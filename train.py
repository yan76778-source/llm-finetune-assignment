# file: train.py
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from utils import parse_final_answer, create_chat_messages

def load_model_and_tokenizer(model_id, use_8bit=True):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    try:
        if use_8bit:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                device_map="auto",
                load_in_8bit=True,
                trust_remote_code=True,
            )
        else:
            raise RuntimeError("skip 8bit")
    except Exception as e:
        print(">>>> Warning: failed to load in 8-bit or bitsandbytes not available:", e)
        print(">>>> Falling back to load without 8-bit (may use more VRAM).")
        model = AutoModelForCausalLM.from_pretrained(
            model_id,
            device_map="auto",
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True,
        )
    return model, tokenizer

def get_peft_config():
    return LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        task_type="CAUSAL_LM",
    )

def main(args):
    print(f"--- 1. 加载模型和Tokenizer: {args.model_id} (try 8-bit: {not args.no_8bit}) ---")
    model, tokenizer = load_model_and_tokenizer(args.model_id, use_8bit=not args.no_8bit)

    print(f"--- 2. 加载和处理数据集 (模式: {args.mode}) ---")
    dataset = load_dataset("gsm8k", "main", split="train")

    def map_fn(ex):
        return create_chat_messages(ex, mode=args.mode)

    processed = dataset.map(map_fn, remove_columns=dataset.column_names)
    processed = processed.filter(lambda ex: ex is not None and "text" in ex and ex["text"] is not None)

    print(f"成功处理 {len(processed)} 条样本。")

    peft_config = get_peft_config()

    # NOTE: TrainingArguments does NOT accept max_seq_length, so we remove it here.
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        optim="paged_adamw_8bit" if not args.no_8bit else "adamw_torch",
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        save_strategy="epoch",
        logging_steps=50,
        report_to="none",
        fp16=(not args.no_bf16) and torch.cuda.is_available(),
        bf16=args.bf16 and torch.cuda.is_available(),
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed,
        peft_config=peft_config,
        dataset_text_field="text",   # our utils returns "text"
        max_seq_length=args.max_seq_length,  # pass here, not to TrainingArguments
        args=training_args,
    )

    print("--- 4. 开始训练 ---")
    trainer.train()

    adapter_path = os.path.join(args.adapter_dir, f"gsm8k-{args.mode}")
    print(f"--- 5. 训练完成，保存适配器到 {adapter_path} ---")
    os.makedirs(adapter_path, exist_ok=True)
    trainer.model.save_pretrained(adapter_path)
    print("Saved adapter to", adapter_path)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Qwen-family on GSM8K (direct vs cot)")

    parser.add_argument("--mode", type=str, required=True, choices=["direct", "cot"],
                        help="Finetuning mode: 'direct' or 'cot'")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen1.5-1.8B-Chat",
                        help="Base model ID from Hugging Face")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for training outputs (logs, checkpoints)")
    parser.add_argument("--adapter_dir", type=str, default="./adapters",
                        help="Directory to save the final LoRA adapters")
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--grad_accum", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--max_seq_length", type=int, default=1024)
    parser.add_argument("--no_8bit", action="store_true", help="Disable 8-bit loading (force full/FP16)")
    parser.add_argument("--bf16", action="store_true", help="Use bf16 if available")
    parser.add_argument("--no_bf16", action="store_true", help="Disable bf16")
    args = parser.parse_args()
    main(args)
