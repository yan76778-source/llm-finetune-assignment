#%%writefile project_code/train.py
import os
import argparse
import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from peft import LoraConfig
from trl import SFTTrainer

from utils import create_chat_messages

def load_model_and_tokenizer(model_id):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    print("🚫 禁用 8-bit，使用 FP16 加载模型")

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        trust_remote_code=True,
        torch_dtype=torch.float16,
        device_map="auto",
    )
    return model, tokenizer


def main(args):
    print(f"✅ 加载模型: {args.model_id}")
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    print(f"✅ 加载数据集 gsm8k (mode={args.mode})")
    dataset = load_dataset("gsm8k", "main", split="train")

    processed = dataset.map(
        lambda ex: create_chat_messages(ex, mode=args.mode),
        remove_columns=dataset.column_names
    ).filter(lambda ex: ex is not None and "text" in ex)

    peft_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        num_train_epochs=1,
        logging_steps=50,
        save_strategy="epoch",
        fp16=True,
        report_to="none"
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed,
        dataset_text_field="text",
        peft_config=peft_config,
        max_seq_length=1024,
        args=training_args,
    )

    os.makedirs(args.adapter_dir, exist_ok=True)
    trainer.model.save_pretrained(args.adapter_dir)
    print("🎉 训练完成！LoRA adapter 已保存到：", args.adapter_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", required=True, choices=["direct","cot"])
    parser.add_argument("--model_id", default="Qwen/Qwen1.5-1.8B-Chat")
    parser.add_argument("--output_dir", default="./results")
    parser.add_argument("--adapter_dir", default="./adapters")
    args = parser.parse_args()
    main(args)
