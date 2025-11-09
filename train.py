# file: train.py
import torch
import os
import argparse  # <-- 专业加分项：使用命令行参数
from datasets import load_dataset, Dataset
from transformers import (
    AutoModelForCausalLM, AutoTokenizer,
    BitsAndBytesConfig, TrainingArguments
)
from peft import LoraConfig
from trl import SFTTrainer

# 导入我们自己的辅助函数
from utils import parse_final_answer, create_chat_messages

# --- 1. QLoRA 和模型加载配置 ---
def load_model_and_tokenizer(model_id):
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        quantization_config=bnb_config,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    return model, tokenizer

# --- 2. PEFT (LoRA) 配置 ---
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

# --- 3. 训练器主函数 ---
def main(args):
    """主训练函数"""
    
    print(f"--- 1. 加载模型和Tokenizer: {args.model_id} ---")
    model, tokenizer = load_model_and_tokenizer(args.model_id)

    print(f"--- 2. 加载和处理数据集 (模式: {args.mode}) ---")
    dataset = load_dataset("gsm8k", "main", split="train")
    
    # 使用 .map() 高效处理
    processed_dataset = dataset.map(
        lambda ex: create_chat_messages(ex, mode=args.mode),
        batched=False, # 一次处理一个样本
        remove_columns=dataset.column_names # 删除旧列
    ).filter(lambda ex: ex["messages"] is not None) # 过滤掉解析失败的样本

    print(f"成功处理 {len(processed_dataset)} 条样本。")
    
    peft_config = get_peft_config()
    
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        per_device_train_batch_size=4,
        gradient_accumulation_steps=4,
        optim="paged_adamw_8bit",
        learning_rate=2e-4,
        lr_scheduler_type="cosine",
        num_train_epochs=1,
        save_strategy="epoch",
        logging_steps=25,
        report_to="none",
        fp16=False,
        bf16=True,
        max_seq_length=1024,
        packing=True,
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=processed_dataset,
        peft_config=peft_config,
        dataset_text_field="messages",
        max_seq_length=1024,
        args=training_args,
    )
    
    print("--- 4. 开始训练 ---")
    trainer.train()
    
    adapter_path = os.path.join(args.adapter_dir, f"gsm8k-{args.mode}")
    print(f"--- 5. 训练完成，保存适配器到 {adapter_path} ---")
    trainer.model.save_pretrained(adapter_path)

# --- 4. 命令行入口 ---
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Finetune Qwen1.5-1.8B on GSM8K")
    
    parser.add_argument("--mode", type=str, required=True, choices=["direct", "cot"],
                        help="Finetuning mode: 'direct' or 'cot'")
    parser.add_argument("--model_id", type=str, default="Qwen/Qwen1.5-1.8B-Chat",
                        help="Base model ID from Hugging Face")
    parser.add_argument("--output_dir", type=str, default="./results",
                        help="Directory for training outputs (logs, checkpoints)")
    parser.add_argument("--adapter_dir", type=str, default="./adapters",
                        help="Directory to save the final LoRA adapters")
    
    args = parser.parse_args()
    main(args)