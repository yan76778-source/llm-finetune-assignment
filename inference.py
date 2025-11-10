# file: inference.py
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
import sys, os

def load_adapter_and_tokenizer(base_model, adapter_path):
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    # Load base model and then load adapter weights (if LoRA saved via save_pretrained)
    model = AutoModelForCausalLM.from_pretrained(base_model, device_map="auto", torch_dtype=torch.float16, trust_remote_code=True)
    # If adapter saved with PEFT, we can load via from_pretrained on model
    try:
        # Try to load adapter weights (PEFT style)
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter_path)
    except Exception as e:
        print("Warning: failed to load adapter via PeftModel:", e)
    return model, tokenizer

def generate(base_model, adapter_path, prompt, max_new_tokens=128):
    model, tokenizer = load_adapter_and_tokenizer(base_model, adapter_path)
    pipe = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0)
    out = pipe(prompt, max_new_tokens=max_new_tokens)
    return out

if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage: python inference.py <base_model> <adapter_path> <prompt>")
        sys.exit(1)
    base_model = sys.argv[1]
    adapter_path = sys.argv[2]
    prompt = sys.argv[3]
    print(generate(base_model, adapter_path, prompt))
