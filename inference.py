import os
os.environ["TRANSFORMERS_NO_TF"] = "1"  # 禁止 transformers import tensorflow

from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from peft import PeftModel


def generate(model_id, adapter_dir, prompt):
    tokenizer = AutoTokenizer.from_pretrained(model_id)

    base_model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype="auto",
        device_map="auto"
    )

    model = PeftModel.from_pretrained(base_model, adapter_dir)
    model.eval()

    pipe = pipeline(
        "text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=128
    )

    result = pipe(prompt)
    return result[0]["generated_text"]
