# file: utils.py
import re

def parse_final_answer(text):
    """从 CoT 或直接输出中解析出 \boxed{} 或最后的数字/文本"""
    if text is None:
        return None
    # 优先匹配 \boxed{}
    match = re.search(r"\\boxed\{([^\}]+)\}", text)
    if match:
        return match.group(1).strip()

    # 如果没有 \boxed，尝试匹配“答案：...”后的文本
    m = re.search(r"答案[:：]\s*(.+)", text, flags=re.IGNORECASE)
    if m:
        return m.group(1).strip()

    # 最后尝试抓最后的数字/token
    matches = re.findall(r"([\d\.\,]+)", text)
    if matches:
        return matches[-1].replace(",", "")
    # 否则返回原文本的最后一行（作为兜底）
    lines = [ln.strip() for ln in text.strip().splitlines() if ln.strip()]
    if lines:
        return lines[-1]
    return None

def create_chat_messages(example, mode="cot"):
    """
    把 gsm8k 的实例转换为用于 SFTTrainer 的单字段 "text"
    返回字典 {'text': '<prompt and response text>'}
    """
    # gsm8k fields: 'question', 'answer' (answer may contain solution steps for cot)
    question = example.get("question") or example.get("Problem") or ""
    answer_text = example.get("answer") or example.get("Answer") or ""

    # Build prompt -> we design a simple chat-like prompt.
    if mode == "direct":
        # For direct, we want the assistant to output just the final answer.
        final = parse_final_answer(answer_text)
        if final is None:
            # skip if cannot parse final answer (trainer will filter None)
            return None
        prompt = f"User: {question}\nAssistant:"
        # We set text to: prompt + final answer (so trainer learns mapping)
        full_text = prompt + " " + final
    else:
        # For CoT, include the full answer text (which contains reasoning and final)
        prompt = f"User: {question}\nAssistant:"
        # sometimes answer_text already contains steps; ensure spacing
        full_text = prompt + " " + answer_text

    return {"text": full_text}
