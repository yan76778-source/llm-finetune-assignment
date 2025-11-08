import re
def parse_final_answer(text):
    """从CoT或直接输出中解析出 \boxed{} 或最后的数字"""
    # 优先匹配 \boxed{}
    match = re.search(r"\\boxed\{([\d\.,]+)\}", text)
    if match:
        return match.group(1).replace(",", "")
    
    # ... (其他解析逻辑)
    matches = re.findall(r"([\d\.,]+)", text)
    if matches:
        return matches[-1].replace(",", "")
    return None

def create_chat_messages(example, mode="cot"):
    """
    根据模式(direct/cot)创建 ChatML 格式的训练样本
    """
    question = "Question: " + example['question']
    answer_text = example['answer']
    
    if mode == "direct":
        final_answer = parse_final_answer(answer_text)
        if final_answer is None:
            return None # 如果无法解析答案，跳过此样本
        assistant_content = final_answer
    else:
        assistant_content = answer_text
        
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": assistant_content}
    ]
    return {"messages": messages} # <--- 注意, 为了 'map' 函数，返回一个字典