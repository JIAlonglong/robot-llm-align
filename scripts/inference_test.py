"""
快速推理测试：对比 base 模型和 SFT 模型的回答
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-7B-Instruct"
SFT_MODEL  = "checkpoints/sft_qwen2.5_7b_v2"

TEST_QUESTIONS = [
    "什么是马尔可夫决策过程（MDP）？",
    "PID控制器中，积分项的作用是什么？",
    "模型预测控制（MPC）相比PID有什么优势？",
    "Q-learning和SARSA的主要区别是什么？",
    "如何调整PID参数来减少超调？",
]

def build_prompt(tokenizer, question):
    messages = [
        {"role": "system", "content": "你是一个专业的机器人控制与强化学习专家。"},
        {"role": "user", "content": question},
    ]
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

def generate(model, tokenizer, prompt, max_new_tokens=256):
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            temperature=1.0,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)

def main():
    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

    print("加载 base 模型...")
    base_model = AutoModelForCausalLM.from_pretrained(
        BASE_MODEL, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )

    print("加载 SFT 模型...")
    sft_model = PeftModel.from_pretrained(base_model, SFT_MODEL)
    sft_model.eval()

    for i, q in enumerate(TEST_QUESTIONS, 1):
        prompt = build_prompt(tokenizer, q)
        print(f"\n{'='*60}")
        print(f"[Q{i}] {q}")
        print(f"{'='*60}")

        # base 模型（禁用 LoRA adapter）
        sft_model.disable_adapter_layers()
        base_ans = generate(sft_model, tokenizer, prompt)
        print(f"\n[Base]\n{base_ans}")

        # SFT 模型（启用 LoRA adapter）
        sft_model.enable_adapter_layers()
        sft_ans = generate(sft_model, tokenizer, prompt)
        print(f"\n[SFT]\n{sft_ans}")

    print("\n✅ 测试完成")

if __name__ == "__main__":
    main()
