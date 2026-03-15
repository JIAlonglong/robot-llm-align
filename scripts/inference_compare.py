#!/usr/bin/env python3
"""推理对比测试：Base vs SFT vs DPO"""

import os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_LORA   = os.path.abspath("checkpoints/sft_qwen1.5b")
DPO_LORA   = os.path.abspath("checkpoints/dpo_qwen1.5b_merged")

TEST_QUESTIONS = [
    "请解释PID控制器的工作原理，并说明如何整定Kp、Ki、Kd参数？",
    "什么是马尔可夫决策过程（MDP）？请给出数学定义和直觉解释。",
    "机器人逆运动学问题有哪些常见解法？各有什么优缺点？",
    "PPO算法相比TRPO有什么改进？为什么PPO在实践中更常用？",
    "如何设计一个MPC控制器来控制移动机器人的轨迹跟踪？",
]

SYSTEM = "你是一个机器人控制与强化学习领域的专家，擅长强化学习、PID控制、MPC、机器人运动学等技术。"


def load_model(base, lora_path=None):
    tok_path = lora_path if (lora_path and os.path.exists(lora_path) and not os.path.exists(os.path.join(lora_path, "adapter_config.json"))) else base
    tokenizer = AutoTokenizer.from_pretrained(tok_path, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    if lora_path and os.path.exists(lora_path):
        if os.path.exists(os.path.join(lora_path, "adapter_config.json")):
            model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
            model = PeftModel.from_pretrained(model, lora_path)
        else:
            model = AutoModelForCausalLM.from_pretrained(lora_path, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    else:
        model = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    model.eval()
    return model, tokenizer


def generate(model, tokenizer, question, max_new=400):
    messages = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question}]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(**inputs, max_new_tokens=max_new, temperature=0.3, do_sample=True, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def run_comparison():
    results = []

    for name, lora in [("Base", None), ("SFT", SFT_LORA), ("DPO", DPO_LORA)]:
        print(f"\n{'='*60}")
        print(f"加载 {name} 模型...")
        model, tokenizer = load_model(BASE_MODEL, lora)

        model_results = []
        for i, q in enumerate(TEST_QUESTIONS):
            print(f"  [{i+1}/{len(TEST_QUESTIONS)}] {q[:50]}...")
            ans = generate(model, tokenizer, q)
            model_results.append({"question": q, "answer": ans})
            print(f"  回答: {ans[:200]}...")

        results.append({"model": name, "answers": model_results})

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 输出对比报告
    print("\n" + "="*60)
    print("对比报告")
    print("="*60)
    for i, q in enumerate(TEST_QUESTIONS):
        print(f"\n【问题 {i+1}】{q}")
        print("-"*60)
        for r in results:
            print(f"\n[{r['model']}]")
            print(r["answers"][i]["answer"][:500])
        print("="*60)

    # 保存结果
    import json
    os.makedirs("logs", exist_ok=True)
    with open("logs/comparison_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print("\n✅ 结果保存到: logs/comparison_results.json")


if __name__ == "__main__":
    run_comparison()
