#!/usr/bin/env python3
"""
LLM-as-a-Judge 评估脚本
对比 base / SFT / DPO 三个模型在机器人控制领域的回答质量

运行方式：
    python scripts/evaluate.py
    python scripts/evaluate.py --mode sft        # 只评估 SFT
    python scripts/evaluate.py --mode dpo        # 只评估 DPO
    python scripts/evaluate.py --no-judge        # 跳过 LLM 打分，只生成回答

依赖：
    pip install transformers peft openai tqdm
"""

import argparse
import json
import os
import torch
from dataclasses import dataclass, field
from typing import List, Optional
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import openai

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class EvalConfig:
    base_model: str = "Qwen/Qwen2.5-7B-Instruct"
    sft_model:  str = "checkpoints/sft_qwen2.5_7b_v2"
    dpo_model:  str = "checkpoints/dpo_qwen2.5_7b_v1"

    # Judge 用 DeepSeek（硅基流动）
    judge_api_key:  str = os.environ.get("SILICONFLOW_API_KEY", "")
    judge_base_url: str = "https://api.siliconflow.cn/v1"
    judge_model:    str = "Pro/deepseek-ai/DeepSeek-V3.2"

    output_file: str = "logs/eval_results.jsonl"
    max_new_tokens: int = 512


# 评估问题集（覆盖各子领域）
EVAL_QUESTIONS = [
    # 强化学习
    {"topic": "rl", "question": "什么是马尔可夫决策过程（MDP）？请给出形式化定义。"},
    {"topic": "rl", "question": "Q-learning 和 SARSA 的本质区别是什么？各自适用于什么场景？"},
    {"topic": "rl", "question": "策略梯度方法中，基准线（baseline）的作用是什么？"},
    # PID
    {"topic": "pid", "question": "PID 控制器中积分饱和问题如何产生？有哪些解决方案？"},
    {"topic": "pid", "question": "如何用 Ziegler-Nichols 方法整定 PID 参数？"},
    # MPC
    {"topic": "mpc", "question": "模型预测控制（MPC）相比 PID 的核心优势是什么？"},
    {"topic": "mpc", "question": "MPC 中预测时域和控制时域的选取对控制性能有何影响？"},
    # 机器人
    {"topic": "robotics", "question": "机器人正运动学和逆运动学的区别是什么？逆运动学为何更难求解？"},
    {"topic": "robotics", "question": "什么是雅可比矩阵？在机器人速度控制中如何使用？"},
    # 综合
    {"topic": "general", "question": "强化学习中探索与利用的权衡（exploration-exploitation tradeoff）如何处理？"},
]

JUDGE_PROMPT_TEMPLATE = """你是一位机器人控制与强化学习领域的专家评审。请对以下回答进行评分。

问题：{question}

回答：
{answer}

请从以下4个维度打分（每项1-5分）：
1. 准确性：技术内容是否正确，无幻觉
2. 完整性：是否覆盖了问题的核心要点
3. 专业性：是否使用了正确的专业术语和公式
4. 清晰度：表达是否清晰易懂

输出 JSON（不要 markdown 标记）：
{{"accuracy": <1-5>, "completeness": <1-5>, "professionalism": <1-5>, "clarity": <1-5>, "total": <4-20>, "comment": "<一句话总结>"}}
"""


# ============================================================
# 2. 模型推理
# ============================================================

def load_model(base_model: str, adapter_path: Optional[str], merge: bool = False):
    """加载模型，可选挂载 LoRA adapter"""
    tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        base_model, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )

    if adapter_path and os.path.exists(adapter_path):
        model = PeftModel.from_pretrained(model, adapter_path)
        if merge:
            model = model.merge_and_unload()
        model.eval()
        print(f"✅ 加载 adapter: {adapter_path}")
    else:
        model.eval()
        if adapter_path:
            print(f"⚠️  adapter 不存在，跳过: {adapter_path}")

    return model, tokenizer


def generate_answer(model, tokenizer, question: str, max_new_tokens: int) -> str:
    messages = [
        {"role": "system", "content": "你是一个机器人控制与强化学习领域的专家，请用专业、准确的语言回答问题。"},
        {"role": "user", "content": question},
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================
# 3. LLM Judge 打分
# ============================================================

def judge_answer(client, judge_model: str, question: str, answer: str) -> dict:
    prompt = JUDGE_PROMPT_TEMPLATE.format(question=question, answer=answer)
    try:
        resp = client.chat.completions.create(
            model=judge_model,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=300,
        )
        content = resp.choices[0].message.content.strip()
        start, end = content.find("{"), content.rfind("}") + 1
        return json.loads(content[start:end])
    except Exception as e:
        print(f"  Judge 失败: {e}")
        return {"accuracy": 0, "completeness": 0, "professionalism": 0, "clarity": 0, "total": 0, "comment": "评分失败"}


# ============================================================
# 4. 主流程
# ============================================================

def run_eval(config: EvalConfig, mode: str, use_judge: bool):
    os.makedirs("logs", exist_ok=True)

    # 确定要评估哪些模型
    models_to_eval = []
    if mode in ("all", "base"):
        models_to_eval.append(("base", None, False))
    if mode in ("all", "sft"):
        models_to_eval.append(("sft", config.sft_model, False))
    if mode in ("all", "dpo"):
        models_to_eval.append(("dpo", config.dpo_model, True))

    judge_client = None
    if use_judge:
        judge_client = openai.OpenAI(api_key=config.judge_api_key, base_url=config.judge_base_url)

    all_results = []

    for model_name, adapter_path, merge in models_to_eval:
        print(f"\n{'='*60}")
        print(f"评估模型: {model_name.upper()}")
        print(f"{'='*60}")

        model, tokenizer = load_model(config.base_model, adapter_path, merge)
        scores = []

        for item in tqdm(EVAL_QUESTIONS, desc=f"[{model_name}] 生成回答"):
            answer = generate_answer(model, tokenizer, item["question"], config.max_new_tokens)

            result = {
                "model": model_name,
                "topic": item["topic"],
                "question": item["question"],
                "answer": answer,
                "scores": None,
            }

            if use_judge and judge_client:
                result["scores"] = judge_answer(judge_client, config.judge_model, item["question"], answer)
                scores.append(result["scores"].get("total", 0))

            all_results.append(result)

            # 实时写入，防止中途崩溃丢数据
            with open(config.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(result, ensure_ascii=False) + "\n")

        if scores:
            avg = sum(scores) / len(scores)
            print(f"\n[{model_name}] 平均分: {avg:.2f}/20 ({len(scores)} 题)")

        # 释放显存
        del model
        torch.cuda.empty_cache()

    # 汇总对比
    if use_judge:
        print(f"\n{'='*60}")
        print("汇总对比")
        print(f"{'='*60}")
        for model_name in ["base", "sft", "dpo"]:
            model_results = [r for r in all_results if r["model"] == model_name and r["scores"]]
            if not model_results:
                continue
            avg = sum(r["scores"]["total"] for r in model_results) / len(model_results)
            print(f"  {model_name.upper():6s}: {avg:.2f}/20")

    print(f"\n✅ 评估完成，结果保存到: {config.output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["all", "base", "sft", "dpo"], default="all")
    parser.add_argument("--no-judge", action="store_true", help="跳过 LLM 打分")
    args = parser.parse_args()

    config = EvalConfig()
    run_eval(config, mode=args.mode, use_judge=not args.no_judge)


if __name__ == "__main__":
    main()
