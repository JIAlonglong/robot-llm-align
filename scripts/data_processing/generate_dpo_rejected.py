#!/usr/bin/env python3
"""用 SFT 模型生成 DPO rejected 数据"""

import json, os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from tqdm import tqdm

SFT_BASE  = "Qwen/Qwen2.5-1.5B-Instruct"
SFT_LORA  = "checkpoints/sft_qwen1.5b"
CHOSEN    = "dataset/dpo_chosen_with_cot_tools.jsonl"
OUTPUT    = "dataset/dpo_pairs.jsonl"
MAX_NEW   = 512


def load_model():
    tokenizer = AutoTokenizer.from_pretrained(SFT_BASE, trust_remote_code=True, padding_side="left")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(SFT_BASE, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)
    model = PeftModel.from_pretrained(model, SFT_LORA)
    model.eval()
    return model, tokenizer


def generate_rejected(model, tokenizer, question: str) -> str:
    messages = [
        {"role": "system", "content": "你是一个机器人控制与强化学习领域的专家。"},
        {"role": "user",   "content": question},
    ]
    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = tokenizer(text, return_tensors="pt").to("cuda")
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW,
            temperature=0.9,   # 高温，让 rejected 质量差一些
            do_sample=True,
            top_p=0.9,
            pad_token_id=tokenizer.pad_token_id,
        )
    return tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


def main():
    # 断点续传
    done = set()
    if os.path.exists(OUTPUT):
        with open(OUTPUT, "r") as f:
            for line in f:
                d = json.loads(line)
                done.add(d["id"])
        print(f"已有 {len(done)} 条，断点续传...")

    chosen_data = []
    with open(CHOSEN, "r") as f:
        for line in f:
            d = json.loads(line)
            if d["id"] not in done:
                chosen_data.append(d)

    print(f"需要生成 {len(chosen_data)} 条 rejected")
    model, tokenizer = load_model()

    with open(OUTPUT, "a", encoding="utf-8") as fout:
        for item in tqdm(chosen_data, desc="生成 rejected"):
            rejected = generate_rejected(model, tokenizer, item["prompt"])
            record = {
                "id":       item["id"],
                "topic":    item["topic"],
                "prompt":   item["prompt"],
                "chosen":   item["chosen"],
                "rejected": rejected,
                "has_tool": item.get("has_tool", False),
                "has_cot":  item.get("has_cot", False),
            }
            fout.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 完成！保存到: {OUTPUT}")


if __name__ == "__main__":
    main()
