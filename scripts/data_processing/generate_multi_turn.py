#!/usr/bin/env python3
"""
基于单轮QA生成多轮对话数据
使用硅基流动API（DeepSeek-V3.2）
"""

import json
import os
import re
from typing import List, Dict
import openai
from tqdm import tqdm


class MultiTurnGenerator:
    def __init__(self, api_key: str):
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        self.model = "Pro/deepseek-ai/DeepSeek-V3.2"

    def generate_multi_turn(self, question: str, answer: str, topic: str) -> List[Dict]:
        """从单轮QA生成多轮对话"""

        prompt = f"""基于以下单轮问答，生成2轮自然的追问对话。

原始问答：
Q: {question}
A: {answer}

要求：
1. 第2轮：深入追问（如"为什么"、"如何证明"、"公式推导"）
2. 第3轮：应用场景或对比（如"适用于什么场景"、"与XX的区别"）
3. 保持专业性和连贯性，答案包含数学公式（LaTeX格式）
4. 每轮答案150-250字

输出JSON（不要markdown标记）：
{{
  "conversations": [
    {{"role": "user", "content": "{question}"}},
    {{"role": "assistant", "content": "{answer}"}},
    {{"role": "user", "content": "追问1的内容"}},
    {{"role": "assistant", "content": "回答1的内容"}},
    {{"role": "user", "content": "追问2的内容"}},
    {{"role": "assistant", "content": "回答2的内容"}}
  ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是机器人控制领域专家，擅长生成高质量的多轮对话数据。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.4,
                max_tokens=2000,
            )

            content = response.choices[0].message.content

            # 提取JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                return []

            json_str = content[start:end]

            try:
                result = json.loads(json_str)
                return result.get("conversations", [])
            except:
                json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)
                result = json.loads(json_str)
                return result.get("conversations", [])

        except Exception as e:
            print(f"生成失败: {e}")
            return []


def main():
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    generator = MultiTurnGenerator(api_key)

    input_file = "dataset/sft_rl_deepseek_50.jsonl"
    output_file = "dataset/sft_multi_turn_50.jsonl"

    # 读取单轮QA数据
    single_turn_data = []
    with open(input_file, "r", encoding="utf-8") as f:
        for line in f:
            single_turn_data.append(json.loads(line.strip()))

    print(f"读取到 {len(single_turn_data)} 条单轮QA数据")

    # 清空旧文件
    if os.path.exists(output_file):
        os.remove(output_file)

    success = 0
    for idx, item in enumerate(tqdm(single_turn_data, desc="生成多轮对话")):
        convs = item["conversations"]

        # 提取问题和答案
        question = next((c["content"] for c in convs if c["role"] == "user"), "")
        answer = next((c["content"] for c in convs if c["role"] == "assistant"), "")
        topic = item["metadata"].get("topic", "reinforcement_learning")

        if not question or not answer:
            continue

        # 生成多轮对话
        multi_turn_convs = generator.generate_multi_turn(question, answer, topic)

        if len(multi_turn_convs) >= 4:  # 至少2轮
            # 添加system消息
            full_convs = [
                {
                    "role": "system",
                    "content": "你是一个机器人控制领域的专家，擅长强化学习、PID控制、MPC等技术。请用专业、准确的语言回答问题，必要时包含数学公式。"
                }
            ] + multi_turn_convs

            new_item = {
                "id": f"sft_multi_turn_{idx:03d}",
                "source": "multi_turn_generated",
                "conversations": full_convs,
                "metadata": {
                    "topic": topic,
                    "difficulty": item["metadata"].get("difficulty", "medium"),
                    "has_formula": True,
                    "num_turns": len([c for c in multi_turn_convs if c["role"] == "user"]),
                    "created_at": "2026-03-12"
                }
            }

            with open(output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

            success += 1
            print(f"  ✅ [{idx+1}/{len(single_turn_data)}] 生成 {new_item['metadata']['num_turns']} 轮对话")
        else:
            print(f"  ❌ [{idx+1}/{len(single_turn_data)}] 生成失败")

    print(f"\n🎉 完成！共生成 {success} 条多轮对话数据")
    print(f"📁 保存到: {output_file}")


if __name__ == "__main__":
    main()
