#!/usr/bin/env python3
"""
自动生成领域专业QA对
使用GPT-4o从教材中提取高质量问答对
"""

import json
import os
from typing import List, Dict
from pypdf import PdfReader
import openai
from tqdm import tqdm

# 配置
openai.api_key = os.getenv("OPENAI_API_KEY")  # 从环境变量读取

class QAGenerator:
    def __init__(self, model: str = "gpt-4o"):
        self.model = model

    def extract_chapter(self, pdf_path: str, start_page: int, end_page: int) -> str:
        """提取PDF章节文本"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[start_page:end_page]:
            text += page.extract_text()
        return text

    def generate_qa_pairs(
        self,
        chapter_text: str,
        topic: str,
        num_pairs: int = 10
    ) -> List[Dict]:
        """使用GPT-4o生成QA对"""

        # 限制输入长度（避免超token）
        max_chars = 12000  # 约3000 tokens
        if len(chapter_text) > max_chars:
            chapter_text = chapter_text[:max_chars] + "\n...(内容截断)"

        prompt = f"""你是一个机器人控制领域的专家。请基于以下教材内容，生成{num_pairs}个高质量的问答对。

要求：
1. 问题应该覆盖核心概念、公式推导、应用场景
2. 答案必须准确、专业，包含数学公式（使用LaTeX格式，如 $Q(s,a)$）
3. 答案长度：150-300字
4. 难度分布：easy(30%), medium(50%), hard(20%)
5. 必须严格遵循教材内容，不要编造

教材主题：{topic}

教材内容：
{chapter_text}

输出JSON格式（不要包含markdown代码块标记）：
{{
  "qa_pairs": [
    {{
      "question": "解释Q-learning的off-policy特性",
      "answer": "Q-learning是一种off-policy时序差分控制算法。所谓off-policy，是指行为策略（用于探索环境）和目标策略（用于更新价值函数）可以不同。\\n\\n更新公式：$Q(s,a) \\leftarrow Q(s,a) + \\alpha[r + \\gamma \\max_{{a'}} Q(s',a') - Q(s,a)]$\\n\\n关键点：使用max操作选择下一状态的最优动作（贪婪策略），但实际执行时可以使用ε-greedy（探索策略）。",
      "topic": "{topic}",
      "difficulty": "medium",
      "has_formula": true
    }}
  ]
}}
"""

        try:
            response = openai.ChatCompletion.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,  # 降低随机性，提高准确性
                max_tokens=3000,
            )

            content = response.choices[0].message.content

            # 解析JSON（移除可能的markdown标记）
            content = content.strip()
            if content.startswith("```json"):
                content = content[7:]
            if content.startswith("```"):
                content = content[3:]
            if content.endswith("```"):
                content = content[:-3]

            result = json.loads(content.strip())
            return result["qa_pairs"]

        except Exception as e:
            print(f"生成失败: {e}")
            return []

    def convert_to_sft_format(self, qa_pairs: List[Dict], source: str) -> List[Dict]:
        """转换为SFT训练格式"""
        sft_data = []

        for idx, qa in enumerate(qa_pairs):
            sft_data.append({
                "id": f"sft_{source}_{idx:03d}",
                "source": source,
                "conversations": [
                    {
                        "role": "system",
                        "content": "你是一个机器人控制领域的专家，擅长强化学习、PID控制、MPC等技术。请用专业、准确的语言回答问题，必要时包含数学公式。"
                    },
                    {
                        "role": "user",
                        "content": qa["question"]
                    },
                    {
                        "role": "assistant",
                        "content": qa["answer"]
                    }
                ],
                "metadata": {
                    "topic": qa.get("topic", "unknown"),
                    "difficulty": qa.get("difficulty", "medium"),
                    "has_formula": qa.get("has_formula", False),
                    "created_at": "2026-03-11"
                }
            })

        return sft_data


def main():
    """主函数：批量生成QA对"""

    generator = QAGenerator()

    # 定义要处理的章节
    chapters = [
        {
            "pdf_path": "textbooks/sutton_barto_rl.pdf",
            "chapters": [
                {"name": "Q-learning", "start": 130, "end": 145, "topic": "reinforcement_learning", "num_qa": 15},
                {"name": "SARSA", "start": 145, "end": 155, "topic": "reinforcement_learning", "num_qa": 10},
                {"name": "Policy Gradient", "start": 320, "end": 340, "topic": "reinforcement_learning", "num_qa": 15},
            ]
        },
        {
            "pdf_path": "textbooks/control_theory.pdf",
            "chapters": [
                {"name": "PID Control", "start": 50, "end": 70, "topic": "pid_control", "num_qa": 20},
                {"name": "State Space", "start": 100, "end": 120, "topic": "modern_control", "num_qa": 15},
            ]
        },
        {
            "pdf_path": "textbooks/mpc_book.pdf",
            "chapters": [
                {"name": "MPC Basics", "start": 20, "end": 40, "topic": "mpc", "num_qa": 15},
            ]
        }
    ]

    all_qa_pairs = []

    # 批量生成
    for book in chapters:
        pdf_path = book["pdf_path"]

        if not os.path.exists(pdf_path):
            print(f"⚠️  教材文件不存在: {pdf_path}")
            continue

        print(f"\n处理教材: {pdf_path}")

        for chapter in tqdm(book["chapters"], desc="生成QA对"):
            # 提取章节文本
            chapter_text = generator.extract_chapter(
                pdf_path,
                chapter["start"],
                chapter["end"]
            )

            # 生成QA对
            qa_pairs = generator.generate_qa_pairs(
                chapter_text,
                chapter["topic"],
                chapter["num_qa"]
            )

            # 转换为SFT格式
            sft_data = generator.convert_to_sft_format(
                qa_pairs,
                f"textbook_{chapter['topic']}"
            )

            all_qa_pairs.extend(sft_data)

            print(f"  ✅ {chapter['name']}: 生成 {len(qa_pairs)} 个QA对")

    # 保存结果
    output_file = "dataset/sft_robotics_auto_generated.jsonl"
    os.makedirs("dataset", exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_qa_pairs:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 完成！共生成 {len(all_qa_pairs)} 个QA对")
    print(f"📁 保存到: {output_file}")

    # 统计信息
    topics = {}
    for item in all_qa_pairs:
        topic = item["metadata"]["topic"]
        topics[topic] = topics.get(topic, 0) + 1

    print("\n📊 主题分布:")
    for topic, count in topics.items():
        print(f"  - {topic}: {count} 条")


if __name__ == "__main__":
    main()
