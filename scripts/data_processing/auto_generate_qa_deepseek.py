#!/usr/bin/env python3
"""
使用硅基流动API（DeepSeek-V3.2）生成QA对
速度快、成本低、质量高
"""

import json
import os
from typing import List, Dict
from pypdf import PdfReader
import openai
from tqdm import tqdm

class DeepSeekQAGenerator:
    def __init__(self, api_key: str):
        """
        初始化DeepSeek API客户端

        Args:
            api_key: 硅基流动API密钥
        """
        self.client = openai.OpenAI(
            api_key=api_key,
            base_url="https://api.siliconflow.cn/v1"
        )
        self.model = "Pro/deepseek-ai/DeepSeek-V3.2"
        print(f"✅ 使用模型: {self.model}")

    def extract_chapter(self, pdf_path: str, start_page: int, end_page: int) -> str:
        """提取PDF章节文本"""
        reader = PdfReader(pdf_path)
        text = ""
        for page in reader.pages[start_page-1:end_page]:
            text += page.extract_text()
        return text

    def generate_qa_batch(
        self,
        chapter_text: str,
        topic: str,
        num_pairs: int = 5
    ) -> List[Dict]:
        """使用DeepSeek API生成QA对"""

        # 限制输入长度
        if len(chapter_text) > 3000:
            chapter_text = chapter_text[:3000]

        prompt = f"""你是机器人控制领域的专家。基于以下教材内容，生成{num_pairs}个高质量的问答对。

要求：
1. 问题覆盖核心概念、公式推导、应用场景
2. 答案准确专业，包含数学公式（LaTeX格式，如 $Q(s,a)$）
3. 答案长度150-300字
4. 严格遵循教材内容，不要编造

教材内容：
{chapter_text}

输出JSON格式（不要markdown标记）：
{{
  "qa_pairs": [
    {{
      "question": "具体问题",
      "answer": "详细答案（包含定义、公式、应用场景）",
      "topic": "{topic}",
      "difficulty": "medium",
      "has_formula": true
    }}
  ]
}}
"""

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "你是机器人控制领域的专家，擅长从教材提取高质量QA对。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000,
            )

            content = response.choices[0].message.content

            # 提取JSON
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                print(f"未找到JSON: {content[:200]}")
                return []

            json_str = content[start:end]

            # 解析JSON
            try:
                result = json.loads(json_str)
                return result.get("qa_pairs", [])
            except:
                import re
                json_str = re.sub(r'(?<!\\)\\(?!["\\/bfnrtu])', r'\\\\', json_str)
                result = json.loads(json_str)
                return result.get("qa_pairs", [])

        except Exception as e:
            print(f"生成失败: {e}")
            return []

    def convert_to_sft_format(self, qa_pairs: List[Dict], source: str, start_id: int = 0) -> List[Dict]:
        """转换为SFT格式"""
        sft_data = []
        for idx, qa in enumerate(qa_pairs):
            sft_data.append({
                "id": f"sft_{source}_{start_id + idx:03d}",
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
                    "created_at": "2026-03-12"
                }
            })
        return sft_data


def main():
    # API配置
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    generator = DeepSeekQAGenerator(api_key)

    # 定义章节
    sections = [
        {"name": "MDP Basics", "start": 48, "end": 58, "topic": "reinforcement_learning", "batches": 2},
        {"name": "Bellman Equations", "start": 58, "end": 65, "topic": "reinforcement_learning", "batches": 2},
        {"name": "Q-learning", "start": 131, "end": 140, "topic": "reinforcement_learning", "batches": 3},
        {"name": "SARSA", "start": 145, "end": 152, "topic": "reinforcement_learning", "batches": 2},
        {"name": "Policy Gradient", "start": 321, "end": 330, "topic": "reinforcement_learning", "batches": 2},
    ]

    pdf_path = "references/textbooks/RLbook2020.pdf"
    output_file = "dataset/sft_rl_deepseek_50.jsonl"

    # 清空旧文件
    if os.path.exists(output_file):
        os.remove(output_file)

    all_qa_pairs = []
    total_id = 0

    print(f"\n📖 处理教材: {pdf_path}\n")

    for section in sections:
        print(f"📝 处理章节: {section['name']}")

        # 提取章节文本
        chapter_text = generator.extract_chapter(
            pdf_path,
            section["start"],
            section["end"]
        )

        # 分批生成
        for batch_idx in range(section["batches"]):
            print(f"  批次 {batch_idx+1}/{section['batches']}...", end=" ", flush=True)

            # 每批使用不同的文本片段
            start_pos = batch_idx * 3000
            text_chunk = chapter_text[start_pos:start_pos + 3000]

            if len(text_chunk) < 500:
                print("文本不足，跳过")
                continue

            qa_pairs = generator.generate_qa_batch(text_chunk, section["topic"], num_pairs=5)

            if qa_pairs:
                sft_data = generator.convert_to_sft_format(
                    qa_pairs,
                    f"textbook_{section['topic']}",
                    total_id
                )

                # 立即保存
                os.makedirs("dataset", exist_ok=True)
                with open(output_file, "a", encoding="utf-8") as f:
                    for item in sft_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                all_qa_pairs.extend(sft_data)
                total_id += len(qa_pairs)
                print(f"✅ 生成 {len(qa_pairs)} 个")
            else:
                print("❌ 失败")

    print(f"\n🎉 完成！共生成 {len(all_qa_pairs)} 个QA对")
    print(f"📁 保存到: {output_file}")

    # 统计信息
    print(f"\n📊 统计信息:")
    print(f"  - 总QA对数: {len(all_qa_pairs)}")
    print(f"  - 文件大小: {os.path.getsize(output_file) / 1024:.1f} KB")


if __name__ == "__main__":
    main()
