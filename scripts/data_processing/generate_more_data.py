#!/usr/bin/env python3
"""
扩充数据集：生成更多 RL/PID/MPC 相关的 QA 对
目标：从 390 条扩充到 1000+ 条
"""

import json
import os
from typing import List, Dict
from pypdf import PdfReader
import openai
from tqdm import tqdm

class DeepSeekQAGenerator:
    def __init__(self, api_key: str):
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

    def generate_qa_from_text(
        self,
        chapter_text: str,
        topic: str,
        num_pairs: int = 5
    ) -> List[Dict]:
        """从教材文本生成QA对"""

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
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                return []

            json_str = content[start:end]
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

    def generate_qa_from_topic(
        self,
        topic_name: str,
        topic_description: str,
        num_pairs: int = 10
    ) -> List[Dict]:
        """直接从主题生成QA对（不需要PDF）"""

        prompt = f"""你是机器人控制领域的专家。请生成{num_pairs}个关于"{topic_name}"的高质量问答对。

主题描述：
{topic_description}

要求：
1. 问题覆盖核心概念、公式推导、实际应用
2. 答案准确专业，包含数学公式（LaTeX格式）
3. 答案长度150-300字
4. 基于真实的控制理论知识，不要编造

输出JSON格式（不要markdown标记）：
{{
  "qa_pairs": [
    {{
      "question": "具体问题",
      "answer": "详细答案（包含定义、公式、应用场景）",
      "topic": "{topic_name}",
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
                    {"role": "system", "content": "你是机器人控制领域的专家，精通强化学习、PID控制、MPC等技术。"},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2500,
            )

            content = response.choices[0].message.content
            start = content.find("{")
            end = content.rfind("}") + 1
            if start == -1 or end == 0:
                return []

            json_str = content[start:end]
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
                    "created_at": "2026-03-13"
                }
            })
        return sft_data


def main():
    # API配置
    api_key = os.environ.get("SILICONFLOW_API_KEY", "")
    generator = DeepSeekQAGenerator(api_key)

    # ========== 第一部分：从 RL 教材生成更多数据 ==========
    print("\n" + "="*60)
    print("第一部分：从 RL 教材生成数据")
    print("="*60)

    rl_sections = [
        # 已有的章节（保留）
        {"name": "MDP Basics", "start": 48, "end": 58, "topic": "reinforcement_learning", "batches": 3},
        {"name": "Bellman Equations", "start": 58, "end": 65, "topic": "reinforcement_learning", "batches": 3},
        {"name": "Dynamic Programming", "start": 73, "end": 85, "topic": "reinforcement_learning", "batches": 3},
        {"name": "Monte Carlo Methods", "start": 91, "end": 105, "topic": "reinforcement_learning", "batches": 3},
        {"name": "TD Learning", "start": 119, "end": 130, "topic": "reinforcement_learning", "batches": 3},
        {"name": "Q-learning", "start": 131, "end": 140, "topic": "reinforcement_learning", "batches": 4},
        {"name": "SARSA", "start": 145, "end": 152, "topic": "reinforcement_learning", "batches": 3},
        {"name": "Function Approximation", "start": 197, "end": 210, "topic": "reinforcement_learning", "batches": 3},
        {"name": "Policy Gradient", "start": 321, "end": 335, "topic": "reinforcement_learning", "batches": 4},
        {"name": "Actor-Critic", "start": 335, "end": 345, "topic": "reinforcement_learning", "batches": 3},
    ]

    pdf_path = "references/textbooks/RLbook2020.pdf"
    output_file_rl = "dataset/sft_rl_extended.jsonl"

    if os.path.exists(output_file_rl):
        os.remove(output_file_rl)

    all_rl_data = []
    total_id = 0

    for section in rl_sections:
        print(f"\n📝 处理章节: {section['name']}")

        chapter_text = generator.extract_chapter(
            pdf_path,
            section["start"],
            section["end"]
        )

        for batch_idx in range(section["batches"]):
            print(f"  批次 {batch_idx+1}/{section['batches']}...", end=" ", flush=True)

            start_pos = batch_idx * 3000
            text_chunk = chapter_text[start_pos:start_pos + 3000]

            if len(text_chunk) < 500:
                print("文本不足，跳过")
                continue

            qa_pairs = generator.generate_qa_from_text(text_chunk, section["topic"], num_pairs=5)

            if qa_pairs:
                sft_data = generator.convert_to_sft_format(
                    qa_pairs,
                    f"textbook_{section['topic']}",
                    total_id
                )

                os.makedirs("dataset", exist_ok=True)
                with open(output_file_rl, "a", encoding="utf-8") as f:
                    for item in sft_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                all_rl_data.extend(sft_data)
                total_id += len(qa_pairs)
                print(f"✅ 生成 {len(qa_pairs)} 个")
            else:
                print("❌ 失败")

    print(f"\n✅ RL 部分完成！共生成 {len(all_rl_data)} 个QA对")

    # ========== 第二部分：直接从主题生成 PID 数据 ==========
    print("\n" + "="*60)
    print("第二部分：生成 PID 控制数据")
    print("="*60)

    pid_topics = [
        {
            "name": "pid_basics",
            "description": "PID控制器的基本原理，包括比例(P)、积分(I)、微分(D)三个环节的作用，传递函数，控制效果分析"
        },
        {
            "name": "pid_tuning",
            "description": "PID参数整定方法，包括Ziegler-Nichols法、临界比例度法、经验法、自适应整定"
        },
        {
            "name": "pid_design",
            "description": "PID控制器设计，包括系统建模、性能指标、稳定性分析、抗干扰设计"
        },
        {
            "name": "pid_applications",
            "description": "PID控制在机器人中的应用，包括位置控制、速度控制、力控制、温度控制"
        },
    ]

    output_file_pid = "dataset/sft_pid_extended.jsonl"
    if os.path.exists(output_file_pid):
        os.remove(output_file_pid)

    all_pid_data = []
    pid_id = 0

    for topic in pid_topics:
        print(f"\n📝 生成主题: {topic['name']}")

        # 每个主题生成 15 个 QA 对
        for batch in range(3):
            print(f"  批次 {batch+1}/3...", end=" ", flush=True)

            qa_pairs = generator.generate_qa_from_topic(
                topic["name"],
                topic["description"],
                num_pairs=5
            )

            if qa_pairs:
                sft_data = generator.convert_to_sft_format(
                    qa_pairs,
                    f"topic_{topic['name']}",
                    pid_id
                )

                with open(output_file_pid, "a", encoding="utf-8") as f:
                    for item in sft_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                all_pid_data.extend(sft_data)
                pid_id += len(qa_pairs)
                print(f"✅ 生成 {len(qa_pairs)} 个")
            else:
                print("❌ 失败")

    print(f"\n✅ PID 部分完成！共生成 {len(all_pid_data)} 个QA对")

    # ========== 第三部分：直接从主题生成 MPC 数据 ==========
    print("\n" + "="*60)
    print("第三部分：生成 MPC 数据")
    print("="*60)

    mpc_topics = [
        {
            "name": "mpc_basics",
            "description": "模型预测控制(MPC)的基本原理，包括预测模型、滚动优化、反馈校正，与传统控制的区别"
        },
        {
            "name": "mpc_optimization",
            "description": "MPC的优化问题，包括目标函数设计、约束处理、求解算法（QP、SQP）"
        },
        {
            "name": "mpc_stability",
            "description": "MPC的稳定性分析，包括终端约束、终端代价、Lyapunov稳定性"
        },
        {
            "name": "mpc_applications",
            "description": "MPC在机器人中的应用，包括轨迹跟踪、避障、多机器人协同"
        },
    ]

    output_file_mpc = "dataset/sft_mpc_extended.jsonl"
    if os.path.exists(output_file_mpc):
        os.remove(output_file_mpc)

    all_mpc_data = []
    mpc_id = 0

    for topic in mpc_topics:
        print(f"\n📝 生成主题: {topic['name']}")

        # 每个主题生成 15 个 QA 对
        for batch in range(3):
            print(f"  批次 {batch+1}/3...", end=" ", flush=True)

            qa_pairs = generator.generate_qa_from_topic(
                topic["name"],
                topic["description"],
                num_pairs=5
            )

            if qa_pairs:
                sft_data = generator.convert_to_sft_format(
                    qa_pairs,
                    f"topic_{topic['name']}",
                    mpc_id
                )

                with open(output_file_mpc, "a", encoding="utf-8") as f:
                    for item in sft_data:
                        f.write(json.dumps(item, ensure_ascii=False) + "\n")

                all_mpc_data.extend(sft_data)
                mpc_id += len(qa_pairs)
                print(f"✅ 生成 {len(qa_pairs)} 个")
            else:
                print("❌ 失败")

    print(f"\n✅ MPC 部分完成！共生成 {len(all_mpc_data)} 个QA对")

    # ========== 汇总统计 ==========
    print("\n" + "="*60)
    print("📊 最终统计")
    print("="*60)
    print(f"RL 数据: {len(all_rl_data)} 条")
    print(f"PID 数据: {len(all_pid_data)} 条")
    print(f"MPC 数据: {len(all_mpc_data)} 条")
    print(f"新增总计: {len(all_rl_data) + len(all_pid_data) + len(all_mpc_data)} 条")
    print(f"\n加上原有 390 条，总数据量约: {390 + len(all_rl_data) + len(all_pid_data) + len(all_mpc_data)} 条")


if __name__ == "__main__":
    main()
