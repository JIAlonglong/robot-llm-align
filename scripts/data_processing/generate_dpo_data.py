#!/usr/bin/env python3
"""
DPO 偏好数据生成 - 第一步
用 DeepSeek-V3.2 生成 500 个问题 + chosen 高质量回答
- RL 相关问题：喂教材章节作为上下文
- 其他主题：直接生成
"""

import json
import os
import time
import re
from typing import List, Dict, Optional
from pypdf import PdfReader
import openai
from tqdm import tqdm

API_KEY  = os.environ.get("SILICONFLOW_API_KEY", "")
BASE_URL = "https://api.siliconflow.cn/v1"
MODEL    = "Pro/deepseek-ai/DeepSeek-V3.2"
OUTPUT   = "dataset/dpo_chosen.jsonl"
PDF_RL       = "references/textbooks/RLbook2020.pdf"
PDF_CONTROL  = "references/textbooks/feedback_systems_astrom_murray.pdf"
PDF_ROBOTICS = "references/textbooks/modern_robotics_lynch_park.pdf"

SYSTEM_PROMPT = "你是一个机器人控制与强化学习领域的专家，擅长强化学习、PID控制、MPC、机器人运动学等技术。"

# RL 教材章节（RLbook2020）
RL_CHAPTERS = [
    {"name": "MDP",                 "start": 47,  "end": 65,  "pdf": PDF_RL},
    {"name": "Dynamic Programming", "start": 73,  "end": 90,  "pdf": PDF_RL},
    {"name": "Monte Carlo",         "start": 91,  "end": 110, "pdf": PDF_RL},
    {"name": "TD Learning",         "start": 119, "end": 145, "pdf": PDF_RL},
    {"name": "Function Approx",     "start": 197, "end": 215, "pdf": PDF_RL},
    {"name": "Policy Gradient",     "start": 321, "end": 345, "pdf": PDF_RL},
    {"name": "Actor-Critic",        "start": 335, "end": 355, "pdf": PDF_RL},
]

# 控制理论教材章节（Feedback Systems）
CONTROL_CHAPTERS = [
    {"name": "PID Control",         "start": 11, "end": 11, "pdf": PDF_CONTROL},
    {"name": "Frequency Analysis",  "start": 10, "end": 10, "pdf": PDF_CONTROL},
    {"name": "State Feedback",      "start": 7,  "end": 7,  "pdf": PDF_CONTROL},
    {"name": "Stability",           "start": 5,  "end": 5,  "pdf": PDF_CONTROL},
    {"name": "Robust Performance",  "start": 13, "end": 13, "pdf": PDF_CONTROL},
]

# 机器人学教材章节（Modern Robotics）
ROBOTICS_CHAPTERS = [
    {"name": "Kinematics",           "start": 4,  "end": 6,  "pdf": PDF_ROBOTICS},
    {"name": "Velocity Kinematics",  "start": 5,  "end": 6,  "pdf": PDF_ROBOTICS},
    {"name": "Inverse Kinematics",   "start": 6,  "end": 7,  "pdf": PDF_ROBOTICS},
    {"name": "Dynamics",             "start": 8,  "end": 9,  "pdf": PDF_ROBOTICS},
    {"name": "Trajectory Generation","start": 9,  "end": 10, "pdf": PDF_ROBOTICS},
    {"name": "Robot Control",        "start": 11, "end": 12, "pdf": PDF_ROBOTICS},
]

# 各主题配置
TOPICS = [
    {"name": "reinforcement_learning", "desc": "强化学习：MDP、值函数、策略梯度、Q-learning、Actor-Critic、PPO、SAC、奖励设计", "count": 120, "chapters": RL_CHAPTERS},
    {"name": "pid_control",            "desc": "PID控制：比例积分微分原理、参数整定（Z-N法）、稳定性分析、抗积分饱和、工程应用", "count": 80,  "chapters": CONTROL_CHAPTERS},
    {"name": "mpc",                    "desc": "模型预测控制MPC：滚动优化、约束处理、终端条件、稳定性、非线性MPC、经济MPC", "count": 80,  "chapters": CONTROL_CHAPTERS},
    {"name": "robot_kinematics",       "desc": "机器人运动学与动力学：正逆运动学、DH参数、雅可比矩阵、轨迹规划、力控制", "count": 80,  "chapters": ROBOTICS_CHAPTERS},
    {"name": "robot_learning",         "desc": "机器人学习：模仿学习、sim-to-real迁移、多任务RL、课程学习、机器人抓取", "count": 80,  "chapters": ROBOTICS_CHAPTERS},
    {"name": "general_control",        "desc": "通用控制理论：状态空间、李雅普诺夫稳定性、鲁棒控制、自适应控制、卡尔曼滤波", "count": 60,  "chapters": CONTROL_CHAPTERS},
]


def get_client():
    return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def extract_pdf_text(pdf_path: str, start_page: int, end_page: int) -> str:
    """提取 PDF 章节文本"""
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    text = ""
    for page in reader.pages[min(start_page - 1, total - 1):min(end_page, total)]:
        text += page.extract_text() or ""
    return text[:4000]


def get_context(chapters) -> tuple:
    """随机取一个章节作为上下文"""
    import random
    chapter = random.choice(chapters)
    text = extract_pdf_text(chapter["pdf"], chapter["start"], chapter["end"])
    return text, chapter["name"]


def generate_questions_with_context(client, topic: Dict, context: str, chapter_name: str, batch_size: int) -> List[str]:
    """基于教材上下文生成 RL 问题"""
    prompt = f"""基于以下教材内容（{chapter_name}章节），生成 {batch_size} 个深度技术问题。

教材内容：
{context}

要求：
- 问题要有深度，覆盖概念理解、公式推导、算法对比、实际应用
- 问题要具体，不要太宽泛
- 中文提问，每行一个问题，不要编号

直接输出问题列表："""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.7,
            max_tokens=800,
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        return [l.strip().lstrip("0123456789.-、） ") for l in lines if l.strip() and len(l.strip()) > 5][:batch_size]
    except Exception as e:
        print(f"  生成问题失败: {e}")
        return []



def generate_chosen_with_context(client, question: str, context: str) -> str:
    """基于教材上下文生成 chosen（RL 专用）"""
    prompt = f"""请基于以下教材内容，对问题给出专业、详细、准确的回答。

教材参考：
{context[:2000]}

问题：{question}

要求：
- 回答要全面，包含定义、原理、公式（LaTeX格式）、应用场景
- 严格基于教材内容，确保准确性
- 长度 200-400 字"""

    try:
        resp = client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.2,
            max_tokens=600,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  生成 chosen 失败: {e}")
        return ""


def main():
    client = get_client()
    os.makedirs("dataset", exist_ok=True)

    # 断点续传：读取已有数据，统计各主题已生成数量
    existing = {}
    total = 0
    if os.path.exists(OUTPUT):
        with open(OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                existing[rec["topic"]] = existing.get(rec["topic"], 0) + 1
                total += 1
        print(f"📂 已有 {total} 条数据，断点续传...")
        for t, c in existing.items():
            print(f"  {t}: {c} 条")

    for topic in TOPICS:
        already = existing.get(topic["name"], 0)
        needed  = topic["count"] - already

        if needed <= 0:
            print(f"\n✅ {topic['name']} 已完成，跳过")
            continue

        print(f"\n{'='*60}")
        print(f"主题: {topic['name']} | 已有: {already} | 还需: {needed}")
        print(f"{'='*60}")

        count = 0
        with tqdm(total=needed, desc=topic["name"]) as pbar:
            while count < needed:
                batch = min(10, needed - count)

                # 所有主题都用教材上下文
                context, chapter_name = get_context(topic["chapters"])
                questions = generate_questions_with_context(client, topic, context, chapter_name, batch)

                if not questions:
                    time.sleep(2)
                    continue

                for q in questions:
                    if count >= needed:
                        break

                    chosen = generate_chosen_with_context(client, q, context)
                    if not chosen:
                        continue

                    record = {
                        "id":       f"dpo_{topic['name']}_{total:04d}",
                        "topic":    topic["name"],
                        "prompt":   q,
                        "chosen":   chosen,
                        "rejected": "",
                    }

                    with open(OUTPUT, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    count += 1
                    total += 1
                    pbar.update(1)

                time.sleep(0.5)

    print(f"\n✅ 完成！共生成 {total} 条 chosen 数据")
    print(f"📁 保存到: {OUTPUT}")


if __name__ == "__main__":
    main()
