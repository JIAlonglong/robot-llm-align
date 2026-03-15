#!/usr/bin/env python3
"""
DPO 偏好数据生成 - 终极增强版
- 混合模型：每主题 5 条 R1（带思维链），其余用 V3.2
- 真实工具执行
- 多轮对话（最多10轮）
"""

import json
import os
import sys
import time
import re
from typing import List, Dict, Tuple
from pypdf import PdfReader
import openai
from tqdm import tqdm

sys.path.insert(0, "/home/liujl/big_model/robot-llm-align/scripts/agent")
from tool_registry import ToolRegistry
from tools.python_robotics_tools import register_robotics_tools

API_KEY   = os.environ.get("SILICONFLOW_API_KEY", "")
BASE_URL  = "https://api.siliconflow.cn/v1"
MODEL_V32 = "Pro/deepseek-ai/DeepSeek-V3.2"
MODEL_R1  = "Pro/deepseek-ai/DeepSeek-R1"
OUTPUT    = "dataset/dpo_chosen_with_cot_tools.jsonl"

PDF_RL       = "references/textbooks/RLbook2020.pdf"
PDF_CONTROL  = "references/textbooks/feedback_systems_astrom_murray.pdf"
PDF_ROBOTICS = "references/textbooks/modern_robotics_lynch_park.pdf"

SYSTEM_PROMPT = """你是一个机器人控制与强化学习领域的专家，擅长强化学习、PID控制、MPC、机器人运动学等技术。

你可以调用以下工具：
- rrt_planning(start_x, start_y, goal_x, goal_y, obstacle_list) - RRT 路径规划
- astar_planning(start_x, start_y, goal_x, goal_y, grid_size, robot_radius, obstacle_list) - A* 规划
- cubic_spline_planning(waypoints) - 三次样条轨迹，waypoints="x1,y1;x2,y2;..."
- simulate_pid(kp, ki, kd, setpoint, duration) - PID 仿真
- lqr_steering_control(x, y, yaw, v, ref_path) - LQR 控制
- mpc_control(x, y, yaw, v, ref_path, horizon) - MPC 控制
- ekf_localization(state, control, measurement) - EKF 定位，state="x,y,yaw"
- particle_filter_localization(initial_state, measurements, num_particles) - 粒子滤波
- arm_forward_kinematics(joint_angles, link_lengths) - 正运动学，joint_angles="θ1,θ2,θ3"
- plot_path_comparison(paths, labels, title) - 路径对比图

调用格式：<tool_call>function_name(param1=value1, param2=value2)</tool_call>
只在需要仿真、规划、计算时调用工具，纯理论问题直接回答。"""

RL_CHAPTERS = [
    {"name": "MDP",                 "start": 47,  "end": 65,  "pdf": PDF_RL},
    {"name": "Dynamic Programming", "start": 73,  "end": 90,  "pdf": PDF_RL},
    {"name": "Monte Carlo",         "start": 91,  "end": 110, "pdf": PDF_RL},
    {"name": "TD Learning",         "start": 119, "end": 145, "pdf": PDF_RL},
    {"name": "Function Approx",     "start": 197, "end": 215, "pdf": PDF_RL},
    {"name": "Policy Gradient",     "start": 321, "end": 345, "pdf": PDF_RL},
    {"name": "Actor-Critic",        "start": 335, "end": 355, "pdf": PDF_RL},
]

CONTROL_CHAPTERS = [
    {"name": "PID Control",        "start": 11, "end": 11, "pdf": PDF_CONTROL},
    {"name": "Frequency Analysis", "start": 10, "end": 10, "pdf": PDF_CONTROL},
    {"name": "State Feedback",     "start": 7,  "end": 7,  "pdf": PDF_CONTROL},
    {"name": "Stability",          "start": 5,  "end": 5,  "pdf": PDF_CONTROL},
    {"name": "Robust Performance", "start": 13, "end": 13, "pdf": PDF_CONTROL},
]

ROBOTICS_CHAPTERS = [
    {"name": "Kinematics",            "start": 4,  "end": 6,  "pdf": PDF_ROBOTICS},
    {"name": "Velocity Kinematics",   "start": 5,  "end": 6,  "pdf": PDF_ROBOTICS},
    {"name": "Inverse Kinematics",    "start": 6,  "end": 7,  "pdf": PDF_ROBOTICS},
    {"name": "Dynamics",              "start": 8,  "end": 9,  "pdf": PDF_ROBOTICS},
    {"name": "Trajectory Generation", "start": 9,  "end": 10, "pdf": PDF_ROBOTICS},
    {"name": "Robot Control",         "start": 11, "end": 12, "pdf": PDF_ROBOTICS},
]

# r1_count: 每主题用 R1 生成的条数，其余用 V3.2
TOPICS = [
    {"name": "reinforcement_learning", "desc": "强化学习：MDP、值函数、策略梯度、Q-learning、Actor-Critic、PPO、SAC", "count": 120, "chapters": RL_CHAPTERS,       "tool_ratio": 0.4, "r1_count": 5},
    {"name": "pid_control",            "desc": "PID控制：参数整定、稳定性分析、抗积分饱和",                          "count": 80,  "chapters": CONTROL_CHAPTERS,  "tool_ratio": 0.6, "r1_count": 5},
    {"name": "mpc",                    "desc": "模型预测控制MPC：滚动优化、约束处理、稳定性",                        "count": 80,  "chapters": CONTROL_CHAPTERS,  "tool_ratio": 0.6, "r1_count": 5},
    {"name": "robot_kinematics",       "desc": "机器人运动学：正逆运动学、DH参数、雅可比矩阵、轨迹规划",            "count": 80,  "chapters": ROBOTICS_CHAPTERS, "tool_ratio": 0.5, "r1_count": 5},
    {"name": "robot_learning",         "desc": "机器人学习：模仿学习、sim-to-real、多任务RL",                       "count": 80,  "chapters": ROBOTICS_CHAPTERS, "tool_ratio": 0.4, "r1_count": 5},
    {"name": "general_control",        "desc": "通用控制：状态空间、李雅普诺夫稳定性、鲁棒控制",                    "count": 60,  "chapters": CONTROL_CHAPTERS,  "tool_ratio": 0.3, "r1_count": 5},
]


def get_client():
    return openai.OpenAI(api_key=API_KEY, base_url=BASE_URL)


def extract_pdf_text(pdf_path: str, start_page: int, end_page: int) -> str:
    reader = PdfReader(pdf_path)
    total = len(reader.pages)
    text = ""
    for page in reader.pages[min(start_page - 1, total - 1):min(end_page, total)]:
        text += page.extract_text() or ""
    return text[:4000]


def get_context(chapters) -> Tuple[str, str]:
    import random
    chapter = random.choice(chapters)
    text = extract_pdf_text(chapter["pdf"], chapter["start"], chapter["end"])
    return text, chapter["name"]


def generate_questions(client, topic: Dict, context: str, chapter_name: str, batch_size: int) -> List[str]:
    tool_hint = f"\n- 其中约 {int(batch_size * topic['tool_ratio'])} 个问题需要仿真/计算/规划（会用到工具）" if topic['tool_ratio'] > 0 else ""
    prompt = f"""基于以下教材内容（{chapter_name}章节），生成 {batch_size} 个深度技术问题。

教材内容：
{context}

要求：
- 问题要有深度，覆盖概念理解、公式推导、算法对比、实际应用{tool_hint}
- 问题要具体，不要太宽泛
- 中文提问，每行一个问题，不要编号

直接输出问题列表："""
    try:
        resp = client.chat.completions.create(
            model=MODEL_V32,  # 问题生成始终用 V3.2，快
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": prompt},
            ],
            temperature=0.8,
            max_tokens=800,
        )
        lines = resp.choices[0].message.content.strip().split("\n")
        return [l.strip().lstrip("0123456789.-、） ") for l in lines if l.strip() and len(l.strip()) > 5][:batch_size]
    except Exception as e:
        print(f"  生成问题失败: {e}")
        return []


def execute_tool_call(tool_registry: ToolRegistry, tool_call_str: str) -> str:
    try:
        result = tool_registry.execute(tool_call_str)
        return json.dumps(result, ensure_ascii=False)
    except Exception as e:
        return json.dumps({"error": str(e)}, ensure_ascii=False)


def generate_chosen(client, tool_registry: ToolRegistry, question: str, context: str, model: str, max_turns: int = 10) -> Tuple[str, bool]:
    """生成 chosen，支持真实工具执行和多轮对话"""
    prompt = f"""请基于以下教材内容，对问题给出专业、详细、准确的回答。

教材参考：
{context[:2000]}

问题：{question}

要求：
- 先思考（分析问题类型、选择方法、判断是否需要工具）
- 如果需要仿真/计算/规划，用 <tool_call>function_name(param1=value1)</tool_call> 调用工具
- 回答要包含定义、原理、公式（LaTeX格式）
- 长度 300-500 字"""

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": prompt},
    ]

    full_response = ""
    has_tool = False

    try:
        for turn in range(max_turns):
            resp = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=0.3,
                max_tokens=1200 if model == MODEL_R1 else 800,
            )
            response = resp.choices[0].message.content.strip()

            # R1 的思维链在 reasoning_content 字段
            reasoning = ""
            if model == MODEL_R1 and hasattr(resp.choices[0].message, "reasoning_content"):
                reasoning = resp.choices[0].message.reasoning_content or ""

            # 拼接思维链 + 回答
            if reasoning:
                turn_text = f"<think>\n{reasoning}\n</think>\n\n{response}"
            else:
                turn_text = response

            full_response += turn_text
            messages.append({"role": "assistant", "content": response})

            # 检测工具调用
            tool_calls = re.findall(r'<tool_call>(.*?)</tool_call>', response, re.DOTALL)
            if not tool_calls:
                break

            has_tool = True
            # 执行工具，把结果追加到对话
            tool_results_text = ""
            for tc in tool_calls:
                result = execute_tool_call(tool_registry, f"<tool_call>{tc}</tool_call>")
                tool_results_text += f"<tool_result>{result}</tool_result>\n"
                full_response += f"\n{tool_results_text}"

            follow_up = f"""{tool_results_text}
请基于以上工具结果继续分析。如果结果有误，可以：
1. 思考原因并调整参数重新调用工具
2. 或者基于现有结果给出最终结论"""
            messages.append({"role": "user", "content": follow_up})

        return full_response.strip(), has_tool

    except Exception as e:
        print(f"  生成 chosen 失败 ({model}): {e}")
        return "", False


def main():
    client = get_client()

    print("初始化工具...")
    tool_registry = ToolRegistry()
    register_robotics_tools(tool_registry)

    os.makedirs("dataset", exist_ok=True)

    # 断点续传：记录每个主题已有多少条，以及其中 R1 有多少条
    existing = {}   # topic -> total count
    r1_done  = {}   # topic -> r1 count
    total = 0

    if os.path.exists(OUTPUT):
        with open(OUTPUT, "r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                t = rec["topic"]
                existing[t] = existing.get(t, 0) + 1
                if rec.get("model") == MODEL_R1:
                    r1_done[t] = r1_done.get(t, 0) + 1
                total += 1
        print(f"已有 {total} 条数据，断点续传...")
        for t, c in existing.items():
            print(f"  {t}: {c} 条 (R1: {r1_done.get(t, 0)})")

    for topic in TOPICS:
        already   = existing.get(topic["name"], 0)
        r1_already = r1_done.get(topic["name"], 0)
        needed    = topic["count"] - already

        if needed <= 0:
            print(f"\n✅ {topic['name']} 已完成，跳过")
            continue

        r1_needed  = max(0, topic["r1_count"] - r1_already)
        v32_needed = needed - r1_needed

        print(f"\n{'='*60}")
        print(f"主题: {topic['name']} | 还需: {needed} (R1: {r1_needed}, V3.2: {v32_needed})")
        print(f"{'='*60}")

        count = 0
        with tqdm(total=needed, desc=topic["name"]) as pbar:
            while count < needed:
                batch = min(5, needed - count)
                context, chapter_name = get_context(topic["chapters"])
                questions = generate_questions(client, topic, context, chapter_name, batch)

                if not questions:
                    time.sleep(2)
                    continue

                for q in questions:
                    if count >= needed:
                        break

                    # 决定用哪个模型
                    use_r1 = (count < r1_needed)
                    model  = MODEL_R1 if use_r1 else MODEL_V32

                    chosen, has_tool = generate_chosen(client, tool_registry, q, context, model)
                    if not chosen:
                        continue

                    record = {
                        "id":       f"dpo_{topic['name']}_{total:04d}",
                        "topic":    topic["name"],
                        "model":    model,
                        "prompt":   q,
                        "chosen":   chosen,
                        "rejected": "",
                        "has_tool": has_tool,
                        "has_cot":  "<think>" in chosen,
                    }

                    with open(OUTPUT, "a", encoding="utf-8") as f:
                        f.write(json.dumps(record, ensure_ascii=False) + "\n")

                    count += 1
                    total += 1
                    pbar.update(1)
                    time.sleep(2 if use_r1 else 0.5)

    print(f"\n✅ 完成！共生成 {total} 条数据")

    # 统计
    tool_count = cot_count = r1_count = 0
    with open(OUTPUT, "r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            if d.get("has_tool"): tool_count += 1
            if d.get("has_cot"):  cot_count  += 1
            if d.get("model") == MODEL_R1: r1_count += 1
    print(f"R1 数据: {r1_count}/{total} ({r1_count/total:.1%})")
    print(f"工具调用: {tool_count}/{total} ({tool_count/total:.1%})")
    print(f"思维链: {cot_count}/{total} ({cot_count/total:.1%})")


if __name__ == "__main__":
    main()
