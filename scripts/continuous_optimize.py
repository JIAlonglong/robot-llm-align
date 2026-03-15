"""
continuous_optimize.py — 12 小时持续收集 + 优化 system prompt

架构：
  - 提问模型（DeepSeek-V3.2）：生成多样化机器人控制任务
  - 执行层（LitRobotAgent）：用 rule_based_policy 执行任务，记录奖励
  - 优化模型（DeepSeek-V3.2）：分析失败案例，重写 system prompt
  - 每轮结果写入 dataset/optimization_log.jsonl

运行：
    conda run -n LLM python scripts/continuous_optimize.py
    conda run -n LLM python scripts/continuous_optimize.py --hours 12 --rounds-per-cycle 10
"""

import sys
import os
import json
import time
import argparse
import random
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from openai import OpenAI

# ── 硅基流动配置 ──────────────────────────────────────────────
SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
TASK_GEN_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"
OPTIMIZER_MODEL = "Pro/deepseek-ai/DeepSeek-V3.2"

# ── 工具说明（告诉提问模型有哪些工具）────────────────────────
TOOLS_DESCRIPTION = """
可用工具列表（机器人控制领域）：

1. simulate_pid(kp, ki, kd, setpoint=1.0, duration=10.0)
   - 仿真 PID 控制器（一阶系统），返回超调量、调节时间、稳态误差
   - 示例：kp=1.5, ki=0.1, kd=0.05

2. rrt_planning(start_x, start_y, goal_x, goal_y, obstacle_list=None)
   - RRT 随机树路径规划，返回路径点列表和路径长度
   - obstacle_list 格式："x1,y1,r1;x2,y2,r2"（圆形障碍物）

3. astar_planning(start_x, start_y, goal_x, goal_y, grid_size=1.0)
   - A* 网格路径规划，适合有障碍物的结构化环境

4. cubic_spline_planning(waypoints)
   - 三次样条轨迹生成，waypoints 格式："x1,y1;x2,y2;x3,y3"

5. lqr_steering_control(x, y, yaw, v, ref_path)
   - LQR 最优转向控制，返回转向角和跟踪误差

6. ekf_localization(state, control, measurement)
   - 扩展卡尔曼滤波定位，state="x,y,yaw", control="v,omega"

7. arm_forward_kinematics(joint_angles, link_lengths="1.0,1.0,1.0")
   - 机械臂正运动学，joint_angles="theta1,theta2,theta3"（弧度）

8. cartpole_reset() / cartpole_step(action)
   - CartPole 倒立摆仿真，action=0（左推）或 1（右推）

9. plot_path_comparison(paths, labels)
   - 绘制多条路径对比图
"""

# ── 初始 system prompt ────────────────────────────────────────
INITIAL_SYSTEM_PROMPT = """你是一个专业的机器人控制 Agent。根据用户的任务描述，选择合适的工具完成任务。

工具调用格式（严格遵守）：
Thought: <分析当前任务，选择工具和参数>
<tool_call>tool_name(arg1=val1, arg2=val2)</tool_call>

规则：
- 每次只调用一个工具
- 参数必须是数值或字符串字面量
- 根据工具返回结果决定下一步
- 任务完成后输出 Final Answer: <结论>"""


def make_client() -> OpenAI:
    return OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)


# ── 提问模型：生成任务 ────────────────────────────────────────

TASK_CATEGORIES = [
    ("pid",    "PID 控制器调参"),
    ("rrt",    "RRT 路径规划"),
    ("astar",  "A* 路径规划"),
    ("ekf",    "EKF 定位"),
    ("arm_fk", "机械臂正运动学"),
    ("cartpole", "CartPole 平衡"),
]

TASK_GEN_SYSTEM = f"""你是一个机器人控制领域的任务生成器。
你的职责是生成多样化的机器人控制任务，供 Agent 使用工具完成。

{TOOLS_DESCRIPTION}

输出格式（严格 JSON，不要有其他内容）：
{{
  "type": "pid|rrt|astar|ekf|arm_fk|cartpole",
  "description": "任务描述（中文，1-2句话）",
  "params": {{参数字典，根据任务类型填写}}
}}

params 示例：
- pid: {{"kp": 1.5, "ki": 0.1, "kd": 0.05}}
- rrt: {{"start_x": 0, "start_y": 0, "goal_x": 7, "goal_y": 5}}
- astar: {{"start_x": 0, "start_y": 0, "goal_x": 8, "goal_y": 8}}
- ekf: {{"state": "0,0,0", "control": "1.0,0.1", "measurement": "0.1,0.05"}}
- arm_fk: {{"joint_angles": "0.5,1.0,0.3", "link_lengths": "1,1,0.5"}}
- cartpole: {{"max_steps": 200}}
"""


def generate_task(client: OpenAI, cycle: int) -> dict:
    """用提问模型生成一个任务"""
    category, category_name = random.choice(TASK_CATEGORIES)

    # 每隔几轮换一个类别，增加多样性
    if cycle % 3 == 0:
        category, category_name = TASK_CATEGORIES[cycle % len(TASK_CATEGORIES)]

    prompt = (
        f"请生成一个关于【{category_name}】的机器人控制任务。"
        f"要求参数有一定变化，不要总是用默认值。这是第 {cycle} 轮任务。"
    )

    try:
        resp = client.chat.completions.create(
            model=TASK_GEN_MODEL,
            messages=[
                {"role": "system", "content": TASK_GEN_SYSTEM},
                {"role": "user", "content": prompt},
            ],
            temperature=0.9,
            max_tokens=256,
        )
        raw = resp.choices[0].message.content.strip()
        # 提取 JSON
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        task = json.loads(raw)
        # 确保有必要字段
        if "type" not in task or "params" not in task:
            raise ValueError("缺少必要字段")
        return task
    except Exception as e:
        # 降级：随机返回一个固定任务
        fallback_tasks = [
            {"type": "pid",    "params": {"kp": round(random.uniform(0.5, 3.0), 2),
                                          "ki": round(random.uniform(0, 0.5), 2),
                                          "kd": round(random.uniform(0, 0.2), 2)}},
            {"type": "rrt",    "params": {"start_x": 0, "start_y": 0,
                                          "goal_x": random.randint(3, 9),
                                          "goal_y": random.randint(3, 9)}},
            {"type": "cartpole", "params": {"max_steps": 200}},
        ]
        print(f"  [TaskGen] 生成失败({e})，使用 fallback 任务")
        return random.choice(fallback_tasks)


# ── 执行层：运行任务，返回结果和奖励 ─────────────────────────

def execute_task(task: dict, system_prompt: str) -> dict:
    """执行任务，返回 {result, reward, task_type, success}"""
    from tool_registry import ToolRegistry
    from tools.python_robotics_tools import register_robotics_tools
    from reward import tool_call_reward
    import math, random as rnd

    # 构建 registry
    reg = ToolRegistry()

    class _CartPole:
        def __init__(self):
            self.state = [0.0, 0.0, 0.0, 0.0]
        def reset(self):
            self.state = [rnd.uniform(-0.05, 0.05) for _ in range(4)]
            return {"obs": list(self.state)}
        def step(self, action: int):
            x, x_dot, theta, theta_dot = self.state
            force = 10.0 if action == 1 else -10.0
            g, m_c, m_p, l = 9.8, 1.0, 0.1, 0.5
            total_mass = m_c + m_p
            dt = 0.02
            cos_t = math.cos(theta); sin_t = math.sin(theta)
            temp = (force + m_p * l * theta_dot**2 * sin_t) / total_mass
            theta_acc = (g * sin_t - cos_t * temp) / (l * (4/3 - m_p * cos_t**2 / total_mass))
            x_acc = temp - m_p * l * theta_acc * cos_t / total_mass
            x += dt * x_dot; x_dot += dt * x_acc
            theta += dt * theta_dot; theta_dot += dt * theta_acc
            self.state = [x, x_dot, theta, theta_dot]
            done = abs(x) > 2.4 or abs(theta) > 0.2095
            return {"obs": list(self.state), "reward": 1.0, "done": done}

    sim = _CartPole()
    reg.register("cartpole_reset", sim.reset)
    reg.register("cartpole_step", sim.step)
    register_robotics_tools(reg)

    task_type = task.get("type", "pid")
    params = task.get("params", {})

    tool_map = {
        "pid":    ("simulate_pid",          ["kp", "ki", "kd"]),
        "rrt":    ("rrt_planning",           ["start_x", "start_y", "goal_x", "goal_y"]),
        "astar":  ("astar_planning",         ["start_x", "start_y", "goal_x", "goal_y"]),
        "ekf":    ("ekf_localization",       ["state", "control", "measurement"]),
        "arm_fk": ("arm_forward_kinematics", ["joint_angles", "link_lengths"]),
    }

    if task_type == "cartpole":
        from agent_executor import AgentExecutor, rule_based_policy
        max_steps = params.get("max_steps", 200)
        executor = AgentExecutor(registry=reg, policy=rule_based_policy, max_steps=max_steps)
        result = executor.run(task="balance cartpole")
    elif task_type in tool_map:
        tool_name, arg_keys = tool_map[task_type]
        args_str = ", ".join(f"{k}={params[k]}" for k in arg_keys if k in params)
        call_str = f"<tool_call>{tool_name}({args_str})</tool_call>"
        result = reg.execute(call_str)
    else:
        result = {"error": f"未知任务类型: {task_type}"}

    reward = tool_call_reward(result, task_type)
    success = reward > 0.5

    # 去掉 plot_base64 避免日志太大
    if isinstance(result, dict):
        result.pop("plot_base64", None)

    return {"result": result, "reward": reward, "task_type": task_type, "success": success}


# ── 优化模型：分析结果，重写 system prompt ────────────────────

OPTIMIZER_SYSTEM = """你是一个 AI Agent 系统优化专家。
你的任务是分析 Agent 的执行记录，找出 system prompt 的不足，并给出改进版本。

分析维度：
1. 工具调用格式是否清晰
2. 参数选择策略是否合理
3. 任务理解是否准确
4. 失败案例的根本原因

输出格式（严格遵守）：
## 问题分析
<简要分析 2-3 个主要问题>

## 优化后的 System Prompt
```
<完整的新 system prompt，直接可用>
```
"""


def optimize_prompt(client: OpenAI, current_prompt: str,
                    execution_records: list, cycle: int) -> str:
    """用优化模型分析执行记录，返回新的 system prompt"""

    # 整理执行记录摘要
    total = len(execution_records)
    success_count = sum(1 for r in execution_records if r["success"])
    avg_reward = sum(r["reward"] for r in execution_records) / max(total, 1)

    # 找出失败案例
    failures = [r for r in execution_records if not r["success"]][:5]
    failure_summary = "\n".join(
        f"- 任务类型: {r['task_type']}, 奖励: {r['reward']:.2f}, 结果: {str(r['result'])[:100]}"
        for r in failures
    ) or "无失败案例"

    # 找出成功案例
    successes = [r for r in execution_records if r["success"]][:3]
    success_summary = "\n".join(
        f"- 任务类型: {r['task_type']}, 奖励: {r['reward']:.2f}"
        for r in successes
    ) or "无成功案例"

    user_msg = f"""第 {cycle} 轮优化分析

当前 System Prompt：
```
{current_prompt}
```

本轮执行统计：
- 总任务数: {total}
- 成功数: {success_count} ({success_count/max(total,1)*100:.1f}%)
- 平均奖励: {avg_reward:.3f}

失败案例：
{failure_summary}

成功案例：
{success_summary}

请分析问题并给出优化后的 system prompt。"""

    try:
        resp = client.chat.completions.create(
            model=OPTIMIZER_MODEL,
            messages=[
                {"role": "system", "content": OPTIMIZER_SYSTEM},
                {"role": "user", "content": user_msg},
            ],
            temperature=0.7,
            max_tokens=1024,
        )
        output = resp.choices[0].message.content.strip()

        # 提取 ``` 之间的 prompt
        if "```" in output:
            parts = output.split("```")
            for i, part in enumerate(parts):
                if i % 2 == 1 and len(part.strip()) > 50:
                    new_prompt = part.strip()
                    if new_prompt.startswith("prompt") or new_prompt.startswith("\n"):
                        new_prompt = new_prompt.lstrip("prompt").strip()
                    return new_prompt
        return current_prompt  # 提取失败，保留原 prompt

    except Exception as e:
        print(f"  [Optimizer] 优化失败: {e}")
        return current_prompt


# ── 主循环 ────────────────────────────────────────────────────

def run(hours: float, rounds_per_cycle: int, log_path: str):
    client = make_client()
    system_prompt = INITIAL_SYSTEM_PROMPT
    end_time = datetime.now() + timedelta(hours=hours)

    os.makedirs(os.path.dirname(log_path), exist_ok=True)

    cycle = 0
    total_tasks = 0
    total_success = 0
    prompt_history = [{"cycle": 0, "prompt": system_prompt, "avg_reward": None}]

    print(f"\n{'='*60}")
    print(f"开始持续优化训练")
    print(f"计划运行: {hours} 小时")
    print(f"每轮任务数: {rounds_per_cycle}")
    print(f"结束时间: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"日志路径: {log_path}")
    print(f"{'='*60}\n")

    while datetime.now() < end_time:
        cycle += 1
        remaining = end_time - datetime.now()
        print(f"\n[Cycle {cycle}] 剩余时间: {str(remaining).split('.')[0]}")
        print(f"  当前 prompt 长度: {len(system_prompt)} 字符")

        # ── 阶段1：生成并执行任务 ──────────────────────────────
        cycle_records = []
        print(f"  生成并执行 {rounds_per_cycle} 个任务...")

        for i in range(rounds_per_cycle):
            # 生成任务
            task = generate_task(client, cycle * rounds_per_cycle + i)

            # 执行任务
            try:
                exec_result = execute_task(task, system_prompt)
            except Exception as e:
                exec_result = {
                    "result": {"error": str(e)},
                    "reward": 0.0,
                    "task_type": task.get("type", "unknown"),
                    "success": False,
                }

            cycle_records.append({
                **exec_result,
                "task": task,
                "cycle": cycle,
                "timestamp": datetime.now().isoformat(),
            })

            total_tasks += 1
            if exec_result["success"]:
                total_success += 1

            status = "✓" if exec_result["success"] else "✗"
            print(f"    [{i+1:2d}/{rounds_per_cycle}] {status} {task.get('type','?'):8s} "
                  f"reward={exec_result['reward']:.3f}")

        # ── 阶段2：统计本轮结果 ────────────────────────────────
        cycle_avg_reward = sum(r["reward"] for r in cycle_records) / len(cycle_records)
        cycle_success_rate = sum(1 for r in cycle_records if r["success"]) / len(cycle_records)
        print(f"  本轮: avg_reward={cycle_avg_reward:.3f}, success={cycle_success_rate*100:.1f}%")
        print(f"  累计: {total_tasks} 任务, {total_success/total_tasks*100:.1f}% 成功率")

        # ── 阶段3：写入日志 ────────────────────────────────────
        with open(log_path, "a", encoding="utf-8") as f:
            for record in cycle_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        # ── 阶段4：优化 system prompt ──────────────────────────
        print(f"  调用优化模型分析结果...")
        new_prompt = optimize_prompt(client, system_prompt, cycle_records, cycle)

        if new_prompt != system_prompt:
            print(f"  [Optimizer] prompt 已更新 ({len(system_prompt)} → {len(new_prompt)} 字符)")
            system_prompt = new_prompt
        else:
            print(f"  [Optimizer] prompt 无变化")

        prompt_history.append({
            "cycle": cycle,
            "prompt": system_prompt,
            "avg_reward": cycle_avg_reward,
            "success_rate": cycle_success_rate,
            "timestamp": datetime.now().isoformat(),
        })

        # 保存 prompt 历史
        prompt_log = log_path.replace(".jsonl", "_prompts.json")
        with open(prompt_log, "w", encoding="utf-8") as f:
            json.dump(prompt_history, f, ensure_ascii=False, indent=2)

        # 检查是否还有时间
        if datetime.now() >= end_time:
            break

        # 短暂休息避免 API 限速
        time.sleep(2)

    # ── 最终报告 ──────────────────────────────────────────────
    print(f"\n{'='*60}")
    print(f"训练完成！")
    print(f"总轮次: {cycle}")
    print(f"总任务: {total_tasks}")
    print(f"总成功率: {total_success/max(total_tasks,1)*100:.1f}%")
    print(f"最终 prompt 长度: {len(system_prompt)} 字符")
    print(f"日志: {log_path}")
    print(f"Prompt 历史: {prompt_log}")
    print(f"{'='*60}")
    print(f"\n最终优化后的 System Prompt:\n{'-'*40}")
    print(system_prompt)


# ── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="12小时持续优化训练")
    parser.add_argument("--hours", type=float, default=12.0, help="运行时长（小时）")
    parser.add_argument("--rounds-per-cycle", type=int, default=15, help="每轮任务数")
    parser.add_argument(
        "--log",
        default="/home/liujl/big_model/robot-llm-align/dataset/optimization_log.jsonl",
        help="日志输出路径",
    )
    args = parser.parse_args()
    run(hours=args.hours, rounds_per_cycle=args.rounds_per_cycle, log_path=args.log)


if __name__ == "__main__":
    main()
