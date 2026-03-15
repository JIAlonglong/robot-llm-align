"""
train_agent_lightning.py — 用 agent-lightning 训练 RobotControlAgent

两种模式：
  --mode apo    : 自动提示优化（APO），不需要 GPU，优化 system prompt
  --mode collect: 只收集轨迹到 SQLite，用于后续 DPO 数据生成
  --mode dev    : 快速单步调试（不训练）

运行方式：
    conda run -n LLM python scripts/train_agent_lightning.py --mode dev
    conda run -n LLM python scripts/train_agent_lightning.py --mode collect --rollouts 50
"""

import sys
import os
import argparse

# 把 agent 目录加入路径
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from agentlightning import Trainer
from agentlightning.store import InMemoryLightningStore
from agentlightning.execution import SharedMemoryExecutionStrategy
from agentlightning.types import PromptTemplate

# ReAct system prompt — APO 会自动优化这个
REACT_SYSTEM_PROMPT = (
    "你是一个机器人控制 Agent。根据任务描述，选择合适的工具完成任务。\n"
    "可用工具：simulate_pid, rrt_planning, astar_planning, ekf_localization, "
    "arm_forward_kinematics, cartpole_reset, cartpole_step\n"
    "输出格式：\nThought: <分析>\n<tool_call>tool_name(arg=val)</tool_call>"
)

INITIAL_RESOURCES = {
    "system_prompt": PromptTemplate(template=REACT_SYSTEM_PROMPT, engine="f-string")
}

# ── 任务数据集 ────────────────────────────────────────────────

TRAIN_TASKS = [
    # CartPole 平衡
    {"type": "cartpole", "params": {"max_steps": 200}},
    {"type": "cartpole", "params": {"max_steps": 200}},
    {"type": "cartpole", "params": {"max_steps": 200}},
    # PID 调参
    {"type": "pid", "params": {"kp": 1.0, "ki": 0.1, "kd": 0.05}},
    {"type": "pid", "params": {"kp": 2.0, "ki": 0.0, "kd": 0.1}},
    {"type": "pid", "params": {"kp": 0.5, "ki": 0.5, "kd": 0.0}},
    # RRT 路径规划
    {"type": "rrt", "params": {"start_x": 0, "start_y": 0, "goal_x": 5, "goal_y": 5}},
    {"type": "rrt", "params": {"start_x": 0, "start_y": 0, "goal_x": 8, "goal_y": 3}},
    # A* 路径规划
    {"type": "astar", "params": {"start_x": 0, "start_y": 0, "goal_x": 8, "goal_y": 8}},
    # EKF 定位
    {"type": "ekf", "params": {"state": "0,0,0", "control": "1,0.1", "measurement": "0.1,0.1"}},
    # 机械臂正运动学
    {"type": "arm_fk", "params": {"joint_angles": "0.5,0.5,0.5", "link_lengths": "1,1,1"}},
]

VAL_TASKS = [
    {"type": "cartpole", "params": {"max_steps": 200}},
    {"type": "pid",      "params": {"kp": 1.5, "ki": 0.2, "kd": 0.08}},
    {"type": "rrt",      "params": {"start_x": 1, "start_y": 1, "goal_x": 6, "goal_y": 6}},
]


# ── 模式：collect（只收集轨迹）────────────────────────────────

def run_collect(args):
    from lit_robot_agent import LitRobotAgent

    store = InMemoryLightningStore()

    trainer = Trainer(
        n_runners=1,
        max_rollouts=args.rollouts,
        store=store,
        strategy=SharedMemoryExecutionStrategy(),
        initial_resources=INITIAL_RESOURCES,
    )

    agent = LitRobotAgent(model_path=args.model_path)
    print(f"[collect] 开始收集轨迹，目标 {args.rollouts} 条 rollout...")
    trainer.fit(agent, TRAIN_TASKS, val_dataset=VAL_TASKS)
    print(f"[collect] 完成，轨迹已存入 dataset/agent_traces.db")


# ── 模式：apo（自动提示优化）─────────────────────────────────

def run_apo(args):
    from openai import AsyncOpenAI
    from agentlightning.algorithm.apo import APO
    from lit_robot_agent import LitRobotAgent

    client = AsyncOpenAI(
        api_key=os.environ.get("OPENAI_API_KEY", ""),
        base_url=os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1"),
    )

    store = InMemoryLightningStore()

    algorithm = APO(
        async_openai_client=client,
        gradient_model=args.gradient_model,
        apply_edit_model=args.gradient_model,
        beam_width=2,
        branch_factor=2,
        beam_rounds=args.rounds,
        val_batch_size=len(VAL_TASKS),
    )

    trainer = Trainer(
        n_runners=1,
        max_rollouts=args.rollouts,
        store=store,
        algorithm=algorithm,
        strategy=SharedMemoryExecutionStrategy(),
        initial_resources=INITIAL_RESOURCES,
    )

    agent = LitRobotAgent(model_path=args.model_path)
    print(f"[apo] 开始 APO 优化，beam_rounds={args.rounds}...")
    trainer.fit(agent, TRAIN_TASKS, val_dataset=VAL_TASKS)
    print("[apo] 完成")


# ── 模式：dev（单步调试）─────────────────────────────────────

def run_dev(args):
    from lit_robot_agent import LitRobotAgent

    trainer = Trainer(n_runners=1, max_rollouts=3, strategy=SharedMemoryExecutionStrategy(), initial_resources=INITIAL_RESOURCES)
    # no_model=True 时跳过 GPU 模型加载，用 rule_based_policy 验证框架
    model_path = "" if args.no_model else args.model_path
    agent = LitRobotAgent(model_path=model_path)
    print("[dev] 快速调试模式，执行 3 条 rollout...")
    trainer.dev(agent, TRAIN_TASKS[:3])
    print("[dev] 完成")


# ── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Robot Agent Lightning 训练")
    parser.add_argument("--mode", choices=["apo", "collect", "dev"], default="dev")
    parser.add_argument("--rollouts", type=int, default=20, help="最大 rollout 数量")
    parser.add_argument("--rounds", type=int, default=3, help="APO beam rounds")
    parser.add_argument(
        "--model-path",
        default="/home/liujl/big_model/robot-llm-align/checkpoints/sft_qwen1.5b_with_tools",
    )
    parser.add_argument(
        "--gradient-model",
        default="gpt-4o-mini",
        help="APO 使用的梯度模型（需要 OpenAI API）",
    )
    parser.add_argument("--no-model", action="store_true", help="跳过 LLM 加载，用规则策略（调试用）")
    args = parser.parse_args()

    if args.mode == "collect":
        run_collect(args)
    elif args.mode == "apo":
        run_apo(args)
    else:
        run_dev(args)


if __name__ == "__main__":
    main()
