"""
LitRobotAgent — 用 agent-lightning 的 LitAgent 包装现有 AgentExecutor
支持：
  - CartPole 平衡（ReAct 循环）
  - 单次工具调用任务（PID/RRT/A*/EKF/机械臂）

训练时通过 emit_reward 发出奖励信号，供 APO/VERL 算法消费。
"""

import sys
import os
import torch

sys.path.insert(0, os.path.dirname(__file__))

from agentlightning import LitAgent
from agentlightning.emitter import emit_reward

from tool_registry import ToolRegistry
from agent_executor import AgentExecutor
from reward import tool_call_reward


# ── 任务类型定义 ──────────────────────────────────────────────
TASK_TYPES = ["cartpole", "pid", "rrt", "astar", "ekf", "arm_fk"]


def _build_registry() -> ToolRegistry:
    """构建工具注册表（复用 app.py 的逻辑）"""
    import math, random

    class MockCartpole:
        def __init__(self):
            self.state = [0.0, 0.0, 0.0, 0.0]

        def reset(self):
            self.state = [random.uniform(-0.05, 0.05) for _ in range(4)]
            return {"obs": list(self.state)}

        def step(self, action: int):
            x, x_dot, theta, theta_dot = self.state
            force = 10.0 if action == 1 else -10.0
            g, m_c, m_p, l = 9.8, 1.0, 0.1, 0.5
            total_mass = m_c + m_p
            dt = 0.02
            cos_t = math.cos(theta)
            sin_t = math.sin(theta)
            temp = (force + m_p * l * theta_dot**2 * sin_t) / total_mass
            theta_acc = (g * sin_t - cos_t * temp) / (l * (4/3 - m_p * cos_t**2 / total_mass))
            x_acc = temp - m_p * l * theta_acc * cos_t / total_mass
            x += dt * x_dot
            x_dot += dt * x_acc
            theta += dt * theta_dot
            theta_dot += dt * theta_acc
            self.state = [x, x_dot, theta, theta_dot]
            done = abs(x) > 2.4 or abs(theta) > 0.2095
            return {"obs": list(self.state), "reward": 1.0, "done": done}

    sim = MockCartpole()
    reg = ToolRegistry()
    reg.register("cartpole_reset", sim.reset)
    reg.register("cartpole_step", sim.step)

    from tools.python_robotics_tools import register_robotics_tools
    register_robotics_tools(reg)
    return reg


def _load_model(model_path: str):
    """加载 Qwen 模型，失败时返回 None（降级到 rule_based_policy）"""
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            model_path, torch_dtype=torch.float16,
            device_map="cuda:0", trust_remote_code=True
        )
        model.eval()
        print(f"[LitRobotAgent] 模型加载成功: {model_path}")
        return model, tokenizer
    except Exception as e:
        print(f"[LitRobotAgent] 模型加载失败，降级到 rule_based_policy: {e}")
        return None, None


class LitRobotAgent(LitAgent):
    """
    机器人控制 Agent，支持多种任务类型。
    task 格式：{"type": "cartpole"|"pid"|"rrt"|..., "params": {...}}
    """

    def __init__(
        self,
        model_path: str = "/home/liujl/big_model/robot-llm-align/checkpoints/sft_qwen1.5b_with_tools",
        max_cartpole_steps: int = 200,
    ):
        super().__init__()
        self.model_path = model_path
        self.max_cartpole_steps = max_cartpole_steps
        self.registry = _build_registry()
        self.model, self.tokenizer = _load_model(model_path)
        self._current_system_prompt = (
            "你是一个机器人控制 Agent，根据任务选择合适的工具完成任务。\n"
            "输出格式：\nThought: <分析>\n<tool_call>tool_name(arg=val)</tool_call>"
        )

    # ── LLM 推理 ─────────────────────────────────────────────

    def _llm_generate(self, messages: list, max_new_tokens: int = 128) -> str:
        if self.model is None:
            return ""
        text = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.tokenizer(text, return_tensors="pt").to(self.model.device)
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=0.7,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id,
            )
        new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)

    def _llm_cartpole_policy(self, obs: list) -> str:
        """用 LLM 生成 CartPole 动作，失败时降级到规则策略"""
        if self.model is None:
            from agent_executor import rule_based_policy
            return rule_based_policy(obs)

        system_prompt = (
            "你是一个 CartPole 平衡控制 Agent。根据当前观测决定动作。\n"
            "观测格式：[cart_pos, cart_vel, pole_angle, pole_vel]\n"
            "输出格式（严格遵守）：\n"
            "Thought: <分析>\n"
            "<tool_call>cartpole_step(action=0或1)</tool_call>"
        )
        obs_str = f"[{', '.join(f'{v:.3f}' for v in obs)}]"
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"当前观测: {obs_str}，请决定动作。"},
        ]
        output = self._llm_generate(messages, max_new_tokens=128)
        # 如果 LLM 没有输出合法 tool_call，降级
        if "<tool_call>" not in output:
            from agent_executor import rule_based_policy
            return rule_based_policy(obs)
        return output

    # ── CartPole 任务 ─────────────────────────────────────────

    def _run_cartpole(self, params: dict) -> dict:
        executor = AgentExecutor(
            registry=self.registry,
            policy=self._llm_cartpole_policy,
            max_steps=params.get("max_steps", self.max_cartpole_steps),
        )
        return executor.run(task="balance cartpole")

    # ── 单次工具调用任务 ──────────────────────────────────────

    def _run_tool_task(self, task_type: str, params: dict) -> dict:
        """构造 tool_call 字符串并执行"""
        tool_map = {
            "pid":    ("simulate_pid",         ["kp", "ki", "kd"]),
            "rrt":    ("rrt_planning",          ["start_x", "start_y", "goal_x", "goal_y"]),
            "astar":  ("astar_planning",        ["start_x", "start_y", "goal_x", "goal_y"]),
            "ekf":    ("ekf_localization",      ["state", "control", "measurement"]),
            "arm_fk": ("arm_forward_kinematics",["joint_angles", "link_lengths"]),
        }
        if task_type not in tool_map:
            return {"error": f"未知任务类型: {task_type}"}

        tool_name, arg_keys = tool_map[task_type]
        args_str = ", ".join(
            f"{k}={params[k]}" for k in arg_keys if k in params
        )
        call_str = f"<tool_call>{tool_name}({args_str})</tool_call>"
        return self.registry.execute(call_str)

    # ── LitAgent 接口 ─────────────────────────────────────────

    def training_rollout(self, task: dict, resources=None, rollout=None):
        task_type = task.get("type", "cartpole")
        params = task.get("params", {})

        # 如果 APO 更新了 system_prompt，用新的提示词
        if resources and "system_prompt" in resources:
            self._current_system_prompt = str(resources["system_prompt"])

        if task_type == "cartpole":
            result = self._run_cartpole(params)
        else:
            result = self._run_tool_task(task_type, params)

        reward = tool_call_reward(result, task_type)
        emit_reward(reward)

        print(f"[LitRobotAgent] task={task_type} reward={reward:.3f}")
        # 返回 None，奖励已通过 emit_reward 发出

    def validation_rollout(self, task: dict, resources=None, rollout=None):
        return self.training_rollout(task, resources=resources, rollout=rollout)
