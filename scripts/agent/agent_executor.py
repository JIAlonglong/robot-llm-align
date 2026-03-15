"""
AgentExecutor — ReAct 循环
Thought → Action(<tool_call>) → Observation → Thought → ...

LLM 层现在是 rule-based 占位，接口固定：
    policy(obs) -> action_str (含 <tool_call> 标签)
后续直接替换成 model.generate() 即可。
"""
import math
from tool_registry import ToolRegistry


# ══════════════════════════════════════════════════════════
# Rule-based 占位 Policy（后续换成 LLM）
# 策略：pole 向右倾 → 向右推；向左倾 → 向左推
# ══════════════════════════════════════════════════════════
def rule_based_policy(obs: list) -> str:
    _, _, theta, theta_dot = obs
    # 简单能量平衡策略
    action = 1 if (theta + 0.1 * theta_dot) > 0 else 0
    thought = (
        f"Thought: pole_angle={theta:.3f} rad, pole_vel={theta_dot:.3f} rad/s. "
        f"{'向右倾，右推' if action == 1 else '向左倾，左推'}。"
    )
    tool_call = f"<tool_call>cartpole_step(action={action})</tool_call>"
    return thought + "\n" + tool_call


# ══════════════════════════════════════════════════════════
# AgentExecutor
# ══════════════════════════════════════════════════════════
class AgentExecutor:
    def __init__(self, registry: ToolRegistry, policy=None, max_steps: int = 500):
        self.registry  = registry
        self.policy    = policy or rule_based_policy
        self.max_steps = max_steps

    def run(self, task: str = "balance cartpole") -> dict:
        print(f"\n{'='*60}")
        print(f"Task: {task}")
        print(f"{'='*60}")

        # ── Step 0: reset ──────────────────────────────────
        reset_result = self.registry.execute(
            "<tool_call>cartpole_reset()</tool_call>"
        )
        obs = reset_result.get("obs", [0, 0, 0, 0])
        print(f"[Reset] obs={obs}")

        total_reward = 0.0
        step = 0

        # ── ReAct 循环 ─────────────────────────────────────
        while step < self.max_steps:
            # Thought + Action
            agent_output = self.policy(obs)
            print(f"\n[Step {step+1}]")
            print(agent_output)

            # Execute tool
            result = self.registry.execute(agent_output)
            obs    = result.get("obs", obs)
            reward = result.get("reward", 0.0)
            done   = result.get("done", False)
            total_reward += reward

            print(f"Observation: obs={[round(v,3) for v in obs]}, "
                  f"reward={reward}, done={done}")

            step += 1
            if done:
                break

        summary = {
            "steps_survived": step,
            "total_reward":   total_reward,
            "final_obs":      obs,
        }
        print(f"\n{'='*60}")
        print(f"Episode 结束: 存活 {step} 步, 总奖励 {total_reward:.1f}")
        print(f"{'='*60}")
        return summary
