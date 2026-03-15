#!/usr/bin/env python3
"""
Robot Control Agent - Web UI
支持两种模式：
  - 对话模式：直接问答机器人控制领域知识
  - Agent 模式：ReAct 循环，自动调用工具完成任务

运行方式：
    cd /home/liujl/big_model/robot-llm-align
    conda run -n LLM python scripts/agent/app.py
    # 本地访问：ssh -L 7860:localhost:7860 user@server → http://localhost:7860
"""

import sys
import os
import glob
import argparse
import math
import random
import json
import base64
import re
import torch

sys.path.insert(0, __file__.rsplit("/", 1)[0])

from tool_registry import ToolRegistry
from agent_executor import AgentExecutor
from transformers import AutoModelForCausalLM, AutoTokenizer

# ============================================================
# 加载 DPO 模型（自动选最新 checkpoint）
# ============================================================

BASE_DIR   = "/home/liujl/big_model/robot-llm-align"
CKPT_DIR   = f"{BASE_DIR}/checkpoints"
PROMPT_PATH = f"{BASE_DIR}/dataset/best_prompt.txt"
FALLBACK_CKPT = f"{CKPT_DIR}/sft_qwen1.5b_with_tools"

def _find_latest_checkpoint() -> str:
    """找 dpo_pipeline_cycleN 中 N 最大且非空的，没有则用 fallback"""
    import re as _re
    pattern = f"{CKPT_DIR}/dpo_pipeline_cycle*"
    dirs = [d for d in glob.glob(pattern)
            if os.path.isdir(d) and len(os.listdir(d)) > 0]
    if not dirs:
        return FALLBACK_CKPT
    def _cycle_num(p):
        m = _re.search(r"cycle(\d+)", p)
        return int(m.group(1)) if m else 0
    return max(dirs, key=_cycle_num)

def _load_best_prompt() -> str:
    if os.path.exists(PROMPT_PATH):
        with open(PROMPT_PATH, "r", encoding="utf-8") as f:
            return f.read().strip()
    return ""

MODEL_PATH = _find_latest_checkpoint()
print(f"加载模型: {MODEL_PATH}")
_tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH, trust_remote_code=True)
_model = AutoModelForCausalLM.from_pretrained(
    MODEL_PATH, dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
)
_model.eval()
_best_prompt = _load_best_prompt()
print(f"模型加载完成  |  best_prompt: {'已加载' if _best_prompt else '未找到'}")


def llm_generate(messages: list, max_new_tokens: int = 512) -> str:
    """调用本地模型生成回复"""
    text = _tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
    with torch.no_grad():
        outputs = _model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=0.7,
            do_sample=True,
            pad_token_id=_tokenizer.eos_token_id,
        )
    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    return _tokenizer.decode(new_tokens, skip_special_tokens=True)


# ============================================================
# Mock CartPole 仿真
# ============================================================

class MockCartpole:
    def __init__(self):
        self.state = [0.0, 0.0, 0.0, 0.0]

    def reset(self) -> dict:
        self.state = [random.uniform(-0.05, 0.05) for _ in range(4)]
        return {"obs": list(self.state)}

    def step(self, action: int) -> dict:
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
        done = (abs(x) > 2.4 or abs(theta) > 0.2095)
        return {"obs": list(self.state), "reward": 1.0, "done": done}

    def ping(self) -> dict:
        return {"status": "ok", "mode": "mock"}


# ============================================================
# Registry
# ============================================================

def build_registry() -> ToolRegistry:
    sim = MockCartpole()
    reg = ToolRegistry()
    reg.register("cartpole_reset", sim.reset)
    reg.register("cartpole_step", sim.step)
    reg.register("cartpole_ping", sim.ping)
    from tools.python_robotics_tools import register_robotics_tools
    register_robotics_tools(reg)
    return reg


# ============================================================
# Agent 模式：ReAct 循环
# ============================================================

def render_cartpole_frame(state: list) -> str:
    """渲染 CartPole 当前帧，返回 base64 png"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from io import BytesIO
    import base64

    x, x_dot, theta, theta_dot = state
    cart_w, cart_h, pole_l = 0.4, 0.2, 1.0

    fig, ax = plt.subplots(figsize=(5, 3))
    ax.set_xlim(-3, 3); ax.set_ylim(-0.5, 2.0)
    ax.set_aspect("equal"); ax.axis("off")
    ax.axhline(0, color="gray", linewidth=1)

    # 小车
    cart = patches.FancyBboxPatch(
        (x - cart_w/2, -cart_h/2), cart_w, cart_h,
        boxstyle="round,pad=0.02", linewidth=1.5,
        edgecolor="black", facecolor="#4C72B0"
    )
    ax.add_patch(cart)

    # 摆杆
    px = x + pole_l * np.sin(theta)
    py = pole_l * np.cos(theta)
    ax.plot([x, px], [0, py], "-", color="#DD8452", linewidth=6, solid_capstyle="round")
    ax.plot(px, py, "o", color="#DD8452", markersize=10)

    ax.set_title(f"x={x:.2f}  θ={np.degrees(theta):.1f}°", fontsize=9)
    buf = BytesIO(); plt.savefig(buf, format="png", dpi=80, bbox_inches="tight"); buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode(); plt.close()
    return b64


    """CartPole 平衡任务，yield 流式 markdown"""
    from agent_executor import rule_based_policy
    _policy = policy or rule_based_policy

    reset_result = registry.execute("<tool_call>cartpole_reset()</tool_call>")
    obs = reset_result.get("obs", [0, 0, 0, 0])
    yield {"type": "text", "content": f"🔄 **Episode 开始** — 初始观测: `{[round(v, 3) for v in obs]}`\n\n---\n"}
    yield {"type": "image", "content": render_cartpole_frame(obs)}

    total_reward = 0.0
    for step in range(1, max_steps + 1):
        agent_output = _policy(obs)
        lines = agent_output.strip().split("\n")
        thought = lines[0] if lines else ""
        action_line = lines[1] if len(lines) > 1 else ""

        result = registry.execute(agent_output)
        obs = result.get("obs", obs)
        reward = result.get("reward", 0.0)
        done = result.get("done", False)
        total_reward += reward

        yield {"type": "text", "content": (
            f"**Step {step}**\n"
            f"💭 `{thought}`\n"
            f"🔧 `{action_line.strip()}`\n"
            f"📊 obs=`{[round(v,3) for v in obs]}` reward={reward} done={done}\n\n"
        )}
        yield {"type": "image", "content": render_cartpole_frame(obs)}
        if done:
            break

    yield {"type": "text", "content": f"\n---\n✅ **Episode 结束** — 存活 {step} 步，总奖励 {total_reward:.1f}"}


def run_tool_call(registry: ToolRegistry, user_msg: str):
    """解析用户直接输入的 tool_call 并执行，yield 结果"""
    import re
    # 支持 tool_name(args) 或 <tool_call>tool_name(args)</tool_call>
    raw = user_msg.strip()
    if not raw.startswith("<tool_call>"):
        raw = f"<tool_call>{raw}</tool_call>"

    yield {"type": "text", "content": f"🔧 执行: `{user_msg}`\n\n"}

    result = registry.execute(raw)

    # 如果有图表，单独 yield
    plot_b64 = result.pop("plot_base64", None)

    yield {"type": "text", "content": f"📊 结果:\n```json\n{json.dumps(result, ensure_ascii=False, indent=2)}\n```\n"}

    if plot_b64:
        yield {"type": "image", "content": plot_b64}


# ============================================================
# 对话模式：规则回复（占位，后续接 LLM）
# ============================================================

KNOWLEDGE_BASE = {
    "pid": "**PID 控制器** 由比例（P）、积分（I）、微分（D）三项组成：\n\n$$u(t) = K_p e(t) + K_i \\int e(t)dt + K_d \\frac{de(t)}{dt}$$\n\n- **P 项**：消除当前误差，增大 Kp 加快响应但易超调\n- **I 项**：消除稳态误差，增大 Ki 消除静差但易积分饱和\n- **D 项**：预测误差趋势，增大 Kd 抑制超调但对噪声敏感",
    "mpc": "**模型预测控制（MPC）** 在每个时刻求解有限时域最优控制问题：\n\n$$\\min_{u} \\sum_{k=0}^{N} \\|x_k - x_{ref}\\|_Q^2 + \\|u_k\\|_R^2$$\n\n优势：可处理约束、多变量、非线性系统，比 PID 更灵活。",
    "rl": "**强化学习** 通过智能体与环境交互学习最优策略：\n\n- **Q-learning**（off-policy）：$Q(s,a) \\leftarrow Q(s,a) + \\alpha[r + \\gamma \\max_{a'} Q(s',a') - Q(s,a)]$\n- **SARSA**（on-policy）：$Q(s,a) \\leftarrow Q(s,a) + \\alpha[r + \\gamma Q(s',a') - Q(s,a)]$\n\n区别：Q-learning 用贪心策略更新，SARSA 用实际执行的动作更新。",
    "rrt": "**RRT（快速随机树）** 是一种采样型路径规划算法：\n\n1. 随机采样状态空间中的点\n2. 找到树中最近节点\n3. 向采样点扩展一步\n4. 重复直到到达目标\n\n适合高维、复杂约束的规划问题。可用 `rrt_planning(sx, sy, gx, gy)` 直接调用。",
    "ekf": "**扩展卡尔曼滤波（EKF）** 对非线性系统进行状态估计：\n\n预测步：$\\hat{x}_{k|k-1} = f(\\hat{x}_{k-1}, u_k)$\n\n更新步：$\\hat{x}_k = \\hat{x}_{k|k-1} + K_k(z_k - h(\\hat{x}_{k|k-1}))$\n\n通过雅可比矩阵线性化非线性函数。",
}

TOOL_HELP = """**可用工具列表**（直接输入调用）：

| 工具 | 示例 |
|------|------|
| PID 仿真 | `simulate_pid(kp=1.0, ki=0.1, kd=0.05)` |
| RRT 规划 | `rrt_planning(start_x=0, start_y=0, goal_x=5, goal_y=5)` |
| A* 规划 | `astar_planning(start_x=0, start_y=0, goal_x=8, goal_y=8)` |
| 样条轨迹 | `cubic_spline_planning(waypoints="0,0;2,3;5,2;7,4")` |
| LQR 控制 | `lqr_steering_control(x=0, y=0, yaw=0, v=1.0, ref_path="0,0,0;5,0,0")` |
| EKF 定位 | `ekf_localization(state="0,0,0", control="1,0.1", measurement="0.1,0.1")` |
| 机械臂FK | `arm_forward_kinematics(joint_angles="0.5,0.5,0.5", link_lengths="1,1,1")` |
| 路径对比图 | `plot_path_comparison(paths="0,0;1,1;2,2|0,0;0,1;0,2", labels="RRT,AStar")` |
| CartPole | `cartpole_reset()` / `cartpole_step(action=1)` |
"""

def chat_reply(user_msg: str, history: list = None) -> str:
    """使用 DPO 模型生成回复"""
    if any(w in user_msg.lower() for w in ["工具", "tool", "help", "帮助", "能做什么"]):
        return TOOL_HELP

    system_prompt = (
        "你是一个机器人控制领域的 AI 助手，擅长控制理论（PID、MPC、LQR）、"
        "强化学习（Q-learning、PPO、SAC）、路径规划（RRT、A*）和状态估计（EKF、粒子滤波）。"
        "当需要调用工具时，使用 <tool_call>tool_name(arg=val)</tool_call> 格式。"
    )
    messages = [{"role": "system", "content": system_prompt}]

    # 加入历史（最近 3 轮）
    if history:
        for h in history[-3:]:
            if h.get("role") in ("user", "assistant") and h.get("content"):
                messages.append({"role": h["role"], "content": h["content"]})

    messages.append({"role": "user", "content": user_msg})
    return llm_generate(messages)


def llm_agent_policy(obs: list, history: list = None) -> str:
    """用 LLM 生成 ReAct 动作（含 <tool_call>）"""
    system_prompt = (
        "你是一个 CartPole 平衡控制 Agent。根据当前观测决定动作。\n"
        "观测格式：[cart_pos, cart_vel, pole_angle, pole_vel]\n"
        "输出格式：\nThought: <分析>\n<tool_call>cartpole_step(action=0或1)</tool_call>"
    )
    obs_str = f"[{', '.join(f'{v:.3f}' for v in obs)}]"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": f"当前观测: {obs_str}，请决定动作。"},
    ]
    return llm_generate(messages, max_new_tokens=128)


# ============================================================
# Gradio UI
# ============================================================

def build_ui(registry: ToolRegistry):
    import gradio as gr

    # ── 判断输入类型 ──────────────────────────────────────────
    def is_agent_trigger(msg: str) -> bool:
        return any(w in msg.lower() for w in ["开始", "run", "balance", "平衡", "cartpole"])

    def is_tool_call(msg: str) -> bool:
        import re
        return bool(re.match(r'\s*\w+\s*\(', msg.strip()))

    # ── Agent 模式：本地模型意图理解 + 规则工具路由 ──────────
    # 优先用 pipeline 优化过的 best_prompt，否则用默认
    AGENT_SYSTEM = _best_prompt if _best_prompt else (
        "你是专业的机器人控制 Agent。分析用户任务，判断需要调用哪个工具，提取参数。\n"
        "可用工具：simulate_pid, rrt_planning, astar_planning, cubic_spline_planning, "
        "ekf_localization, arm_forward_kinematics, cartpole_reset\n"
        "请用一句话描述你的分析，然后说明选择哪个工具和参数值。"
    )

    # 工具意图关键词映射
    TOOL_INTENT = [
        (["pid", "比例", "积分", "微分", "kp", "ki", "kd", "控制器调参"],
         "simulate_pid", {"kp": 1.0, "ki": 0.1, "kd": 0.05}),
        (["rrt", "随机树", "路径规划", "path"],
         "rrt_planning", {"start_x": 0, "start_y": 0, "goal_x": 5, "goal_y": 5}),
        (["astar", "a*", "a星", "网格规划"],
         "astar_planning", {"start_x": 0, "start_y": 0, "goal_x": 8, "goal_y": 8}),
        (["样条", "spline", "轨迹"],
         "cubic_spline_planning", {"waypoints": "0,0;2,3;5,2;7,4"}),
        (["ekf", "卡尔曼", "定位", "localization"],
         "ekf_localization", {"state": "0,0,0", "control": "1.0,0.1", "measurement": "0.1,0.05"}),
        (["机械臂", "arm", "正运动学", "fk", "关节"],
         "arm_forward_kinematics", {"joint_angles": "0.5,1.0,0.3", "link_lengths": "1,1,0.5"}),
        (["cartpole", "倒立摆", "平衡"],
         "cartpole_reset", {}),
    ]

    def extract_numbers(text: str, keys: list) -> dict:
        """从文本中提取数字参数，按 keys 顺序匹配"""
        nums = re.findall(r"[-+]?\d*\.?\d+", text)
        result = {}
        for i, k in enumerate(keys):
            if i < len(nums):
                result[k] = float(nums[i]) if "." in nums[i] else int(nums[i])
        return result

    def route_to_tool(user_msg: str) -> tuple:
        """根据用户消息路由到对应工具，返回 (tool_name, params_dict)"""
        msg_lower = user_msg.lower()
        for keywords, tool_name, default_params in TOOL_INTENT:
            if any(kw in msg_lower for kw in keywords):
                params = dict(default_params)
                # 尝试从消息中提取数字覆盖默认参数
                extracted = extract_numbers(user_msg, list(params.keys()))
                params.update(extracted)
                return tool_name, params
        return None, {}

    def build_tool_call_str(tool_name: str, params: dict) -> str:
        parts = []
        for k, v in params.items():
            if isinstance(v, str):
                parts.append(f'{k}="{v}"')
            else:
                parts.append(f"{k}={v}")
        return f"<tool_call>{tool_name}({', '.join(parts)})</tool_call>"

    def run_react_loop(user_msg: str, history: list):
        """本地模型分析意图 → 规则路由工具 → 执行 → 本地模型总结"""
        buf = ""

        # Step 1: 本地模型分析任务
        buf += "**▸ [分析任务]** 本地模型思考中...\n"
        history[-1]["content"] = buf
        yield history, history

        analysis = llm_generate(
            [{"role": "system", "content": AGENT_SYSTEM},
             {"role": "user", "content": user_msg}],
            max_new_tokens=200
        )
        buf += f"{analysis}\n\n"
        history[-1]["content"] = buf
        yield history, history

        # Step 2: 规则路由到工具
        tool_name, params = route_to_tool(user_msg)
        if not tool_name:
            # 没匹配到工具，纯对话回复
            buf += "**▸ [结论]** 未识别到具体工具任务，以上为分析结果。\n"
            history[-1]["content"] = buf
            yield history, history
            return

        call_str = build_tool_call_str(tool_name, params)
        buf += f"**▸ [工具调用]** `{tool_name}`\n```\n{call_str}\n```\n"
        history[-1]["content"] = buf
        yield history, history

        # Step 3: 执行工具
        result = registry.execute(call_str)
        plot_b64 = result.pop("plot_base64", None) if isinstance(result, dict) else None
        result_str = json.dumps(result, ensure_ascii=False, indent=2)[:800]

        buf += f"\n**▸ [执行结果]**\n```json\n{result_str}\n```\n"
        if plot_b64:
            buf += f"\n![{tool_name}](data:image/png;base64,{plot_b64})\n"
        history[-1]["content"] = buf
        yield history, history

        # Step 4: 本地模型总结结果
        buf += "\n**▸ [结果分析]** 本地模型总结中...\n"
        history[-1]["content"] = buf
        yield history, history

        summary = llm_generate(
            [{"role": "system", "content": "你是机器人控制专家，用中文简洁分析工具执行结果。"},
             {"role": "user", "content": f"任务：{user_msg}\n工具：{tool_name}\n结果：{result_str}\n请分析结果是否合理，给出结论。"}],
            max_new_tokens=256
        )
        buf = buf.rstrip("本地模型总结中...\n") + f"\n{summary}\n"
        history[-1]["content"] = buf
        yield history, history

    # ── 直接执行工具调用（快捷按钮 / 手动输入）────────────────
    def exec_tool_direct(user_msg: str, history: list):
        """直接执行 tool_call，显示结果 + 可视化图"""
        text_buf = ""
        for chunk in run_tool_call(registry, user_msg):
            if chunk["type"] == "text":
                text_buf += chunk["content"]
            elif chunk["type"] == "image":
                text_buf += f"\n\n![result](data:image/png;base64,{chunk['content']})\n"
            history[-1]["content"] = text_buf
            yield history, history

    def handle(user_msg: str, history: list, mode: str):
        history = history or []
        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": ""})

        if mode == "🤖 Agent 模式":
            yield from run_react_loop(user_msg, history)
        elif is_tool_call(user_msg):
            yield from exec_tool_direct(user_msg, history)
        else:
            reply = chat_reply(user_msg, history=history[:-2])
            history[-1]["content"] = reply
            yield history, history
            # 检测回复中的 tool_call 并执行
            tool_calls = re.findall(r"<tool_call>(.*?)</tool_call>", reply, re.DOTALL)
            if tool_calls:
                obs_buf = "\n\n**🔧 工具执行结果：**\n"
                for tc in tool_calls:
                    result = registry.execute(f"<tool_call>{tc}</tool_call>")
                    plot_b64 = result.pop("plot_base64", None) if isinstance(result, dict) else None
                    obs_buf += f"\n`{tc.split('(')[0]}`:\n```json\n{json.dumps(result, ensure_ascii=False, indent=2)[:500]}\n```\n"
                    if plot_b64:
                        obs_buf += f"\n![result](data:image/png;base64,{plot_b64})\n"
                history[-1]["content"] = reply + obs_buf
                yield history, history

    # ── UI 布局 ───────────────────────────────────────────────
    css = """
    @import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Rajdhani:wght@400;500;600;700&display=swap');

    /* ── Root Variables ── */
    :root {
        --bg-void:    #050810;
        --bg-deep:    #080d1a;
        --bg-panel:   #0c1220;
        --bg-card:    #101828;
        --bg-hover:   #162035;
        --cyan:       #00e5ff;
        --cyan-dim:   #00b8cc;
        --cyan-glow:  rgba(0,229,255,0.15);
        --amber:      #ffab00;
        --amber-dim:  #cc8800;
        --green:      #00e676;
        --red:        #ff1744;
        --border:     rgba(0,229,255,0.18);
        --border-dim: rgba(0,229,255,0.07);
        --text-hi:    #e8f4f8;
        --text-mid:   #7a9bb5;
        --text-lo:    #3a5570;
        --font-mono:  'JetBrains Mono', monospace;
        --font-ui:    'Rajdhani', sans-serif;
    }

    /* ── Global Reset ── */
    *, *::before, *::after { box-sizing: border-box; }

    body, .gradio-container {
        background: var(--bg-void) !important;
        font-family: var(--font-mono) !important;
        color: var(--text-hi) !important;
    }

    /* Scanline texture overlay */
    .gradio-container::before {
        content: '';
        position: fixed;
        inset: 0;
        background: repeating-linear-gradient(
            0deg,
            transparent,
            transparent 2px,
            rgba(0,0,0,0.08) 2px,
            rgba(0,0,0,0.08) 4px
        );
        pointer-events: none;
        z-index: 9999;
    }

    /* Grid dot background */
    .gradio-container::after {
        content: '';
        position: fixed;
        inset: 0;
        background-image: radial-gradient(rgba(0,229,255,0.06) 1px, transparent 1px);
        background-size: 32px 32px;
        pointer-events: none;
        z-index: 0;
    }

    /* ── Header ── */
    .rca-header {
        position: relative;
        padding: 20px 28px 16px;
        border-bottom: 1px solid var(--border);
        background: linear-gradient(135deg, var(--bg-deep) 0%, var(--bg-panel) 100%);
        overflow: hidden;
    }
    .rca-header::before {
        content: '';
        position: absolute;
        top: 0; left: 0; right: 0;
        height: 2px;
        background: linear-gradient(90deg, transparent, var(--cyan), var(--amber), transparent);
        animation: scanbar 4s linear infinite;
    }
    @keyframes scanbar {
        0%   { transform: translateX(-100%); }
        100% { transform: translateX(100%); }
    }
    .rca-header h1 {
        font-family: var(--font-ui) !important;
        font-size: 26px !important;
        font-weight: 700 !important;
        letter-spacing: 3px !important;
        text-transform: uppercase !important;
        color: var(--cyan) !important;
        text-shadow: 0 0 20px var(--cyan), 0 0 40px rgba(0,229,255,0.3) !important;
        margin: 0 0 4px !important;
    }
    .rca-header p {
        font-size: 11px !important;
        color: var(--text-mid) !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        margin: 0 !important;
    }
    .rca-status-dot {
        display: inline-block;
        width: 7px; height: 7px;
        border-radius: 50%;
        background: var(--green);
        box-shadow: 0 0 8px var(--green);
        animation: pulse-dot 2s ease-in-out infinite;
        margin-right: 6px;
        vertical-align: middle;
    }
    @keyframes pulse-dot {
        0%, 100% { opacity: 1; box-shadow: 0 0 8px var(--green); }
        50%       { opacity: 0.5; box-shadow: 0 0 3px var(--green); }
    }

    /* ── Tabs ── */
    .tabs { background: transparent !important; border: none !important; }
    .tab-nav {
        background: var(--bg-deep) !important;
        border-bottom: 1px solid var(--border) !important;
        padding: 0 16px !important;
        gap: 0 !important;
    }
    .tab-nav button {
        font-family: var(--font-ui) !important;
        font-size: 13px !important;
        font-weight: 600 !important;
        letter-spacing: 1.5px !important;
        text-transform: uppercase !important;
        color: var(--text-mid) !important;
        background: transparent !important;
        border: none !important;
        border-bottom: 2px solid transparent !important;
        padding: 12px 20px !important;
        border-radius: 0 !important;
        transition: all 0.2s !important;
    }
    .tab-nav button:hover {
        color: var(--cyan) !important;
        background: var(--cyan-glow) !important;
    }
    .tab-nav button.selected {
        color: var(--cyan) !important;
        border-bottom-color: var(--cyan) !important;
        text-shadow: 0 0 12px var(--cyan) !important;
    }

    /* ── Panels & Cards ── */
    .panel-box {
        background: var(--bg-panel);
        border: 1px solid var(--border-dim);
        border-radius: 4px;
        padding: 14px;
        position: relative;
    }
    .panel-box::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 3px; height: 100%;
        background: var(--cyan);
        border-radius: 4px 0 0 4px;
        opacity: 0.6;
    }
    .section-label {
        font-family: var(--font-ui) !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 2.5px !important;
        text-transform: uppercase !important;
        color: var(--cyan-dim) !important;
        margin: 16px 0 8px !important;
        padding-left: 8px !important;
        border-left: 2px solid var(--cyan-dim) !important;
    }

    /* ── Buttons ── */
    button, .gr-button {
        font-family: var(--font-mono) !important;
        border-radius: 3px !important;
        transition: all 0.15s ease !important;
        cursor: pointer !important;
    }

    /* Tool buttons */
    .tool-btn {
        font-size: 11px !important;
        font-weight: 500 !important;
        letter-spacing: 0.5px !important;
        padding: 7px 10px !important;
        width: 100% !important;
        text-align: left !important;
        background: var(--bg-card) !important;
        color: var(--text-mid) !important;
        border: 1px solid var(--border-dim) !important;
        margin-bottom: 4px !important;
        position: relative !important;
        overflow: hidden !important;
    }
    .tool-btn::after {
        content: '▶';
        position: absolute;
        right: 10px;
        top: 50%;
        transform: translateY(-50%);
        font-size: 8px;
        color: var(--text-lo);
        transition: all 0.15s;
    }
    .tool-btn:hover {
        background: var(--bg-hover) !important;
        color: var(--cyan) !important;
        border-color: var(--border) !important;
        box-shadow: inset 0 0 12px var(--cyan-glow), 0 0 8px var(--cyan-glow) !important;
    }
    .tool-btn:hover::after { color: var(--cyan); right: 8px; }

    /* Primary button (CartPole / Send) */
    .gr-button-primary, button.primary {
        background: transparent !important;
        border: 1px solid var(--cyan) !important;
        color: var(--cyan) !important;
        font-family: var(--font-ui) !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 12px var(--cyan-glow), inset 0 0 12px var(--cyan-glow) !important;
        text-shadow: 0 0 8px var(--cyan) !important;
    }
    .gr-button-primary:hover, button.primary:hover {
        background: var(--cyan-glow) !important;
        box-shadow: 0 0 24px rgba(0,229,255,0.4), inset 0 0 20px rgba(0,229,255,0.1) !important;
    }

    /* Search button */
    .search-btn {
        background: transparent !important;
        border: 1px solid var(--amber) !important;
        color: var(--amber) !important;
        font-family: var(--font-ui) !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 10px rgba(255,171,0,0.2) !important;
    }
    .search-btn:hover {
        background: rgba(255,171,0,0.08) !important;
        box-shadow: 0 0 20px rgba(255,171,0,0.35) !important;
    }

    /* Refresh button */
    .refresh-btn {
        background: transparent !important;
        border: 1px solid var(--green) !important;
        color: var(--green) !important;
        font-family: var(--font-ui) !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        box-shadow: 0 0 10px rgba(0,230,118,0.2) !important;
    }
    .refresh-btn:hover {
        background: rgba(0,230,118,0.08) !important;
        box-shadow: 0 0 20px rgba(0,230,118,0.35) !important;
    }

    /* ── Chatbot ── */
    .chatbot-wrap .wrap {
        background: var(--bg-deep) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 4px !important;
    }
    .message.user {
        background: var(--bg-hover) !important;
        border: 1px solid var(--border) !important;
        border-radius: 3px !important;
        color: var(--text-hi) !important;
        font-size: 13px !important;
    }
    .message.bot {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-dim) !important;
        border-left: 2px solid var(--cyan) !important;
        border-radius: 3px !important;
        color: var(--text-hi) !important;
        font-size: 13px !important;
    }
    .message.bot code {
        background: rgba(0,229,255,0.07) !important;
        color: var(--cyan) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 2px !important;
        font-family: var(--font-mono) !important;
        font-size: 12px !important;
    }

    /* ── Textbox / Input ── */
    input[type="text"], textarea, .gr-textbox textarea, .gr-textbox input {
        background: var(--bg-card) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 3px !important;
        color: var(--text-hi) !important;
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
        caret-color: var(--cyan) !important;
        transition: border-color 0.2s, box-shadow 0.2s !important;
    }
    input[type="text"]:focus, textarea:focus,
    .gr-textbox textarea:focus, .gr-textbox input:focus {
        border-color: var(--cyan) !important;
        box-shadow: 0 0 0 2px var(--cyan-glow) !important;
        outline: none !important;
    }

    /* ── Radio ── */
    .gr-radio label {
        font-family: var(--font-mono) !important;
        font-size: 12px !important;
        color: var(--text-mid) !important;
    }
    .gr-radio input[type="radio"]:checked + span {
        color: var(--cyan) !important;
    }

    /* ── Slider ── */
    input[type="range"] {
        accent-color: var(--amber) !important;
    }

    /* ── Markdown output ── */
    .gr-markdown, .prose {
        font-family: var(--font-mono) !important;
        font-size: 13px !important;
        color: var(--text-hi) !important;
        line-height: 1.7 !important;
    }
    .gr-markdown h1, .gr-markdown h2, .gr-markdown h3 {
        font-family: var(--font-ui) !important;
        color: var(--cyan) !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
    }
    .gr-markdown a { color: var(--amber) !important; }
    .gr-markdown code {
        background: rgba(0,229,255,0.07) !important;
        color: var(--cyan) !important;
        border: 1px solid var(--border-dim) !important;
        border-radius: 2px !important;
        padding: 1px 5px !important;
    }
    .gr-markdown pre {
        background: var(--bg-deep) !important;
        border: 1px solid var(--border-dim) !important;
        border-left: 3px solid var(--cyan) !important;
        border-radius: 3px !important;
        padding: 12px !important;
        overflow-x: auto !important;
    }
    .gr-markdown table {
        border-collapse: collapse !important;
        width: 100% !important;
        font-size: 12px !important;
    }
    .gr-markdown th {
        background: var(--bg-hover) !important;
        color: var(--cyan) !important;
        font-family: var(--font-ui) !important;
        letter-spacing: 1px !important;
        text-transform: uppercase !important;
        padding: 8px 12px !important;
        border: 1px solid var(--border-dim) !important;
    }
    .gr-markdown td {
        padding: 6px 12px !important;
        border: 1px solid var(--border-dim) !important;
        color: var(--text-mid) !important;
    }
    .gr-markdown tr:hover td { background: var(--bg-hover) !important; color: var(--text-hi) !important; }

    /* ── Labels ── */
    label, .gr-form > label, .block > label {
        font-family: var(--font-ui) !important;
        font-size: 10px !important;
        font-weight: 700 !important;
        letter-spacing: 2px !important;
        text-transform: uppercase !important;
        color: var(--text-lo) !important;
    }

    /* ── Scrollbars ── */
    ::-webkit-scrollbar { width: 5px; height: 5px; }
    ::-webkit-scrollbar-track { background: var(--bg-deep); }
    ::-webkit-scrollbar-thumb { background: var(--border); border-radius: 3px; }
    ::-webkit-scrollbar-thumb:hover { background: var(--cyan-dim); }

    /* ── Divider ── */
    hr {
        border: none !important;
        border-top: 1px solid var(--border-dim) !important;
        margin: 12px 0 !important;
    }

    /* ── Footer ── */
    footer { display: none !important; }

    /* ── Misc ── */
    .gap { gap: 6px !important; }
    .contain { padding: 12px !important; }
    """

    with gr.Blocks(title="RCA // Robot Control Agent") as demo:

        gr.HTML("""
        <div class="rca-header">
            <h1><span class="rca-status-dot"></span>Robot Control Agent</h1>
            <p>PythonRobotics · ReAct Agent · DPO-Aligned · Neural Terminal v2</p>
        </div>
        """)

        with gr.Tabs():

            # ── Tab 1: Agent 对话 ─────────────────────────────
            with gr.Tab("◈  AGENT  CONSOLE"):
                with gr.Row():
                    # 左侧面板
                    with gr.Column(scale=1, min_width=220):
                        mode_radio = gr.Radio(
                            choices=["💬 对话模式", "🤖 Agent 模式"],
                            value="💬 对话模式",
                            label="运行模式",
                        )

                        gr.HTML('<div class="section-label">// SIMULATION TOOLS</div>')
                        with gr.Column():
                            btn_pid    = gr.Button("▸ PID  CONTROLLER",  elem_classes="tool-btn")
                            btn_rrt    = gr.Button("▸ RRT  PLANNER",      elem_classes="tool-btn")
                            btn_astar  = gr.Button("▸ A*   PLANNER",      elem_classes="tool-btn")
                            btn_spline = gr.Button("▸ CUBIC SPLINE",      elem_classes="tool-btn")
                            btn_arm    = gr.Button("▸ ARM  KINEMATICS",   elem_classes="tool-btn")
                            btn_ekf    = gr.Button("▸ EKF  LOCALIZATION", elem_classes="tool-btn")
                            btn_agent  = gr.Button("▶  RUN CARTPOLE", variant="primary", elem_classes="tool-btn")

                        gr.HTML('<div class="section-label">// KNOWLEDGE BASE</div>')
                        with gr.Column():
                            btn_pid_qa = gr.Button("? PID CONTROL",       elem_classes="tool-btn")
                            btn_mpc_qa = gr.Button("? MPC THEORY",        elem_classes="tool-btn")
                            btn_rl_qa  = gr.Button("? Q-LEARN vs SARSA",  elem_classes="tool-btn")
                            btn_rrt_qa = gr.Button("? RRT ALGORITHM",    elem_classes="tool-btn")

                    # 右侧对话区
                    with gr.Column(scale=4):
                        chatbot = gr.Chatbot(
                            label="",
                            height=560,
                            buttons=["copy", "copy_all"],
                            placeholder="// READY — 输入指令或点击左侧工具快捷键\n// TYPE A COMMAND OR SELECT A TOOL FROM THE SIDEBAR",
                            layout="bubble",
                            elem_classes="chatbot-wrap",
                        )
                        state = gr.State([])

                        with gr.Row():
                            msg_box = gr.Textbox(
                                placeholder="> ENTER COMMAND OR QUERY...",
                                show_label=False,
                                lines=1,
                                scale=6,
                            )
                            send_btn  = gr.Button("SEND ↵", variant="primary", scale=1)
                            clear_btn = gr.Button("CLR", scale=0, min_width=52)

                # 事件绑定
                send_btn.click(handle, inputs=[msg_box, state, mode_radio], outputs=[chatbot, state])
                msg_box.submit(handle, inputs=[msg_box, state, mode_radio], outputs=[chatbot, state])
                clear_btn.click(lambda: ([], []), outputs=[chatbot, state])

                TOOL_PRESETS = {
                    btn_pid:    "simulate_pid(kp=1.0, ki=0.1, kd=0.05)",
                    btn_rrt:    "rrt_planning(start_x=0, start_y=0, goal_x=5, goal_y=5, obstacle_list=2,2,0.5;3,3,0.5)",
                    btn_astar:  "astar_planning(start_x=0, start_y=0, goal_x=8, goal_y=8)",
                    btn_spline: "cubic_spline_planning(waypoints=0,0;2,3;5,2;7,4)",
                    btn_arm:    "arm_forward_kinematics(joint_angles=0.5,0.5,0.5, link_lengths=1,1,1)",
                    btn_ekf:    "ekf_localization(state=0,0,0, control=1,0.1, measurement=0.1,0.1)",
                }
                for btn, preset in TOOL_PRESETS.items():
                    btn.click(lambda p=preset: p, outputs=[msg_box]).then(
                        handle, inputs=[msg_box, state, mode_radio], outputs=[chatbot, state]
                    )

                btn_agent.click(
                    lambda: ("🤖 Agent 模式", "开始"), outputs=[mode_radio, msg_box]
                ).then(handle, inputs=[msg_box, state, mode_radio], outputs=[chatbot, state])

                QA_PRESETS = {
                    btn_pid_qa: "什么是 PID 控制？",
                    btn_mpc_qa: "什么是 MPC？",
                    btn_rl_qa:  "Q-learning 和 SARSA 的区别？",
                    btn_rrt_qa: "RRT 算法原理？",
                }
                for btn, preset in QA_PRESETS.items():
                    btn.click(lambda p=preset: p, outputs=[msg_box]).then(
                        handle, inputs=[msg_box, state, mode_radio], outputs=[chatbot, state]
                    )

            # ── Tab 2: 论文搜索 ───────────────────────────────
            with gr.Tab("◈  DEEP  SEARCH"):
                gr.HTML('<div class="section-label">// ARXIV NEURAL SEARCH + AI SYNTHESIS</div>')
                with gr.Row():
                    search_box = gr.Textbox(
                        placeholder="> QUERY: robot learning, DPO, path planning, SLAM...",
                        show_label=False,
                        scale=5,
                    )
                    search_btn = gr.Button("SEARCH", elem_classes="search-btn", scale=1)
                    max_results = gr.Slider(3, 20, value=8, step=1, label="MAX RESULTS", scale=1)
                search_out = gr.Markdown(value="")
                search_btn.click(
                    fn=search_and_summarize,
                    inputs=[search_box, max_results],
                    outputs=search_out,
                )
                search_box.submit(
                    fn=search_and_summarize,
                    inputs=[search_box, max_results],
                    outputs=search_out,
                )

            # ── Tab 3: Pipeline 监控 ──────────────────────────
            with gr.Tab("◈  PIPELINE  MONITOR"):
                gr.HTML('<div class="section-label">// TRAINING PIPELINE STATUS · DPO CYCLES · CHECKPOINT TRACKER</div>')
                refresh_btn = gr.Button("↺  REFRESH STATUS", elem_classes="refresh-btn")
                status_md = gr.Markdown(value=get_pipeline_status())
                refresh_btn.click(fn=get_pipeline_status, outputs=status_md)

    return demo, css


# ============================================================
# 论文搜索后端
# ============================================================

SILICONFLOW_API_KEY = os.environ.get("SILICONFLOW_API_KEY", "")  # set via env: export SILICONFLOW_API_KEY=...

def search_and_summarize(query: str, max_results: int = 8):
    """arxiv 搜索 + 大模型总结，yield 流式文本"""
    import arxiv

    yield "🔍 正在搜索 arxiv...\n\n"

    try:
        client = arxiv.Client()
        search = arxiv.Search(
            query=query,
            max_results=max_results,
            sort_by=arxiv.SortCriterion.Relevance,
        )
        results = list(client.results(search))
    except Exception as e:
        yield f"搜索失败: {e}"
        return

    if not results:
        yield "未找到相关论文，请换个关键词。"
        return

    # 展示论文列表
    papers_md = f"### 找到 {len(results)} 篇相关论文\n\n"
    papers_text = ""
    for i, r in enumerate(results, 1):
        authors = ", ".join(a.name for a in r.authors[:3])
        if len(r.authors) > 3:
            authors += " et al."
        papers_md += (
            f"**{i}. [{r.title}]({r.entry_id})**\n"
            f"*{authors} · {r.published.strftime('%Y-%m')}*\n\n"
        )
        papers_text += f"[{i}] {r.title}\n摘要: {r.summary[:400]}\n\n"

    yield papers_md + "\n---\n\n🤖 正在总结...\n\n"

    # 大模型总结
    from openai import OpenAI
    llm = OpenAI(api_key=SILICONFLOW_API_KEY, base_url="https://api.siliconflow.cn/v1")
    prompt = (
        f"用户查询：{query}\n\n"
        f"以下是 arxiv 上的相关论文：\n\n{papers_text}\n\n"
        "请用中文：\n"
        "1. 概括这些论文的主要研究方向和核心贡献（3-5句）\n"
        "2. 列出最值得精读的 3 篇，说明理由\n"
        "3. 指出该领域当前的主要挑战和未来方向"
    )
    try:
        stream = llm.chat.completions.create(
            model="Pro/deepseek-ai/DeepSeek-V3.2",
            messages=[
                {"role": "system", "content": "你是一个机器人控制和AI领域的论文分析专家。"},
                {"role": "user", "content": prompt},
            ],
            max_tokens=1024, temperature=0.5, stream=True, timeout=60,
        )
        summary = "### AI 总结\n\n"
        for chunk in stream:
            delta = chunk.choices[0].delta.content or ""
            summary += delta
            yield papers_md + "\n---\n\n" + summary
    except Exception as e:
        yield papers_md + f"\n---\n\n总结失败: {e}"

PIPELINE_LOG_DIR     = "/home/liujl/big_model/robot-llm-align/logs"
PIPELINE_DATASET_DIR = "/home/liujl/big_model/robot-llm-align/dataset"

def get_pipeline_status() -> str:
    """读取最新 pipeline 日志，返回 markdown 格式状态"""

    # 当前 app 加载的 checkpoint
    active_ckpt = os.path.basename(MODEL_PATH)
    latest_ckpt = os.path.basename(_find_latest_checkpoint())
    ckpt_status = "✅ 已是最新" if active_ckpt == latest_ckpt else f"⚠️ 最新为 `{latest_ckpt}`，需重启 app 加载"
    header_md = (
        f"### 当前运行状态\n"
        f"- **加载的 Checkpoint**: `{active_ckpt}` — {ckpt_status}\n"
        f"- **System Prompt**: {'已加载 best_prompt.txt' if _best_prompt else '使用默认 prompt'}\n\n"
    )

    # 找最新日志
    logs = sorted(glob.glob(f"{PIPELINE_LOG_DIR}/pipeline_2*.log"), reverse=True)
    if not logs:
        return header_md + "未找到 pipeline 日志，请先启动 pipeline.py"

    log_path = logs[0]
    log_name = os.path.basename(log_path)

    # 读最后 40 行
    with open(log_path, "r", encoding="utf-8") as f:
        lines = f.readlines()
    tail = "".join(lines[-40:])

    # 读 summary
    summary_path = f"{PIPELINE_DATASET_DIR}/pipeline_summary.json"
    summary_md = ""
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        rows = []
        for r in summary:
            instr = f"{r.get('instruction_acc', 0)*100:.0f}%" if 'instruction_acc' in r else "—"
            halluc = f"{r.get('hallucination_rate', 0)*100:.0f}%" if 'hallucination_rate' in r else "—"
            rows.append(
                f"| {r['cycle']} | {r['tasks']} | {r['dpo_pairs']} "
                f"| {r['avg_reward']:.3f} | {instr} | {halluc} "
                f"| {os.path.basename(r['checkpoint'])} |"
            )
        summary_md = (
            "### 历史轮次\n"
            "| 轮次 | 任务数 | DPO对 | 平均奖励 | 指令准确率 | 幻觉率 | Checkpoint |\n"
            "|------|--------|-------|----------|------------|--------|------------|\n"
            + "\n".join(rows)
        )

    # 读最优 prompt
    prompt_path = f"{PIPELINE_DATASET_DIR}/best_prompt.txt"
    prompt_md = ""
    if os.path.exists(prompt_path):
        with open(prompt_path, "r", encoding="utf-8") as f:
            prompt = f.read().strip()
        prompt_md = f"### 当前最优 System Prompt\n```\n{prompt}\n```"

    return (
        header_md
        + f"**日志文件**: `{log_name}`\n\n"
        + f"### 最新日志（后40行）\n```\n{tail}\n```\n\n"
        + f"{summary_md}\n\n"
        + f"{prompt_md}"
    )



# ============================================================
# 入口
# ============================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--share", action="store_true", help="生成公网链接")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()

    registry = build_registry()
    demo, css = build_ui(registry)
    demo.launch(server_name="0.0.0.0", server_port=args.port, share=args.share, css=css)


if __name__ == "__main__":
    main()
