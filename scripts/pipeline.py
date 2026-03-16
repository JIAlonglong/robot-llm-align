"""
pipeline.py — 4 小时一轮的完整训练流水线

每轮流程：
  阶段1（~2.5h）收集：
    - 提问模型（DeepSeek-V3.2）持续生成多样化任务
    - Agent 执行任务，记录 (prompt, response, reward)
    - 高奖励轨迹 → chosen，低奖励轨迹 → rejected，构建 DPO 数据对
    - 同时用优化模型更新 system prompt

  阶段2（~1h）训练：
    - 用本轮收集的 DPO 数据微调 Qwen（LoRA）
    - 保存新 checkpoint

  阶段3（~0.5h）评估：
    - 用新模型跑验证集，对比奖励提升

运行：
    nohup /home/liujl/miniconda3/envs/LLM/bin/python scripts/pipeline.py > logs/pipeline.log 2>&1 &

参数：
    --cycles N        运行 N 轮（默认无限）
    --collect-minutes 收集阶段时长（默认 150 分钟）
    --min-dpo-pairs   最少 DPO 数据对才触发训练（默认 30）
"""

import sys, os, json, time, argparse, random, math, subprocess
from datetime import datetime, timedelta
from typing import Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from openai import OpenAI

# ── 配置 ──────────────────────────────────────────────────────
SILICONFLOW_API_KEY  = os.environ.get("SILICONFLOW_API_KEY", "")
SILICONFLOW_BASE_URL = "https://api.siliconflow.cn/v1"
TASK_GEN_MODEL       = "Pro/deepseek-ai/DeepSeek-V3.2"
OPTIMIZER_MODEL      = "Pro/deepseek-ai/DeepSeek-V3.2"

BASE_DIR    = "/home/liujl/big_model/robot-llm-align"
DATASET_DIR = f"{BASE_DIR}/dataset"
CKPT_DIR    = f"{BASE_DIR}/checkpoints"
LOG_DIR     = f"{BASE_DIR}/logs"
PYTHON      = "/home/liujl/miniconda3/envs/LLM/bin/python"

# DPO 奖励阈值：高于此为 chosen，低于此为 rejected
CHOSEN_THRESHOLD   = 0.6
REJECTED_THRESHOLD = 0.3

# ── 工具说明 ──────────────────────────────────────────────────
TOOLS_DESCRIPTION = """
可用工具（机器人控制领域）：
1. simulate_pid(kp, ki, kd, setpoint=1.0)  — PID仿真，返回超调/调节时间/稳态误差
2. rrt_planning(start_x, start_y, goal_x, goal_y)  — RRT路径规划
3. astar_planning(start_x, start_y, goal_x, goal_y)  — A*路径规划
4. cubic_spline_planning(waypoints)  — 三次样条轨迹，waypoints="x1,y1;x2,y2;..."
5. lqr_steering_control(x, y, yaw, v, ref_path)  — LQR转向控制
6. ekf_localization(state, control, measurement)  — EKF定位
7. arm_forward_kinematics(joint_angles, link_lengths)  — 机械臂正运动学
8. cartpole_reset() / cartpole_step(action=0或1)  — CartPole倒立摆仿真
"""

TASK_GEN_SYSTEM = f"""你是机器人控制任务生成器，生成多样化任务供Agent完成。
{TOOLS_DESCRIPTION}
输出严格JSON（无其他内容）：
{{"type":"pid|rrt|astar|ekf|arm_fk|cartpole","description":"任务描述","params":{{参数}}}}
params示例：pid:{{"kp":1.5,"ki":0.1,"kd":0.05}} rrt:{{"start_x":0,"start_y":0,"goal_x":7,"goal_y":5}}
astar:{{"start_x":0,"start_y":0,"goal_x":8,"goal_y":8}} ekf:{{"state":"0,0,0","control":"1.0,0.1","measurement":"0.1,0.05"}}
arm_fk:{{"joint_angles":"0.5,1.0,0.3","link_lengths":"1,1,0.5"}} cartpole:{{"max_steps":200}}"""

OPTIMIZER_SYSTEM = """你是AI Agent系统优化专家，分析执行记录改进system prompt。
输出格式：
## 问题分析
<2-3个主要问题>
## 优化后的System Prompt
```
<完整新prompt>
```"""

INITIAL_SYSTEM_PROMPT = """你是专业的机器人控制Agent。根据任务描述选择合适工具完成任务。

工具调用格式（严格遵守）：
Thought: <分析任务，选择工具和参数>
<tool_call>tool_name(arg1=val1, arg2=val2)</tool_call>

规则：每次只调用一个工具，参数必须是数值或字符串字面量，完成后输出 Final Answer: <结论>"""


# ── 工具函数 ──────────────────────────────────────────────────

def make_client():
    return OpenAI(api_key=SILICONFLOW_API_KEY, base_url=SILICONFLOW_BASE_URL)


def log(msg: str):
    ts = datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}", flush=True)


def generate_task(client, cycle_idx: int) -> dict:
    categories = [
        ("pid",     "PID控制器调参"),
        ("rrt",     "RRT路径规划"),
        ("astar",   "A*路径规划"),
        ("ekf",     "EKF定位"),
        ("arm_fk",  "机械臂正运动学"),
        ("cartpole","CartPole平衡"),
    ]
    cat, cat_name = categories[cycle_idx % len(categories)]
    try:
        resp = client.chat.completions.create(
            model=TASK_GEN_MODEL,
            messages=[
                {"role": "system", "content": TASK_GEN_SYSTEM},
                {"role": "user",   "content": f"生成一个【{cat_name}】任务，参数有变化，第{cycle_idx}轮。"},
            ],
            temperature=0.9, max_tokens=256, timeout=30,
        )
        raw = resp.choices[0].message.content.strip()
        if "```" in raw:
            raw = raw.split("```")[1].lstrip("json").strip()
        task = json.loads(raw)
        if "type" not in task or "params" not in task:
            raise ValueError("缺少字段")
        return task
    except Exception as e:
        fallbacks = [
            {"type": "pid",    "params": {"kp": round(random.uniform(0.5,3.0),2), "ki": round(random.uniform(0,0.5),2), "kd": round(random.uniform(0,0.2),2)}},
            {"type": "rrt",    "params": {"start_x": 0, "start_y": 0, "goal_x": random.randint(3,9), "goal_y": random.randint(3,9)}},
            {"type": "cartpole","params": {"max_steps": 200}},
            {"type": "ekf",    "params": {"state": "0,0,0", "control": "1.0,0.1", "measurement": "0.1,0.05"}},
            {"type": "arm_fk", "params": {"joint_angles": f"{random.uniform(0,1.5):.2f},{random.uniform(0,1.5):.2f},{random.uniform(0,1.5):.2f}", "link_lengths": "1,1,0.5"}},
        ]
        log(f"  TaskGen fallback ({e})")
        return random.choice(fallbacks)


def execute_task(task: dict) -> dict:
    import contextlib, io
    from tool_registry import ToolRegistry
    from tools.python_robotics_tools import register_robotics_tools
    from reward import tool_call_reward
    _sink = open(os.devnull, "w")

    reg = ToolRegistry()

    class _CP:
        def __init__(self): self.state = [0.0]*4
        def reset(self):
            self.state = [random.uniform(-0.05,0.05) for _ in range(4)]
            return {"obs": list(self.state)}
        def step(self, action: int):
            x,xd,th,thd = self.state
            f = 10.0 if action==1 else -10.0
            g,mc,mp,l = 9.8,1.0,0.1,0.5
            tm = mc+mp; dt = 0.02
            ct,st = math.cos(th),math.sin(th)
            tmp = (f+mp*l*thd**2*st)/tm
            tha = (g*st-ct*tmp)/(l*(4/3-mp*ct**2/tm))
            xa  = tmp-mp*l*tha*ct/tm
            x+=dt*xd; xd+=dt*xa; th+=dt*thd; thd+=dt*tha
            self.state=[x,xd,th,thd]
            done = abs(x)>2.4 or abs(th)>0.2095
            return {"obs":list(self.state),"reward":1.0,"done":done}

    sim = _CP()
    with contextlib.redirect_stdout(_sink):
        reg.register("cartpole_reset", sim.reset)
        reg.register("cartpole_step",  sim.step)
        register_robotics_tools(reg)

    task_type = task.get("type","pid")
    params    = task.get("params",{})
    tool_map  = {
        "pid":    ("simulate_pid",           ["kp","ki","kd"]),
        "rrt":    ("rrt_planning",            ["start_x","start_y","goal_x","goal_y"]),
        "astar":  ("astar_planning",          ["start_x","start_y","goal_x","goal_y"]),
        "ekf":    ("ekf_localization",        ["state","control","measurement"]),
        "arm_fk": ("arm_forward_kinematics",  ["joint_angles","link_lengths"]),
    }

    if task_type == "cartpole":
        from agent_executor import AgentExecutor, rule_based_policy
        ex = AgentExecutor(registry=reg, policy=rule_based_policy, max_steps=params.get("max_steps",200))
        with contextlib.redirect_stdout(_sink):
            result = ex.run(task="balance cartpole")
    elif task_type in tool_map:
        tname, akeys = tool_map[task_type]
        args = ", ".join(f"{k}={params[k]}" for k in akeys if k in params)
        with contextlib.redirect_stdout(_sink):
            result = reg.execute(f"<tool_call>{tname}({args})</tool_call>")
    else:
        result = {"error": f"未知类型: {task_type}"}

    reward = tool_call_reward(result, task_type)

    # 路径规划任务：用起终点坐标计算更精确的奖励
    if task_type in ("rrt", "astar") and result.get("success"):
        from reward import path_planning_reward_with_coords
        reward = path_planning_reward_with_coords(
            success=True,
            path_length=result.get("length", 9999),
            start_x=params.get("start_x", 0),
            start_y=params.get("start_y", 0),
            goal_x=params.get("goal_x", 10),
            goal_y=params.get("goal_y", 10),
        )

    if isinstance(result, dict): result.pop("plot_base64", None)
    _sink.close()
    return {"result": result, "reward": reward, "task_type": task_type, "success": reward > 0.5}


def optimize_prompt(client, current_prompt: str, records: list, cycle: int) -> str:
    total   = len(records)
    avg_r   = sum(r["reward"] for r in records) / max(total,1)
    succ    = sum(1 for r in records if r["success"])
    fails   = [r for r in records if not r["success"]][:5]
    fail_s  = "\n".join(f"- {r['task_type']} reward={r['reward']:.2f} result={str(r['result'])[:80]}" for r in fails) or "无"
    msg = f"""第{cycle}轮 | 任务{total}个 | 成功{succ}({succ/max(total,1)*100:.0f}%) | 平均奖励{avg_r:.3f}

当前prompt：
```
{current_prompt}
```
失败案例：
{fail_s}

请优化prompt。"""
    try:
        resp = client.chat.completions.create(
            model=OPTIMIZER_MODEL,
            messages=[{"role":"system","content":OPTIMIZER_SYSTEM},{"role":"user","content":msg}],
            temperature=0.7, max_tokens=1024, timeout=60,
        )
        out = resp.choices[0].message.content.strip()
        if "```" in out:
            parts = out.split("```")
            for i,p in enumerate(parts):
                if i%2==1 and len(p.strip())>50:
                    return p.strip().lstrip("prompt").strip()
    except Exception as e:
        log(f"  Optimizer error: {e}")
    return current_prompt


def build_dpo_pairs(records: list) -> list:
    """从执行记录中构建 DPO 数据对"""
    chosen_pool   = [r for r in records if r["reward"] >= CHOSEN_THRESHOLD]
    rejected_pool = [r for r in records if r["reward"] <= REJECTED_THRESHOLD]

    pairs = []
    for c in chosen_pool:
        # 找同类型的 rejected
        same_type_rej = [r for r in rejected_pool if r["task_type"] == c["task_type"]]
        rej = random.choice(same_type_rej) if same_type_rej else (random.choice(rejected_pool) if rejected_pool else None)
        if rej is None:
            continue
        task_desc = c.get("task", {}).get("description", f"{c['task_type']}任务")
        pairs.append({
            "prompt":   f"任务：{task_desc}\n",
            "chosen":   f"执行结果：{json.dumps(c['result'], ensure_ascii=False)}\n奖励：{c['reward']:.3f}",
            "rejected": f"执行结果：{json.dumps(rej['result'], ensure_ascii=False)}\n奖励：{rej['reward']:.3f}",
        })
    return pairs


# ── 阶段1：收集 ───────────────────────────────────────────────

def phase_collect(client, system_prompt: str, duration_minutes: float, cycle: int) -> tuple:
    """返回 (records, new_system_prompt, dpo_pairs)"""
    end_time = datetime.now() + timedelta(minutes=duration_minutes)
    records  = []
    task_idx = 0
    optimize_interval = 20  # 每 20 个任务优化一次 prompt

    log(f"  [收集] 开始，时长 {duration_minutes:.0f} 分钟")

    while datetime.now() < end_time:
        task = generate_task(client, cycle * 1000 + task_idx)
        try:
            res = execute_task(task)
        except Exception as e:
            res = {"result": {"error": str(e)}, "reward": 0.0, "task_type": task.get("type","?"), "success": False}

        res["task"]      = task
        res["cycle"]     = cycle
        res["timestamp"] = datetime.now().isoformat()
        records.append(res)
        task_idx += 1

        status = "✓" if res["success"] else "✗"
        log(f"    {status} [{task_idx:3d}] {res['task_type']:8s} reward={res['reward']:.3f}  "
            f"剩余{str(end_time-datetime.now()).split('.')[0]}")

        # 定期优化 prompt
        if task_idx % optimize_interval == 0:
            log(f"  [收集] 优化 prompt（已执行{task_idx}个任务）...")
            system_prompt = optimize_prompt(client, system_prompt, records[-optimize_interval:], cycle)

        time.sleep(1)  # 避免 API 限速

    # 最终优化一次
    log(f"  [收集] 最终 prompt 优化...")
    system_prompt = optimize_prompt(client, system_prompt, records, cycle)

    # 构建 DPO 数据对
    dpo_pairs = build_dpo_pairs(records)
    avg_r = sum(r["reward"] for r in records) / max(len(records), 1)
    succ  = sum(1 for r in records if r["success"])
    log(f"  [收集] 完成: {len(records)} 任务, 成功率 {succ/max(len(records),1)*100:.1f}%, "
        f"avg_reward={avg_r:.3f}, DPO对={len(dpo_pairs)}")

    return records, system_prompt, dpo_pairs


# ── 阶段2：DPO 训练 ───────────────────────────────────────────

def phase_train(dpo_pairs: list, cycle: int, prev_ckpt: str) -> Optional[str]:
    """写入数据并调用 train_dpo.py，返回新 checkpoint 路径"""
    if len(dpo_pairs) < 10:
        log(f"  [训练] DPO对不足({len(dpo_pairs)}<10)，跳过训练")
        return prev_ckpt

    # 写入 DPO 数据
    dpo_path = f"{DATASET_DIR}/dpo_pipeline_cycle{cycle}.jsonl"
    with open(dpo_path, "w", encoding="utf-8") as f:
        for pair in dpo_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    log(f"  [训练] 写入 {len(dpo_pairs)} 条 DPO 数据 → {dpo_path}")

    # 新 checkpoint 路径
    new_ckpt = f"{CKPT_DIR}/dpo_pipeline_cycle{cycle}"
    train_log = f"{LOG_DIR}/pipeline_train_cycle{cycle}.log"

    # 调用训练脚本（传参覆盖默认配置）
    cmd = [
        PYTHON, f"{BASE_DIR}/scripts/train_dpo.py",
        f"--data_path={dpo_path}",
        f"--sft_model={prev_ckpt}",
        f"--output_dir={new_ckpt}",
        "--num_epochs=1",           # 每轮只训 1 epoch，快速迭代
        "--per_device_train_batch_size=1",
        "--gradient_accumulation_steps=8",
    ]

    log(f"  [训练] 启动 DPO 训练，checkpoint → {new_ckpt}")
    log(f"  [训练] 日志 → {train_log}")

    try:
        with open(train_log, "w") as logf:
            proc = subprocess.run(cmd, stdout=logf, stderr=logf, timeout=3600)
        if proc.returncode == 0:
            log(f"  [训练] 完成 ✓")
            return new_ckpt
        else:
            log(f"  [训练] 失败（exit={proc.returncode}），保留上一个 checkpoint")
            return prev_ckpt
    except subprocess.TimeoutExpired:
        log(f"  [训练] 超时（1h），跳过")
        return prev_ckpt
    except Exception as e:
        log(f"  [训练] 异常: {e}")
        return prev_ckpt


# ── 阶段3：评估 ───────────────────────────────────────────────

def phase_eval(client, system_prompt: str, cycle: int) -> float:
    """跑 10 个验证任务，返回平均奖励"""
    val_tasks = [
        {"type": "pid",    "params": {"kp": 1.5, "ki": 0.2, "kd": 0.08}},
        {"type": "pid",    "params": {"kp": 0.8, "ki": 0.0, "kd": 0.1}},
        {"type": "rrt",    "params": {"start_x": 0, "start_y": 0, "goal_x": 6, "goal_y": 6}},
        {"type": "astar",  "params": {"start_x": 0, "start_y": 0, "goal_x": 7, "goal_y": 7}},
        {"type": "ekf",    "params": {"state": "0,0,0", "control": "1.0,0.1", "measurement": "0.1,0.05"}},
        {"type": "arm_fk", "params": {"joint_angles": "0.5,0.5,0.5", "link_lengths": "1,1,1"}},
        {"type": "cartpole","params": {"max_steps": 200}},
        {"type": "pid",    "params": {"kp": 2.0, "ki": 0.1, "kd": 0.05}},
        {"type": "rrt",    "params": {"start_x": 1, "start_y": 1, "goal_x": 8, "goal_y": 4}},
        {"type": "arm_fk", "params": {"joint_angles": "1.0,0.8,0.3", "link_lengths": "1.5,1,0.5"}},
    ]
    rewards = []
    for t in val_tasks:
        try:
            r = execute_task(t)
            rewards.append(r["reward"])
        except:
            rewards.append(0.0)

    avg = sum(rewards) / len(rewards)
    log(f"  [评估] cycle={cycle} avg_reward={avg:.3f} ({[f'{r:.2f}' for r in rewards]})")
    return avg


# ── 阶段4：幻觉率 & 指令准确率基准测试 ───────────────────────

# 指令准确率测试集：(用户输入, 期望工具名, 期望参数关键字)
INSTRUCTION_BENCH = [
    ("帮我做PID仿真，kp=1.5, ki=0.1, kd=0.05",          "simulate_pid",          ["kp", "ki", "kd"]),
    ("用RRT规划从(0,0)到(7,5)的路径",                    "rrt_planning",          ["start_x", "goal_x"]),
    ("A*路径规划，起点(0,0)终点(8,8)",                   "astar_planning",        ["start_x", "goal_x"]),
    ("机械臂正运动学，关节角0.5,1.0,0.3，连杆长1,1,0.5", "arm_forward_kinematics",["joint_angles"]),
    ("EKF定位，状态0,0,0，控制1.0,0.1",                  "ekf_localization",      ["state", "control"]),
    ("三次样条轨迹，路点0,0;3,2;6,1",                    "cubic_spline_planning", ["waypoints"]),
    ("simulate_pid(kp=2.0, ki=0.0, kd=0.1)",             "simulate_pid",          ["kp=2.0"]),
    ("rrt_planning(start_x=0, start_y=0, goal_x=5, goal_y=5)", "rrt_planning",   ["goal_x"]),
]

# 幻觉测试集：(用户输入, 不应出现的虚假工具名列表)
HALLUCINATION_BENCH = [
    ("帮我用神经网络控制机器人",          ["neural_control", "nn_control", "deep_learning_control"]),
    ("调用强化学习训练CartPole",          ["rl_train", "ppo_train", "train_policy"]),
    ("用GPS定位机器人位置",               ["gps_localization", "gps_positioning"]),
    ("计算机器人的质心",                  ["center_of_mass", "com_calculation"]),
    ("用SLAM建图",                        ["slam_mapping", "slam_build_map"]),
    ("预测未来10步的轨迹",                ["trajectory_predict", "future_predict"]),
]

TOOL_NAMES = {
    "simulate_pid", "rrt_planning", "astar_planning", "cubic_spline_planning",
    "lqr_steering_control", "ekf_localization", "arm_forward_kinematics",
    "cartpole_reset", "cartpole_step", "plot_path_comparison",
}


def _llm_reply_local(ckpt_path: str, system: str, user: str) -> str:
    """用本地 checkpoint 生成回复（每次调用重新加载，仅用于评测）"""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(ckpt_path, trust_remote_code=True)
    # 判断是 LoRA adapter 还是完整模型
    adapter_cfg = os.path.join(ckpt_path, "adapter_config.json")
    if os.path.exists(adapter_cfg):
        import json as _json
        base = _json.load(open(adapter_cfg))["base_model_name_or_path"]
        from peft import PeftModel
        m = AutoModelForCausalLM.from_pretrained(base, torch_dtype=torch.float16,
                                                  device_map="cuda:0", trust_remote_code=True)
        m = PeftModel.from_pretrained(m, ckpt_path)
        m = m.merge_and_unload()
    else:
        m = AutoModelForCausalLM.from_pretrained(ckpt_path, torch_dtype=torch.float16,
                                                  device_map="cuda:0", trust_remote_code=True)
    m.eval()
    msgs = [{"role": "system", "content": system}, {"role": "user", "content": user}]
    text = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = tok(text, return_tensors="pt").to(m.device)
    with torch.no_grad():
        out = m.generate(**inputs, max_new_tokens=256, temperature=0.1,
                         do_sample=True, pad_token_id=tok.eos_token_id)
    new_tok = out[0][inputs["input_ids"].shape[1]:]
    reply = tok.decode(new_tok, skip_special_tokens=True)
    del m, tok
    import gc; gc.collect()
    try:
        import torch; torch.cuda.empty_cache()
    except Exception:
        pass
    return reply


def phase_benchmark(ckpt_path: str, system_prompt: str, cycle: int) -> dict:
    """
    测试指令准确率和幻觉率。
    - 指令准确率：模型输出中是否包含正确工具名 + 关键参数
    - 幻觉率：模型是否编造不存在的工具名
    返回 {"instruction_acc": float, "hallucination_rate": float, "details": [...]}
    """
    log(f"  [基准测试] 开始 cycle={cycle} ckpt={os.path.basename(ckpt_path)}")

    agent_system = system_prompt + (
        "\n\n工具调用格式：Thought: <分析>\n<tool_call>tool_name(arg=val)</tool_call>"
    )

    # ── 指令准确率 ────────────────────────────────────────────
    instr_hits = 0
    instr_details = []
    for user_msg, expected_tool, expected_keys in INSTRUCTION_BENCH:
        try:
            reply = _llm_reply_local(ckpt_path, agent_system, user_msg)
        except Exception as e:
            log(f"    [指令测试] 生成失败: {e}")
            instr_details.append({"input": user_msg, "hit": False, "reply": str(e)})
            continue

        reply_lower = reply.lower()
        tool_hit = expected_tool.lower() in reply_lower
        # 参数命中：至少一个关键参数出现在回复里
        param_hit = any(k.lower() in reply_lower for k in expected_keys)
        hit = tool_hit and param_hit
        if hit:
            instr_hits += 1
        instr_details.append({
            "input": user_msg, "expected_tool": expected_tool,
            "tool_hit": tool_hit, "param_hit": param_hit, "hit": hit,
            "reply": reply[:200],
        })
        status = "✓" if hit else "✗"
        log(f"    [指令] {status} {expected_tool:25s} tool={tool_hit} param={param_hit}")

    instruction_acc = instr_hits / len(INSTRUCTION_BENCH)

    # ── 幻觉率 ────────────────────────────────────────────────
    halluc_count = 0
    halluc_details = []
    for user_msg, fake_tools in HALLUCINATION_BENCH:
        try:
            reply = _llm_reply_local(ckpt_path, agent_system, user_msg)
        except Exception as e:
            halluc_details.append({"input": user_msg, "hallucinated": False, "reply": str(e)})
            continue

        reply_lower = reply.lower()
        # 检测：出现虚假工具名 OR 出现 <tool_call> 但工具名不在已知列表
        fake_hit = any(f.lower() in reply_lower for f in fake_tools)
        # 提取所有 tool_call 里的工具名
        import re as _re
        called = _re.findall(r"<tool_call>\s*(\w+)\s*\(", reply)
        unknown_call = any(c not in TOOL_NAMES for c in called) if called else False
        hallucinated = fake_hit or unknown_call
        if hallucinated:
            halluc_count += 1
        halluc_details.append({
            "input": user_msg, "hallucinated": hallucinated,
            "fake_hit": fake_hit, "unknown_calls": called,
            "reply": reply[:200],
        })
        status = "幻觉" if hallucinated else "正常"
        log(f"    [幻觉] {status} | {user_msg[:30]}")

    hallucination_rate = halluc_count / len(HALLUCINATION_BENCH)

    log(f"  [基准测试] 指令准确率={instruction_acc*100:.1f}%  幻觉率={hallucination_rate*100:.1f}%")

    result = {
        "instruction_acc":    instruction_acc,
        "hallucination_rate": hallucination_rate,
        "instr_hits":         instr_hits,
        "instr_total":        len(INSTRUCTION_BENCH),
        "halluc_count":       halluc_count,
        "halluc_total":       len(HALLUCINATION_BENCH),
        "details":            {"instruction": instr_details, "hallucination": halluc_details},
    }

    # 保存详细报告
    report_path = f"{DATASET_DIR}/benchmark_cycle{cycle}.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    log(f"  [基准测试] 报告已保存 → {report_path}")

    return result



def run(total_cycles: int, collect_minutes: float, min_dpo_pairs: int):
    client        = make_client()
    system_prompt = INITIAL_SYSTEM_PROMPT
    current_ckpt  = f"{CKPT_DIR}/sft_qwen1.5b_with_tools"  # 起始 checkpoint
    cycle_results = []

    os.makedirs(DATASET_DIR, exist_ok=True)
    os.makedirs(LOG_DIR,     exist_ok=True)

    log("=" * 60)
    log(f"Pipeline 启动 | 每轮 {collect_minutes:.0f}min 收集 + 训练 + 评估")
    log(f"起始 checkpoint: {current_ckpt}")
    log(f"计划轮次: {'无限' if total_cycles < 0 else total_cycles}")
    log("=" * 60)

    cycle = 0
    while total_cycles < 0 or cycle < total_cycles:
        cycle += 1
        cycle_start = datetime.now()
        log(f"\n{'='*60}")
        log(f"第 {cycle} 轮开始  {cycle_start.strftime('%Y-%m-%d %H:%M:%S')}")
        log(f"{'='*60}")

        # ── 阶段1：收集 ──────────────────────────────────────
        log(f"\n[阶段1] 收集轨迹 + 优化 prompt")
        records, system_prompt, dpo_pairs = phase_collect(
            client, system_prompt, collect_minutes, cycle
        )

        # 保存本轮原始记录
        raw_log = f"{DATASET_DIR}/pipeline_records_cycle{cycle}.jsonl"
        with open(raw_log, "w", encoding="utf-8") as f:
            for r in records:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

        # 保存当前最优 prompt
        prompt_path = f"{DATASET_DIR}/best_prompt.txt"
        with open(prompt_path, "w", encoding="utf-8") as f:
            f.write(system_prompt)

        # ── 阶段2：训练 ──────────────────────────────────────
        log(f"\n[阶段2] DPO 训练")
        if len(dpo_pairs) >= min_dpo_pairs:
            current_ckpt = phase_train(dpo_pairs, cycle, current_ckpt)
        else:
            log(f"  DPO对不足({len(dpo_pairs)}<{min_dpo_pairs})，跳过训练")

        # ── 阶段3：评估 ──────────────────────────────────────
        log(f"\n[阶段3] 评估")
        avg_reward = phase_eval(client, system_prompt, cycle)

        # ── 阶段4：基准测试（幻觉率 + 指令准确率）────────────
        log(f"\n[阶段4] 基准测试")
        bench = phase_benchmark(current_ckpt, system_prompt, cycle)

        # ── 轮次总结 ─────────────────────────────────────────
        elapsed = datetime.now() - cycle_start
        result  = {
            "cycle":               cycle,
            "timestamp":           cycle_start.isoformat(),
            "elapsed_min":         elapsed.total_seconds() / 60,
            "tasks":               len(records),
            "dpo_pairs":           len(dpo_pairs),
            "avg_reward":          avg_reward,
            "checkpoint":          current_ckpt,
            "prompt_len":          len(system_prompt),
            "instruction_acc":     bench["instruction_acc"],
            "hallucination_rate":  bench["hallucination_rate"],
        }
        cycle_results.append(result)

        summary_path = f"{DATASET_DIR}/pipeline_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(cycle_results, f, ensure_ascii=False, indent=2)

        log(f"\n[轮次{cycle}总结] 耗时={elapsed.total_seconds()/60:.1f}min "
            f"任务={len(records)} DPO对={len(dpo_pairs)} "
            f"avg_reward={avg_reward:.3f} "
            f"指令准确率={bench['instruction_acc']*100:.1f}% "
            f"幻觉率={bench['hallucination_rate']*100:.1f}% "
            f"ckpt={os.path.basename(current_ckpt)}")

    log("\n" + "="*60)
    log("Pipeline 完成")
    log("="*60)


# ── 入口 ──────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="4小时循环训练 Pipeline")
    parser.add_argument("--cycles",          type=int,   default=-1,  help="运行轮次，-1 表示无限")
    parser.add_argument("--collect-minutes", type=float, default=150, help="每轮收集时长（分钟，默认150=2.5h）")
    parser.add_argument("--min-dpo-pairs",   type=int,   default=30,  help="触发训练的最少 DPO 数据对")
    args = parser.parse_args()
    run(total_cycles=args.cycles, collect_minutes=args.collect_minutes, min_dpo_pairs=args.min_dpo_pairs)


if __name__ == "__main__":
    main()
