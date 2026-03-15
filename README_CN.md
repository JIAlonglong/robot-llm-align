# Robot-LLM-Align

[![GitHub](https://img.shields.io/badge/GitHub-JIAlonglong-blue?logo=github)](https://github.com/JIAlonglong/robot-llm-align)
[![License](https://img.shields.io/badge/license-MIT-green)](./LICENSE)

> 机器人控制领域大模型偏好对齐 + Agent 系统

**中文文档 | [English](./README.md)**

---

## 项目简介

以机器人控制领域（PID、路径规划、EKF、机械臂运动学等）为切入点，构建专属数据集并训练对齐模型，同时集成 PythonRobotics 工具库，打造具备真实工具调用能力的 Robot Control Agent。

**两条主线并行推进：**
- **对齐主线**：SFT → DPO，减少大模型在专业领域的幻觉
- **Agent 主线**：ReAct 循环 + 规则路由工具调用，让模型真正能"做事"

**自动化流水线**：每 4 小时一轮 — 收集轨迹 → DPO 训练 → 评估 → 基准测试

---

## 项目结构

```
robot-llm-align/
├── dataset/                              # 训练数据（运行时输出已 gitignore）
│   ├── sft_combined_v2.jsonl             # SFT 通用数据
│   ├── sft_with_tools.jsonl              # SFT 含工具调用数据
│   ├── dpo_pairs.jsonl                   # DPO 偏好对数据
│   └── dpo_train.jsonl                   # DPO 训练数据（流水线输出）
├── scripts/
│   ├── pipeline.py                       # 4 小时自动循环（收集→训练→评估→基准）
│   ├── continuous_optimize.py            # 12 小时独立提示词优化
│   ├── train_sft.py                      # SFT 训练（Qwen2.5-7B + LoRA）
│   ├── train_sft_1.5b.py                 # SFT 训练（Qwen2.5-1.5B + LoRA）
│   ├── train_dpo.py                      # DPO 训练（7B）
│   ├── train_dpo_1.5b.py                 # DPO 训练（1.5B）
│   ├── evaluate.py                       # LLM-as-a-Judge 评估
│   ├── agent/
│   │   ├── app.py                        # Web UI（3 标签：Agent / 深度搜索 / 流水线监控）
│   │   ├── agent_executor.py             # ReAct 执行引擎
│   │   ├── tool_registry.py              # 工具注册与分发
│   │   ├── reward.py                     # 奖励函数
│   │   └── tools/
│   │       └── python_robotics_tools.py  # PythonRobotics 工具适配器
│   └── data_processing/                  # 数据生成脚本
├── checkpoints/                          # 已训练模型权重（gitignore）
├── PythonRobotics/                       # 机器人算法库（工具来源）
└── logs/                                 # 训练日志（gitignore）
```

---

## 环境配置

### 前置要求

- Python 3.10+，PyTorch 2.0+，CUDA 11.8+（仅训练需要）
- conda 环境：`LLM`

### 安装依赖

```bash
pip install transformers peft trl datasets accelerate bitsandbytes
pip install gradio openai arxiv
```

### 配置 API Key

```bash
export SILICONFLOW_API_KEY="your_key_here"   # 流水线和深度搜索必需
export WANDB_API_KEY="your_key_here"         # 可选，用于训练追踪
```

在 [siliconflow.cn](https://siliconflow.cn) 获取硅基流动 API Key，用于 DeepSeek-V3.2 任务生成和提示词优化。

---

## 快速开始

### 启动 Web UI

```bash
cd /path/to/robot-llm-align
conda run -n LLM python scripts/agent/app.py

# 远程服务器访问：
ssh -L 7860:localhost:7860 user@server
# 打开 http://localhost:7860
```

UI 自动加载最新 DPO 流水线 checkpoint 和最优 system prompt。三个标签页：

| 标签 | 说明 |
|------|------|
| **Agent 控制台** | 与模型对话；点击工具按钮（PID、RRT、A*、EKF 等）直接运行工具 |
| **深度搜索** | 按关键词搜索 arxiv 论文，获取 AI 生成摘要 |
| **流水线监控** | 查看训练轮次历史、当前 checkpoint、基准测试指标 |

### 运行自动化流水线

```bash
# 单次 4 小时循环
conda run -n LLM python scripts/pipeline.py

# 持续运行（每 4 小时一轮，无限循环）
conda run -n LLM python scripts/pipeline.py --continuous
```

每轮流程：
1. **收集** — DeepSeek 生成任务，规则路由 Agent 执行，记录轨迹
2. **训练** — 基于收集的偏好对进行 DPO 训练
3. **评估** — 10 个验证任务，计算平均奖励
4. **基准测试** — 指令准确率（8 个用例）+ 幻觉率（6 个用例）

结果保存到 `dataset/pipeline_summary.json`。

### 独立提示词优化（12 小时）

```bash
conda run -n LLM python scripts/continuous_optimize.py --hours 12 --rounds-per-cycle 15
```

---

## 工具集

| 类别 | 工具名 | 功能 |
|------|--------|------|
| 仿真 | `simulate_pid(kp, ki, kd)` | PID 控制仿真，返回超调/稳态误差/图表 |
| 仿真 | `cartpole_reset()` / `cartpole_step(action)` | CartPole 物理仿真 |
| 路径规划 | `rrt_planning(sx, sy, gx, gy)` | RRT 随机树规划 |
| 路径规划 | `astar_planning(sx, sy, gx, gy)` | A* 网格规划 |
| 路径规划 | `cubic_spline_planning(waypoints)` | 三次样条轨迹 |
| 控制 | `lqr_steering_control(x, y, yaw, v, ref_path)` | LQR 跟踪控制 |
| 定位 | `ekf_localization(state, control, measurement)` | 扩展卡尔曼滤波 |
| 机械臂 | `arm_forward_kinematics(joint_angles, link_lengths)` | 正运动学 |
| 可视化 | `plot_path_comparison(paths, labels)` | 多路径对比图 |

---

## 手动训练

```bash
# SFT（1.5B，显存友好）
conda run -n LLM python scripts/train_sft_1.5b.py

# DPO（1.5B）
conda run -n LLM python scripts/train_dpo_1.5b.py

# 评估
conda run -n LLM python scripts/evaluate.py --mode all
```

---

## 技术栈

- **模型**：Qwen2.5-1.5B-Instruct / Qwen2.5-7B-Instruct + LoRA（PEFT）
- **训练**：SFT → DPO（TRL）
- **工具库**：PythonRobotics（路径规划/控制/定位/机械臂）
- **Web UI**：Gradio + 自定义深色主题 CSS
- **数据生成**：DeepSeek-V3.2（硅基流动 API）
- **实验追踪**：Weights & Biases（可选）

---

## License

MIT

## 作者

**JIAlonglong** — [GitHub](https://github.com/JIAlonglong)

如有问题，欢迎提交 Issue 或 PR。
