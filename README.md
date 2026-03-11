# Robot-LLM-Align

> 机器人控制领域的大模型偏好对齐项目

## 项目简介

本项目专注于在机器人控制领域（强化学习、PID 控制、MPC 等）构建专属偏好数据集，并使用 DPO 算法进行模型对齐，旨在减少大模型在专业领域的幻觉问题。

## 核心特性

- 自动化生成领域专属偏好数据（覆盖强化学习、经典控制、现代控制等）
- 基于 DPO 算法的偏好对齐训练
- LLM-as-a-Judge 评估框架
- 完整的训练和评估流程

## 项目结构

```
robot-llm-align/
├── dataset/              # 数据集目录
├── scripts/              # 训练和评估脚本
├── checkpoints/          # 模型权重
├── results/              # 实验结果
├── README.md
└── requirements.txt
```

## 快速开始

详细的两周实施计划请参考：[DPO_TWO_WEEK_PLAN.md](./DPO_TWO_WEEK_PLAN.md)

### 环境要求

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+
- 至少一张 24GB 显存的 GPU（推荐 RTX 4090）

### 安装依赖

```bash
pip install -r requirements.txt
```

## 数据构建策略

### 三阶段数据方案

#### 阶段 1：通用指令遵循能力（200-300 条）
**目的**：让模型学会基本的对话格式和指令遵循

```python
# 使用 HuggingFace 高质量通用数据
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:300]")
```

**优点**：快速建立基础对话能力
**缺点**：缺少领域专业性

#### 阶段 2：领域基础知识注入（核心，500-800 条）
**目的**：构建机器人控制领域的专业知识库

**方案 A：爬取专业问答**
- 来源：Stack Overflow (`reinforcement-learning`, `control-theory` 标签)、Robotics Stack Exchange
- 优点：真实问题，逻辑严密
- 缺点：需要清洗，格式不统一

**方案 B：利用 LLM 生成（推荐用于偏好数据）**
```python
# 用 GPT-4o 生成高质量的领域 QA 对
system_prompt = """
你是一个机器人控制领域的教授。请生成一个关于 {topic} 的问答对：
- 问题要具体、有深度（不要泛泛而谈）
- 答案要包含：原理解释 + 数学公式 + 实际应用场景
- 答案长度：200-400 字
"""

topics = [
    "Q-learning 的 off-policy 特性",
    "PID 控制器的参数整定方法",
    "MPC 的滚动优化原理",
    # ... 50 个精选主题
]
```

**方案 C：从教材/论文提取（本项目采用）**
- 来源：
  - Sutton & Barto 的《Reinforcement Learning: An Introduction》
  - 经典控制理论教材（如《自动控制原理》）
  - 顶会论文（ICRA, IROS, RSS）
- 优点：权威、准确、逻辑严密
- 缺点：需要人工整理成 QA 格式
- 实施方式：
  1. 提取关键概念和定理
  2. 将其改写为问答对
  3. 保留数学公式和推导过程

#### 阶段 3：多轮对话能力（可选，200 条）
**目的**：模拟用户追问的场景

示例：
```
Q1: "什么是 Q-learning？"
A1: [基础解释]
Q2: "它和 SARSA 有什么区别？"
A2: [对比分析]
Q3: "哪个更适合机器人导航？"
A3: [应用场景分析]
```

### 数据质量标准

**SFT 数据要求**：
- 答案准确无误（可引用权威来源验证）
- 包含数学公式（LaTeX 格式）
- 指出常见误区
- 给出实际应用场景

**偏好数据要求**：
- Chosen 答案：逻辑严密、无幻觉
- Rejected 答案：看似合理但包含致命错误（概念混淆、公式错误、适用场景错误）

## 开发计划

- [ ] Week 1: 基建验证与领域数据构建
  - [ ] Day 1: 环境验证
  - [ ] Day 2-3: SFT 基线（通用数据 300 条 + 领域数据 500 条）
  - [ ] Day 4-7: 构建偏好数据（150 对）
- [ ] Week 2: DPO 训练与效果验证
  - [ ] Day 8-9: DPO 训练
  - [ ] Day 10-11: LLM-as-a-Judge 评估
  - [ ] Day 12-13: 超参数优化
  - [ ] Day 14: 项目打包

## 技术栈

- PyTorch
- Transformers
- PEFT (LoRA)
- TRL (DPO)
- Weights & Biases

## License

MIT

## 联系方式

如有问题，欢迎提交 Issue。
