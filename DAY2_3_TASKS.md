# Day 2-3 详细任务清单

> 构建 SFT 数据集（预计 12 小时，分两天完成）

---

## 任务目标

构建高质量的 SFT 训练数据集，包含：
- 300 条通用指令数据（来自 HuggingFace）
- 100 条机器人控制领域专业数据（从教材提取）

---

## Day 2 任务（6 小时）

### Task 2.1：下载通用指令数据（30 分钟）

**目标**：获取 300 条高质量的通用对话数据

**步骤**：

```bash
cd /home/liujl/big_model/robot-llm-align

# 创建数据处理脚本
mkdir -p scripts/data_processing
```

**创建下载脚本**：`scripts/data_processing/download_general_data.py`

```python
#!/usr/bin/env python3
"""
下载通用指令数据
从 HuggingFace 下载 ultrachat_200k 数据集的前 300 条
"""

from datasets import load_dataset
import json
import os

def download_and_convert():
    """下载并转换数据格式"""
    print("正在下载数据集...")
    
    # 下载数据
    dataset = load_dataset(
        "HuggingFaceH4/ultrachat_200k",
        split="train_sft[:300]"
    )
    
    print(f"下载完成，共 {len(dataset)} 条数据")
    
    # 转换格式
    converted_data = []
    for idx, item in enumerate(dataset):
        converted_item = {
            "id": f"sft_general_{idx:03d}",
            "source": "ultrachat",
            "conversations": item["messages"],
            "metadata": {
                "topic": "general",
                "difficulty": "medium",
                "has_formula": False,
                "created_at": "2026-03-11"
            }
        }
        converted_data.append(converted_item)
    
    # 保存为 JSONL
    output_file = "dataset/sft_general_300.jsonl"
    os.makedirs("dataset", exist_ok=True)
    
    with open(output_file, "w", encoding="utf-8") as f:
        for item in converted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"数据已保存到: {output_file}")
    
    # 显示示例
    print("\n示例数据:")
    print(json.dumps(converted_data[0], ensure_ascii=False, indent=2))

if __name__ == "__main__":
    download_and_convert()
```

**运行脚本**：
```bash
python scripts/data_processing/download_general_data.py
```

**验证标准**：
- [ ] 生成 `dataset/sft_general_300.jsonl`
- [ ] 文件包含 300 条数据
- [ ] 数据格式符合规范

---

### Task 2.2：准备教材资源（1 小时）

**目标**：准备用于提取领域知识的教材资源

**推荐教材**：

1. **强化学习**：
   - 《Reinforcement Learning: An Introduction》(Sutton & Barto, 2nd Edition)
   - 重点章节：Ch3-Ch6（MDP、动态规划、MC、TD、Q-learning、SARSA）
   - 下载地址：http://incompleteideas.net/book/RLbook2020.pdf

2. **经典控制**：
   - 《自动控制原理》（胡寿松）或任何经典控制教材
   - 重点章节：PID 控制、状态空间、频域分析

3. **现代控制**：
   - 《Model Predictive Control》相关资料
   - 重点：MPC 基础、约束优化

**步骤**：
```bash
# 创建教材目录
mkdir -p references/textbooks

# 下载 Sutton & Barto RL 书（如果网络允许）
cd references/textbooks
wget http://incompleteideas.net/book/RLbook2020.pdf

# 或者手动下载后放到这个目录
```

**验证标准**：
- [ ] 至少准备好一本教材（PDF 或纸质书）
- [ ] 确认可以访问关键章节

---

### Task 2.3：提取强化学习 QA 对（3 小时）

**目标**：从 RL 教材提取 50 个高质量 QA 对

**提取策略**：

**核心概念清单**（从 Sutton & Barto 书中提取）：
1. MDP（马尔可夫决策过程）
2. Bellman 方程
3. 动态规划（值迭代、策略迭代）
4. Monte Carlo 方法
5. TD Learning（时序差分学习）
6. Q-learning
7. SARSA
8. On-policy vs Off-policy
9. Exploration vs Exploitation
10. ε-greedy 策略

**提取模板**：

```
【教材原文】
"Q-learning is an off-policy TD control algorithm. The update rule is:
Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]"

【改写为 QA 对】
Q: "解释 Q-learning 的 off-policy 特性及其更新公式"
A: "Q-learning 是一种 off-policy 时序差分控制算法。

**Off-policy 的含义**：
行为策略（用于探索环境）和目标策略（用于更新价值函数）可以不同。

**更新公式**：
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

**关键点**：
1. 使用 max 操作选择下一状态的最优动作（贪婪策略）
2. 但实际执行时可以使用 ε-greedy（探索策略）
3. 这种解耦使得 Q-learning 收敛更稳定

**应用场景**：
机器人导航、游戏 AI、自动驾驶决策"
```

**创建数据收集模板**：`dataset/rl_qa_template.jsonl`

手动填写 50 个 QA 对（这是最耗时的部分）

**验证标准**：
- [ ] 至少提取 30 个 QA 对（Day 2 目标）
- [ ] 每个 QA 对包含：概念解释 + 数学公式（如适用）+ 应用场景
- [ ] 答案准确无误（可用 GPT-4o 验证）

---

### Task 2.4：格式化 RL 数据（1.5 小时）

**目标**：将提取的 QA 对转换为标准格式

**创建格式化脚本**：`scripts/data_processing/format_rl_data.py`

```python
#!/usr/bin/env python3
"""
格式化强化学习 QA 数据
将手动提取的 QA 对转换为标准 JSONL 格式
"""

import json
import os
from datetime import datetime

# 手动提取的 QA 对（示例）
rl_qa_pairs = [
    {
        "question": "解释 Q-learning 的 off-policy 特性及其更新公式",
        "answer": """Q-learning 是一种 off-policy 时序差分控制算法。

**Off-policy 的含义**：
行为策略（用于探索环境）和目标策略（用于更新价值函数）可以不同。

**更新公式**：
Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

**关键点**：
1. 使用 max 操作选择下一状态的最优动作（贪婪策略）
2. 但实际执行时可以使用 ε-greedy（探索策略）
3. 这种解耦使得 Q-learning 收敛更稳定

**应用场景**：
机器人导航、游戏 AI、自动驾驶决策""",
        "difficulty": "medium",
        "has_formula": True,
        "keywords": ["q-learning", "off-policy", "td-learning"]
    },
    # ... 添加更多 QA 对
]

def format_data():
    """格式化数据"""
    formatted_data = []
    
    for idx, qa in enumerate(rl_qa_pairs):
        item = {
            "id": f"sft_rl_{idx:03d}",
            "source": "textbook",
            "conversations": [
                {
                    "role": "system",
                    "content": "你是一个机器人控制领域的专家，擅长强化学习、PID 控制、MPC 等技术。请用专业、准确的语言回答问题，必要时包含数学公式。"
                },
                {
                    "role": "user",
                    "content": qa["question"]
                },
                {
                    "role": "assistant",
                    "content": qa["answer"]
                }
            ],
            "metadata": {
                "topic": "reinforcement_learning",
                "difficulty": qa.get("difficulty", "medium"),
                "has_formula": qa.get("has_formula", False),
                "keywords": qa.get("keywords", []),
                "created_at": datetime.now().strftime("%Y-%m-%d")
            }
        }
        formatted_data.append(item)
    
    # 保存
    output_file = "dataset/sft_robotics_rl.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in formatted_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"已格式化 {len(formatted_data)} 条数据")
    print(f"保存到: {output_file}")

if __name__ == "__main__":
    format_data()
```

**验证标准**：
- [ ] 生成 `dataset/sft_robotics_rl.jsonl`
- [ ] 数据格式符合 DATA_FORMAT_SPEC.md 规范
- [ ] 每条数据包含完整的 metadata

---

## Day 3 任务（6 小时）

### Task 3.1：继续提取 RL QA 对（2 小时）

**目标**：完成剩余的 20 个 RL QA 对

**重点概念**（继续提取）：
- Actor-Critic 算法
- Policy Gradient
- PPO（Proximal Policy Optimization）
- DQN（Deep Q-Network）
- Experience Replay
- Target Network

**验证标准**：
- [ ] 总共完成 50 个 RL QA 对
- [ ] 覆盖 RL 的核心概念

---

### Task 3.2：提取 PID 控制 QA 对（2 小时）

**目标**：提取 30 个 PID 控制相关的 QA 对

**核心概念清单**：
1. PID 控制器的基本原理
2. 比例（P）、积分（I）、微分（D）的作用
3. PID 参数整定方法（Ziegler-Nichols）
4. 积分饱和问题
5. 微分噪声问题
6. PID 的适用场景和局限性
7. 数字 PID 实现
8. 增量式 PID vs 位置式 PID

**提取示例**：

```
Q: "解释 PID 控制器中积分饱和问题及其解决方法"
A: "积分饱和（Integral Windup）是 PID 控制中的常见问题。

**问题描述**：
当系统长时间存在偏差时，积分项会不断累积，导致控制量过大，系统响应变慢甚至不稳定。

**产生原因**：
1. 执行器存在物理限制（如阀门开度 0-100%）
2. 积分项持续累积超出执行器能力范围

**解决方法**：
1. **积分限幅**：限制积分项的最大值
2. **条件积分**：只在偏差较小时进行积分
3. **抗饱和算法**：检测到饱和时停止积分累积

**代码示例**：
```python
# 积分限幅
integral += error * dt
integral = max(min(integral, integral_max), integral_min)
```

**应用场景**：
温度控制、电机速度控制、液位控制"
```

**验证标准**：
- [ ] 完成 30 个 PID QA 对
- [ ] 包含常见问题和解决方案

---

### Task 3.3：提取 MPC/现代控制 QA 对（1.5 小时）

**目标**：提取 20 个 MPC 和现代控制相关的 QA 对

**核心概念清单**：
1. MPC 的基本原理
2. 滚动优化（Receding Horizon）
3. 约束处理
4. 预测模型
5. 状态空间表示
6. LQR（线性二次调节器）
7. 卡尔曼滤波

**验证标准**：
- [ ] 完成 20 个 MPC/现代控制 QA 对
- [ ] 覆盖核心概念

---

### Task 3.4：合并所有数据（30 分钟）

**目标**：将所有数据合并为一个完整的训练集

**创建合并脚本**：`scripts/data_processing/merge_datasets.py`

```python
#!/usr/bin/env python3
"""
合并所有 SFT 数据集
"""

import json
import random

def merge_datasets():
    """合并数据集"""
    all_data = []
    
    # 读取通用数据
    with open("dataset/sft_general_300.jsonl", "r", encoding="utf-8") as f:
        for line in f:
            all_data.append(json.loads(line))
    
    # 读取领域数据
    domain_files = [
        "dataset/sft_robotics_rl.jsonl",
        "dataset/sft_robotics_pid.jsonl",
        "dataset/sft_robotics_mpc.jsonl"
    ]
    
    for file in domain_files:
        try:
            with open(file, "r", encoding="utf-8") as f:
                for line in f:
                    all_data.append(json.loads(line))
        except FileNotFoundError:
            print(f"警告: {file} 不存在，跳过")
    
    # 打乱顺序
    random.seed(42)
    random.shuffle(all_data)
    
    # 保存
    output_file = "dataset/sft_combined.jsonl"
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"合并完成，共 {len(all_data)} 条数据")
    print(f"保存到: {output_file}")
    
    # 统计信息
    topics = {}
    for item in all_data:
        topic = item["metadata"].get("topic", "unknown")
        topics[topic] = topics.get(topic, 0) + 1
    
    print("\n数据分布:")
    for topic, count in topics.items():
        print(f"  {topic}: {count} 条")

if __name__ == "__main__":
    merge_datasets()
```

**运行脚本**：
```bash
python scripts/data_processing/merge_datasets.py
```

**验证标准**：
- [ ] 生成 `dataset/sft_combined.jsonl`
- [ ] 总数据量约 400 条（300 通用 + 100 领域）
- [ ] 数据已打乱顺序

---

## 交付物清单

完成 Day 2-3 后，应该有以下文件：

```
robot-llm-align/
├── dataset/
│   ├── sft_general_300.jsonl          ✅ 通用数据
│   ├── sft_robotics_rl.jsonl          ✅ RL 数据
│   ├── sft_robotics_pid.jsonl         ✅ PID 数据
│   ├── sft_robotics_mpc.jsonl         ✅ MPC 数据
│   ├── sft_combined.jsonl             ✅ 合并数据
│   └── README.md                      ✅ 数据说明
├── scripts/data_processing/
│   ├── download_general_data.py       ✅ 下载脚本
│   ├── format_rl_data.py              ✅ 格式化脚本
│   └── merge_datasets.py              ✅ 合并脚本
└── references/
    └── textbooks/                     ✅ 教材资源
        └── RLbook2020.pdf
```

---

## 数据质量检查清单

在进入训练之前，务必检查：

- [ ] 所有数据格式符合 DATA_FORMAT_SPEC.md 规范
- [ ] 每条数据都有唯一的 id
- [ ] conversations 字段包含完整的对话
- [ ] metadata 字段完整
- [ ] 答案准确无误（可抽查 10 条用 GPT-4o 验证）
- [ ] 数学公式格式正确
- [ ] 没有重复数据

---

## 常见问题

### Q1: 手动提取 QA 对太慢怎么办？
**方案 A**：使用 GPT-4o 辅助生成
```python
# 给 GPT-4o 提供教材片段，让它生成 QA 对
prompt = f"""
基于以下教材内容，生成一个高质量的 QA 对：

【教材内容】
{textbook_excerpt}

【要求】
1. 问题要具体、有深度
2. 答案要包含：原理解释 + 数学公式 + 应用场景
3. 答案长度 200-400 字
"""
```

**方案 B**：降低数量要求
- RL: 30 个（而不是 50 个）
- PID: 20 个（而不是 30 个）
- MPC: 10 个（而不是 20 个）
- 总计：60 个领域数据

### Q2: 没有教材怎么办？
使用在线资源：
- Sutton & Barto RL 书：http://incompleteideas.net/book/
- Wikipedia 相关词条
- 知乎/CSDN 的高质量文章

### Q3: 如何验证数据质量？
```python
# 使用 GPT-4o 验证答案准确性
def verify_answer(question, answer):
    prompt = f"""
    请评估以下答案的准确性（1-5 分）：
    
    问题：{question}
    答案：{answer}
    
    评分标准：
    5分：完全准确，逻辑严密
    4分：基本准确，有小瑕疵
    3分：部分正确，有明显错误
    2分：大部分错误
    1分：完全错误
    """
    # 调用 GPT-4o API
```

---

## 下一步

完成 Day 2-3 后，进入 **Day 4：SFT 训练**。

需要我帮你：
1. 生成一些 QA 对的示例？
2. 写数据验证脚本？
3. 或者直接开始提取数据？
