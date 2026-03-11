# 大模型 DPO 偏好对齐两周实战计划

> **项目目标**：在机器人控制领域构建专属偏好数据集，使用 DPO 算法进行模型对齐，产出可用于简历的高质量项目成果

## 硬件资源情况

- **GPU 配置**：8 × RTX 4090 (24GB)
- **可用资源**：GPU 3、5 完全空闲，GPU 4、7 部分空闲
- **建议使用**：`export CUDA_VISIBLE_DEVICES=3`（独占 GPU 3）

## 已有基础

- ✅ minimind 项目（包含基础训练代码）
- ✅ minimind_ljl 工作目录（已配置 uv 环境）
- ✅ 核心依赖已安装：transformers 4.57.1, peft 0.7.1, trl 0.13.0

---

## 第一周：基建验证与领域数据构建（Day 1-7）

### Day 1：环境验证与资源分配（2小时）

**目标**：确保能独占一张 GPU 跑通完整流程

#### 任务清单

```bash
# 1. 指定使用 GPU 3（空闲）
export CUDA_VISIBLE_DEVICES=3

# 2. 验证 transformers 版本
python -c "import transformers; print(transformers.__version__)"

# 3. 测试模型加载（使用 Qwen2.5-1.5B）
python -c "from transformers import AutoModelForCausalLM; \
model = AutoModelForCausalLM.from_pretrained('Qwen/Qwen2.5-1.5B-Instruct', device_map='auto')"
```

#### 交付物

- `env_check.py`：环境检查脚本
  - 输出 GPU 信息
  - 输出库版本
  - 输出模型加载成功标志

---

### Day 2-3：SFT 基线建立（关键里程碑）

**目标**：在 1000 条通用数据上跑通 LoRA 微调

#### 数据准备（2小时）

```python
# 使用 HuggingFace 的高质量数据集
from datasets import load_dataset
dataset = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:1000]")
```

#### 训练配置

**LoRA 参数**：

- `r=8`
- `lora_alpha=16`
- `target_modules=["q_proj", "v_proj"]`

**训练参数**：

- `batch_size=4`
- `gradient_accumulation_steps=4`
- `max_steps=500`
- 预计训练时间：1.5 小时（单卡 4090）

#### 验证标准

```python
# 测试指令遵循能力
prompt = "解释什么是强化学习中的 Q-learning"
# 模型应该能给出结构化回答，而不是乱码
```

#### 交付物

- `sft_baseline.py`：训练脚本
- `checkpoints/sft_baseline/`：模型权重
- `sft_log.txt`：训练日志（包含 loss 曲线）

---

### Day 4-7：构建机器人控制领域偏好数据（核心壁垒）

#### Day 4：设计数据生成策略（4小时）

**关键问题清单**（先回答这些再写代码）：

1. **覆盖哪些子领域？**
  - 强化学习：Q-learning, PPO, SAC
  - 经典控制：PID 控制
  - 现代控制：MPC（模型预测控制）
  - 运动规划：轨迹规划
2. **什么样的错误答案最有价值？**
  - **类型 A**：概念混淆（把 Q-learning 和 SARSA 的更新公式搞混）
  - **类型 B**：参数误用（PID 调参时 Kp/Ki/Kd 的作用说反）
  - **类型 C**：适用场景错误（在非线性系统上推荐线性控制器）

**System Prompt 设计**（数据质量的生命线）：

```python
SYSTEM_PROMPT = """
你是一个机器人控制领域的专家。请针对以下问题生成两个答案：

【Chosen 答案要求】
1. 引用经典教材/论文（如 Sutton & Barto 的 RL 书）
2. 包含数学公式（用 LaTeX 格式）
3. 指出常见误区
4. 给出实际应用场景

【Rejected 答案要求】
1. 看似专业但包含致命错误（如公式推导错误、概念张冠李戴）
2. 不能是明显的胡言乱语（要让人第一眼看不出问题）
3. 错误类型：{error_type}

问题：{question}
"""
```

#### Day 5-6：批量生成数据（自动化流程）

**数据生成脚本**（`generate_preference_data.py`）：

```python
import openai
import json
from tqdm import tqdm

# 问题库（手动精选 50 个高质量问题）
questions = [
    "解释 Q-learning 和 SARSA 的本质区别",
    "为什么 PID 控制器在积分饱和时会失效？",
    "MPC 如何处理约束优化问题？",
    "PPO 算法中 clip 操作的作用是什么？",
    "如何为四旋翼无人机设计轨迹跟踪控制器？",
    # ... 补充到 50 个
]

error_types = ["公式推导错误", "概念混淆", "适用场景错误"]

preference_data = []

for q in tqdm(questions):
    for error_type in error_types:
        # 调用 API 生成 chosen 和 rejected
        response = openai.ChatCompletion.create(
            model="gpt-4o",
            messages=[{
                "role": "system",
                "content": SYSTEM_PROMPT.format(
                    question=q,
                    error_type=error_type
                )
            }]
        )

        # 解析返回结果
        result = parse_response(response)
        preference_data.append({
            "prompt": q,
            "chosen": result["chosen"],
            "rejected": result["rejected"]
        })

# 保存为 JSONL 格式
with open("preference_data.jsonl", "w") as f:
    for item in preference_data:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")
```

**目标产出**：

- 150 对偏好数据（50 问题 × 3 错误类型）
- 每对数据包含：`{"prompt": "...", "chosen": "...", "rejected": "..."}`

#### Day 7：数据质量验证（人工抽检）

**验证流程**：

1. 随机抽取 20 对数据
2. 检查 rejected 答案是否真的有问题（让 GPT-4o 当裁判）
3. 如果 rejected 答案被判定为正确，说明 prompt 设计有问题，需要迭代

**验证脚本**：

```python
# 让 GPT-4o 评估哪个答案更好
judge_prompt = f"""
问题：{prompt}

答案 A：{chosen}

答案 B：{rejected}

请判断哪个答案更专业准确，并说明理由。
"""
```

#### 交付物

- `preference_data.jsonl`：150 对偏好数据
- `data_quality_report.md`：包含抽检结果、bad case 分析

---

## 第二周：DPO 训练与效果验证（Day 8-14）

### Day 8-9：DPO 训练链路打通

**训练脚本**（基于 trl 库）：

```python
from trl import DPOTrainer, DPOConfig
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载 SFT 模型
sft_model = AutoModelForCausalLM.from_pretrained("checkpoints/sft_baseline")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# DPO 配置
config = DPOConfig(
    beta=0.1,  # DPO 的温度参数
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_steps=300,
    logging_steps=10,
    save_steps=100,
    output_dir="checkpoints/dpo_model",
)

# 加载偏好数据
from datasets import load_dataset
preference_dataset = load_dataset("json", data_files="preference_data.jsonl")

# 训练
trainer = DPOTrainer(
    model=sft_model,
    ref_model=sft_model,  # 参考模型（冻结）
    train_dataset=preference_dataset["train"],
    tokenizer=tokenizer,
    args=config,
)

trainer.train()
```

**关键指标监控**：

- `rewards/chosen`：应该持续上升
- `rewards/rejected`：应该持续下降
- `rewards/margins`：两者差距应该拉大（目标 > 0.5）

**预计训练时间**：2 小时（单卡 4090）

#### 交付物

- `train_dpo.py`：DPO 训练脚本
- `checkpoints/dpo_model/`：DPO 模型权重
- `dpo_training_log.txt`：训练日志

---

### Day 10-11：LLM-as-a-Judge 评估

**评估脚本**（`eval_dpo.py`）：

```python
import openai
from transformers import AutoModelForCausalLM, AutoTokenizer

# 加载模型
sft_model = AutoModelForCausalLM.from_pretrained("checkpoints/sft_baseline")
dpo_model = AutoModelForCausalLM.from_pretrained("checkpoints/dpo_model")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-1.5B-Instruct")

# 准备 50 个测试问题（不在训练集中）
test_questions = [
    "解释 Actor-Critic 算法的核心思想",
    "如何为倒立摆系统设计 LQR 控制器？",
    # ... 补充到 50 个
]

# 生成答案
sft_answers = []
dpo_answers = []

for q in test_questions:
    inputs = tokenizer(q, return_tensors="pt")

    sft_output = sft_model.generate(**inputs, max_length=512)
    sft_ans = tokenizer.decode(sft_output[0])
    sft_answers.append(sft_ans)

    dpo_output = dpo_model.generate(**inputs, max_length=512)
    dpo_ans = tokenizer.decode(dpo_output[0])
    dpo_answers.append(dpo_ans)

# GPT-4o 裁判
win_count = 0
total_count = len(test_questions)

for q, sft_ans, dpo_ans in zip(test_questions, sft_answers, dpo_answers):
    judge_prompt = f"""
    请评估以下两个答案的质量，重点关注：
    1. 专业准确性
    2. 是否存在幻觉或错误
    3. 逻辑严密性

    问题：{q}

    答案 A（SFT 模型）：{sft_ans}

    答案 B（DPO 模型）：{dpo_ans}

    请选择更好的答案（A 或 B），并说明理由。
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    result = response.choices[0].message.content
    if "B" in result or "答案 B" in result:
        win_count += 1

win_rate = win_count / total_count
print(f"DPO 模型胜率：{win_rate:.2%}")
```

**目标指标**：

- DPO 模型在"专业准确性"维度的胜率 > 65%
- 在"避免幻觉"维度的胜率 > 70%

#### 交付物

- `eval_dpo.py`：评估脚本
- `eval_results.json`：详细对比数据
- `win_rate_chart.png`：可视化图表

---

### Day 12-13：Kaggle 经验迁移与超参数优化

#### 任务清单

**1. 阅读 Kaggle LMSYS 比赛 Top 5 方案**（重点关注）：

- 数据清洗策略（如何过滤低质量偏好对）
- LoRA 参数选择（r=16 vs r=32 的权衡）
- 训练稳定性技巧（梯度裁剪、warmup 步数）

**2. 实验至少 3 组超参数**：

```python
experiments = [
    {
        "name": "baseline",
        "beta": 0.1,
        "lora_r": 8,
        "warmup_steps": 0
    },
    {
        "name": "higher_beta",
        "beta": 0.2,
        "lora_r": 16,
        "warmup_steps": 0
    },
    {
        "name": "with_warmup",
        "beta": 0.15,
        "lora_r": 16,
        "warmup_steps": 50
    },
]

for exp in experiments:
    print(f"Running experiment: {exp['name']}")
    # 训练模型
    # 评估 win rate
    # 记录结果
```

**3. 记录每组实验的 win rate 变化**

#### 交付物

- `hyperparameter_tuning.md`：实验记录
- `checkpoints/best_model/`：最优模型的 checkpoint

---

### Day 14：项目打包与简历材料准备

#### GitHub README 结构

```markdown
# 机器人控制领域的 LLM 偏好对齐

## 🎯 核心亮点

- 自动化生成 150 对领域专属偏好数据（覆盖强化学习、PID 控制、MPC 等）
- 使用 DPO 算法将模型在专业准确性上的表现提升 XX%
- 通过 LLM-as-a-Judge 验证，DPO 模型在消除"控制理论幻觉"上的胜率达 70%

## 🛠️ 技术栈

- **模型**：Qwen2.5-1.5B + LoRA (r=16)
- **算法**：SFT → DPO
- **评估**：GPT-4o 作为裁判
- **框架**：PyTorch, Transformers, PEFT, TRL

## 📊 实验结果

[插入 win_rate_chart.png]

| 指标 | SFT 基线 | DPO 模型 | 提升 |
|------|---------|---------|------|
| 专业准确性胜率 | - | XX% | +XX% |
| 避免幻觉胜率 | - | XX% | +XX% |
| Reward Margin | - | 0.XX | - |

## 🚀 复现步骤

### 1. 环境安装
```bash
pip install -r requirements.txt
```

### 2. 数据生成

```bash
python generate_preference_data.py
```

### 3. SFT 训练

```bash
export CUDA_VISIBLE_DEVICES=0
python sft_baseline.py
```

### 4. DPO 训练

```bash
python train_dpo.py
```

### 5. 评估

```bash
python eval_dpo.py
```

## 📁 项目结构

```
.
├── generate_preference_data.py  # 偏好数据生成
├── sft_baseline.py              # SFT 训练脚本
├── train_dpo.py                 # DPO 训练脚本
├── eval_dpo.py                  # 评估脚本
├── preference_data.jsonl        # 偏好数据集
├── checkpoints/
│   ├── sft_baseline/            # SFT 模型
│   └── dpo_model/               # DPO 模型
└── results/
    ├── eval_results.json
    └── win_rate_chart.png
```

## 💡 关键技术细节

### 偏好数据生成策略

- 使用 GPT-4o 生成高质量的 chosen/rejected 对
- 设计三种错误类型：公式推导错误、概念混淆、适用场景错误
- 通过 LLM 裁判进行数据质量验证

### DPO 训练技巧

- Beta 参数设置为 0.15（经过超参数搜索）
- 使用 LoRA (r=16) 进行高效微调
- 添加 50 步 warmup 提升训练稳定性

## 📚 参考资料

- [DPO 论文](https://arxiv.org/abs/2305.18290)
- [Kaggle L](https://www.kaggle.com/competitions/lmsys-chatbot-arena)
- [MSYS 比赛](https://www.kaggle.com/competitions/lmsys-chatbot-arena)
- [TRL 文档](https://huggingface.co/docs/trl)

```

#### 简历描述模板

```

【大模型偏好对齐项目】2026.03

- 针对机器人控制领域，设计自动化流程生成 150 对高质量偏好数据，覆盖强化学习（Q-learning, PPO, SAC）、PID 控制、MPC 等核心算法
- 使用 DPO 算法在 Qwen2.5-1.5B 上进行偏好对齐，通过 GPT-4o 裁判验证，模型在专业准确性上的胜率提升 XX%，在消除"控制理论幻觉"上的胜率达 70%
- 参考 Kaggle LMSYS 比赛 Top 方案，优化 LoRA 超参数（r=16, alpha=32）和训练策略（warmup + gradient clipping），最终模型 Reward Margin 达到 0.XX
- 技术栈：PyTorch, Transformers, PEFT, TRL, Weights & Biases

```

#### 交付物
- `README.md`：完整的项目文档
- `RESUME_DESCRIPTION.md`：简历描述模板
- 所有代码和数据整理到 GitHub 仓库

---

## ⚠️ 关键风险与应对策略

| 风险 | 应对策略 |
|------|---------|
| **API 调用成本过高** | 先用 50 个问题测试，验证 prompt 质量后再批量生成 |
| **DPO 训练不收敛** | 降低 beta 参数（从 0.1 降到 0.05），增加 warmup 步数 |
| **评估结果不理想** | 回到 Day 7 检查数据质量，可能需要重新设计 rejected 答案的生成逻辑 |
| **GPU 资源被占用** | 提前和同事协调，或者使用 GPU 5（完全空闲） |
| **模型过拟合** | 增加训练数据量，或者使用更小的 LoRA rank |
| **生成的 rejected 答案质量不够** | 迭代 System Prompt，增加更多约束条件 |

---

## 📈 进度追踪

### Week 1
- [ ] Day 1: 环境验证
- [ ] Day 2-3: SFT 基线
- [ ] Day 4: 数据生成策略设计
- [ ] Day 5-6: 批量生成偏好数据
- [ ] Day 7: 数据质量验证

### Week 2
- [ ] Day 8-9: DPO 训练
- [ ] Day 10-11: LLM-as-a-Judge 评估
- [ ] Day 12-13: 超参数优化
- [ ] Day 14: 项目打包

---

## 🎓 学习资源

### 必读论文
1. **DPO**: Direct Preference Optimization (Rafailov et al., 2023)
2. **LoRA**: Low-Rank Adaptation of Large Language Models (Hu et al., 2021)
3. **RLHF**: Training language models to follow instructions with human feedback (Ouyang et al., 2022)

### 推荐教程
- [Hugging Face TRL 文档](https://huggingface.co/docs/trl)
- [Kaggle LMSYS 比赛讨论区](https://www.kaggle.com/competitions/lmsys-chatbot-arena/discussion)
- [强化学习圣经：Sutton & Barto](http://incompleteideas.net/book/the-book-2nd.html)

---

## 联系方式

如有问题，欢迎通过以下方式联系：
- GitHub Issues
- Email: [your-email]

---

**最后更新**：2026-03-11
```

