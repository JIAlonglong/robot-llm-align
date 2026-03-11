# 技术路线图（基于教材/论文提取数据）

## 核心策略调整

**原计划问题**：依赖 API 生成数据，成本高且质量不可控

**新方案优势**：
- 数据来源权威（教材 + 顶会论文）
- 逻辑严密，无幻觉
- 成本低（无 API 调用费用）
- 可复现性强

---

## 完整技术路线

### 第一周：数据构建与 SFT 训练

#### Day 1：环境验证（2 小时）
**任务**：
- 验证 GPU 资源（使用 GPU 3）
- 测试 Qwen2.5-1.5B 模型加载
- 运行 `scripts/env_check.py`

**交付物**：
- 环境检查通过的日志

---

#### Day 2-3：构建 SFT 数据集（核心）

**数据来源**：

1. **通用指令数据（300 条）**
   ```python
   from datasets import load_dataset
   general_data = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft[:300]")
   ```

2. **领域专业数据（500-800 条）- 从教材提取**

   **推荐教材**：
   - 《Reinforcement Learning: An Introduction》(Sutton & Barto, 2nd Edition)
     - 章节：Ch3-Ch6（MDP、动态规划、MC、TD、Q-learning、SARSA）
   - 《自动控制原理》（胡寿松）
     - 章节：PID 控制、状态空间、频域分析
   - 《Model Predictive Control》(Camacho & Bordons)
     - 章节：MPC 基础、约束优化

   **提取方法**：
   ```
   教材内容 → 关键概念/定理 → 改写为 QA 对

   示例：
   【教材原文】
   "Q-learning 是一种 off-policy TD 控制算法，更新公式为：
    Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]"

   【改写为 QA】
   Q: "解释 Q-learning 的 off-policy 特性及其更新公式"
   A: "Q-learning 是一种 off-policy 时序差分控制算法。
       所谓 off-policy，是指行为策略（用于探索）和目标策略（用于更新）可以不同。

       更新公式：
       Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]

       关键点：
       1. 使用 max 操作选择下一状态的最优动作（贪婪策略）
       2. 但实际执行时可以使用 ε-greedy（探索策略）
       3. 这种解耦使得 Q-learning 收敛更稳定

       应用场景：机器人导航、游戏 AI"
   ```

   **数据格式**（JSONL）：
   ```json
   {
     "conversations": [
       {"role": "system", "content": "你是一个机器人控制领域的专家"},
       {"role": "user", "content": "解释 Q-learning 的 off-policy 特性"},
       {"role": "assistant", "content": "[详细回答]"}
     ]
   }
   ```

3. **数据整理工具**
   ```bash
   # 创建数据整理脚本
   scripts/prepare_sft_data.py
   ```

**任务分解**：
- Day 2 上午：提取 RL 相关概念（50 个 QA）
- Day 2 下午：提取 PID/经典控制概念（30 个 QA）
- Day 3 上午：提取 MPC/现代控制概念（20 个 QA）
- Day 3 下午：合并数据 + 格式验证

**交付物**：
- `dataset/sft_general_300.jsonl`（通用数据）
- `dataset/sft_robotics_100.jsonl`（领域数据）
- `dataset/sft_combined.jsonl`（合并后的完整数据集）

---

#### Day 4：SFT 训练

**训练脚本**（`scripts/train_sft.py`）：
```python
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments
from trl import SFTTrainer
from peft import LoraConfig, get_peft_model

# 模型配置
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
tokenizer = AutoTokenizer.from_pretrained(model_name)

# LoRA 配置
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM"
)

# 训练参数
training_args = TrainingArguments(
    output_dir="checkpoints/sft_baseline",
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=5e-5,
    num_train_epochs=3,
    logging_steps=10,
    save_steps=100,
    warmup_steps=50,
)

# 训练
trainer = SFTTrainer(
    model=model,
    args=training_args,
    train_dataset=dataset,
    peft_config=lora_config,
)

trainer.train()
```

**预计训练时间**：2-3 小时（400 条数据，单卡 4090）

**交付物**：
- `checkpoints/sft_baseline/`（模型权重）
- `sft_training_log.txt`（训练日志）

---

#### Day 5-7：构建偏好数据（DPO 数据集）

**策略**：同样从教材提取，但构造 Chosen/Rejected 对

**Rejected 答案的构造方法**：

1. **公式推导错误**
   ```
   Chosen: Q(s,a) ← Q(s,a) + α[r + γ max Q(s',a') - Q(s,a)]
   Rejected: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]  # 缺少 max
   ```

2. **概念混淆**
   ```
   Chosen: "Q-learning 是 off-policy，SARSA 是 on-policy"
   Rejected: "Q-learning 是 on-policy，SARSA 是 off-policy"  # 说反了
   ```

3. **适用场景错误**
   ```
   Chosen: "PID 适用于线性系统，非线性系统建议用 MPC"
   Rejected: "PID 可以完美处理任何非线性系统"  # 过度泛化
   ```

**数据格式**（JSONL）：
```json
{
  "prompt": "解释 Q-learning 和 SARSA 的本质区别",
  "chosen": [
    {"role": "user", "content": "解释 Q-learning 和 SARSA 的本质区别"},
    {"role": "assistant", "content": "[正确答案：off-policy vs on-policy]"}
  ],
  "rejected": [
    {"role": "user", "content": "解释 Q-learning 和 SARSA 的本质区别"},
    {"role": "assistant", "content": "[错误答案：概念说反]"}
  ]
}
```

**任务分解**：
- Day 5：构造 50 对 RL 偏好数据
- Day 6：构造 30 对 PID/经典控制偏好数据
- Day 7：构造 20 对 MPC/现代控制偏好数据 + 质量验证

**质量验证**：
- 随机抽取 20 对数据
- 人工检查 Rejected 答案是否真的有问题
- 如果 Rejected 答案实际上是正确的，需要重新构造

**交付物**：
- `dataset/dpo_preference_100.jsonl`（100 对偏好数据）
- `dataset/data_quality_report.md`（质量验证报告）

---

### 第二周：DPO 训练与评估

#### Day 8-9：DPO 训练

**训练脚本**（`scripts/train_dpo.py`）：
```python
from trl import DPOTrainer, DPOConfig

config = DPOConfig(
    beta=0.1,  # KL 惩罚系数
    learning_rate=5e-5,
    per_device_train_batch_size=2,
    gradient_accumulation_steps=8,
    max_steps=300,
    logging_steps=10,
    output_dir="checkpoints/dpo_model",
)

trainer = DPOTrainer(
    model=sft_model,  # 加载 Day 4 训练的 SFT 模型
    ref_model=sft_model,  # 参考模型（冻结）
    train_dataset=preference_dataset,
    tokenizer=tokenizer,
    args=config,
)

trainer.train()
```

**关键指标监控**：
- `rewards/chosen`：应该持续上升
- `rewards/rejected`：应该持续下降
- `rewards/margins`：目标 > 0.5

**预计训练时间**：2 小时（100 对数据，单卡 4090）

**交付物**：
- `checkpoints/dpo_model/`（DPO 模型权重）
- `dpo_training_log.txt`（训练日志）

---

#### Day 10-11：LLM-as-a-Judge 评估

**评估脚本**（`scripts/eval_dpo.py`）：
```python
# 准备 50 个测试问题（不在训练集中）
test_questions = [
    "解释 Actor-Critic 算法的核心思想",
    "如何为倒立摆系统设计 LQR 控制器？",
    # ... 50 个
]

# 让 SFT 模型和 DPO 模型分别回答
sft_answers = [sft_model.generate(q) for q in test_questions]
dpo_answers = [dpo_model.generate(q) for q in test_questions]

# 调用 GPT-4o 裁判
for q, sft_ans, dpo_ans in zip(test_questions, sft_answers, dpo_answers):
    judge_prompt = f"""
    请评估以下两个答案的质量，重点关注：
    1. 专业准确性（公式是否正确）
    2. 逻辑严密性（推导是否合理）
    3. 是否存在幻觉（虚构的概念或错误的适用场景）

    问题：{q}
    答案 A：{sft_ans}
    答案 B：{dpo_ans}

    请选择更好的答案（A 或 B）并说明理由。
    """

    result = openai.ChatCompletion.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": judge_prompt}]
    )

    # 统计 DPO 胜率
```

**目标指标**：
- DPO 模型在"专业准确性"维度的胜率 > 65%
- 在"避免幻觉"维度的胜率 > 70%

**交付物**：
- `results/eval_results.json`（详细对比数据）
- `results/win_rate_chart.png`（可视化图表）

---

#### Day 12-13：超参数优化

**实验组**：
```python
experiments = [
    {"beta": 0.1, "lora_r": 8, "learning_rate": 5e-5},
    {"beta": 0.2, "lora_r": 16, "learning_rate": 3e-5},
    {"beta": 0.15, "lora_r": 16, "learning_rate": 5e-5, "warmup_steps": 50},
]
```

**参考 Kaggle LMSYS 比赛经验**：
- 数据清洗：过滤长度 < 50 或 > 2048 的样本
- LoRA 参数：r=16 通常优于 r=8（在小数据集上）
- 训练稳定性：使用 warmup + gradient clipping

**交付物**：
- `results/hyperparameter_tuning.md`（实验记录）
- 最优模型的 checkpoint

---

#### Day 14：项目打包

**GitHub README 更新**：
```markdown
## 实验结果

### SFT 训练
- 数据集：300 条通用 + 100 条领域专业数据
- 训练时长：2.5 小时（单卡 4090）
- 最终 Loss：0.XX

### DPO 训练
- 偏好数据：100 对（来自教材提取）
- Reward Margin：0.XX
- 训练时长：2 小时

### 评估结果
- DPO 模型在专业准确性上的胜率：XX%
- 在避免幻觉上的胜率：XX%

[插入 win_rate_chart.png]
```

**简历描述**：
```
【机器人控制领域的 LLM 偏好对齐】2026.03

- 从经典教材（Sutton & Barto RL 书、自动控制原理）提取 100 条高质量 QA 对，构建机器人控制领域的 SFT 数据集
- 设计偏好数据构造方法（公式错误、概念混淆、场景误用），生成 100 对 DPO 训练数据
- 使用 LoRA + DPO 算法在 Qwen2.5-1.5B 上进行偏好对齐，通过 GPT-4o 裁判验证，模型在专业准确性上的胜率提升 XX%
- 技术栈：PyTorch, Transformers, PEFT, TRL, Weights & Biases
```

---

## 关键优势

### 相比原计划的改进

| 维度 | 原计划（API 生成） | 新方案（教材提取） |
|------|-------------------|-------------------|
| 数据质量 | 依赖 prompt 质量，不可控 | 来自权威教材，准确性高 |
| 成本 | API 调用费用（150 对 × 3 次调用 ≈ $5-10） | 零成本 |
| 可复现性 | API 返回结果不稳定 | 完全可复现 |
| 学习价值 | 黑盒生成 | 深入理解领域知识 |
| 面试亮点 | "调用 API 生成数据" | "从经典教材提取知识，构建高质量数据集" |

### 技术亮点

1. **数据构建方法论**：从教材到 QA 对的系统化流程
2. **偏好数据设计**：三种错误类型（公式、概念、场景）
3. **完整训练流程**：SFT → DPO → 评估 → 优化
4. **可量化的效果**：LLM-as-a-Judge 胜率指标

---

## 风险与应对

| 风险 | 应对策略 |
|------|---------|
| 数据量不足（100 条太少） | 优先保证质量，后期可扩展到 200-300 条 |
| 人工提取效率低 | 使用 OCR + GPT-4o 辅助提取（但人工审核） |
| DPO 训练不收敛 | 降低 beta 参数，增加 warmup 步数 |
| 评估结果不理想 | 回到数据质量检查，可能需要重新构造 Rejected 答案 |

---

## 时间分配建议

**Week 1**：
- Day 1：2 小时（环境验证）
- Day 2-3：12 小时（数据提取，最耗时）
- Day 4：3 小时（SFT 训练）
- Day 5-7：10 小时（偏好数据构造）

**Week 2**：
- Day 8-9：4 小时（DPO 训练）
- Day 10-11：6 小时（评估）
- Day 12-13：6 小时（优化）
- Day 14：3 小时（打包）

**总计**：约 46 小时（平均每天 3.3 小时）

---

## 下一步行动

1. **立即开始**：运行 `scripts/env_check.py` 验证环境
2. **准备教材**：下载 Sutton & Barto RL 书 PDF
3. **创建数据模板**：设计 QA 对的标准格式
4. **开始提取**：从 Q-learning 章节开始（最经典）

需要我帮你生成 `scripts/prepare_sft_data.py`（数据整理脚本）吗？
