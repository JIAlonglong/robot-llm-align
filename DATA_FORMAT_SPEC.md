# 数据格式与接口规范

> 本文档定义项目中所有数据格式、API 接口、文件命名规范

---

## 1. 数据集格式规范

### 1.1 SFT 数据格式（JSONL）

**文件命名**：`dataset/sft_{source}_{count}.jsonl`
- 示例：`sft_general_300.jsonl`、`sft_robotics_100.jsonl`

**数据结构**：
```json
{
  "id": "sft_001",
  "source": "ultrachat",
  "conversations": [
    {
      "role": "system",
      "content": "你是一个机器人控制领域的专家，擅长强化学习、PID 控制、MPC 等技术。"
    },
    {
      "role": "user",
      "content": "解释 Q-learning 的 off-policy 特性"
    },
    {
      "role": "assistant",
      "content": "Q-learning 是一种 off-policy 时序差分控制算法..."
    }
  ],
  "metadata": {
    "topic": "reinforcement_learning",
    "difficulty": "medium",
    "has_formula": true,
    "created_at": "2026-03-11"
  }
}
```

**字段说明**：
- `id`：唯一标识符（必需）
- `source`：数据来源（`ultrachat`、`textbook`、`manual`）
- `conversations`：对话列表（必需）
  - `role`：角色（`system`、`user`、`assistant`）
  - `content`：内容
- `metadata`：元数据（可选）
  - `topic`：主题标签（`reinforcement_learning`、`pid_control`、`mpc`）
  - `difficulty`：难度（`easy`、`medium`、`hard`）
  - `has_formula`：是否包含数学公式
  - `created_at`：创建日期

### 1.2 DPO 偏好数据格式（JSONL）

**文件命名**：`dataset/dpo_{topic}_{count}.jsonl`
- 示例：`dpo_robotics_100.jsonl`

**数据结构**：
```json
{
  "id": "dpo_001",
  "prompt": "解释 Q-learning 和 SARSA 的本质区别",
  "chosen": [
    {
      "role": "user",
      "content": "解释 Q-learning 和 SARSA 的本质区别"
    },
    {
      "role": "assistant",
      "content": "Q-learning 是 off-policy 算法，SARSA 是 on-policy 算法。\n\n核心区别：\n1. Q-learning 使用 max Q(s',a') 更新（贪婪策略）\n2. SARSA 使用实际执行的 Q(s',a') 更新（当前策略）\n\n数学公式：\nQ-learning: Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]\nSARSA: Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]"
    }
  ],
  "rejected": [
    {
      "role": "user",
      "content": "解释 Q-learning 和 SARSA 的本质区别"
    },
    {
      "role": "assistant",
      "content": "Q-learning 是 on-policy 算法，SARSA 是 off-policy 算法。两者的更新公式基本相同。"
    }
  ],
  "metadata": {
    "error_type": "concept_confusion",
    "topic": "reinforcement_learning",
    "created_at": "2026-03-11"
  }
}
```

**字段说明**：
- `id`：唯一标识符（必需）
- `prompt`：用户问题（必需）
- `chosen`：正确答案的对话（必需）
- `rejected`：错误答案的对话（必需）
- `metadata`：元数据（可选）
  - `error_type`：错误类型
    - `concept_confusion`：概念混淆
    - `formula_error`：公式错误
    - `wrong_scenario`：适用场景错误
  - `topic`：主题标签
  - `created_at`：创建日期

---

## 2. 模型配置规范

### 2.1 训练配置文件（YAML）

**文件路径**：`configs/train_config.yaml`

```yaml
# 模型配置
model:
  name: "Qwen/Qwen2.5-1.5B-Instruct"
  device_map: "auto"
  torch_dtype: "bfloat16"

# LoRA 配置
lora:
  r: 8
  lora_alpha: 16
  target_modules:
    - "q_proj"
    - "v_proj"
    - "k_proj"
    - "o_proj"
  lora_dropout: 0.05
  bias: "none"
  task_type: "CAUSAL_LM"

# 训练参数
training:
  output_dir: "checkpoints/sft_baseline"
  per_device_train_batch_size: 4
  gradient_accumulation_steps: 4
  learning_rate: 5e-5
  num_train_epochs: 3
  max_seq_length: 512
  logging_steps: 10
  save_steps: 100
  warmup_steps: 50
  fp16: false
  bf16: true

# 数据配置
data:
  train_file: "dataset/sft_combined.jsonl"
  max_samples: null  # null 表示使用全部数据
  shuffle: true
  seed: 42

# GPU 配置
gpu:
  visible_devices: "3"  # 使用 GPU 3
```

### 2.2 DPO 训练配置

**文件路径**：`configs/dpo_config.yaml`

```yaml
# 继承 SFT 配置
base_config: "configs/train_config.yaml"

# DPO 特定参数
dpo:
  beta: 0.1  # KL 惩罚系数
  reference_model: "checkpoints/sft_baseline"  # 参考模型路径
  max_prompt_length: 256
  max_length: 512

# 训练参数（覆盖基础配置）
training:
  output_dir: "checkpoints/dpo_model"
  per_device_train_batch_size: 2
  gradient_accumulation_steps: 8
  max_steps: 300
  learning_rate: 5e-5

# 数据配置
data:
  train_file: "dataset/dpo_robotics_100.jsonl"
```

---

## 3. 脚本接口规范

### 3.1 训练脚本接口

**文件路径**：`scripts/train_sft.py`

**命令行参数**：
```bash
python scripts/train_sft.py \
  --config configs/train_config.yaml \
  --data_file dataset/sft_combined.jsonl \
  --output_dir checkpoints/sft_baseline \
  --gpu_id 3 \
  --resume_from_checkpoint checkpoints/sft_baseline/checkpoint-100  # 可选
```

**参数说明**：
- `--config`：配置文件路径（必需）
- `--data_file`：训练数据路径（可选，覆盖配置文件）
- `--output_dir`：输出目录（可选，覆盖配置文件）
- `--gpu_id`：GPU ID（可选，覆盖配置文件）
- `--resume_from_checkpoint`：从检查点恢复（可选）

**返回值**：
- 成功：退出码 0，输出训练日志
- 失败：退出码 1，输出错误信息

### 3.2 评估脚本接口

**文件路径**：`scripts/eval_dpo.py`

**命令行参数**：
```bash
python scripts/eval_dpo.py \
  --sft_model checkpoints/sft_baseline \
  --dpo_model checkpoints/dpo_model \
  --test_file dataset/test_questions.jsonl \
  --output_file results/eval_results.json \
  --judge_model gpt-4o
```

**参数说明**：
- `--sft_model`：SFT 模型路径（必需）
- `--dpo_model`：DPO 模型路径（必需）
- `--test_file`：测试问题文件（必需）
- `--output_file`：结果输出文件（必需）
- `--judge_model`：裁判模型（`gpt-4o`、`claude-3-opus`）

**输出格式**（JSON）：
```json
{
  "metadata": {
    "sft_model": "checkpoints/sft_baseline",
    "dpo_model": "checkpoints/dpo_model",
    "test_samples": 50,
    "judge_model": "gpt-4o",
    "evaluated_at": "2026-03-15T10:30:00"
  },
  "overall_metrics": {
    "dpo_win_rate": 0.72,
    "sft_win_rate": 0.18,
    "tie_rate": 0.10
  },
  "dimension_metrics": {
    "accuracy": {
      "dpo_win_rate": 0.75,
      "sft_win_rate": 0.15
    },
    "hallucination_avoidance": {
      "dpo_win_rate": 0.80,
      "sft_win_rate": 0.12
    },
    "clarity": {
      "dpo_win_rate": 0.65,
      "sft_win_rate": 0.25
    }
  },
  "detailed_results": [
    {
      "question_id": "test_001",
      "question": "解释 Q-learning 的 off-policy 特性",
      "sft_answer": "...",
      "dpo_answer": "...",
      "winner": "dpo",
      "judge_reasoning": "DPO 模型的回答更准确，包含了数学公式和实际应用场景"
    }
  ]
}
```

---

## 4. 文件命名规范

### 4.1 数据集文件
```
dataset/
├── sft_general_300.jsonl          # 通用 SFT 数据
├── sft_robotics_100.jsonl         # 领域 SFT 数据
├── sft_combined.jsonl             # 合并后的 SFT 数据
├── dpo_robotics_100.jsonl         # DPO 偏好数据
├── test_questions.jsonl           # 测试问题
└── README.md                      # 数据集说明
```

### 4.2 模型权重文件
```
checkpoints/
├── sft_baseline/                  # SFT 基线模型
│   ├── adapter_config.json
│   ├── adapter_model.bin
│   └── checkpoint-100/            # 训练检查点
├── dpo_model/                     # DPO 模型
│   ├── adapter_config.json
│   └── adapter_model.bin
└── README.md                      # 模型说明
```

### 4.3 结果文件
```
results/
├── eval_results.json              # 评估结果
├── win_rate_chart.png             # 胜率图表
├── training_curves.png            # 训练曲线
└── README.md                      # 结果说明
```

---

## 5. 日志格式规范

### 5.1 训练日志格式

**文件路径**：`logs/train_sft_20260311.log`

**格式**：
```
[2026-03-11 10:30:00] [INFO] Starting SFT training
[2026-03-11 10:30:05] [INFO] Model loaded: Qwen/Qwen2.5-1.5B-Instruct
[2026-03-11 10:30:10] [INFO] Dataset loaded: 400 samples
[2026-03-11 10:30:15] [INFO] Training started
[2026-03-11 10:30:25] [TRAIN] Step 10/500 | Loss: 2.345 | LR: 5e-5
[2026-03-11 10:30:35] [TRAIN] Step 20/500 | Loss: 2.123 | LR: 5e-5
...
[2026-03-11 12:30:00] [INFO] Training completed
[2026-03-11 12:30:05] [INFO] Model saved to: checkpoints/sft_baseline
```

### 5.2 评估日志格式

**文件路径**：`logs/eval_dpo_20260315.log`

**格式**：
```
[2026-03-15 10:30:00] [INFO] Starting evaluation
[2026-03-15 10:30:05] [INFO] SFT model loaded: checkpoints/sft_baseline
[2026-03-15 10:30:10] [INFO] DPO model loaded: checkpoints/dpo_model
[2026-03-15 10:30:15] [INFO] Test questions loaded: 50 samples
[2026-03-15 10:30:20] [EVAL] Question 1/50 | Winner: dpo
[2026-03-15 10:30:25] [EVAL] Question 2/50 | Winner: sft
...
[2026-03-15 11:00:00] [INFO] Evaluation completed
[2026-03-15 11:00:05] [INFO] DPO win rate: 72%
[2026-03-15 11:00:10] [INFO] Results saved to: results/eval_results.json
```

---

## 6. API 接口规范（可选）

### 6.1 模型推理 API

**端点**：`POST /v1/chat/completions`

**请求格式**：
```json
{
  "model": "robot-control-agent",
  "messages": [
    {
      "role": "system",
      "content": "你是一个机器人控制领域的专家"
    },
    {
      "role": "user",
      "content": "解释 Q-learning 的 off-policy 特性"
    }
  ],
  "temperature": 0.7,
  "max_tokens": 512
}
```

**响应格式**：
```json
{
  "id": "chatcmpl-123",
  "object": "chat.completion",
  "created": 1710144000,
  "model": "robot-control-agent",
  "choices": [
    {
      "index": 0,
      "message": {
        "role": "assistant",
        "content": "Q-learning 是一种 off-policy 时序差分控制算法..."
      },
      "finish_reason": "stop"
    }
  ],
  "usage": {
    "prompt_tokens": 50,
    "completion_tokens": 200,
    "total_tokens": 250
  }
}
```

---

## 7. 版本控制规范

### 7.1 Git 提交信息格式

**格式**：`<type>(<scope>): <subject>`

**类型（type）**：
- `feat`：新功能
- `fix`：修复 bug
- `data`：数据相关
- `train`：训练相关
- `eval`：评估相关
- `docs`：文档更新
- `refactor`：代码重构
- `test`：测试相关

**范围（scope）**：
- `sft`：SFT 训练
- `dpo`：DPO 训练
- `data`：数据处理
- `eval`：评估
- `config`：配置
- `scripts`：脚本

**示例**：
```bash
git commit -m "feat(sft): 添加 SFT 训练脚本"
git commit -m "data(robotics): 添加 100 条机器人控制领域数据"
git commit -m "fix(dpo): 修复 DPO 训练中的 loss 计算错误"
git commit -m "eval(metrics): 添加胜率统计功能"
```

### 7.2 版本标签规范

**格式**：`v<major>.<minor>.<patch>`

**示例**：
- `v0.1.0`：初始版本（SFT 基线）
- `v0.2.0`：添加 DPO 训练
- `v0.2.1`：修复 DPO 训练 bug
- `v1.0.0`：完整的基础版本

---

## 8. 错误处理规范

### 8.1 错误代码

| 错误代码 | 说明 | 处理方式 |
|---------|------|---------|
| `E001` | 数据文件不存在 | 检查文件路径 |
| `E002` | 数据格式错误 | 检查 JSON 格式 |
| `E003` | 模型加载失败 | 检查模型路径和权限 |
| `E004` | GPU 内存不足 | 减小 batch size |
| `E005` | 训练中断 | 从检查点恢复 |
| `E006` | 评估失败 | 检查测试数据 |

### 8.2 错误日志格式

```
[2026-03-11 10:30:00] [ERROR] E001: Data file not found
[2026-03-11 10:30:00] [ERROR] File path: dataset/sft_combined.jsonl
[2026-03-11 10:30:00] [ERROR] Traceback:
  File "scripts/train_sft.py", line 50, in load_dataset
    with open(data_file, 'r') as f:
FileNotFoundError: [Errno 2] No such file or directory: 'dataset/sft_combined.jsonl'
```

---

## 9. 测试数据格式

### 9.1 测试问题文件

**文件路径**：`dataset/test_questions.jsonl`

**格式**：
```json
{
  "id": "test_001",
  "question": "解释 Q-learning 的 off-policy 特性",
  "topic": "reinforcement_learning",
  "difficulty": "medium",
  "expected_keywords": ["off-policy", "max", "贪婪策略"],
  "expected_formula": "Q(s,a) ← Q(s,a) + α[r + γ max_a' Q(s',a') - Q(s,a)]"
}
```

---

## 10. 配置文件优先级

当多个配置源存在时，优先级从高到低：
1. 命令行参数
2. 环境变量
3. 配置文件（YAML）
4. 默认值

**示例**：
```bash
# 配置文件中 gpu_id=3
# 命令行参数 --gpu_id 5
# 最终使用 GPU 5
python scripts/train_sft.py --config configs/train_config.yaml --gpu_id 5
```

---

## 附录：完整示例

### A.1 完整的 SFT 数据示例

```json
{
  "id": "sft_rl_001",
  "source": "textbook",
  "conversations": [
    {
      "role": "system",
      "content": "你是一个机器人控制领域的专家，擅长强化学习、PID 控制、MPC 等技术。请用专业、准确的语言回答问题，必要时包含数学公式。"
    },
    {
      "role": "user",
      "content": "解释 Q-learning 的 off-policy 特性，并说明它与 SARSA 的区别"
    },
    {
      "role": "assistant",
      "content": "Q-learning 是一种 off-policy 时序差分控制算法。\n\n**Off-policy 的含义**：\n行为策略（用于探索环境）和目标策略（用于更新价值函数）可以不同。\n\n**更新公式**：\nQ(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]\n\n**与 SARSA 的区别**：\n1. Q-learning 使用 max 操作选择下一状态的最优动作（贪婪策略）\n2. SARSA 使用实际执行的动作（当前策略）\n3. Q-learning 收敛到最优策略，SARSA 收敛到当前策略的价值\n\n**应用场景**：\n- Q-learning 适用于需要找到最优策略的场景（如机器人导航）\n- SARSA 适用于需要考虑探索风险的场景（如悬崖行走问题）"
    }
  ],
  "metadata": {
    "topic": "reinforcement_learning",
    "difficulty": "medium",
    "has_formula": true,
    "keywords": ["q-learning", "off-policy", "sarsa"],
    "created_at": "2026-03-11",
    "verified": true
  }
}
```

### A.2 完整的 DPO 数据示例

```json
{
  "id": "dpo_rl_001",
  "prompt": "解释 Q-learning 和 SARSA 的本质区别",
  "chosen": [
    {
      "role": "user",
      "content": "解释 Q-learning 和 SARSA 的本质区别"
    },
    {
      "role": "assistant",
      "content": "Q-learning 和 SARSA 的本质区别在于 **on-policy vs off-policy**：\n\n**Q-learning（off-policy）**：\n- 更新公式：Q(s,a) ← Q(s,a) + α[r + γ max_{a'} Q(s',a') - Q(s,a)]\n- 使用 max 操作选择下一状态的最优动作\n- 行为策略和目标策略可以不同\n- 收敛到最优策略\n\n**SARSA（on-policy）**：\n- 更新公式：Q(s,a) ← Q(s,a) + α[r + γ Q(s',a') - Q(s,a)]\n- 使用实际执行的动作\n- 行为策略和目标策略相同\n- 收敛到当前策略的价值\n\n**实际影响**：\n在悬崖行走问题中，Q-learning 会学习走悬崖边缘（最优路径），而 SARSA 会学习远离悬崖（更安全的路径）。"
    }
  ],
  "rejected": [
    {
      "role": "user",
      "content": "解释 Q-learning 和 SARSA 的本质区别"
    },
    {
      "role": "assistant",
      "content": "Q-learning 是 on-policy 算法，SARSA 是 off-policy 算法。两者的更新公式基本相同，只是在选择下一个动作时有细微差别。在实际应用中，两者的性能差不多。"
    }
  ],
  "metadata": {
    "error_type": "concept_confusion",
    "error_description": "将 on-policy 和 off-policy 的定义说反了",
    "topic": "reinforcement_learning",
    "difficulty": "medium",
    "created_at": "2026-03-11",
    "verified": true
  }
}
```
