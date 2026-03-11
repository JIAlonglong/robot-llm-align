# Day 2-3 详细任务清单（自动化版本）

> 构建 SFT 数据集（预计 4-6 小时，使用 GPT-4o 自动生成）

---

## 核心变化：从手动提取 → 自动生成

**原方案问题**：
- 手动从教材提取 QA 对，耗时 12+ 小时
- 人工效率低，容易出错

**新方案优势**：
- 使用 GPT-4o 自动生成，1-2 小时完成
- 质量可控（通过 prompt 工程）
- 成本低（约 $5-7）
- 可复现性强

---

## 任务目标

构建高质量的 SFT 训练数据集，包含：
- 300 条通用指令数据（来自 HuggingFace）
- 100 条机器人控制领域专业数据（GPT-4o 自动生成）

---

## Day 2 任务（3 小时）

### Task 2.1：下载通用指令数据（30 分钟）

**目标**：获取 300 条高质量的通用对话数据

**步骤**：

```bash
cd /home/liujl/big_model/robot-llm-align

# 运行下载脚本（已在 Day 1 创建）
python scripts/data_processing/download_general_data.py
```

**验证标准**：
- [ ] 生成 `dataset/sft_general_300.jsonl`
- [ ] 文件大小约 500KB-1MB
- [ ] 随机抽查 3 条数据，格式正确

---

### Task 2.2：准备教材资源（30 分钟）

**目标**：下载机器人控制领域的经典教材

**推荐教材**：

1. **强化学习**（必需）
   - 《Reinforcement Learning: An Introduction》(Sutton & Barto, 2nd Edition)
   - 下载地址：http://incompleteideas.net/book/RLbook2020.pdf

2. **经典控制**（可选）
   - 任何 PID 控制教材
   - 或使用在线资源（Wikipedia、知乎）

3. **现代控制**（可选）
   - MPC 相关资料

**步骤**：
```bash
# 创建教材目录
mkdir -p textbooks

# 下载 Sutton & Barto RL 书
wget http://incompleteideas.net/book/RLbook2020.pdf -O textbooks/sutton_barto_rl.pdf

# 验证下载
ls -lh textbooks/sutton_barto_rl.pdf
```

**验证标准**：
- [ ] `textbooks/sutton_barto_rl.pdf` 存在
- [ ] 文件大小约 5-10 MB
- [ ] PDF 可以正常打开

---

### Task 2.3：配置 OpenAI API（15 分钟）

**目标**：设置 API Key，准备调用 GPT-4o

**步骤**：

```bash
# 方法 1：设置环境变量（临时）
export OPENAI_API_KEY="sk-your-api-key-here"

# 方法 2：写入配置文件（永久）
echo 'export OPENAI_API_KEY="sk-your-api-key-here"' >> ~/.bashrc
source ~/.bashrc

# 验证
echo $OPENAI_API_KEY
```

**测试 API 连接**：
```bash
python -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
response = openai.ChatCompletion.create(
    model='gpt-4o',
    messages=[{'role': 'user', 'content': 'Hello'}],
    max_tokens=10
)
print('API 连接成功:', response.choices[0].message.content)
"
```

**验证标准**：
- [ ] API Key 已设置
- [ ] 测试调用成功
- [ ] 账户余额充足（至少 $10）

---

### Task 2.4：自动生成强化学习 QA 对（1 小时）⭐ 核心任务

**目标**：使用 GPT-4o 从教材中自动生成 50 个高质量 QA 对

**创建配置文件**：`configs/qa_generation_rl.json`

```bash
mkdir -p configs
cat > configs/qa_generation_rl.json << 'EOFCONFIG'
{
  "chapters": [
    {
      "pdf_path": "textbooks/sutton_barto_rl.pdf",
      "sections": [
        {
          "name": "MDP Basics",
          "start_page": 47,
          "end_page": 65,
          "topic": "reinforcement_learning",
          "num_qa": 10,
          "focus": "MDP定义、状态转移、奖励函数、Bellman方程"
        },
        {
          "name": "Q-learning",
          "start_page": 130,
          "end_page": 145,
          "topic": "reinforcement_learning",
          "num_qa": 15,
          "focus": "off-policy特性、更新公式、收敛性证明"
        },
        {
          "name": "SARSA",
          "start_page": 145,
          "end_page": 155,
          "topic": "reinforcement_learning",
          "num_qa": 10,
          "focus": "on-policy特性、与Q-learning的区别"
        },
        {
          "name": "Policy Gradient",
          "start_page": 320,
          "end_page": 340,
          "topic": "reinforcement_learning",
          "num_qa": 15,
          "focus": "REINFORCE算法、Actor-Critic、优势函数"
        }
      ]
    }
  ],
  "model": "gpt-4o",
  "temperature": 0.3,
  "output_file": "dataset/sft_rl_auto_50.jsonl"
}
EOFCONFIG
```

**运行自动生成脚本**：
```bash
# 安装依赖（如果还没安装）
pip install pypdf openai

# 运行生成脚本
python scripts/data_processing/auto_generate_qa.py \
  --config configs/qa_generation_rl.json \
  --verify  # 自动验证质量
```

**脚本工作流程**：
1. 读取配置文件
2. 从 PDF 提取指定章节文本
3. 调用 GPT-4o API 生成 QA 对
4. 自动格式化为 JSONL
5. 可选：使用 GPT-4o 验证答案准确性（抽查 10%）

**预计时间**：
- 提取文本：5 分钟
- API 调用：30-40 分钟（50 个 QA 对）
- 质量验证：10 分钟
- 总计：约 1 小时

**成本估算**：
- 50 个 QA 对 × $0.08/对 ≈ $4-5
- 验证成本（可选）：$1-2
- 总计：约 $5-7

**验证标准**：
- [ ] 生成 `dataset/sft_rl_auto_50.jsonl`
- [ ] 包含 50 个 QA 对
- [ ] 每个答案包含：定义 + 公式 + 应用场景
- [ ] 答案长度：150-300 字
- [ ] 自动验证分数 >= 4 分的比例 > 90%

---

### Task 2.5：人工抽查与修正（30 分钟）

**目标**：人工审核自动生成的数据，确保质量

**抽查策略**：
- 随机抽取 10 个 QA 对（20%）
- 重点检查：公式正确性、概念准确性、逻辑严密性

**运行质量检查脚本**：
```bash
python scripts/data_processing/quality_check.py \
  --input dataset/sft_rl_auto_50.jsonl \
  --sample_rate 0.2 \
  --output logs/quality_check_rl.txt
```

**人工审核要点**：
1. ✅ 公式是否正确（LaTeX 格式）
2. ✅ 概念是否准确（不能说反）
3. ✅ 适用场景是否合理
4. ❌ 是否存在幻觉（编造的概念）

**修正方法**：
```bash
# 如果发现错误，重新生成有问题的 QA 对
python scripts/data_processing/auto_generate_qa.py \
  --regenerate \
  --ids "sft_rl_005,sft_rl_012"  # 指定需要重新生成的 ID
```

**验证标准**：
- [ ] 抽查 10 个 QA 对
- [ ] 错误率 < 10%
- [ ] 所有错误已修正
- [ ] 生成质量报告 `logs/quality_check_rl.txt`

---

## Day 3 任务（3 小时）

### Task 3.1：自动生成 PID 控制 QA 对（1 小时）

**目标**：生成 30 个 PID 控制相关的 QA 对

**方案 A：如果有 PID 教材 PDF**
```bash
# 创建配置文件
cat > configs/qa_generation_pid.json << 'EOFCONFIG'
{
  "chapters": [
    {
      "pdf_path": "textbooks/control_theory.pdf",
      "sections": [
        {
          "name": "PID Basics",
          "start_page": 50,
          "end_page": 70,
          "topic": "pid_control",
          "num_qa": 15,
          "focus": "比例、积分、微分作用，参数整定"
        },
        {
          "name": "PID Tuning",
          "start_page": 70,
          "end_page": 90,
          "topic": "pid_control",
          "num_qa": 15,
          "focus": "Ziegler-Nichols法、临界比例度法"
        }
      ]
    }
  ],
  "model": "gpt-4o",
  "temperature": 0.3,
  "output_file": "dataset/sft_pid_auto_30.jsonl"
}
EOFCONFIG

# 运行生成
python scripts/data_processing/auto_generate_qa.py \
  --config configs/qa_generation_pid.json
```

**方案 B：如果没有教材，使用 GPT-4o 直接生成**
```bash
# 创建直接生成脚本
python scripts/data_processing/generate_qa_from_topics.py \
  --topics "PID控制基础,PID参数整定,PID控制器设计,PID稳定性分析" \
  --num_per_topic 8 \
  --output dataset/sft_pid_auto_30.jsonl
```

**验证标准**：
- [ ] 生成 `dataset/sft_pid_auto_30.jsonl`
- [ ] 包含 30 个 QA 对
- [ ] 覆盖 PID 核心概念

---

### Task 3.2：自动生成 MPC 相关 QA 对（1 小时）

**目标**：生成 20 个 MPC 相关的 QA 对

**使用直接生成方案**（推荐）：
```bash
python scripts/data_processing/generate_qa_from_topics.py \
  --topics "MPC基础,滚动优化,约束处理,MPC稳定性" \
  --num_per_topic 5 \
  --output dataset/sft_mpc_auto_20.jsonl
```

**验证标准**：
- [ ] 生成 `dataset/sft_mpc_auto_20.jsonl`
- [ ] 包含 20 个 QA 对
- [ ] 覆盖 MPC 核心概念

---

### Task 3.3：合并所有数据集（30 分钟）

**目标**：将所有数据合并为一个完整的训练集

**创建合并脚本**：`scripts/data_processing/merge_datasets.py`

```python
#!/usr/bin/env python3
"""
合并所有 SFT 数据集
"""

import json
import os

def merge_datasets(input_files, output_file):
    """合并多个 JSONL 文件"""
    all_data = []
    
    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            continue
        
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                all_data.append(json.loads(line))
        
        print(f"✅ 读取 {file_path}: {len(all_data)} 条")
    
    # 保存合并结果
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    
    print(f"\n🎉 合并完成！共 {len(all_data)} 条数据")
    print(f"📁 保存到: {output_file}")

if __name__ == "__main__":
    merge_datasets(
        input_files=[
            "dataset/sft_general_300.jsonl",
            "dataset/sft_rl_auto_50.jsonl",
            "dataset/sft_pid_auto_30.jsonl",
            "dataset/sft_mpc_auto_20.jsonl",
        ],
        output_file="dataset/sft_combined.jsonl"
    )
```

**运行合并**：
```bash
python scripts/data_processing/merge_datasets.py
```

**验证标准**：
- [ ] 生成 `dataset/sft_combined.jsonl`
- [ ] 总数据量：300 + 50 + 30 + 20 = 400 条
- [ ] 文件大小约 1-2 MB

---

### Task 3.4：数据质量最终验证（30 分钟）

**目标**：全面检查数据质量

**运行完整验证**：
```bash
python scripts/data_processing/final_validation.py \
  --input dataset/sft_combined.jsonl \
  --output logs/final_validation_report.txt
```

**验证项目**：
1. ✅ 格式正确性（JSON 格式、必需字段）
2. ✅ 数据完整性（无空值、无重复）
3. ✅ 长度分布（答案长度 150-300 字）
4. ✅ 主题分布（通用 75%、领域 25%）
5. ✅ 难度分布（easy 30%、medium 50%、hard 20%）

**生成统计报告**：
```bash
# 查看报告
cat logs/final_validation_report.txt
```

**验证标准**：
- [ ] 所有格式检查通过
- [ ] 无重复数据
- [ ] 主题分布合理
- [ ] 生成最终报告

---

## 交付物清单

完成 Day 2-3 后，应该有以下文件：

```
robot-llm-align/
├── dataset/
│   ├── sft_general_300.jsonl          ✅ 通用数据（300 条）
│   ├── sft_rl_auto_50.jsonl           ✅ RL 数据（50 条）
│   ├── sft_pid_auto_30.jsonl          ✅ PID 数据（30 条）
│   ├── sft_mpc_auto_20.jsonl          ✅ MPC 数据（20 条）
│   └── sft_combined.jsonl             ✅ 合并数据（400 条）
├── textbooks/
│   └── sutton_barto_rl.pdf            ✅ RL 教材
├── configs/
│   ├── qa_generation_rl.json          ✅ RL 生成配置
│   ├── qa_generation_pid.json         ✅ PID 生成配置（可选）
├── logs/
│   ├── quality_check_rl.txt           ✅ RL 质量报告
│   └── final_validation_report.txt    ✅ 最终验证报告
└── scripts/data_processing/
    ├── auto_generate_qa.py            ✅ 自动生成脚本
    ├── generate_qa_from_topics.py     ✅ 主题生成脚本
    ├── quality_check.py               ✅ 质量检查脚本
    ├── merge_datasets.py              ✅ 合并脚本
    └── final_validation.py            ✅ 最终验证脚本
```

---

## 常见问题

### Q1: API 调用失败怎么办？
```bash
# 检查 API Key
echo $OPENAI_API_KEY

# 检查网络连接
curl https://api.openai.com/v1/models -H "Authorization: Bearer $OPENAI_API_KEY"

# 如果网络问题，使用代理
export https_proxy=http://your-proxy:port
```

### Q2: 生成的数据质量不好怎么办？
- 调整 prompt（在 `auto_generate_qa.py` 中）
- 降低 temperature（从 0.3 → 0.1）
- 增加验证步骤（使用 `--verify` 参数）

### Q3: 成本超预算怎么办？
- 减少生成数量（50 → 30）
- 使用 Claude-3.5-Sonnet（更便宜）
- 混合使用开源数据集

---

## 下一步

完成 Day 2-3 后，进入 **Day 4：SFT 训练**。

需要我帮你：
1. 创建 `generate_qa_from_topics.py` 脚本？
2. 创建质量检查脚本？
3. 或者直接开始运行 Task 2.1？
