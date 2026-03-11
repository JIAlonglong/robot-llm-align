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

## 开发计划

- [ ] Week 1: 基建验证与领域数据构建
- [ ] Week 2: DPO 训练与效果验证

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
