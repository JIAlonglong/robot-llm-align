#!/bin/bash
# SFT训练启动脚本

# 设置使用的GPU（GPU 0 空闲）
export CUDA_VISIBLE_DEVICES=0

# 激活conda环境
source ~/miniconda3/etc/profile.d/conda.sh
conda activate LLM

# 运行训练
python scripts/train_sft.py 2>&1 | tee logs/train_sft_$(date +%Y%m%d_%H%M%S).log

echo "训练完成！"
