#!/bin/bash
# 完整训练流程：SFT → DPO → 评估
# 预计总时长：2-3 小时

set -e  # 遇到错误立即退出

LOG_DIR="logs/training_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

echo "=========================================="
echo "开始完整训练流程"
echo "模型: Qwen2.5-1.5B-Instruct"
echo "日志目录: $LOG_DIR"
echo "=========================================="

# 1. 等待 DPO chosen 数据生成完成
echo "[1/6] 等待 DPO chosen 数据生成..."
while [ ! -f dataset/dpo_chosen_with_cot_tools.jsonl ] || [ $(wc -l < dataset/dpo_chosen_with_cot_tools.jsonl) -lt 470 ]; do
    COUNT=$(wc -l < dataset/dpo_chosen_with_cot_tools.jsonl 2>/dev/null || echo 0)
    echo "  当前进度: $COUNT/470"
    sleep 60
done
echo "✅ DPO chosen 数据生成完成"

# 2. SFT 训练
echo "[2/6] 开始 SFT 训练..."
if [ -f "checkpoints/sft_qwen1.5b/adapter_config.json" ]; then
    echo "✅ SFT checkpoint 已存在，跳过"
else
    /home/liujl/miniconda3/envs/LLM/bin/python scripts/train_sft_1.5b.py > "$LOG_DIR/sft.log" 2>&1
    echo "✅ SFT 训练完成"
fi

# 3. 生成 DPO rejected 数据
echo "[3/6] 生成 DPO rejected 数据..."
if [ -f "dataset/dpo_pairs.jsonl" ] && [ $(wc -l < dataset/dpo_pairs.jsonl) -ge 470 ]; then
    echo "✅ DPO pairs 已存在 ($(wc -l < dataset/dpo_pairs.jsonl) 条)，跳过"
else
    /home/liujl/miniconda3/envs/LLM/bin/python scripts/data_processing/generate_dpo_rejected.py > "$LOG_DIR/rejected.log" 2>&1
    echo "✅ DPO rejected 数据生成完成"
fi

# 4. DPO 训练
echo "[4/6] 开始 DPO 训练..."
/home/liujl/miniconda3/envs/LLM/bin/python scripts/train_dpo_1.5b.py > "$LOG_DIR/dpo.log" 2>&1
echo "✅ DPO 训练完成"

# 5. 推理对比测试
echo "[5/6] 运行推理对比测试..."
/home/liujl/miniconda3/envs/LLM/bin/python scripts/inference_compare.py > "$LOG_DIR/compare.log" 2>&1
echo "✅ 推理对比完成"

# 6. 生成报告
echo "[6/6] 生成训练报告..."
cat > "$LOG_DIR/summary.txt" <<EOF
训练完成时间: $(date)

模型: Qwen2.5-1.5B-Instruct
SFT 数据: 625 条
DPO 数据: 470 条

Checkpoints:
- SFT: checkpoints/sft_qwen1.5b/
- DPO: checkpoints/dpo_qwen1.5b/

详细日志:
- SFT: $LOG_DIR/sft.log
- DPO rejected: $LOG_DIR/rejected.log
- DPO: $LOG_DIR/dpo.log
- 对比测试: $LOG_DIR/compare.log
EOF

echo "=========================================="
echo "✅ 全部完成！"
echo "查看报告: cat $LOG_DIR/summary.txt"
echo "查看对比: cat $LOG_DIR/compare.log"
echo "=========================================="
