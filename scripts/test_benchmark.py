"""
快速测试 benchmark 指标（不跑完整流水线）
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

CKPT = "/home/liujl/big_model/robot-llm-align/checkpoints/dpo_pipeline_cycle9"
PROMPT_FILE = "/home/liujl/big_model/robot-llm-align/dataset/best_prompt.txt"

if os.path.exists(PROMPT_FILE):
    with open(PROMPT_FILE) as f:
        system_prompt = f.read().strip()
else:
    system_prompt = "你是机器人控制专家，可以调用工具完成任务。"

print(f"Checkpoint: {CKPT}")
print(f"System prompt: {system_prompt[:80]}...")
print("=" * 60)

# 导入 benchmark 函数
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from pipeline import phase_benchmark

result = phase_benchmark(CKPT, system_prompt, cycle=9)

print("\n" + "=" * 60)
print("Benchmark 结果")
print("=" * 60)
print(f"指令准确率: {result['instruction_acc']:.1%}")
print(f"幻觉率:     {result['hallucination_rate']:.1%}")
print(f"\n详细结果:")
for d in result["details"]:
    status = "✓" if d["pass"] else "✗"
    print(f"  {status} {d['type']:12s} | {d['input'][:50]:50s} | {d.get('reason', '')[:40]}")
