# 构建高质量的SFT数量集
# - 300 条通用指令数据（来自 HuggingFace）
# - 100 条机器人控制领域专业数据（从教材提取）

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
    