#!/usr/bin/env python3
"""
合并所有SFT数据集
"""

import json
import random
import os

def merge_datasets():
    """合并多个JSONL文件"""

    input_files = [
        "dataset/sft_general_300.jsonl",
        "dataset/sft_rl_deepseek_50.jsonl",
        "dataset/sft_rl_extended.jsonl",
        "dataset/sft_pid_extended.jsonl",
        "dataset/sft_mpc_extended.jsonl",
        "dataset/sft_multi_turn_50.jsonl",
    ]

    output_file = "dataset/sft_combined_v3.jsonl"

    all_data = []

    for file_path in input_files:
        if not os.path.exists(file_path):
            print(f"⚠️  文件不存在: {file_path}")
            continue

        count = 0
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                all_data.append(json.loads(line.strip()))
                count += 1

        print(f"✅ 读取 {file_path}: {count} 条")

    # 打乱顺序
    random.seed(42)
    random.shuffle(all_data)

    # 保存
    with open(output_file, "w", encoding="utf-8") as f:
        for item in all_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"\n🎉 合并完成！共 {len(all_data)} 条数据")
    print(f"📁 保存到: {output_file}")

    # 统计信息
    topics = {}
    sources = {}
    for item in all_data:
        topic = item.get("metadata", {}).get("topic", "unknown")
        source = item.get("source", "unknown")
        topics[topic] = topics.get(topic, 0) + 1
        sources[source] = sources.get(source, 0) + 1

    print(f"\n📊 主题分布:")
    for topic, count in sorted(topics.items()):
        print(f"  - {topic}: {count} 条")

    print(f"\n📊 来源分布:")
    for source, count in sorted(sources.items()):
        print(f"  - {source}: {count} 条")

    # 文件大小
    file_size = os.path.getsize(output_file) / 1024 / 1024
    print(f"\n💾 文件大小: {file_size:.2f} MB")


if __name__ == "__main__":
    merge_datasets()
