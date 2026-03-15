#!/usr/bin/env python3
"""
SFT（监督微调）训练脚本
使用 LoRA 对 Qwen2.5-7B-Instruct 进行参数高效微调

运行方式：
    python scripts/train_sft.py

依赖：
    pip install transformers peft trl datasets accelerate bitsandbytes wandb
"""
# 「加载→单条格式化→批量过滤→数据集封装→拆分→训练时动态补齐」
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import dataclass, field
from typing import List, Dict, Optional

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
)

# 设置 wandb API key
os.environ.setdefault("WANDB_API_KEY", "")  # set via env: export WANDB_API_KEY=...


# ============================================================
# 1. 配置区（超参数集中管理）
# ============================================================

@dataclass
class SFTConfig:
    # 模型
    model_name: str = "Qwen/Qwen2.5-7B-Instruct"

    # 数据
    data_path: str = "dataset/sft_combined_v2.jsonl"
    max_length: int = 2048           # 最大序列长度，超出截断
    val_ratio: float = 0.05          # 验证集比例

    # LoRA 参数
    lora_r: int = 16                 # LoRA 秩，从 8 增加到 16
    lora_alpha: int = 32             # LoRA 缩放系数，通常设为 2*r
    lora_dropout: float = 0.05
    # 只对注意力层的 q/v 矩阵做 LoRA（节省显存）
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # 训练参数
    output_dir: str = "checkpoints/sft_qwen2.5_7b_v2"
    num_epochs: int = 10
    per_device_train_batch_size: int = 1  # 减小到 1
    gradient_accumulation_steps: int = 16  # 增加到 16，保持等效 batch_size = 16
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    save_steps: int = 100
    logging_steps: int = 10
    fp16: bool = True                # 混合精度训练，节省显存
    gradient_checkpointing: bool = True  # 启用梯度检查点，节省显存


# ============================================================
# 2. 数据加载（已实现）
# ============================================================

def load_jsonl(path: str) -> List[Dict]:
    """读取 JSONL 格式的数据集"""
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"✅ 读取数据: {len(data)} 条 ({path})")
    return data


# ============================================================
# 3. 数据预处理（TODO：请你实现）
# ============================================================

def format_conversation(item: Dict, tokenizer, max_length: int) -> Optional[Dict]:
    """
    将一条对话数据转换为模型输入格式。
    只对 assistant 的回复部分计算 loss，其余位置设为 -100。
    """
    conversations = item.get("conversations", [])
    if not conversations:
        return None

    # 完整对话的 input_ids
    result = tokenizer.apply_chat_template(
        conversations,
        tokenize=True,
        add_generation_prompt=False,
    )
    # 提取 input_ids（BatchEncoding 有 input_ids 属性）
    if hasattr(result, "input_ids"):
        input_ids = result.input_ids
    elif isinstance(result, dict):
        input_ids = result["input_ids"]
    else:
        input_ids = result

    if len(input_ids) > max_length:
        return None

    # 构造 labels：全部初始化为 -100
    labels = [-100] * len(input_ids)

    # 逐步累积，找到每个 assistant 回复的 token 范围
    prev_len = 0
    for i in range(len(conversations)):
        # 编码到第 i 条消息为止的完整对话
        partial_result = tokenizer.apply_chat_template(
            conversations[:i + 1],
            tokenize=True,
            add_generation_prompt=False,
        )
        if hasattr(partial_result, "input_ids"):
            partial_ids = partial_result.input_ids
        elif isinstance(partial_result, dict):
            partial_ids = partial_result["input_ids"]
        else:
            partial_ids = partial_result

        cur_len = len(partial_ids)

        if conversations[i]["role"] == "assistant":
            # 当前 assistant 回复的 token 范围是 [prev_len, cur_len)
            labels[prev_len:cur_len] = input_ids[prev_len:cur_len]

        prev_len = cur_len

    attention_mask = [1] * len(input_ids)

    # Debug: 检查第一条数据
    if not hasattr(format_conversation, '_debug_printed'):
        print(f"DEBUG format_conversation: input_ids len={len(input_ids)}, labels len={len(labels)}, non-100 labels={sum(1 for x in labels if x != -100)}")
        format_conversation._debug_printed = True

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
    }


def build_dataset(data: List[Dict], tokenizer, max_length: int) -> Dataset:
    """
    批量处理所有数据，过滤掉太长的样本，返回 HuggingFace Dataset。
    """
    processed = []
    skipped = 0

    for item in data:
        result = format_conversation(item, tokenizer, max_length)
        if result is None:
            skipped += 1
        else:
            # 转换为 tensor（Dataset 会自动处理）
            processed.append({
                "input_ids": result["input_ids"],
                "attention_mask": result["attention_mask"],
                "labels": result["labels"],
            })

    print(f"📊 数据预处理完成：{len(processed)} 条保留，{skipped} 条因超长被过滤")
    return Dataset.from_list(processed)


# ============================================================
# 4. 模型加载（已实现）
# ============================================================

def load_model_and_tokenizer(config: SFTConfig):
    """加载基座模型和分词器，并挂载 LoRA 适配器"""

    print(f"📦 加载模型: {config.model_name}")

    # 加载分词器
    tokenizer = AutoTokenizer.from_pretrained(
        config.model_name,
        trust_remote_code=True,
        padding_side="right",   # 训练时右填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # 加载模型（fp16 节省显存）
    model = AutoModelForCausalLM.from_pretrained(
        config.model_name,
        dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # 挂载 LoRA 适配器
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)

    # 启用梯度检查点
    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()  # 打印可训练参数量

    return model, tokenizer


# ============================================================
# 5. 训练入口（已实现）
# ============================================================

def main():
    config = SFTConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    # 加载数据
    raw_data = load_jsonl(config.data_path)

    # 加载模型
    model, tokenizer = load_model_and_tokenizer(config)

    # 预处理数据（需要你实现上面两个函数后才能运行）
    dataset = build_dataset(raw_data, tokenizer, config.max_length)

    # 划分训练集/验证集
    split = dataset.train_test_split(test_size=config.val_ratio, seed=42)
    train_dataset = split["train"]
    eval_dataset = split["test"]
    print(f"📊 训练集: {len(train_dataset)} 条，验证集: {len(eval_dataset)} 条")

    # 训练参数
    training_args = TrainingArguments(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_steps=50,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=3,
        eval_strategy="no",           # 关闭 eval，避免 OOM 和中断
        load_best_model_at_end=False,  # 关闭，避免依赖 eval
        report_to="wandb",
        run_name="sft-qwen2.5-7b-robotics-v2",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        remove_unused_columns=False,   # 保留所有列，让自定义 collator 正常工作
    )

    # 数据整理器（手动 padding）
    pad_id = tokenizer.pad_token_id
    _debug_printed = [False]

    def data_collator(features):
        # features 是 list of dict，每个 dict 的 value 可能是嵌套 dict（accelerate 行为）
        # 提取真正的 input_ids 和 labels
        input_ids_list = []
        labels_list = []

        for f in features:
            ids = f["input_ids"]
            lbls = f["labels"]

            # 如果 ids 是 dict，说明是嵌套结构，取其中的 input_ids
            if isinstance(ids, dict):
                ids = ids["input_ids"]
            if isinstance(lbls, dict):
                lbls = lbls["labels"]

            input_ids_list.append(torch.tensor(ids))
            labels_list.append(torch.tensor(lbls))

        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids_list, batch_first=True, padding_value=pad_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels_list, batch_first=True, padding_value=-100)
        attention_mask = (input_ids != pad_id).long()

        if not _debug_printed[0]:
            print(f"DEBUG: batch_size={len(features)}, input_ids shape={input_ids.shape}, labels shape={labels.shape}")
            _debug_printed[0] = True

        return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

    # 启动训练
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
    )

    print("🚀 开始训练...")
    trainer.train()

    # 保存最终模型（只保存 LoRA 权重）
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"✅ 训练完成！模型保存到: {config.output_dir}")


if __name__ == "__main__":
    main()
