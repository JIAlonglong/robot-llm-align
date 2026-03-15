#!/usr/bin/env python3
"""
DPO（直接偏好优化）训练脚本
在 SFT 模型基础上，用偏好数据进一步对齐

运行方式：
    python scripts/train_dpo.py

依赖：
    pip install transformers peft trl datasets accelerate bitsandbytes wandb
"""

import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

from dataclasses import dataclass, field
from typing import List

import torch
import wandb
from datasets import Dataset
from peft import LoraConfig, PeftModel, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

os.environ.setdefault("WANDB_API_KEY", "")  # set via env: export WANDB_API_KEY=...


# ============================================================
# 1. 配置
# ============================================================

@dataclass
class TrainConfig:
    # 模型
    base_model: str = "Qwen/Qwen2.5-1.5B-Instruct"
    sft_model:  str = "checkpoints/sft_qwen2.5_7b_v2"   # SFT LoRA 权重目录

    # 数据
    data_path:  str = "dataset/dpo_train.jsonl"
    val_ratio:  float = 0.05

    # LoRA（在 SFT adapter 基础上再加一层，或直接 merge 后重新挂）
    lora_r:      int = 16
    lora_alpha:  int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(
        default_factory=lambda: ["q_proj", "v_proj"]
    )

    # DPO 超参
    beta: float = 0.1          # KL 惩罚系数，越大越保守
    max_length:        int = 1024
    max_prompt_length: int = 512

    # 训练
    output_dir:                    str = "checkpoints/dpo_qwen2.5_7b_v1"
    num_epochs:                    int = 3
    per_device_train_batch_size:   int = 1
    gradient_accumulation_steps:   int = 16
    learning_rate:               float = 5e-5   # DPO 用更小的 lr
    warmup_ratio:                float = 0.1
    lr_scheduler_type:             str = "cosine"
    save_steps:                    int = 50
    logging_steps:                 int = 5
    fp16:                         bool = True
    gradient_checkpointing:       bool = True


# ============================================================
# 2. 数据加载
# ============================================================

def load_dpo_dataset(path: str, val_ratio: float):
    """
    读取 DPO JSONL，期望每行格式：
    {"prompt": "...", "chosen": "...", "rejected": "..."}
    返回 HuggingFace Dataset（已划分 train/test）
    """
    records = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            # 过滤掉 rejected 为空的条目（数据还没生成完）
            if not rec.get("chosen") or not rec.get("rejected"):
                continue
            records.append({
                "prompt":   rec["prompt"],
                "chosen":   rec["chosen"],
                "rejected": rec["rejected"],
            })

    print(f"✅ 读取 DPO 数据: {len(records)} 条 ({path})")
    dataset = Dataset.from_list(records)
    split = dataset.train_test_split(test_size=val_ratio, seed=42)
    print(f"📊 训练集: {len(split['train'])} 条，验证集: {len(split['test'])} 条")
    return split["train"], split["test"]


# ============================================================
# 3. 模型加载
# ============================================================

def load_model_and_tokenizer(config: TrainConfig):
    """
    加载 base model，合并 SFT LoRA 权重，再挂新的 DPO LoRA adapter。
    reference model（DPOTrainer 内部自动处理）会 clone 合并后的模型。
    """
    print(f"📦 加载 base model: {config.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(
        config.base_model,
        trust_remote_code=True,
        padding_side="left",   # DPO 推理时左填充
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    base = AutoModelForCausalLM.from_pretrained(
        config.base_model,
        torch_dtype=torch.float16,
        device_map="cuda:0",
        trust_remote_code=True,
    )

    # 合并 SFT LoRA → 得到完整的 SFT 模型权重
    print(f"🔗 合并 SFT LoRA: {config.sft_model}")
    model = PeftModel.from_pretrained(base, config.sft_model)
    model = model.merge_and_unload()   # 合并后卸载 adapter，得到普通模型

    # 挂新的 DPO LoRA adapter
    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    from peft import get_peft_model
    model = get_peft_model(model, lora_config)

    if config.gradient_checkpointing:
        model.enable_input_require_grads()
        model.gradient_checkpointing_enable()

    model.print_trainable_parameters()
    return model, tokenizer


# ============================================================
# 4. 训练入口
# ============================================================

def main():
    import argparse
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data_path",                      default=None)
    parser.add_argument("--sft_model",                      default=None)
    parser.add_argument("--output_dir",                     default=None)
    parser.add_argument("--num_epochs",                     type=int,   default=None)
    parser.add_argument("--per_device_train_batch_size",    type=int,   default=None)
    parser.add_argument("--gradient_accumulation_steps",    type=int,   default=None)
    known, _ = parser.parse_known_args()

    config = TrainConfig()
    if known.data_path:                   config.data_path                   = known.data_path
    if known.sft_model:                   config.sft_model                   = known.sft_model
    if known.output_dir:                  config.output_dir                  = known.output_dir
    if known.num_epochs is not None:      config.num_epochs                  = known.num_epochs
    if known.per_device_train_batch_size: config.per_device_train_batch_size = known.per_device_train_batch_size
    if known.gradient_accumulation_steps: config.gradient_accumulation_steps = known.gradient_accumulation_steps
    os.makedirs(config.output_dir, exist_ok=True)

    train_dataset, eval_dataset = load_dpo_dataset(config.data_path, config.val_ratio)
    model, tokenizer = load_model_and_tokenizer(config)

    dpo_config = DPOConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.num_epochs,
        per_device_train_batch_size=config.per_device_train_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        lr_scheduler_type=config.lr_scheduler_type,
        fp16=config.fp16,
        logging_steps=config.logging_steps,
        save_steps=config.save_steps,
        save_total_limit=2,
        eval_strategy="no",
        load_best_model_at_end=False,
        report_to="wandb",
        run_name="dpo-qwen2.5-7b-robotics-v1",
        dataloader_num_workers=0,
        gradient_checkpointing=config.gradient_checkpointing,
        remove_unused_columns=False,
        # DPO 专属
        beta=config.beta,
        max_length=config.max_length,
    )

    trainer = DPOTrainer(
        model=model,
        ref_model=None,
        args=dpo_config,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
    )

    print("🚀 开始 DPO 训练...")
    trainer.train()

    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"✅ DPO 训练完成！模型保存到: {config.output_dir}")


if __name__ == "__main__":
    main()
