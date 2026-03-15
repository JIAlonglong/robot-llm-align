#!/usr/bin/env python3
"""SFT 训练 - Qwen2.5-1.5B-Instruct + LoRA"""

import json, os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"

from dataclasses import dataclass, field
from typing import List, Dict, Optional
from datasets import Dataset
from peft import LoraConfig, get_peft_model, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, Trainer
import wandb

os.environ.setdefault("WANDB_API_KEY", "")  # set via env: export WANDB_API_KEY=...

@dataclass
class SFTConfig:
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct"
    data_path:  str = "dataset/sft_with_tools.jsonl"
    max_length: int = 1024
    val_ratio:  float = 0.05
    lora_r:     int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "v_proj", "k_proj", "o_proj"])
    output_dir: str = "checkpoints/sft_qwen1.5b_with_tools"
    num_epochs: int = 20
    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 8
    learning_rate: float = 2e-4
    warmup_ratio: float = 0.05
    lr_scheduler_type: str = "cosine"
    save_steps: int = 100
    logging_steps: int = 10
    fp16: bool = True
    gradient_checkpointing: bool = True


def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                data.append(json.loads(line))
    print(f"✅ 读取数据: {len(data)} 条")
    return data


def format_conversation(item, tokenizer, max_length):
    conversations = item.get("conversations", [])
    if not conversations:
        return None

    result = tokenizer.apply_chat_template(conversations, tokenize=True, add_generation_prompt=False)
    input_ids = result if isinstance(result, list) else result["input_ids"]

    if len(input_ids) > max_length:
        return None

    labels = [-100] * len(input_ids)
    prev_len = 0
    for i, turn in enumerate(conversations):
        partial = tokenizer.apply_chat_template(conversations[:i+1], tokenize=True, add_generation_prompt=False)
        partial_ids = partial if isinstance(partial, list) else partial["input_ids"]
        cur_len = len(partial_ids)
        if turn["role"] == "assistant":
            labels[prev_len:cur_len] = input_ids[prev_len:cur_len]
        prev_len = cur_len

    return {"input_ids": input_ids, "attention_mask": [1]*len(input_ids), "labels": labels}


def build_dataset(data, tokenizer, max_length):
    processed, skipped = [], 0
    for item in data:
        r = format_conversation(item, tokenizer, max_length)
        if r is None:
            skipped += 1
        else:
            processed.append(r)
    print(f"📊 保留 {len(processed)} 条，过滤 {skipped} 条")
    return Dataset.from_list(processed)


def main():
    config = SFTConfig()
    os.makedirs(config.output_dir, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(config.model_name, trust_remote_code=True, padding_side="right")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(config.model_name, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=config.lora_r, lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        target_modules=config.lora_target_modules,
        bias="none",
    )
    model = get_peft_model(model, lora_config)
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    model.print_trainable_parameters()

    raw_data = load_jsonl(config.data_path)
    dataset = build_dataset(raw_data, tokenizer, config.max_length)
    split = dataset.train_test_split(test_size=config.val_ratio, seed=42)
    print(f"训练集: {len(split['train'])} | 验证集: {len(split['test'])}")

    pad_id = tokenizer.pad_token_id
    def data_collator(features):
        input_ids = torch.nn.utils.rnn.pad_sequence([torch.tensor(f["input_ids"]) for f in features], batch_first=True, padding_value=pad_id)
        labels    = torch.nn.utils.rnn.pad_sequence([torch.tensor(f["labels"])    for f in features], batch_first=True, padding_value=-100)
        return {"input_ids": input_ids, "attention_mask": (input_ids != pad_id).long(), "labels": labels}

    training_args = TrainingArguments(
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
        report_to="wandb",
        run_name="sft-qwen1.5b-robotics",
        dataloader_num_workers=0,
        gradient_checkpointing=True,
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model, args=training_args,
        train_dataset=split["train"], eval_dataset=split["test"],
        data_collator=data_collator,
    )

    print("🚀 开始 SFT 训练...")
    trainer.train()
    model.save_pretrained(config.output_dir)
    tokenizer.save_pretrained(config.output_dir)
    print(f"✅ SFT 完成！保存到: {config.output_dir}")


if __name__ == "__main__":
    main()
