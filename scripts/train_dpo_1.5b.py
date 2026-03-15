#!/usr/bin/env python3
"""DPO 训练 - Qwen2.5-1.5B + LoRA"""

import json, os, torch
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["PYTORCH_ALLOC_CONF"] = "expandable_segments:True"
os.environ.setdefault("WANDB_API_KEY", "")  # set via env: export WANDB_API_KEY=...

from datasets import Dataset
from peft import LoraConfig, TaskType
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig as TRLDPOConfig

MODEL_NAME  = "Qwen/Qwen2.5-1.5B-Instruct"
DATA_PATH   = "dataset/dpo_pairs.jsonl"
OUTPUT_DIR  = "checkpoints/dpo_qwen1.5b"
MAX_LENGTH  = 1024
NUM_EPOCHS  = 3
BATCH_SIZE  = 1
GRAD_ACCUM  = 8
LR          = 5e-5
BETA        = 0.1
LORA_R      = 8
LORA_ALPHA  = 16


def load_data(path):
    data = []
    with open(path, "r") as f:
        for line in f:
            d = json.loads(line)
            data.append({"prompt": d["prompt"], "chosen": d["chosen"], "rejected": d["rejected"]})
    print(f"✅ 读取 {len(data)} 条 DPO 数据")
    return Dataset.from_list(data)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("加载 tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("加载模型...")
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, torch_dtype=torch.float16, device_map="cuda:0", trust_remote_code=True
    )

    dataset = load_data(DATA_PATH)

    lora_config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R, lora_alpha=LORA_ALPHA, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"], bias="none",
    )

    training_args = TRLDPOConfig(
        output_dir=OUTPUT_DIR,
        num_train_epochs=NUM_EPOCHS,
        per_device_train_batch_size=BATCH_SIZE,
        gradient_accumulation_steps=GRAD_ACCUM,
        learning_rate=LR,
        fp16=True,
        logging_steps=10,
        save_steps=100,
        save_total_limit=2,
        report_to="wandb",
        run_name="dpo-qwen1.5b-robotics",
        beta=BETA,
        max_length=MAX_LENGTH,
    )

    trainer = DPOTrainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print("🚀 开始 DPO 训练...")
    trainer.train()
    # 合并 LoRA 权重后保存完整模型
    merged_model = trainer.model.merge_and_unload()
    merged_model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"✅ DPO 完成！保存到: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
