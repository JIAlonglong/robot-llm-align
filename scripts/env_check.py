from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# 选用的模型
model_name = "Qwen/Qwen2.5-1.5B-Instruct"

print(f"Loading model from {model_name}")
# 加载分词器和模型（半精度 bfloat16 加载，完美适配 4090）
tokenizer = AutoTokenizer.from_pretrained(model_name, device_map="auto")
model = AutoModelForCausalLM.from_pretrained(model_name, device_map="auto")
print("Model loaded successfully")
print(f"Tokenizer: {tokenizer}")
print(f"Model: {model}")
print("Environment check completed successfully")

# 构造具有严格逻辑边界的对话输入
messages = [
    {"role": "system", "content": "你是一个专注于机器人和控制算法的专业代理。请严格遵守系统指令和逻辑边界，仅输出专业、硬核的技术解析，拒绝任何废话和泛泛而谈。"},
    {"role": "user", "content": "请简述强化学习（RL）中的奖励函数与模型预测控制（MPC）在四足机器人运动规划中的核心差异。"},
]

# 调用分词器的对话模板，将上面的结构化文本拼接成模型可以读懂的Tensor
text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer(text, return_tensors="pt").to(model.device)

print("\n模型已成功加载到 GPU 显存！正在根据严苛指令生成专业回答...")

# 生成输出
generated_ids = model.generate(
    model_inputs["input_ids"], 
    attention_mask=model_inputs.attention_mask,
    max_new_tokens=200,
    temperature=0.3
    )
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)   # 从完整输出中切片，去掉输入部分
]

response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

print("\n专业回答生成完成！最终答案：", response)