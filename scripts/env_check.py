import os,sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


# 检查gpu
def check_gpu():
    """检查 GPU 可用性"""
    print("=" * 50)
    print("1. GPU 检查")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False
    gpu_count = torch.cuda.device_count()
    print(f"GPU 数量: {gpu_count}")
    for i in range(gpu_count):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(f"GPU {i} 显存: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.1f} GB")
    return True

def check_libraries():
    """检查依赖库版本"""
    print("\n" + "=" * 50)
    print("2. 依赖库检查")
    print("=" * 50)

    libraries = {
        "torch": torch.__version__,
        "transformers": None,
        "peft": None,
        "trl": None,
    }

    try:
        import transformers
        libraries["transformers"] = transformers.__version__
    except ImportError:
        libraries["transformers"] = "未安装"

    try:
        import peft
        libraries["peft"] = peft.__version__
    except ImportError:
        libraries["peft"] = "未安装"

    try:
        import trl
        libraries["trl"] = trl.__version__
    except ImportError:
        libraries["trl"] = "未安装"

    for lib, version in libraries.items():
        status = "✅" if version != "未安装" else "❌"
        print(f"{status} {lib}: {version}")

    return all(v != "未安装" for v in libraries.values())

def check_model_loading():
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
        max_new_tokens=100000,
        temperature=0.3
        )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)   # 从完整输出中切片，去掉输入部分
    ]

    response = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

    print("\n专业回答生成完成！最终答案：", response)

    return True

def main():
    """主函数"""
    print("\n🚀 开始环境检查...\n")

    results = {
        "GPU": check_gpu(),
        "依赖库": check_libraries(),
        "模型加载": check_model_loading()
    }

    print("\n" + "=" * 50)
    print("检查结果汇总")
    print("=" * 50)

    for item, status in results.items():
        status_icon = "✅" if status else "❌"
        print(f"{status_icon} {item}: {'通过' if status else '失败'}")

    if all(results.values()):
        print("\n🎉 所有检查通过！环境配置正确。")
        return 0
    else:
        print("\n⚠️  部分检查失败，请根据上述错误信息修复。")
        return 1

if __name__ == "__main__":
    sys.exit(main())