# Day 1 详细任务清单

> 环境验证与资源分配（预计 2-3 小时）

---

## 任务目标

确保能独占一张 GPU 跑通完整的模型加载和推理流程，为后续训练做好准备。

---

## 任务清单

### Task 1.1：GPU 资源确认（15 分钟）

**目标**：确认可用的 GPU 资源

**步骤**：
```bash
# 1. 查看 GPU 状态
nvidia-smi

# 2. 确认 GPU 3 是否空闲（根据你的实际情况）
# 如果 GPU 3 被占用，选择其他空闲的 GPU

# 3. 设置环境变量（后续所有命令都使用这个 GPU）
export CUDA_VISIBLE_DEVICES=3
echo $CUDA_VISIBLE_DEVICES
```

**验证标准**：
- [ ] 能看到 GPU 列表
- [ ] 确认至少有一张 GPU 显存使用率 < 10%
- [ ] 环境变量设置成功

---

### Task 1.2：Python 环境验证（15 分钟）

**目标**：确认所有依赖库已正确安装

**步骤**：
```bash
cd /home/liujl/big_model/robot-llm-align

# 1. 检查 Python 版本
python --version  # 应该 >= 3.8

# 2. 安装依赖（如果还没安装）
pip install -r requirements.txt -i https://mirrors.aliyun.com/pypi/simple

# 3. 验证核心库版本
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import peft; print(f'PEFT: {peft.__version__}')"
python -c "import trl; print(f'TRL: {trl.__version__}')"
```

**验证标准**：
- [ ] PyTorch >= 2.0.0
- [ ] Transformers >= 4.57.0
- [ ] PEFT >= 0.7.0
- [ ] TRL >= 0.13.0

---

### Task 1.3：模型下载与加载测试（30-60 分钟）

**目标**：下载 Qwen2.5-1.5B 模型并测试加载

**步骤**：

#### 方案 A：自动下载（推荐，需要网络）
```bash
# 运行环境检查脚本
python scripts/env_check.py
```

#### 方案 B：手动下载（网络不稳定时）
```bash
# 1. 使用 modelscope 下载（国内更快）
pip install modelscope

# 2. 下载模型
python -c "
from modelscope import snapshot_download
model_dir = snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')
print(f'模型已下载到: {model_dir}')
"

# 3. 修改 env_check.py 中的模型路径
# model_name = "./models/qwen/Qwen2.5-1.5B-Instruct"
```

**验证标准**：
- [ ] 模型下载成功（约 3GB）
- [ ] 模型加载成功（无 OOM 错误）
- [ ] 能够生成文本（输出不是乱码）

---

### Task 1.4：创建环境检查脚本（30 分钟）

**目标**：编写一个完整的环境检查脚本

**文件路径**：`scripts/env_check.py`

**脚本内容**：
```python
#!/usr/bin/env python3
"""
环境检查脚本
验证 GPU、依赖库、模型加载是否正常
"""

import sys
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

def check_gpu():
    """检查 GPU 可用性"""
    print("=" * 50)
    print("1. GPU 检查")
    print("=" * 50)

    if not torch.cuda.is_available():
        print("❌ CUDA 不可用")
        return False

    gpu_count = torch.cuda.device_count()
    print(f"✅ 检测到 {gpu_count} 张 GPU")

    for i in range(gpu_count):
        gpu_name = torch.cuda.get_device_name(i)
        gpu_memory = torch.cuda.get_device_properties(i).total_memory / 1024**3
        print(f"   GPU {i}: {gpu_name} ({gpu_memory:.1f} GB)")

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
    """检查模型加载"""
    print("\n" + "=" * 50)
    print("3. 模型加载检查")
    print("=" * 50)

    model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    try:
        print(f"正在加载模型: {model_name}")
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            device_map="auto",
            torch_dtype=torch.bfloat16
        )
        print("✅ 模型加载成功")

        # 测试推理
        print("\n正在测试推理...")
        messages = [
            {"role": "system", "content": "你是一个机器人控制领域的专家。"},
            {"role": "user", "content": "请简述强化学习中的 Q-learning 算法。"}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            model_inputs.input_ids,
            max_new_tokens=200,
            temperature=0.3,
            do_sample=True
        )

        # 只保留新生成的部分
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        print("\n" + "-" * 50)
        print("模型回答:")
        print(response)
        print("-" * 50)

        print("\n✅ 推理测试成功")
        return True

    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return False

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
```

**运行脚本**：
```bash
python scripts/env_check.py
```

**验证标准**：
- [ ] 所有检查项都显示 ✅
- [ ] 模型能够生成合理的回答（不是乱码）

---

### Task 1.5：记录环境信息（15 分钟）

**目标**：记录环境配置，便于后续复现

**步骤**：
```bash
# 1. 导出依赖版本
pip freeze > requirements_frozen.txt

# 2. 记录 GPU 信息
nvidia-smi > logs/gpu_info.txt

# 3. 记录 CUDA 版本
nvcc --version > logs/cuda_version.txt

# 4. 创建环境信息文件
cat > logs/environment_info.txt << EOF
日期: $(date)
Python 版本: $(python --version)
PyTorch 版本: $(python -c "import torch; print(torch.__version__)")
CUDA 版本: $(python -c "import torch; print(torch.version.cuda)")
GPU 型号: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)
可用显存: $(nvidia-smi --query-gpu=memory.free --format=csv,noheader | head -1)
EOF

cat logs/environment_info.txt
```

**验证标准**：
- [ ] 生成 `requirements_frozen.txt`
- [ ] 生成 `logs/environment_info.txt`
- [ ] 文件内容完整

---

## 交付物清单

完成 Day 1 后，应该有以下文件：

```
robot-llm-align/
├── scripts/
│   └── env_check.py              ✅ 环境检查脚本
├── logs/
│   ├── gpu_info.txt              ✅ GPU 信息
│   ├── cuda_version.txt          ✅ CUDA 版本
│   └── environment_info.txt      ✅ 环境汇总
├── requirements_frozen.txt       ✅ 冻结的依赖版本
└── models/                       ✅ 下载的模型（可选）
    └── qwen/
        └── Qwen2.5-1.5B-Instruct/
```

---

## 常见问题与解决方案

### Q1: CUDA Out of Memory
**原因**：GPU 显存不足
**解决**：
```bash
# 1. 确认 GPU 是否被其他进程占用
nvidia-smi

# 2. 杀死占用进程（谨慎操作）
kill -9 <PID>

# 3. 或者选择其他空闲的 GPU
export CUDA_VISIBLE_DEVICES=5
```

### Q2: 模型下载速度慢
**原因**：网络问题
**解决**：
```bash
# 使用 modelscope（国内镜像）
pip install modelscope
python -c "
from modelscope import snapshot_download
snapshot_download('qwen/Qwen2.5-1.5B-Instruct', cache_dir='./models')
"
```

### Q3: 依赖库版本冲突
**原因**：已有环境中的库版本不兼容
**解决**：
```bash
# 创建新的虚拟环境
python -m venv venv_robot_llm
source venv_robot_llm/bin/activate
pip install -r requirements.txt
```

---

## 下一步

完成 Day 1 后，进入 **Day 2-3：SFT 数据准备**。

需要我继续细化 Day 2-3 的任务吗？
