# 升级版技术路线：融合 LangChain + OpenAgents + AutoGPT

> 基于三大开源项目的核心思想，构建一个**具备 Agent 能力的机器人控制领域 LLM**

---

## 核心洞察：从三大项目学到什么？

### 1. LangChain 的启发
**核心价值**：模块化 + 可组合性
- **Chain 思想**：将复杂任务分解为可组合的步骤
- **Memory 机制**：对话历史管理
- **Tool Use**：LLM 调用外部工具的标准接口

**对我们的启发**：
- 不仅训练一个"会回答问题"的模型
- 而是训练一个"会使用工具解决问题"的 Agent

### 2. OpenAgents 的启发
**核心价值**：三种 Agent 类型
- **Data Agent**：数据分析（写代码、执行、可视化）
- **Plugins Agent**：调用 200+ 工具
- **Web Agent**：自主浏览网页

**对我们的启发**：
- 机器人控制领域也需要"工具调用"能力
- 例如：调用仿真器、绘制控制曲线、执行 PID 调参脚本

### 3. AutoGPT 的启发
**核心价值**：自主规划 + 多步推理
- 给定目标，自动分解为子任务
- 循环执行：思考 → 行动 → 观察 → 反思

**对我们的启发**：
- 训练模型不仅回答"什么是 Q-learning"
- 还能回答"如何为四足机器人设计强化学习控制器"（需要多步推理）

---

## 升级后的项目目标

### 原目标（基础版）
训练一个在机器人控制领域**减少幻觉**的 LLM

### 新目标（Agent 版）
训练一个具备以下能力的**机器人控制 Agent**：
1. **专业知识问答**（原目标）
2. **工具调用**：调用仿真器、绘图工具、代码执行器
3. **多步推理**：分解复杂任务（如"设计一个倒立摆控制器"）
4. **代码生成与执行**：生成控制算法代码并验证

---

## 完整技术路线（3-4 周）

### 第一周：SFT 基线 + 工具调用能力

#### Day 1-2：环境验证 + SFT 数据准备
**任务**：
- 验证 GPU 资源
- 准备 SFT 数据（300 条通用 + 100 条领域）

**数据格式升级**：
```json
{
  "conversations": [
    {"role": "system", "content": "你是一个机器人控制专家，可以调用工具"},
    {"role": "user", "content": "帮我绘制 Q-learning 的收敛曲线"},
    {"role": "assistant", "content": "我将使用 plot_tool 绘制曲线\n<tool_call>plot_convergence(algorithm='q_learning')</tool_call>"}
  ]
}
```

**关键变化**：
- 引入 `<tool_call>` 标签（参考 LangChain 的 Tool Use）
- 模型不仅回答问题，还会"调用工具"

#### Day 3-4：定义工具集（Tool Registry）
**灵感来源**：OpenAgents 的 Plugins Agent

**工具清单**：
1. **仿真工具**
   - `simulate_pid(kp, ki, kd)` - 运行 PID 仿真
   - `simulate_rl(algorithm, env)` - 运行强化学习仿真

2. **可视化工具**
   - `plot_convergence(data)` - 绘制收敛曲线
   - `plot_phase_portrait(system)` - 绘制相图

3. **代码执行工具**
   - `execute_python(code)` - 执行 Python 代码
   - `validate_control_law(code)` - 验证控制律正确性

4. **知识检索工具**
   - `search_paper(query)` - 搜索相关论文
   - `lookup_formula(concept)` - 查找公式

**实现方式**：
```python
# scripts/tools/tool_registry.py
from typing import Callable, Dict

class ToolRegistry:
    def __init__(self):
        self.tools: Dict[str, Callable] = {}

    def register(self, name: str, func: Callable):
        self.tools[name] = func

    def execute(self, tool_call: str):
        # 解析 <tool_call>simulate_pid(kp=1.0, ki=0.1, kd=0.05)</tool_call>
        # 执行对应的函数
        pass

# 注册工具
registry = ToolRegistry()
registry.register("simulate_pid", simulate_pid_func)
registry.register("plot_convergence", plot_convergence_func)
```

#### Day 5-7：SFT 训练（带工具调用）
**训练数据示例**：
```json
{
  "conversations": [
    {"role": "user", "content": "帮我调试这个 PID 控制器，Kp=2.0, Ki=0.5, Kd=0.1"},
    {"role": "assistant", "content": "我将运行仿真来测试这组参数\n<tool_call>simulate_pid(kp=2.0, ki=0.5, kd=0.1)</tool_call>\n\n仿真结果显示超调量为 15%，建议降低 Kp 到 1.5"}
  ]
}
```

**训练脚本**：
- 使用 `scripts/train_sft_with_tools.py`
- 模型学会在合适的时机插入 `<tool_call>` 标签

**交付物**：
- `checkpoints/sft_with_tools/`（具备工具调用能力的模型）

---

### 第二周：DPO 训练 + 多步推理能力

#### Day 8-10：构建偏好数据（工具调用场景）
**Chosen vs Rejected 示例**：

**场景 1：工具选择错误**
```json
{
  "prompt": "帮我分析这个强化学习算法的性能",
  "chosen": "<tool_call>simulate_rl(algorithm='ppo', env='cartpole')</tool_call>",
  "rejected": "<tool_call>plot_convergence(data=None)</tool_call>  # 错误：没有数据就绘图"
}
```

**场景 2：参数传递错误**
```json
{
  "prompt": "运行 PID 仿真，Kp=1.0",
  "chosen": "<tool_call>simulate_pid(kp=1.0, ki=0.0, kd=0.0)</tool_call>",
  "rejected": "<tool_call>simulate_pid(kp='1.0')</tool_call>  # 错误：类型错误"
}
```

**场景 3：工具调用时机错误**
```json
{
  "prompt": "解释什么是 PID 控制",
  "chosen": "PID 控制是一种经典的反馈控制方法...",  # 不需要调用工具
  "rejected": "<tool_call>simulate_pid()</tool_call>\nPID 控制是..."  # 错误：不必要的工具调用
}
```

#### Day 11-12：DPO 训练
**目标**：
- 模型学会"何时调用工具"
- 模型学会"调用哪个工具"
- 模型学会"如何传递参数"

**训练脚本**：
- `scripts/train_dpo_with_tools.py`

#### Day 13-14：多步推理能力（ReAct 模式）
**灵感来源**：AutoGPT 的自主规划

**ReAct 模式**：
```
Thought: 我需要先运行仿真，然后分析结果
Action: <tool_call>simulate_pid(kp=1.0, ki=0.1, kd=0.05)</tool_call>
Observation: 仿真结果显示超调量 20%，稳态误差 2%
Thought: 超调量过大，需要降低 Kp
Action: <tool_call>simulate_pid(kp=0.8, ki=0.1, kd=0.05)</tool_call>
Observation: 超调量降低到 10%，稳态误差 1%
Answer: 建议使用 Kp=0.8, Ki=0.1, Kd=0.05
```

**训练数据**：
- 构造 50 个"多步推理"的示例
- 每个示例包含 2-4 轮 Thought-Action-Observation

**交付物**：
- `checkpoints/dpo_with_react/`（具备多步推理能力的模型）

---

### 第三周：Agent 系统集成

#### Day 15-17：构建 Agent 执行引擎
**灵感来源**：LangChain 的 Agent Executor

**核心组件**：
```python
# scripts/agent/agent_executor.py
class RobotControlAgent:
    def __init__(self, model, tool_registry):
        self.model = model
        self.tools = tool_registry
        self.memory = []  # 对话历史

    def run(self, user_query: str, max_steps: int = 5):
        """
        执行 Agent 循环：
        1. 模型生成响应（可能包含 <tool_call>）
        2. 解析并执行工具调用
        3. 将结果反馈给模型
        4. 重复直到任务完成或达到最大步数
        """
        for step in range(max_steps):
            # 1. 模型生成
            response = self.model.generate(user_query, self.memory)

            # 2. 检查是否有工具调用
            if "<tool_call>" in response:
                tool_result = self.tools.execute(response)
                self.memory.append({"role": "observation", "content": tool_result})
            else:
                # 任务完成
                return response

        return "达到最大步数限制"
```

**实现工具**：
```python
# scripts/tools/simulation_tools.py
def simulate_pid(kp: float, ki: float, kd: float) -> dict:
    """运行 PID 仿真"""
    # 调用 scipy 或自定义仿真器
    result = run_pid_simulation(kp, ki, kd)
    return {
        "overshoot": result.overshoot,
        "settling_time": result.settling_time,
        "steady_state_error": result.sse
    }

def plot_convergence(data: list) -> str:
    """绘制收敛曲线"""
    import matplotlib.pyplot as plt
    plt.plot(data)
    plt.savefig("convergence.png")
    return "图表已保存到 convergence.png"
```

#### Day 18-19：端到端测试
**测试场景**：

**场景 1：PID 调参助手**
```
User: 帮我为倒立摆系统设计 PID 控制器
Agent:
  Thought: 我需要先尝试一组初始参数
  Action: <tool_call>simulate_pid(kp=10.0, ki=1.0, kd=2.0)</tool_call>
  Observation: 超调量 50%，系统不稳定
  Thought: Kp 过大，需要降低
  Action: <tool_call>simulate_pid(kp=5.0, ki=1.0, kd=2.0)</tool_call>
  Observation: 超调量 15%，稳态误差 3%
  Answer: 建议使用 Kp=5.0, Ki=1.0, Kd=2.0
```

**场景 2：强化学习算法对比**
```
User: 对比 Q-learning 和 SARSA 在 CartPole 环境中的性能
Agent:
  Thought: 我需要分别运行两个算法
  Action: <tool_call>simulate_rl(algorithm='q_learning', env='cartpole')</tool_call>
  Observation: Q-learning 平均奖励 195
  Action: <tool_call>simulate_rl(algorithm='sarsa', env='cartpole')</tool_call>
  Observation: SARSA 平均奖励 180
  Action: <tool_call>plot_comparison(data=[195, 180], labels=['Q-learning', 'SARSA'])</tool_call>
  Answer: Q-learning 性能更优，图表已生成
```

#### Day 20-21：Web UI 开发
**灵感来源**：OpenAgents 的 Chat Web UI

**技术栈**：
- Streamlit（快速原型）或 Gradio
- 显示 Agent 的思考过程（Thought-Action-Observation）
- 实时显示工具调用结果（图表、仿真输出）

**界面设计**：
```
┌─────────────────────────────────────┐
│  Robot Control Agent                │
├─────────────────────────────────────┤
│  User: 帮我调试 PID 控制器          │
│                                     │
│  Agent:                             │
│  💭 Thought: 需要运行仿真           │
│  🔧 Action: simulate_pid(...)       │
│  📊 Observation: [仿真结果图表]     │
│  💭 Thought: 参数需要调整           │
│  🔧 Action: simulate_pid(...)       │
│  ✅ Answer: 建议使用 Kp=...         │
└─────────────────────────────────────┘
```

---

### 第四周：评估与优化

#### Day 22-24：Agent 能力评估
**评估维度**：

1. **工具调用准确率**
   - 是否选择了正确的工具？
   - 参数传递是否正确？

2. **多步推理能力**
   - 能否分解复杂任务？
   - 推理步骤是否合理？

3. **任务完成率**
   - 给定 50 个测试任务，Agent 能完成多少？

**评估脚本**：
```python
# scripts/eval_agent.py
test_tasks = [
    "为倒立摆设计 PID 控制器",
    "对比 PPO 和 SAC 在 Hopper 环境中的性能",
    "分析这段控制代码的稳定性",
    # ... 50 个任务
]

for task in test_tasks:
    result = agent.run(task)
    # 评估结果质量
```

#### Day 25-27：与 Baseline 对比
**对比对象**：
- GPT-4o（通用模型）
- 我们的 SFT 模型（无工具调用）
- 我们的 Agent 模型（有工具调用）

**对比指标**：
- 专业准确性
- 工具调用成功率
- 任务完成率

#### Day 28：项目打包与展示
**GitHub README 更新**：
```markdown
## 核心亮点

### 1. 领域专业知识
- 从经典教材提取 100 条高质量 QA 对
- DPO 训练减少幻觉，专业准确性提升 XX%

### 2. 工具调用能力
- 定义 8 个机器人控制领域的专用工具
- 模型学会在合适时机调用工具
- 工具调用准确率达 XX%

### 3. 多步推理能力
- 实现 ReAct 模式（Thought-Action-Observation）
- 能够分解复杂任务（如"设计控制器"）
- 平均推理步数 X 步

### 4. Agent 系统
- 完整的 Agent 执行引擎
- Web UI 展示思考过程
- 端到端任务完成率 XX%

## Demo 视频

[插入 Agent 运行的 GIF/视频]
```

**简历描述**：
```
【机器人控制领域的 LLM Agent】2026.03-04

- 从经典教材（Sutton & Barto RL 书）提取 100 条高质量 QA 对，使用 DPO 算法进行偏好对齐，模型在专业准确性上的胜率提升 XX%
- 参考 LangChain 的 Tool Use 机制，定义 8 个机器人控制领域的专用工具（仿真器、绘图工具、代码执行器），训练模型具备工具调用能力
- 实现 ReAct 模式（Thought-Action-Observation），使模型具备多步推理能力，能够分解复杂任务（如"设计 PID 控制器"）
- 构建完整的 Agent 执行引擎和 Web UI，端到端任务完成率达 XX%
- 技术栈：PyTorch, Transformers, PEFT, TRL, LangChain, Streamlit
```

---

## 关键技术对比

### 原计划 vs 升级版

| 维度 | 原计划（基础版） | 升级版（Agent 版） |
|------|-----------------|-------------------|
| **核心能力** | 问答（减少幻觉） | 问答 + 工具调用 + 多步推理 |
| **数据类型** | QA 对 | QA 对 + 工具调用示例 + ReAct 轨迹 |
| **训练阶段** | SFT + DPO | SFT + DPO + ReAct 微调 |
| **系统复杂度** | 单模型推理 | Agent 执行引擎 + 工具注册表 |
| **应用场景** | 知识问答 | 实际任务执行（调参、仿真、代码生成） |
| **面试亮点** | "减少幻觉" | "构建 Agent 系统" |
| **时间成本** | 2 周 | 3-4 周 |

---

## 从三大项目借鉴的核心模式

### 1. LangChain 的模式
- **Chain**：任务分解（Thought → Action → Observation）
- **Tool Use**：标准化的工具调用接口
- **Memory**：对话历史管理

### 2. OpenAgents 的模式
- **Tool Registry**：工具注册表
- **Web UI**：可视化 Agent 思考过程
- **多 Agent 类型**：Data Agent / Plugins Agent / Web Agent

### 3. AutoGPT 的模式
- **ReAct**：Reasoning + Acting
- **自主规划**：给定目标，自动分解子任务
- **循环执行**：直到任务完成或达到最大步数

---

## 实施建议

### 渐进式开发策略

**阶段 1（Week 1-2）**：先完成基础版
- SFT + DPO
- 验证数据质量和训练流程

**阶段 2（Week 3）**：引入工具调用
- 定义 2-3 个简单工具（如 `plot_convergence`）
- 训练模型识别 `<tool_call>` 标签

**阶段 3（Week 4）**：完整 Agent 系统
- 实现 ReAct 模式
- 构建 Web UI

### 风险控制

| 风险 | 应对策略 |
|------|---------|
| 工具调用训练困难 | 先用少量数据（50 条）验证可行性 |
| ReAct 模式不收敛 | 降低推理步数（max_steps=3） |
| Agent 系统复杂度高 | 先实现最小可行版本（MVP） |
| 时间不足 | 优先完成基础版，Agent 功能作为可选项 |

---

## 下一步行动

### 立即开始（Day 1）
1. 运行 `scripts/env_check.py` 验证环境
2. 决定是否采用升级版路线（需要额外 1-2 周）
3. 如果采用，先完成基础版（Week 1-2），再扩展 Agent 功能

### 关键决策点
**问题**：是否值得投入额外时间构建 Agent 系统？

**考虑因素**：
- ✅ **面试价值**：Agent 系统比单纯的 DPO 更有亮点
- ✅ **技术深度**：涉及工具调用、多步推理、系统集成
- ✅ **实用性**：Agent 能解决实际问题（不只是回答问题）
- ❌ **时间成本**：需要额外 1-2 周
- ❌ **复杂度**：调试难度更高

**建议**：
- 如果时间充裕（4 周+），强烈推荐升级版
- 如果时间紧张（2 周），先完成基础版，后续再扩展

需要我帮你：
1. 生成工具调用的训练数据示例？
2. 实现 Tool Registry 的代码框架？
3. 或者先从基础版开始，逐步迭代？
