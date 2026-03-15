#!/usr/bin/env python3
"""生成工具调用训练数据"""

import json
import os

OUTPUT = "dataset/tool_call_sft.jsonl"

# 工具调用示例数据
TOOL_EXAMPLES = [
    # RRT 规划
    {
        "user": "帮我用RRT算法规划一条从(0,0)到(5,5)的路径，避开障碍物(2,2,半径0.5)和(3,3,半径0.5)",
        "assistant": "我将使用RRT路径规划工具为您规划路径。\n\n<tool_call>rrt_planning(start_x=0, start_y=0, goal_x=5, goal_y=5, obstacle_list=\"2,2,0.5;3,3,0.5\")</tool_call>"
    },
    {
        "user": "从原点到(10,10)规划路径，有一个圆形障碍物在(5,5)半径1",
        "assistant": "好的，我使用RRT算法规划路径。\n\n<tool_call>rrt_planning(start_x=0, start_y=0, goal_x=10, goal_y=10, obstacle_list=\"5,5,1\")</tool_call>"
    },

    # A* 规划
    {
        "user": "用A*算法从(0,0)到(8,8)规划路径",
        "assistant": "我将使用A*网格规划算法。\n\n<tool_call>astar_planning(start_x=0, start_y=0, goal_x=8, goal_y=8)</tool_call>"
    },
    {
        "user": "A*规划从(1,1)到(10,10)",
        "assistant": "使用A*算法规划路径。\n\n<tool_call>astar_planning(start_x=1, start_y=1, goal_x=10, goal_y=10)</tool_call>"
    },

    # 样条轨迹
    {
        "user": "生成一条经过点(0,0), (2,3), (5,2), (7,4)的平滑轨迹",
        "assistant": "我将使用三次样条插值生成平滑轨迹。\n\n<tool_call>cubic_spline_planning(waypoints=\"0,0;2,3;5,2;7,4\")</tool_call>"
    },
    {
        "user": "用样条曲线连接这些路径点：(0,0), (1,2), (3,3)",
        "assistant": "使用三次样条规划平滑轨迹。\n\n<tool_call>cubic_spline_planning(waypoints=\"0,0;1,2;3,3\")</tool_call>"
    },

    # PID 仿真
    {
        "user": "仿真一个PID控制器，Kp=1.0, Ki=0.1, Kd=0.05，目标值100",
        "assistant": "我将运行PID控制仿真。\n\n<tool_call>simulate_pid(kp=1.0, ki=0.1, kd=0.05, setpoint=100)</tool_call>"
    },
    {
        "user": "测试PID参数Kp=2.0, Ki=0.5, Kd=0.1",
        "assistant": "运行PID仿真测试这组参数。\n\n<tool_call>simulate_pid(kp=2.0, ki=0.5, kd=0.1)</tool_call>"
    },
    {
        "user": "帮我调试PID控制器，先试试Kp=0.8, Ki=0.05, Kd=0.02",
        "assistant": "好的，我先用这组参数运行仿真看看效果。\n\n<tool_call>simulate_pid(kp=0.8, ki=0.05, kd=0.02)</tool_call>"
    },

    # 机械臂正运动学
    {
        "user": "计算机械臂正运动学，关节角度[0.5, 0.5, 0.5]，连杆长度[1, 1, 1]",
        "assistant": "我将计算机械臂的末端位置。\n\n<tool_call>arm_forward_kinematics(joint_angles=\"0.5,0.5,0.5\", link_lengths=\"1,1,1\")</tool_call>"
    },
    {
        "user": "三连杆机械臂，关节角[0.3, 0.6, 0.9]，连杆长[1.5, 1.2, 0.8]，求末端位置",
        "assistant": "计算正运动学。\n\n<tool_call>arm_forward_kinematics(joint_angles=\"0.3,0.6,0.9\", link_lengths=\"1.5,1.2,0.8\")</tool_call>"
    },

    # EKF 定位
    {
        "user": "用EKF估计机器人位置，当前状态(0,0,0)，控制输入(1,0.1)，观测(0.1,0.1)",
        "assistant": "我将使用扩展卡尔曼滤波进行状态估计。\n\n<tool_call>ekf_localization(state=\"0,0,0\", control=\"1,0.1\", measurement=\"0.1,0.1\")</tool_call>"
    },
    {
        "user": "EKF定位，状态(1,2,0.5)，控制(0.5,0.2)，观测(1.1,2.1)",
        "assistant": "使用EKF更新状态估计。\n\n<tool_call>ekf_localization(state=\"1,2,0.5\", control=\"0.5,0.2\", measurement=\"1.1,2.1\")</tool_call>"
    },

    # 路径对比
    {
        "user": "对比两条路径：路径1经过(0,0),(2,2),(4,4)，路径2经过(0,0),(1,3),(4,4)",
        "assistant": "我将绘制路径对比图。\n\n<tool_call>plot_path_comparison(paths=\"0,0;2,2;4,4|0,0;1,3;4,4\", labels=\"路径1,路径2\", title=\"路径对比\")</tool_call>"
    },

    # 混合场景：先规划再对比
    {
        "user": "用RRT和A*分别规划从(0,0)到(5,5)的路径，然后对比",
        "assistant": "好的，我先用RRT规划一条路径。\n\n<tool_call>rrt_planning(start_x=0, start_y=0, goal_x=5, goal_y=5)</tool_call>\n\n然后用A*规划。\n\n<tool_call>astar_planning(start_x=0, start_y=0, goal_x=5, goal_y=5)</tool_call>"
    },

    # CartPole 控制
    {
        "user": "开始CartPole平衡任务",
        "assistant": "我将启动CartPole仿真环境。\n\n<tool_call>cartpole_reset()</tool_call>"
    },
    {
        "user": "CartPole向右推",
        "assistant": "执行向右推动作。\n\n<tool_call>cartpole_step(action=1)</tool_call>"
    },
    {
        "user": "CartPole向左推",
        "assistant": "执行向左推动作。\n\n<tool_call>cartpole_step(action=0)</tool_call>"
    },
]

def main():
    os.makedirs("dataset", exist_ok=True)

    system_prompt = (
        "你是一个机器人控制领域的AI助手，擅长控制理论、强化学习、路径规划和状态估计。"
        "当需要调用工具时，使用 <tool_call>tool_name(arg=val)</tool_call> 格式。"
    )

    with open(OUTPUT, "w", encoding="utf-8") as f:
        for i, example in enumerate(TOOL_EXAMPLES):
            record = {
                "id": f"tool_{i:03d}",
                "conversations": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": example["user"]},
                    {"role": "assistant", "content": example["assistant"]},
                ]
            }
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"✅ 生成 {len(TOOL_EXAMPLES)} 条工具调用训练数据")
    print(f"保存到: {OUTPUT}")

if __name__ == "__main__":
    main()
