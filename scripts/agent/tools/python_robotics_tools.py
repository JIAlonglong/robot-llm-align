#!/usr/bin/env python3
"""
PythonRobotics 工具适配器
将 PythonRobotics 库的算法封装成 tool_call 接口

使用方式:
    from tools.python_robotics_tools import register_robotics_tools
    registry = ToolRegistry()
    register_robotics_tools(registry)
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')  # 无 GUI 后端
import matplotlib.pyplot as plt
from typing import List, Tuple, Optional
import tempfile
import base64
from io import BytesIO

# 添加 PythonRobotics 到路径
PYTHON_ROBOTICS_PATH = "/home/liujl/big_model/robot-llm-align/PythonRobotics"
if PYTHON_ROBOTICS_PATH not in sys.path:
    sys.path.insert(0, PYTHON_ROBOTICS_PATH)


# ============================================================
# 1. 路径规划工具
# ============================================================

def rrt_planning(
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    obstacle_list: Optional[str] = None,
    rand_area_x: float = 10.0,
    rand_area_y: float = 10.0,
) -> dict:
    """
    RRT 路径规划

    Args:
        start_x, start_y: 起点坐标
        goal_x, goal_y: 终点坐标
        obstacle_list: 障碍物列表，格式 "x1,y1,r1;x2,y2,r2" (圆形障碍物)
        rand_area_x, rand_area_y: 随机采样区域范围

    Returns:
        {"path": [[x,y], ...], "success": bool, "length": float}
    """
    try:
        from PathPlanning.RRT import rrt

        # 解析障碍物
        obstacles = []
        if obstacle_list:
            for obs in obstacle_list.split(";"):
                parts = obs.strip().split(",")
                if len(parts) == 3:
                    obstacles.append([float(parts[0]), float(parts[1]), float(parts[2])])

        # 创建 RRT 规划器
        planner = rrt.RRT(
            start=[start_x, start_y],
            goal=[goal_x, goal_y],
            rand_area=[-rand_area_x, rand_area_x, -rand_area_y, rand_area_y],
            obstacle_list=obstacles,
            expand_dis=0.5,
            path_resolution=0.1,
        )

        # 规划（禁用动画）
        path = planner.planning(animation=False)

        if path is None:
            return {"success": False, "error": "未找到路径"}

        # 计算路径长度
        length = sum(
            np.hypot(path[i+1][0] - path[i][0], path[i+1][1] - path[i][1])
            for i in range(len(path) - 1)
        )

        path_xy = [[round(float(p[0]), 3), round(float(p[1]), 3)] for p in path]

        # 绘图
        fig, ax = plt.subplots(figsize=(6, 6))
        # 树边
        for node in planner.node_list:
            if node.parent is not None:
                ax.plot([node.x, node.parent.x], [node.y, node.parent.y],
                        "-g", linewidth=0.5, alpha=0.4)
        # 障碍物
        for ox, oy, size in obstacles:
            circle = plt.Circle((ox, oy), size, color="k", alpha=0.6)
            ax.add_patch(circle)
        # 路径
        px, py = zip(*path)
        ax.plot(px, py, "-r", linewidth=2, label="Path")
        ax.plot(start_x, start_y, "bs", markersize=8, label="Start")
        ax.plot(goal_x, goal_y, "g*", markersize=12, label="Goal")
        ax.set_xlim(-rand_area_x, rand_area_x)
        ax.set_ylim(-rand_area_y, rand_area_y)
        ax.legend(); ax.grid(True); ax.set_title(f"RRT Planning (len={length:.2f})")
        buf = BytesIO(); plt.savefig(buf, format="png", dpi=100, bbox_inches="tight"); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode(); plt.close()

        return {
            "success": True,
            "path": path_xy,
            "length": round(float(length), 3),
            "num_nodes": len(planner.node_list),
            "plot_base64": img_b64,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def astar_planning(
    start_x: float,
    start_y: float,
    goal_x: float,
    goal_y: float,
    grid_size: float = 1.0,
    robot_radius: float = 0.5,
    obstacle_list: Optional[str] = None,
) -> dict:
    """
    A* 网格路径规划

    Args:
        start_x, start_y: 起点
        goal_x, goal_y: 终点
        grid_size: 网格分辨率
        robot_radius: 机器人半径
        obstacle_list: 障碍物 "x1,y1;x2,y2;..."

    Returns:
        {"path": [[x,y], ...], "success": bool}
    """
    try:
        from PathPlanning.AStar import a_star

        # 解析障碍物
        ox, oy = [], []
        if obstacle_list:
            for obs in obstacle_list.split(";"):
                parts = obs.strip().split(",")
                if len(parts) == 2:
                    ox.append(float(parts[0]))
                    oy.append(float(parts[1]))

        # 如果没有障碍物，添加边界
        if not ox:
            for i in range(-10, 11):
                ox.extend([i, i, -10, 10])
                oy.extend([-10, 10, i, i])

        planner = a_star.AStarPlanner(ox, oy, grid_size, robot_radius)
        rx, ry = planner.planning(start_x, start_y, goal_x, goal_y)

        if not rx:
            return {"success": False, "error": "未找到路径"}

        path = [[round(x, 3), round(y, 3)] for x, y in zip(rx, ry)]

        # 绘图
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(ox, oy, ".k", markersize=4, label="Obstacle")
        ax.plot([p[0] for p in path], [p[1] for p in path], "-r", linewidth=2, label="Path")
        ax.plot(start_x, start_y, "bs", markersize=8, label="Start")
        ax.plot(goal_x, goal_y, "g*", markersize=12, label="Goal")
        ax.legend(); ax.grid(True); ax.set_title(f"A* Planning ({len(path)} steps)")
        ax.set_aspect("equal")
        buf = BytesIO(); plt.savefig(buf, format="png", dpi=100, bbox_inches="tight"); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode(); plt.close()

        return {"success": True, "path": path, "length": len(path), "plot_base64": img_b64}

    except Exception as e:
        return {"success": False, "error": str(e)}


def cubic_spline_planning(waypoints: str) -> dict:
    """
    三次样条曲线轨迹生成

    Args:
        waypoints: 路径点 "x1,y1;x2,y2;x3,y3;..."

    Returns:
        {"path": [[x,y,yaw,curvature], ...], "success": bool}
    """
    try:
        from PathPlanning.CubicSpline import cubic_spline_planner

        # 解析路径点
        ax, ay = [], []
        for wp in waypoints.split(";"):
            parts = wp.strip().split(",")
            if len(parts) == 2:
                ax.append(float(parts[0]))
                ay.append(float(parts[1]))

        if len(ax) < 2:
            return {"success": False, "error": "至少需要 2 个路径点"}

        # 生成样条曲线
        cx, cy, cyaw, ck, s = cubic_spline_planner.calc_spline_course(ax, ay, ds=0.1)

        path = [
            [round(float(x), 3), round(float(y), 3), round(float(yaw), 3), round(float(k), 3)]
            for x, y, yaw, k in zip(cx, cy, cyaw, ck)
        ]

        # 绘图
        fig, axes = plt.subplots(1, 2, figsize=(10, 4))
        axes[0].plot(cx, cy, "-r", linewidth=2, label="Spline")
        axes[0].plot(ax, ay, "ob", markersize=8, label="Waypoints")
        axes[0].legend(); axes[0].grid(True); axes[0].set_title("Cubic Spline Path")
        axes[0].set_aspect("equal")
        axes[1].plot(s, np.degrees(cyaw), "-g", label="Yaw (deg)")
        axes[1].plot(s, ck, "-b", label="Curvature")
        axes[1].legend(); axes[1].grid(True); axes[1].set_title("Yaw & Curvature")
        axes[1].set_xlabel("Arc length")
        buf = BytesIO(); plt.savefig(buf, format="png", dpi=100, bbox_inches="tight"); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode(); plt.close()

        return {"success": True, "path": path, "length": round(float(s[-1]), 3), "plot_base64": img_b64}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# 2. 控制工具
# ============================================================

def lqr_steering_control(
    x: float,
    y: float,
    yaw: float,
    v: float,
    ref_path: str,
    lookahead: float = 5.0,
) -> dict:
    """
    LQR 转向控制

    Args:
        x, y, yaw, v: 当前状态 (位置、航向、速度)
        ref_path: 参考路径 "x1,y1,yaw1;x2,y2,yaw2;..."
        lookahead: 前视距离

    Returns:
        {"steering": float, "target_idx": int, "error": float}
    """
    try:
        from PathTracking.lqr_steer_control import lqr_steer_control

        # 解析参考路径
        cx, cy, cyaw = [], [], []
        for pt in ref_path.split(";"):
            parts = pt.strip().split(",")
            if len(parts) >= 3:
                cx.append(float(parts[0]))
                cy.append(float(parts[1]))
                cyaw.append(float(parts[2]))

        if len(cx) < 2:
            return {"success": False, "error": "参考路径至少需要 2 个点"}

        # 计算曲率（简化）
        ck = [0.0] * len(cx)

        # LQR 控制
        state = lqr_steer_control.State(x=x, y=y, yaw=yaw, v=v)
        pe, pth_e = 0.0, 0.0  # 误差初始化

        delta, target_idx, e, th_e = lqr_steer_control.lqr_steering_control(
            state, cx, cy, cyaw, ck, pe, pth_e
        )

        return {
            "success": True,
            "steering": round(delta, 4),
            "target_idx": target_idx,
            "cross_track_error": round(e, 4),
            "heading_error": round(th_e, 4),
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def mpc_control(
    x: float,
    y: float,
    yaw: float,
    v: float,
    ref_path: str,
    horizon: int = 10,
) -> dict:
    """
    MPC 模型预测控制（倒立摆场景）

    Args:
        x: 位置
        y: 速度
        yaw: 角度
        v: 角速度
        ref_path: 参考轨迹（暂未使用，保留接口）
        horizon: 预测时域

    Returns:
        {"control": [u1, u2, ...], "predicted_trajectory": [[x,y], ...]}
    """
    try:
        from InvertedPendulum import inverted_pendulum_mpc_control as mpc

        # 简化：直接用倒立摆 MPC
        x0 = np.array([x, y, yaw, v])

        # 调用 MPC（需要 cvxpy）
        # 注意：原始代码可能需要修改以支持参数化调用
        return {
            "success": True,
            "control": [0.0],  # 占位，需要实际实现
            "note": "MPC 需要进一步适配，当前为占位实现"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# 3. 定位/滤波工具
# ============================================================

def ekf_localization(
    state: str,
    control: str,
    measurement: str,
) -> dict:
    """
    扩展卡尔曼滤波定位

    Args:
        state: 当前状态 "x,y,yaw"
        control: 控制输入 "v,omega"
        measurement: 观测 "x_obs,y_obs"

    Returns:
        {"estimated_state": [x, y, yaw], "covariance": [[...], ...]}
    """
    try:
        from Localization.extended_kalman_filter import extended_kalman_filter as ekf

        # 解析输入 — EKF 需要列向量，state 为 4 维 [x,y,yaw,v]
        parts = state.split(",")
        if len(parts) == 3:
            parts.append("1.0")  # 补速度默认值
        x = np.array([[float(v)] for v in parts])              # (4,1)
        u = np.array([[float(v)] for v in control.split(",")]) # (2,1)
        z = np.array([[float(v)] for v in measurement.split(",")]) # (2,1)

        # 运动模型
        x_pred = ekf.motion_model(x, u)
        z_pred = ekf.observation_model(x_pred)

        # 绘图：显示真实状态、预测状态、观测
        fig, ax = plt.subplots(figsize=(5, 5))
        xf, xpf = x.flatten(), x_pred.flatten()
        zf = z.flatten()
        ax.plot(xf[0], xf[1], "bs", markersize=10, label=f"Prior ({xf[0]:.2f},{xf[1]:.2f})")
        ax.plot(xpf[0], xpf[1], "r^", markersize=10, label=f"Predicted ({xpf[0]:.2f},{xpf[1]:.2f})")
        ax.plot(zf[0], zf[1], "g*", markersize=12, label=f"Measurement ({zf[0]:.2f},{zf[1]:.2f})")
        ax.annotate("", xy=(xpf[0], xpf[1]), xytext=(xf[0], xf[1]),
                    arrowprops=dict(arrowstyle="->", color="red", lw=2))
        ax.legend(); ax.grid(True); ax.set_title("EKF Localization Step")
        ax.set_aspect("equal")
        buf = BytesIO(); plt.savefig(buf, format="png", dpi=100, bbox_inches="tight"); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode(); plt.close()

        return {
            "success": True,
            "predicted_state": [round(float(v), 4) for v in x_pred.flatten().tolist()],
            "predicted_measurement": [round(float(v), 4) for v in z_pred.flatten().tolist()],
            "plot_base64": img_b64,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# 4. 机械臂工具
# ============================================================

def arm_forward_kinematics(
    joint_angles: str,
    link_lengths: str = "1.0,1.0,1.0",
) -> dict:
    """
    机械臂正运动学

    Args:
        joint_angles: 关节角度 "theta1,theta2,theta3,..." (弧度)
        link_lengths: 连杆长度 "L1,L2,L3,..."

    Returns:
        {"end_effector": [x, y], "joint_positions": [[x,y], ...]}
    """
    try:
        from ArmNavigation.n_joint_arm_to_point_control import NLinkArm

        # 解析输入
        angles = [float(a) for a in joint_angles.split(",")]
        lengths = [float(l) for l in link_lengths.split(",")]

        if len(angles) != len(lengths):
            return {"success": False, "error": "关节角度和连杆长度数量不匹配"}

        # 创建机械臂
        arm = NLinkArm.NLinkArm(lengths, angles, [0, 0], show_animation=False)
        arm.update_joints(angles)

        # 获取末端位置和所有关节位置
        end_effector = arm.end_effector
        joint_positions = arm.points

        # 绘图
        fig, ax = plt.subplots(figsize=(5, 5))
        pts = [[round(float(p[0]), 4), round(float(p[1]), 4)] for p in joint_positions]
        xs = [p[0] for p in pts]
        ys = [p[1] for p in pts]
        ax.plot(xs, ys, "o-b", linewidth=3, markersize=8, label="Arm")
        ax.plot(xs[0], ys[0], "gs", markersize=12, label="Base")
        ax.plot(xs[-1], ys[-1], "r*", markersize=14, label="End-effector")
        total = sum(lengths)
        ax.set_xlim(-total - 0.5, total + 0.5)
        ax.set_ylim(-total - 0.5, total + 0.5)
        ax.set_aspect("equal"); ax.grid(True); ax.legend()
        ax.set_title(f"Arm FK — EE=({xs[-1]:.3f}, {ys[-1]:.3f})")
        buf = BytesIO(); plt.savefig(buf, format="png", dpi=100, bbox_inches="tight"); buf.seek(0)
        img_b64 = base64.b64encode(buf.read()).decode(); plt.close()

        return {
            "success": True,
            "end_effector": [round(float(end_effector[0]), 4), round(float(end_effector[1]), 4)],
            "joint_positions": pts,
            "num_joints": len(angles),
            "plot_base64": img_b64,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# 5. 仿真工具
# ============================================================

def simulate_pid(
    kp: float,
    ki: float = 0.0,
    kd: float = 0.0,
    setpoint: float = 1.0,
    duration: float = 10.0,
) -> dict:
    """
    PID 控制器仿真（一阶系统）

    Args:
        kp, ki, kd: PID 参数
        setpoint: 目标值
        duration: 仿真时长（秒）

    Returns:
        {"overshoot": float, "settling_time": float, "steady_state_error": float, "plot": str}
    """
    try:
        dt = 0.01
        steps = int(duration / dt)

        # 一阶系统: dx/dt = -x + u
        x = 0.0
        integral = 0.0
        prev_error = 0.0

        time_data = []
        output_data = []
        control_data = []

        for i in range(steps):
            t = i * dt
            error = setpoint - x
            integral += error * dt
            derivative = (error - prev_error) / dt

            u = kp * error + ki * integral + kd * derivative
            x += dt * (-x + u)

            time_data.append(t)
            output_data.append(x)
            control_data.append(u)
            prev_error = error

        # 计算性能指标
        output_arr = np.array(output_data)
        overshoot = max(0, (output_arr.max() - setpoint) / setpoint * 100)

        # 稳态时间（2% 误差带）
        settling_idx = np.where(np.abs(output_arr - setpoint) < 0.02 * setpoint)[0]
        settling_time = time_data[settling_idx[0]] if len(settling_idx) > 0 else duration

        # 稳态误差
        sse = abs(output_data[-1] - setpoint)

        # 绘图
        plt.figure(figsize=(10, 4))
        plt.subplot(1, 2, 1)
        plt.plot(time_data, output_data, label='Output')
        plt.axhline(setpoint, color='r', linestyle='--', label='Setpoint')
        plt.xlabel('Time (s)')
        plt.ylabel('Output')
        plt.legend()
        plt.grid(True)
        plt.title(f'PID Response (Kp={kp}, Ki={ki}, Kd={kd})')

        plt.subplot(1, 2, 2)
        plt.plot(time_data, control_data, label='Control Signal', color='orange')
        plt.xlabel('Time (s)')
        plt.ylabel('Control')
        plt.legend()
        plt.grid(True)

        # 保存为 base64
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()

        return {
            "success": True,
            "overshoot": round(float(overshoot), 2),
            "settling_time": round(float(settling_time), 3),
            "steady_state_error": round(float(sse), 4),
            "plot_base64": img_base64,
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def particle_filter_localization(
    initial_state: str,
    measurements: str,
    num_particles: int = 100,
) -> dict:
    """
    粒子滤波定位

    Args:
        initial_state: 初始状态 "x,y,yaw"
        measurements: 观测序列 "x1,y1;x2,y2;..."
        num_particles: 粒子数量

    Returns:
        {"estimated_trajectory": [[x,y,yaw], ...], "plot": str}
    """
    try:
        from Localization.particle_filter import particle_filter as pf

        # 解析输入
        x0 = np.array([float(v) for v in initial_state.split(",")])
        z_list = []
        for z_str in measurements.split(";"):
            z_list.append(np.array([float(v) for v in z_str.split(",")]))

        # 简化实现：返回占位结果
        return {
            "success": True,
            "estimated_trajectory": [[round(float(x0[0]), 3), round(float(x0[1]), 3), round(float(x0[2]), 3)]],
            "note": "粒子滤波需要完整实现，当前为占位"
        }

    except Exception as e:
        return {"success": False, "error": str(e)}


def plot_path_comparison(
    paths: str,
    labels: str,
    title: str = "Path Comparison",
) -> dict:
    """
    绘制多条路径对比图

    Args:
        paths: 路径数据 "path1_x1,y1;x2,y2|path2_x1,y1;x2,y2"
        labels: 标签 "label1,label2"
        title: 图表标题

    Returns:
        {"plot_base64": str}
    """
    try:
        path_list = []
        for path_str in paths.split("|"):
            path = []
            for pt in path_str.split(";"):
                parts = pt.strip().split(",")
                if len(parts) == 2:
                    path.append([float(parts[0]), float(parts[1])])
            path_list.append(path)

        label_list = labels.split(",")

        plt.figure(figsize=(8, 6))
        for path, label in zip(path_list, label_list):
            if path:
                xs, ys = zip(*path)
                plt.plot(xs, ys, marker='o', label=label, linewidth=2)

        plt.xlabel('X')
        plt.ylabel('Y')
        plt.title(title)
        plt.legend()
        plt.grid(True)
        plt.axis('equal')

        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        img_base64 = base64.b64encode(buf.read()).decode()
        plt.close()

        return {"success": True, "plot_base64": img_base64}

    except Exception as e:
        return {"success": False, "error": str(e)}


# ============================================================
# 6. 注册所有工具
# ============================================================

def register_robotics_tools(registry):
    """将所有 PythonRobotics 工具注册到 ToolRegistry"""

    # 路径规划
    registry.register("rrt_planning", rrt_planning)
    registry.register("astar_planning", astar_planning)
    registry.register("cubic_spline_planning", cubic_spline_planning)

    # 控制
    registry.register("lqr_steering_control", lqr_steering_control)
    registry.register("mpc_control", mpc_control)

    # 定位
    registry.register("ekf_localization", ekf_localization)
    registry.register("particle_filter_localization", particle_filter_localization)

    # 机械臂
    registry.register("arm_forward_kinematics", arm_forward_kinematics)

    # 仿真
    registry.register("simulate_pid", simulate_pid)

    # 可视化
    registry.register("plot_path_comparison", plot_path_comparison)

    print("[PythonRobotics] 已注册 11 个工具")


# ============================================================
# 6. 测试入口
# ============================================================

if __name__ == "__main__":
    # 简单测试
    print("测试 RRT 规划...")
    result = rrt_planning(0, 0, 5, 5, obstacle_list="2,2,0.5;3,3,0.5")
    print(result)

    print("\n测试机械臂正运动学...")
    result = arm_forward_kinematics("0.5,0.5,0.5", "1.0,1.0,1.0")
    print(result)

    print("\n测试三次样条...")
    result = cubic_spline_planning("0,0;1,2;3,3;5,2")
    print(result)
