"""
奖励函数模块 — 为各类机器人控制任务定义奖励信号
用于 agent-lightning 的 emit_reward 接口
"""


def cartpole_reward(steps_survived: int, max_steps: int = 500) -> float:
    """CartPole 平衡任务：存活步数归一化"""
    return steps_survived / max_steps


def pid_reward(overshoot: float, settling_time: float, steady_state_error: float,
               max_settling: float = 10.0) -> float:
    """
    PID 仿真任务奖励（收紧版）
    - 超调越小越好（0% 最优）
    - 调节时间越短越好
    - 稳态误差越小越好
    优秀: 0.85+, 良好: 0.65-0.85, 及格: 0.4-0.65, 差: <0.4
    """
    # 超调惩罚更严格：>50% 直接 0 分
    if overshoot > 50.0:
        overshoot_score = 0.0
    else:
        overshoot_score = max(0.0, 1.0 - overshoot / 30.0)  # 30% 以上快速衰减

    # 调节时间：<2s 满分，>8s 接近 0
    settling_score = max(0.0, 1.0 - settling_time / 8.0)

    # 稳态误差：<0.05 满分，>0.2 接近 0
    sse_score = max(0.0, 1.0 - steady_state_error * 5.0)

    # 加权：超调最重要（0.5），调节时间次之（0.3），稳态误差（0.2）
    return overshoot_score * 0.5 + settling_score * 0.3 + sse_score * 0.2


def path_planning_reward(success: bool, path_length: float,
                         optimal_length: float = None) -> float:
    """
    路径规划任务奖励（收紧版）
    - 规划失败得 0
    - 成功按路径效率评分，无参考长度给 0.5
    """
    if not success:
        return 0.0
    if optimal_length is None or optimal_length <= 0:
        return 0.5
    ratio = optimal_length / max(path_length, 1e-6)
    # 路径效率 >90% 才能拿高分，用平方压缩中间段
    return min(1.0, ratio ** 1.5)


def path_planning_reward_with_coords(success: bool, path_length: float,
                                     start_x: float, start_y: float,
                                     goal_x: float, goal_y: float) -> float:
    """用起终点直线距离作为参考基准"""
    import math
    straight = math.hypot(goal_x - start_x, goal_y - start_y)
    return path_planning_reward(success, path_length, optimal_length=straight)


def tool_call_reward(result: dict, task_type: str) -> float:
    """
    根据工具执行结果和任务类型计算奖励
    统一入口，供 LitAgent 调用
    """
    if result.get("error"):
        return 0.0

    if task_type == "cartpole":
        steps = result.get("steps_survived", 0)
        return cartpole_reward(steps)

    elif task_type == "pid":
        return pid_reward(
            overshoot=result.get("overshoot", 100.0),
            settling_time=result.get("settling_time", 10.0),
            steady_state_error=result.get("steady_state_error", 1.0),
        )

    elif task_type in ("rrt", "astar", "spline"):
        return path_planning_reward(
            success=result.get("success", False),
            path_length=result.get("length", 9999),
        )

    elif task_type == "ekf":
        # EKF：根据预测精度给分（假设 result 有 rmse 字段）
        rmse = result.get("rmse", 999.0)
        if rmse > 10.0:
            return 0.1
        # RMSE < 1.0 优秀，1-5 良好，5-10 及格
        return max(0.1, min(1.0, 1.0 - rmse / 10.0))

    elif task_type == "arm_fk":
        # 机械臂：根据末端位置误差给分（假设 result 有 position_error）
        error = result.get("position_error", 999.0)
        if error > 5.0:
            return 0.1
        # 误差 < 0.5 优秀，0.5-2 良好，2-5 及格
        return max(0.1, min(1.0, 1.0 - error / 5.0))

    return 0.3  # 未知任务类型，给低分（不再给 0.5）
