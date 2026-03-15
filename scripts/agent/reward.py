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
    PID 仿真任务奖励
    - 超调越小越好（0% 最优）
    - 调节时间越短越好
    - 稳态误差越小越好
    """
    overshoot_score = max(0.0, 1.0 - overshoot / 100.0)
    settling_score = max(0.0, 1.0 - settling_time / max_settling)
    sse_score = max(0.0, 1.0 - steady_state_error * 10.0)
    return (overshoot_score + settling_score + sse_score) / 3.0


def path_planning_reward(success: bool, path_length: float,
                         optimal_length: float = None) -> float:
    """
    路径规划任务奖励
    - 规划失败得 0
    - 成功则按路径长度评分（越短越好）
    """
    if not success:
        return 0.0
    if optimal_length is None or optimal_length <= 0:
        return 0.7  # 成功但无参考长度，给基础分
    ratio = optimal_length / max(path_length, 1e-6)
    return min(1.0, ratio)


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
        # EKF：预测成功即得分
        return 0.8 if result.get("success") else 0.0

    elif task_type == "arm_fk":
        return 0.8 if result.get("success") else 0.0

    return 0.5  # 未知任务类型，给中性分
