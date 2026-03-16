"""
测试新奖励函数的区分度
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "agent"))

from reward import tool_call_reward

# ── PID 测试用例 ──────────────────────────────────────────────────
pid_cases = [
    {"name": "优秀", "overshoot": 5.0, "settling_time": 1.5, "steady_state_error": 0.02},
    {"name": "良好", "overshoot": 15.0, "settling_time": 3.0, "steady_state_error": 0.08},
    {"name": "及格", "overshoot": 25.0, "settling_time": 5.0, "steady_state_error": 0.15},
    {"name": "差", "overshoot": 40.0, "settling_time": 7.0, "steady_state_error": 0.25},
    {"name": "很差", "overshoot": 60.0, "settling_time": 9.0, "steady_state_error": 0.5},
]

print("=" * 60)
print("PID 奖励分布测试")
print("=" * 60)
for case in pid_cases:
    result = {
        "overshoot": case["overshoot"],
        "settling_time": case["settling_time"],
        "steady_state_error": case["steady_state_error"],
    }
    reward = tool_call_reward(result, "pid")
    print(f"{case['name']:6s} | 超调={case['overshoot']:5.1f}% 调节={case['settling_time']:4.1f}s 稳态={case['steady_state_error']:.3f} → reward={reward:.3f}")

# ── 路径规划测试用例 ──────────────────────────────────────────────
import math
path_cases = [
    {"name": "失败", "success": False, "length": 0, "sx": 0, "sy": 0, "gx": 10, "gy": 10},
    {"name": "优秀", "success": True, "length": 14.5, "sx": 0, "sy": 0, "gx": 10, "gy": 10},  # 直线14.14
    {"name": "良好", "success": True, "length": 17.0, "sx": 0, "sy": 0, "gx": 10, "gy": 10},
    {"name": "及格", "success": True, "length": 22.0, "sx": 0, "sy": 0, "gx": 10, "gy": 10},
    {"name": "差", "success": True, "length": 30.0, "sx": 0, "sy": 0, "gx": 10, "gy": 10},
]

print("\n" + "=" * 60)
print("路径规划奖励分布测试")
print("=" * 60)
for case in path_cases:
    straight = math.hypot(case["gx"] - case["sx"], case["gy"] - case["sy"])
    result = {
        "success": case["success"],
        "length": case["length"],
    }
    from reward import path_planning_reward
    reward = path_planning_reward(result["success"], result["length"], optimal_length=straight)
    print(f"{case['name']:6s} | 成功={case['success']} 长度={case['length']:5.1f} 直线={straight:.1f} → reward={reward:.3f}")

# ── EKF 测试用例 ──────────────────────────────────────────────────
ekf_cases = [
    {"name": "优秀", "rmse": 0.5},
    {"name": "良好", "rmse": 2.0},
    {"name": "及格", "rmse": 5.0},
    {"name": "差", "rmse": 8.0},
    {"name": "很差", "rmse": 12.0},
]

print("\n" + "=" * 60)
print("EKF 奖励分布测试")
print("=" * 60)
for case in ekf_cases:
    result = {"rmse": case["rmse"]}
    reward = tool_call_reward(result, "ekf")
    print(f"{case['name']:6s} | RMSE={case['rmse']:5.1f} → reward={reward:.3f}")

# ── CartPole 测试用例 ──────────────────────────────────────────────
cartpole_cases = [
    {"name": "优秀", "steps": 480},
    {"name": "良好", "steps": 350},
    {"name": "及格", "steps": 200},
    {"name": "差", "steps": 80},
    {"name": "很差", "steps": 20},
]

print("\n" + "=" * 60)
print("CartPole 奖励分布测试")
print("=" * 60)
for case in cartpole_cases:
    result = {"steps_survived": case["steps"]}
    reward = tool_call_reward(result, "cartpole")
    print(f"{case['name']:6s} | 存活步数={case['steps']:3d} → reward={reward:.3f}")

# ── 统计分析 ──────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("奖励分布统计")
print("=" * 60)

all_rewards = []
for case in pid_cases:
    r = {"overshoot": case["overshoot"], "settling_time": case["settling_time"], "steady_state_error": case["steady_state_error"]}
    all_rewards.append(tool_call_reward(r, "pid"))

for case in [c for c in path_cases if c["success"]]:
    straight = math.hypot(case["gx"] - case["sx"], case["gy"] - case["sy"])
    from reward import path_planning_reward
    all_rewards.append(path_planning_reward(True, case["length"], optimal_length=straight))

for case in ekf_cases:
    all_rewards.append(tool_call_reward({"rmse": case["rmse"]}, "ekf"))

for case in cartpole_cases:
    all_rewards.append(tool_call_reward({"steps_survived": case["steps"]}, "cartpole"))

import statistics
print(f"样本数: {len(all_rewards)}")
print(f"平均值: {statistics.mean(all_rewards):.3f}")
print(f"中位数: {statistics.median(all_rewards):.3f}")
print(f"标准差: {statistics.stdev(all_rewards):.3f}")
print(f"最小值: {min(all_rewards):.3f}")
print(f"最大值: {max(all_rewards):.3f}")
print(f"\n区分度: {'✓ 良好' if statistics.stdev(all_rewards) > 0.25 else '✗ 不足'}")
