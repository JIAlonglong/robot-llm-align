"""
run_agent.py — 启动入口
用法：
  python scripts/agent/run_agent.py [--host 127.0.0.1] [--port 5555] [--episodes 3]

前置条件：
  Webots 已启动并加载 webots_worlds/cartpole.wbt
  （见 README 或 scripts/start_webots.sh）
"""
import sys
import argparse

# 把 scripts/agent 加入路径
sys.path.insert(0, __file__.rsplit("/", 1)[0])

from tools.webots_tool import WebotsClient
from tool_registry import ToolRegistry
from agent_executor import AgentExecutor


def build_registry(client: WebotsClient) -> ToolRegistry:
    reg = ToolRegistry()
    reg.register("cartpole_reset", client.reset)
    reg.register("cartpole_step",  client.step)
    reg.register("cartpole_ping",  client.ping)
    return reg


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host",     default="127.0.0.1")
    parser.add_argument("--port",     type=int, default=5555)
    parser.add_argument("--episodes", type=int, default=3)
    args = parser.parse_args()

    client = WebotsClient(host=args.host, port=args.port)
    client.connect()

    # 连通性检查
    pong = client.ping()
    print(f"[Ping] {pong}")

    registry = build_registry(client)
    agent    = AgentExecutor(registry, max_steps=500)

    results = []
    for ep in range(1, args.episodes + 1):
        print(f"\n\n{'#'*60}")
        print(f"# Episode {ep}/{args.episodes}")
        print(f"{'#'*60}")
        result = agent.run(task="balance the cartpole")
        results.append(result)

    # 汇总
    avg_steps  = sum(r["steps_survived"] for r in results) / len(results)
    avg_reward = sum(r["total_reward"]   for r in results) / len(results)
    print(f"\n{'='*60}")
    print(f"汇总 ({args.episodes} episodes)")
    print(f"  平均存活步数: {avg_steps:.1f}")
    print(f"  平均总奖励:   {avg_reward:.1f}")
    print(f"{'='*60}")

    client.close()


if __name__ == "__main__":
    main()
