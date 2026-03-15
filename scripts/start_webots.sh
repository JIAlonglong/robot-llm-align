#!/bin/bash
# start_webots.sh — 在远程服务器上启动 Webots (streaming 模式)
# 本地访问：ssh -L 1234:localhost:1234 user@server
# 然后浏览器打开 http://localhost:1234

set -e

REPO_DIR="$(cd "$(dirname "$0")/.." && pwd)"
WORLD="$REPO_DIR/webots_worlds/cartpole.wbt"
DISPLAY_NUM=99
STREAM_PORT=1234

# ── 1. 启动虚拟显示 ───────────────────────────────────────
if ! pgrep -f "Xvfb :$DISPLAY_NUM" > /dev/null; then
    echo "[1/3] 启动 Xvfb :$DISPLAY_NUM ..."
    Xvfb :$DISPLAY_NUM -screen 0 1280x1024x24 &
    sleep 1
else
    echo "[1/3] Xvfb :$DISPLAY_NUM 已在运行"
fi
export DISPLAY=:$DISPLAY_NUM

# ── 2. 把 controllers 目录告诉 Webots ────────────────────
export WEBOTS_HOME=/usr/local/webots
export WEBOTS_CONTROLLER_URL="tcp://localhost:5555"

# ── 3. 启动 Webots streaming ─────────────────────────────
echo "[2/3] 启动 Webots (streaming port $STREAM_PORT) ..."
echo "      世界文件: $WORLD"
webots \
    --stream="port=$STREAM_PORT" \
    --no-sandbox \
    --batch \
    "$WORLD" &

WEBOTS_PID=$!
echo "[3/3] Webots PID=$WEBOTS_PID"
echo ""
echo "========================================"
echo "  本地查看仿真画面："
echo "  ssh -L $STREAM_PORT:localhost:$STREAM_PORT user@$(hostname -I | awk '{print $1}')"
echo "  浏览器打开 http://localhost:$STREAM_PORT"
echo ""
echo "  Agent TCP 端口: 5555"
echo "  运行 Agent："
echo "  python scripts/agent/run_agent.py"
echo "========================================"

wait $WEBOTS_PID
