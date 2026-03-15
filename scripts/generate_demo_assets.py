"""
Generate demo images for README assets.
Run: conda run -n LLM python scripts/generate_demo_assets.py
Output: assets/
"""
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as patches

ASSETS = os.path.join(os.path.dirname(__file__), "..", "assets")
os.makedirs(ASSETS, exist_ok=True)

# ── 1. PID response curve ──────────────────────────────────────────────────
def gen_pid():
    dt, T = 0.01, 5.0
    t = np.arange(0, T, dt)
    kp, ki, kd = 2.0, 0.8, 0.3
    setpoint = 1.0
    y, integral, prev_err = 0.0, 0.0, 0.0
    ys = []
    for _ in t:
        err = setpoint - y
        integral += err * dt
        derivative = (err - prev_err) / dt
        u = kp * err + ki * integral + kd * derivative
        y += u * dt * 0.5
        prev_err = err
        ys.append(y)

    fig, ax = plt.subplots(figsize=(7, 3.5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(t, ys, color="#00d4ff", lw=2, label=f"Response (Kp={kp}, Ki={ki}, Kd={kd})")
    ax.axhline(setpoint, color="#ff6b35", lw=1.5, ls="--", label="Setpoint")
    ax.set_xlabel("Time (s)", color="#8b949e")
    ax.set_ylabel("Output", color="#8b949e")
    ax.set_title("PID Control Simulation", color="#e6edf3", fontsize=13)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "pid_demo.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ pid_demo.png")

# ── 2. RRT path planning ───────────────────────────────────────────────────
def gen_rrt():
    np.random.seed(42)
    fig, ax = plt.subplots(figsize=(6, 6), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")

    # obstacles
    obstacles = [(20, 20, 5), (40, 30, 6), (30, 50, 7), (55, 55, 5), (15, 60, 6)]
    for ox, oy, r in obstacles:
        c = plt.Circle((ox, oy), r, color="#ff6b35", alpha=0.6)
        ax.add_patch(c)

    # fake RRT tree edges
    nodes = [(10, 10)]
    edges = []
    for _ in range(120):
        rx, ry = np.random.uniform(0, 70), np.random.uniform(0, 70)
        dists = [np.hypot(rx - nx, ry - ny) for nx, ny in nodes]
        nearest = nodes[np.argmin(dists)]
        step = 4.0
        angle = np.arctan2(ry - nearest[1], rx - nearest[0])
        nx2 = nearest[0] + step * np.cos(angle)
        ny2 = nearest[1] + step * np.sin(angle)
        # skip if inside obstacle
        if any(np.hypot(nx2 - ox, ny2 - oy) < r + 1 for ox, oy, r in obstacles):
            continue
        nodes.append((nx2, ny2))
        edges.append((nearest, (nx2, ny2)))

    for (x1, y1), (x2, y2) in edges:
        ax.plot([x1, x2], [y1, y2], color="#30363d", lw=0.8)

    # highlight path (fake)
    path = [(10,10),(14,12),(18,16),(22,22),(28,30),(34,38),(42,44),(50,52),(58,58),(63,63)]
    px, py = zip(*path)
    ax.plot(px, py, color="#00d4ff", lw=2.5, label="Path")

    ax.plot(10, 10, "o", color="#3fb950", ms=10, label="Start")
    ax.plot(63, 63, "*", color="#f0c040", ms=14, label="Goal")
    ax.set_xlim(0, 70); ax.set_ylim(0, 70)
    ax.set_title("RRT Path Planning", color="#e6edf3", fontsize=13)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "rrt_demo.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ rrt_demo.png")

# ── 3. Pipeline flow diagram ───────────────────────────────────────────────
def gen_pipeline():
    fig, ax = plt.subplots(figsize=(10, 3), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.axis("off")

    steps = [
        ("Collect\nTrajectories", "#1f6feb"),
        ("DPO\nTraining", "#388bfd"),
        ("Evaluate\nReward", "#3fb950"),
        ("Benchmark\nAccuracy", "#f0c040"),
    ]
    labels_sub = ["DeepSeek generates\ntasks + executes", "LoRA fine-tuning\non preferences",
                  "10 validation\ntasks", "Instruction acc\n+ hallucination rate"]

    n = len(steps)
    xs = np.linspace(0.1, 0.9, n)
    for i, ((label, color), sub) in enumerate(zip(steps, labels_sub)):
        # box
        rect = patches.FancyBboxPatch((xs[i]-0.09, 0.35), 0.18, 0.45,
                                       boxstyle="round,pad=0.02",
                                       facecolor=color, edgecolor="none", alpha=0.85)
        ax.add_patch(rect)
        ax.text(xs[i], 0.575, label, ha="center", va="center",
                color="white", fontsize=10, fontweight="bold")
        ax.text(xs[i], 0.18, sub, ha="center", va="center",
                color="#8b949e", fontsize=7.5)
        # arrow
        if i < n - 1:
            ax.annotate("", xy=(xs[i+1]-0.09, 0.575), xytext=(xs[i]+0.09, 0.575),
                        arrowprops=dict(arrowstyle="->", color="#8b949e", lw=1.5))

    # cycle arrow
    ax.annotate("", xy=(xs[0], 0.35), xytext=(xs[-1], 0.35),
                arrowprops=dict(arrowstyle="->", color="#f0c040", lw=1.5,
                                connectionstyle="arc3,rad=0.4"))
    ax.text(0.5, 0.02, "↻  4-hour cycle", ha="center", color="#f0c040", fontsize=9)

    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Automated Training Pipeline", color="#e6edf3", fontsize=13, pad=8)
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "pipeline_flow.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ pipeline_flow.png")

# ── 4. EKF localization ────────────────────────────────────────────────────
def gen_ekf():
    np.random.seed(7)
    T = 50
    true_x = np.cumsum(np.cos(np.linspace(0, 2*np.pi, T))) * 3
    true_y = np.cumsum(np.sin(np.linspace(0, 2*np.pi, T))) * 3
    meas_x = true_x + np.random.randn(T) * 1.2
    meas_y = true_y + np.random.randn(T) * 1.2
    # simple smoothed estimate
    est_x = np.convolve(meas_x, np.ones(5)/5, mode="same")
    est_y = np.convolve(meas_y, np.ones(5)/5, mode="same")

    fig, ax = plt.subplots(figsize=(6, 5), facecolor="#0d1117")
    ax.set_facecolor("#0d1117")
    ax.plot(true_x, true_y, color="#3fb950", lw=2, label="Ground Truth")
    ax.scatter(meas_x, meas_y, color="#ff6b35", s=15, alpha=0.6, label="Noisy Measurements")
    ax.plot(est_x, est_y, color="#00d4ff", lw=2, ls="--", label="EKF Estimate")
    ax.set_title("EKF Localization", color="#e6edf3", fontsize=13)
    ax.tick_params(colors="#8b949e")
    for spine in ax.spines.values():
        spine.set_edgecolor("#30363d")
    ax.legend(facecolor="#161b22", edgecolor="#30363d", labelcolor="#e6edf3")
    fig.tight_layout()
    fig.savefig(os.path.join(ASSETS, "ekf_demo.png"), dpi=150, bbox_inches="tight")
    plt.close()
    print("✓ ekf_demo.png")


if __name__ == "__main__":
    gen_pid()
    gen_rrt()
    gen_pipeline()
    gen_ekf()
    print(f"\nAll assets saved to: {os.path.abspath(ASSETS)}")
