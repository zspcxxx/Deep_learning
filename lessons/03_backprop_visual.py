#!/usr/bin/env python3
from __future__ import annotations

import numpy as np
from matplotlib.patches import Circle

from dllab.io_utils import dump_metrics
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def run() -> dict:
    out = ensure_artifacts("03_backprop")
    x = np.array([0.5, 0.8])
    y = 1.0
    W1 = np.array([[0.1, 0.2], [0.3, 0.4]])
    b1 = np.array([0.1, -0.1])
    W2 = np.array([[0.5], [0.6]])
    b2 = np.array([0.2])

    z1 = x @ W1 + b1
    a1 = sigmoid(z1)
    z2 = a1 @ W2 + b2
    a2 = np.asarray(sigmoid(z2)).reshape(-1)
    L = float(-(y * np.log(a2[0] + 1e-12) + (1 - y) * np.log(1 - a2[0] + 1e-12)))
    dz2 = np.asarray(a2 - y).reshape(-1)
    dW2 = a1.reshape(-1, 1) * dz2
    db2 = dz2.copy()
    da1 = (dz2.ravel() @ W2.T).ravel()
    dz1 = (da1 * a1 * (1 - a1)).ravel()
    dW1 = x.reshape(-1, 1) * dz1
    db1 = dz1.copy()

    print("=" * 70)
    print("前向传播")
    print("=" * 70)
    print(f"x={x}, z1={z1}, a1={a1}, z2={z2}, a2={a2}, L={L:.4f}")
    print("=" * 70)
    print("反向传播")
    print("=" * 70)
    print(f"dz2={dz2}, dW2={dW2.ravel()}, da1={da1}, dz1={dz1}")

    def draw_network(ax, direction="forward"):
        x_in, x_hid, x_out = 0.12, 0.50, 0.88
        pos_in = [(x_in, 0.30), (x_in, 0.70)]
        pos_hid = [(x_hid, 0.30), (x_hid, 0.70)]
        pos_out = [(x_out, 0.50)]
        color = "#2ca02c" if direction == "forward" else "#d62728"
        pairs = []
        src, dst = (pos_in, pos_hid) if direction == "forward" else (pos_out, pos_hid)
        for p1 in src:
            for p2 in dst:
                pairs.append((p1, p2))
        src2, dst2 = (pos_hid, pos_out) if direction == "forward" else (pos_hid, pos_in)
        for p1 in src2:
            for p2 in dst2:
                pairs.append((p1, p2))
        for (x1, y1), (x2, y2) in pairs:
            ax.annotate("", xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.8, alpha=0.6, shrinkA=12, shrinkB=12))

        def node(px, py, label, facecolor):
            ax.add_patch(Circle((px, py), 0.055, facecolor=facecolor, edgecolor="black", lw=1.5, zorder=5))
            ax.text(px, py, label, ha="center", va="center", fontsize=10, zorder=6, fontweight="bold")

        node(*pos_in[0], f"x1\n{x[0]:.2f}", "lightyellow")
        node(*pos_in[1], f"x2\n{x[1]:.2f}", "lightyellow")
        node(*pos_hid[0], f"h1\n{a1[0]:.2f}", "lightblue")
        node(*pos_hid[1], f"h2\n{a1[1]:.2f}", "lightblue")
        node(*pos_out[0], f"ŷ\n{a2[0]:.2f}", "lightpink")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.axis("off")

    fig, axes = plt.subplots(2, 2, figsize=(15, 11))
    draw_network(axes[0][0], "forward")
    axes[0][0].set_title("Forward pass")
    draw_network(axes[0][1], "backward")
    axes[0][1].set_title("Backward pass")
    axes[1][0].axis("off")
    axes[1][0].text(0.05, 0.6, f"L={L:.4f}\ndz2={dz2[0]:+.4f}\ndz1={dz1}", family="monospace", fontsize=11)
    axes[1][0].set_title("Chain rule numbers")
    norms = [np.linalg.norm(dW1), np.linalg.norm(db1), np.linalg.norm(dW2), np.linalg.norm(db2)]
    axes[1][1].bar(["dW1", "db1", "dW2", "db2"], norms, color=["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"])
    axes[1][1].set_title("Gradient magnitudes")
    image = save_figure(fig, out / "backprop_visualization.png")
    metrics = {
        "loss": L,
        "a2": float(a2[0]),
        "images": [str(image.relative_to(out.parent.parent))],
        "passed": bool(0 < float(a2[0]) < 1 and L > 0),
    }
    dump_metrics(out / "visual_metrics.json", metrics)
    print(f"结果图已保存: {image}")
    return metrics


if __name__ == "__main__":
    run()
