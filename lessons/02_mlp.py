#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dllab.datasets import generate_moons_dataset, generate_xor_dataset
from dllab.io_utils import dump_metrics
from dllab.mlp import MLP
from dllab.paths import ensure_artifacts
from dllab.plotting import plot_decision_boundary, plt, save_figure


def run() -> dict:
    out = ensure_artifacts("02_mlp")
    X_xor, y_xor = generate_xor_dataset()
    print("=" * 90)
    print("多层感知机演示 - XOR")
    print("=" * 90)
    mlp_xor = MLP([2, 8, 8, 1], learning_rate=0.8, n_iters=1200, activation="relu", seed=42)
    mlp_xor.fit(X_xor, y_xor, verbose=True, print_every=300)
    acc_xor = float(np.mean(mlp_xor.predict(X_xor) == y_xor) * 100)
    print(f"  → 最终训练准确率: {acc_xor:.2f}%  损失: {mlp_xor.loss_[-1]:.6f}")

    X_moon, y_moon = generate_moons_dataset()
    print("\n" + "=" * 90)
    print("多层感知机演示 - Two Moons")
    print("=" * 90)
    mlp_moon = MLP([2, 16, 16, 1], learning_rate=0.8, n_iters=1500, activation="relu", seed=42)
    mlp_moon.fit(X_moon, y_moon, verbose=True, print_every=500)
    acc_moon = float(np.mean(mlp_moon.predict(X_moon) == y_moon) * 100)
    print(f"  → 最终训练准确率: {acc_moon:.2f}%  损失: {mlp_moon.loss_[-1]:.6f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_decision_boundary(axes[0][0], mlp_xor, X_xor, y_xor, f"MLP on XOR (acc={acc_xor:.1f}%)")
    axes[0][1].plot(mlp_xor.loss_, color="tab:red"); axes[0][1].set_title("XOR loss"); axes[0][1].grid(alpha=0.3)
    axes[0][2].plot(mlp_xor.accuracy_, color="tab:blue"); axes[0][2].set_title("XOR acc"); axes[0][2].set_ylim(-2, 102); axes[0][2].grid(alpha=0.3)
    plot_decision_boundary(axes[1][0], mlp_moon, X_moon, y_moon, f"MLP on Moons (acc={acc_moon:.1f}%)")
    axes[1][1].plot(mlp_moon.loss_, color="tab:red"); axes[1][1].set_title("Moons loss"); axes[1][1].grid(alpha=0.3)
    axes[1][2].plot(mlp_moon.accuracy_, color="tab:blue"); axes[1][2].set_title("Moons acc"); axes[1][2].set_ylim(-2, 102); axes[1][2].grid(alpha=0.3)
    image = save_figure(fig, out / "mlp_result.png")

    metrics = {
        "xor_accuracy": acc_xor,
        "xor_loss": mlp_xor.loss_[-1],
        "moons_accuracy": acc_moon,
        "moons_loss": mlp_moon.loss_[-1],
        "images": [str(image.relative_to(out.parent.parent))],
        "passed": acc_xor >= 95 and acc_moon >= 90,
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"结果图已保存: {image}")
    return metrics


if __name__ == "__main__":
    run()
