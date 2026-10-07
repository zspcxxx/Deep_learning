#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dllab.backprop import BackpropNN, gradient_check
from dllab.datasets import generate_moons_dataset, generate_xor_dataset
from dllab.io_utils import dump_metrics
from dllab.paths import ensure_artifacts
from dllab.plotting import plot_decision_boundary, plt, save_figure


def run() -> dict:
    out = ensure_artifacts("03_backprop")
    print("=" * 90)
    print("梯度检查（验证反向传播正确性）")
    print("=" * 90)
    X_ck, y_ck = generate_xor_dataset(n_per_quadrant=10, noise=0.15, seed=0)
    model_ck = BackpropNN(n_in=2, n_hidden=4, activation="sigmoid", seed=42)
    results = gradient_check(model_ck, X_ck, y_ck)
    grad_ok = True
    grad_report = {}
    for name in ["W1", "b1", "W2", "b2"]:
        err = results[name]["max_rel_err"]
        ok = err < 1e-6
        grad_ok = grad_ok and ok
        grad_report[name] = {"max_rel_err": err, "passed": ok}
        print(f"  {name:>3s} : max relative error = {err:.3e}   {'✓ 通过' if ok else '✗ 未通过'}")

    X_xor, y_xor = generate_xor_dataset()
    print("\n" + "=" * 90)
    print("训练演示 1：XOR")
    print("=" * 90)
    model_xor = BackpropNN(n_in=2, n_hidden=8, learning_rate=1.5, n_iters=1200, activation="sigmoid", seed=42)
    model_xor.fit(X_xor, y_xor, verbose=True, print_every=300)
    acc_xor = float(np.mean(model_xor.predict(X_xor) == y_xor) * 100)
    print(f"  → 最终准确率: {acc_xor:.2f}%  损失: {model_xor.loss_[-1]:.6f}")

    X_moon, y_moon = generate_moons_dataset()
    print("\n" + "=" * 90)
    print("训练演示 2：Two Moons")
    print("=" * 90)
    model_moon = BackpropNN(n_in=2, n_hidden=16, learning_rate=1.5, n_iters=1500, activation="sigmoid", seed=42)
    model_moon.fit(X_moon, y_moon, verbose=True, print_every=500)
    acc_moon = float(np.mean(model_moon.predict(X_moon) == y_moon) * 100)
    print(f"  → 最终准确率: {acc_moon:.2f}%  损失: {model_moon.loss_[-1]:.6f}")

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    plot_decision_boundary(axes[0][0], model_xor, X_xor, y_xor, f"Backprop XOR (acc={acc_xor:.1f}%)")
    axes[0][1].plot(model_xor.loss_, color="tab:red"); axes[0][1].set_title("XOR loss"); axes[0][1].grid(alpha=0.3)
    axes[0][2].plot(model_xor.accuracy_, color="tab:blue"); axes[0][2].set_title("XOR acc"); axes[0][2].set_ylim(-2, 102); axes[0][2].grid(alpha=0.3)
    plot_decision_boundary(axes[1][0], model_moon, X_moon, y_moon, f"Backprop Moons (acc={acc_moon:.1f}%)")
    axes[1][1].plot(model_moon.loss_, color="tab:red"); axes[1][1].set_title("Moons loss"); axes[1][1].grid(alpha=0.3)
    axes[1][2].plot(model_moon.accuracy_, color="tab:blue"); axes[1][2].set_title("Moons acc"); axes[1][2].set_ylim(-2, 102); axes[1][2].grid(alpha=0.3)
    image = save_figure(fig, out / "backprop_result.png")

    metrics = {
        "gradient_check": grad_report,
        "xor_accuracy": acc_xor,
        "moons_accuracy": acc_moon,
        "images": [str(image.relative_to(out.parent.parent))],
        "passed": grad_ok and acc_xor >= 90 and acc_moon >= 85,
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"结果图已保存: {image}")
    return metrics


if __name__ == "__main__":
    run()
