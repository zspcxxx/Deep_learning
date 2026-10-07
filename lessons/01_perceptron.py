#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dllab.datasets import generate_2d_dataset, generate_logic_gate_dataset
from dllab.io_utils import dump_metrics
from dllab.paths import ensure_artifacts
from dllab.perceptron import Perceptron
from dllab.plotting import plt, save_figure


def run() -> dict:
    out = ensure_artifacts("01_perceptron")
    X_gate_base = np.array([[0, 0], [0, 2], [2, 0], [2, 2]], dtype=float)
    y_and_base = np.array([-1, -1, -1, 1])
    y_or_base = np.array([-1, 1, 1, 1])
    y_xor_base = np.array([-1, 1, 1, -1])
    n_per, sigma, hw = 50, 0.30, 0.49

    gates = {
        "AND": generate_logic_gate_dataset(X_gate_base, y_and_base, n_per, sigma, hw, seed=1),
        "OR": generate_logic_gate_dataset(X_gate_base, y_or_base, n_per, sigma, hw, seed=2),
        "XOR": generate_logic_gate_dataset(X_gate_base, y_xor_base, n_per, sigma, hw, seed=3),
    }

    print("=" * 105)
    print(f"逻辑门演示（每门 {4 * n_per} 个样本，截断高斯 σ={sigma}，半宽 ±{hw}）")
    print("=" * 105)

    gate_results = {}
    metrics = {"gates": {}}
    for name, (X_g, y_g) in gates.items():
        print(f"\n[{name}] 开始训练（样本数 = {len(y_g)}）：")
        clf = Perceptron(learning_rate=0.1, n_iters=400 if name == "XOR" else 4000)
        clf.fit(X_g, y_g, verbose=True, print_every=100)
        acc = float(np.mean(clf.predict(X_g) == y_g) * 100)
        converged = clf.errors_[-1] == 0
        print(f"  → {'在第 ' + str(len(clf.errors_)) + ' 轮收敛' if converged else '达到最大轮数，仍有误分类'}")
        print(f"  → 最终权重 w = {clf.w}, 偏置 b = {clf.b:.4f}")
        print(f"  → 训练集准确率: {acc:.2f}%")
        if name == "XOR":
            print("  → XOR 线性不可分，单层感知机无法完美拟合")
        gate_results[name] = (clf, X_g, y_g)
        metrics["gates"][name] = {
            "accuracy": acc,
            "epochs": len(clf.errors_),
            "converged": converged,
            "final_errors": int(clf.errors_[-1]),
        }

    X_2d, y_2d = generate_2d_dataset()
    print("\n" + "=" * 105)
    print(f"二维数据（共 {len(y_2d)} 个样本）")
    print("=" * 105)
    clf_2d = Perceptron(learning_rate=0.1, n_iters=4000)
    clf_2d.fit(X_2d, y_2d, verbose=True, print_every=50)
    acc_2d = float(np.mean(clf_2d.predict(X_2d) == y_2d) * 100)
    print(f"  → 在第 {len(clf_2d.errors_)} 轮收敛，准确率 {acc_2d:.2f}%")
    metrics["two_d"] = {"accuracy": acc_2d, "epochs": len(clf_2d.errors_), "converged": clf_2d.errors_[-1] == 0}

    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    for ax, (name, (clf, X_g, y_g)) in zip(axes[0], gate_results.items()):
        pos, neg = y_g == 1, y_g == -1
        ax.scatter(X_g[pos, 0], X_g[pos, 1], c="red", marker="o", s=18, alpha=0.5, edgecolor="none", label="+1")
        ax.scatter(X_g[neg, 0], X_g[neg, 1], c="blue", marker="x", s=18, alpha=0.5, label="-1")
        if abs(clf.w[1]) > 1e-8:
            x1_vals = np.array([-0.7, 2.7])
            ax.plot(x1_vals, -(clf.w[0] * x1_vals + clf.b) / clf.w[1], "k--", linewidth=2)
        ax.set_xlim(-0.7, 2.7); ax.set_ylim(-0.7, 2.7)
        ax.set_title(f"{name} Gate (epochs={len(clf.errors_)})")
        ax.grid(alpha=0.3); ax.legend(fontsize=8)

    for ax, (name, (clf, _, _)) in zip(axes[1], gate_results.items()):
        ax.plot(range(1, len(clf.errors_) + 1), clf.errors_, linewidth=1.5, color="crimson")
        ax.set_title(f"{name} errors"); ax.grid(alpha=0.3)

    ax = axes[2][0]
    pos, neg = y_2d == 1, y_2d == -1
    ax.scatter(X_2d[pos, 0], X_2d[pos, 1], c="red", marker="o", s=12, alpha=0.4, edgecolor="none")
    ax.scatter(X_2d[neg, 0], X_2d[neg, 1], c="blue", marker="x", s=12, alpha=0.4)
    x1_vals = np.array([X_2d[:, 0].min() - 1, X_2d[:, 0].max() + 1])
    ax.plot(x1_vals, -(clf_2d.w[0] * x1_vals + clf_2d.b) / clf_2d.w[1], "k--", linewidth=2)
    ax.set_title("2D decision boundary"); ax.grid(alpha=0.3)

    axes[2][1].plot(range(1, len(clf_2d.errors_) + 1), clf_2d.errors_, color="green")
    axes[2][1].set_title("2D errors"); axes[2][1].grid(alpha=0.3)

    colors = {"AND": "tab:red", "OR": "tab:blue", "XOR": "tab:purple"}
    for name, (clf, _, _) in gate_results.items():
        axes[2][2].plot(range(1, len(clf.errors_) + 1), clf.errors_, color=colors[name], label=name)
    axes[2][2].plot(range(1, len(clf_2d.errors_) + 1), clf_2d.errors_, color="tab:green", label="2D")
    axes[2][2].legend(); axes[2][2].set_title("Error summary"); axes[2][2].grid(alpha=0.3)

    image = save_figure(fig, out / "perceptron_result.png")
    metrics["images"] = [str(image.relative_to(out.parent.parent))]
    metrics["passed"] = metrics["gates"]["AND"]["converged"] and metrics["gates"]["OR"]["converged"] and not metrics["gates"]["XOR"]["converged"]
    dump_metrics(out / "metrics.json", metrics)
    print(f"结果图已保存: {image}")
    return metrics


if __name__ == "__main__":
    run()
