#!/usr/bin/env python3
from __future__ import annotations

import numpy as np

from dllab.activations import (
    PLOT_FUNCTIONS, SUMMARY_ROWS, TABLE_FUNCTIONS,
    gelu_grad, relu_grad, sigmoid_grad, softmax, tanh_grad,
)
from dllab.io_utils import dump_metrics
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure


def run() -> dict:
    out = ensure_artifacts("04_activations")
    z = np.linspace(-6, 6, 500)
    fig, axes = plt.subplots(4, 4, figsize=(18, 16))
    for ax, (name, f, g) in zip(axes.ravel(), PLOT_FUNCTIONS):
        ax.plot(z, f(z), color="tab:blue", linewidth=2.5, label="f(z)")
        ax.plot(z, g(z), color="tab:red", linewidth=2.0, linestyle="--", label="f'(z)")
        ax.axhline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.axvline(0, color="black", linewidth=0.7, alpha=0.5)
        ax.set_xlim(-6, 6); ax.set_ylim(-2.5, 3.5); ax.set_title(name); ax.legend(fontsize=8); ax.grid(alpha=0.3)
    img1 = save_figure(fig, out / "activation_functions_all.png")

    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    logits = np.array([2.0, 1.0, 0.1, -1.0, -2.0]).reshape(1, -1)
    for T, c in zip([0.5, 1.0, 2.0, 5.0], ["tab:red", "tab:blue", "tab:green", "tab:purple"]):
        axes2[0].plot(range(5), softmax(logits / T).ravel(), marker="o", color=c, label=f"T={T}")
    axes2[0].set_title("Softmax temperatures"); axes2[0].legend(); axes2[0].grid(alpha=0.3)
    probs = softmax(logits).ravel()
    axes2[1].bar(range(5), probs)
    axes2[1].set_title(f"Softmax T=1  sum={probs.sum():.4f}")
    img2 = save_figure(fig2, out / "softmax_demo.png")

    fig3, ax3 = plt.subplots(figsize=(10, 6))

    def grad_chain(grad_func, n, z_test=1.0):
        chain = [1.0]
        for _ in range(n):
            chain.append(chain[-1] * abs(grad_func(np.array([z_test]))[0]))
        return chain

    n_layers = 15
    ax3.semilogy(range(n_layers + 1), grad_chain(sigmoid_grad, n_layers), marker="o", label="Sigmoid")
    ax3.semilogy(range(n_layers + 1), grad_chain(tanh_grad, n_layers), marker="s", label="Tanh")
    ax3.semilogy(range(n_layers + 1), grad_chain(relu_grad, n_layers), marker="^", label="ReLU")
    ax3.semilogy(range(n_layers + 1), grad_chain(gelu_grad, n_layers), marker="D", label="GELU")
    ax3.set_title("Gradient vanishing"); ax3.legend(); ax3.grid(alpha=0.3, which="both")
    img3 = save_figure(fig3, out / "gradient_vanishing.png")

    table = []
    print("=" * 90)
    print("激活函数数值表")
    print("=" * 90)
    for name, func, grad in TABLE_FUNCTIONS:
        vals_f = [float(func(np.array([v]))[0]) for v in [-2, 0, 2]]
        vals_g = [float(grad(np.array([v]))[0]) for v in [-2, 0, 2]]
        table.append({"name": name, "f": vals_f, "grad": vals_g})
        print(f"{name:<14s} {vals_f[0]:8.4f} {vals_f[1]:8.4f} {vals_f[2]:8.4f}")

    metrics = {
        "softmax_sum": float(probs.sum()),
        "table": table,
        "summary": [{"name": n, "range": r, "use": u, "drawback": d} for n, r, u, d in SUMMARY_ROWS],
        "images": [str(p.relative_to(out.parent.parent)) for p in (img1, img2, img3)],
        "passed": bool(abs(float(probs.sum()) - 1.0) < 1e-6),
    }
    dump_metrics(out / "metrics.json", metrics)
    print("激活函数图已保存")
    return metrics


if __name__ == "__main__":
    run()
