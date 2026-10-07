#!/usr/bin/env python3
"""Compare how different learning rates affect training."""
from __future__ import annotations

import numpy as np

from dllab.datasets import generate_xor_dataset
from dllab.io_utils import dump_metrics
from dllab.mlp import MLP
from dllab.paths import ensure_artifacts
from dllab.plotting import plot_decision_boundary, plt, save_figure

# Same init / architecture; only lr changes.
LAYER_SIZES = [2, 8, 8, 1]
N_ITERS = 600
SEED = 42
LEARNING_RATES = [
    (0.001, "too small"),
    (0.5, "good"),
    (8.0, "too large"),
]


def gd_1d(lr: float, n_steps: int = 25, w0: float = -3.0) -> list[float]:
    """Minimize L(w)=(w-2)^2 with gradient descent; grad = 2(w-2)."""
    ws = [w0]
    for _ in range(n_steps):
        w = ws[-1]
        ws.append(w - lr * 2.0 * (w - 2.0))
    return ws


def run() -> dict:
    out = ensure_artifacts("05_learning_rate")
    X, y = generate_xor_dataset(n_per_quadrant=100, noise=0.15, seed=0)

    print("=" * 90)
    print("学习率对比实验（同一 XOR 数据、同一网络结构与随机种子）")
    print("=" * 90)
    print(f"结构 {LAYER_SIZES}，迭代 {N_ITERS}，seed={SEED}")
    print(f"学习率: {[lr for lr, _ in LEARNING_RATES]}")

    results = {}
    for lr, label in LEARNING_RATES:
        print(f"\n--- lr = {lr} ({label}) ---")
        model = MLP(LAYER_SIZES, learning_rate=lr, n_iters=N_ITERS, activation="relu", seed=SEED)
        model.fit(X, y, verbose=True, print_every=200)
        # Guard against NaN from exploding lr
        loss_curve = [float(v) if np.isfinite(v) else float("nan") for v in model.loss_]
        acc_curve = [float(v) if np.isfinite(v) else 0.0 for v in model.accuracy_]
        final_loss = loss_curve[-1]
        weights_ok = all(np.all(np.isfinite(w)) for w in model.W)
        if weights_ok and np.isfinite(final_loss):
            final_acc = float(np.mean(model.predict(X) == y) * 100)
        else:
            final_acc = 0.0
        diverged = (not weights_ok) or (not np.isfinite(final_loss)) or final_loss > 5.0
        print(f"  → 最终 loss={final_loss}  acc={final_acc:.2f}%  diverged={diverged}")
        results[lr] = {
            "label": label,
            "loss": loss_curve,
            "accuracy": acc_curve,
            "final_loss": final_loss if np.isfinite(final_loss) else None,
            "final_acc": final_acc,
            "diverged": diverged,
            "model": model,
        }

    # ---- Figure 1: loss + accuracy curves ----
    colors = {0.001: "#3b82f6", 0.5: "#0f766e", 8.0: "#be123c"}
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for lr, label in LEARNING_RATES:
        r = results[lr]
        axes[0].plot(r["loss"], color=colors[lr], linewidth=1.8, label=f"lr={lr} ({label})")
        axes[1].plot(r["accuracy"], color=colors[lr], linewidth=1.8, label=f"lr={lr} ({label})")
    axes[0].set(xlabel="Epoch", ylabel="BCE Loss", title="Loss vs learning rate")
    finite_losses = [v for r in results.values() for v in r["loss"] if np.isfinite(v)]
    ymax = min(3.0, max(1.0, max(finite_losses) * 1.1)) if finite_losses else 3.0
    axes[0].set_ylim(0, ymax)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy vs learning rate")
    axes[1].set_ylim(-2, 102)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    img_curves = save_figure(fig, out / "lr_curves.png")

    # ---- Figure 2: decision boundaries ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (lr, label) in zip(axes2, LEARNING_RATES):
        r = results[lr]
        if r["diverged"] or not np.isfinite(r["final_loss"] or np.nan):
            ax.text(0.5, 0.5, f"lr={lr}\ndiverged", ha="center", va="center", fontsize=14, color="#be123c",
                    transform=ax.transAxes)
            ax.set_title(f"lr={lr} · {label}")
            ax.axis("off")
        else:
            plot_decision_boundary(
                ax, r["model"], X, y,
                f"lr={lr}  acc={r['final_acc']:.1f}%\n{label}",
            )
    img_bounds = save_figure(fig2, out / "lr_boundaries.png")

    # ---- Figure 3: 1D toy landscape ----
    w_grid = np.linspace(-4, 6, 400)
    loss_grid = (w_grid - 2) ** 2
    toy_lrs = [(0.05, "too small"), (0.4, "good"), (1.05, "too large")]
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.2))
    toy_paths = {}
    for ax, (lr, tag) in zip(axes3, toy_lrs):
        path = gd_1d(lr)
        toy_paths[lr] = path
        ax.plot(w_grid, loss_grid, color="#64748b", linewidth=2, label="L(w)=(w-2)^2")
        ax.plot(path, [(w - 2) ** 2 for w in path], "o-", color="#1f6feb", markersize=4, linewidth=1.5, label="GD path")
        ax.axvline(2, color="#0f766e", linestyle="--", alpha=0.6, label="optimum w=2")
        ax.set(xlabel="w", ylabel="L(w)", title=f"1D GD  lr={lr} ({tag})")
        ax.legend(fontsize=8); ax.grid(alpha=0.3)
        ax.set_ylim(-0.5, 30)
    img_toy = save_figure(fig3, out / "lr_1d_landscape.png")

    slow = results[0.001]
    good = results[0.5]
    bad = results[8.0]
    # Good should beat too-small; too-large should diverge or stay clearly worse.
    passed = (
        good["final_acc"] >= 90
        and good["final_acc"] > slow["final_acc"] + 5
        and (bad["diverged"] or bad["final_acc"] < good["final_acc"] - 10)
    )

    summary_rows = []
    for lr, label in LEARNING_RATES:
        r = results[lr]
        summary_rows.append({
            "lr": lr,
            "label": label,
            "final_acc": r["final_acc"],
            "final_loss": r["final_loss"],
            "diverged": r["diverged"],
        })
        print(f"  lr={lr:<4}  acc={r['final_acc']:6.2f}%  loss={r['final_loss']}  diverged={r['diverged']}")

    metrics = {
        "learning_rates": summary_rows,
        "toy_1d": {str(lr): {"final_w": path[-1], "steps": len(path) - 1} for lr, path in toy_paths.items()},
        "images": [
            str(img_curves.relative_to(out.parent.parent)),
            str(img_bounds.relative_to(out.parent.parent)),
            str(img_toy.relative_to(out.parent.parent)),
        ],
        "passed": bool(passed),
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
