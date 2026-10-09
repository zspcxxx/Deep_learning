#!/usr/bin/env python3
"""Compare input standardization and batch normalization on poorly scaled features."""
from __future__ import annotations

import numpy as np

from dllab.datasets import generate_moons_dataset
from dllab.io_utils import dump_metrics
from dllab.mlp import MLP
from dllab.paths import ensure_artifacts
from dllab.plotting import plot_decision_boundary, plt, save_figure

LAYER_SIZES = [2, 16, 16, 1]
N_ITERS = 700
LR = 0.35
SEED = 42
# One axis large, one small — raw SGD struggles; input std / BN recover.
SCALE = np.array([10.0, 0.15])
SHIFT = np.array([10.0, -1.0])

SETUPS = [
    ("raw", "无归一化", False, False),
    ("input", "输入标准化", True, False),
    ("bn", "BatchNorm", False, True),
]
COLORS = {"raw": "#be123c", "input": "#3b82f6", "bn": "#0f766e"}


def make_scaled_moons():
    X, y = generate_moons_dataset(n_per_class=200, noise=0.12, seed=0)
    X_raw = X * SCALE + SHIFT
    return X_raw, y


def standardize(X: np.ndarray):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return (X - mean) / std, mean, std


def run() -> dict:
    out = ensure_artifacts("07_normalization")
    X_raw, y = make_scaled_moons()

    print("=" * 90)
    print("归一化对比实验（月牙数据故意拉成极端尺度）")
    print("=" * 90)
    print(f"原始特征 mean={X_raw.mean(axis=0)}  std={X_raw.std(axis=0)}")
    print(f"结构 {LAYER_SIZES}，lr={LR}，迭代 {N_ITERS}，seed={SEED}")

    results = {}
    for key, label, use_input_std, use_bn in SETUPS:
        print(f"\n--- {key}: {label} ---")
        if use_input_std:
            X_train, mean, std = standardize(X_raw)
            transform = ("standardize", mean, std)
        else:
            X_train = X_raw
            transform = ("identity", None, None)

        model = MLP(
            LAYER_SIZES,
            learning_rate=LR,
            n_iters=N_ITERS,
            activation="relu",
            seed=SEED,
            init="he",
            batch_norm=use_bn,
        )
        model.fit(X_train, y, verbose=True, print_every=200)
        loss_curve = [float(v) if np.isfinite(v) else float("nan") for v in model.loss_]
        acc_curve = [float(v) if np.isfinite(v) else 0.0 for v in model.accuracy_]
        final_loss = loss_curve[-1]
        weights_ok = all(np.all(np.isfinite(w)) for w in model.W)
        if weights_ok and np.isfinite(final_loss):
            final_acc = float(np.mean(model.predict(X_train) == y) * 100)
        else:
            final_acc = 0.0
        diverged = (not weights_ok) or (not np.isfinite(final_loss)) or final_loss > 5.0
        print(f"  → 最终 loss={final_loss}  acc={final_acc:.2f}%  diverged={diverged}")
        results[key] = {
            "label": label,
            "loss": loss_curve,
            "accuracy": acc_curve,
            "final_loss": final_loss if np.isfinite(final_loss) else None,
            "final_acc": final_acc,
            "diverged": diverged,
            "model": model,
            "X_train": X_train,
            "transform": transform,
        }

    # ---- Figure 1: feature scatter before / after input std ----
    X_std, _, _ = standardize(X_raw)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, X_plot, title in [
        (axes[0], X_raw, "Raw features (extreme scale)"),
        (axes[1], X_std, "After input standardization"),
    ]:
        ax.scatter(X_plot[y == 0, 0], X_plot[y == 0, 1], s=12, alpha=0.7, c="#3b82f6", label="0")
        ax.scatter(X_plot[y == 1, 0], X_plot[y == 1, 1], s=12, alpha=0.7, c="#f59e0b", label="1")
        ax.set(title=title, xlabel="x0", ylabel="x1")
        ax.legend(); ax.grid(alpha=0.3)
        ax.set_aspect("auto")
    img_features = save_figure(fig, out / "norm_features.png")

    # ---- Figure 2: loss / accuracy ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5))
    for key, label, _, _ in SETUPS:
        r = results[key]
        axes2[0].plot(r["loss"], color=COLORS[key], linewidth=1.8, label=key)
        axes2[1].plot(r["accuracy"], color=COLORS[key], linewidth=1.8, label=key)
    axes2[0].set(xlabel="Epoch", ylabel="BCE Loss", title="Loss vs normalization")
    finite = [v for r in results.values() for v in r["loss"] if np.isfinite(v)]
    ymax = min(3.0, max(1.0, max(finite) * 1.1)) if finite else 3.0
    axes2[0].set_ylim(0, ymax)
    axes2[0].legend(); axes2[0].grid(alpha=0.3)
    axes2[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy vs normalization")
    axes2[1].set_ylim(-2, 102)
    axes2[1].legend(); axes2[1].grid(alpha=0.3)
    img_curves = save_figure(fig2, out / "norm_curves.png")

    # ---- Figure 3: decision boundaries in each model's training space ----
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label, _, _) in zip(axes3, SETUPS):
        r = results[key]
        if r["diverged"]:
            ax.text(0.5, 0.5, f"{label}\ndiverged", ha="center", va="center",
                    fontsize=14, color="#be123c", transform=ax.transAxes)
            ax.set_title(label)
            ax.axis("off")
        else:
            plot_decision_boundary(
                ax, r["model"], r["X_train"], y,
                f"{label}\nacc={r['final_acc']:.1f}%",
            )
    img_bounds = save_figure(fig3, out / "norm_boundaries.png")

    raw = results["raw"]
    inp = results["input"]
    bn = results["bn"]
    passed = (
        inp["final_acc"] >= 90
        and bn["final_acc"] >= 90
        and inp["final_acc"] > raw["final_acc"] + 10
        and bn["final_acc"] > raw["final_acc"] + 10
    )

    summary_rows = []
    for key, label, use_input_std, use_bn in SETUPS:
        r = results[key]
        summary_rows.append({
            "setup": key,
            "label": label,
            "input_std": use_input_std,
            "batch_norm": use_bn,
            "final_acc": r["final_acc"],
            "final_loss": r["final_loss"],
            "diverged": r["diverged"],
        })
        print(f"  {key:<6}  acc={r['final_acc']:6.2f}%  loss={r['final_loss']}  diverged={r['diverged']}")

    metrics = {
        "feature_raw_mean": X_raw.mean(axis=0).tolist(),
        "feature_raw_std": X_raw.std(axis=0).tolist(),
        "setups": summary_rows,
        "images": [
            str(img_features.relative_to(out.parent.parent)),
            str(img_curves.relative_to(out.parent.parent)),
            str(img_bounds.relative_to(out.parent.parent)),
        ],
        "passed": bool(passed),
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
