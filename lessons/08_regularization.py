#!/usr/bin/env python3
"""Compare no-reg / L2 / too-strong L2 (overfit → regularize → underfit)."""
from __future__ import annotations

import numpy as np

from dllab.datasets import generate_moons_dataset
from dllab.io_utils import dump_metrics
from dllab.mlp import MLP
from dllab.paths import ensure_artifacts
from dllab.plotting import plot_decision_boundary, plt, save_figure

LAYER_SIZES = [2, 64, 64, 1]
N_ITERS = 1600
LR = 0.5
SEED = 42
N_TRAIN = 28
NOISE = 0.32
N_FLIP = 8  # label noise on train only — encourages memorization

SETUPS = [
    ("none", "无正则化", 0.0),
    ("l2", "适量 L2", 0.15),
    ("strong", "过强 L2", 2.0),
]
COLORS = {"none": "#be123c", "l2": "#0f766e", "strong": "#64748b"}


def split_moons(n_train: int = N_TRAIN, n_flip: int = N_FLIP, seed: int = 0):
    X, y = generate_moons_dataset(n_per_class=250, noise=NOISE, seed=seed)
    rng = np.random.RandomState(seed + 1)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    X_train, y_train = X[:n_train], y[:n_train].copy()
    X_test, y_test = X[n_train:], y[n_train:]
    flip_idx = rng.choice(n_train, size=n_flip, replace=False)
    y_train[flip_idx] = 1 - y_train[flip_idx]
    return X_train, y_train, X_test, y_test, flip_idx


def weight_l2_norm(model: MLP) -> float:
    return float(np.sqrt(sum(np.sum(w * w) for w in model.W)))


def run() -> dict:
    out = ensure_artifacts("08_regularization")
    X_train, y_train, X_test, y_test, flip_idx = split_moons()

    print("=" * 90)
    print("正则化对比实验（小训练集 + 标签噪声 + 宽网络）")
    print("=" * 90)
    print(f"训练 {len(y_train)}（其中翻转 {len(flip_idx)} 个标签）/ 测试 {len(y_test)}")
    print(f"噪声={NOISE}，结构 {LAYER_SIZES}，lr={LR}，迭代 {N_ITERS}")

    results = {}
    for key, label, l2 in SETUPS:
        print(f"\n--- {key}: {label} (l2={l2}) ---")
        model = MLP(
            LAYER_SIZES,
            learning_rate=LR,
            n_iters=N_ITERS,
            activation="relu",
            seed=SEED,
            init="he",
            l2=l2,
        )
        model.fit(X_train, y_train, verbose=True, print_every=400)
        train_acc = float(np.mean(model.predict(X_train) == y_train) * 100)
        test_acc = float(np.mean(model.predict(X_test) == y_test) * 100)
        gap = train_acc - test_acc
        wnorm = weight_l2_norm(model)
        print(f"  → train={train_acc:.2f}%  test={test_acc:.2f}%  gap={gap:.2f}  ||W||={wnorm:.3f}")
        results[key] = {
            "label": label,
            "loss": [float(v) for v in model.loss_],
            "train_curve": [float(v) for v in model.accuracy_],
            "train_acc": train_acc,
            "test_acc": test_acc,
            "gap": gap,
            "weight_norm": wnorm,
            "model": model,
        }

    # ---- Figure 1: training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, _, _ in SETUPS:
        r = results[key]
        axes[0].plot(r["loss"], color=COLORS[key], linewidth=1.6, label=key)
        axes[1].plot(r["train_curve"], color=COLORS[key], linewidth=1.6, label=key)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss (BCE + L2)")
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Train acc (%)", title="Training accuracy")
    axes[1].set_ylim(-2, 102)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    img_curves = save_figure(fig, out / "reg_curves.png")

    # ---- Figure 2: train/test bars + weight norms ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(12, 5))
    xs = np.arange(len(SETUPS))
    width = 0.35
    train_vals = [results[k]["train_acc"] for k, *_ in SETUPS]
    test_vals = [results[k]["test_acc"] for k, *_ in SETUPS]
    axes2[0].bar(xs - width / 2, train_vals, width, label="train", color="#94a3b8")
    axes2[0].bar(xs + width / 2, test_vals, width, label="test", color="#0f766e")
    axes2[0].set_xticks(xs)
    axes2[0].set_xticklabels([k for k, *_ in SETUPS])
    axes2[0].set(ylabel="Accuracy (%)", title="Train vs test", ylim=(0, 105))
    axes2[0].legend(); axes2[0].grid(axis="y", alpha=0.3)

    wnorms = [results[k]["weight_norm"] for k, *_ in SETUPS]
    axes2[1].bar(xs, wnorms, color=[COLORS[k] for k, *_ in SETUPS])
    axes2[1].set_xticks(xs)
    axes2[1].set_xticklabels([k for k, *_ in SETUPS])
    axes2[1].set(ylabel="||W||_2", title="Weight magnitude")
    axes2[1].grid(axis="y", alpha=0.3)
    img_bars = save_figure(fig2, out / "reg_train_test.png")

    # ---- Figure 3: decision boundaries ----
    fig3, axes3 = plt.subplots(1, 3, figsize=(15, 4.5))
    for ax, (key, label, _) in zip(axes3, SETUPS):
        r = results[key]
        plot_decision_boundary(
            ax, r["model"], X_train, y_train,
            f"{key}  test={r['test_acc']:.1f}%\n||W||={r['weight_norm']:.1f}",
        )
        ax.scatter(X_test[::4, 0], X_test[::4, 1], s=8, c="#64748b", alpha=0.2, zorder=0)
    img_bounds = save_figure(fig3, out / "reg_boundaries.png")

    none, mild, strong = results["none"], results["l2"], results["strong"]
    # Spectrum: memorize → shrink weights → underfit
    passed = (
        none["train_acc"] >= 90
        and none["gap"] >= 15
        and mild["weight_norm"] < none["weight_norm"] * 0.6
        and strong["train_acc"] < 70
        and strong["weight_norm"] < mild["weight_norm"] * 0.5
    )

    summary_rows = []
    for key, label, l2 in SETUPS:
        r = results[key]
        summary_rows.append({
            "setup": key,
            "label": label,
            "l2": l2,
            "train_acc": r["train_acc"],
            "test_acc": r["test_acc"],
            "gap": r["gap"],
            "weight_norm": r["weight_norm"],
        })
        print(f"  {key:<8} train={r['train_acc']:6.2f}% test={r['test_acc']:6.2f}% "
              f"gap={r['gap']:5.2f} ||W||={r['weight_norm']:.3f}")

    metrics = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "n_flip": int(len(flip_idx)),
        "noise": NOISE,
        "setups": summary_rows,
        "images": [
            str(img_curves.relative_to(out.parent.parent)),
            str(img_bars.relative_to(out.parent.parent)),
            str(img_bounds.relative_to(out.parent.parent)),
        ],
        "passed": bool(passed),
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
