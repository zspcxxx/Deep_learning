#!/usr/bin/env python3
"""Compare multiple regularization methods on a small noisy train set."""
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
N_VAL = 60
NOISE = 0.32
N_FLIP = 8
SPARSE_EPS = 1e-3

# key, label, l1, l2, dropout, early_stop_patience (None = off)
SETUPS = [
    ("none", "无正则", 0.0, 0.0, 0.0, None),
    ("l2", "L2", 0.0, 0.15, 0.0, None),
    ("l1", "L1", 0.04, 0.0, 0.0, None),
    ("dropout", "Dropout", 0.0, 0.0, 0.4, None),
    ("elastic", "L1+L2", 0.02, 0.08, 0.0, None),
    ("l2_drop", "L2+Dropout", 0.0, 0.08, 0.35, None),
    ("early", "早停", 0.0, 0.0, 0.0, 80),
    ("strong", "过强 L2", 0.0, 2.0, 0.0, None),
]
COLORS = {
    "none": "#be123c",
    "l2": "#0f766e",
    "l1": "#7c3aed",
    "dropout": "#2563eb",
    "elastic": "#0891b2",
    "l2_drop": "#ca8a04",
    "early": "#ea580c",
    "strong": "#64748b",
}


def split_moons(n_train: int = N_TRAIN, n_val: int = N_VAL, n_flip: int = N_FLIP, seed: int = 0):
    X, y = generate_moons_dataset(n_per_class=250, noise=NOISE, seed=seed)
    rng = np.random.RandomState(seed + 1)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    X_train, y_train = X[:n_train], y[:n_train].copy()
    X_val, y_val = X[n_train : n_train + n_val], y[n_train : n_train + n_val]
    X_test, y_test = X[n_train + n_val :], y[n_train + n_val :]
    flip_idx = rng.choice(n_train, size=n_flip, replace=False)
    y_train[flip_idx] = 1 - y_train[flip_idx]
    return X_train, y_train, X_val, y_val, X_test, y_test, flip_idx


def weight_l2_norm(model: MLP) -> float:
    return float(np.sqrt(sum(np.sum(w * w) for w in model.W)))


def weight_sparsity(model: MLP, eps: float = SPARSE_EPS) -> float:
    total = sum(w.size for w in model.W)
    near_zero = sum(int(np.sum(np.abs(w) < eps)) for w in model.W)
    return float(near_zero / total)


def run() -> dict:
    out = ensure_artifacts("08_regularization")
    X_train, y_train, X_val, y_val, X_test, y_test, flip_idx = split_moons()

    print("=" * 90)
    print("多种正则化对比（小训练集 + 标签噪声 + 宽网络）")
    print("=" * 90)
    print(f"训练 {len(y_train)}（翻转 {len(flip_idx)}）/ 验证 {len(y_val)} / 测试 {len(y_test)}")
    print(f"噪声={NOISE}，结构 {LAYER_SIZES}，lr={LR}，迭代 {N_ITERS}")
    print(f"方法: {[k for k, *_ in SETUPS]}")

    results = {}
    for key, label, l1, l2, dropout, patience in SETUPS:
        print(f"\n--- {key}: {label} (l1={l1}, l2={l2}, dropout={dropout}, early={patience}) ---")
        model = MLP(
            LAYER_SIZES,
            learning_rate=LR,
            n_iters=N_ITERS,
            activation="relu",
            seed=SEED,
            init="he",
            l1=l1,
            l2=l2,
            dropout=dropout,
        )
        fit_kw = {"verbose": True, "print_every": 400}
        if patience is not None:
            fit_kw.update(X_val=X_val, y_val=y_val, early_stop_patience=patience)
        model.fit(X_train, y_train, **fit_kw)
        train_acc = float(np.mean(model.predict(X_train) == y_train) * 100)
        val_acc = float(np.mean(model.predict(X_val) == y_val) * 100)
        test_acc = float(np.mean(model.predict(X_test) == y_test) * 100)
        gap = train_acc - test_acc
        wnorm = weight_l2_norm(model)
        sparse = weight_sparsity(model)
        stopped = model.best_epoch_ + 1 if model.best_epoch_ is not None else N_ITERS
        print(
            f"  → train={train_acc:.2f}% val={val_acc:.2f}% test={test_acc:.2f}% "
            f"gap={gap:.2f} ||W||={wnorm:.3f} sparse={sparse:.3f} epochs={stopped}"
        )
        results[key] = {
            "label": label,
            "l1": l1,
            "l2": l2,
            "dropout": dropout,
            "early_stop": patience,
            "loss": [float(v) for v in model.loss_],
            "train_curve": [float(v) for v in model.accuracy_],
            "val_loss": [float(v) for v in model.val_loss_],
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "gap": gap,
            "weight_norm": wnorm,
            "sparsity": sparse,
            "epochs_used": int(stopped),
            "model": model,
        }

    keys = [k for k, *_ in SETUPS]

    # ---- Figure 1: training curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key in keys:
        r = results[key]
        axes[0].plot(r["loss"], color=COLORS[key], linewidth=1.4, label=key)
        axes[1].plot(r["train_curve"], color=COLORS[key], linewidth=1.4, label=key)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss")
    axes[0].legend(fontsize=8, ncol=2); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Train acc (%)", title="Training accuracy")
    axes[1].set_ylim(-2, 102)
    axes[1].legend(fontsize=8, ncol=2); axes[1].grid(alpha=0.3)
    img_curves = save_figure(fig, out / "reg_curves.png")

    # ---- Figure 2: train/test + weight / sparsity ----
    fig2, axes2 = plt.subplots(1, 3, figsize=(16, 4.8))
    xs = np.arange(len(keys))
    width = 0.35
    train_vals = [results[k]["train_acc"] for k in keys]
    test_vals = [results[k]["test_acc"] for k in keys]
    axes2[0].bar(xs - width / 2, train_vals, width, label="train", color="#94a3b8")
    axes2[0].bar(xs + width / 2, test_vals, width, label="test", color="#0f766e")
    axes2[0].set_xticks(xs)
    axes2[0].set_xticklabels(keys, rotation=30, ha="right")
    axes2[0].set(ylabel="Accuracy (%)", title="Train vs test", ylim=(0, 105))
    axes2[0].legend(); axes2[0].grid(axis="y", alpha=0.3)

    wnorms = [results[k]["weight_norm"] for k in keys]
    axes2[1].bar(xs, wnorms, color=[COLORS[k] for k in keys])
    axes2[1].set_xticks(xs)
    axes2[1].set_xticklabels(keys, rotation=30, ha="right")
    axes2[1].set(ylabel="||W||_2", title="Weight magnitude")
    axes2[1].grid(axis="y", alpha=0.3)

    spars = [results[k]["sparsity"] for k in keys]
    axes2[2].bar(xs, spars, color=[COLORS[k] for k in keys])
    axes2[2].set_xticks(xs)
    axes2[2].set_xticklabels(keys, rotation=30, ha="right")
    axes2[2].set(ylabel=f"frac |w|<{SPARSE_EPS}", title="Sparsity", ylim=(0, 1))
    axes2[2].grid(axis="y", alpha=0.3)
    img_bars = save_figure(fig2, out / "reg_train_test.png")

    # ---- Figure 3: decision boundaries (selected methods) ----
    show = ["none", "l2", "l1", "dropout", "elastic", "early", "l2_drop", "strong"]
    fig3, axes3 = plt.subplots(2, 4, figsize=(16, 8))
    for ax, key in zip(axes3.ravel(), show):
        r = results[key]
        plot_decision_boundary(
            ax, r["model"], X_train, y_train,
            f"{key}  test={r['test_acc']:.1f}%\n||W||={r['weight_norm']:.1f}",
        )
        ax.scatter(X_test[::5, 0], X_test[::5, 1], s=6, c="#64748b", alpha=0.2, zorder=0)
    img_bounds = save_figure(fig3, out / "reg_boundaries.png")

    # ---- Figure 4: early-stop val curve vs none ----
    fig4, ax4 = plt.subplots(figsize=(9, 4.5))
    if results["early"]["val_loss"]:
        ax4.plot(results["early"]["val_loss"], color=COLORS["early"], linewidth=1.8, label="early val loss")
        if results["early"]["epochs_used"]:
            ax4.axvline(results["early"]["epochs_used"] - 1, color=COLORS["early"], linestyle="--", alpha=0.7,
                        label=f"stop @ {results['early']['epochs_used']}")
    # also plot none val by evaluating... none wasn't tracked; skip
    ax4.set(xlabel="Epoch", ylabel="Val BCE", title="Early stopping on validation loss")
    ax4.legend(); ax4.grid(alpha=0.3)
    img_early = save_figure(fig4, out / "reg_early_stop.png")

    none, l2r, l1r, dropr, elastic, l2d, early, strong = (
        results["none"], results["l2"], results["l1"], results["dropout"],
        results["elastic"], results["l2_drop"], results["early"], results["strong"],
    )
    passed = (
        none["train_acc"] >= 90
        and none["gap"] >= 12
        and l2r["weight_norm"] < none["weight_norm"] * 0.65
        and l1r["sparsity"] > none["sparsity"] + 0.05
        and strong["train_acc"] < 70
        and early["epochs_used"] < N_ITERS
        and dropr["train_acc"] >= 70  # dropout still learns something
        and elastic["weight_norm"] < none["weight_norm"]
        and l2d["weight_norm"] < none["weight_norm"]
    )

    summary_rows = []
    for key, label, l1, l2, dropout, patience in SETUPS:
        r = results[key]
        summary_rows.append({
            "setup": key,
            "label": label,
            "l1": l1,
            "l2": l2,
            "dropout": dropout,
            "early_stop_patience": patience,
            "train_acc": r["train_acc"],
            "val_acc": r["val_acc"],
            "test_acc": r["test_acc"],
            "gap": r["gap"],
            "weight_norm": r["weight_norm"],
            "sparsity": r["sparsity"],
            "epochs_used": r["epochs_used"],
        })
        print(
            f"  {key:<8} train={r['train_acc']:6.2f}% test={r['test_acc']:6.2f}% "
            f"gap={r['gap']:5.2f} ||W||={r['weight_norm']:7.3f} sparse={r['sparsity']:.3f} "
            f"epochs={r['epochs_used']}"
        )

    metrics = {
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_flip": int(len(flip_idx)),
        "noise": NOISE,
        "methods": [k for k, *_ in SETUPS],
        "setups": summary_rows,
        "images": [
            str(img_curves.relative_to(out.parent.parent)),
            str(img_bars.relative_to(out.parent.parent)),
            str(img_bounds.relative_to(out.parent.parent)),
            str(img_early.relative_to(out.parent.parent)),
        ],
        "passed": bool(passed),
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
