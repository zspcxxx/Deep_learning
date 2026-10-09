#!/usr/bin/env python3
"""Capstone: train one model that combines lessons 01–08 best practices."""
from __future__ import annotations

import numpy as np

from dllab.datasets import generate_moons_dataset
from dllab.io_utils import dump_metrics
from dllab.mlp import MLP
from dllab.paths import ensure_artifacts
from dllab.plotting import plot_decision_boundary, plt, save_figure

# Deeper net — needs good init / norm / reg to train stably.
LAYER_SIZES = [2, 32, 32, 32, 1]
N_ITERS = 1200
SEED = 42
N_TRAIN, N_VAL = 80, 80
NOISE = 0.22
N_FLIP = 6
# Deliberately awkward feature scale (lesson 07).
SCALE = np.array([12.0, 0.12])
SHIFT = np.array([8.0, -1.5])

# Checklist tying each knob back to earlier lessons.
RECIPE = [
    ("02/04", "deep MLP + ReLU", "non-linear boundary"),
    ("03", "backprop + full-batch SGD", "hand-written grads"),
    ("05", "sensible learning rate", "naive lr=6.0, tuned lr=0.4"),
    ("06", "He init", "naive uses large Gaussian"),
    ("07", "input std + BatchNorm", "naive uses raw scales"),
    ("08", "L2 + Dropout + early stop", "naive: no reg, full epochs"),
]

PIPELINES = {
    "naive": {
        "label": "naive (broken knobs)",
        "color": "#be123c",
        "standardize": False,
        "kwargs": dict(
            learning_rate=6.0,
            n_iters=N_ITERS,
            activation="relu",
            seed=SEED,
            init="large",
            batch_norm=False,
            l2=0.0,
            dropout=0.0,
        ),
        "early_stop": None,
    },
    "tuned": {
        "label": "tuned (lessons 01-08)",
        "color": "#0f766e",
        "standardize": True,
        "kwargs": dict(
            learning_rate=0.4,
            n_iters=N_ITERS,
            activation="relu",
            seed=SEED,
            init="he",
            batch_norm=True,
            l2=0.06,
            dropout=0.25,
        ),
        "early_stop": 100,
    },
}


def make_data(seed: int = 0):
    X, y = generate_moons_dataset(n_per_class=220, noise=NOISE, seed=seed)
    X = X * SCALE + SHIFT
    rng = np.random.RandomState(seed + 3)
    idx = rng.permutation(len(y))
    X, y = X[idx], y[idx]
    X_train, y_train = X[:N_TRAIN], y[:N_TRAIN].copy()
    X_val, y_val = X[N_TRAIN : N_TRAIN + N_VAL], y[N_TRAIN : N_TRAIN + N_VAL]
    X_test, y_test = X[N_TRAIN + N_VAL :], y[N_TRAIN + N_VAL :]
    flip = rng.choice(N_TRAIN, size=N_FLIP, replace=False)
    y_train[flip] = 1 - y_train[flip]
    return X_train, y_train, X_val, y_val, X_test, y_test, flip


def standardize_fit(X):
    mean = X.mean(axis=0)
    std = X.std(axis=0)
    std = np.where(std < 1e-8, 1.0, std)
    return mean, std


def apply_standardize(X, mean, std):
    return (X - mean) / std


def run() -> dict:
    out = ensure_artifacts("09_full_pipeline")
    X_train, y_train, X_val, y_val, X_test, y_test, flip = make_data()

    print("=" * 90)
    print("综合实战：把 01–08 串成一条训练流水线")
    print("=" * 90)
    print(f"结构 {LAYER_SIZES} | train={len(y_train)} (flip {len(flip)}) val={len(y_val)} test={len(y_test)}")
    print(f"特征尺度 std={X_train.std(axis=0)}")
    print("知识清单：")
    for lesson, knob, note in RECIPE:
        print(f"  [{lesson}] {knob} — {note}")

    results = {}
    for key, cfg in PIPELINES.items():
        print(f"\n--- {key}: {cfg['label']} ---")
        if cfg["standardize"]:
            mean, std = standardize_fit(X_train)
            Xt = apply_standardize(X_train, mean, std)
            Xv = apply_standardize(X_val, mean, std)
            Xe = apply_standardize(X_test, mean, std)
            transform = {"mean": mean.tolist(), "std": std.tolist()}
        else:
            Xt, Xv, Xe = X_train, X_val, X_test
            transform = None

        model = MLP(LAYER_SIZES, **cfg["kwargs"])
        fit_kw = {"verbose": True, "print_every": 300}
        if cfg["early_stop"] is not None:
            fit_kw.update(X_val=Xv, y_val=y_val, early_stop_patience=cfg["early_stop"])
        model.fit(Xt, y_train, **fit_kw)

        train_acc = float(np.mean(model.predict(Xt) == y_train) * 100)
        val_acc = float(np.mean(model.predict(Xv) == y_val) * 100)
        test_acc = float(np.mean(model.predict(Xe) == y_test) * 100)
        epochs = (model.best_epoch_ + 1) if model.best_epoch_ is not None else len(model.loss_)
        finite = all(np.all(np.isfinite(w)) for w in model.W) and np.isfinite(model.loss_[-1])
        print(f"  → train={train_acc:.2f}% val={val_acc:.2f}% test={test_acc:.2f}% epochs={epochs} ok={finite}")
        results[key] = {
            "label": cfg["label"],
            "loss": [float(v) if np.isfinite(v) else float("nan") for v in model.loss_],
            "train_curve": [float(v) for v in model.accuracy_],
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "epochs": int(epochs),
            "finite": bool(finite),
            "transform": transform,
            "model": model,
            "X_plot": Xt,
        }

    # ---- Figure 1: recipe card as text table image-ish via bars ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for key, cfg in PIPELINES.items():
        r = results[key]
        axes[0].plot(r["loss"], color=cfg["color"], linewidth=1.8, label=key)
        axes[1].plot(r["train_curve"], color=cfg["color"], linewidth=1.8, label=key)
    axes[0].set(xlabel="Epoch", ylabel="Loss", title="Training loss: naive vs tuned")
    finite_losses = [v for r in results.values() for v in r["loss"] if np.isfinite(v)]
    ymax = min(4.0, max(1.0, max(finite_losses) * 1.1)) if finite_losses else 4.0
    axes[0].set_ylim(0, ymax)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Train acc (%)", title="Training accuracy")
    axes[1].set_ylim(-2, 102)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    img_curves = save_figure(fig, out / "pipeline_curves.png")

    # ---- Figure 2: metric comparison ----
    fig2, ax2 = plt.subplots(figsize=(8, 5))
    keys = list(PIPELINES)
    xs = np.arange(3)
    width = 0.35
    for i, key in enumerate(keys):
        r = results[key]
        vals = [r["train_acc"], r["val_acc"], r["test_acc"]]
        ax2.bar(xs + (i - 0.5) * width, vals, width, label=key, color=PIPELINES[key]["color"])
    ax2.set_xticks(xs)
    ax2.set_xticklabels(["train", "val", "test"])
    ax2.set(ylabel="Accuracy (%)", title="Hold-out comparison", ylim=(0, 105))
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    img_bars = save_figure(fig2, out / "pipeline_metrics.png")

    # ---- Figure 3: boundaries in each model's input space ----
    fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
    for ax, key in zip(axes3, keys):
        r = results[key]
        cfg = PIPELINES[key]
        if not r["finite"] or r["test_acc"] < 55:
            ax.text(0.5, 0.5, f"{key}\nfailed / weak", ha="center", va="center",
                    fontsize=14, color="#be123c", transform=ax.transAxes)
            ax.set_title(cfg["label"])
            ax.axis("off")
        else:
            y_for_plot = y_train
            plot_decision_boundary(
                ax, r["model"], r["X_plot"], y_for_plot,
                f"{key}  test={r['test_acc']:.1f}%",
            )
    img_bounds = save_figure(fig3, out / "pipeline_boundaries.png")

    # ---- Figure 4: lesson checklist (static annotation plot) ----
    fig4, ax4 = plt.subplots(figsize=(10, 4.5))
    ax4.axis("off")
    lines = ["Knowledge checklist used by tuned pipeline:\n"]
    for lesson, knob, note in RECIPE:
        lines.append(f"  [{lesson}]  {knob}  →  {note}")
    lines.append("\nnaive deliberately breaks 05/06/07/08 to show the gap.")
    ax4.text(0.02, 0.95, "\n".join(lines), va="top", ha="left", family="monospace", fontsize=11,
             transform=ax4.transAxes)
    ax4.set_title("How earlier lessons plug into this training run")
    img_recipe = save_figure(fig4, out / "pipeline_recipe.png")

    naive, tuned = results["naive"], results["tuned"]
    passed = (
        tuned["finite"]
        and tuned["test_acc"] >= 85
        and tuned["test_acc"] > naive["test_acc"] + 15
        and (not naive["finite"] or naive["test_acc"] < 75)
    )

    summary = []
    for key, cfg in PIPELINES.items():
        r = results[key]
        summary.append({
            "pipeline": key,
            "label": cfg["label"],
            "standardize": cfg["standardize"],
            "init": cfg["kwargs"]["init"],
            "lr": cfg["kwargs"]["learning_rate"],
            "batch_norm": cfg["kwargs"]["batch_norm"],
            "l2": cfg["kwargs"]["l2"],
            "dropout": cfg["kwargs"]["dropout"],
            "early_stop": cfg["early_stop"],
            "train_acc": r["train_acc"],
            "val_acc": r["val_acc"],
            "test_acc": r["test_acc"],
            "epochs": r["epochs"],
            "finite": r["finite"],
        })
        print(f"  {key:<6} test={r['test_acc']:6.2f}% train={r['train_acc']:6.2f}% "
              f"epochs={r['epochs']} finite={r['finite']}")

    metrics = {
        "recipe": [{"lesson": a, "knob": b, "note": c} for a, b, c in RECIPE],
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "n_flip": int(len(flip)),
        "pipelines": summary,
        "images": [
            str(img_curves.relative_to(out.parent.parent)),
            str(img_bars.relative_to(out.parent.parent)),
            str(img_bounds.relative_to(out.parent.parent)),
            str(img_recipe.relative_to(out.parent.parent)),
        ],
        "passed": bool(passed),
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
