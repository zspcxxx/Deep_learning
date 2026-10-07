#!/usr/bin/env python3
"""Compare how different weight-initialization schemes affect training."""
from __future__ import annotations

import numpy as np

from dllab.datasets import generate_xor_dataset
from dllab.io_utils import dump_metrics
from dllab.mlp import MLP
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure

# Deeper net so bad init shows vanishing / exploding more clearly.
LAYER_SIZES = [2, 32, 32, 32, 1]
N_ITERS = 800
LR = 0.5
SEED = 42
SCHEMES = [
    ("zeros", "全零（对称崩溃）"),
    ("small", "过小高斯 σ=0.01"),
    ("large", "过大高斯 σ=2.0"),
    ("he", "He / Kaiming"),
]
COLORS = {
    "zeros": "#64748b",
    "small": "#3b82f6",
    "large": "#be123c",
    "he": "#0f766e",
}


def activation_stats(model: MLP, X: np.ndarray) -> list[dict]:
    """Mean / std of each hidden activation right after init (or current weights)."""
    acts, _ = model._forward(X)
    rows = []
    # acts[0]=input, acts[1..-2]=hidden, acts[-1]=sigmoid out
    for i, a in enumerate(acts[1:-1], start=1):
        rows.append({
            "layer": i,
            "mean": float(np.mean(a)),
            "std": float(np.std(a)),
            "frac_zero": float(np.mean(a == 0)),
        })
    return rows


def weight_stds(model: MLP) -> list[float]:
    return [float(np.std(w)) for w in model.W]


def run() -> dict:
    out = ensure_artifacts("06_weight_init")
    X, y = generate_xor_dataset(n_per_quadrant=100, noise=0.15, seed=0)

    print("=" * 90)
    print("参数初始化对比实验（同一 XOR、同一结构 / lr / seed，只改 init）")
    print("=" * 90)
    print(f"结构 {LAYER_SIZES}，lr={LR}，迭代 {N_ITERS}，seed={SEED}")

    results = {}
    for scheme, label in SCHEMES:
        print(f"\n--- init = {scheme} ({label}) ---")
        model = MLP(
            LAYER_SIZES,
            learning_rate=LR,
            n_iters=N_ITERS,
            activation="relu",
            seed=SEED,
            init=scheme,
        )
        init_acts = activation_stats(model, X)
        init_wstd = weight_stds(model)
        wstd_s = [f"{s:.4f}" for s in init_wstd]
        act_s = [f"{r['std']:.4f}" for r in init_acts]
        print(f"  初始权重 std: {wstd_s}")
        print(f"  初始隐藏层激活 std: {act_s}")

        model.fit(X, y, verbose=True, print_every=200)
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
        results[scheme] = {
            "label": label,
            "loss": loss_curve,
            "accuracy": acc_curve,
            "final_loss": final_loss if np.isfinite(final_loss) else None,
            "final_acc": final_acc,
            "diverged": diverged,
            "init_acts": init_acts,
            "init_wstd": init_wstd,
            "model": model,
        }

    # ---- Figure 1: loss + accuracy ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    for scheme, label in SCHEMES:
        r = results[scheme]
        axes[0].plot(r["loss"], color=COLORS[scheme], linewidth=1.8, label=f"{scheme}")
        axes[1].plot(r["accuracy"], color=COLORS[scheme], linewidth=1.8, label=f"{scheme}")
    axes[0].set(xlabel="Epoch", ylabel="BCE Loss", title="Loss vs weight init")
    finite_losses = [v for r in results.values() for v in r["loss"] if np.isfinite(v)]
    ymax = min(3.0, max(1.0, max(finite_losses) * 1.1)) if finite_losses else 3.0
    axes[0].set_ylim(0, ymax)
    axes[0].legend(); axes[0].grid(alpha=0.3)
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Accuracy vs weight init")
    axes[1].set_ylim(-2, 102)
    axes[1].legend(); axes[1].grid(alpha=0.3)
    img_curves = save_figure(fig, out / "init_curves.png")

    # ---- Figure 2: initial activation std by depth ----
    fig2, ax2 = plt.subplots(figsize=(9, 5))
    n_hidden = len(LAYER_SIZES) - 2
    x = np.arange(1, n_hidden + 1)
    width = 0.18
    for i, (scheme, _) in enumerate(SCHEMES):
        stds = [row["std"] for row in results[scheme]["init_acts"]]
        ax2.bar(x + (i - 1.5) * width, stds, width=width, color=COLORS[scheme], label=scheme)
    ax2.set(xlabel="Hidden layer index", ylabel="Activation std (at init)",
            title="Forward signal scale right after initialization")
    ax2.set_xticks(x)
    ax2.legend(); ax2.grid(axis="y", alpha=0.3)
    img_acts = save_figure(fig2, out / "init_act_std.png")

    # ---- Figure 3: activation histograms at init (layer 2) ----
    fig3, axes3 = plt.subplots(1, 4, figsize=(16, 3.8), sharey=True)
    for ax, (scheme, label) in zip(axes3, SCHEMES):
        model = results[scheme]["model"]
        # Re-init a fresh model to histogram true initial activations
        fresh = MLP(LAYER_SIZES, learning_rate=LR, n_iters=1, activation="relu", seed=SEED, init=scheme)
        acts, _ = fresh._forward(X)
        # middle hidden layer
        mid = acts[2].ravel()
        ax.hist(mid, bins=40, color=COLORS[scheme], alpha=0.85, density=True)
        ax.set_title(f"{scheme}\nstd={mid.std():.3f}")
        ax.set_xlabel("a (layer 2)")
        ax.grid(alpha=0.3)
    axes3[0].set_ylabel("density")
    fig3.suptitle("Hidden-layer activation histogram at initialization", y=1.02)
    img_hist = save_figure(fig3, out / "init_act_hist.png")

    zeros = results["zeros"]
    small = results["small"]
    large = results["large"]
    he = results["he"]
    # He should dominate; zeros stuck; large clearly worse or diverged.
    passed = (
        he["final_acc"] >= 90
        and he["final_acc"] > zeros["final_acc"] + 20
        and he["final_acc"] > small["final_acc"] + 5
        and (large["diverged"] or large["final_acc"] < he["final_acc"] - 10)
    )

    summary_rows = []
    for scheme, label in SCHEMES:
        r = results[scheme]
        summary_rows.append({
            "scheme": scheme,
            "label": label,
            "final_acc": r["final_acc"],
            "final_loss": r["final_loss"],
            "diverged": r["diverged"],
            "init_wstd": r["init_wstd"],
            "init_act_std": [row["std"] for row in r["init_acts"]],
        })
        print(f"  {scheme:<6}  acc={r['final_acc']:6.2f}%  loss={r['final_loss']}  diverged={r['diverged']}")

    metrics = {
        "schemes": summary_rows,
        "images": [
            str(img_curves.relative_to(out.parent.parent)),
            str(img_acts.relative_to(out.parent.parent)),
            str(img_hist.relative_to(out.parent.parent)),
        ],
        "passed": bool(passed),
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
