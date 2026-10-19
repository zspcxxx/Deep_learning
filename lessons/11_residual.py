#!/usr/bin/env python3
"""Lesson 11: residual connections — deep plain MLPs degrade, h + F(h) keeps them trainable."""
from __future__ import annotations

import time

import numpy as np

from dllab.data_store import cached_dataset
from dllab.datasets import generate_spiral_dataset
from dllab.io_utils import dump_metrics
from dllab.nn import gradient_check
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure
from dllab.resnet import block_input_grads, block_outputs, build_deep_mlp, n_weight_layers

LESSON = "11_residual"
SEED = 42
SPIRAL = dict(n_per_class=300, n_classes=3, turns=2.5, noise=0.15)
WIDTH = 32
BLOCKS = [2, 4, 8, 16, 24]
FIT = dict(n_epochs=100, batch_size=64, lr=2e-3, min_lr=1e-4, warmup_epochs=2, weight_decay=0.0,
           early_stop_patience=None, optimizer="adamw", seed=0)
VARIANTS = {
    "plain": {"label": "plain", "residual": False, "branch_init": "he", "color": "#be123c"},
    "residual": {"label": "residual (zero-init branch)", "residual": True, "branch_init": "zeros", "color": "#0f766e"},
    "residual_he": {"label": "residual (He-init branch)", "residual": True, "branch_init": "he", "color": "#ca8a04"},
}


def make_model(key: str, n_blocks: int):
    v = VARIANTS[key]
    return build_deep_mlp(n_blocks, v["residual"], width=WIDTH, n_classes=SPIRAL["n_classes"],
                          branch_init=v["branch_init"], seed=SEED)


def plot_boundary(ax, model, X, y, title):
    xx, yy = np.meshgrid(np.linspace(-1.15, 1.15, 160), np.linspace(-1.15, 1.15, 160))
    pred = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    ax.contourf(xx, yy, pred, levels=[-0.5, 0.5, 1.5, 2.5], colors=["#fecaca", "#bbf7d0", "#bfdbfe"], alpha=0.8)
    ax.scatter(X[:, 0], X[:, 1], c=np.array(["#dc2626", "#16a34a", "#2563eb"])[y], s=6)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([])
    ax.set_yticks([])


def run() -> dict:
    out = ensure_artifacts(LESSON)
    print("=" * 94)
    print("残差连接（11）：网络越深越难训练？给每个块加一条 h + F(h) 的捷径")
    print("=" * 94)

    X_tr, y_tr = cached_dataset(LESSON, "spiral_train", generate_spiral_dataset, **SPIRAL, seed=0)
    X_te, y_te = cached_dataset(LESSON, "spiral_test", generate_spiral_dataset, **SPIRAL, seed=1)
    print(f"数据: data/{LESSON}/spiral_*.npz | {SPIRAL['n_classes']} 条螺旋臂，每条绕 {SPIRAL['turns']} 圈 | "
          f"train={len(y_tr)} test={len(y_te)}")
    print(f"网络: 宽度 {WIDTH}，每个块 = ReLU → Dense → ReLU → Dense；深度 = 1 + 2×块数 + 1 个 Dense 层")

    # ---- 1. gradient check of the residual block's backward pass ----
    ck_model = build_deep_mlp(2, True, width=6, n_classes=3, branch_init="he", seed=1)
    ck = gradient_check(ck_model, X_tr[:16], y_tr[:16], n_checks=10)
    grad_ok = all(err < 1e-5 for err in ck.values())
    print(f"\n梯度检查（2 个残差块，16 个样本）：")
    for key, err in ck.items():
        print(f"  {key:<16s} max relative error = {err:.3e}   {'✓ 通过' if err < 1e-5 else '✗ 未通过'}")

    # ---- 2. signal propagation at initialization (deepest nets) ----
    deep = BLOCKS[-1]
    print(f"\n信号传播（初始化后、训练前，{n_weight_layers(deep)} 层网络）：")
    signal = {}
    for key in VARIANTS:
        model = make_model(key, deep)
        fwd = [float(h.std()) for h in block_outputs(model, X_tr)]
        bwd = block_input_grads(model, X_tr, y_tr)
        signal[key] = {"forward_std": fwd, "grad_norm": bwd}
        print(f"  [{key}] 前向 h 的标准差: 块0 {fwd[0]:.3g} → 块{deep} {fwd[-1]:.3g} | "
              f"反向 ‖dL/dh‖: 块{deep} {bwd[-1]:.3g} → 块1 {bwd[0]:.3g}")

    # ---- 3. depth sweep ----
    results = {key: [] for key in VARIANTS}
    curves = {key: {} for key in VARIANTS}
    models = {}
    for n_blocks in BLOCKS:
        depth = n_weight_layers(n_blocks)
        print(f"\n--- 深度 {depth} 层（{n_blocks} 个块） ---")
        for key in VARIANTS:
            t0 = time.time()
            model = make_model(key, n_blocks)
            model.fit(X_tr, y_tr, **FIT)
            tr_loss, tr_acc = model.evaluate(X_tr, y_tr)
            _, te_acc = model.evaluate(X_te, y_te)
            secs = time.time() - t0
            finite = bool(np.isfinite(tr_loss))
            results[key].append({"blocks": n_blocks, "depth": depth, "train_loss": tr_loss if finite else None,
                                 "train_acc": tr_acc, "test_acc": te_acc, "params": model.n_params,
                                 "train_sec": round(secs, 2)})
            curves[key][depth] = model.loss_
            models[(key, depth)] = model
            print(f"  [{key}] train_loss={tr_loss:.4f} train_acc={tr_acc:.2f}% test_acc={te_acc:.2f}% "
                  f"params={model.n_params} time={secs:.1f}s")

    depths = [n_weight_layers(b) for b in BLOCKS]
    print("\n汇总（训练集准确率，看的是“能不能学会”，不是泛化）：")
    for key in VARIANTS:
        print(f"  {key:<12s} " + "  ".join(f"{d}层={r['train_acc']:.1f}%" for d, r in zip(depths, results[key])))

    # ---- Figure 1: accuracy / loss vs depth ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for key, v in VARIANTS.items():
        axes[0].plot(depths, [r["train_acc"] for r in results[key]], "o-", lw=2, color=v["color"], label=v["label"])
        axes[1].plot(depths, [r["train_loss"] if r["train_loss"] is not None else np.nan for r in results[key]],
                     "o-", lw=2, color=v["color"], label=v["label"])
    axes[0].set(xlabel="Depth (Dense layers)", ylabel="Train accuracy (%)", title="Can the network fit the training set?", ylim=(25, 102))
    axes[0].axhline(100 / SPIRAL["n_classes"], color="#64748b", ls=":", lw=1)
    axes[1].set(xlabel="Depth (Dense layers)", ylabel="Train loss (log)", title="Final training loss", yscale="log")
    for ax in axes:
        ax.set_xticks(depths)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    img_depth = save_figure(fig, out / "res_depth.png")

    # ---- Figure 2: training curves per depth ----
    fig, axes = plt.subplots(1, 3, figsize=(17, 4.6), sharey=True)
    cmap = plt.get_cmap("viridis")
    for ax, (key, v) in zip(axes, VARIANTS.items()):
        for k, depth in enumerate(depths):
            ax.plot(curves[key][depth], color=cmap(k / max(1, len(depths) - 1)), lw=1.8, label=f"{depth} layers")
        ax.set(xlabel="Epoch", title=v["label"], yscale="log")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Train loss (log)")
    img_curves = save_figure(fig, out / "res_curves.png")

    # ---- Figure 3: signal propagation at init ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for key, v in VARIANTS.items():
        axes[0].plot(range(len(signal[key]["forward_std"])), signal[key]["forward_std"], "o-", ms=3, lw=2,
                     color=v["color"], label=v["label"])
        axes[1].plot(range(1, deep + 1), signal[key]["grad_norm"], "o-", ms=3, lw=2, color=v["color"], label=v["label"])
    axes[0].set(xlabel="Block (0 = after stem)", ylabel="std of h (log)", yscale="log",
                title=f"Forward: activation scale at init ({n_weight_layers(deep)} layers)")
    axes[1].set(xlabel="Block (1 = closest to input)", ylabel="‖dL/dh‖ (log)", yscale="log",
                title="Backward: gradient reaching each block at init")
    for ax in axes:
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    img_signal = save_figure(fig, out / "res_signal.png")

    # ---- Figure 4: decision boundaries ----
    mid = n_weight_layers(8)
    panels = [("plain", mid), ("plain", depths[-1]), ("residual", depths[-1]), ("residual_he", depths[-1])]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
    for ax, (key, depth) in zip(axes, panels):
        r = results[key][depths.index(depth)]
        plot_boundary(ax, models[(key, depth)], X_tr, y_tr, f"{VARIANTS[key]['label']}\n{depth} layers, train {r['train_acc']:.1f}%")
    img_bounds = save_figure(fig, out / "res_boundaries.png")

    plain, res = results["plain"], results["residual"]
    best_plain = max(r["train_acc"] for r in plain)
    passed = bool(
        grad_ok
        and res[-1]["train_acc"] >= 95.0
        and plain[-1]["train_acc"] <= res[-1]["train_acc"] - 20.0
        and plain[-1]["train_acc"] < best_plain - 5.0
    )
    metrics = {
        "dataset": f"{SPIRAL['n_classes']}-arm spiral, {SPIRAL['turns']} turns",
        "n_train": len(y_tr), "n_test": len(y_te), "width": WIDTH, "depths": depths,
        "gradient_check": {k: {"max_rel_err": v, "passed": v < 1e-5} for k, v in ck.items()},
        "signal_at_init": {k: {"depth": n_weight_layers(deep), **v} for k, v in signal.items()},
        "results": results,
        "images": [str(p.relative_to(out.parent.parent)) for p in (img_depth, img_curves, img_signal, img_bounds)],
        "passed": passed,
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
