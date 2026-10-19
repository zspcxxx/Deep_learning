#!/usr/bin/env python3
"""Lesson 10: detect cats in 32x32 colour photos (CIFAR-10 cat vs not-cat) with a hand-written CNN."""
from __future__ import annotations

import time

import numpy as np

from dllab.cnn import SimpleCNN, augment_flip_crop, gradient_check
from dllab.data_store import cached_dataset
from dllab.datasets import load_cat_dataset, load_cat_debug_batch
from dllab.io_utils import dump_metrics
from dllab.mlp import MulticlassMLP
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure

LESSON = "10_cnn"
SEED = 42
N_VAL = 1000
CNN_CFG = dict(channels=(8, 16, 32), hidden=64, n_classes=2, image_size=32, in_channels=3, dropout=0.3, seed=SEED)
CNN_FIT = dict(n_epochs=25, batch_size=64, lr=2e-3, min_lr=1e-4, warmup_epochs=1, weight_decay=1e-2, early_stop_patience=5)
# Same recipe as the lesson-09 tuned pipeline.
MLP_CFG = dict(
    learning_rate=0.0025, min_lr=0.0001, n_epochs=40, batch_size=128, warmup_epochs=5,
    lr_schedule="warmup_cosine", activation="gelu", init="he", norm="rmsnorm", loss_type="ce",
    optimizer="adamw", weight_decay=0.03, l1=5e-5, l2=3e-3, dropout=0.22, seed=SEED,
)
MLP_LAYERS = [32 * 32 * 3, 128, 64, 2]
MODELS = {
    "mlp": {"label": "MLP", "color": "#be123c"},
    "cnn": {"label": "CNN", "color": "#2563eb"},
    "cnn_aug": {"label": "CNN + augmentation", "color": "#0f766e"},
}
THRESHOLDS = np.linspace(0.05, 0.95, 19)
OCCLUSION_PATCH, OCCLUSION_STRIDE = 8, 2


def detection_stats(p_cat: np.ndarray, y: np.ndarray, threshold: float = 0.5) -> dict:
    pred = p_cat >= threshold
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    precision = tp / max(1, tp + fp)
    recall = tp / max(1, tp + fn)
    return {
        "accuracy": (tp + tn) / len(y) * 100, "precision": precision * 100, "recall": recall * 100,
        "f1": 2 * precision * recall / max(1e-12, precision + recall) * 100,
        "tp": tp, "fp": fp, "fn": fn, "tn": tn,
    }


def cat_logit(model: SimpleCNN, x: np.ndarray) -> np.ndarray:
    """log(P(cat) / P(not cat)); unlike the probability it does not saturate near 0 or 1."""
    p = np.clip(model.predict_proba(x), 1e-7, 1.0)
    return np.log(p[:, 1]) - np.log(p[:, 0])


def occlusion_map(model: SimpleCNN, img: np.ndarray) -> tuple[float, np.ndarray]:
    """Slide a mean-colour patch over the image; heat = how much the cat log-odds drop when that region is hidden."""
    _, h, w = img.shape
    base = float(model.predict_proba(img[None])[0, 1])
    coords = [(r, c) for r in range(0, h - OCCLUSION_PATCH + 1, OCCLUSION_STRIDE)
              for c in range(0, w - OCCLUSION_PATCH + 1, OCCLUSION_STRIDE)]
    batch = np.repeat(img[None], len(coords), axis=0)
    for k, (r, c) in enumerate(coords):
        batch[k, :, r : r + OCCLUSION_PATCH, c : c + OCCLUSION_PATCH] = 0.0  # 0 = mean colour after normalization
    drops = cat_logit(model, img[None])[0] - cat_logit(model, batch)
    heat, count = np.zeros((h, w)), np.zeros((h, w))
    for (r, c), dlt in zip(coords, drops):
        heat[r : r + OCCLUSION_PATCH, c : c + OCCLUSION_PATCH] += dlt
        count[r : r + OCCLUSION_PATCH, c : c + OCCLUSION_PATCH] += 1
    return base, heat / count


def run() -> dict:
    out = ensure_artifacts(LESSON)
    print("=" * 94)
    print("卷积神经网络（10）：检测图片里有没有猫（CIFAR-10 猫 vs 非猫）")
    print("=" * 94)

    # ---- 1. data ----
    d = load_cat_dataset(n_val=N_VAL, seed=SEED)
    names = d["class_names"]
    mean = d["x_train"].reshape(-1, 3).mean(axis=0) / 255.0
    std = d["x_train"].reshape(-1, 3).std(axis=0) / 255.0

    def to_input(x_uint8: np.ndarray) -> np.ndarray:
        return ((x_uint8 / 255.0 - mean) / std).transpose(0, 3, 1, 2).astype(np.float32)

    X = {s: to_input(d[f"x_{s}"]) for s in ("train", "val", "test")}
    Y = {s: d[f"y_{s}"] for s in ("train", "val", "test")}
    print(f"数据: data/cat/cat_vs_noncat.npz | 32x32 RGB | train={len(Y['train'])} val={len(Y['val'])} "
          f"test={len(Y['test'])} | 猫占比 train={Y['train'].mean():.1%} test={Y['test'].mean():.1%}")
    print(f"按通道归一化 mean(RGB)={np.round(mean, 4).tolist()} std(RGB)={np.round(std, 4).tolist()}")

    # ---- 2. gradient check on a cached debug batch ----
    dbg_x, dbg_y = cached_dataset(LESSON, "gradcheck_batch", load_cat_debug_batch, n=4, seed=0)
    ck_model = SimpleCNN(channels=(4, 6), hidden=8, n_classes=2, image_size=32, in_channels=3,
                         dropout=0.0, seed=1, dtype=np.float64)
    ck = gradient_check(ck_model, to_input(dbg_x).astype(np.float64), dbg_y, n_checks=10)
    grad_ok = all(err < 1e-5 for err in ck.values())
    print(f"\n梯度检查（{len(dbg_y)} 张调试图片，data/{LESSON}/gradcheck_batch.npz）：")
    for key, err in ck.items():
        print(f"  {key:<8s} max relative error = {err:.3e}   {'✓ 通过' if err < 1e-5 else '✗ 未通过'}")

    # ---- 3. models ----
    trained = {}
    print(f"\n--- [mlp] {MLP_LAYERS} GELU + RMSNorm + AdamW（09 课 tuned 配方） ---")
    t0 = time.time()
    mlp = MulticlassMLP(MLP_LAYERS, **MLP_CFG)
    mlp.fit(X["train"].reshape(len(Y["train"]), -1), Y["train"], X_val=X["val"].reshape(len(Y["val"]), -1),
            y_val=Y["val"], early_stop_patience=8, verbose=True, print_every=10)
    mlp_params = int(sum(w.size for w in mlp.W) + sum(b.size for b in mlp.b) + sum(g.size for g in mlp.gamma))
    trained["mlp"] = (mlp, lambda a: a.reshape(len(a), -1), mlp_params, time.time() - t0)

    for key, aug in (("cnn", None), ("cnn_aug", augment_flip_crop)):
        tag = "随机水平翻转 + 随机平移裁剪" if aug else "不做数据增强"
        print(f"\n--- [{key}] conv3x3(8)-pool-conv3x3(16)-pool-conv3x3(32)-pool-FC(64)-FC(2)，{tag} ---")
        t0 = time.time()
        cnn = SimpleCNN(**CNN_CFG)
        cnn.fit(X["train"], Y["train"], X["val"], Y["val"], augment=aug, seed=SEED, verbose=True, **CNN_FIT)
        trained[key] = (cnn, lambda a: a, cnn.n_params, time.time() - t0)

    # ---- 4. evaluation ----
    print("\n评估（测试集，阈值 0.5：P(猫) ≥ 0.5 判为猫）：")
    results = {}
    for key, (model, prep, n_params, secs) in trained.items():
        p_test = model.predict_proba(prep(X["test"]))[:, 1]
        stats = detection_stats(p_test, Y["test"])
        train_acc = model.evaluate(prep(X["train"]), Y["train"])[1]
        val_acc = model.evaluate(prep(X["val"]), Y["val"])[1]
        epochs = (model.best_epoch_ + 1) if model.best_epoch_ is not None else len(model.loss_)
        results[key] = {"p_test": p_test, "train_acc": train_acc, "val_acc": val_acc, "n_params": n_params,
                        "epochs": int(epochs), "train_sec": round(secs, 1), **stats}
        print(f"  [{key}] acc={stats['accuracy']:.2f}% precision={stats['precision']:.2f}% recall={stats['recall']:.2f}% "
              f"f1={stats['f1']:.2f}% train_acc={train_acc:.2f}% params={n_params} best_epoch={epochs} time={secs:.1f}s")

    best = results["cnn_aug"]
    neg = Y["test"] == 0
    fp_by_class = {}
    for c in sorted(set(d["source_test"][neg].tolist())):
        sel = neg & (d["source_test"] == c)
        fp_by_class[names[c]] = float(np.mean(best["p_test"][sel] >= 0.5) * 100)
    ranked = sorted(fp_by_class.items(), key=lambda kv: -kv[1])
    print("  非猫图片被误判为猫的比例（cnn_aug）：" + ", ".join(f"{k}={v:.1f}%" for k, v in ranked))
    sweep = [detection_stats(best["p_test"], Y["test"], t) for t in THRESHOLDS]

    # ---- Figure 1: curves ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for key, (model, *_rest) in trained.items():
        color, label = MODELS[key]["color"], MODELS[key]["label"]
        ep = range(1, len(model.loss_) + 1)
        axes[0].plot(ep, model.loss_, color=color, lw=2, label=f"{label} train")
        axes[0].plot(range(1, len(model.val_loss_) + 1), model.val_loss_, color=color, lw=2, ls="--", label=f"{label} val")
        axes[1].plot(range(1, len(model.val_accuracy_) + 1), model.val_accuracy_, color=color, lw=2, label=label)
    axes[0].set(xlabel="Epoch", ylabel="Cross-entropy", title="Loss (solid = train, dashed = val)")
    axes[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="Validation accuracy", ylim=(50, 90))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    img_curves = save_figure(fig, out / "cnn_curves.png")

    # ---- Figure 2: filters and feature maps ----
    cnn_aug = trained["cnn_aug"][0]
    cat_idx = int(np.argmax(np.where(Y["test"] == 1, best["p_test"], -1)))
    maps = cnn_aug.feature_maps(X["test"][cat_idx : cat_idx + 1])
    w1 = cnn_aug.layers[0].W
    fig = plt.figure(figsize=(16, 7.5))
    grid = fig.add_gridspec(4, 16, hspace=0.55)
    ax = fig.add_subplot(grid[0:2, 0:3])
    ax.imshow(d["x_test"][cat_idx])
    ax.set_title(f"input: cat, P(cat)={best['p_test'][cat_idx]:.2f}")
    ax.axis("off")
    for i in range(w1.shape[0]):
        k = w1[i].transpose(1, 2, 0)
        ax = fig.add_subplot(grid[0, 4 + i])
        ax.imshow((k - k.min()) / (k.max() - k.min() + 1e-12))
        ax.set_title(f"k{i}", fontsize=8)
        ax.axis("off")
        ax = fig.add_subplot(grid[1, 4 + i])
        ax.imshow(maps["relu1"][0, i], cmap="viridis")
        ax.axis("off")
    for i in range(maps["relu2"].shape[1]):
        ax = fig.add_subplot(grid[2 + i // 8, 4 + i % 8])
        ax.imshow(maps["relu2"][0, i], cmap="viridis")
        ax.set_title(f"c2-{i}", fontsize=7)
        ax.axis("off")
    fig.text(0.29, 0.93, "conv1 3x3 RGB kernels (row 1) and their ReLU feature maps 32x32 (row 2)", fontsize=10)
    fig.text(0.29, 0.47, "conv2 ReLU feature maps 16x16 (16 channels)", fontsize=10)
    img_filters = save_figure(fig, out / "cnn_filters.png")

    # ---- Figure 3: detections on sample test images ----
    rng = np.random.RandomState(SEED)
    sample = np.concatenate([rng.choice(np.where(Y["test"] == c)[0], 8, replace=False) for c in (1, 0)])
    fig, axes = plt.subplots(2, 8, figsize=(16, 5.4), gridspec_kw={"hspace": 0.35})
    for ax, i in zip(axes.ravel(), sample):
        p = best["p_test"][i]
        ok = (p >= 0.5) == (Y["test"][i] == 1)
        ax.imshow(d["x_test"][i])
        ax.set_title(f"{names[d['source_test'][i]]}\nP(cat)={p:.2f}", fontsize=9, color="#0f766e" if ok else "#be123c")
        ax.axis("off")
    fig.suptitle("CNN + augmentation on test images (top: cats, bottom: not cats; red = wrong at threshold 0.5)")
    img_detect = save_figure(fig, out / "cnn_detect.png")

    # ---- Figure 4: occlusion sensitivity ----
    cats_ok = np.where((Y["test"] == 1) & (best["p_test"] >= 0.5))[0]
    picks = list(cats_ok[np.argsort(-best["p_test"][cats_ok])[:: max(1, len(cats_ok) // 5)][:5]])
    fooled = np.where(neg & (best["p_test"] >= 0.5))[0]
    if len(fooled):
        picks.append(int(fooled[np.argmax(best["p_test"][fooled])]))
    fig, axes = plt.subplots(2, len(picks), figsize=(2.6 * len(picks), 6.4), gridspec_kw={"hspace": 0.3})
    for j, i in enumerate(picks):
        p0, heat = occlusion_map(cnn_aug, X["test"][i])
        axes[0, j].imshow(d["x_test"][i])
        axes[0, j].set_title(f"{names[d['source_test'][i]]}  P(cat)={p0:.2f}", fontsize=9)
        axes[1, j].imshow(d["x_test"][i])
        axes[1, j].imshow(heat, cmap="jet", alpha=0.55, vmin=0, vmax=max(1e-3, heat.max()))
        axes[1, j].set_title(f"max log-odds drop {heat.max():.1f}", fontsize=9)
        for ax in axes[:, j]:
            ax.axis("off")
    fig.suptitle(f"Occlusion: hide an {OCCLUSION_PATCH}x{OCCLUSION_PATCH} patch; red = the cat score drops most")
    img_occl = save_figure(fig, out / "cnn_occlusion.png")

    # ---- Figure 5: detection metrics ----
    fig, axes = plt.subplots(2, 2, figsize=(14, 9.5))
    keys = list(MODELS)
    xs = np.arange(4)
    for k, key in enumerate(keys):
        r = results[key]
        vals = [r["accuracy"], r["precision"], r["recall"], r["f1"]]
        axes[0, 0].bar(xs + (k - 1) * 0.27, vals, 0.27, color=MODELS[key]["color"],
                       label=f"{MODELS[key]['label']} ({r['n_params']:,} params)")
    axes[0, 0].set_xticks(xs)
    axes[0, 0].set_xticklabels(["accuracy", "precision", "recall", "F1"])
    axes[0, 0].set(ylim=(50, 95), ylabel="%", title="Test set, threshold 0.5")
    axes[0, 0].legend(fontsize=8)
    axes[0, 0].grid(axis="y", alpha=0.3)
    axes[0, 1].plot(THRESHOLDS, [s["precision"] for s in sweep], "o-", color="#7c3aed", label="precision")
    axes[0, 1].plot(THRESHOLDS, [s["recall"] for s in sweep], "o-", color="#ea580c", label="recall")
    axes[0, 1].axvline(0.5, color="#64748b", ls="--", lw=1)
    axes[0, 1].set(xlabel="Threshold on P(cat)", ylabel="%", title="CNN + aug: precision / recall vs threshold", ylim=(0, 102))
    axes[0, 1].legend()
    axes[0, 1].grid(alpha=0.3)
    axes[1, 0].bar([k for k, _ in ranked], [v for _, v in ranked], color="#be123c")
    axes[1, 0].set(ylabel="% predicted as cat", title="False alarms by true class (CNN + aug)")
    axes[1, 0].tick_params(axis="x", rotation=30)
    axes[1, 0].grid(axis="y", alpha=0.3)
    cm = np.array([[best["tn"], best["fp"]], [best["fn"], best["tp"]]])
    axes[1, 1].imshow(cm, cmap="Blues")
    for r in range(2):
        for c in range(2):
            axes[1, 1].text(c, r, cm[r, c], ha="center", va="center", fontsize=14,
                            color="white" if cm[r, c] > cm.max() / 2 else "#0f172a")
    axes[1, 1].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["not cat", "cat"], yticklabels=["not cat", "cat"],
                   xlabel="Predicted", ylabel="True", title="Confusion matrix (CNN + aug)")
    img_metrics = save_figure(fig, out / "cnn_metrics.png")

    mlp_r, cnn_r = results["mlp"], results["cnn_aug"]
    passed = bool(
        grad_ok
        and cnn_r["accuracy"] >= 75.0
        and cnn_r["accuracy"] > mlp_r["accuracy"]
        and cnn_r["n_params"] < mlp_r["n_params"]
    )
    metrics = {
        "dataset": "CIFAR-10 cat vs not-cat (32x32 RGB)",
        "n_train": len(Y["train"]), "n_val": len(Y["val"]), "n_test": len(Y["test"]),
        "gradient_check": {k: {"max_rel_err": v, "passed": v < 1e-5} for k, v in ck.items()},
        "models": {k: {kk: vv for kk, vv in r.items() if kk != "p_test"} for k, r in results.items()},
        "false_alarm_by_class": dict(ranked),
        "threshold_sweep": [{"threshold": round(float(t), 2), "precision": s["precision"], "recall": s["recall"]}
                            for t, s in zip(THRESHOLDS, sweep)],
        "images": [str(p.relative_to(out.parent.parent))
                   for p in (img_curves, img_filters, img_detect, img_occl, img_metrics)],
        "passed": passed,
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
