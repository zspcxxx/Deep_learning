#!/usr/bin/env python3
"""Capstone (Lesson 09): MNIST handwritten digit recognition integrating all 11 core techniques."""
from __future__ import annotations

import numpy as np

from dllab.datasets import load_mnist_dataset
from dllab.io_utils import dump_metrics
from dllab.mlp import MulticlassMLP
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure

LAYER_SIZES = [784, 128, 64, 10]
N_EPOCHS = 35
BATCH_SIZE = 128
SEED = 42
N_TRAIN = 4000
N_VAL = 800
N_TEST = 1200

# Core Knowledge Table (11 items matching the course summary diagram)
RECIPE = [
    ("参数初始化", "He / Xavier", "信号尺度稳定，梯度可传播"),
    ("激活函数", "Sigmoid / ReLU / GELU / SiLU", "引入非线性，保持梯度通畅"),
    ("Dropout", "随机关闭神经元", "防止过拟合，特征冗余"),
    ("归一化", "BN / LN / RMSNorm", "稳定激活分布，加速收敛"),
    ("损失函数", "MSE / Cross-Entropy", "量化误差，提供梯度信号"),
    ("L1 / L2 正则化", "损失加惩罚项", "限制模型复杂度"),
    ("反向传播", "链式法则", "计算每个参数的梯度"),
    ("优化器", "SGD+Momentum / AdamW", "高效稳定地更新参数"),
    ("学习率调度", "Warmup + Cosine", "不同阶段适配不同步长"),
    ("权重衰减", "AdamW 解耦衰减", "参数持续向零收缩，提升泛化"),
    ("Early Stopping", "监控验证集", "在过拟合之前停止训练"),
]

# English summary rows for matplotlib figure rendering (avoids CJK font missing warnings)
RECIPE_PLOT_ROWS = [
    ("1. Param Init", "He / Xavier", "Stable signal scale & gradient flow"),
    ("2. Activation", "Sigmoid / ReLU / GELU / SiLU", "Non-linearity & smooth gradients"),
    ("3. Dropout", "Randomly drop neurons (p=0.12)", "Prevent overfitting & feature co-adaptation"),
    ("4. Normalization", "BN / LN / RMSNorm", "Stabilize activation distribution & speed up"),
    ("5. Loss Function", "MSE / Cross-Entropy", "Quantify error & provide gradient signal"),
    ("6. L1 / L2 Reg", "Loss penalty terms", "Constrain model complexity"),
    ("7. Backprop", "Chain rule", "Compute exact gradients for all parameters"),
    ("8. Optimizer", "SGD+Momentum / AdamW", "Update parameters efficiently & stably"),
    ("9. LR Schedule", "Warmup + Cosine Annealing", "Adapt step size across training stages"),
    ("10. Weight Decay", "AdamW decoupled decay", "Shrink weights toward zero for generalization"),
    ("11. Early Stopping", "Monitor validation set", "Stop training before overfitting"),
]

PIPELINES = {
    "naive": {
        "label": "naive (large init + Sigmoid + MSE + raw SGD, no norm/reg)",
        "color": "#be123c",
        "standardize": False,
        "kwargs": dict(
            learning_rate=0.08,
            min_lr=0.08,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            warmup_epochs=0,
            lr_schedule="constant",
            activation="sigmoid",
            init="large",
            norm=None,
            loss_type="mse",
            optimizer="sgd",
            weight_decay=0.0,
            l1=0.0,
            l2=0.0,
            dropout=0.0,
            seed=SEED,
        ),
        "early_stop": None,
    },
    "momentum_bn": {
        "label": "momentum_bn (He + ReLU + BN + CE + SGD+Momentum + L2 + Cosine)",
        "color": "#2563eb",
        "standardize": True,
        "kwargs": dict(
            learning_rate=0.06,
            min_lr=0.002,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            warmup_epochs=4,
            lr_schedule="warmup_cosine",
            activation="relu",
            init="he",
            norm="bn",
            loss_type="ce",
            optimizer="momentum",
            momentum=0.9,
            weight_decay=0.0,
            l1=1e-4,
            l2=5e-3,
            dropout=0.10,
            seed=SEED,
        ),
        "early_stop": 10,
    },
    "tuned": {
        "label": "tuned (He/Xavier + GELU + RMSNorm + CE + AdamW + WarmupCosine + Drop/L2)",
        "color": "#0f766e",
        "standardize": True,
        "kwargs": dict(
            learning_rate=0.0025,
            min_lr=0.0001,
            n_epochs=N_EPOCHS,
            batch_size=BATCH_SIZE,
            warmup_epochs=5,
            lr_schedule="warmup_cosine",
            activation="gelu",
            init="he",
            norm="rmsnorm",
            loss_type="ce",
            optimizer="adamw",
            weight_decay=0.03,
            l1=5e-5,
            l2=3e-3,
            dropout=0.22,
            seed=SEED,
        ),
        "early_stop": 10,
    },
}


def standardize_fit(X: np.ndarray):
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True)
    std = np.where(std < 1.0, 1.0, std)
    return mean, std


def apply_standardize(X: np.ndarray, mean: np.ndarray, std: np.ndarray):
    return (X - mean) / std


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int = 10) -> np.ndarray:
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred):
        cm[int(t), int(p)] += 1
    return cm


def run() -> dict:
    out = ensure_artifacts("09_full_pipeline")
    X_train, y_train, X_val, y_val, X_test, y_test, data_source = load_mnist_dataset(
        n_train=N_TRAIN, n_val=N_VAL, n_test=N_TEST, seed=SEED
    )

    print("=" * 94)
    print("综合实战（09）：使用核心知识体系实现 MNIST 手写数字识别（10 分类）")
    print("=" * 94)
    print(
        f"数据源: {data_source} | 输入维度: 28x28=784 | 网络结构: {LAYER_SIZES} | "
        f"train={len(y_train)} val={len(y_val)} test={len(y_test)}"
    )
    print("\n总结：核心知识清单（11 项关键技术）：")
    print(f"  {'环节':<16s} | {'关键技术':<28s} | {'解决的问题'}")
    print("  " + "-" * 82)
    for stage, tech, purpose in RECIPE:
        print(f"  {stage:<16s} | {tech:<28s} | {purpose}")

    mean, std = standardize_fit(X_train)
    Xt_std = apply_standardize(X_train, mean, std)
    Xv_std = apply_standardize(X_val, mean, std)
    Xe_std = apply_standardize(X_test, mean, std)

    # Raw pixel scale [0, 255] divided only by 255 or unstandardized for naive
    Xt_raw = X_train / 255.0
    Xv_raw = X_val / 255.0
    Xe_raw = X_test / 255.0

    results = {}
    for key, cfg in PIPELINES.items():
        print(f"\n--- [{key}] {cfg['label']} ---")
        if cfg["standardize"]:
            Xt, Xv, Xe = Xt_std, Xv_std, Xe_std
        else:
            Xt, Xv, Xe = Xt_raw, Xv_raw, Xe_raw

        model = MulticlassMLP(LAYER_SIZES, **cfg["kwargs"])
        model.fit(
            Xt,
            y_train,
            X_val=Xv,
            y_val=y_val,
            early_stop_patience=cfg["early_stop"],
            verbose=True,
            print_every=10,
        )

        train_loss, train_acc = model.evaluate(Xt, y_train)
        val_loss, val_acc = model.evaluate(Xv, y_val)
        test_loss, test_acc = model.evaluate(Xe, y_test)
        y_pred_test = model.predict(Xe)
        probs_test = model.predict_proba(Xe)
        epochs = (model.best_epoch_ + 1) if model.best_epoch_ is not None else len(model.loss_)
        finite = bool(
            all(np.all(np.isfinite(w)) for w in model.W)
            and np.isfinite(train_loss)
            and np.isfinite(test_loss)
        )
        w_norm = float(np.sqrt(sum(float(np.sum(w * w)) for w in model.W)))
        print(
            f"  → train={train_acc:.2f}% | val={val_acc:.2f}% | test={test_acc:.2f}% | "
            f"best_epoch={epochs} | ||W||={w_norm:.2f} | finite={finite}"
        )
        results[key] = {
            "label": cfg["label"],
            "loss": [float(v) for v in model.loss_],
            "val_loss": [float(v) for v in model.val_loss_],
            "train_curve": [float(v) for v in model.accuracy_],
            "val_curve": [float(v) for v in model.val_accuracy_],
            "lr_history": [float(v) for v in model.lr_history_],
            "train_acc": train_acc,
            "val_acc": val_acc,
            "test_acc": test_acc,
            "test_loss": test_loss,
            "epochs": int(epochs),
            "weight_norm": w_norm,
            "finite": finite,
            "y_pred": y_pred_test,
            "probs": probs_test,
        }

    # ---- Figure 1: Loss, Val Accuracy & Warmup+Cosine LR Schedule ----
    fig1, axes1 = plt.subplots(1, 3, figsize=(16, 4.6))
    for key, cfg in PIPELINES.items():
        r = results[key]
        axes1[0].plot(r["loss"], color=cfg["color"], linewidth=2.0, label=f"{key} (train)")
        axes1[1].plot(r["val_curve"], color=cfg["color"], linewidth=2.0, label=f"{key} (val)")
        axes1[2].plot(r["lr_history"], color=cfg["color"], linewidth=2.0, label=key)
    axes1[0].set(xlabel="Epoch", ylabel="Loss", title="Training Loss Curve")
    axes1[0].legend(fontsize=8)
    axes1[0].grid(alpha=0.3)

    axes1[1].set(xlabel="Epoch", ylabel="Accuracy (%)", title="MNIST Validation Accuracy", ylim=(0, 102))
    axes1[1].legend(fontsize=8)
    axes1[1].grid(alpha=0.3)

    axes1[2].set(xlabel="Epoch", ylabel="Learning Rate", title="LR Schedule (Warmup + Cosine)")
    axes1[2].set_yscale("log")
    axes1[2].legend(fontsize=8)
    axes1[2].grid(alpha=0.3)
    img_curves = save_figure(fig1, out / "pipeline_curves.png")

    # ---- Figure 2: Hold-out Accuracy Bar Chart + Confusion Matrix (Tuned) ----
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 5.2))
    keys = list(PIPELINES.keys())
    xs = np.arange(3)
    width = 0.25
    for i, key in enumerate(keys):
        r = results[key]
        vals = [r["train_acc"], r["val_acc"], r["test_acc"]]
        bars = axes2[0].bar(
            xs + (i - 1.0) * width,
            vals,
            width,
            label=key,
            color=PIPELINES[key]["color"],
        )
        for b in bars:
            h = b.get_height()
            axes2[0].text(
                b.get_x() + b.get_width() / 2.0,
                h + 1.0,
                f"{h:.1f}%",
                ha="center",
                va="bottom",
                fontsize=8,
            )
    axes2[0].set_xticks(xs)
    axes2[0].set_xticklabels(["Train", "Validation", "Test"])
    axes2[0].set(ylabel="Accuracy (%)", title="MNIST 10-Class Accuracy Comparison", ylim=(0, 110))
    axes2[0].legend(fontsize=9)
    axes2[0].grid(axis="y", alpha=0.3)

    cm = compute_confusion_matrix(y_test, results["tuned"]["y_pred"], n_classes=10)
    im = axes2[1].imshow(cm, cmap="Blues")
    axes2[1].set(
        xlabel="Predicted Digit",
        ylabel="True Digit",
        title=f"Tuned Confusion Matrix (Test Acc = {results['tuned']['test_acc']:.2f}%)",
    )
    axes2[1].set_xticks(range(10))
    axes2[1].set_yticks(range(10))
    thresh = cm.max() * 0.5
    for r_idx in range(10):
        for c_idx in range(10):
            val = int(cm[r_idx, c_idx])
            if val > 0:
                axes2[1].text(
                    c_idx,
                    r_idx,
                    str(val),
                    ha="center",
                    va="center",
                    color="white" if val > thresh else "#0f172a",
                    fontsize=8,
                )
    fig2.colorbar(im, ax=axes2[1], fraction=0.046, pad=0.04)
    img_bars = save_figure(fig2, out / "pipeline_metrics.png")

    # ---- Figure 3: Sample MNIST Predictions (2x8 Grid) ----
    fig3, axes3 = plt.subplots(2, 8, figsize=(15, 4.4))
    tuned_preds = results["tuned"]["y_pred"]
    tuned_probs = results["tuned"]["probs"]
    naive_preds = results["naive"]["y_pred"]
    for idx, ax in enumerate(axes3.ravel()):
        img28 = X_test[idx].reshape(28, 28)
        true_d = int(y_test[idx])
        pred_d = int(tuned_preds[idx])
        conf = float(tuned_probs[idx, pred_d] * 100.0)
        naive_d = int(naive_preds[idx])
        ax.imshow(img28, cmap="gray_r")
        ok = pred_d == true_d
        title_color = "#0f766e" if ok else "#be123c"
        ax.set_title(
            f"T:{true_d} P:{pred_d} ({conf:.0f}%)\nnaive:{naive_d}",
            fontsize=8.5,
            color=title_color,
        )
        ax.axis("off")
    fig3.suptitle("MNIST Test Samples: True (T) vs Tuned Prediction (P) & Confidence vs Naive", fontsize=12)
    img_samples = save_figure(fig3, out / "pipeline_boundaries.png")

    # ---- Figure 4: Core Knowledge Table Card ----
    fig4, ax4 = plt.subplots(figsize=(12, 5.2))
    ax4.axis("off")
    header = f"{'Stage':<20s} | {'Key Technique':<30s} | {'Problem Solved'}"
    sep = "-" * 92
    lines = [
        "Core Deep Learning Techniques Applied to MNIST Digit Recognition:",
        "",
        header,
        sep,
    ]
    for stage, tech, prob in RECIPE_PLOT_ROWS:
        lines.append(f"{stage:<20s} | {tech:<30s} | {prob}")
    lines.append(sep)
    lines.append(
        f"Result Summary -> naive: {results['naive']['test_acc']:.2f}%  |  "
        f"momentum_bn: {results['momentum_bn']['test_acc']:.2f}%  |  "
        f"tuned (AdamW+RMSNorm+GELU): {results['tuned']['test_acc']:.2f}%"
    )
    ax4.text(
        0.02,
        0.96,
        "\n".join(lines),
        va="top",
        ha="left",
        family="monospace",
        fontsize=9.8,
        transform=ax4.transAxes,
    )
    ax4.set_title("Summary of Core Knowledge (11 Pillars in Lesson 09 MNIST Pipeline)")
    img_recipe = save_figure(fig4, out / "pipeline_recipe.png")

    naive = results["naive"]
    momentum_bn = results["momentum_bn"]
    tuned = results["tuned"]
    passed = bool(
        tuned["finite"]
        and momentum_bn["finite"]
        and tuned["test_acc"] >= 90.0
        and momentum_bn["test_acc"] >= 88.0
        and tuned["test_acc"] > naive["test_acc"] + 25.0
    )

    summary = []
    print("\n汇总对比：")
    for key, cfg in PIPELINES.items():
        r = results[key]
        kw = cfg["kwargs"]
        summary.append(
            {
                "pipeline": key,
                "label": cfg["label"],
                "standardize": cfg["standardize"],
                "init": kw["init"],
                "activation": kw["activation"],
                "norm": kw["norm"] or "none",
                "loss_type": kw["loss_type"],
                "optimizer": kw["optimizer"],
                "lr": kw["learning_rate"],
                "lr_schedule": kw["lr_schedule"],
                "weight_decay": kw["weight_decay"],
                "l1": kw["l1"],
                "l2": kw["l2"],
                "dropout": kw["dropout"],
                "early_stop": cfg["early_stop"],
                "train_acc": round(r["train_acc"], 2),
                "val_acc": round(r["val_acc"], 2),
                "test_acc": round(r["test_acc"], 2),
                "epochs": r["epochs"],
                "weight_norm": round(r["weight_norm"], 2),
                "finite": r["finite"],
            }
        )
        print(
            f"  {key:<12s} | test={r['test_acc']:6.2f}% | val={r['val_acc']:6.2f}% | "
            f"train={r['train_acc']:6.2f}% | epochs={r['epochs']:2d} | ||W||={r['weight_norm']:6.2f}"
        )

    metrics = {
        "dataset": "MNIST (28x28 handwritten digits, 10 classes)",
        "data_source": data_source,
        "architecture": LAYER_SIZES,
        "n_train": int(len(y_train)),
        "n_val": int(len(y_val)),
        "n_test": int(len(y_test)),
        "recipe": [{"stage": s, "tech": t, "purpose": p} for s, t, p in RECIPE],
        "pipelines": summary,
        "images": [
            str(img_curves.relative_to(out.parent.parent)),
            str(img_bars.relative_to(out.parent.parent)),
            str(img_samples.relative_to(out.parent.parent)),
            str(img_recipe.relative_to(out.parent.parent)),
        ],
        "passed": passed,
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
