#!/usr/bin/env python3
"""Lesson 15: assemble a small Transformer from lessons 01–14, then remove one part at a time."""
from __future__ import annotations

import time

import numpy as np

from dllab.data_store import cached_dataset
from dllab.io_utils import dump_metrics
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure
from dllab.rnn import RecurrentNet
from dllab.transformer import MLPBaseline, TinyTransformer, gradient_check, make_sort_dataset, train

LESSON = "15_transformer"
SEQ_LEN, VOCAB = 8, 10
N_TRAIN, N_VAL, N_TEST = 1000, 500, 2000
MODEL = dict(d_model=32, n_heads=4, n_layers=4, d_ff=64, dropout=0.1)
FIT = dict(steps=600, batch_size=128, lr=0.1, min_lr=1e-3, warmup=100, weight_decay=0.01, clip=1.0,
           eval_every=50, patience=6)
SEED = 0

# Each ablation changes exactly one thing relative to the full model.
ABLATIONS = [
    ("no_pos", "去掉位置编码", "14", {"model": {"positional": False}}),
    ("no_residual", "去掉残差连接", "11", {"model": {"residual": False}}),
    ("no_layernorm", "去掉 LayerNorm", "07", {"model": {"layer_norm": False}}),
    ("no_warmup", "去掉学习率 warmup", "05 / 09", {"fit": {"warmup": 1}}),
    ("big_init", "初始化放大 4 倍", "06", {"model": {"init_scale": 4.0}}),
    ("no_reg", "去掉 Dropout 和权重衰减", "08", {"model": {"dropout": 0.0}, "fit": {"weight_decay": 0.0}}),
    ("linear_ffn", "FFN 去掉激活函数", "02 / 04", {"model": {"activation": "linear"}}),
]
CRITICAL = ("no_pos", "no_residual", "no_layernorm", "no_warmup", "big_init")


def datasets():
    return {
        split: cached_dataset(LESSON, f"sort_{split}", make_sort_dataset, n, SEQ_LEN, VOCAB, seed=seed)
        for split, n, seed in (("train", N_TRAIN, 0), ("val", N_VAL, 1), ("test", N_TEST, 2))
    }


def fmt(ev: dict) -> str:
    return f"token={ev['token_acc']:.1f}% seq={ev['seq_acc']:.1f}%"


def run_transformer(data, label: str, model_kw=None, fit_kw=None, quiet: bool = False):
    model = TinyTransformer(VOCAB, SEQ_LEN, **{**MODEL, **(model_kw or {})}, seed=SEED, dtype=np.float32)
    fit = {**FIT, **(fit_kw or {})}

    def log(step, loss, lr_t, ev):
        if not quiet or step % 200 == 0:
            print(f"  step {step:4d} | loss={loss:.3f} | lr={lr_t:.4f} | val {fmt(ev)}")

    t0 = time.time()
    hist = train(model, *data["train"], *data["val"], seed=SEED, log_fn=log, **fit)
    test = model.evaluate(*data["test"])
    diverged = not np.isfinite(hist["loss"]).all() or max(hist["loss"]) > 10.0
    print(f"  [{label}] test {fmt(test)} | params={model.n_params} | 最优步={hist['best_step']}"
          f"{' | 训练中损失爆到 %.1f' % max(hist['loss']) if diverged else ''} | {time.time() - t0:.1f}s")
    return model, hist, {**test, "params": model.n_params, "best_step": hist["best_step"],
                         "stopped_at": hist["stopped_at"], "max_train_loss": float(np.nanmax(hist["loss"])),
                         "diverged": bool(diverged), "train_sec": round(time.time() - t0, 1)}


def rnn_io(X, Y):
    """LSTM reads the 8 digits, then gets 8 'write' steps (token VOCAB) and outputs the sorted digits."""
    n = len(X)
    x = np.zeros((n, 2 * SEQ_LEN, VOCAB + 1), dtype=np.float32)
    x[np.arange(n)[:, None], np.arange(SEQ_LEN)[None, :], X] = 1.0
    x[:, SEQ_LEN:, VOCAB] = 1.0
    y = np.concatenate([np.zeros_like(Y), Y], axis=1)
    mask = np.concatenate([np.zeros((n, SEQ_LEN)), np.ones((n, SEQ_LEN))], axis=1).astype(np.float32)
    return x, y, mask


def lstm_eval(model, X, Y) -> dict:
    logits, _ = model.logits(rnn_io(X, Y)[0], "all")
    pred = np.asarray(logits)[:, SEQ_LEN:].argmax(-1)
    return {"token_acc": float((pred == Y).mean() * 100), "seq_acc": float((pred == Y).all(1).mean() * 100)}


def run_lstm(data):
    X_tr, Y_tr = data["train"]
    model = RecurrentNet("lstm", VOCAB + 1, 64, VOCAB, seed=SEED, dtype=np.float32)
    rng = np.random.RandomState(SEED)

    def batch(step):
        idx = rng.randint(0, len(X_tr), size=FIT["batch_size"])
        return rnn_io(X_tr[idx], Y_tr[idx])

    def log(step, loss, norm):
        print(f"  step {step:4d} | loss={loss:.3f} | val {fmt(lstm_eval(model, *data['val']))}")

    t0 = time.time()
    model.fit(batch, 800, lr=1e-2, clip=1.0, mode="all", log_every=200, log_fn=log)
    test = lstm_eval(model, *data["test"])
    print(f"  [LSTM] test {fmt(test)} | params={model.n_params} | {time.time() - t0:.1f}s")
    return {**test, "params": model.n_params, "train_sec": round(time.time() - t0, 1)}


def plot_map(out):
    fig, ax = plt.subplots(figsize=(13, 7.4))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 10.6)
    ax.axis("off")

    def box(x, y, w, text, lessons, color):
        ax.add_patch(plt.Rectangle((x, y), w, 0.62, fc=color, ec="#334155", lw=1.1, zorder=2))
        ax.text(x + 0.15, y + 0.31, text, va="center", fontsize=10, zorder=3)
        ax.text(x + w - 0.12, y + 0.31, lessons, va="center", ha="right", fontsize=9.5,
                color="#1f6feb", fontweight="bold", zorder=3)

    stack = [
        ("Linear + softmax per position, cross-entropy", "L01 L09", "#e2e8f0"),
        ("Final LayerNorm", "L07", "#fef3c7"),
        ("+ residual shortcut", "L11", "#ede9fe"),
        ("FFN: Linear - GELU - Linear (same MLP at every position)", "L02 L04 L10", "#dcfce7"),
        ("LayerNorm", "L07", "#fef3c7"),
        ("+ residual shortcut", "L11", "#ede9fe"),
        ("Multi-head self-attention  softmax(QK^T / sqrt(d)) V", "L13 L14", "#ccfbf1"),
        ("LayerNorm", "L07", "#fef3c7"),
        ("Token embedding + learned position vectors", "L14", "#dbeafe"),
        ("Input: 8 digits   ->   output: the same digits sorted", "", "#f8fafc"),
    ]
    y = 9.6
    for i, (text, lessons, color) in enumerate(stack):
        box(0.3, y, 7.4, text, lessons, color)
        if i < len(stack) - 1:
            ax.annotate("", xy=(4.0, y), xytext=(4.0, y - 0.38),
                        arrowprops=dict(arrowstyle="->", color="#334155", lw=1.2))
        y -= 1.0
    ax.add_patch(plt.Rectangle((0.15, 3.48), 7.7, 4.86, fill=False, ec="#64748b", ls="--", lw=1.2))
    ax.text(7.95, 5.9, "x 4 blocks", rotation=90, va="center", fontsize=10, color="#475569")

    train_parts = [
        ("Init: weights ~ N(0, 1/fan_in)", "L06"),
        ("Backprop by hand + gradient check", "L03"),
        ("AdamW, gradient clipping", "L09 L12"),
        ("Warmup + cosine learning rate", "L05 L09"),
        ("Dropout + weight decay", "L08"),
        ("Train / val / test, early stopping", "L08 L09"),
        ("Baselines: MLP, LSTM read-then-write", "L02 L12 L13"),
    ]
    ax.text(8.7, 9.95, "Training & evaluation", fontsize=11, fontweight="bold")
    for i, (text, lessons) in enumerate(train_parts):
        box(8.6, 9.0 - i * 0.92, 4.2, text, lessons, "#fff7ed")
    ax.set_title("Where each lesson lives inside the model", fontsize=13)
    return save_figure(fig, out / "capstone_map.png")


def plot_training(out, hist):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.4))
    axes[0].plot(hist["loss"], color="#be123c", lw=1, alpha=0.8)
    axes[0].set(yscale="log", xlabel="Step", ylabel="Train loss (log)", title="Training loss")
    steps = [v["step"] for v in hist["val"]]
    axes[1].plot(steps, [v["seq_acc"] for v in hist["val"]], "o-", color="#0f766e", label="whole sequence")
    axes[1].plot(steps, [v["token_acc"] for v in hist["val"]], "s--", color="#1f6feb", label="per digit")
    if hist["best_step"]:
        axes[1].axvline(hist["best_step"], color="#64748b", ls=":", label=f"best step {hist['best_step']}")
    axes[1].set(xlabel="Step", ylabel="Validation accuracy (%)", ylim=(-3, 103), title="Validation")
    axes[1].legend(fontsize=9)
    axes[2].plot(hist["lr"], color="#b45309", lw=2)
    axes[2].set(xlabel="Step", ylabel="Learning rate", title="Warmup + cosine schedule")
    for ax in axes:
        ax.grid(alpha=0.3)
    return save_figure(fig, out / "capstone_training.png")


def plot_ablation(out, full, ablations, curves):
    keys = ["full"] + [a[0] for a in ABLATIONS]
    eng = {"full": "Full model", "no_pos": "No positions (L14)", "no_residual": "No residual (L11)",
           "no_layernorm": "No LayerNorm (L07)", "no_warmup": "No warmup (L05/L09)",
           "big_init": "Init x4 (L06)", "no_reg": "No dropout / weight decay (L08)",
           "linear_ffn": "FFN without activation (L02/L04)"}
    res = {"full": full, **ablations}
    seq = [res[k]["seq_acc"] for k in keys]
    tok = [res[k]["token_acc"] for k in keys]
    fig, axes = plt.subplots(1, 2, figsize=(16, 5.2), gridspec_kw={"width_ratios": [1.15, 1]})
    y = np.arange(len(keys))[::-1]
    colors = ["#0f766e" if k == "full" else ("#be123c" if s < 50 else "#94a3b8") for k, s in zip(keys, seq)]
    axes[0].barh(y, seq, color=colors, height=0.6, label="whole sequence")
    axes[0].scatter(tok, y, color="#1f2937", marker="|", s=260, zorder=3, label="per digit")
    axes[0].set_yticks(y, [eng[k] for k in keys])
    axes[0].set(xlim=(0, 105), xlabel="Test accuracy (%)", title="Remove one part at a time")
    axes[0].legend(fontsize=9, loc="center right")
    for yy, s in zip(y, seq):
        axes[0].text(min(s + 1.5, 92), yy, f"{s:.1f}%", va="center", fontsize=9)
    for k in keys:
        loss = np.asarray(curves[k], dtype=float)
        k_s = 15
        sm = np.convolve(loss, np.ones(k_s) / k_s, mode="valid") if len(loss) > k_s else loss
        axes[1].plot(sm, lw=2.2 if k == "full" else 1.3, color="#0f766e" if k == "full" else None,
                     label=eng[k])
    axes[1].set(yscale="log", xlabel="Step", ylabel="Train loss (smoothed, log)", title="Training loss")
    axes[1].legend(fontsize=8, ncol=2)
    for ax in axes:
        ax.grid(alpha=0.3)
    return save_figure(fig, out / "capstone_ablation.png")


def plot_baselines(out, baselines):
    keys = ["mlp", "lstm", "transformer"]
    labels = ["MLP (L02/L09)", "LSTM read-then-write (L12/L13)", "Transformer (L15)"]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.6))
    x = np.arange(len(keys))
    seq = [baselines[k]["seq_acc"] for k in keys]
    tok = [baselines[k]["token_acc"] for k in keys]
    params = [baselines[k]["params"] / 1000 for k in keys]
    axes[0].bar(x - 0.18, seq, 0.36, color="#0f766e", label="whole sequence")
    axes[0].bar(x + 0.18, tok, 0.36, color="#93c5fd", label="per digit")
    for xx, s, t in zip(x, seq, tok):
        axes[0].text(xx - 0.18, s + 1, f"{s:.1f}", ha="center", fontsize=9)
        axes[0].text(xx + 0.18, t + 1, f"{t:.1f}", ha="center", fontsize=9)
    axes[0].set_xticks(x, labels, fontsize=9)
    axes[0].set(ylim=(0, 112), ylabel="Test accuracy (%)", title=f"Same task, {N_TRAIN} training sequences")
    axes[0].legend(fontsize=9, loc="lower center", ncol=2)
    axes[1].bar(x, params, 0.5, color="#b45309")
    for xx, p in zip(x, params):
        axes[1].text(xx, p + 1.5, f"{p:.1f}k", ha="center", fontsize=9)
    axes[1].set_xticks(x, labels, fontsize=9)
    axes[1].set(ylim=(0, max(params) * 1.15), ylabel="Parameters (thousands)", title="Model size")
    for ax in axes:
        ax.grid(axis="y", alpha=0.3)
    return save_figure(fig, out / "capstone_baselines.png")


def plot_attention(out, model, x, y):
    maps = model.attention_maps(x[None])
    fig, axes = plt.subplots(1, len(maps), figsize=(4.2 * len(maps), 4.2))
    labels = [str(int(d)) for d in x]
    for i, (ax, a) in enumerate(zip(axes, maps)):
        ax.imshow(a[0].mean(axis=0), cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(SEQ_LEN), labels)
        ax.set_yticks(range(SEQ_LEN), [f"{j}:{int(t)}" for j, t in enumerate(y)])
        ax.set(xlabel="key: input digit", title=f"Block {i + 1} (mean over heads)")
        if i == 0:
            ax.set_ylabel("query: output position : target digit")
    return save_figure(fig, out / "capstone_attention.png")


def run() -> dict:
    out = ensure_artifacts(LESSON)
    print("=" * 94)
    print("综合串联（15）：用前 14 课的零件从零组装一个小 Transformer，再一次去掉一个零件")
    print("=" * 94)
    data = datasets()
    print(f"数据: 每条样本是 {SEQ_LEN} 个 0–{VOCAB - 1} 的数字（可重复），目标是同样的数字从小到大排好。"
          f"训练 {N_TRAIN} 条、验证 {N_VAL} 条、测试 {N_TEST} 条。")
    print(f"完整模型: {MODEL['n_layers']} 个 pre-LN 块，d_model={MODEL['d_model']}，{MODEL['n_heads']} 个头，"
          f"FFN 宽 {MODEL['d_ff']}，GELU，Dropout {MODEL['dropout']}；AdamW lr={FIT['lr']}，"
          f"warmup {FIT['warmup']} 步 + 余弦，权重衰减 {FIT['weight_decay']}，梯度裁剪 {FIT['clip']}，"
          f"最多 {FIT['steps']} 步，按验证损失早停")

    print("\n梯度检查（float64 小模型，覆盖嵌入、位置、LayerNorm、Q/K/V/O、FFN、输出层）：")
    xs, ys = make_sort_dataset(3, 5, 6, seed=1)
    ck = {}
    # The plain variant only needs one block to cover the no-residual / no-LayerNorm code paths;
    # stacked without them, later Q/K gradients get ~1000x smaller and central differences lose precision.
    for tag, kw in (("full", {"n_layers": 2}), ("plain", {"n_layers": 1, "residual": False, "layer_norm": False})):
        small = TinyTransformer(6, 5, d_model=8, n_heads=2, d_ff=16, seed=1, **kw)
        for key, err in gradient_check(small, xs, ys).items():
            ck[f"{tag}/{key}"] = err
    worst = sorted(ck.items(), key=lambda kv: -kv[1])
    for key, err in worst[:6]:
        print(f"  {key:<30s} max relative error = {err:.3e}   {'✓ 通过' if err < 1e-4 else '✗ 未通过'}")
    print(f"  共检查 {len(ck)} 组参数，最大相对误差 {worst[0][1]:.3e}")
    grad_ok = all(e < 1e-4 for e in ck.values())

    print("\n--- 完整模型 ---")
    model, hist, full = run_transformer(data, "完整模型")

    print("\n--- 对照：MLP（第 02 / 09 课）---")
    mlp = MLPBaseline(VOCAB, SEQ_LEN, hidden=(256, 256), seed=SEED, dtype=np.float32)
    t0 = time.time()
    train(mlp, *data["train"], *data["val"], **{**FIT, "lr": 1e-2, "warmup": 50}, seed=SEED,
          log_fn=lambda s, l, lr_t, ev: print(f"  step {s:4d} | loss={l:.3f} | val {fmt(ev)}") if s % 200 == 0 else None)
    mlp_test = {**mlp.evaluate(*data["test"]), "params": mlp.n_params, "train_sec": round(time.time() - t0, 1)}
    mlp_train = mlp.evaluate(*data["train"])
    print(f"  [MLP] test {fmt(mlp_test)} | train {fmt(mlp_train)} | params={mlp.n_params} | {mlp_test['train_sec']}s")

    print("\n--- 对照：LSTM 先读后写（第 12 / 13 课）---")
    lstm = run_lstm(data)

    ablations, curves = {}, {"full": hist["loss"]}
    for i, (key, label, lesson, change) in enumerate(ABLATIONS):
        print(f"\n--- 消融 {chr(65 + i)}：{label}（第 {lesson} 课）---")
        _, h, res = run_transformer(data, label, change.get("model"), change.get("fit"), quiet=True)
        ablations[key] = {**res, "label": label, "lesson": lesson}
        curves[key] = h["loss"]

    X_te, Y_te = data["test"]
    pred = model.predict(X_te[:1])[0]
    example = {"input": X_te[0].tolist(), "target": Y_te[0].tolist(), "pred": pred.tolist()}
    print(f"\n  例子 输入={example['input']} 目标={example['target']} 预测={example['pred']}")

    baselines = {"mlp": mlp_test, "lstm": lstm, "transformer": {k: full[k] for k in ("token_acc", "seq_acc", "params")}}
    images = [
        plot_map(out),
        plot_training(out, hist),
        plot_baselines(out, baselines),
        plot_ablation(out, full, ablations, curves),
        plot_attention(out, model, X_te[0], Y_te[0]),
    ]

    passed = bool(
        grad_ok
        and full["seq_acc"] >= 95.0
        and all(ablations[k]["seq_acc"] <= 30.0 for k in CRITICAL)
        and mlp_test["seq_acc"] < full["seq_acc"]
    )
    metrics = {
        "task": {"seq_len": SEQ_LEN, "vocab": VOCAB, "n_train": N_TRAIN, "n_val": N_VAL, "n_test": N_TEST},
        "model": MODEL,
        "fit": FIT,
        "gradient_check": {k: {"max_rel_err": v, "passed": v < 1e-4} for k, v in ck.items()},
        "full": full,
        "baselines": baselines,
        "mlp_train": mlp_train,
        "ablations": ablations,
        "critical": list(CRITICAL),
        "example": example,
        "images": [str(p.relative_to(out.parent.parent)) for p in images],
        "passed": passed,
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
