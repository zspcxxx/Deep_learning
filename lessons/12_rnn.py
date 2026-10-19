#!/usr/bin/env python3
"""Lesson 12: recurrent networks — BPTT, long-term memory (RNN vs LSTM vs GRU) and character-level name generation."""
from __future__ import annotations

import time
from collections import Counter

import numpy as np

from dllab.data_store import cached_dataset
from dllab.datasets import NAME_VOCAB, encode_names, generate_memory_dataset, load_names
from dllab.io_utils import dump_metrics
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure
from dllab.rnn import RecurrentNet, gradient_check, input_grad_norms

LESSON = "12_rnn"
SEED = 42
N_SYMBOLS = 4
HIDDEN = 32
DELAYS = [5, 10, 20, 40]
MEMORY_STEPS = 1500
FORGET_BIAS = 3.0
UPDATE_BIAS = 3.0
NAME_HIDDEN = 64
NAME_STEPS = 1500
NAME_TRAIN = 28000
TEMPERATURES = (0.5, 1.0, 1.5)
MODELS = {
    "rnn": {"label": "vanilla RNN", "cell": "rnn", "forget_bias": 1.0, "update_bias": 0.0, "color": "#be123c"},
    "lstm": {"label": f"LSTM (forget bias {FORGET_BIAS:g})", "cell": "lstm", "forget_bias": FORGET_BIAS, "update_bias": 0.0, "color": "#0f766e"},
    "lstm_fb1": {"label": "LSTM (forget bias 1)", "cell": "lstm", "forget_bias": 1.0, "update_bias": 0.0, "color": "#ca8a04"},
    "gru": {"label": f"GRU (update bias {UPDATE_BIAS:g})", "cell": "gru", "forget_bias": 1.0, "update_bias": UPDATE_BIAS, "color": "#2563eb"},
    "gru_ub0": {"label": "GRU (update bias 0)", "cell": "gru", "forget_bias": 1.0, "update_bias": 0.0, "color": "#7c3aed"},
}


def make(key: str, n_in: int, hidden: int, n_out: int, seed: int = SEED) -> RecurrentNet:
    v = MODELS[key]
    return RecurrentNet(
        v["cell"], n_in, hidden, n_out, seed=seed,
        forget_bias=v["forget_bias"], update_bias=v["update_bias"],
    )


def decode(seq) -> str:
    return "".join(NAME_VOCAB[t] for t in seq)


def smooth(xs, k: int = 25):
    xs = np.asarray(xs, dtype=float)
    return np.convolve(xs, np.ones(k) / k, mode="valid") if len(xs) > k else xs


def run() -> dict:
    out = ensure_artifacts(LESSON)
    print("=" * 94)
    print("循环神经网络（12）：随时间反向传播、长期记忆（RNN vs LSTM vs GRU）、按字符生成名字")
    print("=" * 94)
    names = load_names()
    print(f"数据: 记忆任务（{N_SYMBOLS} 种符号，序列结束时说出第 1 个符号）+ 名字 data/names/names.txt（{len(names)} 个）")

    # ---- 1. gradient check of BPTT ----
    print("\n梯度检查（随时间反向传播 BPTT）：")
    ck_report = {}
    Xc, yc = generate_memory_dataset(4, 6, N_SYMBOLS, seed=1)
    Xn, Yn, Mn = encode_names(["emma", "bo", "anna"])
    for cell in ("rnn", "lstm", "gru"):
        for mode, args in (("last", (Xc, yc, None, N_SYMBOLS, N_SYMBOLS)), ("all", (Xn, Yn, Mn, len(NAME_VOCAB), len(NAME_VOCAB)))):
            X, Y, M, n_in, n_out = args
            m = RecurrentNet(cell, n_in, 5, n_out, seed=1)
            for key, err in gradient_check(m, X, Y, mode=mode, mask=M).items():
                ck_report[f"{cell}/{mode}/{key}"] = err
                print(f"  {cell:<5s}{mode:<5s}{key:<8s} max relative error = {err:.3e}   {'✓ 通过' if err < 1e-6 else '✗ 未通过'}")
    grad_ok = all(e < 1e-6 for e in ck_report.values())

    # ---- 2. long-term memory task ----
    test_sets = {d: cached_dataset(LESSON, f"memory_test_d{d}", generate_memory_dataset, 1000, d, N_SYMBOLS, seed=999)
                 for d in DELAYS}
    memory = {k: {} for k in MODELS}
    curves = {}
    trained = {}

    def train_memory(key, delay):
        Xte, yte = test_sets[delay]
        m = make(key, N_SYMBOLS, HIDDEN, N_SYMBOLS)
        t0 = time.time()
        m.fit(lambda s: generate_memory_dataset(64, delay, N_SYMBOLS, seed=10_000 * delay + s), MEMORY_STEPS, lr=3e-3, clip=1.0)
        acc = float(np.mean(m.predict(Xte) == yte) * 100)
        memory[key][delay] = {"accuracy": acc, "final_loss": float(np.mean(m.loss_[-50:])), "train_sec": round(time.time() - t0, 1)}
        curves[(key, delay)] = m.loss_
        trained[(key, delay)] = m
        print(f"  [{key}] acc={acc:.1f}% final_loss={memory[key][delay]['final_loss']:.3f} "
              f"params={m.n_params} time={time.time() - t0:.1f}s")

    for delay in DELAYS:
        print(f"\n--- 记忆任务：延迟 {delay} 步（随机猜对率 {100 / N_SYMBOLS:.0f}%） ---")
        for key in ("rnn", "lstm", "gru"):
            train_memory(key, delay)
    long = DELAYS[-1]
    print(f"\n--- 门偏置对比：延迟 {long} 步，LSTM 遗忘门偏置 1、GRU 更新门偏置 0 ---")
    train_memory("lstm_fb1", long)
    train_memory("gru_ub0", long)

    print(f"\n梯度沿时间回传（初始化时，延迟 {long} 步，第 1 步输入收到的梯度 / 最后一步输入收到的梯度）：")
    Xte, yte = test_sets[long]
    grad_flow = {}
    for key in MODELS:
        g = input_grad_norms(make(key, N_SYMBOLS, HIDDEN, N_SYMBOLS), Xte[:256], yte[:256])
        grad_flow[key] = (g / g[-1]).tolist()
        print(f"  [{key}] ‖dL/dx_1‖ / ‖dL/dx_{long}‖ = {g[0] / g[-1]:.1e}")
    g = input_grad_norms(trained[("lstm", long)], Xte[:256], yte[:256])
    grad_flow["lstm_trained"] = (g / g[-1]).tolist()
    f_gate = trained[("lstm", long)].forget_gates(Xte[:256])[:, 1:].mean(axis=(0, 1))
    z_gate = trained[("gru", long)].update_gates(Xte[:256])[:, 1:].mean(axis=(0, 1))
    print(f"  训练后 LSTM 的遗忘门均值：最大 {f_gate.max():.3f}，有 {int(np.sum(f_gate > 0.95))} / {HIDDEN} 个单元 > 0.95")
    print(f"  训练后 GRU 的更新门均值：最大 {z_gate.max():.3f}，有 {int(np.sum(z_gate > 0.95))} / {HIDDEN} 个单元 > 0.95")

    # ---- 3. character-level names ----
    perm = np.random.RandomState(0).permutation(len(names))
    train_names = [names[i] for i in perm[:NAME_TRAIN]]
    val_names = [names[i] for i in perm[NAME_TRAIN:]]
    Xtr, Ytr, Mtr = encode_names(train_names)
    Xva, Yva, Mva = encode_names(val_names)
    counts = Counter("".join(train_names) + "." * len(train_names))
    freq = np.array([counts[c] for c in NAME_VOCAB], dtype=float)
    freq /= freq.sum()
    unigram = float(-(freq * np.log(freq)).sum())
    uniform = float(np.log(len(NAME_VOCAB)))
    print(f"\n名字数据：训练 {len(train_names)} / 验证 {len(val_names)}，字符表 {len(NAME_VOCAB)} 个（'.' 表示开始 / 结束）")
    print(f"  基线：均匀猜测 {uniform:.3f} nats/字符；只按字母频率猜（unigram） {unigram:.3f} nats/字符")

    train_set = set(train_names)
    name_results = {}
    for key in ("rnn", "lstm", "gru"):
        print(f"\n--- 生成名字：[{key}] 隐藏层 {NAME_HIDDEN}，{NAME_STEPS} 步 ---")
        m = make(key, len(NAME_VOCAB), NAME_HIDDEN, len(NAME_VOCAB))
        val_curve = []
        t0 = time.time()

        def batch(s):
            idx = np.random.RandomState(s).randint(len(train_names), size=64)
            return Xtr[idx], Ytr[idx], Mtr[idx]

        def log(step, loss, norm, m=m, val_curve=val_curve):
            v = m.sequence_loss(Xva, Yva, Mva)
            val_curve.append((step, v))
            smp = ", ".join(decode(s) for s in m.sample(0, 0, 6, 1.0, seed=step))
            print(f"  step {step:5d} | val_loss={v:.3f} | 样本: {smp}")

        m.fit(batch, NAME_STEPS, lr=5e-3, clip=5.0, mode="all", log_every=NAME_STEPS // 5, log_fn=log)
        samples = {}
        for T in TEMPERATURES:
            gen = [decode(s) for s in m.sample(0, 0, 200, T, seed=7)]
            gen = [g for g in gen if g]
            novel = float(np.mean([g not in train_set for g in gen]) * 100)
            samples[str(T)] = {"novel_pct": novel, "examples": gen[:12]}
            print(f"  温度 T={T}: 新名字占比={novel:.0f}% 样本: {', '.join(gen[:8])}")
        name_results[key] = {"val_loss": val_curve[-1][1], "val_curve": val_curve, "train_loss": m.loss_,
                             "params": m.n_params, "train_sec": round(time.time() - t0, 1), "samples": samples}

    # ---- Figure 1: memory task ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for key in ("rnn", "lstm", "gru"):
        axes[0].plot(DELAYS, [memory[key][d]["accuracy"] for d in DELAYS], "o-", lw=2,
                     color=MODELS[key]["color"], label=MODELS[key]["label"])
    axes[0].plot([long], [memory["lstm_fb1"][long]["accuracy"]], "s", ms=10, color=MODELS["lstm_fb1"]["color"],
                 label=MODELS["lstm_fb1"]["label"])
    axes[0].plot([long], [memory["gru_ub0"][long]["accuracy"]], "D", ms=8, color=MODELS["gru_ub0"]["color"],
                 label=MODELS["gru_ub0"]["label"])
    axes[0].axhline(100 / N_SYMBOLS, color="#64748b", ls=":", lw=1)
    axes[0].set(xlabel="Delay (steps between the symbol and the question)", ylabel="Test accuracy (%)",
                title="Remember the first symbol", ylim=(0, 105), xticks=DELAYS)
    for key in MODELS:
        axes[1].plot(smooth(curves[(key, long)]), lw=2, color=MODELS[key]["color"], label=MODELS[key]["label"])
    axes[1].axhline(np.log(N_SYMBOLS), color="#64748b", ls=":", lw=1)
    axes[1].set(xlabel="Training step", ylabel="Loss (smoothed)", title=f"Training at delay {long}")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    img_memory = save_figure(fig, out / "rnn_memory.png")

    # ---- Figure 2: gradient flow + forget gates ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    steps = np.arange(1, long + 1)
    for key in MODELS:
        axes[0].semilogy(steps, grad_flow[key], lw=2, color=MODELS[key]["color"], label=f"{MODELS[key]['label']} (init)")
    axes[0].semilogy(steps, grad_flow["lstm_trained"], lw=2, ls="--", color=MODELS["lstm"]["color"],
                     label=f"{MODELS['lstm']['label']} (trained)")
    axes[0].set(xlabel="Input step t", ylabel="‖dL/dx_t‖ / ‖dL/dx_T‖ (log)",
                title=f"How much gradient reaches each step (delay {long})")
    axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(fontsize=8)
    order_f = np.argsort(f_gate)[::-1]
    order_z = np.argsort(z_gate)[::-1]
    axes[1].plot(range(HIDDEN), f_gate[order_f], lw=2, color=MODELS["lstm"]["color"], label="LSTM forget f")
    axes[1].plot(range(HIDDEN), z_gate[order_z], lw=2, color=MODELS["gru"]["color"], label="GRU update z")
    axes[1].axhline(1.0, color="#64748b", ls=":", lw=1)
    axes[1].set(xlabel="Unit (sorted)", ylabel="mean gate", ylim=(0, 1.05),
                title=f"Trained gates at delay {long}")
    axes[1].legend(fontsize=9)
    axes[1].grid(axis="y", alpha=0.3)
    img_grad = save_figure(fig, out / "rnn_gradflow.png")

    # ---- Figure 3: names ----
    fig, axes = plt.subplots(1, 2, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1, 1.25]})
    for key in ("rnn", "lstm", "gru"):
        st, vl = zip(*name_results[key]["val_curve"])
        axes[0].plot(smooth(name_results[key]["train_loss"], 50), lw=1.2, alpha=0.5, color=MODELS[key]["color"])
        axes[0].plot(st, vl, "o-", lw=2, color=MODELS[key]["color"], label=f"{MODELS[key]['label']} val")
    axes[0].axhline(uniform, color="#64748b", ls=":", lw=1)
    axes[0].text(NAME_STEPS * 0.55, uniform + 0.03, "uniform guess", fontsize=8, color="#64748b")
    axes[0].axhline(unigram, color="#64748b", ls="--", lw=1)
    axes[0].text(NAME_STEPS * 0.55, unigram + 0.03, "letter frequency only", fontsize=8, color="#64748b")
    axes[0].set(xlabel="Training step", ylabel="nats per character", title="Next-character loss (thin = train, dots = validation)")
    axes[0].grid(alpha=0.3)
    axes[0].legend(fontsize=9)
    axes[1].axis("off")
    lines = ["Names invented by the GRU (bold = not in the training set)", ""]
    axes[1].text(0.0, 1.0, lines[0], fontsize=11, fontweight="bold", va="top", transform=axes[1].transAxes)
    y = 0.9
    for T in TEMPERATURES:
        s = name_results["gru"]["samples"][str(T)]
        axes[1].text(0.0, y, f"T = {T}   ({s['novel_pct']:.0f}% new)", fontsize=10, va="top", color="#2563eb",
                     transform=axes[1].transAxes)
        y -= 0.07
        for row in range(3):
            x = 0.02
            for name in s["examples"][row * 4 : row * 4 + 4]:
                axes[1].text(x, y, name, fontsize=10, va="top", family="monospace", transform=axes[1].transAxes,
                             fontweight="bold" if name not in train_set else "normal")
                x += 0.25
            y -= 0.065
        y -= 0.03
    img_names = save_figure(fig, out / "rnn_names.png")

    lstm_names = name_results["lstm"]
    gru_names = name_results["gru"]
    passed = bool(
        grad_ok
        and memory["rnn"][DELAYS[0]]["accuracy"] >= 95.0
        and memory["lstm"][long]["accuracy"] >= 95.0
        and memory["gru"][long]["accuracy"] >= 95.0
        and memory["rnn"][long]["accuracy"] <= 40.0
        and memory["gru_ub0"][long]["accuracy"] <= 40.0
        and lstm_names["val_loss"] < unigram - 0.4
        and gru_names["val_loss"] < unigram - 0.4
        and lstm_names["samples"]["1.0"]["novel_pct"] >= 50.0
    )
    metrics = {
        "memory": {k: {str(d): v for d, v in r.items()} for k, r in memory.items()},
        "delays": DELAYS, "n_symbols": N_SYMBOLS, "hidden": HIDDEN,
        "forget_bias": FORGET_BIAS, "update_bias": UPDATE_BIAS,
        "gradient_check": {k: {"max_rel_err": v, "passed": v < 1e-6} for k, v in ck_report.items()},
        "grad_flow_first_over_last": {k: v[0] for k, v in grad_flow.items()},
        "forget_gate": {"max": float(f_gate.max()), "min": float(f_gate.min()),
                        "units_above_0.95": int(np.sum(f_gate > 0.95)), "hidden": HIDDEN},
        "update_gate": {"max": float(z_gate.max()), "min": float(z_gate.min()),
                        "units_above_0.95": int(np.sum(z_gate > 0.95)), "hidden": HIDDEN},
        "names": {
            "n_train": len(train_names), "n_val": len(val_names), "uniform_loss": uniform, "unigram_loss": unigram,
            **{k: {kk: vv for kk, vv in r.items() if kk not in ("train_loss",)} for k, r in name_results.items()},
        },
        "images": [str(p.relative_to(out.parent.parent)) for p in (img_memory, img_grad, img_names)],
        "passed": passed,
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
