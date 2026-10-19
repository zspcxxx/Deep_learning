#!/usr/bin/env python3
"""Lesson 13: encoder–decoder seq2seq. Reverse a random digit string, with and without attention."""
from __future__ import annotations

import time

import numpy as np

from dllab.data_store import cached_dataset
from dllab.io_utils import dump_metrics
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure
from dllab.seq2seq import EncoderDecoder, gradient_check, make_reverse_batch

LESSON = "13_seq2seq"
SEED = 42
N_SYM = 5
SOS = N_SYM
EOS = N_SYM + 1
VOCAB = N_SYM + 2
HIDDEN = 32
UPDATE_BIAS = 3.0
LENGTHS = [4, 8, 12]
STEPS = {4: 500, 8: 1600, 12: 1600}
BATCH = 64
LR = 4e-3
TEST_N = 400


def smooth(loss, k=25):
    if len(loss) < k:
        return np.asarray(loss, dtype=float)
    ker = np.ones(k) / k
    return np.convolve(loss, ker, mode="valid")


def test_set(length: int):
    def build(n, length, n_sym, sos, eos, seed):
        src, _, target = make_reverse_batch(n, length, n_sym, sos, eos, np.random.RandomState(seed))
        return src, target

    return cached_dataset(LESSON, f"reverse_test_l{length}", build, TEST_N, length, N_SYM, SOS, EOS, seed=999)


def run() -> dict:
    out = ensure_artifacts(LESSON)
    print("=" * 94)
    print("编码器–解码器（13）：把一串随机数字倒过来。对比“一个向量”和注意力")
    print("=" * 94)
    print(f"数据: 源序列是 {N_SYM} 种符号，目标是把它倒过来并在末尾输出 EOS。训练数据每一步现场生成。")

    print("\n梯度检查（编码器、解码器、注意力）：")
    rng = np.random.RandomState(1)
    src, dec_in, target = make_reverse_batch(3, 4, N_SYM, SOS, EOS, rng)
    ck = {}
    for attention, name in ((False, "seq2seq"), (True, "attention")):
        model = EncoderDecoder(VOCAB, 5, attention=attention, update_bias=UPDATE_BIAS, seed=1)
        for key, err in gradient_check(model, src, dec_in, target, n_checks=4).items():
            ck[f"{name}/{key}"] = err
            print(f"  {name:<10s}{key:<8s} max relative error = {err:.3e}   {'✓ 通过' if err < 1e-4 else '✗ 未通过'}")
    grad_ok = all(e < 1e-4 for e in ck.values())

    tests = {length: test_set(length) for length in LENGTHS}
    results = {"seq2seq": {}, "attention": {}}
    curves = {}
    attention_map = None
    example = None

    def train(attention: bool, length: int):
        key = "attention" if attention else "seq2seq"
        steps = STEPS[length]
        print(f"\n--- 倒序任务：长度 {length}，{'注意力' if attention else '固定向量'}，{steps} 步 ---")
        model = EncoderDecoder(VOCAB, HIDDEN, attention=attention, update_bias=UPDATE_BIAS, seed=SEED)
        src_te, tgt_te = tests[length]

        def batch(step, length=length):
            return make_reverse_batch(BATCH, length, N_SYM, SOS, EOS, np.random.RandomState(step))

        def log(step, loss, norm, model=model, src_te=src_te, tgt_te=tgt_te):
            acc = model.exact_match(src_te, tgt_te, SOS, EOS)
            print(f"  step {step:4d} | loss={loss:.3f} | exact={acc:.1f}%")

        t0 = time.time()
        model.fit(batch, steps, lr=LR, clip=1.0, log_every=max(steps // 4, 1), log_fn=log)
        acc = model.exact_match(src_te, tgt_te, SOS, EOS)
        pred, alphas = model.decode(src_te[:64], SOS, EOS)
        tok = float(np.mean(pred == tgt_te[:64]) * 100)
        results[key][length] = {
            "exact": acc, "token": tok, "final_loss": float(np.mean(model.loss_[-30:])),
            "params": model.n_params, "train_sec": round(time.time() - t0, 1), "steps": steps,
        }
        curves[(key, length)] = model.loss_
        print(f"  [{key}] exact={acc:.1f}% token={tok:.1f}% params={model.n_params} time={time.time() - t0:.1f}s")
        if attention and length == LENGTHS[-1]:
            return pred, alphas, src_te, tgt_te
        return None

    for length in LENGTHS:
        train(False, length)
        got = train(True, length)
        if got is not None:
            pred, alphas, src_te, tgt_te = got
            attention_map = alphas.mean(axis=0)
            example = {
                "src": src_te[0].tolist(),
                "target": tgt_te[0].tolist(),
                "pred": pred[0].tolist(),
            }
            print(f"  例子 src={example['src']} target={example['target']} pred={example['pred']}")

    long = LENGTHS[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for key, color, label in (
        ("seq2seq", "#be123c", "fixed vector"),
        ("attention", "#0f766e", "attention"),
    ):
        axes[0].plot(LENGTHS, [results[key][n]["exact"] for n in LENGTHS], "o-", lw=2, color=color, label=label)
        axes[1].plot(smooth(curves[(key, long)]), lw=2, color=color, label=label)
    axes[0].set(xlabel="Source length", ylabel="Exact-match accuracy (%)", ylim=(-3, 105),
                title="Reverse the sequence", xticks=LENGTHS)
    axes[1].set(xlabel="Training step", ylabel="Loss (smoothed)", title=f"Training at length {long}")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    img_acc = save_figure(fig, out / "seq2seq_accuracy.png")

    fig, ax = plt.subplots(figsize=(6.2, 5.2))
    im = ax.imshow(attention_map, cmap="Blues", vmin=0, vmax=max(0.2, float(attention_map.max())), aspect="auto")
    ax.set(xlabel="Source position", ylabel="Decoder step (EOS last)",
           title=f"Where attention looks (length {long})")
    fig.colorbar(im, ax=ax, fraction=0.046)
    img_att = save_figure(fig, out / "seq2seq_attention.png")

    passed = bool(
        grad_ok
        and results["seq2seq"][LENGTHS[0]]["exact"] >= 90.0
        and results["attention"][LENGTHS[0]]["exact"] >= 90.0
        and results["attention"][long]["exact"] >= 95.0
        and results["seq2seq"][long]["exact"] <= 20.0
    )
    metrics = {
        "lengths": LENGTHS, "n_sym": N_SYM, "hidden": HIDDEN, "update_bias": UPDATE_BIAS,
        "results": {k: {str(n): v for n, v in r.items()} for k, r in results.items()},
        "gradient_check": {k: {"max_rel_err": v, "passed": v < 1e-4} for k, v in ck.items()},
        "example": example,
        "images": [str(p.relative_to(out.parent.parent)) for p in (img_acc, img_att)],
        "passed": passed,
    }
    dump_metrics(out / "metrics.json", metrics)
    print(f"\n结果已保存到 {out}")
    return metrics


if __name__ == "__main__":
    run()
