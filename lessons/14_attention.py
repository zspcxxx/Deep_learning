#!/usr/bin/env python3
"""Lesson 14: attention as Query / Key / Value soft lookup vs mean pooling."""
from __future__ import annotations

import time

import numpy as np

from dllab.attention import LookupNet, gradient_check, make_lookup_batch
from dllab.data_store import cached_dataset
from dllab.io_utils import dump_metrics
from dllab.paths import ensure_artifacts
from dllab.plotting import plt, save_figure

LESSON = "14_attention"
SEED = 42
N_KEYS = 16
N_VALUES = 8
D_MODEL = 32
SLOT_SIZES = [4, 8, 16]
STEPS = {4: 400, 8: 600, 16: 900}
BATCH = 64
LR = 4e-3
TEST_N = 500


def smooth(loss, k=25):
    if len(loss) < k:
        return np.asarray(loss, dtype=float)
    ker = np.ones(k) / k
    return np.convolve(loss, ker, mode="valid")


def test_set(n_slots: int):
    def build(n, n_slots, n_keys, n_values, seed):
        return make_lookup_batch(n, n_slots, n_keys, n_values, np.random.RandomState(seed))

    return cached_dataset(LESSON, f"lookup_test_s{n_slots}", build, TEST_N, n_slots, N_KEYS, N_VALUES, seed=999)


def run() -> dict:
    out = ensure_artifacts(LESSON)
    print("=" * 94)
    print("注意力机制（14）：按键查值。对比平均池化与缩放点积注意力（Query / Key / Value）")
    print("=" * 94)
    print(f"数据: 每条样本是 {N_KEYS} 种键、{N_VALUES} 种值里抽出的若干互不相同的 (键, 值) 对；"
          "给定其中一个键，预测对应的值。训练数据每一步现场生成。")

    print("\n梯度检查（平均池化 / 缩放点积注意力）：")
    rng = np.random.RandomState(1)
    keys, values, query, target = make_lookup_batch(3, 4, N_KEYS, N_VALUES, rng)
    ck = {}
    for mode in ("mean", "attention"):
        model = LookupNet(N_KEYS, N_VALUES, d_model=8, mode=mode, seed=1)
        for key, err in gradient_check(model, keys, values, query, target, n_checks=4).items():
            ck[f"{mode}/{key}"] = err
            print(f"  {mode:<10s}{key:<12s} max relative error = {err:.3e}   "
                  f"{'✓ 通过' if err < 1e-4 else '✗ 未通过'}")
    grad_ok = all(e < 1e-4 for e in ck.values())

    tests = {n: test_set(n) for n in SLOT_SIZES}
    results = {"mean": {}, "attention": {}}
    curves = {}
    attention_map = None
    example = None

    def train(mode: str, n_slots: int):
        steps = STEPS[n_slots]
        print(f"\n--- 按键查值：{n_slots} 个槽位，{'缩放点积注意力' if mode == 'attention' else '平均池化'}，{steps} 步 ---")
        model = LookupNet(N_KEYS, N_VALUES, d_model=D_MODEL, mode=mode, seed=SEED)
        te = tests[n_slots]

        def batch(step, n_slots=n_slots):
            return make_lookup_batch(BATCH, n_slots, N_KEYS, N_VALUES, np.random.RandomState(step))

        def log(step, loss, norm, model=model, te=te):
            acc = model.accuracy(*te)
            print(f"  step {step:4d} | loss={loss:.3f} | acc={acc:.1f}%")

        t0 = time.time()
        model.fit(batch, steps, lr=LR, clip=1.0, log_every=max(steps // 4, 1), log_fn=log)
        acc = model.accuracy(*te)
        results[mode][n_slots] = {
            "acc": acc,
            "final_loss": float(np.mean(model.loss_[-30:])),
            "params": model.n_params,
            "train_sec": round(time.time() - t0, 1),
            "steps": steps,
        }
        curves[(mode, n_slots)] = model.loss_
        print(f"  [{mode}] acc={acc:.1f}% params={model.n_params} time={time.time() - t0:.1f}s")
        if mode == "attention" and n_slots == SLOT_SIZES[-1]:
            return model, te
        return None

    for n_slots in SLOT_SIZES:
        train("mean", n_slots)
        got = train("attention", n_slots)
        if got is not None:
            model, te = got
            keys_te, values_te, query_te, target_te = te
            weights = model.attention_weights(keys_te[:64], values_te[:64], query_te[:64])
            # Align weights so column 0 is the true matching slot (for a readable heatmap).
            true_slot = np.argmax(keys_te[:64] == query_te[:64, None], axis=1)
            aligned = np.zeros_like(weights)
            for i, s in enumerate(true_slot):
                order = np.roll(np.arange(weights.shape[1]), -int(s))
                aligned[i] = weights[i, order]
            attention_map = aligned.mean(axis=0)
            i = 0
            pred = model.predict(keys_te[i : i + 1], values_te[i : i + 1], query_te[i : i + 1])[0]
            example = {
                "keys": keys_te[i].tolist(),
                "values": values_te[i].tolist(),
                "query": int(query_te[i]),
                "target": int(target_te[i]),
                "pred": int(pred),
                "weights": weights[i].tolist(),
                "true_slot": int(true_slot[i]),
            }
            print(f"  例子 keys={example['keys']} values={example['values']} "
                  f"query={example['query']} target={example['target']} pred={example['pred']} "
                  f"weights={[round(w, 3) for w in example['weights']]}")

    long = SLOT_SIZES[-1]
    fig, axes = plt.subplots(1, 2, figsize=(14, 4.8))
    for mode, color, label in (
        ("mean", "#be123c", "mean pool"),
        ("attention", "#0f766e", "scaled dot-product"),
    ):
        axes[0].plot(SLOT_SIZES, [results[mode][n]["acc"] for n in SLOT_SIZES],
                     "o-", lw=2, color=color, label=label)
        axes[1].plot(smooth(curves[(mode, long)]), lw=2, color=color, label=label)
    axes[0].axhline(100.0 / N_VALUES, color="#64748b", ls="--", lw=1, label="chance")
    axes[0].set(xlabel="Number of slots", ylabel="Accuracy (%)", ylim=(-3, 105),
                title="Lookup by key", xticks=SLOT_SIZES)
    axes[1].set(xlabel="Training step", ylabel="Loss (smoothed)", title=f"Training at {long} slots")
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend(fontsize=9)
    img_acc = save_figure(fig, out / "attention_accuracy.png")

    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    x = np.arange(long)
    ax.bar(x, attention_map, color="#0f766e", width=0.8)
    ax.set(xlabel="Slot (0 = matching key, then wrap-around)", ylabel="Mean attention weight",
           title=f"Where attention looks ({long} slots, aligned to true key)",
           xticks=x, ylim=(0, max(0.35, float(attention_map.max()) * 1.15)))
    ax.grid(axis="y", alpha=0.3)
    img_att = save_figure(fig, out / "attention_weights.png")

    passed = bool(
        grad_ok
        and results["attention"][SLOT_SIZES[0]]["acc"] >= 90.0
        and results["attention"][long]["acc"] >= 95.0
        and results["mean"][long]["acc"] <= 40.0
    )
    metrics = {
        "n_keys": N_KEYS, "n_values": N_VALUES, "d_model": D_MODEL, "slot_sizes": SLOT_SIZES,
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
