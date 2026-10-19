from __future__ import annotations

import math

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    gc = m.get("gradient_check", {})
    if gc:
        worst_name, worst = max(((k, v["max_rel_err"]) for k, v in gc.items()), key=lambda kv: kv[1])
        ok = all(v["passed"] for v in gc.values())
        out.append(finding(
            f"梯度检查{'全部通过' if ok else '未通过'}：最大相对误差 {worst:.1e}（{worst_name}）",
            "残差块的反向传播是 dL/dx = dL/dy + dL/dy · ∂F/∂x：上游梯度一份直接走捷径到 x，一份穿过 F 回来，两份相加。"
            "数值梯度和这个公式对上了。个别参数的误差在 1e-6 量级，是因为中心差分的扰动偶尔跨过 ReLU 在 0 处的折点，"
            "所以阈值取 1e-5（第 03 课用的是处处光滑的 sigmoid，可以更严格）。",
            "ok" if ok else "error",
        ))

    res = m.get("results", {})
    plain, resid, res_he = res.get("plain"), res.get("residual"), res.get("residual_he")
    if not (plain and resid):
        return out
    n_classes = 3
    chance = 100 / n_classes
    best = max(plain, key=lambda r: r["train_acc"])
    deep_p, deep_r = plain[-1], resid[-1]
    out.append(finding(
        f"退化问题：普通网络 {best['depth']} 层时训练准确率 {pct(best['train_acc'])}，{deep_p['depth']} 层只剩 {pct(deep_p['train_acc'])}",
        "更深的网络表达能力只会更强——理论上它可以让多出来的层什么都不做（恒等映射），至少不比浅网络差。"
        "但实际训练时，普通网络超过一定深度后连训练集都拟合不了：" +
        "，".join(f"{r['depth']} 层 {r['train_acc']:.1f}%" for r in plain) + "。"
        "注意这是训练集准确率，所以不是过拟合，而是优化器找不到好的参数——这就是 ResNet 论文里说的“退化问题”。" +
        (f"{deep_p['depth']} 层时损失停在 {deep_p['train_loss']:.4f} ≈ ln{n_classes}，准确率 {chance:.1f}%，"
         "等于对每个样本都均匀猜测三个类，网络完全没有学到东西。"
         if deep_p["train_loss"] is not None and abs(deep_p["train_loss"] - math.log(n_classes)) < 0.01 else ""),
        "warn",
    ))
    out.append(finding(
        f"残差网络：{deep_r['depth']} 层训练准确率 {pct(deep_r['train_acc'])}，测试 {pct(deep_r['test_acc'])}",
        "同样的层、同样的参数量（" + "，".join(f"{r['depth']} 层 {r['params']:,}" for r in resid[-2:]) + "），"
        "只是每个块从 h ← F(h) 改成 h ← h + F(h)。结果：" +
        "，".join(f"{r['depth']} 层 {r['train_acc']:.1f}%" for r in resid) + "。"
        "网络越深，最终训练损失反而越低（见训练曲线），深度终于变成了优势。",
        "ok" if deep_r["train_acc"] >= 95 else "warn",
    ))

    sig = m.get("signal_at_init", {})
    if "plain" in sig and "residual" in sig:
        sp, sr = sig["plain"], sig["residual"]
        depth = sp["depth"]
        out.append(finding(
            f"为什么：初始化时信号在 {depth} 层普通网络里逐层衰减",
            f"训练开始前，把数据送进普通网络：隐藏状态的标准差从 {sp['forward_std'][0]:.3g} 衰减到最后一块的 {sp['forward_std'][-1]:.3g}；"
            f"反向传播时，到达最后一块的梯度 ‖dL/dh‖ 是 {sp['grad_norm'][-1]:.2e}，传到第 1 块只剩 {sp['grad_norm'][0]:.2e}"
            f"（缩小约 {sp['grad_norm'][-1] / max(sp['grad_norm'][0], 1e-12):.0f} 倍）。前面的层几乎收不到梯度，学不动。\n"
            f"残差网络（分支末层零初始化）：前向标准差每一块都是 {sr['forward_std'][-1]:.3g}，梯度每一块都是 {sr['grad_norm'][0]:.2e}——"
            "因为 ∂(h + F(h))/∂h = I + ∂F/∂h，不管 F 怎样，梯度至少有一份能原封不动地沿捷径传回去。",
        ))
    if res_he and "residual_he" in sig:
        sh = sig["residual_he"]
        out.append(finding(
            f"光有捷径还不够：分支用 He 初始化时，{res_he[-1]['depth']} 层只有 {pct(res_he[-1]['train_acc'])}",
            "如果残差分支 F 一开始就输出和输入同样大小的信号，每经过一块方差就大约翻倍（h + F(h) 是两份相加）："
            f"前向标准差从 {sh['forward_std'][0]:.3g} 涨到 {sh['forward_std'][-1]:.3g}，第 1 块收到的梯度是 {sh['grad_norm'][0]:.3g}。"
            "信号爆炸让训练很不稳定。把每个分支的最后一层初始化为 0，块一开始就是恒等映射 h + 0 = h，"
            "网络从“一个很浅的网络”起步，再逐渐学会用上每一块。实际的 ResNet 用 BatchNorm（最后一个 BN 的 γ 初始化为 0），"
            "Transformer 用 LayerNorm 并缩小残差分支，目的都一样：让残差块在初始化时接近恒等。",
            "warn",
        ))
    shallow_p, shallow_r = plain[0], resid[0]
    if shallow_r["train_acc"] < shallow_p["train_acc"]:
        out.append(finding(
            f"浅网络上残差没有优势：{shallow_p['depth']} 层时普通 {pct(shallow_p['train_acc'])}，残差 {pct(shallow_r['train_acc'])}",
            "残差连接解决的是“深”带来的优化困难。只有两个块时普通网络本来就好训练；而零初始化的残差块一开始什么都不做，"
            "需要先花一些轮次把分支“长”出来，在只训练 100 轮的预算下反而略慢。",
            "info",
        ))
    return out


NOTES = LessonNotes(
    intro="在三条螺旋臂分类任务上，把同样的全连接网络从 6 层加深到 50 层，对比普通网络和加了残差连接（h + F(h)）的网络："
          "普通网络越深越训练不动，残差网络越深越好。再从初始化时信号的前向 / 反向传播解释原因，并说明残差分支为什么要零初始化。",
    steps=[
        "加载螺旋数据（data/11_residual/spiral_*.npz）",
        "梯度检查残差块的反向传播",
        "初始化时的信号传播（50 层）",
        "深度扫描：6 / 10 / 18 / 34 / 50 层 × 三种网络",
        "汇总",
        "画准确率-深度曲线、训练曲线、信号传播、决策边界",
    ],
    log_notes=[
        LogNote(r"数据: data/11_residual", "数据：三条螺旋臂",
                "900 个训练点、900 个测试点，三类各占三分之一，每条螺旋绕 2.5 圈，彼此紧紧缠在一起，需要很弯曲的决策边界。"
                "输入只有 2 维，训练很快，适合拿来做“只改深度”的对照实验。", step=0),
        LogNote(r"^网络: 宽度", "网络结构",
                "先用一个 Dense 把 2 维输入映射到 32 维（stem），然后是若干个块，每块 F(h) = W2·ReLU(W1·ReLU(h) + b1) + b2，"
                "最后 ReLU → Dense 输出 3 类。普通网络：h ← F(h)；残差网络：h ← h + F(h)。两者参数完全相同。"
                "统一用 AdamW（lr 2e-3）、batch 64、训练 100 轮，不加权重衰减和早停，只看能不能把训练集拟合好。"),
        LogNote(r"梯度检查（2 个残差块", "梯度检查",
                "用一个很小的残差网络（宽度 6、2 个块）对每组参数抽 10 个元素做中心差分，验证 Residual 层的反向传播。", step=1),
        LogNote(r"(?P<p>\S+) +max relative error = (?P<e>[\d.e+-]+) +✓", "{p} 通过（{e}）",
                "每行是一组参数：stem / head 是首尾两层，blockK.fc_a / fc_b 是第 K 个块里的两个 Dense。误差低于 1e-5 即通过。", level="ok"),
        LogNote(r"(?P<p>\S+) +max relative error = (?P<e>[\d.e+-]+) +✗", "{p} 未通过（{e}）",
                "解析梯度和数值梯度对不上。", once=False, level="error"),
        LogNote(r"信号传播（初始化后", "初始化时的信号传播",
                "还没开始训练，先看信号能不能在 50 层网络里正常传递：前向看每块输出 h 的标准差，反向看传到每块输入处的梯度大小 ‖dL/dh‖。"
                "两者都应该在各层之间保持在差不多的量级。", step=2),
        LogNote(r"\[plain\] 前向 h 的标准差: 块0 (?P<a>\S+) → 块\d+ (?P<b>\S+) \| 反向 ‖dL/dh‖: 块\d+ (?P<c>\S+) → 块1 (?P<d>\S+)",
                "普通网络：前向 {a} → {b}，梯度 {c} → {d}",
                "前向信号越传越弱，反向梯度传到靠近输入的第 1 块时又小了一个多数量级。前面的层几乎收不到更新信号。", level="warn"),
        LogNote(r"\[residual\] 前向 h 的标准差: 块0 (?P<a>\S+) → 块\d+ (?P<b>\S+) \| 反向 ‖dL/dh‖: 块\d+ (?P<c>\S+) → 块1 (?P<d>\S+)",
                "残差网络（零初始化）：前向 {a} → {b}，梯度 {c} → {d}",
                "每个分支的最后一层初始化为 0，所以每块一开始都是 h + 0 = h：信号和梯度都沿捷径原样通过，每一块完全相同。", level="ok"),
        LogNote(r"\[residual_he\] 前向 h 的标准差: 块0 (?P<a>\S+) → 块\d+ (?P<b>\S+) \| 反向 ‖dL/dh‖: 块\d+ (?P<c>\S+) → 块1 (?P<d>\S+)",
                "残差网络（He 初始化）：前向 {a} → {b}，梯度 {c} → {d}",
                "分支一开始就输出“满尺寸”的信号，每块 h + F(h) 都让方差大约翻倍，24 块下来放大了上千倍——信号爆炸。", level="warn"),
        LogNote(r"--- 深度 6 层", "开始深度扫描",
                "每个深度训练三个网络：plain（普通）、residual（残差 + 分支零初始化）、residual_he（残差 + 分支 He 初始化）。"
                "train_acc 是训练集准确率，衡量“能不能学会”；test_acc 是另一批螺旋点上的准确率。", step=3),
        LogNote(r"--- 深度 50 层", "最深的 50 层", "看普通网络在这里会发生什么。"),
        LogNote(r"\[plain\] train_loss=1\.098\d", "普通网络完全没学会",
                "训练损失 1.0986 = ln 3，准确率 33.33%：网络对每个样本都给三个类相同的概率，等于随机猜。"
                "100 轮里损失一动不动——梯度传不到前面的层，参数几乎没有被有效更新。", level="error"),
        LogNote(r"汇总（训练集准确率", "汇总", "按深度列出三种网络的训练准确率，接着画图。", step=4),
        LogNote(r"结果已保存到", "画图完成", "", step=5),
    ],
    image_notes={
        "res_depth.png":
            "左：训练集准确率随深度的变化，虚线是随机猜测的 33.3%。普通网络（红）在 10–18 层最好，之后一路下滑，50 层掉到随机水平；"
            "残差网络（绿）从 10 层起一直是 100%。He 初始化的残差网络（黄）介于两者之间，越深越差。"
            "右：最终训练损失（对数坐标），34 层和 50 层时残差网络的损失比普通网络低两到四个数量级。",
        "res_curves.png":
            "三种网络在各个深度下的训练损失曲线（颜色越亮越深）。普通网络：18 层最好，34 层下降很慢，50 层是一条贴在 ln3 ≈ 1.1 的水平线。"
            "残差网络（零初始化）：前几十轮下降较慢（分支要先从 0“长”出来），之后深网络下降得最快、最低；6 层的最浅网络反而最差。"
            "He 初始化的残差网络：开头损失有尖峰，深网络整体停在较高的位置。",
        "res_signal.png":
            "训练前的 50 层网络。左：每块输出 h 的标准差（对数坐标）——普通网络逐层衰减到原来的约 1/30，零初始化残差网络是一条水平线，"
            "He 初始化残差网络指数式上涨到 1000 以上。右：反向传播到每块输入的梯度大小——普通网络越靠近输入越小，"
            "零初始化残差网络每块完全一样，He 初始化残差网络越靠近输入越大（爆炸）。",
        "res_boundaries.png":
            "决策边界（背景颜色是模型预测的类别，点是训练数据）。18 层普通网络能沿着螺旋画出边界；50 层普通网络整张图只预测一个类；"
            "50 层残差网络（零初始化）的边界严丝合缝地贴着三条螺旋；50 层 He 初始化残差网络的边界出现大量放射状的碎片，外圈分错很多。",
    },
    explain=explain,
)
