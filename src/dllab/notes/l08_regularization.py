from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct

COMMENTS = {
    "l2": "L2 在损失里加 (λ/2m)·Σw²，每步把权重往 0 拉一点，权重整体变小、边界更平滑。",
    "l1": "L1 加 (λ/m)·Σ|w|，梯度是 sign(w)，会把不重要的权重直接推到 0，产生稀疏的权重。",
    "dropout": "训练时随机让 40% 的隐藏神经元输出为 0，网络不能依赖某几个神经元“背答案”。推理时关闭。",
    "elastic": "L1 + L2 组合：既压小权重又产生稀疏性。",
    "l2_drop": "L2 + Dropout 组合约束。",
    "early": "不加任何惩罚，只是在验证损失不再下降时停止训练，并回滚到验证集最好的那一轮。",
    "strong": "λ₂ = 2.0 远超合理范围：惩罚项压倒数据项，权重几乎全被压成 0，模型退化成常数（欠拟合）。",
}


def explain(m: dict) -> list[dict]:
    setups = {s["setup"]: s for s in m.get("setups", [])}
    base = setups.get("none")
    if not base:
        return []
    out = [finding(
        f"实验设定：训练集只有 {m.get('n_train')} 个点，其中 {m.get('n_flip')} 个标签被故意翻转",
        f"训练 {m.get('n_train')} / 验证 {m.get('n_val')} / 测试 {m.get('n_test')}，宽网络 [2, 64, 64, 1]。"
        "样本少、有错误标签、模型容量大，是最容易过拟合的组合：模型会把那些翻转的错误标签也“背”下来。",
    ), finding(
        f"none（无正则）：train {pct(base['train_acc'])} vs test {pct(base['test_acc'])}，差距 {base['gap']:.1f} 个百分点",
        f"训练准确率很高，测试却低得多——典型的过拟合。权重范数 ‖W‖ = {base['weight_norm']:.1f}，是后面各方法的对照基线。",
        "warn",
    )]
    for key in ("l2", "l1", "dropout", "elastic", "l2_drop", "early", "strong"):
        s = setups.get(key)
        if not s:
            continue
        stats = (f"train {pct(s['train_acc'])} / test {pct(s['test_acc'])}，gap {s['gap']:.1f}，"
                 f"‖W‖ {s['weight_norm']:.2f}（none 为 {base['weight_norm']:.2f}），稀疏度 {s['sparsity']:.0%}")
        body = COMMENTS[key] + "\n" + stats
        level = "info"
        if key == "early":
            body += (f"\n只训练了 {s.get('epochs_used', '?')} 轮就停下。训练准确率反而低于验证准确率（gap 为负），"
                     "说明模型还没开始背噪声就被叫停了。在这份数据上它的测试准确率最高。")
            level = "ok"
        elif key == "strong":
            level = "warn"
        elif key in ("l1", "elastic") and s["sparsity"] > 0.5:
            body += f"\n{s['sparsity']:.0%} 的权重绝对值小于 1e-3，L1 的稀疏效果非常明显。"
        out.append(finding(f"{key}（{s['label']}）", body, level))

    best = max(setups.values(), key=lambda s: s["test_acc"])
    out.append(finding(
        f"测试集最好的是 {best['setup']}（{pct(best['test_acc'])}）",
        "正则化的目标不是提高训练准确率，而是缩小训练与测试之间的差距。这份数据非常小、噪声又大，"
        "各方法测试准确率的差别只有几个百分点，单次实验的结论不一定稳定；更可靠的做法是多个随机种子取平均，并用验证集选择超参数。",
    ))
    return out


NOTES = LessonNotes(
    intro="在小样本 + 标签噪声 + 宽网络这种极易过拟合的设定下，并排对比无正则、L2、L1、Dropout、L1+L2、L2+Dropout、早停、过强 L2 八种设置。",
    steps=[
        "加载月牙数据并划分、翻转部分标签（data/08_regularization/moons.npz）",
        "无正则基线",
        "L2 / L1 / Dropout",
        "组合方法：L1+L2、L2+Dropout",
        "早停",
        "过强 L2（欠拟合对照）",
        "画曲线、柱状图、边界和早停验证曲线",
    ],
    log_notes=[
        LogNote(r"训练 (?P<n>\d+)（翻转 (?P<f>\d+)）", "数据：训练 {n} 个点，翻转 {f} 个标签",
                "故意制造过拟合条件：训练集小，且有约 30% 的标签是错的。验证集用于早停，测试集只在最后评估。", step=0),
        LogNote(r"--- none:", "无正则基线", "预期：训练准确率很高，测试明显更低。", step=1),
        LogNote(r"--- l2:", "单一正则手段", "接下来依次是 L2、L1、Dropout，每组只打开一种正则，其余条件和 none 完全相同。", step=2),
        LogNote(r"--- elastic:", "组合正则", "L1+L2（elastic net）和 L2+Dropout：两种约束叠加，强度各自调小一些。", step=3),
        LogNote(r"--- early:", "早停", "每轮在验证集上计算损失，patience = 80 轮无改善就停。", step=4),
        LogNote(r"--- strong:", "过强 L2", "预期：欠拟合，train 和 test 都接近随机猜测。", step=5, level="warn"),
        LogNote(r"→ train=(?P<tr>[\d.]+)% val=(?P<va>[\d.]+)% test=(?P<te>[\d.]+)% gap=(?P<gap>[-\d.]+) \|\|W\|\|=(?P<w>[\d.]+) sparse=(?P<sp>[\d.]+)",
                "train {tr}% / test {te}%，gap {gap}",
                "gap = train − test，越大说明过拟合越严重；‖W‖ 是全部权重的 L2 范数；sparse 是绝对值 < 1e-3 的权重占比。",
                once=False),
        LogNote(r"^  none +train=", "汇总", "八组结果汇总，接着画图。", step=6),
    ],
    image_notes={
        "reg_curves.png":
            "各方法的训练损失和训练准确率。训练集只有 28 个点，每多分对或分错一个点准确率就跳约 3.6 个百分点，所以曲线都是锯齿状。"
            "none（红）的训练损失最低——它把训练集（包括错误标签）拟合得最彻底；strong（灰）很快锁定在 53.6% 不再变化。",
        "reg_train_test.png":
            "三组柱状图：train / test 准确率、权重范数 ‖W‖、稀疏度（|w| < 0.001 的比例）。重点看：l2 / elastic / l2_drop 的 ‖W‖ 比 none 小一半以上；"
            "l1 / elastic 的稀疏度超过 90%；strong 的 ‖W‖ 几乎为 0、稀疏度 100%；early 是唯一 test 高于 train 的方法。",
        "reg_boundaries.png":
            "八个模型的决策边界（黑色虚线），标题里是测试准确率和 ‖W‖。红点 / 蓝叉是 28 个训练样本（含被翻转的错误标签），"
            "浅灰小点是部分测试样本。none 和 l1 的边界伸出细长的“触手”去包住个别训练点；early 的边界最简单；strong 整个平面一片空白，没有边界。",
        "reg_early_stop.png":
            "早停实验的验证损失曲线。第 4 轮左右验证损失降到最低，之后持续上升并越来越震荡——模型开始拟合噪声。"
            "虚线标出被选中的轮次；训练又继续了 patience 轮确认没有改善才停下，最终参数回滚到虚线处。",
    },
    explain=explain,
)
