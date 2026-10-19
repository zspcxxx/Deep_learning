from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    std = m.get("feature_raw_std")
    if std:
        ratio = max(std) / min(std)
        out.append(finding(
            f"原始特征尺度相差约 {ratio:.0f} 倍",
            f"两个特征的标准差分别为 {std[0]:.3f} 和 {std[1]:.3f}。同一个学习率对大尺度特征太大、对小尺度特征又太小："
            "损失曲面被拉成一条又窄又长的峡谷，梯度下降只能在峡谷两壁之间来回震荡，很难沿着谷底前进。",
            "warn",
        ))
    for s in m.get("setups", []):
        head = f"{s['setup']}（{s['label']}）：准确率 {pct(s['final_acc'])}，loss {s['final_loss']:.4f}"
        if s["setup"] == "raw":
            body = "直接用原始尺度训练，loss 在 0.65 左右震荡、准确率在 50%–60% 徘徊，基本没有学会月牙的形状。"
            level = "warn" if s["final_acc"] < 80 else "info"
        elif s["setup"] == "input":
            body = ("训练前用训练集的均值和标准差把每个特征变成均值 0、方差 1。只改了这一步，网络和学习率都没变，准确率就接近 100%。"
                    "注意：均值和标准差必须只用训练集计算，再套用到验证 / 测试集。")
            level = "ok"
        elif s["setup"] == "bn":
            body = ("输入不做标准化，而是在每个隐藏层的线性变换后加 BatchNorm：按当前 batch 把每个神经元的输出标准化，"
                    "再用可学习的 γ、β 缩放平移。它同样把尺度问题解决了，而且对隐藏层内部也有效。")
            level = "ok"
        else:
            body, level = "", "info"
        out.append(finding(head, body, level))
    return out


NOTES = LessonNotes(
    intro="把月牙数据的两个特征故意拉成悬殊的尺度（一个 ×10、一个 ×0.15），对比不做归一化、输入标准化、BatchNorm 三种处理。",
    steps=[
        "加载月牙数据并拉伸尺度（data/07_normalization/moons.npz）",
        "无归一化",
        "输入标准化",
        "BatchNorm",
        "画特征分布、训练曲线和决策边界",
    ],
    log_notes=[
        LogNote(r"原始特征 mean=", "看特征尺度",
                "第一个特征标准差约 8.7，第二个只有约 0.077，相差上百倍。这种情况在真实数据里很常见，例如“收入（元）”和“年龄（岁）”放在一起。",
                step=0),
        LogNote(r"--- raw:", "无归一化", "预期：同一个学习率无法同时适应两个尺度，训练困难。", step=1),
        LogNote(r"--- input:", "输入标准化", "(x − mean) / std：把每个特征都变成均值 0、方差 1。", step=2),
        LogNote(r"--- bn:", "BatchNorm",
                "在每个隐藏层后标准化，再用 γ、β 恢复表达能力。推理时用训练过程中累积的滑动平均统计量。", step=3),
        LogNote(r"^  raw +acc=", "汇总", "三组结果汇总，接着画图。", step=4),
    ],
    image_notes={
        "norm_features.png":
            "左：原始特征。matplotlib 自动缩放了坐标轴，所以月牙形状看起来正常，但看刻度：横轴 x0 从 0 到 33，纵轴 x1 只有 −1.1 到 −0.85，"
            "两者尺度相差上百倍。右：标准化后两个轴都在 −2 到 2 之间。对网络来说，左图才是它“看到”的真实尺度。",
        "norm_curves.png":
            "三种设置的训练损失和准确率。raw（红）停在 0.65 左右并不时出现尖峰，准确率在 55% 上下；"
            "input（蓝）最平稳，损失一路降到接近 0；bn（绿）前 350 轮左右波动较大，之后同样降到接近 0、准确率接近 100%。",
        "norm_boundaries.png":
            "三个模型的决策边界（黑色虚线）。raw 只学到一条接近竖直的直线，基本是按 x1 一刀切；"
            "input（在标准化坐标下画）和 bn（在原始坐标下画）都沿着两个月牙之间弯曲穿过。",
    },
    explain=explain,
)
