from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    xa, xl = m.get("xor_accuracy"), m.get("xor_loss")
    if xa is not None:
        out.append(finding(
            f"XOR：训练准确率 {pct(xa)}，损失 {xl:.2e}",
            "01 课的感知机在 XOR 上最多只能分对一部分；加了两层各 8 个 ReLU 神经元后就能完全分开。"
            "隐藏层先把输入映射到新的特征空间（相当于把平面“折”几下），在那个空间里 XOR 变成线性可分，"
            "输出层再画一条直线即可。损失接近 0 说明模型对每个样本都给出了接近 0 或 1 的高置信度预测。",
            "ok" if xa >= 95 else "warn",
        ))
    ma, ml = m.get("moons_accuracy"), m.get("moons_loss")
    if ma is not None:
        out.append(finding(
            f"Two Moons：训练准确率 {pct(ma)}，损失 {ml:.2e}",
            "两个互相嵌套的月牙，边界是弯曲的。[2,16,16,1] 的网络能拟合出这条弯曲边界。",
            "ok" if ma >= 90 else "warn",
        ))
    out.append(finding(
        "注意：这里只看了训练集",
        "本课的准确率都在训练数据上计算，100% 只说明模型“记住”了这些点，不代表在新数据上也一样好。"
        "如何用验证集 / 测试集衡量泛化，以及如何防止过拟合，见第 08 课。",
        "warn",
    ))
    return out


NOTES = LessonNotes(
    intro="在感知机前面加隐藏层和 ReLU 非线性，变成多层感知机（MLP），验证它能解决感知机学不会的 XOR，以及边界弯曲的月牙数据。",
    steps=[
        "加载 XOR 数据（data/02_mlp/xor.npz）",
        "训练 [2, 8, 8, 1] MLP 拟合 XOR",
        "加载 Two Moons 数据（data/02_mlp/moons.npz）",
        "训练 [2, 16, 16, 1] MLP 拟合月牙",
        "画决策边界、损失和准确率曲线",
    ],
    log_notes=[
        LogNote(r"多层感知机演示 - XOR", "XOR 数据与网络",
                "XOR 四个象限各 100 个点（高斯噪声 0.15）。网络结构 [2, 8, 8, 1]：输入 2 维，两层各 8 个 ReLU 神经元，"
                "输出 1 个 sigmoid 概率。学习率 0.8，全批量梯度下降 1200 轮。", step=0),
        LogNote(r"Epoch +300 \| loss", "训练进行中",
                "损失在几百轮内就降到 0.01 以下，准确率 100%——隐藏层让 XOR 变得可分。", step=1),
        LogNote(r"多层感知机演示 - Two Moons", "Two Moons 数据与网络",
                "两个月牙各 200 个点（噪声 0.1），边界是弯曲的。用更宽的 [2, 16, 16, 1] 网络训练 1500 轮。", step=2),
        LogNote(r"Epoch +500 \| loss", "月牙训练进行中", "损失持续下降，弯曲的边界逐渐成形。", step=3),
        LogNote(r"结果图已保存", "画图完成", "左列是决策边界，中列是损失曲线，右列是准确率曲线。", step=4),
    ],
    image_notes={
        "mlp_result.png":
            "上排 XOR、下排 Two Moons。左：背景颜色是模型输出的概率（红≈1、蓝≈0），黑色虚线是 0.5 决策边界——"
            "XOR 上它把平面分成四块，月牙上它是一条弯曲的线。中：损失随迭代下降。右：训练准确率很快升到 100%。",
    },
    explain=explain,
)
