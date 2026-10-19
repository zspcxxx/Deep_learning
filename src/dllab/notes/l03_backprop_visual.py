from __future__ import annotations

import math

from dllab.notes.base import LessonNotes, LogNote, finding


def explain(m: dict) -> list[dict]:
    a2, loss = m.get("a2"), m.get("loss")
    if a2 is None:
        return []
    dz2 = a2 - 1.0
    return [
        finding(
            f"前向：预测概率 ŷ = {a2:.4f}，损失 L = {loss:.4f}",
            f"真实标签 y = 1，模型给出 ŷ = {a2:.4f}。二元交叉熵 L = −ln ŷ = −ln {a2:.4f} = {-math.log(a2):.4f}。"
            "ŷ 越接近 1，损失越接近 0；如果 ŷ 接近 0（完全判反），损失会趋于无穷大。",
        ),
        finding(
            f"输出层误差 dz2 = ŷ − y = {dz2:+.4f}",
            "sigmoid + 交叉熵组合的好处：对输出层加权和 z2 的梯度化简成 ŷ − y，不需要单独乘 sigmoid 的导数。"
            f"它是负数，意思是“z2 应该变大”：梯度下降 z2 ← z2 − η·dz2 会把 z2 往上推，从而把 ŷ 推向 1。",
        ),
        finding(
            "链式法则逐层往回传",
            "dW2 = a1 · dz2：输出层权重的梯度 = 上一层激活 × 本层误差（激活越大的连接，责任越大）。\n"
            "da1 = dz2 · W2：误差按权重分配回隐藏层。\n"
            "dz1 = da1 ⊙ a1(1 − a1)：穿过 sigmoid 时要乘它的导数，而 sigmoid 导数最大只有 0.25。\n"
            "所以日志里 |dz1| 比 |dz2| 小了一个数量级——这就是深层 sigmoid 网络梯度消失的来源（第 04 课）。",
        ),
        finding(
            "这个例子用来手算核对",
            "所有数字都是固定的（x、W1、b1、W2、b2 都写死在代码里），可以拿纸笔按 docs/03_backprop.html 的步骤逐项验证。"
            "下一个实验（03）会用数值梯度自动做同样的核对。",
            "ok",
        ),
    ]


NOTES = LessonNotes(
    intro="用一个 2-2-1 的小网络和一组固定数字，把前向传播和反向传播的每一个中间量都打印出来，方便对照公式手算。",
    steps=[
        "前向传播：x → z1 → a1 → z2 → ŷ → 损失 L",
        "反向传播：dz2 → dW2 → da1 → dz1",
        "画出网络数值图与梯度大小",
    ],
    log_notes=[
        LogNote(r"^前向传播", "前向传播", "输入 x 经过 z1 = x·W1 + b1、a1 = sigmoid(z1)、z2 = a1·W2 + b2、ŷ = sigmoid(z2)，最后算交叉熵损失。", step=0),
        LogNote(r"a2=\[(?P<a2>[\d.]+)\], L=(?P<L>[\d.]+)", "前向结果：ŷ = {a2}，L = {L}",
                "z1 / a1 是隐藏层两个神经元的加权和与激活；a2 就是预测概率 ŷ。真实标签 y = 1，所以 L = −ln ŷ。"),
        LogNote(r"^反向传播", "反向传播", "从损失出发，用链式法则依次求每个中间量的梯度。", step=1),
        LogNote(r"dz2=\[(?P<dz2>[-\d.]+)\]", "输出误差 dz2 = {dz2}",
                "dz2 = ŷ − y。dW2 = a1·dz2，da1 = dz2·W2，dz1 = da1 ⊙ a1(1 − a1)。注意 dz1 比 dz2 小得多——sigmoid 导数 ≤ 0.25。"),
        LogNote(r"结果图已保存", "画图完成", "", step=2),
    ],
    image_notes={
        "backprop_visualization.png":
            "左上：前向传播，节点里是各层激活值（输入 x、隐藏层 h、输出 ŷ），绿色连线是信号从左往右的路径。"
            "右上：同一个网络，红色连线表示反向传播时梯度从 ŷ 往回流的路径。"
            "左下：损失 L、输出误差 dz2 和隐藏层误差 dz1 的具体数字。"
            "右下：四组参数梯度的范数——靠近输出的 dW2 / db2（约 0.25 / 0.30）是靠近输入的 dW1 / db1（约 0.05）的 5 倍左右，"
            "梯度每往回穿过一层 sigmoid 就被缩小一截。",
    },
    explain=explain,
)
