from __future__ import annotations

import numpy as np

from dllab.activations import gelu_grad, relu_grad, sigmoid_grad, tanh_grad
from dllab.notes.base import LessonNotes, LogNote, finding

N_LAYERS = 15


def explain(m: dict) -> list[dict]:
    out = []
    s = m.get("softmax_sum")
    if s is not None:
        out.append(finding(
            f"Softmax 输出之和 = {s:.6f}",
            "Softmax 把任意实数向量变成概率分布：每项为正、总和为 1。代码先减去最大值再取指数，"
            "数值上更稳定（避免 exp 溢出），但不改变结果。这是本实验的通过条件。",
            "ok" if abs(s - 1) < 1e-6 else "error",
        ))

    z = np.array([1.0])
    chains = {name: float(abs(g(z)[0])) ** N_LAYERS
              for name, g in (("Sigmoid", sigmoid_grad), ("Tanh", tanh_grad), ("ReLU", relu_grad), ("GELU", gelu_grad))}
    out.append(finding(
        f"梯度消失：{N_LAYERS} 层连乘后 Sigmoid 只剩 {chains['Sigmoid']:.1e}",
        f"反向传播每穿过一层激活函数，梯度就乘一次它的导数。在 z = 1 处连乘 {N_LAYERS} 层："
        f"Sigmoid {chains['Sigmoid']:.1e}，Tanh {chains['Tanh']:.1e}，ReLU {chains['ReLU']:.0f}，GELU {chains['GELU']:.2f}。"
        "Sigmoid 导数最大才 0.25，十几层之后靠近输入的层几乎收不到梯度；ReLU 正半轴导数恒为 1，梯度原样传回。"
        "这就是隐藏层默认用 ReLU 家族而不是 Sigmoid 的原因。",
        "info",
    ))

    rows = {r["name"]: r for r in m.get("table", [])}
    if "ReLU" in rows and "GELU" in rows:
        relu, gelu = rows["ReLU"], rows["GELU"]
        out.append(finding(
            "数值表里的几个细节",
            f"ReLU 在 z = −2 处输出 {relu['f'][0]:.2f}、导数 {relu['grad'][0]:.2f}：负半轴完全“关闭”，"
            "如果一个神经元的输入总是负的，它就永远得不到梯度（死亡 ReLU）。"
            f"GELU 在 z = −2 处输出 {gelu['f'][0]:.4f}，是一个很小的负数：它是 ReLU 的平滑版本，负半轴仍有微弱梯度，"
            "Transformer 里常用。LeakyReLU / ELU / SiLU 也都是为了给负半轴留一点梯度。",
        ))
    return out


NOTES = LessonNotes(
    intro="把常见激活函数及其导数画出来，演示 Softmax 的温度效应，并量化 Sigmoid / Tanh / ReLU / GELU 在深层网络里的梯度衰减。"
          "本实验不训练模型，只做计算和画图。",
    steps=[
        "画 16 种激活函数与导数",
        "Softmax 温度演示",
        "梯度消失：15 层导数连乘",
        "打印激活函数数值表",
    ],
    log_notes=[
        LogNote(r"激活函数数值表", "激活函数数值表",
                "图在打印之前已经画好。下面每行是一个激活函数在 z = −2、0、2 三点的取值 f(z)。"
                "对比要点：Sigmoid / Tanh 两端被压扁（饱和），ReLU 负半轴恒为 0，GELU / SiLU / Mish 负半轴有一个小“凹陷”。",
                step=3),
        LogNote(r"^Sigmoid +0\.1192", "Sigmoid", "f(−2)=0.12、f(0)=0.5、f(2)=0.88：输出在 (0,1)，适合表示概率，但两端饱和导致梯度很小。"),
        LogNote(r"^ReLU +0\.0000", "ReLU", "负半轴输出 0，正半轴原样输出：计算最便宜，梯度要么 0 要么 1。"),
        LogNote(r"^Step +", "Step", "阶跃函数是感知机用的激活，导数处处为 0（在 0 点不可导），所以无法用梯度下降训练。"),
        LogNote(r"激活函数图已保存", "画图完成", "三张图已写入 artifacts/04_activations/，见下方运行结果。"),
    ],
    image_notes={
        "activation_functions_all.png":
            "16 个子图，每个是一种激活函数：蓝色实线是 f(z)，红色虚线是导数 f'(z)。"
            "看导数曲线最重要：Sigmoid / Tanh 的导数只在 0 附近较大，两侧迅速趋于 0；ReLU 的导数是 0/1 的阶梯；"
            "GELU、SiLU、Mish 的导数是平滑的，并在负半轴保留少量梯度。",
        "softmax_demo.png":
            "左：同一组 logits 在不同温度 T 下的 Softmax 输出。T 越小分布越尖（接近 argmax），T 越大越平（接近均匀）。"
            "右：T = 1 时的概率柱状图，总和为 1。",
        "gradient_vanishing.png":
            "纵轴是对数刻度，表示梯度穿过 n 层激活函数后剩下的倍数（在 z = 1 处计算）。"
            "Sigmoid 和 Tanh 是直线下降——每层乘一个小于 1 的常数，指数衰减；ReLU 保持 1；GELU 在 z = 1 处导数略大于 1，会缓慢变大。",
    },
    explain=explain,
)
