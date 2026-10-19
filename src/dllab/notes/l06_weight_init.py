from __future__ import annotations

import math

from dllab.notes.base import LessonNotes, LogNote, finding, pct

LN2 = math.log(2)


def _fmt(xs: list[float]) -> str:
    return " → ".join(f"{x:.3g}" for x in xs)


def explain(m: dict) -> list[dict]:
    out = []
    for s in m.get("schemes", []):
        acts = s["init_act_std"]
        head = f"{s['scheme']}（{s['label']}）：准确率 {pct(s['final_acc'])}，loss {s['final_loss']:.4f}"
        stuck = abs(s["final_loss"] - LN2) < 1e-3
        if s["scheme"] == "zeros":
            body = ("所有权重都是 0，同一层每个神经元的输出、梯度都完全相同，更新后也仍然相同（对称性无法打破），"
                    "相当于每层只有一个神经元。加上 ReLU(0)=0，隐藏层激活全是 0，网络一开始就是常数输出，loss 恰为 ln2。")
            level = "error"
        elif s["scheme"] == "small":
            body = (f"初始激活标准差逐层缩小：{_fmt(acts)}。每层乘以一个很小的权重，信号越传越弱，"
                    "反向传播时梯度同样被逐层缩小，所以 loss 几乎没离开 ln2。")
            if s["final_acc"] > 55:
                body += f"准确率 {pct(s['final_acc'])} 只是因为输出在 0.5 附近有极微小的偏移，模型并没有真正学会。"
            level = "warn"
        elif s["scheme"] == "large":
            body = (f"初始激活标准差逐层放大：{_fmt(acts)}。输出层输入极大，sigmoid 饱和，"
                    "第一步的梯度更新幅度巨大，把大部分 ReLU 推入负半轴后网络退化成常数输出。")
            if stuck:
                body += "最终 loss 停在 ln2，和全零初始化一样没学到任何东西。"
            level = "error"
        elif s["scheme"] == "he":
            body = (f"He 初始化按 std = √(2 / fan_in) 缩放权重，激活标准差逐层保持稳定：{_fmt(acts)}。"
                    "ReLU 会把一半输入置零、方差减半，He 的系数 2 正好补回来。信号既不消失也不爆炸，训练顺利收敛。")
            level = "ok"
        else:
            body, level = "", "info"
        out.append(finding(head, body, level))
    if out:
        out.append(finding(
            "结论",
            "四组实验的数据、结构、学习率、随机种子完全相同，只有初始化不同，结局却天差地别。"
            "深层网络训练能否启动，很大程度上取决于初始化是否让每层的信号尺度保持在合理范围。"
            "ReLU 网络用 He，tanh / sigmoid 网络用 Xavier。",
        ))
    return out


NOTES = LessonNotes(
    intro="用一个较深的 [2, 32, 32, 32, 1] ReLU 网络训练 XOR，四组实验只改初始化方式（全零 / 过小 / 过大 / He），"
          "观察初始化如何决定信号在网络中能否正常传递。",
    steps=[
        "加载 XOR 数据（data/06_weight_init/xor.npz）",
        "全零初始化",
        "过小高斯 σ = 0.01",
        "过大高斯 σ = 2.0",
        "He 初始化",
        "画训练曲线与激活分布",
    ],
    log_notes=[
        LogNote(r"参数初始化对比实验", "受控对比",
                "网络比前几课更深（3 个隐藏层），初始化的影响会被逐层放大，更容易看清差别。", step=0),
        LogNote(r"--- init = zeros", "全零初始化", "预期：对称性无法打破，网络学不动。", step=1),
        LogNote(r"--- init = small", "过小初始化", "预期：激活和梯度逐层衰减（信号消失）。", step=2),
        LogNote(r"--- init = large", "过大初始化", "预期：激活逐层放大（信号爆炸）。", step=3),
        LogNote(r"--- init = he", "He 初始化", "预期：各层激活尺度稳定，正常收敛。", step=4),
        LogNote(r"初始权重 std", "初始权重标准差",
                "每层权重矩阵在训练开始前的标准差。He 初始化第一层较大（fan_in = 2），后面几层约 0.25（fan_in = 32）。"),
        LogNote(r"初始隐藏层激活 std", "初始激活标准差",
                "把训练数据送进还没训练的网络，各隐藏层输出的标准差。理想情况是逐层大致不变；逐层变小是信号消失，逐层变大是信号爆炸。"),
        LogNote(r"loss = 0\.693147 \|", "loss 恒为 0.693147", "这正是 ln2：网络对每个样本都输出 0.5，梯度几乎为 0，参数不再变化。",
                level="warn"),
        LogNote(r"^  zeros +acc=", "汇总", "四组结果汇总，接着画图。", step=5),
    ],
    image_notes={
        "init_curves.png":
            "左：四种初始化的训练损失。只有 He（绿色）降了下去；large（红）开头有一个很高的尖峰，随后和 zeros、small 一起"
            "贴在 0.693 附近（三条线重叠）。右：训练准确率——He 很快到 100%；small 的准确率慢慢爬到 80% 左右，"
            "但它的损失始终是 0.693，说明输出只是在 0.5 附近有微小偏移；zeros 和 large 停在 50%。",
        "init_act_std.png":
            "训练开始前每个隐藏层激活的标准差（线性纵轴）。large（红）从第 1 层的约 0.8 涨到第 3 层的约 44（信号爆炸）；"
            "He（绿）每层都在 0.3–0.4，保持稳定；small 小到 1e-3 以下、zeros 为 0，所以在图上看不到柱子（信号消失）。",
        "init_act_hist.png":
            "第 2 个隐藏层在初始化时的激活值直方图，标题里是标准差，注意每张图的横轴范围："
            "zeros 全部为 0；small 全部挤在 0–0.0008 之间；large 分布到 35 左右；He 分布在 0–2 之间。",
    },
    explain=explain,
)
