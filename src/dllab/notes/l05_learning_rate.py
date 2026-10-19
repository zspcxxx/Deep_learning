from __future__ import annotations

import math

from dllab.notes.base import LessonNotes, LogNote, finding, pct

LN2 = math.log(2)


def explain(m: dict) -> list[dict]:
    out = []
    rows = {r["label"]: r for r in m.get("learning_rates", [])}
    small, good, large = rows.get("too small"), rows.get("good"), rows.get("too large")
    if small:
        out.append(finding(
            f"lr = {small['lr']}（太小）：600 轮后 loss {small['final_loss']:.3f}，准确率 {pct(small['final_acc'])}",
            f"初始损失约为 ln2 ≈ {LN2:.3f}，600 轮只降到 {small['final_loss']:.3f}。方向是对的，但每步走得太短，"
            "要达到同样效果需要成百上千倍的迭代次数。训练曲线表现为一条缓慢下降的斜线。",
            "warn",
        ))
    if good:
        out.append(finding(
            f"lr = {good['lr']}（合适）：loss {good['final_loss']:.4f}，准确率 {pct(good['final_acc'])}",
            "几百轮内损失就降到接近 0，准确率 100%。合适的学习率让每一步既足够大、又不会越过谷底。",
            "ok",
        ))
    if large:
        stuck = large["final_loss"] is not None and abs(large["final_loss"] - LN2) < 0.01
        body = f"最终 loss {large['final_loss']}，准确率 {pct(large['final_acc'])}。"
        if large["diverged"]:
            body += "损失爆炸或出现 NaN：步子太大，每次更新都越过谷底并且越跳越远。"
        elif stuck:
            body += ("损失停在 ln2 ≈ 0.693、准确率 50%：没有出现 NaN，但网络已经“坏掉”，对所有输入都输出 0.5。"
                     "典型原因是前几步巨大的更新把 ReLU 神经元推到了输入恒为负的区域（死亡 ReLU），之后梯度几乎为 0，再也学不动。"
                     "这是学习率过大的另一种常见结局，比 NaN 更隐蔽。")
        out.append(finding(f"lr = {large['lr']}（太大）：训练失败", body, "error" if large["diverged"] else "warn"))

    toy = m.get("toy_1d", {})
    if toy:
        parts = []
        for lr_s, r in sorted(toy.items(), key=lambda kv: float(kv[0])):
            lr = float(lr_s)
            factor = abs(1 - 2 * lr)
            parts.append(f"lr={lr}：|1−2η| = {factor:.2f}，{r['steps']} 步后 w = {r['final_w']:.3f}")
        out.append(finding(
            "一维例子：L(w) = (w − 2)²",
            "梯度下降 w ← w − η·2(w − 2) 等价于 (w − 2) ← (1 − 2η)(w − 2)，每步误差乘以 |1 − 2η|。"
            "小于 1 就收敛，越接近 0 越快；大于 1 就发散。\n" + "\n".join(parts),
        ))
    return out


NOTES = LessonNotes(
    intro="同一个 XOR 数据、同一个 [2, 8, 8, 1] 网络、同一个随机种子，只改学习率（0.001 / 0.5 / 8.0），看训练的三种结局；"
          "再用一维二次函数直观展示步长和收敛的关系。",
    steps=[
        "加载 XOR 数据（data/05_learning_rate/xor.npz）",
        "lr = 0.001：太小",
        "lr = 0.5：合适",
        "lr = 8.0：太大",
        "画曲线、决策边界和一维示意图",
    ],
    log_notes=[
        LogNote(r"学习率对比实验", "受控对比",
                "除了学习率，其他条件完全相同（数据、网络结构、初始化种子、迭代 600 轮），所以结果差异只能归因于学习率。", step=0),
        LogNote(r"--- lr = (?P<lr>[\d.]+) \(too small\)", "lr = {lr}：太小",
                "预期：损失会下降，但非常慢，600 轮远远不够。", step=1),
        LogNote(r"--- lr = (?P<lr>[\d.]+) \(good\)", "lr = {lr}：合适", "预期：几百轮内损失降到接近 0。", step=2),
        LogNote(r"--- lr = (?P<lr>[\d.]+) \(too large\)", "lr = {lr}：太大",
                "预期：要么损失爆炸（NaN），要么网络被打坏后停滞。", step=3, level="warn"),
        LogNote(r"loss = 0\.69\d+ \| accuracy =  50\.00%", "损失卡在 ln2、准确率 50%",
                "loss ≈ 0.693 = ln2 恰好是对所有样本都输出 0.5 时的交叉熵：网络退化成常数输出。", level="warn"),
        LogNote(r"→ 最终 loss=(?P<loss>nan|\d+(?:\.\d{1,4})?)\S*  acc=(?P<acc>[\d.]+)%  diverged=(?P<d>\w+)",
                "本组结束：loss={loss}，acc={acc}%", "diverged=True 表示损失 > 5 或出现 NaN / Inf。", once=False),
        LogNote(r"^  lr=", "汇总", "三组结果汇总，接着画图。", step=4),
    ],
    image_notes={
        "lr_curves.png":
            "左：三种学习率的损失曲线。蓝色（0.001）几乎是水平线；绿色（0.5）迅速降到 0；红色（8.0）停在 0.693 附近不动。"
            "右：对应的训练准确率。",
        "lr_boundaries.png":
            "三个模型最终的决策边界。lr = 0.001 的边界还很粗糙，只分对一部分；lr = 0.5 把四个象限干净分开；"
            "lr = 8.0 整个平面一片白色（概率恒为 0.5），也没有黑色边界线——模型对所有输入都输出同一个值。",
        "lr_1d_landscape.png":
            "一维函数 L(w) = (w − 2)² 上的梯度下降轨迹（蓝点），绿色虚线是最优点 w = 2。"
            "lr = 0.05 一步步慢慢挪向谷底；lr = 0.4 几步就到；lr = 1.05 在谷底两侧来回跳，而且越跳越高（发散）。",
    },
    explain=explain,
)
