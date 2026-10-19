from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    gates = m.get("gates", {})
    for name, rule in (("AND", "只有 (1,1) 为 +1"), ("OR", "只有 (0,0) 为 −1")):
        r = gates.get(name)
        if not r:
            continue
        if r["converged"]:
            out.append(finding(
                f"{name}：第 {r['epochs']} 轮收敛，训练准确率 {pct(r['accuracy'])}",
                f"{name} 门{rule}，一条直线就能把两类分开（线性可分）。感知机收敛定理保证：只要数据线性可分，"
                "感知机在有限次更新后一定找到一条完全分开两类的直线。这里两类点之间间隔（margin）较宽，"
                "所以只需要很少几轮就没有分错的点了，之后权重不再更新。",
                "ok",
            ))
        else:
            out.append(finding(f"{name}：未收敛", "线性可分的数据理论上必收敛，可能是轮数上限太小或学习率设置有问题。", "warn"))

    x = gates.get("XOR")
    if x:
        body = (f"跑满 {x['epochs']} 轮仍有 {x['final_errors']} 个样本分错，准确率 {pct(x['accuracy'])}。"
                "XOR 的同类点位于对角线两端，任何一条直线都至少把一组角点分错，所以这不是学习率或轮数不够，"
                "而是单层线性模型的能力上限。")
        if x["accuracy"] < 50:
            body += ("准确率甚至低于 50%：感知机不收敛时会一直被错分样本来回拉扯，最终权重只取决于最后几次更新，"
                     "停在哪里带有偶然性，并不对应任何“最优”直线。")
        body += "解决办法见第 02 课：加一个隐藏层。"
        out.append(finding(f"XOR：{'意外收敛' if x['converged'] else '无法收敛（符合预期）'}", body,
                           "warn" if x["converged"] else "ok"))

    t = m.get("two_d")
    if t:
        out.append(finding(
            f"二维两簇数据：第 {t['epochs']} 轮收敛，准确率 {pct(t['accuracy'])}",
            "两团点分别位于 (10,10) 和 (−10,−10) 附近，相距很远，同样线性可分；"
            "样本数是逻辑门的三倍，但收敛轮数并不取决于样本多少，而取决于两类之间的间隔大小。",
            "ok" if t["converged"] else "warn",
        ))
    return out


NOTES = LessonNotes(
    intro="用最简单的线性分类器——感知机，分别学习 AND、OR、XOR 三个逻辑门和一份二维两簇数据，"
          "观察“线性可分就能学会、线性不可分就学不会”。",
    steps=[
        "加载逻辑门数据（data/01_perceptron/*.npz）",
        "训练 AND、OR：线性可分，应当很快收敛",
        "训练 XOR：线性不可分，预期跑满轮数仍有误分类",
        "训练二维两簇数据",
        "画出决策线与误分类曲线",
    ],
    log_notes=[
        LogNote(r"逻辑门演示", "数据准备",
                "四个角点 (0,0)(0,2)(2,0)(2,2) 周围各撒 50 个点（截断高斯扰动，半宽 0.49 保证点不会越过象限），"
                "标签取 ±1。数据缓存在 data/01_perceptron/ 下，参数不变就直接复用。", step=0),
        LogNote(r"\[AND\] 开始训练", "开始训练 AND",
                "感知机预测 sign(w·x + b)。只有分错的点会触发更新：w ← w + η·y·x，b ← b + η·y，"
                "即把决策线朝能把这个点分对的方向推一点。", step=1),
        LogNote(r"Epoch +\d+ \| errors", "怎么读这一行",
                "errors 是本轮仍被分错的样本数，accuracy 是训练集准确率，w 和 b 是当前决策线 w1·x1 + w2·x2 + b = 0 的参数。"
                "errors 降到 0 说明找到了一条完全分开两类的直线。"),
        LogNote(r"在第 (?P<n>\d+) 轮收敛$", "第 {n} 轮收敛",
                "这一轮没有任何样本被分错，权重不再变化，训练提前结束。", once=False, level="ok"),
        LogNote(r"\[OR\] 开始训练", "开始训练 OR", "OR 只有左下角为 −1，同样一条直线可分。", step=1),
        LogNote(r"\[XOR\] 开始训练", "开始训练 XOR",
                "XOR 中 (0,0)(2,2) 为 −1、(0,2)(2,0) 为 +1，同类点在对角。不存在一条直线能分开它们，"
                "预期 errors 会一直震荡，不会降到 0。轮数上限设为 400 以节省时间。", step=2),
        LogNote(r"达到最大轮数", "跑满轮数仍未收敛", "这正是线性不可分的表现：每次修正一个点都会把另一些点推到错误一侧。",
                level="warn"),
        LogNote(r"二维数据（共", "二维两簇数据",
                "两团高斯点分别以 (10,10) 和 (−10,−10) 为中心，间隔很大，感知机应当几轮内收敛。", step=3),
        LogNote(r"结果图已保存", "画图完成", "第一行是三个逻辑门的数据和学到的决策线，第二行是每轮误分类数。", step=4),
    ],
    image_notes={
        "perceptron_result.png":
            "第一行：AND / OR / XOR 的样本（红点 +1、蓝叉 −1）和最终决策线（黑虚线）。AND、OR 的线干净地分开两类；"
            "XOR 的线无论放在哪里都会切错一部分点。第二行：每轮误分类数——AND、OR 很快降到 0，XOR 一直在高位震荡。"
            "第三行：二维两簇数据的决策线、误分类曲线，以及四条误分类曲线的汇总对比。",
    },
    explain=explain,
)
