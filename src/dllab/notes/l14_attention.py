from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    gc = m.get("gradient_check", {})
    if gc:
        worst_name, worst = max(gc.items(), key=lambda kv: kv[1]["max_rel_err"])
        ok = all(v["passed"] for v in gc.values())
        out.append(finding(
            f"缩放点积注意力的反向传播{'通过' if ok else '未通过'}梯度检查：最大相对误差 {worst['max_rel_err']:.1e}（{worst_name}）",
            "Query、Key、Value 各自有线性投影；分数是 QKᵀ/√d，再经 softmax 得到权重，上下文是权重乘 Value。"
            "平均池化基线没有这些投影，只是把所有槽位取平均后和查询拼在一起分类。"
            "通过线设在 1e-4；实现反了的话相对误差会接近 1。",
            "ok" if ok else "error",
        ))
    res = m.get("results", {})
    sizes = [str(n) for n in m.get("slot_sizes", [])]
    mean, att = res.get("mean", {}), res.get("attention", {})
    if mean and att and sizes:
        long = sizes[-1]
        out.append(finding(
            f"按键查值：{sizes[0]} 个槽位时注意力 {pct(att[sizes[0]]['acc'])}；"
            f"{long} 个槽位时注意力 {pct(att[long]['acc'])}、平均池化 {pct(mean[long]['acc'])}",
            "任务是在一袋 (键, 值) 里，根据查询键找回对应的值。"
            "平均池化：" + "，".join(f"{n} 槽 {mean[n]['acc']:.0f}%" for n in sizes) + "。"
            "注意力：" + "，".join(f"{n} 槽 {att[n]['acc']:.0f}%" for n in sizes) + "。"
            "槽位一多，平均值把匹配项稀释掉了；注意力只给匹配键附近的槽位大权重。",
            "ok" if att[long]["acc"] >= 95 and mean[long]["acc"] <= 40 else "warn",
        ))
    ex = m.get("example") or {}
    if ex:
        w = ex.get("weights") or []
        peak = max(range(len(w)), key=lambda i: w[i]) if w else None
        out.append(finding(
            "注意力权重集中在匹配键所在的槽位",
            f"一条样本：键 {ex.get('keys')}，值 {ex.get('values')}，查询 {ex.get('query')}，"
            f"目标 {ex.get('target')}，预测 {ex.get('pred')}。"
            f"各槽位权重 {[round(x, 3) for x in w]}；真值槽位是第 {ex.get('true_slot')} 个"
            + (f"，权重峰值也在第 {peak} 个。" if peak is not None else "。"),
        ))
    return out


NOTES = LessonNotes(
    intro="第 13 课在编码器–解码器里用过注意力。本课把它拆成 Query / Key / Value："
          "用缩放点积注意力按键查值，并和平均池化对照，看记忆变长以后谁还能找对。",
    steps=[
        "梯度检查（平均池化 / 注意力）",
        "按键查值：4 个槽位",
        "按键查值：8 个槽位",
        "按键查值：16 个槽位",
        "画准确率曲线和注意力权重",
    ],
    log_notes=[
        LogNote(r"^数据: 每条样本", "按键查值任务",
                "每条样本是若干互不相同的 (键, 值) 对。模型看到全部槽位和一个查询键，要输出对应的值。"
                "和翻译不同：这里没有循环网络，只有一次软查找。"),
        LogNote(r"^梯度检查", "梯度检查",
                "对嵌入、Q/K/V 投影和输出层用中心差分核对。平均池化路径也要过，避免基线写错。", step=0),
        LogNote(r"(?P<p>\S+) +max relative error = (?P<e>[\d.e+-]+) +✗", "梯度检查未通过（{e}）",
                "反向传播和数值梯度对不上。", once=False, level="error"),
        LogNote(r"--- 按键查值：4 个槽位", "短记忆",
                "槽位少时，平均值里匹配项占比还不算小，平均池化偶尔也能猜对一些。", step=1),
        LogNote(r"--- 按键查值：8 个槽位", "中等记忆", "匹配项被更多无关值稀释。", step=2),
        LogNote(r"--- 按键查值：16 个槽位", "长记忆",
                "平均池化接近随机猜（1/值种类数）。注意力应把几乎全部权重放在匹配键上。", step=3),
        LogNote(r"\[mean\] acc=(?P<a>[\d.]+)%", "平均池化：{a}%",
                "把所有槽位的嵌入取平均，再和查询拼在一起分类。槽位越多，匹配信号越弱。", once=False),
        LogNote(r"\[attention\] acc=(?P<a>[\d.]+)%", "注意力：{a}%",
                "查询投影成 Q，每个槽位投影成 K 和 V；softmax(QKᵀ/√d) 选出该读哪个 V。", once=False, level="ok"),
        LogNote(r"^  例子 keys=", "一条样本", "对照键、值、查询、预测，以及模型打在每个槽位上的注意力权重。"),
        LogNote(r"结果已保存到", "画图完成", "", step=4),
    ],
    image_notes={
        "attention_accuracy.png":
            "左：测试准确率随槽位数变化。红线是平均池化，绿线是缩放点积注意力，灰虚线是随机猜对率。"
            "槽位变多后红线掉向随机水平，绿线仍接近 100%。右：16 个槽位时的训练损失。",
        "attention_weights.png":
            "把测试集上注意力权重对齐到“真值槽位 = 0”再取平均。柱子越高表示模型越常盯住那个相对位置。"
            "学会以后，几乎所有质量都集中在第 0 根柱（匹配键）。",
    },
    explain=explain,
)
