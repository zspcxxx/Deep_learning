from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    gc = m.get("gradient_check", {})
    if gc:
        worst_name, worst = max(((k, v["max_rel_err"]) for k, v in gc.items()), key=lambda kv: kv[1])
        ok = all(v["passed"] for v in gc.values())
        out.append(finding(
            f"梯度检查{'全部通过' if ok else '未通过'}：最大相对误差 {worst:.1e}（{worst_name}）",
            "对每个参数做中心差分 (L(θ+ε) − L(θ−ε)) / 2ε 得到数值梯度，再和反向传播算出的解析梯度比较。"
            "相对误差 |a − n| / (|a| + |n|) 在 1e-7 以下说明实现正确；1e-4 量级通常意味着某处公式写错。"
            f"这里最差也只有 {worst:.1e}，四组参数（W1、b1、W2、b2）的反向传播都可信。",
            "ok" if ok else "error",
        ))
    xa, ma = m.get("xor_accuracy"), m.get("moons_accuracy")
    if xa is not None:
        out.append(finding(
            f"XOR 训练准确率 {pct(xa)}，Two Moons 训练准确率 {pct(ma)}",
            "这里的网络只有一个 sigmoid 隐藏层（BackpropNN），学习率 1.5。和 02 课两层 ReLU 网络的 100% 相比稍低、"
            "收敛也慢一些：sigmoid 导数最大 0.25，梯度传回来时被打折，学得更慢。",
            "ok" if xa >= 90 and ma >= 85 else "warn",
        ))
    return out


NOTES = LessonNotes(
    intro="手写反向传播并用数值梯度检查证明它是对的，然后用同一份实现训练 XOR 和月牙数据。",
    steps=[
        "梯度检查：解析梯度 vs 数值梯度",
        "训练 XOR（单隐藏层 sigmoid 网络）",
        "训练 Two Moons",
        "画决策边界与训练曲线",
    ],
    log_notes=[
        LogNote(r"梯度检查（验证反向传播正确性）", "梯度检查",
                "用 40 个点的小 XOR 数据（data/03_backprop/gradcheck_xor.npz）。对每个参数逐个加减一个极小量 ε，"
                "看损失变化多少，得到数值梯度；再和反向传播给出的梯度比较。", step=0),
        LogNote(r"(?P<p>\w+) : max relative error = (?P<e>[\d.e+-]+) +✓", "{p} 通过（相对误差 {e}）",
                "每行是一组参数：W1 / b1 是隐藏层的权重和偏置，W2 / b2 是输出层的。相对误差远小于阈值 1e-6，"
                "说明这组参数的梯度公式没有写错；后面各行同理。", level="ok"),
        LogNote(r"(?P<p>\w+) : max relative error = (?P<e>[\d.e+-]+) +✗", "{p} 未通过（相对误差 {e}）",
                "解析梯度和数值梯度对不上，反向传播里这组参数的公式有错。", once=False, level="error"),
        LogNote(r"训练演示 1：XOR", "训练 XOR",
                "单隐藏层 8 个 sigmoid 神经元，学习率 1.5，全批量 1200 轮。", step=1),
        LogNote(r"训练演示 2：Two Moons", "训练 Two Moons", "单隐藏层 16 个 sigmoid 神经元，1500 轮。", step=2),
        LogNote(r"结果图已保存", "画图完成", "", step=3),
    ],
    image_notes={
        "backprop_result.png":
            "上排 XOR、下排 Two Moons，用的都是手写反向传播训练出的网络。左：决策边界；中：损失曲线；右：训练准确率。"
            "注意月牙的损失曲线前期下降较慢——sigmoid 隐藏层梯度偏小，需要更多轮才进入快速下降阶段。",
    },
    explain=explain,
)
