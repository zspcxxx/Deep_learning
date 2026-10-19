from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    gc = m.get("gradient_check", {})
    if gc:
        worst_name, worst = max(gc.items(), key=lambda kv: kv[1]["max_rel_err"])
        ok = all(v["passed"] for v in gc.values())
        out.append(finding(
            f"编码器和解码器的反向传播{'通过' if ok else '未通过'}梯度检查：最大相对误差 {worst['max_rel_err']:.1e}（{worst_name}）",
            "训练时解码器的输入是正确答案的上一个符号（teacher forcing），损失是每个输出位置的交叉熵。"
            "固定向量模型只把编码器最后一个隐藏状态交给解码器；注意力模型在每一步都对全部编码器状态做加权和。"
            "有些参数的梯度本身只有 1e-7，中心差分的绝对误差大约 1e-11，相对误差会到 1e-5 左右。通过线设在 1e-4；实现反了的话相对误差会接近 1。",
            "ok" if ok else "error",
        ))
    res = m.get("results", {})
    lengths = [str(n) for n in m.get("lengths", [])]
    seq, att = res.get("seq2seq", {}), res.get("attention", {})
    if seq and att and lengths:
        long = lengths[-1]
        out.append(finding(
            f"整句翻对：长度 {lengths[0]} 时两种模型都可以，长度 {long} 时注意力 {pct(att[long]['exact'])}、固定向量 {pct(seq[long]['exact'])}",
            "任务是把随机符号倒过来，再输出一个结束符。整句必须每个位置都对才算对。"
            "固定向量：" + "，".join(f"长度 {n} 为 {seq[n]['exact']:.0f}%" for n in lengths) + "。"
            "注意力：" + "，".join(f"长度 {n} 为 {att[n]['exact']:.0f}%" for n in lengths) + "。"
            "源序列变长以后，一个固定长度的隐藏向量装不下整句；注意力让解码器在需要哪个符号时回去看对应的位置。",
            "ok" if att[long]["exact"] >= 95 and seq[long]["exact"] <= 20 else "warn",
        ))
    ex = m.get("example") or {}
    if ex:
        out.append(finding(
            "注意力对齐的是“正在输出的那个源位置”",
            f"一条长度 {len(ex.get('src', []))} 的测试序列：源 {ex.get('src')}，目标 {ex.get('target')}，模型输出 {ex.get('pred')}。"
            "热力图的每一行是解码器的一步，每一列是源序列的一个位置。倒序任务里，第 1 个输出应对齐源序列的最后一位，所以亮带是反对角线。",
        ))
    return out


NOTES = LessonNotes(
    intro="把第 12 课的 GRU 接成编码器–解码器：编码器读完整句，解码器一个符号一个符号地写出来。"
          "先用最后一个隐藏向量传递整句，再让解码器在每一步用注意力回头看编码器的全部状态。任务是把随机符号倒过来。",
    steps=[
        "梯度检查（固定向量 / 注意力）",
        "倒序：长度 4",
        "倒序：长度 8",
        "倒序：长度 12",
        "画准确率曲线和注意力热力图",
    ],
    log_notes=[
        LogNote(r"^数据: 源序列", "倒序任务",
                "每条样本是一串随机符号。编码器只看见源序列；解码器要输出倒过来的序列，并在最后输出 EOS。"
                "训练时用 teacher forcing：不管模型上一步猜得对不对，下一步都喂给它正确的上一个符号。评估时改成贪心解码，把模型自己的输出喂回去。"),
        LogNote(r"^梯度检查", "梯度检查",
                "对编码器 GRU、解码器 GRU、输出层，以及注意力的 Wa、Ua、va，用中心差分核对反向传播。", step=0),
        LogNote(r"(?P<p>\S+) +max relative error = (?P<e>[\d.e+-]+) +✗", "梯度检查未通过（{e}）",
                "反向传播和数值梯度对不上。", once=False, level="error"),
        LogNote(r"--- 倒序任务：长度 4", "短序列",
                "两种模型隐藏层都是 32 维，更新门偏置为 3。短序列可以塞进一个向量，所以固定向量模型也能学会。", step=1),
        LogNote(r"--- 倒序任务：长度 8", "中等长度", "源序列加长，固定向量要记住的内容变多。", step=2),
        LogNote(r"--- 倒序任务：长度 12", "长序列",
                "固定向量仍然只有 32 个数。注意力则在解码的每一步对 12 个编码器状态做 softmax 加权。", step=3),
        LogNote(r"\[seq2seq\] exact=(?P<a>[\d.]+)%", "固定向量：整句 {a}%",
                "整句准确率要求每个位置（含结束符）都正确。长度一长，它往往只能对上靠近末尾的几个符号。", once=False),
        LogNote(r"\[attention\] exact=(?P<a>[\d.]+)%", "注意力：整句 {a}%",
                "解码器不需要把整句都记在一个向量里，需要哪个符号就去看哪个位置。", once=False, level="ok"),
        LogNote(r"^  例子 src=", "一条样本", "对比源序列、正确的倒序（末尾是 EOS）和模型贪心解码的结果。"),
        LogNote(r"结果已保存到", "画图完成", "", step=4),
    ],
    image_notes={
        "seq2seq_accuracy.png":
            "左：整句翻对的比例。红线是只传递编码器最后一个向量的 seq2seq，绿线是每一步都做注意力的编码器–解码器。"
            "短序列两者都高；源序列变长后红线掉下去，绿线仍然接近 100%。右：最长序列上的训练损失。",
        "seq2seq_attention.png":
            "把测试集上的注意力权重按样本取平均。横轴是源序列的位置，纵轴是解码器的第几步（最后一行是 EOS）。"
            "倒序时第 1 步要输出源序列的最后一个符号，所以颜色最深的一条带从右上走向左下。",
    },
    explain=explain,
)
