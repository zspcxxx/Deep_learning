from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct

# Why removing each part breaks (or doesn't break) training, tied to the lesson that introduced it.
WHY = {
    "no_pos": "注意力只看内容、不看位置（第 14 课的软查找就是按内容匹配）。没有位置向量，每个输出只取决于自己那个输入数字"
              "和整条序列有哪些数字，与它排在第几位无关，所以没法知道自己该输出第几小的数。",
    "no_residual": "4 个块直接串联，梯度要穿过 8 个子层才能回到嵌入层，和第 11 课的深层普通网络一样学不动；"
                   "有残差时，h + F(h) 的捷径让梯度原样传回。",
    "no_layernorm": "没有 LayerNorm，每个块的输出尺度没人管，残差流越加越大。学习率升到 0.1 附近时（第 100 步前后）损失被推爆，"
                    "这正是第 07 课“尺度失控就难学”的情形。",
    "no_warmup": "学习率第一步就是 0.1。Adam 刚开始时二阶矩估计还不准，参数一下被推得太远，"
                 "之后学习率按余弦降下来也没能爬回去。第 05 课讲过步子太大会发散；第 09 课的 warmup 就是先用小步子起跑。",
    "big_init": "Q、K 的权重各放大 4 倍，分数就放大 16 倍，softmax 一开始就几乎是 one-hot："
                "实测初始化时每个查询最大的那个注意力权重平均约 0.9（正常初始化约 0.3），梯度几乎传不过 softmax。"
                "第 06 课的结论在这里重现：初始化尺度不对，网络一开始就卡住。",
    "no_reg": "去掉 Dropout 和权重衰减后结果只差一点。这里泛化的主要来源是结构本身：同一组权重在 8 个位置共享（第 10 课的思想），"
              "参数比 MLP 少得多，过拟合空间小。正则化在第 08 课那种参数多、数据少的情形里作用更明显。",
    "linear_ffn": "FFN 不加激活，块里仍有 softmax 注意力这一处非线性，所以排序照样能学会。第 02 / 04 课说的"
                  "“全线性网络叠多少层都等于一层”，前提是整个网络没有任何非线性。",
}


def explain(m: dict) -> list[dict]:
    out = []
    gc = m.get("gradient_check", {})
    if gc:
        worst_name, worst = max(gc.items(), key=lambda kv: kv[1]["max_rel_err"])
        ok = all(v["passed"] for v in gc.values())
        out.append(finding(
            f"手写反向传播{'全部通过' if ok else '未通过'}梯度检查：{len(gc)} 组参数，最大相对误差 {worst['max_rel_err']:.1e}",
            "检查覆盖嵌入、位置向量、LayerNorm、Q/K/V/O 投影、FFN 和输出层，以及关掉残差和 LayerNorm 的变体（消融实验要用）。"
            f"误差最大的是 {worst_name}。做法和第 03 课一样：对每个参数加减一个很小的数，用损失差除以 2ε 得到数值梯度，再和反向传播的结果比。",
            "ok" if ok else "error",
        ))

    full = m.get("full") or {}
    model, fit, task = m.get("model", {}), m.get("fit", {}), m.get("task", {})
    if full:
        out.append(finding(
            f"完整模型：测试集整句全对 {pct(full['seq_acc'])}，逐位 {pct(full['token_acc'])}",
            f"{model.get('n_layers')} 个 pre-LN 块，{full['params']} 个参数，训练集只有 {task.get('n_train')} 条。"
            f"最优验证损失出现在第 {full.get('best_step')} 步，最终用的是那一步的参数（第 09 课的早停与回滚）。"
            "整句全对要求 8 个位置都对，比逐位准确率严格得多。",
            "ok" if full["seq_acc"] >= 95 else "warn",
        ))

    base = m.get("baselines") or {}
    mlp, lstm, tf = base.get("mlp"), base.get("lstm"), base.get("transformer")
    if mlp and tf:
        tr = m.get("mlp_train") or {}
        out.append(finding(
            f"MLP 参数是 Transformer 的 {mlp['params'] / tf['params']:.1f} 倍，测试整句只有 {pct(mlp['seq_acc'])}",
            f"MLP 把 8 个 one-hot 拼成 80 维一次性输入，每个位置都有自己的一套权重；"
            f"训练集整句 {pct(tr.get('seq_acc', 0))}，测试集 {pct(mlp['seq_acc'])}，差距就是过拟合（第 08 课）。"
            f"Transformer 的 Q/K/V 和 FFN 在所有位置共用同一组权重，和第 10 课卷积核在图像各处共用一样，"
            f"所以 {task.get('n_train')} 条数据就够它学会“比较大小”这条规则：测试整句 {pct(tf['seq_acc'])}。",
            "ok" if mlp["seq_acc"] < tf["seq_acc"] else "warn",
        ))
    if lstm and tf:
        out.append(finding(
            f"LSTM 先读后写也能做到 {pct(lstm['seq_acc'])}，但它必须一步一步走 16 个时间步",
            "做法和第 13 课的编码器–解码器一样：先把 8 个数字逐个读进记忆，再用 8 个“写”信号逐个吐出排好的数字。"
            "门控记忆（第 12 课）足够记住每个数字出现了几次，所以在这个短任务上不比 Transformer 差。"
            "区别在计算方式：LSTM 的第 t 步要等第 t−1 步算完；Transformer 一层之内所有位置同时计算，"
            "序列一长，这就是它能用 GPU 大规模训练的原因。",
        ))

    abl = m.get("ablations") or {}
    critical = [k for k in m.get("critical", []) if k in abl]
    if abl and critical:
        bad = [k for k in critical if abl[k]["seq_acc"] > 30]
        out.append(finding(
            f"一次去掉一个关键零件，整句准确率从 {pct(full.get('seq_acc', 0))} 掉到 "
            f"{min(abl[k]['seq_acc'] for k in critical):.1f}%–{max(abl[k]['seq_acc'] for k in critical):.1f}%",
            "；".join(f"{abl[k]['label']}（第 {abl[k]['lesson']} 课）{abl[k]['seq_acc']:.1f}%" for k in critical) + "。"
            "每一项都只改了一处，其余配置和完整模型完全相同，所以掉下来的准确率就是这个零件的贡献。",
            "ok" if not bad else "warn",
        ))
    for k, a in abl.items():
        extra = ""
        if a.get("diverged"):
            extra = f"训练损失最高到 {a['max_train_loss']:.1f}（正常应从 2.3 一路降到 0.01 以下）。"
        out.append(finding(
            f"{a['label']}（第 {a['lesson']} 课）：整句 {pct(a['seq_acc'])}，逐位 {pct(a['token_acc'])}",
            extra + WHY.get(k, ""),
            "info" if k not in critical else ("ok" if a["seq_acc"] <= 30 else "warn"),
        ))

    ex = m.get("example") or {}
    if ex:
        out.append(finding(
            "一条测试样本",
            f"输入 {ex.get('input')}，目标 {ex.get('target')}，模型输出 {ex.get('pred')}"
            f"{'，完全正确。' if ex.get('pred') == ex.get('target') else '。'}",
        ))
    return out


NOTES = LessonNotes(
    intro="把前 14 课的零件拼成一个小 Transformer：嵌入和位置向量、多头注意力、残差、LayerNorm、GELU 的 FFN、"
          "AdamW + Warmup + 余弦、Dropout 和权重衰减、早停。任务是给 8 个数字排序。"
          "先和 MLP、LSTM 对照，再一次只去掉一个零件，看对应那一课讲过的问题会不会重新出现。",
    steps=[
        "梯度检查（整个模型）",
        "训练完整模型",
        "对照：MLP 与 LSTM",
        "消融：一次去掉一个零件",
        "画图",
    ],
    log_notes=[
        LogNote(r"^数据: 每条样本是", "排序任务",
                "每个输出位置都要知道整条序列（有几个比自己小的数）和自己是第几位，"
                "所以既考“看全局”（注意力），又考“知道位置”（位置向量）。训练集故意只给 1000 条。"),
        LogNote(r"^完整模型: ", "完整配置：每一项都来自前面某一课",
                "pre-LN 块（07 LayerNorm + 11 残差 + 14 注意力 + 02/04 FFN 与 GELU），06 按 1/√fan_in 初始化，"
                "09 的 AdamW + Warmup + 余弦，08 的 Dropout 与权重衰减，12 的梯度裁剪。"
                "学习率故意开到 0.1（平常多用 1e-3 量级），用来检验 warmup 和 LayerNorm 是否在保护训练。"),
        LogNote(r"^梯度检查", "梯度检查（第 03 课）",
                "整个 Transformer 的反向传播都是手写的。先用中心差分逐组核对，再开始训练。", step=0),
        LogNote(r"(?P<p>\S+) +max relative error = (?P<e>[\d.e+-]+) +✗", "梯度检查未通过（{p}：{e}）",
                "反向传播和数值梯度对不上，后面的训练结果不可信。", once=False, level="error"),
        LogNote(r"共检查 (?P<n>\d+) 组参数，最大相对误差 (?P<e>\S+)", "{n} 组参数，最大误差 {e}",
                "通过线是 1e-4。只列出误差最大的几组。"),
        LogNote(r"^--- 完整模型 ---", "训练完整模型", "", step=1),
        LogNote(r"step +\d+ \| loss=[\d.]+ \| lr=", "怎么读训练日志",
                "每 50 步在验证集上评估一次：loss 是这 50 步的平均训练损失，lr 是当前学习率（前 100 步线性升到 0.1，再按余弦降），"
                "token 是逐位准确率，seq 是 8 位全对的比例。"),
        LogNote(r"\[完整模型\] test token=(?P<t>[\d.]+)% seq=(?P<s>[\d.]+)%", "完整模型：整句 {s}%",
                "这是后面所有消融实验的对照基准。", level="ok"),
        LogNote(r"^--- 对照：MLP", "对照 1：MLP（第 02 / 09 课）",
                "把 8 个 one-hot 拼成 80 维，直接输出 8×10 个分数。没有位置共享，参数约是 Transformer 的 3 倍。", step=2),
        LogNote(r"\[MLP\] test token=(?P<t>[\d.]+)% seq=(?P<s>[\d.]+)% \| train token=(?P<tt>[\d.]+)% seq=(?P<ts>[\d.]+)%",
                "MLP：测试 {s}%，训练 {ts}%", "训练集几乎全对、测试集差一截，是典型的过拟合。"),
        LogNote(r"^--- 对照：LSTM", "对照 2：LSTM 先读后写（第 12 / 13 课）",
                "先逐个读入 8 个数字，再给 8 个“写”信号逐个输出，和第 13 课的编码器–解码器同一思路。"),
        LogNote(r"\[LSTM\] test token=(?P<t>[\d.]+)% seq=(?P<s>[\d.]+)%", "LSTM：整句 {s}%",
                "门控记忆足够数清每个数字出现几次。它和 Transformer 的区别在于必须按时间步串行计算。"),
        LogNote(r"^--- 消融 A：", "消融：一次只改一处", "下面每个实验只改一项配置，其余和完整模型完全相同。", step=3),
        LogNote(r"消融 .：去掉位置编码", "去掉位置编码（第 14 课）", WHY["no_pos"]),
        LogNote(r"消融 .：去掉残差连接", "去掉残差（第 11 课）", WHY["no_residual"]),
        LogNote(r"消融 .：去掉 LayerNorm", "去掉 LayerNorm（第 07 课）", WHY["no_layernorm"]),
        LogNote(r"消融 .：去掉学习率 warmup", "去掉 warmup（第 05 / 09 课）", WHY["no_warmup"]),
        LogNote(r"消融 .：初始化放大", "初始化放大 4 倍（第 06 课）", WHY["big_init"]),
        LogNote(r"消融 .：去掉 Dropout", "去掉正则化（第 08 课）", WHY["no_reg"]),
        LogNote(r"消融 .：FFN 去掉激活", "FFN 去掉激活（第 02 / 04 课）", WHY["linear_ffn"]),
        LogNote(r"\[(?!完整模型|MLP|LSTM)(?P<name>[^\]]+)\] test token=(?P<t>[\d.]+)% seq=(?P<s>[\d.]+)%",
                "{name}：整句 {s}%（逐位 {t}%）", "和完整模型比，差多少就是这个零件的贡献。", once=False),
        LogNote(r"训练中损失爆到 (?P<l>[\d.]+)", "训练损失爆到 {l}",
                "正常训练的损失从 2.3（随机猜 10 个数字）一路降到 0.01 以下。冲到两位数说明数值已经失控。",
                once=False, level="warn"),
        LogNote(r"^  例子 输入=", "一条测试样本", "对照输入、目标和模型输出。"),
        LogNote(r"结果已保存到", "画图完成", "", step=4),
    ],
    image_notes={
        "capstone_map.png":
            "知识地图：左边从下往上是数据流经的每个零件，右边是训练和评估用到的方法，蓝色标签是它第一次出现的课。"
            "虚线框里的块重复 4 次。整节课的消融实验就是在这张图上一次拿掉一个框。",
        "capstone_training.png":
            "完整模型的训练过程。左：训练损失（对数坐标），从约 2.3 降到 0.01 以下。"
            "中：验证集准确率，蓝虚线是逐位、绿线是 8 位全对；全对更严格，起步慢但最终追上。灰色竖线是验证损失最低的一步。"
            "右：学习率，前 100 步线性升到 0.1，然后余弦降到 0.001（第 09 课的调度）。",
        "capstone_baselines.png":
            "同样 1000 条训练数据。左：测试准确率，深绿是整句全对、浅蓝是逐位。右：参数量。"
            "MLP 参数最多却最差：每个位置一套独立权重，数据不够它学成通用规则。LSTM 参数最少也能学会，但要串行走 16 步。",
        "capstone_ablation.png":
            "左：每一行只去掉一个零件后的测试准确率，柱子是整句全对，黑色竖线是逐位准确率。"
            "红色（看不见的短柱）表示整句几乎为 0：位置编码、残差、LayerNorm、warmup、初始化，任一项出问题模型都学不会。"
            "灰色两项（正则化、FFN 激活）在本任务上影响很小。"
            "右：各次训练的损失曲线（平滑、对数坐标）。学会的曲线一路下降到 0.01 以下；"
            "学不会的停在 1 到 2 多一点；去掉 LayerNorm 的那条在第 100 步前后冲到 20 以上。曲线长短不同是因为早停。",
        "capstone_attention.png":
            "完整模型对一条测试样本的注意力（每块 4 个头取平均）。行是输出位置（冒号后是该位置的目标数字），"
            "列是输入的 8 个数字，颜色越亮权重越大。有的块权重比较分散，像在把整条序列的信息汇总到每个位置；"
            "有的块里几行会集中盯住某一个输入数字。排序是 4 个块配合完成的，单看一块不一定能读出“第 i 小”。",
    },
    explain=explain,
)
