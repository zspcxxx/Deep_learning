from __future__ import annotations

import math

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    gc = m.get("gradient_check", {})
    if gc:
        worst_name, worst = max(gc.items(), key=lambda kv: kv[1]["max_rel_err"])
        ok = all(v["passed"] for v in gc.values())
        out.append(finding(
            f"随时间反向传播（BPTT）{'全部通过' if ok else '未通过'}梯度检查：最大相对误差 {worst['max_rel_err']:.1e}",
            "RNN 在每个时间步用的是同一组权重，所以 Wx、Wh 的梯度是所有时间步贡献之和；反向传播从最后一步开始，"
            "把 dL/dh_t 一步步往前传（dh_{t-1} = (dh_t ⊙ tanh′) · Whᵀ），同时累加每一步的权重梯度。"
            "RNN、LSTM 两种单元，在“只看最后一步”（记忆任务）和“每一步都预测下一个字符”（名字）两种用法下，都和数值梯度对上了。"
            "tanh、sigmoid 处处光滑，所以阈值可以用严格的 1e-6。",
            "ok" if ok else "error",
        ))

    mem = m.get("memory", {})
    delays = [str(d) for d in m.get("delays", [])]
    rnn, lstm, fb1 = mem.get("rnn", {}), mem.get("lstm", {}), mem.get("lstm_fb1", {})
    n_sym = m.get("n_symbols", 4)
    if rnn and lstm and delays:
        long = delays[-1]
        out.append(finding(
            f"长期记忆：延迟 {long} 步时 RNN {pct(rnn[long]['accuracy'])}，LSTM {pct(lstm[long]['accuracy'])}",
            "任务：读一串随机符号，读完后说出第 1 个是什么。后面的符号和第 1 个来自同一个字母表，网络必须一直“记着”开头、同时忽略中间那些长得一样的干扰。"
            "各延迟下的测试准确率——RNN：" + "，".join(f"{d} 步 {rnn[d]['accuracy']:.0f}%" for d in delays) +
            "；LSTM：" + "，".join(f"{d} 步 {lstm[d]['accuracy']:.0f}%" for d in delays) + "。"
            f"随机猜的准确率是 {100 / n_sym:.0f}%：RNN 在长延迟下的训练损失一直停在 ln{n_sym} ≈ {math.log(n_sym):.3f}，等于完全没学到。",
            "ok" if lstm[long]["accuracy"] >= 95 else "warn",
        ))
    gf = m.get("grad_flow_first_over_last", {})
    if gf:
        out.append(finding(
            "为什么 RNN 记不住：梯度沿时间指数衰减",
            f"训练开始前，把损失对第 1 步输入的梯度和对最后一步输入的梯度相比："
            f"RNN 只有 {gf.get('rnn', 0):.1e}，LSTM（遗忘门偏置 {m.get('forget_bias', 3):g}）有 {gf.get('lstm', 0):.2f}。"
            "RNN 每往前一步，梯度就要乘一次 Whᵀ 和 tanh 的导数（≤ 1），几十步连乘下来几乎为 0——这和第 04、11 课里深层网络的梯度消失是同一件事，"
            "只不过“层”换成了“时间步”。第 1 步的信息对损失的影响传不回去，网络也就学不会去记它。\n"
            "LSTM 的细胞状态 c_t = f ⊙ c_{t-1} + i ⊙ g 是加法更新：沿 c 往回传梯度时每步只乘遗忘门 f，f 接近 1 时梯度几乎原样通过——"
            "和第 11 课残差连接的 h + F(h) 是同一个思路。",
        ))
    if fb1 and lstm and delays:
        long = delays[-1]
        out.append(finding(
            f"遗忘门偏置很关键：偏置 1 时 LSTM 在延迟 {long} 步只有 {pct(fb1[long]['accuracy'])}",
            f"遗忘门 f = sigmoid(… + b_f)。偏置为 1 时，初始 f ≈ sigmoid(1) ≈ 0.73，{long} 步连乘 0.73^{long} ≈ {0.73 ** int(long):.0e}，"
            f"梯度同样传不回去（初始化时第 1 步的梯度比例只有 {gf.get('lstm_fb1', 0):.1e}）。"
            f"偏置设为 {m.get('forget_bias', 3):g} 时 f ≈ sigmoid({m.get('forget_bias', 3):g}) ≈ 0.95，细胞一开始就倾向于“保留记忆”。"
            "这和第 11 课“让残差块从恒等映射起步”是同一个道理：先让信息和梯度能通过，再让网络学会该忘什么。",
            "warn",
        ))
    fg = m.get("forget_gate", {})
    if fg:
        out.append(finding(
            f"训练后的遗忘门：{fg['units_above_0.95']} / {fg['hidden']} 个单元的平均 f > 0.95",
            f"在延迟 {delays[-1] if delays else ''} 步的测试序列上统计训练好的 LSTM，遗忘门均值最大 {fg['max']:.3f}、最小 {fg['min']:.3f}。"
            "大部分单元几乎从不遗忘，负责把第 1 个符号一路带到最后。同时，训练后的梯度曲线显示中间各步输入的梯度很小——"
            "网络学会了用输入门把后面的干扰挡在外面。",
            "ok",
        ))

    names = m.get("names", {})
    nr, nl = names.get("rnn"), names.get("lstm")
    if nr and nl:
        out.append(finding(
            f"生成名字：验证集损失 LSTM {nl['val_loss']:.3f}、RNN {nr['val_loss']:.3f} nats/字符",
            f"模型每读一个字符就预测下一个字符。基线：均匀乱猜 {names['uniform_loss']:.3f}，只按字母频率猜 {names['unigram_loss']:.3f}。"
            f"两个模型都远好于基线，说明学到了字母之间的搭配和名字的结构（在训练数据里，q 后面约 3/4 跟的是 u；约 42% 的名字以 n 或 a 结尾）。"
            f"名字很短（最长 15 个字母），不需要长期记忆，所以普通 RNN 和 LSTM 差别不大；参数量 RNN {nr['params']:,}、LSTM {nl['params']:,}，"
            f"训练耗时 RNN {nr['train_sec']} 秒、LSTM {nl['train_sec']} 秒。",
            "ok",
        ))
        s = nl["samples"]
        out.append(finding(
            "温度：在“像训练集”和“有创意”之间取舍",
            "生成时把输出 logits 除以温度 T 再做 softmax：T < 1 分布更尖，更常选最可能的字母；T > 1 分布更平，更敢冒险。"
            + "；".join(f"T={t}：{v['novel_pct']:.0f}% 是训练集里没有的新名字，例如 {', '.join(v['examples'][:4])}" for t, v in s.items())
            + "。T 太低时很多名字是“背”出来的，T 太高时开始出现不像名字的字母组合。",
        ))
    return out


NOTES = LessonNotes(
    intro="手写 RNN 和 LSTM 的随时间反向传播（BPTT）。先用“记住第 1 个符号”的任务对比两者的长期记忆能力，"
          "解释梯度为什么会沿时间消失、LSTM 的门和遗忘门偏置怎样解决它；再训练一个按字符生成英文名字的模型，看它在训练中一步步学会“造名字”。",
    steps=[
        "梯度检查（RNN / LSTM × 最后一步 / 每一步）",
        "记忆任务：延迟 5 / 10 / 20 / 40 步，RNN vs LSTM",
        "遗忘门偏置对比（延迟 40 步）",
        "梯度沿时间回传与遗忘门分析",
        "生成名字：RNN",
        "生成名字：LSTM",
        "画记忆曲线、梯度回传、名字生成图",
    ],
    log_notes=[
        LogNote(r"^数据: 记忆任务", "两个任务",
                "记忆任务：每条序列由随机符号组成（4 种），读到最后时要说出第 1 个符号是什么；延迟越长越难，训练数据每一步都现场随机生成，"
                "测试集固定缓存在 data/12_rnn/。名字任务：3.2 万个英文名（data/names/names.txt），学习“看到前面的字母，预测下一个字母”。"),
        LogNote(r"^梯度检查（随时间反向传播", "梯度检查 BPTT",
                "对 RNN、LSTM 两种单元，分别在“只在最后一步计算损失”和“每一步都计算损失（带填充掩码）”两种模式下，"
                "用中心差分核对 BPTT 算出的梯度。cell.Wx / Wh / b 是循环单元的参数，out.Wy / by 是输出层。", step=0),
        LogNote(r"(?P<p>\S+) +max relative error = (?P<e>[\d.e+-]+) +✗", "梯度检查未通过（{e}）", "BPTT 实现有误。",
                once=False, level="error"),
        LogNote(r"--- 记忆任务：延迟 5 步", "记忆任务开始",
                "两个模型隐藏层都是 32 维。RNN：h_t = tanh(x_t·Wx + h_{t-1}·Wh + b)；LSTM 多了输入门 i、遗忘门 f、输出门 o 和细胞状态 c。"
                "Adam（lr 3e-3，余弦衰减）训练 1500 步，每步 64 条新序列，梯度范数裁剪到 1——RNN 训练时梯度偶尔会突然变很大，裁剪能防止一步走飞。",
                step=1),
        LogNote(r"--- 记忆任务：延迟 40 步", "最难的延迟：40 步", "第 1 个符号后面跟着 39 个干扰符号。"),
        LogNote(r"\[rnn\] acc=(?P<a>[\d.]+)% final_loss=1\.38", "RNN 学不会（{a}%）",
                "损失停在 ln4 ≈ 1.386：对 4 个符号给出均等的概率，等于随机猜。不是训练不够久——梯度根本传不回第 1 步。", level="error"),
        LogNote(r"--- 遗忘门偏置对比", "遗忘门偏置对比",
                "同样的 LSTM，只把遗忘门偏置从 3 改回常见的默认值 1，看看初始化对长期记忆的影响。", step=2),
        LogNote(r"\[lstm_fb1\] acc=(?P<a>[\d.]+)%", "遗忘门偏置 1：{a}%",
                "初始遗忘门 ≈ 0.73，40 步后梯度只剩约 0.73^40 ≈ 3e-6，和 RNN 一样学不会。", level="warn"),
        LogNote(r"梯度沿时间回传（初始化时", "梯度沿时间回传",
                "只在最后一步计算损失，看损失对每一步输入 x_t 的梯度有多大，再除以最后一步的值。这个比例越接近 1，说明越早的输入越能影响训练。",
                step=3),
        LogNote(r"\[rnn\] ‖dL/dx_1‖ / ‖dL/dx_\d+‖ = (?P<r>\S+)", "RNN：第 1 步只收到 {r}",
                "每往前一步都要乘一次 Whᵀ 和 tanh 的导数，几十步连乘后指数衰减。", level="warn"),
        LogNote(r"\[lstm\] ‖dL/dx_1‖ / ‖dL/dx_\d+‖ = (?P<r>\S+)", "LSTM：第 1 步收到 {r}",
                "梯度沿细胞状态往回传时每步只乘遗忘门 f ≈ 0.95，衰减慢得多。", level="ok"),
        LogNote(r"训练后 LSTM 的遗忘门均值", "训练后的遗忘门",
                "遗忘门接近 1 的单元就是“记忆单元”：它们把第 1 个符号的信息一路保存到序列结束。"),
        LogNote(r"^名字数据：训练", "名字数据",
                "每个名字前后各加一个 '.'：输入 '.emma'，目标 'emma.'。不同长度的名字补齐到同样长度，用掩码让补齐的位置不计入损失。"),
        LogNote(r"基线：均匀猜测", "怎么判断学得好不好",
                "损失单位是 nats/字符（交叉熵）。均匀乱猜是 ln27；只按字母出现频率猜是另一个基线。模型必须明显低于它们才算学到了字母之间的规律。"),
        LogNote(r"--- 生成名字：\[rnn\]", "生成名字：RNN",
                "Adam（lr 5e-3）训练 1500 步，每步 64 个名字，梯度裁剪到 5。每 300 步在验证集上算一次损失，并用当前模型随机生成 6 个名字。",
                step=4),
        LogNote(r"step +1 \| val_loss", "训练前",
                "还没训练时生成的就是随机字母串，验证损失约等于 ln27 ≈ 3.30。", once=False),
        LogNote(r"step +300 \| val_loss", "300 步后",
                "很快就开始出现元音辅音交替、像名字的字母串，损失降到 2.2 左右。后面的进步主要在细节上。"),
        LogNote(r"--- 生成名字：\[lstm\]", "生成名字：LSTM", "同样的设置换成 LSTM（遗忘门偏置 3）。", step=5),
        LogNote(r"温度 T=1\.5", "温度",
                "T 越低越保守、越像训练集里的名字（重复率高）；T 越高越有创意，但开始出现不像名字的组合。"),
        LogNote(r"结果已保存到", "画图完成", "", step=6),
    ],
    image_notes={
        "rnn_memory.png":
            "左：不同延迟下的测试准确率，虚线是随机猜的 25%。RNN（红）在 20 步以内全对，40 步时掉到随机水平；"
            "LSTM（绿，遗忘门偏置 3）一直是 100%；黄色方块是遗忘门偏置 1 的 LSTM，在 40 步时同样失败。"
            "右：延迟 40 步时的训练损失。LSTM（偏置 3）三百步内就降到接近 0；另外两条一直贴在 ln4 ≈ 1.39 的虚线上，一点都没学到。",
        "rnn_gradflow.png":
            "左：延迟 40 步时，损失对每一步输入的梯度（除以最后一步的值，对数坐标）。实线是训练前：RNN（红）往前每一步都在衰减，第 1 步只剩 1e-5；"
            "遗忘门偏置 1 的 LSTM（黄）衰减慢一些；偏置 3 的 LSTM（绿）几乎是平的。虚线是训练好的 LSTM：第 1 步的梯度明显高于中间各步——"
            "它学会了只关心第 1 个符号、忽略中间的干扰。右：训练好的 LSTM 每个单元的平均遗忘门（从大到小排序），大部分在 0.95 以上，几乎从不遗忘。",
        "rnn_names.png":
            "左：按字符预测的损失，细线是训练集（平滑后），圆点是验证集；两条虚线是“均匀乱猜”和“只按字母频率猜”的基线。"
            "两个模型都在 300 步内就明显超过基线，LSTM 最终略好。右：LSTM 在三种温度下生成的名字，加粗的是训练集里不存在的新名字；"
            "温度越高新名字越多，但 T = 1.5 时开始出现不像名字的拼写。",
    },
    explain=explain,
)
