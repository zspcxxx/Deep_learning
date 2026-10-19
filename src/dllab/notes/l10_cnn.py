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
            "三通道（RGB）输入下，卷积层（im2col 后 dW = dᵀ·cols，再用 col2im 把 dcols 加回输入位置）、"
            "最大池化（梯度只回传给每个窗口的最大值位置）、全连接层的反向传播全部和数值梯度对上了。"
            "检查用的是一个更小的 float64 网络（4 / 6 个通道），dropout 关闭，保证前向是确定性的。",
            "ok" if ok else "error",
        ))

    models = m.get("models", {})
    mlp, cnn, aug = models.get("mlp"), models.get("cnn"), models.get("cnn_aug")
    if not (mlp and cnn and aug):
        return out
    out.append(finding(
        f"CNN + 增强：准确率 {pct(aug['accuracy'])}，MLP {pct(mlp['accuracy'])}；参数 {aug['n_params']:,} vs {mlp['n_params']:,}",
        f"CNN 的参数只有 MLP 的约 {aug['n_params'] / mlp['n_params']:.0%}，准确率却高 {aug['accuracy'] - mlp['accuracy']:.1f} 个百分点。"
        "MLP 把 32×32×3 = 3072 个数直接连到 128 个神经元，第一层就有 39 万个参数，而且每个像素位置的权重都不一样——"
        "猫出现在左边和右边，对它来说是两件完全不同的事。CNN 用同一组 3×3 卷积核扫过整张图（权重共享），"
        "再逐层池化，把“边缘 → 纹理 → 耳朵、眼睛这样的局部部件”组合起来，这正是识别物体需要的结构。",
        "ok" if aug["accuracy"] > mlp["accuracy"] else "warn",
    ))
    out.append(finding(
        f"数据增强：{pct(cnn['accuracy'])} → {pct(aug['accuracy'])}",
        f"同一个 CNN，不做增强时第 {cnn['epochs']} 轮验证集就最好了，之后训练损失继续下降、验证损失回升，开始记忆训练图片。"
        "加上随机水平翻转和随机平移 2 个像素后，每一轮看到的都是“新”图片：一只朝左的猫翻转后仍然是猫。"
        f"验证集最好的轮次推迟到第 {aug['epochs']} 轮，训练和验证损失始终贴在一起。数据增强相当于免费扩充数据集，是图像任务最常用的正则化手段。",
        "ok" if aug["accuracy"] >= cnn["accuracy"] else "warn",
    ))
    out.append(finding(
        f"检出率（recall）{pct(aug['recall'])}，精确率（precision）{pct(aug['precision'])}",
        f"测试集里 {aug['tp'] + aug['fn']} 张猫，找出了 {aug['tp']} 张，漏掉 {aug['fn']} 张；"
        f"判为“猫”的 {aug['tp'] + aug['fp']} 张里有 {aug['fp']} 张其实不是猫（误报）。"
        "检测任务通常同时看这两个数：recall 低是“漏检”，precision 低是“误报”。准确率把两种错误混在一起，看不出模型偏向哪一边。",
    ))
    fa = m.get("false_alarm_by_class", {})
    if fa:
        ranked = sorted(fa.items(), key=lambda kv: -kv[1])
        top, rate = ranked[0]
        rows = "，".join(f"{k} {v:.0f}%" for k, v in ranked)
        out.append(finding(
            f"最容易被当成猫的是 {top}（{rate:.0f}% 被误报）",
            f"各类非猫图片被判为猫的比例：{rows}。"
            "狗和猫都是四条腿、有毛、尖耳朵或圆脸的小动物，在 32×32 的分辨率下非常难区分；"
            "而汽车、卡车、飞机这类形状规则的人造物体很少被误认。误报集中在“长得像”的类别，说明模型学到的是动物的外观特征，而不是背景之类的偶然线索。",
            "warn",
        ))
    sweep = m.get("threshold_sweep", [])
    if sweep:
        hi = max(sweep, key=lambda s: s["threshold"])
        lo = min(sweep, key=lambda s: s["threshold"])
        out.append(finding(
            "阈值决定“宁可错报”还是“宁可漏报”",
            "模型输出的是 P(猫)，超过阈值才算检测到猫，默认阈值 0.5。"
            f"把阈值降到 {lo['threshold']}，几乎所有猫都能找到（recall {lo['recall']:.0f}%），但 precision 只有 {lo['precision']:.0f}%；"
            f"升到 {hi['threshold']}，判为猫的基本都对（precision {hi['precision']:.0f}%），但只能找到 {hi['recall']:.0f}% 的猫。"
            "实际应用按代价选阈值：漏掉猫代价高就调低，误报代价高就调高。",
        ))
    out.append(finding(
        f"训练耗时：CNN + 增强 {aug['train_sec']} 秒，CNN {cnn['train_sec']} 秒，MLP {mlp['train_sec']} 秒",
        "纯 NumPy 实现下卷积很慢：每个 3×3 窗口都要先展开（im2col），计算量和内存搬运都比全连接大。"
        "在 GPU 上用 cuDNN 这类优化库，同样的网络可以快上百倍。",
    ))
    return out


NOTES = LessonNotes(
    intro="用手写的卷积神经网络判断一张 32×32 的彩色照片里有没有猫（CIFAR-10：猫 vs 其他 9 类）。"
          "先用梯度检查验证实现，再对比 MLP、CNN、CNN + 数据增强，最后用检出率 / 误报率和遮挡热力图分析模型到底在看哪里。",
    steps=[
        "加载猫 / 非猫数据（data/cat/cat_vs_noncat.npz），按通道归一化",
        "梯度检查（data/10_cnn/gradcheck_batch.npz 中的 4 张图）",
        "训练 MLP 基线（09 课 tuned 配方）",
        "训练 CNN（不做数据增强）",
        "训练 CNN + 数据增强（翻转 + 平移）",
        "评估：准确率、检出率、精确率、各类误报",
        "画曲线、卷积核 / 特征图、检测结果、遮挡热力图、指标图",
    ],
    log_notes=[
        LogNote(r"数据: data/cat", "数据：CIFAR-10 猫 vs 非猫",
                "猫是 CIFAR-10 的第 3 类；“非猫”从另外 9 类（飞机、汽车、鸟、鹿、狗、青蛙、马、船、卡车）中均匀抽取，两类各占一半。"
                "训练 5000 张、验证 1000 张（用于早停）、测试 2000 张（只在最后评估一次）。图片只有 32×32，人眼看也有点模糊。", step=0),
        LogNote(r"按通道归一化", "按通道归一化",
                "R、G、B 三个通道分别减去训练集的均值、除以标准差。不用逐像素标准化，是为了保留像素之间的空间关系——卷积要靠这个。"),
        LogNote(r"梯度检查（", "梯度检查",
                "用一个小号 CNN（4 / 6 个通道、float64、关闭 dropout），对每组参数随机抽 10 个元素做中心差分，"
                "和反向传播结果比较。阈值 1e-5。", step=1),
        LogNote(r"(?P<p>\S+) +max relative error = (?P<e>[\d.e+-]+) +✓", "{p} 通过（{e}）",
                "每行是一组参数（conv 的卷积核 W 和偏置 b，fc 的权重和偏置）。相对误差远低于阈值 1e-5，"
                "说明这组参数的梯度公式正确；后面各行同理。", level="ok"),
        LogNote(r"(?P<p>\S+) +max relative error = (?P<e>[\d.e+-]+) +✗", "{p} 未通过（{e}）",
                "解析梯度和数值梯度对不上。", once=False, level="error"),
        LogNote(r"--- \[mlp\]", "MLP 基线",
                "把 32×32×3 的图片拉平成 3072 维向量，沿用 09 课 tuned 的结构和超参数。它看不到“哪些像素挨在一起”。", step=2),
        LogNote(r"--- \[cnn\]", "CNN（不增强）",
                "三个卷积块，每块是 3×3 卷积 → ReLU → 2×2 最大池化，通道数 8 → 16 → 32，图片尺寸 32 → 16 → 8 → 4；"
                "然后接 64 个神经元的全连接层、Dropout 0.3、2 分类输出。AdamW + Warmup/Cosine，patience 5。每轮约 3.5 秒。", step=3),
        LogNote(r"--- \[cnn_aug\]", "CNN + 数据增强",
                "网络和超参数完全相同，只是每个训练 batch 在送进网络前随机水平翻转、并随机平移最多 2 个像素（边缘补 0）。"
                "验证集和测试集不做增强。", step=4),
        LogNote(r"^评估", "评估",
                "accuracy 是整体准确率；precision（精确率）= 判为猫的里面真是猫的比例；recall（检出率）= 所有猫里被找出来的比例；"
                "F1 是两者的调和平均。", step=5),
        LogNote(r"非猫图片被误判为猫的比例", "误报来自哪些类别",
                "把测试集中的非猫图片按原始类别分组，统计每组被判为猫的比例，可以看出模型最容易把什么当成猫。"),
        LogNote(r"结果已保存到", "画图完成", "", step=6),
    ],
    image_notes={
        "cnn_curves.png":
            "左：训练损失（实线）和验证损失（虚线）。CNN（蓝）不增强时，训练损失一路下降到 0.33，验证损失却在第 6 轮后回升——过拟合；"
            "CNN + 增强（绿）的两条线一直贴在一起。MLP（红）的验证损失最高。右：验证准确率，增强后的 CNN 稳定在 80% 左右，MLP 在 70% 上下。",
        "cnn_filters.png":
            "左：测试集里 CNN 最确定是猫的一张图。第一行：conv1 学到的 8 个 3×3 卷积核，按 RGB 显示（每个核单独拉伸到 0–1）；"
            "第二行：每个卷积核在这张图上的 ReLU 特征图（32×32）——有的对颜色区域响应，有的对边缘响应，猫的耳朵、脸部轮廓清晰可见。"
            "下面两行：conv2 的 16 张特征图（16×16），是在 conv1 特征上的再组合，开始变得抽象。",
        "cnn_detect.png":
            "随机抽取的 16 张测试图（CNN + 增强）。上排是猫，下排是非猫；标题是原始类别和模型给出的 P(猫)。"
            "绿色表示在阈值 0.5 下判断正确，红色表示判错。注意那些 P(猫) 接近 0.5 的图——模型自己也拿不准。",
        "cnn_occlusion.png":
            "遮挡热力图：用 8×8 的灰块（均值颜色）依次遮住图片的每个位置，看“是猫”的得分（log-odds）下降多少。"
            "红色区域被遮住后得分掉得最多，就是模型判断“这是猫”时最依赖的地方。前五列是被正确检出的猫（按把握从高到低），"
            "热点落在猫的头部（脸、耳朵）或身体上，而不是背景——说明模型确实在看猫本身。"
            "最后一列是模型最有把握的一次误报（标题是它真实的类别），可以看看是图里哪块区域让它以为是猫。",
        "cnn_metrics.png":
            "左上：三个模型在测试集上的准确率、精确率、检出率、F1。右上：CNN + 增强在不同阈值下的精确率和检出率——阈值越高越“谨慎”，"
            "精确率上升、检出率下降，虚线是默认阈值 0.5。左下：各类非猫图片被误判为猫的比例，狗遥遥领先。"
            "右下：混淆矩阵，右上角是误报（不是猫却判成猫），左下角是漏检（是猫却没认出来）。",
    },
    explain=explain,
)
