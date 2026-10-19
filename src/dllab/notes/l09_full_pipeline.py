from __future__ import annotations

from dllab.notes.base import LessonNotes, LogNote, finding, pct


def explain(m: dict) -> list[dict]:
    out = []
    if m.get("data_source") != "mnist_npz":
        out.append(finding("数据源不是真实 MNIST",
                           f"data_source = {m.get('data_source')}：没有找到 data/mnist/mnist.npz，使用了内置合成数字，结果仅供参考。",
                           "warn"))
    p = {r["pipeline"]: r for r in m.get("pipelines", [])}
    naive, mbn, tuned = p.get("naive"), p.get("momentum_bn"), p.get("tuned")
    if naive:
        out.append(finding(
            f"naive：测试准确率只有 {pct(naive['test_acc'])}",
            f"large 初始化 + Sigmoid + MSE + 固定学习率 SGD，没有归一化和正则，跑满 {naive['epochs']} 轮。"
            f"权重范数 ‖W‖ = {naive['weight_norm']:.0f}：过大的初始权重让 sigmoid 一开始就饱和，梯度极小；"
            "MSE 配 Softmax 时，预测越错梯度反而可能越小，进一步拖慢学习。10 分类随机猜是 10%，它只比随机好一点。",
            "warn",
        ))
    if mbn:
        out.append(finding(
            f"momentum_bn：测试准确率 {pct(mbn['test_acc'])}，第 {mbn['epochs']} 轮最佳",
            "换成 He 初始化 + ReLU + BatchNorm + 交叉熵 + SGD+Momentum + Warmup/Cosine，第 1 轮验证准确率就超过 80%。"
            "动量把连续几步的梯度方向累积起来，在一致的方向上加速、在来回震荡的方向上相互抵消。",
            "ok",
        ))
    if tuned:
        out.append(finding(
            f"tuned：测试准确率 {pct(tuned['test_acc'])}，第 {tuned['epochs']} 轮最佳",
            "He/Xavier + GELU + RMSNorm + 交叉熵 + AdamW（解耦权重衰减）+ Warmup/Cosine + Dropout/L2 + 早停。"
            "AdamW 为每个参数单独调整步长，并把权重衰减从梯度里拆出来直接作用在权重上，比在损失里加 L2 更稳定。",
            "ok",
        ))
    if naive and tuned:
        out.append(finding(
            f"差距：{tuned['test_acc'] - naive['test_acc']:.1f} 个百分点",
            "三条流水线用的是同一份数据、同一个网络结构 [784, 128, 64, 10]，差别全部来自训练配方。"
            "momentum_bn 和 tuned 结果接近，说明只要初始化、归一化、损失函数、优化器这几个关键环节都选对，具体用哪一种变体影响不大。"
            "两者训练准确率都接近 100%、测试在 92% 左右，还有约 7 个百分点的泛化差距——这是 MLP 处理图像的天花板之一。第 10 课换成更难的彩色照片（找猫），会看到 CNN 的优势更明显。",
        ))
    return out


NOTES = LessonNotes(
    intro="在 MNIST 手写数字（10 分类）上，用同一个 [784, 128, 64, 10] 网络对比三条训练流水线：故意踩坑的 naive、经典的 SGD+Momentum+BN、"
          "以及把 11 项核心技术装齐的 tuned。",
    steps=[
        "加载 MNIST（data/mnist/mnist.npz，抽取 4000 / 800 / 1200）",
        "naive：large init + Sigmoid + MSE + SGD",
        "momentum_bn：He + ReLU + BN + CE + Momentum",
        "tuned：He + GELU + RMSNorm + CE + AdamW",
        "画曲线、混淆矩阵、样本预测和知识清单",
    ],
    log_notes=[
        LogNote(r"数据源: (?P<src>\w+)", "数据：{src}",
                "28×28 灰度图拉平成 784 维向量。训练集 4000 张用于更新参数，验证集 800 张用于早停和模型选择，"
                "测试集 1200 张只在最后评估一次。naive 只把像素除以 255；另外两条做逐像素标准化。", step=0),
        LogNote(r"总结：核心知识清单", "11 项核心技术",
                "下表列出本课用到的环节、具体技术和它解决的问题。三条流水线的区别就是在这些环节上做了不同选择。"),
        LogNote(r"--- \[naive\]", "naive 流水线",
                "故意选择一组“坏”配置：大初始化让 sigmoid 饱和、MSE 不适合分类、固定学习率 SGD、不做输入标准化。预期学得很慢。", step=1, level="warn"),
        LogNote(r"--- \[momentum_bn\]", "momentum_bn 流水线",
                "经典组合：He 初始化 + ReLU + BatchNorm + 交叉熵 + SGD+Momentum(0.9) + Warmup/Cosine + L1/L2 + Dropout + 早停。", step=2),
        LogNote(r"--- \[tuned\]", "tuned 流水线",
                "现代组合：GELU 激活 + RMSNorm + AdamW（weight_decay = 0.03）+ Warmup 5 轮后 Cosine 退火 + Dropout 0.22 + 早停。", step=3),
        LogNote(r"→ train=(?P<tr>[\d.]+)% \| val=(?P<va>[\d.]+)% \| test=(?P<te>[\d.]+)%", "本条流水线：test {te}%",
                "train / val / test 分别是训练、验证、测试准确率。best_epoch 是早停选中的轮次；‖W‖ 是所有权重的 L2 范数。",
                once=False),
        LogNote(r"汇总对比", "汇总", "三条流水线结果对比，接着画图。", step=4),
    ],
    image_notes={
        "pipeline_curves.png":
            "左：训练损失。注意 naive 用的是 MSE，另外两条是交叉熵，数值不能直接比较大小，只看下降趋势。"
            "中：验证准确率——momentum_bn 和 tuned 几轮内突破 90%，naive 缓慢爬升到 30% 左右。"
            "右：学习率（对数坐标），可以看到 Warmup 的上升段和 Cosine 的下降段；naive 是一条水平线。",
        "pipeline_metrics.png":
            "左：三条流水线的 train / val / test 准确率柱状图。右：tuned 在测试集上的混淆矩阵，行是真实数字、列是预测数字，"
            "对角线是预测正确的数量；非对角线上较大的数字说明哪些数字容易混淆（常见是 4↔9、3↔5、7↔9）。",
        "pipeline_boundaries.png":
            "16 张测试图片。标题第一行：T 是真实标签，P 是 tuned 的预测和置信度；第二行是 naive 的预测。"
            "绿色标题表示 tuned 预测正确，红色表示错误。",
        "pipeline_recipe.png": "11 项核心技术的对照卡片，底部是三条流水线的测试准确率。",
    },
    explain=explain,
)
