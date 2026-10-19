from __future__ import annotations

LESSONS = [
    {
        "id": "01",
        "keys": ("01", "1", "perceptron", "01_perceptron"),
        "title": "单层感知机",
        "module": "lessons.01_perceptron",
        "doc": "01_perceptron.html",
        "help": "线性分类：AND / OR 能学会，XOR 学不会",
    },
    {
        "id": "02",
        "keys": ("02", "2", "mlp", "02_mlp"),
        "title": "多层感知机",
        "module": "lessons.02_mlp",
        "doc": "02_mlp.html",
        "help": "用隐藏层解决 XOR 和月牙形分类",
    },
    {
        "id": "03-visual",
        "keys": ("03-visual", "03v", "backprop-visual", "03_backprop_visual"),
        "title": "反向传播（数值走一遍）",
        "module": "lessons.03_backprop_visual",
        "doc": "03_backprop.html",
        "help": "用一组固定数字手算前向 / 反向",
    },
    {
        "id": "03",
        "keys": ("03", "3", "backprop", "03_backpropagation"),
        "title": "反向传播（训练 + 梯度检查）",
        "module": "lessons.03_backpropagation",
        "doc": "03_backprop.html",
        "help": "手写梯度，并用数值梯度核对",
    },
    {
        "id": "04",
        "keys": ("04", "4", "activations", "04_activations"),
        "title": "激活函数",
        "module": "lessons.04_activations",
        "doc": "04_activations.html",
        "help": "常见激活函数、导数与梯度消失",
    },
    {
        "id": "05",
        "keys": ("05", "5", "lr", "learning-rate", "learning_rate", "05_learning_rate"),
        "title": "学习率",
        "module": "lessons.05_learning_rate",
        "doc": "05_learning_rate.html",
        "help": "对比太小 / 合适 / 太大的学习率对训练的影响",
    },
    {
        "id": "06",
        "keys": ("06", "6", "init", "weight-init", "weight_init", "initialization", "06_weight_init"),
        "title": "参数初始化",
        "module": "lessons.06_weight_init",
        "doc": "06_weight_init.html",
        "help": "对比全零 / 过小 / 过大 / He 初始化对训练的影响",
    },
    {
        "id": "07",
        "keys": ("07", "7", "norm", "normalization", "batchnorm", "bn", "07_normalization"),
        "title": "归一化",
        "module": "lessons.07_normalization",
        "doc": "07_normalization.html",
        "help": "对比无归一化 / 输入标准化 / BatchNorm",
    },
    {
        "id": "08",
        "keys": ("08", "8", "reg", "regularization", "l2", "dropout", "08_regularization"),
        "title": "正则化",
        "module": "lessons.08_regularization",
        "doc": "08_regularization.html",
        "help": "对比 L1/L2/Dropout/组合/早停/过强 L2 等多种正则",
    },
    {
        "id": "09",
        "keys": ("09", "9", "pipeline", "full-pipeline", "capstone", "mnist", "09_full_pipeline"),
        "title": "综合实战：MNIST 数字识别",
        "module": "lessons.09_full_pipeline",
        "doc": "09_full_pipeline.html",
        "help": "集齐 11 项核心技术（He/Xavier、GELU、RMSNorm/BN、CE、AdamW、Warmup+Cosine、早停等）实现 MNIST 手写数字识别",
    },
    {
        "id": "10",
        "keys": ("10", "cnn", "conv", "convolution", "cat", "10_cnn"),
        "title": "卷积神经网络：检测小猫",
        "module": "lessons.10_cnn",
        "doc": "10_cnn.html",
        "help": "手写 CNN 判断彩色照片里有没有猫：对比 MLP 与数据增强，看检出率、误报和遮挡热力图",
    },
    {
        "id": "11",
        "keys": ("11", "res", "resnet", "residual", "skip", "11_residual"),
        "title": "残差连接",
        "module": "lessons.11_residual",
        "doc": "11_residual.html",
        "help": "普通深层网络越深越难训练；h + F(h) 的捷径让 50 层网络也能学会",
    },
    {
        "id": "12",
        "keys": ("12", "rnn", "lstm", "gru", "recurrent", "sequence", "12_rnn"),
        "title": "循环神经网络 RNN / LSTM / GRU",
        "module": "lessons.12_rnn",
        "doc": "12_rnn.html",
        "help": "手写随时间反向传播：RNN 记不住 40 步前的信息，LSTM 和 GRU 可以；再按字符生成新名字",
    },
    {
        "id": "13",
        "keys": ("13", "seq2seq", "encoder", "decoder", "13_seq2seq"),
        "title": "编码器–解码器 Seq2Seq",
        "module": "lessons.13_seq2seq",
        "doc": "13_seq2seq.html",
        "help": "编码器读完源序列，解码器逐符号生成倒序；固定向量装不下长序列，注意力可以",
    },
    {
        "id": "14",
        "keys": ("14", "attention", "qkv", "dot-product", "self-attention", "14_attention"),
        "title": "注意力机制",
        "module": "lessons.14_attention",
        "doc": "14_attention.html",
        "help": "Query / Key / Value 缩放点积注意力按键查值；平均池化在记忆变长后失败",
    },
]


for _item in LESSONS:
    # Run / summary name, e.g. "03_backprop_visual" (also the lesson file stem).
    _item["name"] = _item["module"].split(".", 1)[1]


def find_lesson(key: str) -> dict:
    needle = key.strip().lower()
    for item in LESSONS:
        if needle == item["id"].lower() or needle in item["keys"]:
            return item
    known = ", ".join(item["id"] for item in LESSONS)
    raise KeyError(f"未知实验 {key!r}。可选：{known}")
