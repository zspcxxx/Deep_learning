from __future__ import annotations

LESSONS = [
    {
        "id": "01",
        "keys": ("01", "1", "perceptron", "01_perceptron"),
        "title": "单层感知机",
        "module": "lessons.01_perceptron",
        "doc": "01_perceptron.html",
        "needs_torch": False,
        "help": "线性分类：AND / OR 能学会，XOR 学不会",
    },
    {
        "id": "02",
        "keys": ("02", "2", "mlp", "02_mlp"),
        "title": "多层感知机",
        "module": "lessons.02_mlp",
        "doc": "02_mlp.html",
        "needs_torch": False,
        "help": "用隐藏层解决 XOR 和月牙形分类",
    },
    {
        "id": "03-visual",
        "keys": ("03-visual", "03v", "backprop-visual", "03_backprop_visual"),
        "title": "反向传播（数值走一遍）",
        "module": "lessons.03_backprop_visual",
        "doc": "03_backprop.html",
        "needs_torch": False,
        "help": "用一组固定数字手算前向 / 反向",
    },
    {
        "id": "03",
        "keys": ("03", "3", "backprop", "03_backpropagation"),
        "title": "反向传播（训练 + 梯度检查）",
        "module": "lessons.03_backpropagation",
        "doc": "03_backprop.html",
        "needs_torch": False,
        "help": "手写梯度，并用数值梯度核对",
    },
    {
        "id": "04",
        "keys": ("04", "4", "activations", "04_activations"),
        "title": "激活函数",
        "module": "lessons.04_activations",
        "doc": "04_activations.html",
        "needs_torch": False,
        "help": "常见激活函数、导数与梯度消失",
    },
    {
        "id": "05",
        "keys": ("05", "5", "lr", "learning-rate", "learning_rate", "05_learning_rate"),
        "title": "学习率",
        "module": "lessons.05_learning_rate",
        "doc": "05_learning_rate.html",
        "needs_torch": False,
        "help": "对比太小 / 合适 / 太大的学习率对训练的影响",
    },
    {
        "id": "06",
        "keys": ("06", "6", "mnist", "06_mnist"),
        "title": "MNIST 手写数字",
        "module": "lessons.06_mnist",
        "doc": "06_mnist.html",
        "needs_torch": True,
        "help": "PyTorch 训练 MLP 与 CNN",
    },
]


def find_lesson(key: str) -> dict:
    needle = key.strip().lower()
    for item in LESSONS:
        if needle == item["id"].lower() or needle in item["keys"]:
            return item
    known = ", ".join(item["id"] for item in LESSONS)
    raise KeyError(f"未知实验 {key!r}。可选：{known}")
