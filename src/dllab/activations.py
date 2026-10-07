from __future__ import annotations

import numpy as np


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))


def sigmoid_grad(z):
    s = sigmoid(z)
    return s * (1 - s)


def tanh(z):
    return np.tanh(z)


def tanh_grad(z):
    return 1.0 - np.tanh(z) ** 2


def hard_sigmoid(z):
    return np.clip(z / 6.0 + 0.5, 0.0, 1.0)


def hard_sigmoid_grad(z):
    return np.where((z > -3) & (z < 3), 1.0 / 6.0, 0.0)


def logsigmoid(z):
    return -np.log1p(np.exp(-np.abs(z))) + np.minimum(z, 0)


def logsigmoid_grad(z):
    return sigmoid(-z)


def tanhshrink(z):
    return z - np.tanh(z)


def tanhshrink_grad(z):
    return np.tanh(z) ** 2


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return (z > 0).astype(float)


def leaky_relu(z, alpha=0.01):
    return np.where(z > 0, z, alpha * z)


def leaky_relu_grad(z, alpha=0.01):
    return np.where(z > 0, 1.0, alpha)


def prelu(z, alpha=0.25):
    return np.where(z > 0, z, alpha * z)


def prelu_grad(z, alpha=0.25):
    return np.where(z > 0, 1.0, alpha)


def rrelu(z, alpha_low=0.125, alpha_high=1.0 / 3.0, seed=0):
    rng = np.random.RandomState(seed)
    alpha = rng.uniform(alpha_low, alpha_high, size=z.shape)
    return np.where(z > 0, z, alpha * z)


def rrelu_grad(z, seed=0):
    alpha = (0.125 + 1.0 / 3.0) / 2.0
    return np.where(z > 0, 1.0, alpha)


def elu(z, alpha=1.0):
    return np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1))


def elu_grad(z, alpha=1.0):
    return np.where(z > 0, 1.0, alpha * np.exp(np.clip(z, -500, 0)))


def selu(z, alpha=1.67326, scale=1.0507):
    return scale * np.where(z > 0, z, alpha * (np.exp(np.clip(z, -500, 0)) - 1))


def selu_grad(z, alpha=1.67326, scale=1.0507):
    return scale * np.where(z > 0, 1.0, alpha * np.exp(np.clip(z, -500, 0)))


def gelu(z):
    return 0.5 * z * (1 + np.tanh(np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)))


def gelu_grad(z):
    inner = np.sqrt(2 / np.pi) * (z + 0.044715 * z ** 3)
    cdf = 0.5 * (1 + np.tanh(inner))
    pdf = 0.5 * z * (1 - np.tanh(inner) ** 2) * np.sqrt(2 / np.pi) * (1 + 3 * 0.044715 * z ** 2)
    return cdf + pdf


def silu(z):
    return z * sigmoid(z)


def silu_grad(z):
    s = sigmoid(z)
    return s + z * s * (1 - s)


def swish(z, beta=1.0):
    return z * sigmoid(beta * z)


def swish_grad(z, beta=1.0):
    s = sigmoid(beta * z)
    return s + beta * z * s * (1 - s)


def mish(z):
    sp = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
    return z * np.tanh(sp)


def mish_grad(z):
    sp = np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)
    tanh_sp = np.tanh(sp)
    sig_sp = 1.0 / (1.0 + np.exp(-sp))
    return tanh_sp + z * (1 - tanh_sp ** 2) * sig_sp


def softplus(z):
    return np.log1p(np.exp(-np.abs(z))) + np.maximum(z, 0)


def softplus_grad(z):
    return sigmoid(z)


def hardswish(z):
    return z * np.clip(z + 3, 0, 6) / 6


def hardswish_grad(z):
    return np.where((z <= -3) | (z >= 3), 0.0, np.where(z > 0, (2 * z + 3) / 6, (z + 3) / 3))


def hardtanh(z):
    return np.clip(z, -1, 1)


def hardtanh_grad(z):
    return ((z > -1) & (z < 1)).astype(float)


def step(z):
    return (z >= 0).astype(float)


def sign(z):
    return np.sign(z)


def linear(z):
    return z


def linear_grad(z):
    return np.ones_like(z)


def softmax(z):
    z = z - np.max(z, axis=-1, keepdims=True)
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=-1, keepdims=True)


def maxout(z1, z2):
    return np.maximum(z1, z2)


PLOT_FUNCTIONS = [
    ("Sigmoid", sigmoid, sigmoid_grad),
    ("Tanh", tanh, tanh_grad),
    ("Hard Sigmoid", hard_sigmoid, hard_sigmoid_grad),
    ("LogSigmoid", logsigmoid, logsigmoid_grad),
    ("ReLU", relu, relu_grad),
    ("Leaky ReLU", leaky_relu, leaky_relu_grad),
    ("PReLU", prelu, prelu_grad),
    ("ELU", elu, elu_grad),
    ("SELU", selu, selu_grad),
    ("GELU", gelu, gelu_grad),
    ("SiLU / Swish", silu, silu_grad),
    ("Mish", mish, mish_grad),
    ("Softplus", softplus, softplus_grad),
    ("HardSwish", hardswish, hardswish_grad),
    ("HardTanh", hardtanh, hardtanh_grad),
    ("Step", step, lambda z: np.zeros_like(z)),
]

TABLE_FUNCTIONS = [
    ("Sigmoid", sigmoid, sigmoid_grad),
    ("Tanh", tanh, tanh_grad),
    ("HardSigmoid", hard_sigmoid, hard_sigmoid_grad),
    ("LogSigmoid", logsigmoid, logsigmoid_grad),
    ("ReLU", relu, relu_grad),
    ("LeakyReLU", lambda z: leaky_relu(z, 0.1), lambda z: leaky_relu_grad(z, 0.1)),
    ("PReLU", lambda z: prelu(z, 0.25), lambda z: prelu_grad(z, 0.25)),
    ("ELU", elu, elu_grad),
    ("SELU", selu, selu_grad),
    ("GELU", gelu, gelu_grad),
    ("SiLU/Swish", silu, silu_grad),
    ("Mish", mish, mish_grad),
    ("Softplus", softplus, softplus_grad),
    ("HardSwish", hardswish, hardswish_grad),
    ("HardTanh", hardtanh, hardtanh_grad),
    ("Step", step, lambda z: np.zeros_like(z)),
]

SUMMARY_ROWS = [
    ("Sigmoid", "(0,1)", "二分类输出", "梯度消失、非零中心"),
    ("Tanh", "(-1,1)", "RNN 隐藏层", "梯度消失"),
    ("HardSigmoid", "(0,1)", "移动端", "近似精度低"),
    ("LogSigmoid", "(-inf,0)", "损失函数", "单独使用较少"),
    ("ReLU", "[0,+inf)", "默认隐藏层", "死亡 ReLU"),
    ("LeakyReLU", "(-inf,+inf)", "ReLU 替代", "alpha 需调"),
    ("PReLU", "(-inf,+inf)", "深层网络", "增加参数"),
    ("RReLU", "(-inf,+inf)", "防止过拟合", "随机性"),
    ("ELU", "(-1,+inf)", "深层网络", "计算稍慢"),
    ("SELU", "(-1.7,+inf)", "自归一化网络", "参数固定"),
    ("GELU", "(-0.17,+inf)", "Transformer", "计算稍慢"),
    ("SiLU/Swish", "(-0.28,+inf)", "深层网络", "计算稍慢"),
    ("Mish", "(-0.31,+inf)", "YOLOv4", "计算慢"),
    ("Softplus", "(0,+inf)", "ReLU 平滑版", "无上界"),
    ("HardSwish", "(-0.375,+inf)", "MobileNetV3", "近似精度低"),
    ("HardTanh", "[-1,1]", "裁剪输出", "梯度为 0 区域多"),
    ("Step", "{0,1}", "感知机", "不可导"),
    ("Sign", "{-1,0,+1}", "理论模型", "不可导"),
    ("Linear", "(-inf,+inf)", "回归输出", "无非线性"),
    ("Softmax", "(0,1) 和=1", "多分类输出", "仅用于输出层"),
]
