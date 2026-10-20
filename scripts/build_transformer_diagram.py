#!/usr/bin/env python3
"""Draw docs/16_transformer_architecture.svg: the full Transformer with every technique it uses.

Main area: the original encoder–decoder (data flows bottom → top, as in Vaswani et al. 2017),
each box tagged with the lesson of this project that builds it. Right column: zoom-ins on
multi-head attention, masks, Post-LN vs Pre-LN, FFN / norm variants, position encodings, and
the training / inference recipe. Run again after editing; the SVG is committed.
"""
from __future__ import annotations

from html import escape
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "docs" / "16_transformer_architecture.svg"
W, H = 1720, 1630
FONT = "Noto Sans SC, Noto Sans CJK SC, PingFang SC, Microsoft YaHei, sans-serif"
MONO = "JetBrains Mono, Noto Sans Mono CJK SC, Menlo, Consolas, monospace"

COLORS = {
    "io": ("#f8fafc", "#64748b"),
    "embed": ("#dbeafe", "#2563eb"),
    "pos": ("#e0e7ff", "#4f46e5"),
    "attn": ("#ccfbf1", "#0f766e"),
    "norm": ("#fef3c7", "#b45309"),
    "ffn": ("#dcfce7", "#15803d"),
    "drop": ("#f1f5f9", "#94a3b8"),
    "out": ("#e2e8f0", "#334155"),
    "train": ("#fff7ed", "#c2410c"),
    "infer": ("#fae8ff", "#a21caf"),
    "note": ("#ffffff", "#cbd5e1"),
}

svg: list[str] = []


def text(x, y, s, size=13, anchor="middle", weight="normal", color="#1a2332", font=FONT, italic=False):
    style = ' font-style="italic"' if italic else ""
    svg.append(f'<text x="{x}" y="{y}" font-size="{size}" text-anchor="{anchor}" font-weight="{weight}" '
               f'fill="{color}" font-family="{font}" style="white-space:pre"{style}>{escape(s)}</text>')


def tags(x_right, y_top, lessons):
    """Lesson pills sitting on the top-right border of a box."""
    x = x_right - 6
    for lesson in reversed(lessons):
        w = 8 + 7.2 * len(lesson)
        x -= w
        svg.append(f'<rect x="{x:.1f}" y="{y_top - 9}" width="{w:.1f}" height="17" rx="8.5" fill="#1f6feb"/>')
        text(x + w / 2, y_top + 3.5, lesson, size=10.5, weight="bold", color="#ffffff", font=MONO)
        x -= 4


def box(x, y, w, h, title, sub=None, kind="note", lessons=(), size=13.5, sub_size=11.5, dashed=False):
    fill, stroke = COLORS[kind]
    dash = ' stroke-dasharray="5 4"' if dashed else ""
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="9" fill="{fill}" stroke="{stroke}" '
               f'stroke-width="1.4"{dash}/>')
    cx = x + w / 2
    if sub:
        text(cx, y + h / 2 - 3, title, size=size, weight="bold")
        text(cx, y + h / 2 + 13, sub, size=sub_size, color="#475569")
    else:
        text(cx, y + h / 2 + 5, title, size=size, weight="bold")
    if lessons:
        tags(x + w, y, lessons)


def arrow(x1, y1, x2, y2, color="#334155", width=1.6, marker="a"):
    svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}" '
               f'marker-end="url(#{marker})"/>')


def path(d, color="#334155", width=1.6, marker="a", dashed=False):
    dash = ' stroke-dasharray="6 4"' if dashed else ""
    svg.append(f'<path d="{d}" fill="none" stroke="{color}" stroke-width="{width}"{dash} marker-end="url(#{marker})"/>')


def frame(x, y, w, h, label=None, color="#64748b", fill="none", label_size=13):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{fill}" stroke="{color}" '
               f'stroke-width="1.3" stroke-dasharray="7 5"/>')
    if label:
        text(x + 12, y - 8, label, size=label_size, anchor="start", weight="bold", color="#334155")


def panel(x, y, w, h, title):
    svg.append(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#ffffff" stroke="#cbd5e1" stroke-width="1.3"/>')
    text(x + 16, y + 26, title, size=15, anchor="start", weight="bold", color="#1f2937")


def lines(x, y, rows, size=12.5, gap=21, color="#1f2937"):
    for i, row in enumerate(rows):
        text(x, y + i * gap, row, size=size, anchor="start", color=color)


def plus(cx, cy, r=13):
    svg.append(f'<circle cx="{cx}" cy="{cy}" r="{r}" fill="#ffffff" stroke="#334155" stroke-width="1.6"/>')
    svg.append(f'<line x1="{cx - 7}" y1="{cy}" x2="{cx + 7}" y2="{cy}" stroke="#334155" stroke-width="1.8"/>')
    svg.append(f'<line x1="{cx}" y1="{cy - 7}" x2="{cx}" y2="{cy + 7}" stroke="#334155" stroke-width="1.8"/>')


# ---------------------------------------------------------------- header

def header():
    svg.append(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
               f'xml:space="preserve" role="img" aria-label="Transformer 详细技术架构图">')
    svg.append('<defs>'
               '<marker id="a" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">'
               '<path d="M0,0 L7,3.5 L0,7 Z" fill="#334155"/></marker>'
               '<marker id="ab" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">'
               '<path d="M0,0 L7,3.5 L0,7 Z" fill="#1f6feb"/></marker>'
               '<marker id="ar" markerWidth="9" markerHeight="9" refX="7" refY="3.5" orient="auto">'
               '<path d="M0,0 L7,3.5 L0,7 Z" fill="#7c3aed"/></marker>'
               '</defs>')
    svg.append(f'<rect width="{W}" height="{H}" fill="#f8fafc"/>')
    text(30, 48, "Transformer 详细技术架构图", size=26, anchor="start", weight="bold")
    text(30, 78, "主体：原论文的编码器–解码器（Vaswani et al., 2017，数据自下而上流动）；右侧：各模块内部与现代变体。"
                 "蓝色标签 Lxx = 本项目第 xx 课。", size=14, anchor="start", color="#475569")


# ---------------------------------------------------------------- top-left info boxes

def overview():
    frame(115, 125, 870, 170, "三种架构家族（同一套零件的不同拼法）")
    fams = [
        ("Encoder-only：BERT、RoBERTa", ["只用左侧编码器栈", "双向自注意力（只有 padding mask）", "用途：分类、抽取、表示学习"]),
        ("Decoder-only：GPT、LLaMA", ["只用右侧解码器栈，去掉交叉注意力", "因果掩码，逐个预测下一个 token", "用途：语言模型、对话、代码"]),
        ("Encoder–decoder：原论文、T5", ["左右两栈都用，交叉注意力相连", "编码器读全句，解码器逐个生成", "用途：翻译、摘要（L13 的同一思路）"]),
    ]
    for i, (title, rows) in enumerate(fams):
        x = 130 + i * 285
        svg.append(f'<rect x="{x}" y="145" width="270" height="135" rx="10" fill="#ffffff" stroke="#cbd5e1"/>')
        text(x + 14, 170, title, size=13.5, anchor="start", weight="bold", color="#0f766e")
        lines(x + 14, 196, rows, size=12, gap=24, color="#334155")

    panel(115, 315, 430, 195, "张量形状速查")
    lines(135, 368, [
        "B  批大小            S  源序列长度     T  目标序列长度",
        "V  词表大小          d = d_model（原论文 512）",
        "h  头数（8）          d_k = d_v = d / h（64）",
        "d_ff  前馈层宽度（2048 ≈ 4d）",
        "注意力分数 (B, h, T, S)；输出 logits (B, T, V)",
    ], size=12.5, gap=26)

    panel(565, 315, 420, 195, "图例")
    box(585, 355, 120, 30, "数据流", kind="out", size=12)
    arrow(715, 370, 770, 370)
    svg.append('<path d="M585,420 H700" fill="none" stroke="#7c3aed" stroke-width="1.8" stroke-dasharray="6 4" marker-end="url(#ar)"/>')
    text(715, 425, "残差捷径 x + Sublayer(x)（L11）", size=12.5, anchor="start")
    svg.append('<path d="M585,460 H700" fill="none" stroke="#1f6feb" stroke-width="2.2" marker-end="url(#ab)"/>')
    text(715, 465, "编码器输出 memory → 交叉注意力的 K、V", size=12.5, anchor="start")
    tags(632, 495, ["L14"])
    text(640, 499, "= 这个零件在本项目第 14 课里亲手实现过", size=12.5, anchor="start")

    panel(135, 555, 330, 205, "原论文 base / big 配置")
    lines(155, 605, [
        "base：N=6, d=512, h=8, d_ff=2048",
        "      dropout 0.1，约 6500 万参数",
        "big：N=6, d=1024, h=16, d_ff=4096",
        "      dropout 0.3，约 2.13 亿参数",
        "本项目 L15：N=4, d=32, h=4, d_ff=64",
        "      pre-LN，约 3.5 万参数",
    ], size=12.5, gap=24)


# ---------------------------------------------------------------- encoder

def encoder():
    x0, w, cx = 135, 330, 300
    box(x0, 1480, w, 44, "源序列 x₁ … x_S", "token id，形状 (B, S)", kind="io")
    arrow(cx, 1480, cx, 1456)
    box(x0, 1410, w, 46, "分词 Tokenizer", "BPE / WordPiece：文本 → 子词 id", kind="io")
    arrow(cx, 1410, cx, 1388)
    box(x0, 1340, w, 48, "Input Embedding", "查表 E（V×d 矩阵），再乘 √d → (B, S, d)", kind="embed", lessons=["L14", "L15"])
    arrow(cx, 1340, cx, 1315)
    plus(cx, 1300)
    box(20, 1272, 100, 56, "位置编码", "sin / cos", kind="pos", lessons=["L14"])
    arrow(120, 1300, cx - 14, 1300)
    arrow(cx, 1287, cx, 1276)
    box(x0, 1232, w, 44, "Dropout p = 0.1", kind="drop", lessons=["L08"])
    arrow(cx, 1232, cx, 1172)

    frame(115, 860, 380, 345, "编码器层 × N（原论文 N = 6）")
    box(x0, 1118, w, 54, "多头自注意力 Multi-Head Self-Attention", "h 个头并行，padding mask 遮掉补齐位置", kind="attn",
        lessons=["L13", "L14"])
    arrow(cx, 1118, cx, 1096)
    box(x0, 1052, w, 44, "Dropout → Add & Norm", "LayerNorm(x + Sublayer(x))", kind="norm", lessons=["L07", "L08"])
    arrow(cx, 1052, cx, 1012)
    box(x0, 952, w, 60, "逐位置前馈网络 FFN", "Linear d→4d · ReLU / GELU · Linear 4d→d", kind="ffn",
        lessons=["L02", "L04", "L10"])
    arrow(cx, 952, cx, 926)
    box(x0, 882, w, 44, "Dropout → Add & Norm", "LayerNorm(x + Sublayer(x))", kind="norm", lessons=["L07", "L08"])
    path(f"M{cx},1196 H480 V1074 H{x0 + w + 2}", color="#7c3aed", width=1.8, marker="ar", dashed=True)
    path(f"M{cx},1032 H480 V904 H{x0 + w + 2}", color="#7c3aed", width=1.8, marker="ar", dashed=True)
    text(502, 1160, "残差", size=11.5, anchor="start", color="#7c3aed")
    tags(552, 1135, ["L11"])
    arrow(cx, 882, cx, 816)
    box(x0, 770, w, 46, "编码器输出 memory", "(B, S, d)，每个源位置一个上下文向量", kind="out")
    path(f"M{x0 + w},793 H560 V1036 H623", color="#1f6feb", width=2.4, marker="ab")
    text(568, 930, "K、V", size=12.5, anchor="start", weight="bold", color="#1f6feb")


# ---------------------------------------------------------------- decoder

def decoder():
    x0, w, cx = 625, 330, 790
    box(x0, 1480, w, 44, "目标序列右移一位：<bos> y₁ … y_(T−1)", "训练时喂真值（teacher forcing），形状 (B, T)", kind="io",
        lessons=["L13"], size=13)
    arrow(cx, 1480, cx, 1458)
    box(x0, 1410, w, 48, "Output Embedding", "与输入嵌入、输出层共享权重（weight tying）", kind="embed", lessons=["L14"])
    arrow(cx, 1410, cx, 1385)
    plus(cx, 1370)
    box(500, 1342, 100, 56, "位置编码", "sin / cos", kind="pos", lessons=["L14"])
    arrow(600, 1370, cx - 14, 1370)
    arrow(cx, 1357, cx, 1344)
    box(x0, 1300, w, 44, "Dropout p = 0.1", kind="drop", lessons=["L08"])
    arrow(cx, 1300, cx, 1250)

    frame(605, 770, 380, 505, "解码器层 × N")
    box(x0, 1196, w, 54, "带掩码的多头自注意力", "因果掩码：位置 t 只能看 ≤ t 的位置", kind="attn", lessons=["L13", "L14"])
    arrow(cx, 1196, cx, 1174)
    box(x0, 1130, w, 44, "Dropout → Add & Norm", kind="norm", lessons=["L07", "L11"])
    arrow(cx, 1130, cx, 1064)
    box(x0, 1010, w, 54, "交叉注意力 Cross-Attention", "Q ← 解码器，K / V ← 编码器 memory", kind="attn", lessons=["L13"])
    arrow(cx, 1010, cx, 988)
    box(x0, 944, w, 44, "Dropout → Add & Norm", kind="norm", lessons=["L07", "L11"])
    arrow(cx, 944, cx, 912)
    box(x0, 852, w, 60, "逐位置前馈网络 FFN", "Linear d→4d · ReLU / GELU · Linear 4d→d", kind="ffn",
        lessons=["L02", "L04"])
    arrow(cx, 852, cx, 836)
    box(x0, 792, w, 44, "Dropout → Add & Norm", kind="norm", lessons=["L07", "L11"])
    for y_from, y_to in ((1274, 1152), (1097, 966), (928, 814)):
        path(f"M{cx},{y_from} H972 V{y_to} H{x0 + w + 2}", color="#7c3aed", width=1.8, marker="ar", dashed=True)

    arrow(cx, 792, cx, 746)
    box(x0, 700, w, 46, "Linear d → V", "得到每个位置的 logits (B, T, V)", kind="out", lessons=["L01", "L09"])
    arrow(cx, 700, cx, 676)
    box(x0, 630, w, 46, "Softmax", "P(y_t | y_<t, x)：每个位置一个词表分布", kind="out", lessons=["L09"])
    arrow(cx - 85, 630, cx - 85, 600)
    arrow(cx + 85, 630, cx + 85, 600)
    box(x0, 530, 160, 70, "训练", "交叉熵 + label smoothing", kind="train", lessons=["L09"], size=13.5, sub_size=11)
    text(x0 + 80, 593, "所有位置并行算损失", size=10.5, color="#9a3412")
    box(x0 + 170, 530, 160, 70, "推理", "自回归逐个生成", kind="infer", lessons=["L12"], size=13.5, sub_size=11)
    text(x0 + 250, 593, "greedy / beam / top-k", size=10.5, color="#86198f")


# ---------------------------------------------------------------- right column

def attention_zoom():
    panel(1040, 110, 660, 700, "① 多头注意力内部（编码器、解码器里的每个注意力框都是这样）")
    box(1090, 148, 560, 44, "输入 X：(B, T, d)", "自注意力 Q、K、V 都来自 X；交叉注意力 Q 来自解码器，K、V 来自 memory",
        kind="io", size=13, sub_size=11)
    for i, (name, sub) in enumerate((("Q = X W_Q", "查询：我要找什么"), ("K = X W_K", "键：我能被怎样找到"),
                                     ("V = X W_V", "值：被取走的内容"))):
        x = 1095 + i * 190
        arrow(x + 85, 192, x + 85, 220)
        box(x, 220, 170, 48, name, sub, kind="embed", size=13, sub_size=11, lessons=["L14"] if i == 0 else ())
        arrow(x + 85, 268, x + 85, 292)
    box(1095, 292, 550, 38, "切成 h 个头：(B, T, d) → (B, h, T, d_k)，每个头在 d_k 维子空间里独立做注意力", kind="pos", size=12.5)

    frame(1085, 362, 570, 318, None, color="#0f766e")
    text(1100, 384, "缩放点积注意力  Attention(Q, K, V) = softmax(Q Kᵀ / √d_k + M) V", size=13, anchor="start",
         weight="bold", color="#0f766e")
    steps = [
        ("MatMul：S = Q Kᵀ → (B, h, T, T)", "attn", ["L14"]),
        ("Scale：S / √d_k（防止 softmax 过尖）", "attn", ["L14"]),
        ("Mask（可选）：被遮位置加 −∞", "attn", ["L13"]),
        ("Softmax（沿 key 维）→ 权重 α，每行和为 1", "attn", ["L14"]),
        ("Dropout（attention dropout）", "drop", ["L08"]),
    ]
    arrow(1280, 330, 1280, 396)
    for i, (label, kind, les) in enumerate(steps):
        y = 396 + i * 48
        box(1115, y, 330, 36, label, kind=kind, size=12.5, lessons=les)
        if i < len(steps) - 1:
            arrow(1280, y + 36, 1280, y + 48)
    arrow(1280, 396 + 4 * 48 + 36, 1280, 636)
    path("M1560,330 V636", color="#334155")
    text(1568, 480, "V", size=13, anchor="start", weight="bold")
    box(1115, 636, 530, 36, "MatMul：α V → (B, h, T, d_k)", kind="attn", size=12.5)
    arrow(1380, 672, 1380, 700)
    box(1095, 700, 550, 36, "拼接各头 Concat → (B, T, d)", kind="pos", size=12.5)
    arrow(1380, 736, 1380, 756)
    box(1095, 756, 550, 36, "Linear W_O → 输出 (B, T, d)，交给 Dropout → Add & Norm", kind="embed", size=12.5)


def masks_zoom():
    panel(1040, 830, 330, 250, "② 两种掩码（M = 0 或 −∞）")
    cell = 22
    for g, (gx, title, masked) in enumerate(((1062, "padding mask", lambda r, c: c >= 3),
                                              (1222, "因果掩码", lambda r, c: c > r))):
        text(gx + 2.5 * cell, 885, title, size=12.5, weight="bold")
        for r in range(5):
            for c in range(5):
                fill = "#334155" if masked(r, c) else "#ccfbf1"
                svg.append(f'<rect x="{gx + c * cell}" y="{895 + r * cell}" width="{cell - 2}" height="{cell - 2}" '
                           f'rx="3" fill="{fill}"/>')
        text(gx + 2.5 * cell, 1022, "行 = query，列 = key", size=11, color="#475569")
    lines(1060, 1046, ["深色 = 加 −∞，softmax 后权重为 0", "padding：后 2 个位置是补齐的空位"], size=11.5, gap=17,
          color="#334155")


def norm_zoom():
    panel(1385, 830, 315, 250, "③ Post-LN 与 Pre-LN")
    for i, (cx, title, sub, order) in enumerate(((1462, "Post-LN", "原论文、BERT", ("Sublayer", "⊕", "LayerNorm")),
                                                  (1622, "Pre-LN", "GPT-2 起、L15", ("LayerNorm", "Sublayer", "⊕")))):
        text(cx, 880, title, size=13, weight="bold")
        text(cx, 897, sub, size=11, color="#475569")
        box(cx - 50, 1036, 100, 26, "x", kind="io", size=12)
        ys = [998, 960, 922]
        prev = 1036
        for label, y in zip(order, ys):
            arrow(cx, prev, cx, y + 26)
            if label == "⊕":
                plus(cx, y + 13, r=11)
            else:
                box(cx - 50, y, 100, 26, label, kind="norm" if label == "LayerNorm" else "attn", size=11.5)
            prev = y
        plus_y = ys[order.index("⊕")] + 13
        start_y = 1049
        path(f"M{cx - 50},{start_y} H{cx - 64} V{plus_y} H{cx - 13}", color="#7c3aed", width=1.5, marker="ar", dashed=True)
        arrow(cx, ys[-1], cx, 908)


def variants_zoom():
    panel(1040, 1100, 660, 160, "④ 前馈网络与归一化的变体")
    lines(1060, 1150, [
        "FFN(x) = W₂ φ(W₁x + b₁) + b₂，d_ff ≈ 4d；φ = ReLU（原论文）/ GELU（BERT、GPT）   L02 L04",
        "SwiGLU：FFN(x) = (Swish(x W₁) ⊙ x V) W₂，d_ff ≈ 8d/3（PaLM、LLaMA）",
        "LayerNorm：γ·(x − μ)/σ + β，按每个 token 的特征维归一化，不依赖批大小   L07",
        "RMSNorm：γ·x / RMS(x)，省掉减均值和 β，计算更省（T5、LLaMA）",
        "残差流：每个子层只“加”一点修改，梯度沿捷径原样回传，几十上百层也能训练   L11",
    ], size=12.5, gap=22)


def position_zoom():
    panel(1040, 1280, 660, 140, "⑤ 位置信息（注意力本身不看顺序）")
    lines(1060, 1330, [
        "正弦：PE(pos, 2i) = sin(pos / 10000^(2i/d))，PE(pos, 2i+1) = cos(…)（原论文，可外推）",
        "可学习位置向量：每个位置一个向量（BERT、GPT-2、本项目 L15）   L14 L15",
        "RoPE：按位置旋转 Q、K 的每对维度，点积只依赖相对距离（LLaMA、Qwen 等）",
        "ALiBi：不加位置向量，给注意力分数加一个与距离成正比的负偏置",
    ], size=12.5, gap=23)


def recipe_zoom():
    panel(1040, 1440, 660, 175, "⑥ 训练与推理")
    lines(1060, 1490, [
        "优化：Adam β₁=0.9, β₂=0.98, ε=1e-9；现在多用 AdamW + 权重衰减   L09",
        "学习率：lr = d^(−0.5) · min(step^(−0.5), step · warmup^(−1.5))，warmup 4000 步   L05 L09",
        "正则与稳定：Dropout 0.1、label smoothing 0.1、梯度裁剪、混合精度   L08 L12",
        "推理：KV cache 复用已算的 K / V；MQA / GQA 让多个头共享 K / V；FlashAttention 分块省显存",
        "代价：自注意力的时间 O(T²·d)、显存 O(T²)，序列越长越贵",
    ], size=12.5, gap=23)


def build() -> str:
    svg.clear()
    header()
    overview()
    encoder()
    decoder()
    attention_zoom()
    masks_zoom()
    norm_zoom()
    variants_zoom()
    position_zoom()
    recipe_zoom()
    svg.append("</svg>")
    return "\n".join(svg) + "\n"


if __name__ == "__main__":
    OUT.write_text(build(), encoding="utf-8")
    print(f"wrote {OUT}")
