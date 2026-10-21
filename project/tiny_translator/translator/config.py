"""Training configuration: defaults, presets, the form schema shown in the web UI, and validation."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields


@dataclass
class TrainConfig:
    name: str = "demo"
    # data
    dataset: str = "cmn-eng"
    max_pairs: int = 0
    zh_simplified: bool = True
    max_src_len: int = 20
    max_tgt_len: int = 30
    min_freq: int = 2
    val_ratio: float = 0.05
    test_ratio: float = 0.05
    # model
    d_model: int = 256
    n_heads: int = 4
    n_enc_layers: int = 3
    n_dec_layers: int = 3
    d_ff: int = 1024
    dropout: float = 0.1
    norm_first: bool = True
    activation: str = "gelu"
    tie_embeddings: bool = True
    # training
    epochs: int = 30
    batch_size: int = 128
    lr: float = 7e-4
    schedule: str = "cosine"
    warmup_steps: int = 400
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    patience: int = 0
    device: str = "auto"
    seed: int = 42

    def to_dict(self) -> dict:
        return asdict(self)


GROUPS = [
    ("data", "数据"),
    ("model", "模型结构"),
    ("train", "训练"),
]

# key -> (group, label, help, extra). extra holds min/max/step for numbers, or choices.
FIELDS: dict[str, tuple[str, str, str, dict]] = {
    "name": ("data", "运行名称", "用于区分不同训练，保存在 runs/ 下的目录名里", {}),
    "dataset": ("data", "数据集", "句对文件 data/<名称>.tsv，每行“英文<Tab>中文”", {"choices": []}),
    "max_pairs": ("data", "最多使用句对数", "0 表示全部；先用少量数据试跑可以更快看到效果", {"min": 0, "max": 2_000_000, "step": 1000}),
    "zh_simplified": ("data", "中文统一为简体", "Tatoeba 的中文繁简混杂，统一后字表更小、BLEU 更可比（需要 zhconv）", {}),
    "max_src_len": ("data", "英文最大词数", "超过此长度的句对被丢弃", {"min": 4, "max": 128}),
    "max_tgt_len": ("data", "中文最大字数", "超过此长度的句对被丢弃；也决定译文最长能生成多少字", {"min": 4, "max": 160}),
    "min_freq": ("data", "词表最低频次", "训练集中出现少于此次数的词记为 <unk>", {"min": 1, "max": 20}),
    "val_ratio": ("data", "验证集比例", "每轮结束用来计算验证损失和 BLEU", {"min": 0.01, "max": 0.3, "step": 0.01}),
    "test_ratio": ("data", "测试集比例", "训练全程不参与，只在最后和验证页评估", {"min": 0.01, "max": 0.3, "step": 0.01}),
    "d_model": ("model", "模型宽度 d_model", "词向量和每层隐藏状态的维度，必须能被头数整除", {"min": 32, "max": 1024, "step": 32}),
    "n_heads": ("model", "注意力头数", "多头注意力的头数", {"min": 1, "max": 16}),
    "n_enc_layers": ("model", "编码器层数", "", {"min": 1, "max": 12}),
    "n_dec_layers": ("model", "解码器层数", "", {"min": 1, "max": 12}),
    "d_ff": ("model", "前馈层宽度 d_ff", "通常取 d_model 的 4 倍", {"min": 32, "max": 8192, "step": 32}),
    "dropout": ("model", "Dropout", "注意力权重、残差支路和嵌入上的丢弃率", {"min": 0.0, "max": 0.5, "step": 0.05}),
    "norm_first": ("model", "Pre-LN", "勾选为 Pre-LN（先归一化再进子层），不勾选为原论文的 Post-LN", {}),
    "activation": ("model", "前馈激活函数", "", {"choices": [["gelu", "GELU"], ["relu", "ReLU"]]}),
    "tie_embeddings": ("model", "共享输出层权重", "解码器嵌入矩阵兼作输出投影，参数更少", {}),
    "epochs": ("train", "训练轮数", "", {"min": 1, "max": 500}),
    "batch_size": ("train", "批大小", "每步的句对数", {"min": 8, "max": 1024, "step": 8}),
    "lr": ("train", "峰值学习率", "预热结束时达到的学习率", {"min": 1e-5, "max": 1e-2, "step": 1e-4}),
    "schedule": ("train", "学习率计划", "", {"choices": [["cosine", "预热 + 余弦衰减"], ["inverse_sqrt", "预热 + 平方根倒数（原论文）"], ["constant", "预热后恒定"]]}),
    "warmup_steps": ("train", "预热步数", "学习率从 0 线性升到峰值所用的步数", {"min": 0, "max": 20000, "step": 50}),
    "weight_decay": ("train", "权重衰减", "AdamW 的解耦权重衰减，偏置和归一化参数不衰减", {"min": 0.0, "max": 0.3, "step": 0.01}),
    "label_smoothing": ("train", "标签平滑", "", {"min": 0.0, "max": 0.3, "step": 0.05}),
    "grad_clip": ("train", "梯度裁剪", "全局梯度范数上限，0 表示不裁剪", {"min": 0.0, "max": 10.0, "step": 0.5}),
    "patience": ("train", "早停耐心", "验证 BLEU 连续多少轮没有提升就停止，0 表示不早停", {"min": 0, "max": 100}),
    "device": ("train", "设备", "", {"choices": [["auto", "自动（有 GPU 用 GPU）"], ["cuda", "GPU (CUDA)"], ["cpu", "CPU"]]}),
    "seed": ("train", "随机种子", "决定数据划分和初始化", {"min": 0, "max": 2**31 - 1}),
}

PRESETS = {
    "quick": {
        "label": "快速试跑（约 1 分钟）",
        "values": {"name": "quick", "d_model": 128, "n_heads": 4, "n_enc_layers": 2, "n_dec_layers": 2,
                   "d_ff": 512, "epochs": 12, "warmup_steps": 200, "lr": 1e-3},
    },
    "standard": {
        "label": "标准（约 4 分钟）",
        "values": {"name": "standard"},
    },
    "large": {
        "label": "加大模型（更久、通常更准）",
        "values": {"name": "large", "d_model": 384, "n_heads": 6, "n_enc_layers": 4, "n_dec_layers": 4,
                   "d_ff": 1536, "dropout": 0.2, "epochs": 50, "lr": 5e-4, "warmup_steps": 800, "patience": 8},
    },
}


def schema(datasets: list[dict]) -> dict:
    defaults = TrainConfig().to_dict()
    out = []
    for f in fields(TrainConfig):
        group, label, help_text, extra = FIELDS[f.name]
        kind = {"bool": "bool", "int": "int", "float": "float"}.get(f.type, "str")
        extra = dict(extra)
        if f.name == "dataset":
            extra["choices"] = [[d["name"], d["label"]] for d in datasets]
        if "choices" in extra:
            kind = "choice"
        out.append({"key": f.name, "group": group, "label": label, "help": help_text, "kind": kind,
                    "default": defaults[f.name], **extra})
    return {"groups": GROUPS, "fields": out, "presets": PRESETS, "defaults": defaults}


def _coerce(kind: str, value):
    if kind == "bool":
        if isinstance(value, str):
            return value.lower() in {"1", "true", "on", "yes"}
        return bool(value)
    if kind == "int":
        if isinstance(value, float) and not value.is_integer():
            raise ValueError("需要整数")
        return int(value)
    if kind == "float":
        return float(value)
    return str(value).strip()


def validate(raw: dict, dataset_names: list[str]) -> tuple[TrainConfig | None, dict[str, str]]:
    """Return a config built from raw form values, plus per-field error messages."""
    errors: dict[str, str] = {}
    values = TrainConfig().to_dict()
    for f in fields(TrainConfig):
        if f.name not in raw:
            continue
        kind = {"bool": "bool", "int": "int", "float": "float"}.get(f.type, "str")
        try:
            values[f.name] = _coerce(kind, raw[f.name])
        except (TypeError, ValueError):
            errors[f.name] = "格式不正确"
            continue
        extra = FIELDS[f.name][3]
        v = values[f.name]
        if kind in {"int", "float"}:
            if "min" in extra and v < extra["min"]:
                errors[f.name] = f"不能小于 {extra['min']}"
            elif "max" in extra and v > extra["max"]:
                errors[f.name] = f"不能大于 {extra['max']}"
        elif f.name != "dataset" and "choices" in extra and v not in [c[0] for c in extra["choices"]]:
            errors[f.name] = "不在可选范围内"

    if not values["name"]:
        errors["name"] = "请填写运行名称"
    if values["dataset"] not in dataset_names:
        errors["dataset"] = "数据集不存在"
    if "d_model" not in errors and "n_heads" not in errors and values["d_model"] % values["n_heads"]:
        errors["n_heads"] = f"d_model={values['d_model']} 不能被头数整除"
    if values["val_ratio"] + values["test_ratio"] >= 0.5:
        errors["test_ratio"] = "验证集与测试集合计应小于 50%"
    if errors:
        return None, errors
    return TrainConfig(**values), {}
