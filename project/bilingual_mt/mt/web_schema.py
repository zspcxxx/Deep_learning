"""Web form schema / presets / validation for bilingual_mt."""
from __future__ import annotations

from dataclasses import fields

from .config import DIRECTIONS, TrainConfig

GROUPS = [
    ("data", "数据"),
    ("model", "模型结构"),
    ("train", "训练"),
    ("decode", "解码"),
]

# key -> (group, label, help, extra)
FIELDS: dict[str, tuple[str, str, str, dict]] = {
    "name": ("data", "运行名称", "保存在 runs/ 下的目录名", {}),
    "dataset": ("data", "数据集", "data/<名称>.tsv，每行「英文\\t中文」", {"choices": []}),
    "direction": (
        "data",
        "翻译方向",
        "both = 中英互译（共享一套模型，源端加方向标签）",
        {"choices": [["both", "中英互译 (both)"], ["en2zh", "仅英→中"], ["zh2en", "仅中→英"]]},
    ),
    "max_pairs": ("data", "最多使用句对数", "0 表示全部", {"min": 0, "max": 2_000_000, "step": 1000}),
    "zh_simplified": ("data", "中文统一为简体", "需要 zhconv", {}),
    "max_src_len": ("data", "源端最大子词数", "含方向标签与 </s>", {"min": 8, "max": 256}),
    "max_tgt_len": ("data", "目标端最大子词数", "", {"min": 8, "max": 256}),
    "val_ratio": ("data", "验证集比例", "", {"min": 0.01, "max": 0.3, "step": 0.01}),
    "test_ratio": ("data", "测试集比例", "", {"min": 0.01, "max": 0.3, "step": 0.01}),
    "vocab_size": ("data", "联合 BPE 词表大小", "中英共词表；过小会自动抬升", {"min": 2000, "max": 32000, "step": 500}),
    "seed": ("data", "随机种子", "", {"min": 0, "max": 2**31 - 1}),
    "d_model": ("model", "模型宽度 d_model", "须能被头数整除", {"min": 32, "max": 1024, "step": 32}),
    "n_heads": ("model", "注意力头数", "", {"min": 1, "max": 16}),
    "n_enc_layers": ("model", "编码器层数", "", {"min": 1, "max": 12}),
    "n_dec_layers": ("model", "解码器层数", "", {"min": 1, "max": 12}),
    "d_ff": ("model", "前馈层宽度 d_ff", "通常为 d_model×4", {"min": 32, "max": 8192, "step": 32}),
    "dropout": ("model", "Dropout", "", {"min": 0.0, "max": 0.5, "step": 0.05}),
    "norm_first": ("model", "Pre-LN", "勾选为 Pre-LN", {}),
    "activation": ("model", "前馈激活", "", {"choices": [["gelu", "GELU"], ["relu", "ReLU"]]}),
    "tie_embeddings": ("model", "共享输出层权重", "嵌入矩阵兼作输出投影", {}),
    "epochs": ("train", "训练轮数", "", {"min": 1, "max": 500}),
    "batch_size": ("train", "批大小", "", {"min": 4, "max": 512, "step": 4}),
    "lr": ("train", "峰值学习率", "", {"min": 1e-5, "max": 1e-2, "step": 1e-4}),
    "schedule": (
        "train",
        "学习率计划",
        "",
        {
            "choices": [
                ["cosine", "预热 + 余弦衰减"],
                ["inverse_sqrt", "预热 + 平方根倒数"],
                ["constant", "预热后恒定"],
            ]
        },
    ),
    "warmup_steps": ("train", "预热步数", "", {"min": 0, "max": 20000, "step": 50}),
    "weight_decay": ("train", "权重衰减", "", {"min": 0.0, "max": 0.3, "step": 0.01}),
    "label_smoothing": ("train", "标签平滑", "", {"min": 0.0, "max": 0.3, "step": 0.05}),
    "grad_clip": ("train", "梯度裁剪", "0 = 不裁剪", {"min": 0.0, "max": 10.0, "step": 0.5}),
    "patience": ("train", "早停耐心", "0 = 不早停", {"min": 0, "max": 100}),
    "amp": ("train", "混合精度 AMP", "仅 CUDA 生效", {}),
    "device": (
        "train",
        "设备",
        "",
        {"choices": [["auto", "自动"], ["cuda", "GPU (CUDA)"], ["cpu", "CPU"]]},
    ),
    "log_every": ("train", "日志间隔（步）", "", {"min": 1, "max": 500}),
    "beam_size": ("decode", "默认束宽", "训练内验证用贪心；交互翻译可用束搜索", {"min": 1, "max": 10}),
    "max_decode_len": ("decode", "最大解码长度", "", {"min": 16, "max": 256}),
}

# Fields present on TrainConfig but omitted from the web form.
_HIDDEN = {"spm_model", "num_workers", "share_encoder_decoder_emb"}

PRESETS = {
    "kids": {
        "label": "儿童对话互译（推荐）",
        "values": {
            "name": "kids",
            "dataset": "kids-dialogue",
            "direction": "both",
            "max_pairs": 0,
            "max_src_len": 32,
            "max_tgt_len": 32,
            "vocab_size": 4000,
            "d_model": 256,
            "n_heads": 4,
            "n_enc_layers": 3,
            "n_dec_layers": 3,
            "d_ff": 1024,
            "dropout": 0.15,
            "epochs": 40,
            "batch_size": 32,
            "lr": 7e-4,
            "warmup_steps": 500,
            "patience": 6,
            "beam_size": 4,
            "max_decode_len": 40,
        },
    },
    "quick": {
        "label": "快速试跑（Jetson ≈ 数分钟）",
        "values": {
            "name": "quick",
            "dataset": "kids-dialogue",
            "direction": "both",
            "max_pairs": 0,
            "max_src_len": 32,
            "max_tgt_len": 32,
            "vocab_size": 4000,
            "d_model": 128,
            "n_heads": 4,
            "n_enc_layers": 2,
            "n_dec_layers": 2,
            "d_ff": 512,
            "epochs": 12,
            "batch_size": 64,
            "lr": 1e-3,
            "warmup_steps": 200,
            "patience": 3,
        },
    },
    "jetson": {
        "label": "Jetson 新闻互译（中等，非儿童）",
        "values": {
            "name": "jetson",
            "dataset": "news-commentary",
            "direction": "both",
            "max_pairs": 80000,
            "max_src_len": 48,
            "max_tgt_len": 48,
            "vocab_size": 6000,
            "d_model": 256,
            "n_heads": 4,
            "n_enc_layers": 4,
            "n_dec_layers": 4,
            "d_ff": 1024,
            "epochs": 20,
            "batch_size": 16,
            "lr": 5e-4,
            "warmup_steps": 2000,
            "patience": 4,
        },
    },
    "en2zh": {
        "label": "仅英→中（儿童语料）",
        "values": {"name": "en2zh_kids", "direction": "en2zh", "dataset": "kids-dialogue"},
    },
    "zh2en": {
        "label": "仅中→英（儿童语料）",
        "values": {"name": "zh2en_kids", "direction": "zh2en", "dataset": "kids-dialogue"},
    },
}


def schema(datasets: list[dict]) -> dict:
    defaults = TrainConfig().to_dict()
    out = []
    for f in fields(TrainConfig):
        if f.name in _HIDDEN or f.name not in FIELDS:
            continue
        group, label, help_text, extra = FIELDS[f.name]
        kind = {"bool": "bool", "int": "int", "float": "float"}.get(f.type, "str")
        extra = dict(extra)
        if f.name == "dataset":
            extra["choices"] = [[d["name"], d["label"]] for d in datasets]
        if "choices" in extra:
            kind = "choice"
        out.append(
            {
                "key": f.name,
                "group": group,
                "label": label,
                "help": help_text,
                "kind": kind,
                "default": defaults[f.name],
                **extra,
            }
        )
    presets = [{"key": k, **p} for k, p in PRESETS.items()]
    return {"groups": GROUPS, "fields": out, "presets": presets, "defaults": defaults}


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
    errors: dict[str, str] = {}
    values = TrainConfig().to_dict()
    for f in fields(TrainConfig):
        if f.name in _HIDDEN or f.name not in raw:
            continue
        if f.name not in FIELDS and f.name not in raw:
            continue
        kind = {"bool": "bool", "int": "int", "float": "float"}.get(f.type, "str")
        if f.name in FIELDS:
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
    if values["direction"] not in DIRECTIONS:
        errors["direction"] = f"必须是 {DIRECTIONS}"
    if "d_model" not in errors and "n_heads" not in errors and values["d_model"] % values["n_heads"]:
        errors["n_heads"] = f"d_model={values['d_model']} 不能被头数整除"
    if values["val_ratio"] + values["test_ratio"] >= 0.5:
        errors["test_ratio"] = "验证集与测试集合计应小于 50%"
    if errors:
        return None, errors
    return TrainConfig(**{k: values[k] for k in (f.name for f in fields(TrainConfig))}), {}
