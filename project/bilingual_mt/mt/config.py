"""Load YAML configs into a typed TrainConfig."""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[1]
CONFIGS = ROOT / "configs"
RUNS_DIR = ROOT / "runs"

# Shared by all translation projects under project/ (tiny_translator, bilingual_mt, …).
DATA_DIR = ROOT.parent / "mt_data"
# Legacy per-project folder still accepted as a fallback when listing/loading.
LEGACY_DATA_DIR = ROOT / "data"

DIRECTIONS = ("en2zh", "zh2en", "both")


@dataclass
class TrainConfig:
    name: str = "bilingual"
    # data
    dataset: str = "kids-dialogue"  # stem under project/mt_data/, or a built-in name
    direction: str = "both"  # en2zh | zh2en | both
    max_pairs: int = 0
    zh_simplified: bool = True
    max_src_len: int = 64
    max_tgt_len: int = 64
    val_ratio: float = 0.02
    test_ratio: float = 0.02
    seed: int = 42
    # vocabulary (joint SentencePiece BPE)
    vocab_size: int = 8000
    spm_model: str = ""  # filled after prepare; relative to run dir or absolute
    # model
    d_model: int = 512
    n_heads: int = 8
    n_enc_layers: int = 6
    n_dec_layers: int = 6
    d_ff: int = 2048
    dropout: float = 0.1
    norm_first: bool = True
    activation: str = "gelu"
    tie_embeddings: bool = True
    share_encoder_decoder_emb: bool = True  # one embedding table for bilingual vocab
    # training
    epochs: int = 30
    batch_size: int = 64
    lr: float = 5e-4
    schedule: str = "cosine"  # cosine | inverse_sqrt | constant
    warmup_steps: int = 4000
    weight_decay: float = 0.01
    label_smoothing: float = 0.1
    grad_clip: float = 1.0
    patience: int = 5
    amp: bool = True
    device: str = "auto"
    num_workers: int = 0
    log_every: int = 50
    # decode
    beam_size: int = 4
    max_decode_len: int = 80

    def to_dict(self) -> dict:
        return asdict(self)


def _merge(base: dict, override: dict) -> dict:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge(out[k], v)
        else:
            out[k] = v
    return out


def _flatten(raw: dict) -> dict:
    """Allow nested YAML groups (data/model/train) or a flat mapping."""
    flat: dict[str, Any] = {}
    known = {f.name for f in fields(TrainConfig)}
    for key, value in raw.items():
        if key == "inherit":
            continue
        if isinstance(value, dict) and key in {"data", "model", "train", "decode", "vocab"}:
            for kk, vv in value.items():
                flat[kk] = vv
        elif key in known:
            flat[key] = value
    return flat


def load_config(path: str | Path) -> TrainConfig:
    path = Path(path)
    if not path.is_file():
        alt = CONFIGS / path
        if alt.is_file():
            path = alt
        elif (CONFIGS / f"{path}.yaml").is_file():
            path = CONFIGS / f"{path}.yaml"
        else:
            raise FileNotFoundError(f"找不到配置文件：{path}")
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if "inherit" in raw:
        parent = load_config(raw["inherit"]).to_dict()
        raw = _merge(parent, _flatten(raw))
    else:
        raw = _flatten(raw)
    cfg = TrainConfig(**{f.name: raw[f.name] for f in fields(TrainConfig) if f.name in raw})
    if cfg.direction not in DIRECTIONS:
        raise ValueError(f"direction 必须是 {DIRECTIONS}，收到 {cfg.direction!r}")
    return cfg


def resolve_device(spec: str) -> str:
    import torch

    if spec == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return spec
