"""Prepare splits + joint SentencePiece vocabulary."""
from __future__ import annotations

import json
from pathlib import Path

from .config import RUNS_DIR, TrainConfig, load_config
from .data import (
    download_builtin,
    expand_directions,
    load_en_zh_tsv,
    split_pairs,
    write_split_tsv,
)
from .vocab import train_sentencepiece


def prepare(cfg: TrainConfig, run_dir: Path | None = None, force_download: bool = False) -> Path:
    run_dir = run_dir or (RUNS_DIR / cfg.name)
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )

    tsv = download_builtin(cfg.dataset, force=force_download)
    en_zh = load_en_zh_tsv(
        tsv, max_pairs=cfg.max_pairs, simplified=cfg.zh_simplified, seed=cfg.seed
    )
    print(f"清洗后平行句对：{len(en_zh)}")
    pairs = expand_directions(en_zh, cfg.direction)
    print(f"方向 {cfg.direction} → 训练样本：{len(pairs)}")
    train, val, test = split_pairs(
        pairs, val_ratio=cfg.val_ratio, test_ratio=cfg.test_ratio, seed=cfg.seed
    )
    write_split_tsv(run_dir / "train.tsv", train)
    write_split_tsv(run_dir / "val.tsv", val)
    write_split_tsv(run_dir / "test.tsv", test)
    print(f"划分 train/val/test = {len(train)}/{len(val)}/{len(test)}")

    # Joint corpus: all EN + ZH text (unique enough via raw lines).
    texts: list[str] = []
    for en, zh in en_zh:
        texts.append(en)
        texts.append(zh)
    spm_prefix = run_dir / "spm"
    model_path = train_sentencepiece(texts, spm_prefix, vocab_size=cfg.vocab_size)
    cfg.spm_model = str(model_path)
    (run_dir / "config.json").write_text(
        json.dumps(cfg.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"SentencePiece → {model_path}")
    return run_dir


def prepare_from_config(config_path: str, run_name: str | None = None, force: bool = False) -> Path:
    cfg = load_config(config_path)
    if run_name:
        cfg.name = run_name
    return prepare(cfg, force_download=force)
