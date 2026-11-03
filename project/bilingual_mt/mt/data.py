"""Corpus build, TSV loading, bilingual direction expansion."""
from __future__ import annotations

import csv
import importlib.util
import random
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR, LEGACY_DATA_DIR
from .text import clean_pair

KIDS_BUILDER = Path(__file__).resolve().parents[1] / "scripts" / "build_kids_corpora.py"

# Built-in corpora: all built by scripts/build_kids_corpora.py from OPUS (shared with tiny_translator).
BUILTINS: dict[str, dict] = {
    "kids-mix": {"label": "儿童口语·合集（人工种子 + 影视字幕 + Tatoeba，去重）"},
    "kids-subtitles": {"label": "儿童口语·影视字幕（OpenSubtitles 2024 按儿童话题筛选）"},
    "kids-tatoeba": {"label": "儿童口语·日常短句（Tatoeba 2026 筛选，量小质优）"},
}


@dataclass
class Pair:
    src: str
    tgt: str
    direction: str  # "en2zh" | "zh2en"


def _find_existing_tsv(name: str) -> Path | None:
    for root in (DATA_DIR, LEGACY_DATA_DIR):
        for cand in (root / f"{name}.tsv", root / name / f"{name}.tsv"):
            if cand.is_file():
                return cand
    return None


def _kids_builder():
    spec = importlib.util.spec_from_file_location("build_kids_corpora", KIDS_BUILDER)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"无法加载 {KIDS_BUILDER}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def download_builtin(name: str, force: bool = False) -> Path:
    """Ensure mt_data/<name>.tsv exists (building it if it is a built-in); return its path."""
    out = DATA_DIR / f"{name}.tsv"
    if out.is_file() and not force:
        return out
    existing = _find_existing_tsv(name)
    if existing and not force:
        if existing.resolve() == out.resolve():
            return out
        DATA_DIR.mkdir(parents=True, exist_ok=True)
        if not out.exists():
            existing.replace(out)
            print(f"已迁入共享语料目录 → {out}")
        return out
    if name not in BUILTINS:
        raise FileNotFoundError(
            f"未知数据集 {name!r}。可选内置：{list(BUILTINS)}，或把 TSV 放到 {DATA_DIR}/{name}.tsv"
        )
    return _kids_builder().build(name)


def load_en_zh_tsv(
    path: Path,
    *,
    max_pairs: int = 0,
    simplified: bool = True,
    seed: int = 42,
) -> list[tuple[str, str]]:
    """Load TSV as list of (en, zh). Columns: en \\t zh."""
    pairs: list[tuple[str, str]] = []
    with path.open(encoding="utf-8", errors="replace") as fh:
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            cleaned = clean_pair(parts[0], parts[1], simplified=simplified)
            if cleaned is None:
                continue
            pairs.append(cleaned)
    if max_pairs and len(pairs) > max_pairs:
        rng = random.Random(seed)
        pairs = rng.sample(pairs, max_pairs)
    return pairs


def expand_directions(
    en_zh: list[tuple[str, str]],
    direction: str,
) -> list[Pair]:
    """Expand (en, zh) into training Pairs for the requested direction(s)."""
    out: list[Pair] = []
    if direction in ("en2zh", "both"):
        for en, zh in en_zh:
            out.append(Pair(src=en, tgt=zh, direction="en2zh"))
    if direction in ("zh2en", "both"):
        for en, zh in en_zh:
            out.append(Pair(src=zh, tgt=en, direction="zh2en"))
    return out


def split_pairs(
    pairs: list[Pair],
    *,
    val_ratio: float,
    test_ratio: float,
    seed: int,
) -> tuple[list[Pair], list[Pair], list[Pair]]:
    rng = random.Random(seed)
    idx = list(range(len(pairs)))
    rng.shuffle(idx)
    n = len(idx)
    n_test = max(1, int(n * test_ratio)) if test_ratio > 0 and n > 50 else 0
    n_val = max(1, int(n * val_ratio)) if val_ratio > 0 and n > 50 else 0
    test_i = idx[:n_test]
    val_i = idx[n_test : n_test + n_val]
    train_i = idx[n_test + n_val :]
    pick = lambda ids: [pairs[i] for i in ids]
    return pick(train_i), pick(val_i), pick(test_i)


def write_split_tsv(path: Path, pairs: list[Pair]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        w.writerow(["src", "tgt", "direction"])
        for p in pairs:
            w.writerow([p.src, p.tgt, p.direction])


def read_split_tsv(path: Path) -> list[Pair]:
    pairs: list[Pair] = []
    with path.open(encoding="utf-8") as fh:
        reader = csv.DictReader(fh, delimiter="\t")
        for row in reader:
            pairs.append(Pair(src=row["src"], tgt=row["tgt"], direction=row["direction"]))
    return pairs
