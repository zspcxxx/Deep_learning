"""Corpus download, TSV loading, bilingual direction expansion."""
from __future__ import annotations

import csv
import gzip
import io
import random
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path

from .config import DATA_DIR, LEGACY_DATA_DIR
from .text import clean_pair

# Built-in corpora (same sources as tiny_translator where possible).
BUILTINS: dict[str, dict] = {
    "kids-dialogue": {
        "url": "",
        "kind": "kids_build",
        "label": "儿童对话中英（家庭/学校口语，适合双向互译）",
    },
    "cmn-eng": {
        "url": "https://www.manythings.org/anki/cmn-eng.zip",
        "kind": "anki",
        "label": "Tatoeba 中英短句 (cmn-eng)",
    },
    "news-commentary": {
        "url": "https://data.statmt.org/news-commentary/v18/training/news-commentary-v18.en-zh.tsv.gz",
        "kind": "tsv_gz",
        "label": "News Commentary v18 新闻评论 (en-zh)",
    },
    "ted2020": {
        "url": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-zh.zip",
        "kind": "opus_moses",
        "label": "TED2020 演讲字幕 (en-zh)",
    },
}


@dataclass
class Pair:
    src: str
    tgt: str
    direction: str  # "en2zh" | "zh2en"


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    print(f"下载 {url}")
    # manythings.org 等站点对无 UA 的请求会返回 406。
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Mozilla/5.0 (compatible; bilingual_mt/0.1; +local-lab)"},
    )
    with urllib.request.urlopen(req, timeout=600) as resp, tmp.open("wb") as out:
        while True:
            chunk = resp.read(1024 * 1024)
            if not chunk:
                break
            out.write(chunk)
    tmp.replace(dest)


def _find_existing_tsv(name: str) -> Path | None:
    for root in (DATA_DIR, LEGACY_DATA_DIR):
        for cand in (root / f"{name}.tsv", root / name / f"{name}.tsv"):
            if cand.is_file():
                return cand
    return None


def _parse_anki_zip(zip_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        names = [n for n in zf.namelist() if n.endswith(".txt")]
        if not names:
            raise RuntimeError(f"anki zip 中没有 .txt：{zip_path}")
        with zf.open(names[0]) as fh:
            text = io.TextIOWrapper(fh, encoding="utf-8")
            for line in text:
                parts = line.rstrip("\n").split("\t")
                if len(parts) < 2:
                    continue
                en, zh = parts[0], parts[1]
                pairs.append((en, zh))
    return pairs


def _parse_tsv_gz(path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    opener = gzip.open if path.suffix == ".gz" or path.name.endswith(".tsv.gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as fh:  # type: ignore[arg-type]
        for line in fh:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 2:
                continue
            # news-commentary: en \t zh
            pairs.append((parts[0], parts[1]))
    return pairs


def _parse_opus_moses(zip_path: Path) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    with zipfile.ZipFile(zip_path) as zf:
        en_name = next((n for n in zf.namelist() if n.endswith(".en")), None)
        zh_name = next((n for n in zf.namelist() if n.endswith(".zh")), None)
        if not en_name or not zh_name:
            raise RuntimeError(f"OPUS moses zip 缺少 .en/.zh：{list(zf.namelist())[:10]}")
        with zf.open(en_name) as fe, zf.open(zh_name) as fz:
            for en, zh in zip(
                io.TextIOWrapper(fe, encoding="utf-8", errors="replace"),
                io.TextIOWrapper(fz, encoding="utf-8", errors="replace"),
            ):
                pairs.append((en.rstrip("\n"), zh.rstrip("\n")))
    return pairs


def download_builtin(name: str, force: bool = False) -> Path:
    """Ensure data/<name>.tsv exists; return its path."""
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

    meta = BUILTINS.get(name)
    if meta is None:
        raise FileNotFoundError(
            f"未知数据集 {name!r}。可选内置：{list(BUILTINS)}，或把 TSV 放到 {DATA_DIR}/{name}.tsv"
        )

    kind = meta["kind"]
    if kind == "kids_build":
        import importlib.util

        script = Path(__file__).resolve().parents[1] / "scripts" / "build_kids_dialogue.py"
        spec = importlib.util.spec_from_file_location("build_kids_dialogue", script)
        if spec is None or spec.loader is None:
            raise RuntimeError(f"无法加载 {script}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        return mod.build(out, force_download=force)

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    raw = DATA_DIR / f"_raw_{name}"
    raw.mkdir(exist_ok=True)
    url = meta["url"]
    if kind == "anki":
        zpath = raw / "cmn-eng.zip"
        if not zpath.is_file() or force:
            _download(url, zpath)
        pairs = _parse_anki_zip(zpath)
    elif kind == "tsv_gz":
        gpath = raw / "news-commentary.tsv.gz"
        if not gpath.is_file() or force:
            _download(url, gpath)
        pairs = _parse_tsv_gz(gpath)
    elif kind == "opus_moses":
        zpath = raw / "ted2020.zip"
        if not zpath.is_file() or force:
            _download(url, zpath)
        pairs = _parse_opus_moses(zpath)
    else:
        raise ValueError(kind)

    with out.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh, delimiter="\t", lineterminator="\n")
        for en, zh in pairs:
            w.writerow([en, zh])
    print(f"写入 {out} （{len(pairs)} 行）")
    return out


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
