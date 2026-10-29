"""Parallel corpus: download, tokenisation, vocabularies, splits and batching."""
from __future__ import annotations

import io
import random
import re
import unicodedata
import urllib.request
import zipfile
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
# Shared by all translation projects under project/ (see project/mt_data/).
DATA_DIR = ROOT.parent / "mt_data"
LEGACY_DATA_DIR = ROOT / "data"

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

BUILTIN = {
    "kids-dialogue": {
        "label": "儿童对话中英（家庭/学校口语，共享 mt_data，适合小朋友）",
        "url": "",
        "format": "shared_ready",  # 由 bilingual_mt 构建；本页只消费共享文件
    },
    "cmn-eng": {
        "label": "Tatoeba 英→中（manythings.org，约 3 万短句，CC-BY 2.0）",
        "url": "https://www.manythings.org/anki/cmn-eng.zip",
        "format": "anki_zip",
        "member": "cmn.txt",
    },
    # WMT News Commentary：新闻评论平行语料，质量高、规模大，比短句 Tatoeba 更适合提升 BLEU。
    # 句子偏长，建议把英文最大词数 / 中文最大字数调到约 40 / 60。
    "news-commentary": {
        "label": "News Commentary 英→中（WMT/statmt，约 31 万对）",
        "url": "https://data.statmt.org/news-commentary/v16/training/news-commentary-v16.en-zh.tsv.gz",
        "format": "tsv_gz",
    },
    "ted2020": {
        "label": "TED2020 英→中（OPUS，约 1.6 万口语句对）",
        "url": "https://object.pouta.csc.fi/OPUS-TED2020/v1/moses/en-zh.txt.zip",
        "format": "opus_moses",
        "prefix": "TED2020.en-zh",
    },
}

EN_TOKEN = re.compile(r"[a-z0-9]+(?:'[a-z]+)?|[^\sa-z0-9]")


def tokenize_en(text: str) -> list[str]:
    text = unicodedata.normalize("NFKC", text).lower().replace("\u2019", "'").replace("\u2018", "'")
    return EN_TOKEN.findall(text)


def tokenize_zh(text: str) -> list[str]:
    # Character level; full-width punctuation is kept so the output reads like normal Chinese.
    return [ch for ch in text.strip() if not ch.isspace()]


def detokenize_zh(tokens: list[str]) -> str:
    return "".join(tokens)


def to_simplified(text: str) -> str:
    try:
        import zhconv
    except ImportError as exc:
        raise RuntimeError("“中文统一为简体”需要 zhconv：pip install zhconv，或在配置里取消勾选") from exc
    return zhconv.convert(text, "zh-cn")


# ---------------------------------------------------------------- datasets

def dataset_path(name: str) -> Path:
    """Prefer shared mt_data/; fall back to this project's legacy data/."""
    shared = DATA_DIR / f"{name}.tsv"
    if shared.is_file():
        return shared
    legacy = LEGACY_DATA_DIR / f"{name}.tsv"
    if legacy.is_file():
        return legacy
    return shared


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def list_datasets() -> list[dict]:
    labels = {name: spec["label"] for name, spec in BUILTIN.items()}
    for root in (DATA_DIR, LEGACY_DATA_DIR):
        if root.is_dir():
            for path in sorted(root.glob("*.tsv")):
                labels.setdefault(path.stem, f"自定义句对文件 {path.name}")
    out = []
    for name, label in labels.items():
        path = dataset_path(name)
        ready = path.is_file()
        spec = BUILTIN.get(name)
        downloadable = bool(spec) and spec.get("format") != "shared_ready"
        out.append({"name": name, "label": label, "ready": ready, "downloadable": downloadable,
                    "pairs": _count_lines(path) if ready else None})
    return out


def _fetch(url: str, timeout: int = 600) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 (tiny-translator)"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return resp.read()


def _ascii_ratio(text: str) -> float:
    if not text:
        return 0.0
    return sum(1 for c in text if ("a" <= c <= "z") or ("A" <= c <= "Z")) / len(text)


def _normalize_en_zh_pairs(pairs: list[tuple[str, str]]) -> list[tuple[str, str]]:
    """Ensure (English, Chinese) order; some dumps put Chinese first."""
    if not pairs:
        return pairs
    sample = pairs[:200]
    swaps = sum(1 for en, zh in sample if _ascii_ratio(en) < _ascii_ratio(zh))
    if swaps > len(sample) // 2:
        pairs = [(zh, en) for en, zh in pairs]
    # de-duplicate while keeping order
    seen: set[tuple[str, str]] = set()
    out: list[tuple[str, str]] = []
    for pair in pairs:
        if pair in seen:
            continue
        seen.add(pair)
        out.append(pair)
    return out


def _pairs_from_anki_zip(blob: bytes, member: str) -> list[tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(member).decode("utf-8")
    pairs = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def _pairs_from_opus_moses(blob: bytes, prefix: str) -> list[tuple[str, str]]:
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        ens = zf.read(f"{prefix}.en").decode("utf-8").splitlines()
        zhs = zf.read(f"{prefix}.zh").decode("utf-8").splitlines()
    return [(e.strip(), z.strip()) for e, z in zip(ens, zhs) if e.strip() and z.strip()]


def _pairs_from_tsv_gz(blob: bytes) -> list[tuple[str, str]]:
    import gzip

    text = gzip.decompress(blob).decode("utf-8", errors="replace")
    pairs = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def download_dataset(name: str, log=print) -> int:
    spec = BUILTIN.get(name)
    if spec is None:
        raise ValueError(f"数据集 {name} 不支持自动下载，请把句对文件放到 {DATA_DIR}/{name}.tsv")
    dest = DATA_DIR / f"{name}.tsv"
    fmt = spec.get("format", "anki_zip")
    if fmt == "shared_ready":
        if dest.is_file():
            n = _count_lines(dest)
            log(f"{name} 已在共享目录就绪：{dest}（{n} 行）")
            return n
        raise FileNotFoundError(
            f"{name} 需由 bilingual_mt 构建后放入 {dest}（本机下载后 rsync 到设备）"
        )
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log(f"下载 {spec['url']} …")
    blob = _fetch(spec["url"])
    if fmt == "anki_zip":
        pairs = _pairs_from_anki_zip(blob, spec["member"])
    elif fmt == "opus_moses":
        pairs = _pairs_from_opus_moses(blob, spec["prefix"])
    elif fmt == "tsv_gz":
        pairs = _pairs_from_tsv_gz(blob)
    else:
        raise ValueError(f"未知数据集格式：{fmt}")
    pairs = _normalize_en_zh_pairs(pairs)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    write_pairs(dest, pairs)
    try:
        shown = dest.relative_to(ROOT.parent)
    except ValueError:
        shown = dest
    log(f"已保存 {len(pairs)} 个句对到 {shown}")
    return len(pairs)


def read_pairs(path: Path) -> list[tuple[str, str]]:
    pairs = []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            pairs.append((parts[0].strip(), parts[1].strip()))
    return pairs


def write_pairs(path: Path, pairs: list[tuple[str, str]]) -> None:
    path.write_text("".join(f"{en}\t{zh}\n" for en, zh in pairs), encoding="utf-8")


# ---------------------------------------------------------------- vocabulary

class Vocab:
    def __init__(self, itos: list[str]):
        self.itos = itos
        self.stoi = {tok: i for i, tok in enumerate(itos)}

    def __len__(self) -> int:
        return len(self.itos)

    @classmethod
    def build(cls, sentences: list[list[str]], min_freq: int) -> "Vocab":
        counts = Counter(tok for sent in sentences for tok in sent)
        words = sorted((t for t, n in counts.items() if n >= min_freq), key=lambda t: (-counts[t], t))
        return cls(SPECIALS + words)

    def encode(self, tokens: list[str]) -> list[int]:
        return [self.stoi.get(tok, UNK) for tok in tokens]

    def decode(self, ids: list[int]) -> list[str]:
        out = []
        for i in ids:
            if i == EOS:
                break
            if i in (PAD, BOS):
                continue
            out.append(self.itos[i] if 0 <= i < len(self.itos) else "<unk>")
        return out

    def to_list(self) -> list[str]:
        return list(self.itos)


# ---------------------------------------------------------------- prepared splits

@dataclass
class Example:
    en: str
    zh: str
    src: list[int]
    tgt: list[int]


@dataclass
class Prepared:
    train: list[Example]
    val: list[Example]
    test: list[Example]
    src_vocab: Vocab
    tgt_vocab: Vocab
    stats: dict


def prepare(cfg, log=print) -> Prepared:
    path = dataset_path(cfg.dataset)
    if not path.is_file():
        raise FileNotFoundError(f"找不到数据集文件 {path}")
    pairs = read_pairs(path)
    total = len(pairs)
    rng = random.Random(cfg.seed)
    rng.shuffle(pairs)
    if cfg.max_pairs:
        pairs = pairs[: cfg.max_pairs]
    if cfg.zh_simplified:
        pairs = [(en, to_simplified(zh)) for en, zh in pairs]

    kept = []
    for en, zh in pairs:
        s, t = tokenize_en(en), tokenize_zh(zh)
        if 0 < len(s) <= cfg.max_src_len and 0 < len(t) <= cfg.max_tgt_len:
            kept.append((en, zh, s, t))
    if len(kept) < 50:
        raise ValueError(f"过滤后只剩 {len(kept)} 个句对，请放宽长度限制或增加数据")

    n_test = max(1, int(len(kept) * cfg.test_ratio))
    n_val = max(1, int(len(kept) * cfg.val_ratio))
    test, val, train = kept[:n_test], kept[n_test:n_test + n_val], kept[n_test + n_val:]

    src_vocab = Vocab.build([x[2] for x in train], cfg.min_freq)
    tgt_vocab = Vocab.build([x[3] for x in train], cfg.min_freq)

    def encode(rows):
        return [Example(en, zh, src_vocab.encode(s), tgt_vocab.encode(t)) for en, zh, s, t in rows]

    train_ex, val_ex, test_ex = encode(train), encode(val), encode(test)
    val_src_tokens = sum(len(e.src) for e in val_ex) or 1
    stats = {
        "pairs_in_file": total,
        "pairs_used": len(pairs),
        "pairs_kept": len(kept),
        "dropped_by_length": len(pairs) - len(kept),
        "train": len(train_ex),
        "val": len(val_ex),
        "test": len(test_ex),
        "src_vocab": len(src_vocab),
        "tgt_vocab": len(tgt_vocab),
        "avg_src_len": round(sum(len(e.src) for e in train_ex) / len(train_ex), 2),
        "avg_tgt_len": round(sum(len(e.tgt) for e in train_ex) / len(train_ex), 2),
        "val_src_unk_rate": round(sum(i == UNK for e in val_ex for i in e.src) / val_src_tokens, 4),
    }
    log(f"句对 {total} → 使用 {len(pairs)} → 长度过滤后 {len(kept)}；"
        f"训练/验证/测试 = {len(train_ex)}/{len(val_ex)}/{len(test_ex)}；"
        f"英文词表 {len(src_vocab)}，中文字表 {len(tgt_vocab)}")
    return Prepared(train_ex, val_ex, test_ex, src_vocab, tgt_vocab, stats)


# ---------------------------------------------------------------- batching

def make_batches(examples: list[Example], batch_size: int, rng: random.Random | None = None) -> list[list[int]]:
    """Group sentences of similar length so batches carry little padding."""
    order = list(range(len(examples)))
    if rng is not None:
        rng.shuffle(order)
    order.sort(key=lambda i: len(examples[i].src))
    chunks = [order[i:i + batch_size] for i in range(0, len(order), batch_size)]
    if rng is not None:
        rng.shuffle(chunks)
    return chunks


def _pad(rows: list[list[int]]) -> torch.Tensor:
    width = max(len(r) for r in rows)
    return torch.tensor([r + [PAD] * (width - len(r)) for r in rows], dtype=torch.long)


def src_tensor(id_lists: list[list[int]]) -> torch.Tensor:
    return _pad([ids + [EOS] for ids in id_lists])


def collate(examples: list[Example], idx: list[int], device) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Source ids, decoder input (<bos> + y) and decoder target (y + <eos>)."""
    batch = [examples[i] for i in idx]
    src = src_tensor([e.src for e in batch])
    tgt = _pad([[BOS] + e.tgt + [EOS] for e in batch])
    return src.to(device), tgt[:, :-1].to(device), tgt[:, 1:].to(device)
