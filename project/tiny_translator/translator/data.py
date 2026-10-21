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
DATA_DIR = ROOT / "data"

PAD, BOS, EOS, UNK = 0, 1, 2, 3
SPECIALS = ["<pad>", "<bos>", "<eos>", "<unk>"]

BUILTIN = {
    "cmn-eng": {
        "label": "Tatoeba 英→中句对（manythings.org，约 2.9 万对，CC-BY 2.0）",
        "url": "https://www.manythings.org/anki/cmn-eng.zip",
        "member": "cmn.txt",
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
    return DATA_DIR / f"{name}.tsv"


def _count_lines(path: Path) -> int:
    with path.open("rb") as fh:
        return sum(1 for line in fh if line.strip())


def list_datasets() -> list[dict]:
    labels = {name: spec["label"] for name, spec in BUILTIN.items()}
    if DATA_DIR.is_dir():
        for path in sorted(DATA_DIR.glob("*.tsv")):
            labels.setdefault(path.stem, f"自定义句对文件 data/{path.name}")
    out = []
    for name, label in labels.items():
        path = dataset_path(name)
        ready = path.is_file()
        out.append({"name": name, "label": label, "ready": ready, "downloadable": name in BUILTIN,
                    "pairs": _count_lines(path) if ready else None})
    return out


def download_dataset(name: str, log=print) -> int:
    spec = BUILTIN.get(name)
    if spec is None:
        raise ValueError(f"数据集 {name} 不支持自动下载，请把句对文件放到 data/{name}.tsv")
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    log(f"下载 {spec['url']} …")
    req = urllib.request.Request(spec["url"], headers={"User-Agent": "Mozilla/5.0 (tiny-translator)"})
    with urllib.request.urlopen(req, timeout=120) as resp:
        blob = resp.read()
    with zipfile.ZipFile(io.BytesIO(blob)) as zf:
        text = zf.read(spec["member"]).decode("utf-8")
    rows = []
    for line in text.splitlines():
        parts = line.split("\t")
        if len(parts) >= 2 and parts[0].strip() and parts[1].strip():
            rows.append(f"{parts[0].strip()}\t{parts[1].strip()}")
    dest = dataset_path(name)
    tmp = dest.with_suffix(".part")
    tmp.write_text("\n".join(rows) + "\n", encoding="utf-8")
    tmp.replace(dest)
    log(f"已保存 {len(rows)} 个句对到 {dest.relative_to(ROOT)}")
    return len(rows)


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
        raise FileNotFoundError(f"找不到数据集文件 {path.relative_to(ROOT)}")
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
