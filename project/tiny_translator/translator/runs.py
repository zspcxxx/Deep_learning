"""Run directories on disk, and loading trained models for translation and evaluation."""
from __future__ import annotations

import json
import re
import shutil
import threading
import time
from collections import OrderedDict
from pathlib import Path

import torch

from .config import TrainConfig
from .data import ROOT, UNK, Vocab, detokenize_zh, read_pairs, tokenize_en, tokenize_zh
from .decode import cross_attention, translate_ids
from .metrics import corpus_bleu, sentence_bleu
from .model import TinyTransformer

RUNS_DIR = ROOT / "runs"
RUN_ID_RE = re.compile(r"^[\w.-]+$")


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


def write_json(path: Path, data) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    tmp.replace(path)


def new_run_dir(name: str) -> Path:
    safe = re.sub(r"[^\w-]+", "-", name).strip("-")[:40] or "run"
    path = RUNS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}_{safe}"
    path.mkdir(parents=True, exist_ok=False)
    return path


def run_dir(run_id: str) -> Path:
    if not RUN_ID_RE.match(run_id):
        raise KeyError(run_id)
    path = RUNS_DIR / run_id
    if not path.is_dir():
        raise KeyError(run_id)
    return path


def list_runs(active_id: str | None = None) -> list[dict]:
    if not RUNS_DIR.is_dir():
        return []
    out = []
    for path in sorted(RUNS_DIR.iterdir(), reverse=True):
        if not path.is_dir():
            continue
        summary = read_json(path / "summary.json", {})
        cfg = read_json(path / "config.json", {})
        status = summary.get("status", "unknown")
        if status in {"preparing", "running", "stopping"} and path.name != active_id:
            status = "interrupted"
        out.append({
            "id": path.name,
            "name": cfg.get("name", path.name),
            "status": status,
            "created": summary.get("started_at"),
            "epochs_done": summary.get("epochs_done", 0),
            "epochs": cfg.get("epochs"),
            "best_val_bleu": summary.get("best_val_bleu"),
            "test_bleu": summary.get("test_bleu"),
            "params": summary.get("params"),
            "has_checkpoint": (path / "best.pt").is_file(),
        })
    return out


def run_detail(run_id: str) -> dict:
    path = run_dir(run_id)
    return {
        "id": run_id,
        "config": read_json(path / "config.json", {}),
        "summary": read_json(path / "summary.json", {}),
        "history": read_json(path / "history.json", {"epochs": [], "steps": []}),
    }


def delete_run(run_id: str) -> None:
    path = run_dir(run_id)
    with _cache_lock:
        for key in [k for k in _cache if k[0] == run_id]:
            del _cache[key]
    shutil.rmtree(path)


# ---------------------------------------------------------------- inference

def resolve_device(name: str) -> torch.device:
    if name == "cpu":
        return torch.device("cpu")
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("当前环境没有可用的 CUDA GPU")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Translator:
    def __init__(self, path: Path, which: str = "best"):
        self.path = path
        self.cfg = TrainConfig(**read_json(path / "config.json"))
        self.src_vocab = Vocab(read_json(path / "vocab_src.json"))
        self.tgt_vocab = Vocab(read_json(path / "vocab_tgt.json"))
        self.device = resolve_device(self.cfg.device)
        self.model = TinyTransformer(len(self.src_vocab), len(self.tgt_vocab), self.cfg).to(self.device)
        ckpt = torch.load(path / f"{which}.pt", map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["model"])
        self.model.eval()
        self.checkpoint = {k: v for k, v in ckpt.items() if k != "model"}
        self.max_len = self.cfg.max_tgt_len + 10
        self.lock = threading.Lock()

    def translate(self, text: str, beam: int = 1) -> dict:
        tokens = tokenize_en(text)
        if not tokens:
            raise ValueError("请输入英文句子")
        ids = self.src_vocab.encode(tokens)
        t0 = time.perf_counter()
        with self.lock:
            out_ids = translate_ids(self.model, [ids], self.device, self.max_len, beam)[0]
            attn = cross_attention(self.model, ids, out_ids, self.device) if out_ids else []
        out_tokens = self.tgt_vocab.decode(out_ids)
        return {
            "input": text,
            "src_tokens": tokens,
            "src_unk": [i == UNK for i in ids],
            "out_tokens": out_tokens,
            "output": detokenize_zh(out_tokens),
            "attention": attn,
            "elapsed_ms": round(1000 * (time.perf_counter() - t0), 1),
        }

    def evaluate(self, split: str = "test", limit: int = 0, beam: int = 1) -> dict:
        pairs = read_pairs(self.path / f"{split}.tsv")
        if limit:
            pairs = pairs[:limit]
        src_ids = [self.src_vocab.encode(tokenize_en(en)) for en, _ in pairs]
        t0 = time.perf_counter()
        with self.lock:
            outs = translate_ids(self.model, src_ids, self.device, self.max_len, beam)
        elapsed = time.perf_counter() - t0
        hyps = [self.tgt_vocab.decode(o) for o in outs]
        refs = [tokenize_zh(zh) for _, zh in pairs]
        score = corpus_bleu(hyps, refs)
        samples = [
            {"en": en, "ref": zh, "hyp": detokenize_zh(h), "bleu": sentence_bleu(h, r)}
            for (en, zh), h, r in zip(pairs, hyps, refs)
        ]
        exact = sum(detokenize_zh(h) == zh for (_, zh), h in zip(pairs, hyps))
        unk = sum(i == UNK for ids in src_ids for i in ids)
        return {
            "split": split, "beam": beam, "count": len(pairs), **score,
            "exact_match": round(100 * exact / max(1, len(pairs)), 2),
            "src_unk_rate": round(100 * unk / max(1, sum(map(len, src_ids))), 2),
            "seconds": round(elapsed, 2), "samples": samples,
        }


_cache: OrderedDict[tuple[str, str], Translator] = OrderedDict()
_cache_lock = threading.Lock()


def load_translator(run_id: str, which: str = "best") -> Translator:
    if which not in {"best", "last"}:
        raise KeyError(which)
    path = run_dir(run_id)
    ckpt = path / f"{which}.pt"
    if not ckpt.is_file():
        raise FileNotFoundError(f"{run_id} 还没有 {which}.pt 检查点")
    key = (run_id, which)
    with _cache_lock:
        cached = _cache.get(key)
        if cached is not None and cached.checkpoint.get("mtime") == ckpt.stat().st_mtime:
            _cache.move_to_end(key)
            return cached
        translator = Translator(path, which)
        translator.checkpoint["mtime"] = ckpt.stat().st_mtime
        _cache[key] = translator
        while len(_cache) > 3:
            _cache.popitem(last=False)
        return translator
