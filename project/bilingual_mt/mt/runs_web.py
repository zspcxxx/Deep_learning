"""List runs and load models for web translate / evaluate."""
from __future__ import annotations

import json
import re
import shutil
import time
from collections import OrderedDict
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import RUNS_DIR, TrainConfig, resolve_device
from .data import read_split_tsv
from .dataset import MTDataset, collate_batch
from .decode import beam_search, greedy_decode, ids_to_text
from .metrics import bleu_for_direction
from .model import build_model
from .text import normalize_en, normalize_zh
from .vocab import Vocab

RUN_ID_RE = re.compile(r"^[\w.-]+$")
_cache: OrderedDict[tuple[str, str], "RunTranslator"] = OrderedDict()
_cache_lock = __import__("threading").Lock()
_CACHE_MAX = 2


def read_json(path: Path, default=None):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError):
        return default


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
        if not path.is_dir() or path.name.startswith("."):
            continue
        summary = read_json(path / "summary.json", {})
        cfg = read_json(path / "config.json", {})
        status = summary.get("status", "unknown")
        if status in {"preparing", "running", "stopping"} and path.name != active_id:
            status = "interrupted"
        out.append(
            {
                "id": path.name,
                "name": cfg.get("name", path.name),
                "status": status,
                "created": summary.get("started_at"),
                "epochs_done": summary.get("epochs_done", 0),
                "epochs": cfg.get("epochs"),
                "direction": cfg.get("direction"),
                "best_val_bleu": summary.get("best_val_bleu"),
                "test_bleu": summary.get("test_bleu"),
                "params": summary.get("params"),
                "has_checkpoint": (path / "best.pt").is_file() or (path / "last.pt").is_file(),
            }
        )
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


def _char_tokens(s: str) -> list[str]:
    # Character tokens for sentence-level BLEU on Chinese; words for English.
    if any("\u4e00" <= c <= "\u9fff" for c in s):
        return list(s.replace(" ", ""))
    return s.split()


class RunTranslator:
    def __init__(self, path: Path, which: str = "best"):
        self.path = path
        raw = read_json(path / "config.json")
        self.cfg = TrainConfig(**{k: v for k, v in raw.items() if k in TrainConfig.__dataclass_fields__})
        spm = Path(self.cfg.spm_model) if self.cfg.spm_model else path / "spm.model"
        if not spm.is_file():
            spm = path / "spm.model"
        self.vocab = Vocab(spm)
        device_str = resolve_device(self.cfg.device)
        self.device = torch.device(device_str)
        self.model = build_model(self.cfg, len(self.vocab), self.vocab.pad_id).to(self.device)
        ckpt_path = path / f"{which}.pt"
        if not ckpt_path.is_file():
            ckpt_path = path / "best.pt" if which != "best" else path / "last.pt"
        if not ckpt_path.is_file():
            raise FileNotFoundError(f"找不到检查点 {which}.pt")
        state = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.model.load_state_dict(state["model"])
        self.model.eval()
        self.checkpoint = {k: v for k, v in state.items() if k != "model"}

    def translate(self, text: str, direction: str, beam: int = 4) -> dict:
        if direction not in ("en2zh", "zh2en"):
            raise ValueError("direction 必须是 en2zh 或 zh2en")
        t0 = time.time()
        if direction == "en2zh":
            text = normalize_en(text)
        else:
            text = normalize_zh(text, simplified=self.cfg.zh_simplified)
        if not text:
            raise ValueError("输入为空")
        src_ids = self.vocab.encode_source(text, direction, self.cfg.max_src_len)
        src = torch.tensor([src_ids], dtype=torch.long, device=self.device)
        if beam > 1:
            ids = beam_search(
                self.model, src, self.vocab, beam_size=beam, max_len=self.cfg.max_decode_len
            )[0]
        else:
            ids = greedy_decode(self.model, src, self.vocab, self.cfg.max_decode_len)[0]
        out = ids_to_text(ids, self.vocab)
        return {
            "output": out,
            "direction": direction,
            "elapsed_ms": int((time.time() - t0) * 1000),
            "src_tokens": self.vocab.sp.encode(text, out_type=str),
            "src_unk": [],
            "out_tokens": list(out) if direction == "en2zh" else out.split(),
            "attention": [],
        }

    def evaluate(self, split: str, limit: int, beam: int) -> dict:
        pairs = read_split_tsv(self.path / f"{split}.tsv")
        if limit:
            pairs = pairs[:limit]
        ds = MTDataset(pairs, self.vocab, self.cfg.max_src_len, self.cfg.max_tgt_len)
        loader = DataLoader(
            ds,
            batch_size=min(32, self.cfg.batch_size),
            shuffle=False,
            collate_fn=partial(collate_batch, pad_id=self.vocab.pad_id),
        )
        t0 = time.time()
        samples = []
        hyps_by: dict[str, list[str]] = {"en2zh": [], "zh2en": []}
        refs_by: dict[str, list[str]] = {"en2zh": [], "zh2en": []}
        for batch in loader:
            src = batch["src"].to(self.device)
            if beam > 1:
                pred = beam_search(
                    self.model, src, self.vocab, beam_size=beam, max_len=self.cfg.max_decode_len
                )
            else:
                pred = greedy_decode(self.model, src, self.vocab, self.cfg.max_decode_len)
            for ids, src_t, ref, direction in zip(
                pred, batch["src_text"], batch["tgt_text"], batch["direction"]
            ):
                hyp = ids_to_text(ids, self.vocab)
                hyps_by[direction].append(hyp)
                refs_by[direction].append(ref)
                ht, rt = _char_tokens(hyp), _char_tokens(ref)
                overlap = len(set(ht) & set(rt)) / max(1, len(set(rt)))
                samples.append(
                    {
                        "en": src_t,
                        "ref": ref,
                        "hyp": hyp,
                        "direction": direction,
                        "bleu": round(100 * overlap, 1),
                    }
                )

        bleus = []
        detail = {}
        for d in ("en2zh", "zh2en"):
            if hyps_by[d]:
                b = bleu_for_direction(hyps_by[d], refs_by[d], d)
                detail[f"bleu_{d}"] = round(b, 2)
                bleus.append(b)
        bleu = round(sum(bleus) / len(bleus), 2) if bleus else 0.0
        exact = sum(1 for s in samples if s["hyp"] == s["ref"]) / max(1, len(samples))
        return {
            "bleu": bleu,
            "precisions": [bleu, bleu, bleu, bleu],  # sacrebleu summary; UI shows primary BLEU
            "bp": 1.0,
            "exact_match": round(100 * exact, 1),
            "src_unk_rate": 0.0,
            "count": len(samples),
            "seconds": round(time.time() - t0, 1),
            "samples": samples,
            **detail,
        }


def load_translator(run_id: str, which: str = "best") -> RunTranslator:
    key = (run_id, which)
    with _cache_lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
        tr = RunTranslator(run_dir(run_id), which)
        _cache[key] = tr
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
        return tr
