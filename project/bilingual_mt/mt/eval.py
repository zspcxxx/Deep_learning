"""Evaluate a checkpoint on val/test with BLEU."""
from __future__ import annotations

import json
from functools import partial
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from .config import RUNS_DIR, resolve_device
from .data import read_split_tsv
from .dataset import MTDataset, collate_batch
from .decode import beam_search, greedy_decode, ids_to_text
from .metrics import bleu_for_direction
from .model import build_model
from .vocab import Vocab
from .config import TrainConfig


def load_run(run_dir: Path, device: str = "auto"):
    run_dir = Path(run_dir)
    cfg_path = run_dir / "config.json"
    cfg = TrainConfig(**json.loads(cfg_path.read_text(encoding="utf-8")))
    device = resolve_device(device if device != "auto" else cfg.device)
    spm = Path(cfg.spm_model) if cfg.spm_model else run_dir / "spm.model"
    vocab = Vocab(spm)
    model = build_model(cfg, len(vocab), vocab.pad_id)
    ckpt_file = run_dir / "best.pt"
    if not ckpt_file.is_file():
        ckpt_file = run_dir / "checkpoint.pt"
    state = torch.load(ckpt_file, map_location=device)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return cfg, vocab, model, device


@torch.no_grad()
def evaluate_split(
    run_dir: Path,
    split: str = "test",
    *,
    beam_size: int | None = None,
    device: str = "auto",
    max_examples: int = 0,
) -> dict:
    cfg, vocab, model, device = load_run(run_dir, device=device)
    pairs = read_split_tsv(run_dir / f"{split}.tsv")
    if max_examples:
        pairs = pairs[:max_examples]
    ds = MTDataset(pairs, vocab, cfg.max_src_len, cfg.max_tgt_len)
    loader = DataLoader(
        ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        collate_fn=partial(collate_batch, pad_id=vocab.pad_id),
    )
    beam = beam_size if beam_size is not None else cfg.beam_size
    hyps_by_dir: dict[str, list[str]] = {"en2zh": [], "zh2en": []}
    refs_by_dir: dict[str, list[str]] = {"en2zh": [], "zh2en": []}

    for batch in loader:
        src = batch["src"].to(device)
        if beam and beam > 1:
            pred_ids = beam_search(
                model, src, vocab, beam_size=beam, max_len=cfg.max_decode_len
            )
        else:
            pred_ids = greedy_decode(model, src, vocab, cfg.max_decode_len)
        for ids, ref, direction in zip(pred_ids, batch["tgt_text"], batch["direction"]):
            hyps_by_dir[direction].append(ids_to_text(ids, vocab))
            refs_by_dir[direction].append(ref)

    out: dict = {"split": split, "n": len(ds)}
    bleus = []
    for d in ("en2zh", "zh2en"):
        if hyps_by_dir[d]:
            b = bleu_for_direction(hyps_by_dir[d], refs_by_dir[d], d)
            out[f"bleu_{d}"] = b
            out[f"n_{d}"] = len(hyps_by_dir[d])
            bleus.append(b)
            # Save sample predictions
            samples = list(zip(hyps_by_dir[d][:20], refs_by_dir[d][:20]))
            out[f"samples_{d}"] = [{"hyp": h, "ref": r} for h, r in samples]
    out["bleu"] = sum(bleus) / len(bleus) if bleus else 0.0
    out_path = Path(run_dir) / f"eval_{split}.json"
    out_path.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: v for k, v in out.items() if not k.startswith("samples")}, ensure_ascii=False, indent=2))
    return out


def evaluate_run(run_name: str, split: str = "test", **kwargs) -> dict:
    return evaluate_split(RUNS_DIR / run_name, split=split, **kwargs)
