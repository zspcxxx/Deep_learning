"""Interactive / batch translation with a trained checkpoint."""
from __future__ import annotations

from pathlib import Path

import torch

from .config import RUNS_DIR
from .decode import beam_search, greedy_decode, ids_to_text
from .eval import load_run
from .text import normalize_en, normalize_zh


def translate_text(
    text: str,
    direction: str,
    *,
    run_dir: Path,
    beam_size: int | None = None,
    device: str = "auto",
) -> str:
    if direction not in ("en2zh", "zh2en"):
        raise ValueError("direction 必须是 en2zh 或 zh2en")
    cfg, vocab, model, device = load_run(run_dir, device=device)
    if direction == "en2zh":
        text = normalize_en(text)
    else:
        text = normalize_zh(text, simplified=cfg.zh_simplified)
    src_ids = vocab.encode_source(text, direction, cfg.max_src_len)
    src = torch.tensor([src_ids], dtype=torch.long, device=device)
    beam = beam_size if beam_size is not None else cfg.beam_size
    if beam and beam > 1:
        ids = beam_search(model, src, vocab, beam_size=beam, max_len=cfg.max_decode_len)[0]
    else:
        ids = greedy_decode(model, src, vocab, cfg.max_decode_len)[0]
    return ids_to_text(ids, vocab)


def translate_file(
    path: Path,
    direction: str,
    *,
    run_dir: Path,
    out_path: Path | None = None,
    beam_size: int | None = None,
    device: str = "auto",
) -> Path:
    lines = path.read_text(encoding="utf-8").splitlines()
    outs = [
        translate_text(line, direction, run_dir=run_dir, beam_size=beam_size, device=device)
        for line in lines
        if line.strip()
    ]
    out_path = out_path or path.with_suffix(path.suffix + f".{direction}.txt")
    out_path.write_text("\n".join(outs) + "\n", encoding="utf-8")
    print(f"写入 {out_path} （{len(outs)} 行）")
    return out_path


def resolve_run(run: str) -> Path:
    p = Path(run)
    if p.is_dir():
        return p
    return RUNS_DIR / run
