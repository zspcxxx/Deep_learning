"""Greedy and beam-search decoding."""
from __future__ import annotations

import torch
import torch.nn.functional as F

from .model import TransformerMT
from .vocab import Vocab


@torch.no_grad()
def greedy_decode(
    model: TransformerMT,
    src: torch.Tensor,
    vocab: Vocab,
    max_len: int,
) -> list[list[int]]:
    model.eval()
    memory, mem_pad = model.encode(src)
    B = src.size(0)
    device = src.device
    ys = torch.full((B, 1), vocab.bos_id, dtype=torch.long, device=device)
    finished = torch.zeros(B, dtype=torch.bool, device=device)
    for _ in range(max_len - 1):
        logits = model.decode(ys, memory, mem_pad)[:, -1, :]  # (B, V)
        next_id = logits.argmax(dim=-1)
        next_id = torch.where(finished, torch.full_like(next_id, vocab.pad_id), next_id)
        ys = torch.cat([ys, next_id.unsqueeze(1)], dim=1)
        finished = finished | next_id.eq(vocab.eos_id)
        if bool(finished.all()):
            break
    return ys.tolist()


@torch.no_grad()
def beam_search(
    model: TransformerMT,
    src: torch.Tensor,
    vocab: Vocab,
    *,
    beam_size: int = 4,
    max_len: int = 80,
    length_penalty: float = 0.6,
) -> list[list[int]]:
    """Per-example beam search (batch size can be >1; beams expand per item)."""
    model.eval()
    results: list[list[int]] = []
    for i in range(src.size(0)):
        results.append(
            _beam_one(
                model,
                src[i : i + 1],
                vocab,
                beam_size=beam_size,
                max_len=max_len,
                length_penalty=length_penalty,
            )
        )
    return results


def _beam_one(
    model: TransformerMT,
    src: torch.Tensor,
    vocab: Vocab,
    *,
    beam_size: int,
    max_len: int,
    length_penalty: float,
) -> list[int]:
    memory, mem_pad = model.encode(src)
    device = src.device
    # Each hypothesis: (score, token_ids tensor of shape (1, T))
    beams: list[tuple[float, torch.Tensor]] = [(0.0, torch.tensor([[vocab.bos_id]], device=device))]
    finished: list[tuple[float, torch.Tensor]] = []

    for _ in range(max_len - 1):
        if not beams:
            break
        # Expand all live beams in one forward by stacking.
        scores = []
        seqs = []
        for sc, seq in beams:
            scores.append(sc)
            seqs.append(seq)
        # Pad to same length (they share length in sync decoding).
        T = seqs[0].size(1)
        batch_seq = torch.cat(seqs, dim=0)  # (B, T)
        mem = memory.expand(batch_seq.size(0), -1, -1)
        mpad = mem_pad.expand(batch_seq.size(0), -1)
        logits = model.decode(batch_seq, mem, mpad)[:, -1, :]  # (B, V)
        logp = F.log_softmax(logits, dim=-1)
        topv, topi = logp.topk(beam_size, dim=-1)

        candidates: list[tuple[float, torch.Tensor]] = []
        for b in range(batch_seq.size(0)):
            base = scores[b]
            for k in range(beam_size):
                tok = int(topi[b, k].item())
                new_sc = base + float(topv[b, k].item())
                new_seq = torch.cat(
                    [batch_seq[b : b + 1], topi[b : b + 1, k : k + 1]], dim=1
                )
                if tok == vocab.eos_id:
                    # Normalize by length
                    lp = ((5 + new_seq.size(1)) / 6) ** length_penalty
                    finished.append((new_sc / lp, new_seq))
                else:
                    candidates.append((new_sc, new_seq))

        candidates.sort(key=lambda x: x[0], reverse=True)
        beams = candidates[:beam_size]
        if len(finished) >= beam_size:
            # Early stop if enough finished and best finished beats best live
            finished.sort(key=lambda x: x[0], reverse=True)
            if beams and finished[0][0] >= beams[0][0]:
                break

    if not finished:
        finished = beams
    finished.sort(key=lambda x: x[0], reverse=True)
    best = finished[0][1].squeeze(0).tolist()
    return best


def ids_to_text(ids: list[int], vocab: Vocab) -> str:
    # Drop leading BOS if present; stop at EOS.
    if ids and ids[0] == vocab.bos_id:
        ids = ids[1:]
    out: list[int] = []
    for i in ids:
        if i == vocab.eos_id:
            break
        if i == vocab.pad_id:
            continue
        out.append(i)
    return vocab.decode(out)
