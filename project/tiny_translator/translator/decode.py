"""Greedy and beam-search decoding."""
from __future__ import annotations

import torch

from .data import BOS, EOS, PAD, src_tensor


@torch.no_grad()
def greedy(model, src: torch.Tensor, max_len: int) -> list[list[int]]:
    """Batched greedy decoding; returns token ids without <bos>/<eos>."""
    mem, src_mask = model.encode(src)
    B = src.size(0)
    ys = torch.full((B, 1), BOS, dtype=torch.long, device=src.device)
    done = torch.zeros(B, dtype=torch.bool, device=src.device)
    for _ in range(max_len):
        logits, _ = model.decode(ys, mem, src_mask)
        nxt = logits[:, -1].argmax(dim=-1).masked_fill(done, PAD)
        ys = torch.cat([ys, nxt[:, None]], dim=1)
        done |= nxt == EOS
        if bool(done.all()):
            break
    out = []
    for row in ys[:, 1:].tolist():
        out.append(row[: row.index(EOS)] if EOS in row else [t for t in row if t != PAD])
    return out


@torch.no_grad()
def beam_search(model, src: torch.Tensor, max_len: int, beam: int = 4, alpha: float = 0.6) -> list[int]:
    """Beam search for one sentence (src: [1, S]) with GNMT length normalisation."""
    mem, src_mask = model.encode(src)
    device = src.device
    alive = [([BOS], 0.0)]
    finished: list[tuple[list[int], float]] = []

    def norm(seq_len: int, score: float) -> float:
        return score / (((5 + seq_len) / 6) ** alpha)

    for _ in range(max_len):
        seqs = torch.tensor([s for s, _ in alive], dtype=torch.long, device=device)
        n = seqs.size(0)
        logits, _ = model.decode(seqs, mem.expand(n, -1, -1), src_mask.expand(n, -1, -1, -1))
        logp = logits[:, -1].log_softmax(dim=-1)
        scores = torch.tensor([s for _, s in alive], device=device)[:, None] + logp
        top = scores.view(-1).topk(min(2 * beam, scores.numel()))
        vocab = logp.size(-1)
        candidates = []
        for score, flat in zip(top.values.tolist(), top.indices.tolist()):
            row, tok = divmod(flat, vocab)
            seq = alive[row][0] + [tok]
            if tok == EOS:
                finished.append((seq, norm(len(seq) - 1, score)))
            else:
                candidates.append((seq, score))
            if len(candidates) == beam:
                break
        alive = candidates
        if not alive:
            break
        best_alive = max(norm(len(s) - 1, sc) for s, sc in alive)
        if len(finished) >= beam and max(f[1] for f in finished) >= best_alive:
            break
    if not finished:
        finished = [(s, norm(len(s) - 1, sc)) for s, sc in alive]
    seq = max(finished, key=lambda f: f[1])[0]
    return [t for t in seq[1:] if t != EOS]


def translate_ids(model, id_lists: list[list[int]], device, max_len: int, beam: int = 1,
                  batch_size: int = 128) -> list[list[int]]:
    model.eval()
    out: list[list[int]] = []
    if beam <= 1:
        order = sorted(range(len(id_lists)), key=lambda i: len(id_lists[i]))
        result: dict[int, list[int]] = {}
        for start in range(0, len(order), batch_size):
            idx = order[start:start + batch_size]
            src = src_tensor([id_lists[i] for i in idx]).to(device)
            for i, ids in zip(idx, greedy(model, src, max_len)):
                result[i] = ids
        return [result[i] for i in range(len(id_lists))]
    for ids in id_lists:
        out.append(beam_search(model, src_tensor([ids]).to(device), max_len, beam))
    return out


@torch.no_grad()
def cross_attention(model, src_ids: list[int], out_ids: list[int], device) -> list[list[float]]:
    """Last decoder layer's cross-attention: one row per output token over source tokens + <eos>."""
    model.eval()
    src = src_tensor([src_ids]).to(device)
    mem, src_mask = model.encode(src)
    tgt_in = torch.tensor([[BOS] + out_ids], dtype=torch.long, device=device)
    _, weights = model.decode(tgt_in, mem, src_mask, need_weights=True)
    return weights[0, : len(out_ids)].tolist()
