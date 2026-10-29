"""PyTorch Dataset / collate for bilingual MT."""
from __future__ import annotations

import torch
from torch.utils.data import Dataset

from .data import Pair
from .vocab import Vocab


class MTDataset(Dataset):
    def __init__(
        self,
        pairs: list[Pair],
        vocab: Vocab,
        max_src_len: int,
        max_tgt_len: int,
    ):
        self.pairs = pairs
        self.vocab = vocab
        self.max_src_len = max_src_len
        self.max_tgt_len = max_tgt_len
        # Filter pairs that become empty after encoding.
        self._kept: list[Pair] = []
        for p in pairs:
            src = vocab.encode_source(p.src, p.direction, max_src_len)
            tgt = vocab.encode_target(p.tgt, max_tgt_len)
            if len(src) >= 2 and len(tgt) >= 3:
                self._kept.append(p)
        self.pairs = self._kept

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        p = self.pairs[idx]
        src = self.vocab.encode_source(p.src, p.direction, self.max_src_len)
        tgt = self.vocab.encode_target(p.tgt, self.max_tgt_len)
        # Teacher forcing: input = tgt[:-1], label = tgt[1:]
        return {
            "src": torch.tensor(src, dtype=torch.long),
            "tgt_in": torch.tensor(tgt[:-1], dtype=torch.long),
            "tgt_out": torch.tensor(tgt[1:], dtype=torch.long),
            "direction": p.direction,
            "src_text": p.src,
            "tgt_text": p.tgt,
        }


def collate_batch(batch: list[dict], pad_id: int) -> dict:
    def pad_stack(key: str) -> torch.Tensor:
        seqs = [b[key] for b in batch]
        T = max(s.size(0) for s in seqs)
        out = torch.full((len(seqs), T), pad_id, dtype=torch.long)
        for i, s in enumerate(seqs):
            out[i, : s.size(0)] = s
        return out

    return {
        "src": pad_stack("src"),
        "tgt_in": pad_stack("tgt_in"),
        "tgt_out": pad_stack("tgt_out"),
        "direction": [b["direction"] for b in batch],
        "src_text": [b["src_text"] for b in batch],
        "tgt_text": [b["tgt_text"] for b in batch],
    }
