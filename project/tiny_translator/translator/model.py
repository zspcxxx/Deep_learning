"""A small encoder-decoder Transformer, written out layer by layer."""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import nn

from .data import PAD


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512):
        super().__init__()
        pos = torch.arange(max_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[: x.size(1)]


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float):
        super().__init__()
        self.h = n_heads
        self.dk = d_model // n_heads
        self.q = nn.Linear(d_model, d_model)
        self.k = nn.Linear(d_model, d_model)
        self.v = nn.Linear(d_model, d_model)
        self.o = nn.Linear(d_model, d_model)
        self.drop = nn.Dropout(dropout)

    def forward(self, x, mem, mask, need_weights: bool = False):
        """x: [B, Tq, D] queries; mem: [B, Tk, D] keys/values; mask: bool, True = may attend."""
        B, Tq, _ = x.shape
        Tk = mem.size(1)
        q = self.q(x).view(B, Tq, self.h, self.dk).transpose(1, 2)
        k = self.k(mem).view(B, Tk, self.h, self.dk).transpose(1, 2)
        v = self.v(mem).view(B, Tk, self.h, self.dk).transpose(1, 2)
        scores = q @ k.transpose(-2, -1) / math.sqrt(self.dk)
        if mask is not None:
            scores = scores.masked_fill(~mask, float("-inf"))
        attn = scores.softmax(dim=-1)
        out = (self.drop(attn) @ v).transpose(1, 2).reshape(B, Tq, self.h * self.dk)
        return self.o(out), (attn if need_weights else None)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float, activation: str):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.act = F.gelu if activation == "gelu" else F.relu
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        return self.fc2(self.drop(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation, norm_first):
        super().__init__()
        self.attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x, mask):
        if self.norm_first:
            h = self.n1(x)
            x = x + self.drop(self.attn(h, h, mask)[0])
            return x + self.drop(self.ffn(self.n2(x)))
        x = self.n1(x + self.drop(self.attn(x, x, mask)[0]))
        return self.n2(x + self.drop(self.ffn(x)))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, dropout, activation, norm_first):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ffn = FeedForward(d_model, d_ff, dropout, activation)
        self.n1 = nn.LayerNorm(d_model)
        self.n2 = nn.LayerNorm(d_model)
        self.n3 = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, y, mem, self_mask, cross_mask, need_weights=False):
        if self.norm_first:
            h = self.n1(y)
            y = y + self.drop(self.self_attn(h, h, self_mask)[0])
            out, w = self.cross_attn(self.n2(y), mem, cross_mask, need_weights)
            y = y + self.drop(out)
            return y + self.drop(self.ffn(self.n3(y))), w
        y = self.n1(y + self.drop(self.self_attn(y, y, self_mask)[0]))
        out, w = self.cross_attn(y, mem, cross_mask, need_weights)
        y = self.n2(y + self.drop(out))
        return self.n3(y + self.drop(self.ffn(y))), w


class TinyTransformer(nn.Module):
    def __init__(self, src_vocab: int, tgt_vocab: int, cfg):
        super().__init__()
        d = cfg.d_model
        self.d_model = d
        self.src_emb = nn.Embedding(src_vocab, d, padding_idx=PAD)
        self.tgt_emb = nn.Embedding(tgt_vocab, d, padding_idx=PAD)
        self.pos = PositionalEncoding(d, max_len=max(256, cfg.max_src_len + 8, cfg.max_tgt_len + 16))
        self.drop = nn.Dropout(cfg.dropout)
        args = (d, cfg.n_heads, cfg.d_ff, cfg.dropout, cfg.activation, cfg.norm_first)
        self.encoder = nn.ModuleList(EncoderLayer(*args) for _ in range(cfg.n_enc_layers))
        self.decoder = nn.ModuleList(DecoderLayer(*args) for _ in range(cfg.n_dec_layers))
        # Pre-LN leaves the residual stream unnormalised, so it needs a final LayerNorm.
        self.enc_norm = nn.LayerNorm(d) if cfg.norm_first else nn.Identity()
        self.dec_norm = nn.LayerNorm(d) if cfg.norm_first else nn.Identity()
        self.out = nn.Linear(d, tgt_vocab, bias=False)
        self._init_weights()
        if cfg.tie_embeddings:
            self.out.weight = self.tgt_emb.weight

    def _init_weights(self):
        for name, p in self.named_parameters():
            if p.dim() > 1 and "emb" not in name:
                nn.init.xavier_uniform_(p)
            elif "bias" in name:
                nn.init.zeros_(p)
        for emb in (self.src_emb, self.tgt_emb):
            nn.init.normal_(emb.weight, mean=0.0, std=self.d_model ** -0.5)
            with torch.no_grad():
                emb.weight[PAD].zero_()

    def num_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @staticmethod
    def causal_mask(t: int, device) -> torch.Tensor:
        return torch.tril(torch.ones(t, t, dtype=torch.bool, device=device))

    def encode(self, src: torch.Tensor):
        src_mask = (src != PAD)[:, None, None, :]
        x = self.drop(self.pos(self.src_emb(src) * math.sqrt(self.d_model)))
        for layer in self.encoder:
            x = layer(x, src_mask)
        return self.enc_norm(x), src_mask

    def decode(self, tgt_in, mem, src_mask, need_weights: bool = False):
        """Returns logits [B, T, V] and, if asked, last-layer cross-attention averaged over heads [B, T, S]."""
        T = tgt_in.size(1)
        self_mask = self.causal_mask(T, tgt_in.device)[None, None] & (tgt_in != PAD)[:, None, None, :]
        y = self.drop(self.pos(self.tgt_emb(tgt_in) * math.sqrt(self.d_model)))
        weights = None
        for i, layer in enumerate(self.decoder):
            last = i == len(self.decoder) - 1
            y, w = layer(y, mem, self_mask, src_mask, need_weights and last)
            if w is not None:
                weights = w.mean(dim=1)
        return self.out(self.dec_norm(y)), weights

    def forward(self, src, tgt_in):
        mem, src_mask = self.encode(src)
        return self.decode(tgt_in, mem, src_mask)[0]
