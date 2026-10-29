"""Encoder–Decoder Transformer for bilingual MT."""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 512, dropout: float = 0.1):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        pe = torch.zeros(max_len, d_model)
        pos = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer("pe", pe.unsqueeze(0), persistent=False)  # (1, L, D)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[:, : x.size(1)]
        return self.dropout(x)


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float = 0.1):
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        *,
        key_padding_mask: torch.Tensor | None = None,
        attn_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, Tq, _ = q.shape
        Tk = k.size(1)
        q = self.q_proj(q).view(B, Tq, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(k).view(B, Tk, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(v).view(B, Tk, self.n_heads, self.d_k).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)  # (B,H,Tq,Tk)
        if attn_mask is not None:
            scores = scores + attn_mask
        if key_padding_mask is not None:
            # key_padding_mask: (B, Tk) True = pad
            scores = scores.masked_fill(key_padding_mask[:, None, None, :], float("-inf"))
        weights = F.softmax(scores, dim=-1)
        weights = torch.nan_to_num(weights, nan=0.0)
        weights = self.dropout(weights)
        out = torch.matmul(weights, v)  # (B,H,Tq,d_k)
        out = out.transpose(1, 2).contiguous().view(B, Tq, -1)
        return self.out_proj(out)


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.1, activation: str = "gelu"):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        self.act = F.gelu if activation == "gelu" else F.relu

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.dropout(self.act(self.fc1(x))))


class EncoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        norm_first: bool,
        activation: str,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(self, x: torch.Tensor, key_padding_mask: torch.Tensor | None) -> torch.Tensor:
        if self.norm_first:
            h = self.norm1(x)
            x = x + self.drop1(self.self_attn(h, h, h, key_padding_mask=key_padding_mask))
            h = self.norm2(x)
            x = x + self.drop2(self.ff(h))
        else:
            x = self.norm1(x + self.drop1(self.self_attn(x, x, x, key_padding_mask=key_padding_mask)))
            x = self.norm2(x + self.drop2(self.ff(x)))
        return x


class DecoderLayer(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        norm_first: bool,
        activation: str,
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.cross_attn = MultiHeadAttention(d_model, n_heads, dropout)
        self.ff = FeedForward(d_model, d_ff, dropout, activation)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.norm3 = nn.LayerNorm(d_model)
        self.drop1 = nn.Dropout(dropout)
        self.drop2 = nn.Dropout(dropout)
        self.drop3 = nn.Dropout(dropout)
        self.norm_first = norm_first

    def forward(
        self,
        x: torch.Tensor,
        memory: torch.Tensor,
        *,
        tgt_key_padding_mask: torch.Tensor | None,
        memory_key_padding_mask: torch.Tensor | None,
        causal_mask: torch.Tensor,
    ) -> torch.Tensor:
        if self.norm_first:
            h = self.norm1(x)
            x = x + self.drop1(
                self.self_attn(h, h, h, key_padding_mask=tgt_key_padding_mask, attn_mask=causal_mask)
            )
            h = self.norm2(x)
            x = x + self.drop2(
                self.cross_attn(h, memory, memory, key_padding_mask=memory_key_padding_mask)
            )
            h = self.norm3(x)
            x = x + self.drop3(self.ff(h))
        else:
            x = self.norm1(
                x
                + self.drop1(
                    self.self_attn(x, x, x, key_padding_mask=tgt_key_padding_mask, attn_mask=causal_mask)
                )
            )
            x = self.norm2(
                x
                + self.drop2(
                    self.cross_attn(x, memory, memory, key_padding_mask=memory_key_padding_mask)
                )
            )
            x = self.norm3(x + self.drop3(self.ff(x)))
        return x


def _causal_mask(sz: int, device: torch.device) -> torch.Tensor:
    # Additive mask: 0 allow, -inf block future.
    mask = torch.triu(torch.full((sz, sz), float("-inf"), device=device), diagonal=1)
    return mask  # (T, T) broadcast over (B, H, T, T)


class TransformerMT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        d_model: int = 512,
        n_heads: int = 8,
        n_enc_layers: int = 6,
        n_dec_layers: int = 6,
        d_ff: int = 2048,
        dropout: float = 0.1,
        max_len: int = 256,
        pad_id: int = 0,
        norm_first: bool = True,
        activation: str = "gelu",
        tie_embeddings: bool = True,
    ):
        super().__init__()
        self.pad_id = pad_id
        self.d_model = d_model
        self.emb = nn.Embedding(vocab_size, d_model, padding_idx=pad_id)
        self.pos = PositionalEncoding(d_model, max_len=max_len, dropout=dropout)
        self.encoder = nn.ModuleList(
            [
                EncoderLayer(d_model, n_heads, d_ff, dropout, norm_first, activation)
                for _ in range(n_enc_layers)
            ]
        )
        self.decoder = nn.ModuleList(
            [
                DecoderLayer(d_model, n_heads, d_ff, dropout, norm_first, activation)
                for _ in range(n_dec_layers)
            ]
        )
        self.enc_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.dec_norm = nn.LayerNorm(d_model) if norm_first else nn.Identity()
        self.out_proj = nn.Linear(d_model, vocab_size, bias=False)
        if tie_embeddings:
            self.out_proj.weight = self.emb.weight
        self._reset()

    def _reset(self) -> None:
        nn.init.normal_(self.emb.weight, mean=0.0, std=self.d_model**-0.5)
        if self.pad_id is not None:
            with torch.no_grad():
                self.emb.weight[self.pad_id].zero_()

    def encode(self, src: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # src: (B, S)
        pad_mask = src.eq(self.pad_id)
        x = self.pos(self.emb(src) * math.sqrt(self.d_model))
        for layer in self.encoder:
            x = layer(x, key_padding_mask=pad_mask)
        return self.enc_norm(x), pad_mask

    def decode(
        self,
        tgt_in: torch.Tensor,
        memory: torch.Tensor,
        memory_pad: torch.Tensor,
    ) -> torch.Tensor:
        # tgt_in: (B, T) — usually [bos] + tokens (no final eos for teacher-forcing input)
        pad_mask = tgt_in.eq(self.pad_id)
        causal = _causal_mask(tgt_in.size(1), tgt_in.device)
        x = self.pos(self.emb(tgt_in) * math.sqrt(self.d_model))
        for layer in self.decoder:
            x = layer(
                x,
                memory,
                tgt_key_padding_mask=pad_mask,
                memory_key_padding_mask=memory_pad,
                causal_mask=causal,
            )
        x = self.dec_norm(x)
        return self.out_proj(x)  # (B, T, V)

    def forward(self, src: torch.Tensor, tgt_in: torch.Tensor) -> torch.Tensor:
        memory, mem_pad = self.encode(src)
        return self.decode(tgt_in, memory, mem_pad)

    def num_parameters(self, trainable_only: bool = True) -> int:
        params = self.parameters() if not trainable_only else (p for p in self.parameters() if p.requires_grad)
        return sum(p.numel() for p in params)


def build_model(cfg, vocab_size: int, pad_id: int) -> TransformerMT:
    return TransformerMT(
        vocab_size=vocab_size,
        d_model=cfg.d_model,
        n_heads=cfg.n_heads,
        n_enc_layers=cfg.n_enc_layers,
        n_dec_layers=cfg.n_dec_layers,
        d_ff=cfg.d_ff,
        dropout=cfg.dropout,
        max_len=max(cfg.max_src_len, cfg.max_tgt_len, cfg.max_decode_len) + 8,
        pad_id=pad_id,
        norm_first=cfg.norm_first,
        activation=cfg.activation,
        tie_embeddings=cfg.tie_embeddings,
    )
