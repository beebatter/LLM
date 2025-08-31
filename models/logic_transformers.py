"""
Lightweight Transformer models for clause scoring and (optional) bi-encoding.

Primary target: Passive Scores Mode (PSM) in iProver interactive mode,
scoring clauses independently from conjecture to be used as external_score.

Secondary (optional): Bi-encoder for query/clause if conjecture text is available.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import math
import torch
import torch.nn as nn


def _default_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 4096):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, D]
        l = x.size(1)
        return x + self.pe[:, :l]


@dataclass
class TransformerConfig:
    vocab_size: int
    d_model: int = 256
    n_heads: int = 4
    n_layers: int = 4
    d_ff: int = 512
    dropout: float = 0.1
    max_len: int = 1024
    pad_id: int = 0


class TransformerEncoder(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model, padding_idx=cfg.pad_id)
        self.pos = PositionalEncoding(cfg.d_model, max_len=cfg.max_len)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=cfg.d_model,
            nhead=cfg.n_heads,
            dim_feedforward=cfg.d_ff,
            dropout=cfg.dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=cfg.n_layers)
        self.norm = nn.LayerNorm(cfg.d_model)

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # input_ids: [B, L]; attention_mask: [B, L] with 1 for valid, 0 for pad
        x = self.embed(input_ids)
        x = self.pos(x)
        if attention_mask is not None:
            # Transformer expects True for masked positions, so invert
            key_padding_mask = attention_mask == 0  # [B, L]
        else:
            key_padding_mask = None
        h = self.encoder(x, src_key_padding_mask=key_padding_mask)
        return self.norm(h)  # [B, L, D]


class MeanPooler(nn.Module):
    def forward(self, hidden: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        # hidden: [B, L, D], mask: [B, L] (1 valid, 0 pad)
        if mask is None:
            return hidden.mean(dim=1)
        lengths = mask.sum(dim=1).clamp_min(1).unsqueeze(1)
        summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        return summed / lengths


class ClauseScorer(nn.Module):
    """
    Clause-only scorer suitable for iProver Passive Scores Mode.

    Input: token ids and attention mask for a single clause with optional structure prefix tokens.
    Output: scalar score per clause (higher -> more priority).
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.encoder = TransformerEncoder(cfg)
        self.pool = MeanPooler()
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        h = self.encoder(input_ids, attention_mask)
        z = self.pool(h, attention_mask)
        return self.head(z).squeeze(-1)  # [B]


class BiEncoder(nn.Module):
    """
    Optional dual-encoder for (query, clause). Not used by default in EA.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.q_enc = TransformerEncoder(cfg)
        self.d_enc = TransformerEncoder(cfg)
        self.pool = MeanPooler()

    def encode(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None, which: str = "q") -> torch.Tensor:
        enc = self.q_enc if which == "q" else self.d_enc
        h = enc(input_ids, attention_mask)
        return self.pool(h, attention_mask)  # [B, D]

    def score(self, q_ids: torch.Tensor, q_mask: torch.Tensor, d_ids: torch.Tensor, d_mask: torch.Tensor) -> torch.Tensor:
        q = self.encode(q_ids, q_mask, which="q")
        d = self.encode(d_ids, d_mask, which="d")
        return (q * d).sum(dim=1)  # dot product [B]


class CrossEncoder(nn.Module):
    """
    Minimal cross-encoder: reuse ClauseScorer on concatenated (<Q> ... </Q> <D> ... </D>) IDs.
    Prepare inputs in the dataloader; this module mirrors ClauseScorer head.
    """

    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.encoder = TransformerEncoder(cfg)
        self.pool = MeanPooler()
        self.head = nn.Sequential(
            nn.Linear(cfg.d_model, cfg.d_model),
            nn.GELU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.d_model, 1),
        )

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        h = self.encoder(input_ids, attention_mask)
        z = self.pool(h, attention_mask)
        return self.head(z).squeeze(-1)
