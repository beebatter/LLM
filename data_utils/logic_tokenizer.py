#!/usr/bin/env python3
"""
Shared normalization + SentencePiece wrapper for ATP clause text.

Defaults:
- SentencePiece model: /home/ks/Training/models/spm_logic.model
- Normalization: variable unification -> VAR; keep ()=,~&| as standalone; <SYM_LONG> for very long symbols
- Structure prefix tokens from features: <H*><U*><E*><C*><B*>

Use this in Bi-/Cross-Encoder data pipelines to ensure consistent preprocessing.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

try:
    import sentencepiece as spm
except Exception:  # pragma: no cover
    spm = None


var_pat = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")
punct_pat = re.compile(r"([()=,~&|])")
sym_long_pat = re.compile(r"\b[A-Za-z0-9_]{41,}\b")
space_pat = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = sym_long_pat.sub("<SYM_LONG>", s)
    s = var_pat.sub("VAR", s)
    s = punct_pat.sub(r" \1 ", s)
    s = space_pat.sub(" ", s).strip()
    return s


@dataclass
class PrefixBuckets:
    conj_cuts: Tuple[int, int, int] = (0, 2, 5)
    born_cuts: Tuple[int, int, int] = (0, 8, 32)

    def conj_bucket(self, val: Optional[int]) -> int:
        if val is None:
            return 0
        a, b, c = self.conj_cuts
        if val <= a:
            return 0
        if val <= b:
            return 1
        if val <= c:
            return 2
        return 3

    def born_bucket(self, val: Optional[int]) -> int:
        if val is None:
            return 0
        a, b, c = self.born_cuts
        if val <= a:
            return 0
        if val <= b:
            return 1
        if val <= c:
            return 2
        return 3


def features_to_prefix(features: Dict, buckets: PrefixBuckets = PrefixBuckets()) -> str:
    if not isinstance(features, dict):
        return ""
    def as_int(x):
        try:
            return int(x)
        except Exception:
            return 0
    horn = 1 if as_int(features.get("horn", 0)) else 0
    unit = 1 if as_int(features.get("unit", 0)) else 0
    epr = 1 if as_int(features.get("epr", 0)) else 0
    conj_dist = features.get("conj_dist")
    born = features.get("born")
    try:
        conj_dist = int(conj_dist) if conj_dist is not None else None
    except Exception:
        conj_dist = None
    try:
        born = int(born) if born is not None else None
    except Exception:
        born = None
    cbin = buckets.conj_bucket(conj_dist)
    bbin = buckets.born_bucket(born)
    return f"<H{horn}><U{unit}><E{epr}><C{cbin}><B{bbin}> "


class LogicSentencePiece:
    def __init__(self, model_path: str = "/home/ks/Training/models/spm_logic.model") -> None:
        if spm is None:
            raise RuntimeError("sentencepiece is not installed. pip install sentencepiece")
        self.model_path = str(model_path)
        self.sp = spm.SentencePieceProcessor(model_file=self.model_path)

    def encode(self, s: str) -> List[int]:
        return self.sp.encode(s, out_type=int)

    def decode(self, ids: Sequence[int]) -> str:
        return self.sp.decode(ids)

    @property
    def vocab_size(self) -> int:
        return self.sp.vocab_size()

    def token_to_id(self, token: str) -> int:
        return self.sp.piece_to_id(token)


def wrap_q(s: str) -> str:
    return f"<Q> {s} </Q>"


def wrap_d(s: str, features: Optional[Dict] = None, buckets: PrefixBuckets = PrefixBuckets()) -> str:
    prefix = features_to_prefix(features or {}, buckets)
    return f"{prefix}<D> {s} </D>"


def truncate_ids(ids: List[int], max_len: int) -> List[int]:
    if max_len and len(ids) > max_len:
        return ids[:max_len]
    return ids
