#!/usr/bin/env python3
"""
Lightweight datasets for Bi-Encoder and Cross-Encoder training using the shared
logic tokenizer/normalizer with structure prefix and <Q>/<D> wrappers.

Each JSONL is expected to contain fields:
  - problem_name (optional)
  - conjecture_text or conjecture_sig (string)
  - text (clause string)
  - features (dict) optional
  - label (int) optional for classification

This is minimal and can be adapted to your existing trainer loops.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from .logic_tokenizer import (
    LogicSentencePiece,
    normalize_text,
    wrap_q,
    wrap_d,
    truncate_ids,
)


class JsonlReader:
    def __init__(self, path: str | Path) -> None:
        self.path = str(path)

    def __iter__(self) -> Iterator[Dict]:
        with open(self.path, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except Exception:
                    continue


class BiEncoderDataset:
    """
    Emits (q_ids, d_ids, label?) where label is optional for contrastive training.
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        spm_model: str = "/home/ks/Training/models/spm_logic.model",
        max_len_q: int = 256,
        max_len_d: int = 256,
        include_label: bool = False,
    ) -> None:
        self.reader = JsonlReader(jsonl_path)
        self.tok = LogicSentencePiece(spm_model)
        self.max_len_q = max_len_q
        self.max_len_d = max_len_d
        self.include_label = include_label

    def __iter__(self) -> Iterator[Tuple[List[int], List[int]]]:
        for j in self.reader:
            d = j.get("text")
            if not d:
                continue
            q = j.get("conjecture_text") or j.get("conjecture_sig") or ""
            features = j.get("features") or {}
            qn = normalize_text(q) if q else ""
            dn = normalize_text(d)
            qwrap = wrap_q(qn) if qn else ""
            dwrap = wrap_d(dn, features)
            if not qwrap:
                # Some setups may still train with only clauses; skip if q missing.
                continue
            q_ids = truncate_ids(self.tok.encode(qwrap), self.max_len_q)
            d_ids = truncate_ids(self.tok.encode(dwrap), self.max_len_d)
            if self.include_label:
                yield (q_ids, d_ids, int(j.get("label", 0)))
            else:
                yield (q_ids, d_ids)


class CrossEncoderDataset:
    """
    Emits (qd_ids, label?) where qd is concatenated string: [Q] ... [/Q] [D] ... [/D]
    """

    def __init__(
        self,
        jsonl_path: str | Path,
        spm_model: str = "/home/ks/Training/models/spm_logic.model",
        max_len: int = 256,
        include_label: bool = True,
    ) -> None:
        self.reader = JsonlReader(jsonl_path)
        self.tok = LogicSentencePiece(spm_model)
        self.max_len = max_len
        self.include_label = include_label

    def __iter__(self) -> Iterator[List[int]]:
        for j in self.reader:
            d = j.get("text")
            if not d:
                continue
            q = j.get("conjecture_text") or j.get("conjecture_sig") or ""
            if not q:
                continue
            features = j.get("features") or {}
            qn = normalize_text(q)
            dn = normalize_text(d)
            qwrap = wrap_q(qn)
            dwrap = wrap_d(dn, features)
            qd = f"{qwrap} {dwrap}"
            qd_ids = truncate_ids(self.tok.encode(qd), self.max_len)
            if self.include_label:
                yield (qd_ids, int(j.get("label", 0)))
            else:
                yield qd_ids
