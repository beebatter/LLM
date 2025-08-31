from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import torch
from torch.utils.data import Dataset, DataLoader

from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets


def _wrap_clause(text: str, features: Optional[Dict] = None) -> str:
    prefix = ""
    if features:
        buckets = PrefixBuckets()
        prefix = features_to_prefix(features, buckets)
    return f"{prefix}<D> {normalize_text(text)} </D>"


@dataclass
class ClauseRecord:
    text: str
    features: Optional[Dict]
    label: Optional[float] = None


class ClauseDataset(Dataset):
    """Reads JSONL with keys: text, optional features, optional label/score."""

    def __init__(self, jsonl_paths: List[str]) -> None:
        self.items: List[ClauseRecord] = []
        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    text = j.get("text")
                    if not text:
                        continue
                    features = j.get("features") or None
                    label = j.get("label") or j.get("score")
                    if isinstance(label, (int, float)):
                        label = float(label)
                    else:
                        label = None
                    self.items.append(ClauseRecord(text=text, features=features, label=label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> ClauseRecord:
        return self.items[idx]


class Collate:
    def __init__(self, spm_model: str, max_len: int = 512, pad_id: int = 0) -> None:
        self.tok = LogicSentencePiece(spm_model)
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, batch: List[ClauseRecord]) -> Dict[str, torch.Tensor]:
        ids_list: List[List[int]] = []
        mask_list: List[List[int]] = []
        labels: List[float] = []
        for rec in batch:
            s = _wrap_clause(rec.text, rec.features)
            ids = self.tok.encode(s)
            ids = ids[: self.max_len]
            ids_list.append(ids)
            mask_list.append([1] * len(ids))
            labels.append(float(rec.label) if rec.label is not None else 0.0)

        maxl = max(len(x) for x in ids_list)
        padded_ids = [x + [self.pad_id] * (maxl - len(x)) for x in ids_list]
        padded_mask = [m + [0] * (maxl - len(m)) for m in mask_list]

        return {
            "input_ids": torch.tensor(padded_ids, dtype=torch.long),
            "attention_mask": torch.tensor(padded_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.float),
        }


def build_dataloader(jsonl_paths: List[str], spm_model: str, batch_size: int = 64, shuffle: bool = True, num_workers: int = 2, max_len: int = 512) -> DataLoader:
    ds = ClauseDataset(jsonl_paths)
    collate = Collate(spm_model=spm_model, max_len=max_len)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=collate)
