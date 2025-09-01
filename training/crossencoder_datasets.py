from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader

from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets


def _wrap_qd(q: str, d: str, features: Optional[Dict] = None) -> str:
    qn = f"<Q> {normalize_text(q)} </Q>"
    prefix = features_to_prefix(features or {}, PrefixBuckets())
    dn = f"{prefix}<D> {normalize_text(d)} </D>"
    return f"{qn} {dn}"


@dataclass
class QDPair:
    q: str
    d: str
    features: Optional[Dict]
    label: float


class CrossDataset(Dataset):
    def __init__(self, jsonl_paths: List[str]) -> None:
        self.items: List[QDPair] = []

        def _get_first(dct: Dict, keys: List[str]):
            for k in keys:
                v = dct.get(k)
                if v is not None:
                    return v
            return None

        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    # Support both unified schema (text_a/text_b) and older fields
                    q = _get_first(j, ["text_a", "conjecture_text", "conjecture_sig", "query", "q", "conjecture"])  # type: ignore[assignment]
                    d = _get_first(j, ["text_b", "text", "doc", "d", "clause", "premise"])  # type: ignore[assignment]
                    if not q or not d:
                        continue
                    features = _get_first(j, ["features", "meta"]) or None
                    lab = j.get("label")
                    try:
                        label = float(lab) if lab is not None else 0.0
                    except Exception:
                        label = 0.0
                    self.items.append(QDPair(q=q, d=d, features=features, label=label))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> QDPair:
        return self.items[idx]


class CrossCollate:
    def __init__(self, spm_model: str, max_len: int = 256, pad_id: int = 0) -> None:
        self.tok = LogicSentencePiece(spm_model)
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, batch: List[QDPair]) -> Dict[str, torch.Tensor]:
        ids_list: List[List[int]] = []
        mask_list: List[List[int]] = []
        labels: List[float] = []
        for it in batch:
            s = _wrap_qd(it.q, it.d, it.features)
            ids = self.tok.encode(s)[: self.max_len]
            ids_list.append(ids)
            mask_list.append([1] * len(ids))
            labels.append(float(it.label))
        maxl = max(len(x) for x in ids_list)
        ids = torch.tensor([x + [self.pad_id] * (maxl - len(x)) for x in ids_list], dtype=torch.long)
        mask = torch.tensor([m + [0] * (maxl - len(m)) for m in mask_list], dtype=torch.long)
        y = torch.tensor(labels, dtype=torch.float)
        return {"input_ids": ids, "attention_mask": mask, "labels": y}


def build_cross_dataloader(paths: List[str], spm_model: str, batch_size: int = 128, shuffle: bool = True, num_workers: int = 2, max_len: int = 256, limit: Optional[int] = None) -> DataLoader:
    ds = CrossDataset(paths)  # limit is unused here for now
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=CrossCollate(spm_model, max_len=max_len))
