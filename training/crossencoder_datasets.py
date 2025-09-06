from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Iterable, Iterator

import torch
from torch.utils.data import Dataset, DataLoader

from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets
import hashlib


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
    group: str  # grouping key (e.g., problem_name or query id)


class CrossDataset(Dataset):
    def __init__(self, jsonl_paths: List[str]) -> None:
        self.items: List[QDPair] = []
        self.group_to_indices: Dict[str, List[int]] = {}

        def get_first(dct: Dict, keys: List[str]):
            for k in keys:
                v = dct.get(k)
                if v is not None:
                    return v
            return None

        def md5(s: str) -> str:
            return hashlib.md5(s.encode('utf-8')).hexdigest()

        for p in jsonl_paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    # accept both pair and unified schemas
                    q = get_first(j, ["conjecture_text", "conjecture_sig", "query", "q", "conjecture", "text_a"])  # type: ignore[assignment]
                    d = get_first(j, ["text", "doc", "d", "clause", "premise", "text_b"])  # type: ignore[assignment]
                    if not q or not d:
                        continue
                    features = get_first(j, ["features", "meta"]) or None
                    # grouping key: prefer explicit problem/query id, else md5 of normalized query
                    g = get_first(j, ["problem_name", "problem", "query_id", "qid", "qid_str"]) or md5(normalize_text(q))
                    lab = j.get("label")
                    try:
                        label = float(lab) if lab is not None else 0.0
                    except Exception:
                        label = 0.0
                    self.items.append(QDPair(q=q, d=d, features=features, label=label, group=str(g)))
        # build group index
        for idx, it in enumerate(self.items):
            self.group_to_indices.setdefault(it.group, []).append(idx)

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


class CrossCollateGrouped(CrossCollate):
    def __call__(self, batch: List[QDPair]) -> Dict[str, torch.Tensor]:
        out = super().__call__(batch)
        # map group strings in batch to small integer ids
        groups = []
        gid_map: Dict[str, int] = {}
        next_id = 0
        for it in batch:
            g = it.group
            if g not in gid_map:
                gid_map[g] = next_id
                next_id += 1
            groups.append(gid_map[g])
        out["group_ids"] = torch.tensor(groups, dtype=torch.long)
        return out


def build_cross_dataloader(paths: List[str], spm_model: str, batch_size: int = 128, shuffle: bool = True, num_workers: int = 2, max_len: int = 256, limit: Optional[int] = None) -> DataLoader:
    ds = CrossDataset(paths)  # limit is unused here for now
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=CrossCollate(spm_model, max_len=max_len))


class GroupedBatchSampler:
    """Yield batches composed of multiple groups, each contributing up to group_size samples.

    If groups_per_batch is None or <=1, one group per batch with up to group_size samples.
    """
    def __init__(self, dataset: CrossDataset, group_size: int = 16, groups_per_batch: int = 4, shuffle: bool = True) -> None:
        self.ds = dataset
        self.group_size = max(1, int(group_size))
        self.groups_per_batch = max(1, int(groups_per_batch))
        self.shuffle = shuffle

    def __iter__(self) -> Iterator[List[int]]:
        groups = list(self.ds.group_to_indices.items())
        if self.shuffle:
            import random
            random.shuffle(groups)
        for i in range(0, len(groups), self.groups_per_batch):
            chunk = groups[i:i + self.groups_per_batch]
            # assemble batch from up to groups_per_batch groups
            batch_idx: List[int] = []
            for g, idxs in chunk:
                if self.shuffle:
                    import random
                    random.shuffle(idxs)
                batch_idx.extend(idxs[: self.group_size])
            if batch_idx:
                yield batch_idx

    def __len__(self) -> int:
        # rough estimate: number of batches is number of groups divided by groups_per_batch
        import math
        return math.ceil(len(self.ds.group_to_indices) / self.groups_per_batch)


def build_cross_grouped_dataloader(paths: List[str], spm_model: str, group_size: int = 16, groups_per_batch: int = 4, num_workers: int = 2, max_len: int = 256, shuffle: bool = True) -> DataLoader:
    ds = CrossDataset(paths)
    sampler = GroupedBatchSampler(ds, group_size=group_size, groups_per_batch=groups_per_batch, shuffle=shuffle)
    return DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, collate_fn=CrossCollateGrouped(spm_model, max_len=max_len))
