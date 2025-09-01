from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch.utils.data import Dataset, DataLoader
import random
import math
from collections import defaultdict, deque

from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets
import hashlib


def _wrap_q(text: str) -> str:
    return f"<Q> {normalize_text(text)} </Q>"


def _wrap_d(text: str, features: Optional[Dict] = None) -> str:
    prefix = features_to_prefix(features or {}, PrefixBuckets())
    return f"{prefix}<D> {normalize_text(text)} </D>"


@dataclass
class QDItem:
    q: str
    d: str
    features: Optional[Dict]
    label: float
    problem_name: str


class BiPairDataset(Dataset):
    """JSONL schema is flexible. Accepts any of these keys:
    - q: conjecture_text | conjecture_sig | query | q
    - d: text | doc | d | clause | premise
    - features: features | meta
    - label: label (defaults to 0.0 when missing)
    - problem_name: problem_name | problem | qid (defaults to '<UNK>')
    """

    def __init__(self, jsonl_paths: List[str], limit: Optional[int] = None) -> None:
        self.items: List[QDItem] = []

        def get_first(dct: Dict, keys: List[str]) -> Optional[str]:
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
                    q = get_first(j, ["conjecture_text", "conjecture_sig", "query", "q", "conjecture"])  # type: ignore[assignment]
                    d = get_first(j, ["text", "doc", "d", "clause", "premise"])  # type: ignore[assignment]
                    if not q or not d:
                        continue
                    features = get_first(j, ["features", "meta"])  # may be dict or None
                    lab = j.get("label")
                    try:
                        label = float(lab) if lab is not None else 0.0
                    except Exception:
                        label = 0.0
                    problem_name = get_first(j, ["problem_name", "problem", "qid"]) or "<UNK>"
                    self.items.append(QDItem(q=q, d=d, features=features, label=label, problem_name=problem_name))
                    if limit is not None and len(self.items) >= limit:
                        break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> QDItem:
        return self.items[idx]


class BiCollate:
    def __init__(self, spm_model: str, max_len: int = 256, pad_id: int = 0) -> None:
        self.tok = LogicSentencePiece(spm_model)
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, batch: List[QDItem]) -> Dict[str, torch.Tensor]:
        q_ids_list: List[List[int]] = []
        q_masks: List[List[int]] = []
        d_ids_list: List[List[int]] = []
        d_masks: List[List[int]] = []
        labels: List[float] = []
        buckets: List[str] = []
        problems: List[str] = []
        d_hashes: List[str] = []
        for it in batch:
            q_s = _wrap_q(it.q)
            d_s = _wrap_d(it.d, it.features)
            q_ids = self.tok.encode(q_s)[: self.max_len]
            d_ids = self.tok.encode(d_s)[: self.max_len]
            q_ids_list.append(q_ids)
            d_ids_list.append(d_ids)
            q_masks.append([1] * len(q_ids))
            d_masks.append([1] * len(d_ids))
            labels.append(float(it.label))
            # bucket key derived from features (same logic as prefix)
            bucket_key = features_to_prefix(it.features or {}, PrefixBuckets())
            buckets.append(bucket_key or "<UNK>")
            problems.append(it.problem_name or "<UNK>")
            # stable id for doc: md5 of normalized raw doc string
            norm_d = normalize_text(it.d)
            d_hash = hashlib.md5(norm_d.encode("utf-8")).hexdigest()
            d_hashes.append(d_hash)

        def pad(arrs: List[List[int]], pad_id: int) -> torch.Tensor:
            maxl = max(len(x) for x in arrs)
            return torch.tensor([x + [pad_id] * (maxl - len(x)) for x in arrs], dtype=torch.long)

        def padm(arrs: List[List[int]]) -> torch.Tensor:
            maxl = max(len(x) for x in arrs)
            return torch.tensor([x + [0] * (maxl - len(x)) for x in arrs], dtype=torch.long)

        return {
            "q_ids": pad(q_ids_list, self.pad_id),
            "q_mask": padm(q_masks),
            "d_ids": pad(d_ids_list, self.pad_id),
            "d_mask": padm(d_masks),
            "labels": torch.tensor(labels, dtype=torch.float),
            "bucket": buckets,
            "problem": problems,
            "d_hash": d_hashes,
        }


def build_bi_dataloader(paths: List[str], spm_model: str, batch_size: int = 256, shuffle: bool = True, num_workers: int = 2, max_len: int = 256, limit: Optional[int] = None) -> DataLoader:
    ds = BiPairDataset(paths, limit=limit)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, collate_fn=BiCollate(spm_model, max_len=max_len))


class ProblemGroupedBatchSampler:
    """
    Batch sampler that groups samples by problem_name to provide multiple positives per batch.
    Heuristic: in each batch, include chunks of size 'chunk' from distinct problems until reaching batch_size.
    This increases the chance that SupCon sees multi-positive structure.
    """

    def __init__(self, dataset: BiPairDataset, batch_size: int, seed: int = 42):
        self.dataset = dataset
        self.batch_size = batch_size
        self.seed = seed
        self._epoch = 0
        # build problem -> deque(indices)
        groups: dict[str, deque[int]] = defaultdict(deque)
        rng = random.Random(seed)
        for i, it in enumerate(dataset.items):
            groups[it.problem_name or "<UNK>"].append(i)
        # shuffle within each group
        for k in list(groups.keys()):
            arr = list(groups[k])
            rng.shuffle(arr)
            groups[k] = deque(arr)
        self.groups = groups
        self.group_keys = list(groups.keys())
        rng.shuffle(self.group_keys)

    def __iter__(self):
        # New iterator that guarantees full coverage each epoch by round-robin over non-empty groups
        rng = random.Random(self.seed + self._epoch)
        self._epoch += 1
        # working copy of groups with per-epoch reshuffle
        groups: dict[str, deque[int]] = {}
        for k, v in self.groups.items():
            arr = list(v)
            rng.shuffle(arr)
            groups[k] = deque(arr)
        active_keys = [k for k, q in groups.items() if len(q) > 0]
        rng.shuffle(active_keys)
        chunk = max(2, self.batch_size // 8)  # at least 2 per problem
        batch: list[int] = []
        while active_keys:
            next_active: list[str] = []
            for k in active_keys:
                if len(batch) >= self.batch_size:
                    yield batch
                    batch = []
                q = groups[k]
                if not q:
                    continue
                take = min(chunk, self.batch_size - len(batch), len(q))
                for _ in range(take):
                    batch.append(q.popleft())
                if len(q) > 0:
                    next_active.append(k)
            active_keys = next_active
            rng.shuffle(active_keys)
        if batch:
            yield batch

    def __len__(self) -> int:
        n = len(self.dataset)
        return max(1, math.ceil(n / max(1, self.batch_size)))


def build_bi_dataloader_grouped(paths: List[str], spm_model: str, batch_size: int = 256, num_workers: int = 2, max_len: int = 256, limit: Optional[int] = None) -> DataLoader:
    ds = BiPairDataset(paths, limit=limit)
    sampler = ProblemGroupedBatchSampler(ds, batch_size=batch_size)
    return DataLoader(ds, batch_sampler=sampler, num_workers=num_workers, collate_fn=BiCollate(spm_model, max_len=max_len))
