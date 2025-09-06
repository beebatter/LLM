#!/usr/bin/env python3
"""
Prepare listwise training data from flat CE pairs JSONL.

Input: pairs.*.jsonl with fields like:
  {"query": str, "doc": str, "label": 0/1 or float, "problem_name": str, ...}

Optional: join teacher scores from ce_scored.*.jsonl produced by score_cross_encoder.py:
  {"text_a": str, "text_b": str, "ce_score": float, "problem_name": str, ...}

Output: groups.*.jsonl where each line is:
  {
    "problem_name": str,
    "query": str,
    "group_id": str,  # stable composite key
    "candidates": [
      {"text": str, "label": int|float, "weight": float, "ce_score": float | null,
       "meta": dict | null}
    ],
    "target_scores": [float] | null  # group-wise distribution from teacher (e.g., ce_score)
  }

Usage example:
  python prepare_listwise_data.py \
    --input /root/autodl-tmp/Training/datasets/pairs.full.train.jsonl \
    --output /root/autodl-tmp/Training/datasets/groups.k48.train.jsonl \
    --k 48 --min-positives 1 --seed 42 --teacher ce --tau 1.0 \
    --ce-scored /root/autodl-tmp/Training/datasets/ce_scored.train.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Tuple


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # best-effort skip
                continue


def write_jsonl(path: str, rows: Iterable[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def softmax(xs: List[float], tau: float = 1.0) -> List[float]:
    if not xs:
        return []
    # numerical stability
    mx = max(xs)
    exps = [math.exp((x - mx) / max(1e-8, tau)) for x in xs]
    s = sum(exps)
    if s <= 0:
        n = len(xs)
        return [1.0 / n] * n
    return [e / s for e in exps]


def norm_label(v) -> float:
    try:
        return float(v)
    except Exception:
        return 1.0 if v in (True, "1", 1) else 0.0


def make_group_key(problem_name: Optional[str], query: str) -> str:
    pn = (problem_name or "").strip()
    return f"{pn}||{query}"


def load_teacher_scores_ce(path: str) -> Dict[Tuple[str, str, str], float]:
    """Map (problem_name, query(text_a), doc(text_b)) -> ce_score."""
    mapping: Dict[Tuple[str, str, str], float] = {}
    for r in read_jsonl(path):
        q = r.get("text_a")
        d = r.get("text_b")
        pn = r.get("problem_name") or ""
        s = r.get("ce_score")
        if q is None or d is None or s is None:
            continue
        mapping[(pn, q, d)] = float(s)
    return mapping


def build_groups(
    pairs_path: str,
    k: int,
    min_positives: int,
    seed: int,
    teacher_kind: Optional[str],
    teacher_tau: float,
    ce_scored_path: Optional[str],
) -> Iterable[dict]:
    random.seed(seed)

    # Load teacher scores if provided
    ce_scores: Dict[Tuple[str, str, str], float] = {}
    if teacher_kind == "ce" and ce_scored_path:
        ce_scores = load_teacher_scores_ce(ce_scored_path)

    groups: Dict[str, List[dict]] = defaultdict(list)
    for r in read_jsonl(pairs_path):
        q = r.get("query")
        d = r.get("doc")
        if not q or not d:
            continue
        pn = r.get("problem_name") or ""
        key = make_group_key(pn, q)
        groups[key].append(r)

    keys = list(groups.keys())
    random.shuffle(keys)

    for key in keys:
        items = groups[key]
        if not items:
            continue
        # separate positives and negatives
        pos = [it for it in items if norm_label(it.get("label", 0)) > 0.0]
        neg = [it for it in items if norm_label(it.get("label", 0)) <= 0.0]
        if len(pos) < min_positives:
            continue

        # select up to k items, include all positives, fill with negatives
        random.shuffle(pos)
        random.shuffle(neg)
        chosen = pos[:k]
        if len(chosen) < k:
            need = k - len(chosen)
            chosen.extend(neg[:need])

        # If still short (small groups), just take what we have
        cand = []
        teacher_scores: List[Optional[float]] = []
        for it in chosen:
            q = it.get("query")
            d = it.get("doc")
            pn = it.get("problem_name") or ""
            label = norm_label(it.get("label", 0))
            weight = float(it.get("weight", 1.0))
            meta = it.get("meta") or None
            ce_score = None
            if ce_scores:
                ce_score = ce_scores.get((pn, q, d))
                if ce_score is None:
                    # try fallback by dropping pn
                    ce_score = ce_scores.get(("", q, d))
            cand.append({
                "text": d,
                "label": label,
                "weight": weight,
                "ce_score": ce_score,
                "meta": meta,
            })
            teacher_scores.append(ce_score)

        # Compute target distribution if requested and teacher available
        target_scores = None
        if teacher_kind == "ce" and any(s is not None for s in teacher_scores):
            # fill missing with min score - margin for stability
            filled = []
            finite_vals = [s for s in teacher_scores if s is not None]
            base = min(finite_vals) - 1.0 if finite_vals else 0.0
            for s in teacher_scores:
                filled.append(float(s) if s is not None else base)
            target_scores = softmax(filled, tau=teacher_tau)

        # unpack group key
        try:
            pn, q = key.split("||", 1)
        except ValueError:
            pn, q = "", key

        yield {
            "problem_name": pn,
            "query": q,
            "group_id": key,
            "candidates": cand,
            "target_scores": target_scores,
        }


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Prepare listwise groups from CE pairs JSONL")
    p.add_argument("--input", required=True, help="Path to pairs.*.jsonl")
    p.add_argument("--output", required=True, help="Path to write groups.*.jsonl")
    p.add_argument("--k", type=int, default=48, help="Max candidates per group")
    p.add_argument("--min-positives", type=int, default=1, help="Min positives per group")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--teacher", choices=["ce"], default=None, help="Which teacher to use for target_scores")
    p.add_argument("--tau", type=float, default=1.0, help="Temperature for teacher softmax")
    p.add_argument("--ce-scored", default=None, help="Optional ce_scored.*.jsonl for teacher scores")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = build_groups(
        pairs_path=args.input,
        k=args.k,
        min_positives=args.min_positives,
        seed=args.seed,
        teacher_kind=args.teacher,
        teacher_tau=args.tau,
        ce_scored_path=args.ce_scored,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote groups to {args.output}")


if __name__ == "__main__":
    main()
