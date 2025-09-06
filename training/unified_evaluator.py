#!/usr/bin/env python3
"""
Unified evaluator for ranking metrics across different models.

Expected input: predictions JSONL with fields per item:
  {
    "problem_name": str,
    "query": str,              # or use "text_a" as fallback
    "group_id": str,           # stable key (problem||query)
    "doc": str,                # or use "text_b" as fallback (optional)
    "label": int|float,        # ground truth (0/1 or real-valued) (aliases: y, target)
    "score": float,            # model score (higher is better) (aliases: ce_score, bi_score, llm_score)
    "rank": int                # optional, ignored if provided
  }

This script groups by group_id (or problem_name+query fallback) and computes:
  - MRR@K, Recall@K, NDCG@K for K in a set
  - Classification AUC/AP on the whole set (if labels are binary)
  - Diagnostics: groups, avg_candidates, avg_positives, hit@1, hit@1_pos

Usage:
  python unified_evaluator.py --pred path1.jsonl [--name CE] \
    --pred path2.jsonl --name BI --ks 10 32 64 --out metrics.json
"""
from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
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
                continue


def dcg(rels: List[float], k: Optional[int] = None) -> float:
    if k is not None:
        rels = rels[:k]
    return sum((2 ** r - 1) / math.log2(i + 2) for i, r in enumerate(rels))


def ndcg(labels_sorted_by_score: List[float], k: Optional[int] = None) -> float:
    dcg_val = dcg(labels_sorted_by_score, k)
    ideal = sorted(labels_sorted_by_score, reverse=True)
    idcg_val = dcg(ideal, k)
    return (dcg_val / idcg_val) if idcg_val > 0 else 0.0


def compute_group_metrics(items: List[Tuple[float, float]], ks: List[int]):
    # items: list of (score, label)
    items = sorted(items, key=lambda x: x[0], reverse=True)
    labels_sorted = [lbl for _, lbl in items]
    pos_total = sum(1 for l in labels_sorted if l > 0)
    res = {}
    for k in ks:
        topk = labels_sorted[:k]
        hit_at_k = 1.0 if any(l > 0 for l in topk) else 0.0
        recall_k = (sum(1 for l in topk if l > 0) / max(1, pos_total)) if pos_total > 0 else 0.0
        ndcg_k = ndcg(labels_sorted, k)
        # MRR@K: first positive rank in top-k
        mrr_k = 0.0
        for i, l in enumerate(topk, start=1):
            if l > 0:
                mrr_k = 1.0 / i
                break
        res[k] = {"hit": hit_at_k, "recall": recall_k, "ndcg": ndcg_k, "mrr": mrr_k}
    # diagnostics
    first_is_pos = 1.0 if (labels_sorted and labels_sorted[0] > 0) else 0.0
    return res, {"positives": pos_total, "first_is_pos": first_is_pos, "candidates": len(items)}


def try_auc_ap(all_scores: List[float], all_labels: List[int]):
    # Optional dependency: scikit-learn
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        ap = float(average_precision_score(all_labels, all_scores))
        auc = float(roc_auc_score(all_labels, all_scores))
        return {"ap": ap, "auc": auc}
    except Exception:
        return None


def evaluate_predictions(pred_path: str, ks: List[int]):
    groups: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
    all_scores: List[float] = []
    all_labels: List[int] = []

    for r in read_jsonl(pred_path):
        q = r.get("query") or r.get("text_a") or ""
        pn = r.get("problem_name") or ""
        gid = r.get("group_id") or f"{pn}||{q}"
        score = r.get("score")
        if score is None:
            for alt in ("ce_score", "bi_score", "llm_score"):
                if r.get(alt) is not None:
                    score = r.get(alt)
                    break
        label = r.get("label")
        if label is None:
            for alt in ("y", "target"):
                if r.get(alt) is not None:
                    label = r.get(alt)
                    break
        if score is None or label is None:
            continue
        try:
            score = float(score)
            label_f = float(label)
        except Exception:
            continue
        groups[gid].append((score, label_f))
        # Treat labels >0 as positives for pairwise metrics
        all_scores.append(score)
        all_labels.append(1 if label_f > 0 else 0)

    # aggregate group metrics
    totals = {k: {"hit": 0.0, "recall": 0.0, "ndcg": 0.0, "mrr": 0.0} for k in ks}
    diag_counts = {"groups": 0, "candidates": 0, "positives": 0, "hit@1": 0.0, "hit@1_pos": 0.0}
    for gid, items in groups.items():
        per_k, diag = compute_group_metrics(items, ks)
        diag_counts["groups"] += 1
        diag_counts["candidates"] += diag.get("candidates", 0)
        diag_counts["positives"] += diag.get("positives", 0)
        diag_counts["hit@1"] += per_k[1]["hit"] if 1 in per_k else 0.0
        diag_counts["hit@1_pos"] += diag.get("first_is_pos", 0.0)
        for k in ks:
            for m in ("hit", "recall", "ndcg", "mrr"):
                totals[k][m] += per_k[k][m]

    n_groups = max(1, diag_counts["groups"])
    averaged = {k: {m: totals[k][m] / n_groups for m in totals[k]} for k in ks}
    diag_out = {
        **{k: v for k, v in diag_counts.items()},
        "avg_candidates": diag_counts["candidates"] / n_groups,
        "avg_positives": diag_counts["positives"] / n_groups,
    }

    cls_metrics = try_auc_ap(all_scores, all_labels)
    return {"ranking": averaged, "classification": cls_metrics, "diagnostics": diag_out}


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Unified ranking evaluator for predictions JSONL")
    p.add_argument("--pred", action="append", required=True, help="Path to predictions JSONL")
    p.add_argument("--name", action="append", help="Optional model name for each --pred")
    p.add_argument("--ks", type=int, nargs="+", default=[1, 10, 32, 64], help="K values")
    p.add_argument("--out", default=None, help="Optional output metrics JSON path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    names = args.name or []
    if names and len(names) != len(args.pred):
        raise SystemExit("--name count must equal --pred count or be omitted")

    all_results = {}
    for i, pred_path in enumerate(args.pred):
        name = names[i] if i < len(names) else f"run_{i+1}"
        metrics = evaluate_predictions(pred_path, args.ks)
        all_results[name] = metrics

    text = json.dumps(all_results, ensure_ascii=False, indent=2)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(text + "\n")
        print(f"Wrote metrics to {args.out}")
    print(text)


if __name__ == "__main__":
    main()
