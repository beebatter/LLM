#!/usr/bin/env python3
"""
Build many listwise groups (K=64) per query from large candidate pools.

Why: Each query may have thousands of candidates. To increase training data, we
produce multiple group replicas with different candidate subsets per query.

Strategies:
  - random: include as many positives as possible, fill remaining with random negatives.
  - topmix: if teacher CE scores available, take top_frac*K high-score anchors (e.g., 16 of 64),
            then fill the rest with diverse negatives sampled from the tail.

Target soft labels (optional): join CE scores and apply softmax(tau) per group.

Outputs: groups JSONL where group_id is suffixed with "#rep{r}" to keep replicas distinct.
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
            except Exception:
                continue


def write_jsonl(path: str, rows: Iterable[dict]):
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def softmax(xs: List[float], tau: float = 1.0) -> List[float]:
    if not xs:
        return []
    m = max(xs)
    exps = [math.exp((x - m) / max(1e-8, tau)) for x in xs]
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


def load_teacher_scores_ce(path: Optional[str]) -> Dict[Tuple[str, str, str], float]:
    mapping: Dict[Tuple[str, str, str], float] = {}
    if not path:
        return mapping
    for r in read_jsonl(path):
        q = r.get("text_a"); d = r.get("text_b"); pn = r.get("problem_name") or ""; s = r.get("ce_score")
        if q is None or d is None or s is None:
            continue
        mapping[(pn, q, d)] = float(s)
    return mapping


@dataclass
class Pair:
    problem_name: str
    query: str
    doc: str
    label: float
    weight: float
    meta: Optional[dict]


def load_pairs(path: str) -> Dict[str, List[Pair]]:
    groups: Dict[str, List[Pair]] = defaultdict(list)
    for r in read_jsonl(path):
        q = r.get("query"); d = r.get("doc")
        if not q or not d:
            continue
        pn = r.get("problem_name") or ""
        key = make_group_key(pn, q)
        groups[key].append(Pair(
            problem_name=pn,
            query=q,
            doc=d,
            label=norm_label(r.get("label", 0.0)),
            weight=float(r.get("weight", 1.0)),
            meta=r.get("meta") or None,
        ))
    return groups


def select_random(pos: List[Pair], neg: List[Pair], k: int, rng: random.Random) -> List[Pair]:
    rng.shuffle(pos); rng.shuffle(neg)
    out = []
    if pos:
        if len(pos) > k:
            out.extend(rng.sample(pos, k))
            return out
        out.extend(pos)
    if len(out) < k:
        need = k - len(out)
        if len(neg) > need:
            out.extend(rng.sample(neg, need))
        else:
            out.extend(neg)
    return out


def select_topmix(all_items: List[Tuple[Pair, float]], k: int, top_frac: float, rng: random.Random) -> List[Pair]:
    # Sort by teacher score desc (missing -> very small)
    scored = sorted(all_items, key=lambda x: (x[1] if x[1] is not None else -1e9), reverse=True)
    top_k = max(1, int(round(k * max(0.0, min(1.0, top_frac)))))
    anchors = [p for p, _ in scored[: top_k]]
    tail = [p for p, _ in scored[top_k:]]
    rng.shuffle(tail)
    need = k - len(anchors)
    fill = tail[: max(0, need)]
    return anchors + fill


def build_groups_multi(
    pairs_path: str,
    k: int,
    replicas: int,
    seed: int,
    teacher_tau: float,
    ce_scored_path: Optional[str],
    strategy: str,
    top_frac: float,
) -> Iterable[dict]:
    base_rng = random.Random(seed)
    ce_scores = load_teacher_scores_ce(ce_scored_path)
    groups = load_pairs(pairs_path)

    keys = list(groups.keys())
    base_rng.shuffle(keys)

    total_groups, emitted = 0, 0
    for key in keys:
        items = groups[key]
        if not items:
            continue
        total_groups += 1
        pn, q = ("", key)
        if "||" in key:
            pn, q = key.split("||", 1)

        pos = [it for it in items if it.label > 0.0]
        neg = [it for it in items if it.label <= 0.0]
        if len(pos) == 0:
            continue  # require at least one positive per replica

        # Pre-cache teacher score per candidate for this query (if any)
        scored_all: List[Tuple[Pair, Optional[float]]] = []
        for it in items:
            s = ce_scores.get((it.problem_name, it.query, it.doc))
            if s is None:
                s = ce_scores.get(("", it.query, it.doc))
            scored_all.append((it, s if s is not None else None))

        # Multiple replicas per query
        for r in range(replicas):
            rng = random.Random(seed * 1_000_003 + r * 97 + hash(key) % 2_147_483_647)
            if strategy == "topmix" and ce_scored_path:
                chosen = select_topmix(scored_all, k, top_frac, rng)
            else:
                chosen = select_random(pos, neg, k, rng)

            # Compute target_scores from teacher if available
            ts = []
            finite_vals = []
            for it in chosen:
                s = ce_scores.get((it.problem_name, it.query, it.doc))
                if s is None:
                    s = ce_scores.get(("", it.query, it.doc))
                ts.append(s)
                if s is not None:
                    finite_vals.append(s)
            target_scores = None
            if any(v is not None for v in ts):
                base = (min(finite_vals) - 1.0) if finite_vals else 0.0
                filled = [float(v) if v is not None else base for v in ts]
                target_scores = softmax(filled, tau=teacher_tau)

            yield {
                "problem_name": pn,
                "query": q,
                "group_id": f"{key}#rep{r}",
                "candidates": [
                    {
                        "text": it.doc,
                        "label": it.label,
                        "weight": it.weight,
                        "ce_score": ce_scores.get((it.problem_name, it.query, it.doc))
                            or ce_scores.get(("", it.query, it.doc)),
                        "meta": it.meta,
                    }
                    for it in chosen
                ],
                "target_scores": target_scores,
            }
            emitted += 1

    print(f"built from queries={total_groups}, emitted groups={emitted} (replicas={replicas}, k={k})")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Build multiple listwise groups per query from large candidate pools")
    p.add_argument("--input", required=True, help="Path to pairs.*.jsonl")
    p.add_argument("--output", required=True, help="Path to write groups.*.jsonl")
    p.add_argument("--k", type=int, default=64, help="Candidates per group")
    p.add_argument("--replicas", type=int, default=4, help="How many groups per query")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--tau", type=float, default=0.5, help="Temperature for teacher softmax")
    p.add_argument("--ce-scored", default=None, help="Optional ce_scored.*.jsonl for teacher scores")
    p.add_argument("--strategy", choices=["random","topmix"], default="topmix")
    p.add_argument("--top-frac", type=float, default=0.25, help="Fraction of K to take from top teacher scores in topmix")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    rows = build_groups_multi(
        pairs_path=args.input,
        k=args.k,
        replicas=args.replicas,
        seed=args.seed,
        teacher_tau=args.tau,
        ce_scored_path=args.ce_scored,
        strategy=args.strategy,
        top_frac=args.top_frac,
    )
    write_jsonl(args.output, rows)
    print(f"Wrote groups to {args.output}")


if __name__ == "__main__":
    main()
