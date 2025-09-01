#!/usr/bin/env python3
"""
Build listwise windows for LLM reranking fine-tuning.

Input: one or more JSONL files with clause-level samples. Expected fields per line
(best-effort, multiple aliases supported):

- problem_name: str (grouping key)
- conjecture_text|conjecture|query|text_a: str
- text|clause|text_b|premise: str
- features|meta: dict (optional)
- label: int/bool (1 positive, 0 negative)
- clause_id|id: int (optional; will be auto-assigned if missing)
- neg_bucket: str optional for weighted sampling

Output: JSONL with one item per listwise window, including:
- problem_name, ids (length K), input (prompt text),
- target_json: the expected JSON string {"scores": [s1..sK]} using a soft target distribution,
- target_scores: the numeric list for convenience (KL training later)

This script aligns with the whitepaper §4 (LLM Listwise Rerank) and batch_ranker.py's prompt schema.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from LLM.data_utils.logic_tokenizer import normalize_text, features_to_prefix, PrefixBuckets


def _first(d: Dict, keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _build_candidate_lines(ids: List[int], texts: List[str], tags: List[str]) -> str:
    lines = []
    for i, (cid, t, tg) in enumerate(zip(ids, texts, tags), start=1):
        lines.append(f"- ID {cid}\n- TEXT: {t}\n- TAGS: {tg}")
    return "\n".join(lines)


def _prompt_for_window(conj: str, ids: List[int], texts: List[str], tags: List[str]) -> str:
    K = len(ids)
    header = (
        "你是自动定理证明的子句打分器。给定一个猜想 Q 和 K 个候选子句，"
        "请仅输出 JSON：{\"scores\":[...]}，长度为 K，数字用小数，且和可不必等于 1（下游会 softmax）。\n"
    )
    conj_block = f"[CONJECTURE]\n{conj}\n"
    spec = (
        f"\n[CANDIDATES]  共 {K} 条。每条格式：\n"
        "- ID <整型唯一ID>\n- TEXT: <子句文本>\n- TAGS: <紧凑特征token>\n\n"
        "（以下为候选）\n"
    )
    body = _build_candidate_lines(ids, texts, tags)
    tail = "\n\n请仅输出：\n{\"scores\":[s1, s2, ..., sK]}\n"
    return header + conj_block + spec + body + tail


def _soft_targets(pos_mask: List[int], tau: float = 1.0) -> List[float]:
    # Uniform mass on positives; if none positive, uniform over all.
    K = len(pos_mask)
    n_pos = sum(1 for x in pos_mask if x)
    if n_pos <= 0:
        return [1.0 / K] * K
    base = [1.0 / n_pos if m else 0.0 for m in pos_mask]
    if abs(tau - 1.0) < 1e-9:
        return base
    # Optional sharpening/softening
    import numpy as np
    x = np.array(base, dtype=float)
    x = np.power(x + 1e-12, 1.0 / tau)
    x = x / x.sum()
    return x.tolist()


def _neg_weight(bucket: Optional[str]) -> float:
    if not bucket:
        return 1.0
    b = str(bucket)
    if b == "NEG_given_nonproof":
        return 1.0
    if b == "NEG_simplified":
        return 0.8
    if b == "NEG_passive_only":
        return 0.5
    # failed/others
    return 0.25


def _build_windows_for_problem(
    items: List[Dict],
    conj: str,
    K: int,
    max_windows: Optional[int],
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    pos = [x for x in items if int(_first(x, ["label"]) or 0) == 1]
    neg = [x for x in items if int(_first(x, ["label"]) or 0) == 0]
    if not items:
        return []
    if not pos:
        # ensure at least one window with all negatives (will be uniform targets)
        neg_pool = rng.sample(neg, min(len(neg), K)) if neg else rng.sample(items, min(len(items), K))
        return [neg_pool]

    # Weight negatives by bucket for harder sampling
    weights = [(_neg_weight(_first(x, ["neg_bucket"])) if x in neg else 0.0) for x in items]
    neg_weights = [w for x, w in zip(items, weights) if int(_first(x, ["label"]) or 0) == 0]
    neg_only = neg

    windows = []
    # Strategy: for each positive, make one window seeded by that positive, fill with hard negatives
    for p in pos:
        w = [p]
        need = K - 1
        if neg_only:
            # weighted sampling without replacement (approximate)
            pool = list(neg_only)
            pool_w = list(neg_weights)
            sel: List[Dict] = []
            for _ in range(min(need, len(pool))):
                s = sum(pool_w) or 1.0
                r = rng.random() * s
                acc = 0.0
                idx = 0
                for i, ww in enumerate(pool_w):
                    acc += ww
                    if acc >= r:
                        idx = i
                        break
                sel.append(pool.pop(idx))
                pool_w.pop(idx)
            w.extend(sel)
        # fill if still short
        if len(w) < K:
            more = [x for x in items if x not in w]
            rng.shuffle(more)
            w.extend(more[: K - len(w)])
        windows.append(w[:K])
        if max_windows and len(windows) >= max_windows:
            break
    return windows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Make listwise windows for LLM reranker training")
    ap.add_argument("--input", action="append", required=True, help="JSONL path(s) with clause-level samples")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path (listwise)")
    ap.add_argument("--window", type=int, default=64, help="Window size K")
    ap.add_argument("--max-windows-per-problem", type=int, default=4, help="Cap windows per problem")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    # group by problem
    by_prob: Dict[str, List[Dict]] = defaultdict(list)
    conj_map: Dict[str, str] = {}
    next_cid = 1
    for p in args.input:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                prob = _first(j, ["problem_name", "problem", "name"]) or "_"
                conj = _first(j, ["conjecture_text", "conjecture", "query", "text_a"]) or ""
                if conj and prob not in conj_map:
                    conj_map[prob] = normalize_text(conj)
                # normalize fields for convenience
                row = {
                    "id": _as_int(_first(j, ["clause_id", "id"]) or next_cid),
                    "text": normalize_text(_first(j, ["text", "clause", "text_b", "premise"]) or ""),
                    "features": _first(j, ["features", "meta"]) or {},
                    "label": _as_int(_first(j, ["label"]) or 0),
                    "neg_bucket": _first(j, ["neg_bucket"]) or None,
                }
                if _first(j, ["clause_id", "id"]) is None:
                    next_cid += 1
                by_prob[prob].append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_windows = 0
    with open(out_path, "w", encoding="utf-8") as wf:
        for prob, items in by_prob.items():
            conj = conj_map.get(prob, "")
            if not items or not conj:
                continue
            windows = _build_windows_for_problem(items, conj, args.window, args.max_windows_per_problem, args.seed)
            for w in windows:
                ids = [int(x["id"]) for x in w]
                tags = [features_to_prefix(x.get("features") or {}, PrefixBuckets()).strip() for x in w]
                texts = [x.get("text") or "" for x in w]
                pos_mask = [1 if int(x.get("label") or 0) == 1 else 0 for x in w]
                tgt = _soft_targets(pos_mask, tau=1.0)
                prompt = _prompt_for_window(conj, ids, texts, tags)
                obj = {
                    "problem_name": prob,
                    "K": len(ids),
                    "ids": ids,
                    "input": prompt,
                    "target_json": json.dumps({"scores": tgt}, ensure_ascii=False),
                    "target_scores": tgt,
                }
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_windows += 1
    print(f"Wrote {n_windows} listwise windows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
