#!/usr/bin/env python3
"""
Sanitize listwise JSONL: ensure each line has target_scores as a finite probability vector of length K.

Strategy per line:
- If target_scores exists and has K elements, coerce to floats, replace non-finite with 0, renormalize; if sum<=0 -> fallback.
- Else try target_json.scores as above.
- Else try derive from labels (one-hot on first label==1; else uniform).
Writes a new JSONL with corrected target_scores and target_json.

Usage:
python -m LLM.scripts.sanitize_listwise_jsonl \
  --in  /root/autodl-tmp/Training/datasets/listwise/train.listwise.fused.jsonl \
  --out /root/autodl-tmp/Training/datasets/listwise/train.listwise.fused.sanitized.jsonl
"""
from __future__ import annotations

import argparse
import json
from typing import Any, List, Optional

import numpy as np


def _uniform(K: int) -> List[float]:
    K = max(1, int(K))
    return [1.0 / K] * K


def _from_existing(j: dict, K: int) -> Optional[List[float]]:
    tgt = j.get("target_scores")
    if not (isinstance(tgt, list) and len(tgt) == K):
        obj = j.get("target_json")
        if obj is not None:
            try:
                if isinstance(obj, str):
                    obj = json.loads(obj)
                sc = obj.get("scores") if isinstance(obj, dict) else None
                if isinstance(sc, list) and len(sc) == K:
                    tgt = sc
            except Exception:
                tgt = None
    if isinstance(tgt, list) and len(tgt) == K:
        v = np.asarray([float(x) for x in tgt], dtype=np.float64)
        v[~np.isfinite(v)] = 0.0
        s = float(np.sum(v))
        # If sum is non-positive or non-finite after sanitization, treat as invalid -> let caller try labels
        if not (s > 0 and np.isfinite(s)):
            return None
        v = v / s
        return [float(x) for x in v.tolist()]
    return None


def _from_labels(j: dict, K: int) -> Optional[List[float]]:
    labs = j.get("candidates")
    if isinstance(labs, list) and len(labs) == K:
        idx = None
        for i, c in enumerate(labs):
            try:
                if int(c.get("label", 0)) == 1:
                    idx = i
                    break
            except Exception:
                continue
        if idx is not None:
            v = [0.0] * K
            v[idx] = 1.0
            return v
    return None


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Sanitize listwise JSONL target_scores to be finite probabilities")
    ap.add_argument("--in", dest="inp", required=True, help="Input JSONL path")
    ap.add_argument("--out", dest="out", required=True, help="Output JSONL path")
    ap.add_argument("--limit", type=int, default=0, help="Limit lines (0=all)")
    ap.add_argument("--prefer-labels", action="store_true", help="If existing scores invalid, prefer one-hot from labels before uniform")
    args = ap.parse_args(argv)

    n_in = 0
    n_out = 0
    n_fixed = 0
    with open(args.inp, "r", encoding="utf-8", errors="ignore") as fr, open(args.out, "w", encoding="utf-8") as fw:
        for ln in fr:
            if not ln.strip():
                continue
            try:
                j = json.loads(ln)
            except Exception:
                continue
            n_in += 1
            K = int(j.get("K") or 0)
            if K <= 0:
                continue
            ok = _from_existing(j, K)
            if ok is None:
                ok = _from_labels(j, K)
            if ok is None:
                ok = _uniform(K)
            prev = j.get("target_scores")
            if prev != ok:
                n_fixed += 1
            j["target_scores"] = ok
            j["target_json"] = json.dumps({"scores": ok}, ensure_ascii=False)
            fw.write(json.dumps(j, ensure_ascii=False) + "\n")
            n_out += 1
            if args.limit and n_in >= args.limit:
                break
    print(f"[sanitize] processed {n_in} lines; wrote {n_out}; fixed {n_fixed} -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
