#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import List, Dict, Any


def load_jsonl(path: Path, limit: int = 0) -> List[Dict[str, Any]]:
    out = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for i, ln in enumerate(f, start=1):
            if limit and len(out) >= limit:
                break
            try:
                out.append(json.loads(ln))
            except Exception:
                continue
    return out


def analyze(split: str, rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    n = len(rows)
    ks = Counter()
    empty_text = 0
    first_onehot = 0
    uniform_all_equal = 0
    non_finite = 0
    id_pos_counts = Counter()
    problem_names = Counter()

    for r in rows:
        k = int(r.get("K") or 0)
        ks[k] += 1
        problem_names[r.get("problem_name", "")] += 1

        # check candidates text presence by simple heuristic in our flat prompt format
        prompt: str = r.get("input") or ""
        # TEXT: lines are empty in current data -> count as empty text
        if "- TEXT:" in prompt and "- TEXT:\n" in prompt:
            empty_text += 1

        # labels
        scores = r.get("target_scores")
        if not isinstance(scores, list):
            try:
                obj = r.get("target_json")
                if isinstance(obj, str):
                    obj = json.loads(obj)
                scores = obj.get("scores") if isinstance(obj, dict) else None
            except Exception:
                scores = None

        if isinstance(scores, list) and scores:
            # one-hot at first position?
            if scores[0] == 1.0 and sum(1 for x in scores if float(x) == 1.0) == 1 and sum(scores) == 1.0:
                first_onehot += 1
            # uniform distribution?
            if all(float(x) == float(scores[0]) for x in scores):
                uniform_all_equal += 1
            # non-finite values
            if any((isinstance(x, float) and (math.isnan(x) or math.isinf(x))) for x in scores):
                non_finite += 1

        # id positional bias: collect the argmax position in the list if one-hot
        try:
            if isinstance(scores, list) and max(scores, default=-1) > 0:
                argmax = max(range(len(scores)), key=lambda i: scores[i])
                id_pos_counts[argmax] += 1
        except Exception:
            pass

    return {
        "split": split,
        "n": n,
        "K_hist": ks,
        "problems": len(problem_names),
        "empty_TEXT_lines": empty_text,
        "first_item_onehot": first_onehot,
        "uniform_labels": uniform_all_equal,
        "non_finite_labels": non_finite,
        "argmax_pos_hist": id_pos_counts,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Check listwise dataset integrity and common pitfalls")
    ap.add_argument("--train", type=Path, required=True)
    ap.add_argument("--val", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    train = load_jsonl(args.train, args.limit)
    val = load_jsonl(args.val, args.limit)

    rep = {
        "train": analyze("train", train),
        "val": analyze("val", val),
    }

    # Pretty print with simple JSON serializable counters
    def to_jsonable(d):
        if isinstance(d, Counter):
            return {str(k): int(v) for k, v in d.items()}
        if isinstance(d, dict):
            return {k: to_jsonable(v) for k, v in d.items()}
        return d

    print(json.dumps(to_jsonable(rep), ensure_ascii=False, indent=2))
    # Quick guidance
    print("\n[HINT] If empty_TEXT_lines > 0: fill candidate TEXT and keep TAGS. If first_item_onehot is large: avoid positional bias. If uniform_labels is large: prefer soft labels with positives spread.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
