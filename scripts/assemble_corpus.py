"""
Assemble final corpus for training by merging positives and mixing negatives with caps and target ratio.

Inputs:
 - /home/ks/LLM/datasets/interactive_sampled_small_aligned.jsonl
 - /home/ks/LLM/data/teacher_from_casc.jsonl
 - /home/ks/LLM/datasets/negatives_from_diff.jsonl

Output:
 - /home/ks/LLM/datasets/corpus_merged.jsonl

Behavior:
 - Dedupe positives by (problem_name, normalized text).
 - Cap positives per problem (default 2000) to reduce dominance.
 - Sample negatives to reach target neg:pos ratio (default 1.0), respecting available counts and per-problem caps.
 - Keep records schema as in teacher/neg files.
"""

from __future__ import annotations

import json
import os
import random
import re
from pathlib import Path
from typing import Dict, List, Tuple, Set

POS_A = Path('/home/ks/LLM/datasets/interactive_sampled_small_aligned.jsonl')
POS_B = Path('/home/ks/LLM/data/teacher_from_casc.jsonl')
NEG   = Path('/home/ks/LLM/datasets/negatives_from_diff.jsonl')
OUT   = Path('/home/ks/LLM/datasets/corpus_merged.jsonl')

POS_CAP_PER_PROBLEM = 2000
NEG_CAP_PER_PROBLEM = 3000
TARGET_NEG_POS_RATIO = 1.0
SEED = 42


def normalize_clause_text(s: str) -> str:
    s2 = s.strip()
    if s2.endswith('.'): s2 = s2[:-1]
    s2 = re.sub(r"\s+", " ", s2)  # keep spaces but collapse runs
    return s2


def read_jsonl(path: Path):
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def main() -> None:
    random.seed(SEED)
    # Merge positives
    pos_seen: Set[Tuple[str,str]] = set()
    pos_per_problem: Dict[str, List[dict]] = {}
    total_pos = 0
    for path in (POS_A, POS_B):
        for r in read_jsonl(path):
            if r.get('label', 1) != 1:
                continue
            pn = r.get('problem_name')
            txt = r.get('text')
            if not pn or not txt:
                continue
            key = (pn, normalize_clause_text(txt))
            if key in pos_seen:
                continue
            pos_seen.add(key)
            pos_per_problem.setdefault(pn, []).append(r)
            total_pos += 1

    # Cap positives per problem
    for pn, lst in pos_per_problem.items():
        if len(lst) > POS_CAP_PER_PROBLEM:
            random.shuffle(lst)
            pos_per_problem[pn] = lst[:POS_CAP_PER_PROBLEM]

    capped_pos = sum(len(v) for v in pos_per_problem.values())

    # Load negatives grouped
    neg_per_problem: Dict[str, List[dict]] = {}
    for r in read_jsonl(NEG):
        if r.get('label', 0) != 0:
            continue
        pn = r.get('problem_name')
        if not pn:
            continue
        neg_per_problem.setdefault(pn, []).append(r)
    for pn, lst in neg_per_problem.items():
        if len(lst) > NEG_CAP_PER_PROBLEM:
            random.shuffle(lst)
            neg_per_problem[pn] = lst[:NEG_CAP_PER_PROBLEM]

    # Determine negatives target overall and per problem
    target_neg_total = int(TARGET_NEG_POS_RATIO * capped_pos)
    # Distribute proportionally to available negatives per problem
    avail = {pn: len(neg_per_problem.get(pn, [])) for pn in pos_per_problem.keys()}
    total_avail = sum(avail.values()) or 1
    neg_pick_per_problem: Dict[str,int] = {
        pn: min(avail[pn], max(0, int(target_neg_total * (avail[pn]/total_avail))))
        for pn in avail
    }

    # Write out merged corpus
    os.makedirs(OUT.parent, exist_ok=True)
    total_written_pos = 0
    total_written_neg = 0
    with OUT.open('w', encoding='utf-8') as fout:
        for pn, lst in pos_per_problem.items():
            for r in lst:
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
                total_written_pos += 1
        for pn, lst in neg_per_problem.items():
            k = neg_pick_per_problem.get(pn, 0)
            if k <= 0:
                continue
            random.shuffle(lst)
            for r in lst[:k]:
                fout.write(json.dumps(r, ensure_ascii=False) + '\n')
                total_written_neg += 1

    print(f"wrote corpus: {OUT}")
    print(f"positives: {total_written_pos}, negatives: {total_written_neg}, ratio={total_written_neg/(total_written_pos or 1):.2f}")


if __name__ == '__main__':
    main()
