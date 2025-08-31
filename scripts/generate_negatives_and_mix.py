"""
Generate negatives as the set difference: all-clauses minus positives (interactive+teacher),
with per-problem quotas and origin-based mixing. Also print key distribution stats.

Inputs (fixed paths for now):
 - /home/ks/LLM/curated_300.json
 - /home/ks/LLM/datasets/interactive_sampled_small_aligned.jsonl
 - /home/ks/LLM/data/teacher_from_casc.jsonl
 - /home/ks/LLM/datasets/clauses_with_origin.jsonl

Outputs:
 - /home/ks/LLM/datasets/negatives_from_diff.jsonl
 - Summary printed to stdout
"""

from __future__ import annotations

import json
import os
import re
import sys
import collections
from pathlib import Path
from typing import Dict, Set, Tuple

CURATED = Path('/home/ks/LLM/curated_300.json')
INTER = Path('/home/ks/LLM/datasets/interactive_sampled_small_aligned.jsonl')
TEACHER = Path('/home/ks/LLM/data/teacher_from_casc.jsonl')
CLAUSES = Path('/home/ks/LLM/datasets/clauses_with_origin.jsonl')
OUT_NEG = Path('/home/ks/LLM/datasets/negatives_from_diff.jsonl')

# Per-problem negative sampling policy
NEG_R_POS = 3           # target negatives per positive
NEG_MIN = 100           # ensure at least this many negatives if available
NEG_MAX = 3000          # cap per problem
MIX = {                 # mix proportions per origin bucket (normalized per problem on the fly)
    'given': 0.5,
    'simplified': 0.3,
    'passive': 0.18,
    'never': 0.02,
}


def normalize_clause_text(s: str) -> str:
    # Remove whitespace and trailing dot; collapse newlines/tabs
    s2 = s.strip()
    if s2.endswith('.'): s2 = s2[:-1]
    s2 = re.sub(r"\s+", "", s2)
    return s2


def split_top_level_disj(formula: str):
    parts = []
    depth = 0
    start = 0
    for i,ch in enumerate(formula):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '|' and depth == 0:
            parts.append(formula[start:i].strip())
            start = i+1
    parts.append(formula[start:].strip())
    return parts


def feat_is_unit(formula: str) -> int:
    return 1 if len(split_top_level_disj(formula)) == 1 else 0


def feat_is_horn(formula: str) -> int:
    pos = 0
    for lit in split_top_level_disj(formula):
        t = lit.strip()
        if t.startswith('~'): continue
        if '!=' in t and not t.startswith('~'): continue
        pos += 1
        if pos > 1: return 0
    return 1


def feat_is_epr(formula: str) -> int:
    depth = 0
    for ch in formula:
        if ch == '(':
            depth += 1
            if depth >= 2:
                return 0
        elif ch == ')':
            depth -= 1
    return 1


def main() -> None:
    # Load curated problems
    with CURATED.open() as f:
        curated = json.load(f)
    problems = [p['problem_name'] for p in curated['problems'] if 'problem_name' in p]
    PSET: Set[str] = set(problems)

    # Build positive set: (problem_name, norm_text)
    pos_set: Set[Tuple[str,str]] = set()
    pos_counts = collections.Counter()

    for path in (INTER, TEACHER):
        with path.open() as f:
            for line in f:
                try:
                    r = json.loads(line)
                except Exception:
                    continue
                pn = r.get('problem_name')
                if pn not in PSET:
                    continue
                txt = r.get('text')
                if not txt:
                    continue
                key = (pn, normalize_clause_text(txt))
                if key not in pos_set:
                    pos_set.add(key)
                    pos_counts[pn] += 1

    print(f"positives: problems covered {len([p for p in problems if pos_counts[p]>0])}/300, total uniq clauses {len(pos_set)}")

    # Compute per-problem negative targets
    neg_targets: Dict[str,int] = {}
    for pn in problems:
        npos = pos_counts[pn]
        tgt = max(NEG_MIN if npos==0 else 0, min(NEG_MAX, NEG_R_POS * npos))
        # ensure a small baseline for very small npos
        if npos>0:
            tgt = max(tgt, min(NEG_MIN, 5*npos))
        neg_targets[pn] = tgt

    # Streaming pass over all clauses to produce negatives until quotas met
    picked = collections.Counter()          # per problem
    picked_by_origin = {pn: collections.Counter() for pn in problems}
    total_written = 0
    os.makedirs(OUT_NEG.parent, exist_ok=True)
    with CLAUSES.open() as fin, OUT_NEG.open('w', encoding='utf-8') as fout:
        for line in fin:
            try:
                r = json.loads(line)
            except Exception:
                continue
            pn = r.get('problem_name')
            if pn not in PSET:
                continue
            if picked[pn] >= neg_targets[pn]:
                continue
            txt = r.get('text')
            if not txt:
                continue
            norm = normalize_clause_text(txt)
            if (pn, norm) in pos_set:
                continue  # it's a positive clause, skip

            origin = (r.get('origin') or 'unknown').lower()
            if origin not in MIX:
                # map unknown to passive bucket by default
                origin = 'passive'

            # Soft per-origin mix control: only accept if this origin is under its proportion
            tgt = neg_targets[pn]
            # desired count for this origin so far
            desired = int(MIX[origin] * tgt)
            if picked_by_origin[pn][origin] >= desired and picked[pn] < tgt:
                # allow spillover after we have tried hitting desired for all origins
                # simple heuristic: allow if total picked for pn is less than 50% of target
                if picked[pn] > tgt * 0.5:
                    pass
                else:
                    continue

            # Compose output record matching teacher schema but label=0
            unit = feat_is_unit(norm)
            horn = feat_is_horn(norm)
            epr = feat_is_epr(norm)
            out = {
                'problem_name': pn,
                'division': None,
                'url': None,
                'conjecture_text': '',
                'text': txt,
                'features': {
                    'horn': horn,
                    'epr': epr,
                    'unit': unit,
                    'born': -1,
                    'conj_dist': -1,
                },
                'label': 0,
                'neg_bucket': f"NEG_{origin}_nonproof",
                'source': 'neg_allclauses',
                'proof_solver': 'none',
                'sample_weight': 0.5 if origin in ('passive','never') else 0.8,
                'item_id': None,
                'role': 'plain',
                'item_kind': 'cnf',
            }
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            picked[pn] += 1
            picked_by_origin[pn][origin] += 1
            total_written += 1

    # Summary
    done_probs = sum(1 for pn in problems if picked[pn]>0)
    print(f"negatives written: {total_written} across {done_probs} problems")
    # show a few per-origin distributions
    agg = collections.Counter()
    for pn in problems:
        agg.update(picked_by_origin[pn])
    print('negatives origin mix:', dict(agg))


if __name__ == '__main__':
    main()
