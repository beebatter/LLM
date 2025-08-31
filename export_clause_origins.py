#!/usr/bin/env python3
"""
Export all registered clauses with their first-seen origin from iProver raw logs.

Output JSONL rows shaped as:
{
  "problem_name": "SWB012+3",
  "text": "~iext(uri_owl_hasValue,X0,X1)| ... | iext(X2,X3,X1)",
  "origin": "given"  # or simplified / passive / never / unknown
  # optionally, for negatives, add: "neg_bucket": "NEG_given_nonproof"
}

Usage examples:
  # From a directory of *.raw.log files (e.g., produced by run_batch_pipeline)
  python3 export_clause_origins.py --raw-dir datasets/logs \
      --output datasets/clauses_with_origin.jsonl --strip-tcf

  # From a single raw log
  python3 export_clause_origins.py --raw-file path/to/ALG050+1.raw.log \
      --output out.jsonl --problem-name ALG050+1
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob
from typing import Dict, Iterable, List, Optional, Set, Tuple

# Reuse the robust parser from iplog_to_dataset
try:
    from iplog_to_dataset import parse_log, determine_neg_bucket  # type: ignore
except Exception:
    # Add repo root to path if running from subdir
    REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, REPO_ROOT)
    from iplog_to_dataset import parse_log, determine_neg_bucket  # type: ignore


def infer_problem_name_from_path(path: str) -> str:
    """Infer problem name from filename like 'ALG050+1.raw.log' -> 'ALG050+1'."""
    base = os.path.basename(path)
    if base.endswith('.raw.log'):
        return base[:-len('.raw.log')]
    if base.endswith('.log'):
        return base[:-len('.log')]
    return os.path.splitext(base)[0]


def strip_tcf_text(text: Optional[str]) -> Optional[str]:
    """Extract the clause body from a tcf(...) string when possible.

    Examples of inputs:
      tcf(c_141,plain, (~sP36),file('clausifier', u539)).
      tcf(c_374,plain, (A|B|~C),inference(...), ...)

    We look for the 'plain,' marker, then take the balanced parenthesis
    block that follows it as the clause body.
    """
    if not text or 'tcf(' not in text:
        return text
    try:
        # Find 'plain,'
        key = 'plain,'
        i = text.find(key)
        if i == -1:
            return text
        j = i + len(key)
        # Skip spaces
        while j < len(text) and text[j].isspace():
            j += 1
        if j >= len(text) or text[j] != '(':
            return text
        # Balanced parentheses extraction
        depth = 0
        start = j + 1  # after '('
        k = j
        while k < len(text):
            ch = text[k]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
                if depth == 0:
                    # k is the matching ')'
                    return text[start:k].strip()
            k += 1
        return text
    except Exception:
        return text


def determine_origin(cid: int, seen: Dict[str, Set[int]], registered: bool) -> str:
    """Origin precedence: given > simplified > passive > never > unknown."""
    if cid in seen.get('given', set()):
        return 'given'
    if cid in seen.get('simplified', set()):
        return 'simplified'
    if cid in seen.get('passive', set()):
        return 'passive'
    if registered:
        return 'never'
    return 'unknown'


def iter_raw_logs(raw_file: Optional[str], raw_dir: Optional[str]) -> Iterable[Tuple[str, str]]:
    """Yield (problem_name, path) pairs from a single file or a directory of logs."""
    if raw_file:
        problem = infer_problem_name_from_path(raw_file)
        yield problem, raw_file
        return
    if raw_dir:
        pattern = os.path.join(raw_dir, '*.raw.log')
        files = sorted(glob(pattern))
        for p in files:
            yield infer_problem_name_from_path(p), p


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description='Export per-clause origins from iProver raw logs.')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--raw-file', help='Path to a single *.raw.log file')
    src.add_argument('--raw-dir', help='Directory containing *.raw.log files')
    ap.add_argument('--output', required=True, help='Output JSONL file path')
    ap.add_argument('--problem-name', default=None, help='Override problem_name (for --raw-file)')
    ap.add_argument('--strip-tcf', action='store_true', help='Extract clause body from tcf(...)')
    args = ap.parse_args(argv)

    out_path = args.output
    count = 0
    with open(out_path, 'w', encoding='utf-8') as out:
        for inferred_problem, path in iter_raw_logs(args.raw_file, args.raw_dir):
            problem_name = args.problem_name or inferred_problem
            clauses, proof_ids, seen = parse_log(path)
            # Emit one row per registered clause
            for cid, info in clauses.items():
                text = info.get('text')
                if args.strip_tcf:
                    text = strip_tcf_text(text)
                origin = determine_origin(cid, seen, registered=True)
                row = {
                    'problem_name': problem_name,
                    'text': text,
                    'origin': origin,
                }
                # For negatives, optionally include neg_bucket
                nb = determine_neg_bucket(cid, proof_ids, seen)
                if nb is not None:
                    row['neg_bucket'] = nb
                out.write(json.dumps(row, ensure_ascii=False) + '\n')
                count += 1
    print(f'Wrote {count} rows to {out_path}')


if __name__ == '__main__':
    main()
