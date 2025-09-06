#!/usr/bin/env python3
import argparse
import json
import os
import random
import sys
import re
import hashlib
from collections import defaultdict, Counter
from typing import Dict, List, Tuple


def norm_text(s: str) -> str:
    if s is None:
        return ''
    s = s.strip()
    s = re.sub(r"\s+", " ", s)
    return s


def md5_text(s: str) -> str:
    return hashlib.md5(s.encode('utf-8', errors='ignore')).hexdigest()


def maybe_wrap(text: str, open_tag: str, close_tag: str) -> str:
    if text is None:
        text = ''
    if open_tag in text and close_tag in text:
        return text
    return f"{open_tag}{text}{close_tag}"


def parse_keep_probs(arg: str, max_bucket: int = 8) -> Dict[int, float]:
    """
    Parse comma- or colon-separated mapping like:
      "1.0,0.8,0.6"  -> {0:1.0,1:0.8,2:0.6}
      "0:1.0,1:0.7,2:0.5" -> {0:1.0,1:0.7,2:0.5}
    Unknown buckets default to last provided value.
    """
    mapping: Dict[int, float] = {}
    if not arg:
        return mapping
    parts = [p.strip() for p in arg.split(',') if p.strip()]
    if all(':' in p for p in parts):
        for p in parts:
            k, v = p.split(':', 1)
            try:
                mapping[int(k)] = float(v)
            except Exception:
                continue
    else:
        for i, p in enumerate(parts):
            try:
                mapping[i] = float(p)
            except Exception:
                continue
    if mapping:
        last = mapping[max(mapping.keys())]
        for b in range(max_bucket + 1):
            if b not in mapping:
                mapping[b] = last
    return mapping


def load_stream(path: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def group_by_problem(items):
    groups = defaultdict(list)
    for j in items:
        pn = j.get('problem_name')
        if pn is None:
            continue
        groups[pn].append(j)
    return groups


def dedup_within_problem(examples: List[dict], key_field: str) -> Tuple[List[dict], int]:
    seen = set()
    out = []
    dups = 0
    for j in examples:
        key = md5_text(norm_text(j.get(key_field) or ''))
        if key in seen:
            dups += 1
            continue
        seen.add(key)
        out.append(j)
    return out, dups


def cap_by_label(examples: List[dict], max_pos: int, max_neg: int) -> List[dict]:
    pos, neg = [], []
    for j in examples:
        lab = int(j.get('label') or 0)
        (pos if lab == 1 else neg).append(j)
    # keep order randomized but stable given seed
    random.shuffle(pos)
    random.shuffle(neg)
    if max_pos is not None:
        pos = pos[:max_pos]
    if max_neg is not None:
        neg = neg[:max_neg]
    return pos + neg


def downsample_negs_by_bucket(examples: List[dict], keep_probs: Dict[int, float]) -> List[dict]:
    if not keep_probs:
        return examples
    out = []
    for j in examples:
        lab = int(j.get('label') or 0)
        if lab == 1:
            out.append(j)
            continue
        meta = j.get('meta') or {}
        b = meta.get('neg_bucket')
        try:
            b = int(b)
        except Exception:
            b = None
        p = keep_probs.get(b, keep_probs.get(max(keep_probs.keys()) if keep_probs else 0, 1.0))
        if random.random() <= float(p):
            out.append(j)
    return out


def main():
    ap = argparse.ArgumentParser(description='Prepare cross-encoder dataset: dedup, balance, split by problem (train/val/test).')
    ap.add_argument('--in', dest='inp', required=True, help='Input crossencoder.jsonl')
    ap.add_argument('--out-prefix', required=True, help='Output prefix; will write {prefix}.train.jsonl/.val.jsonl/.test.jsonl')
    ap.add_argument('--val-problem-frac', type=float, default=0.1, help='Fraction of problems for validation split')
    ap.add_argument('--test-problem-frac', type=float, default=0.1, help='Fraction of problems for test split')
    ap.add_argument('--seed', type=int, default=42)
    ap.add_argument('--dedup', action='store_true', help='Deduplicate within each problem by text_b')
    ap.add_argument('--max-pos-per-problem', type=int, default=None, help='Cap positives per problem (after dedup)')
    ap.add_argument('--max-neg-per-problem', type=int, default=None, help='Cap negatives per problem (after dedup/downsample)')
    ap.add_argument('--neg-keep-probs', type=str, default='', help='Neg keep probs by bucket, e.g., "1.0,0.8,0.6,0.4" or "0:1.0,1:0.7,2:0.5"')
    ap.add_argument('--add-q-d-tags', action='store_true', help='Wrap <Q>…</Q> and <D>…</D> if missing in text_a/text_b')
    ap.add_argument('--exclusive-docs-across-splits', action='store_true', help='Ensure no text_b/doc appears across splits (drop from val/test)')
    args = ap.parse_args()

    random.seed(args.seed)

    keep_probs = parse_keep_probs(args.neg_keep_probs)

    # Stream load and optionally tag-wrap
    def _wrap_if_needed(j: dict) -> dict:
        if args.add_q_d_tags:
            a = j.get('text_a')
            b = j.get('text_b')
            a = maybe_wrap(a or '', '<Q>', '</Q>')
            b = maybe_wrap(b or '', '<D>', '</D>')
            j['text_a'] = a
            j['text_b'] = b
        return j

    items = [_wrap_if_needed(j) for j in load_stream(args.inp)]
    if not items:
        print('No items loaded. Check input path.', file=sys.stderr)
        sys.exit(1)

    groups = group_by_problem(items)

    # Split by problems
    problems = list(groups.keys())
    random.shuffle(problems)
    test_n = max(1, int(len(problems) * args.test_problem_frac))
    val_n = max(1, int(len(problems) * args.val_problem_frac))
    # allocate problems: first test, next val, rest train
    test_probs = set(problems[:test_n])
    val_probs = set(problems[test_n:test_n + val_n])

    stats = {
        'problems_total': len(problems),
    'test_problems': len(test_probs),
    'val_problems': len(val_probs),
        'train_problems': len(problems) - len(val_probs),
        'dups_removed': 0,
        'before_counts': {'pos': 0, 'neg': 0},
        'after_counts': {'pos': 0, 'neg': 0},
    }

    train_out, val_out, test_out = [], [], []

    for pn, exs in groups.items():
        # label counts before
        for j in exs:
            if int(j.get('label') or 0) == 1:
                stats['before_counts']['pos'] += 1
            else:
                stats['before_counts']['neg'] += 1

        # dedup within problem on text_b
        if args.dedup:
            exs, dups = dedup_within_problem(exs, key_field='text_b')
            stats['dups_removed'] += dups

        # downsample negatives by bucket
        exs = downsample_negs_by_bucket(exs, keep_probs)

        # cap per label
        exs = cap_by_label(exs, args.max_pos_per_problem, args.max_neg_per_problem)

        # record after counts
        for j in exs:
            if int(j.get('label') or 0) == 1:
                stats['after_counts']['pos'] += 1
            else:
                stats['after_counts']['neg'] += 1

        if pn in test_probs:
            test_out.extend(exs)
        elif pn in val_probs:
            val_out.extend(exs)
        else:
            train_out.extend(exs)

    # Optionally enforce doc-level exclusivity across splits
    exclus_removed = {'val': 0, 'test': 0}
    if args.exclusive_docs_across_splits:
        def doc_text_md5(j: dict) -> str:
            tb = j.get('text_b') or j.get('doc') or j.get('text') or ''
            return md5_text(norm_text(tb))
        train_md5s = {doc_text_md5(j) for j in train_out}
        # remove from val any doc present in train
        new_val = []
        for j in val_out:
            if doc_text_md5(j) in train_md5s:
                exclus_removed['val'] += 1
                continue
            new_val.append(j)
        val_out = new_val
        # remove from test any doc present in train or val
        val_md5s = {doc_text_md5(j) for j in val_out}
        new_test = []
        for j in test_out:
            md = doc_text_md5(j)
            if md in train_md5s or md in val_md5s:
                exclus_removed['test'] += 1
                continue
            new_test.append(j)
        test_out = new_test

    # Shuffle outputs for better mixing
    random.shuffle(train_out)
    random.shuffle(val_out)
    random.shuffle(test_out)

    out_train = f"{args.out_prefix}.train.jsonl"
    out_val = f"{args.out_prefix}.val.jsonl"
    out_test = f"{args.out_prefix}.test.jsonl"

    with open(out_train, 'w', encoding='utf-8') as ft:
        for j in train_out:
            ft.write(json.dumps(j, ensure_ascii=False) + '\n')
    with open(out_val, 'w', encoding='utf-8') as fv:
        for j in val_out:
            fv.write(json.dumps(j, ensure_ascii=False) + '\n')
    with open(out_test, 'w', encoding='utf-8') as fte:
        for j in test_out:
            fte.write(json.dumps(j, ensure_ascii=False) + '\n')

    # Summarize
    def cnt(exs: List[dict]):
        c = Counter(int(j.get('label') or 0) for j in exs)
        return {'pos': c.get(1, 0), 'neg': c.get(0, 0), 'total': len(exs)}

    summary = {
        'stats': stats,
        'exclusive_docs_removed': exclus_removed,
        'train_counts': cnt(train_out),
        'val_counts': cnt(val_out),
        'test_counts': cnt(test_out),
        'out_train': out_train,
        'out_val': out_val,
        'out_test': out_test,
    }

    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
