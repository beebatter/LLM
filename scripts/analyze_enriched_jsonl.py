#!/usr/bin/env python3
"""
Analyze an enriched JSONL dataset for quality and visualize parameters.

Inputs: JSONL with fields like
  - query: str
  - doc: str
  - problem_name: str
  - label: int (0/1)
  - weight: float
  - meta: dict with keys: source, proof_solver, neg_bucket, overlap_bucket,
          conj_pred_overlap (float), conj_token_jaccard (float),
          unit (0/1), horn (0/1), epr (0/1), born (int), conj_dist (int)

Outputs under --out-dir (default Logs/dataset):
  - report.md: human-readable summary with ASCII charts (fallback if plots unavailable)
  - metrics.json: aggregate metrics and distributions
  - samples.csv: a small table of outlier/issue examples
  - If matplotlib is available:
      - PNG charts for distributions (label, weight, jaccard, overlap, unit/horn/epr, born, conj_dist, sources)

Usage:
  python LLM/scripts/analyze_enriched_jsonl.py \
    --input /root/autodl-tmp/Training/datasets/val_full.enriched.jsonl \
    --out-dir /root/Logs/dataset
"""
from __future__ import annotations

import argparse
import collections
import csv
import json
import math
import os
import statistics
import sys
from typing import Any, Dict, Iterable, List, Tuple


def slen(x: Any) -> int:
    return len(str(x)) if x is not None else 0


def try_import_matplotlib():
    try:
        import matplotlib
        matplotlib.use('Agg')  # headless
        import matplotlib.pyplot as plt
        return plt
    except Exception:
        return None


def read_jsonl(path: str) -> Iterable[Dict[str, Any]]:
    with open(path, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception as e:
                sys.stderr.write(f"[warn] JSON parse error at line {i}: {e}\n")


def histogram(values: List[float], bins: int = 20, lo: float | None = None, hi: float | None = None) -> List[Tuple[float, float, int]]:
    if not values:
        return []
    if lo is None:
        lo = min(values)
    if hi is None:
        hi = max(values)
    if hi == lo:
        hi = lo + 1e-9
    step = (hi - lo) / bins
    edges = [lo + i * step for i in range(bins + 1)]
    counts = [0] * bins
    for v in values:
        idx = int((v - lo) / step)
        if idx < 0:
            idx = 0
        if idx >= bins:
            idx = bins - 1
        counts[idx] += 1
    return [(edges[i], edges[i + 1], counts[i]) for i in range(bins)]


def ascii_bar(count: int, max_count: int, width: int = 40) -> str:
    if max_count <= 0:
        return ''
    n = int(round(count / max_count * width))
    return '█' * n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True, help='Path to enriched JSONL')
    ap.add_argument('--out-dir', default='Logs/dataset', help='Directory for outputs')
    ap.add_argument('--limit', type=int, default=None, help='Optional max lines to read')
    ap.add_argument('--plots', action='store_true', help='Force plot generation if matplotlib is available')
    args = ap.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    plt = try_import_matplotlib()

    # Aggregates
    n_total = 0
    n_errors = 0

    labels: List[int] = []
    weights: List[float] = []
    q_len: List[int] = []
    d_len: List[int] = []
    q_tok: List[int] = []
    d_tok: List[int] = []
    probs: collections.Counter[str] = collections.Counter()
    sources: collections.Counter[str] = collections.Counter()
    solvers: collections.Counter[str] = collections.Counter()
    neg_buckets: collections.Counter[str] = collections.Counter()
    overlap_buckets: collections.Counter[str] = collections.Counter()
    unit_vals: List[int] = []
    horn_vals: List[int] = []
    epr_vals: List[int] = []
    born_vals: List[int] = []
    conj_dist_vals: List[int] = []
    conj_pred_overlap_vals: List[float] = []
    conj_token_jaccard_vals: List[float] = []

    # Issues / outliers collections
    missing_fields: List[Tuple[int, str]] = []
    weird_weights: List[Tuple[int, float]] = []
    long_text_samples: List[Tuple[str, int, int]] = []  # (problem, q_len, d_len)

    # Read
    for idx, rec in enumerate(read_jsonl(args.input), start=1):
        n_total += 1
        if args.limit and n_total > args.limit:
            break
        try:
            lbl = int(rec.get('label'))
            w = float(rec.get('weight', 1.0))
            q = rec.get('query')
            d = rec.get('doc')
            prob = str(rec.get('problem_name', ''))
            meta = rec.get('meta') or {}

            if q is None or d is None:
                missing_fields.append((idx, 'query/doc'))
                continue

            labels.append(lbl)
            weights.append(w)
            qlen = slen(q)
            dlen = slen(d)
            q_len.append(qlen)
            d_len.append(dlen)
            q_tok.append(len(str(q).split()))
            d_tok.append(len(str(d).split()))
            probs[prob] += 1

            src = str(meta.get('source', ''))
            sources[src] += 1
            solvers[str(meta.get('proof_solver', ''))] += 1
            neg_buckets[str(meta.get('neg_bucket', ''))] += 1
            overlap_buckets[str(meta.get('overlap_bucket', ''))] += 1
            for key, dest in (
                ('unit', unit_vals), ('horn', horn_vals), ('epr', epr_vals),
            ):
                v = meta.get(key)
                if v is not None:
                    try:
                        dest.append(int(v))
                    except Exception:
                        pass
            for key, dest in (
                ('born', born_vals), ('conj_dist', conj_dist_vals),
            ):
                v = meta.get(key)
                if v is not None:
                    try:
                        dest.append(int(v))
                    except Exception:
                        pass
            v = meta.get('conj_pred_overlap')
            if v is not None:
                try:
                    conj_pred_overlap_vals.append(float(v))
                except Exception:
                    pass
            v = meta.get('conj_token_jaccard')
            if v is not None:
                try:
                    conj_token_jaccard_vals.append(float(v))
                except Exception:
                    pass

            if not (0 <= w <= 1.0):
                weird_weights.append((idx, w))

            if qlen > 5000 or dlen > 5000:
                long_text_samples.append((prob, qlen, dlen))

        except Exception as e:
            n_errors += 1
            sys.stderr.write(f"[warn] record {idx} error: {e}\n")

    # Aggregate metrics
    def safe_mean(vals: List[float]):
        return (sum(vals) / len(vals)) if vals else None

    def safe_median(vals: List[float]):
        return statistics.median(vals) if vals else None

    def to_pct(counts: collections.Counter[str]) -> Dict[str, float]:
        total = sum(counts.values()) or 1
        return {k: v * 100.0 / total for k, v in counts.items()}

    metrics: Dict[str, Any] = {
        'n_total': n_total,
        'n_errors': n_errors,
        'n_missing': len(missing_fields),
        'label_dist': collections.Counter(labels),
        'weight_mean': safe_mean(weights),
        'weight_median': safe_median(weights),
        'query_len_mean': safe_mean(q_len),
        'query_len_median': safe_median(q_len),
        'doc_len_mean': safe_mean(d_len),
        'doc_len_median': safe_median(d_len),
        'query_tok_mean': safe_mean(q_tok),
        'doc_tok_mean': safe_mean(d_tok),
        'sources': sources,
        'solvers': solvers,
        'neg_buckets': neg_buckets,
        'overlap_buckets': overlap_buckets,
        'unit_ratio': safe_mean(unit_vals),
        'horn_ratio': safe_mean(horn_vals),
        'epr_ratio': safe_mean(epr_vals),
        'born_mean': safe_mean(born_vals),
        'born_median': safe_median(born_vals),
        'conj_dist_mean': safe_mean(conj_dist_vals),
        'conj_dist_median': safe_median(conj_dist_vals),
        'conj_pred_overlap_mean': safe_mean(conj_pred_overlap_vals),
        'conj_pred_overlap_median': safe_median(conj_pred_overlap_vals),
        'conj_token_jaccard_mean': safe_mean(conj_token_jaccard_vals),
        'conj_token_jaccard_median': safe_median(conj_token_jaccard_vals),
        'weird_weights_count': len(weird_weights),
        'long_text_samples_count': len(long_text_samples),
    }

    # Write metrics.json (convert Counters to dicts)
    def serialize(o: Any) -> Any:
        if isinstance(o, collections.Counter):
            return dict(o)
        return o

    with open(os.path.join(args.out_dir, 'metrics.json'), 'w', encoding='utf-8') as f:
        json.dump({k: serialize(v) for k, v in metrics.items()}, f, ensure_ascii=False, indent=2)

    # Prepare report.md with ASCII visuals
    report_lines: List[str] = []
    report_lines.append(f"# Dataset quality report\n")
    report_lines.append(f"Input: `{args.input}`\n\n")
    report_lines.append(f"- Total records: {n_total}\n")
    report_lines.append(f"- Parse errors: {n_errors}\n")
    report_lines.append(f"- Missing query/doc: {len(missing_fields)}\n")
    lab_counter = collections.Counter(labels)
    report_lines.append("\n## Label distribution\n")
    total_labs = sum(lab_counter.values()) or 1
    max_lab = max(lab_counter.values()) if lab_counter else 0
    for lab in sorted(lab_counter.keys()):
        cnt = lab_counter[lab]
        pct = cnt * 100.0 / total_labs
        report_lines.append(f"- label={lab}: {cnt} ({pct:.2f}%) {ascii_bar(cnt, max_lab)}\n")

    # Numeric histograms
    def add_hist_section(title: str, values: List[float], bins: int = 20):
        report_lines.append(f"\n## {title}\n")
        if not values:
            report_lines.append("(no data)\n")
            return
        h = histogram(values, bins=bins)
        if not h:
            report_lines.append("(no variation)\n")
            return
        maxc = max(c for _, _, c in h) or 1
        for lo, hi, c in h:
            report_lines.append(f"[{lo:.3f}, {hi:.3f}) {c:6d} {ascii_bar(c, maxc)}\n")
        report_lines.append("\n")

    add_hist_section('Weights', weights)
    add_hist_section('Query length (chars)', [float(x) for x in q_len])
    add_hist_section('Doc length (chars)', [float(x) for x in d_len])
    add_hist_section('Query tokens', [float(x) for x in q_tok])
    add_hist_section('Doc tokens', [float(x) for x in d_tok])
    add_hist_section('Conj token Jaccard', conj_token_jaccard_vals)
    add_hist_section('Conj predicate overlap', conj_pred_overlap_vals)
    add_hist_section('born', [float(x) for x in born_vals])
    add_hist_section('conj_dist', [float(x) for x in conj_dist_vals])

    # Categorical top-k
    def add_topk_section(title: str, counter: collections.Counter[str], k: int = 10):
        report_lines.append(f"\n## {title} (top {k})\n")
        if not counter:
            report_lines.append("(no data)\n")
            return
        total = sum(counter.values()) or 1
        maxc = max(counter.values()) or 1
        for key, cnt in counter.most_common(k):
            pct = cnt * 100.0 / total
            report_lines.append(f"- {key or '(empty)'}: {cnt} ({pct:.2f}%) {ascii_bar(cnt, maxc)}\n")

    add_topk_section('Sources', sources)
    add_topk_section('Proof solvers', solvers)
    add_topk_section('Neg buckets', neg_buckets)
    add_topk_section('Overlap buckets', overlap_buckets)
    add_topk_section('Problems', probs)

    # Issues
    report_lines.append("\n## Potential issues\n")
    report_lines.append(f"- Weird weights (outside [0,1]): {len(weird_weights)}\n")
    if weird_weights:
        preview = ", ".join([f"#{i}:{w}" for i, w in weird_weights[:10]])
        report_lines.append(f"  - examples: {preview}\n")
    report_lines.append(f"- Excessively long samples (>5k chars): {len(long_text_samples)}\n")
    if long_text_samples:
        preview = ", ".join([f"{p}(q={ql},d={dl})" for p, ql, dl in long_text_samples[:10]])
        report_lines.append(f"  - examples: {preview}\n")

    # Write report
    with open(os.path.join(args.out_dir, 'report.md'), 'w', encoding='utf-8') as f:
        f.write("\n".join(report_lines))

    # Save some samples.csv (long texts or random empty)
    samples_path = os.path.join(args.out_dir, 'samples.csv')
    with open(samples_path, 'w', encoding='utf-8', newline='') as f:
        w = csv.writer(f)
        w.writerow(['problem', 'query_len', 'doc_len'])
        for p, ql, dl in long_text_samples[:50]:
            w.writerow([p, ql, dl])

    # Optional plotting if matplotlib present or --plots requested
    if plt and (args.plots or True):
        def save_hist(vals: List[float], title: str, fname: str, bins: int = 30):
            if not vals:
                return
            plt.figure(figsize=(6, 4))
            plt.hist(vals, bins=bins, color='#1f77b4', edgecolor='white')
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, fname))
            plt.close()

        def save_bar(counter: collections.Counter[str], title: str, fname: str, topk: int = 15):
            if not counter:
                return
            items = counter.most_common(topk)
            labels_, counts_ = zip(*items)
            plt.figure(figsize=(8, max(3, len(items) * 0.3)))
            plt.barh(range(len(items)), counts_, color='#2ca02c')
            plt.yticks(range(len(items)), [x if x else '(empty)' for x in labels_])
            plt.title(title)
            plt.tight_layout()
            plt.savefig(os.path.join(args.out_dir, fname))
            plt.close()

        save_bar(collections.Counter(labels), 'Label distribution', 'labels.png')
        save_hist(weights, 'Weights', 'weights.png')
        save_hist([float(x) for x in q_len], 'Query length (chars)', 'q_len.png')
        save_hist([float(x) for x in d_len], 'Doc length (chars)', 'd_len.png')
        save_hist([float(x) for x in q_tok], 'Query tokens', 'q_tok.png')
        save_hist([float(x) for x in d_tok], 'Doc tokens', 'd_tok.png')
        save_hist(conj_token_jaccard_vals, 'Conj token Jaccard', 'conj_token_jaccard.png')
        save_hist(conj_pred_overlap_vals, 'Conj predicate overlap', 'conj_pred_overlap.png')
        save_hist([float(x) for x in born_vals], 'born', 'born.png')
        save_hist([float(x) for x in conj_dist_vals], 'conj_dist', 'conj_dist.png')
        save_bar(sources, 'Sources (top 15)', 'sources.png')
        save_bar(solvers, 'Proof solvers (top 15)', 'solvers.png')
        save_bar(neg_buckets, 'Neg buckets (top 15)', 'neg_buckets.png')
        save_bar(overlap_buckets, 'Overlap buckets (top 15)', 'overlap_buckets.png')

    print(f"[done] analyzed {n_total} records -> {args.out_dir}")


if __name__ == '__main__':
    main()
