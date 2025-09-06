#!/usr/bin/env python3
"""Three-way grid search for LLM / Bi / CE fusion weights.

Reads an enriched listwise JSONL (each line is a window with 'candidates' and per-candidate
fields 'llm_raw_scores', 'bi_scores', 'ce_scores', and optional 'label').

Writes two outputs next to the input path with the given prefix:
- <out_prefix>.grid.json  : list of grid entries {w_llm, w_bi, w_ce, avg_ndcg, avg_mrr}
- <out_prefix>.per_window.csv : per-window metrics and best local fusion weights

Usage: python scripts/grid_search_three_way.py --input /path/to/enriched.jsonl --step 0.05 --out-prefix /path/to/val.smoke.enriched
"""
import argparse
import json
import math
import csv
from pathlib import Path
import numpy as np


def ndcg_at_k(rels, scores, k=8):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    dcg = 0.0
    for rank, idx in enumerate(order[:k], start=1):
        rel = rels[idx]
        dcg += (2**rel - 1) / math.log2(rank + 1)
    ideal = sorted(rels, reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal[:k], start=1):
        idcg += (2**rel - 1) / math.log2(rank + 1)
    return dcg / idcg if idcg > 0 else 0.0


def mrr(rels, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for rank, idx in enumerate(order, start=1):
        if rels[idx] > 0:
            return 1.0 / rank
    return 0.0


def zscore(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return a.tolist()
    mu = a.mean()
    sd = a.std()
    if sd < 1e-6:
        sd = 1.0
    return ((a - mu) / sd).tolist()


def minmax(arr):
    a = np.array(arr, dtype=float)
    if a.size == 0:
        return a.tolist()
    lo = a.min(); hi = a.max(); den = hi - lo if hi > lo else 1.0
    return ((a - lo) / den).tolist()


def load_windows(path):
    wins = []
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip():
                continue
            wins.append(json.loads(line))
    return wins


def build_normed(wins):
    normed = []
    for w in wins:
        K = len(w.get('candidates', []))
        rels = [int(c.get('label', 0)) for c in w.get('candidates', [])]
        llm = w.get('llm_raw_scores', [0.0] * K)
        bi = w.get('bi_scores', [0.0] * K)
        ce = w.get('ce_scores', [0.0] * K)
        llmz = zscore(llm) if len(llm) > 0 else [0.0] * K
        bi_n = minmax(bi)
        ce_n = minmax(ce)
        normed.append({'rels': rels, 'llm': llmz, 'bi': bi_n, 'ce': ce_n})
    return normed


def run_grid(normed, step=0.05):
    vals = [round(i * step, 10) for i in range(int(1 / step) + 1)]
    grid = []
    for w_llm in vals:
        for w_bi in vals:
            w_ce = 1.0 - w_llm - w_bi
            if w_ce < -1e-12:
                continue
            w_ce = round(w_ce, 10)
            if w_ce < 0.0 - 1e-12:
                continue
            ndcgs = []
            mrrs = []
            for w in normed:
                fused = [w_llm * ll + w_bi * b + w_ce * c for ll, b, c in zip(w['llm'], w['bi'], w['ce'])]
                k = min(8, len(w['rels']))
                ndcgs.append(ndcg_at_k(w['rels'], fused, k=k))
                mrrs.append(mrr(w['rels'], fused))
            avg_ndcg = sum(ndcgs) / len(ndcgs) if ndcgs else 0.0
            avg_mrr = sum(mrrs) / len(mrrs) if mrrs else 0.0
            grid.append({'w_llm': w_llm, 'w_bi': w_bi, 'w_ce': w_ce, 'avg_ndcg': avg_ndcg, 'avg_mrr': avg_mrr})
    # sort by ndcg desc
    grid.sort(key=lambda x: x['avg_ndcg'], reverse=True)
    return grid


def per_window_best(normed, grid):
    # For each window, find best grid entry locally (by ndcg)
    rows = []
    for i, w in enumerate(normed):
        best_local = None
        for g in grid:
            fused = [g['w_llm'] * ll + g['w_bi'] * b + g['w_ce'] * c for ll, b, c in zip(w['llm'], w['bi'], w['ce'])]
            k = min(8, len(w['rels']))
            nd = ndcg_at_k(w['rels'], fused, k=k)
            mr = mrr(w['rels'], fused)
            if best_local is None or nd > best_local['ndcg']:
                best_local = {'ndcg': nd, 'mrr': mr, 'weights': (g['w_llm'], g['w_bi'], g['w_ce'])}
        # baseline single-signal metrics
        nd_llm = ndcg_at_k(w['rels'], w['llm'], k=min(8, len(w['rels'])))
        mr_llm = mrr(w['rels'], w['llm'])
        nd_bi = ndcg_at_k(w['rels'], w['bi'], k=min(8, len(w['rels'])))
        mr_bi = mrr(w['rels'], w['bi'])
        nd_ce = ndcg_at_k(w['rels'], w['ce'], k=min(8, len(w['rels'])))
        mr_ce = mrr(w['rels'], w['ce'])
        rows.append({'window_idx': i, 'ndcg_llm': nd_llm, 'mrr_llm': mr_llm, 'ndcg_bi': nd_bi, 'mrr_bi': mr_bi, 'ndcg_ce': nd_ce, 'mrr_ce': mr_ce, 'best_ndcg': best_local['ndcg'], 'best_mrr': best_local['mrr'], 'best_w_llm': best_local['weights'][0], 'best_w_bi': best_local['weights'][1], 'best_w_ce': best_local['weights'][2]})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--input', required=True, help='enriched jsonl input')
    p.add_argument('--step', type=float, default=0.05, help='grid step (default 0.05)')
    p.add_argument('--out-prefix', required=True, help='output path prefix (no extension)')
    args = p.parse_args()

    inpath = Path(args.input)
    out_prefix = Path(args.out_prefix)
    wins = load_windows(inpath)
    if not wins:
        print('no windows found in', inpath)
        return
    normed = build_normed(wins)
    grid = run_grid(normed, step=args.step)
    # write grid json
    grid_path = out_prefix.with_suffix(out_prefix.suffix + '.grid.json') if out_prefix.suffix else Path(str(out_prefix) + '.grid.json')
    # simpler: just write <out_prefix>.grid.json
    grid_path = Path(str(out_prefix) + '.grid.json')
    with open(grid_path, 'w', encoding='utf-8') as f:
        json.dump(grid, f, ensure_ascii=False, indent=2)
    print('wrote grid to', grid_path)

    # per-window best
    rows = per_window_best(normed, grid)
    csv_path = Path(str(out_prefix) + '.per_window.csv')
    with open(csv_path, 'w', newline='', encoding='utf-8') as cf:
        writer = csv.DictWriter(cf, fieldnames=['window_idx','ndcg_llm','mrr_llm','ndcg_bi','mrr_bi','ndcg_ce','mrr_ce','best_ndcg','best_mrr','best_w_llm','best_w_bi','best_w_ce'])
        writer.writeheader()
        for r in rows:
            writer.writerow(r)
    print('wrote per-window CSV to', csv_path)

    # print top 5 global combos
    print('top 5 grid combos (by avg_ndcg):')
    for g in grid[:5]:
        print(f"w_llm={g['w_llm']:.3f} w_bi={g['w_bi']:.3f} w_ce={g['w_ce']:.3f} avg_ndcg={g['avg_ndcg']:.4f} avg_mrr={g['avg_mrr']:.4f}")


if __name__ == '__main__':
    main()
