#!/usr/bin/env python3
"""Three-way fusion pipeline explorer.

Reads listwise JSONL (train/val). Expects per-window fields:
- 'llm_raw_scores' : list of avg_logP(clause|conj) (float)
- optionally per-window 'llm_marg_raw_scores' : list of avg_logP(clause) (float)
- optionally per-candidate 'bi' and 'ce' fields inside 'candidates'

Performs s_llm = cond - lambda * marg, per-window z-score, softmax with temperature,
computes p^bi/p^ce similarly, then fused = Normalize(w_bi*p_bi + w_ce*p_ce + w_llm*p_llm).
Does grid search over lambda, temperature, and weights; evaluates NDCG/MRR on val set.
"""
import argparse, json, math
from pathlib import Path
import itertools
import numpy as np

def load_jsonl(path):
    rows = []
    with open(path,'r') as f:
        for l in f:
            rows.append(json.loads(l))
    return rows

def zscore(xs):
    a = np.array(xs, dtype=np.float64)
    if a.size==0:
        return a
    m = a.mean(); s = a.std()
    if s < 1e-8:
        return a - m
    return (a - m) / s

def softmax_from_scores(xs, temp=1.0):
    a = np.array(xs, dtype=np.float64) / float(temp)
    if a.size==0:
        return a
    a = a - a.max()
    exp = np.exp(np.clip(a, -60, 60))
    s = exp.sum()
    if s == 0:
        return np.ones_like(exp) / float(exp.size)
    return (exp / s)

def get_array_or_default(win, key, K):
    arr = win.get(key)
    if arr is None:
        return np.zeros(K, dtype=np.float64)
    return np.array(arr, dtype=np.float64)

def compute_method_probs(win, key_raw, temp=1.0, length_norm=False):
    raw = get_array_or_default(win, key_raw, len(win.get('candidates',[])))
    if length_norm:
        # if token counts available as candidate['token_count'], divide
        toks = np.array([c.get('token_count', 1) for c in win.get('candidates',[])], dtype=np.float64)
        toks = np.clip(toks, 1.0, None)
        raw = raw / toks
    z = zscore(raw)
    probs = softmax_from_scores(z, temp=temp)
    return probs

def compute_llm_probs(win, lam=0.0, temp=1.0):
    cond = get_array_or_default(win, 'llm_raw_scores', len(win.get('candidates',[])))
    marg = get_array_or_default(win, 'llm_marg_raw_scores', len(win.get('candidates',[])))
    s = cond - lam * marg
    z = zscore(s)
    return softmax_from_scores(z, temp=temp)

def get_bi_ce_probs(win, temp=1.0):
    K = len(win.get('candidates',[]))
    bi = np.array([c.get('bi', 0.0) for c in win.get('candidates',[])], dtype=np.float64)
    ce = np.array([c.get('ce', 0.0) for c in win.get('candidates',[])], dtype=np.float64)
    pbi = softmax_from_scores(zscore(bi), temp=temp)
    pce = softmax_from_scores(zscore(ce), temp=temp)
    return pbi, pce

def fused_from_components(pbi, pce, pll, w_bi, w_ce, w_llm):
    arr = w_bi * pbi + w_ce * pce + w_llm * pll
    s = arr.sum()
    if s==0:
        K = arr.size
        return np.ones(K, dtype=np.float64)/float(K)
    return arr / s

def ndcg_at_k(rels, scores, k=None):
    if k is None: k = len(scores)
    order = np.argsort(-np.array(scores))
    dcg = 0.0
    for i, idx in enumerate(order[:k]):
        rel = rels[idx]
        dcg += (2**rel - 1) / math.log2(i+2)
    ideal = sorted(rels, reverse=True)[:k]
    idcg = sum((2**r - 1) / math.log2(i+2) for i,r in enumerate(ideal))
    return dcg / idcg if idcg>0 else 0.0

def mrr(rels, scores):
    order = np.argsort(-np.array(scores))
    for i, idx in enumerate(order):
        if rels[idx] > 0:
            return 1.0/(i+1)
    return 0.0

def evaluate_dataset(wins, fused_list):
    ndcgs=[]; mrrs=[]
    for w, fused in zip(wins, fused_list):
        rels = [int(c.get('label',0)) for c in w.get('candidates',[])]
        if len(rels)==0: continue
        ndcgs.append(ndcg_at_k(rels, fused))
        mrrs.append(mrr(rels, fused))
    return float(np.mean(ndcgs)) if ndcgs else 0.0, float(np.mean(mrrs)) if mrrs else 0.0

def grid_search(val_wins, lambdas, temps, weight_triples):
    best = None
    for lam in lambdas:
        for temp in temps:
            # precompute components for val
            pll_list = [compute_llm_probs(w, lam=lam, temp=temp) for w in val_wins]
            pbi_list = []
            pce_list = []
            for w in val_wins:
                pbi, pce = get_bi_ce_probs(w, temp=1.0)
                pbi_list.append(pbi); pce_list.append(pce)

            for (w_llm, w_ce, w_bi) in weight_triples:
                fused = [fused_from_components(pbi_list[i], pce_list[i], pll_list[i], w_bi, w_ce, w_llm) for i in range(len(val_wins))]
                ndcg, mrrv = evaluate_dataset(val_wins, fused)
                score = ndcg  # primary metric
                if best is None or score > best['score']:
                    best = dict(score=score, ndcg=ndcg, mrr=mrrv, lam=lam, temp=temp, w_llm=w_llm, w_ce=w_ce, w_bi=w_bi)
    return best

def enumerated_weight_triples(total=1.0, steps=(0.0,0.2,0.4,0.6,0.8,1.0)):
    triples = []
    for w_llm in steps:
        for w_ce in steps:
            for w_bi in steps:
                if abs(w_llm + w_ce + w_bi - total) < 1e-6:
                    triples.append((w_llm, w_ce, w_bi))
    return triples

def write_fused_jsonl(wins, fused_list, outpath):
    p = Path(outpath)
    with p.open('w') as f:
        for w, fused in zip(wins, fused_list):
            obj = dict(w)
            obj['fused_scores'] = fused.tolist() if isinstance(fused, np.ndarray) else fused
            f.write(json.dumps(obj, ensure_ascii=False) + '\n')

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--val', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--lambda-cands', default='0.0,0.2,0.5,1.0')
    ap.add_argument('--temps', default='0.5,1.0,2.0')
    ap.add_argument('--steps', default='0.0,0.2,0.4,0.6,0.8,1.0')
    args = ap.parse_args()

    val_wins = load_jsonl(args.val)
    lambdas = [float(x) for x in args.lambda_cands.split(',')]
    temps = [float(x) for x in args.temps.split(',')]
    steps = [float(x) for x in args.steps.split(',')]
    weight_triples = enumerated_weight_triples(total=1.0, steps=steps)

    best = grid_search(val_wins, lambdas, temps, weight_triples)
    print('[info] best params:', best)

    # produce fused with best params
    pll_list = [compute_llm_probs(w, lam=best['lam'], temp=best['temp']) for w in val_wins]
    fused = [fused_from_components(*get_bi_ce_probs(w, temp=1.0), pll_list[i], best['w_bi'], best['w_ce'], best['w_llm']) for i,w in enumerate(val_wins)]
    outp = args.out or (args.val + '.fused3.jsonl')
    write_fused_jsonl(val_wins, fused, outp)
    print('[info] wrote fused file to', outp)

if __name__=='__main__':
    main()
