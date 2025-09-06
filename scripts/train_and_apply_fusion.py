#!/usr/bin/env python3
"""Train a small fusion model (LLM-dominant) on listwise JSONL and apply it to produce fused listwise teacher scores.

Produces: <input>.fused.jsonl and a training CSV.

Fallback behavior: if scikit-learn is not available, uses a simple heuristic fusion (alpha on LLM z-score).
"""
import argparse
import json
import math
import random
from pathlib import Path

def load_windows(path):
    wins = []
    with open(path, 'r') as f:
        for i,l in enumerate(f,1):
            wins.append(json.loads(l))
    return wins

def build_candidate_rows(wins):
    rows = []
    for w_idx, w in enumerate(wins):
        raw = w.get('llm_raw_scores') or []
        K = len(raw)
        # per-window z-score of raw logits
        if K>0:
            mean = sum(raw)/K
            var = sum((x-mean)**2 for x in raw)/K
            std = math.sqrt(var) if var>1e-12 else 1.0
        else:
            mean = 0.0; std = 1.0

        entropy = w.get('llm_softmax_entropy', 0.0)
        truncated = 1.0 if w.get('llm_truncated_any') else 0.0

        cands = w.get('candidates', [])
        for i in range(K):
            cand = cands[i] if i < len(cands) else {}
            text = cand.get('text','')
            lab = int(cand.get('label', 0))
            llm_raw = float(raw[i])
            llm_z = (llm_raw - mean)/std
            # length feature (tokens approx by whitespace)
            text_len = len(text.split())
            tag = cand.get('tags','')
            tag_b3 = 1.0 if '<B3>' in tag else 0.0
            rows.append({
                'win_idx': w_idx,
                'problem_name': w.get('problem_name'),
                'cand_idx': i,
                'llm_raw': llm_raw,
                'llm_z': llm_z,
                'entropy': float(entropy),
                'truncated': truncated,
                'text_len': text_len,
                'tag_b3': tag_b3,
                'label': lab,
            })
    return rows

def train_fusion(rows, holdout_frac=0.2, random_seed=42):
    # split by window (so all candidates of a window go to same split)
    wins = {}
    for r in rows:
        wins.setdefault(r['win_idx'], []).append(r)
    win_idxs = sorted(wins.keys())
    random.Random(random_seed).shuffle(win_idxs)
    n_hold = max(1, int(len(win_idxs)*holdout_frac))
    hold = set(win_idxs[:n_hold])
    train_rows = [r for r in rows if r['win_idx'] not in hold]
    val_rows = [r for r in rows if r['win_idx'] in hold]

    # feature matrix: llm_z, entropy, truncated, text_len, tag_b3
    def build_Xy(rs):
        X = [[r['llm_z'], r['entropy'], r['truncated'], float(r['text_len']), r['tag_b3']] for r in rs]
        y = [r['label'] for r in rs]
        return X,y

    Xtr,ytr = build_Xy(train_rows)
    Xv,yv = build_Xy(val_rows)

    model = None
    scaler = None
    try:
        from sklearn.linear_model import LogisticRegression
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler().fit(Xtr)
        Xtr_s = scaler.transform(Xtr)
        Xv_s = scaler.transform(Xv)
        clf = LogisticRegression(max_iter=200, class_weight='balanced')
        clf.fit(Xtr_s, ytr)
        model = ('sklearn_logreg', clf, scaler)
    except Exception as e:
        print('[warn] sklearn unavailable or failed to train, falling back to heuristic fusion:', e)
        # fallback: fused = alpha * llm_z + (1-alpha) * (-text_len normalized)
        # choose alpha=0.85 to prefer LLM
        model = ('heuristic', 0.85)

    return model, (train_rows, val_rows)

def apply_model_to_windows(model, wins, rows):
    # produce fused probabilities per candidate per window
    fused_by_win = {}
    if model[0]=='sklearn_logreg':
        clf = model[1]
        scaler = model[2]
        # for each row compute prob
        feats = [[r['llm_z'], r['entropy'], r['truncated'], float(r['text_len']), r['tag_b3']] for r in rows]
        Xs = scaler.transform(feats)
        probs = clf.predict_proba(Xs)[:,1]
        for r,p in zip(rows,probs):
            fused_by_win.setdefault(r['win_idx'], []).append((r['cand_idx'], float(p)))
    else:
        alpha = model[1]
        # normalize text_len to [0,1] per-window
        by_win = {}
        for r in rows:
            by_win.setdefault(r['win_idx'], []).append(r)
        for win_idx, rs in by_win.items():
            lens = [r['text_len'] for r in rs]
            minl = min(lens); maxl = max(lens)
            den = max(1, maxl-minl)
            for r in rs:
                len_norm = (r['text_len']-minl)/den
                fused = alpha * r['llm_z'] + (1-alpha) * (1.0 - len_norm)
                # map fused via sigmoid to [0,1]
                p = 1.0/(1.0+math.exp(-fused))
                fused_by_win.setdefault(win_idx, []).append((r['cand_idx'], float(p)))

    # assemble outputs: for each window keep candidates in original order
    outputs = []
    for win_idx, w in enumerate(wins):
        K = len(w.get('candidates', []))
        arr = [0.0]*K
        items = fused_by_win.get(win_idx, [])
        for cid, p in items:
            if 0 <= cid < K:
                arr[cid] = p
        outputs.append(arr)
    return outputs

def ndcg_at_k(relevances, scores, k=8):
    # relevances: list of binary labels, scores: predicted scores
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    dcg = 0.0
    for rank, idx in enumerate(order[:k], start=1):
        rel = relevances[idx]
        dcg += (2**rel - 1) / math.log2(rank+1)
    # ideal
    ideal = sorted(relevances, reverse=True)
    idcg = 0.0
    for rank, rel in enumerate(ideal[:k], start=1):
        idcg += (2**rel - 1) / math.log2(rank+1)
    return dcg / idcg if idcg>0 else 0.0

def mrr(relevances, scores):
    order = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    for rank, idx in enumerate(order, start=1):
        if relevances[idx]>0:
            return 1.0/rank
    return 0.0

def evaluate_fused(wins, fused_scores):
    ndcgs=[]; mrrs=[]
    for w, fs in zip(wins, fused_scores):
        rels = [int(c.get('label',0)) for c in w.get('candidates',[])]
        ndcgs.append(ndcg_at_k(rels, fs, k=len(fs)))
        mrrs.append(mrr(rels, fs))
    return sum(ndcgs)/len(ndcgs) if ndcgs else 0.0, sum(mrrs)/len(mrrs) if mrrs else 0.0

def write_fused_jsonl(wins, fused, outpath):
    p = Path(outpath)
    with p.open('w') as f:
        for w, fs in zip(wins, fused):
            obj = dict(w)
            obj['fused_scores'] = fs
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def write_training_csv(rows, path):
    import csv
    keys = ['win_idx','problem_name','cand_idx','llm_raw','llm_z','entropy','truncated','text_len','tag_b3','label']
    with open(path,'w',newline='') as f:
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        for r in rows:
            writer.writerow({k:r.get(k,'') for k in keys})

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--input', required=True)
    ap.add_argument('--out', default=None)
    ap.add_argument('--train-csv', default=None)
    ap.add_argument('--holdout-frac', type=float, default=0.2)
    args = ap.parse_args()

    wins = load_windows(args.input)
    rows = build_candidate_rows(wins)
    model, (train_rows, val_rows) = train_fusion(rows, holdout_frac=args.holdout_frac)

    fused = apply_model_to_windows(model, wins, rows)
    ndcg, mean_mrr = evaluate_fused(wins, fused)
    print(f'[info] fused ndcg (avg)={ndcg:.4f}, mrr={mean_mrr:.4f}')

    outp = args.out or (args.input + '.fused.jsonl')
    write_fused_jsonl(wins, fused, outp)
    print('[info] wrote fused jsonl to', outp)
    csvp = args.train_csv or (args.input + '.training.csv')
    write_training_csv(rows, csvp)
    print('[info] wrote training csv to', csvp)

if __name__=='__main__':
    main()
