#!/usr/bin/env python3
import json, os, sys, collections, math, re
from pathlib import Path
from statistics import mean

ROOT = Path(__file__).resolve().parent
P_CURATED = ROOT / 'curated_200.json'
P_JSONL = ROOT / 'interactive_sampled_small.jsonl'
P_FAILED = ROOT / 'failed.jsonl'


def norm_text(obj: dict):
    for k in ('text','clause','canonical_formula','clause_text'):
        if obj.get(k):
            return obj[k]
    return None


def safe_float(x, default=None):
    try:
        return float(x)
    except Exception:
        return default


def analyze_curated(path: Path):
    if not path.exists():
        return {'exists': False}
    try:
        data = json.load(path.open('r', encoding='utf-8'))
    except Exception:
        return {'exists': True, 'error': 'json_load_failed'}
    probs = data.get('problems', []) if isinstance(data, dict) else []
    domains = [p.get('domain') for p in probs if p.get('domain')]
    dom_ctr = collections.Counter(domains)
    return {
        'exists': True,
        'problems': len(probs),
        'domains_top5': dom_ctr.most_common(5),
        'domains_unique': len(dom_ctr),
        'metadata': data.get('metadata', {}),
    }


def analyze_jsonl(path: Path, head_n: int = 3):
    if not path.exists():
        return {'exists': False}
    labels = collections.Counter()
    buckets = collections.Counter()
    problems = collections.Counter()
    texts = set(); dup_total = 0
    per_problem = {}
    features_presence = collections.Counter()
    len_stats = []
    born_vals = []
    conj_vals = []
    cross_problem_dups = 0
    text_owner = {}
    head_samples = []
    n = 0
    with path.open('r', encoding='utf-8', errors='ignore') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            n += 1
            if len(head_samples) < head_n:
                head_samples.append({k: obj.get(k) for k in ('problem_name','label','neg_bucket','text')})
            pn = obj.get('problem_name') or obj.get('problem')
            if pn:
                problems[pn] += 1
                per_problem.setdefault(pn, {'pos':0,'neg':0})
            lab = obj.get('label')
            if lab is not None:
                labels[lab] += 1
                if pn:
                    if lab == 1:
                        per_problem[pn]['pos'] += 1
                    else:
                        per_problem[pn]['neg'] += 1
            b = obj.get('neg_bucket')
            if b:
                buckets[b] += 1
            txt = norm_text(obj)
            if txt:
                if txt in texts:
                    dup_total += 1
                    owner = text_owner.get(txt)
                    if owner is not None and owner != pn:
                        cross_problem_dups += 1
                texts.add(txt)
                if pn and txt not in text_owner:
                    text_owner[txt] = pn
                len_stats.append(len(txt))
            feats = obj.get('features') or {}
            for k in ('unit','horn','epr','born','conj_dist'):
                if k in feats:
                    features_presence[k] += 1
            if 'born' in feats and isinstance(feats['born'], (int,float)):
                born_vals.append(feats['born'])
            if 'conj_dist' in feats and isinstance(feats['conj_dist'], (int,float)):
                conj_vals.append(feats['conj_dist'])
    pos = labels.get(1,0); neg = labels.get(0,0)
    pos_rate = (pos/(pos+neg)) if (pos+neg)>0 else None
    per_prob_summary = []
    for k,v in per_problem.items():
        tot = v['pos']+v['neg']
        pr = (v['pos']/tot) if tot>0 else 0.0
        per_prob_summary.append((k, v['pos'], v['neg'], pr))
    per_prob_summary.sort(key=lambda x: (-x[1], x[3], x[0]))
    return {
        'exists': True,
        'lines': n,
        'problems': len(problems),
        'labels': dict(labels),
        'pos_rate': pos_rate,
        'buckets': dict(buckets),
        'distinct_texts': len(texts),
        'dup_total': dup_total,
        'cross_problem_dups': cross_problem_dups,
        'features_presence': dict(features_presence),
        'len_avg': mean(len_stats) if len_stats else None,
        'len_p95': sorted(len_stats)[int(0.95*len(len_stats))] if len_stats else None,
        'born_minmax': (min(born_vals), max(born_vals)) if born_vals else None,
        'conj_dist_minmax': (min(conj_vals), max(conj_vals)) if conj_vals else None,
        'head_samples': head_samples,
        'per_problem_top10': per_prob_summary[:10],
    }


def analyze_failed(path: Path, head_n: int = 3):
    if not path.exists():
        return {'exists': False}
    reasons = collections.Counter()
    problems = collections.Counter()
    head = []
    n = 0
    # Stats on last_given_clauses and passive_sample
    lg_count = []  # per-entry count
    ps_count = []
    # feature coverage in last_given_clauses
    feat_cov = collections.Counter()
    try:
        with path.open('r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                n += 1
                if len(head) < head_n:
                    head.append({k: obj.get(k) for k in ('problem_name','reason','elapsed_sec')})
                pn = obj.get('problem_name') or obj.get('problem')
                if pn:
                    problems[pn] += 1
                r = obj.get('reason') or 'unknown'
                reasons[r] += 1
                lg = obj.get('last_given_clauses') or []
                ps = obj.get('passive_sample') or []
                lg_count.append(len(lg))
                ps_count.append(len(ps))
                # feature coverage
                for entry in lg:
                    feats = (entry.get('features') or {})
                    for k in ('unit','horn','epr','born','conj_dist'):
                        if k in feats:
                            feat_cov[k] += 1
    except Exception:
        pass
    def _minmax(xs):
        return (min(xs), max(xs)) if xs else None
    def _avg(xs):
        return (sum(xs)/len(xs)) if xs else None
    return {
        'exists': True,
        'lines': n,
        'problems': len(problems),
        'reasons_top': reasons.most_common(6),
        'last_given_avg_count': _avg(lg_count),
        'last_given_minmax': _minmax(lg_count),
        'passive_sample_avg_count': _avg(ps_count),
        'passive_sample_minmax': _minmax(ps_count),
        'last_given_feat_coverage': dict(feat_cov),
        'head_samples': head,
    }


def main():
    cur = analyze_curated(P_CURATED)
    js = analyze_jsonl(P_JSONL)
    fl = analyze_failed(P_FAILED)
    report = {
        'curated_200.json': cur,
        'interactive_sampled_small.jsonl': js,
        'failed.jsonl': fl,
    }
    out_path = ROOT / 'corpus_quality_report.json'
    try:
        with out_path.open('w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
        print(str(out_path))
    except Exception:
        # still print to stdout
        print(json.dumps(report, ensure_ascii=False, indent=2))

    # Also print brief human-readable summaries
    print('=== curated_200.json ===')
    print(json.dumps(cur, ensure_ascii=False, indent=2))
    print('\n=== interactive_sampled_small.jsonl ===')
    print(json.dumps(js, ensure_ascii=False, indent=2))
    print('\n=== failed.jsonl ===')
    print(json.dumps(fl, ensure_ascii=False, indent=2))

if __name__ == '__main__':
    main()
