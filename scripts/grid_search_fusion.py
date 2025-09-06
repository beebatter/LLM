#!/usr/bin/env python3
from __future__ import annotations

"""
Grid-search calibration and fusion for listwise teachers on a dev JSONL.

This script:
  1) Loads listwise windows (conjecture + K candidates with labels and optional target_scores)
  2) Computes raw per-candidate scores once for available teachers:
     - LLM: direct logits scoring (sum-LL or mean-LL; TEXT or FULL segment); optional PMI baseline (empty conj)
     - CE: Cross-Encoder scores (if ckpt+spm provided)
     - Bi: Bi-Encoder dot-product sims (if ckpt+spm provided)
  3) For each grid setting (z-score?, tau, bias, PMI lambda; fusion weights), produces per-window fused probability p*
  4) Evaluates metrics: Hit@K, NDCG@K (from candidate labels), and optional Pearson to target_scores
  5) Writes a JSON report of the best settings and optionally outputs a fused JSONL with target_scores replaced

Usage example:
python -m LLM.scripts.grid_search_fusion \
  --in-listwise /root/autodl-tmp/Training/datasets/listwise/val.listwise.teacher.jsonl \
  --model /root/autodl-tmp/models/Goedel-Prover-V2-32B --bits 8 --attn eager \
  --input-max 1536 --target-max 128 --row-batch 8 --wins-per-batch 2 \
  --score-type mean-ll --segment text --zscore \
  --tau-grid 0.7,1.0,1.3 --pmi-grid 0.0,0.5,0.7,1.0 \
  --weights-grid 0.15,0.45,0.40;0.20,0.40,0.40 \
  --hit-ks 10,32 --ndcg-ks 10,32 \
  --report /root/autodl-tmp/Training/logs/grid_search_report.json \
  --out-fused /root/autodl-tmp/Training/datasets/listwise/val.listwise.fused.best.jsonl
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from transformers import BitsAndBytesConfig  # type: ignore

from LLM.scripts.make_listwise_targets_from_teacher import (
    _parse_prompt_for_q_and_candidates,
    build_cross_model,
    build_bi_model,
    score_cross,
    score_bi,
)
from LLM.training.eval_llm_listwise import _direct_scores_for_many_windows


def _read_listwise(path: str, limit: int = 0) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            try:
                j = json.loads(ln)
            except Exception:
                continue
            items.append(j)
            if limit and len(items) >= limit:
                break
    return items


def _labels_from_candidates(j: Dict[str, Any]) -> List[int]:
    cands = j.get('candidates') or []
    labs = []
    for c in cands:
        try:
            labs.append(int(c.get('label') or 0))
        except Exception:
            labs.append(0)
    return labs


def _softmax_np(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = (x - x.max()) / max(1e-8, float(tau))
    e = np.exp(x)
    s = float(e.sum())
    if s <= 0:
        return np.ones_like(x) / max(1, len(x))
    return e / s


def _zscore_np(v: np.ndarray) -> np.ndarray:
    mu = float(v.mean())
    sd = float(v.std())
    return (v - mu) / sd if sd > 1e-8 else (v - mu)


def _pearson(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return float('nan')
    k = min(len(a), len(b))
    if k < 2:
        return float('nan')
    av = np.asarray(a[:k], dtype=np.float64)
    bv = np.asarray(b[:k], dtype=np.float64)
    if np.all(av == av[0]) or np.all(bv == bv[0]):
        return float('nan')
    return float(np.corrcoef(av, bv)[0, 1])


def _dcg_at_k(rel: List[int], k: int) -> float:
    if k <= 0:
        return 0.0
    r = np.array(rel[:k], dtype=float)
    discounts = 1.0 / np.log2(np.arange(2, k + 2))
    return float((r * discounts).sum())


def _ndcg_at_k(labels: List[int], scores: List[float], k: int) -> Optional[float]:
    if not labels:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float))
    rel_sorted = [int(labels[i]) for i in order]
    dcg = _dcg_at_k(rel_sorted, k)
    ideal = sorted(labels, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg <= 0:
        return None  # undefined (no positives)
    return dcg / idcg


def _hit_at_k(labels: List[int], scores: List[float], k: int) -> Optional[float]:
    if not labels:
        return None
    order = np.argsort(-np.asarray(scores, dtype=float))
    topk = [int(labels[i]) for i in order[:k]]
    if sum(labels) <= 0:
        return None  # undefined (no positives)
    return 1.0 if any(topk) else 0.0


def _load_hf_llm(model_path: str, lora: Optional[str], bits: int, attn: str):
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if getattr(tok, 'pad_token', None) is None:
        tok.pad_token = tok.eos_token
    try:
        tok.truncation_side = 'left'
    except Exception:
        pass
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = 'auto' if torch.cuda.is_available() else None
    quant: Optional[BitsAndBytesConfig] = None
    if bits == 4:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype)
    elif bits == 8:
        quant = BitsAndBytesConfig(load_in_8bit=True)
    attn_impl = None
    if attn == 'flash':
        attn_impl = 'flash_attention_2'
    elif attn == 'eager':
        attn_impl = 'eager'
    if quant is not None:
        mdl = AutoModelForCausalLM.from_pretrained(model_path, quantization_config=quant, device_map=device_map, torch_dtype=torch_dtype, attn_implementation=attn_impl)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_path, device_map=device_map, torch_dtype=torch_dtype, attn_implementation=attn_impl)
    if lora:
        from peft import PeftModel  # type: ignore
        mdl = PeftModel.from_pretrained(mdl, lora)
    mdl.eval()
    return tok, mdl


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Grid-search calibration and fusion for listwise teachers')
    ap.add_argument('--in-listwise', required=True, help='Dev listwise JSONL')
    ap.add_argument('--limit', type=int, default=0)
    # LLM
    ap.add_argument('--model', required=True)
    ap.add_argument('--lora', default=None)
    ap.add_argument('--bits', type=int, choices=[4,8,16], default=8)
    ap.add_argument('--attn', choices=['auto','flash','eager'], default='eager')
    ap.add_argument('--input-max', type=int, default=1536)
    ap.add_argument('--target-max', type=int, default=128)
    ap.add_argument('--row-batch', type=int, default=8)
    ap.add_argument('--wins-per-batch', type=int, default=2)
    ap.add_argument('--score-type', choices=['sum-ll','mean-ll'], default='mean-ll')
    ap.add_argument('--segment', choices=['full','text'], default='text')
    # CE/Bi (optional)
    ap.add_argument('--cross-ckpt', default=None)
    ap.add_argument('--bi-ckpt', default=None)
    ap.add_argument('--spm', default=None)
    # Grids
    ap.add_argument('--zscore', action='store_true')
    ap.add_argument('--tau-grid', default='1.0')
    ap.add_argument('--bias-grid', default='0.0')
    ap.add_argument('--pmi-grid', default='0.0')
    ap.add_argument('--weights-grid', default='0.20,0.40,0.40')
    # Metrics / outputs
    ap.add_argument('--hit-ks', default='10,32')
    ap.add_argument('--ndcg-ks', default='10,32')
    ap.add_argument('--report', type=Path, required=True)
    ap.add_argument('--out-fused', type=Path, default=None)
    args = ap.parse_args(argv)

    items = _read_listwise(args.in_listwise, args.limit)
    if not items:
        raise SystemExit('No items loaded.')
    # Windows for LLM scoring
    wins: List[Tuple[str, List[Tuple[int,str,str]]]] = []
    labels_per_win: List[List[int]] = []
    tgt_scores_per_win: List[List[float]] = []
    for j in items:
        q = j.get('conjecture') or j.get('conjecture_text')
        cands = j.get('candidates') or []
        labs = _labels_from_candidates(j)
        if (not q or not cands) and j.get('input'):
            qq, tuples = _parse_prompt_for_q_and_candidates(j['input'])
            if qq and tuples:
                q = qq
                cands = [{'id': cid, 'text': text, 'tags': tags} for (cid, text, tags) in tuples]
                labs = [int(x.get('label') or 0) for x in j.get('candidates') or []] if j.get('candidates') else [0]*len(tuples)
        if not q or not cands:
            continue
        triple = [(int(c.get('id') or i+1), str(c.get('text') or ''), str(c.get('tags') or '')) for i, c in enumerate(cands)]
        wins.append((q, triple))
        labels_per_win.append(labs if labs and len(labs)==len(triple) else [0]*len(triple))
        ts = j.get('target_scores') or []
        if not ts and j.get('target_json'):
            try:
                obj = j['target_json'] if isinstance(j['target_json'], dict) else json.loads(j['target_json'])
                if isinstance(obj, dict) and isinstance(obj.get('scores'), list):
                    ts = obj['scores']
            except Exception:
                ts = []
        tgt_scores_per_win.append([float(x) for x in ts] if ts else [])

    # Build teachers
    tok_llm, mdl_llm = _load_hf_llm(args.model, args.lora, args.bits, args.attn)
    use_ce_bi = bool(args.cross_ckpt and args.bi_ckpt and args.spm)
    if use_ce_bi:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        enc, head, tok_ce, _ = build_cross_model(args.cross_ckpt, args.spm, device)
        bi, _ = build_bi_model(args.bi_ckpt, device)

    # Compute raw scores in slices to control memory
    def batched_scores(_wins: List[Tuple[str, List[Tuple[int,str,str]]]]) -> List[List[float]]:
        out: List[List[float]] = []
        for i in range(0, len(_wins), max(1, int(args.wins_per_batch))):
            sl = _wins[i:i+max(1, int(args.wins_per_batch))]
            scs = _direct_scores_for_many_windows(
                mdl_llm, tok_llm, sl, args.input_max, args.target_max, args.row_batch,
                score_type=args.score_type, segment=args.segment,
            )
            out.extend(scs)
        return out

    raw_llm = batched_scores(wins)
    base_llm = batched_scores([("", w[1]) for w in wins])  # PMI baseline

    if use_ce_bi:
        ce_raw: List[List[float]] = []
        bi_raw: List[List[float]] = []
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        for (q, cands) in wins:
            ce_raw.append(score_cross(enc, head, tok_ce, q, cands, max_len=256, batch=256, device=device))
            bi_raw.append(score_bi(bi, tok_ce, q, cands, max_len=256, batch=256, device=device))
    else:
        ce_raw = [[] for _ in wins]
        bi_raw = [[] for _ in wins]

    tau_list = [float(x) for x in str(args.tau_grid).split(',') if x]
    bias_list = [float(x) for x in str(args.bias_grid).split(',') if x]
    pmi_list = [float(x) for x in str(args.pmi_grid).split(',') if x]
    w_sets: List[Tuple[float,float,float]] = []
    for trip in str(args.weights_grid).split(';'):
        if not trip.strip():
            continue
        parts = [float(x) for x in trip.split(',') if x]
        if len(parts) == 3:
            w_sets.append((parts[0], parts[1], parts[2]))

    hitKs = [int(x) for x in str(args.hit_ks).split(',') if x]
    ndcgKs = [int(x) for x in str(args.ndcg_ks).split(',') if x]

    def build_probs(zscore: bool, tau: float, bias: float, pmi_lambda: float, w_bi: float, w_ce: float, w_llm: float):
        preds: List[List[float]] = []
        for wi in range(len(wins)):
            v_llm = np.asarray(raw_llm[wi], dtype=np.float64)
            b_llm = np.asarray(base_llm[wi], dtype=np.float64)
            if v_llm.shape != b_llm.shape:
                # safety
                b_llm = np.zeros_like(v_llm)
            v_llm = v_llm - float(pmi_lambda) * b_llm if pmi_lambda > 0 else v_llm
            if zscore:
                v_llm = _zscore_np(v_llm)
            v_llm = (v_llm - bias) / max(1e-8, tau)
            p_llm = _softmax_np(v_llm, tau=1.0)

            parts = [w_llm * p_llm]
            if use_ce_bi:
                v_ce = np.asarray(ce_raw[wi], dtype=np.float64)
                v_bi = np.asarray(bi_raw[wi], dtype=np.float64)
                if zscore:
                    v_ce = _zscore_np(v_ce)
                    v_bi = _zscore_np(v_bi)
                v_ce = (v_ce - bias) / max(1e-8, tau)
                v_bi = (v_bi - bias) / max(1e-8, tau)
                parts.append(w_ce * _softmax_np(v_ce, tau=1.0))
                parts.append(w_bi * _softmax_np(v_bi, tau=1.0))
            else:
                # fallback to teacher vector if present
                tgt = tgt_scores_per_win[wi]
                if tgt and len(tgt) == len(p_llm):
                    p_exist = np.asarray(tgt, dtype=np.float64)
                    s = float(p_exist.sum())
                    if s > 0:
                        p_exist = p_exist / s
                    else:
                        p_exist = np.ones_like(p_exist) / len(p_exist)
                    parts.append(w_ce * p_exist)

            p = sum(parts)
            s = float(p.sum())
            p = p / s if s > 0 else np.ones_like(p) / len(p)
            preds.append(p.tolist())
        return preds

    def eval_metrics(preds: List[List[float]]):
        # returns dict of aggregated metrics
        hit_res: Dict[int, List[float]] = {k: [] for k in hitKs}
        ndcg_res: Dict[int, List[float]] = {k: [] for k in ndcgKs}
        pears: List[float] = []
        for wi, p in enumerate(preds):
            labs = labels_per_win[wi]
            for k in hitKs:
                hv = _hit_at_k(labs, p, k)
                if hv is not None:
                    hit_res[k].append(hv)
            for k in ndcgKs:
                nv = _ndcg_at_k(labs, p, k)
                if nv is not None:
                    ndcg_res[k].append(nv)
            tgt = tgt_scores_per_win[wi]
            if tgt and len(tgt) == len(p):
                pr = _pearson(tgt, p)
                if not np.isnan(pr):
                    pears.append(float(pr))
        out = {f'hit@{k}': float(np.mean(v)) if v else None for k, v in hit_res.items()}
        out.update({f'ndcg@{k}': float(np.mean(v)) if v else None for k, v in ndcg_res.items()})
        out['pearson_to_teacher'] = float(np.mean(pears)) if pears else None
        out['_counts'] = {'n': len(preds), 'hit_counts': {k: len(v) for k, v in hit_res.items()}, 'ndcg_counts': {k: len(v) for k, v in ndcg_res.items()}, 'pear_cnt': len(pears)}
        return out

    best = {
        'by_ndcg@10': None,
        'by_ndcg@32': None,
        'by_hit@10': None,
        'by_hit@32': None,
        'by_pearson': None,
    }
    best_scores = {k: -1e9 for k in best.keys()}
    tried: List[Dict[str, Any]] = []

    for tau in tau_list:
        for bias in bias_list:
            for pmi in pmi_list:
                for (w_bi, w_ce, w_llm) in w_sets:
                    preds = build_probs(args.zscore, tau, bias, pmi, w_bi, w_ce, w_llm)
                    mets = eval_metrics(preds)
                    row = {
                        'zscore': bool(args.zscore), 'tau': float(tau), 'bias': float(bias), 'pmi': float(pmi),
                        'w_bi': float(w_bi), 'w_ce': float(w_ce), 'w_llm': float(w_llm), 'metrics': mets,
                    }
                    tried.append(row)
                    # update leaders
                    def upd(key: str, val: Optional[float]):
                        if val is None:
                            return
                        if float(val) > float(best_scores[key]):
                            best_scores[key] = float(val)
                            best[key] = row
                    upd('by_ndcg@10', mets.get('ndcg@10'))
                    upd('by_ndcg@32', mets.get('ndcg@32'))
                    upd('by_hit@10',  mets.get('hit@10'))
                    upd('by_hit@32',  mets.get('hit@32'))
                    upd('by_pearson', mets.get('pearson_to_teacher'))

    report = {'input': str(args.in_listwise), 'limit': int(args.limit), 'use_ce_bi': use_ce_bi, 'grid': {'tau': tau_list, 'bias': bias_list, 'pmi': pmi_list, 'weights': w_sets}, 'best': best, 'tried': tried}
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with open(args.report, 'w', encoding='utf-8') as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"[grid] wrote report to {args.report}")

    # Optionally write fused outputs using the best-by-ndcg@32 setting
    if args.out_fused:
        pick = best['by_ndcg@32'] or best['by_hit@10'] or best['by_pearson']
        if pick:
            preds = build_probs(pick['zscore'], pick['tau'], pick['bias'], pick['pmi'], pick['w_bi'], pick['w_ce'], pick['w_llm'])
            args.out_fused.parent.mkdir(parents=True, exist_ok=True)
            with open(args.in_listwise, 'r', encoding='utf-8', errors='ignore') as fr, open(args.out_fused, 'w', encoding='utf-8') as fw:
                i = 0
                for ln in fr:
                    try:
                        j = json.loads(ln)
                    except Exception:
                        continue
                    if i >= len(preds):
                        break
                    j['target_scores'] = preds[i]
                    j['target_json'] = json.dumps({'scores': preds[i]}, ensure_ascii=False)
                    fw.write(json.dumps(j, ensure_ascii=False) + '\n')
                    i += 1
            print(f"[grid] wrote fused outputs to {args.out_fused}")
        else:
            print('[grid] no best setting available to write fused outputs')

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
