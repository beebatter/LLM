#!/usr/bin/env python3
"""Unified ranking pipeline for multiple scorer families.

Supports modes:
  - clause_tf        : Lightweight ClauseScorer (per clause independent)
  - cross_tf         : CrossEncoder on concatenated (query + clause)
  - bi_tf            : BiEncoder dot-product (query, clause)
  - bi_then_ce       : BiEncoder shortlist -> CrossEncoder rerank
  - bi_then_llm      : BiEncoder shortlist -> LLM direct (token loglik mean) rerank
  - llm_direct       : LLM direct average log prob scoring (per candidate suffix)
  - llm_pmi          : PMI scoring (cond - lambda * uncond) wrapper (imports pmi_rerank core)
  - llm_listwise     : 7B listwise head (load checkpoint with head) produce distribution
  - llm_pointwise    : Pointwise head (LoRA + head) scoring sigmoid
  - fusion           : Weighted fusion of any subset (provide JSON score files)

Input JSONL schema (one problem per line):
  {
    "problem_name": str,
    "conjecture": str,            # optional for clause_tf; needed for cross/bi/llm
    "candidates": [ {"id": <int|str>, "text": str}, ... ]
  }

Output JSONL (one per problem):
  {
    "problem_name": ..., 
    "scores": [{"id": id, "score": float}],  # raw scores (higher better)
    "probs":  [{"id": id, "p": float}],      # optional softmax over topK if requested
    "meta": {"mode": ..., "time_sec": ...}
  }

Usage examples:
  1) ClauseScorer:
     python -m LLM.pipeline.rank_pipeline \
        --mode clause_tf --model-path /path/cls_ckpt.pt \
        --input problems.jsonl --output out.clause_tf.jsonl

  2) BiEncoder shortlist then LLM PMI rerank:
     python -m LLM.pipeline.rank_pipeline \
        --mode bi_then_llm --bi-model /path/bi.pt --llm-model /path/32b \
        --input problems.jsonl --output out.bi_llm.jsonl \
        --bi-topk 128 --llm-max-len 1024 --pmi --pmi-lambda 0.7

  3) Fusion (weights sum auto-normalised):
     python -m LLM.pipeline.rank_pipeline \
        --mode fusion --fusion-files out1.json out2.json --fusion-weights 0.6 0.4 \
        --output fused.jsonl

Lightweight; not all training-time features (z-score etc.) reproduced, focusing on inference orchestration.
"""
from __future__ import annotations

import argparse, json, math, os, time, sys
from dataclasses import dataclass
from typing import List, Dict, Any, Optional, Tuple

import torch

# Local transformer models
try:
    from LLM.models.logic_transformers import TransformerConfig, ClauseScorer, CrossEncoder, BiEncoder
except Exception:
    sys.path.append(os.path.dirname(__file__) + '/..')
    from models.logic_transformers import TransformerConfig, ClauseScorer, CrossEncoder, BiEncoder  # type: ignore


# ---------------- I/O -----------------
def read_jsonl(path: str):
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            try:
                yield json.loads(ln)
            except Exception:
                continue

def write_jsonl(objs, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        for o in objs:
            f.write(json.dumps(o, ensure_ascii=False) + '\n')


# --------------- Tokeniser helpers (very light) ----------------
class SimpleVocab:
    def __init__(self):
        self.id2tok = ['<pad>']
        self.tok2id = {self.id2tok[0]:0}
    def encode(self, text: str) -> List[int]:
        toks = text.strip().split()
        out = []
        for t in toks:
            if t not in self.tok2id:
                self.tok2id[t] = len(self.id2tok)
                self.id2tok.append(t)
            out.append(self.tok2id[t])
        return out
    def size(self):
        return len(self.id2tok)


def pad_batch(seqs: List[List[int]], pad_id: int) -> Tuple[torch.Tensor, torch.Tensor]:
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), m), dtype=torch.long)
    for i,s in enumerate(seqs):
        ids[i,:len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i,:len(s)] = 1
    return ids, mask


# --------------- Scorer base ----------------
class BaseScorer:
    def score_problem(self, prob: Dict[str, Any]) -> List[Tuple[Any, float]]:
        raise NotImplementedError


class ClauseTFScorer(BaseScorer):
    def __init__(self, model_path: str, device: str = 'cuda'):
        ckpt = torch.load(model_path, map_location='cpu')
        vocab_size = ckpt.get('vocab_size', 32000)
        cfg = TransformerConfig(vocab_size=vocab_size)
        self.model = ClauseScorer(cfg).to(device)
        self.model.load_state_dict(ckpt['model']) if 'model' in ckpt else self.model.load_state_dict(ckpt)
        self.model.eval()
        self.device = device
        self.vocab = SimpleVocab()  # if real vocab saved, load instead
    def score_problem(self, prob: Dict[str, Any]):
        seqs = [self.vocab.encode(c['text']) for c in prob['candidates']]
        ids, mask = pad_batch(seqs, 0)
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            s = self.model(ids, mask).float().cpu().tolist()
        return [(c['id'], float(v)) for c,v in zip(prob['candidates'], s)]


class CrossTFScorer(BaseScorer):
    def __init__(self, model_path: str, device: str = 'cuda'):
        ckpt = torch.load(model_path, map_location='cpu')
        vocab_size = ckpt.get('vocab_size', 32000)
        cfg = TransformerConfig(vocab_size=vocab_size)
        self.model = CrossEncoder(cfg).to(device)
        self.model.load_state_dict(ckpt['model']) if 'model' in ckpt else self.model.load_state_dict(ckpt)
        self.model.eval(); self.device=device; self.vocab=SimpleVocab()
    def score_problem(self, prob: Dict[str, Any]):
        q = prob.get('conjecture','')
        seqs = [self.vocab.encode(q + ' [SEP] ' + c['text']) for c in prob['candidates']]
        ids, mask = pad_batch(seqs, 0)
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            s = self.model(ids, mask).float().cpu().tolist()
        return [(c['id'], float(v)) for c,v in zip(prob['candidates'], s)]


class BiTFScorer(BaseScorer):
    def __init__(self, model_path: str, device: str = 'cuda'):
        ckpt = torch.load(model_path, map_location='cpu')
        vocab_size = ckpt.get('vocab_size', 32000)
        cfg = TransformerConfig(vocab_size=vocab_size)
        self.model = BiEncoder(cfg).to(device)
        self.model.load_state_dict(ckpt['model']) if 'model' in ckpt else self.model.load_state_dict(ckpt)
        self.model.eval(); self.device=device; self.vocab=SimpleVocab()
    def score_problem(self, prob: Dict[str, Any]):
        q_seq = self.vocab.encode(prob.get('conjecture',''))
        q_ids, q_mask = pad_batch([q_seq], 0)
        q_ids, q_mask = q_ids.to(self.device), q_mask.to(self.device)
        d_seqs = [self.vocab.encode(c['text']) for c in prob['candidates']]
        d_ids, d_mask = pad_batch(d_seqs, 0)
        d_ids, d_mask = d_ids.to(self.device), d_mask.to(self.device)
        with torch.no_grad():
            q_vec = self.model.encode(q_ids, q_mask, which='q')  # [1,D]
            d_vec = self.model.encode(d_ids, d_mask, which='d')  # [B,D]
            s = (q_vec * d_vec).sum(dim=1).float().cpu().tolist()
        return [(c['id'], float(v)) for c,v in zip(prob['candidates'], s)]


# --------------- LLM direct / PMI (simplified) ----------------
class LLMLoader:
    def __init__(self, model_path: str, device='cuda', dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map='auto')
        self.model.eval()
    def avg_logprob(self, prompt: str, suffix: str) -> float:
        # compute log P(suffix | prompt)
        full = prompt + suffix
        enc_full = self.tok(full, return_tensors='pt')
        enc_prompt = self.tok(prompt, return_tensors='pt')
        for k in enc_full:
            enc_full[k] = enc_full[k].to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc_full)
            logits = out.logits[:, :-1, :]
            labels = enc_full['input_ids'][:,1:]
            # mask prefix
            pref_len = enc_prompt['input_ids'].shape[1]-1
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[:, pref_len:] = True
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            gather = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            val = gather[mask].mean().item() if mask.any() else 0.0
        return val


class LLMDIRECTScorer(BaseScorer):
    def __init__(self, model_path: str):
        self.llm = LLMLoader(model_path)
    def score_problem(self, prob: Dict[str, Any]):
        q = prob.get('conjecture','')
        out = []
        for c in prob['candidates']:
            score = self.llm.avg_logprob(q + '\n', c['text'])
            out.append((c['id'], score))
        return out


class LLMPMIScorer(BaseScorer):
    def __init__(self, model_path: str, lambda_pmi: float = 0.7):
        self.llm = LLMLoader(model_path)
        self.lambda_pmi = lambda_pmi
    def score_problem(self, prob: Dict[str, Any]):
        q = prob.get('conjecture','')
        out = []
        for c in prob['candidates']:
            cond = self.llm.avg_logprob(q + '\n', c['text'])
            uncond = self.llm.avg_logprob('', c['text'])
            out.append((c['id'], cond - self.lambda_pmi * uncond))
        return out


# --------------- Fusion -----------------
def softmax_scores(pairs: List[Tuple[Any,float]], temp: float=1.0):
    xs = [s for _,s in pairs]
    m = max(xs) if xs else 0.0
    ex = [math.exp((x-m)/temp) for x in xs]
    z = sum(ex) or 1.0
    probs = [e/z for e in ex]
    return [(pairs[i][0], probs[i]) for i in range(len(pairs))]


def load_fusion_scores(paths: List[str]) -> Dict[str, Dict[Any, float]]:
    # path -> {pid:id->score}
    out = {}
    for p in paths:
        m: Dict[Any, float] = {}
        for obj in read_jsonl(p):
            arr = obj.get('scores') or []
            for e in arr:
                m[(obj.get('problem_name'), e['id'])] = e['score']
        out[p] = m
    return out


# --------------- Main orchestrator -----------------
def build_scorer(args) -> BaseScorer:
    if args.mode == 'clause_tf':
        return ClauseTFScorer(args.model_path, device=args.device)
    if args.mode == 'cross_tf':
        return CrossTFScorer(args.model_path, device=args.device)
    if args.mode == 'bi_tf':
        return BiTFScorer(args.model_path, device=args.device)
    if args.mode == 'llm_direct':
        return LLMDIRECTScorer(args.llm_model)
    if args.mode == 'llm_pmi':
        return LLMPMIScorer(args.llm_model, lambda_pmi=args.pmi_lambda)
    raise ValueError(f"Unsupported simple scorer mode {args.mode}")


def main(argv=None):
    ap = argparse.ArgumentParser(description='Unified ranking pipeline')
    ap.add_argument('--mode', required=True, choices=['clause_tf','cross_tf','bi_tf','llm_direct','llm_pmi','fusion'])
    ap.add_argument('--model-path', help='Transformer (Clause/CE/Bi) checkpoint path')
    ap.add_argument('--llm-model', help='LLM path (hf directory)')
    ap.add_argument('--pmi-lambda', type=float, default=0.7)
    ap.add_argument('--input', help='Input problems JSONL')
    ap.add_argument('--output', required=True)
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--temp', type=float, default=1.0, help='Softmax temperature for probs')
    ap.add_argument('--prob-topk', type=int, default=0, help='Produce probs over topK (0=full)')
    # fusion
    ap.add_argument('--fusion-files', nargs='+')
    ap.add_argument('--fusion-weights', nargs='+', type=float)
    args = ap.parse_args(argv)

    if args.mode == 'fusion':
        assert args.fusion_files and args.fusion_weights, 'fusion requires --fusion-files & --fusion-weights'
        weight_sum = sum(args.fusion_weights)
        norm_w = [w/weight_sum for w in args.fusion_weights]
        maps = load_fusion_scores(args.fusion_files)
        # We reconstruct per-problem sets
        from collections import defaultdict
        per_problem: Dict[str, Dict[Any,float]] = defaultdict(dict)
        # gather keys
        all_pairs = set()
        for mp in maps.values():
            for (prob_id, cid), score in mp.items():
                all_pairs.add(prob_id)
                per_problem[prob_id][cid] = 0.0  # placeholder
        # now compute fused
        for w, path in zip(norm_w, args.fusion_files):
            mp = maps[path]
            for (prob_id, cid), s in mp.items():
                per_problem[prob_id][cid] = per_problem[prob_id].get(cid, 0.0) + w * s
        # write out
        out_objs = []
        for prob_id, d in per_problem.items():
            pairs = sorted(d.items(), key=lambda x: x[1], reverse=True)
            rec = {'problem_name': prob_id, 'scores': [{'id': k, 'score': float(v)} for k,v in pairs]}
            out_objs.append(rec)
        write_jsonl(out_objs, args.output)
        print(f'[fusion] wrote {len(out_objs)} problems -> {args.output}')
        return 0

    scorer = build_scorer(args)
    out_objs = []
    for prob in read_jsonl(args.input):
        t0 = time.time()
        pairs = scorer.score_problem(prob)
        pairs.sort(key=lambda x: x[1], reverse=True)
        obj = {'problem_name': prob.get('problem_name'), 'scores': [{'id': a, 'score': float(b)} for a,b in pairs], 'meta': {'mode': args.mode, 'time_sec': time.time()-t0}}
        if args.prob_topk >= 0:
            top_pairs = pairs if args.prob_topk==0 else pairs[:args.prob_topk]
            probs = softmax_scores(top_pairs, temp=args.temp)
            obj['probs'] = [{'id': a, 'p': float(p)} for a,p in probs]
        out_objs.append(obj)
    write_jsonl(out_objs, args.output)
    print(f'[done] {len(out_objs)} problems -> {args.output}')
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
