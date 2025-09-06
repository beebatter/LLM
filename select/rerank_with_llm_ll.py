#!/usr/bin/env python3
from __future__ import annotations

"""
Non-generative LLM reranker using conditional log-likelihood.

For each (query, candidate), compute the log-probability of candidate TEXT tokens
given a prompt that contains the conjecture and compact TAGS. This avoids relying
on any JSON-formatted generation. Softmax is applied across candidates to get
normalized scores per query.

Input (same as cross-encoder reranker): JSONL where each line has
  {"query": str, "candidate_ids": [int,...]} or {"query": str, "candidates": [str,...]}
If using candidate_ids, --index-meta is required to map id -> text + features.

Output: JSONL with {"scores": [{"id": int|None, "text": str, "score": float}, ...]} per line,
sorted by score desc (optionally top-k capped).
"""

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from transformers import BitsAndBytesConfig  # type: ignore

from LLM.data_utils.logic_tokenizer import normalize_text, features_to_prefix, PrefixBuckets


def _wrap_pair(conj: str, text: str, features: Optional[Dict]) -> Tuple[str, str]:
    conj_block = f"[CONJECTURE]\n{normalize_text(conj)}\n\n"
    prefix = features_to_prefix(features or {}, PrefixBuckets())
    cand_block = f"[CANDIDATE]\n- TEXT: {normalize_text(text)}\n- TAGS: {prefix.strip()}\n"
    # We will compute log P(target | prompt). Return (prompt, target_text)
    return conj_block, cand_block


@torch.no_grad()
def _sum_target_logprob(model, tok, prompt: str, target: str, input_max: int, target_max_toks: int, model_dev: torch.device) -> float:
    # Tokenize target with truncation (hard cap per-candidate)
    if target_max_toks and target_max_toks > 0:
        t_ids = tok.encode(target, add_special_tokens=False, truncation=True, max_length=int(target_max_toks))
    else:
        t_ids = tok.encode(target, add_special_tokens=False)
    # Compute prompt budget and encode prompt with left-truncation
    keep_p = max(0, int(input_max) - len(t_ids))
    if keep_p > 0:
        p_ids = tok.encode(prompt, add_special_tokens=False, truncation=True, max_length=int(keep_p))
        # enforce left-side by slicing tail if tokenizer didn't honor side
        if len(p_ids) > keep_p:
            p_ids = p_ids[-keep_p:]
    else:
        p_ids = []
    ids = p_ids + t_ids
    if not ids or not t_ids:
        return float('-inf')

    input_ids = torch.tensor([ids], dtype=torch.long, device=model_dev)
    outputs = model(input_ids=input_ids, use_cache=False)
    logits = outputs.logits  # [1, L, V]
    # Shift logits by one to align with labels (standard LM next-token)
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]

    # We only want the region corresponding to target tokens. Target starts at idx = len(p_ids)
    start = max(0, len(p_ids) - 1)  # because labels are shifted by one
    end = start + len(t_ids)
    # Guard bounds
    L = labels.shape[1]
    start = min(max(0, start), L)
    end = min(max(start, end), L)
    if end <= start:
        return float('-inf')
    # Gather token logprobs for the target span
    span_lp = logprobs[0, start:end, :].gather(-1, labels[0, start:end].unsqueeze(-1)).squeeze(-1)
    return float(span_lp.sum().item())


@torch.no_grad()
def _batch_sum_logprob(model, input_id_batches: List[List[int]], p_lens: List[int], t_lens: List[int], device: torch.device) -> List[float]:
    """Compute sum log-prob for multiple sequences in one forward pass.
    Each sequence is [prompt_ids + target_ids]. p_lens[i] is len(prompt_ids), t_lens[i] is len(target_ids).
    Returns list of summed logprobs over the target span per sequence.
    """
    if not input_id_batches:
        return []
    max_len = max(len(x) for x in input_id_batches)
    pad_id = 0
    input_ids = torch.full((len(input_id_batches), max_len), pad_id, dtype=torch.long, device=device)
    attn = torch.zeros_like(input_ids)
    for i, ids in enumerate(input_id_batches):
        L = len(ids)
        input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=device)
        attn[i, :L] = 1
    outputs = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
    logits = outputs.logits  # [B, L, V]
    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
    labels = input_ids[:, 1:]
    res: List[float] = []
    for i in range(len(input_id_batches)):
        start = max(0, p_lens[i] - 1)
        end = start + t_lens[i]
        L = labels.shape[1]
        start = min(max(0, start), L)
        end = min(max(start, end), L)
        if end <= start:
            res.append(float('-inf'))
            continue
        lp = logprobs[i, start:end, :].gather(-1, labels[i, start:end].unsqueeze(-1)).squeeze(-1)
        res.append(float(lp.sum().item()))
    return res


@dataclass
class Cand:
    id: Optional[int]
    text: str
    features: Optional[Dict]


def load_meta(meta_path: Path) -> Tuple[List[str], List[Optional[Dict]]]:
    texts: List[str] = []
    feats: List[Optional[Dict]] = []
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            texts.append(j.get("text") or j.get("canonical_formula") or "")
            feats.append(j.get("features") or None)
    return texts, feats


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rerank with LLM conditional log-likelihood (no JSON parsing)")
    ap.add_argument("--input", type=Path, required=True, help="JSONL with {query, candidate_ids|candidates}")
    ap.add_argument("--index-meta", type=Path, help=".meta.jsonl to map ids->text+features (required if using candidate_ids)")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    ap.add_argument("--topk", type=int, default=64, help="Top-K to emit per query (cap)")
    ap.add_argument("--limit", type=int, default=0, help="limit number of queries processed (0=all)")
    ap.add_argument("--batch", type=int, default=8, help="candidate scoring batch size per query")
    # LLM load
    ap.add_argument("--model", type=str, required=True)
    ap.add_argument("--lora", type=str, default=None)
    ap.add_argument("--bits", type=int, choices=[4, 8, 16], default=4)
    ap.add_argument("--bf16", action="store_true")
    # Context control
    ap.add_argument("--input-max", type=int, default=2048, help="cap (prompt+target) tokens; left-trunc")
    ap.add_argument("--target-max-toks", type=int, default=256, help="cap candidate target tokens")
    # Optional RoPE scaling
    ap.add_argument("--rope-type", type=str, default="none", choices=["none", "linear", "dynamic", "yarn"])
    ap.add_argument("--rope-factor", type=float, default=1.0)
    ap.add_argument("--rope-base", type=int, default=None)
    args = ap.parse_args(argv)

    # Tokenizer & model
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    try:
        tok.truncation_side = "left"  # keep tail
    except Exception:
        pass
    # Raise tokenizer max length to avoid spurious warnings; we control length explicitly
    try:
        tok.model_max_length = max(getattr(tok, "model_max_length", 4096) or 4096, 100000)  # type: ignore[attr-defined]
    except Exception:
        pass
    torch_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16
    device_map = "auto" if torch.cuda.is_available() else None
    quant: Optional[BitsAndBytesConfig] = None
    if args.bits == 4:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype)
    elif args.bits == 8:
        quant = BitsAndBytesConfig(load_in_8bit=True)

    if quant is not None:
        mdl = AutoModelForCausalLM.from_pretrained(args.model, quantization_config=quant, device_map=device_map, torch_dtype=torch_dtype)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(args.model, device_map=device_map, torch_dtype=torch_dtype)
    if args.rope_type != "none" and args.rope_factor and args.rope_factor > 1.0:
        try:
            cfg = mdl.config
            orig = int(args.rope_base) if args.rope_base else int(getattr(cfg, "max_position_embeddings", 4096))
            cfg.rope_scaling = {"type": args.rope_type, "factor": float(args.rope_factor), "original_max_position_embeddings": int(orig)}
            if hasattr(cfg, "max_position_embeddings"):
                try:
                    cfg.max_position_embeddings = int(orig * args.rope_factor)
                except Exception:
                    pass
            print(f"[INFO] RoPE scaling: {cfg.rope_scaling}")
        except Exception as e:
            print(f"[WARN] RoPE scaling failed: {e}")
    if args.lora:
        from peft import PeftModel  # type: ignore
        mdl = PeftModel.from_pretrained(mdl, args.lora)
    mdl.eval()
    # Use the embedding device to place input IDs safely even with device_map=auto
    try:
        model_dev = mdl.get_input_embeddings().weight.device
    except Exception:
        try:
            model_dev = getattr(mdl, "device", None) or next(mdl.parameters()).device
        except Exception:
            model_dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load meta for candidate_ids
    meta_texts: List[str] = []
    meta_feats: List[Optional[Dict]] = []
    if args.index_meta is not None:
        meta_texts, meta_feats = load_meta(args.index_meta)

    out_f = open(args.out, "w", encoding="utf-8", buffering=1)  # line-buffered for visibility
    n = 0
    # Precount queries for ETA
    total = 0
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if line.strip():
                total += 1
            if args.limit and total >= args.limit:
                break
    import time
    t0 = time.time()
    print(f"[INFO] queries={total} | batch={args.batch} | input_max={args.input_max} | target_max_toks={args.target_max_toks}")
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            if args.limit and n >= args.limit:
                break
            q = j.get("query") or j.get("conjecture_text") or j.get("q") or ""
            if not q:
                out_f.write(json.dumps({"scores": []}, ensure_ascii=False) + "\n")
                continue
            topk = int(j.get("topk", args.topk))
            cand_ids = j.get("candidate_ids")
            cand_texts = j.get("candidates")
            cands: List[Cand] = []
            if cand_ids is not None:
                if not meta_texts:
                    raise RuntimeError("--index-meta is required when using candidate_ids")
                for cid in cand_ids:
                    try:
                        t = meta_texts[int(cid)]
                        ftr = meta_feats[int(cid)] if meta_feats else None
                    except Exception:
                        t, ftr = "", None
                    cands.append(Cand(id=int(cid), text=t, features=ftr))
            elif cand_texts is not None:
                for t in cand_texts:
                    cands.append(Cand(id=None, text=str(t), features=None))
            else:
                out_f.write(json.dumps({"scores": []}, ensure_ascii=False) + "\n")
                continue

            # Prepare batches of input ids for this query
            seqs: List[List[int]] = []
            p_lens: List[int] = []
            t_lens: List[int] = []
            for c in cands:
                prompt, target = _wrap_pair(q, c.text, c.features)
                # Tokenize target with truncation
                if args.target_max_toks and args.target_max_toks > 0:
                    t_ids = tok.encode(target, add_special_tokens=False, truncation=True, max_length=int(args.target_max_toks))
                else:
                    t_ids = tok.encode(target, add_special_tokens=False)
                keep_p = max(0, int(args.input_max) - len(t_ids))
                if keep_p > 0:
                    p_ids = tok.encode(prompt, add_special_tokens=False, truncation=True, max_length=int(keep_p))
                    if len(p_ids) > keep_p:
                        p_ids = p_ids[-keep_p:]
                else:
                    p_ids = []
                seqs.append(p_ids + t_ids)
                p_lens.append(len(p_ids))
                t_lens.append(len(t_ids))

            # Run in mini-batches to trade throughput/VRAM
            ll_scores: List[float] = []
            for i in range(0, len(seqs), args.batch):
                chunk = seqs[i:i+args.batch]
                pl = p_lens[i:i+args.batch]
                tl = t_lens[i:i+args.batch]
                ll_scores.extend(_batch_sum_logprob(mdl, chunk, pl, tl, device=model_dev))

            # Softmax across candidates to get normalized scores
            import math
            if ll_scores:
                m = max(ll_scores)
                exps = [math.exp(x - m) for x in ll_scores]
                z = sum(exps) or 1.0
                probs = [x / z for x in exps]
            else:
                probs = []

            scored = [
                {"id": c.id, "text": c.text, "score": float(p)}
                for c, p in zip(cands, probs)
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            out = {"scores": scored[:topk]}
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n += 1
            if n % 5 == 0 or n == total:
                dt = time.time() - t0
                qps = n / dt if dt > 0 else 0.0
                eta = (total - n) / qps if qps > 0 else float('inf')
                print(f"[PROG] {n}/{total} | qps={qps:.2f} | eta={int(eta)//60:02d}m{int(eta)%60:02d}s")
    out_f.close()
    print(f"wrote {n} lines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
