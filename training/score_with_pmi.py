#!/usr/bin/env python3
from __future__ import annotations

"""
Compute PMI-based teacher distributions for listwise groups.

Input: groups.*.jsonl with fields: {"problem_name","query","candidates":[{"text":...}], ...}
Output: same JSONL but with target_scores replaced by a PMI-based softmax distribution per group.

Score per candidate c_i:
  s_i = avg_logP(c_i | conj) - lambda * avg_logP(c_i)
Then z-score within the group and apply (tau, b):
  p_i = softmax(( (s_i - mu) / (sigma + 1e-6) - b ) / tau)

Notes:
 - Uses a simple non-chat prompt:
     [CONJECTURE]\n{conj}\n\n[CANDIDATE]\n{cand}\n
     For unconditional: [CANDIDATE]\n{cand}\n
 - Truncation is applied on raw strings before tokenization via --q-max-chars/--cand-max-chars.
 - Batching is supported across candidates; unconditional scores are cached in-memory and can optionally be saved.
"""

import argparse
import json
import math
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue


def write_jsonl(path: str, rows: Iterable[dict]):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fmt_prefix_conj(conj: str) -> str:
    return f"[CONJECTURE]\n{conj}\n\n[CANDIDATE]\n"


def fmt_prefix_uncond() -> str:
    return f"[CANDIDATE]\n"


def trunc_text(s: str, max_chars: int) -> str:
    if max_chars and isinstance(s, str) and len(s) > max_chars:
        return s[:max_chars]
    return s


@torch.no_grad()
def avg_logp_batch(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    pairs: List[Tuple[str, str]],  # list of (prefix_text, span_text)
    max_len: int,
    device: torch.device,
    dtype: torch.dtype,
) -> List[float]:
    """Compute average log-prob over span_text tokens for each (prefix, span) pair.

    Returns a list of average log-probs (float) per input; if a span is empty after truncation,
    returns -inf to signal unusable candidate.
    """
    if not pairs:
        return []

    # Encode prefix-only lengths to locate span start; then encode full to actual ids
    pref_ids_list = []
    full_ids_list = []
    full_attn_list = []
    span_masks = []  # mask over shifted positions where labels should be counted

    # Pre-encode and build per-sample mask
    for (pref, span) in pairs:
        # Full encoding
        enc_full = tok(pref + span, return_tensors="pt", truncation=True, max_length=max_len)
        ids = enc_full["input_ids"][0]
        attn = enc_full.get("attention_mask", None)
        # Prefix length (standalone); best-effort alignment
        enc_pref = tok(pref, return_tensors="pt", truncation=True, max_length=max_len)
        pref_len = enc_pref["input_ids"].shape[1]
        T = ids.shape[0]
        # shifted positions length
        if T < 2:
            # too short
            span_mask = torch.zeros((0,), dtype=torch.bool)
        else:
            # span starts at pref_len, but ensure within [1, T-1] for shifted labels
            start = max(1, min(pref_len, T - 1))
            # span ends at T-1 (inclusive index for shifted labels)
            end = T - 1
            if end <= start - 1:
                span_mask = torch.zeros((T - 1,), dtype=torch.bool)
            else:
                span_mask = torch.zeros((T - 1,), dtype=torch.bool)
                span_mask[start - 1 : end] = True
        pref_ids_list.append(enc_pref["input_ids"][0])
        full_ids_list.append(ids)
        full_attn_list.append(attn[0] if attn is not None else torch.ones_like(ids))
        span_masks.append(span_mask)

    # Pad to batch
    max_T = max(x.shape[0] for x in full_ids_list)
    B = len(full_ids_list)
    input_ids = torch.full((B, max_T), tok.pad_token_id or tok.eos_token_id, dtype=torch.long)
    attn_mask = torch.zeros((B, max_T), dtype=torch.long)
    for i in range(B):
        T = full_ids_list[i].shape[0]
        input_ids[i, :T] = full_ids_list[i]
        attn_mask[i, :T] = full_attn_list[i]
    input_ids = input_ids.to(device)
    attn_mask = attn_mask.to(device)

    # Forward
    out = model(input_ids=input_ids, attention_mask=attn_mask)
    logits = out.logits[:, :-1, :].to(dtype=torch.float32)
    next_ids = input_ids[:, 1:]
    log_probs = F.log_softmax(logits, dim=-1).gather(-1, next_ids.unsqueeze(-1)).squeeze(-1)

    # Average over span mask per sample
    avgs: List[float] = []
    for i in range(B):
        mask = span_masks[i].to(device)
        Tm = min(mask.shape[0], log_probs.shape[1])
        if Tm <= 0:
            avgs.append(float("-inf"))
            continue
        m = mask[:Tm]
        lp = log_probs[i, :Tm]
        denom = m.sum().item()
        if denom <= 0:
            avgs.append(float("-inf"))
        else:
            val = float((lp[m].sum() / denom).item())
            avgs.append(val)
    return avgs


def process_groups(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    groups_path: str,
    out_path: str,
    lambda_pmi: float,
    tau: float,
    bias: float,
    max_len: int,
    q_max_chars: int,
    cand_max_chars: int,
    batch: int,
    device: torch.device,
) -> None:
    # In-memory cache for unconditional scores
    uncond_cache: Dict[str, float] = {}

    def gen():
        for row in read_jsonl(groups_path):
            q = row.get("query") or ""
            cands = row.get("candidates") or []
            if not q or not cands:
                yield row
                continue
            # Prepare texts
            q_t = trunc_text(q, q_max_chars)
            cand_texts = [trunc_text(c.get("text", ""), cand_max_chars) for c in cands]
            # Compute conditional avg_logP(c|q)
            pref = fmt_prefix_conj(q_t)
            pairs = [(pref, c) for c in cand_texts]
            cond_scores: List[float] = []
            for i in range(0, len(pairs), batch):
                chunk = pairs[i : i + batch]
                cond_scores.extend(avg_logp_batch(model, tok, chunk, max_len=max_len, device=device, dtype=next(model.parameters()).dtype))
            # Compute unconditional avg_logP(c)
            pref_u = fmt_prefix_uncond()
            uncond_scores: List[float] = []
            need_pairs: List[Tuple[str, str]] = []
            need_idx: List[int] = []
            for i, c in enumerate(cand_texts):
                if c in uncond_cache:
                    uncond_scores.append(uncond_cache[c])
                else:
                    uncond_scores.append(0.0)  # placeholder
                    need_pairs.append((pref_u, c))
                    need_idx.append(i)
            for i in range(0, len(need_pairs), batch):
                chunk = need_pairs[i : i + batch]
                vals = avg_logp_batch(model, tok, chunk, max_len=max_len, device=device, dtype=next(model.parameters()).dtype)
                for j, v in enumerate(vals):
                    idx = need_idx[i + j]
                    uncond_scores[idx] = v
                    uncond_cache[cand_texts[idx]] = v
            # Combine: PMI -> z-score -> softmax
            K = len(cand_texts)
            # Handle invalid spans (-inf)
            valid = [i for i in range(K) if math.isfinite(cond_scores[i]) and math.isfinite(uncond_scores[i])]
            if len(valid) < 2:
                # fallback to uniform
                row["target_scores"] = [1.0 / K] * K
                yield row
                continue
            raw = [cond_scores[i] - lambda_pmi * uncond_scores[i] for i in range(K)]
            mu = sum(raw) / K
            var = sum((x - mu) ** 2 for x in raw) / K
            sigma = math.sqrt(max(var, 0.0))
            if not math.isfinite(sigma) or sigma < 1e-6:
                row["target_scores"] = [1.0 / K] * K
                yield row
                continue
            z = [ (x - mu) / (sigma + 1e-6) for x in raw ]
            logits = [ (zi - bias) / max(1e-6, tau) for zi in z ]
            m = max(logits)
            exps = [math.exp(x - m) for x in logits]
            s = sum(exps)
            p = [e / s for e in exps]
            row["target_scores"] = p
            yield row

    write_jsonl(out_path, gen())


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Compute PMI-based teacher distributions for groups JSONL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--in", dest="inp", required=True, help="Input groups.*.jsonl")
    ap.add_argument("--out", required=True, help="Output groups.*.jsonl with PMI target_scores")
    ap.add_argument("--lambda-pmi", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=0.9)
    ap.add_argument("--bias", type=float, default=0.0)
    ap.add_argument("--max-len", type=int, default=1536)
    ap.add_argument("--q-max-chars", type=int, default=256)
    ap.add_argument("--cand-max-chars", type=int, default=256)
    ap.add_argument("--batch", type=int, default=4, help="Micro-batch size (number of candidates per forward)")
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, local_files_only=Path(args.model).exists())
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        device_map="auto",
        local_files_only=Path(args.model).exists(),
    )
    model.eval()

    process_groups(
        model=model,
        tok=tok,
        groups_path=args.inp,
        out_path=args.out,
        lambda_pmi=args.lambda_pmi,
        tau=args.tau,
        bias=args.bias,
        max_len=args.max_len,
        q_max_chars=args.q_max_chars,
        cand_max_chars=args.cand_max_chars,
        batch=max(1, args.batch),
        device=device,
    )
    print(f"Wrote PMI groups to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
