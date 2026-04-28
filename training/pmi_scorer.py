#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from tqdm.auto import tqdm

from LLM.training.token_utils import ensure_special_tokens


def avg_logprob_for_span(model, tok, q: str, c: str, max_len: int = 1024) -> float:
    # Build prompt: keep conjecture and candidate, but only score candidate tokens
    text_q = f"[CONJECTURE]\n<Q> {q} </Q>\n"
    text_c = f"[CANDIDATE]\n<CAND_START> {c} <CAND_END>\n"
    full = text_q + text_c
    enc = tok(full, max_length=max_len, truncation=True, return_tensors="pt")
    input_ids = enc.input_ids.to(model.device)
    attn = enc.attention_mask.to(model.device)

    # find candidate span tokens (after the first <CAND_START>)
    cand_id = tok.convert_tokens_to_ids("<CAND_START>")
    ids = input_ids[0].tolist()
    try:
        start = ids.index(cand_id)
    except ValueError:
        return -1e9
    # score tokens strictly after start, up to <CAND_END> or sequence end
    end_id = tok.convert_tokens_to_ids("<CAND_END>")
    try:
        end = ids.index(end_id, start + 1)
    except ValueError:
        end = len(ids) - 1
    if end <= start + 1:
        return -1e9

    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attn)
        logits = out.logits  # [1,T,V]
        logp = torch.log_softmax(logits[0, :-1, :], dim=-1)  # align with next token
    # Next-token indices to score: positions start..end-1 predict tokens start+1..end
    targets = input_ids[0, 1: end + 1]
    span_logp = logp[start:end, :].gather(1, targets.unsqueeze(1)).squeeze(1)
    return float(span_logp.mean().item())


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="PMI scorer: per-candidate avg log-prob with optional PMI correction; outputs unified predictions JSONL")
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", action="append", required=True, help="pairs.*.jsonl with fields query/doc/label/problem_name")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--lambda-pmi", type=float, default=0.7, help="PMI lambda; 0 means pure cond. avg logP")
    args = ap.parse_args(argv)

    # Load with token alignment (special tokens added)
    res = ensure_special_tokens(args.model, load_kwargs={"device_map": "auto"})
    tok, model = res.tokenizer, res.model
    model.eval()

    # Optional: unconditional scorer cache
    uncond_cache = {}

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for path in args.data:
            with open(path, "r", encoding="utf-8", errors="ignore") as fin:
                for line in tqdm(fin, desc=f"PMI {Path(path).name}"):
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    q = j.get("query") or j.get("text_a")
                    d = j.get("doc") or j.get("text_b")
                    if not q or not d:
                        continue
                    pn = j.get("problem_name") or ""
                    y = float(j.get("label", 0))

                    s_cond = avg_logprob_for_span(model, tok, q, d, max_len=args.max_len)
                    if args.lambda_pmi > 0:
                        if d not in uncond_cache:
                            s_uncond = avg_logprob_for_span(model, tok, "", d, max_len=args.max_len)
                            uncond_cache[d] = s_uncond
                        else:
                            s_uncond = uncond_cache[d]
                        s = s_cond - args.lambda_pmi * s_uncond
                    else:
                        s = s_cond

                    f.write(json.dumps({
                        "problem_name": pn,
                        "query": q,
                        "group_id": f"{pn}||{q}",
                        "doc": d,
                        "label": y,
                        "score": float(s),
                        "llm_score": float(s),
                    }, ensure_ascii=False) + "\n")

    print(f"wrote predictions: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
