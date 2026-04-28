#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from typing import Iterable, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def read_jsonl(path: str) -> Iterable[dict]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except Exception:
                continue


def parse_args():
    ap = argparse.ArgumentParser(description="Score groups by LM internal rating tokens at <RATE> positions")
    ap.add_argument("--ckpt", required=True, help="Path to fine-tuned token-scorer (from save_pretrained)")
    ap.add_argument("--groups", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--bins", type=int, default=10)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bf16", action="store_true")
    return ap.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.ckpt, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.ckpt,
        torch_dtype=torch.bfloat16 if args.bf16 else None,
        device_map=None,
    ).to(device).eval()

    rating_tokens = [f"<R{i}>" for i in range(args.bins)]
    rate_id = tok.convert_tokens_to_ids("<RATE>")
    rating_ids = [tok.convert_tokens_to_ids(t) for t in rating_tokens]

    fout = open(args.out, "w", encoding="utf-8")
    try:
        for r in read_jsonl(args.groups):
            q = r.get("query") or ""
            cand = r.get("candidates") or []
            text = [f"<Q> {q} </Q>"]
            for c in cand:
                txt = c.get("text") or ""
                text.append(f"<CAND_START> {txt} <CAND_END> <RATE>")
            s = "\n".join(text)
            enc = tok(
                s,
                max_length=args.max_len,
                truncation=True,
                padding="max_length",
                return_tensors="pt",
                add_special_tokens=True,
            ).to(device)

            with torch.no_grad():
                out = model(**enc)
                logits = out.logits  # [1, T, V]
                ids = enc.input_ids[0]
                # find <RATE> positions, take next-token distribution
                pos_list = [i for i, tid in enumerate(ids.tolist()) if tid == rate_id]
                scores = []
                for pos in pos_list:
                    # distribution for token at pos+1
                    if pos + 1 >= logits.size(1):
                        scores.append(0.0)
                        continue
                    prob = logits[0, pos + 1].float().softmax(dim=-1)
                    # expected rating = sum p(Rk)*k / (bins-1) in [0,1]
                    num = 0.0
                    den = 0.0
                    for k, tid in enumerate(rating_ids):
                        p = float(prob[tid].item()) if 0 <= tid < prob.size(-1) else 0.0
                        num += p * k
                        den += p
                    val = (num / max(den, 1e-8)) / max(1, (args.bins - 1))
                    scores.append(val)

            # write unified predictions
            for c, sc in zip(cand, scores):
                obj = {
                    "problem_name": r.get("problem_name") or "",
                    "query": q,
                    "group_id": r.get("group_id") or "",
                    "doc": c.get("text") or "",
                    "label": float(c.get("label", 0.0)),
                    "score": float(sc),
                }
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        fout.close()
    print(f"wrote: {args.out}")


if __name__ == "__main__":
    main()
