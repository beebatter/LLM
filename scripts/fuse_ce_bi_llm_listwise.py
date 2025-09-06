#!/usr/bin/env python3
from __future__ import annotations
"""
Fuse Cross-Encoder, Bi-Encoder, and LLM (direct log-likelihood) scores into soft labels for listwise training.

- Input: listwise JSONL with fields {input, K, ...} per line (as produced by make_listwise_chunks)
- Output: same lines with target_scores replaced by fused soft labels and target_json updated

Fusion: fused = λ_ce * s_ce + λ_bi * s_bi + λ_llm * s_llm_cal
Then p = softmax(fused / tau). s_llm_cal optionally z-scored per window.

Example:
python -m LLM.scripts.fuse_ce_bi_llm_listwise \
  --in /root/autodl-tmp/Training/datasets/listwise/val.listwise.teacher.jsonl \
  --out /root/autodl-tmp/Training/datasets/listwise/val.listwise.teacher.fused32b.jsonl \
  --cross-ckpt /root/autodl-tmp/Training/models/cross_encoder_best.pt \
  --bi-ckpt /root/autodl-tmp/Training/models/biencoder_best.pt \
  --spm /root/autodl-tmp/Training/models/spm_logic_32k.model \
  --llm-model /root/autodl-tmp/models/Goedel-Prover-V2-32B \
  --bits 8 --row-batch 16 --input-max 2048 --target-max 256 \
  --lambda-ce 1.0 --lambda-bi 0.3 --lambda-llm 1.0 --tau 1.0 --llm-zscore
"""

import argparse
import json
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from transformers import BitsAndBytesConfig  # type: ignore

from LLM.scripts.make_listwise_targets_from_teacher import (
    build_cross_model,
    build_bi_model,
    score_cross,
    score_bi,
    _parse_prompt_for_q_and_candidates,
)
from LLM.training.eval_llm_listwise import _direct_scores_for_many_windows


def softmax_np(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    if x.size == 0:
        return x
    # sanitize NaN/Inf before softmax
    x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
    # stable softmax with temperature
    m = float(np.max(x)) if x.size > 0 else 0.0
    x = (x - m) / max(float(tau), 1e-8)
    e = np.exp(np.clip(x, -60.0, 60.0))  # clip to avoid overflow in extreme cases
    s = float(e.sum())
    if not np.isfinite(s) or s <= 0:
        return np.ones_like(x) / max(1, len(x))
    p = e / s
    # final guard
    if not np.all(np.isfinite(p)):
        return np.ones_like(x) / max(1, len(x))
    return p


def build_llm(model_path: str, lora: Optional[str], bits: int, attn: str) -> tuple[AutoModelForCausalLM, AutoTokenizer]:
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if getattr(tok, "pad_token", None) is None:
        tok.pad_token = tok.eos_token
    try:
        tok.truncation_side = "left"
    except Exception:
        pass
    quant: Optional[BitsAndBytesConfig] = None
    torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    device_map = "auto" if torch.cuda.is_available() else None
    if bits == 4:
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=torch_dtype,
        )
    elif bits == 8:
        quant = BitsAndBytesConfig(load_in_8bit=True)
    attn_impl = None
    if attn == "flash":
        attn_impl = "flash_attention_2"
    elif attn == "eager":
        attn_impl = "eager"
    if quant is not None:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path,
            quantization_config=quant,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    else:
        mdl = AutoModelForCausalLM.from_pretrained(
            model_path,
            device_map=device_map,
            torch_dtype=torch_dtype,
            attn_implementation=attn_impl,
        )
    # match tokenizer size
    try:
        target_vocab_size = int(len(tok))
        if getattr(mdl.config, "vocab_size", None) != target_vocab_size:
            mdl.resize_token_embeddings(target_vocab_size)
            try:
                mdl.config.vocab_size = target_vocab_size
            except Exception:
                pass
    except Exception:
        pass
    if lora:
        try:
            from peft import PeftModel  # type: ignore
            mdl = PeftModel.from_pretrained(mdl, lora)
        except Exception as e:
            print(f"[WARN] failed to load LoRA: {e}")
    mdl.eval()
    return mdl, tok


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Fuse CE+Bi+LLM (direct) into listwise soft labels")
    ap.add_argument("--in", dest="in_path", type=Path, required=True)
    ap.add_argument("--out", dest="out_path", type=Path, required=True)
    # CE/Bi
    ap.add_argument("--cross-ckpt", type=str, required=True)
    ap.add_argument("--bi-ckpt", type=str, required=True)
    ap.add_argument("--spm", type=str, required=True)
    # LLM
    ap.add_argument("--llm-model", type=str, required=True)
    ap.add_argument("--llm-lora", type=str, default=None)
    ap.add_argument("--bits", type=int, choices=[4, 8, 16], default=8)
    ap.add_argument("--attn", type=str, choices=["auto", "flash", "eager"], default="eager")
    ap.add_argument("--input-max", type=int, default=2048)
    ap.add_argument("--target-max", type=int, default=256)
    ap.add_argument("--row-batch", type=int, default=16)
    ap.add_argument("--score-type", type=str, choices=["sum-ll", "mean-ll"], default="mean-ll")
    ap.add_argument("--segment", type=str, choices=["full", "text"], default="text")
    ap.add_argument("--llm-zscore", action="store_true")
    ap.add_argument("--llm-calib-tau", type=float, default=1.0)
    ap.add_argument("--llm-calib-bias", type=float, default=0.0)
    # Fusion weights
    ap.add_argument("--lambda-ce", type=float, default=1.0)
    ap.add_argument("--lambda-bi", type=float, default=0.3)
    ap.add_argument("--lambda-llm", type=float, default=1.0)
    ap.add_argument("--tau", type=float, default=1.0, help="final softmax temperature")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # build CE/Bi
    enc, head, tok_ce, _ = build_cross_model(args.cross_ckpt, args.spm, device)
    bi, _ = build_bi_model(args.bi_ckpt, device)
    tok_bi = tok_ce  # same SPM

    # build LLM
    mdl, tok = build_llm(args.llm_model, args.llm_lora, args.bits, args.attn)

    # Read and collect windows
    lines: List[dict] = []
    windows: List[tuple[str, List[tuple[int, str, str]]]] = []
    with open(args.in_path, "r", encoding="utf-8", errors="ignore") as f:
        for li, ln in enumerate(f, start=1):
            if args.limit and len(lines) >= args.limit:
                break
            if not ln.strip():
                continue
            try:
                j = json.loads(ln)
            except Exception:
                continue
            prompt = j.get("input") or j.get("prompt") or ""
            K = int(j.get("K") or 0)
            if not prompt or K <= 0:
                continue
            q, cands = _parse_prompt_for_q_and_candidates(prompt)
            if not q or not cands or len(cands) != K:
                continue
            lines.append(j)
            windows.append((q, cands))

    if not windows:
        print("no valid windows to process")
        return 1

    # Score CE/Bi per window
    s_ce_all: List[List[float]] = []
    s_bi_all: List[List[float]] = []
    for (q, cands) in windows:
        s_ce_all.append(score_cross(enc, head, tok_ce, q, cands, max_len=256, batch=256, device=device))
        s_bi_all.append(score_bi(bi, tok_bi, q, cands, max_len=256, batch=256, device=device))

    # Score LLM direct in global batches
    s_llm_all: List[List[float]] = _direct_scores_for_many_windows(
        mdl, tok, windows,
        input_max=args.input_max,
        target_max=args.target_max,
        row_batch=args.row_batch,
        score_type=args.score_type,
        segment=args.segment,
    )

    # Fuse and write
    wrote = 0
    with open(args.out_path, "w", encoding="utf-8") as w:
        for j, s_ce, s_bi, s_llm in zip(lines, s_ce_all, s_bi_all, s_llm_all):
            v_ce = np.asarray(s_ce, dtype=np.float64)
            v_bi = np.asarray(s_bi, dtype=np.float64)
            v_llm = np.asarray(s_llm, dtype=np.float64)
            if v_ce.size == 0 or v_bi.size == 0 or v_llm.size == 0:
                continue
            # sanitize inputs: replace NaN/Inf to safe finite values
            v_ce = np.nan_to_num(v_ce, nan=0.0, posinf=0.0, neginf=0.0)
            v_bi = np.nan_to_num(v_bi, nan=0.0, posinf=0.0, neginf=0.0)
            v_llm = np.nan_to_num(v_llm, nan=0.0, posinf=0.0, neginf=0.0)
            # Calibrate LLM per-window
            if args.llm_zscore:
                mu = float(v_llm.mean())
                sd = float(v_llm.std())
                if sd > 1e-8:
                    v_llm = (v_llm - mu) / sd
                else:
                    v_llm = v_llm - mu
            v_llm = (v_llm - float(args.llm_calib_bias)) / max(1e-8, float(args.llm_calib_tau))
            fused = float(args.lambda_ce) * v_ce + float(args.lambda_bi) * v_bi + float(args.lambda_llm) * v_llm
            fused = np.nan_to_num(fused, nan=0.0, posinf=0.0, neginf=0.0)
            tgt_arr = softmax_np(fused, tau=float(args.tau))
            # final safety: ensure proper distribution
            if tgt_arr.size == 0 or not np.all(np.isfinite(tgt_arr)):
                tgt_arr = np.ones_like(fused) / max(1, len(fused))
            # normalize to sum=1
            s = float(tgt_arr.sum())
            if s <= 0 or not np.isfinite(s):
                tgt_arr = np.ones_like(fused) / max(1, len(fused))
            else:
                tgt_arr = tgt_arr / s
            tgt = tgt_arr.tolist()
            j["target_scores"] = tgt
            j["target_json"] = json.dumps({"scores": tgt}, ensure_ascii=False)
            w.write(json.dumps(j, ensure_ascii=False) + "\n")
            wrote += 1
    print(f"processed={len(lines)} wrote={wrote} -> {args.out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
