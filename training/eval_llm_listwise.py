#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Any, Optional, Tuple

import numpy as np
import requests

# optional in-process inference
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from transformers import BitsAndBytesConfig  # type: ignore

# reuse JSON extractor from batch_ranker for robustness
from LLM.batch_ranker import extract_json
from LLM.scripts.make_listwise_targets_from_teacher import _parse_prompt_for_q_and_candidates


def _flatten_scores(x: Any) -> List[float]:
    """Coerce model/target scores into a list[float]. Accepts:
    - {"scores": [float,...]}
    - {"scores": [{"score":..}, ...]}
    - [float,...]
    - any other -> []
    """
    data = x
    if isinstance(x, dict) and "scores" in x:
        data = x["scores"]
    if isinstance(data, list):
        out = []
        for it in data:
            if isinstance(it, (int, float)):
                out.append(float(it))
            elif isinstance(it, dict) and ("score" in it):
                try:
                    out.append(float(it.get("score", 0.0)))
                except Exception:
                    out.append(0.0)
            else:
                # ignore unknown shapes
                continue
        return out
    return []


def _pearsonr(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return float('nan')
    n = min(len(a), len(b))
    if n < 2:
        return float('nan')
    a1 = np.asarray(a[:n], dtype=np.float64)
    b1 = np.asarray(b[:n], dtype=np.float64)
    if np.all(a1 == a1[0]) or np.all(b1 == b1[0]):
        return float('nan')
    return float(np.corrcoef(a1, b1)[0, 1])


@torch.no_grad()
def _inprocess_generate(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    prompts: List[str],
    device: Optional[torch.device],
    input_max_len: int,
    max_new_tokens: int,
    temperature: float,
    batch_size: int = 4,
) -> List[str]:
    outs: List[str] = []
    # Resolve a concrete device for inputs to avoid CPU/CUDA mismatches under PEFT/device_map=auto
    try:
        model_dev = getattr(model, "device", None)
        if model_dev is None:
            model_dev = next(model.parameters()).device  # type: ignore
    except Exception:
        import torch as _torch
        model_dev = _torch.device("cuda" if _torch.cuda.is_available() else "cpu")
    for i in range(0, len(prompts), batch_size):
        batch = prompts[i : i + batch_size]
        # Left-side truncation to keep the task instruction and candidate list tail
        enc = tok(
            batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=int(input_max_len),
        )
        # Always move to model device to satisfy embedding lookup
        enc = enc.to(model_dev)
        gen = model.generate(
            **enc,
            do_sample=False,
            temperature=1.0,
            max_new_tokens=int(max_new_tokens),
            eos_token_id=tok.eos_token_id,
            pad_token_id=getattr(tok, "pad_token_id", tok.eos_token_id),
            use_cache=True,
        )
        texts = tok.batch_decode(gen, skip_special_tokens=True)
        for p, t in zip(batch, texts):
            if t.startswith(p):
                t = t[len(p):]
            outs.append(t.strip())
    return outs


@torch.no_grad()
def _direct_scores_for_window(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    conj: str,
    cands: List[tuple[int, str, str]],
    input_max: int,
    target_max: int,
    score_type: str = "sum-ll",
    segment: str = "text",
) -> List[float]:
    conj_block = f"[CONJECTURE]\n{conj}\n\n"
    t_ids_list: List[List[int]] = []
    t_text_len_list: List[int] = []  # number of tokens belonging to the TEXT line (without TAGS)
    p_ids_list: List[List[int]] = []
    for (_, text, tags) in cands:
        blk_full = f"[CANDIDATE]\n- TEXT: {text}\n- TAGS: {tags}\n"
        t_ids = tok.encode(blk_full, add_special_tokens=False, truncation=True, max_length=int(target_max))
        # Text-only token span length (prefix '- TEXT: ' + text + '\n') truncated the same way
        blk_text = f"[CANDIDATE]\n- TEXT: {text}\n"
        t_text_ids = tok.encode(blk_text, add_special_tokens=False, truncation=True, max_length=int(target_max))
        t_text_len = min(len(t_text_ids), len(t_ids))
        keep_p = max(0, int(input_max) - len(t_ids))
        if keep_p > 0:
            p_ids = tok.encode(conj_block, add_special_tokens=False, truncation=True, max_length=int(keep_p))
            if len(p_ids) > keep_p:
                p_ids = p_ids[-keep_p:]
        else:
            p_ids = []
        p_ids_list.append(p_ids)
        t_ids_list.append(t_ids)
        t_text_len_list.append(t_text_len)
    if not t_ids_list:
        return []
    # device
    try:
        dev = model.get_input_embeddings().weight.device
    except Exception:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Process in micro-batches to reduce memory footprint
    rows = [p + t for p, t in zip(p_ids_list, t_ids_list)]
    N = len(rows)
    pad_id = tok.pad_token_id or 0
    scores = torch.empty(N, dtype=torch.float32, device=dev)
    tok_counts = torch.zeros(N, dtype=torch.float32, device=dev)
    chunk = 8  # tune if still OOM
    for s in range(0, N, chunk):
        e = min(N, s + chunk)
        rs = rows[s:e]
        max_len = max(len(r) for r in rs)
        input_ids = torch.full((len(rs), max_len), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros_like(input_ids)
        for i, ids in enumerate(rs):
            L = len(ids)
            input_ids[i, :L] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[i, :L] = 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits  # [B, T, V]
            # shift for next-token prediction
            logits = logits[:, :-1, :]
            labels = input_ids[:, 1:]
            # compute log probs for labels without materializing full log_softmax tensor
            logZ = torch.logsumexp(logits, dim=-1)  # [B, T]
            label_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
            lp_labels = label_logits - logZ  # [B, T]
            for i in range(s, e):
                # window-relative index within chunk
                ci = i - s
                start = max(0, len(p_ids_list[i]) - 1)
                if segment == "text":
                    seg_len = t_text_len_list[i]
                else:
                    seg_len = len(t_ids_list[i])
                end = start + seg_len
                Lr = lp_labels.shape[1]
                start = min(max(0, start), Lr)
                end = min(max(start, end), Lr)
                if end <= start:
                    scores[i] = -1e9
                else:
                    # sum-LL or mean-LL
                    s_sum = lp_labels[ci, start:end].sum()
                    cnt = float(end - start)
                    tok_counts[i] = cnt
                    if score_type == "mean-ll" and cnt > 0:
                        scores[i] = s_sum / cnt
                    else:
                        scores[i] = s_sum
        # free per-chunk tensors
        del input_ids, attn, logits, labels, logZ, label_logits, lp_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return scores.detach().float().cpu().tolist()


@torch.no_grad()
def _direct_scores_for_many_windows(
    model: AutoModelForCausalLM,
    tok: AutoTokenizer,
    windows: List[tuple[str, List[tuple[int, str, str]]]],
    input_max: int,
    target_max: int,
    row_batch: int,
    score_type: str = "sum-ll",
    segment: str = "text",
) -> List[List[float]]:
    """Compute candidate scores for many windows in global batches of rows.
    Returns a list of per-window score lists, aligned with `windows`.
    """
    # Pre-encode all rows
    conj_blocks: List[List[int]] = []
    t_ids_list: List[List[int]] = []
    t_text_len_list: List[int] = []
    p_ids_list: List[List[int]] = []
    win_slices: List[tuple[int, int]] = []  # start,end over rows for each window
    start_idx = 0
    for conj, cands in windows:
        conj_block = f"[CONJECTURE]\n{conj}\n\n"
        # Encode conjecture once at a generous cap; we will slice per target length
        conj_ids_full = tok.encode(conj_block, add_special_tokens=False, truncation=True, max_length=int(input_max))
        row_count_before = len(t_ids_list)
        for (_, text, tags) in cands:
            blk_full = f"[CANDIDATE]\n- TEXT: {text}\n- TAGS: {tags}\n"
            t_ids = tok.encode(blk_full, add_special_tokens=False, truncation=True, max_length=int(target_max))
            blk_text = f"[CANDIDATE]\n- TEXT: {text}\n"
            t_text_ids = tok.encode(blk_text, add_special_tokens=False, truncation=True, max_length=int(target_max))
            t_text_len = min(len(t_text_ids), len(t_ids))
            keep_p = max(0, int(input_max) - len(t_ids))
            if keep_p > 0:
                # keep tail (left truncation) of conj_ids_full
                p_ids = conj_ids_full[-keep_p:]
            else:
                p_ids = []
            p_ids_list.append(p_ids)
            t_ids_list.append(t_ids)
            t_text_len_list.append(t_text_len)
        row_count_after = len(t_ids_list)
        conj_blocks.append(conj_ids_full)
        win_slices.append((start_idx, start_idx + (row_count_after - row_count_before)))
        start_idx += (row_count_after - row_count_before)

    if not t_ids_list:
        return [[] for _ in windows]

    # Resolve device
    try:
        dev = model.get_input_embeddings().weight.device
    except Exception:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Global batched scoring
    pad_id = tok.pad_token_id or 0
    N = len(t_ids_list)
    row_scores = torch.empty(N, dtype=torch.float32, device=dev)
    tok_counts = torch.zeros(N, dtype=torch.float32, device=dev)
    bsz = max(1, int(row_batch))
    for s in range(0, N, bsz):
        e = min(N, s + bsz)
        max_len = 0
        for i in range(s, e):
            max_len = max(max_len, len(p_ids_list[i]) + len(t_ids_list[i]))
        input_ids = torch.full((e - s, max_len), pad_id, dtype=torch.long, device=dev)
        attn = torch.zeros_like(input_ids)
        for i in range(s, e):
            ids = p_ids_list[i] + t_ids_list[i]
            L = len(ids)
            input_ids[i - s, :L] = torch.tensor(ids, dtype=torch.long, device=dev)
            attn[i - s, :L] = 1
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
            out = model(input_ids=input_ids, attention_mask=attn, use_cache=False)
            logits = out.logits  # [B, T, V]
            logits = logits[:, :-1, :]
            labels = input_ids[:, 1:]
            logZ = torch.logsumexp(logits, dim=-1)  # [B, T]
            label_logits = logits.gather(-1, labels.unsqueeze(-1)).squeeze(-1)  # [B, T]
            lp_labels = label_logits - logZ

            for i in range(s, e):
                ci = i - s
                start_pos = max(0, len(p_ids_list[i]) - 1)
                if segment == "text":
                    seg_len = t_text_len_list[i]
                else:
                    seg_len = len(t_ids_list[i])
                end_pos = start_pos + seg_len
                Lr = lp_labels.shape[1]
                start_pos = min(max(0, start_pos), Lr)
                end_pos = min(max(start_pos, end_pos), Lr)
                if end_pos <= start_pos:
                    row_scores[i] = -1e9
                else:
                    s_sum = lp_labels[ci, start_pos:end_pos].sum()
                    cnt = float(end_pos - start_pos)
                    tok_counts[i] = cnt
                    if score_type == "mean-ll" and cnt > 0:
                        row_scores[i] = s_sum / cnt
                    else:
                        row_scores[i] = s_sum
        del input_ids, attn, logits, labels, logZ, label_logits, lp_labels
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # Unflatten to per-window
    out_scores: List[List[float]] = []
    for (st, ed) in win_slices:
        if ed <= st:
            out_scores.append([])
        else:
            out_scores.append(row_scores[st:ed].detach().float().cpu().tolist())
    return out_scores


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Eval LLM listwise reranker on val JSONL: JSON rate/len match/MAE/MSE/Pearson")
    ap.add_argument("--data", type=Path, required=True, help="val_listwise.jsonl")
    ap.add_argument("--limit", type=int, default=200, help="max samples to evaluate (0=all)")
    ap.add_argument("--out", type=Path, default=None, help="If set, write per-line predictions to this JSONL file")
    ap.add_argument("--out-mode", type=str, choices=["pred", "replace-target"], default="pred", help="pred: add pred_scores; replace-target: overwrite target_scores/target_json with predictions")
    ap.add_argument("--input-max", type=int, default=2048, help="cap prompt tokens (left-truncated) to speed up prefill")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=64)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--endpoint", type=str, default=os.getenv("LLM_LOCAL_ENDPOINT", ""), help="POST endpoint /generate; default from env LLM_LOCAL_ENDPOINT")
    # in-process fast path (skip HTTP)
    ap.add_argument("--model", type=str, default=None, help="HF model id/path for in-process eval (overrides --endpoint if set)")
    ap.add_argument("--lora", type=str, default=None, help="Optional PEFT LoRA adapter path")
    ap.add_argument("--batch", type=int, default=4, help="in-process batch size")
    ap.add_argument("--bits", type=int, choices=[4, 8, 16], default=4, help="Quantization: 4/8-bit (bnb) or 16 (no quant)")
    ap.add_argument("--mode", type=str, choices=["json","direct"], default="json", help="json: parse model text; direct: compute scores via logits softmax")
    ap.add_argument("--target-max", type=int, default=256, help="per-candidate token cap in direct mode")
    ap.add_argument("--row-batch", type=int, default=64, help="global row batch size for direct mode (bigger = faster, needs more VRAM)")
    ap.add_argument("--attn", type=str, default="auto", choices=["auto","flash","eager"], help="attention impl: flash uses flash_attention_2 if supported")
    # scoring options
    ap.add_argument("--score-type", type=str, choices=["sum-ll", "mean-ll"], default="sum-ll", help="Use sum or mean of token log-likelihoods per candidate")
    ap.add_argument("--segment", type=str, choices=["full", "text"], default="text", help="Which part of candidate block to score: full block (TEXT+TAGS) or just TEXT line")
    ap.add_argument("--zscore", action="store_true", help="Apply per-window z-score to scores before softmax")
    ap.add_argument("--calib-tau", type=float, default=1.0, help="Temperature for softmax calibration in direct mode")
    ap.add_argument("--calib-bias", type=float, default=0.0, help="Bias shift for calibration before softmax")
    ap.add_argument("--pmi-lambda", type=float, default=0.0, help="If >0, subtract lambda*logP(c) by scoring candidate without conjecture (PMI)")
    # Sharding for multi-GPU parallel eval
    ap.add_argument("--shard", type=int, default=0, help="0-indexed shard id")
    ap.add_argument("--nshards", type=int, default=1, help="total number of shards")
    # Optional: RoPE scaling for longer context at eval time
    ap.add_argument("--rope-type", type=str, default="none", choices=["none", "linear", "dynamic", "yarn"], help="Apply RoPE scaling to extend context")
    ap.add_argument("--rope-factor", type=float, default=1.0, help="Scaling factor, e.g., 2.0 for ~8k if base is 4k")
    ap.add_argument("--rope-base", type=int, default=None, help="Original max_position_embeddings; if None, try from model config")
    args = ap.parse_args(argv)

    use_inprocess = bool(args.model)
    if not use_inprocess and not args.endpoint:
        raise SystemExit("Endpoint not set: pass --endpoint or export LLM_LOCAL_ENDPOINT, or use --model for in-process eval.")

    n_total = 0
    n_json_ok = 0
    n_len_match = 0
    maes: List[float] = []
    mses: List[float] = []
    pears: List[float] = []

    prompts: List[str] = []
    tgt_vecs: List[List[float]] = []
    orig_objs: List[dict] = []
    direct_windows: List[tuple[str, List[tuple[int,str,str]]]] = []
    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        for li, ln in enumerate(f, start=1):
            # dataset sharding
            if args.nshards > 1:
                if (li - 1) % max(1, args.nshards) != max(0, args.shard):
                    continue
            if args.limit and (n_total >= args.limit):
                break
            try:
                j = json.loads(ln)
            except Exception:
                continue
            prompt = j.get("input") or j.get("prompt")
            if not prompt:
                continue
            if args.mode == "json":
                prompt_full = str(prompt)
                prompts.append(prompt_full)
            else:
                conj, tuples = _parse_prompt_for_q_and_candidates(prompt)
                if not conj or not tuples:
                    continue
                direct_windows.append((conj, tuples))
            # Build target vector
            tgt_vec: List[float] = []
            if j.get("target_scores"):
                try:
                    tgt_vec = [float(x) for x in j["target_scores"]]
                except Exception:
                    tgt_vec = []
            elif j.get("target_json"):
                try:
                    tgt_obj = json.loads(j["target_json"]) if isinstance(j["target_json"], str) else j["target_json"]
                    tgt_vec = _flatten_scores(tgt_obj)
                except Exception:
                    tgt_vec = []
            tgt_vecs.append(tgt_vec)
            orig_objs.append(j)
            n_total += 1

    # Generate all predictions
    texts: List[str] = []
    if use_inprocess:
        # speed hints
        try:
            torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
            torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
        except Exception:
            pass
        # Prefer quantized loading to avoid OOM
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
        # Keep the tail (instruction and placeholder) by left-side truncation
        try:
            tok.truncation_side = "left"  # type: ignore[attr-defined]
        except Exception:
            pass
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        quant: Optional[BitsAndBytesConfig] = None
        torch_dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch.cuda.is_available() else None
        if args.bits == 4:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch_dtype,
            )
        elif args.bits == 8:
            quant = BitsAndBytesConfig(load_in_8bit=True)

        attn_impl = None
        if args.attn == "flash":
            attn_impl = "flash_attention_2"
        elif args.attn == "eager":
            attn_impl = "eager"

        if quant is not None:
            mdl = AutoModelForCausalLM.from_pretrained(
                args.model,
                quantization_config=quant,
                device_map=device_map,
                torch_dtype=torch_dtype,
                attn_implementation=attn_impl,
            )
        else:
            mdl = AutoModelForCausalLM.from_pretrained(
                args.model,
                torch_dtype=torch_dtype,
                device_map=device_map,
                attn_implementation=attn_impl,
            )
        # IMPORTANT: match training-time embedding size before loading LoRA
        try:
            target_vocab_size = int(len(tok))
            if getattr(mdl.config, "vocab_size", None) != target_vocab_size:
                mdl.resize_token_embeddings(target_vocab_size)
                try:
                    mdl.config.vocab_size = target_vocab_size
                except Exception:
                    pass
        except Exception as _e:
            print(f"[WARN] Could not resize token embeddings pre-PEFT: {_e}")
        # Apply optional RoPE scaling
        try:
            if args.rope_type != "none" and args.rope_factor and args.rope_factor > 1.0:
                cfg = mdl.config
                orig = int(args.rope_base) if args.rope_base else int(getattr(cfg, "max_position_embeddings", 4096))
                cfg.rope_scaling = {  # type: ignore[attr-defined]
                    "type": args.rope_type,
                    "factor": float(args.rope_factor),
                    "original_max_position_embeddings": int(orig),
                }
                if hasattr(cfg, "max_position_embeddings"):
                    try:
                        cfg.max_position_embeddings = int(orig * args.rope_factor)
                    except Exception:
                        pass
                print(f"[INFO] RoPE scaling enabled at eval: {cfg.rope_scaling}")
        except Exception as e:
            print(f"[WARN] Failed to apply RoPE scaling at eval: {e}")
        if args.lora:
            # Be flexible: accept a parent dir that contains a `lora/` subdir with adapter files
            lora_path = args.lora
            try:
                p = Path(lora_path)
                if p.is_dir():
                    cfg_here = p / "adapter_config.json"
                    if not cfg_here.exists():
                        cfg_sub = p / "lora" / "adapter_config.json"
                        if cfg_sub.exists():
                            lora_path = str(p / "lora")
            except Exception:
                pass
            from peft import PeftModel  # type: ignore
            mdl = PeftModel.from_pretrained(mdl, lora_path)
        mdl.eval()
        # If using device_map auto, avoid forcing .to(device); pass device=None to generator
        device_for_inputs: Optional[torch.device] = None
        if device_map is None:
            device_for_inputs = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if args.mode == "json":
            texts = _inprocess_generate(
                mdl,
                tok,
                prompts,
                device_for_inputs,
                input_max_len=args.input_max,
                max_new_tokens=args.max_new,
                temperature=args.temperature,
                batch_size=args.batch,
            )
        else:
            pred_vecs: List[List[float]] = []
            # Global-batched scoring for better GPU utilization
            raw_scores = _direct_scores_for_many_windows(
                mdl, tok, direct_windows, args.input_max, args.target_max, args.row_batch,
                score_type=args.score_type, segment=args.segment,
            )
            # Optional PMI baseline: score candidates without conjecture
            if args.pmi_lambda and args.pmi_lambda > 0.0:
                null_windows: List[tuple[str, List[tuple[int, str, str]]]] = [("", ws[1]) for ws in direct_windows]
                base_scores = _direct_scores_for_many_windows(
                    mdl, tok, null_windows, 0, args.target_max, args.row_batch,
                    score_type=args.score_type, segment=args.segment,
                )
            else:
                base_scores = [[] for _ in raw_scores]
            for xs in raw_scores:
                pred_vecs.append([])
            # Calibrate per-window
            for wi, xs in enumerate(raw_scores):
                if not xs:
                    continue
                v = np.asarray(xs, dtype=np.float64)
                if args.pmi_lambda and args.pmi_lambda > 0.0:
                    bs = np.asarray(base_scores[wi], dtype=np.float64)
                    if bs.shape == v.shape:
                        v = v - float(args.pmi_lambda) * bs
                # Optional z-score normalization
                if args.zscore:
                    mu = float(v.mean())
                    sd = float(v.std())
                    if sd > 1e-8:
                        v = (v - mu) / sd
                    else:
                        v = v - mu
                # Calibration then softmax to produce a distribution
                v = (v - float(args.calib_bias)) / max(1e-8, float(args.calib_tau))
                vmax = float(v.max())
                v = v - vmax
                p = np.exp(v)
                s = float(p.sum())
                if s <= 0:
                    p = np.ones_like(p) / max(1, len(p))
                else:
                    p = p / s
                pred_vecs[wi] = p.tolist()
    else:
        if args.mode == "json":
            for prompt_full in prompts:
                try:
                    r = requests.post(
                        args.endpoint,
                        json={"prompt": prompt_full, "temperature": args.temperature, "max_new_tokens": args.max_new},
                        timeout=args.timeout,
                    )
                    r.raise_for_status()
                    obj = r.json()
                    texts.append(obj.get("text") or obj.get("output") or obj.get("generated_text") or "")
                except Exception:
                    texts.append("")
        else:
            raise SystemExit("Direct mode requires --model (in-process)")

    # Evaluate
    if args.mode == "json":
        for text, tgt_vec in zip(texts, tgt_vecs):
            parsed = extract_json(text)
            pred_vec = _flatten_scores(parsed)
            json_ok = isinstance(parsed, dict) and ("scores" in parsed)
            if json_ok:
                n_json_ok += 1
            len_match = bool(tgt_vec) and (len(pred_vec) == len(tgt_vec))
            if len_match:
                n_len_match += 1

            k = min(len(pred_vec), len(tgt_vec))
            if k >= 1:
                a = np.asarray(pred_vec[:k], dtype=np.float64)
                b = np.asarray(tgt_vec[:k], dtype=np.float64)
                mae_v = float(np.mean(np.abs(a - b)))
                mse_v = float(np.mean((a - b) ** 2))
                if not np.isnan(mae_v):
                    maes.append(mae_v)
                if not np.isnan(mse_v):
                    mses.append(mse_v)
                pr = _pearsonr(pred_vec[:k], tgt_vec[:k])
                if not np.isnan(pr):
                    pears.append(float(pr))
        # Optional write-out for json mode
        if args.out is not None:
            with open(args.out, "w", encoding="utf-8") as w:
                for j, text in zip(orig_objs, texts):
                    parsed = extract_json(text)
                    preds = _flatten_scores(parsed)
                    if args.out_mode == "replace-target":
                        j["target_scores"] = preds
                        j["target_json"] = json.dumps({"scores": preds}, ensure_ascii=False)
                    else:
                        j["pred_scores"] = preds
                        j["pred_json"] = json.dumps({"scores": preds}, ensure_ascii=False)
                    w.write(json.dumps(j, ensure_ascii=False) + "\n")
    else:
        # direct mode
        for pred_vec, tgt_vec in zip(pred_vecs, tgt_vecs):
            len_match = bool(tgt_vec) and (len(pred_vec) == len(tgt_vec))
            if len_match:
                n_len_match += 1
            k = min(len(pred_vec), len(tgt_vec))
            if k >= 1:
                a = np.asarray(pred_vec[:k], dtype=np.float64)
                b = np.asarray(tgt_vec[:k], dtype=np.float64)
                mae_v = float(np.mean(np.abs(a - b)))
                mse_v = float(np.mean((a - b) ** 2))
                if not np.isnan(mae_v):
                    maes.append(mae_v)
                if not np.isnan(mse_v):
                    mses.append(mse_v)
                pr = _pearsonr(pred_vec[:k], tgt_vec[:k])
                if not np.isnan(pr):
                    pears.append(float(pr))
        # Optional write-out for direct mode
        if args.out is not None:
            with open(args.out, "w", encoding="utf-8") as w:
                for j, preds in zip(orig_objs, pred_vecs):
                    if args.out_mode == "replace-target":
                        j["target_scores"] = preds
                        j["target_json"] = json.dumps({"scores": preds}, ensure_ascii=False)
                    else:
                        j["pred_scores"] = preds
                        j["pred_json"] = json.dumps({"scores": preds}, ensure_ascii=False)
                    w.write(json.dumps(j, ensure_ascii=False) + "\n")

    if n_total == 0:
        raise SystemExit("No samples evaluated.")

    # Prepare aggregate sums for easy merge across shards
    sum_mae = float(sum(maes)) if maes else 0.0
    sum_mse = float(sum(mses)) if mses else 0.0
    pear_sum = float(sum(pears)) if pears else 0.0
    pear_cnt = int(len(pears))
    out = {
        "n": n_total,
        "json_ok_rate": (n_json_ok / n_total) if args.mode == "json" else None,
        "len_match_rate": n_len_match / n_total,
        "mae": (sum(maes) / len(maes)) if maes else None,
        "mse": (sum(mses) / len(mses)) if mses else None,
        "pearson": (sum(pears) / len(pears)) if pears else None,
        "mode": args.mode,
        "score_type": args.score_type if args.mode == "direct" else None,
        "segment": args.segment if args.mode == "direct" else None,
        "zscore": bool(args.zscore) if args.mode == "direct" else None,
        "calibration": {"tau": args.calib_tau, "bias": args.calib_bias} if args.mode == "direct" else None,
        "pmi_lambda": args.pmi_lambda if args.mode == "direct" else None,
        "_sums": {
            "sum_mae": sum_mae,
            "sum_mse": sum_mse,
            "pear_sum": pear_sum,
            "pear_cnt": pear_cnt,
            "json_ok_sum": int(n_json_ok),
            "len_match_sum": int(n_len_match)
        }
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
