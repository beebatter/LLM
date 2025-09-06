#!/usr/bin/env python3
"""
Build listwise windows for LLM reranking fine-tuning.

Input: one or more JSONL files with clause-level samples. Expected fields per line
(best-effort, multiple aliases supported):

- problem_name: str (grouping key)
- conjecture_text|conjecture|query|text_a: str
- text|clause|text_b|premise: str
- features|meta: dict (optional)
- label: int/bool (1 positive, 0 negative)
- clause_id|id: int (optional; will be auto-assigned if missing)
- neg_bucket: str optional for weighted sampling

Output: JSONL with one item per listwise window, including:
- problem_name, ids (length K), input (prompt text),
- target_json: the expected JSON string {"scores": [s1..sK]} using a soft target distribution,
- target_scores: the numeric list for convenience (KL training later)

This script aligns with the whitepaper §4 (LLM Listwise Rerank) and batch_ranker.py's prompt schema.
"""
from __future__ import annotations

import argparse
import json
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as _np
try:
    from bitsandbytes import BitsAndBytesConfig
except Exception:
    BitsAndBytesConfig = None

from LLM.data_utils.logic_tokenizer import normalize_text, features_to_prefix, PrefixBuckets


def _first(d: Dict, keys: List[str]):
    for k in keys:
        if k in d and d[k] is not None:
            return d[k]
    return None


def _as_int(x, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _build_candidate_lines(ids: List[int], texts: List[str], tags: List[str]) -> str:
    lines = []
    for i, (cid, t, tg) in enumerate(zip(ids, texts, tags), start=1):
        lines.append(f"- ID {cid}\n- TEXT: {t}\n- TAGS: {tg}")
    return "\n".join(lines)


def _prompt_for_window(conj: str, ids: List[int], texts: List[str], tags: List[str], require_sum1: bool = False) -> str:
    K = len(ids)
    if require_sum1:
        header = (
            "你是自动定理证明的子句打分器。给定一个猜想 Q 和 K 个候选子句，"
            "请仅输出 JSON：{\"scores\":[...]}，长度为 K，且分数为 softmax 概率（总和 = 1）。\n"
        )
    else:
        header = (
            "你是自动定理证明的子句打分器。给定一个猜想 Q 和 K 个候选子句，"
            "请仅输出 JSON：{\"scores\":[...]}，长度为 K，数字用小数，且和可不必等于 1（下游会 softmax）。\n"
        )
    conj_block = f"[CONJECTURE]\n{conj}\n"
    spec = (
        f"\n[CANDIDATES]  共 {K} 条。每条格式：\n"
        "- ID <整型唯一ID>\n- TEXT: <子句文本>\n- TAGS: <紧凑特征token>\n\n"
        "（以下为候选）\n"
    )
    body = _build_candidate_lines(ids, texts, tags)
    if require_sum1:
        tail = "\n\n请仅输出（确保 scores 总和为 1）：\n{\"scores\":[s1, s2, ..., sK]}\n"
    else:
        tail = "\n\n请仅输出：\n{\"scores\":[s1, s2, ..., sK]}\n"
    return header + conj_block + spec + body + tail


def _soft_targets(pos_mask: List[int], tau: float = 1.0) -> List[float]:
    # Uniform mass on positives; if none positive, uniform over all.
    K = len(pos_mask)
    n_pos = sum(1 for x in pos_mask if x)
    if n_pos <= 0:
        return [1.0 / K] * K
    base = [1.0 / n_pos if m else 0.0 for m in pos_mask]
    if abs(tau - 1.0) < 1e-9:
        return base
    # Optional sharpening/softening
    import numpy as np
    x = np.array(base, dtype=float)
    x = np.power(x + 1e-12, 1.0 / tau)
    x = x / x.sum()
    return x.tolist()


def _neg_weight(bucket: Optional[str]) -> float:
    if not bucket:
        return 1.0
    b = str(bucket)
    if b == "NEG_given_nonproof":
        return 1.0
    if b == "NEG_simplified":
        return 0.8
    if b == "NEG_passive_only":
        return 0.5
    # failed/others
    return 0.25


def _build_windows_for_problem(
    items: List[Dict],
    conj: str,
    K: int,
    max_windows: Optional[int],
    seed: int,
) -> List[Dict]:
    rng = random.Random(seed)
    pos = [x for x in items if int(_first(x, ["label"]) or 0) == 1]
    neg = [x for x in items if int(_first(x, ["label"]) or 0) == 0]
    if not items:
        return []
    if not pos:
        # ensure at least one window with all negatives (will be uniform targets)
        neg_pool = rng.sample(neg, min(len(neg), K)) if neg else rng.sample(items, min(len(items), K))
        # Only keep exact-K windows
        return [neg_pool] if len(neg_pool) == K else []

    # Weight negatives by bucket for harder sampling
    weights = [(_neg_weight(_first(x, ["neg_bucket"])) if x in neg else 0.0) for x in items]
    neg_weights = [w for x, w in zip(items, weights) if int(_first(x, ["label"]) or 0) == 0]
    neg_only = neg

    windows = []
    # Strategy: for each positive, make one window seeded by that positive, fill with hard negatives
    for p in pos:
        w = [p]
        need = K - 1
        if neg_only:
            # weighted sampling without replacement (approximate)
            pool = list(neg_only)
            pool_w = list(neg_weights)
            sel: List[Dict] = []
            for _ in range(min(need, len(pool))):
                s = sum(pool_w) or 1.0
                r = rng.random() * s
                acc = 0.0
                idx = 0
                for i, ww in enumerate(pool_w):
                    acc += ww
                    if acc >= r:
                        idx = i
                        break
                sel.append(pool.pop(idx))
                pool_w.pop(idx)
            w.extend(sel)
        # fill if still short
        if len(w) < K:
            more = [x for x in items if x not in w]
            rng.shuffle(more)
            w.extend(more[: K - len(w)])
        windows.append(w[:K])
        if max_windows and len(windows) >= max_windows:
            break
    return windows


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Make listwise windows for LLM reranker training")
    ap.add_argument("--input", action="append", required=True, help="JSONL path(s) with clause-level samples")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path (listwise)")
    ap.add_argument("--window", type=int, default=64, help="Window size K")
    ap.add_argument("--max-windows-per-problem", type=int, default=4, help="Cap windows per problem")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-text-chars", type=int, default=256, help="Truncate TEXT to this many characters (per candidate)")
    ap.add_argument("--llm-model", type=str, default=None, help="Path or HF id of causal LLM to use for scoring (optional)")
    ap.add_argument("--llm-softmax", action="store_true", help="If set and --llm-model provided, compute real model log-probs for each candidate and replace target_scores with the softmax of those scores")
    ap.add_argument("--llm-batch-size", type=int, default=8, help="Batch size when scoring candidates within a window")
    ap.add_argument("--llm-4bit", action="store_true", help="If set, attempt to load the LLM in 4-bit via bitsandbytes (load_in_4bit=True)")
    ap.add_argument("--llm-device-map", type=str, default='auto', help="device_map to pass when loading large models (default 'auto')")
    ap.add_argument("--llm-max-memory", type=str, default=None, help="Max memory mapping for dispatch (JSON string or 'auto' to infer from nvidia-smi)")
    ap.add_argument("--llm-low-cpu-mem-usage", action='store_true', help="If set, pass low_cpu_mem_usage=True to from_pretrained when available")
    ap.add_argument("--prompt-softmax", action="store_true", help="If set, ask the model to output softmax-normalized scores (sum=1)")
    ap.add_argument("--llm-temp", type=float, default=1.0, help="Temperature applied to scores before softmax (default 1.0)")
    ap.add_argument("--llm-length-norm", type=str, default='sum', choices=['sum', 'mean'], help="Whether to sum token log-probs ('sum') or average per-token ('mean') before softmax")
    ap.add_argument("--llm-exclude-eos", action='store_true', help="If set, exclude final EOS token from per-candidate log-prob sums")
    ap.add_argument("--llm-max-tokens", type=int, default=None, help="If set, truncate per-candidate token span to this many tokens before summing/averaging")
    args = ap.parse_args(argv)

    # group by problem
    by_prob: Dict[str, List[Dict]] = defaultdict(list)
    conj_map: Dict[str, str] = {}
    next_cid = 1
    for p in args.input:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                prob = _first(j, ["problem_name", "problem", "name"]) or "_"
                conj = _first(j, ["conjecture_text", "conjecture", "query", "text_a"]) or ""
                if conj and prob not in conj_map:
                    conj_map[prob] = normalize_text(conj)
                # normalize fields for convenience
                raw_text = _first(j, ["text", "clause", "text_b", "premise", "doc"]) or ""
                norm_text = normalize_text(raw_text)
                if not norm_text:
                    # drop empty TEXT to avoid blank candidates
                    continue
                row = {
                    "id": _as_int(_first(j, ["clause_id", "id"]) or next_cid),
                    "text": norm_text,
                    "features": _first(j, ["features", "meta"]) or {},
                    "label": _as_int(_first(j, ["label"]) or 0),
                    "neg_bucket": _first(j, ["neg_bucket"]) or None,
                }
                if _first(j, ["clause_id", "id"]) is None:
                    next_cid += 1
                by_prob[prob].append(row)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_windows = 0
    rng = random.Random(args.seed)
    # If requested, load LLM tokenizer+model for direct scoring
    llm_tokenizer = None
    llm_model = None
    device = None
    ap_llm_opts = getattr(args, 'llm_opts', None)
    if args.llm_model:
        try:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            llm_tokenizer = AutoTokenizer.from_pretrained(args.llm_model, use_fast=False)
            # support optional 4-bit loading for very large models
            llm_loaded_in_4bit = False
            if getattr(args, 'llm_4bit', False):
                try:
                    import bitsandbytes as bnb  # ensure bitsandbytes is available
                    bnb_ok = True
                except Exception:
                    bnb_ok = False
                if bnb_ok:
                    try:
                        # Build BitsAndBytesConfig if available
                        if BitsAndBytesConfig is None:
                            raise RuntimeError('BitsAndBytesConfig not available in bitsandbytes; please install compatible bitsandbytes')
                        bnb_config = BitsAndBytesConfig(
                            load_in_4bit=True,
                            bnb_4bit_use_double_quant=True,
                            bnb_4bit_compute_dtype=torch.float16,
                        )
                        load_kwargs = dict(
                            quantization_config=bnb_config,
                            device_map=args.llm_device_map,
                            trust_remote_code=True,
                        )
                        if getattr(args, 'llm_low_cpu_mem_usage', False):
                            load_kwargs['low_cpu_mem_usage'] = True

                        # prepare max_memory mapping if requested; transformers prefers integer GPU indices when using device_map='auto'
                        max_mem = None
                        if getattr(args, 'llm_max_memory', None):
                            mm = args.llm_max_memory
                            if mm.strip().lower() == 'auto':
                                try:
                                    import subprocess
                                    out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'], encoding='utf-8')
                                    vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
                                    max_mem = {}
                                    for i, v in enumerate(vals):
                                        cap_mb = int(v * 0.8)
                                        # use integer keys as required by newer transformers for GPU mapping
                                        max_mem[i] = f"{cap_mb}MB"
                                except Exception as e:
                                    print(f"Warning: failed to auto-compute max_memory via nvidia-smi: {e}")
                                    max_mem = None
                            else:
                                try:
                                    max_mem = json.loads(mm)
                                except Exception:
                                    max_mem = None
                        if max_mem is not None:
                            load_kwargs['max_memory'] = max_mem
                        llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model, **load_kwargs)
                        llm_loaded_in_4bit = True
                    except Exception as e:
                        print(f"Warning: 4-bit load with bitsandbytes failed, falling back to standard load: {e}")
                        llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model, trust_remote_code=True)
                else:
                    print("Warning: bitsandbytes not installed; cannot load in 4-bit. Falling back to standard load.")
                    llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model, trust_remote_code=True)
            else:
                llm_model = AutoModelForCausalLM.from_pretrained(args.llm_model, torch_dtype=torch.float16 if device == 'cuda' else torch.float32, trust_remote_code=True)
            # If model was not automatically moved (e.g., non-4bit CPU/GPU path), move it
            try:
                # move CPU fallback models to CUDA when possible. If we actually loaded in 4-bit
                # (bnb device_map dispatch), avoid forcing a .to('cuda') which can be wrong.
                if device == 'cuda' and not llm_loaded_in_4bit:
                    llm_model.to(device)
            except Exception:
                pass
            llm_model.eval()
            # determine a sensible primary device for the model's parameters
            def _model_primary_device(m):
                try:
                    for p in m.parameters():
                        if getattr(p, 'device', None) is not None and p.device.type != 'meta':
                            return p.device
                except Exception:
                    pass
                return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model_primary_device = _model_primary_device(llm_model)
            print(f"[info] llm_model primary device: {model_primary_device}")
            # If model is on CPU but CUDA is available and this isn't a 4-bit dispatched model,
            # attempt to move the model to CUDA to speed up scoring and avoid device mismatch.
            try:
                if str(model_primary_device).startswith('cpu') and torch.cuda.is_available() and not getattr(args, 'llm_4bit', False):
                    try:
                        llm_model.to('cuda')
                        model_primary_device = _model_primary_device(llm_model)
                        print(f"[info] moved llm_model to device: {model_primary_device}")
                    except Exception:
                        pass
            except Exception:
                pass
        except Exception as e:
            print(f"Warning: failed to load LLM model {args.llm_model}: {e}")
            llm_tokenizer = None
            llm_model = None
            device = None

    def _score_window_with_llm(prompt_conj: str, cand_texts: List[str], batch_size: Optional[int] = None) -> Tuple[List[float], List[bool], List[float]]:
        """Batch-score K candidates for a single window.
        Returns (scores_list, truncation_flags, entropies).
        Scores are either summed log-probs or mean per-token log-probs depending on --llm-length-norm.
        If llm_model/tokenizer not loaded, returns ([], [], []).
        """
        if llm_model is None or llm_tokenizer is None:
            return [], [], []
        if batch_size is None:
            batch_size = len(cand_texts)
        scores: List[float] = []
        trunc_flags: List[bool] = []
        entropies: List[float] = []
        base_prompt = f"[CONJECTURE]\n{prompt_conj}\n[CANDIDATE]\n"
        # Precompute base length
        base_enc = llm_tokenizer(base_prompt, return_tensors='pt', truncation=True, max_length=llm_tokenizer.model_max_length)
        base_len = base_enc['input_ids'].size(1)
        # prefer actual model device to avoid CPU/CUDA mismatches when using device_map
        try:
            device_local = model_primary_device
        except Exception:
            device_local = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')

        pad_id = llm_tokenizer.pad_token_id or llm_tokenizer.eos_token_id or 0
        # diagnostics counters
        empty_span_counter = 0
        trunc_counter = 0

        for i in range(0, len(cand_texts), batch_size):
            batch = cand_texts[i : i + batch_size]
            inputs = [base_prompt + t for t in batch]
            try:
                enc = llm_tokenizer(inputs, return_tensors='pt', padding=True, truncation=True, max_length=llm_tokenizer.model_max_length, add_special_tokens=True)
                input_ids = enc['input_ids']
                attention_mask = enc.get('attention_mask')
                # move tensors to the model's device to avoid device mismatch
                try:
                    input_ids = input_ids.to(device_local)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device_local)
                except Exception:
                    input_ids = input_ids.cuda() if torch.cuda.is_available() else input_ids
                    if attention_mask is not None:
                        attention_mask = attention_mask.cuda() if torch.cuda.is_available() else attention_mask
                with torch.no_grad():
                    out = llm_model(input_ids=input_ids, attention_mask=attention_mask)
                    logits = out.logits  # (B, L, V)
                    logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
                    B, L, V = logprobs.shape
                    for bi in range(B):
                        ids = input_ids[bi]
                        # compute real length for this example (exclude padding)
                        if attention_mask is not None:
                            real_len = int(attention_mask[bi].sum().item())
                        else:
                            real_len = L
                        # default: mark not truncated
                        truncated = False
                        if base_len >= real_len:
                            # nothing left for candidate text (truncated or too long base)
                            scores.append(float('-inf'))
                            trunc_flags.append(True)
                            entropies.append(float('nan'))
                            empty_span_counter += 1
                            continue
                        # token span to consider
                        start = base_len
                        end = min(real_len, L)
                        token_ids = ids[start:end]
                        # optionally drop final EOS token
                        if args.llm_exclude_eos and token_ids.numel() > 0:
                            eos_mask = token_ids == llm_tokenizer.eos_token_id if llm_tokenizer.eos_token_id is not None else torch.zeros_like(token_ids, dtype=torch.bool)
                            if eos_mask.any():
                                if eos_mask[-1]:
                                    token_ids = token_ids[:-1]
                                    end -= 1
                        if token_ids.numel() == 0:
                            scores.append(float('-inf'))
                            trunc_flags.append(True)
                            entropies.append(float('nan'))
                            empty_span_counter += 1
                            continue
                        # mark truncation if candidate tokens were truncated by tokenizer
                        if len(token_ids) < (real_len - start):
                            truncated = True
                            trunc_counter += 1
                        gather_positions = torch.arange(start, end, device=device_local) - 1
                        lp = logprobs[bi, gather_positions, token_ids]
                        # length normalization option
                        if args.llm_length_norm == 'mean':
                            total_lp = float(lp.mean().item())
                        else:
                            total_lp = float(lp.sum().item())
                        scores.append(total_lp)
                        trunc_flags.append(truncated)
                        # compute entropy of next-token distribution averaged over positions for diagnostics
                        probs = torch.exp(logprobs[bi, gather_positions])  # (T, V)
                        ent = - (probs * torch.log(probs + 1e-12)).sum(dim=-1).mean().item()
                        entropies.append(ent)
            except Exception as e:
                # include device debug info to diagnose mismatches
                try:
                    mdev = model_primary_device
                except Exception:
                    mdev = None
                print(f"LLM batch scoring error: {e} -- model_device={mdev} input_device_check: input_ids_device={getattr(locals().get('input_ids',None),'device',None)}")
                for _ in batch:
                    scores.append(float('-inf'))
                    trunc_flags.append(False)
                    entropies.append(float('nan'))

        # report diagnostics
        if empty_span_counter:
            print(f"[warn] empty candidate token spans encountered: {empty_span_counter}")
        if trunc_counter:
            print(f"[info] truncated candidate token spans: {trunc_counter}")

        # apply temperature scaling when returning; caller will do softmax
        if args.llm_temp and args.llm_temp != 1.0:
            scores = [s / float(args.llm_temp) if _np.isfinite(s) else s for s in scores]
        return scores, trunc_flags, entropies

    with open(out_path, "w", encoding="utf-8") as wf:
        for prob, items in by_prob.items():
            conj = conj_map.get(prob, "")
            if not items or not conj:
                continue
            windows = _build_windows_for_problem(items, conj, args.window, args.max_windows_per_problem, args.seed)
            for w in windows:
                # Enforce exact-K windows only
                if len(w) != args.window:
                    continue
                # Shuffle order to remove position bias
                rng.shuffle(w)
                # Build fields and truncate TEXT if needed
                ids = [int(x["id"]) for x in w]
                tags = [features_to_prefix(x.get("features") or {}, PrefixBuckets()).strip() for x in w]
                texts = [ (x.get("text") or "")[: args.max_text_chars] for x in w ]
                pos_mask = [1 if int(x.get("label") or 0) == 1 else 0 for x in w]
                tgt = _soft_targets(pos_mask, tau=1.0)
                prompt = _prompt_for_window(conj, ids, texts, tags, require_sum1=bool(args.prompt_softmax))
                cands = [
                    {
                        "id": int(x.get("id")),
                        "text": (x.get("text") or "")[: args.max_text_chars],
                        "tags": features_to_prefix(x.get("features") or {}, PrefixBuckets()).strip(),
                        "label": int(x.get("label") or 0),
                    }
                    for x in w
                ]
                obj = {
                    "problem_name": prob,
                    "K": len(ids),
                    "ids": ids,
                    "conjecture": conj,
                    "candidates": cands,
                    "input": prompt,
                    "target_json": json.dumps({"scores": tgt}, ensure_ascii=False),
                    "target_scores": tgt,
                }
                # If requested, compute real model softmax scores and replace target_scores
                if args.llm_model and args.llm_softmax and llm_model is not None:
                    try:
                        cand_texts = [c['text'] for c in cands]
                        lls, trunc_flags, ents = _score_window_with_llm(conj, cand_texts, batch_size=args.llm_batch_size)
                        # numeric guard: replace -inf/NaN with large negative
                        if not lls:
                            # no scores computed; keep existing target
                            pass
                        else:
                            a = _np.array([(-1e9 if (not _np.isfinite(x)) else x) for x in lls], dtype=float)
                            # numeric stable softmax
                            if a.size == 0 or not _np.isfinite(a).any():
                                # fallback to previous target
                                pass
                            else:
                                a_max = a.max()
                                ex = _np.exp(a - a_max)
                                if ex.sum() == 0 or not _np.isfinite(ex.sum()):
                                    pass
                                else:
                                    soft = (ex / ex.sum()).tolist()
                                    obj['target_scores'] = soft
                                    obj['target_json'] = json.dumps({"scores": soft}, ensure_ascii=False)
                                    # diagnostics
                                    obj['llm_softmax_entropy'] = float(_np.nanmean([e for e in ents if _np.isfinite(e)]) if any(_np.isfinite(ents)) else float('nan'))
                                    obj['llm_truncated_any'] = any(trunc_flags)
                                    obj['llm_raw_scores'] = [ (None if (not _np.isfinite(x)) else float(x)) for x in lls ]
                    except Exception as e:
                        print(f"Warning: failed to compute llm softmax for problem {prob}: {e}")
                wf.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_windows += 1
    print(f"Wrote {n_windows} listwise windows to {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
