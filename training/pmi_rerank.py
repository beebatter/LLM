#!/usr/bin/env python3
"""PMI 重排脚本

功能：批量计算 avg_logP(c|q) 及 avg_logP(c)，做 PMI 校准后组内 softmax。
特点：
  - 不改模型结构；仅依赖 HF Transformers，可直接用于 DeepSeek / Goedel / Qwen 等因果 LLM。
  - 支持『攒到 N 个候选再一次前向』(--batch-size) 以提高吞吐。
  - 输入 JSONL：每行可为 {"conjecture": "...", "candidates": ["...","..."]}
              或 candidates 为对象列表：{"candidates":[{"text":"..."}, ...]}
  - 输出 JSONL：为每行追加 {"pmi_scores":[...], "probs":[...], "K":K}

示例：
python -m LLM.training.pmi_rerank \
  --model /root/autodl-tmp/models/DeepSeek-Prover-V2-7B \
  --in /root/autodl-tmp/Training/datasets/sample_groups.jsonl \
  --out /root/autodl-tmp/Training/datasets/sample_groups.pmi.jsonl \
  --batch-size 32 --lambda-pmi 0.7 --tau 1.0 --q-max 384 --c-max 96 --bf16

作为库函数：from LLM.training.pmi_rerank import PMIConfig, compute_pmi_for_group
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Union
from contextlib import nullcontext

# Reuse existing prompt parser (for lines that only have a combined prompt string)
try:  # optional import; if structure not present we fall back silently
    from LLM.scripts.make_listwise_targets_from_teacher import _parse_prompt_for_q_and_candidates as _parse_prompt  # type: ignore
except Exception:  # pragma: no cover
    _parse_prompt = None  # type: ignore

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore

__all__ = [
    "PMIConfig",
    "compute_pmi_for_group",
    "avg_logp_batch",
]


# ------------------------- 平均对数似然（仅 suffix 记分） -------------------------

def avg_logp_batch(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prefixes: List[str],
    suffixes: List[str],
    device: torch.device,
    max_len: int,
    autocast_dtype: torch.dtype | None = None,
) -> Tuple[List[float], List[int]]:
    """计算一批样本的 suffix 平均 logP：avg_logP(suffix | prefix+suffix)。
    仅 suffix token 计入（通过 labels = -100 屏蔽 prefix）。
    返回： (avg_logp_list, effective_token_count_list)
    """
    assert len(prefixes) == len(suffixes)
    B = len(prefixes)
    if B == 0:
        return [], []

    enc_p = tokenizer(prefixes, add_special_tokens=False)
    enc_s = tokenizer(suffixes, add_special_tokens=False)

    pad_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    input_ids: List[List[int]] = []
    attn_masks: List[List[int]] = []
    label_ids: List[List[int]] = []
    tok_counts: List[int] = []

    for i in range(B):
        p_ids = enc_p["input_ids"][i]
        s_ids = enc_s["input_ids"][i]
        merged = (p_ids + s_ids)[-max_len:]  # 右侧截断：保留结尾（保证 suffix 保留优先级）
        # 计算被截掉多少 prefix token
        cut = max(0, len(p_ids) + len(s_ids) - len(merged))
        # suffix 在 merged 中的起点：原 prefix 长度 - 被截掉的 token
        suffix_start = max(0, len(p_ids) - cut)
        labels = [-100] * suffix_start + merged[suffix_start:]
        labels = labels[: len(merged)]
        eff = sum(1 for x in labels if x != -100)
        tok_counts.append(eff)
        input_ids.append(merged)
        attn_masks.append([1] * len(merged))
        label_ids.append(labels)

    maxT = max(len(x) for x in input_ids)
    input_ids = [x + [pad_id] * (maxT - len(x)) for x in input_ids]
    attn_masks = [x + [0] * (maxT - len(x)) for x in attn_masks]
    label_ids = [x + [-100] * (maxT - len(x)) for x in label_ids]

    input_ids_t = torch.tensor(input_ids, device=device)
    attn_t = torch.tensor(attn_masks, device=device)
    labels_t = torch.tensor(label_ids, device=device)

    with torch.no_grad():
        ctx = (torch.autocast(device_type="cuda", dtype=autocast_dtype) if (torch.cuda.is_available() and autocast_dtype is not None) else nullcontext())
        with ctx:
            out = model(input_ids=input_ids_t, attention_mask=attn_t, use_cache=False)
            logits = out.logits  # [B,T,V]
            logprobs = torch.log_softmax(logits, dim=-1)
            gather_labels = labels_t.clone()
            gather_labels[gather_labels < 0] = 0
            token_ll = logprobs.gather(-1, gather_labels.unsqueeze(-1)).squeeze(-1)
            mask = (labels_t != -100).float()
            sum_ll = (token_ll * mask).sum(dim=1)
            counts = mask.sum(dim=1).clamp(min=1.0)
            avg = (sum_ll / counts).tolist()
            counts_list = counts.long().tolist()
    return avg, counts_list


# ------------------------- PMI 主逻辑 -------------------------

@dataclass
class PMIConfig:
    lambda_pmi: float = 0.7  # λ (0.5~1.0 常用区间)
    tau: float = 1.0         # 温度 softmax
    q_max: int = 384         # 猜想字符裁剪
    c_max: int = 96          # 候选字符裁剪
    max_len: int = 2048      # token 级最大长度 (右裁剪)
    batch_size: int = 32     # 批处理候选个数
    bf16: bool = False       # autocast bf16
    cache_uncond: bool = False  # 是否缓存 avg_logP(c)


def _trim(s: str, m: int) -> str:
    return s if (m is None or m <= 0 or len(s) <= m) else s[:m]


def _cand_texts(cands: List[Union[str, Dict[str, Any]]]) -> List[str]:
    out: List[str] = []
    for c in cands:
        if isinstance(c, str):
            out.append(c)
        elif isinstance(c, dict):
            out.append(str(c.get("text", "")))
        else:
            out.append(str(c))
    return out


def compute_pmi_for_group(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    conjecture: str,
    candidates: List[Union[str, Dict[str, Any]]],
    cfg: PMIConfig,
    device: torch.device,
    _uncond_cache: Dict[str, float] | None = None,
) -> Dict[str, Any]:
    """对同一 conjecture 的一组候选计算 PMI 分布。
    返回 {"pmi_scores": [...], "probs": [...], "K": K}。
    """
    texts = _cand_texts(candidates)
    q = _trim(conjecture, cfg.q_max)
    texts = [_trim(t, cfg.c_max) for t in texts]
    K = len(texts)
    if K == 0:
        return {"pmi_scores": [], "probs": [], "K": 0}

    # 构造 (prefix, suffix)
    conj_prefix = f"[CONJECTURE]\n{q}\n[CANDIDATE]\n"
    prefixes = [conj_prefix] * K
    suffixes = texts

    # 条件 avg_logP(c|q)
    cond_scores: List[float] = []
    for i in range(0, K, cfg.batch_size):
        sl = slice(i, i + cfg.batch_size)
        avg_lp, _ = avg_logp_batch(
            model, tokenizer,
            prefixes[sl], suffixes[sl],
            device, cfg.max_len,
            autocast_dtype=(torch.bfloat16 if (cfg.bf16 and torch.cuda.is_available()) else None),
        )
        cond_scores.extend(avg_lp)

    # 无条件 avg_logP(c)
    uncond_scores: List[float] = []
    base_prefix = "[CANDIDATE]\n"
    for i, t in enumerate(texts):
        if cfg.cache_uncond and _uncond_cache is not None and t in _uncond_cache:
            uncond_scores.append(_uncond_cache[t])
        else:
            avg_lp, _ = avg_logp_batch(
                model, tokenizer, [base_prefix], [t], device, cfg.max_len,
                autocast_dtype=(torch.bfloat16 if (cfg.bf16 and torch.cuda.is_available()) else None),
            )
            uncond_scores.append(avg_lp[0])
            if cfg.cache_uncond and _uncond_cache is not None:
                _uncond_cache[t] = avg_lp[0]

    lam = cfg.lambda_pmi
    pmi_scores = [c - lam * u for c, u in zip(cond_scores, uncond_scores)]

    # 组内 z-score + 温度 softmax
    import numpy as np
    v = np.array(pmi_scores, dtype=np.float64)
    if v.size > 1:
        m = float(v.mean()); s = float(v.std())
        if s > 1e-8:
            v = (v - m) / s
        else:
            v = v - m
    tau = max(1e-4, float(cfg.tau))
    v = v / tau
    v = v - float(v.max())
    e = np.exp(v)
    p = e / max(e.sum(), 1e-12)

    return {"pmi_scores": pmi_scores, "probs": p.tolist(), "K": K}


# ------------------------- CLI -------------------------

def _build_argparser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="PMI rerank: avg_logP(c|q) 与 avg_logP(c) → PMI → softmax")
    ap.add_argument("--model", required=True, help="HF 模型路径或名称")
    ap.add_argument("--in", dest="inp", required=False, help="输入 JSONL")
    ap.add_argument("--out", dest="outp", required=False, help="输出 JSONL")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--lambda-pmi", type=float, default=0.7)
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--q-max", type=int, default=384)
    ap.add_argument("--c-max", type=int, default=96)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--cache-uncond", action="store_true", help="缓存无条件 avg_logP(c) 以复用")
    return ap


def main(argv: List[str] | None = None) -> int:
    ap = _build_argparser()
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token is not None:
        tok.pad_token = tok.eos_token
    dtype = torch.bfloat16 if (args.bf16 and torch.cuda.is_available()) else (torch.float16 if torch.cuda.is_available() else torch.float32)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.eval()

    cfg = PMIConfig(
        lambda_pmi=args.lambda_pmi,
        tau=args.tau,
        q_max=args.q_max,
        c_max=args.c_max,
        max_len=args.max_len,
        batch_size=args.batch_size,
        bf16=bool(args.bf16),
        cache_uncond=bool(args.cache_uncond),
    )

    uncond_cache: Dict[str, float] | None = {} if cfg.cache_uncond else None

    if args.inp and args.outp:
        inp_path = Path(args.inp)
        out_path = Path(args.outp)
        n_in = 0; n_out = 0
        with open(inp_path, "r", encoding="utf-8", errors="ignore") as fin, open(out_path, "w", encoding="utf-8") as fout:
            for line in fin:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                conj = obj.get("conjecture") or obj.get("query") or ""
                cands = obj.get("candidates") or []
                # If still missing, try to parse from a prompt-style field
                if (not conj or not cands) and _parse_prompt is not None:
                    prompt_field = obj.get("input") or obj.get("prompt")
                    if isinstance(prompt_field, str) and prompt_field.strip():
                        pq, tuples = _parse_prompt(prompt_field)
                        if pq and tuples:
                            conj = conj or pq
                            # Only build candidates if original list absent
                            if not cands:
                                cands = [{"text": t, "meta": {"tags": tags}} for (_cid, t, tags) in tuples]
                if not conj or not cands:
                    continue
                res = compute_pmi_for_group(model, tok, conj, cands, cfg, device, _uncond_cache=uncond_cache)
                obj.update(res)
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
                n_out += 1
                n_in += 1
        print(f"[PMI] processed={n_out} written={out_path}")
    else:
        # Demo 模式
        demo = {
            "conjecture": "![X] (P(X) => Q(X))",
            "candidates": ["P(a)", "~P(a) | Q(a)", "R(a)"]
        }
        res = compute_pmi_for_group(model, tok, demo["conjecture"], demo["candidates"], cfg, device, _uncond_cache=uncond_cache)
        print(json.dumps(res, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    if "ipykernel" in sys.modules:
        # Notebook 环境下不自动执行 CLI
        pass
    else:
        raise SystemExit(main())
