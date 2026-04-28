#!/usr/bin/env python3
"""
Unified listwise ranking evaluator for multiple model types over a common dataset:
- bi: Bi-Encoder (LogicSentencePiece + cosine sim)
- ce: Cross-Encoder (TransformerEncoder + CrossHead)
- llm-direct: HF CausalLM direct scoring by token log-likelihood (supports LoRA, 4/8-bit, segment/text or full)
- llm-head: 7B LLM + span-pooled scoring head checkpoint trained by train_llm_head.py

Dataset formats supported (auto-detected per line):
1) Structured: {"problem_name", "query", "candidates":[{"text","meta",...},...], "target_scores"?}
2) Prompt:     {"input": "[CONJECTURE]\n... [CANDIDATE] - TEXT: ... - TAGS: ..."}

Outputs metrics: len_match_rate, MAE, MSE, KL(target||pred), Pearson, Spearman, NDCG@1/3/5/10, Hit@1/3/5; optional JSONL with predictions.

Examples:
python -m LLM.training.eval_unified_rank \
  --data /path/val.jsonl --limit 200 --model-type llm-direct \
  --hf-model /root/autodl-tmp/models/Goedel-Prover-V2-32B \
  --lora /root/autodl-tmp/Training/models/goedel32b-listwise-lora-v2/lora \
  --bits 4 --input-max 896 --target-max 128 --row-batch 8 \
  --score-type mean-ll --segment text --zscore --tau 1.0 --bias 0.0 \
  --out /tmp/val.unified.llm32.pred.jsonl

python -m LLM.training.eval_unified_rank \
  --data /path/val_structured.jsonl --model-type llm-head \
  --hf-model /root/autodl-tmp/models/DeepSeek-Prover-V2-7B \
  --head-ckpt /root/autodl-tmp/Training/models/7b-head/ckpt.pt \
  --max-len 1024 --batch 1 --zscore --tau 1.0 --bias 0.0

python -m LLM.training.eval_unified_rank \
  --data /path/val.jsonl --model-type bi \
  --bi-ckpt /root/Training/models/biencoder_best.pt --spm /root/Training/models/spm_logic_24k.model

python -m LLM.training.eval_unified_rank \
  --data /path/val.jsonl --model-type ce \
  --cross-ckpt /root/Training/models/cross_encoder_best.pt --spm /root/Training/models/spm_logic_24k.model
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np

# Optional torch/transformers imports (for LLM modes)
try:
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
    from transformers import BitsAndBytesConfig  # type: ignore
except Exception:  # pragma: no cover
    torch = None  # type: ignore
    AutoModelForCausalLM = None  # type: ignore
    AutoTokenizer = None  # type: ignore
    BitsAndBytesConfig = None  # type: ignore

# Reuse existing utilities
from LLM.scripts.make_listwise_targets_from_teacher import (
    _parse_prompt_for_q_and_candidates as parse_from_prompt,
    build_bi_model,
    build_cross_model,
    score_bi as score_bi_encoder,
    score_cross as score_cross_encoder,
)
from LLM.training.eval_llm_listwise import _direct_scores_for_many_windows as direct_llm_scores
from LLM.training.train_llm_head import ScoreHead, format_prompt
from LLM.training.token_utils import ensure_special_tokens


# ------------------ Common dataset adapter ------------------

@dataclass
class Example:
    problem_name: str
    query: str
    candidates: List[Dict[str, Any]]  # [{'text': str, 'meta': {...}}]
    target: Optional[List[float]]     # soft labels or None


def parse_line_to_example(j: Dict[str, Any]) -> Optional[Example]:
    # Format 1: structured
    q = j.get("query")
    cands = j.get("candidates")
    if isinstance(q, str) and isinstance(cands, list) and cands:
        tgt = j.get("target_scores")
        if isinstance(tgt, list):
            tgt = [float(x) for x in tgt if isinstance(x, (int, float))]
        else:
            tgt = None
        return Example(
            problem_name=str(j.get("problem_name") or j.get("id") or ""),
            query=q,
            candidates=cands,
            target=tgt,
        )
    # Format 2: prompt-only
    prompt = j.get("input") or j.get("prompt")
    if isinstance(prompt, str) and prompt.strip():
        conj, tuples = parse_from_prompt(prompt)
        if conj and tuples:
            c_list: List[Dict[str, Any]] = []
            for (_cid, text, tags) in tuples:
                meta = {"tags": tags}
                c_list.append({"text": text, "meta": meta})
            tgt = j.get("target_scores")
            if isinstance(tgt, list):
                tgt = [float(x) for x in tgt if isinstance(x, (int, float))]
            else:
                tgt = None
            return Example(
                problem_name=str(j.get("problem_name") or j.get("id") or ""),
                query=conj,
                candidates=c_list,
                target=tgt,
            )
    return None


def iter_examples(path: Path, limit: int = 0) -> Iterable[Example]:
    n = 0
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            try:
                j = json.loads(ln)
            except Exception:
                continue
            ex = parse_line_to_example(j)
            if ex is None:
                continue
            yield ex
            n += 1
            if limit and n >= limit:
                break


# ------------------ Metrics ------------------

def _entropy(p: List[float]) -> float:
    s = 0.0
    for x in p:
        if x > 0.0:
            s -= x * math.log(x + 1e-12)
    return s


def _ndcg_at_k(rel: List[float], pred: List[float], k: int) -> float:
    idx = list(range(len(rel)))
    order = sorted(idx, key=lambda i: pred[i], reverse=True)[:k]
    dcg = sum((rel[i]) / math.log2(j + 2) for j, i in enumerate(order))
    order2 = sorted(idx, key=lambda i: rel[i], reverse=True)[:k]
    idcg = sum((rel[i]) / math.log2(j + 2) for j, i in enumerate(order2))
    return (dcg / idcg) if idcg > 0 else 0.0


def _pearson(a: List[float], b: List[float]) -> Optional[float]:
    if not a or not b:
        return None
    n = min(len(a), len(b))
    if n < 2:
        return None
    a1 = np.asarray(a[:n], dtype=np.float64)
    b1 = np.asarray(b[:n], dtype=np.float64)
    if np.all(a1 == a1[0]) or np.all(b1 == b1[0]):
        return None
    return float(np.corrcoef(a1, b1)[0, 1])


def _spearman(a: List[float], b: List[float]) -> Optional[float]:
    try:
        import scipy.stats as ss  # type: ignore
    except Exception:
        return None
    n = min(len(a), len(b))
    if n < 2:
        return None
    rho, _ = ss.spearmanr(a[:n], b[:n])
    if isinstance(rho, float) and (rho == rho):  # not NaN
        return float(rho)
    return None


def _softmax(v: np.ndarray, tau: float = 1.0, bias: float = 0.0) -> np.ndarray:
    v = (v - float(bias)) / max(1e-8, float(tau))
    v = v - float(v.max())
    e = np.exp(v)
    s = float(e.sum())
    if s <= 0:
        return np.ones_like(v) / max(1, len(v))
    return e / s


# ------------------ Model adapters ------------------

@dataclass
class EvalConfig:
    zscore: bool
    tau: float
    bias: float
    segment: str
    score_type: str


class BiCEScorer:
    def __init__(self, model_type: str, cross_ckpt: Optional[str], bi_ckpt: Optional[str], spm_path: str, device: Optional[str] = None):
        assert model_type in ("bi", "ce")
        self.model_type = model_type
        dev = torch.device(device or ("cuda" if torch and torch.cuda.is_available() else "cpu")) if torch else None
        self.device = dev
        if model_type == "ce":
            assert cross_ckpt and spm_path
            self.ce_enc, self.ce_head, self.ce_tok, _ = build_cross_model(cross_ckpt, spm_path, dev)  # type: ignore
            self.bi = None
            self.bi_tok = None
        else:
            assert bi_ckpt and spm_path
            self.bi, _ = build_bi_model(bi_ckpt, dev)  # type: ignore
            from LLM.data_utils.logic_tokenizer import LogicSentencePiece  # type: ignore
            self.bi_tok = LogicSentencePiece(spm_path)
            self.ce_enc = None
            self.ce_head = None
            self.ce_tok = None

    @torch.no_grad()
    def score_window(self, q: str, cands: List[Dict[str, Any]], eval_cfg: EvalConfig) -> List[float]:
        # Return probability vector per window
        if self.model_type == "ce":
            tuples = [(i, c.get("text", ""), str((c.get("meta") or {}).get("tags", ""))) for i, c in enumerate(cands)]
            raw = np.array(score_cross_encoder(self.ce_enc, self.ce_head, self.ce_tok, q, tuples, max_len=256, batch=256, device=self.device), dtype=float)
        else:
            tuples = [(i, c.get("text", ""), str((c.get("meta") or {}).get("tags", ""))) for i, c in enumerate(cands)]
            raw = np.array(score_bi_encoder(self.bi, self.bi_tok, q, tuples, max_len=256, batch=256, device=self.device), dtype=float)
        v = raw.astype(np.float64)
        if eval_cfg.zscore and v.size > 1:
            mu = float(v.mean()); sd = float(v.std())
            v = (v - mu) / (sd if sd > 1e-8 else 1.0)
        p = _softmax(v, tau=eval_cfg.tau, bias=eval_cfg.bias).tolist()
        return p


class LLMDirectScorer:
    def __init__(self, hf_model: str, lora: Optional[str], bits: int = 4, attn: str = "auto"):
        assert AutoModelForCausalLM is not None and AutoTokenizer is not None
        tok = AutoTokenizer.from_pretrained(hf_model, use_fast=True)
        try:
            tok.truncation_side = "left"  # keep task tail
        except Exception:
            pass
        if getattr(tok, "pad_token", None) is None:
            tok.pad_token = tok.eos_token
        quant: Optional[BitsAndBytesConfig] = None
        torch_dtype = torch.bfloat16 if torch and torch.cuda.is_available() else torch.float32
        device_map = "auto" if torch and torch.cuda.is_available() else None
        if bits == 4:
            quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype)
        elif bits == 8:
            quant = BitsAndBytesConfig(load_in_8bit=True)
        attn_impl = None
        if attn == "flash":
            attn_impl = "flash_attention_2"
        elif attn == "eager":
            attn_impl = "eager"
        if quant is not None:
            mdl = AutoModelForCausalLM.from_pretrained(hf_model, quantization_config=quant, device_map=device_map, torch_dtype=torch_dtype, attn_implementation=attn_impl)
        else:
            mdl = AutoModelForCausalLM.from_pretrained(hf_model, device_map=device_map, torch_dtype=torch_dtype, attn_implementation=attn_impl)
        # resize embeddings to tokenizer size
        try:
            if getattr(mdl.config, "vocab_size", None) != len(tok):
                mdl.resize_token_embeddings(len(tok))
        except Exception:
            pass
        if lora:
            from peft import PeftModel  # type: ignore
            p = Path(lora)
            if p.is_dir() and not (p / "adapter_config.json").exists() and (p / "lora" / "adapter_config.json").exists():
                lora = str(p / "lora")
            mdl = PeftModel.from_pretrained(mdl, lora)
        mdl.eval()
        self.model = mdl
        self.tok = tok

    @torch.no_grad()
    def score_window(self, q: str, cands: List[Dict[str, Any]], eval_cfg: EvalConfig, input_max: int, target_max: int, row_batch: int) -> List[float]:
        tuples: List[Tuple[int, str, str]] = []
        for i, c in enumerate(cands):
            text = c.get("text", "")
            tags = (c.get("meta") or {}).get("tags", "")
            tuples.append((i, text, str(tags)))
        scores = direct_llm_scores(
            self.model, self.tok,
            [(q, tuples)],
            input_max=input_max,
            target_max=target_max,
            row_batch=row_batch,
            score_type=eval_cfg.score_type,
            segment=eval_cfg.segment,
        )[0]
        v = np.asarray(scores, dtype=np.float64)
        if eval_cfg.zscore and v.size > 1:
            mu = float(v.mean()); sd = float(v.std())
            v = (v - mu) / (sd if sd > 1e-8 else 1.0)
        p = _softmax(v, tau=eval_cfg.tau, bias=eval_cfg.bias).tolist()
        return p


class LLMHeadScorer:
    def __init__(self, hf_model: str, ckpt_path: str, max_len: int = 1024, use_flash: bool = False):
        # Load base + tokens
        res = ensure_special_tokens(hf_model, load_kwargs={"device_map": "auto"})
        self.tok = res.tokenizer
        self.base = res.model
        if use_flash and hasattr(self.base.config, "attn_implementation"):
            try:
                self.base.config.attn_implementation = "flash_attention_2"
            except Exception:
                pass
        # Build head and load ckpt
        hidden = self.base.config.hidden_size
        self.head = ScoreHead(hidden, pool="mean").to(next(self.base.parameters()).dtype)
        # ckpt format is from train_llm_head.py
        state = torch.load(ckpt_path, map_location="cpu")
        # Load LoRA weights into base
        from peft import LoraConfig, get_peft_model  # type: ignore
        lcfg = LoraConfig(
            r=state.get("config", {}).get("lora_r", 16),
            lora_alpha=state.get("config", {}).get("lora_alpha", 32),
            lora_dropout=state.get("config", {}).get("lora_drop", 0.05),
            bias="none",
            task_type="CAUSAL_LM",
            target_modules=["q_proj","k_proj","v_proj","o_proj"],
        )
        self.base = get_peft_model(self.base, lcfg)
        self.base.load_state_dict(state["lora"], strict=False)
        self.head.load_state_dict(state["head"], strict=True)
        self.base.eval(); self.head.eval()
        self.max_len = max_len

    @torch.no_grad()
    def score_window(self, q: str, cands: List[Dict[str, Any]], eval_cfg: EvalConfig) -> List[float]:
        # Build prompt and locate spans
        prompt = format_prompt(q, cands)
        enc = self.tok([prompt], max_length=self.max_len, truncation=True, padding=True, return_tensors="pt")
        start_id = self.tok.convert_tokens_to_ids("<CAND_START>")
        end_id = self.tok.convert_tokens_to_ids("<CAND_END>")
        # find spans
        inp = enc.input_ids
        spans: List[List[Tuple[int,int]]] = [[]]
        seq = inp[0].tolist()
        starts = []
        for i, tid in enumerate(seq):
            if tid == start_id:
                starts.append(i)
            elif tid == end_id and starts:
                st = starts.pop(0)
                spans[0].append((st, i))
        # forward
        device = next(self.base.parameters()).device
        for k in enc:
            enc[k] = enc[k].to(device)
        out = self.base(**enc, output_hidden_states=True)
        last = out.hidden_states[-1]
        logits = self.head(last, spans)
        v = logits[0].detach().float().cpu().numpy().astype(np.float64)
        # mask -1e9 paddings if any (shouldn't happen with per-window)
        v = v[~np.isneginf(v)] if np.any(np.isneginf(v)) else v
        if eval_cfg.zscore and v.size > 1:
            mu = float(v.mean()); sd = float(v.std())
            v = (v - mu) / (sd if sd > 1e-8 else 1.0)
        p = _softmax(v, tau=eval_cfg.tau, bias=eval_cfg.bias).tolist()
        return p


# ------------------ Runner ------------------

def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Unified ranking evaluator across BI/CE/LLM-direct/LLM-head")
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", type=Path, default=None, help="Write per-line predictions JSONL with pred_scores")

    ap.add_argument("--model-type", choices=["bi","ce","llm-direct","llm-head"], required=True)

    # BI/CE
    ap.add_argument("--cross-ckpt", type=str, default=None)
    ap.add_argument("--bi-ckpt", type=str, default=None)
    ap.add_argument("--spm", type=str, default=None)

    # LLM-direct
    ap.add_argument("--hf-model", type=str, default=None)
    ap.add_argument("--lora", type=str, default=None)
    ap.add_argument("--bits", type=int, choices=[4,8,16], default=4)
    ap.add_argument("--score-type", choices=["sum-ll","mean-ll"], default="mean-ll")
    ap.add_argument("--segment", choices=["text","full"], default="text")
    ap.add_argument("--input-max", type=int, default=896)
    ap.add_argument("--target-max", type=int, default=128)
    ap.add_argument("--row-batch", type=int, default=8)

    # LLM-head
    ap.add_argument("--head-ckpt", type=str, default=None)
    ap.add_argument("--max-len", type=int, default=1024)

    # Calibration
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--tau", type=float, default=1.0)
    ap.add_argument("--bias", type=float, default=0.0)

    args = ap.parse_args(argv)

    # Build scorer
    eval_cfg = EvalConfig(zscore=bool(args.zscore), tau=float(args.tau), bias=float(args.bias), segment=str(getattr(args, "segment", "text")), score_type=str(getattr(args, "score_type", "mean-ll")))

    if args.model_type in ("bi","ce"):
        if not args.spm:
            raise SystemExit("--spm is required for bi/ce")
        scorer = BiCEScorer(args.model_type, args.cross_ckpt, args.bi_ckpt, args.spm)
    elif args.model_type == "llm-direct":
        if not args.hf_model:
            raise SystemExit("--hf-model is required for llm-direct")
        scorer = LLMDirectScorer(args.hf_model, args.lora, bits=args.bits)
    else:  # llm-head
        if not args.hf_model or not args.head_ckpt:
            raise SystemExit("--hf-model and --head-ckpt are required for llm-head")
        scorer = LLMHeadScorer(args.hf_model, args.head_ckpt, max_len=args.max_len)

    # Iterate and score
    n_total = 0
    n_len_match = 0
    maes: List[float] = []
    mses: List[float] = []
    pears: List[float] = []
    spears: List[float] = []
    kls: List[float] = []
    ents_t: List[float] = []
    ents_p: List[float] = []
    ndcg1: List[float] = []; ndcg3: List[float] = []; ndcg5: List[float] = []; ndcg10: List[float] = []
    hit1: List[float] = []; hit3: List[float] = []; hit5: List[float] = []

    out_lines: List[str] = []

    for ex in iter_examples(args.data, args.limit):
        # Skip empty windows
        if not ex.query or not ex.candidates:
            continue
        # Produce window probs
        if args.model_type in ("bi","ce"):
            pred = scorer.score_window(ex.query, ex.candidates, eval_cfg)
        elif args.model_type == "llm-direct":
            pred = scorer.score_window(ex.query, ex.candidates, eval_cfg, input_max=args.input_max, target_max=args.target_max, row_batch=args.row_batch)
        else:
            pred = scorer.score_window(ex.query, ex.candidates, eval_cfg)

        tgt = ex.target
        if tgt is None:
            # If no target, we still can write out predictions
            j = {
                "problem_name": ex.problem_name,
                "query": ex.query,
                "candidates": ex.candidates,
                "pred_scores": pred,
            }
            if args.out is not None:
                out_lines.append(json.dumps(j, ensure_ascii=False))
            n_total += 1
            continue

        # Normalize target/pred to probabilities
        a = np.asarray(pred, dtype=np.float64)
        b = np.asarray([float(x) for x in tgt], dtype=np.float64)
        K = min(a.size, b.size)
        if K <= 0:
            continue
        a = a[:K]; b = b[:K]
        sa = float(a.sum()); sb = float(b.sum())
        if sa > 0: a = a / sa
        if sb > 0: b = b / sb

        n_total += 1
        if a.size == b.size:
            n_len_match += 1

        ents_t.append(_entropy(b.tolist()))
        ents_p.append(_entropy(a.tolist()))
        maes.append(float(np.mean(np.abs(a - b))))
        mses.append(float(np.mean((a - b) ** 2)))
        pr = _pearson(a.tolist(), b.tolist())
        if pr is not None:
            pears.append(pr)
        sr = _spearman(a.tolist(), b.tolist())
        if sr is not None:
            spears.append(sr)
        # KL target||pred
        eps = 1e-12
        kl = float(np.sum(b * np.log((b + eps) / (a + eps))))
        if kl == kl:  # not NaN
            kls.append(kl)
        # Ranking metrics
        ndcg1.append(_ndcg_at_k(b.tolist(), a.tolist(), 1))
        ndcg3.append(_ndcg_at_k(b.tolist(), a.tolist(), 3))
        ndcg5.append(_ndcg_at_k(b.tolist(), a.tolist(), 5))
        ndcg10.append(_ndcg_at_k(b.tolist(), a.tolist(), 10))
        top_t = int(np.argmax(b))
        order = np.argsort(-a)
        hit1.append(1.0 if top_t in order[:1] else 0.0)
        hit3.append(1.0 if top_t in order[:3] else 0.0)
        hit5.append(1.0 if top_t in order[:5] else 0.0)

        if args.out is not None:
            j = {
                "problem_name": ex.problem_name,
                "query": ex.query,
                "candidates": ex.candidates,
                "target_scores": b.tolist(),
                "pred_scores": a.tolist(),
            }
            out_lines.append(json.dumps(j, ensure_ascii=False))

    if args.out is not None and out_lines:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as w:
            for ln in out_lines:
                w.write(ln + "\n")

    if n_total == 0:
        raise SystemExit("No samples evaluated.")

    def _avg(xs: List[float]) -> Optional[float]:
        return (float(np.mean(xs)) if xs else None)

    out = {
        "n": n_total,
        "len_match_rate": n_len_match / n_total,
        "MAE": _avg(maes),
        "MSE": _avg(mses),
        "KL_target||pred": _avg(kls),
        "Pearson": _avg(pears),
        "Spearman": _avg(spears),
        "entropy_target_avg": _avg(ents_t),
        "entropy_pred_avg": _avg(ents_p),
        "NDCG@1": _avg(ndcg1),
        "NDCG@3": _avg(ndcg3),
        "NDCG@5": _avg(ndcg5),
        "NDCG@10": _avg(ndcg10),
        "Hit@1": _avg(hit1),
        "Hit@3": _avg(hit3),
        "Hit@5": _avg(hit5),
        "model_type": args.model_type,
        "calibration": {"zscore": bool(args.zscore), "tau": float(args.tau), "bias": float(args.bias)},
        "llm_scoring": {"score_type": getattr(args, "score_type", None), "segment": getattr(args, "segment", None)} if args.model_type == "llm-direct" else None,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
