#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers import BitsAndBytesConfig  # type: ignore

from LLM.training.eval_llm_listwise import _direct_scores_for_many_windows


def load_meta_texts(meta_path: Path) -> List[str]:
    texts: List[str] = []
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            texts.append(j.get("text") or j.get("canonical_formula") or "")
    return texts


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rerank queries' topK candidates with an LLM (direct log-likelihood)")
    ap.add_argument("--queries", type=Path, required=True, help="queries_with_topK_ids.jsonl (has query, candidate_ids)")
    ap.add_argument("--index-meta", type=Path, required=True, help="Index .meta.jsonl mapping id -> text")
    ap.add_argument("--model", type=str, required=True, help="HF model id/path for in-process eval")
    ap.add_argument("--lora", type=str, default=None, help="Optional PEFT LoRA adapter path or parent dir containing lora/")
    ap.add_argument("--bits", type=int, choices=[4, 8, 16], default=4, help="Quantization: 4/8-bit (bnb) or 16 (no quant)")
    ap.add_argument("--attn", type=str, default="auto", choices=["auto","flash","eager"], help="attention impl: flash uses flash_attention_2 if supported")
    ap.add_argument("--input-max", type=int, default=1536)
    ap.add_argument("--target-max", type=int, default=192)
    ap.add_argument("--row-batch", type=int, default=64)
    # scoring controls (match eval_llm_listwise)
    ap.add_argument("--score-type", type=str, choices=["sum-ll", "mean-ll"], default="sum-ll",
                    help="Use sum or mean of token log-likelihoods per candidate")
    ap.add_argument("--segment", type=str, choices=["full", "text"], default="text",
                    help="Which part of candidate block to score: full block (TEXT+TAGS) or just TEXT")
    ap.add_argument("--zscore", action="store_true", help="Apply per-query z-score normalization before softmax")
    ap.add_argument("--calib-tau", type=float, default=1.0, help="Temperature for calibration before softmax")
    ap.add_argument("--calib-bias", type=float, default=0.0, help="Bias shift before softmax")
    ap.add_argument("--pmi-lambda", type=float, default=0.0, help="If >0, subtract lambda*logP(c) scored without conjecture")
    ap.add_argument("--limit", type=int, default=0, help="0 = all")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL with {query, scores:[{id,text,score}]} per line")
    args = ap.parse_args(argv)

    # tokenizer/model
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
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
    # resize embeddings to tokenizer length (LoRA expects matched shapes)
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
    if args.lora:
        # accept parent dir containing lora/
        lora_path = args.lora
        p = Path(lora_path)
        if p.is_dir() and not (p / "adapter_config.json").exists() and (p / "lora" / "adapter_config.json").exists():
            lora_path = str(p / "lora")
        from peft import PeftModel  # type: ignore
        mdl = PeftModel.from_pretrained(mdl, lora_path)
    mdl.eval()

    # speed hints
    try:
        torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore[attr-defined]
        torch.set_float32_matmul_precision("high")  # type: ignore[attr-defined]
    except Exception:
        pass

    # load meta texts
    meta_texts = load_meta_texts(args.index_meta)

    # read queries, build windows in small groups to control memory
    with open(args.queries, "r", encoding="utf-8", errors="ignore") as fq, \
         open(args.out, "w", encoding="utf-8") as fo:
        buffer: List[Tuple[str, List[Tuple[int, str, str]]]] = []
        q_rows: List[dict] = []
        n = 0
        for line in fq:
            if args.limit and n >= args.limit:
                break
            try:
                j = json.loads(line)
            except Exception:
                continue
            q = j.get("query") or j.get("text_a") or j.get("conjecture_text") or j.get("q") or ""
            cand_ids = j.get("candidate_ids") or []
            cands: List[Tuple[int, str, str]] = []
            for cid in cand_ids:
                try:
                    txt = meta_texts[int(cid)]
                except Exception:
                    txt = ""
                cands.append((int(cid), str(txt), ""))
            if not cands:
                continue
            buffer.append((str(q), cands))
            q_rows.append({"query": q, "candidate_ids": cand_ids})
            n += 1

            # process in chunks of queries to bound latency/memory
            if len(buffer) >= 16:
                scores_list = _direct_scores_for_many_windows(
                    mdl, tok, buffer, args.input_max, args.target_max, args.row_batch,
                    score_type=args.score_type, segment=args.segment,
                )
                # Optional PMI baseline scores (no conjecture)
                if args.pmi_lambda and args.pmi_lambda > 0.0:
                    null_windows: List[Tuple[str, List[Tuple[int, str, str]]]] = [("", ws[1]) for ws in buffer]
                    base_scores_list = _direct_scores_for_many_windows(
                        mdl, tok, null_windows, 0, args.target_max, args.row_batch,
                        score_type=args.score_type, segment=args.segment,
                    )
                else:
                    base_scores_list = [[] for _ in scores_list]
                for row, scores in zip(q_rows, scores_list):
                    cand_ids2 = row["candidate_ids"]
                    # Per-query normalization/calibration
                    v = torch.tensor(scores, dtype=torch.float32)
                    if args.pmi_lambda and args.pmi_lambda > 0.0:
                        bs = torch.tensor(base_scores_list[q_rows.index(row)], dtype=torch.float32)
                        if bs.shape == v.shape:
                            v = v - float(args.pmi_lambda) * bs
                    if args.zscore:
                        mu = float(v.mean())
                        sd = float(v.std())
                        v = (v - mu) / (sd if sd > 1e-8 else 1.0)
                    v = (v - float(args.calib_bias)) / max(1e-8, float(args.calib_tau))
                    v = v - float(v.max())
                    p = torch.exp(v)
                    ssum = float(p.sum())
                    if ssum > 0:
                        p = p / ssum
                    items = []
                    for cid, sc in zip(cand_ids2, p.tolist()):
                        txt = meta_texts[int(cid)] if 0 <= int(cid) < len(meta_texts) else ""
                        items.append({"id": int(cid), "text": txt, "score": float(sc)})
                    items.sort(key=lambda x: x["score"], reverse=True)
                    fo.write(json.dumps({"query": row["query"], "scores": items}, ensure_ascii=False) + "\n")
                buffer.clear()
                q_rows.clear()

        # flush remainder
        if buffer:
            scores_list = _direct_scores_for_many_windows(
                mdl, tok, buffer, args.input_max, args.target_max, args.row_batch,
                score_type=args.score_type, segment=args.segment,
            )
            if args.pmi_lambda and args.pmi_lambda > 0.0:
                null_windows = [("", ws[1]) for ws in buffer]
                base_scores_list = _direct_scores_for_many_windows(
                    mdl, tok, null_windows, 0, args.target_max, args.row_batch,
                    score_type=args.score_type, segment=args.segment,
                )
            else:
                base_scores_list = [[] for _ in scores_list]
            for row_idx, (row, scores) in enumerate(zip(q_rows, scores_list)):
                cand_ids2 = row["candidate_ids"]
                v = torch.tensor(scores, dtype=torch.float32)
                if args.pmi_lambda and args.pmi_lambda > 0.0:
                    bs = torch.tensor(base_scores_list[row_idx], dtype=torch.float32)
                    if bs.shape == v.shape:
                        v = v - float(args.pmi_lambda) * bs
                if args.zscore:
                    mu = float(v.mean())
                    sd = float(v.std())
                    v = (v - mu) / (sd if sd > 1e-8 else 1.0)
                v = (v - float(args.calib_bias)) / max(1e-8, float(args.calib_tau))
                v = v - float(v.max())
                p = torch.exp(v)
                ssum = float(p.sum())
                if ssum > 0:
                    p = p / ssum
                items = []
                for cid, sc in zip(cand_ids2, p.tolist()):
                    txt = meta_texts[int(cid)] if 0 <= int(cid) < len(meta_texts) else ""
                    items.append({"id": int(cid), "text": txt, "score": float(sc)})
                items.sort(key=lambda x: x["score"], reverse=True)
                fo.write(json.dumps({"query": row["query"], "scores": items}, ensure_ascii=False) + "\n")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
