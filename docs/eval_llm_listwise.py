#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple, Any

import numpy as np
import requests

# reuse JSON extractor from batch_ranker for robustness
from LLM.batch_ranker import extract_json


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


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Eval LLM listwise reranker on val JSONL: JSON rate/len match/MAE/MSE/Pearson")
    ap.add_argument("--data", type=Path, required=True, help="val_listwise.jsonl")
    ap.add_argument("--limit", type=int, default=200, help="max samples to evaluate (0=all)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--max-new", type=int, default=256)
    ap.add_argument("--timeout", type=float, default=60.0)
    ap.add_argument("--endpoint", type=str, default=os.getenv("LLM_LOCAL_ENDPOINT", ""), help="POST endpoint /generate; default from env LLM_LOCAL_ENDPOINT")
    args = ap.parse_args(argv)

    if not args.endpoint:
        raise SystemExit("LLM_LOCAL_ENDPOINT not set and --endpoint empty.")

    n_total = 0
    n_json_ok = 0
    n_len_match = 0
    maes: List[float] = []
    mses: List[float] = []
    pears: List[float] = []

    with open(args.data, "r", encoding="utf-8", errors="ignore") as f:
        for li, ln in enumerate(f, start=1):
            if args.limit and (n_total >= args.limit):
                break
            try:
                j = json.loads(ln)
            except Exception:
                continue
            prompt = j.get("input") or j.get("prompt")
            if not prompt:
                continue
            # Add the same strict JSON instruction used in training to stabilize outputs
            sys_inst = (
                "你是自动定理证明的子句打分器。只输出合法 JSON（无多余文本）。\n"
                "必须严格输出：{\"scores\":[...]}，且长度等于候选数 K。\n"
            )
            prompt_full = sys_inst + "\n" + prompt + "\n只输出 JSON："
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
            # Call endpoint
            try:
                r = requests.post(
                    args.endpoint,
                    json={
                        "prompt": prompt_full,
                        "temperature": args.temperature,
                        "max_new_tokens": args.max_new,
                    },
                    timeout=args.timeout,
                )
                r.raise_for_status()
                obj = r.json()
                text = obj.get("text") or obj.get("output") or obj.get("generated_text") or ""
            except Exception:
                text = ""
            parsed = extract_json(text)
            pred_vec = _flatten_scores(parsed)

            n_total += 1
            json_ok = isinstance(parsed, dict) and ("scores" in parsed)
            if json_ok:
                n_json_ok += 1
            len_match = bool(tgt_vec) and (len(pred_vec) == len(tgt_vec))
            if len_match:
                n_len_match += 1

            # numeric metrics on overlap
            n = min(len(pred_vec), len(tgt_vec))
            if n >= 1:
                a = np.asarray(pred_vec[:n], dtype=np.float64)
                b = np.asarray(tgt_vec[:n], dtype=np.float64)
                maes.append(float(np.mean(np.abs(a - b))))
                mses.append(float(np.mean((a - b) ** 2)))
                pr = _pearsonr(pred_vec[:n], tgt_vec[:n])
                if not np.isnan(pr):
                    pears.append(float(pr))

    if n_total == 0:
        raise SystemExit("No samples evaluated.")

    out = {
        "n": n_total,
        "json_ok_rate": n_json_ok / n_total,
        "len_match_rate": n_len_match / n_total,
        "mae": (sum(maes) / len(maes)) if maes else None,
        "mse": (sum(mses) / len(mses)) if mses else None,
        "pearson": (sum(pears) / len(pears)) if pears else None,
    }
    print(json.dumps(out, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
