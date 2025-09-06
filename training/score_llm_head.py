#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from tqdm.auto import tqdm
from peft import LoraConfig, get_peft_model

from LLM.training.token_utils import ensure_special_tokens, locate_token_positions


def format_prompt(q: str, cands: List[dict]) -> str:
    parts = ["[CONJECTURE]\n", f"<Q> {q} </Q>\n"]
    for i, c in enumerate(cands, 1):
        tags = c.get("meta") or {}
        tag_str = " ".join(f"<{k}={v}>" for k, v in tags.items() if v is not None)
        parts.append(f"[CANDIDATE {i}]\n<CAND_START>\n{c.get('text','')}\n{tag_str}\n<CAND_END>\n")
    return "".join(parts)


class ScoreHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, 1)

    def forward(self, last_hidden: torch.Tensor, cand_pos: List[List[int]]) -> torch.Tensor:
        B, T, H = last_hidden.size()
        out = []
        for b in range(B):
            idxs = cand_pos[b]
            if not idxs:
                out.append(torch.zeros(1, device=last_hidden.device))
                continue
            hs = last_hidden[b, idxs, :]
            s = self.proj(hs).squeeze(-1)
            out.append(s)
        K = max(x.size(0) for x in out)
        padded = []
        for s in out:
            if s.size(0) < K:
                pad = torch.full((K - s.size(0),), -1e9, device=s.device)
                s = torch.cat([s, pad], dim=0)
            padded.append(s)
        return torch.stack(padded, dim=0)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Score listwise groups with a trained LLM scoring head (LoRA)")
    ap.add_argument("--ckpt", type=Path, required=True, help="Checkpoint saved by train_llm_head.py")
    ap.add_argument("--groups", action="append", required=True, help="groups.*.jsonl files")
    ap.add_argument("--out", type=Path, required=True, help="Output unified predictions JSONL")
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--bf16", action="store_true")
    args = ap.parse_args(argv)

    dtype = torch.bfloat16 if args.bf16 else torch.float32

    # Load checkpoint
    ckpt = torch.load(args.ckpt, map_location="cpu")
    base = ckpt.get("base_model")
    if base is None:
        raise ValueError("Checkpoint missing base_model path")
    lora_cfg_d = ckpt.get("lora_cfg") or {"r": 16, "lora_alpha": 32, "lora_dropout": 0.05, "target_modules": ["q_proj","k_proj","v_proj","o_proj"]}

    # Load model/tokenizer with special tokens
    res = ensure_special_tokens(base, load_kwargs={"torch_dtype": dtype, "device_map": "auto"})
    tok, base_model = res.tokenizer, res.model
    device = next(base_model.parameters()).device

    # Attach LoRA and head
    lora_cfg = LoraConfig(
        r=lora_cfg_d.get("r", 16),
        lora_alpha=lora_cfg_d.get("lora_alpha", 32),
        lora_dropout=lora_cfg_d.get("lora_dropout", 0.05),
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=lora_cfg_d.get("target_modules", ["q_proj","k_proj","v_proj","o_proj"]),
    )
    model = get_peft_model(base_model, lora_cfg)
    model.load_state_dict(ckpt["lora"], strict=False)
    head = ScoreHead(base_model.config.hidden_size).to(device)
    head.load_state_dict(ckpt["head"], strict=True)
    model.eval(); head.eval()

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fout:
        for gp in args.groups:
            with open(gp, "r", encoding="utf-8", errors="ignore") as fin:
                for line in tqdm(fin, desc=f"score {Path(gp).name}"):
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    q = j.get("query") or ""
                    cands = j.get("candidates") or []
                    if not q or not cands:
                        continue
                    pn = j.get("problem_name") or ""
                    gid = j.get("group_id") or f"{pn}||{q}"

                    text = format_prompt(q, cands)
                    enc = tok(text, max_length=args.max_len, truncation=True, return_tensors="pt")
                    for k in enc:
                        enc[k] = enc[k].to(device)
                    cand_id = tok.convert_tokens_to_ids("<CAND_START>")
                    cand_pos = locate_token_positions(enc["input_ids"], cand_id)

                    with torch.no_grad():
                        out = model(**enc, output_hidden_states=True)
                        last = out.hidden_states[-1]
                        s = head(last, cand_pos)  # [1,K]
                        s = s[0]
                        s_norm = (s - s.mean()) / (s.std() + 1e-6)

                    for i, c in enumerate(cands):
                        d = c.get("text", "")
                        y = float(c.get("label", 0))
                        if i >= s.size(0):
                            score = -1e9
                            raw = -1e9
                        else:
                            raw = float(s[i].item())
                            score = float(s_norm[i].item())
                        fout.write(json.dumps({
                            "problem_name": pn,
                            "query": q,
                            "group_id": gid,
                            "doc": d,
                            "label": y,
                            "score": score,
                            "llm_score": score,
                            "llm_score_raw": raw,
                        }, ensure_ascii=False) + "\n")

    print(f"wrote predictions: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
