#!/usr/bin/env python3
"""
Fuse Cross-Encoder and Bi-Encoder scores to produce soft labels for listwise windows.

Input: listwise JSONL (from make_listwise_chunks.py), each line with fields:
 - problem_name, K, ids, input, target_json, target_scores

We will parse the prompt to extract conjecture and candidate TEXT/TAGS if needed (but prefer using ids + an index meta),
then score each candidate with:
 - Cross-Encoder: TransformerEncoder + CrossHead (BCE logits→sigmoid)
 - Bi-Encoder: dot(Q, D) on normalized embeddings

Output: write a new JSONL file with the same fields but target_scores replaced by softmax(lambda_cross*ce + lambda_bi*bi),
and target_json updated accordingly.

Usage example:
python -m LLM.scripts.make_listwise_targets_from_teacher \
  --in-listwise /root/Training/datasets/train_listwise.rebuilt.jsonl \
  --out         /root/Training/datasets/train_listwise.teacher.jsonl \
  --cross-ckpt  /root/Training/models/cross_encoder_best.pt \
  --bi-ckpt     /root/Training/models/biencoder_best.pt \
  --spm         /root/Training/models/spm_logic_24k.model \
  --lambda-cross 1.0 --lambda-bi 0.3 --max-len 256 --batch 256
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets
from LLM.models.logic_transformers import TransformerConfig, TransformerEncoder, BiEncoder
from LLM.training.train_cross_encoder import CrossHead


def _parse_prompt_for_q_and_candidates(prompt: str) -> Tuple[str, List[Tuple[int, str, str]]]:
    """Best-effort extraction aligning make_listwise_chunks prompt schema.
    Returns: (conjecture_text, [(id, text, tags_str), ...])
    """
    q = ""
    cands: List[Tuple[int, str, str]] = []
    try:
        # [CONJECTURE]\n{conj}\n ... then candidates blocks "- ID <id>\n- TEXT: ...\n- TAGS: ..."
        # Extract conj between "[CONJECTURE]\n" and "\n[CANDIDATES]"
        start = prompt.find("[CONJECTURE]\n")
        if start >= 0:
            start += len("[CONJECTURE]\n")
            end = prompt.find("\n[CANDIDATES]", start)
            if end > start:
                q = prompt[start:end].strip()
        # Extract candidate lines
        body_start = prompt.find("（以下为候选）")
        if body_start < 0:
            body_start = prompt.find("(以下为候选)")
        if body_start >= 0:
            body = prompt[body_start:]
        else:
            body = prompt
        # Split by "- ID"
        parts = body.split("- ID ")
        for part in parts[1:]:
            try:
                id_end = part.find("\n")
                cid = int(part[:id_end].strip())
                t_pos = part.find("- TEXT:")
                g_pos = part.find("- TAGS:")
                text = part[t_pos + 7:g_pos].strip() if (t_pos >= 0 and g_pos > t_pos) else ""
                tags = part[g_pos + 7:].splitlines()[0].strip() if g_pos >= 0 else ""
                cands.append((cid, text, tags))
            except Exception:
                continue
    except Exception:
        pass
    return q, cands


def build_cross_model(ckpt_path: str, spm_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])  # type: ignore[arg-type]
    enc = TransformerEncoder(cfg).to(device).eval()
    head = CrossHead(cfg.d_model).to(device).eval()
    enc.load_state_dict(ckpt["encoder_state"])  # type: ignore[arg-type]
    head.load_state_dict(ckpt["head_state"])  # type: ignore[arg-type]
    tok = LogicSentencePiece(spm_path)
    return enc, head, tok, cfg


def build_bi_model(ckpt_path: str, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])  # type: ignore[arg-type]
    model = BiEncoder(cfg).to(device).eval()
    model.load_state_dict(ckpt["model_state"])  # type: ignore[arg-type]
    return model, cfg


@torch.no_grad()
def score_cross(enc, head, tok: LogicSentencePiece, q: str, cands: List[Tuple[int, str, str]], max_len: int, batch: int, device: torch.device) -> List[float]:
    scores: List[float] = []
    qn = f"<Q> {normalize_text(q)} </Q>"
    for i in range(0, len(cands), batch):
        chunk = cands[i:i+batch]
        ids_list: List[List[int]] = []
        mask_list: List[List[int]] = []
        for (_, text, tags) in chunk:
            dn = f"{tags}<D> {normalize_text(text)} </D>"
            s = f"{qn} {dn}"
            ids = tok.encode(s)[: max_len]
            ids_list.append(ids)
            mask_list.append([1] * len(ids))
        maxl = max(len(x) for x in ids_list) if ids_list else 0
        ids_t = torch.tensor([x + [0]*(maxl-len(x)) for x in ids_list], dtype=torch.long, device=device)
        mask_t = torch.tensor([m + [0]*(maxl-len(m)) for m in mask_list], dtype=torch.long, device=device)
        h = enc(ids_t, mask_t)
        logits = head(h, mask_t)
        prob = torch.sigmoid(logits).detach().float().cpu().numpy().tolist()
        scores.extend(prob)
    return scores


@torch.no_grad()
def score_bi(model: BiEncoder, tok: LogicSentencePiece, q: str, cands: List[Tuple[int, str, str]], max_len: int, batch: int, device: torch.device) -> List[float]:
    # Encode q once
    def wrap_q(text: str) -> str:
        return f"<Q> {normalize_text(text)} </Q>"
    def wrap_d(text: str, tags: str) -> str:
        return f"{tags}<D> {normalize_text(text)} </D>"
    q_ids = tok.encode(wrap_q(q))[: max_len]
    q_ids_t = torch.tensor([q_ids], dtype=torch.long, device=device)
    q_mask_t = torch.tensor([[1]*len(q_ids)], dtype=torch.long, device=device)
    q_vec = model.encode(q_ids_t, q_mask_t, which="q")  # [1, D]
    q_vec = torch.nn.functional.normalize(q_vec, dim=1)
    scores: List[float] = []
    # Encode docs in batches
    for i in range(0, len(cands), batch):
        chunk = cands[i:i+batch]
        ids_list: List[List[int]] = []
        mask_list: List[List[int]] = []
        for (_, text, tags) in chunk:
            ids = tok.encode(wrap_d(text, tags))[: max_len]
            ids_list.append(ids)
            mask_list.append([1]*len(ids))
        maxl = max(len(x) for x in ids_list) if ids_list else 0
        d_ids = torch.tensor([x + [0]*(maxl-len(x)) for x in ids_list], dtype=torch.long, device=device)
        d_mask = torch.tensor([m + [0]*(maxl-len(m)) for m in mask_list], dtype=torch.long, device=device)
        d_vec = model.encode(d_ids, d_mask, which="d")  # [N, D]
        d_vec = torch.nn.functional.normalize(d_vec, dim=1)
        sim = torch.matmul(q_vec, d_vec.t()).squeeze(0)  # [N]
        scores.extend(sim.detach().float().cpu().numpy().tolist())
    return scores


def softmax(x: np.ndarray, tau: float = 1.0) -> np.ndarray:
    x = (x - x.max()) / max(tau, 1e-6)
    e = np.exp(x)
    s = e.sum()
    if s <= 0:
        return np.ones_like(x) / len(x)
    return e / s


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Make listwise targets from fused teacher scores (cross+bi)")
    ap.add_argument("--in-listwise", required=True, help="Input listwise JSONL")
    ap.add_argument("--out", required=True, help="Output JSONL with replaced target_scores")
    ap.add_argument("--cross-ckpt", required=True, help="Cross-Encoder checkpoint path")
    ap.add_argument("--bi-ckpt", required=True, help="Bi-Encoder checkpoint path")
    ap.add_argument("--spm", required=True, help="SentencePiece model path")
    ap.add_argument("--lambda-cross", type=float, default=1.0)
    ap.add_argument("--lambda-bi", type=float, default=0.3)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--tau", type=float, default=1.0, help="temperature for softmax")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # Build models
    enc, head, tok_ce, _ = build_cross_model(args.cross_ckpt, args.spm, device)
    bi, _ = build_bi_model(args.bi_ckpt, device)
    tok_bi = LogicSentencePiece(args.spm)

    n_in, n_out = 0, 0
    with open(args.in_listwise, "r", encoding="utf-8", errors="ignore") as f, open(args.out, "w", encoding="utf-8") as w:
        for line in f:
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            n_in += 1
            prompt = j.get("input") or ""
            K = int(j.get("K") or 0)
            if not prompt or K <= 0:
                continue
            q, cands = _parse_prompt_for_q_and_candidates(prompt)
            if not q or not cands or len(cands) != K:
                continue
            # score
            s_cross = np.array(score_cross(enc, head, tok_ce, q, cands, args.max_len, args.batch, device), dtype=float)
            s_bi = np.array(score_bi(bi, tok_bi, q, cands, args.max_len, args.batch, device), dtype=float)
            if s_cross.shape[0] != K or s_bi.shape[0] != K:
                continue
            fused = args.lambda_cross * s_cross + args.lambda_bi * s_bi
            tgt = softmax(fused, tau=args.tau).tolist()
            j["target_scores"] = tgt
            j["target_json"] = json.dumps({"scores": tgt}, ensure_ascii=False)
            w.write(json.dumps(j, ensure_ascii=False) + "\n")
            n_out += 1

    print(f"processed {n_in} lines, wrote {n_out} with teacher targets to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
