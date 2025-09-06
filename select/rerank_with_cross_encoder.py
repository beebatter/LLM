#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from LLM.models.logic_transformers import TransformerEncoder, TransformerConfig
from LLM.training.train_cross_encoder import CrossHead  # reuse the same head definition
from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets


def _wrap_qd(q: str, d: str, features: Optional[Dict] = None) -> str:
    qn = f"<Q> {normalize_text(q)} </Q>"
    prefix = features_to_prefix(features or {}, PrefixBuckets())
    dn = f"{prefix}<D> {normalize_text(d)} </D>"
    return f"{qn} {dn}"


@dataclass
class Cand:
    id: Optional[int]
    text: str
    features: Optional[Dict]


def load_meta(meta_path: Path) -> Tuple[List[str], List[Optional[Dict]]]:
    texts: List[str] = []
    feats: List[Optional[Dict]] = []
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            texts.append(j.get("text") or j.get("canonical_formula") or "")
            feats.append(j.get("features") or None)
    return texts, feats


def build_model(ckpt_path: Path, spm_override: Optional[Path] = None, device: torch.device = torch.device("cpu")):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])  # type: ignore[arg-type]
    enc = TransformerEncoder(cfg).to(device).eval()
    head = CrossHead(cfg.d_model).to(device).eval()
    enc.load_state_dict(ckpt["encoder_state"])  # type: ignore[arg-type]
    head.load_state_dict(ckpt["head_state"])  # type: ignore[arg-type]
    spm_path = spm_override or Path(ckpt.get("spm_model", ""))
    if not spm_path or not Path(spm_path).exists():
        raise RuntimeError("SentencePiece model path missing; pass --spm explicitly or ensure checkpoint contains 'spm_model'.")
    tok = LogicSentencePiece(str(spm_path))
    return enc, head, tok, cfg


def score_pairs(enc, head, tok: LogicSentencePiece, pairs: List[Tuple[str, Cand]], max_len: int, device: torch.device, batch: int = 256) -> List[float]:
    scores: List[float] = []
    for i in range(0, len(pairs), batch):
        chunk = pairs[i:i+batch]
        ids_list: List[List[int]] = []
        mask_list: List[List[int]] = []
        for q, c in chunk:
            s = _wrap_qd(q, c.text, c.features)
            ids = tok.encode(s)[: max_len]
            ids_list.append(ids)
            mask_list.append([1]*len(ids))
        maxl = max(len(x) for x in ids_list) if ids_list else 0
        ids_t = torch.tensor([x + [0]*(maxl-len(x)) for x in ids_list], dtype=torch.long, device=device)
        mask_t = torch.tensor([m + [0]*(maxl-len(m)) for m in mask_list], dtype=torch.long, device=device)
        with torch.no_grad():
            h = enc(ids_t, mask_t)
            logits = head(h, mask_t)
            prob = torch.sigmoid(logits).detach().float().cpu().numpy().tolist()
            scores.extend(prob)
    return scores


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Rerank candidates with Cross-Encoder")
    ap.add_argument("--input", type=Path, required=True, help="JSONL with {query, candidate_ids|candidates, topk?}")
    ap.add_argument("--index-meta", type=Path, help=".meta.jsonl to map ids->text+features (required if using candidate_ids)")
    ap.add_argument("--model", type=Path, required=True, help="Cross-Encoder checkpoint path")
    ap.add_argument("--spm", type=Path, default=None, help="SentencePiece model path (override)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--topk", type=int, default=64, help="Top-K to emit per query (cap)")
    ap.add_argument("--out", type=Path, required=True, help="Output JSONL path")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, head, tok, cfg = build_model(args.model, args.spm, device)

    meta_texts: List[str] = []
    meta_feats: List[Optional[Dict]] = []
    if args.index_meta is not None:
        meta_texts, meta_feats = load_meta(args.index_meta)

    out_f = open(args.out, "w", encoding="utf-8")
    n_lines = 0
    with open(args.input, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                j = json.loads(line)
            except Exception:
                continue
            q = j.get("query") or j.get("conjecture_text") or j.get("q") or ""
            topk = int(j.get("topk", args.topk))
            cand_ids = j.get("candidate_ids")
            cand_texts = j.get("candidates")
            cands: List[Cand] = []
            if cand_ids is not None:
                if not meta_texts:
                    raise RuntimeError("--index-meta is required when using candidate_ids")
                for cid in cand_ids:
                    try:
                        t = meta_texts[int(cid)]
                        ftr = meta_feats[int(cid)] if meta_feats else None
                    except Exception:
                        t, ftr = "", None
                    cands.append(Cand(id=int(cid), text=t, features=ftr))
            elif cand_texts is not None:
                for t in cand_texts:
                    cands.append(Cand(id=None, text=str(t), features=None))
            else:
                # Nothing to rerank
                out_f.write(json.dumps({"scores": []}, ensure_ascii=False) + "\n")
                continue

            pairs = [(q, c) for c in cands]
            scores = score_pairs(enc, head, tok, pairs, max_len=args.max_len, device=device, batch=args.batch)
            scored = [
                {"id": c.id, "text": c.text, "score": float(s)}
                for c, s in zip(cands, scores)
            ]
            scored.sort(key=lambda x: x["score"], reverse=True)
            out = {"scores": scored[:topk]}
            out_f.write(json.dumps(out, ensure_ascii=False) + "\n")
            n_lines += 1
    out_f.close()
    print(f"wrote {n_lines} lines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
