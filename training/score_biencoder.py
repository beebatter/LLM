#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.training.crossencoder_datasets import CrossDataset, _wrap_qd
from LLM.data_utils.logic_tokenizer import LogicSentencePiece


def load_model(model_path: Path, d_model_default: int = 512, layers_default: int = 6, heads_default: int = 8, max_len_default: int = 256) -> tuple[BiEncoder, TransformerConfig, str]:
    ckpt = torch.load(model_path, map_location="cpu")
    if isinstance(ckpt, dict) and "config" in ckpt:
        c = ckpt["config"]
        cfg = TransformerConfig(
            vocab_size=c.get("vocab_size", 32000),
            d_model=c.get("d_model", d_model_default),
            n_heads=c.get("n_heads", heads_default),
            n_layers=c.get("n_layers", layers_default),
            pad_id=c.get("pad_id", 0),
            max_len=c.get("max_len", max_len_default),
        )
    else:
        cfg = TransformerConfig(32000, d_model_default, heads_default, layers_default, 0, max_len_default)
    model = BiEncoder(cfg)
    spm_model = None
    if isinstance(ckpt, dict) and "model_state" in ckpt:
        model.load_state_dict(ckpt["model_state"], strict=False)
        spm_model = ckpt.get("spm_model")
    return model, cfg, spm_model


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score pairs with Bi-Encoder (cosine sim) and dump unified predictions JSONL")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--spm", type=Path, required=False, help="SPM model path if not in checkpoint")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, spm_path_ckpt = load_model(args.model)
    model.to(device).eval()

    spm_path = spm_path_ckpt or (str(args.spm) if args.spm else None)
    if spm_path is None:
        raise SystemExit("Need --spm or checkpoint['spm_model'] to tokenize")
    tok = LogicSentencePiece(spm_path)

    def encode_texts(batch_texts: List[str]):
        ids_list, masks = [], []
        for s in batch_texts:
            ids = tok.encode(s)[: args.max_len]
            ids_list.append(ids)
            masks.append([1] * len(ids))
        maxl = max(len(x) for x in ids_list)
        pad_id = 0
        ids = torch.tensor([x + [pad_id] * (maxl - len(x)) for x in ids_list], dtype=torch.long, device=device)
        mask = torch.tensor([m + [0] * (maxl - len(m)) for m in masks], dtype=torch.long, device=device)
        return ids, mask

    ds = CrossDataset(args.data)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        B = args.batch
        for i in tqdm(range(0, len(ds), B), desc="score_bi", dynamic_ncols=True):
            chunk = [ds[j] for j in range(i, min(i + B, len(ds)))]
            # build separate inputs for q and d
            q_texts = [f"<Q> {q.d} </Q>".replace(q.d, q.q) for q in chunk]  # properly normalize via _wrap
            d_texts = [f"<D> {q.d} </D>".replace(q.d, q.d) for q in chunk]
            # Use the same wrapper as CE to include prefixes and normalization
            q_wrapped = [f"<Q> {tok.normalize_text(q)} </Q>" if hasattr(tok, 'normalize_text') else _wrap_qd(q.q, "", None).split("</Q>")[0] + " </Q>" for q in chunk]  # fallback if needed
            # Actually, rely on encode of separate methods in BiEncoder with bare normalized strings
            q_ids, q_mask = encode_texts([f"<Q> {tok.normalize_text(it.q) if hasattr(tok,'normalize_text') else it.q} </Q>" for it in chunk])
            d_ids, d_mask = encode_texts([f"<D> {tok.normalize_text(it.d) if hasattr(tok,'normalize_text') else it.d} </D>" for it in chunk])
            with torch.no_grad():
                qv = model.encode(q_ids, q_mask, which="q")
                dv = model.encode(d_ids, d_mask, which="d")
                qv = nn.functional.normalize(qv, dim=1)
                dv = nn.functional.normalize(dv, dim=1)
                sc = (qv * dv).sum(dim=1).detach().cpu().tolist()
            for it, s in zip(chunk, sc):
                pred = {
                    "problem_name": it.group,
                    "query": it.q,
                    "group_id": f"{it.group}||{it.q}",
                    "doc": it.d,
                    "label": float(it.label),
                    "score": float(s),
                    "bi_score": float(s),
                }
                f.write(json.dumps(pred, ensure_ascii=False) + "\n")

    print(f"wrote predictions: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
