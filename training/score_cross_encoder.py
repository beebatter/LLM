#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from LLM.models.logic_transformers import TransformerEncoder, TransformerConfig
from LLM.training.crossencoder_datasets import CrossDataset, _wrap_qd


class CrossHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        lengths = mask.sum(dim=1).clamp_min(1).unsqueeze(1)
        summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        z = summed / lengths
        return self.proj(z).squeeze(-1)


def load_model(model_path: Path, d_model_default: int = 512, layers_default: int = 6, heads_default: int = 8, max_len_default: int = 256) -> tuple[TransformerEncoder, CrossHead, TransformerConfig]:
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
    enc = TransformerEncoder(cfg)
    head = CrossHead(cfg.d_model)
    if isinstance(ckpt, dict) and "encoder_state" in ckpt and "head_state" in ckpt:
        enc.load_state_dict(ckpt["encoder_state"], strict=False)
        head.load_state_dict(ckpt["head_state"], strict=False)
    return enc, head, cfg


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Score pairs with Cross-Encoder and dump ce_scored JSONL")
    ap.add_argument("--data", action="append", required=True)
    ap.add_argument("--model", type=Path, required=True)
    ap.add_argument("--spm", type=Path, required=False, help="Unused here; model cfg holds vocab size")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc, head, cfg = load_model(args.model)
    enc.to(device).eval(); head.to(device).eval()

    ds = CrossDataset(args.data)
    tok = None
    # Use the same tokenizer as training datapath
    from LLM.data_utils.logic_tokenizer import LogicSentencePiece
    # Try to recover SPM model path from checkpoint if stored
    spm_model_path = None
    try:
        ckpt = torch.load(args.model, map_location="cpu")
        spm_model_path = ckpt.get("spm_model") if isinstance(ckpt, dict) else None
    except Exception:
        spm_model_path = None
    if spm_model_path is None and args.spm is not None:
        spm_model_path = str(args.spm)
    if spm_model_path is None:
        raise SystemExit("Need --spm or checkpoint['spm_model'] to tokenize")
    tok = LogicSentencePiece(spm_model_path)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    fout = args.out.open("w", encoding="utf-8")
    try:
        B = args.batch
        for i in tqdm(range(0, len(ds), B), desc="score", dynamic_ncols=True):
            chunk = [ds[j] for j in range(i, min(i + B, len(ds)))]
            ids_list, masks = [], []
            for it in chunk:
                s = _wrap_qd(it.q, it.d, it.features)
                ids = tok.encode(s)[: args.max_len]
                ids_list.append(ids)
                masks.append([1] * len(ids))
            maxl = max(len(x) for x in ids_list)
            pad_id = 0
            ids = torch.tensor([x + [pad_id] * (maxl - len(x)) for x in ids_list], dtype=torch.long, device=device)
            mask = torch.tensor([m + [0] * (maxl - len(m)) for m in masks], dtype=torch.long, device=device)
            with torch.no_grad():
                h = enc(ids, mask)
                sc = head(h, mask).sigmoid().detach().cpu().tolist()
            for it, s in zip(chunk, sc):
                obj = {
                    "text_a": it.q,
                    "text_b": it.d,
                    "label": float(it.label),
                    "ce_score": float(s),
                    "problem_name": it.group,
                }
                if it.features is not None:
                    obj["features"] = it.features
                fout.write(json.dumps(obj, ensure_ascii=False) + "\n")
    finally:
        fout.close()

    print(f"wrote: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
