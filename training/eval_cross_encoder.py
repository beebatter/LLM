#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import torch
from tqdm.auto import tqdm

from LLM.models.logic_transformers import TransformerEncoder, TransformerConfig
from LLM.training.train_cross_encoder import CrossHead
from LLM.training.crossencoder_datasets import build_cross_dataloader


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Eval Cross-Encoder on JSONL (ROC AUC / AP)")
    ap.add_argument("--data", action="append", required=True, help="JSONL path(s) to evaluate")
    ap.add_argument("--model", type=Path, required=True, help="Cross-Encoder checkpoint path")
    ap.add_argument("--spm", type=str, default=None, help="SentencePiece model path (override)")
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON to write metrics")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])  # type: ignore
    enc = TransformerEncoder(cfg).to(device).eval()
    head = CrossHead(cfg.d_model).to(device).eval()
    enc.load_state_dict(ckpt["encoder_state"])  # type: ignore
    head.load_state_dict(ckpt["head_state"])  # type: ignore
    spm_path = args.spm or ckpt.get("spm_model")
    if not spm_path:
        raise RuntimeError("SPM model path missing; pass --spm or ensure checkpoint has 'spm_model'.")

    loader = build_cross_dataloader(args.data, spm_model=str(spm_path), batch_size=args.batch, shuffle=False, max_len=args.max_len)

    ys, ps = [], []
    with torch.no_grad():
        for batch in tqdm(loader, desc="eval", dynamic_ncols=True):
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            h = enc(ids, mask)
            logits = head(h, mask)
            p = logits.sigmoid()
            ys.append(y.cpu())
            ps.append(p.cpu())

    import numpy as np
    from sklearn.metrics import roc_auc_score, average_precision_score
    ycat = torch.cat(ys).numpy()
    pcat = torch.cat(ps).numpy()
    try:
        auc = float(roc_auc_score(ycat, pcat))
    except Exception:
        auc = float("nan")
    try:
        ap = float(average_precision_score(ycat, pcat))
    except Exception:
        ap = float("nan")
    metrics = {"auc": auc, "ap": ap, "n": int(len(ycat))}
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
