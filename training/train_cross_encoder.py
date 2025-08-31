#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW

from LLM.models.logic_transformers import TransformerEncoder, TransformerConfig
from LLM.training.crossencoder_datasets import build_cross_dataloader


class CrossHead(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d((1, d_model))
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: [B,L,D], mask: [B,L]
        lengths = mask.sum(dim=1).clamp_min(1).unsqueeze(1)
        summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        z = summed / lengths
        return self.proj(z).squeeze(-1)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train Cross-Encoder for RERANK")
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--val", action="append")
    ap.add_argument("--spm", type=str, default="/home/ks/Training/models/spm_logic.model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--save", type=Path, default=Path("/home/ks/Training/models/cross_encoder_best.pt"))
    args = ap.parse_args(argv)

    spm_vocab = Path(args.spm).with_suffix(".vocab")
    with open(spm_vocab, "r", encoding="utf-8") as f:
        vocab_size = sum(1 for _ in f)

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        pad_id=0,
        max_len=args.max_len,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc = TransformerEncoder(cfg).to(device)
    head = CrossHead(cfg.d_model).to(device)
    params = list(enc.parameters()) + list(head.parameters())
    opt = AdamW(params, lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    train_loader = build_cross_dataloader(args.train, spm_model=args.spm, batch_size=args.batch, shuffle=True, max_len=args.max_len)
    val_loader = build_cross_dataloader(args.val or args.train, spm_model=args.spm, batch_size=args.batch, shuffle=False, max_len=args.max_len)

    best_auc = -1.0
    from sklearn.metrics import roc_auc_score

    for ep in range(1, args.epochs + 1):
        enc.train(); head.train()
        total, n = 0.0, 0
        for batch in train_loader:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            h = enc(ids, mask)
            logits = head(h, mask)
            loss = loss_fn(logits, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss.item()) * ids.size(0)
            n += ids.size(0)
        # eval
        enc.eval(); head.eval()
        ys, ps = [], []
        with torch.no_grad():
            for batch in val_loader:
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                h = enc(ids, mask)
                logits = head(h, mask)
                ys.append(y.cpu())
                ps.append(logits.sigmoid().cpu())
        import numpy as np
        ycat = torch.cat(ys).numpy()
        pcat = torch.cat(ps).numpy()
        try:
            auc = float(roc_auc_score(ycat, pcat))
        except Exception:
            auc = 0.5
        print(f"epoch {ep}: train_loss={total/max(1,n):.4f} val_auc={auc:.4f}")
        if auc > best_auc:
            best_auc = auc
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "encoder_state": enc.state_dict(),
                "head_state": head.state_dict(),
                "config": cfg.__dict__,
                "spm_model": args.spm,
            }, args.save)
            print(f"saved: {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
