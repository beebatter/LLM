#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import DataLoader

from LLM.models.logic_transformers import ClauseScorer, TransformerConfig
from LLM.training.logic_datasets import build_dataloader


def train_epoch(model: nn.Module, loader: DataLoader, optim: AdamW, device: torch.device) -> float:
    model.train()
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        optim.zero_grad(set_to_none=True)
        scores = model(ids, mask)
        loss = loss_fn(scores, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optim.step()
        total += float(loss.item()) * ids.size(0)
        n += ids.size(0)
    return total / max(1, n)


@torch.no_grad()
def eval_epoch(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    model.eval()
    loss_fn = nn.MSELoss()
    total = 0.0
    n = 0
    for batch in loader:
        ids = batch["input_ids"].to(device)
        mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)
        scores = model(ids, mask)
        loss = loss_fn(scores, labels)
        total += float(loss.item()) * ids.size(0)
        n += ids.size(0)
    return total / max(1, n)


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train clause-only Transformer scorer for iProver PSM")
    ap.add_argument("--train", action="append", required=True, help="Path(s) to training JSONL files")
    ap.add_argument("--val", action="append", help="Path(s) to validation JSONL files")
    ap.add_argument("--spm", type=str, default="/home/ks/Training/models/spm_logic.model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--max-len", type=int, default=512)
    ap.add_argument("--d-model", type=int, default=256)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--save", type=Path, default=Path("/home/ks/Training/models/clause_scorer.pt"))
    args = ap.parse_args(argv)

    # Load vocab size from SentencePiece vocab file (model path -> .vocab path)
    spm_model = Path(args.spm)
    spm_vocab = spm_model.with_suffix(".vocab")
    if not spm_vocab.exists():
        raise FileNotFoundError(f"SPM vocab not found: {spm_vocab}")
    with open(spm_vocab, "r", encoding="utf-8") as f:
        vocab_size = sum(1 for _ in f)

    cfg = TransformerConfig(
        vocab_size=vocab_size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        pad_id=0,
    )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ClauseScorer(cfg).to(device)
    optimizer = AdamW(model.parameters(), lr=args.lr)

    train_loader = build_dataloader(args.train, spm_model=str(spm_model), batch_size=args.batch, shuffle=True, max_len=args.max_len)
    val_loader = build_dataloader(args.val or args.train, spm_model=str(spm_model), batch_size=args.batch, shuffle=False, max_len=args.max_len)

    best = float("inf")
    for ep in range(1, args.epochs + 1):
        tr_loss = train_epoch(model, train_loader, optimizer, device)
        va_loss = eval_epoch(model, val_loader, device)
        print(f"epoch {ep}: train_loss={tr_loss:.4f} val_loss={va_loss:.4f}")
        if va_loss < best:
            best = va_loss
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "spm_model": str(spm_model),
            }, args.save)
            print(f"saved: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
