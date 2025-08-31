#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.training.biencoder_datasets import build_bi_dataloader


class InfoNCE(nn.Module):
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, q: torch.Tensor, d: torch.Tensor) -> torch.Tensor:
        # q, d: [B, D]
        q = nn.functional.normalize(q, dim=1)
        d = nn.functional.normalize(d, dim=1)
        logits = (q @ d.t()) / self.t  # [B, B]
        labels = torch.arange(q.size(0), device=q.device)
        return nn.functional.cross_entropy(logits, labels)


@torch.no_grad()
def eval_recall(model: BiEncoder, loader, device: torch.device, topk: int = 64) -> float:
    model.eval()
    all_q, all_d = [], []
    for batch in loader:
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        d_ids = batch["d_ids"].to(device)
        d_mask = batch["d_mask"].to(device)
        q = model.encode(q_ids, q_mask, which="q")
        d = model.encode(d_ids, d_mask, which="d")
        all_q.append(q)
        all_d.append(d)
    Q = torch.cat(all_q, dim=0)
    D = torch.cat(all_d, dim=0)
    Q = nn.functional.normalize(Q, dim=1)
    D = nn.functional.normalize(D, dim=1)
    sims = Q @ D.t()  # [N, N]
    vals, idx = sims.topk(k=min(topk, D.size(0)), dim=1, largest=True)
    # hit if the gold doc (same row index) is in topk
    hits = (idx == torch.arange(D.size(0), device=idx.device).unsqueeze(1)).any(dim=1).float()
    return float(hits.mean().item())


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train Bi-Encoder (InfoNCE) for SELECT")
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--val", action="append")
    ap.add_argument("--spm", type=str, default="/home/ks/Training/models/spm_logic.model")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--save", type=Path, default=Path("/home/ks/Training/models/biencoder_best.pt"))
    ap.add_argument("--limit", type=int, default=0, help="Limit number of training examples (for quick sanity runs)")
    args = ap.parse_args(argv)

    # vocab size from spm vocab
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
    model = BiEncoder(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)
    loss_fn = InfoNCE(temperature=0.07)

    limit = args.limit if args.limit and args.limit > 0 else None
    train_loader = build_bi_dataloader(args.train, spm_model=args.spm, batch_size=args.batch, shuffle=True, max_len=args.max_len, limit=limit)
    val_loader = build_bi_dataloader(args.val or args.train, spm_model=args.spm, batch_size=args.batch, shuffle=False, max_len=args.max_len, limit=limit)

    best = -1.0
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        for batch in train_loader:
            q_ids = batch["q_ids"].to(device)
            q_mask = batch["q_mask"].to(device)
            d_ids = batch["d_ids"].to(device)
            d_mask = batch["d_mask"].to(device)
            opt.zero_grad(set_to_none=True)
            q = model.encode(q_ids, q_mask, which="q")
            d = model.encode(d_ids, d_mask, which="d")
            loss = loss_fn(q, d)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * q_ids.size(0)
            n += q_ids.size(0)
        r64 = eval_recall(model, val_loader, device, topk=64)
        print(f"epoch {ep}: train_loss={total/max(1,n):.4f} val_R@64={r64:.4f}")
        if r64 > best:
            best = r64
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "model_state": model.state_dict(),
                "config": cfg.__dict__,
                "spm_model": args.spm,
            }, args.save)
            print(f"saved: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
