#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from LLM.models.logic_transformers import TransformerEncoder, TransformerConfig
from LLM.training.crossencoder_datasets import build_cross_dataloader, build_cross_grouped_dataloader
from LLM.training.vis_utils import plot_curves, save_metrics_json, plot_hist

try:
    import sentencepiece as spm  # type: ignore
except Exception:
    spm = None


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
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--save", type=Path, default=Path("/home/ks/Training/models/cross_encoder_best.pt"))
    ap.add_argument("--logdir", type=Path, default=Path("/home/ks/Training/logs/crossencoder"))
    # loss & grouping options
    ap.add_argument("--loss", type=str, choices=["bce", "listwise"], default="bce")
    ap.add_argument("--group-size", type=int, default=16, help="Max candidates per query/group in listwise mode")
    ap.add_argument("--groups-per-batch", type=int, default=4, help="Groups per batch in listwise mode")
    args = ap.parse_args(argv)

    # robust vocab size from SPM model or .vocab fallback
    vocab_size = None
    if spm is not None:
        try:
            spp = spm.SentencePieceProcessor()
            spp.load(str(args.spm))
            vocab_size = int(spp.get_piece_size())
        except Exception:
            vocab_size = None
    if vocab_size is None:
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

    if args.loss == "listwise":
        train_loader = build_cross_grouped_dataloader(
            args.train, spm_model=args.spm, group_size=args.group_size, groups_per_batch=args.groups_per_batch, max_len=args.max_len, shuffle=True
        )
    else:
        train_loader = build_cross_dataloader(args.train, spm_model=args.spm, batch_size=args.batch, shuffle=True, max_len=args.max_len)
    val_loader = build_cross_dataloader(args.val or args.train, spm_model=args.spm, batch_size=args.batch, shuffle=False, max_len=args.max_len)
    writer = SummaryWriter(log_dir=str(args.logdir))
    history = {"train_loss": [], "val_auc": []}

    best_auc = -1.0
    from sklearn.metrics import roc_auc_score

    if len(train_loader) == 0:
        print("Empty train loader: check input paths and schema (need query/text fields).", flush=True)
        return 1

    for ep in range(1, args.epochs + 1):
        enc.train(); head.train()
        total, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            opt.zero_grad(set_to_none=True)
            h = enc(ids, mask)
            logits = head(h, mask)
            if args.loss == "listwise":
                # group-wise softmax CE over positives
                group_ids = batch.get("group_ids")
                if group_ids is None:
                    # fallback to BCE if grouping missing
                    loss = loss_fn(logits, y)
                    eff_bs = ids.size(0)
                else:
                    gid = group_ids.to(device)
                    loss_sum = 0.0
                    groups = gid.unique().tolist()
                    eff_groups = 0
                    for g in groups:
                        m = (gid == g)
                        lg = logits[m]
                        yg = y[m]
                        if (yg > 0.5).sum() == 0:
                            continue  # skip groups with no positives
                        logp = lg - torch.logsumexp(lg, dim=0)
                        loss_g = -logp[yg > 0.5].mean()
                        loss_sum = loss_sum + loss_g
                        eff_groups += 1
                    loss = loss_sum / max(1, eff_groups)
                    eff_bs = eff_groups
            else:
                loss = loss_fn(logits, y)
                eff_bs = ids.size(0)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            total += float(loss.item()) * eff_bs
            n += eff_bs
            writer.add_scalar("train/loss_step", float(loss.item()), (ep-1)*len(train_loader)+pbar.n)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total/max(1,n)):.4f}")
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
        if len(ys) == 0:
            print("Warning: empty val loader; skipping AUC this epoch.")
            auc = float('nan')
        else:
            ycat = torch.cat(ys).numpy()
            pcat = torch.cat(ps).numpy()
            try:
                auc = float(roc_auc_score(ycat, pcat))
            except Exception:
                auc = 0.5
        avg_loss = total / max(1, n)
        writer.add_scalar("train/loss_epoch", avg_loss, ep)
        writer.add_scalar("val/auc", auc, ep)
        history["train_loss"].append(avg_loss)
        history["val_auc"].append(auc)
        print(f"epoch {ep}: train_loss={avg_loss:.4f} val_auc={auc:.4f}")
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

    # end-of-training artifacts
    plot_curves(history, args.logdir, prefix="crossencoder")
    # optional: score histogram on a small slice of val
    try:
        import torch as _T
        enc.eval(); head.eval()
        pos_scores, neg_scores = [], []
        with torch.no_grad():
            for i, batch in enumerate(val_loader):
                if i >= 20:  # small slice
                    break
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                h = enc(ids, mask)
                logits = head(h, mask)
                p = logits.sigmoid()
                pos_scores += p[y > 0.5].cpu().tolist()
                neg_scores += p[y <= 0.5].cpu().tolist()
        plot_hist(pos_scores, neg_scores, args.logdir, name="score_hist.png")
    except Exception:
        pass
    save_metrics_json({"best_auc": best_auc, **history}, args.logdir)
    writer.close()

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
