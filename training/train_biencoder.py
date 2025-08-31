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

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.training.biencoder_datasets import build_bi_dataloader
from LLM.training.vis_utils import plot_curves, save_metrics_json, plot_hist, plot_bucket_box, save_bucket_recalls, plot_topk_curve

try:
    import sentencepiece as spm  # type: ignore
except Exception:
    spm = None  # will fallback to .vocab file only


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
def eval_recall(model: BiEncoder, loader, device: torch.device, topk: int = 64, neg_per_query: int = 16, max_neg_samples: int = 50000):
    model.eval()
    # First pass: encode all pairs
    all_q, all_d, all_bucket = [], [], []
    for batch in loader:
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        d_ids = batch["d_ids"].to(device)
        d_mask = batch["d_mask"].to(device)
        q = model.encode(q_ids, q_mask, which="q")
        d = model.encode(d_ids, d_mask, which="d")
        all_q.append(q.detach())
        all_d.append(d.detach())
        if "bucket" in batch:
            all_bucket += list(batch["bucket"])  # type: ignore
    Q = torch.cat(all_q, dim=0)
    D = torch.cat(all_d, dim=0)
    Q = nn.functional.normalize(Q, dim=1)
    D = nn.functional.normalize(D, dim=1)

    N = D.size(0)
    K = min(topk, N)
    # Move D to device memory if possible
    D_dev = D
    try:
        if D.device != device:
            D_dev = D.to(device, non_blocking=True)
    except RuntimeError:
        D_dev = D  # fallback

    hits_start = torch.zeros(K, dtype=torch.long)
    pos_scores: List[float] = []
    neg_scores: List[float] = []
    bucket_hits = {}
    bucket_counts = {}

    start = 0
    for q_batch in torch.split(Q, split_size_or_sections=max(1, 1024 if N > 8192 else 4096), dim=0):
        bsz = q_batch.size(0)
        end = start + bsz
        # compute sims against all docs (B x N) without materializing NxN across all batches
        sims_bn = (q_batch.to(D_dev.device) @ D_dev.t())  # on device
        # positive scores: diagonal elements
        row_idx = torch.arange(bsz, device=sims_bn.device)
        col_idx = torch.arange(start, end, device=sims_bn.device)
        pos_scores.extend(sims_bn[row_idx, col_idx].detach().float().cpu().tolist())
        # top-K indices per row
        vals, idx = torch.topk(sims_bn, k=K, dim=1, largest=True)
        # hits@K: find position of gold index
        gold = torch.arange(start, end, device=idx.device).unsqueeze(1)
        eq = (idx == gold)
        # position where gold appears (if any)
        any_hit = eq.any(dim=1)
        if any_hit.any():
            # positions of first True in each row
            pos = torch.where(eq)[1]  # flattened column indices for True entries
            # map to per-row position by grouping; simpler: loop small K
            for r in range(bsz):
                if any_hit[r]:
                    p = int(torch.nonzero(eq[r], as_tuple=False)[0, 0].item())
                    hits_start[p] += 1
        # bucket stats and negative sampling
        if all_bucket:
            for r in range(bsz):
                b = all_bucket[start + r]
                bucket_counts[b] = bucket_counts.get(b, 0) + 1
                # hit at K? reuse any_hit
                hit = bool(any_hit[r].item())
                bucket_hits[b] = bucket_hits.get(b, 0) + (1 if hit else 0)
        # negative sampling: take top negatives (exclude diagonal) limited per query
        if len(neg_scores) < max_neg_samples:
            sims_neg = sims_bn.clone()
            sims_neg[row_idx, col_idx] = float('-inf')
            kneg = min(neg_per_query, N - 1)
            if kneg > 0:
                nv, _ = torch.topk(sims_neg, k=kneg, dim=1, largest=True)
                neg_scores.extend(nv.detach().float().cpu().flatten().tolist())
                if len(neg_scores) > max_neg_samples:
                    neg_scores = neg_scores[:max_neg_samples]
        start = end
        # free
        del sims_bn

    recall_at_k = torch.cumsum(hits_start, dim=0).float().div(max(1, N)).cpu().tolist()
    recallK = float(recall_at_k[K - 1]) if K > 0 else 0.0
    recalls_by_bucket = {k: (float(bucket_hits.get(k, 0)) / max(1, v)) for k, v in bucket_counts.items()}
    return recallK, recall_at_k, recalls_by_bucket, pos_scores, neg_scores, all_bucket


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
    ap.add_argument("--logdir", type=Path, default=Path("/home/ks/Training/logs/biencoder"))
    args = ap.parse_args(argv)

    # vocab size from spm model or .vocab fallback
    vocab_size = None
    # Try SentencePiece model directly first
    if spm is not None:
        try:
            spp = spm.SentencePieceProcessor()
            spp.load(str(args.spm))
            vocab_size = int(spp.get_piece_size())
        except Exception:
            vocab_size = None
    if vocab_size is None:
        spm_vocab = Path(args.spm).with_suffix(".vocab")
        try:
            with open(spm_vocab, "r", encoding="utf-8") as f:
                vocab_size = sum(1 for _ in f)
        except FileNotFoundError:
            raise FileNotFoundError(f"Cannot infer vocab size: neither {spm_vocab} exists nor can the SPM model be loaded. Check --spm path.")

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
    writer = SummaryWriter(log_dir=str(args.logdir))

    limit = args.limit if args.limit and args.limit > 0 else None
    train_loader = build_bi_dataloader(args.train, spm_model=args.spm, batch_size=args.batch, shuffle=True, max_len=args.max_len, limit=limit)
    val_loader = build_bi_dataloader(args.val or args.train, spm_model=args.spm, batch_size=args.batch, shuffle=False, max_len=args.max_len, limit=limit)

    best = -1.0
    history = {"train_loss": [], "val_R@64": []}
    global_step = 0
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
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
            global_step += 1
            if global_step % 10 == 0:
                writer.add_scalar("train/loss_step", float(loss.item()), global_step)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total/max(1,n)):.4f}")
        avg_loss = total / max(1, n)
        writer.add_scalar("train/loss_epoch", avg_loss, ep)
        r64, topk_curve, recalls_by_bucket, pos_scores, neg_scores, buckets = eval_recall(model, val_loader, device, topk=64)
        writer.add_scalar("val/recall@64", r64, ep)
        history["train_loss"].append(avg_loss)
        history["val_R@64"].append(r64)
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
        # per-epoch optional visuals on small slice
        try:
            plot_hist(pos_scores, neg_scores, args.logdir, name=f"score_hist_ep{ep}.png")
            plot_topk_curve(topk_curve, args.logdir, name=f"topk_curve_ep{ep}.png")
            if recalls_by_bucket:
                save_bucket_recalls(recalls_by_bucket, args.logdir, name=f"bucket_recalls_ep{ep}.json")
        except Exception:
            pass
    # End-of-training artifacts
    plot_curves(history, args.logdir, prefix="biencoder")
    save_metrics_json({"best_R@64": best, **history}, args.logdir)
    # final: per-bucket box for diagonal scores
    try:
        # re-eval to get scores and buckets for final plots
        r64, topk_curve, recalls_by_bucket, pos_scores, neg_scores, buckets = eval_recall(model, val_loader, device, topk=64)
        import collections
        by_bucket = collections.defaultdict(list)
        if buckets and len(buckets) == len(pos_scores):
            for b, s in zip(buckets, pos_scores):
                by_bucket[b].append(s)
        if by_bucket:
            plot_bucket_box(by_bucket, args.logdir, name="bucket_box.png")
        plot_topk_curve(topk_curve, args.logdir, name="topk_curve_final.png")
        if recalls_by_bucket:
            save_bucket_recalls(recalls_by_bucket, args.logdir, name="bucket_recalls_final.json")
        plot_hist(pos_scores, neg_scores, args.logdir, name="score_hist_final.png")
    except Exception:
        pass
    writer.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
