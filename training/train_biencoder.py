#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.tensorboard import SummaryWriter
from tqdm.auto import tqdm

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.training.biencoder_datasets import build_bi_dataloader, build_bi_dataloader_grouped
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


class SupConLoss(nn.Module):
    """
    Supervised contrastive loss for bi-encoder with many positives per query in-batch.
    positives_mask: [B, B] with True where (q_i, d_j) is a positive pair; exclude i==j only if it's not positive.
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.t = temperature

    def forward(self, q: torch.Tensor, d: torch.Tensor, positives_mask: torch.Tensor) -> torch.Tensor:
        # q,d normalized
        q = nn.functional.normalize(q, dim=1)
        d = nn.functional.normalize(d, dim=1)
        logits = (q @ d.t()) / self.t  # [B,B]
        # For each row i, compute log prob mass over its positives j
        # Mask invalid positions (no positives -> skip row)
        pos_mask = positives_mask.float()
        # subtract max for stability
        logits = logits - logits.max(dim=1, keepdim=True).values
        exp_logits = torch.exp(logits)
        denom = exp_logits.sum(dim=1, keepdim=True) + 1e-9
        pos_sum = (exp_logits * pos_mask).sum(dim=1)
        # Only keep rows with at least one positive
        valid = pos_mask.sum(dim=1) > 0
        if valid.any():
            log_prob = torch.log(pos_sum[valid] / denom[valid].squeeze(1) + 1e-9)
            loss = -log_prob.mean()
        else:
            loss = torch.tensor(0.0, device=q.device, requires_grad=True)
        return loss


@torch.no_grad()
def eval_recall(model: BiEncoder, loader, device: torch.device, topk: int = 64):
    """
    True Recall@K per problem_name:
    - collect unique docs (by d_hash) and their embeddings;
    - for each problem, collect all its positive docs' hashes;
    - for each problem's query embedding (take one per sample, average if multiple),
      retrieve over the global doc index and check if any of its positive hashes appears in top-K.
    """
    model.eval()
    all_q, all_d = [], []
    problems: List[str] = []
    d_hashes: List[str] = []
    buckets: List[str] = []
    all_labels: list[float] = []
    for batch in loader:
        q_ids = batch["q_ids"].to(device)
        q_mask = batch["q_mask"].to(device)
        d_ids = batch["d_ids"].to(device)
        d_mask = batch["d_mask"].to(device)
        q = model.encode(q_ids, q_mask, which="q")
        d = model.encode(d_ids, d_mask, which="d")
        all_q.append(q.detach().cpu())
        all_d.append(d.detach().cpu())
        problems.extend(batch.get("problem", ["<UNK>"] * q.size(0)))
        d_hashes.extend(batch.get("d_hash", [""] * d.size(0)))
        lbs = batch.get("labels")
        if lbs is not None:
            all_labels.extend([float(x) for x in lbs.tolist()])
        else:
            all_labels.extend([1.0] * q.size(0))
        if "bucket" in batch:
            buckets += list(batch["bucket"])  # type: ignore
    Q = torch.cat(all_q, dim=0)
    D = torch.cat(all_d, dim=0)
    Q = nn.functional.normalize(Q, dim=1)
    D = nn.functional.normalize(D, dim=1)

    # Build unique doc index
    uniq_map = {}
    uniq_d_vecs = []
    for i, h in enumerate(d_hashes):
        if h not in uniq_map:
            uniq_map[h] = len(uniq_d_vecs)
            uniq_d_vecs.append(D[i])
    Dmat = torch.stack(uniq_d_vecs, dim=0)  # [Nd, D]

    # positives per problem (set of doc hashes)
    pos_by_prob = {}
    for i, prob in enumerate(problems):
        h = d_hashes[i]
        if h == "" or all_labels[i] < 0.5:
            continue
        s = pos_by_prob.get(prob)
        if s is None:
            s = set()
            pos_by_prob[prob] = s
        s.add(h)

    # query embedding per problem: average of all q for that problem
    import collections
    q_sum = collections.defaultdict(lambda: torch.zeros(Q.size(1)))
    q_cnt = collections.defaultdict(int)
    for i, prob in enumerate(problems):
        q_sum[prob] = q_sum[prob] + Q[i]
        q_cnt[prob] += 1
    q_by_prob = {k: (v / max(1, q_cnt[k])) for k, v in q_sum.items()}

    Nd = Dmat.size(0)
    K = min(topk, Nd)
    hits = 0
    total = 0
    topk_curve = [0] * K
    # brute-force retrieval per problem
    Dt = Dmat.t()  # [D, Nd]
    for prob, qv in q_by_prob.items():
        # skip problems that have zero positives (shouldn't happen if data prepared)
        pos_hashes = pos_by_prob.get(prob, set())
        if not pos_hashes:
            continue
        sims = (qv @ Dt).squeeze(0)  # [Nd]
        vals, idx = torch.topk(sims, k=K, dim=0, largest=True)
        # check hit
        ranked_hashes = [list(uniq_map.keys())[j] for j in idx.tolist()]
        hit_pos = -1
        for rank, h in enumerate(ranked_hashes):
            if h in pos_hashes:
                hit_pos = rank
                break
        total += 1
        if hit_pos >= 0:
            hits += 1
            for r in range(hit_pos, K):
                topk_curve[r] += 1
    if total == 0:
        recallK = 0.0
        topk_curve = [0.0] * K
    else:
        recallK = hits / total
        topk_curve = [v / total for v in topk_curve]
    # Pairwise (diagonal) scores for histogram: use labeled pairs in the dataset
    diag = (Q * D).sum(dim=1)
    pos_scores, neg_scores = [], []
    for i, lbl in enumerate(all_labels):
        if lbl >= 0.5:
            pos_scores.append(float(diag[i].item()))
        else:
            neg_scores.append(float(diag[i].item()))
    # no bucket breakdown in this path (could add via majority bucket per problem)
    recalls_by_bucket = {}
    return recallK, topk_curve, recalls_by_bucket, pos_scores, neg_scores, list(pos_by_prob.keys())


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Train Bi-Encoder (InfoNCE) for SELECT")
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--val", action="append")
    ap.add_argument("--spm", type=str, default="/home/ks/Training/models/spm_logic.model")
    ap.add_argument("--epochs", type=int, default=6)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--save", type=Path, default=Path("/home/ks/Training/models/biencoder_best.pt"))
    ap.add_argument("--limit", type=int, default=0, help="Limit number of training examples (for quick sanity runs)")
    ap.add_argument("--logdir", type=Path, default=Path("/home/ks/Training/logs/biencoder"))
    ap.add_argument("--loss", type=str, default="info_nce", choices=["info_nce", "supcon"], help="Training loss: InfoNCE (requires positives aligned by row) or supervised contrastive (uses labels in-batch)")
    ap.add_argument("--temperature", type=float, default=0.07)
    ap.add_argument("--grouped", action="store_true", help="Use problem-grouped batch sampler (recommended for supcon)")
    ap.add_argument("--warmup-steps", type=int, default=1000, help="Linear warmup steps for LR scheduler (0 to disable)")
    ap.add_argument("--scheduler", type=str, default="cosine", choices=["none", "cosine"], help="LR scheduler type")
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
    opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    loss_fn = InfoNCE(temperature=args.temperature) if args.loss == "info_nce" else SupConLoss(temperature=args.temperature)
    writer = SummaryWriter(log_dir=str(args.logdir))

    limit = args.limit if args.limit and args.limit > 0 else None
    if args.grouped or args.loss == "supcon":
        train_loader = build_bi_dataloader_grouped(args.train, spm_model=args.spm, batch_size=args.batch, max_len=args.max_len, limit=limit)
    else:
        train_loader = build_bi_dataloader(args.train, spm_model=args.spm, batch_size=args.batch, shuffle=True, max_len=args.max_len, limit=limit)
    val_loader = build_bi_dataloader(args.val or args.train, spm_model=args.spm, batch_size=args.batch, shuffle=False, max_len=args.max_len, limit=limit)

    # Prepare scheduler after we know total steps
    best = -1.0
    history = {"train_loss": [], "val_R@64": []}
    global_step = 0
    # Build loaders first to compute total training steps for scheduler
    # (re-order creation moved earlier to compute scheduler)
    # Done above
    total_train_steps = (len(train_loader) * args.epochs) if hasattr(train_loader, "__len__") else 0
    if args.scheduler == "cosine" and total_train_steps > 0:
        warmup = max(0, args.warmup_steps)
        def lr_lambda(step: int):
            if step < warmup:
                return float(step) / float(max(1, warmup))
            progress = float(step - warmup) / float(max(1, total_train_steps - warmup))
            # cosine decay to 0
            import math
            return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))
        scheduler = LambdaLR(opt, lr_lambda)
    else:
        scheduler = None
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
            if args.loss == "info_nce":
                loss = loss_fn(q, d)
                valid_count = q_ids.size(0)
            else:
                # Supervised contrastive: for each anchor i, any doc j is positive iff
                # problems[i] == problems[j] and label[j] >= 0.5 (j is a positive doc for that problem).
                labels = batch.get("labels")
                problems = batch.get("problem")
                B = q.size(0)
                pos_mask = torch.zeros((B, B), dtype=torch.bool, device=q.device)
                valid_count = 0
                if labels is not None and problems is not None:
                    # labs on CPU for control flow; torch.bool list for indexing decisions
                    labs = (labels >= 0.5).cpu().tolist()
                    probs = problems  # list[str]
                    # Build: problem -> indices of positive docs in this batch
                    prob_to_pos = {}
                    for j in range(B):
                        if labs[j]:
                            prob_to_pos.setdefault(probs[j], []).append(j)
                    # For each anchor i, mark all positive docs of the same problem as positives
                    for i in range(B):
                        idxs = prob_to_pos.get(probs[i])
                        if idxs:
                            pos_mask[i, idxs] = True
                    # Count anchors that actually have at least one positive
                    valid_count = int((pos_mask.sum(dim=1) > 0).sum().item())
                loss = loss_fn(q, d, pos_mask)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if scheduler is not None:
                scheduler.step()
            # Average tracking: for SupCon, only anchors with at least one positive contribute to the mean;
            # use valid_count to keep a comparable running average.
            if args.loss == "supcon" and 'valid_count' in locals() and valid_count > 0:
                total += float(loss.item()) * valid_count
                n += valid_count
            else:
                total += float(loss.item()) * q_ids.size(0)
                n += q_ids.size(0)
            global_step += 1
            if global_step % 10 == 0:
                writer.add_scalar("train/loss_step", float(loss.item()), global_step)
                if args.loss == "supcon":
                    # anchors with positives ratio and mean positives per valid anchor
                    if 'pos_mask' in locals():
                        per_anchor_pos = pos_mask.sum(dim=1).float()
                        anchors_with_pos = (per_anchor_pos > 0).float().mean().item()
                        mean_pos_per_valid = per_anchor_pos[per_anchor_pos > 0].mean().item() if (per_anchor_pos > 0).any() else 0.0
                        writer.add_scalar("train/anchors_with_pos_ratio", anchors_with_pos, global_step)
                        writer.add_scalar("train/mean_pos_per_valid_anchor", mean_pos_per_valid, global_step)
                # log lr
                for i, pg in enumerate(opt.param_groups):
                    if "lr" in pg:
                        writer.add_scalar(f"train/lr/group{i}", pg["lr"], global_step)
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
            # Plot per-epoch artifacts and rolling curves
            plot_hist(pos_scores, neg_scores, args.logdir, name=f"score_hist_ep{ep}.png")
            plot_topk_curve(topk_curve, args.logdir, name=f"topk_curve_ep{ep}.png")
            if recalls_by_bucket:
                save_bucket_recalls(recalls_by_bucket, args.logdir, name=f"bucket_recalls_ep{ep}.json")
            # Rolling curves
            plot_curves({"train_loss": history["train_loss"], "val_R@64": history["val_R@64"]}, args.logdir, prefix="biencoder")
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
