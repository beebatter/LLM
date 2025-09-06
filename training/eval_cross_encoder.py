#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
<<<<<<< HEAD
from typing import Dict, List, Tuple, Union
import statistics

import torch
import torch.nn as nn
from tqdm.auto import tqdm

from LLM.models.logic_transformers import TransformerEncoder, TransformerConfig
from LLM.training.crossencoder_datasets import (
    build_cross_dataloader,
    build_cross_grouped_dataloader,
)


class CrossHead(nn.Module):
    """Same scoring head used in training: masked mean-pool + MLP to 1 logit."""

    def __init__(self, d_model: int, dropout: float = 0.1):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(dropout), nn.Linear(d_model, 1)
        )

    def forward(self, hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # hidden: [B,L,D], mask: [B,L]
        lengths = mask.sum(dim=1).clamp_min(1).unsqueeze(1)
        summed = (hidden * mask.unsqueeze(-1)).sum(dim=1)
        z = summed / lengths
        return self.proj(z).squeeze(-1)


def load_model(model_path: Path, spm_model: Path, d_model_default: int = 512, layers_default: int = 6, heads_default: int = 8, max_len_default: int = 256) -> Tuple[TransformerEncoder, CrossHead]:
    """Load encoder+head from a checkpoint saved by train_cross_encoder.

    Supports checkpoints saved as dict with keys: encoder_state, head_state, config, spm_model.
    Falls back to defaults when config missing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(model_path, map_location="cpu")

    # Extract config or fall back
    if isinstance(ckpt, dict) and "config" in ckpt:
        cfg_dict = ckpt["config"]
        cfg = TransformerConfig(
            vocab_size=cfg_dict.get("vocab_size", 32000),
            d_model=cfg_dict.get("d_model", d_model_default),
            n_heads=cfg_dict.get("n_heads", heads_default),
            n_layers=cfg_dict.get("n_layers", layers_default),
            pad_id=cfg_dict.get("pad_id", 0),
            max_len=cfg_dict.get("max_len", max_len_default),
        )
    else:
        # Configless ckpt: use safe defaults; vocab_size is only needed for embedding matrix
        cfg = TransformerConfig(
            vocab_size=32000,
            d_model=d_model_default,
            n_heads=heads_default,
            n_layers=layers_default,
            pad_id=0,
            max_len=max_len_default,
        )

    enc = TransformerEncoder(cfg).to(device)
    head = CrossHead(cfg.d_model).to(device)

    # Load weights if available
    if isinstance(ckpt, dict) and "encoder_state" in ckpt and "head_state" in ckpt:
        enc.load_state_dict(ckpt["encoder_state"], strict=False)
        head.load_state_dict(ckpt["head_state"], strict=False)
    elif isinstance(ckpt, dict) and "model_state" in ckpt:
        state = ckpt["model_state"]
        # heuristic split
        try:
            enc.load_state_dict({k.replace("encoder.", ""): v for k, v in state.items() if k.startswith("encoder.")}, strict=False)
            head.load_state_dict({k.replace("head.", ""): v for k, v in state.items() if k.startswith("head.")}, strict=False)
        except Exception:
            # Best-effort load into encoder
            enc.load_state_dict(state, strict=False)
    elif isinstance(ckpt, dict):
        # Try loading directly if it looks like a state_dict
        try:
            enc.load_state_dict(ckpt, strict=False)
        except Exception:
            pass
    else:
        # Unknown format; ignore and use random init
        pass

    enc.eval(); head.eval()
    return enc, head


def compute_pairwise_metrics(y_true: torch.Tensor, y_prob: torch.Tensor) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score, average_precision_score

    y = y_true.cpu().numpy()
    p = y_prob.cpu().numpy()
    out = {
        "auc": float(roc_auc_score(y, p)) if y.size > 0 else float("nan"),
        "ap": float(average_precision_score(y, p)) if y.size > 0 else float("nan"),
        "n": int(y.size),
    }
    return out


def _recall_at_k(rels: List[int], k: int) -> float:
    if not rels:
        return 0.0
    num_pos = sum(rels)
    if num_pos == 0:
        return 0.0
    topk = rels[:k]
    return sum(topk) / num_pos


def _dcg_at_k(rels: List[int], k: int) -> float:
    import math

    dcg = 0.0
    for i, r in enumerate(rels[:k], start=1):
        dcg += (2**r - 1) / math.log2(i + 1)
    return dcg


def _ndcg_at_k(rels: List[int], k: int) -> float:
    if not rels:
        return 0.0
    ideal = sorted(rels, reverse=True)
    idcg = _dcg_at_k(ideal, k)
    if idcg == 0.0:
        return 0.0
    return _dcg_at_k(rels, k) / idcg


def _mrr(rels: List[int]) -> float:
    for i, r in enumerate(rels, start=1):
        if r > 0:
            return 1.0 / i
    return 0.0


def compute_groupwise_metrics(groups: Dict[Union[int, str], List[Tuple[float, int]]], ks: List[int]) -> Dict[str, float]:
    """Compute ranking metrics by grouping predictions per query.

    groups: gid -> list of (score, label)
    """
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    mrrs: List[float] = []
    cand_counts: List[int] = []
    pos_counts: List[int] = []
    hits_all: List[int] = []  # top-1 hit across all groups
    hits_pos_only: List[int] = []  # top-1 hit among groups with at least one positive
    counted = 0
    with_pos = 0
    for gid, items in groups.items():
        counted += 1
        # sort by score desc, then get binary relevance list
        items_sorted = sorted(items, key=lambda x: x[0], reverse=True)
        rels = [int(lbl > 0.5) for _, lbl in items_sorted]
        pos_c = sum(rels)
        cand_counts.append(len(rels))
        pos_counts.append(pos_c)
        if pos_c > 0:
            with_pos += 1
        for k in ks:
            recalls[k].append(_recall_at_k(rels, k))
            ndcgs[k].append(_ndcg_at_k(rels, k))
        mrrs.append(_mrr(rels))
        top_rel = rels[0] if len(rels) > 0 else 0
        hits_all.append(top_rel)
        if pos_c > 0:
            hits_pos_only.append(top_rel)

    out = {
        "groups": counted,
        "groups_with_pos": with_pos,
        "mrr": float(sum(mrrs) / max(1, len(mrrs))),
    "avg_candidates": float(sum(cand_counts) / max(1, len(cand_counts))),
    "avg_positives": float(sum(pos_counts) / max(1, len(pos_counts))),
    "median_positives": float(statistics.median(pos_counts) if pos_counts else 0.0),
    "hit@1": float(sum(hits_all) / max(1, len(hits_all))),
    "hit@1_pos": float(sum(hits_pos_only) / max(1, len(hits_pos_only))) if hits_pos_only else 0.0,
    }
    for k in ks:
        out[f"recall@{k}"] = float(sum(recalls[k]) / max(1, len(recalls[k])))
        out[f"ndcg@{k}"] = float(sum(ndcgs[k]) / max(1, len(ndcgs[k])))
    return out


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate Cross-Encoder (pairwise + ranking)")
    ap.add_argument("--data", action="append", required=True, help="JSONL file(s) of pairs")
    ap.add_argument("--model", type=Path, required=True, help="Checkpoint path (.pt/.pkl)")
    ap.add_argument("--spm", type=Path, default=Path("/home/ks/Training/models/spm_logic.model"))
    ap.add_argument("--batch", type=int, default=256, help="Batch size for pairwise eval")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--ranking", action="store_true", help="Also compute groupwise ranking metrics")
    ap.add_argument("--ks", type=int, nargs="*", default=[1, 5, 10, 20, 50], help="K values for Recall/NDCG")
    ap.add_argument("--out", type=Path, help="Optional path to write JSON results")
    args = ap.parse_args(argv)

    enc, head = load_model(args.model, args.spm)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # 1) Pairwise metrics on a standard dataloader
    dl = build_cross_dataloader(args.data, spm_model=str(args.spm), batch_size=args.batch, shuffle=False, max_len=args.max_len)
    ys_all, ps_all = [], []
    with torch.no_grad():
        pbar = tqdm(dl, desc="eval", dynamic_ncols=True)
        for batch in pbar:
=======
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
>>>>>>> ddb8195ae6c9e30062ec225eba6c1f87730e53e2
            ids = batch["input_ids"].to(device)
            mask = batch["attention_mask"].to(device)
            y = batch["labels"].to(device)
            h = enc(ids, mask)
            logits = head(h, mask)
<<<<<<< HEAD
            ys_all.append(y.cpu())
            ps_all.append(logits.sigmoid().cpu())
    if len(ys_all) == 0:
        pairwise = {"auc": float("nan"), "ap": float("nan"), "n": 0}
    else:
        ycat = torch.cat(ys_all)
        pcat = torch.cat(ps_all)
        pairwise = compute_pairwise_metrics(ycat, pcat)

    results: Dict[str, float] = {**pairwise}

    # 2) Optional ranking metrics using grouped batches (requires group ids in dataset)
    if args.ranking:
        # Use grouped dataloader; set a large group_size to avoid truncation
        gdl = build_cross_grouped_dataloader(
            args.data,
            spm_model=str(args.spm),
            group_size=10_000_000,
            groups_per_batch=4,
            max_len=args.max_len,
            shuffle=False,
        )
        groups: Dict[Union[int, str], List[Tuple[float, int]]] = {}
        with torch.no_grad():
            pbar = tqdm(gdl, desc="ranking", dynamic_ncols=True)
            for bidx, batch in enumerate(pbar):
                ids = batch["input_ids"].to(device)
                mask = batch["attention_mask"].to(device)
                y = batch["labels"].to(device)
                gid = batch.get("group_ids")
                if gid is None:
                    continue
                gid = gid.cpu().tolist()
                h = enc(ids, mask)
                logits = head(h, mask).sigmoid().cpu().tolist()
                ys = y.cpu().tolist()
                for g, s, lbl in zip(gid, logits, ys):
                    # Compose a unique key per batch to avoid collisions when group ids are re-used per-batch
                    gkey = f"{bidx}:{int(g)}"
                    groups.setdefault(gkey, []).append((s, lbl))
        ranking = compute_groupwise_metrics(groups, args.ks)
        results.update(ranking)

    # Output
    print(json.dumps(results, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

=======
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
>>>>>>> ddb8195ae6c9e30062ec225eba6c1f87730e53e2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
