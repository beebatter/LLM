#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import torch
from tqdm.auto import tqdm

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.training.biencoder_datasets import build_bi_dataloader


def _ndcg_at_k(labels: List[int], k: int) -> float:
    import math
    k = min(k, len(labels))
    if k <= 0:
        return 0.0
    gains = [1.0 / math.log2(i + 2) if labels[i] > 0 else 0.0 for i in range(k)]
    dcg = sum(gains)
    ideal = sorted(labels, reverse=True)[:k]
    idcg = sum(1.0 / math.log2(i + 2) if ideal[i] > 0 else 0.0 for i in range(len(ideal)))
    return (dcg / idcg) if idcg > 0 else 0.0


@torch.no_grad()
def eval_retrieval_multiK(
    model: BiEncoder,
    loader,
    device: torch.device,
    ks: List[int],
):
    """
    Compute per-problem retrieval with global doc pool from the loader batch stream.
    Metrics: Hit@K (any positive in top-K), Recall@K (fraction of positives in top-K), NDCG@K.
    """
    model.eval()
    all_q, all_d = [], []
    problems: List[str] = []
    d_hashes: List[str] = []
    labels_all: List[float] = []

    for batch in tqdm(loader, desc="encode", dynamic_ncols=True):
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
            labels_all.extend([float(x) for x in lbs.tolist()])
        else:
            labels_all.extend([0.0] * q.size(0))

    if not all_q or not all_d:
        return {
            "n_problems": 0,
            "hit@K": {int(k): 0.0 for k in ks},
            "recall@K": {int(k): 0.0 for k in ks},
            "ndcg@K": {int(k): 0.0 for k in ks},
        }

    Q = torch.nn.functional.normalize(torch.cat(all_q, dim=0), dim=1)
    D = torch.nn.functional.normalize(torch.cat(all_d, dim=0), dim=1)

    # Unique doc vectors by hash
    uniq_map: Dict[str, int] = {}
    uniq_vecs: List[torch.Tensor] = []
    for i, h in enumerate(d_hashes):
        if h not in uniq_map:
            uniq_map[h] = len(uniq_vecs)
            uniq_vecs.append(D[i])
    if not uniq_vecs:
        return {
            "n_problems": 0,
            "hit@K": {int(k): 0.0 for k in ks},
            "recall@K": {int(k): 0.0 for k in ks},
            "ndcg@K": {int(k): 0.0 for k in ks},
        }
    Dmat = torch.stack(uniq_vecs, dim=0)  # [Nd, d]
    Dt = Dmat.t()

    # Positives by problem (set of doc hashes)
    pos_by_prob: Dict[str, Set[str]] = {}
    for i, prob in enumerate(problems):
        if labels_all[i] >= 0.5:
            h = d_hashes[i]
            if h:
                pos_by_prob.setdefault(prob, set()).add(h)

    # Query embedding per problem (average of its rows)
    import collections
    q_sum = collections.defaultdict(lambda: torch.zeros(Q.size(1)))
    q_cnt = collections.defaultdict(int)
    for i, prob in enumerate(problems):
        q_sum[prob] = q_sum[prob] + Q[i]
        q_cnt[prob] += 1
    q_by_prob = {k: (v / max(1, q_cnt[k])) for k, v in q_sum.items()}

    Ks = sorted(set(int(k) for k in ks))
    hit = {k: 0 for k in Ks}
    rec = {k: 0.0 for k in Ks}
    ndcg = {k: 0.0 for k in Ks}
    n_considered = 0

    for prob, qv in q_by_prob.items():
        pos = list(pos_by_prob.get(prob, set()))
        if not pos:
            continue
        n_considered += 1
        sims = (qv @ Dt).squeeze(0)  # [Nd]
        # argsort desc
        order = torch.argsort(sims, dim=0, descending=True)
        ranked_hashes = [list(uniq_map.keys())[int(i)] for i in order.tolist()]
        # build binary relevance list up to max K
        maxK = max(Ks)
        labels_ranked = [1 if (i < len(ranked_hashes) and ranked_hashes[i] in pos) else 0 for i in range(min(maxK, len(ranked_hashes)))]
        for k in Ks:
            k2 = min(k, len(labels_ranked))
            hits = sum(labels_ranked[:k2])
            hit[k] += 1 if hits > 0 else 0
            rec[k] += (hits / float(len(pos))) if len(pos) > 0 else 0.0
            ndcg[k] += _ndcg_at_k(labels_ranked, k)

    if n_considered == 0:
        return {
            "n_problems": 0,
            "hit@K": {int(k): 0.0 for k in ks},
            "recall@K": {int(k): 0.0 for k in ks},
            "ndcg@K": {int(k): 0.0 for k in ks},
        }

    return {
        "n_problems": n_considered,
        "hit@K": {int(k): hit[int(k)] / n_considered for k in ks},
        "recall@K": {int(k): rec[int(k)] / n_considered for k in ks},
        "ndcg@K": {int(k): ndcg[int(k)] / n_considered for k in ks},
    }


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Eval Bi-Encoder on JSONL (Hit@K / Recall@K / NDCG@K)")
    ap.add_argument("--data", action="append", required=True, help="JSONL path(s) to evaluate")
    ap.add_argument("--model", type=Path, required=True, help="Bi-Encoder checkpoint path (.pt)")
    ap.add_argument("--spm", type=str, default=None, help="SentencePiece model path (override)")
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--k", type=str, default="10,32,64", help="Comma-separated K values")
    ap.add_argument("--out", type=Path, default=None, help="Optional JSON to write metrics")
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])  # type: ignore
    model = BiEncoder(cfg)
    model.load_state_dict(ckpt["model_state"])  # type: ignore
    model.to(device).eval()
    spm_path = args.spm or ckpt.get("spm_model")
    if not spm_path:
        raise RuntimeError("SPM model path missing; pass --spm or ensure checkpoint has 'spm_model'.")

    loader = build_bi_dataloader(args.data, spm_model=str(spm_path), batch_size=args.batch, shuffle=False, max_len=args.max_len)
    ks = [int(x) for x in args.k.split(",") if x.strip()]
    metrics = eval_retrieval_multiK(model, loader, device, ks)
    print(json.dumps(metrics, ensure_ascii=False, indent=2))
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, ensure_ascii=False, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
