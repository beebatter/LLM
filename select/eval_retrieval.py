#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import math

from LLM.data_utils.logic_tokenizer import normalize_text


def load_pos_labels(labels_paths: List[Path]) -> Dict[str, Set[str]]:
    """Map normalized query text -> set of normalized positive doc texts."""
    q2pos: Dict[str, Set[str]] = {}
    for p in labels_paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                if not line.strip():
                    continue
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                qa = j.get("text_a") or j.get("query") or j.get("conjecture_text") or j.get("q")
                db = j.get("text_b") or j.get("doc") or j.get("text")
                if qa is None or db is None:
                    continue
                if int(j.get("label", 0)) != 1:
                    continue
                qn = normalize_text(str(qa))
                dn = normalize_text(str(db))
                q2pos.setdefault(qn, set()).add(dn)
    return q2pos


def load_meta_texts(meta_path: Optional[Path]) -> List[str]:
    if not meta_path:
        return []
    texts: List[str] = []
    with open(meta_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            texts.append(j.get("text") or j.get("canonical_formula") or "")
    return texts


def ndcg_at_k(labels: List[int], k: int) -> float:
    # Binary labels list in ranked order; compute NDCG@K
    k = min(k, len(labels))
    if k == 0:
        return 0.0
    gains = [1.0 / math.log2(i + 2) if labels[i] > 0 else 0.0 for i in range(k)]
    dcg = sum(gains)
    ideal_labels = sorted(labels, reverse=True)[:k]
    ideal_gains = [1.0 / math.log2(i + 2) if ideal_labels[i] > 0 else 0.0 for i in range(len(ideal_labels))]
    idcg = sum(ideal_gains)
    return (dcg / idcg) if idcg > 0 else 0.0


def eval_files(
    queries_file: Path,
    reranked_file: Path,
    labels_files: List[Path],
    index_meta: Optional[Path],
    ks: List[int],
) -> Dict[str, Dict[int, float]]:
    q2pos = load_pos_labels(labels_files)
    meta_texts = load_meta_texts(index_meta)

    # Accumulators
    n_considered = 0
    # For each K: hit rate, recall (fraction of positives retrieved), ndcg
    bi_hit = {k: 0 for k in ks}
    bi_rec = {k: 0.0 for k in ks}
    bi_ndcg = {k: 0.0 for k in ks}
    ce_hit = {k: 0 for k in ks}
    ce_rec = {k: 0.0 for k in ks}
    ce_ndcg = {k: 0.0 for k in ks}

    with open(queries_file, "r", encoding="utf-8", errors="ignore") as fq, \
         open(reranked_file, "r", encoding="utf-8", errors="ignore") as fr:
        for li, (lq, lr) in enumerate(zip(fq, fr), start=1):
            try:
                jq = json.loads(lq)
                jr = json.loads(lr)
            except Exception:
                continue
            q = jq.get("query") or jq.get("text_a") or jq.get("conjecture_text") or jq.get("q") or ""
            qn = normalize_text(str(q))
            pos = q2pos.get(qn)
            if not pos:
                # skip queries without positives in labels
                continue

            # CE ranked texts
            ce_texts = [normalize_text(str(it.get("text", ""))) for it in jr.get("scores", [])]
            # Bi baseline texts (map candidate_ids via meta)
            bi_ids = jq.get("candidate_ids") or []
            bi_texts: List[str] = []
            if meta_texts:
                for cid in bi_ids:
                    try:
                        bi_texts.append(normalize_text(meta_texts[int(cid)]))
                    except Exception:
                        bi_texts.append("")

            # Labels (binary) for top-K sequences
            ce_labels = [1 if t in pos else 0 for t in ce_texts]
            bi_labels = [1 if t in pos else 0 for t in bi_texts]

            n_pos = len(pos)
            if n_pos == 0:
                continue

            n_considered += 1
            for k in ks:
                # CE
                k_ce = min(k, len(ce_labels))
                hits_ce = sum(ce_labels[:k_ce])
                ce_hit[k] += 1 if hits_ce > 0 else 0
                ce_rec[k] += hits_ce / n_pos
                ce_ndcg[k] += ndcg_at_k(ce_labels, k)
                # Bi
                k_bi = min(k, len(bi_labels))
                hits_bi = sum(bi_labels[:k_bi])
                bi_hit[k] += 1 if hits_bi > 0 else 0
                bi_rec[k] += hits_bi / n_pos
                bi_ndcg[k] += ndcg_at_k(bi_labels, k)

    if n_considered == 0:
        raise SystemExit("No queries with positives found; check labels/inputs alignment.")

    # Aggregate to rates
    out: Dict[str, Dict[int, float]] = {}
    out["n_queries"] = {0: float(n_considered)}
    out["bi_hit@K"] = {k: bi_hit[k] / n_considered for k in ks}
    out["ce_hit@K"] = {k: ce_hit[k] / n_considered for k in ks}
    out["bi_recall@K"] = {k: bi_rec[k] / n_considered for k in ks}
    out["ce_recall@K"] = {k: ce_rec[k] / n_considered for k in ks}
    out["bi_ndcg@K"] = {k: bi_ndcg[k] / n_considered for k in ks}
    out["ce_ndcg@K"] = {k: ce_ndcg[k] / n_considered for k in ks}
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Evaluate CE reranking vs Bi-Encoder retrieval")
    ap.add_argument("--queries", type=Path, required=True, help="queries_with_topK_ids.jsonl")
    ap.add_argument("--reranked", type=Path, required=True, help="queries_topK_reranked.jsonl")
    ap.add_argument("--labels", type=Path, nargs="+", required=True, help="One or more CE dataset JSONL files with labels (val)")
    ap.add_argument("--index-meta", type=Path, required=True, help="Index .meta.jsonl to map candidate_ids to texts")
    ap.add_argument("--k", type=str, default="10,32,64,100,200", help="Comma-separated K values")
    args = ap.parse_args(argv)

    ks = [int(x) for x in args.k.split(",") if x.strip()]
    metrics = eval_files(args.queries, args.reranked, list(args.labels), args.index_meta, ks)
    print(json.dumps(metrics, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
