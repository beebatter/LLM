#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text


def _wrap_q(text: str) -> str:
    return f"<Q> {normalize_text(text)} </Q>"


def load_queries_from_jsonl(paths: List[Path], max_queries: Optional[int] = None) -> List[Dict]:
    out: List[Dict] = []
    seen = set()
    for p in paths:
        with open(p, "r", encoding="utf-8", errors="ignore") as f:
            for line in f:
                try:
                    j = json.loads(line)
                except Exception:
                    continue
                q = j.get("query") or j.get("text_a") or j.get("conjecture_text") or j.get("q")
                if not q:
                    continue
                pn = j.get("problem_name")
                key = (pn, q)
                if key in seen:
                    continue
                seen.add(key)
                out.append({"query": q, "problem_name": pn})
                if max_queries is not None and len(out) >= max_queries:
                    return out
    return out


def load_npz_index(path: Path) -> Tuple[np.ndarray, np.ndarray]:
    data = np.load(path)
    ids = data["ids"]  # shape [N]
    vecs = data["vecs"].astype("float32")  # shape [N,d]
    return ids, vecs


def load_faiss_index(path: Path):
    import faiss  # type: ignore
    index = faiss.read_index(str(path))
    return index


def build_bi_encoder(ckpt_path: Path, device: torch.device):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])  # type: ignore
    enc = BiEncoder(cfg)
    enc.load_state_dict(ckpt["model_state"])  # type: ignore
    enc.to(device).eval()
    return enc, cfg


def encode_queries(tok: LogicSentencePiece, enc: BiEncoder, queries: List[str], max_len: int, device: torch.device, batch_size: int = 256) -> np.ndarray:
    embs: List[np.ndarray] = []
    for i in range(0, len(queries), batch_size):
        chunk = queries[i:i+batch_size]
        ids_list: List[List[int]] = []
        mask_list: List[List[int]] = []
        for q in chunk:
            s = _wrap_q(q)
            ids = tok.encode(s)[: max_len]
            ids_list.append(ids)
            mask_list.append([1]*len(ids))
        maxl = max(len(x) for x in ids_list) if ids_list else 0
        ids_t = torch.tensor([x + [0]*(maxl-len(x)) for x in ids_list], dtype=torch.long, device=device)
        mask_t = torch.tensor([m + [0]*(maxl-len(m)) for m in mask_list], dtype=torch.long, device=device)
        with torch.no_grad():
            q = enc.encode(ids_t, mask_t, which="q")
            q = torch.nn.functional.normalize(q, dim=1)
        embs.append(q.detach().float().cpu().numpy())
    return np.vstack(embs) if embs else np.zeros((0, 1), dtype="float32")


def topk_ip_np(qs: np.ndarray, vecs: np.ndarray, ids: np.ndarray, k: int) -> Tuple[np.ndarray, np.ndarray]:
    # qs: [M,d], vecs: [N,d] (both L2-normalized); return (indices, scores)
    sims = qs @ vecs.T  # [M,N]
    k = min(k, vecs.shape[0])
    top_idx = np.argpartition(-sims, kth=k-1, axis=1)[:, :k]
    top_scores = np.take_along_axis(sims, top_idx, axis=1)
    # sort within top-k
    order = np.argsort(-top_scores, axis=1)
    top_idx = np.take_along_axis(top_idx, order, axis=1)
    top_scores = np.take_along_axis(top_scores, order, axis=1)
    top_ids = ids[top_idx]
    return top_ids, top_scores


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Batch select top-K ids for queries using Bi-Encoder and NPZ/FAISS index")
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--queries-jsonl", type=Path, help="JSONL with per-line {query,...} (also accepts text_a/conjecture_text)")
    g.add_argument("--from-crossencoder", type=Path, nargs="+", help="Use CE dataset JSONL(s) to extract unique queries")
    ap.add_argument("--index", type=Path, required=True, help=".npz or .faiss index file")
    ap.add_argument("--model", type=Path, required=True, help="Bi-Encoder checkpoint path")
    ap.add_argument("--spm", type=Path, required=True, help="SentencePiece model path")
    ap.add_argument("--k", type=int, default=200)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tok = LogicSentencePiece(str(args.spm))
    enc, cfg = build_bi_encoder(args.model, device)

    # Load queries
    if args.queries_jsonl:
        queries = load_queries_from_jsonl([args.queries_jsonl])
    else:
        queries = load_queries_from_jsonl([Path(p) for p in args.from_crossencoder])
    if not queries:
        raise RuntimeError("No queries found in input.")

    # Encode queries
    q_texts = [q["query"] for q in queries]
    q_embs = encode_queries(tok, enc, q_texts, max_len=args.max_len, device=device, batch_size=args.batch)

    # Search
    if str(args.index).endswith('.npz'):
        ids, vecs = load_npz_index(args.index)
        top_ids, top_scores = topk_ip_np(q_embs, vecs, ids, args.k)
    else:
        import faiss  # type: ignore
        index = load_faiss_index(args.index)
        # Expect Flat IP index; produce top-k per batch
        k = min(args.k, index.ntotal)
        top_ids = []
        top_scores = []
        for i in range(0, q_embs.shape[0], 512):
            q = q_embs[i:i+512].astype('float32')
            D, I = index.search(q, k)
            top_scores.append(D)
            top_ids.append(I.astype('int64'))
        top_ids = np.vstack(top_ids)
        top_scores = np.vstack(top_scores)

    # Write JSONL
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with open(args.out, 'w', encoding='utf-8') as f:
        for i, q in enumerate(queries):
            rec = {
                "query": q["query"],
                "problem_name": q.get("problem_name"),
                "candidate_ids": top_ids[i].tolist(),
                "bi_scores": [float(s) for s in top_scores[i].tolist()],
                "topk": int(args.k),
            }
            f.write(json.dumps(rec, ensure_ascii=False) + '\n')
    print(f"wrote {len(queries)} lines to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
