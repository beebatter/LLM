#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets


def _clean_clause_text(tcf_clause: str) -> str:
    """Best-effort: drop provenance and tcf wrapper, keep the formula.

    Expected input like:
      tcf(c_77,plain, (![X0:$i]: (~p(X0)|q(X0))),file('x', tag)).

    Strategy:
      - cut at ",file(" if present (drop provenance)
      - take substring after "plain," as the formula
      - strip outer whitespace/dot
    """
    s = tcf_clause.strip()
    i = s.find(",file(")
    if i != -1:
        s = s[:i]
    # remove trailing ")." if present
    s = s.rstrip().rstrip('.')
    j = s.find("plain,")
    if j != -1:
        s = s[j + len("plain,") :]
    # remove possible leading/trailing parens
    s = s.strip()
    # common wrapping: tcf(...,plain, <formula>)
    # at this point s should be the formula or close to it
    return s


def _wrap_q(text: str) -> str:
    return f"<Q> {normalize_text(text)} </Q>"


def _wrap_d(text: str, features: Optional[Dict] = None) -> str:
    prefix = features_to_prefix(features or {}, PrefixBuckets())
    return f"{prefix}<D> {normalize_text(text)} </D>"


def _parse_nul_file(path: Path) -> Tuple[Dict[int, Dict], List[int]]:
    with path.open('rb') as f:
        parts = [seg for seg in f.read().split(b"\x00") if seg.strip()]
    if not parts:
        raise RuntimeError("Empty input")
    reg = json.loads(parts[0].decode('utf-8'))
    if reg.get('tag') != 'register_clauses':
        raise RuntimeError("First message must be register_clauses")
    # collect clauses map
    clauses: Dict[int, Dict] = {}
    for c in reg.get('clauses', []):
        cid = int(c.get('clause_id'))
        raw = c.get('clause') or ''
        feats = c.get('clause_features') or {}
        formula = _clean_clause_text(raw)
        clauses[cid] = {'text': formula, 'features': feats}
    # find scores_req
    scores_ids: List[int] = []
    for seg in parts[1:]:
        try:
            msg = json.loads(seg.decode('utf-8'))
        except Exception:
            continue
        if msg.get('tag') == 'scores_req':
            scores_ids = [int(x) for x in msg.get('clause_ids', [])]
            break
    if not scores_ids:
        raise RuntimeError("scores_req not found")
    return clauses, scores_ids


@torch.no_grad()
def rank_topk(
    clauses: Dict[int, Dict],
    scores_ids: List[int],
    model_path: Path,
    spm_path: Path,
    k: int = 200,
    max_len: int = 256,
    batch_size: int = 512,
    device: Optional[str] = None,
) -> Tuple[List[int], Dict[int, float]]:
    # find conjecture: conj_dist == 0 if present
    conj_id = None
    for cid, info in clauses.items():
        feats = info.get('features') or {}
        if feats.get('conj_dist') == 0:
            conj_id = cid
            break
    if conj_id is None:
        # fallback: first id of scores_ids
        conj_id = scores_ids[0]
    q_text = clauses[conj_id]['text']

    # load encoder
    ckpt = torch.load(str(model_path), map_location='cpu')
    cfg = TransformerConfig(**ckpt['config'])  # type: ignore
    model = BiEncoder(cfg)
    model.load_state_dict(ckpt['model_state'])  # type: ignore
    dev = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    model.to(dev).eval()
    tok = LogicSentencePiece(str(spm_path))

    # encode query
    q_ids = tok.encode(_wrap_q(q_text))[: max_len]
    q_ids_t = torch.tensor([q_ids], dtype=torch.long, device=dev)
    q_mask_t = torch.tensor([[1] * len(q_ids)], dtype=torch.long, device=dev)
    q_vec = model.encode(q_ids_t, q_mask_t, which='q')
    q_vec = torch.nn.functional.normalize(q_vec, dim=1)

    # encode docs (only requested IDs)
    cand_ids = [cid for cid in scores_ids if cid in clauses]
    d_vecs: List[np.ndarray] = []
    for i in range(0, len(cand_ids), batch_size):
        batch = cand_ids[i:i+batch_size]
        texts = [_wrap_d(clauses[cid]['text'], clauses[cid].get('features')) for cid in batch]
        ids_list = [tok.encode(t)[: max_len] for t in texts]
        maxl = max(1, max(len(x) for x in ids_list))
        pad = 0
        ids_t = torch.tensor([x + [pad] * (maxl - len(x)) for x in ids_list], dtype=torch.long, device=dev)
        mask_t = torch.tensor([[1] * len(x) + [0] * (maxl - len(x)) for x in ids_list], dtype=torch.long, device=dev)
        z = model.encode(ids_t, mask_t, which='d')
        z = torch.nn.functional.normalize(z, dim=1)
        d_vecs.append(z.detach().cpu().numpy().astype('float32'))
    if d_vecs:
        D = np.concatenate(d_vecs, axis=0)
    else:
        D = np.zeros((0, q_vec.shape[1]), dtype='float32')

    # compute sims and top-k
    sims = (D @ q_vec.cpu().numpy().T).reshape(-1)
    order = np.argsort(-sims)
    top_idx = order[:k].tolist()
    id2score = {int(cand_ids[i]): float(sims[i]) for i in range(len(cand_ids))}
    top_ids = [int(cand_ids[i]) for i in top_idx]
    return top_ids, id2score


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Rank top-K clause_ids from iProver raw (NUL-separated) using Bi-Encoder")
    ap.add_argument('--raw', type=Path, required=True, help='Path to *.jsonl.nul file (NUL-separated JSON messages)')
    ap.add_argument('--model', type=Path, required=True, help='Bi-Encoder checkpoint (.pt)')
    ap.add_argument('--spm', type=Path, required=True, help='SentencePiece model')
    ap.add_argument('-k', type=int, default=200)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--batch', type=int, default=512)
    args = ap.parse_args(argv)

    clauses, ids = _parse_nul_file(args.raw)
    top_ids, id2score = rank_topk(
        clauses, ids, model_path=args.model, spm_path=args.spm,
        k=args.k, max_len=args.max_len, batch_size=args.batch,
    )
    out = {
        'top_ids': top_ids,
        'scores': {str(k): v for k, v in id2score.items()},
    }
    print(json.dumps(out, ensure_ascii=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
