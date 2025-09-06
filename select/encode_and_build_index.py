#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets


def _wrap_d(text: str, features: Optional[Dict] = None) -> str:
    prefix = features_to_prefix(features or {}, PrefixBuckets())
    return f"{prefix}<D> {normalize_text(text)} </D>"


@torch.no_grad()
def encode_docs(docs_path: str, model_path: str, spm_path: str, max_len: int = 256, batch_size: int = 512, limit: Optional[int] = None, device: Optional[str] = None):
    # Load checkpoint
    ckpt = torch.load(model_path, map_location="cpu")
    cfg_d = ckpt.get("config")
    if cfg_d is None:
        raise RuntimeError("Checkpoint missing config")
    cfg = TransformerConfig(**cfg_d)
    model = BiEncoder(cfg)
    model.load_state_dict(ckpt["model_state"])
    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.to(dev).eval()
    tok = LogicSentencePiece(spm_path)

    # Deduplicate by normalized text hash
    import hashlib
    seen = {}
    texts: list[str] = []
    feats: list[Optional[Dict]] = []
    metas: list[Dict] = []
    with open(docs_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            try:
                j = json.loads(line)
            except Exception:
                continue
            t = j.get("text") or j.get("doc") or j.get("d") or j.get("clause")
            if not t:
                continue
            norm = normalize_text(t)
            h = hashlib.md5(norm.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen[h] = True
            ftr = j.get("features") or j.get("meta")
            texts.append(t)
            feats.append(ftr)
            metas.append({
                "text": t,
                "hash": h,
                "item_id": j.get("item_id"),
                "problem_name": j.get("problem_name"),
                "features": ftr,
            })
            if limit is not None and len(texts) >= limit:
                break

    ids = np.arange(len(texts), dtype="int64")
    vecs: list[np.ndarray] = []
    # batch encode
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i+batch_size]
        batch_feats = feats[i:i+batch_size]
        d_ids_list = []
        d_masks = []
        for t, ftr in zip(batch_texts, batch_feats):
            s = _wrap_d(t, ftr)
            ids_ = tok.encode(s)[: max_len]
            d_ids_list.append(ids_)
            d_masks.append([1] * len(ids_))
        if not d_ids_list:
            continue
        maxl = max(len(x) for x in d_ids_list)
        pad_id = 0
        ids_t = torch.tensor([x + [pad_id] * (maxl - len(x)) for x in d_ids_list], dtype=torch.long, device=dev)
        mask_t = torch.tensor([x + [0] * (maxl - len(x)) for x in d_masks], dtype=torch.long, device=dev)
        Z = model.encode(ids_t, mask_t, which="d")
        Z = torch.nn.functional.normalize(Z, dim=1)
        vecs.append(Z.detach().cpu().numpy().astype("float32"))

    V = np.concatenate(vecs, axis=0) if vecs else np.zeros((0, cfg.d_model), dtype="float32")
    return ids, V, metas


def build_faiss_index(vecs: np.ndarray, metric: str = "ip"):
    try:
        import faiss  # type: ignore
    except Exception:
        return None
    d = vecs.shape[1]
    if metric == "ip":
        index = faiss.IndexFlatIP(d)
    else:
        index = faiss.IndexFlatL2(d)
    index.add(vecs)
    return index


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Encode docs and build index (FAISS if available, else NPZ)")
    ap.add_argument("--docs", type=str, required=True, help="JSONL with fields: text, features (optional)")
    ap.add_argument("--out", type=Path, required=True, help="Output prefix or file (.npz for numpy index or .faiss for FAISS)")
    ap.add_argument("--model", type=str, required=True, help="Path to BiEncoder checkpoint (.pt)")
    ap.add_argument("--spm", type=str, required=True, help="Path to sentencepiece model")
    ap.add_argument("--max-len", type=int, default=256)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"])
    args = ap.parse_args(argv)

    limit = args.limit if args.limit and args.limit > 0 else None
    ids, vecs, metas = encode_docs(args.docs, args.model, args.spm, max_len=args.max_len, batch_size=args.batch, limit=limit)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    # Try FAISS
    index = build_faiss_index(vecs, metric=args.metric)
    if index is not None and str(args.out).endswith(".faiss"):
        import faiss  # type: ignore
        faiss.write_index(index, str(args.out))
        # save mapping (both ids.txt for quick scan and meta.jsonl for rich info)
        with (args.out.with_suffix(".ids.txt")).open("w", encoding="utf-8") as f:
            for i, m in zip(ids.tolist(), metas):
                f.write(json.dumps({"id": int(i), "text": m.get("text", "")}, ensure_ascii=False) + "\n")
        with (args.out.with_suffix(".meta.jsonl")).open("w", encoding="utf-8") as f:
            for i, m in zip(ids.tolist(), metas):
                rec = {"id": int(i), **m}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"FAISS index saved: {args.out} with {vecs.shape[0]} vectors")
    else:
        # Save NPZ index
        np.savez_compressed(args.out, ids=ids, vecs=vecs)
        with (Path(str(args.out) + ".meta.jsonl")).open("w", encoding="utf-8") as f:
            for i, m in zip(ids.tolist(), metas):
                rec = {"id": int(i), **m}
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"NPZ index saved: {args.out} with {vecs.shape[0]} vectors")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
