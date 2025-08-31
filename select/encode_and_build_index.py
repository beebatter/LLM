#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List

import numpy as np

# NOTE: To keep dependencies light, we postpone FAISS import until needed.

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Encode docs and build FAISS/HNSW index (stub)")
    ap.add_argument("--docs", type=str, required=True, help="JSONL with fields: text, features (optional)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--model", type=str, default="/home/ks/Training/models/biencoder.pt")
    ap.add_argument("--spm", type=str, default="/home/ks/Training/models/spm_logic.model")
    ap.add_argument("--dim", type=int, default=512)
    args = ap.parse_args(argv)

    # TODO: load BiEncoder, encode texts into embeddings; here we store random vectors as a stub.
    vecs = []
    ids = []
    with open(args.docs, "r", encoding="utf-8", errors="ignore") as f:
        for i, line in enumerate(f):
            try:
                j = json.loads(line)
            except Exception:
                continue
            if not j.get("text"):
                continue
            ids.append(i)
            vecs.append(np.random.randn(args.dim).astype("float32"))

    vecs = np.stack(vecs) if vecs else np.zeros((0, args.dim), dtype="float32")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, ids=np.array(ids, dtype="int64"), vecs=vecs)
    print(f"stub index saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
