#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets


def _wrap_q(text: str) -> str:
    return f"<Q> {normalize_text(text)} </Q>"


class SelectHandler(BaseHTTPRequestHandler):
    index_vecs: np.ndarray = np.zeros((0, 1), dtype="float32")
    index_ids: np.ndarray = np.zeros((0,), dtype="int64")
    metric: str = "ip"
    # encoder
    device: torch.device
    encoder: Optional[BiEncoder] = None
    tok: Optional[LogicSentencePiece] = None
    max_len: int = 256

    def do_POST(self):
        if self.path != "/select":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get('Content-Length', '0'))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8"))
        except Exception:
            self.send_response(400)
            self.end_headers()
            return
        # req: {"query": "...", "k": 64}
        k = int(req.get("k", 64))
        text = req.get("query") or ""
        if self.encoder is None or self.tok is None or not text:
            self._respond({"ids": []})
            return
        # encode query
        with torch.no_grad():
            s = _wrap_q(text)
            ids_ = self.tok.encode(s)[: self.max_len]
            pad = 0
            ids_t = torch.tensor([ids_], dtype=torch.long, device=self.device)
            mask_t = torch.tensor([[1] * len(ids_)], dtype=torch.long, device=self.device)
            q = self.encoder.encode(ids_t, mask_t, which="q")
            q = torch.nn.functional.normalize(q, dim=1).cpu().numpy().astype("float32")[0]
        if self.index_vecs.size == 0:
            res = []
        else:
            sims = self.index_vecs @ q if self.metric == "ip" else -((self.index_vecs - q) ** 2).sum(axis=1)
            topk = np.argsort(-sims)[:k]
            res = [int(self.index_ids[i]) for i in topk]
        self._respond({"ids": res})

    def _respond(self, obj):
        out = json.dumps(obj).encode("utf-8")
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select service (BiEncoder + NPZ/FAISS index)")
    ap.add_argument("--index", type=Path, required=True, help=".npz or .faiss index file")
    ap.add_argument("--model", type=str, required=True, help="BiEncoder checkpoint path")
    ap.add_argument("--spm", type=str, required=True, help="SentencePiece model path")
    ap.add_argument("--metric", type=str, default="ip", choices=["ip", "l2"])
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    ap.add_argument("--max-len", type=int, default=256)
    args = ap.parse_args(argv)

    # load index
    vecs = None
    ids = None
    if str(args.index).endswith('.npz'):
        data = np.load(args.index)
        ids = data["ids"]
        vecs = data["vecs"]
    else:
        try:
            import faiss  # type: ignore
        except Exception:
            raise RuntimeError("FAISS not available, provide .npz index instead")
        index = faiss.read_index(str(args.index))
        # For service simplicity, also keep vectors in RAM (if index is flat)
        if isinstance(index, faiss.IndexFlat):
            xb = faiss.vector_to_array(index.xb).reshape(index.ntotal, index.d)
            vecs = xb.astype('float32')
            ids = np.arange(index.ntotal, dtype='int64')
        else:
            raise RuntimeError("Only flat FAISS index supported in this minimal service")

    # load encoder
    ckpt = torch.load(args.model, map_location="cpu")
    cfg = TransformerConfig(**ckpt["config"])  # type: ignore
    enc = BiEncoder(cfg)
    enc.load_state_dict(ckpt["model_state"])  # type: ignore
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    enc.to(device).eval()
    tok = LogicSentencePiece(args.spm)

    SelectHandler.index_ids = ids if ids is not None else np.zeros((0,), dtype="int64")
    SelectHandler.index_vecs = vecs if vecs is not None else np.zeros((0, cfg.d_model), dtype="float32")
    SelectHandler.encoder = enc
    SelectHandler.tok = tok
    SelectHandler.device = device
    SelectHandler.metric = args.metric
    SelectHandler.max_len = args.max_len
    print(f"index loaded: shape={SelectHandler.index_vecs.shape}; service on http://{args.host}:{args.port}")

    server = HTTPServer((args.host, args.port), SelectHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
