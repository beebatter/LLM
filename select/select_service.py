#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Tuple

import numpy as np


class SelectHandler(BaseHTTPRequestHandler):
    index_vecs: np.ndarray = np.zeros((0, 1), dtype="float32")
    index_ids: np.ndarray = np.zeros((0,), dtype="int64")

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
        # TODO: encode query by BiEncoder; here random vector to demo
        q = np.random.randn(self.index_vecs.shape[1]).astype("float32")
        if self.index_vecs.size == 0:
            res = []
        else:
            sims = self.index_vecs @ q
            topk = np.argsort(-sims)[:k]
            res = [int(self.index_ids[i]) for i in topk]
        out = json.dumps({"ids": res}).encode("utf-8")
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(out)))
        self.end_headers()
        self.wfile.write(out)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Select service (stub)")
    ap.add_argument("--index", type=Path, required=True)
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9000)
    args = ap.parse_args(argv)

    data = np.load(args.index)
    SelectHandler.index_ids = data["ids"]
    SelectHandler.index_vecs = data["vecs"]
    print(f"index loaded: {args.index} shape={SelectHandler.index_vecs.shape}")

    server = HTTPServer((args.host, args.port), SelectHandler)
    print(f"select service on http://{args.host}:{args.port}")
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
