#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import socket
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

from LLM.models.logic_transformers import ClauseScorer, TransformerConfig
from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets


def wrap_clause(text: str, features: Optional[Dict] = None) -> str:
    prefix = ""
    if features:
        prefix = features_to_prefix(features, PrefixBuckets())
    return f"{prefix}<D> {normalize_text(text)} </D>"


@dataclass
class ClauseInfo:
    clause_id: int
    text: str
    features: Optional[Dict]


class EAState:
    def __init__(self, model_path: Path, spm_model: Path, device: Optional[str] = None):
        ckpt = torch.load(model_path, map_location="cpu")
        cfg_dict = ckpt["config"]
        cfg = TransformerConfig(**cfg_dict)
        self.model = ClauseScorer(cfg)
        self.model.load_state_dict(ckpt["model_state"])
        self.model.eval()
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model.to(self.device)
        self.tok = LogicSentencePiece(str(spm_model))
        self.pad_id = 0
        self.clauses: Dict[int, ClauseInfo] = {}

    @torch.no_grad()
    def score_ids(self, clause_ids: List[int]) -> List[float]:
        texts: List[str] = []
        for cid in clause_ids:
            ci = self.clauses.get(cid)
            if not ci:
                texts.append("<D> </D>")
            else:
                texts.append(wrap_clause(ci.text, ci.features))
        ids_list = [self.tok.encode(x)[:512] for x in texts]
        maxl = max(1, max(len(x) for x in ids_list))
        pad = self.pad_id
        ids = torch.tensor([x + [pad] * (maxl - len(x)) for x in ids_list], dtype=torch.long, device=self.device)
        mask = torch.tensor([[1] * len(x) + [0] * (maxl - len(x)) for x in ids_list], dtype=torch.long, device=self.device)
        scores = self.model(ids, mask).float().cpu().tolist()
        return [float(s) for s in scores]


class EAServer:
    def __init__(self, host: str, port: int, state: EAState):
        self.host = host
        self.port = port
        self.state = state
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

    def start(self):
        self.sock.bind((self.host, self.port))
        self.sock.listen(1)
        print(f"EA listening on {self.host}:{self.port}")
        while True:
            conn, addr = self.sock.accept()
            print(f"client connected: {addr}")
            threading.Thread(target=self.handle_client, args=(conn,), daemon=True).start()

    def handle_client(self, conn: socket.socket):
        with conn:
            buf = b""
            while True:
                data = conn.recv(8192)
                if not data:
                    break
                buf += data
                # Assuming line-delimited JSON
                while b"\n" in buf:
                    line, buf = buf.split(b"\n", 1)
                    if not line.strip():
                        continue
                    try:
                        msg = json.loads(line.decode("utf-8"))
                    except Exception:
                        continue
                    resp = self.dispatch(msg)
                    if resp is not None:
                        out = (json.dumps(resp) + "\n").encode("utf-8")
                        conn.sendall(out)

    def dispatch(self, msg: Dict) -> Optional[Dict]:
        tag = msg.get("tag")
        if tag == "register_clauses":
            for c in msg.get("clauses", []):
                cid = int(c.get("clause_id"))
                clause_str = c.get("clause") or ""
                # Extract main formula inside (...) best-effort; fallback to full string
                text = clause_str
                features = c.get("clause_features") or {}
                self.state.clauses[cid] = ClauseInfo(cid, text=text, features=features)
            return None
        if tag == "scores_req":
            ids = [int(x) for x in msg.get("clause_ids", [])]
            scores = self.state.score_ids(ids)
            return {"tag": "scores_res", "scores": scores}
        if tag == "server_queries_start":
            # We don't query iProver at the moment
            return None
        if tag == "server_queries_end":
            return None
        # Other tags can be safely ignored or logged
        return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="External Agent (PSM) for iProver using Transformer clause scorer")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=12345)
    ap.add_argument("--model", type=Path, default=Path("/home/ks/Training/models/clause_scorer.pt"))
    ap.add_argument("--spm", type=Path, default=Path("/home/ks/Training/models/spm_logic.model"))
    ap.add_argument("--device", type=str, default=None)
    args = ap.parse_args(argv)

    state = EAState(model_path=args.model, spm_model=args.spm, device=args.device)
    EAServer(args.host, args.port, state).start()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
