#!/usr/bin/env python3
from __future__ import annotations

"""
Minimal local text generation service for scheme B (LLM rerank).

POST /generate {"prompt": str, "temperature": float?, "max_new_tokens": int?}
-> {"text": str}

Intended to work with batch_ranker.py via LLM_LOCAL_ENDPOINT=http://127.0.0.1:8001/generate
Use any causal LM (e.g., deepseek-ai/deepseek-math-7b-instruct) optionally with a LoRA adapter.
"""

import argparse
from http.server import BaseHTTPRequestHandler, HTTPServer
import json
from typing import Optional

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM


class GenHandler(BaseHTTPRequestHandler):
    tok: Optional[AutoTokenizer] = None  # type: ignore
    mdl: Optional[AutoModelForCausalLM] = None  # type: ignore
    device: torch.device

    def do_POST(self):
        if self.path != "/generate":
            self.send_response(404)
            self.end_headers()
            return
        length = int(self.headers.get("Content-Length", "0"))
        body = self.rfile.read(length)
        try:
            req = json.loads(body.decode("utf-8", errors="ignore"))
        except Exception:
            self._respond({"text": ""})
            return
        prompt = req.get("prompt") or ""
        temp = float(req.get("temperature", 0.0))
        max_new = int(req.get("max_new_tokens", 256))
        if not prompt or self.mdl is None or self.tok is None:
            self._respond({"text": ""})
            return
        inputs = self.tok(prompt, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.mdl.generate(
                **inputs,
                do_sample=(temp > 0),
                temperature=max(0.01, temp),
                max_new_tokens=max_new,
                eos_token_id=self.tok.eos_token_id,
            )
        text = self.tok.decode(out[0], skip_special_tokens=True)
        # Return only the assistant/completion segment: heuristic split by prompt
        if text.startswith(prompt):
            text = text[len(prompt):]
        self._respond({"text": text.strip()})

    def _respond(self, obj):
        data = json.dumps(obj, ensure_ascii=False).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Local LLM generation service")
    ap.add_argument("--model", type=str, required=True, help="HF model id/path")
    ap.add_argument("--lora", type=str, help="Optional LoRA adapter path")
    ap.add_argument("--host", type=str, default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8001)
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    if args.lora:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, args.lora)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()

    GenHandler.tok = tok
    GenHandler.mdl = mdl
    GenHandler.device = device
    print(f"ranker service on http://{args.host}:{args.port}")
    server = HTTPServer((args.host, args.port), GenHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
