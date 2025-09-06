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
    input_max_len: int = 2048

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
        inputs = self.tok(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.input_max_len,
        ).to(self.device)
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
    ap.add_argument("--input-max", type=int, default=2048, help="cap prompt tokens; left-truncation is used")
    # Optional RoPE scaling for longer context
    ap.add_argument("--rope-type", type=str, default="none", choices=["none", "linear", "dynamic", "yarn"], help="Apply RoPE scaling to extend context")
    ap.add_argument("--rope-factor", type=float, default=1.0, help="Scaling factor, e.g., 2.0 for ~8k if base is 4k")
    ap.add_argument("--rope-base", type=int, default=None, help="Original max_position_embeddings; if None, try from model config")
    args = ap.parse_args(argv)

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    mdl = AutoModelForCausalLM.from_pretrained(args.model, torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    # Apply optional RoPE scaling
    try:
        if args.rope_type != "none" and args.rope_factor and args.rope_factor > 1.0:
            cfg = mdl.config
            orig = int(args.rope_base) if args.rope_base else int(getattr(cfg, "max_position_embeddings", 4096))
            cfg.rope_scaling = {  # type: ignore[attr-defined]
                "type": args.rope_type,
                "factor": float(args.rope_factor),
                "original_max_position_embeddings": int(orig),
            }
            if hasattr(cfg, "max_position_embeddings"):
                try:
                    cfg.max_position_embeddings = int(orig * args.rope_factor)
                except Exception:
                    pass
            print(f"[INFO] RoPE scaling enabled in service: {cfg.rope_scaling}")
    except Exception as e:
        print(f"[WARN] Failed to apply RoPE scaling in service: {e}")
    if args.lora:
        from peft import PeftModel
        mdl = PeftModel.from_pretrained(mdl, args.lora)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mdl.to(device).eval()

    GenHandler.tok = tok
    GenHandler.mdl = mdl
    GenHandler.device = device
    # Prefer left truncation to retain the tail with candidates and JSON spec
    try:
        tok.truncation_side = "left"  # type: ignore[attr-defined]
    except Exception:
        pass
    GenHandler.input_max_len = int(args.input_max)
    print(f"ranker service on http://{args.host}:{args.port}")
    server = HTTPServer((args.host, args.port), GenHandler)
    server.serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
