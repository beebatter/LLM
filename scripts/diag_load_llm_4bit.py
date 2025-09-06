#!/usr/bin/env python3
"""
Interactive diagnostic loader for large causal LMs with bitsandbytes 4-bit.

It prints detailed environment info, auto-computes a safe max_memory mapping
from `nvidia-smi`, and attempts to load a model using:
  - BitsAndBytesConfig(load_in_4bit=True)
  - device_map=auto (configurable)
  - max_memory (auto or user-provided JSON)
  - low_cpu_mem_usage (optional)

On failures, it prints full tracebacks and optionally tries a fallback load.

Example:
  python -m LLM.scripts.diag_load_llm_4bit \
    --model /root/autodl-tmp/models/Goedel-Prover-V2-32B \
    --device-map auto --max-memory auto --low-cpu-mem-usage
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import traceback

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig, __version__ as _tf_ver

try:
    import bitsandbytes as bnb  # noqa: F401
    _bnb_ver = getattr(bnb, "__version__", "unknown")
except Exception:
    bnb = None
    _bnb_ver = "unavailable"


def _auto_max_memory() -> dict | None:
    try:
        out = subprocess.check_output([
            "nvidia-smi",
            "--query-gpu=memory.total",
            "--format=csv,noheader,nounits",
        ], encoding="utf-8")
        vals = [int(x.strip()) for x in out.strip().splitlines() if x.strip()]
        mm = {}
        for i, v in enumerate(vals):
            cap_mb = int(v * 0.8)
            mm[i] = f"{cap_mb}MB"
        return mm
    except Exception as e:
        print(f"[warn] failed to read nvidia-smi for max_memory: {e}")
        return None


def _print_env():
    print("python:", sys.version)
    print("torch:", torch.__version__, "cuda_available=", torch.cuda.is_available(), "cuda_count=", torch.cuda.device_count())
    print("transformers:", _tf_ver)
    print("bitsandbytes:", _bnb_ver)
    if torch.cuda.is_available():
        try:
            names = []
            for i in range(torch.cuda.device_count()):
                names.append(torch.cuda.get_device_name(i))
            print("cuda devices:", names)
        except Exception:
            pass


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Diagnose 4-bit loading via bitsandbytes")
    ap.add_argument("--model", required=True, help="Local path or HF id of the model")
    ap.add_argument("--device-map", default="auto", help="device_map to pass (default: auto)")
    ap.add_argument("--max-memory", default="auto", help="'auto' or JSON mapping, e.g. '{\"0\":\"40GB\",\"1\":\"40GB\"}'")
    ap.add_argument("--low-cpu-mem-usage", action="store_true", help="Set low_cpu_mem_usage=True when loading")
    ap.add_argument("--no-quant", action="store_true", help="If set, skip 4-bit quantization path (for control)")
    ap.add_argument("--bf16", action="store_true", help="Use bfloat16 compute dtype for 4-bit path (default fp16)")
    args = ap.parse_args(argv)

    print("[env]")
    _print_env()
    print("\n[config]")
    print("  model:", args.model)
    print("  device_map:", args.device_map)
    print("  low_cpu_mem_usage:", args.low_cpu_mem_usage)
    print("  quantization:", (not args.no_quant))

    # Resolve max_memory mapping
    max_mem = None
    if args.max_memory:
        if args.max_memory.strip().lower() == "auto":
            max_mem = _auto_max_memory()
        else:
            try:
                parsed = json.loads(args.max_memory)
                if isinstance(parsed, dict):
                    # Coerce string keys like "0" to integer indices {0: "40GB"}
                    max_mem = { (int(k) if isinstance(k, str) and k.isdigit() else k): v for k, v in parsed.items() }
                else:
                    max_mem = parsed
            except Exception as e:
                print(f"[warn] failed to parse --max-memory JSON: {e}")
                max_mem = None
    if max_mem is not None:
        print("  max_memory:", max_mem)
    else:
        print("  max_memory: None")

    # Try loading tokenizer first (lightweight sanity)
    try:
        tok = AutoTokenizer.from_pretrained(args.model, use_fast=False, trust_remote_code=True)
        print("[ok] tokenizer loaded. vocab_size=", getattr(tok, "vocab_size", None))
        del tok
    except Exception:
        print("[err] tokenizer load failed:")
        traceback.print_exc()

    # 4-bit path
    if not args.no_quant:
        print("\n[step] 4-bit loading via BitsAndBytesConfig ...")
        if bnb is None or BitsAndBytesConfig is None:
            print("[err] bitsandbytes/quantization not available; cannot do 4-bit. Please install bitsandbytes and ensure compatible CUDA.")
        else:
            try:
                bnb_conf = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
                )
                load_kwargs = dict(
                    quantization_config=bnb_conf,
                    device_map=args.device_map,
                    trust_remote_code=True,
                )
                if args.low_cpu_mem_usage:
                    load_kwargs["low_cpu_mem_usage"] = True
                if max_mem is not None:
                    load_kwargs["max_memory"] = max_mem

                model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
                print("[ok] model loaded in 4-bit.")
                try:
                    total = sum(p.numel() for p in model.parameters())
                    print("  total_params:", total)
                except Exception:
                    pass
                del model
            except Exception:
                print("[err] 4-bit load failed with traceback:")
                traceback.print_exc()

    # Fallback (non-quant) path
    print("\n[step] fallback standard load (CPU) just to validate checkpoint integrity ...")
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, trust_remote_code=True, device_map="cpu")
        print("[ok] fallback CPU load succeeded (not usable for inference speed, only integrity check).")
        del model
    except Exception:
        print("[err] fallback CPU load also failed with traceback:")
        traceback.print_exc()

    print("\n[done] review logs above for the first failing step and message.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
