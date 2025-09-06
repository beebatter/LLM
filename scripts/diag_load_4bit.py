#!/usr/bin/env python3
"""
Diagnostic utility: attempt to load a causal LM with BitsAndBytesConfig (4-bit) using transformers dispatch.

Usage (example):
  python LLM/scripts/diag_load_4bit.py --model /path/to/model --load-4bit --device-map auto --max-memory auto

This script prints detailed environment info, attempts the load, prints any traceback, then releases resources.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback

import torch


def run(cmd):
    try:
        return subprocess.check_output(cmd, encoding='utf-8', stderr=subprocess.STDOUT)
    except Exception as e:
        return str(e)


def infer_max_memory_map():
    """Return a dict mapping integer GPU indices to safe max_memory strings like {'0':'51200MB'} or None on failure."""
    try:
        out = run(['nvidia-smi', '--query-gpu=memory.total', '--format=csv,noheader,nounits'])
        lines = [l.strip() for l in out.splitlines() if l.strip()]
        vals = [int(x) for x in lines]
        mm = {}
        for i, v in enumerate(vals):
            cap_mb = int(v * 0.8)  # safety margin
            mm[i] = f"{cap_mb}MB"
        return mm
    except Exception as e:
        print('Failed to infer max_memory via nvidia-smi:', e)
        return None


def main(argv=None):
    ap = argparse.ArgumentParser(description="Diagnose 4-bit model loading with BitsAndBytesConfig + transformers dispatch")
    ap.add_argument('--model', required=True, help='Path to the model directory or HF id')
    ap.add_argument('--load-4bit', action='store_true', help='Use BitsAndBytesConfig/4-bit quantization')
    ap.add_argument('--device-map', default='auto', help="device_map to pass (default 'auto')")
    ap.add_argument('--max-memory', default=None, help="Max memory mapping (JSON or 'auto')")
    ap.add_argument('--low-cpu-mem-usage', action='store_true', help='Pass low_cpu_mem_usage=True when loading')
    args = ap.parse_args(argv)

    print('=== Environment ===')
    print('python:', sys.version.replace('\n', ' '))
    print('cwd:', os.getcwd())
    try:
        import transformers

        print('transformers', transformers.__version__)
    except Exception as e:
        print('transformers import failed:', e)
    try:
        import bitsandbytes as bnb

        print('bitsandbytes', getattr(bnb, '__version__', 'unknown'))
    except Exception as e:
        print('bitsandbytes import failed:', e)
    print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available(), 'cuda_count', torch.cuda.device_count())

    from transformers import AutoModelForCausalLM

    bnb_conf = None
    if args.load_4bit:
        try:
            from bitsandbytes import BitsAndBytesConfig

            bnb_conf = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.float16)
            print('BitsAndBytesConfig created')
        except Exception:
            print('Failed to import or create BitsAndBytesConfig:')
            traceback.print_exc()
            bnb_conf = None

    # prepare max_memory mapping
    max_memory = None
    if args.max_memory:
        mm = args.max_memory.strip()
        if mm.lower() == 'auto':
            print('Attempting to infer max_memory via nvidia-smi...')
            max_memory = infer_max_memory_map()
            print('inferred max_memory:', max_memory)
        else:
            try:
                max_memory = json.loads(mm)
            except Exception:
                print('Could not parse max_memory as JSON; ignoring')
                max_memory = None

    load_kwargs = dict(trust_remote_code=True)
    if args.device_map:
        load_kwargs['device_map'] = args.device_map
    if args.low_cpu_mem_usage:
        load_kwargs['low_cpu_mem_usage'] = True
    if max_memory is not None:
        load_kwargs['max_memory'] = max_memory

    if bnb_conf is not None:
        load_kwargs['quantization_config'] = bnb_conf

    print('\n=== Load attempt ===')
    print('model:', args.model)
    print('load_kwargs keys:', list(load_kwargs.keys()))

    start = time.time()
    try:
        model = AutoModelForCausalLM.from_pretrained(args.model, **load_kwargs)
        elapsed = time.time() - start
        print(f'Loaded model successfully in {elapsed:.1f}s')
        try:
            total = sum(p.numel() for p in model.parameters())
            print('total params (approx):', total)
        except Exception:
            print('Could not compute total params')
        # quick sanity forward small input if CUDA available
        if torch.cuda.is_available():
            try:
                import transformers
                tok = transformers.AutoTokenizer.from_pretrained(args.model, use_fast=False)
                input_ids = tok('Hello', return_tensors='pt').input_ids.to(next(model.parameters()).device)
                with torch.no_grad():
                    out = model(input_ids=input_ids)
                print('Performed a tiny forward, output logits shape:', getattr(out, 'logits', None).shape)
            except Exception:
                print('Tiny forward failed:')
                traceback.print_exc()
    except Exception:
        print('Exception during from_pretrained:')
        traceback.print_exc()
        return 2
    finally:
        try:
            # release
            del model
            torch.cuda.empty_cache()
            time.sleep(1.0)
            print('Released model and cleared CUDA cache')
        except Exception:
            pass

    print('\nDone. If loading failed, re-run with --max-memory auto and capture its stdout/stderr to review the traceback above.')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
