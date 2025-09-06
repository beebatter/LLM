#!/usr/bin/env bash
set -euo pipefail

# Simple wrapper for LLM/docs/eval_llm_listwise.py

VAL_JSONL="/root/autodl-tmp/Training/datasets/listwise/val.listwise.jsonl"  # or *.teacher.jsonl
ENDPOINT="http://127.0.0.1:8000/generate"  # set your local inference endpoint
LIMIT=200
TEMP=0.0
MAX_NEW=256
TIMEOUT=60

python -m LLM.docs.eval_llm_listwise \
  --data "$VAL_JSONL" \
  --limit "$LIMIT" \
  --temperature "$TEMP" \
  --max-new "$MAX_NEW" \
  --timeout "$TIMEOUT" \
  --endpoint "$ENDPOINT"
