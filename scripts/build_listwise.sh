#!/usr/bin/env bash
set -euo pipefail

# Ensure repo is on PYTHONPATH
export PYTHONPATH="/root:${PYTHONPATH:-}"

# Build listwise windows and (optionally) fuse teacher scores.
# Edit the variables below to match your paths.

# Inputs (clause-level supervised JSONL)
TRAIN_SRC="/root/autodl-tmp/Training/datasets/ce_prepared_full_excl.train.jsonl"
VAL_SRC="/root/autodl-tmp/Training/datasets/ce_prepared_full_excl.val.jsonl"

# Output dir
OUT_DIR="/root/autodl-tmp/Training/datasets/listwise"
K=64            # window size (64/128)
SEED=2025
MAX_W_PER_PROB=4
MAX_TEXT_CHARS=256

# Optional teacher fusion
# Preferred: your explicit paths below; if missing, we'll auto-detect under /root/autodl-tmp/Training
CROSS_CKPT="${CROSS_CKPT:-/root/autodl-tmp/Training/models/cross_encoder_best.pt}"
BI_CKPT="${BI_CKPT:-/root/autodl-tmp/Training/models/biencoder_best.pt}"
SPM_MODEL="${SPM_MODEL:-/root/autodl-tmp/Training/models/spm_logic_24k.model}"
LAMBDA_CROSS=1.0
LAMBDA_BI=0.3
TAU=1.0

TRAIN_ROOT="/root/autodl-tmp/Training"

find_largest() {
  # usage: find_largest <root> <maxdepth> <pattern>
  local root="$1"; local depth="$2"; local pattern="$3"
  [ -d "$root" ] || return 1
  # print size and path, sort by size desc, pick first path
  local out
  out=$(find "$root" -maxdepth "$depth" -type f -iname "$pattern" -printf '%s\t%p\n' 2>/dev/null | sort -nr | head -n 1 | cut -f2-)
  if [ -n "${out:-}" ] && [ -f "$out" ]; then
    echo "$out"
    return 0
  fi
  return 1
}

autodetect_teacher_artifacts() {
  if [ -z "${CROSS_CKPT:-}" ] || [ ! -f "$CROSS_CKPT" ]; then
    CROSS_CKPT=$(find_largest "$TRAIN_ROOT/models" 3 '*cross*encod*best*.pt' || true)
    [ -z "$CROSS_CKPT" ] && CROSS_CKPT=$(find_largest "$TRAIN_ROOT" 4 '*cross*encod*.pt' || true)
  fi
  if [ -z "${BI_CKPT:-}" ] || [ ! -f "$BI_CKPT" ]; then
    BI_CKPT=$(find_largest "$TRAIN_ROOT/models" 3 '*bi*encod*best*.pt' || true)
    [ -z "$BI_CKPT" ] && BI_CKPT=$(find_largest "$TRAIN_ROOT" 4 '*bi*encod*.pt' || true)
    # fallback generic best.pt
    [ -z "$BI_CKPT" ] && BI_CKPT=$(find_largest "$TRAIN_ROOT/models" 3 '*biencoder*best*.pt' || true)
  fi
  if [ -z "${SPM_MODEL:-}" ] || [ ! -f "$SPM_MODEL" ]; then
    SPM_MODEL=$(find_largest "$TRAIN_ROOT/models" 3 '*spm*.model' || true)
    [ -z "$SPM_MODEL" ] && SPM_MODEL=$(find_largest "$TRAIN_ROOT" 4 '*spm*.model' || true)
  fi
}

mkdir -p "$OUT_DIR"

echo "[Build] train windows -> $OUT_DIR/train.listwise.jsonl"
python -m LLM.scripts.make_listwise_chunks \
  --input "$TRAIN_SRC" \
  --out   "$OUT_DIR/train.listwise.jsonl" \
  --window "$K" \
  --max-windows-per-problem "$MAX_W_PER_PROB" \
  --seed "$SEED" \
  --max-text-chars "$MAX_TEXT_CHARS"

echo "[Build] val windows   -> $OUT_DIR/val.listwise.jsonl"
python -m LLM.scripts.make_listwise_chunks \
  --input "$VAL_SRC" \
  --out   "$OUT_DIR/val.listwise.jsonl" \
  --window "$K" \
  --max-windows-per-problem "$MAX_W_PER_PROB" \
  --seed "$SEED" \
  --max-text-chars "$MAX_TEXT_CHARS"

autodetect_teacher_artifacts

echo "[Teacher] CROSS_CKPT=${CROSS_CKPT:-}"
echo "[Teacher] BI_CKPT=${BI_CKPT:-}"
echo "[Teacher] SPM_MODEL=${SPM_MODEL:-}"

if [[ -n "${CROSS_CKPT:-}" && -f "$CROSS_CKPT" && -n "${BI_CKPT:-}" && -f "$BI_CKPT" && -n "${SPM_MODEL:-}" && -f "$SPM_MODEL" ]]; then
  echo "[Teacher] fuse scores for train..."
  python -m LLM.scripts.make_listwise_targets_from_teacher \
    --in-listwise "$OUT_DIR/train.listwise.jsonl" \
    --out         "$OUT_DIR/train.listwise.teacher.jsonl" \
    --cross-ckpt  "$CROSS_CKPT" \
    --bi-ckpt     "$BI_CKPT" \
    --spm         "$SPM_MODEL" \
    --lambda-cross "$LAMBDA_CROSS" \
    --lambda-bi    "$LAMBDA_BI" \
    --tau          "$TAU"

  echo "[Teacher] fuse scores for val..."
  python -m LLM.scripts.make_listwise_targets_from_teacher \
    --in-listwise "$OUT_DIR/val.listwise.jsonl" \
    --out         "$OUT_DIR/val.listwise.teacher.jsonl" \
    --cross-ckpt  "$CROSS_CKPT" \
    --bi-ckpt     "$BI_CKPT" \
    --spm         "$SPM_MODEL" \
    --lambda-cross "$LAMBDA_CROSS" \
    --lambda-bi    "$LAMBDA_BI" \
    --tau          "$TAU"
else
  echo "[Teacher] Skipped (missing artifacts). Please set CROSS_CKPT/BI_CKPT/SPM_MODEL or place them under $TRAIN_ROOT."
fi

echo "[Done] Listwise data are in $OUT_DIR"
