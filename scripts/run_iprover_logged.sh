#!/usr/bin/env bash
# Run iProver and save full stdout/stderr to a timestamped log for later comparison.
# Usage:
#   bash LLM/scripts/run_iprover_logged.sh ./iprover-master/iproveropt [iprover-args ...] /path/to/problem.p
# Env:
#   LOG_DIR   Optional. Where to store logs. Default: Logs/iprover
#   LOG_TAG   Optional. Label in filename, e.g., "bi+CE", "bi+LLM", "iprover-only".
#             If unset, will try to derive from EA_MODE (bi_then_cross -> bi+CE, bi_then_llm -> bi+LLM).
set -euo pipefail

LOG_DIR="${LOG_DIR:-Logs/iprover}"
mkdir -p "$LOG_DIR"
TS="$(date +%Y%m%d_%H%M%S)"
PROBLEM=""
for arg in "$@"; do
  if [[ "$arg" == *.p ]]; then PROBLEM="$arg"; fi
done
BASE="session"
if [[ -n "$PROBLEM" ]]; then
  BASE="$(basename "$PROBLEM" .p)"
fi

# Determine tag for filename
TAG="${LOG_TAG:-}"
if [[ -z "$TAG" ]]; then
  case "${EA_MODE:-}" in
    bi_then_cross) TAG="bi+CE" ;;
    bi_then_llm)   TAG="bi+LLM" ;;
    *)             TAG="" ;;
  esac
fi
sanitize() { echo "$1" | tr ' /\\:' '____' | tr -cd '[:alnum:]_+.-'; }
TAG_SAFE=""; if [[ -n "$TAG" ]]; then TAG_SAFE="__$(sanitize "$TAG")"; fi

LOG_FILE="$LOG_DIR/${TS}_${BASE}${TAG_SAFE}.log"

# Announce
echo "[wrap] logging to $LOG_FILE"
# Store the full command line
{
  echo "[wrap] cmd: $*"
  if [[ -n "$TAG" ]]; then echo "[wrap] tag: $TAG"; fi
} | tee -a "$LOG_FILE" >/dev/null

# Execute and tee output
"$@" 2>&1 | tee -a "$LOG_FILE"
RET=${PIPESTATUS[0]}
{
  echo "[wrap] exit code: $RET"
} | tee -a "$LOG_FILE" >/dev/null
exit $RET
