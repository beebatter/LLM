# Source this file to set up TPTP path for iProver and tools
# usage: source LLM/env.sh

# TPTP root (adjust if your dataset is elsewhere)
export TPTP=/root/autodl-tmp/TPTP-v9.0.0

# Optional: add helpful aliases
alias ip-bsl='(cd /root/iprover-master && ./iproveropt --include_path "$TPTP" )'
# Example: ip-bsl $TPTP/Problems/NUM/NUM322+1.p
