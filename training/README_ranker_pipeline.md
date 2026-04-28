# Ranker Pipeline: Data -> Models -> Evaluation -> iProver

This guide shows how to use your existing CE listwise pairs to build listwise groups (optional), produce predictions with different models, compare them with a unified evaluator, and outline how to integrate into iProver.

## 1) Data: keep your pairs, optionally build listwise groups

- You already have pairs JSONL at:
  - `/root/autodl-tmp/Training/datasets/pairs.full.train.jsonl`
  - `/root/autodl-tmp/Training/datasets/pairs.full.dev.jsonl`
  - `/root/autodl-tmp/Training/datasets/pairs.full.test.jsonl`

- To build group-oriented data with K candidates per query and (optional) CE teacher soft labels:

  python3 LLM/training/prepare_listwise_data.py \
    --input /root/autodl-tmp/Training/datasets/pairs.full.train.jsonl \
    --output /root/autodl-tmp/Training/datasets/groups.k48.train.jsonl \
    --k 48 --min-positives 1 --seed 42 \
    --teacher ce --tau 1.0 \
    --ce-scored /root/autodl-tmp/Training/datasets/ce_scored.train.jsonl

Notes:
- You do not need to regenerate pairs; use them directly for CE/BI scoring. Listwise groups are useful for LLM scoring-head training.

## 2) Produce predictions in a unified format

Cross-Encoder predictions (optionally also emit ce_scored.*):

  python3 LLM/training/score_cross_encoder.py \
    --data /root/autodl-tmp/Training/datasets/pairs.full.dev.jsonl \
    --model /home/ks/Training/models/cross_encoder_best.pt \
    --spm /home/ks/Training/models/spm_logic.model \
    --out /root/autodl-tmp/Training/datasets/ce_scored.dev.jsonl \
    --pred-out /root/autodl-tmp/Training/datasets/pred.ce.dev.jsonl

Bi-Encoder cosine predictions:

  python3 LLM/training/score_biencoder.py \
    --data /root/autodl-tmp/Training/datasets/pairs.full.dev.jsonl \
    --model /home/ks/Training/models/biencoder_best.pt \
    --spm /home/ks/Training/models/spm_logic.model \
    --out /root/autodl-tmp/Training/datasets/pred.bi.dev.jsonl

LLM scoring-head predictions: once trained, dump the same unified JSONL schema:

  {"problem_name": str, "query": str, "group_id": str, "doc": str, "label": 0/1, "score": float}

## 3) Compare models with the unified evaluator

  python3 LLM/training/unified_evaluator.py \
    --pred /root/autodl-tmp/Training/datasets/pred.ce.dev.jsonl --name CE \
    --pred /root/autodl-tmp/Training/datasets/pred.bi.dev.jsonl --name BI \
    --ks 1 10 32 64 \
    --out /root/autodl-tmp/Training/datasets/metrics.dev.json

Outputs per model: ranking (MRR/Recall/NDCG@K), classification (AUC/AP), and diagnostics (avg candidates/positives, hit@1).

## 4) iProver integration (hook overview)

The existing GNN server (`iprover-gnn-server-master/server/iserver_mp.py`) accepts clause batches, builds a temporary CNF file, and calls a scorer (PIEGNN) to return scores.

Options to integrate your rankers:

1) External re-ranker (low-risk):
   - Keep iProver + GNN server flow untouched.
   - Periodically export clause batches (via `--evaldata_dir`) and run your CE/BI/LLM re-ranker offline to produce new scores, then feed them back for ablation/analysis.

2) Drop-in scorer replacement (medium risk):
   - Replace `network.predict(messages)` in `iserver_mp.py` with a thin adapter that:
     - Converts `GraphData` back to text clause strings (or pass-through if you record the original clauses from `register_clauses`).
     - Wraps conjecture + clause text with your tokenizer (`_wrap_qd`) and batches through CE/BI/LLM.
     - Returns a list of floats as scores.
   - Start with CE (fastest to wire) and keep the rest identical.

3) Sidecar HTTP/IPC service (safe and flexible):
   - Run a small Python HTTP server exposing `/score` that accepts a list of {query, candidates[]} and returns scores.
   - In `gpu_worker`, replace `network.predict` with an HTTP client call. Add batching and timeouts.

Practical tips:
- Respect `query_max_size` and context sizes; if K×len is too large, micro-batch candidates.
- Cache the conjecture encoding to reduce recomputation (prefix-cache); only the clause part changes per candidate.
- Keep a switch to fall back to the native GNN if the sidecar is down.

## 5) Calibration and fusion

- After generating predictions, perform per-group z-score and optional (tau, b) calibration on dev and reuse for test/online (save calib.yaml).
- For fusion with BI: score_fused = w_bi * bi_score + w_main * score; tune weights on dev for Hit@10 and NDCG@32.

## 6) Next steps

- Train 7B scoring-head with groups.k48.* (listwise CE/KL on teacher scores) and export unified predictions for eval.
- Optionally run 32B PMI as teacher; prepare `target_scores` for group-wise distillation.
- Integrate the fast scorer into iProver via sidecar first, then consider in-process replacement.
