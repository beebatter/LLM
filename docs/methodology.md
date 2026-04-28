# Methodology

## Overview
This document details the end‑to‑end external reasoning assistance pipeline integrating iProver's interactive clause processing with a modular neural scoring architecture (Bi / Cross / LLM / Fusion). The system targets long first‑order logic (FOL) clause retrieval and reranking with robustness to heterogeneous checkpoints and extreme input lengths.

## Interactive Protocol Layer
- Message Types: `register_clauses`, `server_queries_start`, `scores_req`, `scores_res`, `server_queries_end`.
- Transport: newline‐delimited JSON (no legacy ACK packets). NUL separators were removed for compatibility.
- Buffering: `scores_req` messages received before `server_queries_start` are queued then flushed once queries begin.
- Fallback: If no explicit conjecture (distance==0) is registered, a heuristic fallback selects the earliest plausible clause.
- Queue Configuration Dependency: iProver must enable `external_score` priority queue (positive weight) for `scores_req` generation.

## Backend Architecture
Each backend implements a uniform `score(queries, candidates)` contract returning float scores.

Backends:
1. BiTFBackend (bi‑encoder retrieval)
2. CrossTFBackend (cross encoder rerank)
3. BiThenCross (two‑stage: ANN-style bi retrieval -> cross rerank)
4. BiThenLLM (bi retrieval -> lightweight LLM scoring)
5. LLMDIRECT / LLMPMI (prompt or PMI token log‑prob scoring)
6. FusionBackend (weighted combination of prior backends; optional normalization: none | softmax | minmax | zscore)

Shared Features:
- Unified checkpoint loader tolerant to top-level keys: `model_state`, `model`, `state_dict`.
- Verbosity gating: diagnostic prints emitted only under `--verbose`.
- Caching: Bi encoder embeddings and rerank intermediate scores cached per clause id (in‑memory) to reduce recomputation.

## Long Sequence Handling
Challenge: Clauses frequently exceed the transformer pretraining window; naive truncation degraded recall.

Solution: Chunked sliding window encoding (Bi encoders) activated when input length > runtime threshold.
- Parameters: `--bi-chunk-encode` flag set plus:
  - `--bi-chunk-len` (window length)
  - `--bi-chunk-stride` (hop size)
  - `--bi-chunk-max` (cap on number of chunks per example)
  - `--bi-chunk-agg` aggregation strategy: `mean` | `max` | `attn`
- Aggregation:
  - mean: arithmetic mean of normalized chunk embeddings
  - max: elementwise max
  - attn: learned scalar attention weights over chunks (lightweight MLP)
- Trigger Logic: Only applied when (raw_token_count > `override_max_len` > model.config.max_len OR raw_token_count > model.config.max_len without override). This preserves efficiency for short inputs.

## Max Length Override
- Flag: `--max-len-override` (server side) defers truncation up to a higher ceiling (currently 2048 tokens) independent of the baked model config.
- Purpose: Allows reuse of shorter‑trained checkpoints while experimenting with extended inference context via chunking.

## Dimension Mismatch Resilience
Problem: Heterogeneous checkpoints produced embeddings of varying dimensionalities (e.g., 256, 384, 512, 640, 691, 1231) causing dot‑product failures.

Mitigations:
1. Adaptive Pooling: Ensures sequence -> vector via masked mean before similarity.
2. Projection Layer (only if needed): Dynamically instantiated linear `Linear(in_dim, target_dim)` with Xavier init when a mismatch is first detected; message emitted under verbose mode.
3. Normalization: cosine similarity computed after `L2` normalization for stability across projections.

## Training Pipelines
### Bi‑Encoder Distillation
Script: `training/train_biencoder_distill.py`
- Objective: Regress or distill cross‑encoder / teacher scores.
- Loss Options: MSE (default) or KL over per‑query softmax distribution (temperature adjustable).
- Defaults: `--max-len=2048`, `d_model=512`, 6 layers, 8 heads.
- Data Schema: Flexible JSONL with multiple accepted key aliases for queries, clauses, scores, grouping, and optional feature dictionaries.

### Cross Encoder Training
Script: `training/train_cross_encoder.py`
- Objectives: Binary classification (BCE) or listwise group softmax ranking (`--loss=listwise`).
- Architecture: Shared transformer encoder + lightweight pooling/scoring head.
- Defaults: `--max-len=2048`, `batch=128`, same base transformer depth as bi encoder.
- Evaluation: ROC AUC per epoch; optional histogram artifacts.

### Additional Training Utilities
- Pointwise & listwise dataset preparation (`prepare_pointwise_dataset.py`, `prepare_listwise_data*.py`).
- Clause scorer / transformer ranker variants for ablations.
- LLM fine‑tuning scripts (LoRA, listwise SFT) present; LLM head ranking pathway currently experimental and not production‑stable.

## Two‑Stage & Fusion Scoring
- BiThenCross: Retrieves top-K via bi encoder similarity, then recomputes cross scores on that subset, replacing or fusing scores.
- BiThenLLM: Similar retrieval front; passes top candidates into LLM scoring (token logprob or PMI scheme) when enabled.
- FusionBackend: Applies optional score normalization then linear combination with configurable weights; supports extension (e.g., weight search) without modifying base backends.

## Caching Strategy
- Keyed by (clause_id, encoder_variant) to avoid recomputation across multiple queries in the same interactive session.
- Cross rerank stage caches pairwise embeddings or final logits where safe (depends on deterministic preprocessing).
- Cache invalidation: Fresh server run implies empty caches; runtime flags can disable caching (future toggle not yet implemented—current behavior always on).

## Verbosity & Diagnostics
- `--verbose` reveals: checkpoint load pathway, detected dimension mismatches, projection creation, chunk counts (future enhancement), and timing summaries.
- Default (non‑verbose) output limited to essential high‑level lifecycle notices (server start, backend ready).

## Robust Checkpoint Loading
Pseudo‑algorithm:
1. Load checkpoint (torch.load).
2. Extract state dict via first matching key in `[model_state, model, state_dict]` else treat whole object as state dict.
3. Filter unexpected keys if necessary (strict=False load).
4. Read config dict for transformer hyperparams; fall back to runtime args when absent.

## Evaluation (Planned / Ongoing)
Metrics:
- Retrieval: Recall@K, MRR on held‑out interactive problems using bi encoder.
- Rerank: AUC / group NDCG for cross encoder outputs.
- Fusion: Ablation of normalization strategies & weight sensitivity.
- Long Clause Benefit: Performance delta with and without chunked encoding on overlength inputs.
Instrumentation TODO:
- Log average / max chunk count per clause.
- Record projection usage rate (percentage requiring dimension mapping).

## Limitations
- Cross encoder lacks runtime `--max-len-override` symmetry (current override only applied in bi path).
- LLM direct / PMI scoring path still exploratory; latency & stability not tuned for production scale.
- No persistent on‑disk embedding cache or ANN index (in‑memory only); scaling beyond moderate problem sets may require FAISS / disk caching.
- Chunk aggregation 'attn' currently lightweight; could be replaced with trainable transformer over chunk embeddings.

## Future Work
1. Cross Encoder Override: Add unified truncation override & optional chunk support.
2. Telemetry: Expose chunk usage, projection stats, and per‑stage latency to structured logs.
3. ANN Integration: Precompute bi embeddings + FAISS index for sub‑millisecond top‑K retrieval.
4. Advanced Fusion: Learnable weight optimization or gating network conditioned on query features.
5. LLM Ranking Stabilization: Consolidate token‑logprob and PMI methods; prune ineffective LLM head approach.
6. Distillation Extensions: Multi‑teacher ensemble distillation (cross + PMI) with uncertainty weighting.
7. Feature‑Aware Encoding: Incorporate structured symbolic features via prefix tokens more aggressively (curr. shallow prefix only).
8. Persistent Caching: Clause embedding store with versioned invalidation.
9. Mixed Precision & Throughput: AMP + gradient checkpointing and flash attention for deeper encoders.
10. Reproducibility Suite: Scripted benchmark harness capturing seeds, flag snapshots, model hashes, and protocol transcripts.

## Reproducibility Checklist
- Max sequence length (training): 2048 tokens for both bi and cross encoders.
- Runtime override flag: `--max-len-override` (applies to bi encoder chunk logic).
- Chunk params: (len, stride, max, agg) explicitly logged in run config.
- Checkpoint key flexibility ensures legacy models load without manual edits.
- Projection layer creation deterministic (Xavier init, single instantiation per dimension pair).

## Summary
The system delivers a resilient, extensible neural reasoning assistant: adaptive long‑sequence handling via chunked encoding, dynamic embedding dimension reconciliation, multi‑backend scoring orchestration, and 2048‑token training context. Pending enhancements focus on symmetry (cross override), richer telemetry, scalable retrieval infrastructure, and stabilization of LLM scoring variants.
