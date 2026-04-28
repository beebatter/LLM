# iProver LLM External Agent Whitepaper (Experimental Progress and Training Improvement Plan)

## Background and Goals

- Goal: Use an LLM as iProver’s external re-ranker. Given “conjecture + K candidate clauses,” output a per-candidate score distribution to guide given-clause selection and clause ordering, thereby improving proof efficiency.
- Two routes:
  - LLM reranking (listwise learning; outputs a distribution/scores).
  - Classic Transformer stack (Bi-Encoder retrieval + Cross-Encoder rerank) as teacher and for fusion.
- Output protocol:
  - Online we prefer returning per-candidate scores directly (ideally a softmax distribution or a dedicated scoring head), avoiding the fragility of JSON parsing.

## Current Experimental Setup (Data / Models / Scripts)

Data
- Listwise teacher-fusion data:
  - Train: `/root/autodl-tmp/Training/datasets/listwise/train.listwise.teacher.jsonl`
  - Val: `/root/autodl-tmp/Training/datasets/listwise/val.listwise.teacher.jsonl`
- Generation: `scripts/build_listwise.sh` (supports Cross+Bi teacher fusion: softmax(λ_ce·s_ce + λ_bi·s_bi; τ)).

Models and Training
- 7B: `DeepSeek-Prover-V2-7B` (local), LoRA, 2 epochs, bf16 / 8bit / none; max_len ≈ 2048.
- 32B: `Goedel-Prover-V2-32B` (local), QLoRA (4bit), 2 epochs; max_len ≈ 1024.
- Training script: `training/train_llm_listwise_sft.py`
  - Key robustness: pad→eos, resize_token_embeddings, ignore pad/out-of-range in labels, use_cache=False, gradient checkpointing.
  - Quantization switch: `--quant {4bit,8bit,none}`.

Evaluation and Reranking
- Listwise consistency evaluation (direct softmax): `training/eval_llm_listwise.py --mode direct` (independent forward pass per candidate; sum of per-token log-likelihood → softmax).
- Retrieval evaluation (Bi/CE/LLM/Fusion):
  - Generate CE/LLM reranks: `select/rerank_with_cross_encoder.py`, `select/rerank_with_llm.py`.
  - Evaluate fusion: `select/eval_retrieval.py` (supports λ grid search).

## Results Summary (Offline)

Listwise consistency (alignment with teacher targets; val subset n ≈ 102)
- 7B: len_match = 1.0, MAE ≈ 0.030, MSE ≈ 0.0137, Pearson ≈ 0.05.
- 32B: len_match = 1.0, MAE ≈ 0.030, MSE ≈ 0.0137, Pearson ≈ 0.27.

Retrieval metrics (LLM as reranker; same queries/labels/meta, n ≈ 27)
- Bi vs CE: CE generally outperforms Bi.
- LLM 7B: Early-K Hit@10 ≈ CE; overall close.
- LLM 32B: Slightly weaker than CE at early K; approaches Bi at mid/deep K.
- Bi+CE fusion λ grid (fix λ_ce = 1.0):
  - Best Hit@10: λ_bi = 0.1 (Hit@10 ≈ 0.333, NDCG@32 ≈ 0.242).
  - Best NDCG@32: λ_bi = 0.2 (NDCG@32 ≈ 0.249, Hit@10 ≈ 0.259).

Short conclusion
- As-is, LLM reranking is not yet stably better than CE/Bi (7B at early K ≈ CE; 32B early K weaker; listwise direct eval correlation ≈ 0 for 7B, ≈ 0.27 for 32B).
- This looks like “fine-tuning not yet aligned/evaluation not fully calibrated,” not a fundamental model limitation.

## Diagnosis and Causes

- Objective-metric mismatch: Training fits a teacher soft distribution (listwise CE+Bi), while evaluation emphasizes Hit@K/NDCG. Add ranking objectives (ListMLE/Pairwise) or distill CE logits (KL).
- Length bias: Direct eval uses “sum log-likelihood,” which penalizes longer candidates; use “mean per-token log-likelihood” or add a length penalty.
- Context truncation: Input_max/target_max were reduced to avoid OOM, losing critical information; with H800, enable flash-attn/TF32 and try 2048/256.
- Quantization gap: 7B trained in bf16 but evaluated in 4bit reduces performance; try 8bit/none baselines for 7B.
- Training signal skewed to “template imitation”: Positives evenly weighted; lacks strength differentiation among positives and candidate-level signals; loss operates at token level rather than candidate level.
- Data size and hyperparameters: Only 2 epochs; LoRA r/α/dropout/scheduler need tuning; increase hard negative ratio and data augmentation.
- Prompt consistency: Re-verify training/eval prompts match exactly (delimiters/fields/order).

## Implementation Plan for Training Improvements (Repo-aligned)

1) Evaluation protocol calibration (wins without changing the model)
- Switch scoring to “mean LL”: `s_mean = (1/|c|) Σ log p(x_t | prompt)`; add `--score-type {sum-ll, mean-ll}`.
- Per-candidate independent forward pass: avoid attention leakage among K candidates; keep current `--mode direct`, ensuring one forward per candidate.
- Within-window normalization: z-score over {s_i} or subtract median/divide by MAD (add `--zscore`).
- Temperature/bias calibration: fit p̂ = softmax((s − b)/τ) on dev; add `--calib-tau/--calib-bias`.
- Optional PMI: `s_PMI = log P(c | conj) − λ·log P(c)` (λ ∈ [0.5, 1.0]); requires an unconditional baseline forward.

2) Candidate-level objective learning (replace “JSON SFT”)
- Add a “candidate scoring head” to the model: wrap each candidate with `<CAND_START> … <CAND_END>`, take the `<CAND_START>` hidden state h_i, feed a linear head to get s_i; p̂ = softmax(s).
- Loss: cross-entropy/KL to align with teacher soft distribution p*; optionally mix in a small pairwise (Bradley–Terry/margin) as regularization.
- Implementation: add a `--score-head` mode to `training/train_llm_listwise_sft.py` or introduce a new script `train_llm_listwise_scorehead.py`; apply LoRA to top layers and the scoring head to reduce VRAM and instability.

3) Enhance teacher distribution (sharpen positive-internal shape)
- Use dev grid search over (pos_mass m, temperature τ, blend α) with `target_dist_tuner.py`, generating sharper target_scores:
  - pos_weights = (1 − α) · uniform + α · softmax(teacher/τ);
  - Apply only to positive subset; distribute remaining mass uniformly across negatives; add ε smoothing and renormalize.
- Pipeline: extend `scripts/build_listwise.sh` to read the best YAML and rewrite `train/val.listwise.teacher.jsonl` target_scores (without changing sample bodies).

4) Training polish (current recipe tweaks)
- Learning rate: for 7B LoRA use 5e-5 ~ 1e-4; for 32B QLoRA use 5e-5; warmup_ratio ≈ 0.03; cosine schedule; epochs 2–3.
- Candidate randomization: shuffle candidate order each epoch; shuffle targets accordingly.
- Hard negatives: raise `NEG_given_nonproof` proportion (e.g., 0.6) to improve early-K hits.
- Logits distillation: directly distill the CE K-dim raw scores with τ = 1–2; richer than discrete soft distributions.
- Performance: on H800 enable flash-attn/TF32, use_cache=False, gradient checkpointing; for 7B evaluate with 8bit/none as controls.

5) Fusion and evaluation (with Bi/CE)
- Fusion: S = λ_ce·S_ce + λ_bi·S_bi + λ_llm·S_llm; allow per-query normalization (min-max/z-score/RRF).
- Evaluation: Hit@{10,32,64}, NDCG@{10,32,64}, Recall@K; pick λ on dev with a small grid.

6) Two-day landing plan
- Day 1:
  - Evaluation protocol: `--score-type mean-ll`, `--zscore`, `--calib-*`; optional PMI; on H800 raise to input_max=2048/target_max=256 and re-evaluate.
  - Target distribution: run `target_dist_tuner.py` to search m, τ, α; rewrite target_scores in `train/val.listwise.teacher.jsonl`.
- Day 2:
  - 7B: LoRA lr=1e-4, epochs=2–3, listwise CE/KL (or score-head); randomized candidates; hard negative ratio 0.6; evaluate with 8bit/none as controls.
  - 32B: QLoRA lr=5e-5, epochs=2, same as above.
  - Evaluation: listwise consistency (mean-LL), Bi top-200 rerank (Hit/NDCG), three-way fusion small grid; fix best λ.

## Operational Guide (Optional Reference)

- Generate listwise (with teacher fusion): `bash scripts/build_listwise.sh` (set CROSS/BI/SPM paths, τ, λ_bi as needed).
- Train 7B (single GPU) example:
  - `CUDA_VISIBLE_DEVICES=0 python training/train_llm_listwise_sft.py --model /root/autodl-tmp/models/DeepSeek-Prover-V2-7B --train .../train.listwise.teacher.jsonl --val .../val.listwise.teacher.jsonl --out .../deepseek7b-listwise-lora --batch 2 --grad-accum 8 --epochs 2 --lr 1e-4 --max-len 2048 --bf16 --quant 8bit`
- Train 32B (single GPU) example:
  - `CUDA_VISIBLE_DEVICES=1 python training/train_llm_listwise_sft.py --model /root/autodl-tmp/models/Goedel-Prover-V2-32B --train ... --val ... --out .../goedel32b-listwise-lora --batch 1 --grad-accum 16 --epochs 2 --lr 8e-5 --max-len 1024 --bf16 --quant 4bit`
- Listwise evaluation (direct):
  - `python -m LLM.training.eval_llm_listwise --data .../val.listwise.teacher.jsonl --mode direct --model <base> --lora <adapter_dir> --bits 4 --input-max 1536 --target-max 192`
- LLM reranking → retrieval evaluation:
  - `python -m LLM.select.rerank_with_llm ...` → `python -m LLM.select.eval_retrieval ... --lambda-ce ... --lambda-bi ...`

## Artifacts and Paths (Current)

- Data: `/root/autodl-tmp/Training/datasets/listwise/{train,val}.listwise{.teacher}.jsonl`
- Models:
  - 7B LoRA: `/root/autodl-tmp/Training/models/deepseek7b-listwise-lora`
  - 32B LoRA: `/root/autodl-tmp/Training/models/goedel32b-listwise-lora`
- Evaluation:
  - LLM direct listwise metrics: `eval_listwise_direct.json` under each model’s out dir
  - Retrieval evaluation: `metrics_llm7b.json`, `metrics_llm32b.json` (comparable to Bi/CE)

---

Conclusion: LLM reranking is usable today but not yet consistently better than CE/Bi. Prioritize evaluation calibration and sharpening target distributions, then introduce a candidate-level scoring head and ranking losses; also run a small fusion grid among Bi/CE/LLM to lift mid/deep NDCG. With two H800s, the above plan can complete one closed loop in two days and re-validate metrics.


# iProver–LLM Parallel Plan Whitepaper (incl. dataset generation details + two shippable options)
> Goal: Based on your existing code (`run_batch_pipeline.py / process_iprover_v3.py / iplog_to_dataset.py / batch_ranker.py`), implement and evaluate two parallel routes:
> Plan A (All Transformer): Transformer → Vector Retrieval (Bi-Encoder) → Transformer Cross-Encoder Rerank.
> Plan B (Hybrid): Transformer → Vector Retrieval (Bi-Encoder) → Local LLM (DeepSeekMath‑32B, LoRA) Rerank.
> Emphasis on dataset/corpus quality and reproducible metrics.

---

# Project Background and Overview (Introduction)

Project Objective: Use LLM + Transformer encoders to provide Prover Guidance for saturation-based automated theorem provers (e.g., iProver) to improve given-clause selection quality and online efficiency, without changing the prover kernel’s soundness. Ultimately improve “problems solved within a fixed timeout” and reduce average given steps/latency.

Background: Saturation-based ATPs (E/iProver/Vampire) rely on the given-clause loop, facing candidate explosion, weak heuristic signals, and poor cross-domain transfer. We adopt a two-stage strategy:
- SELECT (Recall): Train a contrastive Bi‑Encoder to independently encode the target conjecture q and a candidate clause d and use vector similarity for fast Top‑K retrieval over a large corpus.
- RERANK: Apply a finer-grained scoring to the SELECT candidates. Two parallel implementations:
  1) Plan A: Transformer Cross‑Encoder reranking.
  2) Plan B: Local LLM (DeepSeekMath‑32B, QLoRA) listwise reranking.

Scope and Constraints:
- Do not modify iProver’s proof logic or correctness; only influence “given clause priority” via an external agent.
- Keep the LLM’s native tokenizer unchanged; train a dedicated tokenizer only for our own Transformers (Bi-/Cross-Encoder) and use light normalization (variable normalization/structural prefixes/signature hints).
- Keep heuristic safety nets (oldest/lightest/random exploration) for robustness and fallback.

---

## Project Structure (Scripts and Directories)

## 1. Code Layout (by responsibility)
- `process_iprover_v3.py` (online EA service): Connects to iProver, collects windows (register/passive/given/simplified), constructs batches on scores_req → prefilter → call rerank backend → respond; optional SAT-grounding assistance and score cache.
- `batch_ranker.py` (unified rerank backend): Wraps backends (heuristic/future cross_encoder/llm_local/llm_api), handles chunking, score parsing, and fallback.
- `run_batch_pipeline.py` (offline batch runner): Orchestrates iProver+EA; outputs raw NUL‑delimited logs, supervised JSONL dataset, and failures; contains negative bucketing and frontier/born sampling strategies.
- `iplog_to_dataset.py` (logs → dataset): Parses CNFRefutation and run traces to write `text, features, label, neg_bucket`; recommended to add `conjecture_text` for the Bi‑Encoder.

## 2. Online/Offline Data and Control Flow
```
iProver ──register/passive/given/simplified──▶ EA
      └────────────scores_req(ids)────────────▶  prefilter → batch_ranker(backend) → scores_res
EA ──szs_result_out/server_queries_end───────▶ iProver

run_batch_pipeline.py → Logs/*.raw.log → iplog_to_dataset.py → datasets/*.jsonl / failed.jsonl
```

## 3. Capabilities Implemented
- Online: window management, prefiltering (target functor, unit clause, shared constants, length penalty, etc.), score cache, silent/truncated logs, optional SAT grounding.
- Offline: negative bucketing (given_nonproof/simplified/passive_only/never_seen), frontier/born sampling, failure set (with last given/passive clause text and features).

## 4. Attachment Points for the Two Plans
- Plan A (Transformer two-stage):
  - Hook SELECT (Bi‑Encoder) before `_ea_handle_scores_req`; pass Top‑K into `batch_ranker.py --backend cross_encoder`.
  - EA-side fusion `S = λ_bi·S_bi + λ_ce·S_ce + λ_h·S_heur` and mixing with oldest/lightest/random.
- Plan B (SELECT + LLM rerank):
  - Same SELECT; pass Top‑K into `--backend llm_local` (local LoRA LLM listwise scoring).
  - Optionally distill LLM scores into Cross‑Encoder to reduce cost.

## 5. Minimal Modification Points
- `iplog_to_dataset.py`: add `conjecture_text`, unify bucket priority, output `sample_weight`.
- `run_batch_pipeline.py`: parameterize negative sampling and bucket ratios, output `dataset_report.json` and `problem_splits.json`.
- New: `train_sentencepiece.py / train_biencoder.py / encode_and_build_index.py / select_service.py / train_cross_encoder.py / make_listwise_chunks.py / train_llm_lora.py / eval_rank_metrics.py`.
- `process_iprover_v3.py`: insert SELECT; `--pipeline {A_transformer_ce, B_llm_rerank}`; record Top‑K recall.
- `batch_ranker.py`: add `cross_encoder/llm_local/llm_api` backends and score fusion.

## 6. Evaluation and Rollout Order (Summary)
- Offline: Recall@K for SELECT; NDCG@K for RERANK.
- Online: number of solved problems within timeout, average given count, latency/GPU.
- Order: Dataset Gate → Tokenizer → Bi‑Encoder → Index service → Cross‑Encoder/LLM → Fusion/Scheduler → A/B.

> Goal: Based on your existing code (`run_batch_pipeline.py / process_iprover_v3.py / iplog_to_dataset.py / batch_ranker.py`), implement and evaluate two parallel routes:
> Plan A (All Transformer): Transformer → Vector Retrieval (Bi-Encoder) → Transformer Cross-Encoder Rerank.
> Plan B (Hybrid): Transformer → Vector Retrieval (Bi-Encoder) → Local LLM (DeepSeekMath‑32B, QLoRA) Rerank.
> Emphasis on dataset/corpus quality and reproducible metrics.

---
## Table of Contents
1. Project overview and plan comparison
2. Corpus (dataset) construction: from raw logs to train set (focus)
3. Plan A: All Transformer (SELECT + Cross-Encoder RERANK)
4. Plan B: Bi‑Encoder SELECT + Local LLM RERANK (DeepSeekMath‑32B, QLoRA)
5. Unified evaluation (offline & online)
6. Engineering rollout: change list, CLI, service, resource estimate
7. Risks and fallback, milestones and acceptance

---
> This section summarizes responsibilities and interactions for “existing scripts + planned scripts.” See other sections for CLI details.

### A. Core online path

- `process_iprover_v3.py` (EA bridge)
  Encodes/decodes messages between iProver and the external agent, logs interactions; on `scores_req` runs SELECT (Bi‑Encoder retrieval) and RERANK (Cross‑Encoder or LLM), outputs scores/fusion results; records recall and scheduling details.

- `batch_ranker.py` (unified rerank backend)
  Wraps multiple scoring backends: `cross_encoder / llm_local / llm_api / heuristic`; supports score fusion and fallback; called by `process_iprover_v3.py`.

### B. Batch processing and data loop

- `run_batch_pipeline.py` (offline batch + data collection)
  Runs iProver+EA in batch; generates `*.raw.log`, `datasets/*.jsonl`, and `failed.jsonl`; unifies failure reasons and sample statistics.

- `iplog_to_dataset.py` (logs → supervised samples)
  Parses CNFRefutation/given traces; outputs samples with `label (pos/neg/bucket)` and `features`; supports adding `conjecture_text`, weak-negative weighting, and bucket strategy.

### C. SELECT/retrieval and training (planned/extended scripts)

- `train_sentencepiece.py` (Tokenizer training)
  Train SentencePiece (Unigram/BPE) for custom Transformers only; LLM uses its native tokenizer.

- `train_biencoder.py` (SELECT training)
  Train a dual-tower model with contrastive learning (InfoNCE + hard negatives). Export `best.pt` and SPM.

- `encode_and_build_index.py` / `select_service.py` (vector index and online retrieval)
  Offline encode static corpus + per-problem incremental pool (FAISS/HNSW). Provide online Top‑K retrieval.

- `train_cross_encoder.py` (Plan A rerank)
  Train cross-encoder with binary/pairwise/listwise objectives; online batched forward.

- `make_listwise_chunks.py` / `train_llm_lora.py` (Plan B rerank)
  Generate listwise windows (K=64) and fine-tune a local LLM with QLoRA; used by `batch_ranker.py --backend llm_local`.

- `eval_rank_metrics.py` (unified evaluation)
  Offline: Recall@K, NDCG@K, MAP/AUC; online: solved count, average given, latency and resource use.

### D. Assets and directories

- `datasets/`: supervised samples and splits (train/val/test); failure set (weak negatives only).
- `models/`: tokenizer, Bi‑Encoder, Cross‑Encoder, LoRA adapters, etc.
- `indexes/`: FAISS/HNSW vector index and caches.
- `Logs/`: EA raw interaction logs and batch outputs.

### E. Compatibility and legacy (from initial README)
- `auto_fof_corpus_builder_local.py` / `html_fof_extractor.py`: earlier integration and problem-list extraction tools (now largely replaced by batch/data pipeline; still useful for generating lists and HTML extraction).
- `process_iprover_v2.patched.py` / `setup_and_run_fof.sh`: historical EA version and one-click script; kept as backup.

---

## 1) Project overview and parallel plans

### 1.1 Shared architecture (common to both plans)
- SELECT (recall): Contrastive Bi‑Encoder encodes q and d independently and retrieves Top‑K by cosine or dot product.
- RERANK: Refines Top‑K scores and returns final priority; mix with oldest/lightest/random exploration before feeding iProver’s given selector.
- Data loop: iProver↔EA (`process_iprover_v3.py`) logs → `iplog_to_dataset.py` labels (pos/neg/bucket) → training → EA online integration → A/B evaluation → continue logging and distillation.

### 1.2 Two parallel plans
- Plan A (all Transformer)
  - STEP1: Bi‑Encoder SELECT (fast Top‑K recall)
  - STEP2: Transformer Cross‑Encoder RERANK (smaller specialized cross-encoder; input like `[CLS] <Q> … </Q> <D> … </D>`; one forward per (q,d); high precision at lower cost than LLM)
- Plan B (hybrid)
  - STEP1: Bi‑Encoder SELECT (same as above)
  - STEP2: Local LLM (DeepSeekMath‑32B, QLoRA) listwise reranking via `batch_ranker.py --backend llm_local`.

> Develop both in parallel and compare on unified offline/online metrics. Plan A is lighter and easier to deploy; Plan B has a higher accuracy ceiling but higher cost.

---

## 2) Corpus (dataset) construction: from raw logs to training set (focus)

### 2.1 Data sources and toolchain
- Raw interaction logs: `run_batch_pipeline.py` runs problems to produce `*.raw.log` (NUL‑delimited JSON), plus `datasets/*.jsonl` and `failed.jsonl` (failed problem metadata).
- Parsing and labeling: `iplog_to_dataset.py` extracts positives from CNFRefutation (c_# appearing in the proof) and buckets other clauses by source/state into negatives:
  - `NEG_given_nonproof`: was given but not used in the proof.
  - `NEG_simplified`: simplified/eliminated.
  - `NEG_passive_only`: only appeared in passive set.
  - `NEG_never_seen`: missing registration/exception (may be empty).
- Failures: `failed.jsonl` used only as weak negatives (e.g., last_given_clauses/passive_sample) and must NOT enter val/test.

### 2.2 Sample fields (strongly recommended)
Each JSON line:
```json
{
  "problem_name": "ALG050+1",
  "conjecture_text": "...",          
  "text": "clause/cnf text ...",
  "features": {"horn":1,"epr":1,"unit":0,"born":12,"conj_dist":2,...},
  "label": 1,                        
  "neg_bucket": null,                
  "source": "run_batch_pipeline",   
  "sample_weight": 1.0               
}
```
> New field: `conjecture_text` (or `conjecture_sig`) for Bi‑Encoder training and online SELECT.

### 2.3 Quality gates (acceptance)
1) Schema complete: all fields present; positives have `neg_bucket = null`.
2) Split by problem: train/val/test grouped by `problem_name`; no cross-split leakage.
3) Negative sampling and bucket ratios (train set): positive:negative ≈ 1:3–1:5; suggested bucket mix:
   `given_nonproof ≈ 50% / simplified ≈ 20% / passive_only ≈ 25% / never_seen ≈ 5%`.
4) Dedup: val/test must not contain the same `text` as train; cross-problem duplicates should preferably be in train or down-weighted.
5) Length and normalization: variable normalization; training max sequence length (e.g., 256 tokens); drop a portion of extremely long entries.
6) Failure samples: only for weak negatives in train with low weight; per-problem cap (50–200).

### 2.4 Preprocessing and tokenization
- SentencePiece 16k–32k; include `()=,~|&` as user-defined symbols; normalize variables to `VAR`.
- Prefix structural features as tokens: discretize `features` into a text prefix like `<H1><U0><E1><C2><B3> … clause …`; bucketize `conj_dist/born` before mapping.

### 2.5 Produce two kinds of training sets
- SELECT (Bi‑Encoder):
  - Form A: `(q, d+, {d-…})` for InfoNCE (in-batch all negatives + periodic hard negatives).
  - Form B: pairwise `(q, d+, d-)` for margin/softmax.
- RERANK:
  - Cross‑Encoder: binary/pairwise (BPR/hinge)/listwise (use small windows, e.g., 16–32).
  - LLM (DeepSeekMath‑32B): listwise (window=64) with entries like `{"ids":[...], "texts":[...], "target":[softmax distribution]}`.

### 2.6 Command patterns (examples)
```bash
# (1) Generate supervised samples (with conjecture_text)
python3 iplog_to_dataset.py \
  --raw-dir Logs/EA.* \
  --out datasets/clauses.jsonl \
  --add-conjecture \
  --bucket-priority given_nonproof,simplified,passive_only,never_seen

# (2) Split by problem + dedup + resample buckets + inject weak negatives (failures)
python3 make_splits.py \
  --input datasets/clauses.jsonl \
  --failed datasets/failed.jsonl \
  --train-out datasets/train.jsonl \
  --val-out datasets/val.jsonl \
  --test-out datasets/test.jsonl \
  --neg-ratio 4 \
  --bucket-quota 0.5,0.2,0.25,0.05 \
  --weak-failed-cap 100 --weak-weight 0.25

# (3) Vocab and feature prefixes
python3 train_sentencepiece.py --input datasets/train.jsonl --vocab 24000 --out models/sp/

# (4) Generate listwise samples for RERANK
python3 make_listwise_chunks.py \
  --input datasets/train.jsonl \
  --window 64 --smooth 0.1 \
  --out datasets/train_listwise.jsonl
```

---

---

## 2.7 Tokenizer strategy and “name invariance” (only train for custom Transformers; keep LLM native)

> We only train a tokenizer for our custom Transformers (Bi-/Cross-Encoder). LLMs (e.g., DeepSeekMath‑32B) strictly use their native tokenizer. For LLMs, apply only light text normalization (variable normalization, structural prefixes, signature hints); do not change the vocab.

### 2.7.1 Which tokenizer to train?
- Train: tokenizer for the custom Transformers (SentencePiece Unigram or BPE).
- Do NOT train: any tokenizer for large LLMs (keep native to ensure fine-tune/inference compatibility).

### 2.7.2 Training corpus (for tokenizer only, not supervised labels)
Create a plain-text file (one entry per line) from `interactive_sampled_small.jsonl` (or a larger set) by concatenating:
- `conjecture_text` (or `conjecture_sig`)
- `text` (clause)
- Structural feature prefix (see §2.4), placed at the beginning of each line

Preprocessing rules:
- Variable normalization: unify all first-order variables as `VAR` (e.g., X0,X1,... → VAR).
- Keep logic/parenthesis symbols: `()=,~|&` as standalone tokens.
- Truncate extremely long rare symbols: >40 chars → `<SYM_LONG>` (optional).
- Include structural prefix, e.g., `<H1><U0><E1><C2><B3> ...` in the same line.

### 2.7.3 SentencePiece training command (Unigram recommended)
```bash
spm_train \
  --input=corpus_for_tokenizer.txt \
  --model_prefix=spm_logic \
  --vocab_size=24000 \
  --model_type=unigram \
  --character_coverage=1.0 \
  --input_sentence_size=5000000 --shuffle_input_sentence=true \
  --user_defined_symbols='(<Q>,</Q>,<D>,</D>,<H0>,<H1>,<U0>,<U1>,<E0>,<E1>,<C0>,<C1>,<C2>,<C3>,<B0>,<B1>,<B2>,<B3>,VAR,(,),=,~,&,|,"," )' \
  --unk_piece=<UNK> --pad_piece=<PAD>
```
Key points:
- Add `<Q>, </Q>, <D>, </D>`, all structural prefixes, and `VAR`, parentheses and logic symbols to `--user_defined_symbols` so they become independent tokens.
- `vocab_size` suggested 16k–32k; start with 24k.
- If clauses are long, consider increasing `vocab_size` or adding more `user_defined_symbols`.

Acceptance checks (must pass):
- Parentheses and logic symbols remain independent tokens; variables mostly become `VAR`.
- Average tokenized length for train/val ≤ 160 tokens.
- OOV near 0 (Unigram is robust and usually satisfies this).

### 2.7.4 How to handle “name invariance” (α‑equivalence)?
- Strongly recommend: only variable normalization (`X* → VAR`) — high gain, low effort.
- Predicate/functor names: no global anonymization (high coupling/complexity; may lose useful weak semantics). Use a middle ground:
  1) Signature prefix: add `<SIG: f/2,g/1,h/3>` (can be hashed) to the sample prefix;
  2) Data augmentation: for 20–30% of batches, locally rename symbols (consistent within each sample) to improve robustness;
  3) Symbol dropout: with 5–10% probability replace rare symbols with `<SYM_UNK>` (apply to both pos and neg).
- On the LLM side: keep tokenizer unchanged, only add normalized views in the prompt, e.g.:
  ```
  <Q_SIG: f/2,g/1; VARS:5> conjecture...
  <D_SIG: f/2,h/2; VARS:3> clause...
  ```

### 2.7.5 Ablations and metrics (suggest 3 variants)
1) Baseline: no variable normalization.
2) Var‑Norm: variable normalization + structural prefix.
3) Var‑Norm + Aug: Var‑Norm + signature prefix + 20% random renaming augmentation.

Offline metrics: Bi‑Encoder R@32/64, MAP; Cross/LLM NDCG@K, Pairwise Acc.
Online metrics: problems solved, average given, latency.

Conclusion in advance: training a tokenizer only for custom Transformers is necessary and beneficial; keeping the LLM tokenizer native significantly reduces engineering risk and complexity.

## 3) Plan A: All Transformer (SELECT + Cross‑Encoder RERANK)

### 3.1 Bi‑Encoder (SELECT) training
- Model: twin-tower Transformer (6–8 layers, hidden 512–768; shared or not). Pooling: `[CLS]` or mean.
- Loss: InfoNCE (τ = 0.05–0.1); in-batch all negatives; periodically mine hard negatives (use high-scoring mistakes).
- Optimization: AdamW(lr=1e‑4, wd=0.01), mixed precision; early-stopping by val Recall@K (K=32/64).
- Export: `best.pt` + `spm.model`.

### 3.2 FAISS index and online SELECT
- Static index: encode common axioms/common clauses offline (HNSW/IVF‑PQ), read-only.
- Per-problem pool: encode and append newly registered clauses from EA; maintain a `text→vector` LRU cache.
- EA integration: call SELECT before `scores_req` → return Top‑K (e.g., 512/1024) candidates for reranking.

### 3.3 Cross‑Encoder (Transformer) RERANK training
- Input: concatenated `(q,d)`; template: `"[CLS] <Q> {q_text} </Q> <D> {d_text} </D>"`.
- Objective: binary (logit→sigmoid) or pairwise (BPR/hinge) or listwise (small windows, e.g., 16–32).
- Hyperparams: 6 layers, hidden 512, max length 256; AdamW 5e‑5; early-stopping by AUC / NDCG@K.
- Online inference: forward once per `(q,d)` in Top‑K (batched); wrapped by `batch_ranker.py --backend cross_encoder`.

### 3.4 Score fusion and scheduling
- Fusion: `S = λ1·S_cross + λ2·S_bi + λ3·S_heur`; grid search λ on the validation set.
- Scheduling: `Top‑M(cross)` + `Oldest M1` + `Lightest M2` + `ε random`.

---

## 5) Unified evaluation and comparison metrics

### 5.1 Offline (aggregate per problem)
- SELECT: Recall@{10,32,64}, MAP; latency (vectorization + retrieval).
- RERANK: NDCG@K, MAP, AUC, Pairwise Acc, Kendall’s τ; tokens/latency per 64‑candidate batch.
- End‑to‑end simulation: on saved candidate sets simulate “SELECT → RERANK → schedule” for the first N steps, report Top‑1~K positive hit rate and cumulative gains.

### 5.2 Online (fixed problem set)
- Primary: problems solved (within T seconds), average given count, average latency.
- Secondary: SELECT hit rate (probability that a true proof clause is recalled), RERANK lift (positive proportion within Top‑M).
- Efficiency: GPU/CPU utilization, VRAM, throughput (clauses/sec).
- Statistics: bootstrap 95% CI; keep per‑problem JSON reports for reproducibility.

---

## 6) Engineering rollout: changes, CLI, service, resource estimates

### 6.1 Minimal change list
1) `iplog_to_dataset.py`
   - Add `--add-conjecture` to output `conjecture_text`/`conjecture_sig`.
   - Unify bucket priority: `given_nonproof > simplified > passive_only > never_seen`.
   - Output `sample_weight` (bucket‑based weights).
2) `run_batch_pipeline.py`
   - Parameterize negative quotas and bucket target ratios; output `dataset_report.json`.
   - Produce `problem_splits.json` (problem sets for train/val/test).
3) New scripts
   - `make_splits.py / train_sentencepiece.py / train_biencoder.py / encode_and_build_index.py / select_service.py / make_listwise_chunks.py / train_cross_encoder.py / train_llm_lora.py / eval_rank_metrics.py` (templates can be bootstrapped).
4) `process_iprover_v3.py`
   - Insert SELECT before `_ea_handle_scores_req`.
   - Add `--pipeline`: `"A_transformer_ce"` (Plan A) / `"B_llm_rerank"` (Plan B).
   - Record Top‑K recall and scheduling details.
5) `batch_ranker.py`
   - Extend `--backend`: `cross_encoder | llm_local | llm_api`.
   - `--blend "llm=0.6,bi=0.3,heur=0.1"`; unified failure fallback.

### 6.2 CLI chain examples
```bash
# Data prep
python3 iplog_to_dataset.py --raw-dir Logs/ --out datasets/clauses.jsonl --add-conjecture
python3 make_splits.py --input datasets/clauses.jsonl --failed datasets/failed.jsonl \
  --train-out datasets/train.jsonl --val-out datasets/val.jsonl --test-out datasets/test.jsonl

# Train Bi‑Encoder (SELECT)
python3 train_biencoder.py --train datasets/train.jsonl --val datasets/val.jsonl \
  --spm models/sp/spm.model --epochs 6 --batch 256 --neg-per-pos 4 --hard-negative

# Build index and service
python3 encode_and_build_index.py --model models/biencoder/best.pt --spm models/sp/spm.model \
  --docs datasets/train.jsonl --out indexes/global.faiss
python3 select_service.py --index indexes/global.faiss --model models/biencoder/best.pt --spm models/sp/spm.model

# Plan A: Cross‑Encoder rerank
python3 train_cross_encoder.py --train datasets/train.jsonl --val datasets/val.jsonl \
  --spm models/sp/spm.model --epochs 3 --maxlen 256
python3 process_iprover_v3.py serve --pipeline A_transformer_ce --select-top-k 1024 \
  --select-service http://127.0.0.1:9000 --ranker-backend cross_encoder --blend "llm=0.0,bi=0.3,heur=0.7"

# Plan B: LLM rerank (DeepSeekMath‑32B, QLoRA)
python3 make_listwise_chunks.py --input datasets/train.jsonl --out datasets/train_listwise.jsonl --window 64
python3 train_llm_lora.py --data datasets/train_listwise.jsonl --base deepseek-math-32b --out models/ds32b-lora/
python3 process_iprover_v3.py serve --pipeline B_llm_rerank --select-top-k 1024 \
  --select-service http://127.0.0.1:9000 --ranker-backend llm_local --blend "llm=0.6,bi=0.3,heur=0.1"
```

### 6.3 Resource estimates
- Bi‑Encoder training: single 24GB GPU is sufficient (batch 256 may need grad accumulation), 6–8 layers/hidden 512.
- Cross‑Encoder training: single 24GB GPU; online RERANK: Top‑K=512 → ≈512 forwards per batch (batchable).
- LLM QLoRA: 40–48GB VRAM (32B base, QLoRA); inference can use vLLM tensor parallelism.

---

## 7) Risks and fallback, milestones and acceptance

- Risks: LLM parsing failures/latency; insufficient SELECT recall; dataset shift (e.g., too many passive-only negatives).
- Fallback: if RERANK fails then `S = S_bi`; if SELECT service fails, send directly to RERANK with smaller K; always keep oldest/lightest/random.
- Milestones:
  - Milestone 1 (P0): SELECT online + Plan A beats heuristic by ≥15% in R@64 offline.
  - Milestone 2 (P1): Plan B improves online solved count by ≥10% (same timeout).
  - Milestone 3 (P1): After distillation + hard negatives, SELECT recall improves by ≥5%.
- Acceptance: submit `metrics_offline.json` and online A/B reports (with confidence intervals).

---

> The minimal code changes above come with runnable command chains. You can first land Plan A (no LLM service needed, low engineering friction) while preparing Plan B’s QLoRA data and scripts in parallel to run comparative evaluation within two weeks.

---

## Completed (Plan A alignment progress and next steps)

### Overview
- Completed: §3.1 Bi‑Encoder training, §3.2 index construction (offline), §3.3 Cross‑Encoder training and offline rerank/evaluation.
- To do: §3.2 online integration (EA wiring), §3.4 score fusion and scheduling, plus per-problem incremental pool and LRU cache.

### 3.1 Bi‑Encoder (SELECT) training
- Done
  - Twin-tower Transformer; SupCon (group positives by problem) contrastive learning; AMP + AdamW + warmup+cosine.
  - Validation by problem Recall@K (K=32/64); R@64 ≈ 0.926 (stable).
  - Exported best.pt and SentencePiece spm.model.
- Partial
  - Periodic hard negative mining not yet integrated (currently only in-batch negatives and natural hard cases).
- Note
  - SupCon and InfoNCE are similar objectives; can continue as-is and add hard negatives.

### 3.2 FAISS index and online SELECT
- Done
  - Offline encoded and built index (Flat/IP, L2 norm ≈ cosine); produced .npz/.faiss and rich .meta.jsonl.
  - `select_service.py` can load index and model to serve Top‑K; `batch_select.py` used for candidate generation.
- Partial
  - Per-problem incremental add and text→vector LRU cache not implemented.
  - EA integration: `process_iprover_v3.py` has not yet called SELECT before scores_req.
- Plan differences
  - Whitepaper suggests HNSW/IVF‑PQ for large-scale acceleration; current setup is Flat and can be replaced after evaluation.

### 3.3 Cross‑Encoder (Transformer) RERANK training
- Done
  - Data: split by problem and document-level exclusivity enforced across splits (no leakage).
  - Training: 6 layers / hidden 512, BCEWithLogits; supports init-from and pos_weight; val AUC ≈ 0.988–0.989.
  - Tools: `rerank_with_cross_encoder.py` for offline reranking; `eval_retrieval.py` for Hit/Recall/NDCG.
- Partial
  - Online: `batch_ranker.py` cross_encoder backend and EA wiring not yet landed.
  - Pairwise/Listwise objectives not yet enabled (currently binary only).

### 3.4 Score fusion and scheduling
- Todo
  - Implement `S = λ1·S_cross + λ2·S_bi + λ3·S_heur` grid search and online fusion.
  - Scheduling: Top‑M(cross) + Oldest M1 + Lightest M2 + ε random; failure fallback and detailed logs.

### Offline end‑to‑end evaluation (no-leak test, small sample)
- Setup: Bi‑Encoder K=200 → Cross‑Encoder rerank; n_queries = 27.
- Metrics:
  - Bi Hit@K: @64 ≈ 0.518, @200 ≈ 0.630; CE Hit@K: @64 ≈ 0.556, @200 ≈ 0.630.
  - Bi Recall@K: @64 ≈ 0.0626, @200 ≈ 0.1778; CE Recall@K: @64 ≈ 0.1097, @200 ≈ 0.1778.
  - CE clearly lifts NDCG and Hit at small/medium K; at K=200 it ties Bi (candidate cap bound).
- Diagnosis:
  - Index coverage ≈ 82%; avg positives per query ≈ 180.
  - Theoretical upper bounds due to K limit: @10≈0.056, @32≈0.178, @64≈0.356, @100≈0.556, @200≈1.0 — current results match expectations.

### Next steps (priority)
1) Online wiring: `process_iprover_v3.py` calls SELECT before scores_req → forward to `batch_ranker.py (cross_encoder)` → respond; keep heuristic fallback and detailed logs.
2) Recall & stability: raise candidate K to 500/1000; expand the test set; report reachable recall (discount uncovered positives).
3) Fusion & scheduling: offline grid search for λ1/λ2/λ3 (target NDCG@64/100); implement fusion and Top‑M + Oldest/Lightest/ε scheduling in EA.
4) Hard negatives & index: add periodic hard-negative mining to fine-tune Bi; evaluate/possibly switch to HNSW/IVF‑PQ; implement per-problem incremental pool and text→vector LRU cache.
