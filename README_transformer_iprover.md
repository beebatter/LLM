# Transformer training and iProver External Agent

This folder provides a minimal pipeline to train a clause-only Transformer scorer using your SentencePiece tokenizer and to serve scores to iProver in Interactive Mode (Passive Scores Mode, PSM).

## Files
- `models/logic_transformers.py`: Transformer encoder and `ClauseScorer`.
- `training/logic_datasets.py`: JSONL dataset reader and collate using `spm_logic.model`.
- `training/train_clause_scorer.py`: Trainer script (MSE loss on optional `label`/`score`).
- `ea/iprover_external_agent.py`: External Agent server that serves scores over TCP.

## Train
Assumes tokenizer at `/home/ks/Training/models/spm_logic.model` and datasets under `/home/ks/Training/datasets`.

```bash
# Example: train for 3 epochs; adjust JSONL list to your split
python3 /home/ks/LLM/training/train_clause_scorer.py \
  --train /home/ks/Training/datasets/train.jsonl \
  --val /home/ks/Training/datasets/val.jsonl \
  --spm /home/ks/Training/models/spm_logic.model \
  --epochs 3 --batch 64 --lr 3e-4 --max-len 512 \
  --d-model 256 --layers 4 --heads 4 \
  --save /home/ks/Training/models/clause_scorer.pt
```

Notes:
- If your JSONL lacks labels, the trainer will optimize to 0.0; provide weak labels (e.g., normalized age, -conj_dist, or success heuristics) or switch to pairwise ranking later.
- Vocab size is inferred from `.vocab` next to the model.

## Run External Agent
Start EA, then run iProver in interactive mode using PSM options.

```bash
# Start EA server
python3 /home/ks/LLM/ea/iprover_external_agent.py \
  --host 127.0.0.1 --port 12345 \
  --model /home/ks/Training/models/clause_scorer.pt \
  --spm /home/ks/Training/models/spm_logic.model
```

In a separate terminal, launch iProver with external scores queue, as per iProver README:

```bash
cd /home/ks/iprover-master
./iproveropt --interactive_mode true --external_ip_address "127.0.0.1" --external_port 12345 \
  --schedule none --preprocessing_flag false --instantiation_flag false --superposition_flag true \
  --sup_iter_deepening 0 --comb_sup_deep_mult 0 \
  --sup_passive_queue_type priority_queues --sup_passive_queues_freq "[1]" \
  --sup_passive_queues "[[+external_score]]" Examples/PUZ001-1.p
```

## Next steps
- Replace MSE with listwise/pairwise ranking using proof outcomes or heuristic targets.
- Add query-aware bi-encoder and integrate `<Q>` context when available.
- Log score distributions and add safeguards to avoid degenerate constant outputs.