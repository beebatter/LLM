#!/usr/bin/env python3
"""
QLoRA fine-tuning for a listwise LLM reranker that outputs strictly JSON: {"scores":[...]}

Data: JSONL produced by scripts/make_listwise_chunks.py with fields:
  - input: prompt text (includes conjecture + K candidates)
  - target_json: stringified {"scores":[...]} or
  - target_scores: [float] (optional; used for auxiliary KL if enabled)

We do supervised fine-tuning to teach stable JSON output. Optionally, an auxiliary
KL loss over the numeric scores can be added later; for now we do pure SFT for robustness.

This script uses Hugging Face Transformers + PEFT (LoRA/QLoRA) and is designed to run on a single GPU.
"""
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset


@dataclass
class ListwiseItem:
    prompt: str
    target: str


class ListwiseDataset(Dataset):
    def __init__(self, paths: List[str]) -> None:
        self.items: List[ListwiseItem] = []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    prompt = j.get("input")
                    # prefer explicit JSON string; fallback to building from scores
                    tgt = j.get("target_json")
                    if not tgt:
                        scores = j.get("target_scores")
                        if isinstance(scores, list) and scores:
                            tgt = json.dumps({"scores": scores}, ensure_ascii=False)
                    if not prompt or not tgt:
                        continue
                    # Strong instruction to only output JSON
                    sys_inst = (
                        "你是自动定理证明的子句打分器。只输出合法 JSON（无多余文本）。\n"
                        "必须严格输出：{\"scores\":[...]}，且长度等于候选数 K。\n"
                    )
                    prompt_full = sys_inst + "\n" + prompt + "\n只输出 JSON："
                    self.items.append(ListwiseItem(prompt=prompt_full, target=tgt))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> ListwiseItem:
        return self.items[idx]


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="QLoRA SFT for LLM listwise reranker")
    ap.add_argument("--model", type=str, required=True, help="HF model id/path, e.g., deepseek-ai/deepseek-math-7b-instruct")
    ap.add_argument("--train", action="append", required=True, help="Train JSONL listwise file(s)")
    ap.add_argument("--val", action="append", help="Val JSONL listwise file(s)")
    ap.add_argument("--out", type=Path, required=True, help="Output directory for LoRA adapter")
    ap.add_argument("--max-len", type=int, default=4096)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    ap.add_argument("--local-only", action="store_true", help="Load tokenizer/model only from local cache or path (offline mode)")
    ap.add_argument("--no-trust-remote", action="store_true", help="Disable trust_remote_code when loading models/tokenizers")
    args = ap.parse_args(argv)

    # Lazy imports to keep repo lightweight if HF not installed
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    train_ds = ListwiseDataset(args.train)
    val_ds = ListwiseDataset(args.val or []) if args.val else None

    trust_remote = not args.no_trust_remote
    load_kwargs = dict(local_files_only=args.local_only, trust_remote_code=trust_remote)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, **load_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    # 4-bit QLoRA
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        quantization_config=quant,
        device_map="auto",
        **load_kwargs,
    )
    base = prepare_model_for_kbit_training(base)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)

    def _tokenize(batch: List[ListwiseItem]) -> Dict[str, Any]:
        texts = []
        for it in batch:
            # Supervised format: prompt + target JSON
            texts.append(it.prompt + it.target)
        toks = tok(texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        # Build labels: supervise on the whole sequence (simple SFT). Could also mask prompt if desired.
        toks["labels"] = toks["input_ids"].clone()
        return toks

    class _Collator:
        def __call__(self, features):
            return _tokenize(features)

    training_args = TrainingArguments(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(1, args.batch),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        save_steps=500,
        evaluation_strategy="steps" if val_ds else "no",
        eval_steps=200 if val_ds else None,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=_Collator(),
        tokenizer=tok,
    )

    trainer.train()
    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"Saved LoRA adapter to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
