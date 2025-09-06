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
import inspect
from dataclasses import dataclass
import os
import sys
import time
import math
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
    # Optional RoPE scaling for longer context windows at training time
    ap.add_argument("--rope-type", type=str, default="none", choices=["none", "linear", "dynamic", "yarn"], help="Apply RoPE scaling to extend context")
    ap.add_argument("--rope-factor", type=float, default=1.0, help="Scaling factor, e.g., 2.0 for ~8k if base is 4k")
    ap.add_argument("--rope-base", type=int, default=None, help="Original max_position_embeddings; if None, try from model config")
    args = ap.parse_args(argv)

    # Lazy imports to keep repo lightweight if HF not installed
    from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from transformers import BitsAndBytesConfig

    train_ds = ListwiseDataset(args.train)
    val_ds = ListwiseDataset(args.val or []) if args.val else None
    print(f"[INFO] PID={os.getpid()} | train_items={len(train_ds)} | val_items={len(val_ds) if val_ds else 0}", flush=True)

    trust_remote = not args.no_trust_remote
    load_kwargs = dict(local_files_only=args.local_only, trust_remote_code=trust_remote)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True, **load_kwargs)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    print(f"[INFO] Tokenizer loaded | pad_token={tok.pad_token!r} | eos_token={tok.eos_token!r}", flush=True)

    # 4-bit QLoRA
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16)
    # If launched with torchrun (DDP), do NOT use device_map="auto"; let HF Trainer place the model per process.
    use_ddp = False
    try:
        import torch.distributed as dist  # type: ignore
        use_ddp = dist.is_available() and dist.is_initialized()
    except Exception:
        use_ddp = False

    base_load_kwargs = dict(
        quantization_config=quant,
        **load_kwargs,
    )
    if not use_ddp:
        base_load_kwargs["device_map"] = "auto"

    base = AutoModelForCausalLM.from_pretrained(
        args.model,
        **base_load_kwargs,
    )
    # Apply optional RoPE scaling for longer contexts before preparing for k-bit training
    try:
        if args.rope_type != "none" and args.rope_factor and args.rope_factor > 1.0:
            cfg = base.config
            orig = int(args.rope_base) if args.rope_base else int(getattr(cfg, "max_position_embeddings", 4096))
            cfg.rope_scaling = {  # type: ignore[attr-defined]
                "type": args.rope_type,
                "factor": float(args.rope_factor),
                "original_max_position_embeddings": int(orig),
            }
            # Optionally lift max_position_embeddings so tokenizer truncation can go higher if model supports it
            if hasattr(cfg, "max_position_embeddings"):
                try:
                    cfg.max_position_embeddings = int(orig * args.rope_factor)
                except Exception:
                    pass
            print(f"[INFO] RoPE scaling enabled: {cfg.rope_scaling}", flush=True)
    except Exception as e:
        print(f"[WARN] Failed to apply RoPE scaling: {e}", flush=True)
    base = prepare_model_for_kbit_training(base)
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
    lcfg = LoraConfig(r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=args.lora_dropout, target_modules=target_modules, bias="none", task_type="CAUSAL_LM")
    model = get_peft_model(base, lcfg)
    print(f"[INFO] Model + LoRA ready | target_modules={target_modules} | r={args.lora_r} alpha={args.lora_alpha} dropout={args.lora_dropout}", flush=True)

    def _tokenize(batch: List[ListwiseItem]) -> Dict[str, Any]:
        # Tokenize per-sample so we can mask the prompt part (labels=-100 for prompt tokens)
        input_ids: List[List[int]] = []
        attention_mask: List[List[int]] = []
        labels: List[List[int]] = []

        for it in batch:
            # Encode without adding special tokens to keep prompt/target boundaries aligned
            p_ids = tok.encode(it.prompt, add_special_tokens=False)
            t_ids = tok.encode(it.target, add_special_tokens=False)

            # Ensure we never drop the target JSON; if over max_len, truncate prompt from the left
            max_len = int(args.max_len)
            keep_p = max(0, max_len - len(t_ids))
            if len(p_ids) > keep_p:
                p_ids = p_ids[-keep_p:] if keep_p > 0 else []

            ids = p_ids + t_ids
            lbl = ([-100] * len(p_ids)) + t_ids  # loss only on assistant JSON

            input_ids.append(ids)
            attention_mask.append([1] * len(ids))
            labels.append(lbl)

        # Pad to the longest in batch
        max_blen = max(len(x) for x in input_ids) if input_ids else 0
        pad_id = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
        for i in range(len(input_ids)):
            need = max_blen - len(input_ids[i])
            if need > 0:
                input_ids[i].extend([pad_id] * need)
                attention_mask[i].extend([0] * need)
                labels[i].extend([-100] * need)

        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    class _Collator:
        def __call__(self, features):
            return _tokenize(features)

    # Build TrainingArguments in a version-compatible way (HF v4/v5)
    ta_kwargs = dict(
        output_dir=str(args.out),
        per_device_train_batch_size=args.batch,
        per_device_eval_batch_size=max(1, args.batch),
        gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs,
        learning_rate=args.lr,
        logging_steps=10,
        logging_first_step=True,
        save_steps=500,
        bf16=args.bf16,
        fp16=not args.bf16,
        gradient_checkpointing=True,
        optim="paged_adamw_8bit",
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.01,
        dataloader_num_workers=2,
        report_to=[],  # disable external loggers for clean console output
        ddp_find_unused_parameters=False,
    )
    sig_params = set(inspect.signature(TrainingArguments.__init__).parameters.keys())
    eval_value = "steps" if val_ds else "no"
    if "evaluation_strategy" in sig_params:
        ta_kwargs["evaluation_strategy"] = eval_value
    elif "eval_strategy" in sig_params:
        ta_kwargs["eval_strategy"] = eval_value
    if val_ds and "eval_steps" in sig_params:
        ta_kwargs["eval_steps"] = 200

    training_args = TrainingArguments(**ta_kwargs)

    # ---- Console progress callback ----
    from transformers import TrainerCallback, TrainerState, TrainerControl  # type: ignore

    class ConsoleLogger(TrainerCallback):
        def __init__(self):
            self.t0 = time.time()
            self.last_step = 0
            self.last_time = self.t0
        def on_train_begin(self, args, state: TrainerState, control: TrainerControl, **kwargs):
            ws = int(os.environ.get("WORLD_SIZE", "1"))
            rank = int(os.environ.get("RANK", "0"))
            print(f"[RUN] output_dir={args.output_dir} | epochs={args.num_train_epochs} | batch={args.per_device_train_batch_size} | grad_accum={args.gradient_accumulation_steps} | world_size={ws} rank={rank}", flush=True)
            # Reset CUDA peak memory stats so peak reflects training run
            if torch.cuda.is_available():
                try:
                    torch.cuda.reset_peak_memory_stats()
                except Exception:
                    pass
        def on_log(self, args, state: TrainerState, control: TrainerControl, logs=None, **kwargs):
            now = time.time()
            step = state.global_step or 0
            max_steps = getattr(state, "max_steps", None) or 0
            dt = now - self.last_time
            self.last_time = now
            # ETA
            eta = None
            if step and max_steps and now > self.t0:
                frac = min(1.0, max(0.0, step / float(max_steps)))
                remaining = (now - self.t0) * (1.0 - frac) / max(1e-6, frac)
                eta = remaining
            # GPU mem
            mem_alloc = mem_res = mem_peak = None
            if torch.cuda.is_available():
                try:
                    mem_alloc = torch.cuda.memory_allocated() / (1024**3)
                    mem_res = torch.cuda.memory_reserved() / (1024**3)
                    mem_peak = torch.cuda.max_memory_allocated() / (1024**3)
                except Exception:
                    mem_alloc = mem_res = mem_peak = None
            loss = (logs or {}).get("loss")
            lr = (logs or {}).get("learning_rate")
            grad_norm = (logs or {}).get("grad_norm")
            msg = f"[STEP {step}/{max_steps}] loss={loss:.4f} lr={lr:.2e}"
            if grad_norm is not None:
                msg += f" | grad_norm={float(grad_norm):.3f}"
            if mem_alloc is not None:
                # Show allocated/reserved/peak
                msg += f" | gpu_mem={mem_alloc:.2f}G alloc/{mem_res:.2f}G resv/{mem_peak:.2f}G peak"
            if eta is not None:
                msg += f" | ETA={int(eta)//60:02d}m{int(eta)%60:02d}s"
            print(msg, flush=True)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=_Collator(),
        tokenizer=tok,
        callbacks=[ConsoleLogger()],
    )

    # Print expected total steps for visibility
    try:
        n_train = len(train_ds)
    world = int(os.environ.get("WORLD_SIZE", "1"))
    eff_bs = args.batch * max(1, world)
        steps_per_epoch = math.ceil(n_train / max(1, eff_bs)) // max(1, args.grad_accum)
        print(f"[INFO] Approx steps/epoch={steps_per_epoch} | total_epochs={args.epochs}", flush=True)
    except Exception:
        pass

    trainer.train()
    model.save_pretrained(str(args.out))
    tok.save_pretrained(str(args.out))
    print(f"Saved LoRA adapter to {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
