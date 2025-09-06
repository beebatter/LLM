#!/usr/bin/env python3
from __future__ import annotations

import argparse, json, os
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Any

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, BitsAndBytesConfig
from transformers import DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
import torch.nn.functional as F


def load_jsonl(path: str) -> List[Dict[str, Any]]:
    data: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for ln in f:
            try:
                j = json.loads(ln)
            except Exception:
                continue
            if not j.get("input"):
                # Fallback to build a compact prompt on the fly
                conj = j.get("conjecture", "")
                cands = j.get("candidates") or []
                K = int(j.get("K") or len(cands))
                lines = [f"[CONJECTURE]\n{conj}\n", f"[CANDIDATES] 共 {K} 条。每条：\n"]
                for idx, c in enumerate(cands, start=1):
                    cid = c.get("id", idx)
                    lines.append(f"- ID {cid}\n- TEXT: {c.get('text','')}\n- TAGS: {c.get('tags','')}")
                prompt = "你是自动定理证明的子句打分器。只输出严格 JSON：{\"scores\":[...]}。\n" + "\n".join(lines)
            else:
                prompt = j["input"]
            target = j.get("target_json")
            if isinstance(target, dict):
                target = json.dumps(target, ensure_ascii=False)
            if not target and j.get("target_scores"):
                target = json.dumps({"scores": j["target_scores"]}, ensure_ascii=False)
            if not target:
                # create a uniform placeholder; not ideal but prevents drop
                K = int(j.get("K") or len(j.get("candidates") or []))
                if K <= 0:
                    continue
                target = json.dumps({"scores": [1.0/float(K)]*K}, ensure_ascii=False)
            data.append({"prompt": prompt, "target": target})
    return data


class JsonOnlySFT(Dataset):
    def __init__(self, items: List[Dict[str, str]], tok, max_len: int = 2048):
        self.items = items
        self.tok = tok
        self.max_len = max_len

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int):
        it = self.items[idx]
        text = it["prompt"] + "\n" + it["target"]
        enc = self.tok(text, truncation=True, max_length=self.max_len, padding=False, return_tensors=None)
        # Do NOT attach labels here; collator will create labels and mask pads as -100 safely
        return enc


def main():
    ap = argparse.ArgumentParser(description="QLoRA SFT for JSON-only listwise outputs")
    ap.add_argument("--model", required=True, help="base model path or HF name")
    ap.add_argument("--train", required=True, help="train listwise jsonl")
    ap.add_argument("--val", required=True, help="val listwise jsonl")
    ap.add_argument("--out", required=True, help="output dir for checkpoints")
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--quant", choices=["4bit","8bit","none"], default="4bit", help="quantization mode for base model")
    ap.add_argument("--lora-r", type=int, default=32)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-dropout", type=float, default=0.05)
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    # Prefer right padding for causal LM and avoid adding new special tokens if possible
    try:
        tok.padding_side = "right"
        if tok.pad_token is None and tok.eos_token is not None:
            tok.pad_token = tok.eos_token
    except Exception:
        pass

    # Configure quantization per args.quant
    qcfg = None
    if args.quant == "4bit":
        qcfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            bnb_4bit_use_double_quant=True,
        )
    elif args.quant == "8bit":
        qcfg = BitsAndBytesConfig(
            load_in_8bit=True,
        )

    if qcfg is not None:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            quantization_config=qcfg,
            device_map="auto",
        )
    else:
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            torch_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            device_map="auto",
        )

    # Ensure model/tokenizer vocab alignment to prevent out-of-range label ids
    try:
        if hasattr(model, "get_input_embeddings"):
            cur = model.get_input_embeddings().weight.shape[0]
            tgt = len(tok)
            if tgt != cur:
                model.resize_token_embeddings(tgt)
        # Align config vocab_size to output head and tie weights
        try:
            out_v = model.get_output_embeddings().weight.shape[0]
            model.config.vocab_size = int(out_v)
        except Exception:
            pass
        if hasattr(model, "tie_weights"):
            try:
                model.tie_weights()
            except Exception:
                pass
    except Exception:
        # Best-effort; if resizing fails under quantization, rely on pad->eos mapping to avoid unseen ids
        pass

    # Make sure pad/eos ids are consistent on model side and disable cache for grad checkpointing
    try:
        if tok.pad_token_id is not None:
            model.config.pad_token_id = tok.pad_token_id
        if tok.eos_token_id is not None:
            model.config.eos_token_id = tok.eos_token_id
        if hasattr(model, "generation_config"):
            if tok.pad_token_id is not None:
                model.generation_config.pad_token_id = tok.pad_token_id
            if tok.eos_token_id is not None:
                model.generation_config.eos_token_id = tok.eos_token_id
        model.config.use_cache = False
    except Exception:
        pass
    # Ensure inputs require grads for gradient checkpointing + LoRA, even when not quantized
    try:
        if hasattr(model, "enable_input_require_grads"):
            model.enable_input_require_grads()
    except Exception:
        pass
    # Prepare model for k-bit/LoRA training (safe to call even when not quantized)
    model = prepare_model_for_kbit_training(model)
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"],
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, lora_cfg)

    train_items = load_jsonl(args.train)
    val_items = load_jsonl(args.val)
    train_ds = JsonOnlySFT(train_items, tok, max_len=args.max_len)
    val_ds = JsonOnlySFT(val_items, tok, max_len=args.max_len)

    targs = TrainingArguments(
        output_dir=args.out,
        per_device_train_batch_size=args.batch,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        num_train_epochs=args.epochs,
        warmup_ratio=0.03,
        bf16=args.bf16,
        logging_steps=50,
        save_strategy="epoch",
        gradient_checkpointing=True,
        optim="paged_adamw_32bit",
    )

    # Custom collator: pad, then set labels = input_ids with pads -> -100 and guard OOR ids
    _collator_calls = {"n": 0}
    def data_collator(features):
        # features already truncated in dataset; pad to longest in batch
        batch = tok.pad(features, padding=True, return_tensors="pt")
        # enforce integer types
        batch["input_ids"] = batch["input_ids"].to(torch.long)
        if "attention_mask" in batch:
            batch["attention_mask"] = batch["attention_mask"].to(torch.long)
        labels = batch["input_ids"].clone()
        # mask out pads
        if "attention_mask" in batch:
            labels[batch["attention_mask"] == 0] = -100
        # also mask explicit pad token id, if present
        if tok.pad_token_id is not None:
            labels[labels == tok.pad_token_id] = -100
        # guard any out-of-range token ids against the actual vocab used in loss
        try:
            vsz_out = model.get_output_embeddings().weight.shape[0]
        except Exception:
            vsz_out = len(tok)
        n_classes = int(getattr(model.config, "vocab_size", vsz_out))
        if n_classes <= 0:
            n_classes = int(vsz_out)
        labels[(labels >= n_classes) | (labels < 0)] = -100
        batch["labels"] = labels.to(torch.long)
        # one-time diagnostics
        if _collator_calls["n"] == 0:
            _collator_calls["n"] += 1
            with torch.no_grad():
                mx = labels[labels != -100].max().item() if (labels != -100).any() else -1
                mn = labels[labels != -100].min().item() if (labels != -100).any() else -1
                print(f"[collator] lm_head_out={vsz_out} config.vocab_size={n_classes} labels[min,max]=[{mn},{mx}] pad_id={tok.pad_token_id}")
        return batch

    # One-batch sanity check: ensure labels are within logits dim after forward
    try:
        dl = DataLoader(train_ds, batch_size=1, shuffle=False, collate_fn=data_collator)
        first = next(iter(dl))
        device = next(model.parameters()).device
        first = {k: v.to(device) if isinstance(v, torch.Tensor) else v for k, v in first.items()}
        with torch.no_grad():
            out = model(input_ids=first["input_ids"], attention_mask=first.get("attention_mask"), use_cache=False)
            logits = out.logits  # [B, L, V]
            V = logits.size(-1)
            mx = int(first["labels"][first["labels"] != -100].max().item()) if (first["labels"] != -100).any() else -1
            print(f"[debug] logits_vocab={V} labels_max={mx} matches? {mx < V}")
            # manual shifted CE to catch issues early
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = first["labels"][:, 1:].contiguous()
            loss_dbg = torch.nn.functional.cross_entropy(
                shift_logits.view(-1, V),
                shift_labels.view(-1),
                ignore_index=-100,
            )
            print(f"[debug] pre-train CE ok, loss={float(loss_dbg):.4f}")
    except Exception as e:
        print(f"[debug] sanity-check skipped: {e}")

    class MyTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            labels = inputs.pop("labels")
            outputs = model(**inputs, use_cache=False)
            logits = outputs.get("logits") if isinstance(outputs, dict) else outputs.logits
            V = logits.size(-1)
            # shifted LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = labels[:, 1:].contiguous()
            loss = F.cross_entropy(shift_logits.view(-1, V), shift_labels.view(-1), ignore_index=-100)
            return (loss, outputs) if return_outputs else loss

    trainer = MyTrainer(
        model=model,
        args=targs,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tok,
        data_collator=data_collator,
    )
    trainer.train()

    # Save LoRA adapter only
    model.save_pretrained(os.path.join(args.out, "lora"))
    tok.save_pretrained(os.path.join(args.out, "tokenizer"))


if __name__ == "__main__":
    main()
