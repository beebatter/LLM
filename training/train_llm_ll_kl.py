#!/usr/bin/env python3
from __future__ import annotations

"""
Train LLM-LL (non-generative) with KL to teacher soft labels.

Data: JSONL from make_listwise_chunks.py including fields:
  - conjecture: str
  - candidates: [{text, tags, id?, label?}, ...] (length K)
  - target_scores: [float] soft labels from teacher fusion

Objective: For each window, compute s_i = log P(target_i | prompt) for each candidate i,
then minimize KL(softmax(s) || target_scores). This avoids JSON generation.

Supports LoRA on the base model. Uses 4/8/16-bit load. Truncation controlled by --input-max and --target-max-toks.
"""

import argparse
import os
import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Dict, Optional

import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM  # type: ignore
from transformers import BitsAndBytesConfig  # type: ignore
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from LLM.data_utils.logic_tokenizer import features_to_prefix, PrefixBuckets, normalize_text
# Reuse the prompt parser used when building teacher targets
from LLM.scripts.make_listwise_targets_from_teacher import _parse_prompt_for_q_and_candidates


@dataclass
class Window:
    conj: str
    targets: List[str]  # serialized candidate TEXT+TAGS blocks
    y: List[float]


class ListwiseKLDataset(Dataset):
    def __init__(self, paths: List[str], max_items: int = 0):
        self.items: List[Window] = []
        for p in paths:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                for ln in f:
                    try:
                        j = json.loads(ln)
                    except Exception:
                        continue
                    # teacher scores (soft labels)
                    y = j.get('target_scores') or []
                    if not y:
                        # try to parse from target_json if present
                        try:
                            tj = j.get('target_json')
                            if isinstance(tj, str):
                                obj = json.loads(tj)
                            else:
                                obj = tj
                            if obj and isinstance(obj, dict) and isinstance(obj.get('scores'), list):
                                y = obj['scores']
                        except Exception:
                            pass
                    # Prefer explicit fields; otherwise parse from the prompt
                    conj = j.get('conjecture') or j.get('conjecture_text') or ''
                    cands = j.get('candidates') or []
                    if (not conj or not cands) and j.get('input'):
                        try:
                            pq, tuples = _parse_prompt_for_q_and_candidates(j['input'])
                            if not conj:
                                conj = pq or ''
                            if not cands and tuples:
                                # tuples: List[(id, text, tags)]
                                cands = [
                                    {"id": cid, "text": text, "features": {}, "tags": tags}
                                    for (cid, text, tags) in tuples
                                ]
                        except Exception:
                            pass
                    if not conj or not cands or not y:
                        continue
                    if len(cands) != len(y):
                        continue
                    targets: List[str] = []
                    for c in cands:
                        if isinstance(c, dict):
                            # prefer explicit tags if provided by parser; else synthesize from features
                            tags = c.get('tags') or features_to_prefix(c.get('features') or {}, PrefixBuckets())
                            txt = normalize_text((c.get('text') or '')).strip()
                        else:
                            tags = ''
                            txt = normalize_text(str(c) or '')
                        blk = f"[CANDIDATE]\n- TEXT: {txt}\n- TAGS: {str(tags).strip()}\n"
                        targets.append(blk)
                    self.items.append(Window(conj=normalize_text(conj), targets=targets, y=[float(v) for v in y]))
                    if max_items and len(self.items) >= max_items:
                        break

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> Window:
        return self.items[idx]


def build_model_and_tok(model_id: str, lora: Optional[str], bits: int, bf16: bool, rope_type: str, rope_factor: float, rope_base: Optional[int],
                        lora_r: int, lora_alpha: int, lora_dropout: float, lora_target: Optional[str],
                        ddp: bool = False, local_rank: int = 0):
    tok = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    if getattr(tok, 'pad_token', None) is None:
        tok.pad_token = tok.eos_token
    try:
        tok.truncation_side = 'left'
    except Exception:
        pass
    # lift tokenizer limit to suppress warnings; we enforce our own caps
    try:
        tok.model_max_length = max(getattr(tok, 'model_max_length', 4096) or 4096, 100000)
    except Exception:
        pass

    torch_dtype = torch.bfloat16 if bf16 and torch.cuda.is_available() else torch.float16
    # Use per-rank device placement in DDP; otherwise allow auto device_map
    if ddp and torch.cuda.is_available():
        device_map = {"": int(local_rank)}
        torch.cuda.set_device(int(local_rank))
    else:
        device_map = 'auto' if torch.cuda.is_available() else None
    quant: Optional[BitsAndBytesConfig] = None
    if bits == 4:
        quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True, bnb_4bit_compute_dtype=torch_dtype)
    elif bits == 8:
        quant = BitsAndBytesConfig(load_in_8bit=True)

    if quant is not None:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, quantization_config=quant, device_map=device_map, torch_dtype=torch_dtype)
    else:
        mdl = AutoModelForCausalLM.from_pretrained(model_id, device_map=device_map, torch_dtype=torch_dtype)
    try:
        mdl.config.use_cache = False  # reduce memory during training
    except Exception:
        pass

    if rope_type != 'none' and rope_factor and rope_factor > 1.0:
        try:
            cfg = mdl.config
            orig = int(rope_base) if rope_base else int(getattr(cfg, 'max_position_embeddings', 4096))
            cfg.rope_scaling = {'type': rope_type, 'factor': float(rope_factor), 'original_max_position_embeddings': int(orig)}
            if hasattr(cfg, 'max_position_embeddings'):
                try:
                    cfg.max_position_embeddings = int(orig * rope_factor)
                except Exception:
                    pass
            print(f"[INFO] RoPE scaling: {cfg.rope_scaling}")
        except Exception as e:
            print(f"[WARN] RoPE scaling failed: {e}")

    # LoRA handling: if an existing adapter path is given, load it; otherwise create a fresh LoRA on common transformer modules
    try:
        if lora:
            from peft import PeftModel  # type: ignore
            mdl = PeftModel.from_pretrained(mdl, lora)
        else:
            from peft import get_peft_model, LoraConfig, TaskType  # type: ignore
            # target modules: cover q/k/v/o + MLP (gate/up/down) for common archs (Qwen/LLaMA/DeepSeek)
            if lora_target and lora_target.strip():
                target_modules = [t.strip() for t in lora_target.split(',') if t.strip()]
            else:
                target_modules = [
                    'q_proj','k_proj','v_proj','o_proj',
                    'gate_proj','up_proj','down_proj'
                ]
            lcfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=target_modules,
                bias='none',
                task_type=TaskType.CAUSAL_LM,
            )
            mdl = get_peft_model(mdl, lcfg)
            try:
                mdl.print_trainable_parameters()  # quick visibility
            except Exception:
                pass
    except Exception as e:
        print(f"[WARN] LoRA setup failed or unavailable: {e}")
    mdl.train()
    # choose device for inputs
    try:
        model_dev = mdl.get_input_embeddings().weight.device
    except Exception:
        model_dev = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    return tok, mdl, model_dev


def collate_batch(batch: List[Window], tok, input_max: int, target_max: int):
    prompts: List[List[int]] = []
    targets: List[List[int]] = []
    # For each window, we will concatenate per-candidate sequences later
    data = []
    for w in batch:
        conj_block = f"[CONJECTURE]\n{w.conj}\n\n"
        ps: List[List[int]] = []
        ts: List[List[int]] = []
        for tgt in w.targets:
            t_ids = tok.encode(tgt, add_special_tokens=False, truncation=True, max_length=int(target_max))
            keep_p = max(0, int(input_max) - len(t_ids))
            if keep_p > 0:
                p_ids = tok.encode(conj_block, add_special_tokens=False, truncation=True, max_length=int(keep_p))
                if len(p_ids) > keep_p:
                    p_ids = p_ids[-keep_p:]
            else:
                p_ids = []
            ps.append(p_ids)
            ts.append(t_ids)
        data.append((ps, ts, torch.tensor(w.y, dtype=torch.float32)))
    return data


def kl_train_loop(args):
    # DDP setup (optional)
    use_ddp = bool(getattr(args, 'ddp', False)) and torch.cuda.is_available() and int(os.environ.get('WORLD_SIZE', '1')) > 1
    local_rank = int(os.environ.get('LOCAL_RANK', '0')) if use_ddp else 0
    if use_ddp and not dist.is_initialized():
        dist.init_process_group(backend='nccl', timeout=torch.distributed.datetime.timedelta(seconds=18000) if hasattr(torch.distributed, 'datetime') else None)
        torch.cuda.set_device(local_rank)
    world_size = dist.get_world_size() if use_ddp else 1
    rank = dist.get_rank() if use_ddp else 0
    ds = ListwiseKLDataset(args.train, max_items=args.max_items)
    vs = ListwiseKLDataset(args.val or [], max_items=args.val_max_items) if args.val else None
    tok, mdl, device = build_model_and_tok(
        args.model, args.lora, args.bits, args.bf16,
        args.rope_type, args.rope_factor, args.rope_base,
        args.lora_r, args.lora_alpha, args.lora_dropout, args.lora_target_modules,
        ddp=use_ddp, local_rank=local_rank,
    )
    # Wrap with DDP for multi-GPU training
    runner = mdl
    if use_ddp:
        try:
            runner = DDP(mdl, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
        except Exception as e:
            if rank == 0:
                print(f"[WARN] DDP wrap failed, continuing single-GPU on rank {rank}: {e}")
            use_ddp = False
            runner = mdl

    # Optional gradient checkpointing to save memory
    if getattr(args, 'grad_checkpointing', False):
        try:
            # Different wrappers expose different methods; try broadly
            if hasattr(mdl, 'gradient_checkpointing_enable'):
                mdl.gradient_checkpointing_enable()
            try:
                base = getattr(mdl, 'base_model', None)
                core = getattr(base, 'model', None)
                if core and hasattr(core, 'gradient_checkpointing_enable'):
                    core.gradient_checkpointing_enable()
            except Exception:
                pass
            if hasattr(mdl, 'enable_input_require_grads'):
                mdl.enable_input_require_grads()
        except Exception:
            pass

    opt = torch.optim.AdamW(mdl.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = None
    ce = torch.nn.KLDivLoss(reduction='batchmean')

    def run_epoch(split: str):
        nonlocal scaler
        dset = ds if split == 'train' else vs
        if dset is None or len(dset) == 0:
            return 0.0
        mdl.train() if split == 'train' else mdl.eval()
        total_loss = 0.0
        n_win = 0
        # Split data across ranks without sampler
        start_idx = rank * args.batch
        step = args.batch * world_size
        for i in range(start_idx, len(dset), step):
            chunk = [dset[j] for j in range(i, min(i+args.batch, len(dset)))]
            pack = collate_batch(chunk, tok, args.input_max, args.target_max_toks)

            # Build a list of per-candidate sequences across windows, then micro-batch them to control memory
            all_ids: List[List[int]] = []
            p_lens: List[int] = []
            t_lens: List[int] = []
            y_list: List[torch.Tensor] = []
            offsets: List[int] = [0]
            for (ps, ts, y) in pack:
                for p_ids, t_ids in zip(ps, ts):
                    all_ids.append(p_ids + t_ids)
                    p_lens.append(len(p_ids))
                    t_lens.append(len(t_ids))
                y_list.append(y)
                offsets.append(offsets[-1] + len(y))

            # Compute per-row scores in chunks to reduce peak memory
            row_scores_all = torch.empty(len(all_ids), dtype=torch.float32, device=device)
            for start_idx in range(0, len(all_ids), max(1, int(args.cand_chunk))):
                end_idx = min(start_idx + int(args.cand_chunk), len(all_ids))
                idxs = list(range(start_idx, end_idx))
                subset = [all_ids[r] for r in idxs]
                if not subset:
                    continue
                max_len = max(len(x) for x in subset)
                input_ids = torch.full((len(subset), max_len), 0, dtype=torch.long, device=device)
                attn = torch.zeros_like(input_ids)
                for i_local, ids in enumerate(subset):
                    L = len(ids)
                    input_ids[i_local, :L] = torch.tensor(ids, dtype=torch.long, device=device)
                    attn[i_local, :L] = 1

                with torch.autocast(device_type='cuda', dtype=torch.bfloat16, enabled=args.bf16 and torch.cuda.is_available()):
                    out = runner(input_ids=input_ids, attention_mask=attn, use_cache=False)
                    logits = out.logits
                    logprobs = torch.log_softmax(logits[:, :-1, :], dim=-1)
                    labels = input_ids[:, 1:]
                    for i_local, r in enumerate(idxs):
                        start = max(0, p_lens[r] - 1)
                        end = start + t_lens[r]
                        Lr = labels.shape[1]
                        start = min(max(0, start), Lr)
                        end = min(max(start, end), Lr)
                        if end <= start:
                            row_scores_all[r] = -1e9
                            continue
                        lp = logprobs[i_local, start:end, :].gather(-1, labels[i_local, start:end].unsqueeze(-1)).squeeze(-1)
                        row_scores_all[r] = lp.sum()

            s = row_scores_all
            # reshape back to windows
            preds: List[torch.Tensor] = []
            for wi in range(len(y_list)):
                a, b = offsets[wi], offsets[wi+1]
                preds.append(s[a:b])
            # compute KL per window: log_softmax(preds) vs target y
            loss = 0.0
            for wi, (pred, y) in enumerate(zip(preds, y_list)):
                logp = torch.log_softmax(pred, dim=-1)
                tgt = y.to(logp.dtype).to(device)
                # normalize target in case of tiny drift
                tgt = tgt / max(tgt.sum(), torch.tensor(1e-6, device=device))
                loss = loss + ce(logp, tgt)

            if split == 'train':
                opt.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(mdl.parameters(), 1.0)
                opt.step()

            total_loss += float(loss.detach().cpu())
            n_win += len(chunk)
            if split == 'train' and rank == 0 and ((i - start_idx) // step) % 10 == 0:
                print(f"[TRAIN] step={((i - start_idx)//step)} loss={float(loss):.4f} windows={n_win} (rank {rank})")
        # Aggregate across ranks
        if use_ddp:
            t = torch.tensor([total_loss, float(n_win)], device=device)
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
            total_loss, n_win = float(t[0].item()), int(t[1].item())
        return total_loss / max(1, n_win)

    for ep in range(args.epochs):
        tr = run_epoch('train')
        print(f"[EPOCH {ep}] train_kl={tr:.4f}")
        if vs is not None and len(vs) > 0:
            with torch.no_grad():
                vl = run_epoch('val')
            print(f"[EPOCH {ep}] val_kl={vl:.4f}")
    # 保存 LoRA 权重（如果使用）或整模型配置
    # Save adapter or model
    # Save only on rank 0
    if (not use_ddp) or rank == 0:
        try:
            mdl.save_pretrained(args.out)
        except Exception as e:
            print(f"[WARN] save_pretrained failed: {e}")
        print(f"saved to {args.out}")
    if use_ddp:
        try:
            dist.barrier()
            dist.destroy_process_group()
        except Exception:
            pass


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description='Train LLM-LL via KL to teacher soft labels')
    ap.add_argument('--model', type=str, required=True)
    ap.add_argument('--lora', type=str, default=None)
    ap.add_argument('--bits', type=int, choices=[4,8,16], default=4)
    ap.add_argument('--bf16', action='store_true')
    ap.add_argument('--train', action='append', required=True)
    ap.add_argument('--val', action='append')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--epochs', type=int, default=1)
    ap.add_argument('--batch', type=int, default=2, help='windows per step')
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--input-max', type=int, default=1536)
    ap.add_argument('--target-max-toks', type=int, default=192)
    ap.add_argument('--max-items', type=int, default=0, help='subset train (0=all)')
    ap.add_argument('--val-max-items', type=int, default=0)
    ap.add_argument('--rope-type', type=str, default='none', choices=['none','linear','dynamic','yarn'])
    ap.add_argument('--rope-factor', type=float, default=1.0)
    ap.add_argument('--rope-base', type=int, default=None)
    # LoRA hyperparams (used when --lora is not provided to create a fresh adapter)
    ap.add_argument('--lora-r', type=int, default=16)
    ap.add_argument('--lora-alpha', type=int, default=32)
    ap.add_argument('--lora-dropout', type=float, default=0.05)
    ap.add_argument('--lora-target-modules', type=str, default='', help='comma-separated; default covers q/k/v/o and gate/up/down proj')
    ap.add_argument('--grad-checkpointing', action='store_true', help='enable gradient checkpointing to save memory')
    ap.add_argument('--cand-chunk', type=int, default=16, help='number of candidates to score per forward pass')
    ap.add_argument('--ddp', action='store_true', help='enable multi-GPU DDP training (use torchrun)')
    args = ap.parse_args(argv)

    kl_train_loop(args)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
