#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

from LLM.training.token_utils import ensure_special_tokens, locate_token_positions


class GroupItem:
    __slots__ = ("problem_name", "query", "cands", "target")
    def __init__(self, pn: str, q: str, cands: List[dict], target: Optional[List[float]]):
        self.problem_name = pn
        self.query = q
        self.cands = cands
        self.target = target


class GroupsDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.items: List[GroupItem] = []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    pn = j.get("problem_name") or ""
                    q = j.get("query") or ""
                    cands = j.get("candidates") or []
                    tgt = j.get("target_scores")
                    if not q or not cands:
                        continue
                    self.items.append(GroupItem(pn, q, cands, tgt))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i: int) -> GroupItem:
        return self.items[i]


def format_prompt(q: str, cands: List[dict], cand_max_chars: int = 0) -> str:
    parts = ["[CONJECTURE]\n", f"<Q> {q} </Q>\n"]
    for i, c in enumerate(cands, 1):
        tags = c.get("meta") or {}
        # simple tag rendering
        tag_str = " ".join(f"<{k}={v}>" for k, v in tags.items() if v is not None)
        text = c.get('text', '')
        if cand_max_chars and cand_max_chars > 0 and isinstance(text, str) and len(text) > cand_max_chars:
            text = text[:cand_max_chars]
        parts.append(f"[CANDIDATE {i}]\n<CAND_START>\n{text}\n{tag_str}\n<CAND_END>\n")
    return "".join(parts)


class ScoreHead(nn.Module):
    def __init__(self, hidden: int):
        super().__init__()
        self.proj = nn.Linear(hidden, 1)

    def forward(self, last_hidden: torch.Tensor, cand_pos_or_spans: List[List]):
        # last_hidden: [B,T,H]
        B, T, H = last_hidden.size()
        out = []
        for b in range(B):
            entries = cand_pos_or_spans[b]
            if not entries:
                out.append(torch.zeros(1, device=last_hidden.device))
                continue
            scores_b = []
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    st, ed = int(e[0]), int(e[1])
                    # pool tokens inside span (exclusive of markers)
                    st_eff = max(0, st + 1)
                    ed_eff = max(st_eff + 1, min(T, ed))
                    hs = last_hidden[b, st_eff:ed_eff, :]  # [L,H]
                    if hs.numel() == 0:
                        # fallback to start token if empty span
                        hs = last_hidden[b, st:st+1, :]
                    h = hs.mean(dim=0)  # [H]
                else:
                    # assume it's a single position
                    pos = int(e)
                    h = last_hidden[b, pos, :]  # [H]
                s = self.proj(h)
                scores_b.append(s)
            s = torch.stack(scores_b, dim=0).squeeze(-1)  # [K]
            out.append(s)
        # pad to same K per batch by right-padding with very negative scores
        K = max(x.size(0) for x in out)
        padded = []
        for s in out:
            if s.size(0) < K:
                pad = torch.full((K - s.size(0),), -1e9, device=s.device)
                s = torch.cat([s, pad], dim=0)
            padded.append(s)
        return torch.stack(padded, dim=0)  # [B,K]


def _locate_spans(input_ids: torch.Tensor, start_id: int, end_id: int) -> List[List[tuple]]:
    spans: List[List[tuple]] = []
    for seq in input_ids.tolist():
        starts = []
        tmp = []
        for i, tid in enumerate(seq):
            if tid == start_id:
                starts.append(i)
            elif tid == end_id and starts:
                st = starts.pop(0)
                tmp.append((st, i))
        spans.append(tmp)
    return spans


def build_batch(
    batch: List[GroupItem],
    tok: AutoTokenizer,
    max_len: int,
    cand_max_chars: int = 0,
    reorder_by_target: bool = False,
    max_cands_per_group: int = 0,
    pseudo_target_from_labels: bool = False,
):
    texts: List[str] = []
    targets: List[Optional[List[float]]] = []
    for x in batch:
        cands = x.cands
        tgt = x.target
        # reorder by target if requested and sizes match
        if reorder_by_target and tgt is not None and isinstance(tgt, (list, tuple)):
            try:
                idxs = list(range(len(cands)))
                # pad tgt to len(cands) if shorter so zip doesn't drop
                if len(tgt) < len(cands):
                    tgt = list(tgt) + [0.0] * (len(cands) - len(tgt))
                # sort by target desc
                idxs.sort(key=lambda i: float(tgt[i]) if i < len(tgt) and isinstance(tgt[i], (int, float)) else 0.0, reverse=True)
                if max_cands_per_group and max_cands_per_group > 0:
                    idxs = idxs[:max_cands_per_group]
                cands = [cands[i] for i in idxs]
                tgt = [tgt[i] if i < len(tgt) else 0.0 for i in idxs]
            except Exception:
                # fall back to original order
                pass
        else:
            if max_cands_per_group and max_cands_per_group > 0:
                cands = cands[:max_cands_per_group]
                if isinstance(tgt, (list, tuple)):
                    tgt = list(tgt)[:len(cands)]

        texts.append(format_prompt(x.query, cands, cand_max_chars=cand_max_chars))
        if tgt is None and pseudo_target_from_labels:
            # Build a simple non-uniform pseudo target from label*weight
            vals = []
            for c in cands:
                lab = c.get("label", 0.0)
                w = c.get("weight", 1.0)
                try:
                    v = float(lab) * float(w)
                except Exception:
                    v = 0.0
                if not (v == v) or v == float("inf") or v == float("-inf"):
                    v = 0.0
                vals.append(max(0.0, v))
            targets.append(vals)
        else:
            targets.append(tgt if tgt is not None else None)

    enc = tok(texts, max_length=max_len, truncation=True, padding=True, return_tensors="pt")
    start_id = tok.convert_tokens_to_ids("<CAND_START>")
    end_id = tok.convert_tokens_to_ids("<CAND_END>")
    cand_spans = _locate_spans(enc.input_ids, start_id, end_id)
    return enc, cand_spans, targets


def kl_divergence(p_star: torch.Tensor, p_hat: torch.Tensor, eps: float = 1e-8):
    p_star = p_star.clamp_min(eps)
    p_hat = p_hat.clamp_min(eps)
    return torch.sum(p_star * (p_star.log() - p_hat.log()), dim=-1).mean()


def _sanitize_target(tgt_list: Optional[List[float]], k_eff: int, device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    """Convert raw target list to a safe probability vector of length k_eff.
    - Replaces NaN/Inf with 0
    - Clamps negatives to 0
    - Truncates/pads to k_eff
    - Renormalizes; if sum==0 -> uniform
    """
    if k_eff <= 0:
        return torch.empty((0,), device=device, dtype=dtype)
    if tgt_list is None or (isinstance(tgt_list, (list, tuple)) and len(tgt_list) == 0):
        return torch.full((k_eff,), 1.0 / k_eff, device=device, dtype=dtype)
    t = torch.tensor(tgt_list, device=device, dtype=dtype)
    # Replace non-finite and negatives
    t = torch.nan_to_num(t, nan=0.0, posinf=0.0, neginf=0.0)
    t = torch.clamp(t, min=0.0)
    # Align length
    if t.numel() > k_eff:
        t = t[:k_eff]
    elif t.numel() < k_eff:
        t = torch.cat([t, torch.zeros(k_eff - t.numel(), device=device, dtype=dtype)], dim=0)
    s = t.sum()
    if not torch.isfinite(s) or s.item() <= 1e-12:
        t = torch.full((k_eff,), 1.0 / k_eff, device=device, dtype=dtype)
    else:
        t = t / s
    return t


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description="Train LLM scoring head with LoRA on listwise groups")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--dev", action="append", required=True)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-drop", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--cand-max-chars", type=int, default=512, help="Truncate each candidate text to this many chars to fit more candidates in context; 0 disables")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (may require disable use_cache)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    ap.add_argument("--reorder-by-target", action="store_true", help="Reorder candidates by target_scores desc to keep high-signal ones in context")
    ap.add_argument("--max-cands-per-group", type=int, default=16, help="Cap number of candidates per group included in the prompt")
    ap.add_argument("--pseudo-target-from-labels", action="store_true", help="If target_scores is missing, build pseudo targets from label*weight")
    ap.add_argument("--save", type=Path, required=True)
    args = ap.parse_args(argv)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and align special tokens
    res = ensure_special_tokens(args.model, load_kwargs={"torch_dtype": dtype, "device_map": "auto"})
    tok, base_model = res.tokenizer, res.model

    # Prepare LoRA
    if args.grad_checkpoint:
        try:
            base_model.gradient_checkpointing_enable()
            if hasattr(base_model.config, "use_cache"):
                base_model.config.use_cache = False
        except Exception:
            pass
    base_model.enable_input_require_grads()
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    )
    model = get_peft_model(base_model, lora_cfg)
    # Ensure head dtype matches base model to avoid BF16/FP32 matmul mismatch
    head = ScoreHead(base_model.config.hidden_size).to(device=device, dtype=next(base_model.parameters()).dtype)

    # Data
    train_ds = GroupsDataset(args.train)
    dev_ds = GroupsDataset(args.dev)
    # simple python batching to avoid collate complexity (batch=1 default)

    # Quick dataset diagnostics on a small sample
    try:
        sample_n = min(200, len(train_ds))
        has_target = 0
        target_uniform = 0
        k_stats = []
        for j in range(sample_n):
            gi = train_ds[j]
            enc, cand_pos, targets = build_batch([gi], tok, args.max_len, cand_max_chars=args.cand_max_chars, reorder_by_target=args.reorder_by_target, max_cands_per_group=args.max_cands_per_group, pseudo_target_from_labels=args.pseudo_target_from_labels)
            k_eff = len(cand_pos[0]) if cand_pos else 0
            k_stats.append(k_eff)
            tlist = gi.target
            if tlist is not None and isinstance(tlist, (list, tuple)) and len(tlist) > 0 and all([isinstance(x, (int, float)) for x in tlist]):
                # finite sum?
                import math
                ssum = sum([0.0 if (x is None or (isinstance(x, float) and (math.isnan(x) or math.isinf(x)))) else float(x) for x in tlist])
                if ssum > 0:
                    has_target += 1
                    # check near-uniform in first k_eff entries
                    tfirst = tlist[:k_eff] if k_eff > 0 else tlist
                    if len(tfirst) > 1:
                        mx, mn = max(tfirst), min(tfirst)
                        if (mx - mn) <= 1e-6:
                            target_uniform += 1
        if k_stats:
            import statistics as _st
            k_avg = sum(k_stats) / len(k_stats)
            k_min, k_max = min(k_stats), max(k_stats)
        else:
            k_avg = k_min = k_max = 0
        print(f"[diag] train items={len(train_ds)}, sample={sample_n}, with_target={has_target}/{sample_n}, uniform_targets~={target_uniform}, k_eff(avg/min/max)={k_avg:.1f}/{k_min}/{k_max}")
        if has_target == 0:
            print("[warn] No non-empty target_scores found. Training will default to uniform targets and will not learn. Build groups with teacher scores.")
        if k_avg <= 6:
            print("[hint] Very small k_eff. Consider reducing --max-len or increasing --cand-max-chars truncation to fit more candidates.")
    except Exception as _e:
        print(f"[diag] dataset diagnostics skipped due to: {_e}")

    from transformers import get_linear_schedule_with_warmup
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=args.lr, weight_decay=args.wd)
    steps_per_epoch = max(1, (len(train_ds) + args.batch - 1) // args.batch)
    total_updates = steps_per_epoch * args.epochs
    warmup_updates = int(args.warmup * total_updates)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_updates, num_training_steps=total_updates)
    best = float("inf")

    for ep in range(1, args.epochs + 1):
        model.train(); head.train()
        pbar = tqdm(range(0, len(train_ds), args.batch), desc=f"train ep{ep}", dynamic_ncols=True)
        accum = 0
        for i in pbar:
            batch = [train_ds[j] for j in range(i, min(i + args.batch, len(train_ds)))]
            enc, cand_pos, targets = build_batch(batch, tok, args.max_len, cand_max_chars=args.cand_max_chars, reorder_by_target=args.reorder_by_target, max_cands_per_group=args.max_cands_per_group, pseudo_target_from_labels=args.pseudo_target_from_labels)
            for k in enc:
                enc[k] = enc[k].to(device)
            out = model(**enc, output_hidden_states=True)
            last = out.hidden_states[-1]
            # ensure head on same device/dtype as last
            if (next(head.parameters()).device != last.device) or (next(head.parameters()).dtype != last.dtype):
                head.to(device=last.device, dtype=last.dtype)
            s = head(last, cand_pos)  # [B,K]
            # mask padded slots
            B, K = s.size(0), s.size(1)
            mask = torch.zeros((B, K), dtype=torch.bool, device=s.device)
            for b, idxs in enumerate(cand_pos):
                k = min(len(idxs), K)
                if k > 0:
                    mask[b, :k] = True
            # masked mean/std in float32 for stability
            # sanitize raw scores first
            s_f = torch.nan_to_num(s.float(), nan=0.0, posinf=0.0, neginf=0.0)
            counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)
            sum_s = (s_f.masked_fill(~mask, 0.0)).sum(dim=-1, keepdim=True)
            mean = sum_s / counts
            var = (((s_f - mean).masked_fill(~mask, 0.0)) ** 2).sum(dim=-1, keepdim=True) / counts
            std = var.sqrt().clamp_min(1e-4)
            s_norm = ((s_f - mean) / std).masked_fill(~mask, -1e9)
            s_norm = torch.clamp(torch.nan_to_num(s_norm, nan=-1e9, posinf=30.0, neginf=-30.0), min=-30.0, max=30.0)
            # Per-sample CE over valid candidates only (in float32)
            valid_losses = []
            for b, tgt in enumerate(targets):
                k_eff = int(mask[b].sum().item())
                if k_eff == 0:
                    continue
                logp = torch.log_softmax(s_norm[b, :k_eff], dim=-1)
                if not torch.isfinite(logp).all():
                    # fallback to uniform if numerical issues
                    import math
                    logp = torch.full((k_eff,), -math.log(max(1, k_eff)), device=s_norm.device, dtype=s_norm.dtype)
                t = _sanitize_target(tgt, k_eff, device=logp.device, dtype=logp.dtype)
                loss_b = -(t * logp).sum()
                valid_losses.append(loss_b)

            if not valid_losses:
                # no valid candidates in this batch, skip backward and step accumulation counter to avoid hang
                accum += 1
                if accum % args.grad_accum == 0:
                    opt.zero_grad(set_to_none=True); sched.step()
                pbar.set_postfix(loss="skip")
                continue

            loss = (torch.stack(valid_losses).mean()) / max(1, args.grad_accum)
            # Guard against non-finite loss
            if not torch.isfinite(loss):
                pbar.set_postfix(loss="nan-skip")
                accum += 1
                if accum % args.grad_accum == 0:
                    opt.zero_grad(set_to_none=True); sched.step()
                continue
            loss.backward()
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), args.max_grad_norm)
            accum += 1
            if accum % args.grad_accum == 0:
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
            pbar.set_postfix(loss=f"{(loss.item()*max(1,args.grad_accum)):.4f}")

        # Dev loss
        model.eval(); head.eval()
        dev_losses = []
        with torch.no_grad():
            for j in range(len(dev_ds)):
                enc, cand_pos, targets = build_batch([dev_ds[j]], tok, args.max_len, cand_max_chars=args.cand_max_chars, reorder_by_target=args.reorder_by_target, max_cands_per_group=args.max_cands_per_group, pseudo_target_from_labels=args.pseudo_target_from_labels)
                for k in enc:
                    enc[k] = enc[k].to(device)
                out = model(**enc, output_hidden_states=True)
                last = out.hidden_states[-1]
                if (next(head.parameters()).device != last.device) or (next(head.parameters()).dtype != last.dtype):
                    head.to(device=last.device, dtype=last.dtype)
                s = head(last, cand_pos)
                B, K = s.size(0), s.size(1)
                mask = torch.zeros((B, K), dtype=torch.bool, device=s.device)
                for b, idxs in enumerate(cand_pos):
                    k = min(len(idxs), K)
                    if k > 0:
                        mask[b, :k] = True
                counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                s_f = torch.nan_to_num(s.float(), nan=0.0, posinf=0.0, neginf=0.0)
                sum_s = (s_f.masked_fill(~mask, 0.0)).sum(dim=-1, keepdim=True)
                mean = sum_s / counts
                var = (((s_f - mean).masked_fill(~mask, 0.0)) ** 2).sum(dim=-1, keepdim=True) / counts
                std = var.sqrt().clamp_min(1e-4)
                s_norm = ((s_f - mean) / std).masked_fill(~mask, -1e9)
                s_norm = torch.clamp(torch.nan_to_num(s_norm, nan=-1e9, posinf=30.0, neginf=-30.0), min=-30.0, max=30.0)
                # CE on dev
                k_eff = int(mask[0].sum().item())
                if k_eff == 0:
                    continue
                logp = torch.log_softmax(s_norm[0, :k_eff], dim=-1)
                if not torch.isfinite(logp).all():
                    import math
                    logp = torch.full((k_eff,), -math.log(max(1, k_eff)), device=s_norm.device, dtype=s_norm.dtype)
                t = _sanitize_target(targets[0], k_eff, device=logp.device, dtype=logp.dtype)
                dev_losses.append((-(t * logp).sum()).item())
        dev_loss = sum(dev_losses) / max(1, len(dev_losses))
        print(f"dev_kl={dev_loss:.4f}")
        if dev_loss < best:
            best = dev_loss
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                "base_model": args.model,
                "lora": model.state_dict(),
                "head": head.state_dict(),
                "tokenizer_len": len(tok),
                "lora_cfg": {
                    "r": args.lora_r,
                    "lora_alpha": args.lora_alpha,
                    "lora_dropout": args.lora_drop,
                    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
                },
                "special": {
                    "cand_start": res.cand_start_id,
                    "cand_end": res.cand_end_id,
                },
            }, args.save)
            print(f"saved: {args.save}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
