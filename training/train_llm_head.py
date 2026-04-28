#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import List, Optional, Tuple

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


def format_prompt(q: str, cands: List[dict], cand_max_chars: int = 0, q_max_chars: int = 0, no_tags: bool = False) -> str:
    if isinstance(q, str) and q_max_chars and q_max_chars > 0 and len(q) > q_max_chars:
        q = q[:q_max_chars]
    parts = ["[CONJECTURE]\n", f"<Q> {q} </Q>\n"]
    for i, c in enumerate(cands, 1):
        tags = c.get("meta") or {}
        # simple tag rendering
        tag_str = "" if no_tags else " ".join(f"<{k}={v}>" for k, v in tags.items() if v is not None)
        text = c.get('text', '')
        if cand_max_chars and cand_max_chars > 0 and isinstance(text, str) and len(text) > cand_max_chars:
            text = text[:cand_max_chars]
        parts.append(f"[CANDIDATE {i}]\n<CAND_START>\n{text}\n{tag_str}\n<CAND_END>\n")
    return "".join(parts)


class ScoreHead(nn.Module):
    def __init__(self, hidden: int, pool: str = "end", query_aware: bool = False, mlp_hidden: int = 0):
        super().__init__()
        self.pool = pool  # 'mean'|'start'|'end'
        self.query_aware = query_aware
        in_dim = hidden * 3 if query_aware else hidden
        if mlp_hidden and mlp_hidden > 0:
            self.proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Linear(in_dim, mlp_hidden),
                nn.GELU(),
                nn.Linear(mlp_hidden, 1),
            )
        else:
            self.proj = nn.Sequential(
                nn.LayerNorm(in_dim),
                nn.GELU(),
                nn.Linear(in_dim, 1),
            )

    def _pool_span(self, last_hidden: torch.Tensor, b: int, T: int, st: int, ed: int) -> torch.Tensor:
        if self.pool == "mean":
            st_eff = max(0, st + 1)
            ed_eff = max(st_eff + 1, min(T, ed))
            hs = last_hidden[b, st_eff:ed_eff, :]
            if hs.numel() == 0:
                hs = last_hidden[b, st:st+1, :]
            return hs.mean(dim=0)
        elif self.pool == "start":
            pos = max(0, min(T - 1, st))
            return last_hidden[b, pos, :]
        else:  # 'end'
            pos = max(0, min(T - 1, ed))
            return last_hidden[b, pos, :]

    def forward(self, last_hidden: torch.Tensor, cand_pos_or_spans: List[List], query_vecs: Optional[torch.Tensor] = None):
        B, T, H = last_hidden.size()
        out = []
        for b in range(B):
            entries = cand_pos_or_spans[b]
            if not entries:
                out.append(torch.zeros(1, device=last_hidden.device))
                continue
            scores_b = []
            qv = query_vecs[b] if (self.query_aware and query_vecs is not None) else None
            for e in entries:
                if isinstance(e, (list, tuple)) and len(e) == 2:
                    st, ed = int(e[0]), int(e[1])
                    h = self._pool_span(last_hidden, b, T, st, ed)
                else:
                    pos = int(e)
                    h = last_hidden[b, pos, :]
                if qv is not None:
                    hv = torch.cat([h, qv, h * qv], dim=-1)
                else:
                    hv = h
                s = self.proj(hv)
                scores_b.append(s)
            s = torch.stack(scores_b, dim=0).squeeze(-1)
            out.append(s)
        K = max(x.size(0) for x in out)
        padded = []
        for s in out:
            if s.size(0) < K:
                pad = torch.full((K - s.size(0),), -1e9, device=s.device)
                s = torch.cat([s, pad], dim=0)
            padded.append(s)
        return torch.stack(padded, dim=0)


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

def _locate_query_vec(last_hidden: torch.Tensor, input_ids: torch.Tensor, q_id: int, q_end_id: int, pool: str = "mean") -> torch.Tensor:
    B, T, H = last_hidden.size()
    q_vecs: List[torch.Tensor] = []
    for b, seq in enumerate(input_ids.tolist()):
        st = None; ed = None
        for i, tid in enumerate(seq):
            if tid == q_id and st is None:
                st = i
            elif tid == q_end_id and st is not None:
                ed = i; break
        if st is None or ed is None:
            qv = last_hidden[b].mean(dim=0)
        else:
            if pool == "mean":
                st_eff = max(0, st + 1)
                ed_eff = max(st_eff + 1, min(T, ed))
                hs = last_hidden[b, st_eff:ed_eff, :]
                if hs.numel() == 0:
                    hs = last_hidden[b, st:st+1, :]
                qv = hs.mean(dim=0)
            elif pool == "start":
                qv = last_hidden[b, max(0, st), :]
            else:
                qv = last_hidden[b, max(0, min(T-1, ed)), :]
        q_vecs.append(qv)
    return torch.stack(q_vecs, dim=0)


def build_batch(
    batch: List[GroupItem],
    tok: AutoTokenizer,
    max_len: int,
    cand_max_chars: int = 0,
    q_max_chars: int = 0,
    no_tags: bool = False,
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
        texts.append(format_prompt(x.query, cands, cand_max_chars=cand_max_chars, q_max_chars=q_max_chars, no_tags=no_tags))
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
    ap.add_argument("--lora-include-mlp", action="store_true", help="Also apply LoRA to MLP projection layers (gate/up/down)")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--max-len", type=int, default=1024)
    ap.add_argument("--cand-max-chars", type=int, default=512, help="Truncate each candidate text to this many chars to fit more candidates in context; 0 disables")
    ap.add_argument("--q-max-chars", type=int, default=0, help="Truncate query text to this many chars; 0 disables")
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=16)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--grad-checkpoint", action="store_true", help="Enable gradient checkpointing (may require disable use_cache)")
    ap.add_argument("--max-grad-norm", type=float, default=1.0, help="Gradient clipping norm (0 to disable)")
    ap.add_argument("--reorder-by-target", action="store_true", help="Reorder candidates by target_scores desc to keep high-signal ones in context")
    ap.add_argument("--max-cands-per-group", type=int, default=16, help="Cap number of candidates per group included in the prompt")
    ap.add_argument("--pseudo-target-from-labels", action="store_true", help="If target_scores is missing, build pseudo targets from label*weight")
    ap.add_argument("--no-tags", action="store_true", help="Do not render candidate meta tags in the prompt")
    ap.add_argument("--pool", choices=["mean", "start", "end", "span"], default="end", help="Token pooling inside a candidate span ('span' is alias of 'mean')")
    ap.add_argument("--no-zscore", action="store_true", help="Disable per-group z-score normalization of logits before softmax")
    ap.add_argument("--tau", type=float, default=1.0, help="Temperature after z-score: p=softmax((z-b)/tau)")
    ap.add_argument("--bias", type=float, default=0.0, help="Bias after z-score: p=softmax((z-b)/tau)")
    ap.add_argument("--filter-uniform", type=float, default=-1.0, help="Skip groups whose target entropy H >= ln(K)-threshold; set negative to disable")
    ap.add_argument("--freeze-lora", action="store_true", help="Freeze LoRA/base model, train head only")
    ap.add_argument("--head-lr", type=float, default=None, help="Learning rate for head (defaults to --lr)")
    ap.add_argument("--lora-lr", type=float, default=None, help="Learning rate for LoRA/base model (defaults to --lr)")
    ap.add_argument("--rank-loss", choices=["none","pairwise","listmle"], default="none")
    ap.add_argument("--rank-weight", type=float, default=1.0)
    ap.add_argument("--pair-r", type=int, default=3)
    ap.add_argument("--pair-temp", type=float, default=1.0)
    ap.add_argument("--pair-margin", type=float, default=0.0)
    ap.add_argument("--query-aware", action="store_true")
    ap.add_argument("--head-hidden", type=int, default=0, help="Hidden size for MLP head (0=single layer)")
    ap.add_argument("--diag-only", action="store_true", help="Run diagnostics and exit before training")
    ap.add_argument("--diag-sample", type=int, default=200, help="How many training items to sample for diagnostics (-1 for all)")
    ap.add_argument("--sanity", action="store_true", help="Print a detailed sanity check for one sample")
    ap.add_argument("--sanity-index", type=int, default=0, help="Index of the sample to use for sanity check")
    ap.add_argument("--use-flash", action="store_true", help="Try to enable flash attention if supported")
    ap.add_argument("--save", type=Path, required=True)
    args = ap.parse_args(argv)

    dtype = torch.bfloat16 if args.bf16 else torch.float32
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load and align special tokens
    res = ensure_special_tokens(args.model, load_kwargs={"dtype": dtype, "device_map": "auto"})
    tok, base_model = res.tokenizer, res.model
    if args.use_flash:
        try:
            if hasattr(base_model.config, "attn_implementation"):
                base_model.config.attn_implementation = "flash_attention_2"
        except Exception:
            pass

    # Prepare LoRA
    if args.grad_checkpoint:
        try:
            base_model.gradient_checkpointing_enable()
            if hasattr(base_model.config, "use_cache"):
                base_model.config.use_cache = False
        except Exception:
            pass
    base_model.enable_input_require_grads()
    target_modules = ["q_proj", "k_proj", "v_proj", "o_proj"]
    if args.lora_include_mlp:
        target_modules += ["gate_proj", "up_proj", "down_proj"]
    lora_cfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )
    model = get_peft_model(base_model, lora_cfg)
    # Ensure head dtype matches base model to avoid BF16/FP32 matmul mismatch
    head_pool = "mean" if args.pool == "span" else args.pool
    head = ScoreHead(base_model.config.hidden_size, pool=head_pool, query_aware=args.query_aware, mlp_hidden=args.head_hidden).to(device=device, dtype=next(base_model.parameters()).dtype)

    # Data
    train_ds = GroupsDataset(args.train)
    dev_ds = GroupsDataset(args.dev)
    # simple python batching to avoid collate complexity (batch=1 default)

    # Quick dataset diagnostics on a small sample
    try:
        sample_n = len(train_ds) if (args.diag_sample is not None and args.diag_sample < 0) else min(args.diag_sample or 200, len(train_ds))
        has_target = 0
        target_uniform = 0
        zero_k = 0
        k_stats = []
        for j in range(sample_n):
            gi = train_ds[j]
            enc, cand_pos, targets = build_batch(
                [gi], tok, args.max_len,
                cand_max_chars=args.cand_max_chars,
                q_max_chars=args.q_max_chars,
                no_tags=args.no_tags,
                reorder_by_target=args.reorder_by_target,
                max_cands_per_group=args.max_cands_per_group,
                pseudo_target_from_labels=args.pseudo_target_from_labels,
            )
            k_eff = len(cand_pos[0]) if cand_pos else 0
            k_stats.append(k_eff)
            if k_eff == 0:
                zero_k += 1
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
        # quartiles
        if k_stats:
            k_sorted = sorted(k_stats)
            p25 = k_sorted[int(0.25 * (len(k_sorted) - 1))]
            p50 = k_sorted[int(0.50 * (len(k_sorted) - 1))]
            p75 = k_sorted[int(0.75 * (len(k_sorted) - 1))]
            print(f"[diag] train items={len(train_ds)}, sample={sample_n}, with_target={has_target}/{sample_n}, uniform_targets~={target_uniform}, zero_k={zero_k}, k_eff(avg/min/max;p25/median/p75)={k_avg:.1f}/{k_min}/{k_max};{p25}/{p50}/{p75}")
        else:
            print(f"[diag] train items={len(train_ds)}, sample={sample_n}, with_target={has_target}/{sample_n}, uniform_targets~={target_uniform}, zero_k={zero_k}, k_eff(avg/min/max;p25/median/p75)=0/0/0;0/0/0")
        if has_target == 0:
            print("[warn] No non-empty target_scores found. Training will default to uniform targets and will not learn. Build groups with teacher scores.")
        if k_avg <= 6:
            print("[hint] Very small k_eff. Consider reducing --max-len or increasing --cand-max-chars truncation to fit more candidates.")
    except Exception as _e:
        print(f"[diag] dataset diagnostics skipped due to: {_e}")

    # Optional sanity check on a single sample
    if args.sanity and len(train_ds) > 0:
        idx = max(0, min(len(train_ds) - 1, args.sanity_index))
        gi = train_ds[idx]
        print(f"[sanity] sample_idx={idx}, problem_name={gi.problem_name}")
        enc, cand_spans, targets = build_batch(
            [gi], tok, args.max_len,
            cand_max_chars=args.cand_max_chars,
            q_max_chars=args.q_max_chars,
            no_tags=args.no_tags,
            reorder_by_target=args.reorder_by_target,
            max_cands_per_group=args.max_cands_per_group,
            pseudo_target_from_labels=args.pseudo_target_from_labels,
        )
        starts = [st for (st, ed) in cand_spans[0]] if cand_spans and cand_spans[0] else []
        print(f"[sanity] cand_positions(len)={len(starts)}, positions={starts}")
        for k in enc: enc[k] = enc[k].to(device)
        out = model(**enc, output_hidden_states=True)
        last = out.hidden_states[-1]
        # candidate vectors per pooling
        vecs = []
        for (st, ed) in cand_spans[0]:
            if head_pool == "mean":
                st_eff = max(0, st + 1); ed_eff = max(st_eff + 1, min(last.size(1), ed))
                hs = last[0, st_eff:ed_eff, :]
                if hs.numel() == 0:
                    hs = last[0, st:st+1, :]
                v = hs.mean(dim=0)
            elif head_pool == "start":
                v = last[0, max(0, st), :]
            else:
                v = last[0, max(0, min(last.size(1)-1, ed)), :]
            vecs.append(v)
        if len(vecs) >= 2:
            v0 = vecs[0]
            sims = []
            for v in vecs:
                num = torch.dot(v0, v)
                den = (v0.norm() * v.norm()).clamp_min(1e-8)
                sims.append((num/den).detach().float().cpu().item())
            print(f"[sanity] cosine_vs_first(min/max)={min(sims):.4f}/{max(sims):.4f}; first10={sims[:10]}")
        s = head(last, cand_spans)
        print(f"[sanity] head_scores(min/mean/max)={float(s.min()):.4f}/{float(s.mean()):.4f}/{float(s.max()):.4f}")
        # teacher entropy
        k_eff = len(cand_spans[0]) if cand_spans else 0
        if k_eff > 0:
            t = _sanitize_target(gi.target, k_eff, device=s.device, dtype=torch.float32)
            eps = 1e-12
            H = float((-(t * (t.clamp_min(eps).log()))).sum().cpu().item())
            import math
            lnK = float(math.log(k_eff))
            tmax = float(t.max().cpu().item())
            print(f"[sanity] teacher_entropy={H:.4f} vs ln(k)={lnK:.4f}, teacher_maxprob={tmax:.4f}")
        # trainable params
        try:
            n_model_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
            n_head_train = sum(p.numel() for p in head.parameters() if p.requires_grad)
            print(f"[sanity] trainable_params(model)={n_model_train}, trainable_params(head)={n_head_train}")
        except Exception:
            pass
        # one backward pass to see head grad magnitude
        try:
            B, K = s.size(0), s.size(1)
            mask = torch.zeros((B, K), dtype=torch.bool, device=s.device)
            mask[0, :k_eff] = True
            s_f = s.float()
            if args.no_zscore:
                s_norm = s_f.masked_fill(~mask, -1e9)
            else:
                counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                sum_s = (s_f.masked_fill(~mask, 0.0)).sum(dim=-1, keepdim=True)
                mean = sum_s / counts
                var = (((s_f - mean).masked_fill(~mask, 0.0)) ** 2).sum(dim=-1, keepdim=True) / counts
                std = var.sqrt().clamp_min(1e-4)
                s_norm = ((s_f - mean) / std).masked_fill(~mask, -1e9)
            logp = torch.log_softmax(s_norm[0, :k_eff], dim=-1)
            t = _sanitize_target(gi.target, k_eff, device=logp.device, dtype=logp.dtype)
            loss_s = -(t * logp).sum()
            for m in model.parameters():
                if m.grad is not None: m.grad = None
            for m in head.parameters():
                if m.grad is not None: m.grad = None
            loss_s.backward()
            g = head.proj.weight.grad
            gnorm = float((g.norm().detach().cpu().item()) if g is not None else 0.0)
            print(f"[sanity] head_grad_l2={gnorm:.6f}")
        except Exception:
            pass
    if args.diag_only:
        return 0

    from transformers import get_linear_schedule_with_warmup
    # Freeze LoRA/base if requested
    if args.freeze_lora:
        for p in model.parameters():
            p.requires_grad = False
        for p in head.parameters():
            p.requires_grad = True
    lr_head = args.head_lr if args.head_lr is not None else args.lr
    lr_lora = args.lora_lr if args.lora_lr is not None else args.lr
    # Build param groups
    params = []
    head_params = [p for p in head.parameters() if p.requires_grad]
    if head_params:
        params.append({"params": head_params, "lr": lr_head})
    lora_params = [p for p in model.parameters() if p.requires_grad]
    if lora_params:
        params.append({"params": lora_params, "lr": lr_lora})
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)
    micro_steps_per_epoch = max(1, (len(train_ds) + args.batch - 1) // args.batch)
    optimizer_steps_per_epoch = max(1, (micro_steps_per_epoch + args.grad_accum - 1) // args.grad_accum)
    total_optimizer_steps = optimizer_steps_per_epoch * args.epochs
    warmup_optimizer_steps = int(args.warmup * total_optimizer_steps)
    sched = get_linear_schedule_with_warmup(opt, num_warmup_steps=warmup_optimizer_steps, num_training_steps=total_optimizer_steps)
    best = float("inf")

    # quick summary of trainable params
    try:
        n_total = sum(p.numel() for p in model.parameters()) + sum(p.numel() for p in head.parameters())
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad) + sum(p.numel() for p in head.parameters() if p.requires_grad)
        print(f"trainable_params={n_train}/{n_total} ({n_train/n_total*100:.2f}%)")
    except Exception:
        pass

    for ep in range(1, args.epochs + 1):
        model.train(); head.train()
        pbar = tqdm(range(0, len(train_ds), args.batch), desc=f"train ep{ep}", dynamic_ncols=True)
        accum = 0
        step_idx = 0
        for i in pbar:
            batch = [train_ds[j] for j in range(i, min(i + args.batch, len(train_ds)))]
            enc, cand_spans, targets = build_batch(
                batch, tok, args.max_len,
                cand_max_chars=args.cand_max_chars,
                q_max_chars=args.q_max_chars,
                no_tags=args.no_tags,
                reorder_by_target=args.reorder_by_target,
                max_cands_per_group=args.max_cands_per_group,
                pseudo_target_from_labels=args.pseudo_target_from_labels,
            )
            for k in enc:
                enc[k] = enc[k].to(device)
            out = model(**enc, output_hidden_states=True)
            last = out.hidden_states[-1]
            # ensure head on same device/dtype as last
            if (next(head.parameters()).device != last.device) or (next(head.parameters()).dtype != last.dtype):
                head.to(device=last.device, dtype=last.dtype)
            # query-aware vector
            q_vecs = None
            if args.query_aware:
                q_id = tok.convert_tokens_to_ids("<Q>")
                q_end_id = tok.convert_tokens_to_ids("</Q>")
                q_vecs = _locate_query_vec(last, enc["input_ids"], q_id, q_end_id, pool=head_pool)
            # choose pooling: pass spans and let head pick start/end/mean
            s = head(last, cand_spans, q_vecs)  # [B,K]
            # mask padded slots
            B, K = s.size(0), s.size(1)
            mask = torch.zeros((B, K), dtype=torch.bool, device=s.device)
            for b, idxs in enumerate(cand_spans):
                k = min(len(idxs), K)
                if k > 0:
                    mask[b, :k] = True
            # sanitize raw scores first (float32 for stability) + zscore + calibration
            s_f = torch.nan_to_num(s.float(), nan=0.0, posinf=0.0, neginf=0.0)
            if args.no_zscore:
                s_logits = s_f
            else:
                counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                sum_s = (s_f.masked_fill(~mask, 0.0)).sum(dim=-1, keepdim=True)
                mean = sum_s / counts
                var = (((s_f - mean).masked_fill(~mask, 0.0)) ** 2).sum(dim=-1, keepdim=True) / counts
                std = var.sqrt().clamp_min(1e-6)
                s_logits = (s_f - mean) / std
            s_norm = ((s_logits - float(args.bias)) / max(1e-6, float(args.tau))).masked_fill(~mask, -1e9)
            s_norm = torch.clamp(torch.nan_to_num(s_norm, nan=-1e9, posinf=30.0, neginf=-30.0), min=-30.0, max=30.0)
            # Per-sample CE over valid candidates only (in float32)
            valid_losses = []
            rank_losses = []
            skipped_uniform = 0
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
                # Filter near-uniform teacher groups if requested
                if args.filter_uniform is not None and args.filter_uniform >= 0.0:
                    import math
                    H = (-(t * (t.clamp_min(1e-12).log()))).sum().item()
                    if H >= math.log(k_eff) - float(args.filter_uniform):
                        skipped_uniform += 1
                        continue
                loss_b = -(t * logp).sum()
                valid_losses.append(loss_b)
                # Ranking loss
                if args.rank_loss != "none":
                    s_slice = s_norm[b, :k_eff]
                    if args.rank_loss == "pairwise":
                        vals = t.detach().cpu().tolist()
                        idxs = list(range(k_eff))
                        idxs.sort(key=lambda i: vals[i], reverse=True)
                        r = min(args.pair_r, max(1, k_eff // 2))
                        pos_idx = idxs[:r]; neg_idx = idxs[-r:]
                        diffs = []
                        for pi in pos_idx:
                            for ni in neg_idx:
                                diffs.append(s_slice[pi] - s_slice[ni])
                        if diffs:
                            diffs_t = torch.stack(diffs)
                            if args.pair_margin and args.pair_margin > 0:
                                rl = torch.relu(float(args.pair_margin) - diffs_t).mean()
                            else:
                                rl = torch.log1p(torch.exp(-diffs_t / max(1e-6, float(args.pair_temp)))).mean()
                            rank_losses.append(rl)
                    else:  # listmle
                        order = torch.argsort(t, descending=True)
                        s_ord = s_slice[order]
                        parts = []
                        for j in range(s_ord.size(0)):
                            parts.append(torch.logsumexp(s_ord[j:], dim=0) - s_ord[j])
                        if parts:
                            rank_losses.append(torch.stack(parts).sum())

            performed_opt_step = False
            if not valid_losses:
                # no valid candidates in this batch, skip backward and step accumulation counter to avoid hang
                accum += 1
                if accum % args.grad_accum == 0:
                    # don't advance scheduler without an optimizer step
                    opt.zero_grad(set_to_none=True)
                pbar.set_postfix(loss="skip")
                continue

            loss_ce = torch.stack(valid_losses).mean()
            if rank_losses:
                loss_rank = torch.stack(rank_losses).mean()
                loss_total = loss_ce + float(args.rank_weight) * loss_rank
            else:
                loss_rank = None
                loss_total = loss_ce
            loss = loss_total / max(1, args.grad_accum)
            # Guard against non-finite loss
            if not torch.isfinite(loss):
                pbar.set_postfix(loss="nan-skip")
                accum += 1
                if accum % args.grad_accum == 0:
                    opt.zero_grad(set_to_none=True)
                continue
            loss.backward()
            # lightweight grad norms (every ~200 micro-steps)
            step_idx += 1
            if step_idx % 200 == 0:
                try:
                    import math as _m
                    head_g = head.proj.weight.grad
                    gnorm_head = float(head_g.norm().item()) if head_g is not None else 0.0
                    # pick a representative LoRA param if present
                    gnorm_lora = None
                    for n, p in model.named_parameters():
                        if p.requires_grad and p.grad is not None and ("lora_A" in n or "lora_B" in n):
                            gnorm_lora = float(p.grad.norm().item())
                            break
                    if gnorm_lora is None:
                        gnorm_lora = 0.0
                    pbar.set_postfix(loss=f"{(loss.item()*max(1,args.grad_accum)):.4f}", g_head=f"{gnorm_head:.2e}", g_lora=f"{gnorm_lora:.2e}")
                except Exception:
                    pass
            if args.max_grad_norm and args.max_grad_norm > 0:
                torch.nn.utils.clip_grad_norm_(list(model.parameters()) + list(head.parameters()), args.max_grad_norm)
            accum += 1
            if accum % args.grad_accum == 0:
                opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
                performed_opt_step = True
            # show lr occasionally
            if step_idx % 100 == 0:
                try:
                    lrs = [pg.get("lr", 0.0) for pg in opt.param_groups]
                    lr_cur = max(lrs) if lrs else 0.0
                    extra = {} if skipped_uniform == 0 else {"skipU": skipped_uniform}
                    if loss_rank is not None:
                        extra.update({"rank": f"{(loss_rank.item() if isinstance(loss_rank, torch.Tensor) else loss_rank):.3f}"})
                    pbar.set_postfix(loss=f"{(loss.item()*max(1,args.grad_accum)):.4f}", lr=f"{lr_cur:.2e}", **extra)
                except Exception:
                    pbar.set_postfix(loss=f"{(loss.item()*max(1,args.grad_accum)):.4f}")

        # Dev loss
        model.eval(); head.eval()
        dev_losses = []
        with torch.no_grad():
            for j in range(len(dev_ds)):
                enc, cand_spans, targets = build_batch(
                    [dev_ds[j]], tok, args.max_len,
                    cand_max_chars=args.cand_max_chars,
                    q_max_chars=args.q_max_chars,
                    no_tags=args.no_tags,
                    reorder_by_target=args.reorder_by_target,
                    max_cands_per_group=args.max_cands_per_group,
                    pseudo_target_from_labels=args.pseudo_target_from_labels,
                )
                for k in enc:
                    enc[k] = enc[k].to(device)
                out = model(**enc, output_hidden_states=True)
                last = out.hidden_states[-1]
                if (next(head.parameters()).device != last.device) or (next(head.parameters()).dtype != last.dtype):
                    head.to(device=last.device, dtype=last.dtype)
                q_vecs = None
                if args.query_aware:
                    q_id = tok.convert_tokens_to_ids("<Q>")
                    q_end_id = tok.convert_tokens_to_ids("</Q>")
                    q_vecs = _locate_query_vec(last, enc["input_ids"], q_id, q_end_id, pool=head_pool)
                s = head(last, cand_spans, q_vecs)
                B, K = s.size(0), s.size(1)
                mask = torch.zeros((B, K), dtype=torch.bool, device=s.device)
                for b, idxs in enumerate(cand_spans):
                    k = min(len(idxs), K)
                    if k > 0:
                        mask[b, :k] = True
                s_f = torch.nan_to_num(s.float(), nan=0.0, posinf=0.0, neginf=0.0)
                if args.no_zscore:
                    s_logits = s_f
                else:
                    counts = mask.sum(dim=-1, keepdim=True).clamp(min=1)
                    sum_s = (s_f.masked_fill(~mask, 0.0)).sum(dim=-1, keepdim=True)
                    mean = sum_s / counts
                    var = (((s_f - mean).masked_fill(~mask, 0.0)) ** 2).sum(dim=-1, keepdim=True) / counts
                    std = var.sqrt().clamp_min(1e-6)
                    s_logits = (s_f - mean) / std
                s_norm = ((s_logits - float(args.bias)) / max(1e-6, float(args.tau))).masked_fill(~mask, -1e9)
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
                    "target_modules": target_modules,
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
