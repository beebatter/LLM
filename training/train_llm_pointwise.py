#!/usr/bin/env python3
"""Pointwise LLM training (线路 B): 二分类 / 回归 / 可选分布 KL (组内 softmax)

支持：
  - 任务类型: cls / reg / cls+reg
  - 数据格式: JSONL 每行包含
      {"query": "...", "candidate": "...", "label": 0/1, "score": 0.0-1.0, "group": "qid"}
    兼容字段别名: conjecture -> query; doc/text -> candidate; problem_name -> group
  - LoRA / QLoRA (4/8-bit) 量化加载
  - BCE + label smoothing + pos_weight (auto)；回归 MSE 或 Huber；可同时启用
  - 组内分布 KL: 先对同 group 内 logits 做 z-score + 温度 softmax，再与归一化 target score 比；可加权
  - 指标: AUC (cls), MSE/MAE/Pearson/Spearman (reg), NDCG@k / Hit@k (按 group)；支持早停

示例：
python -m LLM.training.train_llm_pointwise \
  --model /root/autodl-tmp/models/Goedel-Prover-V2-32B \
  --train /root/autodl-tmp/Training/datasets/pointwise.train.jsonl \
  --dev   /root/autodl-tmp/Training/datasets/pointwise.dev.jsonl \
  --task cls \
  --lora-r 16 --lora-alpha 32 --lora-drop 0.05 --bits 4 \
  --lr 5e-5 --epochs 2 --batch 1 --grad-accum 64 --bf16 --warmup 0.03 \
  --max-len 896 --label-smoothing 0.05 --pos-weight auto \
  --save /root/autodl-tmp/Training/models/pointwise32b.pt
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
from transformers import BitsAndBytesConfig  # type: ignore
from peft import LoraConfig, get_peft_model  # type: ignore


@dataclass
class Sample:
    query: str
    candidate: str
    label: Optional[float]
    score: Optional[float]
    group: str


class PointwiseDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.samples: List[Sample] = []
        for p in paths:
            with open(p, "r", encoding="utf-8", errors="ignore") as f:
                for ln in f:
                    try:
                        j = json.loads(ln)
                    except Exception:
                        continue
                    q = j.get("query") or j.get("conjecture") or ""
                    cand = j.get("candidate") or j.get("text") or j.get("doc") or j.get("candidate_text") or ""
                    if not cand and isinstance(j.get("candidate"), dict):  # candidate dict case
                        cdict = j.get("candidate")
                        cand = cdict.get("text", "") if isinstance(cdict, dict) else ""
                    group = j.get("group") or j.get("problem_name") or j.get("qid") or q[:32]
                    label = j.get("label")
                    score = j.get("score")
                    if not q or not cand:
                        continue
                    # Coerce numeric
                    try:
                        if label is not None:
                            label = float(label)
                    except Exception:
                        label = None
                    try:
                        if score is not None:
                            score = float(score)
                    except Exception:
                        score = None
                    self.samples.append(Sample(q, cand, label, score, str(group)))
        self.groups: Dict[str, List[int]] = defaultdict(list)
        for i, s in enumerate(self.samples):
            self.groups[s.group].append(i)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx: int) -> Sample:
        return self.samples[idx]


def build_prompt(q: str, cand: str) -> str:
    return f"[SCORE]\n[CONJECTURE]\n{q}\n[CANDIDATE]\n{cand}"


class ScoreHead(nn.Module):
    def __init__(self, hidden: int, mlp: int = 0, dropout: float = 0.0):
        super().__init__()
        layers: List[nn.Module] = []
        if mlp and mlp > 0:
            layers.extend([
                nn.LayerNorm(hidden),
                nn.Linear(hidden, mlp),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mlp, 1),
            ])
        else:
            layers.extend([
                nn.LayerNorm(hidden),
                nn.Linear(hidden, 1),
            ])
        self.net = nn.Sequential(*layers)

    def forward(self, h_cls: torch.Tensor) -> torch.Tensor:
        return self.net(h_cls).squeeze(-1)


def bce_smooth_logits(logits: torch.Tensor, targets: torch.Tensor, eps: float, pos_weight: Optional[torch.Tensor] = None):
    # label smoothing: y' = y*(1-eps) + 0.5*eps
    if eps > 0:
        targets = targets * (1 - eps) + 0.5 * eps
    # manual binary cross entropy with logits
    # loss = - y*logsigmoid(x) - (1-y)*logsigmoid(-x)
    lpos = torch.nn.functional.logsigmoid(logits)
    lneg = torch.nn.functional.logsigmoid(-logits)
    if pos_weight is not None:
        loss = -targets * lpos * pos_weight - (1 - targets) * lneg
    else:
        loss = -targets * lpos - (1 - targets) * lneg
    return loss.mean()


def kl_div(p_star: torch.Tensor, p_hat: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    p_star = p_star.clamp_min(eps)
    p_hat = p_hat.clamp_min(eps)
    return torch.sum(p_star * (p_star.log() - p_hat.log()), dim=-1).mean()


def ndcg_at_k(rel: List[float], pred: List[float], k: int) -> float:
    if not rel or not pred:
        return 0.0
    idx = list(range(len(rel)))
    order = sorted(idx, key=lambda i: pred[i], reverse=True)[:k]
    dcg = 0.0
    for j, i in enumerate(order):
        dcg += rel[i] / math.log2(j + 2)
    order2 = sorted(idx, key=lambda i: rel[i], reverse=True)[:k]
    idcg = 0.0
    for j, i in enumerate(order2):
        idcg += rel[i] / math.log2(j + 2)
    return dcg / idcg if idcg > 0 else 0.0


def spearman(a: List[float], b: List[float]) -> float:
    try:
        import scipy.stats as ss  # type: ignore
        r, _ = ss.spearmanr(a, b)
        return float(r) if isinstance(r, float) and r == r else float("nan")
    except Exception:
        return float("nan")


def build_argparser():
    ap = argparse.ArgumentParser(description="Train pointwise LLM (cls/reg/cls+reg) with LoRA")
    ap.add_argument("--model", required=True)
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--dev", action="append", required=True)
    ap.add_argument("--task", choices=["cls", "reg", "cls+reg"], default="cls")
    ap.add_argument("--reg-prob", action="store_true", help="Apply sigmoid before regression loss (score in [0,1])")
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--lora-drop", type=float, default=0.05)
    ap.add_argument("--bits", type=int, choices=[4,8,16], default=4)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--warmup", type=float, default=0.03)
    ap.add_argument("--batch", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=64)
    ap.add_argument("--max-len", type=int, default=896)
    ap.add_argument("--bf16", action="store_true")
    ap.add_argument("--max-grad-norm", type=float, default=1.0)
    ap.add_argument("--mlp-hidden", type=int, default=0)
    ap.add_argument("--mlp-drop", type=float, default=0.0)
    ap.add_argument("--label-smoothing", type=float, default=0.0)
    ap.add_argument("--pos-weight", type=str, default="none", help="none|auto|float value")
    ap.add_argument("--kl-weight", type=float, default=0.0, help=">0 启用组内 softmax KL")
    ap.add_argument("--kl-temp", type=float, default=1.0)
    ap.add_argument("--zscore", action="store_true")
    ap.add_argument("--eval-steps", type=int, default=1000)
    ap.add_argument("--early-patience", type=int, default=3)
    ap.add_argument("--save", type=Path, required=True)
    return ap


def main(argv: Optional[List[str]] = None) -> int:
    ap = build_argparser()
    args = ap.parse_args(argv)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cuda.matmul.allow_tf32 = True  # type: ignore
    torch.set_float32_matmul_precision("high")  # type: ignore

    # Tokenizer & special token
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None and tok.eos_token_id is not None:
        tok.pad_token = tok.eos_token
    added = 0
    if "[SCORE]" not in tok.get_vocab():
        tok.add_special_tokens({"additional_special_tokens": ["[SCORE]"]})
        added = 1

    # Quantization config
    quant: Optional[BitsAndBytesConfig] = None
    if args.bits in (4, 8):
        if args.bits == 4:
            quant = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
                bnb_4bit_compute_dtype=torch.bfloat16 if args.bf16 else torch.float16,
            )
        else:
            quant = BitsAndBytesConfig(load_in_8bit=True)
    torch_dtype = torch.bfloat16 if args.bf16 and torch.cuda.is_available() else torch.float16 if torch.cuda.is_available() else torch.float32
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch_dtype,
        device_map="auto" if torch.cuda.is_available() else None,
        quantization_config=quant,
    )
    if added:
        model.resize_token_embeddings(len(tok))

    # LoRA
    lcfg = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_drop,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj","k_proj","v_proj","o_proj"],
    )
    model = get_peft_model(model, lcfg)
    model.train()

    # 将 head 放到主 GPU；同时保持 dtype 与主参数一致
    primary_param = next(model.parameters())
    head = ScoreHead(model.config.hidden_size, mlp=args.mlp_hidden, dropout=args.mlp_drop)
    head.to(device=primary_param.device, dtype=primary_param.dtype)
    head.train()

    train_ds = PointwiseDataset(args.train)
    dev_ds = PointwiseDataset(args.dev)

    # Pos weight
    pos_weight_t: Optional[torch.Tensor] = None
    if args.task in ("cls", "cls+reg") and args.pos_weight != "none":
        if args.pos_weight == "auto":
            pos_cnt = sum(1 for s in train_ds.samples if (s.label is not None and s.label >= 0.5))
            neg_cnt = sum(1 for s in train_ds.samples if (s.label is not None and s.label < 0.5))
            if pos_cnt > 0:
                w = math.sqrt((neg_cnt + 1) / (pos_cnt))
                pos_weight_t = torch.tensor(w, device=device, dtype=next(model.parameters()).dtype)
        else:
            try:
                w = float(args.pos_weight)
                pos_weight_t = torch.tensor(w, device=device, dtype=next(model.parameters()).dtype)
            except Exception:
                pos_weight_t = None

    # Optim
    # Separate head vs base (optional same lr)
    params = [
        {"params": [p for p in model.parameters() if p.requires_grad], "lr": args.lr, "weight_decay": args.wd},
        {"params": head.parameters(), "lr": args.lr, "weight_decay": 0.0},
    ]
    optim = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.wd)

    total_steps = (len(train_ds) // max(1, args.batch) // max(1, args.grad_accum) + 1) * args.epochs
    warmup_steps = int(total_steps * args.warmup)
    global_step = 0

    def lr_schedule(step: int):
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        # cosine decay
        pct = (step - warmup_steps) / max(1, (total_steps - warmup_steps))
        pct = min(1.0, max(0.0, pct))
        return 0.5 * (1 + math.cos(math.pi * pct))

    scaler = torch.cuda.amp.GradScaler(enabled=False)  # not using fp16 scaler, bf16 safe

    best_auc = -1.0
    patience = 0

    def iterate_batches(ds: PointwiseDataset, shuffle: bool = True):
        idxs = list(range(len(ds)))
        if shuffle:
            np.random.shuffle(idxs)
        for i in range(0, len(idxs), args.batch):
            batch_idx = idxs[i:i+args.batch]
            yield [ds[s] for s in batch_idx]

    def forward_batch(samples: List[Sample]):
        texts = [build_prompt(s.query, s.candidate) for s in samples]
        enc = tok(texts, padding=True, truncation=True, max_length=args.max_len, return_tensors="pt")
        for k in enc:
            enc[k] = enc[k].to(device)
        out = model(**enc, output_hidden_states=True, use_cache=False)
        last = out.hidden_states[-1]  # [B,T,H]
        # 若隐藏状态所在设备与 head 不一致，动态迁移 head
        if next(head.parameters()).device != last.device:
            head.to(last.device)
        score_id = tok.convert_tokens_to_ids("[SCORE]")
        pos = (enc.input_ids == score_id).long().argmax(dim=1)  # assume first occurrence
        h_cls = last[torch.arange(last.size(0)), pos, :]
        logits = head(h_cls)
        return logits, samples

    def evaluate() -> Dict[str, Any]:
        model.eval(); head.eval()
        all_logits: List[float] = []
        all_labels: List[float] = []
        all_scores: List[float] = []
        by_group_logits: Dict[str, List[float]] = defaultdict(list)
        by_group_target: Dict[str, List[float]] = defaultdict(list)
        with torch.no_grad():
            for batch in iterate_batches(dev_ds, shuffle=False):
                logits, smp = forward_batch(batch)
                probs = torch.sigmoid(logits).detach().float().cpu().tolist()
                for l, s, g, prob in zip([x.label for x in smp], [x.score for x in smp], [x.group for x in smp], probs):
                    if l is not None:
                        all_labels.append(float(l))
                    if s is not None:
                        all_scores.append(float(s))
                    all_logits.append(prob)
                # group metrics
                for li, samp in enumerate(smp):
                    by_group_logits[samp.group].append(probs[li])
                    if samp.score is not None:
                        by_group_target[samp.group].append(float(samp.score))
        metrics: Dict[str, Any] = {}
        # AUC
        if all_labels:
            try:
                from sklearn.metrics import roc_auc_score  # type: ignore
                metrics["auc"] = float(roc_auc_score(all_labels, all_logits))
            except Exception:
                metrics["auc"] = None
        # Regression stats
        if all_scores:
            a = np.asarray(all_logits, dtype=np.float64)
            b = np.asarray(all_scores, dtype=np.float64)
            metrics["mse"] = float(np.mean((a - b) ** 2))
            metrics["mae"] = float(np.mean(np.abs(a - b)))
            if not (np.all(a == a[0]) or np.all(b == b[0])):
                metrics["pearson"] = float(np.corrcoef(a, b)[0, 1])
        # Ranking metrics (NDCG@10/32 + Hit@10)
        nd10 = []; nd32 = []; hit10 = []
        for gid, preds in by_group_logits.items():
            t = by_group_target.get(gid)
            if not t or len(t) != len(preds):
                continue
            rel = np.asarray(t, dtype=np.float64)
            # normalize rel
            srel = rel.sum()
            if srel > 0:
                rel = rel / srel
            p = preds
            nd10.append(ndcg_at_k(rel.tolist(), p, 10))
            nd32.append(ndcg_at_k(rel.tolist(), p, 32))
            # hit10: teacher top1 是否出现在 pred top10
            top_t = int(np.argmax(rel))
            order = np.argsort(-np.asarray(p))[:10]
            hit10.append(1.0 if top_t in order else 0.0)
        if nd10:
            metrics["ndcg@10"] = float(np.mean(nd10))
        if nd32:
            metrics["ndcg@32"] = float(np.mean(nd32))
        if hit10:
            metrics["hit@10"] = float(np.mean(hit10))
        model.train(); head.train()
        return metrics

    # -------- training loop --------
    accum = 0
    for epoch in range(1, args.epochs + 1):
        for batch in iterate_batches(train_ds, shuffle=True):
            logits, smp = forward_batch(batch)
            loss_total = torch.zeros((), device=device, dtype=logits.dtype)
            # classification
            if args.task in ("cls", "cls+reg"):
                labels = []
                for s in smp:
                    if s.label is None:
                        continue
                    labels.append(float(1.0 if s.label >= 0.5 else 0.0))
                if labels:  # filter aligned positions
                    # mask logits to those entries
                    mask_idx = [i for i, s in enumerate(smp) if s.label is not None]
                    log_sel = logits[mask_idx]
                    y = torch.tensor(labels, device=log_sel.device, dtype=log_sel.dtype)
                    loss_cls = bce_smooth_logits(log_sel, y, args.label_smoothing, pos_weight_t)
                    loss_total = loss_total + loss_cls
            # regression
            if args.task in ("reg", "cls+reg"):
                scores = []
                mask_idx2 = []
                for i, s in enumerate(smp):
                    if s.score is not None:
                        scores.append(float(s.score))
                        mask_idx2.append(i)
                if scores:
                    log_sel = logits[mask_idx2]
                    if args.reg_prob:
                        pred = torch.sigmoid(log_sel)
                    else:
                        pred = log_sel
                    tgt = torch.tensor(scores, device=pred.device, dtype=pred.dtype)
                    loss_reg = torch.nn.functional.mse_loss(pred, tgt)
                    loss_total = loss_total + loss_reg
            # group KL (optional)
            if args.kl_weight > 0:
                # organize by group inside this mini-batch
                grouped: Dict[str, List[int]] = defaultdict(list)
                for i, s in enumerate(smp):
                    grouped[s.group].append(i)
                kl_losses = []
                for gid, idxs in grouped.items():
                    if len(idxs) < 2:
                        continue
                    tgt_scores = [smp[i].score for i in idxs]
                    if any(ts is None for ts in tgt_scores):
                        continue
                    tgt = torch.tensor([float(x) for x in tgt_scores], device=logits.device, dtype=logits.dtype)
                    st = tgt.sum()
                    if st <= 0:
                        continue
                    tgt = tgt / st
                    raw = logits[idxs]
                    if args.zscore:
                        mu = raw.mean(); sd = raw.std()
                        if sd > 1e-6:
                            raw = (raw - mu) / sd
                        else:
                            raw = raw - mu
                    dist = torch.softmax(raw / max(1e-6, args.kl_temp), dim=-1)
                    kl_losses.append(kl_div(tgt.unsqueeze(0), dist.unsqueeze(0)))
                if kl_losses:
                    loss_kl = torch.stack(kl_losses).mean() * args.kl_weight
                    loss_total = loss_total + loss_kl

            loss_total = loss_total / max(1, args.grad_accum)
            loss_total.backward()
            accum += 1
            if accum >= args.grad_accum:
                # schedule
                lr_scale = lr_schedule(global_step)
                for g in optim.param_groups:
                    g["lr"] = args.lr * lr_scale
                if args.max_grad_norm and args.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    torch.nn.utils.clip_grad_norm_(head.parameters(), args.max_grad_norm)
                optim.step(); optim.zero_grad(); accum = 0
                global_step += 1
                if args.eval_steps and global_step % args.eval_steps == 0:
                    metrics = evaluate()
                    print(f"[step {global_step}] metrics: {json.dumps(metrics, ensure_ascii=False)}")
                    # early stop on AUC (if classification) else on ndcg@10
                    key_metric = None
                    if "auc" in metrics and metrics["auc"] is not None:
                        key_metric = metrics["auc"]
                    elif "ndcg@10" in metrics:
                        key_metric = metrics["ndcg@10"]
                    if key_metric is not None and key_metric > best_auc:
                        best_auc = key_metric
                        patience = 0
                        args.save.parent.mkdir(parents=True, exist_ok=True)
                        torch.save({
                            "config": vars(args),
                            "lora": model.state_dict(),
                            "head": head.state_dict(),
                            "tokenizer_len": len(tok),
                        }, args.save)
                        print(f"[save] improved checkpoint -> {args.save}")
                    else:
                        patience += 1
                        if patience >= args.early_patience:
                            print("[early_stop] patience reached")
                            return 0
        # epoch end eval
        metrics = evaluate()
        print(f"[epoch {epoch}] metrics: {json.dumps(metrics, ensure_ascii=False)}")
    # final save
    args.save.parent.mkdir(parents=True, exist_ok=True)
    torch.save({
        "config": vars(args),
        "lora": model.state_dict(),
        "head": head.state_dict(),
        "tokenizer_len": len(tok),
    }, args.save)
    print(f"[done] saved final checkpoint: {args.save}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
