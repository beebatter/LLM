#!/usr/bin/env python3
"""Pure Transformer ranker training (clause / bi / cross modes).

数据输入 (JSONL) 支持两种：
  1) 点式 (pointwise) 每行：
     {"query": str, "clause": str, "label": 0/1, "score": float?, "group": "qid"}
     - query 可为空 (clause 模式忽略)
  2) 组内 JSON (listwise-ish) 每行：
     {"problem_name": str, "conjecture": str, "candidates": [{"id":..,"text":..,"label":0/1,"score":..}, ...]}
     读入时会展开为多行点式。

训练目标：
  - BCE: 对 label 二分类 (>=0.5 为正)
  - (可选) MSE: 对 score 回归 (若提供)
  - (可选) 组内 softmax + KL (score 归一化后) —— 简化对比增强 (默认关闭)

运行示例 (clause 模式)：
python -m LLM.training.train_transformer_ranker \
  --mode clause \
  --train /root/autodl-tmp/Training/datasets/pointwise.train.jsonl \
  --dev   /root/autodl-tmp/Training/datasets/pointwise.dev.jsonl \
  --epochs 2 --lr 1e-4 --batch 64 --max-len 256 \
  --save /root/autodl-tmp/Training/models/ckpt_clause_tf.pt

bi 模式：
python -m LLM.training.train_transformer_ranker --mode bi --train ... --dev ... --save ckpt_bi.pt

cross 模式：
python -m LLM.training.train_transformer_ranker --mode cross --train ... --dev ... --save ckpt_ce.pt

完成后可用：
python -m LLM.pipeline.rank_pipeline --mode clause_tf --model-path ckpt_clause_tf.pt --input problems.jsonl --output out.clause.jsonl
"""
from __future__ import annotations

import argparse
import json
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Any, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    from LLM.models.logic_transformers import TransformerConfig, ClauseScorer, BiEncoder, CrossEncoder
except Exception:
    from models.logic_transformers import TransformerConfig, ClauseScorer, BiEncoder, CrossEncoder  # type: ignore


# ------------------ Simple whitespace vocab ------------------
class SimpleVocab:
    def __init__(self):
        self.id2tok = ['<pad>']
        self.tok2id = {'<pad>': 0}
    def encode(self, text: str) -> List[int]:
        toks = text.strip().split()
        ids = []
        for t in toks:
            if t not in self.tok2id:
                self.tok2id[t] = len(self.id2tok)
                self.id2tok.append(t)
            ids.append(self.tok2id[t])
        return ids
    @property
    def size(self):
        return len(self.id2tok)


def pad_batch(seqs: List[List[int]], pad_id: int = 0):
    if not seqs:
        return torch.empty(0,0,dtype=torch.long), torch.empty(0,0,dtype=torch.long)
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), m), dtype=torch.long)
    for i,s in enumerate(seqs):
        if len(s):
            ids[i,:len(s)] = torch.tensor(s,dtype=torch.long)
            mask[i,:len(s)] = 1
    return ids, mask


@dataclass
class Example:
    query: str
    clause: str
    label: Optional[int]
    score: Optional[float]
    group: str


class RankDataset(Dataset):
    def __init__(self, paths: List[str]):
        self.examples: List[Example] = []
        for p in paths:
            with open(p,'r',encoding='utf-8',errors='ignore') as f:
                for ln in f:
                    ln=ln.strip()
                    if not ln:
                        continue
                    try:
                        j=json.loads(ln)
                    except Exception:
                        continue
                    if 'candidates' in j:  # group style
                        q = j.get('conjecture') or ''
                        gid = j.get('problem_name') or q[:32]
                        for c in j.get('candidates',[]):
                            text = c.get('text') or ''
                            if not text:
                                continue
                            lbl = c.get('label')
                            sc  = c.get('score')
                            try:
                                if lbl is not None:
                                    lbl = 1 if float(lbl) >= 0.5 else 0
                            except Exception:
                                lbl=None
                            try:
                                if sc is not None:
                                    sc=float(sc)
                            except Exception:
                                sc=None
                            self.examples.append(Example(q,text,lbl,sc,gid))
                    else:  # pointwise line
                        q = j.get('query') or ''
                        c = j.get('clause') or j.get('candidate') or j.get('text') or ''
                        if not c:
                            continue
                        gid = j.get('group') or j.get('problem_name') or q[:32]
                        lbl = j.get('label')
                        sc  = j.get('score')
                        try:
                            if lbl is not None:
                                lbl = 1 if float(lbl) >= 0.5 else 0
                        except Exception:
                            lbl=None
                        try:
                            if sc is not None:
                                sc=float(sc)
                        except Exception:
                            sc=None
                        self.examples.append(Example(q,c,lbl,sc,gid))
        self.groups = {}
        for i,e in enumerate(self.examples):
            self.groups.setdefault(e.group, []).append(i)

    def __len__(self):
        return len(self.examples)
    def __getitem__(self, idx):
        return self.examples[idx]


def build_argparser():
    ap = argparse.ArgumentParser(description='Train pure Transformer ranker (clause/bi/cross)')
    ap.add_argument('--mode', choices=['clause','bi','cross'], required=True)
    ap.add_argument('--train', action='append', required=True)
    ap.add_argument('--dev', action='append', required=True)
    ap.add_argument('--epochs', type=int, default=2)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--lr', type=float, default=1e-4)
    ap.add_argument('--wd', type=float, default=0.01)
    ap.add_argument('--warmup', type=float, default=0.05)
    ap.add_argument('--max-len', type=int, default=256)
    ap.add_argument('--d-model', type=int, default=256)
    ap.add_argument('--layers', type=int, default=4)
    ap.add_argument('--heads', type=int, default=4)
    ap.add_argument('--d-ff', type=int, default=512)
    ap.add_argument('--dropout', type=float, default=0.1)
    ap.add_argument('--label-smoothing', type=float, default=0.0)
    ap.add_argument('--mse-weight', type=float, default=0.0)
    ap.add_argument('--kl-weight', type=float, default=0.0, help='组内 softmax KL (score 分布)')
    ap.add_argument('--kl-temp', type=float, default=1.0)
    ap.add_argument('--save', required=True)
    ap.add_argument('--eval-steps', type=int, default=500)
    ap.add_argument('--device', default='cuda')
    return ap


def bce_loss_with_smoothing(logits: torch.Tensor, labels: torch.Tensor, eps: float):
    if eps>0:
        labels = labels*(1-eps) + 0.5*eps
    return nn.functional.binary_cross_entropy_with_logits(logits, labels)


def kl_div(p_star: torch.Tensor, p_hat: torch.Tensor, eps: float=1e-8):
    p_star = p_star.clamp_min(eps)
    p_hat  = p_hat.clamp_min(eps)
    return torch.sum(p_star * (p_star.log()-p_hat.log()), dim=-1).mean()


def main(argv=None):
    args = build_argparser().parse_args(argv)
    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    train_ds = RankDataset(args.train)
    dev_ds   = RankDataset(args.dev)

    vocab = SimpleVocab()

    # build model config (vocab size grows online, finalize after first pass of a small sample)
    # We will encode on the fly; since vocab grows, we finalize before model build by scanning train set once.
    for e in train_ds.examples[:10000]:  # partial warm vocab
        vocab.encode(e.clause)
        if args.mode in ('bi','cross'):
            vocab.encode(e.query)
    cfg = TransformerConfig(
        vocab_size=vocab.size,
        d_model=args.d_model,
        n_heads=args.heads,
        n_layers=args.layers,
        d_ff=args.d_ff,
        dropout=args.dropout,
        max_len=args.max_len,
        pad_id=0,
    )
    if args.mode == 'clause':
        model = ClauseScorer(cfg)
    elif args.mode == 'bi':
        model = BiEncoder(cfg)
    else:
        model = CrossEncoder(cfg)
    model.to(device)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    total_steps = math.ceil(len(train_ds)/args.batch)*args.epochs
    warmup = int(total_steps*args.warmup)

    def lr_scheduler(step):
        if step < warmup:
            return step/max(1,warmup)
        pct = (step-warmup)/max(1,(total_steps-warmup))
        pct = min(1.0,max(0.0,pct))
        return 0.5*(1+math.cos(math.pi*pct))

    global_step=0

    def batch_iter(ds: RankDataset, shuffle=True):
        idxs = list(range(len(ds)))
        if shuffle:
            np.random.shuffle(idxs)
        for i in range(0,len(idxs), args.batch):
            yield [ds.examples[j] for j in idxs[i:i+args.batch]]

    def forward_clause(ex: List[Example]):
        seqs = [vocab.encode(x.clause)[:args.max_len] for x in ex]
        ids, mask = pad_batch(seqs,0)
        ids, mask = ids.to(device), mask.to(device)
        logits = model(ids, mask)  # [B]
        return logits

    def forward_bi(ex: List[Example]):
        q_seqs = [vocab.encode(x.query)[:args.max_len] for x in ex]
        d_seqs = [vocab.encode(x.clause)[:args.max_len] for x in ex]
        q_ids,q_mask = pad_batch(q_seqs,0); d_ids,d_mask = pad_batch(d_seqs,0)
        q_ids,q_mask = q_ids.to(device), q_mask.to(device)
        d_ids,d_mask = d_ids.to(device), d_mask.to(device)
        logits = model.score(q_ids,q_mask,d_ids,d_mask)  # [B]
        return logits

    def forward_cross(ex: List[Example]):
        seqs = [vocab.encode((x.query or '') + ' [SEP] ' + x.clause)[:args.max_len] for x in ex]
        ids,mask = pad_batch(seqs,0)
        ids,mask = ids.to(device), mask.to(device)
        logits = model(ids,mask)  # [B]
        return logits

    def run_eval():
        model.eval()
        preds=[]; labels=[]; scores=[]
        with torch.no_grad():
            for b in batch_iter(dev_ds, shuffle=False):
                if args.mode=='clause': logit=forward_clause(b)
                elif args.mode=='bi': logit=forward_bi(b)
                else: logit=forward_cross(b)
                probs=torch.sigmoid(logit).cpu().tolist()
                for ex,p in zip(b,probs):
                    preds.append(p)
                    if ex.label is not None:
                        labels.append(ex.label)
                    if ex.score is not None:
                        scores.append((p,ex.score))
        metrics={}
        if labels:
            try:
                from sklearn.metrics import roc_auc_score
                metrics['auc']=float(roc_auc_score(labels,preds))
            except Exception:
                metrics['auc']=None
        if scores:
            arr=np.array(scores,dtype=float)
            metrics['mse']=float(np.mean((arr[:,0]-arr[:,1])**2))
        model.train()
        return metrics

    best=-1e9
    for epoch in range(1, args.epochs+1):
        for batch in batch_iter(train_ds, shuffle=True):
            if args.mode=='clause': logits=forward_clause(batch)
            elif args.mode=='bi': logits=forward_bi(batch)
            else: logits=forward_cross(batch)
            loss = torch.zeros((), device=device, dtype=logits.dtype)
            # BCE part
            lbl_mask=[i for i,e in enumerate(batch) if e.label is not None]
            if lbl_mask:
                y=torch.tensor([batch[i].label for i in lbl_mask], dtype=logits.dtype, device=device)
                loss += bce_loss_with_smoothing(logits[lbl_mask], y, args.label_smoothing)
            # MSE part
            if args.mse_weight>0:
                sc_mask=[i for i,e in enumerate(batch) if e.score is not None]
                if sc_mask:
                    y=torch.tensor([float(batch[i].score) for i in sc_mask],dtype=logits.dtype,device=device)
                    pred=torch.sigmoid(logits[sc_mask])
                    loss += args.mse_weight*nn.functional.mse_loss(pred,y)
            # group KL (simple per mini-batch group wise)
            if args.kl_weight>0:
                from collections import defaultdict
                grp=defaultdict(list)
                for i,e in enumerate(batch):
                    if e.score is not None:
                        grp[e.group].append(i)
                kl_list=[]
                for g,idxs in grp.items():
                    if len(idxs)<2: continue
                    tgt=torch.tensor([float(batch[i].score) for i in idxs], dtype=logits.dtype, device=device)
                    s=tgt.sum()
                    if s<=0: continue
                    tgt=tgt/s
                    dist=torch.softmax(logits[idxs]/max(1e-6,args.kl_temp), dim=-1)
                    kl_list.append(kl_div(tgt.unsqueeze(0), dist.unsqueeze(0)))
                if kl_list:
                    loss+=args.kl_weight*torch.stack(kl_list).mean()
            loss.backward()
            global_step+=1
            # schedule
            lr_scale=lr_scheduler(global_step)
            for g in opt.param_groups:
                g['lr']=args.lr*lr_scale
            opt.step(); opt.zero_grad()
            if args.eval_steps and global_step % args.eval_steps==0:
                metrics=run_eval()
                key=metrics.get('auc') if 'auc' in metrics else None
                print(f"[step {global_step}] metrics={metrics}")
                if key is not None and key>best:
                    best=key
                    torch.save({'model':model.state_dict(),'vocab_size':vocab.size,'config':vars(args)}, args.save)
                    print(f"[save] {args.save}")
        metrics=run_eval()
        print(f"[epoch {epoch}] metrics={metrics}")
    torch.save({'model':model.state_dict(),'vocab_size':vocab.size,'config':vars(args)}, args.save)
    print(f"[done] saved final {args.save}")
    return 0


if __name__=='__main__':
    raise SystemExit(main())
