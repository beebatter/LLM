#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List, Dict, Optional

import torch
import torch.nn as nn
from torch.optim import AdamW
from torch.utils.data import Dataset, DataLoader
from tqdm.auto import tqdm

from LLM.models.logic_transformers import BiEncoder, TransformerConfig
from LLM.data_utils.logic_tokenizer import LogicSentencePiece, normalize_text, features_to_prefix, PrefixBuckets


class DistillItem:
    def __init__(self, q: str, d: str, score: float, features: Optional[Dict], group: str):
        self.q = q
        self.d = d
        self.score = float(score)
        self.features = features
        self.group = group


class DistillDataset(Dataset):
    """JSONL schema: supports fields
    - query keys: conjecture_text|conjecture_sig|query|q|conjecture|text_a
    - doc keys: text|doc|d|clause|premise|text_b
    - score: ce_score|score|label (float)
    - group: problem_name|problem|query_id|qid|qid_str (else md5 of normalized query)
    - features: features|meta (optional)
    """

    def __init__(self, jsonl_paths: List[str]):
        import json, hashlib

        def get_first(dct: Dict, keys: List[str]):
            for k in keys:
                v = dct.get(k)
                if v is not None:
                    return v
            return None

        def md5(s: str) -> str:
            return hashlib.md5(s.encode('utf-8')).hexdigest()

        self.items: List[DistillItem] = []
        for p in jsonl_paths:
            with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    q = get_first(j, ["conjecture_text", "conjecture_sig", "query", "q", "conjecture", "text_a"]) or ""
                    d = get_first(j, ["text", "doc", "d", "clause", "premise", "text_b"]) or ""
                    if not q or not d:
                        continue
                    features = get_first(j, ["features", "meta"]) or None
                    g = get_first(j, ["problem_name", "problem", "query_id", "qid", "qid_str"]) or md5(normalize_text(q))
                    s = get_first(j, ["ce_score", "score", "label"])  # prefer explicit ce_score or score
                    try:
                        score = float(s)
                    except Exception:
                        score = 0.0
                    self.items.append(DistillItem(q, d, score, features, str(g)))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, idx: int) -> DistillItem:
        return self.items[idx]


class DistillCollate:
    def __init__(self, spm_model: str, max_len: int = 256, pad_id: int = 0):
        self.tok = LogicSentencePiece(spm_model)
        self.max_len = max_len
        self.pad_id = pad_id

    def __call__(self, batch: List[DistillItem]) -> Dict[str, torch.Tensor]:
        def wrap_q(text: str) -> str:
            return f"<Q> {normalize_text(text)} </Q>"

        def wrap_d(text: str, features: Optional[Dict]) -> str:
            prefix = features_to_prefix(features or {}, PrefixBuckets())
            return f"{prefix}<D> {normalize_text(text)} </D>"

        q_ids_list, d_ids_list, q_masks, d_masks = [], [], [], []
        scores: List[float] = []
        # group string to int mapping within batch (optional use)
        gid_map: Dict[str, int] = {}
        next_gid = 0
        g_ids: List[int] = []
        for it in batch:
            qi = self.tok.encode(wrap_q(it.q))[: self.max_len]
            di = self.tok.encode(wrap_d(it.d, it.features))[: self.max_len]
            q_ids_list.append(qi)
            d_ids_list.append(di)
            q_masks.append([1] * len(qi))
            d_masks.append([1] * len(di))
            scores.append(float(it.score))
            if it.group not in gid_map:
                gid_map[it.group] = next_gid
                next_gid += 1
            g_ids.append(gid_map[it.group])

        def pad(arrs: List[List[int]], pad_id: int) -> torch.Tensor:
            maxl = max(len(x) for x in arrs)
            return torch.tensor([x + [pad_id] * (maxl - len(x)) for x in arrs], dtype=torch.long)

        def padm(arrs: List[List[int]]) -> torch.Tensor:
            maxl = max(len(x) for x in arrs)
            return torch.tensor([x + [0] * (maxl - len(x)) for x in arrs], dtype=torch.long)

        return {
            "q_ids": pad(q_ids_list, self.pad_id),
            "q_mask": padm(q_masks),
            "d_ids": pad(d_ids_list, self.pad_id),
            "d_mask": padm(d_masks),
            "scores": torch.tensor(scores, dtype=torch.float),
            "group_ids": torch.tensor(g_ids, dtype=torch.long),
        }


def main(argv: List[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Distill Cross-Encoder into Bi-Encoder (soft score regression)")
    ap.add_argument("--train", action="append", required=True)
    ap.add_argument("--val", action="append")
    ap.add_argument("--spm", type=str, default="/home/ks/Training/models/spm_logic.model")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--d-model", type=int, default=512)
    ap.add_argument("--layers", type=int, default=6)
    ap.add_argument("--heads", type=int, default=8)
    ap.add_argument("--max-len", type=int, default=2048)
    ap.add_argument("--save", type=Path, default=Path("/home/ks/Training/models/biencoder_distill.pt"))
    ap.add_argument("--logdir", type=Path, default=Path("/home/ks/Training/logs/biencoder_distill"))
    ap.add_argument("--loss", type=str, choices=["mse", "kl"], default="mse")
    ap.add_argument("--temperature", type=float, default=1.0, help="Softening temperature for KL distillation")
    args = ap.parse_args(argv)

    # model
    # vocab size: try to read from SPM model directly
    try:
        import sentencepiece as spm  # type: ignore
        spp = spm.SentencePieceProcessor(); spp.load(str(args.spm))
        vocab_size = int(spp.get_piece_size())
    except Exception:
        from pathlib import Path as _P
        spm_vocab = _P(args.spm).with_suffix('.vocab')
        with open(spm_vocab, 'r', encoding='utf-8') as f:
            vocab_size = sum(1 for _ in f)

    cfg = TransformerConfig(
        vocab_size=vocab_size, d_model=args.d_model, n_heads=args.heads, n_layers=args.layers, pad_id=0, max_len=args.max_len
    )
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = BiEncoder(cfg).to(device)
    opt = AdamW(model.parameters(), lr=args.lr)

    # data
    train_ds = DistillDataset(args.train)
    val_ds = DistillDataset(args.val or args.train)
    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True, num_workers=2, collate_fn=DistillCollate(args.spm, max_len=args.max_len))
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False, num_workers=2, collate_fn=DistillCollate(args.spm, max_len=args.max_len))

    # loss
    if args.loss == 'kl':
        # convert scores to per-group softmax and match distributions
        def group_softmax(scores: torch.Tensor, gids: torch.Tensor, T: float) -> torch.Tensor:
            # scores [B], gids [B] -> probs [B] normalized within group
            out = torch.zeros_like(scores)
            for g in gids.unique():
                m = (gids == g)
                s = scores[m] / max(1e-6, T)
                p = torch.softmax(s, dim=0)
                out[m] = p
            return out
        kl = nn.KLDivLoss(reduction='batchmean')
    else:
        mse = nn.MSELoss()

    best = float('inf')
    for ep in range(1, args.epochs + 1):
        model.train()
        total, n = 0.0, 0
        pbar = tqdm(train_loader, desc=f"epoch {ep}/{args.epochs}", dynamic_ncols=True)
        for batch in pbar:
            q_ids = batch['q_ids'].to(device)
            q_mask = batch['q_mask'].to(device)
            d_ids = batch['d_ids'].to(device)
            d_mask = batch['d_mask'].to(device)
            scores = batch['scores'].to(device)
            gids = batch['group_ids'].to(device)
            opt.zero_grad(set_to_none=True)
            q = model.encode(q_ids, q_mask, which='q')
            d = model.encode(d_ids, d_mask, which='d')
            sim = (nn.functional.normalize(q, dim=1) * nn.functional.normalize(d, dim=1)).sum(dim=1)
            if args.loss == 'kl':
                # distributions within group
                p_t = group_softmax(scores, gids, args.temperature)
                p_s = group_softmax(sim, gids, args.temperature)
                loss = kl(torch.log(p_s + 1e-8), p_t)
            else:
                loss = mse(sim, scores)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            total += float(loss.item()) * q_ids.size(0)
            n += q_ids.size(0)
            pbar.set_postfix(loss=f"{loss.item():.4f}", avg=f"{(total/max(1,n)):.4f}")

        # simple val loss
        model.eval(); vtotal, vn = 0.0, 0
        with torch.no_grad():
            for batch in val_loader:
                q_ids = batch['q_ids'].to(device)
                q_mask = batch['q_mask'].to(device)
                d_ids = batch['d_ids'].to(device)
                d_mask = batch['d_mask'].to(device)
                scores = batch['scores'].to(device)
                gids = batch['group_ids'].to(device)
                q = model.encode(q_ids, q_mask, which='q')
                d = model.encode(d_ids, d_mask, which='d')
                sim = (nn.functional.normalize(q, dim=1) * nn.functional.normalize(d, dim=1)).sum(dim=1)
                if args.loss == 'kl':
                    p_t = group_softmax(scores, gids, args.temperature)
                    p_s = group_softmax(sim, gids, args.temperature)
                    vloss = kl(torch.log(p_s + 1e-8), p_t)
                else:
                    vloss = mse(sim, scores)
                vtotal += float(vloss.item()) * q_ids.size(0)
                vn += q_ids.size(0)
        avg = vtotal / max(1, vn)
        print(f"epoch {ep}: val_loss={avg:.4f}")
        if avg < best:
            best = avg
            args.save.parent.mkdir(parents=True, exist_ok=True)
            torch.save({
                'model_state': model.state_dict(),
                'config': cfg.__dict__,
                'spm_model': args.spm,
            }, args.save)
            print(f"saved: {args.save}")

    return 0


if __name__ == '__main__':
    raise SystemExit(main())
