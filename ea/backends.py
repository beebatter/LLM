#!/usr/bin/env python3
"""Unified scoring backends for interactive EA / iProver pipeline.

Expose a simple interface:

    backend = load_backend(
        mode="clause_tf|cross_tf|bi_tf|bi_then_llm|llm_direct|llm_pmi|fusion",
        **kwargs
    )
    scores = backend.score(conjecture_text: str, candidates: List[Dict])

Where each candidate dict minimally contains {'id': <any>, 'text': <str>}.

Design principles:
  - Stateless (except cached model weights / tokenizer)
  - Return raw scores (higher = better). Caller handles calibration/softmax.
  - Optional shortlist (bi_then_llm) to reduce LLM passes.
  - Fusion takes pre-computed score mappings.

Lightweight; real production should persist vocabulary + configs.
"""
from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple, Optional

import torch

try:
    from LLM.models.logic_transformers import TransformerConfig, ClauseScorer, CrossEncoder, BiEncoder
except Exception:  # pragma: no cover
    from models.logic_transformers import TransformerConfig, ClauseScorer, CrossEncoder, BiEncoder  # type: ignore


# ----------------------- Simple ad-hoc tokenizer -----------------------
class SimpleVocab:
    """Either dynamic whitespace vocab or load a sentencepiece vocab file (.vocab).

    If vocab_path provided and looks like sentencepiece (.vocab), we parse first column tokens.
    IDs follow file order; <pad> inserted as id0 if absent.
    """
    def __init__(self, vocab_path: Optional[str] = None, dynamic: bool = True, pad_id: int = 0):
        self.id2tok: List[str] = []
        self.tok2id: Dict[str, int] = {}
        self.dynamic = dynamic and (vocab_path is None)
        self.pad_id = pad_id
        self.unk_id = 0  # will be set below
        if vocab_path and os.path.isfile(vocab_path):
            # sentencepiece .vocab lines: token<tab>score
            with open(vocab_path, 'r', encoding='utf-8', errors='ignore') as f:
                for ln in f:
                    ln = ln.strip()
                    if not ln:
                        continue
                    if '\t' in ln:
                        tok, _score = ln.split('\t', 1)
                    else:
                        tok = ln.split()[0]
                    if tok in self.tok2id:
                        continue
                    self.tok2id[tok] = len(self.id2tok)
                    self.id2tok.append(tok)
            # Ensure <pad> exists at index 0 (model default); if absent, insert and shift
            if '<pad>' not in self.tok2id:
                self.id2tok = ['<pad>'] + self.id2tok
                self.tok2id = {tok: i+1 for tok, i in self.tok2id.items()}
                self.tok2id['<pad>'] = 0
            # Ensure <unk> exists and is NOT pad
            if '<unk>' not in self.tok2id:
                self.tok2id['<unk>'] = len(self.id2tok)
                self.id2tok.append('<unk>')
            self.unk_id = self.tok2id.get('<unk>', 1)
        else:
            # dynamic building mode
            self.id2tok = ['<pad>']
            self.tok2id = {'<pad>': 0}
            # reserve <unk>
            self.tok2id['<unk>'] = 1
            self.id2tok.append('<unk>')
            self.unk_id = 1

    def encode(self, text: str) -> List[int]:
        if not text:
            return []
        if self.dynamic:
            toks = text.strip().split()
            out: List[int] = []
            for t in toks:
                if t not in self.tok2id:
                    self.tok2id[t] = len(self.id2tok)
                    self.id2tok.append(t)
                out.append(self.tok2id[t])
            return out
        # static vocab mode: apply a lightweight regex tokenizer to avoid collapsing to single <unk>
        import re
        # tokens: identifiers / numbers / separate punctuation important in logic
        toks = re.findall(r"[A-Za-z0-9_]+|[()=,~|&!]|:\$i|:\$o", text)
        if not toks:
            toks = [text]
        return [self.tok2id.get(t, self.unk_id) for t in toks]

    @property
    def size(self) -> int:  # type: ignore
        return len(self.id2tok)


def _pad_batch(seqs: List[List[int]], pad_id: int = 0):
    if not seqs:
        return torch.empty(0, 0, dtype=torch.long), torch.empty(0, 0, dtype=torch.long)
    m = max(len(s) for s in seqs)
    ids = torch.full((len(seqs), m), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), m), dtype=torch.long)
    for i, s in enumerate(seqs):
        if not s:
            continue
        ids[i, :len(s)] = torch.tensor(s, dtype=torch.long)
        mask[i, :len(s)] = 1
    return ids, mask


def _sanitize_ids_tensor(ids: torch.Tensor, model_vocab_size: int, unk_id: int, *, verbose: bool = False, who: str = "") -> torch.Tensor:
    """Replace any indices <0 or >= model_vocab_size with unk_id to avoid embedding OOB.

    Returns a (potentially) cloned tensor with safe indices.
    """
    if ids.numel() == 0:
        return ids
    hi = ids >= model_vocab_size
    lo = ids < 0
    if hi.any() or lo.any():
        if verbose:
            try:
                total = int(hi.sum().item() + lo.sum().item())
                max_id = int(ids.max().item()) if ids.numel() else -1
                min_id = int(ids.min().item()) if ids.numel() else -1
                print(f"[EA DBG] sanitize({who}): replacing {total} OOR ids (min={min_id}, max={max_id}) with unk_id={unk_id}, model_vocab_size={model_vocab_size}", flush=True)
            except Exception:
                pass
        ids = ids.clone()
        ids[hi | lo] = int(unk_id)
    return ids


# ----------------------------- Base -----------------------------------
class BaseBackend:
    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:  # pragma: no cover - interface
        raise NotImplementedError


# ---------------------- Transformer Backends --------------------------
def _extract_state(ckpt: Any):  # helper to unify checkpoint formats
    if isinstance(ckpt, dict):
        for key in ('model', 'state_dict', 'model_state'):
            if key in ckpt and isinstance(ckpt[key], (dict,)):
                return ckpt[key]
    return ckpt


class ClauseTFBackend(BaseBackend):
    def __init__(self, ckpt_path: str, device: str = 'cuda', vocab_path: Optional[str] = None):
        self.device = device
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = _extract_state(ckpt)
        cfg_dict = {}
        for k in ['vocab_size', 'd_model', 'n_heads', 'n_layers', 'd_ff', 'dropout', 'max_len', 'pad_id']:
            if k in ckpt.get('config', {}):
                cfg_dict[k] = ckpt['config'][k]
        if 'vocab_size' not in cfg_dict:
            cfg_dict['vocab_size'] = ckpt.get('vocab_size', 32000)
        cfg = TransformerConfig(**cfg_dict)
        self.model = ClauseScorer(cfg).to(device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.vocab = SimpleVocab(vocab_path=vocab_path, pad_id=getattr(cfg, 'pad_id', 0))
    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        seqs = [self.vocab.encode(c['text']) for c in candidates]
        ids, mask = _pad_batch(seqs)
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            s = self.model(ids, mask).float().cpu().tolist()
        return [float(x) for x in s]


class CrossTFBackend(BaseBackend):
    def __init__(self, ckpt_path: str, device: str = 'cuda', vocab_path: Optional[str] = None,
                 override_max_len: Optional[int] = None, verbose: bool = True):
        self.device = device
        self.verbose = verbose
        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = _extract_state(ckpt)
        cfg_dict = {}
        for k in ['vocab_size', 'd_model', 'n_heads', 'n_layers', 'd_ff', 'dropout', 'max_len', 'pad_id']:
            if isinstance(ckpt, dict) and 'config' in ckpt and k in ckpt['config']:
                cfg_dict[k] = ckpt['config'][k]
        if 'vocab_size' not in cfg_dict:
            cfg_dict['vocab_size'] = ckpt.get('vocab_size', 32000)
        cfg = TransformerConfig(**cfg_dict)
        self.model = CrossEncoder(cfg).to(device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.vocab = SimpleVocab(vocab_path=vocab_path, pad_id=getattr(cfg, 'pad_id', 0))
        # Diagnostics: embedding vs vocab sizes
        try:
            emb = self.model.encoder.embed
            model_vs = int(getattr(emb, 'num_embeddings', cfg.vocab_size))
            if self.verbose:
                print(
                    f"[EA] CrossTF embed/vocab: model_emb={model_vs} ckpt_vocab={cfg.vocab_size} file_vocab={self.vocab.size} unk_id={self.vocab.unk_id} pad_id={getattr(cfg, 'pad_id', 0)}",
                    flush=True,
                )
        except Exception:
            model_vs = cfg.vocab_size
        # Determine effective max length: honor override if provided; else model cfg; fallback to 256
        try:
            model_max = getattr(self.model.encoder.cfg, 'max_len', None)
        except Exception:
            model_max = None
        if override_max_len is not None and model_max is not None:
            self.max_len = int(min(override_max_len, model_max))
        elif override_max_len is not None:
            self.max_len = int(override_max_len)
        else:
            self.max_len = int(model_max or 256)
        if self.verbose:
            try:
                print(f"[EA] CrossTFBackend loaded: eff_max_len={self.max_len} (model_max={model_max})", flush=True)
            except Exception:
                pass
    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        q = conjecture or ''
        max_len = self.max_len
        
        # Advanced truncation strategy
        seqs = []
        truncation_stats = {'full_fit': 0, 'truncated': 0, 'extreme_truncated': 0}
        
        for c in candidates:
            # Strategy 1: Try full sequence first
            full_text = q + ' [SEP] ' + c['text']
            full_seq = self.vocab.encode(full_text)
            
            if len(full_seq) <= max_len:
                seqs.append(full_seq)
                truncation_stats['full_fit'] += 1
            else:
                # Strategy 2: Intelligent truncation preserving structure
                q_seq = self.vocab.encode(q)
                sep_seq = self.vocab.encode(' [SEP] ')
                c_seq = self.vocab.encode(c['text'])
                
                overhead = len(sep_seq)
                available = max_len - overhead
                
                if available <= 2:  # Extreme case: not enough space even for sep
                    seqs.append(full_seq[:max_len])
                    truncation_stats['extreme_truncated'] += 1
                    continue
                
                # Strategy 3: Preserve key parts with sliding window fallback
                truncation_stats['truncated'] += 1
                
                # For very long sequences, use sliding window to capture different parts
                total_original = len(q_seq) + len(c_seq)
                
                if total_original > max_len * 3:  # Very long, use sliding window
                    # Take beginning and end of each part
                    q_start = max(1, len(q_seq) // 4)
                    q_end = max(1, len(q_seq) // 4) 
                    c_start = max(1, len(c_seq) // 4)
                    c_end = max(1, len(c_seq) // 4)
                    
                    q_truncated = q_seq[:q_start] + q_seq[-q_end:] if len(q_seq) > q_start + q_end else q_seq
                    c_truncated = c_seq[:c_start] + c_seq[-c_end:] if len(c_seq) > c_start + c_end else c_seq
                    
                    # Adjust if still too long
                    combined_len = len(q_truncated) + len(c_truncated)
                    if combined_len > available:
                        # Proportional reduction
                        q_ratio = len(q_truncated) / combined_len if combined_len > 0 else 0.5
                        q_allocation = int(available * q_ratio)
                        c_allocation = available - q_allocation
                        q_truncated = q_truncated[:q_allocation]
                        c_truncated = c_truncated[:c_allocation]
                        
                else:
                    # Standard proportional allocation for moderate lengths
                    q_ratio = len(q_seq) / total_original if total_original > 0 else 0.5
                    q_allocation = max(1, int(available * q_ratio))
                    q_allocation = min(q_allocation, len(q_seq), available - 1)
                    c_allocation = available - q_allocation
                    
                    q_truncated = q_seq[:q_allocation]
                    c_truncated = c_seq[:c_allocation]
                
                # Rebuild and add
                result_seq = q_truncated + sep_seq + c_truncated
                seqs.append(result_seq[:max_len])  # Final safety
        
        # Log truncation statistics occasionally
        if truncation_stats['truncated'] > 0 and len(candidates) > 100:
            try:
                print(f"[EA DBG] Truncation stats: {truncation_stats} out of {len(candidates)} candidates", flush=True)
            except:
                pass
        
        ids, mask = _pad_batch(seqs)
        # Sanitize ids to avoid OOB in embedding
        try:
            emb = self.model.encoder.embed
            model_vs = int(getattr(emb, 'num_embeddings', self.vocab.size))
        except Exception:
            model_vs = self.vocab.size
        eff_unk = int(self.vocab.unk_id)
        if eff_unk >= model_vs:
            # Fallback unk id within embedding range
            if self.verbose:
                try:
                    print(f"[EA WARN] CrossTF unk_id {eff_unk} >= model_vocab_size {model_vs}; using fallback unk_id {model_vs-1}", flush=True)
                except Exception:
                    pass
            eff_unk = model_vs - 1
        ids = _sanitize_ids_tensor(ids, model_vs, eff_unk, verbose=self.verbose, who="cross")
        # Move to device after sanitization
        ids, mask = ids.to(self.device), mask.to(self.device)
        
        try:
            print(f"[EA DBG] CrossTF input shapes after truncation: ids={tuple(ids.shape)}, mask={tuple(mask.shape)}", flush=True)
        except Exception:
            pass
            
        with torch.no_grad():
            try:
                s = self.model(ids, mask).float().cpu().tolist()
            except Exception as e:
                try:
                    print(f"[EA DBG] CrossEncoder forward failed: {str(e)}", flush=True)
                    # Try to get intermediate shapes for debugging
                    h = self.model.encoder(ids, mask)  # [B, L, D]
                    print(f"[EA DBG] encoder output shape: {tuple(h.shape)}", flush=True)
                    z = self.model.pool(h, mask)  # [B, D]
                    print(f"[EA DBG] pool output shape: {tuple(z.shape)}", flush=True)
                except Exception as e2:
                    print(f"[EA DBG] Debug failed: {str(e2)}", flush=True)
                raise e
                
        return [float(x) for x in s]


class BiTFBackend(BaseBackend):
    def __init__(self, ckpt_path: str, device: str = 'cuda', vocab_path: Optional[str] = None,
                 chunk_encode: bool = False, chunk_len: int = 256, chunk_stride: int = 192,
                 chunk_max: int = 4, chunk_agg: str = 'mean', override_max_len: Optional[int] = None, verbose: bool = False):
        self.device = device
        self.chunk_encode = chunk_encode
        self.chunk_len = chunk_len
        self.chunk_stride = chunk_stride
        self.chunk_max = chunk_max
        self.chunk_agg = chunk_agg
        self.override_max_len = override_max_len
        self.verbose = verbose

        ckpt = torch.load(ckpt_path, map_location='cpu')
        state = _extract_state(ckpt)
        cfg_dict: Dict[str, Any] = {}
        for k in ['vocab_size', 'd_model', 'n_heads', 'n_layers', 'd_ff', 'dropout', 'max_len', 'pad_id']:
            if isinstance(ckpt, dict) and 'config' in ckpt and k in ckpt['config']:
                cfg_dict[k] = ckpt['config'][k]
        if 'vocab_size' not in cfg_dict:
            cfg_dict['vocab_size'] = ckpt.get('vocab_size', 32000)

        cfg = TransformerConfig(**cfg_dict)
        self.model = BiEncoder(cfg).to(device)
        self.model.load_state_dict(state, strict=False)
        self.model.eval()
        self.vocab = SimpleVocab(vocab_path=vocab_path, pad_id=getattr(cfg, 'pad_id', 0))
        # Diagnostics: embedding vs vocab sizes for both encoders
        try:
            q_vs = int(getattr(self.model.q_enc.embed, 'num_embeddings', cfg.vocab_size))
        except Exception:
            q_vs = cfg.vocab_size
        try:
            d_vs = int(getattr(self.model.d_enc.embed, 'num_embeddings', cfg.vocab_size))
        except Exception:
            d_vs = cfg.vocab_size
        if self.verbose:
            try:
                print(
                    f"[EA] BiTF embed/vocab: q_emb={q_vs} d_emb={d_vs} ckpt_vocab={cfg.vocab_size} file_vocab={self.vocab.size} unk_id={self.vocab.unk_id} pad_id={getattr(cfg, 'pad_id', 0)}",
                    flush=True,
                )
            except Exception:
                pass

        # Attention aggregator params (if needed)
        if self.chunk_encode and self.chunk_agg == 'attn':
            import torch.nn as nn
            self.attn_w = nn.Linear(cfg.d_model, 1, bias=False).to(device)
        if self.verbose:
            try:
                print(
                    f"[EA] BiTFBackend loaded: q_embed_dim={self.model.q_enc.embed.embedding_dim} "
                    f"d_embed_dim={self.model.d_enc.embed.embedding_dim} chunk={self.chunk_encode} "
                    f"max_len_override={self.override_max_len}",
                    flush=True,
                )
            except Exception:
                pass

    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        # tokenize
        q_tokens = self.vocab.encode(conjecture or '')
        d_tokens_list = [self.vocab.encode(c['text']) for c in candidates]

        # optional truncation to model max_len (to avoid huge length mismatch like 615 vs 256)
        eff_limit: Optional[int] = None
        try:
            model_max = getattr(self.model.q_enc.cfg, 'max_len', None)
        except Exception:
            model_max = None
        if self.override_max_len is not None and model_max is not None:
            eff_limit = min(self.override_max_len, model_max)
        elif self.override_max_len is not None:
            eff_limit = self.override_max_len
        elif model_max is not None:
            eff_limit = model_max
        if eff_limit is not None:
            if len(q_tokens) > eff_limit:
                q_tokens = q_tokens[:eff_limit]
            d_tokens_list = [t[:eff_limit] if len(t) > eff_limit else t for t in d_tokens_list]

        def _encode_chunks(tokens: List[int], which: str):
            if (not self.chunk_encode) or len(tokens) <= self.chunk_len:
                ids, msk = _pad_batch([tokens])
                # Sanitize before device move
                try:
                    enc = self.model.q_enc if which == 'q' else self.model.d_enc
                    model_vs = int(getattr(enc.embed, 'num_embeddings', self.vocab.size))
                except Exception:
                    model_vs = self.vocab.size
                eff_unk = int(self.vocab.unk_id)
                if eff_unk >= model_vs:
                    if self.verbose:
                        try:
                            print(f"[EA WARN] BiTF unk_id {eff_unk} >= model_vocab_size {model_vs}; using fallback unk_id {model_vs-1}", flush=True)
                        except Exception:
                            pass
                    eff_unk = model_vs - 1
                ids = _sanitize_ids_tensor(ids, model_vs, eff_unk, verbose=self.verbose, who=f"bi-{which}")
                ids, msk = ids.to(self.device), msk.to(self.device)
                vec = self.model.encode(ids, msk, which=which)  # [1,D] (already pooled by BiEncoder)
                return vec  # [1,D]
            vecs = []
            taken = 0
            for start in range(0, len(tokens), self.chunk_stride):
                if taken >= self.chunk_max:
                    break
                window = tokens[start:start + self.chunk_len]
                if not window:
                    break
                ids, msk = _pad_batch([window])
                # Sanitize before device move
                try:
                    enc = self.model.q_enc if which == 'q' else self.model.d_enc
                    model_vs = int(getattr(enc.embed, 'num_embeddings', self.vocab.size))
                except Exception:
                    model_vs = self.vocab.size
                eff_unk = int(self.vocab.unk_id)
                if eff_unk >= model_vs:
                    if self.verbose:
                        try:
                            print(f"[EA WARN] BiTF unk_id {eff_unk} >= model_vocab_size {model_vs}; using fallback unk_id {model_vs-1}", flush=True)
                        except Exception:
                            pass
                    eff_unk = model_vs - 1
                ids = _sanitize_ids_tensor(ids, model_vs, eff_unk, verbose=self.verbose, who=f"bi-{which}")
                ids, msk = ids.to(self.device), msk.to(self.device)
                v = self.model.encode(ids, msk, which=which)  # [1,D] (already pooled by BiEncoder)
                vecs.append(v)
                taken += 1
                if len(window) < self.chunk_len:
                    break
            if len(vecs) == 1:
                return vecs[0]
            mat = torch.cat(vecs, dim=0)  # [K,D]
            if self.chunk_agg == 'mean':
                return mat.mean(dim=0, keepdim=True)
            if self.chunk_agg == 'max':
                return mat.max(dim=0, keepdim=True)[0]
            if self.chunk_agg == 'attn':
                w = self.attn_w(mat)  # [K,1]
                att = torch.softmax(w.squeeze(-1), dim=0).unsqueeze(-1)
                return (mat * att).sum(dim=0, keepdim=True)
            return mat.mean(dim=0, keepdim=True)

        with torch.no_grad():
            q_vec = _encode_chunks(q_tokens, 'q')  # [1,D]
            d_vec_list = []
            for toks in d_tokens_list:
                v = _encode_chunks(toks, 'd')  # [1,D]
                d_vec_list.append(v)
            d_vec = torch.cat(d_vec_list, dim=0)  # [B,D]

            # Debug shapes before any processing
            try:
                print(f"[EA DBG] q_vec shape: {tuple(q_vec.shape)}, d_vec shape: {tuple(d_vec.shape)}", flush=True)
            except Exception:
                pass

            # Handle dimension mismatch with projection if needed
            if q_vec.size(1) != d_vec.size(1):
                # First mismatch: log detailed shapes
                if self.verbose:
                    try:
                        print(
                            f"[EA DBG] mismatch before projection q_shape={tuple(q_vec.shape)} d_shape={tuple(d_vec.shape)}",
                            flush=True,
                        )
                    except Exception:
                        pass
                min_d = min(q_vec.size(1), d_vec.size(1))
                # dynamic projection layer (cache on self)
                if not hasattr(self, '_proj_layer'):
                    import torch.nn as nn
                    bigger = q_vec if q_vec.size(1) > d_vec.size(1) else d_vec
                    self._proj_layer = nn.Linear(bigger.size(1), min_d, bias=False).to(self.device)
                    if self.verbose:
                        try:
                            print(f"[EA DBG] created projection layer {bigger.size(1)} -> {min_d}", flush=True)
                        except Exception:
                            pass
                if q_vec.size(1) > min_d:
                    q_vec = self._proj_layer(q_vec)
                if d_vec.size(1) > min_d:
                    # if projection layer made for q, reuse weights (transpose if needed) else create second
                    if q_vec.size(1) == min_d and getattr(self, '_proj_layer', None) is not None and d_vec.size(1) != min_d:
                        # create a second if necessary
                        import torch.nn as nn
                        if not hasattr(self, '_proj_layer_d'):
                            self._proj_layer_d = nn.Linear(d_vec.size(1), min_d, bias=False).to(self.device)
                        d_vec = self._proj_layer_d(d_vec)
                    else:
                        d_vec = self._proj_layer(d_vec)
                if q_vec.size(1) != d_vec.size(1):  # final safeguard
                    min_d2 = min(q_vec.size(1), d_vec.size(1))
                    q_vec = q_vec[:, :min_d2]
                    d_vec = d_vec[:, :min_d2]

            try:
                print(
                    f"[EA DBG] final shapes before dot product: q={tuple(q_vec.shape)}, d={tuple(d_vec.shape)}",
                    flush=True,
                )
            except Exception:
                pass

            s = (q_vec * d_vec).sum(dim=1).float().cpu().tolist()
        return [float(x) for x in s]


# --------------------------- LLM Backends -----------------------------
class _LLMWrapper:
    def __init__(self, model_path: str, dtype=torch.bfloat16):
        from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        self.tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
        if self.tok.pad_token_id is None and self.tok.eos_token_id is not None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(model_path, torch_dtype=dtype, device_map='auto')
        # Safety: ensure tokenizer and model embedding sizes are aligned to avoid GPU index OOB asserts
        try:
            emb = self.model.get_input_embeddings()
            tok_size = len(self.tok)
            if emb is not None and getattr(emb, 'num_embeddings', None) is not None:
                if emb.num_embeddings < tok_size:
                    # Resize model embeddings to tokenizer size (common when pad/eos tokens differ)
                    self.model.resize_token_embeddings(tok_size)
        except Exception:
            pass
        self.model.eval()
    def avg_logprob(self, prompt: str, suffix: str) -> float:
        # Compute mean log P(suffix | prompt)
        full = prompt + suffix
        enc_full = self.tok(full, return_tensors='pt')
        enc_prompt = self.tok(prompt, return_tensors='pt')
        for k in enc_full:
            enc_full[k] = enc_full[k].to(self.model.device)
        with torch.no_grad():
            out = self.model(**enc_full)
            logits = out.logits[:, :-1, :]
            labels = enc_full['input_ids'][:, 1:]
            # Guard against negative offset when prompt is empty or tokenizes to a single special token
            pref_len = max(0, int(enc_prompt['input_ids'].shape[1]) - 1)
            mask = torch.zeros_like(labels, dtype=torch.bool)
            mask[:, pref_len:] = True
            logprobs = torch.nn.functional.log_softmax(logits, dim=-1)
            gather = logprobs.gather(-1, labels.unsqueeze(-1)).squeeze(-1)
            if mask.any():
                return float(gather[mask].mean().item())
        return 0.0


class LLMDIRECTBackend(BaseBackend):
    def __init__(self, model_path: str):
        self.llm = _LLMWrapper(model_path)
    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        q = (conjecture or '') + '\n'
        out: List[float] = []
        for c in candidates:
            out.append(self.llm.avg_logprob(q, c['text']))
        return out


class LLMPMIBackend(BaseBackend):
    def __init__(self, model_path: str, lambda_pmi: float = 0.7):
        self.llm = _LLMWrapper(model_path)
        self.lambda_pmi = lambda_pmi
    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        q = (conjecture or '') + '\n'
        out: List[float] = []
        for c in candidates:
            cond = self.llm.avg_logprob(q, c['text'])
            uncond = self.llm.avg_logprob('', c['text'])
            out.append(cond - self.lambda_pmi * uncond)
        return out


class BiThenLLMBackend(BaseBackend):
    """Shortlist with BiEncoder then rerank topK using LLM (direct or PMI)."""
    def __init__(self, bi_ckpt: str, llm_path: str, topk: int = 128, pmi: bool = False, lambda_pmi: float = 0.7,
                 device: str = 'cuda', chunk_encode: bool = False, chunk_len: int = 256,
                 chunk_stride: int = 192, chunk_max: int = 4, chunk_agg: str = 'mean'):
        self.topk = topk
        self.pmi = pmi
        self.lambda_pmi = lambda_pmi
        self.bi = BiTFBackend(bi_ckpt, device=device,
                              chunk_encode=chunk_encode, chunk_len=chunk_len,
                              chunk_stride=chunk_stride, chunk_max=chunk_max, chunk_agg=chunk_agg)
        self.llm = _LLMWrapper(llm_path)
        # caches
        self._bi_cache: Dict[int, float] = {}
        self._rerank_cache: Dict[int, float] = {}
        # debug / stats containers
        self.last_timing: Dict[str, float] = {}
        self.last_bi_scores: List[float] = []
        self.last_rerank_scores: Dict[int, float] = {}

    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        import time
        t0 = time.time()
        bi_scores_full = self.bi.score(conjecture, candidates)
        self.last_bi_scores = list(bi_scores_full)
        bi_hits = 0
        for idx, s in enumerate(bi_scores_full):
            cid = candidates[idx]['id'] if 'id' in candidates[idx] else idx
            if cid in self._bi_cache and abs(self._bi_cache[cid] - s) < 1e-6:
                bi_hits += 1
            self._bi_cache[cid] = s
        order = sorted(range(len(candidates)), key=lambda i: bi_scores_full[i], reverse=True)
        top_idx = order[: min(self.topk, len(order))]
        t_bi = time.time()
        q = (conjecture or '') + '\n'
        rerank_scores: Dict[int, float] = {}
        rerank_hits = 0
        self.last_rerank_scores = {}
        for i in top_idx:
            cid = candidates[i]['id'] if 'id' in candidates[i] else i
            if cid in self._rerank_cache:
                rerank_scores[i] = self._rerank_cache[cid]
                rerank_hits += 1
                continue
            text = candidates[i]['text']
            cond = self.llm.avg_logprob(q, text)
            if self.pmi:
                uncond = self.llm.avg_logprob('', text)
                val = cond - self.lambda_pmi * uncond
            else:
                val = cond
            self._rerank_cache[cid] = val
            rerank_scores[i] = val
            self.last_rerank_scores[cid] = val
        t_rerank = time.time()
        final = []
        for i in range(len(candidates)):
            final.append(rerank_scores.get(i, bi_scores_full[i]))
        self.last_timing = {
            'cands': len(candidates),
            'topk': len(top_idx),
            'bi_time': t_bi - t0,
            'rerank_time': t_rerank - t_bi,
            'total_time': t_rerank - t0,
            'bi_cache_hits': bi_hits,
            'rerank_cache_hits': rerank_hits,
        }
        return final


class BiThenCrossBackend(BaseBackend):
    """Shortlist with BiEncoder then rerank topK using CrossEncoder."""
    def __init__(self, bi_ckpt: str, cross_ckpt: str, topk: int = 256, device: str = 'cuda', vocab: Optional[str] = None,
                 chunk_encode: bool = False, chunk_len: int = 256, chunk_stride: int = 192,
                 chunk_max: int = 4, chunk_agg: str = 'mean', override_max_len: Optional[int] = None):
        self.topk = topk
        self.bi = BiTFBackend(bi_ckpt, device=device, vocab_path=vocab,
                              chunk_encode=chunk_encode, chunk_len=chunk_len,
                              chunk_stride=chunk_stride, chunk_max=chunk_max, chunk_agg=chunk_agg,
                              override_max_len=override_max_len)
        self.cross = CrossTFBackend(cross_ckpt, device=device, vocab_path=vocab,
                                    override_max_len=override_max_len)
        self._bi_cache: Dict[int, float] = {}
        self._rerank_cache: Dict[int, float] = {}
        self.last_timing: Dict[str, float] = {}
        self.last_bi_scores: List[float] = []
        self.last_rerank_scores: Dict[int, float] = {}

    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        import time
        t0 = time.time()
        bi_scores = self.bi.score(conjecture, candidates)
        self.last_bi_scores = list(bi_scores)
        bi_hits = 0
        for idx, s in enumerate(bi_scores):
            cid = candidates[idx]['id'] if 'id' in candidates[idx] else idx
            if cid in self._bi_cache and abs(self._bi_cache[cid] - s) < 1e-6:
                bi_hits += 1
            self._bi_cache[cid] = s
        order = sorted(range(len(candidates)), key=lambda i: bi_scores[i], reverse=True)
        top_idx = order[: min(self.topk, len(order))]
        t_bi = time.time()
        need_ids: List[int] = []
        need_subset: List[Dict[str, Any]] = []
        for i in top_idx:
            cid = candidates[i]['id'] if 'id' in candidates[i] else i
            if cid not in self._rerank_cache:
                need_ids.append(i)
                need_subset.append(candidates[i])
        self.last_rerank_scores = {}
        if need_subset:
            scored = self.cross.score(conjecture, need_subset)
            for local_i, val in zip(need_ids, scored):
                cid = candidates[local_i]['id'] if 'id' in candidates[local_i] else local_i
                self._rerank_cache[cid] = val
                self.last_rerank_scores[cid] = val
        rerank_hits = len(top_idx) - len(need_ids)
        t_rerank = time.time()
        final: List[float] = []
        for i in range(len(candidates)):
            cid = candidates[i]['id'] if 'id' in candidates[i] else i
            final.append(self._rerank_cache.get(cid, bi_scores[i]))
        self.last_timing = {
            'cands': len(candidates),
            'topk': len(top_idx),
            'bi_time': t_bi - t0,
            'rerank_time': t_rerank - t_bi,
            'total_time': t_rerank - t0,
            'bi_cache_hits': bi_hits,
            'rerank_cache_hits': rerank_hits,
        }
        return final


# ----------------------------- Fusion ---------------------------------
class FusionBackend(BaseBackend):
    def __init__(self, sources: List[Dict[str, float]], weights: List[float]):
        assert len(sources) == len(weights)
        s = sum(weights)
        self.weights = [w / s for w in weights]
        self.sources = sources
    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> List[float]:
        ids = [c['id'] for c in candidates]
        out: List[float] = []
        for cid in ids:
            acc = 0.0
            for w, src in zip(self.weights, self.sources):
                acc += w * src.get(cid, 0.0)
            out.append(acc)
        return out


# --------------------------- Loader -----------------------------------
def load_backend(mode: str, **kwargs) -> BaseBackend:
    if mode == 'clause_tf':
        return ClauseTFBackend(
            kwargs['ckpt'],
            device=kwargs.get('device', 'cuda'),
            vocab_path=kwargs.get('vocab'),
        )
    elif mode == 'cross_tf':
        return CrossTFBackend(
            kwargs['ckpt'],
            device=kwargs.get('device', 'cuda'),
            vocab_path=kwargs.get('vocab'),
            override_max_len=kwargs.get('override_max_len'),
        )
    elif mode == 'bi_tf':
        return BiTFBackend(
            kwargs['ckpt'],
            device=kwargs.get('device', 'cuda'),
            vocab_path=kwargs.get('vocab'),
            chunk_encode=kwargs.get('chunk_encode', False),
            chunk_len=kwargs.get('chunk_len', 256),
            chunk_stride=kwargs.get('chunk_stride', 192),
            chunk_max=kwargs.get('chunk_max', 4),
            chunk_agg=kwargs.get('chunk_agg', 'mean'),
            override_max_len=kwargs.get('override_max_len'),
        )
    elif mode == 'llm_direct':
        return LLMDIRECTBackend(kwargs['llm'])
    elif mode == 'llm_pmi':
        return LLMPMIBackend(
            kwargs['llm'],
            lambda_pmi=kwargs.get('lambda_pmi', 0.7),
        )
    elif mode == 'bi_then_llm':
        return BiThenLLMBackend(
            bi_ckpt=kwargs['bi_ckpt'],
            llm_path=kwargs['llm'],
            topk=kwargs.get('topk', 128),
            pmi=kwargs.get('pmi', False),
            lambda_pmi=kwargs.get('lambda_pmi', 0.7),
            device=kwargs.get('device', 'cuda'),
            chunk_encode=kwargs.get('chunk_encode', False),
            chunk_len=kwargs.get('chunk_len', 256),
            chunk_stride=kwargs.get('chunk_stride', 192),
            chunk_max=kwargs.get('chunk_max', 4),
            chunk_agg=kwargs.get('chunk_agg', 'mean'),
        )
    elif mode == 'bi_then_cross':
        return BiThenCrossBackend(
            bi_ckpt=kwargs['bi_ckpt'],
            cross_ckpt=kwargs['cross_ckpt'],
            topk=kwargs.get('topk', 256),
            device=kwargs.get('device', 'cuda'),
            vocab=kwargs.get('vocab'),
            chunk_encode=kwargs.get('chunk_encode', False),
            chunk_len=kwargs.get('chunk_len', 256),
            chunk_stride=kwargs.get('chunk_stride', 192),
            chunk_max=kwargs.get('chunk_max', 4),
            chunk_agg=kwargs.get('chunk_agg', 'mean'),
            override_max_len=kwargs.get('override_max_len'),
        )
    elif mode == 'fusion':
        return FusionBackend(kwargs['sources'], kwargs['weights'])
    else:
        raise ValueError(f'Unknown backend mode {mode}')


# ---------------------- Utility: softmax (optional) -------------------
def to_probs(scores: List[float], temp: float = 1.0) -> List[float]:
    if not scores:
        return []
    m = max(scores)
    ex = [math.exp((x - m) / max(1e-6, temp)) for x in scores]
    z = sum(ex) or 1.0
    return [e / z for e in ex]


__all__ = [
    'BaseBackend', 'ClauseTFBackend', 'CrossTFBackend', 'BiTFBackend',
    'LLMDIRECTBackend', 'LLMPMIBackend', 'BiThenLLMBackend', 'FusionBackend',
    'load_backend', 'to_probs'
]
