#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Tuple

from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
)


DEFAULT_SPECIAL_TOKENS = ["<CAND_START>", "<CAND_END>", "<Q>", "</Q>", "<D>", "</D>"]


@dataclass
class TokenAlignResult:
    tokenizer: any
    model: any
    added_tokens: int
    cand_start_id: int
    cand_end_id: int


def ensure_special_tokens(
    model_name_or_path: str,
    load_kwargs: Optional[dict] = None,
    extra_tokens: Optional[List[str]] = None,
    pad_to_eos: bool = True,
) -> TokenAlignResult:
    """Load tokenizer+model, register special tokens, resize embeddings, and set pad id.

    Returns tokenizer, model, number of tokens added, and ids for <CAND_START>/<CAND_END>.
    """
    load_kwargs = load_kwargs or {}
    # If model path is local, prefer offline loading to avoid HF access
    is_local = False
    try:
        is_local = Path(model_name_or_path).exists()
    except Exception:
        is_local = False

    tok_kwargs = {"use_fast": True}
    if is_local:
        tok_kwargs["local_files_only"] = True
    tok = AutoTokenizer.from_pretrained(model_name_or_path, **tok_kwargs)
    specials = list(dict.fromkeys((extra_tokens or []) + DEFAULT_SPECIAL_TOKENS))
    added = tok.add_special_tokens({"additional_special_tokens": specials})
    if pad_to_eos and tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model_load = dict(load_kwargs)
    if is_local:
        model_load.setdefault("local_files_only", True)
    model = AutoModelForCausalLM.from_pretrained(model_name_or_path, **model_load)
    model.resize_token_embeddings(len(tok))
    model.config.pad_token_id = tok.pad_token_id

    cand_start_id = tok.convert_tokens_to_ids("<CAND_START>")
    cand_end_id = tok.convert_tokens_to_ids("<CAND_END>")
    return TokenAlignResult(tok, model, added, cand_start_id, cand_end_id)


def locate_token_positions(input_ids, token_id: int):
    """Find positions of a single token id in batched input_ids tensor [B,T] or list.
    Returns List[List[int]] of positions per batch row.
    """
    if hasattr(input_ids, "tolist"):
        rows = input_ids.tolist()
    else:
        rows = input_ids
    pos = []
    for r in rows:
        pos.append([i for i, x in enumerate(r) if x == token_id])
    return pos
