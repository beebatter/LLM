#!/usr/bin/env python3
"""
Train a SentencePiece tokenizer (Unigram/BPE) for Transformer encoders on ATP clause data.

Features:
- Builds a tokenizer training corpus from JSONL datasets with fields: `conjecture_text|conjecture_sig`, `text`, and optional `features`.
- Normalizes variables (TPTP-style) to VAR, preserves logic symbols ()=,~&| as standalone tokens, truncates ultra-long symbols to <SYM_LONG>.
- Adds structure-prefix tokens derived from features: <H0/1>, <U0/1>, <E0/1>, <C0..3>, <B0..3> before each clause line.
- Trains SentencePiece with user_defined_symbols covering tags, prefixes, VAR, logic symbols.
- Provides a quick validation: symbol preservation, tag presence, and average tokenized length over sample lines.

Usage examples:
  python3 scripts/train_sentencepiece_tokenizer.py \
    --input-glob "/home/ks/Training/datasets/**/*.jsonl" \
    --out-dir /home/ks/LLM/models/sp --model-prefix spm_logic \
    --vocab-size 24000 --model-type unigram \
    --build-corpus --train --validate

  # Also include LLM/datasets as extra source
  python3 scripts/train_sentencepiece_tokenizer.py \
    --input-glob "/home/ks/Training/datasets/**/*.jsonl" \
    --input-glob "/home/ks/LLM/datasets/**/*.jsonl" \
    --out-dir /home/ks/LLM/models/sp --model-prefix spm_logic \
    --build-corpus --dedup --max-lines 5000000 --train --validate

Notes:
- This tokenizer is for Bi-/Cross-Encoder only. Keep LLM tokenizer native.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import hashlib
from dataclasses import dataclass
from glob import glob
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


try:
    import sentencepiece as spm
except Exception as e:  # pragma: no cover
    spm = None


# Do not include "," as a user-defined symbol (it conflicts with the comma-separated API);
# it's fine to let comma be learned normally.
LOGIC_SYMS = ["(", ")", "=", "~", "&", "|"]
STRUCT_PREFIX_TOKENS = [
    "<H0>", "<H1>",
    "<U0>", "<U1>",
    "<E0>", "<E1>",
    "<C0>", "<C1>", "<C2>", "<C3>",
    "<B0>", "<B1>", "<B2>", "<B3>",
]
TAG_TOKENS = ["<Q>", "</Q>", "<D>", "</D>"]
SPECIAL_SYMS = ["VAR", "<SYM_LONG>"]


var_pat = re.compile(r"\b[A-Z][A-Za-z0-9_]*\b")  # TPTP variables typically start with uppercase
punct_pat = re.compile(r"([()=,~&|])")
sym_long_pat = re.compile(r"\b[A-Za-z0-9_]{41,}\b")
space_pat = re.compile(r"\s+")


def normalize_text(s: str) -> str:
    s = sym_long_pat.sub("<SYM_LONG>", s)
    s = var_pat.sub("VAR", s)
    s = punct_pat.sub(r" \1 ", s)
    s = space_pat.sub(" ", s).strip()
    return s


@dataclass
class Buckets:
    conj_cuts: Tuple[int, int, int] = (0, 2, 5)  # bins: [0], [1..2], [3..5], [6+]
    born_cuts: Tuple[int, int, int] = (0, 8, 32)  # bins: [0], [1..8], [9..32], [33+]

    def conj_bucket(self, val: Optional[int]) -> int:
        if val is None:
            return 0
        a, b, c = self.conj_cuts
        if val <= a:
            return 0
        if val <= b:
            return 1
        if val <= c:
            return 2
        return 3

    def born_bucket(self, val: Optional[int]) -> int:
        if val is None:
            return 0
        a, b, c = self.born_cuts
        if val <= a:
            return 0
        if val <= b:
            return 1
        if val <= c:
            return 2
        return 3


def features_to_prefix(features: Dict, buckets: Buckets) -> str:
    if not isinstance(features, dict):
        return ""
    horn = 1 if int(features.get("horn", 0)) else 0
    unit = 1 if int(features.get("unit", 0)) else 0
    epr = 1 if int(features.get("epr", 0)) else 0
    conj_dist = features.get("conj_dist")
    born = features.get("born")
    try:
        conj_dist = int(conj_dist) if conj_dist is not None else None
    except Exception:
        conj_dist = None
    try:
        born = int(born) if born is not None else None
    except Exception:
        born = None
    cbin = buckets.conj_bucket(conj_dist)
    bbin = buckets.born_bucket(born)

    tokens = [
        f"<H{horn}>",
        f"<U{unit}>",
        f"<E{epr}>",
        f"<C{cbin}>",
        f"<B{bbin}>",
    ]
    return "".join(tokens) + " "


def iter_jsonl_files(globs: List[str]) -> Iterator[Path]:
    seen = set()
    for g in globs:
        for p in glob(g, recursive=True):
            if p.endswith(".jsonl"):
                if p not in seen:
                    seen.add(p)
                    yield Path(p)


def build_corpus(
    inputs: List[str],
    out_path: Path,
    buckets: Buckets,
    dedup: bool = False,
    max_lines: Optional[int] = None,
    include_q: bool = True,
) -> int:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    n_written = 0
    seen = set() if dedup else None

    with out_path.open("w", encoding="utf-8") as w:
        for fp in iter_jsonl_files(inputs):
            try:
                f = fp.open("r", encoding="utf-8", errors="ignore")
            except Exception:
                continue
            with f:
                for line in f:
                    try:
                        j = json.loads(line)
                    except Exception:
                        continue
                    d = j.get("text")
                    if not d:
                        continue
                    q = j.get("conjecture_text") or j.get("conjecture_sig") or ""
                    features = j.get("features") or {}

                    dn = normalize_text(d)
                    prefix = features_to_prefix(features, buckets)
                    dline = f"{prefix}<D> {dn} </D>".strip()
                    if not seen or dline not in seen:
                        w.write(dline + "\n")
                        n_written += 1
                        if seen is not None:
                            seen.add(dline)

                    if include_q and q:
                        qn = normalize_text(q)
                        qline = f"<Q> {qn} </Q>"
                        if not seen or qline not in seen:
                            w.write(qline + "\n")
                            n_written += 1
                            if seen is not None:
                                seen.add(qline)

                    if max_lines and n_written >= max_lines:
                        return n_written
    return n_written


def train_sentencepiece(
    corpus_path: Path,
    out_dir: Path,
    model_prefix: str,
    vocab_size: int,
    model_type: str = "unigram",
    input_sentence_size: int = 5_000_000,
    character_coverage: float = 1.0,
    user_defined_symbols: Optional[List[str]] = None,
) -> Tuple[Path, Path]:
    if spm is None:
        raise RuntimeError("sentencepiece is not installed. pip install sentencepiece")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_prefix = str(out_dir / model_prefix)
    uds = [u for u in (user_defined_symbols or []) if isinstance(u, str) and len(u) > 0 and u != ","]
    def _try_train(vsz: int):
        spm.SentencePieceTrainer.Train(
            input=str(corpus_path),
            model_prefix=out_prefix,
            vocab_size=vsz,
            model_type=model_type,
            character_coverage=character_coverage,
            input_sentence_size=input_sentence_size,
            shuffle_input_sentence=True,
            user_defined_symbols=",".join(uds),
            unk_piece="<UNK>",
        )

    try:
        _try_train(vocab_size)
    except RuntimeError as e:
        msg = str(e)
        if "Vocabulary size too high" in msg and "<=" in msg:
            # parse allowed maximum
            try:
                allowed = int(msg.split("<=")[-1].split()[0].strip().strip(". "))
                allowed = max(2000, allowed)  # enforce a minimum to avoid tiny vocabs
            except Exception:
                allowed = max(2000, vocab_size // 2)
            _try_train(allowed)
        else:
            raise
    return Path(out_prefix + ".model"), Path(out_prefix + ".vocab")


def validate_model(model_path: Path, sample_path: Path, sample_lines: int = 10_000) -> Dict[str, float | bool]:
    if spm is None:
        raise RuntimeError("sentencepiece is not installed. pip install sentencepiece")
    sp = spm.SentencePieceProcessor(model_file=str(model_path))
    ok_parens = True
    ok_tags = True
    lens: List[int] = []
    # probe fixed samples
    probes = [
        "<Q> VAR & f(VAR) -> g(VAR) </Q>",
        "<H1><U0><E1><C2><B3> <D> ( p(VAR) & q(VAR) ) -> r(VAR) </D>",
    ]
    for s in probes:
        toks = sp.id_to_piece(sp.encode(s, out_type=int))
        # Only check symbols/tags that actually appear in the probe string
        parens_in = [t for t in ["(", ")", "&", "|", "~", "="] if t in s]
        tags_in = [t for t in TAG_TOKENS if t in s]
        ok_parens &= all(x in toks for x in parens_in)
        ok_tags &= all(x in toks for x in tags_in)

    with open(sample_path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f):
            if i >= sample_lines:
                break
            line = line.strip()
            if not line:
                continue
            lens.append(len(sp.encode(line, out_type=int)))
    avg_len = sum(lens) / max(1, len(lens))
    return {"ok_parens": ok_parens, "ok_tags": ok_tags, "avg_len": float(avg_len)}


def hash_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while True:
            b = f.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()[:12]


def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(description="Train SentencePiece tokenizer for ATP clauses with structure-prefix.")
    p.add_argument("--input-glob", action="append", required=False,
                   help="Glob(s) for jsonl sources; can be used multiple times. Default: /home/ks/Training/datasets/**/*.jsonl",
    )
    p.add_argument("--out-corpus", type=Path, default=Path("/home/ks/LLM/tmp/corpus_for_tokenizer.txt"))
    p.add_argument("--dedup", action="store_true", help="Deduplicate lines in-memory (may use large RAM)")
    p.add_argument("--max-lines", type=int, default=None, help="Stop after writing this many lines")
    p.add_argument("--no-q", dest="include_q", action="store_false", help="Do not include <Q> lines")

    p.add_argument("--conj-cuts", type=str, default="0,2,5", help="Conjecture distance cuts a,b,c for 4 bins")
    p.add_argument("--born-cuts", type=str, default="0,8,32", help="Born cuts a,b,c for 4 bins")

    p.add_argument("--out-dir", type=Path, default=Path("/home/ks/Training/models"))
    p.add_argument("--model-prefix", type=str, default="spm_logic")
    p.add_argument("--vocab-size", type=int, default=24000)
    p.add_argument("--model-type", type=str, default="unigram", choices=["unigram", "bpe"])
    p.add_argument("--input-sentence-size", type=int, default=5_000_000)
    p.add_argument("--character-coverage", type=float, default=1.0)

    p.add_argument("--build-corpus", action="store_true")
    p.add_argument("--train", action="store_true")
    p.add_argument("--validate", action="store_true")

    args = p.parse_args(argv)

    default_glob = "/home/ks/Training/datasets/**/*.jsonl"
    input_globs = args.input_glob or [default_glob]

    # Bucketing
    try:
        conj_cuts = tuple(int(x) for x in args.conj_cuts.split(","))  # type: ignore
        born_cuts = tuple(int(x) for x in args.born_cuts.split(","))  # type: ignore
        if len(conj_cuts) != 3 or len(born_cuts) != 3:
            raise ValueError
    except Exception:
        print("Invalid cuts; expect three comma-separated integers for each of conj-cuts and born-cuts", file=sys.stderr)
        return 2
    buckets = Buckets(conj_cuts=conj_cuts, born_cuts=born_cuts)

    # user defined symbols set
    uds = TAG_TOKENS + STRUCT_PREFIX_TOKENS + SPECIAL_SYMS + LOGIC_SYMS

    # Build corpus
    if args.build_corpus:
        n = build_corpus(
            inputs=input_globs,
            out_path=args.out_corpus,
            buckets=buckets,
            dedup=args.dedup,
            max_lines=args.max_lines,
            include_q=args.include_q,
        )
        print(f"Corpus written: {args.out_corpus} (lines={n})")

    # Train tokenizer
    model_path: Optional[Path] = None
    if args.train:
        if not args.out_corpus.exists():
            print(f"Corpus not found: {args.out_corpus}. Run with --build-corpus first.", file=sys.stderr)
            return 2
        model_path, vocab_path = train_sentencepiece(
            corpus_path=args.out_corpus,
            out_dir=args.out_dir,
            model_prefix=args.model_prefix,
            vocab_size=args.vocab_size,
            model_type=args.model_type,
            input_sentence_size=args.input_sentence_size,
            character_coverage=args.character_coverage,
            user_defined_symbols=uds,
        )
        print(f"Tokenizer trained: {model_path} | vocab: {vocab_path}")
        print(f"Model hash: {hash_file(model_path)}")

    # Validate tokenizer
    if args.validate:
        model_file = model_path or (args.out_dir / f"{args.model_prefix}.model")
        if not model_file.exists():
            print(f"Model not found for validation: {model_file}", file=sys.stderr)
            return 2
        if not args.out_corpus.exists():
            print(f"Sample corpus required for validation: {args.out_corpus}", file=sys.stderr)
            return 2
        report = validate_model(model_file, args.out_corpus)
        print("Validation:", report)
        # quick gates
        if not report.get("ok_parens", False):
            print("[WARN] Logic symbols may not be isolated tokens.")
        if not report.get("ok_tags", False):
            print("[WARN] Tag tokens may not be recognized.")
        avg_len = float(report.get("avg_len", 1e9))
        if avg_len > 160:
            print(f"[WARN] Average tokenized length is high ({avg_len:.1f}). Consider larger vocab or more UDS.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
