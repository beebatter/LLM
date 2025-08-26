#!/usr/bin/env python3
"""
Utility to convert a raw iProver interactive log into a training dataset.

This script reads a file produced by iProver in interactive mode (commonly
called ``iprover_raw.log``).  The log file contains a stream of JSON
messages separated by either newline characters or NUL characters.  Each
message describes an event in the proof search: new clauses being
registered, clauses entering/leaving various queues, and (if a proof is
found) the list of clause identifiers used in the final proof.  By
collecting these events we can label each generated clause as either a
positive example (appears in the proof) or a negative example (never
used in the proof).

The output of this script is a JSONL (one JSON object per line) where
each entry contains the following keys:

* ``clause_id``: the unique identifier assigned by iProver.
* ``basic_clause_id``: the underlying clause identifier ignoring
  component copies (see iProver docs for details).
* ``text``: the original textual representation of the clause (as sent
  in the ``register_clauses`` message).  You can post‑process this
  further if desired.
* ``features``: a dictionary of clause features provided by iProver
  (e.g. ``conj_dist``, ``born``, ``horn``, ``epr``).
* ``label``: ``1`` if the clause appears in the proof, ``0``
  otherwise.
* ``neg_bucket``: for negative examples, a more fine‑grained category
  indicating why the clause is considered negative.  The possible
  values are::

      NEG_given_nonproof  – clause was selected as a given clause but
                             does not occur in the final proof.
      NEG_passive_only    – clause lived only in the passive queue and
                             was never selected.
      NEG_simplified      – clause was simplified away.
      NEG_never_seen      – clause never left the registration phase.

  Positive examples always have ``neg_bucket=None``.

* ``component`` and ``component_id``: copied from the ``register_clauses``
  message when available (some versions of iProver run multiple
  components concurrently).  If absent, these fields are omitted.

You can extend the output schema by editing the code below.  For
example, you might want to include a canonicalised formula, tags from
the EA, or additional heuristic scores.

Usage::

    python iplog_to_dataset.py --input iprover_raw.log --output out.jsonl

When combined with a script that invokes iProver and the external
agent, you can iterate over a corpus of problems to build a large
labeled dataset for training.
"""

import argparse

from typing import Set

from typing import Set as _Set  # alias to avoid shadowing


from typing import Set as _Set  # alias to avoid shadowing
import re

def extract_proof_ids_from_szs_text(text: str) -> _Set[int]:
    """Extract clause IDs from the SZS CNFRefutation block (plain text)."""
    ids: _Set[int] = set()
    if not text:
        return ids
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith('% SZS output start CNFRefutation'):
            in_block = True
            continue
        if in_block and line.startswith('% SZS output end CNFRefutation'):
            break
        if in_block:
            m = re.search(r"tcf\(\s*c_(\d+)\s*,", line)
            if m:
                try:
                    ids.add(int(m.group(1)))
                except Exception:
                    pass
    return ids

import json
import os
import sys
from typing import Dict, List, Set, Optional


def iter_messages(path: str):
    """Yield JSON messages from a raw interactive log.

    The log may use either NUL (\x00) delimiters, newlines, or a mix.
    This function attempts to split on both.  Blank segments are skipped.
    """
    with open(path, 'rb') as f:
        data = f.read()
    # split on NUL first
    parts = []
    for segment in data.split(b"\x00"):
        # then split on newlines within each segment
        for line in segment.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                msg = json.loads(line.decode('utf-8', errors='ignore'))
            except Exception:
                # best effort: skip malformed lines
                continue
            yield msg


def parse_log(path: str):
    """Parse a raw iProver log and return structures for clause info and labels.

    Returns a tuple ``(clauses, proof_ids, seen)`` where

    * ``clauses`` is a dict mapping clause_id -> dict with keys ``text``,
      ``features``, ``basic_clause_id``, and optionally ``component`` and
      ``component_id``.
    * ``proof_ids`` is a set of clause identifiers that appear in the proof.
    * ``seen`` is a dict with keys ``given``, ``passive``, ``simplified``
      mapping to sets of clause ids that were observed in the corresponding
      events.
    """
    clauses: Dict[int, Dict] = {}
    proof_ids: Set[int] = set()
    seen = {
        'given': set(),
        'passive': set(),
        'simplified': set(),
    }
    # iterate through messages
    for msg in iter_messages(path):
        tag = msg.get('tag')
        if tag == 'register_clauses':
            for entry in msg.get('clauses', []):
                cid = entry.get('clause_id')
                if cid is None:
                    continue
                info = clauses.setdefault(cid, {})
                # raw clause string; may include inference annotations
                info['text'] = entry.get('clause')
                # copy clause_features if present
                info['features'] = entry.get('clause_features', {})
                # basic_clause_id may be useful for deduplication
                if 'basic_clause_id' in info['features']:
                    info['basic_clause_id'] = info['features'].get('basic_clause_id')
                # propagate component and component_id if present at top level
                comp = msg.get('component')
                comp_id = msg.get('component_id')
                if comp is not None:
                    info['component'] = comp
                if comp_id is not None:
                    info['component_id'] = comp_id
        elif tag == 'proof_out':
            ids = msg.get('clause_ids') or []
            # ensure ints
            for cid in ids:
                try:
                    proof_ids.add(int(cid))
                except Exception:
                    pass
        elif tag == 'given_clause':
            ids = msg.get('clause_ids') or []
            for cid in ids:
                try:
                    seen['given'].add(int(cid))
                except Exception:
                    pass
        elif tag == 'passive_clauses':
            ids = msg.get('clause_ids') or []
            for cid in ids:
                try:
                    seen['passive'].add(int(cid))
                except Exception:
                    pass
        elif tag == 'simplified_clauses':
            ids = msg.get('clause_ids') or []
            for cid in ids:
                try:
                    seen['simplified'].add(int(cid))
                except Exception:
                    pass
        # ignore other tags
    # Fallback: if no proof_out was seen in JSON, try to parse SZS CNFRefutation text
    # to recover the list of clause IDs used in the proof. This supports non-interactive
    # proof outputs or logs where only the SZS block was captured.
    ##__PATCH_FALLBACK_SZS__##
    if not proof_ids:
        try:
            with open(path, 'r', encoding='utf-8', errors='ignore') as _f:
                _text = _f.read()
            szs_ids = extract_proof_ids_from_szs_text(_text)
            if szs_ids:
                proof_ids = set(szs_ids)
        except Exception:
            pass
    return clauses, proof_ids, seen


def determine_neg_bucket(cid: int, proof_ids: Set[int], seen: Dict[str, Set[int]]) -> Optional[str]:
    """Given a clause id and the sets of seen events, return a negative bucket label.

    Returns ``None`` for positive examples.  For negative examples, returns
    one of the bucket strings defined in the module docstring.
    """
    if cid in proof_ids:
        return None
    # Precedence matters as events overlap. Use the following priority:
    # 1) given (hardest), 2) simplified, 3) passive-only, 4) never-seen
    # Rationale: many simplified clauses have also appeared in passive;
    # classifying passive before simplified would collapse them into passive_only.
    # Likewise, given_nonproof should take precedence over any later status.
    if cid in seen['given']:
        return 'NEG_given_nonproof'
    # simplified away at some point (even if it passed through passive)
    if cid in seen['simplified']:
        return 'NEG_simplified'
    # clause was in passive queues but never selected as given
    if cid in seen['passive']:
        return 'NEG_passive_only'
    # never seen beyond registration
    return 'NEG_never_seen'


def build_dataset(clauses: Dict[int, Dict], proof_ids: Set[int], seen: Dict[str, Set[int]], problem_name: str = None) -> List[Dict]:
    """Construct a list of dataset entries from parsed log structures."""
    rows = []
    for cid, info in clauses.items():
        row = {
            'problem_name': problem_name,
            'clause_id': cid,
            'basic_clause_id': info.get('basic_clause_id'),
            'text': info.get('text'),
            'features': info.get('features', {}),
            'label': 1 if cid in proof_ids else 0,
        }
        # optional fields
        if 'component' in info:
            row['component'] = info['component']
        if 'component_id' in info:
            row['component_id'] = info['component_id']
        # determine neg_bucket for negatives
        neg_bucket = determine_neg_bucket(cid, proof_ids, seen)
        if neg_bucket:
            row['neg_bucket'] = neg_bucket
        else:
            row['neg_bucket'] = None
        rows.append(row)
    return rows


def main(argv: Optional[List[str]] = None) -> None:
    ap = argparse.ArgumentParser(description='Convert iProver raw log into labeled dataset.')
    ap.add_argument('--input', required=True, help='Path to iprover_raw.log file')
    ap.add_argument('--output', required=True, help='Path to output JSONL file')
    ap.add_argument('--problem-name', default=None, help='Optional problem name to include in each row')
    args = ap.parse_args(argv)

    clauses, proof_ids, seen = parse_log(args.input)
    dataset = build_dataset(clauses, proof_ids, seen, problem_name=args.problem_name)
    # write out JSON lines
    with open(args.output, 'w', encoding='utf-8') as f:
        for row in dataset:
            f.write(json.dumps(row, ensure_ascii=False) + '\n')
    print(f'Wrote {len(dataset)} entries to {args.output}')


if __name__ == '__main__':
    main()