"""
Align interactive JSONL to the teacher_from_casc.jsonl schema.

Input:  /home/ks/LLM/datasets/interactive_sampled_small.jsonl
Output: /home/ks/LLM/datasets/interactive_sampled_small_aligned.jsonl

Rules:
- Preserve problem_name and inner formula text (extract from tcf/cnf wrapper).
- features.horn/epr/unit as ints. Use existing horn/epr booleans if present; compute unit from text.
- features.born and features.conj_dist preserved if present, else defaults born=0, conj_dist=-1.
- Add fields: division=None, url=None, conjecture_text="", source="teacher_iprover", proof_solver="iProver---3.9",
  sample_weight=0.5, item_id (e.g., c_123), role (from wrapper), item_kind (tcf or cnf), label, neg_bucket.
"""

from __future__ import annotations

import json
import os
import re
from typing import Dict, List

SRC = "/home/ks/LLM/datasets/interactive_sampled_small.jsonl"
DST = "/home/ks/LLM/datasets/interactive_sampled_small_aligned.jsonl"

# Regexes similar to the main pipeline
FOF_HEAD_RE = re.compile(r"\bfof\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)
CNF_HEAD_RE = re.compile(r"\bcnf\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)
TCF_HEAD_RE = re.compile(r"\btcf\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)


def extract_formula_from_raw(raw: str) -> str:
    open_pos = raw.find('(')
    if open_pos == -1:
        return ""
    depth, commas, third_start = 0, 0, -1
    for idx in range(open_pos, len(raw)):
        ch = raw[idx]
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 1:
            commas += 1
            if commas == 2:
                j = idx + 1
                while j < len(raw) and raw[j].isspace():
                    j += 1
                if j < len(raw):
                    third_start = j
                break
    if third_start == -1:
        return ""
    d, k = 0, third_start
    out: List[str] = []
    while k < len(raw):
        ch = raw[k]
        out.append(ch)
        if ch == '(':
            d += 1
        elif ch == ')':
            d -= 1
            if d == 0:
                break
        k += 1
    inner = "".join(out).strip()
    if inner.startswith('(') and inner.endswith(')'):
        inner = inner[1:-1].strip()
    return inner


def split_top_level_disj(formula: str) -> List[str]:
    parts: List[str] = []
    s = formula
    depth, start = 0, 0
    for i, ch in enumerate(s):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == '|' and depth == 0:
            parts.append(s[start:i].strip())
            start = i + 1
    parts.append(s[start:].strip())
    return parts


def feat_is_unit(formula: str) -> int:
    return 1 if len(split_top_level_disj(formula)) == 1 else 0


def align_record(rec: Dict) -> Dict:
    raw_text: str = rec.get("text", "")
    # When interactive holds full tcf(...)., extract inner formula and attributes
    item_kind = None
    item_id = None
    role = None
    m = TCF_HEAD_RE.search(raw_text)
    if m:
        item_kind = "tcf"
        item_id = m.group(1)
        role = m.group(2).lower()
    else:
        m2 = CNF_HEAD_RE.search(raw_text)
        if m2:
            item_kind = "cnf"
            item_id = m2.group(1)
            role = m2.group(2).lower()
    inner = extract_formula_from_raw(raw_text) if (m or m2) else raw_text

    features_in = rec.get("features", {}) or {}
    horn = features_in.get("horn")
    if isinstance(horn, bool):
        horn_i = 1 if horn else 0
    else:
        horn_i = int(horn) if isinstance(horn, int) else 0
    epr = features_in.get("epr")
    if isinstance(epr, bool):
        epr_i = 1 if epr else 0
    else:
        epr_i = int(epr) if isinstance(epr, int) else 0
    born = int(features_in.get("born", 0))
    conj_dist = int(features_in.get("conj_dist", -1))
    unit_i = feat_is_unit(inner)

    clause_id = rec.get("clause_id")
    if not item_id and clause_id is not None:
        try:
            item_id = f"c_{int(clause_id)}"
        except Exception:
            item_id = str(clause_id)
    if not role:
        role = "plain"
    if not item_kind:
        item_kind = "tcf" if raw_text.strip().lower().startswith("tcf(") else "cnf"

    out = {
        "problem_name": rec.get("problem_name"),
        "division": None,
        "url": None,
        "conjecture_text": "",
        "text": inner,
        "features": {
            "horn": horn_i,
            "epr": epr_i,
            "unit": unit_i,
            "born": born,
            "conj_dist": conj_dist,
        },
        "label": rec.get("label", 1),
        "neg_bucket": rec.get("neg_bucket"),
        "source": "teacher_iprover",
        "proof_solver": "iProver---3.9",
        "sample_weight": 0.5,
        "item_id": item_id,
        "role": role,
        "item_kind": item_kind,
    }
    return out


def main() -> None:
    count_in = 0
    count_out = 0
    os.makedirs(os.path.dirname(DST), exist_ok=True)
    with open(SRC, "r", encoding="utf-8") as fin, open(DST, "w", encoding="utf-8") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            count_in += 1
            try:
                rec = json.loads(line)
            except Exception:
                continue
            out = align_record(rec)
            fout.write(json.dumps(out, ensure_ascii=False) + "\n")
            count_out += 1
    print(f"aligned {count_out}/{count_in} records -> {DST}")


if __name__ == "__main__":
    main()
