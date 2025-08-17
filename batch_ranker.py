#!/usr/bin/env python3
"""
Batch ranker: uses the JSON produced by process_iprover_v2.py to
1) generate a compact background summary (once),
2) split candidate clauses into chunks (64/128),
3) ask the LLM to score each chunk (0..1), and
4) normalize + anchor-align scores to produce a global ranking.

Default model is GPT-5 Thinking ("gpt-5").

Usage example:
    python batch_ranker.py \
      --input iprover_llm_output.json \
      --out out_scores.json \
      --chunk-size 128 \
      --anchors 8 \
      --summary-max-tokens 500 \
      --context-summary-k 64 \
      --model gpt-5

Notes
- Set OPENAI_API_KEY in environment for real calls.
- Use --dry-run to test the pipeline without calling the API.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import time
from collections import defaultdict
import hashlib
import shutil
from statistics import mean, pstdev
from typing import Dict, List, Tuple, Any

# --------------------------- I/O ---------------------------

def load_dataset(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(obj: Any, path: str) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)

# ------------------------ Utilities ------------------------
 # --------------------- Semantic helpers ---------------------

_LIT_SPLIT_RE = re.compile(r"\|")
_PRED_RE = re.compile(r"^(~)?\s*([A-Za-z]*P\d+)(?:/(\d+))?")

def parse_literals(formula: str) -> List[Tuple[str, str]]:
    """Return list of literals as tuples: (pol+pred, pred/arity).
    Example: "~P4/2(V0,V1) | Q/1(V1)" -> [("-P4/2","P4/2"),("+Q/1","Q/1")]
    Equality is tagged as "EQ/2" and counted as "+EQ/2" (polarityless here).
    """
    if not formula:
        return []
    lits = []
    parts = [p.strip() for p in _LIT_SPLIT_RE.split(formula)] if "|" in formula else [formula.strip()]
    for raw in parts:
        if not raw:
            continue
        # crude equality detection
        if "=" in raw and not raw.lstrip().startswith("~"):
            lits.append(("+EQ/2", "EQ/2"))
            continue
        if "=" in raw and raw.lstrip().startswith("~"):
            lits.append(("-EQ/2", "EQ/2"))
            continue
        m = _PRED_RE.match(raw)
        if not m:
            # fallback: try to find P#/arity anywhere
            m2 = re.search(r"(P\d+)(?:/(\d+))?", raw)
            if m2:
                pred = m2.group(1)
                ar = m2.group(2) or "?"
                lits.append(("+%s/%s" % (pred, ar), f"{pred}/{ar}"))
            continue
        neg, pred, ar = m.groups()
        ar = ar or "?"
        pol = '-' if neg else '+'
        lits.append((f"{pol}{pred}/{ar}", f"{pred}/{ar}"))
    return lits

def extract_goal_frontier(conjecture_formula: str) -> Dict[str, Any]:
    lits = parse_literals(conjecture_formula)
    pos = {p for pp, p in lits if pp.startswith('+')}
    neg = {p for pp, p in lits if pp.startswith('-')}
    preds = {p.split('/')[0] for p in (pos | neg)}
    return {"pos": pos, "neg": neg, "preds": preds, "has_eq": ("EQ/2" in pos or "EQ/2" in neg)}

TARGET_FUN_RE = re.compile(r"\b(F\d+)\s*\(")
CONST_RE = re.compile(r"\bC\d+\b")

def infer_target_info(conjecture_formula: str) -> dict:
    """
    从猜想里推断“目标函子/参数格局/关键常量集合”。
    重点支持：F1/3(C1, C2, C3) != F1/3(C1, C4, C5) 这类 EUF 风格目标。
    """
    text = conjecture_formula or ""
    # 找出出现的 F#/k（通过 /k 或通过括号判断）
    funs = set(re.findall(r"\b(F\d+/\d+)\b", text))
    # 兜底：有的式子只写 F1(...) 不带 /3
    funs |= set(TARGET_FUN_RE.findall(text))
    # 收集常量
    goal_consts = set(CONST_RE.findall(text))

    # 粗略解析 F#(a,b,c) 形的三元，抓“首参为常量”的信息（比如 C1）
    first_arg_consts = set()
    for m in re.finditer(r"\b(F\d+)\s*\(\s*([A-Z]\d+)\s*,", text):
        first_arg_consts.add(m.group(2))

    return {
        "target_functors": sorted(funs),
        "goal_consts": sorted(goal_consts),
        "first_arg_consts": sorted(first_arg_consts),
    }

def format_target_context_text(ti: dict) -> str:
    parts = []
    if ti.get("target_functors"):
        parts.append("target functors: " + ", ".join(ti["target_functors"]))
    if ti.get("first_arg_consts"):
        parts.append("first-arg consts: " + ", ".join(ti["first_arg_consts"]))
    if ti.get("goal_consts"):
        parts.append("goal consts: " + ", ".join(ti["goal_consts"]))
    return "\n".join(parts)

def _candidate_pred_set(formula: str) -> Tuple[set, set, set]:
    lits = parse_literals(formula)
    pos = {p for pp, p in lits if pp.startswith('+')}
    neg = {p for pp, p in lits if pp.startswith('-')}
    preds = {p.split('/')[0] for p in (pos | neg)}
    return pos, neg, preds

def attach_sat_metrics(cands: List[Dict[str, Any]], sat_map: Dict[str, Any]) -> None:
    """Attach _sat_support/_sat_pressure to candidate dicts, if available.
    sat_map is expected to be {cid: [bool, ...]}.
    """
    if not isinstance(sat_map, dict):
        return
    for c in cands:
        cid = int(c.get("id"))
        vals = sat_map.get(str(cid)) or sat_map.get(cid) or []
        try:
            vals = [bool(v) for v in vals]
        except Exception:
            vals = []
        n = len(vals)
        support = (sum(1 for v in vals if v) / n) if n else None
        pressure = (1.0 - support) if (support is not None) else None
        if support is not None:
            c["_sat_support"] = float(support)
            c["_sat_pressure"] = float(pressure)
            c["_sat_n"] = int(n)

def _count_goal_consts_occurs(s: str, goal_consts: set) -> int:
    return sum(len(re.findall(rf"\b{re.escape(c)}\b", s)) for c in goal_consts)

def _touches_target_functor(s: str, target_functors: set) -> bool:
    if not target_functors:
        return False
    # 同时兼容 F1/3 与 F1( 的写法
    return any((tf in s) or re.search(rf"\b{re.escape(tf.split('/')[0])}\s*\(", s) for tf in target_functors)

def _eq_of_target_functor(s: str, target_functors: set) -> bool:
    if not _touches_target_functor(s, target_functors):
        return False
    # 粗判：出现 F#(...) = F#(...) 或 !=
    return bool(re.search(r"F\d+\s*\([^)]*\)\s*(!=|=)\s*F\d+\s*\(", s))

def _first_arg_in_goal(s: str, first_arg_consts: set) -> bool:
    if not first_arg_consts:
        return False
    # 捕捉 F#(Ck, ... 的格局
    return any(re.search(rf"\bF\d+\s*\(\s*{re.escape(c)}\s*,", s) for c in first_arg_consts)

def semantic_tags_for_clause(c: Dict[str, Any], goal: Dict[str, Any], target_info: Dict[str, Any]) -> List[str]:
    f = c.get("features", {})
    s = c.get("canonical_formula", "") or ""
    unit = ('|' not in s)
    has_eq = ("=" in s) or ("EQ/2" in s)

    # 原有：resolvable / horn / eq / sat / goal_pred_overlap
    pos, neg, preds = _candidate_pred_set(s)
    resolvable = []
    for p in pos:
        if p in goal.get("neg", set()):
            resolvable.append(p)
    for p in neg:
        if p in goal.get("pos", set()):
            resolvable.append(p)

    tags: List[str] = []
    if unit: tags.append("unit")
    if has_eq: tags.append("eq")
    if f.get("horn"): tags.append("horn")
    if resolvable:
        tags.append("resolvable:" + ','.join(sorted(set(resolvable))[:3]))

    if c.get("_sat_support") is not None:
        tags.append(f"sat_support={c['_sat_support']:.2f}")
    if c.get("_sat_pressure") is not None:
        tags.append(f"sat_pressure={c['_sat_pressure']:.2f}")
    ovlp = len(preds & goal.get("preds", set()))
    tags.append(f"goal_pred_overlap={ovlp}")

    # 新增：目标相关的强信号
    tfs = set(target_info.get("target_functors", []))
    first_args = set(target_info.get("first_arg_consts", []))
    goal_consts = set(target_info.get("goal_consts", []))

    if _touches_target_functor(s, tfs):
        tags.append("touches_target_functor")
    if _eq_of_target_functor(s, tfs):
        tags.append("eq_of_target_functor")
    if _first_arg_in_goal(s, first_args):
        tags.append("first_arg_in_goal")
    k = _count_goal_consts_occurs(s, goal_consts)
    tags.append(f"shares_goal_consts:{k}")

    return tags

def compute_and_attach_sem_tags(cands: List[Dict[str, Any]], goal: Dict[str, Any], target_info: Dict[str, Any]) -> None:
    for c in cands:
        c["_sem_tags"] = semantic_tags_for_clause(c, goal, target_info)

def compute_and_attach_sem_tags(cands: List[Dict[str, Any]], goal: Dict[str, Any]) -> None:
    for c in cands:
        c["_sem_tags"] = semantic_tags_for_clause(c, goal)

def select_anchors_semantic(cands: List[Dict[str, Any]], A: int, goal: Dict[str, Any]) -> List[Dict[str, Any]]:
    if A <= 0:
        return []
    def score(c: Dict[str, Any]) -> Tuple:
        tags = c.get("_sem_tags", [])
        s = 0.0
        s += 3.0 if any(t.startswith("resolvable:") and "unit" in tags for t in tags) else 0.0
        s += 2.0 if any(t.startswith("resolvable:") for t in tags) else 0.0
        s += 1.5 if "eq" in tags else 0.0
        s += 1.0 if "horn" in tags else 0.0
        s += 0.5 * float(c.get("_sat_pressure", 0.0))
        # tie-breakers: shorter formula, lower max_func_arity
        f = c.get("features", {})
        return (
            -s,
            len(c.get("canonical_formula", "")),
            f.get("max_func_arity", 99),
        )
    ranked = sorted(cands, key=score)
    # take unique top A
    seen = set()
    out = []
    for c in ranked:
        if c["id"] in seen:
            continue
        out.append(c)
        seen.add(c["id"])
        if len(out) >= A:
            break
    return out

def format_goal_frontier_text(goal: Dict[str, Any]) -> str:
    pos = sorted(goal.get("pos", []))
    neg = sorted(goal.get("neg", []))
    lines = []
    if pos:
        lines.append("+ goals: " + ", ".join(pos[:12]))
    if neg:
        lines.append("- goals: " + ", ".join(neg[:12]))
    if goal.get("has_eq"):
        lines.append("= goals: EQ/2 (equality present)")
    return "\n".join(lines) or "(empty)"


def format_conjecture_targets_text(targets: Dict[str, Any]) -> str:
    """
    Pretty-print conjecture target functors/constants (produced by process_iprover_v2.py).
    targets example:
      {"functors": ["F1/3"], "first_arg_consts": ["C1"], "goal_consts": ["C2","C3","C4","C5"]}
    """
    if not isinstance(targets, dict):
        return ""
    parts = []
    funs = targets.get("functors") or []
    firsts = targets.get("first_arg_consts") or []
    gconsts = targets.get("goal_consts") or []
    if funs:
        parts.append("target functors: " + ", ".join(funs))
    if firsts:
        parts.append("first-arg consts: " + ", ".join(firsts))
    if gconsts:
        parts.append("goal consts: " + ", ".join(gconsts))
    return "\n".join(parts)


def build_reasoning_rules_text() -> str:
    return (
        "Resolution: if ~P(t) is in clause C and P(s) is in a goal clause and t unifies s, resolve to eliminate P.\n"
        "Superposition/Demodulation: if l=r is available and l matches a subterm in a goal/clause, rewrite l→r to simplify.\n"
        "Instantiation (Horn): if A1 & ... & Ak -> B and B unifies a goal, then A1..Ak become subgoals."
    )

SCHEMA = {
    "type": "object",
    "properties": {
        "scores": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "integer"},
                    "score": {"type": "number"}
                },
                "required": ["id", "score"]
            }
        }
    },
    "required": ["scores"]
}

def extract_json(text: str) -> dict:
    """Best-effort extraction of a JSON object from model text output."""
    if not text:
        return {}
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        try:
            return json.loads(text)
        except Exception:
            pass
    start = text.find("{")
    end = text.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(text[start:end+1])
        except Exception:
            pass
    # Fallback: parse lines like "ID 123: 0.08" into a scores JSON
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    pairs = []
    for ln in lines:
        m = re.search(r"ID\s+(\d+)\s*[:：]\s*([-+]?\d+(?:\.\d+)?)", ln)
        if m:
            try:
                pairs.append({"id": int(m.group(1)), "score": float(m.group(2))})
            except Exception:
                continue
    if pairs:
        return {"scores": pairs}
    return {}

# ----------------------- Heuristics -------------------------

def compute_overlap(conj_syms: set, clause: Dict[str, Any]) -> float:
    s = sym_keys(clause)
    if not conj_syms:
        return 0.0
    return len(s & conj_syms) / max(1, len(conj_syms))

def pick_context_for_summary(ctx: List[Dict[str, Any]], K: int = 64) -> List[Dict[str, Any]]:
    def key(c: Dict[str, Any]):
        f = c.get("features", {})
        return (
            f.get("conj_dist", 10**9),
            0 if f.get("horn") else 1,
            0 if f.get("epr") else 1,
            len(c.get("canonical_formula", "")),
            -c.get("_ovlp", 0.0),
        )
    return sorted(ctx, key=key)[:K]

def make_chunks(cands: List[Dict[str, Any]], chunk_payload_size: int, anchors: List[Dict[str, Any]]) -> List[List[Dict[str, Any]]]:
    # Simplified: no anchors, split candidates into chunks of given payload size
    pool = list(cands)
    if chunk_payload_size <= 0:
        return [pool] if pool else []
    chunks: List[List[Dict[str, Any]]] = [pool[i:i+chunk_payload_size] for i in range(0, len(pool), chunk_payload_size)]
    if not chunks and pool:
        chunks = [pool[:chunk_payload_size]]
    return chunks

# ---------------------- Prompt Builders ---------------------

def build_symbol_cheatsheet(symbol_map: Dict[str, Any], keys: List[str], max_items: int = 120) -> str:
    rows = []
    for k in sorted(keys)[:max_items]:
        info = symbol_map.get(k)
        if not info:
            continue
        orig = ", ".join(info.get("original", []))
        rows.append(f"{k} ⇨ {orig}  ({info.get('kind')}, arity={info.get('arity')})")
    return "\n".join(rows)

def _find_run_root(out_dir: str) -> str:
    """Best-effort detection of the EA run root directory.
    We expect args.out like: .../EA.<port>.<ts>/requests/scores_req_xxx/out_scores.json
    Return the parent of 'requests' if found; otherwise return out_dir.
    """
    cur = os.path.abspath(out_dir)
    # Limit the climb depth to avoid walking to filesystem root indefinitely
    for _ in range(5):
        base = os.path.basename(cur)
        if base == "requests":
            return os.path.dirname(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(out_dir)

def _prompt_fingerprint(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

def build_scoring_prompt(summary: str, cheatsheet: str, conjecture_formula: str, chunk: List[Dict[str, Any]], goal_text: str, rules_text: str, prompt_path: str = None) -> str:
    """
    构建评分prompt，支持从文件加载模板。
    prompt_path: 指定prompt模板文件路径，默认为prompts/scoring_prompt.txt。
    """
    import pathlib
    def _merge_tags(c: Dict[str, Any]) -> List[str]:
        ea_tags = c.get("tags", []) or []
        sem_tags = c.get("_sem_tags", []) or []
        merged = []
        for t in (ea_tags + sem_tags):
            if t not in merged:
                merged.append(t)
        return merged
    def _line(c: Dict[str, Any]) -> str:
        tag_str = ", ".join(_merge_tags(c))
        return f"- ID {c['id']} | tags: [{tag_str}]\n  formula: {c['canonical_formula']}"
    lines = "\n".join(_line(c) for c in chunk)
    example = """
{
  "scores": [
    {"id": 12345, "score": 42, "why": "unit resolvable"},
    {"id": 23456, "score": 5,  "why": "no bridge"}
  ]
}
""".strip()
    # 默认prompt路径
    if prompt_path is None:
        prompt_path = str(pathlib.Path(__file__).parent / "prompts/scoring_prompt.txt")
    # 读取模板
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    # 替换变量
    prompt = template.replace("{{goal_text}}", goal_text)
    prompt = prompt.replace("{{rules_text}}", rules_text)
    prompt = prompt.replace("{{summary}}", summary)
    prompt = prompt.replace("{{cheatsheet}}", cheatsheet)
    prompt = prompt.replace("{{conjecture_formula}}", conjecture_formula)
    prompt = prompt.replace("{{lines}}", lines)
    prompt = prompt.replace("{{example}}", example)
    return prompt

# -------------------------- LLM -----------------------------

class LLMClient:
    def __init__(self, model: str = "gpt-5", temperature: float = 1.0, dry_run: bool = False, max_retries: int = 3, verbose: bool = False):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.verbose = verbose
        self._goal_text = ""
        self._rules_text = ""
        # Auto-fix temperature for models that require default=1
        if ("gpt-5" in self.model) and (not self.dry_run) and (temperature != 1.0):
            self.temperature = 1.0
            if self.verbose:
                print("[LLM] Model requires default temperature; overriding to 1.0")
        self._client = None
        if not dry_run:
            try:
                # New-style SDK
                from openai import OpenAI  # type: ignore
                self._client = OpenAI()
            except Exception:
                # Old-style SDK fallback
                try:
                    import openai  # type: ignore
                    self._client = openai
                except Exception:
                    raise RuntimeError("OpenAI SDK not available. Install `openai` and set OPENAI_API_KEY.")

    def _chat(self, prompt: str) -> str:
        if self.dry_run:
            text = prompt.strip()
            # Detect scoring prompt more flexibly
            if ("ATP 子句打分器" in text) or ("子句打分器" in text):
                # Robustly extract IDs even if bullets/colons vary
                ids = [int(m) for m in re.findall(r"ID\s+(\d+)", text)]
                # produce 0..50 scores with tiny bias by id
                payload = {
                    "scores": [
                        {"id": i, "score": int((i % 51)), "why": "mock"} for i in ids
                    ]
                }
                return json.dumps(payload, ensure_ascii=False)
            # Otherwise treat it as a summary prompt
            return "(占位) 背景摘要：…"
        last_err = None
        for _ in range(self.max_retries):
            if self.verbose:
                print(f"[LLM] request try {_+1}/{self.max_retries}...")
            try:
                # New-style SDK path
                if hasattr(self._client, "chat"):
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": "You are a careful assistant. Always follow the user instructions strictly."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return resp.choices[0].message.content or ""
                # Old-style SDK path
                if hasattr(self._client, "ChatCompletion"):
                    resp = self._client.ChatCompletion.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": "You are a careful assistant. Always follow the user instructions strictly."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return resp["choices"][0]["message"]["content"]
            except Exception as e:  # pragma: no cover
                if self.verbose:
                    print(f"[LLM] error: {e}. retrying...")
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

    def set_reasoning_context(self, goal_text: str, rules_text: str) -> None:
        self._goal_text = goal_text
        self._rules_text = rules_text

    def summarize(self, conjecture_formula: str, cheatsheet: str, ctx_list: List[Dict[str, Any]], max_tokens: int) -> str:
        prompt = build_summary_prompt(conjecture_formula, cheatsheet, ctx_list, max_tokens)
        return self._chat(prompt)

    def score_chunk(self, summary: str, cheatsheet: str, conjecture_formula: str, chunk: List[Dict[str, Any]]) -> Dict[str, Any]:
        prompt = build_scoring_prompt(summary, cheatsheet, conjecture_formula, chunk, self._goal_text, self._rules_text)
        text = self._chat(prompt)
        parsed = extract_json(text)
        coerced = coerce_scores(parsed, chunk)
        return {"scores": coerced}

# ---------------- Score coercion and flat-spread helpers -----------------
from typing import Any, Tuple

def coerce_scores(obj: Any, chunk: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Normalize various model output shapes into a list of {id, score} dicts.
    Accepts:
      - {"scores": [{"id":..,"score":..,("why":..)}, ...]}
      - {"scores": {"123": 0.8, "124": {"score": 12, "why": "..."}, ...}}
      - {"123": 0.8, "124": 0.2, ...}   (top-level mapping)
      - [0.1, 0.2, ...]                 (aligned by order of IDs in `chunk`)
    Falls back to zeros if nothing usable is found.
    """
    ids = [int(c.get("id")) for c in (chunk or []) if "id" in c]
    out: List[Dict[str, Any]] = []

    # If obj is a dict, try to dig out the "scores" field; otherwise treat the dict as a mapping.
    data = None
    if isinstance(obj, dict):
        data = obj.get("scores", obj)
    else:
        data = obj

    # Case 1: list of dicts
    if isinstance(data, list) and data and all(isinstance(x, dict) and "id" in x and "score" in x for x in data):
        for x in data:
            try:
                out.append({"id": int(x["id"]), "score": float(x["score"]), **({"why": x.get("why")} if "why" in x else {})})
            except Exception:
                pass
        if out:
            return out

    # Case 2: dict mapping id -> score or id -> {"score":..., ...}
    if isinstance(data, dict):
        for k, v in data.items():
            try:
                cid = int(k)
                if isinstance(v, dict) and "score" in v:
                    sc = float(v.get("score", 0.0))
                    rec = {"id": cid, "score": sc}
                    if "why" in v:
                        rec["why"] = v["why"]
                    out.append(rec)
                else:
                    sc = float(v)
                    out.append({"id": cid, "score": sc})
            except Exception:
                continue
        if out:
            return out

    # Case 3: list of scalars aligned with chunk IDs
    if isinstance(data, list) and data and all(isinstance(x, (int, float, str)) for x in data) and len(ids) == len(data):
        for cid, val in zip(ids, data):
            try:
                out.append({"id": int(cid), "score": float(val)})
            except Exception:
                out.append({"id": int(cid), "score": 0.0})
        if out:
            return out

    # Fallback: zeros aligned to chunk
    if ids:
        return [{"id": int(cid), "score": 0.0} for cid in ids]
    return out

def spread_if_chunk_flat(scores_arr: List[Dict[str, Any]], chunk: List[Dict[str, Any]], eps: float = 1e-9) -> Tuple[List[Dict[str, Any]], bool]:
    # 保留以备兼容，但不再对扁平分数做扩散（简化版不需要）
    return scores_arr, False

def sym_keys(clause: Dict[str, Any]) -> set:
    return set(clause.get("local_symbols", {}).keys())

# ---------------------- Prompt Builders ---------------------

def build_symbol_cheatsheet(symbol_map: Dict[str, Any], keys: List[str], max_items: int = 120) -> str:
    rows = []
    for k in sorted(keys)[:max_items]:
        info = symbol_map.get(k)
        if not info:
            continue
        orig = ", ".join(info.get("original", []))
        rows.append(f"{k} ⇨ {orig}  ({info.get('kind')}, arity={info.get('arity')})")
    return "\n".join(rows)

def _find_run_root(out_dir: str) -> str:
    """Best-effort detection of the EA run root directory.
    We expect args.out like: .../EA.<port>.<ts>/requests/scores_req_xxx/out_scores.json
    Return the parent of 'requests' if found; otherwise return out_dir.
    """
    cur = os.path.abspath(out_dir)
    # Limit the climb depth to avoid walking to filesystem root indefinitely
    for _ in range(5):
        base = os.path.basename(cur)
        if base == "requests":
            return os.path.dirname(cur)
        parent = os.path.dirname(cur)
        if parent == cur:
            break
        cur = parent
    return os.path.abspath(out_dir)

def _prompt_fingerprint(prompt_text: str) -> str:
    return hashlib.sha256(prompt_text.encode("utf-8")).hexdigest()

def build_summary_prompt(conjecture_formula: str, cheatsheet: str, ctx_list: List[Dict[str, Any]], max_tokens: int, prompt_path: str = None) -> str:
    """
    构建摘要prompt，支持从文件加载模板。
    prompt_path: 指定prompt模板文件路径，默认为prompts/summary_prompt.txt。
    """
    import pathlib
    ctx_text = "\n".join(f"- ID {c['id']}: {c['canonical_formula']}" for c in ctx_list[:80])
    if prompt_path is None:
        prompt_path = str(pathlib.Path(__file__).parent / "prompts/summary_prompt.txt")
    with open(prompt_path, "r", encoding="utf-8") as f:
        template = f.read()
    prompt = template.replace("{{max_tokens}}", str(max_tokens))
    prompt = prompt.replace("{{conjecture_formula}}", conjecture_formula)
    prompt = prompt.replace("{{cheatsheet}}", cheatsheet)
    prompt = prompt.replace("{{ctx_text}}", ctx_text)
    return prompt

def build_scoring_prompt(summary: str, cheatsheet: str, conjecture_formula: str, chunk: List[Dict[str, Any]], goal_text: str, rules_text: str) -> str:
    def _merge_tags(c: Dict[str, Any]) -> List[str]:
        # Merge EA tags (from process_iprover_v2.py) with local semantic tags
        ea_tags = c.get("tags", []) or []
        sem_tags = c.get("_sem_tags", []) or []
        merged = []
        for t in (ea_tags + sem_tags):
            if t not in merged:
                merged.append(t)
        return merged

    def _line(c: Dict[str, Any]) -> str:
        tag_str = ", ".join(_merge_tags(c))
        return f"- ID {c['id']} | tags: [{tag_str}]\n  formula: {c['canonical_formula']}"

    lines = "\n".join(_line(c) for c in chunk)
    example = """
{
  "scores": [
    {"id": 12345, "score": 42, "why": "unit resolvable"},
    {"id": 23456, "score": 5,  "why": "no bridge"}
  ]
}
""".strip()
    return f"""
你是 ATP 子句打分器。请仅基于推理可用性为每个候选打分，并输出紧凑 JSON。
- 分值范围：0–50 的整数或小数；
- 仅基于推理可用性打分（0–50）：
  A: 45–50 直接触达 F1/3（含 F1 等式/不等式，或一跳可重写到目标式样）
  B: 30–44 需一跳桥接到 F1（上下文/anchors 可见明确桥接）
  C: 15–29 结构可用但当前缺桥（投影/置换/读参模板未与 F1 接通）
  D: 0–14 与目标无关或仅变元自等
- 优先级：一跳解析/重写 > 注入/观测 > 其他 Horn；禁止使用频度/重合度/长度为理由。
- **使用 tags（强信号）**：
  - `eq_of_target_functor` ⇒ A 档强加分；
  - `touches_target_functor` 且 `first_arg_in_goal` ⇒ B 档；
  - `shares_goal_consts:k` 仅微调（k 越大越高），`horn`/`unit` 仅用于并列打破；
  - `sat_support=..`/`sat_pressure=..` 只作轻微微调。
- 只输出 JSON，每条包含 id、score、why（≤12字；如：含F1等式/一跳可重写/有桥可投影/投影缺桥/与目标无桥）。

【目标前沿（抽象+目标模式）】
{goal_text}

【推理规则（参考）】
{rules_text}

【背景摘要（全局共享）】
{summary}

【符号速查表（节选）】
{cheatsheet}

【猜想】
{conjecture_formula}

【待评分子句（每条含 tags + 公式）】
{lines}

【输出示例】
{example}
""".strip()

# -------------------------- LLM -----------------------------

class LLMClient:
    def __init__(self, model: str = "gpt-5", temperature: float = 1.0, dry_run: bool = False, max_retries: int = 3, verbose: bool = False):
        self.model = model
        self.temperature = temperature
        self.dry_run = dry_run
        self.max_retries = max_retries
        self.verbose = verbose
        self._goal_text = ""
        self._rules_text = ""
        # Auto-fix temperature for models that require default=1
        if ("gpt-5" in self.model) and (not self.dry_run) and (temperature != 1.0):
            self.temperature = 1.0
            if self.verbose:
                print("[LLM] Model requires default temperature; overriding to 1.0")
        self._client = None
        if not dry_run:
            try:
                # New-style SDK
                from openai import OpenAI  # type: ignore
                self._client = OpenAI()
            except Exception:
                # Old-style SDK fallback
                try:
                    import openai  # type: ignore
                    self._client = openai
                except Exception:
                    raise RuntimeError("OpenAI SDK not available. Install `openai` and set OPENAI_API_KEY.")

    def _chat(self, prompt: str) -> str:
        if self.dry_run:
            text = prompt.strip()
            # Detect scoring prompt more flexibly
            if ("ATP 子句打分器" in text) or ("子句打分器" in text):
                # Robustly extract IDs even if bullets/colons vary
                ids = [int(m) for m in re.findall(r"ID\s+(\d+)", text)]
                # produce 0..50 scores with tiny bias by id
                payload = {
                    "scores": [
                        {"id": i, "score": int((i % 51)), "why": "mock"} for i in ids
                    ]
                }
                return json.dumps(payload, ensure_ascii=False)
            # Otherwise treat it as a summary prompt
            return "(占位) 背景摘要：…"
        last_err = None
        for _ in range(self.max_retries):
            if self.verbose:
                print(f"[LLM] request try {_+1}/{self.max_retries}...")
            try:
                # New-style SDK path
                if hasattr(self._client, "chat"):
                    resp = self._client.chat.completions.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": "You are a careful assistant. Always follow the user instructions strictly."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return resp.choices[0].message.content or ""
                # Old-style SDK path
                if hasattr(self._client, "ChatCompletion"):
                    resp = self._client.ChatCompletion.create(
                        model=self.model,
                        temperature=self.temperature,
                        messages=[
                            {"role": "system", "content": "You are a careful assistant. Always follow the user instructions strictly."},
                            {"role": "user", "content": prompt},
                        ],
                    )
                    return resp["choices"][0]["message"]["content"]
            except Exception as e:  # pragma: no cover
                if self.verbose:
                    print(f"[LLM] error: {e}. retrying...")
                last_err = e
                time.sleep(1.0)
        raise RuntimeError(f"LLM request failed after retries: {last_err}")

    def set_reasoning_context(self, goal_text: str, rules_text: str) -> None:
        self._goal_text = goal_text
        self._rules_text = rules_text

    def summarize(self, conjecture_formula: str, cheatsheet: str, ctx_list: List[Dict[str, Any]], max_tokens: int, prompt_path: str = None) -> str:
        prompt = build_summary_prompt(conjecture_formula, cheatsheet, ctx_list, max_tokens, prompt_path=prompt_path)
        return self._chat(prompt)

    def score_chunk(self, summary: str, cheatsheet: str, conjecture_formula: str, chunk: List[Dict[str, Any]], prompt_path: str = None) -> Dict[str, Any]:
        prompt = build_scoring_prompt(summary, cheatsheet, conjecture_formula, chunk, self._goal_text, self._rules_text, prompt_path=prompt_path)
        text = self._chat(prompt)
        parsed = extract_json(text)
        coerced = coerce_scores(parsed, chunk)
        return {"scores": coerced}

# ---------------------- Normalization ----------------------

def normalize_and_align(chunks_json: List[Dict[str, Any]], anchor_ids: List[int]) -> Dict[int, float]:
    # 简化：不再做 z-score 或锚点对齐，直接聚合并做全局 min-max 归一化（模型输出应为 0..50）
    bag: Dict[int, List[float]] = defaultdict(list)
    for js in chunks_json:
        for x in js.get("scores", []):
            try:
                bag[int(x["id"])].append(float(x["score"]))
            except Exception:
                pass
    # 平均
    agg = {cid: (sum(v)/len(v)) for cid, v in bag.items() if v}
    if not agg:
        return {}
    vals = list(agg.values())
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        return {cid: 0.5 for cid in agg.keys()}
    return {cid: (sc - vmin) / (vmax - vmin) for cid, sc in agg.items()}

# --------------------------- Main ---------------------------

def main():
    ap = argparse.ArgumentParser(description="Chunked clause ranking with background summary + anchor alignment")
    ap.add_argument("--input", type=str, default="iprover_llm_output.json", help="dataset JSON from process_iprover_v2.py")
    ap.add_argument("--out", type=str, default="out_scores.json", help="final scores JSON path")
    ap.add_argument("--chunk-size", type=int, default=128, help="number of NON-ANCHOR clauses per chunk")
    ap.add_argument("--anchors", type=int, default=8, help="anchor clauses replicated across chunks for alignment")
    ap.add_argument("--context-summary-k", type=int, default=64, help="how many context clauses to feed into summary")
    ap.add_argument("--summary-max-tokens", type=int, default=500, help="summary token budget (approx)")
    ap.add_argument("--cheatsheet-max-items", type=int, default=120, help="max symbol entries in cheatsheet")
    ap.add_argument("--model", type=str, default="gpt-5", help="LLM model name (default: gpt-5)")
    ap.add_argument("--temperature", type=float, default=1.0, help="LLM temperature")
    ap.add_argument("--max-retries", type=int, default=3, help="LLM call retries")
    ap.add_argument("--dry-run", action="store_true", help="do not call API; generate mock scores")
    ap.add_argument("--verbose", action="store_true", help="print debug info about chunking and scoring")
    # save-prompts hard-wired on by default; keep the flag for compatibility but it is ignored
    ap.add_argument("--save-prompts", action="store_true", help="(ignored; always on) dump prompts and raw responses to files")
    ap.add_argument("--progress", action="store_true", help="print live progress while LLM is running")
    args = ap.parse_args()

    ds = load_dataset(args.input)
    sym_map: Dict[str, Any] = ds.get("symbol_map", {})
    conj = ds.get("conjecture")
    ctx = ds.get("context_clauses", [])
    cands = ds.get("candidate_clauses", [])
    conj_targets: Dict[str, Any] = ds.get("conjecture_targets", {}) or {}
    eqf: Dict[str, Any] = ds.get("ea_query_features", {}) or {}
    sat_map: Dict[str, Any] = eqf.get("sat_lit_gr_vals", {}) or {}

    if args.verbose:
        print(f"[DEBUG] clauses: context={len(ctx)}, candidates={len(cands)}")

    if not conj:
        raise SystemExit("No conjecture found in dataset.")

    # overlap feature
    conj_syms = sym_keys(conj)
    for c in ctx:
        c["_ovlp"] = compute_overlap(conj_syms, c)
    for c in cands:
        c["_ovlp"] = compute_overlap(conj_syms, c)

    # pick summary context & cheatsheet keys (from conjecture + summary context)
    sum_ctx = pick_context_for_summary(ctx, K=args.context_summary_k)
    used_keys = set(conj.get("local_symbols", {}).keys())
    for c in sum_ctx:
        used_keys |= set(c.get("local_symbols", {}).keys())
    cheatsheet = build_symbol_cheatsheet(sym_map, sorted(used_keys), max_items=args.cheatsheet_max_items)

    # Determine output base directory (hard-coded policy):
    # If args.out has a directory (e.g., when invoked by EA under artifacts), use that.
    # Otherwise, place under /home/ks/logs/Ranker.<pid>.<ts>/
    out_dir = os.path.dirname(args.out) or "."
    if out_dir == ".":
        try:
            run_base = f"/home/ks/LLM/Logs/Ranker.{os.getpid()}.{int(time.time())}"
            os.makedirs(run_base, exist_ok=True)
            args.out = os.path.join(run_base, os.path.basename(args.out))
            out_dir = run_base
        except Exception:
            os.makedirs(out_dir, exist_ok=True)

    # LLM clients (primary + optional small-batch client)
    llm = LLMClient(model=args.model, temperature=args.temperature, dry_run=args.dry_run, max_retries=args.max_retries, verbose=(args.progress or args.verbose))
    SMALL_BATCH_THRESHOLD = 17
    # Use a standard chat-completions model, not a realtime model
    SMALL_BATCH_MODEL = "gpt-4o-mini"
    llm_small = None  # lazy init

    # Background summary: cache per run using the exact prompt as the key
    os.makedirs(out_dir, exist_ok=True)
    summary_prompt = build_summary_prompt(conj.get("canonical_formula", ""), cheatsheet, sum_ctx, args.summary_max_tokens)
    fp = _prompt_fingerprint(summary_prompt)
    run_root = _find_run_root(out_dir)
    cache_dir = os.path.join(run_root, "summary_cache", fp)
    cached_resp_path = os.path.join(cache_dir, "summary_response.txt")
    cached_prompt_path = os.path.join(cache_dir, "summary_prompt.txt")
    os.makedirs(cache_dir, exist_ok=True)

    summary: str
    if os.path.exists(cached_resp_path):
        # Reuse cached summary and copy artifacts into current out_dir for convenience
        with open(cached_resp_path, "r", encoding="utf-8") as f:
            summary = f.read()
        # Ensure cached prompt is present (for completeness)
        if not os.path.exists(cached_prompt_path):
            with open(cached_prompt_path, "w", encoding="utf-8") as f:
                f.write(summary_prompt)
        # Mirror to request directory
        shutil.copyfile(cached_prompt_path, os.path.join(out_dir, "summary_prompt.txt"))
        shutil.copyfile(cached_resp_path, os.path.join(out_dir, "summary_response.txt"))
        if args.progress:
            print("[LLM] Using cached background summary.")
    else:
        if args.progress:
            print("[LLM] Generating background summary...")
        # Save prompt to cache and out_dir
        with open(cached_prompt_path, "w", encoding="utf-8") as sp:
            sp.write(summary_prompt)
        with open(os.path.join(out_dir, "summary_prompt.txt"), "w", encoding="utf-8") as sp2:
            sp2.write(summary_prompt)
        # Generate via LLM (or dry-run), then save in cache and out_dir
        summary = llm.summarize(conj.get("canonical_formula", ""), cheatsheet, sum_ctx, args.summary_max_tokens)
        with open(cached_resp_path, "w", encoding="utf-8") as sr:
            sr.write(summary)
        with open(os.path.join(out_dir, "summary_response.txt"), "w", encoding="utf-8") as sr2:
            sr2.write(summary)
        if args.progress:
            print("[LLM] Summary ready.")

    # Semantic helpers: compute goal frontier, rules, attach SAT metrics & semantic tags, set LLM context

    conj_formula = conj.get("canonical_formula", "")
    target_info = infer_target_info(conj_formula)
    target_text = format_target_context_text(target_info)

    # 把 target_text 拼到 goal_text（不改函数签名，最小侵入）
    goal = extract_goal_frontier(conj_formula)
    goal_text = format_goal_frontier_text(goal)
    if target_text:
        goal_text = goal_text + "\n" + target_text
    goal_text_with_targets = goal_text + (("\n" + targets_text) if targets_text else "")
    rules_text = build_reasoning_rules_text()

    # attach SAT metrics (if any) and compute semantic tags
    attach_sat_metrics(cands, sat_map)
    compute_and_attach_sem_tags(cands, goal, target_info)

    # Provide reasoning context to the client
    llm.set_reasoning_context(goal_text_with_targets, rules_text)

    # anchors: 简化版不使用 anchors
    requested_A = 0
    anchors: List[Dict[str, Any]] = []
    anchor_ids: List[int] = []

    # chunking（不带 anchors）
    chunks = make_chunks(cands, args.chunk_size, anchors)
    if args.verbose:
        pool_size = len(cands)
        print(f"[DEBUG] anchors={len(anchor_ids)} pool={pool_size} chunks={len(chunks)}")
        if chunks:
            print(f"[DEBUG] chunk[0] size={len(chunks[0])}")

    # per-chunk scoring（保持 artifacts 输出）
    chunk_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_results: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks):
        use_small = len(ch) < SMALL_BATCH_THRESHOLD
        client = llm_small if (use_small and llm_small is not None) else llm
        if use_small and llm_small is None:
            llm_small = LLMClient(model=SMALL_BATCH_MODEL, temperature=args.temperature, dry_run=args.dry_run, max_retries=args.max_retries, verbose=(args.progress or args.verbose))
            client = llm_small
        if args.progress:
            mdl = getattr(client, "model", "?")
            print(f"[LLM] Scoring chunk {idx+1}/{len(chunks)} (size={len(ch)}) with model={mdl}...")
        prompt = build_scoring_prompt(summary, cheatsheet, conj_formula, ch, goal_text_with_targets, rules_text)
        with open(os.path.join(chunk_dir, f"prompt_{idx:03d}.txt"), "w", encoding="utf-8") as pf:
            pf.write(prompt)
        client.set_reasoning_context(goal_text, rules_text)
        text = ""
        try:
            text = client._chat(prompt)
        except Exception as e:
            if args.progress:
                print(f"[LLM] model failed: {e}. Using zero scores for this chunk.")
            text = ""
        with open(os.path.join(chunk_dir, f"response_{idx:03d}.txt"), "w", encoding="utf-8") as rf:
            rf.write(text)
        parsed = extract_json(text)
        scores_arr = coerce_scores(parsed, ch)
        scores_arr, used_spread = spread_if_chunk_flat(scores_arr, ch)
        res = {"scores": scores_arr, "meta": {"model": getattr(client, "model", args.model)}}
        save_json(res, os.path.join(chunk_dir, f"chunk_{idx:03d}.json"))
        chunk_results.append(res)
        if args.verbose:
            print(f"[DEBUG] chunk_{idx:03d}: scores={len(res.get('scores', []))}")

    # 简化汇总：全局 min-max 归一化到 0..1
    total_scored = sum(len(js.get("scores", [])) for js in chunk_results)
    if args.verbose:
        print(f"[DEBUG] total_scored entries across chunks: {total_scored}")
    if total_scored == 0:
        print("[WARN] No scores produced by chunks. Saving empty scores.")
        save_json({"scores": []}, args.out)
        return

    final_scores = normalize_and_align(chunk_results, anchor_ids)
    vals_dbg = list(final_scores.values())
    if vals_dbg:
        print(f"[CAL] global min={min(vals_dbg):.3f} max={max(vals_dbg):.3f} mean={sum(vals_dbg)/len(vals_dbg):.3f}")

    # 输出（按分数降序）
    ranking_items = sorted(((int(cid), float(sc)) for cid, sc in final_scores.items()), key=lambda kv: kv[1], reverse=True)
    result = {
        "meta": {
            "model": args.model,
            "temperature": args.temperature,
            "chunk_size": args.chunk_size,
            "anchors": 0,
            "context_summary_k": args.context_summary_k,
            "summary_max_tokens": args.summary_max_tokens,
            "dry_run": args.dry_run,
            "timestamp": int(time.time()),
            "debug": {
                "context": len(ctx),
                "candidates": len(cands),
                "anchors": 0,
                "chunks": len(chunks),
            },
        },
        "conjecture_id": conj.get("id"),
        "anchor_ids": [],  # simplified
        "scores": [{"id": int(cid), "score": float(score)} for cid, score in ranking_items],
    }
    save_json(result, args.out)
    topk = min(20, len(ranking_items))
    print(f"Saved {len(ranking_items)} scores to {args.out}. Top {topk}:")
    for i, (cid, sc) in enumerate(ranking_items[:topk], 1):
        print(f"  {i:>2}. id={cid}  score={sc:.3f}")

if __name__ == "__main__":
    main()