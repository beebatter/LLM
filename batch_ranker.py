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

def semantic_tags_for_clause(c: Dict[str, Any], goal: Dict[str, Any]) -> List[str]:
    f = c.get("features", {})
    s = c.get("canonical_formula", "")
    unit = ('|' not in s)
    has_eq = ("=" in s) or ("EQ/2" in s)
    pos, neg, preds = _candidate_pred_set(s)
    # resolvable: opposite polarity predicate exists in goal frontier
    resolvable = []
    for p in pos:
        if p in goal.get("neg", set()):
            resolvable.append(p)
    for p in neg:
        if p in goal.get("pos", set()):
            resolvable.append(p)
    tags: List[str] = []
    if unit:
        tags.append("unit")
    if has_eq:
        tags.append("eq")
    if f.get("horn"):
        tags.append("horn")
    if resolvable:
        # keep at most 3 for brevity
        short = ','.join(sorted(set(resolvable))[:3])
        tags.append(f"resolvable:{short}")
    # SAT metrics
    if c.get("_sat_support") is not None:
        tags.append(f"sat_support={c['_sat_support']:.2f}")
    if c.get("_sat_pressure") is not None:
        tags.append(f"sat_pressure={c['_sat_pressure']:.2f}")
    # overlap with goal predicates (symbolic, polarity-agnostic) just for info, not scoring
    ovlp = len(preds & goal.get("preds", set()))
    tags.append(f"goal_pred_overlap={ovlp}")
    return tags

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
            return {}
    return {}

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

def build_summary_prompt(conjecture_formula: str, cheatsheet: str, ctx_list: List[Dict[str, Any]], max_tokens: int) -> str:
    ctx_text = "\n".join(f"- ID {c['id']}: {c['canonical_formula']}" for c in ctx_list[:80])
    return f"""
    你是自动定理证明（ATP）助手。请基于“命名不变”的公式写一段**精炼的背景摘要**（<= {max_tokens} tokens）：
    - 使用下面的“符号速查表”理解 P#/F#/C# 的原名语义；
    - 保持命名不变，不替换 P#/F#/C#；
    - 摘要重点：与猜想直接相关的关系/构造、常见模式、明显的蕴含或等价线索、易混淆的符号区别；
    - 覆盖**等式类与非等式类**两种模式，并考虑**不同谓词元数**之间可组合的推理链；
    - 列出 6–10 条要点；避免复述每条子句；不要输出任何评分。

    【猜想】:
    {conjecture_formula}

    【符号速查表】:
    {cheatsheet}

    【上下文(节选)】:
    {ctx_text}

    只输出摘要正文，不要额外说明。
    """.strip()

def build_scoring_prompt(summary: str, cheatsheet: str, conjecture_formula: str, chunk: List[Dict[str, Any]], goal_text: str, rules_text: str) -> str:
    def _line(c: Dict[str, Any]) -> str:
        tags = c.get("_sem_tags", [])
        tag_str = ", ".join(tags) if tags else ""
        return f"- ID {c['id']} | tags: [{tag_str}]\n  formula: {c['canonical_formula']}"
    lines = "\n".join(_line(c) for c in chunk)
    return f"""
你是 ATP 子句打分器。请基于**推理可用性**而非“频度/重合度/长度”对候选打分。
- **分值范围**：0–10 的实数，至少保留两位小数；
- **评分依据（只允许这些）**：
  1) 与【目标前沿】的可解性（resolvable:*），是否能一跳解析/关闭；
  2) 等式可重写潜力（eq + rewrite 直觉）；
  3) Horn 规则可实例化潜力（horn）；
  4) SAT 观测（sat_support / sat_pressure）仅作**微调**，不得喧宾夺主；
- **硬性规则（Bridge 约束）**：
  A) 若候选至少一侧可与猜想目标项**直接统一**或能被等式**一跳重写**到目标式样，显著加分；
  B) 若候选仅为 `F*/... =/!= C*` 这类“等于常量/不等于常量”，且看不到把目标项一跳桥接到该常量的规则链，则强烈降权（此类通常偏题）；
  C) 等式方向优先：能让目标项**复杂→简单**、或消顶符号/合一参数者优先。
- **分档指引（用于拉开分数）**：
  9–10：可与目标一侧直接互补/解析，或一跳重写/实例化即可推进；
  7–8：Horn 结论可统一目标（前提短），或具备多个有效重写位点；
  4–6：间接相关（需≥2步桥接）或弱重写；
  0–3：与目标两侧**无桥接**，或纯 `...=/!=常量` 且无可见桥接。
- **并列打破顺序**（仅当分值恰好相同）：
  1) unit 且 resolvable 优先；
  2) 文字数（lit_count）更少者优先；
  3) max_func_arity 更低者优先；
  4) Horn 优先；EPR 优先；
  5) canonical_formula 更短者优先。

【目标前沿（抽象）】
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
""".strip()

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
    anchor_ids = {a["id"] for a in anchors}
    # Exclude anchors from the main pool to avoid duplicate scoring within a chunk
    pool = [c for c in cands if c["id"] not in anchor_ids]

    # Split pool into payload-sized chunks
    chunks: List[List[Dict[str, Any]]] = [pool[i:i+chunk_payload_size] for i in range(0, len(pool), chunk_payload_size)]

    def _append_anchors(cap: int, items: List[Dict[str, Any]]):
        """Append anchors to items up to cap, without duplicate IDs."""
        used = {x["id"] for x in items}
        for a in anchors:
            if len(items) >= cap:
                break
            if a["id"] in used:
                continue
            items.append(a)
            used.add(a["id"])

    # Normal case: extend each chunk with anchors (dedup, respect size)
    for ch in chunks:
        _append_anchors(chunk_payload_size, ch)

    # Edge case: if no chunks were created (e.g., pool empty or all candidates selected as anchors),
    # build a single chunk from candidates and ensure anchors are included, respecting the size limit.
    if not chunks:
        base = list(pool) if pool else list(cands)
        base = base[:chunk_payload_size]
        _append_anchors(chunk_payload_size, base)
        # If still empty (no candidates), but anchors exist, fall back to anchors-only chunk
        if not base and anchors:
            base = anchors[:min(len(anchors), chunk_payload_size)]
        # Only create a chunk if we have something to score
        if base:
            chunks = [base]

    return chunks

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
                payload = {"scores": [{"id": i, "score": round(random.random(), 3)} for i in ids]}
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
        obj = extract_json(text)
        if not obj or "scores" not in obj:
            ids = [c["id"] for c in chunk]
            obj = {"scores": [{"id": i, "score": 0.0} for i in ids]}
        return obj

# ---------------------- Normalization ----------------------

def normalize_and_align(chunks_json: List[Dict[str, Any]], anchor_ids: List[int]) -> Dict[int, float]:
    # 1) per-chunk z-score normalize
    per_chunk: List[Dict[int, float]] = []
    for js in chunks_json:
        arr = js.get("scores", [])
        vals = [float(x.get("score", 0.0)) for x in arr]
        mu = mean(vals) if vals else 0.0
        sd = pstdev(vals) if len(vals) > 1 else 1.0
        if sd == 0: sd = 1.0
        per_chunk.append({int(x["id"]): (float(x["score"]) - mu) / sd for x in arr})
    # 2) global anchor target (average across chunks)
    anchor_vals: Dict[int, List[float]] = defaultdict(list)
    for norm in per_chunk:
        for aid in anchor_ids:
            if aid in norm:
                anchor_vals[aid].append(norm[aid])
    global_anchor = {aid: (sum(v)/len(v) if v else 0.0) for aid, v in anchor_vals.items()}
    # 3) align each chunk by linear fit a*s+b using anchors
    aligned: List[Dict[int, float]] = []
    for norm in per_chunk:
        xs, ys = [], []
        for aid in anchor_ids:
            if aid in norm and aid in global_anchor:
                xs.append(norm[aid]); ys.append(global_anchor[aid])
        a, b = 1.0, 0.0
        if len(xs) >= 2:
            xbar = sum(xs)/len(xs); ybar = sum(ys)/len(ys)
            num = sum((x - xbar)*(y - ybar) for x, y in zip(xs, ys))
            den = sum((x - xbar)**2 for x in xs) or 1.0
            a = num/den
            b = ybar - a * xbar
        elif len(xs) == 1:
            b = ys[0] - xs[0]
        aligned.append({cid: a * v + b for cid, v in norm.items()})
    # 4) aggregate by mean across chunks
    bag: Dict[int, List[float]] = defaultdict(list)
    for al in aligned:
        for cid, v in al.items():
            bag[cid].append(v)
    return {cid: sum(v)/len(v) for cid, v in bag.items()}


# ---------------------- Calibration & Jitter ----------------------

def _fun_keys(local_symbols: Dict[str, Any]) -> set:
    if not isinstance(local_symbols, dict):
        return set()
    return {k for k in local_symbols.keys() if isinstance(k, str) and k.startswith("F")}

def _pred_overlap_count(formula: str, goal: Dict[str, Any]) -> int:
    pos, neg, preds = _candidate_pred_set(formula or "")
    return len(preds & goal.get("preds", set()))

def calibrate_scores(
    raw_scores: Dict[int, float],
    id2clause: Dict[int, Dict[str, Any]],
    goal: Dict[str, Any],
    conj_fun_syms: set,
    scale_hi: float = 100.0,
    penalty_offtarget_eq: float = 0.2,
    bump_resolvable: float = 0.6,
    bump_unit: float = 0.4,
    bump_horn: float = 0.3,
    bump_fun_overlap: float = 0.2,
) -> Tuple[Dict[int, float], Dict[str, Any]]:
    """Min-max scale to [0,scale_hi], penalize off-target eq-only clauses with no bridge,
    and add small positive bumps for semantically useful tags. Return (scaled_scores, stats)."""
    if not raw_scores:
        return {}, {"note": "empty"}

    vals = list(raw_scores.values())
    pre_min = min(vals); pre_max = max(vals)
    pre_mean = sum(vals) / len(vals)

    # Min-max scaling
    scaled: Dict[int, float] = {}
    if pre_max - pre_min > 1e-12:
        for cid, v in raw_scores.items():
            scaled[cid] = (v - pre_min) / (pre_max - pre_min) * scale_hi
    else:
        # Flat case: start everyone from the midpoint; later jitter will separate.
        mid = 0.5 * scale_hi
        scaled = {cid: mid for cid in raw_scores.keys()}

    penalized = 0
    bumped = 0

    for cid, s in list(scaled.items()):
        c = id2clause.get(cid, {})
        formula = c.get("canonical_formula", "") or ""
        tags = c.get("_sem_tags", []) or []
        loc = c.get("local_symbols", {}) or {}

        has_eq = ("=" in formula)
        fun_ovlp = len(_fun_keys(loc) & conj_fun_syms)
        pred_ovlp = _pred_overlap_count(formula, goal)

        # Off-target eq: equality present, but neither goal predicates nor conjecture functions overlap
        if has_eq and fun_ovlp == 0 and pred_ovlp == 0:
            s *= penalty_offtarget_eq
            penalized += 1

        # Positive micro-bumps (small; only for ordering and to help external_score distinguish)
        bump = 0.0
        if any(isinstance(t, str) and t.startswith("resolvable:") for t in tags):
            bump += bump_resolvable
        if "unit" in tags:
            bump += bump_unit
        if "horn" in tags:
            bump += bump_horn
        if fun_ovlp > 0:
            bump += bump_fun_overlap
        if bump:
            s += bump
            bumped += 1

        scaled[cid] = float(s)

    post_vals = list(scaled.values())
    post_min = min(post_vals); post_max = max(post_vals)
    post_mean = sum(post_vals) / len(post_vals)

    stats = {
        "pre_min": pre_min, "pre_max": pre_max, "pre_mean": pre_mean,
        "post_min": post_min, "post_max": post_max, "post_mean": post_mean,
        "penalized_offtarget_eq": penalized, "bumped_semantic": bumped,
        "scale_hi": scale_hi,
    }
    return scaled, stats

def apply_groupwise_jitter(sorted_items: List[Tuple[int, float]], eps: float = 1e-6) -> List[Tuple[int, float]]:
    """Within groups of (nearly) equal scores, add decreasing tiny jitter so numbers are strictly ordered.
    `sorted_items` must already be in the desired final order (after tie-break rules)."""
    if not sorted_items:
        return sorted_items
    out: List[Tuple[int, float]] = []
    i = 0
    n = len(sorted_items)
    while i < n:
        j = i + 1
        base = sorted_items[i][1]
        # find group with almost equal scores
        while j < n and abs(sorted_items[j][1] - base) <= 1e-12:
            j += 1
        # apply jitter within [i, j)
        group = sorted_items[i:j]
        g = len(group)
        for k, (cid, sc) in enumerate(group):
            # earlier items (better by tie-break) get slightly higher jitter
            jittered = sc + eps * (g - k)
            out.append((cid, jittered))
        i = j
    return out

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
    goal = extract_goal_frontier(conj_formula)
    goal_text = format_goal_frontier_text(goal)
    rules_text = build_reasoning_rules_text()

    # attach SAT metrics (if any) and compute semantic tags
    attach_sat_metrics(cands, sat_map)
    compute_and_attach_sem_tags(cands, goal)

    # Provide reasoning context to the client
    llm.set_reasoning_context(goal_text, rules_text)

    # anchors: semantic anchor selection with fallback
    # Choose anchors conservatively for small candidate sets to avoid the "all anchors, empty pool" degenerate case.
    # Effective anchors: at most len(cands)-1 so there's always some non-anchor payload (when possible).
    requested_A = max(0, int(args.anchors))
    A = min(requested_A, max(0, len(cands) - 1))
    anchors = select_anchors_semantic(cands, A, goal)
    if len(anchors) < A:
        # fallback to previous overlap-based anchors to fill remaining slots
        rest = [c for c in cands if c not in anchors]
        extra = sorted(rest, key=lambda c: (-c.get("_ovlp", 0.0), len(c.get("canonical_formula", ""))))[:(A - len(anchors))]
        anchors = anchors + extra
    anchor_ids = [a["id"] for a in anchors]

    # chunking
    chunks = make_chunks(cands, args.chunk_size, anchors)
    if args.verbose:
        pool_size = sum(1 for c in cands if c["id"] not in set(anchor_ids))
        print(f"[DEBUG] anchors={len(anchor_ids)} pool={pool_size} chunks={len(chunks)}")
        if chunks:
            print(f"[DEBUG] chunk[0] size={len(chunks[0])} (payload+anchors)")

    # per-chunk scoring
    chunk_dir = os.path.join(out_dir, "chunks")
    os.makedirs(chunk_dir, exist_ok=True)

    chunk_results: List[Dict[str, Any]] = []
    for idx, ch in enumerate(chunks):
        # Choose model based on chunk size
        use_small = len(ch) < SMALL_BATCH_THRESHOLD
        client = llm
        if use_small:
            if llm_small is None:
                llm_small = LLMClient(model=SMALL_BATCH_MODEL, temperature=args.temperature, dry_run=args.dry_run, max_retries=args.max_retries, verbose=(args.progress or args.verbose))
                # ensure it has the same reasoning context
                # (set below once goal_text/rules_text are available)
            client = llm_small
        if args.progress:
            mdl = getattr(client, "model", "?")
            print(f"[LLM] Scoring chunk {idx+1}/{len(chunks)} (size={len(ch)}) with model={mdl}...")
        prompt = build_scoring_prompt(summary, cheatsheet, conj_formula, ch, goal_text, rules_text)
        with open(os.path.join(chunk_dir, f"prompt_{idx:03d}.txt"), "w", encoding="utf-8") as pf:
            pf.write(prompt)
        if args.verbose:
            ids_in_prompt = [int(m) for m in re.findall(r"ID\s+(\d+)", prompt)]
            print(f"[DEBUG] chunk_{idx:03d}: ids_in_prompt={len(ids_in_prompt)}")
        # Call LLM and persist raw response alongside parsed JSON
        # Ensure reasoning context is present on chosen client
        client.set_reasoning_context(goal_text, rules_text)
        text = ""
        try:
            text = client._chat(prompt)
        except Exception as e:
            if use_small:
                if args.progress:
                    print(f"[LLM] small-batch model failed: {e}. Falling back to primary model {llm.model}.")
                try:
                    client = llm
                    client.set_reasoning_context(goal_text, rules_text)
                    text = client._chat(prompt)
                except Exception as e2:
                    if args.progress:
                        print(f"[LLM] primary model also failed: {e2}. Using zero scores for this chunk.")
                    text = ""
            else:
                if args.progress:
                    print(f"[LLM] model failed: {e}. Using zero scores for this chunk.")
                text = ""
        with open(os.path.join(chunk_dir, f"response_{idx:03d}.txt"), "w", encoding="utf-8") as rf:
            rf.write(text)
        res = extract_json(text)
        if not res or "scores" not in res:
            ids = [c["id"] for c in ch]
            res = {"scores": [{"id": i, "score": 0.0} for i in ids]}
        # annotate chunk with model used
        try:
            res_meta = res.setdefault("meta", {})
            res_meta["model"] = getattr(client, "model", args.model)
        except Exception:
            pass
        if args.progress:
            print(f"[LLM]   -> received {len(res.get('scores', []))} scores.")
        save_json(res, os.path.join(chunk_dir, f"chunk_{idx:03d}.json"))
        chunk_results.append(res)
        if args.verbose:
            print(f"[DEBUG] chunk_{idx:03d}: scores={len(res.get('scores', []))}")

    # normalize + align across chunks
    total_scored = sum(len(js.get("scores", [])) for js in chunk_results)
    if args.verbose:
        print(f"[DEBUG] total_scored entries across chunks: {total_scored}")
    if total_scored == 0:
        print("[WARN] No scores produced by chunks. Possible reasons: empty candidates; all candidates selected as anchors; or prompt parse failure. Falling back to anchors-only scores (zeros).")
        fallback_scores = {int(aid): 0.0 for aid in anchor_ids}
        ranking = sorted(fallback_scores.items(), key=lambda kv: kv[1], reverse=True)
        result = {
            "meta": {
                "model": args.model,
                "temperature": args.temperature,
                "chunk_size": args.chunk_size,
                "anchors": args.anchors,
                "context_summary_k": args.context_summary_k,
                "summary_max_tokens": args.summary_max_tokens,
                "dry_run": args.dry_run,
                "timestamp": int(time.time()),
                "debug": {
                    "context": len(ctx),
                    "candidates": len(cands),
                    "anchors": len(anchor_ids),
                    "chunks": len(chunks),
                    "total_scored": total_scored,
                }
            },
            "conjecture_id": conj.get("id"),
            "anchor_ids": anchor_ids,
            "scores": [{"id": int(cid), "score": float(score)} for cid, score in ranking],
        }
        save_json(result, args.out)
        topk = min(20, len(ranking))
        print(f"Saved {len(ranking)} scores to {args.out} (fallback). Top {topk}:")
        for i, (cid, sc) in enumerate(ranking[:topk], 1):
            print(f"  {i:>2}. id={cid}  score={sc:.3f}")
        return

    final_scores = normalize_and_align(chunk_results, anchor_ids)
    # Build id -> clause map (context + candidates) for calibration and tie-breaking
    id2clause: Dict[int, Dict[str, Any]] = {c["id"]: c for c in ctx}
    id2clause.update({c["id"]: c for c in cands})

    # Conjecture-side function symbols for bridge checks
    conj_fun_syms = {k for k in (conj.get("local_symbols", {}) or {}).keys() if isinstance(k, str) and k.startswith("F")}

    # Calibrate scores to a wide numeric range and apply semantic adjustments
    calibrated_scores, calib_stats = calibrate_scores(final_scores, id2clause, goal, conj_fun_syms)
    final_scores = calibrated_scores

    if args.progress or args.verbose:
        print(f"[SCALE] pre(min/mean/max)=({calib_stats['pre_min']:.4f}/{calib_stats['pre_mean']:.4f}/{calib_stats['pre_max']:.4f}) "
              f"-> post(min/mean/max)=({calib_stats['post_min']:.2f}/{calib_stats['post_mean']:.2f}/{calib_stats['post_max']:.2f}); "
              f"penalized(offtarget_eq)={calib_stats['penalized_offtarget_eq']}, bumped={calib_stats['bumped_semantic']}")


    def _tiebreak_tuple(cid: int, score: float):
        c = id2clause.get(cid, {})
        ovlp = 0.0
        if c:
            ovlp = compute_overlap(conj_syms, c)
        f = c.get("features", {}) if c else {}
        conjd = f.get("conj_dist", 10**9)
        hornp = 0 if f.get("horn") else 1
        eprp = 0 if f.get("epr") else 1
        clen = len(c.get("canonical_formula", "")) if c else 10**9
        # Primary sort: score DESC; then: overlap DESC; conj_dist ASC; horn True first; epr True first; shorter clause first
        return (-score, -ovlp, conjd, hornp, eprp, clen)

    ranking_items = [(int(cid), float(sc)) for cid, sc in final_scores.items()]
    ranking_items.sort(key=lambda kv: _tiebreak_tuple(kv[0], kv[1]))
    # Make numeric scores strictly ordered within ties to help iProver's external_score
    ranking_items = apply_groupwise_jitter(ranking_items, eps=1e-6)

    # If scores are degenerate (all equal or near-equal), spread them by rank to avoid all zeros reaching iProver.
    def _spread_if_degenerate(items):
        if not items:
            return items, False
        vals = [v for _, v in items]
        if (max(vals) - min(vals)) < 1e-9:
            n = len(items)
            if n == 1:
                return [(items[0][0], 1.0)], True
            # Map ranks to [1.0 .. 0.0]
            mapped = [(cid, (n - i - 1) / (n - 1)) for i, (cid, _) in enumerate(items)]
            return mapped, True
        return items, False

    ranking, used_spread_fallback = _spread_if_degenerate(ranking_items)
    models_used = sorted({js.get("meta", {}).get("model", args.model) for js in chunk_results})
    result = {
        "meta": {
            "model": args.model,
            "temperature": args.temperature,
            "chunk_size": args.chunk_size,
            "anchors": args.anchors,
            "context_summary_k": args.context_summary_k,
            "summary_max_tokens": args.summary_max_tokens,
            "dry_run": args.dry_run,
            "timestamp": int(time.time()),
            "debug": {
                "context": len(ctx),
                "candidates": len(cands),
                "anchors": len(anchor_ids),
                "chunks": len(chunks),
                "requested_anchors": requested_A,
                "used_spread_fallback": used_spread_fallback,
                "models_used": models_used,
                "calibration": calib_stats,
            },
        },
        "conjecture_id": conj.get("id"),
        "anchor_ids": anchor_ids,
        "scores": [{"id": int(cid), "score": float(score)} for cid, score in ranking],
    }
    save_json(result, args.out)
    # quick view
    topk = min(20, len(ranking))
    print(f"Saved {len(ranking)} scores to {args.out}. Top {topk}:")
    for i, (cid, sc) in enumerate(ranking[:topk], 1):
        print(f"  {i:>2}. id={cid}  score={sc:.3f}")

if __name__ == "__main__":
    main()