#!/usr/bin/env python3
"""
Unified clause preprocessing + heuristic ranking tool.

功能概述 (Overview)
===================
本脚本将 `process_iprover_v3.py` (子句预处理/规范化) 与 `batch_ranker.py` (打分逻辑)
的核心思路合并，形成一个“一步到位”的流水：

1. 读取 iProver 交互模式原始日志 (NUL 分隔 JSON)，定位 `register_clauses` 与最新一次
   `scores_req`（或全部合并）。
2. 清洗每个子句文本：去掉 file(...)/inference(...) 溯源后缀；抽取 tcf(...) 内公式。
3. 两阶段规范化 (canonicalisation)：
   - 识别猜想 (conj_dist == 0) 子句，用于后来 semantic tags。
   - 建立全局符号映射：谓词 -> P#, 函子 -> F#, 常量 -> C#；变量按子句局部次序编号 V0,V1,...
     (扫描策略为简化启发式，并非完整 TPTP 解析器，但对常见 iProver 产生式足够稳健)。
4. 构造数据集结构：conjecture / context / candidate 子句集合（可由参数控制大小）。
5. 语义标注 (semantic tags)：
   - unit / horn / eq / resolvable / touches_target_functor / eq_of_target_functor /
     first_arg_in_goal / shares_goal_consts:k / goal_pred_overlap=n 等。
6. 启发式打分：把每个候选映射到 0–50 的 raw 分，然后做全局 min‑max 归一化生成 0..1 分数。
7. 产物：
   - canonical_dataset.json  (全量 canonical 表示)
   - scoring_details.json    (每个子句 tags + raw_score + reason + norm_score)
   - out_scores.json         (供 iProver EA 使用的最终 {id, score} 列表)
   - process.log             (详细流水日志，可选控制台回显)

设计原则
========
尽量保持纯 Python 依赖，只使用标准库；便于在受限环境中直接运行。

限制 / 说明
===========
1. 提供的 `process_iprover_v3.py` 片段中有大量未完成的占位，本实现采用自包含简化解析逻辑。
2. 不是严格的 TPTP 解析（不处理全部语法角落），但保留常用算子 ~,|,&,=,!= 以及量词外观。
3. 规范化策略：
   - 顶层文字拆分：按 "|" 分割 (括号深度 0)；
   - 谓词头 token 认定：每个文字去除前导 '~' 后首个标识符（跟随 '('）视为谓词；
   - 函子：出现在任意参数内部且跟 '(' 的非谓词头符号；
   - 常量：未跟 '(' 的小写 / 数字混合 / 或首字母小写符号；
   - 变量：首字母大写且可能带 :$i 等类型后缀。
4. equality 直接依赖于字符串中出现 '=' 或 '!=', tags 中表现为 eq。

使用示例
========
python preprocess_and_rank.py \
  --raw-log iprover_raw.log \
  --out out_scores.json \
  --context-size 128 \
  --candidate-size 256 \
  --log-dir LogsRun --verbose

Dry‑run 不需要，因为本脚本仅使用启发式；但保留参数占位 (--dry-run) 以对齐其它流水脚本接口。
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional

# ---------------------------------------------------------------------------
# Logging utilities
# ---------------------------------------------------------------------------

class Logger:
    def __init__(self, log_path: str, verbose: bool):
        self.log_path = log_path
        self.verbose = verbose
        os.makedirs(os.path.dirname(log_path) or '.', exist_ok=True)
        self._fh = open(log_path, 'a', encoding='utf-8')

    def _ts(self) -> str:
        return time.strftime('%H:%M:%S')

    def log(self, msg: str):
        line = f"[{self._ts()}] {msg}"
        self._fh.write(line + '\n')
        self._fh.flush()
        if self.verbose:
            print(line, flush=True)

    def close(self):
        try:
            self._fh.close()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Raw clause preprocessing (adapted / simplified)
# ---------------------------------------------------------------------------

PROVENANCE_SPLIT_PATTERNS = ['file(', 'inference(']

def preprocess_clause_str(clause_str: str) -> str:
    """Remove trailing provenance suffix (file(...)/inference(...)), keep tcf wrapper."""
    for pat in PROVENANCE_SPLIT_PATTERNS:
        if pat in clause_str:
            clause_str = clause_str.split(pat, 1)[0]
            break
    clause_str = clause_str.rstrip()
    if clause_str.endswith(','):
        clause_str = clause_str[:-1].rstrip()
    return clause_str

TCF_RE = re.compile(r"\s*tcf\s*\(", re.IGNORECASE)

def extract_formula_from_tcf(clause_str: str) -> str:
    m = TCF_RE.match(clause_str)
    if not m:
        return clause_str
    body = clause_str[m.end():]
    depth = 0
    comma_count = 0
    start_idx: Optional[int] = None
    for idx, ch in enumerate(body):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        elif ch == ',' and depth == 0:
            comma_count += 1
            if comma_count == 2:
                start_idx = idx + 1
                break
    if start_idx is None:
        return clause_str
    formula = body[start_idx:].strip()
    if formula.endswith('.'):
        formula = formula[:-1].strip()
    # strip balanced outer parens
    def strip_parens(s: str) -> str:
        while s.startswith('(') and s.endswith(')'):
            d = 0; ok = True
            for i, c in enumerate(s):
                if c == '(':
                    d += 1
                elif c == ')':
                    d -= 1
                    if d < 0:
                        ok = False; break
                if i < len(s) - 1 and d == 0:
                    ok = False; break
            if ok and d == 0:
                s = s[1:-1].strip()
            else:
                break
        return s
    return strip_parens(formula)


# ---------------------------------------------------------------------------
# Canonicalisation
# ---------------------------------------------------------------------------

TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_:$]*")

@dataclass
class CanonicalMaps:
    pred_map: Dict[str, str] = field(default_factory=dict)
    fun_map: Dict[str, str] = field(default_factory=dict)
    const_map: Dict[str, str] = field(default_factory=dict)
    pred_counter: int = 0
    fun_counter: int = 0
    const_counter: int = 0

    def predicate(self, name: str) -> str:
        if name not in self.pred_map:
            self.pred_counter += 1
            self.pred_map[name] = f"P{self.pred_counter}"
        return self.pred_map[name]

    def functor(self, name: str) -> str:
        if name not in self.fun_map:
            self.fun_counter += 1
            self.fun_map[name] = f"F{self.fun_counter}"
        return self.fun_map[name]

    def constant(self, name: str) -> str:
        if name not in self.const_map:
            self.const_counter += 1
            self.const_map[name] = f"C{self.const_counter}"
        return self.const_map[name]


def split_top_level_disj(formula: str) -> List[str]:
    parts: List[str] = []
    depth = 0
    start = 0
    for i, ch in enumerate(formula):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth = max(0, depth - 1)
        elif ch == '|' and depth == 0:
            parts.append(formula[start:i].strip())
            start = i + 1
    parts.append(formula[start:].strip())
    return [p for p in parts if p]


def identify_predicate_heads(lits: List[str]) -> set:
    heads = set()
    for lit in lits:
        lit2 = lit.lstrip()
        neg = lit2.startswith('~')
        if neg:
            lit2 = lit2[1:].lstrip()
        m = TOKEN_RE.match(lit2)
        if m and m.end() < len(lit2) and lit2[m.end()] == '(':
            heads.add(m.group(0))
    return heads


def canonicalise_formula(formula: str, cmap: CanonicalMaps) -> Tuple[str, Dict[str, str], Dict[str, set]]:
    """Return canonical formula + variable mapping + local symbol usage sets.

    简化策略：
      1) 先标出顶层文字的谓词头 -> P#；
      2) 再次扫描整句：变量 (首字母大写) -> V#；若 token 在谓词头集合，替换为对应 P#；
         若后随 '(' -> F# (函子)；否则 -> C# (常量)。
      3) local_symbols: 记录使用到的 canonical 名 -> 原名集合。
    """
    lits = split_top_level_disj(formula)
    pred_heads = identify_predicate_heads(lits)
    var_map: Dict[str, str] = {}
    var_counter = 0
    local_syms: Dict[str, set] = {}

    def reg_local(canon: str, orig: str):
        local_syms.setdefault(canon, set()).add(orig)

    out_parts: List[str] = []
    i = 0
    n = len(formula)
    while i < n:
        ch = formula[i]
        if not ch.isalpha() and ch not in '_':
            out_parts.append(ch)
            i += 1
            continue
        m = TOKEN_RE.match(formula, i)
        if not m:
            out_parts.append(ch)
            i += 1
            continue
        token = m.group(0)
        base = token.split(':', 1)[0]
        j = m.end()
        # variable?
        if base and base[0].isupper():
            if base not in var_map:
                var_map[base] = f"V{var_counter}"
                var_counter += 1
            canon = var_map[base]
            out_parts.append(canon)
            reg_local(canon, base)
            i = j
            continue
        # predicate head?
        next_non_ws_j = j
        while next_non_ws_j < n and formula[next_non_ws_j].isspace():
            next_non_ws_j += 1
        is_follow_paren = next_non_ws_j < n and formula[next_non_ws_j] == '('
        if token in pred_heads and is_follow_paren:
            canon = cmap.predicate(token)
            out_parts.append(canon)
            reg_local(canon, token)
            i = j
            continue
        # function vs constant
        if is_follow_paren:
            canon = cmap.functor(token)
            out_parts.append(canon)
            reg_local(canon, token)
        else:
            canon = cmap.constant(token)
            out_parts.append(canon)
            reg_local(canon, token)
        i = j
    canonical = ''.join(out_parts)
    return canonical, var_map, local_syms


# ---------------------------------------------------------------------------
# Semantic tagging & heuristic scoring (subset from batch_ranker)
# ---------------------------------------------------------------------------

LIT_SPLIT = re.compile(r"\|")
PRED_TAG_RE = re.compile(r"^(~)?\s*([A-Za-z]*P\d+)(?:/(\d+))?")

def parse_literals(formula: str) -> List[Tuple[str, str]]:
    if not formula:
        return []
    parts = [p.strip() for p in LIT_SPLIT.split(formula)] if '|' in formula else [formula.strip()]
    lits: List[Tuple[str, str]] = []
    for raw in parts:
        if not raw:
            continue
        if '=' in raw and not raw.lstrip().startswith('~'):
            lits.append(("+EQ/2", "EQ/2")); continue
        if '=' in raw and raw.lstrip().startswith('~'):
            lits.append(("-EQ/2", "EQ/2")); continue
        m = PRED_TAG_RE.match(raw)
        if not m:
            m2 = re.search(r"(P\d+)(?:/(\d+))?", raw)
            if m2:
                pred = m2.group(1); ar = m2.group(2) or '?'
                lits.append((f"+{pred}/{ar}", f"{pred}/{ar}"))
            continue
        neg, pred, ar = m.groups(); ar = ar or '?'
        pol = '-' if neg else '+'
        lits.append((f"{pol}{pred}/{ar}", f"{pred}/{ar}"))
    return lits

def extract_goal_frontier(conj_formula: str) -> Dict[str, Any]:
    lits = parse_literals(conj_formula)
    pos = {p for pp, p in lits if pp.startswith('+')}
    neg = {p for pp, p in lits if pp.startswith('-')}
    preds = {p.split('/')[0] for p in (pos | neg)}
    return {'pos': pos, 'neg': neg, 'preds': preds, 'has_eq': ('EQ/2' in pos or 'EQ/2' in neg)}

TARGET_FUN_RE = re.compile(r"\b(F\d+)\s*\(")
CONST_RE = re.compile(r"\bC\d+\b")

def infer_target_info(conjecture_formula: str) -> dict:
    text = conjecture_formula or ''
    funs = set(re.findall(r"\b(F\d+/\d+)\b", text))
    funs |= set(re.findall(r"\b(F\d+)\s*\(", text))
    goal_consts = set(CONST_RE.findall(text))
    first_arg_consts = set()
    for m in re.finditer(r"\b(F\d+)\s*\(\s*(C\d+)\s*,", text):
        first_arg_consts.add(m.group(2))
    return {
        'target_functors': sorted(funs),
        'goal_consts': sorted(goal_consts),
        'first_arg_consts': sorted(first_arg_consts),
    }

def semantic_tags_for_clause(clause: Dict[str, Any], goal: Dict[str, Any], target_info: Dict[str, Any]) -> List[str]:
    s = clause.get('canonical_formula', '')
    feats = clause.get('features', {}) or {}
    unit = '|' not in s
    has_eq = ('=' in s) or ('!=' in s) or ('EQ/2' in s)
    lits = parse_literals(s)
    pos = {p for pp, p in lits if pp.startswith('+')}
    neg = {p for pp, p in lits if pp.startswith('-')}
    resolvable = []
    for p in pos:
        if p in goal.get('neg', set()):
            resolvable.append(p)
    for p in neg:
        if p in goal.get('pos', set()):
            resolvable.append(p)
    tags: List[str] = []
    if unit: tags.append('unit')
    if feats.get('horn'): tags.append('horn')
    if has_eq: tags.append('eq')
    if resolvable:
        tags.append('resolvable:' + ','.join(sorted(set(resolvable))[:3]))
    # target related
    tfs = set(target_info.get('target_functors', []))
    firsts = set(target_info.get('first_arg_consts', []))
    gconsts = set(target_info.get('goal_consts', []))
    if any((tf in s) or re.search(rf"\b{re.escape(tf.split('/')[0])}\s*\(", s) for tf in tfs):
        tags.append('touches_target_functor')
    if re.search(r"F\d+\s*\([^)]*\)\s*(!=|=)\s*F\d+\s*\(", s):
        tags.append('eq_of_target_functor')
    if any(re.search(rf"\bF\d+\s*\(\s*{re.escape(c)}\s*,", s) for c in firsts):
        tags.append('first_arg_in_goal')
    shared = sum(len(re.findall(rf"\b{re.escape(c)}\b", s)) for c in gconsts)
    tags.append(f'shares_goal_consts:{shared}')
    ovlp = len({p.split('/')[0] for p in (pos|neg)} & goal.get('preds', set()))
    tags.append(f'goal_pred_overlap={ovlp}')
    return tags

def score_clause_heuristic(tags: List[str], formula: str) -> Tuple[float, str]:
    tags_set = set(tags)
    # extract k
    k = 0
    for t in tags:
        if t.startswith('shares_goal_consts:'):
            try: k = int(t.split(':',1)[1])
            except Exception: k = 0
    commas = formula.count(',')
    parens = formula.count('(') + formula.count(')')
    symbols = len(re.findall(r"\b[FPC]\d+\b|[=]|neq|EQ", formula))
    weight = commas + parens + symbols
    weight_norm = int((weight + 5)//6)
    depth = 0; cur=0
    maxd=0
    for ch in formula:
        if ch=='(':
            cur+=1; maxd=max(maxd,cur)
        elif ch==')' and cur>0:
            cur-=1
    depth = maxd
    S_pos = 0; why = ''
    if 'eq_of_target_functor' in tags_set:
        S_pos += 32; why = why or '含F等式'
    if (('EQ' in formula) or ('=' in formula) or ('!=' in formula)) and ('touches_target_functor' in tags_set):
        S_pos += 10; why = why or '一跳重写'
    if 'first_arg_in_goal' in tags_set:
        S_pos += 8; why = why or '首参对齐'
    if 'touches_target_functor' in tags_set:
        S_pos += 12; why = why or '含F'
    if k >= 2:
        S_pos += 10; why = why or '同投影桥'
    if k > 0:
        S_pos += min(8, 2*k); why = why or '含目标常量'
    if any(t.startswith('goal_pred_overlap=1') for t in tags_set):
        S_pos += 6; why = why or '谓词重合'
    if 'unit' in tags_set:
        S_pos += 3; why = why or 'unit'
    if 'horn' in tags_set:
        S_pos += 1; why = why or 'horn'
    S_pos += max(0, 8 - weight_norm)
    if not why and S_pos>0:
        why = '轻量'
    # penalties
    S_pen = 0
    touches_or_eq = ('touches_target_functor' in tags_set) or ('eq_of_target_functor' in tags_set)
    has_eq = ('=' in formula) or ('!=' in formula) or ('EQ' in formula)
    if (not touches_or_eq) and (not has_eq) and k==0:
        S_pen += 12; why = why or '无桥'
    if weight_norm > 10:
        S_pen += (weight_norm - 10); why = why or '超重'
    if depth > 3:
        S_pen += (depth - 3)*2; why = why or '过深'
    var_cnt = len(re.findall(r"\bV\d+\b", formula))
    const_cnt = len(re.findall(r"\bC\d+\b", formula))
    if (var_cnt >= 2*(const_cnt+1)) and (not touches_or_eq) and k==0:
        S_pen += 4; why = why or '变量过多'
    if re.search(r"\bEQ\s*\(\s*([A-Za-z0-9_]+)\s*,\s*\1\s*\)", formula):
        S_pen += 3; why = why or '自等冗余'
    raw = max(0.0, min(50.0, float(S_pos - S_pen)))
    if raw == 0.0:
        h = 0
        for ch in formula:
            h = (h*33 + ord(ch)) & 0xffffffff
        raw = (h % 1000)/10000.0
    return raw, why or 'other'


# ---------------------------------------------------------------------------
# Dataset building
# ---------------------------------------------------------------------------

def parse_raw_log(path: str, logger: Logger) -> Tuple[Dict[int, Dict], List[int]]:
    with open(path, 'rb') as f:
        segs = [s for s in f.read().split(b'\x00') if s.strip()]
    msgs = []
    for idx, seg in enumerate(segs):
        try:
            obj = json.loads(seg.decode('utf-8'))
            msgs.append(obj)
        except Exception as e:
            logger.log(f"[WARN] segment {idx} JSON decode failed: {e}")
    clauses: Dict[int, Dict] = {}
    scores_req_ids: List[int] = []
    for obj in msgs:
        tag = obj.get('tag')
        if tag == 'register_clauses':
            for entry in obj.get('clauses', []):
                cid = int(entry['clause_id'])
                raw = entry['clause']
                feats = entry.get('clause_features', {}) or {}
                clean = preprocess_clause_str(raw)
                formula = extract_formula_from_tcf(clean)
                clauses[cid] = {
                    'id': cid,
                    'raw_clause': raw,
                    'clean_clause': clean,
                    'formula': formula,
                    'features': feats,
                }
        elif tag == 'scores_req':
            scores_req_ids = [int(i) for i in obj.get('clause_ids', [])]
    if not clauses:
        raise RuntimeError('No register_clauses found in log')
    if not scores_req_ids:
        logger.log('[INFO] No scores_req found; using all clause IDs as candidates (excluding conjecture).')
        scores_req_ids = sorted(clauses.keys())
    return clauses, scores_req_ids


def build_dataset(clauses: Dict[int, Dict], candidate_ids: List[int], context_size: int, candidate_size: int, logger: Logger) -> Dict[str, Any]:
    cmap = CanonicalMaps()
    # Identify conjecture (first conj_dist == 0)
    conjecture_id: Optional[int] = None
    for cid, info in clauses.items():
        if info.get('features', {}).get('conj_dist') == 0:
            conjecture_id = cid
            break
    # context selection (in registration order) excluding candidates + conjecture
    reg_order = list(clauses.keys())
    cand_set = set(candidate_ids)
    context: List[int] = []
    for cid in reg_order:
        if cid == conjecture_id:
            continue
        if cid in cand_set:
            continue
        context.append(cid)
        if len(context) >= context_size:
            break
    # candidate list truncated
    candidates_final: List[int] = []
    for cid in candidate_ids:
        if cid == conjecture_id:
            continue
        if cid not in clauses:
            continue
        candidates_final.append(cid)
        if len(candidates_final) >= candidate_size:
            break
    # canonicalise all needed clauses (conjecture + context + candidates)
    needed_ids = set(context + candidates_final)
    if conjecture_id is not None:
        needed_ids.add(conjecture_id)
    logger.log(f"[CANON] total needed clauses: {len(needed_ids)} (context={len(context)}, candidates={len(candidates_final)}, conjecture={'yes' if conjecture_id else 'no'})")
    for cid in needed_ids:
        entry = clauses[cid]
        cformula, var_map, local_syms = canonicalise_formula(entry['formula'], cmap)
        entry['canonical_formula'] = cformula
        entry['variable_mapping'] = var_map
        entry['local_symbols'] = {k: sorted(list(v)) for k, v in local_syms.items()}
    # build symbol_map
    symbol_map = {}
    for name, canon in cmap.pred_map.items():
        symbol_map[f"{canon}/?"] = {'original': [name], 'kind': 'predicate', 'arity': '?'}
    for name, canon in cmap.fun_map.items():
        symbol_map[f"{canon}/?"] = {'original': [name], 'kind': 'function', 'arity': '?'}
    for name, canon in cmap.const_map.items():
        symbol_map[canon] = {'original': [name], 'kind': 'constant', 'arity': 0}
    conj_repr = None
    if conjecture_id is not None:
        conj_repr = {k: clauses[conjecture_id][k] for k in ('id','canonical_formula','formula','features','variable_mapping','local_symbols') if k in clauses[conjecture_id]}
    dataset = {
        'metadata': {
            'schema_version': 'u1',
            'timestamp': int(time.time()),
            'context_size': context_size,
            'candidate_size': candidate_size,
        },
        'symbol_map': symbol_map,
        'conjecture': conj_repr,
        'context_clauses': [clauses[cid] for cid in context],
        'candidate_clauses': [clauses[cid] for cid in candidates_final],
    }
    return dataset


# ---------------------------------------------------------------------------
# Scoring pipeline
# ---------------------------------------------------------------------------

def score_dataset(ds: Dict[str, Any], logger: Logger) -> Tuple[Dict[int, float], List[Dict[str, Any]]]:
    conj = ds.get('conjecture') or {}
    conj_formula = conj.get('canonical_formula', '')
    goal = extract_goal_frontier(conj_formula)
    target_info = infer_target_info(conj_formula)
    cands = ds.get('candidate_clauses', []) or []
    details: List[Dict[str, Any]] = []
    raw_scores: Dict[int, float] = {}
    logger.log(f"[SCORE] candidates={len(cands)}")
    for c in cands:
        tags = semantic_tags_for_clause(c, goal, target_info)
        c['tags'] = tags
        raw, reason = score_clause_heuristic(tags, c.get('canonical_formula',''))
        raw_scores[int(c['id'])] = raw
        details.append({
            'id': int(c['id']),
            'canonical_formula': c.get('canonical_formula',''),
            'tags': tags,
            'raw_score_0_50': raw,
            'reason': reason,
        })
    if not raw_scores:
        return {}, details
    vals = list(raw_scores.values())
    vmin, vmax = min(vals), max(vals)
    if abs(vmax - vmin) < 1e-12:
        logger.log('[SCORE] flat raw scores; injecting jitter by id')
        norm = {}
        for cid, raw in raw_scores.items():
            h = int(cid * 2654435761 & 0xffffffff)
            norm[cid] = (h % 1009)/1009.0
    else:
        norm = {cid: (raw - vmin)/(vmax - vmin) for cid, raw in raw_scores.items()}
    for d in details:
        d['norm_score_0_1'] = norm.get(d['id'], 0.0)
    return norm, details


# ---------------------------------------------------------------------------
# I/O helpers
# ---------------------------------------------------------------------------

def save_json(obj: Any, path: str):
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


# ---------------------------------------------------------------------------
# Main CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[List[str]] = None):
    ap = argparse.ArgumentParser(description='Unified preprocess + heuristic ranker (简化版)')
    ap.add_argument('--raw-log', type=str, required=True, help='Path to NUL-delimited iprover interactive raw log')
    ap.add_argument('--out', type=str, default='out_scores.json', help='Output scores JSON file')
    ap.add_argument('--log-dir', type=str, default='Logs', help='Directory to store logs & intermediates')
    ap.add_argument('--context-size', type=int, default=128)
    ap.add_argument('--candidate-size', type=int, default=256)
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--dry-run', action='store_true', help='(kept for interface symmetry; heuristic is deterministic)')
    args = ap.parse_args(argv)

    ts_dir = os.path.join(args.log_dir, f"UPR.{int(time.time())}")
    os.makedirs(ts_dir, exist_ok=True)
    logger = Logger(os.path.join(ts_dir, 'process.log'), verbose=args.verbose)
    logger.log('=== Unified Preprocess + Rank START ===')
    logger.log(f"Args: {vars(args)}")

    # Step 1 parse raw log
    logger.log('[STEP] parse raw log')
    clauses, candidate_ids = parse_raw_log(args.raw_log, logger)
    logger.log(f"[PARSE] clauses={len(clauses)} last_scores_req_ids={len(candidate_ids)}")

    # Step 2 build dataset
    logger.log('[STEP] build dataset & canonicalise')
    ds = build_dataset(clauses, candidate_ids, args.context_size, args.candidate_size, logger)
    dataset_path = os.path.join(ts_dir, 'canonical_dataset.json')
    save_json(ds, dataset_path)
    logger.log(f"[CANON] dataset saved -> {dataset_path}")

    # Step 3 scoring
    logger.log('[STEP] scoring')
    norm_scores, details = score_dataset(ds, logger)
    details_path = os.path.join(ts_dir, 'scoring_details.json')
    save_json(details, details_path)
    logger.log(f"[SCORE] details saved -> {details_path}")

    # Step 4 final out
    scores_arr = [{'id': int(cid), 'score': float(sc)} for cid, sc in sorted(norm_scores.items(), key=lambda kv: kv[1], reverse=True)]
    out_obj = {'scores': scores_arr, 'meta': {'timestamp': int(time.time()), 'dataset_path': dataset_path, 'details_path': details_path}}
    save_json(out_obj, args.out)
    logger.log(f"[OUT] wrote {len(scores_arr)} scores -> {args.out}")
    if scores_arr:
        topk = min(10, len(scores_arr))
        logger.log('[TOP] top scores:')
        for i, (cid, sc) in enumerate(scores_arr[:topk], 1):
            logger.log(f"  {i:02d}. id={cid} score={sc:.4f}")
    logger.log('=== DONE ===')
    logger.close()


if __name__ == '__main__':  # pragma: no cover
    main()
