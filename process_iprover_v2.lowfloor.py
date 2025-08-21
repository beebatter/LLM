
# --- Prefilter controls (added) ---
PREFILTER_ENABLED = True        # If True, only a subset of candidates is sent to the LLM
LOW_FLOOR_SCORE   = 0.02        # Very low score assigned to filtered-out clauses (0..1 scale)
# ----------------------------------

"""
this is new version of process_iprover.py
Utility script to convert iProver interactive mode logs into a format
suitable for consumption by a Large Language Model (LLM).  The goal
is to hide problem‑specific symbol names behind canonical identifiers
so that an LLM can generalise across different problems.  The script
supports the following high‑level operations:

1. Parse a raw interactive log file (e.g. ``iprover_raw.log``) that
   contains JSON messages from iProver separated by a NUL character.
   Typically, the first message will be a ``register_clauses``
   request describing newly generated clauses and their features,
   followed by a ``scores_req`` message listing clause IDs that
   require scoring.

2. Clean each clause's textual representation by removing trailing
   provenance annotations such as ``file(...)`` or ``inference(...)``.
   This relies on the same heuristic used in the iprover‑gnn server.

3. Extract the logical formula from the cleaned clause.  In iProver
   output, clauses are wrapped in a ``tcf(id,plain,<formula>)``
   constructor.  The ``extract_formula`` function discards the ``tcf``
   wrapper and the clause ID, returning only the underlying formula.

4. Canonicalise symbol names across the entire problem.  All
   variables are renamed to ``V0``, ``V1``, etc. on a per‑clause
   basis.  Non‑variable symbols (predicates, functions and constants)
   are assigned stable canonical names (``P1``, ``P2``...) across
   the entire dataset.  Symbols that never occur with arguments are
   treated as constants; those that do appear followed by a ``(`` are
   treated as function/predicate symbols.  The script records a
   mapping from canonical names back to the original symbol names.

5. Build a structured representation for each clause.  Each entry
   includes the clause ID, the canonicalised formula, the original
   cleaned formula, and any clause features supplied by iProver
   (e.g. ``conj_dist``, ``horn``, ``epr``).  This representation can
   be serialised to JSON for consumption by an LLM.

6. Select a subset of clauses to serve as context and candidates.
   By default, the script selects the first N context clauses from
   ``register_clauses`` and the first M candidate clause IDs from
   the ``scores_req`` message.  The user can adjust these numbers
   via command‑line arguments.  The clause with ``conj_dist == 0``
   (if present) is treated as the conjecture.

Example usage::

    python process_iprover.py \
      --raw-log iprover_raw.log \
      --context-size 128 \
      --candidate-size 128 \
      --output json

This will produce a JSON document containing the symbol mapping,
conjecture, context clauses and candidate clauses.  See the ``main``
function for details.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

# ---------- Lightweight parser to classify symbols (pred/function/const) ----------
from typing import Any

def _kw_skip_ws(s: str, i: int) -> int:
    n = len(s)
    while i < n and s[i].isspace():
        i += 1
    return i

def _kw_parse_name(s: str, i: int):
    n = len(s); j = i
    while j < n and (s[j].isalnum() or s[j] in ['_', '$', ':']):
        j += 1
    return s[i:j], j

def _kw_strip_type(name: str) -> str:
    return name.split(':',1)[0] if ':' in name else name

def _kw_parse_term_kindscan(s: str, i: int, func_set: set) -> int:
    i = _kw_skip_ws(s, i)
    name, j = _kw_parse_name(s, i)
    if not name:
        return i
    base = _kw_strip_type(name)
    j = _kw_skip_ws(s, j)
    if j < len(s) and s[j] == '(':
        depth = 1; args = 1; k = j+1
        while k < len(s):
            c = s[k]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
                if depth == 0:
                    break
            elif c == ',' and depth == 1:
                args += 1
            k += 1
        func_set.add((base, args))
        return k+1 if k < len(s) and s[k] == ')' else j+1
    return j

def _kw_split_top_level_disj(s: str):
    parts=[]; depth=0; start=0
    for i,ch in enumerate(s):
        if ch=='(':
            depth+=1
        elif ch==')':
            depth-=1
        elif ch=='|' and depth==0:
            parts.append(s[start:i].strip()); start=i+1
    parts.append(s[start:].strip())
    return parts

def _strip_outer_parens(s: str) -> str:
    s = s.strip()
    while s.startswith('(') and s.endswith(')'):
        depth = 0
        balanced = True
        for i, c in enumerate(s):
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and i < len(s) - 1:
                balanced = False
                break
        if balanced:
            s = s[1:-1].strip()
        else:
            break
    return s

def _classify_symbols_in_formula(formula: str) -> tuple[set, set]:
    """
    Return (func_terms, pred_heads) where each element is a set of (name, arity).

    func_terms  : symbols that occur in term position (as a function symbol)
    pred_heads  : symbols that occur as the head of an atomic literal (predicate)
    """
    ast = parse_raw_formula_ast(formula)
    return _collect_occurrences_for_kinds(ast)
#
# -------- Minimal AST parser for RAW (non-canonical) formulas + occurrence collection --------

def parse_raw_formula_ast(s: str):
    """
    Parse a *raw* TPTP-like formula (with typed tokens allowed) into an AST compatible
    with `parse_canonical_formula_ast`'s shape, but token regex allows ':' and '$'.
    Nodes:
      - quantifier: { 'quantifier': 'forall'|'exists', 'vars': [...], 'body': ... }
      - op: { 'op': 'or'|'and'|'='|'neq'|'not', 'args': [...] }  (not: uses 'arg')
      - atom: { 'atom': { 'pred': name, 'args': [...] } }
      - leaf terms: variables/constants as strings (we don't need full term nodes)
    """
    import re

    def strip_outer_parens(expr: str) -> str:
        expr = expr.strip()
        while expr.startswith('(') and expr.endswith(')'):
            depth = 0
            balanced = True
            for i, c in enumerate(expr):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(expr) - 1:
                    balanced = False
                    break
            if balanced:
                expr = expr[1:-1].strip()
            else:
                break
        return expr

    s = s.strip()
    # Quantifiers
    if s.startswith('![') or s.startswith('?['):
        quant = 'forall' if s.startswith('![') else 'exists'
        end_idx = s.find(']:')
        if end_idx != -1:
            var_list = s[2:end_idx]
            vars_ = [v.strip() for v in var_list.split(',') if v.strip()]
            body = s[end_idx + 2:].strip()
            return {'quantifier': quant, 'vars': vars_, 'body': parse_raw_formula_ast(body)}

    s = strip_outer_parens(s)

    def split_top_level(expr: str, op_char: str) -> list:
        parts = []
        depth = 0
        last = 0
        i = 0
        n = len(expr)
        while i < n:
            c = expr[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == op_char and depth == 0:
                parts.append(expr[last:i])
                last = i + 1
            i += 1
        if parts:
            parts.append(expr[last:].strip())
            return [p.strip() for p in parts]
        else:
            return [expr]

    # Disjunction
    parts = split_top_level(s, '|')
    if len(parts) > 1:
        return {'op': 'or', 'args': [parse_raw_formula_ast(p) for p in parts]}

    # Conjunction
    parts = split_top_level(s, '&')
    if len(parts) > 1:
        return {'op': 'and', 'args': [parse_raw_formula_ast(p) for p in parts]}

    # Equality / inequality at top level
    def find_top_level(expr: str, target: str) -> int:
        depth = 0
        i = 0
        n = len(expr)
        L = len(target)
        while i <= n - L:
            c = expr[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if depth == 0 and expr.startswith(target, i):
                return i
            i += 1
        return -1

    idx = find_top_level(s, '!=')
    if idx != -1:
        left = s[:idx].strip()
        right = s[idx + 2:].strip()
        return {'op': 'neq', 'args': [parse_raw_formula_ast(left), parse_raw_formula_ast(right)]}
    idx = find_top_level(s, '=')
    if idx != -1:
        left = s[:idx].strip()
        right = s[idx + 1:].strip()
        return {'op': '=', 'args': [parse_raw_formula_ast(left), parse_raw_formula_ast(right)]}

    # Negation
    if s.startswith('~'):
        return {'op': 'not', 'arg': parse_raw_formula_ast(s[1:].strip())}

    # Atom or term (raw token permits ':', '$')
    s = strip_outer_parens(s)
    m = re.match(r"[A-Za-z][A-Za-z0-9_:$]*", s)
    if m:
        token = m.group(0)
        base = _kw_strip_type(token)
        rest = s[len(token):].lstrip()
        if rest.startswith('('):
            # parse arguments
            depth = 1
            arg_parts = []
            last = 1
            i = 1
            n = len(rest)
            while i < n:
                c = rest[i]
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        arg_parts.append(rest[last:i].strip())
                        break
                elif c == ',' and depth == 1:
                    arg_parts.append(rest[last:i].strip())
                    last = i + 1
                i += 1
            args = [parse_raw_formula_ast(arg) for arg in arg_parts]
            return {'atom': {'pred': base, 'args': args}}
        else:
            # variable or constant symbol
            return base
    return s


def _collect_occurrences_for_kinds(ast) -> tuple[set, set]:
    """Collect (func_terms, pred_heads) from a raw-formula AST.

    - pred_heads: symbols that appear as atom heads at *literal* level (not inside terms)
    - func_terms: symbols that appear in *term* position (arguments of atoms or sides of equality)
                  Constants are recorded as (name, 0)
    """
    func_terms: set = set()
    pred_heads: set = set()

    def walk(node, in_term: bool = False):
        if isinstance(node, dict):
            if 'quantifier' in node:
                walk(node['body'], in_term=False)
            elif 'op' in node:
                op = node['op']
                if op in ('or', 'and'):
                    for a in node['args']:
                        walk(a, in_term=False)
                elif op in ('=', 'neq'):
                    # both sides are terms
                    for a in node['args']:
                        walk(a, in_term=True)
                elif op == 'not':
                    walk(node['arg'], in_term=False)
                else:
                    for a in node.get('args', []):
                        walk(a, in_term=False)
            elif 'atom' in node:
                name = node['atom']['pred']
                args = node['atom']['args']
                if in_term:
                    # function symbol in term position
                    ar = len(args)
                    func_terms.add((name, ar))
                    for a in args:
                        walk(a, in_term=True)
                else:
                    # predicate head at literal level
                    pred_heads.add((name, len(args)))
                    for a in args:
                        walk(a, in_term=True)
        elif isinstance(node, str):
            # leaf term: variable or constant
            if in_term and node and (not node[0].isupper()):
                # lowercase-starting symbol as constant (0-arity)
                func_terms.add((node, 0))
        else:
            # lists or others fall back
            if isinstance(node, list):
                for a in node:
                    walk(a, in_term=in_term)

    walk(ast, in_term=False)
    return func_terms, pred_heads


def preprocess_clause_str(clause_str: str) -> str:
    """Remove provenance annotations from a raw TPTP clause.

    iProver prints each clause with a ``file(...)`` or
    ``inference(...)`` suffix containing metadata.  For the purposes
    of proof guidance, this information is not needed.  This helper
    removes the suffix by splitting on ``file(`` or ``inference(`` and
    trimming trailing parentheses.  The returned string still
    contains the ``tcf`` wrapper.

    Parameters
    ----------
    clause_str : str
        The raw clause string from iProver's JSON.  For example::

            "tcf(c_77,plain, (![X0:$i,X1:$i]: (~end_point(X0,X1)|open(X1))),file('clausifier', u123))."

    Returns
    -------
    str
        The cleaned clause without the provenance suffix.  In the
        example above this would be::

            "tcf(c_77,plain, (![X0:$i,X1:$i]: (~end_point(X0,X1)|open(X1)))".
    """
    # Drop provenance after file( or inference(
    if 'file(' in clause_str:
        clause_str = clause_str.split('file(', 1)[0]
    elif 'inference(' in clause_str:
        clause_str = clause_str.split('inference(', 1)[0]
    # Trim trailing comma and whitespace only.  Do not strip
    # parentheses here because the formula may legitimately end with
    # multiple closing parens (e.g. nested function calls) and
    # removing them would corrupt the syntax.  The caller is
    # responsible for balancing parentheses as needed.
    clause_str = clause_str.rstrip()
    # Remove a single trailing comma if present
    if clause_str.endswith(','):
        clause_str = clause_str[:-1].rstrip()
    return clause_str


def extract_formula_from_tcf(clause_str: str) -> str:
    """Extract the logical formula from a cleaned ``tcf`` wrapper.

    A clause from iProver is wrapped as ``tcf(id,plain,<formula>)``.
    After the clause has been cleaned of provenance suffixes, this
    helper locates the substring following the second comma (after
    the ID and the 'plain' indicator) and removes any enclosing
    parentheses.  It does not strip quantifiers or change the
    internal structure of the formula.

    Parameters
    ----------
    clause_str : str
        The cleaned clause starting with ``tcf``.  For example::

            "tcf(c_77,plain, (![X0:$i,X1:$i]: (~end_point(X0,X1)|open(X1)))"

    Returns
    -------
    str
        The extracted formula, for example::

            "![X0:$i,X1:$i]: (~end_point(X0,X1)|open(X1))"

        If the clause does not appear to be a ``tcf`` wrapper, the
        original string is returned unchanged.
    """
    # Ensure the string contains the expected wrapper
    m = re.match(r"\s*tcf\s*\(", clause_str)
    if not m:
        return clause_str
    # Remove leading 'tcf('
    body = clause_str[m.end():]
    # We want to skip over the first two comma‑separated fields:
    #   id, plain, <formula>
    depth = 0
    comma_count = 0
    start_idx: Optional[int] = None
    for idx, ch in enumerate(body):
        if ch == '(':
            depth += 1
        elif ch == ')':
            depth -= 1
        elif ch == ',' and depth == 0:
            comma_count += 1
            if comma_count == 2:
                # start of the formula is immediately after this comma
                start_idx = idx + 1
                break
    if start_idx is None:
        # Could not locate formula, return unchanged
        return clause_str
    formula = body[start_idx:].strip()
    # Remove outer parentheses if they wrap the entire formula
    def strip_parens(s: str) -> str:
        """Repeatedly strip one pair of matching outer parentheses.

        Only strips if the first and last parentheses form a matching
        pair without prematurely closing inside.  This prevents
        removing needed parentheses for nested expressions like
        "(~p|q)".
        """
        while s.startswith('(') and s.endswith(')'):
            depth = 0
            balanced = True
            for i, c in enumerate(s):
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                if depth == 0 and i < len(s) - 1:
                    balanced = False
                    break
            if balanced:
                s = s[1:-1].strip()
            else:
                break
        return s
    formula = strip_parens(formula)
    # Remove trailing full stop if present
    if formula.endswith('.'):
        formula = formula[:-1].strip()
    return formula


@dataclass
class Canonicaliser:
    """Maintain canonical names for symbols across multiple clauses.

    Each non‑variable symbol (predicate, function or constant) is
    assigned a canonical name on first encounter and reused
    thereafter.  Variables are renamed on a per‑clause basis and
    therefore do not need to be stored globally.
    """
    # Mapping from (original symbol name, arity, kind) to canonical name
    # kind ∈ {"predicate","function","constant"}
    symbol_mapping: Dict[Tuple[str, int, str], str] = field(default_factory=dict)
    # Reverse mapping: canonical name -> list of original names
    reverse_mapping: Dict[str, List[str]] = field(default_factory=dict)
    # Information about symbols: canonical name -> details (original names, kind, arity)
    symbol_info: Dict[str, Dict[str, object]] = field(default_factory=dict)

    def get_canonical_symbol(self, name: str, is_constant: bool, arity: int, kind: Optional[str] = None) -> str:
        """Return the canonical name for a symbol, using P# for predicates, F# for functions, C# for constants.
        The mapping key uses (name, arity, kind) to prevent collisions when the same name appears
        as both a predicate and a function.
        """
        if kind is None:
            kind = 'constant' if is_constant else 'predicate'
        prefix = 'C' if kind == 'constant' else ('F' if kind == 'function' else 'P')
        key = (name, 0 if kind == 'constant' else arity, kind)
        if key in self.symbol_mapping:
            canonical = self.symbol_mapping[key]
            info = self.symbol_info.setdefault(canonical, {})
            originals = info.setdefault('original', [])
            # store original name (optionally include arity for non-constants)
            orig_repr = name if kind == 'constant' else f"{name}/{arity}"
            if orig_repr not in originals:
                originals.append(orig_repr)
            # keep the maximal observed arity (constants stay 0)
            prev_arity = int(info.get('arity', 0) or 0)
            info['arity'] = max(prev_arity, 0 if kind == 'constant' else arity)
            info['kind'] = kind
            return canonical
        # allocate next index for this prefix
        existing_indices = [int(s[1:]) for s in self.symbol_mapping.values() if s.startswith(prefix)]
        next_idx = max(existing_indices, default=0) + 1
        canonical = f"{prefix}{next_idx}"
        self.symbol_mapping[key] = canonical
        self.reverse_mapping.setdefault(canonical, []).append(name)
        self.symbol_info[canonical] = {
            'original': [name if kind == 'constant' else f"{name}/{arity}"],
            'kind': kind,
            'arity': 0 if kind == 'constant' else arity,
        }
        return canonical


def canonicalise_formula(
    formula: str,
    canonicaliser: Canonicaliser,
    clause_var_counter_start: int = 0,
    kind_func_terms: Optional[set] = None,
    pred_head_set: Optional[set] = None
) -> Tuple[str, Dict[str, str], Dict[str, set]]:
    """Replace variable and symbol names in a formula with canonical identifiers.

    The function processes the formula character by character to
    identify tokens that correspond to variables or non‑variable
    symbols.  Variable names (those starting with an uppercase
    alphabetic character, possibly followed by digits or underscores,
    optionally with a type suffix like ':$i') are renamed per clause
    to 'V0', 'V1', etc.  Non‑variable symbols are looked up or
    registered in the provided ``canonicaliser``.  A symbol is
    considered a constant if the next non‑whitespace character after
    the token is *not* an opening parenthesis.

    Parameters
    ----------
    formula : str
        The logical formula as a string.
    canonicaliser : Canonicaliser
        Object used to assign canonical names to non‑variable symbols.
    clause_var_counter_start : int, optional
        The starting index for numbering variables within this
        clause.  This is useful if the caller wishes to reserve
        initial variable identifiers (default is 0).

    Returns
    -------
    Tuple[str, Dict[str, str], Dict[str, set]]
        A triple consisting of the canonicalised formula, a mapping
        from the original variable names to their canonical names for
        this clause, and a mapping of canonical symbols to the set of
        original names used in this clause.
    """
    result_chars: List[str] = []
    var_mapping: Dict[str, str] = {}
    local_symbols: Dict[str, set] = {}
    var_counter = clause_var_counter_start
    i = 0
    n = len(formula)
    while i < n:
        ch = formula[i]
        # Copy punctuation and whitespace verbatim
        if not (ch.isalpha() or ch == '_' or ch.isdigit()):
            result_chars.append(ch)
            i += 1
            continue
        # Extract a candidate name token (allow letters, digits, underscores, '$', ':')
        j = i
        while j < n and (formula[j].isalnum() or formula[j] in ['_', '$', ':']):
            j += 1
        token = formula[i:j]
        # Determine if this token is a variable.  In typed TPTP, variables
        # start with an uppercase letter (A–Z) and may include a type
        # suffix after a colon, e.g. 'X0:$i'.  Strip the suffix for
        # canonicalisation.
        # Identify base name up to colon, if present
        base_name = token
        type_suffix = ''
        if ':' in token:
            base_name, type_suffix = token.split(':', 1)
        # A variable if the first character of base_name is uppercase
        is_variable = base_name and base_name[0].isupper()
        if is_variable:
            if base_name not in var_mapping:
                canonical_var = f"V{var_counter}"
                var_mapping[base_name] = canonical_var
                var_counter += 1
            result_chars.append(var_mapping[base_name])
        else:
            # Determine if the symbol is followed by '(' after any whitespace
            k = j
            while k < n and formula[k].isspace():
                k += 1
            is_constant = True
            if k < n and formula[k] == '(':  # has arguments
                is_constant = False
            # Compute arity for non-constant symbols
            arity = 0
            if not is_constant:
                # Count the number of top-level comma-separated arguments
                # starting from the character after '('
                par_depth = 1
                arg_count = 1
                idx2 = k + 1
                while idx2 < n:
                    ch2 = formula[idx2]
                    if ch2 == '(':
                        par_depth += 1
                    elif ch2 == ')':
                        par_depth -= 1
                        if par_depth == 0:
                            break
                    elif ch2 == ',' and par_depth == 1:
                        arg_count += 1
                    idx2 += 1
                arity = arg_count
            # Decide kind using global occurrence sets, with local tie-breaker
            in_func = bool(kind_func_terms and (base_name, arity) in kind_func_terms)
            in_pred = bool(pred_head_set and (base_name, arity) in pred_head_set)
            if is_constant:
                sym_kind = 'constant'
            elif in_pred and not in_func:
                sym_kind = 'predicate'
            elif in_func and not in_pred:
                sym_kind = 'function'
            elif in_func and in_pred:
                # Local heuristic: if after this call's closing ')', the next non-space char is '=' then this is a term
                # Find matching ')'
                kparen = k  # k currently points at '(' after the token
                par_depth = 1
                idx2 = kparen + 1
                while idx2 < n and par_depth > 0:
                    ch2 = formula[idx2]
                    if ch2 == '(':
                        par_depth += 1
                    elif ch2 == ')':
                        par_depth -= 1
                    idx2 += 1
                # idx2 is the position after the matching ')'
                m = idx2
                while m < n and formula[m].isspace():
                    m += 1
                if m + 1 < n and formula[m] == '!' and formula[m+1] == '=':
                    sym_kind = 'function'
                elif m < n and formula[m] == '=':
                    sym_kind = 'function'
                else:
                    sym_kind = 'predicate'
            else:
                # Default to predicate if unknown
                sym_kind = 'predicate'
            canonical_sym = canonicaliser.get_canonical_symbol(base_name, is_constant, arity, sym_kind)
            local_key = canonical_sym if canonical_sym.startswith('C') else f"{canonical_sym}/{arity}"
            local_symbols.setdefault(local_key, set()).add(base_name)
            result_chars.append(canonical_sym)
        # Skip the token in the input
        i = j
    canonical_formula = ''.join(result_chars)
    return canonical_formula, var_mapping, local_symbols

def parse_canonical_formula_ast(s: str):
    """
    Parse a canonical formula string into a nested AST.

    The AST uses dictionaries with keys:
      - 'quantifier': 'forall' or 'exists', with 'vars' and 'body'
      - 'op': 'or', 'and', '=', 'neq', 'not'
      - 'args': list of subexpressions (for binary/multi operators)
      - 'atom': {'pred': name, 'args': [...]}
      - atomic terms are strings (variables or constants)

    This parser is minimal and assumes the canonical formula uses only
    the following constructs: universal/existential quantifiers, negation (~),
    disjunction (|), conjunction (&), equality (=), inequality (!=),
    parentheses, and atoms of the form Pn(arg1,arg2,...).
    """

    import re

    def strip_outer_parens(expr: str) -> str:
        expr = expr.strip()
        while expr.startswith('(') and expr.endswith(')'):
            depth = 0
            balanced = True
            for i, c in enumerate(expr):
                if c == '(':  # pragma: no cover
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0 and i < len(expr) - 1:
                        balanced = False
                        break
            if balanced:
                expr = expr[1:-1].strip()
            else:
                break
        return expr

    s = s.strip()
    # Handle quantifiers
    if s.startswith('![') or s.startswith('?['):
        quant = 'forall' if s.startswith('![') else 'exists'
        # find closing ']:' to split vars and body
        end_idx = s.find(']:')
        if end_idx != -1:
            var_list = s[2:end_idx]
            vars_ = [v.strip() for v in var_list.split(',') if v.strip()]
            body = s[end_idx + 2:].strip()
            return {'quantifier': quant, 'vars': vars_, 'body': parse_canonical_formula_ast(body)}
    # Remove outer parentheses
    s = strip_outer_parens(s)

    # Helper to split at top-level operator
    def split_top_level(expr: str, op_char: str) -> list:
        parts = []
        depth = 0
        last = 0
        i = 0
        n = len(expr)
        while i < n:
            c = expr[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            elif c == op_char and depth == 0:
                parts.append(expr[last:i])
                last = i + 1
            i += 1
        if parts:
            parts.append(expr[last:].strip())
            return [p.strip() for p in parts]
        else:
            return [expr]

    # Disjunction
    parts = split_top_level(s, '|')
    if len(parts) > 1:
        return {'op': 'or', 'args': [parse_canonical_formula_ast(p) for p in parts]}

    # Conjunction
    parts = split_top_level(s, '&')
    if len(parts) > 1:
        return {'op': 'and', 'args': [parse_canonical_formula_ast(p) for p in parts]}

    # Equality or inequality at top level
    # Look for '!=' first
    # We treat '=' and '!=' only if they occur at top level
    def find_top_level(expr: str, target: str) -> int:
        depth = 0
        i = 0
        n = len(expr)
        while i <= n - len(target):
            c = expr[i]
            if c == '(':
                depth += 1
            elif c == ')':
                depth -= 1
            if expr.startswith(target, i) and depth == 0:
                return i
            i += 1
        return -1

    idx = find_top_level(s, '!=')
    if idx != -1:
        left = s[:idx].strip()
        right = s[idx + 2:].strip()
        return {'op': 'neq', 'args': [parse_canonical_formula_ast(left), parse_canonical_formula_ast(right)]}
    idx = find_top_level(s, '=')
    if idx != -1:
        left = s[:idx].strip()
        right = s[idx + 1:].strip()
        return {'op': '=', 'args': [parse_canonical_formula_ast(left), parse_canonical_formula_ast(right)]}

    # Negation
    if s.startswith('~'):
        return {'op': 'not', 'arg': parse_canonical_formula_ast(s[1:].strip())}

    # Atom or term
    s = strip_outer_parens(s)
    m = re.match(r"[A-Za-z][A-Za-z0-9_]*", s)
    if m:
        token = m.group(0)
        rest = s[len(token):].lstrip()
        if rest.startswith('('):
            # parse arguments inside parentheses
            depth = 1
            arg_parts = []
            last = 1
            i = 1
            n = len(rest)
            while i < n:
                c = rest[i]
                if c == '(':
                    depth += 1
                elif c == ')':
                    depth -= 1
                    if depth == 0:
                        arg_parts.append(rest[last:i].strip())
                        break
                elif c == ',' and depth == 1:
                    arg_parts.append(rest[last:i].strip())
                    last = i + 1
                i += 1
            args = [parse_canonical_formula_ast(arg) for arg in arg_parts]
            return {'atom': {'pred': token, 'args': args}}
        else:
            return token
    # Fall back: return the string itself
    return s


def parse_raw_log(path: str) -> Tuple[Dict[int, Dict], List[int], Optional[str]]:
    """Parse a raw iProver interactive log into clauses and scores.

    The raw log is expected to contain two JSON objects separated
    by a NUL (\x00) character.  The first object must have ``tag``
    equal to ``register_clauses`` and include a list of clauses with
    associated identifiers and features.  The second object must
    contain a ``scores_req`` message with a list of clause IDs to
    score.

    Parameters
    ----------
    path : str
        Path to the raw log file.

    Returns
    -------
    Tuple[Dict[int, Dict], List[int], Optional[str]]
        A mapping from clause_id to its data (containing cleaned
        clause, features, etc.), the list of clause IDs in the
        ``scores_req`` request, and the value of the 'component'
        field from the scores request (or None if absent).
    """
    with open(path, 'rb') as f:
        data = f.read().split(b'\x00')
    # Filter out empty segments
    segments = [seg for seg in data if seg.strip()]
    if not segments:
        raise ValueError("No JSON segments found in raw log")
    # First segment: register_clauses
    register_msg = json.loads(segments[0].decode('utf-8'))
    if register_msg.get('tag') != 'register_clauses':
        raise ValueError("First message is not a register_clauses message")
    # Build clause dict keyed by id
    clauses: Dict[int, Dict] = {}
    for entry in register_msg.get('clauses', []):
        cid = entry['clause_id']
        raw_clause: str = entry['clause']
        features = entry.get('clause_features', {})
        cleaned = preprocess_clause_str(raw_clause)
        # Extract formula from tcf wrapper if present
        formula = extract_formula_from_tcf(cleaned)
        clauses[cid] = {
            'raw_clause': raw_clause,
            'clean_clause': cleaned,
            'formula': formula,
            'features': features,
        }
    # Second segment: scores_req (may start with newline)
    scores_msg = None
    for seg in segments[1:]:
        try:
            msg = json.loads(seg.decode('utf-8'))
        except json.JSONDecodeError:
            continue
        if msg.get('tag') == 'scores_req':
            scores_msg = msg
            break
    if scores_msg is None:
        raise ValueError("No scores_req message found in raw log")
    clause_ids = scores_msg.get('clause_ids', [])
    component = scores_msg.get('component')
    return clauses, clause_ids, component


def build_canonical_dataset(
    clauses: Dict[int, Dict],
    clause_ids: List[int],
    context_size: int,
    candidate_size: int,
    mapping_scope: str = 'all',
    component: Optional[str] = None,
    include_ast: bool = False
) -> Dict:
    """Construct a canonical representation of the problem for the LLM.

    This helper selects a subset of clauses to act as context and
    candidates, canonicalises their symbols and variables, and
    identifies the conjecture clause (the one with ``conj_dist == 0``).

    Parameters
    ----------
    clauses : Dict[int, Dict]
        Mapping from clause ID to clause information as returned by
        :func:`parse_raw_log`.
    clause_ids : List[int]
        The list of clause IDs from the ``scores_req`` message.
    context_size : int
        Number of context clauses to select from the registered
        clauses.  These provide background information for the LLM.
    candidate_size : int
        Number of candidate clause IDs to take from ``clause_ids``.

    Returns
    -------
    Dict
        A dictionary with keys:

        ``metadata`` : dict
            Metadata about this batch including the schema version,
            mapping scope, batch identifier, and the component from the
            scores request.

        ``symbol_map`` : dict
            A mapping from canonical symbol names to details including
            the list of original names, the kind (constant or symbol) and
            arity.

        ``conjecture`` : dict or None
            The conjecture clause (if one exists) with its canonical
            formula, AST, original formula, features, variable mapping and
            local symbols.

        ``context_clauses`` : list
            A list of context clause dictionaries similar to the
            conjecture representation.

        ``candidate_clauses`` : list
            A list of candidate clause dictionaries similar to the
            conjecture representation.
    """
    canonicaliser = Canonicaliser()
    # Identify the conjecture clause: choose the first clause with conj_dist == 0
    conjecture_entry: Optional[Tuple[int, Dict]] = None
    for cid, info in clauses.items():
        if info['features'].get('conj_dist') == 0:
            conjecture_entry = (cid, info)
            break
    # Determine context clause IDs in order of registration, excluding conjecture
    # and excluding any IDs explicitly requested for scoring (to avoid starving candidates)
    register_order = list(clauses.keys())
    requested = set(clause_ids)
    context_ids: List[int] = []
    for cid in register_order:
        if conjecture_entry is not None and cid == conjecture_entry[0]:
            continue
        if cid in requested:
            continue
        context_ids.append(cid)
        if len(context_ids) >= context_size:
            break
    # Determine candidate clause IDs from scores_req, excluding conjecture, unknown ids, and any context ids
    context_set = set(context_ids)
    seen = set()  # to preserve order while deduplicating
    candidate_ids: List[int] = []
    for cid in clause_ids:
        if cid not in clauses:
            continue
        if conjecture_entry is not None and cid == conjecture_entry[0]:
            continue
        if cid in context_set:
            continue
        if cid in seen:
            continue
        seen.add(cid)
        candidate_ids.append(cid)
        if len(candidate_ids) >= candidate_size:
            break
    # Build function-kind lookup: which symbols appear as functions in term position
    mapping_ids = list(clauses.keys()) if mapping_scope == 'all' else (
        ([conjecture_entry[0]] if conjecture_entry is not None else []) + context_ids + candidate_ids
    )
    kind_funcs: set = set()
    kind_preds: set = set()
    for _cid in mapping_ids:
        fset, pset = _classify_symbols_in_formula(clauses[_cid]['formula'])
        kind_funcs |= fset
        kind_preds |= pset
    kind_conflicts = sorted({f"{n}/{a}" for (n,a) in (kind_funcs & kind_preds) if a > 0})
    # Seed the canonicaliser with symbols from mapping_ids (depends on mapping_scope)
    for cid in mapping_ids:
        entry = clauses[cid]
        formula = entry['formula']
        i = 0
        n = len(formula)
        while i < n:
            ch = formula[i]
            if not (ch.isalpha() or ch == '_' or ch.isdigit()):
                i += 1
                continue
            j = i
            while j < n and (formula[j].isalnum() or formula[j] in ['_', '$', ':']):
                j += 1
            token = formula[i:j]
            base_name = token
            if ':' in token:
                base_name, _ = token.split(':', 1)
            is_variable = base_name and base_name[0].isupper()
            if not is_variable:
                k = j
                while k < n and formula[k].isspace():
                    k += 1
                is_constant = True
                arity = 0
                if k < n and formula[k] == '(':  # has arguments
                    is_constant = False
                    par_depth = 1
                    arg_count = 1
                    idx2 = k + 1
                    while idx2 < n:
                        ch2 = formula[idx2]
                        if ch2 == '(':
                            par_depth += 1
                        elif ch2 == ')':
                            par_depth -= 1
                            if par_depth == 0:
                                break
                        elif ch2 == ',' and par_depth == 1:
                            arg_count += 1
                        idx2 += 1
                    arity = arg_count
                if is_constant:
                    sym_kind = 'constant'
                else:
                    in_func = (base_name, arity) in kind_funcs
                    in_pred = (base_name, arity) in kind_preds
                    if in_pred and not in_func:
                        sym_kind = 'predicate'
                    elif in_func and not in_pred:
                        sym_kind = 'function'
                    elif in_func and in_pred:
                        # Ambiguous globally: default to predicate for seeding
                        sym_kind = 'predicate'
                    else:
                        sym_kind = 'predicate'
                canonicaliser.get_canonical_symbol(base_name, is_constant, arity, sym_kind)
            i = j
    # Build conjecture representation
    conj_repr = None
    if conjecture_entry is not None:
        conj_id, conj_info = conjecture_entry
        cformula, var_map, local_syms = canonicalise_formula(
            conj_info['formula'], canonicaliser,
            kind_func_terms=kind_funcs, pred_head_set=kind_preds
        )
        # NOTE: This is the single source of truth for conjecture representation.
        # Do not rebuild conj_repr later in this function to avoid divergence.
        conj_repr = {
            'id': conj_id,
            'canonical_formula': cformula,
            'original_formula': conj_info['formula'],
            'features': conj_info['features'],
            'variable_mapping': var_map,
            'local_symbols': {canon: sorted(list(names)) for canon, names in local_syms.items()},
        }
        if include_ast:
            conj_repr['canonical_formula_ast'] = parse_canonical_formula_ast(cformula)

    # <<< 新增：从猜想中提取目标函数/常量集合 >>>
    conj_targets = _extract_conjecture_targets(conj_repr)

    # Local helper to canonicalise a clause + tags（闭包里用到 conj_targets）
    def canonicalise_clause(cid: int) -> Dict:
        entry = clauses[cid]
        formula = entry['formula']
        cformula, var_map, local_syms = canonicalise_formula(
            formula, canonicaliser,
            kind_func_terms=kind_funcs, pred_head_set=kind_preds
        )
        record = {
            'id': cid,
            'canonical_formula': cformula,
            'original_formula': formula,
            'features': entry['features'],
            'variable_mapping': var_map,
            'local_symbols': {canon: sorted(list(names)) for canon, names in local_syms.items()},
        }
        ast = None
        if include_ast:
            ast = parse_canonical_formula_ast(cformula)
            record['canonical_formula_ast'] = ast
        # <<< 新增：计算 reasoning tags >>>
        tags, tag_info = _compute_reasoning_tags(cformula, ast, conj_targets)
        record['tags'] = tags
        record['tag_info'] = tag_info
        # 兜底：若缺少 lit_count，可用顶层 "or" 拆分估算
        if 'lit_count' not in record['features']:
            lits = _kw_split_top_level_disj(cformula)
            record['features']['lit_count'] = len(lits) if lits else 1
        return record
    # Canonicalise context and candidate clauses
    context_clauses = [canonicalise_clause(cid) for cid in context_ids]
    candidate_clauses = [canonicalise_clause(cid) for cid in candidate_ids]
    # Collect actually-used canonical keys from local_symbols across this batch
    used_keys: set = set()
    if conj_repr is not None:
        used_keys.update(conj_repr['local_symbols'].keys())
    for c in context_clauses:
        used_keys.update(c['local_symbols'].keys())
    for c in candidate_clauses:
        used_keys.update(c['local_symbols'].keys())
    # Build symbol map with kind and arity, filtered to only those used in this batch
    symbol_map: Dict[str, Dict[str, object]] = {}
    for canon, info in canonicaliser.symbol_info.items():
        orig = sorted(set(info.get('original', [])))
        kind = info.get('kind', 'symbol')
        arity = info.get('arity', 0)
        key = canon if kind == 'constant' else f"{canon}/{arity}"
        if key not in used_keys:
            continue
        symbol_map[key] = {
            'original': orig,
            'kind': kind,
            'arity': arity,
        }
    # Metadata
    symbols_total_parsed = len(canonicaliser.symbol_info)
    symbol_map_size_used = len(symbol_map)
    import time
    batch_id = str(int(time.time()))
    return {
        'metadata': {
            'schema_version': '1.0',
            'mapping_scope': mapping_scope,
            'batch_id': batch_id,
            'component': component,
            'kind_conflicts': kind_conflicts,
            'symbols_total_parsed': symbols_total_parsed,
            'symbol_map_size_used': symbol_map_size_used,
        },
        'symbol_map': symbol_map,
        'conjecture': conj_repr,
        'conjecture_targets': conj_targets,   # <<< 新增
        'context_clauses': context_clauses,
        'candidate_clauses': candidate_clauses,
    }

# ====== Reasoning tags helpers (add below parse_canonical_formula_ast) ======
# --------- EA helpers: stable send/print routines ---------

def _ea_send(conn, obj, msg_delim: str) -> None:
    """Send a JSON message to iProver and mirror it to stdout with a stable prefix."""
    import json
    try:
        print(f"[EA OUT] {json.dumps(obj, ensure_ascii=False)}", flush=True)
    except Exception:
        pass
    try:
        conn.sendall((json.dumps(obj) + msg_delim).encode("utf-8"))
    except Exception:
        # If the connection is broken, just ignore; server loop will exit soon.
        pass

def _ea_print_in(obj) -> None:
    """Stable input logging to stdout (used for '[EA IN] ...')."""
    import json
    try:
        print(f"[EA IN] {json.dumps(obj, ensure_ascii=False)}", flush=True)
    except Exception:
        pass

def _collect_constants_from_canonical(formula: str) -> set[str]:
    return set(re.findall(r"C\d+", formula))

def _term_head_name(node):
    """Return (head_name, arity) if node is a function term of form f(args), else None."""
    if isinstance(node, dict) and "atom" in node and isinstance(node["atom"].get("args"), list):
        return node["atom"]["pred"], len(node["atom"]["args"])
    return None

def _extract_conjecture_targets(conj_repr: Optional[dict]) -> dict:
    """
    From the conjecture canonical formula, extract:
      - target functors (names/arity) appearing as heads on both sides of =/!=
      - the first-argument constants of those functors (e.g., {C1})
      - all constants used inside those functor arguments (goal const set)
    """
    targets = {"functors": set(), "first_arg_consts": set(), "goal_consts": set()}
    if not conj_repr:
        return {"functors": [], "first_arg_consts": [], "goal_consts": []}

    ast = parse_canonical_formula_ast(conj_repr["canonical_formula"])
    if isinstance(ast, dict) and ast.get("op") in ("=", "neq"):
        for side in ast["args"]:
            head = _term_head_name(side)
            if head:
                name, ar = head
                targets["functors"].add((name, ar))
                args = side["atom"]["args"]
                if args:
                    a0 = args[0]
                    if isinstance(a0, str) and re.fullmatch(r"C\d+", a0):
                        targets["first_arg_consts"].add(a0)
                for a in args:
                    if isinstance(a, str) and re.fullmatch(r"C\d+", a):
                        targets["goal_consts"].add(a)

    return {
        "functors": [f"{n}/{k}" for (n,k) in sorted(targets["functors"])],
        "first_arg_consts": sorted(targets["first_arg_consts"]),
        "goal_consts": sorted(targets["goal_consts"]),
    }

def _compute_reasoning_tags(canonical_formula: str, ast, targets: dict) -> tuple[list[str], dict]:
    """
    Compute lightweight reasoning tags for one clause, given the conjecture targets.
    Returns (tags_list, tag_info_dict).
    """
    tags=[]; info={}
    tf_names = {t.split('/')[0] for t in targets.get("functors", [])}

    # parse AST if caller didn't provide
    if not ast:
        ast = parse_canonical_formula_ast(canonical_formula)

    # touches_target_functor: cheap check by string
    touches = any((name + "(") in canonical_formula for name in tf_names)
    if touches:
        tags.append("touches_target_functor")

    # eq_of_target_functor: both sides of =/!= are target head
    is_eq = isinstance(ast, dict) and ast.get("op") in ("=", "neq")
    if is_eq:
        left, right = ast["args"]
        def head_name(n):
            h = _term_head_name(n)
            return h[0] if h else None
        if head_name(left) in tf_names and head_name(right) in tf_names:
            tags.append("eq_of_target_functor")

    # first_arg_in_goal: exists F*(C_goal,*,*)
    first_consts = set(targets.get("first_arg_consts", []))
    def has_first_arg_const(node) -> bool:
        if isinstance(node, dict) and "atom" in node:
            name = node["atom"]["pred"]
            if name in tf_names and node["atom"]["args"]:
                a0 = node["atom"]["args"][0]
                return isinstance(a0, str) and a0 in first_consts
        return False

    def walk(n) -> bool:
        if has_first_arg_const(n):
            return True
        if isinstance(n, dict):
            if "atom" in n:
                for a in n["atom"]["args"]:
                    if walk(a): return True
            # check generic substructures
            for k in ("args","body"):
                v = n.get(k)
                if isinstance(v, list):
                    for a in v:
                        if walk(a): return True
                elif isinstance(v, dict):
                    if walk(v): return True
        return False

    if first_consts and walk(ast):
        tags.append("first_arg_in_goal")

    # shares_goal_consts:k
    clause_consts = _collect_constants_from_canonical(canonical_formula)
    goal_consts = set(targets.get("goal_consts", []))
    shared = clause_consts & goal_consts
    if shared:
        tags.append(f"shares_goal_consts:{len(shared)}")

    info = {
        "touches_target_functor": bool(touches),
        "eq_of_target_functor": ("eq_of_target_functor" in tags),
        "first_arg_in_goal": ("first_arg_in_goal" in tags),
        "shares_goal_consts_n": len(shared),
        "constants_shared": sorted(shared),
    }
    return tags, info
# ====== end helpers ======


def make_batch_for_scores_req(
    all_clauses: Dict[int, Dict],
    req_ids: List[int],
    context_size: int = 128,
    mapping_scope: str = 'batch',
    include_ast: bool = False,
    component: Optional[str] = None,
) -> Dict:
    """Build a canonical batch tailored for a single scores_req interaction.

    Ensures the context never overlaps requested IDs and that all requested IDs
    appear in the candidate set (no truncation).
    """
    return build_canonical_dataset(
        clauses=all_clauses,
        clause_ids=req_ids,
        context_size=context_size,
        candidate_size=len(req_ids),
        mapping_scope=mapping_scope,
        component=component,
        include_ast=True,  # <-- was False
    )

# ---------------------- Minimal interactive EA server (optional) ----------------------
class _EAState:
    def __init__(self):
        self.clauses: Dict[int, Dict[str, Any]] = {}
        # Store the last pending scores_req until server_queries_start arrives
        self.pending_scores: Optional[Dict[str, Any]] = None
        # Latest SAT ground literal evaluations keyed by clause id
        self.last_sat_eval: Dict[int, List[bool]] = {}
        # Last SAT solver exec result ("sat"/"unsat"), if requested
        self.last_sat_result: Optional[str] = None
        # Cache: (req_ids tuple, component, component_id, sat_hash) -> scores list
        self.scores_cache: Dict[tuple, List[float]] = {}
        # Throttle counter for SAT query rounds
        self.sat_round: int = 0


def _ea_make_cache_key(req_ids: List[int], component, component_id, sat_map: Optional[Dict[int, List[bool]]]) -> tuple:
    """Build a stable cache key from req_ids, component identifiers, and SAT snapshot."""
    import hashlib, json
    if sat_map:
        norm = {int(k): list(map(bool, v or [])) for k, v in sat_map.items()}
        sat_hash = hashlib.sha1(json.dumps(norm, sort_keys=True).encode("utf-8")).hexdigest()
    else:
        sat_hash = "nosat"
    return (tuple(req_ids), component, component_id, sat_hash)


def _ea_fallback_scores_heuristic(req_ids: List[int], state: _EAState) -> Dict[int, float]:
    """Heuristic non-zero fallback to keep iProver moving when the ranker fails."""
    vals: Dict[int, float] = {}
    for cid in req_ids:
        info = state.clauses.get(int(cid), {})
        formula = info.get("formula", "") or ""
        f = info.get("features", {}) or {}
        s = 0.0
        cd = f.get("conj_dist", -1)
        if isinstance(cd, int) and cd >= 0:
            s += 2.0 / (1.0 + cd)
        else:
            s += 0.2
        if '|' not in formula:
            s += 0.5
        if '=' in formula:
            s += 0.3
        if f.get("horn", False):
            s += 0.2
        sat_vals = state.last_sat_eval.get(int(cid), [])
        if sat_vals:
            support = sum(1 for v in sat_vals if v) / max(1, len(sat_vals))
            pressure = 1.0 - support
            s += 0.3 * float(pressure)
        vals[int(cid)] = s
    if not vals:
        return {}
    vs = list(vals.values())
    vmin, vmax = min(vs), max(vs)
    if vmax - vmin < 1e-9:
        return {int(cid): 0.5 for cid in req_ids}
    return {int(cid): (vals[int(cid)] - vmin) / (vmax - vmin) for cid in req_ids}


def _ea_iter_json_messages(conn):
    import json
    buf = b""
    dec = json.JSONDecoder()
    while True:
        chunk = conn.recv(8192)
        if not chunk:
            break
        buf += chunk
        # First split by NUL terminator if present
        parts = buf.split(b"\x00")
        for part in parts[:-1]:
            part = part.strip()
            if part:
                try:
                    yield json.loads(part.decode("utf-8", errors="ignore"))
                except Exception:
                    pass
        buf = parts[-1]
        # Then try concatenated JSON without delimiters
        s = buf.decode("utf-8", errors="ignore").lstrip()
        while s:
            try:
                obj, end = dec.raw_decode(s)
                yield obj
                s = s[end:].lstrip()
            except json.JSONDecodeError:
                break
        buf = s.encode("utf-8")


def _ea_handle_register_clauses(state: _EAState, msg: Dict[str, Any], log_fp=None) -> None:
    for entry in msg.get("clauses", []):
        cid = int(entry["clause_id"])
        raw = entry["clause"]
        feats = entry.get("clause_features", {})
        clean = preprocess_clause_str(raw)
        formula = extract_formula_from_tcf(clean)
        state.clauses[cid] = {
            "raw_clause": raw,
            "clean_clause": clean,
            "formula": formula,
            "features": feats,
        }
    if log_fp is not None:
        log_fp.write(json.dumps(msg, ensure_ascii=False) + "\x00")
        log_fp.flush()


def _ea_score_with_ranker(batch: Dict, ranker_script: str, model: str,
                          chunk_size: int, anchors: int,
                          context_summary_k: int, summary_max_tokens: int,
                          dry_run: bool, progress: bool, verbose: bool,
                          save_prompts: bool, artifacts_dir: Optional[str],
                          python_exec: Optional[str] = None) -> Dict[int, float]:
    import json, tempfile, subprocess, sys, os, time, hashlib
    exe = python_exec or sys.executable

    def _mk_req_dir(base: str) -> str:
        ts = int(time.time() * 1000)
        cand_ids = [c.get('id') for c in batch.get('candidate_clauses', [])]
        h = hashlib.sha1((','.join(map(str, cand_ids))).encode('utf-8')).hexdigest()[:8]
        d = os.path.join(base, f"scores_req_{ts}_{len(cand_ids)}_{h}")
        os.makedirs(d, exist_ok=True)
        return d

    if artifacts_dir:
        req_dir = _mk_req_dir(artifacts_dir)
        in_path  = os.path.join(req_dir, 'iprover_llm_output.json')
        out_path = os.path.join(req_dir, 'out_scores.json')
        with open(in_path, 'w', encoding='utf-8') as f:
            json.dump(batch, f, ensure_ascii=False)
        cmd = [
            exe, ranker_script,
            '--input', in_path,
            '--out', out_path,
            '--chunk-size', str(chunk_size),
            '--anchors', str(anchors),
            '--context-summary-k', str(context_summary_k),
            '--summary-max-tokens', str(summary_max_tokens),
            '--model', model,
        ]
        if dry_run:
            cmd.append('--dry-run')
        if progress:
            cmd.append('--progress')
        if verbose:
            cmd.append('--verbose')
        if save_prompts:
            cmd.append('--save-prompts')
        # 在 req_dir 下运行，确保 prompts/chunks 也写到该目录
        if verbose:
            print(f"[EA] ranker python: {exe}")
        subprocess.run(cmd, check=True, cwd=req_dir)
        data = json.load(open(out_path, 'r', encoding='utf-8'))
        print(f"[EA] artifacts saved under: {req_dir}")
        return {int(e['id']): float(e['score']) for e in data.get('scores', [])}
    else:
        with tempfile.TemporaryDirectory() as td:
            in_path  = os.path.join(td, 'iprover_llm_output.json')
            out_path = os.path.join(td, 'out_scores.json')
            with open(in_path, 'w', encoding='utf-8') as f:
                json.dump(batch, f, ensure_ascii=False)
            cmd = [
                exe, ranker_script,
                '--input', in_path,
                '--out', out_path,
                '--chunk-size', str(chunk_size),
                '--anchors', str(anchors),
                '--context-summary-k', str(context_summary_k),
                '--summary-max-tokens', str(summary_max_tokens),
                '--model', model,
            ]
            if dry_run:
                cmd.append('--dry-run')
            if progress:
                cmd.append('--progress')
            if verbose:
                cmd.append('--verbose')
            if save_prompts:
                cmd.append('--save-prompts')
            if verbose:
                print(f"[EA] ranker python: {exe}")
            subprocess.run(cmd, check=True)
            data = json.load(open(out_path, 'r', encoding='utf-8'))
            return {int(e['id']): float(e['score']) for e in data.get('scores', [])}


def _ea__size_tier_and_budget(n: int) -> tuple[str, int]:
    r"""
    Size tiers and LLM budgets (K) — lightweight version:
      S (n ≤ 80)     : K = n (all)
      A (80 < n ≤ 400): K = 200
      B (400 < n ≤ 2k): K = 128
      C (2k < n ≤ 10k): K = 64
      D (n > 10k)     : K = 48
    r"""
    if n <= 80:
        return ("S", n)
    if n <= 400:
        return ("A", 200)
    if n <= 2000:
        return ("B", 128)
    if n <= 10000:
        return ("C", 64)
    return ("D", 48)


def _ea__token_weight(canonical_formula: str) -> int:
    r"""
    Crude weight estimate: count of symbol-like tokens (F#, P#, C#, EQ, variables),
    plus a small penalty for disjunctions.
    r"""
    import re as _re
    tok = len(_re.findall(r"[A-Z][A-Za-z0-9_]*", canonical_formula or ""))
    tok += canonical_formula.count("|") * 2
    return tok


def _ea_prefilter_select(batch: dict) -> list[int]:
    r"""
    Choose a focused subset of candidates for LLM scoring.
    Priority:
      1) eq_of_target_functor
      2) touches_target_functor & first_arg_in_goal
      3) shares_goal_consts:k
      4) unit / short weight
    Returns a list of candidate IDs to send to the LLM in priority order.
    r"""
    cands = list(batch.get("candidate_clauses", []))
    n = len(cands)
    tier, K = _ea__size_tier_and_budget(n)

    # If tiny, just return all
    if K >= n:
        return [int(c.get("id")) for c in cands]

    # Prepare priority tuples
    items = []
    for c in cands:
        cid = int(c.get("id"))
        cf = c.get("canonical_formula") or c.get("formula") or ""
        tags = c.get("tags") or []
        tag_info = c.get("tag_info") or {}
        unit = ("|" not in cf)
        # Hook-like score
        hook = 0.0
        if tag_info.get("eq_of_target_functor"):
            hook = 1.0
        elif tag_info.get("touches_target_functor") and tag_info.get("first_arg_in_goal"):
            hook = 0.85
        else:
            # shares_goal_consts:k encoded as tag or in tag_info
            k = 0
            # Try tag_info first
            k = int(tag_info.get("shares_goal_consts_count", 0) or 0)
            if not k:
                # Fallback: parse tag "shares_goal_consts:k"
                for t in tags:
                    if t.startswith("shares_goal_consts:"):
                        try:
                            k = int(t.split(":")[1])
                        except Exception:
                            k = 0
                        break
            hook = min(0.6, 0.15 * k)

        w = _ea__token_weight(cf)
        # Make eq/touches/unit more attractive, penalize heavy clauses
        prio = (
            (3 if tag_info.get("eq_of_target_functor") else 0) * 1000 +
            (2 if (tag_info.get("touches_target_functor") and tag_info.get("first_arg_in_goal")) else 0) * 500 +
            int(hook * 100) * 5 +
            (50 if unit else 0) -
            int(0.3 * w)
        )
        items.append((prio, cid))

    items.sort(reverse=True)  # high priority first
    selected = [cid for _, cid in items[:K]]
    return selected


def _ea_handle_scores_req(state: _EAState, msg: Dict[str, Any], args, log_fp=None) -> Dict[str, Any]:
    req_ids = [int(i) for i in msg.get("clause_ids", [])]
    component = msg.get("component")
    component_id = msg.get("component_id")
    # Cache lookup (consider SAT snapshot when enabled)
    sat_for_cache = state.last_sat_eval if getattr(args, 'include_sat_eval', False) else None
    cache_key = _ea_make_cache_key(req_ids, component, component_id, sat_for_cache)
    if cache_key in state.scores_cache:
        scores_list = list(state.scores_cache[cache_key])
        res: Dict[str, Any] = {"tag": "scores_res", "scores": scores_list}
        if component is not None:
            res["component"] = component
        if component_id is not None:
            res["component_id"] = component_id
        if getattr(args, 'echo_clause_ids', False):
            res["clause_ids"] = req_ids
        if log_fp is not None:
            delim = getattr(args, '_ea_log_delim', "\x00")
            log_fp.write(json.dumps(msg, ensure_ascii=False) + delim)
            log_fp.write(json.dumps(res, ensure_ascii=False) + delim)
            log_fp.flush()
        try:
            print(f"[EA] cache hit for {len(req_ids)} clauses (component={component}/{component_id}).")
        except Exception:
            pass
        return res
    batch = make_batch_for_scores_req(
        all_clauses=state.clauses,
        req_ids=req_ids,
        context_size=args.context_size,
        mapping_scope="batch",
        include_ast=False,
        component=component,
    )
    # Thread optional EA-query features (e.g., SAT ground literal evaluations) into the batch
    if getattr(args, 'include_sat_eval', False) and getattr(state, 'last_sat_eval', None):
        # Keep a compact mapping cid -> list[bool]
        batch.setdefault('ea_query_features', {})['sat_lit_gr_vals'] = {
            int(cid): list(vals) for cid, vals in state.last_sat_eval.items()
        }
    if getattr(state, 'last_sat_result', None) is not None:
        batch.setdefault('ea_query_features', {})['sat_solver_exec_result'] = state.last_sat_result
    try:
        
        # --- Prefilter + low-floor scoring (added) ---
        if PREFILTER_ENABLED:
            selected_ids = _ea_prefilter_select(batch)
            selected_set = set(int(x) for x in selected_ids)
            # Build a reduced batch for the LLM
            reduced_batch = dict(batch)
            reduced_batch['candidate_clauses'] = [c for c in batch.get('candidate_clauses', []) if int(c.get('id')) in selected_set]
            # Call ranker on the reduced set
            selected_scores = _ea_score_with_ranker(
                reduced_batch,
                ranker_script=args.ranker,
                model=args.model,
                chunk_size=args.chunk_size,
                anchors=args.anchors,
                context_summary_k=args.context_summary_k,
                summary_max_tokens=args.summary_max_tokens,
                dry_run=args.dry_run,
                progress=args.progress,
                verbose=args.verbose,
                save_prompts=args.save_prompts,
                artifacts_dir=args.artifacts_dir,
                python_exec=getattr(args, 'python_exec', None),
            )
            # Initialize all as very low floor, then overlay selected
            scores_map = {int(cid): float(LOW_FLOOR_SCORE) for cid in req_ids}
            scores_map.update({int(k): float(v) for k, v in selected_scores.items()})
        else:
            scores_map = _ea_score_with_ranker(
                    batch=batch,
                    ranker_script=args.ranker_script,
                    model=args.model,
                    chunk_size=args.chunk_size,
                    anchors=args.anchors,
                    context_summary_k=args.context_summary_k,
                    summary_max_tokens=args.summary_max_tokens,
                    dry_run=args.dry_run,
                    progress=args.progress,
                    verbose=args.verbose,
                    save_prompts=args.save_prompts,
                    artifacts_dir=args.artifacts_dir,
                    python_exec=getattr(args, 'python_exec', None),
                )
    except Exception as e:
                # Robust fallback: keep iProver running even if the ranker fails
                try:
                    print(f"[EA] ranker failed: {e}. Using heuristic fallback for {len(req_ids)} clauses.")
                except Exception:
                    pass
                scores_map = _ea_fallback_scores_heuristic(req_ids, state)
    scores_list = [scores_map.get(cid, 0.0) for cid in req_ids]
            # Save to cache
    state.scores_cache[cache_key] = list(map(float, scores_list))
        # --- end added block ---

    # Include component identifiers for robustness across iProver builds
    res: Dict[str, Any] = {"tag": "scores_res", "scores": scores_list}
    if component is not None:
        res["component"] = component
    if component_id is not None:
        res["component_id"] = component_id
    # Optionally echo clause_ids to make mapping explicit (some agents rely on this)
    if getattr(args, 'echo_clause_ids', False):
        res["clause_ids"] = req_ids
    if log_fp is not None:
        delim = getattr(args, '_ea_log_delim', "\x00")
        log_fp.write(json.dumps(msg, ensure_ascii=False) + delim)
        log_fp.write(json.dumps(res, ensure_ascii=False) + delim)
        log_fp.flush()
    return res


def run_ea_server(host: str, port: int, args) -> None:
    import socket, threading, json, sys, os, time
    state = _EAState()
    # Hard-code unified logs root and derive a per-run directory
    LOGS_ROOT = "/home/ks/LLM/Logs"
    run_dir = os.path.join(LOGS_ROOT, f"EA.{port}.{int(time.time())}")
    os.makedirs(run_dir, exist_ok=True)

    # Default locations under run_dir (respect user-provided values)
    if not args.log_file:
        args.log_file = os.path.join(run_dir, "EA.raw.jsonl.nul")
    if not args.artifacts_dir:
        args.artifacts_dir = os.path.join(run_dir, "requests")
    os.makedirs(args.artifacts_dir, exist_ok=True)

    # Tee stdout to a file capturing console output (use sys.__stdout__ to avoid recursion)
    class _Tee:
        def __init__(self, *streams):
            self._streams = streams
        def write(self, data):
            for s in self._streams:
                try:
                    s.write(data)
                    s.flush()
                except Exception:
                    pass
            return len(data)
        def flush(self):
            for s in self._streams:
                try:
                    s.flush()
                except Exception:
                    pass

    try:
        stdout_log_path = os.path.join(run_dir, "EA.stdout.log")
        _stdout_f = open(stdout_log_path, "a", encoding="utf-8")
        base_stream = getattr(sys, "__stdout__", None) or sys.stdout
        sys.stdout = _Tee(base_stream, _stdout_f)
    except Exception:
        _stdout_f = None

    # Open raw JSON log (NUL-delimited)
    try:
        log_fp = open(args.log_file, "a", encoding="utf-8")
    except Exception:
        log_fp = None
    # Message delimiter configuration
    # iProver writes JSON followed by "\n\x00\n" and reads incoming JSON line-by-line.
    # If we only send a NUL without a trailing newline, iProver will block waiting for a line end.
    # Therefore, when 'nul' is selected, we send the exact iProver delimiter "\n\x00\n";
    # otherwise we send a plain newline-only framing.
    NUL_NEWLINE_DELIM = "\n\x00\n" 
    msg_delim = NUL_NEWLINE_DELIM if getattr(args, 'delimiter', 'nul') == 'nul' else "\n"
    # stash for logging helper
    setattr(args, '_ea_log_delim', msg_delim)
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.bind((host, port))
    sock.listen(1)
    # Make accept interruptible so we can exit promptly after finish
    sock.settimeout(1.0)
    print(f"[EA] logs dir: {run_dir}")
    print(f"[EA] listening on {host}:{port} (delimiter={'NUL+NEWLINE' if msg_delim==NUL_NEWLINE_DELIM else 'NEWLINE'})")

    shutdown_event = threading.Event()
    # Throttle frequency for SAT ground-literal evaluation (every N rounds)
    SAT_EVAL_EVERY = 3

    def handle(conn, addr):
        with conn:
            print(f"[EA] connected from {addr}")
            for msg in _ea_iter_json_messages(conn):
                tag = msg.get("tag")
                _ea_print_in(msg)
                if tag == "register_clauses":
                    _ea_handle_register_clauses(state, msg, log_fp)
                elif tag == "scores_req":
                    if getattr(args, 'use_server_queries', False):
                        # Optional: reset old SAT evals when a new scores_req arrives
                        state.last_sat_eval = {}
                        state.last_sat_result = None
                        # Defer scoring until we finish the optional query window
                        state.pending_scores = {
                            "req_ids": [int(i) for i in msg.get("clause_ids", [])],
                            "component": msg.get("component"),
                            "component_id": msg.get("component_id"),
                        }
                        if log_fp is not None:
                            log_fp.write(json.dumps(msg, ensure_ascii=False) + msg_delim)
                            log_fp.flush()
                    else:
                        # Store the scores request for processing after server_queries_start/end
                        state.pending_scores = {
                            "req_ids": [int(i) for i in msg.get("clause_ids", [])],
                            "component": msg.get("component"),
                            "component_id": msg.get("component_id"),
                            "msg": msg
                        }
                        if log_fp is not None:
                            log_fp.write(json.dumps(msg, ensure_ascii=False) + msg_delim)
                            log_fp.flush()
                elif tag == "server_queries_start":
                    if getattr(args, 'use_server_queries', False) and state.pending_scores:
                        state.sat_round += 1
                        do_sat = (state.sat_round % SAT_EVAL_EVERY == 0)
                        if do_sat:
                            # Ask iProver to evaluate the current candidates against its SAT model
                            q = {"tag": "cls_sat_eval_gr_req", "clause_ids": state.pending_scores["req_ids"]}
                            if log_fp is not None:
                                log_fp.write(json.dumps(msg, ensure_ascii=False) + msg_delim)
                                log_fp.write(json.dumps(q, ensure_ascii=False) + msg_delim)
                                log_fp.flush()
                            _ea_send(conn, q, msg_delim)
                            # Will send server_queries_end after *_res
                        else:
                            # Throttled: end query window immediately and proceed to scoring
                            end = {"tag": "server_queries_end"}
                            if log_fp is not None:
                                log_fp.write(json.dumps(msg, ensure_ascii=False) + msg_delim)
                                log_fp.write(json.dumps(end, ensure_ascii=False) + msg_delim)
                                log_fp.flush()
                            _ea_send(conn, end, msg_delim)
                            # Process pending scores right away
                            if state.pending_scores:
                                pend = state.pending_scores
                                msg2 = pend.get("msg") or {
                                    "tag": "scores_req",
                                    "clause_ids": pend["req_ids"],
                                    "component": pend.get("component"),
                                    "component_id": pend.get("component_id"),
                                }
                                res = _ea_handle_scores_req(state, msg2, args, log_fp)
                                _ea_send(conn, res, msg_delim)
                                state.pending_scores = None
                    else:
                        # No server queries needed, send end immediately and then process scores
                        end = {"tag": "server_queries_end"}
                        if log_fp is not None:
                            log_fp.write(json.dumps(msg, ensure_ascii=False) + msg_delim)
                            log_fp.write(json.dumps(end, ensure_ascii=False) + msg_delim)
                            log_fp.flush()
                        _ea_send(conn, end, msg_delim)

                        # Now process the pending scores request
                        if state.pending_scores:
                            pend = state.pending_scores
                            msg2 = pend.get("msg") or {
                                "tag": "scores_req",
                                "clause_ids": pend["req_ids"],
                                "component": pend.get("component"),
                                "component_id": pend.get("component_id"),
                            }
                            res = _ea_handle_scores_req(state, msg2, args, log_fp)
                            _ea_send(conn, res, msg_delim)
                            state.pending_scores = None
                elif tag == "cls_sat_eval_gr_res":
                    # Some older builds use a typo 'cause_ids' — fall back if needed
                    ids = msg.get("clause_ids") or msg.get("cause_ids") or []
                    try:
                        cids = [int(x) for x in ids]
                    except Exception:
                        cids = []
                    vals = msg.get("sat_lit_gr_vals", []) or []
                    state.last_sat_eval = {cid: (vals[i] if i < len(vals) else []) for i, cid in enumerate(cids)}
                    # Finish the query window
                    end = {"tag": "server_queries_end"}
                    if log_fp is not None:
                        log_fp.write(json.dumps(end, ensure_ascii=False) + msg_delim)
                        log_fp.flush()
                    _ea_send(conn, end, msg_delim)
                    # Now score and respond to the earlier scores_req
                    if state.pending_scores:
                        pend = state.pending_scores
                        # Synthesize a scores_req message so we can reuse the handler
                        msg2 = {
                            "tag": "scores_req",
                            "clause_ids": pend["req_ids"],
                            "component": pend.get("component"),
                            "component_id": pend.get("component_id"),
                        }
                        res = _ea_handle_scores_req(state, msg2, args, log_fp)
                        _ea_send(conn, res, msg_delim)
                        state.pending_scores = None
                # Detect finish/timeout from iProver and request shutdown
                elif tag in ("szs_result_out", "szs_status_out", "proof_out"):
                    # Many builds emit szs_result_out with a 'status' field; proof_out usually follows Unsatisfiable
                    status = msg.get("status") or msg.get("szs_status") or msg.get("result")
                    try:
                        print(f"[EA] finish signal received (tag={tag}, status={status}), exiting", flush=True)
                    except Exception:
                        pass
                    if log_fp is not None:
                        log_fp.write(json.dumps(msg, ensure_ascii=False) + msg_delim)
                        log_fp.flush()
                    if getattr(args, 'exit_on_finish', True):
                        shutdown_event.set()
                        break
                # passive_clauses / given_clause / simplified_clauses 等可选消息，此处无需回包
                else:
                    # Still append to log file for completeness
                    if log_fp is not None:
                        try:
                            log_fp.write(json.dumps(msg, ensure_ascii=False) + msg_delim)
                            log_fp.flush()
                        except Exception:
                            pass
            # Connection ended; if configured, exit as well
            if getattr(args, 'exit_on_finish', True) and not shutdown_event.is_set():
                try:
                    print("[EA] connection closed by iProver; exiting", flush=True)
                except Exception:
                    pass
                shutdown_event.set()

    while True:
        if shutdown_event.is_set():
            break
        try:
            conn, addr = sock.accept()
        except socket.timeout:
            continue
        threading.Thread(target=handle, args=(conn, addr), daemon=True).start()

    # Cleanup and exit
    try:
        sock.close()
    except Exception:
        pass
    if log_fp is not None:
        try:
            log_fp.close()
        except Exception:
            pass
    if _stdout_f is not None:
        try:
            _stdout_f.close()
        except Exception:
            pass
    sys.exit(0)


def main(argv: Optional[List[str]] = None) -> None:
    parser = argparse.ArgumentParser(description="Canonicalise iProver interactive log for LLM input OR run EA server")
    sub = parser.add_subparsers(dest="cmd", required=True)

    # convert (original behavior)
    p_conv = sub.add_parser("convert", help="Parse raw log -> canonical JSON (original mode)")
    p_conv.add_argument('--raw-log', type=str, required=True, help="Path to iprover_raw.log")
    p_conv.add_argument('--context-size', type=int, default=128, help="Number of context clauses to include")
    p_conv.add_argument('--candidate-size', type=int, default=128, help="Number of candidate clauses to score")
    p_conv.add_argument('--mapping-scope', type=str, default='all', choices=['all', 'batch'], help="Build symbol mapping on all registered clauses or only on the selected batch")
    p_conv.add_argument('--output', type=str, default='json', choices=['json', 'pretty'], help="Output format")
    p_conv.add_argument('--include-ast', action='store_true', help='Include canonical_formula_ast in JSON output (omitted by default)')

    # serve (EA server)
    p_srv = sub.add_parser("serve", help="Run EA server for iProver interactive mode")
    p_srv.add_argument('--host', type=str, default='127.0.0.1')
    p_srv.add_argument('--port', type=int, default=12345)
    p_srv.add_argument('--ranker-script', type=str, default='/Users/songkunwei/Desktop/LLM/batch_ranker.py', help='Path to batch_ranker.py')
    p_srv.add_argument('--python-exec', dest='python_exec', type=str, default=None,
                       help='Python interpreter to run the ranker (default: same as EA sys.executable)')
    p_srv.add_argument('--model', type=str, default='gpt-5', help='LLM model for ranking (passed to ranker)')
    p_srv.add_argument('--chunk-size', type=int, default=64)
    p_srv.add_argument('--artifacts-dir', type=str, default=None,
                   help='If set, persist each scores_req artifacts for BOTH EA and ranker under this directory. \n'
                        'EA: raw I/O (if --log-file), per-request batch JSON. Ranker: summary prompt/response, chunk prompts and raw responses, out_scores.json.')
    p_srv.add_argument('--anchors', type=int, default=8)
    p_srv.add_argument('--context-summary-k', type=int, default=64)
    p_srv.add_argument('--summary-max-tokens', type=int, default=500)
    p_srv.add_argument('--dry-run', action='store_true')
    p_srv.add_argument('--progress', action='store_true')
    p_srv.add_argument('--verbose', action='store_true')
    p_srv.add_argument('--save-prompts', action='store_true')
    p_srv.add_argument('--use-server-queries', action='store_true',
                       help='Between scores_req and scores_res, request cls_sat_eval_gr from iProver and feed results into the ranker')
    p_srv.add_argument('--include-sat-eval', dest='include_sat_eval', action='store_true',
                       help='Include cls_sat_eval_gr results in batch JSON as ea_query_features.sat_lit_gr_vals')
    p_srv.add_argument('--delimiter', type=str, choices=['nul','newline'], default='nul',
                       help='Message delimiter when sending to iProver (default NUL). Use newline for older/debug tools that dislike NUL.')
    p_srv.add_argument('--echo-clause-ids', dest='echo_clause_ids', action='store_true',
                       help='Include clause_ids in scores_res for explicit mapping.')
    p_srv.add_argument('--context-size', type=int, default=128, help='Context size used when building each batch for a scores_req')
    p_srv.add_argument('--log-file', type=str, default=None, help='Append raw request/response JSON (NUL-separated) to this file')
    # Auto-exit on finish/timeout from iProver (default: enabled)
    p_srv.add_argument('--exit-on-finish', dest='exit_on_finish', action='store_true', default=True,
                       help='Exit EA automatically when iProver sends final status (e.g., szs_result_out/proof_out) or closes the connection')
    p_srv.add_argument('--no-exit-on-finish', dest='exit_on_finish', action='store_false',
                       help='Do not exit EA automatically when iProver finishes')

    args = parser.parse_args(argv)

    if args.cmd == 'convert':
        clauses, clause_ids, component = parse_raw_log(args.raw_log)
        dataset = build_canonical_dataset(
            clauses, clause_ids, args.context_size, args.candidate_size,
            mapping_scope=args.mapping_scope, component=component,
            include_ast=args.include_ast
        )
        if args.output == 'json':
            with open('iprover_llm_output.json', 'w', encoding='utf-8') as f:
                json.dump(dataset, f, indent=2, ensure_ascii=False)
            print("输出已保存到 iprover_llm_output.json")
        else:
            print("Symbol mapping:")
            for canon, info in sorted(dataset['symbol_map'].items()):
                originals = ', '.join(info['original'])
                kind = info.get('kind')
                arity = info.get('arity')
                print(f"  {canon} (kind={kind}, arity={arity}): {originals}")
            if dataset['conjecture']:
                print("\nConjecture:")
                conj = dataset['conjecture']
                print(f"  ID {conj['id']}: {conj['canonical_formula']}")
            print("\nContext clauses:")
            for c in dataset['context_clauses'][:5]:
                print(f"  ID {c['id']}: {c['canonical_formula']}")
            print("\nCandidate clauses:")
            for c in dataset['candidate_clauses'][:5]:
                print(f"  ID {c['id']}: {c['canonical_formula']}")

    elif args.cmd == 'serve':
        run_ea_server(args.host, args.port, args)


if __name__ == '__main__':
    main()