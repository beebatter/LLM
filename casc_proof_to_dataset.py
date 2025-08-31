
"""
casc_proof_to_dataset.py
------------------------
Parse CASC J12 iProver 3.9 result pages (local .txt or fetched) and extract *clause-level* positives
from the proof block, emitting your training JSONL with the requested schema.

Output schema per line:
{
  "problem_name": "...",
  "conjecture_text": "...",          # best-effort (from proof FOF 'conjecture' / 'negated_conjecture')
  "text": "clause/cnf text ...",     # the clause formula
  "features": {"horn":1,"epr":1,"unit":0,"born":12,"conj_dist":2},
  "label": 1,
  "neg_bucket": null,
  "source": "web_refutation",
  "sample_weight": 1.0,
  "item_id": "c_210",
  "role": "plain"
}

Notes
- Prefer parsing clause-level entries present in the proof block:
  * "cnf(...)." (untyped CNF)
  * "tcf(...)." (typed clause form; treated as clause-level too)
- We also parse "fof(...)." only to:
  * map f-ids (e.g., f544) to roles (axiom/conjecture/negated_conjecture/...)
  * extract a human-readable conjecture_text
- "conj_dist" is computed as the minimal backward distance in the proof graph from a clause to any FOF
  whose role is conjecture/negated_conjecture. If unreachable, we set -1.
- "born" defaults to the numeric part of the clause id (e.g., c_210 -> 210), falling back to order index.

This script uses only Python stdlib.
"""

import argparse
import json
import os
import re
import sys
import urllib.request
from typing import List, Dict, Optional, Tuple
import time
from collections import deque
from urllib.parse import urljoin

DEFAULT_BASE = "https://tptp.org/CASC/J12/WWWFiles/Results"
DEFAULT_SOLVER = "iProver---3.9"

# Proof window markers
START_RE = re.compile(r"%\s*SZS\s+output\s+start\s+.*Refutation", re.IGNORECASE)
END_RE   = re.compile(r"%\s*SZS\s+output\s+end\s+.*Refutation", re.IGNORECASE)

# Header regexes for quick id/role pickup
FOF_HEAD_RE = re.compile(r"\bfof\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)
CNF_HEAD_RE = re.compile(r"\bcnf\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)
TCF_HEAD_RE = re.compile(r"\btcf\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)

# For extracting inference parent ids
INF_PARENTS_RE = re.compile(r"inference\s*\(\s*[^)]*?\[([^\]]*)\]\s*\)", re.IGNORECASE|re.DOTALL)

def fetch(url: str, timeout: int = 30) -> Optional[str]:
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None

def read_local(path: str) -> Optional[str]:
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            return f.read()
    except Exception:
        return None

def clean_line_noise(s: str) -> str:
    # Remove report time prefixes like "315.96/41.88\t" from the beginning of lines
    return re.sub(r'(?m)^\s*\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*\t?', '', s)

def locate_proof_window(text: str) -> Optional[str]:
    m = START_RE.search(text)
    if not m:
        return None
    start = m.end()
    m2 = END_RE.search(text, start)
    end = m2.start() if m2 else len(text)
    return text[start:end]

def iter_items_balanced(body: str, kind: str):
    """Yield raw blocks for 'kind' (fof|cnf|tcf) items terminated by a period.
       Balanced-parentheses scan; robust to multi-line formulas.
    """
    s = body
    token = f"{kind}("
    i, n = 0, len(s)
    while True:
        p = s.find(token, i)
        if p == -1: break
        j = p + len(kind)
        if j >= n or s[j] != '(':
            i = p + 1
            continue
        depth = 1
        k = j + 1
        while k < n and depth > 0:
            ch = s[k]
            if ch == '(':
                depth += 1
            elif ch == ')':
                depth -= 1
            k += 1
        if k < n and s[k] == '.':
            k += 1
        yield s[p:k].strip()
        i = k

def extract_formula_from_raw(raw: str) -> str:
    """Extract 3rd argument (formula) from raw 'fof|cnf|tcf(...)' block."""
    open_pos = raw.find('(')
    if open_pos == -1:
        return ""
    # scan commas at depth 1 (arguments of outer functor)
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
                if j < len(raw) and raw[j] == '(':
                    third_start = j
                break
    if third_start == -1:
        return ""
    # read balanced parens for 3rd arg
    d, k = 0, third_start
    out = []
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
    inner = "".join(out)
    if inner.startswith('(') and inner.endswith(')'):
        inner = inner[1:-1].strip()
    return inner

def parse_parents_from_inference(raw: str) -> List[str]:
    m = INF_PARENTS_RE.search(raw)
    if not m:
        return []
    # Parents could be f### or c_### etc.
    return re.findall(r"\b([cf][A-Za-z0-9_]*\d+)\b", m.group(1))

def split_top_level_disj(formula: str) -> List[str]:
    parts = []
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

def feat_is_horn(formula: str) -> int:
    # at most one positive literal
    lits = split_top_level_disj(formula)
    pos = 0
    for lit in lits:
        t = lit.strip()
        if t.startswith('~'):
            continue
        # treat "A != B" as negative equality
        if '!=' in t and not t.startswith('~'):
            continue
        pos += 1
        if pos > 1:
            return 0
    return 1

def feat_is_epr(formula: str) -> int:
    # heuristic: if any '(' occurs at depth >= 2 (nested function terms), mark non-EPR
    depth = 0
    for ch in formula:
        if ch == '(':
            depth += 1
            if depth >= 2:
                return 0
        elif ch == ')':
            depth -= 1
    return 1

def compute_conj_dist(c_parents: Dict[str, List[str]], fof_roles: Dict[str, str]) -> Dict[str, int]:
    """Minimal backward steps from clause cid to any FOF with role conjecture/negated_conjecture.
       If unreachable, -1.
    """
    conj_f = {fid for fid, r in fof_roles.items() if r in ('conjecture', 'negated_conjecture')}
    dist = {}
    for cid in c_parents.keys():
        seen = set()
        q = deque([(cid, 0)])
        hit = None
        while q:
            node, d = q.popleft()
            if node in seen:
                continue
            seen.add(node)
            pars = c_parents.get(node, [])
            # direct FOF ancestor that is conjecture?
            if any(p.startswith('f') and p in conj_f for p in pars):
                hit = d + 1
                break
            # otherwise traverse clause parents
            for p in pars:
                if p.startswith('c'):
                    q.append((p, d + 1))
        dist[cid] = hit if hit is not None else -1
    return dist

def build_url(base: str, division: str, solver: str, problem_name: str) -> str:
    return f"{base.rstrip('/')}/{division}/{solver}/{problem_name}"

def try_load_problem(problem_name: str,
                     divisions: List[str],
                     base: str,
                     solver: str,
                     proof_dir: Optional[str],
                     from_web: bool,
                     verbose: bool = False) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Try local file '<proof_dir>/<problem>.txt', else fetch from web across divisions.
       Return (text, division, url). division could be None for local.
    """
    if proof_dir:
        local_path = os.path.join(proof_dir, f"{problem_name}.txt")
        txt = read_local(local_path)
        if txt:
            if verbose:
                print(f"[load] {problem_name}: local file OK -> {local_path}")
            return txt, None, f"file://{os.path.abspath(local_path)}"
    if from_web:
        for div in divisions:
            for cand in (problem_name, f"{problem_name}.p"):
                url = build_url(base, div, solver, cand)
                if verbose:
                    print(f"[fetch] {problem_name}: trying {url}")
                txt = fetch(url)
                if txt and "% SZS" in txt:
                    if verbose:
                        print(f"[fetch] {problem_name}: fetched from {div} as {cand}")
                    return txt, div, url
                else:
                    if verbose:
                        print(f"[fetch] {problem_name}: not found in {div} as {cand}")
    return None, None, None

def process_single_problem(name: str, txt: str, division: Optional[str], url: Optional[str], out_f, verbose: bool = False) -> Tuple[int, Optional[str]]:
    cleaned = clean_line_noise(txt)
    body = locate_proof_window(cleaned)
    if not body:
        if verbose:
            print(f"[skip] {name}: no SZS refutation window found")
        return 0, "no_szs_refutation_window"

    # map FOF ids -> roles
    fof_roles = {}
    for m in FOF_HEAD_RE.finditer(body):
        fid, role = m.group(1), m.group(2).lower()
        fof_roles[fid] = role

    # grab a conjecture_text (prefer 'conjecture', fallback to 'negated_conjecture')
    conjecture_text = ""
    # iterate over actual FOF items to extract formula bodies
    for raw in iter_items_balanced(body, "fof"):
        m = FOF_HEAD_RE.search(raw)
        if not m: 
            continue
        role = m.group(2).lower()
        if role in ("conjecture", "negated_conjecture"):
            conjecture_text = extract_formula_from_raw(raw)
            if conjecture_text:
                break

    positives = 0

    # parse clause-level items (CNF and TCF)
    clause_heads = [("cnf", CNF_HEAD_RE), ("tcf", TCF_HEAD_RE)]
    # collect parents map for conj_dist
    c_parents = {}
    c_rows = []

    kind_counts = {"cnf": 0, "tcf": 0}
    for kind, head_re in clause_heads:
        for raw in iter_items_balanced(body, kind):
            m = head_re.search(raw)
            if not m: 
                continue
            cid, role = m.group(1), m.group(2).lower()
            formula = extract_formula_from_raw(raw)
            parents = parse_parents_from_inference(raw)
            c_parents[cid] = parents
            kind_counts[kind] += 1

            # features
            # born: numeric part of cid if present, else enumerate later
            mnum = re.search(r"\d+", cid)
            born = int(mnum.group(0)) if mnum else None

            c_rows.append({
                "problem_name": name,
                "division": division,
                "url": url,
                "conjecture_text": conjecture_text,
                "text": formula,
                "features": {
                    # placeholder; conj_dist added after we compute on full graph
                    "horn": 0, "epr": 0, "unit": 0, "born": born if born is not None else -1, "conj_dist": -1
                },
                "label": 1,
                "neg_bucket": None,
                "source": "web_refutation",
                "sample_weight": 1.0,
                "item_id": cid,
                "role": role,
                "item_kind": kind
            })

    # compute conj_dist on the full clause graph
    conj_dist = compute_conj_dist(c_parents, fof_roles)

    # finalize features and emit
    for idx, row in enumerate(c_rows):
        formula = row["text"]
        # fill born if missing
        if row["features"]["born"] == -1:
            row["features"]["born"] = idx
        row["features"]["unit"] = 1 if feat_is_unit(formula) else 0
        row["features"]["horn"] = 1 if feat_is_horn(formula) else 0
        row["features"]["epr"]  = 1 if feat_is_epr(formula) else 0
        row["features"]["conj_dist"] = conj_dist.get(row["item_id"], -1)
        if out_f is not None:
            out_f.write(json.dumps(row, ensure_ascii=False) + "\n")
        positives += 1

    if verbose:
        conj_short = (conjecture_text[:80] + '...') if conjecture_text and len(conjecture_text) > 80 else conjecture_text
        print(f"[emit] {name}: cnf={kind_counts['cnf']} tcf={kind_counts['tcf']} -> rows={positives}; conjecture_len={len(conjecture_text) if conjecture_text else 0}")

    return positives, None

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True,
                    help=(
                        "Path to a problem list. Supported formats: "
                        "(1) JSON with key 'problems' listing objects with 'problem_name'; "
                        "(2) JSON array of objects or strings; "
                        "(3) JSONL where each line has 'problem_name'; "
                        "(4) Plain text with one problem name per line."
                    ))
    ap.add_argument("--proof-dir", default=None,
                    help="Directory of saved CASC result pages named '<problem_name>.txt'.")
    ap.add_argument("--from-web", action="store_true",
                    help="Fetch from web when local file not found.")
    ap.add_argument("--divisions", nargs="+", default=["FNE", "FEQ"],
                    help="Divisions to try when fetching (default: FNE FEQ).")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="CASC base url.")
    ap.add_argument("--solver", default=DEFAULT_SOLVER, help="Solver subdir.")
    ap.add_argument("-o", "--out", default="positives_from_casc.jsonl", help="Output JSONL path.")
    ap.add_argument("--verbose", action="store_true", help="Print intermediate progress logs.")
    ap.add_argument("--log-every", type=int, default=25, help="Print progress every N problems (in addition to --verbose).")
    ap.add_argument("--miss-report", default=None, help="Optional path to write per-problem miss reasons (TSV: problem_name\treason).")
    ap.add_argument("--report-only", action="store_true", help="Do not write positives, only compute and write miss report.")
    args = ap.parse_args()

    def load_problem_names(path: str, verbose: bool = False) -> List[str]:
        # Try JSON object/array first
        try:
            with open(path, "r", encoding="utf-8") as f:
                obj = json.load(f)
            names: List[str] = []
            if isinstance(obj, dict):
                if "problems" in obj and isinstance(obj["problems"], list):
                    for p in obj["problems"]:
                        if isinstance(p, dict) and "problem_name" in p:
                            names.append(p["problem_name"])
                        elif isinstance(p, str):
                            names.append(p)
                elif "problem_name" in obj and isinstance(obj["problem_name"], str):
                    names.append(obj["problem_name"])
            elif isinstance(obj, list):
                for p in obj:
                    if isinstance(p, dict) and "problem_name" in p:
                        names.append(p["problem_name"])
                    elif isinstance(p, str):
                        names.append(p)
            if names:
                return names
        except json.JSONDecodeError:
            # Fall back to JSONL or plain text
            pass
        except Exception:
            pass

        # Try JSONL: one JSON object per line with problem_name
        names = []
        try:
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = json.loads(line)
                        if isinstance(rec, dict):
                            name = rec.get("problem_name") or rec.get("problem")
                            if isinstance(name, str):
                                names.append(name)
                                continue
                    except Exception:
                        # Not JSON - treat as plain text
                        pass
                    # Plain text fallback: accept the line as a problem name if no JSON parsed
                    if line and not line.startswith("{"):
                        names.append(line)
        except Exception:
            return []
        return names

    names = load_problem_names(args.problems, verbose=args.verbose)
    if not names:
        print(f"[error] Could not parse problem names from: {args.problems}", file=sys.stderr)
        sys.exit(2)

    total_problems = 0
    total_items = 0
    attempted = 0
    t0 = time.perf_counter()

    misses: List[Tuple[str, str]] = []  # (problem_name, reason)

    out_f_handle = None
    if not args.report_only:
        out_f_handle = open(args.out, "w", encoding="utf-8")
    try:
        for idx, name in enumerate(names, 1):
            if args.verbose or (args.log_every and idx % args.log_every == 1):
                print(f"[progress] {idx}/{len(names)}: {name}")
            txt, div, url = try_load_problem(
                problem_name=name,
                divisions=args.divisions,
                base=args.base_url,
                solver=args.solver,
                proof_dir=args.proof_dir,
                from_web=args.from_web,
                verbose=args.verbose
            )
            if not txt:
                if args.verbose:
                    print(f"[miss] {name}: no result found (local/web)")
                misses.append((name, "not_found_local_web"))
                continue
            attempted += 1
            items, reason = process_single_problem(name, txt, div, url, out_f_handle, verbose=args.verbose)
            if items and items > 0:
                total_problems += 1
                total_items += items
                if args.verbose:
                    print(f"[ok] {name}: emitted {items} rows")
            else:
                # fetched but could not emit (e.g., no SZS window)
                misses.append((name, reason or "no_items"))
        # write miss report if requested
        if args.miss_report:
            with open(args.miss_report, "w", encoding="utf-8") as mf:
                for n, r in misses:
                    mf.write(f"{n}\t{r}\n")
            if args.verbose:
                print(f"[report] wrote miss report: {args.miss_report} (count={len(misses)})")
    finally:
        if out_f_handle is not None:
            out_f_handle.close()

    dt = time.perf_counter() - t0
    print(f"[done] problems_emitted={total_problems}, positives_emitted={total_items}, attempted={attempted}, total={len(names)}, time_sec={dt:.2f}, out={args.out}")

if __name__ == "__main__":
    main()
