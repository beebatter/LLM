"""
other_prover_casc_to_cnf_dataset.py
-----------------------------------
Fetch CASC result pages (local or web), extract cnf/tcf clauses from SZS Proof/CNFRefutation
windows, and if only fof(...) items are present, clausify them via an external tool
(user template or automatic fallbacks: Vampire -> tptp4X -> E release -> eprover-preprocess).

Also supports parsing a local Results index HTML (FOF division) to get per-problem links and
trying solvers in fallback order iProver -> Vampire 4.9 -> E 3.2.0.

Outputs JSONL with teacher-style records and includes proof_solver to indicate which solver
page the proof came from.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Dict, List, Optional, Tuple

try:
    import urllib.request  # type: ignore
except Exception:  # pragma: no cover
    urllib = None  # type: ignore


# Defaults
DEFAULT_BASE = "https://tptp.org/CASC/J12/WWWFiles/Results"
DEFAULT_PROVERS = ["Vampire---4.9", "E---3.2.0"]


# Regexes
START_RE = re.compile(r"%\s*SZS\s+output\s+start\s+(?:CNFRefutation|Proof)\b", re.IGNORECASE)
END_RE = re.compile(r"%\s*SZS\s+output\s+end\s+(?:CNFRefutation|Proof)\b", re.IGNORECASE)

FOF_HEAD_RE = re.compile(r"\bfof\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)
CNF_HEAD_RE = re.compile(r"\bcnf\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)
TCF_HEAD_RE = re.compile(r"\btcf\s*\(\s*([A-Za-z0-9_]+)\s*,\s*([a-z_]+)\s*,", re.IGNORECASE)
INF_PARENTS_RE = re.compile(r"inference\s*\(\s*[^)]*?\[([^\]]*)\]\s*\)", re.IGNORECASE | re.DOTALL)


def fetch(url: str, timeout: int = 30) -> Optional[str]:
    if not urllib or not hasattr(urllib, "request"):
        return None
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:  # type: ignore
            return resp.read().decode("utf-8", errors="replace")
    except Exception:
        return None


def parse_parents_from_inference(raw: str) -> List[str]:
    m = INF_PARENTS_RE.search(raw)
    if not m:
        return []
    return re.findall(r"\b([cf][A-Za-z0-9_]*\d+)\b", m.group(1))


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


def feat_is_horn(formula: str) -> int:
    lits = split_top_level_disj(formula)
    pos = 0
    for lit in lits:
        t = lit.strip()
        if t.startswith('~'):
            continue
        if '!=' in t and not t.startswith('~'):
            continue
        pos += 1
        if pos > 1:
            return 0
    return 1


def feat_is_epr(formula: str) -> int:
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
    conj_f = {fid for fid, r in fof_roles.items() if r in ('conjecture', 'negated_conjecture')}
    dist: Dict[str, int] = {}
    from collections import deque
    for cid in c_parents.keys():
        seen = set()
        q = deque([(cid, 0)])
        hit: Optional[int] = None
        while q:
            node, d = q.popleft()
            if node in seen:
                continue
            seen.add(node)
            pars = c_parents.get(node, [])
            if any(p.startswith('f') and p in conj_f for p in pars):
                hit = d + 1
                break
            for p in pars:
                if p.startswith('c'):
                    q.append((p, d + 1))
        dist[cid] = hit if hit is not None else -1
    return dist


def build_url(base: str, division: str, solver: str, problem_name: str) -> str:
    return f"{base.rstrip('/')}/{division}/{solver}/{problem_name}"


def parse_division_from_url(url: str) -> Optional[str]:
    try:
        parts = url.split("/Results/")
        if len(parts) < 2:
            return None
        rest = parts[1]
        return rest.split("/")[0]
    except Exception:
        return None


def parse_index_html(index_path: str) -> Dict[str, Dict[str, str]]:
    """Parse a local CASC FOF results HTML and extract solver result links for
    iProver---3.9, Vampire---4.9, E---3.2.0. Returns: { problem: { solver: url } }
    """
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception:
        return {}
    rx = re.compile(
        r'href="(?P<url>[^"\s]+/Results/[^/]+/(?P<solver>iProver---3\.9|Vampire---4\.9|E---3\.2\.0)/(?P<prob>[A-Za-z0-9_+\-^]+))"',
        re.IGNORECASE,
    )
    out: Dict[str, Dict[str, str]] = {}
    for m in rx.finditer(html):
        url = m.group("url"); solver = m.group("solver"); prob = m.group("prob")
        d = out.setdefault(prob, {})
        if solver not in d:
            d[solver] = url
    return out


def solver_to_source_tag(solver: str) -> str:
    s = solver.lower()
    if "vampire" in s:
        return "teacher_vampire"
    if s == "e" or s.startswith("e---"):
        return "teacher_e"
    if "iprover" in s:
        return "teacher_iprover"
    return f"teacher_{solver.replace('---','_').replace('-','_')}"


def clean_line_noise(s: str) -> str:
    # Remove leading timing prefixes like "3.09/1.93\t"
    return re.sub(r'(?m)^\s*\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*\t?', '', s)


def try_fetch_result(
    problem_name: str,
    divisions: List[str],
    base: str,
    solver: str,
    *,
    verbose: bool = False,
    proof_dir: Optional[str] = None,
    urls_override: Optional[List[str]] = None,
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    # Local file first
    if proof_dir:
        for cand in (problem_name, f"{problem_name}.txt", f"{problem_name}.p", f"{problem_name}.out"):
            local_path = os.path.join(proof_dir, cand)
            if os.path.exists(local_path):
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        return f.read(), "LOCAL", local_path
                except Exception:
                    pass

    # Direct URLs (from index) if provided
    if urls_override:
        for u in urls_override:
            if verbose:
                print(f"[fetch] {problem_name} @ {solver}: {u}")
            txt = fetch(u)
            if txt and "% SZS" in txt:
                return txt, parse_division_from_url(u), u

    # Web fallback: constructed URLs
    for div in divisions:
        for cand in (problem_name, f"{problem_name}.p"):
            url = build_url(base, div, solver, cand)
            if verbose:
                print(f"[fetch] {problem_name} @ {solver}: {url}")
            txt = fetch(url)
            if txt and "% SZS" in txt:
                return txt, div, url
    return None, None, None


def _cmd_exists(cmd: str) -> bool:
    return bool(shutil.which(cmd))


def _is_e_debug() -> bool:
    if not _cmd_exists("eprover"):
        return False
    try:
        out = subprocess.check_output(["eprover", "--version"], text=True, stderr=subprocess.STDOUT, timeout=3)
        return "DEBUG" in out.upper()
    except Exception:
        return False


def _parse_cnf_blocks_from_text(content: str) -> List[str]:
    blocks: List[str] = []
    s = content
    token = "cnf("
    i, n = 0, len(s)
    while True:
        p = s.find(token, i)
        if p == -1:
            break
        j = p + 3
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
        blocks.append(s[p:k].strip())
        i = k
    return blocks


def run_external_clausifier(
    fof_items: List[str],
    cmd_template: Optional[str],
    tmpdir: str,
    *,
    verbose: bool = False,
) -> List[str]:
    # Write input
    in_path = os.path.join(tmpdir, "in_fof.p")
    out_path = os.path.join(tmpdir, "out_cnf.p")
    with open(in_path, "w", encoding="utf-8") as f:
        for raw in fof_items:
            line = raw.strip()
            if not line.endswith('.'):
                line += '.'
            f.write(line + "\n")

    # Build candidate commands: user template -> Vampire -> tptp4X -> E -> eprover-preprocess
    candidates: List[str] = []
    if cmd_template and "{in}" in cmd_template and "{out}" in cmd_template:
        tmpl = re.sub(r"--output-file\s+", "--output-file=", cmd_template)
        candidates.append(tmpl)

    vamp = shutil.which("vampire") or ("/usr/local/bin/vampire" if os.path.exists("/usr/local/bin/vampire") else None)
    if vamp:
        candidates.append(f"{vamp} --mode clausify --input_syntax tptp --output_syntax tptp -t 10 {{in}} > {{out}}")

    tptp4x = shutil.which("tptp4X") or ("/home/ks/TPTP4X/tptp4X" if os.path.exists("/home/ks/TPTP4X/tptp4X") else None)
    if tptp4x:
        candidates.append(f"{tptp4x} -x cnf -f tptp {{in}} > {{out}}")

    if _cmd_exists("eprover") and not _is_e_debug():
        candidates.append(
            "eprover --tstp-in --tstp-out --cnf --output-level=0 --cpu-limit=10 --memory-limit=1024 "
            "--output-file={out} {in}"
        )
    if _cmd_exists("eprover-preprocess"):
        candidates.append("eprover-preprocess --tstp-in --tstp-out --cnf --output-level=0 --output-file={out} {in}")

    if not candidates:
        if verbose:
            print("[clausify] no available clausifier in PATH")
        return []

    if verbose:
        print("[clausify] candidates:")
        for i, c in enumerate(candidates):
            print(f"  [{i}] {c}")

    for idx, tmpl in enumerate(candidates):
        tmpl2 = re.sub(r"--output-file\s+", "--output-file=", tmpl)
        shell_cmd = tmpl2.format_map({"in": in_path, "out": out_path})
        if verbose:
            print(("[clausify] running" if idx == 0 else "[clausify] fallback") + f": {shell_cmd}")
        try:
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            subprocess.run(shell_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"[clausify] command failed (rc={e.returncode})")
            continue
        # Read output
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            continue
        if content and "cnf(" in content:
            blocks = _parse_cnf_blocks_from_text(content)
            if blocks:
                return blocks
        else:
            if verbose:
                print("[clausify] no cnf(...) in output")
    return []


def process_one(
    problem_name: str,
    text: str,
    solver: str,
    division: Optional[str],
    url: Optional[str],
    out_f,
    *,
    sample_weight: float,
    verbose: bool = False,
    clausify_cmd: Optional[str] = None,
) -> int:
    cleaned = clean_line_noise(text)
    body = locate_proof_window(cleaned)
    if not body:
        body = cleaned
        if verbose:
            print(f"[info] {problem_name} @ {solver}: no SZS window; scanning entire file")

    fof_roles: Dict[str, str] = {}
    for m in FOF_HEAD_RE.finditer(body):
        fid, role = m.group(1), m.group(2).lower()
        fof_roles[fid] = role

    conjecture_text = ""
    fof_items_raw: List[str] = []
    for raw in iter_items_balanced(body, "fof"):
        fof_items_raw.append(raw)
        m = FOF_HEAD_RE.search(raw)
        if m and m.group(2).lower() in ("conjecture", "negated_conjecture"):
            form = extract_formula_from_raw(raw)
            if form:
                conjecture_text = form

    total = 0
    c_parents: Dict[str, List[str]] = {}
    rows: List[Dict] = []
    for kind, head_re in (("cnf", CNF_HEAD_RE), ("tcf", TCF_HEAD_RE)):
        for raw in iter_items_balanced(body, kind):
            m = head_re.search(raw)
            if not m:
                continue
            cid, role = m.group(1), m.group(2).lower()
            formula = extract_formula_from_raw(raw)
            parents = parse_parents_from_inference(raw)
            c_parents[cid] = parents
            mnum = re.search(r"\d+", cid)
            born = int(mnum.group(0)) if mnum else -1
            rows.append({
                "problem_name": problem_name,
                "division": division,
                "url": url,
                "conjecture_text": conjecture_text,
                "text": formula,
                "features": {"horn": 0, "epr": 0, "unit": 0, "born": born, "conj_dist": -1},
                "label": 1,
                "neg_bucket": None,
                "source": solver_to_source_tag(solver),
                "proof_solver": solver,
                "sample_weight": float(sample_weight),
                "item_id": cid,
                "role": role,
                "item_kind": kind,
            })

    # If no clause-level items but have FOF, try clausification automatically
    if not rows and fof_items_raw:
        with tempfile.TemporaryDirectory() as tmpd:
            cnf_blocks = run_external_clausifier(fof_items_raw, clausify_cmd, tmpd, verbose=verbose)
        for raw in cnf_blocks:
            m = CNF_HEAD_RE.search(raw)
            if not m:
                continue
            cid, role = m.group(1), m.group(2).lower()
            formula = extract_formula_from_raw(raw)
            rows.append({
                "problem_name": problem_name,
                "division": division,
                "url": url,
                "conjecture_text": conjecture_text,
                "text": formula,
                "features": {"horn": 0, "epr": 0, "unit": 0, "born": -1, "conj_dist": -1},
                "label": 1,
                "neg_bucket": None,
                "source": solver_to_source_tag(solver),
                "proof_solver": solver,
                "sample_weight": float(sample_weight),
                "item_id": cid,
                "role": role,
                "item_kind": "cnf",
            })

    if not rows:
        if verbose:
            print(f"[miss] {problem_name} @ {solver}: no clause-level items (and no clausify)")
        return 0

    conj_dist = {}
    if rows and c_parents:
        conj_dist = compute_conj_dist(c_parents, fof_roles)

    for idx, r in enumerate(rows):
        if r["features"]["born"] == -1:
            r["features"]["born"] = idx
        form = r["text"]
        r["features"]["unit"] = 1 if feat_is_unit(form) else 0
        r["features"]["horn"] = 1 if feat_is_horn(form) else 0
        r["features"]["epr"] = 1 if feat_is_epr(form) else 0
        if conj_dist:
            r["features"]["conj_dist"] = conj_dist.get(r["item_id"], -1)
        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += 1
    return total


def load_problem_names(path: str) -> List[str]:
    # Try JSON or JSONL first
    try:
        with open(path, "r", encoding="utf-8") as f:
            content = f.read()
        # Try JSON object/array
        try:
            obj = json.loads(content)
            names: List[str] = []
            if isinstance(obj, dict) and "problems" in obj and isinstance(obj["problems"], list):
                for p in obj["problems"]:
                    if isinstance(p, dict) and "problem_name" in p:
                        names.append(p["problem_name"])
                    elif isinstance(p, str):
                        names.append(p)
                return names
            if isinstance(obj, list):
                for p in obj:
                    if isinstance(p, dict) and "problem_name" in p:
                        names.append(p["problem_name"])
                    elif isinstance(p, str):
                        names.append(p)
                return names
        except json.JSONDecodeError:
            # Try JSONL lines
            names = []
            for line in content.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                    if isinstance(rec, dict) and "problem_name" in rec:
                        names.append(rec["problem_name"])
                        continue
                except Exception:
                    pass
                names.append(line)
            return names
    except Exception:
        # Fallback: plain text file
        pass

    names: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    names.append(line)
    except Exception:
        return []
    return names


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=False,
                    help="Path to problem list (txt/json/jsonl; one name per line or objects with 'problem_name').")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--divisions", nargs="+", default=["FNE", "FEQ", "FOF"],
                    help="CASC divisions to try (default: FNE FEQ FOF).")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="CASC base url (J12).")
    ap.add_argument("--provers", nargs="+", default=DEFAULT_PROVERS,
                    help="Which solver directories to query, e.g. 'Vampire---4.9' 'E---3.2.0'.")
    ap.add_argument("--index-html", default=None,
                    help="Local CASC Results HTML (FOF) to parse for solver result links (iProver/Vampire/E).")
    ap.add_argument("--index-as-problems", action="store_true",
                    help="Use all problems found in --index-html as the problem list (ignores --problems file).")
    ap.add_argument("--sample-weight", type=float, default=0.5,
                    help="Sample weight for teacher data (default 0.5).")
    ap.add_argument("--clausify-cmd", default=None,
                    help="External clausifier command template; must contain {in} and {out}.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    ap.add_argument("--miss-report", default=None, help="Optional TSV of (problem,solver,reason).")
    ap.add_argument("--proof-dir", default=None,
                    help="Directory containing local result pages named <problem>.txt/.p; used before web.")

    args = ap.parse_args()

    index_map: Dict[str, Dict[str, str]] = {}
    names: List[str] = []
    if args.index_html:
        index_map = parse_index_html(args.index_html)
        if args.index_as_problems and index_map:
            names = sorted(index_map.keys())
    if not names:
        if not args.problems:
            print("[error] provide --problems or --index-html with --index-as-problems", file=sys.stderr)
            sys.exit(2)
        names = load_problem_names(args.problems)
        if not names:
            src = args.index_html or args.problems
            print(f"[error] could not load problem names from: {src}", file=sys.stderr)
            sys.exit(2)

    misses: List[Tuple[str, str, str]] = []
    count_problems = 0
    count_rows = 0

    with open(args.out, "w", encoding="utf-8") as out_f:
        for name in names:
            emitted_this_problem = 0
            # If index_map provided, prefer links and solver fallback order
            if index_map and name in index_map:
                fallback_order = ["iProver---3.9", "Vampire---4.9", "E---3.2.0"]
                links = index_map[name]
                tried_any = False
                for solver in fallback_order:
                    txt = div = url = None  # type: ignore[assignment]
                    if solver in links:
                        tried_any = True
                        base_url = links[solver]
                        candidate_urls = [base_url]
                        if not base_url.endswith(".p"):
                            candidate_urls.append(base_url + ".p")
                        txt, div, url = try_fetch_result(
                            name, args.divisions, args.base_url, solver,
                            verbose=args.verbose, proof_dir=args.proof_dir, urls_override=candidate_urls,
                        )
                    if not txt:
                        # Constructed URL fallback even if no index link or fetch failed
                        txt, div, url = try_fetch_result(
                            name, args.divisions, args.base_url, solver,
                            verbose=args.verbose, proof_dir=args.proof_dir,
                        )
                    if not txt:
                        misses.append((name, solver, "not_found"))
                        continue
                    rows = process_one(
                        name, txt, solver, div, url, out_f,
                        sample_weight=args.sample_weight, verbose=args.verbose, clausify_cmd=args.clausify_cmd,
                    )
                    if rows > 0:
                        emitted_this_problem += rows
                        # Stop at first solver that yields rows
                        break
                    else:
                        misses.append((name, solver, "no_rows"))
                if not tried_any and args.verbose:
                    print(f"[warn] {name}: no index links for preferred solvers")
            else:
                # Iterate provided solvers and try divisions/base
                for solver in args.provers:
                    txt, div, url = try_fetch_result(
                        name, args.divisions, args.base_url, solver,
                        verbose=args.verbose, proof_dir=args.proof_dir,
                    )
                    if not txt:
                        misses.append((name, solver, "not_found"))
                        continue
                    rows = process_one(
                        name, txt, solver, div, url, out_f,
                        sample_weight=args.sample_weight, verbose=args.verbose, clausify_cmd=args.clausify_cmd,
                    )
                    if rows > 0:
                        emitted_this_problem += rows
                        break
                    else:
                        misses.append((name, solver, "no_rows"))

            if emitted_this_problem > 0:
                count_problems += 1
                count_rows += emitted_this_problem
            elif args.verbose:
                print(f"[warn] {name}: no rows from any solver")

    if args.miss_report:
        try:
            with open(args.miss_report, "w", encoding="utf-8") as f:
                for n, s, r in misses:
                    f.write(f"{n}\t{s}\t{r}\n")
        except Exception as e:
            print(f"[warn] failed to write miss report: {e}", file=sys.stderr)

    print(f"[done] problems_with_any_rows={count_problems}, rows_emitted={count_rows}, out={args.out}")
    if args.miss_report:
        print(f"[report] miss_report={args.miss_report}, count={len(misses)}")


if __name__ == "__main__":
    main()
                    continue
                if '!=' in t and not t.startswith('~'):
                    continue
                pos += 1
                if pos > 1:
                    return 0
            return 1

        def feat_is_epr(formula: str) -> int:
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
            conj_f = {fid for fid, r in fof_roles.items() if r in ('conjecture', 'negated_conjecture')}
            dist = {}
            from collections import deque
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
                    if any(p.startswith('f') and p in conj_f for p in pars):
                        hit = d + 1
                        break
                    for p in pars:
                        if p.startswith('c'):
                            q.append((p, d + 1))
                dist[cid] = hit if hit is not None else -1
            return dist

        def build_url(base: str, division: str, solver: str, problem_name: str) -> str:
            return f"{base.rstrip('/')}/{division}/{solver}/{problem_name}"

        def parse_division_from_url(url: str) -> Optional[str]:
            try:
                parts = url.split("/Results/")
                if len(parts) < 2:
                    return None
                rest = parts[1]
                return rest.split("/")[0]
            except Exception:
                return None

        def parse_index_html(index_path: str) -> Dict[str, Dict[str, str]]:
            """
            Parse a local CASC FOF results HTML and extract solver result links for
            iProver---3.9, Vampire---4.9, E---3.2.0.
            Returns: { problem_name: { solver: url, ... }, ... }
            """
            try:
                with open(index_path, "r", encoding="utf-8") as f:
                    html = f.read()
            except Exception:
                return {}
            rx = re.compile(
                r'href="(?P<url>[^"\s]+/Results/[^/]+/(?P<solver>iProver---3\.9|Vampire---4\.9|E---3\.2\.0)/(?P<prob>[A-Za-z0-9_+\-^]+))"',
                re.IGNORECASE)
            out: Dict[str, Dict[str, str]] = {}
            for m in rx.finditer(html):
                url = m.group("url"); solver = m.group("solver"); prob = m.group("prob")
                d = out.setdefault(prob, {})
                if solver not in d:
                    d[solver] = url
            return out

        def solver_to_source_tag(solver: str) -> str:
            s = solver.lower()
            if "vampire" in s: return "teacher_vampire"
            if s == "e" or s.startswith("e---"): return "teacher_e"
            if "iprover" in s: return "teacher_iprover"
            return f"teacher_{solver.replace('---','_').replace('-','_')}"

        def clean_line_noise(s: str) -> str:
            return re.sub(r'(?m)^\s*\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*\t?', '', s)

        def try_fetch_result(problem_name: str, divisions: List[str], base: str, solver: str,
                             verbose=False, proof_dir: Optional[str]=None,
                             urls_override: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
            # local first if provided
            if proof_dir:
                for cand in (problem_name, f"{problem_name}.txt", f"{problem_name}.p", f"{problem_name}.out"):
                    local_path = os.path.join(proof_dir, cand)
                    if os.path.exists(local_path):
                        try:
                            with open(local_path, "r", encoding="utf-8") as f:
                                txt = f.read()
                            return txt, "LOCAL", local_path
                        except Exception:
                            pass
            # direct URLs override
            if urls_override:
                for u in urls_override:
                    if verbose:
                        print(f"[fetch] {problem_name} @ {solver}: {u}")
                    txt = fetch(u)
                    if txt and "% SZS" in txt:
                        return txt, parse_division_from_url(u), u
            # web fallback
            for div in divisions:
                for cand in (problem_name, f"{problem_name}.p"):
                    url = build_url(base, div, solver, cand)
                    if verbose:
                        print(f"[fetch] {problem_name} @ {solver}: {url}")
                    txt = fetch(url)
                    if txt and "% SZS" in txt:
                        return txt, div, url
            return None, None, None

        def run_external_clausifier(fof_items: List[str], cmd_template: str, tmpdir: str, verbose=False) -> List[str]:
            """
            Try to clausify a set of FOF items using external tools.
            Order: user template -> Vampire (--mode clausify) -> tptp4X -> eprover (release) -> eprover-preprocess.
            Returns list of cnf(...) blocks on success, or [] otherwise.
            """
            if not cmd_template or "{in}" not in cmd_template or "{out}" not in cmd_template:
                cmd_template = ""
            in_path = os.path.join(tmpdir, "in_fof.p")
            out_path = os.path.join(tmpdir, "out_cnf.p")
            with open(in_path, "w", encoding="utf-8") as f:
                for raw in fof_items:
                    line = raw.strip()
                    if not line.endswith('.'):
                        line += '.'
                    f.write(line + "\n")

            candidates: List[str] = []
            import re as _re
            if cmd_template:
                candidates.append(_re.sub(r"--output-file\s+", "--output-file=", cmd_template))

            def _exists(cmd: str) -> bool:
                return bool(shutil.which(cmd))

            # Prefer Vampire clausify if available
            vamp = shutil.which("vampire") or ("/usr/local/bin/vampire" if os.path.exists("/usr/local/bin/vampire") else None)
            if vamp:
                candidates.append(f"{vamp} --mode clausify --input_syntax tptp --output_syntax tptp -t 10 {{in}} > {{out}}")

            # tptp4X fallback
            tptp4x = shutil.which("tptp4X") or ("/home/ks/TPTP4X/tptp4X" if os.path.exists("/home/ks/TPTP4X/tptp4X") else None)
            if tptp4x:
                candidates.append(f"{tptp4x} -x cnf -f tptp {{in}} > {{out}}")

            # E fallbacks (skip DEBUG)
            def _is_e_debug() -> bool:
                if not _exists("eprover"):
                    return False
                try:
                    out = subprocess.check_output(["eprover", "--version"], text=True, stderr=subprocess.STDOUT, timeout=3)
                    return "DEBUG" in out.upper()
                except Exception:
                    return False
            if _exists("eprover") and not _is_e_debug():
                candidates.append("eprover --tstp-in --tstp-out --cnf --output-level=0 --cpu-limit=10 --memory-limit=1024 --output-file={out} {in}")
            if _exists("eprover-preprocess"):
                candidates.append("eprover-preprocess --tstp-in --tstp-out --cnf --output-level=0 --output-file={out} {in}")

            if not candidates:
                if verbose:
                    print("[clausify] no available clausifier in PATH")
                return []

            if verbose:
                print("[clausify] candidates:")
                for i, c in enumerate(candidates):
                    print(f"  [{i}] {c}")

            for idx, tmpl in enumerate(candidates):
                tmpl2 = _re.sub(r"--output-file\s+", "--output-file=", tmpl)
                shell_cmd = tmpl2.format_map({"in": in_path, "out": out_path})
                if verbose:
                    print(("[clausify] running" if idx == 0 else "[clausify] fallback") + f": {shell_cmd}")
                try:
                    if os.path.exists(out_path):
                        try: os.remove(out_path)
                        except Exception: pass
                    subprocess.run(shell_cmd, shell=True, check=True)
                except subprocess.CalledProcessError as e:
                    if verbose:
                        print(f"[clausify] command failed (rc={e.returncode})")
                    continue
                try:
                    with open(out_path, "r", encoding="utf-8") as f:
                        content = f.read()
                except Exception:
                    continue
                if content and "cnf(" in content:
                    s = content
                    cnf_blocks = []
                    token = "cnf("
                    i, n = 0, len(s)
                    while True:
                        p = s.find(token, i)
                        if p == -1:
                            break
                        j = p + 3
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
                        cnf_blocks.append(s[p:k].strip())
                        i = k
                    if cnf_blocks:
                        return cnf_blocks
                else:
                    if verbose:
                        print("[clausify] no cnf(...) in output")
            return []

        def process_one(problem_name: str, text: str, solver: str, division: Optional[str], url: Optional[str],
                        out_f, sample_weight: float, verbose=False, clausify_cmd: Optional[str]=None) -> int:
            # Prefer SZS window; if absent, fall back to scanning the whole file.
            cleaned = clean_line_noise(text)
            body = locate_proof_window(cleaned)
            if not body:
                body = cleaned
                if verbose:
                    print(f"[info] {problem_name} @ {solver}: no SZS window; scanning entire file")
            fof_roles: Dict[str, str] = {}
            for m in FOF_HEAD_RE.finditer(body):
                fid, role = m.group(1), m.group(2).lower()
                fof_roles[fid] = role

            conjecture_text = ""
            fof_items_raw = []
            for raw in iter_items_balanced(body, "fof"):
                fof_items_raw.append(raw)
                m = FOF_HEAD_RE.search(raw)
                if m and m.group(2).lower() in ("conjecture", "negated_conjecture"):
                    form = extract_formula_from_raw(raw)
                    if form:
                        conjecture_text = form

            total = 0
            c_parents: Dict[str, List[str]] = {}
            rows = []
            for kind, head_re in (("cnf", CNF_HEAD_RE), ("tcf", TCF_HEAD_RE)):
                for raw in iter_items_balanced(body, kind):
                    m = head_re.search(raw)
                    if not m:
                        continue
                    cid, role = m.group(1), m.group(2).lower()
                    formula = extract_formula_from_raw(raw)
                    parents = parse_parents_from_inference(raw)
                    c_parents[cid] = parents
                    mnum = re.search(r"\d+", cid)
                    born = int(mnum.group(0)) if mnum else -1
                    rows.append({
                        "problem_name": problem_name,
                        "division": division,
                        "url": url,
                        "conjecture_text": conjecture_text,
                        "text": formula,
                        "features": {"horn": 0, "epr": 0, "unit": 0, "born": born, "conj_dist": -1},
                        "label": 1,
                        "neg_bucket": None,
                        "source": solver_to_source_tag(solver),
                        "proof_solver": solver,
                        "sample_weight": float(sample_weight),
                        "item_id": cid,
                        "role": role,
                        "item_kind": kind
                    })

            # If no clause-level items but have fof, try clausification automatically
            if not rows and fof_items_raw:
                with tempfile.TemporaryDirectory() as tmpd:
                    cnf_blocks = run_external_clausifier(fof_items_raw, clausify_cmd or "", tmpd, verbose=verbose)
                for raw in cnf_blocks:
                    m = CNF_HEAD_RE.search(raw)
                    if not m:
                        continue
                    cid, role = m.group(1), m.group(2).lower()
                    formula = extract_formula_from_raw(raw)
                    rows.append({
                        "problem_name": problem_name,
                        "division": division,
                        "url": url,
                        "conjecture_text": conjecture_text,
                        "text": formula,
                        "features": {"horn": 0, "epr": 0, "unit": 0, "born": -1, "conj_dist": -1},
                        "label": 1,
                        "neg_bucket": None,
                        "source": solver_to_source_tag(solver),
                        "proof_solver": solver,
                        "sample_weight": float(sample_weight),
                        "item_id": cid,
                        "role": role,
                        "item_kind": "cnf"
                    })

            if not rows:
                if verbose:
                    print(f"[miss] {problem_name} @ {solver}: no clause-level items (and no clausify)")
                return 0

            conj_dist = {}
            if rows and c_parents:
                conj_dist = compute_conj_dist(c_parents, fof_roles)

            for idx, r in enumerate(rows):
                if r["features"]["born"] == -1:
                    r["features"]["born"] = idx
                form = r["text"]
                r["features"]["unit"] = 1 if feat_is_unit(form) else 0
                r["features"]["horn"] = 1 if feat_is_horn(form) else 0
                r["features"]["epr"]  = 1 if feat_is_epr(form) else 0
                if conj_dist:
                    r["features"]["conj_dist"] = conj_dist.get(r["item_id"], -1)
                out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
                total += 1
            return total

        def load_problem_names(path: str) -> List[str]:
            # Try JSON
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
                elif isinstance(obj, list):
                    for p in obj:
                        if isinstance(p, dict) and "problem_name" in p:
                            names.append(p["problem_name"])
                        elif isinstance(p, str):
                            names.append(p)
                if names:
                    return names
            except json.JSONDecodeError:
                pass
            except Exception:
                pass
            # Try JSONL or plain text
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
                                    names.append(name); continue
                        except Exception:
                            pass
                        names.append(line)
            except Exception:
                return []
            return names

        def main():
            ap = argparse.ArgumentParser()
            ap.add_argument("--problems", required=False,
                            help="Path to problem list (txt/json/jsonl; one name per line or objects with 'problem_name').")
            ap.add_argument("--out", required=True, help="Output JSONL path.")
            ap.add_argument("--divisions", nargs="+", default=["FNE", "FEQ", "FOF"],
                            help="CASC divisions to try (default: FNE FEQ FOF).")
            ap.add_argument("--base-url", default=DEFAULT_BASE, help="CASC base url (J12).")
            ap.add_argument("--provers", nargs="+", default=DEFAULT_PROVERS,
                            help="Which solver directories to query, e.g. 'Vampire---4.9' 'E---3.2.0'.")
            ap.add_argument("--sample-weight", type=float, default=0.5, help="Sample weight for teacher data (default 0.5).")
            ap.add_argument("--clausify-cmd", default=None,
                            help="External clausifier command template; must contain {in} and {out}.")
            ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
            ap.add_argument("--miss-report", default=None, help="Optional TSV of (problem,solver,reason).")
            ap.add_argument("--proof-dir", default=None,
                            help="Directory containing local result pages named <problem>.txt/.p; used before web.")
            ap.add_argument("--index-html", default=None,
                            help="Local CASC FOF Results HTML to parse for result links (iProver/Vampire/E).")
            ap.add_argument("--index-as-problems", action="store_true",
                            help="Use all problems parsed from --index-html as the problem list.")

            args = ap.parse_args()

            index_map: Dict[str, Dict[str, str]] = {}
            names: List[str] = []
            if args.index_html:
                index_map = parse_index_html(args.index_html)
                if args.index_as_problems and index_map:
                    names = sorted(index_map.keys())
            if not names:
                if not args.problems:
                    print("[error] provide --problems or --index-html with --index-as-problems", file=sys.stderr)
                    sys.exit(2)
                names = load_problem_names(args.problems)
                if not names:
                    print(f"[error] could not load problem names from: {args.problems}", file=sys.stderr)
                    sys.exit(2)

            out_f = open(args.out, "w", encoding="utf-8")
            misses = []
            count_problems = 0
            count_rows = 0

            try:
                for name in names:
                    emitted_this_problem = 0
                    if index_map and name in index_map:
                        fallback_order = ["iProver---3.9", "Vampire---4.9", "E---3.2.0"]
                        links = index_map[name]
                        tried_any = False
                        for solver in fallback_order:
                            # Use index link if available, else try constructed URLs
                            txt = div = url = None
                            if solver in links:
                                tried_any = True
                                base_url = links[solver]
                                candidate_urls = [base_url]
                                if not base_url.endswith(".p"):
                                    candidate_urls.append(base_url + ".p")
                                t, d, u = try_fetch_result(name, args.divisions, args.base_url, solver,
                                                           verbose=args.verbose, proof_dir=args.proof_dir,
                                                           urls_override=candidate_urls)
                                txt, div, url = t, d, u
                            if not txt:
                                t2, d2, u2 = try_fetch_result(name, args.divisions, args.base_url, solver,
                                                              verbose=args.verbose, proof_dir=args.proof_dir)
                                if t2:
                                    tried_any = True
                                    txt, div, url = t2, d2, u2
                            if not txt:
                                misses.append((name, solver, "not_found"))
                                continue
                            rows = process_one(name, txt, solver, div, url, out_f,
                                               sample_weight=args.sample_weight,
                                               verbose=args.verbose,
                                               clausify_cmd=args.clausify_cmd)
                            if rows > 0:
                                emitted_this_problem += rows
                                break
                            else:
                                misses.append((name, solver, "no_clause_items"))
                        if not tried_any and args.verbose:
                            print(f"[warn] {name}: no index links for preferred solvers")
                    else:
                        for solver in args.provers:
                            txt, div, url = try_fetch_result(name, args.divisions, args.base_url, solver,
                                                             verbose=args.verbose, proof_dir=args.proof_dir)
                            if not txt:
                                misses.append((name, solver, "not_found"))
                                continue
                            rows = process_one(name, txt, solver, div, url, out_f,
                                               sample_weight=args.sample_weight,
                                               verbose=args.verbose,
                                               clausify_cmd=args.clausify_cmd)
                            if rows > 0:
                                emitted_this_problem += rows
                                break
                            else:
                                misses.append((name, solver, "no_clause_items"))
                    if emitted_this_problem > 0:
                        count_problems += 1
                        count_rows += emitted_this_problem
                    elif args.verbose:
                        print(f"[warn] {name}: no rows from any solver")
            finally:
                out_f.close()

            if args.miss_report:
                with open(args.miss_report, "w", encoding="utf-8") as f:
                    for n, s, r in misses:
                        f.write(f"{n}\t{s}\t{r}\n")

            print(f"[done] problems_with_any_rows={count_problems}, rows_emitted={count_rows}, out={args.out}")
            if args.miss_report:
                print(f"[report] miss_report={args.miss_report}, count={len(misses)}")

        if __name__ == "__main__":
            main()
    if third_start == -1:
        return ""
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
    return re.findall(r"\b([cf][A-Za-z0-9_]*\d+)\b", m.group(1))

def split_top_level_disj(formula: str):
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
    lits = split_top_level_disj(formula)
    pos = 0
    for lit in lits:
        t = lit.strip()
        if t.startswith('~'):
            continue
        if '!=' in t and not t.startswith('~'):
            continue
        pos += 1
        if pos > 1:
            return 0
    return 1

def feat_is_epr(formula: str) -> int:
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
    conj_f = {fid for fid, r in fof_roles.items() if r in ('conjecture', 'negated_conjecture')}
    dist = {}
    from collections import deque
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
            if any(p.startswith('f') and p in conj_f for p in pars):
                hit = d + 1
                break
            for p in pars:
                if p.startswith('c'):
                    q.append((p, d + 1))
        dist[cid] = hit if hit is not None else -1
    return dist

def build_url(base: str, division: str, solver: str, problem_name: str) -> str:
    return f"{base.rstrip('/')}/{division}/{solver}/{problem_name}"

def parse_division_from_url(url: str) -> Optional[str]:
    try:
        # Expect .../Results/<DIVISION>/<SOLVER>/<PROBLEM>[.p]
        parts = url.split("/Results/")
        if len(parts) < 2:
            return None
        rest = parts[1]
        return rest.split("/")[0]
    except Exception:
        return None

def parse_index_html(index_path: str) -> Dict[str, Dict[str, str]]:
    """
    Parse a local CASC results HTML page and extract solver result links for
    iProver---3.9, Vampire---4.9, and E---3.2.0.
    Returns: { problem_name: { solver: url, ... }, ... }
    """
    try:
        with open(index_path, "r", encoding="utf-8") as f:
            html = f.read()
    except Exception:
        return {}
    # Match anchors to Results URLs for target solvers
    rx = re.compile(
        r'href="(?P<url>[^"\s]+/Results/[^/]+/(?P<solver>iProver---3\.9|Vampire---4\.9|E---3\.2\.0)/(?P<prob>[A-Za-z0-9_+\-^]+))"',
        re.IGNORECASE)
    out: Dict[str, Dict[str, str]] = {}
    for m in rx.finditer(html):
        url = m.group("url")
        solver = m.group("solver")
        prob = m.group("prob")
        d = out.setdefault(prob, {})
        # Keep the first seen link per solver (divisions might repeat; first is fine)
        if solver not in d:
            d[solver] = url
    return out

def solver_to_source_tag(solver: str) -> str:
    s = solver.lower()
    if "vampire" in s: return "teacher_vampire"
    if s == "e" or s.startswith("e---"): return "teacher_e"
    return f"teacher_{solver.replace('---','_').replace('-','_')}"

def clean_line_noise(s: str) -> str:
    return re.sub(r'(?m)^\s*\d+(?:\.\d+)?/\d+(?:\.\d+)?\s*\t?', '', s)

def try_fetch_result(problem_name: str, divisions: List[str], base: str, solver: str,
                     verbose=False, proof_dir: Optional[str]=None,
                     urls_override: Optional[List[str]] = None) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    import urllib.parse
    # local first if provided
    if proof_dir:
        for cand in (problem_name, f"{problem_name}.txt", f"{problem_name}.p", f"{problem_name}.out"):
            local_path = os.path.join(proof_dir, cand)
            if os.path.exists(local_path):
                try:
                    with open(local_path, "r", encoding="utf-8") as f:
                        txt = f.read()
                    return txt, "LOCAL", local_path
                except Exception:
                    pass
    # direct URLs if provided (in order)
    if urls_override:
        for u in urls_override:
            if verbose:
                print(f"[fetch] {problem_name} @ {solver}: {u}")
            txt = fetch(u)
            if txt and "% SZS" in txt:
                return txt, parse_division_from_url(u), u
    # web fallback
    for div in divisions:
        for cand in (problem_name, f"{problem_name}.p"):
            url = build_url(base, div, solver, cand)
            if verbose:
                print(f"[fetch] {problem_name} @ {solver}: {url}")
            txt = fetch(url)
            if txt and "% SZS" in txt:
                return txt, div, url
    return None, None, None

def run_external_clausifier(fof_items: List[str], cmd_template: str, tmpdir: str, verbose=False) -> List[str]:
    """
    Try to clausify a set of FOF items using an external tool.
    Strategy:
      1) Use the given command template (expects {in} and {out}).
      2) If it fails, try safe fallbacks automatically when available:
         - eprover with resource limits, without --no-preprocessing
         - eprover-preprocess (if installed)
         - tptp4X (if installed), via shell redirection to the out file
    Returns list of cnf(...) blocks on success, or [] otherwise.
    """
    if not cmd_template or "{in}" not in cmd_template or "{out}" not in cmd_template:
        # Even if user didn't provide a template, we'll attempt auto fallbacks below.
        cmd_template = ""
    in_path = os.path.join(tmpdir, "in_fof.p")
    out_path = os.path.join(tmpdir, "out_cnf.p")
    with open(in_path, "w", encoding="utf-8") as f:
        for raw in fof_items:
            line = raw.strip()
            if not line.endswith('.'):
                line += '.'
            f.write(line + "\n")
    # Build candidate commands (primary + fallbacks)
    candidates: List[str] = []
    import re as _re
    if cmd_template:
        candidates.append(_re.sub(r"--output-file\s+", "--output-file=", cmd_template))

    # Helper: add candidate if binary seems available
    def _exists(cmd_name: str) -> bool:
        return bool(shutil.which(cmd_name))

    # Detect if the installed eprover appears to be a DEBUG build; if so, deprioritize/skip it
    def _is_e_debug() -> bool:
        if not _exists("eprover"):
            return False
        try:
            out = subprocess.check_output(["eprover", "--version"], text=True, stderr=subprocess.STDOUT, timeout=3)
            return "DEBUG" in out.upper()
        except Exception:
            return False

    # If primary contains eprover but without limits, add a limited variant
    if not candidates or ("eprover" in candidates[0] and "cpu-limit" not in candidates[0]):
        if _exists("eprover") and not _is_e_debug():
            candidates.append(
                "eprover --tstp-in --tstp-out --cnf --output-level=0 "
                "--cpu-limit=10 --memory-limit=1024 --output-file={out} {in}"
            )

    # eprover-preprocess if present (no searching involved, tends to be robust)
    if _exists("eprover-preprocess"):
        candidates.append(
            "eprover-preprocess --tstp-in --tstp-out --cnf --output-level=0 --output-file={out} {in}"
        )

    # tptp4X fallback (writes to stdout; we redirect). Works when available in PATH or common local dir.
    tptp4x_path = shutil.which("tptp4X") or (
        "/home/ks/TPTP4X/tptp4X" if os.path.exists("/home/ks/TPTP4X/tptp4X") else None
    )
    if tptp4x_path:
        # Prefer tptp4X early when available (robust CNF conversion)
        candidates.insert(0, f"{tptp4x_path} -x cnf -f tptp {{in}} > {{out}}")

    if not candidates:
        if verbose:
            print("[clausify] no available clausifier command found in PATH")
        return []

    # Try candidates in order
    if verbose:
        try:
            print("[clausify] candidates:")
            for ci, c in enumerate(candidates):
                print(f"  [{ci}] {c}")
        except Exception:
            pass

    for idx, tmpl in enumerate(candidates):
        # Normalize E long option style: use --output-file=<arg>
        tmpl2 = _re.sub(r"--output-file\s+", "--output-file=", tmpl)
        shell_cmd = tmpl2.format_map({"in": in_path, "out": out_path})
        if verbose:
            prefix = "[clausify] running" if idx == 0 else "[clausify] fallback"
            print(f"{prefix}: {shell_cmd}")
        try:
            # Ensure previous out_path is cleared to avoid reading stale data
            if os.path.exists(out_path):
                try:
                    os.remove(out_path)
                except Exception:
                    pass
            subprocess.run(shell_cmd, shell=True, check=True)
        except subprocess.CalledProcessError as e:
            if verbose:
                print(f"[clausify] command failed (rc={e.returncode}): {e}")
            # On failure, continue to next candidate
            continue
        # Read output
        try:
            with open(out_path, "r", encoding="utf-8") as f:
                content = f.read()
        except Exception:
            # If this candidate uses stdout redirection, out_path should exist; if not, try next
            continue
        # If we got any cnf(...) tokens, accept and parse
        if content and "cnf(" in content:
            s = content
            cnf_blocks = []
            token = "cnf("
            i, n = 0, len(s)
            while True:
                p = s.find(token, i)
                if p == -1:
                    break
                j = p + 3
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
                cnf_blocks.append(s[p:k].strip())
                i = k
            if cnf_blocks:
                return cnf_blocks
        else:
            if verbose:
                print("[clausify] no cnf(...) found in output; trying next candidate")
            continue
    # All candidates failed
    return []
    cnf_blocks = []
    token = "cnf("
    s = content
    i, n = 0, len(s)
    while True:
        p = s.find(token, i)
        if p == -1:
            break
        j = p + 3
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
        cnf_blocks.append(s[p:k].strip())
        i = k
    return cnf_blocks

def process_one(problem_name: str, text: str, solver: str, division: Optional[str], url: Optional[str],
                out_f, sample_weight: float, verbose=False, clausify_cmd: Optional[str]=None) -> int:
    # Prefer SZS window; if absent, fall back to scanning the whole file.
    cleaned = clean_line_noise(text)
    body = locate_proof_window(cleaned)
    if not body:
        body = cleaned
        if verbose:
            print(f"[info] {problem_name} @ {solver}: no SZS window; scanning entire file for fof/cnf/tcf")
    fof_roles: Dict[str, str] = {}
    for m in FOF_HEAD_RE.finditer(body):
        fid, role = m.group(1), m.group(2).lower()
        fof_roles[fid] = role

    conjecture_text = ""
    fof_items_raw = []
    for raw in iter_items_balanced(body, "fof"):
        fof_items_raw.append(raw)
        m = FOF_HEAD_RE.search(raw)
        if m and m.group(2).lower() in ("conjecture", "negated_conjecture"):
            form = extract_formula_from_raw(raw)
            if form:
                conjecture_text = form

    total = 0
    c_parents: Dict[str, List[str]] = {}
    rows = []
    for kind, head_re in (("cnf", CNF_HEAD_RE), ("tcf", TCF_HEAD_RE)):
        for raw in iter_items_balanced(body, kind):
            m = head_re.search(raw)
            if not m:
                continue
            cid, role = m.group(1), m.group(2).lower()
            formula = extract_formula_from_raw(raw)
            parents = parse_parents_from_inference(raw)
            c_parents[cid] = parents
            mnum = re.search(r"\d+", cid)
            born = int(mnum.group(0)) if mnum else -1
            rows.append({
                "problem_name": problem_name,
                "division": division,
                "url": url,
                "conjecture_text": conjecture_text,
                "text": formula,
                "features": {"horn": 0, "epr": 0, "unit": 0, "born": born, "conj_dist": -1},
                "label": 1,
                "neg_bucket": None,
                "source": solver_to_source_tag(solver),
                "sample_weight": float(sample_weight),
                "item_id": cid,
                "role": role,
                "item_kind": kind
            })

    if not rows and fof_items_raw:
        # Attempt clausification using provided template or automatic fallbacks.
        with tempfile.TemporaryDirectory() as tmpd:
            cnf_blocks = run_external_clausifier(fof_items_raw, clausify_cmd or "", tmpd, verbose=verbose)
        for raw in cnf_blocks:
            m = CNF_HEAD_RE.search(raw)
            if not m:
                continue
            cid, role = m.group(1), m.group(2).lower()
            formula = extract_formula_from_raw(raw)
            rows.append({
                "problem_name": problem_name,
                "division": division,
                "url": url,
                "conjecture_text": conjecture_text,
                "text": formula,
                "features": {"horn": 0, "epr": 0, "unit": 0, "born": -1, "conj_dist": -1},
                "label": 1,
                "neg_bucket": None,
                "source": solver_to_source_tag(solver),
                "sample_weight": float(sample_weight),
                "item_id": cid,
                "role": role,
                "item_kind": "cnf"
            })

    if not rows:
        if verbose:
            print(f"[miss] {problem_name} @ {solver}: no clause-level items (and no clausify)")
        return 0

    conj_dist = {}
    if rows and c_parents:
        conj_dist = compute_conj_dist(c_parents, fof_roles)

    for idx, r in enumerate(rows):
        if r["features"]["born"] == -1:
            r["features"]["born"] = idx
        form = r["text"]
        r["features"]["unit"] = 1 if feat_is_unit(form) else 0
        r["features"]["horn"] = 1 if feat_is_horn(form) else 0
        r["features"]["epr"]  = 1 if feat_is_epr(form) else 0
        if conj_dist:
            r["features"]["conj_dist"] = conj_dist.get(r["item_id"], -1)
        out_f.write(json.dumps(r, ensure_ascii=False) + "\n")
        total += 1
    return total

def load_problem_names(path: str) -> List[str]:
    # Try JSON
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
        elif isinstance(obj, list):
            for p in obj:
                if isinstance(p, dict) and "problem_name" in p:
                    names.append(p["problem_name"])
                elif isinstance(p, str):
                    names.append(p)
        if names:
            return names
    except json.JSONDecodeError:
        pass
    except Exception:
        pass
    # Try JSONL or plain text
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
                            names.append(name); continue
                except Exception:
                    pass
                names.append(line)
    except Exception:
        return []
    return names

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--problems", required=True,
                    help="Path to problem list (txt/json/jsonl; one name per line or objects with 'problem_name').")
    ap.add_argument("--out", required=True, help="Output JSONL path.")
    ap.add_argument("--divisions", nargs="+", default=["FNE", "FEQ", "FOF"],
                    help="CASC divisions to try (default: FNE FEQ FOF).")
    ap.add_argument("--base-url", default=DEFAULT_BASE, help="CASC base url (J12).")
    ap.add_argument("--provers", nargs="+", default=DEFAULT_PROVERS,
                    help="Which solver directories to query, e.g. 'Vampire---4.9' 'E---3.2.0'.")
    ap.add_argument("--index-html", default="/home/ks/LLM/iprover_FOF_result.html",
                    help="Local CASC Results HTML (FOF) to parse for solver result links (iProver/Vampire/E).")
    ap.add_argument("--index-as-problems", action="store_true",
                    help="Use all problems found in --index-html as the problem list (ignores --problems file).")
    ap.add_argument("--sample-weight", type=float, default=0.5, help="Sample weight for teacher data (default 0.5).")
    ap.add_argument("--clausify-cmd", default=None,
                    help="External clausifier command template; must contain {in} and {out}.")
    ap.add_argument("--verbose", action="store_true", help="Verbose logging.")
    ap.add_argument("--miss-report", default=None, help="Optional TSV of (problem,solver,reason).")
    ap.add_argument("--proof-dir", default=None,
                    help="Directory containing local result pages named <problem>.txt/.p; used before web.")
    args = ap.parse_args()

    index_map: Dict[str, Dict[str, str]] = {}
    if args.index_html:
        index_map = parse_index_html(args.index_html)
        if args.index_as_problems and index_map:
            names = sorted(index_map.keys())
        else:
            names = load_problem_names(args.problems)
    else:
        names = load_problem_names(args.problems)
    if not names:
        src = args.index_html or args.problems
        print(f"[error] could not load problem names from: {src}", file=sys.stderr)
        sys.exit(2)

    out_f = open(args.out, "w", encoding="utf-8")
    misses = []
    count_problems = 0
    count_rows = 0

    try:
        for name in names:
            emitted_this_problem = 0
            # If index_map provided, use fallback order iProver -> Vampire 4.9 -> E 3.2.0
            if index_map and name in index_map:
                fallback_order = ["iProver---3.9", "Vampire---4.9", "E---3.2.0"]
                links = index_map[name]
                tried_any = False
                for solver in fallback_order:
                    if solver not in links:
                        continue
                    tried_any = True
                    base_url = links[solver]
                    # Try both the naked URL and the .p variant
                    candidate_urls = [base_url]
                    if not base_url.endswith(".p"):
                        candidate_urls.append(base_url + ".p")
                    txt, div, url = try_fetch_result(name, args.divisions, args.base_url, solver,
                                                     verbose=args.verbose, proof_dir=args.proof_dir,
                                                     urls_override=candidate_urls)
                    if not txt:
                        misses.append((name, solver, "not_found"))
                        continue
                    rows = process_one(name, txt, solver, div, url, out_f,
                                       sample_weight=args.sample_weight,
                                       verbose=args.verbose,
                                       clausify_cmd=args.clausify_cmd)
                    if rows > 0:
                        emitted_this_problem += rows
                        break  # stop after first successful solver per problem
                    else:
                        misses.append((name, solver, "no_clause_items"))
                if not tried_any and args.verbose:
                    print(f"[warn] {name}: no links for preferred solvers in index")
            else:
                # Original behavior: iterate provided solvers and try divisions/base
                for solver in args.provers:
                    txt, div, url = try_fetch_result(name, args.divisions, args.base_url, solver,
                                                     verbose=args.verbose, proof_dir=args.proof_dir)
                    if not txt:
                        misses.append((name, solver, "not_found"))
                        continue
                    rows = process_one(name, txt, solver, div, url, out_f,
                                       sample_weight=args.sample_weight,
                                       verbose=args.verbose,
                                       clausify_cmd=args.clausify_cmd)
                    if rows > 0:
                        emitted_this_problem += rows
                    else:
                        misses.append((name, solver, "no_clause_items"))
            if emitted_this_problem > 0:
                count_problems += 1
                count_rows += emitted_this_problem
            elif args.verbose:
                print(f"[warn] {name}: no rows from any solver")
    finally:
        out_f.close()

    if args.miss_report:
        with open(args.miss_report, "w", encoding="utf-8") as f:
            for n, s, r in misses:
                f.write(f"{n}\t{s}\t{r}\n")

    print(f"[done] problems_with_any_rows={count_problems}, rows_emitted={count_rows}, out={args.out}")
    if args.miss_report:
        print(f"[report] miss_report={args.miss_report}, count={len(misses)}")

if __name__ == "__main__":
    main()
