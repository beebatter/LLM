#!/usr/bin/env python3
"""
End‑to‑end driver to collect training data from a batch of TPTP FOF problems with
support for failure logging and high‑quality negative sampling.

This script automates the following workflow for each problem in a JSON list:

1. Start the external agent (EA) implemented in ``process_iprover_v3.py`` in
   interactive mode.  The EA listens on a chosen TCP port, assigns heuristic
   scores to candidate clauses (in dry‑run mode so that no large language
   models are called) and logs all messages exchanged between iProver and the
   EA to a specified file.

2. Launch iProver in interactive mode on the target problem, connecting to the
   EA via the reserved port.  iProver will register all generated clauses,
   emit events when clauses move through passive/given/simplified queues and
   optionally send a ``proof_out`` event listing the clause identifiers used
   in a final proof.  The EA uses the scores to guide clause selection.

3. Once iProver terminates (either by proving unsatisfiability, giving up
   without a proof or being killed on timeout), the EA exits as well.  The
   raw log is parsed into labelled clause examples using the helper
   functions from ``iplog_to_dataset.py``.

4. Optionally sample the negative examples to produce a balanced dataset.
   Positive examples are always retained in full (deduplicated by
   ``basic_clause_id``).  Negative examples are divided into buckets
   (``NEG_given_nonproof``, ``NEG_simplified``, ``NEG_passive_only``,
   ``NEG_never_seen``) and sampled according to quotas favouring hard
   negatives near the search frontier.

5. Append the resulting examples to a cumulative JSONL file.  Failures
   (timeouts, iProver errors, absence of proofs) are recorded in a
   separate JSONL fail log together with the last few ``given_clause`` IDs
   to aid manual triage.

Usage example::

    python run_batch_pipeline.py \
        --problems fof_problems_from_html.json \
        --iprover iproveropt \
        --output dataset.jsonl \
        --fail-log failed_problems.jsonl \
        --timeout 60

Additional options allow you to control negative sampling:

``--no-sample-negatives``
    Disable negative sampling entirely and write all parsed examples.

``--neg-mult`` (default 8)
    Target ratio of negatives to positives.  Each problem will include up to
    ``neg_mult × (# positive examples)`` negatives, subject to caps.

``--neg-cap-per-problem`` (default 1000)
    Upper bound on the number of negatives for a single problem.

``--fallback-neg-cap`` (default 300)
    Maximum number of negatives to keep when a problem has zero positive
    examples (e.g. no proof found but you choose to keep negatives with
    ``--keep-failed-negatives``).

``--frontier-window`` (default 64)
    Number of most recent ``given_clause`` IDs considered as the search
    frontier when sampling hard negatives from ``NEG_given_nonproof``.

See the command‑line help for full details.
"""

import argparse
import json
import os
import socket
import subprocess
import sys
import time
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Helper functions for failure logging.  These mirror logic used in
# previous versions but are bundled here for clarity.

def _append_jsonl(path: str, obj: dict) -> None:
    """Append a JSON object as a line to a JSONL file, creating parent dirs."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(obj, ensure_ascii=False) + "\n")


def extract_last_given(log_path: str, limit: int = 32) -> List[int]:
    """Return up to ``limit`` clause IDs from the most recent given_clause events.

    The raw log is scanned backwards (splitting on NUL and newlines) and
    ``clause_ids`` entries are collected from ``given_clause`` messages.
    If the log cannot be read or no given clause events are found, an
    empty list is returned.  The returned list is in chronological order.
    """
    if not os.path.exists(log_path):
        return []
    try:
        with open(log_path, "rb") as f:
            data = f.read()
        parts = [seg for seg in data.split(b"\x00") if seg.strip()]
        last_ids: List[int] = []
        for segment in reversed(parts):
            for line in reversed(segment.splitlines()):
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                if msg.get("tag") == "given_clause":
                    for cid in msg.get("clause_ids", []) or []:
                        try:
                            last_ids.append(int(cid))
                        except Exception:
                            pass
                    if len(last_ids) >= limit:
                        break
            if len(last_ids) >= limit:
                break
        return list(reversed(last_ids[-limit:]))
    except Exception:
        return []


def _build_clause_index(log_path: str) -> Dict[int, Dict[str, Any]]:
    """Scan the raw EA log and build a mapping: clause_id -> {clause, features}.

    Looks for register_clauses messages and records per-clause text and
    features. Returns an empty dict on errors or when no entries are found.
    """
    index: Dict[int, Dict[str, Any]] = {}
    if not os.path.exists(log_path):
        return index
    try:
        with open(log_path, "rb") as f:
            data = f.read()
        # The EA log typically uses NULs to separate message groups
        parts = [seg for seg in data.split(b"\x00") if seg.strip()]
        for segment in parts:
            for line in segment.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                if msg.get("tag") == "register_clauses":
                    for cl in msg.get("clauses", []) or []:
                        try:
                            cid = int(cl.get("clause_id"))
                        except Exception:
                            continue
                        entry = {
                            "clause": cl.get("clause"),
                            "features": (cl.get("clause_features") or {}),
                        }
                        index[cid] = entry
    except Exception:
        return index
    return index


def extract_last_given_details(log_path: str, limit: int = 32) -> List[Dict[str, Any]]:
    """Return details for the most recent given_clause events.

    Each element includes the transient ID (for correlation), the clause text,
    and its features if available, e.g.::

        {"id": 245, "clause": "tcf(...).", "features": {"born": 3, ...}}
    """
    ids = extract_last_given(log_path, limit=limit)
    if not ids:
        return []
    index = _build_clause_index(log_path)
    details: List[Dict[str, Any]] = []
    for cid in ids:
        info = index.get(cid, {})
        details.append({
            "id": cid,
            "clause": info.get("clause"),
            "features": info.get("features"),
        })
    return details



def extract_sample_passive_details(log_path: str, limit: int = 8) -> List[Dict[str, Any]]:
    """Return up to `limit` passive-only clause details for failure logs."""
    try:
        idx = _build_clause_index(log_path)
        given: Set[int] = set()
        passive: Set[int] = set()
        simplified: Set[int] = set()
        with open(log_path, "rb") as f:
            data = f.read()
        parts = [seg for seg in data.split(b"\x00") if seg.strip()]
        for segment in parts:
            for line in segment.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode("utf-8", errors="ignore"))
                except Exception:
                    continue
                tag = msg.get("tag")
                if tag == "given_clause":
                    for cid in msg.get("clause_ids", []) or []:
                        try: given.add(int(cid))
                        except Exception: pass
                elif tag == "passive_clauses":
                    for cid in msg.get("clause_ids", []) or []:
                        try: passive.add(int(cid))
                        except Exception: pass
                elif tag == "simplified_clauses":
                    for cid in msg.get("clause_ids", []) or []:
                        try: simplified.add(int(cid))
                        except Exception: pass
        cids = [cid for cid in passive if cid not in given and cid not in simplified]
        items = []
        for cid in cids:
            info = idx.get(cid, {})
            feats = info.get("features") or {}
            born = feats.get("born", 1<<30)
            horn = feats.get("horn", False)
            conj = feats.get("conj_dist", 1<<30)
            score = (1 if horn else 0) - 0.001*born - 0.0001*conj
            items.append((score, {"id": cid, "clause": info.get("clause"), "features": feats}))
        items.sort(key=lambda x: x[0], reverse=True)
        return [d for _, d in items[:limit]]
    except Exception:
        return []
def log_failure(
    fail_log_path: str,
    *,
    problem_name: str,
    file_path: str,
    domain: Optional[str],
    reason: str,
    timeout_sec: Optional[int],
    elapsed_sec: Optional[float],
    iprover_exit: Optional[int],
    external_port: Optional[int],
    log_path: Optional[str],
) -> None:
    """Record a failed run to the failure log.

    Captures problem metadata, the failure reason (``timeout``, ``iprover_error``,
    or ``no_proof``), timing information and the port used.  To aid
    debugging, the last few given clause IDs are extracted from the raw log
    if available.  Each entry is appended as one JSON object per line.
    """
    entry: Dict[str, Any] = {
        "problem_name": problem_name,
        "file_path": file_path,
        "domain": domain,
        "reason": reason,
        "timeout_sec": timeout_sec,
        "elapsed_sec": elapsed_sec,
        "iprover_exit": iprover_exit,
        "external_port": external_port,
        "log_path": log_path,
        "timestamp": int(time.time()),
        "note": "manual_proof_needed",
    }
    if log_path:
        entry["last_given_ids"] = extract_last_given(log_path, limit=32)
        # Richer context: include the clause texts and features for the last given clauses
        entry["last_given_clauses"] = extract_last_given_details(log_path, limit=32)
        entry["passive_sample"] = extract_sample_passive_details(log_path, limit=8)
    else:
        entry["last_given_ids"] = []
        entry["last_given_clauses"] = []
    _append_jsonl(fail_log_path, entry)


def has_proof_out(path: str) -> bool:
    """Quickly check whether a raw log contains a ``proof_out`` event."""
    if not os.path.exists(path):
        return False
    try:
        with open(path, "rb") as f:
            data = f.read()
        return b'"proof_out"' in data
    except Exception:
        return False


def has_szs_success_in_ea_log(path: str) -> bool:
    """Check EA raw log for a szs_result_out event indicating success.

    Treats either Theorem or Unsatisfiable as success signals.
    """
    if not os.path.exists(path):
        return False
    try:
        with open(path, 'rb') as f:
            data = f.read()
        # Fast path: look for the tag, then cheaply scan JSON lines
        if b'"szs_result_out"' not in data:
            return False
        for segment in data.split(b"\x00"):
            for line in segment.splitlines():
                line = line.strip()
                if not line:
                    continue
                try:
                    msg = json.loads(line.decode('utf-8', errors='ignore'))
                except Exception:
                    continue
                if msg.get('tag') == 'szs_result_out':
                    status = str(msg.get('szs_status') or '')
                    if re.search(r"SZS\\s+status\\s+(Theorem|Unsatisfiable)\\b", status):
                        return True
        return False
    except Exception:
        return False

# Import parsers from iplog_to_dataset.  If import fails, augment sys.path.
try:
    from iplog_to_dataset import parse_log, build_dataset
except Exception:
    sys.path.append(os.path.dirname(__file__))
    from iplog_to_dataset import parse_log, build_dataset  # type: ignore



def has_szs_proof(text: str) -> bool:
    """Detects whether iProver stdout contains a successful proof indication."""
    if not text:
        return False
    if "% SZS output start CNFRefutation" in text:
        return True
    # Some runs print only SZS status without the full CNFRefutation block
    if re.search(r"%\s*SZS\s+status\s+(Theorem|Unsatisfiable)\b", text):
        return True
    return False


def extract_proof_ids_from_szs_text(text: str) -> Set[int]:
    """Extract tcf(c_<ID>, ...) IDs from a SZS CNFRefutation block in stdout."""
    ids: Set[int] = set()
    if not text:
        return ids
    in_block = False
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        if line.startswith("% SZS output start CNFRefutation"):
            in_block = True
            continue
        if in_block and line.startswith("% SZS output end CNFRefutation"):
            break
        if in_block:
            m = re.search(r"tcf\(\s*c_(\d+)\s*,", line)
            if m:
                try:
                    ids.add(int(m.group(1)))
                except Exception:
                    pass
    return ids


def find_free_port() -> int:
    """Reserve and return an unused TCP port by binding a temporary socket."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def launch_ea(port: int, log_file: str, python_exec: str = sys.executable) -> subprocess.Popen:
    """Spawn the external agent server on the given port."""
    here = Path(__file__).resolve().parent
    ea_script = here / 'process_iprover_v3.py'
    if not ea_script.exists():
        raise FileNotFoundError(f"Cannot locate process_iprover_v3.py at {ea_script}")
    cmd = [
        python_exec,
        str(ea_script),
        'serve',
        '--host', '127.0.0.1',
        '--port', str(port),
        '--dry-run',
        '--exit-on-finish',
        '--python-exec', python_exec,
        '--verbose', '--progress',
        '--log-file', log_file,
        '--no-prefilter',
    ]
    return subprocess.Popen(cmd, stdout=None, stderr=None)


def launch_iprover(problem_path: str, port: int, iprover_bin: str,
                   stdout_path: str, stderr_path: str) -> subprocess.Popen:
    """Run iProver in interactive mode on ``problem_path`` connecting to ``port``."""
    cmd = [
        iprover_bin,
        '--interactive_mode', 'true',
        '--external_ip_address', '127.0.0.1',
        '--external_port', str(port),
        '--schedule', 'none',
        '--preprocessing_flag', 'false',
        '--instantiation_flag', 'true',
        '--superposition_flag', 'true',
        '--resolution_flag', 'false',
        '--sup_iter_deepening', '0',
        '--comb_sup_deep_mult', '0',
        '--sup_passive_queue_type', 'priority_queues',
        '--sup_passive_queues_freq', '[1]',
        '--sup_passive_queues', '[[+external_score]]',
        problem_path,
    ]
    os.makedirs(os.path.dirname(stdout_path) or '.', exist_ok=True)
    out = open(stdout_path, 'wb')
    err = open(stderr_path, 'wb')
    return subprocess.Popen(cmd, stdout=out, stderr=err, cwd='/home/ks/iprover-master')


def collect_dataset_from_log(log_path: str, problem_name: str) -> List[Dict[str, Any]]:
    """Parse a raw log file and return dataset entries via iplog_to_dataset."""
    clauses, proof_ids, seen = parse_log(log_path)
    return build_dataset(clauses, proof_ids, seen, problem_name=problem_name)


###############################
# Negative sampling functions  #
###############################

def _neg_score(row: Dict[str, Any], frontier: Set[int], born_min: int, born_max: int) -> float:
    """Compute a lightweight score for ranking negative examples.

    The score encourages clauses that are closer to the proof frontier and
    structurally simple.  It combines the following factors:

    * Presence in the frontier (12 if yes, 0 otherwise)
    * Normalised born (0–5 scale)
    * Unit clauses (+3)
    * Horn clauses (+1)
    * Short conj_dist (+ up to 5 for small distances)

    Missing features default to neutral values.  The parameters ``born_min``
    and ``born_max`` are used to normalise the born values across all
    negative examples for the current problem.
    """
    feats = row.get('features', {}) or {}
    is_frontier = 1 if row.get('clause_id') in frontier else 0
    born = feats.get('born')
    if born is None or born_max <= born_min:
        born_norm = 0.0
    else:
        born_norm = (born - born_min) / (born_max - born_min)
    born_score = 5.0 * born_norm
    unit_score = 3.0 if feats.get('unit') else 0.0
    horn_score = 1.0 if feats.get('horn') else 0.0
    conj_dist = feats.get('conj_dist')
    if conj_dist is None:
        conj_score = 0.0
    else:
        conj_score = max(0.0, 5.0 - float(conj_dist))
    return 12.0 * is_frontier + born_score + unit_score + horn_score + conj_score


def _dedup_positives(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Deduplicate positive examples by basic_clause_id, retaining earliest born."""
    dedup: Dict[Any, Dict[str, Any]] = {}
    for row in rows:
        key = row.get('basic_clause_id') if row.get('basic_clause_id') is not None else row.get('clause_id')
        current = dedup.get(key)
        if current is None:
            dedup[key] = row
            continue
        feats_new = row.get('features', {}) or {}
        feats_old = current.get('features', {}) or {}
        born_new = feats_new.get('born', 0)
        born_old = feats_old.get('born', 0)
        if born_new < born_old:
            dedup[key] = row
    return list(dedup.values())


def _slice_by_born(rows: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Split rows into (early, mid, late) groups based on born order."""
    if not rows:
        return [], [], []
    rows_sorted = sorted(rows, key=lambda r: (r.get('features', {}).get('born', 0)))
    n = len(rows_sorted)
    if n < 3:
        return [], [], rows_sorted
    third = max(1, n // 3)
    early = rows_sorted[:third]
    mid = rows_sorted[third:2 * third]
    late = rows_sorted[2 * third:]
    return early, mid, late


def sample_problem_entries(
    rows: List[Dict[str, Any]],
    frontier: Set[int],
    *,
    neg_mult: int = 8,
    neg_cap: int = 1000,
    fallback_neg_cap: int = 300,
) -> List[Dict[str, Any]]:
    """Given all rows for a problem, sample negative examples according to quotas."""
    positives = [r for r in rows if r.get('label') == 1]
    negatives = [r for r in rows if r.get('label') == 0]
    pos_dedup = _dedup_positives(positives)
    pos_count = len(pos_dedup)
    if pos_count > 0:
        target_neg = neg_mult * pos_count
    else:
        target_neg = fallback_neg_cap
    target_neg = min(target_neg, neg_cap)
    # Group negatives by bucket
    buckets: Dict[str, List[Dict[str, Any]]] = {
        'NEG_given_nonproof': [],
        'NEG_simplified': [],
        'NEG_passive_only': [],
        'NEG_never_seen': [],
    }
    for row in negatives:
        bucket = row.get('neg_bucket') or 'NEG_never_seen'
        if bucket not in buckets:
            # Unknown buckets are lumped into passive_only
            buckets.setdefault('NEG_passive_only', []).append(row)
        else:
            buckets[bucket].append(row)
    if not negatives or target_neg <= 0:
        return pos_dedup
    # Precompute born range for scoring
    born_vals = []
    for row in negatives:
        bv = row.get('features', {}).get('born')
        if bv is not None:
            born_vals.append(bv)
    if born_vals:
        born_min = min(born_vals)
        born_max = max(born_vals)
    else:
        born_min = 0
        born_max = 0
    for row in negatives:
        row['_score'] = _neg_score(row, frontier, born_min, born_max)
    # Quotas
    q_given = int(0.50 * target_neg)
    q_simpl = int(0.20 * target_neg)
    q_passive = int(0.25 * target_neg)
    q_never = target_neg - (q_given + q_simpl + q_passive)
    if q_never < 0:
        q_never = 0
    # NEG_given_nonproof sampling
    selected_given: List[Dict[str, Any]] = []
    given_rows = buckets.get('NEG_given_nonproof', [])
    if given_rows and q_given > 0:
        frontier_rows = [r for r in given_rows if r.get('clause_id') in frontier]
        nonfrontier_rows = [r for r in given_rows if r.get('clause_id') not in frontier]
        take_frontier = max(0, min(len(frontier_rows), q_given // 3))
        if q_given > 0 and take_frontier == 0 and frontier_rows:
            take_frontier = 1
        if take_frontier > 0:
            frontier_sorted = sorted(frontier_rows, key=lambda r: r['_score'], reverse=True)
            selected_given.extend(frontier_sorted[:take_frontier])
        remaining = q_given - len(selected_given)
        if remaining > 0 and nonfrontier_rows:
            early, mid, late = _slice_by_born(nonfrontier_rows)
            sec_quota = max(1, remaining // 3) if remaining > 0 else 0
            def _sel(section: List[Dict[str, Any]], k: int) -> Tuple[List[Dict[str, Any]], int]:
                if not section or k <= 0:
                    return [], 0
                section_sorted = sorted(section, key=lambda r: r['_score'], reverse=True)
                take = min(k, len(section_sorted))
                return section_sorted[:take], take
            # Early
            chosen, used = _sel(early, sec_quota)
            selected_given.extend(chosen); remaining -= used
            # Mid
            if remaining > 0:
                chosen, used = _sel(mid, sec_quota)
                selected_given.extend(chosen); remaining -= used
            # Late
            if remaining > 0:
                chosen, used = _sel(late, sec_quota)
                selected_given.extend(chosen); remaining -= used
            if remaining > 0:
                selected_ids = {r['clause_id'] for r in selected_given}
                leftovers = [r for r in nonfrontier_rows if r['clause_id'] not in selected_ids]
                if leftovers:
                    extras = sorted(leftovers, key=lambda r: r['_score'], reverse=True)[:remaining]
                    selected_given.extend(extras)
    # NEG_simplified sampling
    selected_simpl: List[Dict[str, Any]] = []
    simpl_rows = buckets.get('NEG_simplified', [])
    if simpl_rows and q_simpl > 0:
        selected_simpl = sorted(simpl_rows, key=lambda r: r['_score'], reverse=True)[:min(q_simpl, len(simpl_rows))]
    # NEG_passive_only sampling
    selected_passive: List[Dict[str, Any]] = []
    passive_rows = buckets.get('NEG_passive_only', [])
    if passive_rows and q_passive > 0:
        selected_passive = sorted(passive_rows, key=lambda r: r['_score'], reverse=True)[:min(q_passive, len(passive_rows))]
    # NEG_never_seen sampling
    selected_never: List[Dict[str, Any]] = []
    never_rows = buckets.get('NEG_never_seen', [])
    if never_rows and q_never > 0:
        selected_never = sorted(never_rows, key=lambda r: r['clause_id'])[:min(q_never, len(never_rows))]
    selected_neg_set: Set[int] = set()
    selected_negatives: List[Dict[str, Any]] = []
    for coll in (selected_given, selected_simpl, selected_passive, selected_never):
        for r in coll:
            cid = r.get('clause_id')
            if cid is not None and cid not in selected_neg_set:
                selected_neg_set.add(cid)
                selected_negatives.append(r)
    remaining_quota = target_neg - len(selected_negatives)
    if remaining_quota > 0:
        leftovers = [r for r in negatives if r.get('clause_id') not in selected_neg_set]
        if leftovers:
            extras = sorted(leftovers, key=lambda r: r['_score'], reverse=True)[:remaining_quota]
            for r in extras:
                cid = r.get('clause_id')
                if cid is not None and cid not in selected_neg_set:
                    selected_neg_set.add(cid)
                    selected_negatives.append(r)
    # Remove temporary scores
    for r in negatives:
        if '_score' in r:
            del r['_score']
    final_rows = []
    final_rows.extend(pos_dedup)
    final_rows.extend(selected_negatives)
    return final_rows


####################################
# Driver to run each problem       #
####################################

def run_problem(
    problem: Dict[str, Any],
    *,
    iprover_bin: str,
    output_dir: Path,
    cumulative_file: str,
    simulate: bool,
    timeout: int,
    fail_log_path: Optional[str],
    keep_failed_negatives: bool,
    no_sample_negatives: bool,
    neg_mult: int,
    neg_cap_per_problem: int,
    fallback_neg_cap: int,
    frontier_window: int,
) -> None:
    """Process a single problem: run EA + iProver, parse log, sample and append."""
    problem_name = problem.get('problem_name') or Path(problem.get('file_path')).stem
    problem_path = problem.get('file_path')
    log_file = output_dir / f"{problem_name}.raw.log"
    iprover_stdout = output_dir / f"{problem_name}.iprover.out"
    iprover_stderr = output_dir / f"{problem_name}.iprover.err"
    port = find_free_port()
    start_time = time.time()
    failure_reason: Optional[str] = None
    iprover_exit: Optional[int] = None
    ea_proc: Optional[subprocess.Popen] = None
    ip_proc: Optional[subprocess.Popen] = None
    # Launch EA and iProver
    ea_proc = launch_ea(port, str(log_file))
    time.sleep(0.5)
    try:
        if not simulate:
            ip_proc = launch_iprover(problem_path, port, iprover_bin, str(iprover_stdout), str(iprover_stderr))
            try:
                iprover_exit = ip_proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                failure_reason = 'timeout'
                ip_proc.kill(); ip_proc.wait()
        try:
            ea_proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            ea_proc.kill(); ea_proc.wait()
    finally:
        for proc in (ip_proc, ea_proc):
            if proc and proc.poll() is None:
                proc.kill(); proc.wait()
    # Read iProver stdout for SZS proof detection
    szs_text = ''
    try:
        if iprover_stdout.exists():
            szs_text = iprover_stdout.read_text(encoding='utf-8', errors='ignore')
    except Exception:
        szs_text = ''

    elapsed = time.time() - start_time
    # Failure determination
    if not simulate:
        szs_ok = has_szs_proof(szs_text) or has_szs_success_in_ea_log(str(log_file))
        # Non-zero exit is not a failure if SZS indicates success
        if failure_reason is None and iprover_exit is not None and iprover_exit != 0 and not szs_ok:
            failure_reason = 'iprover_error'
        # Zero exit but no proof markers anywhere -> no_proof
        if failure_reason is None and iprover_exit == 0 and not (has_proof_out(str(log_file)) or szs_ok):
            failure_reason = 'no_proof'
    else:
        szs_ok = has_szs_proof(szs_text) or has_szs_success_in_ea_log(str(log_file))
        if not (has_proof_out(str(log_file)) or szs_ok):
            failure_reason = 'no_proof'
    # Record failures
    if failure_reason and fail_log_path:
        log_failure(
            fail_log_path,
            problem_name=problem_name,
            file_path=problem_path,
            domain=problem.get('domain'),
            reason=failure_reason,
            timeout_sec=timeout if failure_reason == 'timeout' else None,
            elapsed_sec=elapsed,
            iprover_exit=(iprover_exit if failure_reason == 'iprover_error' else (0 if failure_reason == 'no_proof' else None)),
            external_port=port,
            log_path=str(log_file) if log_file.exists() else None,
        )
    # Decide if dataset should be generated
    if not log_file.exists():
        print(f"[warn] missing log {log_file}; skip {problem_name}")
        return
    if failure_reason and not keep_failed_negatives:
        print(f"Skipped dataset generation for {problem_name} due to failure: {failure_reason}")
        return
    # Build dataset
    # Build dataset with SZS fallback: if no proof_out, recover tcf IDs from iProver stdout
    clauses, proof_ids, seen = parse_log(str(log_file))
    if not proof_ids and szs_text:
        szs_ids = extract_proof_ids_from_szs_text(szs_text)
        if szs_ids:
            proof_ids = set(szs_ids)
    dataset_entries = build_dataset(clauses, proof_ids, seen, problem_name=problem_name)
    # Sample negatives if requested
    if not no_sample_negatives:
        frontier_ids = set(extract_last_given(str(log_file), limit=frontier_window))
        sampled_entries = sample_problem_entries(
            dataset_entries,
            frontier_ids,
            neg_mult=neg_mult,
            neg_cap=neg_cap_per_problem,
            fallback_neg_cap=fallback_neg_cap,
        )
    else:
        sampled_entries = dataset_entries
    # Append to cumulative output
    with open(cumulative_file, 'a', encoding='utf-8') as outf:
        for row in sampled_entries:
            outf.write(json.dumps(row, ensure_ascii=False) + '\n')
    status = f"processed {len(sampled_entries)} clauses"
    if failure_reason:
        status += f" (failed run: {failure_reason})"
    print(f"{problem_name}: {status}")


def main(argv: Optional[List[str]] = None) -> None:
    """Main entry: parse args and process each problem."""
    ap = argparse.ArgumentParser(description='Batch run iProver + EA and collect labelled datasets with negative sampling.')
    ap.add_argument('--problems', required=True, help='JSON file listing problems')
    ap.add_argument('--iprover', default='iproveropt', help='iProver executable')
    ap.add_argument('--output', required=True, help='Cumulative output JSONL file')
    ap.add_argument('--limit', type=int, default=None, help='Limit number of problems to process')
    ap.add_argument('--simulate', action='store_true', help='Parse existing logs without running iProver')
    ap.add_argument('--timeout', type=int, default=300, help='Timeout for each iProver run (seconds)')
    ap.add_argument('--fail-log', default='failed_problems.jsonl', help='Path to write failed problem entries')
    ap.add_argument('--keep-failed-negatives', action='store_true', help='Include negatives from failed runs')
    # Sampling options
    ap.add_argument('--no-sample-negatives', action='store_true', help='Disable negative sampling')
    ap.add_argument('--neg-mult', type=int, default=8, help='Multiplier for negatives relative to positives')
    ap.add_argument('--neg-cap-per-problem', type=int, default=1000, help='Maximum negatives per problem')
    ap.add_argument('--fallback-neg-cap', type=int, default=300, help='Negatives cap when no positives found')
    ap.add_argument('--frontier-window', type=int, default=64, help='Number of last given_clause IDs used as frontier')
    args = ap.parse_args(argv)
    # Load problems
    with open(args.problems, 'r', encoding='utf-8') as f:
        data = json.load(f)
    if isinstance(data, list):
        problems = data
    elif isinstance(data, dict) and 'problems' in data:
        problems = data['problems']
    else:
        raise ValueError('Invalid problems JSON: expected a list or object with "problems"')
    # Prepare directories
    cumulative_path = Path(args.output).resolve()
    log_dir = cumulative_path.parent / 'logs'
    log_dir.mkdir(parents=True, exist_ok=True)
    # Ensure output exists
    if not cumulative_path.exists():
        cumulative_path.parent.mkdir(parents=True, exist_ok=True)
        cumulative_path.touch()
    processed = 0
    for problem in problems:
        if args.limit is not None and processed >= args.limit:
            break
        run_problem(
            problem,
            iprover_bin=args.iprover,
            output_dir=log_dir,
            cumulative_file=str(cumulative_path),
            simulate=args.simulate,
            timeout=args.timeout,
            fail_log_path=args.fail_log,
            keep_failed_negatives=args.keep_failed_negatives,
            no_sample_negatives=args.no_sample_negatives,
            neg_mult=args.neg_mult,
            neg_cap_per_problem=args.neg_cap_per_problem,
            fallback_neg_cap=args.fallback_neg_cap,
            frontier_window=args.frontier_window,
        )
        processed += 1
    print(f"Finished processing {processed} problem(s)")


if __name__ == '__main__':
    main()