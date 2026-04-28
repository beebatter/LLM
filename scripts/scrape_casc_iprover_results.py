#!/usr/bin/env python3
"""
Scrape iProver results from a local CASC results HTML table and fetch per-problem pages
to extract SZS status and proof presence.

Inputs:
  - --html /path/to/local/results.html (e.g., LLM/iprover_FOF_result.html)
  - --out-json Logs/iprover/casc_fof_iprover_summary.json
  - --out-csv  Logs/iprover/casc_fof_iprover_summary.csv
  - --limit N  (optional) only process first N problems for quick testing
    - --pages-dir DIR  (optional) use offline pages in DIR named '<problem_id>.txt' or '.html'
    - --no-network     (optional) do not attempt network fetch; rely on --pages-dir only
    - --links-out PATH (optional) write the list of iProver links to a CSV for manual download

Outputs:
  JSON/CSV with fields:
    problem_id, iprover_url, iprover_time_or_tag, szs_status, proof_kind, has_cnf_refutation

Notes:
  - We rely on the header column that contains "iProver---3.9" to locate the correct cell per row.
  - Network timeouts are handled; unavailable pages are recorded with szs_status="FETCH_ERROR".
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from typing import List, Optional, Tuple

try:
    # Prefer stdlib
    from urllib.request import Request, urlopen
    from urllib.error import URLError, HTTPError
except Exception as e:
    print(f"[fatal] urllib not available: {e}", file=sys.stderr)
    sys.exit(2)


IPROVER_HEADER_TOKEN = "iProver---3.9"


@dataclass
class ProblemEntry:
    problem_id: str
    iprover_url: Optional[str]
    iprover_time_or_tag: Optional[str]


@dataclass
class ProblemResult:
    problem_id: str
    iprover_url: Optional[str]
    iprover_time_or_tag: Optional[str]
    szs_status: Optional[str]
    proof_kind: Optional[str]
    has_cnf_refutation: bool
    fetch_error: Optional[str] = None
    proof_fof_count: Optional[int] = None
    proof_cnf_count: Optional[int] = None


def read_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def find_iprover_col_index(html: str) -> int:
    """Return the 0-based index among the TDs corresponding to iProver column.

    The header row contains multiple <th> including the first 'First-order Theorems'.
    For data rows, there are N <td> cells, one per system column; so we return
    header_index_of_iprover - 1.
    """
    # Find header row (first <tr> under <tbody>) that contains our token
    m = re.search(r"<tbody>\s*<tr>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE)
    if not m:
        raise ValueError("Could not find header row under <tbody><tr> ... </tr>")
    header_row = m.group(1)
    ths = re.findall(r"<th[^>]*>(.*?)</th>", header_row, flags=re.DOTALL | re.IGNORECASE)
    if not ths:
        raise ValueError("Header <th> cells not found")
    ip_col_header_index = None
    for idx, th_inner in enumerate(ths):
        if IPROVER_HEADER_TOKEN in th_inner:
            ip_col_header_index = idx
            break
    if ip_col_header_index is None:
        raise ValueError(f"Header with token '{IPROVER_HEADER_TOKEN}' not found")
    # Translate header index to TD index in rows (subtract the problem column)
    td_index = ip_col_header_index - 1
    if td_index < 0:
        raise ValueError("Computed iProver TD index is negative; header parsing mismatch")
    return td_index


def parse_rows(html: str, ip_td_index: int) -> List[ProblemEntry]:
    rows: List[ProblemEntry] = []
    # Each data row begins with a <tr><th align="LEFT"> containing the problem link
    for m in re.finditer(r"<tr>\s*<th[^>]*>(.*?)</th>(.*?)</tr>", html, re.DOTALL | re.IGNORECASE):
        th_inner = m.group(1)
        tds_inner = m.group(2)
        prob_m = re.search(r"<tt>([^<]+)</tt>", th_inner)
        if not prob_m:
            # Not a problem row
            continue
        problem_id = prob_m.group(1).strip()
        # Collect TDs for this row
        tds = re.findall(r"<td[^>]*>(.*?)</td>", tds_inner, flags=re.DOTALL | re.IGNORECASE)
        if not tds or ip_td_index >= len(tds):
            # Skip rows without iProver cell
            continue
        ip_td = tds[ip_td_index]
        # Extract anchor href and label text (time or tag)
        a_m = re.search(r"<a[^>]*href=\"([^\"]+)\"[^>]*>([^<]+)</a>", ip_td, flags=re.IGNORECASE)
        ip_url = a_m.group(1).strip() if a_m else None
        ip_text = a_m.group(2).strip() if a_m else None
        rows.append(ProblemEntry(problem_id=problem_id, iprover_url=ip_url, iprover_time_or_tag=ip_text))
    return rows


def fetch_url(url: str, timeout: float = 20.0) -> Tuple[Optional[str], Optional[str]]:
    try:
        req = Request(url, headers={"User-Agent": "Mozilla/5.0 (compatible; CASC-iProver-Scraper/1.0)"})
        with urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            content = resp.read().decode(charset, errors="replace")
            return content, None
    except HTTPError as e:
        return None, f"HTTPError {e.code}: {e.reason}"
    except URLError as e:
        return None, f"URLError: {e.reason}"
    except Exception as e:
        return None, f"Exception: {e}"


def parse_iprover_page(text: str) -> Tuple[Optional[str], Optional[str], bool, Optional[int], Optional[int]]:
    # Example lines:
    # % SZS status Theorem for theBenchmark.p
    # % SZS output start CNFRefutation for theBenchmark.p
    # Multiple SZS status lines may appear (Started, then Theorem, etc.). Prefer a non-Started status,
    # otherwise take the last occurrence.
    szs_status = None
    statuses = re.findall(r"%\s*SZS\s+status\s+([A-Za-z_]+)\b", text)
    if statuses:
        # prefer last non-Started
        for s in reversed(statuses):
            if s != "Started":
                szs_status = s
                break
        if szs_status is None:
            szs_status = statuses[-1]
    proof_kind = None
    m2 = re.search(r"%\s*SZS\s+output\s+start\s+([^\n\r]+)", text)
    if m2:
        proof_kind = m2.group(1).strip()
    has_cnf = bool(re.search(r"SZS\s+output\s+start\s+CNFRefutation", text))
    # Try to extract the proof block to count items
    proof_fof = None
    proof_cnf = None
    mblock = re.search(r"%\s*SZS\s+output\s+start[^\n]*\n(.*?)%\s*SZS\s+output\s+end", text, re.DOTALL | re.IGNORECASE)
    if mblock:
        block = mblock.group(1)
        proof_fof = len(re.findall(r"^\s*fof\s*\(", block, flags=re.IGNORECASE | re.MULTILINE))
        proof_cnf = len(re.findall(r"^\s*cnf\s*\(", block, flags=re.IGNORECASE | re.MULTILINE))
    return szs_status, proof_kind, has_cnf, proof_fof, proof_cnf


def ensure_dir(path: str) -> None:
    d = os.path.dirname(path)
    if d and not os.path.exists(d):
        os.makedirs(d, exist_ok=True)


def write_json(path: str, rows: List[ProblemResult]) -> None:
    ensure_dir(path)
    with open(path, "w", encoding="utf-8") as f:
        json.dump([asdict(r) for r in rows], f, ensure_ascii=False, indent=2)


def write_csv(path: str, rows: List[ProblemResult]) -> None:
    ensure_dir(path)
    fieldnames = [
        "problem_id",
        "iprover_url",
        "iprover_time_or_tag",
        "szs_status",
        "proof_kind",
        "has_cnf_refutation",
        "proof_fof_count",
        "proof_cnf_count",
        "fetch_error",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(asdict(r))


def write_links_csv(path: str, entries: List[ProblemEntry]) -> None:
    ensure_dir(path)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["problem_id", "iprover_url", "iprover_time_or_tag"])
        for e in entries:
            w.writerow([e.problem_id, e.iprover_url or "", e.iprover_time_or_tag or ""])


def main():
    ap = argparse.ArgumentParser(description="Scrape iProver SZS statuses from CASC results HTML")
    ap.add_argument("--html", required=True, help="Path to local CASC results HTML (FOF)")
    ap.add_argument("--out-json", default="Logs/iprover/casc_fof_iprover_summary.json")
    ap.add_argument("--out-csv", default="Logs/iprover/casc_fof_iprover_summary.csv")
    ap.add_argument("--limit", type=int, default=None, help="Only process first N problems")
    ap.add_argument("--max-workers", type=int, default=12)
    ap.add_argument("--pages-dir", default=None, help="Directory with offline pages named '<problem_id>.txt' or '.html'")
    ap.add_argument("--no-network", action="store_true", help="Disable network fetching; rely on --pages-dir only")
    ap.add_argument("--links-out", default=None, help="Write CSV with (problem_id, iprover_url, iprover_time_or_tag)")
    args = ap.parse_args()

    html = read_file(args.html)
    ip_td_index = find_iprover_col_index(html)
    entries = parse_rows(html, ip_td_index)
    if args.limit is not None:
        entries = entries[: args.limit]

    print(f"[info] Parsed {len(entries)} problem rows; iProver TD index={ip_td_index}")

    if args.links_out:
        write_links_csv(args.links_out, entries)
        print(f"[info] Wrote links CSV -> {args.links_out}")

    results: List[ProblemResult] = []

    # Helper to read offline page if present
    def read_offline(e: ProblemEntry) -> Optional[str]:
        if not args.pages_dir:
            return None
        base = os.path.join(args.pages_dir, e.problem_id)
        for ext in (".txt", ".html", ".htm", ""):
            p = base + ext
            if os.path.isfile(p):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        return f.read()
                except Exception:
                    try:
                        with open(p, "r", encoding="latin-1") as f:
                            return f.read()
                    except Exception:
                        return None
        return None

    # First pass: populate results from offline pages (if any)
    pending: List[ProblemEntry] = []
    for e in entries:
        offline_text = read_offline(e)
        if offline_text is not None:
            szs_status, proof_kind, has_cnf, pfof, pcnf = parse_iprover_page(offline_text)
            results.append(ProblemResult(
                problem_id=e.problem_id,
                iprover_url=e.iprover_url,
                iprover_time_or_tag=e.iprover_time_or_tag,
                szs_status=szs_status,
                proof_kind=proof_kind,
                has_cnf_refutation=has_cnf,
                proof_fof_count=pfof,
                proof_cnf_count=pcnf,
                fetch_error=None,
            ))
        else:
            pending.append(e)

    # If network disabled, record NO_FILE / NO_URL for remaining
    if args.no_network:
        for e in pending:
            results.append(ProblemResult(
                problem_id=e.problem_id,
                iprover_url=e.iprover_url,
                iprover_time_or_tag=e.iprover_time_or_tag,
                szs_status=None,
                proof_kind=None,
                has_cnf_refutation=False,
                fetch_error="NO_FILE" if e.iprover_url else "NO_URL",
            ))
    else:
        # Fetch remaining in parallel
        with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
            fut_to_entry = {}
            for e in pending:
                if e.iprover_url:
                    fut = ex.submit(fetch_url, e.iprover_url)
                    fut_to_entry[fut] = e
                else:
                    results.append(ProblemResult(
                        problem_id=e.problem_id,
                        iprover_url=None,
                        iprover_time_or_tag=e.iprover_time_or_tag,
                        szs_status=None,
                        proof_kind=None,
                        has_cnf_refutation=False,
                        fetch_error="NO_URL",
                    ))
            for fut in as_completed(fut_to_entry):
                e = fut_to_entry[fut]
                text, err = fut.result()
                if err or not text:
                    results.append(ProblemResult(
                        problem_id=e.problem_id,
                        iprover_url=e.iprover_url,
                        iprover_time_or_tag=e.iprover_time_or_tag,
                        szs_status=None,
                        proof_kind=None,
                        has_cnf_refutation=False,
                        fetch_error=err or "EMPTY",
                    ))
                else:
                    szs_status, proof_kind, has_cnf, pfof, pcnf = parse_iprover_page(text)
                    results.append(ProblemResult(
                        problem_id=e.problem_id,
                        iprover_url=e.iprover_url,
                        iprover_time_or_tag=e.iprover_time_or_tag,
                        szs_status=szs_status,
                        proof_kind=proof_kind,
                        has_cnf_refutation=has_cnf,
                        proof_fof_count=pfof,
                        proof_cnf_count=pcnf,
                    ))

    # Sort by problem_id for stability
    results.sort(key=lambda r: r.problem_id)

    # Write outputs
    write_json(args.out_json, results)
    write_csv(args.out_csv, results)

    # Summary
    solved = [r for r in results if r.szs_status in {"Theorem", "Unsatisfiable", "ContradictoryAxioms"}]
    with_proof = [r for r in results if r.has_cnf_refutation]
    errors = [r for r in results if r.fetch_error]
    print(f"[done] total={len(results)} solved={len(solved)} with_cnf_refutation={len(with_proof)} errors={len(errors)}")
    if errors:
        print("[warn] sample errors:")
        for r in errors[:5]:
            print(f"  - {r.problem_id}: {r.fetch_error}")


if __name__ == "__main__":
    main()
