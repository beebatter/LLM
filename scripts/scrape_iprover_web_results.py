#!/usr/bin/env python3
"""
Scrape iProver results from a CASC results HTML table and extract proof-related data.

Inputs:
- --html: Path to a local CASC results HTML file (e.g., iprover_FOF_result.html)
- --system: System identifier to select the column (default: iProver---3.9)
- --limit: Max number of problems to process (optional)
- --out-json: Output JSON path
- --out-csv: Output CSV path
- --offline-dir: Optional directory with per-problem logs (e.g., SWW473+1.txt) to use instead of network

Behavior:
- Parse problem rows, extract problem ID and the iProver result link + time/value from that cell
- Fetch each iProver result page (or read offline file if available)
- Parse SZS status, presence of CNFRefutation, and count proof steps (cnf( lines) within the CNFRefutation block
- Emit JSON and CSV summaries
"""
from __future__ import annotations

import argparse
import csv
import html
import json
import os
import re
import sys
from typing import List, Dict, Optional, Tuple


def load_text(path: str) -> str:
    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def find_problem_rows(doc: str) -> List[Tuple[str, str]]:
    """
    Return list of (row_html, problem_id) for each problem row in the table.
    """
    rows: List[Tuple[str, str]] = []
    # Pattern to find row start with problem id in <tt>PROB</tt>
    row_iter = re.finditer(r"<tr><th[^>]*>\s*<a[^>]*>\s*<tt>([A-Z0-9\+]+)</tt>\s*</a>", doc)
    for m in row_iter:
        prob = m.group(1)
        start = m.start()
        # find the end of this row by nearest closing </tr>
        end = doc.find("</tr>", start)
        if end == -1:
            continue
        row_html = doc[start:end]
        rows.append((row_html, prob))
    return rows


def extract_iprover_cell(row_html: str, system: str) -> Optional[Tuple[str, str]]:
    """
    From a row_html, extract (url, cell_value_text) for the given system column.
    System is like 'iProver---3.9'. The link pattern is '/Results/.../iProver---3.9/PROB'.
    """
    # Find href with system
    href_match = re.search(r'href="(https?://[^\"]+/Results/[^\"]+/%s/[^\"]+)"[^>]*>([^<]+)</a>' % re.escape(system), row_html)
    if not href_match:
        return None
    url = html.unescape(href_match.group(1))
    value_text = html.unescape(href_match.group(2)).strip()
    return url, value_text


def fetch_url(url: str) -> Optional[str]:
    """
    Fetch a URL content using stdlib. Avoid external dependencies.
    """
    try:
        import urllib.request
        with urllib.request.urlopen(url, timeout=20) as resp:
            charset = resp.headers.get_content_charset() or 'utf-8'
            return resp.read().decode(charset, errors='ignore')
    except Exception as e:
        print(f"[warn] fetch failed {url}: {e}", file=sys.stderr)
        return None


def parse_iprover_page(text: str) -> Dict[str, Optional[object]]:
    """
    Extract SZS status and CNFRefutation details from a page text.
    Works if the page is plain text or HTML containing the text (we strip tags crudely).
    """
    # If HTML, drop tags to expose text
    if '<' in text and '>' in text:
        # crude removal of tags
        text_no_tags = re.sub(r'<[^>]+>', '\n', text)
    else:
        text_no_tags = text

    # Normalize whitespace
    lines = [ln.strip() for ln in text_no_tags.splitlines()]
    norm = "\n".join(lines)

    # SZS status: sometimes appears multiple times (Started, then Theorem). Take the last occurrence.
    matches = re.findall(r"SZS\s+status\s+([A-Za-z_\-]+)", norm)
    status = matches[-1] if matches else None

    # Find CNFRefutation block
    start_m = re.search(r"SZS\s+output\s+start\s+CNFRefutation", norm, re.IGNORECASE)
    end_m = re.search(r"SZS\s+output\s+end\s+CNFRefutation", norm, re.IGNORECASE)
    has_proof = bool(start_m and end_m and end_m.start() > start_m.end())
    proof_steps = None
    cnf_lines = None
    fof_lines = None
    block = None
    if has_proof:
        block = norm[start_m.end(): end_m.start()]
        # Count cnf( and fof( occurrences as a proxy for proof steps. Use word boundary;
        # CASC logs often prefix lines with timestamps, so don't anchor at line start.
        cnf_lines = len(re.findall(r"(?i)\bcnf\(", block))
        fof_lines = len(re.findall(r"(?i)\bfof\(", block))
        # Use cnf count as 'proof_steps' proxy
        proof_steps = cnf_lines

    # Extract passive sizes if present anywhere in the text (outside or inside the block)
    # Examples that we try to capture (case-insensitive):
    #  "Peak passive size: 1234" or "Peak passive size = 1234"
    #  "Mean passive size: 56.78"
    peak_passive_size = None
    mean_passive_size = None
    m_peak = re.search(r"(?i)peak\s+passive\s+size\s*[:=]\s*([0-9]+)", norm)
    if m_peak:
        try:
            peak_passive_size = int(m_peak.group(1))
        except Exception:
            peak_passive_size = None
    m_mean = re.search(r"(?i)mean\s+passive\s+size\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", norm)
    if m_mean:
        try:
            mean_passive_size = float(m_mean.group(1))
        except Exception:
            mean_passive_size = None

    return {
        'status': status,
        'has_proof': has_proof,
        'proof_steps': proof_steps,
        'cnf_lines': cnf_lines,
        'fof_lines': fof_lines,
        'peak_passive_size': peak_passive_size,
        'mean_passive_size': mean_passive_size,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--html', required=True, help='Path to CASC results HTML file')
    ap.add_argument('--system', default='iProver---3.9', help='System id to extract (default: iProver---3.9)')
    ap.add_argument('--limit', type=int, default=None, help='Max number of problems to process')
    ap.add_argument('--out-json', default='Logs/iprover/web_results.json', help='Output JSON path')
    ap.add_argument('--out-csv', default='Logs/iprover/web_results.csv', help='Output CSV path')
    ap.add_argument('--offline-dir', default='LLM', help='Directory with optional per-problem .txt logs (e.g., SWW473+1.txt)')
    ap.add_argument('--offline-only', action='store_true', help='Do not fetch from the web; only use local per-problem logs if present')
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)

    doc = load_text(args.html)
    rows = find_problem_rows(doc)
    if args.limit:
        rows = rows[: args.limit]

    results: List[Dict[str, object]] = []
    for i, (row_html, prob) in enumerate(rows, start=1):
        cell = extract_iprover_cell(row_html, args.system)
        if not cell:
            # skip if no iProver cell
            continue
        url, cell_value = cell
        cell_time: Optional[float] = None
        # Try to parse numeric time from the cell value
        try:
            cell_time = float(cell_value)
        except Exception:
            cell_time = None

        # Prefer local offline file if available
        offline_path = None
        for base in [args.offline_dir, os.path.dirname(args.html)]:
            cand = os.path.join(base, f"{prob}.txt")
            if os.path.isfile(cand):
                offline_path = cand
                break

        if offline_path:
            text = load_text(offline_path)
        else:
            if args.offline_only:
                # Skip this problem; no offline data and network disabled
                continue
            text = fetch_url(url) or ''

        parsed = parse_iprover_page(text) if text else {
            'status': None,
            'has_proof': False,
            'proof_steps': None,
            'cnf_lines': None,
            'fof_lines': None,
        }

        rec = {
            'idx': i,
            'problem': prob,
            'url': url,
            'cell_value': cell_value,
            'time_sec': cell_time,
            **parsed,
        }
        results.append(rec)

    # Save JSON
    with open(args.out_json, 'w', encoding='utf-8') as f:
        json.dump({'system': args.system, 'count': len(results), 'rows': results}, f, ensure_ascii=False, indent=2)

    # Save CSV
    cols = ['idx', 'problem', 'time_sec', 'cell_value', 'status', 'has_proof', 'proof_steps', 'cnf_lines', 'fof_lines', 'peak_passive_size', 'mean_passive_size', 'url']
    with open(args.out_csv, 'w', encoding='utf-8', newline='') as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k) for k in cols})

    # Console summary
    status_counts: Dict[str, int] = {}
    proof_yes = 0
    for r in results:
        s = r.get('status') or 'UNKNOWN'
        status_counts[s] = status_counts.get(s, 0) + 1
        if r.get('has_proof'):
            proof_yes += 1

    print(f"Processed {len(results)} problems for system {args.system}")
    print("SZS status counts:")
    for s, c in sorted(status_counts.items(), key=lambda x: (-x[1], x[0])):
        print(f"  {s}: {c}")
    print(f"Proofs found (CNFRefutation block present): {proof_yes}")
    print(f"Wrote JSON: {args.out_json}")
    print(f"Wrote CSV:  {args.out_csv}")

    # Aggregate statistics
    def _avg(vals):
        return sum(vals) / len(vals) if vals else None
    def _median(vals):
        if not vals:
            return None
        vs = sorted(vals)
        n = len(vs)
        mid = n // 2
        if n % 2 == 1:
            return vs[mid]
        return (vs[mid - 1] + vs[mid]) / 2

    proof_lengths = [int(r['proof_steps']) for r in results if r.get('proof_steps') is not None]
    peaks = [int(r['peak_passive_size']) for r in results if r.get('peak_passive_size') is not None]
    means = [float(r['mean_passive_size']) for r in results if r.get('mean_passive_size') is not None]

    avg_len = _avg(proof_lengths)
    med_len = _median(proof_lengths)
    avg_peak = _avg(peaks)
    med_peak = _median(peaks)
    avg_mean_pass = _avg(means)
    med_mean_pass = _median(means)

    print("Summary metrics:")
    print(f"  Proof length count: {len(proof_lengths)}; average: {avg_len}; median: {med_len}")
    if peaks:
        print(f"  Peak passive size: n={len(peaks)}; average: {avg_peak}; median: {med_peak}")
    else:
        print("  Peak passive size: n=0 (no data found)")
    if means:
        print(f"  Mean passive size: n={len(means)}; average: {avg_mean_pass}; median: {med_mean_pass}")
    else:
        print("  Mean passive size: n=0 (no data found)")


if __name__ == '__main__':
    main()
