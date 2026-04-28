#!/usr/bin/env python3
"""
Download iProver result pages listed in a links CSV and save them locally for offline parsing.

Input CSV format (as produced by scrape_casc_iprover_results.py --links-out):
  problem_id, iprover_url, iprover_time_or_tag

Usage example:
  python LLM/scripts/download_iprover_pages.py \
    --links Logs/iprover/casc_fof_iprover_links.csv \
    --out-dir LLM/iprover_pages \
    --max-workers 12 --timeout 15
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, Tuple


def ensure_dir(path: str) -> None:
    d = os.path.abspath(path)
    os.makedirs(d, exist_ok=True)


def fetch_url(url: str, timeout: float) -> Tuple[Optional[str], Optional[str]]:
    try:
        from urllib.request import Request, urlopen
        req = Request(url, headers={
            "User-Agent": "Mozilla/5.0 (compatible; iProver-Downloader/1.0)",
            "Accept": "text/plain, text/html;q=0.9, */*;q=0.8",
        })
        with urlopen(req, timeout=timeout) as resp:
            charset = resp.headers.get_content_charset() or "utf-8"
            data = resp.read().decode(charset, errors="replace")
            return data, None
    except Exception as e:
        return None, str(e)


def main():
    ap = argparse.ArgumentParser(description="Download iProver pages from links CSV")
    ap.add_argument('--links', required=True, help='Path to links CSV (problem_id, iprover_url, iprover_time_or_tag)')
    ap.add_argument('--out-dir', required=True, help='Directory to save downloaded pages (as <problem_id>.txt)')
    ap.add_argument('--max-workers', type=int, default=12, help='Parallel downloads')
    ap.add_argument('--timeout', type=float, default=15.0, help='Per-request timeout in seconds')
    ap.add_argument('--limit', type=int, default=None, help='Optional limit of rows to download')
    ap.add_argument('--skip-existing', action='store_true', help='Skip files that already exist')
    args = ap.parse_args()

    ensure_dir(args.out_dir)

    rows = []
    with open(args.links, 'r', encoding='utf-8') as f:
        r = csv.DictReader(f)
        for rec in r:
            pid = rec.get('problem_id') or ''
            url = rec.get('iprover_url') or ''
            if not pid or not url:
                continue
            rows.append((pid, url))

    if args.limit:
        rows = rows[: args.limit]

    print(f"[info] To download: {len(rows)} pages -> {args.out_dir}")

    # Prepare tasks
    todo = []
    for pid, url in rows:
        out_path = os.path.join(args.out_dir, f"{pid}.txt")
        if args.skip_existing and os.path.exists(out_path):
            continue
        todo.append((pid, url, out_path))

    print(f"[info] Will download: {len(todo)} (skipping existing: {args.skip_existing})")

    ok = 0
    fail = 0
    errors = []
    with ThreadPoolExecutor(max_workers=args.max_workers) as ex:
        futs = {}
        for pid, url, out_path in todo:
            futs[ex.submit(fetch_url, url, args.timeout)] = (pid, out_path, url)
        for fut in as_completed(futs):
            pid, out_path, url = futs[fut]
            text, err = fut.result()
            if err or not text:
                fail += 1
                if len(errors) < 10:
                    errors.append((pid, url, err or 'empty'))
                continue
            try:
                with open(out_path, 'w', encoding='utf-8') as f:
                    f.write(text)
                ok += 1
            except Exception as e:
                fail += 1
                if len(errors) < 10:
                    errors.append((pid, url, f"write_error: {e}"))

    print(f"[done] downloaded={ok} failed={fail} out_dir={args.out_dir}")
    if errors:
        print("[warn] sample errors:")
        for pid, url, msg in errors:
            print(f"  - {pid}: {msg}")


if __name__ == '__main__':
    main()
