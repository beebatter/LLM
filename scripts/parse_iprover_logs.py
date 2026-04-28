#!/usr/bin/env python3
"""
Parse iProver logs (from scripts/run_iprover_logged.sh) to extract baseline metrics:

Outputs per-problem and an aggregate summary:
- solved (bool), status, total_time (s)
- proof_steps (count of tcf/cnf lines in CNFRefutation)
- inst/sup/res passive sizes (final snapshot; proxy for passive)
- inst/sup/res loop counters

Note: iProver does not print peak/mean passive sizes over time in standard logs.
We report the final snapshot numbers as proxies and mark Peak/Mean passive as N/A
in the aggregate unless a better signal is present.
"""
import argparse
import json
import os
import re
import statistics as stats
from typing import Dict, Any, List


def parse_log(path: str) -> Dict[str, Any]:
    data: Dict[str, Any] = {
        'file': os.path.basename(path),
        'status': 'Unknown',
        'solved': False,
        'total_time': None,
        'proof_steps': 0,
        'inst_num_in_passive': None,
        'sup_num_in_passive': None,
        'res_num_in_passive': None,
        'inst_num_of_loops': None,
        'sup_num_of_loops': None,
        'res_num_of_loops': None,
        'interactive_mode': None,
    }

    try:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            text = f.read()
    except Exception as e:
        data['error'] = str(e)
        return data

    # Determine status
    m = re.search(r"%\s*SZS status\s+(\w+)", text)
    if m:
        data['status'] = m.group(1)
        data['solved'] = data['status'] in {'Theorem', 'Unsatisfiable', 'Satisfiable'}

    # Interactive mode flag (baseline should be false)
    m2 = re.search(r"--interactive_mode\s+\s*(true|false)", text)
    if m2:
        data['interactive_mode'] = (m2.group(1).lower() == 'true')

    # Total time
    mt = re.search(r"total_time:\s+([0-9.]+)", text)
    if mt:
        try:
            data['total_time'] = float(mt.group(1))
        except Exception:
            pass

    # Proof steps: count lines between start/end markers
    # Lines usually start with 'tcf(' or 'cnf(' in the refutation block
    ref_start = re.search(r"^%\s*SZS output start CNFRefutation.*$", text, re.M)
    ref_end = re.search(r"^%\s*SZS output end CNFRefutation.*$", text, re.M)
    if ref_start and ref_end and ref_end.start() > ref_start.end():
        block = text[ref_start.end():ref_end.start()]
        steps = 0
        for line in block.splitlines():
            if re.match(r"\s*(tcf|cnf)\(", line):
                steps += 1
        data['proof_steps'] = steps

    # Passive sizes (final snapshot)
    def grab_int(label: str):
        m = re.search(rf"{label}:\s+(-?\d+)", text)
        return int(m.group(1)) if m else None

    data['inst_num_in_passive'] = grab_int('inst_num_in_passive')
    data['sup_num_in_passive'] = grab_int('sup_num_in_passive')
    data['res_num_in_passive'] = grab_int('res_num_in_passive')
    data['inst_num_of_loops'] = grab_int('inst_num_of_loops')
    data['sup_num_of_loops'] = grab_int('sup_num_of_loops')
    data['res_num_of_loops'] = grab_int('res_num_of_loops')

    return data


def aggregate(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    solved = [r for r in rows if r.get('solved')]
    agg: Dict[str, Any] = {
        'count': len(rows),
        'solved': len(solved),
    }
    # Mean proof steps over solved
    steps = [r['proof_steps'] for r in solved if r.get('proof_steps')]
    agg['mean_proof_steps'] = round(stats.mean(steps), 3) if steps else None
    # Mean wall time over solved
    times = [r['total_time'] for r in solved if r.get('total_time') is not None]
    agg['mean_wall_time_solved_s'] = round(stats.mean(times), 3) if times else None
    # Passive sizes (report mean of final snapshot as a proxy)
    def mean_or_none(vals):
        vals = [v for v in vals if v is not None]
        return round(stats.mean(vals), 3) if vals else None
    agg['mean_inst_passive_final'] = mean_or_none([r.get('inst_num_in_passive') for r in solved])
    agg['mean_sup_passive_final'] = mean_or_none([r.get('sup_num_in_passive') for r in solved])
    agg['mean_res_passive_final'] = mean_or_none([r.get('res_num_in_passive') for r in solved])
    # Loop counters (sum over categories; proxy for looping activity, not looping cases)
    def sum_or_zero(vals):
        return sum([v for v in vals if isinstance(v, int)])
    agg['sum_inst_loops'] = sum_or_zero([r.get('inst_num_of_loops') for r in rows])
    agg['sum_sup_loops'] = sum_or_zero([r.get('sup_num_of_loops') for r in rows])
    agg['sum_res_loops'] = sum_or_zero([r.get('res_num_of_loops') for r in rows])
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--dir', default='Logs/iprover', help='Directory with *.log files')
    ap.add_argument('--baseline-only', action='store_true', help='Filter to interactive_mode=false runs')
    ap.add_argument('--json-out', default='', help='Optional path to write JSON with rows and aggregate')
    args = ap.parse_args()

    if not os.path.isdir(args.dir):
        print(f"No directory: {args.dir}")
        return 1
    logs = [os.path.join(args.dir, f) for f in os.listdir(args.dir) if f.endswith('.log')]
    logs.sort()
    rows = []
    for p in logs:
        r = parse_log(p)
        if args.baseline_only and r.get('interactive_mode') is True:
            continue
        rows.append(r)

    agg = aggregate(rows)
    # Print a compact table-like summary
    print("files:", len(rows), "solved:", agg['solved'], "of", agg['count'])
    print("mean proof steps (solved):", agg.get('mean_proof_steps'))
    print("mean wall time / solved (s):", agg.get('mean_wall_time_solved_s'))
    print("mean passive final (inst/sup/res):",
          agg.get('mean_inst_passive_final'),
          agg.get('mean_sup_passive_final'),
          agg.get('mean_res_passive_final'))
    print("sum loops (inst/sup/res):",
          agg.get('sum_inst_loops'),
          agg.get('sum_sup_loops'),
          agg.get('sum_res_loops'))

    if args.json_out:
        with open(args.json_out, 'w') as f:
            json.dump({'rows': rows, 'aggregate': agg}, f, indent=2)
        print('wrote', args.json_out)


if __name__ == '__main__':
    raise SystemExit(main())
