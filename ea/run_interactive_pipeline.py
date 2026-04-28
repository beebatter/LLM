#!/usr/bin/env python3
"""End-to-end mini pipeline:
1. Launch minimal EA server (interactive_server_minimal) on a free port.
2. Launch iProver in interactive mode pointing to that port (superposition passive external_score).
3. Stream iProver stdout/stderr to log files.
4. On completion, optionally parse whether a proof / SZS success occurred.

Usage:
  python -m LLM.ea.run_interactive_pipeline \
     --iprover-bin /root/iprover-master/iproveropt \
     --problem /root/iprover-master/Examples/problem.p \
     --mode bi_then_cross --bi-ckpt ... --cross-ckpt ... --vocab ... \
     --out-dir /tmp/ea_run

Notes:
- Requires that interactive_server_minimal.py is importable as LLM.ea.interactive_server_minimal
- Uses NUL-delimited protocol consistent with minimal server
"""
from __future__ import annotations
import argparse, os, sys, socket, subprocess, time, json, shutil, textwrap
from pathlib import Path

from . import interactive_server_minimal as ea_mod


def find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Mini interactive pipeline")
    p.add_argument('--iprover-bin', required=True)
    p.add_argument('--problem', required=True)
    p.add_argument('--out-dir', required=True)
    p.add_argument('--timeout', type=int, default=300)
    # EA backend params (subset)
    p.add_argument('--mode', required=True)
    p.add_argument('--ckpt')
    p.add_argument('--bi-ckpt')
    p.add_argument('--cross-ckpt')
    p.add_argument('--llm-model')
    p.add_argument('--vocab')
    p.add_argument('--topk', type=int, default=128)
    p.add_argument('--pmi', action='store_true')
    p.add_argument('--lambda-pmi', type=float, default=0.7)
    p.add_argument('--score-norm', choices=['none','softmax','minmax','zscore'], default='none')
    p.add_argument('--temp', type=float, default=1.0)
    p.add_argument('--device', default='cuda')
    p.add_argument('--use-canonical', action='store_true')
    p.add_argument('--log-ea', action='store_true')
    p.add_argument('--sup-passive-queues-freq', default='[1]')
    p.add_argument('--sup-passive-queues', default='[[+external_score]]')
    p.add_argument('--extra-iprover-args', nargs='*', default=[])
    return p.parse_args(argv)


def launch_ea(args, port: int, run_dir: Path):
    # Build argv for minimal server (import and run in-process via subprocess to isolate GPU context)
    ea_cmd = [sys.executable, '-m', 'LLM.ea.interactive_server_minimal', '--host','127.0.0.1', '--port', str(port), '--mode', args.mode,
              '--topk', str(args.topk), '--score-norm', args.score_norm, '--temp', str(args.temp), '--device', args.device]
    if args.ckpt: ea_cmd += ['--ckpt', args.ckpt]
    if args.bi_ckpt: ea_cmd += ['--bi-ckpt', args.bi_ckpt]
    if args.cross_ckpt: ea_cmd += ['--cross-ckpt', args.cross_ckpt]
    if args.llm_model: ea_cmd += ['--llm-model', args.llm_model]
    if args.vocab: ea_cmd += ['--vocab', args.vocab]
    if args.use_canonical: ea_cmd += ['--use-canonical']
    if args.pmi: ea_cmd += ['--pmi']
    ea_log = open(run_dir/'ea_stdout.log', 'w', encoding='utf-8') if args.log_ea else subprocess.DEVNULL
    proc = subprocess.Popen(ea_cmd, stdout=ea_log, stderr=ea_log)
    return proc


def launch_iprover(args, port: int, run_dir: Path):
    out = open(run_dir/'iprover_stdout.log','w',encoding='utf-8')
    err = open(run_dir/'iprover_stderr.log','w',encoding='utf-8')
    cmd = [args.iprover_bin,
           '--interactive_mode','true', '--external_ip_address','127.0.0.1', '--external_port', str(port),
           '--schedule','none','--preprocessing_flag','false','--resolution_flag','false','--instantiation_flag','false','--superposition_flag','true',
           '--sup_iter_deepening','0','--comb_sup_deep_mult','0',
           '--sup_passive_queue_type','priority_queues',
           '--sup_passive_queues_freq', args.sup_passive_queues_freq,
           '--sup_passive_queues', args.sup_passive_queues,
           args.problem]
    cmd += args.extra_iprover_args
    proc = subprocess.Popen(cmd, stdout=out, stderr=err, cwd=Path(args.iprover_bin).resolve().parent)
    return proc


def detect_success(log_text: str) -> str:
    if '% SZS status Theorem' in log_text: return 'Theorem'
    if '% SZS status Unsatisfiable' in log_text: return 'Unsatisfiable'
    if '% SZS status Satisfiable' in log_text: return 'Satisfiable'
    if '% SZS status CounterSatisfiable' in log_text: return 'CounterSatisfiable'
    return 'Unknown'


def main(argv=None):
    args = parse_args(argv)
    run_dir = Path(args.out_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    port = find_free_port()
    print(f"[PIPE] allocating port {port}")
    ea_proc = launch_ea(args, port, run_dir)
    time.sleep(0.8)  # give EA time to bind
    ip_proc = launch_iprover(args, port, run_dir)
    start = time.time()
    try:
        while True:
            if ip_proc.poll() is not None:
                break
            if time.time() - start > args.timeout:
                print('[PIPE] timeout reached – terminating')
                ip_proc.terminate(); ea_proc.terminate()
                break
            time.sleep(1.0)
    finally:
        # Ensure EA stops once iProver done
        if ea_proc.poll() is None:
            ea_proc.terminate()
    # Summarize
    szs = 'Unknown'
    ip_out_path = run_dir/'iprover_stdout.log'
    try:
        text = ip_out_path.read_text(encoding='utf-8')
        szs = detect_success(text)
    except Exception:
        pass
    summary = {
        'port': port,
        'elapsed_sec': round(time.time()-start,3),
        'iprover_exit': ip_proc.returncode,
        'ea_exit': ea_proc.returncode,
        'szs_status': szs,
    }
    (run_dir/'summary.json').write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding='utf-8')
    print('[PIPE] summary:', summary)
    return 0

if __name__=='__main__':
    raise SystemExit(main())
