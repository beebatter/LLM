#!/usr/bin/env python3
"""Interactive EA server integrating Transformer & LLM scorers.

Protocol (iProver interactive style, minimal subset):
  Incoming JSON messages (NUL or NEWLINE delimited) with tags:
    - register_clauses: {"tag":"register_clauses", "clauses":[{"clause_id":int, "clause":str, "clause_features":{...}}], "component":str?, "component_id":int?}
    - scores_req: {"tag":"scores_req", "clause_ids":[...], "component":..., "component_id":...}
    - terminate (optional)

  Outgoing:
    - scores_res: {"tag":"scores_res", "scores":[float...], "component":..., "component_id":...}
    - ack: generic acknowledgement

Scoring backends loaded from `ea.backends` (Clause / Cross / Bi / Bi+LLM / LLM direct / PMI / Fusion).

Example:
  python -m LLM.ea.interactive_server serve \
    --host 127.0.0.1 --port 12346 \
    --mode bi_then_llm --bi-ckpt /path/bi.pt --llm-model /root/autodl-tmp/models/Goedel-Prover-V2-32B \
    --topk 128 --pmi --lambda-pmi 0.7 --verbose

Fusion usage (pre-computed JSON score maps id->score):
  python -m LLM.ea.interactive_server serve \
    --host 127.0.0.1 --port 12346 --mode fusion \
    --fusion-files ce_scores.json bi_scores.json \
    --fusion-weights 0.6 0.4

Notes:
  - This is a lightweight server; disable for production or add batching if clause count large.
  - Clause text currently uses raw "clause" string; you can swap to canonical form if available.
"""
from __future__ import annotations

import argparse
import json
import os
import socket
import sys
import time
from typing import Dict, Any, List

from .backends import load_backend
try:  # canonical helpers
    from LLM.process_iprover_v3 import preprocess_clause_str, extract_formula_from_tcf
except Exception:  # pragma: no cover
    def preprocess_clause_str(s: str) -> str:  # fallback no-op
        return s
    def extract_formula_from_tcf(s: str) -> str:
        return s


def log(msg: str, verbose: bool):  # simple logger
    if verbose:
        ts = time.strftime('%H:%M:%S')
        print(f"[{ts}] {msg}", flush=True)


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Interactive EA unified scorer")
    ap.add_argument('command', choices=['serve'])
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, required=True)
    ap.add_argument('--mode', required=True,
                    choices=['clause_tf','cross_tf','bi_tf','bi_then_llm','bi_then_cross','llm_direct','llm_pmi','fusion'])
    # transformer ckpts
    ap.add_argument('--ckpt', help='Transformer checkpoint (clause/cross/bi)')
    ap.add_argument('--bi-ckpt', help='BiEncoder ckpt for bi_then_llm / bi_then_cross')
    ap.add_argument('--cross-ckpt', help='CrossEncoder ckpt for bi_then_cross')
    ap.add_argument('--llm-model', help='HF model path for llm modes / bi_then_llm')
    ap.add_argument('--vocab', help='SentencePiece .vocab file for transformer tokenization (optional)')
    ap.add_argument('--topk', type=int, default=128, help='Shortlist size for bi_then_llm')
    ap.add_argument('--pmi', action='store_true', help='Use PMI in bi_then_llm')
    ap.add_argument('--lambda-pmi', type=float, default=0.7)
    # fusion
    ap.add_argument('--fusion-files', nargs='+')
    ap.add_argument('--fusion-weights', nargs='+', type=float)
    # runtime
    ap.add_argument('--device', default='cuda')
    ap.add_argument('--verbose', action='store_true')
    ap.add_argument('--exit-after-scores', action='store_true', help='Debug: exit after first scores_res')
    # score normalization
    ap.add_argument('--score-norm', choices=['none','softmax','minmax','zscore'], default='none',
                    help='Post-process scores before returning (primarily for bi_tf large dot products)')
    ap.add_argument('--temp', type=float, default=1.0, help='Temperature for softmax normalization')
    # long sequence handling (bi encoders)
    ap.add_argument('--bi-chunk-encode', action='store_true', help='Enable chunked encoding for long sequences in bi encoders')
    ap.add_argument('--bi-chunk-len', type=int, default=256, help='Chunk window length')
    ap.add_argument('--bi-chunk-stride', type=int, default=192, help='Chunk stride (overlap = len - stride)')
    ap.add_argument('--bi-chunk-max', type=int, default=4, help='Maximum number of chunks per sequence')
    ap.add_argument('--bi-chunk-agg', choices=['mean','max','attn'], default='mean', help='Aggregation over chunk vectors')
    ap.add_argument('--max-len-override', type=int, default=2048, help='Truncate only if length exceeds this (applies to Bi/Cross encoders)')
    # canonical / preprocessing
    ap.add_argument('--use-canonical', action='store_true', help='Use cleaned formula (preprocess+extract) for scoring and conjecture')
    # logging & artifacts
    ap.add_argument('--print-io', action='store_true', help='Print raw IN/OUT JSON lines with prefixes')
    ap.add_argument('--log-dir', default=None, help='Base directory to store artifacts of requests/responses')
    ap.add_argument('--save-requests', action='store_true', help='Persist each register_clauses / scores_req / scores_res as separate JSON file')
    # conjecture fallback
    ap.add_argument('--fallback-conjecture', choices=['first','shortest','longest'], default='first',
                    help='Heuristic if no clause has conj_dist==0 (pick clause text)')
    ap.add_argument('--require-conjecture', action='store_true', help='If set, delay scoring until conjecture discovered')
    # server queries handling
    ap.add_argument('--auto-server-queries-end', action='store_true',
                    help='Automatically reply with {"tag":"server_queries_end"} after server_queries_start to release iProver loop')
    ap.add_argument('--scores-before-end', action='store_true',
                    help='When set (default new behavior), send scores_res before server_queries_end. If unset, old behavior (end then scores). Use to test protocol expectations.')
    ap.add_argument('--no-ack', action='store_true',
                    help='Do not send any ack messages (iProver reference implementation does not expect them). Recommended to keep ON for current iProver builds.')
    # delimiter compatibility: some iProver builds expect NEWLINE-delimited JSON (no NUL). Default to newline for safety.
    ap.add_argument('--delimiter', choices=['newline','nul','both'], default='newline',
                    help='Message framing delimiter. Use newline for maximum compatibility. "nul" for NUL (\x00). "both" appends NUL then newline.')
    # auto-exit policy
    ap.add_argument('--auto-exit-on-szs', action='store_true', default=True,
                    help='Exit EA when iProver sends final SZS status or proof_out (default: on)')
    ap.add_argument('--auto-exit-on-timeout', action='store_true', default=True,
                    help='Exit EA when iProver signals timeout or closes the connection (default: on)')
    return ap.parse_args(argv)


class BackendWrapper:
    def __init__(self, args):
        m = args.mode
        if m == 'fusion':
            assert args.fusion_files and args.fusion_weights, 'fusion requires --fusion-files & --fusion-weights'
            # load all score maps: expect JSON: {problem_name: {clause_id: score, ...}} OR flat lines? Simplify: each file is JSONL lines with {problem_name, scores:[{id,score}]}
            src_maps = []
            for p in args.fusion_files:
                mm = {}
                with open(p, 'r', encoding='utf-8', errors='ignore') as f:
                    for ln in f:
                        try:
                            obj = json.loads(ln)
                        except Exception:
                            continue
                        for e in obj.get('scores', []):
                            mm[e['id']] = e['score']
                src_maps.append(mm)
            from .backends import load_backend as _lb
            self.backend = _lb('fusion', sources=src_maps, weights=args.fusion_weights)
        elif m == 'bi_then_llm':
            assert args.bi_ckpt and args.llm_model
            self.backend = load_backend('bi_then_llm', bi_ckpt=args.bi_ckpt, llm=args.llm_model,
                                        topk=args.topk, pmi=args.pmi, lambda_pmi=args.lambda_pmi,
                                        device=args.device,
                                        chunk_encode=args.bi_chunk_encode,
                                        chunk_len=args.bi_chunk_len,
                                        chunk_stride=args.bi_chunk_stride,
                                        chunk_max=args.bi_chunk_max,
                                        chunk_agg=args.bi_chunk_agg,
                                        override_max_len=args.max_len_override)
        elif m == 'bi_then_cross':
            assert args.bi_ckpt and args.cross_ckpt
            self.backend = load_backend('bi_then_cross', bi_ckpt=args.bi_ckpt, cross_ckpt=args.cross_ckpt,
                                        topk=args.topk, device=args.device, vocab=args.vocab,
                                        chunk_encode=args.bi_chunk_encode,
                                        chunk_len=args.bi_chunk_len,
                                        chunk_stride=args.bi_chunk_stride,
                                        chunk_max=args.bi_chunk_max,
                                        chunk_agg=args.bi_chunk_agg,
                                        override_max_len=args.max_len_override)
        elif m in ('clause_tf','cross_tf','bi_tf'):
            assert args.ckpt, f'{m} requires --ckpt'
            if m == 'bi_tf':
                self.backend = load_backend(m, ckpt=args.ckpt, device=args.device, vocab=args.vocab,
                                            chunk_encode=args.bi_chunk_encode,
                                            chunk_len=args.bi_chunk_len,
                                            chunk_stride=args.bi_chunk_stride,
                                            chunk_max=args.bi_chunk_max,
                                            chunk_agg=args.bi_chunk_agg,
                                            override_max_len=args.max_len_override)
            else:
                self.backend = load_backend(m, ckpt=args.ckpt, device=args.device, vocab=args.vocab,
                                            override_max_len=args.max_len_override)
        elif m in ('llm_direct','llm_pmi'):
            assert args.llm_model
            kw = {'llm': args.llm_model}
            if m == 'llm_pmi':
                kw['lambda_pmi'] = args.lambda_pmi
            self.backend = load_backend(m, **kw)
        else:
            raise ValueError(f'Unsupported mode {m}')

    def score(self, conjecture: str, candidates: List[Dict[str, Any]]):
        return self.backend.score(conjecture, candidates)


def serve(args):
    backend = BackendWrapper(args)
    # network init
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    # Match reference log style (no timestamp) for initial lines
    print(f"[EA] listening on {args.host}:{args.port} (delimiter=NUL+NEWLINE)", flush=True)

    # artifacts directory setup
    run_dir = None
    if args.log_dir:
        import time as _t, pathlib
        ts = int(_t.time()*1000)
        run_dir = pathlib.Path(args.log_dir)/f"EA.{os.getpid()}.{ts}"
        try:
            run_dir.mkdir(parents=True, exist_ok=True)
            # Reference style: "[EA] logs dir: <path>"
            print(f"[EA] logs dir: {run_dir}", flush=True)
        except Exception as e:  # pragma: no cover
            print(f"[EA] failed to create log dir {run_dir}: {e}", flush=True)
    req_counter = 0
    conn, addr = srv.accept()
    print(f"[EA] connected from {addr}", flush=True)
    conn.settimeout(0.5)

    # state
    clause_index: Dict[int, Dict[str, Any]] = {}
    conjecture_formula: str = ''  # simple heuristic: first clause with conj_dist==0, else fallback
    fallback_pool: Dict[int, str] = {}
    buffer = b''
    running = True
    printed_ranker_python = False
    req_artifact_counter = 0

    def send(obj: Dict[str, Any]):
        # Configurable delimiter: many iProver builds split on NEWLINE only; NUL causes Yojson "Junk after end" if concatenated.
        if args.delimiter == 'nul':
            delim = '\x00'
        elif args.delimiter == 'both':
            delim = '\x00\n'
        else:  # newline
            delim = '\n'
        data = (json.dumps(obj, ensure_ascii=False) + delim).encode('utf-8')
        try:
            conn.sendall(data)
        except Exception:
            pass

    # Protocol state for ordering: hold scores_req until server_queries_start arrives so that
    # ordering becomes: register_clauses -> scores_req (buffered) -> server_queries_start -> server_queries_end -> scores_res
    pending_scores_req: List[Dict[str, Any]] = []
    while running:
        # read
        try:
            chunk = conn.recv(65536)
            if chunk is None:
                time.sleep(0.01)
            elif len(chunk) == 0:
                # peer closed connection
                if args.verbose:
                    log('[EA] peer closed connection', True)
                if args.auto_exit_on_timeout:
                    running = False
                break
            else:
                buffer += chunk
        except socket.timeout:
            pass
        except Exception:
            break

        # split by NUL first; if none, fall back to newline
        while True:
            if b'\x00' in buffer:
                part, buffer = buffer.split(b'\x00', 1)
            elif b'\n' in buffer:
                part, buffer = buffer.split(b'\n', 1)
            else:
                break
            part = part.strip()
            if not part:
                continue
            raw_txt = part.decode('utf-8', errors='ignore')
            if args.print_io:
                short = raw_txt if len(raw_txt) <= 120 else (raw_txt[:120] + '…')
                print(f"[EA IN] {short}", flush=True)
            try:
                msg = json.loads(raw_txt)
            except Exception:
                log(f"[WARN] malformed JSON: {raw_txt[:120]!r}", args.verbose)
                continue
            tag = msg.get('tag')
            if tag == 'register_clauses':
                for entry in msg.get('clauses', []):
                    cid = entry.get('clause_id')
                    if cid is None:
                        continue
                    raw_clause = entry.get('clause') or ''
                    if args.use_canonical:
                        cleaned = preprocess_clause_str(raw_clause)
                        formula = extract_formula_from_tcf(cleaned)
                        canon = formula or raw_clause
                    else:
                        canon = raw_clause
                    clause_index[cid] = {
                        'text': raw_clause,
                        'canon': canon,
                        'features': entry.get('clause_features', {})
                    }
                    feat = clause_index[cid]['features']
                    if feat and feat.get('conj_dist') == 0 and not conjecture_formula:
                        conjecture_formula = clause_index[cid]['canon' if args.use_canonical else 'text']
                    # accumulate for potential fallback if no conj_dist==0 appears
                    if feat and feat.get('conj_dist', 9999) != 0:
                        fallback_pool[cid] = clause_index[cid]['canon' if args.use_canonical else 'text']
                if (not args.no_ack):
                    send({'tag': 'ack', 'what': 'register_clauses', 'count': len(msg.get('clauses', []))})
                if args.save_requests and run_dir:
                    import pathlib, time as _t
                    req_counter += 1
                    (run_dir/"requests").mkdir(exist_ok=True)
                    with open(run_dir/"requests"/f"register_{req_counter}.json", 'w', encoding='utf-8') as f:
                        json.dump(msg, f, ensure_ascii=False)
                if not conjecture_formula and fallback_pool:
                    # apply heuristic selection
                    if args.fallback_conjecture == 'first':
                        sel_id = sorted(fallback_pool.keys())[0]
                    elif args.fallback_conjecture == 'shortest':
                        sel_id = min(fallback_pool.keys(), key=lambda k: len(fallback_pool[k]))
                    else:  # longest
                        sel_id = max(fallback_pool.keys(), key=lambda k: len(fallback_pool[k]))
                    conjecture_formula = fallback_pool[sel_id]
                    if args.verbose:
                        log(f"[EA] Fallback conjecture <- clause_id={sel_id} len={len(conjecture_formula)}", True)
            elif tag == 'scores_req':
                # Buffer the request to be served after server_queries_start (to match expected ordering)
                pending_scores_req.append(msg)
                if args.print_io:
                    print(f"[EA] buffered scores_req (count={len(pending_scores_req)})", flush=True)
            elif tag in ('szs_result_out', 'szs_status_out', 'proof_out'):
                # Detect SZS finalization or proof emission
                status = str(msg.get('status') or msg.get('szs_status') or msg.get('result') or '')
                if args.verbose:
                    log(f"[EA] received finalization tag={tag} status={status[:80]}", True)
                should_exit = False
                txt = status.strip().lower()
                if 'szs' in txt or tag in ('szs_result_out','szs_status_out'):
                    # common finals: theorem, unsatisfiable, satisfiable, timeout, counterSatisfiable
                    for kw in ('theorem','unsatisfiable','satisfiable','timeout','countersatisfiable','gaveup','unknown'):
                        if kw in txt:
                            should_exit = True
                            break
                if tag == 'proof_out':
                    should_exit = True
                if should_exit and args.auto_exit_on_szs:
                    running = False
            elif tag == 'timeout':
                if args.verbose:
                    log('[EA] received timeout tag from iProver', True)
                if args.auto_exit_on_timeout:
                    running = False
            elif tag == 'server_queries_start':
                # Depending on flag, either old ordering (end first) or new ordering (scores then end)
                # Force legacy safe ordering: server_queries_end -> scores_res (disable scores-before-end if set)
                force_scores_before_end = False
                if args.scores_before_end:
                    force_scores_before_end = False  # override
                if args.auto_server_queries_end:
                    end_obj = {'tag': 'server_queries_end'}
                    send(end_obj)
                    if args.print_io:
                        print(f"[EA OUT] {json.dumps(end_obj, ensure_ascii=False)}", flush=True)
                    if not printed_ranker_python:
                        print(f"[EA] ranker python: {sys.executable}", flush=True)
                        printed_ranker_python = True
                if args.auto_server_queries_end:
                    if not printed_ranker_python:
                        print(f"[EA] ranker python: {sys.executable}", flush=True)
                        printed_ranker_python = True
                    while pending_scores_req:
                        sreq = pending_scores_req.pop(0)
                        req_ids = sreq.get('clause_ids', [])
                        if args.verbose:
                            print(f"[EA DBG] processing scores_req clause_ids={len(req_ids)}", flush=True)
                        cands = []
                        missing = 0
                        for cid in req_ids:
                            info = clause_index.get(cid)
                            if not info:
                                missing += 1
                                continue
                            chosen_text = info['canon'] if args.use_canonical else info['text']
                            cands.append({'id': cid, 'text': chosen_text})
                        if args.verbose:
                            print(f"[EA DBG] candidates_kept={len(cands)} conjecture_len={len(conjecture_formula) if conjecture_formula else 0}", flush=True)
                            if missing:
                                print(f"[EA DBG] warning: {missing} clause_ids not registered yet; their scores will default to 0.0", flush=True)
                        if not cands:
                            send({'tag': 'scores_res', 'scores': [], 'component': sreq.get('component'), 'component_id': sreq.get('component_id')})
                            continue
                        if not conjecture_formula:
                            if args.require_conjecture:
                                if args.verbose:
                                    log('[EA] No conjecture yet; returning zeros', True)
                                send({'tag': 'scores_res', 'scores': [0.0]*len(cands), 'component': sreq.get('component'), 'component_id': sreq.get('component_id')})
                                continue
                            else:
                                conjecture_formula = cands[0]['text']
                                if args.verbose:
                                    log(f"[EA] Emergency fallback conjecture from candidate id={cands[0]['id']}", True)
                        req_dir = None
                        if args.save_requests and run_dir:
                            import pathlib, time as _t, random, hashlib
                            (run_dir/"requests").mkdir(exist_ok=True)
                            ts_ms = int(time.time()*1000)
                            req_artifact_counter += 1
                            salt = hashlib.md5(f"{ts_ms}_{req_artifact_counter}_{random.random()}".encode()).hexdigest()[:8]
                            req_dir = run_dir/"requests"/f"scores_req_{ts_ms}_{req_artifact_counter}_{salt}"
                            try:
                                req_dir.mkdir(parents=True, exist_ok=True)
                            except Exception:
                                req_dir = None
                            if req_dir is not None:
                                with open(req_dir/"scores_req.json", 'w', encoding='utf-8') as f:
                                    json.dump(sreq, f, ensure_ascii=False)
                                print(f"[EA] artifacts saved under: {req_dir}", flush=True)
                        try:
                            conj = conjecture_formula if conjecture_formula else ''
                            if args.use_canonical and conj:
                                pass
                            scores = backend.score(conj, cands)
                            # Intermediate scores
                            bi_scores = getattr(backend.backend, 'last_bi_scores', None)
                            rerank_scores_map = getattr(backend.backend, 'last_rerank_scores', None)
                            if args.verbose and bi_scores is not None:
                                try:
                                    print(f"[EA DBG] bi_scores_sample={bi_scores[:5]}", flush=True)
                                except Exception:
                                    pass
                            if args.verbose and isinstance(rerank_scores_map, dict) and rerank_scores_map:
                                try:
                                    sample_items = list(rerank_scores_map.items())[:5]
                                    print(f"[EA DBG] rerank_scores_sample={sample_items}", flush=True)
                                except Exception:
                                    pass
                        except Exception as e:
                            log(f"[ERROR] scoring failed: {e}", True)
                            scores = [0.0] * len(cands)
                        if args.score_norm != 'none' and scores:
                            import math
                            if args.score_norm == 'softmax':
                                m = max(scores)
                                ex = [math.exp((s - m)/max(1e-6, args.temp)) for s in scores]
                                z = sum(ex) or 1.0
                                scores = [e/z for e in ex]
                            elif args.score_norm == 'minmax':
                                mn = min(scores); mx = max(scores)
                                if mx - mn < 1e-9:
                                    scores = [0.5]*len(scores)
                                else:
                                    scores = [(s - mn)/(mx - mn) for s in scores]
                            elif args.score_norm == 'zscore':
                                mean = sum(scores)/len(scores)
                                var = sum((s-mean)**2 for s in scores)/max(1,len(scores)-1)
                                std = math.sqrt(var) or 1.0
                                scores = [(s-mean)/std for s in scores]
                        score_map = {c['id']: s for c, s in zip(cands, scores)}
                        ordered = [float(score_map.get(cid, 0.0)) for cid in req_ids]
                        if args.verbose and not scores:
                            print("[EA DBG] empty score list; sending empty scores_res", flush=True)
                        if args.verbose and missing:
                            print(f"[EA DBG] {missing} missing ids caused {sum(1 for cid in req_ids if cid not in score_map)} zeros in ordered output", flush=True)
                        res_obj = {'tag': 'scores_res', 'scores': ordered, 'component': sreq.get('component'), 'component_id': sreq.get('component_id')}
                        send(res_obj)
                        if args.print_io:
                            print(f"[EA OUT] {json.dumps(res_obj, ensure_ascii=False)}", flush=True)
                        if args.save_requests and run_dir:
                            if req_dir is not None:
                                with open(req_dir/"scores_res.json", 'w', encoding='utf-8') as f:
                                    json.dump(res_obj, f, ensure_ascii=False)
                                try:
                                    debug_meta = {
                                        'conjecture': conjecture_formula,
                                        'first_candidates': cands[:10],
                                        'bi_scores_first': getattr(backend.backend, 'last_bi_scores', [])[:10],
                                        'rerank_scores': list(getattr(backend.backend, 'last_rerank_scores', {}).items())[:20]
                                    }
                                    with open(req_dir/"debug_meta.json", 'w', encoding='utf-8') as df:
                                        json.dump(debug_meta, df, ensure_ascii=False)
                                except Exception:
                                    pass
                            else:
                                import pathlib
                                (run_dir/"responses").mkdir(exist_ok=True)
                                with open(run_dir/"responses"/f"scores_res_{int(time.time()*1000)}.json", 'w', encoding='utf-8') as f:
                                    json.dump(res_obj, f, ensure_ascii=False)
                        stats = getattr(backend.backend, 'last_timing', None)
                        if stats and args.verbose:
                            log(f"TIMING cands={stats.get('cands')} topk={stats.get('topk')} bi={stats.get('bi_time'):.4f}s rerank={stats.get('rerank_time'):.4f}s total={stats.get('total_time'):.4f}s bi_hits={stats.get('bi_cache_hits')} rerank_hits={stats.get('rerank_cache_hits')}", True)
                        if args.exit_after_scores:
                            running = False
                    # already sent end before scores for compatibility
            elif tag == 'terminate':
                if not args.no_ack:
                    res = {'tag': 'ack', 'what': 'terminate'}
                    send(res)
                    if args.print_io:
                        print(f"[EA OUT] {json.dumps(res, ensure_ascii=False)}", flush=True)
                running = False
            else:
                if not args.no_ack:
                    res = {'tag': 'ack', 'what': tag or 'unknown'}
                    send(res)
                    if args.print_io:
                        print(f"[EA OUT] {json.dumps(res, ensure_ascii=False)}", flush=True)
    try:
        conn.close()
    except Exception:
        pass
    try:
        srv.close()
    except Exception:
        pass
    log('[EA] server stopped', args.verbose)


def main(argv=None):  # pragma: no cover
    args = parse_args(argv)
    if args.command == 'serve':
        return serve(args)
    return 0


if __name__ == '__main__':  # pragma: no cover
    raise SystemExit(main())
