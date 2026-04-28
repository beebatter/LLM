#!/usr/bin/env python3
"""Unified interactive EA server with canonical preprocessing + bi / cross / LLM fusion.

Protocol order (expected by iProver interactive README):
  scores_req -> server_queries_start -> server_queries_end -> scores_res

Features:
  - Clause registration with optional canonical extraction (reuse logic from preprocess_and_rank).
  - Multi-source scoring pipeline when scores_req is answered:
        * Bi-encoder (mandatory if provided) over ALL candidates.
        * Cross-encoder rerank on topK (optional).
        * LLM direct / PMI scoring on same topK (optional, slower).
        * Heuristic semantic scoring (lightweight) from canonical forms.
        * Linear fusion with user weights; per-source min-max normalization to 0..1.
  - Semantic tags: unit / horn / eq / resolvable / touches_target_functor / eq_of_target_functor /
                    first_arg_in_goal / shares_goal_consts:k / goal_pred_overlap=n
  - Detailed runtime logging (per scores_req) including timing + cache hits if backends expose them.
  - Softmax / minmax / zscore / none final post-normalization (after fusion) if requested.
  - NUL ("\x00") terminated JSON messages (no trailing newline) for iProver parser compatibility.

CLI example:
  python -m LLM.ea.interactive_server_unified \\
    --host 127.0.0.1 --port 7300 \\
    --bi-ckpt /path/bi.pt --cross-ckpt /path/cross.pt \\
    --llm-model /path/llm --topk 256 \\
    --fuse-lambda-bi 1.0 --fuse-lambda-cross 1.0 --fuse-lambda-llm 1.0 --fuse-lambda-heur 0.3 \\
    --use-canonical --score-norm softmax --temp 0.3 --verbose

Notes:
  - If only --bi-ckpt is given, fusion degenerates to bi (plus optional heuristic if weight>0).
  - Heuristic component always computed from canonical text (enabled automatically if use-canonical).
  - LLM scoring is expensive; keep topk moderate. It uses avg logprob (see backends _LLMWrapper).
  - This server is stateless across scores_req except clause storage and optional backend caches.
"""
from __future__ import annotations
import argparse, json, math, os, socket, sys, time, traceback
from typing import Any, Dict, List, Tuple, Optional

try:  # Import canonical helpers & semantic scoring pieces
    from LLM.preprocess_and_rank import (
        preprocess_clause_str, extract_formula_from_tcf,
        canonicalise_formula, CanonicalMaps, parse_literals,
        extract_goal_frontier, infer_target_info, semantic_tags_for_clause,
        score_clause_heuristic
    )  # type: ignore
except Exception:  # fallback minimal implementations
    def preprocess_clause_str(s: str) -> str: return s
    def extract_formula_from_tcf(s: str) -> str: return s
    class CanonicalMaps:  # type: ignore
        #!/usr/bin/env python3
        """REMOVED MODULE STUB

        This file previously contained the 'unified' interactive server with fusion + canonical pipeline.
        Per user request it has been replaced by a stub to avoid accidental usage.

        Use instead:
            python -m LLM.ea.interactive_server serve ...

        If you need the old implementation, recover it from version control history before this cleanup.
        """
        import sys

        def main():  # pragma: no cover
                sys.stderr.write("[stub] interactive_server_unified has been removed. Use interactive_server.\n")
                sys.exit(1)

        if __name__ == '__main__':  # pragma: no cover
                main()
        mn, mx = min(scores), max(scores)
        return [0.5]*len(scores) if mx-mn < 1e-12 else [(s-mn)/(mx-mn) for s in scores]
    if kind == 'zscore':
        mean = sum(scores)/len(scores)
        var = sum((s-mean)**2 for s in scores)/max(1,len(scores)-1)
        std = math.sqrt(var) or 1.0
        return [(s-mean)/std for s in scores]
    return scores


# ----------- Server State ------------
class ClauseStore:
    def __init__(self, use_canonical: bool):
        self.use_canonical = use_canonical
        self.clauses: Dict[int, Dict[str, Any]] = {}
        self.conjecture_text: str = ''
        self.cmap = CanonicalMaps() if use_canonical else None

    def add(self, clause_id: int, raw_clause: str, features: Dict[str, Any]):
        if self.use_canonical:
            cleaned = preprocess_clause_str(raw_clause)
            formula = extract_formula_from_tcf(cleaned)
            canon, var_map, local_syms = canonicalise_formula(formula, self.cmap)  # type: ignore[arg-type]
            rec = {
                'id': clause_id,
                'raw': raw_clause,
                'formula': formula,
                'text': canon,
                'features': features,
                'variable_mapping': var_map,
                'local_symbols': {k: sorted(list(v)) for k,v in local_syms.items()},
            }
        else:
            rec = {'id': clause_id, 'raw': raw_clause, 'text': raw_clause, 'features': features}
        self.clauses[clause_id] = rec
        if features.get('conj_dist') == 0 and not self.conjecture_text:
            self.conjecture_text = rec['text']

    def get_batch(self, ids: List[int]) -> List[Dict[str, Any]]:
        out = []
        for cid in ids:
            if cid in self.clauses:
                out.append(self.clauses[cid])
        return out


class ScoringPipeline:
    def __init__(self, args):
        self.args = args
        self.bi = BiTFBackend(args.bi_ckpt, device=args.device, vocab_path=args.vocab)
        self.cross = CrossTFBackend(args.cross_ckpt, device=args.device, vocab_path=args.vocab) if args.cross_ckpt else None
        if args.llm_model:
            if args.llm_pmi:
                self.llm = LLMPMIBackend(args.llm_model, lambda_pmi=args.lambda_pmi)
            else:
                self.llm = LLMDIRECTBackend(args.llm_model)
        else:
            self.llm = None
        self.last_detail: Dict[str, Any] = {}

    def score(self, conjecture: str, candidates: List[Dict[str, Any]]) -> Dict[int, float]:
        t0 = time.time()
        # Bi stage
        bi_scores_list = self.bi.score(conjecture, [{'id': c['id'], 'text': c['text']} for c in candidates])
        bi_map = {c['id']: float(s) for c, s in zip(candidates, bi_scores_list)}
        t_bi = time.time()
        # Determine rerank subset
        order = sorted(candidates, key=lambda c: bi_map[c['id']], reverse=True)
        rerank_subset = order[: min(self.args.topk, len(order))]
        # Cross stage
        cross_map: Dict[int, float] = {}
        if self.cross and rerank_subset:
            cross_scores = self.cross.score(conjecture, [{'id': c['id'], 'text': c['text']} for c in rerank_subset])
            cross_map = {c['id']: float(s) for c, s in zip(rerank_subset, cross_scores)}
        t_cross = time.time()
        # LLM stage
        llm_map: Dict[int, float] = {}
        if self.llm and rerank_subset:
            llm_scores = self.llm.score(conjecture, [{'id': c['id'], 'text': c['text']} for c in rerank_subset])
            llm_map = {c['id']: float(s) for c, s in zip(rerank_subset, llm_scores)}
        t_llm = time.time()
        # Heuristic stage (if weight > 0 or verbose, compute for all candidates)
        heur_map: Dict[int, float] = {}
        if self.args.fuse_lambda_heur > 0 or self.args.verbose:
            goal = extract_goal_frontier(conjecture)
            targets = infer_target_info(conjecture)
            for c in candidates:
                tags = semantic_tags_for_clause(c, goal, targets)
                c['tags'] = tags  # record for possible logging
                raw, _why = score_clause_heuristic(tags, c['text'])
                heur_map[c['id']] = raw  # raw 0..50
        # Normalize each component for fusion
        comp_norms: Dict[str, Dict[int,float]] = {}
        comp_norms['bi'] = _minmax_norm(bi_map)
        if cross_map: comp_norms['cross'] = _minmax_norm(cross_map)
        if llm_map: comp_norms['llm'] = _minmax_norm(llm_map)
        if heur_map: comp_norms['heur'] = _minmax_norm(heur_map)
        # Fusion
        w_bi = max(0.0, self.args.fuse_lambda_bi)
        w_cross = max(0.0, self.args.fuse_lambda_cross) if cross_map else 0.0
        w_llm = max(0.0, self.args.fuse_lambda_llm) if llm_map else 0.0
        w_heur = max(0.0, self.args.fuse_lambda_heur) if heur_map else 0.0
        denom = w_bi + w_cross + w_llm + w_heur
        if denom <= 0:
            denom = 1.0; w_bi = 1.0  # fallback to bi only
        fused: Dict[int, float] = {}
        ids_all = {c['id'] for c in candidates}
        for cid in ids_all:
            sc = 0.0
            if cid in comp_norms['bi']: sc += w_bi * comp_norms['bi'][cid]
            if 'cross' in comp_norms and cid in comp_norms['cross']: sc += w_cross * comp_norms['cross'][cid]
            if 'llm' in comp_norms and cid in comp_norms['llm']: sc += w_llm * comp_norms['llm'][cid]
            if 'heur' in comp_norms and cid in comp_norms['heur']: sc += w_heur * comp_norms['heur'][cid]
            fused[cid] = sc / denom
        t_end = time.time()
        top_preview = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:10]
        self.last_detail = {
            'timing': {
                'bi': t_bi - t0,
                'cross': t_cross - t_bi,
                'llm': t_llm - t_cross,
                'fuse': t_end - t_llm,
                'total': t_end - t0,
            },
            'sizes': {
                'candidates': len(candidates),
                'rerank_subset': len(rerank_subset),
                'cross_scored': len(cross_map),
                'llm_scored': len(llm_map),
            },
            'weights': {'bi': w_bi, 'cross': w_cross, 'llm': w_llm, 'heur': w_heur},
            'components_raw': {
                'bi': bi_map,
                **({'cross': cross_map} if cross_map else {}),
                **({'llm': llm_map} if llm_map else {}),
                **({'heur': heur_map} if heur_map else {}),
            },
            'components_norm': comp_norms,
            'fused_top10': top_preview,
        }
        return fused


# ------------- Main Loop -------------
def serve(args):
    os.makedirs(args.log_dir, exist_ok=True)
    log_path = os.path.join(args.log_dir, f"EAUnified.{args.port}.{int(time.time())}.log")
    def log(msg: str):
        ts = time.strftime('%H:%M:%S')
        with open(log_path, 'a', encoding='utf-8') as f:
            f.write(f"[{ts}] {msg}\n")
        if args.verbose:
            print(f"[{ts}] {msg}")

    log("=== Unified EA server start ===")
    log(f"Args: {vars(args)}")

    store = ClauseStore(args.use_canonical)
    scorer = ScoringPipeline(args)
    stats = {'register_clauses': 0, 'scores_req': 0, 'scores_res': 0}
    request_counter = 0
    last_scores_req_time = 0.0

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((args.host, args.port))
    srv.listen(1)
    print(f"[EA] listening {args.host}:{args.port} (unified, NUL delimited)")
    conn, addr = srv.accept()
    print(f"[EA] connected {addr}")
    conn.settimeout(0.5)
    buf = b''
    pending: Dict[Tuple[Optional[str], Optional[int]], Dict[str, Any]] = {}

    # Choose delimiter
    if args.delimiter == 'nul':
        OUT_DELIM = "\x00"
    elif args.delimiter == 'nul_full':
        OUT_DELIM = "\n\x00\n"
    else:
        OUT_DELIM = "\n"

    def send(obj: Dict[str, Any]):
        frame = json.dumps(obj, ensure_ascii=False)
        if args.print_io:
            print('[EA OUT]', frame)
        try:
            # Chunk large frames to avoid OS TCP segmentation edge cases with very big JSON (safety)
            data = (frame + OUT_DELIM).encode('utf-8')
            if len(data) > 256 * 1024:  # 256KB threshold arbitrary
                for i in range(0, len(data), 64 * 1024):
                    conn.sendall(data[i:i+64*1024])
            else:
                conn.sendall(data)
        except Exception:
            pass
        log(f"OUT {obj.get('tag')} size={len(frame)} delim={args.delimiter}")

    last_activity = time.time()
    running = True
    while running:
        try:
            chunk = conn.recv(65536)
            if not chunk:
                time.sleep(0.01)
            else:
                buf += chunk
                last_activity = time.time()
        except socket.timeout:
            pass
        except Exception:
            break
        # Heartbeat
        if args.heartbeat_sec > 0 and (time.time() - last_activity) >= args.heartbeat_sec:
            log(f"HEARTBEAT pending={len(pending)} stored_clauses={len(store.clauses)} conj_set={bool(store.conjecture_text)}")
            last_activity = time.time()
        progressed = True
        while progressed:
            progressed = False
            if b'\x00' in buf:
                raw, buf = buf.split(b'\x00', 1); progressed = True
            elif b'\n' in buf:  # fallback newline delim
                raw, buf = buf.split(b'\n', 1); progressed = True
            if not progressed:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                msg = json.loads(raw.decode('utf-8', 'ignore'))
            except Exception:
                log('WARN bad JSON frame')
                continue
            tag = msg.get('tag')
            if args.print_io:
                print('[EA IN]', tag)
            log(f"IN {tag}")
            if tag == 'terminate':
                if args.emit_ack:
                    send({'tag': 'ack', 'what': 'terminate'})
                else:
                    log('TERMINATE received (no ack sent)')
                running = False
                break
            if tag == 'register_clauses':
                added = 0
                for c in msg.get('clauses', []):
                    cid = c.get('clause_id')
                    if cid is None:
                        continue
                    if cid in store.clauses:
                        continue
                    store.add(int(cid), c.get('clause') or '', c.get('clause_features', {}) or {})
                    added += 1
                stats['register_clauses'] += 1
                if args.emit_ack:
                    send({'tag': 'ack', 'what': 'register_clauses', 'count': added})
                else:
                    log(f"REGISTER_CLAUSES stored={added} (no ack sent)")
            elif tag == 'scores_req':
                comp = msg.get('component'); cid = msg.get('component_id')
                pending[(comp, cid)] = msg
                stats['scores_req'] += 1
                last_scores_req_time = time.time()
            elif tag == 'server_queries_start':
                # 根据参数决定顺序
                if not args.scores_before_end:
                    send({'tag': 'server_queries_end'})
                # Answer all pending scores_req (可能多个)
                for key, sreq in list(pending.items()):
                    request_counter += 1
                    req_ids = sreq.get('clause_ids', [])[: args.max_candidates]
                    batch = store.get_batch(req_ids)
                    if not store.conjecture_text and batch:
                        store.conjecture_text = batch[0]['text']
                    conj = store.conjecture_text or ''
                    if not batch:
                        send({'tag': 'scores_res', 'scores': [], 'component': sreq.get('component'), 'component_id': sreq.get('component_id')})
                        continue
                    try:
                        fused = scorer.score(conj, batch)
                    except Exception as e:
                        log(f"ERROR scoring: {e}\n{traceback.format_exc()}")
                        fused = {b['id']: 0.0 for b in batch}
                    # Build ordered list matching req clause_ids
                    ordered = [float(fused.get(cid, 0.0)) for cid in req_ids]
                    # Post global normalization if requested
                    ordered = _post_normalize(ordered, args.score_norm, args.temp)
                    send({'tag': 'scores_res', 'scores': ordered, 'component': sreq.get('component'), 'component_id': sreq.get('component_id')})
                    stats['scores_res'] += 1
                    # Dump artifacts
                    if args.log_requests:
                        req_root = os.path.join(args.log_dir, 'requests')
                        os.makedirs(req_root, exist_ok=True)
                        rdir = os.path.join(req_root, f"req_{request_counter:05d}")
                        os.makedirs(rdir, exist_ok=True)
                        try:
                            with open(os.path.join(rdir, 'scores_req.json'), 'w', encoding='utf-8') as f:
                                json.dump(sreq, f, ensure_ascii=False, indent=2)
                            with open(os.path.join(rdir, 'candidates.json'), 'w', encoding='utf-8') as f:
                                json.dump(batch, f, ensure_ascii=False, indent=2)
                            with open(os.path.join(rdir, 'fused_ordered.json'), 'w', encoding='utf-8') as f:
                                json.dump({'ordered_clause_ids': req_ids, 'scores': ordered}, f, ensure_ascii=False, indent=2)
                            if scorer.last_detail:
                                with open(os.path.join(rdir, 'timing.json'), 'w', encoding='utf-8') as f:
                                    json.dump(scorer.last_detail.get('timing', {}), f, ensure_ascii=False, indent=2)
                                with open(os.path.join(rdir, 'sizes.json'), 'w', encoding='utf-8') as f:
                                    json.dump(scorer.last_detail.get('sizes', {}), f, ensure_ascii=False, indent=2)
                                with open(os.path.join(rdir, 'weights.json'), 'w', encoding='utf-8') as f:
                                    json.dump(scorer.last_detail.get('weights', {}), f, ensure_ascii=False, indent=2)
                                if args.dump_components:
                                    with open(os.path.join(rdir, 'components_raw.json'), 'w', encoding='utf-8') as f:
                                        json.dump(scorer.last_detail.get('components_raw', {}), f, ensure_ascii=False)
                                    with open(os.path.join(rdir, 'components_norm.json'), 'w', encoding='utf-8') as f:
                                        json.dump(scorer.last_detail.get('components_norm', {}), f, ensure_ascii=False)
                        except Exception as _e:
                            log(f"WARN dump failed: {_e}")
                    # Scoreboard summary
                    try:
                        scoreboard = {
                            'requests': request_counter,
                            'stats': stats,
                            'last_detail': scorer.last_detail,
                            'ts': int(time.time()),
                        }
                        with open(os.path.join(args.log_dir, 'scoreboard.json'), 'w', encoding='utf-8') as f:
                            json.dump(scoreboard, f, ensure_ascii=False, indent=2)
                    except Exception:
                        pass
                    if scorer.last_detail and 'fused_top10' in scorer.last_detail:
                        log(f"FUSED_TOP10 {scorer.last_detail['fused_top10']}")
                    # Log details
                    if scorer.last_detail:
                        log(f"DETAIL timing={scorer.last_detail['timing']} sizes={scorer.last_detail['sizes']} weights={scorer.last_detail['weights']}")
                pending.clear()
                # 如果选择 scores_before_end, 现在再发 server_queries_end
                if args.scores_before_end:
                    send({'tag': 'server_queries_end'})
            else:
                # ignore other tags for now (szs_result_out, proof_out...)
                pass
        # Idle detection (after processing loop)
        if args.max_idle_sec > 0 and stats['scores_req'] > 0 and (time.time() - last_scores_req_time) > args.max_idle_sec:
            log(f"IDLE no_new_scores_req_for={int(time.time()-last_scores_req_time)}s total_rounds={request_counter} (可能 iProver 未再调度 external_score)")
            last_scores_req_time = time.time()
    try:
        conn.close()
    except Exception:
        pass
    try:
        srv.close()
    except Exception:
        pass
    log('=== Unified EA server stop ===')
    print('[EA] stopped')


def main():  # pragma: no cover
    serve(parse_args())

if __name__ == '__main__':  # pragma: no cover
    main()
