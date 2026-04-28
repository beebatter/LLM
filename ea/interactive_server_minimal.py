#!/usr/bin/env python3
"""Minimal interactive external agent server for iProver (PSM mode).

Features:
- Follows README-interactive ordering: scores_req -> server_queries_start -> server_queries_end -> scores_res.
- Buffers the last scores_req per component until server_queries_start.
- Uses a bi-then-cross (or single clause_tf) backend via existing backends loader.
- Supports softmax/minmax/zscore/none normalization.
- Optional canonical text extraction via process_iprover_v3 helpers.
- NUL ("\x00") terminated JSON protocol (no trailing newline) to match iProver parser.
- Simple logging + optional artifact saving.

Differences from deleted advanced server:
- No multi-request artifact directories.
- No debug meta dumps.
- Only retains most recent scores_req (drops older if multiple arrive before start).

Extendable: add SAT query handling or LLM rerank later.
"""
from __future__ import annotations
import argparse, json, socket, sys, time, os, math
from typing import Dict, Any, List

try:
    from LLM.process_iprover_v3 import preprocess_clause_str, extract_formula_from_tcf  # type: ignore
except Exception:  # noqa: E722
    def preprocess_clause_str(s: str) -> str: return s
    def extract_formula_from_tcf(s: str) -> str: return s

from .backends import load_backend


def parse_args(argv=None):
    p = argparse.ArgumentParser(description="Minimal iProver EA server")
    p.add_argument('--host', default='127.0.0.1')
    p.add_argument('--port', type=int, required=True)
    p.add_argument('--mode', required=True, choices=['clause_tf','cross_tf','bi_tf','bi_then_cross','bi_then_llm','llm_direct','llm_pmi','fusion'])
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
    p.add_argument('--use-canonical', action='store_true')
    p.add_argument('--log', action='store_true')
    p.add_argument('--device', default='cuda')
    p.add_argument('--fusion-files', nargs='+')
    p.add_argument('--fusion-weights', nargs='+', type=float)
    return p.parse_args(argv)

class Backend:
    def __init__(self, a):
        if a.mode == 'fusion':
            assert a.fusion_files and a.fusion_weights
        if a.mode == 'bi_then_llm':
            assert a.bi_ckpt and a.llm_model
            self.backend = load_backend('bi_then_llm', bi_ckpt=a.bi_ckpt, llm=a.llm_model, topk=a.topk, pmi=a.pmi, lambda_pmi=a.lambda_pmi, device=a.device)
        elif a.mode == 'bi_then_cross':
            assert a.bi_ckpt and a.cross_ckpt
            self.backend = load_backend('bi_then_cross', bi_ckpt=a.bi_ckpt, cross_ckpt=a.cross_ckpt, topk=a.topk, device=a.device, vocab=a.vocab)
        elif a.mode in ('clause_tf','cross_tf','bi_tf'):
            assert a.ckpt
            self.backend = load_backend(a.mode, ckpt=a.ckpt, device=a.device, vocab=a.vocab)
        elif a.mode in ('llm_direct','llm_pmi'):
            assert a.llm_model
            kw={'llm':a.llm_model}
            if a.mode=='llm_pmi': kw['lambda_pmi']=a.lambda_pmi
            self.backend = load_backend(a.mode, **kw)
        elif a.mode=='fusion':
            self.backend = load_backend('fusion', sources=a.fusion_files, weights=a.fusion_weights)
        else:
            raise ValueError(a.mode)
    def score(self, conj: str, cands: List[Dict[str,Any]]):
        return self.backend.score(conj, cands)

def normalize(scores: List[float], args) -> List[float]:
    if not scores: return scores
    if args.score_norm=='none': return scores
    if args.score_norm=='softmax':
        m=max(scores); ex=[math.exp((s-m)/max(1e-6,args.temp)) for s in scores]; z=sum(ex) or 1.0; return [e/z for e in ex]
    if args.score_norm=='minmax':
        mn=min(scores); mx=max(scores); return [0.5]*len(scores) if mx-mn<1e-9 else [(s-mn)/(mx-mn) for s in scores]
    if args.score_norm=='zscore':
        mean=sum(scores)/len(scores); var=sum((s-mean)**2 for s in scores)/max(1,len(scores)-1); std=math.sqrt(var) or 1.0; return [(s-mean)/std for s in scores]
    return scores

def serve(args):
    backend=Backend(args)
    srv=socket.socket(socket.AF_INET,socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1)
    srv.bind((args.host,args.port))
    srv.listen(1)
    print(f"[EA] listening {args.host}:{args.port} (NUL delimited)")
    conn,addr=srv.accept()
    print(f"[EA] connected {addr}")
    conn.settimeout(0.5)
    clause_index: Dict[int,Dict[str,Any]]={}
    conj=""
    pending: Dict[Tuple[str,int],Dict[str,Any]]={}  # (component,component_id)->scores_req msg
    buf=b''
    def send(obj:Dict[str,Any]):
        try: conn.sendall((json.dumps(obj,ensure_ascii=False)+"\x00").encode('utf-8'))
        except Exception: pass
    running=True
    while running:
        try:
            chunk=conn.recv(65536)
            if not chunk:
                time.sleep(0.01)
            else:
                buf+=chunk
        except socket.timeout: pass
        except Exception: break
        # parse
        progressed=True
        while progressed:
            progressed=False
            if b'\x00' in buf:
                raw,buf=buf.split(b'\x00',1); progressed=True
            elif b'\n' in buf:
                raw,buf=buf.split(b'\n',1); progressed=True
            if not progressed: break
            raw=raw.strip()
            if not raw: continue
            try: msg=json.loads(raw.decode('utf-8','ignore'))
            except Exception:
                if args.log: print('[EA] bad json',raw[:80])
                continue
            tag=msg.get('tag')
            if args.log: print('[EA IN]',tag)
            if tag=='terminate':
                send({'tag':'ack','what':'terminate'}); running=False; break
            if tag=='register_clauses':
                for c in msg.get('clauses',[]):
                    cid=c.get('clause_id');
                    if cid is None: continue
                    raw_clause=c.get('clause') or ''
                    if args.use_canonical:
                        cleaned=preprocess_clause_str(raw_clause)
                        formula=extract_formula_from_tcf(cleaned)
                        canon=formula or raw_clause
                    else:
                        canon=raw_clause
                    clause_index[cid]={'text':raw_clause,'canon':canon,'features':c.get('clause_features',{})}
                    feat=clause_index[cid]['features']
                    if feat and feat.get('conj_dist')==0 and not conj:
                        conj=clause_index[cid]['canon' if args.use_canonical else 'text']
                send({'tag':'ack','what':'register_clauses','count':len(msg.get('clauses',[]))})
            elif tag=='scores_req':
                comp=msg.get('component'); cid=msg.get('component_id'); pending[(comp,cid)]=msg
            elif tag=='server_queries_start':
                # Immediately finish query phase then reply scores_res for each pending request (README order)
                send({'tag':'server_queries_end'})
                for key,sreq in list(pending.items()):
                    req_ids=sreq.get('clause_ids',[])
                    cands=[]
                    for rcid in req_ids:
                        info=clause_index.get(rcid)
                        if not info: continue
                        chosen=info['canon'] if args.use_canonical else info['text']
                        cands.append({'id':rcid,'text':chosen})
                    if not conj and cands:
                        conj=cands[0]['text']
                    if not cands:
                        send({'tag':'scores_res','scores':[],'component':sreq.get('component'),'component_id':sreq.get('component_id')})
                        continue
                    try:
                        raw_scores=backend.score(conj,cands)
                    except Exception:
                        raw_scores=[0.0]*len(cands)
                    raw_scores=normalize(raw_scores,args)
                    m={c['id']:s for c,s in zip(cands,raw_scores)}
                    ordered=[float(m.get(i,0.0)) for i in req_ids]
                    send({'tag':'scores_res','scores':ordered,'component':sreq.get('component'),'component_id':sreq.get('component_id')})
                pending.clear()
            else:
                # ignore others
                pass
    try: conn.close()
    except Exception: pass
    try: srv.close()
    except Exception: pass
    print('[EA] stopped')

if __name__=='__main__':
    serve(parse_args())
