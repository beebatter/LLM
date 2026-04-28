#!/usr/bin/env python3
"""Minimal example client that mimics iProver interaction with interactive_server.

Usage:
  1) Start server, e.g.:
     python -m LLM.ea.interactive_server serve \
        --host 127.0.0.1 --port 7001 \
        --mode bi_tf --ckpt /root/autodl-tmp/Training/models/biencoder_distill_kl.pt \
        --vocab /root/autodl-tmp/Training/models/spm_logic_24k.vocab \
        --score-norm softmax --temp 0.3 --verbose
  2) Run this client:
     python -m LLM.ea.iprover_client_example --host 127.0.0.1 --port 7001

Simulates:
  - register_clauses: sends conjecture + 3 candidate clauses
  - scores_req: asks for their scores
  - terminate
"""
from __future__ import annotations
import argparse, json, socket, time


def send(sock: socket.socket, obj):
    data = (json.dumps(obj, ensure_ascii=False) + '\x00').encode('utf-8')
    sock.sendall(data)


def recv_loop(sock: socket.socket, timeout=2.0):
    sock.settimeout(0.2)
    buf = b''
    out = []
    t0 = time.time()
    while time.time() - t0 < timeout:
        try:
            chunk = sock.recv(4096)
            if not chunk:
                time.sleep(0.05)
                continue
            buf += chunk
        except socket.timeout:
            pass
        while True:
            if b'\x00' in buf:
                part, buf = buf.split(b'\x00', 1)
            elif b'\n' in buf:
                part, buf = buf.split(b'\n', 1)
            else:
                break
            if not part.strip():
                continue
            try:
                out.append(json.loads(part.decode('utf-8', errors='ignore')))
            except Exception:
                continue
    return out


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument('--host', default='127.0.0.1')
    ap.add_argument('--port', type=int, default=7001)
    args = ap.parse_args(argv)

    s = socket.create_connection((args.host, args.port))

    # register clauses: include one with conj_dist=0 to set conjecture
    clauses = [
        {"clause_id": 1, "clause": "goal: prove eq(s(a), s(a))", "clause_features": {"conj_dist": 0}},
        {"clause_id": 2, "clause": "clause eq(s(a), s(a)).", "clause_features": {"conj_dist": 1}},
        {"clause_id": 3, "clause": "clause eq(s(a), s(b)).", "clause_features": {"conj_dist": 1}},
        {"clause_id": 4, "clause": "clause implies(P(a),Q(a)).", "clause_features": {"conj_dist": 2}},
    ]
    send(s, {"tag": "register_clauses", "clauses": clauses})
    print('Sent register_clauses')
    time.sleep(0.1)
    for r in recv_loop(s):
        print('Recv:', r)

    # request scores for subset
    send(s, {"tag": "scores_req", "clause_ids": [2,3,4], "component": "demo", "component_id": 0})
    print('Sent scores_req')
    for r in recv_loop(s):
        print('Recv:', r)

    send(s, {"tag": "terminate"})
    for r in recv_loop(s):
        print('Recv:', r)
    s.close()

if __name__ == '__main__':
    main()
