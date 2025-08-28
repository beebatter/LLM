稳健版（Balanced）——按比例丢弃 70%，最少保留 90，Top-K=192
python3 /home/ks/LLM/process_iprover_v3.py serve \
  --host 127.0.0.1 --port 12346 \
  --ranker-script /home/ks/LLM/batch_ranker.py \
  --model gpt-4.1 \
  --chunk-size 92 --anchors 6 \
  --context-size 128 --context-summary-k 129 --summary-max-tokens 700 \
  --use-server-queries --include-sat-eval \
  --verbose --progress \
  --python-exec /home/ks/Proof-Guidance-for-Automated-Theorem-Proving-Using-Large-Language-Models/LLM/LLM_py311/bin/python \
  --prefilter-drop 0.7 --prefilter-min-keep 90 --prefilter-top-k 192

稳健版（Balanced）——提高外部打分权重、收紧搜索面、略降重启倍率、加大实例化占比
./iproveropt \
  --interactive_mode true \
  --external_ip_address 127.0.0.1 \
  --external_port 12346 \
  --schedule none \
  --time_out_real 2000 \
  --preprocessing_flag true \
  --instantiation_flag true \
  --superposition_flag true \
  --resolution_flag false \
  --sup_iter_deepening 1 \
  --comb_mode clause_based \
  --comb_inst_mult 8 \
  --comb_sup_mult 6 \
  --sup_passive_queue_type priority_queues \
  --sup_passive_queues '[[+external_score;-num_symb];[-conj_dist;+conj_symb;-num_symb];[+age;-num_symb]]' \
  --sup_passive_queues_freq '[8;2;1]' \
  --inst_passive_queue_type priority_queues \
  --inst_passive_queues '[[+external_score;-num_var];[-conj_dist;+conj_symb;-num_var];[+age;-num_var]]' \
  --inst_passive_queues_freq '[8;2;1]' \
  --sup_unprocessed_bound 900 \
  --inst_unprocessed_bound 800 \
  --sup_restarts_mult 8 \
  --sup_to_prop_solver passive \
  --sup_prop_simpl_new true \
  --sup_prop_simpl_given true \
  --sup_smt_interval 3000 \
  /home/ks/TPTP-v9.0.0/Problems/NUN/NUN/NUN066+1.p





./iproveropt \
  --interactive_mode true \
  --external_ip_address 127.0.0.1 \
  --external_port 12346 \
  --schedule none \
  --time_out_real 2000 \
  --preprocessing_flag true \
  --instantiation_flag true \
  --superposition_flag true \
  --resolution_flag false \
  --sup_iter_deepening 1 \
  --comb_mode clause_based \
  --comb_inst_mult 8 \
  --comb_sup_mult 6 \
  --sup_passive_queue_type priority_queues \
  --sup_passive_queues '[[+external_score;-num_symb];[-conj_dist;+conj_symb;-num_symb];[+age;-num_symb]]' \
  --sup_passive_queues_freq '[6;3;1]' \
  --inst_passive_queue_type priority_queues \
  --inst_passive_queues '[[+external_score;-num_var];[-conj_dist;+conj_symb;-num_var];[+age;-num_var]]' \
  --inst_passive_queues_freq '[6;3;1]' \
  --sup_unprocessed_bound 900 \
  --inst_unprocessed_bound 800 \
  --sup_restarts_mult 8 \
  --sup_to_prop_solver passive \
  --sup_prop_simpl_new true \
  --sup_prop_simpl_given true \
  --sup_smt_interval 3000 \
  --sup_share_score_frac 0.1 \
  --sup_share_max_num_cl 120 \
  /home/ks/TPTP-v9.0.0/Problems/PRO/PRO002+2.p


  ./iproveropt \

  --schedule default \
  --time_out_real 2000 \
  --preprocessing_flag true \
  --instantiation_flag true \
  --superposition_flag true \
  --resolution_flag false \
  --sup_iter_deepening 1 \
  --comb_mode clause_based \
  --comb_inst_mult 8 \
  --comb_sup_mult 6 \

  --sup_restarts_mult 8 \
  --sup_to_prop_solver passive \
  --sup_prop_simpl_new true \
  --sup_prop_simpl_given true \
  --sup_smt_interval 3000 \

  /home/ks/TPTP-v9.0.0/Problems/PRO/PRO002+2.p



  ./iproveropt     --schedule none   --time_out_real 2000   --preprocessing_flag true   --instantiation_flag true   --superposition_flag true   --resolution_flag true   --sup_iter_deepening 1   --comb_mode clause_based   --comb_inst_mult 8   --comb_sup_mult 6   --sup_passive_queue_type priority_queues      --inst_passive_queue_type priority_queues   --sup_unprocessed_bound 900   --inst_unprocessed_bound 800   --sup_restarts_mult 8   --sup_to_prop_solver passive   --sup_prop_simpl_new true   --sup_prop_simpl_given true   --sup_smt_interval 3000   /home/ks/TPTP-v9.0.0/Problems/AGT/AGT011+2.p




  python3 run_batch_pipeline.py \
  --problems fof_problems_from_html.json \
  --iprover iproveropt \
  --output dataset.jsonl \
  --fail-log failed_problems.jsonl \
  --timeout 300



  s": {"basic_clause_id": 8826, "conj_dist": -1, "born": 2, "horn": true, "epr": true}}]}
[EA IN] {"tag": "scores_req", "clause_ids": [9928], "component": "sup", "component_id": 1}
[EA IN] {"tag": "server_queries_start"}
[EA OUT] {"tag": "server_queries_end"}
[EA] ranker python: /home/ks/Proof-Guidance-for-Automated-Theorem-Proving-Using-Large-Language-Models/LLM/LLM_py311/bin/python3
[DEBUG] clauses: context=128, candidates=1
[LLM] Using cached background summary.
[DEBUG] anchors=0 pool=1 chunks=1
[DEBUG] chunk[0] size=1
[LLM] Scoring chunk 1/1 (size=1) with model=gpt-4o-mini...
[DEBUG] chunk_000: scores=1
[DEBUG] total_scored entries across chunks: 1
[CAL] global min=0.948 max=0.948 mean=0.948
Saved 1 scores to /home/ks/LLM/Logs/EA.57437.1756133650/requests/scores_req_1756133705427_1_0ffceb2f/out_scores.json. Top 1:
   1. id=9928  score=0.948
[EA] artifacts saved under: /home/ks/LLM/Logs/EA.57437.1756133650/requests/scores_req_1756133705427_1_0ffceb2f
[EA OUT] {"tag": "scores_res", "scores": [0.9484638255698712], "component": "sup", "component_id": 1}
Exception in thread Thread-1 (handle):
Traceback (most recent call last):
  File "/usr/lib/python3.11/threading.py", line 1045, in _bootstrap_inner
    self.run()
  File "/usr/lib/python3.11/threading.py", line 982, in run
    self._target(*self._args, **self._kwargs)
  File "/home/ks/LLM/process_iprover_v3.py", line 1816, in handle
    for msg in _ea_iter_json_messages(conn):
  File "/home/ks/LLM/process_iprover_v3.py", line 1394, in _ea_iter_json_messages
    chunk = conn.recv(8192)
            ^^^^^^^^^^^^^^^
ConnectionResetError: [Errno 104] Connection reset by peer