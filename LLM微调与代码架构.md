收到。我会给出一套面向在线 iProver 交互的清晰代码架构与落地步骤，覆盖两条方案（A: 全 Transformer；B: LLM 精排），把“SELECT→RERANK→融合→调度”的流水线做成稳定、可观测、可回退的组件。

Checklist

定义在线流水线分层与模块边界（SELECT/RERANK/融合/调度）
规划代码结构与接口契约（inputs/outputs/错误模式）
配置与服务化形态（本地函数/HTTP 服务）
可靠性与观测（缓存、超时、回退、日志/指标）
优先落地的最小改造点与任务序列
架构总览（在线链路）
SELECT（Bi-Encoder 召回）
输入：conjecture（文本/特征）
输出：Top-K 候选 {id, text, meta, S_bi}
能力：静态库检索 + 题内增量库 + text→embedding LRU 缓存
RERANK（精排）
方案 A：Cross‑Encoder（批量前向）→ S_cross
方案 B：LLM 本地服务（vLLM/TGI/Transformers）listwise 或 pairwise → S_llm
分数融合
S = λ_cross·S_cross + λ_bi·S_bi + λ_heur·S_heur（离线网格搜索 λ，在线读取配置）
调度与回包
Top‑M（融合）+ Oldest M1 + Lightest M2 + ε 随机，形成最终优先级；记录召回与调度明细
代码结构（在现有仓库上最小增量）
新增包：LLM/ea/（在线执行层）

LLM/ea/types.py
定义数据契约
Clause{id:int, text:str, features:dict, meta:dict}
SelectRequest/Response, RerankRequest/Response, FusionRequest/Response
LLM/ea/select/selector.py
class Selector: load(model, spm, faiss_static), encode_query, search(K), add_in_problem(ids,texts,embeds), cache
LRU: text→embedding；线程安全
LLM/ea/backends/base.py
abstract RankerBackend.score(req: RerankRequest) -> RerankResponse
LLM/ea/backends/cross_encoder.py（从现有训练/推理代码抽内核）
复用 tokenization/forward/batching；支持 max_len/batch/设备/超时
LLM/ea/backends/llm_local.py（方案 B 预留）
vLLM/TGI/HF 三选一适配；listwise/pairwise 输入组装；解析失败回退
LLM/ea/fusion.py
fuse(scores_dicts, lambdas, norm='z'|'minmax')，默认 minmax；可读“bi/heur”来自字段或外部 JSON
LLM/ea/scheduler.py
schedule(candidates, scores, policy_config) → ordered ids（Top‑M + Oldest/Lightest/ε）
LLM/ea/pipelines.py
class PipelineA(Pipeline): SELECT→CrossEncoder→Fusion→Schedule
class PipelineB(Pipeline): SELECT→LLM→Fusion→Schedule
LLM/ea/config.py
dataclasses + YAML 读取：模型路径、索引、K、M、λ、时间预算、回退策略
LLM/ea/telemetry.py
结构化日志/计时/Counter；可选 Prometheus
对接层改造（最小侵入）

process_iprover_v3.py
在 _ea_handle_scores_req 中调用 pipelines.Pipeline*.run(request)
支持 --pipeline {A_transformer_ce,B_llm_rerank} 与 YAML 配置
记录 per-request 指标：select_K、召回命中率、精排耗时、融合与调度输出
batch_ranker.py
保留为离线/调试 CLI；已加融合能力（你刚完成）
抽出可 import 的函数（可作为在线备选回退路径）
服务化（可选）

select_service.py 继续提供 HTTP；同时暴露 importable Selector（在线内嵌优先，HTTP 为跨进程选项）
新增 ranker_service.py（可选）：把 Cross‑Encoder/LLM 封装成 HTTP 服务，EA 侧只做编排
接口契约（简版）
Selector
encode_query(conj:str)->np.ndarray
search(query_vec,K:int)->List[Candidate{id,text,meta,score_bi}]
add_in_problem(docs:List[Clause])、warm_cache(texts)
错误：IndexEmptyError / EncodeTimeoutError / SearchTimeoutError
RankerBackend
score(req:{conjecture:str, candidates:[{id,text,meta,score_bi}], topk:int})->[{id, score_cross}]
错误：OOM/Timeout/BatchError（全部异常→返回空数组，由上层回退）
Fusion
fuse(bags: dict(name->id->score), lambdas: dict)->id->score_fused
若某分量缺失→按 0 处理；空结果→退回主分量
Scheduler
schedule(cands, fused_scores, cfg)->final_ids
配置与运行形态
YAML（示例）
pipeline: A_transformer_ce
select: {index: /.../corpus.faiss, model:/.../biencoder.pt, spm:..., K:1024, in_problem_index:true, lru:50000}
ranker: {type: cross_encoder, model:/.../cross.pt, spm:..., batch:256, max_len:256, timeout_ms:800}
fusion: {lambda_cross:0.6, lambda_bi:0.3, lambda_heur:0.1, norm:minmax}
schedule: {top_m:64, oldest:8, lightest:8, epsilon:0.02}
budgets: {select_ms:200, rerank_ms:1000, total_ms:1600}
CLI
process_iprover_v3.py serve --config ea_pipeline.yaml --pipeline A_transformer_ce
可靠性与观测
超时与回退
SELECT 超时/失败→直接下放启发式或精排少量（K 取小）
RERANK 超时/失败→用 S_bi（或启发式）回包
融合失败→用主分量（Cross 或 Bi）回包
缓存
text→embedding LRU；conjecture 与热点子句预热
in-problem index 增量 add，定期去重
观测
每 request JSON 摘要：K、命中、各阶段耗时、回退标志、Top‑N 预览
可选 Prometheus：阶段耗时直方图、失败率、GPU/CPU 利用率采样
最小落地任务（按优先级）
打通在线 Pipeline A
LLM/ea/types.py、select/selector.py、backends/cross_encoder.py、fusion.py、scheduler.py、pipelines.py、config.py、telemetry.py（最简实现）
process_iprover_v3.py：scores_req 前调用 PipelineA.run()
Selector：支持静态 FAISS + 题内 add + LRU
Cross‑Encoder 后端：复用现有推理代码（batch、max_len、设备绑定）
配置与回退
新增 YAML 配置，budgets+回退策略生效
日志落地 per-request JSON，记录所有关键指标
融合搜索（离线）
在现有 select/eval_retrieval.py 上加 λ 网格搜索（目标 NDCG@64/100）
将 λ 写回在线 YAML
Pipeline B 支架
backends/llm_local.py：vLLM/TGI 适配器 & 一个 dry-run/mock 实现（先通路再优化）
process_iprover_v3.py 加 --pipeline B_llm_rerank
索引加速与增强
HNSW/IVF‑PQ 可插拔（与 Flat 并存）
题内增量库持久化策略（可选）
质量门与测试
单测：Selector 编解码/增量 add；Cross‑Encoder 批推理；融合边界（空分量/全等分）与调度
端到端 dry‑run：固定输入→稳定输出（CI 可跑）
误差与边界（快速提示）
空候选或全零分：融合 re-normalize 后扁平，需抖动或主分量回退（已在实现中考虑）
大量正例/低 K：报告 reachable recall（扣除未覆盖正例），增大 K=500/1000
并发：Selector LRU/FAISS 增量 add 做锁粒度控制；RERANK 批处理合并请求
下一步你可以直接做
新增 LLM/ea 包与最小类骨架（types/selector/backends/fusion/scheduler/pipelines/config/telemetry）
在 process_iprover_v3.py 注入 PipelineA.run(…)（保留旧启发式回路作为回退）
用现有 Cross‑Encoder 与 Bi‑Encoder 资源跑一次在线 dry-run（单题），产出 per-request JSON 指标
用 eval_retrieval 做 λ 网格搜，更新 YAML 并再跑一次对比