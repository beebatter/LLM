
# iProver–LLM 工程执行计划（TODO & 里程碑）
> 合并自《白皮书_并行方案_数据训练_含Tokenizer修订_加Introduction与结构》和《架构_数据流_对接计划》

---

## 0. 项目目标（复述）
- 在**不修改 iProver 证明内核**的前提下，用 **Bi‑Encoder SELECT** + **RERANK**（两条并行方案：A=Transformer Cross‑Encoder、B=本地大模型 QLoRA）提升 **给定子句选择**质量与效率。  
- 成果衡量：在固定超时内的**已证题数**、**平均 given 数**、**平均时延**显著优于启发式基线；离线 **Recall@K（SELECT）** 与 **NDCG@K（RERANK）** 达标。

---

## 1. 工程里程碑（Milestones）
里程碑以“最小可用 → 并行对比 → 持续优化”的顺序推进。每个里程碑列出**验收标准（Deliverables & KPI）**。

### M0（第 0 周）：环境与基线可复现
**任务**
- [ ] 固化题集与随机种子；冻结一份“启发式-only”基线配置；
- [ ] 复现 `run_batch_pipeline.py` → `Logs/*.raw.log` → `iplog_to_dataset.py` → `datasets/*.jsonl`；产出数据报告；
- [ ] 建立统一评测脚本（离线 & 在线）：`eval_rank_metrics.py`（雏形）。

**交付**
- `datasets/clauses.jsonl`、`datasets/failed.jsonl`、`dataset_report.json`、`metrics_baseline.json`  
**KPI**
- 启发式基线在线评测跑通；日志/数据路径固定。

---

### M1（第 1–2 周）：高质量语料库（Acceptance Gates 全通过）
**任务**
- [ ] `iplog_to_dataset.py`：新增 `conjecture_text`；统一负例桶优先级；输出 `sample_weight`；
- [ ] `make_splits.py`：按 **problem_name** 分组切分 train/val/test；去重；控制每桶占比（训练集建议：`given≈50%/simpl≈20%/passive≈25%/never≈5%`）；
- [ ] `train_sentencepiece.py`：训练自研 **SentencePiece（Unigram，24k）**；变量归一 `VAR`、保留 `()=,~|&` 与 `<H*/U*/E*/C*/B*>`、`<Q>/<D>`；
- [ ] 验收脚本：长度/OOV/桶分布/正负比例/跨集去重。

**交付**
- `train.jsonl / val.jsonl / test.jsonl`、`spm_logic.model`、`dataset_gate_report.json`  
**KPI**
- 平均长度 ≤ 160 tokens、OOV≈0、按题切分无泄漏；训练负例占比 1:3～1:5 达成。

---

### M2（第 2–3 周）：SELECT（Bi‑Encoder）训练 + 向量索引
**任务**
- [ ] `train_biencoder.py`：InfoNCE + 难负挖掘；结构前缀与 `conjecture_text` 输入模板落地；
- [ ] `encode_and_build_index.py`：离线编码静态库 + 题内增量库；`select_service.py`：HTTP Top‑K 检索；
- [ ] `process_iprover_v3.py`：在 `_ea_handle_scores_req` *前*插入 SELECT；落地 `--select-top-k`。

**交付**
- `models/biencoder/best.pt`、`indexes/global.faiss`、`select_service`、`metrics_select_offline.json`  
**KPI**
- 验证集 **Recall@64 ≥ 启发式候选**；在线接入后，不启用精排仅 SELECT→调度，已证题数≥基线 +5%。

---

### M3A（第 3–4 周）：方案 A – Cross‑Encoder 精排（全 Transformer）
**任务**
- [ ] `train_cross_encoder.py`：pairwise/listwise 训练；max_len 256；
- [ ] `batch_ranker.py`：新增 `--backend cross_encoder`；分数融合接口；
- [ ] `process_iprover_v3.py`：`--pipeline A_transformer_ce`；记录 RERANK 延迟与分布。

**交付**
- `models/cross_encoder/best.pt`、`metrics_rerank_offline.json`、`metrics_online_pipelineA.json`  
**KPI**
- 离线 **NDCG@64** 较仅 SELECT 提升 ≥ 8%；在线 **已证题数 +10%**、平均时延不升高 >15%。

---

### M3B（第 3–5 周，并行）：方案 B – 本地大模型 RERANK（DeepSeekMath‑32B + QLoRA）
**任务**
- [ ] `make_listwise_chunks.py`：K=64 listwise 样本（目标分布=one‑hot/软标签/蒸馏）；
- [ ] `train_llm_lora.py`：QLoRA 训练；`batch_ranker.py --backend llm_local`；
- [ ] `process_iprover_v3.py`：`--pipeline B_llm_rerank`；解析失败回退；融合权重网格搜索。

**交付**
- `models/ds32b-lora/`、`metrics_rerank_offline_llm.json`、`metrics_online_pipelineB.json`  
**KPI**
- 离线 **NDCG@64** ≥ 方案 A；在线 **已证题数 +10%~15%**，资源与延迟可接受（VRAM、吞吐可观测）。

---

### M4（第 5–6 周）：A/B 汇总与蒸馏优化
**任务**
- [ ] 统一离线/在线评测面板与报表导出；统计每域/每难度指标；
- [ ] 将方案 B 的精排分数蒸馏到 Cross‑Encoder（可选），降低在线成本；
- [ ] 召回率与融合 λ 自动化调参（启发式回退保持）。

**交付**
- `report_ab_summary.md`、`cross_encoder_distilled.pt`、`metrics_online_final.json`  
**KPI**
- 最终线上 **已证题数 ≥ 基线 +15%**；给定效率与时延维持红线以内。

---

## 2. 并行工作面（按领域划分）

### 数据 & 预处理
- 负责人：Data Owner（待指派）  
- 产出：`train/val/test.jsonl`、`spm_logic.model`、`dataset_gate_report.json`

### 检索（SELECT）
- 负责人：Retrieval Owner（待指派）  
- 产出：`biencoder.pt`、`faiss index`、`select_service`、`metrics_select_offline.json`

### 精排（RERANK）A：Cross‑Encoder
- 负责人：CE Owner（待指派）  
- 产出：`cross_encoder.pt`、`metrics_rerank_offline.json`

### 精排（RERANK）B：LLM QLoRA
- 负责人：LLM Owner（待指派）  
- 产出：`ds32b-lora`、`metrics_rerank_offline_llm.json`

### 在线集成 & A/B
- 负责人：Integration Owner（待指派）  
- 产出：`metrics_online_*`、A/B 报告与看板

---

## 3. 任务清单（TODO，大纲）

### 3.1 代码变更（最小改造位）
- [ ] `iplog_to_dataset.py`：`--add-conjecture`、桶优先级统一、`sample_weight`  
- [ ] `run_batch_pipeline.py`：参数化负采样/桶占比、输出 `dataset_report.json`  
- [ ] `process_iprover_v3.py`：SELECT 插入点、`--pipeline`、日志指标  
- [ ] `batch_ranker.py`：`cross_encoder / llm_local` 后端与容错回退、融合权重接口  
- [ ] 新增：`make_splits.py / train_sentencepiece.py / train_biencoder.py / encode_and_build_index.py / select_service.py / train_cross_encoder.py / make_listwise_chunks.py / train_llm_lora.py / eval_rank_metrics.py`

### 3.2 数据治理
- [ ] 按题切分 & 跨集去重  
- [ ] 负例桶配额与 `NEG_given_nonproof/simplified/passive_only/never_seen` 目标占比  
- [ ] 变量归一、结构前缀 token、签名前缀（可选）与随机重命名增强（20%）  
- [ ] 失败集（weak negatives）仅注入训练，低权重，题内上限

### 3.3 模型训练 & 评测
- [ ] Tokenizer 训练与验收（长度/OOV）  
- [ ] Bi‑Encoder：InfoNCE + 难负；R@K 指标  
- [ ] Cross‑Encoder：pair/listwise；NDCG@K、AUC  
- [ ] LLM QLoRA：listwise；解析容错 & 回退  
- [ ] 统一评测：离线（R@K/NDCG）与在线（已证题数/given/时延/资源）

### 3.4 上线保障
- [ ] 回退路径：RERANK 失败→仅 SELECT；SELECT 服务异常→启发式；始终混入“最老/最轻/随机”  
- [ ] 指标与告警：召回率、缓存命中率、SAT 使用率、延迟/VRAM/吞吐  
- [ ] 版本化：模型/词表/索引/题集固定；随机种子固化

---

## 4. 风险清单与缓解
- **LLM 解析失败或超时**：严格 JSON 模式+容错；回退到 Bi‑Encoder/启发式  
- **召回不足**：扩大 K、提升难负、特征前缀增强  
- **分布漂移**：周期性再采数与重训；数据看板监控桶分布/长度漂移  
- **成本过高**：优先 SELECT；RERANK 缩短 K；蒸馏到 Cross‑Encoder

---

## 5. 成果包清单（交付物）
- 数据：`train/val/test.jsonl`、`failed.jsonl`、`dataset_report.json`  
- 模型：`spm_logic.model`、`biencoder.pt`、`cross_encoder.pt`、`ds32b-lora`  
- 索引与服务：`indexes/*.faiss`、`select_service`  
- 报告：`metrics_select_offline.json`、`metrics_rerank_offline*.json`、`metrics_online_*.json`、`report_ab_summary.md`

---

### 附：甘特式时间线（建议）
- 周 0：M0  
- 周 1–2：M1  
- 周 2–3：M2  
- 周 3–4：M3A（并行启动 M3B）  
- 周 3–5：M3B  
- 周 5–6：M4

