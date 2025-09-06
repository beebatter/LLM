# iProver LLM 外部代理白皮书（实验进展与训练改进计划）

## 背景与目标

- 目标：让 LLM 作为 iProver 的外部重排器，对“猜想 + K 条候选子句”输出候选级分布，用于给定选择与子句排序，提升证明效率。
- 两条路线：
  - LLM 精排（listwise 学习，输出分布/分数）。
  - 传统 Transformer（Bi-Encoder 召回 + Cross-Encoder 精排），作为教师与融合成员。
- 输出协议：
  - 在线期望直接得到每候选的打分分布（建议直算 softmax 分布或候选打分头），避免依赖 JSON 解析的脆弱性。

## 当前实验设置（数据/模型/脚本）

数据
- listwise 教师融合数据：
  - 训练：`/root/autodl-tmp/Training/datasets/listwise/train.listwise.teacher.jsonl`
  - 验证：`/root/autodl-tmp/Training/datasets/listwise/val.listwise.teacher.jsonl`
- 生成：`scripts/build_listwise.sh`（支持 Cross+Bi 教师融合：softmax(λ_ce·s_ce + λ_bi·s_bi; τ)）。

模型与训练
- 7B：`DeepSeek-Prover-V2-7B`（本地），LoRA，2 epoch，bf16 / 8bit / none；max_len≈2048。
- 32B：`Goedel-Prover-V2-32B`（本地），QLoRA（4bit），2 epoch；max_len≈1024。
- 训练脚本：`training/train_llm_listwise_sft.py`
  - 关键稳健化：pad→eos、resize_token_embeddings、labels 忽略 pad/越界、use_cache=False、梯度检查点。
  - 量化开关：`--quant {4bit,8bit,none}`。

评测与重排
- 列表一致性评测（直算 softmax）：`training/eval_llm_listwise.py --mode direct`（候选独立前向，逐 token 对数似然求和 → softmax）。
- 检索评测（Bi/CE/LLM/Fusion）：
  - 生成 CE/LLM 重排：`select/rerank_with_cross_encoder.py`、`select/rerank_with_llm.py`。
  - 评测融合：`select/eval_retrieval.py`（支持 λ 网格）。

## 结果摘要（离线）

列表一致性（与教师目标的对齐；验证子集 n≈102）
- 7B：len_match=1.0，MAE≈0.030，MSE≈0.0137，Pearson≈0.05。
- 32B：len_match=1.0，MAE≈0.030，MSE≈0.0137，Pearson≈0.27。

检索指标（把 LLM 当重排器；同一 queries/labels/meta，n≈27）
- Bi 与 CE：CE 普遍优于 Bi；
- LLM 7B：早段 Hit@10 ≈ CE，整体相近；
- LLM 32B：早段略弱于 CE，中深段接近 Bi；
- Bi+CE 融合 λ 网格（固定 λ_ce=1.0）：
  - 最优 Hit@10：λ_bi=0.1（Hit@10≈0.333，NDCG@32≈0.242）。
  - 最优 NDCG@32：λ_bi=0.2（NDCG@32≈0.249，Hit@10≈0.259）。

简短结论
- 现状下，LLM 重排并未稳定优于 CE/Bi（7B 早段≈CE，32B 早段更弱；listwise 直评 7B 相关性≈0，32B≈0.27）。
- 更像“微调尚未到位/评测不完全对齐”，非模型无能。

## 诊断与成因

- 目标-指标不匹配：训练拟合教师软分布（listwise CE+Bi），评测偏向 Hit@K/NDCG；建议加排序目标（ListMLE/Pairwise）或蒸馏 CE logits（KL）。
- 长度偏置：直评按“对数似然总和”，长候选吃亏；应使用“平均每 token 对数似然”或加长度惩罚。
- 上下文截断：为避 OOM 降低 input_max/target_max，损失关键信息；H800 可开 flash-attn/TF32，尝试 2048/256。
- 量化差异：7B 训练 bf16、评测 4bit 会拉低表现；建议 7B 用 8bit/none 做对照。
- 训练信号偏“模板模仿”：正例均分、缺少“正例内部强弱”与候选级信号；损失落在 token 层而非候选层。
- 样本量与超参：仅 2 epoch；LoRA r/α/dropout/调度需继续调优；难负占比与数据增强可强化。
- 提示一致性：需再核验训练/评测 prompt 完全一致（分隔符/字段/顺序）。

## 训练改进实现计划（适配本仓库）

一、评测口径校准（不改模型即可增益）
- 直评改“平均 LL”：`s_mean = (1/|c|)Σ log p(x_t|prompt)`；新增选项：`--score-type {sum-ll, mean-ll}`。
- 逐候选独立前向：避免 K 候选同上下文引入注意力泄漏；保留现有 `--mode direct`，确保每候选单独前向。
- 组内归一：对窗口 {s_i} 做 z-score 或减中位数/除 MAD（新增 `--zscore`）。
- 温度/偏置标定：dev 上拟合 p̂=softmax((s−b)/τ)，加入 `--calib-tau/--calib-bias`。
- 可选 PMI：`s_PMI = logP(c|conj) − λ·logP(c)`（λ∈[0.5,1.0]，需无条件提示的基线前向）。

二、候选级目标学习（替代“生成 JSON 的 SFT”）
- 模型端增加“候选打分头”：为每候选包裹 `<CAND_START>…<CAND_END>`，取 `<CAND_START>` 隐状态 h_i，经线性头得 s_i；p̂=softmax(s)。
- 损失：对齐教师软分布 p* 的 CE/KL；可混入少量 pairwise（Bradley–Terry/margin）作正则。
- 实现：在 `training/train_llm_listwise_sft.py` 增加 `--score-head` 模式或新脚本 `train_llm_listwise_scorehead.py`；LoRA 仅接上层与打分头，降低显存与不稳。

三、教师分布增强（调尖正例内部形状）
- 使用 dev 网格搜 `(pos_mass m, temperature τ, blend α)`（脚本 `target_dist_tuner.py`），生成更“尖锐”的 target_scores：
  - pos_weights = (1−α)·uniform + α·softmax(teacher/τ)；
  - 仅作用于正例子集，负例均分剩余质量；整体加 ε 平滑再归一。
- 管道：在 `scripts/build_listwise.sh` 中增加读取最佳 YAML 并重写 target_scores（不改样本本体）。

四、训练细节抛光（当前配方微调）
- 学习率：7B LoRA 建议 5e-5～1e-4；32B QLoRA 5e-5；warmup_ratio≈0.03，cosine 调度，epoch 2–3。
- 候选随机化：每 epoch 重排候选顺序，目标同步重排。
- 难负注入：提高 `NEG_given_nonproof` 占比（如 0.6），增强早段命中能力。
- 蒸馏 logits：直接蒸馏 CE 的 K 维原始分，τ=1–2，比离散软分更细腻。
- 性能：H800 开启 flash-attn/TF32，use_cache=False，梯度检查点；7B 用 8bit/none 评测对照。

五、融合与评测（与 Bi/CE 协同）
- 融合：S = λ_ce·S_ce + λ_bi·S_bi + λ_llm·S_llm； per-query 归一策略（min-max/z-score/RRF）可选。
- 评测：Hit@{10,32,64}、NDCG@{10,32,64}、Recall@K；dev 小网格选 λ。

六、两天落地排期
- Day 1：
  - 评测口径：`--score-type mean-ll`、`--zscore`、`--calib-*`；可选 PMI；H800 上提至 input_max=2048/target_max=256 复测。
  - 目标分布：运行 `target_dist_tuner.py` 搜 m,τ,α，重写 `train/val.listwise.teacher.jsonl` 的 target_scores。
- Day 2：
  - 7B：LoRA lr=1e-4，epoch=2–3，listwise CE/KL（或 score-head）；随机候选；难负 0.6；8bit/none 评测对照。
  - 32B：QLoRA lr=5e-5，epoch=2，同上；
  - 评测：listwise 一致性（mean‑LL 口径）、Bi top‑200 重排（Hit/NDCG）、三方融合小网格；固化最佳 λ。

## 操作指引（可选参考）

- 生成 listwise（含教师融合）：`bash scripts/build_listwise.sh`（按需设置 CROSS/BI/SPM 路径、τ、λ_bi）。
- 训练 7B（单卡）示例：
  - `CUDA_VISIBLE_DEVICES=0 python training/train_llm_listwise_sft.py --model /root/autodl-tmp/models/DeepSeek-Prover-V2-7B --train .../train.listwise.teacher.jsonl --val .../val.listwise.teacher.jsonl --out .../deepseek7b-listwise-lora --batch 2 --grad-accum 8 --epochs 2 --lr 1e-4 --max-len 2048 --bf16 --quant 8bit`
- 训练 32B（单卡）示例：
  - `CUDA_VISIBLE_DEVICES=1 python training/train_llm_listwise_sft.py --model /root/autodl-tmp/models/Goedel-Prover-V2-32B --train ... --val ... --out .../goedel32b-listwise-lora --batch 1 --grad-accum 16 --epochs 2 --lr 8e-5 --max-len 1024 --bf16 --quant 4bit`
- 列表一致性评测（直算）：
  - `python -m LLM.training.eval_llm_listwise --data .../val.listwise.teacher.jsonl --mode direct --model <base> --lora <adapter_dir> --bits 4 --input-max 1536 --target-max 192`
- LLM 重排 → 检索评测：
  - `python -m LLM.select.rerank_with_llm ...` → `python -m LLM.select.eval_retrieval ... --lambda-ce ... --lambda-bi ...`

## 产物与路径（当前）

- 数据：`/root/autodl-tmp/Training/datasets/listwise/{train,val}.listwise{.teacher}.jsonl`
- 模型：
  - 7B LoRA：`/root/autodl-tmp/Training/models/deepseek7b-listwise-lora`
  - 32B LoRA：`/root/autodl-tmp/Training/models/goedel32b-listwise-lora`
- 评测：
  - LLM 直评指标：各自 out 目录下 `eval_listwise_direct.json`
  - 检索评测：`metrics_llm7b.json`、`metrics_llm32b.json`（与 Bi/CE 可对比）

---

结论：当前 LLM 重排已具备可用性，但尚未稳定优于 CE/Bi。优先推进评测口径校准与目标分布调尖，随后引入候选级打分头与排序损失；同时做三方融合的小网格提升中深段 NDCG。在两张 H800 的资源下，上述计划可在两天内完成一轮闭环并复核指标。


# iProver–LLM 并行方案白皮书（含：数据集生成详解 + 两条可落地方案）
> 目标：基于你现有代码（`run_batch_pipeline.py / process_iprover_v3.py / iplog_to_dataset.py / batch_ranker.py`）同时实现并评测两条并行路线：  
> **方案 A（全 Transformer）**：Transformer→**向量检索（Bi‑Encoder）**→**Transformer Cross‑Encoder 精排**。  
> **方案 B（混合式）**：Transformer→**向量检索（Bi‑Encoder）**→**本地大模型（DeepSeekMath‑32B，LoRA）精排**。  
> 核心强调 **数据集（语料库）生成质量** 与 **可复现实验指标**。

---

# 项目背景与总览（Introduction）

**项目目标**：用大模型（LLM）+ Transformer 编码器为**饱和式自动定理证明器**（如 iProver）提供 **Prover Guidance**，在不改动证明内核正确性的前提下，显著提升**给定子句（given clause）选择**质量与在线效率，最终提高在固定超时内的**已证题数**与**平均给定步数/时延**表现。

**背景**：饱和式 ATP（E/iProver/Vampire 等）依赖“给定子句循环（given-clause loop）”，面临**候选子句爆炸**、启发式信号弱、跨领域迁移差等问题。我们引入两级策略：  
- **SELECT（召回）**：用对比学习训练的 **Bi‑Encoder** 生成向量嵌入，对 *目标猜想 q* 与 *候选子句 d* 做独立编码，以相似度在大规模库中**极速 Top‑K 检索**；  
- **RERANK（精排）**：对 SELECT 的候选再做**细粒度打分**。我们提供两条并行实现：  
  1) **方案 A**：Transformer **Cross‑Encoder** 精排；  
  2) **方案 B**：**本地大模型**（如 DeepSeekMath‑32B，QLoRA）做 **listwise** 精排。

**范围与约束**：  
- 不修改 iProver 的证明逻辑与正确性；仅通过外部代理影响“给定子句优先级”。  
- LLM 侧**保持原生 tokenizer**；只为自研 Transformer（Bi-/Cross-Encoder）训练专用 tokenizer，并做轻量规范化（变量归一/结构前缀/签名提示）。  
- 保留**启发式安全网**（最老/最轻/随机探索）保证鲁棒性与可回退。

---

## 项目结构（脚本与目录职责）

## 1. 代码结构（按职责）
- **process_iprover_v3.py**（在线 EA 服务）：连接 iProver，收集窗口（register/passive/given/simplified），在 `scores_req` 时构造 batch → 预筛 → 调用精排后端 → 回包；可选 SAT ground 辅助与打分缓存。
- **batch_ranker.py**（精排后端统一入口）：封装后端（heuristic/未来 cross_encoder/llm_local/llm_api），负责 chunk 化输入与分数解析、失败回退。
- **run_batch_pipeline.py**（离线批跑）：调度 iProver+EA；输出原始 NUL JSON 日志、监督样本 JSONL、失败集；内置负例分桶与 frontier/born 采样策略。
- **iplog_to_dataset.py**（日志→数据集）：解析 CNFRefutation 与运行轨迹，写出 `text, features, label, neg_bucket`；建议新增 `conjecture_text` 支持 Bi‑Encoder。

## 2. 在线/离线数据与控制流
```
iProver ──register/passive/given/simplified──▶ EA
      └────────────scores_req(ids)────────────▶  预筛 → batch_ranker(后端) → scores_res
EA ──szs_result_out/server_queries_end───────▶ iProver

run_batch_pipeline.py → Logs/*.raw.log → iplog_to_dataset.py → datasets/*.jsonl / failed.jsonl
```

## 3. 已实现能力
- 在线：窗口管理、预筛（目标函子、单位子句、共享常量、长度惩罚等）、打分缓存、可静音/截断日志、可选 SAT ground。  
- 离线：负例分桶（given_nonproof/simplified/passive_only/never_seen）、frontier/出生轮次、失败集（含最后一次给定/被动子句文本与特征）。

## 4. 与两条并行方案的挂载点
- **方案 A（Transformer 两层）**：
  - 在 `_ea_handle_scores_req` 前挂 **SELECT（Bi‑Encoder）**；Top‑K 进入 `batch_ranker.py --backend cross_encoder`；
  - EA 侧融合 `S=λ_bi·S_bi+λ_ce·S_ce+λ_h·S_heur`，并混合“最老/最轻/随机”。
- **方案 B（SELECT + LLM 精排）**：
  - SELECT 同上；Top‑K 进入 `--backend llm_local`（本地 LoRA LLM，listwise 打分）；
  - 可把 LLM 分数蒸馏至 Cross‑Encoder，降低成本。

## 5. 最小改造位
- `iplog_to_dataset.py`：新增 `conjecture_text`、统一桶优先级、输出 `sample_weight`；
- `run_batch_pipeline.py`：参数化负采样与桶占比，输出 `dataset_report.json` 与 `problem_splits.json`；
- 新增：`train_sentencepiece.py / train_biencoder.py / encode_and_build_index.py / select_service.py / train_cross_encoder.py / make_listwise_chunks.py / train_llm_lora.py / eval_rank_metrics.py`；
- `process_iprover_v3.py`：插入 SELECT；`--pipeline {A_transformer_ce,B_llm_rerank}`；记录 Top‑K 召回；
- `batch_ranker.py`：增 `cross_encoder/llm_local/llm_api` 后端与分数融合。

## 6. 评测与落地顺序（摘要）
- 离线：SELECT 的 Recall@K、RERANK 的 NDCG@K；  
- 在线：已证题数、平均 given、延迟/GPU；  
- 顺序：Dataset Gate → Tokenizer → Bi‑Encoder → 索引服务 → Cross‑Encoder/LLM → 融合/调度 → A/B。


> 目标：基于你现有代码（`run_batch_pipeline.py / process_iprover_v3.py / iplog_to_dataset.py / batch_ranker.py`）同时实现并评测两条并行路线：  
> **方案 A（全 Transformer）**：Transformer→**向量检索（Bi‑Encoder）**→**Transformer Cross‑Encoder 精排**。  
> **方案 B（混合式）**：Transformer→**向量检索（Bi‑Encoder）**→**本地大模型（DeepSeekMath‑32B，LoRA）精排**。  
> 核心强调 **数据集（语料库）生成质量** 与 **可复现实验指标**。

---
## 目录
1. 项目概览与并行方案对比  
2. **语料库（数据集）构建：从原始日志到训练集**（重点）  
3. 方案 A：全 Transformer（SELECT + Cross‑Encoder RERANK）  
4. 方案 B：Bi‑Encoder SELECT + 本地大模型 RERANK（DeepSeekMath‑32B, QLoRA）  
5. 统一评测与对比指标（离线 & 在线）  
6. 工程落地：改动点、CLI、服务化、资源预估  
7. 风险与回退、里程碑与验收

---

> 本节汇总“现有脚本 + 规划脚本”的**职责边界**与**交互关系**。代码与命令行细节在白皮书其他章节逐步落地。

### A. 核心在线链路

- **`process_iprover_v3.py`**（EA 服务桥接）  
  负责 iProver ↔ 外部代理的消息编解码与日志；在收到 `scores_req` 时调用 SELECT（Bi‑Encoder 向量检索）与 RERANK（Cross‑Encoder 或 LLM），并输出分数/混合调度结果；记录召回率与调度明细。

- **`batch_ranker.py`**（精排后端统一入口）  
  封装多种打分后端：`cross_encoder / llm_local / llm_api / heuristic`，并支持**分数融合**与**回退**；对接 `process_iprover_v3.py`。

### B. 批处理与数据闭环

- **`run_batch_pipeline.py`**（离线批跑与采数）  
  批量调度 iProver+EA，生成 `*.raw.log`、`datasets/*.jsonl` 与 `failed.jsonl`；统一**失败原因**与**样本统计**。

- **`iplog_to_dataset.py`**（日志→监督样本）  
  解析 CNFRefutation/给定轨迹，输出带 **label（正/负/桶）** 与 **features** 的样本；支持新增 `conjecture_text` 字段、弱负权重与分桶策略。

### C. 选择/检索与训练（规划/扩展脚本）

- **`train_sentencepiece.py`**（Tokenizer 训练）  
  仅为自研 Transformer 训练 **SentencePiece（Unigram/BPE）**；LLM 侧 tokenizer 不改。

- **`train_biencoder.py`**（SELECT 训练）  
  对比学习（InfoNCE + 难负）训练双塔模型；导出 `best.pt` 与 SPM。

- **`encode_and_build_index.py` / `select_service.py`**（向量库与在线检索）  
  离线编码静态库 + 题内增量库（FAISS/HNSW），提供在线 Top‑K 检索服务。

- **`train_cross_encoder.py`**（方案 A 精排）  
  以二分类/Pairwise/Listwise 训练交叉编码器；在线批量前向。

- **`make_listwise_chunks.py` / `train_llm_lora.py`**（方案 B 精排）  
  生成 **listwise** 窗口样本（K=64），对本地大模型做 QLoRA 微调；推理落在 `batch_ranker.py --backend llm_local`。

- **`eval_rank_metrics.py`**（统一评测）  
  离线：Recall@K、NDCG@K、MAP/AUC；在线：已证题数、平均 given、耗时与资源占用。

### D. 资产与目录

- **`datasets/`**：监督样本与分割集（train/val/test）、失败集（仅作弱负）、报告 JSON。  
- **`models/`**：Tokenizer、Bi‑Encoder、Cross‑Encoder、LoRA 适配器等。  
- **`indexes/`**：FAISS/HNSW 向量索引与缓存。  
- **`Logs/`**：EA 原始交互日志与批跑输出。

### E. 兼容与传承（来自初版 README 的要点）
- **`auto_fof_corpus_builder_local.py` / `html_fof_extractor.py`**：早期集成与问题列表抽取工具，可继续用于生成题单与 HTML 抽取。（现已由批跑与数据链路替代为主）  
- **`process_iprover_v2.patched.py` / `setup_and_run_fof.sh`**：历史版本的 EA 与一键脚本；作为备用方案保留。

---


## 1) 项目概览与并行方案对比

### 1.1 架构共识（两条方案共享）
- **SELECT（召回）**：使用对比学习训练的 **Bi‑Encoder** 将 *猜想/目标（q）* 与 *子句（d）* 独立编码，余弦或内积相似度检索 Top‑K。  
- **RERANK（精排）**：对 SELECT 的 Top‑K 进行更精细的打分，返回最终优先级；并与“最老/最轻/随机探索”混合调度，喂给 iProver 的 given 选择器。  
- **数据闭环**：iProver↔EA（`process_iprover_v3.py`）交互日志 → `iplog_to_dataset.py` 标注（正/负/桶）→ 训练 → EA 在线接入 → A/B 评测 → 继续采数与蒸馏。

### 1.2 两条并行方案
- **方案 A（全 Transformer）**
  - **STEP1：Bi‑Encoder SELECT**（快速 Top‑K 召回）  
  - **STEP2：Transformer Cross‑Encoder RERANK**（较小但专用的交叉编码器，输入形如 `[CLS] <Q> … </Q> <D> … </D>`，对每个(q,d) 前向一次，精度高、代价较 LLM 低）
- **方案 B（混合式）**
  - **STEP1：Bi‑Encoder SELECT**（相同）  
  - **STEP2：**本地大模型 **DeepSeekMath‑32B + QLoRA** 做 **listwise/JSON‑SFT 精排**，适配 `batch_ranker.py --backend llm_local`。

> 两条路线**并行开发**、统一离线/在线指标对比。A 更轻、更易部署；B 精度上限更高、代价更大。

---

## 2) 语料库（数据集）构建：从原始日志到训练集（重点）

### 2.1 数据来源与工具链
- **原始交互日志**：`run_batch_pipeline.py` 批跑问题 → 产生 `*.raw.log`（NUL 分隔 JSON），以及 `datasets/*.jsonl` 与 `failed.jsonl`（失败问题元信息）。  
- **解析与标注**：`iplog_to_dataset.py` 从 SZS 证明片段解析**正例**（出现在 CNFRefutation 的 c_#），其它子句按来源与状态分为**负例桶**：  
  - `NEG_given_nonproof`：给定过但未进入证明；  
  - `NEG_simplified`：被化简/消去；  
  - `NEG_passive_only`：仅在被动集出现；  
  - `NEG_never_seen`：注册缺失/异常（可为空）。  
- **失败日志**：`failed.jsonl` 仅作**弱负**来源（`last_given_clauses` / `passive_sample`），**不得进入 val/test**。

### 2.2 样本字段（建议**强制**）
每行 JSON：
```json
{
  "problem_name": "ALG050+1",
  "conjecture_text": "...",          // 新增，或改为 conjecture_sig
  "text": "clause/cnf text ...",
  "features": {"horn":1,"epr":1,"unit":0,"born":12,"conj_dist":2,...},
  "label": 1,                        // 正例=1, 负例=0
  "neg_bucket": null,                // 正例为 null; 负例为 NEG_*
  "source": "run_batch_pipeline",    // or "failed_weak"
  "sample_weight": 1.0               // 负例可按桶下调，如 given:1.0, simpl:0.8, passive:0.5, failed:0.25
}
```
> **新增字段**：`conjecture_text`（或 `conjecture_sig`），为 Bi‑Encoder 训练与在线 SELECT 做准备。

### 2.3 质量闸门（Acceptance Gates）
1) **Schema 完整**：字段齐全；正例 `neg_bucket=null`。  
2) **按题切分**：train/val/test **按 problem_name 分组**；同题不可跨集。  
3) **负例采样与桶占比（训练集）**：相对正例 1:3～1:5；桶占比建议  
   `given_nonproof≈50% / simplified≈20% / passive_only≈25% / never_seen≈5%`。  
4) **去重**：val/test 不得含与 train 相同 `text`；跨题重复尽量放到 train 或降权。  
5) **长度与规范化**：变量归一；训练时最大序列长度（建议 256 tokens）；极长条目可剔除一部分。  
6) **失败样本**：仅作为弱负注入 train，设低权重，且每题上限（50–200）。

### 2.4 预处理与分词
- **SentencePiece 16k–32k**：将 `()=,~|&` 加入 user defined symbols；变量统一 `VAR`。  
- **前缀特征 token**：把 `features` 离散化到文本前缀，如：`<H1><U0><E1><C2><B3> … clause …`；`conj_dist/born` 先分桶再映射。

### 2.5 产出两类训练集
- **SELECT（Bi‑Encoder）用**：三元组/对比形态  
  - 形式 A：`(q, d+, {d-…})` 做 InfoNCE（batch 内全负 + 周期性**难负**）；  
  - 形式 B：pairwise `(q, d+, d-)` 做 margin/softmax。  
- **RERANK 用**：  
  - **Cross‑Encoder**：二分类/排序（pairwise 或 listwise），输入拼接 `[CLS] <Q>…</Q> <D>…</D>`；  
  - **LLM（DeepSeekMath‑32B）**：**listwise**（窗口=64），产生 `{"ids":[...], "texts":[...], "target":[softmax分布]}`。

### 2.6 命令范式（示例）
```bash
# (1) 生成监督样本（带 conjecture_text）
python3 iplog_to_dataset.py \
  --raw-dir Logs/EA.* \
  --out datasets/clauses.jsonl \
  --add-conjecture \
  --bucket-priority given_nonproof,simplified,passive_only,never_seen

# (2) 按题切分 + 去重 + 桶重采样 + 弱负注入（失败集）
python3 make_splits.py \
  --input datasets/clauses.jsonl \
  --failed datasets/failed.jsonl \
  --train-out datasets/train.jsonl \
  --val-out datasets/val.jsonl \
  --test-out datasets/test.jsonl \
  --neg-ratio 4 \
  --bucket-quota 0.5,0.2,0.25,0.05 \
  --weak-failed-cap 100 --weak-weight 0.25

# (3) 词表与特征前缀
python3 train_sentencepiece.py --input datasets/train.jsonl --vocab 24000 --out models/sp/

# (4) 生成 RERANK 用 listwise 样本
python3 make_listwise_chunks.py \
  --input datasets/train.jsonl \
  --window 64 --smooth 0.1 \
  --out datasets/train_listwise.jsonl
```

---



---

## 2.7 专用 Tokenizer 策略与“命名不变性”（**只训练 Transformer 的 Tokenizer；LLM 保持原生**）

> 本项目**仅为自研 Transformer（Bi-/Cross-Encoder）训练专用分词器**；本地大模型（如 DeepSeekMath‑32B）**严格使用其原生 tokenizer**。在 LLM 侧仅做**文本规范化**（变量归一、结构前缀、签名提示），不改其词表。

### 2.7.1 训练哪一个 Tokenizer？
- **需要训练**：用于 **Bi-Encoder / Cross-Encoder** 的自研 tokenizer（SentencePiece Unigram 或 BPE）。
- **不需要训练**：任何大模型（DeepSeekMath‑32B）的 tokenizer —— **保持原生**，确保微调与推理兼容。

### 2.7.2 训练语料（只用于 tokenizer 训练，不是监督标签集）
将 `interactive_sampled_small.jsonl`（或更大集）中的以下字段写到一个纯文本文件（每行一条）：
- `conjecture_text`（或 `conjecture_sig`）
- `text`（子句）
- **结构前缀特征**（见 §2.4），放在每行最前面

预处理规则：
- **变量归一**：所有一阶变量统一为 `VAR`（示例：`X0,X1,... → VAR`）。
- **保留逻辑/括号符号**：`()=,~|&` 等作为独立符号保留。
- **异常长符号截尾**：长度 > 40 的罕见符号替换为 `<SYM_LONG>`（可选）。
- **结构前缀**：如 `<H1><U0><E1><C2><B3> ...` 一并写入。

### 2.7.3 SentencePiece 训练命令（推荐 Unigram）
```bash
spm_train \
  --input=corpus_for_tokenizer.txt \
  --model_prefix=spm_logic \
  --vocab_size=24000 \
  --model_type=unigram \
  --character_coverage=1.0 \
  --input_sentence_size=5000000 --shuffle_input_sentence=true \
  --user_defined_symbols='(<Q>,</Q>,<D>,</D>,<H0>,<H1>,<U0>,<U1>,<E0>,<E1>,<C0>,<C1>,<C2>,<C3>,<B0>,<B1>,<B2>,<B3>,VAR,(,),=,~,&,|,"," )' \
  --unk_piece=<UNK> --pad_piece=<PAD>
```
**要点**：
- 将 `<Q>, </Q>, <D>, </D>`、**全部结构前缀**以及 `VAR`、括号与逻辑符号，加入 `--user_defined_symbols`，保证它们是**独立 token**。
- `vocab_size` 建议 16k–32k；起步 24k。
- 若式子较长，可增大 `vocab_size` 或补充更多 `user_defined_symbols`。

**验收检查（必须过线）**：
- 括号与逻辑符号均保持为**独立 token**；变量几乎总为 `VAR`；
- 训练/验证集的平均长度 ≤ 160 tokens；
- OOV 接近 0（Unigram 自带鲁棒性，通常满足）。

### 2.7.4 “命名不变性”（α‑等价）如何处理？
- **强烈建议**：仅做**变量归一**（`X* → VAR`），收益显著、实现轻量。
- **谓词/函子名**：不做全局匿名化（耦合与复杂度高、可能丢失有用的弱语义）。采用**折中策略**：
  1) **签名前缀**：在样本前缀中增加 `<SIG: f/2,g/1,h/3>`（可做哈希压缩）；
  2) **数据增强**：训练时对 20–30% batch 做**局部随机重命名**（每条内一致），提升对名字的鲁棒性；
  3) **符号 dropout**：以 5–10% 概率将稀有符号替换 `<SYM_UNK>`（正负同施）。
- **LLM 侧**：不改 tokenizer，仅在 prompt 中附加**规范化视图**如：
  ```
  <Q_SIG: f/2,g/1; VARS:5> conjecture...
  <D_SIG: f/2,h/2; VARS:3> clause...
  ```

### 2.7.5 消融与度量（建议做三组）
1) **Baseline**：不做变量归一；
2) **Var‑Norm**：仅变量归一 + 结构前缀；
3) **Var‑Norm + Aug**：变量归一 + 签名前缀 + 20% 随机重命名增强。

离线指标：Bi‑Encoder 的 R@32/64、MAP；Cross‑/LLM 的 NDCG@K、Pairwise Acc；  
在线指标：已证题数、平均 given、耗时。

> 结论前置：**只训练 Transformer 的 tokenizer** 是必要且收益显著的；**LLM tokenizer 保持原生**能显著降低工程风险与复杂度。


## 3) 方案 A：全 Transformer（SELECT + Cross‑Encoder RERANK）

### 3.1 Bi‑Encoder（SELECT）训练
- **模型**：双塔 Transformer（6–8 层，hidden 512–768；共享或不共享参数均可）。Pooling：`[CLS]` 或 mean。  
- **损失**：InfoNCE（τ=0.05–0.1）；batch 内全负；每 N 步触发一次**难负挖掘**（按当前模型高分错例）。  
- **优化**：AdamW(lr=1e‑4, wd=0.01)，mix‑precision；早停指标=验证集 **Recall@K（K=32/64）**。  
- **导出**：`best.pt` + `spm.model`。

### 3.2 FAISS 建库与在线 SELECT
- **静态库**：常见公理/公共子句离线编码入索引（HNSW/IVF‑PQ），只读。  
- **题内库**：EA 注册新子句即刻编码与追加，维持 `text→vector` LRU 缓存。  
- **EA 接入**：在 `scores_req` 前调用 SELECT → 返回 Top‑K（如 512/1024）候选进入精排。

### 3.3 Cross‑Encoder（Transformer）RERANK 训练
- **输入**：拼接 `(q,d)`；模板：`"[CLS] <Q> {q_text} </Q> <D> {d_text} </D>"`。  
- **目标**：二分类（logit→sigmoid）或 pairwise（BPR/hinge）或 listwise（小模型可用 16–32 窗口）。  
- **超参**：6 层 Transformer，hidden 512，max length 256；AdamW 5e‑5；基于 **AUC / NDCG@K** 早停。  
- **在线推理**：对 Top‑K 中每个 `(q,d)` 前向一次（可 batch），输出打分；落在 `batch_ranker.py --backend cross_encoder`。

### 3.4 分数融合与调度
- 融合：`S = λ1·S_cross + λ2·S_bi + λ3·S_heur`；验证集网格搜索 λ。  
- 调度：`Top‑M(cross)` + `Oldest M1` + `Lightest M2` + `ε 随机`。

---

## 4) 方案 B：Bi‑Encoder SELECT + 本地大模型（DeepSeekMath‑32B, QLoRA）RERANK

### 4.1 SELECT 同 3.1–3.2

### 4.2 LLM RERANK（DeepSeekMath‑32B）微调
- **数据**：`make_listwise_chunks.py` 产出的 **listwise**（窗口 64，目标为 softmax 分布）。  
- **训练**：QLoRA（rank=16, α=32, lr=1e‑4, bf16）；目标为 **listwise CE/KL**；或先做 JSON‑SFT（输出 `{"scores":[...]}`），在线再 softmax。  
- **推理**：vLLM/TGI/HF‑Transformers 任选；落在 `batch_ranker.py --backend llm_local`；解析失败→回退 Bi‑Encoder/启发式。  
- **融合与调度**：同 3.4，`S = λ1·S_llm + λ2·S_bi + λ3·S_heur`。

---

## 5) 统一评测与对比指标

### 5.1 离线（per problem 聚合）
- **SELECT 指标**：Recall@{10,32,64}、MAP；**延迟**（向量化 + 检索）。  
- **RERANK 指标**：NDCG@K、MAP、AUC、Pairwise Acc、Kendall’s τ；**每 64 条 batch 的 token/latency**。  
- **端到端模拟**：在保存的候选集上模拟“SELECT→RERANK→调度”的前 N 步，统计 Top‑1~K 的正例命中率与累计收益。

### 5.2 在线（固定题集）
- **主指标**：已证题数（within T 秒）、平均 given 数、平均耗时。  
- **辅指标**：SELECT 命中率（真·证明子句被召回的概率）、RERANK 提升（Top‑M 中正例比例）。  
- **效率**：GPU/CPU 利用率、VRAM、吞吐量（子句/秒）。  
- **统计**：bootstrap 95% CI；记录 per‑problem 报告 JSON 便于复现。

---

## 6) 工程落地：改动点、CLI、服务化、资源预估

### 6.1 代码改动清单（最小化）
1) **`iplog_to_dataset.py`**  
   - 新增 `--add-conjecture` 输出 `conjecture_text/或sig`；  
   - 统一桶优先级：`given_nonproof > simplified > passive_only > never_seen`；  
   - 输出 `sample_weight`（按桶分配）。
2) **`run_batch_pipeline.py`**  
   - 参数化负例配额与桶目标占比，输出 `dataset_report.json`；  
   - 生成 `problem_splits.json`（train/val/test 的题名集合）。
3) **新增脚本**  
   - `make_splits.py / train_sentencepiece.py / train_biencoder.py / encode_and_build_index.py / select_service.py / make_listwise_chunks.py / train_cross_encoder.py / train_llm_lora.py / eval_rank_metrics.py`（模板可直接拷贝使用）。
4) **`process_iprover_v3.py`**  
   - 在 `_ea_handle_scores_req` 前插入 SELECT；  
   - 新增 `--pipeline`：`"A_transformer_ce"`（方案A）/`"B_llm_rerank"`（方案B）；  
   - 记录 Top‑K 召回率、最终调度明细。
5) **`batch_ranker.py`**  
   - 扩展 `--backend`：`cross_encoder | llm_local | llm_api`；  
   - `--blend "llm=0.6,bi=0.3,heur=0.1"`；**解析失败统一回退**。

### 6.2 CLI 串联示例
```bash
# 数据准备
python3 iplog_to_dataset.py --raw-dir Logs/ --out datasets/clauses.jsonl --add-conjecture
python3 make_splits.py --input datasets/clauses.jsonl --failed datasets/failed.jsonl \
  --train-out datasets/train.jsonl --val-out datasets/val.jsonl --test-out datasets/test.jsonl

# 训练 Bi‑Encoder（SELECT）
python3 train_biencoder.py --train datasets/train.jsonl --val datasets/val.jsonl \
  --spm models/sp/spm.model --epochs 6 --batch 256 --neg-per-pos 4 --hard-negative

# 建库与服务
python3 encode_and_build_index.py --model models/biencoder/best.pt --spm models/sp/spm.model \
  --docs datasets/train.jsonl --out indexes/global.faiss
python3 select_service.py --index indexes/global.faiss --model models/biencoder/best.pt --spm models/sp/spm.model

# 方案 A：Cross‑Encoder 精排
python3 train_cross_encoder.py --train datasets/train.jsonl --val datasets/val.jsonl \
  --spm models/sp/spm.model --epochs 3 --maxlen 256
python3 process_iprover_v3.py serve --pipeline A_transformer_ce --select-top-k 1024 \
  --select-service http://127.0.0.1:9000 --ranker-backend cross_encoder --blend "llm=0.0,bi=0.3,heur=0.7"

# 方案 B：LLM 精排（DeepSeekMath‑32B, QLoRA）
python3 make_listwise_chunks.py --input datasets/train.jsonl --out datasets/train_listwise.jsonl --window 64
python3 train_llm_lora.py --data datasets/train_listwise.jsonl --base deepseek-math-32b --out models/ds32b-lora/
python3 process_iprover_v3.py serve --pipeline B_llm_rerank --select-top-k 1024 \
  --select-service http://127.0.0.1:9000 --ranker-backend llm_local --blend "llm=0.6,bi=0.3,heur=0.1"
```

### 6.3 资源预估
- **Bi‑Encoder 训练**：单卡 24GB 可跑（batch 256 需梯度累积），6–8 层/hidden 512。  
- **Cross‑Encoder 训练**：单卡 24GB；在线 RERANK：Top‑K=512 → 前向约 512 次/批（可批处理）。  
- **LLM QLoRA**：40–48GB 显存（32B 基座，QLoRA）；推理可用 vLLM 张量并行。

---

## 7) 风险与回退、里程碑与验收

- **风险**：LLM 解析失败/耗时；SELECT 召回不足；数据分布漂移（某域被动桶偏多）。  
- **回退**：RERANK 失败时 `S=S_bi`；SELECT 服务异常时直送 RERANK（K 取小）；始终保留“最老/最轻/随机”。  
- **里程碑**：  
  - **里程碑1（P0）**：SELECT 上线 + 方案A 离线优于启发式 15% 的 R@64；  
  - **里程碑2（P1）**：方案B 在线 #已证题数 提升 ≥10%（同 timeout）；  
  - **里程碑3（P1）**：蒸馏 + 难负后，方案A 的 SELECT 召回进一步提升 ≥5%。  
- **验收**：提交 `metrics_offline.json` 与在线 A/B 报表（含置信区间）。

---

> 本白皮书对应的最小代码改动点均已列出，并提供了可执行的命令串。你可以先实现 **方案 A**（无需大模型服务，工程阻力小），并并行准备 **方案 B** 的 QLoRA 数据与脚本，两周内即可拉起对比评测。

---

## 已完成（方案 A 对齐进度与下一步）

### 概览
- 已完成：§3.1 Bi‑Encoder 训练、§3.2 索引构建（离线）、§3.3 Cross‑Encoder 训练与离线精排/评测。
- 待完成：§3.2 在线接入（EA 串联）、§3.4 分数融合与调度，以及题内增量库与 LRU 缓存。

### 3.1 Bi‑Encoder（SELECT）训练
- Done
  - 双塔 Transformer，采用 SupCon（按 problem 分组正例）对比学习；AMP + AdamW + warmup+cosine。
  - 验证集按 problem 的 Recall@K（K=32/64）评测；R@64 ≈ 0.926（稳定）。
  - 导出 best.pt 与 SentencePiece spm.model。
- Partial
  - 周期性难负挖掘尚未并入训练循环（当前仅 in‑batch 负与自然难例）。
- Note
  - SupCon 与 InfoNCE 为同类对比学习目标，可继续沿用或在此基础上加入难负。

### 3.2 FAISS 建库与在线 SELECT
- Done
  - 已离线编码并构建索引（Flat/IP，向量 L2 归一化≈余弦）；产出 .npz/.faiss 与丰富 .meta.jsonl。
  - select_service.py 可加载索引与模型进行 Top‑K；batch_select.py 已用于候选生成。
- Partial
  - 题内库增量 add 与 text→vector LRU 缓存未实现。
  - EA 接入：process_iprover_v3.py 尚未在 scores_req 前串联 SELECT 调用。
- 计划差异
  - 白皮书建议 HNSW/IVF‑PQ 用于大库加速；当前为 Flat，后续可替换评估。

### 3.3 Cross‑Encoder（Transformer）RERANK 训练
- Done
  - 数据：按题切分并做文档级排他，构建 train/val/test（无泄漏）。
  - 训练：6 层 / hidden 512，BCEWithLogits；支持 init‑from 与 pos_weight；val AUC ≈ 0.988–0.989。
  - 工具：rerank_with_cross_encoder.py 离线精排；eval_retrieval.py 计算 Hit/Recall/NDCG。
- Partial
  - 在线：batch_ranker.py 的 cross_encoder 后端与 EA 串联尚未落地。
  - Pairwise/Listwise 目标暂未启用（当前主用二分类）。

### 3.4 分数融合与调度
- Todo
  - 实现 S = λ1·S_cross + λ2·S_bi + λ3·S_heur 的网格搜索与在线融合。
  - 调度：Top‑M(cross) + Oldest M1 + Lightest M2 + ε 随机；失败回退与日志保留。

### 离线端到端评测（无泄漏 test，小样本）
- 设置：Bi‑Encoder 检索 K=200 → Cross‑Encoder 精排；n_queries = 27。
- 指标：
  - Bi Hit@K：@64 ≈ 0.518，@200 ≈ 0.630；CE Hit@K：@64 ≈ 0.556，@200 ≈ 0.630。
  - Bi Recall@K：@64 ≈ 0.0626，@200 ≈ 0.1778；CE Recall@K：@64 ≈ 0.1097，@200 ≈ 0.1778。
  - CE 在小/中 K 明显提升 NDCG 与 Hit；K=200 与 Bi 持平（受候选上限影响）。
- 诊断：
  - 索引覆盖 ≈ 82%；单 query 平均正例 ≈ 180。
  - K 限制的理论上限：@10≈0.056、@32≈0.178、@64≈0.356、@100≈0.556、@200≈1.0；当前结果符合预期。

### 下一步（优先级）
1) 在线打通：process_iprover_v3.py 在 scores_req 前调用 SELECT → 交给 batch_ranker.py(cross_encoder) → 回包；保留启发式回退与详尽日志。
2) 召回与稳定性：将候选 K 提至 500/1000；扩大 test 题集；同步报告 reachable recall（扣除未覆盖正例）。
3) 融合与调度：离线网格搜索 λ1/λ2/λ3（目标 NDCG@64/100）；在 EA 端实现融合与 Top‑M + Oldest/Lightest/ε 调度策略。
4) 难负与索引：加入周期性难负挖掘微调 Bi；评估并切换 HNSW/IVF‑PQ；实现题内增量库与 text→vector LRU 缓存。
