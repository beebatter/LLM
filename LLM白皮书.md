LLM项目计划白皮书
1) 项目的整体概述
这是一个“iProver 交互模式 + 外部智能代理（EA）”的端到端体系，目标是两件事：
•	数据侧：从 TPTP FOF 题库高质量、可追溯地构建“子句选择”训练语料（证明用到的子句=正例，没用到的=负例，细分负例桶），并支撑后续对比学习/监督学习与离线评测。你的 run_batch_pipeline.py 就是这个流程的编排器（拉起 EA→启动 iProver→收集原始交互日志→解析并采样→写入 JSONL 数据集）；负例桶与标签的生成逻辑在 iplog_to_dataset.py 里定义并实现（含对 SZS CNFRefutation 的解析）。
•	推理侧：把模型打分能力（当前以 LLM 精排为主）接入 iProver 的“被动队列/给定子句”调度，形成选择—精排—调度闭环。你现在的 EA/打分器由 process_iprover_v3.py（日志清洗、符号规范化、候选集与上下文组织、可选前置筛选）和 batch_ranker.py（分块召回、背景摘要、锚点对齐、归一化与容错）共同完成；iProver 的交互协议与“注册子句/请求打分/回传分数”的工作方式在官方交互模式说明里有清晰定义，你的实现与其一致。
核心理念与演进
短期用 LLM 做 Cross-Encoder 式精排（你已有）；中期上线 Bi-Encoder/索引器式 Transformer 做 SELECT（向量检索 Top-K）+ LLM 精排的 SELECT-RERANK 框架；长期加入蒸馏/难负挖掘与结构偏置，形成稳定、可扩展的神经引导子句选择系统。你的 README 与脚本注释也在往这个形态靠拢（“外部智能代理”“被动分数/给定子句模式”“日志—数据集”链路已经打通）。
2) 现有功能列表 & 未来功能清单
2.1 现有功能（已可用）
•	批处理数据管线（dataset builder）
o	统一编排：逐题拉起 EA 与 iProver、设定超时、收集原始交互日志、失败记录与原因、并把样本累积写入 JSONL（含负例采样参数如 --neg-mult/--neg-cap-per-problem/--frontier-window）。
o	失败日志增强：记录最近的 given_clause，并已扩展为从 register_clauses 反查子句原文与特征（_build_clause_index / extract_last_given_details），便于跨会话诊断（ID 不稳定时仍可定位）。
•	原始日志 → 标注样本（正/负/负例桶）
o	解析 NUL/换行混合分隔的 JSON 消息流；
o	从 SZS 证明片段抽取出现过的 c_# 子句 ID 作为正例；
o	其它按照“被动仅驻留/给定但非证明/被简化/未出注册”等分到负例桶；输出统一样本架构（text/features/label/neg_bucket/...）。
•	EA 侧清洗与规范化
o	基于 process_iprover_v3.py 的条目级清洗（去除 file(...)/inference(...) 尾注，抽出 tcf(...,<formula>) 的公式体）、变量与符号规范化（变量重命名、全题内稳定的谓词/函数/常量编号）、上下文/候选集选择，以及**（可开关）前置筛选与低保底分**（PREFILTER_ENABLED / LOW_FLOOR_SCORE）。
•	LLM 批量精排器（可 dry-run）
o	将候选子句按 64/128 分块，先生成一份背景摘要，然后对每块请求模型打分，最后做锚点对齐与归一化，并对 JSON 解析失败/空响应等进行容错与回退；支持 --dry-run 验证全链路不依赖真实 API（默认模型注释为 “GPT-5 Thinking”）。
o	语义小助手：从猜想里抽“目标函子/关键常量/参数格局”，给候选打结构标签（是否触达目标函子、是否等式等），用于启发式/提示词增强或后续蒸馏特征。
•	文档与使用指南
o	iProver/交互模式说明与能力清单（含构建/运行范式）；
o	你的 LLM-EA 搭建 README（问题源抽取、三种策略、输出结构示例）为团队成员提供一键起步的操作口径。
2.2 计划中的功能（建议优先级与落地点）
P0（直接提升效果/稳定性的关键项）
•	Bi-Encoder “索引器式 Transformer” 训练与接入：基于现有 JSONL 语料训练 对比学习 双塔编码器（q=猜想，d=子句），上线 FAISS 检索作为 SELECT；在 EA 内把 LLM 精排前的候选从万级压到 Top-K（512/1024），并保留“老的/更轻的/随机”的混合调度通道。
•	多源分数融合：S = λ1·S_LLM + λ2·S_bi + λ3·S_heur，离线验证集网格搜索 λ；在线写入 per-problem metrics.json。
•	稳健性与可观测性：LLM JSON 解析失败的回退路径统一、日志统一埋点（召回率、R@K、已证题数/平均 given/平均时长）。
P1（性价比高的模型质量项）
•	难负挖掘：在同题内用当前 Bi-Encoder 得分找“最易混的负例”加入训练；
•	蒸馏：把 LLM 的 listwise 分布蒸馏到 Bi-Encoder（KL + InfoNCE），对召回提升显著；
•	结构偏置：加入“符号类型/项深度/量词作用域/参数次序”等额外嵌入，强化逻辑感（位置编码之外的 inductive bias）。
P2（工程与产品化）
•	本地大模型精排（如 DeepSeekMath-32B/本地推理）：提供 LoRA/QLoRA 微调脚本与 vLLM/llama.cpp 推理封装；支持 listwise softmax 与 pairwise 两种学习范式，作为在线 RERANK 的可选后端。
•	离线评测工具箱：按题分组的 Recall@{10,32,64}/MAP、A/B 报表模板与可视化；
•	数据治理：跨题重复去重、分布均衡、训练/验证按题切分的自动校验；
•	环境健壮性：TPTP_DIR/axioms 校验、端口/进程清理、失败自动重试与极限兜底。
3) 技术栈与落地步骤
3.1 技术栈选择（核心组件）
•	定理证明与交互
o	iProver v3.x（交互模式）
o	你的 EA（外部智能代理）：process_iprover_v3.py 作为消息编排/清洗与日志核心
•	数据构建与分析
o	Python 3.11、JSONL、pandas（可选）
o	数据构建：run_batch_pipeline.py（批跑多题产出数据集）
o	日志转样本：iplog_to_dataset.py
•	表示学习与检索
o	PyTorch（Transformer 训练）
o	SentencePiece（自定义子词表）
o	FAISS（向量检索，HNSW/IVF-PQ）
o	HuggingFace Transformers（双塔/交叉编码器）
o	peft / bitsandbytes / accelerate（LoRA/QLoRA）
•	LLM 精排（RERANK）
o	本地大模型（例如 DeepSeekMath-32B），LoRA 微调
o	推理服务：vLLM / TGI / 纯 HF generate（按资源选择）
•	可观测性与运维
o	统一日志目录（EA 日志、数据集、评测报告）
o	端口进程清理与 TPTP 环境检查（脚本内置）
o	评测脚本（离线 Recall@K/MAP、在线已证题数/步数/时长）
3.2 落地步骤（从零到全链路）
1.	数据就绪
o	用 run_batch_pipeline.py 跑一批 TPTP 题，得到 *.raw.log 与 datasets/*.jsonl（你已完成基础版）。
o	在 iplog_to_dataset.py 中确保样本含：problem_name / text / features / label / neg_bucket，并附带 conjecture/符号签名（为双塔 SELECT 做准备）。
2.	预处理与分词器
o	训练 SentencePiece（16k–32k），保留符号 ()=,~|&，变量归一化。
o	把结构特征离散化为前缀 token：<H0/1><U0/1><E0/1><C*><B*>。
3.	训练 SELECT（Bi-Encoder 双塔）
o	q=conjecture，d=clause；InfoNCE + 难负挖掘（优先 NEG_given_nonproof / NEG_simplified）。
o	导出 best.pt + tokenizer。
4.	建立向量索引
o	FAISS 索引（GPU/CPU 任选），两类库：
	静态全库：常见公理/子句离线编码；
	题内动态库：EA 收到新子句即时编码追加（LRU 缓存文本→向量）。
5.	接入 EA（SELECT→RERANK→调度）
o	在 process_iprover_v3.py 的 scores_req 处理链路前插入 SELECT：
	用双塔编码 q，检索 Top-K 子句；
	只把 Top-K 分块交给 batch_ranker.py（LLM 精排）。
o	调度：LLM 高分 + “最老” + “最轻” + 少量随机探索 → 给 iProver。
o	融合：S = λ1·S_LLM + λ2·S_bi + λ3·S_heur，验证集网格搜索 λ。
6.	训练 RERANK（本地大模型 LoRA）
o	用 listwise 样本（按题/按块，softmax 目标分布）微调 DeepSeekMath-32B（QLoRA）。
o	推理服务：vLLM/TGI；batch_ranker.py 增设后端 --backend llm_local。
7.	评测与 A/B
o	离线：题内 Recall@{10,32,64} / MAP；
o	在线：固定题集，统计已证题数/平均 given/平均时间；
o	方案对比：启发式 / 单塔先验 / Bi-Encoder / Bi+LLM / 蒸馏 Bi+LLM。
 
4) 功能模块与交互关系
┌──────────────────────────────────────────────────────────────────────────────┐
│                         数据侧（构建与评测）                                 │
│  run_batch_pipeline.py → *.raw.log → iplog_to_dataset.py → *.jsonl → splits │
│         ↑                 ↑                           ↑                     │
│     失败日志           证明片段抽取                负例分桶/特征             │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         训练侧（选择与精排）                                  │
│  Bi-Encoder 训练（q,d,InfoNCE+HN）→ best.pt                                   │
│  LLM LoRA（listwise softmax）→ adapter.safetensors                             │
└──────────────────────────────────────────────────────────────────────────────┘

┌──────────────────────────────────────────────────────────────────────────────┐
│                         在线推理（EA + iProver）                              │
│ iProver ⇄ process_iprover_v3.py (EA)                                          │
│                │                                                              │
│         [SELECT] Bi-Encoder + FAISS（Top-K）                                  │
│                │                                                              │
│         [RERANK] batch_ranker.py（LLM，本地或云）                             │
│                │                                                              │
│         [SCHEDULE] 混合优先队列（LLM高分/最老/最轻/随机）                     │
│                │                                                              │
│             回传 scores_res / 子句优先序                                      │
└──────────────────────────────────────────────────────────────────────────────┘
•	数据流
o	iProver 产生日志 → EA 记录 NUL 分隔 JSON → iplog_to_dataset.py 解析 SZS 证明 → 生成样本（正/负）
o	训练：样本 →（分词/特征 token）→ 双塔训练 & LLM LoRA
o	在线：scores_req 到达 → SELECT（Top-K）→ RERANK（分批打分）→ 混合调度 → scores_res
•	接口点（消息标签）
o	register_clauses / given_clause / scores_req / scores_res
o	在 scores_req 之前做 SELECT；RERANK 解析失败时回退启发式/双塔分数
•	缓存与索引
o	子句文本哈希 → 向量/分数缓存（会话内复用，跨会话失效不影响）
o	FAISS 动态添加/移除题内子句；静态全库只读
 
5) 现有代码的详细分步实现计划
目标：最小改动接入 SELECT-RERANK，逐步增强到蒸馏与本地 LLM 精排。
5.1 数据与预处理（P0）
•	iplog_to_dataset.py
1.	输出字段中增加：conjecture_text（或 conjecture_sig 目标符号签名）
2.	结构特征完善：unit/epr/horn/conj_dist/born，必要时补 contains_target_functor/const_eq/arity 等
3.	生成 listwise chunk 工具（新脚本 make_listwise_chunks.py）：
	按题、按窗口（32/64）成块；
	计算 softmax 目标分布（ε=0.1 平滑）；保存 {ids, features, text, target_distribution}
•	SentencePiece 词表（已给示例脚本）
o	词表 16k–32k；user_defined_symbols 加 <H*/U*/E*/C*/B*>
o	变量统一替换为 VAR，防止稀疏
•	切分
o	按 problem_name 分组切 train/val/test（避免泄漏）
o	自动校验：跨题重复子句文本 → 验证/测试中可降权或排除
5.2 SELECT（Bi-Encoder）训练与上线（P0）
•	新建目录：models/biencoder/
o	train_biencoder.py（基于你已有的 train_encoder_contrastive.py 改为双塔）
	输入：(q, d, label)；正对来自证明子句；负对同题其它子句（硬负优先）
	损失：InfoNCE（以 q 为 anchor）；可叠加 BPR/hinge
	超参：6–8 层、512 hidden、τ=0.07、AdamW(lr=1e-4, wd=0.01)
o	encode_and_build_index.py
	加载 best.pt & tokenizer；
	编码静态库（公理/常见子句）；落地 FAISS 索引（HNSW 或 IVF-PQ）
•	EA 接入（process_iprover_v3.py）
o	新增参数：--select-top-k 1024、--select-service http://127.0.0.1:9000（或本地函数）
o	在收到 scores_req：
1.	聚合候选子句文本与特征 → 生成前缀 token → 子词编码；
2.	编码 q=conjecture；
3.	查询 FAISS：返回 Top-K 子句 ID；
4.	仅将 Top-K 传给 batch_ranker.py 继续 RERANK；
5.	混合调度（LLM 高分 M + 最老 M₁ + 最轻 M₂ + 随机 ε）；
o	缓存：{clause_text_hash -> vec/score}；会话内有效
•	向量索引服务（新）
o	select_service.py：封装 index.search()、add_with_ids()、remove_ids() 接口
o	题开始时创建“题内动态库”，题结束时释放；静态库初始化一次
5.3 RERANK（LLM/DeepSeekMath-32B）训练与上线（P1）
•	数据：使用 make_listwise_chunks.py 产物
•	训练：train_llm_lora.py
o	模型：DeepSeekMath-32B（或等价开源数学 LLM）；QLoRA(rank 16, α 32, lr 1e-4)
o	目标：listwise softmax JSON 输出（或直接数值头；简化先 SFT JSON）
o	解析容错：缺失 ID→补 0、再二次 softmax
•	推理接入（batch_ranker.py）
o	新增后端 --backend llm_local（vLLM/TGI/HF generate）
o	解析 JSON → 对齐 ID → 二次 softmax → 回传 scores_res
o	失败回退：用 Bi-Encoder 分数 / 启发式融合
5.4 融合与调度（P0）
•	融合：在 batch_ranker.py 或 EA 汇总处实现
o	S = λ1·S_LLM + λ2·S_bi + λ3·S_heur，可使用 --blend "llm=0.6,bi=0.3,heur=0.1"
o	验证集网格搜索 λ：选择使 Recall@64 或在线“已证题数”最优
•	调度（process_iprover_v3.py）
o	参数化：--sched-top-m 50 --sched-oldest 8 --sched-lightest 8 --sched-random-p 0.02
o	拆分来源：Top-M(LLM)，M₁(年龄)，M₂(轻量)，ε(随机)
o	可记录最终交付给 iProver 的子句队列，以便回放
5.5 蒸馏与难负（P1）
•	蒸馏：distill_biencoder_from_llm.py
o	用 LLM 的 listwise 软标签（每块分布）对 Bi-Encoder 做 KL；同时保留 InfoNCE
o	目标：提高 SELECT 的召回，使其更贴近 RERANK 偏好
•	难负挖掘：训练过程中周期性用 Bi-Encoder 打分，采 “高分负例” 追加到负池
5.6 评测与 A/B（P0）
•	离线：eval_rank_metrics.py
o	输入：(problem_name, clause_id/text, label, score_bi/score_llm/score_blend)
o	输出：每题 Recall@K/MAP、宏平均、CSV/JSON 报告
•	在线：EA 记录
o	每题：已证（0/1）、总 given、总用时、proof_out 是否出现
o	汇总：A/B 报表与统计脚本
5.7 运维与健壮性（P0）
•	环境检查：启动时校验 TPTP_DIR/Axioms、iProver 路径、端口可用性
•	故障回退：LLM 超时/解析失败→Bi/启发式；FAISS 查询失败→直接 LLM 精排；总有兜底
•	日志：统一 Logs/EA.<pid>.<ts>/ 结构；提供 log_inspector.py 一键查问题
5.8 里程碑（建议 3 个迭代）
•	迭代 1（P0）：双塔 SELECT + FAISS 接入 + 调度 + 融合 + 离线/在线评测
交付：train_biencoder.py / select_service.py / EA 接入 patch / eval_rank_metrics.py
•	迭代 2（P1）：LLM LoRA RERANK（DeepSeek 本地）+ listwise 数据 + 解析容错
交付：make_listwise_chunks.py / train_llm_lora.py / batch_ranker.py 后端适配
•	迭代 3（P1）：蒸馏 + 难负 + 结构偏置嵌入 + 报表与可视化
交付：distill_biencoder_from_llm.py / hard_negative_miner.py / dash

