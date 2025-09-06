# 方案 B（LLM 重排）微调训练清单与增强路线图

> 目标：让大模型在收到“猜想 + K 个候选子句”后，稳定输出严格 JSON：`{"scores":[...K...]}`，作为在线重排器（RERANK），并与 Bi‑Encoder/启发式融合，提升在线已证题数、降低平均 given/延迟可控。

本清单覆盖最小可行闭环（数据→QLoRA SFT→本地服务→离线评测→在线接入）以及增强路线（蒸馏、KL、课程学习、自举、部署优化、观测与回退）。

---

## 0. 仓库内关键脚本（已就绪）
- 数据构造（listwise）：`LLM/scripts/make_listwise_chunks.py`
- QLoRA 训练（SFT）：`LLM/training/train_llm_lora.py`
- 简易推理服务（HTTP）：`LLM/select/ranker_service.py`
- LLM 重排器（离线/可接在线）：`LLM/batch_ranker.py`（支持本地端点）
- Tokenizer/特征前缀：`LLM/data_utils/logic_tokenizer.py`

依赖（已列出）：`/root/LLM/requirements-transformer.txt`

---

## 1. 最小闭环（可直接执行）

### 1.1 环境
```bash
pip install -r /root/LLM/requirements-transformer.txt
```

GPU 建议：≥24GB 显存（QLoRA + 梯度累积可更省）。

### 1.2 数据构造（listwise K=64）
输入 JSONL 每行至少含：
- problem_name；
- conjecture_text|conjecture|query|text_a（任选其一）；
- text|clause|text_b|premise（任选其一）；
- label（1/0）；可选 features|meta、neg_bucket。

生成训练/验证窗口：
```bash
# 训练集
python -m LLM.scripts.make_listwise_chunks \
  --input /root/Training/datasets/train.jsonl \
  --out   /root/Training/datasets/train_listwise.jsonl \
  --window 64

# 验证集（如有）
python -m LLM.scripts.make_listwise_chunks \
  --input /root/Training/datasets/val.jsonl \
  --out   /root/Training/datasets/val_listwise.jsonl \
  --window 64
```
说明：
- 至少 1 个正例；若全负 → 目标分布均匀；
- 负例采样按 neg_bucket 加权：`given_nonproof=1.0 > simplified=0.8 > passive_only=0.5 > 其它=0.25`；
- features→`<H?><U?><E?><C?><B?>` 前缀，拼到 TAGS。

验收：`train_listwise.jsonl` 每行含 `input`、`ids`、`K`、`target_json`、`target_scores`。

### 1.3 QLoRA SFT（稳定 JSON 输出）
推荐基座：`deepseek-ai/deepseek-math-7b-instruct`。
```bash
python -m LLM.training.train_llm_lora \
  --model deepseek-ai/deepseek-math-7b-instruct \
  --train /root/Training/datasets/train_listwise.jsonl \
  --val   /root/Training/datasets/val_listwise.jsonl \
  --out   /root/Training/models/ds7b-instruct-lora \
  --epochs 1 --batch 1 --grad-accum 16 --lr 1e-4 --bf16

# 若网络无法访问 Hugging Face，可切换离线/本地缓存模式：
# 1) 先将模型权重下载到本地目录（/path/to/local_model），或确保已缓存；
# 2) 再加 --local-only 运行；必要时 export 环境变量 HF_HUB_OFFLINE=1。
# 示例：
# HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 \
# python -m LLM.training.train_llm_lora \
#   --model /path/to/local_model \
#   --train /root/Training/datasets/train_listwise.jsonl \
#   --val   /root/Training/datasets/val_listwise.jsonl \
#   --out   /root/Training/models/ds7b-instruct-lora \
#   --epochs 1 --batch 1 --grad-accum 16 --lr 1e-4 --bf16 --local-only
```
提示：这是“格式与任务对齐”的 SFT；先保格式稳定，再考虑进一步精度优化。

小贴士（下载/缓存常见报错与处理）：
- Connection reset by peer（transfer.xethub.hf.co / xet-core）
  - 原因：启用了 hf_transfer（Xet 加速），网络抖动/防火墙导致连接被对端重置；会反复重试但不稳定。
  - 处理：禁用加速，改走普通 HTTP 下载，并将并发降到 1。
    - 临时：`export HF_HUB_ENABLE_HF_TRANSFER=0`；或卸载：`pip uninstall -y hf_transfer`。
    - CLI（示例）：`hf download deepseek-ai/deepseek-math-7b-instruct --local-dir /autodl-tmp/models/deepseek-math-7b-instruct --max-workers 1`。
- No space left on device（os error 28）
  - 原因：大权重分片（~10GB）先写入临时目录再移动，TMP 或缓存分区空间不足；并发下载会放大量临时块文件。
  - 处理：把缓存与 TMP 指到大盘，并清理残留 .incomplete。
    - 环境变量：
      - `export HF_HOME=/autodl-tmp/hf`
      - `export HF_HUB_CACHE=/autodl-tmp/hf/cache`
      - `export TRANSFORMERS_CACHE=/autodl-tmp/hf/transformers`
      - `export TMPDIR=/autodl-tmp/tmp`
      - 创建目录：`mkdir -p $HF_HOME $HF_HUB_CACHE $TRANSFORMERS_CACHE $TMPDIR`
    - 清理残留：
      - `find "$HF_HOME" -name "*.incomplete" -delete || true`
      - 如需重来：`rm -rf "$HF_HOME"/hub/models--deepseek-ai--deepseek-math-7b-instruct`
    - 重新下载（单线程）：
      - `HF_HUB_ENABLE_HF_TRANSFER=0 hf download deepseek-ai/deepseek-math-7b-instruct --local-dir /autodl-tmp/models/deepseek-math-7b-instruct --max-workers 1`
    - 训练时使用本地目录并加 `--local-only`：`--model /autodl-tmp/models/deepseek-math-7b-instruct --local-only`

### 1.4 本地推理服务（HTTP）
```bash
python -m LLM.select.ranker_service \
  --model deepseek-ai/deepseek-math-7b-instruct \
  --lora  /root/Training/models/ds7b-instruct-lora \
  --port  8001
```
支持任意 CausalLM；如需高性能可替换为 vLLM/TGI（保证接口：POST `{"prompt":...}`→返回`{"text":...}`）。

### 1.5 离线评测与融合
```bash
export LLM_LOCAL_ENDPOINT=http://127.0.0.1:8001/generate

python -m LLM.batch_ranker \
  --input  /root/Training/datasets/sample_scores_req.json \
  --out    /root/Training/logs/rank_llm.json \
  --chunk-size 64 --progress \
  --lambda-cross 1.0 --lambda-bi 0.3 --lambda-heur 0.1
```
说明：
- `batch_ranker` 会将每个 chunk 的 prompt 发送到 `LLM_LOCAL_ENDPOINT`，解析 `{"scores":[...]}`，失败/空则回退为 0 并融合；
- 最终分数 min-max 归一后输出。

### 1.6 在线接入（EA / process_iprover_v3.py）
- 在 `_ea_handle_scores_req` 前：生成 chunk → 调用 `batch_ranker`（或其内部函数）→ 回包；
- 失败/超时（>800–1200ms）→ 回退 Bi/启发式；
- 融合权重 λ 在验证集网格搜索后写入配置。

---

## 2. 参数建议与注意事项
- 窗口大小：`K=64/128`，先用 64；
- 上下文长度：控制在 4k–8k tokens 内，必要时对子句做安全截断；
- SFT 超参：QLoRA(NF4, r=16/32, α=32)，`lr=1e-4`，`warmup=0.03`，`epochs=1–2`；
- 输出约束：指令中硬性要求“只输出 JSON”，温度 0 或极低；
- 解析健壮性：`batch_ranker` 已做容错/抖动；保持严格 JSON 可显著降低回退率。

---

## 3. 质量门与验收
- 训练：1 epoch 正常收敛并保存 LoRA；
- 离线：
  - 解析成功率≈100%；
  - NDCG@64 / Recall@K 优于纯 Bi‑Encoder；
  - 延迟 P50/P95 在预算内；
- 在线：固定题集与时限，`#已证题数 ↑`、`平均 given ↓/≈`，资源占用可控（记录 GPU/CPU 时间、吞吐）。

---

## 4. 增强路线（逐步开启）

### 4.1 教师‑学生蒸馏（建议优先级：高）
思路：用更强的“教师”离线产生软分布，再用 KL 蒸馏到在线小模型（7B/13B）。
- 教师候选：
  - 强 LLM：`DeepSeek-R1-Distill-Qwen-32B`（或你的最强本地 LLM）；
  - Cross‑Encoder：你现有 CE（AUC 高），离线打分填入 `target_scores`；
- 数据：用 `make_listwise_chunks` 生成窗口后，将 `target_scores` 替换为教师分布；
- 训练：继续用 `train_llm_lora.py` 做 SFT（先不改损失），已能学到排序相对关系；后续切换到 KL 混合损失（见 4.2）。

### 4.2 数值 KL/CE 混合损失（建议优先级：中）
目的：让模型生成的 scores 的 softmax 分布更贴近目标分布。
- 实现路线：在 SFT 时将目标 JSON 强制对齐，同时在训练 loop 中，解析模型输出分数（或在 logits 空间近似），与 `target_scores` 做 `KLDivLoss`（或 CE）；
- 混合权重：`L = NLL(JSON) + λ·KL(scores||target)`，`λ` 从 0.5～1.0 网格搜；
- 注意：需要轻量定制 Trainer 的 `compute_loss`，脚本后续可添加 `--use-kl` 选项。

### 4.3 课程学习与专家迭代（高性价比）
- 课程学习：从中等难度子库开始（按家族/符号规模/深度分层），逐步放开更难域；
- 专家迭代：上线后周期性采数，将新成功证明的窗口与标签回灌到训练集滚动 SFT/蒸馏；
- 指标：每轮报告 Recall@K / NDCG@K、在线已证题数增量与置信区间。

### 4.4 自举与难负挖掘
- 自举：用当前最优 A/B 方案跑更大题单，吸入新增正例；
- 难负：从 `failed.jsonl` 抽“前沿弱负”与“差一点命中”的窗口，以低权重加入训练（已在 `make_listwise_chunks` 里通过 neg_bucket 权重体现）。

### 4.5 高性能部署（vLLM/TGI）
- 目标：降低 P95 延迟、提升吞吐与稳定性；
- 要点：
  - 固定 `temperature=0`、`max_new_tokens=256–512`；
  - 合理设置 `stop`（如首个 `\n{`/`\n[` 等）减少跑偏；
  - KV Cache / 张量并行 / 静态批次；
  - 观测：QPS、延迟直方图、OOM/超时率。

### 4.6 解析稳健性与 Schema 约束
- 在 prompt 中再次强调“只输出 JSON”；
- 在线解析失败：固定回退（Bi/启发式），并记录失败示例；
- 可选：服务端做 JSON Schema 校验与重试 1 次（超时/格式错不重试）。

### 4.7 预算与回退策略
- 超时阈值：`rerank_ms ∈ [800, 1200]`；
- 缓存：热点窗口与摘要缓存、text→embedding 的 LRU；
- 回退序：LLM→Bi→启发式；
- 调度：`Top‑M + Oldest + Lightest + ε` 混合，防止局部最优。

### 4.8 融合 λ 与 K 曲线
- 网格搜索：`λ_llm, λ_bi, λ_heur` 在 val 上按 NDCG@64/100 搜；
- K 对比：64 vs 128 的收益/延迟曲线；
- 写回在线 YAML。

### 4.9 观测与日志
- 每 request JSON 摘要：K、各阶段耗时、解析失败标志、Top‑N 预览；
- 可接 Prometheus：阶段耗时直方图、失败率、GPU/CPU 利用率采样；
- 关键工件落盘（prompt/response/分数 等），便于复现实验。

---

## 5. 常见问题（FAQ）
**Q：没有 `conjecture_text` 怎么办？**
先用 `interactive_sampled_small.jsonl` 打通流程；并升级 `iplog_to_dataset.py` 输出 `conjecture_text`（白皮书 §2.2 已建议），再换更大集重跑。

**Q：7B‑RL 能用吗？**
不建议直用（话痨倾向）。优先 `7B‑instruct`；要更强效果，用 32B 作为离线教师蒸馏到 7B 学生。

**Q：输出不是严格 JSON？**
提高“只输出 JSON”的系统指令占比；温度 0；训练集中混入 5–10% “格式纠错”样本；服务端加一次 schema 校验与重试。

---

## 6. 快速验收清单（勾选即过线）
- [ ] `train_listwise.jsonl / val_listwise.jsonl` 生成成功；
- [ ] QLoRA 1 epoch 训练完成，保存 LoRA 目录；
- [ ] 本地服务 `ranker_service.py` 能返回文本；`batch_ranker` 端到端生成 `scores`；
- [ ] 离线 NDCG@64/Recall@K 提升，解析失败≈0，延迟在预算内；
- [ ] 在线 A/B：`#已证题数` 提升或等同但延迟可控；
- [ ] 融合 λ/K 在 val 上网格搜索并固化到配置；
- [ ] 日志/观测齐全，可复现实验。

---

## 7. 附录：命令速查
```bash
# 数据
python -m LLM.scripts.make_listwise_chunks --input /root/Training/datasets/train.jsonl --out /root/Training/datasets/train_listwise.jsonl --window 64

# 训练（QLoRA SFT）
python -m LLM.training.train_llm_lora --model deepseek-ai/deepseek-math-7b-instruct \
  --train /root/Training/datasets/train_listwise.jsonl \
  --val /root/Training/datasets/val_listwise.jsonl \
  --out  /root/Training/models/ds7b-instruct-lora --epochs 1 --batch 1 --grad-accum 16 --lr 1e-4 --bf16

# 本地服务
python -m LLM.select.ranker_service --model deepseek-ai/deepseek-math-7b-instruct --lora /root/Training/models/ds7b-instruct-lora --port 8001

# 离线评分与融合
export LLM_LOCAL_ENDPOINT=http://127.0.0.1:8001/generate
python -m LLM.batch_ranker --input /root/Training/datasets/sample_scores_req.json --out /root/Training/logs/rank_llm.json --chunk-size 64 --lambda-cross 1.0 --lambda-bi 0.3 --lambda-heur 0.1 --progress
```

---

如需：
- 我可以将 `process_iprover_v3.py` 注入 Pipeline B 调用并加 YAML 配置；
- 增补 `train_llm_lora.py` 的 KL 蒙版与 `--use-kl` 选项；
- 生成 vLLM/TGI 的部署清单与 Prometheus 指标配置示例。

---

## 8. 按 DeepSeek‑Prover 方法落地：从数据到训练（FOL 版）

本节把“自举+合成数据 / 两阶段训练 / 教师‑学生 / 课程学习”的关键做法，落地到你的 FOL 子句筛选场景，给出可执行的实施方法与脚本串联。

### 8.1 数据侧：网站“证明 + 全集 CNF”对齐（大规模正/负）
目的：绕开本地求解率瓶颈，直接用公开的 TPTP/CASC/TSTP 金标生成高质量样本。

步骤：
1) 收集题库与已验证反驳（TSTP 证明）
   - 下载目标家族的 TPTP FOF/ CNF 问题与对应 TSTP 证明（可选官网打包）。
2) 生成“全集 CNF”并规范化
   - 用 `tptp4X` 将 FOF 归约到 CNF（含所有公理与猜想子句）。
   - 规范化：变量归一（`X*→VAR`，本仓 `normalize_text` 已支持）、字面量排序（按谓词/极性/项序），得到 `canonical_formula`。注：排序可在生成脚本中实现，或以“目标符号优先”作为近似。
3) 对齐金标正例
## 9. 训练改进实现计划（基于当前评测现象）

结论解读（现象→原因）
- 7B Pearson≈0.05 但 MAE≈0.03：学到了“平均模板”，对窗口内相对强弱不敏感，排序信号不足。
- 32B Pearson≈0.27：已开始捕捉差异，但相关性不高，仍受评测口径偏置与目标过于平滑影响。
- 主要两类原因：
  1) 打分/评测口径偏置（长度、频率/模板、候选相互干扰）；
  2) 训练目标更像“生成 JSON 的 SFT”，而不是“候选级分布/排序学习”。

### 9.1 立刻可做的评测校准（不改模型即可增益）
- 平均对数似然（mean-LL）替代“总和”以消除长度偏置：s_mean = (1/|c|)·Σ log p(x_t|prompt)。
- 逐候选独立前向，避免候选间注意力泄漏（不要同一前向里切片取每个候选分）。
- 组内归一：对窗口内 {s_i} 做 z-score 或减中位数/除 MAD，再 softmax。
- 温度/偏置标定：在 dev 拟合 p̂=softmax((s−b)/τ)，最小化 KL/CE 找 τ,b。
- 可选 PMI 校准：s_PMI = log P(c|conj) − λ·log P(c)（λ∈[0.5,1.0] 网格搜）。

实现落地（脚本改动最小）：
- `LLM/training/eval_llm_listwise.py`
  - 新增 `--score-type {sum-ll,mean-ll,pmi}`、`--isolate`（逐候选独立前向）、`--zscore`、`--calib-tau`、`--calib-bias`；
  - PMI 需要额外跑一次“无猜想/通用提示”的基线得分（或缓存语料频率模型）。
- `LLM/select/rerank_with_llm.py`
  - 同步支持上述参数，作为离线重排器产出 CE 兼容格式。

验收：Pearson、NDCG 明显上升，len_match_rate=1.0 保持。

### 9.2 训练目标升级（候选级分布学习，替代生成 JSON）
- 在每个候选段落外侧加 special tokens：`<CAND_START> ... <CAND_END>`。
- 前向后在 `<CAND_START>` 位置取隐藏向量 h_i（或在候选范围内做平均池化）。
- 线性打分头：s_i = w^T h_i + b；窗口分布 p̂=softmax(s)。
- 损失：CE/KL 对齐教师分布 p*：L = CE(p*, p̂)；可混合少量 pairwise（margin/BT）作正则。
- LoRA 仅加在打分头或上层 block，减少显存与收敛不稳。

实现落地：
- 新增/扩展训练脚本（建议）：`LLM/training/train_llm_listwise_sft.py` 支持 `--score-head` 模式：
  - 构造含 `<CAND_START>/<CAND_END>` 的输入；
  - 从模型隐层抽取候选向量，接线线性头，计算 listwise CE/KL；
  - 保留已有 SFT 路径作为备选（JSON 输出模式）。

### 9.3 教师分布增强（让正例内部“有强弱”）
- 用 CE 的细粒度分数+温度 τ，给正例再分配权重：
  - pos_weights = (1−α)·uniform + α·softmax(teacher/τ)；
  - 在 dev 上对 (α, τ, 总质量 m) 网格搜，避免“平均模板”。
- 没有高质量 CE 时，可用 TAGS 启发式分（如命中 conjecture 符号/常量≻其它），再配温度与混合权重。

落地：
- 增加 `scripts/target_dist_tuner.py`（或在现有融合脚本加调参流程），输出新的 `target_scores` 并重写训练集（仅替换 scores，不改样本）。

### 9.4 训练细节优化（在你现配方上抛光）
- 学习率：7B LoRA 建议 5e-5～1e-4；32B QLoRA 5e-5；epoch 2–3，带 warmup（0.03）与 cosine。
- 候选随机化：每个 epoch 打乱候选顺序，目标同步重排。
- 难负提升：提高 `given_nonproof` 类负例采样占比（例如 0.6），增强早段命中。
- 混合教师蒸馏：直接蒸馏 CE logits（K 维 raw 分），小温度 τ=1–2。
- 性能：H800 开启 flash‑attn/TF32；use_cache=False；梯度检查点；保证 tokenizer/embedding 尺寸一致。

### 9.5 对症排查清单（按顺序执行）
1) 评测改口径：mean‑LL + 逐候选独立 + z‑score + (τ,b) 标定 + 可选 PMI；
2) 检查 direct‑softmax 实现：仅累积候选 TEXT token；正确 shift；剔除 PAD/EOS；各候选独立上下文；
3) 用 `target_dist_tuner` 调尖分布（m, τ, α）；
4) 若允许改模型：启用候选打分头 + listwise CE/KL；
5) 复测 Hit@K/NDCG@K，并做 Bi+CE+LLM 三方网格融合（λ_ce, λ_bi, λ_llm）。

### 9.6 两天落地排期（可直接执行）
- Day 1：
  - 实装评测口径参数：`--score-type mean-ll/pmi`、`--isolate`、`--zscore`、`--calib-*`；
  - 跑 dev 标定 τ,b，选定是否用 PMI；
  - 跑 `target_dist_tuner` 重写训练集 `target_scores`（更尖）。
- Day 2：
  - 7B：LoRA lr=1e-4，2–3 epoch，listwise KL；随机候选；难负 0.6；
  - 32B：QLoRA lr=5e-5，2 epoch，同上；
  - 评测：listwise 一致性（分数头/mean‑LL 口径）、Bi top‑200 重排 Hit@K/NDCG@K、三方融合小网格；
  - 记录 metrics.json 与 λ 配置，固化到在线配置。

### 9.7 成功判据与回退
- 成功：Pearson 与 NDCG@32/64 提升；Hit@10 不下降或小幅上升；解析失败≈0；延迟在预算内；
- 回退：若 7B 不收敛，先用 32B 作为教师蒸馏到 7B；或仅作为融合一员（提高中深段 NDCG）。

   - 从 TSTP 证明中抽取 c_# 列表，与全集 CNF 的条目按 `canonical_formula` 匹配 → 得到 label=1；其余条目按出现状态标负（given_nonproof / simplified / passive_only / never_seen）。
4) 写出统一样本
   - 形如：`{problem_name, conjecture_text, text, canonical_formula, features, label, neg_bucket}`。

提示：
- 变量归一/符号标准化用现有 `LLM/data_utils/logic_tokenizer.py`；
- 字面量排序可从“谓词名/极性/项长度”简单排序起步；
- 这一步的收割规模决定了 SFT 上限，建议优先做成批量脚本（命名示例：`scripts/tptp_tstp_align.py`）。

### 8.2 自举采样（Expert Iteration）与弱负/难负
1) 用现有 A/B 方案批跑更多题，产出交互日志：
```bash
python -m LLM.run_batch_pipeline --problems /path/to/list --timeout 10 \
  --log-dir /root/Training/logs/EA.run1
```
2) 抽取子句样本（带 neg_bucket 与 features）并加入训练池：
```bash
python -m LLM.iplog_to_dataset \
  --input  /root/Training/logs/EA.run1/*.raw.log \
  --output /root/Training/datasets/run1_clauses.jsonl
```
3) 弱负：将 `failed.jsonl` 中“前沿但未进证明”的子句作为低权重负例，仅用于训练：
- 在 `make_listwise_chunks` 中已按 `neg_bucket` 权重偏置抽样；
- 验证/测试集中不混入弱负来源（避免评测污染）。

### 8.3 生成 listwise 训练窗
同第 1.2 节，确保每窗 ≥1 正例，K=64/128 依据显存与延迟折中选择。软标签：正例均分或基于启发式加权（如 `shares_goal_consts`/`touches_target_functor`）。

### 8.4 两阶段训练：SFT →（可选）KL/蒸馏 →（可选）RL
阶段 1（必做）：SFT（JSON 规约输出）
- 目标：稳定输出严格 JSON；混入 5–10% 的“格式纠错”样本（错误→正确）提升鲁棒性。
- 命令参考：见第 1.3 节。

阶段 2（建议）：Listwise 分布对齐（KL/CE）
- 目标：让模型输出的分布更贴近目标 soft 分布；
- 现状：脚本已完成纯 SFT，KL 需要在 `train_llm_lora.py` 增加 `--use-kl` 与自定义 `compute_loss`；
- 做法（先行替代）：将目标 JSON 中的 `scores` 写成“更尖锐”的软标签（如正例写 0.9/0.8，其余均摊），继续 SFT 可获得明显排序增益；
- 后续：添加 KL/CE 混合损失 `L = NLL + λ·KL`，λ∈[0.5,1.0] 网格搜。

阶段 3（可选）：教师‑学生与轻量 RL
- 教师蒸馏：
  1) 用更强的教师离线打分窗口，得到 `teacher_scores`；
  2) 生成新的 listwise 数据，将 `target_scores ← teacher_scores`；
  3) 继续 SFT/或 KL 蒸馏到 7B 学生；
  - 教师来源：强 LLM（如 `DeepSeek-R1-Distill-Qwen-32B`）或 Cross‑Encoder（你已训练）。
- 轻量 RL：
  - 在 EA 环路内，以“找到证明=1，否则0”为回合奖励，对 listwise 策略做小步 PPO/GRPO；
  - 或对成功轨迹做回放/偏好优化（DPO）强化“把正例排前”的倾向；
  - 建议在 SFT/蒸馏稳定后开启，以降低波动与工程复杂度。

教师分布产生示例（两条路径）：
```bash
# (A) 用强 LLM 作为教师
export LLM_LOCAL_ENDPOINT=http://127.0.0.1:8002/generate  # 教师服务
python -m LLM.batch_ranker \
  --input  /root/Training/datasets/sample_scores_req.json \
  --out    /root/Training/logs/teacher_llm_scores.json \
  --chunk-size 64 --progress

# (B) 用 Cross-Encoder 作为教师（离线打分脚本示意）
python -m LLM.select.rerank_with_cross_encoder \
  --conjectures /root/Training/datasets/conjs.jsonl \
  --candidates  /root/Training/datasets/cands.jsonl \
  --spm /root/Training/models/spm_logic_24k.model \
  --model /root/Training/models/cross_encoder_best_full.pt \
  --out  /root/Training/logs/teacher_ce_scores.json
```
将教师分布注入 listwise（示意）：
```bash
# 伪代码：把 id->score 映射写回每个窗口的 target_scores
# 也可新增脚本 scripts/inject_teacher_scores.py 完成该步骤
```

### 8.5 课程学习与数据滚动（Expert Iteration）
- 题库分层（家族/符号规模/深度）→ 先中等难度，逐轮滚动吸入新证题；
- 每轮固定验证集，监控 Recall@K（SELECT）/NDCG@K（RERANK）/在线已证题数与 P95 延迟；
- 失败前沿（`failed.jsonl`）补充为难负池，提升区分力。

### 8.6 部署与接线
- 部署：优先 vLLM/TGI，温度 0、max_new_tokens 256–512、合理 stop，观测 QPS 与 P95/P99；
- 接线：`batch_ranker` 通过 `LLM_LOCAL_ENDPOINT` 调用后端；在线失败/超时回退 Bi/启发式；
- 融合：`S = λ_llm·S_llm + λ_bi·S_bi + λ_heur·S_heur`；在验证集做 λ 网格搜索，固化到配置。

### 8.7 成功标准（对齐 DeepSeek‑Prover 评测套路）
- 离线：SELECT Recall@{32,64,128}、RERANK NDCG@K/MAP、解析成功率、吞吐/延迟；
- 在线：固定时限下的 `#已证题数`、平均 given、GPU/CPU 时间与 P95 延迟；
- A/B：记录 λ、K、超时阈值，产出带 CI 的对比报告；
- 实验可复现：保留 prompt/response/分数与 per‑request JSON 摘要。
