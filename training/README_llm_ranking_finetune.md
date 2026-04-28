# LLM 排序器微调与部署指南（7B/32B｜评分头 & PMI）

本指南给出可直接落地的两条并行路线：
- A）候选打分头（一次前向输出 K 分，推荐主线）
- B）逐候选独立前向（PMI/平均对数似然，零改模，可作教师/兜底）

面向两类模型：DeepSeek‑Prover‑V2‑7B（在线主力）与 Goedel‑Prover‑V2‑32B（高精教师/离线重排）。

---

## 1. 共同前置（数据/Tokenizer/模板）

- 数据形态（listwise，K=64 推荐）
  - 每条样本：`{conjecture, candidates[i].text, candidates[i].tags, target_scores[i]}`
  - target_scores：教师融合软分布 p*（可先用 CE 或 CE+Bi 融合；在 dev 上已调好形状 m/τ/α）
  - 采样：难负占比 ≥60%（NEG_given_nonproof、NEG_simplified），每个 query 至少 1 个正
  - 每个 epoch 重排候选顺序，并同步重排 target_scores

- 特殊 tokens（必须）
  - 注册 `<CAND_START>`、`<CAND_END>`（及你已有的结构前缀 `<Q>/<D>/…`）
  - `tokenizer.pad_token = tokenizer.eos_token`（或新增 PAD）
  - `model.resize_token_embeddings(len(tokenizer))` 且 `model.config.pad_token_id = tokenizer.pad_token_id`

- 输入模板（示意）
  - 段落：
    - `[CONJECTURE]` 段（<Q>…</Q>）
    - K 个 `[CANDIDATE i]` 段：`<CAND_START> text_i tags_i <CAND_END>`
  - TAGS（例如 `<unit><horn><len=..>`）为强提示，建议保留

- 生成 K=64 的 groups（可选，用于评分头训练）
```bash
python3 LLM/training/prepare_listwise_data.py \
  --input /root/autodl-tmp/Training/datasets/pairs.full.train.jsonl \
  --output /root/autodl-tmp/Training/datasets/groups.k64.train.jsonl \
  --k 64 --min-positives 1 --seed 42 \
  --teacher ce --tau 1.0 \
  --ce-scored /root/autodl-tmp/Training/datasets/ce_scored.train.jsonl
```

> 注：无需重做 pairs；groups 仅用于 LLM 评分头 listwise 训练。

---

## 2. DeepSeek‑Prover‑V2‑7B（在线主力）

### 路线 A：LoRA + 候选打分头（推荐）
- 目标：一次前向产 K 个分；组内 softmax 与教师分布对齐（CE/KL）
- 前向/头部：
  - `model(..., output_hidden_states=True)`
  - 每个候选取 `<CAND_START>` 的 hidden（或 [start,end) 平均池化）
  - 线性头：`s_i = w^T h_i + b` → [B,K]
  - 组内 z-score → (τ,b) 标定（dev 拟合）→ softmax 得 p̂；loss=CE(p*, p̂)
- LoRA/优化（稳定省显存）：
  - target_modules: `[q_proj,k_proj,v_proj,o_proj]`（可再含 gate/up/down 测试）
  - r=16（或 8），alpha=32（或16），dropout=0.05
  - bf16，gradient_checkpointing=True，use_cache=False；可用 flash‑attn 则开启
  - 批量：batch=1，grad_accum=16（等效 bs=16），max_len=1024（必要时 896/768）
  - AdamW lr=1e-4（或 5e-5），wd=0.01，cosine + warmup_ratio=0.03
  - 2–3 epoch；dev 监控 NDCG@32 / KL 早停
- 评测/标定：NDCG@{10,32}、Hit@{10,32}、与教师分布 KL/MAE/Pearson；dev 拟合 (τ,b)，保存 calib.yaml
 - 训练/推理命令（与仓库脚本对齐）
```bash
# 依赖
pip install "transformers>=4.43" "peft>=0.11" "accelerate" "bitsandbytes" "sentencepiece" "scikit-learn"

# 训练（已实现 LoRA/头部/候选定位）
python3 LLM/training/train_llm_head.py \
  --model /root/autodl-tmp/models/DeepSeek-Prover-V2-7B \
  --train /root/autodl-tmp/Training/datasets/groups.k64.train.jsonl \
  --dev   /root/autodl-tmp/Training/datasets/groups.k64.dev.jsonl \
  --lora-r 16 --lora-alpha 32 --lora-drop 0.05 \
  --lr 1e-4 --wd 0.01 --epochs 2 --warmup 0.03 \
  --max-len 1024 --batch 1 --grad-accum 16 --bf16 --grad-checkpoint \
  --save /root/autodl-tmp/Training/models/llmhead_7b.pt

# 评分（统一预测 JSONL）
python3 LLM/training/score_llm_head.py \
  --ckpt /root/autodl-tmp/Training/models/llmhead_7b.pt \
  --groups /root/autodl-tmp/Training/datasets/groups.k64.dev.jsonl \
  --out /root/autodl-tmp/Training/datasets/pred.llmhead.dev.jsonl \
  --bf16 --max-len 1024

# 评测
python3 LLM/training/unified_evaluator.py \
  --pred /root/autodl-tmp/Training/datasets/pred.llmhead.dev.jsonl --name LLM-HEAD \
  --ks 1 10 32 64 \
  --out /root/autodl-tmp/Training/datasets/metrics.llmhead.dev.json
```

> 提示：若出现 `No module named 'LLM'`，先执行 `export PYTHONPATH=/root:$PYTHONPATH`，或改用模块方式运行：`python3 -m LLM.training.train_llm_head ...`、`python3 -m LLM.training.score_llm_head ...`。
- 部署（在线）：
  - 实现 `LLMHeadScorer7B`：输入 {conjecture, candidates[], tags[]} → 前向得到 K 分；应用 calib.yaml；输出口径与 CE 一致
  - 与 Bi 轻融合（`w_bi≈0.2, w_llm≈0.8`）并监控线上 KPI；使用 prefix-cache 降时延

### 路线 B：逐候选平均对数似然/PMI（兜底/教师）
- 每候选独立前向（避免互看）；仅统计候选 span 的平均对数似然 avg_logP(c|q)
- PMI 校正：`score = avg_logP(c|q) − λ·avg_logP(c)`，λ∈[0.5,1.0] 在 dev 拟合
- 组内 z-score → (τ,b) → softmax；写 calib.yaml
- 优点：零改造，立刻可用；可作为教师或线上兜底；缺点：时延约 K 倍

运行（已实现 `pmi_scorer.py`）
```bash
python3 LLM/training/pmi_scorer.py \
  --model /root/autodl-tmp/models/DeepSeek-Prover-V2-7B \
  --data  /root/autodl-tmp/Training/datasets/pairs.full.dev.jsonl \
  --out   /root/autodl-tmp/Training/datasets/pred.pmi.dev.jsonl \
  --lambda-pmi 0.7 --max-len 1024

python3 LLM/training/unified_evaluator.py \
  --pred /root/autodl-tmp/Training/datasets/pred.pmi.dev.jsonl --name PMI \
  --ks 1 10 32 64 \
  --out /root/autodl-tmp/Training/datasets/metrics.pmi.dev.json
```

---

## 3. Goedel‑Prover‑V2‑32B（高精教师/离线重排）

### 路线 A：QLoRA + 候选打分头（推荐）
- 角色：强教师/离线重排器（对齐 p*）
- QLoRA/优化：
  - 量化（可选）：如需 4-bit/QLoRA，请按环境配置 bitsandbytes；本仓库训练脚本默认全精/半精加载
  - LoRA：r=16（或 8），alpha=32（或 16），dropout=0.05；target_modules 同 7B
  - AdamW lr=5e-5，wd=0.01，cosine + warmup_ratio=0.03
  - 1–2 epoch；bf16，gradient_checkpointing=True，use_cache=False，flash‑attn 可开
  - 显存：80G 直上；48G 建议 max_len≤896 或 K=48；或多卡 DP/FSDP
- 训练/推理（示例）
```bash
python3 LLM/training/train_llm_head.py \
  --model /root/autodl-tmp/models/Goedel-Prover-V2-32B \
  --train /root/autodl-tmp/Training/datasets/groups.k64.train.jsonl \
  --dev   /root/autodl-tmp/Training/datasets/groups.k64.dev.jsonl \
  --lora-r 16 --lora-alpha 32 --lora-drop 0.05 \
  --lr 5e-5 --wd 0.01 --epochs 2 --warmup 0.03 \
  --max-len 896 --batch 1 --grad-accum 16 --bf16 --grad-checkpoint \
  --save /root/autodl-tmp/Training/models/llmhead_32b.pt

python3 LLM/training/score_llm_head.py \
  --ckpt /root/autodl-tmp/Training/models/llmhead_32b.pt \
  --groups /root/autodl-tmp/Training/datasets/groups.k64.dev.jsonl \
  --out /root/autodl-tmp/Training/datasets/pred.llmhead32.dev.jsonl \
  --bf16 --max-len 896
```
- 用途：离线重排 CE Top‑M（如 256）并缓存；与 Bi/CE/PMI 融合为 p*，反向蒸馏给 7B

### 路线 B：PMI（零改造）
- 同 7B 的 PMI；更适合强教师/关键样本复核；建议只对 Top‑M 运行以控时延

---

## 4. 协同与融合
- 教师融合：`p* = w_bi·p_bi + w_ce·p_ce + w_llm32·p_llm32`（dev 网格搜索；早段看 Hit@10，中深看 NDCG@32）
- 蒸馏：用 p* 训练 7B/32B 的评分头（listwise CE/KL）
- 上线：7B‑Head 主力 + Bi 融合；离线：32B 重排/复核低置信度；统一 calib.yaml

---

## 5. 标定与统一评测
- 标定：
  - 组内 z-score（每 query 的 K 窗口）；dev 上拟合 (τ,b)（以及 PMI 的 λ），保存 calib.yaml；训练/评测/上线统一使用
- 统一评测（仓库内已有工具）：
  - 统一预测 JSONL：`{problem_name, query/text_a, group_id, doc/text_b, label, score}`
  - CE/BI 产出：`score_cross_encoder.py --pred-out`、`score_biencoder.py`
```bash
python3 LLM/training/unified_evaluator.py \
  --pred /root/autodl-tmp/Training/datasets/pred.ce.dev.jsonl --name CE \
  --pred /root/autodl-tmp/Training/datasets/pred.bi.dev.jsonl --name BI \
  --ks 1 10 32 64 \
  --out /root/autodl-tmp/Training/datasets/metrics.dev.json
```

---

## 6. iProver 接入
- Sidecar（首选）：启一个 Python HTTP `/score` 服务 → `iserver_mp.py` 的 `gpu_worker`/`eval_files` 调用 HTTP 替代 `network.predict`；保留 GNN 作兜底
- 原位替换（可选）：在 `iserver_mp.py` 中用 State.register_clauses 保存的子句文本，组合 {conjecture + candidates} → 7B‑Head/PMI 评分；注意 micro‑batch、`query_max_size`、前缀缓存

---

## 7. 关键易错点
- 绝不把 pad_id 当标签；若存在 token‑level 训练，padding 处 `labels=-100`；评分头路线不做 token CE
- tokenizer/embedding 对齐：`resize_token_embeddings + pad_token_id`
- `<CAND_START>` 位置定位必须正确；被截断候选需丢弃或 mask
- 打分头允许候选互看；PMI 必须逐候选独立前向
- PMI 长度偏置：用平均 LL，而非总和
- 训练采样：候选顺序每 epoch 打乱；提升难负
- 早停：dev 的 NDCG@32/Hit@10 为主，辅以 KL/MAE/Pearson

---

## 8. 伪代码（评分头）
```python
# 前置：model(..., output_hidden_states=True); tokenizer.pad->eos; resize_token_embeddings

def build_inputs(conjecture, candidates):
    text = format_prompt(conjecture, candidates)  # 见上模板
    enc = tokenizer(text, max_length=MAX_LEN, truncation=True, padding="max_length", return_tensors="pt")
    cand_starts = locate_cand_starts(enc.input_ids, tokenizer.convert_tokens_to_ids("<CAND_START>"))
    return enc, cand_starts  # 每样本 K 个索引

class ScoreHead(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.proj = nn.Linear(hidden, 1)
    def forward(self, H, idxs):  # H: [B,T,Hs]; idxs: 展平后 [B*K]
        hs = H.view(-1, H.size(-1))[idxs]  # 或按批 gather
        return self.proj(hs).view(B, K)

outputs = model(**enc)
H = outputs.hidden_states[-1]
s = score_head(H, cand_starts)     # [B,K]
s = zscore_per_window(s)
p_hat = softmax((s - b)/tau, dim=-1)
loss = CE(p_star, p_hat)
```

---

## 9. 最小可执行清单
- 用 CE 生成 `ce_scored.*` → 如需评分头，`prepare_listwise_data.py --k 64` 生成 `groups.*`
- 7B‑Head：r=16, α=32, lr=1e-4, epoch=2–3, bf16, max_len=1024, bs=1×accum=16；dev 标定 (τ,b)
- 7B‑PMI：逐候选，拟合 λ、τ、b；生成统一预测并对比
- 32B‑Head：QLoRA（lr=5e-5, 1–2 epoch, max_len≈896）或 32B‑PMI（Top‑M）
- 统一评估：`unified_evaluator.py` 对比 CE/BI/7B‑Head/7B‑PMI/32B‑PMI（或 32B‑Head）
- 上线：7B‑Head 主 + Bi 融合；Sidecar 接 iProver；定期离线 32B 重排/校准

---

参考：
- 排序管线与评测（CE/BI）：`LLM/training/README_ranker_pipeline.md`
- 工具：`prepare_listwise_data.py`、`score_cross_encoder.py`、`score_biencoder.py`、`score_llm_head.py`、`pmi_scorer.py`、`unified_evaluator.py`
