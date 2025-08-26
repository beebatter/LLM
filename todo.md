目标一：训练一个 Transformer 做“快速预筛”（对比学习/排序）
1) 数据改进（越早做越省后期弯路）

按题切分：用我给你的 split（train/val/test 按 problem_name 分组）来避免泄漏。

去重 & 去共公理：我们发现有不少跨题重复子句。两种做法：

训练时对出现于多题的子句降低采样概率或权重；

或者把“是否跨题出现”作为一个二值特征，作为 token 加到序列里（下文会讲）。

负例多样化：现在几乎全是 NEG_passive_only，建议追加更硬的负例：

NEG_given_nonproof（被 given 过但不在 proof 中）；

NEG_simplified（被化简/删除）；

NEG_frontier（给定窗口内的邻近子句但非正例）。
用这些桶做分层采样，训练会更稳定。

（可选）引入“目标侧”信息：如果你能把conjecture 的文本或“目标符号签名”（目标中的谓词/函数集合）也存进样本，快速预筛能更“题相关”。没有也能做“无条件先验”预筛，但有目标侧信息效果通常更好。

2) 输入建模与分词

Tokenizer：用 SentencePiece/Unigram 训练一个 16k–32k 的子词表，在整个 TPTP 语料（子句+猜想）上训练。保留 (, ), ,, =, ~, | 等符号为原子 token，变量名建议归一化（如 X0→VAR0）以减小稀疏。

结构提示：把“易学”的结构特征写成特制 token拼接在前面（比单独喂标量特征更兼容 Transformer）：

例：<HORN=1> <EPR=1> <UNIT=0> <BORN=123> <CONJ_DIST=2> 再接子句文本；

或精简：<H1> <E1> <U0> <C2> <B123>（自定义离散化 bins，避免太多不同的 BORN 值）。

双塔 or 单塔？

无目标侧：单塔（encoder）即可，直接把子句编码成向量，做监督对比/pairwise 排序；

有目标侧：推荐双塔（Encoder_q 编 conjecture，Encoder_d 编 clause），用内积或双线性头做相似度。推理时很快（clause 向量缓存）。

3) 损失函数与采样

Supervised Contrastive / InfoNCE（题内）

一个 batch 内来自同一题或少量题；设 anchor 是正例，正样本为“同题正例”，负样本为“同题负例”（或 in-batch negatives）。

温度 τ=0.07 左右；每题正样本数量不足时，用 hard-negative 复用（见下）。

Pairwise Ranking（BPR/hinge）

采 (<clause^+>, <clause^->) 同题对，最小化 -log σ(s^+ - s^-) 或 max(0, m - s^+ + s^-)。

Hard negative mining

从 NEG_given_nonproof、NEG_frontier 优先采样；

训练进行到 1/3 steps 后，加入在线难负例：用当前模型在同题内打分，挑靠前但为负例的样本并加入队列（MoCo/queue 思路）。

采样比例

每题：pos:neg ≈ 1:3 ~ 1:5；

桶内：NEG_given_nonproof：NEG_frontier：NEG_passive_only ≈ 2:2:1（按你数据量微调）。

4) 模型与超参建议（先跑通为主）

小型 Transformer encoder（自训或从小模型初始化）

层数 6–8，隐藏 384–512，头数 6–8，max len 256；

投影头：[CLS]/mean-pool → MLP(2 层，隐藏 256) → L2 normalize；

优化：AdamW, lr=1e-4, wd=0.01, warmup 5%，总步数 30k–100k（看数据）；

梯度裁剪 1.0，bf16/FP16，可多卡 DDP；

Batch 按“题内混合小批”组织：例如一次采 8 个题，每题 8 个样本，batch=64。

双塔（如果有 conjecture 文本）

两个塔共享或不共享参数均可；共享更省参数；

训练时 InfoNCE：以 q 为 anchor，正为同题正例子句，负为同题负例；in-batch negatives 自然有效。

5) 评测（离线 + 在线）

离线：按题内 Recall@K / MAP / NDCG@K（我给你的指标表就是这个）。

在线：接回 EA，观测

已证题数↑、平均 given 次数↓、平均时间↓，

以及 proof_out 出现比例↑。

Ablation：比较有/无结构特征 token、有/无 hard negatives、有/无 conjecture 双塔。

6) 接入 EA（超快）

预筛 = 用你训练的 encoder 先给每个 clause 一个 prior 分数；

batch_ranker.py 接口里：

对新子句做一次编码并缓存（cache key 用“子句文本哈希”）；

如果有双塔：conjecture 在该题只编码一次；

返回 scores = sim(q, d) 或 head(d)。

你已有“日志静音/截断”，吞吐会更稳；把 batch_size 与 GPU 显存配套。

目标二：用语料微调“大模型”，让它直接返回 softmax 评分
1) 更合适的“监督信号”形式

你的离线语料是点标签（0/1）；但你想让 LLM对“一个子句列表”输出归一化 softmax。

建议把每个 chunk（题内一批子句） 制作成目标分布（listwise target）：

平滑分布：

设正例个数为 P，总数 N；

令正例目标分布为 (1-ε)/P（均分），负例为 ε/(N-P)，ε=0.05~0.1；

这样避免全 0/1 的不可微问题，也匹配“softmax 归一化”。

或者 pairwise 格式：随机采 (pos, neg) 对，让 LLM输出“二选一更相关”的偏好；用 DPO/SFT 都能做，但 listwise 更贴目标。

Chunk 组织：与现在 batch_ranker.py 的 prompt 对齐（“- ID xxx\n formula: …”），保持固定模板以便接入。

2) 训练方式（两条路线）
路线 A：SFT + 解析归一化

训练 LLM以 JSON 输出：

{"scores":[["ID1",0.12],["ID2",0.03],...]}


训练时的 label 就是上面的平滑分布（按 ID 对齐）；

损失：标准 LM 损失（teacher forcing）虽然是“文本损失”，但在数值上很容易收敛到你给定的分布；

推理端：即便模型输出不完全归一，也二次 softmax 归一化；并加容错（非法 JSON / 不足项 → 回退启发式）。

优点：易实现、能直接兼容你现有 batch_ranker.py；
缺点：严格的“概率校准”不是 LM 的强项，但对排序影响不大（我们会重归一且只看相对大小）。

路线 B：加个“数值头”（如果你能改模型）

把 LLM当 cross-encoder：把每个子句拼接到 prompt，取特殊 token 的隐状态，过一个线性头得到logit；

对该 列表的 logits 做 softmax，最小化KLDiv到你的目标分布；

这需要能拿到中间隐藏态，常见 LoRA 也能做。

效果稳定、可精确控制，但工程上比 A 路线复杂。

两条路线都建议 LoRA/QLoRA，节省显存，学习率 1e-42e-4（LoRA 层），rank 816，α=16~32。

3) Prompt 细节（通吃训练与推理）

强约束 JSON（和你现在的一致），并强调“ID 对齐”、“分数 ∈ [0,1] 且和为 1”；

上文摘要要控制在你设的 --summary-max-tokens 内（我看你是 500/700），以免漂移；

指令里列出评分要点：Horn/Unit/ConjDistance/是否出现目标符号/是否等式等（这些线索 LLM 能学会复用），和你的启发式指标保持一致。

4) 训练样本的构造建议

按题切 chunk，chunk_size 不宜太大（32/64），太大时 LLM 容易走形；

正/负比例：确保每个 chunk 至少有 2~4 个正例；如果题里正例极少，合并相邻 given 窗口或正例上采样；

将结构特征作为显式提示（和上面 encoder 一样），例如：

- ID 12345 | tags: [horn, epr, unit, conj_dist=2, born=37]
  formula: ~ p(X) | q(f(X))


错误防护：prepare 脚本里生成 expected_ids 列表，推理解析后强制对齐，缺失的 ID 补 0 分，最后重归一化。

5) 评测 & 接入

离线：和 encoder 一样看 per-problem Recall@K/MAP；

在线：把 --ranker-script 指向你微调 LLM 的脚本，跑 EA；

加入回退：若解析失败或低置信（你可让 LLM附带 confidence 字段），回到启发式或 encoder 分数融合。

推荐的立即可做的 10 件小事

把 conjecture 文本或“符号签名”加入样本（作为 <CONJ> ... token 串），为将来的双塔/交叉打基础。

补充负例桶（NEG_given_nonproof 等），并在训练采样里按桶配比采样。

训练一个 6 层 512 隐层的 encoder（先无目标侧也可），用 Supervised Contrastive / Pairwise 跑 30k step，看离线 Recall 提升。

Clause embedding 缓存：在 batch_ranker.py 里加 LRU cache（key=子句文本哈希），显著降延迟。

SentencePiece 词表：用子句+猜想训练 16k 词表，并替换 tokenizer。

把结构特征变为 token：<H1> <U0> <C2> <B37> 前缀到文本。

hard negative mining：训练到 1/3 步数后，加入“模型打分靠前但为负”的子句。

LLM SFT 准备：用当前 chunk 格式导出 listwise 目标分布（带 ε 平滑），做一版 LoRA 微调。

解析兜底：batch_ranker.py 里，解析 JSON 失败→回退启发式/encoder；并做二次 softmax。

统一评测脚本：离线同一个 eval_rank_metrics.py 跑 encoder 与 LLM 的分数，便于对比。

先做的通用改进（训练和评测前）

按题切分 + 去泄漏

只用“按 problem_name 分组”的 train/val/test（不要按行随机）。

训练/验证/测试三份数据互不共享相同子句文本（如果有跨题重复文本，训练时降低这些样本权重或在验证/测试中去掉它们）。

分布均衡与采样策略

按题均衡采样：每个 batch 选若干题（如 8 题），每题采相同数量样本，避免 ALG104+1/ALG047+1/ALG128+1 这类大题“主导训练”。

负例分桶采样：优先采 NEG_given_nonproof 和 NEG_simplified（各 40%），NEG_passive_only（20%）。没有时再回退。

把结构特征变成 token 前缀（对 Transformer/LLM 都有益）
例如：

<H1> <U0> <E1> <C2> <B37>    （H:horn U:unit E:EPR C:conj_dist=2 B:born=37）
formula: ~ p(X) | q(f(X))


连续数值（born/conj_dist）要离散化成桶（如 B0/B1/…/B5），减少稀疏与过拟合。

尽量加入“目标侧”语境（如果能提取到 conjecture 或目标符号签名）

轻量做法：在每条样本前加 <CONJ> ... 或 <SIG> [pred1, fun2, ...]；

后续升级到双塔（conjecture 塔 + clause 塔）时直接复用。

三类评测统一起来

离线：题内 Recall@K (10/32/64)、MAP；

在线：EA+iProver 的“已证题数↑ / 平均 given 次数↓ / 平均时间↓”；

A/B：与“纯启发式”和“旧版 LLM dry-run”对比。

目标 1：Transformer 编码器做对比学习→ 快速预筛
数据形成（强烈建议）

Pairwise / Triplet（题内）：

每个正例 c+，配 k 个负例 c-；优先从 NEG_given_nonproof、NEG_simplified 采；不够再用 NEG_passive_only。

比例：pos:neg = 1:3 ~ 1:5，其中硬负例 ≥ 50%。

Batch 结构：一次采 T 个题（如 T=8），每题 P 个正 + N 个负（如 P=8, N=24）；这样 InfoNCE 可以题内成对比，in-batch negatives 自然有效。

Hard negative mining（第 1/3 训练后启用）：用当前模型在同题内打分，从非正例里挑“排在前面的那些”追加到负例池。

模型与超参（先小后大）

Tokenizer：SentencePiece/Unigram 16k~32k；保留括号、逗号、等号等符号；变量归一化（X7→VAR）。

Encoder：6–8 层、hidden=512、heads=8、max_len=256；池化用 [CLS] 或 mean-pool；投影头 2 层 MLP（hidden=256，最后 L2 norm）。

损失：

InfoNCE / SupCon（题内）：温度 τ=0.07；一个正例对所有同题正例为正对，其余为负。

或 BPR/hinge（Pairwise）：-log σ(s+ - s-) 或 max(0, m - s+ + s-)，m=0.2。

优化：AdamW，lr=1e-4，wd=0.01，warmup=5%，总步数 50k 左右（看早停）；FP16/bf16；grad clip=1.0。

正则：dropout=0.1；label smoothing（对于 pairwise 不用）。

特征前缀 token如上，加速收敛且“可解释”。

评测与期望

离线每题 Recall@K 明显高于启发式（R@64 提升 10–30 个点很常见）；

在线 EA：同样问题集下已证题数↑、证明步数↓。

如果只做“无目标侧”的先验预筛，也能拉开与启发式的差距；加入 conjecture（双塔）后，提升更稳。

接入 EA 的注意点

缓存：以“子句文本哈希”为 key 缓存向量/分数（ID 每次会话不同）。

吞吐：把 ranker 做成批量（一次 encode 多个子句）；GPU 上可轻松几千条/s。

回退：编码失败/超长 → 启发式权重小幅回退（比如 10% 权重混合）。

目标 2：用当前语料微调 LLM，让它直接输出 softmax 评分
数据组织：listwise 目标分布

以“题内一个 chunk”的形式构建样本（和你现在 scores_req 的格式一致）：

- ID 12345 | tags: [H1,E1,U0,C2,B37]
  formula: ...
- ID 12346 | tags: [H1,E0,U1,C1,B12]
  formula: ...
...


把该 chunk 的标签转成平滑分布 y（避免 0/1 尖分布）：

假设 P 个正例、N 总数，设 ε=0.1：正例目标是 (1-ε)/P，负例目标是 ε/(N-P)。

训练目标：KLDiv( softmax(logits) || y ) 或直接 CrossEntropy 到按 y 排序的目标（listwise 更稳）。

如果不改模型结构，也可以 SFT 让 LLM输出 JSON（{"scores":[["ID",0.12],...]}），训练时用 LM 损失；推理后再二次 softmax 归一化（解析容错要做好）。
效果略弱于“直接在隐藏态加数值头”的做法，但实现最快、与你的 batch_ranker.py 完全兼容。

训练配置（LoRA/QLoRA）

LoRA rank=816，α=1632；lr=1e-42e-4（LoRA 层）；warmup=3%；epoch=13（看早停）；

chunk_size=32/64；保证每个 chunk 至少含 2–4 个正例；

Prompt 要非常明确：

“只输出 JSON；分数 ∈[0,1] 且和为 1；ID 必须覆盖输入列表；可选给 confidence”；

明示评分要点（Horn/Unit/ConjDist/是否含目标符号/是否等式），和启发式一致，利于学习。

推理端：

解析失败 → 回退启发式/encoder；

对输出再做softmax 归一（防止漂移）；

混合分数 S = λ * S_LLM + (1-λ) * S_encoder（λ 在验证集上网格搜索，如 0.6）。

你这份数据的针对性优化建议

题目数量（18）偏少：强烈建议扩到 50+ 题再做严肃对比，否则泛化不稳定。短期可以做 “留一题”交叉验证（LOPO-CV） 做稳健评测。

负例结构已不错：有 968 个 NEG_given_nonproof 与 345 个 NEG_simplified，足够做 hard negatives；训练时把这两类权重/采样概率拉高。

失败 42 条都是 timeout：调大 iProver 超时或减少问题难度可以进一步扩大语料；也可把 timeout 的题目纳入“困难集”做在线评测。

正例分布头重：如 ALG104+1/ALG047+1/ALG128+1 正例很多，训练采样要按题均衡，并对每题设“每 epoch 上限”，避免它们压制其他题。




方案	SELECT	RERANK	调度	已证题数	平均 given	平均时间	R@64 (离线)
启发式基线	规则	无	FIFO	–	–	–	–
只用 Bi-Encoder	Bi	无	age/轻/随机	↑	↓	↓	↑
Bi + LLM	Bi	LLM	混合	↑↑	↓↓	↓↓	↑↑
+ 蒸馏	Bi(蒸馏)	LLM	混合	最佳	最佳	最佳	最高