# iProver FOF语料库构建器

一个集成的工具，用于构建大语言模型的FOF（First-Order Logic）定理证明语料库。

## 核心文件

- **`auto_fof_corpus_builder_local.py`** - 主要脚本，集成所有功能
- **`html_fof_extractor.py`** - HTML问题列表提取器
- **`batch_ranker.py`** - 子句打分模块
- **`process_iprover_v2.patched.py`** - EA服务器
- **`setup_and_run_fof.sh`** - 一键运行脚本
- **`iprover_FOF_result.html`** - FOF问题源文件

## 主要功能

### 🎯 三种处理策略

1. **混合策略** (`hybrid`) - **推荐**
   - 简单问题：直接证明提取
   - 复杂问题：EA交互引导
   - 自动选择最优方案

2. **直接证明** (`direct_only`)
   - 纯粹运行iProver获取证明
   - 适合快速收集正例数据

3. **EA交互** (`ea_only`)
   - 全程EA引导
   - 适合困难问题的详细分析

### 🛠 核心能力

- **统一的证明提取** - 支持CNF/FOF/TCF多种格式
- **早期终止处理** - 自动识别并处理快速证明的问题
- **EA日志解析** - 从EA交互日志提取子句信息
- **对比学习数据集** - 自动构建正例/负例数据集
- **性能分析** - 详细的统计和改进建议

## 快速开始

### 基本用法

```bash
# 默认模式：使用HTML提取的488个FOF问题（推荐）
python auto_fof_corpus_builder_local.py \
    --max-problems 50 \
    --timeout 60 \
    --strategy hybrid

# 使用指定目录的问题
python auto_fof_corpus_builder_local.py \
    --fof-dir /path/to/fof/problems \
    --output-dir fof_corpus \
    --strategy hybrid

# 使用自定义问题列表文件
python auto_fof_corpus_builder_local.py \
    --problem-list problems.json \
    --output-dir fof_corpus \
    --strategy direct_only

# 一键运行（交互式）
./setup_and_run_fof.sh
```

### 参数说明

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--fof-dir` | FOF问题目录路径 | 自动使用HTML提取 |
| `--problem-list` | 问题列表JSON文件 | `fof_problems_from_html.json` |
| `--output-dir` | 输出目录 | `fof_corpus_refactored` |
| `--max-problems` | 最大处理问题数 | 50 |
| `--timeout` | 单个问题超时时间（秒） | 60 |
| `--strategy` | 处理策略 | `hybrid` |
| `--ea-port` | EA服务器端口 | 12346 |
| `--iprover-path` | iProver路径 | `/home/ks/iprover-master/iproveropt` |

### 🎯 智能问题源选择

系统会按以下优先级自动选择问题源：
1. **用户指定的问题列表** (`--problem-list`)
2. **用户指定的目录** (`--fof-dir`) 
3. **HTML自动提取** (默认，488个精选FOF问题)

HTML自动提取使用 `iprover_FOF_result.html` 中的问题名称，自动匹配到TPTP目录：
- 支持37个不同领域（AGT、ALG、BIO等）
- 自动域名映射（如BIO003+1 → `/TPTP/Problems/BIO/BIO003+1.p`）
- 100%成功匹配率

## 输出结构

```
fof_corpus/
├── dataset/                    # 对比学习数据集
│   ├── positive_examples.jsonl    # 正例（有证明的子句）
│   ├── negative_examples.jsonl    # 负例（无证明的子句）
│   └── contrastive_dataset.jsonl  # 混合数据集
├── Logs/                       # 详细日志
│   ├── ea_server.log              # EA服务器日志
│   └── problem_XXX/               # 每个问题的专用日志
├── processing_results.json    # 详细处理结果
└── processing_summary.json    # 处理摘要和建议
```

## 数据格式

### 正例数据（有证明的子句）
```json
{
  "problem_id": "ALG032+1",
  "problem_file": "/path/to/problem.p",
  "status": "theorem",
  "szs_status": "Theorem",
  "strategy": "direct_proof",
  "clause": {
    "clause_id": "c_64",
    "clause_text": "(op(e5,e5)=e4)",
    "clause_type": "plain",
    "source": "tcf_internal",
    "format_type": "tcf"
  },
  "label": "positive"
}
```

### 负例数据（无证明的子句）
```json
{
  "problem_id": "BIO006+1", 
  "status": "unknown",
  "clause": { /* 子句信息 */ },
  "label": "negative"
}
```

## 性能优化建议

### 基于策略选择

- **大量简单问题** → 使用 `direct_only`
- **混合难度问题** → 使用 `hybrid`（默认）
- **复杂研究问题** → 使用 `ea_only`

### 参数调优

```bash
# 快速验证（小数据集）
--max-problems 10 --timeout 30 --strategy direct_only

# 平衡模式（中等数据集）
--max-problems 100 --timeout 60 --strategy hybrid

# 深度分析（大数据集）  
--max-problems 500 --timeout 300 --strategy ea_only
```

## 常见问题

### Q: 大部分问题都是早期终止怎么办？
A: 这是正常的，说明你的问题相对简单。重构版本会自动处理早期终止并提取证明子句。

### Q: 如何获得更多EA交互数据？
A: 使用更复杂的问题集，或者增加timeout时间，使用`ea_only`策略。

### Q: 证明子句提取失败怎么办？
A: 检查iProver路径是否正确，确保`--proof_out true`参数生效。

## 技术细节

- **SZS状态映射**: Theorem/Unsatisfiable → 正例, Satisfiable/Unknown → 负例
- **早期终止检测**: 自动识别EA连接后立即断开的情况
- **混合策略时间分配**: 快速尝试用时 = min(timeout/4, 30秒)
- **子句格式支持**: CNF, FOF, TCF, EA内部格式

---

## 版本历史

- **v2.0** (重构版本) - 集成所有功能，支持三种策略
- **v1.x** - 分离的脚本版本（已废弃）

有问题请查看日志文件或联系开发者。