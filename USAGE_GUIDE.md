# LLM引导的iProver自动推理系统使用指南

## 🎯 系统概述

这是一个集成了LLM引导的自动定理证明系统，由三个核心组件构成：
- **auto_fof_corpus_builder_local.py** - 主调度器
- **process_iprover_v2.patched.py** - EA服务器（LLM引导模块）  
- **batch_ranker.py** - 子句打分器（支持启发式评分）

## 🚀 快速开始

### 1. 基本运行（使用HTML提取的FOF问题）
```bash
cd /home/ks/LLM

# 基础运行 - 处理5个问题，每个60秒超时
python auto_fof_corpus_builder_local.py \
  --max-problems 5 \
  --timeout 60

# 详细模式
python auto_fof_corpus_builder_local.py \
  --max-problems 10 \
  --timeout 120 \
  --verbose
```

### 2. 指定FOF问题源
```bash
# 使用本地FOF目录
python auto_fof_corpus_builder_local.py \
  --fof-dir /path/to/fof/problems \
  --max-problems 20 \
  --timeout 90

# 使用问题列表JSON文件  
python auto_fof_corpus_builder_local.py \
  --problem-list /path/to/problem_list.json \
  --max-problems 15 \
  --timeout 120
```

### 3. 数据收集模式（Proof Guidance训练数据）
```bash
# 启用数据收集模式 - 收集正负例训练数据
python auto_fof_corpus_builder_local.py \
  --collect-proof-data \
  --dataset-output-format jsonl \
  --max-problems 50 \
  --timeout 180 \
  --verbose

# JSON格式数据集
python auto_fof_corpus_builder_local.py \
  --collect-proof-data \
  --dataset-output-format json \
  --max-problems 30
```

## 🔧 高级配置

### 完整参数示例
```bash
python auto_fof_corpus_builder_local.py \
  --fof-dir /home/ks/TPTP-v9.0.0/Problems \
  --output-dir my_fof_corpus \
  --iprover-path /home/ks/iprover-master/iproveropt \
  --max-problems 100 \
  --timeout 300 \
  --ea-port 12347 \
  --strategy ea_only \
  --collect-proof-data \
  --dataset-output-format jsonl \
  --verbose
```

### 参数说明
- `--fof-dir`: FOF问题目录路径  
- `--problem-list`: 问题列表JSON文件
- `--output-dir`: 输出目录（默认: fof_corpus_refactored）
- `--max-problems`: 最大处理问题数（默认: 50）
- `--timeout`: 单个问题超时时间秒数（默认: 60）
- `--ea-port`: EA服务器端口（默认: 12346）
- `--collect-proof-data`: 启用数据收集模式
- `--dataset-output-format`: 数据集格式（json/jsonl）

## 📊 输出文件结构

### 基础模式输出
```
fof_corpus_refactored/
├── Logs/                           # 日志目录
│   ├── ea_server.log              # EA服务器日志
│   └── diagnostic_*.log           # 问题诊断日志
├── processing_results.json        # 详细处理结果
├── processing_summary.json        # 处理摘要统计
└── dataset/                       # 传统对比学习数据集
    ├── positive_examples.jsonl   
    ├── negative_examples.jsonl
    └── contrastive_dataset.jsonl
```

### 数据收集模式输出
```
fof_corpus_refactored/
├── proof_guidance_datasets/        # 单个问题数据集
│   ├── problem1_dataset.jsonl     # 问题1的训练数据
│   ├── problem2_dataset.jsonl     # 问题2的训练数据
│   └── ...
├── aggregated_datasets/            # 聚合数据集
│   ├── proof_guidance_corpus.jsonl # 完整训练语料库
│   └── dataset_report.md          # 数据集统计报告
└── Logs/                          # 运行日志
```

## 🎯 使用场景

### 场景1: 快速测试（干运行模式）
```bash
# 使用启发式打分，无需API调用
python batch_ranker.py \
  --input test_problem.json \
  --out test_scores.json \
  --dry-run \
  --verbose
```

### 场景2: 生产环境证明搜索
```bash  
# 设置OpenAI API Key
export OPENAI_API_KEY="your-api-key-here"

python auto_fof_corpus_builder_local.py \
  --max-problems 100 \
  --timeout 600 \
  --verbose
```

### 场景3: 大规模数据集构建
```bash
python auto_fof_corpus_builder_local.py \
  --fof-dir /home/ks/TPTP-v9.0.0/Problems \
  --collect-proof-data \
  --dataset-output-format jsonl \
  --max-problems 500 \
  --timeout 300 \
  --output-dir large_proof_corpus
```

## 🔍 监控和调试

### 查看实时进度
```bash
# 监控EA服务器日志
tail -f fof_corpus_refactored/Logs/ea_server.log

# 查看处理进度
watch -n 5 "ls -la fof_corpus_refactored/proof_guidance_datasets/ | wc -l"
```

### 常见问题排查
1. **EA服务器启动失败**: 检查端口12346是否被占用
2. **iProver找不到**: 确认iProver路径 `/home/ks/iprover-master/iproveropt`  
3. **API调用失败**: 检查 `OPENAI_API_KEY` 环境变量
4. **内存不足**: 减少 `--max-problems` 参数

## 📈 性能优化

### 启发式模式（推荐用于开发测试）
- 使用 `--dry-run` 无需API调用
- 基于A/B/C/D分级算法智能评分
- 适合快速验证和调试

### 生产模式配置
- 合理设置 `--timeout` (推荐180-600秒)
- 使用 `--verbose` 监控进度  
- 定期清理日志文件避免磁盘满

## 🎓 训练数据格式

### JSONL格式示例
```json
{"problem_id": "ALG001-1", "clause_id": 123, "label": 1, "example_type": "positive", "clause_text": "~P(X) | Q(X,f(X))", "clause_features": {...}}
{"problem_id": "ALG001-1", "clause_id": 124, "label": 0, "example_type": "negative", "negative_type": "passive_only", "clause_text": "R(a,b)", "clause_features": {...}}
```

### 数据集统计
系统会自动生成数据集报告，包含：
- 正例/负例数量统计
- 负例类型分布（passive_only, given_nonproof, simplified, never_seen）
- 处理成功率和平均时间

---

**🔥 现在你可以开始使用了！建议从小规模测试开始，然后逐步扩大到生产规模。**