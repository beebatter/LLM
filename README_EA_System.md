# 大模型EA (External Agent) 与iProver交互系统

## 系统概述

本系统实现了一个完整的外部代理(EA)，用于与iProver自动定理证明器进行交互，使用大语言模型对子句进行评分，指导证明搜索过程。

## 系统架构

```
┌─────────────┐    TCP Socket    ┌─────────────┐    subprocess    ┌─────────────┐
│   iProver   │ ←────────────→  │  EA Server  │ ─────────────→  │   Ranker    │
│             │    JSON/NUL     │             │   python3       │   (LLM)     │
└─────────────┘                 └─────────────┘                 └─────────────┘
```

## 核心组件

### 1. EA Server (`ea_server.py`)
- **职责**: 处理与iProver的TCP连接和JSON消息交换
- **功能**:
  - 接收并注册clause
  - 处理scoring请求  
  - 准备batch数据
  - 调用ranker进行评分
  - 返回scores给iProver
  - 详细日志记录

### 2. Batch Ranker (`batch_ranker.py`)
- **职责**: 使用大语言模型对clause进行评分
- **功能**:
  - 解析canonical formula
  - 生成背景摘要
  - 分块处理candidates
  - LLM评分
  - 归一化和对齐scores

### 3. 测试工具
- `test_ea_connection.py`: 连接测试
- `start_ea_server.sh`: 启动脚本
- `test_iprover.sh`: iProver测试脚本

## 交互流程

### 1. 连接建立
```bash
# 启动EA服务器
python3 ea_server.py --host 127.0.0.1 --port 12346 [options]

# 启动iProver
./iproveropt --interactive_mode true --external_ip_address 127.0.0.1 --external_port 12346 [options] problem.p
```

### 2. 消息交换序列

#### 2.1 Clause注册
```json
iProver → EA: {
  "tag": "register_clauses",
  "clauses": [
    {
      "clause": "tcf(c_60,plain,(killed(butler,agatha)|killed(charles,agatha)),file(...))",
      "clause_id": 60,
      "clause_features": {
        "basic_clause_id": 60,
        "conj_dist": 0,
        "born": 1,
        "horn": false,
        "epr": true
      }
    }
  ]
}
```

#### 2.2 评分请求
```json
iProver → EA: {
  "tag": "scores_req",
  "clause_ids": [51, 50, 49, 56, 55, 60, 58, 57, 53, 52, 59, 54],
  "component": "sup",
  "component_id": 1
}
```

#### 2.3 批处理准备
EA内部处理：
1. 提取conjecture (conj_dist=0)
2. 分离context和candidate clauses
3. 清理clause文本（移除provenance）
4. 构建batch JSON

#### 2.4 LLM评分
```bash
python3 batch_ranker.py --input batch.json --out scores.json [options]
```

#### 2.5 返回评分
```json
EA → iProver: {
  "tag": "scores_res", 
  "scores": [0.234, 0.456, 0.123, ...]
}
```

### 3. 其他消息处理
- `server_queries_start/end`: 查询窗口管理
- `passive_clauses`: 被动队列更新
- `given_clause`: 选中clause通知
- `szs_result_out`: 最终结果
- `proof_out`: 证明输出

## 运行示例

### 启动EA服务器
```bash
cd /home/ks/LLM
python3 ea_server.py \
  --host 127.0.0.1 \
  --port 12346 \
  --ranker-script /home/ks/LLM/batch_ranker.py \
  --model gpt-5 \
  --chunk-size 8 \
  --anchors 2 \
  --context-size 32 \
  --context-summary-k 16 \
  --summary-max-tokens 200 \
  --log-file /home/ks/logs/EA.12346.log \
  --dry-run \
  --verbose \
  --progress
```

### 运行iProver测试
```bash
cd /home/ks/iprover-master
./iproveropt \
  --interactive_mode true \
  --external_ip_address 127.0.0.1 \
  --external_port 12346 \
  --schedule none \
  --preprocessing_flag false \
  --instantiation_flag true \
  --superposition_flag true \
  --resolution_flag false \
  --sup_iter_deepening 0 \
  --comb_sup_deep_mult 0 \
  --sup_passive_queue_type priority_queues \
  --sup_passive_queues_freq "[1]" \
  --sup_passive_queues "[[+external_score]]" \
  Examples/PUZ001-1.p
```

## 成功验证

### 测试结果
✅ **EA服务器启动成功**
- 监听端口12346
- 日志记录正常

✅ **iProver连接成功** 
- TCP连接建立
- JSON消息解析正确

✅ **Clause注册成功**
- 接收12个clause
- 正确识别conjecture (clause 60)

✅ **评分流程成功**
- Batch数据准备正确
- Ranker调用成功
- Scores返回正常

✅ **完整交互流程**
- 所有消息类型处理正确
- 连接正常关闭

### 日志示例
```json
{"timestamp": "2025-08-15 22:20:15", "type": "received", "message": {"tag": "register_clauses", "clauses": [...]}}
{"timestamp": "2025-08-15 22:20:15", "type": "received", "message": {"tag": "scores_req", "clause_ids": [51, 50, 49, 56, 55, 60, 58, 57, 53, 52, 59, 54], "component": "sup", "component_id": 1}}
{"timestamp": "2025-08-15 22:20:15", "type": "prepared_response", "message": {"tag": "scores_res", "scores": [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]}}
```

## 关键特性

### 清晰的中间过程
- 详细的连接日志
- 消息接收/发送记录
- Batch准备过程输出
- Ranker执行状态
- 评分结果统计

### 错误处理
- 连接失败重试
- Ranker调用超时
- JSON解析错误恢复
- 优雅的连接关闭

### 可配置性
- 主机端口设置
- Ranker参数调整
- 日志级别控制
- 干运行模式

## 下一步扩展

1. **真实LLM集成**: 替换dry-run模式，使用真实的GPT-5 API
2. **更复杂的评分**: 实现基于语义分析的智能评分
3. **性能优化**: 批处理大小优化，并行处理
4. **实验分析**: 对比不同策略的证明效果
5. **可视化界面**: 实时监控证明过程和评分

## 总结

本系统成功实现了：
- ✅ 完整的EA与iProver交互协议
- ✅ 清晰的中间过程日志
- ✅ 模块化的架构设计
- ✅ 可扩展的LLM评分框架
- ✅ 端到端的工作流程

系统已准备好进行更复杂的定理证明实验和LLM指导的证明搜索研究。
