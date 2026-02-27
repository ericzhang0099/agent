# Mem0 + Zep (Graphiti) 核心算法速查表

## 1. 记忆压缩引擎 (Mem0风格)

### 核心流程
```
对话输入 → 提取事实 → 检索相关记忆 → LLM决策 → 执行操作
```

### 决策类型
| 操作 | 说明 | 场景 |
|------|------|------|
| ADD | 添加新记忆 | 新信息 |
| UPDATE | 更新现有记忆 | 信息变化 |
| DELETE | 删除记忆 | 信息过时 |
| NONE | 无操作 | 重复/无关信息 |

### 关键代码
```python
# 记忆压缩
compressor = MemoryCompressor(llm_client, vector_store, embedder)
decisions = compressor.compress(messages, user_id="user123")

# 执行决策
for d in decisions:
    if d.action == MemoryAction.ADD:
        store.add(d.content)
    elif d.action == MemoryAction.UPDATE:
        store.update(d.memory_id, d.content)
```

---

## 2. 时序知识图谱 (Zep风格)

### 三层架构
```
Community Layer (社区层)
    └── Label Propagation 算法
    └── 社区摘要 (Map-Reduce)

Semantic Layer (语义层)
    ├── Entity Nodes: 名称 + 摘要
    └── Entity Edges: 关系 + Fact + 时序

Episode Layer (原始层)
    └── 原始消息/文本 + 双向索引
```

### 双时序模型
| 时间线 | 字段 | 含义 |
|--------|------|------|
| 事件时间 T | valid_at / invalid_at | 事实实际有效时间 |
| 事务时间 T' | created_at / expired_at | 数据录入系统时间 |

### 关键代码
```python
# 初始化
tkg = TemporalKnowledgeGraph(neo4j_driver)

# 添加事实（自动处理冲突）
tkg.extract_and_add_fact(
    subject="用户", predicate="喜欢", object="中餐",
    fact_text="用户喜欢中餐",
    valid_at=datetime(2024, 6, 1)
)

# 时序查询（查询历史状态）
results = tkg.temporal_search(
    "用户偏好", 
    as_of=datetime(2024, 3, 1)  # 查询3月时的状态
)
```

---

## 3. 混合检索

### 三路召回
```
┌─────────────────┐
│   查询 Query     │
└────────┬────────┘
         │
    ┌────┼────┐
    ↓    ↓    ↓
┌──────┐┌──────┐┌──────┐
│向量搜索││全文搜索││图谱搜索│
│语义  ││关键词 ││关系  │
└──┬───┘└──┬───┘└──┬───┘
   └────┬──┘       │
        ↓          │
   ┌─────────┐     │
   │ RRF融合  │←────┘
   │ 重排序  │
   └────┬────┘
        ↓
   最终结果
```

### 重排序算法

#### RRF (Reciprocal Rank Fusion)
```python
score = Σ(1 / (k + rank))  # k=60
```

#### MMR (Maximal Marginal Relevance)
```python
MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected))
```

### 关键代码
```python
retriever = HybridRetriever(
    vector_store=vector_store,
    graph_store=graph_store,
    fulltext_index=fulltext
)

# 混合搜索
results = retriever.search(
    query="用户偏好",
    user_id="user123",
    use_vector=True,
    use_fulltext=True,
    use_graph=True,
    rerank_method="rrf"
)
```

---

## 4. Pinecone 最佳实践

### 索引设计
```python
# 混合索引（密集+稀疏向量）
index = pc.create_index(
    name="memory-index",
    dimension=1536,  # text-embedding-3-large
    metric="cosine"
)
```

### 记录结构
```python
{
    "_id": "user123#memory#2024-01-15#001",
    "chunk_text": "用户偏好...",
    "user_id": "user123",
    "category": "preference",
    "confidence": 0.95,
    "created_at": "2024-01-15T10:30:00Z"
}
```

### 混合搜索
```python
results = index.query(
    vector=dense_vector,      # 语义向量
    sparse_vector=sparse_vec,  # BM25稀疏向量
    filter={"user_id": {"$eq": "user123"}},
    alpha=0.5  # 0=纯关键词, 1=纯语义
)
```

---

## 5. 性能对比

| 指标 | Mem0 (向量) | Zep (图谱) | 混合方案 |
|------|-------------|------------|----------|
| 准确率 | 7.5% | 11.1% | 10%+ |
| 加载速度 | 基准 | 慢86.5% | 中等 |
| 查询延迟 | 低 | 中 | 中 |
| 成本 | 低 | 高40% | 中等 |
| 时序推理 | 弱 | 强 | 强 |
| 关系推理 | 弱 | 强 | 强 |

---

## 6. 快速集成清单

### 基础版（Mem0风格）
- [ ] 向量数据库（Pinecone/Qdrant）
- [ ] 嵌入模型（text-embedding-3-small）
- [ ] 记忆压缩逻辑
- [ ] 元数据过滤

### 进阶版（Zep风格）
- [ ] Neo4j图谱数据库
- [ ] 实体抽取Pipeline
- [ ] 时序事实管理
- [ ] 社区检测

### 完整版（混合）
- [ ] 三路召回（向量+全文+图谱）
- [ ] RRF/MMR重排序
- [ ] 上下文组装器
- [ ] 缓存层（Redis）

---

## 7. 关键Prompt模板

### 实体抽取
```
从对话中提取实体节点：
1. 始终提取说话者作为第一个节点
2. 提取其他重要实体、概念
3. 不要为关系或动作创建节点
4. 不要为时间信息创建节点
```

### 事实抽取
```
从消息中提取所有事实：
1. 仅在提供的实体之间提取事实
2. 每个事实代表两个不同节点之间的明确关系
3. relation_type: 简洁的大写描述
4. fact: 包含所有相关信息的详细描述
```

### 时序抽取
```
从事实中提取时间信息：
- valid_at: 关系成立的时间
- invalid_at: 关系结束的时间
使用ISO 8601格式，根据参考时间计算相对时间
```

---

## 8. 常见陷阱

1. **时序冲突**：新事实应与现有事实的时序范围不重叠
2. **实体消歧**：相似名称的实体需要合并
3. **记忆爆炸**：需要定期压缩和归档旧记忆
4. **检索质量**：单一检索方式召回率有限，需要混合
5. **成本控制**：LLM调用次数需要优化（使用轻量级模型做抽取）
