# 长期记忆压缩优化方案 v2.0
## 基于智能分层压缩与动态重要性评分

---

## 1. 现有系统问题分析

### 1.1 当前架构痛点

| 问题 | 影响 | 严重程度 |
|------|------|----------|
| 文本存储冗余 | 相同概念重复存储，空间浪费 | 🔴 高 |
| 缺乏重要性评估 | 所有记忆同等对待，检索噪音大 | 🔴 高 |
| 检索效率瓶颈 | 纯语义检索，缺乏关键词快速过滤 | 🟡 中 |
| 无自动压缩机制 | 记忆无限增长，无清理策略 | 🔴 高 |
| 上下文窗口限制 | 检索结果过多，超出LLM处理能力 | 🟡 中 |

### 1.2 数据流分析

```
当前流程:
用户输入 → 短期记忆 → (手动整理) → 情景记忆 → (手动整理) → 长期记忆

问题:
- 整理依赖人工，无法自动进行
- 无压缩机制，数据线性增长
- 无去重机制，相似内容重复存储
```

---

## 2. 优化目标

### 2.1 核心指标

| 指标 | 当前 | 目标 | 优化方向 |
|------|------|------|----------|
| 存储压缩率 | 0% | ≥60% | 去除冗余，保留关键 |
| 检索准确率 | ~70% | ≥90% | 语义+关键词混合 |
| 检索延迟 | ~200ms | ≤50ms | 分层索引+缓存 |
| 记忆重要性区分 | 无 | 5级评分 | 动态评分机制 |
| 自动压缩比例 | 0% | ≥80% | 智能压缩算法 |

### 2.2 设计原则

1. **最小高信号原则**: 只保留对当前任务最有价值的信息
2. **渐进式压缩**: 随时间自动降低细节保留度
3. **重要性驱动**: 高频访问、关键决策的记忆优先保留
4. **可逆压缩**: 保留原始数据索引，必要时可恢复

---

## 3. 优化架构设计

### 3.1 分层压缩存储架构

```
┌─────────────────────────────────────────────────────────────────┐
│                     记忆压缩存储架构 v2.0                          │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐      │
│  │  热数据层    │    │  温数据层    │    │  冷数据层    │      │
│  │  (7天内)     │    │  (30天内)    │    │  (永久)      │      │
│  │              │    │              │    │              │      │
│  │ • 完整文本   │    │ • 压缩摘要   │    │ • 核心要点   │      │
│  │ • 完整元数据 │    │ • 关键引用   │    │ • 知识图谱   │      │
│  │ • 高频访问   │    │ • 中等访问   │    │ • 低频访问   │      │
│  └──────┬───────┘    └──────┬───────┘    └──────┬───────┘      │
│         │                   │                   │              │
│         └───────────────────┼───────────────────┘              │
│                             ↓                                  │
│                   ┌─────────────────┐                          │
│                   │  智能压缩引擎   │                          │
│                   │  • 去重检测     │                          │
│                   │  • 摘要生成     │                          │
│                   │  • 重要性评分   │                          │
│                   │  • 知识提取     │                          │
│                   └─────────────────┘                          │
│                             │                                  │
│                             ↓                                  │
│                   ┌─────────────────┐                          │
│                   │  混合检索引擎   │                          │
│                   │  • 语义向量检索 │                          │
│                   │  • BM25关键词   │                          │
│                   │  • 重要性加权   │                          │
│                   │  • 时间衰减     │                          │
│                   └─────────────────┘                          │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 3.2 数据压缩流程

```
原始记忆
    ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 1: 去重检测                                              │
│ - 语义相似度 > 0.85 视为重复                                  │
│ - 合并重复内容，保留最新版本                                  │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 2: 重要性评分                                            │
│ - 访问频率 (30%)                                              │
│ - 决策关键度 (25%)                                            │
│ - 信息密度 (20%)                                              │
│ - 时效性 (15%)                                                │
│ - 用户显式标记 (10%)                                          │
└──────────────────────────────────────────────────────────────┘
    ↓
┌──────────────────────────────────────────────────────────────┐
│ Step 3: 分层压缩                                              │
│ - 高重要性(4-5分): 保留完整文本                               │
│ - 中重要性(2-3分): 生成结构化摘要                             │
│ - 低重要性(0-1分): 提取关键要点，归档到冷存储                 │
└──────────────────────────────────────────────────────────────┘
    ↓
压缩后记忆
```

---

## 4. 核心算法设计

### 4.1 记忆重要性评分算法

```python
class MemoryImportanceScorer:
    """
    记忆重要性评分算法
    
    评分维度:
    1. 访问频率 (access_frequency): 被检索的次数
    2. 决策关键度 (decision_criticality): 是否涉及关键决策
    3. 信息密度 (information_density): 单位字符的信息量
    4. 时效性 (recency): 越新的记忆分值越高（指数衰减）
    5. 用户标记 (user_marked): 用户显式标记的重要记忆
    """
    
    def calculate_importance(self, memory: MemoryRecord) -> float:
        """计算记忆重要性评分 (0-5分)"""
        
        # 1. 访问频率分数 (0-1.5)
        access_score = min(memory.access_count / 10, 1.5)
        
        # 2. 决策关键度 (0-1.25)
        decision_keywords = ['决策', '决定', '选择', '批准', '拒绝', '关键']
        decision_score = sum(1 for kw in decision_keywords if kw in memory.content) * 0.25
        decision_score = min(decision_score, 1.25)
        
        # 3. 信息密度 (0-1.0)
        # 基于实体数量、关键词密度计算
        entity_count = len(self._extract_entities(memory.content))
        keyword_density = len(set(self._extract_keywords(memory.content))) / len(memory.content)
        density_score = min((entity_count * 0.1 + keyword_density * 100), 1.0)
        
        # 4. 时效性 (0-0.75) - 指数衰减
        days_old = (datetime.now() - memory.created_at).days
        recency_score = 0.75 * np.exp(-days_old / 30)  # 30天半衰期
        
        # 5. 用户标记 (0-0.5)
        user_score = 0.5 if memory.user_marked_important else 0
        
        # 总分 (0-5)
        total_score = access_score + decision_score + density_score + recency_score + user_score
        
        return round(total_score, 2)
```

### 4.2 智能压缩算法

```python
class MemoryCompressor:
    """
    记忆智能压缩算法
    
    压缩策略基于重要性评分:
    - 5分: 保留完整文本 + 完整元数据
    - 4分: 保留完整文本 + 精简元数据
    - 3分: 生成结构化摘要 (保留60%信息)
    - 2分: 提取关键要点 (保留30%信息)
    - 1分: 提取核心实体和关系
    - 0分: 仅保留索引，归档到冷存储
    """
    
    def compress(self, memory: MemoryRecord, importance: float) -> CompressedMemory:
        """根据重要性评分压缩记忆"""
        
        if importance >= 4.0:
            # 高重要性: 完整保留
            return self._preserve_full(memory)
        
        elif importance >= 3.0:
            # 中高重要性: 结构化摘要
            return self._generate_summary(memory, ratio=0.6)
        
        elif importance >= 2.0:
            # 中等重要性: 关键要点
            return self._extract_key_points(memory)
        
        elif importance >= 1.0:
            # 低重要性: 核心实体
            return self._extract_entities_only(memory)
        
        else:
            # 极低重要性: 仅保留索引
            return self._create_index_only(memory)
    
    def _generate_summary(self, memory: MemoryRecord, ratio: float) -> CompressedMemory:
        """生成结构化摘要"""
        # 使用提取式摘要算法
        sentences = self._split_sentences(memory.content)
        
        # 计算句子重要性
        sentence_scores = {}
        for i, sent in enumerate(sentences):
            score = self._calculate_sentence_importance(sent, memory.content)
            sentence_scores[i] = score
        
        # 选择Top-K句子
        k = max(1, int(len(sentences) * ratio))
        top_sentences = sorted(sentence_scores.items(), key=lambda x: x[1], reverse=True)[:k]
        top_sentences.sort(key=lambda x: x[0])  # 按原文顺序排列
        
        summary = ' '.join([sentences[i] for i, _ in top_sentences])
        
        return CompressedMemory(
            original_id=memory.id,
            compressed_content=summary,
            compression_ratio=len(summary) / len(memory.content),
            key_entities=self._extract_entities(memory.content),
            importance=memory.importance_score,
            compression_level='summary'
        )
```

### 4.3 混合检索算法

```python
class HybridRetriever:
    """
    语义 + 关键词 + 重要性 混合检索
    
    检索流程:
    1. 关键词预过滤 (快速缩小候选集)
    2. 语义相似度计算 (精确匹配)
    3. 重要性加权 (优先高价值记忆)
    4. 时间衰减调整 (平衡新旧记忆)
    """
    
    def retrieve(self, query: str, top_k: int = 5) -> List[RetrievalResult]:
        """混合检索主函数"""
        
        # Step 1: 关键词预过滤 (使用倒排索引)
        query_keywords = self._extract_keywords(query)
        candidate_ids = self._keyword_filter(query_keywords, max_candidates=100)
        
        # Step 2: 语义相似度计算
        query_embedding = self.embedding_model.encode(query)
        semantic_scores = {}
        for mem_id in candidate_ids:
            mem_embedding = self.get_embedding(mem_id)
            similarity = cosine_similarity(query_embedding, mem_embedding)
            semantic_scores[mem_id] = similarity
        
        # Step 3: 重要性加权
        weighted_scores = {}
        for mem_id, sem_score in semantic_scores.items():
            importance = self.get_importance(mem_id)
            # 重要性加权: 高重要性记忆获得额外加成
            weighted_scores[mem_id] = sem_score * (1 + importance * 0.2)
        
        # Step 4: 时间衰减调整
        final_scores = {}
        for mem_id, score in weighted_scores.items():
            age_days = self.get_memory_age(mem_id)
            time_decay = np.exp(-age_days / 60)  # 60天半衰期
            # 平衡新旧: 保留70%原始分，30%时间衰减
            final_scores[mem_id] = score * 0.7 + score * time_decay * 0.3
        
        # 返回Top-K结果
        top_results = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        return [self._format_result(mem_id, score) for mem_id, score in top_results]
```

---

## 5. 存储结构优化

### 5.1 优化后的记忆记录结构

```python
@dataclass
class OptimizedMemoryRecord:
    """优化后的记忆记录结构"""
    
    # 基础标识
    id: str                          # 唯一标识
    parent_id: Optional[str]         # 父记忆ID（用于版本追踪）
    
    # 内容（分层存储）
    content_full: Optional[str]      # 完整内容（热数据层）
    content_summary: Optional[str]   # 摘要（温数据层）
    content_keypoints: List[str]     # 关键要点（冷数据层）
    
    # 压缩元数据
    compression_level: int           # 压缩级别 0-5
    compression_ratio: float         # 压缩比例
    original_length: int             # 原始长度
    compressed_length: int           # 压缩后长度
    
    # 重要性评分
    importance_score: float          # 0-5分
    importance_factors: Dict         # 各维度得分详情
    
    # 访问统计
    access_count: int                # 访问次数
    last_accessed: datetime          # 最后访问时间
    access_pattern: List[datetime]   # 访问时间序列
    
    # 分类标签
    memory_type: str                 # 记忆类型
    categories: List[str]            # 分类标签
    entities: List[str]              # 提取的实体
    keywords: List[str]              # 关键词
    
    # 时间戳
    created_at: datetime             # 创建时间
    updated_at: datetime             # 更新时间
    compressed_at: Optional[datetime] # 压缩时间
    
    # 来源追踪
    source: str                      # 来源文件/系统
    context: Dict                    # 上下文信息
    
    # 用户交互
    user_marked_important: bool      # 用户显式标记
    user_notes: Optional[str]        # 用户备注
```

### 5.2 索引结构

```python
class MemoryIndex:
    """多维度记忆索引"""
    
    # 1. 向量索引 (语义检索)
    vector_index: Dict[str, np.ndarray]  # memory_id -> embedding
    
    # 2. 倒排索引 (关键词检索)
    inverted_index: Dict[str, Set[str]]  # keyword -> memory_ids
    
    # 3. 重要性索引
    importance_index: Dict[float, Set[str]]  # importance_score -> memory_ids
    
    # 4. 时间索引
    time_index: Dict[str, Set[str]]  # date_ymd -> memory_ids
    
    # 5. 类型索引
    type_index: Dict[str, Set[str]]  # memory_type -> memory_ids
    
    # 6. 实体索引
    entity_index: Dict[str, Set[str]]  # entity -> memory_ids
```

---

## 6. 实施路线图

### Phase 1: 基础优化 (1-2周)

- [x] 实现重要性评分算法
- [x] 实现基础压缩算法
- [ ] 集成到现有记忆系统
- [ ] 添加访问统计追踪

### Phase 2: 检索优化 (2-3周)

- [ ] 实现混合检索引擎
- [ ] 优化BM25关键词索引
- [ ] 添加检索结果排序优化
- [ ] 实现检索缓存机制

### Phase 3: 智能压缩 (3-4周)

- [ ] 实现自动压缩调度
- [ ] 添加去重检测
- [ ] 实现分层存储迁移
- [ ] 添加压缩效果评估

### Phase 4: 高级功能 (4-6周)

- [ ] 实现知识图谱提取
- [ ] 添加记忆关联分析
- [ ] 实现预测性预加载
- [ ] 添加可视化监控面板

---

## 7. 性能预期

### 7.1 压缩效果预期

| 记忆类型 | 原始大小 | 压缩后 | 压缩率 | 信息保留 |
|----------|----------|--------|--------|----------|
| 高重要性 | 100% | 100% | 0% | 100% |
| 中重要性 | 100% | 40% | 60% | 85% |
| 低重要性 | 100% | 15% | 85% | 60% |
| 归档记忆 | 100% | 5% | 95% | 30% |

### 7.2 检索性能预期

| 指标 | 优化前 | 优化后 | 提升 |
|------|--------|--------|------|
| 平均检索时间 | 200ms | 50ms | 4x |
| 检索准确率 | 70% | 90% | +28% |
| 存储空间 | 100% | 40% | -60% |
| 缓存命中率 | 0% | 75% | +75% |

---

## 8. 风险评估与缓解

| 风险 | 可能性 | 影响 | 缓解措施 |
|------|--------|------|----------|
| 过度压缩 | 中 | 高 | 保留原始数据索引，可恢复 |
| 重要性评分偏差 | 中 | 中 | 多维度评分，定期校准 |
| 检索质量下降 | 低 | 高 | A/B测试，逐步上线 |
| 系统复杂度增加 | 高 | 中 | 模块化设计，完善文档 |

---

## 9. 总结

本方案通过以下核心机制实现长期记忆压缩优化:

1. **智能重要性评分**: 多维度评估记忆价值，区分优先级
2. **分层压缩策略**: 根据重要性采用不同压缩级别
3. **混合检索引擎**: 语义+关键词+重要性加权，提升检索质量
4. **自动压缩调度**: 定期执行压缩，保持系统健康

预期效果:
- 存储空间减少 60%
- 检索速度提升 4x
- 检索准确率提升至 90%
- 系统可维护性显著改善
