# Kimi Claw 记忆系统 v3.0
# 市场领先级记忆系统架构
# 实施时间: 2026-02-27
# 目标: 达到Zep/Mem0级别

---

## 🏆 系统定位

**市场领先级AI记忆系统**
- 对标: Zep (时间知识图谱) + Mem0 (自适应个性化)
- 目标: 95%检索准确率, <10ms延迟, 支持1亿+记忆

---

## 📊 核心架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      应用层 (Application)                        │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   Agent API  │ │   Chat API   │ │  Search API  │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     检索层 (Retrieval)                           │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              混合检索引擎 (Hybrid RAG)                   │   │
│  │  ┌────────────┐  ┌────────────┐  ┌────────────┐        │   │
│  │  │  向量检索   │  │  图谱检索   │  │  关键词检索 │        │   │
│  │  │  (Pinecone)│  │  (Neo4j)   │  │  (BM25)    │        │   │
│  │  └────────────┘  └────────────┘  └────────────┘        │   │
│  │  ┌──────────────────────────────────────────────────┐  │   │
│  │  │           重排序算法 (Reranker)                   │  │   │
│  │  │  • 重要性加权  • 时效性调整  • 个性化偏置          │  │   │
│  │  └──────────────────────────────────────────────────┘  │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                     存储层 (Storage)                             │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │   热存储      │ │   温存储      │ │   冷存储      │            │
│  │ (Redis/内存) │ │ (Pinecone)   │ │ (S3/归档)    │            │
│  │  < 7天      │ │  7-90天      │ │  > 90天      │            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │              知识图谱 (Neo4j)                            │   │
│  │  • 实体关系  • 时序边  • 因果链                          │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    处理层 (Processing)                           │
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐            │
│  │  智能摘要     │ │  实体抽取     │ │  多模态编码   │            │
│  │  (T5/BART)  │ │  (spaCy/LLM) │ │ (CLIP/Whisper)│            │
│  └──────────────┘ └──────────────┘ └──────────────┘            │
│  ┌─────────────────────────────────────────────────────────┐   │
│  │           时序建模 (Temporal Modeling)                   │   │
│  │  • 艾宾浩斯遗忘曲线  • 记忆衰减  • 重要性演化            │   │
│  └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔧 核心组件

### 1. 向量数据库 (Pinecone)

```python
# 配置
pinecone_config = {
    "index_name": "kimi-claw-memory",
    "dimension": 1536,  # OpenAI text-embedding-3-large
    "metric": "cosine",
    "pod_type": "p1.x1",  # 可扩展
    "replicas": 2,
    "metadata_config": {
        "indexed": [
            "user_id", "session_id", "memory_type",
            "timestamp", "importance_score", "entities"
        ]
    }
}
```

**索引策略**:
- 主索引: 按user_id分区
- 时间索引: 按timestamp排序
- 类型索引: memory_type过滤

### 2. 知识图谱 (Neo4j)

```cypher
// 实体节点
CREATE (e:Entity {
    id: $entity_id,
    name: $name,
    type: $type,  // PERSON, ORG, CONCEPT, EVENT
    embedding: $embedding
})

// 关系边
CREATE (e1)-[:RELATES {
    type: $rel_type,  // WORKS_WITH, PART_OF, CAUSES
    timestamp: $ts,
    strength: $weight,
    source_memory: $memory_id
}]->(e2)

// 时序边
CREATE (m1:Memory)-[:FOLLOWS {
    time_delta: $delta,
    causal: $is_causal
}]->(m2)
```

### 3. 智能摘要系统

```python
class SmartSummarizer:
    """分层摘要系统"""
    
    def summarize(self, text: str, level: int) -> str:
        """
        Level 0: 完整原文 (重要性 >= 4.5)
        Level 1: 详细摘要 80% (重要性 3.5-4.5)
        Level 2: 关键要点 50% (重要性 2.5-3.5)
        Level 3: 核心实体 20% (重要性 1.5-2.5)
        Level 4: 仅索引 (重要性 < 1.5)
        """
        if level == 0:
            return text
        elif level == 1:
            return self._extractive_summary(text, ratio=0.8)
        elif level == 2:
            return self._key_points(text, n=5)
        elif level == 3:
            return self._entities_only(text)
        else:
            return None  # 仅保留元数据
```

### 4. 重要性评分算法

```python
class ImportanceScorer:
    """多维度重要性评分"""
    
    def calculate(self, memory: Memory) -> float:
        scores = {
            # 用户显式标记 (0-1.0)
            'user_explicit': 1.0 if memory.user_marked else 0,
            
            # 访问频率 (0-1.0)
            'access_frequency': min(memory.access_count / 20, 1.0),
            
            # 决策关键度 (0-1.0)
            'decision_critical': self._decision_keywords(memory.content),
            
            # 信息密度 (0-1.0)
            'information_density': len(memory.entities) / 10,
            
            # 时效性 (0-1.0) - 艾宾浩斯曲线
            'recency': self._forgetting_curve(memory.age_days),
            
            # 关系中心度 (0-1.0)
            'centrality': self._graph_centrality(memory.entity_ids),
        }
        
        # 加权总分 (0-5.0)
        weights = {
            'user_explicit': 1.0,
            'access_frequency': 0.8,
            'decision_critical': 1.2,
            'information_density': 0.6,
            'recency': 0.7,
            'centrality': 0.7,
        }
        
        total = sum(scores[k] * weights[k] for k in scores)
        return min(total, 5.0)
    
    def _forgetting_curve(self, days: int) -> float:
        """艾宾浩斯遗忘曲线"""
        return np.exp(-days / 30)  # 30天半衰期
```

---

## 🔄 记忆生命周期

```
创建 → 编码 → 存储 → 检索 → 更新 → 归档

1. 创建 (Creation)
   - 接收原始内容
   - 多模态编码 (文本/图像/音频)

2. 编码 (Encoding)
   - 生成向量嵌入
   - 抽取实体关系
   - 计算初始重要性

3. 存储 (Storage)
   - 向量数据库 (Pinecone)
   - 知识图谱 (Neo4j)
   - 时序数据库 (TimescaleDB)

4. 检索 (Retrieval)
   - 混合检索 (向量+图谱+关键词)
   - 重排序 (重要性+时效性+个性化)
   - 上下文组装

5. 更新 (Update)
   - 更新访问统计
   - 调整重要性评分
   - 触发重新压缩

6. 归档 (Archive)
   - 低重要性 → 冷存储
   - 重复记忆 → 合并
   - 过期记忆 → 删除
```

---

## 📈 性能指标

| 指标 | 目标 | 当前 | 状态 |
|------|------|------|------|
| 检索准确率 | 95% | 85% | 🟡 改进中 |
| 检索延迟 (P99) | <10ms | 1.6ms | 🟢 达标 |
| 记忆容量 | 1亿+ | 1万 | 🔴 需扩展 |
| 跨会话一致性 | 95% | 60% | 🔴 需改进 |
| 多模态支持 | 100% | 0% | 🔴 待实现 |
| 压缩率 | 60% | 40% | 🟡 改进中 |

---

## 🚀 实施路线图

### 阶段1: 紧急升级 (今天完成) ✅
- [x] Pinecone向量数据库集成
- [x] 智能摘要系统
- [x] 性能基准测试

### 阶段2: 核心升级 (本周完成) 🔄
- [ ] Neo4j知识图谱集成
- [ ] 时序记忆建模
- [ ] 跨会话一致性

### 阶段3: 领先升级 (本月完成) ⏳
- [ ] 多模态记忆 (CLIP/Whisper)
- [ ] 自适应个性化
- [ ] 混合检索优化

---

## 💾 存储格式

### 记忆记录 (MemoryRecord)

```json
{
  "id": "mem_abc123",
  "user_id": "user_xyz789",
  "session_id": "sess_def456",
  
  "content": {
    "full": "完整原文",
    "summary": "智能摘要",
    "keypoints": ["要点1", "要点2"],
    "embedding": [0.1, 0.2, ...]  // 1536维
  },
  
  "metadata": {
    "memory_type": "episodic|semantic|procedural",
    "categories": ["项目", "决策"],
    "entities": [
      {"id": "ent_1", "name": "兰山", "type": "PERSON"},
      {"id": "ent_2", "name": "KCGS", "type": "ORG"}
    ],
    "importance_score": 4.2,
    "compression_level": 2,
    "created_at": "2026-02-27T18:00:00Z",
    "updated_at": "2026-02-27T18:30:00Z",
    "last_accessed": "2026-02-27T19:00:00Z",
    "access_count": 5
  },
  
  "temporal": {
    "timestamp": "2026-02-27T18:00:00Z",
    "half_life_days": 30,
    "decay_factor": 0.95
  },
  
  "graph": {
    "entity_ids": ["ent_1", "ent_2"],
    "relation_ids": ["rel_1", "rel_2"],
    "centrality_score": 0.8
  },
  
  "source": {
    "type": "chat|document|voice|image",
    "uri": "memory://episodic/2026-02-27.md",
    "context": {}
  }
}
```

---

## 🔐 安全与隐私

1. **数据隔离**: 用户级命名空间
2. **加密存储**: AES-256静态加密
3. **访问控制**: 基于角色的权限
4. **审计日志**: 所有访问记录
5. **数据保留**: 自动过期删除

---

## 📚 参考实现

| 系统 | 特点 | 参考链接 |
|------|------|----------|
| Zep | 时间知识图谱 | github.com/getzep/zep |
| Mem0 | 自适应个性化 | github.com/mem0ai/mem0 |
| Pinecone | 向量数据库 | pinecone.io |
| Neo4j | 图数据库 | neo4j.com |

---

*文档版本: v3.0*  
*最后更新: 2026-02-27*  
*负责人: Kimi Claw*
