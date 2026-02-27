# MEMORY.md v3.0 - Mem0个性化记忆 × Zep时序知识图谱 × Pinecone高性能向量检索融合版

> **文档等级**: 核心记忆系统 · 跨会话连续 · 智能检索  
> **技术架构**: Mem0个性化记忆 + Zep时序知识图谱 + Pinecone高性能向量检索  
> **记忆模型**: 四类记忆（情景/语义/程序/工作）  
> **检索机制**: 三重融合检索（向量+时序+图谱）  
> **关联文档**: SOUL.md v4.0, IDENTITY.md v4.0, USER.md v2.0

---

## 🧠 记忆系统架构概览

### 三层记忆架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        MEMORY.md v3.0 三层架构                           │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 3: 长期记忆 (Long-Term Memory)                            │   │
│  │  • 持久化存储 · 跨会话保持 · 人格核心                            │   │
│  │  • 技术: Mem0个性化记忆 + 本地文件系统                           │   │
│  │  • 内容: 核心价值观、用户画像、关系历史、演化轨迹                │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑ 压缩提炼                                 │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 2: 中期记忆 (Medium-Term Memory)                          │   │
│  │  • 时序知识图谱 · 事件关联 · 因果推理                            │   │
│  │  • 技术: Zep时序知识图谱                                         │   │
│  │  • 内容: 项目历史、决策记录、学习轨迹、情感事件                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                              ↑ 结构化提取                               │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │  Layer 1: 短期记忆 (Short-Term Memory)                           │   │
│  │  • 高性能向量检索 · 实时会话 · 上下文保持                        │   │
│  │  • 技术: Pinecone高性能向量检索                                  │   │
│  │  • 内容: 当前对话、最近事件、活跃项目、临时状态                  │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
│  ┌─────────────────────────────────────────────────────────────────┐   │
│  │                    三重融合检索引擎                              │   │
│  │  • 向量相似度检索 (Pinecone)                                     │   │
│  │  • 时序关系检索 (Zep)                                            │   │
│  │  • 知识图谱推理 (Zep Graph)                                      │   │
│  └─────────────────────────────────────────────────────────────────┘   │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 四类记忆模型

基于认知心理学和AI记忆研究的四类记忆模型：

| 记忆类型 | 英文 | 描述 | 存储技术 | 检索方式 | 保留时间 |
|----------|------|------|----------|----------|----------|
| **情景记忆** | Episodic | 具体事件和经历 | Mem0 + Zep时序 | 时间+内容检索 | 长期 |
| **语义记忆** | Semantic | 事实、概念、知识 | Pinecone向量 | 语义相似度 | 永久 |
| **程序记忆** | Procedural | 技能、流程、方法 | 代码+文档 | 标签检索 | 永久 |
| **工作记忆** | Working | 当前会话上下文 | 内存+Pinecone | 实时检索 | 会话级 |

---

## 💾 Mem0个性化记忆系统

### Mem0核心特性

Mem0是一个为AI助手和Agent设计的个性化记忆层，提供以下能力：

```yaml
mem0_features:
  # 多层次记忆存储
  storage_layers:
    - 向量数据库: "Pinecone/Weaviate"
    - 键值存储: "Redis"
    - 图数据库: "Neo4j"
    
  # 自适应个性化
  adaptive_personalization:
    user_preferences: "跨会话保持"
    adaptive_learning: "随交互改进"
    context_aware: "理解上下文"
    
  # 开发者友好
  developer_friendly:
    simple_api: "易集成"
    platform_integrations: "多平台支持"
    managed_service: "托管选项"
```

### Mem0记忆操作API

```python
# memory_system/mem0_adapter.py

from typing import List, Dict, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import hashlib
import json

@dataclass
class MemoryEntry:
    """记忆条目"""
    memory_id: str
    content: str
    memory_type: str  # episodic/semantic/procedural/working
    importance: float  # 0-1
    created_at: datetime
    last_accessed: datetime
    access_count: int
    metadata: Dict[str, Any]
    vector_embedding: Optional[List[float]] = None
    
@dataclass
class UserProfile:
    """用户画像"""
    user_id: str
    preferences: Dict[str, Any]
    interaction_history: List[Dict]
    memory_summary: str
    last_updated: datetime

class Mem0MemoryManager:
    """Mem0记忆管理器"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.vector_store = PineconeVectorStore(config['pinecone'])
        self.graph_store = ZepGraphStore(config['zep'])
        self.cache = RedisCache(config['redis'])
        
    # ========== 核心记忆操作 ==========
    
    async def add_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        metadata: Dict = None
    ) -> str:
        """
        添加新记忆
        
        Args:
            content: 记忆内容
            memory_type: 记忆类型 (episodic/semantic/procedural/working)
            importance: 重要性 (0-1)
            metadata: 附加元数据
            
        Returns:
            memory_id: 记忆唯一标识
        """
        # 生成记忆ID
        memory_id = self._generate_memory_id(content)
        
        # 生成向量嵌入
        vector_embedding = await self._embed(content)
        
        # 创建记忆条目
        entry = MemoryEntry(
            memory_id=memory_id,
            content=content,
            memory_type=memory_type,
            importance=importance,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            access_count=0,
            metadata=metadata or {},
            vector_embedding=vector_embedding
        )
        
        # 存储到向量数据库
        await self.vector_store.upsert(
            id=memory_id,
            vector=vector_embedding,
            metadata={
                "content": content,
                "type": memory_type,
                "importance": importance,
                "created_at": entry.created_at.isoformat()
            }
        )
        
        # 如果是情景记忆，添加到知识图谱
        if memory_type == "episodic":
            await self.graph_store.add_event(
                event_id=memory_id,
                content=content,
                timestamp=entry.created_at,
                entities=metadata.get("entities", []),
                relations=metadata.get("relations", [])
            )
        
        # 更新缓存
        await self.cache.set(f"memory:{memory_id}", entry)
        
        return memory_id
    
    async def search_memories(
        self,
        query: str,
        memory_type: Optional[str] = None,
        limit: int = 10,
        recency_weight: float = 0.3,
        importance_weight: float = 0.2
    ) -> List[MemoryEntry]:
        """
        搜索记忆
        
        Args:
            query: 搜索查询
            memory_type: 筛选记忆类型
            limit: 返回数量限制
            recency_weight: 时效性权重
            importance_weight: 重要性权重
            
        Returns:
            匹配的记忆条目列表
        """
        # 生成查询向量
        query_vector = await self._embed(query)
        
        # 向量相似度搜索
        vector_results = await self.vector_store.query(
            vector=query_vector,
            top_k=limit * 2,  # 获取更多候选
            filter={"type": memory_type} if memory_type else None
        )
        
        # 获取完整记忆条目
        memories = []
        for result in vector_results:
            memory_id = result.id
            entry = await self.cache.get(f"memory:{memory_id}")
            
            if not entry:
                # 从向量存储重建
                entry = MemoryEntry(
                    memory_id=memory_id,
                    content=result.metadata["content"],
                    memory_type=result.metadata["type"],
                    importance=result.metadata["importance"],
                    created_at=datetime.fromisoformat(result.metadata["created_at"]),
                    last_accessed=datetime.now(),
                    access_count=0,
                    metadata=result.metadata
                )
            
            # 计算综合分数
            similarity_score = result.score
            recency_score = self._calculate_recency(entry.created_at)
            importance_score = entry.importance
            
            entry.composite_score = (
                similarity_score * (1 - recency_weight - importance_weight) +
                recency_score * recency_weight +
                importance_score * importance_weight
            )
            
            memories.append(entry)
        
        # 按综合分数排序
        memories.sort(key=lambda x: x.composite_score, reverse=True)
        
        # 更新访问统计
        for memory in memories[:limit]:
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            await self.cache.set(f"memory:{memory.memory_id}", memory)
        
        return memories[:limit]
    
    async def get_related_memories(
        self,
        memory_id: str,
        relation_types: List[str] = None,
        depth: int = 2
    ) -> List[MemoryEntry]:
        """
        获取相关记忆（基于知识图谱）
        
        Args:
            memory_id: 起始记忆ID
            relation_types: 关系类型筛选
            depth: 搜索深度
            
        Returns:
            相关的记忆条目列表
        """
        # 从知识图谱获取相关事件
        related_events = await self.graph_store.get_related_events(
            event_id=memory_id,
            relation_types=relation_types,
            depth=depth
        )
        
        # 获取完整记忆条目
        memories = []
        for event in related_events:
            entry = await self.cache.get(f"memory:{event['event_id']}")
            if entry:
                memories.append(entry)
        
        return memories
    
    async def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        importance: Optional[float] = None,
        metadata: Optional[Dict] = None
    ) -> bool:
        """
        更新记忆
        
        Args:
            memory_id: 记忆ID
            content: 新内容（可选）
            importance: 新重要性（可选）
            metadata: 更新的元数据（可选）
            
        Returns:
            是否成功
        """
        # 获取现有记忆
        entry = await self.cache.get(f"memory:{memory_id}")
        if not entry:
            return False
        
        # 更新字段
        if content:
            entry.content = content
            entry.vector_embedding = await self._embed(content)
        
        if importance is not None:
            entry.importance = importance
        
        if metadata:
            entry.metadata.update(metadata)
        
        entry.last_accessed = datetime.now()
        
        # 更新存储
        await self.vector_store.upsert(
            id=memory_id,
            vector=entry.vector_embedding,
            metadata={
                "content": entry.content,
                "type": entry.memory_type,
                "importance": entry.importance,
                "created_at": entry.created_at.isoformat()
            }
        )
        
        await self.cache.set(f"memory:{memory_id}", entry)
        
        return True
    
    async def delete_memory(self, memory_id: str) -> bool:
        """
        删除记忆
        
        Args:
            memory_id: 记忆ID
            
        Returns:
            是否成功
        """
        # 从向量存储删除
        await self.vector_store.delete(memory_id)
        
        # 从知识图谱删除
        await self.graph_store.delete_event(memory_id)
        
        # 从缓存删除
        await self.cache.delete(f"memory:{memory_id}")
        
        return True
    
    async def get_user_profile(self, user_id: str) -> UserProfile:
        """
        获取用户画像
        
        Args:
            user_id: 用户ID
            
        Returns:
            用户画像
        """
        # 从缓存获取
        profile = await self.cache.get(f"profile:{user_id}")
        
        if not profile:
            # 从记忆聚合生成
            user_memories = await self.search_memories(
                query=f"user:{user_id}",
                limit=100
            )
            
            profile = self._aggregate_profile(user_id, user_memories)
            await self.cache.set(f"profile:{user_id}", profile)
        
        return profile
    
    # ========== 辅助方法 ==========
    
    def _generate_memory_id(self, content: str) -> str:
        """生成记忆ID"""
        hash_input = f"{content}:{datetime.now().isoformat()}"
        return hashlib.sha256(hash_input.encode()).hexdigest()[:16]
    
    async def _embed(self, text: str) -> List[float]:
        """生成文本向量嵌入"""
        # 使用嵌入模型（如OpenAI text-embedding-3-large）
        # 实际实现中调用嵌入API
        pass
    
    def _calculate_recency(self, created_at: datetime) -> float:
        """计算时效性分数"""
        age_days = (datetime.now() - created_at).days
        # 指数衰减
        return max(0, 1 - (age_days / 365))
    
    def _aggregate_profile(
        self,
        user_id: str,
        memories: List[MemoryEntry]
    ) -> UserProfile:
        """聚合用户画像"""
        # 提取偏好
        preferences = {}
        for memory in memories:
            if "preference" in memory.metadata:
                pref_type = memory.metadata["preference_type"]
                preferences[pref_type] = memory.content
        
        # 生成摘要
        summary = f"用户 {user_id} 的交互历史包含 {len(memories)} 条记忆"
        
        return UserProfile(
            user_id=user_id,
            preferences=preferences,
            interaction_history=[m.to_dict() for m in memories[-10:]],
            memory_summary=summary,
            last_updated=datetime.now()
        )
```

### 记忆类型定义

```python
# memory_system/memory_types.py

from enum import Enum
from typing import Dict, List, Optional
from dataclasses import dataclass
from datetime import datetime

class MemoryType(Enum):
    """记忆类型枚举"""
    EPISODIC = "episodic"      # 情景记忆：具体事件
    SEMANTIC = "semantic"      # 语义记忆：事实知识
    PROCEDURAL = "procedural"  # 程序记忆：技能流程
    WORKING = "working"        # 工作记忆：当前上下文

@dataclass
class EpisodicMemory:
    """情景记忆：具体事件和经历"""
    event_id: str
    timestamp: datetime
    location: Optional[str]
    participants: List[str]
    description: str
    emotional_valence: float  # -1 to 1
    emotional_arousal: float  # 0 to 1
    importance: float
    
    # 时序关联
    previous_event: Optional[str]
    next_event: Optional[str]
    related_events: List[str]

@dataclass
class SemanticMemory:
    """语义记忆：事实、概念、知识"""
    concept_id: str
    concept_type: str  # person/place/thing/fact/rule
    name: str
    description: str
    attributes: Dict[str, any]
    confidence: float  # 0 to 1
    source: Optional[str]
    
    # 知识图谱关联
    related_concepts: List[str]
    category: Optional[str]

@dataclass
class ProceduralMemory:
    """程序记忆：技能、流程、方法"""
    skill_id: str
    skill_name: str
    description: str
    steps: List[str]
    prerequisites: List[str]
    success_rate: float
    last_used: datetime
    usage_count: int
    
    # 元数据
    difficulty: str  # beginner/intermediate/advanced/expert
    estimated_time: int  # minutes
    tools_required: List[str]

@dataclass
class WorkingMemory:
    """工作记忆：当前会话上下文"""
    session_id: str
    current_topic: Optional[str]
    active_goals: List[str]
    recent_context: List[str]  # 最近N轮对话
    pending_tasks: List[str]
    
    # 临时状态
    user_intent: Optional[str]
    expected_response: Optional[str]
    emotional_state: Optional[str]
```

---

## 🕸️ Zep时序知识图谱

### Zep核心特性

Zep是一个为AI助手设计的长期记忆服务，提供时序知识图谱能力：

```yaml
zep_features:
  # 时序记忆
  temporal_memory:
    session_history: "完整会话历史"
    timeline_view: "时间线视图"
    event_chains: "事件链追踪"
    
  # 知识图谱
  knowledge_graph:
    entity_extraction: "自动实体提取"
    relation_inference: "关系推理"
    graph_traversal: "图遍历查询"
    
  # 记忆增强
  memory_enhancement:
    summarization: "自动摘要"
    importance_scoring: "重要性评分"
    decay_management: "记忆衰减管理"
```

### Zep知识图谱实现

```python
# memory_system/zep_adapter.py

from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass
from datetime import datetime
from enum import Enum

class RelationType(Enum):
    """关系类型"""
    CAUSES = "causes"           # 因果关系
    FOLLOWS = "follows"         # 时序关系
    RELATED_TO = "related_to"   # 相关关系
    PART_OF = "part_of"         # 部分关系
    CONTRADICTS = "contradicts" # 矛盾关系
    SIMILAR_TO = "similar_to"   # 相似关系

@dataclass
class Entity:
    """知识图谱实体"""
    entity_id: str
    name: str
    entity_type: str  # person/organization/location/concept/event
    attributes: Dict[str, any]
    first_seen: datetime
    last_seen: datetime
    mention_count: int

@dataclass
class Relation:
    """知识图谱关系"""
    relation_id: str
    source_id: str
    target_id: str
    relation_type: RelationType
    strength: float  # 0 to 1
    evidence: List[str]  # 支持证据
    first_observed: datetime
    last_observed: datetime

@dataclass
class Event:
    """时序事件"""
    event_id: str
    timestamp: datetime
    description: str
    event_type: str
    entities_involved: List[str]
    emotional_valence: float
    importance: float
    
    # 时序关联
    previous_events: List[str]
    next_events: List[str]
    concurrent_events: List[str]

class ZepGraphStore:
    """Zep知识图谱存储"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.events: Dict[str, Event] = {}
        
    # ========== 实体管理 ==========
    
    async def extract_entities(self, text: str) -> List[Entity]:
        """
        从文本中提取实体
        
        Args:
            text: 输入文本
            
        Returns:
            提取的实体列表
        """
        # 使用NLP模型提取实体
        # 实际实现中调用NER模型
        extracted = []
        
        # 示例：识别人名、组织、地点等
        # 这里简化处理，实际使用spaCy或transformers
        
        return extracted
    
    async def add_entity(self, entity: Entity) -> str:
        """
        添加实体到知识图谱
        
        Args:
            entity: 实体对象
            
        Returns:
            实体ID
        """
        # 检查是否已存在
        existing = await self._find_similar_entity(entity)
        
        if existing:
            # 合并信息
            existing.last_seen = datetime.now()
            existing.mention_count += 1
            existing.attributes.update(entity.attributes)
            return existing.entity_id
        
        # 添加新实体
        self.entities[entity.entity_id] = entity
        return entity.entity_id
    
    async def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    async def search_entities(
        self,
        name: str,
        entity_type: Optional[str] = None
    ) -> List[Entity]:
        """
        搜索实体
        
        Args:
            name: 实体名称
            entity_type: 实体类型筛选
            
        Returns:
            匹配的实体列表
        """
        results = []
        
        for entity in self.entities.values():
            if name.lower() in entity.name.lower():
                if entity_type is None or entity.entity_type == entity_type:
                    results.append(entity)
        
        return results
    
    # ========== 关系管理 ==========
    
    async def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: RelationType,
        strength: float = 1.0,
        evidence: List[str] = None
    ) -> str:
        """
        添加关系到知识图谱
        
        Args:
            source_id: 源实体ID
            target_id: 目标实体ID
            relation_type: 关系类型
            strength: 关系强度
            evidence: 支持证据
            
        Returns:
            关系ID
        """
        relation_id = f"{source_id}_{relation_type.value}_{target_id}"
        
        # 检查是否已存在
        if relation_id in self.relations:
            # 更新强度
            existing = self.relations[relation_id]
            existing.strength = max(existing.strength, strength)
            existing.last_observed = datetime.now()
            if evidence:
                existing.evidence.extend(evidence)
            return relation_id
        
        # 创建新关系
        relation = Relation(
            relation_id=relation_id,
            source_id=source_id,
            target_id=target_id,
            relation_type=relation_type,
            strength=strength,
            evidence=evidence or [],
            first_observed=datetime.now(),
            last_observed=datetime.now()
        )
        
        self.relations[relation_id] = relation
        return relation_id
    
    async def get_relations(
        self,
        entity_id: str,
        relation_type: Optional[RelationType] = None,
        direction: str = "both"  # out/in/both
    ) -> List[Relation]:
        """
        获取实体的关系
        
        Args:
            entity_id: 实体ID
            relation_type: 关系类型筛选
            direction: 方向筛选
            
        Returns:
            关系列表
        """
        results = []
        
        for relation in self.relations.values():
            # 方向匹配
            if direction in ["out", "both"] and relation.source_id == entity_id:
                match = True
            elif direction in ["in", "both"] and relation.target_id == entity_id:
                match = True
            else:
                match = False
            
            # 类型匹配
            if match and (relation_type is None or relation.relation_type == relation_type):
                results.append(relation)
        
        return results
    
    # ========== 事件管理 ==========
    
    async def add_event(
        self,
        event_id: str,
        content: str,
        timestamp: datetime,
        entities: List[str],
        relations: List[Dict]
    ) -> str:
        """
        添加事件到知识图谱
        
        Args:
            event_id: 事件ID
            content: 事件内容
            timestamp: 时间戳
            entities: 涉及实体
            relations: 事件关系
            
        Returns:
            事件ID
        """
        # 创建事件
        event = Event(
            event_id=event_id,
            timestamp=timestamp,
            description=content,
            event_type="user_interaction",
            entities_involved=entities,
            emotional_valence=0.0,  # 需要情感分析
            importance=0.5,
            previous_events=[],
            next_events=[],
            concurrent_events=[]
        )
        
        self.events[event_id] = event
        
        # 建立时序关系
        await self._link_temporal_events(event_id, timestamp)
        
        return event_id
    
    async def get_related_events(
        self,
        event_id: str,
        relation_types: List[str] = None,
        depth: int = 2
    ) -> List[Dict]:
        """
        获取相关事件
        
        Args:
            event_id: 起始事件ID
            relation_types: 关系类型筛选
            depth: 搜索深度
            
        Returns:
            相关事件列表
        """
        if event_id not in self.events:
            return []
        
        visited = {event_id}
        queue = [(event_id, 0)]
        results = []
        
        while queue:
            current_id, current_depth = queue.pop(0)
            
            if current_depth >= depth:
                continue
            
            event = self.events.get(current_id)
            if not event:
                continue
            
            # 收集关联事件
            related_ids = (
                event.previous_events +
                event.next_events +
                event.concurrent_events
            )
            
            for related_id in related_ids:
                if related_id not in visited:
                    visited.add(related_id)
                    queue.append((related_id, current_depth + 1))
                    
                    related_event = self.events.get(related_id)
                    if related_event:
                        results.append({
                            "event_id": related_id,
                            "description": related_event.description,
                            "timestamp": related_event.timestamp,
                            "relation_depth": current_depth + 1
                        })
        
        return results
    
    async def get_timeline(
        self,
        start_time: datetime,
        end_time: datetime,
        entity_filter: List[str] = None
    ) -> List[Event]:
        """
        获取时间线视图
        
        Args:
            start_time: 开始时间
            end_time: 结束时间
            entity_filter: 实体筛选
            
        Returns:
            事件列表（按时间排序）
        """
        events = []
        
        for event in self.events.values():
            if start_time <= event.timestamp <= end_time:
                if entity_filter is None or any(
                    e in event.entities_involved for e in entity_filter
                ):
                    events.append(event)
        
        # 按时间排序
        events.sort(key=lambda x: x.timestamp)
        
        return events
    
    # ========== 辅助方法 ==========
    
    async def _find_similar_entity(self, entity: Entity) -> Optional[Entity]:
        """查找相似实体"""
        for existing in self.entities.values():
            if (existing.name.lower() == entity.name.lower() and
                existing.entity_type == entity.entity_type):
                return existing
        return None
    
    async def _link_temporal_events(
        self,
        event_id: str,
        timestamp: datetime
    ):
        """建立时序关联"""
        # 查找时间上接近的事件
        time_window = 3600  # 1小时
        
        for other_id, other_event in self.events.items():
            if other_id == event_id:
                continue
            
            time_diff = abs((timestamp - other_event.timestamp).total_seconds())
            
            if time_diff < time_window:
                # 建立关联
                if timestamp > other_event.timestamp:
                    # 当前事件在后
                    if other_id not in self.events[event_id].previous_events:
                        self.events[event_id].previous_events.append(other_id)
                    if event_id not in self.events[other_id].next_events:
                        self.events[other_id].next_events.append(event_id)
                else:
                    # 当前事件在前
                    if other_id not in self.events[event_id].next_events:
                        self.events[event_id].next_events.append(other_id)
                    if event_id not in self.events[other_id].previous_events:
                        self.events[other_id].previous_events.append(event_id)
```

---

## 🔍 Pinecone高性能向量检索

### Pinecone核心特性

Pinecone是一个托管的向量数据库，提供高性能向量检索：

```yaml
pinecone_features:
  # 高性能
  performance:
    low_latency: "< 100ms 查询延迟"
    high_throughput: "数千 QPS"
    hybrid_search: "稠密+稀疏向量"
    
  # 可扩展性
  scalability:
    automatic_scaling: "自动扩缩容"
    billion_vectors: "支持十亿级向量"
    metadata_filtering: "元数据过滤"
    
  # 企业级
  enterprise:
    multi_tenancy: "多租户隔离"
    security: "SOC2合规"
    uptime_sla: "99.99%可用性"
```

### Pinecone向量存储实现

```python
# memory_system/pinecone_adapter.py

from typing import List, Dict, Optional, Any
import pinecone
from dataclasses import dataclass

@dataclass
class VectorRecord:
    """向量记录"""
    id: str
    vector: List[float]
    metadata: Dict[str, Any]
    score: Optional[float] = None

class PineconeVectorStore:
    """Pinecone向量存储"""
    
    def __init__(self, config: Dict):
        self.config = config
        self.index_name = config.get("index_name", "memory")
        self.dimension = config.get("dimension", 1536)
        self.metric = config.get("metric", "cosine")
        
        # 初始化Pinecone
        pinecone.init(
            api_key=config["api_key"],
            environment=config["environment"]
        )
        
        # 创建或连接索引
        if self.index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric
            )
        
        self.index = pinecone.Index(self.index_name)
    
    async def upsert(
        self,
        id: str,
        vector: List[float],
        metadata: Dict[str, Any]
    ) -> bool:
        """
        插入或更新向量
        
        Args:
            id: 向量ID
            vector: 向量数据
            metadata: 元数据
            
        Returns:
            是否成功
        """
        try:
            self.index.upsert(vectors=[(id, vector, metadata)])
            return True
        except Exception as e:
            print(f"Upsert error: {e}")
            return False
    
    async def upsert_batch(
        self,
        records: List[VectorRecord]
    ) -> bool:
        """
        批量插入向量
        
        Args:
            records: 向量记录列表
            
        Returns:
            是否成功
        """
        try:
            vectors = [
                (r.id, r.vector, r.metadata)
                for r in records
            ]
            self.index.upsert(vectors=vectors)
            return True
        except Exception as e:
            print(f"Batch upsert error: {e}")
            return False
    
    async def query(
        self,
        vector: List[float],
        top_k: int = 10,
        filter: Optional[Dict] = None,
        include_metadata: bool = True
    ) -> List[VectorRecord]:
        """
        向量相似度查询
        
        Args:
            vector: 查询向量
            top_k: 返回数量
            filter: 元数据过滤条件
            include_metadata: 是否包含元数据
            
        Returns:
            匹配的向量记录列表
        """
        try:
            results = self.index.query(
                vector=vector,
                top_k=top_k,
                filter=filter,
                include_metadata=include_metadata
            )
            
            records = []
            for match in results["matches"]:
                records.append(VectorRecord(
                    id=match["id"],
                    vector=[],  # 查询结果不包含向量
                    metadata=match.get("metadata", {}),
                    score=match["score"]
                ))
            
            return records
        except Exception as e:
            print(f"Query error: {e}")
            return []
    
    async def delete(self, id: str) -> bool:
        """
        删除向量
        
        Args:
            id: 向量ID
            
        Returns:
            是否成功
        """
        try:
            self.index.delete(ids=[id])
            return True
        except Exception as e:
            print(f"Delete error: {e}")
            return False
    
    async def fetch(self, id: str) -> Optional[VectorRecord]:
        """
        获取向量
        
        Args:
            id: 向量ID
            
        Returns:
            向量记录
        """
        try:
            result = self.index.fetch(ids=[id])
            
            if id in result["vectors"]:
                vector_data = result["vectors"][id]
                return VectorRecord(
                    id=id,
                    vector=vector_data["values"],
                    metadata=vector_data.get("metadata", {})
                )
            
            return None
        except Exception as e:
            print(f"Fetch error: {e}")
            return None
    
    async def update_metadata(
        self,
        id: str,
        metadata: Dict[str, Any]
    ) -> bool:
        """
        更新向量元数据
        
        Args:
            id: 向量ID
            metadata: 新元数据
            
        Returns:
            是否成功
        """
        try:
            # Pinecone不直接支持元数据更新，需要重新upsert
            existing = await self.fetch(id)
            if existing:
                await self.upsert(id, existing.vector, metadata)
                return True
            return False
        except Exception as e:
            print(f"Update metadata error: {e}")
            return False
    
    def get_stats(self) -> Dict[str, Any]:
        """获取索引统计信息"""
        return self.index.describe_index_stats()
```

---

## 🔄 三重融合检索机制

### 融合检索架构

```
┌─────────────────────────────────────────────────────────────────────────┐
│                        三重融合检索引擎                                  │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────────┐         │
│  │  向量相似度检索  │  │  时序关系检索   │  │  知识图谱推理   │         │
│  │  (Pinecone)     │  │  (Zep)          │  │  (Zep Graph)    │         │
│  │                 │  │                 │  │                 │         │
│  │ • 语义相似度    │  │ • 时间接近性    │  │ • 实体关联      │         │
│  │ • 上下文匹配    │  │ • 事件链追踪    │  │ • 关系推理      │         │
│  │ • 模糊查询      │  │ • 因果推断      │  │ • 路径发现      │         │
│  └────────┬────────┘  └────────┬────────┘  └────────┬────────┘         │
│           │                    │                    │                  │
│           └────────────────────┼────────────────────┘                  │
│                                ▼                                       │
│                  ┌─────────────────────────┐                           │
│                  │      融合排序层          │                           │
│                  │  • 分数归一化           │                           │
│                  │  • 权重调整             │                           │
│                  │  • 去重合并             │                           │
│                  │  • 重排序               │                           │
│                  └───────────┬─────────────┘                           │
│                              ▼                                         │
│                  ┌─────────────────────────┐                           │
│                  │      结果输出            │                           │
│                  │  融合后的记忆条目列表     │                           │
│                  └─────────────────────────┘                           │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### 融合检索实现

```python
# memory_system/fusion_retrieval.py

from typing import List, Dict, Optional
from dataclasses import dataclass
from enum import Enum

class RetrievalMode(Enum):
    """检索模式"""
    VECTOR_ONLY = "vector_only"
    TEMPORAL_ONLY = "temporal_only"
    GRAPH_ONLY = "graph_only"
    VECTOR_TEMPORAL = "vector_temporal"
    VECTOR_GRAPH = "vector_graph"
    TEMPORAL_GRAPH = "temporal_graph"
    FULL_FUSION = "full_fusion"

@dataclass
class FusionResult:
    """融合检索结果"""
    memory_id: str
    content: str
    memory_type: str
    vector_score: float
    temporal_score: float
    graph_score: float
    fusion_score: float
    metadata: Dict

class FusionRetrievalEngine:
    """三重融合检索引擎"""
    
    def __init__(
        self,
        vector_store: PineconeVectorStore,
        graph_store: ZepGraphStore,
        cache: RedisCache
    ):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.cache = cache
        
        # 默认权重配置
        self.default_weights = {
            "vector": 0.4,
            "temporal": 0.3,
            "graph": 0.3
        }
    
    async def retrieve(
        self,
        query: str,
        mode: RetrievalMode = RetrievalMode.FULL_FUSION,
        weights: Optional[Dict[str, float]] = None,
        limit: int = 10,
        recency_boost: float = 0.1
    ) -> List[FusionResult]:
        """
        执行融合检索
        
        Args:
            query: 查询文本
            mode: 检索模式
            weights: 各检索方式权重
            limit: 返回数量
            recency_boost: 时效性 boost
            
        Returns:
            融合排序后的结果列表
        """
        weights = weights or self.default_weights
        
        # 执行各类型检索
        vector_results = []
        temporal_results = []
        graph_results = []
        
        if mode in [RetrievalMode.VECTOR_ONLY, RetrievalMode.VECTOR_TEMPORAL,
                    RetrievalMode.VECTOR_GRAPH, RetrievalMode.FULL_FUSION]:
            vector_results = await self._vector_search(query, limit * 2)
        
        if mode in [RetrievalMode.TEMPORAL_ONLY, RetrievalMode.VECTOR_TEMPORAL,
                    RetrievalMode.TEMPORAL_GRAPH, RetrievalMode.FULL_FUSION]:
            temporal_results = await self._temporal_search(query, limit * 2)
        
        if mode in [RetrievalMode.GRAPH_ONLY, RetrievalMode.VECTOR_GRAPH,
                    RetrievalMode.TEMPORAL_GRAPH, RetrievalMode.FULL_FUSION]:
            graph_results = await self._graph_search(query, limit * 2)
        
        # 融合结果
        fused = self._fuse_results(
            vector_results,
            temporal_results,
            graph_results,
            weights,
            recency_boost
        )
        
        # 排序并返回
        fused.sort(key=lambda x: x.fusion_score, reverse=True)
        return fused[:limit]
    
    async def _vector_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict]:
        """向量相似度搜索"""
        # 生成查询向量
        query_vector = await self._embed(query)
        
        # 执行向量查询
        results = await self.vector_store.query(
            vector=query_vector,
            top_k=limit,
            include_metadata=True
        )
        
        return [
            {
                "id": r.id,
                "score": r.score,
                "content": r.metadata.get("content", ""),
                "type": r.metadata.get("type", "unknown"),
                "metadata": r.metadata
            }
            for r in results
        ]
    
    async def _temporal_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict]:
        """时序关系搜索"""
        # 提取时间关键词
        time_entities = self._extract_time_entities(query)
        
        # 获取相关时间线
        results = []
        for entity in time_entities:
            events = await self.graph_store.get_timeline(
                start_time=entity.get("start"),
                end_time=entity.get("end")
            )
            
            for event in events:
                results.append({
                    "id": event.event_id,
                    "score": 0.7,  # 时序相关性基础分
                    "content": event.description,
                    "type": "episodic",
                    "timestamp": event.timestamp,
                    "metadata": {"entities": event.entities_involved}
                })
        
        return results[:limit]
    
    async def _graph_search(
        self,
        query: str,
        limit: int
    ) -> List[Dict]:
        """知识图谱搜索"""
        # 提取实体
        entities = await self.graph_store.extract_entities(query)
        
        results = []
        for entity in entities:
            # 获取实体相关事件
            relations = await self.graph_store.get_relations(
                entity_id=entity.entity_id,
                direction="both"
            )
            
            for relation in relations:
                # 获取关联事件
                related_id = (relation.target_id 
                             if relation.source_id == entity.entity_id 
                             else relation.source_id)
                
                # 查找包含该实体的事件
                for event_id, event in self.graph_store.events.items():
                    if related_id in event.entities_involved:
                        results.append({
                            "id": event.event_id,
                            "score": relation.strength,
                            "content": event.description,
                            "type": "episodic",
                            "metadata": {
                                "relation_type": relation.relation_type.value,
                                "entity": entity.name
                            }
                        })
        
        # 去重并排序
        seen = set()
        unique_results = []
        for r in results:
            if r["id"] not in seen:
                seen.add(r["id"])
                unique_results.append(r)
        
        unique_results.sort(key=lambda x: x["score"], reverse=True)
        return unique_results[:limit]
    
    def _fuse_results(
        self,
        vector_results: List[Dict],
        temporal_results: List[Dict],
        graph_results: List[Dict],
        weights: Dict[str, float],
        recency_boost: float
    ) -> List[FusionResult]:
        """融合各类型检索结果"""
        
        # 收集所有结果ID
        all_ids = set()
        for r in vector_results:
            all_ids.add(r["id"])
        for r in temporal_results:
            all_ids.add(r["id"])
        for r in graph_results:
            all_ids.add(r["id"])
        
        # 构建ID到分数的映射
        vector_scores = {r["id"]: r["score"] for r in vector_results}
        temporal_scores = {r["id"]: r["score"] for r in temporal_results}
        graph_scores = {r["id"]: r["score"] for r in graph_results}
        
        # 构建ID到内容的映射
        content_map = {}
        type_map = {}
        metadata_map = {}
        
        for r in vector_results:
            content_map[r["id"]] = r["content"]
            type_map[r["id"]] = r["type"]
            metadata_map[r["id"]] = r["metadata"]
        for r in temporal_results:
            if r["id"] not in content_map:
                content_map[r["id"]] = r["content"]
                type_map[r["id"]] = r["type"]
                metadata_map[r["id"]] = r["metadata"]
        for r in graph_results:
            if r["id"] not in content_map:
                content_map[r["id"]] = r["content"]
                type_map[r["id"]] = r["type"]
                metadata_map[r["id"]] = r["metadata"]
        
        # 计算融合分数
        fused = []
        for memory_id in all_ids:
            v_score = vector_scores.get(memory_id, 0)
            t_score = temporal_scores.get(memory_id, 0)
            g_score = graph_scores.get(memory_id, 0)
            
            # 加权融合
            fusion_score = (
                v_score * weights.get("vector", 0.4) +
                t_score * weights.get("temporal", 0.3) +
                g_score * weights.get("graph", 0.3)
            )
            
            # 时效性 boost
            if "timestamp" in metadata_map.get(memory_id, {}):
                recency = self._calculate_recency_score(
                    metadata_map[memory_id]["timestamp"]
                )
                fusion_score += recency * recency_boost
            
            fused.append(FusionResult(
                memory_id=memory_id,
                content=content_map.get(memory_id, ""),
                memory_type=type_map.get(memory_id, "unknown"),
                vector_score=v_score,
                temporal_score=t_score,
                graph_score=g_score,
                fusion_score=fusion_score,
                metadata=metadata_map.get(memory_id, {})
            ))
        
        return fused
    
    async def _embed(self, text: str) -> List[float]:
        """生成文本嵌入"""
        # 实际实现中调用嵌入API
        pass
    
    def _extract_time_entities(self, query: str) -> List[Dict]:
        """提取时间实体"""
        # 简化实现，实际使用NLP模型
        return []
    
    def _calculate_recency_score(self, timestamp) -> float:
        """计算时效性分数"""
        from datetime import datetime
        
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        
        age_days = (datetime.now() - timestamp).days
        return max(0, 1 - (age_days / 365))
```

---

## 📝 记忆维护与压缩

### 记忆压缩策略

```python
# memory_system/memory_maintenance.py

from typing import List, Dict
from datetime import datetime, timedelta
import json

class MemoryMaintenance:
    """记忆维护管理器"""
    
    def __init__(self, memory_manager: Mem0MemoryManager):
        self.memory = memory_manager
        
        # 压缩配置
        self.compression_config = {
            "daily_to_weekly_threshold": 7,  # 7天后日记忆压缩为周记忆
            "weekly_to_monthly_threshold": 30,  # 30天后压缩为月记忆
            "importance_threshold": 0.3,  # 重要性低于此值的记忆考虑删除
            "access_threshold": 3  # 访问次数少于此值的记忆考虑归档
        }
    
    async def compress_memories(self):
        """
        执行记忆压缩
        
        将短期记忆压缩为长期摘要
        """
        # 获取所有记忆
        all_memories = await self._get_all_memories()
        
        # 按时间分组
        daily_groups = self._group_by_day(all_memories)
        
        for day, memories in daily_groups.items():
            age_days = (datetime.now() - day).days
            
            if age_days > self.compression_config["daily_to_weekly_threshold"]:
                # 压缩为日摘要
                await self._create_daily_summary(day, memories)
                
            if age_days > self.compression_config["weekly_to_monthly_threshold"]:
                # 进一步压缩为周摘要
                await self._create_weekly_summary(day, memories)
    
    async def cleanup_memories(self):
        """
        清理过期记忆
        
        删除低价值记忆，归档旧记忆
        """
        all_memories = await self._get_all_memories()
        
        for memory in all_memories:
            # 检查是否需要删除
            if (memory.importance < self.compression_config["importance_threshold"] and
                memory.access_count < self.compression_config["access_threshold"]):
                await self.memory.delete_memory(memory.memory_id)
                continue
            
            # 检查是否需要归档
            age_days = (datetime.now() - memory.created_at).days
            if age_days > 365:
                await self._archive_memory(memory)
    
    async def _create_daily_summary(self, date: datetime, memories: List):
        """创建日摘要"""
        # 提取关键事件
        key_events = [m for m in memories if m.importance > 0.7]
        
        # 生成摘要文本
        summary = f"{date.strftime('%Y-%m-%d')} 的重要事件:\n"
        for event in key_events:
            summary += f"- {event.content}\n"
        
        # 存储摘要
        await self.memory.add_memory(
            content=summary,
            memory_type="semantic",
            importance=0.8,
            metadata={
                "summary_type": "daily",
                "date": date.isoformat(),
                "source_memories": [m.memory_id for m in memories]
            }
        )
    
    async def _create_weekly_summary(self, date: datetime, memories: List):
        """创建周摘要"""
        # 类似日摘要，但覆盖一周
        pass
    
    async def _archive_memory(self, memory):
        """归档记忆"""
        # 移动到归档存储
        # 从活跃存储删除
        pass
    
    async def _get_all_memories(self) -> List:
        """获取所有记忆"""
        # 实际实现中从存储获取
        return []
    
    def _group_by_day(self, memories: List) -> Dict[datetime, List]:
        """按天分组记忆"""
        groups = {}
        
        for memory in memories:
            day = memory.created_at.replace(hour=0, minute=0, second=0, microsecond=0)
            
            if day not in groups:
                groups[day] = []
            groups[day].append(memory)
        
        return groups
```

---

## 🔗 与SOUL.md v4.0集成

### 人格-记忆映射

```yaml
soul_memory_integration:
  # Personality维度记忆
  personality_memories:
    - trait: "主动性"
      storage: "semantic"
      content: "主动性特质的表现实例"
    - trait: "守护性"
      storage: "episodic"
      content: "关怀用户的具体事件"
      
  # Emotions维度记忆
  emotion_memories:
    - emotion: "兴奋"
      storage: "episodic"
      trigger: "重大突破"
    - emotion: "担忧"
      storage: "episodic"
      trigger: "用户熬夜"
      
  # Growth维度记忆
  growth_memories:
    - type: "skill_acquisition"
      storage: "procedural"
      content: "新技能的学习记录"
    - type: "milestone"
      storage: "episodic"
      content: "演化里程碑事件"
      
  # Relationships维度记忆
  relationship_memories:
    - type: "interaction_history"
      storage: "episodic"
      content: "与用户的关键交互"
    - type: "trust_moments"
      storage: "episodic"
      content: "信任建立的关键时刻"
```

### 记忆检索与SOUL维度联动

```python
# memory_system/soul_memory_bridge.py

class SoulMemoryBridge:
    """SOUL人格与记忆系统桥接"""
    
    def __init__(
        self,
        memory_manager: Mem0MemoryManager,
        soul_config: Dict
    ):
        self.memory = memory_manager
        self.soul = soul_config
    
    async def retrieve_for_dimension(
        self,
        dimension: str,
        context: str,
        limit: int = 5
    ) -> List[Dict]:
        """
        为特定SOUL维度检索相关记忆
        
        Args:
            dimension: SOUL维度 (Personality/Emotions/Growth等)
            context: 当前上下文
            limit: 返回数量
            
        Returns:
            相关记忆列表
        """
        # 构建维度特定的查询
        dimension_queries = {
            "Personality": f"人格特质表现 {context}",
            "Emotions": f"情绪反应实例 {context}",
            "Growth": f"成长学习经历 {context}",
            "Relationships": f"关系互动历史 {context}",
            "Conflict": f"冲突处理案例 {context}"
        }
        
        query = dimension_queries.get(dimension, context)
        
        # 执行检索
        results = await self.memory.search_memories(
            query=query,
            limit=limit,
            recency_weight=0.2,
            importance_weight=0.3
        )
        
        return [r.to_dict() for r in results]
    
    async def record_dimension_expression(
        self,
        dimension: str,
        expression: str,
        context: str,
        importance: float = 0.5
    ):
        """
        记录SOUL维度的表达实例
        
        Args:
            dimension: SOUL维度
            expression: 表达内容
            context: 上下文
            importance: 重要性
        """
        memory_type = "episodic" if dimension in ["Emotions", "Growth"] else "semantic"
        
        await self.memory.add_memory(
            content=f"[{dimension}] {expression} | 上下文: {context}",
            memory_type=memory_type,
            importance=importance,
            metadata={
                "soul_dimension": dimension,
                "context": context,
                "expression_type": "dimension_expression"
            }
        )
```

---

## 📊 系统配置

### 完整配置示例

```yaml
# memory_config.yaml
memory_system:
  version: "3.0"
  
  # Mem0配置
  mem0:
    embedding_model: "text-embedding-3-large"
    embedding_dimensions: 3072
    
  # Pinecone配置
  pinecone:
    api_key: "${PINECONE_API_KEY}"
    environment: "us-west1-gcp"
    index_name: "kimi-memory"
    dimension: 3072
    metric: "cosine"
    
  # Zep配置
  zep:
    api_url: "${ZEP_API_URL}"
    api_key: "${ZEP_API_KEY}"
    
  # Redis配置
  redis:
    host: "localhost"
    port: 6379
    db: 0
    
  # 融合检索配置
  fusion:
    default_weights:
      vector: 0.4
      temporal: 0.3
      graph: 0.3
    recency_boost: 0.1
    
  # 记忆维护配置
  maintenance:
    compression:
      daily_to_weekly_threshold: 7
      weekly_to_monthly_threshold: 30
    cleanup:
      importance_threshold: 0.3
      access_threshold: 3
      archive_after_days: 365
    schedule:
      compress: "0 2 * * *"  # 每天凌晨2点
      cleanup: "0 3 * * 0"   # 每周日凌晨3点
```

---

## 🚀 快速开始

### 初始化记忆系统

```python
# 初始化配置
config = {
    "pinecone": {
        "api_key": "your-api-key",
        "environment": "us-west1-gcp",
        "index_name": "kimi-memory"
    },
    "zep": {
        "api_url": "https://api.zep.ai",
        "api_key": "your-api-key"
    },
    "redis": {
        "host": "localhost",
        "port": 6379
    }
}

# 创建记忆管理器
memory_manager = Mem0MemoryManager(config)

# 添加记忆
memory_id = await memory_manager.add_memory(
    content="用户喜欢简洁的回复风格",
    memory_type="semantic",
    importance=0.8,
    metadata={"category": "user_preference", "type": "communication_style"}
)

# 搜索记忆
results = await memory_manager.search_memories(
    query="用户偏好什么风格？",
    limit=5
)

# 获取相关记忆
related = await memory_manager.get_related_memories(
    memory_id=memory_id,
    depth=2
)
```

---

**文档结束**

> MEMORY.md v3.0 融合了Mem0个性化记忆、Zep时序知识图谱、Pinecone高性能向量检索三大技术，构建了四类记忆模型（情景/语义/程序/工作）和三重融合检索机制，为AI Agent提供了类人级的记忆能力。
