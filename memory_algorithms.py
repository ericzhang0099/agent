"""
Mem0 + Zep (Graphiti) 核心算法实现
可直接集成的代码片段
"""

import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Literal, Any
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import numpy as np
from enum import Enum

# =============================================================================
# 1. Mem0 风格记忆压缩引擎
# =============================================================================

class MemoryAction(Enum):
    ADD = "ADD"
    UPDATE = "UPDATE"
    DELETE = "DELETE"
    NONE = "NONE"

@dataclass
class MemoryDecision:
    """记忆操作决策"""
    action: MemoryAction
    content: str
    memory_id: Optional[str] = None
    reason: Optional[str] = None
    confidence: float = 1.0

@dataclass
class Memory:
    """记忆条目"""
    id: str
    content: str
    user_id: str
    created_at: datetime
    updated_at: datetime
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    metadata: Dict = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "content": self.content,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "access_count": self.access_count,
            "metadata": self.metadata
        }

class LLMClient(ABC):
    """LLM客户端抽象"""
    
    @abstractmethod
    def extract_facts(self, messages: List[Dict]) -> List[Dict]:
        """从对话中提取事实"""
        pass
    
    @abstractmethod
    def evaluate_memories(self, new_facts: List[Dict], 
                         existing_memories: List[Memory]) -> List[MemoryDecision]:
        """评估记忆操作决策"""
        pass

class VectorStore(ABC):
    """向量存储抽象"""
    
    @abstractmethod
    def search(self, query: str, user_id: str, top_k: int = 10) -> List[Memory]:
        """语义搜索"""
        pass
    
    @abstractmethod
    def add(self, memory: Memory, embedding: List[float]):
        """添加记忆"""
        pass
    
    @abstractmethod
    def update(self, memory_id: str, content: str, embedding: List[float]):
        """更新记忆"""
        pass
    
    @abstractmethod
    def delete(self, memory_id: str):
        """删除记忆"""
        pass

class MemoryCompressor:
    """
    Mem0风格的记忆压缩引擎
    
    核心算法：
    1. 使用LLM提取结构化事实
    2. 检索相关现有记忆
    3. LLM决策（ADD/UPDATE/DELETE/NONE）
    4. 执行决策并更新存储
    """
    
    def __init__(self, llm_client: LLMClient, vector_store: VectorStore, 
                 embedder: Any):
        self.llm = llm_client
        self.store = vector_store
        self.embedder = embedder
        
    def compress(self, messages: List[Dict], user_id: str) -> List[MemoryDecision]:
        """
        压缩对话历史，提取并管理记忆
        
        Args:
            messages: 对话消息列表 [{"role": "user/assistant", "content": "..."}]
            user_id: 用户标识
            
        Returns:
            记忆操作决策列表
        """
        # Step 1: 提取事实
        facts = self.llm.extract_facts(messages)
        
        # Step 2: 检索相关记忆
        query = " ".join([m.get("content", "") for m in messages[-3:]])
        existing_memories = self.store.search(query, user_id=user_id, top_k=10)
        
        # Step 3: LLM决策
        decisions = self.llm.evaluate_memories(facts, existing_memories)
        
        # Step 4: 执行决策
        self._execute_decisions(decisions, user_id)
        
        return decisions
    
    def _execute_decisions(self, decisions: List[MemoryDecision], user_id: str):
        """执行记忆操作决策"""
        for decision in decisions:
            if decision.action == MemoryAction.ADD:
                self._add_memory(decision.content, user_id)
            elif decision.action == MemoryAction.UPDATE and decision.memory_id:
                self._update_memory(decision.memory_id, decision.content)
            elif decision.action == MemoryAction.DELETE and decision.memory_id:
                self._delete_memory(decision.memory_id)
    
    def _add_memory(self, content: str, user_id: str) -> str:
        """添加新记忆"""
        memory_id = hashlib.md5(f"{user_id}:{content}:{datetime.now()}".encode()).hexdigest()
        memory = Memory(
            id=memory_id,
            content=content,
            user_id=user_id,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )
        embedding = self.embedder.embed(content)
        self.store.add(memory, embedding)
        return memory_id
    
    def _update_memory(self, memory_id: str, content: str):
        """更新现有记忆"""
        embedding = self.embedder.embed(content)
        self.store.update(memory_id, content, embedding)
    
    def _delete_memory(self, memory_id: str):
        """删除记忆"""
        self.store.delete(memory_id)


# =============================================================================
# 2. Zep (Graphiti) 风格时序知识图谱
# =============================================================================

@dataclass
class TemporalFact:
    """时序事实（边）"""
    id: str
    subject: str
    predicate: str
    object: str
    fact_text: str
    valid_at: datetime
    invalid_at: Optional[datetime] = None
    created_at: datetime = field(default_factory=datetime.now)
    source_episode_id: Optional[str] = None
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        """检查在指定时间点是否有效"""
        if timestamp < self.valid_at:
            return False
        if self.invalid_at and timestamp >= self.invalid_at:
            return False
        return True

@dataclass
class Entity:
    """实体节点"""
    id: str
    name: str
    summary: str
    embedding: Optional[List[float]] = None
    created_at: datetime = field(default_factory=datetime.now)

@dataclass
class Episode:
    """原始对话记录"""
    id: str
    content: str
    user_id: str
    timestamp: datetime
    episode_type: str = "message"  # message, text, json
    speaker: Optional[str] = None

class TemporalKnowledgeGraph:
    """
    Zep风格的时序知识图谱
    
    核心特性：
    1. 双时序模型（事件时间T + 事务时间T'）
    2. 事实版本管理（valid_at/invalid_at）
    3. 三层架构（Episode -> Entity -> Community）
    """
    
    def __init__(self, neo4j_driver=None, vector_store=None):
        self.driver = neo4j_driver
        self.vector_store = vector_store
        self.episodes: Dict[str, Episode] = {}
        self.entities: Dict[str, Entity] = {}
        self.facts: Dict[str, TemporalFact] = {}
        
    def add_episode(self, content: str, user_id: str, 
                    timestamp: Optional[datetime] = None,
                    episode_type: str = "message",
                    speaker: Optional[str] = None) -> str:
        """
        添加原始对话记录（Episode）
        
        这是知识图谱构建的第一步，保存原始数据用于溯源
        """
        episode_id = hashlib.md5(f"{user_id}:{content}:{timestamp or datetime.now()}".encode()).hexdigest()[:16]
        
        episode = Episode(
            id=episode_id,
            content=content,
            user_id=user_id,
            timestamp=timestamp or datetime.now(),
            episode_type=episode_type,
            speaker=speaker
        )
        
        self.episodes[episode_id] = episode
        
        # 如果配置了Neo4j，同步写入
        if self.driver:
            with self.driver.session() as session:
                session.run("""
                    CREATE (e:Episode {
                        id: $id,
                        content: $content,
                        user_id: $user_id,
                        timestamp: $timestamp,
                        type: $type,
                        speaker: $speaker
                    })
                """, id=episode_id, content=content, user_id=user_id,
                     timestamp=episode.timestamp.isoformat(),
                     type=episode_type, speaker=speaker)
        
        return episode_id
    
    def extract_and_add_fact(self, subject: str, predicate: str, object: str,
                            fact_text: str,
                            valid_at: datetime,
                            invalid_at: Optional[datetime] = None,
                            source_episode_id: Optional[str] = None,
                            user_id: Optional[str] = None) -> str:
        """
        提取并添加时序事实
        
        自动处理：
        1. 实体创建/更新
        2. 事实冲突检测与失效
        3. 双向索引建立
        """
        # 创建或获取实体
        subject_entity = self._get_or_create_entity(subject, user_id)
        object_entity = self._get_or_create_entity(object, user_id)
        
        # 检查冲突并失效旧事实
        self._invalidate_conflicting_facts(
            subject_entity.id, object_entity.id, 
            predicate, valid_at
        )
        
        # 创建新事实
        fact_id = hashlib.md5(f"{subject}:{predicate}:{object}:{valid_at}".encode()).hexdigest()[:16]
        
        fact = TemporalFact(
            id=fact_id,
            subject=subject,
            predicate=predicate,
            object=object,
            fact_text=fact_text,
            valid_at=valid_at,
            invalid_at=invalid_at,
            source_episode_id=source_episode_id
        )
        
        self.facts[fact_id] = fact
        
        # 同步到Neo4j
        if self.driver:
            with self.driver.session() as session:
                session.run("""
                    MATCH (s:Entity {id: $subject_id})
                    MATCH (o:Entity {id: $object_id})
                    CREATE (s)-[r:FACT {
                        id: $fact_id,
                        predicate: $predicate,
                        fact_text: $fact_text,
                        valid_at: $valid_at,
                        invalid_at: $invalid_at,
                        created_at: datetime()
                    }]->(o)
                """, subject_id=subject_entity.id, object_id=object_entity.id,
                     fact_id=fact_id, predicate=predicate,
                     fact_text=fact_text,
                     valid_at=valid_at.isoformat(),
                     invalid_at=invalid_at.isoformat() if invalid_at else None)
                
                if source_episode_id:
                    session.run("""
                        MATCH (e:Episode {id: $episode_id})
                        MATCH ()-[r:FACT {id: $fact_id}]-()
                        CREATE (e)-[:EXTRACTED_FROM]->(r)
                    """, episode_id=source_episode_id, fact_id=fact_id)
        
        return fact_id
    
    def _get_or_create_entity(self, name: str, user_id: Optional[str] = None) -> Entity:
        """获取或创建实体"""
        # 简单实现：基于名称哈希
        entity_id = hashlib.md5(name.encode()).hexdigest()[:16]
        
        if entity_id not in self.entities:
            entity = Entity(
                id=entity_id,
                name=name,
                summary=name  # 初始摘要就是名称
            )
            self.entities[entity_id] = entity
            
            if self.driver:
                with self.driver.session() as session:
                    session.run("""
                        MERGE (e:Entity {id: $id})
                        SET e.name = $name, e.summary = $summary
                    """, id=entity_id, name=name, summary=name)
        
        return self.entities[entity_id]
    
    def _invalidate_conflicting_facts(self, subject_id: str, object_id: str,
                                     predicate: str, new_valid_at: datetime):
        """
        检测并失效冲突的事实
        
        当新事实与现有事实冲突时，将旧事实标记为失效
        """
        # 查找相同主谓宾的事实
        conflicting = [
            fact for fact in self.facts.values()
            if fact.subject == self.entities.get(subject_id, Entity("", "", "")).name
            and fact.object == self.entities.get(object_id, Entity("", "", "")).name
            and fact.predicate == predicate
            and fact.invalid_at is None  # 当前仍有效
            and fact.valid_at < new_valid_at  # 新事实更晚
        ]
        
        for old_fact in conflicting:
            # 失效旧事实
            old_fact.invalid_at = new_valid_at
            
            if self.driver:
                with self.driver.session() as session:
                    session.run("""
                        MATCH ()-[r:FACT {id: $fact_id}]->()
                        SET r.invalid_at = $invalid_at
                    """, fact_id=old_fact.id, invalid_at=new_valid_at.isoformat())
    
    def temporal_search(self, query: str, user_id: str,
                       as_of: Optional[datetime] = None,
                       top_k: int = 10) -> List[Dict]:
        """
        时序感知搜索
        
        返回在指定时间点有效的所有相关事实
        """
        results = []
        
        # 默认查询当前时间
        if as_of is None:
            as_of = datetime.now()
        
        # 过滤有效事实
        for fact in self.facts.values():
            if fact.is_valid_at(as_of):
                results.append({
                    "fact_id": fact.id,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "fact_text": fact.fact_text,
                    "valid_from": fact.valid_at,
                    "valid_until": fact.invalid_at
                })
        
        # 按相关度排序（简化实现：按时间倒序）
        results.sort(key=lambda x: x["valid_from"], reverse=True)
        
        return results[:top_k]
    
    def get_entity_timeline(self, entity_name: str) -> List[Dict]:
        """
        获取实体的时间线
        
        返回该实体参与的所有事实，按时间排序
        """
        timeline = []
        
        for fact in self.facts.values():
            if fact.subject == entity_name or fact.object == entity_name:
                timeline.append({
                    "fact": fact.fact_text,
                    "valid_at": fact.valid_at,
                    "invalid_at": fact.invalid_at,
                    "is_active": fact.invalid_at is None
                })
        
        timeline.sort(key=lambda x: x["valid_at"])
        return timeline


# =============================================================================
# 3. 混合检索（向量 + 图谱 + 全文）
# =============================================================================

class Reranker:
    """重排序算法"""
    
    @staticmethod
    def reciprocal_rank_fusion(results_lists: List[List[Dict]], 
                                k: int = 60) -> List[Dict]:
        """
        RRF (Reciprocal Rank Fusion) 算法
        
        公式: score = Σ(1 / (k + rank))
        
        用于融合多个搜索结果列表
        """
        scores = {}
        
        for results in results_lists:
            for rank, item in enumerate(results):
                doc_id = item.get("id") or item.get("fact_id") or hashlib.md5(
                    str(item).encode()).hexdigest()[:16]
                
                if doc_id not in scores:
                    scores[doc_id] = {"item": item, "score": 0.0}
                
                # RRF分数计算
                scores[doc_id]["score"] += 1.0 / (k + rank + 1)
        
        # 按融合分数排序
        fused = sorted(scores.values(), key=lambda x: x["score"], reverse=True)
        return [x["item"] for x in fused]
    
    @staticmethod
    def maximal_marginal_relevance(query_embedding: List[float],
                                   results: List[Dict],
                                   lambda_param: float = 0.5,
                                   top_k: int = 10) -> List[Dict]:
        """
        MMR (Maximal Marginal Relevance) 算法
        
        平衡相关性和多样性：
        MMR = λ * Sim(query, doc) - (1-λ) * max(Sim(doc, selected))
        
        Args:
            lambda_param: 0-1之间，越大越注重相关性，越小越注重多样性
        """
        selected = []
        remaining = list(results)
        
        while len(selected) < top_k and remaining:
            mmr_scores = []
            
            for item in remaining:
                # 与查询的相似度
                query_sim = item.get("score", 0.5)
                
                # 与已选结果的最大相似度
                max_sim = 0
                for sel in selected:
                    # 简化：使用预存分数，实际应计算向量相似度
                    sim = item.get("vector_similarity", 0.5)
                    max_sim = max(max_sim, sim)
                
                # MMR分数
                mmr_score = lambda_param * query_sim - (1 - lambda_param) * max_sim
                mmr_scores.append((item, mmr_score))
            
            # 选择MMR分数最高的
            mmr_scores.sort(key=lambda x: x[1], reverse=True)
            best_item = mmr_scores[0][0]
            
            selected.append(best_item)
            remaining.remove(best_item)
        
        return selected

class HybridRetriever:
    """
    混合检索引擎
    
    融合三种检索方式：
    1. 向量语义搜索（语义相似度）
    2. 全文关键词搜索（BM25）
    3. 图谱关系搜索（BFS遍历）
    """
    
    def __init__(self, vector_store=None, graph_store=None, 
                 fulltext_index=None, embedder=None):
        self.vector_store = vector_store
        self.graph_store = graph_store
        self.fulltext = fulltext_index
        self.embedder = embedder
        self.reranker = Reranker()
    
    def search(self, query: str, user_id: str, 
               top_k: int = 10,
               use_vector: bool = True,
               use_fulltext: bool = True,
               use_graph: bool = True,
               rerank_method: str = "rrf") -> List[Dict]:
        """
        执行混合检索
        
        Args:
            query: 查询字符串
            user_id: 用户标识
            top_k: 返回结果数量
            use_vector: 是否使用向量搜索
            use_fulltext: 是否使用全文搜索
            use_graph: 是否使用图谱搜索
            rerank_method: 重排序方法 (rrf/mmr/none)
        """
        results_lists = []
        
        # 1. 向量语义搜索
        if use_vector and self.vector_store:
            vector_results = self._vector_search(query, user_id, top_k * 2)
            results_lists.append(vector_results)
        
        # 2. 全文关键词搜索
        if use_fulltext and self.fulltext:
            fulltext_results = self._fulltext_search(query, user_id, top_k * 2)
            results_lists.append(fulltext_results)
        
        # 3. 图谱关系搜索
        if use_graph and self.graph_store:
            graph_results = self._graph_search(query, user_id, top_k * 2)
            results_lists.append(graph_results)
        
        # 4. 结果融合
        if rerank_method == "rrf" and len(results_lists) > 1:
            combined = self.reranker.reciprocal_rank_fusion(results_lists)
        elif rerank_method == "mmr" and self.embedder:
            all_results = [item for sublist in results_lists for item in sublist]
            query_embedding = self.embedder.embed(query)
            combined = self.reranker.maximal_marginal_relevance(
                query_embedding, all_results, top_k=top_k
            )
        else:
            # 简单合并去重
            seen = set()
            combined = []
            for sublist in results_lists:
                for item in sublist:
                    item_id = item.get("id") or str(item)
                    if item_id not in seen:
                        seen.add(item_id)
                        combined.append(item)
        
        return combined[:top_k]
    
    def _vector_search(self, query: str, user_id: str, top_k: int) -> List[Dict]:
        """向量语义搜索"""
        if not self.vector_store:
            return []
        
        embedding = self.embedder.embed(query) if self.embedder else None
        
        # 这里调用实际的向量存储
        # 简化实现
        return [{
            "id": f"vec_{i}",
            "content": f"Vector result for: {query}",
            "score": 0.9 - i * 0.05,
            "source": "vector"
        } for i in range(top_k)]
    
    def _fulltext_search(self, query: str, user_id: str, top_k: int) -> List[Dict]:
        """全文关键词搜索（BM25）"""
        if not self.fulltext:
            return []
        
        # 简化实现
        return [{
            "id": f"text_{i}",
            "content": f"Text result for: {query}",
            "score": 0.85 - i * 0.05,
            "source": "fulltext"
        } for i in range(top_k)]
    
    def _graph_search(self, query: str, user_id: str, top_k: int) -> List[Dict]:
        """图谱关系搜索（BFS遍历）"""
        if not self.graph_store:
            return []
        
        # 简化实现
        return [{
            "id": f"graph_{i}",
            "content": f"Graph result for: {query}",
            "score": 0.8 - i * 0.05,
            "source": "graph"
        } for i in range(top_k)]


# =============================================================================
# 4. 上下文组装器
# =============================================================================

class ContextAssembler:
    """
    上下文组装器
    
    将检索结果格式化为LLM可用的上下文字符串
    """
    
    def __init__(self, max_tokens: int = 4000):
        self.max_tokens = max_tokens
        # 粗略估计：1 token ≈ 4字符
        self.chars_per_token = 4
    
    def assemble(self, facts: List[Dict], entities: List[Dict] = None,
                 communities: List[Dict] = None) -> str:
        """
        组装上下文
        
        输出格式：
        <FACTS>
        fact_text (Date range: valid_at - invalid_at)
        ...
        </FACTS>
        
        <ENTITIES>
        entity_name: summary
        ...
        </ENTITIES>
        """
        parts = []
        current_length = 0
        max_chars = self.max_tokens * self.chars_per_token
        
        # 1. 添加事实
        if facts:
            facts_str = self._format_facts(facts)
            if current_length + len(facts_str) < max_chars:
                parts.append(f"<FACTS>\n{facts_str}\n</FACTS>")
                current_length += len(facts_str)
        
        # 2. 添加实体
        if entities and current_length < max_chars:
            entities_str = self._format_entities(entities)
            if current_length + len(entities_str) < max_chars:
                parts.append(f"<ENTITIES>\n{entities_str}\n</ENTITIES>")
                current_length += len(entities_str)
        
        # 3. 添加社区（如空间允许）
        if communities and current_length < max_chars:
            communities_str = self._format_communities(communities)
            remaining = max_chars - current_length
            if len(communities_str) > remaining:
                communities_str = communities_str[:remaining] + "..."
            parts.append(f"<COMMUNITIES>\n{communities_str}\n</COMMUNITIES>")
        
        return "\n\n".join(parts)
    
    def _format_facts(self, facts: List[Dict]) -> str:
        """格式化事实列表"""
        lines = []
        for fact in facts:
            text = fact.get("fact_text") or fact.get("content", "")
            valid_from = fact.get("valid_at") or fact.get("valid_from")
            valid_until = fact.get("invalid_at") or fact.get("valid_until")
            
            if valid_from:
                date_range = f" (Date range: {valid_from} - {valid_until or 'present'})"
                lines.append(f"{text}{date_range}")
            else:
                lines.append(text)
        
        return "\n".join(lines)
    
    def _format_entities(self, entities: List[Dict]) -> str:
        """格式化实体列表"""
        lines = []
        for entity in entities:
            name = entity.get("name", "")
            summary = entity.get("summary", "")
            lines.append(f"{name}: {summary}")
        
        return "\n".join(lines)
    
    def _format_communities(self, communities: List[Dict]) -> str:
        """格式化社区列表"""
        lines = []
        for comm in communities:
            name = comm.get("name", "")
            summary = comm.get("summary", "")
            lines.append(f"{name}: {summary}")
        
        return "\n".join(lines)


# =============================================================================
# 5. 使用示例
# =============================================================================

def example_usage():
    """使用示例"""
    
    # 1. 初始化组件
    # llm_client = YourLLMClient()
    # vector_store = YourVectorStore()
    # embedder = YourEmbedder()
    
    # 2. 创建记忆压缩器
    # compressor = MemoryCompressor(llm_client, vector_store, embedder)
    
    # 3. 压缩对话
    # messages = [
    #     {"role": "user", "content": "我喜欢吃意大利菜"},
    #     {"role": "assistant", "content": "好的，我记住了您的偏好"},
    #     {"role": "user", "content": "其实我现在更喜欢中餐了"}
    # ]
    # decisions = compressor.compress(messages, user_id="user123")
    
    # 4. 创建时序知识图谱
    tkg = TemporalKnowledgeGraph()
    
    # 5. 添加对话记录
    episode_id = tkg.add_episode(
        content="用户喜欢意大利菜",
        user_id="user123",
        timestamp=datetime(2024, 1, 1)
    )
    
    # 6. 提取事实
    tkg.extract_and_add_fact(
        subject="用户123",
        predicate="喜欢",
        object="意大利菜",
        fact_text="用户123喜欢意大利菜",
        valid_at=datetime(2024, 1, 1),
        source_episode_id=episode_id,
        user_id="user123"
    )
    
    # 7. 更新偏好（使旧事实失效）
    tkg.extract_and_add_fact(
        subject="用户123",
        predicate="喜欢",
        object="中餐",
        fact_text="用户123现在更喜欢中餐",
        valid_at=datetime(2024, 6, 1),  # 新事实生效时间
        source_episode_id=episode_id,
        user_id="user123"
    )
    
    # 8. 时序查询
    # 查询2024年3月的偏好（应该是意大利菜）
    results_march = tkg.temporal_search(
        "用户的饮食偏好",
        user_id="user123",
        as_of=datetime(2024, 3, 1)
    )
    print("2024年3月的偏好:", results_march)
    
    # 查询2024年7月的偏好（应该是中餐）
    results_july = tkg.temporal_search(
        "用户的饮食偏好",
        user_id="user123",
        as_of=datetime(2024, 7, 1)
    )
    print("2024年7月的偏好:", results_july)
    
    # 9. 混合检索
    retriever = HybridRetriever()
    results = retriever.search(
        query="用户的饮食偏好",
        user_id="user123",
        top_k=5
    )
    
    # 10. 组装上下文
    assembler = ContextAssembler(max_tokens=2000)
    context = assembler.assemble(
        facts=results,
        entities=[{"name": "用户123", "summary": "喜欢中餐"}]
    )
    print("组装后的上下文:", context)


if __name__ == "__main__":
    example_usage()
