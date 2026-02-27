#!/usr/bin/env python3
"""
长期记忆压缩优化系统 v2.0
实现智能压缩、重要性评分、混合检索
"""

import os
import json
import hashlib
import re
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, asdict, field
from collections import defaultdict
import heapq

# 配置
DEFAULT_CONFIG = {
    "hot_storage_days": 7,
    "warm_storage_days": 30,
    "compression_threshold_high": 4.0,
    "compression_threshold_medium": 2.0,
    "compression_threshold_low": 1.0,
    "recency_half_life_days": 30,
    "time_decay_half_life": 60,
    "semantic_similarity_threshold": 0.85,
    "max_summary_ratio": 0.6,
    "keyword_weight": 0.3,
    "semantic_weight": 0.7,
}

# 确保所有配置项都有默认值
DEFAULT_CONFIG_FULL = {
    **DEFAULT_CONFIG,
    "embedding_dim": 384,
    "max_candidates": 100,
    "summary_sentences": 3,
}


@dataclass
class MemoryRecord:
    """优化后的记忆记录结构"""
    
    # 基础标识
    id: str
    parent_id: Optional[str] = None
    
    # 内容（分层存储）
    content_full: Optional[str] = None
    content_summary: Optional[str] = None
    content_keypoints: List[str] = field(default_factory=list)
    
    # 压缩元数据
    compression_level: int = 5  # 0-5, 5=未压缩
    compression_ratio: float = 1.0
    original_length: int = 0
    compressed_length: int = 0
    
    # 重要性评分
    importance_score: float = 0.0
    importance_factors: Dict[str, float] = field(default_factory=dict)
    
    # 访问统计
    access_count: int = 0
    last_accessed: Optional[datetime] = None
    access_pattern: List[datetime] = field(default_factory=list)
    
    # 分类标签
    memory_type: str = "general"
    categories: List[str] = field(default_factory=list)
    entities: List[str] = field(default_factory=list)
    keywords: List[str] = field(default_factory=list)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    compressed_at: Optional[datetime] = None
    
    # 来源追踪
    source: str = ""
    context: Dict[str, Any] = field(default_factory=dict)
    
    # 用户交互
    user_marked_important: bool = False
    user_notes: Optional[str] = None
    
    def to_dict(self) -> Dict:
        """转换为字典（用于序列化）"""
        data = asdict(self)
        # 转换datetime为ISO格式字符串
        for key in ['created_at', 'updated_at', 'last_accessed', 'compressed_at']:
            if data[key]:
                data[key] = data[key].isoformat() if isinstance(data[key], datetime) else data[key]
        for key in ['access_pattern']:
            data[key] = [d.isoformat() if isinstance(d, datetime) else d for d in data[key]]
        return data
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryRecord':
        """从字典创建实例"""
        # 转换ISO格式字符串为datetime
        for key in ['created_at', 'updated_at', 'last_accessed', 'compressed_at']:
            if data.get(key):
                data[key] = datetime.fromisoformat(data[key])
        for key in ['access_pattern']:
            if data.get(key):
                data[key] = [datetime.fromisoformat(d) for d in data[key]]
        return cls(**data)


class TextProcessor:
    """文本处理工具类"""
    
    @staticmethod
    def split_sentences(text: str) -> List[str]:
        """将文本分割为句子"""
        # 支持中英文句子分割
        sentences = re.split(r'(?<=[。！？.!?])\s+', text)
        return [s.strip() for s in sentences if s.strip()]
    
    @staticmethod
    def extract_keywords(text: str, top_k: int = 10) -> List[str]:
        """提取关键词（简单TF-IDF近似）"""
        # 分词
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 停用词过滤
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                     '的', '了', '在', '是', '和', '有', '我', '你', '他'}
        words = [w for w in words if w not in stopwords and len(w) > 1]
        
        # 统计词频
        word_freq = defaultdict(int)
        for word in words:
            word_freq[word] += 1
        
        # 返回Top-K
        return [word for word, _ in heapq.nlargest(top_k, word_freq.items(), key=lambda x: x[1])]
    
    @staticmethod
    def extract_entities(text: str) -> List[str]:
        """提取实体（简化版，基于大写和引号）"""
        # 提取引号内的内容
        quoted = re.findall(r'["""]([^"""]+)["""]', text)
        # 提取大写单词（可能是专有名词）
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        # 合并去重
        entities = list(set(quoted + capitalized))
        return entities[:20]  # 限制数量
    
    @staticmethod
    def calculate_similarity(text1: str, text2: str) -> float:
        """计算两段文本的相似度（基于Jaccard）"""
        set1 = set(TextProcessor.extract_keywords(text1, top_k=50))
        set2 = set(TextProcessor.extract_keywords(text2, top_k=50))
        
        if not set1 or not set2:
            return 0.0
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0


class MemoryImportanceScorer:
    """
    记忆重要性评分算法
    
    评分维度:
    1. 访问频率 (30%)
    2. 决策关键度 (25%)
    3. 信息密度 (20%)
    4. 时效性 (15%)
    5. 用户显式标记 (10%)
    """
    
    DECISION_KEYWORDS = [
        '决策', '决定', '选择', '批准', '拒绝', '关键', '重要',
        'decision', 'decide', 'choose', 'approve', 'reject', 'critical', 'important'
    ]
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG
    
    def calculate_importance(self, memory: MemoryRecord) -> Tuple[float, Dict[str, float]]:
        """
        计算记忆重要性评分
        
        Returns:
            (总分, 各维度得分详情)
        """
        factors = {}
        
        # 1. 访问频率分数 (0-1.5)
        access_score = min(memory.access_count / 10, 1.5)
        factors['access_frequency'] = round(access_score, 2)
        
        # 2. 决策关键度 (0-1.25)
        content_lower = memory.content_full.lower() if memory.content_full else ""
        decision_matches = sum(1 for kw in self.DECISION_KEYWORDS if kw in content_lower)
        decision_score = min(decision_matches * 0.25, 1.25)
        factors['decision_criticality'] = round(decision_score, 2)
        
        # 3. 信息密度 (0-1.0)
        entities = memory.entities or TextProcessor.extract_entities(content_lower)
        keywords = memory.keywords or TextProcessor.extract_keywords(content_lower, top_k=50)
        
        entity_score = min(len(entities) * 0.1, 0.5)
        keyword_score = min(len(keywords) / 20, 0.5)
        density_score = entity_score + keyword_score
        factors['information_density'] = round(density_score, 2)
        
        # 4. 时效性 (0-0.75) - 指数衰减
        days_old = (datetime.now() - memory.created_at).days
        recency_score = 0.75 * np.exp(-days_old / self.config['recency_half_life_days'])
        factors['recency'] = round(recency_score, 2)
        
        # 5. 用户标记 (0-0.5)
        user_score = 0.5 if memory.user_marked_important else 0
        factors['user_marked'] = user_score
        
        # 总分 (0-5)
        total_score = access_score + decision_score + density_score + recency_score + user_score
        
        return round(total_score, 2), factors
    
    def batch_score(self, memories: List[MemoryRecord]) -> List[Tuple[MemoryRecord, float, Dict]]:
        """批量评分"""
        results = []
        for memory in memories:
            score, factors = self.calculate_importance(memory)
            results.append((memory, score, factors))
        return results


class MemoryCompressor:
    """
    记忆智能压缩算法
    
    压缩策略:
    - 5分: 保留完整文本 + 完整元数据
    - 4分: 保留完整文本 + 精简元数据
    - 3分: 生成结构化摘要 (保留60%信息)
    - 2分: 提取关键要点 (保留30%信息)
    - 1分: 提取核心实体和关系
    - 0分: 仅保留索引，归档到冷存储
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG
        self.text_processor = TextProcessor()
    
    def compress(self, memory: MemoryRecord, importance: float) -> MemoryRecord:
        """
        根据重要性评分压缩记忆
        
        Returns:
            压缩后的记忆记录（原地修改）
        """
        if memory.content_full is None:
            return memory
        
        original_content = memory.content_full
        original_length = len(original_content)
        memory.original_length = original_length
        
        if importance >= 4.0:
            # 高重要性: 完整保留
            memory.compression_level = 5
            memory.compression_ratio = 1.0
            memory.compressed_length = original_length
            
        elif importance >= 3.0:
            # 中高重要性: 结构化摘要
            summary = self._generate_summary(original_content, ratio=self.config['max_summary_ratio'])
            memory.content_summary = summary
            memory.compression_level = 3
            memory.compression_ratio = len(summary) / original_length
            memory.compressed_length = len(summary)
            
        elif importance >= 2.0:
            # 中等重要性: 关键要点
            keypoints = self._extract_key_points(original_content)
            memory.content_keypoints = keypoints
            memory.content_summary = None  # 清除摘要层
            memory.compression_level = 2
            compressed_text = ' '.join(keypoints)
            memory.compression_ratio = len(compressed_text) / original_length
            memory.compressed_length = len(compressed_text)
            
        elif importance >= 1.0:
            # 低重要性: 核心实体
            entities = self.text_processor.extract_entities(original_content)
            memory.entities = entities
            memory.content_keypoints = []  # 清除要点层
            memory.content_summary = None
            memory.compression_level = 1
            compressed_text = ', '.join(entities)
            memory.compression_ratio = len(compressed_text) / original_length if original_length > 0 else 0
            memory.compressed_length = len(compressed_text)
            
        else:
            # 极低重要性: 仅保留索引
            memory.content_full = None  # 清除完整内容
            memory.content_summary = None
            memory.content_keypoints = []
            memory.compression_level = 0
            memory.compression_ratio = 0.0
            memory.compressed_length = 0
        
        memory.compressed_at = datetime.now()
        memory.updated_at = datetime.now()
        
        return memory
    
    def _generate_summary(self, text: str, ratio: float = 0.6) -> str:
        """生成结构化摘要（基于句子重要性）"""
        sentences = self.text_processor.split_sentences(text)
        
        if len(sentences) <= 3:
            return text
        
        # 计算句子重要性
        sentence_scores = {}
        keywords = set(self.text_processor.extract_keywords(text, top_k=30))
        
        for i, sent in enumerate(sentences):
            score = 0
            # 关键词匹配
            sent_keywords = set(self.text_processor.extract_keywords(sent, top_k=10))
            score += len(sent_keywords & keywords) * 2
            
            # 位置权重（开头和结尾的句子更重要）
            if i == 0:
                score += 5
            elif i == len(sentences) - 1:
                score += 3
            elif i < len(sentences) * 0.2:
                score += 2
            
            # 长度惩罚（过长或过短的句子减分）
            sent_len = len(sent)
            if sent_len < 10:
                score -= 2
            elif sent_len > 200:
                score -= 1
            
            sentence_scores[i] = score
        
        # 选择Top-K句子
        k = max(1, int(len(sentences) * ratio))
        top_sentences = heapq.nlargest(k, sentence_scores.items(), key=lambda x: x[1])
        top_sentences.sort(key=lambda x: x[0])  # 按原文顺序排列
        
        summary = ' '.join([sentences[i] for i, _ in top_sentences])
        return summary
    
    def _extract_key_points(self, text: str) -> List[str]:
        """提取关键要点"""
        sentences = self.text_processor.split_sentences(text)
        
        if len(sentences) <= 5:
            return sentences
        
        # 选择最重要的30%句子作为关键要点
        keywords = set(self.text_processor.extract_keywords(text, top_k=20))
        sentence_scores = []
        
        for i, sent in enumerate(sentences):
            sent_keywords = set(self.text_processor.extract_keywords(sent, top_k=5))
            score = len(sent_keywords & keywords)
            if i == 0:  # 首句加分
                score += 3
            sentence_scores.append((i, score))
        
        # 选择Top 30%
        k = max(1, int(len(sentences) * 0.3))
        top_sentences = heapq.nlargest(k, sentence_scores, key=lambda x: x[1])
        top_sentences.sort(key=lambda x: x[0])
        
        return [sentences[i] for i, _ in top_sentences]
    
    def decompress(self, memory: MemoryRecord) -> str:
        """
        解压记忆（获取最佳可用内容）
        
        Returns:
            可用的文本内容
        """
        if memory.content_full:
            return memory.content_full
        elif memory.content_summary:
            return memory.content_summary
        elif memory.content_keypoints:
            return '\n'.join(memory.content_keypoints)
        elif memory.entities:
            return '相关实体: ' + ', '.join(memory.entities)
        else:
            return f"[已归档记忆: {memory.id}]"


class DuplicateDetector:
    """重复记忆检测器"""
    
    def __init__(self, similarity_threshold: float = 0.85):
        self.similarity_threshold = similarity_threshold
        self.text_processor = TextProcessor()
    
    def find_duplicates(self, memories: List[MemoryRecord]) -> List[Tuple[str, str, float]]:
        """
        查找重复的记忆对
        
        Returns:
            [(id1, id2, similarity), ...]
        """
        duplicates = []
        n = len(memories)
        
        for i in range(n):
            for j in range(i + 1, n):
                mem1 = memories[i]
                mem2 = memories[j]
                
                content1 = mem1.content_full or mem1.content_summary or ''
                content2 = mem2.content_full or mem2.content_summary or ''
                
                if not content1 or not content2:
                    continue
                
                similarity = self.text_processor.calculate_similarity(content1, content2)
                
                if similarity >= self.similarity_threshold:
                    duplicates.append((mem1.id, mem2.id, similarity))
        
        return duplicates
    
    def merge_duplicates(self, mem1: MemoryRecord, mem2: MemoryRecord) -> MemoryRecord:
        """合并重复的记忆（保留信息更完整的版本）"""
        # 选择更长的、更新的版本
        if mem1.created_at > mem2.created_at:
            newer, older = mem1, mem2
        else:
            newer, older = mem2, mem1
        
        # 合并访问统计
        newer.access_count += older.access_count
        newer.access_pattern.extend(older.access_pattern)
        newer.access_pattern.sort()
        
        # 合并实体和关键词
        newer.entities = list(set(newer.entities + older.entities))
        newer.keywords = list(set(newer.keywords + older.keywords))
        
        # 如果旧版本有用户标记，保留
        if older.user_marked_important:
            newer.user_marked_important = True
        
        newer.updated_at = datetime.now()
        
        return newer


class HybridRetriever:
    """
    语义 + 关键词 + 重要性 混合检索引擎
    
    检索流程:
    1. 关键词预过滤 (快速缩小候选集)
    2. 语义相似度计算 (精确匹配)
    3. 重要性加权 (优先高价值记忆)
    4. 时间衰减调整 (平衡新旧记忆)
    """
    
    def __init__(self, config: Dict = None):
        self.config = config or DEFAULT_CONFIG
        self.text_processor = TextProcessor()
        
        # 索引结构
        self.inverted_index: Dict[str, Set[str]] = defaultdict(set)
        self.importance_index: Dict[float, Set[str]] = defaultdict(set)
        self.time_index: Dict[str, Set[str]] = defaultdict(set)
        self.type_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 向量索引（简化版，使用关键词向量）
        self.vector_index: Dict[str, np.ndarray] = {}
    
    def build_index(self, memories: List[MemoryRecord]):
        """构建索引"""
        for memory in memories:
            self._index_memory(memory)
    
    def _index_memory(self, memory: MemoryRecord):
        """索引单个记忆"""
        # 倒排索引
        keywords = memory.keywords or []
        for kw in keywords:
            self.inverted_index[kw].add(memory.id)
        
        # 重要性索引
        importance_bucket = round(memory.importance_score)
        self.importance_index[importance_bucket].add(memory.id)
        
        # 时间索引
        date_key = memory.created_at.strftime('%Y-%m-%d')
        self.time_index[date_key].add(memory.id)
        
        # 类型索引
        self.type_index[memory.memory_type].add(memory.id)
        
        # 向量索引（关键词的one-hot近似）
        self.vector_index[memory.id] = self._compute_vector(memory)
    
    def _compute_vector(self, memory: MemoryRecord) -> np.ndarray:
        """计算记忆的向量表示（简化版）"""
        # 使用关键词的哈希作为向量
        keywords = memory.keywords or []
        vector = np.zeros(128)
        
        for kw in keywords:
            # 使用哈希值填充向量
            hash_val = int(hashlib.md5(kw.encode()).hexdigest(), 16)
            for i in range(128):
                if (hash_val >> i) & 1:
                    vector[i] += 1
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector
    
    def retrieve(self, query: str, memories: Dict[str, MemoryRecord], 
                 top_k: int = 5, memory_type: Optional[str] = None) -> List[Dict]:
        """
        混合检索主函数
        
        Args:
            query: 查询字符串
            memories: 记忆字典 {id: MemoryRecord}
            top_k: 返回结果数量
            memory_type: 可选的记忆类型过滤
        
        Returns:
            检索结果列表
        """
        # Step 1: 关键词预过滤
        query_keywords = self.text_processor.extract_keywords(query, top_k=10)
        candidate_ids = self._keyword_filter(query_keywords, memory_type)
        
        if not candidate_ids:
            return []
        
        # Step 2: 计算查询向量
        query_vector = self._compute_query_vector(query_keywords)
        
        # Step 3: 语义相似度计算 + 重要性加权 + 时间衰减
        scored_results = []
        
        for mem_id in candidate_ids:
            if mem_id not in memories:
                continue
            
            memory = memories[mem_id]
            
            # 语义相似度
            mem_vector = self.vector_index.get(mem_id, np.zeros(128))
            semantic_sim = np.dot(query_vector, mem_vector)
            
            # 关键词匹配度
            keyword_match = len(set(query_keywords) & set(memory.keywords or []))
            keyword_score = min(keyword_match / len(query_keywords), 1.0) if query_keywords else 0
            
            # 混合基础分
            base_score = (semantic_sim * self.config['semantic_weight'] + 
                         keyword_score * self.config['keyword_weight'])
            
            # 重要性加权
            importance_boost = 1 + memory.importance_score * 0.2
            weighted_score = base_score * importance_boost
            
            # 时间衰减调整
            age_days = (datetime.now() - memory.created_at).days
            time_decay = np.exp(-age_days / self.config['time_decay_half_life'])
            final_score = weighted_score * 0.7 + weighted_score * time_decay * 0.3
            
            scored_results.append({
                'id': mem_id,
                'score': final_score,
                'semantic_score': semantic_sim,
                'keyword_score': keyword_score,
                'importance': memory.importance_score,
                'memory': memory
            })
        
        # 排序并返回Top-K
        scored_results.sort(key=lambda x: x['score'], reverse=True)
        return scored_results[:top_k]
    
    def _keyword_filter(self, query_keywords: List[str], 
                        memory_type: Optional[str] = None) -> Set[str]:
        """关键词预过滤"""
        candidate_ids = set()
        
        for kw in query_keywords:
            candidate_ids.update(self.inverted_index.get(kw, set()))
        
        # 类型过滤
        if memory_type and candidate_ids:
            type_ids = self.type_index.get(memory_type, set())
            candidate_ids = candidate_ids & type_ids
        
        return candidate_ids
    
    def _compute_query_vector(self, keywords: List[str]) -> np.ndarray:
        """计算查询向量"""
        vector = np.zeros(128)
        
        for kw in keywords:
            hash_val = int(hashlib.md5(kw.encode()).hexdigest(), 16)
            for i in range(128):
                if (hash_val >> i) & 1:
                    vector[i] += 1
        
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector


class MemoryCompressionSystem:
    """
    记忆压缩系统主类
    整合所有组件，提供统一接口
    """
    
    def __init__(self, storage_dir: str = "./compressed_memory", config: Dict = None):
        self.storage_dir = storage_dir
        # 合并默认配置和用户配置
        self.config = {**DEFAULT_CONFIG, **(config or {})}
        
        # 初始化组件
        self.scorer = MemoryImportanceScorer(self.config)
        self.compressor = MemoryCompressor(self.config)
        self.duplicate_detector = DuplicateDetector(
            self.config.get('semantic_similarity_threshold', 0.85)
        )
        self.retriever = HybridRetriever(self.config)
        
        # 内存存储
        self.memories: Dict[str, MemoryRecord] = {}
        
        # 确保存储目录存在
        os.makedirs(storage_dir, exist_ok=True)
    
    def add_memory(self, content: str, source: str, memory_type: str = "general",
                   metadata: Dict = None) -> MemoryRecord:
        """添加新记忆"""
        # 生成ID
        memory_id = hashlib.md5(f"{content}:{source}:{datetime.now()}".encode()).hexdigest()[:16]
        
        # 提取元数据
        metadata = metadata or {}
        keywords = TextProcessor.extract_keywords(content, top_k=20)
        entities = TextProcessor.extract_entities(content)
        
        # 创建记忆记录
        memory = MemoryRecord(
            id=memory_id,
            content_full=content,
            memory_type=memory_type,
            source=source,
            keywords=keywords,
            entities=entities,
            categories=metadata.get('categories', []),
            context=metadata.get('context', {}),
            user_marked_important=metadata.get('user_marked_important', False)
        )
        
        # 计算重要性并压缩
        importance, factors = self.scorer.calculate_importance(memory)
        memory.importance_score = importance
        memory.importance_factors = factors
        
        memory = self.compressor.compress(memory, importance)
        
        # 存储
        self.memories[memory_id] = memory
        
        return memory
    
    def search(self, query: str, top_k: int = 5, 
               memory_type: Optional[str] = None) -> List[Dict]:
        """搜索记忆"""
        # 更新访问统计
        results = self.retriever.retrieve(query, self.memories, top_k, memory_type)
        
        for result in results:
            memory = result['memory']
            memory.access_count += 1
            memory.last_accessed = datetime.now()
            memory.access_pattern.append(datetime.now())
        
        return results
    
    def compress_all(self, target_ratio: float = 0.6):
        """压缩所有记忆"""
        for memory in self.memories.values():
            # 重新计算重要性（考虑新的访问统计）
            importance, factors = self.scorer.calculate_importance(memory)
            memory.importance_score = importance
            memory.importance_factors = factors
            
            # 重新压缩
            self.compressor.compress(memory, importance)
        
        # 计算压缩统计
        total_original = sum(m.original_length for m in self.memories.values())
        total_compressed = sum(m.compressed_length for m in self.memories.values())
        actual_ratio = total_compressed / total_original if total_original > 0 else 0
        
        return {
            'total_memories': len(self.memories),
            'total_original_bytes': total_original,
            'total_compressed_bytes': total_compressed,
            'compression_ratio': actual_ratio,
            'target_ratio': target_ratio
        }
    
    def deduplicate(self) -> Dict:
        """去重"""
        duplicates = self.duplicate_detector.find_duplicates(list(self.memories.values()))
        
        removed_ids = []
        for id1, id2, similarity in duplicates:
            if id1 in self.memories and id2 in self.memories:
                merged = self.duplicate_detector.merge_duplicates(
                    self.memories[id1], self.memories[id2]
                )
                self.memories[id1] = merged
                del self.memories[id2]
                removed_ids.append(id2)
        
        return {
            'duplicates_found': len(duplicates),
            'removed_ids': removed_ids
        }
    
    def save(self):
        """保存到磁盘"""
        data = {
            'memories': {k: v.to_dict() for k, v in self.memories.items()},
            'config': self.config,
            'saved_at': datetime.now().isoformat()
        }
        
        filepath = os.path.join(self.storage_dir, 'memory_store.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=2)
    
    def load(self):
        """从磁盘加载"""
        filepath = os.path.join(self.storage_dir, 'memory_store.json')
        if not os.path.exists(filepath):
            return False
        
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.memories = {
            k: MemoryRecord.from_dict(v) for k, v in data['memories'].items()
        }
        self.config = data.get('config', DEFAULT_CONFIG)
        
        # 重建索引
        self.retriever.build_index(list(self.memories.values()))
        
        return True
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        if not self.memories:
            return {'total_memories': 0}
        
        total_original = sum(m.original_length for m in self.memories.values())
        total_compressed = sum(m.compressed_length for m in self.memories.values())
        
        compression_levels = defaultdict(int)
        for m in self.memories.values():
            compression_levels[m.compression_level] += 1
        
        return {
            'total_memories': len(self.memories),
            'total_original_bytes': total_original,
            'total_compressed_bytes': total_compressed,
            'overall_compression_ratio': total_compressed / total_original if total_original > 0 else 0,
            'compression_level_distribution': dict(compression_levels),
            'avg_importance_score': sum(m.importance_score for m in self.memories.values()) / len(self.memories),
            'total_access_count': sum(m.access_count for m in self.memories.values())
        }


# ============ 测试与演示 ============

def demo():
    """演示记忆压缩系统的功能"""
    print("=" * 60)
    print("长期记忆压缩优化系统 v2.0 - 功能演示")
    print("=" * 60)
    
    # 创建系统实例
    system = MemoryCompressionSystem(storage_dir="./demo_memory")
    
    # 示例记忆数据
    sample_memories = [
        {
            "content": """
            2026-02-27 重要决策记录
            
            今天董事长兰山决定启动记忆系统优化项目。这是一个关键决策，将显著提升团队的长期记忆能力。
            项目目标包括：
            1. 实现记忆压缩，减少60%存储空间
            2. 优化检索效率，提升4倍速度
            3. 建立重要性评分机制
            
            这是一个战略性决策，需要全力以赴。
            """,
            "source": "decisions/memory_optimization.md",
            "type": "decision",
            "metadata": {"user_marked_important": True}
        },
        {
            "content": """
            日常会议记录 - 团队周会
            
            参会人员：CEO Kimi Claw, research-lead, dev-arch
            会议时间：2026-02-27 10:00
            
            讨论内容：
            - review上周进度
            - 讨论下周计划
            - 资源分配调整
            
            没有特别重要的决策，主要是例行沟通。
            """,
            "source": "meetings/weekly_2026-02-27.md",
            "type": "meeting",
            "metadata": {}
        },
        {
            "content": """
            技术调研笔记 - ChromaDB向量数据库
            
            ChromaDB是一个开源的向量数据库，适合存储和检索高维向量。
            主要特点：
            - 支持多种相似度度量（余弦、欧氏距离等）
            - 提供Python客户端，易于集成
            - 支持持久化存储
            - 支持元数据过滤
            
            在我们的记忆系统中，ChromaDB可以用来存储记忆向量，实现语义检索。
            """,
            "source": "research/chromadb_notes.md",
            "type": "research",
            "metadata": {}
        },
        {
            "content": """
            重复内容测试 - 记忆压缩的重要性
            
            记忆压缩对于长期记忆系统非常重要。通过压缩，我们可以：
            1. 减少存储空间
            2. 提高检索效率
            3. 保留关键信息
            
            压缩算法需要根据重要性进行分层处理。
            """,
            "source": "test/duplicate_test.md",
            "type": "test",
            "metadata": {}
        },
        {
            "content": """
            记忆压缩的重要性说明
            
            记忆压缩是长期记忆系统的核心功能。它的重要性体现在：
            1. 可以显著减少存储空间需求
            2. 能够提高检索速度和效率
            3. 帮助保留最关键的信息
            
            我们需要根据记忆的重要性进行分层压缩处理。
            """,
            "source": "docs/compression_importance.md",
            "type": "documentation",
            "metadata": {}
        }
    ]
    
    print("\n📥 步骤1: 添加示例记忆")
    print("-" * 40)
    
    for i, mem_data in enumerate(sample_memories, 1):
        memory = system.add_memory(
            content=mem_data["content"],
            source=mem_data["source"],
            memory_type=mem_data["type"],
            metadata=mem_data["metadata"]
        )
        print(f"  [{i}] {memory.memory_type:12} | "
              f"重要性: {memory.importance_score:.2f} | "
              f"压缩级别: {memory.compression_level} | "
              f"来源: {memory.source}")
    
    print("\n📊 步骤2: 查看系统统计")
    print("-" * 40)
    stats = system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 步骤3: 测试检索功能")
    print("-" * 40)
    
    test_queries = [
        "记忆压缩优化",
        "ChromaDB向量数据库",
        "重要决策",
        "团队会议"
    ]
    
    for query in test_queries:
        print(f"\n  查询: '{query}'")
        results = system.search(query, top_k=3)
        for r in results:
            memory = r['memory']
            print(f"    → [{memory.memory_type}] 相关度: {r['score']:.3f} | "
                  f"重要性: {memory.importance_score:.2f} | "
                  f"来源: {memory.source}")
    
    print("\n🗑️  步骤4: 去重检测")
    print("-" * 40)
    dedup_result = system.deduplicate()
    print(f"  发现重复: {dedup_result['duplicates_found']} 对")
    print(f"  移除ID: {dedup_result['removed_ids']}")
    
    print("\n💾 步骤5: 保存系统状态")
    print("-" * 40)
    system.save()
    print(f"  已保存到: {system.storage_dir}/memory_store.json")
    
    print("\n📈 最终统计")
    print("-" * 40)
    final_stats = system.get_stats()
    for key, value in final_stats.items():
        print(f"  {key}: {value}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return system


if __name__ == "__main__":
    demo()
