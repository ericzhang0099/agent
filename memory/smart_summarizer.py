#!/usr/bin/env python3
"""
智能摘要系统 v3.0
实现分层压缩、重要性评分、自适应摘要
"""

import re
import heapq
from typing import List, Dict, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
from collections import defaultdict
import numpy as np


@dataclass
class SummaryLevel:
    """摘要级别定义"""
    level: int
    name: str
    description: str
    target_ratio: float  # 目标压缩率
    min_importance: float  # 最低重要性阈值


# 预定义摘要级别
SUMMARY_LEVELS = {
    0: SummaryLevel(0, "完整", "保留完整原文", 1.0, 4.5),
    1: SummaryLevel(1, "详细", "保留80%信息", 0.8, 3.5),
    2: SummaryLevel(2, "要点", "保留50%信息", 0.5, 2.5),
    3: SummaryLevel(3, "精简", "保留20%信息", 0.2, 1.5),
    4: SummaryLevel(4, "索引", "仅保留元数据", 0.0, 0.0),
}


class TextProcessor:
    """文本处理工具"""
    
    # 停用词
    STOPWORDS = {
        'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'must', 'shall', 'can', 'need', 'dare',
        'ought', 'used', 'to', 'of', 'in', 'for', 'on', 'with', 'at', 'by',
        'from', 'as', 'into', 'through', 'during', 'before', 'after', 'above',
        'below', 'between', 'under', 'again', 'further', 'then', 'once', 'here',
        'there', 'when', 'where', 'why', 'how', 'all', 'each', 'few', 'more',
        'most', 'other', 'some', 'such', 'no', 'nor', 'not', 'only', 'own',
        'same', 'so', 'than', 'too', 'very', 'just', '的', '了', '在', '是',
        '和', '有', '我', '你', '他', '她', '它', '我们', '你们', '他们',
        '这', '那', '这些', '那些', '一个', '这个', '那个', '什么', '怎么'
    }
    
    # 决策关键词
    DECISION_KEYWORDS = [
        '决策', '决定', '选择', '批准', '拒绝', '关键', '重要', '战略',
        'decision', 'decide', 'choose', 'approve', 'reject', 'critical',
        'important', 'strategic', 'approve', 'authorize', 'finalize'
    ]
    
    # 行动关键词
    ACTION_KEYWORDS = [
        '完成', '实施', '启动', '部署', '发布', '上线', '交付',
        'complete', 'implement', 'launch', 'deploy', 'release', 'deliver'
    ]
    
    @classmethod
    def split_sentences(cls, text: str) -> List[str]:
        """分割句子 (支持中英文)"""
        # 中文句子结束符
        chinese_ends = r'[。！？；]'
        # 英文句子结束符
        english_ends = r'[.!?;]'
        
        # 保护常见缩写
        text = re.sub(r'(Mr|Mrs|Dr|Prof|Inc|Ltd|Jr|Sr|vs|Vol|vol)\.', r'\1&lt;DOT&gt;', text)
        
        # 分割句子
        pattern = f'({chinese_ends}|{english_ends})'
        sentences = re.split(pattern, text)
        
        # 合并分割符和句子
        result = []
        i = 0
        while i < len(sentences):
            if i + 1 < len(sentences) and re.match(pattern, sentences[i + 1]):
                result.append(sentences[i] + sentences[i + 1])
                i += 2
            else:
                if sentences[i].strip():
                    result.append(sentences[i])
                i += 1
        
        # 恢复缩写
        result = [s.replace('&lt;DOT&gt;', '.') for s in result]
        
        return [s.strip() for s in result if s.strip()]
    
    @classmethod
    def extract_keywords(cls, text: str, top_k: int = 20) -> List[str]:
        """提取关键词 (TF-IDF近似)"""
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 过滤停用词和短词
        filtered = [w for w in words 
                   if w not in cls.STOPWORDS and len(w) > 1]
        
        # 统计词频
        word_freq = defaultdict(int)
        for word in filtered:
            word_freq[word] += 1
        
        # 返回Top-K
        top_words = heapq.nlargest(top_k, word_freq.items(), key=lambda x: x[1])
        return [word for word, _ in top_words]
    
    @classmethod
    def extract_entities(cls, text: str) -> List[Dict]:
        """提取实体 (简化版)"""
        entities = []
        
        # 提取引号内的内容
        quoted = re.findall(r'["""]([^"""]+)["""]', text)
        for q in quoted:
            entities.append({"name": q, "type": "QUOTED"})
        
        # 提取大写单词组合 (可能是专有名词)
        capitalized = re.findall(r'\b[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*\b', text)
        for c in capitalized:
            if c.lower() not in cls.STOPWORDS:
                entities.append({"name": c, "type": "PROPER_NOUN"})
        
        # 提取数字和度量
        numbers = re.findall(r'\b\d+(?:\.\d+)?\s*(?:%|percent|个|条|次|天|小时|分钟)\b', text)
        for n in numbers:
            entities.append({"name": n, "type": "METRIC"})
        
        # 去重
        seen = set()
        unique_entities = []
        for e in entities:
            key = e["name"].lower()
            if key not in seen:
                seen.add(key)
                unique_entities.append(e)
        
        return unique_entities[:20]
    
    @classmethod
    def calculate_similarity(cls, text1: str, text2: str) -> float:
        """计算文本相似度 (Jaccard)"""
        keywords1 = set(cls.extract_keywords(text1, top_k=50))
        keywords2 = set(cls.extract_keywords(text2, top_k=50))
        
        if not keywords1 or not keywords2:
            return 0.0
        
        intersection = len(keywords1 & keywords2)
        union = len(keywords1 | keywords2)
        
        return intersection / union if union > 0 else 0.0


class ImportanceScorer:
    """
    记忆重要性评分器
    
    评分维度:
    1. 用户显式标记 (权重1.0)
    2. 决策关键度 (权重1.2)
    3. 访问频率 (权重0.8)
    4. 信息密度 (权重0.6)
    5. 时效性 (权重0.7)
    6. 行动导向 (权重0.5)
    """
    
    def __init__(self):
        self.weights = {
            'user_explicit': 1.0,
            'decision_critical': 1.2,
            'access_frequency': 0.8,
            'information_density': 0.6,
            'recency': 0.7,
            'action_oriented': 0.5,
        }
    
    def calculate(self, 
                  content: str,
                  access_count: int = 0,
                  user_marked: bool = False,
                  age_days: int = 0) -> Tuple[float, Dict]:
        """
        计算重要性评分
        
        Returns:
            (总分, 各维度得分)
        """
        scores = {}
        
        # 1. 用户显式标记 (0-1.0)
        scores['user_explicit'] = 1.0 if user_marked else 0
        
        # 2. 决策关键度 (0-1.0)
        content_lower = content.lower()
        decision_matches = sum(1 for kw in TextProcessor.DECISION_KEYWORDS 
                              if kw in content_lower)
        scores['decision_critical'] = min(decision_matches * 0.25, 1.0)
        
        # 3. 访问频率 (0-1.0)
        scores['access_frequency'] = min(access_count / 20, 1.0)
        
        # 4. 信息密度 (0-1.0)
        entities = TextProcessor.extract_entities(content)
        keywords = TextProcessor.extract_keywords(content, top_k=50)
        entity_score = min(len(entities) * 0.1, 0.5)
        keyword_score = min(len(keywords) / 20, 0.5)
        scores['information_density'] = entity_score + keyword_score
        
        # 5. 时效性 - 艾宾浩斯遗忘曲线 (0-1.0)
        scores['recency'] = np.exp(-age_days / 30)  # 30天半衰期
        
        # 6. 行动导向 (0-1.0)
        action_matches = sum(1 for kw in TextProcessor.ACTION_KEYWORDS 
                            if kw in content_lower)
        scores['action_oriented'] = min(action_matches * 0.3, 1.0)
        
        # 计算加权总分 (0-5.0)
        total = sum(scores[k] * self.weights[k] for k in scores)
        
        return round(min(total, 5.0), 2), scores
    
    def get_compression_level(self, importance: float) -> int:
        """根据重要性确定压缩级别"""
        for level in sorted(SUMMARY_LEVELS.keys()):
            sl = SUMMARY_LEVELS[level]
            if importance >= sl.min_importance:
                return level
        return 4  # 默认最低级别


class SmartSummarizer:
    """
    智能摘要器
    实现分层摘要、提取式摘要、关键要点提取
    """
    
    def __init__(self):
        self.text_processor = TextProcessor()
        self.importance_scorer = ImportanceScorer()
    
    def summarize(self, 
                  content: str,
                  importance: float = 0,
                  level: Optional[int] = None) -> Dict:
        """
        智能摘要主函数
        
        Args:
            content: 原文内容
            importance: 重要性评分 (0-5)
            level: 指定摘要级别 (0-4), None则自动选择
        
        Returns:
            摘要结果字典
        """
        # 确定压缩级别
        if level is None:
            level = self.importance_scorer.get_compression_level(importance)
        
        summary_level = SUMMARY_LEVELS.get(level, SUMMARY_LEVELS[4])
        
        # 根据级别生成摘要
        if level == 0:
            return self._level_0_full(content, summary_level)
        elif level == 1:
            return self._level_1_detailed(content, summary_level)
        elif level == 2:
            return self._level_2_keypoints(content, summary_level)
        elif level == 3:
            return self._level_3_concise(content, summary_level)
        else:
            return self._level_4_index(content, summary_level)
    
    def _level_0_full(self, content: str, level: SummaryLevel) -> Dict:
        """级别0: 完整保留"""
        return {
            "level": 0,
            "level_name": level.name,
            "full_text": content,
            "summary": None,
            "keypoints": [],
            "entities": self.text_processor.extract_entities(content),
            "compression_ratio": 1.0,
            "original_length": len(content),
            "compressed_length": len(content)
        }
    
    def _level_1_detailed(self, content: str, level: SummaryLevel) -> Dict:
        """级别1: 详细摘要 (保留80%)"""
        sentences = self.text_processor.split_sentences(content)
        
        if len(sentences) <= 3:
            return self._level_0_full(content, level)
        
        # 选择最重要的80%句子
        k = max(1, int(len(sentences) * level.target_ratio))
        selected = self._select_top_sentences(sentences, k)
        
        summary = ' '.join(selected)
        
        return {
            "level": 1,
            "level_name": level.name,
            "full_text": content,  # 仍保留完整文本
            "summary": summary,
            "keypoints": selected[:5],
            "entities": self.text_processor.extract_entities(content),
            "compression_ratio": len(summary) / len(content) if content else 1.0,
            "original_length": len(content),
            "compressed_length": len(summary)
        }
    
    def _level_2_keypoints(self, content: str, level: SummaryLevel) -> Dict:
        """级别2: 关键要点 (保留50%)"""
        sentences = self.text_processor.split_sentences(content)
        
        if len(sentences) <= 5:
            keypoints = sentences
        else:
            # 选择最重要的50%句子作为要点
            k = max(1, int(len(sentences) * level.target_ratio))
            keypoints = self._select_top_sentences(sentences, k)
        
        # 生成简短摘要
        summary = ' '.join(keypoints[:3]) if keypoints else ""
        
        return {
            "level": 2,
            "level_name": level.name,
            "full_text": None,  # 不保留完整文本
            "summary": summary,
            "keypoints": keypoints,
            "entities": self.text_processor.extract_entities(content),
            "compression_ratio": sum(len(k) for k in keypoints) / len(content) if content else 0,
            "original_length": len(content),
            "compressed_length": sum(len(k) for k in keypoints)
        }
    
    def _level_3_concise(self, content: str, level: SummaryLevel) -> Dict:
        """级别3: 精简 (保留20%)"""
        # 提取关键实体
        entities = self.text_processor.extract_entities(content)
        
        # 提取最重要的句子
        sentences = self.text_processor.split_sentences(content)
        
        if len(sentences) <= 2:
            keypoints = sentences
        else:
            k = max(1, int(len(sentences) * level.target_ratio))
            keypoints = self._select_top_sentences(sentences, k)
        
        # 构建精简描述
        entity_names = [e["name"] for e in entities[:5]]
        concise = f"涉及: {', '.join(entity_names)}" if entity_names else ""
        
        return {
            "level": 3,
            "level_name": level.name,
            "full_text": None,
            "summary": concise,
            "keypoints": keypoints,
            "entities": entities,
            "compression_ratio": 0.2,
            "original_length": len(content),
            "compressed_length": len(concise) + sum(len(k) for k in keypoints)
        }
    
    def _level_4_index(self, content: str, level: SummaryLevel) -> Dict:
        """级别4: 仅索引"""
        entities = self.text_processor.extract_entities(content)
        keywords = self.text_processor.extract_keywords(content, top_k=10)
        
        return {
            "level": 4,
            "level_name": level.name,
            "full_text": None,
            "summary": None,
            "keypoints": [],
            "entities": entities,
            "keywords": keywords,
            "compression_ratio": 0.0,
            "original_length": len(content),
            "compressed_length": 0
        }
    
    def _select_top_sentences(self, sentences: List[str], k: int) -> List[str]:
        """选择最重要的k个句子"""
        if len(sentences) <= k:
            return sentences
        
        # 计算句子重要性分数
        all_text = ' '.join(sentences)
        global_keywords = set(self.text_processor.extract_keywords(all_text, top_k=30))
        
        sentence_scores = []
        for i, sent in enumerate(sentences):
            score = 0
            
            # 关键词匹配
            sent_keywords = set(self.text_processor.extract_keywords(sent, top_k=10))
            score += len(sent_keywords & global_keywords) * 2
            
            # 位置权重
            if i == 0:  # 首句
                score += 5
            elif i == len(sentences) - 1:  # 尾句
                score += 3
            elif i < len(sentences) * 0.2:  # 前20%
                score += 2
            
            # 决策关键词加分
            sent_lower = sent.lower()
            for kw in TextProcessor.DECISION_KEYWORDS:
                if kw in sent_lower:
                    score += 3
            
            # 行动关键词加分
            for kw in TextProcessor.ACTION_KEYWORDS:
                if kw in sent_lower:
                    score += 2
            
            # 长度惩罚
            sent_len = len(sent)
            if sent_len < 10:
                score -= 2
            elif sent_len > 200:
                score -= 1
            
            sentence_scores.append((i, score))
        
        # 选择Top-K
        top_sentences = heapq.nlargest(k, sentence_scores, key=lambda x: x[1])
        top_sentences.sort(key=lambda x: x[0])  # 按原文顺序排列
        
        return [sentences[i] for i, _ in top_sentences]
    
    def batch_summarize(self, 
                       contents: List[str],
                       importances: Optional[List[float]] = None) -> List[Dict]:
        """批量摘要"""
        results = []
        for i, content in enumerate(contents):
            importance = importances[i] if importances and i < len(importances) else 0
            results.append(self.summarize(content, importance))
        return results


class MemoryCompressionEngine:
    """
    记忆压缩引擎
    整合重要性评分和智能摘要
    """
    
    def __init__(self, target_compression: float = 0.6):
        self.summarizer = SmartSummarizer()
        self.scorer = ImportanceScorer()
        self.target_compression = target_compression
    
    def compress(self, 
                 content: str,
                 access_count: int = 0,
                 user_marked: bool = False,
                 age_days: int = 0) -> Dict:
        """
        压缩记忆
        
        Returns:
            压缩结果，包含所有层级的内容
        """
        # 1. 计算重要性
        importance, factors = self.scorer.calculate(
            content=content,
            access_count=access_count,
            user_marked=user_marked,
            age_days=age_days
        )
        
        # 2. 根据重要性确定压缩级别
        level = self.scorer.get_compression_level(importance)
        
        # 3. 生成摘要
        summary_result = self.summarizer.summarize(content, importance, level)
        
        return {
            "importance_score": importance,
            "importance_factors": factors,
            "compression_level": level,
            **summary_result
        }
    
    def decompress(self, compressed: Dict) -> str:
        """
        解压记忆 (获取最佳可用内容)
        """
        if compressed.get("full_text"):
            return compressed["full_text"]
        elif compressed.get("summary"):
            return compressed["summary"]
        elif compressed.get("keypoints"):
            return '\n'.join(compressed["keypoints"])
        elif compressed.get("entities"):
            entities = [e["name"] for e in compressed["entities"]]
            return f"相关实体: {', '.join(entities)}"
        else:
            return "[已归档记忆]"
    
    def get_stats(self, compressed_memories: List[Dict]) -> Dict:
        """获取压缩统计"""
        if not compressed_memories:
            return {}
        
        total_original = sum(m.get("original_length", 0) for m in compressed_memories)
        total_compressed = sum(m.get("compressed_length", 0) for m in compressed_memories)
        
        level_distribution = defaultdict(int)
        for m in compressed_memories:
            level_distribution[m.get("compression_level", 4)] += 1
        
        return {
            "total_memories": len(compressed_memories),
            "total_original_bytes": total_original,
            "total_compressed_bytes": total_compressed,
            "compression_ratio": total_compressed / total_original if total_original > 0 else 0,
            "level_distribution": dict(level_distribution),
            "avg_importance": sum(m.get("importance_score", 0) for m in compressed_memories) / len(compressed_memories)
        }


def demo():
    """演示智能摘要系统"""
    print("=" * 70)
    print("智能摘要系统 v3.0 - 演示")
    print("=" * 70)
    
    # 测试文本
    test_memories = [
        {
            "content": """
            2026-02-27 重要决策记录
            
            今天董事长兰山在战略会议上做出了关键决策：正式启动KCGS记忆系统升级项目。
            这是一个具有战略意义的决定，将决定公司未来3年的技术竞争力。
            
            项目目标：
            1. 实现记忆压缩，减少60%存储空间
            2. 优化检索效率，提升4倍速度
            3. 建立重要性评分机制
            4. 集成知识图谱能力
            
            预算：500万元
            时间线：3个月完成核心功能
            团队：10人精英团队
            
            兰山强调："这是公司最重要的技术投资之一，必须全力以赴。"
            """,
            "user_marked": True,
            "access_count": 15,
            "age_days": 0
        },
        {
            "content": """
            日常开发日志 - 2026-02-27
            
            今天完成了以下工作：
            - 修复了3个bug
            - 完成了ChromaDB的性能测试
            - 编写了技术文档
            
            明天计划：
            - 开始Pinecone集成
            - 继续优化检索算法
            
            没有遇到重大问题。
            """,
            "user_marked": False,
            "access_count": 2,
            "age_days": 0
        },
        {
            "content": """
            团队周会记录
            
            参会人员：CEO Kimi Claw, 研发负责人, 产品经理
            时间：2026-02-27 10:00
            
            讨论内容：
            - 回顾上周进度
            - 讨论下周计划
            - 资源分配调整
            
            主要是例行沟通，没有重要决策。
            """,
            "user_marked": False,
            "access_count": 1,
            "age_days": 7
        }
    ]
    
    engine = MemoryCompressionEngine()
    
    print("\n📊 记忆压缩演示\n")
    print("-" * 70)
    
    compressed_memories = []
    
    for i, mem in enumerate(test_memories, 1):
        print(f"\n【记忆 {i}】")
        
        # 压缩
        result = engine.compress(
            content=mem["content"],
            access_count=mem["access_count"],
            user_marked=mem["user_marked"],
            age_days=mem["age_days"]
        )
        
        compressed_memories.append(result)
        
        print(f"  重要性评分: {result['importance_score']}/5.0")
        print(f"  压缩级别: {result['compression_level']} ({result['level_name']})")
        print(f"  压缩率: {result['compression_ratio']:.1%}")
        print(f"  原始长度: {result['original_length']} 字符")
        print(f"  压缩后: {result['compressed_length']} 字符")
        
        if result.get("keypoints"):
            print(f"\n  关键要点:")
            for j, point in enumerate(result["keypoints"][:3], 1):
                print(f"    {j}. {point[:50]}...")
        
        if result.get("entities"):
            entities = [e["name"] for e in result["entities"][:5]]
            print(f"\n  关键实体: {', '.join(entities)}")
    
    print("\n" + "=" * 70)
    print("📈 压缩统计")
    print("=" * 70)
    
    stats = engine.get_stats(compressed_memories)
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.2%}" if "ratio" in key else f"  {key}: {value:.2f}")
        else:
            print(f"  {key}: {value}")
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


if __name__ == "__main__":
    demo()
