#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
reflection_system.py - 反思日志系统

功能：
1. 反思日志记录与管理
2. 经验提取与总结
3. 反思模式识别
4. 改进建议生成

作者：元认知研究项目
版本：1.0.0
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict
import threading


class ReflectionLevel(Enum):
    """反思深度级别"""
    SURFACE = auto()      # 表层反思 - 发生了什么
    ANALYTICAL = auto()   # 分析反思 - 为什么发生
    CRITICAL = auto()     # 批判反思 - 如何改进
    TRANSFORMATIVE = auto()  # 变革反思 - 深层信念改变


class ReflectionType(Enum):
    """反思类型"""
    ACTION = "action"           # 行动反思
    DECISION = "decision"       # 决策反思
    EMOTION = "emotion"         # 情绪反思
    LEARNING = "learning"       # 学习反思
    INTERACTION = "interaction" # 交互反思
    SYSTEM = "system"           # 系统反思


@dataclass
class ReflectionEntry:
    """反思日志条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    timestamp: datetime = field(default_factory=datetime.now)
    level: ReflectionLevel = ReflectionLevel.ANALYTICAL
    type: ReflectionType = ReflectionType.ACTION
    
    # 反思内容
    context: str = ""           # 情境描述
    what_happened: str = ""     # 发生了什么
    why_happened: str = ""      # 为什么发生
    feelings: str = ""          # 感受如何
    lessons_learned: str = ""   # 学到什么
    action_items: List[str] = field(default_factory=list)
    
    # 元数据
    tags: List[str] = field(default_factory=list)
    related_entries: List[str] = field(default_factory=list)
    effectiveness_score: Optional[float] = None  # 0-10
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        data = asdict(self)
        data['level'] = self.level.name
        data['type'] = self.type.value
        data['timestamp'] = self.timestamp.isoformat()
        return data
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'ReflectionEntry':
        """从字典创建"""
        data['level'] = ReflectionLevel[data['level']]
        data['type'] = ReflectionType(data['type'])
        data['timestamp'] = datetime.fromisoformat(data['timestamp'])
        return cls(**data)


@dataclass
class ExperiencePattern:
    """经验模式"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    name: str = ""
    description: str = ""
    frequency: int = 0
    first_seen: datetime = field(default_factory=datetime.now)
    last_seen: datetime = field(default_factory=datetime.now)
    related_entries: List[str] = field(default_factory=list)
    insights: List[str] = field(default_factory=list)
    recommendations: List[str] = field(default_factory=list)


class ReflectionJournal:
    """反思日志系统"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "reflection_journal.json"
        self.entries: Dict[str, ReflectionEntry] = {}
        self.patterns: Dict[str, ExperiencePattern] = {}
        self.tags_index: Dict[str, List[str]] = defaultdict(list)
        self.type_index: Dict[ReflectionType, List[str]] = defaultdict(list)
        self._lock = threading.RLock()
        self._load()
    
    def add_entry(self, entry: ReflectionEntry) -> str:
        """添加反思条目"""
        with self._lock:
            self.entries[entry.id] = entry
            
            # 更新索引
            for tag in entry.tags:
                self.tags_index[tag].append(entry.id)
            self.type_index[entry.type].append(entry.id)
            
            self._save()
            return entry.id
    
    def create_entry(
        self,
        context: str,
        what_happened: str,
        level: ReflectionLevel = ReflectionLevel.ANALYTICAL,
        type: ReflectionType = ReflectionType.ACTION,
        why_happened: str = "",
        feelings: str = "",
        lessons_learned: str = "",
        action_items: Optional[List[str]] = None,
        tags: Optional[List[str]] = None
    ) -> ReflectionEntry:
        """创建并添加反思条目"""
        entry = ReflectionEntry(
            level=level,
            type=type,
            context=context,
            what_happened=what_happened,
            why_happened=why_happened,
            feelings=feelings,
            lessons_learned=lessons_learned,
            action_items=action_items or [],
            tags=tags or []
        )
        self.add_entry(entry)
        return entry
    
    def get_entry(self, entry_id: str) -> Optional[ReflectionEntry]:
        """获取反思条目"""
        return self.entries.get(entry_id)
    
    def search_entries(
        self,
        type: Optional[ReflectionType] = None,
        level: Optional[ReflectionLevel] = None,
        tags: Optional[List[str]] = None,
        start_date: Optional[datetime] = None,
        end_date: Optional[datetime] = None,
        keyword: Optional[str] = None
    ) -> List[ReflectionEntry]:
        """搜索反思条目"""
        results = list(self.entries.values())
        
        if type:
            results = [e for e in results if e.type == type]
        
        if level:
            results = [e for e in results if e.level == level]
        
        if tags:
            tag_set = set(tags)
            results = [e for e in results if tag_set.intersection(set(e.tags))]
        
        if start_date:
            results = [e for e in results if e.timestamp >= start_date]
        
        if end_date:
            results = [e for e in results if e.timestamp <= end_date]
        
        if keyword:
            keyword_lower = keyword.lower()
            results = [
                e for e in results
                if (keyword_lower in e.context.lower() or
                    keyword_lower in e.what_happened.lower() or
                    keyword_lower in e.why_happened.lower() or
                    keyword_lower in e.lessons_learned.lower())
            ]
        
        return sorted(results, key=lambda e: e.timestamp, reverse=True)
    
    def get_recent_entries(self, days: int = 7, limit: int = 10) -> List[ReflectionEntry]:
        """获取最近的反思条目"""
        cutoff = datetime.now() - timedelta(days=days)
        entries = [e for e in self.entries.values() if e.timestamp >= cutoff]
        return sorted(entries, key=lambda e: e.timestamp, reverse=True)[:limit]
    
    def analyze_patterns(self) -> List[ExperiencePattern]:
        """分析反思模式"""
        patterns = []
        
        # 按类型分析
        for ref_type in ReflectionType:
            type_entries = self.search_entries(type=ref_type)
            if len(type_entries) >= 3:
                pattern = self._extract_pattern(ref_type.value, type_entries)
                if pattern:
                    patterns.append(pattern)
        
        # 按标签分析
        tag_patterns = self._analyze_tag_patterns()
        patterns.extend(tag_patterns)
        
        # 保存模式
        for pattern in patterns:
            self.patterns[pattern.id] = pattern
        
        return patterns
    
    def _extract_pattern(
        self,
        name: str,
        entries: List[ReflectionEntry]
    ) -> Optional[ExperiencePattern]:
        """从条目中提取模式"""
        if not entries:
            return None
        
        # 提取共同主题
        common_lessons = self._find_common_themes(
            [e.lessons_learned for e in entries if e.lessons_learned]
        )
        
        # 生成洞察
        insights = []
        if len(entries) > 5:
            insights.append(f"该类型反思出现频率较高({len(entries)}次)，值得关注")
        
        # 生成建议
        recommendations = []
        if common_lessons:
            recommendations.append(f"关注共同主题: {', '.join(common_lessons[:3])}")
        
        return ExperiencePattern(
            name=f"{name}_pattern",
            description=f"基于{len(entries)}条{name}类型反思的模式",
            frequency=len(entries),
            first_seen=min(e.timestamp for e in entries),
            last_seen=max(e.timestamp for e in entries),
            related_entries=[e.id for e in entries],
            insights=insights,
            recommendations=recommendations
        )
    
    def _analyze_tag_patterns(self) -> List[ExperiencePattern]:
        """分析标签模式"""
        patterns = []
        
        for tag, entry_ids in self.tags_index.items():
            if len(entry_ids) >= 3:
                entries = [self.entries[eid] for eid in entry_ids]
                pattern = ExperiencePattern(
                    name=f"tag_{tag}",
                    description=f"标签'{tag}'相关的反思模式",
                    frequency=len(entries),
                    first_seen=min(e.timestamp for e in entries),
                    last_seen=max(e.timestamp for e in entries),
                    related_entries=entry_ids,
                    insights=[f"'{tag}'是一个 recurring theme，出现{len(entries)}次"],
                    recommendations=[f"考虑为'{tag}'制定专门的改进策略"]
                )
                patterns.append(pattern)
        
        return patterns
    
    def _find_common_themes(self, texts: List[str]) -> List[str]:
        """查找共同主题（简化实现）"""
        if not texts:
            return []
        
        # 简单的关键词提取
        word_freq = defaultdict(int)
        for text in texts:
            words = text.lower().split()
            for word in set(words):
                if len(word) > 3:  # 过滤短词
                    word_freq[word] += 1
        
        # 返回出现频率高的词
        sorted_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)
        return [word for word, freq in sorted_words[:5] if freq >= 2]
    
    def generate_insights(self) -> Dict[str, Any]:
        """生成洞察报告"""
        total_entries = len(self.entries)
        
        if total_entries == 0:
            return {"message": "暂无反思记录"}
        
        # 类型分布
        type_distribution = {
            t.value: len(entries)
            for t, entries in self.type_index.items()
        }
        
        # 级别分布
        level_distribution = defaultdict(int)
        for entry in self.entries.values():
            level_distribution[entry.level.name] += 1
        
        # 时间趋势
        entries_by_week = defaultdict(int)
        for entry in self.entries.values():
            week_key = entry.timestamp.strftime("%Y-W%W")
            entries_by_week[week_key] += 1
        
        # 行动项统计
        total_action_items = sum(
            len(e.action_items) for e in self.entries.values()
        )
        
        return {
            "total_entries": total_entries,
            "type_distribution": dict(type_distribution),
            "level_distribution": dict(level_distribution),
            "entries_by_week": dict(entries_by_week),
            "total_action_items": total_action_items,
            "patterns_identified": len(self.patterns),
            "top_tags": sorted(
                self.tags_index.items(),
                key=lambda x: len(x[1]),
                reverse=True
            )[:5]
        }
    
    def get_improvement_suggestions(self) -> List[Dict[str, Any]]:
        """获取改进建议"""
        suggestions = []
        
        # 基于模式的建议
        for pattern in self.patterns.values():
            if pattern.recommendations:
                suggestions.append({
                    "source": f"模式: {pattern.name}",
                    "suggestions": pattern.recommendations,
                    "priority": "high" if pattern.frequency > 5 else "medium"
                })
        
        # 基于频率的建议
        for ref_type, entries in self.type_index.items():
            if len(entries) > 10:
                suggestions.append({
                    "source": f"高频反思类型: {ref_type.value}",
                    "suggestions": [
                        f"{ref_type.value}类型反思频繁({len(entries)}次)，建议制定系统性改进方案"
                    ],
                    "priority": "high"
                })
        
        return suggestions
    
    def _save(self):
        """保存到文件"""
        try:
            data = {
                "entries": {k: v.to_dict() for k, v in self.entries.items()},
                "patterns": {k: asdict(v) for k, v in self.patterns.items()},
                "tags_index": dict(self.tags_index),
                "type_index": {k.value: v for k, v in self.type_index.items()}
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存反思日志失败: {e}")
    
    def _load(self):
        """从文件加载"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载条目
            for entry_id, entry_data in data.get("entries", {}).items():
                self.entries[entry_id] = ReflectionEntry.from_dict(entry_data)
            
            # 加载模式
            for pattern_id, pattern_data in data.get("patterns", {}).items():
                self.patterns[pattern_id] = ExperiencePattern(**pattern_data)
            
            # 加载索引
            self.tags_index = defaultdict(list, data.get("tags_index", {}))
            type_index_data = data.get("type_index", {})
            for type_val, entry_ids in type_index_data.items():
                self.type_index[ReflectionType(type_val)] = entry_ids
                
        except FileNotFoundError:
            pass  # 文件不存在，使用空状态
        except Exception as e:
            print(f"加载反思日志失败: {e}")


class ReflectionPrompts:
    """反思提示模板"""
    
    TEMPLATES = {
        ReflectionLevel.SURFACE: {
            "prompts": [
                "今天发生了什么重要的事？",
                "我做了什么？",
                "遇到了什么挑战？"
            ]
        },
        ReflectionLevel.ANALYTICAL: {
            "prompts": [
                "为什么会发生这件事？",
                "我当时是怎么想的？",
                "有哪些因素影响了结果？",
                "我学到了什么？"
            ]
        },
        ReflectionLevel.CRITICAL: {
            "prompts": [
                "我的假设是否正确？",
                "有哪些不同的视角？",
                "下次如何做得更好？",
                "需要改变什么方法？"
            ]
        },
        ReflectionLevel.TRANSFORMATIVE: {
            "prompts": [
                "这件事如何改变了我的看法？",
                "我的核心价值观是否受到影响？",
                "这对我未来的方向有何启示？",
                "如何将这次经历转化为成长？"
            ]
        }
    }
    
    @classmethod
    def get_prompts(cls, level: ReflectionLevel) -> List[str]:
        """获取指定级别的反思提示"""
        return cls.TEMPLATES.get(level, {}).get("prompts", [])
    
    @classmethod
    def get_all_prompts(cls) -> Dict[ReflectionLevel, List[str]]:
        """获取所有反思提示"""
        return {level: data["prompts"] for level, data in cls.TEMPLATES.items()}


# 便捷函数
def create_reflection_journal(storage_path: Optional[str] = None) -> ReflectionJournal:
    """创建反思日志实例"""
    return ReflectionJournal(storage_path)


def quick_reflect(
    journal: ReflectionJournal,
    context: str,
    what_happened: str,
    lessons: str = "",
    tags: Optional[List[str]] = None
) -> ReflectionEntry:
    """快速记录反思"""
    return journal.create_entry(
        context=context,
        what_happened=what_happened,
        lessons_learned=lessons,
        tags=tags or ["quick"]
    )


if __name__ == "__main__":
    # 示例用法
    print("=== 反思日志系统 ===\n")
    
    # 创建反思日志
    journal = create_reflection_journal("demo_reflections.json")
    
    # 添加示例反思条目
    entry1 = journal.create_entry(
        context="项目开发",
        what_happened="在实现新功能时遇到了架构设计问题",
        why_happened="前期需求分析不够充分，对扩展性考虑不足",
        lessons_learned="需要在开发前进行更详细的设计评审",
        action_items=["制定设计评审清单", "建立架构决策记录"],
        tags=["development", "architecture", "lesson"],
        level=ReflectionLevel.CRITICAL,
        type=ReflectionType.ACTION
    )
    print(f"添加反思条目: {entry1.id}")
    
    entry2 = journal.create_entry(
        context="团队协作",
        what_happened="与团队成员就技术方案产生分歧",
        why_happened="沟通方式不够开放，没有充分听取对方意见",
        lessons_learned="技术讨论应该更加开放，尊重不同观点",
        action_items=["改进沟通技巧", "建立技术讨论规范"],
        tags=["collaboration", "communication"],
        level=ReflectionLevel.ANALYTICAL,
        type=ReflectionType.INTERACTION
    )
    print(f"添加反思条目: {entry2.id}")
    
    # 分析模式
    print("\n=== 模式分析 ===")
    patterns = journal.analyze_patterns()
    for pattern in patterns:
        print(f"\n模式: {pattern.name}")
        print(f"  描述: {pattern.description}")
        print(f"  频率: {pattern.frequency}")
        print(f"  洞察: {pattern.insights}")
    
    # 生成洞察
    print("\n=== 洞察报告 ===")
    insights = journal.generate_insights()
    for key, value in insights.items():
        print(f"{key}: {value}")
    
    # 获取改进建议
    print("\n=== 改进建议 ===")
    suggestions = journal.get_improvement_suggestions()
    for suggestion in suggestions:
        print(f"\n来源: {suggestion['source']}")
        print(f"优先级: {suggestion['priority']}")
        for s in suggestion['suggestions']:
            print(f"  - {s}")
    
    print("\n=== 反思提示 ===")
    for level, prompts in ReflectionPrompts.get_all_prompts().items():
        print(f"\n{level.name}级别提示:")
        for prompt in prompts:
            print(f"  • {prompt}")
