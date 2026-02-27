#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
learning_loop.py - 学习循环系统

功能：
1. 持续学习循环管理
2. 经验知识库维护
3. 改进策略生成与跟踪
4. 学习效果评估

作者：元认知研究项目
版本：1.0.0
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Set
from dataclasses import dataclass, field, asdict
from enum import Enum, auto
from collections import defaultdict
import threading
import time


class LearningStage(Enum):
    """学习阶段"""
    OBSERVE = "observe"       # 观察
    REFLECT = "reflect"       # 反思
    CONCEPTUALIZE = "conceptualize"  # 概念化
    EXPERIMENT = "experiment" # 实验
    INTEGRATE = "integrate"   # 整合


class ImprovementStatus(Enum):
    """改进状态"""
    PLANNED = "planned"       # 计划中
    IN_PROGRESS = "in_progress"  # 进行中
    COMPLETED = "completed"   # 已完成
    ABANDONED = "abandoned"   # 已放弃
    DEFERRED = "deferred"     # 推迟


@dataclass
class LearningGoal:
    """学习目标"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    created_at: datetime = field(default_factory=datetime.now)
    target_date: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ImprovementStatus = ImprovementStatus.PLANNED
    progress: float = 0.0  # 0-100
    related_reflections: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['target_date'] = self.target_date.isoformat() if self.target_date else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data


@dataclass
class ImprovementAction:
    """改进行动"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    description: str = ""
    goal_id: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    status: ImprovementStatus = ImprovementStatus.PLANNED
    priority: int = 3  # 1-5, 1最高
    effort_estimate: Optional[int] = None  # 小时
    actual_effort: Optional[int] = None
    outcome: str = ""
    effectiveness_rating: Optional[float] = None  # 1-10
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['status'] = self.status.value
        data['created_at'] = self.created_at.isoformat()
        data['started_at'] = self.started_at.isoformat() if self.started_at else None
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data


@dataclass
class KnowledgeItem:
    """知识条目"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    content: str = ""
    category: str = "general"
    source_reflections: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)
    confidence: float = 0.5  # 0-1
    usage_count: int = 0
    last_used: Optional[datetime] = None
    related_knowledge: List[str] = field(default_factory=list)
    tags: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['created_at'] = self.created_at.isoformat()
        data['updated_at'] = self.updated_at.isoformat()
        data['last_used'] = self.last_used.isoformat() if self.last_used else None
        return data


@dataclass
class LearningCycle:
    """学习周期记录"""
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    started_at: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    stages_completed: List[LearningStage] = field(default_factory=list)
    current_stage: LearningStage = LearningStage.OBSERVE
    observations: List[str] = field(default_factory=list)
    reflections: List[str] = field(default_factory=list)
    concepts: List[str] = field(default_factory=list)
    experiments: List[str] = field(default_factory=list)
    outcomes: Dict[str, Any] = field(default_factory=dict)
    is_active: bool = True
    
    def to_dict(self) -> Dict[str, Any]:
        data = asdict(self)
        data['current_stage'] = self.current_stage.value
        data['stages_completed'] = [s.value for s in self.stages_completed]
        data['started_at'] = self.started_at.isoformat()
        data['completed_at'] = self.completed_at.isoformat() if self.completed_at else None
        return data


class LearningLoop:
    """学习循环系统"""
    
    def __init__(self, storage_path: Optional[str] = None):
        self.storage_path = storage_path or "learning_loop.json"
        self.goals: Dict[str, LearningGoal] = {}
        self.actions: Dict[str, ImprovementAction] = {}
        self.knowledge_base: Dict[str, KnowledgeItem] = {}
        self.cycles: Dict[str, LearningCycle] = {}
        self.active_cycle: Optional[str] = None
        
        self._lock = threading.RLock()
        self._observers: List[Callable] = []
        self._load()
    
    def start_learning_cycle(self, context: str = "") -> LearningCycle:
        """开始新的学习周期"""
        with self._lock:
            # 完成之前的周期
            if self.active_cycle:
                self.complete_cycle(self.active_cycle)
            
            cycle = LearningCycle(
                observations=[context] if context else []
            )
            self.cycles[cycle.id] = cycle
            self.active_cycle = cycle.id
            self._save()
            return cycle
    
    def advance_stage(self, cycle_id: Optional[str] = None) -> Optional[LearningStage]:
        """推进到下一个学习阶段"""
        with self._lock:
            cycle_id = cycle_id or self.active_cycle
            if not cycle_id or cycle_id not in self.cycles:
                return None
            
            cycle = self.cycles[cycle_id]
            stages = list(LearningStage)
            
            # 记录当前阶段为已完成
            if cycle.current_stage not in cycle.stages_completed:
                cycle.stages_completed.append(cycle.current_stage)
            
            # 推进到下一阶段
            current_idx = stages.index(cycle.current_stage)
            if current_idx < len(stages) - 1:
                cycle.current_stage = stages[current_idx + 1]
            else:
                # 所有阶段完成
                cycle.is_active = False
                cycle.completed_at = datetime.now()
            
            self._save()
            self._notify_observers("stage_advanced", cycle)
            return cycle.current_stage
    
    def add_observation(self, observation: str, cycle_id: Optional[str] = None) -> bool:
        """添加观察"""
        with self._lock:
            cycle_id = cycle_id or self.active_cycle
            if not cycle_id or cycle_id not in self.cycles:
                return False
            
            cycle = self.cycles[cycle_id]
            cycle.observations.append(observation)
            self._save()
            return True
    
    def add_reflection(self, reflection: str, cycle_id: Optional[str] = None) -> bool:
        """添加反思"""
        with self._lock:
            cycle_id = cycle_id or self.active_cycle
            if not cycle_id or cycle_id not in self.cycles:
                return False
            
            cycle = self.cycles[cycle_id]
            cycle.reflections.append(reflection)
            
            # 自动推进到概念化阶段
            if cycle.current_stage == LearningStage.OBSERVE:
                self.advance_stage(cycle_id)
            
            self._save()
            return True
    
    def add_concept(self, concept: str, cycle_id: Optional[str] = None) -> bool:
        """添加概念"""
        with self._lock:
            cycle_id = cycle_id or self.active_cycle
            if not cycle_id or cycle_id not in self.cycles:
                return False
            
            cycle = self.cycles[cycle_id]
            cycle.concepts.append(concept)
            
            # 自动推进到实验阶段
            if cycle.current_stage == LearningStage.REFLECT:
                self.advance_stage(cycle_id)
            
            self._save()
            return True
    
    def add_experiment(self, experiment: str, cycle_id: Optional[str] = None) -> bool:
        """添加实验"""
        with self._lock:
            cycle_id = cycle_id or self.active_cycle
            if not cycle_id or cycle_id not in self.cycles:
                return False
            
            cycle = self.cycles[cycle_id]
            cycle.experiments.append(experiment)
            
            # 自动推进到整合阶段
            if cycle.current_stage == LearningStage.CONCEPTUALIZE:
                self.advance_stage(cycle_id)
            
            self._save()
            return True
    
    def complete_cycle(self, cycle_id: Optional[str] = None) -> Optional[LearningCycle]:
        """完成学习周期"""
        with self._lock:
            cycle_id = cycle_id or self.active_cycle
            if not cycle_id or cycle_id not in self.cycles:
                return None
            
            cycle = self.cycles[cycle_id]
            cycle.is_active = False
            cycle.completed_at = datetime.now()
            
            # 确保所有阶段都标记为完成
            for stage in LearningStage:
                if stage not in cycle.stages_completed:
                    cycle.stages_completed.append(stage)
            
            if self.active_cycle == cycle_id:
                self.active_cycle = None
            
            self._save()
            self._notify_observers("cycle_completed", cycle)
            return cycle
    
    def create_goal(
        self,
        description: str,
        target_date: Optional[datetime] = None,
        related_reflections: Optional[List[str]] = None
    ) -> LearningGoal:
        """创建学习目标"""
        with self._lock:
            goal = LearningGoal(
                description=description,
                target_date=target_date,
                related_reflections=related_reflections or []
            )
            self.goals[goal.id] = goal
            self._save()
            self._notify_observers("goal_created", goal)
            return goal
    
    def update_goal_progress(self, goal_id: str, progress: float) -> bool:
        """更新目标进度"""
        with self._lock:
            if goal_id not in self.goals:
                return False
            
            goal = self.goals[goal_id]
            goal.progress = max(0.0, min(100.0, progress))
            
            if goal.progress >= 100.0:
                goal.status = ImprovementStatus.COMPLETED
                goal.completed_at = datetime.now()
                self._notify_observers("goal_completed", goal)
            
            self._save()
            return True
    
    def create_action(
        self,
        description: str,
        goal_id: Optional[str] = None,
        priority: int = 3,
        effort_estimate: Optional[int] = None,
        tags: Optional[List[str]] = None
    ) -> ImprovementAction:
        """创建改进行动"""
        with self._lock:
            action = ImprovementAction(
                description=description,
                goal_id=goal_id,
                priority=priority,
                effort_estimate=effort_estimate,
                tags=tags or []
            )
            self.actions[action.id] = action
            
            # 关联到目标
            if goal_id and goal_id in self.goals:
                goal = self.goals[goal_id]
                if "action_ids" not in goal.metrics:
                    goal.metrics["action_ids"] = []
                goal.metrics["action_ids"].append(action.id)
            
            self._save()
            return action
    
    def start_action(self, action_id: str) -> bool:
        """开始行动"""
        with self._lock:
            if action_id not in self.actions:
                return False
            
            action = self.actions[action_id]
            action.status = ImprovementStatus.IN_PROGRESS
            action.started_at = datetime.now()
            self._save()
            return True
    
    def complete_action(
        self,
        action_id: str,
        outcome: str = "",
        effectiveness_rating: Optional[float] = None
    ) -> bool:
        """完成行动"""
        with self._lock:
            if action_id not in self.actions:
                return False
            
            action = self.actions[action_id]
            action.status = ImprovementStatus.COMPLETED
            action.completed_at = datetime.now()
            action.outcome = outcome
            action.effectiveness_rating = effectiveness_rating
            
            # 计算实际投入
            if action.started_at:
                duration = datetime.now() - action.started_at
                action.actual_effort = int(duration.total_seconds() / 3600)
            
            self._save()
            self._notify_observers("action_completed", action)
            return True
    
    def add_knowledge(
        self,
        content: str,
        category: str = "general",
        source_reflections: Optional[List[str]] = None,
        confidence: float = 0.5,
        tags: Optional[List[str]] = None
    ) -> KnowledgeItem:
        """添加知识"""
        with self._lock:
            # 检查是否已存在相似知识
            existing = self._find_similar_knowledge(content)
            if existing:
                # 更新现有知识
                existing.confidence = max(existing.confidence, confidence)
                existing.updated_at = datetime.now()
                if source_reflections:
                    existing.source_reflections.extend(source_reflections)
                self._save()
                return existing
            
            knowledge = KnowledgeItem(
                content=content,
                category=category,
                source_reflections=source_reflections or [],
                confidence=confidence,
                tags=tags or []
            )
            self.knowledge_base[knowledge.id] = knowledge
            self._save()
            return knowledge
    
    def _find_similar_knowledge(self, content: str) -> Optional[KnowledgeItem]:
        """查找相似知识（简化实现）"""
        content_lower = content.lower()
        for item in self.knowledge_base.values():
            # 简单的相似度检查
            if content_lower in item.content.lower() or item.content.lower() in content_lower:
                return item
        return None
    
    def search_knowledge(
        self,
        query: str,
        category: Optional[str] = None,
        min_confidence: float = 0.0
    ) -> List[KnowledgeItem]:
        """搜索知识"""
        results = []
        query_lower = query.lower()
        
        for item in self.knowledge_base.values():
            if category and item.category != category:
                continue
            if item.confidence < min_confidence:
                continue
            
            if (query_lower in item.content.lower() or
                any(query_lower in tag.lower() for tag in item.tags)):
                results.append(item)
        
        # 按置信度和使用次数排序
        results.sort(key=lambda x: (x.confidence, x.usage_count), reverse=True)
        return results
    
    def use_knowledge(self, knowledge_id: str) -> Optional[KnowledgeItem]:
        """使用知识（更新使用统计）"""
        with self._lock:
            if knowledge_id not in self.knowledge_base:
                return None
            
            knowledge = self.knowledge_base[knowledge_id]
            knowledge.usage_count += 1
            knowledge.last_used = datetime.now()
            self._save()
            return knowledge
    
    def get_learning_stats(self) -> Dict[str, Any]:
        """获取学习统计"""
        total_goals = len(self.goals)
        completed_goals = sum(1 for g in self.goals.values() if g.status == ImprovementStatus.COMPLETED)
        
        total_actions = len(self.actions)
        completed_actions = sum(1 for a in self.actions.values() if a.status == ImprovementStatus.COMPLETED)
        
        total_cycles = len(self.cycles)
        completed_cycles = sum(1 for c in self.cycles.values() if not c.is_active)
        
        # 计算平均行动效果
        ratings = [a.effectiveness_rating for a in self.actions.values() if a.effectiveness_rating]
        avg_effectiveness = sum(ratings) / len(ratings) if ratings else 0
        
        # 知识库统计
        knowledge_by_category = defaultdict(int)
        for item in self.knowledge_base.values():
            knowledge_by_category[item.category] += 1
        
        return {
            "goals": {
                "total": total_goals,
                "completed": completed_goals,
                "completion_rate": completed_goals / total_goals if total_goals > 0 else 0
            },
            "actions": {
                "total": total_actions,
                "completed": completed_actions,
                "completion_rate": completed_actions / total_actions if total_actions > 0 else 0,
                "average_effectiveness": avg_effectiveness
            },
            "cycles": {
                "total": total_cycles,
                "completed": completed_cycles,
                "active": self.active_cycle
            },
            "knowledge_base": {
                "total_items": len(self.knowledge_base),
                "by_category": dict(knowledge_by_category)
            }
        }
    
    def generate_improvement_plan(self) -> Dict[str, Any]:
        """生成改进计划"""
        plan = {
            "created_at": datetime.now().isoformat(),
            "goals": [],
            "priority_actions": [],
            "recommendations": []
        }
        
        # 未完成的目标
        incomplete_goals = [
            g for g in self.goals.values()
            if g.status != ImprovementStatus.COMPLETED
        ]
        
        for goal in sorted(incomplete_goals, key=lambda x: x.created_at):
            plan["goals"].append({
                "id": goal.id,
                "description": goal.description,
                "progress": goal.progress,
                "status": goal.status.value
            })
        
        # 高优先级未完成行动
        pending_actions = [
            a for a in self.actions.values()
            if a.status in [ImprovementStatus.PLANNED, ImprovementStatus.IN_PROGRESS]
        ]
        
        for action in sorted(pending_actions, key=lambda x: x.priority):
            plan["priority_actions"].append({
                "id": action.id,
                "description": action.description,
                "priority": action.priority,
                "status": action.status.value
            })
        
        # 生成建议
        if not incomplete_goals:
            plan["recommendations"].append("建议设定新的学习目标以持续改进")
        
        if len(pending_actions) > 5:
            plan["recommendations"].append(f"待办行动较多({len(pending_actions)}个)，建议优先处理高优先级项")
        
        low_confidence_knowledge = [
            k for k in self.knowledge_base.values() if k.confidence < 0.5
        ]
        if low_confidence_knowledge:
            plan["recommendations"].append(
                f"有{len(low_confidence_knowledge)}条知识置信度较低，建议验证和巩固"
            )
        
        return plan
    
    def add_observer(self, callback: Callable):
        """添加观察者"""
        self._observers.append(callback)
    
    def _notify_observers(self, event: str, data: Any):
        """通知观察者"""
        for observer in self._observers:
            try:
                observer(event, data)
            except Exception as e:
                print(f"Observer error: {e}")
    
    def _save(self):
        """保存到文件"""
        try:
            data = {
                "goals": {k: v.to_dict() for k, v in self.goals.items()},
                "actions": {k: v.to_dict() for k, v in self.actions.items()},
                "knowledge_base": {k: v.to_dict() for k, v in self.knowledge_base.items()},
                "cycles": {k: v.to_dict() for k, v in self.cycles.items()},
                "active_cycle": self.active_cycle
            }
            with open(self.storage_path, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception as e:
            print(f"保存学习循环数据失败: {e}")
    
    def _load(self):
        """从文件加载"""
        try:
            with open(self.storage_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # 加载目标
            for goal_id, goal_data in data.get("goals", {}).items():
                goal_data['status'] = ImprovementStatus(goal_data['status'])
                goal_data['created_at'] = datetime.fromisoformat(goal_data['created_at'])
                if goal_data['target_date']:
                    goal_data['target_date'] = datetime.fromisoformat(goal_data['target_date'])
                if goal_data['completed_at']:
                    goal_data['completed_at'] = datetime.fromisoformat(goal_data['completed_at'])
                self.goals[goal_id] = LearningGoal(**goal_data)
            
            # 加载行动
            for action_id, action_data in data.get("actions", {}).items():
                action_data['status'] = ImprovementStatus(action_data['status'])
                action_data['created_at'] = datetime.fromisoformat(action_data['created_at'])
                if action_data['started_at']:
                    action_data['started_at'] = datetime.fromisoformat(action_data['started_at'])
                if action_data['completed_at']:
                    action_data['completed_at'] = datetime.fromisoformat(action_data['completed_at'])
                self.actions[action_id] = ImprovementAction(**action_data)
            
            # 加载知识
            for knowledge_id, knowledge_data in data.get("knowledge_base", {}).items():
                knowledge_data['created_at'] = datetime.fromisoformat(knowledge_data['created_at'])
                knowledge_data['updated_at'] = datetime.fromisoformat(knowledge_data['updated_at'])
                if knowledge_data['last_used']:
                    knowledge_data['last_used'] = datetime.fromisoformat(knowledge_data['last_used'])
                self.knowledge_base[knowledge_id] = KnowledgeItem(**knowledge_data)
            
            # 加载周期
            for cycle_id, cycle_data in data.get("cycles", {}).items():
                cycle_data['current_stage'] = LearningStage(cycle_data['current_stage'])
                cycle_data['stages_completed'] = [LearningStage(s) for s in cycle_data['stages_completed']]
                cycle_data['started_at'] = datetime.fromisoformat(cycle_data['started_at'])
                if cycle_data['completed_at']:
                    cycle_data['completed_at'] = datetime.fromisoformat(cycle_data['completed_at'])
                self.cycles[cycle_id] = LearningCycle(**cycle_data)
            
            self.active_cycle = data.get("active_cycle")
            
        except FileNotFoundError:
            pass
        except Exception as e:
            print(f"加载学习循环数据失败: {e}")


class ContinuousImprovement:
    """持续改进管理器"""
    
    def __init__(self, learning_loop: LearningLoop):
        self.loop = learning_loop
        self.review_interval_days = 7
        self.last_review: Optional[datetime] = None
    
    def should_review(self) -> bool:
        """检查是否应该进行回顾"""
        if not self.last_review:
            return True
        days_since = (datetime.now() - self.last_review).days
        return days_since >= self.review_interval_days
    
    def conduct_review(self) -> Dict[str, Any]:
        """进行定期回顾"""
        self.last_review = datetime.now()
        
        stats = self.loop.get_learning_stats()
        plan = self.loop.generate_improvement_plan()
        
        review = {
            "review_date": self.last_review.isoformat(),
            "statistics": stats,
            "improvement_plan": plan,
            "insights": self._generate_insights(stats),
            "next_review": (self.last_review + timedelta(days=self.review_interval_days)).isoformat()
        }
        
        return review
    
    def _generate_insights(self, stats: Dict[str, Any]) -> List[str]:
        """生成洞察"""
        insights = []
        
        # 目标完成情况
        goal_rate = stats["goals"]["completion_rate"]
        if goal_rate >= 0.8:
            insights.append("目标完成率很高，保持这个节奏！")
        elif goal_rate < 0.5:
            insights.append("目标完成率较低，建议检视目标设定是否合理")
        
        # 行动效果
        avg_effectiveness = stats["actions"]["average_effectiveness"]
        if avg_effectiveness >= 8:
            insights.append("改进行动效果显著，方法有效")
        elif avg_effectiveness < 5:
            insights.append("改进行动效果一般，可能需要调整策略")
        
        # 知识积累
        kb_size = stats["knowledge_base"]["total_items"]
        if kb_size > 50:
            insights.append(f"知识库已积累{kb_size}条知识，建议定期整理和复习")
        
        return insights


# 便捷函数
def create_learning_loop(storage_path: Optional[str] = None) -> LearningLoop:
    """创建学习循环实例"""
    return LearningLoop(storage_path)


def create_continuous_improvement(
    loop: Optional[LearningLoop] = None
) -> ContinuousImprovement:
    """创建持续改进管理器"""
    if loop is None:
        loop = create_learning_loop()
    return ContinuousImprovement(loop)


if __name__ == "__main__":
    # 示例用法
    print("=== 学习循环系统 ===\n")
    
    # 创建学习循环
    loop = create_learning_loop("demo_learning.json")
    
    # 开始新的学习周期
    print("1. 开始学习周期")
    cycle = loop.start_learning_cycle("优化代码审查流程")
    print(f"   周期ID: {cycle.id}")
    print(f"   当前阶段: {cycle.current_stage.value}")
    
    # 添加观察
    print("\n2. 添加观察")
    loop.add_observation("代码审查经常延迟，影响发布进度")
    loop.add_observation("审查意见质量参差不齐")
    
    # 添加反思
    print("3. 添加反思")
    loop.add_reflection("缺乏明确的审查标准和检查清单")
    loop.add_reflection("审查者时间分配不合理")
    print(f"   当前阶段: {cycle.current_stage.value}")
    
    # 添加概念
    print("4. 添加概念")
    loop.add_concept("需要建立标准化的代码审查流程")
    loop.add_concept("引入审查轮值制度平衡负载")
    print(f"   当前阶段: {cycle.current_stage.value}")
    
    # 添加实验
    print("5. 添加实验")
    loop.add_experiment("制定代码审查检查清单")
    loop.add_experiment("试行审查轮值制度")
    print(f"   当前阶段: {cycle.current_stage.value}")
    
    # 完成周期
    print("6. 完成学习周期")
    completed = loop.complete_cycle()
    print(f"   完成阶段数: {len(completed.stages_completed)}")
    
    # 创建学习目标
    print("\n7. 创建学习目标")
    goal = loop.create_goal(
        description="将代码审查平均时间缩短50%",
        target_date=datetime.now() + timedelta(days=30)
    )
    print(f"   目标ID: {goal.id}")
    print(f"   描述: {goal.description}")
    
    # 创建改进行动
    print("\n8. 创建改进行动")
    action = loop.create_action(
        description="制定并实施代码审查检查清单",
        goal_id=goal.id,
        priority=1,
        effort_estimate=8,
        tags=["process", "code-review"]
    )
    print(f"   行动ID: {action.id}")
    print(f"   优先级: {action.priority}")
    
    # 开始并完成任务
    loop.start_action(action.id)
    loop.complete_action(
        action_id=action.id,
        outcome="检查清单已制定并开始使用",
        effectiveness_rating=8.5
    )
    
    # 更新目标进度
    loop.update_goal_progress(goal.id, 50.0)
    
    # 添加知识
    print("\n9. 添加知识")
    knowledge = loop.add_knowledge(
        content="代码审查检查清单应包括：功能正确性、代码风格、测试覆盖、性能影响",
        category="best_practice",
        confidence=0.9,
        tags=["code-review", "checklist"]
    )
    print(f"   知识ID: {knowledge.id}")
    
    # 获取学习统计
    print("\n=== 学习统计 ===")
    stats = loop.get_learning_stats()
    for category, data in stats.items():
        print(f"\n{category}:")
        for key, value in data.items():
            print(f"  {key}: {value}")
    
    # 生成改进计划
    print("\n=== 改进计划 ===")
    plan = loop.generate_improvement_plan()
    print(f"目标数: {len(plan['goals'])}")
    print(f"优先行动数: {len(plan['priority_actions'])}")
    print("建议:")
    for rec in plan['recommendations']:
        print(f"  • {rec}")
    
    # 持续改进回顾
    print("\n=== 持续改进回顾 ===")
    ci = create_continuous_improvement(loop)
    review = ci.conduct_review()
    print(f"回顾日期: {review['review_date']}")
    print("洞察:")
    for insight in review['insights']:
        print(f"  • {insight}")
    
    print("\n=== 演示完成 ===")
