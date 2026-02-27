"""
strategy_selector.py - 策略选择器
元认知系统第二部分：策略选择与优化

功能：
1. 策略库管理 - 维护可用策略集合
2. 策略选择算法 - 根据任务特征选择最优策略
3. 策略匹配评分 - 评估策略与任务的匹配度
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
import json
import time
from collections import defaultdict


class StrategyType(Enum):
    """策略类型枚举"""
    RESEARCH = auto()      # 研究型策略
    ANALYSIS = auto()      # 分析型策略
    CREATIVE = auto()      # 创意型策略
    EXECUTION = auto()     # 执行型策略
    COLLABORATION = auto() # 协作型策略
    OPTIMIZATION = auto()  # 优化型策略


class TaskComplexity(Enum):
    """任务复杂度等级"""
    SIMPLE = 1
    MODERATE = 2
    COMPLEX = 3
    VERY_COMPLEX = 4


@dataclass
class TaskProfile:
    """任务特征画像"""
    task_id: str
    task_type: str
    complexity: TaskComplexity
    required_skills: List[str]
    time_constraints: Optional[int] = None  # 分钟
    quality_requirements: float = 0.5  # 0-1
    collaboration_needed: bool = False
    domain: str = "general"
    
    def to_vector(self) -> Dict[str, float]:
        """转换为特征向量"""
        return {
            "complexity": self.complexity.value / 4.0,
            "skill_count": len(self.required_skills) / 10.0,
            "time_pressure": 1.0 - (self.time_constraints / 120.0) if self.time_constraints else 0.0,
            "quality_focus": self.quality_requirements,
            "collaboration": 1.0 if self.collaboration_needed else 0.0,
        }


@dataclass
class Strategy:
    """策略定义"""
    strategy_id: str
    name: str
    strategy_type: StrategyType
    description: str
    applicable_domains: List[str]
    required_resources: List[str]
    estimated_time_range: Tuple[int, int]  # (min, max) 分钟
    success_rate: float = 0.5
    avg_quality_score: float = 0.5
    usage_count: int = 0
    last_used: Optional[float] = None
    
    # 策略特征向量
    characteristics: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.characteristics:
            self.characteristics = self._default_characteristics()
    
    def _default_characteristics(self) -> Dict[str, float]:
        """生成默认特征向量"""
        type_weights = {
            StrategyType.RESEARCH: {"depth": 0.9, "breadth": 0.7, "speed": 0.3},
            StrategyType.ANALYSIS: {"depth": 0.8, "breadth": 0.6, "speed": 0.5},
            StrategyType.CREATIVE: {"depth": 0.6, "breadth": 0.9, "speed": 0.4},
            StrategyType.EXECUTION: {"depth": 0.5, "breadth": 0.4, "speed": 0.9},
            StrategyType.COLLABORATION: {"depth": 0.6, "breadth": 0.8, "speed": 0.5},
            StrategyType.OPTIMIZATION: {"depth": 0.8, "breadth": 0.5, "speed": 0.7},
        }
        return type_weights.get(self.strategy_type, {"depth": 0.5, "breadth": 0.5, "speed": 0.5})


@dataclass
class SelectionResult:
    """策略选择结果"""
    selected_strategy: Strategy
    confidence: float
    reasoning: str
    alternatives: List[Tuple[Strategy, float]]  # (策略, 得分)
    estimated_outcome: Dict[str, float]


class StrategyLibrary:
    """策略库管理器"""
    
    def __init__(self):
        self._strategies: Dict[str, Strategy] = {}
        self._domain_index: Dict[str, List[str]] = defaultdict(list)
        self._type_index: Dict[StrategyType, List[str]] = defaultdict(list)
        self._version = "1.0.0"
    
    def register(self, strategy: Strategy) -> bool:
        """注册新策略"""
        if strategy.strategy_id in self._strategies:
            return False
        
        self._strategies[strategy.strategy_id] = strategy
        
        # 更新索引
        for domain in strategy.applicable_domains:
            self._domain_index[domain].append(strategy.strategy_id)
        
        self._type_index[strategy.strategy_type].append(strategy.strategy_id)
        return True
    
    def unregister(self, strategy_id: str) -> bool:
        """注销策略"""
        if strategy_id not in self._strategies:
            return False
        
        strategy = self._strategies[strategy_id]
        
        # 更新索引
        for domain in strategy.applicable_domains:
            if strategy_id in self._domain_index[domain]:
                self._domain_index[domain].remove(strategy_id)
        
        if strategy_id in self._type_index[strategy.strategy_type]:
            self._type_index[strategy.strategy_type].remove(strategy_id)
        
        del self._strategies[strategy_id]
        return True
    
    def get(self, strategy_id: str) -> Optional[Strategy]:
        """获取策略"""
        return self._strategies.get(strategy_id)
    
    def list_all(self) -> List[Strategy]:
        """列出所有策略"""
        return list(self._strategies.values())
    
    def find_by_domain(self, domain: str) -> List[Strategy]:
        """按领域查找策略"""
        ids = self._domain_index.get(domain, [])
        return [self._strategies[sid] for sid in ids if sid in self._strategies]
    
    def find_by_type(self, strategy_type: StrategyType) -> List[Strategy]:
        """按类型查找策略"""
        ids = self._type_index.get(strategy_type, [])
        return [self._strategies[sid] for sid in ids if sid in self._strategies]
    
    def update_stats(self, strategy_id: str, success: bool, quality: float):
        """更新策略统计信息"""
        if strategy_id not in self._strategies:
            return
        
        strategy = self._strategies[strategy_id]
        strategy.usage_count += 1
        strategy.last_used = time.time()
        
        # 更新成功率 (指数移动平均)
        alpha = 0.1
        success_val = 1.0 if success else 0.0
        strategy.success_rate = (1 - alpha) * strategy.success_rate + alpha * success_val
        
        # 更新质量分数
        strategy.avg_quality_score = (1 - alpha) * strategy.avg_quality_score + alpha * quality
    
    def get_library_stats(self) -> Dict[str, Any]:
        """获取策略库统计"""
        total = len(self._strategies)
        type_distribution = {
            t.name: len(ids) for t, ids in self._type_index.items()
        }
        domain_distribution = {
            d: len(ids) for d, ids in self._domain_index.items()
        }
        
        avg_success = 0.0
        avg_quality = 0.0
        if total > 0:
            avg_success = sum(s.success_rate for s in self._strategies.values()) / total
            avg_quality = sum(s.avg_quality_score for s in self._strategies.values()) / total
        
        return {
            "version": self._version,
            "total_strategies": total,
            "type_distribution": type_distribution,
            "domain_distribution": domain_distribution,
            "avg_success_rate": avg_success,
            "avg_quality_score": avg_quality,
        }
    
    def export_to_json(self) -> str:
        """导出策略库为JSON"""
        data = {
            "version": self._version,
            "strategies": [
                {
                    "strategy_id": s.strategy_id,
                    "name": s.name,
                    "strategy_type": s.strategy_type.name,
                    "description": s.description,
                    "applicable_domains": s.applicable_domains,
                    "required_resources": s.required_resources,
                    "estimated_time_range": s.estimated_time_range,
                    "success_rate": s.success_rate,
                    "avg_quality_score": s.avg_quality_score,
                    "usage_count": s.usage_count,
                    "characteristics": s.characteristics,
                }
                for s in self._strategies.values()
            ]
        }
        return json.dumps(data, indent=2)


class StrategySelector:
    """策略选择器"""
    
    def __init__(self, strategy_library: StrategyLibrary):
        self.library = strategy_library
        self._selection_history: List[Dict[str, Any]] = []
        self._weights = {
            "domain_match": 0.25,
            "complexity_match": 0.20,
            "time_match": 0.20,
            "skill_match": 0.15,
            "historical_performance": 0.15,
            "recency": 0.05,
        }
    
    def select(
        self,
        task_profile: TaskProfile,
        top_k: int = 3,
        constraints: Optional[Dict[str, Any]] = None
    ) -> SelectionResult:
        """
        为任务选择最优策略
        
        Args:
            task_profile: 任务特征画像
            top_k: 返回前k个候选策略
            constraints: 可选约束条件
        
        Returns:
            SelectionResult: 选择结果
        """
        constraints = constraints or {}
        
        # 1. 候选策略过滤
        candidates = self._filter_candidates(task_profile, constraints)
        
        if not candidates:
            # 返回默认策略
            default_strategy = Strategy(
                strategy_id="default",
                name="通用执行策略",
                strategy_type=StrategyType.EXECUTION,
                description="适用于未知任务的默认策略",
                applicable_domains=["general"],
                required_resources=[],
                estimated_time_range=(30, 120),
            )
            return SelectionResult(
                selected_strategy=default_strategy,
                confidence=0.3,
                reasoning="无匹配策略，使用默认策略",
                alternatives=[],
                estimated_outcome={"success_prob": 0.5, "quality": 0.5, "time": 60}
            )
        
        # 2. 计算匹配分数
        scored_candidates = []
        for strategy in candidates:
            score, details = self._calculate_match_score(strategy, task_profile)
            scored_candidates.append((strategy, score, details))
        
        # 3. 排序并选择
        scored_candidates.sort(key=lambda x: x[1], reverse=True)
        
        best_strategy, best_score, best_details = scored_candidates[0]
        alternatives = [(s, score) for s, score, _ in scored_candidates[1:top_k]]
        
        # 4. 生成推理说明
        reasoning = self._generate_reasoning(best_strategy, best_details, task_profile)
        
        # 5. 预估结果
        estimated_outcome = self._estimate_outcome(best_strategy, task_profile)
        
        # 6. 记录选择历史
        self._selection_history.append({
            "timestamp": time.time(),
            "task_id": task_profile.task_id,
            "selected_strategy": best_strategy.strategy_id,
            "confidence": best_score,
            "alternatives": [s.strategy_id for s, _ in alternatives],
        })
        
        return SelectionResult(
            selected_strategy=best_strategy,
            confidence=best_score,
            reasoning=reasoning,
            alternatives=alternatives,
            estimated_outcome=estimated_outcome
        )
    
    def _filter_candidates(
        self,
        task_profile: TaskProfile,
        constraints: Dict[str, Any]
    ) -> List[Strategy]:
        """过滤候选策略"""
        candidates = []
        
        # 按领域查找
        domain_strategies = self.library.find_by_domain(task_profile.domain)
        if not domain_strategies and task_profile.domain != "general":
            domain_strategies = self.library.find_by_domain("general")
        
        for strategy in domain_strategies:
            # 检查约束条件
            if "max_time" in constraints:
                if strategy.estimated_time_range[1] > constraints["max_time"]:
                    continue
            
            if "required_type" in constraints:
                if strategy.strategy_type != constraints["required_type"]:
                    continue
            
            if "min_success_rate" in constraints:
                if strategy.success_rate < constraints["min_success_rate"]:
                    continue
            
            candidates.append(strategy)
        
        return candidates
    
    def _calculate_match_score(
        self,
        strategy: Strategy,
        task: TaskProfile
    ) -> Tuple[float, Dict[str, float]]:
        """计算策略与任务的匹配分数"""
        details = {}
        
        # 1. 领域匹配度
        if task.domain in strategy.applicable_domains:
            details["domain_match"] = 1.0
        elif "general" in strategy.applicable_domains:
            details["domain_match"] = 0.6
        else:
            details["domain_match"] = 0.2
        
        # 2. 复杂度匹配度
        task_complexity = task.complexity.value / 4.0
        strategy_depth = strategy.characteristics.get("depth", 0.5)
        complexity_diff = abs(task_complexity - strategy_depth)
        details["complexity_match"] = 1.0 - complexity_diff
        
        # 3. 时间匹配度
        if task.time_constraints:
            est_min, est_max = strategy.estimated_time_range
            if est_max <= task.time_constraints:
                details["time_match"] = 1.0
            elif est_min <= task.time_constraints:
                details["time_match"] = 0.7
            else:
                details["time_match"] = 0.3
        else:
            details["time_match"] = 0.8
        
        # 4. 技能匹配度
        if task.required_skills:
            matching_skills = set(task.required_skills) & set(strategy.required_resources)
            details["skill_match"] = len(matching_skills) / len(task.required_skills)
        else:
            details["skill_match"] = 0.5
        
        # 5. 历史表现
        details["historical_performance"] = (
            strategy.success_rate * 0.6 + strategy.avg_quality_score * 0.4
        )
        
        # 6. 时效性 (避免一直使用同一策略)
        if strategy.last_used:
            hours_since_last = (time.time() - strategy.last_used) / 3600
            details["recency"] = min(1.0, hours_since_last / 24.0)
        else:
            details["recency"] = 1.0  # 从未使用过，给予新鲜感
        
        # 计算加权总分
        total_score = sum(
            details[key] * self._weights[key] for key in self._weights.keys()
        )
        
        return total_score, details
    
    def _generate_reasoning(
        self,
        strategy: Strategy,
        details: Dict[str, float],
        task: TaskProfile
    ) -> str:
        """生成选择推理说明"""
        reasons = []
        
        if details["domain_match"] >= 0.8:
            reasons.append(f"策略'{strategy.name}'在'{task.domain}'领域有专门优化")
        
        if details["complexity_match"] >= 0.7:
            reasons.append("策略复杂度与任务需求高度匹配")
        
        if details["time_match"] >= 0.8:
            reasons.append("策略执行时间符合任务时间约束")
        
        if details["historical_performance"] >= 0.7:
            reasons.append(f"策略历史表现优秀(成功率{strategy.success_rate:.1%})")
        
        if not reasons:
            reasons.append(f"策略'{strategy.name}'综合评分最优")
        
        return "; ".join(reasons)
    
    def _estimate_outcome(
        self,
        strategy: Strategy,
        task: TaskProfile
    ) -> Dict[str, float]:
        """预估执行结果"""
        # 基于历史表现和任务特征预估
        base_success = strategy.success_rate
        base_quality = strategy.avg_quality_score
        
        # 调整因子
        complexity_factor = 1.0 - (task.complexity.value - 1) * 0.1
        time_pressure = 0.0
        if task.time_constraints:
            avg_time = sum(strategy.estimated_time_range) / 2
            if task.time_constraints < avg_time:
                time_pressure = (avg_time - task.time_constraints) / avg_time * 0.2
        
        adjusted_success = max(0.1, min(0.95, base_success * complexity_factor - time_pressure))
        adjusted_quality = max(0.1, min(0.95, base_quality * complexity_factor))
        
        return {
            "success_prob": round(adjusted_success, 3),
            "quality": round(adjusted_quality, 3),
            "time": sum(strategy.estimated_time_range) // 2,
        }
    
    def set_weights(self, weights: Dict[str, float]):
        """设置评分权重"""
        total = sum(weights.values())
        self._weights = {k: v / total for k, v in weights.items()}
    
    def get_selection_history(self, limit: int = 10) -> List[Dict[str, Any]]:
        """获取选择历史"""
        return self._selection_history[-limit:]


# 预定义策略库初始化
def create_default_strategies() -> List[Strategy]:
    """创建默认策略集合"""
    return [
        Strategy(
            strategy_id="research_deep_dive",
            name="深度研究策略",
            strategy_type=StrategyType.RESEARCH,
            description="系统性收集和分析信息，适用于需要全面了解的复杂问题",
            applicable_domains=["research", "analysis", "general"],
            required_resources=["web_search", "document_analysis"],
            estimated_time_range=(60, 180),
            characteristics={"depth": 0.95, "breadth": 0.8, "speed": 0.2},
        ),
        Strategy(
            strategy_id="quick_analysis",
            name="快速分析策略",
            strategy_type=StrategyType.ANALYSIS,
            description="快速识别关键信息，适用于时间敏感的分析任务",
            applicable_domains=["analysis", "business", "general"],
            required_resources=["data_processing"],
            estimated_time_range=(15, 45),
            characteristics={"depth": 0.5, "breadth": 0.6, "speed": 0.9},
        ),
        Strategy(
            strategy_id="creative_brainstorm",
            name="创意头脑风暴",
            strategy_type=StrategyType.CREATIVE,
            description="生成多样化创意方案，适用于创新性和设计类任务",
            applicable_domains=["creative", "design", "product"],
            required_resources=["ideation_tools"],
            estimated_time_range=(30, 90),
            characteristics={"depth": 0.5, "breadth": 0.95, "speed": 0.6},
        ),
        Strategy(
            strategy_id="agile_execution",
            name="敏捷执行策略",
            strategy_type=StrategyType.EXECUTION,
            description="快速迭代执行，适用于明确的开发或实施任务",
            applicable_domains=["development", "engineering", "general"],
            required_resources=["coding", "testing"],
            estimated_time_range=(20, 60),
            characteristics={"depth": 0.4, "breadth": 0.4, "speed": 0.95},
        ),
        Strategy(
            strategy_id="collaborative_workflow",
            name="协作工作流策略",
            strategy_type=StrategyType.COLLABORATION,
            description="多Agent协作完成任务，适用于需要多方参与的复杂项目",
            applicable_domains=["project", "team", "general"],
            required_resources=["multi_agent", "communication"],
            estimated_time_range=(45, 120),
            characteristics={"depth": 0.6, "breadth": 0.8, "speed": 0.5},
        ),
        Strategy(
            strategy_id="optimization_iterative",
            name="迭代优化策略",
            strategy_type=StrategyType.OPTIMIZATION,
            description="持续迭代改进解决方案，适用于优化和调优任务",
            applicable_domains=["optimization", "performance", "general"],
            required_resources=["profiling", "testing"],
            estimated_time_range=(40, 100),
            characteristics={"depth": 0.8, "breadth": 0.5, "speed": 0.7},
        ),
    ]


# 便捷函数
def create_default_library() -> StrategyLibrary:
    """创建并初始化默认策略库"""
    library = StrategyLibrary()
    for strategy in create_default_strategies():
        library.register(strategy)
    return library
