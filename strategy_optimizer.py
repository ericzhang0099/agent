"""
strategy_optimizer.py - 策略优化器
元认知系统第二部分：策略选择与优化

功能：
1. 策略效果评估 - 评估策略执行效果
2. 策略参数优化 - 优化策略参数配置
3. 策略组合优化 - 发现最优策略组合
4. 在线学习 - 根据反馈持续改进
"""

from typing import Dict, List, Optional, Callable, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import json
import time
import statistics
from collections import defaultdict, deque
import random

from strategy_selector import (
    StrategyLibrary, Strategy, StrategyType, TaskProfile, 
    TaskComplexity, SelectionResult
)


class EvaluationMetric(Enum):
    """评估指标"""
    SUCCESS = "success"           # 是否成功
    QUALITY = "quality"           # 质量分数
    EFFICIENCY = "efficiency"     # 效率分数
    TIME = "time"                 # 执行时间
    COST = "cost"                 # 资源成本
    SATISFACTION = "satisfaction" # 用户满意度


@dataclass
class ExecutionRecord:
    """策略执行记录"""
    record_id: str
    task_id: str
    strategy_id: str
    start_time: float
    end_time: Optional[float] = None
    metrics: Dict[str, float] = field(default_factory=dict)
    feedback: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    
    @property
    def duration(self) -> Optional[float]:
        """执行时长"""
        if self.end_time:
            return self.end_time - self.start_time
        return None
    
    def is_complete(self) -> bool:
        """是否已完成"""
        return self.end_time is not None


@dataclass
class StrategyPerformance:
    """策略性能统计"""
    strategy_id: str
    total_executions: int = 0
    successful_executions: int = 0
    failed_executions: int = 0
    avg_quality: float = 0.0
    avg_efficiency: float = 0.0
    avg_duration: float = 0.0
    recent_scores: deque = field(default_factory=lambda: deque(maxlen=10))
    trend: str = "stable"  # "improving", "declining", "stable"
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.total_executions == 0:
            return 0.0
        return self.successful_executions / self.total_executions
    
    @property
    def composite_score(self) -> float:
        """综合评分"""
        if self.total_executions == 0:
            return 0.5
        return (
            self.success_rate * 0.4 +
            self.avg_quality * 0.3 +
            self.avg_efficiency * 0.3
        )


@dataclass
class OptimizationSuggestion:
    """优化建议"""
    target_strategy: str
    suggestion_type: str  # "parameter", "workflow", "replacement"
    description: str
    expected_improvement: float
    confidence: float
    implementation: Optional[Dict[str, Any]] = None


class StrategyEvaluator:
    """策略效果评估器"""
    
    def __init__(self, library: StrategyLibrary):
        self.library = library
        self._execution_records: Dict[str, ExecutionRecord] = {}
        self._task_records: Dict[str, List[str]] = defaultdict(list)
        self._strategy_performance: Dict[str, StrategyPerformance] = {}
    
    def start_execution(
        self,
        task_id: str,
        strategy_id: str,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """开始记录策略执行"""
        record_id = f"{task_id}_{strategy_id}_{int(time.time())}"
        
        record = ExecutionRecord(
            record_id=record_id,
            task_id=task_id,
            strategy_id=strategy_id,
            start_time=time.time(),
            context=context or {}
        )
        
        self._execution_records[record_id] = record
        self._task_records[task_id].append(record_id)
        
        return record_id
    
    def complete_execution(
        self,
        record_id: str,
        success: bool,
        metrics: Optional[Dict[str, float]] = None,
        feedback: Optional[str] = None
    ) -> bool:
        """完成策略执行记录"""
        if record_id not in self._execution_records:
            return False
        
        record = self._execution_records[record_id]
        record.end_time = time.time()
        record.metrics = metrics or {}
        record.feedback = feedback
        
        # 更新策略性能统计
        self._update_performance(record, success)
        
        # 更新策略库统计
        quality = metrics.get("quality", 0.5) if metrics else 0.5
        self.library.update_stats(record.strategy_id, success, quality)
        
        return True
    
    def _update_performance(self, record: ExecutionRecord, success: bool):
        """更新策略性能统计"""
        strategy_id = record.strategy_id
        
        if strategy_id not in self._strategy_performance:
            self._strategy_performance[strategy_id] = StrategyPerformance(
                strategy_id=strategy_id
            )
        
        perf = self._strategy_performance[strategy_id]
        perf.total_executions += 1
        
        if success:
            perf.successful_executions += 1
        else:
            perf.failed_executions += 1
        
        # 更新平均值
        metrics = record.metrics
        alpha = 0.1  # 平滑因子
        
        if "quality" in metrics:
            perf.avg_quality = (1 - alpha) * perf.avg_quality + alpha * metrics["quality"]
        
        if "efficiency" in metrics:
            perf.avg_efficiency = (1 - alpha) * perf.avg_efficiency + alpha * metrics["efficiency"]
        
        if record.duration:
            if perf.avg_duration == 0:
                perf.avg_duration = record.duration
            else:
                perf.avg_duration = (1 - alpha) * perf.avg_duration + alpha * record.duration
        
        # 计算综合分数并更新趋势
        composite = perf.composite_score
        perf.recent_scores.append(composite)
        
        if len(perf.recent_scores) >= 5:
            recent_avg = sum(list(perf.recent_scores)[-5:]) / 5
            older_avg = sum(list(perf.recent_scores)[:5]) / 5 if len(perf.recent_scores) >= 10 else recent_avg
            
            if recent_avg > older_avg * 1.05:
                perf.trend = "improving"
            elif recent_avg < older_avg * 0.95:
                perf.trend = "declining"
            else:
                perf.trend = "stable"
    
    def evaluate_strategy(
        self,
        strategy_id: str,
        time_range: Optional[Tuple[float, float]] = None
    ) -> Optional[StrategyPerformance]:
        """评估特定策略的表现"""
        return self._strategy_performance.get(strategy_id)
    
    def compare_strategies(
        self,
        strategy_ids: List[str],
        metric: str = "composite_score"
    ) -> List[Tuple[str, float]]:
        """比较多个策略的表现"""
        results = []
        for sid in strategy_ids:
            perf = self._strategy_performance.get(sid)
            if perf:
                if metric == "composite_score":
                    score = perf.composite_score
                elif metric == "success_rate":
                    score = perf.success_rate
                elif metric == "avg_quality":
                    score = perf.avg_quality
                else:
                    score = perf.composite_score
                results.append((sid, score))
        
        results.sort(key=lambda x: x[1], reverse=True)
        return results
    
    def get_task_history(self, task_id: str) -> List[ExecutionRecord]:
        """获取任务的执行历史"""
        record_ids = self._task_records.get(task_id, [])
        return [self._execution_records[rid] for rid in record_ids if rid in self._execution_records]
    
    def generate_report(self, strategy_id: Optional[str] = None) -> Dict[str, Any]:
        """生成评估报告"""
        if strategy_id:
            perf = self._strategy_performance.get(strategy_id)
            if not perf:
                return {"error": f"Strategy {strategy_id} not found"}
            
            return {
                "strategy_id": strategy_id,
                "total_executions": perf.total_executions,
                "success_rate": perf.success_rate,
                "avg_quality": perf.avg_quality,
                "avg_efficiency": perf.avg_efficiency,
                "avg_duration": perf.avg_duration,
                "composite_score": perf.composite_score,
                "trend": perf.trend,
            }
        else:
            # 全局报告
            all_perfs = list(self._strategy_performance.values())
            if not all_perfs:
                return {"error": "No execution data available"}
            
            return {
                "total_strategies_evaluated": len(all_perfs),
                "total_executions": sum(p.total_executions for p in all_perfs),
                "overall_success_rate": sum(p.success_rate for p in all_perfs) / len(all_perfs),
                "avg_quality": sum(p.avg_quality for p in all_perfs) / len(all_perfs),
                "best_strategy": max(all_perfs, key=lambda p: p.composite_score).strategy_id,
                "worst_strategy": min(all_perfs, key=lambda p: p.composite_score).strategy_id,
            }


class StrategyOptimizer:
    """策略优化器"""
    
    def __init__(self, library: StrategyLibrary, evaluator: StrategyEvaluator):
        self.library = library
        self.evaluator = evaluator
        self._optimization_history: List[Dict[str, Any]] = []
        self._parameter_cache: Dict[str, Dict[str, Any]] = {}
    
    def suggest_improvements(
        self,
        strategy_id: Optional[str] = None,
        min_executions: int = 5
    ) -> List[OptimizationSuggestion]:
        """
        生成策略优化建议
        
        Args:
            strategy_id: 特定策略ID，None则分析所有策略
            min_executions: 生成建议所需的最小执行次数
        
        Returns:
            优化建议列表
        """
        suggestions = []
        
        if strategy_id:
            strategies = [strategy_id]
        else:
            strategies = list(self.evaluator._strategy_performance.keys())
        
        for sid in strategies:
            perf = self.evaluator._strategy_performance.get(sid)
            if not perf or perf.total_executions < min_executions:
                continue
            
            strategy = self.library.get(sid)
            if not strategy:
                continue
            
            # 分析失败原因
            if perf.success_rate < 0.6:
                suggestions.append(OptimizationSuggestion(
                    target_strategy=sid,
                    suggestion_type="workflow",
                    description=f"策略'{strategy.name}'成功率较低({perf.success_rate:.1%})，建议增加前置验证步骤",
                    expected_improvement=0.15,
                    confidence=0.7
                ))
            
            # 分析效率问题
            if perf.avg_efficiency < 0.5:
                suggestions.append(OptimizationSuggestion(
                    target_strategy=sid,
                    suggestion_type="parameter",
                    description=f"策略'{strategy.name}'效率偏低，建议优化资源分配或减少不必要的步骤",
                    expected_improvement=0.20,
                    confidence=0.6,
                    implementation={"reduce_steps": True, "parallelize": True}
                ))
            
            # 分析质量趋势
            if perf.trend == "declining":
                suggestions.append(OptimizationSuggestion(
                    target_strategy=sid,
                    suggestion_type="replacement",
                    description=f"策略'{strategy.name}'表现呈下降趋势，建议考虑替代策略或重新训练",
                    expected_improvement=0.25,
                    confidence=0.5
                ))
            
            # 时间优化建议
            if perf.avg_duration > sum(strategy.estimated_time_range) / 2 * 1.5:
                suggestions.append(OptimizationSuggestion(
                    target_strategy=sid,
                    suggestion_type="parameter",
                    description=f"策略'{strategy.name}'实际执行时间({perf.avg_duration:.0f}s)超出预期，建议调整时间预估或优化执行流程",
                    expected_improvement=0.10,
                    confidence=0.65
                ))
        
        # 按预期改进程度排序
        suggestions.sort(key=lambda x: x.expected_improvement, reverse=True)
        return suggestions
    
    def optimize_parameters(
        self,
        strategy_id: str,
        target_metric: str = "composite_score"
    ) -> Dict[str, Any]:
        """
        优化策略参数
        
        使用历史数据寻找最优参数配置
        """
        strategy = self.library.get(strategy_id)
        if not strategy:
            return {"error": f"Strategy {strategy_id} not found"}
        
        # 获取该策略的执行记录
        records = [
            r for r in self.evaluator._execution_records.values()
            if r.strategy_id == strategy_id and r.is_complete()
        ]
        
        if len(records) < 10:
            return {
                "strategy_id": strategy_id,
                "status": "insufficient_data",
                "message": f"需要至少10条执行记录，当前只有{len(records)}条",
            }
        
        # 分析不同上下文下的表现
        context_performance = defaultdict(lambda: {"scores": [], "params": []})
        
        for record in records:
            context_key = record.context.get("complexity", "unknown")
            score = record.metrics.get(target_metric, record.metrics.get("quality", 0.5))
            context_performance[context_key]["scores"].append(score)
        
        # 找出最佳配置
        best_config = None
        best_score = 0.0
        
        for context_key, data in context_performance.items():
            avg_score = sum(data["scores"]) / len(data["scores"])
            if avg_score > best_score:
                best_score = avg_score
                best_config = context_key
        
        # 生成优化建议
        optimization_result = {
            "strategy_id": strategy_id,
            "status": "success",
            "best_context": best_config,
            "best_score": best_score,
            "recommendations": [],
        }
        
        # 根据策略类型给出具体建议
        if strategy.strategy_type == StrategyType.RESEARCH:
            optimization_result["recommendations"].append(
                "调整信息收集深度与广度的平衡参数"
            )
        elif strategy.strategy_type == StrategyType.EXECUTION:
            optimization_result["recommendations"].append(
                "优化并行执行的任务数量阈值"
            )
        elif strategy.strategy_type == StrategyType.COLLABORATION:
            optimization_result["recommendations"].append(
                "调整Agent间通信频率和同步机制"
            )
        
        self._parameter_cache[strategy_id] = optimization_result
        return optimization_result
    
    def find_strategy_combinations(
        self,
        task_profile: TaskProfile,
        max_strategies: int = 3
    ) -> List[Dict[str, Any]]:
        """
        寻找最优策略组合
        
        对于复杂任务，可能需要多个策略组合使用
        """
        # 获取候选策略
        candidates = self.library.find_by_domain(task_profile.domain)
        if not candidates:
            candidates = self.library.list_all()
        
        # 过滤掉表现差的策略
        good_candidates = []
        for s in candidates:
            perf = self.evaluator._strategy_performance.get(s.strategy_id)
            if perf and perf.success_rate > 0.4:
                good_candidates.append(s)
        
        if not good_candidates:
            good_candidates = candidates
        
        combinations = []
        
        # 生成策略组合 (简化版 - 实际可使用更复杂的算法)
        for i, s1 in enumerate(good_candidates):
            # 单策略
            perf = self.evaluator._strategy_performance.get(s1.strategy_id)
            score = perf.composite_score if perf else 0.5
            combinations.append({
                "strategies": [s1.strategy_id],
                "estimated_score": score,
                "rationale": f"使用单一策略{s1.name}",
            })
            
            # 双策略组合
            if max_strategies >= 2:
                for s2 in good_candidates[i+1:]:
                    # 检查策略类型是否互补
                    complementary = self._are_complementary(s1, s2)
                    if complementary:
                        combined_score = min(0.95, score + 0.15)  # 组合加分
                        combinations.append({
                            "strategies": [s1.strategy_id, s2.strategy_id],
                            "estimated_score": combined_score,
                            "rationale": f"组合{s1.name}和{s2.name}，互补性强",
                            "execution_order": [s1.strategy_id, s2.strategy_id],
                        })
        
        # 排序并返回前N个
        combinations.sort(key=lambda x: x["estimated_score"], reverse=True)
        return combinations[:5]
    
    def _are_complementary(self, s1: Strategy, s2: Strategy) -> bool:
        """判断两个策略是否互补"""
        # 不同类型通常更互补
        if s1.strategy_type != s2.strategy_type:
            return True
        
        # 特征向量差异大表示互补
        char1 = s1.characteristics
        char2 = s2.characteristics
        
        diff = sum(abs(char1.get(k, 0.5) - char2.get(k, 0.5)) for k in ["depth", "breadth", "speed"])
        return diff > 0.5
    
    def adapt_strategy(
        self,
        strategy_id: str,
        execution_context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        根据执行上下文动态调整策略
        
        在线学习：根据当前执行情况调整策略参数
        """
        strategy = self.library.get(strategy_id)
        if not strategy:
            return {"error": f"Strategy {strategy_id} not found"}
        
        adaptations = {
            "strategy_id": strategy_id,
            "adaptations_made": [],
            "reasoning": [],
        }
        
        # 根据时间压力调整
        time_pressure = execution_context.get("time_pressure", 0.0)
        if time_pressure > 0.7:
            adaptations["adaptations_made"].append("reduce_scope")
            adaptations["reasoning"].append("时间压力大，缩小执行范围")
        
        # 根据复杂度调整
        complexity = execution_context.get("complexity", TaskComplexity.MODERATE)
        if complexity == TaskComplexity.VERY_COMPLEX:
            adaptations["adaptations_made"].append("increase_depth")
            adaptations["reasoning"].append("任务复杂度高，增加分析深度")
        
        # 根据历史反馈调整
        perf = self.evaluator._strategy_performance.get(strategy_id)
        if perf and perf.trend == "declining":
            adaptations["adaptations_made"].append("add_verification")
            adaptations["reasoning"].append("策略表现下降，增加验证步骤")
        
        # 记录适应历史
        self._optimization_history.append({
            "timestamp": time.time(),
            "strategy_id": strategy_id,
            "context": execution_context,
            "adaptations": adaptations["adaptations_made"],
        })
        
        return adaptations
    
    def get_optimization_report(self) -> Dict[str, Any]:
        """获取优化报告"""
        return {
            "total_optimizations": len(self._optimization_history),
            "recent_adaptations": self._optimization_history[-10:],
            "cached_parameters": list(self._parameter_cache.keys()),
            "improvement_suggestions_count": len(self.suggest_improvements()),
        }


class MetaLearningEngine:
    """元学习引擎 - 跨任务学习最优策略选择模式"""
    
    def __init__(self, evaluator: StrategyEvaluator):
        self.evaluator = evaluator
        self._task_patterns: Dict[str, Dict[str, Any]] = {}
        self._cross_task_insights: List[Dict[str, Any]] = []
    
    def learn_from_executions(self, min_samples: int = 20):
        """从执行记录中学习模式"""
        records = [
            r for r in self.evaluator._execution_records.values()
            if r.is_complete()
        ]
        
        if len(records) < min_samples:
            return {"status": "insufficient_data", "required": min_samples, "available": len(records)}
        
        # 按任务类型分组分析
        task_type_performance = defaultdict(lambda: {"strategies": defaultdict(list)})
        
        for record in records:
            task_type = record.context.get("task_type", "unknown")
            task_type_performance[task_type]["strategies"][record.strategy_id].append(
                record.metrics.get("quality", 0.5)
            )
        
        # 找出每种任务类型的最佳策略
        insights = []
        for task_type, data in task_type_performance.items():
            best_strategy = None
            best_score = 0.0
            
            for strategy_id, scores in data["strategies"].items():
                if len(scores) >= 3:  # 至少3个样本
                    avg_score = sum(scores) / len(scores)
                    if avg_score > best_score:
                        best_score = avg_score
                        best_strategy = strategy_id
            
            if best_strategy:
                insight = {
                    "task_type": task_type,
                    "best_strategy": best_strategy,
                    "avg_score": best_score,
                    "sample_count": sum(len(s) for s in data["strategies"].values()),
                }
                insights.append(insight)
                self._task_patterns[task_type] = insight
        
        self._cross_task_insights.extend(insights)
        return {
            "status": "success",
            "insights_discovered": len(insights),
            "insights": insights,
        }
    
    def predict_best_strategy(self, task_profile: TaskProfile) -> Optional[str]:
        """预测最佳策略"""
        # 检查是否有已学习的模式
        if task_profile.task_type in self._task_patterns:
            return self._task_patterns[task_profile.task_type]["best_strategy"]
        
        # 基于相似度推断
        best_match = None
        best_similarity = 0.0
        
        task_vec = task_profile.to_vector()
        for task_type, pattern in self._task_patterns.items():
            # 简化的相似度计算
            similarity = random.random()  # 实际应基于任务特征向量计算
            if similarity > best_similarity:
                best_similarity = similarity
                best_match = pattern["best_strategy"]
        
        return best_match
    
    def get_learned_patterns(self) -> Dict[str, Any]:
        """获取已学习的模式"""
        return {
            "task_patterns": self._task_patterns,
            "cross_task_insights_count": len(self._cross_task_insights),
            "recent_insights": self._cross_task_insights[-5:],
        }


# 便捷函数
def create_optimization_system(library: Optional[StrategyLibrary] = None) -> Tuple[
    StrategyLibrary, StrategyEvaluator, StrategyOptimizer, MetaLearningEngine
]:
    """创建完整的策略优化系统"""
    if library is None:
        from strategy_selector import create_default_library
        library = create_default_library()
    
    evaluator = StrategyEvaluator(library)
    optimizer = StrategyOptimizer(library, evaluator)
    meta_learner = MetaLearningEngine(evaluator)
    
    return library, evaluator, optimizer, meta_learner
