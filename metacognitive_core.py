"""
Metacognitive Core - 元认知与自我改进系统
基于ACT-R认知架构简化版实现

核心功能：
1. 自我监控循环 (Self-Monitoring Loop)
2. 策略选择器 (Strategy Selector)
3. 反思日志系统 (Reflection Log System)

架构设计：
- Declarative Memory: 存储事实性知识
- Procedural Memory: 存储程序性知识（技能/策略）
- Goal Buffer: 当前目标状态
- Metacognitive Buffer: 元认知状态
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
from collections import deque
import json
import time
from abc import ABC, abstractmethod


# ============================================================================
# 基础类型定义
# ============================================================================

class ConfidenceLevel(Enum):
    """置信度级别"""
    VERY_LOW = 0.1
    LOW = 0.3
    MEDIUM = 0.5
    HIGH = 0.7
    VERY_HIGH = 0.9


class StrategyType(Enum):
    """策略类型"""
    ANALYTICAL = auto()      # 分析型
    INTUITIVE = auto()       # 直觉型
    SYSTEMATIC = auto()      # 系统型
    HEURISTIC = auto()       # 启发式
    ADAPTIVE = auto()        # 自适应


@dataclass
class Chunk:
    """ACT-R 记忆块 - 知识的基本单元"""
    id: str
    chunk_type: str
    slots: Dict[str, Any] = field(default_factory=dict)
    activation: float = 0.0  # 激活水平（可及性）
    creation_time: float = field(default_factory=time.time)
    last_access: float = field(default_factory=time.time)
    access_count: int = 0
    
    def update_activation(self, current_time: float):
        """更新激活水平（基于时间衰减和访问频率）"""
        time_diff = current_time - self.last_access
        decay = 0.5  # 衰减率
        # ACT-R 激活公式简化版
        self.activation = (self.access_count * 0.1) - (decay * time_diff)
        self.activation = max(0.0, self.activation)


@dataclass
class Production:
    """ACT-R 产生式规则 - IF-THEN 规则"""
    name: str
    conditions: List[Callable[[Dict], bool]]
    actions: List[Callable[[Dict], Any]]
    utility: float = 0.0  # 效用值（学习后更新）
    success_count: int = 0
    failure_count: int = 0
    
    def calculate_utility(self) -> float:
        """计算效用值（基于成功/失败历史）"""
        total = self.success_count + self.failure_count
        if total == 0:
            return 0.5
        return self.success_count / total


@dataclass
class Goal:
    """目标状态"""
    id: str
    description: str
    priority: int = 5  # 1-10
    status: str = "active"  # active, completed, failed
    subgoals: List['Goal'] = field(default_factory=list)
    progress: float = 0.0  # 0.0 - 1.0
    created_at: float = field(default_factory=time.time)
    deadline: Optional[float] = None


@dataclass
class ReflectionEntry:
    """反思日志条目"""
    timestamp: float
    phase: str  # before, during, after
    observation: str
    evaluation: str
    strategy_used: str
    outcome: str
    lessons_learned: List[str]
    confidence_before: float
    confidence_after: float


@dataclass
class MetacognitiveState:
    """元认知状态 - 对认知的认知"""
    current_task: Optional[str] = None
    confidence: float = 0.5
    cognitive_load: float = 0.5  # 认知负荷 0-1
    attention_focus: Optional[str] = None
    strategy_in_use: Optional[str] = None
    monitoring_active: bool = False
    reflection_triggered: bool = False


# ============================================================================
# 声明性记忆系统 (Declarative Memory)
# ============================================================================

class DeclarativeMemory:
    """声明性记忆 - 存储事实性知识"""
    
    def __init__(self, capacity: int = 1000):
        self.chunks: Dict[str, Chunk] = {}
        self.capacity = capacity
        self.retrieval_threshold = 0.0
        
    def add_chunk(self, chunk: Chunk) -> bool:
        """添加记忆块"""
        if len(self.chunks) >= self.capacity:
            # 遗忘激活最低的记忆块
            self._forget_least_active()
        self.chunks[chunk.id] = chunk
        return True
    
    def retrieve(self, chunk_type: str, **slot_constraints) -> Optional[Chunk]:
        """检索记忆块（基于激活水平和匹配度）"""
        candidates = []
        current_time = time.time()
        
        for chunk in self.chunks.values():
            chunk.update_activation(current_time)
            
            # 类型匹配
            if chunk.chunk_type != chunk_type:
                continue
                
            # 槽位约束匹配
            match_score = 0
            for slot, value in slot_constraints.items():
                if slot in chunk.slots and chunk.slots[slot] == value:
                    match_score += 1
            
            if match_score == len(slot_constraints):
                # 匹配度 + 激活水平 = 总得分
                total_score = match_score + chunk.activation
                candidates.append((chunk, total_score))
        
        if not candidates:
            return None
            
        # 选择得分最高的
        candidates.sort(key=lambda x: x[1], reverse=True)
        best_chunk = candidates[0][0]
        
        # 更新访问统计
        best_chunk.last_access = current_time
        best_chunk.access_count += 1
        
        return best_chunk
    
    def retrieve_by_activation(self, min_activation: float = 0.0) -> List[Chunk]:
        """按激活水平检索记忆块"""
        current_time = time.time()
        result = []
        for chunk in self.chunks.values():
            chunk.update_activation(current_time)
            if chunk.activation >= min_activation:
                result.append(chunk)
        return sorted(result, key=lambda c: c.activation, reverse=True)
    
    def _forget_least_active(self):
        """遗忘激活最低的记忆块"""
        if not self.chunks:
            return
        least_active = min(self.chunks.values(), key=lambda c: c.activation)
        del self.chunks[least_active.id]


# ============================================================================
# 程序性记忆系统 (Procedural Memory)
# ============================================================================

class ProceduralMemory:
    """程序性记忆 - 存储技能和策略（产生式规则）"""
    
    def __init__(self):
        self.productions: Dict[str, Production] = {}
        self.utility_learning_rate = 0.1
        
    def add_production(self, production: Production):
        """添加产生式规则"""
        self.productions[production.name] = production
        
    def match_productions(self, buffer_state: Dict) -> List[Production]:
        """匹配当前状态下的可用产生式"""
        matched = []
        for production in self.productions.values():
            if all(condition(buffer_state) for condition in production.conditions):
                matched.append(production)
        return matched
    
    def select_production(self, buffer_state: Dict) -> Optional[Production]:
        """选择效用最高的产生式（冲突消解）"""
        matched = self.match_productions(buffer_state)
        if not matched:
            return None
        
        # 按效用排序，选择最高的
        matched.sort(key=lambda p: p.calculate_utility(), reverse=True)
        return matched[0]
    
    def update_utility(self, production_name: str, success: bool):
        """更新产生式效用（学习）"""
        if production_name not in self.productions:
            return
            
        production = self.productions[production_name]
        if success:
            production.success_count += 1
        else:
            production.failure_count += 1
        
        production.utility = production.calculate_utility()


# ============================================================================
# 策略选择器 (Strategy Selector)
# ============================================================================

class StrategySelector:
    """策略选择器 - 根据任务特征选择最优策略"""
    
    def __init__(self, procedural_memory: ProceduralMemory):
        self.pm = procedural_memory
        self.strategy_history: deque = deque(maxlen=50)
        
    def analyze_task(self, task_description: str, context: Dict) -> Dict:
        """分析任务特征"""
        features = {
            'complexity': self._estimate_complexity(task_description),
            'novelty': self._estimate_novelty(task_description),
            'time_pressure': context.get('time_pressure', 0.5),
            'accuracy_required': context.get('accuracy_required', 0.5),
            'cognitive_load': context.get('cognitive_load', 0.5)
        }
        return features
    
    def select_strategy(self, task_description: str, context: Dict) -> Tuple[str, float]:
        """选择最佳策略，返回策略名称和置信度"""
        features = self.analyze_task(task_description, context)
        
        # 基于任务特征选择策略
        if features['time_pressure'] > 0.7 and features['accuracy_required'] < 0.6:
            strategy = StrategyType.HEURISTIC
            confidence = 0.7
        elif features['complexity'] > 0.7 and features['accuracy_required'] > 0.8:
            strategy = StrategyType.SYSTEMATIC
            confidence = 0.8
        elif features['novelty'] > 0.7:
            strategy = StrategyType.ADAPTIVE
            confidence = 0.6
        elif features['cognitive_load'] > 0.7:
            strategy = StrategyType.INTUITIVE
            confidence = 0.65
        else:
            strategy = StrategyType.ANALYTICAL
            confidence = 0.75
        
        # 记录选择历史
        self.strategy_history.append({
            'task': task_description[:50],
            'strategy': strategy.name,
            'features': features,
            'confidence': confidence
        })
        
        return strategy.name, confidence
    
    def _estimate_complexity(self, task: str) -> float:
        """估计任务复杂度（简化版）"""
        # 基于任务描述长度、关键词等估计
        complexity_indicators = ['复杂', '多步骤', '依赖', '协调', '分析']
        score = sum(1 for indicator in complexity_indicators if indicator in task)
        return min(1.0, score / len(complexity_indicators) + len(task) / 500)
    
    def _estimate_novelty(self, task: str) -> float:
        """估计任务新颖度（基于历史）"""
        if not self.strategy_history:
            return 0.5
        
        # 检查相似任务历史
        similar_tasks = sum(1 for h in self.strategy_history 
                          if self._similarity(h['task'], task) > 0.7)
        return 1.0 - (similar_tasks / len(self.strategy_history))
    
    def _similarity(self, task1: str, task2: str) -> float:
        """计算任务相似度（简化版）"""
        words1 = set(task1.lower().split())
        words2 = set(task2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union)


# ============================================================================
# 反思日志系统 (Reflection Log System)
# ============================================================================

class ReflectionLog:
    """反思日志系统 - 记录和分析认知过程"""
    
    def __init__(self, max_entries: int = 100):
        self.entries: deque = deque(maxlen=max_entries)
        self.patterns: Dict[str, Any] = {}
        
    def log(self, entry: ReflectionEntry):
        """记录反思条目"""
        self.entries.append(entry)
        self._update_patterns(entry)
    
    def _update_patterns(self, entry: ReflectionEntry):
        """更新认知模式"""
        strategy = entry.strategy_used
        if strategy not in self.patterns:
            self.patterns[strategy] = {
                'usage_count': 0,
                'success_rate': 0.0,
                'avg_confidence_gain': 0.0
            }
        
        pattern = self.patterns[strategy]
        pattern['usage_count'] += 1
        
        # 计算成功（置信度提升）
        confidence_gain = entry.confidence_after - entry.confidence_before
        pattern['avg_confidence_gain'] = (
            (pattern['avg_confidence_gain'] * (pattern['usage_count'] - 1) + confidence_gain)
            / pattern['usage_count']
        )
        
        # 简单成功判断：置信度提升且结果不为失败
        success = confidence_gain > 0 and entry.outcome != "failed"
        pattern['success_rate'] = (
            (pattern['success_rate'] * (pattern['usage_count'] - 1) + (1.0 if success else 0.0))
            / pattern['usage_count']
        )
    
    def get_insights(self) -> List[str]:
        """获取认知洞察"""
        insights = []
        
        # 分析最有效的策略
        if self.patterns:
            best_strategy = max(self.patterns.items(), 
                              key=lambda x: x[1]['success_rate'])
            insights.append(f"最有效的策略: {best_strategy[0]} "
                          f"(成功率: {best_strategy[1]['success_rate']:.2f})")
        
        # 分析置信度变化趋势
        if len(self.entries) >= 5:
            recent = list(self.entries)[-5:]
            avg_gain = sum(e.confidence_after - e.confidence_before for e in recent) / 5
            if avg_gain > 0:
                insights.append(f"近期认知能力呈上升趋势 (平均置信度提升: {avg_gain:.2f})")
            elif avg_gain < 0:
                insights.append(f"近期认知能力呈下降趋势 (平均置信度变化: {avg_gain:.2f})")
        
        # 常见失败模式
        failures = [e for e in self.entries if e.outcome == "failed"]
        if failures:
            common_strategy = max(set(f.strategy_used for f in failures),
                                key=lambda s: sum(1 for f in failures if f.strategy_used == s))
            insights.append(f"需要改进的策略: {common_strategy} (失败次数: "
                          f"{sum(1 for f in failures if f.strategy_used == common_strategy)})")
        
        return insights
    
    def export_log(self) -> str:
        """导出日志为JSON"""
        log_data = {
            'entries': [
                {
                    'timestamp': e.timestamp,
                    'phase': e.phase,
                    'observation': e.observation,
                    'evaluation': e.evaluation,
                    'strategy_used': e.strategy_used,
                    'outcome': e.outcome,
                    'lessons_learned': e.lessons_learned,
                    'confidence_before': e.confidence_before,
                    'confidence_after': e.confidence_after
                }
                for e in self.entries
            ],
            'patterns': self.patterns,
            'insights': self.get_insights()
        }
        return json.dumps(log_data, indent=2, ensure_ascii=False)


# ============================================================================
# 自我监控循环 (Self-Monitoring Loop)
# ============================================================================

class SelfMonitoringLoop:
    """自我监控循环 - 持续监控认知过程"""
    
    def __init__(self):
        self.state = MetacognitiveState()
        self.monitoring_interval = 1.0  # 监控间隔（秒）
        self.performance_history: deque = deque(maxlen=20)
        self.anomaly_threshold = 0.3
        
    def start_monitoring(self, task: str):
        """开始监控任务"""
        self.state.current_task = task
        self.state.monitoring_active = True
        self.state.confidence = 0.5
        self.performance_history.clear()
        
    def update(self, progress: float, error_rate: float, 
               cognitive_load: float) -> Dict:
        """更新监控状态"""
        self.state.cognitive_load = cognitive_load
        
        # 计算性能指标
        performance = self._calculate_performance(progress, error_rate)
        self.performance_history.append(performance)
        
        # 检测异常
        anomalies = self._detect_anomalies()
        
        # 更新置信度
        self._update_confidence(performance, error_rate)
        
        # 检查是否需要反思
        if self._should_reflect():
            self.state.reflection_triggered = True
        
        return {
            'performance': performance,
            'anomalies': anomalies,
            'confidence': self.state.confidence,
            'cognitive_load': self.state.cognitive_load,
            'reflection_needed': self.state.reflection_triggered
        }
    
    def _calculate_performance(self, progress: float, error_rate: float) -> float:
        """计算综合性能指标"""
        # 进度权重 0.4, 准确率权重 0.6
        accuracy = 1.0 - error_rate
        return 0.4 * progress + 0.6 * accuracy
    
    def _detect_anomalies(self) -> List[str]:
        """检测认知异常"""
        anomalies = []
        
        if len(self.performance_history) < 3:
            return anomalies
        
        recent = list(self.performance_history)[-3:]
        avg_performance = sum(recent) / len(recent)
        
        # 性能突然下降
        if len(self.performance_history) >= 5:
            older = list(self.performance_history)[-5:-2]
            older_avg = sum(older) / len(older)
            if avg_performance < older_avg - self.anomaly_threshold:
                anomalies.append("performance_drop")
        
        # 认知负荷过高
        if self.state.cognitive_load > 0.8:
            anomalies.append("high_cognitive_load")
        
        # 置信度过低
        if self.state.confidence < 0.3:
            anomalies.append("low_confidence")
        
        return anomalies
    
    def _update_confidence(self, performance: float, error_rate: float):
        """更新置信度（基于贝叶斯更新简化版）"""
        # 性能越好，置信度越高；错误率越高，置信度越低
        performance_factor = performance * 0.3
        error_penalty = error_rate * 0.3
        
        # 平滑更新
        target_confidence = 0.5 + performance_factor - error_penalty
        target_confidence = max(0.1, min(0.9, target_confidence))
        
        # 向目标置信度移动
        self.state.confidence += (target_confidence - self.state.confidence) * 0.2
    
    def _should_reflect(self) -> bool:
        """判断是否需要触发反思"""
        if not self.performance_history:
            return False
        
        # 性能持续下降
        if len(self.performance_history) >= 5:
            recent = list(self.performance_history)[-5:]
            if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                return True
        
        # 置信度与性能不匹配
        if self.state.confidence > 0.7 and self.performance_history[-1] < 0.4:
            return True
        
        return self.state.reflection_triggered
    
    def get_state(self) -> MetacognitiveState:
        """获取当前元认知状态"""
        return self.state


# ============================================================================
# 元认知核心系统 (Metacognitive Core)
# ============================================================================

class MetacognitiveCore:
    """元认知核心系统 - 整合所有组件"""
    
    def __init__(self):
        # 初始化记忆系统
        self.declarative_memory = DeclarativeMemory()
        self.procedural_memory = ProceduralMemory()
        
        # 初始化认知组件
        self.strategy_selector = StrategySelector(self.procedural_memory)
        self.monitoring_loop = SelfMonitoringLoop()
        self.reflection_log = ReflectionLog()
        
        # 目标缓冲区
        self.goal_buffer: Optional[Goal] = None
        
        # 初始化默认产生式
        self._init_default_productions()
    
    def _init_default_productions(self):
        """初始化默认产生式规则"""
        # 目标分解规则
        self.procedural_memory.add_production(Production(
            name="decompose_complex_goal",
            conditions=[
                lambda buf: buf.get('goal_complexity', 0) > 0.7,
                lambda buf: buf.get('has_subgoals', False) is False
            ],
            actions=[
                lambda buf: buf.update({'action': 'decompose'})
            ],
            utility=0.8
        ))
        
        # 寻求帮助规则
        self.procedural_memory.add_production(Production(
            name="seek_help_when_stuck",
            conditions=[
                lambda buf: buf.get('stuck_time', 0) > 300,  # 卡住5分钟
                lambda buf: buf.get('retry_count', 0) > 3
            ],
            actions=[
                lambda buf: buf.update({'action': 'seek_help'})
            ],
            utility=0.7
        ))
        
        # 切换策略规则
        self.procedural_memory.add_production(Production(
            name="switch_strategy_on_failure",
            conditions=[
                lambda buf: buf.get('consecutive_failures', 0) > 2,
                lambda buf: buf.get('current_strategy') is not None
            ],
            actions=[
                lambda buf: buf.update({'action': 'switch_strategy'})
            ],
            utility=0.75
        ))
    
    def start_task(self, task_description: str, context: Dict = None) -> Dict:
        """开始新任务"""
        context = context or {}
        
        # 创建目标
        self.goal_buffer = Goal(
            id=f"goal_{int(time.time())}",
            description=task_description
        )
        
        # 启动监控
        self.monitoring_loop.start_monitoring(task_description)
        
        # 选择策略
        strategy, confidence = self.strategy_selector.select_strategy(
            task_description, context
        )
        
        # 记录初始反思
        self.reflection_log.log(ReflectionEntry(
            timestamp=time.time(),
            phase="before",
            observation=f"开始任务: {task_description[:100]}",
            evaluation="任务初始化",
            strategy_used=strategy,
            outcome="started",
            lessons_learned=[],
            confidence_before=confidence,
            confidence_after=confidence
        ))
        
        return {
            'goal_id': self.goal_buffer.id,
            'strategy': strategy,
            'confidence': confidence,
            'cognitive_load': self.monitoring_loop.state.cognitive_load
        }
    
    def monitor(self, progress: float, error_rate: float = 0.0,
                cognitive_load: float = 0.5) -> Dict:
        """执行监控循环"""
        status = self.monitoring_loop.update(progress, error_rate, cognitive_load)
        
        # 如果需要反思，生成反思条目
        if status['reflection_needed']:
            self._trigger_reflection(status)
        
        return status
    
    def _trigger_reflection(self, status: Dict):
        """触发反思过程"""
        state = self.monitoring_loop.get_state()
        
        # 分析当前情况
        observations = []
        if 'performance_drop' in status['anomalies']:
            observations.append("性能出现下降")
        if 'high_cognitive_load' in status['anomalies']:
            observations.append("认知负荷过高")
        if 'low_confidence' in status['anomalies']:
            observations.append("置信度偏低")
        
        # 生成教训
        lessons = []
        if status['performance'] < 0.5:
            lessons.append("当前策略效果不佳，考虑切换")
        if state.cognitive_load > 0.8:
            lessons.append("需要降低认知负荷，简化任务或寻求帮助")
        
        # 记录反思
        self.reflection_log.log(ReflectionEntry(
            timestamp=time.time(),
            phase="during",
            observation="; ".join(observations) if observations else "常规检查点",
            evaluation=f"性能: {status['performance']:.2f}, 置信度: {status['confidence']:.2f}",
            strategy_used=state.strategy_in_use or "unknown",
            outcome="reflection_triggered",
            lessons_learned=lessons,
            confidence_before=state.confidence,
            confidence_after=state.confidence * 0.95  # 反思时暂时降低
        ))
        
        self.monitoring_loop.state.reflection_triggered = False
    
    def complete_task(self, outcome: str, final_confidence: float):
        """完成任务"""
        if self.goal_buffer:
            self.goal_buffer.status = "completed" if outcome == "success" else "failed"
            self.goal_buffer.progress = 1.0
        
        state = self.monitoring_loop.get_state()
        
        # 记录最终反思
        self.reflection_log.log(ReflectionEntry(
            timestamp=time.time(),
            phase="after",
            observation=f"任务完成，结果: {outcome}",
            evaluation="任务结束评估",
            strategy_used=state.strategy_in_use or "unknown",
            outcome=outcome,
            lessons_learned=self._generate_lessons(outcome),
            confidence_before=state.confidence,
            confidence_after=final_confidence
        ))
        
        # 更新产生式效用
        if state.strategy_in_use:
            self.procedural_memory.update_utility(
                f"strategy_{state.strategy_in_use}",
                outcome == "success"
            )
    
    def _generate_lessons(self, outcome: str) -> List[str]:
        """生成经验教训"""
        lessons = []
        
        if outcome == "success":
            lessons.append("当前策略组合有效")
        else:
            lessons.append("需要重新评估策略选择")
        
        # 从历史中学习
        insights = self.reflection_log.get_insights()
        lessons.extend(insights[:2])  # 添加前两条洞察
        
        return lessons
    
    def get_status(self) -> Dict:
        """获取系统状态"""
        state = self.monitoring_loop.get_state()
        
        return {
            'current_task': state.current_task,
            'confidence': state.confidence,
            'cognitive_load': state.cognitive_load,
            'monitoring_active': state.monitoring_active,
            'goal_status': self.goal_buffer.status if self.goal_buffer else None,
            'reflections_count': len(self.reflection_log.entries),
            'insights': self.reflection_log.get_insights()
        }
    
    def export_reflections(self) -> str:
        """导出反思日志"""
        return self.reflection_log.export_log()


# ============================================================================
# 使用示例
# ============================================================================

def demo():
    """演示元认知系统的使用"""
    print("=" * 60)
    print("元认知核心系统演示")
    print("=" * 60)
    
    # 创建系统实例
    mc = MetacognitiveCore()
    
    # 场景1: 开始一个复杂任务
    print("\n[场景1] 开始复杂分析任务")
    print("-" * 40)
    
    result = mc.start_task(
        task_description="分析大规模数据集并生成报告",
        context={
            'time_pressure': 0.3,
            'accuracy_required': 0.9,
            'cognitive_load': 0.4
        }
    )
    print(f"目标ID: {result['goal_id']}")
    print(f"选择策略: {result['strategy']}")
    print(f"初始置信度: {result['confidence']:.2f}")
    
    # 场景2: 模拟任务执行过程中的监控
    print("\n[场景2] 任务执行监控")
    print("-" * 40)
    
    # 模拟进度更新
    for i, (progress, error, load) in enumerate([
        (0.2, 0.05, 0.5),
        (0.4, 0.08, 0.6),
        (0.5, 0.15, 0.7),  # 错误率上升，可能触发反思
        (0.6, 0.10, 0.65),
        (0.8, 0.05, 0.5),
    ]):
        status = mc.monitor(progress, error, load)
        print(f"步骤 {i+1}: 进度={progress:.0%}, 错误率={error:.0%}, "
              f"认知负荷={load:.0%}")
        print(f"  -> 性能: {status['performance']:.2f}, "
              f"置信度: {status['confidence']:.2f}")
        if status['anomalies']:
            print(f"  -> 异常检测: {status['anomalies']}")
        if status['reflection_needed']:
            print(f"  -> [反思触发]")
    
    # 场景3: 完成任务
    print("\n[场景3] 任务完成")
    print("-" * 40)
    
    mc.complete_task(outcome="success", final_confidence=0.85)
    print("任务状态: 已完成")
    
    # 场景4: 查看系统状态和洞察
    print("\n[场景4] 系统状态与洞察")
    print("-" * 40)
    
    status = mc.get_status()
    print(f"反思条目数: {status['reflections_count']}")
    print("\n认知洞察:")
    for insight in status['insights']:
        print(f"  • {insight}")
    
    # 场景5: 导出反思日志
    print("\n[场景5] 反思日志导出（摘要）")
    print("-" * 40)
    
    log = mc.export_reflections()
    log_data = json.loads(log)
    print(f"总条目: {len(log_data['entries'])}")
    print(f"策略模式: {list(log_data['patterns'].keys())}")
    
    print("\n" + "=" * 60)
    print("演示完成")
    print("=" * 60)


if __name__ == "__main__":
    demo()
