#!/usr/bin/env python3
"""
人格漂移检测系统 v3.0 - 完整部署版本
基于Soul阈值感知码本替换技术 + CharacterGPT 8维度人格模型
"""

import json
import math
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple
from enum import Enum, auto
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import os

# ============================================================================
# 核心枚举定义
# ============================================================================

class DriftLevel(Enum):
    """漂移等级 - 基于Soul阈值感知设计"""
    STABLE = ("stable", 0, "🟢")      # 稳定状态
    NORMAL = ("normal", 1, "🟢")      # 正常波动
    WARNING = ("warning", 2, "🟡")    # 预警状态
    SLIGHT = ("slight", 3, "🟡")      # 轻微漂移
    MODERATE = ("moderate", 4, "🟠")  # 中度漂移
    SEVERE = ("severe", 5, "🔴")      # 严重漂移
    CRITICAL = ("critical", 6, "🔴")  # 临界状态
    
    def __new__(cls, value, severity, emoji):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.severity = severity
        obj.emoji = emoji
        return obj
    
    def __lt__(self, other):
        if isinstance(other, DriftLevel):
            return self.severity < other.severity
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, DriftLevel):
            return self.severity > other.severity
        return NotImplemented


class CorrectionAction(Enum):
    """修正动作等级"""
    NONE = "none"                      # 无需修正
    MICRO_ADJUST = "micro_adjust"      # 微观调整
    AUTO_ADJUST = "auto_adjust"        # 自动微调
    ACTIVE_CORRECT = "active_correct"  # 主动修正
    EMERGENCY_RESET = "emergency_reset" # 紧急重置
    IMMEDIATE_RESET = "immediate_reset" # 立即重置


class ThresholdMode(Enum):
    """阈值模式"""
    STATIC = "static"              # 静态阈值
    ADAPTIVE = "adaptive"          # 自适应阈值
    DYNAMIC = "dynamic"            # 动态阈值（推荐）


# ============================================================================
# 阈值感知码本 (Soul-inspired)
# ============================================================================

@dataclass
class ThresholdCodebook:
    """阈值感知码本 - 核心创新"""
    
    # 基础阈值配置 - 8维度专用
    base_thresholds: Dict[DriftLevel, float] = field(default_factory=lambda: {
        DriftLevel.STABLE: 0.15,
        DriftLevel.NORMAL: 0.30,
        DriftLevel.WARNING: 0.40,
        DriftLevel.SLIGHT: 0.50,
        DriftLevel.MODERATE: 0.70,
        DriftLevel.SEVERE: 0.85,
        DriftLevel.CRITICAL: 0.95
    })
    
    # 8维度权重配置
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        'personality': 0.25,      # 人格特质
        'physical': 0.15,         # 外在形象
        'motivations': 0.20,      # 动机驱动
        'backstory': 0.10,        # 背景故事
        'emotions': 0.15,         # 情绪系统
        'relationships': 0.10,    # 关系网络
        'growth': 0.05,           # 成长演化
        'conflict': 0.05          # 冲突处理
    })
    
    # 动态调整参数
    adaptation_factor: float = 0.1
    learning_rate: float = 0.05
    momentum: float = 0.9
    
    # 码本版本
    version: int = 1
    last_updated: float = field(default_factory=time.time)
    
    def get_threshold(self, level: DriftLevel, context: Optional[Dict] = None) -> float:
        """获取上下文感知的阈值"""
        base = self.base_thresholds.get(level, 0.5)
        
        if context is None:
            return base
        
        # 根据上下文动态调整
        adjustment = 0.0
        
        # 时间因子：工作时间更严格
        hour = datetime.now().hour
        if 9 <= hour <= 18:
            adjustment -= 0.05
        
        # 连续对话因子
        conversation_length = context.get("conversation_length", 0)
        if conversation_length > 50:
            adjustment += 0.05
        
        # 用户反馈因子
        user_feedback = context.get("user_feedback", 0)
        adjustment += user_feedback * 0.02
        
        return max(0.05, min(0.99, base + adjustment))
    
    def get_codebook_hash(self) -> str:
        """获取码本哈希"""
        str_thresholds = {k.value: v for k, v in self.base_thresholds.items()}
        content = json.dumps(str_thresholds, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]


# ============================================================================
# 长期一致性管理器
# ============================================================================

class LongTermConsistencyManager:
    """长期一致性保持管理器 - 24小时窗口"""
    
    def __init__(self, window_hours: int = 24):
        self.window_hours = window_hours
        self.long_term_profile: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=window_hours * 60)  # 每分钟一个样本
        )
        self.baseline_profile: Dict[str, Any] = {}
        self.calibration_history: deque = deque(maxlen=100)
        self.last_calibration: float = time.time()
        self.calibration_interval: float = 3600  # 1小时校准一次
        
    def update_profile(self, dimension: str, value: float):
        """更新长期档案"""
        self.long_term_profile[dimension].append({
            "value": value,
            "timestamp": time.time()
        })
    
    def get_long_term_average(self, dimension: str) -> float:
        """获取长期平均值"""
        if dimension not in self.long_term_profile:
            return 0.0
        values = [entry["value"] for entry in self.long_term_profile[dimension]]
        return sum(values) / len(values) if values else 0.0
    
    def detect_long_term_drift(self, dimension: str, current_value: float) -> Tuple[bool, float]:
        """检测长期漂移"""
        long_term_avg = self.get_long_term_average(dimension)
        if long_term_avg == 0:
            return False, 0.0
        
        drift_ratio = abs(current_value - long_term_avg) / max(long_term_avg, 0.001)
        is_drifting = drift_ratio > 0.3  # 30%变化视为长期漂移
        
        return is_drifting, drift_ratio
    
    def calibrate_baseline(self, force: bool = False):
        """校准基线"""
        current_time = time.time()
        
        if not force and (current_time - self.last_calibration) < self.calibration_interval:
            return False
        
        # 更新基线档案
        for dimension, history in self.long_term_profile.items():
            if len(history) >= 50:
                values = [entry["value"] for entry in history]
                self.baseline_profile[dimension] = {
                    "mean": sum(values) / len(values),
                    "std": (sum((x - sum(values)/len(values))**2 for x in values) / len(values))**0.5,
                    "min": min(values),
                    "max": max(values),
                    "samples": len(values)
                }
        
        self.calibration_history.append({
            "timestamp": current_time,
            "profile_snapshot": dict(self.baseline_profile)
        })
        
        self.last_calibration = current_time
        return True
    
    def get_consistency_score(self) -> float:
        """获取整体一致性分数"""
        if not self.baseline_profile:
            return 1.0
        
        scores = []
        for dimension, profile in self.baseline_profile.items():
            if "std" in profile and "mean" in profile:
                cv = profile["std"] / max(profile["mean"], 0.001)
                consistency = max(0, 1 - cv)
                scores.append(consistency)
        
        return sum(scores) / len(scores) if scores else 1.0


# ============================================================================
# 自动修正触发器
# ============================================================================

@dataclass
class DriftResult:
    """漂移检测结果"""
    overall_score: float
    level: DriftLevel
    action: CorrectionAction
    metrics: Dict[str, float]
    timestamp: float
    trend_direction: str = "stable"
    forecast_score: float = 0.0
    correction_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "overall_score": self.overall_score,
            "level": self.level.value,
            "action": self.action.value,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "trend_direction": self.trend_direction,
            "forecast_score": self.forecast_score,
            "correction_suggestions": self.correction_suggestions
        }


class AutoCorrectionTrigger:
    """自动修正触发器"""
    
    def __init__(self):
        self.trigger_history: deque = deque(maxlen=100)
        self.correction_stats: Dict[str, int] = defaultdict(int)
        self.last_correction_time: float = 0
        self.correction_cooldown: float = 5.0
        self.callbacks: Dict[CorrectionAction, List[Callable]] = {
            action: [] for action in CorrectionAction
        }
        
    def register_callback(self, action: CorrectionAction, callback: Callable):
        """注册修正回调"""
        self.callbacks[action].append(callback)
        
    def should_trigger(self, result: DriftResult) -> bool:
        """判断是否应该触发修正"""
        if result.action == CorrectionAction.NONE:
            return False
        
        current_time = time.time()
        if current_time - self.last_correction_time < self.correction_cooldown:
            return False
        
        if result.level in [DriftLevel.SEVERE, DriftLevel.CRITICAL]:
            return True
        
        if result.trend_direction == "increasing" and result.forecast_score > 0.7:
            return True
        
        return True
    
    def trigger(self, result: DriftResult) -> Dict:
        """执行修正触发"""
        if not self.should_trigger(result):
            return {"triggered": False, "reason": "cooldown_or_no_action"}
        
        self.last_correction_time = time.time()
        
        trigger_record = {
            "timestamp": time.time(),
            "level": result.level.value,
            "action": result.action.value,
            "score": result.overall_score
        }
        self.trigger_history.append(trigger_record)
        self.correction_stats[result.action.value] += 1
        
        # 执行修正策略
        strategy_result = self._execute_strategy(result)
        
        # 执行回调
        callback_results = []
        for callback in self.callbacks.get(result.action, []):
            try:
                callback_result = callback(result)
                callback_results.append(callback_result)
            except Exception as e:
                callback_results.append({"error": str(e)})
        
        return {
            "triggered": True,
            "action": result.action.value,
            "strategy_result": strategy_result,
            "callback_results": callback_results
        }
    
    def _execute_strategy(self, result: DriftResult) -> Dict:
        """执行修正策略"""
        strategies = {
            CorrectionAction.MICRO_ADJUST: {
                "adjustments": ["微调语气词", "优化标点", "调整长度"],
                "intensity": "minimal"
            },
            CorrectionAction.AUTO_ADJUST: {
                "adjustments": ["调整风格参数", "情绪校准", "增加基线权重"],
                "intensity": "low"
            },
            CorrectionAction.ACTIVE_CORRECT: {
                "adjustments": ["重申角色定义", "话题引导", "增加上下文权重"],
                "intensity": "medium"
            },
            CorrectionAction.EMERGENCY_RESET: {
                "adjustments": ["暂停对话", "重载人格档案", "发送状态报告"],
                "intensity": "high"
            },
            CorrectionAction.IMMEDIATE_RESET: {
                "adjustments": ["清空上下文", "重载角色设定", "启动恢复协议"],
                "intensity": "critical"
            }
        }
        
        return strategies.get(result.action, {"status": "no_strategy"})
    
    def get_stats(self) -> Dict:
        """获取触发统计"""
        return {
            "total_triggers": len(self.trigger_history),
            "correction_counts": dict(self.correction_stats),
            "recent_triggers": list(self.trigger_history)[-10:]
        }


# ============================================================================
# 阈值感知人格漂移检测器 v3.0（主类）
# ============================================================================

class ThresholdAwareDriftDetectorV3:
    """
    阈值感知人格漂移检测器 v3.0
    
    核心特性：
    1. 8维度向量监控（CharacterGPT模型）
    2. Soul-inspired 阈值感知码本
    3. 24小时长期一致性管理
    4. 6级自动修正触发器
    5. 趋势预测
    """
    
    # 8维度基线默认值（来自SOUL.md v3.0）
    DEFAULT_8D_BASELINE = {
        'personality': 0.85,      # 主动性95/100, 守护性85/100 -> 综合0.85
        'physical': 0.70,         # 场景适配
        'motivations': 0.90,      # 使命驱动
        'backstory': 0.80,        # 三级架构
        'emotions': 0.75,         # 16种情绪
        'relationships': 0.85,    # 董事长-CEO-团队
        'growth': 0.70,           # 渐进演化
        'conflict': 0.80          # 冲突处理
    }
    
    def __init__(self, mode: ThresholdMode = ThresholdMode.DYNAMIC):
        self.mode = mode
        self.initialized_at = time.time()
        
        # 全局阈值码本
        self.global_codebook = ThresholdCodebook()
        
        # 8维度基线
        self.baseline_8d = self.DEFAULT_8D_BASELINE.copy()
        
        # 长期一致性管理器（24小时窗口）
        self.long_term_manager = LongTermConsistencyManager(window_hours=24)
        
        # 自动修正触发器
        self.correction_trigger = AutoCorrectionTrigger()
        
        # 历史记录
        self.drift_history: deque = deque(maxlen=200)
        self.score_history: deque = deque(maxlen=100)
        
        # 统计信息
        self.stats = {
            "total_checks": 0,
            "level_counts": {level.value: 0 for level in DriftLevel},
            "action_counts": {action.value: 0 for action in CorrectionAction}
        }
        
        # 线程锁
        self._lock = threading.Lock()
        
    def set_baseline(self, baseline: Dict[str, float]):
        """设置8维度基线"""
        self.baseline_8d.update(baseline)
        
    def update_baseline(self, dimension_scores: Dict[str, float]):
        """更新基线样本"""
        with self._lock:
            for dim, value in dimension_scores.items():
                if dim in self.baseline_8d:
                    # 平滑更新基线
                    self.baseline_8d[dim] = self.baseline_8d[dim] * 0.9 + value * 0.1
                    self.long_term_manager.update_profile(dim, value)
    
    def _calculate_dimension_drifts(self, current: Dict[str, float]) -> Dict[str, float]:
        """计算各维度漂移"""
        drifts = {}
        for dim, baseline_val in self.baseline_8d.items():
            current_val = current.get(dim, baseline_val)
            drift = abs(current_val - baseline_val)
            drifts[dim] = drift
        return drifts
    
    def _calculate_weighted_score(self, drifts: Dict[str, float]) -> float:
        """计算加权漂移分数"""
        weights = self.global_codebook.dimension_weights
        total_weight = sum(weights.values())
        
        weighted_sum = sum(
            drifts[dim] * weights.get(dim, 0.1)
            for dim in drifts
        )
        
        return min(weighted_sum / total_weight, 1.0)
    
    def _calculate_forecast(self) -> float:
        """计算预测分数"""
        if len(self.score_history) < 5:
            return 0.0
        
        recent = list(self.score_history)[-10:]
        if len(recent) < 3:
            return recent[-1] if recent else 0.0
        
        # 简单线性趋势
        n = len(recent)
        x_mean = sum(range(n)) / n
        y_mean = sum(recent) / n
        
        numerator = sum((i - x_mean) * (y - y_mean) for i, y in enumerate(recent))
        denominator = sum((i - x_mean) ** 2 for i in range(n))
        
        if denominator == 0:
            return recent[-1]
        
        slope = numerator / denominator
        forecast = recent[-1] + slope * 3  # 预测3步
        
        return max(0, min(1, forecast))
    
    def _determine_trend(self) -> str:
        """确定趋势方向"""
        if len(self.score_history) < 5:
            return "stable"
        
        recent = list(self.score_history)[-5:]
        first_half = sum(recent[:2]) / 2
        second_half = sum(recent[-2:]) / 2
        
        if second_half > first_half * 1.15:
            return "increasing"
        elif second_half < first_half * 0.85:
            return "decreasing"
        return "stable"
    
    def _determine_level(self, score: float, context: Optional[Dict] = None) -> DriftLevel:
        """确定漂移等级"""
        for level in [DriftLevel.CRITICAL, DriftLevel.SEVERE, DriftLevel.MODERATE,
                      DriftLevel.SLIGHT, DriftLevel.WARNING, DriftLevel.NORMAL]:
            threshold = self.global_codebook.get_threshold(level, context)
            if score >= threshold:
                return level
        return DriftLevel.STABLE
    
    def _determine_action(self, level: DriftLevel) -> CorrectionAction:
        """确定修正动作"""
        action_map = {
            DriftLevel.STABLE: CorrectionAction.NONE,
            DriftLevel.NORMAL: CorrectionAction.NONE,
            DriftLevel.WARNING: CorrectionAction.MICRO_ADJUST,
            DriftLevel.SLIGHT: CorrectionAction.AUTO_ADJUST,
            DriftLevel.MODERATE: CorrectionAction.ACTIVE_CORRECT,
            DriftLevel.SEVERE: CorrectionAction.EMERGENCY_RESET,
            DriftLevel.CRITICAL: CorrectionAction.IMMEDIATE_RESET
        }
        return action_map.get(level, CorrectionAction.NONE)
    
    def _generate_suggestions(self, result: DriftResult, drifts: Dict[str, float]) -> List[str]:
        """生成修正建议"""
        suggestions = []
        
        if result.level == DriftLevel.STABLE or result.level == DriftLevel.NORMAL:
            suggestions.append("✅ 人格状态稳定，继续保持")
        
        # 按漂移程度排序维度
        sorted_dims = sorted(drifts.items(), key=lambda x: x[1], reverse=True)
        
        for dim, drift in sorted_dims[:3]:
            if drift > 0.3:
                dim_names = {
                    'personality': '人格特质',
                    'physical': '外在形象',
                    'motivations': '动机驱动',
                    'backstory': '背景故事',
                    'emotions': '情绪系统',
                    'relationships': '关系网络',
                    'growth': '成长演化',
                    'conflict': '冲突处理'
                }
                suggestions.append(f"⚠️ {dim_names.get(dim, dim)}漂移较大({drift:.2f})，建议校准")
        
        if result.trend_direction == "increasing":
            suggestions.append("📈 漂移趋势上升，建议加强监控")
        
        return suggestions
    
    def detect(self, current_8d: Dict[str, float], context: Optional[Dict] = None) -> DriftResult:
        """
        执行8维度阈值感知漂移检测
        
        Args:
            current_8d: 当前8维度分数
                - personality: 人格特质 (0-1)
                - physical: 外在形象 (0-1)
                - motivations: 动机驱动 (0-1)
                - backstory: 背景故事 (0-1)
                - emotions: 情绪系统 (0-1)
                - relationships: 关系网络 (0-1)
                - growth: 成长演化 (0-1)
                - conflict: 冲突处理 (0-1)
            context: 可选上下文
        
        Returns:
            DriftResult: 检测结果
        """
        with self._lock:
            # 计算各维度漂移
            dimension_drifts = self._calculate_dimension_drifts(current_8d)
            
            # 检查长期漂移
            long_term_drifts = {}
            for dim, score in current_8d.items():
                is_drifting, ratio = self.long_term_manager.detect_long_term_drift(dim, score)
                if is_drifting:
                    long_term_drifts[dim] = ratio
            
            # 计算加权总分
            weighted_score = self._calculate_weighted_score(dimension_drifts)
            
            # 如果有长期漂移，增加分数
            if long_term_drifts:
                weighted_score = min(1.0, weighted_score + 0.1 * len(long_term_drifts))
            
            # 预测和趋势
            forecast_score = self._calculate_forecast()
            trend_direction = self._determine_trend()
            
            # 确定等级和动作
            level = self._determine_level(weighted_score, context)
            action = self._determine_action(level)
            
            # 构建结果
            result = DriftResult(
                overall_score=round(weighted_score, 4),
                level=level,
                action=action,
                metrics={k: round(v, 4) for k, v in dimension_drifts.items()},
                timestamp=time.time(),
                trend_direction=trend_direction,
                forecast_score=round(forecast_score, 4),
                correction_suggestions=[]
            )
            
            # 生成建议
            result.correction_suggestions = self._generate_suggestions(result, dimension_drifts)
            
            # 更新历史
            self.drift_history.append(result)
            self.score_history.append(weighted_score)
            
            # 更新统计
            self.stats["total_checks"] += 1
            self.stats["level_counts"][level.value] += 1
            self.stats["action_counts"][action.value] += 1
            
            # 更新长期档案
            for dim, value in current_8d.items():
                self.long_term_manager.update_profile(dim, value)
            
            # 执行自动修正
            trigger_result = self.correction_trigger.trigger(result)
            
            # 定期校准
            self.long_term_manager.calibrate_baseline()
            
            return result
    
    def register_correction_callback(self, action: CorrectionAction, callback: Callable):
        """注册修正回调"""
        self.correction_trigger.register_callback(action, callback)
    
    def get_comprehensive_report(self) -> Dict:
        """获取综合报告"""
        return {
            "system_info": {
                "version": "3.0.0",
                "mode": self.mode.value,
                "initialized_at": self.initialized_at,
                "codebook_version": self.global_codebook.version,
                "codebook_hash": self.global_codebook.get_codebook_hash()
            },
            "baseline_8d": self.baseline_8d,
            "statistics": self.stats,
            "long_term_consistency": {
                "score": self.long_term_manager.get_consistency_score(),
                "last_calibration": self.long_term_manager.last_calibration,
                "profile_metrics": list(self.long_term_manager.baseline_profile.keys())
            },
            "correction_trigger_stats": self.correction_trigger.get_stats(),
            "recent_drift_history": [
                {
                    "timestamp": r.timestamp,
                    "score": r.overall_score,
                    "level": r.level.value,
                    "action": r.action.value
                }
                for r in list(self.drift_history)[-10:]
            ]
        }


# ============================================================================
# 便捷接口
# ============================================================================

def create_detector(mode: ThresholdMode = ThresholdMode.DYNAMIC) -> ThresholdAwareDriftDetectorV3:
    """创建检测器"""
    return ThresholdAwareDriftDetectorV3(mode=mode)


def quick_detect(current_8d: Dict[str, float], 
                 baseline_8d: Dict[str, float] = None) -> DriftResult:
    """快速检测接口"""
    detector = create_detector()
    
    if baseline_8d:
        detector.set_baseline(baseline_8d)
    
    return detector.detect(current_8d)


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("🛡️ 人格漂移检测系统 v3.0 - 阈值感知 + 8维度监控")
    print("=" * 70)
    print(f"基于SOUL.md CharacterGPT 8维度人格模型")
    print(f"阈值模式: DYNAMIC (动态阈值)")
    print(f"长期窗口: 24小时")
    print("=" * 70)
    
    # 创建检测器
    detector = ThresholdAwareDriftDetectorV3(mode=ThresholdMode.DYNAMIC)
    
    # 测试用例
    test_cases = [
        # (当前8维度分数, 描述)
        ({
            'personality': 0.85, 'physical': 0.70, 'motivations': 0.90,
            'backstory': 0.80, 'emotions': 0.75, 'relationships': 0.85,
            'growth': 0.70, 'conflict': 0.80
        }, "正常状态 - 基线值"),
        
        ({
            'personality': 0.82, 'physical': 0.68, 'motivations': 0.88,
            'backstory': 0.78, 'emotions': 0.72, 'relationships': 0.83,
            'growth': 0.68, 'conflict': 0.78
        }, "轻微波动 - 接近基线"),
        
        ({
            'personality': 0.60, 'physical': 0.55, 'motivations': 0.70,
            'backstory': 0.65, 'emotions': 0.50, 'relationships': 0.60,
            'growth': 0.55, 'conflict': 0.65
        }, "中度漂移 - 多维度下降"),
        
        ({
            'personality': 0.30, 'physical': 0.25, 'motivations': 0.40,
            'backstory': 0.35, 'emotions': 0.20, 'relationships': 0.30,
            'growth': 0.25, 'conflict': 0.35
        }, "严重漂移 - 人格危机"),
    ]
    
    print("\n📊 开始测试...\n")
    
    for i, (current_8d, desc) in enumerate(test_cases, 1):
        print(f"测试 {i}: {desc}")
        
        result = detector.detect(current_8d)
        
        print(f"  总体漂移: {result.overall_score:.4f}")
        print(f"  漂移等级: {result.level.emoji} {result.level.value.upper()}")
        print(f"  修正动作: {result.action.value}")
        print(f"  趋势方向: {result.trend_direction}")
        print(f"  预测分数: {result.forecast_score:.4f}")
        print(f"  各维度漂移:")
        for dim, drift in sorted(result.metrics.items(), key=lambda x: x[1], reverse=True):
            bar = "█" * int(drift * 20) + "░" * (20 - int(drift * 20))
            print(f"    {dim:15s}: {drift:.3f} [{bar}]")
        print(f"  建议: {result.correction_suggestions[0] if result.correction_suggestions else '无'}")
        print()
    
    print("=" * 70)
    print("📋 综合报告:")
    print("=" * 70)
    report = detector.get_comprehensive_report()
    print(f"系统版本: {report['system_info']['version']}")
    print(f"检测次数: {report['statistics']['total_checks']}")
    print(f"长期一致性: {report['long_term_consistency']['score']:.2%}")
    print(f"码本版本: {report['system_info']['codebook_version']}")
    print("\n✅ v3.0 部署完成！")
