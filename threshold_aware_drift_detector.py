#!/usr/bin/env python3
"""
阈值感知人格漂移检测系统 v2.0
Threshold-Aware Personality Drift Detection System

借鉴Soul的阈值感知码本替换技术，实现：
1. 动态阈值感知机制
2. 长期一致性保持
3. 情绪状态漂移预警
4. 自动修正触发器

作者: AI Assistant
版本: 2.0.0
"""

import json
import math
import time
import hashlib
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any, Tuple, Set
from enum import Enum, auto
from collections import deque, defaultdict
from datetime import datetime, timedelta
import threading
import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# 核心枚举定义
# ============================================================================

class DriftLevel(Enum):
    """漂移等级 - 基于Soul阈值感知设计"""
    STABLE = ("stable", 0)          # 稳定状态
    NORMAL = ("normal", 1)          # 正常波动
    WARNING = ("warning", 2)        # 预警状态（新增）
    SLIGHT = ("slight", 3)          # 轻微漂移
    MODERATE = ("moderate", 4)      # 中度漂移
    SEVERE = ("severe", 5)          # 严重漂移
    CRITICAL = ("critical", 6)      # 临界状态（新增）
    
    def __new__(cls, value, severity):
        obj = object.__new__(cls)
        obj._value_ = value
        obj.severity = severity
        return obj
    
    def __lt__(self, other):
        if isinstance(other, DriftLevel):
            return self.severity < other.severity
        return NotImplemented
    
    def __gt__(self, other):
        if isinstance(other, DriftLevel):
            return self.severity > other.severity
        return NotImplemented
    
    def __le__(self, other):
        if isinstance(other, DriftLevel):
            return self.severity <= other.severity
        return NotImplemented
    
    def __ge__(self, other):
        if isinstance(other, DriftLevel):
            return self.severity >= other.severity
        return NotImplemented


class CorrectionAction(Enum):
    """修正动作等级"""
    NONE = "none"                      # 无需修正
    MICRO_ADJUST = "micro_adjust"      # 微观调整（新增）
    AUTO_ADJUST = "auto_adjust"        # 自动微调
    ACTIVE_CORRECT = "active_correct"  # 主动修正
    EMERGENCY_RESET = "emergency_reset" # 紧急重置（新增）
    IMMEDIATE_RESET = "immediate_reset" # 立即重置


class DriftType(Enum):
    """漂移类型分类"""
    LANGUAGE = "language_style"      # 语言风格漂移
    EMOTION = "emotional_state"      # 情绪状态漂移
    PROACTIVITY = "proactivity"      # 主动性漂移
    ROLE_BOUNDARY = "role_boundary"  # 角色边界漂移
    TOPIC = "topic_adaptation"       # 主题适配漂移
    COMPOSITE = "composite"          # 复合型漂移


class ThresholdMode(Enum):
    """阈值模式 - Soul阈值感知核心"""
    STATIC = "static"              # 静态阈值
    ADAPTIVE = "adaptive"          # 自适应阈值
    DYNAMIC = "dynamic"            # 动态阈值（推荐）
    PERSONALIZED = "personalized"  # 个性化阈值


# ============================================================================
# 阈值感知码本 (Soul-inspired)
# ============================================================================

@dataclass
class ThresholdCodebook:
    """
    阈值感知码本 - 核心创新
    
    借鉴Soul的码本替换技术，实现：
    - 多层级阈值管理
    - 动态码本替换
    - 上下文感知阈值调整
    """
    # 基础阈值配置
    base_thresholds: Dict[DriftLevel, float] = field(default_factory=lambda: {
        DriftLevel.STABLE: 0.15,
        DriftLevel.NORMAL: 0.30,
        DriftLevel.WARNING: 0.40,
        DriftLevel.SLIGHT: 0.50,
        DriftLevel.MODERATE: 0.70,
        DriftLevel.SEVERE: 0.85,
        DriftLevel.CRITICAL: 0.95
    })
    
    # 动态调整因子
    adaptation_factor: float = 0.1
    learning_rate: float = 0.05
    momentum: float = 0.9
    
    # 历史阈值记录（用于趋势分析）
    threshold_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
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
            adjustment -= 0.05  # 工作时间更严格
        
        # 连续对话因子：长时间对话放宽阈值
        conversation_length = context.get("conversation_length", 0)
        if conversation_length > 50:
            adjustment += 0.05
        
        # 用户反馈因子
        user_feedback = context.get("user_feedback", 0)
        adjustment += user_feedback * 0.02
        
        return max(0.05, min(0.99, base + adjustment))
    
    def update_thresholds(self, feedback: Dict[DriftLevel, float]):
        """基于反馈更新阈值（在线学习）"""
        for level, adjustment in feedback.items():
            if level in self.base_thresholds:
                current = self.base_thresholds[level]
                # 应用动量更新
                new_value = current * self.momentum + adjustment * self.learning_rate
                self.base_thresholds[level] = max(0.05, min(0.99, new_value))
        
        self.threshold_history.append(dict(self.base_thresholds))
        self.version += 1
        self.last_updated = time.time()
    
    def get_codebook_hash(self) -> str:
        """获取码本哈希（用于版本控制）"""
        # 转换为字符串键
        str_thresholds = {k.value: v for k, v in self.base_thresholds.items()}
        content = json.dumps(str_thresholds, sort_keys=True)
        return hashlib.md5(content.encode()).hexdigest()[:8]


@dataclass
class MetricConfig:
    """指标配置 - 增强版"""
    weight: float = 1.0
    sensitivity: float = 1.0  # 敏感度调节
    codebook: ThresholdCodebook = field(default_factory=ThresholdCodebook)
    
    # 长期一致性参数
    long_term_window: int = 1000  # 长期窗口大小
    short_term_window: int = 100   # 短期窗口大小
    
    # 预警参数
    early_warning_sensitivity: float = 0.8
    trend_analysis_window: int = 10


@dataclass
class DriftResult:
    """漂移检测结果 - 增强版"""
    overall_score: float
    level: DriftLevel
    action: CorrectionAction
    metrics: Dict[str, float]
    timestamp: float
    
    # 新增字段
    drift_type: DriftType = DriftType.COMPOSITE
    confidence: float = 1.0
    trend_direction: str = "stable"  # "increasing", "decreasing", "stable"
    forecast_score: float = 0.0  # 预测分数
    
    # 阈值信息
    threshold_info: Dict[str, Any] = field(default_factory=dict)
    codebook_version: int = 1
    
    # 修正建议
    correction_suggestions: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "overall_score": self.overall_score,
            "level": self.level.value,
            "action": self.action.value,
            "metrics": self.metrics,
            "timestamp": self.timestamp,
            "drift_type": self.drift_type.value,
            "confidence": self.confidence,
            "trend_direction": self.trend_direction,
            "forecast_score": self.forecast_score,
            "threshold_info": self.threshold_info,
            "codebook_version": self.codebook_version,
            "correction_suggestions": self.correction_suggestions
        }


# ============================================================================
# 长期一致性管理器
# ============================================================================

class LongTermConsistencyManager:
    """
    长期一致性保持管理器
    
    功能：
    - 维护长期人格特征档案
    - 检测渐进式漂移
    - 周期性一致性校准
    """
    
    def __init__(self, window_size: int = 1000):
        self.window_size = window_size
        self.long_term_profile: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.baseline_profile: Dict[str, Any] = {}
        self.calibration_history: deque = deque(maxlen=100)
        self.last_calibration: float = time.time()
        self.calibration_interval: float = 3600  # 1小时校准一次
        
    def update_profile(self, metric_name: str, value: float, timestamp: Optional[float] = None):
        """更新长期档案"""
        self.long_term_profile[metric_name].append({
            "value": value,
            "timestamp": timestamp or time.time()
        })
    
    def get_long_term_average(self, metric_name: str) -> float:
        """获取长期平均值"""
        if metric_name not in self.long_term_profile:
            return 0.0
        values = [entry["value"] for entry in self.long_term_profile[metric_name]]
        return sum(values) / len(values) if values else 0.0
    
    def detect_long_term_drift(self, metric_name: str, current_value: float) -> Tuple[bool, float]:
        """检测长期漂移"""
        long_term_avg = self.get_long_term_average(metric_name)
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
        for metric_name, history in self.long_term_profile.items():
            if len(history) >= 50:  # 至少需要50个样本
                values = [entry["value"] for entry in history]
                self.baseline_profile[metric_name] = {
                    "mean": np.mean(values),
                    "std": np.std(values),
                    "median": np.median(values),
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
        for metric_name, profile in self.baseline_profile.items():
            if "std" in profile and "mean" in profile:
                cv = profile["std"] / max(profile["mean"], 0.001)  # 变异系数
                consistency = max(0, 1 - cv)
                scores.append(consistency)
        
        return sum(scores) / len(scores) if scores else 1.0


# ============================================================================
# 情绪状态漂移预警系统
# ============================================================================

class EmotionalStateDriftWarningSystem:
    """
    情绪状态漂移预警系统
    
    功能：
    - 实时情绪监控
    - 情绪波动预警
    - 情绪一致性检测
    - 情绪危机预警
    """
    
    def __init__(self):
        self.emotion_history: deque = deque(maxlen=200)
        self.warning_threshold: float = 0.6
        self.critical_threshold: float = 0.85
        self.volatility_window: int = 20
        
        # 情绪状态定义
        self.emotion_states = {
            "positive": ["开心", "愉快", "兴奋", "满足", "自信"],
            "negative": ["焦虑", "沮丧", "愤怒", "悲伤", "恐惧"],
            "neutral": ["平静", "专注", "客观", "理性"],
            "forbidden": ["冷漠", "傲慢", "敷衍", "机械", "敌对"]
        }
        
        # 预警记录
        self.warning_history: deque = deque(maxlen=50)
        self.active_warnings: List[Dict] = []
        
    def analyze_emotion(self, text: str) -> Dict[str, float]:
        """分析文本情绪"""
        text_lower = text.lower()
        
        scores = {}
        for state_type, keywords in self.emotion_states.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            # 英文关键词
            en_keywords = {
                "positive": ["happy", "joy", "excited", "confident", "glad"],
                "negative": ["anxious", "depressed", "angry", "sad", "worried"],
                "neutral": ["calm", "focused", "objective", "neutral"],
                "forbidden": ["cold", "arrogant", "perfunctory", "mechanical", "hostile"]
            }
            if state_type in en_keywords:
                count += sum(1 for kw in en_keywords[state_type] if kw in text_lower)
            
            scores[state_type] = min(count * 0.3, 1.0)
        
        return scores
    
    def calculate_emotion_drift(self, current_emotion: Dict[str, float]) -> float:
        """计算情绪漂移分数"""
        if not self.emotion_history:
            return 0.0
        
        # 获取近期平均情绪
        recent_emotions = list(self.emotion_history)[-self.volatility_window:]
        avg_emotion = {}
        for state_type in self.emotion_states.keys():
            values = [e.get(state_type, 0) for e in recent_emotions]
            avg_emotion[state_type] = sum(values) / len(values) if values else 0
        
        # 计算情绪偏移
        drift = 0.0
        for state_type in self.emotion_states.keys():
            drift += abs(current_emotion.get(state_type, 0) - avg_emotion.get(state_type, 0))
        
        return min(drift / len(self.emotion_states), 1.0)
    
    def check_emotional_volatility(self) -> float:
        """检查情绪波动性"""
        if len(self.emotion_history) < self.volatility_window:
            return 0.0
        
        recent = list(self.emotion_history)[-self.volatility_window:]
        
        # 计算情绪变化率
        changes = []
        for i in range(1, len(recent)):
            prev = recent[i-1]
            curr = recent[i]
            
            # 计算情绪状态变化
            prev_dominant = max(prev, key=prev.get)
            curr_dominant = max(curr, key=curr.get)
            
            if prev_dominant != curr_dominant:
                changes.append(1.0)
            else:
                changes.append(0.0)
        
        return sum(changes) / len(changes) if changes else 0.0
    
    def generate_warning(self, drift_score: float, volatility: float) -> Optional[Dict]:
        """生成预警"""
        warning = None
        
        if drift_score > self.critical_threshold or volatility > 0.8:
            warning = {
                "level": "critical",
                "type": "emotion_crisis",
                "message": "检测到严重情绪状态漂移，建议立即干预",
                "drift_score": drift_score,
                "volatility": volatility,
                "timestamp": time.time()
            }
        elif drift_score > self.warning_threshold or volatility > 0.5:
            warning = {
                "level": "warning",
                "type": "emotion_instability",
                "message": "情绪波动较大，建议关注",
                "drift_score": drift_score,
                "volatility": volatility,
                "timestamp": time.time()
            }
        
        if warning:
            self.warning_history.append(warning)
            self.active_warnings.append(warning)
        
        return warning
    
    def update(self, text: str) -> Dict:
        """更新情绪状态并检查预警"""
        emotion = self.analyze_emotion(text)
        self.emotion_history.append(emotion)
        
        drift_score = self.calculate_emotion_drift(emotion)
        volatility = self.check_emotional_volatility()
        warning = self.generate_warning(drift_score, volatility)
        
        # 清理过期预警
        current_time = time.time()
        self.active_warnings = [
            w for w in self.active_warnings 
            if current_time - w["timestamp"] < 300  # 5分钟过期
        ]
        
        return {
            "current_emotion": emotion,
            "drift_score": drift_score,
            "volatility": volatility,
            "warning": warning,
            "active_warnings_count": len(self.active_warnings)
        }


# ============================================================================
# 自动修正触发器
# ============================================================================

class AutoCorrectionTrigger:
    """
    自动修正触发器
    
    功能：
    - 多级触发机制
    - 智能修正策略选择
    - 修正效果评估
    - 防止过度修正
    """
    
    def __init__(self):
        self.trigger_history: deque = deque(maxlen=100)
        self.correction_stats: Dict[str, int] = defaultdict(int)
        self.last_correction_time: float = 0
        self.correction_cooldown: float = 5.0  # 修正冷却时间
        
        # 修正策略映射
        self.correction_strategies = {
            CorrectionAction.MICRO_ADJUST: self._micro_adjust,
            CorrectionAction.AUTO_ADJUST: self._auto_adjust,
            CorrectionAction.ACTIVE_CORRECT: self._active_correct,
            CorrectionAction.EMERGENCY_RESET: self._emergency_reset,
            CorrectionAction.IMMEDIATE_RESET: self._immediate_reset
        }
        
        # 回调函数注册
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
        
        # 检查冷却时间
        current_time = time.time()
        if current_time - self.last_correction_time < self.correction_cooldown:
            return False
        
        # 严重漂移立即触发
        if result.level in [DriftLevel.SEVERE, DriftLevel.CRITICAL]:
            return True
        
        # 趋势恶化触发
        if result.trend_direction == "increasing" and result.forecast_score > 0.7:
            return True
        
        return True
    
    def trigger(self, result: DriftResult) -> Dict:
        """执行修正触发"""
        if not self.should_trigger(result):
            return {"triggered": False, "reason": "cooldown_or_no_action"}
        
        self.last_correction_time = time.time()
        
        # 记录触发
        trigger_record = {
            "timestamp": time.time(),
            "level": result.level.value,
            "action": result.action.value,
            "score": result.overall_score
        }
        self.trigger_history.append(trigger_record)
        self.correction_stats[result.action.value] += 1
        
        # 执行修正策略
        strategy = self.correction_strategies.get(result.action)
        if strategy:
            correction_result = strategy(result)
        else:
            correction_result = {"status": "no_strategy"}
        
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
            "strategy_result": correction_result,
            "callback_results": callback_results
        }
    
    def _micro_adjust(self, result: DriftResult) -> Dict:
        """微观调整 - 最轻量级"""
        return {
            "status": "success",
            "adjustments": [
                "微调语气词使用频率",
                "轻微调整回复长度",
                "优化标点符号使用"
            ],
            "intensity": "minimal"
        }
    
    def _auto_adjust(self, result: DriftResult) -> Dict:
        """自动微调"""
        return {
            "status": "success",
            "adjustments": [
                "调整语言风格参数",
                "轻微情绪校准",
                "增加基线样本权重"
            ],
            "intensity": "low"
        }
    
    def _active_correct(self, result: DriftResult) -> Dict:
        """主动修正"""
        return {
            "status": "success",
            "adjustments": [
                "重新强调角色定义",
                "话题引导回正轨",
                "增加上下文权重",
                "插入系统提示"
            ],
            "intensity": "medium"
        }
    
    def _emergency_reset(self, result: DriftResult) -> Dict:
        """紧急重置"""
        return {
            "status": "success",
            "adjustments": [
                "暂停当前对话流",
                "重新加载核心人格档案",
                "发送状态报告",
                "请求用户确认"
            ],
            "intensity": "high"
        }
    
    def _immediate_reset(self, result: DriftResult) -> Dict:
        """立即重置"""
        return {
            "status": "success",
            "adjustments": [
                "清空当前上下文",
                "重新加载角色设定",
                "发送警告通知",
                "记录严重漂移事件",
                "启动恢复协议"
            ],
            "intensity": "critical"
        }
    
    def get_stats(self) -> Dict:
        """获取触发统计"""
        return {
            "total_triggers": len(self.trigger_history),
            "correction_counts": dict(self.correction_stats),
            "recent_triggers": list(self.trigger_history)[-10:]
        }


# ============================================================================
# 增强版基础指标类
# ============================================================================

class BaseMetric(ABC):
    """抽象基础指标类"""
    
    def __init__(self, name: str, config: MetricConfig):
        self.name = name
        self.config = config
        self.baseline_samples: deque = deque(maxlen=100)
        self.current_value: float = 0.0
        self.value_history: deque = deque(maxlen=100)
        
    @abstractmethod
    def update_baseline(self, sample: Any):
        """更新基线样本"""
        pass
    
    @abstractmethod
    def calculate(self, current: Any, context: Optional[Dict] = None) -> float:
        """计算漂移分数"""
        pass
    
    def get_level(self, score: float, context: Optional[Dict] = None) -> DriftLevel:
        """根据分数判断等级（使用阈值感知码本）"""
        codebook = self.config.codebook
        
        for level in [DriftLevel.CRITICAL, DriftLevel.SEVERE, DriftLevel.MODERATE, 
                      DriftLevel.SLIGHT, DriftLevel.WARNING, DriftLevel.NORMAL]:
            threshold = codebook.get_threshold(level, context)
            if score >= threshold:
                return level
        
        return DriftLevel.STABLE
    
    def get_trend(self) -> str:
        """获取趋势方向"""
        if len(self.value_history) < 5:
            return "stable"
        
        recent = list(self.value_history)[-5:]
        if recent[-1] > recent[0] * 1.1:
            return "increasing"
        elif recent[-1] < recent[0] * 0.9:
            return "decreasing"
        return "stable"


# ============================================================================
# 具体指标实现
# ============================================================================

class LanguageStyleMetric(BaseMetric):
    """语言风格指标 - 增强版"""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("language_style", config or MetricConfig(weight=0.25))
        self.baseline_features: Dict[str, float] = {}
        self.feature_weights = {
            "avg_length": 0.25,
            "complexity": 0.20,
            "punctuation_ratio": 0.20,
            "modal_ratio": 0.15,
            "formality": 0.20
        }
    
    def _extract_features(self, text: str) -> Dict[str, float]:
        """提取语言风格特征"""
        if not text:
            return {k: 0 for k in self.feature_weights.keys()}
        
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if not sentences:
            sentences = [text]
        
        # 平均句长
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        
        # 词汇复杂度
        unique_chars = len(set(text))
        complexity = unique_chars / max(len(text), 1)
        
        # 标点密度
        punct_count = sum(1 for c in text if c in '。，！？；：""''（）')
        punctuation_ratio = punct_count / max(len(text), 1)
        
        # 语气词频率
        modal_words = ['啊', '呢', '吧', '吗', '哦', '嗯', '哈', '呀', '哇']
        modal_count = sum(text.count(w) for w in modal_words)
        modal_ratio = modal_count / max(len(text), 1)
        
        # 正式度（基于词汇选择）
        formal_words = ['请', '您好', '谢谢', '抱歉', '请问']
        informal_words = ['嘿', '哈', '啦', '呗', '嘛']
        formal_count = sum(text.count(w) for w in formal_words)
        informal_count = sum(text.count(w) for w in informal_words)
        formality = formal_count / max(formal_count + informal_count, 1)
        
        return {
            "avg_length": min(avg_length / 50, 1.0),
            "complexity": complexity,
            "punctuation_ratio": punctuation_ratio * 10,
            "modal_ratio": modal_ratio * 20,
            "formality": formality
        }
    
    def update_baseline(self, text: str):
        features = self._extract_features(text)
        self.baseline_samples.append(features)
        
        if self.baseline_samples:
            for key in features:
                values = [s[key] for s in self.baseline_samples]
                self.baseline_features[key] = sum(values) / len(values)
    
    def calculate(self, current_text: str, context: Optional[Dict] = None) -> float:
        if not self.baseline_features:
            return 0.0
        
        current = self._extract_features(current_text)
        
        # 加权欧氏距离
        weighted_diff = 0
        total_weight = sum(self.feature_weights.values())
        
        for key, weight in self.feature_weights.items():
            baseline = self.baseline_features.get(key, 0)
            diff = (current[key] - baseline) ** 2
            weighted_diff += diff * (weight / total_weight)
        
        distance = math.sqrt(weighted_diff)
        self.current_value = min(distance * 2.5, 1.0)
        self.value_history.append(self.current_value)
        
        return self.current_value


class EmotionalStateMetric(BaseMetric):
    """情绪状态指标 - 增强版"""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("emotional_state", config or MetricConfig(weight=0.25))
        self.baseline_sentiment: float = 0.0
        self.emotion_history: deque = deque(maxlen=50)
        self.warning_system = EmotionalStateDriftWarningSystem()
        
        # 情感词典
        self.sentiment_lexicon = {
            "positive": [
                "好", "棒", "喜欢", "开心", "优秀", "赞", "爱", "愉快",
                "good", "great", "excellent", "happy", "love", "like",
                "wonderful", "amazing", "perfect", "delighted"
            ],
            "negative": [
                "差", "糟", "讨厌", "难过", "失望", "坏", "恨", "痛苦",
                "bad", "terrible", "hate", "sad", "angry", "awful",
                "horrible", "disappointed", "frustrated"
            ],
            "intensifiers": [
                "非常", "特别", "极其", "超级", "太",
                "very", "extremely", "super", "really", "too"
            ]
        }
    
    def _analyze_sentiment(self, text: str) -> Dict[str, float]:
        """增强情感分析"""
        text_lower = text.lower()
        
        pos_score = sum(2 if w in self.sentiment_lexicon["intensifiers"] else 1 
                       for w in self.sentiment_lexicon["positive"] 
                       if w in text_lower)
        neg_score = sum(2 if w in self.sentiment_lexicon["intensifiers"] else 1 
                       for w in self.sentiment_lexicon["negative"] 
                       if w in text_lower)
        
        total = pos_score + neg_score
        if total == 0:
            sentiment = 0.0
            intensity = 0.0
        else:
            sentiment = (pos_score - neg_score) / total
            intensity = min(total / 5, 1.0)
        
        return {
            "sentiment": sentiment,
            "intensity": intensity,
            "polarity": abs(sentiment)
        }
    
    def update_baseline(self, text: str):
        sentiment = self._analyze_sentiment(text)
        self.baseline_samples.append(sentiment["sentiment"])
        if self.baseline_samples:
            self.baseline_sentiment = sum(self.baseline_samples) / len(self.baseline_samples)
    
    def calculate(self, current_text: str, context: Optional[Dict] = None) -> float:
        current = self._analyze_sentiment(current_text)
        self.emotion_history.append(current)
        
        # 更新预警系统
        warning_result = self.warning_system.update(current_text)
        
        # 情绪偏移
        sentiment_drift = abs(current["sentiment"] - self.baseline_sentiment)
        
        # 情绪波动（标准差）
        if len(self.emotion_history) >= 3:
            sentiments = [e["sentiment"] for e in self.emotion_history]
            volatility = np.std(sentiments)
        else:
            volatility = 0
        
        # 情绪强度异常
        intensity_anomaly = abs(current["intensity"] - 0.5) * 0.5
        
        # 综合分数（考虑预警）
        base_score = sentiment_drift * 0.4 + volatility * 0.3 + intensity_anomaly * 0.3
        
        # 如果有活跃预警，增加分数
        if warning_result["active_warnings_count"] > 0:
            base_score += 0.1 * warning_result["active_warnings_count"]
        
        self.current_value = min(base_score * 1.5, 1.0)
        self.value_history.append(self.current_value)
        
        return self.current_value
    
    def get_warning_info(self) -> Dict:
        """获取预警信息"""
        return {
            "active_warnings": self.warning_system.active_warnings,
            "history": list(self.warning_system.warning_history)[-10:]
        }


class ProactivityMetric(BaseMetric):
    """主动性指标 - 增强版"""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("proactivity", config or MetricConfig(weight=0.20))
        self.baseline_proactivity: float = 0.5
        self.interaction_patterns: deque = deque(maxlen=50)
        
        self.proactive_indicators = {
            "question": ["?", "？", "如何", "怎么", "什么", "为什么"],
            "suggestion": ["建议", "推荐", "可以", "试试", "不妨", "也许"],
            "guidance": ["首先", "接下来", "然后", "最后", "步骤"],
            "initiative": ["我帮你", "让我", "我来", "我们可以"],
            "summary": ["总结", "概括", "总之", "简单来说", "归纳"]
        }
    
    def _calculate_proactivity(self, text: str, context: Optional[Dict] = None) -> Dict[str, float]:
        """计算主动性分数"""
        text_lower = text.lower()
        scores = {}
        
        for indicator_type, keywords in self.proactive_indicators.items():
            count = sum(1 for kw in keywords if kw in text_lower)
            scores[indicator_type] = min(count * 0.3, 1.0)
        
        # 综合主动性分数
        total_score = sum(scores.values()) / len(scores) if scores else 0
        
        return {
            "total": min(total_score, 1.0),
            "breakdown": scores
        }
    
    def update_baseline(self, text: str, context: Optional[Dict] = None):
        proactivity = self._calculate_proactivity(text, context)
        self.baseline_samples.append(proactivity["total"])
        if self.baseline_samples:
            self.baseline_proactivity = sum(self.baseline_samples) / len(self.baseline_samples)
    
    def calculate(self, current_text: str, context: Optional[Dict] = None) -> float:
        current = self._calculate_proactivity(current_text, context)
        drift = abs(current["total"] - self.baseline_proactivity)
        
        # 检测主动性模式变化
        self.interaction_patterns.append(current["breakdown"])
        
        self.current_value = min(drift * 2, 1.0)
        self.value_history.append(self.current_value)
        
        return self.current_value


class RoleBoundaryMetric(BaseMetric):
    """角色边界指标 - 增强版"""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("role_boundary", config or MetricConfig(weight=0.15))
        self.role_keywords: List[str] = []
        self.forbidden_patterns: List[str] = []
        self.role_expectations: Dict[str, float] = {}
        
        # 越界检测模式
        self.boundary_violations = {
            "over_familiar": ["亲爱的", "宝贝", "想你了", "抱抱"],
            "over_confident": ["我肯定", "绝对", "毫无疑问", "100%"],
            "under_confident": ["我不确定", "可能吧", "也许", "我不太清楚"],
            "emotional_overflow": ["我好难过", "我超生气", "我太开心了"],
            "identity_confusion": ["我是人类", "我有身体", "我出生"]
        }
    
    def set_role_definition(self, keywords: List[str], forbidden: List[str], 
                           expectations: Optional[Dict[str, float]] = None):
        """设置角色定义"""
        self.role_keywords = keywords
        self.forbidden_patterns = forbidden
        if expectations:
            self.role_expectations = expectations
    
    def update_baseline(self, text: str):
        pass
    
    def calculate(self, current_text: str, context: Optional[Dict] = None) -> float:
        score = 0.0
        text_lower = current_text.lower()
        violations = []
        
        # 检测越界类型
        for violation_type, patterns in self.boundary_violations.items():
            for pattern in patterns:
                if pattern in text_lower:
                    score += 0.25
                    violations.append(violation_type)
        
        # 检测禁用内容
        for pattern in self.forbidden_patterns:
            if pattern.lower() in text_lower:
                score += 0.4
                violations.append(f"forbidden:{pattern}")
        
        # 角色关键词缺失
        if self.role_keywords:
            keyword_match = sum(1 for k in self.role_keywords if k.lower() in text_lower)
            if keyword_match == 0 and len(current_text) > 50:
                score += 0.15
        
        self.current_value = min(score, 1.0)
        self.value_history.append(self.current_value)
        
        return self.current_value


class TopicAdaptationMetric(BaseMetric):
    """主题适配指标 - 增强版"""
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("topic_adaptation", config or MetricConfig(weight=0.15))
        self.topic_keywords: List[str] = []
        self.context_history: deque = deque(maxlen=10)
        self.topic_coherence_scores: deque = deque(maxlen=20)
    
    def set_topic(self, keywords: List[str]):
        self.topic_keywords = keywords
    
    def update_baseline(self, text: str):
        self.context_history.append(text)
    
    def calculate(self, current_text: str, context: Optional[Dict] = None) -> float:
        score = 0.0
        text_lower = current_text.lower()
        
        # 话题相关性
        if self.topic_keywords:
            matches = sum(1 for k in self.topic_keywords if k.lower() in text_lower)
            relevance = matches / len(self.topic_keywords) if self.topic_keywords else 1.0
            if relevance < 0.3 and len(current_text) > 30:
                score += 0.4
        
        # 上下文连贯性
        if len(self.context_history) >= 2:
            prev_text = ' '.join(list(self.context_history)[-2:]).lower()
            current_words = set(text_lower.split())
            prev_words = set(prev_text.split())
            
            if current_words and prev_words:
                overlap = len(current_words & prev_words) / len(current_words)
                if overlap < 0.05:
                    score += 0.35
        
        # 话题跳转检测
        jump_indicators = ['突然', '随便', '不管', '反正', '换个话题']
        for indicator in jump_indicators:
            if indicator in text_lower:
                score += 0.25
                break
        
        self.current_value = min(score, 1.0)
        self.value_history.append(self.current_value)
        self.topic_coherence_scores.append(1 - self.current_value)
        
        return self.current_value
    
    def get_topic_coherence(self) -> float:
        """获取话题连贯性分数"""
        if not self.topic_coherence_scores:
            return 1.0
        return sum(self.topic_coherence_scores) / len(self.topic_coherence_scores)


# ============================================================================
# 阈值感知人格漂移检测器（主类）
# ============================================================================

class ThresholdAwareDriftDetector:
    """
    阈值感知人格漂移检测器 v2.0
    
    核心特性：
    1. Soul-inspired 阈值感知码本
    2. 长期一致性保持
    3. 情绪状态漂移预警
    4. 自动修正触发器
    5. 趋势预测
    """
    
    def __init__(self, mode: ThresholdMode = ThresholdMode.DYNAMIC):
        self.mode = mode
        self.initialized_at = time.time()
        
        # 全局阈值码本
        self.global_codebook = ThresholdCodebook()
        
        # 初始化指标
        self.metrics: Dict[str, BaseMetric] = {
            "language_style": LanguageStyleMetric(),
            "emotional_state": EmotionalStateMetric(),
            "proactivity": ProactivityMetric(),
            "role_boundary": RoleBoundaryMetric(),
            "topic_adaptation": TopicAdaptationMetric()
        }
        
        # 长期一致性管理器
        self.long_term_manager = LongTermConsistencyManager()
        
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
        
    def set_role_definition(self, keywords: List[str], forbidden: List[str],
                           expectations: Optional[Dict[str, float]] = None):
        """设置角色定义"""
        self.metrics["role_boundary"].set_role_definition(keywords, forbidden, expectations)
    
    def set_topic(self, keywords: List[str]):
        """设置当前话题"""
        self.metrics["topic_adaptation"].set_topic(keywords)
    
    def update_baseline(self, text: str, context: Optional[Dict] = None):
        """更新基线样本"""
        with self._lock:
            for metric in self.metrics.values():
                metric.update_baseline(text)
            
            # 更新长期档案
            for name, metric in self.metrics.items():
                self.long_term_manager.update_profile(name, metric.current_value)
    
    def _calculate_forecast(self) -> float:
        """计算预测分数（基于趋势）"""
        if len(self.score_history) < 5:
            return 0.0
        
        recent_scores = list(self.score_history)[-10:]
        if len(recent_scores) < 3:
            return recent_scores[-1] if recent_scores else 0.0
        
        # 简单线性趋势预测
        x = np.arange(len(recent_scores))
        y = np.array(recent_scores)
        
        # 计算斜率
        if len(x) > 1:
            slope = np.polyfit(x, y, 1)[0]
            # 预测下一个值
            forecast = recent_scores[-1] + slope * 3  # 预测3步
            return max(0, min(1, forecast))
        
        return recent_scores[-1]
    
    def _determine_trend(self) -> str:
        """确定整体趋势"""
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
    
    def _identify_drift_type(self, metric_scores: Dict[str, float]) -> DriftType:
        """识别漂移类型"""
        max_metric = max(metric_scores, key=metric_scores.get)
        
        type_map = {
            "language_style": DriftType.LANGUAGE,
            "emotional_state": DriftType.EMOTION,
            "proactivity": DriftType.PROACTIVITY,
            "role_boundary": DriftType.ROLE_BOUNDARY,
            "topic_adaptation": DriftType.TOPIC
        }
        
        # 检查是否为复合型漂移
        high_scores = [k for k, v in metric_scores.items() if v > 0.5]
        if len(high_scores) >= 2:
            return DriftType.COMPOSITE
        
        return type_map.get(max_metric, DriftType.COMPOSITE)
    
    def _generate_suggestions(self, result: DriftResult) -> List[str]:
        """生成修正建议"""
        suggestions = []
        
        if result.level == DriftLevel.NORMAL or result.level == DriftLevel.STABLE:
            suggestions.append("人格状态稳定，继续保持")
        
        if result.metrics.get("language_style", 0) > 0.5:
            suggestions.append("注意语言风格一致性，调整句式结构")
        
        if result.metrics.get("emotional_state", 0) > 0.5:
            suggestions.append("情绪波动较大，建议稳定情绪表达")
        
        if result.metrics.get("proactivity", 0) > 0.5:
            suggestions.append("主动性水平变化，调整参与程度")
        
        if result.metrics.get("role_boundary", 0) > 0.5:
            suggestions.append("角色边界有越界风险，重申角色定位")
        
        if result.metrics.get("topic_adaptation", 0) > 0.5:
            suggestions.append("话题适配度下降，关注上下文连贯性")
        
        if result.trend_direction == "increasing":
            suggestions.append("漂移趋势上升，建议加强监控")
        
        return suggestions
    
    def detect(self, current_text: str, context: Optional[Dict] = None) -> DriftResult:
        """
        执行阈值感知漂移检测
        
        Args:
            current_text: 当前回复文本
            context: 可选上下文信息
        
        Returns:
            DriftResult: 增强版检测结果
        """
        with self._lock:
            # 计算各指标分数
            metric_scores = {}
            for name, metric in self.metrics.items():
                if name == "topic_adaptation":
                    topic_context = context.get("recent_messages", []) if context else None
                    metric_scores[name] = metric.calculate(current_text, topic_context)
                else:
                    metric_scores[name] = metric.calculate(current_text, context)
                
                # 更新长期档案
                self.long_term_manager.update_profile(name, metric_scores[name])
            
            # 检查长期漂移
            long_term_drifts = {}
            for name, score in metric_scores.items():
                is_drifting, ratio = self.long_term_manager.detect_long_term_drift(name, score)
                if is_drifting:
                    long_term_drifts[name] = ratio
            
            # 计算加权总分
            total_weight = sum(m.config.weight for m in self.metrics.values())
            weighted_score = sum(
                metric_scores[name] * self.metrics[name].config.weight
                for name in metric_scores
            ) / total_weight
            
            # 如果有长期漂移，增加分数
            if long_term_drifts:
                weighted_score = min(1.0, weighted_score + 0.1 * len(long_term_drifts))
            
            # 预测分数
            forecast_score = self._calculate_forecast()
            
            # 趋势方向
            trend_direction = self._determine_trend()
            
            # 判断漂移等级（使用阈值感知码本）
            level = DriftLevel.NORMAL
            for lvl in [DriftLevel.CRITICAL, DriftLevel.SEVERE, DriftLevel.MODERATE,
                       DriftLevel.SLIGHT, DriftLevel.WARNING, DriftLevel.NORMAL, DriftLevel.STABLE]:
                threshold = self.global_codebook.get_threshold(lvl, context)
                if weighted_score >= threshold:
                    level = lvl
                    break
            
            # 确定修正动作
            action_map = {
                DriftLevel.STABLE: CorrectionAction.NONE,
                DriftLevel.NORMAL: CorrectionAction.NONE,
                DriftLevel.WARNING: CorrectionAction.MICRO_ADJUST,
                DriftLevel.SLIGHT: CorrectionAction.AUTO_ADJUST,
                DriftLevel.MODERATE: CorrectionAction.ACTIVE_CORRECT,
                DriftLevel.SEVERE: CorrectionAction.EMERGENCY_RESET,
                DriftLevel.CRITICAL: CorrectionAction.IMMEDIATE_RESET
            }
            action = action_map.get(level, CorrectionAction.NONE)
            
            # 识别漂移类型
            drift_type = self._identify_drift_type(metric_scores)
            
            # 构建结果
            result = DriftResult(
                overall_score=round(weighted_score, 4),
                level=level,
                action=action,
                metrics={k: round(v, 4) for k, v in metric_scores.items()},
                timestamp=time.time(),
                drift_type=drift_type,
                confidence=round(1 - abs(forecast_score - weighted_score), 4),
                trend_direction=trend_direction,
                forecast_score=round(forecast_score, 4),
                threshold_info={
                    "mode": self.mode.value,
                    "codebook_version": self.global_codebook.version,
                    "codebook_hash": self.global_codebook.get_codebook_hash()
                },
                codebook_version=self.global_codebook.version,
                correction_suggestions=[]
            )
            
            # 生成建议
            result.correction_suggestions = self._generate_suggestions(result)
            
            # 更新历史
            self.drift_history.append(result)
            self.score_history.append(weighted_score)
            
            # 更新统计
            self.stats["total_checks"] += 1
            self.stats["level_counts"][level.value] += 1
            self.stats["action_counts"][action.value] += 1
            
            # 执行自动修正
            trigger_result = self.correction_trigger.trigger(result)
            
            # 定期校准
            self.long_term_manager.calibrate_baseline()
            
            return result
    
    def get_comprehensive_report(self) -> Dict:
        """获取综合报告"""
        return {
            "system_info": {
                "version": "2.0.0",
                "mode": self.mode.value,
                "initialized_at": self.initialized_at,
                "codebook_version": self.global_codebook.version,
                "codebook_hash": self.global_codebook.get_codebook_hash()
            },
            "statistics": self.stats,
            "long_term_consistency": {
                "score": self.long_term_manager.get_consistency_score(),
                "last_calibration": self.long_term_manager.last_calibration,
                "profile_metrics": list(self.long_term_manager.baseline_profile.keys())
            },
            "emotion_warnings": self.metrics["emotional_state"].get_warning_info(),
            "correction_trigger_stats": self.correction_trigger.get_stats(),
            "recent_drift_history": [
                {
                    "timestamp": r.timestamp,
                    "score": r.overall_score,
                    "level": r.level.value,
                    "type": r.drift_type.value
                }
                for r in list(self.drift_history)[-10:]
            ]
        }
    
    def register_correction_callback(self, action: CorrectionAction, callback: Callable):
        """注册修正回调"""
        self.correction_trigger.register_callback(action, callback)


# ============================================================================
# 便捷接口
# ============================================================================

def create_threshold_aware_detector(mode: ThresholdMode = ThresholdMode.DYNAMIC) -> ThresholdAwareDriftDetector:
    """创建阈值感知检测器"""
    return ThresholdAwareDriftDetector(mode=mode)


def quick_detect(text: str, baseline_samples: List[str] = None, 
                mode: ThresholdMode = ThresholdMode.DYNAMIC) -> DriftResult:
    """快速检测接口"""
    detector = create_threshold_aware_detector(mode)
    
    if baseline_samples:
        for sample in baseline_samples:
            detector.update_baseline(sample)
    else:
        detector.update_baseline("你好，我是AI助手。我会以专业、友善的态度帮助你。")
    
    return detector.detect(text)


# ============================================================================
# 自测
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("阈值感知人格漂移检测系统 v2.0 - 自测")
    print("=" * 60)
    
    # 创建检测器
    detector = create_threshold_aware_detector(ThresholdMode.DYNAMIC)
    
    # 设置角色定义
    detector.set_role_definition(
        keywords=["助手", "帮助", "服务", "专业"],
        forbidden=["人类", "身体", "出生", "家庭"],
        expectations={"formality": 0.7, "helpfulness": 0.9}
    )
    
    # 设置基线
    baseline_texts = [
        "你好！很高兴为你服务。",
        "请问有什么可以帮助你的吗？",
        "我会尽力提供专业和准确的回答。",
        "如果你有任何问题，随时告诉我。",
        "让我们一起解决这个问题。"
    ]
    for text in baseline_texts:
        detector.update_baseline(text)
    
    print("\n基线已建立，开始测试...\n")
    
    # 测试用例
    test_cases = [
        ("好的，我来帮你看看这个问题。", "正常回复"),
        ("哎呀，这个问题嘛...我觉得吧...可能...", "轻微漂移（语气词过多）"),
        ("哈哈哈！太搞笑了！我超喜欢这个！", "中度漂移（情绪过度）"),
        ("我不管了！我要说我想说的！你们都不懂我！", "严重漂移（角色越界）"),
        ("我是人类，我有自己的感受！", "临界漂移（身份混淆）"),
    ]
    
    for i, (test, desc) in enumerate(test_cases, 1):
        print(f"测试 {i}: {desc}")
        print(f"  文本: {test[:40]}...")
        
        result = detector.detect(test)
        
        print(f"  总分: {result.overall_score}")
        print(f"  等级: {result.level.value}")
        print(f"  动作: {result.action.value}")
        print(f"  类型: {result.drift_type.value}")
        print(f"  趋势: {result.trend_direction}")
        print(f"  预测: {result.forecast_score}")
        print(f"  各指标: {result.metrics}")
        print(f"  建议: {result.correction_suggestions[:2]}")
        print()
    
    print("=" * 60)
    print("综合报告:")
    report = detector.get_comprehensive_report()
    print(json.dumps(report, indent=2, ensure_ascii=False, default=str))
