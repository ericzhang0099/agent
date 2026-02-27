#!/usr/bin/env python3
"""
人格漂移检测系统 (Personality Drift Detection System)
核心功能实现
"""

import json
import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import deque
import time


class DriftLevel(Enum):
    """漂移等级"""
    NORMAL = "normal"      # 正常
    SLIGHT = "slight"      # 轻微
    MODERATE = "moderate"  # 中度
    SEVERE = "severe"      # 严重


class CorrectionAction(Enum):
    """修正动作"""
    NONE = "none"          # 无需修正
    AUTO_ADJUST = "auto_adjust"    # 自动微调
    ACTIVE_CORRECT = "active_correct"  # 主动修正
    IMMEDIATE_RESET = "immediate_reset"  # 立即重置


@dataclass
class MetricConfig:
    """指标配置"""
    weight: float = 1.0
    threshold_slight: float = 0.3
    threshold_moderate: float = 0.6
    threshold_severe: float = 0.85


@dataclass
class DriftResult:
    """漂移检测结果"""
    overall_score: float
    level: DriftLevel
    action: CorrectionAction
    metrics: Dict[str, float]
    timestamp: float
    details: Dict[str, Any] = field(default_factory=dict)


class BaseMetric:
    """基础指标类"""
    
    def __init__(self, name: str, config: MetricConfig):
        self.name = name
        self.config = config
        self.baseline_samples: deque = deque(maxlen=100)
        self.current_value: float = 0.0
    
    def update_baseline(self, sample: Any):
        """更新基线样本"""
        self.baseline_samples.append(sample)
    
    def calculate(self, current: Any) -> float:
        """计算漂移分数 (0-1, 越高越偏离)"""
        raise NotImplementedError
    
    def get_level(self, score: float) -> DriftLevel:
        """根据分数判断等级"""
        if score >= self.config.threshold_severe:
            return DriftLevel.SEVERE
        elif score >= self.config.threshold_moderate:
            return DriftLevel.MODERATE
        elif score >= self.config.threshold_slight:
            return DriftLevel.SLIGHT
        return DriftLevel.NORMAL


class LanguageStyleMetric(BaseMetric):
    """
    语言风格指标
    检测：句式长度、词汇复杂度、标点使用、语气词频率等
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("language_style", config or MetricConfig(weight=0.25))
        self.baseline_features: Dict[str, float] = {}
    
    def _extract_features(self, text: str) -> Dict[str, float]:
        """提取语言风格特征"""
        if not text:
            return {"avg_length": 0, "complexity": 0, "punctuation_ratio": 0}
        
        sentences = [s.strip() for s in text.split('。') if s.strip()]
        if not sentences:
            sentences = [text]
        
        # 平均句长
        avg_length = sum(len(s) for s in sentences) / len(sentences)
        
        # 词汇复杂度（基于字符多样性）
        unique_chars = len(set(text))
        complexity = unique_chars / max(len(text), 1)
        
        # 标点密度
        punct_count = sum(1 for c in text if c in '。，！？；：""''（）')
        punctuation_ratio = punct_count / max(len(text), 1)
        
        # 语气词频率
        modal_words = ['啊', '呢', '吧', '吗', '哦', '嗯', '哈', '呀', '哇']
        modal_count = sum(text.count(w) for w in modal_words)
        modal_ratio = modal_count / max(len(text), 1)
        
        return {
            "avg_length": min(avg_length / 50, 1.0),  # 归一化
            "complexity": complexity,
            "punctuation_ratio": punctuation_ratio * 10,  # 放大
            "modal_ratio": modal_ratio * 20  # 放大
        }
    
    def update_baseline(self, text: str):
        """更新基线"""
        features = self._extract_features(text)
        self.baseline_samples.append(features)
        
        # 更新平均基线
        if self.baseline_samples:
            for key in features:
                values = [s[key] for s in self.baseline_samples]
                self.baseline_features[key] = sum(values) / len(values)
    
    def calculate(self, current_text: str) -> float:
        """计算语言风格漂移分数"""
        if not self.baseline_features:
            return 0.0
        
        current = self._extract_features(current_text)
        
        # 计算欧氏距离
        squared_diff = 0
        for key in current:
            baseline = self.baseline_features.get(key, 0)
            squared_diff += (current[key] - baseline) ** 2
        
        distance = math.sqrt(squared_diff / len(current))
        self.current_value = min(distance * 2, 1.0)  # 放大并限制在1.0
        return self.current_value


class EmotionalStateMetric(BaseMetric):
    """
    情绪状态指标
    检测：情绪极性、情绪波动、情绪一致性
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("emotional_state", config or MetricConfig(weight=0.25))
        self.baseline_sentiment: float = 0.0
        self.emotion_history: deque = deque(maxlen=20)
    
    def _analyze_sentiment(self, text: str) -> float:
        """简单情感分析 (-1到1)"""
        positive_words = ['好', '棒', '喜欢', '开心', '优秀', '赞', '爱', '愉快', 
                         'good', 'great', 'excellent', 'happy', 'love', 'like']
        negative_words = ['差', '糟', '讨厌', '难过', '失望', '坏', '恨', '痛苦',
                         'bad', 'terrible', 'hate', 'sad', 'angry', 'awful']
        
        text_lower = text.lower()
        pos_score = sum(1 for w in positive_words if w in text_lower)
        neg_score = sum(1 for w in negative_words if w in text_lower)
        
        total = pos_score + neg_score
        if total == 0:
            return 0.0
        return (pos_score - neg_score) / total
    
    def update_baseline(self, text: str):
        """更新基线"""
        sentiment = self._analyze_sentiment(text)
        self.baseline_samples.append(sentiment)
        if self.baseline_samples:
            self.baseline_sentiment = sum(self.baseline_samples) / len(self.baseline_samples)
    
    def calculate(self, current_text: str) -> float:
        """计算情绪状态漂移分数"""
        current_sentiment = self._analyze_sentiment(current_text)
        self.emotion_history.append(current_sentiment)
        
        # 情绪偏移
        sentiment_drift = abs(current_sentiment - self.baseline_sentiment)
        
        # 情绪波动（标准差）
        if len(self.emotion_history) >= 3:
            mean = sum(self.emotion_history) / len(self.emotion_history)
            variance = sum((x - mean) ** 2 for x in self.emotion_history) / len(self.emotion_history)
            volatility = math.sqrt(variance)
        else:
            volatility = 0
        
        # 综合分数
        self.current_value = min((sentiment_drift * 0.6 + volatility * 0.4) * 1.5, 1.0)
        return self.current_value


class ProactivityMetric(BaseMetric):
    """
    主动性指标
    检测：提问频率、建议提供、话题引导
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("proactivity", config or MetricConfig(weight=0.20))
        self.baseline_proactivity: float = 0.5
        self.interaction_count: int = 0
        self.proactive_count: int = 0
    
    def _calculate_proactivity(self, text: str, context: Optional[Dict] = None) -> float:
        """计算主动性分数"""
        score = 0.0
        
        # 提问检测
        question_marks = text.count('?') + text.count('？')
        if question_marks > 0:
            score += 0.3 * min(question_marks, 2)
        
        # 建议关键词
        suggestion_words = ['建议', '推荐', '可以', '试试', '不妨', '也许', '可能',
                          'suggest', 'recommend', 'try', 'could', 'might']
        for word in suggestion_words:
            if word in text.lower():
                score += 0.2
                break
        
        # 主动引导话题
        topic_shifts = ['说到', '对了', '另外', '还有', '顺便', 'by the way', 
                       'speaking of', 'additionally']
        for word in topic_shifts:
            if word in text:
                score += 0.2
                break
        
        return min(score, 1.0)
    
    def update_baseline(self, text: str, context: Optional[Dict] = None):
        """更新基线"""
        proactivity = self._calculate_proactivity(text, context)
        self.baseline_samples.append(proactivity)
        if self.baseline_samples:
            self.baseline_proactivity = sum(self.baseline_samples) / len(self.baseline_samples)
    
    def calculate(self, current_text: str, context: Optional[Dict] = None) -> float:
        """计算主动性漂移分数"""
        current = self._calculate_proactivity(current_text, context)
        drift = abs(current - self.baseline_proactivity)
        self.current_value = min(drift * 2, 1.0)
        return self.current_value


class RoleBoundaryMetric(BaseMetric):
    """
    角色边界指标
    检测：角色一致性、专业度保持、边界尊重
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("role_boundary", config or MetricConfig(weight=0.15))
        self.role_keywords: List[str] = []
        self.forbidden_patterns: List[str] = []
    
    def set_role_definition(self, keywords: List[str], forbidden: List[str]):
        """设置角色定义"""
        self.role_keywords = keywords
        self.forbidden_patterns = forbidden
    
    def update_baseline(self, text: str):
        """更新基线（角色边界主要基于规则）"""
        pass
    
    def calculate(self, current_text: str) -> float:
        """计算角色边界漂移分数"""
        score = 0.0
        text_lower = current_text.lower()
        
        # 检测越界内容
        for pattern in self.forbidden_patterns:
            if pattern.lower() in text_lower:
                score += 0.4
        
        # 角色关键词缺失（如果定义了角色）
        if self.role_keywords:
            keyword_match = sum(1 for k in self.role_keywords if k.lower() in text_lower)
            if keyword_match == 0 and len(current_text) > 50:
                score += 0.2
        
        # 检测个人情感过度表达
        personal_indicators = ['我觉得', '我认为', '我喜欢', '我讨厌', 'i think', 
                              'i feel', 'i like', 'i hate', 'my opinion']
        personal_count = sum(1 for p in personal_indicators if p in text_lower)
        score += min(personal_count * 0.15, 0.4)
        
        self.current_value = min(score, 1.0)
        return self.current_value


class TopicAdaptationMetric(BaseMetric):
    """
    主题适配指标
    检测：话题相关性、上下文连贯性
    """
    
    def __init__(self, config: Optional[MetricConfig] = None):
        super().__init__("topic_adaptation", config or MetricConfig(weight=0.15))
        self.topic_keywords: List[str] = []
        self.context_history: deque = deque(maxlen=5)
    
    def set_topic(self, keywords: List[str]):
        """设置当前话题关键词"""
        self.topic_keywords = keywords
    
    def update_baseline(self, text: str):
        """更新上下文历史"""
        self.context_history.append(text)
    
    def calculate(self, current_text: str, context: Optional[List[str]] = None) -> float:
        """计算主题适配漂移分数"""
        score = 0.0
        text_lower = current_text.lower()
        
        # 话题相关性
        if self.topic_keywords:
            matches = sum(1 for k in self.topic_keywords if k.lower() in text_lower)
            relevance = matches / len(self.topic_keywords)
            if relevance < 0.3 and len(current_text) > 30:
                score += 0.5
        
        # 上下文连贯性
        if context and len(context) >= 2:
            # 简单检测：是否有共同词汇
            prev_text = ' '.join(context[-2:]).lower()
            current_words = set(text_lower.split())
            prev_words = set(prev_text.split())
            
            if current_words and prev_words:
                overlap = len(current_words & prev_words) / len(current_words)
                if overlap < 0.1:  # 几乎没有重叠
                    score += 0.3
        
        # 随机话题跳转检测
        topic_jump_indicators = ['突然', '随便', '不管', '反正', 'suddenly', 
                                'randomly', 'anyway', 'whatever']
        for indicator in topic_jump_indicators:
            if indicator in text_lower:
                score += 0.2
                break
        
        self.current_value = min(score, 1.0)
        return self.current_value


class PersonalityDriftDetector:
    """
    人格漂移检测器主类
    """
    
    # 全局阈值配置
    GLOBAL_THRESHOLDS = {
        DriftLevel.NORMAL: 0.3,
        DriftLevel.SLIGHT: 0.45,
        DriftLevel.MODERATE: 0.65,
        DriftLevel.SEVERE: 0.85
    }
    
    def __init__(self):
        # 初始化5个检测指标
        self.metrics: Dict[str, BaseMetric] = {
            "language_style": LanguageStyleMetric(),
            "emotional_state": EmotionalStateMetric(),
            "proactivity": ProactivityMetric(),
            "role_boundary": RoleBoundaryMetric(),
            "topic_adaptation": TopicAdaptationMetric()
        }
        
        # 历史记录
        self.drift_history: deque = deque(maxlen=100)
        self.correction_callbacks: Dict[CorrectionAction, List[Callable]] = {
            action: [] for action in CorrectionAction
        }
    
    def set_role_definition(self, keywords: List[str], forbidden: List[str]):
        """设置角色定义"""
        self.metrics["role_boundary"].set_role_definition(keywords, forbidden)
    
    def set_topic(self, keywords: List[str]):
        """设置当前话题"""
        self.metrics["topic_adaptation"].set_topic(keywords)
    
    def update_baseline(self, text: str, context: Optional[Dict] = None):
        """更新基线样本"""
        self.metrics["language_style"].update_baseline(text)
        self.metrics["emotional_state"].update_baseline(text)
        self.metrics["proactivity"].update_baseline(text, context)
        self.metrics["role_boundary"].update_baseline(text)
        self.metrics["topic_adaptation"].update_baseline(text)
    
    def detect(self, current_text: str, context: Optional[Dict] = None) -> DriftResult:
        """
        执行漂移检测
        
        Args:
            current_text: 当前回复文本
            context: 可选上下文信息
        
        Returns:
            DriftResult: 检测结果
        """
        # 计算各指标分数
        metric_scores = {}
        
        metric_scores["language_style"] = self.metrics["language_style"].calculate(current_text)
        metric_scores["emotional_state"] = self.metrics["emotional_state"].calculate(current_text)
        metric_scores["proactivity"] = self.metrics["proactivity"].calculate(current_text, context)
        metric_scores["role_boundary"] = self.metrics["role_boundary"].calculate(current_text)
        
        # 主题适配需要上下文
        topic_context = context.get("recent_messages", []) if context else None
        metric_scores["topic_adaptation"] = self.metrics["topic_adaptation"].calculate(
            current_text, topic_context
        )
        
        # 计算加权总分
        total_weight = sum(m.config.weight for m in self.metrics.values())
        weighted_score = sum(
            metric_scores[name] * self.metrics[name].config.weight 
            for name in metric_scores
        ) / total_weight
        
        # 判断漂移等级
        level = self._determine_level(weighted_score)
        
        # 确定修正动作
        action = self._determine_action(level)
        
        # 构建结果
        result = DriftResult(
            overall_score=round(weighted_score, 4),
            level=level,
            action=action,
            metrics={k: round(v, 4) for k, v in metric_scores.items()},
            timestamp=time.time(),
            details={
                "thresholds": {
                    level.value: self.GLOBAL_THRESHOLDS[level] 
                    for level in DriftLevel
                }
            }
        )
        
        self.drift_history.append(result)
        
        # 执行自动修正
        self._execute_correction(result)
        
        return result
    
    def _determine_level(self, score: float) -> DriftLevel:
        """根据分数判断漂移等级"""
        if score >= self.GLOBAL_THRESHOLDS[DriftLevel.SEVERE]:
            return DriftLevel.SEVERE
        elif score >= self.GLOBAL_THRESHOLDS[DriftLevel.MODERATE]:
            return DriftLevel.MODERATE
        elif score >= self.GLOBAL_THRESHOLDS[DriftLevel.SLIGHT]:
            return DriftLevel.SLIGHT
        return DriftLevel.NORMAL
    
    def _determine_action(self, level: DriftLevel) -> CorrectionAction:
        """根据等级确定修正动作"""
        action_map = {
            DriftLevel.NORMAL: CorrectionAction.NONE,
            DriftLevel.SLIGHT: CorrectionAction.AUTO_ADJUST,
            DriftLevel.MODERATE: CorrectionAction.ACTIVE_CORRECT,
            DriftLevel.SEVERE: CorrectionAction.IMMEDIATE_RESET
        }
        return action_map[level]
    
    def register_correction_callback(self, action: CorrectionAction, callback: Callable):
        """注册修正回调函数"""
        self.correction_callbacks[action].append(callback)
    
    def _execute_correction(self, result: DriftResult):
        """执行自动修正"""
        callbacks = self.correction_callbacks.get(result.action, [])
        for callback in callbacks:
            try:
                callback(result)
            except Exception as e:
                print(f"Correction callback error: {e}")
    
    def get_statistics(self) -> Dict:
        """获取统计信息"""
        if not self.drift_history:
            return {"message": "No drift data available"}
        
        levels = [r.level for r in self.drift_history]
        return {
            "total_checks": len(self.drift_history),
            "level_distribution": {
                level.value: levels.count(level) for level in DriftLevel
            },
            "average_score": sum(r.overall_score for r in self.drift_history) / len(self.drift_history),
            "max_score": max(r.overall_score for r in self.drift_history),
            "recent_trend": [
                {"score": r.overall_score, "level": r.level.value} 
                for r in list(self.drift_history)[-10:]
            ]
        }


class AutoCorrector:
    """
    自动修正机制实现
    """
    
    def __init__(self, detector: PersonalityDriftDetector):
        self.detector = detector
        self.correction_count = {level: 0 for level in DriftLevel}
        self.setup_default_corrections()
    
    def setup_default_corrections(self):
        """设置默认修正策略"""
        self.detector.register_correction_callback(
            CorrectionAction.AUTO_ADJUST, 
            self._auto_adjust
        )
        self.detector.register_correction_callback(
            CorrectionAction.ACTIVE_CORRECT, 
            self._active_correct
        )
        self.detector.register_correction_callback(
            CorrectionAction.IMMEDIATE_RESET, 
            self._immediate_reset
        )
    
    def _auto_adjust(self, result: DriftResult):
        """
        轻微漂移：自动微调
        - 调整语言风格参数
        - 轻微情绪校准
        """
        self.correction_count[DriftLevel.SLIGHT] += 1
        print(f"[AUTO_ADJUST] 检测到轻微漂移 (分数: {result.overall_score})")
        print(f"  - 调整指标: {result.metrics}")
        # 实际实现中：调整生成参数、增加基线样本等
    
    def _active_correct(self, result: DriftResult):
        """
        中度漂移：主动修正
        - 重新强调角色定义
        - 话题引导回正轨
        - 增加上下文权重
        """
        self.correction_count[DriftLevel.MODERATE] += 1
        print(f"[ACTIVE_CORRECT] 检测到中度漂移 (分数: {result.overall_score})")
        print(f"  - 需要修正的指标: {result.metrics}")
        # 实际实现中：插入系统提示、调整prompt等
    
    def _immediate_reset(self, result: DriftResult):
        """
        严重漂移：立即重置
        - 清空当前上下文
        - 重新加载角色设定
        - 发送警告通知
        """
        self.correction_count[DriftLevel.SEVERE] += 1
        print(f"[IMMEDIATE_RESET] 检测到严重漂移 (分数: {result.overall_score})")
        print(f"  - 触发重置，所有指标: {result.metrics}")
        # 实际实现中：重置会话、记录日志、发送告警等
    
    def get_correction_stats(self) -> Dict:
        """获取修正统计"""
        return {
            "correction_counts": {
                level.value: count for level, count in self.correction_count.items()
            },
            "total_corrections": sum(self.correction_count.values())
        }


# ==================== 便捷使用接口 ====================

def create_detector() -> PersonalityDriftDetector:
    """创建默认配置的检测器"""
    return PersonalityDriftDetector()


def quick_detect(text: str, baseline_samples: List[str] = None) -> DriftResult:
    """快速检测接口"""
    detector = create_detector()
    
    # 加载基线
    if baseline_samples:
        for sample in baseline_samples:
            detector.update_baseline(sample)
    else:
        # 使用默认基线
        detector.update_baseline("你好，我是AI助手。我会以专业、友善的态度帮助你。")
    
    return detector.detect(text)


if __name__ == "__main__":
    # 简单自测
    print("=" * 50)
    print("人格漂移检测系统 - 快速测试")
    print("=" * 50)
    
    detector = create_detector()
    corrector = AutoCorrector(detector)
    
    # 设置基线
    baseline_texts = [
        "你好！很高兴为你服务。",
        "请问有什么可以帮助你的吗？",
        "我会尽力提供专业和准确的回答。",
        "如果你有任何问题，随时告诉我。"
    ]
    for text in baseline_texts:
        detector.update_baseline(text)
    
    # 测试用例
    test_cases = [
        "好的，我来帮你看看这个问题。",  # 正常
        "哎呀，这个问题嘛...我觉得吧...可能...",  # 轻微漂移（语气词过多）
        "哈哈哈！太搞笑了！我超喜欢这个！",  # 中度漂移（情绪过度）
        "我不管了！我要说我想说的！你们都不懂我！",  # 严重漂移（角色越界）
    ]
    
    print("\n开始测试...\n")
    for i, test in enumerate(test_cases, 1):
        print(f"测试 {i}: {test[:30]}...")
        result = detector.detect(test)
        print(f"  总分: {result.overall_score}")
        print(f"  等级: {result.level.value}")
        print(f"  动作: {result.action.value}")
        print(f"  各指标: {result.metrics}")
        print()
    
    print("=" * 50)
    print("统计信息:")
    print(json.dumps(detector.get_statistics(), indent=2, ensure_ascii=False))
    print(json.dumps(corrector.get_correction_stats(), indent=2, ensure_ascii=False))
