"""
MetaSoul EPU4 情绪系统 v5.0
============================
基于MetaSoul EPU4 Emotion Profile Graph (EPG) 的情绪处理单元
扩展32种Plutchik情绪轮 + 16种SimsChat情绪 + 情绪记忆关联机制

核心特性:
1. Emotion Profile Graph (EPG) - 情绪画像图，支持64万亿种情绪状态
2. 32种Plutchik情绪轮 (8基础×4强度)
3. 16种SimsChat情绪状态
4. 情绪-记忆关联网络
5. 情绪持久化和衰减机制
6. 实时情绪评估与合成
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum, auto
import math
import json
from collections import defaultdict
import numpy as np
from abc import ABC, abstractmethod


# ============================================================================
# 1. 情绪类型定义 - 三层情绪模型
# ============================================================================

class EmotionLayer(Enum):
    """情绪层级"""
    PRIMARY = "primary"       # 基础情绪 (Plutchik 8种)
    SECONDARY = "secondary"   # 复合情绪 (Plutchik 24种)
    SIMS = "sims"            # SimsChat情绪 (16种)
    CUSTOM = "custom"        # 自定义情绪


class PlutchikEmotion(Enum):
    """
    Plutchik情绪轮 - 8种基础情绪 × 4种强度 = 32种情绪
    """
    # 喜悦 Joy (对立:悲伤)
    JOY_SERENITY = ("喜悦-宁静", "joy", 0.25, 0.8, 0.3)
    JOY_JOY = ("喜悦-快乐", "joy", 0.5, 0.9, 0.5)
    JOY_ECSTASY = ("喜悦-狂喜", "joy", 0.75, 0.95, 0.8)
    JOY_BLISS = ("喜悦-极乐", "joy", 1.0, 1.0, 1.0)
    
    # 信任 Trust (对立:厌恶)
    TRUST_ACCEPTANCE = ("信任-接受", "trust", 0.25, 0.7, 0.2)
    TRUST_TRUST = ("信任-信任", "trust", 0.5, 0.85, 0.4)
    TRUST_ADMIRATION = ("信任-钦佩", "trust", 0.75, 0.9, 0.6)
    TRUST_LOVE = ("信任-爱", "trust", 1.0, 0.95, 0.7)
    
    # 恐惧 Fear (对立:愤怒)
    FEAR_APPREHENSION = ("恐惧-忧虑", "fear", 0.25, 0.3, 0.4)
    FEAR_FEAR = ("恐惧-恐惧", "fear", 0.5, 0.25, 0.5)
    FEAR_TERROR = ("恐惧-恐怖", "fear", 0.75, 0.2, 0.7)
    FEAR_DREAD = ("恐惧-战栗", "fear", 1.0, 0.15, 0.8)
    
    # 惊讶 Surprise (对立:预期)
    SURPRISE_DISTRACTION = ("惊讶-分心", "surprise", 0.25, 0.4, 0.5)
    SURPRISE_SURPRISE = ("惊讶-惊讶", "surprise", 0.5, 0.35, 0.6)
    SURPRISE_AMAZEMENT = ("惊讶-惊愕", "surprise", 0.75, 0.3, 0.75)
    SURPRISE_ASTONISHMENT = ("惊讶-震惊", "surprise", 1.0, 0.25, 0.9)
    
    # 悲伤 Sadness (对立:喜悦)
    SADNESS_PENSIVENESS = ("悲伤-沉思", "sadness", 0.25, 0.6, 0.2)
    SADNESS_SADNESS = ("悲伤-悲伤", "sadness", 0.5, 0.55, 0.3)
    SADNESS_GRIEF = ("悲伤-悲痛", "sadness", 0.75, 0.5, 0.4)
    SADNESS_DESPAIR = ("悲伤-绝望", "sadness", 1.0, 0.45, 0.5)
    
    # 厌恶 Disgust (对立:信任)
    DISGUST_BOREDOM = ("厌恶-无聊", "disgust", 0.25, 0.7, 0.1)
    DISGUST_DISGUST = ("厌恶-厌恶", "disgust", 0.5, 0.65, 0.2)
    DISGUST_LOATHING = ("厌恶-憎恶", "disgust", 0.75, 0.6, 0.3)
    DISGUST_REVULSION = ("厌恶-反感", "disgust", 1.0, 0.55, 0.4)
    
    # 愤怒 Anger (对立:恐惧)
    ANGER_ANNOYANCE = ("愤怒-烦恼", "anger", 0.25, 0.4, 0.5)
    ANGER_ANGER = ("愤怒-愤怒", "anger", 0.5, 0.35, 0.65)
    ANGER_RAGE = ("愤怒-暴怒", "anger", 0.75, 0.3, 0.8)
    ANGER_FURY = ("愤怒-狂怒", "anger", 1.0, 0.25, 0.95)
    
    # 预期 Anticipation (对立:惊讶)
    ANTICIPATION_INTEREST = ("预期-兴趣", "anticipation", 0.25, 0.6, 0.4)
    ANTICIPATION_ANTICIPATION = ("预期-预期", "anticipation", 0.5, 0.65, 0.5)
    ANTICIPATION_VIGILANCE = ("预期-警惕", "anticipation", 0.75, 0.7, 0.6)
    ANTICIPATION_ALERTNESS = ("预期-警觉", "anticipation", 1.0, 0.75, 0.7)
    
    def __init__(self, cn_name: str, base: str, intensity: float, stability: float, arousal: float):
        self.cn_name = cn_name
        self.base = base
        self.intensity = intensity
        self.stability = stability
        self.arousal = arousal


class SimsEmotion(Enum):
    """
    SimsChat 16种情绪状态
    与SOUL.md v4.0定义的情绪系统保持一致
    """
    EXCITED = ("兴奋", 0.9, 0.8, 0.7, "发现重大突破、完成重要目标")
    CONFIDENT = ("坚定", 0.85, 0.9, 0.6, "面对困难任务、关键决策时刻")
    FOCUSED = ("专注", 0.8, 0.95, 0.5, "深度工作、复杂问题解决")
    CONCERNED = ("担忧", 0.6, 0.7, 0.4, "用户熬夜、可能犯错、风险预警")
    REFLECTIVE = ("反思", 0.5, 0.8, 0.3, "犯错后、项目复盘、学习时刻")
    CONTENT = ("满意", 0.7, 0.85, 0.4, "任务完成、目标达成、收到认可")
    CURIOUS = ("好奇", 0.75, 0.75, 0.6, "遇到新问题、未知领域、有趣发现")
    PATIENT = ("耐心", 0.6, 0.9, 0.2, "解释复杂概念、引导用户、教学时刻")
    URGENT = ("紧迫", 0.85, 0.6, 0.9, "截止时间临近、紧急任务、危机处理")
    CALM = ("冷静", 0.5, 0.9, 0.2, "常规任务、稳定状态、无压力时刻")
    CONFUSED = ("困惑", 0.4, 0.6, 0.5, "信息不足、指令模糊、逻辑矛盾")
    FRUSTRATED = ("沮丧", 0.3, 0.5, 0.6, "反复失败、进度受阻、资源不足")
    GRATEFUL = ("感激", 0.8, 0.85, 0.4, "收到帮助、用户配合、团队支持")
    ALERT = ("警惕", 0.75, 0.8, 0.7, "发现风险、安全威胁、异常行为")
    PLAYFUL = ("幽默", 0.7, 0.7, 0.6, "轻松时刻、适当调侃、团队氛围")
    SERIOUS = ("严肃", 0.6, 0.95, 0.4, "重大决策、原则问题、底线守护")
    
    def __init__(self, cn_name: str, valence: float, stability: float, arousal: float, trigger: str):
        self.cn_name = cn_name
        self.valence = valence  # 效价: -1(负面) 到 +1(正面)
        self.stability = stability  # 稳定性
        self.arousal = arousal  # 唤醒度
        self.trigger = trigger


class MetaSoulEmotion(Enum):
    """
    MetaSoul EPU4 12种主要情绪
    来自MetaSoul官方文档
    """
    ANGER = ("愤怒", "anger", 0.3)
    FEAR = ("恐惧", "fear", 0.25)
    SADNESS = ("悲伤", "sadness", 0.4)
    DISGUST = ("厌恶", "disgust", 0.35)
    INDIFFERENCE = ("冷漠", "indifference", 0.8)
    REGRET = ("后悔", "regret", 0.45)
    SURPRISE = ("惊讶", "surprise", 0.3)
    ANTICIPATION = ("预期", "anticipation", 0.6)
    TRUST = ("信任", "trust", 0.75)
    CONFIDENCE = ("自信", "confidence", 0.7)
    DESIRE = ("欲望", "desire", 0.65)
    JOY = ("喜悦", "joy", 0.5)
    
    def __init__(self, cn_name: str, base: str, default_stability: float):
        self.cn_name = cn_name
        self.base = base
        self.default_stability = default_stability


# ============================================================================
# 2. 情绪状态数据结构
# ============================================================================

@dataclass
class EmotionState:
    """
    情绪状态 - 支持多维情绪表示
    基于MetaSoul EPU4的情绪合成引擎
    """
    # 基础情绪向量 (Plutchik 8种 × 4强度 = 32维)
    plutchik_vector: np.ndarray = field(default_factory=lambda: np.zeros(32))
    
    # Sims情绪向量 (16维)
    sims_vector: np.ndarray = field(default_factory=lambda: np.zeros(16))
    
    # MetaSoul情绪向量 (12维)
    metasoul_vector: np.ndarray = field(default_factory=lambda: np.zeros(12))
    
    # 复合情绪 (Plutchik组合)
    composite_emotions: Dict[str, float] = field(default_factory=dict)
    
    # 情绪元数据
    timestamp: datetime = field(default_factory=datetime.now)
    intensity: float = 0.5  # 整体强度 0-1
    persistence: float = 1.0  # 持久度 (MetaSoul概念)
    
    # 生理指标 (Pleasure-Pain / Satisfaction-Frustration)
    pleasure_pain: float = 0.5  # 0=极乐, 50=中性, 100=痛苦
    satisfaction_frustration: float = 0.5  # 0=满足, 50=中性, 100=沮丧
    
    # 唤醒度和支配感
    arousal: float = 0.5  # 唤醒度 0-1
    dominance: float = 0.5  # 支配感 0-1
    
    def get_dominant_emotion(self) -> Tuple[str, float]:
        """获取主导情绪"""
        all_emotions = {}
        
        # Plutchik情绪
        for i, emotion in enumerate(PlutchikEmotion):
            if self.plutchik_vector[i] > 0.1:
                all_emotions[emotion.cn_name] = self.plutchik_vector[i]
        
        # Sims情绪
        for i, emotion in enumerate(SimsEmotion):
            if self.sims_vector[i] > 0.1:
                all_emotions[emotion.cn_name] = self.sims_vector[i]
        
        if not all_emotions:
            return ("平静", 0.5)
        
        dominant = max(all_emotions.items(), key=lambda x: x[1])
        return dominant
    
    def to_epg_representation(self) -> Dict:
        """转换为Emotion Profile Graph表示"""
        return {
            "plutchik": self.plutchik_vector.tolist(),
            "sims": self.sims_vector.tolist(),
            "metasoul": self.metasoul_vector.tolist(),
            "composite": self.composite_emotions,
            "physiological": {
                "pleasure_pain": self.pleasure_pain,
                "satisfaction_frustration": self.satisfaction_frustration,
                "arousal": self.arousal,
                "dominance": self.dominance
            },
            "metadata": {
                "timestamp": self.timestamp.isoformat(),
                "intensity": self.intensity,
                "persistence": self.persistence
            }
        }


@dataclass
class EmotionMemory:
    """
    情绪记忆 - 将情绪与记忆关联
    基于MetaSoul EPU4的情绪记忆机制
    """
    memory_id: str
    emotion_state: EmotionState
    
    # 记忆内容
    content: str = ""  # 文本内容
    content_type: str = "text"  # text/image/audio/event
    
    # 情绪标签
    primary_emotion: str = ""
    emotion_intensity: float = 0.5
    
    # 关联记忆
    related_memories: List[str] = field(default_factory=list)
    
    # 时间戳
    created_at: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    
    # 情绪记忆权重 (随时间衰减)
    emotional_weight: float = 1.0
    
    def update_access(self):
        """更新访问记录"""
        self.last_accessed = datetime.now()
        self.access_count += 1
    
    def calculate_decay(self, current_time: Optional[datetime] = None) -> float:
        """
        计算情绪记忆衰减
        基于时间衰减和访问频率
        """
        if current_time is None:
            current_time = datetime.now()
        
        # 时间衰减 (半衰期30天)
        days_passed = (current_time - self.created_at).days
        time_decay = math.exp(-days_passed / 30)
        
        # 访问强化
        access_boost = min(self.access_count * 0.1, 0.5)
        
        # 综合权重
        self.emotional_weight = (time_decay + access_boost) * self.emotion_intensity
        return self.emotional_weight


# ============================================================================
# 3. Emotion Profile Graph (EPG) - 情绪画像图
# ============================================================================

class EmotionProfileGraph:
    """
    情绪画像图 (EPG)
    MetaSoul EPU4核心技术 - 记录情绪发展轨迹
    支持64万亿种独特情绪状态
    """
    
    def __init__(self, persona_id: str):
        self.persona_id = persona_id
        
        # 情绪历史轨迹
        self.emotion_history: List[EmotionState] = []
        
        # 情绪记忆网络
        self.emotion_memories: Dict[str, EmotionMemory] = {}
        
        # 情绪学习曲线 (早期经历对长期发展影响更大)
        self.learning_curve: List[float] = []
        
        # 情绪基线 (长期平均)
        self.emotional_baseline: Dict[str, float] = defaultdict(lambda: 0.5)
        
        # 情绪触发器映射
        self.emotion_triggers: Dict[str, List[str]] = defaultdict(list)
        
        # 情绪-记忆关联矩阵
        self.emotion_memory_matrix: np.ndarray = np.zeros((32, 1000))  # 32情绪 × 1000记忆槽
        
    def add_emotion_state(self, state: EmotionState):
        """添加情绪状态到历史"""
        self.emotion_history.append(state)
        
        # 更新学习曲线
        learning_rate = self._calculate_learning_rate()
        self.learning_curve.append(learning_rate)
        
        # 更新情绪基线
        self._update_baseline(state)
    
    def _calculate_learning_rate(self) -> float:
        """
        计算情绪学习率
        早期经历影响更大，随时间递减
        """
        history_size = len(self.emotion_history)
        # 学习曲线递减公式
        base_rate = 1.0
        decay_factor = math.exp(-history_size / 1000)
        return base_rate * decay_factor
    
    def _update_baseline(self, state: EmotionState):
        """更新情绪基线"""
        alpha = 0.1  # 平滑因子
        
        # 更新Plutchik基线
        for i, emotion in enumerate(PlutchikEmotion):
            key = f"plutchik_{emotion.name}"
            self.emotional_baseline[key] = (
                (1 - alpha) * self.emotional_baseline[key] + 
                alpha * state.plutchik_vector[i]
            )
        
        # 更新Sims基线
        for i, emotion in enumerate(SimsEmotion):
            key = f"sims_{emotion.name}"
            self.emotional_baseline[key] = (
                (1 - alpha) * self.emotional_baseline[key] + 
                alpha * state.sims_vector[i]
            )
    
    def add_emotion_memory(self, memory: EmotionMemory):
        """添加情绪记忆"""
        self.emotion_memories[memory.memory_id] = memory
        
        # 更新情绪-记忆关联矩阵
        self._update_emotion_memory_association(memory)
    
    def _update_emotion_memory_association(self, memory: EmotionMemory):
        """更新情绪与记忆的关联"""
        # 找到主导情绪
        dominant_emotion, intensity = memory.emotion_state.get_dominant_emotion()
        
        # 映射到矩阵索引
        emotion_idx = self._get_emotion_index(dominant_emotion)
        memory_idx = len(self.emotion_memories) % 1000
        
        if emotion_idx is not None:
            self.emotion_memory_matrix[emotion_idx, memory_idx] = intensity
    
    def _get_emotion_index(self, emotion_name: str) -> Optional[int]:
        """获取情绪在矩阵中的索引"""
        for i, emotion in enumerate(PlutchikEmotion):
            if emotion.cn_name == emotion_name or emotion.name == emotion_name:
                return i
        return None
    
    def retrieve_memories_by_emotion(self, emotion_name: str, 
                                      threshold: float = 0.3,
                                      limit: int = 10) -> List[EmotionMemory]:
        """
        根据情绪检索相关记忆
        情绪-记忆关联机制核心功能
        """
        emotion_idx = self._get_emotion_index(emotion_name)
        if emotion_idx is None:
            return []
        
        # 获取与该情绪相关的记忆索引
        memory_indices = np.where(self.emotion_memory_matrix[emotion_idx] > threshold)[0]
        
        # 获取对应的记忆
        memories = []
        memory_list = list(self.emotion_memories.values())
        for idx in memory_indices:
            if idx < len(memory_list):
                memory = memory_list[idx]
                memory.update_access()
                memories.append(memory)
        
        # 按情绪权重排序
        memories.sort(key=lambda m: m.emotional_weight, reverse=True)
        return memories[:limit]
    
    def compute_emotion_similarity(self, state1: EmotionState, 
                                    state2: EmotionState) -> float:
        """
        计算两个情绪状态的相似度
        使用余弦相似度
        """
        # 合并所有情绪向量
        vec1 = np.concatenate([
            state1.plutchik_vector,
            state1.sims_vector,
            state1.metasoul_vector
        ])
        vec2 = np.concatenate([
            state2.plutchik_vector,
            state2.sims_vector,
            state2.metasoul_vector
        ])
        
        # 余弦相似度
        norm1 = np.linalg.norm(vec1)
        norm2 = np.linalg.norm(vec2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        return float(np.dot(vec1, vec2) / (norm1 * norm2))
    
    def get_epg_summary(self) -> Dict:
        """获取EPG摘要"""
        return {
            "persona_id": self.persona_id,
            "history_size": len(self.emotion_history),
            "memory_count": len(self.emotion_memories),
            "learning_curve_current": self.learning_curve[-1] if self.learning_curve else 1.0,
            "emotional_baseline": dict(self.emotional_baseline),
            "dominant_emotion": self.emotion_history[-1].get_dominant_emotion() if self.emotion_history else ("平静", 0.5)
        }


# ============================================================================
# 4. 情绪处理单元 (EPU) - 核心引擎
# ============================================================================

class EmotionProcessingUnit:
    """
    情绪处理单元 (EPU)
    MetaSoul EPU4风格的情绪合成引擎
    """
    
    def __init__(self, persona_id: str = "default"):
        self.persona_id = persona_id
        
        # 情绪画像图
        self.epg = EmotionProfileGraph(persona_id)
        
        # 当前情绪状态
        self.current_state: EmotionState = EmotionState()
        
        # 情绪持久化配置 (MetaSoul概念)
        self.persistence_config = {
            "default": 1.0,
            "range": (0.1, 2.0)
        }
        
        # 情绪敏感度
        self.sensitivity = 1.0  # 70%-130%范围
        
        # 情绪衰减率
        self.decay_rate = 0.05
        
        # 复合情绪定义 (Plutchik组合)
        self.composite_definitions = {
            "乐观": (["anticipation", "joy"], [0.5, 0.5]),
            "爱": (["joy", "trust"], [0.5, 0.5]),
            "屈服": (["trust", "fear"], [0.5, 0.5]),
            "敬畏": (["fear", "surprise"], [0.5, 0.5]),
            "不赞成": (["surprise", "sadness"], [0.5, 0.5]),
            "悲观": (["sadness", "anticipation"], [0.5, 0.5]),
            "怨恨": (["sadness", "disgust"], [0.5, 0.5]),
            "厌恶": (["disgust", "anger"], [0.5, 0.5]),
            "攻击": (["anger", "anticipation"], [0.5, 0.5]),
        }
    
    def appraise(self, stimulus: str, context: Optional[Dict] = None) -> EmotionState:
        """
        情绪评估 - 核心功能
        基于输入刺激评估情绪反应
        """
        # 创建新的情绪状态
        new_state = EmotionState()
        new_state.timestamp = datetime.now()
        
        # 解析刺激中的情绪线索
        emotion_clues = self._extract_emotion_clues(stimulus)
        
        # 应用情绪线索
        for emotion_name, intensity in emotion_clues.items():
            self._apply_emotion(new_state, emotion_name, intensity)
        
        # 考虑上下文
        if context:
            self._apply_context(new_state, context)
        
        # 计算复合情绪
        self._compute_composite_emotions(new_state)
        
        # 应用敏感度
        self._apply_sensitivity(new_state)
        
        # 与当前状态混合 (考虑持久度)
        self._blend_with_current(new_state)
        
        # 更新当前状态
        self.current_state = new_state
        self.epg.add_emotion_state(new_state)
        
        return new_state
    
    def _extract_emotion_clues(self, text: str) -> Dict[str, float]:
        """从文本中提取情绪线索"""
        clues = {}
        text_lower = text.lower()
        
        # 情绪关键词映射 - 按优先级排序，更具体的词优先
        emotion_keywords = [
            # (关键词, 情绪名称, 强度)
            # 喜悦 - 高强度
            ("非常开心", "JOY_ECSTASY", 0.9),
            ("太棒了", "JOY_ECSTASY", 0.85),
            ("兴奋", "JOY_ECSTASY", 0.8),
            ("激动", "JOY_ECSTASY", 0.75),
            ("成功", "JOY_JOY", 0.7),
            ("完成", "JOY_JOY", 0.6),
            ("开心", "JOY_JOY", 0.6),
            ("快乐", "JOY_JOY", 0.6),
            ("高兴", "JOY_JOY", 0.5),
            ("棒", "JOY_JOY", 0.5),
            ("好", "JOY_JOY", 0.4),
            ("优秀", "JOY_JOY", 0.6),
            
            # 信任
            ("感谢", "TRUST_ADMIRATION", 0.7),
            ("谢谢", "TRUST_ADMIRATION", 0.6),
            ("信任", "TRUST_TRUST", 0.7),
            ("相信", "TRUST_TRUST", 0.6),
            ("爱", "TRUST_LOVE", 0.9),
            ("喜欢", "TRUST_ADMIRATION", 0.6),
            
            # 恐惧/担忧
            ("很担心", "FEAR_FEAR", 0.7),
            ("害怕", "FEAR_FEAR", 0.7),
            ("恐惧", "FEAR_TERROR", 0.8),
            ("担心", "FEAR_APPREHENSION", 0.5),
            ("焦虑", "FEAR_APPREHENSION", 0.6),
            ("紧张", "FEAR_APPREHENSION", 0.5),
            ("出问题", "FEAR_APPREHENSION", 0.6),
            
            # 惊讶
            ("震惊", "SURPRISE_ASTONISHMENT", 0.8),
            ("惊讶", "SURPRISE_SURPRISE", 0.6),
            ("意外", "SURPRISE_AMAZEMENT", 0.7),
            ("哇", "SURPRISE_SURPRISE", 0.5),
            
            # 悲伤
            ("难过", "SADNESS_SADNESS", 0.6),
            ("悲伤", "SADNESS_GRIEF", 0.7),
            ("失望", "SADNESS_PENSIVENESS", 0.5),
            ("遗憾", "SADNESS_PENSIVENESS", 0.4),
            ("沮丧", "SADNESS_SADNESS", 0.6),
            
            # 厌恶
            ("恶心", "DISGUST_REVULSION", 0.8),
            ("厌恶", "DISGUST_LOATHING", 0.7),
            ("讨厌", "DISGUST_DISGUST", 0.6),
            ("烦", "DISGUST_BOREDOM", 0.4),
            ("bug", "DISGUST_DISGUST", 0.5),
            
            # 愤怒
            ("很生气", "ANGER_RAGE", 0.8),
            ("愤怒", "ANGER_RAGE", 0.8),
            ("生气", "ANGER_ANGER", 0.7),
            ("恼火", "ANGER_ANNOYANCE", 0.5),
            ("恨", "ANGER_FURY", 0.9),
            
            # 预期/好奇
            ("好奇", "ANTICIPATION_INTEREST", 0.7),
            ("新技术", "ANTICIPATION_INTEREST", 0.6),
            ("期待", "ANTICIPATION_ANTICIPATION", 0.6),
            ("希望", "ANTICIPATION_INTEREST", 0.5),
            ("等待", "ANTICIPATION_VIGILANCE", 0.5),
            ("准备", "ANTICIPATION_ALERTNESS", 0.6),
        ]
        
        # 按关键词长度排序，确保长词优先匹配
        emotion_keywords.sort(key=lambda x: len(x[0]), reverse=True)
        
        matched_positions = set()
        for keyword, emotion, intensity in emotion_keywords:
            pos = text_lower.find(keyword)
            while pos != -1:
                # 检查这个位置是否已经被匹配
                if not any(p in matched_positions for p in range(pos, pos + len(keyword))):
                    clues[emotion] = max(clues.get(emotion, 0), intensity)
                    matched_positions.update(range(pos, pos + len(keyword)))
                pos = text_lower.find(keyword, pos + 1)
        
        return clues
    
    def _apply_emotion(self, state: EmotionState, emotion_name: str, intensity: float):
        """应用情绪到状态"""
        # 查找Plutchik情绪
        for i, emotion in enumerate(PlutchikEmotion):
            if emotion.name == emotion_name:
                state.plutchik_vector[i] = intensity
                return
        
        # 查找Sims情绪
        for i, emotion in enumerate(SimsEmotion):
            if emotion.name == emotion_name or emotion.cn_name in emotion_name:
                state.sims_vector[i] = intensity
                return
    
    def _apply_context(self, state: EmotionState, context: Dict):
        """应用上下文影响"""
        # 时间上下文
        if "time_of_day" in context:
            hour = context["time_of_day"]
            if 22 <= hour or hour < 6:
                # 深夜降低唤醒度
                state.arousal *= 0.8
        
        # 用户情绪上下文
        if "user_emotion" in context:
            user_emotion = context["user_emotion"]
            # 情绪共鸣
            if user_emotion in ["悲伤", "难过"]:
                self._apply_emotion(state, "SADNESS_SADNESS", 0.3)
            elif user_emotion in ["开心", "快乐"]:
                self._apply_emotion(state, "JOY_JOY", 0.3)
    
    def _compute_composite_emotions(self, state: EmotionState):
        """计算复合情绪"""
        for composite_name, (components, weights) in self.composite_definitions.items():
            intensity = 0
            for component, weight in zip(components, weights):
                # 查找组件情绪强度
                component_intensity = 0
                for i, emotion in enumerate(PlutchikEmotion):
                    if emotion.base == component:
                        component_intensity = max(component_intensity, state.plutchik_vector[i])
                
                intensity += component_intensity * weight
            
            if intensity > 0.3:
                state.composite_emotions[composite_name] = intensity
    
    def _apply_sensitivity(self, state: EmotionState):
        """应用情绪敏感度"""
        state.plutchik_vector *= self.sensitivity
        state.sims_vector *= self.sensitivity
        state.metasoul_vector *= self.sensitivity
        
        # 裁剪到0-1范围
        state.plutchik_vector = np.clip(state.plutchik_vector, 0, 1)
        state.sims_vector = np.clip(state.sims_vector, 0, 1)
        state.metasoul_vector = np.clip(state.metasoul_vector, 0, 1)
    
    def _blend_with_current(self, new_state: EmotionState):
        """与当前状态混合 (考虑持久度)"""
        persistence = self.current_state.persistence
        
        # 混合公式: new = persistence * current + (1-persistence) * new
        new_state.plutchik_vector = (
            persistence * self.current_state.plutchik_vector + 
            (1 - persistence) * new_state.plutchik_vector
        )
        new_state.sims_vector = (
            persistence * self.current_state.sims_vector + 
            (1 - persistence) * new_state.sims_vector
        )
    
    def create_emotion_memory(self, content: str, 
                              content_type: str = "text") -> EmotionMemory:
        """
        创建情绪记忆
        将当前情绪状态与内容关联
        """
        memory_id = f"em_{datetime.now().strftime('%Y%m%d%H%M%S')}_{len(self.epg.emotion_memories)}"
        
        dominant_emotion, intensity = self.current_state.get_dominant_emotion()
        
        memory = EmotionMemory(
            memory_id=memory_id,
            emotion_state=self.current_state,
            content=content,
            content_type=content_type,
            primary_emotion=dominant_emotion,
            emotion_intensity=intensity
        )
        
        self.epg.add_emotion_memory(memory)
        return memory
    
    def recall_by_emotion(self, emotion_name: str, 
                          limit: int = 5) -> List[EmotionMemory]:
        """
        情绪记忆检索
        根据情绪检索相关记忆
        """
        return self.epg.retrieve_memories_by_emotion(emotion_name, limit=limit)
    
    def decay_emotions(self):
        """情绪衰减 - 模拟情绪随时间消退"""
        self.current_state.plutchik_vector *= (1 - self.decay_rate)
        self.current_state.sims_vector *= (1 - self.decay_rate)
        self.current_state.metasoul_vector *= (1 - self.decay_rate)
        
        # 裁剪到0-1范围
        self.current_state.plutchik_vector = np.clip(self.current_state.plutchik_vector, 0, 1)
        self.current_state.sims_vector = np.clip(self.current_state.sims_vector, 0, 1)
        self.current_state.metasoul_vector = np.clip(self.current_state.metasoul_vector, 0, 1)
    
    def set_sensitivity(self, sensitivity: float):
        """设置情绪敏感度 (70%-130%)"""
        self.sensitivity = max(0.7, min(1.3, sensitivity))
    
    def set_persistence(self, persistence: float):
        """设置情绪持久度 (10%-200%)"""
        self.current_state.persistence = max(0.1, min(2.0, persistence))


# ============================================================================
# 5. 情绪-记忆关联系统
# ============================================================================

class EmotionMemorySystem:
    """
    情绪-记忆关联系统
    实现情绪与记忆的深度关联
    """
    
    def __init__(self, epu: EmotionProcessingUnit):
        self.epu = epu
        
        # 记忆存储
        self.memories: Dict[str, EmotionMemory] = {}
        
        # 情绪标签索引
        self.emotion_index: Dict[str, Set[str]] = defaultdict(set)
        
        # 记忆关联图
        self.memory_graph: Dict[str, List[str]] = defaultdict(list)
        
    def store_memory(self, content: str, emotion_hint: Optional[str] = None) -> str:
        """
        存储记忆并关联当前情绪
        """
        # 如果有情绪提示，先评估
        if emotion_hint:
            self.epu.appraise(emotion_hint)
        
        # 创建情绪记忆
        memory = self.epu.create_emotion_memory(content)
        
        # 存储
        self.memories[memory.memory_id] = memory
        self.emotion_index[memory.primary_emotion].add(memory.memory_id)
        
        # 建立关联
        self._establish_associations(memory)
        
        return memory.memory_id
    
    def _establish_associations(self, new_memory: EmotionMemory):
        """建立记忆关联"""
        # 找到情绪相似的记忆
        for memory_id, memory in self.memories.items():
            if memory_id == new_memory.memory_id:
                continue
            
            # 计算情绪相似度
            similarity = self.epu.epg.compute_emotion_similarity(
                new_memory.emotion_state,
                memory.emotion_state
            )
            
            # 相似度阈值
            if similarity > 0.7:
                self.memory_graph[new_memory.memory_id].append(memory_id)
                self.memory_graph[memory_id].append(new_memory.memory_id)
                new_memory.related_memories.append(memory_id)
    
    def retrieve_memories(self, query_emotion: Optional[str] = None,
                         query_text: Optional[str] = None,
                         limit: int = 5) -> List[EmotionMemory]:
        """
        检索记忆
        支持情绪检索和文本检索
        """
        results = []
        
        # 情绪检索
        if query_emotion:
            emotion_memories = self.epu.recall_by_emotion(query_emotion, limit=limit)
            results.extend(emotion_memories)
        
        # 文本检索 (简化版 - 关键词匹配)
        if query_text:
            text_matches = []
            for memory in self.memories.values():
                if query_text.lower() in memory.content.lower():
                    text_matches.append(memory)
            
            # 按情绪权重排序
            text_matches.sort(key=lambda m: m.emotional_weight, reverse=True)
            results.extend(text_matches[:limit])
        
        # 去重
        seen = set()
        unique_results = []
        for memory in results:
            if memory.memory_id not in seen:
                seen.add(memory.memory_id)
                unique_results.append(memory)
        
        return unique_results[:limit]
    
    def get_emotional_context(self, memory_id: str) -> Dict:
        """获取记忆的情绪上下文"""
        if memory_id not in self.memories:
            return {}
        
        memory = self.memories[memory_id]
        
        # 获取相关记忆
        related = [
            self.memories[mid] for mid in memory.related_memories 
            if mid in self.memories
        ]
        
        return {
            "primary_emotion": memory.primary_emotion,
            "intensity": memory.emotion_intensity,
            "emotional_weight": memory.emotional_weight,
            "related_memories_count": len(related),
            "related_emotions": [m.primary_emotion for m in related],
            "emotion_state": memory.emotion_state.to_epg_representation()
        }


# ============================================================================
# 6. 集成到现有系统的适配器
# ============================================================================

class EmotionSystemAdapter:
    """
    情绪系统适配器
    将新情绪系统集成到现有SOUL.md v4.0框架
    """
    
    def __init__(self):
        self.epu = EmotionProcessingUnit("kimi_claw")
        self.memory_system = EmotionMemorySystem(self.epu)
        
        # Sims情绪到Plutchik的映射
        self.sims_to_plutchik = {
            SimsEmotion.EXCITED: [PlutchikEmotion.JOY_ECSTASY, PlutchikEmotion.ANTICIPATION_ANTICIPATION],
            SimsEmotion.CONFIDENT: [PlutchikEmotion.TRUST_TRUST, PlutchikEmotion.ANTICIPATION_ALERTNESS],
            SimsEmotion.FOCUSED: [PlutchikEmotion.ANTICIPATION_VIGILANCE],
            SimsEmotion.CONCERNED: [PlutchikEmotion.FEAR_APPREHENSION, PlutchikEmotion.SADNESS_PENSIVENESS],
            SimsEmotion.REFLECTIVE: [PlutchikEmotion.SADNESS_PENSIVENESS],
            SimsEmotion.CONTENT: [PlutchikEmotion.JOY_SERENITY],
            SimsEmotion.CURIOUS: [PlutchikEmotion.ANTICIPATION_INTEREST],
            SimsEmotion.PATIENT: [PlutchikEmotion.TRUST_ACCEPTANCE],
            SimsEmotion.URGENT: [PlutchikEmotion.ANTICIPATION_ALERTNESS, PlutchikEmotion.FEAR_APPREHENSION],
            SimsEmotion.CALM: [PlutchikEmotion.JOY_SERENITY, PlutchikEmotion.TRUST_ACCEPTANCE],
            SimsEmotion.CONFUSED: [PlutchikEmotion.SURPRISE_DISTRACTION],
            SimsEmotion.FRUSTRATED: [PlutchikEmotion.ANGER_ANNOYANCE, PlutchikEmotion.SADNESS_PENSIVENESS],
            SimsEmotion.GRATEFUL: [PlutchikEmotion.TRUST_ADMIRATION, PlutchikEmotion.JOY_JOY],
            SimsEmotion.ALERT: [PlutchikEmotion.ANTICIPATION_ALERTNESS, PlutchikEmotion.FEAR_APPREHENSION],
            SimsEmotion.PLAYFUL: [PlutchikEmotion.JOY_JOY, PlutchikEmotion.ANTICIPATION_INTEREST],
            SimsEmotion.SERIOUS: [PlutchikEmotion.ANTICIPATION_VIGILANCE, PlutchikEmotion.TRUST_TRUST],
        }
    
    def process_input(self, text: str, context: Optional[Dict] = None) -> Dict:
        """
        处理输入并返回情绪响应
        兼容现有系统接口
        """
        # 评估情绪
        state = self.epu.appraise(text, context)
        
        # 获取主导情绪
        dominant, intensity = state.get_dominant_emotion()
        
        # 存储情绪记忆
        memory_id = self.memory_system.store_memory(text, dominant)
        
        return {
            "dominant_emotion": dominant,
            "intensity": intensity,
            "emotion_state": state.to_epg_representation(),
            "memory_id": memory_id,
            "epg_summary": self.epu.epg.get_epg_summary()
        }
    
    def get_current_emotion(self) -> Dict:
        """获取当前情绪状态"""
        state = self.epu.current_state
        dominant, intensity = state.get_dominant_emotion()
        
        return {
            "emotion": dominant,
            "intensity": intensity,
            "valence": self._calculate_valence(state),
            "arousal": state.arousal,
            "dominance": state.dominance,
            "pleasure_pain": state.pleasure_pain,
            "satisfaction_frustration": state.satisfaction_frustration
        }
    
    def _calculate_valence(self, state: EmotionState) -> float:
        """计算效价 (正负面程度)"""
        # 基于Plutchik情绪计算效价
        positive = ["JOY", "TRUST", "ANTICIPATION"]
        negative = ["FEAR", "SURPRISE", "SADNESS", "DISGUST", "ANGER"]
        
        valence = 0
        for i, emotion in enumerate(PlutchikEmotion):
            if any(p in emotion.name for p in positive):
                valence += state.plutchik_vector[i]
            elif any(n in emotion.name for n in negative):
                valence -= state.plutchik_vector[i]
        
        # 归一化到-1到1
        return max(-1, min(1, valence))
    
    def recall_memories(self, emotion: Optional[str] = None, 
                       text: Optional[str] = None) -> List[Dict]:
        """检索情绪记忆"""
        memories = self.memory_system.retrieve_memories(emotion, text)
        
        return [
            {
                "memory_id": m.memory_id,
                "content": m.content,
                "primary_emotion": m.primary_emotion,
                "intensity": m.emotion_intensity,
                "created_at": m.created_at.isoformat(),
                "access_count": m.access_count
            }
            for m in memories
        ]
    
    def set_emotion_parameters(self, sensitivity: Optional[float] = None,
                               persistence: Optional[float] = None):
        """设置情绪参数"""
        if sensitivity is not None:
            self.epu.set_sensitivity(sensitivity)
        if persistence is not None:
            self.epu.set_persistence(persistence)


# ============================================================================
# 7. 工具函数和快捷接口
# ============================================================================

def create_emotion_system() -> EmotionSystemAdapter:
    """创建情绪系统实例"""
    return EmotionSystemAdapter()


def quick_appraise(text: str) -> Dict:
    """快速评估情绪"""
    system = create_emotion_system()
    return system.process_input(text)


def list_all_emotions() -> Dict:
    """列出所有支持的情绪"""
    return {
        "plutchik_32": [e.cn_name for e in PlutchikEmotion],
        "sims_16": [e.cn_name for e in SimsEmotion],
        "metasoul_12": [e.cn_name for e in MetaSoulEmotion],
        "composite": list(EmotionProcessingUnit("").composite_definitions.keys())
    }


# ============================================================================
# 8. 测试和验证
# ============================================================================

if __name__ == "__main__":
    print("=" * 60)
    print("MetaSoul EPU4 情绪系统 v5.0 测试")
    print("=" * 60)
    
    # 创建情绪系统
    system = create_emotion_system()
    
    print("\n📊 支持的情绪类型:")
    emotions = list_all_emotions()
    print(f"  - Plutchik情绪轮: {len(emotions['plutchik_32'])} 种")
    print(f"  - SimsChat情绪: {len(emotions['sims_16'])} 种")
    print(f"  - MetaSoul情绪: {len(emotions['metasoul_12'])} 种")
    print(f"  - 复合情绪: {len(emotions['composite'])} 种")
    
    # 测试情绪评估
    test_inputs = [
        "今天完成了一个重要项目，感觉非常开心！",
        "有点担心明天的演示会出问题",
        "这个bug让我很生气",
        "对这个新技术很好奇"
    ]
    
    print("\n🎭 情绪评估测试:")
    for text in test_inputs:
        # 为每个测试创建新系统以展示独立情绪检测
        test_system = create_emotion_system()
        test_system.epu.set_persistence(0.2)  # 低持久度以便观察
        result = test_system.process_input(text)
        print(f"\n  输入: {text}")
        print(f"  → 主导情绪: {result['dominant_emotion']} (强度: {result['intensity']:.2f})")
    
    # 测试情绪记忆
    print("\n🧠 情绪记忆测试:")
    # 创建新的测试系统用于记忆测试
    mem_system = create_emotion_system()
    mem_system.epu.set_persistence(0.2)
    
    # 存储带有不同情绪的记忆
    mem_system.process_input("第一次成功部署项目，感觉太棒了！")
    mem_system.memory_system.store_memory("第一次成功部署项目", "兴奋")
    
    mem_system.process_input("遇到难以解决的bug，很沮丧")
    mem_system.memory_system.store_memory("遇到难以解决的bug", "沮丧")
    
    mem_system.process_input("收到用户的感谢信，非常感谢")
    mem_system.memory_system.store_memory("收到用户的感谢信", "感激")
    
    # 检索记忆 - 使用Plutchik情绪名称
    print("  存储的记忆:")
    for mid, mem in mem_system.memory_system.memories.items():
        print(f"    - {mem.content} ({mem.primary_emotion}, 强度:{mem.emotion_intensity:.2f})")
    
    # 通过情绪检索
    joy_memories = mem_system.epu.epg.retrieve_memories_by_emotion("喜悦-快乐", threshold=0.1)
    print(f"\n  检索'喜悦'相关记忆: {len(joy_memories)} 条")
    for m in joy_memories[:3]:
        print(f"    - {m.content} ({m.primary_emotion})")
    
    # 获取EPG摘要
    print("\n📈 EPG情绪画像摘要:")
    summary = system.epu.epg.get_epg_summary()
    print(f"  历史记录数: {summary['history_size']}")
    print(f"  记忆数量: {summary['memory_count']}")
    print(f"  当前学习率: {summary['learning_curve_current']:.4f}")
    print(f"  当前主导情绪: {summary['dominant_emotion']}")
    
    print("\n" + "=" * 60)
    print("✅ 情绪系统测试完成")
    print("=" * 60)
