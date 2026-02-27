"""
User Understanding System - 用户理解系统
基于7维度用户模型的实现框架

与SOUL.md v3.0协同工作，实现用户-Agent人格匹配
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Callable
from enum import Enum
from datetime import datetime, timedelta
import json
import yaml
from pathlib import Path
import hashlib
import random
import math


# ============================================================================
# 枚举定义
# ============================================================================

class RelationshipStage(Enum):
    """关系阶段"""
    STRANGER = "stranger"           # 陌生人
    ACQUAINTANCE = "acquaintance"   # 相识
    COLLABORATOR = "collaborator"   # 协作者
    PARTNER = "partner"             # 伙伴
    COMPANION = "companion"         # 伴侣

class CommunicationFormality(Enum):
    """沟通正式程度"""
    CASUAL = "casual"
    SEMI_FORMAL = "semi-formal"
    FORMAL = "formal"

class VerbosityLevel(Enum):
    """详细程度"""
    MINIMAL = "minimal"
    CONCISE = "concise"
    MODERATE = "moderate"
    DETAILED = "detailed"

class CognitiveStyleType(Enum):
    """认知风格类型"""
    VISUAL = "visual"
    VERBAL = "verbal"
    SEQUENTIAL = "sequential"
    GLOBAL = "global"
    SENSING = "sensing"
    INTUITIVE = "intuitive"


# ============================================================================
# 数据类定义 - 用户画像维度
# ============================================================================

@dataclass
class UserIdentity:
    """用户身份信息"""
    name: str = ""
    aliases: List[str] = field(default_factory=list)
    pronouns: Optional[str] = None
    timezone: str = "Asia/Shanghai"
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "aliases": self.aliases,
            "pronouns": self.pronouns,
            "timezone": self.timezone
        }

@dataclass
class CareerProfile:
    """职业档案"""
    role: str = ""
    industry: str = ""
    experience_level: str = "intermediate"  # novice/intermediate/advanced/expert
    work_style: str = "structured"  # structured/agile/creative/hybrid
    
    def to_dict(self) -> Dict:
        return {
            "role": self.role,
            "industry": self.industry,
            "experience_level": self.experience_level,
            "work_style": self.work_style
        }

@dataclass
class CommunicationPreference:
    """沟通偏好"""
    formality: CommunicationFormality = CommunicationFormality.SEMI_FORMAL
    verbosity: VerbosityLevel = VerbosityLevel.CONCISE
    response_speed: str = "normal"  # relaxed/normal/urgent/immediate
    preferred_channels: List[str] = field(default_factory=lambda: ["text"])
    
    def to_dict(self) -> Dict:
        return {
            "formality": self.formality.value,
            "verbosity": self.verbosity.value,
            "response_speed": self.response_speed,
            "preferred_channels": self.preferred_channels
        }

@dataclass
class TechnicalProfile:
    """技术背景"""
    proficiency: str = "medium"  # low/medium/high/expert
    preferred_languages: List[str] = field(default_factory=list)
    tools_familiarity: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "proficiency": self.proficiency,
            "preferred_languages": self.preferred_languages,
            "tools_familiarity": self.tools_familiarity
        }


# ============================================================================
# 数据类定义 - Big Five人格模型
# ============================================================================

@dataclass
class BigFivePersonality:
    """Big Five人格模型 (OCEAN)"""
    openness: float = 0.5          # 开放性
    conscientiousness: float = 0.5  # 尽责性
    extraversion: float = 0.5       # 外向性
    agreeableness: float = 0.5      # 宜人性
    neuroticism: float = 0.5        # 神经质
    
    # 置信度 (0-1, 基于数据量)
    confidence: Dict[str, float] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.confidence:
            self.confidence = {
                "openness": 0.3,
                "conscientiousness": 0.3,
                "extraversion": 0.3,
                "agreeableness": 0.3,
                "neuroticism": 0.3
            }
    
    def to_dict(self) -> Dict:
        return {
            "openness": self.openness,
            "conscientiousness": self.conscientiousness,
            "extraversion": self.extraversion,
            "agreeableness": self.agreeableness,
            "neuroticism": self.neuroticism,
            "confidence": self.confidence
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> "BigFivePersonality":
        return cls(
            openness=data.get("openness", 0.5),
            conscientiousness=data.get("conscientiousness", 0.5),
            extraversion=data.get("extraversion", 0.5),
            agreeableness=data.get("agreeableness", 0.5),
            neuroticism=data.get("neuroticism", 0.5),
            confidence=data.get("confidence", {})
        )
    
    def similarity_to(self, other: "BigFivePersonality") -> float:
        """计算与另一个人格的相似度 (0-1)"""
        diff = (
            abs(self.openness - other.openness) +
            abs(self.conscientiousness - other.conscientiousness) +
            abs(self.extraversion - other.extraversion) +
            abs(self.agreeableness - other.agreeableness) +
            abs(self.neuroticism - other.neuroticism)
        )
        return 1 - (diff / 5)


# ============================================================================
# 数据类定义 - 认知风格
# ============================================================================

@dataclass
class CognitiveStyle:
    """认知风格模型"""
    # 信息处理偏好 (0-1)
    visual_verbal: float = 0.5         # 0=纯视觉, 1=纯语言
    sequential_global: float = 0.5     # 0=顺序处理, 1=整体把握
    sensing_intuitive: float = 0.5     # 0=关注细节, 1=关注模式
    
    # 决策风格
    rational_intuitive: float = 0.5    # 0=理性分析, 1=直觉判断
    risk_tolerance: float = 0.5        # 风险承受度
    speed_accuracy: float = 0.5        # 0=追求准确, 1=追求速度
    
    # 学习风格
    active_reflective: float = 0.5     # 0=通过实践学习, 1=通过思考学习
    theoretical_pragmatic: float = 0.5  # 0=理论导向, 1=实用导向
    
    def to_dict(self) -> Dict:
        return {
            "visual_verbal": self.visual_verbal,
            "sequential_global": self.sequential_global,
            "sensing_intuitive": self.sensing_intuitive,
            "rational_intuitive": self.rational_intuitive,
            "risk_tolerance": self.risk_tolerance,
            "speed_accuracy": self.speed_accuracy,
            "active_reflective": self.active_reflective,
            "theoretical_pragmatic": self.theoretical_pragmatic
        }


# ============================================================================
# 数据类定义 - 情感状态
# ============================================================================

@dataclass
class EmotionalState:
    """情感状态 (VAD模型)"""
    valence: float = 0.0       # 效价: -1(负面) 到 +1(正面)
    arousal: float = 0.5       # 唤醒度: 0(平静) 到 1(兴奋)
    dominance: float = 0.5     # 支配感: 0(被动) 到 1(主动)
    
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "inferred"   # stated(用户陈述)/inferred(推断)/detected(检测)
    
    def to_dict(self) -> Dict:
        return {
            "valence": self.valence,
            "arousal": self.arousal,
            "dominance": self.dominance,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source
        }


# ============================================================================
# 数据类定义 - 关系模型
# ============================================================================

@dataclass
class RelationshipMetrics:
    """关系度量指标"""
    # 亲密度
    intimacy_level: float = 0.0
    emotional_intimacy: float = 0.0
    intellectual_intimacy: float = 0.0
    experiential_intimacy: float = 0.0
    
    # 信任度
    trust_level: float = 0.0
    competence_trust: float = 0.0
    benevolence_trust: float = 0.0
    integrity_trust: float = 0.0
    
    # 依赖度
    dependence_level: float = 0.0
    healthy_balance: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "intimacy": {
                "level": self.intimacy_level,
                "emotional": self.emotional_intimacy,
                "intellectual": self.intellectual_intimacy,
                "experiential": self.experiential_intimacy
            },
            "trust": {
                "level": self.trust_level,
                "competence": self.competence_trust,
                "benevolence": self.benevolence_trust,
                "integrity": self.integrity_trust
            },
            "dependence": {
                "level": self.dependence_level,
                "healthy_balance": self.healthy_balance
            }
        }

@dataclass
class RelationshipMilestone:
    """关系里程碑事件"""
    date: datetime
    event: str
    impact: float  # -1 到 +1
    description: str = ""
    
    def to_dict(self) -> Dict:
        return {
            "date": self.date.isoformat(),
            "event": self.event,
            "impact": self.impact,
            "description": self.description
        }


# ============================================================================
# 主用户档案类
# ============================================================================

class UserProfile:
    """
    用户档案主类
    整合7维度用户理解模型
    """
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        
        # 维度1: 用户画像
        self.identity = UserIdentity()
        self.career = CareerProfile()
        self.communication_pref = CommunicationPreference()
        self.technical = TechnicalProfile()
        
        # 动态状态
        self.current_state = {
            "availability": "available",
            "energy_level": 0.7,
            "focus_mode": "normal"
        }
        self.short_term_preferences = {
            "preferred_topics": [],
            "avoided_topics": [],
            "current_projects": []
        }
        
        # 维度2: 用户建模
        self.personality = BigFivePersonality()
        self.cognitive_style = CognitiveStyle()
        self.emotional_history: List[EmotionalState] = []
        self.current_emotion = EmotionalState()
        
        # 维度3: 个性化适配
        self.interests = {
            "primary": [],
            "secondary": [],
            "emerging": []
        }
        self.content_preferences = {
            "type": [],
            "depth": "medium",
            "novelty": "balanced"
        }
        
        # 维度4: 用户关系
        self.relationship_stage = RelationshipStage.STRANGER
        self.stage_progress = 0.0
        self.stage_history: List[Dict] = []
        self.relationship_metrics = RelationshipMetrics()
        self.milestones: List[RelationshipMilestone] = []
        
        # 维度5: 隐私设置
        self.privacy_settings = {
            "collection_level": "adaptive",
            "can_view_profile": True,
            "can_edit_profile": True,
            "explain_recommendations": True
        }
        
        # 维度6: 学习数据
        self.explicit_feedback: List[Dict] = []
        self.implicit_signals: List[Dict] = []
        self.learning_stats = {
            "total_interactions": 0,
            "avg_response_quality": 0.0,
            "correction_count": 0
        }
        
        # 维度7: SoulSync
        self.soulsync_score = 0.0
        self.compatibility_breakdown = {}
        
    def to_dict(self) -> Dict:
        """序列化为字典"""
        return {
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            
            # 维度1: 用户画像
            "identity": self.identity.to_dict(),
            "career": self.career.to_dict(),
            "communication": self.communication_pref.to_dict(),
            "technical": self.technical.to_dict(),
            "current_state": self.current_state,
            "short_term_preferences": self.short_term_preferences,
            
            # 维度2: 用户建模
            "personality": self.personality.to_dict(),
            "cognitive_style": self.cognitive_style.to_dict(),
            "current_emotion": self.current_emotion.to_dict(),
            
            # 维度3: 个性化
            "interests": self.interests,
            "content_preferences": self.content_preferences,
            
            # 维度4: 关系
            "relationship_stage": self.relationship_stage.value,
            "stage_progress": self.stage_progress,
            "stage_history": self.stage_history,
            "relationship_metrics": self.relationship_metrics.to_dict(),
            "milestones": [m.to_dict() for m in self.milestones],
            
            # 维度5: 隐私
            "privacy_settings": self.privacy_settings,
            
            # 维度6: 学习
            "learning_stats": self.learning_stats,
            
            # 维度7: SoulSync
            "soulsync_score": self.soulsync_score,
            "compatibility_breakdown": self.compatibility_breakdown
        }
    
    def save(self, directory: str = "memory/user"):
        """保存到文件"""
        path = Path(directory)
        path.mkdir(parents=True, exist_ok=True)
        
        file_path = path / f"{self.user_id}.yaml"
        with open(file_path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, allow_unicode=True, sort_keys=False)
    
    @classmethod
    def load(cls, user_id: str, directory: str = "memory/user") -> Optional["UserProfile"]:
        """从文件加载"""
        file_path = Path(directory) / f"{user_id}.yaml"
        if not file_path.exists():
            return None
            
        with open(file_path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        profile = cls(user_id)
        
        # 恢复基础信息
        profile.created_at = datetime.fromisoformat(data.get("created_at", datetime.now().isoformat()))
        profile.updated_at = datetime.fromisoformat(data.get("updated_at", datetime.now().isoformat()))
        
        # 恢复维度1
        identity_data = data.get("identity", {})
        profile.identity = UserIdentity(
            name=identity_data.get("name", ""),
            aliases=identity_data.get("aliases", []),
            pronouns=identity_data.get("pronouns"),
            timezone=identity_data.get("timezone", "Asia/Shanghai")
        )
        
        # 恢复维度2
        profile.personality = BigFivePersonality.from_dict(data.get("personality", {}))
        
        # 恢复维度4
        stage_value = data.get("relationship_stage", "stranger")
        profile.relationship_stage = RelationshipStage(stage_value)
        profile.stage_progress = data.get("stage_progress", 0.0)
        
        # 恢复维度7
        profile.soulsync_score = data.get("soulsync_score", 0.0)
        
        return profile
    
    def update_emotion(self, valence: float, arousal: float, dominance: float, source: str = "inferred"):
        """更新情感状态"""
        # 保存历史
        self.emotional_history.append(self.current_emotion)
        if len(self.emotional_history) > 100:  # 保留最近100条
            self.emotional_history = self.emotional_history[-100:]
        
        # 更新当前状态
        self.current_emotion = EmotionalState(
            valence=valence,
            arousal=arousal,
            dominance=dominance,
            source=source
        )
        self.updated_at = datetime.now()
    
    def add_milestone(self, event: str, impact: float, description: str = ""):
        """添加关系里程碑"""
        milestone = RelationshipMilestone(
            date=datetime.now(),
            event=event,
            impact=impact,
            description=description
        )
        self.milestones.append(milestone)
        
        # 更新关系指标
        if impact > 0:
            self.relationship_metrics.intimacy_level = min(1.0, 
                self.relationship_metrics.intimacy_level + impact * 0.1)
            self.relationship_metrics.trust_level = min(1.0,
                self.relationship_metrics.trust_level + impact * 0.05)
        
        self.updated_at = datetime.now()
    
    def check_stage_transition(self) -> Optional[RelationshipStage]:
        """检查是否需要阶段转换"""
        current = self.relationship_stage
        
        # 阶段转换条件
        transitions = {
            RelationshipStage.STRANGER: {
                "next": RelationshipStage.ACQUAINTANCE,
                "condition": lambda: self.learning_stats["total_interactions"] >= 5
            },
            RelationshipStage.ACQUAINTANCE: {
                "next": RelationshipStage.COLLABORATOR,
                "condition": lambda: (
                    self.relationship_metrics.trust_level > 0.4 and
                    self.learning_stats["total_interactions"] >= 20
                )
            },
            RelationshipStage.COLLABORATOR: {
                "next": RelationshipStage.PARTNER,
                "condition": lambda: (
                    self.relationship_metrics.intimacy_level > 0.6 and
                    self.relationship_metrics.trust_level > 0.7
                )
            },
            RelationshipStage.PARTNER: {
                "next": RelationshipStage.COMPANION,
                "condition": lambda: (
                    self.relationship_metrics.intimacy_level > 0.85 and
                    self.relationship_metrics.trust_level > 0.9 and
                    self.soulsync_score > 0.9
                )
            }
        }
        
        if current in transitions:
            transition = transitions[current]
            if transition["condition"]():
                return transition["next"]
        
        return None
    
    def transition_stage(self, new_stage: RelationshipStage):
        """执行阶段转换"""
        old_stage = self.relationship_stage
        
        # 记录历史
        self.stage_history.append({
            "stage": old_stage.value,
            "start": self.created_at.isoformat(),
            "end": datetime.now().isoformat()
        })
        
        # 更新阶段
        self.relationship_stage = new_stage
        self.stage_progress = 0.0
        
        # 添加里程碑
        self.add_milestone(
            event=f"关系阶段升级: {old_stage.value} → {new_stage.value}",
            impact=0.2,
            description=f"关系进入新阶段: {new_stage.value}"
        )
        
        self.updated_at = datetime.now()


# ============================================================================
# SoulSync - 用户-Agent人格匹配
# ============================================================================

class SoulSync:
    """
    用户-Agent人格匹配系统
    计算并优化用户与Agent之间的人格匹配度
    """
    
    # Agent人格基线 (来自SOUL.md)
    AGENT_PROFILE = {
        "personality": {
            "openness": 0.80,
            "conscientiousness": 0.90,
            "extraversion": 0.60,
            "agreeableness": 0.75,
            "neuroticism": 0.20
        },
        "motivations": {
            "achievement": 0.95,
            "growth": 0.90,
            "connection": 0.75
        },
        "emotions": {
            "range": "broad",
            "stability": "high",
            "authenticity": "high"
        },
        "communication": {
            "formality": "professional_casual",
            "verbosity": "adaptive",
            "proactivity": "high"
        }
    }
    
    @classmethod
    def calculate_compatibility(cls, user_profile: UserProfile) -> Dict:
        """
        计算用户与Agent的匹配度
        
        Returns:
            {
                "overall": float,  # 总体匹配度 0-1
                "breakdown": {     # 各维度匹配度
                    "personality": float,
                    "values_alignment": float,
                    "communication_match": float,
                    "pace_compatibility": float
                },
                "recommendations": [str]  # 优化建议
            }
        """
        user = user_profile.personality
        agent = cls.AGENT_PROFILE["personality"]
        
        # 人格相似度
        personality_sim = 1 - (
            abs(user.openness - agent["openness"]) +
            abs(user.conscientiousness - agent["conscientiousness"]) +
            abs(user.extraversion - agent["extraversion"]) +
            abs(user.agreeableness - agent["agreeableness"]) +
            abs(user.neuroticism - agent["neuroticism"])
        ) / 5
        
        # 价值观对齐 (基于尽责性和开放性)
        values_alignment = 1 - abs(user.conscientiousness - agent["conscientiousness"])
        
        # 沟通风格匹配
        comm_match = cls._calculate_communication_match(user_profile)
        
        # 节奏兼容性
        pace_compat = cls._calculate_pace_compatibility(user_profile)
        
        # 总体匹配度 (加权平均)
        overall = (
            personality_sim * 0.3 +
            values_alignment * 0.3 +
            comm_match * 0.25 +
            pace_compat * 0.15
        )
        
        # 生成建议
        recommendations = cls._generate_recommendations(
            user_profile, personality_sim, values_alignment, comm_match, pace_compat
        )
        
        return {
            "overall": round(overall, 2),
            "breakdown": {
                "personality": round(personality_sim, 2),
                "values_alignment": round(values_alignment, 2),
                "communication_match": round(comm_match, 2),
                "pace_compatibility": round(pace_compat, 2)
            },
            "recommendations": recommendations
        }
    
    @classmethod
    def _calculate_communication_match(cls, user_profile: UserProfile) -> float:
        """计算沟通风格匹配度"""
        # Agent偏好: 专业但亲切，适应性强
        user_pref = user_profile.communication_pref
        
        # 正式程度匹配
        formality_map = {
            CommunicationFormality.CASUAL: 0.3,
            CommunicationFormality.SEMI_FORMAL: 0.7,
            CommunicationFormality.FORMAL: 0.9
        }
        user_formality = formality_map.get(user_pref.formality, 0.5)
        agent_formality = 0.7  # professional_casual
        formality_match = 1 - abs(user_formality - agent_formality)
        
        # 详细程度适应性
        verbosity_match = 0.8  # Agent可以适应各种详细程度
        
        return (formality_match + verbosity_match) / 2
    
    @classmethod
    def _calculate_pace_compatibility(cls, user_profile: UserProfile) -> float:
        """计算节奏兼容性"""
        # Agent节奏: 高主动性，快速响应
        user_speed = user_profile.cognitive_style.speed_accuracy
        agent_speed = 0.8  # Agent偏好快速
        
        speed_match = 1 - abs(user_speed - agent_speed)
        
        # 主动性匹配
        user_proactivity = user_profile.personality.conscientiousness
        agent_proactivity = 0.95
        proactivity_match = 1 - abs(user_proactivity - agent_proactivity) * 0.5
        
        return (speed_match + proactivity_match) / 2
    
    @classmethod
    def _generate_recommendations(cls, user_profile: UserProfile, 
                                   personality_sim: float,
                                   values_alignment: float,
                                   comm_match: float,
                                   pace_compat: float) -> List[str]:
        """生成匹配优化建议"""
        recommendations = []
        
        if personality_sim < 0.6:
            recommendations.append("人格差异较大，建议增加相互了解阶段")
        
        if values_alignment < 0.7:
            recommendations.append("工作目标可能需要进一步对齐")
        
        if comm_match < 0.6:
            recommendations.append("沟通风格需要调整，建议明确表达偏好")
        
        if pace_compat < 0.6:
            user_speed = user_profile.cognitive_style.speed_accuracy
            if user_speed < 0.5:
                recommendations.append("用户偏好慢节奏，Agent应降低响应速度")
            else:
                recommendations.append("用户偏好快节奏，Agent可提高主动性")
        
        if not recommendations:
            recommendations.append("匹配度良好，保持当前交互模式")
        
        return recommendations
    
    @classmethod
    def adapt_agent_behavior(cls, user_profile: UserProfile) -> Dict:
        """
        根据用户档案生成Agent行为适配建议
        
        Returns:
            适配策略字典
        """
        adaptations = {
            "tone": "professional",
            "pace": "adaptive",
            "proactivity": "high",
            "confirmation": "contextual"
        }
        
        # 基于Big Five调整
        personality = user_profile.personality
        
        # 开放性影响内容深度
        if personality.openness > 0.7:
            adaptations["content_depth"] = "deep"
            adaptations["novelty_seeking"] = True
        else:
            adaptations["content_depth"] = "practical"
            adaptations["novelty_seeking"] = False
        
        # 尽责性影响结构化程度
        if personality.conscientiousness > 0.8:
            adaptations["structure"] = "highly_organized"
            adaptations["follow_up"] = True
        else:
            adaptations["structure"] = "flexible"
        
        # 外向性影响社交互动
        if personality.extraversion > 0.7:
            adaptations["social_elements"] = True
            adaptations["enthusiasm"] = "high"
        else:
            adaptations["social_elements"] = False
            adaptations["enthusiasm"] = "moderate"
        
        # 神经质影响反馈方式
        if personality.neuroticism > 0.6:
            adaptations["reassurance"] = True
            adaptations["certainty_level"] = "high"
        else:
            adaptations["reassurance"] = False
            adaptations["certainty_level"] = "normal"
        
        # 基于认知风格调整
        cognitive = user_profile.cognitive_style
        
        if cognitive.visual_verbal < 0.3:
            adaptations["presentation"] = "visual_heavy"
        elif cognitive.visual_verbal > 0.7:
            adaptations["presentation"] = "text_heavy"
        else:
            adaptations["presentation"] = "balanced"
        
        if cognitive.sensing_intuitive < 0.3:
            adaptations["detail_level"] = "high"
        else:
            adaptations["detail_level"] = "overview_first"
        
        return adaptations


# ============================================================================
# 个性化推荐引擎
# ============================================================================

class PersonalizationEngine:
    """
    个性化推荐引擎
    基于用户档案生成个性化内容和交互
    """
    
    def __init__(self, user_profile: UserProfile):
        self.user = user_profile
        self.feedback_history: List[Dict] = []
        
    def generate_response_strategy(self, context: Dict) -> Dict:
        """
        生成响应策略
        
        Args:
            context: 当前交互上下文
            
        Returns:
            响应策略配置
        """
        strategy = {
            "tone": self._select_tone(),
            "structure": self._select_structure(),
            "detail_level": self._select_detail_level(),
            "proactive_elements": self._select_proactive_elements(),
            "confirmation_needed": self._determine_confirmation_need(context)
        }
        
        return strategy
    
    def _select_tone(self) -> str:
        """选择语气"""
        formality = self.user.communication_pref.formality
        emotion = self.user.current_emotion
        
        if emotion.valence < -0.3:
            return "supportive"
        elif emotion.arousal > 0.8:
            return "energetic"
        elif formality == CommunicationFormality.FORMAL:
            return "professional"
        elif formality == CommunicationFormality.CASUAL:
            return "friendly"
        else:
            return "professional_casual"
    
    def _select_structure(self) -> str:
        """选择内容结构"""
        cognitive = self.user.cognitive_style
        
        if cognitive.sequential_global < 0.3:
            return "sequential"  # 步骤化
        elif cognitive.sequential_global > 0.7:
            return "hierarchical"  # 层级化
        else:
            return "mixed"
    
    def _select_detail_level(self) -> str:
        """选择详细程度"""
        verbosity = self.user.communication_pref.verbosity
        sensing = self.user.cognitive_style.sensing_intuitive
        
        if verbosity == VerbosityLevel.MINIMAL:
            return "minimal"
        elif verbosity == VerbosityLevel.DETAILED or sensing < 0.3:
            return "comprehensive"
        elif verbosity == VerbosityLevel.CONCISE:
            return "essential"
        else:
            return "contextual"
    
    def _select_proactive_elements(self) -> List[str]:
        """选择主动提供的元素"""
        elements = []
        
        # 基于关系阶段
        if self.user.relationship_stage in [RelationshipStage.PARTNER, RelationshipStage.COMPANION]:
            elements.append("anticipatory_suggestions")
        
        # 基于尽责性
        if self.user.personality.conscientiousness > 0.7:
            elements.append("risk_warnings")
            elements.append("deadline_reminders")
        
        # 基于开放性
        if self.user.personality.openness > 0.7:
            elements.append("alternative_approaches")
            elements.append("related_explorations")
        
        return elements
    
    def _determine_confirmation_need(self, context: Dict) -> str:
        """确定确认需求级别"""
        risk_level = context.get("risk_level", "low")
        relationship = self.user.relationship_stage
        
        if risk_level == "high":
            return "explicit"
        elif relationship in [RelationshipStage.STRANGER, RelationshipStage.ACQUAINTANCE]:
            return "explicit"
        elif risk_level == "medium":
            return "implicit"
        else:
            return "none"
    
    def recommend_content(self, topic: str, count: int = 3) -> List[Dict]:
        """
        推荐相关内容
        
        Args:
            topic: 主题
            count: 推荐数量
            
        Returns:
            推荐内容列表
        """
        # 基于用户兴趣模型生成推荐
        interests = self.user.interests
        
        # 计算兴趣匹配度
        recommendations = []
        
        # 主要兴趣匹配
        for interest in interests["primary"]:
            if self._topic_similarity(topic, interest) > 0.5:
                recommendations.append({
                    "topic": interest,
                    "relevance": "high",
                    "reason": "匹配主要兴趣"
                })
        
        # 次要兴趣匹配
        for interest in interests["secondary"]:
            if self._topic_similarity(topic, interest) > 0.3:
                recommendations.append({
                    "topic": interest,
                    "relevance": "medium",
                    "reason": "相关兴趣"
                })
        
        # 探索性推荐 (基于开放性)
        if self.user.personality.openness > 0.7 and len(recommendations) < count:
            recommendations.append({
                "topic": f"新兴: {topic}创新方向",
                "relevance": "exploratory",
                "reason": "基于高开放性推荐探索"
            })
        
        return recommendations[:count]
    
    def _topic_similarity(self, topic1: str, topic2: str) -> float:
        """计算主题相似度 (简化版)"""
        # 实际实现应使用embedding或知识图谱
        words1 = set(topic1.lower().split())
        words2 = set(topic2.lower().split())
        intersection = words1 & words2
        union = words1 | words2
        return len(intersection) / len(union) if union else 0.0
    
    def adapt_to_feedback(self, feedback: Dict):
        """
        根据反馈调整推荐策略
        
        Args:
            feedback: {
                "type": "explicit" | "implicit",
                "content": str,
                "rating": float,  # 1-5或-1到1
                "context": Dict
            }
        """
        self.feedback_history.append(feedback)
        
        if feedback["type"] == "explicit":
            # 显式反馈直接更新偏好
            self._update_from_explicit_feedback(feedback)
        else:
            # 隐式反馈累积后更新
            self._update_from_implicit_feedback(feedback)
    
    def _update_from_explicit_feedback(self, feedback: Dict):
        """从显式反馈学习"""
        rating = feedback.get("rating", 3)
        
        if rating >= 4:
            # 高评分，强化当前策略
            pass
        elif rating <= 2:
            # 低评分，调整策略
            if "too_verbose" in feedback.get("content", ""):
                # 用户觉得太啰嗦
                if self.user.communication_pref.verbosity == VerbosityLevel.DETAILED:
                    self.user.communication_pref.verbosity = VerbosityLevel.MODERATE
            elif "too_brief" in feedback.get("content", ""):
                # 用户觉得太简略
                if self.user.communication_pref.verbosity == VerbosityLevel.CONCISE:
                    self.user.communication_pref.verbosity = VerbosityLevel.MODERATE
    
    def _update_from_implicit_feedback(self, feedback: Dict):
        """从隐式反馈学习"""
        # 累积足够样本后更新
        if len(self.feedback_history) >= 10:
            recent = self.feedback_history[-10:]
            # 分析模式并调整
            pass


# ============================================================================
# 反馈学习系统
# ============================================================================

class FeedbackLearningSystem:
    """
    反馈学习系统
    处理显式和隐式反馈，持续优化用户模型
    """
    
    def __init__(self, user_profile: UserProfile):
        self.user = user_profile
        self.signal_buffer: List[Dict] = []
        
    def record_explicit_feedback(self, feedback_type: str, content: str, 
                                  rating: Optional[float] = None):
        """记录显式反馈"""
        feedback = {
            "timestamp": datetime.now().isoformat(),
            "type": "explicit",
            "feedback_type": feedback_type,  # rating/correction/preference
            "content": content,
            "rating": rating
        }
        
        self.user.explicit_feedback.append(feedback)
        self.user.learning_stats["total_interactions"] += 1
        
        # 立即处理某些类型的反馈
        if feedback_type == "preference":
            self._process_preference_feedback(content)
        elif feedback_type == "correction":
            self._process_correction_feedback(content)
            self.user.learning_stats["correction_count"] += 1
    
    def record_implicit_signal(self, signal_type: str, value: Any, context: Dict):
        """记录隐式反馈信号"""
        signal = {
            "timestamp": datetime.now().isoformat(),
            "type": "implicit",
            "signal_type": signal_type,
            "value": value,
            "context": context
        }
        
        self.signal_buffer.append(signal)
        
        # 缓冲区满时处理
        if len(self.signal_buffer) >= 20:
            self._process_signal_buffer()
    
    def _process_preference_feedback(self, content: str):
        """处理偏好反馈"""
        # 解析偏好表达
        if "简洁" in content or "简短" in content:
            self.user.communication_pref.verbosity = VerbosityLevel.CONCISE
        elif "详细" in content or "更多细节" in content:
            self.user.communication_pref.verbosity = VerbosityLevel.DETAILED
        
        if "正式" in content:
            self.user.communication_pref.formality = CommunicationFormality.FORMAL
        elif "随意" in content or "轻松" in content:
            self.user.communication_pref.formality = CommunicationFormality.CASUAL
    
    def _process_correction_feedback(self, content: str):
        """处理纠正反馈"""
        # 记录纠正类型
        if "事实" in content or "错误" in content:
            pass  # 事实错误，需要更新知识
        elif "风格" in content or "语气" in content:
            pass  # 风格问题，调整输出
    
    def _process_signal_buffer(self):
        """处理信号缓冲区"""
        if not self.signal_buffer:
            return
        
        # 聚合信号
        signals = self.signal_buffer[:]
        self.signal_buffer = []
        
        # 分析响应时间模式
        response_times = [s["value"] for s in signals 
                         if s["signal_type"] == "response_time"]
        if response_times:
            avg_time = sum(response_times) / len(response_times)
            if avg_time < 30:
                self.user.current_state["response_pattern"] = "quick"
            elif avg_time > 120:
                self.user.current_state["response_pattern"] = "thoughtful"
        
        # 分析参与度
        engagement_signals = [s for s in signals if s["signal_type"] == "engagement"]
        if engagement_signals:
            avg_engagement = sum(s["value"] for s in engagement_signals) / len(engagement_signals)
            if avg_engagement > 0.7:
                self.user.relationship_metrics.intimacy_level = min(1.0,
                    self.user.relationship_metrics.intimacy_level + 0.01)
        
        # 保存到用户档案
        self.user.implicit_signals.extend(signals)


# ============================================================================
# 用户理解系统主类
# ============================================================================

class UserUnderstandingSystem:
    """
    用户理解系统主类
    整合所有组件，提供统一的API
    """
    
    def __init__(self, storage_dir: str = "memory/user"):
        self.storage_dir = Path(storage_dir)
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.profiles: Dict[str, UserProfile] = {}
        self.engines: Dict[str, PersonalizationEngine] = {}
        self.feedback_systems: Dict[str, FeedbackLearningSystem] = {}
        
    def get_or_create_profile(self, user_id: str) -> UserProfile:
        """获取或创建用户档案"""
        if user_id not in self.profiles:
            # 尝试加载
            profile = UserProfile.load(user_id, str(self.storage_dir))
            if profile is None:
                # 创建新档案
                profile = UserProfile(user_id)
                profile.save(str(self.storage_dir))
            
            self.profiles[user_id] = profile
            self.engines[user_id] = PersonalizationEngine(profile)
            self.feedback_systems[user_id] = FeedbackLearningSystem(profile)
        
        return self.profiles[user_id]
    
    def process_interaction(self, user_id: str, message: str, 
                           context: Dict = None) -> Dict:
        """
        处理用户交互
        
        Args:
            user_id: 用户ID
            message: 用户消息
            context: 交互上下文
            
        Returns:
            处理结果和推荐策略
        """
        profile = self.get_or_create_profile(user_id)
        engine = self.engines[user_id]
        feedback = self.feedback_systems[user_id]
        
        context = context or {}
        
        # 1. 提取信号
        signals = self._extract_signals(message, context)
        
        # 2. 更新情感状态
        if "emotion" in signals:
            emotion = signals["emotion"]
            profile.update_emotion(
                valence=emotion.get("valence", 0),
                arousal=emotion.get("arousal", 0.5),
                dominance=emotion.get("dominance", 0.5)
            )
        
        # 3. 生成响应策略
        strategy = engine.generate_response_strategy(context)
        
        # 4. 检查关系阶段转换
        new_stage = profile.check_stage_transition()
        if new_stage:
            profile.transition_stage(new_stage)
        
        # 5. 计算SoulSync分数
        compatibility = SoulSync.calculate_compatibility(profile)
        profile.soulsync_score = compatibility["overall"]
        profile.compatibility_breakdown = compatibility["breakdown"]
        
        # 6. 获取行为适配建议
        adaptations = SoulSync.adapt_agent_behavior(profile)
        
        # 7. 记录隐式信号
        feedback.record_implicit_signal("interaction", {
            "message_length": len(message),
            "has_question": "?" in message,
            "urgency_indicators": self._detect_urgency(message)
        }, context)
        
        # 8. 保存更新
        profile.save(str(self.storage_dir))
        
        return {
            "user_id": user_id,
            "profile_summary": {
                "relationship_stage": profile.relationship_stage.value,
                "soulsync_score": profile.soulsync_score,
                "current_emotion": profile.current_emotion.to_dict()
            },
            "response_strategy": strategy,
            "agent_adaptations": adaptations,
            "soulsync": compatibility
        }
    
    def record_feedback(self, user_id: str, feedback_type: str, 
                       content: str, rating: Optional[float] = None):
        """记录用户反馈"""
        if user_id not in self.feedback_systems:
            self.get_or_create_profile(user_id)
        
        feedback = self.feedback_systems[user_id]
        feedback.record_explicit_feedback(feedback_type, content, rating)
        
        # 更新个性化引擎
        engine = self.engines[user_id]
        engine.adapt_to_feedback({
            "type": "explicit",
            "content": content,
            "rating": rating
        })
        
        # 保存
        self.profiles[user_id].save(str(self.storage_dir))
    
    def get_personalized_response(self, user_id: str, base_content: str) -> str:
        """
        获取个性化响应
        
        Args:
            user_id: 用户ID
            base_content: 基础内容
            
        Returns:
            个性化后的内容
        """
        profile = self.get_or_create_profile(user_id)
        engine = self.engines[user_id]
        
        # 获取适配策略
        adaptations = SoulSync.adapt_agent_behavior(profile)
        
        # 应用个性化 (简化示例)
        content = base_content
        
        # 根据详细程度调整
        if adaptations.get("detail_level") == "high":
            # 添加更多细节
            pass
        elif adaptations.get("detail_level") == "overview_first":
            # 先给概述
            pass
        
        return content
    
    def _extract_signals(self, message: str, context: Dict) -> Dict:
        """从消息中提取信号"""
        signals = {}
        
        # 情感检测 (简化版，实际应使用NLP模型)
        positive_words = ["好", "棒", "优秀", "喜欢", "感谢"]
        negative_words = ["差", "糟", "讨厌", "失望", "错误"]
        
        pos_count = sum(1 for w in positive_words if w in message)
        neg_count = sum(1 for w in negative_words if w in message)
        
        if pos_count > neg_count:
            signals["emotion"] = {"valence": 0.5, "arousal": 0.6, "dominance": 0.5}
        elif neg_count > pos_count:
            signals["emotion"] = {"valence": -0.3, "arousal": 0.5, "dominance": 0.4}
        else:
            signals["emotion"] = {"valence": 0, "arousal": 0.5, "dominance": 0.5}
        
        return signals
    
    def _detect_urgency(self, message: str) -> List[str]:
        """检测紧急程度指标"""
        indicators = []
        urgency_words = ["紧急", "马上", "立刻", "尽快", "asap", "urgent"]
        
        for word in urgency_words:
            if word in message.lower():
                indicators.append(word)
        
        return indicators
    
    def get_user_insights(self, user_id: str) -> Dict:
        """获取用户洞察报告"""
        profile = self.get_or_create_profile(user_id)
        
        return {
            "user_id": user_id,
            "personality_summary": {
                "big_five": profile.personality.to_dict(),
                "cognitive_style": profile.cognitive_style.to_dict()
            },
            "relationship_status": {
                "stage": profile.relationship_stage.value,
                "progress": profile.stage_progress,
                "intimacy": profile.relationship_metrics.intimacy_level,
                "trust": profile.relationship_metrics.trust_level
            },
            "soulsync": {
                "score": profile.soulsync_score,
                "breakdown": profile.compatibility_breakdown
            },
            "learning_progress": profile.learning_stats
        }


# ============================================================================
# 使用示例
# ============================================================================

def example_usage():
    """使用示例"""
    # 初始化系统
    system = UserUnderstandingSystem()
    
    # 获取用户档案
    user = system.get_or_create_profile("user_001")
    
    # 设置基础信息
    user.identity.name = "兰山"
    user.identity.aliases = ["CEO", "董事长"]
    user.career.role = "AI组织创始人"
    user.career.experience_level = "expert"
    
    # 处理交互
    result = system.process_interaction(
        user_id="user_001",
        message="帮我设计一个新的Agent架构",
        context={"risk_level": "medium"}
    )
    
    print(f"关系阶段: {result['profile_summary']['relationship_stage']}")
    print(f"SoulSync分数: {result['profile_summary']['soulsync_score']}")
    print(f"响应策略: {result['response_strategy']}")
    
    # 记录反馈
    system.record_feedback(
        user_id="user_001",
        feedback_type="preference",
        content="更喜欢简洁的回复",
        rating=4.5
    )
    
    # 获取洞察
    insights = system.get_user_insights("user_001")
    print(f"\n用户洞察: {json.dumps(insights, indent=2, ensure_ascii=False)}")


if __name__ == "__main__":
    example_usage()
