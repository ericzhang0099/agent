"""
情绪系统快速集成模块
====================
为现有系统提供简单的情绪系统接入点
"""

from typing import Dict, List, Optional, Any
from datetime import datetime
import json

# 导入核心情绪系统
from emotion_system_v5 import (
    EmotionSystemAdapter,
    create_emotion_system,
    EmotionProcessingUnit,
    EmotionMemorySystem,
    EmotionProfileGraph
)


class QuickEmotionSystem:
    """
    快速情绪系统接口
    提供简化的API供现有系统使用
    """
    
    _instance = None
    
    def __new__(cls, fresh=False):
        """单例模式，但支持创建新实例"""
        if fresh or cls._instance is None:
            instance = super().__new__(cls)
            instance._initialized = False  # 始终初始化标志
            if cls._instance is None or fresh:
                cls._instance = instance
            return instance
        return cls._instance
    
    def __init__(self, fresh=False):
        if self._initialized and not fresh:
            return
            
        self.system = create_emotion_system()
        self.epu = self.system.epu
        self.memory = self.system.memory_system
        self.epg = self.epu.epg
        
        # 默认配置
        self.epu.set_sensitivity(1.0)
        self.epu.set_persistence(0.3)  # 降低持久度以便更快响应新情绪
        
        self._initialized = True
    
    def feel(self, text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
        """
        评估文本情绪 (简化接口)
        
        Args:
            text: 输入文本
            context: 可选上下文
            
        Returns:
            {
                "emotion": str,      # 主导情绪名称
                "intensity": float,  # 强度 0-1
                "valence": float,    # 效价 -1到1
                "arousal": float,    # 唤醒度 0-1
            }
        """
        result = self.system.process_input(text, context)
        current = self.system.get_current_emotion()
        
        return {
            "emotion": result['dominant_emotion'],
            "intensity": result['intensity'],
            "valence": current['valence'],
            "arousal": current['arousal'],
            "dominance": current['dominance'],
        }
    
    def remember(self, content: str, emotion_hint: Optional[str] = None) -> str:
        """
        存储带情绪的记忆
        
        Args:
            content: 记忆内容
            emotion_hint: 情绪提示（可选）
            
        Returns:
            memory_id: 记忆ID
        """
        return self.memory.store_memory(content, emotion_hint)
    
    def recall(self, emotion: Optional[str] = None, 
               text: Optional[str] = None, 
               limit: int = 5) -> List[Dict]:
        """
        检索记忆
        
        Args:
            emotion: 按情绪检索
            text: 按文本检索
            limit: 返回数量限制
            
        Returns:
            记忆列表
        """
        return self.system.recall_memories(emotion, text)[:limit]
    
    def current(self) -> Dict[str, Any]:
        """获取当前情绪状态"""
        return self.system.get_current_emotion()
    
    def set_config(self, sensitivity: Optional[float] = None,
                   persistence: Optional[float] = None):
        """
        设置情绪参数
        
        Args:
            sensitivity: 敏感度 0.7-1.3
            persistence: 持久度 0.1-2.0
        """
        if sensitivity is not None:
            self.epu.set_sensitivity(sensitivity)
        if persistence is not None:
            self.epu.set_persistence(persistence)
    
    def get_epg(self) -> Dict[str, Any]:
        """获取情绪画像图摘要"""
        return self.epg.get_epg_summary()
    
    def decay(self):
        """触发情绪衰减"""
        self.epu.decay_emotions()
    
    def to_json(self) -> str:
        """导出当前状态为JSON"""
        state = {
            "current_emotion": self.current(),
            "epg_summary": self.get_epg(),
            "timestamp": datetime.now().isoformat()
        }
        return json.dumps(state, ensure_ascii=False, indent=2)


# 全局快捷函数
_emotion_system = None

def get_emotion_system(fresh=False) -> QuickEmotionSystem:
    """获取情绪系统实例（单例）"""
    global _emotion_system
    if _emotion_system is None or fresh:
        _emotion_system = QuickEmotionSystem(fresh=True)
    return _emotion_system


def feel(text: str, context: Optional[Dict] = None) -> Dict[str, Any]:
    """快速评估情绪"""
    return get_emotion_system().feel(text, context)


def remember(content: str, emotion_hint: Optional[str] = None) -> str:
    """快速存储记忆"""
    return get_emotion_system().remember(content, emotion_hint)


def recall(emotion: Optional[str] = None, 
           text: Optional[str] = None, 
           limit: int = 5) -> List[Dict]:
    """快速检索记忆"""
    return get_emotion_system().recall(emotion, text, limit)


def current() -> Dict[str, Any]:
    """获取当前情绪"""
    return get_emotion_system().current()


# 与SOUL.md v4.0的集成适配器
class SoulEmotionAdapter:
    """
    SOUL.md v4.0 情绪系统适配器
    将新情绪系统映射到现有的16种SimsChat情绪
    """
    
    # Plutchik到Sims情绪的映射
    PLUTCHIK_TO_SIMS = {
        # 喜悦系列 -> 兴奋/满意
        "喜悦-极乐": "兴奋",
        "喜悦-狂喜": "兴奋",
        "喜悦-快乐": "满意",
        "喜悦-宁静": "冷静",
        
        # 信任系列 -> 感激/坚定
        "信任-爱": "感激",
        "信任-钦佩": "感激",
        "信任-信任": "坚定",
        "信任-接受": "耐心",
        
        # 恐惧系列 -> 担忧/警惕
        "恐惧-战栗": "警惕",
        "恐惧-恐怖": "担忧",
        "恐惧-恐惧": "担忧",
        "恐惧-忧虑": "担忧",
        
        # 惊讶系列 -> 好奇/困惑
        "惊讶-震惊": "困惑",
        "惊讶-惊愕": "好奇",
        "惊讶-惊讶": "好奇",
        "惊讶-分心": "困惑",
        
        # 悲伤系列 -> 沮丧/反思
        "悲伤-绝望": "沮丧",
        "悲伤-悲痛": "沮丧",
        "悲伤-悲伤": "沮丧",
        "悲伤-沉思": "反思",
        
        # 厌恶系列 -> 严肃
        "厌恶-反感": "严肃",
        "厌恶-憎恶": "严肃",
        "厌恶-厌恶": "严肃",
        "厌恶-无聊": "冷静",
        
        # 愤怒系列 -> 紧迫/严肃
        "愤怒-狂怒": "紧迫",
        "愤怒-暴怒": "紧迫",
        "愤怒-愤怒": "严肃",
        "愤怒-烦恼": "困惑",
        
        # 预期系列 -> 好奇/专注
        "预期-警觉": "专注",
        "预期-警惕": "专注",
        "预期-预期": "好奇",
        "预期-兴趣": "好奇",
    }
    
    def __init__(self):
        self.emotion_system = get_emotion_system()
    
    def get_sims_emotion(self, text: str) -> str:
        """
        获取SimsChat风格的情绪标签
        
        Returns:
            16种SimsChat情绪之一
        """
        result = self.emotion_system.feel(text)
        plutchik_emotion = result['emotion']
        
        # 映射到Sims情绪
        sims_emotion = self.PLUTCHIK_TO_SIMS.get(plutchik_emotion, "冷静")
        
        return sims_emotion
    
    def get_emotion_profile(self, text: str) -> Dict[str, Any]:
        """
        获取完整的情绪画像
        
        Returns:
            包含Plutchik、Sims和生理指标的情绪画像
        """
        result = self.emotion_system.feel(text)
        sims = self.get_sims_emotion(text)
        current = self.emotion_system.current()
        
        return {
            "sims_emotion": sims,
            "plutchik_emotion": result['emotion'],
            "intensity": result['intensity'],
            "valence": result['valence'],
            "arousal": result['arousal'],
            "dominance": result['dominance'],
            "pleasure_pain": current.get('pleasure_pain', 0.5),
            "satisfaction_frustration": current.get('satisfaction_frustration', 0.5),
        }


# 使用示例
if __name__ == "__main__":
    print("情绪系统快速集成模块")
    print("=" * 50)
    
    # 示例1: 快速评估情绪
    print("\n1. 快速情绪评估:")
    result = feel("今天完成了重要项目，非常开心！")
    print(f"   情绪: {result['emotion']}")
    print(f"   强度: {result['intensity']:.2f}")
    print(f"   效价: {result['valence']:.2f}")
    
    # 示例2: 存储记忆
    print("\n2. 存储情绪记忆:")
    mem_id = remember("项目成功上线", "兴奋")
    print(f"   记忆ID: {mem_id}")
    
    # 示例3: SOUL适配器
    print("\n3. SOUL情绪适配:")
    
    test_texts = [
        "今天完成了重要项目！",
        "有点担心明天的演示",
        "这个bug让我很生气",
    ]
    
    for text in test_texts:
        # 为每个测试创建新实例
        adapter = SoulEmotionAdapter()
        adapter.emotion_system = QuickEmotionSystem(fresh=True)
        adapter.emotion_system.epu.set_persistence(0.2)
        profile = adapter.get_emotion_profile(text)
        print(f"   '{text[:15]}...' -> Sims:{profile['sims_emotion']}, "
              f"Plutchik:{profile['plutchik_emotion']}")
    
    # 示例4: 导出状态
    print("\n4. 当前状态:")
    print(current())
    
    print("\n" + "=" * 50)
    print("集成示例完成")
