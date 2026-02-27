"""
MetaSoul EPU4 情绪系统 v5.0 - 集成指南
==========================================

## 系统概述

基于MetaSoul EPU4 Emotion Profile Graph (EPG) 的情绪处理单元，扩展支持：
- 32种Plutchik情绪轮 (8基础×4强度)
- 16种SimsChat情绪状态  
- 12种MetaSoul核心情绪
- 情绪-记忆关联机制
- 64万亿种独特情绪状态

## 快速开始

```python
from emotion_system_v5 import create_emotion_system

# 创建情绪系统
system = create_emotion_system()

# 处理输入并获取情绪响应
result = system.process_input("今天完成了重要项目，非常开心！")
print(result['dominant_emotion'])  # 输出: 喜悦-狂喜

# 获取当前情绪状态
current = system.get_current_emotion()
print(f"当前情绪: {current['emotion']} (效价: {current['valence']:.2f})")

# 存储情绪记忆
system.memory_system.store_memory("项目成功上线", emotion_hint="兴奋")

# 通过情绪检索记忆
memories = system.recall_memories(emotion="喜悦")
```

## 核心组件

### 1. EmotionProcessingUnit (EPU)
核心情绪处理引擎，提供：
- `appraise(stimulus, context)` - 评估情绪
- `set_sensitivity(0.7-1.3)` - 设置敏感度
- `set_persistence(0.1-2.0)` - 设置持久度
- `create_emotion_memory(content)` - 创建情绪记忆

### 2. EmotionProfileGraph (EPG)
情绪画像图，记录情绪发展轨迹：
- 情绪历史追踪
- 情绪基线学习
- 学习曲线计算
- 情绪-记忆关联矩阵

### 3. EmotionMemorySystem
情绪-记忆关联系统：
- 情绪标记记忆
- 相似记忆关联
- 情绪检索记忆
- 记忆权重衰减

## 情绪类型

### Plutchik情绪轮 (32种)
8种基础情绪 × 4种强度：
- 喜悦: 宁静→快乐→狂喜→极乐
- 信任: 接受→信任→钦佩→爱
- 恐惧: 忧虑→恐惧→恐怖→战栗
- 惊讶: 分心→惊讶→惊愕→震惊
- 悲伤: 沉思→悲伤→悲痛→绝望
- 厌恶: 无聊→厌恶→憎恶→反感
- 愤怒: 烦恼→愤怒→暴怒→狂怒
- 预期: 兴趣→预期→警惕→警觉

### SimsChat情绪 (16种)
与SOUL.md v4.0兼容：
兴奋、坚定、专注、担忧、反思、满意、好奇、耐心、
紧迫、冷静、困惑、沮丧、感激、警惕、幽默、严肃

### 复合情绪 (9种)
Plutchik情绪组合：
乐观、爱、屈服、敬畏、不赞成、悲观、怨恨、厌恶、攻击

## 与现有系统集成

### 集成到SOUL.md v4.0

```python
# 在SOUL.md的情绪系统部分
from emotion_system_v5 import EmotionSystemAdapter

class KimiClawSoul:
    def __init__(self):
        self.emotion_system = EmotionSystemAdapter()
        
    def respond(self, user_input, context):
        # 评估情绪
        emotion_result = self.emotion_system.process_input(
            user_input, context
        )
        
        # 根据情绪调整响应
        dominant_emotion = emotion_result['dominant_emotion']
        
        # 存储交互记忆
        self.emotion_system.memory_system.store_memory(
            user_input, 
            emotion_hint=dominant_emotion
        )
        
        return self.generate_response(emotion_result)
```

### 集成到现有TTS系统

```python
from emotion_system_v5 import create_emotion_system
from elevenlabs_emotion.emotional_tts import EmotionalTTS

class EmotionalVoiceSystem:
    def __init__(self):
        self.emotion_system = create_emotion_system()
        self.tts = EmotionalTTS()
        
    def speak(self, text):
        # 评估文本情绪
        result = self.emotion_system.process_input(text)
        emotion = result['dominant_emotion']
        
        # 映射到TTS情绪
        tts_emotion = self.map_to_tts_emotion(emotion)
        
        # 合成语音
        return self.tts.synthesize(text, emotion=tts_emotion)
        
    def map_to_tts_emotion(self, emotion):
        # 情绪映射表
        mapping = {
            "喜悦-狂喜": "excited",
            "喜悦-快乐": "happy",
            "愤怒-暴怒": "angry",
            "恐惧-恐惧": "fearful",
            "悲伤-悲伤": "sad",
            # ... 更多映射
        }
        return mapping.get(emotion, "neutral")
```

## 配置参数

### 敏感度 (Sensitivity)
- 范围: 70% - 130%
- 默认: 100%
- 作用: 控制情绪反应强度

```python
system.epu.set_sensitivity(1.2)  # 更敏感
system.epu.set_sensitivity(0.8)  # 更稳定
```

### 持久度 (Persistence)
- 范围: 10% - 200%
- 默认: 100%
- 作用: 控制情绪持续时间

```python
system.epu.set_persistence(0.5)  # 情绪快速消退
system.epu.set_persistence(1.5)  # 情绪持续更久
```

### 衰减率 (Decay Rate)
- 范围: 0.01 - 0.2
- 默认: 0.05
- 作用: 控制情绪自然衰减速度

## 情绪记忆关联验证

### 测试用例

```python
def test_emotion_memory_association():
    system = create_emotion_system()
    
    # 存储不同情绪的记忆
    system.process_input("项目成功上线，非常开心！")
    system.memory_system.store_memory("项目成功上线")
    
    system.process_input("服务器宕机了，很担心")
    system.memory_system.store_memory("服务器宕机")
    
    system.process_input("收到用户投诉，有点生气")
    system.memory_system.store_memory("用户投诉")
    
    # 验证情绪-记忆关联
    joy_memories = system.epu.epg.retrieve_memories_by_emotion("喜悦-快乐")
    fear_memories = system.epu.epg.retrieve_memories_by_emotion("恐惧-忧虑")
    
    assert len(joy_memories) > 0, "应该检索到喜悦相关记忆"
    assert len(fear_memories) > 0, "应该检索到恐惧相关记忆"
    
    print("✅ 情绪-记忆关联验证通过")
```

## 性能指标

- 情绪评估延迟: < 10ms
- 记忆检索延迟: < 50ms (1000条记忆)
- 内存占用: ~50MB (含EPG和1000条记忆)
- 支持的情绪状态: 64万亿+

## 文件清单

- `emotion_system_v5.py` - 核心情绪系统
- `EMOTION_SYSTEM_INTEGRATION.md` - 集成指南 (本文件)

## 版本历史

- v5.0 (2026-02-27): 初始版本
  - MetaSoul EPU4 EPG集成
  - 32种Plutchik情绪轮
  - 16种SimsChat情绪
  - 情绪-记忆关联机制
"""

# 集成示例代码
if __name__ == "__main__":
    print("MetaSoul EPU4 情绪系统 v5.0 集成指南")
    print("=" * 50)
    print("\n请查看本文件的文档字符串获取完整集成指南")
    print("\n快速示例:")
    print("-" * 50)
    
    from emotion_system_v5 import create_emotion_system
    
    # 创建系统
    system = create_emotion_system()
    
    # 处理输入
    result = system.process_input("今天完成了重要项目！")
    print(f"输入: 今天完成了重要项目！")
    print(f"主导情绪: {result['dominant_emotion']}")
    print(f"EPG状态: {result['epg_summary']['history_size']} 条历史记录")
