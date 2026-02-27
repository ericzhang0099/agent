# 用户理解系统 - 架构文档

## 系统概述

用户理解系统是一个基于**7维度用户模型**的个性化框架，与SOUL.md v3.0的8维度Agent人格模型协同工作，实现用户-Agent人格匹配和双向进化。

## 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                    用户理解系统架构                               │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │                 UserUnderstandingSystem                   │  │
│  │                      (主控类)                             │  │
│  └──────────────┬─────────────────────────────┬──────────────┘  │
│                 │                             │                 │
│        ┌────────▼────────┐           ┌────────▼────────┐        │
│        │   UserProfile   │           │Personalization  │        │
│        │   (用户档案)     │           │   Engine        │        │
│        └────────┬────────┘           │  (推荐引擎)      │        │
│                 │                    └─────────────────┘        │
│        ┌────────▼────────┐                                      │
│        │  7维度模型       │           ┌─────────────────┐        │
│        │ • Profile       │           │  FeedbackLearning│        │
│        │ • Modeling      │           │     System      │        │
│        │ • Adaptation    │           │   (反馈学习)     │        │
│        │ • Relationship  │           └─────────────────┘        │
│        │ • Privacy       │                                      │
│        │ • Learning      │           ┌─────────────────┐        │
│        │ • SoulSync      │           │    SoulSync     │        │
│        └─────────────────┘           │  (人格匹配)      │        │
│                                      └─────────────────┘        │
└─────────────────────────────────────────────────────────────────┘
```

## 7维度用户模型详解

### 维度1: User Profile (用户画像)

**静态属性**
- 基础身份 (姓名、别名、时区)
- 职业背景 (角色、行业、经验等级)
- 沟通偏好 (正式程度、详细程度、响应速度)
- 技术背景 (熟练度、偏好语言、工具熟悉度)

**动态属性**
- 当前状态 (可用性、能量水平、专注模式)
- 短期偏好 (当前关注主题、避免主题、进行中的项目)

**代码位置**: `user_understanding_system.py` - `UserIdentity`, `CareerProfile`, `CommunicationPreference`, `TechnicalProfile`

### 维度2: User Modeling (用户建模)

**Big Five人格模型 (OCEAN)**
- Openness (开放性): 0-1
- Conscientiousness (尽责性): 0-1
- Extraversion (外向性): 0-1
- Agreeableness (宜人性): 0-1
- Neuroticism (神经质): 0-1

**认知风格**
- 信息处理偏好 (视觉/语言、顺序/整体、感知/直觉)
- 决策风格 (理性/直觉、风险承受、速度/准确)
- 学习风格 (主动/反思、理论/实用)

**情感状态 (VAD模型)**
- Valence (效价): -1 到 +1
- Arousal (唤醒度): 0 到 1
- Dominance (支配感): 0 到 1

**代码位置**: `user_understanding_system.py` - `BigFivePersonality`, `CognitiveStyle`, `EmotionalState`

### 维度3: Personalization (个性化适配)

**沟通风格适配**
- 输出风格 (语气、结构、详细程度)
- 语言特征 (术语使用、隐喻、幽默)
- 格式偏好 (Markdown、emoji、长度)

**内容推荐**
- 兴趣模型 (主要、次要、新兴)
- 内容偏好 (类型、深度、新颖性)
- 推荐策略 (探索-利用平衡)

**交互优化**
- 交互节奏 (响应延迟、打断容忍)
- 主动性级别
- 确认策略

**代码位置**: `personalization_engine.py` - `RecommendationEngine`, `UserInterestModel`

### 维度4: Relationship (用户关系)

**关系阶段模型**
```
Stranger → Acquaintance → Collaborator → Partner → Companion
(陌生人)   (相识)        (协作者)      (伙伴)    (伴侣)
```

**关系度量**
- 亲密度 (情感、智力、经历)
- 信任度 (能力、善意、诚信)
- 依赖度 (健康平衡监测)

**关系演化**
- 里程碑事件追踪
- 关系质量指标
- 风险预警

**代码位置**: `user_understanding_system.py` - `RelationshipMetrics`, `RelationshipMilestone`

### 维度5: Privacy (隐私保护)

**数据最小化**
- 数据分类 (核心/行为/临时)
- 保留策略
- 差分隐私配置

**权限控制**
- 用户控制级别
- 自动决策边界
- 透明度设置

**可遗忘设计**
- 时间衰减
- 相关性过滤
- 用户请求遗忘

**代码位置**: `user_understanding_system.py` - `UserProfile.privacy_settings`

### 维度6: Learning (反馈学习)

**显式反馈**
- 直接评分
- 结构化反馈
- 修正指令

**隐式反馈**
- 行为信号 (响应时间、编辑模式、参与深度)
- 情感信号 (情绪趋势、热情指标、挫折指标)

**持续优化**
- 在线学习
- 批量更新
- A/B测试

**代码位置**: `user_understanding_system.py` - `FeedbackLearningSystem`

### 维度7: SoulSync (用户-Agent人格匹配)

**匹配维度**
- 人格相似度
- 价值观对齐
- 沟通风格匹配
- 节奏兼容性

**动态适配策略**
- 不匹配响应
- 互补策略
- 行为适配建议

**共同演化目标**
- 共享目标
- 互相学习
- 关系愿景

**代码位置**: `user_understanding_system.py` - `SoulSync`

## 数据流

```
用户输入
    │
    ▼
┌─────────────────┐
│   信号提取       │ ← NLP解析、情感分析、行为编码
└────────┬────────┘
         │
    ┌────┴────┬────────────┐
    ▼         ▼            ▼
┌────────┐ ┌────────┐ ┌────────┐
│情感状态 │ │意图识别 │ │行为模式 │
└────┬───┘ └────┬───┘ └────┬───┘
     │          │          │
     └──────────┼──────────┘
                ▼
        ┌───────────────┐
        │  7维度建模更新  │
        └───────┬───────┘
                │
        ┌───────┴───────┐
        ▼               ▼
┌───────────────┐ ┌───────────────┐
│ SoulSync匹配  │ │ 推荐引擎       │
│ 分数计算      │ │ 内容生成       │
└───────┬───────┘ └───────┬───────┘
        │                 │
        └────────┬────────┘
                 ▼
        ┌───────────────┐
        │ 个性化输出     │
        └───────────────┘
```

## API使用模式

### 基础使用

```python
from user_understanding_system import UserUnderstandingSystem

# 初始化
system = UserUnderstandingSystem()

# 获取用户档案
user = system.get_or_create_profile("user_001")

# 处理交互
result = system.process_interaction(
    user_id="user_001",
    message="帮我设计Agent架构",
    context={"risk_level": "medium"}
)

# 记录反馈
system.record_feedback(
    user_id="user_001",
    feedback_type="preference",
    content="喜欢简洁风格",
    rating=4.5
)
```

### 个性化推荐

```python
from personalization_engine import RecommendationEngine, ContentItem

# 创建引擎
engine = RecommendationEngine(user)

# 定义内容池
content_pool = [
    ContentItem(
        id="1",
        title="AI Agent架构设计",
        content_type="article",
        topics=["AI", "Agent"],
        difficulty="advanced",
        estimated_time=15
    ),
    # ...
]

# 生成推荐
recommendations = engine.recommend(content_pool, n=5)
```

### SoulSync人格匹配

```python
from user_understanding_system import SoulSync

# 计算匹配度
compatibility = SoulSync.calculate_compatibility(user)
print(f"匹配度: {compatibility['overall']:.0%}")

# 获取适配建议
adaptations = SoulSync.adapt_agent_behavior(user)
```

## 存储结构

```
memory/
└── user/
    ├── {user_id}.yaml          # 用户档案
    ├── {user_id}_history.json   # 交互历史
    └── {user_id}_feedback.json  # 反馈记录
```

## 配置文件

### 用户档案 YAML 结构

```yaml
user_id: "user_001"
created_at: "2024-01-01T00:00:00"
updated_at: "2024-01-15T12:00:00"

identity:
  name: "兰山"
  aliases: ["CEO", "董事长"]
  timezone: "Asia/Shanghai"

career:
  role: "AI组织创始人"
  experience_level: "expert"

personality:
  openness: 0.85
  conscientiousness: 0.90
  # ...

relationship_stage: "partner"
stage_progress: 0.75

soulsync_score: 0.88
```

## 扩展点

### 添加新的推荐算法

```python
class MyRecommendationAlgorithm:
    def calculate_score(self, item: ContentItem, user: UserProfile) -> float:
        # 实现评分逻辑
        pass

# 在 RecommendationEngine 中注册
engine.algorithm_weights["my_algo"] = 0.2
```

### 自定义信号提取器

```python
def my_signal_extractor(message: str) -> Dict:
    signals = {}
    # 自定义信号提取逻辑
    return signals

# 在 UserUnderstandingSystem 中使用
```

### 添加新的关系阶段

```python
class RelationshipStage(Enum):
    # 现有阶段...
    MENTOR = "mentor"  # 新增：导师阶段
```

## 性能考虑

- **内存使用**: 活跃用户档案常驻内存，非活跃档案按需加载
- **存储优化**: 行为数据定期聚合，原始数据过期删除
- **计算优化**: 相似度计算结果缓存，避免重复计算
- **隐私保护**: 敏感数据本地存储，差分隐私保护

## 与其他系统的集成

### 与SOUL.md集成

```python
# SoulSync自动读取SOUL.md中的Agent人格配置
AGENT_PROFILE = {
    "personality": {
        "openness": 0.80,
        "conscientiousness": 0.90,
        # 来自SOUL.md
    }
}
```

### 与AGENTS.md集成

- 用户档案在每次会话开始时加载
- 交互记录在会话结束时保存
- 长期记忆更新触发档案更新

## 监控指标

| 指标 | 说明 | 目标值 |
|------|------|--------|
| 用户满意度 | 显式反馈平均评分 | >4.5/5 |
| 需求预测准确率 | 用户确认匹配率 | >80% |
| 关系健康度 | 亲密度+信任度综合 | >0.8 |
| SoulSync分数 | 人格匹配度 | >0.7 |
| 推荐多样性 | 主题覆盖度 | >0.6 |

## 最佳实践

1. **渐进式学习**: 新用户从默认配置开始，逐步个性化
2. **透明推荐**: 向用户解释推荐理由，增加信任
3. **尊重边界**: 高风险操作始终请求确认
4. **定期复盘**: 每月评估关系阶段，调整策略
5. **隐私优先**: 默认最小化数据收集，用户可控

## 未来扩展

- [ ] 多模态输入处理 (语音、图像)
- [ ] 跨用户协同过滤 (隐私保护下)
- [ ] 深度学习模型集成 (情感识别、意图理解)
- [ ] 实时关系可视化仪表板
- [ ] 自动化A/B测试框架
