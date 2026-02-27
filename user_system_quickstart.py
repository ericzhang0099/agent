"""
User Understanding System - Quick Start Guide
用户理解系统快速入门指南
"""

# ============================================================================
# 1. 系统初始化
# ============================================================================

from user_understanding_system import (
    UserUnderstandingSystem, 
    UserProfile, 
    SoulSync,
    CommunicationFormality,
    VerbosityLevel,
    RelationshipStage
)
from personalization_engine import RecommendationEngine, ContentItem

# 初始化系统
system = UserUnderstandingSystem(storage_dir="memory/user")

# ============================================================================
# 2. 用户档案管理
# ============================================================================

# 获取或创建用户档案
user = system.get_or_create_profile("user_001")

# 设置基础信息
user.identity.name = "兰山"
user.identity.aliases = ["CEO", "董事长"]
user.identity.timezone = "Asia/Shanghai"

# 设置职业信息
user.career.role = "AI组织创始人"
user.career.industry = "人工智能"
user.career.experience_level = "expert"
user.career.work_style = "agile"

# 设置沟通偏好
user.communication_pref.formality = CommunicationFormality.SEMI_FORMAL
user.communication_pref.verbosity = VerbosityLevel.CONCISE
user.communication_pref.response_speed = "immediate"

# 设置技术背景
user.technical.proficiency = "expert"
user.technical.preferred_languages = ["Python", "TypeScript"]
user.technical.tools_familiarity = ["OpenClaw", "Git", "Docker"]

# 设置Big Five人格 (基于观察)
user.personality.openness = 0.85
user.personality.conscientiousness = 0.90
user.personality.extraversion = 0.65
user.personality.agreeableness = 0.70
user.personality.neuroticism = 0.30

# 设置认知风格
user.cognitive_style.visual_verbal = 0.6
user.cognitive_style.sequential_global = 0.4
user.cognitive_style.sensing_intuitive = 0.7
user.cognitive_style.rational_intuitive = 0.6
user.cognitive_style.risk_tolerance = 0.75
user.cognitive_style.speed_accuracy = 0.8

# 设置兴趣
user.interests["primary"] = ["AI Agent", "系统架构", "团队管理"]
user.interests["secondary"] = ["编程语言", "开发工具", "效率方法"]

# 保存档案
user.save()

# ============================================================================
# 3. 交互处理与实时适配
# ============================================================================

# 处理用户消息
result = system.process_interaction(
    user_id="user_001",
    message="帮我设计一个新的Agent架构",
    context={
        "risk_level": "medium",
        "topic": "architecture"
    }
)

# 查看处理结果
print(f"关系阶段: {result['profile_summary']['relationship_stage']}")
print(f"SoulSync分数: {result['profile_summary']['soulsync_score']}")
print(f"当前情感: {result['profile_summary']['current_emotion']}")
print(f"响应策略: {result['response_strategy']}")
print(f"Agent适配: {result['agent_adaptations']}")

# ============================================================================
# 4. 反馈学习
# ============================================================================

# 记录显式反馈 - 偏好表达
system.record_feedback(
    user_id="user_001",
    feedback_type="preference",
    content="更喜欢简洁的回复",
    rating=4.5
)

# 记录显式反馈 - 纠正
system.record_feedback(
    user_id="user_001",
    feedback_type="correction",
    content="API版本应该是v2不是v1",
    rating=2.0
)

# 记录显式反馈 - 评分
system.record_feedback(
    user_id="user_001",
    feedback_type="rating",
    content="整体满意度",
    rating=5.0
)

# ============================================================================
# 5. 关系演化
# ============================================================================

# 添加关系里程碑
user.add_milestone(
    event="首次深夜工作会话",
    impact=0.1,
    description="用户在深夜与Agent协作完成紧急任务"
)

user.add_milestone(
    event="完成首个重大项目",
    impact=0.15,
    description="共同完成了Agent团队架构设计"
)

# 检查阶段转换
new_stage = user.check_stage_transition()
if new_stage:
    user.transition_stage(new_stage)
    print(f"关系阶段升级至: {new_stage.value}")

# ============================================================================
# 6. SoulSync人格匹配
# ============================================================================

# 计算匹配度
compatibility = SoulSync.calculate_compatibility(user)
print(f"\n=== SoulSync人格匹配 ===")
print(f"总体匹配度: {compatibility['overall']:.0%}")
print(f"人格相似度: {compatibility['breakdown']['personality']:.0%}")
print(f"价值观对齐: {compatibility['breakdown']['values_alignment']:.0%}")
print(f"沟通匹配: {compatibility['breakdown']['communication_match']:.0%}")
print(f"节奏兼容: {compatibility['breakdown']['pace_compatibility']:.0%}")
print(f"优化建议: {compatibility['recommendations']}")

# 获取Agent行为适配建议
adaptations = SoulSync.adapt_agent_behavior(user)
print(f"\n=== Agent行为适配 ===")
for key, value in adaptations.items():
    print(f"{key}: {value}")

# ============================================================================
# 7. 个性化推荐
# ============================================================================

# 创建推荐引擎
rec_engine = RecommendationEngine(user)

# 模拟内容池
content_pool = [
    ContentItem(
        id="1",
        title="AI Agent架构设计最佳实践",
        content_type="article",
        topics=["AI", "Agent", "架构"],
        difficulty="advanced",
        estimated_time=15,
        metadata={"has_visuals": True}
    ),
    ContentItem(
        id="2",
        title="Python异步编程指南",
        content_type="tutorial",
        topics=["Python", "异步", "编程"],
        difficulty="intermediate",
        estimated_time=20
    ),
    ContentItem(
        id="3",
        title="量子机器学习入门",
        content_type="article",
        topics=["量子计算", "机器学习", "前沿"],
        difficulty="expert",
        estimated_time=30
    ),
    ContentItem(
        id="4",
        title="团队管理心理学",
        content_type="article",
        topics=["管理", "心理学", "团队"],
        difficulty="intermediate",
        estimated_time=12
    ),
    ContentItem(
        id="5",
        title="OpenClaw高级技巧",
        content_type="tutorial",
        topics=["OpenClaw", "工具", "效率"],
        difficulty="advanced",
        estimated_time=10
    )
]

# 模拟历史兴趣数据
rec_engine.interest_model.topic_scores["AI"] = 0.9
rec_engine.interest_model.topic_scores["Agent"] = 0.95
rec_engine.interest_model.topic_scores["架构"] = 0.8
rec_engine.interest_model.topic_scores["管理"] = 0.6
rec_engine.interest_model.content_type_scores["article"] = 0.85
rec_engine.interest_model.content_type_scores["tutorial"] = 0.75

# 生成推荐
recommendations = rec_engine.recommend(content_pool, n=3)

print(f"\n=== 个性化推荐 ===")
for i, rec in enumerate(recommendations, 1):
    print(f"\n{i}. {rec.item.title}")
    print(f"   类型: {rec.item.content_type}")
    print(f"   主题: {', '.join(rec.item.topics)}")
    print(f"   推荐分数: {rec.score:.3f}")
    print(f"   推荐理由: {rec.reason}")
    print(f"   置信度: {rec.confidence:.0%}")

# ============================================================================
# 8. 用户洞察报告
# ============================================================================

insights = system.get_user_insights("user_001")

print(f"\n=== 用户洞察报告 ===")
print(f"\n人格摘要:")
print(f"  - 开放性: {insights['personality_summary']['big_five']['openness']:.0%}")
print(f"  - 尽责性: {insights['personality_summary']['big_five']['conscientiousness']:.0%}")
print(f"  - 外向性: {insights['personality_summary']['big_five']['extraversion']:.0%}")

print(f"\n关系状态:")
print(f"  - 当前阶段: {insights['relationship_status']['stage']}")
print(f"  - 亲密度: {insights['relationship_status']['intimacy']:.0%}")
print(f"  - 信任度: {insights['relationship_status']['trust']:.0%}")

print(f"\nSoulSync:")
print(f"  - 匹配分数: {insights['soulsync']['score']:.0%}")

print(f"\n学习进度:")
print(f"  - 总交互数: {insights['learning_progress']['total_interactions']}")
print(f"  - 纠正次数: {insights['learning_progress']['correction_count']}")

# ============================================================================
# 9. 隐私设置
# ============================================================================

# 配置隐私设置
user.privacy_settings = {
    "collection_level": "adaptive",  # minimal/adaptive/comprehensive
    "can_view_profile": True,
    "can_edit_profile": True,
    "can_delete_data": True,
    "can_export_data": True,
    "explain_recommendations": True,
    "show_confidence_scores": False,
    "retention_days": 90
}

# 数据遗忘示例 (删除特定主题的数据)
def forget_topic(user: UserProfile, topic: str):
    """遗忘特定主题的数据"""
    if topic in user.interests["primary"]:
        user.interests["primary"].remove(topic)
    if topic in user.interests["secondary"]:
        user.interests["secondary"].remove(topic)
    
    # 清除相关信号
    user.implicit_signals = [
        s for s in user.implicit_signals 
        if topic not in str(s.get("context", {}))
    ]
    
    print(f"已遗忘主题: {topic}")

# ============================================================================
# 10. 完整工作流示例
# ============================================================================

def full_workflow_example():
    """
    完整工作流示例:
    1. 初始化用户
    2. 处理交互
    3. 学习反馈
    4. 更新关系
    5. 生成推荐
    6. 保存状态
    """
    
    # 1. 初始化
    system = UserUnderstandingSystem()
    user = system.get_or_create_profile("ceo_lanshan")
    
    # 2. 设置基础档案
    user.identity.name = "兰山"
    user.career.experience_level = "expert"
    user.personality.conscientiousness = 0.90
    user.personality.openness = 0.85
    
    # 3. 模拟多次交互
    interactions = [
        ("帮我分析Agent架构", {"risk_level": "low"}),
        ("这个方案效率不够高", {"risk_level": "medium"}),
        ("需要更简洁的实现", {"risk_level": "low"}),
        ("24小时内完成部署", {"risk_level": "high", "urgent": True}),
    ]
    
    for message, context in interactions:
        result = system.process_interaction(user.user_id, message, context)
        print(f"处理: {message[:20]}... | 情感: {result['profile_summary']['current_emotion']['valence']:+.2f}")
    
    # 4. 记录反馈
    system.record_feedback(user.user_id, "preference", "喜欢简洁风格", 4.5)
    system.record_feedback(user.user_id, "preference", "需要更多代码示例", 4.0)
    
    # 5. 添加里程碑
    user.add_milestone("完成架构设计", 0.15)
    
    # 6. 检查阶段
    new_stage = user.check_stage_transition()
    if new_stage:
        user.transition_stage(new_stage)
    
    # 7. 生成推荐
    rec_engine = RecommendationEngine(user)
    # ... 配置推荐引擎 ...
    
    # 8. 保存
    user.save()
    
    print(f"\n工作流完成!")
    print(f"关系阶段: {user.relationship_stage.value}")
    print(f"SoulSync: {user.soulsync_score:.0%}")
    print(f"总交互: {user.learning_stats['total_interactions']}")

# 运行完整示例
# full_workflow_example()

print("\n" + "="*50)
print("用户理解系统快速入门指南加载完成!")
print("="*50)
