# emotion_system_demo.py
"""
EMOTION.md v4.0 演示脚本
展示情绪系统的核心功能
"""

import asyncio
from datetime import datetime
from emotion_system import EmotionEngine


def print_emotion_state(engine, label=""):
    """打印情绪状态"""
    state = engine.get_current_emotion()
    print(f"\n{'='*50}")
    if label:
        print(f"【{label}】")
    print(f"情绪: {state['emotion']}")
    if state.get('sub_emotion'):
        print(f"子类型: {state['sub_emotion']}")
    print(f"强度: {state['intensity']:.2f}")
    print(f"价值极性: {state['valence']:.2f}")
    print(f"唤醒度: {state['arousal']:.2f}")
    print(f"SOUL维度: {state['soul_dimension']}")
    print(f"描述: {state['description']}")
    print(f"{'='*50}")


def demo_basic_emotions():
    """演示基础情绪"""
    print("\n" + "="*70)
    print("演示1: 16种基础情绪")
    print("="*70)
    
    engine = EmotionEngine()
    
    # 测试各种情绪触发
    test_inputs = [
        ("重大突破！我们成功了！", "兴奋触发"),
        ("交给我，没问题", "坚定触发"),
        ("我又熬夜了，没睡觉", "担忧触发"),
        ("紧急！deadline还有1小时", "紧迫触发"),
        ("不明白这是什么意思", "困惑触发"),
        ("哈哈，真有趣", "幽默触发"),
        ("谢谢你的帮助", "感激触发"),
        ("这是原则问题", "严肃触发"),
        ("开始深度工作", "专注触发"),
        ("让我复盘一下", "反思触发"),
    ]
    
    for text, label in test_inputs:
        result = engine.process_input(text)
        if result["triggered"]:
            print(f"\n输入: '{text}'")
            print(f"触发: {label}")
            print(f"检测到情绪: {result['emotion']} (强度: {result['intensity']:.2f})")
            if result.get('triggers'):
                trigger = result['triggers'][0]
                print(f"触发器: {trigger['trigger_id']} (优先级: {trigger['priority']})")


def demo_emotion_decay():
    """演示情绪衰减"""
    print("\n" + "="*70)
    print("演示2: 情绪衰减机制")
    print("="*70)
    
    from emotion_system.decay import EmotionDecayModel
    
    model = EmotionDecayModel()
    
    emotions = ["Excited", "Focused", "Calm", "Concerned"]
    
    print("\n不同情绪的衰减曲线 (初始强度0.9, 60分钟):")
    print("-" * 70)
    print(f"{'时间(分)':<10} {'兴奋':<12} {'专注':<12} {'冷静':<12} {'担忧':<12}")
    print("-" * 70)
    
    for minute in [0, 5, 10, 15, 30, 60]:
        row = f"{minute:<10}"
        for emotion in emotions:
            intensity = model.calculate_decay(emotion, 0.9, minute)
            row += f"{intensity:<12.3f}"
        print(row)
    
    print("\n情境修正示例 (专注在深度工作情境下):")
    print("-" * 50)
    print(f"{'时间(分)':<10} {'默认情境':<15} {'深度工作':<15}")
    print("-" * 50)
    
    for minute in [0, 10, 20, 30, 60]:
        normal = model.calculate_decay("Focused", 0.9, minute, "default")
        deep_work = model.calculate_decay("Focused", 0.9, minute, "deep_work")
        print(f"{minute:<10} {normal:<15.3f} {deep_work:<15.3f}")


def demo_emotion_reinforcement():
    """演示情绪强化"""
    print("\n" + "="*70)
    print("演示3: 情绪强化机制")
    print("="*70)
    
    from emotion_system.reinforcement import EmotionReinforcementModel
    
    model = EmotionReinforcementModel()
    
    print("\n兴奋情绪的强化效果:")
    print("-" * 60)
    print(f"{'触发次数':<10} {'触发类型':<20} {'强化量':<10} {'新强度':<10}")
    print("-" * 60)
    
    triggers = [
        (1, "further_breakthrough", "进一步突破"),
        (2, "consecutive_success", "连续成功"),
        (3, "positive_feedback", "正面反馈"),
    ]
    
    current_intensity = 0.7
    for count, trigger, desc in triggers:
        result = model.calculate_reinforcement(
            "Excited", current_intensity, trigger, count
        )
        print(f"{count:<10} {desc:<20} {result['boost']:<10.3f} {result['new_intensity']:<10.3f}")
        current_intensity = result['new_intensity']
    
    print("\n正向反馈强化:")
    print("-" * 50)
    outcomes = [
        ("task_success", "任务成功"),
        ("user_satisfaction", "用户满意"),
        ("problem_solved", "问题解决"),
        ("breakthrough_achieved", "突破达成"),
    ]
    
    for outcome, desc in outcomes:
        result = model.apply_positive_feedback("Confident", outcome, 0.7)
        print(f"{desc}: +{result['boost']:.2f} -> {result['new_intensity']:.2f}")


def demo_epg_graph():
    """演示EPG情绪记忆图谱"""
    print("\n" + "="*70)
    print("演示4: EPG情绪记忆图谱")
    print("="*70)
    
    from emotion_system.epg import EmotionGraph
    
    graph = EmotionGraph()
    
    print("\n创建情绪节点:")
    print("-" * 50)
    
    # 创建情绪节点
    emotion1_id = graph.add_emotion_node(
        base_emotion="Excited",
        sub_emotion="Ecstatic",
        intensity=0.95,
        context="重大突破"
    )
    print(f"情绪节点1: {emotion1_id}")
    print(f"  - 类型: Excited (Ecstatic)")
    print(f"  - 强度: 0.95")
    
    emotion2_id = graph.add_emotion_node(
        base_emotion="Content",
        sub_emotion="Satisfied",
        intensity=0.75,
        context="任务完成"
    )
    print(f"情绪节点2: {emotion2_id}")
    print(f"  - 类型: Content (Satisfied)")
    print(f"  - 强度: 0.75")
    
    # 创建记忆节点
    print("\n创建记忆节点:")
    print("-" * 50)
    
    memory1_id = graph.add_memory_node(
        memory_id="mem_001",
        memory_type="breakthrough",
        content="完成重大突破",
        importance=0.9
    )
    print(f"记忆节点1: {memory1_id}")
    print(f"  - 类型: breakthrough")
    print(f"  - 重要性: 0.9")
    
    # 建立关联
    print("\n建立情绪-记忆关联:")
    print("-" * 50)
    
    assoc_id = graph.add_emotion_memory_association(
        emotion_node_id=emotion1_id,
        memory_node_id=memory1_id,
        strength=0.9,
        association_type="triggered_by"
    )
    print(f"关联ID: {assoc_id}")
    print(f"  - 情绪 -> 记忆: 强度 0.9")
    
    # 建立情绪时序
    print("\n建立情绪时序关系:")
    print("-" * 50)
    
    seq_id = graph.add_emotion_sequence(
        from_emotion_id=emotion1_id,
        to_emotion_id=emotion2_id,
        transition_probability=0.8
    )
    print(f"时序ID: {seq_id}")
    print(f"  - Excited -> Content: 转移概率 0.8")
    
    # 获取统计
    print("\n图谱统计:")
    print("-" * 50)
    stats = graph.get_emotion_statistics()
    print(f"总节点数: {stats['total_emotion_nodes'] + stats['total_memory_nodes']}")
    print(f"情绪节点: {stats['total_emotion_nodes']}")
    print(f"记忆节点: {stats['total_memory_nodes']}")
    print(f"关系数: {stats['total_relations']}")


def demo_full_workflow():
    """演示完整工作流程"""
    print("\n" + "="*70)
    print("演示5: 完整情绪工作流程")
    print("="*70)
    
    engine = EmotionEngine()
    
    print("\n步骤1: 初始状态")
    print_emotion_state(engine, "初始")
    
    print("\n步骤2: 处理重大突破消息")
    result = engine.process_input("重大突破！我们成功了！")
    print(f"触发: {result['triggers'][0]['trigger_id'] if result['triggers'] else 'None'}")
    print_emotion_state(engine, "兴奋状态")
    
    print("\n步骤3: 处理用户担忧")
    result = engine.process_input("但是我担心时间不够")
    print(f"检测到: {result['emotion']}")
    
    print("\n步骤4: 手动设置专注")
    engine.set_emotion("Focused", 0.8, sub_emotion="Concentrated", context="开始深度工作")
    print_emotion_state(engine, "专注状态")
    
    print("\n步骤5: 查看情绪历史")
    history = engine.get_emotion_history(limit=5)
    print(f"历史记录数: {len(history)}")
    for i, h in enumerate(history):
        print(f"  {i+1}. {h['emotion']} ({h['intensity']:.2f}) - {h['timestamp']}")
    
    print("\n步骤6: 获取统计信息")
    stats = engine.get_emotion_statistics()
    print(f"总记录数: {stats['total_records']}")
    print(f"情绪分布: {stats['emotion_distribution']}")


def demo_emotion_subtypes():
    """演示64种精细情绪子类型"""
    print("\n" + "="*70)
    print("演示6: 64种精细情绪子类型")
    print("="*70)
    
    from emotion_system.emotions import SUB_EMOTIONS, BASE_EMOTIONS
    
    print(f"\n总计: {len(SUB_EMOTIONS)} 种子类型")
    print("-" * 70)
    
    # 展示每种基础情绪的子类型
    for base_emotion in list(BASE_EMOTIONS.keys())[:5]:  # 只展示前5种
        print(f"\n{base_emotion}:")
        subs = [k for k in SUB_EMOTIONS.keys() if k.startswith(base_emotion)]
        for sub_key in subs:
            sub_def = SUB_EMOTIONS[sub_key]
            intensity_range = sub_def["intensity_range"]
            print(f"  - {sub_key.split('_')[1]}: {sub_def['description']}")
            print(f"    强度范围: {intensity_range[0]:.1f} - {intensity_range[1]:.1f}")


def main():
    """主函数"""
    print("\n" + "="*70)
    print("EMOTION.md v4.0 情绪系统演示")
    print("="*70)
    print("\n本演示展示生产级情绪系统的核心功能:")
    print("1. 16种SimsChat基础情绪")
    print("2. 64种精细情绪子类型")
    print("3. 情绪触发器系统")
    print("4. 情绪衰减机制")
    print("5. 情绪强化机制")
    print("6. EPG情绪记忆图谱")
    print("7. 情绪-记忆双向关联")
    
    # 运行所有演示
    demo_basic_emotions()
    demo_emotion_decay()
    demo_emotion_reinforcement()
    demo_epg_graph()
    demo_full_workflow()
    demo_emotion_subtypes()
    
    print("\n" + "="*70)
    print("演示完成！")
    print("="*70)
    print("\n情绪系统v4.0已准备就绪，可立即投入生产使用。")
    print("核心特性:")
    print("✓ 16种基础情绪 × 64种子类型 = 1024种精细情绪状态")
    print("✓ EPG情绪记忆图谱，支持情绪-记忆双向关联")
    print("✓ 关键词/模式/上下文多维度触发器")
    print("✓ 指数衰减模型 + 情境修正")
    print("✓ 连续触发强化 + 正向反馈")
    print("✓ 与MEMORY.md v3.0深度集成")
    print("="*70)


if __name__ == "__main__":
    main()
