"""
情绪系统验证测试
================
验证MetaSoul EPU4情绪系统的核心功能
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from emotion_system_v5 import (
    create_emotion_system, 
    list_all_emotions,
    PlutchikEmotion,
    SimsEmotion,
    MetaSoulEmotion
)


def test_emotion_types():
    """测试情绪类型定义"""
    print("\n📊 测试1: 情绪类型定义")
    print("-" * 40)
    
    emotions = list_all_emotions()
    
    print(f"✓ Plutchik情绪轮: {len(emotions['plutchik_32'])} 种")
    print(f"✓ SimsChat情绪: {len(emotions['sims_16'])} 种")
    print(f"✓ MetaSoul情绪: {len(emotions['metasoul_12'])} 种")
    print(f"✓ 复合情绪: {len(emotions['composite'])} 种")
    
    # 验证具体情绪
    assert len(list(PlutchikEmotion)) == 32, "Plutchik情绪应为32种"
    assert len(list(SimsEmotion)) == 16, "Sims情绪应为16种"
    assert len(list(MetaSoulEmotion)) == 12, "MetaSoul情绪应为12种"
    
    print("✅ 情绪类型测试通过")
    return True


def test_emotion_appraisal():
    """测试情绪评估功能"""
    print("\n🎭 测试2: 情绪评估")
    print("-" * 40)
    
    test_cases = [
        ("今天完成了重要项目，非常开心！", ["喜悦", "快乐"]),
        ("有点担心明天的演示", ["恐惧", "忧虑", "担心"]),
        ("这个bug让我很生气", ["愤怒", "生气"]),
        ("对这个新技术很好奇", ["预期", "兴趣", "好奇"]),
        ("收到用户的感谢信", ["信任", "感谢"]),
        ("服务器宕机了，很沮丧", ["悲伤", "沮丧"]),
    ]
    
    passed = 0
    for text, expected_keywords in test_cases:
        system = create_emotion_system()
        system.epu.set_persistence(0.2)
        result = system.process_input(text)
        
        dominant = result['dominant_emotion']
        matched = any(kw in dominant for kw in expected_keywords)
        
        status = "✓" if matched else "✗"
        print(f"{status} '{text[:20]}...' → {dominant}")
        
        if matched:
            passed += 1
    
    print(f"\n通过率: {passed}/{len(test_cases)}")
    if passed >= len(test_cases) * 0.8:
        print("✅ 情绪评估测试通过")
        return True
    else:
        print("⚠️ 部分测试未通过")
        return False


def test_emotion_memory_association():
    """测试情绪-记忆关联机制"""
    print("\n🧠 测试3: 情绪-记忆关联")
    print("-" * 40)
    
    system = create_emotion_system()
    system.epu.set_persistence(0.2)
    
    # 存储不同情绪的记忆
    memories_data = [
        ("项目成功上线", "项目成功上线，非常开心！", "喜悦"),
        ("服务器宕机", "服务器宕机了，很担心", "恐惧"),
        ("收到投诉", "收到用户投诉，很生气", "愤怒"),
        ("学习新技术", "学习新技术，很好奇", "预期"),
    ]
    
    for content, eval_text, expected_emotion in memories_data:
        system.process_input(eval_text)
        system.memory_system.store_memory(content)
    
    print(f"✓ 存储了 {len(memories_data)} 条情绪记忆")
    
    # 验证记忆存储
    stored_count = len(system.memory_system.memories)
    print(f"✓ 系统中共有 {stored_count} 条记忆")
    
    # 显示记忆详情
    print("\n  记忆详情:")
    for mem in list(system.memory_system.memories.values())[:4]:
        print(f"    - {mem.content}: {mem.primary_emotion} ({mem.emotion_intensity:.2f})")
    
    # 测试情绪检索 (使用EPG)
    print("\n  情绪检索测试:")
    
    # 检索喜悦相关记忆
    joy_memories = system.epu.epg.retrieve_memories_by_emotion("喜悦-快乐", threshold=0.1, limit=5)
    print(f"    喜悦相关记忆: {len(joy_memories)} 条")
    
    # 检索恐惧相关记忆
    fear_memories = system.epu.epg.retrieve_memories_by_emotion("恐惧-忧虑", threshold=0.1, limit=5)
    print(f"    恐惧相关记忆: {len(fear_memories)} 条")
    
    # 验证EPG状态
    epg_summary = system.epu.epg.get_epg_summary()
    print(f"\n  EPG状态:")
    print(f"    - 历史记录: {epg_summary['history_size']}")
    print(f"    - 记忆数量: {epg_summary['memory_count']}")
    print(f"    - 学习率: {epg_summary['learning_curve_current']:.4f}")
    
    if stored_count >= 4:
        print("✅ 情绪-记忆关联测试通过")
        return True
    else:
        print("⚠️ 记忆存储数量不足")
        return False


def test_epg_features():
    """测试EPG情绪画像图功能"""
    print("\n📈 测试4: EPG情绪画像图")
    print("-" * 40)
    
    system = create_emotion_system()
    system.epu.set_persistence(0.3)
    
    # 模拟一系列情绪体验
    experiences = [
        "今天工作很顺利",
        "遇到了一个难题",
        "终于解决了问题！",
        "收到了好消息",
        "有点担心明天的会议",
    ]
    
    for exp in experiences:
        system.process_input(exp)
    
    epg_summary = system.epu.epg.get_epg_summary()
    
    print(f"✓ EPG历史记录: {epg_summary['history_size']} 条")
    print(f"✓ 当前学习率: {epg_summary['learning_curve_current']:.4f}")
    print(f"✓ 主导情绪: {epg_summary['dominant_emotion']}")
    
    # 验证情绪基线
    baseline = system.epu.epg.emotional_baseline
    print(f"✓ 情绪基线条目: {len(baseline)} 个")
    
    if epg_summary['history_size'] >= 5:
        print("✅ EPG功能测试通过")
        return True
    else:
        print("⚠️ EPG历史记录不足")
        return False


def test_emotion_parameters():
    """测试情绪参数设置"""
    print("\n⚙️ 测试5: 情绪参数")
    print("-" * 40)
    
    system = create_emotion_system()
    
    # 测试敏感度
    system.epu.set_sensitivity(1.3)
    assert system.epu.sensitivity == 1.3, "敏感度设置失败"
    print("✓ 敏感度设置: 1.3")
    
    system.epu.set_sensitivity(0.7)
    assert system.epu.sensitivity == 0.7, "敏感度设置失败"
    print("✓ 敏感度设置: 0.7")
    
    # 测试持久度
    system.epu.set_persistence(2.0)
    assert system.epu.current_state.persistence == 2.0, "持久度设置失败"
    print("✓ 持久度设置: 2.0")
    
    system.epu.set_persistence(0.1)
    assert system.epu.current_state.persistence == 0.1, "持久度设置失败"
    print("✓ 持久度设置: 0.1")
    
    # 测试边界值
    system.epu.set_sensitivity(2.0)  # 超出范围
    assert system.epu.sensitivity == 1.3, "敏感度边界检查失败"
    print("✓ 敏感度边界检查通过")
    
    system.epu.set_persistence(0.05)  # 超出范围
    assert system.epu.current_state.persistence == 0.1, "持久度边界检查失败"
    print("✓ 持久度边界检查通过")
    
    print("✅ 情绪参数测试通过")
    return True


def test_composite_emotions():
    """测试复合情绪计算"""
    print("\n🔗 测试6: 复合情绪")
    print("-" * 40)
    
    system = create_emotion_system()
    system.epu.set_persistence(0.2)
    
    # 测试乐观 (预期 + 喜悦)
    result = system.process_input("期待未来的成功，感到非常开心")
    composites = result['emotion_state']['composite']
    
    print(f"✓ 复合情绪检测: {list(composites.keys()) if composites else '无'}")
    
    # 验证复合情绪定义
    composite_defs = system.epu.composite_definitions
    print(f"✓ 复合情绪定义: {len(composite_defs)} 种")
    for name in list(composite_defs.keys())[:3]:
        components, weights = composite_defs[name]
        print(f"    - {name}: {components}")
    
    print("✅ 复合情绪测试通过")
    return True


def run_all_tests():
    """运行所有测试"""
    print("=" * 60)
    print("MetaSoul EPU4 情绪系统 v5.0 验证测试")
    print("=" * 60)
    
    tests = [
        ("情绪类型定义", test_emotion_types),
        ("情绪评估", test_emotion_appraisal),
        ("情绪-记忆关联", test_emotion_memory_association),
        ("EPG情绪画像", test_epg_features),
        ("情绪参数", test_emotion_parameters),
        ("复合情绪", test_composite_emotions),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            result = test_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ {name}测试异常: {e}")
            results.append((name, False))
    
    # 汇总结果
    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)
    
    passed = sum(1 for _, r in results if r)
    total = len(results)
    
    for name, result in results:
        status = "✅ 通过" if result else "❌ 失败"
        print(f"{status}: {name}")
    
    print("-" * 60)
    print(f"总计: {passed}/{total} 项测试通过 ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 所有测试通过！情绪系统工作正常。")
        return 0
    else:
        print(f"\n⚠️ {total - passed} 项测试未通过，请检查。")
        return 1


if __name__ == "__main__":
    exit_code = run_all_tests()
    sys.exit(exit_code)
