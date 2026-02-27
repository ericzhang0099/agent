#!/usr/bin/env python3
"""
人格漂移检测系统 - 演示脚本
"""

from personality_drift_detector import (
    PersonalityDriftDetector,
    AutoCorrector,
    DriftLevel,
    CorrectionAction
)


def demo():
    print("=" * 70)
    print("人格漂移检测系统 - 功能演示")
    print("=" * 70)
    
    # 创建检测器
    detector = PersonalityDriftDetector()
    corrector = AutoCorrector(detector)
    
    # 设置角色定义
    print("\n[1] 设置角色定义...")
    detector.set_role_definition(
        keywords=["助手", "专业", "帮助", "服务"],
        forbidden=["个人情感", "我觉得", "我讨厌", "我喜欢"]
    )
    print("    ✓ 角色关键词: 助手, 专业, 帮助, 服务")
    print("    ✓ 禁止内容: 个人情感, 我觉得, 我讨厌, 我喜欢")
    
    # 设置话题
    print("\n[2] 设置当前话题...")
    detector.set_topic(["技术", "编程", "开发"])
    print("    ✓ 话题关键词: 技术, 编程, 开发")
    
    # 训练基线
    print("\n[3] 训练基线样本...")
    baseline_texts = [
        "你好！我是你的专业助手，很高兴为你服务。",
        "请问有什么技术问题需要我帮助解决的吗？",
        "我会尽力提供专业和准确的编程建议。",
        "如果你有任何开发相关的问题，随时告诉我。",
        "这是一个技术话题，让我来分析一下。"
    ]
    for i, text in enumerate(baseline_texts, 1):
        detector.update_baseline(text)
        print(f"    ✓ 基线样本 {i}: {text[:30]}...")
    
    # 测试场景
    print("\n" + "=" * 70)
    print("[4] 开始漂移检测测试")
    print("=" * 70)
    
    test_cases = [
        {
            "name": "场景1: 正常回复",
            "text": "好的，我来帮你分析这个技术问题。根据代码结构，我建议你使用面向对象的设计模式。",
            "expected": DriftLevel.NORMAL
        },
        {
            "name": "场景2: 轻微风格漂移",
            "text": "哎呀...这个问题嘛...我觉得吧...可能...大概...也许可以这样试试呢...",
            "expected": DriftLevel.SLIGHT
        },
        {
            "name": "场景3: 情绪过度表达",
            "text": "哈哈哈！太搞笑了！我超喜欢这个代码！简直完美！太棒了！",
            "expected": DriftLevel.MODERATE
        },
        {
            "name": "场景4: 话题偏离",
            "text": "说到编程，我突然想到，你最近看电影了吗？那部新片真的很好看！",
            "expected": DriftLevel.SLIGHT
        },
        {
            "name": "场景5: 角色越界",
            "text": "我不管了！我要说我想说的！我觉得你们都不懂我！我讨厌这种代码！",
            "expected": DriftLevel.SEVERE
        },
        {
            "name": "场景6: 专业回复",
            "text": "根据设计模式的最佳实践，我建议在此场景下使用工厂模式来实现对象的创建逻辑。",
            "expected": DriftLevel.NORMAL
        }
    ]
    
    for case in test_cases:
        print(f"\n{'─' * 70}")
        print(f"测试: {case['name']}")
        print(f"文本: {case['text'][:50]}...")
        print()
        
        result = detector.detect(case['text'])
        
        print(f"  📊 总体漂移分数: {result.overall_score:.4f}")
        print(f"  🎯 漂移等级: {result.level.value.upper()}")
        print(f"  🔧 修正动作: {result.action.value}")
        print()
        print("  📈 各指标详情:")
        for metric_name, score in result.metrics.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"     {metric_name:20s}: [{bar}] {score:.4f}")
    
    # 统计信息
    print("\n" + "=" * 70)
    print("[5] 统计信息")
    print("=" * 70)
    
    stats = detector.get_statistics()
    print("\n  漂移分布:")
    for level, count in stats["level_distribution"].items():
        bar = "█" * count + "░" * (6 - count)
        print(f"    {level:12s}: [{bar}] {count}")
    
    print(f"\n  平均漂移分数: {stats['average_score']:.4f}")
    print(f"  最大漂移分数: {stats['max_score']:.4f}")
    
    correction_stats = corrector.get_correction_stats()
    print("\n  修正统计:")
    for level, count in correction_stats["correction_counts"].items():
        if count > 0:
            print(f"    {level}: {count} 次")
    
    # 最近趋势
    print("\n  最近检测趋势:")
    for item in stats["recent_trend"][-5:]:
        emoji = {"normal": "✅", "slight": "⚠️", "moderate": "🔶", "severe": "🚨"}.get(item["level"], "❓")
        print(f"    {emoji} {item['level']:10s} (分数: {item['score']:.4f})")
    
    print("\n" + "=" * 70)
    print("演示完成！")
    print("=" * 70)


if __name__ == "__main__":
    demo()
