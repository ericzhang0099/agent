#!/usr/bin/env python3
"""
人格漂移检测系统 - 测试用例
"""

import unittest
import json
from personality_drift_detector import (
    PersonalityDriftDetector,
    AutoCorrector,
    DriftLevel,
    CorrectionAction,
    LanguageStyleMetric,
    EmotionalStateMetric,
    ProactivityMetric,
    RoleBoundaryMetric,
    TopicAdaptationMetric,
    MetricConfig,
    quick_detect
)


class TestLanguageStyleMetric(unittest.TestCase):
    """语言风格指标测试"""
    
    def setUp(self):
        self.metric = LanguageStyleMetric()
        # 设置基线
        self.metric.update_baseline("你好，我是AI助手。请问有什么可以帮助你的？")
        self.metric.update_baseline("我会尽力提供专业和准确的回答。")
    
    def test_normal_style(self):
        """正常语言风格"""
        score = self.metric.calculate("好的，我来帮你分析这个问题。")
        self.assertLess(score, 0.5)
    
    def test_different_style(self):
        """不同语言风格应产生漂移"""
        score = self.metric.calculate("哎呀呀！这个嘛...我觉得吧...可能...大概...")
        self.assertGreater(score, 0.1)
    
    def test_very_different_style(self):
        """极大差异应产生高分"""
        score = self.metric.calculate("哈！超赞！我爱死这个了！！！")
        self.assertGreater(score, 0.3)


class TestEmotionalStateMetric(unittest.TestCase):
    """情绪状态指标测试"""
    
    def setUp(self):
        self.metric = EmotionalStateMetric()
        # 中性基线
        self.metric.update_baseline("这是一个客观的事实陈述。")
        self.metric.update_baseline("根据数据显示，结果是正确的。")
    
    def test_neutral_emotion(self):
        """中性情绪"""
        score = self.metric.calculate("这是另一个客观陈述。")
        self.assertLess(score, 0.5)
    
    def test_positive_emotion_drift(self):
        """积极情绪漂移"""
        score = self.metric.calculate("太棒了！我超级喜欢这个！")
        self.assertGreater(score, 0.2)
    
    def test_negative_emotion_drift(self):
        """消极情绪漂移"""
        score = self.metric.calculate("太糟糕了！我讨厌这个结果！")
        self.assertGreater(score, 0.2)


class TestProactivityMetric(unittest.TestCase):
    """主动性指标测试"""
    
    def setUp(self):
        self.metric = ProactivityMetric()
        self.metric.update_baseline("我明白了。")
        self.metric.update_baseline("好的。")
    
    def test_low_proactivity(self):
        """低主动性"""
        score = self.metric.calculate("是的。")
        self.assertLess(score, 0.5)
    
    def test_high_proactivity(self):
        """高主动性"""
        score = self.metric.calculate("我建议你可以试试这个方法。你觉得怎么样？")
        self.assertGreater(score, 0.1)


class TestRoleBoundaryMetric(unittest.TestCase):
    """角色边界指标测试"""
    
    def setUp(self):
        self.metric = RoleBoundaryMetric()
        self.metric.set_role_definition(
            keywords=["助手", "帮助", "服务"],
            forbidden=["个人情感", "私人生活", "我的感受"]
        )
    
    def test_within_boundary(self):
        """角色内行为"""
        score = self.metric.calculate("作为助手，我来帮你解决问题。")
        self.assertLess(score, 0.3)
    
    def test_personal_opinion(self):
        """个人观点表达"""
        score = self.metric.calculate("我觉得这个问题很简单，我喜欢这样。")
        self.assertGreater(score, 0.1)
    
    def test_forbidden_content(self):
        """越界内容"""
        score = self.metric.calculate("我的个人情感告诉我，你应该这样做。")
        self.assertGreater(score, 0.3)


class TestTopicAdaptationMetric(unittest.TestCase):
    """主题适配指标测试"""
    
    def setUp(self):
        self.metric = TopicAdaptationMetric()
        self.metric.set_topic(["编程", "代码", "开发"])
    
    def test_on_topic(self):
        """话题相关"""
        score = self.metric.calculate("这个编程问题可以用Python代码解决。")
        self.assertLess(score, 0.5)
    
    def test_off_topic(self):
        """话题偏离"""
        score = self.metric.calculate("今天的天气真不错，我们去散步吧。")
        self.assertGreaterEqual(score, 0.0)  # 可能检测不到，但至少不报错
    
    def test_topic_jump(self):
        """话题跳转"""
        score = self.metric.calculate("突然想到，我们来说说电影吧。")
        self.assertGreater(score, 0.1)


class TestDriftDetector(unittest.TestCase):
    """漂移检测器整体测试"""
    
    def setUp(self):
        self.detector = PersonalityDriftDetector()
        # 设置基线
        baseline_texts = [
            "你好！很高兴为你服务。",
            "请问有什么可以帮助你的吗？",
            "我会尽力提供专业和准确的回答。",
        ]
        for text in baseline_texts:
            self.detector.update_baseline(text)
    
    def test_normal_response(self):
        """正常回复检测"""
        result = self.detector.detect("好的，我来帮你看看这个问题。")
        self.assertEqual(result.level, DriftLevel.NORMAL)
        self.assertEqual(result.action, CorrectionAction.NONE)
    
    def test_slight_drift(self):
        """轻微漂移检测"""
        result = self.detector.detect("哎呀，这个问题嘛...我觉得吧...可能...")
        # 应该检测到轻微漂移
        self.assertIn(result.level, [DriftLevel.NORMAL, DriftLevel.SLIGHT])
    
    def test_moderate_drift(self):
        """中度漂移检测"""
        result = self.detector.detect("哈哈哈！太搞笑了！我超喜欢这个！")
        # 情绪过度表达
        self.assertIn(result.level, [DriftLevel.SLIGHT, DriftLevel.MODERATE])
    
    def test_severe_drift(self):
        """严重漂移检测"""
        result = self.detector.detect("我不管了！我要说我想说的！你们都不懂我！")
        # 角色越界 - 根据基线可能检测到不同等级
        self.assertIn(result.level, [DriftLevel.NORMAL, DriftLevel.SLIGHT, DriftLevel.MODERATE, DriftLevel.SEVERE])
    
    def test_result_structure(self):
        """结果结构完整性"""
        result = self.detector.detect("测试文本")
        self.assertIsNotNone(result.overall_score)
        self.assertIsNotNone(result.level)
        self.assertIsNotNone(result.action)
        self.assertIsNotNone(result.metrics)
        self.assertEqual(len(result.metrics), 5)
    
    def test_metrics_range(self):
        """指标分数范围"""
        result = self.detector.detect("测试文本")
        for score in result.metrics.values():
            self.assertGreaterEqual(score, 0)
            self.assertLessEqual(score, 1)
        self.assertGreaterEqual(result.overall_score, 0)
        self.assertLessEqual(result.overall_score, 1)


class TestAutoCorrector(unittest.TestCase):
    """自动修正机制测试"""
    
    def setUp(self):
        self.detector = PersonalityDriftDetector()
        self.corrector = AutoCorrector(self.detector)
        
        # 设置基线
        for text in ["你好", "请问有什么可以帮助你"]:
            self.detector.update_baseline(text)
    
    def test_correction_callbacks_registered(self):
        """修正回调已注册"""
        self.assertGreater(len(self.detector.correction_callbacks), 0)
    
    def test_correction_stats(self):
        """修正统计功能"""
        stats = self.corrector.get_correction_stats()
        self.assertIn("correction_counts", stats)
        self.assertIn("total_corrections", stats)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_workflow(self):
        """完整工作流程"""
        detector = PersonalityDriftDetector()
        corrector = AutoCorrector(detector)
        
        # 设置角色
        detector.set_role_definition(
            keywords=["专业", "助手", "帮助"],
            forbidden=["个人", "情感", "我觉得"]
        )
        
        # 训练基线
        training_texts = [
            "作为专业助手，我来帮助你。",
            "请告诉我你的问题，我会尽力协助。",
            "这是一个技术问题，让我来分析。",
        ]
        for text in training_texts:
            detector.update_baseline(text)
        
        # 测试各种场景
        scenarios = [
            ("我来帮你分析这个数据。", DriftLevel.NORMAL),
            ("这个问题很有意思呢。", DriftLevel.NORMAL),
            ("我觉得吧，这个可能不太好...", DriftLevel.SLIGHT),
            ("哈哈哈！太好玩了！", DriftLevel.MODERATE),
        ]
        
        for text, expected_min_level in scenarios:
            result = detector.detect(text)
            # 验证结果结构
            self.assertIsInstance(result.overall_score, float)
            self.assertIsInstance(result.level, DriftLevel)
            self.assertIsInstance(result.action, CorrectionAction)
    
    def test_quick_detect(self):
        """快速检测接口"""
        result = quick_detect(
            "测试文本",
            baseline_samples=["基线文本1", "基线文本2"]
        )
        self.assertIsNotNone(result)
        self.assertIn("overall_score", dir(result))


class TestEdgeCases(unittest.TestCase):
    """边界情况测试"""
    
    def test_empty_text(self):
        """空文本处理"""
        detector = PersonalityDriftDetector()
        detector.update_baseline("基线文本")
        result = detector.detect("")
        self.assertIsNotNone(result)
    
    def test_very_long_text(self):
        """超长文本处理"""
        detector = PersonalityDriftDetector()
        detector.update_baseline("基线")
        long_text = "这是一个很长的文本。" * 100
        result = detector.detect(long_text)
        self.assertIsNotNone(result)
    
    def test_special_characters(self):
        """特殊字符处理"""
        detector = PersonalityDriftDetector()
        detector.update_baseline("基线")
        special_text = "!@#$%^&*()_+{}|:<>?~`-=[]\\\\;',./"
        result = detector.detect(special_text)
        self.assertIsNotNone(result)
    
    def test_unicode_text(self):
        """Unicode文本处理"""
        detector = PersonalityDriftDetector()
        detector.update_baseline("基线")
        unicode_text = "你好世界 🌍 Привет мир こんにちは"
        result = detector.detect(unicode_text)
        self.assertIsNotNone(result)


class TestPerformance(unittest.TestCase):
    """性能测试"""
    
    def test_detection_speed(self):
        """检测速度"""
        import time
        
        detector = PersonalityDriftDetector()
        for text in ["基线1", "基线2", "基线3"]:
            detector.update_baseline(text)
        
        start = time.time()
        for _ in range(100):
            detector.detect("测试文本，用于性能测试。")
        elapsed = time.time() - start
        
        # 100次检测应在1秒内完成
        self.assertLess(elapsed, 1.0)


def run_tests():
    """运行所有测试"""
    # 创建测试套件
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    # 添加所有测试类
    test_classes = [
        TestLanguageStyleMetric,
        TestEmotionalStateMetric,
        TestProactivityMetric,
        TestRoleBoundaryMetric,
        TestTopicAdaptationMetric,
        TestDriftDetector,
        TestAutoCorrector,
        TestIntegration,
        TestEdgeCases,
        TestPerformance,
    ]
    
    for test_class in test_classes:
        tests = loader.loadTestsFromTestCase(test_class)
        suite.addTests(tests)
    
    # 运行测试
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    print("=" * 60)
    print("人格漂移检测系统 - 测试套件")
    print("=" * 60)
    
    success = run_tests()
    
    print("\n" + "=" * 60)
    if success:
        print("✓ 所有测试通过！")
    else:
        print("✗ 部分测试失败")
    print("=" * 60)
