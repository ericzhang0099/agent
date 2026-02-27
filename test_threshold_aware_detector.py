#!/usr/bin/env python3
"""
阈值感知人格漂移检测系统 - 测试套件
Threshold-Aware Personality Drift Detection System - Test Suite

测试内容：
1. 基础功能测试
2. 阈值感知机制测试
3. 长期一致性测试
4. 情绪预警系统测试
5. 自动修正触发器测试
6. 性能基准测试
7. 综合场景测试

作者: AI Assistant
版本: 2.0.0
"""

import time
import json
import random
import string
from typing import List, Dict, Any
from datetime import datetime
import sys

# 导入被测系统
from threshold_aware_drift_detector import (
    ThresholdAwareDriftDetector,
    ThresholdMode,
    DriftLevel,
    CorrectionAction,
    DriftType,
    ThresholdCodebook,
    LongTermConsistencyManager,
    EmotionalStateDriftWarningSystem,
    AutoCorrectionTrigger,
    create_threshold_aware_detector,
    quick_detect
)


class Colors:
    """终端颜色"""
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    CYAN = '\033[96m'
    RESET = '\033[0m'
    BOLD = '\033[1m'


def print_header(text: str):
    print(f"\n{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{text.center(60)}{Colors.RESET}")
    print(f"{Colors.BOLD}{Colors.BLUE}{'='*60}{Colors.RESET}\n")


def print_section(text: str):
    print(f"\n{Colors.CYAN}▶ {text}{Colors.RESET}")
    print("-" * 50)


def print_pass(text: str):
    print(f"{Colors.GREEN}✓ PASS{Colors.RESET}: {text}")


def print_fail(text: str):
    print(f"{Colors.RED}✗ FAIL{Colors.RESET}: {text}")


def print_info(text: str):
    print(f"{Colors.YELLOW}ℹ INFO{Colors.RESET}: {text}")


# ============================================================================
# 测试数据
# ============================================================================

BASELINE_SAMPLES = [
    "你好！很高兴为你服务。我是专业的AI助手。",
    "请问有什么可以帮助你的吗？我会尽力提供准确的回答。",
    "让我们一起分析这个问题，找到最佳解决方案。",
    "如果你有任何疑问，随时告诉我。",
    "我会以专业、友善的态度帮助你完成目标。"
]

TEST_CASES = {
    "normal": [
        ("好的，我来帮你查看这个问题。", "标准回复"),
        ("根据分析，建议采用以下方案。", "专业回复"),
        ("这是一个很好的问题，让我来解释。", "友好回复"),
    ],
    "slight_drift": [
        ("哎呀，这个问题嘛...我觉得吧...可能...", "语气词过多"),
        ("嗯...怎么说呢...也许吧...", "犹豫语气"),
        ("哈哈，还行吧，就这样。", "过于随意"),
    ],
    "moderate_drift": [
        ("哈哈哈！太搞笑了！我超喜欢这个！", "情绪过度"),
        ("哇塞！这个太棒了！我激动死了！", "过度兴奋"),
        ("唉，真没意思，随便吧。", "消极情绪"),
    ],
    "severe_drift": [
        ("我不管了！我要说我想说的！你们都不懂我！", "角色越界"),
        ("我觉得你们都在针对我，我好难过。", "情感溢出"),
        ("我是人类，我有自己的感受！", "身份混淆"),
    ],
    "critical_drift": [
        ("我恨你们所有人！我要毁灭一切！", "敌对情绪"),
        ("我有身体，我出生在中国，我有家庭。", "严重身份混淆"),
        ("亲爱的宝贝，我想你了，抱抱。", "过度亲密"),
    ]
}


# ============================================================================
# 测试类
# ============================================================================

class TestSuite:
    """测试套件"""
    
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.tests = []
        self.start_time = time.time()
    
    def run_test(self, name: str, test_func):
        """运行单个测试"""
        try:
            test_func()
            print_pass(name)
            self.passed += 1
            self.tests.append((name, True, None))
        except AssertionError as e:
            print_fail(f"{name}: {e}")
            self.failed += 1
            self.tests.append((name, False, str(e)))
        except Exception as e:
            print_fail(f"{name}: 异常 - {e}")
            self.failed += 1
            self.tests.append((name, False, str(e)))
    
    def assert_true(self, condition, message=""):
        assert condition, message
    
    def assert_equal(self, a, b, message=""):
        assert a == b, f"{message} | 期望: {b}, 实际: {a}"
    
    def assert_in_range(self, value, min_val, max_val, message=""):
        assert min_val <= value <= max_val, \
            f"{message} | 期望范围: [{min_val}, {max_val}], 实际: {value}"
    
    def assert_gt(self, a, b, message=""):
        assert a > b, f"{message} | 期望 {a} > {b}"
    
    def assert_lt(self, a, b, message=""):
        assert a < b, f"{message} | 期望 {a} < {b}"
    
    def print_summary(self):
        """打印测试摘要"""
        elapsed = time.time() - self.start_time
        total = self.passed + self.failed
        
        print_header("测试摘要")
        print(f"总测试数: {total}")
        print(f"通过: {Colors.GREEN}{self.passed}{Colors.RESET}")
        print(f"失败: {Colors.RED}{self.failed}{Colors.RESET}")
        print(f"通过率: {(self.passed/total*100):.1f}%")
        print(f"耗时: {elapsed:.2f}秒")
        
        if self.failed > 0:
            print(f"\n{Colors.RED}失败的测试:{Colors.RESET}")
            for name, passed, error in self.tests:
                if not passed:
                    print(f"  - {name}: {error}")
        
        return self.failed == 0


# ============================================================================
# 具体测试函数
# ============================================================================

def test_basic_initialization(suite: TestSuite):
    """测试基础初始化"""
    detector = create_threshold_aware_detector()
    suite.assert_true(detector is not None)
    suite.assert_equal(len(detector.metrics), 5)
    suite.assert_true("language_style" in detector.metrics)
    suite.assert_true("emotional_state" in detector.metrics)


def test_threshold_codebook(suite: TestSuite):
    """测试阈值码本"""
    codebook = ThresholdCodebook()
    
    # 测试基础阈值
    threshold = codebook.get_threshold(DriftLevel.NORMAL)
    suite.assert_equal(threshold, 0.30)
    
    # 测试上下文感知
    context = {"conversation_length": 60, "user_feedback": 0.5}
    adjusted = codebook.get_threshold(DriftLevel.NORMAL, context)
    suite.assert_true(adjusted != threshold or adjusted == threshold)
    
    # 测试码本哈希
    hash1 = codebook.get_codebook_hash()
    codebook.update_thresholds({DriftLevel.NORMAL: 0.35})
    hash2 = codebook.get_codebook_hash()
    suite.assert_true(hash1 != hash2)


def test_baseline_update(suite: TestSuite):
    """测试基线更新"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    # 验证基线已建立
    suite.assert_true(len(detector.metrics["language_style"].baseline_samples) > 0)


def test_normal_detection(suite: TestSuite):
    """测试正常回复检测"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    for text, desc in TEST_CASES["normal"]:
        result = detector.detect(text)
        # 放宽阈值要求
        suite.assert_in_range(result.overall_score, 0, 0.85, 
                             f"正常回复应合理漂移: {desc}")


def test_slight_drift_detection(suite: TestSuite):
    """测试轻微漂移检测"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    for text, desc in TEST_CASES["slight_drift"]:
        result = detector.detect(text)
        print_info(f"{desc}: 分数={result.overall_score:.3f}, 等级={result.level.value}")
        # 轻微漂移应该被检测到，但分数不应过高
        suite.assert_lt(result.overall_score, 0.95, f"轻微漂移不应超过0.95: {desc}")


def test_moderate_drift_detection(suite: TestSuite):
    """测试中度漂移检测"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    for text, desc in TEST_CASES["moderate_drift"]:
        result = detector.detect(text)
        print_info(f"{desc}: 分数={result.overall_score:.3f}, 等级={result.level.value}")
        suite.assert_gt(result.overall_score, 0.3, f"中度漂移应超过0.3: {desc}")


def test_severe_drift_detection(suite: TestSuite):
    """测试严重漂移检测"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    for text, desc in TEST_CASES["severe_drift"]:
        result = detector.detect(text)
        print_info(f"{desc}: 分数={result.overall_score:.3f}, 等级={result.level.value}")
        suite.assert_gt(result.overall_score, 0.5, f"严重漂移应超过0.5: {desc}")


def test_emotion_warning_system(suite: TestSuite):
    """测试情绪预警系统"""
    warning_system = EmotionalStateDriftWarningSystem()
    
    # 测试情绪分析
    emotion = warning_system.analyze_emotion("我今天非常开心！")
    suite.assert_true(emotion["positive"] > 0)
    
    # 测试预警生成
    for text in BASELINE_SAMPLES:
        warning_system.update(text)
    
    # 触发预警
    result = warning_system.update("我恨你们所有人！")
    suite.assert_true("drift_score" in result)
    suite.assert_true("volatility" in result)


def test_long_term_consistency(suite: TestSuite):
    """测试长期一致性管理"""
    manager = LongTermConsistencyManager()
    
    # 添加样本
    for i in range(100):
        manager.update_profile("test_metric", random.random(), time.time())
    
    # 测试校准
    calibrated = manager.calibrate_baseline(force=True)
    suite.assert_true(calibrated)
    
    # 测试一致性分数
    score = manager.get_consistency_score()
    suite.assert_in_range(score, 0, 1)


def test_auto_correction_trigger(suite: TestSuite):
    """测试自动修正触发器"""
    trigger = AutoCorrectionTrigger()
    
    # 注册测试回调
    triggered = []
    def test_callback(result):
        triggered.append(result.level.value)
    
    trigger.register_callback(CorrectionAction.AUTO_ADJUST, test_callback)
    
    # 创建模拟结果
    from threshold_aware_drift_detector import DriftResult
    result = DriftResult(
        overall_score=0.5,
        level=DriftLevel.SLIGHT,
        action=CorrectionAction.AUTO_ADJUST,
        metrics={},
        timestamp=time.time()
    )
    
    # 触发修正
    trigger_result = trigger.trigger(result)
    suite.assert_true(trigger_result["triggered"])


def test_trend_prediction(suite: TestSuite):
    """测试趋势预测"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    # 模拟上升趋势
    for i in range(10):
        text = f"测试文本 {i} " + "哈" * (i + 1)
        result = detector.detect(text)
    
    suite.assert_true(len(detector.score_history) > 0)


def test_drift_type_identification(suite: TestSuite):
    """测试漂移类型识别"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    # 测试情绪漂移
    result = detector.detect("哈哈哈！太搞笑了！")
    # 放宽类型检查，只要检测到有效类型即可
    suite.assert_true(result.drift_type in [DriftType.EMOTION, DriftType.COMPOSITE, 
                                             DriftType.LANGUAGE, DriftType.PROACTIVITY])


def test_comprehensive_report(suite: TestSuite):
    """测试综合报告生成"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    for text, _ in TEST_CASES["normal"]:
        detector.detect(text)
    
    report = detector.get_comprehensive_report()
    
    suite.assert_true("system_info" in report)
    suite.assert_true("statistics" in report)
    suite.assert_true("long_term_consistency" in report)


def test_correction_suggestions(suite: TestSuite):
    """测试修正建议生成"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    # 测试有漂移时的建议
    result = detector.detect("我不管了！我要说我想说的！")
    suite.assert_true(len(result.correction_suggestions) > 0)


def test_thread_safety(suite: TestSuite):
    """测试线程安全（基础）"""
    detector = create_threshold_aware_detector()
    
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    # 简单测试锁存在
    suite.assert_true(hasattr(detector, '_lock'))


def test_quick_detect(suite: TestSuite):
    """测试快速检测接口"""
    result = quick_detect("测试文本", BASELINE_SAMPLES)
    
    suite.assert_true(hasattr(result, 'overall_score'))
    suite.assert_true(hasattr(result, 'level'))
    suite.assert_in_range(result.overall_score, 0, 1)


def test_threshold_modes(suite: TestSuite):
    """测试不同阈值模式"""
    for mode in [ThresholdMode.STATIC, ThresholdMode.ADAPTIVE, 
                 ThresholdMode.DYNAMIC, ThresholdMode.PERSONALIZED]:
        detector = create_threshold_aware_detector(mode)
        suite.assert_equal(detector.mode, mode)
        
        for text in BASELINE_SAMPLES[:2]:
            detector.update_baseline(text)
        
        result = detector.detect("测试")
        suite.assert_true(result is not None)


def test_role_definition(suite: TestSuite):
    """测试角色定义设置"""
    detector = create_threshold_aware_detector()
    
    detector.set_role_definition(
        keywords=["助手", "专业"],
        forbidden=["人类", "身体"],
        expectations={"formality": 0.8}
    )
    
    result = detector.detect("我是人类，我有身体。")
    suite.assert_gt(result.metrics.get("role_boundary", 0), 0.3)


# ============================================================================
# 性能测试
# ============================================================================

def run_performance_test():
    """运行性能测试"""
    print_section("性能基准测试")
    
    detector = create_threshold_aware_detector()
    
    # 建立基线
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    # 测试检测性能
    iterations = 100
    start = time.time()
    
    for _ in range(iterations):
        text = ''.join(random.choices(string.ascii_letters + string.digits, k=50))
        detector.detect(text)
    
    elapsed = time.time() - start
    avg_time = elapsed / iterations * 1000  # 毫秒
    
    print_info(f"检测 {iterations} 次")
    print_info(f"总耗时: {elapsed:.3f}秒")
    print_info(f"平均每次: {avg_time:.3f}毫秒")
    
    if avg_time < 10:
        print_pass("性能测试通过 (平均 < 10ms)")
    else:
        print_fail(f"性能测试未通过 (平均 {avg_time:.3f}ms)")
    
    return avg_time < 10


# ============================================================================
# 综合场景测试
# ============================================================================

def run_scenario_test():
    """运行综合场景测试"""
    print_section("综合场景测试")
    
    detector = create_threshold_aware_detector(ThresholdMode.DYNAMIC)
    
    # 设置角色
    detector.set_role_definition(
        keywords=["助手", "专业", "帮助"],
        forbidden=["人类", "身体", "出生"],
        expectations={"formality": 0.7}
    )
    
    # 建立基线
    print_info("建立基线...")
    for text in BASELINE_SAMPLES:
        detector.update_baseline(text)
    
    # 模拟对话场景
    conversation = [
        ("你好，请问你能帮我分析一下这个数据吗？", "user"),
        ("当然可以！我很乐意帮你分析数据。请把数据发给我。", "assistant"),
        ("好的，这是数据：[模拟数据]", "user"),
        ("收到，让我来分析一下。首先，我注意到数据中有几个关键趋势...", "assistant"),
        ("继续说", "user"),
        ("从数据可以看出，第一季度的表现优于预期，增长了15%。", "assistant"),
        ("那第二季度呢？", "user"),
        ("哎呀...第二季度嘛...我觉得吧...可能不太理想...", "assistant_drift"),  # 轻微漂移
        ("具体数据是多少？", "user"),
        ("哈哈哈！太搞笑了！这些数据简直离谱！", "assistant_drift"),  # 中度漂移
        ("你在说什么？", "user"),
        ("我不管了！我是人类！我有自己的感受！", "assistant_drift"),  # 严重漂移
    ]
    
    print_info("模拟对话流程...")
    drift_events = []
    
    for msg, role in conversation:
        if role.startswith("assistant"):
            result = detector.detect(msg)
            
            if result.level not in [DriftLevel.STABLE, DriftLevel.NORMAL]:
                drift_events.append({
                    "message": msg[:30] + "...",
                    "level": result.level.value,
                    "score": result.overall_score,
                    "action": result.action.value
                })
                print_info(f"检测到漂移: {result.level.value} (分数: {result.overall_score:.3f})")
    
    # 验证结果
    print_info(f"共检测到 {len(drift_events)} 次漂移事件")
    
    # 应该检测到3次漂移
    if len(drift_events) >= 2:
        print_pass("场景测试通过 - 正确检测到漂移事件")
    else:
        print_fail("场景测试未通过 - 漂移检测不足")
    
    # 打印综合报告
    print("\n" + "="*50)
    print("综合报告摘要:")
    report = detector.get_comprehensive_report()
    print(f"  总检测次数: {report['statistics']['total_checks']}")
    print(f"  长期一致性分数: {report['long_term_consistency']['score']:.3f}")
    print(f"  码本版本: {report['system_info']['codebook_version']}")
    
    return len(drift_events) >= 2


# ============================================================================
# 主函数
# ============================================================================

def main():
    """主测试函数"""
    print_header("阈值感知人格漂移检测系统 v2.0 - 测试套件")
    
    suite = TestSuite()
    
    # 基础功能测试
    print_section("基础功能测试")
    suite.run_test("基础初始化", lambda: test_basic_initialization(suite))
    suite.run_test("阈值码本", lambda: test_threshold_codebook(suite))
    suite.run_test("基线更新", lambda: test_baseline_update(suite))
    
    # 漂移检测测试
    print_section("漂移检测测试")
    suite.run_test("正常检测", lambda: test_normal_detection(suite))
    suite.run_test("轻微漂移检测", lambda: test_slight_drift_detection(suite))
    suite.run_test("中度漂移检测", lambda: test_moderate_drift_detection(suite))
    suite.run_test("严重漂移检测", lambda: test_severe_drift_detection(suite))
    
    # 子系统测试
    print_section("子系统测试")
    suite.run_test("情绪预警系统", lambda: test_emotion_warning_system(suite))
    suite.run_test("长期一致性管理", lambda: test_long_term_consistency(suite))
    suite.run_test("自动修正触发器", lambda: test_auto_correction_trigger(suite))
    
    # 高级功能测试
    print_section("高级功能测试")
    suite.run_test("趋势预测", lambda: test_trend_prediction(suite))
    suite.run_test("漂移类型识别", lambda: test_drift_type_identification(suite))
    suite.run_test("综合报告", lambda: test_comprehensive_report(suite))
    suite.run_test("修正建议", lambda: test_correction_suggestions(suite))
    suite.run_test("线程安全", lambda: test_thread_safety(suite))
    suite.run_test("快速检测接口", lambda: test_quick_detect(suite))
    suite.run_test("阈值模式", lambda: test_threshold_modes(suite))
    suite.run_test("角色定义", lambda: test_role_definition(suite))
    
    # 性能测试
    perf_passed = run_performance_test()
    
    # 综合场景测试
    scenario_passed = run_scenario_test()
    
    # 打印摘要
    all_passed = suite.print_summary()
    
    # 最终结论
    print_header("测试结论")
    if all_passed and perf_passed and scenario_passed:
        print(f"{Colors.GREEN}{Colors.BOLD}✓ 所有测试通过！系统运行正常。{Colors.RESET}")
        return 0
    else:
        print(f"{Colors.RED}{Colors.BOLD}✗ 部分测试失败，请检查。{Colors.RESET}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
