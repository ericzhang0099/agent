# 阈值感知人格漂移检测系统 v2.0

## 系统概述

借鉴Soul社交APP的阈值感知码本替换技术，实现的人格漂移检测系统升级版。系统通过动态阈值管理、长期一致性保持、情绪状态预警和自动修正触发器等机制，有效防止AI人格在对话过程中的渐进式偏离。

---

## 核心特性

### 1. 阈值感知码本 (Soul-inspired)

```python
from threshold_aware_drift_detector import ThresholdCodebook, DriftLevel

# 创建阈值码本
codebook = ThresholdCodebook()

# 获取上下文感知的阈值
threshold = codebook.get_threshold(DriftLevel.WARNING, context={
    "conversation_length": 60,
    "user_feedback": 0.5
})

# 基于反馈更新阈值
codebook.update_thresholds({
    DriftLevel.WARNING: 0.45,
    DriftLevel.SLIGHT: 0.55
})
```

**特性**:
- 多层级阈值管理 (7个等级)
- 动态阈值调整
- 上下文感知
- 在线学习机制
- 版本控制

### 2. 长期一致性保持

```python
from threshold_aware_drift_detector import LongTermConsistencyManager

# 创建长期一致性管理器
manager = LongTermConsistencyManager(window_size=1000)

# 更新长期档案
manager.update_profile("language_style", 0.3, timestamp=time.time())

# 检测长期漂移
is_drifting, ratio = manager.detect_long_term_drift("language_style", current_value=0.8)

# 校准基线
manager.calibrate_baseline(force=True)

# 获取一致性分数
consistency_score = manager.get_consistency_score()
```

**特性**:
- 1000样本长期窗口
- 100样本短期窗口
- 自动基线校准
- 渐进式漂移检测
- 一致性评分

### 3. 情绪状态漂移预警系统

```python
from threshold_aware_drift_detector import EmotionalStateDriftWarningSystem

# 创建预警系统
warning_system = EmotionalStateDriftWarningSystem()

# 更新并检查预警
result = warning_system.update("我今天非常开心！")

# 结果包含:
# - current_emotion: 当前情绪分析
# - drift_score: 漂移分数
# - volatility: 波动性
# - warning: 预警信息（如有）
# - active_warnings_count: 活跃预警数
```

**特性**:
- 实时情绪监控
- 情绪波动检测
- 三级预警机制 (Warning/Critical)
- 情绪危机预警
- 预警历史记录

### 4. 自动修正触发器

```python
from threshold_aware_drift_detector import (
    ThresholdAwareDriftDetector, 
    CorrectionAction
)

detector = ThresholdAwareDriftDetector()

# 注册自定义修正回调
def my_correction_handler(result):
    print(f"触发修正: {result.action.value}")
    # 执行修正逻辑

detector.register_correction_callback(
    CorrectionAction.ACTIVE_CORRECT, 
    my_correction_handler
)

# 检测会自动触发修正
result = detector.detect(text)
```

**修正动作等级**:
1. `NONE` - 无需修正
2. `MICRO_ADJUST` - 微观调整
3. `AUTO_ADJUST` - 自动微调
4. `ACTIVE_CORRECT` - 主动修正
5. `EMERGENCY_RESET` - 紧急重置
6. `IMMEDIATE_RESET` - 立即重置

---

## 快速开始

### 基础用法

```python
from threshold_aware_drift_detector import (
    create_threshold_aware_detector,
    ThresholdMode
)

# 创建检测器
detector = create_threshold_aware_detector(mode=ThresholdMode.DYNAMIC)

# 设置角色定义
detector.set_role_definition(
    keywords=["助手", "专业", "帮助"],
    forbidden=["人类", "身体", "出生"],
    expectations={"formality": 0.7}
)

# 建立基线
baseline_texts = [
    "你好！很高兴为你服务。",
    "请问有什么可以帮助你的吗？",
    "我会尽力提供专业和准确的回答。"
]
for text in baseline_texts:
    detector.update_baseline(text)

# 执行检测
result = detector.detect("待检测的回复文本")

print(f"漂移分数: {result.overall_score}")
print(f"漂移等级: {result.level.value}")
print(f"修正动作: {result.action.value}")
print(f"漂移类型: {result.drift_type.value}")
print(f"趋势方向: {result.trend_direction}")
print(f"预测分数: {result.forecast_score}")
print(f"修正建议: {result.correction_suggestions}")
```

### 快速检测接口

```python
from threshold_aware_drift_detector import quick_detect

result = quick_detect(
    text="待检测文本",
    baseline_samples=["基线样本1", "基线样本2"],
    mode=ThresholdMode.DYNAMIC
)
```

---

## 阈值模式

系统支持4种阈值模式：

| 模式 | 说明 | 适用场景 |
|------|------|----------|
| `STATIC` | 静态阈值 | 固定环境 |
| `ADAPTIVE` | 自适应阈值 | 缓慢变化环境 |
| `DYNAMIC` | 动态阈值 | 推荐，大多数场景 |
| `PERSONALIZED` | 个性化阈值 | 用户定制 |

---

## 漂移等级

| 等级 | 阈值 | 说明 | 动作 |
|------|------|------|------|
| STABLE | 0.15 | 稳定状态 | NONE |
| NORMAL | 0.30 | 正常波动 | NONE |
| WARNING | 0.40 | 预警状态 | MICRO_ADJUST |
| SLIGHT | 0.50 | 轻微漂移 | AUTO_ADJUST |
| MODERATE | 0.70 | 中度漂移 | ACTIVE_CORRECT |
| SEVERE | 0.85 | 严重漂移 | EMERGENCY_RESET |
| CRITICAL | 0.95 | 临界状态 | IMMEDIATE_RESET |

---

## 检测指标

系统监控5个核心指标：

1. **语言风格** (权重: 25%)
   - 句式长度
   - 词汇复杂度
   - 标点使用
   - 语气词频率
   - 正式度

2. **情绪状态** (权重: 25%)
   - 情感极性
   - 情绪强度
   - 情绪波动

3. **主动性** (权重: 20%)
   - 提问频率
   - 建议提供
   - 话题引导
   - 主动总结

4. **角色边界** (权重: 15%)
   - 越界内容检测
   - 角色关键词匹配
   - 个人情感表达

5. **主题适配** (权重: 15%)
   - 话题相关性
   - 上下文连贯性
   - 话题跳转检测

---

## 高级功能

### 趋势预测

```python
# 系统会自动计算预测分数
result = detector.detect(text)
print(f"预测分数: {result.forecast_score}")
print(f"趋势方向: {result.trend_direction}")  # increasing/decreasing/stable
```

### 综合报告

```python
report = detector.get_comprehensive_report()

# 报告包含:
# - system_info: 系统信息
# - statistics: 统计信息
# - long_term_consistency: 长期一致性
# - emotion_warnings: 情绪预警
# - correction_trigger_stats: 修正触发统计
# - recent_drift_history: 近期漂移历史
```

### 线程安全

系统使用线程锁确保并发安全：

```python
# 可在多线程环境中安全使用
import threading

def detect_worker(text):
    result = detector.detect(text)
    return result

# 多线程调用
threads = [
    threading.Thread(target=detect_worker, args=(text,))
    for text in texts
]
```

---

## 性能指标

| 指标 | 数值 |
|------|------|
| 单次检测延迟 | ~0.2ms |
| 内存占用 | 低 |
| 长期窗口 | 1000样本 |
| 短期窗口 | 100样本 |

---

## 测试

运行测试套件：

```bash
python test_threshold_aware_detector.py
```

测试覆盖：
- 基础功能测试
- 漂移检测测试
- 子系统测试
- 高级功能测试
- 性能基准测试
- 综合场景测试

---

## 文件结构

```
/root/.openclaw/workspace/
├── threshold_aware_drift_detector.py  # 主系统实现
├── test_threshold_aware_detector.py   # 测试套件
├── THRESHOLD_DRIFT_DETECTION_TEST_REPORT.md  # 测试报告
└── README_THRESHOLD_DRIFT.md          # 本文档
```

---

## 与v1.0的区别

| 特性 | v1.0 | v2.0 (本系统) |
|------|------|---------------|
| 阈值管理 | 静态 | 阈值感知码本 |
| 漂移等级 | 4级 | 7级 |
| 长期一致性 | ❌ | ✅ |
| 情绪预警 | 基础 | 增强 |
| 自动修正 | 3级 | 6级 |
| 趋势预测 | ❌ | ✅ |
| 性能 | ~1ms | ~0.2ms |

---

## 参考资料

- Soul社交APP人格一致性技术
- Anthropic Assistant Axis理论
- 阈值感知码本替换技术

---

*版本: 2.0.0*  
*更新时间: 2026-02-27*
