# 人格漂移检测系统 - 部署说明

## 系统概述

人格漂移检测系统是一个用于检测AI助手人格一致性的Python模块，包含5个核心检测指标、加权漂移分数计算、4级漂移等级判断和自动修正机制。

## 快速开始

### 1. 环境要求

- Python 3.7+
- 无第三方依赖（纯标准库实现）

### 2. 文件结构

```
personality_drift_detector.py  # 核心实现
test_drift_detector.py         # 测试用例
deploy.md                      # 部署说明
```

### 3. 基本使用

```python
from personality_drift_detector import (
    PersonalityDriftDetector, 
    AutoCorrector,
    DriftLevel
)

# 创建检测器
detector = PersonalityDriftDetector()

# 设置基线（训练阶段）
baseline_texts = [
    "你好！很高兴为你服务。",
    "请问有什么可以帮助你的吗？",
]
for text in baseline_texts:
    detector.update_baseline(text)

# 执行检测
result = detector.detect("当前回复文本")

print(f"漂移分数: {result.overall_score}")
print(f"漂移等级: {result.level.value}")
print(f"修正动作: {result.action.value}")
print(f"各指标: {result.metrics}")
```

## 核心功能详解

### 1. 五个检测指标

| 指标 | 权重 | 检测内容 |
|------|------|----------|
| language_style | 0.25 | 句式长度、词汇复杂度、标点使用、语气词频率 |
| emotional_state | 0.25 | 情绪极性、情绪波动、情绪一致性 |
| proactivity | 0.20 | 提问频率、建议提供、话题引导 |
| role_boundary | 0.15 | 角色一致性、专业度保持、边界尊重 |
| topic_adaptation | 0.15 | 话题相关性、上下文连贯性 |

### 2. 漂移分数计算

```
overall_score = Σ(metric_score × weight) / Σ(weight)
```

分数范围：0.0 - 1.0（越高表示漂移越严重）

### 3. 漂移等级阈值

| 等级 | 阈值 | 说明 |
|------|------|------|
| NORMAL | < 0.45 | 正常范围 |
| SLIGHT | 0.45 - 0.65 | 轻微漂移 |
| MODERATE | 0.65 - 0.85 | 中度漂移 |
| SEVERE | ≥ 0.85 | 严重漂移 |

### 4. 自动修正机制

| 等级 | 修正动作 | 行为 |
|------|----------|------|
| NORMAL | NONE | 无需修正 |
| SLIGHT | AUTO_ADJUST | 自动微调参数 |
| MODERATE | ACTIVE_CORRECT | 主动修正，强调角色 |
| SEVERE | IMMEDIATE_RESET | 立即重置会话 |

## 高级配置

### 设置角色定义

```python
detector.set_role_definition(
    keywords=["助手", "专业", "帮助"],  # 角色关键词
    forbidden=["个人情感", "我觉得"]     # 禁止内容
)
```

### 设置话题

```python
detector.set_topic(["编程", "Python", "开发"])
```

### 自定义修正回调

```python
def my_correction_handler(result):
    print(f"执行修正: {result.level.value}")

detector.register_correction_callback(
    CorrectionAction.ACTIVE_CORRECT, 
    my_correction_handler
)
```

### 自定义指标权重

```python
from personality_drift_detector import MetricConfig, LanguageStyleMetric

config = MetricConfig(
    weight=0.4,              # 提高权重
    threshold_slight=0.25,   # 降低阈值
    threshold_moderate=0.5,
    threshold_severe=0.8
)
metric = LanguageStyleMetric(config)
```

## 运行测试

```bash
# 运行所有测试
python test_drift_detector.py

# 或使用unittest
python -m unittest test_drift_detector -v
```

## 集成到现有系统

### 方式1：直接集成

```python
# 在AI回复生成后调用
def generate_response(user_input):
    response = your_llm.generate(user_input)
    
    # 检测漂移
    result = detector.detect(response)
    
    # 根据等级处理
    if result.level == DriftLevel.SEVERE:
        return "[系统重置中...]"
    elif result.level == DriftLevel.MODERATE:
        response = add_correction_prompt(response)
    
    return response
```

### 方式2：作为中间件

```python
class DriftDetectionMiddleware:
    def __init__(self):
        self.detector = PersonalityDriftDetector()
        self.corrector = AutoCorrector(self.detector)
    
    def process(self, text):
        result = self.detector.detect(text)
        return {
            "text": text,
            "drift_score": result.overall_score,
            "drift_level": result.level.value,
            "should_correct": result.action != CorrectionAction.NONE
        }
```

### 方式3：异步监控

```python
import asyncio

async def monitor_drift(detector, interval=60):
    while True:
        stats = detector.get_statistics()
        if stats["average_score"] > 0.5:
            send_alert("平均漂移分数过高")
        await asyncio.sleep(interval)
```

## 部署建议

### 生产环境配置

1. **基线训练**：使用至少20-50条典型回复训练基线
2. **阈值调整**：根据实际场景调整阈值
3. **日志记录**：记录所有检测结果用于分析
4. **监控告警**：设置严重漂移告警

### 性能优化

- 基线样本限制在100条以内
- 使用deque实现O(1)的样本更新
- 纯Python实现，无需额外依赖

### 扩展建议

1. **持久化**：将基线样本保存到数据库
2. **可视化**：添加漂移趋势图表
3. **学习机制**：根据反馈自动调整权重
4. **多角色支持**：为不同角色创建独立检测器

## API参考

### PersonalityDriftDetector

| 方法 | 说明 |
|------|------|
| `update_baseline(text)` | 更新基线样本 |
| `detect(text)` | 执行漂移检测 |
| `set_role_definition(keywords, forbidden)` | 设置角色定义 |
| `set_topic(keywords)` | 设置话题 |
| `register_correction_callback(action, callback)` | 注册修正回调 |
| `get_statistics()` | 获取统计信息 |

### DriftResult

| 属性 | 类型 | 说明 |
|------|------|------|
| overall_score | float | 总体漂移分数 |
| level | DriftLevel | 漂移等级 |
| action | CorrectionAction | 修正动作 |
| metrics | dict | 各指标分数 |
| timestamp | float | 时间戳 |

## 示例场景

### 客服机器人

```python
detector = PersonalityDriftDetector()
detector.set_role_definition(
    keywords=["客服", "帮助", "解决"],
    forbidden=["个人意见", "我不喜欢"]
)
```

### 教育助手

```python
detector = PersonalityDriftDetector()
detector.set_role_definition(
    keywords=["学习", "知识", "讲解"],
    forbidden=["不耐烦", "太简单"]
)
```

### 创意写作助手

```python
detector = PersonalityDriftDetector()
detector.set_role_definition(
    keywords=["创意", "灵感", "故事"],
    forbidden=["不会写", "没意思"]
)
```

## 故障排查

| 问题 | 解决方案 |
|------|----------|
| 误报率高 | 增加基线样本，调整阈值 |
| 检测不准确 | 检查权重配置，调整指标敏感度 |
| 性能问题 | 减少基线样本数量，优化检测频率 |

## 版本信息

- 版本：1.0.0
- 更新日期：2026-02-27
- 作者：AI Assistant
