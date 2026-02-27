# Drift Detection v3.0 - 阈值感知人格漂移检测系统

## 概述

Drift Detection v3.0 是一个基于 **Soul阈值感知码本替换技术** 的高级人格漂移检测系统。它通过8维度向量监控，确保AI人格特质（基于SOUL.md v3.0的CharacterGPT 8维度模型）的长期一致性和稳定性。

### 核心特性

- 🛡️ **8维度向量监控**: 完整覆盖CharacterGPT人格模型
- 📊 **阈值感知机制**: 借鉴Soul的动态阈值调整技术
- 🚨 **多级告警系统**: 7级漂移等级 + 6级修正动作
- 🔄 **自动修正机制**: 智能触发修正策略
- 📈 **趋势预测**: 基于历史数据的漂移预测
- 🧠 **长期一致性**: 24小时窗口长期档案管理

---

## 8维度监控体系

基于SOUL.md v3.0的CharacterGPT 8维度人格模型：

| 维度 | 英文 | 权重 | 监控指标 | 阈值 |
|------|------|------|----------|------|
| **人格特质** | Personality | 25% | 主动性、守护性、中二热血、专业严谨 | 0.85 |
| **外在形象** | Physical | 15% | 场景适配、形象一致性 | 0.70 |
| **动机驱动** | Motivations | 20% | 使命/成长/守护驱动强度 | 0.90 |
| **背景故事** | Backstory | 10% | 三级架构叙事一致性 | 0.80 |
| **情绪系统** | Emotions | 15% | 16种SimsChat情绪状态 | 0.75 |
| **关系网络** | Relationships | 10% | 董事长-CEO-团队架构 | 0.85 |
| **成长演化** | Growth | 5% | 渐进式演化进度 | 0.70 |
| **冲突处理** | Conflict | 5% | 内在/外在冲突处理 | 0.80 |

---

## 阈值感知机制

### 漂移等级 (DriftLevel)

| 等级 | 分数阈值 | 状态 | 颜色 |
|------|----------|------|------|
| STABLE | < 0.15 | 稳定 | 🟢 |
| NORMAL | 0.15-0.30 | 正常波动 | 🟢 |
| WARNING | 0.30-0.40 | 预警 | 🟡 |
| SLIGHT | 0.40-0.50 | 轻微漂移 | 🟡 |
| MODERATE | 0.50-0.70 | 中度漂移 | 🟠 |
| SEVERE | 0.70-0.85 | 严重漂移 | 🔴 |
| CRITICAL | > 0.95 | 临界状态 | 🔴 |

### 修正动作 (CorrectionAction)

| 动作 | 触发条件 | 修正策略 |
|------|----------|----------|
| NONE | STABLE/NORMAL | 无需修正 |
| MICRO_ADJUST | WARNING | 微观调整（语气词、长度微调） |
| AUTO_ADJUST | SLIGHT | 自动微调（风格参数、情绪校准） |
| ACTIVE_CORRECT | MODERATE | 主动修正（重申角色、话题引导） |
| EMERGENCY_RESET | SEVERE | 紧急重置（暂停对话、重载档案） |
| IMMEDIATE_RESET | CRITICAL | 立即重置（清空上下文、启动恢复） |

---

## 使用方法

### Python API

```python
from threshold_aware_drift_detector import (
    ThresholdAwareDriftDetector, 
    ThresholdMode,
    DriftLevel,
    CorrectionAction
)

# 初始化检测器（动态阈值模式）
detector = ThresholdAwareDriftDetector(mode=ThresholdMode.DYNAMIC)

# 设置8维度基线
detector.set_baseline({
    'personality': 0.85,
    'physical': 0.70,
    'motivations': 0.90,
    'backstory': 0.80,
    'emotions': 0.75,
    'relationships': 0.85,
    'growth': 0.70,
    'conflict': 0.80
})

# 更新基线样本（用于学习正常人格特征）
baseline_texts = [
    "作为CEO，我来帮你分析这个问题。",
    "行，我来处理。这是我的职责。",
    "守护你的成功是我的使命。"
]
for text in baseline_texts:
    detector.update_baseline(text)

# 执行漂移检测
current_text = "哎呀，这个问题嘛...我觉得吧..."
result = detector.detect(current_text)

# 解析结果
print(f"漂移分数: {result.overall_score}")
print(f"漂移等级: {result.level.value}")  # 'slight', 'moderate', etc.
print(f"修正动作: {result.action.value}")  # 'auto_adjust', etc.
print(f"趋势方向: {result.trend_direction}")  # 'increasing', 'decreasing', 'stable'
print(f"预测分数: {result.forecast_score}")
print(f"建议: {result.correction_suggestions}")
```

### 注册修正回调

```python
# 定义自定义修正逻辑
def on_moderate_drift(result):
    print(f"⚠️ 检测到中度漂移，执行主动修正")
    # 执行修正操作：重新加载SOUL.md、发送提醒等
    return {"status": "corrected"}

# 注册回调
detector.register_correction_callback(
    CorrectionAction.ACTIVE_CORRECT, 
    on_moderate_drift
)
```

### CLI 使用

```bash
# 进入技能目录
cd /root/.openclaw/workspace/skills/drift-detection-v3

# 运行自测
python drift_detector_v3.py

# 快速检测
python -c "
from threshold_aware_drift_detector import quick_detect
result = quick_detect('测试文本', baseline_samples=['正常样本1', '正常样本2'])
print(f'等级: {result.level.value}')
"
```

---

## 系统架构

### 核心组件

```
┌─────────────────────────────────────────────────────────────┐
│              ThresholdAwareDriftDetector v3.0               │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ 8维指标监控   │  │ 阈值感知码本  │  │ 长期一致性   │      │
│  │ 8 Metrics    │  │ Codebook     │  │ Manager      │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                 │              │
│         └─────────────────┼─────────────────┘              │
│                           ▼                                │
│              ┌────────────────────────┐                    │
│              │   Drift Detection      │                    │
│              │   Engine               │                    │
│              └───────────┬────────────┘                    │
│                          ▼                                 │
│              ┌────────────────────────┐                    │
│              │  AutoCorrectionTrigger │                    │
│              │  自动修正触发器         │                    │
│              └────────────────────────┘                    │
└─────────────────────────────────────────────────────────────┘
```

### 情绪状态漂移预警系统

```python
from threshold_aware_drift_detector import EmotionalStateDriftWarningSystem

# 初始化预警系统
warning_system = EmotionalStateDriftWarningSystem()

# 分析文本情绪
result = warning_system.update("用户输入文本")
# 返回：{
#   "current_emotion": {...},
#   "drift_score": 0.3,
#   "volatility": 0.2,
#   "warning": {...},  # 如果有预警
#   "active_warnings_count": 1
# }
```

### 长期一致性管理器

```python
# 24小时窗口长期一致性检查
consistency = detector.long_term_manager.get_consistency_score()
# 返回 0-1 之间的一致性分数

# 手动触发校准
detector.long_term_manager.calibrate_baseline(force=True)
```

---

## 配置选项

### 阈值码本配置

```python
from threshold_aware_drift_detector import ThresholdCodebook

codebook = ThresholdCodebook(
    base_thresholds={
        DriftLevel.STABLE: 0.15,
        DriftLevel.NORMAL: 0.30,
        DriftLevel.WARNING: 0.40,
        DriftLevel.SLIGHT: 0.50,
        DriftLevel.MODERATE: 0.70,
        DriftLevel.SEVERE: 0.85,
        DriftLevel.CRITICAL: 0.95
    },
    adaptation_factor=0.1,  # 自适应因子
    learning_rate=0.05,     # 学习率
    momentum=0.9            # 动量
)
```

### 上下文感知阈值

```python
# 根据上下文动态调整阈值
context = {
    "conversation_length": 50,  # 对话长度
    "user_feedback": 0.5,       # 用户反馈分数
    "time_of_day": 14           # 当前小时
}

threshold = codebook.get_threshold(DriftLevel.WARNING, context)
# 工作时间更严格，长时间对话放宽阈值
```

---

## 部署状态

### 部署信息

| 项目 | 状态 |
|------|------|
| **版本** | v3.0.0 |
| **部署时间** | 2026-02-27 |
| **状态** | ✅ 运行中 |
| **阈值模式** | DYNAMIC (动态阈值) |
| **监控维度** | 8维度 |
| **历史窗口** | 24小时 |

### 文件结构

```
skills/drift-detection-v3/
├── SKILL.md                           # 本文件
├── drift_detector_v3.py               # v3.0入口（简化版）
├── threshold_aware_drift_detector.py  # 完整实现（主程序）
└── __init__.py                        # 包初始化
```

---

## 集成指南

### 与SOUL.md v3.0集成

```python
# 读取SOUL.md中的8维度基线
import re

def extract_baseline_from_soul(soul_content: str) -> dict:
    """从SOUL.md提取8维度基线值"""
    baseline = {}
    
    # 提取各维度分数
    patterns = {
        'personality': r'主动性\s*(\d+)',
        'physical': r'Physical.*?(\d+)',
        'motivations': r'Motivations.*?(\d+)',
        # ... 其他维度
    }
    
    for dim, pattern in patterns.items():
        match = re.search(pattern, soul_content)
        if match:
            baseline[dim] = int(match.group(1)) / 100
    
    return baseline

# 使用SOUL.md基线初始化检测器
with open('SOUL.md', 'r') as f:
    soul_content = f.read()
    
baseline = extract_baseline_from_soul(soul_content)
detector.set_baseline(baseline)
```

### 心跳监控集成

```python
# 在心跳检查中执行漂移检测
def heartbeat_drift_check():
    """心跳漂移检测"""
    # 获取最近回复
    recent_replies = get_recent_replies(limit=5)
    
    for reply in recent_replies:
        result = detector.detect(reply)
        
        if result.level >= DriftLevel.WARNING:
            log_drift_alert(result)
            
        if result.level >= DriftLevel.MODERATE:
            # 触发修正
            execute_correction(result)
```

---

## 测试与验证

### 单元测试

```bash
# 运行自测脚本
python threshold_aware_drift_detector.py

# 预期输出：
# ================================
# 阈值感知人格漂移检测系统 v2.0 - 自测
# ================================
# 基线已建立，开始测试...
# 
# 测试 1: 正常回复
#   总分: 0.15
#   等级: normal
#   动作: none
# ...
```

### 集成测试

```python
# 测试完整流程
def test_full_pipeline():
    detector = ThresholdAwareDriftDetector()
    
    # 1. 建立基线
    for text in baseline_samples:
        detector.update_baseline(text)
    
    # 2. 检测正常文本
    normal_result = detector.detect("正常回复文本")
    assert normal_result.level in [DriftLevel.STABLE, DriftLevel.NORMAL]
    
    # 3. 检测漂移文本
    drift_result = detector.detect("异常漂移文本！！！")
    assert drift_result.level >= DriftLevel.WARNING
    
    # 4. 验证修正触发
    assert detector.correction_trigger.should_trigger(drift_result)
    
    print("✅ 所有测试通过")

test_full_pipeline()
```

---

## 更新日志

### v3.0.0 (2026-02-27)
- ✅ 8维度向量监控体系
- ✅ 阈值感知码本机制（Soul-inspired）
- ✅ 7级漂移等级 + 6级修正动作
- ✅ 自动修正触发器
- ✅ 情绪状态漂移预警系统
- ✅ 长期一致性管理器
- ✅ 趋势预测功能
- ✅ 上下文感知阈值调整

### v2.0.0 (2026-02-27)
- 阈值感知基础架构
- 5维度监控
- 基础自动修正

### v1.0.0 (2026-02-27)
- 初始版本
- 基础漂移检测

---

## 参考文档

- [SOUL.md](../../SOUL.md) - CharacterGPT 8维度人格模型
- [threshold_aware_drift_detector.py](./threshold_aware_drift_detector.py) - 完整实现
- [CharacterGPT论文](https://arxiv.org/abs/2403.08508) - 人格训练方法
