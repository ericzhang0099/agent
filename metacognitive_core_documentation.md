# 元认知核心系统文档

## 概述

基于ACT-R认知架构简化版实现的元认知与自我改进系统，聚焦三大核心功能：

1. **自我监控循环** - 持续追踪认知状态
2. **策略选择器** - 智能选择执行策略
3. **反思日志系统** - 记录与学习认知过程

---

## 核心概念

### ACT-R 简化模型

| 组件 | 功能 | 对应类 |
|------|------|--------|
| Declarative Memory | 存储事实性知识 | `DeclarativeMemory` |
| Procedural Memory | 存储程序性知识 | `ProceduralMemory` |
| Goal Buffer | 当前目标状态 | `Goal` |
| Metacognitive Buffer | 元认知状态 | `MetacognitiveState` |

### 元认知三要素

```
┌─────────────────────────────────────────┐
│           元认知核心系统                 │
├─────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────┐ │
│  │ 监控    │  │ 选择    │  │ 反思    │ │
│  │ 循环    │→ │ 策略    │→ │ 日志    │ │
│  │         │  │         │  │         │ │
│  │•性能追踪│  │•任务分析│  │•记录    │ │
│  │•异常检测│  │•策略匹配│  │•模式识别│ │
│  │•置信更新│  │•效用学习│  │•洞察生成│ │
│  └─────────┘  └─────────┘  └─────────┘ │
└─────────────────────────────────────────┘
```

---

## 核心组件详解

### 1. 自我监控循环 (SelfMonitoringLoop)

**功能**：持续监控任务执行状态

**关键方法**：
- `start_monitoring(task)` - 开始监控新任务
- `update(progress, error_rate, cognitive_load)` - 更新状态
- `_detect_anomalies()` - 检测认知异常

**监控指标**：
- 任务进度 (0-1)
- 错误率 (0-1)
- 认知负荷 (0-1)
- 置信度 (0-1)

**异常类型**：
- `performance_drop` - 性能突然下降
- `high_cognitive_load` - 认知负荷过高
- `low_confidence` - 置信度过低

### 2. 策略选择器 (StrategySelector)

**策略类型**：
| 策略 | 适用场景 | 特点 |
|------|----------|------|
| ANALYTICAL | 标准任务 | 逻辑分析 |
| INTUITIVE | 高认知负荷 | 快速决策 |
| SYSTEMATIC | 高复杂度+高精度 | 系统化方法 |
| HEURISTIC | 时间紧迫 | 经验法则 |
| ADAPTIVE | 新颖任务 | 动态调整 |

**选择逻辑**：
```python
if time_pressure > 0.7 and accuracy < 0.6:
    → HEURISTIC
elif complexity > 0.7 and accuracy > 0.8:
    → SYSTEMATIC
elif novelty > 0.7:
    → ADAPTIVE
elif cognitive_load > 0.7:
    → INTUITIVE
else:
    → ANALYTICAL
```

### 3. 反思日志系统 (ReflectionLog)

**反思阶段**：
1. **before** - 任务开始前：策略选择
2. **during** - 任务执行中：异常触发
3. **after** - 任务完成后：总结评估

**数据结构**：
```python
ReflectionEntry:
    timestamp: float          # 时间戳
    phase: str               # 阶段
    observation: str         # 观察
    evaluation: str          # 评估
    strategy_used: str       # 使用策略
    outcome: str             # 结果
    lessons_learned: List    # 教训
    confidence_before: float # 前置信度
    confidence_after: float  # 后置信度
```

---

## 快速开始

```python
from metacognitive_core import MetacognitiveCore

# 创建系统
mc = MetacognitiveCore()

# 开始任务
result = mc.start_task(
    task_description="分析数据并生成报告",
    context={
        'time_pressure': 0.3,
        'accuracy_required': 0.9
    }
)
print(f"选择策略: {result['strategy']}")

# 监控执行
status = mc.monitor(
    progress=0.5,
    error_rate=0.1,
    cognitive_load=0.6
)

# 完成任务
mc.complete_task(outcome="success", final_confidence=0.85)

# 获取洞察
status = mc.get_status()
for insight in status['insights']:
    print(insight)
```

---

## 系统特性

### 学习机制

1. **产生式学习**：根据成功/失败历史更新规则效用
2. **策略适应**：基于任务特征动态选择策略
3. **模式识别**：从反思历史中识别有效策略

### 自适应能力

- 认知负荷过高时自动简化
- 性能下降时触发反思
- 置信度与性能不匹配时调整

---

## 扩展建议

1. **添加更多产生式规则**：针对特定领域
2. **集成外部知识库**：增强声明性记忆
3. **实现更复杂的策略**：深度学习模型选择
4. **可视化监控面板**：实时展示认知状态

---

## 参考

- ACT-R Cognitive Architecture (Anderson et al.)
- Metacognition Theory (Flavell, 1979)
- Self-Regulated Learning (Zimmerman, 2000)
