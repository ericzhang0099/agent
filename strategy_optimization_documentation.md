# 策略优化系统文档

## 概述

策略优化系统是元认知研究的第二部分，实现了智能Agent的策略选择、评估和优化功能。系统包含两个核心模块：

1. **strategy_selector.py** - 策略选择器
2. **strategy_optimizer.py** - 策略优化器

## 核心功能

### 1. 策略库管理 (StrategyLibrary)

策略库管理器负责维护可用策略集合，支持：

- 策略注册与注销
- 按领域/类型索引查询
- 策略使用统计更新
- JSON导出功能

```python
from strategy_selector import StrategyLibrary, Strategy, StrategyType

# 创建策略库
library = StrategyLibrary()

# 注册策略
library.register(Strategy(
    strategy_id="quick_analysis",
    name="快速分析策略",
    strategy_type=StrategyType.ANALYSIS,
    description="快速识别关键信息",
    applicable_domains=["analysis", "business"],
    required_resources=["data_processing"],
    estimated_time_range=(15, 45),
))

# 查询策略
strategies = library.find_by_domain("analysis")
stats = library.get_library_stats()
```

### 2. 策略选择算法 (StrategySelector)

策略选择器根据任务特征选择最优策略，采用多维度评分机制：

**评分维度：**
- 领域匹配度 (25%)
- 复杂度匹配度 (20%)
- 时间匹配度 (20%)
- 技能匹配度 (15%)
- 历史表现 (15%)
- 时效性 (5%)

```python
from strategy_selector import StrategySelector, TaskProfile, TaskComplexity

selector = StrategySelector(library)

# 定义任务画像
task = TaskProfile(
    task_id="task_001",
    task_type="data_analysis",
    complexity=TaskComplexity.MODERATE,
    required_skills=["data_processing", "visualization"],
    time_constraints=60,
    domain="analysis"
)

# 选择策略
result = selector.select(task, top_k=3)
print(f"选中策略: {result.selected_strategy.name}")
print(f"置信度: {result.confidence}")
print(f"推理: {result.reasoning}")
```

### 3. 策略效果评估 (StrategyEvaluator)

策略评估器跟踪策略执行效果，支持：

- 执行记录管理
- 性能统计计算
- 策略对比分析
- 趋势检测

```python
from strategy_optimizer import StrategyEvaluator

evaluator = StrategyEvaluator(library)

# 开始执行记录
record_id = evaluator.start_execution("task_001", "quick_analysis")

# ... 执行策略 ...

# 完成记录
evaluator.complete_execution(
    record_id,
    success=True,
    metrics={"quality": 0.85, "efficiency": 0.9},
    feedback="执行效果良好"
)

# 获取性能报告
perf = evaluator.evaluate_strategy("quick_analysis")
report = evaluator.generate_report()
```

### 4. 策略参数优化 (StrategyOptimizer)

策略优化器提供多种优化功能：

#### 4.1 优化建议生成

```python
from strategy_optimizer import StrategyOptimizer

optimizer = StrategyOptimizer(library, evaluator)

# 生成改进建议
suggestions = optimizer.suggest_improvements()
for s in suggestions:
    print(f"策略: {s.target_strategy}")
    print(f"建议类型: {s.suggestion_type}")
    print(f"描述: {s.description}")
    print(f"预期改进: {s.expected_improvement}")
```

#### 4.2 策略组合优化

```python
# 寻找最优策略组合
combinations = optimizer.find_strategy_combinations(task_profile)
for combo in combinations:
    print(f"组合: {combo['strategies']}")
    print(f"预估分数: {combo['estimated_score']}")
    print(f"理由: {combo['rationale']}")
```

#### 4.3 动态策略调整

```python
# 根据上下文动态调整
adaptations = optimizer.adapt_strategy(
    "quick_analysis",
    execution_context={
        "time_pressure": 0.8,
        "complexity": TaskComplexity.COMPLEX
    }
)
```

### 5. 元学习引擎 (MetaLearningEngine)

元学习引擎支持跨任务学习，发现最优策略选择模式：

```python
from strategy_optimizer import MetaLearningEngine

meta_learner = MetaLearningEngine(evaluator)

# 从执行记录中学习
result = meta_learner.learn_from_executions(min_samples=20)

# 预测最佳策略
best_strategy = meta_learner.predict_best_strategy(task_profile)

# 获取学习到的模式
patterns = meta_learner.get_learned_patterns()
```

## 预定义策略

系统包含6种预定义策略：

| 策略ID | 名称 | 类型 | 适用场景 | 预估时间 |
|--------|------|------|----------|----------|
| research_deep_dive | 深度研究策略 | RESEARCH | 复杂问题全面了解 | 60-180分钟 |
| quick_analysis | 快速分析策略 | ANALYSIS | 时间敏感的分析 | 15-45分钟 |
| creative_brainstorm | 创意头脑风暴 | CREATIVE | 创新设计任务 | 30-90分钟 |
| agile_execution | 敏捷执行策略 | EXECUTION | 开发实施任务 | 20-60分钟 |
| collaborative_workflow | 协作工作流策略 | COLLABORATION | 多Agent协作项目 | 45-120分钟 |
| optimization_iterative | 迭代优化策略 | OPTIMIZATION | 优化调优任务 | 40-100分钟 |

## 快速开始

### 完整示例

```python
from strategy_selector import (
    create_default_library, StrategySelector, 
    TaskProfile, TaskComplexity
)
from strategy_optimizer import (
    create_optimization_system, StrategyEvaluator,
    StrategyOptimizer, MetaLearningEngine
)

# 1. 创建默认策略库
library = create_default_library()

# 2. 创建策略选择器
selector = StrategySelector(library)

# 3. 定义任务
task = TaskProfile(
    task_id="analysis_task_001",
    task_type="market_analysis",
    complexity=TaskComplexity.MODERATE,
    required_skills=["data_analysis", "reporting"],
    time_constraints=45,
    quality_requirements=0.8,
    domain="business"
)

# 4. 选择策略
result = selector.select(task)
print(f"推荐策略: {result.selected_strategy.name}")
print(f"置信度: {result.confidence:.2%}")
print(f"预估成功率: {result.estimated_outcome['success_prob']:.2%}")

# 5. 创建优化系统
evaluator = StrategyEvaluator(library)
optimizer = StrategyOptimizer(library, evaluator)

# 6. 模拟执行并评估
record_id = evaluator.start_execution(task.task_id, result.selected_strategy.strategy_id)
# ... 执行策略 ...
evaluator.complete_execution(
    record_id,
    success=True,
    metrics={"quality": 0.82, "efficiency": 0.88, "time": 42}
)

# 7. 获取优化建议
suggestions = optimizer.suggest_improvements()
print(f"\n优化建议数量: {len(suggestions)}")
```

## 架构设计

```
┌─────────────────────────────────────────────────────────────┐
│                    策略优化系统架构                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  StrategyLibrary │    │  StrategySelector│               │
│  │  (策略库管理)    │◄──►│  (策略选择器)    │               │
│  │                 │    │                 │                │
│  │ • 策略注册      │    │ • 多维度评分    │                │
│  │ • 索引查询      │    │ • 智能匹配      │                │
│  │ • 统计更新      │    │ • 置信度计算    │                │
│  └─────────────────┘    └─────────────────┘                │
│           ▲                      │                          │
│           │                      ▼                          │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │  Strategy       │◄───┤  SelectionResult │               │
│  │  (策略定义)      │    │  (选择结果)      │               │
│  └─────────────────┘    └─────────────────┘                │
│                                                             │
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │ StrategyEvaluator│    │ StrategyOptimizer│               │
│  │  (效果评估)      │◄──►│  (策略优化)      │               │
│  │                 │    │                 │                │
│  │ • 执行记录      │    │ • 参数优化      │                │
│  │ • 性能统计      │    │ • 组合优化      │                │
│  │ • 趋势分析      │    │ • 动态调整      │                │
│  └─────────────────┘    └─────────────────┘                │
│           │                      │                          │
│           ▼                      ▼                          │
│  ┌─────────────────────────────────────────┐               │
│  │       MetaLearningEngine                │               │
│  │       (元学习引擎)                       │               │
│  │                                         │               │
│  │  • 跨任务学习    • 模式发现              │               │
│  │  • 策略预测      • 持续改进              │               │
│  └─────────────────────────────────────────┘               │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 扩展指南

### 添加自定义策略

```python
from strategy_selector import Strategy, StrategyType

my_strategy = Strategy(
    strategy_id="my_custom_strategy",
    name="我的自定义策略",
    strategy_type=StrategyType.ANALYSIS,
    description="针对特定场景的优化策略",
    applicable_domains=["my_domain"],
    required_resources=["resource_1", "resource_2"],
    estimated_time_range=(10, 30),
    characteristics={"depth": 0.7, "breadth": 0.6, "speed": 0.8}
)

library.register(my_strategy)
```

### 自定义评分权重

```python
# 调整评分权重
selector.set_weights({
    "domain_match": 0.3,
    "complexity_match": 0.15,
    "time_match": 0.25,
    "skill_match": 0.1,
    "historical_performance": 0.15,
    "recency": 0.05,
})
```

### 添加自定义评估指标

```python
# 完成执行时添加自定义指标
evaluator.complete_execution(
    record_id,
    success=True,
    metrics={
        "quality": 0.85,
        "efficiency": 0.9,
        "user_satisfaction": 0.92,  # 自定义指标
        "cost": 0.3,
    }
)
```

## API参考

### StrategySelector

| 方法 | 描述 |
|------|------|
| `select(task, top_k=3, constraints=None)` | 为任务选择最优策略 |
| `set_weights(weights)` | 设置评分权重 |
| `get_selection_history(limit=10)` | 获取选择历史 |

### StrategyEvaluator

| 方法 | 描述 |
|------|------|
| `start_execution(task_id, strategy_id, context)` | 开始执行记录 |
| `complete_execution(record_id, success, metrics, feedback)` | 完成执行记录 |
| `evaluate_strategy(strategy_id)` | 评估策略表现 |
| `compare_strategies(strategy_ids, metric)` | 比较多个策略 |
| `generate_report(strategy_id=None)` | 生成评估报告 |

### StrategyOptimizer

| 方法 | 描述 |
|------|------|
| `suggest_improvements(strategy_id, min_executions)` | 生成优化建议 |
| `optimize_parameters(strategy_id, target_metric)` | 优化策略参数 |
| `find_strategy_combinations(task_profile, max_strategies)` | 寻找策略组合 |
| `adapt_strategy(strategy_id, execution_context)` | 动态调整策略 |

## 总结

策略优化系统提供了完整的策略管理、选择、评估和优化能力：

1. **策略库管理** - 灵活的策略注册、查询和统计
2. **智能选择** - 多维度评分机制，支持约束条件
3. **效果评估** - 全面的执行跟踪和性能分析
4. **持续优化** - 参数优化、组合优化和动态调整
5. **元学习** - 跨任务学习，不断提升选择准确性

系统代码约300行，设计简洁、易于扩展，可集成到各类AI Agent系统中。
