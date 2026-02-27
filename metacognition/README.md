# 元认知研究 - 第一部分：自我监控系统

## 概述

自我监控系统是Agent元认知能力的核心组件，使Agent能够实时监控自身状态、评估性能、检测异常，并据此调整行为策略。

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    自我监控系统                              │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────┐  ┌─────────────────┐  ┌─────────────┐ │
│  │   性能监控模块   │  │  认知负荷检测   │  │  异常检测   │ │
│  │                 │  │                 │  │             │ │
│  │ • 准确率追踪    │  │ • 任务复杂度    │  │ • 延迟突增  │ │
│  │ • 延迟统计      │  │ • 上下文使用    │  │ • 准确率下降│ │
│  │ • 资源使用      │  │ • 推理深度      │  │ • 资源耗尽  │ │
│  │ • 吞吐量        │  │ • 综合评估      │  │ • 错误率高  │ │
│  └────────┬────────┘  └────────┬────────┘  └──────┬──────┘ │
│           │                    │                   │        │
│           └────────────────────┼───────────────────┘        │
│                                │                            │
│                                ▼                            │
│                    ┌─────────────────────┐                  │
│                    │    健康状态评估     │                  │
│                    │                     │                  │
│                    │ • 健康分数计算      │                  │
│                    │ • 状态分级          │                  │
│                    │ • 建议生成          │                  │
│                    └─────────────────────┘                  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 核心模块

### 1. 性能监控 (Performance Monitoring)

**功能**：
- 请求成功率追踪
- 延迟统计（平均/P95/P99）
- 系统资源监控（CPU/内存）
- 吞吐量计算

**关键指标**：
| 指标 | 说明 | 正常范围 |
|------|------|----------|
| accuracy_rate | 请求成功率 | > 95% |
| avg_latency_ms | 平均延迟 | < 500ms |
| p95_latency_ms | P95延迟 | < 1000ms |
| cpu_usage_percent | CPU使用率 | < 80% |
| memory_usage_percent | 内存使用率 | < 85% |
| requests_per_second | 每秒请求数 | 视场景 |

**使用示例**：
```python
from self_monitor import SelfMonitor, MonitoringLevel

# 创建监控器
monitor = SelfMonitor(monitoring_level=MonitoringLevel.HIGH)

# 记录请求
start = monitor.record_request_start()
# ... 执行业务逻辑 ...
monitor.record_request_end(start, success=True, tokens_generated=150)

# 获取性能指标
metrics = monitor.get_performance_metrics()
print(f"成功率: {metrics.accuracy_rate:.1%}")
print(f"平均延迟: {metrics.avg_latency_ms:.1f}ms")
```

### 2. 认知负荷检测 (Cognitive Load Detection)

**功能**：
- 任务数量监控
- 任务复杂度评估
- 上下文窗口使用追踪
- 推理深度统计
- 综合负荷等级评估

**负荷等级**：
| 等级 | 分数范围 | 说明 |
|------|----------|------|
| IDLE | 0.0 - 0.1 | 空闲状态 |
| LOW | 0.1 - 0.3 | 低负荷 |
| NORMAL | 0.3 - 0.6 | 正常负荷 |
| HIGH | 0.6 - 0.85 | 高负荷 |
| OVERLOAD | 0.85+ | 超载 |

**使用示例**：
```python
# 更新任务状态
monitor.update_task_status(
    active_tasks=3,
    pending_tasks=2,
    completed_tasks=10
)

# 记录任务复杂度 (0-1)
monitor.record_task_complexity(0.7)

# 更新上下文使用
monitor.update_context_usage(
    tokens_used=2000,
    max_tokens=8000
)

# 评估认知负荷
load = monitor.evaluate_cognitive_load()
print(f"当前负荷: {load.overall_load.value}")
print(f"负荷分数: {load.load_score:.2f}")
```

### 3. 异常检测 (Anomaly Detection)

**检测类型**：
| 异常类型 | 触发条件 | 建议操作 |
|----------|----------|----------|
| LATENCY_SPIKE | 延迟突增2倍以上 | 减少并发任务或优化处理 |
| ACCURACY_DROP | 准确率低于80% | 检查失败原因，调整模型 |
| RESOURCE_EXHAUSTION | 内存使用>90% | 清理缓存或减少批处理大小 |
| ERROR_RATE_HIGH | 错误率>20% | 检查错误日志和系统健康 |
| THROUGHPUT_DROP | 吞吐量下降50% | 检查瓶颈或资源限制 |

**使用示例**：
```python
# 注册异常回调
def on_anomaly(event):
    print(f"[告警] {event.anomaly_type.value}: {event.description}")
    print(f"建议操作: {event.suggested_action}")

monitor.register_anomaly_callback(on_anomaly)

# 启动自动监控
monitor.start_monitoring(interval_seconds=1.0)
```

## 性能追踪器 (PerformanceTracker)

提供高级性能追踪功能，支持装饰器和上下文管理器两种使用方式。

### 装饰器方式
```python
from performance_tracker import track, get_tracker

# 使用全局追踪器
@track()
def process_data(data):
    # 处理逻辑
    return result

# 使用实例追踪器
tracker = PerformanceTracker()

@tracker.track(function_name="custom_name")
def another_function():
    pass
```

### 上下文管理器方式
```python
from performance_tracker import trace

# 追踪代码块
with trace("database_query"):
    # 执行数据库查询
    result = db.query()
```

### 自定义指标
```python
tracker = get_tracker()

# 计数器
tracker.increment_counter("requests", 1)

# 仪表盘
tracker.set_gauge("memory_usage", 1024.5)

# 直方图
tracker.record_histogram("response_time", 150.0)
```

### 生成报告
```python
# 文本报告
print(tracker.generate_report())

# JSON导出
json_data = tracker.export_json()

# 摘要信息
summary = tracker.get_summary()
```

## 健康状态评估

系统综合多个维度计算健康分数(0-100)：

```python
health = monitor.get_health_status()
# {
#     "status": "healthy",  # healthy/degraded/warning/critical
#     "health_score": 85.5,
#     "performance": {...},
#     "cognitive_load": {...},
#     "active_anomalies": 0,
#     "uptime_seconds": 3600
# }
```

**状态分级**：
- **healthy** (90-100): 系统健康
- **degraded** (70-89): 性能下降
- **warning** (50-69): 需要关注
- **critical** (0-49): 严重问题

## 集成示例

### 完整使用示例
```python
from metacognition.self_monitor import SelfMonitor, MonitoringLevel
from metacognition.performance_tracker import PerformanceTracker, track

class MyAgent:
    def __init__(self):
        self.monitor = SelfMonitor(MonitoringLevel.HIGH)
        self.tracker = PerformanceTracker()
        self.monitor.start_monitoring(interval_seconds=1.0)
    
    @track()
    def process_request(self, request):
        # 记录请求开始
        start = self.monitor.record_request_start()
        
        try:
            # 更新认知负荷
            self.monitor.update_task_status(
                active_tasks=1,
                pending_tasks=0,
                completed_tasks=0
            )
            
            # 执行业务逻辑
            result = self._do_process(request)
            
            # 记录成功
            self.monitor.record_request_end(
                start,
                success=True,
                tokens_generated=len(result)
            )
            
            return result
            
        except Exception as e:
            # 记录失败
            self.monitor.record_request_end(start, success=False)
            raise
    
    def get_health(self):
        return self.monitor.get_health_status()
```

## API参考

### SelfMonitor 类

#### 构造函数
```python
SelfMonitor(
    monitoring_level: MonitoringLevel = MonitoringLevel.MEDIUM,
    history_window_size: int = 100,
    anomaly_threshold: float = 0.95
)
```

#### 主要方法
| 方法 | 说明 |
|------|------|
| `start_monitoring(interval_seconds)` | 启动自动监控 |
| `stop_monitoring()` | 停止自动监控 |
| `record_request_start()` | 记录请求开始 |
| `record_request_end(start, success, tokens)` | 记录请求结束 |
| `get_performance_metrics()` | 获取性能指标 |
| `evaluate_cognitive_load()` | 评估认知负荷 |
| `get_health_status()` | 获取健康状态 |
| `register_anomaly_callback(callback)` | 注册异常回调 |

### PerformanceTracker 类

#### 主要方法
| 方法 | 说明 |
|------|------|
| `track(function_name)` | 装饰器工厂 |
| `trace(name)` | 上下文管理器 |
| `increment_counter(name, value)` | 增加计数器 |
| `set_gauge(name, value)` | 设置仪表盘 |
| `record_histogram(name, value)` | 记录直方图 |
| `generate_report()` | 生成报告 |
| `export_json()` | JSON导出 |

## 总结

自我监控系统为Agent提供了：

1. **全面的性能洞察**：准确率、延迟、资源使用等多维度监控
2. **智能的负荷评估**：基于多因素综合评估认知负荷
3. **及时的异常检测**：自动检测并报告各类异常情况
4. **便捷的使用方式**：装饰器、上下文管理器等灵活接口
5. **可扩展的架构**：易于集成到现有Agent系统中

通过自我监控系统，Agent能够实现自我感知、自我调节，从而提升整体性能和可靠性。
