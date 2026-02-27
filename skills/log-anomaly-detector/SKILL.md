---
name: log-anomaly-detector
description: 日志分析与异常检测系统。自动分析系统日志、应用日志和Agent运行日志，检测异常模式、错误趋势和潜在问题。支持实时监控、历史分析、告警生成和可视化报告。当需要排查系统问题、监控系统健康、分析错误趋势或生成日志报告时使用此Skill。
version: 1.0.0
author: KCGS
tags: [log-analysis, anomaly-detection, monitoring, troubleshooting, system-health]
---

# Log Anomaly Detector - 日志分析与异常检测系统

## 功能概述

Log Anomaly Detector 是一个智能日志分析解决方案，能够自动发现日志中的异常模式、错误趋势和潜在系统问题。

### 核心功能

1. **日志采集与解析**
   - 多格式日志支持 (JSON, plain text, structured)
   - 自动格式识别和字段提取
   - 实时日志流处理
   - 历史日志批量分析

2. **异常检测引擎**
   - 基于统计的异常检测 (Z-score, IQR)
   - 模式匹配 (正则表达式、关键字)
   - 频率异常检测 (突发错误、流量异常)
   - 时序异常检测 (趋势变化、周期性异常)

3. **智能分析**
   - 错误聚类与分类
   - 根因分析建议
   - 关联性分析 (多日志源关联)
   - 预测性分析 (错误趋势预测)

4. **告警与报告**
   - 分级告警机制 (INFO/WARNING/CRITICAL)
   - 实时告警通知
   - 可视化分析报告
   - 趋势图表生成

## 使用方法

### 基础用法

```python
from log_anomaly_detector import LogAnomalyDetector, LogEntry, AnomalyLevel

# 初始化检测器
detector = LogAnomalyDetector(data_dir="./log_analysis")

# 分析日志文件
results = detector.analyze_log_file(
    log_path="/var/log/openclaw/agent.log",
    log_type="agent",
    time_window="1h"
)
```

### 实时日志监控

```python
# 启动实时监控
detector.start_monitoring(
    log_paths=[
        "/var/log/openclaw/agent.log",
        "/var/log/openclaw/gateway.log"
    ],
    check_interval=60,  # 每60秒检查一次
    alert_callback=send_alert
)

# 停止监控
detector.stop_monitoring()
```

### 异常检测配置

```python
from log_anomaly_detector import AnomalyConfig

# 自定义异常检测配置
config = AnomalyConfig(
    # 错误关键字模式
    error_patterns=[
        r"ERROR|FATAL|CRITICAL",
        r"Exception|Traceback",
        r"timeout|connection refused",
        r"memory leak|out of memory"
    ],
    
    # 频率阈值
    frequency_thresholds={
        "error_per_minute": 10,
        "warning_per_minute": 50,
        "unique_errors": 5
    },
    
    # 时序异常参数
    time_series_config={
        "window_size": 10,
        "std_threshold": 3.0,
        "trend_change_threshold": 0.3
    }
)

detector = LogAnomalyDetector(config=config)
```

### 分析结果处理

```python
# 获取异常报告
report = detector.generate_report(time_range="24h")

# 遍历检测到的异常
for anomaly in report.anomalies:
    print(f"[{anomaly.level.value}] {anomaly.type}")
    print(f"  时间: {anomaly.timestamp}")
    print(f"  描述: {anomaly.description}")
    print(f"  影响: {anomaly.impact}")
    if anomaly.suggested_action:
        print(f"  建议: {anomaly.suggested_action}")

# 获取错误统计
stats = report.error_stats
print(f"总错误数: {stats.total_errors}")
print(f"唯一错误类型: {stats.unique_error_types}")
print(f"最严重错误: {stats.top_errors[0]}")
```

### 与Agent Health Monitor集成

```python
# 结合Agent健康监控
from agent_health_monitor import AgentHealthMonitor

health_monitor = AgentHealthMonitor()
detector = LogAnomalyDetector()

# 当检测到严重异常时，检查Agent健康状态
def on_critical_anomaly(anomaly):
    agent_id = extract_agent_id(anomaly.log_entry)
    health = health_monitor.check_health(agent_id)
    
    if health.status.value in ["unhealthy", "offline"]:
        send_urgent_alert(f"Agent {agent_id} 异常: {anomaly.description}")

detector.set_anomaly_callback(AnomalyLevel.CRITICAL, on_critical_anomaly)
```

## 配置说明

### 默认检测规则

```python
# 内置异常检测规则
DEFAULT_RULES = {
    # 错误级别检测
    "error_spike": {
        "type": "frequency",
        "pattern": r"ERROR|FATAL",
        "threshold": 10,  # 每分钟超过10个错误
        "level": "WARNING"
    },
    
    # 异常堆栈检测
    "exception_detected": {
        "type": "pattern",
        "pattern": r"Traceback|Exception",
        "level": "CRITICAL"
    },
    
    # 超时检测
    "timeout_pattern": {
        "type": "pattern",
        "pattern": r"timeout|timed out",
        "level": "WARNING"
    },
    
    # 内存问题检测
    "memory_issue": {
        "type": "pattern",
        "pattern": r"out of memory|memory leak",
        "level": "CRITICAL"
    },
    
    # 连接问题检测
    "connection_issue": {
        "type": "pattern",
        "pattern": r"connection refused|connection reset|ECONNREFUSED",
        "level": "WARNING"
    }
}
```

### 日志格式配置

```python
# 支持的日志格式
LOG_FORMATS = {
    "json": {
        "parser": "json",
        "fields": ["timestamp", "level", "message", "source"]
    },
    "plain": {
        "parser": "regex",
        "pattern": r"(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.*)"
    },
    "syslog": {
        "parser": "regex",
        "pattern": r"^(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)"
    }
}
```

## 告警级别

| 级别 | 触发条件 | 响应时间 | 示例 |
|------|----------|----------|------|
| INFO | 一般信息 | 无需响应 | 日志轮转完成 |
| WARNING | 潜在问题 | 4小时内 | 错误率上升 |
| CRITICAL | 严重问题 | 30分钟内 | 异常崩溃、内存泄漏 |
| EMERGENCY | 系统故障 | 立即 | 服务完全不可用 |

## 输出格式

### 异常条目

```python
@dataclass
class Anomaly:
    id: str                    # 异常唯一标识
    level: AnomalyLevel        # 告警级别
    type: str                  # 异常类型
    timestamp: datetime        # 发生时间
    source: str                # 日志来源
    description: str           # 异常描述
    log_entry: LogEntry        # 原始日志
    impact: str                # 影响评估
    suggested_action: str      # 建议操作
    related_anomalies: List[str]  # 关联异常
```

### 分析报告

```python
@dataclass
class AnalysisReport:
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    total_entries: int
    anomalies: List[Anomaly]
    error_stats: ErrorStats
    trends: Dict[str, Trend]
    recommendations: List[str]
```

## CLI使用

```bash
# 分析单个日志文件
python log_anomaly_detector.py analyze /var/log/openclaw/agent.log

# 实时监控多个日志
python log_anomaly_detector.py monitor \
    /var/log/openclaw/agent.log \
    /var/log/openclaw/gateway.log \
    --interval 60

# 生成历史报告
python log_anomaly_detector.py report \
    --log-dir /var/log/openclaw \
    --time-range "24h" \
    --output report.html

# 测试配置
python log_anomaly_detector.py test-config ./my_config.yaml
```

## 集成示例

### 与Cron集成

```python
# 每小时执行日志分析
import schedule
import time

def hourly_log_analysis():
    detector = LogAnomalyDetector()
    
    # 分析过去1小时的日志
    report = detector.analyze_time_range(
        log_paths=["/var/log/openclaw/*.log"],
        time_range="1h"
    )
    
    # 发送告警
    for anomaly in report.anomalies:
        if anomaly.level.value in ["critical", "emergency"]:
            send_alert_notification(anomaly)
    
    # 保存报告
    detector.save_report(report, f"report_{datetime.now().strftime('%Y%m%d_%H')}.html")

schedule.every().hour.do(hourly_log_analysis)
```

### 与Heartbeat集成

```python
# 在心跳检查中分析日志健康度
def on_heartbeat(agent_id):
    detector = LogAnomalyDetector()
    
    # 检查最近日志
    recent_anomalies = detector.get_recent_anomalies(
        source=agent_id,
        time_range="5m"
    )
    
    if len(recent_anomalies) > 5:
        return {"status": "degraded", "reason": "recent_anomalies"}
    
    return {"status": "healthy"}
```

## 文件结构

```
log-anomaly-detector/
├── SKILL.md                    # 本文件
└── log_anomaly_detector.py     # 主程序
```

## 版本信息

- **版本**: v1.0
- **发布日期**: 2026-02-27
- **作者**: KCGS (Kimi Claw Growth System)
- **状态**: 已部署

## 更新日志

- v1.0 (2026-02-27): 初始版本
  - 多格式日志解析
  - 统计异常检测
  - 模式匹配规则
  - 实时监控功能
  - 可视化报告生成
