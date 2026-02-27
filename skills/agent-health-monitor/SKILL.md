---
name: agent-health-monitor
description: Agent健康状态监控系统。用于监控多Agent架构中各个Agent的健康状态、性能指标和资源使用。支持实时健康检查、告警生成、趋势分析。当需要监控Agent运行状态、检测性能问题、生成健康报告时使用此Skill。
---

# Agent Health Monitor - Agent健康状态监控系统

## 功能概述

Agent Health Monitor 是一个专业的Agent健康监控解决方案，用于多Agent架构的运维管理。

### 核心功能

1. **实时监控**
   - 响应时间监控
   - 任务成功率追踪
   - CPU/内存/磁盘使用率
   - 队列深度和负载均衡
   - 心跳检测

2. **健康评估**
   - 5维度综合评分
   - 分级健康状态(Healthy/Degraded/Unhealthy/Offline)
   - 自动问题诊断
   - 优化建议生成

3. **告警系统**
   - 4级告警(INFO/WARNING/CRITICAL/EMERGENCY)
   - 智能告警聚合
   - 告警状态追踪

4. **报告生成**
   - 系统整体概览
   - Agent状态分布
   - 历史趋势分析

## 使用方法

### 基础用法

```python
from agent_health_monitor import AgentHealthMonitor, AgentMetrics, HealthStatus

# 初始化监控器
monitor = AgentHealthMonitor(data_dir="./health_data")

# 注册Agent
monitor.register_agent(
    agent_id="research-lead",
    agent_type="research",
    capabilities=["web_search", "analysis"]
)
```

### 收集指标

```python
# 创建指标数据
metrics = AgentMetrics(
    agent_id="research-lead",
    agent_type="research",
    timestamp=time.time(),
    response_time_ms=450,
    task_success_rate=0.95,
    tasks_completed=150,
    tasks_failed=5,
    cpu_usage=0.45,
    memory_usage=0.62,
    disk_usage=0.30,
    active_tasks=3,
    queue_depth=12,
    avg_load_1m=0.8,
    session_count=2,
    last_heartbeat=time.time()
)

# 提交指标
monitor.collect_metrics(metrics)
```

### 健康检查

```python
# 检查单个Agent
health = monitor.check_health("research-lead")
print(f"状态: {health.status.value}")
print(f"评分: {health.overall_score:.1f}/100")
print("建议:")
for rec in health.recommendations:
    print(f"  - {rec}")

# 检查所有Agent
all_health = monitor.check_all_agents()
for agent_id, health in all_health.items():
    print(f"{agent_id}: {health.status.value}")
```

### 生成告警

```python
# 基于健康检查生成告警
alerts = monitor.generate_alerts(health)
for alert in alerts:
    print(f"[{alert.level.value}] {alert.title}")
    print(f"  {alert.message}")
```

### 系统报告

```python
# 生成完整报告
report = monitor.generate_report()
print(report)

# 获取系统概览
overview = monitor.get_system_overview()
print(f"总Agent数: {overview['total_agents']}")
print(f"平均健康分: {overview['avg_health_score']:.1f}")
print(f"活跃告警: {overview['alerts_count']}")
```

## 配置说明

### 健康阈值

```python
# 默认阈值配置
THRESHOLDS = {
    'response_time_ms': {'warning': 1000, 'critical': 3000},
    'task_success_rate': {'warning': 0.85, 'critical': 0.70},
    'cpu_usage': {'warning': 0.70, 'critical': 0.90},
    'memory_usage': {'warning': 0.80, 'critical': 0.95},
    'disk_usage': {'warning': 0.85, 'critical': 0.95},
    'queue_depth': {'warning': 50, 'critical': 100},
    'heartbeat_timeout': {'warning': 60, 'critical': 120}
}
```

### 评分权重

```python
# 健康评分权重
WEIGHTS = {
    'response_time': 0.15,
    'success_rate': 0.25,
    'resource_usage': 0.20,
    'load_balance': 0.20,
    'heartbeat': 0.20
}
```

## 健康状态定义

| 状态 | 分数范围 | 说明 |
|------|----------|------|
| HEALTHY | ≥90 | 健康运行 |
| DEGRADED | 70-89 | 性能下降，需关注 |
| UNHEALTHY | 50-69 | 不健康，需处理 |
| OFFLINE | <50 | 离线或严重故障 |

## 告警级别

| 级别 | 触发条件 | 响应时间 |
|------|----------|----------|
| INFO | 一般信息 | 无需立即响应 |
| WARNING | 性能警告 | 24小时内处理 |
| CRITICAL | 严重问题 | 1小时内处理 |
| EMERGENCY | 紧急故障 | 立即处理 |

## 集成示例

### 与Cron集成

```python
# 每小时执行健康检查
import schedule
import time

def hourly_health_check():
    monitor = AgentHealthMonitor()
    monitor.load_state()
    
    # 检查所有Agent
    for agent_id in monitor.agents.keys():
        health = monitor.check_health(agent_id)
        alerts = monitor.generate_alerts(health)
        
        # 发送严重告警
        for alert in alerts:
            if alert.level.value in ['critical', 'emergency']:
                send_alert_notification(alert)
    
    monitor.save_state()

schedule.every().hour.do(hourly_health_check)
```

### 与Heartbeat集成

```python
# 在Agent心跳中上报指标
def on_heartbeat(agent_id):
    metrics = collect_agent_metrics(agent_id)
    monitor.collect_metrics(metrics)
    
    health = monitor.check_health(agent_id)
    if health.status != HealthStatus.HEALTHY:
        logger.warning(f"Agent {agent_id} status: {health.status.value}")
```

## CLI使用

```bash
# 运行演示
python agent_health_monitor.py

# 输出示例:
# 🩺 Agent Health Monitor v1.0
# research-lead: healthy (评分: 92.5)
# data-eng: degraded (评分: 78.3)
#   💡 响应时间较高，考虑优化
```

## 文件结构

```
agent-health-monitor/
├── SKILL.md                    # 本文件
└── agent_health_monitor.py     # 主程序
```

## 版本信息

- **版本**: v1.0
- **发布日期**: 2026-02-27
- **作者**: KCGS (Kimi Claw Growth System)
- **状态**: 已部署

## 更新日志

- v1.0 (2026-02-27): 初始版本，实现核心监控功能
  - Agent注册与管理
  - 5维度健康检查
  - 分级告警系统
  - 报告生成
