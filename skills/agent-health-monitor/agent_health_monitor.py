#!/usr/bin/env python3
"""
Agent Health Monitor - Agent健康状态监控系统
实时监控多Agent架构中的各个Agent健康状态
"""

import json
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from enum import Enum
import os

class HealthStatus(Enum):
    HEALTHY = "healthy"           # 健康
    DEGRADED = "degraded"         # 性能下降
    UNHEALTHY = "unhealthy"       # 不健康
    OFFLINE = "offline"           # 离线
    UNKNOWN = "unknown"           # 未知

class AlertLevel(Enum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"

@dataclass
class AgentMetrics:
    """Agent指标数据"""
    agent_id: str
    agent_type: str
    timestamp: float
    
    # 性能指标
    response_time_ms: float       # 响应时间(毫秒)
    task_success_rate: float      # 任务成功率(0-1)
    tasks_completed: int          # 完成任务数
    tasks_failed: int             # 失败任务数
    
    # 资源指标
    cpu_usage: float              # CPU使用率(0-1)
    memory_usage: float           # 内存使用率(0-1)
    disk_usage: float             # 磁盘使用率(0-1)
    
    # 负载指标
    active_tasks: int             # 活跃任务数
    queue_depth: int              # 队列深度
    avg_load_1m: float            # 1分钟平均负载
    
    # 会话指标
    session_count: int            # 活跃会话数
    last_heartbeat: float         # 最后心跳时间

@dataclass
class HealthCheck:
    """健康检查结果"""
    agent_id: str
    status: HealthStatus
    overall_score: float          # 总体健康分数(0-100)
    checks: Dict[str, Tuple[bool, str]]  # 各项检查结果
    timestamp: float
    recommendations: List[str]    # 建议操作

@dataclass
class Alert:
    """告警信息"""
    alert_id: str
    agent_id: str
    level: AlertLevel
    title: str
    message: str
    timestamp: float
    resolved: bool = False
    resolved_at: Optional[float] = None

class AgentHealthMonitor:
    """Agent健康监控器"""
    
    # 健康阈值配置
    THRESHOLDS = {
        'response_time_ms': {'warning': 1000, 'critical': 3000},
        'task_success_rate': {'warning': 0.85, 'critical': 0.70},
        'cpu_usage': {'warning': 0.70, 'critical': 0.90},
        'memory_usage': {'warning': 0.80, 'critical': 0.95},
        'disk_usage': {'warning': 0.85, 'critical': 0.95},
        'queue_depth': {'warning': 50, 'critical': 100},
        'heartbeat_timeout': {'warning': 60, 'critical': 120}  # 秒
    }
    
    # 权重配置
    WEIGHTS = {
        'response_time': 0.15,
        'success_rate': 0.25,
        'resource_usage': 0.20,
        'load_balance': 0.20,
        'heartbeat': 0.20
    }
    
    def __init__(self, data_dir: str = "./health_data"):
        self.data_dir = data_dir
        self.metrics_history: Dict[str, List[AgentMetrics]] = {}
        self.alerts: List[Alert] = []
        self.agents: Dict[str, dict] = {}
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
    def register_agent(self, agent_id: str, agent_type: str, 
                       capabilities: List[str] = None,
                       thresholds: Dict = None):
        """注册Agent到监控系统"""
        self.agents[agent_id] = {
            'agent_id': agent_id,
            'agent_type': agent_type,
            'capabilities': capabilities or [],
            'thresholds': thresholds or {},
            'registered_at': time.time(),
            'status': HealthStatus.UNKNOWN
        }
        self.metrics_history[agent_id] = []
        
    def collect_metrics(self, metrics: AgentMetrics) -> None:
        """收集Agent指标"""
        agent_id = metrics.agent_id
        
        if agent_id not in self.metrics_history:
            self.metrics_history[agent_id] = []
        
        # 保存指标
        self.metrics_history[agent_id].append(metrics)
        
        # 限制历史记录大小(保留最近1000条)
        if len(self.metrics_history[agent_id]) > 1000:
            self.metrics_history[agent_id] = self.metrics_history[agent_id][-1000:]
        
        # 更新Agent状态
        if agent_id in self.agents:
            self.agents[agent_id]['last_metrics'] = asdict(metrics)
    
    def check_health(self, agent_id: str) -> HealthCheck:
        """执行健康检查"""
        if agent_id not in self.metrics_history or not self.metrics_history[agent_id]:
            return HealthCheck(
                agent_id=agent_id,
                status=HealthStatus.UNKNOWN,
                overall_score=0,
                checks={},
                timestamp=time.time(),
                recommendations=["无可用指标数据，请检查Agent是否已注册"]
            )
        
        # 获取最新指标
        latest = self.metrics_history[agent_id][-1]
        
        checks = {}
        scores = {}
        recommendations = []
        
        # 1. 响应时间检查
        rt = latest.response_time_ms
        rt_threshold = self.THRESHOLDS['response_time_ms']
        if rt > rt_threshold['critical']:
            checks['response_time'] = (False, f"响应时间过高: {rt:.0f}ms > {rt_threshold['critical']}ms")
            scores['response_time'] = max(0, 100 - (rt - rt_threshold['warning']) / 10)
            recommendations.append("优化Agent处理逻辑或增加资源")
        elif rt > rt_threshold['warning']:
            checks['response_time'] = (False, f"响应时间警告: {rt:.0f}ms > {rt_threshold['warning']}ms")
            scores['response_time'] = 70
            recommendations.append("监控响应时间趋势")
        else:
            checks['response_time'] = (True, f"响应时间正常: {rt:.0f}ms")
            scores['response_time'] = 100
        
        # 2. 任务成功率检查
        sr = latest.task_success_rate
        sr_threshold = self.THRESHOLDS['task_success_rate']
        if sr < sr_threshold['critical']:
            checks['success_rate'] = (False, f"成功率过低: {sr:.1%} < {sr_threshold['critical']:.1%}")
            scores['success_rate'] = sr * 100
            recommendations.append(f"检查失败原因，最近失败{latest.tasks_failed}次")
        elif sr < sr_threshold['warning']:
            checks['success_rate'] = (False, f"成功率警告: {sr:.1%} < {sr_threshold['warning']:.1%}")
            scores['success_rate'] = 80
            recommendations.append("关注任务失败模式")
        else:
            checks['success_rate'] = (True, f"成功率正常: {sr:.1%}")
            scores['success_rate'] = 100
        
        # 3. 资源使用检查
        max_resource = max(latest.cpu_usage, latest.memory_usage, latest.disk_usage)
        if max_resource > self.THRESHOLDS['memory_usage']['critical']:
            checks['resource_usage'] = (False, f"资源使用率过高: {max_resource:.1%}")
            scores['resource_usage'] = max(0, 100 - (max_resource - 0.8) * 500)
            if latest.memory_usage > self.THRESHOLDS['memory_usage']['critical']:
                recommendations.append("内存使用率过高，考虑重启或扩容")
            if latest.cpu_usage > self.THRESHOLDS['cpu_usage']['critical']:
                recommendations.append("CPU使用率过高，检查是否有死循环")
        elif max_resource > self.THRESHOLDS['memory_usage']['warning']:
            checks['resource_usage'] = (False, f"资源使用率警告: {max_resource:.1%}")
            scores['resource_usage'] = 75
            recommendations.append("监控资源使用趋势")
        else:
            checks['resource_usage'] = (True, f"资源使用正常: CPU {latest.cpu_usage:.1%}, MEM {latest.memory_usage:.1%}")
            scores['resource_usage'] = 100
        
        # 4. 负载均衡检查
        queue = latest.queue_depth
        if queue > self.THRESHOLDS['queue_depth']['critical']:
            checks['load_balance'] = (False, f"队列积压严重: {queue} > {self.THRESHOLDS['queue_depth']['critical']}")
            scores['load_balance'] = max(0, 100 - queue / 2)
            recommendations.append("队列积压严重，需要扩容或限流")
        elif queue > self.THRESHOLDS['queue_depth']['warning']:
            checks['load_balance'] = (False, f"队列深度警告: {queue} > {self.THRESHOLDS['queue_depth']['warning']}")
            scores['load_balance'] = 70
            recommendations.append("关注队列增长趋势")
        else:
            checks['load_balance'] = (True, f"负载正常: 队列深度 {queue}")
            scores['load_balance'] = 100
        
        # 5. 心跳检查
        time_since_hb = time.time() - latest.last_heartbeat
        hb_threshold = self.THRESHOLDS['heartbeat_timeout']
        if time_since_hb > hb_threshold['critical']:
            checks['heartbeat'] = (False, f"心跳超时严重: {time_since_hb:.0f}s > {hb_threshold['critical']}s")
            scores['heartbeat'] = 0
            recommendations.append("Agent可能已离线，需要立即检查")
        elif time_since_hb > hb_threshold['warning']:
            checks['heartbeat'] = (False, f"心跳超时警告: {time_since_hb:.0f}s > {hb_threshold['warning']}s")
            scores['heartbeat'] = 50
            recommendations.append("检查Agent网络连接")
        else:
            checks['heartbeat'] = (True, f"心跳正常: {time_since_hb:.0f}s 前")
            scores['heartbeat'] = 100
        
        # 计算总体分数
        overall_score = sum(
            scores.get(k, 0) * self.WEIGHTS[k] 
            for k in self.WEIGHTS.keys()
        )
        
        # 确定状态
        if overall_score >= 90:
            status = HealthStatus.HEALTHY
        elif overall_score >= 70:
            status = HealthStatus.DEGRADED
        elif overall_score >= 50:
            status = HealthStatus.UNHEALTHY
        else:
            status = HealthStatus.OFFLINE
        
        # 更新Agent状态
        if agent_id in self.agents:
            self.agents[agent_id]['status'] = status
            self.agents[agent_id]['health_score'] = overall_score
        
        return HealthCheck(
            agent_id=agent_id,
            status=status,
            overall_score=overall_score,
            checks=checks,
            timestamp=time.time(),
            recommendations=recommendations if recommendations else ["系统运行正常"]
        )
    
    def check_all_agents(self) -> Dict[str, HealthCheck]:
        """检查所有Agent健康状态"""
        results = {}
        for agent_id in self.agents.keys():
            results[agent_id] = self.check_health(agent_id)
        return results
    
    def generate_alerts(self, health_check: HealthCheck) -> List[Alert]:
        """根据健康检查生成告警"""
        alerts = []
        agent_id = health_check.agent_id
        
        # 根据状态生成告警
        if health_check.status == HealthStatus.OFFLINE:
            alerts.append(Alert(
                alert_id=f"{agent_id}_offline_{int(time.time())}",
                agent_id=agent_id,
                level=AlertLevel.EMERGENCY,
                title=f"Agent {agent_id} 离线",
                message=f"健康评分: {health_check.overall_score:.1f}/100，Agent可能已停止响应",
                timestamp=time.time()
            ))
        elif health_check.status == HealthStatus.UNHEALTHY:
            alerts.append(Alert(
                alert_id=f"{agent_id}_unhealthy_{int(time.time())}",
                agent_id=agent_id,
                level=AlertLevel.CRITICAL,
                title=f"Agent {agent_id} 不健康",
                message=f"健康评分: {health_check.overall_score:.1f}/100，需要立即关注",
                timestamp=time.time()
            ))
        elif health_check.status == HealthStatus.DEGRADED:
            alerts.append(Alert(
                alert_id=f"{agent_id}_degraded_{int(time.time())}",
                agent_id=agent_id,
                level=AlertLevel.WARNING,
                title=f"Agent {agent_id} 性能下降",
                message=f"健康评分: {health_check.overall_score:.1f}/100，建议优化",
                timestamp=time.time()
            ))
        
        # 为每个失败的检查生成详细告警
        for check_name, (passed, message) in health_check.checks.items():
            if not passed:
                level = AlertLevel.WARNING
                if "严重" in message or "过高" in message or "过低" in message:
                    level = AlertLevel.CRITICAL
                
                alerts.append(Alert(
                    alert_id=f"{agent_id}_{check_name}_{int(time.time())}",
                    agent_id=agent_id,
                    level=level,
                    title=f"Agent {agent_id} - {check_name} 异常",
                    message=message,
                    timestamp=time.time()
                ))
        
        # 保存告警
        self.alerts.extend(alerts)
        return alerts
    
    def get_system_overview(self) -> Dict:
        """获取系统整体概览"""
        if not self.agents:
            return {
                'total_agents': 0,
                'status_counts': {},
                'avg_health_score': 0,
                'alerts_count': len([a for a in self.alerts if not a.resolved])
            }
        
        status_counts = {status: 0 for status in HealthStatus}
        total_score = 0
        
        for agent_id, agent_info in self.agents.items():
            status = agent_info.get('status', HealthStatus.UNKNOWN)
            status_counts[status] += 1
            total_score += agent_info.get('health_score', 0)
        
        active_alerts = len([a for a in self.alerts if not a.resolved])
        
        return {
            'timestamp': datetime.now().isoformat(),
            'total_agents': len(self.agents),
            'status_counts': {k.value: v for k, v in status_counts.items()},
            'avg_health_score': total_score / len(self.agents) if self.agents else 0,
            'alerts_count': active_alerts,
            'agents_detail': self.agents
        }
    
    def generate_report(self) -> str:
        """生成健康监控报告"""
        overview = self.get_system_overview()
        
        lines = []
        lines.append("=" * 70)
        lines.append("🩺 Agent Health Monitor - 系统健康报告")
        lines.append(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        lines.append("=" * 70)
        
        # 系统概览
        lines.append("\n📊 系统概览")
        lines.append("-" * 70)
        lines.append(f"总Agent数: {overview['total_agents']}")
        lines.append(f"平均健康分: {overview['avg_health_score']:.1f}/100")
        lines.append(f"活跃告警: {overview['alerts_count']}")
        
        # 状态分布
        lines.append("\n📈 Agent状态分布")
        lines.append("-" * 70)
        status_emojis = {
            'healthy': '🟢',
            'degraded': '🟡',
            'unhealthy': '🟠',
            'offline': '🔴',
            'unknown': '⚪'
        }
        for status, count in overview['status_counts'].items():
            if count > 0:
                emoji = status_emojis.get(status, '⚪')
                lines.append(f"{emoji} {status:12s}: {count}")
        
        # 各Agent详情
        lines.append("\n🔍 Agent详情")
        lines.append("-" * 70)
        for agent_id, agent_info in self.agents.items():
            status = agent_info.get('status', HealthStatus.UNKNOWN)
            score = agent_info.get('health_score', 0)
            emoji = status_emojis.get(status.value, '⚪')
            agent_type = agent_info.get('agent_type', 'unknown')
            
            bar = "█" * int(score / 10) + "░" * (10 - int(score / 10))
            lines.append(f"{emoji} {agent_id:20s} ({agent_type:15s}) |{bar}| {score:.0f}")
        
        # 活跃告警
        if overview['alerts_count'] > 0:
            lines.append("\n🚨 活跃告警")
            lines.append("-" * 70)
            level_emojis = {
                'info': 'ℹ️',
                'warning': '⚠️',
                'critical': '🔴',
                'emergency': '🚨'
            }
            for alert in self.alerts:
                if not alert.resolved:
                    emoji = level_emojis.get(alert.level.value, '⚪')
                    time_str = datetime.fromtimestamp(alert.timestamp).strftime('%H:%M:%S')
                    lines.append(f"{emoji} [{time_str}] {alert.title}")
                    lines.append(f"   {alert.message}")
        
        lines.append("\n" + "=" * 70)
        
        return "\n".join(lines)
    
    def save_state(self) -> None:
        """保存监控状态到文件"""
        state = {
            'agents': self.agents,
            'alerts': [asdict(a) for a in self.alerts],
            'saved_at': time.time()
        }
        
        filepath = os.path.join(self.data_dir, 'monitor_state.json')
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(state, f, indent=2, ensure_ascii=False, default=str)
    
    def load_state(self) -> None:
        """从文件加载监控状态"""
        filepath = os.path.join(self.data_dir, 'monitor_state.json')
        if os.path.exists(filepath):
            with open(filepath, 'r', encoding='utf-8') as f:
                state = json.load(f)
                self.agents = state.get('agents', {})
                # 告警需要重新构建
                alerts_data = state.get('alerts', [])
                self.alerts = [Alert(**a) for a in alerts_data]


# 全局监控器实例
monitor = AgentHealthMonitor()

if __name__ == '__main__':
    print("🩺 Agent Health Monitor v1.0")
    print("=" * 70)
    
    # 演示：注册几个Agent
    monitor.register_agent("research-lead", "research", 
                          capabilities=["web_search", "analysis", "reporting"])
    monitor.register_agent("data-eng", "data",
                          capabilities=["data_processing", "etl", "analytics"])
    monitor.register_agent("quant-strat", "quant",
                          capabilities=["modeling", "backtesting", "risk_analysis"])
    
    # 模拟收集指标
    import random
    for agent_id in monitor.agents.keys():
        metrics = AgentMetrics(
            agent_id=agent_id,
            agent_type=monitor.agents[agent_id]['agent_type'],
            timestamp=time.time(),
            response_time_ms=random.uniform(200, 1500),
            task_success_rate=random.uniform(0.85, 0.99),
            tasks_completed=random.randint(100, 500),
            tasks_failed=random.randint(0, 20),
            cpu_usage=random.uniform(0.3, 0.8),
            memory_usage=random.uniform(0.4, 0.9),
            disk_usage=random.uniform(0.2, 0.6),
            active_tasks=random.randint(1, 10),
            queue_depth=random.randint(0, 80),
            avg_load_1m=random.uniform(0.5, 2.0),
            session_count=random.randint(1, 5),
            last_heartbeat=time.time() - random.uniform(10, 100)
        )
        monitor.collect_metrics(metrics)
    
    # 执行健康检查
    print("\n执行健康检查...")
    for agent_id in monitor.agents.keys():
        health = monitor.check_health(agent_id)
        alerts = monitor.generate_alerts(health)
        print(f"\n{agent_id}: {health.status.value} (评分: {health.overall_score:.1f})")
        for rec in health.recommendations[:2]:
            print(f"  💡 {rec}")
    
    # 生成报告
    print("\n" + monitor.generate_report())
    
    # 保存状态
    monitor.save_state()
    print(f"\n💾 监控状态已保存到 {monitor.data_dir}/monitor_state.json")
