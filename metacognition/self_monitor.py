"""
自我监控核心模块
实现元认知能力中的自我监控功能

功能：
1. 性能监控（准确率/延迟/资源）
2. 认知负荷检测
3. 异常检测机制
"""

import time
import threading
import json
from dataclasses import dataclass, field, asdict
from typing import Dict, List, Optional, Callable, Any
from enum import Enum
from collections import deque
import statistics


class MonitoringLevel(Enum):
    """监控级别"""
    LOW = "low"           # 基础监控
    MEDIUM = "medium"     # 标准监控
    HIGH = "high"         # 详细监控
    DEBUG = "debug"       # 调试级别


class CognitiveLoadLevel(Enum):
    """认知负荷级别"""
    IDLE = "idle"         # 空闲
    LOW = "low"           # 低负荷
    NORMAL = "normal"     # 正常
    HIGH = "high"         # 高负荷
    OVERLOAD = "overload" # 超载


class AnomalyType(Enum):
    """异常类型"""
    LATENCY_SPIKE = "latency_spike"      # 延迟突增
    ACCURACY_DROP = "accuracy_drop"      # 准确率下降
    RESOURCE_EXHAUSTION = "resource_exhaustion"  # 资源耗尽
    ERROR_RATE_HIGH = "error_rate_high"  # 错误率过高
    THROUGHPUT_DROP = "throughput_drop"  # 吞吐量下降
    MEMORY_LEAK = "memory_leak"          # 内存泄漏


@dataclass
class PerformanceMetrics:
    """性能指标数据类"""
    timestamp: float = field(default_factory=time.time)
    
    # 准确率指标
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    accuracy_rate: float = 1.0
    
    # 延迟指标
    avg_latency_ms: float = 0.0
    min_latency_ms: float = float('inf')
    max_latency_ms: float = 0.0
    p95_latency_ms: float = 0.0
    p99_latency_ms: float = 0.0
    
    # 资源指标
    cpu_usage_percent: float = 0.0
    memory_usage_mb: float = 0.0
    memory_usage_percent: float = 0.0
    
    # 吞吐量
    requests_per_second: float = 0.0
    tokens_per_second: float = 0.0


@dataclass
class CognitiveLoadMetrics:
    """认知负荷指标"""
    timestamp: float = field(default_factory=time.time)
    
    # 任务相关
    active_tasks: int = 0
    pending_tasks: int = 0
    completed_tasks: int = 0
    
    # 复杂度评估
    avg_task_complexity: float = 0.0  # 0-1
    max_task_complexity: float = 0.0
    
    # 上下文使用
    context_window_usage: float = 0.0  # 0-1
    context_tokens: int = 0
    
    # 推理深度
    reasoning_steps_avg: float = 0.0
    max_reasoning_depth: int = 0
    
    # 综合负荷评估
    overall_load: CognitiveLoadLevel = CognitiveLoadLevel.NORMAL
    load_score: float = 0.5  # 0-1


@dataclass
class AnomalyEvent:
    """异常事件"""
    timestamp: float = field(default_factory=time.time)
    anomaly_type: AnomalyType = AnomalyType.LATENCY_SPIKE
    severity: str = "medium"  # low, medium, high, critical
    description: str = ""
    metrics_snapshot: Dict = field(default_factory=dict)
    suggested_action: str = ""


class SelfMonitor:
    """
    自我监控核心类
    
    实现Agent的自我监控能力，包括：
    - 性能指标收集与分析
    - 认知负荷实时评估
    - 异常检测与告警
    """
    
    def __init__(
        self,
        monitoring_level: MonitoringLevel = MonitoringLevel.MEDIUM,
        history_window_size: int = 100,
        anomaly_threshold: float = 0.95
    ):
        self.monitoring_level = monitoring_level
        self.history_window_size = history_window_size
        self.anomaly_threshold = anomaly_threshold
        
        # 历史数据
        self.performance_history: deque = deque(maxlen=history_window_size)
        self.cognitive_load_history: deque = deque(maxlen=history_window_size)
        self.anomaly_history: deque = deque(maxlen=50)
        
        # 当前状态
        self.current_performance = PerformanceMetrics()
        self.current_cognitive_load = CognitiveLoadMetrics()
        
        # 监控线程
        self._monitoring_active = False
        self._monitor_thread: Optional[threading.Thread] = None
        self._lock = threading.RLock()
        
        # 回调函数
        self._anomaly_callbacks: List[Callable[[AnomalyEvent], None]] = []
        self._load_change_callbacks: List[Callable[[CognitiveLoadLevel, CognitiveLoadLevel], None]] = []
        
        # 统计
        self._latency_buffer: deque = deque(maxlen=100)
        self._request_times: deque = deque(maxlen=100)
        self._start_time = time.time()
    
    def start_monitoring(self, interval_seconds: float = 1.0):
        """启动监控线程"""
        if self._monitoring_active:
            return
        
        self._monitoring_active = True
        self._monitor_thread = threading.Thread(
            target=self._monitoring_loop,
            args=(interval_seconds,),
            daemon=True
        )
        self._monitor_thread.start()
    
    def stop_monitoring(self):
        """停止监控线程"""
        self._monitoring_active = False
        if self._monitor_thread:
            self._monitor_thread.join(timeout=2.0)
    
    def _monitoring_loop(self, interval: float):
        """监控主循环"""
        while self._monitoring_active:
            try:
                self._collect_system_metrics()
                self._detect_anomalies()
                time.sleep(interval)
            except Exception as e:
                print(f"[SelfMonitor] Monitoring error: {e}")
    
    def _collect_system_metrics(self):
        """收集系统指标"""
        try:
            import psutil
            
            with self._lock:
                # CPU 使用率
                self.current_performance.cpu_usage_percent = psutil.cpu_percent(interval=0.1)
                
                # 内存使用
                memory = psutil.virtual_memory()
                self.current_performance.memory_usage_mb = memory.used / (1024 * 1024)
                self.current_performance.memory_usage_percent = memory.percent
                
        except ImportError:
            pass  # psutil 未安装
    
    # ============ 性能监控接口 ============
    
    def record_request_start(self) -> float:
        """记录请求开始时间，返回开始时间戳"""
        return time.time()
    
    def record_request_end(
        self,
        start_time: float,
        success: bool = True,
        tokens_generated: int = 0
    ):
        """记录请求结束"""
        latency_ms = (time.time() - start_time) * 1000
        
        with self._lock:
            self.current_performance.total_requests += 1
            
            if success:
                self.current_performance.successful_requests += 1
            else:
                self.current_performance.failed_requests += 1
            
            # 更新准确率
            if self.current_performance.total_requests > 0:
                self.current_performance.accuracy_rate = (
                    self.current_performance.successful_requests /
                    self.current_performance.total_requests
                )
            
            # 更新延迟统计
            self._latency_buffer.append(latency_ms)
            self._update_latency_stats()
            
            # 更新吞吐量
            self._request_times.append(time.time())
            self._update_throughput()
            
            # 更新token生成速率
            if tokens_generated > 0 and latency_ms > 0:
                self.current_performance.tokens_per_second = (
                    tokens_generated / (latency_ms / 1000)
                )
    
    def _update_latency_stats(self):
        """更新延迟统计"""
        if not self._latency_buffer:
            return
        
        latencies = list(self._latency_buffer)
        self.current_performance.avg_latency_ms = statistics.mean(latencies)
        self.current_performance.min_latency_ms = min(latencies)
        self.current_performance.max_latency_ms = max(latencies)
        
        if len(latencies) >= 10:
            sorted_latencies = sorted(latencies)
            p95_idx = int(len(sorted_latencies) * 0.95)
            p99_idx = int(len(sorted_latencies) * 0.99)
            self.current_performance.p95_latency_ms = sorted_latencies[min(p95_idx, len(sorted_latencies)-1)]
            self.current_performance.p99_latency_ms = sorted_latencies[min(p99_idx, len(sorted_latencies)-1)]
    
    def _update_throughput(self):
        """更新吞吐量统计"""
        now = time.time()
        # 计算最近1分钟的请求数
        recent_requests = [t for t in self._request_times if now - t < 60]
        if recent_requests:
            self.current_performance.requests_per_second = len(recent_requests) / 60
    
    def get_performance_metrics(self) -> PerformanceMetrics:
        """获取当前性能指标"""
        with self._lock:
            # 保存历史记录
            self.performance_history.append(asdict(self.current_performance))
            return self.current_performance
    
    # ============ 认知负荷监控接口 ============
    
    def update_task_status(
        self,
        active_tasks: int,
        pending_tasks: int,
        completed_tasks: int
    ):
        """更新任务状态"""
        with self._lock:
            self.current_cognitive_load.active_tasks = active_tasks
            self.current_cognitive_load.pending_tasks = pending_tasks
            self.current_cognitive_load.completed_tasks = completed_tasks
    
    def record_task_complexity(self, complexity: float):
        """记录任务复杂度 (0-1)"""
        with self._lock:
            # 使用移动平均
            alpha = 0.3
            current = self.current_cognitive_load.avg_task_complexity
            self.current_cognitive_load.avg_task_complexity = (
                alpha * complexity + (1 - alpha) * current
            )
            self.current_cognitive_load.max_task_complexity = max(
                self.current_cognitive_load.max_task_complexity,
                complexity
            )
    
    def update_context_usage(self, tokens_used: int, max_tokens: int):
        """更新上下文使用情况"""
        with self._lock:
            self.current_cognitive_load.context_tokens = tokens_used
            if max_tokens > 0:
                self.current_cognitive_load.context_window_usage = tokens_used / max_tokens
    
    def record_reasoning_depth(self, steps: int):
        """记录推理深度"""
        with self._lock:
            alpha = 0.3
            current = self.current_cognitive_load.reasoning_steps_avg
            self.current_cognitive_load.reasoning_steps_avg = (
                alpha * steps + (1 - alpha) * current
            )
            self.current_cognitive_load.max_reasoning_depth = max(
                self.current_cognitive_load.max_reasoning_depth,
                steps
            )
    
    def evaluate_cognitive_load(self) -> CognitiveLoadMetrics:
        """
        评估当前认知负荷
        综合考虑多个因素计算负荷等级
        """
        with self._lock:
            load = self.current_cognitive_load
            
            # 计算负荷分数 (0-1)
            factors = []
            weights = []
            
            # 任务数量因子
            total_tasks = load.active_tasks + load.pending_tasks
            if total_tasks > 0:
                task_factor = min(total_tasks / 10, 1.0)
                factors.append(task_factor)
                weights.append(0.25)
            
            # 复杂度因子
            factors.append(load.avg_task_complexity)
            weights.append(0.20)
            
            # 上下文使用因子
            factors.append(load.context_window_usage)
            weights.append(0.25)
            
            # 推理深度因子
            reasoning_factor = min(load.reasoning_steps_avg / 20, 1.0)
            factors.append(reasoning_factor)
            weights.append(0.15)
            
            # 资源使用因子
            resource_factor = self.current_performance.memory_usage_percent / 100
            factors.append(resource_factor)
            weights.append(0.15)
            
            # 加权计算
            if factors and weights:
                load.load_score = sum(f * w for f, w in zip(factors, weights)) / sum(weights)
            
            # 确定负荷等级
            old_level = load.overall_load
            if load.load_score < 0.1:
                load.overall_load = CognitiveLoadLevel.IDLE
            elif load.load_score < 0.3:
                load.overall_load = CognitiveLoadLevel.LOW
            elif load.load_score < 0.6:
                load.overall_load = CognitiveLoadLevel.NORMAL
            elif load.load_score < 0.85:
                load.overall_load = CognitiveLoadLevel.HIGH
            else:
                load.overall_load = CognitiveLoadLevel.OVERLOAD
            
            # 触发回调
            if old_level != load.overall_load:
                for callback in self._load_change_callbacks:
                    try:
                        callback(old_level, load.overall_load)
                    except Exception as e:
                        print(f"[SelfMonitor] Load callback error: {e}")
            
            # 保存历史
            self.cognitive_load_history.append(asdict(load))
            
            return load
    
    # ============ 异常检测接口 ============
    
    def _detect_anomalies(self):
        """检测异常情况"""
        with self._lock:
            perf = self.current_performance
            
            # 检测延迟突增
            if len(self._latency_buffer) >= 10:
                recent_avg = statistics.mean(list(self._latency_buffer)[-5:])
                historical_avg = statistics.mean(self._latency_buffer)
                if historical_avg > 0 and recent_avg > historical_avg * 2:
                    self._report_anomaly(
                        AnomalyType.LATENCY_SPIKE,
                        "high" if recent_avg > historical_avg * 3 else "medium",
                        f"Latency spike detected: {recent_avg:.1f}ms vs avg {historical_avg:.1f}ms",
                        "Consider reducing concurrent tasks or optimizing processing"
                    )
            
            # 检测准确率下降
            if perf.total_requests >= 10 and perf.accuracy_rate < 0.8:
                self._report_anomaly(
                    AnomalyType.ACCURACY_DROP,
                    "high" if perf.accuracy_rate < 0.5 else "medium",
                    f"Accuracy dropped to {perf.accuracy_rate:.1%}",
                    "Review recent failures and consider model adjustment"
                )
            
            # 检测资源耗尽
            if perf.memory_usage_percent > 90:
                self._report_anomaly(
                    AnomalyType.RESOURCE_EXHAUSTION,
                    "critical" if perf.memory_usage_percent > 95 else "high",
                    f"High memory usage: {perf.memory_usage_percent:.1f}%",
                    "Clear cache, reduce batch size, or restart service"
                )
            
            # 检测错误率过高
            if perf.total_requests >= 10:
                error_rate = perf.failed_requests / perf.total_requests
                if error_rate > 0.2:
                    self._report_anomaly(
                        AnomalyType.ERROR_RATE_HIGH,
                        "high" if error_rate > 0.5 else "medium",
                        f"High error rate: {error_rate:.1%}",
                        "Check error logs and system health"
                    )
            
            # 检测吞吐量下降
            if len(self.performance_history) >= 5:
                recent_rps = perf.requests_per_second
                if recent_rps > 0:
                    historical_rps_list = [p.get('requests_per_second', 0) for p in list(self.performance_history)[-5:]]
                    if historical_rps_list:
                        historical_avg_rps = statistics.mean(historical_rps_list)
                        if historical_avg_rps > 0 and recent_rps < historical_avg_rps * 0.5:
                            self._report_anomaly(
                                AnomalyType.THROUGHPUT_DROP,
                                "medium",
                                f"Throughput dropped: {recent_rps:.2f} vs avg {historical_avg_rps:.2f} req/s",
                                "Check for bottlenecks or resource constraints"
                            )
    
    def _report_anomaly(
        self,
        anomaly_type: AnomalyType,
        severity: str,
        description: str,
        suggested_action: str
    ):
        """报告异常事件"""
        # 去重：相同类型的异常在1分钟内不重复报告
        recent_anomalies = [
            a for a in self.anomaly_history
            if a.anomaly_type == anomaly_type and time.time() - a.timestamp < 60
        ]
        if recent_anomalies:
            return
        
        event = AnomalyEvent(
            anomaly_type=anomaly_type,
            severity=severity,
            description=description,
            metrics_snapshot=asdict(self.current_performance),
            suggested_action=suggested_action
        )
        
        self.anomaly_history.append(event)
        
        # 触发回调
        for callback in self._anomaly_callbacks:
            try:
                callback(event)
            except Exception as e:
                print(f"[SelfMonitor] Anomaly callback error: {e}")
    
    def register_anomaly_callback(self, callback: Callable[[AnomalyEvent], None]):
        """注册异常回调函数"""
        self._anomaly_callbacks.append(callback)
    
    def register_load_change_callback(self, callback: Callable[[CognitiveLoadLevel, CognitiveLoadLevel], None]):
        """注册负荷变化回调函数"""
        self._load_change_callbacks.append(callback)
    
    # ============ 查询接口 ============
    
    def get_health_status(self) -> Dict[str, Any]:
        """获取健康状态摘要"""
        with self._lock:
            perf = self.current_performance
            load = self.current_cognitive_load
            
            # 计算健康分数 (0-100)
            health_score = 100
            
            # 准确率影响
            health_score -= (1 - perf.accuracy_rate) * 30
            
            # 延迟影响
            if perf.avg_latency_ms > 1000:
                health_score -= 20
            elif perf.avg_latency_ms > 500:
                health_score -= 10
            
            # 资源影响
            health_score -= perf.memory_usage_percent * 0.3
            
            # 负荷影响
            load_penalty = {
                CognitiveLoadLevel.IDLE: 0,
                CognitiveLoadLevel.LOW: 0,
                CognitiveLoadLevel.NORMAL: 5,
                CognitiveLoadLevel.HIGH: 15,
                CognitiveLoadLevel.OVERLOAD: 30
            }
            health_score -= load_penalty.get(load.overall_load, 0)
            
            health_score = max(0, min(100, health_score))
            
            # 确定状态
            if health_score >= 90:
                status = "healthy"
            elif health_score >= 70:
                status = "degraded"
            elif health_score >= 50:
                status = "warning"
            else:
                status = "critical"
            
            return {
                "status": status,
                "health_score": round(health_score, 1),
                "performance": asdict(perf),
                "cognitive_load": {
                    **asdict(load),
                    "overall_load": load.overall_load.value
                },
                "active_anomalies": len([a for a in self.anomaly_history if time.time() - a.timestamp < 300]),
                "uptime_seconds": time.time() - self._start_time
            }
    
    def get_anomaly_history(self, limit: int = 10) -> List[Dict]:
        """获取异常历史"""
        return [
            {
                "timestamp": a.timestamp,
                "type": a.anomaly_type.value,
                "severity": a.severity,
                "description": a.description,
                "suggested_action": a.suggested_action
            }
            for a in list(self.anomaly_history)[-limit:]
        ]
    
    def export_metrics(self) -> str:
        """导出所有指标为JSON"""
        return json.dumps({
            "current_performance": asdict(self.current_performance),
            "current_cognitive_load": {
                **asdict(self.current_cognitive_load),
                "overall_load": self.current_cognitive_load.overall_load.value
            },
            "performance_history": list(self.performance_history),
            "cognitive_load_history": list(self.cognitive_load_history),
            "anomaly_history": self.get_anomaly_history(50)
        }, indent=2, default=str)


# 全局监控器实例
_default_monitor: Optional[SelfMonitor] = None


def get_monitor() -> SelfMonitor:
    """获取默认监控器实例"""
    global _default_monitor
    if _default_monitor is None:
        _default_monitor = SelfMonitor()
    return _default_monitor


def set_monitor(monitor: SelfMonitor):
    """设置默认监控器实例"""
    global _default_monitor
    _default_monitor = monitor


if __name__ == "__main__":
    # 简单测试
    monitor = SelfMonitor(monitoring_level=MonitoringLevel.HIGH)
    
    print("=== SelfMonitor Test ===")
    
    # 模拟一些请求
    for i in range(10):
        start = monitor.record_request_start()
        time.sleep(0.01)
        monitor.record_request_end(start, success=(i % 5 != 0), tokens_generated=100)
    
    # 更新认知负荷
    monitor.update_task_status(active_tasks=3, pending_tasks=2, completed_tasks=10)
    monitor.record_task_complexity(0.7)
    monitor.update_context_usage(tokens_used=2000, max_tokens=8000)
    monitor.record_reasoning_depth(5)
    
    # 评估负荷
    load = monitor.evaluate_cognitive_load()
    print(f"Cognitive Load: {load.overall_load.value} (score: {load.load_score:.2f})")
    
    # 获取健康状态
    health = monitor.get_health_status()
    print(f"Health Status: {health['status']} (score: {health['health_score']})")
    
    # 导出指标
    print("\n=== Metrics Export ===")
    print(monitor.export_metrics()[:500] + "...")
