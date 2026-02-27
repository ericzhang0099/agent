"""
性能追踪器模块
提供高级性能追踪和分析功能

功能：
- 装饰器方式追踪函数性能
- 上下文管理器方式追踪代码块
- 性能报告生成
- 趋势分析
"""

import time
import functools
import threading
from typing import Dict, List, Optional, Callable, Any, Union
from dataclasses import dataclass, field, asdict
from collections import defaultdict, deque
from enum import Enum
import json
import statistics


class MetricType(Enum):
    """指标类型"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    COUNTER = "counter"
    GAUGE = "gauge"
    HISTOGRAM = "histogram"


@dataclass
class FunctionMetrics:
    """函数性能指标"""
    function_name: str
    call_count: int = 0
    total_time_ms: float = 0.0
    avg_time_ms: float = 0.0
    min_time_ms: float = float('inf')
    max_time_ms: float = 0.0
    success_count: int = 0
    error_count: int = 0
    last_called: Optional[float] = None
    recent_latencies: deque = field(default_factory=lambda: deque(maxlen=100))
    
    def record_call(self, duration_ms: float, success: bool = True):
        """记录一次调用"""
        self.call_count += 1
        self.total_time_ms += duration_ms
        self.avg_time_ms = self.total_time_ms / self.call_count
        self.min_time_ms = min(self.min_time_ms, duration_ms)
        self.max_time_ms = max(self.max_time_ms, duration_ms)
        self.recent_latencies.append(duration_ms)
        self.last_called = time.time()
        
        if success:
            self.success_count += 1
        else:
            self.error_count += 1
    
    @property
    def success_rate(self) -> float:
        """成功率"""
        if self.call_count == 0:
            return 1.0
        return self.success_count / self.call_count
    
    @property
    def p95_latency(self) -> float:
        """P95延迟"""
        if not self.recent_latencies:
            return 0.0
        return self._percentile(list(self.recent_latencies), 0.95)
    
    @property
    def p99_latency(self) -> float:
        """P99延迟"""
        if not self.recent_latencies:
            return 0.0
        return self._percentile(list(self.recent_latencies), 0.99)
    
    @staticmethod
    def _percentile(data: List[float], percentile: float) -> float:
        """计算百分位数"""
        if not data:
            return 0.0
        sorted_data = sorted(data)
        index = int(len(sorted_data) * percentile)
        return sorted_data[min(index, len(sorted_data) - 1)]


@dataclass
class PerformanceSnapshot:
    """性能快照"""
    timestamp: float = field(default_factory=time.time)
    function_metrics: Dict[str, FunctionMetrics] = field(default_factory=dict)
    custom_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            "timestamp": self.timestamp,
            "function_metrics": {
                name: {
                    "function_name": m.function_name,
                    "call_count": m.call_count,
                    "avg_time_ms": round(m.avg_time_ms, 2),
                    "min_time_ms": round(m.min_time_ms, 2) if m.min_time_ms != float('inf') else 0,
                    "max_time_ms": round(m.max_time_ms, 2),
                    "success_rate": round(m.success_rate, 3),
                    "p95_latency": round(m.p95_latency, 2),
                    "p99_latency": round(m.p99_latency, 2),
                    "last_called": m.last_called
                }
                for name, m in self.function_metrics.items()
            },
            "custom_metrics": self.custom_metrics
        }


class PerformanceTracker:
    """
    性能追踪器
    
    提供全面的性能追踪能力：
    - 函数级性能追踪
    - 代码块性能追踪
    - 自定义指标收集
    - 性能报告生成
    """
    
    def __init__(self, history_size: int = 1000):
        self.history_size = history_size
        
        # 函数指标
        self._function_metrics: Dict[str, FunctionMetrics] = {}
        
        # 自定义指标
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, List[float]] = defaultdict(list)
        
        # 历史快照
        self._snapshots: deque = deque(maxlen=history_size)
        
        # 线程安全
        self._lock = threading.RLock()
        
        # 活跃追踪
        self._active_traces: Dict[str, float] = {}
    
    # ============ 装饰器接口 ============
    
    def track(
        self,
        function_name: Optional[str] = None,
        track_args: bool = False,
        track_return: bool = False
    ):
        """
        性能追踪装饰器
        
        用法：
            @tracker.track()
            def my_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            name = function_name or func.__qualname__
            
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                success = True
                
                try:
                    result = func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise e
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self._record_function_call(name, duration_ms, success)
            
            return wrapper
        return decorator
    
    def track_async(
        self,
        function_name: Optional[str] = None
    ):
        """
        异步函数性能追踪装饰器
        
        用法：
            @tracker.track_async()
            async def my_async_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            name = function_name or func.__qualname__
            
            @functools.wraps(func)
            async def wrapper(*args, **kwargs):
                start_time = time.perf_counter()
                success = True
                
                try:
                    result = await func(*args, **kwargs)
                    return result
                except Exception as e:
                    success = False
                    raise e
                finally:
                    duration_ms = (time.perf_counter() - start_time) * 1000
                    self._record_function_call(name, duration_ms, success)
            
            return wrapper
        return decorator
    
    def _record_function_call(self, name: str, duration_ms: float, success: bool):
        """记录函数调用"""
        with self._lock:
            if name not in self._function_metrics:
                self._function_metrics[name] = FunctionMetrics(function_name=name)
            
            self._function_metrics[name].record_call(duration_ms, success)
    
    # ============ 上下文管理器接口 ============
    
    class TraceContext:
        """追踪上下文管理器"""
        
        def __init__(self, tracker: 'PerformanceTracker', name: str):
            self.tracker = tracker
            self.name = name
            self.start_time: Optional[float] = None
            self.success = True
        
        def __enter__(self):
            self.start_time = time.perf_counter()
            self.tracker._active_traces[self.name] = time.time()
            return self
        
        def __exit__(self, exc_type, exc_val, exc_tb):
            if self.start_time:
                duration_ms = (time.perf_counter() - self.start_time) * 1000
                self.success = exc_type is None
                self.tracker._record_function_call(self.name, duration_ms, self.success)
            self.tracker._active_traces.pop(self.name, None)
            return False  # 不抑制异常
    
    def trace(self, name: str) -> 'PerformanceTracker.TraceContext':
        """
        创建追踪上下文
        
        用法：
            with tracker.trace("code_block_name"):
                # 要追踪的代码
                pass
        """
        return self.TraceContext(self, name)
    
    # ============ 自定义指标接口 ============
    
    def increment_counter(self, name: str, value: int = 1):
        """增加计数器"""
        with self._lock:
            self._counters[name] += value
    
    def set_gauge(self, name: str, value: float):
        """设置仪表盘值"""
        with self._lock:
            self._gauges[name] = value
    
    def record_histogram(self, name: str, value: float):
        """记录直方图值"""
        with self._lock:
            self._histograms[name].append(value)
            # 限制历史大小
            if len(self._histograms[name]) > 1000:
                self._histograms[name] = self._histograms[name][-1000:]
    
    # ============ 查询接口 ============
    
    def get_function_metrics(self, name: Optional[str] = None) -> Union[FunctionMetrics, Dict[str, FunctionMetrics]]:
        """获取函数指标"""
        with self._lock:
            if name:
                return self._function_metrics.get(name)
            return dict(self._function_metrics)
    
    def get_counter(self, name: str) -> int:
        """获取计数器值"""
        with self._lock:
            return self._counters.get(name, 0)
    
    def get_gauge(self, name: str) -> float:
        """获取仪表盘值"""
        with self._lock:
            return self._gauges.get(name, 0.0)
    
    def get_histogram_stats(self, name: str) -> Dict[str, float]:
        """获取直方图统计"""
        with self._lock:
            values = self._histograms.get(name, [])
            if not values:
                return {"count": 0, "min": 0, "max": 0, "avg": 0, "p95": 0, "p99": 0}
            
            sorted_values = sorted(values)
            return {
                "count": len(values),
                "min": min(values),
                "max": max(values),
                "avg": statistics.mean(values),
                "p95": sorted_values[int(len(sorted_values) * 0.95)],
                "p99": sorted_values[int(len(sorted_values) * 0.99)]
            }
    
    def get_active_traces(self) -> Dict[str, float]:
        """获取正在进行的追踪"""
        return dict(self._active_traces)
    
    def take_snapshot(self) -> PerformanceSnapshot:
        """创建性能快照"""
        with self._lock:
            snapshot = PerformanceSnapshot(
                function_metrics=dict(self._function_metrics),
                custom_metrics={
                    "counters": dict(self._counters),
                    "gauges": dict(self._gauges),
                    "histograms": {k: self.get_histogram_stats(k) for k in self._histograms.keys()},
                    "active_traces": len(self._active_traces)
                }
            )
            self._snapshots.append(snapshot)
            return snapshot
    
    # ============ 报告生成 ============
    
    def generate_report(self, top_n: int = 10) -> str:
        """生成性能报告"""
        with self._lock:
            lines = [
                "=" * 60,
                "           PERFORMANCE TRACKER REPORT",
                "=" * 60,
                f"Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]
            
            # 函数性能
            lines.append("-" * 60)
            lines.append("FUNCTION PERFORMANCE")
            lines.append("-" * 60)
            
            if self._function_metrics:
                # 按调用次数排序
                sorted_funcs = sorted(
                    self._function_metrics.items(),
                    key=lambda x: x[1].call_count,
                    reverse=True
                )[:top_n]
                
                lines.append(f"{'Function':<30} {'Calls':<8} {'Avg(ms)':<10} {'P95(ms)':<10} {'Success%':<10}")
                lines.append("-" * 60)
                
                for name, metrics in sorted_funcs:
                    lines.append(
                        f"{name:<30} {metrics.call_count:<8} "
                        f"{metrics.avg_time_ms:<10.2f} {metrics.p95_latency:<10.2f} "
                        f"{metrics.success_rate*100:<10.1f}"
                    )
            else:
                lines.append("No function metrics recorded yet.")
            
            lines.append("")
            
            # 自定义指标
            lines.append("-" * 60)
            lines.append("CUSTOM METRICS")
            lines.append("-" * 60)
            
            if self._counters:
                lines.append("Counters:")
                for name, value in sorted(self._counters.items()):
                    lines.append(f"  {name}: {value}")
            
            if self._gauges:
                lines.append("\nGauges:")
                for name, value in sorted(self._gauges.items()):
                    lines.append(f"  {name}: {value:.2f}")
            
            if not self._counters and not self._gauges:
                lines.append("No custom metrics recorded yet.")
            
            lines.append("")
            lines.append("=" * 60)
            
            return "\n".join(lines)
    
    def export_json(self) -> str:
        """导出为JSON格式"""
        snapshot = self.take_snapshot()
        return json.dumps(snapshot.to_dict(), indent=2, default=str)
    
    def get_summary(self) -> Dict[str, Any]:
        """获取摘要信息"""
        with self._lock:
            total_calls = sum(m.call_count for m in self._function_metrics.values())
            total_errors = sum(m.error_count for m in self._function_metrics.values())
            
            avg_latencies = [m.avg_time_ms for m in self._function_metrics.values() if m.call_count > 0]
            overall_avg_latency = statistics.mean(avg_latencies) if avg_latencies else 0
            
            return {
                "total_functions_tracked": len(self._function_metrics),
                "total_calls": total_calls,
                "total_errors": total_errors,
                "overall_success_rate": 1.0 - (total_errors / total_calls) if total_calls > 0 else 1.0,
                "overall_avg_latency_ms": round(overall_avg_latency, 2),
                "active_traces": len(self._active_traces),
                "counters_count": len(self._counters),
                "gauges_count": len(self._gauges),
                "histograms_count": len(self._histograms)
            }
    
    def reset(self):
        """重置所有指标"""
        with self._lock:
            self._function_metrics.clear()
            self._counters.clear()
            self._gauges.clear()
            self._histograms.clear()
            self._snapshots.clear()
            self._active_traces.clear()


# 便捷函数：创建全局追踪器
_global_tracker: Optional[PerformanceTracker] = None


def get_tracker() -> PerformanceTracker:
    """获取全局性能追踪器"""
    global _global_tracker
    if _global_tracker is None:
        _global_tracker = PerformanceTracker()
    return _global_tracker


def set_tracker(tracker: PerformanceTracker):
    """设置全局性能追踪器"""
    global _global_tracker
    _global_tracker = tracker


# 便捷装饰器
def track(function_name: Optional[str] = None):
    """便捷装饰器，使用全局追踪器"""
    return get_tracker().track(function_name)


def track_async(function_name: Optional[str] = None):
    """便捷异步装饰器，使用全局追踪器"""
    return get_tracker().track_async(function_name)


def trace(name: str):
    """便捷上下文管理器，使用全局追踪器"""
    return get_tracker().trace(name)


if __name__ == "__main__":
    # 测试代码
    print("=== PerformanceTracker Test ===\n")
    
    tracker = PerformanceTracker()
    
    # 测试装饰器
    @tracker.track()
    def test_function(n: int):
        time.sleep(0.01)
        return sum(range(n))
    
    # 调用测试函数
    for i in range(10):
        try:
            test_function(100 + i)
        except Exception:
            pass
    
    # 测试上下文管理器
    with tracker.trace("test_block"):
        time.sleep(0.05)
    
    # 测试自定义指标
    tracker.increment_counter("requests", 5)
    tracker.set_gauge("memory_usage", 1024.5)
    tracker.record_histogram("response_time", 150.0)
    tracker.record_histogram("response_time", 200.0)
    tracker.record_histogram("response_time", 180.0)
    
    # 生成报告
    print(tracker.generate_report())
    
    print("\n=== Summary ===")
    print(json.dumps(tracker.get_summary(), indent=2))
