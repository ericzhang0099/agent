#!/usr/bin/env python3
"""
Log Anomaly Detector - 日志分析与异常检测系统
自动分析系统日志，检测异常模式并生成报告

Author: KCGS
Version: 1.0.0
Date: 2026-02-27
"""

import re
import os
import json
import hashlib
import asyncio
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Tuple, Callable, Any
from dataclasses import dataclass, field, asdict
from enum import Enum
from collections import defaultdict, Counter
import statistics


class AnomalyLevel(Enum):
    """异常告警级别"""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    EMERGENCY = "emergency"


class AnomalyType(Enum):
    """异常类型"""
    ERROR_SPIKE = "error_spike"
    EXCEPTION = "exception"
    TIMEOUT = "timeout"
    MEMORY_ISSUE = "memory_issue"
    CONNECTION_ISSUE = "connection_issue"
    FREQUENCY_ANOMALY = "frequency_anomaly"
    PATTERN_MATCH = "pattern_match"


@dataclass
class LogEntry:
    """日志条目"""
    timestamp: datetime
    level: str
    message: str
    source: str
    raw_line: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    """异常检测结果"""
    id: str
    level: AnomalyLevel
    type: AnomalyType
    timestamp: datetime
    source: str
    description: str
    log_entry: Optional[LogEntry]
    impact: str
    suggested_action: Optional[str] = None
    related_anomalies: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "level": self.level.value,
            "type": self.type.value,
            "timestamp": self.timestamp.isoformat(),
            "source": self.source,
            "description": self.description,
            "impact": self.impact,
            "suggested_action": self.suggested_action,
            "related_anomalies": self.related_anomalies
        }


@dataclass
class ErrorStats:
    """错误统计"""
    total_entries: int
    error_count: int
    warning_count: int
    unique_error_types: int
    top_errors: List[Dict]
    error_rate: float  # 错误率
    trend: str  # "increasing", "decreasing", "stable"


@dataclass
class AnomalyConfig:
    """异常检测配置"""
    error_patterns: List[str] = field(default_factory=lambda: [
        r"ERROR|FATAL|CRITICAL",
        r"Exception|Traceback",
        r"timeout|timed out",
        r"memory leak|out of memory",
        r"connection refused|ECONNREFUSED"
    ])
    
    frequency_thresholds: Dict[str, int] = field(default_factory=lambda: {
        "error_per_minute": 10,
        "warning_per_minute": 50,
        "unique_errors": 5
    })
    
    time_series_config: Dict[str, float] = field(default_factory=lambda: {
        "window_size": 10,
        "std_threshold": 3.0,
        "trend_change_threshold": 0.3
    })


class LogParser:
    """日志解析器"""
    
    def __init__(self):
        self.parsers = {
            "json": self._parse_json,
            "plain": self._parse_plain,
            "syslog": self._parse_syslog
        }
    
    def detect_format(self, sample_lines: List[str]) -> str:
        """检测日志格式"""
        if not sample_lines:
            return "plain"
        
        # 尝试JSON格式
        try:
            json.loads(sample_lines[0])
            return "json"
        except:
            pass
        
        # 检测syslog格式
        if re.match(r'^\w+\s+\d+\s+\d{2}:\d{2}:\d{2}', sample_lines[0]):
            return "syslog"
        
        return "plain"
    
    def _parse_json(self, line: str, source: str) -> Optional[LogEntry]:
        """解析JSON格式日志"""
        try:
            data = json.loads(line)
            return LogEntry(
                timestamp=self._parse_timestamp(data.get('timestamp', '')),
                level=data.get('level', 'INFO'),
                message=data.get('message', ''),
                source=source,
                raw_line=line,
                metadata={k: v for k, v in data.items() if k not in ['timestamp', 'level', 'message']}
            )
        except:
            return None
    
    def _parse_plain(self, line: str, source: str) -> Optional[LogEntry]:
        """解析普通文本格式日志"""
        # 尝试匹配常见格式: 2024-01-01 10:00:00 INFO message
        patterns = [
            r'(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.*)',
            r'\[(\d{4}-\d{2}-\d{2}[\sT]\d{2}:\d{2}:\d{2})\]\s+(\w+)\s+(.*)',
            r'(\d{2}/\d{2}/\d{4}\s+\d{2}:\d{2}:\d{2})\s+(\w+)\s+(.*)'
        ]
        
        for pattern in patterns:
            match = re.match(pattern, line)
            if match:
                return LogEntry(
                    timestamp=self._parse_timestamp(match.group(1)),
                    level=match.group(2).upper(),
                    message=match.group(3),
                    source=source,
                    raw_line=line
                )
        
        # 无法解析，作为原始日志
        return LogEntry(
            timestamp=datetime.now(),
            level="UNKNOWN",
            message=line,
            source=source,
            raw_line=line
        )
    
    def _parse_syslog(self, line: str, source: str) -> Optional[LogEntry]:
        """解析syslog格式"""
        pattern = r'^(\w+\s+\d+\s+\d{2}:\d{2}:\d{2})\s+(\S+)\s+(.*)'
        match = re.match(pattern, line)
        if match:
            return LogEntry(
                timestamp=self._parse_timestamp(match.group(1)),
                level="INFO",
                message=match.group(3),
                source=source,
                raw_line=line
            )
        return None
    
    def _parse_timestamp(self, ts_str: str) -> datetime:
        """解析时间戳"""
        formats = [
            "%Y-%m-%d %H:%M:%S",
            "%Y-%m-%dT%H:%M:%S",
            "%Y-%m-%dT%H:%M:%S.%f",
            "%m/%d/%Y %H:%M:%S",
            "%b %d %H:%M:%S"
        ]
        
        for fmt in formats:
            try:
                return datetime.strptime(ts_str, fmt)
            except:
                continue
        
        return datetime.now()
    
    def parse_file(self, file_path: str, format_type: Optional[str] = None) -> List[LogEntry]:
        """解析日志文件"""
        entries = []
        
        if not os.path.exists(file_path):
            return entries
        
        # 检测格式
        if format_type is None:
            with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
                sample = [f.readline().strip() for _ in range(5)]
                format_type = self.detect_format(sample)
        
        # 解析文件
        parser = self.parsers.get(format_type, self._parse_plain)
        with open(file_path, 'r', encoding='utf-8', errors='ignore') as f:
            for line in f:
                line = line.strip()
                if line:
                    entry = parser(line, file_path)
                    if entry:
                        entries.append(entry)
        
        return entries


class AnomalyDetector:
    """异常检测引擎"""
    
    def __init__(self, config: AnomalyConfig):
        self.config = config
        self.compiled_patterns = [re.compile(p, re.IGNORECASE) for p in config.error_patterns]
    
    def detect_pattern_anomalies(self, entries: List[LogEntry]) -> List[Anomaly]:
        """基于模式匹配检测异常"""
        anomalies = []
        
        for entry in entries:
            for i, pattern in enumerate(self.compiled_patterns):
                if pattern.search(entry.message):
                    anomaly_type = self._get_anomaly_type(i)
                    level = self._get_level_for_type(anomaly_type)
                    
                    anomaly = Anomaly(
                        id=self._generate_id(entry),
                        level=level,
                        type=anomaly_type,
                        timestamp=entry.timestamp,
                        source=entry.source,
                        description=f"检测到 {anomaly_type.value}: {entry.message[:100]}",
                        log_entry=entry,
                        impact=self._assess_impact(anomaly_type),
                        suggested_action=self._get_suggested_action(anomaly_type)
                    )
                    anomalies.append(anomaly)
        
        return anomalies
    
    def detect_frequency_anomalies(self, entries: List[LogEntry], window_minutes: int = 5) -> List[Anomaly]:
        """基于频率检测异常"""
        anomalies = []
        
        # 按时间窗口分组
        windows = defaultdict(list)
        for entry in entries:
            window_key = entry.timestamp.replace(
                second=0, microsecond=0
            ).replace(minute=(entry.timestamp.minute // window_minutes) * window_minutes)
            windows[window_key].append(entry)
        
        # 检测错误频率异常
        for window, window_entries in sorted(windows.items()):
            error_count = sum(1 for e in window_entries if e.level in ['ERROR', 'FATAL', 'CRITICAL'])
            
            if error_count > self.config.frequency_thresholds.get('error_per_minute', 10):
                anomaly = Anomaly(
                    id=f"freq_{window.isoformat()}",
                    level=AnomalyLevel.WARNING,
                    type=AnomalyType.FREQUENCY_ANOMALY,
                    timestamp=window,
                    source=window_entries[0].source if window_entries else "unknown",
                    description=f"错误频率异常: {error_count} 个错误在 {window_minutes} 分钟内",
                    log_entry=None,
                    impact="可能存在系统性问题或突发故障",
                    suggested_action="检查系统状态和最近部署的变更"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def detect_time_series_anomalies(self, entries: List[LogEntry]) -> List[Anomaly]:
        """基于时序分析检测异常"""
        anomalies = []
        
        if len(entries) < self.config.time_series_config.get('window_size', 10):
            return anomalies
        
        # 按分钟统计错误数
        minute_counts = defaultdict(int)
        for entry in entries:
            if entry.level in ['ERROR', 'FATAL', 'CRITICAL']:
                minute_key = entry.timestamp.replace(second=0, microsecond=0)
                minute_counts[minute_key] += 1
        
        if not minute_counts:
            return anomalies
        
        # 计算统计值
        counts = list(minute_counts.values())
        mean = statistics.mean(counts)
        std = statistics.stdev(counts) if len(counts) > 1 else 0
        
        if std == 0:
            return anomalies
        
        # 检测Z-score异常
        threshold = self.config.time_series_config.get('std_threshold', 3.0)
        for timestamp, count in minute_counts.items():
            z_score = (count - mean) / std
            if abs(z_score) > threshold:
                level = AnomalyLevel.CRITICAL if z_score > 0 else AnomalyLevel.INFO
                anomaly = Anomaly(
                    id=f"ts_{timestamp.isoformat()}",
                    level=level,
                    type=AnomalyType.FREQUENCY_ANOMALY,
                    timestamp=timestamp,
                    source=entries[0].source if entries else "unknown",
                    description=f"时序异常: {count} 个错误 (Z-score: {z_score:.2f})",
                    log_entry=None,
                    impact="错误数量显著偏离正常范围" if z_score > 0 else "错误数量异常低",
                    suggested_action="调查异常时段的系统活动"
                )
                anomalies.append(anomaly)
        
        return anomalies
    
    def _get_anomaly_type(self, pattern_index: int) -> AnomalyType:
        """根据模式索引获取异常类型"""
        type_map = {
            0: AnomalyType.ERROR_SPIKE,
            1: AnomalyType.EXCEPTION,
            2: AnomalyType.TIMEOUT,
            3: AnomalyType.MEMORY_ISSUE,
            4: AnomalyType.CONNECTION_ISSUE
        }
        return type_map.get(pattern_index, AnomalyType.PATTERN_MATCH)
    
    def _get_level_for_type(self, anomaly_type: AnomalyType) -> AnomalyLevel:
        """根据异常类型获取告警级别"""
        level_map = {
            AnomalyType.ERROR_SPIKE: AnomalyLevel.WARNING,
            AnomalyType.EXCEPTION: AnomalyLevel.CRITICAL,
            AnomalyType.TIMEOUT: AnomalyLevel.WARNING,
            AnomalyType.MEMORY_ISSUE: AnomalyLevel.CRITICAL,
            AnomalyType.CONNECTION_ISSUE: AnomalyLevel.WARNING,
            AnomalyType.FREQUENCY_ANOMALY: AnomalyLevel.WARNING,
            AnomalyType.PATTERN_MATCH: AnomalyLevel.INFO
        }
        return level_map.get(anomaly_type, AnomalyLevel.INFO)
    
    def _assess_impact(self, anomaly_type: AnomalyType) -> str:
        """评估影响"""
        impact_map = {
            AnomalyType.ERROR_SPIKE: "可能影响系统稳定性",
            AnomalyType.EXCEPTION: "功能异常，需要立即处理",
            AnomalyType.TIMEOUT: "响应延迟，影响用户体验",
            AnomalyType.MEMORY_ISSUE: "可能导致服务崩溃",
            AnomalyType.CONNECTION_ISSUE: "外部依赖问题",
            AnomalyType.FREQUENCY_ANOMALY: "系统行为异常",
            AnomalyType.PATTERN_MATCH: "需要关注的日志事件"
        }
        return impact_map.get(anomaly_type, "未知影响")
    
    def _get_suggested_action(self, anomaly_type: AnomalyType) -> Optional[str]:
        """获取建议操作"""
        action_map = {
            AnomalyType.ERROR_SPIKE: "检查错误日志详情和相关服务状态",
            AnomalyType.EXCEPTION: "查看完整堆栈跟踪，定位代码问题",
            AnomalyType.TIMEOUT: "检查网络连接和依赖服务响应时间",
            AnomalyType.MEMORY_ISSUE: "检查内存使用情况，考虑重启服务",
            AnomalyType.CONNECTION_ISSUE: "验证网络配置和外部服务可用性",
            AnomalyType.FREQUENCY_ANOMALY: "分析异常时段的系统负载和变更",
            AnomalyType.PATTERN_MATCH: "根据具体情况评估是否需要处理"
        }
        return action_map.get(anomaly_type)
    
    def _generate_id(self, entry: LogEntry) -> str:
        """生成异常ID"""
        content = f"{entry.timestamp.isoformat()}:{entry.source}:{entry.message[:50]}"
        return hashlib.md5(content.encode()).hexdigest()[:12]


@dataclass
class AnalysisReport:
    """分析报告"""
    generated_at: datetime
    time_range: Tuple[datetime, datetime]
    total_entries: int
    anomalies: List[Anomaly]
    error_stats: ErrorStats
    recommendations: List[str]
    
    def to_dict(self) -> Dict:
        return {
            "generated_at": self.generated_at.isoformat(),
            "time_range": [self.time_range[0].isoformat(), self.time_range[1].isoformat()],
            "total_entries": self.total_entries,
            "anomaly_count": len(self.anomalies),
            "anomalies_by_level": self._count_by_level(),
            "error_stats": asdict(self.error_stats) if self.error_stats else None,
            "recommendations": self.recommendations
        }
    
    def _count_by_level(self) -> Dict[str, int]:
        counts = defaultdict(int)
        for a in self.anomalies:
            counts[a.level.value] += 1
        return dict(counts)
    
    def to_html(self) -> str:
        """生成HTML报告"""
        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>日志分析报告 - {self.generated_at.strftime('%Y-%m-%d %H:%M')}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; background: #f5f5f5; }}
        .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 20px; border-radius: 8px; }}
        h1 {{ color: #333; border-bottom: 2px solid #4CAF50; padding-bottom: 10px; }}
        .summary {{ display: grid; grid-template-columns: repeat(4, 1fr); gap: 15px; margin: 20px 0; }}
        .stat-box {{ background: #f8f9fa; padding: 15px; border-radius: 6px; text-align: center; }}
        .stat-value {{ font-size: 24px; font-weight: bold; color: #4CAF50; }}
        .stat-label {{ color: #666; margin-top: 5px; }}
        .anomaly {{ margin: 10px 0; padding: 15px; border-left: 4px solid; border-radius: 4px; }}
        .anomaly.critical {{ background: #ffebee; border-color: #f44336; }}
        .anomaly.warning {{ background: #fff3e0; border-color: #ff9800; }}
        .anomaly.info {{ background: #e3f2fd; border-color: #2196f3; }}
        .anomaly.emergency {{ background: #fce4ec; border-color: #e91e63; }}
        .level-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 12px; font-weight: bold; }}
        .level-critical {{ background: #f44336; color: white; }}
        .level-warning {{ background: #ff9800; color: white; }}
        .level-info {{ background: #2196f3; color: white; }}
        .level-emergency {{ background: #e91e63; color: white; }}
        .recommendations {{ background: #e8f5e9; padding: 15px; border-radius: 6px; margin-top: 20px; }}
        .recommendations ul {{ margin: 10px 0; }}
        .recommendations li {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>📊 日志分析报告</h1>
        <p>生成时间: {self.generated_at.strftime('%Y-%m-%d %H:%M:%S')}</p>
        <p>分析范围: {self.time_range[0].strftime('%Y-%m-%d %H:%M')} ~ {self.time_range[1].strftime('%Y-%m-%d %H:%M')}</p>
        
        <div class="summary">
            <div class="stat-box">
                <div class="stat-value">{self.total_entries}</div>
                <div class="stat-label">总日志数</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len(self.anomalies)}</div>
                <div class="stat-label">异常数量</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{len([a for a in self.anomalies if a.level == AnomalyLevel.CRITICAL])}</div>
                <div class="stat-label">严重异常</div>
            </div>
            <div class="stat-box">
                <div class="stat-value">{self.error_stats.error_rate:.1%}</div>
                <div class="stat-label">错误率</div>
            </div>
        </div>
        
        <h2>🔍 检测到的异常</h2>
"""
        
        for anomaly in sorted(self.anomalies, key=lambda x: x.timestamp, reverse=True):
            level_class = anomaly.level.value
            html += f"""
        <div class="anomaly {level_class}">
            <span class="level-badge level-{level_class}">{anomaly.level.value.upper()}</span>
            <strong>{anomaly.type.value}</strong> - {anomaly.timestamp.strftime('%Y-%m-%d %H:%M:%S')}
            <p><strong>描述:</strong> {anomaly.description}</p>
            <p><strong>影响:</strong> {anomaly.impact}</p>
            {f"<p><strong>建议:</strong> {anomaly.suggested_action}</p>" if anomaly.suggested_action else ""}
        </div>
"""
        
        if self.recommendations:
            html += """
        <div class="recommendations">
            <h3>💡 改进建议</h3>
            <ul>
"""
            for rec in self.recommendations:
                html += f"                <li>{rec}</li>\n"
            html += """            </ul>
        </div>
"""
        
        html += """
    </div>
</body>
</html>
"""
        return html


class LogAnomalyDetector:
    """日志异常检测器主类"""
    
    def __init__(self, data_dir: str = "./log_analysis", config: Optional[AnomalyConfig] = None):
        self.data_dir = data_dir
        self.config = config or AnomalyConfig()
        self.parser = LogParser()
        self.detector = AnomalyDetector(self.config)
        self.anomaly_callbacks: Dict[AnomalyLevel, List[Callable]] = defaultdict(list)
        self._monitoring = False
        self._monitor_task = None
        
        os.makedirs(data_dir, exist_ok=True)
    
    def analyze_log_file(self, log_path: str, log_type: Optional[str] = None, 
                         time_window: Optional[str] = None) -> AnalysisReport:
        """分析单个日志文件"""
        entries = self.parser.parse_file(log_path, log_type)
        
        # 应用时间窗口过滤
        if time_window:
            entries = self._filter_by_time_window(entries, time_window)
        
        return self._analyze_entries(entries)
    
    def analyze_multiple_files(self, log_paths: List[str], 
                               time_range: Optional[Tuple[datetime, datetime]] = None) -> AnalysisReport:
        """分析多个日志文件"""
        all_entries = []
        
        for path in log_paths:
            entries = self.parser.parse_file(path)
            if time_range:
                entries = [e for e in entries if time_range[0] <= e.timestamp <= time_range[1]]
            all_entries.extend(entries)
        
        all_entries.sort(key=lambda x: x.timestamp)
        return self._analyze_entries(all_entries)
    
    def _analyze_entries(self, entries: List[LogEntry]) -> AnalysisReport:
        """分析日志条目"""
        if not entries:
            return AnalysisReport(
                generated_at=datetime.now(),
                time_range=(datetime.now(), datetime.now()),
                total_entries=0,
                anomalies=[],
                error_stats=None,
                recommendations=["无日志数据可供分析"]
            )
        
        # 执行多种异常检测
        anomalies = []
        anomalies.extend(self.detector.detect_pattern_anomalies(entries))
        anomalies.extend(self.detector.detect_frequency_anomalies(entries))
        anomalies.extend(self.detector.detect_time_series_anomalies(entries))
        
        # 去重
        seen_ids = set()
        unique_anomalies = []
        for a in anomalies:
            if a.id not in seen_ids:
                seen_ids.add(a.id)
                unique_anomalies.append(a)
        
        # 生成统计
        error_stats = self._calculate_error_stats(entries)
        
        # 生成建议
        recommendations = self._generate_recommendations(unique_anomalies, error_stats)
        
        return AnalysisReport(
            generated_at=datetime.now(),
            time_range=(entries[0].timestamp, entries[-1].timestamp),
            total_entries=len(entries),
            anomalies=unique_anomalies,
            error_stats=error_stats,
            recommendations=recommendations
        )
    
    def _filter_by_time_window(self, entries: List[LogEntry], window: str) -> List[LogEntry]:
        """按时间窗口过滤"""
        now = datetime.now()
        
        if window.endswith('h'):
            hours = int(window[:-1])
            cutoff = now - timedelta(hours=hours)
        elif window.endswith('d'):
            days = int(window[:-1])
            cutoff = now - timedelta(days=days)
        elif window.endswith('m'):
            minutes = int(window[:-1])
            cutoff = now - timedelta(minutes=minutes)
        else:
            return entries
        
        return [e for e in entries if e.timestamp >= cutoff]
    
    def _calculate_error_stats(self, entries: List[LogEntry]) -> ErrorStats:
        """计算错误统计"""
        error_levels = ['ERROR', 'FATAL', 'CRITICAL']
        warning_levels = ['WARNING', 'WARN']
        
        error_count = sum(1 for e in entries if e.level in error_levels)
        warning_count = sum(1 for e in entries if e.level in warning_levels)
        
        # 统计唯一错误类型
        error_messages = [e.message for e in entries if e.level in error_levels]
        error_types = Counter(error_messages)
        
        top_errors = [
            {"message": msg[:100], "count": count}
            for msg, count in error_types.most_common(5)
        ]
        
        error_rate = error_count / len(entries) if entries else 0
        
        return ErrorStats(
            total_entries=len(entries),
            error_count=error_count,
            warning_count=warning_count,
            unique_error_types=len(error_types),
            top_errors=top_errors,
            error_rate=error_rate,
            trend="stable"  # 简化处理
        )
    
    def _generate_recommendations(self, anomalies: List[Anomaly], stats: ErrorStats) -> List[str]:
        """生成改进建议"""
        recommendations = []
        
        if stats and stats.error_rate > 0.1:
            recommendations.append("错误率超过10%，建议立即检查系统稳定性")
        
        critical_count = len([a for a in anomalies if a.level == AnomalyLevel.CRITICAL])
        if critical_count > 0:
            recommendations.append(f"发现 {critical_count} 个严重异常，需要优先处理")
        
        if stats and stats.unique_error_types > 10:
            recommendations.append("错误类型较多，建议进行错误分类和根因分析")
        
        memory_anomalies = [a for a in anomalies if a.type == AnomalyType.MEMORY_ISSUE]
        if memory_anomalies:
            recommendations.append("检测到内存相关问题，建议检查内存泄漏")
        
        if not recommendations:
            recommendations.append("系统运行正常，继续保持监控")
        
        return recommendations
    
    def set_anomaly_callback(self, level: AnomalyLevel, callback: Callable):
        """设置异常回调"""
        self.anomaly_callbacks[level].append(callback)
    
    def start_monitoring(self, log_paths: List[str], check_interval: int = 60,
                        alert_callback: Optional[Callable] = None):
        """启动实时监控"""
        self._monitoring = True
        self._monitor_task = asyncio.create_task(
            self._monitor_loop(log_paths, check_interval, alert_callback)
        )
    
    def stop_monitoring(self):
        """停止实时监控"""
        self._monitoring = False
        if self._monitor_task:
            self._monitor_task.cancel()
    
    async def _monitor_loop(self, log_paths: List[str], interval: int, 
                           alert_callback: Optional[Callable]):
        """监控循环"""
        last_positions = {path: 0 for path in log_paths}
        
        while self._monitoring:
            try:
                for path in log_paths:
                    if not os.path.exists(path):
                        continue
                    
                    # 读取新内容
                    current_size = os.path.getsize(path)
                    if current_size <= last_positions[path]:
                        continue
                    
                    with open(path, 'r', encoding='utf-8', errors='ignore') as f:
                        f.seek(last_positions[path])
                        new_lines = f.readlines()
                        last_positions[path] = f.tell()
                    
                    # 解析新条目
                    entries = []
                    for line in new_lines:
                        entry = self.parser._parse_plain(line.strip(), path)
                        if entry:
                            entries.append(entry)
                    
                    # 检测异常
                    if entries:
                        anomalies = self.detector.detect_pattern_anomalies(entries)
                        
                        for anomaly in anomalies:
                            # 触发回调
                            for callback in self.anomaly_callbacks.get(anomaly.level, []):
                                callback(anomaly)
                            
                            if alert_callback:
                                alert_callback(anomaly)
                
                await asyncio.sleep(interval)
                
            except Exception as e:
                print(f"监控错误: {e}")
                await asyncio.sleep(interval)
    
    def save_report(self, report: AnalysisReport, filename: Optional[str] = None):
        """保存报告"""
        if filename is None:
            filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
        
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            f.write(report.to_html())
        
        return filepath
    
    def get_recent_anomalies(self, source: Optional[str] = None, 
                            time_range: str = "1h") -> List[Anomaly]:
        """获取最近异常"""
        # 简化实现，实际应从持久化存储读取
        return []


def demo():
    """演示"""
    print("🔍 Log Anomaly Detector v1.0")
    print("=" * 50)
    
    # 创建检测器
    detector = LogAnomalyDetector(data_dir="./demo_analysis")
    
    # 创建示例日志数据
    sample_logs = """
2026-02-27 14:30:00 INFO Application started successfully
2026-02-27 14:30:05 INFO Connected to database
2026-02-27 14:30:10 WARNING Slow query detected: 2.5s
2026-02-27 14:30:15 ERROR Connection timeout to external API
2026-02-27 14:30:20 ERROR Connection timeout to external API
2026-02-27 14:30:25 ERROR Connection timeout to external API
2026-02-27 14:30:30 CRITICAL Memory usage exceeded 90%
2026-02-27 14:30:35 ERROR Exception in worker thread: NullPointerException
2026-02-27 14:30:40 INFO Retrying connection...
2026-02-27 14:30:45 INFO Connection restored
"""
    
    # 写入临时文件
    temp_log = "/tmp/demo_log.txt"
    with open(temp_log, 'w') as f:
        f.write(sample_logs)
    
    # 分析
    report = detector.analyze_log_file(temp_log)
    
    print(f"\n📊 分析结果:")
    print(f"  总日志数: {report.total_entries}")
    print(f"  异常数量: {len(report.anomalies)}")
    
    if report.error_stats:
        print(f"  错误数: {report.error_stats.error_count}")
        print(f"  错误率: {report.error_stats.error_rate:.1%}")
    
    print(f"\n🔍 检测到的异常:")
    for anomaly in report.anomalies:
        print(f"  [{anomaly.level.value.upper()}] {anomaly.type.value}")
        print(f"    {anomaly.description[:60]}...")
        if anomaly.suggested_action:
            print(f"    💡 {anomaly.suggested_action}")
    
    # 保存报告
    report_path = detector.save_report(report, "demo_report.html")
    print(f"\n📄 报告已保存: {report_path}")
    
    # 清理
    os.remove(temp_log)
    
    return report


if __name__ == "__main__":
    demo()
