#!/usr/bin/env python3
"""
人格漂移监控自动化系统
每小时自动检测Drift Score
"""

import json
import time
from datetime import datetime
from typing import Dict, List

class AutomatedDriftMonitor:
    """自动化人格漂移监控器"""
    
    def __init__(self):
        self.thresholds = {
            'warning': 0.15,
            'moderate': 0.30,
            'critical': 0.50
        }
        self.check_history = []
        self.alert_count = {'warning': 0, 'moderate': 0, 'critical': 0}
        
    def calculate_drift_score(self, current_metrics: Dict) -> float:
        """计算Drift Score"""
        baseline = {
            'personality': 0.85,
            'physical': 0.70,
            'motivations': 0.90,
            'backstory': 0.80,
            'emotions': 0.75,
            'relationships': 0.85,
            'growth': 0.70,
            'conflict': 0.80
        }
        
        total_drift = 0
        for dim, base_val in baseline.items():
            curr_val = current_metrics.get(dim, base_val)
            total_drift += abs(curr_val - base_val)
        
        return total_drift / len(baseline)
    
    def check_and_alert(self, current_metrics: Dict) -> Dict:
        """检查并生成告警"""
        drift_score = self.calculate_drift_score(current_metrics)
        
        # 确定等级
        if drift_score >= self.thresholds['critical']:
            level = 'CRITICAL'
            action = '立即重置人格系统'
        elif drift_score >= self.thresholds['moderate']:
            level = 'MODERATE'
            action = '主动修正并报告'
        elif drift_score >= self.thresholds['warning']:
            level = 'WARNING'
            action = '自动微调参数'
        else:
            level = 'NORMAL'
            action = '继续监控'
        
        # 记录历史
        check_result = {
            'timestamp': datetime.now().isoformat(),
            'drift_score': drift_score,
            'level': level,
            'action': action
        }
        self.check_history.append(check_result)
        
        if level != 'NORMAL':
            self.alert_count[level.lower()] += 1
        
        return check_result
    
    def generate_trend_report(self, hours: int = 24) -> Dict:
        """生成趋势报告"""
        recent_checks = [
            c for c in self.check_history
            if (datetime.now() - datetime.fromisoformat(c['timestamp'])).seconds < hours * 3600
        ]
        
        if not recent_checks:
            return {'status': 'no_data'}
        
        scores = [c['drift_score'] for c in recent_checks]
        
        return {
            'period_hours': hours,
            'check_count': len(recent_checks),
            'avg_drift': sum(scores) / len(scores),
            'max_drift': max(scores),
            'min_drift': min(scores),
            'alert_summary': self.alert_count,
            'trend': 'increasing' if scores[-1] > scores[0] else 'stable'
        }

# 全局实例
drift_monitor = AutomatedDriftMonitor()

if __name__ == '__main__':
    print("🛡️ 人格漂移监控自动化系统")
    print("=" * 50)
    print("告警阈值:")
    print(f"  WARNING:  {drift_monitor.thresholds['warning']}")
    print(f"  MODERATE: {drift_monitor.thresholds['moderate']}")
    print(f"  CRITICAL: {drift_monitor.thresholds['critical']}")
    
    # 演示检测
    test_metrics = {
        'personality': 0.82, 'physical': 0.68, 'motivations': 0.88,
        'backstory': 0.78, 'emotions': 0.72, 'relationships': 0.83,
        'growth': 0.68, 'conflict': 0.78
    }
    
    result = drift_monitor.check_and_alert(test_metrics)
    print(f"\n演示检测结果:")
    print(f"  Drift Score: {result['drift_score']:.3f}")
    print(f"  等级: {result['level']}")
    print(f"  建议行动: {result['action']}")
