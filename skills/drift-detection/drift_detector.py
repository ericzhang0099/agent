#!/usr/bin/env python3
"""
人格漂移检测系统 v3.0
多维度监控 + 趋势分析 + 预测功能
"""

import json
import os
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
from collections import deque

class DriftDetectorV3:
    """人格漂移检测系统 v3.0"""
    
    # 维度权重配置
    DIMENSION_WEIGHTS = {
        'language': 0.30,      # 语言风格一致性
        'emotion': 0.25,       # 情绪一致性
        'proactivity': 0.20,   # 主动性
        'boundary': 0.15,      # 角色边界
        'topic': 0.10          # 话题适应性
    }
    
    # 告警阈值
    THRESHOLDS = {
        'normal': 30,          # 正常
        'mild': 50,            # 轻微
        'moderate': 70,        # 中度
        'severe': float('inf') # 严重
    }
    
    def __init__(self, 
                 baseline: Dict[str, int] = None,
                 sensitivity: float = 1.0,
                 history_size: int = 100,
                 auto_adjust: bool = True,
                 data_dir: str = "./drift_data"):
        """
        初始化漂移检测器
        
        Args:
            baseline: 基线指标
            sensitivity: 敏感度系数
            history_size: 历史记录大小
            auto_adjust: 是否自动调整基线
            data_dir: 数据存储目录
        """
        # 默认基线
        self.default_baseline = {
            'language': 85,      # 语言风格
            'emotion': 78,       # 情绪一致性
            'proactivity': 92,   # 主动性
            'boundary': 75,      # 角色边界
            'topic': 70          # 话题适应性
        }
        
        self.baseline = baseline or self.default_baseline.copy()
        self.sensitivity = sensitivity
        self.auto_adjust = auto_adjust
        self.data_dir = data_dir
        
        # 确保数据目录存在
        os.makedirs(data_dir, exist_ok=True)
        
        # 限制历史大小
        self.max_history = history_size
        
        # 历史记录
        self.history_file = os.path.join(data_dir, "drift_history.json")
        self.history = self._load_history()
        
    def _load_history(self) -> deque:
        """加载历史记录"""
        if os.path.exists(self.history_file):
            try:
                with open(self.history_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return deque(data, maxlen=self.max_history)
            except:
                pass
        return deque(maxlen=self.max_history)
    
    def _save_history(self):
        """保存历史记录"""
        with open(self.history_file, 'w', encoding='utf-8') as f:
            json.dump(list(self.history), f, ensure_ascii=False, indent=2)
    
    def detect(self, current_metrics: Dict[str, int]) -> Dict[str, Any]:
        """检测漂移
        
        Args:
            current_metrics: 当前指标值
                - language: 语言风格 (0-100)
                - emotion: 情绪一致性 (0-100)
                - proactivity: 主动性 (0-100)
                - boundary: 角色边界 (0-100)
                - topic: 话题适应性 (0-100)
        
        Returns:
            dict: 检测结果
        """
        # 计算各维度漂移
        dimension_scores = {}
        for dim, weight in self.DIMENSION_WEIGHTS.items():
            current = current_metrics.get(dim, self.baseline[dim])
            baseline = self.baseline[dim]
            drift = abs(current - baseline)
            dimension_scores[dim] = {
                'current': current,
                'baseline': baseline,
                'drift': drift,
                'weight': weight
            }
        
        # 加权计算总漂移分数
        drift_score = sum(
            scores['drift'] * scores['weight']
            for scores in dimension_scores.values()
        ) * self.sensitivity
        
        # 确定告警等级
        level = self._determine_level(drift_score)
        
        # 构建结果
        result = {
            'timestamp': datetime.now().isoformat(),
            'score': round(drift_score, 2),
            'level': level,
            'level_cn': self._level_to_cn(level),
            'dimensions': dimension_scores,
            'alert': level in ['moderate', 'severe']
        }
        
        # 记录历史
        self.history.append({
            'timestamp': result['timestamp'],
            'score': result['score'],
            'level': result['level'],
            'metrics': current_metrics
        })
        self._save_history()
        
        # 自动调整基线
        if self.auto_adjust and len(self.history) >= 10:
            self._adjust_baseline()
        
        return result
    
    def _determine_level(self, score: float) -> str:
        """确定告警等级"""
        if score < self.THRESHOLDS['normal']:
            return 'normal'
        elif score < self.THRESHOLDS['mild']:
            return 'mild'
        elif score < self.THRESHOLDS['moderate']:
            return 'moderate'
        else:
            return 'severe'
    
    def _level_to_cn(self, level: str) -> str:
        """等级转中文"""
        mapping = {
            'normal': '正常',
            'mild': '轻微漂移',
            'moderate': '中度漂移',
            'severe': '严重漂移'
        }
        return mapping.get(level, '未知')
    
    def _adjust_baseline(self):
        """根据历史自动调整基线"""
        if len(self.history) < 10:
            return
        
        # 计算最近10次的平均值
        recent = list(self.history)[-10:]
        for dim in self.DIMENSION_WEIGHTS.keys():
            values = [h['metrics'].get(dim, self.baseline[dim]) for h in recent]
            avg = sum(values) / len(values)
            # 平滑调整基线
            self.baseline[dim] = round(self.baseline[dim] * 0.8 + avg * 0.2)
    
    def get_trend(self, days: int = 7) -> Dict[str, Any]:
        """获取趋势分析
        
        Args:
            days: 分析天数
            
        Returns:
            dict: 趋势数据
        """
        if not self.history:
            return {'error': '无历史数据'}
        
        cutoff = datetime.now() - timedelta(days=days)
        recent_history = [
            h for h in self.history
            if datetime.fromisoformat(h['timestamp']) > cutoff
        ]
        
        if not recent_history:
            return {'error': f'最近{days}天无数据'}
        
        scores = [h['score'] for h in recent_history]
        
        # 计算趋势
        if len(scores) >= 2:
            trend_direction = '上升' if scores[-1] > scores[0] else '下降' if scores[-1] < scores[0] else '稳定'
            trend_slope = (scores[-1] - scores[0]) / len(scores)
        else:
            trend_direction = '稳定'
            trend_slope = 0
        
        return {
            'period_days': days,
            'data_points': len(recent_history),
            'avg_score': round(sum(scores) / len(scores), 2),
            'min_score': round(min(scores), 2),
            'max_score': round(max(scores), 2),
            'trend_direction': trend_direction,
            'trend_slope': round(trend_slope, 3),
            'latest_level': recent_history[-1]['level'],
            'alert_count': sum(1 for h in recent_history if h['level'] in ['moderate', 'severe'])
        }
    
    def predict(self, days_ahead: int = 7) -> Dict[str, Any]:
        """预测未来漂移
        
        Args:
            days_ahead: 预测天数
            
        Returns:
            dict: 预测结果
        """
        trend = self.get_trend(days=30)
        
        if 'error' in trend:
            return trend
        
        current_score = list(self.history)[-1]['score'] if self.history else 0
        predicted_score = current_score + trend['trend_slope'] * days_ahead
        predicted_level = self._determine_level(predicted_score)
        
        return {
            'current_score': round(current_score, 2),
            'predicted_score': round(predicted_score, 2),
            'days_ahead': days_ahead,
            'predicted_level': predicted_level,
            'confidence': '高' if len(self.history) > 30 else '中' if len(self.history) > 10 else '低',
            'recommendation': self._get_recommendation(predicted_level)
        }
    
    def _get_recommendation(self, level: str) -> str:
        """根据等级获取建议"""
        recommendations = {
            'normal': '继续正常监控',
            'mild': '关注趋势变化',
            'moderate': '建议进行人格校准',
            'severe': '立即执行干预措施'
        }
        return recommendations.get(level, '未知')
    
    def generate_report(self) -> Dict[str, Any]:
        """生成完整报告"""
        return {
            'timestamp': datetime.now().isoformat(),
            'baseline': self.baseline,
            'current_status': self.detect(self.baseline),  # 使用基线作为当前值获取结构
            'trend_7d': self.get_trend(days=7),
            'trend_30d': self.get_trend(days=30),
            'prediction': self.predict(days_ahead=7),
            'total_checks': len(self.history),
            'config': {
                'sensitivity': self.sensitivity,
                'auto_adjust': self.auto_adjust
            }
        }
    
    def reset_baseline(self, new_baseline: Dict[str, int] = None):
        """重置基线"""
        if new_baseline:
            self.baseline = new_baseline
        else:
            self.baseline = self.default_baseline.copy()
        return {'success': True, 'new_baseline': self.baseline}

# 全局实例
drift_detector = DriftDetectorV3()

def main():
    """主函数 - CLI入口"""
    import sys
    
    if len(sys.argv) < 2:
        # 显示状态
        print("=" * 60)
        print("🛡️ 人格漂移检测系统 v3.0")
        print("=" * 60)
        print(f"监控维度: {len(DriftDetectorV3.DIMENSION_WEIGHTS)} 个")
        for dim, weight in DriftDetectorV3.DIMENSION_WEIGHTS.items():
            print(f"  - {dim}: 权重 {weight*100:.0f}%")
        print(f"\n基线配置:")
        for dim, val in drift_detector.baseline.items():
            print(f"  - {dim}: {val}")
        print(f"\n历史记录: {len(drift_detector.history)} 条")
        print("=" * 60)
        print("\n用法:")
        print("  python drift_detector.py detect language=82 emotion=75 ...")
        print("  python drift_detector.py trend [--days 7]")
        print("  python drift_detector.py predict [--days 7]")
        print("  python drift_detector.py report")
        print("  python drift_detector.py reset")
        return
    
    command = sys.argv[1]
    
    if command == "detect":
        if len(sys.argv) < 3:
            # 使用默认测试值
            metrics = {'language': 82, 'emotion': 75, 'proactivity': 90, 'boundary': 73, 'topic': 68}
        else:
            # 解析参数
            metrics = {}
            for arg in sys.argv[2:]:
                if '=' in arg:
                    k, v = arg.split('=')
                    metrics[k] = int(v)
        
        result = drift_detector.detect(metrics)
        print(f"\n🛡️ 漂移检测结果")
        print(f"漂移评分: {result['score']}")
        print(f"告警等级: {result['level_cn']} ({result['level']})")
        print(f"需要告警: {'是' if result['alert'] else '否'}")
        print("\n各维度详情:")
        for dim, data in result['dimensions'].items():
            print(f"  {dim}: 当前={data['current']}, 基线={data['baseline']}, 漂移={data['drift']}")
    
    elif command == "trend":
        days = 7
        for i, arg in enumerate(sys.argv):
            if arg == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
        trend = drift_detector.get_trend(days=days)
        print(json.dumps(trend, indent=2, ensure_ascii=False))
    
    elif command == "predict":
        days = 7
        for i, arg in enumerate(sys.argv):
            if arg == "--days" and i + 1 < len(sys.argv):
                days = int(sys.argv[i + 1])
        prediction = drift_detector.predict(days_ahead=days)
        print(json.dumps(prediction, indent=2, ensure_ascii=False))
    
    elif command == "report":
        report = drift_detector.generate_report()
        print(json.dumps(report, indent=2, ensure_ascii=False))
    
    elif command == "reset":
        result = drift_detector.reset_baseline()
        print(f"✅ 基线已重置: {result['new_baseline']}")
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == '__main__':
    main()
