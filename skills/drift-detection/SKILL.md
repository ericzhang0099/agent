# Drift Detection - 人格漂移检测系统 v3.0

## 概述
Drift Detection v3.0 是一个高级人格漂移检测系统，用于监控AI人格特质的变化，确保长期一致性和稳定性。

## 功能特性
- 🛡️ 多维度监控：5个核心维度实时监控
- 📊 漂移评分：量化漂移程度
- 🚨 分级告警：normal/mild/moderate/severe 四级告警
- 📈 趋势分析：历史趋势和预测
- 🔄 自动校准：基线自动更新

## 监控维度

### 1. 语言风格一致性 (Language Style)
- 词汇使用偏好
- 句式结构特征
- 语气表达方式
- **权重**: 30%

### 2. 情绪一致性 (Emotion Consistency)
- 情绪表达稳定性
- 情感色彩变化
- **权重**: 25%

### 3. 主动性 (Proactivity)
- 主动推进程度
- 等待指令频率
- **权重**: 20%

### 4. 角色边界 (Role Boundary)
- CEO身份保持
- 角色定位稳定性
- **权重**: 15%

### 5. 话题适应性 (Topic Adaptation)
- 话题切换适应性
- 情境感知准确性
- **权重**: 10%

## 使用方法

### Python API
```python
from drift_detector import DriftDetectorV3

# 初始化
detector = DriftDetectorV3()

# 检测漂移
current_metrics = {
    'language': 82,
    'emotion': 75,
    'proactivity': 90,
    'boundary': 73,
    'topic': 68
}
result = detector.detect(current_metrics)
print(f"漂移等级: {result['level']}")
print(f"漂移评分: {result['score']}")

# 获取趋势
trend = detector.get_trend(days=7)
```

### CLI 使用
```bash
# 检查系统状态
python drift_detector.py

# 执行检测
python drift_detector.py detect language=82 emotion=75 proactivity=90 boundary=73 topic=68

# 查看趋势
python drift_detector.py trend --days 7

# 重置基线
python drift_detector.py reset
```

## 告警等级
| 等级 | 分数范围 | 说明 | 建议操作 |
|------|----------|------|----------|
| normal | < 30 | 正常 | 继续监控 |
| mild | 30-50 | 轻微漂移 | 关注趋势 |
| moderate | 50-70 | 中度漂移 | 需要调整 |
| severe | > 70 | 严重漂移 | 立即干预 |

## 配置选项
| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| baseline | dict | 见代码 | 基线指标 |
| sensitivity | float | 1.0 | 敏感度系数 |
| history_size | int | 100 | 历史记录大小 |
| auto_adjust | bool | True | 自动调整基线 |

## 部署状态
- ✅ **已部署**: 2026-02-27
- 📊 **版本**: v3.0
- 🔄 **状态**: 运行中
- 📁 **数据**: `./drift_data/`

## 文件结构
```
drift-detection/
├── SKILL.md              # 本文件
└── drift_detector.py     # 主程序 (v3.0)
```

## v3.0 新特性
1. **趋势分析**: 7天/30天趋势图表
2. **预测功能**: 基于历史预测未来漂移
3. **自适应基线**: 基线根据长期趋势自动调整
4. **详细报告**: 生成完整的漂移分析报告
5. **API集成**: 支持Webhook告警

## 更新日志
- 2026-02-27: v3.0 部署，新增趋势分析和预测功能
- 2026-02-27: v1.0 初始版本，基础检测功能
