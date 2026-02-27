# EmotionTaskMatrix - 情绪-任务矩阵核心

基于SOUL.md v4.0 8维度人格模型的情绪-任务匹配调度系统。

## 🎯 核心特性

- **16种核心情绪**: 基于心理学基本情绪理论，涵盖积极与消极情绪谱系
- **14种任务类型**: 覆盖战略、协调、执行三层架构
- **224个匹配组合**: 预计算的情绪-任务匹配矩阵
- **HEARTBEAT监控**: 实时情绪追踪与健康评分
- **智能调度**: 情绪自适应的任务调度算法

## 📁 文件结构

```
emotion_task_matrix.js      # 核心模块 (23KB)
emotion_task_matrix_test.js # 测试套件 (13KB)
emotion_matrix_demo.js      # 快速演示
```

## 🚀 快速开始

```javascript
const { EmotionTaskMatrix } = require('./emotion_task_matrix.js');

// 创建实例
const etm = new EmotionTaskMatrix();
etm.start();

// 设置情绪
etm.setEmotion('CURIOSITY', 0.8);

// 添加任务
etm.addTask({
  type: 'CREATIVE_IDEATION',
  priority: 5,
  data: { topic: 'AI创新' }
});

// 自动调度
const task = etm.autoSchedule();

// 完成任务
etm.completeTask(task.id, { result: 'success' });
```

## 🎭 16种情绪定义

| 情绪 | 名称 | Valence | Arousal | SOUL维度 |
|------|------|---------|---------|----------|
| JOY | 喜悦 | 1.0 | 0.8 | Emotions |
| GRATITUDE | 感恩 | 0.9 | 0.3 | Relationships |
| HOPE | 希望 | 0.8 | 0.6 | Growth |
| PRIDE | 自豪 | 0.7 | 0.7 | Motivations |
| CURIOSITY | 好奇 | 0.6 | 0.7 | Curiosity |
| CALM | 平静 | 0.5 | 0.2 | Physical |
| NEUTRAL | 中性 | 0.0 | 0.3 | Physical |
| SURPRISE | 惊讶 | 0.0 | 0.9 | Curiosity |
| CONFUSION | 困惑 | -0.3 | 0.5 | Curiosity |
| BOREDOM | 无聊 | -0.4 | 0.1 | Physical |
| ANXIETY | 焦虑 | -0.6 | 0.8 | Conflict |
| FRUSTRATION | 沮丧 | -0.7 | 0.6 | Conflict |
| SADNESS | 悲伤 | -0.8 | 0.3 | Emotions |
| ANGER | 愤怒 | -0.9 | 0.9 | Conflict |
| FEAR | 恐惧 | -0.9 | 0.95 | Backstory |
| DISGUST | 厌恶 | -0.8 | 0.5 | Personality |

## 📋 14种任务类型

### 战略层
- `STRATEGIC_DECISION` - 战略决策
- `CREATIVE_IDEATION` - 创意生成
- `ANALYTICAL_RESEARCH` - 分析研究

### 协调层
- `PROJECT_COORDINATION` - 项目协调
- `TASK_SCHEDULING` - 任务调度

### 执行层
- `CODE_DEVELOPMENT` - 代码开发
- `DEBUGGING` - 调试排错
- `TESTING_QA` - 测试验证
- `DOCUMENTATION` - 文档编写
- `DEPLOYMENT` - 部署运维
- `COMMUNICATION` - 沟通协作
- `LEARNING` - 学习成长
- `CRISIS_RESPONSE` - 危机响应
- `REFLECTION` - 反思总结

## 💓 HEARTBEAT监控

```javascript
const monitor = etm.heartbeat;

// 获取健康报告
const report = monitor.getReport();
console.log(report.health);        // 健康分数 (0-100)
console.log(report.stats);         // 统计信息
console.log(report.trend);         // 情绪趋势
```

## 🧪 运行测试

```bash
node emotion_task_matrix_test.js
```

20个测试用例覆盖：
- 基础初始化
- 16种情绪定义验证
- 14种任务类型验证
- 情绪管理
- 匹配算法
- 任务队列管理
- 智能推荐
- 自适应调度
- HEARTBEAT监控
- 事件系统
- 边界条件

## 🎬 运行演示

```bash
node emotion_matrix_demo.js
```

演示内容包括：
1. 16种情绪按SOUL维度分组展示
2. 不同情绪下的任务推荐
3. 智能任务调度流程
4. HEARTBEAT监控报告
5. 任务-情绪双向推荐

## 🔧 API参考

### EmotionTaskMatrix

| 方法 | 描述 |
|------|------|
| `setEmotion(key, intensity)` | 设置当前情绪 |
| `getEmotion()` | 获取当前情绪 |
| `addTask(config)` | 添加任务到队列 |
| `getNextTask()` | 获取下一个最佳任务 |
| `claimTask(id)` | 认领任务 |
| `completeTask(id, result)` | 完成任务 |
| `autoSchedule()` | 自动调度 |
| `adaptiveSchedule()` | 情绪自适应调度 |
| `recommendTasks(limit)` | 推荐任务 |
| `recommendEmotions(taskType, limit)` | 推荐情绪 |
| `calculateMatch(taskType, emotion)` | 计算匹配度 |
| `getStatus()` | 获取系统状态 |

### HeartbeatMonitor

| 方法 | 描述 |
|------|------|
| `start()` | 启动监控 |
| `stop()` | 停止监控 |
| `recordEmotion(key, intensity, context)` | 记录情绪 |
| `recordTaskMatch(taskType, emotion, score, details)` | 记录任务匹配 |
| `getEmotionTrend(windowSize)` | 获取情绪趋势 |
| `getReport()` | 获取监控报告 |
| `getAlerts()` | 获取告警列表 |

## 📊 匹配算法

匹配分数基于以下因素：

1. **预计算矩阵分数**: 基于valence/arousal与任务复杂度的匹配
2. **SOUL维度匹配**: 情绪维度与任务所需维度的对齐
3. **推荐列表**: 任务preferredEmotions列表中的情绪获得加分
4. **避免列表**: 任务avoidEmotions列表中的情绪被降权

分数范围: 0.0 - 1.0

## 🎨 情绪-任务匹配示例

| 情绪 | 最佳任务 | 避免任务 |
|------|----------|----------|
| 好奇 | 创意生成、分析研究 | 文档编写 |
| 平静 | 文档编写、测试验证 | 危机响应 |
| 喜悦 | 代码开发、创意生成 | 调试排错 |
| 焦虑 | 分析研究(有限) | 沟通协作、危机响应 |

## 🔗 关联文档

- [SOUL.md](./SOUL.md) - 8维度人格模型
- [AGENTS.md](./AGENTS.md) - Multi-Agent架构
- [HEARTBEAT.md](./HEARTBEAT.md) - 心跳监控系统

## 📄 许可证

MIT
