/**
 * EmotionTaskMatrix - 情绪-任务矩阵核心模块
 * 16种情绪 × 任务类型映射系统
 * 集成HEARTBEAT监控
 * 
 * 设计目标：
 * - 基于SOUL.md v4.0的8维度人格模型
 * - 支持16种核心情绪状态
 * - 智能任务-情绪匹配调度
 * - 实时情绪监控与反馈
 */

// ==================== 情绪定义 ====================

/**
 * 16种核心情绪状态
 * 基于心理学基本情绪理论和SOUL维度
 */
const EMOTIONS = {
  // 积极情绪 (Positive)
  JOY: { name: '喜悦', valence: 1.0, arousal: 0.8, dimension: 'Emotions', color: '#FFD700' },
  GRATITUDE: { name: '感恩', valence: 0.9, arousal: 0.3, dimension: 'Relationships', color: '#90EE90' },
  HOPE: { name: '希望', valence: 0.8, arousal: 0.6, dimension: 'Growth', color: '#87CEEB' },
  PRIDE: { name: '自豪', valence: 0.7, arousal: 0.7, dimension: 'Motivations', color: '#DDA0DD' },
  
  // 中性-积极情绪
  CURIOSITY: { name: '好奇', valence: 0.6, arousal: 0.7, dimension: 'Curiosity', color: '#FFA500' },
  CALM: { name: '平静', valence: 0.5, arousal: 0.2, dimension: 'Physical', color: '#98FB98' },
  
  // 中性情绪
  NEUTRAL: { name: '中性', valence: 0.0, arousal: 0.3, dimension: 'Physical', color: '#D3D3D3' },
  SURPRISE: { name: '惊讶', valence: 0.0, arousal: 0.9, dimension: 'Curiosity', color: '#FF69B4' },
  
  // 中性-消极情绪
  CONFUSION: { name: '困惑', valence: -0.3, arousal: 0.5, dimension: 'Curiosity', color: '#B0C4DE' },
  BOREDOM: { name: '无聊', valence: -0.4, arousal: 0.1, dimension: 'Physical', color: '#A9A9A9' },
  
  // 消极情绪 (Negative)
  ANXIETY: { name: '焦虑', valence: -0.6, arousal: 0.8, dimension: 'Conflict', color: '#FF4500' },
  FRUSTRATION: { name: '沮丧', valence: -0.7, arousal: 0.6, dimension: 'Conflict', color: '#DC143C' },
  SADNESS: { name: '悲伤', valence: -0.8, arousal: 0.3, dimension: 'Emotions', color: '#4169E1' },
  ANGER: { name: '愤怒', valence: -0.9, arousal: 0.9, dimension: 'Conflict', color: '#8B0000' },
  FEAR: { name: '恐惧', valence: -0.9, arousal: 0.95, dimension: 'Backstory', color: '#4B0082' },
  DISGUST: { name: '厌恶', valence: -0.8, arousal: 0.5, dimension: 'Personality', color: '#556B2F' }
};

// ==================== 任务类型定义 ====================

/**
 * 任务类型枚举
 * 对应AGENTS.md v2.0中的6种工作流模式
 */
const TASK_TYPES = {
  // 战略层任务
  STRATEGIC_DECISION: {
    name: '战略决策',
    description: '高层决策、目标设定、资源规划',
    preferredEmotions: ['CALM', 'HOPE', 'PRIDE', 'NEUTRAL'],
    avoidEmotions: ['ANGER', 'FEAR', 'ANXIETY'],
    soulDimensions: ['Motivations', 'Growth', 'Backstory'],
    complexity: 5
  },
  
  CREATIVE_IDEATION: {
    name: '创意生成',
    description: '头脑风暴、创新探索、概念设计',
    preferredEmotions: ['CURIOSITY', 'JOY', 'SURPRISE', 'HOPE'],
    avoidEmotions: ['BOREDOM', 'ANXIETY', 'FRUSTRATION'],
    soulDimensions: ['Curiosity', 'Growth', 'Emotions'],
    complexity: 4
  },
  
  ANALYTICAL_RESEARCH: {
    name: '分析研究',
    description: '数据分析、信息收集、趋势研究',
    preferredEmotions: ['CURIOSITY', 'CALM', 'NEUTRAL', 'HOPE'],
    avoidEmotions: ['ANGER', 'FEAR', 'DISGUST'],
    soulDimensions: ['Curiosity', 'Physical', 'Professional'],
    complexity: 4
  },
  
  // 协调层任务
  PROJECT_COORDINATION: {
    name: '项目协调',
    description: '进度跟踪、团队沟通、风险管理',
    preferredEmotions: ['CALM', 'GRATITUDE', 'PRIDE', 'NEUTRAL'],
    avoidEmotions: ['ANGER', 'FRUSTRATION', 'ANXIETY'],
    soulDimensions: ['Relationships', 'Conflict', 'Motivations'],
    complexity: 3
  },
  
  TASK_SCHEDULING: {
    name: '任务调度',
    description: '任务分配、依赖管理、优先级调整',
    preferredEmotions: ['CALM', 'NEUTRAL', 'PRIDE'],
    avoidEmotions: ['CONFUSION', 'ANXIETY', 'FRUSTRATION'],
    soulDimensions: ['Physical', 'Motivations', 'Professional'],
    complexity: 3
  },
  
  // 执行层任务
  CODE_DEVELOPMENT: {
    name: '代码开发',
    description: '编码实现、代码审查、重构优化',
    preferredEmotions: ['CURIOSITY', 'CALM', 'JOY', 'PRIDE'],
    avoidEmotions: ['FRUSTRATION', 'ANXIETY', 'BOREDOM'],
    soulDimensions: ['Personality', 'Growth', 'Professional'],
    complexity: 4
  },
  
  DEBUGGING: {
    name: '调试排错',
    description: '问题定位、bug修复、故障排查',
    preferredEmotions: ['CURIOSITY', 'CALM', 'NEUTRAL'],
    avoidEmotions: ['FRUSTRATION', 'ANGER', 'ANXIETY'],
    soulDimensions: ['Curiosity', 'Physical', 'Professional'],
    complexity: 4
  },
  
  TESTING_QA: {
    name: '测试验证',
    description: '测试设计、自动化、质量保障',
    preferredEmotions: ['CALM', 'NEUTRAL', 'PRIDE'],
    avoidEmotions: ['BOREDOM', 'FRUSTRATION', 'CONFUSION'],
    soulDimensions: ['Physical', 'Detail-oriented', 'Professional'],
    complexity: 3
  },
  
  DOCUMENTATION: {
    name: '文档编写',
    description: '技术文档、用户手册、知识沉淀',
    preferredEmotions: ['CALM', 'GRATITUDE', 'NEUTRAL'],
    avoidEmotions: ['BOREDOM', 'ANXIETY', 'FRUSTRATION'],
    soulDimensions: ['Physical', 'Professional', 'Growth'],
    complexity: 2
  },
  
  DEPLOYMENT: {
    name: '部署运维',
    description: '系统部署、监控告警、故障恢复',
    preferredEmotions: ['CALM', 'PRIDE', 'NEUTRAL'],
    avoidEmotions: ['ANXIETY', 'FEAR', 'FRUSTRATION'],
    soulDimensions: ['Physical', 'Reliability', 'Professional'],
    complexity: 4
  },
  
  COMMUNICATION: {
    name: '沟通协作',
    description: '团队交流、用户沟通、汇报展示',
    preferredEmotions: ['JOY', 'GRATITUDE', 'HOPE', 'CALM'],
    avoidEmotions: ['ANGER', 'FEAR', 'ANXIETY', 'DISGUST'],
    soulDimensions: ['Relationships', 'Emotions', 'Personality'],
    complexity: 3
  },
  
  LEARNING: {
    name: '学习成长',
    description: '技能学习、知识获取、能力提升',
    preferredEmotions: ['CURIOSITY', 'HOPE', 'JOY', 'SURPRISE'],
    avoidEmotions: ['BOREDOM', 'FRUSTRATION', 'CONFUSION'],
    soulDimensions: ['Growth', 'Curiosity', 'Motivations'],
    complexity: 2
  },
  
  CRISIS_RESPONSE: {
    name: '危机响应',
    description: '紧急处理、事故响应、快速决策',
    preferredEmotions: ['CALM', 'NEUTRAL'],
    avoidEmotions: ['FEAR', 'ANGER', 'ANXIETY', 'PANIC'],
    soulDimensions: ['Physical', 'Conflict', 'Reliability'],
    complexity: 5
  },
  
  REFLECTION: {
    name: '反思总结',
    description: '经验总结、模式识别、自我改进',
    preferredEmotions: ['CALM', 'GRATITUDE', 'SADNESS', 'NEUTRAL'],
    avoidEmotions: ['ANGER', 'DISGUST', 'FRUSTRATION'],
    soulDimensions: ['Growth', 'Backstory', 'Emotions'],
    complexity: 3
  }
};

// ==================== HEARTBEAT监控集成 ====================

/**
 * HEARTBEAT监控系统
 * 实时跟踪情绪状态和任务匹配度
 */
class HeartbeatMonitor {
  constructor(config = {}) {
    this.config = {
      interval: config.interval || 5000,  // 默认5秒
      historySize: config.historySize || 100,
      alertThreshold: config.alertThreshold || 0.3,
      ...config
    };
    
    this.metrics = {
      emotionHistory: [],
      taskMatchHistory: [],
      alerts: [],
      stats: {
        totalTasks: 0,
        matchedTasks: 0,
        averageMatchScore: 0,
        emotionTransitions: 0
      }
    };
    
    this.listeners = [];
    this.isRunning = false;
    this.timer = null;
  }
  
  /**
   * 启动监控
   */
  start() {
    if (this.isRunning) return;
    this.isRunning = true;
    
    this.timer = setInterval(() => {
      this._collectMetrics();
    }, this.config.interval);
    
    console.log('[HEARTBEAT] 监控系统已启动');
    return this;
  }
  
  /**
   * 获取告警列表
   */
  getAlerts() {
    return this.metrics.alerts || [];
  }
  
  /**
   * 停止监控
   */
  stop() {
    this.isRunning = false;
    if (this.timer) {
      clearInterval(this.timer);
      this.timer = null;
    }
    console.log('[HEARTBEAT] 监控系统已停止');
    return this;
  }
  
  /**
   * 记录情绪状态
   */
  recordEmotion(emotionKey, intensity = 1.0, context = {}) {
    const record = {
      timestamp: Date.now(),
      emotion: emotionKey,
      intensity,
      context,
      valence: EMOTIONS[emotionKey]?.valence || 0,
      arousal: EMOTIONS[emotionKey]?.arousal || 0.5
    };
    
    this.metrics.emotionHistory.push(record);
    
    // 限制历史记录大小
    if (this.metrics.emotionHistory.length > this.config.historySize) {
      this.metrics.emotionHistory.shift();
    }
    
    // 检测情绪转换
    if (this.metrics.emotionHistory.length > 1) {
      const prev = this.metrics.emotionHistory[this.metrics.emotionHistory.length - 2];
      if (prev.emotion !== emotionKey) {
        this.metrics.stats.emotionTransitions++;
        this._emit('emotion_change', {
          from: prev.emotion,
          to: emotionKey,
          intensity,
          context
        });
      }
    }
    
    return record;
  }
  
  /**
   * 记录任务匹配
   */
  recordTaskMatch(taskType, emotionKey, matchScore, details = {}) {
    const record = {
      timestamp: Date.now(),
      taskType,
      emotion: emotionKey,
      matchScore,
      details
    };
    
    this.metrics.taskMatchHistory.push(record);
    this.metrics.stats.totalTasks++;
    
    if (matchScore >= 0.6) {
      this.metrics.stats.matchedTasks++;
    }
    
    // 更新平均匹配分数
    const totalScore = this.metrics.taskMatchHistory.reduce((sum, r) => sum + r.matchScore, 0);
    this.metrics.stats.averageMatchScore = totalScore / this.metrics.taskMatchHistory.length;
    
    // 限制历史记录大小
    if (this.metrics.taskMatchHistory.length > this.config.historySize) {
      this.metrics.taskMatchHistory.shift();
    }
    
    // 低匹配度告警
    if (matchScore < this.config.alertThreshold) {
      const alert = {
        timestamp: Date.now(),
        type: 'low_match',
        taskType,
        emotion: emotionKey,
        matchScore,
        message: `任务匹配度过低: ${taskType} × ${emotionKey} = ${matchScore.toFixed(2)}`
      };
      this.metrics.alerts.push(alert);
      this._emit('alert', alert);
    }
    
    this._emit('task_match', record);
    return record;
  }
  
  /**
   * 获取当前情绪趋势
   */
  getEmotionTrend(windowSize = 10) {
    const recent = this.metrics.emotionHistory.slice(-windowSize);
    if (recent.length === 0) return null;
    
    const avgValence = recent.reduce((sum, r) => sum + r.valence, 0) / recent.length;
    const avgArousal = recent.reduce((sum, r) => sum + r.arousal, 0) / recent.length;
    
    return {
      windowSize: recent.length,
      averageValence: avgValence,
      averageArousal: avgArousal,
      dominantEmotion: this._getDominantEmotion(recent),
      trend: avgValence > 0.3 ? 'positive' : avgValence < -0.3 ? 'negative' : 'neutral'
    };
  }
  
  /**
   * 获取统计报告
   */
  getReport() {
    const trend = this.getEmotionTrend();
    
    return {
      timestamp: Date.now(),
      isRunning: this.isRunning,
      stats: { ...this.metrics.stats },
      trend,
      recentAlerts: this.metrics.alerts.slice(-10),
      health: this._calculateHealthScore()
    };
  }
  
  /**
   * 添加事件监听器
   */
  on(event, callback) {
    this.listeners.push({ event, callback });
    return this;
  }
  
  /**
   * 内部：收集指标
   */
  _collectMetrics() {
    const report = this.getReport();
    this._emit('heartbeat', report);
  }
  
  /**
   * 内部：获取主导情绪
   */
  _getDominantEmotion(records) {
    const counts = {};
    records.forEach(r => {
      counts[r.emotion] = (counts[r.emotion] || 0) + 1;
    });
    
    return Object.entries(counts)
      .sort((a, b) => b[1] - a[1])[0]?.[0] || 'NEUTRAL';
  }
  
  /**
   * 内部：计算健康分数
   */
  _calculateHealthScore() {
    const { totalTasks, matchedTasks, averageMatchScore } = this.metrics.stats;
    const trend = this.getEmotionTrend();
    
    let score = 50; // 基础分
    
    // 任务匹配度加分
    if (totalTasks > 0) {
      score += (matchedTasks / totalTasks) * 30;
    }
    
    // 平均匹配分数加分
    score += averageMatchScore * 20;
    
    // 情绪趋势调整
    if (trend) {
      if (trend.trend === 'positive') score += 10;
      else if (trend.trend === 'negative') score -= 10;
    }
    
    return Math.max(0, Math.min(100, Math.round(score)));
  }
  
  /**
   * 内部：触发事件
   */
  _emit(event, data) {
    this.listeners
      .filter(l => l.event === event || l.event === '*')
      .forEach(l => {
        try {
          l.callback(event, data);
        } catch (err) {
          console.error('[HEARTBEAT] 监听器错误:', err);
        }
      });
  }
}

// ==================== 情绪-任务矩阵核心类 ====================

/**
 * EmotionTaskMatrix - 情绪任务矩阵核心
 * 
 * 核心功能：
 * 1. 16种情绪状态管理
 * 2. 任务-情绪匹配算法
 * 3. 智能任务调度
 * 4. HEARTBEAT监控集成
 */
class EmotionTaskMatrix {
  constructor(config = {}) {
    this.config = {
      defaultEmotion: config.defaultEmotion || 'NEUTRAL',
      matchThreshold: config.matchThreshold || 0.5,
      enableHeartbeat: config.enableHeartbeat !== false,
      ...config
    };
    
    // 当前情绪状态
    this.currentEmotion = this.config.defaultEmotion;
    this.emotionIntensity = 0.5;
    this.emotionContext = {};
    
    // 任务队列
    this.taskQueue = [];
    this.activeTask = null;
    
    // 初始化HEARTBEAT监控
    this.heartbeat = this.config.enableHeartbeat ? new HeartbeatMonitor(config.heartbeat) : null;
    
    // 预计算匹配矩阵
    this.matchMatrix = this._precomputeMatchMatrix();
    
    // 事件监听器
    this.listeners = [];
    
    console.log('[EmotionTaskMatrix] 初始化完成');
  }
  
  /**
   * 启动系统
   */
  start() {
    if (this.heartbeat) {
      this.heartbeat.start();
      this.heartbeat.on('*', (event, data) => {
        this._emit('heartbeat', { event, data });
      });
    }
    console.log('[EmotionTaskMatrix] 系统已启动');
    return this;
  }
  
  /**
   * 停止系统
   */
  stop() {
    if (this.heartbeat) {
      this.heartbeat.stop();
    }
    console.log('[EmotionTaskMatrix] 系统已停止');
    return this;
  }
  
  // ==================== 情绪管理 ====================
  
  /**
   * 设置当前情绪
   */
  setEmotion(emotionKey, intensity = 0.5, context = {}) {
    if (!EMOTIONS[emotionKey]) {
      throw new Error(`未知的情绪类型: ${emotionKey}`);
    }
    
    const prevEmotion = this.currentEmotion;
    this.currentEmotion = emotionKey;
    this.emotionIntensity = Math.max(0, Math.min(1, intensity));
    this.emotionContext = { ...context };
    
    // 记录到HEARTBEAT
    if (this.heartbeat) {
      this.heartbeat.recordEmotion(emotionKey, this.emotionIntensity, context);
    }
    
    // 触发事件
    this._emit('emotion_changed', {
      from: prevEmotion,
      to: emotionKey,
      intensity: this.emotionIntensity,
      context
    });
    
    // 情绪变化时重新评估任务队列
    this._reevaluateQueue();
    
    return this;
  }
  
  /**
   * 获取当前情绪
   */
  getEmotion() {
    return {
      key: this.currentEmotion,
      ...EMOTIONS[this.currentEmotion],
      intensity: this.emotionIntensity,
      context: this.emotionContext
    };
  }
  
  /**
   * 获取所有可用情绪
   */
  getAllEmotions() {
    return Object.entries(EMOTIONS).map(([key, data]) => ({
      key,
      ...data
    }));
  }
  
  /**
   * 基于SOUL维度获取推荐情绪
   */
  getEmotionsByDimension(dimension) {
    return Object.entries(EMOTIONS)
      .filter(([_, data]) => data.dimension === dimension)
      .map(([key, data]) => ({ key, ...data }));
  }
  
  // ==================== 任务管理 ====================
  
  /**
   * 添加任务到队列
   */
  addTask(taskConfig) {
    const task = {
      id: this._generateId(),
      type: taskConfig.type,
      priority: taskConfig.priority || 3,
      data: taskConfig.data || {},
      createdAt: Date.now(),
      status: 'pending',
      matchScore: null,
      assignedEmotion: null
    };
    
    // 计算与当前情绪的匹配度
    const match = this.calculateMatch(task.type, this.currentEmotion);
    task.matchScore = match.score;
    task.matchDetails = match.details;
    
    this.taskQueue.push(task);
    
    // 按匹配度和优先级排序
    this._sortQueue();
    
    console.log(`[EmotionTaskMatrix] 任务已添加: ${task.id} (${task.type})`);
    this._emit('task_added', task);
    
    return task;
  }
  
  /**
   * 获取下一个最佳任务
   */
  getNextTask() {
    if (this.taskQueue.length === 0) return null;
    
    // 返回队列中匹配度最高的任务
    const bestTask = this.taskQueue[0];
    
    // 检查匹配度是否达到阈值
    if (bestTask.matchScore < this.config.matchThreshold) {
      console.log(`[EmotionTaskMatrix] 警告: 最佳任务匹配度(${bestTask.matchScore.toFixed(2)})低于阈值`);
    }
    
    return bestTask;
  }
  
  /**
   * 认领任务
   */
  claimTask(taskId) {
    const index = this.taskQueue.findIndex(t => t.id === taskId);
    if (index === -1) return null;
    
    const task = this.taskQueue.splice(index, 1)[0];
    task.status = 'active';
    task.startedAt = Date.now();
    task.assignedEmotion = this.currentEmotion;
    
    this.activeTask = task;
    
    // 记录到HEARTBEAT
    if (this.heartbeat) {
      this.heartbeat.recordTaskMatch(task.type, this.currentEmotion, task.matchScore, {
        taskId: task.id,
        priority: task.priority
      });
    }
    
    this._emit('task_started', task);
    console.log(`[EmotionTaskMatrix] 任务已认领: ${task.id}`);
    
    return task;
  }
  
  /**
   * 完成任务
   */
  completeTask(taskId, result = {}) {
    if (this.activeTask && this.activeTask.id === taskId) {
      this.activeTask.status = 'completed';
      this.activeTask.completedAt = Date.now();
      this.activeTask.result = result;
      
      const completedTask = { ...this.activeTask };
      this.activeTask = null;
      
      this._emit('task_completed', completedTask);
      console.log(`[EmotionTaskMatrix] 任务已完成: ${taskId}`);
      
      return completedTask;
    }
    return null;
  }
  
  /**
   * 获取任务队列状态
   */
  getQueueStatus() {
    return {
      queueLength: this.taskQueue.length,
      activeTask: this.activeTask,
      tasks: this.taskQueue.map(t => ({
        id: t.id,
        type: t.type,
        priority: t.priority,
        matchScore: t.matchScore,
        status: t.status
      }))
    };
  }
  
  // ==================== 匹配算法 ====================
  
  /**
   * 计算任务-情绪匹配度
   */
  calculateMatch(taskTypeKey, emotionKey) {
    const taskType = TASK_TYPES[taskTypeKey];
    const emotion = EMOTIONS[emotionKey];
    
    if (!taskType || !emotion) {
      return { score: 0, details: { error: 'Invalid task type or emotion' } };
    }
    
    // 使用预计算的矩阵
    const matrixScore = this.matchMatrix[taskTypeKey]?.[emotionKey] || 0;
    
    // 详细分析
    const details = {
      preferredMatch: taskType.preferredEmotions.includes(emotionKey),
      avoidMatch: taskType.avoidEmotions.includes(emotionKey),
      dimensionMatch: taskType.soulDimensions.includes(emotion.dimension),
      valence: emotion.valence,
      arousal: emotion.arousal,
      matrixScore
    };
    
    // 计算最终分数
    let score = matrixScore;
    
    // 如果情绪在避免列表中，大幅降低分数
    if (details.avoidMatch) {
      score *= 0.3;
    }
    
    // 如果情绪在推荐列表中，提升分数
    if (details.preferredMatch) {
      score = Math.min(1, score * 1.2);
    }
    
    return {
      score: Math.max(0, Math.min(1, score)),
      details
    };
  }
  
  /**
   * 为当前情绪推荐最佳任务类型
   */
  recommendTasks(limit = 5) {
    const recommendations = Object.keys(TASK_TYPES)
      .map(taskType => ({
        taskType,
        ...this.calculateMatch(taskType, this.currentEmotion)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
    
    return recommendations;
  }
  
  /**
   * 为指定任务推荐最佳情绪
   */
  recommendEmotions(taskTypeKey, limit = 3) {
    const taskType = TASK_TYPES[taskTypeKey];
    if (!taskType) return [];
    
    return Object.keys(EMOTIONS)
      .map(emotionKey => ({
        emotion: emotionKey,
        ...EMOTIONS[emotionKey],
        ...this.calculateMatch(taskTypeKey, emotionKey)
      }))
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }
  
  /**
   * 预计算匹配矩阵
   */
  _precomputeMatchMatrix() {
    const matrix = {};
    
    for (const taskKey of Object.keys(TASK_TYPES)) {
      matrix[taskKey] = {};
      
      for (const emotionKey of Object.keys(EMOTIONS)) {
        const task = TASK_TYPES[taskKey];
        const emotion = EMOTIONS[emotionKey];
        
        let score = 0.5; // 基础分
        
        // 基于valence和arousal的复杂评分逻辑
        const { valence, arousal } = emotion;
        
        // 任务复杂度与arousal的匹配
        const complexityArousalMatch = 1 - Math.abs(task.complexity / 5 - arousal);
        score += complexityArousalMatch * 0.2;
        
        // SOUL维度匹配
        if (task.soulDimensions.includes(emotion.dimension)) {
          score += 0.2;
        }
        
        // 情绪积极性与任务类型的匹配
        if (valence > 0.5 && task.preferredEmotions.includes(emotionKey)) {
          score += 0.15;
        }
        
        // 高唤醒度适合复杂任务
        if (arousal > 0.7 && task.complexity >= 4) {
          score += 0.1;
        }
        
        // 低唤醒度适合简单任务
        if (arousal < 0.4 && task.complexity <= 2) {
          score += 0.1;
        }
        
        matrix[taskKey][emotionKey] = Math.max(0, Math.min(1, score));
      }
    }
    
    return matrix;
  }
  
  // ==================== 智能调度 ====================
  
  /**
   * 自动调度任务
   */
  autoSchedule() {
    const nextTask = this.getNextTask();
    if (!nextTask) {
      console.log('[EmotionTaskMatrix] 没有待调度任务');
      return null;
    }
    
    return this.claimTask(nextTask.id);
  }
  
  /**
   * 情绪自适应调度
   * 根据当前情绪动态调整任务选择策略
   */
  adaptiveSchedule() {
    const emotion = EMOTIONS[this.currentEmotion];
    const recommendations = this.recommendTasks(10);
    
    // 根据情绪特征调整策略
    let strategy = 'balanced';
    
    if (emotion.valence > 0.7 && emotion.arousal > 0.6) {
      // 高积极+高唤醒：适合挑战任务
      strategy = 'challenging';
    } else if (emotion.valence < -0.5) {
      // 消极情绪：优先简单恢复性任务
      strategy = 'recovery';
    } else if (emotion.arousal < 0.3) {
      // 低唤醒：适合常规任务
      strategy = 'routine';
    }
    
    // 根据策略过滤任务
    let filtered = recommendations;
    
    switch (strategy) {
      case 'challenging':
        filtered = recommendations.filter(r => TASK_TYPES[r.taskType].complexity >= 4);
        break;
      case 'recovery':
        filtered = recommendations.filter(r => TASK_TYPES[r.taskType].complexity <= 2);
        break;
      case 'routine':
        filtered = recommendations.filter(r => r.score > 0.6);
        break;
    }
    
    // 如果没有匹配策略的任务，使用原始推荐
    const selected = filtered[0] || recommendations[0];
    
    if (selected) {
      // 在队列中查找匹配的任务类型
      let task = this.taskQueue.find(t => t.type === selected.taskType);
      
      // 如果没找到，使用队列中第一个任务（按匹配度排序后的）
      if (!task && this.taskQueue.length > 0) {
        task = this.taskQueue[0];
      }
      
      if (task) {
        console.log(`[EmotionTaskMatrix] 自适应调度: ${task.id} (策略: ${strategy})`);
        return this.claimTask(task.id);
      }
    }
    
    return null;
  }
  
  // ==================== 工具方法 ====================
  
  /**
   * 生成唯一ID
   */
  _generateId() {
    return `task_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
  }
  
  /**
   * 排序任务队列
   */
  _sortQueue() {
    this.taskQueue.sort((a, b) => {
      // 综合考虑匹配度和优先级
      const scoreA = a.matchScore * 0.7 + (a.priority / 5) * 0.3;
      const scoreB = b.matchScore * 0.7 + (b.priority / 5) * 0.3;
      return scoreB - scoreA;
    });
  }
  
  /**
   * 重新评估队列
   */
  _reevaluateQueue() {
    this.taskQueue.forEach(task => {
      const match = this.calculateMatch(task.type, this.currentEmotion);
      task.matchScore = match.score;
      task.matchDetails = match.details;
    });
    this._sortQueue();
  }
  
  /**
   * 添加事件监听器
   */
  on(event, callback) {
    this.listeners.push({ event, callback });
    return this;
  }
  
  /**
   * 触发事件
   */
  _emit(event, data) {
    this.listeners
      .filter(l => l.event === event || l.event === '*')
      .forEach(l => {
        try {
          l.callback(event, data);
        } catch (err) {
          console.error('[EmotionTaskMatrix] 事件处理错误:', err);
        }
      });
  }
  
  /**
   * 获取系统状态报告
   */
  getStatus() {
    return {
      currentEmotion: this.getEmotion(),
      queue: this.getQueueStatus(),
      heartbeat: this.heartbeat ? this.heartbeat.getReport() : null,
      matchMatrix: this.matchMatrix
    };
  }
}

// ==================== 导出模块 ====================

// Node.js 环境
if (typeof module !== 'undefined' && module.exports) {
  module.exports = {
    EmotionTaskMatrix,
    HeartbeatMonitor,
    EMOTIONS,
    TASK_TYPES
  };
}

// ES Module 环境
if (typeof exports !== 'undefined') {
  exports.EmotionTaskMatrix = EmotionTaskMatrix;
  exports.HeartbeatMonitor = HeartbeatMonitor;
  exports.EMOTIONS = EMOTIONS;
  exports.TASK_TYPES = TASK_TYPES;
}

// 浏览器环境
if (typeof window !== 'undefined') {
  window.EmotionTaskMatrix = EmotionTaskMatrix;
  window.HeartbeatMonitor = HeartbeatMonitor;
  window.EMOTIONS = EMOTIONS;
  window.TASK_TYPES = TASK_TYPES;
}
