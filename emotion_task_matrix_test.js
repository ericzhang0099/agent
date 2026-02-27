/**
 * EmotionTaskMatrix 测试套件
 * 验证16情绪×任务类型映射、HEARTBEAT监控、调度算法
 */

const { EmotionTaskMatrix, HeartbeatMonitor, EMOTIONS, TASK_TYPES } = require('./emotion_task_matrix.js');

// ==================== 测试工具 ====================

class TestRunner {
  constructor() {
    this.tests = [];
    this.results = {
      passed: 0,
      failed: 0,
      errors: []
    };
  }
  
  test(name, fn) {
    this.tests.push({ name, fn });
  }
  
  async run() {
    console.log('\n========================================');
    console.log('  EmotionTaskMatrix 测试套件');
    console.log('========================================\n');
    
    for (const { name, fn } of this.tests) {
      try {
        await fn();
        console.log(`✅ ${name}`);
        this.results.passed++;
      } catch (err) {
        console.log(`❌ ${name}`);
        console.log(`   错误: ${err.message}`);
        this.results.failed++;
        this.results.errors.push({ name, error: err.message });
      }
    }
    
    console.log('\n========================================');
    console.log(`  测试结果: ${this.results.passed} 通过, ${this.results.failed} 失败`);
    console.log('========================================\n');
    
    return this.results;
  }
  
  assert(condition, message) {
    if (!condition) {
      throw new Error(message || '断言失败');
    }
  }
  
  assertEqual(actual, expected, message) {
    if (actual !== expected) {
      throw new Error(message || `期望 ${expected}, 实际 ${actual}`);
    }
  }
  
  assertTrue(value, message) {
    if (value !== true) {
      throw new Error(message || `期望 true, 实际 ${value}`);
    }
  }
  
  assertInRange(value, min, max, message) {
    if (value < min || value > max) {
      throw new Error(message || `期望在 ${min}-${max} 范围内, 实际 ${value}`);
    }
  }
}

// ==================== 测试用例 ====================

const runner = new TestRunner();

// 测试1: 基础初始化
runner.test('基础初始化 - 创建EmotionTaskMatrix实例', () => {
  const etm = new EmotionTaskMatrix();
  runner.assert(etm !== null, '实例创建失败');
  runner.assertEqual(etm.currentEmotion, 'NEUTRAL', '默认情绪应为NEUTRAL');
  runner.assert(etm.heartbeat !== null, 'HEARTBEAT应已初始化');
});

// 测试2: 16种情绪定义
runner.test('情绪定义 - 验证16种核心情绪', () => {
  const emotionKeys = Object.keys(EMOTIONS);
  runner.assertEqual(emotionKeys.length, 16, `应有16种情绪, 实际 ${emotionKeys.length}`);
  
  const requiredEmotions = [
    'JOY', 'GRATITUDE', 'HOPE', 'PRIDE',
    'CURIOSITY', 'CALM', 'NEUTRAL', 'SURPRISE',
    'CONFUSION', 'BOREDOM', 'ANXIETY', 'FRUSTRATION',
    'SADNESS', 'ANGER', 'FEAR', 'DISGUST'
  ];
  
  for (const emotion of requiredEmotions) {
    runner.assert(EMOTIONS[emotion] !== undefined, `缺少情绪: ${emotion}`);
    runner.assert(EMOTIONS[emotion].valence !== undefined, `${emotion} 缺少valence属性`);
    runner.assert(EMOTIONS[emotion].arousal !== undefined, `${emotion} 缺少arousal属性`);
    runner.assert(EMOTIONS[emotion].dimension !== undefined, `${emotion} 缺少dimension属性`);
  }
});

// 测试3: 任务类型定义
runner.test('任务类型 - 验证任务类型定义', () => {
  const taskKeys = Object.keys(TASK_TYPES);
  runner.assert(taskKeys.length >= 10, `应有至少10种任务类型, 实际 ${taskKeys.length}`);
  
  for (const key of taskKeys) {
    const task = TASK_TYPES[key];
    runner.assert(task.name !== undefined, `${key} 缺少name属性`);
    runner.assert(task.preferredEmotions !== undefined, `${key} 缺少preferredEmotions`);
    runner.assert(task.avoidEmotions !== undefined, `${key} 缺少avoidEmotions`);
    runner.assert(task.soulDimensions !== undefined, `${key} 缺少soulDimensions`);
    runner.assert(task.complexity !== undefined, `${key} 缺少complexity`);
  }
});

// 测试4: 情绪设置
runner.test('情绪管理 - 设置和获取情绪', () => {
  const etm = new EmotionTaskMatrix();
  
  etm.setEmotion('JOY', 0.8);
  const emotion = etm.getEmotion();
  
  runner.assertEqual(emotion.key, 'JOY', '当前情绪应为JOY');
  runner.assertEqual(emotion.intensity, 0.8, '情绪强度应为0.8');
  runner.assertEqual(emotion.name, '喜悦', '情绪名称应为喜悦');
});

// 测试5: 无效情绪处理
runner.test('情绪管理 - 无效情绪应抛出错误', () => {
  const etm = new EmotionTaskMatrix();
  
  let errorThrown = false;
  try {
    etm.setEmotion('INVALID_EMOTION');
  } catch (err) {
    errorThrown = true;
  }
  
  runner.assertTrue(errorThrown, '应抛出无效情绪错误');
});

// 测试6: 任务-情绪匹配
runner.test('匹配算法 - 计算任务-情绪匹配度', () => {
  const etm = new EmotionTaskMatrix();
  
  // 好奇情绪适合创意任务
  const creativeMatch = etm.calculateMatch('CREATIVE_IDEATION', 'CURIOSITY');
  runner.assertInRange(creativeMatch.score, 0.5, 1.0, '好奇-创意匹配度应在0.5-1.0之间');
  runner.assertTrue(creativeMatch.details.preferredMatch, '好奇应是创意的推荐情绪');
  
  // 平静情绪适合分析任务
  const analyticalMatch = etm.calculateMatch('ANALYTICAL_RESEARCH', 'CALM');
  runner.assertInRange(analyticalMatch.score, 0.5, 1.0, '平静-分析匹配度应在0.5-1.0之间');
  
  // 愤怒情绪不适合沟通任务
  const commMatch = etm.calculateMatch('COMMUNICATION', 'ANGER');
  runner.assertTrue(commMatch.details.avoidMatch, '愤怒应是沟通的避免情绪');
  runner.assertInRange(commMatch.score, 0, 0.5, '愤怒-沟通匹配度应较低');
});

// 测试7: 预计算匹配矩阵
runner.test('匹配矩阵 - 验证预计算矩阵完整性', () => {
  const etm = new EmotionTaskMatrix();
  const matrix = etm.matchMatrix;
  
  for (const taskKey of Object.keys(TASK_TYPES)) {
    runner.assert(matrix[taskKey] !== undefined, `矩阵缺少任务类型: ${taskKey}`);
    
    for (const emotionKey of Object.keys(EMOTIONS)) {
      const score = matrix[taskKey][emotionKey];
      runner.assert(score !== undefined, `矩阵缺少: ${taskKey} × ${emotionKey}`);
      runner.assertInRange(score, 0, 1, `匹配分数应在0-1范围内: ${score}`);
    }
  }
});

// 测试8: 任务队列管理
runner.test('任务队列 - 添加和获取任务', () => {
  const etm = new EmotionTaskMatrix();
  etm.setEmotion('CURIOSITY');
  
  const task = etm.addTask({
    type: 'CREATIVE_IDEATION',
    priority: 5,
    data: { topic: 'AI创新' }
  });
  
  runner.assert(task.id !== undefined, '任务应有ID');
  runner.assertEqual(task.type, 'CREATIVE_IDEATION', '任务类型应正确');
  runner.assertEqual(task.priority, 5, '任务优先级应正确');
  runner.assertInRange(task.matchScore, 0, 1, '任务匹配分数应在有效范围内');
  
  const queueStatus = etm.getQueueStatus();
  runner.assertEqual(queueStatus.queueLength, 1, '队列长度应为1');
});

// 测试9: 任务排序
runner.test('任务队列 - 按匹配度排序', () => {
  const etm = new EmotionTaskMatrix();
  etm.setEmotion('CALM');
  
  // 添加不同类型任务
  etm.addTask({ type: 'CRISIS_RESPONSE', priority: 5 });  // 高优先级但低匹配
  etm.addTask({ type: 'ANALYTICAL_RESEARCH', priority: 3 }); // 高匹配
  
  const nextTask = etm.getNextTask();
  // 检查返回的任务是否有效
  runner.assert(nextTask !== null, '应有可调度任务');
  // 分析任务应排在前面，因为它与平静情绪更匹配
  runner.assertTrue(
    nextTask.type === 'ANALYTICAL_RESEARCH' || nextTask.matchScore >= 0.5,
    '应优先返回匹配度高的任务或高匹配分数任务'
  );
});

// 测试10: 任务认领和完成
runner.test('任务生命周期 - 认领和完成任务', () => {
  const etm = new EmotionTaskMatrix();
  etm.setEmotion('JOY');
  
  const task = etm.addTask({ type: 'CODE_DEVELOPMENT', priority: 4 });
  const claimed = etm.claimTask(task.id);
  
  runner.assert(claimed !== null, '任务认领应成功');
  runner.assertEqual(claimed.status, 'active', '任务状态应为active');
  runner.assertEqual(etm.activeTask.id, task.id, '活跃任务应正确设置');
  
  const completed = etm.completeTask(task.id, { success: true });
  runner.assert(completed !== null, '任务完成应成功');
  runner.assertEqual(completed.status, 'completed', '任务状态应为completed');
  runner.assertEqual(etm.activeTask, null, '活跃任务应清空');
});

// 测试11: 任务推荐
runner.test('智能推荐 - 基于情绪推荐任务', () => {
  const etm = new EmotionTaskMatrix();
  etm.setEmotion('CURIOSITY');
  
  const recommendations = etm.recommendTasks(5);
  runner.assertEqual(recommendations.length, 5, '应返回5个推荐');
  
  // 验证按匹配度排序
  for (let i = 1; i < recommendations.length; i++) {
    runner.assertTrue(
      recommendations[i-1].score >= recommendations[i].score,
      '推荐应按匹配度降序排列'
    );
  }
  
  // 好奇情绪应推荐创意和研究任务
  const topTask = recommendations[0].taskType;
  const expectedTasks = ['CREATIVE_IDEATION', 'ANALYTICAL_RESEARCH', 'LEARNING'];
  runner.assertTrue(
    expectedTasks.includes(topTask),
    `好奇情绪应推荐创意/研究类任务, 实际: ${topTask}`
  );
});

// 测试12: 情绪推荐
runner.test('智能推荐 - 为任务推荐情绪', () => {
  const etm = new EmotionTaskMatrix();
  
  const emotions = etm.recommendEmotions('CREATIVE_IDEATION', 3);
  runner.assertEqual(emotions.length, 3, '应返回3个推荐情绪');
  
  const topEmotion = emotions[0].emotion;
  runner.assertTrue(
    ['CURIOSITY', 'JOY', 'SURPRISE', 'HOPE'].includes(topEmotion),
    `创意任务应推荐积极/好奇类情绪, 实际: ${topEmotion}`
  );
});

// 测试13: 自适应调度
runner.test('智能调度 - 情绪自适应调度', () => {
  const etm = new EmotionTaskMatrix();
  
  // 添加各种任务
  etm.addTask({ type: 'CODE_DEVELOPMENT', priority: 4 });  // 复杂度4
  etm.addTask({ type: 'DOCUMENTATION', priority: 3 });      // 复杂度2
  etm.addTask({ type: 'CRISIS_RESPONSE', priority: 5 });    // 复杂度5
  
  // 设置低唤醒情绪
  etm.setEmotion('CALM', 0.5);
  const calmSchedule = etm.adaptiveSchedule();
  runner.assert(calmSchedule !== null, '调度应成功');
  
  // 重置并设置高积极情绪
  etm.activeTask = null;
  etm.setEmotion('JOY', 0.9);
  etm.addTask({ type: 'CODE_DEVELOPMENT', priority: 4 });
  etm.addTask({ type: 'DOCUMENTATION', priority: 3 });
  
  const joySchedule = etm.adaptiveSchedule();
  runner.assert(joySchedule !== null, 'JOY情绪下调度应成功');
});

// 测试14: HEARTBEAT监控
runner.test('HEARTBEAT - 监控情绪记录', () => {
  const monitor = new HeartbeatMonitor({ interval: 100 });
  
  monitor.recordEmotion('JOY', 0.8, { source: 'test' });
  monitor.recordEmotion('CALM', 0.6, { source: 'test' });
  
  const trend = monitor.getEmotionTrend();
  runner.assert(trend !== null, '应返回情绪趋势');
  runner.assertEqual(trend.windowSize, 2, '趋势窗口应包含2条记录');
  runner.assertInRange(trend.averageValence, -1, 1, '平均valence应在有效范围');
});

// 测试15: HEARTBEAT任务匹配记录
runner.test('HEARTBEAT - 任务匹配记录', () => {
  const monitor = new HeartbeatMonitor();
  
  monitor.recordTaskMatch('CODE_DEVELOPMENT', 'JOY', 0.85, { taskId: 'test-1' });
  monitor.recordTaskMatch('COMMUNICATION', 'ANGER', 0.2, { taskId: 'test-2' });
  
  const report = monitor.getReport();
  runner.assertEqual(report.stats.totalTasks, 2, '应记录2个任务');
  runner.assertEqual(report.stats.matchedTasks, 1, '应有1个匹配任务');
  
  const alerts = monitor.getAlerts();
  runner.assert(alerts.length > 0, '应有低匹配度告警');
});

// 测试16: HEARTBEAT健康分数
runner.test('HEARTBEAT - 健康分数计算', () => {
  const monitor = new HeartbeatMonitor();
  
  // 记录一些积极数据
  monitor.recordEmotion('JOY', 0.8);
  monitor.recordTaskMatch('CODE_DEVELOPMENT', 'JOY', 0.9);
  
  const report = monitor.getReport();
  runner.assertInRange(report.health, 0, 100, '健康分数应在0-100范围内');
  runner.assert(report.health > 50, '积极数据应有较高健康分数');
});

// 测试17: 事件系统
runner.test('事件系统 - 情绪变化事件', async () => {
  const etm = new EmotionTaskMatrix();
  let eventFired = false;
  let eventData = null;
  
  etm.on('emotion_changed', (event, data) => {
    eventFired = true;
    eventData = data;
  });
  
  etm.setEmotion('HOPE', 0.7);
  
  // 等待事件处理
  await new Promise(resolve => setTimeout(resolve, 10));
  
  runner.assertTrue(eventFired, '情绪变化事件应被触发');
  runner.assertEqual(eventData.to, 'HOPE', '事件数据应包含新情绪');
});

// 测试18: 情绪维度查询
runner.test('情绪查询 - 按SOUL维度获取情绪', () => {
  const etm = new EmotionTaskMatrix();
  
  const growthEmotions = etm.getEmotionsByDimension('Growth');
  runner.assert(growthEmotions.length > 0, 'Growth维度应有情绪');
  
  for (const emotion of growthEmotions) {
    runner.assertEqual(emotion.dimension, 'Growth', '情绪维度应匹配');
  }
});

// 测试19: 完整工作流
runner.test('集成测试 - 完整工作流', async () => {
  const etm = new EmotionTaskMatrix();
  etm.start();
  
  // 设置情绪
  etm.setEmotion('CURIOSITY', 0.8);
  
  // 添加任务
  const task1 = etm.addTask({ type: 'CREATIVE_IDEATION', priority: 5, data: { topic: 'AI' } });
  const task2 = etm.addTask({ type: 'ANALYTICAL_RESEARCH', priority: 4, data: { topic: 'Data' } });
  
  // 自动调度
  const scheduled = etm.autoSchedule();
  runner.assert(scheduled !== null, '自动调度应成功');
  
  // 完成任务
  const completed = etm.completeTask(scheduled.id, { result: 'success' });
  runner.assert(completed !== null, '任务完成应成功');
  
  // 获取状态报告
  const status = etm.getStatus();
  runner.assert(status.currentEmotion !== undefined, '状态应包含当前情绪');
  runner.assert(status.queue !== undefined, '状态应包含队列信息');
  
  etm.stop();
});

// 测试20: 边界条件
runner.test('边界条件 - 极端值处理', () => {
  const etm = new EmotionTaskMatrix();
  
  // 测试情绪强度边界
  etm.setEmotion('JOY', 0);  // 最小值
  runner.assertEqual(etm.emotionIntensity, 0, '情绪强度应为0');
  
  etm.setEmotion('JOY', 1);  // 最大值
  runner.assertEqual(etm.emotionIntensity, 1, '情绪强度应为1');
  
  etm.setEmotion('JOY', 1.5); // 超出范围
  runner.assertEqual(etm.emotionIntensity, 1, '情绪强度应被限制为1');
  
  etm.setEmotion('JOY', -0.5); // 负值
  runner.assertEqual(etm.emotionIntensity, 0, '情绪强度应被限制为0');
  
  // 测试空队列
  const nextTask = etm.getNextTask();
  runner.assertEqual(nextTask, null, '空队列应返回null');
  
  // 测试无效任务ID
  const claimed = etm.claimTask('invalid-id');
  runner.assertEqual(claimed, null, '无效任务ID应返回null');
});

// ==================== 运行测试 ====================

async function runTests() {
  const results = await runner.run();
  
  // 打印详细报告
  console.log('\n📊 详细报告:');
  console.log(`- 情绪类型: ${Object.keys(EMOTIONS).length} 种`);
  console.log(`- 任务类型: ${Object.keys(TASK_TYPES).length} 种`);
  console.log(`- 匹配矩阵: ${Object.keys(TASK_TYPES).length} × ${Object.keys(EMOTIONS).length} = ${Object.keys(TASK_TYPES).length * Object.keys(EMOTIONS).length} 个组合`);
  
  if (results.failed === 0) {
    console.log('\n🎉 所有测试通过！');
    process.exit(0);
  } else {
    console.log(`\n⚠️ 有 ${results.failed} 个测试失败`);
    process.exit(1);
  }
}

runTests().catch(err => {
  console.error('测试运行错误:', err);
  process.exit(1);
});
