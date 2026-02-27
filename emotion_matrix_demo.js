/**
 * EmotionTaskMatrix 快速演示
 * 展示16情绪×任务类型映射的核心功能
 */

const { EmotionTaskMatrix, EMOTIONS, TASK_TYPES } = require('./emotion_task_matrix.js');

console.log('╔════════════════════════════════════════════════════════════╗');
console.log('║     EmotionTaskMatrix - 情绪-任务矩阵核心演示              ║');
console.log('╚════════════════════════════════════════════════════════════╝\n');

// 创建实例并启动
const etm = new EmotionTaskMatrix({
  enableHeartbeat: true,
  heartbeat: { interval: 2000 }
});

etm.start();

// 监听事件
etm.on('*', (event, data) => {
  if (event === 'emotion_changed') {
    console.log(`🎭 情绪变化: ${data.from} → ${data.to} (强度: ${data.intensity})`);
  }
  if (event === 'task_started') {
    console.log(`🚀 任务启动: ${data.type} (匹配度: ${(data.matchScore * 100).toFixed(1)}%)`);
  }
  if (event === 'task_completed') {
    console.log(`✅ 任务完成: ${data.type}`);
  }
});

console.log('\n📊 系统概览');
console.log('─────────────────────────────────');
console.log(`情绪类型: ${Object.keys(EMOTIONS).length} 种`);
console.log(`任务类型: ${Object.keys(TASK_TYPES).length} 种`);
console.log(`匹配矩阵: ${Object.keys(TASK_TYPES).length} × ${Object.keys(EMOTIONS).length} = ${Object.keys(TASK_TYPES).length * Object.keys(EMOTIONS).length} 个组合\n`);

// 展示16种情绪
console.log('🎨 16种核心情绪 (基于SOUL 8维度人格模型)');
console.log('─────────────────────────────────');
const emotionsByDimension = {};
Object.entries(EMOTIONS).forEach(([key, data]) => {
  if (!emotionsByDimension[data.dimension]) {
    emotionsByDimension[data.dimension] = [];
  }
  emotionsByDimension[data.dimension].push(`${data.name}(${key})`);
});

Object.entries(emotionsByDimension).forEach(([dimension, emotions]) => {
  console.log(`${dimension}: ${emotions.join(', ')}`);
});

// 演示1: 好奇情绪下的任务推荐
console.log('\n\n🔍 演示1: 好奇(CURIOSITY)情绪下的任务推荐');
console.log('─────────────────────────────────');
etm.setEmotion('CURIOSITY', 0.8);
const curiosityRecs = etm.recommendTasks(5);
curiosityRecs.forEach((rec, i) => {
  const task = TASK_TYPES[rec.taskType];
  console.log(`${i + 1}. ${task.name} - 匹配度: ${(rec.score * 100).toFixed(1)}% ${rec.details.preferredMatch ? '⭐' : ''}`);
});

// 演示2: 平静情绪下的任务推荐
console.log('\n\n🧘 演示2: 平静(CALM)情绪下的任务推荐');
console.log('─────────────────────────────────');
etm.setEmotion('CALM', 0.6);
const calmRecs = etm.recommendTasks(5);
calmRecs.forEach((rec, i) => {
  const task = TASK_TYPES[rec.taskType];
  console.log(`${i + 1}. ${task.name} - 匹配度: ${(rec.score * 100).toFixed(1)}% ${rec.details.preferredMatch ? '⭐' : ''}`);
});

// 演示3: 焦虑情绪下的任务推荐
console.log('\n\n😰 演示3: 焦虑(ANXIETY)情绪下的任务推荐');
console.log('─────────────────────────────────');
etm.setEmotion('ANXIETY', 0.7);
const anxietyRecs = etm.recommendTasks(5);
anxietyRecs.forEach((rec, i) => {
  const task = TASK_TYPES[rec.taskType];
  const warning = rec.details.avoidMatch ? '⚠️ 避免' : '';
  console.log(`${i + 1}. ${task.name} - 匹配度: ${(rec.score * 100).toFixed(1)}% ${warning}`);
});

// 演示4: 任务调度流程
console.log('\n\n⚙️ 演示4: 智能任务调度流程');
console.log('─────────────────────────────────');

// 清空之前的状态
etm.taskQueue = [];
etm.activeTask = null;

// 设置好奇情绪，添加创意任务
etm.setEmotion('CURIOSITY', 0.9);
etm.addTask({ type: 'CREATIVE_IDEATION', priority: 5, data: { topic: 'AI创新' } });
etm.addTask({ type: 'ANALYTICAL_RESEARCH', priority: 4, data: { topic: '数据分析' } });
etm.addTask({ type: 'DOCUMENTATION', priority: 2, data: { topic: '文档整理' } });

console.log('\n当前队列状态:');
const queueStatus = etm.getQueueStatus();
queueStatus.tasks.forEach((t, i) => {
  console.log(`  ${i + 1}. ${TASK_TYPES[t.type].name} - 匹配度: ${(t.matchScore * 100).toFixed(1)}%`);
});

// 自动调度
console.log('\n执行自动调度...');
const scheduled = etm.autoSchedule();
if (scheduled) {
  console.log(`已调度: ${TASK_TYPES[scheduled.type].name}`);
  
  // 模拟任务执行
  setTimeout(() => {
    etm.completeTask(scheduled.id, { result: 'success' });
    
    // 演示5: HEARTBEAT监控报告
    console.log('\n\n💓 演示5: HEARTBEAT监控报告');
    console.log('─────────────────────────────────');
    const report = etm.heartbeat.getReport();
    console.log(`健康分数: ${report.health}/100`);
    console.log(`总任务数: ${report.stats.totalTasks}`);
    console.log(`匹配任务: ${report.stats.matchedTasks}`);
    console.log(`平均匹配分数: ${(report.stats.averageMatchScore * 100).toFixed(1)}%`);
    console.log(`情绪转换次数: ${report.stats.emotionTransitions}`);
    
    if (report.trend) {
      console.log(`\n情绪趋势: ${report.trend.trend}`);
      console.log(`主导情绪: ${EMOTIONS[report.trend.dominantEmotion]?.name || report.trend.dominantEmotion}`);
      console.log(`平均Valence: ${report.trend.averageValence.toFixed(2)}`);
      console.log(`平均Arousal: ${report.trend.averageArousal.toFixed(2)}`);
    }
    
    // 演示6: 为特定任务推荐最佳情绪
    console.log('\n\n🎯 演示6: 为"代码开发"任务推荐最佳情绪');
    console.log('─────────────────────────────────');
    const codeEmotions = etm.recommendEmotions('CODE_DEVELOPMENT', 5);
    codeEmotions.forEach((rec, i) => {
      console.log(`${i + 1}. ${rec.name}(${rec.emotion}) - 匹配度: ${(rec.score * 100).toFixed(1)}%`);
    });
    
    // 结束演示
    console.log('\n\n╔════════════════════════════════════════════════════════════╗');
    console.log('║              演示完成 - 系统停止中...                      ║');
    console.log('╚════════════════════════════════════════════════════════════╝\n');
    
    etm.stop();
    process.exit(0);
  }, 500);
}
