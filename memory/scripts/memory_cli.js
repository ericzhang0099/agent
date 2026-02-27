#!/usr/bin/env node
/**
 * 记忆系统 CLI 工具
 * 命令行接口用于手动操作记忆系统
 */

const memory = require('./memory_system');

const commands = {
  /**
   * 获取帮助信息
   */
  help() {
    console.log(`
记忆系统 CLI 工具

用法: node memory_cli.js <命令> [参数]

命令:
  init                    初始化记忆系统，加载会话上下文
  save <session-id>       保存并归档会话
  
  st-get <key>            获取短期记忆值
  st-set <key> <value>    设置短期记忆值
  st-clear                清除所有短期记忆
  
  project-list            列出所有活跃项目
  project-show <id>       显示项目详情
  project-update <id>     更新项目状态 (交互式)
  
  profile                 显示用户画像
  knowledge-search <query> 搜索知识库
  
  archive <session-id>    归档指定会话
  promote <project-id>    将项目提升到长期记忆

示例:
  node memory_cli.js init
  node memory_cli.js st-set current_task "编写文档"
  node memory_cli.js project-list
  node memory_cli.js knowledge-search "架构"
`);
  },

  /**
   * 初始化会话
   */
  async init() {
    const state = await memory.session.onStart();
    console.log('\n会话初始化完成!');
    console.log('活跃项目:', state.projects.map(p => p.id).join(', ') || '无');
  },

  /**
   * 保存会话
   */
  async save(sessionId) {
    if (!sessionId) {
      console.error('错误: 请提供会话ID');
      process.exit(1);
    }
    await memory.session.onEnd(sessionId);
  },

  /**
   * 获取短期记忆
   */
  async stGet(key) {
    const value = await memory.shortTerm.get(key);
    console.log(value !== null ? value : '(未设置)');
  },

  /**
   * 设置短期记忆
   */
  async stSet(key, ...valueParts) {
    const value = valueParts.join(' ');
    await memory.shortTerm.set(key, value);
    console.log(`已设置: ${key} = ${value}`);
  },

  /**
   * 清除短期记忆
   */
  async stClear() {
    await memory.shortTerm.clear();
    console.log('短期记忆已清除');
  },

  /**
   * 列出项目
   */
  async projectList() {
    const projects = await memory.midTerm.listActiveProjects();
    if (projects.length === 0) {
      console.log('暂无活跃项目');
      return;
    }
    
    console.log('\n活跃项目列表:');
    console.log('-'.repeat(50));
    projects.forEach(p => {
      const completed = p.tasks.tasks?.filter(t => t.status === 'completed').length || 0;
      const total = p.tasks.tasks?.length || 0;
      console.log(`📁 ${p.id}`);
      console.log(`   任务: ${completed}/${total} 完成`);
      console.log(`   更新: ${p.tasks.last_updated || '未知'}`);
    });
  },

  /**
   * 显示项目详情
   */
  async projectShow(projectId) {
    const project = await memory.midTerm.getProject(projectId);
    console.log(`\n项目: ${project.id}`);
    console.log('-'.repeat(50));
    console.log(project.status);
    console.log('\n任务列表:');
    project.tasks.tasks?.forEach(t => {
      const icon = t.status === 'completed' ? '✅' : t.status === 'in_progress' ? '🟡' : '⏳';
      console.log(`  ${icon} ${t.title}`);
    });
  },

  /**
   * 显示用户画像
   */
  async profile() {
    const profile = await memory.longTerm.getUserProfile();
    console.log(profile);
  },

  /**
   * 搜索知识
   */
  async knowledgeSearch(query) {
    const results = await memory.longTerm.searchKnowledge(query);
    if (results.length === 0) {
      console.log('未找到匹配的知识');
      return;
    }
    
    console.log(`\n找到 ${results.length} 条结果:`);
    results.forEach(r => {
      console.log(`\n📚 ${r.domain}`);
      console.log(r.preview);
    });
  },

  /**
   * 归档会话
   */
  async archive(sessionId) {
    const archiveDir = await memory.lifecycle.archiveSession(sessionId);
    console.log(`会话已归档到: ${archiveDir}`);
  },

  /**
   * 提升项目到长期记忆
   */
  async promote(projectId) {
    const archiveDir = await memory.lifecycle.promoteToLongTerm(projectId);
    console.log(`项目已归档到: ${archiveDir}`);
    console.log('经验教训已记录到长期记忆');
  }
};

// 主函数
async function main() {
  const [cmd, ...args] = process.argv.slice(2);
  
  if (!cmd || cmd === 'help' || cmd === '-h' || cmd === '--help') {
    commands.help();
    return;
  }
  
  const commandFn = commands[cmd.replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())];
  
  if (!commandFn) {
    console.error(`未知命令: ${cmd}`);
    console.log('使用 "help" 查看可用命令');
    process.exit(1);
  }
  
  try {
    await commandFn(...args);
  } catch (error) {
    console.error('错误:', error.message);
    process.exit(1);
  }
}

main();
