#!/usr/bin/env node
/**
 * 记忆系统核心模块
 * 提供三层记忆的读写和流转功能
 */

const fs = require('fs').promises;
const path = require('path');

const MEMORY_BASE_PATH = '/root/.openclaw/workspace/memory';

// ==================== 路径工具 ====================

function getShortTermPath(filename) {
  return path.join(MEMORY_BASE_PATH, 'short-term', filename);
}

function getMidTermPath(...parts) {
  return path.join(MEMORY_BASE_PATH, 'mid-term', ...parts);
}

function getLongTermPath(...parts) {
  return path.join(MEMORY_BASE_PATH, 'long-term', ...parts);
}

function getArchivePath(...parts) {
  return path.join(MEMORY_BASE_PATH, 'archive', ...parts);
}

// ==================== 文件操作 ====================

async function readJson(filePath, defaultValue = {}) {
  try {
    const content = await fs.readFile(filePath, 'utf-8');
    return JSON.parse(content);
  } catch (e) {
    return defaultValue;
  }
}

async function writeJson(filePath, data) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, JSON.stringify(data, null, 2), 'utf-8');
}

async function readMarkdown(filePath, defaultValue = '') {
  try {
    return await fs.readFile(filePath, 'utf-8');
  } catch (e) {
    return defaultValue;
  }
}

async function writeMarkdown(filePath, content) {
  await fs.mkdir(path.dirname(filePath), { recursive: true });
  await fs.writeFile(filePath, content, 'utf-8');
}

// ==================== 短期记忆 API ====================

const shortTerm = {
  /**
   * 获取短期记忆值
   * @param {string} key - 键名
   * @param {*} defaultValue - 默认值
   */
  async get(key, defaultValue = null) {
    const data = await readJson(getShortTermPath('context_stack.json'));
    return data.temp_data?.[key] ?? defaultValue;
  },

  /**
   * 设置短期记忆值
   * @param {string} key - 键名
   * @param {*} value - 值
   * @param {number} ttl - 生存时间(秒)，可选
   */
  async set(key, value, ttl = null) {
    const filePath = getShortTermPath('context_stack.json');
    const data = await readJson(filePath);
    
    if (!data.temp_data) data.temp_data = {};
    data.temp_data[key] = {
      value,
      created_at: new Date().toISOString(),
      ttl: ttl ? Date.now() + ttl * 1000 : null
    };
    data.last_updated = new Date().toISOString();
    
    await writeJson(filePath, data);
  },

  /**
   * 清除所有短期记忆
   */
  async clear() {
    const filePath = getShortTermPath('context_stack.json');
    const data = await readJson(filePath);
    data.temp_data = {};
    data.last_updated = new Date().toISOString();
    await writeJson(filePath, data);
  },

  /**
   * 获取当前会话状态
   */
  async getSession() {
    return readMarkdown(getShortTermPath('current_session.md'), '');
  },

  /**
   * 更新当前会话状态
   * @param {string} content - 会话内容
   */
  async updateSession(content) {
    await writeMarkdown(getShortTermPath('current_session.md'), content);
  }
};

// ==================== 中期记忆 API ====================

const midTerm = {
  /**
   * 获取项目记忆
   * @param {string} projectId - 项目ID
   */
  async getProject(projectId) {
    const statusPath = getMidTermPath('projects', projectId, 'status.md');
    const tasksPath = getMidTermPath('projects', projectId, 'tasks.json');
    
    return {
      id: projectId,
      status: await readMarkdown(statusPath),
      tasks: await readJson(tasksPath, { tasks: [] })
    };
  },

  /**
   * 更新项目状态
   * @param {string} projectId - 项目ID
   * @param {Object} updates - 更新内容
   */
  async updateProject(projectId, updates) {
    if (updates.status) {
      await writeMarkdown(
        getMidTermPath('projects', projectId, 'status.md'),
        updates.status
      );
    }
    if (updates.tasks) {
      const tasksPath = getMidTermPath('projects', projectId, 'tasks.json');
      const existing = await readJson(tasksPath, { tasks: [] });
      existing.tasks = updates.tasks;
      existing.last_updated = new Date().toISOString();
      await writeJson(tasksPath, existing);
    }
  },

  /**
   * 列出所有活跃项目
   */
  async listActiveProjects() {
    const contexts = await readJson(getMidTermPath('active_contexts.json'));
    const projects = [];
    
    for (const projectId of contexts.active_projects || []) {
      projects.push(await this.getProject(projectId));
    }
    
    return projects;
  },

  /**
   * 添加活跃项目
   * @param {string} projectId - 项目ID
   */
  async addActiveProject(projectId) {
    const contextsPath = getMidTermPath('active_contexts.json');
    const contexts = await readJson(contextsPath, { active_projects: [] });
    
    if (!contexts.active_projects.includes(projectId)) {
      contexts.active_projects.push(projectId);
      contexts.last_context_switch = new Date().toISOString();
      await writeJson(contextsPath, contexts);
    }
  }
};

// ==================== 长期记忆 API ====================

const longTerm = {
  /**
   * 获取用户画像
   */
  async getUserProfile() {
    return readMarkdown(getLongTermPath('user_profile.md'), '');
  },

  /**
   * 更新用户画像
   * @param {string} content - 新内容或追加内容
   * @param {boolean} append - 是否追加
   */
  async updateUserProfile(content, append = false) {
    const filePath = getLongTermPath('user_profile.md');
    if (append) {
      const existing = await readMarkdown(filePath);
      content = existing + '\n\n' + content;
    }
    await writeMarkdown(filePath, content);
  },

  /**
   * 添加知识
   * @param {string} domain - 领域名称
   * @param {string} content - 知识内容
   * @param {string[]} tags - 标签
   */
  async addKnowledge(domain, content, tags = []) {
    const filePath = getLongTermPath('knowledge_base', 'domains', `${domain}.md`);
    const timestamp = new Date().toISOString();
    const tagStr = tags.length > 0 ? `\n*标签: ${tags.join(', ')}*` : '';
    
    const entry = `\n\n---\n\n## 知识条目 [${timestamp}]${tagStr}\n\n${content}\n`;
    
    const existing = await readMarkdown(filePath, `# ${domain} 领域知识\n`);
    await writeMarkdown(filePath, existing + entry);
  },

  /**
   * 搜索知识 (简单关键词匹配)
   * @param {string} query - 搜索关键词
   */
  async searchKnowledge(query) {
    const domainsPath = getLongTermPath('knowledge_base', 'domains');
    const results = [];
    
    try {
      const files = await fs.readdir(domainsPath);
      for (const file of files) {
        if (file.endsWith('.md')) {
          const content = await readMarkdown(path.join(domainsPath, file));
          if (content.toLowerCase().includes(query.toLowerCase())) {
            results.push({
              domain: file.replace('.md', ''),
              preview: content.substring(0, 200) + '...'
            });
          }
        }
      }
    } catch (e) {
      // 目录不存在
    }
    
    return results;
  },

  /**
   * 记录决策
   * @param {Object} decision - 决策记录
   */
  async recordDecision(decision) {
    const date = new Date().toISOString().split('T')[0];
    const filename = `${date}_${decision.id || 'decision'}.md`;
    const filePath = getLongTermPath('decisions', filename);
    
    const content = `# 决策记录: ${decision.title}\n\n` +
      `**日期**: ${decision.date || date}\n\n` +
      `**背景**: ${decision.background}\n\n` +
      `**决策**: ${decision.decision}\n\n` +
      `**原因**: ${decision.reason}\n\n` +
      (decision.impact ? `**影响**: ${decision.impact}\n\n` : '') +
      (decision.alternatives ? `**备选方案**: ${decision.alternatives}\n\n` : '');
    
    await writeMarkdown(filePath, content);
  }
};

// ==================== 记忆流转 API ====================

const lifecycle = {
  /**
   * 归档会话
   * @param {string} sessionId - 会话ID
   */
  async archiveSession(sessionId) {
    const date = new Date();
    const year = date.getFullYear();
    const month = String(date.getMonth() + 1).padStart(2, '0');
    const day = String(date.getDate()).padStart(2, '0');
    
    const archiveDir = getArchivePath('sessions', String(year), month, day);
    await fs.mkdir(archiveDir, { recursive: true });
    
    // 归档短期记忆
    const shortTermFiles = ['current_session.md', 'context_stack.json'];
    for (const file of shortTermFiles) {
      const srcPath = getShortTermPath(file);
      const destPath = path.join(archiveDir, `${sessionId}_${file}`);
      try {
        await fs.copyFile(srcPath, destPath);
        // 清空原文件
        if (file === 'context_stack.json') {
          await writeJson(srcPath, {
            session_id: sessionId,
            archived_at: new Date().toISOString(),
            context_stack: [],
            temp_data: {}
          });
        }
      } catch (e) {
        // 文件不存在，跳过
      }
    }
    
    return archiveDir;
  },

  /**
   * 将短期记忆提升到中期记忆
   * @param {string[]} keys - 要提升的键名
   * @param {string} projectId - 目标项目ID
   */
  async promoteToMidTerm(keys, projectId) {
    const contextData = await readJson(getShortTermPath('context_stack.json'));
    const extracted = {};
    
    for (const key of keys) {
      if (contextData.temp_data?.[key]) {
        extracted[key] = contextData.temp_data[key];
      }
    }
    
    // 添加到项目笔记
    const notesPath = getMidTermPath('projects', projectId, 'notes.md');
    const timestamp = new Date().toISOString();
    const noteContent = `\n\n## 提取自短期记忆 [${timestamp}]\n\n` +
      '```json\n' + JSON.stringify(extracted, null, 2) + '\n```\n';
    
    const existing = await readMarkdown(notesPath, `# 项目笔记\n`);
    await writeMarkdown(notesPath, existing + noteContent);
    
    return extracted;
  },

  /**
   * 将项目提升到长期记忆 (项目完成时)
   * @param {string} projectId - 项目ID
   */
  async promoteToLongTerm(projectId) {
    const project = await midTerm.getProject(projectId);
    
    // 记录到决策/经验
    await longTerm.recordDecision({
      id: `project_complete_${projectId}`,
      title: `项目完成: ${projectId}`,
      background: `项目 ${projectId} 已完成`,
      decision: '归档项目并提取经验教训',
      reason: '项目周期结束，需要沉淀知识',
      impact: '相关经验已记录到长期记忆'
    });
    
    // 归档项目
    const archiveDir = getArchivePath('projects', projectId);
    await fs.mkdir(archiveDir, { recursive: true });
    
    const projectDir = getMidTermPath('projects', projectId);
    const files = await fs.readdir(projectDir);
    for (const file of files) {
      await fs.copyFile(
        path.join(projectDir, file),
        path.join(archiveDir, file)
      );
    }
    
    // 从活跃项目移除
    const contextsPath = getMidTermPath('active_contexts.json');
    const contexts = await readJson(contextsPath);
    contexts.active_projects = contexts.active_projects.filter(id => id !== projectId);
    await writeJson(contextsPath, contexts);
    
    return archiveDir;
  }
};

// ==================== 会话生命周期 ====================

const session = {
  /**
   * 会话启动时调用
   */
  async onStart() {
    console.log('🧠 记忆系统初始化...');
    
    // 1. 加载短期记忆
    const contextData = await readJson(getShortTermPath('context_stack.json'));
    console.log(`📋 恢复会话: ${contextData.session_id || '新会话'}`);
    
    // 2. 加载活跃项目
    const projects = await midTerm.listActiveProjects();
    console.log(`📁 活跃项目: ${projects.length} 个`);
    projects.forEach(p => console.log(`   - ${p.id}`));
    
    // 3. 加载用户画像摘要
    const profile = await longTerm.getUserProfile();
    console.log('👤 用户画像已加载');
    
    return {
      context: contextData,
      projects,
      profile
    };
  },

  /**
   * 会话结束时调用
   * @param {string} sessionId - 会话ID
   */
  async onEnd(sessionId) {
    console.log('💾 保存会话状态...');
    
    // 1. 归档短期记忆
    const archiveDir = await lifecycle.archiveSession(sessionId);
    console.log(`📦 已归档到: ${archiveDir}`);
    
    // 2. 更新项目状态
    const contexts = await readJson(getMidTermPath('active_contexts.json'));
    for (const projectId of contexts.active_projects || []) {
      const tasksPath = getMidTermPath('projects', projectId, 'tasks.json');
      const tasks = await readJson(tasksPath);
      tasks.last_updated = new Date().toISOString();
      await writeJson(tasksPath, tasks);
    }
    
    console.log('✅ 会话状态已保存');
  }
};

// ==================== 导出 ====================

module.exports = {
  shortTerm,
  midTerm,
  longTerm,
  lifecycle,
  session,
  // 工具函数
  utils: {
    readJson,
    writeJson,
    readMarkdown,
    writeMarkdown
  }
};

// 如果直接运行此脚本，执行演示
if (require.main === module) {
  (async () => {
    console.log('=== 记忆系统演示 ===\n');
    
    // 演示: 会话启动
    await session.onStart();
    
    console.log('\n--- 短期记忆操作 ---');
    await shortTerm.set('demo_key', 'Hello Memory System!', 3600);
    const value = await shortTerm.get('demo_key');
    console.log('读取值:', value);
    
    console.log('\n--- 中期记忆操作 ---');
    const projects = await midTerm.listActiveProjects();
    console.log('活跃项目数:', projects.length);
    
    console.log('\n--- 长期记忆操作 ---');
    const profile = await longTerm.getUserProfile();
    console.log('用户画像长度:', profile.length, '字符');
    
    console.log('\n=== 演示完成 ===');
  })();
}
