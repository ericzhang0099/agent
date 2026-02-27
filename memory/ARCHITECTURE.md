# 分层长期记忆系统架构文档

## 系统概述

本记忆系统为CEO Kimi Claw设计，采用三层架构实现从即时会话到永久知识库的记忆沉淀。

---

## 三层记忆架构

### 1. 短期记忆 (Short-Term Memory)
**位置**: `memory/short-term/`
**生命周期**: 当前会话 ~ 24小时
**用途**: 存储当前活跃会话的上下文、临时变量、用户即时需求

**文件结构**:
```
short-term/
├── current_session.md      # 当前会话状态
├── context_stack.json      # 上下文堆栈
└── temp/                   # 临时文件目录
    └── *.md
```

**内容示例**:
- 当前对话主题
- 待办事项列表
- 临时变量和中间结果
- 用户当前情绪/偏好

---

### 2. 中期记忆 (Mid-Term Memory)
**位置**: `memory/mid-term/`
**生命周期**: 跨会话项目周期 (1天 ~ 数月)
**用途**: 项目状态、进行中任务的上下文、跨会话的工作流

**文件结构**:
```
mid-term/
├── projects/               # 项目目录
│   ├── {project_id}/
│   │   ├── status.md       # 项目状态
│   │   ├── tasks.json      # 任务列表
│   │   └── notes.md        # 项目笔记
├── active_contexts.json    # 活跃上下文索引
└── workflows/              # 工作流模板
    └── *.md
```

**内容示例**:
- 进行中的项目状态
- 待完成的任务清单
- 跨会话的用户偏好设置
- 活跃的工作流定义

---

### 3. 长期记忆 (Long-Term Memory)
**位置**: `memory/long-term/`
**生命周期**: 永久
**用途**: 知识库沉淀、用户画像、经验总结、重要决策记录

**文件结构**:
```
long-term/
├── user_profile.md         # 用户画像
├── knowledge_base/         # 知识库
│   ├── domains/            # 领域知识
│   ├── lessons/            # 经验教训
│   └── patterns/           # 模式识别
├── decisions/              # 重要决策记录
│   └── YYYY-MM-DD_*.md
├── relationships/          # 关系网络
│   └── contacts.md
└── system/                 # 系统级记忆
    ├── capabilities.md     # 能力清单
    └── preferences.md      # 系统偏好
```

**内容示例**:
- 用户长期偏好和习惯
- 重要决策及其原因
- 经验教训总结
- 领域知识积累
- 人际关系网络

---

### 4. 归档区 (Archive)
**位置**: `memory/archive/`
**用途**: 过期短期/中期记忆的归档存储

**文件结构**:
```
archive/
├── sessions/               # 会话归档
│   └── YYYY/MM/DD/
├── projects/               # 已完成项目
│   └── {project_id}/
└── daily/                  # 每日记忆归档
    └── YYYY-MM-DD.md
```

---

## 记忆流转机制

```
┌─────────────────────────────────────────────────────────────┐
│                      记忆流转生命周期                          │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│   [会话开始]                                                │
│       │                                                     │
│       ▼                                                     │
│   ┌──────────────┐    24小时后/会话结束    ┌──────────────┐  │
│   │  短期记忆     │ ──────────────────────▶ │   归档区      │  │
│   │  (临时状态)   │                         │  (历史记录)   │  │
│   └──────────────┘                         └──────────────┘  │
│          │                                                        │
│          │ 重要信息提取                                            │
│          ▼                                                        │
│   ┌──────────────┐    项目完成/重要发现     ┌──────────────┐  │
│   │  中期记忆     │ ──────────────────────▶ │   长期记忆    │  │
│   │  (项目状态)   │                         │  (知识沉淀)   │  │
│   └──────────────┘                         └──────────────┘  │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### 流转规则

| 源 | 目标 | 触发条件 | 处理方式 |
|---|---|---|---|
| 短期记忆 | 归档区 | 会话结束/24小时 | 自动归档，保留原始记录 |
| 短期记忆 | 中期记忆 | 检测到项目上下文 | 提取项目相关信息 |
| 中期记忆 | 长期记忆 | 项目完成/重要决策 | 总结提炼，知识沉淀 |
| 中期记忆 | 归档区 | 项目取消/过期 | 标记状态，移至归档 |
| 长期记忆 | 归档区 | 信息过时/被替代 | 版本控制，保留历史 |

---

## 记忆提取与存储接口

### 核心接口定义

```typescript
interface MemorySystem {
  // 短期记忆操作
  shortTerm: {
    get(key: string): any;
    set(key: string, value: any, ttl?: number): void;
    clear(): void;
  };
  
  // 中期记忆操作
  midTerm: {
    getProject(projectId: string): ProjectMemory;
    updateProject(projectId: string, data: Partial<ProjectMemory>): void;
    listActiveProjects(): ProjectMemory[];
  };
  
  // 长期记忆操作
  longTerm: {
    getUserProfile(): UserProfile;
    updateUserProfile(updates: Partial<UserProfile>): void;
    addKnowledge(domain: string, content: string, tags?: string[]): void;
    searchKnowledge(query: string): KnowledgeItem[];
    recordDecision(decision: DecisionRecord): void;
  };
  
  // 记忆流转
  lifecycle: {
    archive(sessionId: string): void;
    promoteToMidTerm(keys: string[]): void;
    promoteToLongTerm(projectId: string): void;
  };
}
```

---

## 自动化流程

### 1. 会话启动时
1. 读取 `short-term/current_session.md` 恢复上下文
2. 加载 `mid-term/active_contexts.json` 获取活跃项目
3. 读取 `long-term/user_profile.md` 加载用户偏好

### 2. 会话进行中
1. 实时更新 `short-term/current_session.md`
2. 检测项目上下文变化，同步到中期记忆
3. 识别重要信息，提示写入长期记忆

### 3. 会话结束时
1. 总结会话要点
2. 将短期记忆移至归档区
3. 更新中期记忆中的项目状态
4. 触发长期记忆的自动提取流程

### 4. 定期维护 (每日/每周)
1. 清理过期短期记忆
2. 归档已完成的中期项目
3. 整理长期记忆，合并重复内容
4. 生成记忆摘要报告

---

## 记忆优先级与检索

### 检索优先级 (从高到低)
1. **短期记忆** - 当前会话上下文
2. **中期记忆** - 活跃项目状态
3. **长期记忆** - 用户画像和知识库
4. **归档区** - 历史记录 (按需检索)

### 记忆权重算法
```
relevance_score = (
  recency_weight * (1 / days_since_update) +
  frequency_weight * access_count +
  importance_weight * user_marked_importance
)
```

---

## 安全与隐私

1. **敏感信息标记**: 支持标记敏感内容，限制自动提取
2. **访问控制**: 长期记忆中的个人敏感信息需要显式授权
3. **数据加密**: 支持对敏感记忆进行加密存储
4. **定期审计**: 检查长期记忆中的过时或敏感信息

---

## 扩展性设计

1. **插件接口**: 支持自定义记忆处理器
2. **外部集成**: 可对接向量数据库进行语义检索
3. **多模态支持**: 预留图片、音频等非文本记忆存储
4. **分布式存储**: 支持未来扩展到云端同步

---

*文档版本: 1.0*
*创建时间: 2026-02-27*
*作者: CEO Kimi Claw 自我升级系统*
