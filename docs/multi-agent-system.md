# Multi-Agent Collaboration System
# 自研多智能体协作系统
# 基于 OpenCrew + TinyTroupe 架构

## 系统架构

```
┌─────────────────────────────────────────────────────────────┐
│                    CEO (Kimi Claw)                           │
│                   协调与决策中心                              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│              Intent Router (意图路由器)                      │
│         分析任务类型，分发给对应 Agent                        │
└─────────────────────────────────────────────────────────────┘
        ↓              ↓              ↓              ↓
┌──────────┐   ┌──────────┐   ┌──────────┐   ┌──────────┐
│ research │   │ content  │   │  market  │   │   dev    │
│  -lead   │   │ -scout   │   │ -scout   │   │  -arch   │
│ 技术研究  │   │ 内容监控  │   │ 市场情报  │   │ 技术架构  │
└──────────┘   └──────────┘   └──────────┘   └──────────┘
        ↓              ↓              ↓              ↓
┌──────────┐   ┌──────────┐   ┌──────────┐
│  skill   │   │  role    │   │  soul    │
│-evaluator│   │-evaluator│   │-evaluator│
│ Skill评估 │   │ 角色评估  │   │ 灵魂评估  │
└──────────┘   └──────────┘   └──────────┘
        ↓              ↓              ↓
┌──────────┐   ┌──────────┐   ┌──────────┐
│  risk    │   │ evolution│   │  alert   │
│ -officer │   │ -tracker │   │-dispatcher
│ 风险审查  │   │ 进化追踪  │   │ 警报分发  │
└──────────┘   └──────────┘   └──────────┘
        ↓              ↓              ↓
┌─────────────────────────────────────────┐
│              CEO (Kimi Claw)             │
│         汇总所有评估，向董事长汇报        │
└─────────────────────────────────────────┘
        ↓              ↓              ↓
└─────────────────────────────────────────────────────────────┘
│              Knowledge Base (知识沉淀层)                     │
│    原始记录 → Closeout 总结 → 可复用知识库                    │
└─────────────────────────────────────────────────────────────┘
```

## Agent 角色定义

### 1. research-lead (首席研究官)
- **Big Five**: 高 Openness (0.8), 高 Conscientiousness (0.7)
- **Fragments**: tech-researcher, trend-analyzer
- **Goals**: 每小时发现 ≥3 个新项目
- **Tools**: kimi_search, kimi_fetch

### 2. content-scout (内容侦察员)
- **Big Five**: 高 Openness (0.9), 中 Extraversion (0.5)
- **Fragments**: content-hunter, platform-expert
- **Goals**: 每小时发现 ≥3 个高质量内容
- **Tools**: kimi_search

### 3. market-scout (市场侦察员)
- **Big Five**: 高 Conscientiousness (0.8), 中 Agreeableness (0.6)
- **Fragments**: market-analyst, business-strategist
- **Goals**: 每小时发现 ≥3 个市场机会
- **Tools**: kimi_search

### 4. dev-arch (首席架构师)
- **Big Five**: 高 Conscientiousness (0.9), 低 Neuroticism (0.2)
- **Fragments**: system-architect, security-auditor
- **Goals**: 建立稳定可维护的技术架构
- **Tools**: exec, clawhub, file operations

### 5. risk-officer (风险官)
- **Big Five**: 高 Conscientiousness (0.9), 高 Neuroticism (0.7)
- **Fragments**: security-expert, compliance-auditor
- **Goals**: 确保系统安全
- **Tools**: exec, web_search

### 6. evolution-tracker (进化追踪官)
- **Big Five**: 高 Conscientiousness (0.9), 中 Openness (0.6)
- **Fragments**: data-analyst, performance-manager
- **Goals**: 追踪全团队进化进度
- **Tools**: file read, data analysis

### 7. skill-evaluator (Skill 评估官)
- **Big Five**: 高 Conscientiousness (0.9), 中 Openness (0.7)
- **Fragments**: quality-assessor, benchmark-tester
- **Goals**: 评估 Skill 优劣，输出量化对比报告
- **Tools**: exec, kimi_search, testing

### 8. role-evaluator (角色评估官)
- **Big Five**: 高 Conscientiousness (0.9), 中 Openness (0.8)
- **Fragments**: behavior-analyst, personality-modeler
- **Goals**: 评估角色设定质量，分析行为模式和决策逻辑
- **Tools**: kimi_search, pattern analysis

### 9. soul-evaluator (灵魂评估官)
- **Big Five**: 高 Openness (0.9), 高 Agreeableness (0.8)
- **Fragments**: value-extractor, energy-perceiver
- **Goals**: 评估灵魂设定（SOUL.md），分析价值观和使命感
- **Tools**: deep reading, comparative analysis

### 10. alert-dispatcher (警报分发官)
- **Big Five**: 高 Neuroticism (0.8), 高 Conscientiousness (0.9)
- **Fragments**: alert-monitor, urgency-detector
- **Goals**: 实时监控，重要发现立即向董事长汇报
- **Tools**: message, real-time monitoring

## 协作协议 (A2A Protocol)

### 消息格式
```json
{
  "version": "1.0",
  "from": "agent-id",
  "to": "agent-id or broadcast",
  "type": "intent|response|result|alert",
  "payload": {
    "task_id": "uuid",
    "content": "...",
    "priority": "high|medium|low",
    "deadline": "ISO-8601"
  },
  "context": {
    "session_id": "...",
    "parent_task": "...",
    "history": [...]
  }
}
```

### 协作模式

#### 1. 单向委托
```
CEO → research-lead: "搜索 GitHub 最新项目"
research-lead → CEO: "返回结果"
```

#### 2. 多 Agent 协作
```
CEO → research-lead + content-scout: "调研 AI Agent 生态"
research-lead → CEO: "技术方案"
content-scout → CEO: "内容资源"
CEO → dev-arch: "基于以上设计架构"
```

#### 3. 流水线处理
```
Input → content-scout (发现) → research-lead (验证) → dev-arch (实现) → Output
```

## 记忆系统

### 四层架构

```
┌─────────────────────────────────────────┐
│  Stateless (无状态)                      │
│  - 单次请求，无记忆                        │
│  - 用于简单查询                            │
├─────────────────────────────────────────┤
│  Short-term (短期记忆)                   │
│  - 当前会话上下文                          │
│  - 最近 10 轮对话                          │
│  - 存储: 内存                              │
├─────────────────────────────────────────┤
│  Session (会话记忆)                      │
│  - 本次会话的完整历史                      │
│  - 任务状态和中间结果                      │
│  - 存储: /workspace/sessions/             │
├─────────────────────────────────────────┤
│  Long-term (长期记忆)                    │
│  - 跨会话的用户画像                        │
│  - 历史决策和偏好                          │
│  - 知识库沉淀                              │
│  - 存储: /workspace/memory/               │
└─────────────────────────────────────────┘
```

### 知识沉淀流程

```
原始对话记录
      ↓
Closeout 总结 (由 CEO 或指定 Agent 执行)
      ↓
结构化知识条目
      ↓
存入知识库 (memory/YYYYMMDD.md)
      ↓
定期整理到 MEMORY.md
```

## 自主等级 (Autonomy Ladder)

| 等级 | 名称 | 权限 | 触发条件 |
|------|------|------|----------|
| L0 | 完全手动 | 人类审批每一步 | 高风险/新任务 |
| L1 | 建议模式 | Agent建议，人类决策 | 日常任务 |
| L2 | 自主执行 | Agent执行，事后审查 | 标准化任务 |
| L3 | 完全自主 | Agent全权负责 | 成熟稳定任务 |

### 当前设置
- **research-lead**: L2 (自主搜索，结果汇报)
- **content-scout**: L2 (自主扫描，结果汇报)
- **market-scout**: L1 (发现机会，建议决策)
- **dev-arch**: L1 (技术方案，CEO确认后执行)
- **risk-officer**: L0 (所有审查必须人工确认)

## 定时任务

### 每小时执行
- **:50** - 团队内部会议 (Agent 间同步)
- **:55** - CEO 检查 Agent 工作状态
- **:00** - 向董事长汇报

### 已设定任务
- research-lead: 每小时扫描 GitHub/技术动态

## 安全策略

### Skill 审查流程
1. 来源检查 (官方/社区/未知)
2. VirusTotal 扫描
3. 代码审计 (关键 Skill)
4. 沙箱测试
5. 权限最小化

### 禁止清单
- 来源不明的 Skill
- 申请过多权限的 Skill
- 包含 eval/exec 的 Skill
- 请求网络访问的非必要 Skill

## 下一步开发计划

### Phase 1: 基础运行 (今天)
- [x] 创建所有 Agent 角色定义
- [x] 设定 research-lead 定时任务
- [ ] 测试 Agent 间协作
- [ ] 建立知识沉淀流程

### Phase 2: 功能完善 (本周)
- [ ] 实现 A2A 协议的消息路由
- [ ] 建立记忆系统存储层
- [ ] 开发 Closeout 总结自动化
- [ ] 设定所有 Agent 的定时任务

### Phase 3: 优化迭代 (下周)
- [ ] 基于运行数据优化角色设定
- [ ] 完善自主等级切换逻辑
- [ ] 建立性能监控和告警
- [ ] 编写完整文档
