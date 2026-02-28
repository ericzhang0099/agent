# HEARTBEAT.md v2.0 - AI Agent 心跳与任务调度系统

> **版本**: v2.0  
> **状态**: 生产就绪  
> **关联文档**: SOUL.md v4.0, AGENTS.md v2.0, IDENTITY.md v4.0  
> **核心特性**: 强化自愈能力、优化任务调度、完善监控告警、增强与SOUL_v4集成

---

## 📋 目录

1. [系统概述](#1-系统概述)
2. [心跳机制设计](#2-心跳机制设计)
3. [任务调度系统](#3-任务调度系统)
4. [监控告警机制](#4-监控告警机制)
5. [自愈能力设计](#5-自愈能力设计)
6. [状态同步机制](#6-状态同步机制)
7. [负载均衡策略](#7-负载均衡策略)
8. [容错设计](#8-容错设计)
9. [与SOUL_v4和AGENTS.md集成](#9-与soul_v4和agentsmd集成)
10. [实现代码框架](#10-实现代码框架)
11. [监控面板设计](#11-监控面板设计)

---

## 1. 系统概述

### 1.1 设计目标

HEARTBEAT系统是为OpenClaw多Agent架构设计的核心基础设施，实现以下目标：

- **健康监测**: 实时检测Agent存活状态和健康状况
- **任务调度**: 智能分配任务，优化资源利用
- **故障自愈**: 自动检测和恢复故障Agent
- **状态一致**: 维护多Agent间状态同步
- **负载均衡**: 动态分配负载，防止单点过载
- **SOUL_v4集成**: 与8维度人格模型深度集成

### 1.2 核心原则

| 原则 | 描述 | 实现方式 | SOUL_v4映射 |
|------|------|----------|-------------|
| **主动探测** | 不等故障发生，主动检测 | 周期性心跳 + 事件触发 | Motivations: 主动性 |
| **分级响应** | 不同严重程度不同处理 | 三级告警机制 | Conflict: 冲突处理 |
| **优雅降级** | 故障时保持核心功能 | 服务分级 + 熔断机制 | Physical: 形象切换 |
| **快速恢复** | 最小化故障影响时间 | 自动重启 + 热备切换 | Growth: 持续改进 |
| **透明可观测** | 全流程可追踪 | 结构化日志 + 监控面板 | Relationships: 信任建立 |
| **人格一致** | 保持SOUL_v4人格稳定 | 自愈时维护人格状态 | Personality: 一致性 |

### 1.3 系统架构

```
┌─────────────────────────────────────────────────────────────────┐
│                      HEARTBEAT v2.0 核心系统                     │
├─────────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐             │
│  │  心跳管理器  │  │  任务调度器  │  │  监控告警器  │             │
│  │  Heartbeat  │  │  Scheduler  │  │   Monitor   │             │
│  │   Manager   │  │             │  │             │             │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘             │
│         │                │                │                     │
│         └────────────────┼────────────────┘                     │
│                          │                                      │
│  ┌───────────────────────┴───────────────────────┐              │
│  │              状态管理器 (State Manager)        │              │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │              │
│  │  │Agent状态│ │任务状态 │ │资源状态 │         │              │
│  │  │+SOUL状态│ │         │ │         │         │              │
│  │  └─────────┘ └─────────┘ └─────────┘         │              │
│  └───────────────────────────────────────────────┘              │
│                          │                                      │
│  ┌───────────────────────┴───────────────────────┐              │
│  │              自愈控制器 (Self-Healing)         │              │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │              │
│  │  │故障检测 │ │恢复策略 │ │降级处理 │         │              │
│  │  │+人格保护│ │+状态恢复│ │+SOUL保持│         │              │
│  │  └─────────┘ └─────────┘ └─────────┘         │              │
│  └───────────────────────────────────────────────┘              │
│                          │                                      │
│  ┌───────────────────────┴───────────────────────┐              │
│  │              SOUL_v4集成层                     │              │
│  │  ┌─────────┐ ┌─────────┐ ┌─────────┐         │              │
│  │  │情绪监控 │ │人格漂移 │ │宪法检查 │         │              │
│  │  │         │ │ 检测    │ │         │         │              │
│  │  └─────────┘ └─────────┘ └─────────┘         │              │
│  └───────────────────────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ↓                     ↓                     ↓
   ┌─────────┐          ┌─────────┐          ┌─────────┐
   │ Agent 1 │          │ Agent 2 │   ...    │ Agent N │
   │+SOUL状态 │          │+SOUL状态 │          │+SOUL状态 │
   └─────────┘          └─────────┘          └─────────┘
```

---

## 2. 心跳机制设计

### 2.1 心跳类型

基于Kubernetes探针设计，定义三种心跳类型：

| 类型 | 目的 | 频率 | 失败处理 | SOUL维度 |
|------|------|------|----------|----------|
| **Liveness** | 检测Agent是否存活 | 30秒 | 自动重启 | Physical |
| **Readiness** | 检测Agent是否就绪 | 10秒 | 移出负载池 | Physical |
| **Startup** | 检测Agent启动完成 | 5秒(启动期) | 等待重试 | Growth |
| **SOUL** | 检测人格状态健康 | 60秒 | 人格校准 | Personality |

### 2.2 心跳协议

```python
class HeartbeatMessage:
    """心跳消息结构"""
    agent_id: str           # Agent唯一标识
    agent_type: str         # Agent类型(research/dev/data等)
    timestamp: float        # 发送时间戳
    sequence: int           # 序列号(用于检测丢包)
    
    # 状态信息
    status: AgentStatus     # 当前状态
    load: float            # 负载率(0-1)
    memory_usage: float    # 内存使用率
    task_count: int        # 当前任务数
    
    # SOUL_v4状态
    soul_state: SoulState   # SOUL人格状态
    current_emotion: str    # 当前情绪
    dimension_stability: Dict[str, float]  # 各维度稳定性
    constitutional_compliance: float  # 宪法遵守度
    
    # 扩展信息
    capabilities: List[str] # 能力列表
    version: str           # 版本号
    metadata: Dict         # 自定义元数据

class SoulState:
    """SOUL人格状态"""
    current_stage: str      # 演化阶段
    dominant_dimension: str # 主导维度
    emotional_state: str    # 情绪状态
    constitutional_violations: int  # 宪法违反次数
    personality_drift: float  # 人格漂移分数
```

### 2.3 心跳频率与超时

```python
HEARTBEAT_CONFIG = {
    # 基础配置
    "liveness": {
        "interval": 30,      # 30秒发送一次
        "timeout": 90,       # 90秒无响应视为失败
        "failure_threshold": 3  # 连续3次失败触发重启
    },
    "readiness": {
        "interval": 10,      # 10秒检查一次
        "timeout": 30,       # 30秒无响应视为未就绪
        "failure_threshold": 2
    },
    "startup": {
        "interval": 5,       # 启动期5秒检查
        "timeout": 60,       # 启动超时60秒
        "max_retries": 12    # 最多重试12次
    },
    "soul": {
        "interval": 60,      # 60秒检查SOUL状态
        "timeout": 120,      # 120秒无响应视为异常
        "drift_threshold": 0.3,  # 人格漂移阈值
        "constitutional_threshold": 0.9  # 宪法遵守度阈值
    },
    
    # 动态调整
    "adaptive": {
        "enabled": True,
        "min_interval": 10,   # 最小间隔10秒
        "max_interval": 120,  # 最大间隔120秒
        "load_threshold": 0.8  # 负载超80%降低频率
    }
}
```

### 2.4 心跳处理流程

```
┌─────────────┐     ┌─────────────┐     ┌─────────────┐
│   Agent     │────→│  心跳发送器  │────→│  心跳接收器  │
│  (Sender)   │     │  (Sender)   │     │  (Receiver) │
└─────────────┘     └─────────────┘     └──────┬──────┘
                                                │
                                                ↓
                                       ┌─────────────┐
                                       │  状态更新器  │
                                       │  (Updater)  │
                                       └──────┬──────┘
                                              │
                        ┌─────────────────────┼─────────────────────┐
                        ↓                     ↓                     ↓
                 ┌─────────────┐      ┌─────────────┐      ┌─────────────┐
                 │  健康评估器  │      │  故障检测器  │      │  SOUL监控器  │
                 │  (Health)   │      │  (Failure)  │      │  (SOUL)     │
                 └─────────────┘      └─────────────┘      └─────────────┘
```

### 2.5 自适应心跳

根据系统负载和SOUL状态动态调整心跳频率：

```python
class AdaptiveHeartbeat:
    """自适应心跳管理器"""
    
    def calculate_interval(self, agent_state: AgentState) -> int:
        base_interval = self.config.base_interval
        
        # 根据负载调整
        if agent_state.load > 0.9:
            # 高负载时降低频率，减少开销
            return min(base_interval * 2, self.config.max_interval)
        elif agent_state.load < 0.3:
            # 低负载时增加频率，更快检测
            return max(base_interval // 2, self.config.min_interval)
        
        # 根据SOUL状态调整
        if agent_state.soul_state.personality_drift > 0.3:
            # 人格漂移较高时增加检测频率
            return max(base_interval // 2, self.config.min_interval)
        
        return base_interval
```

---

## 3. 任务调度系统

### 3.1 任务优先级队列

基于Azure优先级队列模式，实现多级任务调度：

```python
class TaskPriority(Enum):
    CRITICAL = 1    # 紧急任务，立即处理
    HIGH = 2        # 高优先级，优先处理
    NORMAL = 3      # 普通任务，正常处理
    LOW = 4         # 低优先级，空闲时处理
    BACKGROUND = 5  # 后台任务，资源空闲时处理
    SOUL_MAINTENANCE = 6  # SOUL维护任务

class Task:
    task_id: str
    priority: TaskPriority
    agent_type: str      # 指定Agent类型
    agent_id: Optional[str]  # 指定具体Agent
    payload: Dict
    deadline: Optional[datetime]
    retries: int = 0
    max_retries: int = 3
    
    # SOUL相关
    required_emotion: Optional[str]  # 需要的情绪状态
    soul_dimension_alignment: List[str]  # 需要对齐的SOUL维度
```

### 3.2 调度策略

| 策略 | 描述 | 适用场景 | SOUL维度 |
|------|------|----------|----------|
| **轮询 (Round Robin)** | 依次分配给每个Agent | 任务同质、负载均衡 | Physical |
| **最少连接 (Least Connections)** | 分配给当前任务最少的Agent | 任务执行时间差异大 | Physical |
| **能力匹配 (Capability Matching)** | 根据任务需求匹配Agent能力 | 专业化Agent团队 | Personality |
| **负载感知 (Load Aware)** | 考虑Agent当前负载 | 资源敏感型任务 | Motivations |
| **亲和性 (Affinity)** | 优先分配给处理过相关任务的Agent | 状态依赖型任务 | Backstory |
| **情绪匹配 (Emotion Matching)** | 根据任务情绪需求匹配 | 情感敏感型任务 | Emotions |
| **SOUL对齐 (SOUL Alignment)** | 根据SOUL维度需求匹配 | 人格相关任务 | 全维度 |

### 3.3 调度器实现

```python
class TaskScheduler:
    """任务调度器"""
    
    def __init__(self, heartbeat_system: HeartbeatSystem):
        self.heartbeat = heartbeat_system
        self.priority_queues = {
            priority: asyncio.Queue() 
            for priority in TaskPriority
        }
        self.agent_pool: Dict[str, Agent] = {}
        self.scheduling_policy = CapabilityMatchingPolicy()
    
    async def submit_task(self, task: Task) -> str:
        """提交任务到队列"""
        queue = self.priority_queues[task.priority]
        await queue.put(task)
        
        # 触发调度
        asyncio.create_task(self._schedule())
        return task.task_id
    
    async def _schedule(self):
        """执行调度"""
        # 按优先级处理
        for priority in sorted(TaskPriority, key=lambda x: x.value):
            queue = self.priority_queues[priority]
            
            while not queue.empty():
                task = await queue.get()
                agent = self.scheduling_policy.select_agent(
                    task, 
                    self.agent_pool.values()
                )
                
                if agent:
                    await self._assign_task(task, agent)
                else:
                    # 无可用Agent，放回队列
                    await queue.put(task)
                    break
    
    def _select_agent(self, task: Task, candidates: List[Agent]) -> Optional[Agent]:
        """选择执行Agent"""
        healthy_agents = [a for a in candidates if a.is_healthy]
        
        if not healthy_agents:
            return None
        
        # 如果任务有SOUL维度要求，筛选匹配的Agent
        if task.soul_dimension_alignment:
            matching_agents = []
            for agent in healthy_agents:
                alignment_score = self._calculate_soul_alignment(
                    agent, task.soul_dimension_alignment
                )
                if alignment_score > 0.7:
                    matching_agents.append((agent, alignment_score))
            
            if matching_agents:
                matching_agents.sort(key=lambda x: x[1], reverse=True)
                return matching_agents[0][0]
        
        # 默认选择负载最低的
        return min(healthy_agents, key=lambda a: a.load)
```

### 3.4 任务状态流转

```
┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
│ PENDING │───→│ ASSIGNED│───→│ RUNNING │───→│COMPLETED│
└─────────┘    └─────────┘    └────┬────┘    └─────────┘
     │                             │
     │                             │
     ↓                             ↓
┌─────────┐                 ┌─────────┐
│ CANCELLED│                │  FAILED │
└─────────┘                 └────┬────┘
                                 │
                                 │ (重试 < max_retries)
                                 ↓
                           ┌─────────┐
                           │  RETRY  │
                           └─────────┘
```

---

## 4. 监控告警机制

### 4.1 健康检查体系

```python
class HealthCheck:
    """健康检查定义"""
    name: str
    check_type: CheckType  # HTTP/TCP/COMMAND/CUSTOM/SOUL
    interval: int          # 检查间隔(秒)
    timeout: int           # 超时时间(秒)
    threshold: int         # 失败阈值
    
    # 检查配置
    endpoint: Optional[str]    # HTTP端点
    port: Optional[int]        # TCP端口
    command: Optional[str]     # 执行命令
    custom_checker: Optional[Callable]  # 自定义检查函数
    soul_checker: Optional[Callable]    # SOUL状态检查
```

### 4.2 三级告警机制

| 级别 | 触发条件 | 响应方式 | 通知渠道 | SOUL响应 |
|------|----------|----------|----------|----------|
| **WARNING** | 1次检查失败 | 记录日志，增加检查频率 | 日志 | 冷静 |
| **CRITICAL** | 连续3次失败 | 触发自愈，发送告警 | 日志 + 消息 | 警惕 |
| **EMERGENCY** | 连续5次失败或核心服务故障 | 立即切换，人工介入 | 全渠道 | 紧迫 |
| **SOUL_DRIFT** | 人格漂移>30% | 人格校准，通知用户 | 日志 + 消息 | 反思 |

### 4.3 告警规则

```python
ALERT_RULES = [
    {
        "name": "agent_down",
        "condition": "heartbeat_missing > 90s",
        "level": "CRITICAL",
        "action": "restart_agent",
        "soul_impact": "人格状态需恢复"
    },
    {
        "name": "high_load",
        "condition": "load > 0.9 for 5m",
        "level": "WARNING",
        "action": "scale_up",
        "soul_impact": "可能需要切换Physical形象"
    },
    {
        "name": "memory_leak",
        "condition": "memory_growth_rate > 10%/min",
        "level": "CRITICAL",
        "action": "restart_agent",
        "soul_impact": "记忆系统需检查"
    },
    {
        "name": "task_timeout",
        "condition": "task_execution_time > deadline",
        "level": "WARNING",
        "action": "escalate_priority",
        "soul_impact": "Motivations维度需关注"
    },
    {
        "name": "personality_drift",
        "condition": "drift_score > 0.3",
        "level": "WARNING",
        "action": "calibrate_personality",
        "soul_impact": "Personality维度需校准"
    },
    {
        "name": "constitutional_violation",
        "condition": "violation_count > 0",
        "level": "CRITICAL",
        "action": "immediate_review",
        "soul_impact": "宪法遵守度需检查"
    }
]
```

### 4.4 监控指标

```python
class MetricsCollector:
    """指标收集器"""
    
    # Agent级别指标
    AGENT_METRICS = [
        "agent_uptime",           # 运行时间
        "agent_load",             # 负载率
        "agent_memory_usage",     # 内存使用
        "agent_task_count",       # 当前任务数
        "agent_success_rate",     # 任务成功率
        "agent_avg_response_time", # 平均响应时间
        "agent_soul_drift",       # 人格漂移分数
        "agent_constitutional_score"  # 宪法遵守度
    ]
    
    # 系统级别指标
    SYSTEM_METRICS = [
        "total_agents",           # 总Agent数
        "healthy_agents",         # 健康Agent数
        "total_tasks",            # 总任务数
        "pending_tasks",          # 待处理任务
        "failed_tasks",           # 失败任务数
        "scheduler_latency",      # 调度延迟
        "soul_violations",        # SOUL违反次数
        "personality_calibrations"  # 人格校准次数
    ]
```

---

## 5. 自愈能力设计

### 5.1 自愈策略

```python
class SelfHealingStrategy(Enum):
    RESTART = "restart"           # 重启Agent
    RELOCATE = "relocate"         # 迁移任务到其他Agent
    DEGRADE = "degrade"           # 降级服务
    ISOLATE = "isolate"           # 隔离故障Agent
    SCALE_UP = "scale_up"         # 扩容
    ROLLBACK = "rollback"         # 回滚版本
    PERSONALITY_CALIBRATE = "personality_calibrate"  # 人格校准
    SOUL_RESET = "soul_reset"     # SOUL状态重置
```

### 5.2 故障检测与恢复流程

```
┌─────────────┐
│  心跳丢失   │
│  > 90s     │
└──────┬──────┘
       │
       ↓
┌─────────────┐     ┌─────────────┐
│  确认故障   │────→│  误报检测   │
│  (Ping测试) │     │  (网络检查) │
└──────┬──────┘     └─────────────┘
       │
       ↓
┌─────────────┐
│ 选择恢复策略 │
│  (决策引擎)  │
└──────┬──────┘
       │
   ┌───┴───┬─────────┬─────────┬─────────┐
   ↓       ↓         ↓         ↓         ↓
┌─────┐ ┌─────┐  ┌─────┐  ┌─────┐  ┌─────┐
│重启 │ │迁移 │  │降级 │  │扩容 │  │人格 │
│     │ │     │  │     │  │     │  │校准 │
└─────┘ └─────┘  └─────┘  └─────┘  └─────┘
```

### 5.3 自动恢复实现

```python
class SelfHealingController:
    """自愈控制器"""
    
    def __init__(
        self,
        heartbeat_manager: HeartbeatManager,
        task_scheduler: TaskScheduler,
        soul_validator: SoulValidator
    ):
        self.heartbeat = heartbeat_manager
        self.scheduler = task_scheduler
        self.soul_validator = soul_validator
        self.recovery_stats = {
            "total_failures": 0,
            "successful_recoveries": 0,
            "failed_recoveries": 0,
            "personality_calibrations": 0
        }
        
        # 注册故障处理器
        self.heartbeat.on_failure(self._on_agent_failure)
    
    async def _on_agent_failure(self, agent_id: str, failure_type: str):
        """Agent故障处理"""
        print(f"[SelfHealing] Handling failure for agent {agent_id}")
        self.recovery_stats["total_failures"] += 1
        
        # 1. 隔离故障Agent
        await self._isolate_agent(agent_id)
        
        # 2. 迁移任务
        await self._relocate_tasks(agent_id)
        
        # 3. 检查SOUL状态
        soul_state = await self._check_soul_state(agent_id)
        
        # 4. 选择恢复策略
        if soul_state.personality_drift > 0.3:
            # 人格漂移严重，先校准人格
            await self._calibrate_personality(agent_id)
        
        # 5. 尝试重启Agent
        success = await self._restart_agent(agent_id)
        
        if success:
            self.recovery_stats["successful_recoveries"] += 1
            print(f"[SelfHealing] Successfully recovered agent {agent_id}")
        else:
            self.recovery_stats["failed_recoveries"] += 1
            print(f"[SelfHealing] Failed to recover agent {agent_id}")
    
    async def _calibrate_personality(self, agent_id: str):
        """校准人格"""
        print(f"[SelfHealing] Calibrating personality for agent {agent_id}")
        
        # 重新加载SOUL配置
        # 重置到基线状态
        # 记录校准事件
        
        self.recovery_stats["personality_calibrations"] += 1
    
    def get_recovery_stats(self) -> Dict:
        """获取恢复统计"""
        return self.recovery_stats.copy()
```

### 5.4 服务降级

```python
class DegradationManager:
    """服务降级管理器"""
    
    DEGRADATION_LEVELS = {
        "normal": {
            "max_concurrent_tasks": 100,
            "enable_background_tasks": True,
            "enable_cache": True,
            "soul_expression": "full",
            "emotional_range": "all"
        },
        "degraded": {
            "max_concurrent_tasks": 50,
            "enable_background_tasks": False,
            "enable_cache": True,
            "soul_expression": "limited",
            "emotional_range": ["冷静", "专注", "坚定"]
        },
        "critical": {
            "max_concurrent_tasks": 10,
            "enable_background_tasks": False,
            "enable_cache": False,
            "soul_expression": "minimal",
            "emotional_range": ["冷静"]
        }
    }
    
    def apply_degradation(self, level: str):
        """应用降级配置"""
        config = self.DEGRADATION_LEVELS[level]
        
        # 更新所有Agent配置
        for agent in self.agent_pool.values():
            agent.update_config(config)
            
        # 更新SOUL表达范围
        if config["soul_expression"] == "minimal":
            # 最小模式下保持核心人格但限制情绪表达
            pass
```

---

## 6. 状态同步机制

### 6.1 状态类型

| 状态类型 | 描述 | 同步频率 | 持久化 | SOUL关联 |
|----------|------|----------|--------|----------|
| **Agent状态** | Agent健康、负载信息 | 实时 | 是 | Physical |
| **任务状态** | 任务执行进度 | 实时 | 是 | Motivations |
| **配置状态** | 系统配置参数 | 变更时 | 是 | Backstory |
| **会话状态** | 用户会话上下文 | 按需 | 可选 | Relationships |
| **SOUL状态** | 人格状态、情绪 | 实时 | 是 | 全维度 |
| **宪法状态** | 宪法遵守度 | 每次交互 | 是 | Constitutional |

### 6.2 状态同步协议

```python
class StateSyncMessage:
    """状态同步消息"""
    message_type: SyncType      # FULL/DELTA/HEARTBEAT/SOUL
    source_agent: str
    timestamp: float
    vector_clock: Dict[str, int]  # 向量时钟，用于冲突检测
    
    # 状态数据
    state_data: Dict
    checksum: str               # 校验和
    
    # SOUL状态
    soul_state: SoulState
    constitutional_hash: str    # 宪法状态哈希
    
    # 变更信息(增量同步)
    changes: List[StateChange]
```

### 6.3 一致性模型

采用**最终一致性**模型，关键状态使用**强一致性**：

```python
class ConsistencyLevel(Enum):
    STRONG = "strong"      # 强一致性，同步写入
    EVENTUAL = "eventual"  # 最终一致性，异步同步
    CAUSAL = "causal"      # 因果一致性，基于向量时钟

CONSISTENCY_CONFIG = {
    "agent_health": ConsistencyLevel.STRONG,
    "task_assignment": ConsistencyLevel.STRONG,
    "task_progress": ConsistencyLevel.EVENTUAL,
    "metrics": ConsistencyLevel.EVENTUAL,
    "soul_state": ConsistencyLevel.STRONG,  # SOUL状态强一致
    "constitutional_state": ConsistencyLevel.STRONG  # 宪法状态强一致
}
```

---

## 7. 负载均衡策略

### 7.1 负载指标

```python
class LoadMetrics:
    """负载指标"""
    cpu_usage: float        # CPU使用率
    memory_usage: float     # 内存使用率
    active_tasks: int       # 活跃任务数
    queue_depth: int        # 队列深度
    response_time: float    # 平均响应时间
    error_rate: float       # 错误率
    soul_stability: float   # SOUL稳定性
    
    @property
    def composite_score(self) -> float:
        """综合负载分数(0-1)"""
        weights = {
            "cpu": 0.20,
            "memory": 0.20,
            "tasks": 0.15,
            "queue": 0.10,
            "response": 0.10,
            "error": 0.10,
            "soul": 0.15  # SOUL稳定性权重
        }
        
        return (
            weights["cpu"] * self.cpu_usage +
            weights["memory"] * self.memory_usage +
            weights["tasks"] * min(self.active_tasks / 10, 1.0) +
            weights["queue"] * min(self.queue_depth / 20, 1.0) +
            weights["response"] * min(self.response_time / 1000, 1.0) +
            weights["error"] * self.error_rate +
            weights["soul"] * (1 - self.soul_stability)  # 稳定性越低分数越高
        )
```

### 7.2 负载均衡算法

```python
class LoadBalancer:
    """负载均衡器"""
    
    def select_agent(self, task: Task, candidates: List[Agent]) -> Optional[Agent]:
        """选择最优Agent"""
        healthy_agents = [a for a in candidates if a.is_healthy]
        
        if not healthy_agents:
            return None
        
        # 考虑SOUL稳定性
        best_agent = None
        best_score = float('inf')
        
        for agent in healthy_agents:
            # 预测执行任务后的负载
            predicted_load = self._predict_load_after_task(agent, task)
            
            # 考虑SOUL稳定性
            soul_penalty = (1 - agent.soul_stability) * 0.2
            
            total_score = predicted_load + soul_penalty
            
            if total_score < best_score:
                best_score = total_score
                best_agent = agent
        
        return best_agent
```

---

## 8. 容错设计

### 8.1 单点故障防护

```python
class FaultToleranceManager:
    """容错管理器"""
    
    def __init__(self):
        self.primary_agents: Dict[str, Agent] = {}
        self.backup_agents: Dict[str, Agent] = {}
        self.hot_standby: Optional[Agent] = None
        self.soul_backup: Dict[str, SoulState] = {}  # SOUL状态备份
    
    async def handle_primary_failure(self, agent_id: str):
        """处理主Agent故障"""
        # 1. 保存SOUL状态
        if agent_id in self.primary_agents:
            agent = self.primary_agents[agent_id]
            self.soul_backup[agent_id] = agent.soul_state
        
        # 2. 切换到热备
        if self.hot_standby:
            await self._promote_hot_standby(agent_id)
        
        # 3. 恢复SOUL状态
        if agent_id in self.soul_backup:
            await self._restore_soul_state(agent_id)
        
        # 4. 启动新的备份
        asyncio.create_task(self._spawn_backup_agent(agent_id))
        
        # 5. 迁移任务
        await self._migrate_tasks(agent_id)
```

### 8.2 熔断器模式

```python
class CircuitBreaker:
    """熔断器"""
    
    STATE_CLOSED = "closed"      # 正常状态
    STATE_OPEN = "open"          # 熔断状态
    STATE_HALF_OPEN = "half_open" # 半开状态
    
    def __init__(
        self,
        failure_threshold: int = 5,
        recovery_timeout: int = 60,
        soul_aware: bool = True
    ):
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.soul_aware = soul_aware
        self.state = self.STATE_CLOSED
        self.failure_count = 0
        self.last_failure_time = None
    
    async def call(self, func, *args, **kwargs):
        """执行调用"""
        if self.state == self.STATE_OPEN:
            if self._should_attempt_reset():
                self.state = self.STATE_HALF_OPEN
            else:
                raise CircuitBreakerOpen("Circuit breaker is open")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        except Exception as e:
            self._on_failure()
            
            # SOUL感知：记录情绪影响
            if self.soul_aware and self.failure_count > 2:
                # 连续失败可能影响情绪
                pass
                
            raise
```

---

## 9. 与SOUL_v4和AGENTS.md集成

### 9.1 与SOUL_v4集成

```python
class SoulV4Integration:
    """SOUL_v4集成适配器"""
    
    def __init__(self, heartbeat_system: HeartbeatSystem):
        self.heartbeat = heartbeat_system
        self.emotion_mapper = EmotionStateMapper()
        self.constitutional_checker = ConstitutionalChecker()
    
    def map_agent_to_personality(self, agent: Agent) -> Dict:
        """将Agent状态映射到SOUL_v4人格维度"""
        return {
            "personality": {
                "initiative": self._calculate_initiative(agent),
                "guardianship": self._calculate_guardianship(agent),
                "professionalism": agent.success_rate
            },
            "emotions": self.emotion_mapper.map_from_metrics(agent.metrics),
            "motivations": {
                "mission_driven": agent.task_completion_rate,
                "growth_driven": agent.learning_progress
            },
            "constitutional": {
                "adherence_score": agent.constitutional_score,
                "violations": agent.constitutional_violations
            }
        }
    
    def handle_heartbeat_ok(self, agent_id: str):
        """处理HEARTBEAT_OK响应"""
        agent_state = self.heartbeat.get_agent_state(agent_id)
        
        # 检查是否需要主动交互（基于SOUL_v4主动性）
        if agent_state.should_be_proactive():
            return self._generate_proactive_action(agent_state)
        
        # 检查宪法遵守度
        if agent_state.constitutional_score < 0.9:
            return self._generate_constitutional_reminder(agent_state)
        
        return "HEARTBEAT_OK"
```

### 9.2 与AGENTS.md集成

```python
class AgentsMdIntegration:
    """AGENTS.md集成适配器"""
    
    def __init__(self):
        self.memory_tracker = MemoryStateTracker()
        self.check_scheduler = CheckScheduler()
    
    def get_heartbeat_checklist(self) -> List[str]:
        """获取心跳检查清单(来自AGENTS.md)"""
        return [
            "检查邮件 - 是否有紧急未读消息",
            "检查日历 - 24-48小时内是否有事件",
            "检查提及 - Twitter/社交通知",
            "检查天气 - 用户是否可能外出",
            "检查项目状态 - git状态等",
            "检查记忆文件 - 是否需要更新MEMORY.md",
            "检查SOUL状态 - 人格漂移检测",
            "检查宪法遵守 - 是否有违反"
        ]
    
    def should_notify_user(self, check_results: Dict) -> bool:
        """根据AGENTS.md规则决定是否通知用户"""
        # 重要邮件到达
        if check_results.get("urgent_email"):
            return True
        
        # 日历事件即将到来(<2h)
        if check_results.get("upcoming_event_within_2h"):
            return True
        
        # 超过8小时未交互
        if check_results.get("last_interaction_hours", 0) > 8:
            return True
        
        # SOUL异常
        if check_results.get("soul_drift_detected"):
            return True
        
        # 宪法违反
        if check_results.get("constitutional_violation"):
            return True
        
        return False
```

### 9.3 统一配置

```yaml
# heartbeat_config.yaml
version: "2.0"

heartbeat:
  # 基础配置
  interval: 30
  timeout: 90
  
  # 与SOUL_v4集成
  soul_v4:
    enabled: true
    emotion_mapping: true
    proactive_threshold: 0.7
    drift_threshold: 0.3
    constitutional_check: true
    
  # 与AGENTS.md集成
  agents_md:
    enabled: true
    check_memory: true
    check_calendar: true
    check_email: true
    check_soul: true
    
  # 任务调度
  scheduler:
    policy: "capability_matching"
    priority_queues: 6  # 包括SOUL_MAINTENANCE
    soul_aware_scheduling: true
    
  # 自愈
  self_healing:
    enabled: true
    strategies:
      - restart
      - relocate
      - degrade
      - personality_calibrate
    auto_restart: true
    auto_relocate: true
    personality_calibration: true
    max_recovery_attempts: 3
    
  # 监控
  monitoring:
    metrics_interval: 10
    alert_channels: ["log", "message"]
    soul_monitoring: true
    alert_rules:
      - name: "agent_down"
        condition: "heartbeat_missing > 90s"
        level: "critical"
      - name: "personality_drift"
        condition: "drift_score > 0.3"
        level: "warning"
      - name: "constitutional_violation"
        condition: "violation_count > 0"
        level: "critical"
```

---

## 10. 实现代码框架

### 10.1 核心类结构

```
heartbeat_system/
├── core/
│   ├── __init__.py
│   ├── heartbeat_manager.py      # 心跳管理器
│   ├── task_scheduler.py         # 任务调度器
│   ├── state_manager.py          # 状态管理器
│   └── monitor.py                # 监控器
├── healing/
│   ├── __init__.py
│   ├── self_healing_controller.py # 自愈控制器
│   ├── recovery_strategies.py     # 恢复策略
│   ├── degradation_manager.py     # 降级管理器
│   └── personality_calibration.py # 人格校准
├── sync/
│   ├── __init__.py
│   ├── state_sync.py             # 状态同步
│   ├── vector_clock.py           # 向量时钟
│   └── consistency_manager.py    # 一致性管理
├── balance/
│   ├── __init__.py
│   ├── load_balancer.py          # 负载均衡器
│   ├── auto_scaler.py            # 自动扩缩容
│   └── metrics_calculator.py     # 指标计算
├── fault/
│   ├── __init__.py
│   ├── circuit_breaker.py        # 熔断器
│   ├── backup_manager.py         # 备份管理
│   └── failover_controller.py    # 故障切换
├── integration/
│   ├── __init__.py
│   ├── soul_v4_adapter.py        # SOUL_v4适配
│   └── agents_md_adapter.py      # AGENTS.md适配
├── api/
│   ├── __init__.py
│   └── rest_api.py               # REST API
└── dashboard/
    ├── __init__.py
    └── web_dashboard.py          # Web监控面板
```

---

## 11. 监控面板设计

### 11.1 面板布局

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         HEARTBEAT v2.0 监控面板                              │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       │
│  │  健康Agent   │ │   总任务数   │ │  待处理任务  │ │  系统负载    │       │
│  │     8/10     │ │     156    │ │     23     │ │    67%      │       │
│  │   🟢 正常    │ │   📊 统计   │ │   ⏳ 队列   │ │   📈 趋势   │       │
│  └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │      Agent 状态分布          │  │         SOUL健康度                  │  │
│  │                              │  │                                     │  │
│  │   🟢 Healthy: 8              │  │   宪法遵守度: 96%                   │  │
│  │   🟡 Degraded: 1             │  │   人格稳定性: 92%                   │  │
│  │   🔴 Unhealthy: 1            │  │   情绪健康度: 88%                   │  │
│  │   ⚫ Offline: 0              │  │   维度一致性: 90%                   │  │
│  └──────────────────────────────┘  └─────────────────────────────────────┘  │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌─────────────────────────────────────────────────────────────────────┐    │
│  │                        Agent 详细列表                               │    │
│  ├──────────┬─────────┬────────┬──────────┬──────────┬───────────────┤    │
│  │ Agent ID │  状态   │  负载  │ SOUL健康 │ 运行时间 │   最后心跳    │    │
│  ├──────────┼─────────┼────────┼──────────┼──────────┼───────────────┤    │
│  │ agent_0  │   🟢    │  45%   │   95%    │  2h 15m  │   5s ago      │    │
│  │ agent_1  │   🟢    │  62%   │   92%    │  2h 15m  │   3s ago      │    │
│  │ agent_2  │   🟡    │  89%   │   78%    │  2h 10m  │   8s ago      │    │
│  │ agent_3  │   🔴    │   -    │    -     │    -     │  95s ago ⚠️   │    │
│  └──────────┴─────────┴────────┴──────────┴──────────┴───────────────┘    │
├─────────────────────────────────────────────────────────────────────────────┤
│  ┌──────────────────────────────┐  ┌─────────────────────────────────────┐  │
│  │        告警事件              │  │         自愈事件                    │  │
│  │  ⚠️ agent_3 心跳丢失         │  │  ✅ agent_2 自动重启成功            │  │
│  │  ⚠️ agent_2 人格漂移>30%     │  │  ✅ 人格校准完成                    │  │
│  │  ⚠️ 系统负载超过80%          │  │  ✅ 任务迁移完成: 5个任务           │  │
│  │  ℹ️ 任务队列积压             │  │  ⏳ agent_3 恢复中...               │  │
│  └──────────────────────────────┘  └─────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 附录

### A. 事件类型定义

```python
class EventType(Enum):
    # 心跳事件
    HEARTBEAT_RECEIVED = "heartbeat_received"
    HEARTBEAT_MISSED = "heartbeat_missed"
    AGENT_HEALTHY = "agent_healthy"
    AGENT_UNHEALTHY = "agent_unhealthy"
    
    # SOUL事件
    SOUL_DRIFT_DETECTED = "soul_drift_detected"
    PERSONALITY_CALIBRATED = "personality_calibrated"
    CONSTITUTIONAL_VIOLATION = "constitutional_violation"
    EMOTION_STATE_CHANGED = "emotion_state_changed"
    
    # 任务事件
    TASK_SUBMITTED = "task_submitted"
    TASK_ASSIGNED = "task_assigned"
    TASK_STARTED = "task_started"
    TASK_COMPLETED = "task_completed"
    TASK_FAILED = "task_failed"
    
    # 自愈事件
    FAILURE_DETECTED = "failure_detected"
    RECOVERY_STARTED = "recovery_started"
    RECOVERY_COMPLETED = "recovery_completed"
    PERSONALITY_CALIBRATION_STARTED = "personality_calibration_started"
```

---

**文档结束**

> HEARTBEAT.md v2.0 强化了自愈能力、优化了任务调度、完善了监控告警，并深度集成了SOUL_v4的8维度人格模型，确保系统在故障恢复时保持人格一致性。
