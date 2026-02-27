# Agent工作流系统设计文档

## 1. 概述

本文档设计了一套完整的Agent工作流系统，支持6种核心工作流模式，适用于10人Agent团队的协作场景。

## 2. 六种工作流模式详解

### 2.1 顺序链式 (Sequential Chain)
**定义**: 任务按线性顺序依次执行，每个Agent的输出作为下一个Agent的输入。

**适用场景**: 流水线处理、文档审核、数据清洗

**示例**: 内容创作流程
```
ResearchAgent → WritingAgent → ReviewAgent → PublishAgent
```

### 2.2 路由式 (Router)
**定义**: 根据任务特征智能路由到不同的处理分支。

**适用场景**: 客服分类、任务分发、智能调度

**示例**: 技术支持路由
```
                    ┌→ TechnicalAgent (技术问题)
Input → RouterAgent ┼→ SalesAgent (销售咨询)
                    └→ SupportAgent (一般问题)
```

### 2.3 评估优化式 (Evaluator-Optimizer)
**定义**: 迭代优化循环，评估器提供反馈，生成器持续改进。

**适用场景**: 代码生成、内容优化、方案改进

**示例**: 代码生成优化
```
CodeGeneratorAgent → CodeEvaluatorAgent → [达标?] → 输出
                          ↓ 不达标
                    反馈优化 ← 重新生成
```

### 2.4 并行式 (Parallel)
**定义**: 多个Agent同时处理不同子任务，结果聚合。

**适用场景**: 批量处理、多维度分析、投票决策

**示例**: 多维度市场分析
```
                    ┌→ MarketTrendAgent
Input ──────────────┼→ CompetitorAgent → AggregatorAgent → 输出
                    └→ CustomerInsightAgent
```

### 2.5 规划式 (Planner)
**定义**: 先制定执行计划，再按步骤调度执行。

**适用场景**: 复杂项目管理、多步骤任务、目标分解

**示例**: 产品发布规划
```
PlannerAgent → [Step1, Step2, Step3, ...] → ExecutorAgent → 输出
```

### 2.6 协作式 (Collaborative)
**定义**: 多个Agent通过对话协作完成任务。

**适用场景**: 头脑风暴、方案讨论、决策制定

**示例**: 产品设计讨论
```
ProductManagerAgent ←→ DesignerAgent ←→ EngineerAgent ←→ 达成共识
```

## 3. 10人Agent团队设计

### 3.1 团队角色定义

| 编号 | Agent角色 | 主要职责 | 工作流模式 |
|------|----------|----------|-----------|
| A1 | RouterAgent | 任务分发与路由 | 路由式 |
| A2 | PlannerAgent | 任务规划与分解 | 规划式 |
| A3 | ResearchAgent | 信息收集与研究 | 顺序链式 |
| A4 | WritingAgent | 内容创作与编写 | 顺序链式 |
| A5 | ReviewAgent | 审核与评估 | 评估优化式 |
| A6 | CodeAgent | 代码生成与优化 | 评估优化式 |
| A7 | DataAgent | 数据分析与处理 | 并行式 |
| A8 | MarketAgent | 市场分析与洞察 | 并行式 |
| A9 | FacilitatorAgent | 协调与促进讨论 | 协作式 |
| A10 | AggregatorAgent | 结果聚合与输出 | 全模式 |

### 3.2 团队架构图

```
                    ┌─────────────────────────────────────┐
                    │         RouterAgent (A1)            │
                    │         任务路由中心                 │
                    └─────────────────┬───────────────────┘
                                      │
        ┌─────────────┬───────────────┼───────────────┬─────────────┐
        ↓             ↓               ↓               ↓             ↓
┌──────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│ PlannerAgent │ │ Research │ │  Evaluator   │ │ Parallel │ │Collaborative │
│    (A2)      │ │  Chain   │ │   Chain      │ │  Branch  │ │   Branch     │
└──────┬───────┘ │ (A3-A4)  │ │  (A5-A6)     │ │ (A7-A8)  │ │   (A9)       │
       │         └────┬─────┘ └──────┬───────┘ └────┬─────┘ └──────┬───────┘
       │              │              │              │              │
       ↓              ↓              ↓              ↓              ↓
┌──────────────┐ ┌──────────┐ ┌──────────────┐ ┌──────────┐ ┌──────────────┐
│  Task Plan   │ │  Output  │ │  Optimized   │ │ Aggregated│ │  Consensus   │
│              │ │          │ │   Result     │ │  Result   │ │              │
└──────────────┘ └──────────┘ └──────────────┘ └──────────┘ └──────────────┘
                                      │
                                      ↓
                    ┌─────────────────────────────────────┐
                    │      AggregatorAgent (A10)          │
                    │         结果聚合与输出               │
                    └─────────────────────────────────────┘
```

## 4. 任务分配与路由机制

### 4.1 路由决策引擎

```python
class RouterEngine:
    """
    智能路由决策引擎
    基于任务特征选择最佳工作流模式
    """
    
    ROUTING_RULES = {
        "content_creation": {
            "pattern": "sequential",
            "agents": ["A3", "A4", "A5"],
            "description": "内容创作：研究→写作→审核"
        },
        "code_generation": {
            "pattern": "evaluator_optimizer", 
            "agents": ["A6", "A5"],
            "description": "代码生成：生成→评估→优化"
        },
        "market_analysis": {
            "pattern": "parallel",
            "agents": ["A7", "A8", "A10"],
            "description": "市场分析：并行分析→聚合"
        },
        "project_planning": {
            "pattern": "planner",
            "agents": ["A2", "A3", "A4"],
            "description": "项目规划：规划→执行"
        },
        "brainstorming": {
            "pattern": "collaborative",
            "agents": ["A9", "A3", "A4", "A5"],
            "description": "头脑风暴：多Agent协作"
        },
        "data_processing": {
            "pattern": "parallel",
            "agents": ["A7", "A10"],
            "description": "数据处理：并行处理→聚合"
        }
    }
    
    def route(self, task: Task) -> WorkflowConfig:
        # 基于任务类型和内容特征进行路由
        task_type = self.classify_task(task)
        return self.ROUTING_RULES.get(task_type)
```

### 4.2 动态负载均衡

```python
class LoadBalancer:
    """
    动态负载均衡器
    根据Agent当前负载动态调整任务分配
    """
    
    def assign_task(self, task: Task, available_agents: List[Agent]) -> Agent:
        # 考虑因素：当前负载、历史性能、任务类型匹配度
        scores = []
        for agent in available_agents:
            score = self.calculate_score(agent, task)
            scores.append((agent, score))
        
        # 选择得分最高的Agent
        return max(scores, key=lambda x: x[1])[0]
    
    def calculate_score(self, agent: Agent, task: Task) -> float:
        load_score = 1.0 - (agent.current_load / agent.max_capacity)
        skill_score = agent.skill_match(task.type)
        performance_score = agent.historical_performance(task.type)
        return load_score * 0.4 + skill_score * 0.4 + performance_score * 0.2
```

## 5. 状态共享与同步机制

### 5.1 共享状态架构

```
┌─────────────────────────────────────────────────────────────┐
│                    Shared State Store                        │
│                  (Redis/PostgreSQL)                          │
├─────────────────────────────────────────────────────────────┤
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐       │
│  │  Workflow    │  │   Agent      │  │   Task       │       │
│  │   State      │  │   State      │  │   State      │       │
│  │              │  │              │  │              │       │
│  │ - status     │  │ - status     │  │ - status     │       │
│  │ - progress   │  │ - load       │  │ - result     │       │
│  │ - context    │  │ - history    │  │ - metadata   │       │
│  └──────────────┘  └──────────────┘  └──────────────┘       │
├─────────────────────────────────────────────────────────────┤
│                    Event Bus (Redis Pub/Sub)                 │
│              ┌─────────┐    ┌─────────┐                     │
│              │ Channel │    │ Channel │                     │
│              │  WF-001 │    │  WF-002 │                     │
│              └─────────┘    └─────────┘                     │
└─────────────────────────────────────────────────────────────┘
```

### 5.2 状态同步协议

```python
class StateSyncManager:
    """
    状态同步管理器
    确保分布式Agent间的状态一致性
    """
    
    async def publish_state_change(self, workflow_id: str, state_update: dict):
        """发布状态变更事件"""
        event = {
            "workflow_id": workflow_id,
            "timestamp": datetime.utcnow().isoformat(),
            "update": state_update,
            "version": self.get_next_version(workflow_id)
        }
        await self.event_bus.publish(f"workflow:{workflow_id}", event)
    
    async def subscribe_to_workflow(self, workflow_id: str, callback: Callable):
        """订阅工作流状态变更"""
        await self.event_bus.subscribe(f"workflow:{workflow_id}", callback)
    
    async def sync_agent_state(self, agent_id: str, state: AgentState):
        """同步Agent状态到共享存储"""
        await self.state_store.set(f"agent:{agent_id}", state.to_dict())
        await self.publish_state_change(agent_id, {"agent_state": state.to_dict()})
```

### 5.3 上下文传递机制

```python
class ContextManager:
    """
    上下文管理器
    管理工作流执行过程中的上下文传递
    """
    
    def create_context(self, workflow_id: str, initial_data: dict) -> Context:
        context = Context(
            workflow_id=workflow_id,
            global_vars={},
            step_outputs={},
            metadata={"created_at": datetime.utcnow()}
        )
        context.update(initial_data)
        return context
    
    def pass_context(self, from_step: str, to_step: str, context: Context) -> Context:
        """在步骤间传递上下文"""
        # 支持选择性传递和转换
        new_context = context.copy()
        new_context["previous_step"] = from_step
        new_context["current_step"] = to_step
        return new_context
```

## 6. 工作流执行引擎

### 6.1 核心执行器

```python
class WorkflowEngine:
    """
    工作流执行引擎
    支持6种工作流模式的执行
    """
    
    async def execute(self, workflow: WorkflowConfig, input_data: dict) -> Result:
        pattern = workflow.pattern
        
        if pattern == "sequential":
            return await self.execute_sequential(workflow, input_data)
        elif pattern == "router":
            return await self.execute_router(workflow, input_data)
        elif pattern == "evaluator_optimizer":
            return await self.execute_evaluator_optimizer(workflow, input_data)
        elif pattern == "parallel":
            return await self.execute_parallel(workflow, input_data)
        elif pattern == "planner":
            return await self.execute_planner(workflow, input_data)
        elif pattern == "collaborative":
            return await self.execute_collaborative(workflow, input_data)
        else:
            raise ValueError(f"Unknown pattern: {pattern}")
    
    async def execute_sequential(self, workflow: WorkflowConfig, input_data: dict) -> Result:
        """顺序链式执行"""
        context = self.context_manager.create_context(workflow.id, input_data)
        
        for step in workflow.steps:
            agent = self.agent_registry.get(step.agent_id)
            result = await agent.execute(step.task, context)
            context.step_outputs[step.id] = result
            
            if result.status == "failed":
                return Result(status="failed", error=result.error)
        
        return Result(status="success", data=context.step_outputs)
    
    async def execute_parallel(self, workflow: WorkflowConfig, input_data: dict) -> Result:
        """并行执行"""
        context = self.context_manager.create_context(workflow.id, input_data)
        
        # 并行执行所有步骤
        tasks = []
        for step in workflow.steps:
            agent = self.agent_registry.get(step.agent_id)
            task = agent.execute(step.task, context)
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 聚合结果
        for i, result in enumerate(results):
            context.step_outputs[workflow.steps[i].id] = result
        
        # 调用聚合器
        aggregator = self.agent_registry.get(workflow.aggregator_id)
        final_result = await aggregator.aggregate(context.step_outputs)
        
        return Result(status="success", data=final_result)
```

## 7. 监控与可观测性

### 7.1 指标收集

```python
class MetricsCollector:
    """
    指标收集器
    收集工作流执行的关键指标
    """
    
    METRICS = {
        "workflow_execution_time": "工作流执行时间",
        "step_execution_time": "步骤执行时间",
        "agent_utilization": "Agent利用率",
        "success_rate": "成功率",
        "error_rate": "错误率",
        "queue_depth": "队列深度"
    }
    
    def record_execution(self, workflow_id: str, duration: float, status: str):
        self.counter(f"workflow_execution_total", 
                    {"workflow_id": workflow_id, "status": status}).inc()
        self.histogram("workflow_execution_duration_seconds").observe(duration)
```

## 8. 总结

本设计文档定义了完整的Agent工作流系统，包含：

1. **6种工作流模式**：覆盖从简单到复杂的各种协作场景
2. **10人Agent团队**：角色分明、职责清晰的团队架构
3. **智能路由机制**：基于任务特征的动态路由决策
4. **状态同步机制**：可靠的分布式状态管理
5. **可扩展架构**：支持新模式的插件化扩展

系统采用事件驱动架构，确保高可用性和可扩展性，适用于企业级多Agent协作场景。
