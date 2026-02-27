# Multi-Agent协作系统 - 实现代码框架

## 项目结构

```
openclaw-multi-agent/
├── src/
│   ├── core/
│   │   ├── __init__.py
│   │   ├── agent.py              # Agent基类
│   │   ├── orchestrator.py       # 协调者实现
│   │   ├── coordinator.py        # 调度者实现
│   │   ├── executor.py           # 执行者实现
│   │   ├── validator.py          # 验证者实现
│   │   └── critic.py             # 批评者实现
│   ├── protocol/
│   │   ├── __init__.py
│   │   ├── a2a.py                # A2A协议实现
│   │   ├── mcp.py                # MCP协议实现
│   │   ├── message.py            # 消息格式定义
│   │   └── blackboard.py         # 黑板模式实现
│   ├── workflow/
│   │   ├── __init__.py
│   │   ├── patterns.py           # 工作流模式基类
│   │   ├── pipeline.py           # 顺序管道模式
│   │   ├── parallel.py           # 并行处理模式
│   │   ├── hierarchical.py       # 层级传递模式
│   │   ├── supervisor.py         # 监督者模式
│   │   ├── critic_reviewer.py    # 批评者-审查者模式
│   │   └── democratic.py         # 民主协商模式
│   ├── governance/
│   │   ├── __init__.py
│   │   ├── monitor.py            # 监控实现
│   │   ├── audit.py              # 审计实现
│   │   ├── fault_tolerance.py    # 容错实现
│   │   └── security.py           # 安全实现
│   ├── personality/
│   │   ├── __init__.py
│   │   ├── soul.py               # SOUL人格集成
│   │   ├── emotions.py           # 情绪管理
│   │   └── consistency.py        # 一致性检查
│   ├── task/
│   │   ├── __init__.py
│   │   ├── router.py             # 任务路由
│   │   ├── queue.py              # 优先级队列
│   │   └── dependency.py         # 依赖管理
│   └── utils/
│       ├── __init__.py
│       ├── vector_clock.py       # 向量时钟
│       ├── config.py             # 配置管理
│       └── logger.py             # 日志工具
├── config/
│   ├── agents.yaml               # Agent配置
│   ├── workflow.yaml             # 工作流配置
│   └── personality.yaml          # 人格配置
├── tests/
│   ├── test_agent.py
│   ├── test_workflow.py
│   └── test_protocol.py
├── examples/
│   ├── code_review_example.py
│   ├── research_example.py
│   └── incident_response_example.py
├── requirements.txt
└── README.md
```

## 核心代码实现

### 1. Agent基类 (src/core/agent.py)

```python
"""
Agent基类 - 所有Agent的抽象基类
"""
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional, List
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import uuid
from datetime import datetime

class AgentState(Enum):
    IDLE = "idle"
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    RETRYING = "retrying"
    SHUTDOWN = "shutdown"

class Priority(Enum):
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1

@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    description: str = ""
    input_data: Any = None
    priority: Priority = Priority.NORMAL
    dependencies: List[str] = field(default_factory=list)
    deadline: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class TaskResult:
    task_id: str
    success: bool
    output: Any = None
    error: Optional[str] = None
    duration_ms: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AgentMetrics:
    total_tasks: int = 0
    successful_tasks: int = 0
    failed_tasks: int = 0
    avg_duration_ms: float = 0.0
    last_updated: datetime = field(default_factory=datetime.now)
    
    def record_success(self, duration_ms: int):
        self.total_tasks += 1
        self.successful_tasks += 1
        self.avg_duration_ms = (
            (self.avg_duration_ms * (self.total_tasks - 1) + duration_ms) 
            / self.total_tasks
        )
        self.last_updated = datetime.now()
    
    def record_failure(self, duration_ms: int):
        self.total_tasks += 1
        self.failed_tasks += 1
        self.last_updated = datetime.now()

@dataclass
class AgentConfig:
    agent_id: str
    role: str
    personality_dimensions: Dict[str, int] = field(default_factory=dict)
    capabilities: List[str] = field(default_factory=list)
    max_concurrent_tasks: int = 5
    timeout_seconds: int = 300
    retry_attempts: int = 3

class Agent(ABC):
    """Agent抽象基类"""
    
    def __init__(self, config: AgentConfig):
        self.config = config
        self.state = AgentState.IDLE
        self.current_tasks: Dict[str, Task] = {}
        self.metrics = AgentMetrics()
        self._lock = asyncio.Lock()
        
    @abstractmethod
    async def process(self, task: Task) -> TaskResult:
        """处理任务的核心方法 - 子类必须实现"""
        pass
    
    @abstractmethod
    def get_capabilities(self) -> List[str]:
        """返回Agent能力列表 - 子类必须实现"""
        pass
    
    async def execute(self, task: Task) -> TaskResult:
        """执行任务并记录指标"""
        start_time = asyncio.get_event_loop().time()
        
        try:
            async with self._lock:
                if len(self.current_tasks) >= self.config.max_concurrent_tasks:
                    return TaskResult(
                        task_id=task.id,
                        success=False,
                        error="Agent overloaded"
                    )
                
                self.state = AgentState.RUNNING
                self.current_tasks[task.id] = task
            
            # 设置超时
            result = await asyncio.wait_for(
                self.process(task),
                timeout=self.config.timeout_seconds
            )
            
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            result.duration_ms = duration_ms
            
            if result.success:
                self.metrics.record_success(duration_ms)
            else:
                self.metrics.record_failure(duration_ms)
            
            return result
            
        except asyncio.TimeoutError:
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            self.metrics.record_failure(duration_ms)
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Task timeout after {self.config.timeout_seconds}s",
                duration_ms=duration_ms
            )
        except Exception as e:
            duration_ms = int((asyncio.get_event_loop().time() - start_time) * 1000)
            self.metrics.record_failure(duration_ms)
            return TaskResult(
                task_id=task.id,
                success=False,
                error=str(e),
                duration_ms=duration_ms
            )
        finally:
            async with self._lock:
                if task.id in self.current_tasks:
                    del self.current_tasks[task.id]
                if not self.current_tasks:
                    self.state = AgentState.IDLE
    
    def get_load(self) -> float:
        """返回当前负载 (0-1)"""
        return len(self.current_tasks) / self.config.max_concurrent_tasks
    
    def get_health(self) -> Dict[str, Any]:
        """返回Agent健康状态"""
        return {
            "agent_id": self.config.agent_id,
            "role": self.config.role,
            "state": self.state.value,
            "load": self.get_load(),
            "metrics": {
                "total_tasks": self.metrics.total_tasks,
                "success_rate": (
                    self.metrics.successful_tasks / self.metrics.total_tasks
                    if self.metrics.total_tasks > 0 else 1.0
                ),
                "avg_duration_ms": self.metrics.avg_duration_ms
            }
        }
    
    async def shutdown(self):
        """优雅关闭Agent"""
        self.state = AgentState.SHUTDOWN
        # 等待当前任务完成
        while self.current_tasks:
            await asyncio.sleep(0.1)
```

### 2. Orchestrator实现 (src/core/orchestrator.py)

```python
"""
Orchestrator Agent - 战略协调者
"""
from typing import List, Dict, Optional, Any
import asyncio
from dataclasses import dataclass

from .agent import Agent, AgentConfig, Task, TaskResult, AgentState
from .coordinator import CoordinatorAgent

@dataclass
class TaskAssignment:
    task: Task
    coordinator_id: str
    priority: int

class OrchestratorAgent(Agent):
    """
    战略协调者 - 负责目标分解和资源调度
    人格维度: Personality 90, Motivations 95, Conflict 85
    """
    
    def __init__(self, config: AgentConfig):
        super().__init__(config)
        self.coordinators: Dict[str, CoordinatorAgent] = {}
        self.task_assignments: Dict[str, TaskAssignment] = {}
        self.completed_tasks: List[TaskResult] = []
        
    def register_coordinator(self, coordinator: CoordinatorAgent):
        """注册Coordinator"""
        self.coordinators[coordinator.config.agent_id] = coordinator
    
    async def process(self, task: Task) -> TaskResult:
        """
        处理高层任务：
        1. 目标分解
        2. 任务分配
        3. 进度监控
        4. 结果聚合
        """
        try:
            # 1. 分解目标
            subtasks = await self.decompose_goal(task)
            
            if not subtasks:
                return TaskResult(
                    task_id=task.id,
                    success=True,
                    output=task.input_data,
                    metadata={"message": "No decomposition needed"}
                )
            
            # 2. 分配任务
            assignments = self.assign_tasks(subtasks)
            
            # 3. 监控执行
            results = await self.monitor_execution(assignments)
            
            # 4. 聚合结果
            return self.aggregate_results(task.id, results)
            
        except Exception as e:
            return TaskResult(
                task_id=task.id,
                success=False,
                error=f"Orchestrator error: {str(e)}"
            )
    
    async def decompose_goal(self, task: Task) -> List[Task]:
        """将高层目标分解为可执行子任务"""
        # 这里可以集成LLM进行智能分解
        # 简化实现：根据任务类型预定义分解策略
        
        decomposition_strategies = {
            "code_review": self._decompose_code_review,
            "research": self._decompose_research,
            "data_analysis": self._decompose_data_analysis,
        }
        
        strategy = decomposition_strategies.get(
            task.type, 
            self._default_decomposition
        )
        
        return await strategy(task)
    
    async def _decompose_code_review(self, task: Task) -> List[Task]:
        """代码审查任务分解"""
        return [
            Task(
                type="syntax_check",
                description="语法检查",
                input_data=task.input_data,
                priority=Priority.HIGH
            ),
            Task(
                type="security_scan",
                description="安全扫描",
                input_data=task.input_data,
                priority=Priority.CRITICAL
            ),
            Task(
                type="quality_review",
                description="质量审查",
                input_data=task.input_data,
                priority=Priority.NORMAL
            ),
        ]
    
    async def _decompose_research(self, task: Task) -> List[Task]:
        """研究任务分解"""
        return [
            Task(
                type="data_collection",
                description="数据收集",
                input_data=task.input_data,
                priority=Priority.HIGH
            ),
            Task(
                type="data_analysis",
                description="数据分析",
                input_data=task.input_data,
                priority=Priority.HIGH,
                dependencies=[]  # 将在运行时填充
            ),
            Task(
                type="report_generation",
                description="报告生成",
                input_data=task.input_data,
                priority=Priority.NORMAL,
                dependencies=[]
            ),
        ]
    
    async def _decompose_data_analysis(self, task: Task) -> List[Task]:
        """数据分析任务分解"""
        return [
            Task(
                type="data_loading",
                description="数据加载",
                input_data=task.input_data,
                priority=Priority.HIGH
            ),
            Task(
                type="data_cleaning",
                description="数据清洗",
                input_data=task.input_data,
                priority=Priority.HIGH
            ),
            Task(
                type="analysis_execution",
                description="分析执行",
                input_data=task.input_data,
                priority=Priority.NORMAL
            ),
        ]
    
    async def _default_decomposition(self, task: Task) -> List[Task]:
        """默认分解策略 - 不分解"""
        return [task]
    
    def assign_tasks(self, subtasks: List[Task]) -> Dict[str, List[Task]]:
        """根据负载和能力分配任务给Coordinators"""
        assignments: Dict[str, List[Task]] = {}
        
        for subtask in subtasks:
            # 选择最佳Coordinator
            coordinator = self.select_coordinator(subtask)
            
            if coordinator.config.agent_id not in assignments:
                assignments[coordinator.config.agent_id] = []
            assignments[coordinator.config.agent_id].append(subtask)
            
            # 记录分配
            self.task_assignments[subtask.id] = TaskAssignment(
                task=subtask,
                coordinator_id=coordinator.config.agent_id,
                priority=subtask.priority.value
            )
        
        return assignments
    
    def select_coordinator(self, task: Task) -> CoordinatorAgent:
        """基于负载和能力选择Coordinator"""
        candidates = [
            c for c in self.coordinators.values()
            if self._can_handle(c, task)
        ]
        
        if not candidates:
            # 如果没有匹配的Coordinator，选择负载最低的
            candidates = list(self.coordinators.values())
        
        # 选择负载最低的
        return min(candidates, key=lambda c: c.get_load())
    
    def _can_handle(self, coordinator: CoordinatorAgent, task: Task) -> bool:
        """检查Coordinator是否能处理该任务"""
        coordinator_caps = set(coordinator.get_capabilities())
        # 这里简化处理，实际应该根据任务需求匹配能力
        return True
    
    async def monitor_execution(self, 
                                assignments: Dict[str, List[Task]]) -> List[TaskResult]:
        """监控所有Coordinator的执行进度"""
        tasks = []
        
        for coord_id, subtasks in assignments.items():
            coordinator = self.coordinators.get(coord_id)
            if not coordinator:
                continue
                
            for subtask in subtasks:
                task_future = asyncio.create_task(
                    self._execute_with_retry(coordinator, subtask)
                )
                tasks.append(task_future)
        
        # 等待所有任务完成
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        processed_results = []
        for result in results:
            if isinstance(result, Exception):
                processed_results.append(TaskResult(
                    task_id="unknown",
                    success=False,
                    error=str(result)
                ))
            else:
                processed_results.append(result)
        
        return processed_results
    
    async def _execute_with_retry(self, 
                                  coordinator: CoordinatorAgent, 
                                  task: Task) -> TaskResult:
        """带重试的执行"""
        for attempt in range(self.config.retry_attempts):
            result = await coordinator.execute(task)
            if result.success:
                return result
            
            if attempt < self.config.retry_attempts - 1:
                await asyncio.sleep(2 ** attempt)  # 指数退避
        
        return result
    
    def aggregate_results(self, 
                         parent_task_id: str,
                         results: List[TaskResult]) -> TaskResult:
        """聚合子任务结果"""
        failed_results = [r for r in results if not r.success]
        
        if failed_results:
            return TaskResult(
                task_id=parent_task_id,
                success=False,
                error=f"{len(failed_results)} subtasks failed",
                metadata={
                    "total": len(results),
                    "failed": len(failed_results),
                    "errors": [r.error for r in failed_results]
                }
            )
        
        # 合并所有成功结果
        aggregated_output = {
            "subtask_results": [
                {
                    "task_id": r.task_id,
                    "output": r.output,
                    "duration_ms": r.duration_ms
                }
                for r in results
            ]
        }
        
        total_duration = sum(r.duration_ms for r in results)
        
        return TaskResult(
            task_id=parent_task_id,
            success=True,
            output=aggregated_output,
            duration_ms=total_duration,
            metadata={"subtask_count": len(results)}
        )
    
    def get_capabilities(self) -> List[str]:
        return [
            'goal_decomposition',
            'strategy_planning',
            'resource_scheduling',
            'conflict_arbitration',
            'progress_monitoring',
            'result_aggregation'
        ]
```

### 3. 工作流模式实现 (src/workflow/patterns.py)

```python
"""
工作流模式实现 - 6种核心工作流模式
"""
from abc import ABC, abstractmethod
from typing import List, Callable, Any, Optional
import asyncio
from enum import Enum

from ..core.agent import Agent, Task, TaskResult

class WorkflowPattern(ABC):
    """工作流模式基类"""
    
    @abstractmethod
    async def execute(self, input_data: Any, context: Dict[str, Any] = None) -> Any:
        """执行工作流"""
        pass

class SequentialPipeline(WorkflowPattern):
    """
    顺序管道模式
    Agent A → Agent B → Agent C → [Feedback Loop]
    """
    
    def __init__(self, 
                 agents: List[Agent],
                 allow_feedback: bool = True,
                 max_iterations: int = 3,
                 validator: Optional[Agent] = None):
        self.agents = agents
        self.allow_feedback = allow_feedback
        self.max_iterations = max_iterations
        self.validator = validator
    
    async def execute(self, input_data: Any, context: Dict[str, Any] = None) -> Any:
        current = input_data
        
        for iteration in range(self.max_iterations):
            # 顺序执行每个Agent
            for i, agent in enumerate(self.agents):
                task = Task(
                    id=f"seq_{iteration}_{i}_{agent.config.agent_id}",
                    type='SEQUENTIAL',
                    description=f"Sequential step {i} by {agent.config.agent_id}",
                    input_data=current,
                    priority=context.get('priority', Priority.NORMAL) if context else Priority.NORMAL
                )
                
                result = await agent.execute(task)
                
                if not result.success:
                    return {
                        "success": False,
                        "error": result.error,
                        "iteration": iteration,
                        "agent": agent.config.agent_id
                    }
                
                current = result.output
            
            # 检查是否需要反馈循环
            if not self.allow_feedback:
                break
            
            # 验证结果质量
            if self.validator:
                is_valid = await self._validate(current)
                if is_valid:
                    break
            else:
                # 简化验证：检查输出是否为空
                if current:
                    break
        
        return {
            "success": True,
            "output": current,
            "iterations": iteration + 1
        }
    
    async def _validate(self, data: Any) -> bool:
        """验证结果"""
        if not self.validator:
            return True
        
        task = Task(
            type='VALIDATION',
            input_data=data
        )
        result = await self.validator.execute(task)
        return result.success

class ParallelProcessing(WorkflowPattern):
    """
    并行处理模式
       ┌→ Agent A →┐
Input → ├→ Agent B →├→ Aggregator → Output
       └→ Agent C →┘
    """
    
    def __init__(self, 
                 agents: List[Agent],
                 aggregator: Optional[Callable[[List[Any]], Any]] = None,
                 voting_mode: bool = False,
                 min_success_rate: float = 0.5):
        self.agents = agents
        self.aggregator = aggregator or self._default_aggregator
        self.voting_mode = voting_mode
        self.min_success_rate = min_success_rate
    
    async def execute(self, input_data: Any, context: Dict[str, Any] = None) -> Any:
        # 并行执行所有Agent
        tasks = [
            self._execute_agent(agent, input_data, i)
            for i, agent in enumerate(self.agents)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # 处理结果
        successful_results = []
        failed_results = []
        
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                failed_results.append({
                    "agent": self.agents[i].config.agent_id,
                    "error": str(result)
                })
            elif result.get('success'):
                successful_results.append(result.get('output'))
            else:
                failed_results.append({
                    "agent": self.agents[i].config.agent_id,
                    "error": result.get('error')
                })
        
        # 检查成功率
        success_rate = len(successful_results) / len(self.agents)
        if success_rate < self.min_success_rate:
            return {
                "success": False,
                "error": f"Success rate {success_rate:.2%} below threshold",
                "failed": failed_results
            }
        
        # 聚合结果
        if self.voting_mode:
            aggregated = self._vote_aggregate(successful_results)
        else:
            aggregated = self.aggregator(successful_results)
        
        return {
            "success": True,
            "output": aggregated,
            "success_rate": success_rate,
            "failed_count": len(failed_results)
        }
    
    async def _execute_agent(self, 
                            agent: Agent, 
                            input_data: Any, 
                            index: int) -> Dict[str, Any]:
        """执行单个Agent"""
        task = Task(
            id=f"par_{index}_{agent.config.agent_id}",
            type='PARALLEL',
            description=f"Parallel execution by {agent.config.agent_id}",
            input_data=input_data
        )
        
        result = await agent.execute(task)
        
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "agent": agent.config.agent_id
        }
    
    def _default_aggregator(self, outputs: List[Any]) -> Any:
        """默认聚合器 - 返回列表"""
        return outputs
    
    def _vote_aggregate(self, outputs: List[Any]) -> Any:
        """多数投票聚合"""
        from collections import Counter
        
        # 将输出转换为可哈希类型进行投票
        try:
            votes = Counter(str(o) for o in outputs)
            winner = votes.most_common(1)[0][0]
            
            # 找到对应的原始输出
            for output in outputs:
                if str(output) == winner:
                    return output
        except:
            pass
        
        return outputs[0] if outputs else None

class HierarchicalWorkflow(WorkflowPattern):
    """
    层级传递模式
    Orchestrator → Coordinators → Executors
    """
    
    def __init__(self, orchestrator: 'OrchestratorAgent'):
        self.orchestrator = orchestrator
    
    async def execute(self, input_data: Any, context: Dict[str, Any] = None) -> Any:
        task = Task(
            id=f"hier_{asyncio.get_event_loop().time()}",
            type='HIERARCHICAL',
            description="Hierarchical workflow execution",
            input_data=input_data,
            priority=context.get('priority', Priority.NORMAL) if context else Priority.NORMAL
        )
        
        result = await self.orchestrator.execute(task)
        
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "duration_ms": result.duration_ms
        }

class SupervisorPattern(WorkflowPattern):
    """
    监督者协调模式
    Supervisor监控多个Worker的执行
    """
    
    def __init__(self,
                 supervisor: Agent,
                 workers: List[Agent],
                 check_interval: float = 5.0,
                 timeout: float = 300.0):
        self.supervisor = supervisor
        self.workers = workers
        self.check_interval = check_interval
        self.timeout = timeout
        self.worker_status: Dict[str, Any] = {}
    
    async def execute(self, input_data: Any, context: Dict[str, Any] = None) -> Any:
        start_time = asyncio.get_event_loop().time()
        
        # 启动所有Worker
        worker_tasks = [
            self._monitor_worker(worker, input_data)
            for worker in self.workers
        ]
        
        # 同时启动监督任务
        supervisor_task = asyncio.create_task(
            self._supervise_workers(start_time)
        )
        
        # 等待所有Worker完成或超时
        done, pending = await asyncio.wait(
            worker_tasks + [supervisor_task],
            timeout=self.timeout,
            return_when=asyncio.ALL_COMPLETED
        )
        
        # 取消未完成的任务
        for task in pending:
            task.cancel()
        
        # 收集结果
        results = []
        for task in done:
            if task != supervisor_task:
                try:
                    result = task.result()
                    results.append(result)
                except Exception as e:
                    results.append({"success": False, "error": str(e)})
        
        # 汇总结果
        successful = [r for r in results if r.get('success')]
        failed = [r for r in results if not r.get('success')]
        
        return {
            "success": len(failed) == 0,
            "successful_count": len(successful),
            "failed_count": len(failed),
            "results": results,
            "supervisor_notes": self.worker_status
        }
    
    async def _monitor_worker(self, worker: Agent, input_data: Any) -> Dict[str, Any]:
        """监控单个Worker"""
        task = Task(
            id=f"supervised_{worker.config.agent_id}",
            type='SUPERVISED',
            input_data=input_data
        )
        
        # 定期更新状态
        result = None
        while result is None:
            # 检查Worker健康状态
            self.worker_status[worker.config.agent_id] = worker.get_health()
            
            # 这里简化处理，实际应该支持中断和恢复
            result = await worker.execute(task)
            
            if result is None:
                await asyncio.sleep(self.check_interval)
        
        return {
            "success": result.success,
            "output": result.output,
            "error": result.error,
            "worker": worker.config.agent_id
        }
    
    async def _supervise_workers(self, start_time: float):
        """监督所有Worker"""
        while True:
            elapsed = asyncio.get_event_loop().time() - start_time
            if elapsed > self.timeout:
                break
            
            # 检查整体进度
            all_idle = all(
                status.get('state') == 'idle'
                for status in self.worker_status.values()
            )
            
            if all_idle and self.worker_status:
                break
            
            await asyncio.sleep(self.check_interval)

class CriticReviewerPattern(WorkflowPattern):
    """
    批评者-审查者模式
    Generator → Critic → [Revision Loop] → Validator → Output
    """
    
    def __init__(self,
                 generator: Agent,
                 critic: Agent,
                 validator: Agent,
                 max_revisions: int = 3,
                 quality_threshold: float = 0.8):
        self.generator = generator
        self.critic = critic
        self.validator = validator
        self.max_revisions = max_revisions
        self.quality_threshold = quality_threshold
    
    async def execute(self, input_data: Any, context: Dict[str, Any] = None) -> Any:
        current_output = None
        revision_history = []
        
        for revision in range(self.max_revisions):
            # 1. 生成
            if revision == 0:
                generate_task = Task(
                    id=f"gen_{revision}",
                    type='GENERATE',
                    input_data=input_data
                )
            else:
                # 基于反馈修改
                generate_task = Task(
                    id=f"gen_{revision}",
                    type='REVISE',
                    input_data={
                        "original": input_data,
                        "previous_output": current_output,
                        "feedback": criticism
                    }
                )
            
            gen_result = await self.generator.execute(generate_task)
            
            if not gen_result.success:
                return {
                    "success": False,
                    "error": f"Generation failed: {gen_result.error}",
                    "revision": revision
                }
            
            current_output = gen_result.output
            
            # 2. 批评
            critic_task = Task(
                id=f"critic_{revision}",
                type='CRITICIZE',
                input_data=current_output
            )
            
            critic_result = await self.critic.execute(critic_task)
            criticism = critic_result.output if critic_result.success else None
            
            revision_history.append({
                "revision": revision,
                "output": current_output,
                "criticism": criticism
            })
            
            # 3. 验证
            validator_task = Task(
                id=f"validate_{revision}",
                type='VALIDATE',
                input_data=current_output
            )
            
            validator_result = await self.validator.execute(validator_task)
            
            if validator_result.success:
                # 检查质量分数
                quality_score = validator_result.metadata.get('quality_score', 0)
                if quality_score >= self.quality_threshold:
                    return {
                        "success": True,
                        "output": current_output,
                        "quality_score": quality_score,
                        "revisions": revision + 1,
                        "history": revision_history
                    }
        
        # 达到最大修订次数
        return {
            "success": False,
            "error": "Max revisions reached without meeting quality threshold",
            "final_output": current_output,
            "history": revision_history
        }

class DemocraticPattern(WorkflowPattern):
    """
    民主协商模式
    多个Agent提出方案，通过投票达成共识
    """
    
    def __init__(self,
                 proposers: List[Agent],
                 voters: List[Agent],
                 consensus_threshold: float = 0.67,
                 max_rounds: int = 5,
                 voting_method: str = 'approval'):
        self.proposers = proposers
        self.voters = voters
        self.consensus_threshold = consensus_threshold
        self.max_rounds = max_rounds
        self.voting_method = voting_method
    
    async def execute(self, input_data: Any, context: Dict[str, Any] = None) -> Any:
        # 1. 提案阶段
        proposals = await self._generate_proposals(input_data)
        
        # 2. 协商阶段
        for round_num in range(self.max_rounds):
            # 投票
            votes = await self._collect_votes(proposals)
            
            # 检查共识
            consensus_result = self._check_consensus(votes, proposals)
            
            if consensus_result['reached']:
                return {
                    "success": True,
                    "winner": consensus_result['winner'],
                    "consensus_rate": consensus_result['rate'],
                    "rounds": round_num + 1,
                    "votes": votes
                }
            
            # 如果没有达成共识，进入下一轮协商
            if round_num < self.max_rounds - 1:
                proposals = await self._refine_proposals(proposals, votes)
        
        # 未能在最大轮数内达成共识
        return {
            "success": False,
            "error": "Failed to reach consensus within max rounds",
            "final_votes": votes,
            "proposals": proposals
        }
    
    async def _generate_proposals(self, input_data: Any) -> List[Dict[str, Any]]:
        """生成提案"""
        proposal_tasks = [
            self._get_proposal(proposer, input_data, i)
            for i, proposer in enumerate(self.proposers)
        ]
        
        results = await asyncio.gather(*proposal_tasks)
        return [r for r in results if r is not None]
    
    async def _get_proposal(self, 
                           proposer: Agent, 
                           input_data: Any, 
                           index: int) -> Optional[Dict[str, Any]]:
        """获取单个提案"""
        task = Task(
            id=f"propose_{index}_{proposer.config.agent_id}",
            type='PROPOSE',
            input_data=input_data
        )
        
        result = await proposer.execute(task)
        
        if result.success:
            return {
                "proposal_id": f"p_{index}",
                "proposer": proposer.config.agent_id,
                "content": result.output
            }
        return None
    
    async def _collect_votes(self, proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """收集投票"""
        vote_tasks = [
            self._get_votes(voter, proposals, i)
            for i, voter in enumerate(self.voters)
        ]
        
        results = await asyncio.gather(*vote_tasks)
        
        # 汇总投票
        all_votes = {}
        for voter_id, votes in results:
            all_votes[voter_id] = votes
        
        return all_votes
    
    async def _get_votes(self, 
                        voter: Agent, 
                        proposals: List[Dict[str, Any]], 
                        index: int) -> tuple:
        """获取单个投票者的投票"""
        task = Task(
            id=f"vote_{index}_{voter.config.agent_id}",
            type='VOTE',
            input_data={
                "proposals": proposals,
                "method": self.voting_method
            }
        )
        
        result = await voter.execute(task)
        
        if result.success:
            return (voter.config.agent_id, result.output)
        return (voter.config.agent_id, [])
    
    def _check_consensus(self, 
                        votes: Dict[str, Any], 
                        proposals: List[Dict[str, Any]]) -> Dict[str, Any]:
        """检查是否达成共识"""
        # 统计每个提案的得票数
        vote_counts = {}
        for voter_id, voter_votes in votes.items():
            for proposal_id in voter_votes:
                vote_counts[proposal_id] = vote_counts.get(proposal_id, 0) + 1
        
        if not vote_counts:
            return {'reached': False}
        
        # 找出得票最多的提案
        winner_id = max(vote_counts.keys(), key=lambda k: vote_counts[k])
        winner_votes = vote_counts[winner_id]
        total_voters = len(self.voters)
        
        consensus_rate = winner_votes / total_voters
        
        return {
            'reached': consensus_rate >= self.consensus_threshold,
            'winner': next(p for p in proposals if p['proposal_id'] == winner_id),
            'rate': consensus_rate,
            'vote_counts': vote_counts
        }
    
    async def _refine_proposals(self, 
                               proposals: List[Dict[str, Any]], 
                               votes: Dict[str, Any]) -> List[Dict[str, Any]]:
        """基于投票反馈优化提案"""
        # 简化实现：保留得票最高的几个提案
        vote_counts = {}
        for voter_votes in votes.values():
            for proposal_id in voter_votes:
                vote_counts[proposal_id] = vote_counts.get(proposal_id, 0) + 1
        
        # 按得票数排序
        sorted_proposals = sorted(
            proposals,
            key=lambda p: vote_counts.get(p['proposal_id'], 0),
            reverse=True
        )
        
        # 保留前50%的提案
        keep_count = max(1, len(sorted_proposals) // 2)
        return sorted_proposals[:keep_count]
```

### 4. A2A协议实现 (src/protocol/a2a.py)

```python
"""
A2A (Agent-to-Agent) 协议实现
"""
import json
from typing import Dict, Optional, Callable, Any
from dataclasses import dataclass, asdict, field
from enum import Enum
import asyncio
import uuid
from datetime import datetime

class MessageType(Enum):
    TASK_ASSIGN = "task_assign"
    TASK_RESULT = "task_result"
    STATUS_UPDATE = "status_update"
    HEARTBEAT = "heartbeat"
    ERROR = "error"
    NEGOTIATION = "negotiation"
    CONSENSUS = "consensus"
    QUERY = "query"
    RESPONSE = "response"

class Priority(Enum):
    CRITICAL = 4
    HIGH = 3
    NORMAL = 2
    LOW = 1

@dataclass
class MessageHeader:
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    correlation_id: str = ""
    timestamp: float = field(default_factory=lambda: datetime.now().timestamp())
    version: str = "2.0"
    priority: Priority = Priority.NORMAL

@dataclass
class RoutingInfo:
    from_agent: str = ""
    to_agent: str = ""
    reply_to: Optional[str] = None
    broadcast: bool = False

@dataclass
class MessagePayload:
    type: MessageType = MessageType.QUERY
    content: Dict[str, Any] = field(default_factory=dict)
    metadata: Optional[Dict[str, Any]] = None

@dataclass
class SecurityInfo:
    trace_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    signature: Optional[str] = None
    encryption: Optional[str] = None

@dataclass
class A2AMessage:
    header: MessageHeader = field(default_factory=MessageHeader)
    routing: RoutingInfo = field(default_factory=RoutingInfo)
    payload: MessagePayload = field(default_factory=MessagePayload)
    security: SecurityInfo = field(default_factory=SecurityInfo)
    
    def to_json(self) -> str:
        return json.dumps({
            "header": {
                "message_id": self.header.message_id,
                "correlation_id": self.header.correlation_id,
                "timestamp": self.header.timestamp,
                "version": self.header.version,
                "priority": self.header.priority.value
            },
            "routing": {
                "from_agent": self.routing.from_agent,
                "to_agent": self.routing.to_agent,
                "reply_to": self.routing.reply_to,
                "broadcast": self.routing.broadcast
            },
            "payload": {
                "type": self.payload.type.value,
                "content": self.payload.content,
                "metadata": self.payload.metadata
            },
            "security": {
                "trace_id": self.security.trace_id,
                "signature": self.security.signature,
                "encryption": self.security.encryption
            }
        })
    
    @classmethod
    def from_json(cls, data: str) -> 'A2AMessage':
        obj = json.loads(data)
        return cls(
            header=MessageHeader(
                message_id=obj["header"]["message_id"],
                correlation_id=obj["header"]["correlation_id"],
                timestamp=obj["header"]["timestamp"],
                version=obj["header"]["version"],
                priority=Priority(obj["header"]["priority"])
            ),
            routing=RoutingInfo(
                from_agent=obj["routing"]["from_agent"],
                to_agent=obj["routing"]["to_agent"],
                reply_to=obj["routing"].get("reply_to"),
                broadcast=obj["routing"].get("broadcast", False)
            ),
            payload=MessagePayload(
                type=MessageType(obj["payload"]["type"]),
                content=obj["payload"]["content"],
                metadata=obj["payload"].get("metadata")
            ),
            security=SecurityInfo(
                trace_id=obj["security"]["trace_id"],
                signature=obj["security"].get("signature"),
                encryption=obj["security"].get("encryption")
            )
        )

class A2AProtocol:
    """A2A协议实现"""
    
    def __init__(self, agent_id: str):
        self.agent_id = agent_id
        self.message_handlers: Dict[MessageType, Callable] = {}
        self.pending_responses: Dict[str, asyncio.Future] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self._running = False
    
    def register_handler(self, msg_type: MessageType, 
                        handler: Callable[[A2AMessage], Any]):
        """注册消息处理器"""
        self.message_handlers[msg_type] = handler
    
    async def start(self):
        """启动协议处理循环"""
        self._running = True
        while self._running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(), 
                    timeout=1.0
                )
                await self._handle_message(message)
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                print(f"Error handling message: {e}")
    
    def stop(self):
        """停止协议处理"""
        self._running = False
    
    async def send(self, message: A2AMessage) -> None:
        """发送消息 - 子类需要实现具体传输"""
        raise NotImplementedError("Subclass must implement send method")
    
    async def send_and_wait(self, message: A2AMessage, 
                           timeout: float = 30.0) -> A2AMessage:
        """发送消息并等待响应"""
        future = asyncio.Future()
        self.pending_responses[message.header.message_id] = future
        
        await self.send(message)
        
        try:
            return await asyncio.wait_for(future, timeout)
        except asyncio.TimeoutError:
            del self.pending_responses[message.header.message_id]
            raise
    
    async def _handle_message(self, message: A2AMessage):
        """处理接收到的消息"""
        # 检查是否是响应
        if message.header.correlation_id in self.pending_responses:
            future = self.pending_responses.pop(message.header.correlation_id)
            future.set_result(message)
            return
        
        # 调用对应处理器
        handler = self.message_handlers.get(message.payload.type)
        if handler:
            try:
                result = await handler(message)
                
                # 如果需要回复
                if message.routing.reply_to:
                    response = A2AMessage(
                        header=MessageHeader(
                            correlation_id=message.header.message_id
                        ),
                        routing=RoutingInfo(
                            from_agent=self.agent_id,
                            to_agent=message.routing.from_agent
                        ),
                        payload=MessagePayload(
                            type=MessageType.RESPONSE,
                            content={"result": result}
                        )
                    )
                    await self.send(response)
                    
            except Exception as e:
                # 发送错误响应
                error_response = A2AMessage(
                    header=MessageHeader(
                        correlation_id=message.header.message_id
                    ),
                    routing=RoutingInfo(
                        from_agent=self.agent_id,
                        to_agent=message.routing.from_agent
                    ),
                    payload=MessagePayload(
                        type=MessageType.ERROR,
                        content={"error": str(e)}
                    )
                )
                await self.send(error_response)
    
    def create_message(self,
                      to_agent: str,
                      msg_type: MessageType,
                      content: Dict[str, Any],
                      priority: Priority = Priority.NORMAL) -> A2AMessage:
        """创建消息"""
        return A2AMessage(
            header=MessageHeader(priority=priority),
            routing=RoutingInfo(
                from_agent=self.agent_id,
                to_agent=to_agent,
                reply_to=self.agent_id
            ),
            payload=MessagePayload(
                type=msg_type,
                content=content
            )
        )

# 协商协议实现
class NegotiationProtocol:
    """协商协议实现"""
    
    def __init__(self, a2a: A2AProtocol):
        self.a2a = a2a
        self.negotiations: Dict[str, Any] = {}
    
    async def initiate(self, 
                      to_agent: str, 
                      proposal: Dict[str, Any],
                      deadline: float = 60.0) -> str:
        """发起协商"""
        negotiation_id = str(uuid.uuid4())
        
        message = self.a2a.create_message(
            to_agent=to_agent,
            msg_type=MessageType.NEGOTIATION,
            content={
                "phase": "INITIATE",
                "negotiation_id": negotiation_id,
                "proposal": proposal,
                "deadline": deadline
            }
        )
        
        self.negotiations[negotiation_id] = {
            "status": "pending",
            "proposal": proposal,
            "responses": []
        }
        
        await self.a2a.send(message)
        return negotiation_id
    
    async def respond(self,
                     negotiation_id: str,
                     to_agent: str,
                     decision: str,
                     counter_proposal: Optional[Dict] = None) -> None:
        """响应协商"""
        message = self.a2a.create_message(
            to_agent=to_agent,
            msg_type=MessageType.NEGOTIATION,
            content={
                "phase": "RESPOND",
                "negotiation_id": negotiation_id,
                "decision": decision,  # ACCEPT, REJECT, COUNTER
                "counter_proposal": counter_proposal
            }
        )
        
        await self.a2a.send(message)
```

---

## 使用示例

### 1. 创建并运行Sequential Pipeline

```python
import asyncio
from src.core.agent import AgentConfig
from src.core.executor import ExecutorAgent
from src.core.validator import ValidatorAgent
from src.workflow.patterns import SequentialPipeline

async def main():
    # 创建Agent
    executor1 = ExecutorAgent(AgentConfig(
        agent_id="exec_1",
        role="executor",
        capabilities=["code_analysis"]
    ))
    
    executor2 = ExecutorAgent(AgentConfig(
        agent_id="exec_2",
        role="executor",
        capabilities=["security_scan"]
    ))
    
    validator = ValidatorAgent(AgentConfig(
        agent_id="validator_1",
        role="validator",
        capabilities=["quality_check"]
    ))
    
    # 创建顺序管道
    pipeline = SequentialPipeline(
        agents=[executor1, executor2],
        validator=validator,
        max_iterations=2
    )
    
    # 执行
    result = await pipeline.execute({
        "code": "def hello(): return 'world'"
    })
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

### 2. 创建Hierarchical工作流

```python
from src.core.orchestrator import OrchestratorAgent
from src.core.coordinator import CoordinatorAgent
from src.workflow.patterns import HierarchicalWorkflow

async def main():
    # 创建Orchestrator
    orchestrator = OrchestratorAgent(AgentConfig(
        agent_id="orch_1",
        role="orchestrator",
        personality_dimensions={
            "personality": 90,
            "motivations": 95,
            "conflict": 85
        }
    ))
    
    # 创建并注册Coordinators
    for i in range(3):
        coord = CoordinatorAgent(AgentConfig(
            agent_id=f"coord_{i}",
            role="coordinator"
        ))
        orchestrator.register_coordinator(coord)
    
    # 创建层级工作流
    workflow = HierarchicalWorkflow(orchestrator)
    
    # 执行复杂任务
    result = await workflow.execute({
        "task_type": "research",
        "topic": "Multi-Agent Systems"
    })
    
    print(result)

if __name__ == "__main__":
    asyncio.run(main())
```

---

*代码框架版本: 1.0*  
*更新日期: 2026-02-27*  
*配套文档: AGENTS_v2.md, multi_agent_architecture.md*
