"""
Agent工作流系统 - 核心实现代码
支持6种工作流模式：顺序链式/路由式/评估优化式/并行式/规划式/协作式
"""

import asyncio
import json
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, TypeVar, Generic
from collections import defaultdict
import logging

# 配置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class WorkflowPattern(Enum):
    """工作流模式枚举"""
    SEQUENTIAL = "sequential"
    ROUTER = "router"
    EVALUATOR_OPTIMIZER = "evaluator_optimizer"
    PARALLEL = "parallel"
    PLANNER = "planner"
    COLLABORATIVE = "collaborative"


class TaskStatus(Enum):
    """任务状态枚举"""
    PENDING = "pending"
    ASSIGNED = "assigned"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


class AgentStatus(Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    BUSY = "busy"
    OFFLINE = "offline"


# ==================== 数据模型 ====================

@dataclass
class Task:
    """任务模型"""
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    type: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING
    assigned_agent: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Agent:
    """Agent模型"""
    id: str
    name: str
    type: str
    capabilities: List[str] = field(default_factory=list)
    status: AgentStatus = AgentStatus.IDLE
    current_task: Optional[str] = None
    max_concurrent: int = 5
    load: float = 0.0
    
    async def execute(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        """执行任务（模拟）"""
        self.status = AgentStatus.BUSY
        self.current_task = task.id
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        
        # 模拟执行延迟
        await asyncio.sleep(0.5)
        
        # 模拟执行结果
        result = {
            "agent_id": self.id,
            "agent_name": self.name,
            "task_id": task.id,
            "status": "success",
            "output": f"Processed by {self.name}: {task.input_data}",
            "timestamp": datetime.now().isoformat()
        }
        
        task.output_data = result
        task.status = TaskStatus.COMPLETED
        task.completed_at = datetime.now()
        
        self.status = AgentStatus.IDLE
        self.current_task = None
        
        return result


@dataclass
class WorkflowStep:
    """工作流步骤"""
    id: str
    name: str
    agent_id: str
    task_template: str
    input_from: List[str] = field(default_factory=list)
    output_key: str = ""
    condition: Optional[str] = None


@dataclass
class WorkflowConfig:
    """工作流配置"""
    id: str
    name: str
    pattern: WorkflowPattern
    description: str = ""
    steps: List[WorkflowStep] = field(default_factory=list)
    router_config: Optional[Dict] = None
    evaluator_config: Optional[Dict] = None
    aggregator_id: Optional[str] = None
    config: Dict[str, Any] = field(default_factory=dict)


@dataclass
class WorkflowResult:
    """工作流执行结果"""
    workflow_id: str
    status: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    execution_time: float = 0.0
    step_results: Dict[str, Any] = field(default_factory=dict)


# ==================== 状态管理 ====================

class StateManager:
    """状态管理器 - 管理共享状态"""
    
    def __init__(self):
        self._states: Dict[str, Dict[str, Any]] = defaultdict(dict)
        self._subscribers: Dict[str, List[Callable]] = defaultdict(list)
        self._lock = asyncio.Lock()
    
    async def get(self, key: str) -> Optional[Dict[str, Any]]:
        """获取状态"""
        async with self._lock:
            return self._states.get(key, {}).copy()
    
    async def set(self, key: str, value: Dict[str, Any]):
        """设置状态"""
        async with self._lock:
            old_value = self._states.get(key)
            self._states[key] = value
            
            # 通知订阅者
            for callback in self._subscribers.get(key, []):
                asyncio.create_task(callback(key, value, old_value))
    
    async def update(self, key: str, updates: Dict[str, Any]):
        """更新状态"""
        async with self._lock:
            if key not in self._states:
                self._states[key] = {}
            self._states[key].update(updates)
            
            for callback in self._subscribers.get(key, []):
                asyncio.create_task(callback(key, self._states[key], None))
    
    def subscribe(self, key: str, callback: Callable):
        """订阅状态变更"""
        self._subscribers[key].append(callback)
    
    async def publish_event(self, channel: str, event: Dict[str, Any]):
        """发布事件"""
        logger.info(f"Event published to {channel}: {event}")


# ==================== 路由引擎 ====================

class RouterEngine:
    """路由引擎 - 智能任务路由"""
    
    ROUTING_RULES = {
        "content_creation": {
            "pattern": WorkflowPattern.SEQUENTIAL,
            "agents": ["A3", "A4", "A5"],
            "description": "内容创作：研究→写作→审核"
        },
        "code_generation": {
            "pattern": WorkflowPattern.EVALUATOR_OPTIMIZER,
            "agents": ["A6", "A5"],
            "description": "代码生成：生成→评估→优化"
        },
        "market_analysis": {
            "pattern": WorkflowPattern.PARALLEL,
            "agents": ["A7", "A8", "A10"],
            "description": "市场分析：并行分析→聚合"
        },
        "project_planning": {
            "pattern": WorkflowPattern.PLANNER,
            "agents": ["A2", "A3", "A4"],
            "description": "项目规划：规划→执行"
        },
        "brainstorming": {
            "pattern": WorkflowPattern.COLLABORATIVE,
            "agents": ["A9", "A3", "A4", "A5"],
            "description": "头脑风暴：多Agent协作"
        }
    }
    
    def classify_task(self, task: Task) -> str:
        """任务分类"""
        task_type = task.type.lower()
        
        if any(kw in task_type for kw in ["content", "writing", "article"]):
            return "content_creation"
        elif any(kw in task_type for kw in ["code", "programming", "development"]):
            return "code_generation"
        elif any(kw in task_type for kw in ["market", "analysis", "research"]):
            return "market_analysis"
        elif any(kw in task_type for kw in ["plan", "project", "schedule"]):
            return "project_planning"
        elif any(kw in task_type for kw in ["brainstorm", "discuss", "collaborate"]):
            return "brainstorming"
        else:
            return "content_creation"  # 默认
    
    def route(self, task: Task) -> Dict[str, Any]:
        """路由任务"""
        task_category = self.classify_task(task)
        return self.ROUTING_RULES.get(task_category, self.ROUTING_RULES["content_creation"])


# ==================== 负载均衡器 ====================

class LoadBalancer:
    """负载均衡器 - 动态任务分配"""
    
    def __init__(self, agent_registry: Dict[str, Agent]):
        self.agent_registry = agent_registry
    
    def calculate_score(self, agent: Agent, task: Task) -> float:
        """计算Agent得分"""
        # 负载分数 (越低越好)
        load_score = 1.0 - (agent.load / agent.max_concurrent)
        
        # 能力匹配分数
        task_caps = task.metadata.get("required_capabilities", [])
        if task_caps:
            matched = sum(1 for cap in task_caps if cap in agent.capabilities)
            skill_score = matched / len(task_caps)
        else:
            skill_score = 0.5
        
        # 状态分数
        status_score = 1.0 if agent.status == AgentStatus.IDLE else 0.5
        
        return load_score * 0.4 + skill_score * 0.4 + status_score * 0.2
    
    def select_agent(self, task: Task, candidate_ids: List[str]) -> Optional[Agent]:
        """选择最佳Agent"""
        candidates = [
            self.agent_registry.get(aid) 
            for aid in candidate_ids 
            if aid in self.agent_registry
        ]
        
        if not candidates:
            return None
        
        # 计算得分并排序
        scored = [(agent, self.calculate_score(agent, task)) for agent in candidates]
        scored.sort(key=lambda x: x[1], reverse=True)
        
        return scored[0][0]


# ==================== 工作流执行器 ====================

class WorkflowExecutor:
    """工作流执行器 - 执行各种工作流模式"""
    
    def __init__(self, agent_registry: Dict[str, Agent], state_manager: StateManager):
        self.agent_registry = agent_registry
        self.state_manager = state_manager
        self.router = RouterEngine()
        self.load_balancer = LoadBalancer(agent_registry)
    
    async def execute(self, workflow: WorkflowConfig, input_data: Dict[str, Any]) -> WorkflowResult:
        """执行工作流"""
        start_time = datetime.now()
        workflow_id = str(uuid.uuid4())
        
        logger.info(f"Starting workflow {workflow.name} ({workflow.pattern.value})")
        
        try:
            if workflow.pattern == WorkflowPattern.SEQUENTIAL:
                result = await self._execute_sequential(workflow, input_data)
            elif workflow.pattern == WorkflowPattern.ROUTER:
                result = await self._execute_router(workflow, input_data)
            elif workflow.pattern == WorkflowPattern.EVALUATOR_OPTIMIZER:
                result = await self._execute_evaluator_optimizer(workflow, input_data)
            elif workflow.pattern == WorkflowPattern.PARALLEL:
                result = await self._execute_parallel(workflow, input_data)
            elif workflow.pattern == WorkflowPattern.PLANNER:
                result = await self._execute_planner(workflow, input_data)
            elif workflow.pattern == WorkflowPattern.COLLABORATIVE:
                result = await self._execute_collaborative(workflow, input_data)
            else:
                raise ValueError(f"Unknown pattern: {workflow.pattern}")
            
            execution_time = (datetime.now() - start_time).total_seconds()
            
            return WorkflowResult(
                workflow_id=workflow_id,
                status="success",
                data=result,
                execution_time=execution_time
            )
            
        except Exception as e:
            logger.error(f"Workflow execution failed: {e}")
            return WorkflowResult(
                workflow_id=workflow_id,
                status="failed",
                error=str(e),
                execution_time=(datetime.now() - start_time).total_seconds()
            )
    
    async def _execute_sequential(self, workflow: WorkflowConfig, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """顺序链式执行"""
        context = {"input": input_data, "steps": {}}
        
        for step in workflow.steps:
            agent = self.agent_registry.get(step.agent_id)
            if not agent:
                raise ValueError(f"Agent {step.agent_id} not found")
            
            # 准备任务输入
            task_input = self._prepare_task_input(step, context)
            task = Task(type=step.name, input_data=task_input)
            
            # 执行任务
            result = await agent.execute(task, context)
            context["steps"][step.output_key] = result
            
            # 更新状态
            await self.state_manager.update(
                f"workflow:{workflow.id}",
                {"current_step": step.id, "progress": len(context["steps"]) / len(workflow.steps)}
            )
        
        return context["steps"]
    
    async def _execute_router(self, workflow: WorkflowConfig, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """路由式执行"""
        # 使用路由Agent进行分类
        router_config = workflow.router_config
        router_agent = self.agent_registry.get(router_config["agent_id"])
        
        # 创建分类任务
        classify_task = Task(
            type="classification",
            input_data=input_data
        )
        
        # 获取分类结果
        classification = await router_agent.execute(classify_task, {})
        category = classification.get("category", "default")
        
        # 根据分类选择分支
        for branch in router_config.get("branches", []):
            if branch["condition"] == f"category == '{category}'" or branch["condition"] == "default":
                target_agent = self.agent_registry.get(branch["target_agent"])
                task = Task(type="routed_task", input_data=input_data)
                result = await target_agent.execute(task, {})
                return {"category": category, "result": result}
        
        return {"category": category, "result": None}
    
    async def _execute_evaluator_optimizer(self, workflow: WorkflowConfig, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """评估优化式执行"""
        config = workflow.evaluator_config
        generator_agent = self.agent_registry.get(config["generator_agent_id"])
        evaluator_agent = self.agent_registry.get(config["evaluator_agent_id"])
        
        max_iterations = config.get("max_iterations", 5)
        quality_threshold = config.get("quality_threshold", 8.0)
        
        iteration = 0
        feedback = ""
        best_result = None
        best_score = 0.0
        
        while iteration < max_iterations:
            iteration += 1
            
            # 生成
            gen_task = Task(
                type="generation",
                input_data={**input_data, "feedback": feedback}
            )
            generated = await generator_agent.execute(gen_task, {})
            
            # 评估
            eval_task = Task(
                type="evaluation",
                input_data={"generated": generated, "requirement": input_data}
            )
            evaluation = await evaluator_agent.execute(eval_task, {})
            score = evaluation.get("score", 0.0)
            
            if score > best_score:
                best_score = score
                best_result = generated
            
            if score >= quality_threshold:
                break
            
            feedback = evaluation.get("feedback", "")
        
        return {
            "result": best_result,
            "score": best_score,
            "iterations": iteration
        }
    
    async def _execute_parallel(self, workflow: WorkflowConfig, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """并行式执行"""
        async def execute_step(step: WorkflowStep) -> tuple:
            agent = self.agent_registry.get(step.agent_id)
            task = Task(type=step.name, input_data=input_data)
            result = await agent.execute(task, {})
            return (step.output_key, result)
        
        # 并行执行所有步骤
        tasks = [execute_step(step) for step in workflow.steps]
        results = await asyncio.gather(*tasks)
        
        # 收集结果
        step_results = dict(results)
        
        # 聚合结果
        if workflow.aggregator_id:
            aggregator = self.agent_registry.get(workflow.aggregator_id)
            agg_task = Task(
                type="aggregation",
                input_data=step_results
            )
            aggregated = await aggregator.execute(agg_task, {})
            return {"steps": step_results, "aggregated": aggregated}
        
        return {"steps": step_results}
    
    async def _execute_planner(self, workflow: WorkflowConfig, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """规划式执行"""
        planner_agent = self.agent_registry.get(workflow.config["planner_agent_id"])
        executor_agent = self.agent_registry.get(workflow.config["executor_agent_id"])
        
        # 规划阶段
        plan_task = Task(type="planning", input_data=input_data)
        plan = await planner_agent.execute(plan_task, {})
        
        # 执行阶段
        steps = plan.get("steps", [])
        executed_steps = []
        
        for step in steps:
            exec_task = Task(
                type="execution",
                input_data={"step": step, "plan": plan}
            )
            result = await executor_agent.execute(exec_task, {})
            executed_steps.append({"step": step, "result": result})
        
        return {"plan": plan, "executed_steps": executed_steps}
    
    async def _execute_collaborative(self, workflow: WorkflowConfig, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """协作式执行"""
        facilitator = self.agent_registry.get(workflow.config["facilitator_agent_id"])
        participants = workflow.config.get("participants", [])
        
        max_rounds = workflow.config.get("max_rounds", 5)
        discussion_log = []
        
        for round_num in range(max_rounds):
            round_results = []
            
            for participant in participants:
                agent = self.agent_registry.get(participant["agent_id"])
                task = Task(
                    type="collaboration",
                    input_data={
                        "topic": input_data.get("topic"),
                        "role": participant["role"],
                        "previous_discussion": discussion_log,
                        "round": round_num
                    }
                )
                result = await agent.execute(task, {})
                round_results.append({
                    "agent": participant["agent_id"],
                    "role": participant["role"],
                    "contribution": result
                })
            
            discussion_log.append({"round": round_num, "contributions": round_results})
            
            # 检查是否达成共识
            facilitator_task = Task(
                type="consensus_check",
                input_data={"discussion": discussion_log}
            )
            consensus = await facilitator.execute(facilitator_task, {})
            
            if consensus.get("consensus_reached", False):
                break
        
        return {
            "discussion_log": discussion_log,
            "rounds": len(discussion_log),
            "consensus": consensus
        }
    
    def _prepare_task_input(self, step: WorkflowStep, context: Dict[str, Any]) -> Dict[str, Any]:
        """准备任务输入"""
        # 从上下文中收集输入
        inputs = {}
        for key in step.input_from:
            if key in context["steps"]:
                inputs[key] = context["steps"][key]
        return inputs


# ==================== 工作流管理器 ====================

class WorkflowManager:
    """工作流管理器 - 管理工作流生命周期"""
    
    def __init__(self):
        self.agent_registry: Dict[str, Agent] = {}
        self.workflow_configs: Dict[str, WorkflowConfig] = {}
        self.state_manager = StateManager()
        self.executor: Optional[WorkflowExecutor] = None
    
    def register_agent(self, agent: Agent):
        """注册Agent"""
        self.agent_registry[agent.id] = agent
        logger.info(f"Registered agent: {agent.name} ({agent.id})")
    
    def register_workflow(self, workflow: WorkflowConfig):
        """注册工作流配置"""
        self.workflow_configs[workflow.id] = workflow
        logger.info(f"Registered workflow: {workflow.name} ({workflow.id})")
    
    def initialize(self):
        """初始化执行器"""
        self.executor = WorkflowExecutor(self.agent_registry, self.state_manager)
    
    async def execute_workflow(self, workflow_id: str, input_data: Dict[str, Any]) -> WorkflowResult:
        """执行工作流"""
        workflow = self.workflow_configs.get(workflow_id)
        if not workflow:
            raise ValueError(f"Workflow {workflow_id} not found")
        
        if not self.executor:
            self.initialize()
        
        return await self.executor.execute(workflow, input_data)
    
    async def route_and_execute(self, task: Task) -> WorkflowResult:
        """路由并执行任务"""
        # 自动路由
        route_info = self.executor.router.route(task)
        
        # 找到对应的工作流
        for wf in self.workflow_configs.values():
            if wf.pattern == route_info["pattern"]:
                return await self.execute_workflow(wf.id, task.input_data)
        
        raise ValueError(f"No workflow found for pattern: {route_info['pattern']}")


# ==================== 示例使用 ====================

async def main():
    """主函数 - 演示6种工作流模式"""
    
    # 创建工作流管理器
    manager = WorkflowManager()
    
    # 注册10个Agent
    agents = [
        Agent(id="A1", name="RouterAgent", type="router", capabilities=["routing", "classification"]),
        Agent(id="A2", name="PlannerAgent", type="planner", capabilities=["planning", "decomposition"]),
        Agent(id="A3", name="ResearchAgent", type="researcher", capabilities=["research", "analysis"]),
        Agent(id="A4", name="WritingAgent", type="writer", capabilities=["writing", "content"]),
        Agent(id="A5", name="ReviewAgent", type="evaluator", capabilities=["review", "evaluation"]),
        Agent(id="A6", name="CodeAgent", type="developer", capabilities=["coding", "programming"]),
        Agent(id="A7", name="DataAgent", type="analyst", capabilities=["data", "analysis"]),
        Agent(id="A8", name="MarketAgent", type="analyst", capabilities=["market", "research"]),
        Agent(id="A9", name="FacilitatorAgent", type="coordinator", capabilities=["facilitation", "coordination"]),
        Agent(id="A10", name="AggregatorAgent", type="aggregator", capabilities=["aggregation", "synthesis"]),
    ]
    
    for agent in agents:
        manager.register_agent(agent)
    
    # 创建6种工作流配置
    
    # 1. 顺序链式工作流
    sequential_wf = WorkflowConfig(
        id="wf_sequential",
        name="内容创作工作流",
        pattern=WorkflowPattern.SEQUENTIAL,
        steps=[
            WorkflowStep(id="s1", name="研究", agent_id="A3", task_template="研究主题: {{topic}}", output_key="research"),
            WorkflowStep(id="s2", name="写作", agent_id="A4", task_template="基于研究写作", input_from=["research"], output_key="writing"),
            WorkflowStep(id="s3", name="审核", agent_id="A5", task_template="审核内容质量", input_from=["writing"], output_key="review"),
        ]
    )
    manager.register_workflow(sequential_wf)
    
    # 2. 路由式工作流
    router_wf = WorkflowConfig(
        id="wf_router",
        name="智能路由工作流",
        pattern=WorkflowPattern.ROUTER,
        router_config={
            "agent_id": "A1",
            "branches": [
                {"condition": "category == 'technical'", "target_agent": "A6"},
                {"condition": "category == 'content'", "target_agent": "A4"},
                {"condition": "default", "target_agent": "A9"}
            ]
        }
    )
    manager.register_workflow(router_wf)
    
    # 3. 评估优化式工作流
    eval_opt_wf = WorkflowConfig(
        id="wf_eval_opt",
        name="代码生成优化工作流",
        pattern=WorkflowPattern.EVALUATOR_OPTIMIZER,
        evaluator_config={
            "generator_agent_id": "A6",
            "evaluator_agent_id": "A5",
            "max_iterations": 3,
            "quality_threshold": 8.0
        }
    )
    manager.register_workflow(eval_opt_wf)
    
    # 4. 并行式工作流
    parallel_wf = WorkflowConfig(
        id="wf_parallel",
        name="市场分析工作流",
        pattern=WorkflowPattern.PARALLEL,
        steps=[
            WorkflowStep(id="p1", name="趋势分析", agent_id="A8", task_template="分析市场趋势", output_key="trends"),
            WorkflowStep(id="p2", name="竞品分析", agent_id="A8", task_template="分析竞争对手", output_key="competitors"),
            WorkflowStep(id="p3", name="数据分析", agent_id="A7", task_template="分析用户数据", output_key="data"),
        ],
        aggregator_id="A10"
    )
    manager.register_workflow(parallel_wf)
    
    # 5. 规划式工作流
    planner_wf = WorkflowConfig(
        id="wf_planner",
        name="项目规划工作流",
        pattern=WorkflowPattern.PLANNER,
        config={
            "planner_agent_id": "A2",
            "executor_agent_id": "A3"
        }
    )
    manager.register_workflow(planner_wf)
    
    # 6. 协作式工作流
    collaborative_wf = WorkflowConfig(
        id="wf_collaborative",
        name="头脑风暴工作流",
        pattern=WorkflowPattern.COLLABORATIVE,
        config={
            "facilitator_agent_id": "A9",
            "participants": [
                {"agent_id": "A3", "role": "研究员"},
                {"agent_id": "A4", "role": "创意师"},
                {"agent_id": "A5", "role": "评估师"}
            ],
            "max_rounds": 3
        }
    )
    manager.register_workflow(collaborative_wf)
    
    # 初始化
    manager.initialize()
    
    # 演示执行各种工作流
    print("=" * 60)
    print("Agent工作流系统演示")
    print("=" * 60)
    
    # 1. 顺序链式
    print("\n【1. 顺序链式工作流 - 内容创作】")
    result = await manager.execute_workflow("wf_sequential", {
        "topic": "AI工作流系统",
        "tone": "professional"
    })
    print(f"状态: {result.status}")
    print(f"执行时间: {result.execution_time:.2f}s")
    print(f"步骤结果: {list(result.data.keys())}")
    
    # 2. 路由式
    print("\n【2. 路由式工作流 - 智能路由】")
    result = await manager.execute_workflow("wf_router", {
        "query": "如何优化Python代码性能？",
        "category": "technical"
    })
    print(f"状态: {result.status}")
    print(f"执行时间: {result.execution_time:.2f}s")
    
    # 3. 评估优化式
    print("\n【3. 评估优化式工作流 - 代码生成】")
    result = await manager.execute_workflow("wf_eval_opt", {
        "requirement": "实现一个快速排序算法",
        "language": "python"
    })
    print(f"状态: {result.status}")
    print(f"执行时间: {result.execution_time:.2f}s")
    print(f"迭代次数: {result.data.get('iterations')}")
    print(f"最终得分: {result.data.get('score')}")
    
    # 4. 并行式
    print("\n【4. 并行式工作流 - 市场分析】")
    result = await manager.execute_workflow("wf_parallel", {
        "product": "AI助手",
        "market": "enterprise"
    })
    print(f"状态: {result.status}")
    print(f"执行时间: {result.execution_time:.2f}s")
    print(f"并行步骤: {list(result.data.get('steps', {}).keys())}")
    
    # 5. 规划式
    print("\n【5. 规划式工作流 - 项目规划】")
    result = await manager.execute_workflow("wf_planner", {
        "goal": "开发AI工作流系统",
        "constraints": ["3个月", "10人团队"]
    })
    print(f"状态: {result.status}")
    print(f"执行时间: {result.execution_time:.2f}s")
    
    # 6. 协作式
    print("\n【6. 协作式工作流 - 头脑风暴】")
    result = await manager.execute_workflow("wf_collaborative", {
        "topic": "如何提升用户体验"
    })
    print(f"状态: {result.status}")
    print(f"执行时间: {result.execution_time:.2f}s")
    print(f"讨论轮数: {result.data.get('rounds')}")
    
    # 自动路由演示
    print("\n【自动路由演示】")
    task = Task(type="code_generation", input_data={"requirement": "实现用户认证系统"})
    route_info = manager.executor.router.route(task)
    print(f"任务类型: {task.type}")
    print(f"路由模式: {route_info['pattern'].value}")
    print(f"推荐Agent: {route_info['agents']}")
    
    print("\n" + "=" * 60)
    print("所有工作流执行完成!")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
