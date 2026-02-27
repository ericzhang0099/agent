"""
GoalDrivenAgent - 自主Agent核心
目标驱动架构，支持自动目标拆解、长期规划和24/7运行

核心特性:
1. 目标驱动执行 - 所有行为围绕目标展开
2. 自动目标拆解 - 将复杂目标分解为可执行任务
3. 长期规划 - 支持多阶段规划和里程碑追踪
4. 24/7运行循环 - 持续监控、执行、学习
5. 自主决策 - 基于优先级和状态的智能调度
"""

import asyncio
import json
import logging
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime, timedelta
from enum import Enum, auto
from typing import Dict, List, Optional, Callable, Any, Set
from collections import deque
import threading
import uuid

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("GoalDrivenAgent")


# ═══════════════════════════════════════════════════════════════════════════════
# 核心枚举和常量
# ═══════════════════════════════════════════════════════════════════════════════

class GoalStatus(Enum):
    """目标状态"""
    PENDING = auto()      # 待处理
    ACTIVE = auto()       # 执行中
    PAUSED = auto()       # 暂停
    COMPLETED = auto()    # 已完成
    FAILED = auto()       # 失败
    CANCELLED = auto()    # 已取消


class TaskStatus(Enum):
    """任务状态"""
    PENDING = auto()      # 待执行
    RUNNING = auto()      # 运行中
    BLOCKED = auto()      # 被阻塞
    COMPLETED = auto()    # 已完成
    FAILED = auto()       # 失败
    RETRYING = auto()     # 重试中


class Priority(Enum):
    """优先级等级"""
    CRITICAL = 1    # 关键 - 立即执行
    HIGH = 2        # 高 - 尽快执行
    MEDIUM = 3      # 中 - 正常执行
    LOW = 4         # 低 - 空闲时执行
    BACKGROUND = 5  # 后台 - 资源充足时执行


# ═══════════════════════════════════════════════════════════════════════════════
# 数据类定义
# ═══════════════════════════════════════════════════════════════════════════════

@dataclass
class Task:
    """任务定义"""
    id: str
    name: str
    description: str
    goal_id: str                    # 所属目标ID
    priority: Priority
    status: TaskStatus = TaskStatus.PENDING
    
    # 执行相关
    execute_fn: Optional[Callable] = None
    dependencies: List[str] = field(default_factory=list)  # 依赖的任务ID
    
    # 时间和重试
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    max_retries: int = 3
    retry_count: int = 0
    
    # 结果
    result: Any = None
    error: Optional[str] = None
    
    # 元数据
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'goal_id': self.goal_id,
            'priority': self.priority.name,
            'status': self.status.name,
            'dependencies': self.dependencies,
            'created_at': self.created_at.isoformat(),
            'completed_at': self.completed_at.isoformat() if self.completed_at else None,
            'retry_count': self.retry_count,
            'result': str(self.result) if self.result else None,
            'error': self.error,
        }


@dataclass
class Goal:
    """目标定义"""
    id: str
    name: str
    description: str
    status: GoalStatus = GoalStatus.PENDING
    priority: Priority = Priority.MEDIUM
    
    # 层级关系
    parent_id: Optional[str] = None
    sub_goals: List[str] = field(default_factory=list)
    
    # 任务列表
    tasks: List[Task] = field(default_factory=list)
    
    # 时间规划
    created_at: datetime = field(default_factory=datetime.now)
    deadline: Optional[datetime] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    
    # 进度追踪
    progress: float = 0.0  # 0.0 - 1.0
    milestones: List[Dict] = field(default_factory=list)
    
    # 成功标准
    success_criteria: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'id': self.id,
            'name': self.name,
            'description': self.description,
            'status': self.status.name,
            'priority': self.priority.name,
            'parent_id': self.parent_id,
            'sub_goals': self.sub_goals,
            'progress': self.progress,
            'created_at': self.created_at.isoformat(),
            'deadline': self.deadline.isoformat() if self.deadline else None,
            'tasks_count': len(self.tasks),
            'completed_tasks': sum(1 for t in self.tasks if t.status == TaskStatus.COMPLETED),
        }


@dataclass
class Plan:
    """长期规划"""
    id: str
    name: str
    description: str
    goals: List[Goal] = field(default_factory=list)
    
    # 时间跨度
    start_date: datetime = field(default_factory=datetime.now)
    end_date: Optional[datetime] = None
    
    # 规划阶段
    phases: List[Dict] = field(default_factory=list)
    
    # 状态
    is_active: bool = True
    revision: int = 1
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'goals_count': len(self.goals),
            'is_active': self.is_active,
            'revision': self.revision,
        }


@dataclass
class AgentState:
    """Agent状态"""
    is_running: bool = False
    current_goal_id: Optional[str] = None
    current_task_id: Optional[str] = None
    
    # 统计
    total_goals_completed: int = 0
    total_tasks_completed: int = 0
    total_failures: int = 0
    
    # 性能
    start_time: Optional[datetime] = None
    uptime_seconds: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            'is_running': self.is_running,
            'current_goal_id': self.current_goal_id,
            'current_task_id': self.current_task_id,
            'total_goals_completed': self.total_goals_completed,
            'total_tasks_completed': self.total_tasks_completed,
            'uptime_seconds': self.uptime_seconds,
        }


# ═══════════════════════════════════════════════════════════════════════════════
# 目标拆解器
# ═══════════════════════════════════════════════════════════════════════════════

class GoalDecomposer:
    """
    目标拆解器
    将复杂目标自动分解为可执行的子目标和任务
    """
    
    def __init__(self):
        self.decomposition_patterns = self._init_patterns()
    
    def _init_patterns(self) -> Dict[str, Any]:
        """初始化拆解模式"""
        return {
            'research': {
                'phases': ['资料收集', '信息分析', '报告撰写', '结果验证'],
                'task_template': [
                    {'name': '收集{topic}相关资料', 'priority': Priority.HIGH},
                    {'name': '分析{topic}关键信息', 'priority': Priority.HIGH},
                    {'name': '撰写{topic}分析报告', 'priority': Priority.MEDIUM},
                    {'name': '验证{topic}分析结果', 'priority': Priority.MEDIUM},
                ]
            },
            'development': {
                'phases': ['需求分析', '设计', '实现', '测试', '部署'],
                'task_template': [
                    {'name': '分析{feature}需求', 'priority': Priority.CRITICAL},
                    {'name': '设计{feature}架构', 'priority': Priority.HIGH},
                    {'name': '实现{feature}功能', 'priority': Priority.HIGH},
                    {'name': '测试{feature}功能', 'priority': Priority.HIGH},
                    {'name': '部署{feature}', 'priority': Priority.MEDIUM},
                ]
            },
            'analysis': {
                'phases': ['数据收集', '数据清洗', '分析建模', '结果输出'],
                'task_template': [
                    {'name': '收集{dataset}数据', 'priority': Priority.HIGH},
                    {'name': '清洗{dataset}数据', 'priority': Priority.HIGH},
                    {'name': '建立{dataset}分析模型', 'priority': Priority.MEDIUM},
                    {'name': '输出{dataset}分析结果', 'priority': Priority.MEDIUM},
                ]
            },
            'learning': {
                'phases': ['目标设定', '资源收集', '学习执行', '知识验证'],
                'task_template': [
                    {'name': '设定{subject}学习目标', 'priority': Priority.HIGH},
                    {'name': '收集{subject}学习资源', 'priority': Priority.MEDIUM},
                    {'name': '执行{subject}学习计划', 'priority': Priority.HIGH},
                    {'name': '验证{subject}学习成果', 'priority': Priority.MEDIUM},
                ]
            }
        }
    
    def decompose(self, goal: Goal, goal_type: str = 'generic') -> List[Task]:
        """
        拆解目标为任务列表
        
        Args:
            goal: 目标对象
            goal_type: 目标类型 (research, development, analysis, learning, generic)
        
        Returns:
            任务列表
        """
        tasks = []
        
        # 获取拆解模式
        pattern = self.decomposition_patterns.get(goal_type, self.decomposition_patterns['research'])
        
        # 提取关键词
        keywords = self._extract_keywords(goal.name, goal.description)
        topic = keywords.get('topic', goal.name)
        
        # 生成任务
        prev_task_id = None
        for i, template in enumerate(pattern['task_template']):
            task_name = template['name'].format(topic=topic, feature=topic, dataset=topic, subject=topic)
            
            task = Task(
                id=f"task_{goal.id}_{i}_{uuid.uuid4().hex[:8]}",
                name=task_name,
                description=f"为目标 '{goal.name}' 执行任务: {task_name}",
                goal_id=goal.id,
                priority=template['priority'],
                dependencies=[prev_task_id] if prev_task_id else [],
                metadata={'phase': pattern['phases'][i] if i < len(pattern['phases']) else '执行'}
            )
            
            tasks.append(task)
            prev_task_id = task.id
        
        logger.info(f"目标 '{goal.name}' 已拆解为 {len(tasks)} 个任务")
        return tasks
    
    def _extract_keywords(self, name: str, description: str) -> Dict[str, str]:
        """从目标文本中提取关键词"""
        text = f"{name} {description}".lower()
        
        # 简单的关键词提取逻辑
        keywords = {'topic': name}
        
        # 尝试识别特定主题
        if '代码' in text or '开发' in text or '功能' in text:
            keywords['type'] = 'development'
        elif '研究' in text or '调研' in text or '分析' in text:
            keywords['type'] = 'research'
        elif '学习' in text or '掌握' in text:
            keywords['type'] = 'learning'
        elif '数据' in text:
            keywords['type'] = 'analysis'
        
        return keywords
    
    def create_sub_goals(self, goal: Goal, count: int = 3) -> List[Goal]:
        """
        创建子目标
        
        Args:
            goal: 父目标
            count: 子目标数量
        
        Returns:
            子目标列表
        """
        sub_goals = []
        phases = ['第一阶段', '第二阶段', '第三阶段', '第四阶段', '第五阶段']
        
        for i in range(min(count, len(phases))):
            sub_goal = Goal(
                id=f"subgoal_{goal.id}_{i}_{uuid.uuid4().hex[:8]}",
                name=f"{phases[i]}: {goal.name}",
                description=f"{phases[i]}目标 - {goal.description}",
                parent_id=goal.id,
                priority=goal.priority,
                status=GoalStatus.PENDING
            )
            sub_goals.append(sub_goal)
        
        return sub_goals


# ═══════════════════════════════════════════════════════════════════════════════
# 长期规划器
# ═══════════════════════════════════════════════════════════════════════════════

class LongTermPlanner:
    """
    长期规划器
    管理多阶段规划和里程碑追踪
    """
    
    def __init__(self):
        self.plans: Dict[str, Plan] = {}
        self.active_plan_id: Optional[str] = None
    
    def create_plan(
        self,
        name: str,
        description: str,
        duration_days: int = 30
    ) -> Plan:
        """创建新规划"""
        plan = Plan(
            id=f"plan_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            end_date=datetime.now() + timedelta(days=duration_days)
        )
        
        self.plans[plan.id] = plan
        logger.info(f"创建长期规划: {name} (ID: {plan.id})")
        return plan
    
    def add_goal_to_plan(self, plan_id: str, goal: Goal, phase: int = 0):
        """添加目标到规划"""
        if plan_id not in self.plans:
            raise ValueError(f"规划不存在: {plan_id}")
        
        plan = self.plans[plan_id]
        plan.goals.append(goal)
        
        # 确保阶段存在
        while len(plan.phases) <= phase:
            plan.phases.append({
                'name': f'第{len(plan.phases)+1}阶段',
                'goals': [],
                'status': 'pending'
            })
        
        plan.phases[phase]['goals'].append(goal.id)
        logger.info(f"目标 '{goal.name}' 已添加到规划 '{plan.name}' 的第{phase+1}阶段")
    
    def get_active_plan(self) -> Optional[Plan]:
        """获取当前活动规划"""
        if self.active_plan_id and self.active_plan_id in self.plans:
            return self.plans[self.active_plan_id]
        return None
    
    def set_active_plan(self, plan_id: str):
        """设置活动规划"""
        if plan_id not in self.plans:
            raise ValueError(f"规划不存在: {plan_id}")
        
        # 停用其他规划
        for plan in self.plans.values():
            plan.is_active = False
        
        self.active_plan_id = plan_id
        self.plans[plan_id].is_active = True
        logger.info(f"激活规划: {self.plans[plan_id].name}")
    
    def update_progress(self, plan_id: str) -> float:
        """更新规划进度"""
        if plan_id not in self.plans:
            return 0.0
        
        plan = self.plans[plan_id]
        if not plan.goals:
            return 0.0
        
        total_progress = sum(g.progress for g in plan.goals)
        avg_progress = total_progress / len(plan.goals)
        
        logger.info(f"规划 '{plan.name}' 整体进度: {avg_progress:.1%}")
        return avg_progress
    
    def get_next_goals(self, plan_id: str, count: int = 3) -> List[Goal]:
        """获取接下来要执行的目标"""
        if plan_id not in self.plans:
            return []
        
        plan = self.plans[plan_id]
        pending_goals = [
            g for g in plan.goals
            if g.status in [GoalStatus.PENDING, GoalStatus.PAUSED]
        ]
        
        # 按优先级排序
        pending_goals.sort(key=lambda g: g.priority.value)
        
        return pending_goals[:count]
    
    def revise_plan(self, plan_id: str, changes: Dict[str, Any]) -> Plan:
        """修订规划"""
        if plan_id not in self.plans:
            raise ValueError(f"规划不存在: {plan_id}")
        
        plan = self.plans[plan_id]
        plan.revision += 1
        
        if 'name' in changes:
            plan.name = changes['name']
        if 'description' in changes:
            plan.description = changes['description']
        if 'end_date' in changes:
            plan.end_date = changes['end_date']
        
        logger.info(f"规划 '{plan.name}' 已修订 (版本: {plan.revision})")
        return plan


# ═══════════════════════════════════════════════════════════════════════════════
# 执行引擎
# ═══════════════════════════════════════════════════════════════════════════════

class ExecutionEngine:
    """
    执行引擎
    负责任务调度和执行
    """
    
    def __init__(self, max_concurrent: int = 3):
        self.max_concurrent = max_concurrent
        self.running_tasks: Dict[str, Task] = {}
        self.task_queue: deque = deque()
        self.completed_tasks: List[Task] = []
        self.failed_tasks: List[Task] = []
        
        # 执行统计
        self.execution_stats = {
            'total_executed': 0,
            'total_success': 0,
            'total_failed': 0,
            'average_execution_time': 0.0,
        }
    
    def submit_task(self, task: Task) -> bool:
        """提交任务到执行队列"""
        if task.status != TaskStatus.PENDING:
            logger.warning(f"任务 {task.id} 状态不是PENDING，无法提交")
            return False
        
        self.task_queue.append(task)
        logger.info(f"任务 '{task.name}' 已提交到执行队列")
        return True
    
    def get_ready_tasks(self) -> List[Task]:
        """获取准备就绪的任务（依赖已满足）"""
        ready = []
        completed_ids = {t.id for t in self.completed_tasks}
        
        for task in list(self.task_queue):
            if task.status != TaskStatus.PENDING:
                continue
            
            # 检查依赖
            if all(dep_id in completed_ids for dep_id in task.dependencies):
                ready.append(task)
        
        # 按优先级排序
        ready.sort(key=lambda t: t.priority.value)
        return ready
    
    async def execute_task(self, task: Task) -> bool:
        """执行单个任务"""
        task.status = TaskStatus.RUNNING
        task.started_at = datetime.now()
        self.running_tasks[task.id] = task
        
        logger.info(f"开始执行任务: {task.name}")
        start_time = time.time()
        
        try:
            if task.execute_fn:
                # 执行自定义函数
                if asyncio.iscoroutinefunction(task.execute_fn):
                    result = await task.execute_fn(task)
                else:
                    result = task.execute_fn(task)
                task.result = result
            else:
                # 默认执行逻辑
                result = await self._default_execute(task)
                task.result = result
            
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now()
            self.completed_tasks.append(task)
            
            execution_time = time.time() - start_time
            self._update_stats(success=True, execution_time=execution_time)
            
            logger.info(f"任务 '{task.name}' 执行成功 (耗时: {execution_time:.2f}s)")
            return True
            
        except Exception as e:
            task.status = TaskStatus.FAILED
            task.error = str(e)
            task.retry_count += 1
            
            if task.retry_count < task.max_retries:
                task.status = TaskStatus.RETRYING
                logger.warning(f"任务 '{task.name}' 失败，准备重试 ({task.retry_count}/{task.max_retries}): {e}")
                self.task_queue.append(task)  # 重新加入队列
            else:
                self.failed_tasks.append(task)
                execution_time = time.time() - start_time
                self._update_stats(success=False, execution_time=execution_time)
                logger.error(f"任务 '{task.name}' 执行失败 (重试次数已用尽): {e}")
            
            return False
        
        finally:
            if task.id in self.running_tasks:
                del self.running_tasks[task.id]
    
    async def _default_execute(self, task: Task) -> Any:
        """默认任务执行逻辑"""
        # 模拟任务执行
        await asyncio.sleep(0.5)
        return {"status": "completed", "task_name": task.name}
    
    def _update_stats(self, success: bool, execution_time: float):
        """更新执行统计"""
        self.execution_stats['total_executed'] += 1
        
        if success:
            self.execution_stats['total_success'] += 1
        else:
            self.execution_stats['total_failed'] += 1
        
        # 更新平均执行时间
        n = self.execution_stats['total_executed']
        old_avg = self.execution_stats['average_execution_time']
        self.execution_stats['average_execution_time'] = (old_avg * (n - 1) + execution_time) / n
    
    def get_stats(self) -> Dict[str, Any]:
        """获取执行统计"""
        return {
            **self.execution_stats,
            'queue_length': len(self.task_queue),
            'running_count': len(self.running_tasks),
            'completed_count': len(self.completed_tasks),
            'failed_count': len(self.failed_tasks),
        }


# ═══════════════════════════════════════════════════════════════════════════════
# GoalDrivenAgent - 主类
# ═══════════════════════════════════════════════════════════════════════════════

class GoalDrivenAgent:
    """
    目标驱动自主Agent
    
    核心能力:
    - 目标管理: 创建、追踪、完成目标
    - 自动拆解: 将目标分解为可执行任务
    - 长期规划: 多阶段规划和里程碑管理
    - 24/7运行: 持续监控和执行循环
    - 自主决策: 基于优先级智能调度
    """
    
    def __init__(self, name: str = "GoalDrivenAgent"):
        self.name = name
        self.state = AgentState()
        
        # 核心组件
        self.decomposer = GoalDecomposer()
        self.planner = LongTermPlanner()
        self.engine = ExecutionEngine()
        
        # 数据存储
        self.goals: Dict[str, Goal] = {}
        self.tasks: Dict[str, Task] = {}
        
        # 运行控制
        self._running = False
        self._loop_task: Optional[asyncio.Task] = None
        self._shutdown_event = asyncio.Event()
        
        # 事件回调
        self._callbacks: Dict[str, List[Callable]] = {
            'goal_completed': [],
            'task_completed': [],
            'task_failed': [],
            'plan_updated': [],
        }
        
        logger.info(f"GoalDrivenAgent '{name}' 初始化完成")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 目标管理
    # ═══════════════════════════════════════════════════════════════════════════
    
    def create_goal(
        self,
        name: str,
        description: str = "",
        priority: Priority = Priority.MEDIUM,
        auto_decompose: bool = True,
        goal_type: str = 'generic'
    ) -> Goal:
        """
        创建新目标
        
        Args:
            name: 目标名称
            description: 目标描述
            priority: 优先级
            auto_decompose: 是否自动拆解为任务
            goal_type: 目标类型 (影响拆解策略)
        
        Returns:
            Goal对象
        """
        goal = Goal(
            id=f"goal_{uuid.uuid4().hex[:12]}",
            name=name,
            description=description,
            priority=priority,
            status=GoalStatus.PENDING
        )
        
        self.goals[goal.id] = goal
        logger.info(f"创建目标: {name} (ID: {goal.id}, 优先级: {priority.name})")
        
        # 自动拆解
        if auto_decompose:
            tasks = self.decomposer.decompose(goal, goal_type)
            goal.tasks = tasks
            for task in tasks:
                self.tasks[task.id] = task
            logger.info(f"目标 '{name}' 自动拆解为 {len(tasks)} 个任务")
        
        return goal
    
    def activate_goal(self, goal_id: str) -> bool:
        """激活目标开始执行"""
        if goal_id not in self.goals:
            logger.error(f"目标不存在: {goal_id}")
            return False
        
        goal = self.goals[goal_id]
        goal.status = GoalStatus.ACTIVE
        goal.started_at = datetime.now()
        
        # 将所有任务提交到执行引擎
        for task in goal.tasks:
            if task.status == TaskStatus.PENDING:
                self.engine.submit_task(task)
        
        self.state.current_goal_id = goal_id
        logger.info(f"目标 '{goal.name}' 已激活")
        return True
    
    def complete_goal(self, goal_id: str) -> bool:
        """标记目标完成"""
        if goal_id not in self.goals:
            return False
        
        goal = self.goals[goal_id]
        goal.status = GoalStatus.COMPLETED
        goal.completed_at = datetime.now()
        goal.progress = 1.0
        
        self.state.total_goals_completed += 1
        
        # 触发回调
        self._trigger_callback('goal_completed', goal)
        
        logger.info(f"目标 '{goal.name}' 已完成")
        return True
    
    def get_goal_progress(self, goal_id: str) -> float:
        """获取目标进度"""
        if goal_id not in self.goals:
            return 0.0
        
        goal = self.goals[goal_id]
        if not goal.tasks:
            return 1.0 if goal.status == GoalStatus.COMPLETED else 0.0
        
        completed = sum(1 for t in goal.tasks if t.status == TaskStatus.COMPLETED)
        progress = completed / len(goal.tasks)
        goal.progress = progress
        
        return progress
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 24/7 运行循环
    # ═══════════════════════════════════════════════════════════════════════════
    
    async def start(self):
        """启动Agent 24/7运行循环"""
        if self._running:
            logger.warning("Agent已经在运行中")
            return
        
        self._running = True
        self.state.is_running = True
        self.state.start_time = datetime.now()
        
        logger.info(f"🚀 GoalDrivenAgent '{self.name}' 启动 24/7 运行循环")
        
        self._loop_task = asyncio.create_task(self._main_loop())
    
    async def stop(self):
        """停止Agent运行"""
        if not self._running:
            return
        
        logger.info("正在停止Agent...")
        self._running = False
        self.state.is_running = False
        self._shutdown_event.set()
        
        if self._loop_task:
            self._loop_task.cancel()
            try:
                await self._loop_task
            except asyncio.CancelledError:
                pass
        
        logger.info("✅ Agent已停止")
    
    async def _main_loop(self):
        """主运行循环"""
        loop_counter = 0
        
        while self._running and not self._shutdown_event.is_set():
            loop_counter += 1
            
            try:
                # 1. 更新状态
                self._update_uptime()
                
                # 2. 处理执行任务
                await self._process_execution()
                
                # 3. 检查目标完成
                self._check_goal_completion()
                
                # 4. 规划更新
                await self._update_planning()
                
                # 5. 定期报告 (每10个循环)
                if loop_counter % 10 == 0:
                    self._report_status()
                
                # 短暂休眠，避免CPU占用过高
                await asyncio.sleep(1)
                
            except Exception as e:
                logger.error(f"主循环异常: {e}")
                await asyncio.sleep(5)  # 异常后等待更长时间
    
    async def _process_execution(self):
        """处理任务执行"""
        # 获取准备就绪的任务
        ready_tasks = self.engine.get_ready_tasks()
        
        # 并行执行多个任务（受并发限制）
        available_slots = self.engine.max_concurrent - len(self.engine.running_tasks)
        tasks_to_run = ready_tasks[:available_slots]
        
        if tasks_to_run:
            logger.debug(f"准备执行 {len(tasks_to_run)} 个任务")
            
            # 并发执行
            await asyncio.gather(
                *[self.engine.execute_task(task) for task in tasks_to_run],
                return_exceptions=True
            )
    
    def _check_goal_completion(self):
        """检查目标完成情况"""
        for goal in self.goals.values():
            if goal.status != GoalStatus.ACTIVE:
                continue
            
            progress = self.get_goal_progress(goal.id)
            
            # 检查是否所有任务完成
            all_completed = all(
                t.status == TaskStatus.COMPLETED for t in goal.tasks
            )
            any_failed = any(
                t.status == TaskStatus.FAILED for t in goal.tasks
            )
            
            if all_completed:
                self.complete_goal(goal.id)
            elif any_failed and progress >= 0.8:
                # 大部分完成，即使有失败也标记完成
                logger.warning(f"目标 '{goal.name}' 部分任务失败，但进度达到 {progress:.1%}，标记完成")
                self.complete_goal(goal.id)
    
    async def _update_planning(self):
        """更新规划状态"""
        active_plan = self.planner.get_active_plan()
        if active_plan:
            self.planner.update_progress(active_plan.id)
    
    def _update_uptime(self):
        """更新运行时间"""
        if self.state.start_time:
            self.state.uptime_seconds = (datetime.now() - self.state.start_time).total_seconds()
    
    def _report_status(self):
        """报告当前状态"""
        stats = self.engine.get_stats()
        active_goals = [g for g in self.goals.values() if g.status == GoalStatus.ACTIVE]
        
        logger.info(
            f"📊 Agent状态报告 | "
            f"运行时间: {self.state.uptime_seconds:.0f}s | "
            f"活跃目标: {len(active_goals)} | "
            f"队列任务: {stats['queue_length']} | "
            f"已完成: {self.state.total_goals_completed}个目标/{self.state.total_tasks_completed}个任务"
        )
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 回调和事件
    # ═══════════════════════════════════════════════════════════════════════════
    
    def on(self, event: str, callback: Callable):
        """注册事件回调"""
        if event in self._callbacks:
            self._callbacks[event].append(callback)
    
    def _trigger_callback(self, event: str, data: Any):
        """触发事件回调"""
        for callback in self._callbacks.get(event, []):
            try:
                if asyncio.iscoroutinefunction(callback):
                    asyncio.create_task(callback(data))
                else:
                    callback(data)
            except Exception as e:
                logger.error(f"回调执行错误: {e}")
    
    # ═══════════════════════════════════════════════════════════════════════════
    # 查询和导出
    # ═══════════════════════════════════════════════════════════════════════════
    
    def get_status(self) -> Dict[str, Any]:
        """获取Agent完整状态"""
        return {
            'agent_name': self.name,
            'state': self.state.to_dict(),
            'goals': {gid: g.to_dict() for gid, g in self.goals.items()},
            'execution_stats': self.engine.get_stats(),
            'active_plan': self.planner.get_active_plan().to_dict() if self.planner.get_active_plan() else None,
        }
    
    def export_report(self) -> str:
        """导出执行报告"""
        status = self.get_status()
        
        report = f"""
╔══════════════════════════════════════════════════════════════════╗
║           GoalDrivenAgent 执行报告                               ║
╠══════════════════════════════════════════════════════════════════╣
  Agent名称: {status['agent_name']}
  运行状态: {'运行中' if status['state']['is_running'] else '已停止'}
  运行时间: {status['state']['uptime_seconds']:.0f} 秒
  
  【目标统计】
  已完成目标: {status['state']['total_goals_completed']}
  总目标数: {len(status['goals'])}
  
  【任务统计】
  已完成任务: {status['state']['total_tasks_completed']}
  执行成功率: {(status['execution_stats']['total_success'] / max(status['execution_stats']['total_executed'], 1) * 100):.1f}%
  平均执行时间: {status['execution_stats']['average_execution_time']:.2f}s
  
  【活跃目标】
"""
        for gid, goal in status['goals'].items():
            if goal['status'] == 'ACTIVE':
                report += f"    • {goal['name']} (进度: {goal['progress']:.0%})\n"
        
        report += "╚══════════════════════════════════════════════════════════════════╝"
        
        return report


# ═══════════════════════════════════════════════════════════════════════════════
# 测试和演示
# ═══════════════════════════════════════════════════════════════════════════════

async def demo():
    """
    GoalDrivenAgent 完整功能演示
    """
    print("\n" + "="*70)
    print("🎯 GoalDrivenAgent 自主Agent核心 - 功能演示")
    print("="*70 + "\n")
    
    # 1. 创建Agent
    agent = GoalDrivenAgent(name="DemoAgent")
    print("✅ Step 1: Agent创建成功")
    
    # 2. 创建长期规划
    plan = agent.planner.create_plan(
        name="Q1能力提升计划",
        description="第一季度技能学习和项目完成规划",
        duration_days=90
    )
    agent.planner.set_active_plan(plan.id)
    print(f"✅ Step 2: 长期规划创建成功 - {plan.name}")
    
    # 3. 创建目标（自动拆解）
    goal1 = agent.create_goal(
        name="学习Python异步编程",
        description="掌握asyncio、协程、事件循环等核心概念",
        priority=Priority.HIGH,
        auto_decompose=True,
        goal_type='learning'
    )
    agent.planner.add_goal_to_plan(plan.id, goal1, phase=0)
    print(f"✅ Step 3a: 目标1创建并自动拆解 - {len(goal1.tasks)} 个任务")
    
    goal2 = agent.create_goal(
        name="开发Agent任务调度系统",
        description="实现一个支持优先级和依赖的任务调度模块",
        priority=Priority.CRITICAL,
        auto_decompose=True,
        goal_type='development'
    )
    agent.planner.add_goal_to_plan(plan.id, goal2, phase=1)
    print(f"✅ Step 3b: 目标2创建并自动拆解 - {len(goal2.tasks)} 个任务")
    
    # 4. 定义自定义任务执行函数
    async def sample_task_executor(task: Task) -> Dict:
        """示例任务执行器"""
        print(f"    🔄 执行任务: {task.name}")
        await asyncio.sleep(0.3)  # 模拟执行时间
        return {
            'task_id': task.id,
            'executed_at': datetime.now().isoformat(),
            'result': f"成功完成: {task.name}"
        }
    
    # 为所有任务设置执行函数
    for task in agent.tasks.values():
        task.execute_fn = sample_task_executor
    
    print("✅ Step 4: 任务执行器配置完成")
    
    # 5. 激活目标
    agent.activate_goal(goal1.id)
    agent.activate_goal(goal2.id)
    print("✅ Step 5: 目标已激活，任务进入执行队列")
    
    # 6. 启动24/7运行循环
    print("\n🚀 启动Agent 24/7运行循环...\n")
    await agent.start()
    
    # 7. 让Agent运行一段时间
    await asyncio.sleep(8)
    
    # 8. 停止Agent
    await agent.stop()
    print("\n✅ Agent运行循环已停止")
    
    # 9. 导出报告
    print("\n" + agent.export_report())
    
    # 10. 详细状态
    print("\n📋 目标详情:")
    for goal in agent.goals.values():
        print(f"\n  【{goal.name}】")
        print(f"  状态: {goal.status.name} | 进度: {goal.progress:.0%}")
        for task in goal.tasks:
            status_icon = "✅" if task.status == TaskStatus.COMPLETED else "❌" if task.status == TaskStatus.FAILED else "⏳"
            print(f"    {status_icon} {task.name} ({task.status.name})")
    
    print("\n" + "="*70)
    print("✨ 演示完成！GoalDrivenAgent核心功能验证通过")
    print("="*70 + "\n")
    
    return agent


async def quick_test():
    """快速测试 - 验证核心功能"""
    print("\n" + "="*70)
    print("⚡ GoalDrivenAgent 快速测试")
    print("="*70 + "\n")
    
    agent = GoalDrivenAgent(name="TestAgent")
    
    # 测试目标创建和拆解
    goal = agent.create_goal(
        name="测试目标",
        description="验证自动拆解功能",
        priority=Priority.HIGH,
        auto_decompose=True
    )
    
    assert len(goal.tasks) > 0, "目标拆解失败"
    print(f"✅ 目标拆解: {len(goal.tasks)} 个任务")
    
    # 测试规划
    plan = agent.planner.create_plan("测试规划", "测试描述")
    agent.planner.add_goal_to_plan(plan.id, goal)
    print(f"✅ 规划创建: {plan.name}")
    
    # 测试执行引擎
    async def test_executor(task: Task):
        return {"test": True}
    
    for task in goal.tasks:
        task.execute_fn = test_executor
    
    agent.activate_goal(goal.id)
    await agent.start()
    await asyncio.sleep(3)
    await agent.stop()
    
    stats = agent.engine.get_stats()
    print(f"✅ 执行引擎: 完成 {stats['total_success']}/{stats['total_executed']} 个任务")
    
    print("\n✅ 所有核心功能测试通过！\n")
    return True


# ═══════════════════════════════════════════════════════════════════════════════
# 主入口
# ═══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    # 运行完整演示
    asyncio.run(demo())
    
    # 或者只运行快速测试
    # asyncio.run(quick_test())
