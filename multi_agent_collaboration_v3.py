"""
Multi-Agent协作系统 v3.0 - 对话式协作核心模块
解决"广播倒转"问题：93%独白→对话

核心改进：
1. 对话式消息协议（替代单向广播）
2. Agent间深度对话机制
3. 协作质量评估系统
4. 任务分配与反馈循环优化
5. AGENTS.md v2.0完整集成
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Callable, AsyncIterator, Set, Tuple
from datetime import datetime
from enum import Enum, auto
import asyncio
import uuid
import json
import time
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ==================== 枚举定义 ====================

class DialogueType(Enum):
    """对话类型 - 核心改进：区分独白vs对话"""
    MONOLOGUE = "monologue"           # 独白（单向输出）
    DIALOGUE = "dialogue"             # 双向对话
    DISCUSSION = "discussion"         # 多轮讨论
    DEBATE = "debate"                 # 辩论式对话
    NEGOTIATION = "negotiation"       # 协商对话
    BRAINSTORM = "brainstorm"         # 头脑风暴
    FEEDBACK = "feedback"             # 反馈对话
    CLARIFICATION = "clarification"   # 澄清对话


class MessageIntent(Enum):
    """消息意图 - 对话语义理解"""
    INFORM = "inform"           # 通知
    QUERY = "query"             # 询问
    REQUEST = "request"         # 请求
    PROPOSE = "propose"         # 提议
    AGREE = "agree"             # 同意
    DISAGREE = "disagree"       # 不同意
    CLARIFY = "clarify"         # 澄清
    SUMMARIZE = "summarize"     # 总结
    FEEDBACK = "feedback"       # 反馈
    DELEGATE = "delegate"       # 委派
    ESCALATE = "escalate"       # 升级


class CollaborationPhase(Enum):
    """协作阶段"""
    INITIATION = "initiation"       # 发起
    NEGOTIATION = "negotiation"     # 协商
    EXECUTION = "execution"         # 执行
    REVIEW = "review"               # 审查
    CLOSURE = "closure"             # 结束


class AgentRole(Enum):
    """Agent角色 - AGENTS.md v2.0集成"""
    # 战略层
    CEO = "ceo"                     # CEO Agent
    STRATEGIST = "strategist"       # 战略分析师
    VISIONARY = "visionary"         # 愿景规划师
    
    # 协调层
    PROJECT_MANAGER = "project_manager"
    TASK_SCHEDULER = "task_scheduler"
    RESOURCE_ALLOCATOR = "resource_allocator"
    
    # 执行层
    RESEARCHER = "researcher"
    DATA_ANALYST = "data_analyst"
    DEVELOPER = "developer"
    QA_ENGINEER = "qa_engineer"
    DEVOPS = "devops"
    
    # 协作专用
    FACILITATOR = "facilitator"     # 协调者
    MODERATOR = "moderator"         # 主持人
    CRITIC = "critic"               # 批评者
    SYNTHESIZER = "synthesizer"     # 综合者


# ==================== 数据模型 ====================

@dataclass
class SoulState:
    """SOUL状态 - 8维度人格"""
    personality: float = 0.5
    motivations: float = 0.5
    conflict: float = 0.5
    relationships: float = 0.5
    growth: float = 0.5
    emotions: float = 0.5
    backstory: float = 0.5
    curiosity: float = 0.5
    
    def get_dominant(self) -> str:
        """获取主导维度"""
        dims = {
            "personality": self.personality,
            "motivations": self.motivations,
            "conflict": self.conflict,
            "relationships": self.relationships,
            "growth": self.growth,
            "emotions": self.emotions,
            "backstory": self.backstory,
            "curiosity": self.curiosity
        }
        return max(dims, key=dims.get)


@dataclass
class DialogueMessage:
    """对话消息 - 核心数据结构"""
    message_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dialogue_id: str = ""                          # 所属对话ID
    correlation_id: Optional[str] = None           # 关联消息ID
    
    # 参与者
    sender_id: str = ""
    sender_role: Optional[AgentRole] = None
    receiver_id: Optional[str] = None              # None表示广播/多播
    receiver_roles: List[AgentRole] = field(default_factory=list)
    
    # 内容
    content: str = ""
    intent: MessageIntent = MessageIntent.INFORM
    dialogue_type: DialogueType = DialogueType.DIALOGUE
    
    # SOUL表达
    soul_state: Optional[SoulState] = None
    emotion: str = "neutral"
    tone: str = "professional"
    
    # 元数据
    timestamp: datetime = field(default_factory=datetime.now)
    turn_number: int = 0
    collaboration_phase: CollaborationPhase = CollaborationPhase.INITIATION
    
    # 上下文
    context: Dict[str, Any] = field(default_factory=dict)
    references: List[str] = field(default_factory=list)  # 引用的消息ID
    
    # 质量指标
    response_time_ms: Optional[int] = None
    token_count: Optional[int] = None
    
    def is_response_to(self, other: 'DialogueMessage') -> bool:
        """检查是否是对另一条消息的响应"""
        return self.correlation_id == other.message_id
    
    def is_part_of_dialogue(self, dialogue_id: str) -> bool:
        """检查是否属于某个对话"""
        return self.dialogue_id == dialogue_id
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "message_id": self.message_id,
            "dialogue_id": self.dialogue_id,
            "correlation_id": self.correlation_id,
            "sender_id": self.sender_id,
            "sender_role": self.sender_role.value if self.sender_role else None,
            "receiver_id": self.receiver_id,
            "receiver_roles": [r.value for r in self.receiver_roles],
            "content": self.content,
            "intent": self.intent.value,
            "dialogue_type": self.dialogue_type.value,
            "emotion": self.emotion,
            "tone": self.tone,
            "timestamp": self.timestamp.isoformat(),
            "turn_number": self.turn_number,
            "collaboration_phase": self.collaboration_phase.value,
            "context": self.context,
            "references": self.references
        }


@dataclass
class DialogueSession:
    """对话会话"""
    dialogue_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    dialogue_type: DialogueType = DialogueType.DIALOGUE
    
    # 参与者
    participants: Set[str] = field(default_factory=set)
    initiator: str = ""
    facilitator: Optional[str] = None
    
    # 消息历史
    messages: List[DialogueMessage] = field(default_factory=list)
    
    # 状态
    status: str = "active"  # active, paused, completed, terminated
    current_phase: CollaborationPhase = CollaborationPhase.INITIATION
    current_turn: int = 0
    
    # 主题与目标
    topic: str = ""
    goal: str = ""
    
    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    last_activity: datetime = field(default_factory=datetime.now)
    completed_at: Optional[datetime] = None
    
    # 质量指标
    quality_score: Optional[float] = None
    collaboration_metrics: Dict[str, Any] = field(default_factory=dict)
    
    def add_message(self, message: DialogueMessage) -> None:
        """添加消息"""
        message.dialogue_id = self.dialogue_id
        message.turn_number = self.current_turn
        message.collaboration_phase = self.current_phase
        self.messages.append(message)
        self.current_turn += 1
        self.last_activity = datetime.now()
    
    def get_last_message(self) -> Optional[DialogueMessage]:
        """获取最后一条消息"""
        return self.messages[-1] if self.messages else None
    
    def get_messages_by_sender(self, sender_id: str) -> List[DialogueMessage]:
        """获取某发送者的所有消息"""
        return [m for m in self.messages if m.sender_id == sender_id]
    
    def get_dialogue_ratio(self) -> float:
        """计算对话比例（vs独白）"""
        if not self.messages:
            return 0.0
        
        dialogue_count = sum(1 for m in self.messages 
                           if m.dialogue_type in [DialogueType.DIALOGUE, DialogueType.DISCUSSION])
        return dialogue_count / len(self.messages)
    
    def get_response_rate(self) -> float:
        """计算响应率"""
        if len(self.messages) <= 1:
            return 0.0
        
        responded = sum(1 for m in self.messages[1:] if m.correlation_id)
        return responded / (len(self.messages) - 1)


@dataclass
class CollaborationTask:
    """协作任务"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    description: str = ""
    goal: str = ""
    
    # 分配
    assigned_to: List[str] = field(default_factory=list)
    primary_owner: Optional[str] = None
    
    # 状态
    status: str = "pending"  # pending, in_progress, under_review, completed, failed
    priority: int = 3  # 1-5
    
    # 关联对话
    dialogue_session_id: Optional[str] = None
    
    # 输入输出
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    
    # 需求与要求
    requirements: Dict[str, Any] = field(default_factory=dict)
    acceptance_criteria: Dict[str, Any] = field(default_factory=dict)
    
    # 反馈循环
    feedback_history: List[Dict[str, Any]] = field(default_factory=list)
    revision_count: int = 0
    max_revisions: int = 5
    
    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # 质量
    quality_score: Optional[float] = None
    revision_count: int = 0
    max_revisions: int = 5
    
    # 时间
    created_at: datetime = field(default_factory=datetime.now)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    deadline: Optional[datetime] = None
    
    # 质量
    quality_score: Optional[float] = None
    acceptance_criteria: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CollaborationMetrics:
    """协作质量指标"""
    # 对话质量
    dialogue_ratio: float = 0.0           # 对话vs独白比例（目标>70%）
    response_rate: float = 0.0            # 响应率
    avg_response_time_ms: float = 0.0     # 平均响应时间
    
    # 参与均衡
    participation_balance: float = 0.0    # 参与均衡度（基尼系数逆）
    turn_taking_fairness: float = 0.0     # 轮流公平性
    
    # 协作效果
    consensus_rate: float = 0.0           # 共识达成率
    conflict_resolution_rate: float = 0.0 # 冲突解决率
    task_completion_rate: float = 0.0     # 任务完成率
    
    # 创新指标
    idea_diversity: float = 0.0           # 观点多样性
    build_on_others_rate: float = 0.0     # 基于他人观点构建率
    
    # 整体评分
    overall_score: float = 0.0
    
    def is_healthy(self) -> bool:
        """检查协作健康度"""
        return (
            self.dialogue_ratio >= 0.70 and      # >70%对话
            self.response_rate >= 0.80 and       # >80%响应率
            self.participation_balance >= 0.60 and # 参与相对均衡
            self.overall_score >= 0.70           # 整体>70分
        )


# ==================== 对话式协作核心类 ====================

class DialogueManager:
    """对话管理器 - 核心组件"""
    
    def __init__(self):
        self.sessions: Dict[str, DialogueSession] = {}
        self.agent_dialogues: Dict[str, Set[str]] = defaultdict(set)
        self.message_index: Dict[str, DialogueMessage] = {}
        self._lock = asyncio.Lock()
    
    async def create_session(
        self,
        dialogue_type: DialogueType,
        initiator: str,
        participants: List[str],
        topic: str,
        goal: str,
        facilitator: Optional[str] = None
    ) -> DialogueSession:
        """创建对话会话"""
        async with self._lock:
            session = DialogueSession(
                dialogue_type=dialogue_type,
                initiator=initiator,
                participants=set(participants),
                topic=topic,
                goal=goal,
                facilitator=facilitator
            )
            
            self.sessions[session.dialogue_id] = session
            
            for agent_id in participants:
                self.agent_dialogues[agent_id].add(session.dialogue_id)
            
            logger.info(f"Created {dialogue_type.value} session {session.dialogue_id} "
                       f"with {len(participants)} participants")
            
            return session
    
    async def add_message(
        self,
        dialogue_id: str,
        sender_id: str,
        content: str,
        intent: MessageIntent = MessageIntent.INFORM,
        receiver_id: Optional[str] = None,
        correlation_id: Optional[str] = None,
        soul_state: Optional[SoulState] = None
    ) -> DialogueMessage:
        """添加消息到对话"""
        async with self._lock:
            session = self.sessions.get(dialogue_id)
            if not session:
                raise ValueError(f"Dialogue session {dialogue_id} not found")
            
            message = DialogueMessage(
                dialogue_id=dialogue_id,
                sender_id=sender_id,
                receiver_id=receiver_id,
                content=content,
                intent=intent,
                dialogue_type=session.dialogue_type,
                correlation_id=correlation_id,
                soul_state=soul_state,
                emotion=self._infer_emotion(intent),
                turn_number=session.current_turn
            )
            
            session.add_message(message)
            self.message_index[message.message_id] = message
            
            return message
    
    def _infer_emotion(self, intent: MessageIntent) -> str:
        """从意图推断情绪"""
        emotion_map = {
            MessageIntent.INFORM: "neutral",
            MessageIntent.QUERY: "curious",
            MessageIntent.REQUEST: "earnest",
            MessageIntent.PROPOSE: "enthusiastic",
            MessageIntent.AGREE: "pleased",
            MessageIntent.DISAGREE: "concerned",
            MessageIntent.CLARIFY: "thoughtful",
            MessageIntent.SUMMARIZE: "confident",
            MessageIntent.FEEDBACK: "constructive",
            MessageIntent.DELEGATE: "authoritative",
            MessageIntent.ESCALATE: "urgent"
        }
        return emotion_map.get(intent, "neutral")
    
    async def get_session(self, dialogue_id: str) -> Optional[DialogueSession]:
        """获取对话会话"""
        return self.sessions.get(dialogue_id)
    
    async def get_agent_dialogues(self, agent_id: str) -> List[DialogueSession]:
        """获取Agent参与的所有对话"""
        dialogue_ids = self.agent_dialogues.get(agent_id, set())
        return [self.sessions[did] for did in dialogue_ids if did in self.sessions]
    
    async def close_session(
        self,
        dialogue_id: str,
        final_summary: Optional[str] = None
    ) -> None:
        """关闭对话会话"""
        async with self._lock:
            session = self.sessions.get(dialogue_id)
            if session:
                session.status = "completed"
                session.completed_at = datetime.now()
                
                # 计算最终质量指标
                metrics = await self._calculate_session_metrics(session)
                session.collaboration_metrics = metrics.__dict__
                session.quality_score = metrics.overall_score
                
                logger.info(f"Closed session {dialogue_id} with score {metrics.overall_score:.2f}")
    
    async def _calculate_session_metrics(self, session: DialogueSession) -> CollaborationMetrics:
        """计算会话协作指标"""
        metrics = CollaborationMetrics()
        
        if not session.messages:
            return metrics
        
        # 对话比例
        metrics.dialogue_ratio = session.get_dialogue_ratio()
        
        # 响应率
        metrics.response_rate = session.get_response_rate()
        
        # 平均响应时间
        response_times = []
        for i, msg in enumerate(session.messages[1:], 1):
            if msg.correlation_id:
                prev_msg = self.message_index.get(msg.correlation_id)
                if prev_msg:
                    time_diff = (msg.timestamp - prev_msg.timestamp).total_seconds() * 1000
                    response_times.append(time_diff)
        
        if response_times:
            metrics.avg_response_time_ms = sum(response_times) / len(response_times)
        
        # 参与均衡度
        msg_counts = defaultdict(int)
        for msg in session.messages:
            msg_counts[msg.sender_id] += 1
        
        if msg_counts:
            counts = list(msg_counts.values())
            metrics.participation_balance = self._calculate_balance(counts)
        
        # 整体评分
        metrics.overall_score = (
            metrics.dialogue_ratio * 0.3 +
            metrics.response_rate * 0.2 +
            metrics.participation_balance * 0.2 +
            (1.0 if metrics.avg_response_time_ms < 5000 else 0.5) * 0.3
        )
        
        return metrics
    
    def _calculate_balance(self, counts: List[int]) -> float:
        """计算均衡度（1.0为完全均衡）"""
        if not counts or len(counts) <= 1:
            return 1.0
        
        avg = sum(counts) / len(counts)
        variance = sum((c - avg) ** 2 for c in counts) / len(counts)
        std_dev = variance ** 0.5
        
        # 归一化到0-1
        return max(0.0, 1.0 - std_dev / avg) if avg > 0 else 0.0


class CollaborativeAgent(ABC):
    """协作式Agent基类 - 支持深度对话"""
    
    def __init__(self, agent_id: str, role: AgentRole, name: str):
        self.agent_id = agent_id
        self.role = role
        self.name = name
        
        # SOUL状态
        self.soul_state = SoulState()
        
        # 对话能力
        self.dialogue_manager: Optional[DialogueManager] = None
        self.active_dialogues: Set[str] = set()
        
        # 任务
        self.active_tasks: Dict[str, CollaborationTask] = {}
        self.task_history: List[str] = []
        
        # 协作状态
        self.collaboration_style = "cooperative"  # cooperative, competitive, adaptive
        self.preferred_dialogue_types: List[DialogueType] = [
            DialogueType.DIALOGUE,
            DialogueType.DISCUSSION
        ]
        
        # 统计
        self.stats = {
            "messages_sent": 0,
            "messages_received": 0,
            "dialogues_initiated": 0,
            "dialogues_participated": 0,
            "tasks_completed": 0,
            "feedback_given": 0
        }
    
    def set_dialogue_manager(self, manager: DialogueManager):
        """设置对话管理器"""
        self.dialogue_manager = manager
    
    async def initiate_dialogue(
        self,
        dialogue_type: DialogueType,
        participants: List[str],
        topic: str,
        goal: str,
        opening_message: str
    ) -> str:
        """发起对话"""
        if not self.dialogue_manager:
            raise RuntimeError("Dialogue manager not set")
        
        session = await self.dialogue_manager.create_session(
            dialogue_type=dialogue_type,
            initiator=self.agent_id,
            participants=[self.agent_id] + participants,
            topic=topic,
            goal=goal,
            facilitator=self.agent_id if self.role == AgentRole.FACILITATOR else None
        )
        
        self.active_dialogues.add(session.dialogue_id)
        self.stats["dialogues_initiated"] += 1
        
        # 发送开场消息
        await self.send_message(
            dialogue_id=session.dialogue_id,
            content=opening_message,
            intent=MessageIntent.INFORM
        )
        
        return session.dialogue_id
    
    async def send_message(
        self,
        dialogue_id: str,
        content: str,
        intent: MessageIntent = MessageIntent.INFORM,
        receiver_id: Optional[str] = None,
        reply_to: Optional[str] = None
    ) -> DialogueMessage:
        """发送消息"""
        if not self.dialogue_manager:
            raise RuntimeError("Dialogue manager not set")
        
        message = await self.dialogue_manager.add_message(
            dialogue_id=dialogue_id,
            sender_id=self.agent_id,
            content=content,
            intent=intent,
            receiver_id=receiver_id,
            correlation_id=reply_to,
            soul_state=self.soul_state
        )
        
        self.stats["messages_sent"] += 1
        return message
    
    async def respond_to_message(
        self,
        message: DialogueMessage,
        content: str,
        intent: MessageIntent = MessageIntent.INFORM
    ) -> DialogueMessage:
        """响应消息"""
        return await self.send_message(
            dialogue_id=message.dialogue_id,
            content=content,
            intent=intent,
            receiver_id=message.sender_id,
            reply_to=message.message_id
        )
    
    async def join_dialogue(self, dialogue_id: str) -> None:
        """加入对话"""
        self.active_dialogues.add(dialogue_id)
        self.stats["dialogues_participated"] += 1
    
    async def leave_dialogue(self, dialogue_id: str) -> None:
        """离开对话"""
        self.active_dialogues.discard(dialogue_id)
    
    @abstractmethod
    async def process_message(self, message: DialogueMessage) -> Optional[str]:
        """处理接收到的消息 - 子类实现"""
        pass
    
    @abstractmethod
    async def contribute_to_discussion(
        self,
        dialogue_id: str,
        context: Dict[str, Any]
    ) -> str:
        """为讨论做出贡献 - 子类实现"""
        pass
    
    async def provide_feedback(
        self,
        task_id: str,
        feedback: str,
        score: Optional[float] = None
    ) -> None:
        """提供反馈"""
        task = self.active_tasks.get(task_id)
        if task:
            task.feedback_history.append({
                "from": self.agent_id,
                "feedback": feedback,
                "score": score,
                "timestamp": datetime.now().isoformat()
            })
            self.stats["feedback_given"] += 1
    
    def get_collaboration_report(self) -> Dict[str, Any]:
        """获取协作报告"""
        return {
            "agent_id": self.agent_id,
            "name": self.name,
            "role": self.role.value,
            "stats": self.stats.copy(),
            "active_dialogues": len(self.active_dialogues),
            "active_tasks": len(self.active_tasks),
            "soul_dominant": self.soul_state.get_dominant()
        }


class TaskAllocationSystem:
    """任务分配系统 - 优化分配与反馈循环"""
    
    def __init__(self, dialogue_manager: DialogueManager):
        self.dialogue_manager = dialogue_manager
        self.tasks: Dict[str, CollaborationTask] = {}
        self.agent_capabilities: Dict[str, List[str]] = {}
        self.agent_load: Dict[str, int] = defaultdict(int)
        self.allocation_history: List[Dict[str, Any]] = []
    
    def register_agent_capabilities(self, agent_id: str, capabilities: List[str]):
        """注册Agent能力"""
        self.agent_capabilities[agent_id] = capabilities
    
    async def allocate_task(
        self,
        task: CollaborationTask,
        candidate_agents: List[str],
        use_dialogue: bool = True
    ) -> List[str]:
        """分配任务 - 支持对话式协商"""
        
        if use_dialogue and len(candidate_agents) > 1:
            # 使用对话式协商
            return await self._allocate_via_dialogue(task, candidate_agents)
        else:
            # 使用算法分配
            return self._allocate_via_algorithm(task, candidate_agents)
    
    async def _allocate_via_dialogue(
        self,
        task: CollaborationTask,
        candidates: List[str]
    ) -> List[str]:
        """通过对话协商分配"""
        
        # 创建分配协商对话
        session = await self.dialogue_manager.create_session(
            dialogue_type=DialogueType.NEGOTIATION,
            initiator="system",
            participants=candidates,
            topic=f"Task Allocation: {task.task_type}",
            goal="Determine best agent(s) for the task"
        )
        
        dialogue_id = session.dialogue_id
        
        # 发送任务描述
        await self.dialogue_manager.add_message(
            dialogue_id=dialogue_id,
            sender_id="system",
            content=f"Task: {task.description}\nGoal: {task.goal}\n"
                   f"Requirements: {task.requirements}",
            intent=MessageIntent.REQUEST
        )
        
        # 等待Agent响应（实际实现中需要等待）
        # 这里简化处理
        
        # 基于能力和负载选择
        best_agents = self._select_best_agents(task, candidates)
        
        # 记录分配
        task.assigned_to = best_agents
        if best_agents:
            task.primary_owner = best_agents[0]
        
        self.allocation_history.append({
            "task_id": task.task_id,
            "method": "dialogue",
            "assigned_to": best_agents,
            "candidates": candidates,
            "timestamp": datetime.now().isoformat()
        })
        
        return best_agents
    
    def _allocate_via_algorithm(
        self,
        task: CollaborationTask,
        candidates: List[str]
    ) -> List[str]:
        """通过算法分配"""
        best_agents = self._select_best_agents(task, candidates)
        
        task.assigned_to = best_agents
        if best_agents:
            task.primary_owner = best_agents[0]
        
        self.allocation_history.append({
            "task_id": task.task_id,
            "method": "algorithm",
            "assigned_to": best_agents,
            "candidates": candidates
        })
        
        return best_agents
    
    def _select_best_agents(
        self,
        task: CollaborationTask,
        candidates: List[str]
    ) -> List[str]:
        """选择最佳Agent"""
        scores = []
        
        for agent_id in candidates:
            score = self._calculate_aptitude_score(agent_id, task)
            scores.append((agent_id, score))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 选择前N个
        num_needed = task.requirements.get("num_agents", 1)
        return [agent_id for agent_id, _ in scores[:num_needed]]
    
    def _calculate_aptitude_score(self, agent_id: str, task: CollaborationTask) -> float:
        """计算适配分数"""
        score = 0.0
        
        # 能力匹配
        agent_caps = set(self.agent_capabilities.get(agent_id, []))
        task_caps = set(task.requirements.get("capabilities", []))
        if task_caps:
            score += len(agent_caps & task_caps) / len(task_caps) * 0.5
        
        # 负载（越低越好）
        load = self.agent_load.get(agent_id, 0)
        score += (1.0 - min(load / 10, 1.0)) * 0.3
        
        # 历史表现
        history_score = self._get_agent_history_score(agent_id, task.task_type)
        score += history_score * 0.2
        
        return score
    
    def _get_agent_history_score(self, agent_id: str, task_type: str) -> float:
        """获取Agent历史表现分数"""
        relevant = [
            h for h in self.allocation_history
            if agent_id in h.get("assigned_to", []) and h.get("task_type") == task_type
        ]
        
        if not relevant:
            return 0.5
        
        # 简化：基于最近分配次数
        return max(0.0, 1.0 - len(relevant) * 0.1)
    
    async def process_feedback_loop(
        self,
        task: CollaborationTask,
        dialogue_id: Optional[str] = None
    ) -> bool:
        """处理反馈循环"""
        
        if task.revision_count >= task.max_revisions:
            logger.warning(f"Task {task.task_id} reached max revisions")
            return False
        
        # 获取最新反馈
        if not task.feedback_history:
            return True
        
        latest_feedback = task.feedback_history[-1]
        score = latest_feedback.get("score")
        
        # 检查是否满足接受标准
        if score and score >= task.acceptance_criteria.get("min_score", 0.8):
            task.status = "completed"
            task.quality_score = score
            task.completed_at = datetime.now()
            return True
        
        # 需要修订
        task.revision_count += 1
        task.status = "in_progress"
        
        # 如果有对话，在对话中讨论修订
        if dialogue_id:
            await self.dialogue_manager.add_message(
                dialogue_id=dialogue_id,
                sender_id="system",
                content=f"Revision {task.revision_count} needed. "
                       f"Feedback: {latest_feedback.get('feedback')}",
                intent=MessageIntent.FEEDBACK
            )
        
        return False


class CollaborationQualityMonitor:
    """协作质量监控系统"""
    
    def __init__(self, dialogue_manager: DialogueManager):
        self.dialogue_manager = dialogue_manager
        self.metrics_history: List[CollaborationMetrics] = []
        self.alerts: List[Dict[str, Any]] = []
        self.thresholds = {
            "min_dialogue_ratio": 0.70,
            "min_response_rate": 0.80,
            "max_response_time_ms": 10000,
            "min_participation_balance": 0.50
        }
    
    async def evaluate_session(self, dialogue_id: str) -> CollaborationMetrics:
        """评估对话会话"""
        session = await self.dialogue_manager.get_session(dialogue_id)
        if not session:
            raise ValueError(f"Session {dialogue_id} not found")
        
        metrics = await self._calculate_metrics(session)
        self.metrics_history.append(metrics)
        
        # 检查阈值并生成告警
        await self._check_thresholds(dialogue_id, metrics)
        
        return metrics
    
    async def _calculate_metrics(self, session: DialogueSession) -> CollaborationMetrics:
        """计算协作指标"""
        metrics = CollaborationMetrics()
        
        if not session.messages:
            return metrics
        
        # 基础指标
        metrics.dialogue_ratio = session.get_dialogue_ratio()
        metrics.response_rate = session.get_response_rate()
        
        # 响应时间
        response_times = []
        for msg in session.messages[1:]:
            if msg.response_time_ms:
                response_times.append(msg.response_time_ms)
        
        if response_times:
            metrics.avg_response_time_ms = sum(response_times) / len(response_times)
        
        # 参与均衡
        msg_counts = defaultdict(int)
        for msg in session.messages:
            msg_counts[msg.sender_id] += 1
        
        if msg_counts:
            counts = list(msg_counts.values())
            metrics.participation_balance = self._calculate_gini_inverse(counts)
        
        # 轮流公平性
        metrics.turn_taking_fairness = self._calculate_turn_fairness(session.messages)
        
        # 基于他人观点构建率
        metrics.build_on_others_rate = self._calculate_build_on_rate(session.messages)
        
        # 整体评分
        metrics.overall_score = self._calculate_overall_score(metrics)
        
        return metrics
    
    def _calculate_gini_inverse(self, values: List[int]) -> float:
        """计算基尼系数逆（1为完全均衡）"""
        if not values or len(values) <= 1:
            return 1.0
        
        n = len(values)
        sorted_values = sorted(values)
        cumsum = 0
        for i, v in enumerate(sorted_values, 1):
            cumsum += (2 * i - n - 1) * v
        
        gini = cumsum / (n * sum(values)) if sum(values) > 0 else 0
        return 1.0 - gini
    
    def _calculate_turn_fairness(self, messages: List[DialogueMessage]) -> float:
        """计算轮流公平性"""
        if len(messages) < 2:
            return 1.0
        
        # 检查是否有连续多轮同一Agent发言
        consecutive_counts = []
        current_sender = messages[0].sender_id
        count = 1
        
        for msg in messages[1:]:
            if msg.sender_id == current_sender:
                count += 1
            else:
                consecutive_counts.append(count)
                current_sender = msg.sender_id
                count = 1
        
        consecutive_counts.append(count)
        
        # 平均连续发言轮数（越低越公平）
        avg_consecutive = sum(consecutive_counts) / len(consecutive_counts)
        return max(0.0, 1.0 - (avg_consecutive - 1) * 0.2)
    
    def _calculate_build_on_rate(self, messages: List[DialogueMessage]) -> float:
        """计算基于他人观点构建率"""
        if len(messages) < 2:
            return 0.0
        
        build_on_keywords = ["agree", "disagree", "add", "build", "extend", "however", "but", "also"]
        
        build_on_count = 0
        for msg in messages[1:]:
            content_lower = msg.content.lower()
            if any(kw in content_lower for kw in build_on_keywords):
                build_on_count += 1
        
        return build_on_count / (len(messages) - 1)
    
    def _calculate_overall_score(self, metrics: CollaborationMetrics) -> float:
        """计算整体评分"""
        weights = {
            "dialogue_ratio": 0.25,
            "response_rate": 0.20,
            "participation_balance": 0.20,
            "turn_taking_fairness": 0.15,
            "build_on_others_rate": 0.20
        }
        
        score = (
            metrics.dialogue_ratio * weights["dialogue_ratio"] +
            metrics.response_rate * weights["response_rate"] +
            metrics.participation_balance * weights["participation_balance"] +
            metrics.turn_taking_fairness * weights["turn_taking_fairness"] +
            metrics.build_on_others_rate * weights["build_on_others_rate"]
        )
        
        return min(1.0, max(0.0, score))
    
    async def _check_thresholds(self, dialogue_id: str, metrics: CollaborationMetrics):
        """检查阈值"""
        
        if metrics.dialogue_ratio < self.thresholds["min_dialogue_ratio"]:
            self.alerts.append({
                "type": "low_dialogue_ratio",
                "dialogue_id": dialogue_id,
                "value": metrics.dialogue_ratio,
                "threshold": self.thresholds["min_dialogue_ratio"],
                "timestamp": datetime.now().isoformat()
            })
        
        if metrics.response_rate < self.thresholds["min_response_rate"]:
            self.alerts.append({
                "type": "low_response_rate",
                "dialogue_id": dialogue_id,
                "value": metrics.response_rate,
                "threshold": self.thresholds["min_response_rate"]
            })
        
        if metrics.avg_response_time_ms > self.thresholds["max_response_time_ms"]:
            self.alerts.append({
                "type": "slow_response",
                "dialogue_id": dialogue_id,
                "value": metrics.avg_response_time_ms,
                "threshold": self.thresholds["max_response_time_ms"]
            })
    
    def get_health_report(self) -> Dict[str, Any]:
        """获取健康报告"""
        if not self.metrics_history:
            return {"status": "no_data"}
        
        recent = self.metrics_history[-10:]
        
        return {
            "status": "healthy" if all(m.is_healthy() for m in recent) else "degraded",
            "avg_dialogue_ratio": sum(m.dialogue_ratio for m in recent) / len(recent),
            "avg_response_rate": sum(m.response_rate for m in recent) / len(recent),
            "avg_overall_score": sum(m.overall_score for m in recent) / len(recent),
            "active_alerts": len(self.alerts),
            "recent_alerts": self.alerts[-5:]
        }


class MultiAgentCollaborationSystem:
    """Multi-Agent协作系统 v3.0 - 主入口"""
    
    def __init__(self):
        # 核心组件
        self.dialogue_manager = DialogueManager()
        self.task_allocator = TaskAllocationSystem(self.dialogue_manager)
        self.quality_monitor = CollaborationQualityMonitor(self.dialogue_manager)
        
        # Agent注册
        self.agents: Dict[str, CollaborativeAgent] = {}
        
        # 会话跟踪
        self.active_sessions: Dict[str, str] = {}  # task_id -> dialogue_id
        
        logger.info("Multi-Agent Collaboration System v3.0 initialized")
    
    def register_agent(self, agent: CollaborativeAgent):
        """注册Agent"""
        agent.set_dialogue_manager(self.dialogue_manager)
        self.agents[agent.agent_id] = agent
        
        # 注册能力
        capabilities = self._get_role_capabilities(agent.role)
        self.task_allocator.register_agent_capabilities(agent.agent_id, capabilities)
        
        logger.info(f"Registered agent {agent.name} ({agent.role.value})")
    
    def _get_role_capabilities(self, role: AgentRole) -> List[str]:
        """获取角色能力"""
        capability_map = {
            AgentRole.CEO: ["strategic_planning", "decision_making", "coordination"],
            AgentRole.STRATEGIST: ["analysis", "planning", "forecasting"],
            AgentRole.VISIONARY: ["innovation", "roadmap", "trends"],
            AgentRole.PROJECT_MANAGER: ["planning", "tracking", "risk_management"],
            AgentRole.TASK_SCHEDULER: ["scheduling", "optimization", "load_balancing"],
            AgentRole.RESEARCHER: ["research", "analysis", "information_gathering"],
            AgentRole.DATA_ANALYST: ["data_analysis", "visualization", "reporting"],
            AgentRole.DEVELOPER: ["coding", "debugging", "architecture"],
            AgentRole.QA_ENGINEER: ["testing", "quality_assurance", "automation"],
            AgentRole.DEVOPS: ["deployment", "monitoring", "infrastructure"],
            AgentRole.FACILITATOR: ["facilitation", "moderation", "conflict_resolution"],
            AgentRole.CRITIC: ["review", "critique", "quality_check"],
        }
        return capability_map.get(role, ["general"])
    
    async def start_collaborative_task(
        self,
        task: CollaborationTask,
        dialogue_type: DialogueType = DialogueType.DISCUSSION,
        participants: Optional[List[str]] = None
    ) -> Tuple[str, str]:
        """启动协作任务"""
        
        # 确定参与者
        if not participants:
            participants = list(self.agents.keys())
        
        # 分配任务
        assigned = await self.task_allocator.allocate_task(task, participants)
        
        # 创建协作对话
        dialogue_id = await self._create_collaboration_dialogue(
            task, dialogue_type, assigned
        )
        
        task.dialogue_session_id = dialogue_id
        self.active_sessions[task.task_id] = dialogue_id
        
        # 通知Agent
        for agent_id in assigned:
            agent = self.agents.get(agent_id)
            if agent:
                agent.active_tasks[task.task_id] = task
                await agent.join_dialogue(dialogue_id)
        
        logger.info(f"Started collaborative task {task.task_id} with dialogue {dialogue_id}")
        
        return task.task_id, dialogue_id
    
    async def _create_collaboration_dialogue(
        self,
        task: CollaborationTask,
        dialogue_type: DialogueType,
        participants: List[str]
    ) -> str:
        """创建协作对话"""
        
        session = await self.dialogue_manager.create_session(
            dialogue_type=dialogue_type,
            initiator="system",
            participants=participants,
            topic=task.description,
            goal=task.goal
        )
        
        # 发送任务描述
        await self.dialogue_manager.add_message(
            dialogue_id=session.dialogue_id,
            sender_id="system",
            content=f"🎯 **Task**: {task.description}\n"
                   f"📝 **Goal**: {task.goal}\n"
                   f"👥 **Assigned to**: {', '.join(participants)}\n"
                   f"📊 **Priority**: {task.priority}",
            intent=MessageIntent.INFORM
        )
        
        return session.dialogue_id
    
    async def run_collaboration_round(self, dialogue_id: str) -> None:
        """运行一轮协作"""
        session = await self.dialogue_manager.get_session(dialogue_id)
        if not session:
            return
        
        # 获取当前轮次需要参与的Agent
        for agent_id in session.participants:
            agent = self.agents.get(agent_id)
            if not agent or agent_id == "system":
                continue
            
            # 让Agent做出贡献
            contribution = await agent.contribute_to_discussion(
                dialogue_id=dialogue_id,
                context={
                    "topic": session.topic,
                    "goal": session.goal,
                    "previous_messages": [m.to_dict() for m in session.messages[-5:]]
                }
            )
            
            # 发送贡献
            await agent.send_message(
                dialogue_id=dialogue_id,
                content=contribution,
                intent=MessageIntent.PROPOSE
            )
    
    async def evaluate_and_close(self, dialogue_id: str) -> CollaborationMetrics:
        """评估并关闭协作"""
        
        # 评估质量
        metrics = await self.quality_monitor.evaluate_session(dialogue_id)
        
        # 关闭会话
        await self.dialogue_manager.close_session(dialogue_id)
        
        return metrics
    
    def get_system_report(self) -> Dict[str, Any]:
        """获取系统报告"""
        return {
            "version": "3.0",
            "registered_agents": len(self.agents),
            "active_sessions": len(self.active_sessions),
            "agent_details": [
                agent.get_collaboration_report()
                for agent in self.agents.values()
            ],
            "health": self.quality_monitor.get_health_report()
        }


# ==================== 示例实现 ====================

class ExampleCollaborativeAgent(CollaborativeAgent):
    """示例协作Agent"""
    
    async def process_message(self, message: DialogueMessage) -> Optional[str]:
        """处理消息"""
        logger.info(f"[{self.name}] Received: {message.content[:50]}...")
        
        # 根据意图生成响应
        if message.intent == MessageIntent.QUERY:
            return f"Based on my analysis: {message.content}"
        elif message.intent == MessageIntent.REQUEST:
            return f"I'll help with: {message.content}"
        else:
            return f"I understand: {message.content}"
    
    async def contribute_to_discussion(
        self,
        dialogue_id: str,
        context: Dict[str, Any]
    ) -> str:
        """为讨论做出贡献"""
        topic = context.get("topic", "")
        
        # 基于角色生成贡献
        contributions = {
            AgentRole.RESEARCHER: f"Research insight on {topic}: Key findings suggest...",
            AgentRole.DEVELOPER: f"Technical perspective: Implementation should consider...",
            AgentRole.QA_ENGINEER: f"Quality considerations: We need to verify...",
            AgentRole.PROJECT_MANAGER: f"Project view: Timeline and resources required...",
        }
        
        return contributions.get(
            self.role, 
            f"Input from {self.name}: Consider this aspect..."
        )


async def demo_collaboration_system():
    """演示协作系统"""
    
    print("=" * 70)
    print("Multi-Agent Collaboration System v3.0 - Demo")
    print("=" * 70)
    
    # 创建系统
    system = MultiAgentCollaborationSystem()
    
    # 创建Agent团队
    agents = [
        ExampleCollaborativeAgent("agent_pm", AgentRole.PROJECT_MANAGER, "PM_Alex"),
        ExampleCollaborativeAgent("agent_dev", AgentRole.DEVELOPER, "Dev_Ben"),
        ExampleCollaborativeAgent("agent_qa", AgentRole.QA_ENGINEER, "QA_Carol"),
        ExampleCollaborativeAgent("agent_research", AgentRole.RESEARCHER, "Research_David"),
    ]
    
    for agent in agents:
        system.register_agent(agent)
    
    print("\n【1. Agent注册完成】")
    print(f"注册Agent数量: {len(system.agents)}")
    
    # 创建协作任务
    task = CollaborationTask(
        task_type="feature_development",
        description="Implement user authentication system",
        goal="Create secure, scalable auth system with OAuth support",
        requirements={
            "capabilities": ["coding", "security", "testing"],
            "num_agents": 3
        },
        priority=4
    )
    
    print("\n【2. 启动协作任务】")
    task_id, dialogue_id = await system.start_collaborative_task(
        task=task,
        dialogue_type=DialogueType.DISCUSSION,
        participants=["agent_pm", "agent_dev", "agent_qa"]
    )
    print(f"任务ID: {task_id}")
    print(f"对话ID: {dialogue_id}")
    
    # 运行多轮协作
    print("\n【3. 运行协作轮次】")
    for round_num in range(3):
        print(f"\n--- Round {round_num + 1} ---")
        await system.run_collaboration_round(dialogue_id)
        
        # 显示对话状态
        session = await system.dialogue_manager.get_session(dialogue_id)
        print(f"消息数量: {len(session.messages)}")
        print(f"对话比例: {session.get_dialogue_ratio():.2%}")
    
    # 评估并关闭
    print("\n【4. 评估协作质量】")
    metrics = await system.evaluate_and_close(dialogue_id)
    
    print(f"\n协作质量指标:")
    print(f"  - 对话比例: {metrics.dialogue_ratio:.2%} (目标>70%)")
    print(f"  - 响应率: {metrics.response_rate:.2%} (目标>80%)")
    print(f"  - 参与均衡: {metrics.participation_balance:.2%}")
    print(f"  - 整体评分: {metrics.overall_score:.2%}")
    print(f"  - 健康状态: {'✅ 健康' if metrics.is_healthy() else '⚠️ 需改进'}")
    
    # 系统报告
    print("\n【5. 系统报告】")
    report = system.get_system_report()
    print(f"系统版本: {report['version']}")
    print(f"注册Agent: {report['registered_agents']}")
    print(f"健康状态: {report['health']['status']}")
    
    print("\n" + "=" * 70)
    print("Demo completed successfully!")
    print("=" * 70)
    
    return system


if __name__ == "__main__":
    asyncio.run(demo_collaboration_system())
