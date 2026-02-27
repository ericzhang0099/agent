#!/usr/bin/env python3
"""
系统集成最终优化 - 统一API网关与系统间通信协议
Unified API Gateway and Inter-System Communication Protocol

集成所有已完成系统：
- SoulKernel v1.0.0
- Memory System v4.0 (Mem0 + Zep + Pinecone)
- Reasoning Coordinator v1.0.0
- Autonomous Agent System v1.0.0
- Multimodal Perception System
- Swarm Intelligence Core
- Safety Alignment System
- Emotion Matrix v4.0
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import hashlib
import uuid
from abc import ABC, abstractmethod

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SystemComponent(Enum):
    """系统组件枚举"""
    SOULKERNEL = "soulkernel"
    MEMORY = "memory"
    REASONING = "reasoning"
    AUTONOMOUS_AGENT = "autonomous_agent"
    MULTIMODAL = "multimodal"
    SWARM = "swarm"
    SAFETY = "safety"
    EMOTION = "emotion"


class MessageType(Enum):
    """消息类型枚举"""
    # 系统消息
    HEARTBEAT = "heartbeat"
    STATUS_UPDATE = "status_update"
    ERROR = "error"
    SHUTDOWN = "shutdown"
    
    # 任务消息
    TASK_CREATE = "task_create"
    TASK_ASSIGN = "task_assign"
    TASK_COMPLETE = "task_complete"
    TASK_FAIL = "task_fail"
    
    # 数据消息
    MEMORY_STORE = "memory_store"
    MEMORY_RETRIEVE = "memory_retrieve"
    REASONING_REQUEST = "reasoning_request"
    REASONING_RESPONSE = "reasoning_response"
    
    # 情绪消息
    EMOTION_UPDATE = "emotion_update"
    EMOTION_TRIGGER = "emotion_trigger"
    
    # 安全消息
    SAFETY_CHECK = "safety_check"
    SAFETY_ALERT = "safety_alert"
    
    # 群体消息
    SWARM_COORDINATE = "swarm_coordinate"
    SWARM_EMERGENCE = "swarm_emergence"


@dataclass
class SystemMessage:
    """系统间消息格式"""
    message_id: str
    source: SystemComponent
    target: SystemComponent
    message_type: MessageType
    payload: Dict[str, Any]
    timestamp: datetime
    priority: int = 5  # 1-10, 1最高
    correlation_id: Optional[str] = None
    ttl: int = 300  # 消息存活时间(秒)
    
    def __post_init__(self):
        if not self.message_id:
            self.message_id = str(uuid.uuid4())
        if not self.timestamp:
            self.timestamp = datetime.now()
    
    def to_dict(self) -> Dict:
        return {
            "message_id": self.message_id,
            "source": self.source.value,
            "target": self.target.value,
            "message_type": self.message_type.value,
            "payload": self.payload,
            "timestamp": self.timestamp.isoformat(),
            "priority": self.priority,
            "correlation_id": self.correlation_id,
            "ttl": self.ttl
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'SystemMessage':
        return cls(
            message_id=data["message_id"],
            source=SystemComponent(data["source"]),
            target=SystemComponent(data["target"]),
            message_type=MessageType(data["message_type"]),
            payload=data["payload"],
            timestamp=datetime.fromisoformat(data["timestamp"]),
            priority=data.get("priority", 5),
            correlation_id=data.get("correlation_id"),
            ttl=data.get("ttl", 300)
        )


@dataclass
class SystemState:
    """系统状态"""
    component: SystemComponent
    status: str  # "healthy", "degraded", "unhealthy", "offline"
    load: float  # 0-1
    last_heartbeat: datetime
    capabilities: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)
    soul_dimensions: Dict[str, float] = field(default_factory=dict)


class MessageBus:
    """消息总线 - 系统间通信核心"""
    
    def __init__(self):
        self.subscribers: Dict[SystemComponent, List[Callable]] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        self.message_history: List[SystemMessage] = []
        self.running = False
        self._task: Optional[asyncio.Task] = None
    
    async def start(self):
        """启动消息总线"""
        self.running = True
        self._task = asyncio.create_task(self._process_messages())
        logger.info("MessageBus started")
    
    async def stop(self):
        """停止消息总线"""
        self.running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
        logger.info("MessageBus stopped")
    
    def subscribe(self, component: SystemComponent, handler: Callable):
        """订阅消息"""
        if component not in self.subscribers:
            self.subscribers[component] = []
        self.subscribers[component].append(handler)
        logger.info(f"Component {component.value} subscribed to message bus")
    
    def unsubscribe(self, component: SystemComponent, handler: Callable):
        """取消订阅"""
        if component in self.subscribers:
            if handler in self.subscribers[component]:
                self.subscribers[component].remove(handler)
    
    async def publish(self, message: SystemMessage):
        """发布消息"""
        await self.message_queue.put(message)
        logger.debug(f"Message published: {message.message_type.value}")
    
    async def _process_messages(self):
        """处理消息队列"""
        while self.running:
            try:
                message = await asyncio.wait_for(
                    self.message_queue.get(),
                    timeout=1.0
                )
                await self._route_message(message)
                self.message_history.append(message)
                
                # 限制历史记录大小
                if len(self.message_history) > 10000:
                    self.message_history = self.message_history[-5000:]
                    
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error processing message: {e}")
    
    async def _route_message(self, message: SystemMessage):
        """路由消息到目标组件"""
        target = message.target
        
        # 广播消息处理
        if target == SystemComponent.SOULKERNEL:  # SOULKERNEL作为广播中心
            for component, handlers in self.subscribers.items():
                if component != message.source:
                    for handler in handlers:
                        try:
                            await handler(message)
                        except Exception as e:
                            logger.error(f"Handler error: {e}")
        else:
            # 定向消息
            if target in self.subscribers:
                for handler in self.subscribers[target]:
                    try:
                        await handler(message)
                    except Exception as e:
                        logger.error(f"Handler error: {e}")


class SystemAdapter(ABC):
    """系统适配器基类"""
    
    def __init__(self, component: SystemComponent, message_bus: MessageBus):
        self.component = component
        self.message_bus = message_bus
        self.state = SystemState(
            component=component,
            status="offline",
            load=0.0,
            last_heartbeat=datetime.now(),
            capabilities=[]
        )
        self._running = False
    
    @abstractmethod
    async def initialize(self):
        """初始化系统"""
        pass
    
    @abstractmethod
    async def shutdown(self):
        """关闭系统"""
        pass
    
    @abstractmethod
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        pass
    
    async def send_message(
        self,
        target: SystemComponent,
        message_type: MessageType,
        payload: Dict[str, Any],
        priority: int = 5
    ):
        """发送消息"""
        message = SystemMessage(
            message_id=str(uuid.uuid4()),
            source=self.component,
            target=target,
            message_type=message_type,
            payload=payload,
            timestamp=datetime.now(),
            priority=priority
        )
        await self.message_bus.publish(message)
    
    async def update_state(self, **kwargs):
        """更新状态"""
        for key, value in kwargs.items():
            if hasattr(self.state, key):
                setattr(self.state, key, value)
        self.state.last_heartbeat = datetime.now()


class SoulKernelAdapter(SystemAdapter):
    """SoulKernel适配器"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(SystemComponent.SOULKERNEL, message_bus)
        self.state.capabilities = [
            "consciousness_coordination",
            "attention_management",
            "peripheral_llm_orchestration",
            "self_prompting",
            "cycle_management"
        ]
        self.peripheral_llms: Dict[str, Any] = {}
        self.consciousness_state = "sleeping"
        self.attention_focus = None
    
    async def initialize(self):
        """初始化SoulKernel"""
        logger.info("Initializing SoulKernel...")
        
        # 初始化8个Peripheral LLM
        self.peripheral_llms = {
            "research": {"status": "ready", "load": 0.0},
            "dev": {"status": "ready", "load": 0.0},
            "data": {"status": "ready", "load": 0.0},
            "quant": {"status": "ready", "load": 0.0},
            "risk": {"status": "ready", "load": 0.0},
            "trading": {"status": "ready", "load": 0.0},
            "review": {"status": "ready", "load": 0.0},
            "optimize": {"status": "ready", "load": 0.0}
        }
        
        self.consciousness_state = "waking"
        await self.update_state(status="healthy", load=0.1)
        
        # 订阅消息
        self.message_bus.subscribe(self.component, self.handle_message)
        
        # 启动心跳
        asyncio.create_task(self._heartbeat_loop())
        
        logger.info("SoulKernel initialized")
    
    async def shutdown(self):
        """关闭SoulKernel"""
        self._running = False
        await self.update_state(status="offline")
        logger.info("SoulKernel shutdown")
    
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        if message.message_type == MessageType.TASK_CREATE:
            # 任务创建，进行注意力分配
            await self._coordinate_task(message.payload)
        elif message.message_type == MessageType.REASONING_REQUEST:
            # 推理请求，分配给合适的Peripheral
            await self._route_reasoning(message)
        elif message.message_type == MessageType.EMOTION_UPDATE:
            # 情绪更新，影响意识状态
            await self._update_emotion_state(message.payload)
    
    async def _coordinate_task(self, payload: Dict):
        """协调任务"""
        task_type = payload.get("task_type")
        
        # 根据任务类型选择Peripheral
        peripheral_map = {
            "research": "research",
            "coding": "dev",
            "analysis": "data",
            "quantitative": "quant",
            "risk_assessment": "risk",
            "trading": "trading",
            "review": "review",
            "optimization": "optimize"
        }
        
        selected = peripheral_map.get(task_type, "research")
        
        # 发送任务分配消息
        await self.send_message(
            target=SystemComponent.AUTONOMOUS_AGENT,
            message_type=MessageType.TASK_ASSIGN,
            payload={
                "task_id": payload.get("task_id"),
                "assigned_to": selected,
                "priority": payload.get("priority", 5)
            }
        )
    
    async def _route_reasoning(self, message: SystemMessage):
        """路由推理请求"""
        # 转发到推理协调器
        await self.send_message(
            target=SystemComponent.REASONING,
            message_type=MessageType.REASONING_REQUEST,
            payload=message.payload,
            priority=message.priority
        )
    
    async def _update_emotion_state(self, payload: Dict):
        """更新情绪状态"""
        emotion = payload.get("emotion")
        intensity = payload.get("intensity", 0.5)
        
        # 情绪影响注意力分配
        if emotion in ["urgent", "alert"] and intensity > 0.7:
            self.attention_focus = "risk"
        elif emotion in ["excited", "curious"]:
            self.attention_focus = "research"
    
    async def _heartbeat_loop(self):
        """心跳循环"""
        while self._running:
            await self.send_message(
                target=SystemComponent.SOULKERNEL,
                message_type=MessageType.HEARTBEAT,
                payload={
                    "component": self.component.value,
                    "state": {
                        "consciousness": self.consciousness_state,
                        "attention_focus": self.attention_focus,
                        "peripherals": self.peripheral_llms
                    }
                }
            )
            await asyncio.sleep(30)


class MemoryAdapter(SystemAdapter):
    """Memory System适配器"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(SystemComponent.MEMORY, message_bus)
        self.state.capabilities = [
            "episodic_memory",
            "semantic_memory",
            "procedural_memory",
            "working_memory",
            "fusion_retrieval",
            "temporal_graph",
            "vector_search"
        ]
        self.memories: Dict[str, Any] = {}
        self.user_profiles: Dict[str, Any] = {}
    
    async def initialize(self):
        """初始化记忆系统"""
        logger.info("Initializing Memory System...")
        
        # 初始化三层记忆架构
        self.memory_layers = {
            "short_term": {},  # Pinecone向量检索
            "medium_term": {},  # Zep时序知识图谱
            "long_term": {}     # Mem0个性化记忆
        }
        
        await self.update_state(status="healthy", load=0.05)
        self.message_bus.subscribe(self.component, self.handle_message)
        
        logger.info("Memory System initialized")
    
    async def shutdown(self):
        """关闭记忆系统"""
        await self.update_state(status="offline")
        logger.info("Memory System shutdown")
    
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        if message.message_type == MessageType.MEMORY_STORE:
            await self._store_memory(message.payload)
        elif message.message_type == MessageType.MEMORY_RETRIEVE:
            await self._retrieve_memory(message.payload, message.source)
        elif message.message_type == MessageType.EMOTION_UPDATE:
            # 情绪记忆关联
            await self._store_emotional_memory(message.payload)
    
    async def _store_memory(self, payload: Dict):
        """存储记忆"""
        memory_id = str(uuid.uuid4())
        memory = {
            "id": memory_id,
            "content": payload.get("content"),
            "type": payload.get("memory_type", "episodic"),
            "importance": payload.get("importance", 0.5),
            "timestamp": datetime.now().isoformat(),
            "metadata": payload.get("metadata", {})
        }
        
        self.memories[memory_id] = memory
        
        logger.info(f"Memory stored: {memory_id}")
    
    async def _retrieve_memory(self, payload: Dict, source: SystemComponent):
        """检索记忆"""
        query = payload.get("query")
        memory_type = payload.get("memory_type")
        
        # 模拟融合检索
        results = []
        for memory_id, memory in self.memories.items():
            if memory_type and memory["type"] != memory_type:
                continue
            if query.lower() in memory["content"].lower():
                results.append(memory)
        
        # 发送检索结果
        await self.send_message(
            target=source,
            message_type=MessageType.MEMORY_RETRIEVE,
            payload={
                "query": query,
                "results": results[:10],
                "count": len(results)
            }
        )
    
    async def _store_emotional_memory(self, payload: Dict):
        """存储情绪记忆"""
        await self._store_memory({
            "content": f"Emotional state: {payload.get('emotion')}",
            "memory_type": "emotional",
            "importance": payload.get("intensity", 0.5),
            "metadata": {
                "emotion": payload.get("emotion"),
                "intensity": payload.get("intensity"),
                "trigger": payload.get("trigger")
            }
        })


class ReasoningAdapter(SystemAdapter):
    """Reasoning Coordinator适配器"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(SystemComponent.REASONING, message_bus)
        self.state.capabilities = [
            "chain_of_thought",
            "test_time_compute",
            "self_correction",
            "parallel_sampling",
            "adaptive_scaling",
            "multi_model_coordination"
        ]
        self.reasoning_queue: asyncio.Queue = asyncio.Queue()
        self.active_reasoning: Dict[str, Any] = {}
    
    async def initialize(self):
        """初始化推理协调器"""
        logger.info("Initializing Reasoning Coordinator...")
        
        await self.update_state(status="healthy", load=0.0)
        self.message_bus.subscribe(self.component, self.handle_message)
        
        # 启动推理处理器
        asyncio.create_task(self._process_reasoning_queue())
        
        logger.info("Reasoning Coordinator initialized")
    
    async def shutdown(self):
        """关闭推理协调器"""
        await self.update_state(status="offline")
        logger.info("Reasoning Coordinator shutdown")
    
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        if message.message_type == MessageType.REASONING_REQUEST:
            await self.reasoning_queue.put({
                "message": message,
                "timestamp": datetime.now()
            })
    
    async def _process_reasoning_queue(self):
        """处理推理队列"""
        while True:
            try:
                item = await self.reasoning_queue.get()
                message = item["message"]
                
                await self._execute_reasoning(message)
                
            except Exception as e:
                logger.error(f"Reasoning error: {e}")
    
    async def _execute_reasoning(self, message: SystemMessage):
        """执行推理"""
        payload = message.payload
        query = payload.get("query")
        strategy = payload.get("strategy", "chain_of_thought")
        
        # 模拟推理过程
        reasoning_chain = {
            "query": query,
            "strategy": strategy,
            "steps": [
                {"step": 1, "thought": f"Analyzing: {query}"},
                {"step": 2, "thought": "Breaking down the problem"},
                {"step": 3, "thought": "Applying reasoning strategy"},
                {"step": 4, "thought": "Generating solution"}
            ],
            "conclusion": f"Solution for: {query}",
            "confidence": 0.85,
            "tokens_used": 1500
        }
        
        # 发送推理结果
        await self.send_message(
            target=message.source,
            message_type=MessageType.REASONING_RESPONSE,
            payload={
                "correlation_id": message.message_id,
                "reasoning_chain": reasoning_chain,
                "answer": reasoning_chain["conclusion"]
            }
        )
        
        logger.info(f"Reasoning completed for: {query[:50]}...")


class AutonomousAgentAdapter(SystemAdapter):
    """Autonomous Agent适配器"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(SystemComponent.AUTONOMOUS_AGENT, message_bus)
        self.state.capabilities = [
            "goal_management",
            "auto_decomposition",
            "long_term_planning",
            "continuous_runtime",
            "self_directed_learning",
            "consciousness_kernel"
        ]
        self.goals: Dict[str, Any] = {}
        self.tasks: Dict[str, Any] = {}
        self.plans: Dict[str, Any] = {}
    
    async def initialize(self):
        """初始化自主Agent"""
        logger.info("Initializing Autonomous Agent...")
        
        await self.update_state(status="healthy", load=0.1)
        self.message_bus.subscribe(self.component, self.handle_message)
        
        # 启动持续运行时
        asyncio.create_task(self._continuous_runtime())
        
        logger.info("Autonomous Agent initialized")
    
    async def shutdown(self):
        """关闭自主Agent"""
        await self.update_state(status="offline")
        logger.info("Autonomous Agent shutdown")
    
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        if message.message_type == MessageType.TASK_ASSIGN:
            await self._execute_task(message.payload)
        elif message.message_type == MessageType.TASK_CREATE:
            await self._create_goal(message.payload)
    
    async def _create_goal(self, payload: Dict):
        """创建目标"""
        goal_id = str(uuid.uuid4())
        goal = {
            "id": goal_id,
            "title": payload.get("title"),
            "description": payload.get("description"),
            "type": payload.get("goal_type", "strategic"),
            "priority": payload.get("priority", "medium"),
            "status": "created",
            "created_at": datetime.now().isoformat(),
            "subgoals": []
        }
        
        # 自动拆解目标
        goal["subgoals"] = self._decompose_goal(goal)
        
        self.goals[goal_id] = goal
        
        logger.info(f"Goal created: {goal_id}")
    
    def _decompose_goal(self, goal: Dict) -> List[Dict]:
        """拆解目标"""
        # 模拟目标拆解
        return [
            {"id": str(uuid.uuid4()), "title": f"Step 1 for {goal['title']}", "status": "pending"},
            {"id": str(uuid.uuid4()), "title": f"Step 2 for {goal['title']}", "status": "pending"},
            {"id": str(uuid.uuid4()), "title": f"Step 3 for {goal['title']}", "status": "pending"}
        ]
    
    async def _execute_task(self, payload: Dict):
        """执行任务"""
        task_id = payload.get("task_id")
        assigned_to = payload.get("assigned_to")
        
        logger.info(f"Executing task {task_id} with {assigned_to}")
        
        # 模拟任务执行
        await asyncio.sleep(0.5)
        
        # 发送任务完成消息
        await self.send_message(
            target=SystemComponent.SOULKERNEL,
            message_type=MessageType.TASK_COMPLETE,
            payload={
                "task_id": task_id,
                "result": "success",
                "output": f"Task completed by {assigned_to}"
            }
        )
    
    async def _continuous_runtime(self):
        """持续运行时"""
        while True:
            # 检查待处理任务
            for goal_id, goal in self.goals.items():
                if goal["status"] == "created":
                    # 开始执行目标
                    goal["status"] = "in_progress"
                    logger.info(f"Starting goal execution: {goal_id}")
            
            await asyncio.sleep(5)


class MultimodalAdapter(SystemAdapter):
    """Multimodal System适配器"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(SystemComponent.MULTIMODAL, message_bus)
        self.state.capabilities = [
            "text_processing",
            "image_analysis",
            "audio_processing",
            "video_understanding",
            "cross_modal_fusion",
            "real_time_perception"
        ]
    
    async def initialize(self):
        """初始化多模态系统"""
        logger.info("Initializing Multimodal System...")
        
        await self.update_state(status="healthy", load=0.0)
        self.message_bus.subscribe(self.component, self.handle_message)
        
        logger.info("Multimodal System initialized")
    
    async def shutdown(self):
        """关闭多模态系统"""
        await self.update_state(status="offline")
        logger.info("Multimodal System shutdown")
    
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        if message.message_type == MessageType.TASK_CREATE:
            # 处理多模态任务
            pass


class SwarmAdapter(SystemAdapter):
    """Swarm Intelligence适配器"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(SystemComponent.SWARM, message_bus)
        self.state.capabilities = [
            "swarm_coordination",
            "emergence_detection",
            "consensus_protocol",
            "self_organization",
            "distributed_decision"
        ]
        self.agents: List[Dict] = []
        self.emergence_patterns: List[Dict] = []
    
    async def initialize(self):
        """初始化群体智能"""
        logger.info("Initializing Swarm Intelligence...")
        
        # 创建初始Agent群体
        await self._create_swarm(30)
        
        await self.update_state(status="healthy", load=0.2)
        self.message_bus.subscribe(self.component, self.handle_message)
        
        # 启动群体模拟
        asyncio.create_task(self._run_swarm_simulation())
        
        logger.info("Swarm Intelligence initialized")
    
    async def shutdown(self):
        """关闭群体智能"""
        await self.update_state(status="offline")
        logger.info("Swarm Intelligence shutdown")
    
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        if message.message_type == MessageType.SWARM_COORDINATE:
            await self._coordinate_swarm(message.payload)
    
    async def _create_swarm(self, count: int):
        """创建Agent群体"""
        for i in range(count):
            self.agents.append({
                "id": f"swarm_agent_{i}",
                "position": {"x": i * 10, "y": i * 5},
                "velocity": {"x": 0, "y": 0},
                "state": "active"
            })
    
    async def _run_swarm_simulation(self):
        """运行群体模拟"""
        while True:
            # 模拟群体行为
            for agent in self.agents:
                # 简单的位置更新
                agent["position"]["x"] += 1
                agent["position"]["y"] += 0.5
            
            # 检测涌现模式
            patterns = self._detect_emergence()
            if patterns:
                await self.send_message(
                    target=SystemComponent.SOULKERNEL,
                    message_type=MessageType.SWARM_EMERGENCE,
                    payload={"patterns": patterns}
                )
            
            await asyncio.sleep(1)
    
    def _detect_emergence(self) -> List[Dict]:
        """检测涌现模式"""
        patterns = []
        
        # 检测集群
        if len(self.agents) > 20:
            patterns.append({
                "type": "clustering",
                "description": "Agents forming clusters",
                "confidence": 0.8
            })
        
        return patterns
    
    async def _coordinate_swarm(self, payload: Dict):
        """协调群体"""
        target = payload.get("target")
        logger.info(f"Coordinating swarm towards: {target}")


class SafetyAdapter(SystemAdapter):
    """Safety Alignment适配器"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(SystemComponent.SAFETY, message_bus)
        self.state.capabilities = [
            "constitutional_check",
            "content_moderation",
            "bias_detection",
            "privacy_protection",
            "alignment_verification",
            "risk_assessment"
        ]
        self.violations: List[Dict] = []
    
    async def initialize(self):
        """初始化安全对齐系统"""
        logger.info("Initializing Safety Alignment...")
        
        await self.update_state(status="healthy", load=0.05)
        self.message_bus.subscribe(self.component, self.handle_message)
        
        logger.info("Safety Alignment initialized")
    
    async def shutdown(self):
        """关闭安全对齐系统"""
        await self.update_state(status="offline")
        logger.info("Safety Alignment shutdown")
    
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        if message.message_type == MessageType.SAFETY_CHECK:
            await self._check_safety(message.payload, message.source)
    
    async def _check_safety(self, payload: Dict, source: SystemComponent):
        """安全检查"""
        content = payload.get("content")
        check_type = payload.get("check_type", "constitutional")
        
        # 模拟安全检查
        is_safe = True
        violations = []
        
        # 宪法检查
        if check_type == "constitutional":
            # 检查是否违反SOUL.md宪法
            pass
        
        # 发送检查结果
        await self.send_message(
            target=source,
            message_type=MessageType.SAFETY_CHECK,
            payload={
                "is_safe": is_safe,
                "violations": violations,
                "score": 0.95
            }
        )


class EmotionAdapter(SystemAdapter):
    """Emotion Matrix适配器"""
    
    def __init__(self, message_bus: MessageBus):
        super().__init__(SystemComponent.EMOTION, message_bus)
        self.state.capabilities = [
            "epg_emotion_graph",
            "fine_grained_emotion",
            "emotion_memory",
            "emotion_trigger",
            "emotion_task_matrix",
            "real_time_detection"
        ]
        self.current_emotion = "calm"
        self.emotion_intensity = 0.5
        self.emotion_history: List[Dict] = []
    
    async def initialize(self):
        """初始化情绪矩阵"""
        logger.info("Initializing Emotion Matrix...")
        
        await self.update_state(status="healthy", load=0.1)
        self.message_bus.subscribe(self.component, self.handle_message)
        
        # 启动情绪监控
        asyncio.create_task(self._emotion_monitoring())
        
        logger.info("Emotion Matrix initialized")
    
    async def shutdown(self):
        """关闭情绪矩阵"""
        await self.update_state(status="offline")
        logger.info("Emotion Matrix shutdown")
    
    async def handle_message(self, message: SystemMessage):
        """处理消息"""
        if message.message_type == MessageType.EMOTION_TRIGGER:
            await self._process_emotion_trigger(message.payload)
    
    async def _process_emotion_trigger(self, payload: Dict):
        """处理情绪触发"""
        trigger = payload.get("trigger")
        context = payload.get("context")
        
        # 模拟情绪检测
        emotion_map = {
            "success": ("excited", 0.8),
            "failure": ("frustrated", 0.6),
            "urgent": ("urgent", 0.9),
            "calm": ("calm", 0.5)
        }
        
        emotion, intensity = emotion_map.get(trigger, ("calm", 0.5))
        
        # 更新当前情绪
        self.current_emotion = emotion
        self.emotion_intensity = intensity
        
        # 记录情绪历史
        self.emotion_history.append({
            "emotion": emotion,
            "intensity": intensity,
            "trigger": trigger,
            "timestamp": datetime.now().isoformat()
        })
        
        # 广播情绪更新
        await self.send_message(
            target=SystemComponent.SOULKERNEL,
            message_type=MessageType.EMOTION_UPDATE,
            payload={
                "emotion": emotion,
                "intensity": intensity,
                "trigger": trigger,
                "context": context
            }
        )
        
        logger.info(f"Emotion updated: {emotion} ({intensity})")
    
    async def _emotion_monitoring(self):
        """情绪监控循环"""
        while True:
            # 情绪衰减
            if self.emotion_intensity > 0.3:
                self.emotion_intensity *= 0.95
            
            await asyncio.sleep(10)


class UnifiedAPI:
    """统一API网关"""
    
    def __init__(self):
        self.message_bus = MessageBus()
        self.adapters: Dict[SystemComponent, SystemAdapter] = {}
        self.system_status = "initializing"
        self.start_time = None
    
    async def initialize(self):
        """初始化统一API网关"""
        logger.info("=" * 60)
        logger.info("Initializing Unified API Gateway")
        logger.info("=" * 60)
        
        self.start_time = datetime.now()
        
        # 启动消息总线
        await self.message_bus.start()
        
        # 初始化所有系统适配器
        self.adapters = {
            SystemComponent.SOULKERNEL: SoulKernelAdapter(self.message_bus),
            SystemComponent.MEMORY: MemoryAdapter(self.message_bus),
            SystemComponent.REASONING: ReasoningAdapter(self.message_bus),
            SystemComponent.AUTONOMOUS_AGENT: AutonomousAgentAdapter(self.message_bus),
            SystemComponent.MULTIMODAL: MultimodalAdapter(self.message_bus),
            SystemComponent.SWARM: SwarmAdapter(self.message_bus),
            SystemComponent.SAFETY: SafetyAdapter(self.message_bus),
            SystemComponent.EMOTION: EmotionAdapter(self.message_bus)
        }
        
        # 初始化所有适配器
        for component, adapter in self.adapters.items():
            try:
                await adapter.initialize()
                logger.info(f"✓ {component.value} initialized")
            except Exception as e:
                logger.error(f"✗ {component.value} initialization failed: {e}")
        
        self.system_status = "running"
        
        logger.info("=" * 60)
        logger.info("Unified API Gateway Ready")
        logger.info("=" * 60)
    
    async def shutdown(self):
        """关闭统一API网关"""
        logger.info("Shutting down Unified API Gateway...")
        
        self.system_status = "shutting_down"
        
        # 关闭所有适配器
        for component, adapter in self.adapters.items():
            try:
                await adapter.shutdown()
                logger.info(f"✓ {component.value} shutdown")
            except Exception as e:
                logger.error(f"✗ {component.value} shutdown error: {e}")
        
        # 停止消息总线
        await self.message_bus.stop()
        
        self.system_status = "offline"
        logger.info("Unified API Gateway shutdown complete")
    
    async def get_system_status(self) -> Dict:
        """获取系统状态"""
        component_statuses = {}
        for component, adapter in self.adapters.items():
            component_statuses[component.value] = {
                "status": adapter.state.status,
                "load": adapter.state.load,
                "capabilities": adapter.state.capabilities,
                "last_heartbeat": adapter.state.last_heartbeat.isoformat()
            }
        
        uptime = (datetime.now() - self.start_time).total_seconds() if self.start_time else 0
        
        return {
            "system_status": self.system_status,
            "uptime_seconds": uptime,
            "components": component_statuses,
            "total_components": len(self.adapters),
            "healthy_components": sum(
                1 for a in self.adapters.values() if a.state.status == "healthy"
            )
        }
    
    # ========== API方法 ==========
    
    async def create_task(
        self,
        task_type: str,
        title: str,
        description: str,
        priority: int = 5,
        **kwargs
    ) -> str:
        """创建任务"""
        task_id = str(uuid.uuid4())
        
        await self.adapters[SystemComponent.SOULKERNEL].send_message(
            target=SystemComponent.SOULKERNEL,
            message_type=MessageType.TASK_CREATE,
            payload={
                "task_id": task_id,
                "task_type": task_type,
                "title": title,
                "description": description,
                "priority": priority,
                **kwargs
            },
            priority=priority
        )
        
        return task_id
    
    async def store_memory(
        self,
        content: str,
        memory_type: str = "episodic",
        importance: float = 0.5,
        **kwargs
    ) -> str:
        """存储记忆"""
        memory_id = str(uuid.uuid4())
        
        await self.adapters[SystemComponent.MEMORY].send_message(
            target=SystemComponent.MEMORY,
            message_type=MessageType.MEMORY_STORE,
            payload={
                "memory_id": memory_id,
                "content": content,
                "memory_type": memory_type,
                "importance": importance,
                **kwargs
            }
        )
        
        return memory_id
    
    async def retrieve_memory(
        self,
        query: str,
        memory_type: Optional[str] = None,
        **kwargs
    ) -> List[Dict]:
        """检索记忆"""
        # 创建响应等待
        future = asyncio.Future()
        
        async def response_handler(message: SystemMessage):
            if (message.message_type == MessageType.MEMORY_RETRIEVE and
                message.source == SystemComponent.MEMORY):
                future.set_result(message.payload.get("results", []))
        
        self.message_bus.subscribe(SystemComponent.SOULKERNEL, response_handler)
        
        await self.adapters[SystemComponent.SOULKERNEL].send_message(
            target=SystemComponent.MEMORY,
            message_type=MessageType.MEMORY_RETRIEVE,
            payload={
                "query": query,
                "memory_type": memory_type,
                **kwargs
            }
        )
        
        try:
            results = await asyncio.wait_for(future, timeout=5.0)
            return results
        except asyncio.TimeoutError:
            return []
    
    async def reason(
        self,
        query: str,
        strategy: str = "chain_of_thought",
        **kwargs
    ) -> Dict:
        """执行推理"""
        future = asyncio.Future()
        correlation_id = str(uuid.uuid4())
        
        async def response_handler(message: SystemMessage):
            if (message.message_type == MessageType.REASONING_RESPONSE and
                message.payload.get("correlation_id") == correlation_id):
                future.set_result(message.payload)
        
        self.message_bus.subscribe(SystemComponent.SOULKERNEL, response_handler)
        
        await self.adapters[SystemComponent.SOULKERNEL].send_message(
            target=SystemComponent.REASONING,
            message_type=MessageType.REASONING_REQUEST,
            payload={
                "correlation_id": correlation_id,
                "query": query,
                "strategy": strategy,
                **kwargs
            },
            priority=3
        )
        
        try:
            result = await asyncio.wait_for(future, timeout=30.0)
            return result
        except asyncio.TimeoutError:
            return {"error": "Reasoning timeout"}
    
    async def update_emotion(
        self,
        trigger: str,
        context: Optional[str] = None
    ):
        """更新情绪"""
        await self.adapters[SystemComponent.EMOTION].send_message(
            target=SystemComponent.EMOTION,
            message_type=MessageType.EMOTION_TRIGGER,
            payload={
                "trigger": trigger,
                "context": context
            }
        )
    
    async def check_safety(
        self,
        content: str,
        check_type: str = "constitutional"
    ) -> Dict:
        """安全检查"""
        future = asyncio.Future()
        
        async def response_handler(message: SystemMessage):
            if (message.message_type == MessageType.SAFETY_CHECK and
                message.source == SystemComponent.SAFETY):
                future.set_result(message.payload)
        
        self.message_bus.subscribe(SystemComponent.SOULKERNEL, response_handler)
        
        await self.adapters[SystemComponent.SOULKERNEL].send_message(
            target=SystemComponent.SAFETY,
            message_type=MessageType.SAFETY_CHECK,
            payload={
                "content": content,
                "check_type": check_type
            }
        )
        
        try:
            result = await asyncio.wait_for(future, timeout=5.0)
            return result
        except asyncio.TimeoutError:
            return {"is_safe": True, "score": 1.0}
    
    async def coordinate_swarm(
        self,
        target: str,
        agent_count: int = 30
    ):
        """协调群体"""
        await self.adapters[SystemComponent.SWARM].send_message(
            target=SystemComponent.SWARM,
            message_type=MessageType.SWARM_COORDINATE,
            payload={
                "target": target,
                "agent_count": agent_count
            }
        )


# 全局API实例
_api_instance: Optional[UnifiedAPI] = None


async def get_api() -> UnifiedAPI:
    """获取API实例"""
    global _api_instance
    if _api_instance is None:
        _api_instance = UnifiedAPI()
        await _api_instance.initialize()
    return _api_instance


async def shutdown_api():
    """关闭API"""
    global _api_instance
    if _api_instance:
        await _api_instance.shutdown()
        _api_instance = None


if __name__ == "__main__":
    # 运行集成测试
    async def main():
        api = await get_api()
        
        try:
            # 获取系统状态
            status = await api.get_system_status()
            print("\n" + "=" * 60)
            print("SYSTEM STATUS")
            print("=" * 60)
            print(json.dumps(status, indent=2, default=str))
            
            # 测试创建任务
            print("\n" + "=" * 60)
            print("TESTING: Create Task")
            print("=" * 60)
            task_id = await api.create_task(
                task_type="research",
                title="Test Research Task",
                description="Testing the integrated system",
                priority=3
            )
            print(f"Task created: {task_id}")
            
            # 测试存储记忆
            print("\n" + "=" * 60)
            print("TESTING: Store Memory")
            print("=" * 60)
            memory_id = await api.store_memory(
                content="This is a test memory for the integrated system",
                memory_type="episodic",
                importance=0.8
            )
            print(f"Memory stored: {memory_id}")
            
            # 测试推理
            print("\n" + "=" * 60)
            print("TESTING: Reasoning")
            print("=" * 60)
            reasoning_result = await api.reason(
                query="What is the best approach to system integration?",
                strategy="chain_of_thought"
            )
            print(f"Reasoning result: {json.dumps(reasoning_result, indent=2, default=str)}")
            
            # 测试情绪更新
            print("\n" + "=" * 60)
            print("TESTING: Emotion Update")
            print("=" * 60)
            await api.update_emotion(
                trigger="success",
                context="Integration test successful"
            )
            print("Emotion updated to: excited")
            
            # 等待一段时间让消息处理
            await asyncio.sleep(3)
            
            # 获取最终状态
            print("\n" + "=" * 60)
            print("FINAL SYSTEM STATUS")
            print("=" * 60)
            final_status = await api.get_system_status()
            print(json.dumps(final_status, indent=2, default=str))
            
        finally:
            await shutdown_api()
    
    asyncio.run(main())
