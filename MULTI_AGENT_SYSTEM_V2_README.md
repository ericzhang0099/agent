# 优化后的Multi-Agent协作系统
## Multi-Agent Collaboration System v2.0 - Dialogue-Based

---

## 系统概述

本系统解决了传统Multi-Agent架构中的"广播倒转"问题，实现了真正的对话式协作机制。

### 核心改进

1. **网状通信拓扑**：Agent之间可以直接对话，而非必须通过中心节点
2. **深度对话协议**：基于语用学的对话行为类型（Dialogue Acts）
3. **共享黑板系统**：实时共享中间结果和思维过程
4. **动态任务分配**：基于能力匹配和负载均衡的智能分配
5. **实时反馈循环**：持续监控和自适应调整

---

## 架构对比

### 旧架构（广播倒转）

```
Orchestrator ←→ Agent A
      ↑
      ↓
   Agent B
      ↑
      ↓
   Agent C

问题：
- 所有通信经过Orchestrator
- Agent之间无法直接交流
- 信息孤岛，重复工作
- 单向广播，非真正协作
```

### 新架构（对话式协作）

```
      Facilitator
     (轻量协调者)
          │
    ┌─────┼─────┐
    ↓     ↓     ↓
  Agent A←→Agent B←→Agent C
    │       │       │
    └───────┼───────┘
            ↓
      Shared Blackboard
      (共享黑板)

优势：
- Agent间直接对话
- 网状信息流动
- 实时共享中间结果
- 真正的多向协作
```

---

## 核心组件

### 1. 对话引擎（Dialogue Engine）

```python
from multi_agent_dialogue_system import (
    DialogueAgent, DialogueMessage, DialogueAct,
    FacilitatorAgent, Blackboard
)
```

**对话行为类型（Dialogue Acts）**：
- **信息类**：ASSERT, INFORM, EXPLAIN
- **询问类**：QUESTION, CLARIFY, CONFIRM
- **反馈类**：AGREE, DISAGREE, ACKNOWLEDGE
- **协作类**：SUGGEST, ELABORATE, CORRECT
- **元对话类**：META_DISCUSS, HANDOVER, SUMMARIZE

### 2. 共享黑板（Shared Blackboard）

```python
blackboard = Blackboard(session_id="session_001")

# 写入黑板
await blackboard.write(
    section="workspace",
    key="hypotheses",
    value={"statement": "AI will transform education", "confidence": 0.9},
    agent_id="researcher_1"
)

# 读取黑板
hypotheses = await blackboard.read("workspace", "hypotheses")
```

**黑板分区**：
- **问题定义区**：原始问题、约束条件、成功标准
- **工作区**：假设空间、部分解、待解决问题
- **知识区**：事实、规则、知识图谱
- **控制区**：当前焦点、议程、贡献者状态

### 3. 动态任务分配器

```python
allocator = DynamicTaskAllocator()

# 任务分解
subtasks = allocator.decompose(task)

# 能力匹配
candidates = allocator.matchCapability(subtask, agents)

# 负载均衡
selected = allocator.balanceLoad(candidates, current_loads)
```

### 4. 实时反馈循环

```python
feedback_loop = RealTimeFeedbackLoop(blackboard, agents)

# 监控执行
await feedback_loop.monitor(execution)

# 检测偏差
deviation = feedback_loop.detectDeviation(expected, actual)

# 触发调整
if deviation:
    await feedback_loop.triggerAdjustment(deviation, context)
```

---

## 快速开始

### 安装

```bash
# 依赖
pip install asyncio
```

### 基础使用

```python
import asyncio
from multi_agent_dialogue_system import (
    FacilitatorAgent, ResearchAgent, CreativeAgent, CriticalAgent
)

async def main():
    # 创建协调者
    facilitator = FacilitatorAgent("facilitator_1")
    
    # 创建参与者
    researcher = ResearchAgent("researcher_1")
    creative = CreativeAgent("creative_1")
    critic = CriticalAgent("critic_1")
    
    # 注册参与者
    facilitator.register_participant(researcher)
    facilitator.register_participant(creative)
    facilitator.register_participant(critic)
    
    # 运行对话
    result = await facilitator.run_dialogue(
        topic="Design a new AI-powered education platform",
        objective="Generate a comprehensive design proposal",
        max_rounds=5
    )
    
    print(f"Result: {result['result']}")
    print(f"Consensus: {result['consensus_reached']}")

if __name__ == "__main__":
    asyncio.run(main())
```

---

## Agent角色定义

### 内置角色

| 角色 | 类名 | 专长 | 典型行为 |
|------|------|------|----------|
| 研究者 | ResearchAgent | research, analysis, fact_checking | ASSERT, EXPLAIN, ELABORATE |
| 创意者 | CreativeAgent | brainstorming, ideation, innovation | SUGGEST, ELABORATE |
| 批判者 | CriticalAgent | evaluation, risk_assessment | DISAGREE, QUESTION, SUGGEST |
| 协调者 | FacilitatorAgent | coordination, facilitation | SUMMARIZE, META_DISCUSS |

### 自定义Agent

```python
from multi_agent_dialogue_system import DialogueAgent, DialogueContext, DialogueMessage

class MyCustomAgent(DialogueAgent):
    def __init__(self, agent_id: str):
        super().__init__(
            agent_id,
            "custom_role",
            ["specialty_1", "specialty_2"]
        )
    
    async def generate_response(self, context: DialogueContext) -> DialogueMessage:
        # 根据上下文生成回应
        return DialogueMessage(
            speaker_id=self.agent_id,
            speaker_role=self.role,
            dialogue_act=DialogueAct.ASSERT,
            content="My response...",
            confidence=0.85
        )
    
    async def evaluate_message(self, message: DialogueMessage) -> float:
        # 评估消息与专长的相关度
        return 0.8
```

---

## 对话规则

### 内置规则

1. **必须回应直接提问**：被@提及的Agent必须回应
2. **不同意必须给理由**：DISAGREE必须附带证据
3. **陈述必须有置信度**：ASSERT必须提供confidence
4. **参与度保证**：不能连续多轮不发言
5. **引用必须准确**：引用的内容必须真实存在

### 自定义规则

```python
from multi_agent_dialogue_system import DialogueRules

class MyDialogueRules(DialogueRules):
    @staticmethod
    def custom_rule(message: DialogueMessage) -> bool:
        # 自定义验证逻辑
        return True
```

---

## 工作流模式

### 1. 协商式决策

```python
# 多个Agent就某个问题达成共识
result = await facilitator.run_dialogue(
    topic="Should we adopt microservices architecture?",
    objective="Reach a consensus decision with rationale",
    max_rounds=8
)
```

### 2. 头脑风暴

```python
# 创意型Agent主导，其他Agent补充和扩展
facilitator.register_participant(creative_agent, role="leader")
facilitator.register_participant(research_agent, role="supporter")
facilitator.register_participant(critic_agent, role="evaluator")

result = await facilitator.run_dialogue(
    topic="New features for our product",
    objective="Generate 10 innovative feature ideas",
    max_rounds=6
)
```

### 3. 问题解决

```python
# 针对具体问题，各Agent从不同角度分析
result = await facilitator.run_dialogue(
    topic="System performance degradation",
    objective="Identify root cause and propose solutions",
    max_rounds=10
)
```

---

## 性能优化

### 并发执行

```python
# 多个对话会话并行
sessions = [
    facilitator.run_dialogue(topic=t, objective=o, max_rounds=5)
    for t, o in topics
]
results = await asyncio.gather(*sessions)
```

### 缓存机制

```python
# 黑板支持异步读写缓存
await blackboard.write(section, key, value, agent_id, cache_ttl=300)
```

### 早期终止

```python
# 当达成共识时自动终止
result = await facilitator.run_dialogue(
    topic="...",
    objective="...",
    max_rounds=10,
    early_termination=True  # 达成共识时提前结束
)
```

---

## 监控与调试

### 对话日志

```python
# 启用详细日志
import logging
logging.basicConfig(level=logging.DEBUG)

# 查看对话历史
for message in facilitator.dialogue_history:
    print(f"[{message.speaker_id}] {message.dialogue_act}: {message.content}")
```

### 黑板状态

```python
# 获取黑板摘要
print(blackboard.get_summary())

# 导出完整状态
state = {
    "problem": blackboard.problem,
    "workspace": blackboard.workspace,
    "control": blackboard.control
}
```

### 性能指标

```python
# 收集性能指标
metrics = {
    "total_rounds": result["rounds"],
    "messages_per_agent": {
        agent_id: agent.participation_count
        for agent_id, agent in facilitator.participants.items()
    },
    "consensus_reached": result["consensus_reached"],
    "execution_time": execution_time
}
```

---

## 集成指南

### 与现有系统集成

```python
from multi_agent_dialogue_system import DialogueAgent

class MyExistingAgent(DialogueAgent):
    def __init__(self, existing_agent):
        super().__init__(
            existing_agent.id,
            existing_agent.role,
            existing_agent.capabilities
        )
        self.existing = existing_agent
    
    async def generate_response(self, context):
        # 调用现有Agent的能力
        response = await self.existing.process(context)
        
        return DialogueMessage(
            speaker_id=self.agent_id,
            dialogue_act=DialogueAct.ASSERT,
            content=response
        )
```

### 与LLM集成

```python
class LLMAgent(DialogueAgent):
    def __init__(self, agent_id: str, llm_client):
        super().__init__(agent_id, "llm_agent", ["reasoning", "generation"])
        self.llm = llm_client
    
    async def generate_response(self, context):
        # 构建LLM提示
        prompt = self.build_prompt(context)
        
        # 调用LLM
        response = await self.llm.complete(prompt)
        
        # 解析对话行为
        dialogue_act = self.parse_dialogue_act(response)
        
        return DialogueMessage(
            speaker_id=self.agent_id,
            dialogue_act=dialogue_act,
            content=response
        )
```

---

## 最佳实践

### 1. Agent组合

- **研究+创意+批判**：平衡的创新团队
- **多个专家+协调者**：复杂问题求解
- **辩论双方+裁判**：对立观点分析

### 2. 对话轮数

- 简单问题：3-5轮
- 复杂决策：8-10轮
- 创意生成：5-8轮

### 3. 黑板使用

- 及时写入中间结果
- 引用黑板内容时注明来源
- 定期清理过时假设

### 4. 错误处理

```python
try:
    result = await facilitator.run_dialogue(...)
except DialogueTimeout:
    # 处理超时
    pass
except ConsensusImpossible:
    # 处理无法达成共识
    pass
```

---

## 版本历史

- **v2.0** (2026-02-27): 重构为对话式协作架构，解决广播倒转问题
- **v1.0** (2026-02-20): 初始版本，基于星型拓扑的Multi-Agent系统

---

## 许可证

MIT License
