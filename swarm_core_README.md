# Swarm Intelligence Core - API文档

## 快速开始

```python
import asyncio
from swarm_core import SwarmSystem, SwarmAgent

# 创建群体系统
swarm = SwarmSystem("MySwarm")

# 创建30个Agent
swarm.create_agents(30)

# 运行模拟
asyncio.run(swarm.run(steps=100, delay=0.1))
```

## 核心类

### SwarmAgent

群体智能Agent基类，实现局部感知、自组织和共识决策。

```python
agent = SwarmAgent(
    name="Agent1",                    # Agent名称
    position=Position(100, 100),      # 初始位置
    perception_range=50.0,            # 感知范围
    communication_range=100.0,        # 通信范围
    max_speed=5.0                     # 最大速度
)
```

#### 主要方法

- `perceive(all_agents)` - 感知邻居
- `send_message(recipient, msg_type, content)` - 发送消息
- `broadcast(all_agents, msg_type, content)` - 广播消息
- `self_organize()` - 执行自组织行为
- `propose_value(key, value, all_agents)` - 提出共识提案
- `participate_consensus(key, all_agents)` - 参与共识决策

### SwarmSystem

群体智能系统，管理Agent群体。

```python
swarm = SwarmSystem(name="SwarmName")
swarm.create_agents(count=50)
await swarm.run(steps=100, delay=0.1)
```

### EmergenceDetector

涌现行为检测器。

```python
detector = EmergenceDetector(history_window=100)
patterns = detector.detect_patterns(agents)
analysis = detector.analyze_emergence()
```

## 自组织机制

基于Boids算法的三个核心规则：

1. **分离(Separation)** - 避免碰撞
2. **对齐(Alignment)** - 与邻居速度一致
3. **聚合(Cohesion)** - 向邻居中心移动

```python
# 调整行为权重
agent.separation_weight = 1.5
agent.alignment_weight = 1.0
agent.cohesion_weight = 1.0
```

## 共识决策协议

简单多数制共识：

```python
# Agent提出提案
agent.propose_value("target", "location_A", all_agents)

# Agent参与共识
agent.participate_consensus("target", all_agents)

# 检查是否达成共识
if agent.consensus_value:
    print(f"达成共识: {agent.consensus_value}")
```

## 涌现模式检测

自动检测的涌现模式：

- `clustering` - 集群行为
- `alignment` - 对齐行为
- `division_of_labor` - 分工
- `consensus` - 共识达成
- `self_organization` - 自组织

```python
patterns = detector.detect_patterns(agents)
for pattern in patterns:
    print(f"检测到: {pattern['type']} - {pattern['description']}")
```

## 运行测试

```bash
python3 swarm_core.py
```

## 架构图

```
┌─────────────────────────────────────────┐
│           SwarmSystem                   │
│  ┌─────────────────────────────────┐   │
│  │      EmergenceDetector          │   │
│  └─────────────────────────────────┘   │
│                                         │
│  ┌─────┐ ┌─────┐ ┌─────┐ ┌─────┐      │
│  │Agent│ │Agent│ │Agent│ │Agent│ ...  │
│  └──┬──┘ └──┬──┘ └──┬──┘ └──┬──┘      │
│     │       │       │       │          │
│     └───────┴───────┴───────┘          │
│              局部通信                    │
└─────────────────────────────────────────┘
```
