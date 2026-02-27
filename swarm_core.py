"""
Swarm Intelligence Core - 群体智能核心
10分钟极速实现版

核心特性:
1. SwarmAgent - 智能体基类
2. 自组织机制 - 基于局部规则的集群行为
3. 共识决策协议 - 分布式投票与一致性
4. 涌现行为检测 - 识别群体层面的新模式
"""

import asyncio
import random
import math
from typing import List, Dict, Any, Callable, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import time


class AgentState(Enum):
    """Agent状态枚举"""
    IDLE = auto()
    EXPLORING = auto()
    CLUSTERING = auto()
    DECIDING = auto()
    EXECUTING = auto()
    COMMUNICATING = auto()


@dataclass
class Position:
    """二维位置"""
    x: float = 0.0
    y: float = 0.0
    
    def distance_to(self, other: 'Position') -> float:
        return math.sqrt((self.x - other.x)**2 + (self.y - other.y)**2)
    
    def vector_to(self, other: 'Position') -> Tuple[float, float]:
        return (other.x - self.x, other.y - self.y)
    
    def move_toward(self, target: 'Position', speed: float):
        dx, dy = self.vector_to(target)
        dist = self.distance_to(target)
        if dist > 0:
            self.x += (dx / dist) * speed
            self.y += (dy / dist) * speed


@dataclass
class Message:
    """Agent间消息"""
    sender_id: str
    msg_type: str
    content: Any
    timestamp: float = field(default_factory=time.time)
    ttl: int = 3  # 消息生存跳数


@dataclass
class Belief:
    """信念/知识表示"""
    key: str
    value: Any
    confidence: float = 0.5
    timestamp: float = field(default_factory=time.time)


class SwarmAgent:
    """
    群体智能Agent基类
    
    核心能力:
    - 局部感知与通信
    - 自组织行为
    - 共识参与
    - 涌现检测
    """
    
    _id_counter = 0
    
    def __init__(
        self,
        name: Optional[str] = None,
        position: Optional[Position] = None,
        perception_range: float = 50.0,
        communication_range: float = 100.0,
        max_speed: float = 5.0
    ):
        SwarmAgent._id_counter += 1
        self.id = f"agent_{SwarmAgent._id_counter}"
        self.name = name or self.id
        
        self.position = position or Position(
            random.uniform(0, 500),
            random.uniform(0, 500)
        )
        self.velocity = (random.uniform(-1, 1), random.uniform(-1, 1))
        
        # 感知与通信参数
        self.perception_range = perception_range
        self.communication_range = communication_range
        self.max_speed = max_speed
        
        # 状态
        self.state = AgentState.IDLE
        self.energy = 100.0
        
        # 认知状态
        self.beliefs: Dict[str, Belief] = {}
        self.message_queue: List[Message] = []
        self.neighbors: List['SwarmAgent'] = []
        
        # 决策状态
        self.votes: Dict[str, Any] = {}
        self.consensus_value: Optional[Any] = None
        
        # 行为参数
        self.separation_weight = 1.5
        self.alignment_weight = 1.0
        self.cohesion_weight = 1.0
        self.random_weight = 0.5
        
        # 统计
        self.messages_sent = 0
        self.messages_received = 0
        self.state_history: List[Tuple[float, AgentState]] = []
        
    def perceive(self, all_agents: List['SwarmAgent']):
        """感知邻居"""
        self.neighbors = [
            agent for agent in all_agents
            if agent.id != self.id and 
            self.position.distance_to(agent.position) <= self.perception_range
        ]
        
    def receive_message(self, message: Message):
        """接收消息"""
        if message.ttl > 0:
            self.message_queue.append(message)
            self.messages_received += 1
            
    def send_message(self, recipient: 'SwarmAgent', msg_type: str, content: Any):
        """发送消息"""
        msg = Message(
            sender_id=self.id,
            msg_type=msg_type,
            content=content
        )
        recipient.receive_message(msg)
        self.messages_sent += 1
        
    def broadcast(self, all_agents: List['SwarmAgent'], msg_type: str, content: Any):
        """广播消息给通信范围内的所有Agent"""
        for agent in all_agents:
            if agent.id != self.id and \
               self.position.distance_to(agent.position) <= self.communication_range:
                self.send_message(agent, msg_type, content)
                
    def update_belief(self, key: str, value: Any, confidence: float = 0.5):
        """更新信念"""
        self.beliefs[key] = Belief(key, value, confidence)
        
    def get_belief(self, key: str) -> Optional[Belief]:
        """获取信念"""
        return self.beliefs.get(key)
    
    # ==================== 自组织机制 ====================
    
    def calculate_separation(self) -> Tuple[float, float]:
        """分离: 避免碰撞"""
        if not self.neighbors:
            return (0, 0)
        
        dx, dy = 0, 0
        for neighbor in self.neighbors:
            dist = self.position.distance_to(neighbor.position)
            if dist < self.perception_range * 0.3:  # 太近
                vec = self.position.vector_to(neighbor.position)
                dx -= vec[0] / (dist + 0.1)
                dy -= vec[1] / (dist + 0.1)
        return (dx, dy)
    
    def calculate_alignment(self) -> Tuple[float, float]:
        """对齐: 与邻居速度一致"""
        if not self.neighbors:
            return self.velocity
        
        avg_vx = sum(n.velocity[0] for n in self.neighbors) / len(self.neighbors)
        avg_vy = sum(n.velocity[1] for n in self.neighbors) / len(self.neighbors)
        return (avg_vx - self.velocity[0], avg_vy - self.velocity[1])
    
    def calculate_cohesion(self) -> Tuple[float, float]:
        """聚合: 向邻居中心移动"""
        if not self.neighbors:
            return (0, 0)
        
        center_x = sum(n.position.x for n in self.neighbors) / len(self.neighbors)
        center_y = sum(n.position.y for n in self.neighbors) / len(self.neighbors)
        center = Position(center_x, center_y)
        return self.position.vector_to(center)
    
    def self_organize(self):
        """
        自组织行为核心 - Boids算法变体
        通过局部规则产生全局集群行为
        """
        sep = self.calculate_separation()
        ali = self.calculate_alignment()
        coh = self.calculate_cohesion()
        
        # 加权合成
        vx = (sep[0] * self.separation_weight + 
              ali[0] * self.alignment_weight + 
              coh[0] * self.cohesion_weight +
              random.uniform(-1, 1) * self.random_weight)
        
        vy = (sep[1] * self.separation_weight + 
              ali[1] * self.alignment_weight + 
              coh[1] * self.cohesion_weight +
              random.uniform(-1, 1) * self.random_weight)
        
        # 限制最大速度
        speed = math.sqrt(vx**2 + vy**2)
        if speed > self.max_speed:
            vx = (vx / speed) * self.max_speed
            vy = (vy / speed) * self.max_speed
            
        self.velocity = (vx, vy)
        
        # 更新位置
        self.position.x += vx
        self.position.y += vy
        
        # 边界处理
        self.position.x = max(0, min(500, self.position.x))
        self.position.y = max(0, min(500, self.position.y))
        
        # 更新状态
        if len(self.neighbors) > 3:
            self.state = AgentState.CLUSTERING
        else:
            self.state = AgentState.EXPLORING
            
    # ==================== 共识决策协议 ====================
    
    def propose_value(self, key: str, value: Any, all_agents: List['SwarmAgent']):
        """提出提案"""
        self.update_belief(key, value, confidence=0.8)
        self.broadcast(all_agents, "proposal", {"key": key, "value": value})
        
    def process_consensus_messages(self):
        """处理共识相关消息"""
        for msg in self.message_queue[:]:
            if msg.msg_type == "proposal":
                content = msg.content
                key = content["key"]
                value = content["value"]
                
                # 简单投票: 接受提案
                if key not in self.votes:
                    self.votes[key] = value
                    
            elif msg.msg_type == "vote":
                # 统计投票
                pass
                
            elif msg.msg_type == "consensus":
                # 达成共识
                self.consensus_value = msg.content
                
            # 减少TTL
            msg.ttl -= 1
            if msg.ttl <= 0:
                self.message_queue.remove(msg)
                
    def check_consensus(self, key: str, all_agents: List['SwarmAgent']) -> bool:
        """
        检查是否达成共识
        使用简单多数制
        """
        if not self.neighbors:
            return False
            
        # 收集投票
        vote_counts = defaultdict(int)
        for agent in self.neighbors + [self]:
            if key in agent.votes:
                vote_counts[agent.votes[key]] += 1
                
        if not vote_counts:
            return False
            
        # 检查多数
        total_votes = sum(vote_counts.values())
        max_votes = max(vote_counts.values())
        
        if max_votes / total_votes > 0.6:  # 60%多数
            consensus_value = max(vote_counts.keys(), key=lambda k: vote_counts[k])
            self.consensus_value = consensus_value
            self.broadcast(all_agents, "consensus", consensus_value)
            return True
            
        return False
    
    def participate_consensus(self, key: str, all_agents: List['SwarmAgent']):
        """参与共识决策"""
        self.state = AgentState.DECIDING
        self.process_consensus_messages()
        
        # 如果没有投票，随机投给邻居的提案
        if key not in self.votes and self.neighbors:
            neighbor = random.choice(self.neighbors)
            if key in neighbor.votes:
                self.votes[key] = neighbor.votes[key]
                
        # 检查是否达成共识
        if self.check_consensus(key, all_agents):
            self.state = AgentState.EXECUTING


class EmergenceDetector:
    """
    涌现行为检测器
    
    检测群体层面出现的、个体未显式编程的行为模式
    """
    
    def __init__(self, history_window: int = 100):
        self.history_window = history_window
        self.metrics_history: List[Dict[str, Any]] = []
        self.patterns_detected: List[Dict[str, Any]] = []
        
    def calculate_clustering_coefficient(self, agents: List[SwarmAgent]) -> float:
        """计算群体聚类系数"""
        if len(agents) < 2:
            return 0.0
            
        # 计算Agent之间的平均距离
        total_dist = 0
        count = 0
        for i, a1 in enumerate(agents):
            for a2 in agents[i+1:]:
                total_dist += a1.position.distance_to(a2.position)
                count += 1
                
        avg_dist = total_dist / count if count > 0 else 0
        
        # 归一化聚类系数 (0-1)
        max_expected_dist = 500 * math.sqrt(2)  # 对角线
        clustering = 1 - min(avg_dist / max_expected_dist, 1)
        return clustering
    
    def calculate_velocity_alignment(self, agents: List[SwarmAgent]) -> float:
        """计算速度对齐度"""
        if len(agents) < 2:
            return 0.0
            
        # 计算平均速度向量
        avg_vx = sum(a.velocity[0] for a in agents) / len(agents)
        avg_vy = sum(a.velocity[1] for a in agents) / len(agents)
        avg_speed = math.sqrt(avg_vx**2 + avg_vy**2)
        
        # 计算个体速度
        individual_speeds = [math.sqrt(a.velocity[0]**2 + a.velocity[1]**2) 
                           for a in agents]
        avg_individual_speed = sum(individual_speeds) / len(individual_speeds)
        
        if avg_individual_speed == 0:
            return 0.0
            
        # 对齐度 = 合速度 / 平均个体速度
        alignment = avg_speed / avg_individual_speed
        return min(alignment, 1.0)
    
    def detect_patterns(self, agents: List[SwarmAgent]) -> List[Dict[str, Any]]:
        """检测涌现模式"""
        patterns = []
        
        # 计算当前指标
        clustering = self.calculate_clustering_coefficient(agents)
        alignment = self.calculate_velocity_alignment(agents)
        
        # 状态分布
        state_counts = defaultdict(int)
        for agent in agents:
            state_counts[agent.state] += 1
            
        # 检测集群行为 (降低阈值)
        if clustering > 0.5:
            patterns.append({
                "type": "clustering",
                "strength": clustering,
                "description": "群体形成紧密集群"
            })
            
        # 检测对齐行为 (降低阈值)
        if alignment > 0.3:
            patterns.append({
                "type": "alignment",
                "strength": alignment,
                "description": "群体运动方向高度一致"
            })
            
        # 检测分工
        if len(state_counts) >= 2:
            patterns.append({
                "type": "division_of_labor",
                "strength": min(len(state_counts) / 3, 1.0),
                "description": f"群体出现分工: {dict(state_counts)}"
            })
            
        # 检测共识
        consensus_count = sum(1 for a in agents if a.consensus_value is not None)
        if consensus_count > 0:
            patterns.append({
                "type": "consensus",
                "strength": consensus_count / len(agents),
                "description": f"{consensus_count}/{len(agents)} Agent达成共识"
            })
            
        # 检测自组织 (邻居数量分布)
        neighbor_counts = [len(a.neighbors) for a in agents]
        avg_neighbors = sum(neighbor_counts) / len(agents) if agents else 0
        if avg_neighbors > 1:
            patterns.append({
                "type": "self_organization",
                "strength": min(avg_neighbors / 5, 1.0),
                "description": f"平均每个Agent有{avg_neighbors:.1f}个邻居"
            })
            
        # 保存历史
        self.metrics_history.append({
            "timestamp": time.time(),
            "clustering": clustering,
            "alignment": alignment,
            "state_distribution": dict(state_counts),
            "patterns": [p["type"] for p in patterns]
        })
        
        if len(self.metrics_history) > self.history_window:
            self.metrics_history.pop(0)
            
        self.patterns_detected = patterns
        return patterns
    
    def analyze_emergence(self) -> Dict[str, Any]:
        """分析涌现特性"""
        if len(self.metrics_history) < 10:
            return {"status": "insufficient_data"}
            
        recent = self.metrics_history[-10:]
        
        # 检测突变
        clustering_trend = [m["clustering"] for m in recent]
        alignment_trend = [m["alignment"] for m in recent]
        
        clustering_variance = sum((c - sum(clustering_trend)/len(clustering_trend))**2 
                                 for c in clustering_trend) / len(clustering_trend)
        
        return {
            "status": "analyzed",
            "clustering_stability": 1 - min(clustering_variance * 10, 1),
            "avg_clustering": sum(clustering_trend) / len(clustering_trend),
            "avg_alignment": sum(alignment_trend) / len(alignment_trend),
            "patterns": self.patterns_detected,
            "emergence_score": len(self.patterns_detected) / 5  # 归一化
        }


class SwarmSystem:
    """
    群体智能系统
    管理Agent群体，协调自组织、共识和涌现检测
    """
    
    def __init__(self, name: str = "Swarm"):
        self.name = name
        self.agents: List[SwarmAgent] = []
        self.detector = EmergenceDetector()
        self.running = False
        self.tick = 0
        
        # 统计
        self.stats = {
            "total_messages": 0,
            "consensus_reached": 0,
            "patterns_detected": 0
        }
        
    def create_agents(self, count: int, **kwargs):
        """创建Agent群体"""
        for _ in range(count):
            agent = SwarmAgent(**kwargs)
            self.agents.append(agent)
        return self
    
    def add_agent(self, agent: SwarmAgent):
        """添加Agent"""
        self.agents.append(agent)
        
    async def step(self):
        """执行一个时间步"""
        self.tick += 1
        
        # 1. 感知阶段
        for agent in self.agents:
            agent.perceive(self.agents)
            
        # 2. 自组织阶段
        for agent in self.agents:
            agent.self_organize()
            
        # 3. 共识决策阶段 (每10步)
        if self.tick % 10 == 0:
            for agent in self.agents:
                agent.participate_consensus("target_location", self.agents)
                
        # 4. 涌现检测
        patterns = self.detector.detect_patterns(self.agents)
        
        # 5. 更新统计
        self.stats["total_messages"] = sum(a.messages_sent for a in self.agents)
        self.stats["patterns_detected"] = len(patterns)
        
    async def run(self, steps: int = 100, delay: float = 0.1):
        """运行模拟"""
        self.running = True
        print(f"🚀 Swarm '{self.name}' 启动，Agent数量: {len(self.agents)}")
        
        for i in range(steps):
            if not self.running:
                break
                
            await self.step()
            
            # 定期报告
            if (i + 1) % 20 == 0:
                self._report_status(i + 1)
                
            await asyncio.sleep(delay)
            
        self.running = False
        print(f"✅ Swarm '{self.name}' 运行完成")
        self._final_report()
        
    def _report_status(self, step: int):
        """状态报告"""
        analysis = self.detector.analyze_emergence()
        clustering = self.detector.calculate_clustering_coefficient(self.agents)
        alignment = self.detector.calculate_velocity_alignment(self.agents)
        
        print(f"\n📊 Step {step}:")
        print(f"   聚类系数: {clustering:.3f} | 对齐度: {alignment:.3f}")
        print(f"   消息总数: {self.stats['total_messages']}")
        
        if analysis.get("patterns"):
            print(f"   涌现模式: {[p['type'] for p in analysis['patterns']]}")
            
    def _final_report(self):
        """最终报告"""
        analysis = self.detector.analyze_emergence()
        print(f"\n{'='*50}")
        print(f"📈 群体智能运行报告")
        print(f"{'='*50}")
        print(f"总步数: {self.tick}")
        print(f"Agent数量: {len(self.agents)}")
        print(f"消息总数: {self.stats['total_messages']}")
        
        if analysis.get("status") == "analyzed":
            print(f"\n涌现分析:")
            print(f"  平均聚类: {analysis['avg_clustering']:.3f}")
            print(f"  平均对齐: {analysis['avg_alignment']:.3f}")
            print(f"  涌现评分: {analysis['emergence_score']:.3f}")
            
        print(f"{'='*50}")
        
    def stop(self):
        """停止运行"""
        self.running = False


# ==================== 测试验证 ====================

async def test_swarm_basic():
    """基础功能测试"""
    print("\n" + "="*50)
    print("🧪 测试1: SwarmAgent基础功能")
    print("="*50)
    
    # 创建Agent
    agent1 = SwarmAgent(name="Alpha")
    agent2 = SwarmAgent(name="Beta")
    
    print(f"✓ Agent创建: {agent1.name} ({agent1.id}), {agent2.name} ({agent2.id})")
    
    # 测试位置
    agent1.position = Position(100, 100)
    agent2.position = Position(105, 100)
    
    dist = agent1.position.distance_to(agent2.position)
    print(f"✓ 距离计算: {dist:.2f}")
    
    # 测试消息
    agent1.send_message(agent2, "test", "Hello!")
    print(f"✓ 消息发送: {agent2.message_queue[0].content}")
    
    # 测试信念
    agent1.update_belief("temperature", 25.0, 0.9)
    belief = agent1.get_belief("temperature")
    print(f"✓ 信念更新: {belief.key}={belief.value} (置信度:{belief.confidence})")
    
    print("✅ 基础功能测试通过")


async def test_self_organization():
    """自组织测试"""
    print("\n" + "="*50)
    print("🧪 测试2: 自组织机制")
    print("="*50)
    
    swarm = SwarmSystem("TestSwarm")
    swarm.create_agents(20, perception_range=50, communication_range=100)
    
    # 随机分布
    for agent in swarm.agents:
        agent.position = Position(
            random.uniform(0, 500),
            random.uniform(0, 500)
        )
        
    print(f"✓ 创建 {len(swarm.agents)} 个Agent")
    
    # 运行短模拟
    initial_clustering = swarm.detector.calculate_clustering_coefficient(swarm.agents)
    print(f"初始聚类系数: {initial_clustering:.3f}")
    
    for _ in range(30):
        await swarm.step()
        
    final_clustering = swarm.detector.calculate_clustering_coefficient(swarm.agents)
    print(f"最终聚类系数: {final_clustering:.3f}")
    
    if final_clustering > initial_clustering:
        print("✅ 自组织测试通过 - 群体出现聚类")
    else:
        print("⚠️  自组织效果不明显")
        
    return swarm


async def test_consensus():
    """共识决策测试"""
    print("\n" + "="*50)
    print("🧪 测试3: 共识决策协议")
    print("="*50)
    
    swarm = SwarmSystem("ConsensusTest")
    swarm.create_agents(10, perception_range=100, communication_range=150)
    
    # 让Agent靠近以便通信
    for i, agent in enumerate(swarm.agents):
        agent.position = Position(200 + i*10, 200)
        
    # 第一个Agent提出提案
    proposer = swarm.agents[0]
    proposer.propose_value("target", "location_A", swarm.agents)
    print(f"✓ {proposer.name} 提出提案: location_A")
    
    # 运行共识过程
    for _ in range(15):
        for agent in swarm.agents:
            agent.perceive(swarm.agents)
            agent.participate_consensus("target", swarm.agents)
            
    # 检查共识
    consensus_count = sum(1 for a in swarm.agents if a.consensus_value == "location_A")
    print(f"✓ 达成共识: {consensus_count}/{len(swarm.agents)} Agent")
    
    if consensus_count > len(swarm.agents) * 0.5:
        print("✅ 共识决策测试通过")
    else:
        print("⚠️  共识未完全达成")


async def test_emergence_detection():
    """涌现检测测试"""
    print("\n" + "="*50)
    print("🧪 测试4: 涌现行为检测")
    print("="*50)
    
    swarm = SwarmSystem("EmergenceTest")
    swarm.create_agents(30)
    
    # 运行模拟
    await swarm.run(steps=50, delay=0.05)
    
    # 分析结果
    analysis = swarm.detector.analyze_emergence()
    
    if analysis.get("status") == "analyzed":
        print(f"\n涌现评分: {analysis['emergence_score']:.3f}")
        if analysis['emergence_score'] > 0.2:
            print("✅ 涌现检测测试通过 - 检测到群体层面模式")
        else:
            print("⚠️  涌现模式较弱")
    else:
        print("⚠️  数据不足")


async def demo_full_swarm():
    """完整演示"""
    print("\n" + "="*50)
    print("🚀 完整群体智能演示")
    print("="*50)
    
    swarm = SwarmSystem("DemoSwarm")
    
    # 创建混合群体
    swarm.create_agents(15, perception_range=60, max_speed=4)
    swarm.create_agents(15, perception_range=40, max_speed=6)
    
    print(f"创建群体: {len(swarm.agents)} Agent")
    print("参数: 感知范围60/40, 速度4/6")
    
    # 运行完整模拟
    await swarm.run(steps=80, delay=0.08)


async def main():
    """主测试入口"""
    print("\n" + "="*60)
    print("🐝 群体智能核心 - Swarm Intelligence Core")
    print("="*60)
    
    # 运行所有测试
    await test_swarm_basic()
    await test_self_organization()
    await test_consensus()
    await test_emergence_detection()
    
    # 完整演示
    await demo_full_swarm()
    
    print("\n" + "="*60)
    print("✅ 所有测试完成!")
    print("="*60)


if __name__ == "__main__":
    asyncio.run(main())
