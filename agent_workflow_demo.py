"""
Agent工作流系统 - 快速演示版本
支持6种工作流模式：顺序链式/路由式/评估优化式/并行式/规划式/协作式
"""

import asyncio
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


class WorkflowPattern(Enum):
    SEQUENTIAL = "sequential"
    ROUTER = "router"
    EVALUATOR_OPTIMIZER = "evaluator_optimizer"
    PARALLEL = "parallel"
    PLANNER = "planner"
    COLLABORATIVE = "collaborative"


class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class Task:
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    type: str = ""
    input_data: Dict[str, Any] = field(default_factory=dict)
    output_data: Dict[str, Any] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.PENDING


@dataclass
class Agent:
    id: str
    name: str
    type: str
    capabilities: List[str] = field(default_factory=list)
    
    async def execute(self, task: Task, context: Dict[str, Any]) -> Dict[str, Any]:
        task.status = TaskStatus.RUNNING
        await asyncio.sleep(0.1)  # 模拟执行
        
        result = {
            "agent": self.name,
            "task_type": task.type,
            "output": f"[{self.name}] 完成: {task.type}",
            "timestamp": datetime.now().strftime("%H:%M:%S")
        }
        task.status = TaskStatus.COMPLETED
        return result


class WorkflowExecutor:
    """工作流执行器"""
    
    def __init__(self, agent_registry: Dict[str, Agent]):
        self.agent_registry = agent_registry
    
    async def execute_sequential(self, steps: List[Dict], input_data: Dict) -> Dict:
        """顺序链式执行"""
        print("  → 顺序执行步骤...")
        context = {"input": input_data, "results": {}}
        
        for i, step in enumerate(steps, 1):
            agent = self.agent_registry[step["agent_id"]]
            task = Task(type=step["name"])
            result = await agent.execute(task, context)
            context["results"][step["output_key"]] = result
            print(f"    [{i}/{len(steps)}] {step['name']}: ✓ {agent.name}")
        
        return context["results"]
    
    async def execute_router(self, input_data: Dict, branches: List[Dict]) -> Dict:
        """路由式执行"""
        print("  → 路由决策...")
        # 模拟路由决策
        category = input_data.get("category", "default")
        
        for branch in branches:
            if branch["condition"] == f"category == '{category}'" or branch["condition"] == "default":
                agent = self.agent_registry[branch["target_agent"]]
                task = Task(type="routed_task")
                result = await agent.execute(task, input_data)
                print(f"    路由到: {agent.name} (类别: {category})")
                return {"category": category, "result": result}
        
        return {"category": category}
    
    async def execute_evaluator_optimizer(self, input_data: Dict, max_iter: int = 3) -> Dict:
        """评估优化式执行"""
        print("  → 迭代优化执行...")
        generator = self.agent_registry["A6"]
        evaluator = self.agent_registry["A5"]
        
        best_score = 0
        iteration = 0
        
        for i in range(max_iter):
            iteration += 1
            # 生成
            gen_task = Task(type="generation")
            generated = await generator.execute(gen_task, input_data)
            
            # 评估
            eval_task = Task(type="evaluation")
            evaluation = await evaluator.execute(eval_task, {"generated": generated})
            score = 7.0 + i * 0.5  # 模拟分数提升
            
            print(f"    迭代 {i+1}: 生成→评估 (分数: {score:.1f})")
            
            if score > best_score:
                best_score = score
            
            if score >= 8.5:
                print(f"    ✓ 达到质量阈值，提前停止")
                break
        
        return {"iterations": iteration, "best_score": best_score}
    
    async def execute_parallel(self, steps: List[Dict], input_data: Dict) -> Dict:
        """并行式执行"""
        print("  → 并行执行步骤...")
        
        async def run_step(step):
            agent = self.agent_registry[step["agent_id"]]
            task = Task(type=step["name"])
            result = await agent.execute(task, input_data)
            return step["output_key"], result
        
        # 并行执行
        tasks = [run_step(step) for step in steps]
        results = await asyncio.gather(*tasks)
        
        for key, result in results:
            print(f"    ✓ {key}: {result['agent']}")
        
        # 聚合
        aggregator = self.agent_registry["A10"]
        agg_task = Task(type="aggregation")
        aggregated = await aggregator.execute(agg_task, dict(results))
        print(f"    → 聚合结果: {aggregator.name}")
        
        return {"steps": dict(results), "aggregated": aggregated}
    
    async def execute_planner(self, input_data: Dict) -> Dict:
        """规划式执行"""
        print("  → 规划阶段...")
        planner = self.agent_registry["A2"]
        executor = self.agent_registry["A3"]
        
        # 规划
        plan_task = Task(type="planning")
        plan = await planner.execute(plan_task, input_data)
        print(f"    ✓ 规划完成: {planner.name}")
        
        # 执行
        print("  → 执行阶段...")
        steps = ["需求分析", "架构设计", "开发实现", "测试验证"]
        executed = []
        for step in steps:
            exec_task = Task(type=step)
            result = await executor.execute(exec_task, {})
            executed.append({"step": step, "agent": result["agent"]})
            print(f"    ✓ {step}")
        
        return {"plan": plan, "executed": executed}
    
    async def execute_collaborative(self, input_data: Dict, participants: List[Dict]) -> Dict:
        """协作式执行"""
        print("  → 协作讨论...")
        facilitator = self.agent_registry["A9"]
        
        rounds = 3
        for round_num in range(1, rounds + 1):
            print(f"    第 {round_num} 轮讨论:")
            for p in participants:
                agent = self.agent_registry[p["agent_id"]]
                task = Task(type="collaboration")
                result = await agent.execute(task, input_data)
                print(f"      - {p['role']} ({agent.name}): 贡献想法")
        
        # 检查共识
        consensus_task = Task(type="consensus_check")
        consensus = await facilitator.execute(consensus_task, {})
        print(f"    ✓ 达成共识: {facilitator.name}")
        
        return {"rounds": rounds, "participants": len(participants)}


async def main():
    """演示6种工作流模式"""
    
    print("=" * 70)
    print("🚀 Agent工作流系统 - 6种模式演示")
    print("=" * 70)
    
    # 注册10个Agent
    agents = {
        "A1": Agent("A1", "RouterAgent", "router", ["routing"]),
        "A2": Agent("A2", "PlannerAgent", "planner", ["planning"]),
        "A3": Agent("A3", "ResearchAgent", "researcher", ["research"]),
        "A4": Agent("A4", "WritingAgent", "writer", ["writing"]),
        "A5": Agent("A5", "ReviewAgent", "evaluator", ["review"]),
        "A6": Agent("A6", "CodeAgent", "developer", ["coding"]),
        "A7": Agent("A7", "DataAgent", "analyst", ["data"]),
        "A8": Agent("A8", "MarketAgent", "analyst", ["market"]),
        "A9": Agent("A9", "FacilitatorAgent", "coordinator", ["facilitation"]),
        "A10": Agent("A10", "AggregatorAgent", "aggregator", ["aggregation"]),
    }
    
    executor = WorkflowExecutor(agents)
    
    # 1. 顺序链式
    print("\n📋 【1. 顺序链式工作流 - 内容创作】")
    print("   流程: ResearchAgent → WritingAgent → ReviewAgent")
    result = await executor.execute_sequential([
        {"name": "研究", "agent_id": "A3", "output_key": "research"},
        {"name": "写作", "agent_id": "A4", "output_key": "writing"},
        {"name": "审核", "agent_id": "A5", "output_key": "review"},
    ], {"topic": "AI工作流系统"})
    print(f"   ✓ 完成: 3个步骤顺序执行")
    
    # 2. 路由式
    print("\n🔀 【2. 路由式工作流 - 智能路由】")
    print("   流程: RouterAgent → [TechnicalAgent|SalesAgent|SupportAgent]")
    result = await executor.execute_router(
        {"query": "如何优化代码？", "category": "technical"},
        [
            {"condition": "category == 'technical'", "target_agent": "A6"},
            {"condition": "category == 'sales'", "target_agent": "A4"},
            {"condition": "default", "target_agent": "A9"}
        ]
    )
    print(f"   ✓ 完成: 路由到 {result['result']['agent']}")
    
    # 3. 评估优化式
    print("\n🔄 【3. 评估优化式工作流 - 代码生成优化】")
    print("   流程: CodeAgent ↔ ReviewAgent (迭代优化)")
    result = await executor.execute_evaluator_optimizer(
        {"requirement": "实现快速排序"},
        max_iter=3
    )
    print(f"   ✓ 完成: {result['iterations']}次迭代，最终分数 {result['best_score']:.1f}")
    
    # 4. 并行式
    print("\n⚡ 【4. 并行式工作流 - 市场分析】")
    print("   流程: [DataAgent + MarketAgent] → AggregatorAgent")
    result = await executor.execute_parallel([
        {"name": "趋势分析", "agent_id": "A8", "output_key": "trends"},
        {"name": "竞品分析", "agent_id": "A8", "output_key": "competitors"},
        {"name": "数据分析", "agent_id": "A7", "output_key": "data"},
    ], {"product": "AI助手"})
    print(f"   ✓ 完成: 3个分析并行执行 + 聚合")
    
    # 5. 规划式
    print("\n📊 【5. 规划式工作流 - 项目规划】")
    print("   流程: PlannerAgent → [Step1, Step2, ...] → ExecutorAgent")
    result = await executor.execute_planner({"goal": "开发AI系统", "time": "3个月"})
    print(f"   ✓ 完成: 规划 + {len(result['executed'])}个步骤执行")
    
    # 6. 协作式
    print("\n🤝 【6. 协作式工作流 - 头脑风暴】")
    print("   流程: ResearchAgent ↔ WritingAgent ↔ ReviewAgent (Facilitator协调)")
    result = await executor.execute_collaborative(
        {"topic": "提升用户体验"},
        [
            {"agent_id": "A3", "role": "研究员"},
            {"agent_id": "A4", "role": "创意师"},
            {"agent_id": "A5", "role": "评估师"}
        ]
    )
    print(f"   ✓ 完成: {result['rounds']}轮讨论，{result['participants']}个Agent参与")
    
    # 总结
    print("\n" + "=" * 70)
    print("✅ 所有6种工作流模式执行完成!")
    print("=" * 70)
    print("\n📦 交付物清单:")
    print("   1. 📄 agent-workflow-design.md - 系统设计文档")
    print("   2. ⚙️  agent-workflow-config.yaml - 配置文件")
    print("   3. 🐍 agent_workflow_demo.py - 可运行示例代码")
    print("\n👥 10人Agent团队:")
    print("   A1-Router  A2-Planner  A3-Research  A4-Writing  A5-Review")
    print("   A6-Code    A7-Data     A8-Market    A9-Facilitator  A10-Aggregator")
    print("\n🔄 6种工作流模式:")
    print("   1️⃣  顺序链式    2️⃣  路由式    3️⃣  评估优化式")
    print("   4️⃣  并行式      5️⃣  规划式    6️⃣  协作式")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
