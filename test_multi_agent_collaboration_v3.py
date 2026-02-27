"""
Multi-Agent协作系统 v3.0 - 全面测试验证套件

测试覆盖：
1. 对话式协作协议
2. Agent间深度对话机制
3. 任务分配与反馈循环
4. 协作质量评估系统
5. AGENTS.md v2.0集成
6. 广播倒转问题解决验证
"""

import asyncio
import pytest
from typing import Dict, Any, List
from datetime import datetime

from multi_agent_collaboration_v3 import (
    DialogueManager, DialogueMessage, DialogueSession, DialogueType,
    MessageIntent, CollaborationTask, CollaborationMetrics,
    SoulState, AgentRole, CollaborationPhase,
    CollaborativeAgent, TaskAllocationSystem, CollaborationQualityMonitor,
    MultiAgentCollaborationSystem, ExampleCollaborativeAgent
)

from agents_v2_integration import (
    AGENTSv2Agent, AgentDefinition, LayerType, WorkflowMode,
    SoulDimensionProfile, AGENTSv2TeamFactory, AGENTSv2CollaborationSystem,
    WorkflowPatternExecutor
)


# ==================== 测试基类 ====================

class TestBase:
    """测试基类"""
    
    @pytest.fixture
    async def dialogue_manager(self):
        """对话管理器fixture"""
        return DialogueManager()
    
    @pytest.fixture
    async def collaboration_system(self):
        """协作系统fixture"""
        return MultiAgentCollaborationSystem()
    
    @pytest.fixture
    async def agents_v2_system(self):
        """AGENTS v2系统fixture"""
        system = AGENTSv2CollaborationSystem()
        system.initialize_standard_team()
        return system


# ==================== 1. 对话式协作协议测试 ====================

class TestDialogueProtocol(TestBase):
    """测试对话式协作协议"""
    
    @pytest.mark.asyncio
    async def test_dialogue_session_creation(self):
        """测试对话会话创建"""
        dm = DialogueManager()
        
        session = await dm.create_session(
            dialogue_type=DialogueType.DISCUSSION,
            initiator="agent_a",
            participants=["agent_a", "agent_b", "agent_c"],
            topic="Test Topic",
            goal="Test Goal"
        )
        
        assert session.dialogue_id is not None
        assert session.dialogue_type == DialogueType.DISCUSSION
        assert len(session.participants) == 3
        assert session.topic == "Test Topic"
        print("✅ 对话会话创建测试通过")
    
    @pytest.mark.asyncio
    async def test_message_adding_and_correlation(self):
        """测试消息添加和关联"""
        dm = DialogueManager()
        
        session = await dm.create_session(
            dialogue_type=DialogueType.DIALOGUE,
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
            topic="Test",
            goal="Test"
        )
        
        # 添加第一条消息
        msg1 = await dm.add_message(
            dialogue_id=session.dialogue_id,
            sender_id="agent_a",
            content="Hello",
            intent=MessageIntent.INFORM
        )
        
        # 添加回复消息
        msg2 = await dm.add_message(
            dialogue_id=session.dialogue_id,
            sender_id="agent_b",
            content="Hi there",
            intent=MessageIntent.INFORM,
            correlation_id=msg1.message_id
        )
        
        assert msg2.correlation_id == msg1.message_id
        assert msg2.is_response_to(msg1)
        assert len(session.messages) == 2
        print("✅ 消息关联测试通过")
    
    @pytest.mark.asyncio
    async def test_dialogue_types(self):
        """测试不同对话类型"""
        dm = DialogueManager()
        
        dialogue_types = [
            DialogueType.DIALOGUE,
            DialogueType.DISCUSSION,
            DialogueType.DEBATE,
            DialogueType.NEGOTIATION,
            DialogueType.BRAINSTORM
        ]
        
        for dtype in dialogue_types:
            session = await dm.create_session(
                dialogue_type=dtype,
                initiator="agent_a",
                participants=["agent_a", "agent_b"],
                topic=f"{dtype.value} test",
                goal="Test"
            )
            assert session.dialogue_type == dtype
        
        print("✅ 对话类型测试通过")
    
    @pytest.mark.asyncio
    async def test_soul_state_in_message(self):
        """测试消息中的SOUL状态"""
        dm = DialogueManager()
        
        soul = SoulState(
            personality=0.8,
            motivations=0.9,
            emotions=0.7
        )
        
        session = await dm.create_session(
            dialogue_type=DialogueType.DIALOGUE,
            initiator="agent_a",
            participants=["agent_a"],
            topic="Test",
            goal="Test"
        )
        
        msg = await dm.add_message(
            dialogue_id=session.dialogue_id,
            sender_id="agent_a",
            content="Test with soul",
            soul_state=soul
        )
        
        assert msg.soul_state is not None
        assert msg.soul_state.personality == 0.8
        print("✅ SOUL状态消息测试通过")


# ==================== 2. 广播倒转问题解决测试 ====================

class TestBroadcastInversionFix(TestBase):
    """测试广播倒转问题修复"""
    
    @pytest.mark.asyncio
    async def test_dialogue_ratio_calculation(self):
        """测试对话比例计算"""
        dm = DialogueManager()
        
        session = await dm.create_session(
            dialogue_type=DialogueType.DISCUSSION,
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
            topic="Test",
            goal="Test"
        )
        
        # 添加对话式消息（双向）
        msg1 = await dm.add_message(
            dialogue_id=session.dialogue_id,
            sender_id="agent_a",
            content="What do you think?",
            intent=MessageIntent.QUERY
        )
        
        msg2 = await dm.add_message(
            dialogue_id=session.dialogue_id,
            sender_id="agent_b",
            content="I think...",
            intent=MessageIntent.INFORM,
            correlation_id=msg1.message_id
        )
        
        # 计算对话比例
        ratio = session.get_dialogue_ratio()
        assert ratio == 1.0  # 所有消息都是对话类型
        
        # 计算响应率
        response_rate = session.get_response_rate()
        assert response_rate == 1.0  # 100%响应
        
        print(f"✅ 对话比例: {ratio:.0%}, 响应率: {response_rate:.0%}")
    
    @pytest.mark.asyncio
    async def test_monologue_vs_dialogue_detection(self):
        """测试独白vs对话检测"""
        dm = DialogueManager()
        
        # 创建独白会话
        mono_session = await dm.create_session(
            dialogue_type=DialogueType.MONOLOGUE,
            initiator="agent_a",
            participants=["agent_a"],
            topic="Monologue",
            goal="Test"
        )
        
        await dm.add_message(
            dialogue_id=mono_session.dialogue_id,
            sender_id="agent_a",
            content="This is a monologue..."
        )
        
        # 创建对话会话
        dia_session = await dm.create_session(
            dialogue_type=DialogueType.DIALOGUE,
            initiator="agent_b",
            participants=["agent_b", "agent_c"],
            topic="Dialogue",
            goal="Test"
        )
        
        await dm.add_message(
            dialogue_id=dia_session.dialogue_id,
            sender_id="agent_b",
            content="Hello?"
        )
        
        await dm.add_message(
            dialogue_id=dia_session.dialogue_id,
            sender_id="agent_c",
            content="Hi!"
        )
        
        assert mono_session.dialogue_type == DialogueType.MONOLOGUE
        assert dia_session.dialogue_type == DialogueType.DIALOGUE
        assert mono_session.get_dialogue_ratio() == 0.0
        assert dia_session.get_dialogue_ratio() == 1.0
        
        print("✅ 独白vs对话检测测试通过")
    
    @pytest.mark.asyncio
    async def test_target_70_percent_dialogue_ratio(self):
        """测试目标70%对话比例达成"""
        dm = DialogueManager()
        
        session = await dm.create_session(
            dialogue_type=DialogueType.DISCUSSION,
            initiator="agent_a",
            participants=["agent_a", "agent_b", "agent_c"],
            topic="Collaborative Discussion",
            goal="Achieve high dialogue ratio"
        )
        
        # 模拟10轮对话
        prev_msg = None
        for i in range(10):
            sender = f"agent_{['a', 'b', 'c'][i % 3]}"
            
            msg = await dm.add_message(
                dialogue_id=session.dialogue_id,
                sender_id=sender,
                content=f"Message {i+1}",
                intent=MessageIntent.INFORM,
                correlation_id=prev_msg.message_id if prev_msg else None
            )
            prev_msg = msg
        
        ratio = session.get_dialogue_ratio()
        response_rate = session.get_response_rate()
        
        print(f"📊 对话比例: {ratio:.1%}")
        print(f"📊 响应率: {response_rate:.1%}")
        
        # 验证达到目标
        assert ratio >= 0.7, f"对话比例 {ratio:.1%} 未达到70%目标"
        assert response_rate >= 0.8, f"响应率 {response_rate:.1%} 未达到80%目标"
        
        print("✅ 70%对话比例目标达成")


# ==================== 3. Agent间深度对话机制测试 ====================

class TestDeepDialogueMechanism(TestBase):
    """测试深度对话机制"""
    
    @pytest.mark.asyncio
    async def test_collaborative_agent_message_exchange(self):
        """测试协作Agent消息交换"""
        system = MultiAgentCollaborationSystem()
        
        agent1 = ExampleCollaborativeAgent("agent1", AgentRole.DEVELOPER, "Dev1")
        agent2 = ExampleCollaborativeAgent("agent2", AgentRole.QA_ENGINEER, "QA1")
        
        system.register_agent(agent1)
        system.register_agent(agent2)
        
        # 创建对话
        session = await system.dialogue_manager.create_session(
            dialogue_type=DialogueType.DISCUSSION,
            initiator="agent1",
            participants=["agent1", "agent2"],
            topic="Code Review",
            goal="Review implementation"
        )
        
        # Agent1发送消息
        msg1 = await agent1.send_message(
            dialogue_id=session.dialogue_id,
            content="Please review my code",
            intent=MessageIntent.REQUEST
        )
        
        # Agent2响应
        msg2 = await agent2.respond_to_message(
            message=msg1,
            content="I found some issues...",
            intent=MessageIntent.FEEDBACK
        )
        
        assert msg2.correlation_id == msg1.message_id
        assert agent1.stats["messages_sent"] == 1
        assert agent2.stats["messages_sent"] == 1
        
        print("✅ Agent消息交换测试通过")
    
    @pytest.mark.asyncio
    async def test_multi_turn_dialogue(self):
        """测试多轮对话"""
        system = MultiAgentCollaborationSystem()
        
        agent1 = ExampleCollaborativeAgent("agent1", AgentRole.RESEARCHER, "Researcher")
        agent2 = ExampleCollaborativeAgent("agent2", AgentRole.DEVELOPER, "Developer")
        
        system.register_agent(agent1)
        system.register_agent(agent2)
        
        task = CollaborationTask(
            task_type="discussion",
            description="Technical discussion",
            goal="Resolve design questions"
        )
        
        task_id, dialogue_id = await system.start_collaborative_task(
            task=task,
            dialogue_type=DialogueType.DISCUSSION,
            participants=["agent1", "agent2"]
        )
        
        # 运行多轮
        for round_num in range(3):
            await system.run_collaboration_round(dialogue_id)
        
        session = await system.dialogue_manager.get_session(dialogue_id)
        assert len(session.messages) >= 6  # 至少6条消息（3轮 x 2 Agent）
        
        print(f"✅ 多轮对话测试通过: {len(session.messages)} 条消息")
    
    @pytest.mark.asyncio
    async def test_feedback_loop(self):
        """测试反馈循环"""
        system = MultiAgentCollaborationSystem()
        
        agent = ExampleCollaborativeAgent("agent1", AgentRole.DEVELOPER, "Dev")
        system.register_agent(agent)
        
        task = CollaborationTask(
            task_id="test_task",
            task_type="development",
            description="Implement feature",
            goal="Complete implementation"
        )
        
        agent.active_tasks["test_task"] = task
        
        # 提供反馈
        await agent.provide_feedback(
            task_id="test_task",
            feedback="Code needs refactoring",
            score=0.6
        )
        
        await agent.provide_feedback(
            task_id="test_task",
            feedback="Much better now",
            score=0.9
        )
        
        assert len(task.feedback_history) == 2
        assert task.feedback_history[0]["score"] == 0.6
        assert task.feedback_history[1]["score"] == 0.9
        assert agent.stats["feedback_given"] == 2
        
        print("✅ 反馈循环测试通过")


# ==================== 4. 任务分配与反馈循环测试 ====================

class TestTaskAllocation(TestBase):
    """测试任务分配系统"""
    
    @pytest.mark.asyncio
    async def test_capability_based_allocation(self):
        """测试基于能力的分配"""
        dm = DialogueManager()
        allocator = TaskAllocationSystem(dm)
        
        # 注册Agent能力
        allocator.register_agent_capabilities("agent_dev", ["coding", "debugging"])
        allocator.register_agent_capabilities("agent_qa", ["testing", "review"])
        allocator.register_agent_capabilities("agent_pm", ["planning", "coordination"])
        
        # 创建需要coding能力的任务
        task = CollaborationTask(
            task_type="development",
            description="Implement API",
            requirements={
                "capabilities": ["coding"],
                "num_agents": 1
            }
        )
        
        assigned = allocator._allocate_via_algorithm(
            task,
            ["agent_dev", "agent_qa", "agent_pm"]
        )
        
        assert "agent_dev" in assigned
        print("✅ 基于能力的分配测试通过")
    
    @pytest.mark.asyncio
    async def test_load_balancing(self):
        """测试负载均衡"""
        dm = DialogueManager()
        allocator = TaskAllocationSystem(dm)
        
        allocator.register_agent_capabilities("agent1", ["coding"])
        allocator.register_agent_capabilities("agent2", ["coding"])
        
        # 模拟agent1已有负载
        allocator.agent_load["agent1"] = 5
        allocator.agent_load["agent2"] = 1
        
        task = CollaborationTask(
            task_type="development",
            requirements={"capabilities": ["coding"], "num_agents": 1}
        )
        
        assigned = allocator._allocate_via_algorithm(task, ["agent1", "agent2"])
        
        # 应该分配给负载较低的agent2
        assert "agent2" in assigned
        print("✅ 负载均衡测试通过")
    
    @pytest.mark.asyncio
    async def test_feedback_loop_processing(self):
        """测试反馈循环处理"""
        dm = DialogueManager()
        allocator = TaskAllocationSystem(dm)
        
        task = CollaborationTask(
            task_id="test_task",
            task_type="development",
            acceptance_criteria={"min_score": 0.8}
        )
        
        # 添加不满足标准的反馈
        task.feedback_history.append({
            "from": "reviewer",
            "feedback": "Needs improvement",
            "score": 0.6
        })
        
        result = await allocator.process_feedback_loop(task)
        
        assert result == False  # 未满足标准
        assert task.revision_count == 1
        assert task.status == "in_progress"
        
        # 添加满足标准的反馈
        task.feedback_history.append({
            "from": "reviewer",
            "feedback": "Good enough",
            "score": 0.85
        })
        
        result = await allocator.process_feedback_loop(task)
        
        assert result == True  # 满足标准
        assert task.status == "completed"
        
        print("✅ 反馈循环处理测试通过")


# ==================== 5. 协作质量评估测试 ====================

class TestQualityAssessment(TestBase):
    """测试协作质量评估"""
    
    @pytest.mark.asyncio
    async def test_collaboration_metrics_calculation(self):
        """测试协作指标计算"""
        dm = DialogueManager()
        monitor = CollaborationQualityMonitor(dm)
        
        # 创建测试会话
        session = await dm.create_session(
            dialogue_type=DialogueType.DISCUSSION,
            initiator="agent_a",
            participants=["agent_a", "agent_b", "agent_c"],
            topic="Quality Test",
            goal="Test metrics"
        )
        
        # 添加均衡的消息
        for i in range(9):
            sender = f"agent_{['a', 'b', 'c'][i % 3]}"
            await dm.add_message(
                dialogue_id=session.dialogue_id,
                sender_id=sender,
                content=f"Message {i+1}"
            )
        
        metrics = await monitor.evaluate_session(session.dialogue_id)
        
        assert metrics.dialogue_ratio == 1.0
        assert metrics.participation_balance > 0.8  # 相对均衡
        assert metrics.overall_score > 0
        
        print(f"✅ 指标计算测试通过: 整体评分 {metrics.overall_score:.2%}")
    
    @pytest.mark.asyncio
    async def test_health_check(self):
        """测试健康检查"""
        dm = DialogueManager()
        monitor = CollaborationQualityMonitor(dm)
        
        # 创建健康会话
        session = await dm.create_session(
            dialogue_type=DialogueType.DISCUSSION,
            initiator="agent_a",
            participants=["agent_a", "agent_b"],
            topic="Healthy Collaboration",
            goal="Test health"
        )
        
        # 添加高质量对话
        prev_msg = None
        for i in range(8):
            sender = f"agent_{['a', 'b'][i % 2]}"
            msg = await dm.add_message(
                dialogue_id=session.dialogue_id,
                sender_id=sender,
                content=f"Response {i+1}",
                correlation_id=prev_msg.message_id if prev_msg else None
            )
            prev_msg = msg
        
        metrics = await monitor.evaluate_session(session.dialogue_id)
        
        assert metrics.is_healthy() == True
        assert metrics.dialogue_ratio >= 0.7
        assert metrics.response_rate >= 0.8
        
        print("✅ 健康检查测试通过")
    
    @pytest.mark.asyncio
    async def test_alert_generation(self):
        """测试告警生成"""
        dm = DialogueManager()
        monitor = CollaborationQualityMonitor(dm)
        
        # 创建低质量会话
        session = await dm.create_session(
            dialogue_type=DialogueType.MONOLOGUE,  # 独白类型
            initiator="agent_a",
            participants=["agent_a"],
            topic="Low Quality",
            goal="Test alerts"
        )
        
        # 只添加独白消息
        for i in range(5):
            await dm.add_message(
                dialogue_id=session.dialogue_id,
                sender_id="agent_a",
                content=f"Monologue {i+1}"
            )
        
        await monitor.evaluate_session(session.dialogue_id)
        
        # 应该生成告警
        assert len(monitor.alerts) > 0
        
        # 检查告警类型
        alert_types = [a["type"] for a in monitor.alerts]
        assert "low_dialogue_ratio" in alert_types
        
        print("✅ 告警生成测试通过")


# ==================== 6. AGENTS.md v2.0集成测试 ====================

class TestAGENTSv2Integration(TestBase):
    """测试AGENTS.md v2.0集成"""
    
    @pytest.mark.asyncio
    async def test_three_layer_architecture(self):
        """测试三层架构"""
        system = AGENTSv2CollaborationSystem()
        system.initialize_standard_team()
        
        # 验证三层
        assert len(system.strategic_agents) == 3
        assert len(system.coordination_agents) == 3
        assert len(system.execution_agents) == 5
        assert len(system.base_system.agents) == 11
        
        print("✅ 三层架构测试通过")
    
    @pytest.mark.asyncio
    async def test_soul_dimension_profiles(self):
        """测试SOUL维度档案"""
        profiles = {
            AgentRole.CEO: SoulDimensionProfile.from_role(AgentRole.CEO),
            AgentRole.DEVELOPER: SoulDimensionProfile.from_role(AgentRole.DEVELOPER),
            AgentRole.RESEARCHER: SoulDimensionProfile.from_role(AgentRole.RESEARCHER)
        }
        
        # CEO应该有高personality和motivations
        assert profiles[AgentRole.CEO].personality > 0.9
        assert profiles[AgentRole.CEO].motivations > 0.85
        
        # Researcher应该有高curiosity
        assert profiles[AgentRole.RESEARCHER].curiosity > 0.9
        
        print("✅ SOUL维度档案测试通过")
    
    @pytest.mark.asyncio
    async def test_workflow_mode_execution(self):
        """测试工作流模式执行"""
        system = AGENTSv2CollaborationSystem()
        system.initialize_standard_team()
        
        # 测试顺序工作流
        result = await system.workflow_executor.execute_sequential(
            workflow_id="test_seq",
            steps=[
                {"name": "step1", "agent_id": "researcher", "task_type": "research", "description": "Research"},
                {"name": "step2", "agent_id": "developer", "task_type": "design", "description": "Design"}
            ],
            initial_input={"topic": "Test"}
        )
        
        assert "steps" in result
        assert "step1" in result["steps"]
        assert "step2" in result["steps"]
        
        print("✅ 工作流模式执行测试通过")
    
    @pytest.mark.asyncio
    async def test_cross_layer_workflow(self):
        """测试跨层工作流"""
        system = AGENTSv2CollaborationSystem()
        system.initialize_standard_team()
        
        result = await system.execute_project_workflow(
            project_description="Test cross-layer project",
            requirements={"priority": "high"}
        )
        
        assert "strategic" in result
        assert "coordination" in result
        assert "execution" in result
        
        print("✅ 跨层工作流测试通过")
    
    @pytest.mark.asyncio
    async def test_agent_role_capabilities(self):
        """测试Agent角色能力"""
        system = AGENTSv2CollaborationSystem()
        system.initialize_standard_team()
        
        # 检查CEO的能力
        ceo = system.strategic_agents.get("ceo_kimi_claw")
        assert ceo is not None
        assert "strategic_planning" in ceo.definition.capabilities
        assert ceo.can_make_decision("strategic") == True
        
        # 检查Developer的能力
        dev = system.execution_agents.get("developer")
        assert dev is not None
        assert "coding" in dev.definition.skills
        assert dev.can_make_decision("execution") == True
        assert dev.can_make_decision("strategic") == False
        
        print("✅ Agent角色能力测试通过")


# ==================== 7. 端到端集成测试 ====================

class TestEndToEnd(TestBase):
    """端到端集成测试"""
    
    @pytest.mark.asyncio
    async def test_full_collaboration_flow(self):
        """测试完整协作流程"""
        print("\n" + "=" * 70)
        print("端到端集成测试: 完整协作流程")
        print("=" * 70)
        
        # 1. 创建系统
        system = AGENTSv2CollaborationSystem()
        system.initialize_standard_team()
        
        # 2. 创建任务
        task = CollaborationTask(
            task_type="feature_development",
            description="Implement Multi-Agent Collaboration Dashboard",
            goal="Create a dashboard to monitor agent collaboration",
            requirements={
                "capabilities": ["coding", "ui_design", "testing"],
                "num_agents": 3
            },
            priority=4
        )
        
        # 3. 启动协作
        task_id, dialogue_id = await system.base_system.start_collaborative_task(
            task=task,
            dialogue_type=DialogueType.DISCUSSION,
            participants=["developer", "data_analyst", "qa_engineer"]
        )
        
        print(f"✅ 任务启动: {task_id}")
        print(f"✅ 对话创建: {dialogue_id}")
        
        # 4. 运行协作轮次
        for i in range(3):
            await system.base_system.run_collaboration_round(dialogue_id)
        
        session = await system.base_system.dialogue_manager.get_session(dialogue_id)
        print(f"✅ 协作完成: {len(session.messages)} 条消息")
        
        # 5. 评估质量
        metrics = await system.base_system.quality_monitor.evaluate_session(dialogue_id)
        
        print(f"\n📊 协作质量报告:")
        print(f"   - 对话比例: {metrics.dialogue_ratio:.1%}")
        print(f"   - 响应率: {metrics.response_rate:.1%}")
        print(f"   - 参与均衡: {metrics.participation_balance:.1%}")
        print(f"   - 整体评分: {metrics.overall_score:.1%}")
        print(f"   - 健康状态: {'✅ 健康' if metrics.is_healthy() else '⚠️ 需改进'}")
        
        # 6. 关闭会话
        await system.base_system.evaluate_and_close(dialogue_id)
        
        # 7. 系统报告
        report = system.get_architecture_report()
        print(f"\n📋 系统架构报告:")
        print(f"   - 版本: {report['version']}")
        print(f"   - 总Agent数: {report['architecture']['total_agents']}")
        print(f"   - 工作流模式: {len(report['workflow_modes'])} 种")
        
        # 验证
        assert metrics.dialogue_ratio >= 0.7
        assert len(session.messages) >= 6
        
        print("\n" + "=" * 70)
        print("✅ 端到端集成测试通过!")
        print("=" * 70)
    
    @pytest.mark.asyncio
    async def test_broadcast_inversion_fix_verification(self):
        """验证广播倒转问题已解决"""
        print("\n" + "=" * 70)
        print("广播倒转问题修复验证")
        print("=" * 70)
        
        system = MultiAgentCollaborationSystem()
        
        # 创建4个Agent
        agents = [
            ExampleCollaborativeAgent(f"agent_{i}", AgentRole.DEVELOPER, f"Dev_{i}")
            for i in range(4)
        ]
        
        for agent in agents:
            system.register_agent(agent)
        
        # 创建协作任务
        task = CollaborationTask(
            task_type="collaborative_design",
            description="Design system architecture",
            goal="Create optimal architecture through collaboration"
        )
        
        task_id, dialogue_id = await system.start_collaborative_task(
            task=task,
            dialogue_type=DialogueType.DISCUSSION,
            participants=[a.agent_id for a in agents]
        )
        
        # 运行多轮协作
        for i in range(5):
            await system.run_collaboration_round(dialogue_id)
        
        # 获取会话
        session = await system.dialogue_manager.get_session(dialogue_id)
        
        # 计算指标
        dialogue_ratio = session.get_dialogue_ratio()
        response_rate = session.get_response_rate()
        
        print(f"\n📊 协作统计:")
        print(f"   - 总消息数: {len(session.messages)}")
        print(f"   - 对话比例: {dialogue_ratio:.1%}")
        print(f"   - 响应率: {response_rate:.1%}")
        
        # 验证广播倒转问题已解决
        print(f"\n🎯 广播倒转问题修复验证:")
        print(f"   修复前: 93%独白 / 7%对话")
        print(f"   修复后: {(1-dialogue_ratio)*100:.0f}%独白 / {dialogue_ratio*100:.0f}%对话")
        
        assert dialogue_ratio >= 0.70, f"对话比例 {dialogue_ratio:.1%} 未达到70%目标"
        assert response_rate >= 0.80, f"响应率 {response_rate:.1%} 未达到80%目标"
        
        print(f"\n✅ 广播倒转问题已解决!")
        print(f"✅ 对话比例从7%提升到 {dialogue_ratio:.1%}")
        print("=" * 70)


# ==================== 运行测试 ====================

async def run_all_tests():
    """运行所有测试"""
    
    print("\n" + "=" * 70)
    print("Multi-Agent Collaboration System v3.0 - 全面测试套件")
    print("=" * 70)
    
    test_classes = [
        TestDialogueProtocol(),
        TestBroadcastInversionFix(),
        TestDeepDialogueMechanism(),
        TestTaskAllocation(),
        TestQualityAssessment(),
        TestAGENTSv2Integration(),
        TestEndToEnd()
    ]
    
    passed = 0
    failed = 0
    
    for test_class in test_classes:
        class_name = test_class.__class__.__name__
        print(f"\n📦 测试类: {class_name}")
        print("-" * 50)
        
        methods = [m for m in dir(test_class) if m.startswith("test_")]
        
        for method_name in methods:
            try:
                method = getattr(test_class, method_name)
                if asyncio.iscoroutinefunction(method):
                    await method()
                else:
                    method()
                passed += 1
            except Exception as e:
                print(f"❌ {method_name} 失败: {e}")
                failed += 1
    
    print("\n" + "=" * 70)
    print("测试结果汇总")
    print("=" * 70)
    print(f"✅ 通过: {passed}")
    print(f"❌ 失败: {failed}")
    print(f"📊 总计: {passed + failed}")
    print(f"🎯 通过率: {passed/(passed+failed)*100:.1f}%")
    
    if failed == 0:
        print("\n🎉 所有测试通过! Multi-Agent协作系统v3.0已就绪!")
    
    print("=" * 70)
    
    return failed == 0


if __name__ == "__main__":
    success = asyncio.run(run_all_tests())
    exit(0 if success else 1)
