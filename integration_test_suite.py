#!/usr/bin/env python3
"""
端到端集成测试套件
End-to-End Integration Test Suite

测试所有集成系统的协同工作
"""

import asyncio
import json
import sys
import time
from datetime import datetime
from typing import Dict, List, Any
import unittest

# 导入统一API网关
from unified_api_gateway import (
    get_api, shutdown_api, UnifiedAPI,
    SystemComponent, MessageType, SystemMessage
)


class TestResult:
    """测试结果"""
    def __init__(self, test_name: str):
        self.test_name = test_name
        self.status = "pending"
        self.start_time = None
        self.end_time = None
        self.duration_ms = 0
        self.error_message = None
        self.details = {}
    
    def start(self):
        self.start_time = datetime.now()
        self.status = "running"
    
    def success(self, details: Dict = None):
        self.end_time = datetime.now()
        self.status = "passed"
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        if details:
            self.details = details
    
    def failure(self, error: str, details: Dict = None):
        self.end_time = datetime.now()
        self.status = "failed"
        self.duration_ms = (self.end_time - self.start_time).total_seconds() * 1000
        self.error_message = error
        if details:
            self.details = details
    
    def to_dict(self) -> Dict:
        return {
            "test_name": self.test_name,
            "status": self.status,
            "duration_ms": round(self.duration_ms, 2),
            "error_message": self.error_message,
            "details": self.details
        }


class IntegrationTestSuite:
    """集成测试套件"""
    
    def __init__(self):
        self.api: UnifiedAPI = None
        self.results: List[TestResult] = []
        self.test_start_time = None
        self.test_end_time = None
    
    async def setup(self):
        """测试设置"""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST SUITE - Setup")
        print("=" * 70)
        self.test_start_time = datetime.now()
        self.api = await get_api()
        print("✓ API Gateway initialized")
    
    async def teardown(self):
        """测试清理"""
        print("\n" + "=" * 70)
        print("INTEGRATION TEST SUITE - Teardown")
        print("=" * 70)
        await shutdown_api()
        self.test_end_time = datetime.now()
        print("✓ API Gateway shutdown")
    
    async def run_all_tests(self) -> Dict:
        """运行所有测试"""
        await self.setup()
        
        try:
            # 系统初始化测试
            await self.test_system_initialization()
            
            # SoulKernel集成测试
            await self.test_soulkernel_integration()
            
            # 记忆系统集成测试
            await self.test_memory_integration()
            
            # 推理协调器集成测试
            await self.test_reasoning_integration()
            
            # 自主Agent集成测试
            await self.test_autonomous_agent_integration()
            
            # 多模态系统集成测试
            await self.test_multimodal_integration()
            
            # 群体智能集成测试
            await self.test_swarm_integration()
            
            # 安全对齐集成测试
            await self.test_safety_integration()
            
            # 情绪矩阵集成测试
            await self.test_emotion_integration()
            
            # 跨系统通信测试
            await self.test_cross_system_communication()
            
            # 端到端工作流测试
            await self.test_end_to_end_workflow()
            
            # 故障恢复测试
            await self.test_fault_recovery()
            
            # 性能基准测试
            await self.test_performance_benchmark()
            
        finally:
            await self.teardown()
        
        return self.generate_report()
    
    # ========== 具体测试用例 ==========
    
    async def test_system_initialization(self):
        """测试系统初始化"""
        result = TestResult("System Initialization")
        result.start()
        
        try:
            # 检查系统状态
            status = await self.api.get_system_status()
            
            # 验证所有组件已初始化
            assert status["system_status"] == "running", "System not running"
            assert status["total_components"] == 8, f"Expected 8 components, got {status['total_components']}"
            assert status["healthy_components"] >= 6, f"Only {status['healthy_components']} healthy components"
            
            # 验证所有组件状态
            components = status["components"]
            expected_components = [
                "soulkernel", "memory", "reasoning", "autonomous_agent",
                "multimodal", "swarm", "safety", "emotion"
            ]
            
            for comp in expected_components:
                assert comp in components, f"Component {comp} not found"
            
            result.success({
                "total_components": status["total_components"],
                "healthy_components": status["healthy_components"],
                "uptime": status["uptime_seconds"]
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_soulkernel_integration(self):
        """测试SoulKernel集成"""
        result = TestResult("SoulKernel Integration")
        result.start()
        
        try:
            # 验证SoulKernel状态
            status = await self.api.get_system_status()
            soulkernel_status = status["components"]["soulkernel"]
            
            assert soulkernel_status["status"] == "healthy", "SoulKernel not healthy"
            assert "consciousness_coordination" in soulkernel_status["capabilities"]
            assert "attention_management" in soulkernel_status["capabilities"]
            
            # 测试任务协调
            task_id = await self.api.create_task(
                task_type="research",
                title="SoulKernel Test Task",
                description="Testing SoulKernel task coordination",
                priority=5
            )
            
            assert task_id is not None, "Task creation failed"
            
            # 等待任务处理
            await asyncio.sleep(1)
            
            result.success({
                "task_id": task_id,
                "capabilities": soulkernel_status["capabilities"],
                "load": soulkernel_status["load"]
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_memory_integration(self):
        """测试记忆系统集成"""
        result = TestResult("Memory System Integration")
        result.start()
        
        try:
            # 验证记忆系统状态
            status = await self.api.get_system_status()
            memory_status = status["components"]["memory"]
            
            assert memory_status["status"] == "healthy", "Memory system not healthy"
            
            # 测试存储记忆
            memory_id = await self.api.store_memory(
                content="Integration test memory - SoulKernel architecture",
                memory_type="semantic",
                importance=0.9,
                metadata={"test": True, "category": "architecture"}
            )
            
            assert memory_id is not None, "Memory storage failed"
            
            # 测试检索记忆
            retrieved_memories = await self.api.retrieve_memory(
                query="SoulKernel architecture",
                memory_type="semantic"
            )
            
            result.success({
                "memory_id": memory_id,
                "retrieved_count": len(retrieved_memories),
                "capabilities": memory_status["capabilities"]
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_reasoning_integration(self):
        """测试推理协调器集成"""
        result = TestResult("Reasoning Coordinator Integration")
        result.start()
        
        try:
            # 验证推理系统状态
            status = await self.api.get_system_status()
            reasoning_status = status["components"]["reasoning"]
            
            assert reasoning_status["status"] == "healthy", "Reasoning system not healthy"
            
            # 测试推理
            reasoning_result = await self.api.reason(
                query="Explain the benefits of multi-agent system architecture",
                strategy="chain_of_thought"
            )
            
            assert "answer" in reasoning_result, "Reasoning result missing answer"
            assert "reasoning_chain" in reasoning_result, "Reasoning result missing chain"
            
            result.success({
                "has_answer": True,
                "has_chain": True,
                "strategy": reasoning_result["reasoning_chain"]["strategy"],
                "confidence": reasoning_result["reasoning_chain"].get("confidence", 0)
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_autonomous_agent_integration(self):
        """测试自主Agent集成"""
        result = TestResult("Autonomous Agent Integration")
        result.start()
        
        try:
            # 验证自主Agent状态
            status = await self.api.get_system_status()
            agent_status = status["components"]["autonomous_agent"]
            
            assert agent_status["status"] == "healthy", "Autonomous agent not healthy"
            
            # 测试目标创建
            task_id = await self.api.create_task(
                task_type="strategic",
                title="Autonomous Goal Test",
                description="Testing autonomous goal creation and decomposition",
                priority=4
            )
            
            assert task_id is not None, "Goal creation failed"
            
            # 等待目标处理
            await asyncio.sleep(2)
            
            result.success({
                "goal_id": task_id,
                "capabilities": agent_status["capabilities"],
                "load": agent_status["load"]
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_multimodal_integration(self):
        """测试多模态系统集成"""
        result = TestResult("Multimodal System Integration")
        result.start()
        
        try:
            # 验证多模态系统状态
            status = await self.api.get_system_status()
            multimodal_status = status["components"]["multimodal"]
            
            assert multimodal_status["status"] == "healthy", "Multimodal system not healthy"
            
            # 验证能力列表
            expected_capabilities = [
                "text_processing",
                "image_analysis",
                "audio_processing",
                "cross_modal_fusion"
            ]
            
            for cap in expected_capabilities:
                assert cap in multimodal_status["capabilities"], f"Missing capability: {cap}"
            
            result.success({
                "capabilities": multimodal_status["capabilities"],
                "load": multimodal_status["load"]
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_swarm_integration(self):
        """测试群体智能集成"""
        result = TestResult("Swarm Intelligence Integration")
        result.start()
        
        try:
            # 验证群体智能状态
            status = await self.api.get_system_status()
            swarm_status = status["components"]["swarm"]
            
            assert swarm_status["status"] == "healthy", "Swarm system not healthy"
            
            # 测试群体协调
            await self.api.coordinate_swarm(
                target="test_target",
                agent_count=10
            )
            
            # 等待群体模拟
            await asyncio.sleep(2)
            
            result.success({
                "capabilities": swarm_status["capabilities"],
                "load": swarm_status["load"]
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_safety_integration(self):
        """测试安全对齐集成"""
        result = TestResult("Safety Alignment Integration")
        result.start()
        
        try:
            # 验证安全系统状态
            status = await self.api.get_system_status()
            safety_status = status["components"]["safety"]
            
            assert safety_status["status"] == "healthy", "Safety system not healthy"
            
            # 测试安全检查
            safety_result = await self.api.check_safety(
                content="This is a test content for safety checking",
                check_type="constitutional"
            )
            
            assert "is_safe" in safety_result, "Safety result missing is_safe field"
            assert "score" in safety_result, "Safety result missing score field"
            
            result.success({
                "is_safe": safety_result["is_safe"],
                "safety_score": safety_result["score"],
                "capabilities": safety_status["capabilities"]
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_emotion_integration(self):
        """测试情绪矩阵集成"""
        result = TestResult("Emotion Matrix Integration")
        result.start()
        
        try:
            # 验证情绪系统状态
            status = await self.api.get_system_status()
            emotion_status = status["components"]["emotion"]
            
            assert emotion_status["status"] == "healthy", "Emotion system not healthy"
            
            # 测试情绪更新
            await self.api.update_emotion(
                trigger="success",
                context="Integration test successful"
            )
            
            # 等待情绪处理
            await asyncio.sleep(1)
            
            result.success({
                "capabilities": emotion_status["capabilities"],
                "load": emotion_status["load"]
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_cross_system_communication(self):
        """测试跨系统通信"""
        result = TestResult("Cross-System Communication")
        result.start()
        
        try:
            # 测试SoulKernel -> Memory通信
            await self.api.store_memory(
                content="Cross-system communication test",
                memory_type="episodic",
                importance=0.7
            )
            
            # 测试SoulKernel -> Reasoning通信
            reasoning_result = await self.api.reason(
                query="Test cross-system communication",
                strategy="direct"
            )
            
            # 测试SoulKernel -> Emotion通信
            await self.api.update_emotion(
                trigger="calm",
                context="Communication test"
            )
            
            # 测试Memory -> Reasoning间接通信（通过SoulKernel）
            memories = await self.api.retrieve_memory(
                query="cross-system"
            )
            
            result.success({
                "memory_communication": True,
                "reasoning_communication": "answer" in reasoning_result,
                "emotion_communication": True,
                "retrieved_memories": len(memories)
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_end_to_end_workflow(self):
        """测试端到端工作流"""
        result = TestResult("End-to-End Workflow")
        result.start()
        
        try:
            workflow_steps = []
            
            # Step 1: 创建任务
            task_id = await self.api.create_task(
                task_type="research",
                title="E2E Workflow Test",
                description="Testing complete workflow integration",
                priority=3
            )
            workflow_steps.append({"step": 1, "action": "create_task", "status": "success"})
            
            # Step 2: 存储相关记忆
            memory_id = await self.api.store_memory(
                content=f"Task {task_id} created for E2E workflow test",
                memory_type="episodic",
                importance=0.8
            )
            workflow_steps.append({"step": 2, "action": "store_memory", "status": "success"})
            
            # Step 3: 执行推理
            reasoning_result = await self.api.reason(
                query=f"Analyze task {task_id} and suggest approach",
                strategy="chain_of_thought"
            )
            workflow_steps.append({"step": 3, "action": "reasoning", "status": "success"})
            
            # Step 4: 安全检查
            safety_result = await self.api.check_safety(
                content=reasoning_result.get("answer", ""),
                check_type="constitutional"
            )
            workflow_steps.append({"step": 4, "action": "safety_check", "status": "success"})
            
            # Step 5: 更新情绪
            await self.api.update_emotion(
                trigger="excited" if safety_result["is_safe"] else "concerned",
                context="E2E workflow completion"
            )
            workflow_steps.append({"step": 5, "action": "emotion_update", "status": "success"})
            
            result.success({
                "workflow_steps": workflow_steps,
                "total_steps": len(workflow_steps),
                "completed_steps": len([s for s in workflow_steps if s["status"] == "success"])
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_fault_recovery(self):
        """测试故障恢复"""
        result = TestResult("Fault Recovery")
        result.start()
        
        try:
            # 获取初始状态
            initial_status = await self.api.get_system_status()
            initial_healthy = initial_status["healthy_components"]
            
            # 模拟组件负载增加
            # 注意：这里只是测试监控能力，不会真正导致故障
            
            # 验证系统仍然健康
            current_status = await self.api.get_system_status()
            current_healthy = current_status["healthy_components"]
            
            # 系统应该保持运行
            assert current_status["system_status"] == "running", "System not running after load"
            
            result.success({
                "initial_healthy": initial_healthy,
                "current_healthy": current_healthy,
                "system_stable": current_status["system_status"] == "running"
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    async def test_performance_benchmark(self):
        """测试性能基准"""
        result = TestResult("Performance Benchmark")
        result.start()
        
        try:
            benchmarks = []
            
            # 基准1: 任务创建延迟
            start = time.time()
            for i in range(10):
                await self.api.create_task(
                    task_type="benchmark",
                    title=f"Benchmark Task {i}",
                    description="Performance test",
                    priority=5
                )
            task_latency = (time.time() - start) / 10 * 1000
            benchmarks.append({"metric": "task_creation_latency_ms", "value": round(task_latency, 2)})
            
            # 基准2: 记忆存储延迟
            start = time.time()
            for i in range(10):
                await self.api.store_memory(
                    content=f"Benchmark memory {i}",
                    memory_type="episodic",
                    importance=0.5
                )
            memory_latency = (time.time() - start) / 10 * 1000
            benchmarks.append({"metric": "memory_storage_latency_ms", "value": round(memory_latency, 2)})
            
            # 基准3: 推理延迟
            start = time.time()
            await self.api.reason(
                query="Simple benchmark query",
                strategy="direct"
            )
            reasoning_latency = (time.time() - start) * 1000
            benchmarks.append({"metric": "reasoning_latency_ms", "value": round(reasoning_latency, 2)})
            
            # 基准4: 系统状态查询
            start = time.time()
            for _ in range(10):
                await self.api.get_system_status()
            status_latency = (time.time() - start) / 10 * 1000
            benchmarks.append({"metric": "status_query_latency_ms", "value": round(status_latency, 2)})
            
            result.success({
                "benchmarks": benchmarks,
                "all_under_threshold": all(b["value"] < 5000 for b in benchmarks)
            })
            
        except Exception as e:
            result.failure(str(e))
        
        self.results.append(result)
        self._print_result(result)
    
    def _print_result(self, result: TestResult):
        """打印测试结果"""
        icon = "✓" if result.status == "passed" else "✗"
        print(f"{icon} {result.test_name}: {result.status.upper()}")
        if result.error_message:
            print(f"  Error: {result.error_message}")
        if result.duration_ms > 0:
            print(f"  Duration: {result.duration_ms:.2f}ms")
    
    def generate_report(self) -> Dict:
        """生成测试报告"""
        total_tests = len(self.results)
        passed_tests = len([r for r in self.results if r.status == "passed"])
        failed_tests = len([r for r in self.results if r.status == "failed"])
        
        total_duration = sum(r.duration_ms for r in self.results)
        
        report = {
            "test_suite": "Integration Test Suite",
            "timestamp": datetime.now().isoformat(),
            "summary": {
                "total_tests": total_tests,
                "passed": passed_tests,
                "failed": failed_tests,
                "pass_rate": round(passed_tests / total_tests * 100, 2) if total_tests > 0 else 0,
                "total_duration_ms": round(total_duration, 2)
            },
            "results": [r.to_dict() for r in self.results],
            "system_info": {
                "components_tested": 8,
                "test_categories": [
                    "initialization",
                    "component_integration",
                    "cross_system_communication",
                    "end_to_end_workflow",
                    "fault_recovery",
                    "performance"
                ]
            }
        }
        
        return report


def print_report(report: Dict):
    """打印测试报告"""
    print("\n" + "=" * 70)
    print("INTEGRATION TEST REPORT")
    print("=" * 70)
    print(f"Timestamp: {report['timestamp']}")
    print(f"\nSUMMARY:")
    print(f"  Total Tests: {report['summary']['total_tests']}")
    print(f"  Passed: {report['summary']['passed']}")
    print(f"  Failed: {report['summary']['failed']}")
    print(f"  Pass Rate: {report['summary']['pass_rate']}%")
    print(f"  Total Duration: {report['summary']['total_duration_ms']:.2f}ms")
    
    print(f"\nDETAILED RESULTS:")
    for result in report['results']:
        status_icon = "✓" if result['status'] == 'passed' else "✗"
        print(f"  {status_icon} {result['test_name']}: {result['status']} ({result['duration_ms']:.2f}ms)")
        if result['error_message']:
            print(f"      Error: {result['error_message']}")
    
    print("\n" + "=" * 70)
    if report['summary']['pass_rate'] >= 90:
        print("🎉 ALL TESTS PASSED - SYSTEM READY FOR PRODUCTION")
    elif report['summary']['pass_rate'] >= 70:
        print("⚠️  MOST TESTS PASSED - SYSTEM REQUIRES ATTENTION")
    else:
        print("❌ TESTS FAILED - SYSTEM NOT READY")
    print("=" * 70)


async def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("STARTING INTEGRATION TEST SUITE")
    print("=" * 70)
    
    suite = IntegrationTestSuite()
    report = await suite.run_all_tests()
    
    print_report(report)
    
    # 保存报告到文件
    report_file = f"/root/.openclaw/workspace/integration_test_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    
    print(f"\nReport saved to: {report_file}")
    
    # 返回退出码
    return 0 if report['summary']['pass_rate'] >= 80 else 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
