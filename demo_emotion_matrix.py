#!/usr/bin/env python3
"""
情绪-任务矩阵系统 - 完整演示
展示16情绪×任务类型映射系统的全部功能
"""

import sys
sys.path.insert(0, '/root/.openclaw/workspace')

from emotion_task_matrix import (
    EmotionTaskMatrix,
    EmotionType,
    get_match_score,
    get_optimal_emotions,
    EMOTION_TASK_MATRIX,
    TASK_TYPES,
    TASK_TYPE_NAMES
)


def print_header(title):
    """打印标题"""
    print("\n" + "=" * 70)
    print(f"  {title}")
    print("=" * 70 + "\n")


def demo_basic_matching():
    """演示基础匹配功能"""
    print_header("1. 基础情绪-任务匹配")
    
    # 展示不同任务的最优情绪
    test_tasks = [
        ("coding", "代码开发"),
        ("brainstorming", "头脑风暴"),
        ("incident_response", "应急响应"),
        ("teaching", "教学指导"),
        ("data_analysis", "数据分析"),
        ("research", "研究调研")
    ]
    
    print("任务类型 → 推荐情绪 (匹配分数):")
    print("-" * 50)
    
    for task_type, task_name in test_tasks:
        optimal = get_optimal_emotions(task_type, top_k=3)
        emotions_str = ", ".join([f"{e}({s:.2f})" for e, s in optimal])
        print(f"  {task_name:<12} → {emotions_str}")
    
    print()


def demo_matrix_heatmap():
    """演示匹配矩阵热力图"""
    print_header("2. 情绪-任务匹配矩阵热力图")
    
    # 选择部分情绪和任务展示
    emotions = ["兴奋", "专注", "冷静", "警惕", "紧迫", "幽默"]
    tasks = ["coding", "brainstorming", "incident_response", "teaching", "data_analysis"]
    
    # 打印表头
    header = "情绪\\任务".ljust(10)
    for task in tasks:
        task_short = task[:8]
        header += f"{task_short:<10}"
    print(header)
    print("-" * 60)
    
    # 打印矩阵
    for emotion in emotions:
        row = f"{emotion:<10}"
        for task in tasks:
            score = get_match_score(emotion, task)
            # 根据分数选择显示字符
            if score >= 0.8:
                cell = "███"
            elif score >= 0.6:
                cell = "▓▓▓"
            elif score >= 0.4:
                cell = "▒▒▒"
            else:
                cell = "░░░"
            row += f"{cell:<10}"
        print(row)
    
    print("\n图例: ███ 高(≥0.8)  ▓▓▓ 中(≥0.6)  ▒▒▒ 低(≥0.4)  ░░░ 极低(<0.4)")
    print()


def demo_transition_path():
    """演示情绪过渡路径"""
    print_header("3. 情绪过渡路径规划")
    
    system = EmotionTaskMatrix().initialize()
    
    transitions = [
        ("冷静", "专注", "开始深度工作"),
        ("冷静", "兴奋", "进入头脑风暴"),
        ("兴奋", "冷静", "平复情绪"),
        ("紧迫", "冷静", "危机处理后恢复"),
        ("沮丧", "冷静", "从失败中恢复")
    ]
    
    print("情绪过渡路径:")
    print("-" * 50)
    
    for from_e, to_e, scenario in transitions:
        path = system.get_transition_path(from_e, to_e)
        path_str = " → ".join(path)
        print(f"  {scenario:<20}: {path_str}")
    
    print()


def demo_task_scheduling():
    """演示任务调度"""
    print_header("4. 情绪感知任务调度")
    
    system = EmotionTaskMatrix().initialize()
    
    # 创建模拟Agent
    class MockAgent:
        def __init__(self, name, skills, emotion, load=0.5):
            self.id = name
            self.name = name
            self.skills = skills
            self.current_emotion = emotion
            self.load = load
            self.is_healthy = True
    
    # 注册Agent
    agents = [
        MockAgent("DevAgent", ["coding", "debugging"], "专注", 0.4),
        MockAgent("ResearchAgent", ["research", "analysis"], "好奇", 0.3),
        MockAgent("OpsAgent", ["monitoring", "deployment"], "冷静", 0.6),
    ]
    
    for agent in agents:
        system.register_agent(agent)
    
    # 模拟任务
    class MockTask:
        def __init__(self, task_type, priority="normal"):
            self.task_type = task_type
            self.context = {'priority': priority}
            self.required_skills = [task_type.split('_')[0]] if '_' in task_type else [task_type]
    
    tasks = [
        MockTask("coding", "high"),
        MockTask("research", "normal"),
        MockTask("incident_response", "critical")
    ]
    
    print("任务调度结果:")
    print("-" * 50)
    
    for task in tasks:
        selected = system.schedule_task(task)
        if selected:
            print(f"  任务: {task.task_type:<20} → Agent: {selected.name:>12} (情绪: {selected.current_emotion})")
        else:
            print(f"  任务: {task.task_type:<20} → 无可用Agent")
    
    print()


def demo_emotion_profiles():
    """演示情绪档案"""
    print_header("5. 16种情绪档案")
    
    from emotion_task_matrix.core.emotion_definitions import EMOTION_PROFILES
    
    print(f"{'情绪':<10} {'强度':<8} {'能量':<8} {'极性':<8} {'主要维度':<15} {'描述'}")
    print("-" * 90)
    
    for emotion_type, profile in EMOTION_PROFILES.items():
        polarity_str = "+" if profile.polarity > 0 else ""
        print(f"{profile.name:<10} {profile.intensity:<8.1f} {profile.energy:<8.1f} "
              f"{polarity_str+str(profile.polarity):<8} {profile.primary_dimension:<15} "
              f"{profile.description[:25]}...")
    
    print()


def demo_system_status():
    """演示系统状态"""
    print_header("6. 系统状态")
    
    system = EmotionTaskMatrix().initialize()
    
    # 注册一些Agent
    class MockAgent:
        def __init__(self, name, emotion):
            self.id = name
            self.current_emotion = emotion
            self.current_task_type = "coding"
            self.current_match_score = 0.85
            
            class MockState:
                def __init__(self):
                    self.stability = 0.9
                    self.drift_score = 0.1
                    self.transition_count_1h = 2
            
            self.emotion_state = MockState()
    
    for i, emotion in enumerate(["专注", "冷静", "兴奋"]):
        system.register_agent(MockAgent(f"agent_{i}", emotion))
    
    # 获取系统状态
    status = system.get_system_status()
    
    print(f"系统版本: {status['version']}")
    print(f"初始化状态: {'✓' if status['initialized'] else '✗'}")
    print(f"注册Agent数: {status['registered_agents']}")
    print()
    print("组件状态:")
    for name, state in status['components'].items():
        status_icon = "✓" if state else "✗"
        print(f"  {status_icon} {name}")
    
    print()


def demo_heartbeat():
    """演示HEARTBEAT集成"""
    print_header("7. HEARTBEAT监控")
    
    system = EmotionTaskMatrix().initialize()
    
    # 创建模拟Agent
    class MockAgent:
        def __init__(self, name, emotion, drift=0.1):
            self.id = name
            self.current_emotion = emotion
            self.current_task_type = "coding"
            self.current_match_score = 0.85
            
            class MockState:
                def __init__(self, drift_val):
                    self.stability = 0.9
                    self.drift_score = drift_val
                    self.transition_count_1h = 2
                    self.stuck_duration = 300
                    self.last_transition_time = __import__('time').time() - 600
            
            self.emotion_state = MockState(drift)
    
    # 注册正常Agent
    system.register_agent(MockAgent("normal_agent", "专注", 0.1))
    
    # 注册问题Agent
    system.register_agent(MockAgent("problem_agent", "沮丧", 0.45))
    
    print("Agent心跳状态:")
    print("-" * 50)
    
    for agent_id in ["normal_agent", "problem_agent"]:
        heartbeat = system.generate_heartbeat(agent_id)
        if heartbeat:
            print(f"\n  Agent: {agent_id}")
            print(f"    当前情绪: {heartbeat.current_emotion}")
            print(f"    稳定性: {heartbeat.emotion_stability:.2f}")
            print(f"    漂移: {heartbeat.emotion_drift:.2f}")
            print(f"    告警: {len(heartbeat.alerts)} 个")
            
            for alert in heartbeat.alerts:
                level_icon = "⚠️" if alert.level.value == "warning" else "🚨"
                print(f"      {level_icon} [{alert.level.value}] {alert.message}")
    
    print()


def main():
    """主函数"""
    print("\n" + "=" * 70)
    print("           情绪-任务矩阵系统 v1.0 - 完整演示")
    print("=" * 70)
    
    demo_basic_matching()
    demo_matrix_heatmap()
    demo_transition_path()
    demo_task_scheduling()
    demo_emotion_profiles()
    demo_system_status()
    demo_heartbeat()
    
    print_header("演示完成")
    print("情绪-任务矩阵系统已准备就绪！")
    print("\n核心功能:")
    print("  ✓ 16情绪 × 18任务类型 完整映射")
    print("  ✓ 智能情绪匹配算法")
    print("  ✓ 平滑情绪过渡管理")
    print("  ✓ HEARTBEAT监控集成")
    print("  ✓ 情绪感知任务调度")
    print("  ✓ 与AGENTS.md工作流集成")
    print()


if __name__ == "__main__":
    main()
