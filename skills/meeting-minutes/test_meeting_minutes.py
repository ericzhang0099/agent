#!/usr/bin/env python3
"""
Meeting Minutes Generator - 测试套件
"""

import unittest
import sys
import os

# 添加父目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from meeting_minutes import MeetingMinutesGenerator, MeetingMinutes, ActionItem


class TestMeetingMinutesGenerator(unittest.TestCase):
    """测试会议纪要生成器"""
    
    def setUp(self):
        self.generator = MeetingMinutesGenerator()
    
    def test_extract_participants(self):
        """测试参与人提取"""
        content = "参与人：张三、李四、王五\n会议内容..."
        participants = self.generator._extract_participants(content)
        self.assertIn("张三", participants)
        self.assertIn("李四", participants)
        self.assertIn("王五", participants)
    
    def test_extract_topics(self):
        """测试议题提取"""
        content = """
        议题1：产品规划讨论
        议题2：技术方案评审
        1. 第一季度目标
        2. 团队分工安排
        """
        topics = self.generator._extract_topics(content)
        self.assertTrue(len(topics) > 0)
    
    def test_extract_decisions(self):
        """测试决策提取"""
        content = """
        决定：采用方案A进行开发
        确定：下周一开始执行
        结论：优先处理核心功能
        """
        decisions = self.generator._extract_decisions(content)
        self.assertTrue(len(decisions) >= 2)
    
    def test_extract_action_items(self):
        """测试行动项提取"""
        content = """
        张三负责完成需求文档，下周三前提交
        李四负责技术调研，明天给出方案
        王五跟进第三方接口，3月1日前完成
        """
        action_items = self.generator._extract_action_items(content)
        self.assertTrue(len(action_items) >= 2)
        
        # 检查提取的负责人
        owners = [item.owner for item in action_items]
        self.assertIn("张三", owners)
        self.assertIn("李四", owners)
    
    def test_generate_minutes(self):
        """测试完整会议纪要生成"""
        content = """
        2026年2月27日 团队周会
        
        参与人：张三、李四、王五
        
        议题：
        1. 上周工作回顾
        2. 本周计划讨论
        
        决定：
        - 确定采用微服务架构
        - 下周发布v1.0版本
        
        行动项：
        张三负责完成API文档，3月5日前
        李四负责部署脚本，下周二前
        
        待跟进：
        - 确认第三方服务价格
        - 安排用户测试
        """
        
        minutes = self.generator.generate_minutes(content, "团队周会")
        
        self.assertEqual(minutes.title, "团队周会")
        self.assertIn("张三", minutes.participants)
        self.assertTrue(len(minutes.topics) > 0)
        self.assertTrue(len(minutes.decisions) > 0)
        self.assertTrue(len(minutes.action_items) > 0)
    
    def test_format_markdown(self):
        """测试Markdown格式化"""
        minutes = MeetingMinutes(
            title="测试会议",
            date="2026-02-27",
            duration="30分钟",
            participants=["张三", "李四"],
            topics=["议题1", "议题2"],
            decisions=["决策1"],
            action_items=[
                ActionItem(task="任务1", owner="张三", deadline="下周三")
            ],
            follow_ups=["跟进1"]
        )
        
        output = self.generator._format_markdown(minutes)
        
        self.assertIn("# 测试会议", output)
        self.assertIn("张三", output)
        self.assertIn("议题1", output)
        self.assertIn("决策1", output)
        self.assertIn("任务1", output)
    
    def test_format_text(self):
        """测试纯文本格式化"""
        minutes = MeetingMinutes(
            title="测试会议",
            date="2026-02-27",
            duration="30分钟",
            participants=["张三"],
            topics=["议题1"],
            decisions=[],
            action_items=[],
            follow_ups=[]
        )
        
        output = self.generator._format_text(minutes)
        
        self.assertIn("测试会议", output)
        self.assertIn("2026-02-27", output)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_workflow(self):
        """测试完整工作流"""
        generator = MeetingMinutesGenerator()
        
        # 模拟会议记录文本
        meeting_text = """
        2026年2月27日 产品评审会
        
        参与人：产品经理小王、开发负责人老李、测试负责人小张
        
        今天讨论了v2.0版本的发布计划。
        
        议题：
        1. 新功能需求确认
        2. 发布时间规划
        3. 资源分配
        
        决定：
        - 确定3月15日发布v2.0
        - 优先完成核心功能，非核心功能延后
        - 采用灰度发布策略
        
        行动项：
        小王负责完善PRD文档，3月1日前提交
        老李负责技术方案设计，下周三前评审
        小张准备测试用例，3月10日前完成
        
        待跟进：
        - 确认服务器扩容方案
        - 安排产品经理培训
        """
        
        # 生成会议纪要
        minutes = generator.generate_minutes(meeting_text)
        
        # 验证结果
        self.assertEqual(minutes.date, "2026-02-27")
        self.assertIn("小王", minutes.participants)
        self.assertIn("老李", minutes.participants)
        self.assertTrue(len(minutes.topics) >= 2)
        self.assertTrue(len(minutes.decisions) >= 2)
        self.assertTrue(len(minutes.action_items) >= 2)
        
        # 验证Markdown输出
        md_output = generator.format_minutes(minutes, "markdown")
        self.assertIn("#", md_output)
        self.assertIn("📋", md_output)
        self.assertIn("✅", md_output)
        self.assertIn("📝", md_output)
        
        print("\n✅ 集成测试通过!")
        print(f"   - 提取到 {len(minutes.participants)} 个参与人")
        print(f"   - 提取到 {len(minutes.topics)} 个议题")
        print(f"   - 提取到 {len(minutes.decisions)} 个决策")
        print(f"   - 提取到 {len(minutes.action_items)} 个行动项")


def run_demo():
    """运行演示"""
    print("="*60)
    print("📝 会议纪要生成器 - 演示")
    print("="*60)
    
    generator = MeetingMinutesGenerator()
    
    # 示例会议内容
    meeting_text = """
    2026年2月27日 团队周会
    
    参与人：Kimi Claw、Research Agent、Dev Agent、Data Agent
    
    今天的会议讨论了本周的工作进展和下周计划。
    
    议题：
    1. 本周OKR完成情况回顾
    2. 新Skill开发计划
    3. 系统架构优化方案
    
    决定：
    - 确定开发会议纪要自动生成器Skill
    - 每周五下午进行技术分享
    - 采用新的代码审查流程
    
    行动项：
    Kimi Claw负责完成会议纪要生成器，今天内完成
    Research Agent持续监控GitHub新Skill，每小时汇报
    Dev Agent优化CI/CD流程，下周三前完成
    Data Agent整理数据 pipeline 文档，3月5日前提交
    
    待跟进：
    - 确认飞书API权限申请进度
    - 安排下周团队建设活动
    """
    
    print("\n📥 输入文本:")
    print("-"*40)
    print(meeting_text[:200] + "...")
    
    print("\n🤖 正在生成会议纪要...")
    minutes = generator.generate_minutes(meeting_text, "团队周会")
    
    print("\n📤 输出结果:")
    print("="*60)
    output = generator.format_minutes(minutes, "markdown")
    print(output)
    print("="*60)
    
    print("\n📊 生成统计:")
    print(f"  ✅ 参与人: {len(minutes.participants)} 人")
    print(f"  ✅ 议题: {len(minutes.topics)} 个")
    print(f"  ✅ 决策: {len(minutes.decisions)} 个")
    print(f"  ✅ 行动项: {len(minutes.action_items)} 个")
    print(f"  ✅ 待跟进: {len(minutes.follow_ups)} 个")
    
    return minutes


if __name__ == '__main__':
    # 如果带参数 --demo，运行演示
    if len(sys.argv) > 1 and sys.argv[1] == '--demo':
        run_demo()
    else:
        # 运行单元测试
        unittest.main(verbosity=2)
