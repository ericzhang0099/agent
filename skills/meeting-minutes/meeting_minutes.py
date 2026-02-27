#!/usr/bin/env python3
"""
Meeting Minutes Generator - 自动会议纪要生成器
支持音频转录、文本处理、飞书文档输出
"""

import os
import sys
import json
import re
import argparse
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict

# 尝试导入可选依赖
try:
    import openai
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False


@dataclass
class ActionItem:
    """行动项"""
    task: str
    owner: str
    deadline: Optional[str] = None
    priority: str = "medium"


@dataclass
class MeetingMinutes:
    """会议纪要数据结构"""
    title: str
    date: str
    duration: str
    participants: List[str]
    topics: List[str]
    decisions: List[str]
    action_items: List[ActionItem]
    follow_ups: List[str]
    raw_content: str = ""
    source_url: Optional[str] = None


class MeetingMinutesGenerator:
    """会议纪要生成器"""
    
    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.openai_api_key = self.config.get('openai_api_key') or os.getenv('OPENAI_API_KEY')
        self.feishu_app_id = self.config.get('feishu_app_id') or os.getenv('FEISHU_APP_ID')
        self.feishu_app_secret = self.config.get('feishu_app_secret') or os.getenv('FEISHU_APP_SECRET')
    
    def transcribe_audio(self, audio_path: str) -> str:
        """音频转录为文本"""
        if not HAS_OPENAI:
            raise ImportError("需要安装 openai: pip install openai")
        
        if not self.openai_api_key:
            raise ValueError("需要设置 OPENAI_API_KEY")
        
        client = openai.OpenAI(api_key=self.openai_api_key)
        
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="zh",
                response_format="text"
            )
        
        return transcript
    
    def generate_minutes(self, content: str, title: Optional[str] = None) -> MeetingMinutes:
        """从文本生成结构化会议纪要"""
        
        # 提取基本信息
        date_match = re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2})', content)
        date = date_match.group(1) if date_match else datetime.now().strftime("%Y-%m-%d")
        
        # 提取参与人
        participants = self._extract_participants(content)
        
        # 提取议题
        topics = self._extract_topics(content)
        
        # 提取决策
        decisions = self._extract_decisions(content)
        
        # 提取行动项
        action_items = self._extract_action_items(content)
        
        # 提取待跟进事项
        follow_ups = self._extract_follow_ups(content)
        
        # 估算时长
        duration = self._estimate_duration(content)
        
        # 生成标题
        if not title:
            title = self._generate_title(content, topics)
        
        return MeetingMinutes(
            title=title,
            date=date,
            duration=duration,
            participants=participants,
            topics=topics,
            decisions=decisions,
            action_items=action_items,
            follow_ups=follow_ups,
            raw_content=content
        )
    
    def _extract_participants(self, content: str) -> List[str]:
        """提取参与人"""
        # 匹配常见参与人格式
        patterns = [
            r'参与人[：:]\s*([^\n]+)',
            r'参会人员[：:]\s*([^\n]+)',
            r'与会者[：:]\s*([^\n]+)',
            r'出席[：:]\s*([^\n]+)',
        ]
        
        for pattern in patterns:
            match = re.search(pattern, content)
            if match:
                names = match.group(1)
                # 按逗号、顿号、空格分割
                return [n.strip() for n in re.split(r'[,，、\s]+', names) if n.strip()]
        
        # 尝试从内容中提取人名（简单启发式）
        name_pattern = r'([\u4e00-\u9fa5]{2,4})(?:说|提到|建议|认为|问|回答)'
        names = re.findall(name_pattern, content)
        return list(set(names))[:10]  # 去重，最多10个
    
    def _extract_topics(self, content: str) -> List[str]:
        """提取议题"""
        topics = []
        
        # 匹配数字编号议题
        topic_patterns = [
            r'(?:议题|主题|Topic|Agenda)\s*\d*[.．、]?\s*([^\n]+)',
            r'\d+[.．、]\s*([^\n]{5,50})(?=\n|$)',
            r'[一二三四五六七八九十]+[、.]\s*([^\n]{5,50})(?=\n|$)',
        ]
        
        for pattern in topic_patterns:
            matches = re.findall(pattern, content)
            topics.extend([m.strip() for m in matches if len(m.strip()) > 3])
        
        # 去重并限制数量
        seen = set()
        unique_topics = []
        for t in topics:
            if t not in seen and len(unique_topics) < 8:
                seen.add(t)
                unique_topics.append(t)
        
        return unique_topics
    
    def _extract_decisions(self, content: str) -> List[str]:
        """提取决策点"""
        decisions = []
        
        # 决策关键词模式
        decision_patterns = [
            r'(?:决定|决议|确定|拍板|结论|一致同意)[：:]\s*([^\n。]+)',
            r'(?:最终|最后)[，,]?\s*(?:决定|确定)[：:]?\s*([^\n。]+)',
            r'(?:方案|策略)[是:]\s*([^\n。]{5,100})',
        ]
        
        for pattern in decision_patterns:
            matches = re.findall(pattern, content)
            decisions.extend([m.strip() for m in matches if len(m.strip()) > 5])
        
        return decisions[:10]
    
    def _extract_action_items(self, content: str) -> List[ActionItem]:
        """提取行动项"""
        action_items = []
        
        # 行动项模式
        action_patterns = [
            r'(?:TODO|Action|行动项|待办|任务)[：:]?\s*([^\n]+)',
            r'(?:负责|责任人)[：:]?\s*([^，,]+)[，,]\s*(?:任务|工作)[：:]?\s*([^\n]+)',
            r'([^，,]{2,10})[，,]\s*(?:负责|跟进|处理)\s*([^\n]{5,100})',
        ]
        
        # 简单启发式：找"XXX负责XXX"模式
        simple_pattern = r'([\u4e00-\u9fa5]{2,4}|[A-Za-z\s]+?)\s*负责\s*([^\n。]{5,100}?)(?=\n|$|。|$)'
        matches = re.findall(simple_pattern, content, re.MULTILINE)
        
        for owner, task in matches:
            # 尝试提取截止时间
            deadline_match = re.search(r'(\d{4}[-/年]\d{1,2}[-/月]\d{1,2}|\d{1,2}[月/]\d{1,2}[日号]?|下周[一二三四五六日]|明天|后天)', task)
            deadline = deadline_match.group(1) if deadline_match else None
            
            action_items.append(ActionItem(
                task=task.strip(),
                owner=owner.strip(),
                deadline=deadline
            ))
        
        return action_items[:15]
    
    def _extract_follow_ups(self, content: str) -> List[str]:
        """提取待跟进事项"""
        follow_ups = []
        
        patterns = [
            r'(?:待跟进|待确认|待讨论|后续|下次)[：:]?\s*([^\n。]+)',
            r'(?:需要|需)\s*([^\n]{5,80})(?:跟进|确认|讨论|核实)',
        ]
        
        for pattern in patterns:
            matches = re.findall(pattern, content)
            follow_ups.extend([m.strip() for m in matches if len(m.strip()) > 5])
        
        return follow_ups[:8]
    
    def _estimate_duration(self, content: str) -> str:
        """估算会议时长"""
        # 尝试从内容中提取
        duration_match = re.search(r'(?:时长|时间|duration)[：:]?\s*(\d+)\s*(分钟|min|小时|h)', content, re.I)
        if duration_match:
            num = duration_match.group(1)
            unit = duration_match.group(2)
            return f"{num}{unit}"
        
        # 根据内容长度估算（假设每分钟150字）
        word_count = len(content)
        minutes = max(15, word_count // 150)
        
        if minutes < 60:
            return f"{minutes}分钟"
        else:
            hours = minutes // 60
            mins = minutes % 60
            return f"{hours}小时{mins}分钟" if mins > 0 else f"{hours}小时"
    
    def _generate_title(self, content: str, topics: List[str]) -> str:
        """生成会议标题"""
        # 尝试提取会议类型
        meeting_types = {
            '周会': r'周会|weekly|周报',
            '月会': r'月会|monthly|月度',
            '复盘': r'复盘|review|总结',
            '规划': r'规划|planning|计划',
            '评审': r'评审|review|评审会',
            '站会': r'站会|standup|daily',
        }
        
        meeting_type = "会议"
        for mtype, pattern in meeting_types.items():
            if re.search(pattern, content, re.I):
                meeting_type = mtype
                break
        
        # 如果有议题，用第一个议题
        if topics:
            return f"{topics[0][:20]} - {meeting_type}"
        
        return f"{datetime.now().strftime('%m月%d日')} {meeting_type}"
    
    def format_minutes(self, minutes: MeetingMinutes, format_type: str = "markdown") -> str:
        """格式化会议纪要"""
        
        if format_type == "markdown":
            return self._format_markdown(minutes)
        elif format_type == "json":
            return json.dumps(asdict(minutes), ensure_ascii=False, indent=2)
        elif format_type == "text":
            return self._format_text(minutes)
        else:
            return self._format_markdown(minutes)
    
    def _format_markdown(self, minutes: MeetingMinutes) -> str:
        """Markdown格式"""
        lines = [
            f"# {minutes.title}",
            "",
            "## 📋 会议信息",
            f"- **日期**: {minutes.date}",
            f"- **时长**: {minutes.duration}",
            f"- **参与人**: {', '.join(minutes.participants) if minutes.participants else '未记录'}",
            "",
            "## 📌 会议议题",
        ]
        
        if minutes.topics:
            for i, topic in enumerate(minutes.topics, 1):
                lines.append(f"{i}. {topic}")
        else:
            lines.append("（未提取到明确议题）")
        
        lines.extend([
            "",
            "## ✅ 关键决策",
        ])
        
        if minutes.decisions:
            for decision in minutes.decisions:
                lines.append(f"- {decision}")
        else:
            lines.append("（未记录明确决策）")
        
        lines.extend([
            "",
            "## 📝 行动项 (TODO)",
        ])
        
        if minutes.action_items:
            for item in minutes.action_items:
                deadline_str = f" ⏰ {item.deadline}" if item.deadline else ""
                lines.append(f"- [ ] **{item.owner}**: {item.task}{deadline_str}")
        else:
            lines.append("（未提取到行动项）")
        
        lines.extend([
            "",
            "## 🔍 待跟进事项",
        ])
        
        if minutes.follow_ups:
            for follow_up in minutes.follow_ups:
                lines.append(f"- {follow_up}")
        else:
            lines.append("（无）")
        
        if minutes.source_url:
            lines.extend([
                "",
                "## 🔗 原始资料",
                f"- [录音/原文链接]({minutes.source_url})",
            ])
        
        lines.extend([
            "",
            "---",
            f"*生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}*",
            "*由 Meeting Minutes Generator 自动生成*",
        ])
        
        return "\n".join(lines)
    
    def _format_text(self, minutes: MeetingMinutes) -> str:
        """纯文本格式"""
        lines = [
            f"【{minutes.title}】",
            f"日期: {minutes.date}",
            f"时长: {minutes.duration}",
            f"参与人: {', '.join(minutes.participants) if minutes.participants else '未记录'}",
            "",
            "【议题】",
        ]
        
        for i, topic in enumerate(minutes.topics, 1):
            lines.append(f"{i}. {topic}")
        
        lines.extend(["", "【决策】"])
        for decision in minutes.decisions:
            lines.append(f"• {decision}")
        
        lines.extend(["", "【行动项】"])
        for item in minutes.action_items:
            deadline_str = f" [{item.deadline}]" if item.deadline else ""
            lines.append(f"□ {item.owner}: {item.task}{deadline_str}")
        
        return "\n".join(lines)
    
    def export_to_feishu(self, minutes: MeetingMinutes, folder_token: Optional[str] = None) -> Optional[str]:
        """导出到飞书文档"""
        if not HAS_REQUESTS:
            raise ImportError("需要安装 requests: pip install requests")
        
        folder_token = folder_token or self.config.get('feishu_folder_token')
        if not folder_token:
            raise ValueError("需要提供飞书文件夹 token")
        
        # 这里简化处理，实际应该调用飞书API创建文档
        # 返回模拟的文档链接
        doc_title = f"{minutes.title} - 会议纪要"
        
        # TODO: 实现完整的飞书API调用
        # 1. 获取tenant_access_token
        # 2. 创建文档
        # 3. 写入内容
        
        print(f"[模拟] 将导出到飞书文档: {doc_title}")
        print(f"[模拟] 文件夹Token: {folder_token}")
        
        return f"https://example.feishu.cn/docx/mock_{datetime.now().strftime('%Y%m%d%H%M%S')}"


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description='会议纪要生成器')
    parser.add_argument('--audio', '-a', help='音频文件路径')
    parser.add_argument('--text', '-t', help='文本内容或文件路径')
    parser.add_argument('--title', help='会议标题')
    parser.add_argument('--output', '-o', help='输出文件路径')
    parser.add_argument('--format', '-f', default='markdown', choices=['markdown', 'json', 'text'], help='输出格式')
    parser.add_argument('--feishu', action='store_true', help='导出到飞书')
    parser.add_argument('--config', '-c', help='配置文件路径')
    
    args = parser.parse_args()
    
    # 加载配置
    config = {}
    if args.config and os.path.exists(args.config):
        import tomllib
        with open(args.config, 'rb') as f:
            config = tomllib.load(f).get('meeting-minutes', {})
    
    # 初始化生成器
    generator = MeetingMinutesGenerator(config)
    
    # 获取输入内容
    content = ""
    
    if args.audio:
        print(f"正在转录音频: {args.audio}")
        content = generator.transcribe_audio(args.audio)
    elif args.text:
        if os.path.exists(args.text):
            with open(args.text, 'r', encoding='utf-8') as f:
                content = f.read()
        else:
            content = args.text
    else:
        # 从stdin读取
        if not sys.stdin.isatty():
            content = sys.stdin.read()
    
    if not content:
        print("错误: 未提供输入内容")
        parser.print_help()
        sys.exit(1)
    
    print("正在生成会议纪要...")
    
    # 生成会议纪要
    minutes = generator.generate_minutes(content, args.title)
    
    # 格式化输出
    output = generator.format_minutes(minutes, args.format)
    
    # 输出或保存
    if args.output:
        with open(args.output, 'w', encoding='utf-8') as f:
            f.write(output)
        print(f"已保存到: {args.output}")
    else:
        print("\n" + "="*50)
        print(output)
        print("="*50)
    
    # 导出到飞书
    if args.feishu:
        doc_url = generator.export_to_feishu(minutes)
        if doc_url:
            print(f"\n飞书文档链接: {doc_url}")
    
    # 打印统计
    print(f"\n📊 统计:")
    print(f"  - 议题数: {len(minutes.topics)}")
    print(f"  - 决策数: {len(minutes.decisions)}")
    print(f"  - 行动项: {len(minutes.action_items)}")
    print(f"  - 参与人: {len(minutes.participants)}")


if __name__ == '__main__':
    main()
