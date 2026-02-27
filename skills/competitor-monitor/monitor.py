#!/usr/bin/env python3
"""
竞品监控警报器 - 主程序
自动监控AI Agent/LLM领域竞品动态
"""

import json
import os
import sys
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional
import subprocess

# 竞品配置
COMPETITORS = {
    "openai": {
        "name": "OpenAI",
        "type": "LLM/API",
        "keywords": ["OpenAI", "GPT-4", "GPT-5", "ChatGPT", "o1", "o3"],
        "sources": ["techcrunch", "theverge", "arxiv", "openai.com/blog"],
        "github": "openai",
        "priority": "high"
    },
    "anthropic": {
        "name": "Anthropic",
        "type": "LLM/Agent",
        "keywords": ["Anthropic", "Claude", "Claude 3", "Claude 4", "Sonnet", "Opus"],
        "sources": ["techcrunch", "anthropic.com/news"],
        "github": "anthropics",
        "priority": "high"
    },
    "google_deepmind": {
        "name": "Google DeepMind",
        "type": "LLM/Research",
        "keywords": ["Gemini", "Google AI", "DeepMind", "Gemini 2", "Bard"],
        "sources": ["blog.google", "deepmind.com"],
        "github": "google",
        "priority": "high"
    },
    "microsoft_copilot": {
        "name": "Microsoft Copilot",
        "type": "AI Agent",
        "keywords": ["Copilot", "Microsoft AI", "GitHub Copilot", "Azure AI"],
        "sources": ["microsoft.com/blog", "techcrunch"],
        "github": "microsoft",
        "priority": "high"
    },
    "meta_ai": {
        "name": "Meta AI",
        "type": "LLM/开源",
        "keywords": ["Llama", "Meta AI", "LLaMA 3", "LLaMA 4", "PyTorch"],
        "sources": ["ai.meta.com", "techcrunch"],
        "github": "meta-llama",
        "priority": "medium"
    },
    "autogpt": {
        "name": "AutoGPT",
        "type": "AI Agent框架",
        "keywords": ["AutoGPT", "Auto-GPT", "自主AI Agent"],
        "sources": ["github", "reddit"],
        "github": "Significant-Gravitas/AutoGPT",
        "priority": "medium"
    },
    "langchain": {
        "name": "LangChain",
        "type": "LLM框架",
        "keywords": ["LangChain", "LangSmith", "LangGraph"],
        "sources": ["techcrunch", "blog.langchain.com"],
        "github": "langchain-ai",
        "priority": "medium"
    },
    "crewai": {
        "name": "CrewAI",
        "type": "AI Agent框架",
        "keywords": ["CrewAI", "Multi-Agent", "Agent团队"],
        "sources": ["github", "docs.crewai.com"],
        "github": "joaomdmoura/crewAI",
        "priority": "medium"
    }
}

# 监控维度配置
MONITOR_DIMENSIONS = {
    "product_update": {
        "name": "产品更新",
        "keywords": ["发布", "launch", "release", "update", "新版本", "新功能"],
        "alert_threshold": "immediate"
    },
    "funding": {
        "name": "融资新闻",
        "keywords": ["融资", "funding", "investment", "估值", "valuation", "Series A", "Series B", "Series C"],
        "alert_threshold": "immediate"
    },
    "tech_release": {
        "name": "技术发布",
        "keywords": ["论文", "paper", "arxiv", "研究", "research", "breakthrough", "模型", "model"],
        "alert_threshold": "immediate"
    },
    "github": {
        "name": "GitHub动态",
        "keywords": ["release", "version", "v1.", "v2.", "major"],
        "alert_threshold": "major_release"
    }
}

class CompetitorMonitor:
    def __init__(self):
        self.workspace = "/root/.openclaw/workspace"
        self.memory_dir = os.path.join(self.workspace, "memory", "competitor-monitor")
        self.ensure_directories()
        
    def ensure_directories(self):
        """确保目录结构存在"""
        os.makedirs(self.memory_dir, exist_ok=True)
        
    def search_news(self, query: str, count: int = 5) -> List[Dict]:
        """搜索新闻"""
        try:
            # 使用web_search工具搜索
            result = subprocess.run(
                ["python3", "-c", f"""
import sys
sys.path.insert(0, '{self.workspace}')
from tools.web_search import web_search
results = web_search('{query}', count={count})
print(json.dumps(results, ensure_ascii=False))
"""],
                capture_output=True,
                text=True,
                timeout=30
            )
            if result.returncode == 0:
                return json.loads(result.stdout)
        except Exception as e:
            print(f"搜索失败: {e}")
        return []
        
    def monitor_all(self) -> Dict[str, Any]:
        """执行完整监控"""
        report = {
            "timestamp": datetime.now().isoformat(),
            "competitors": {},
            "alerts": [],
            "summary": {}
        }
        
        for comp_id, comp_config in COMPETITORS.items():
            print(f"🔍 监控 {comp_config['name']}...")
            comp_data = self.monitor_competitor(comp_id, comp_config)
            report["competitors"][comp_id] = comp_data
            
            # 检查警报
            alerts = self.check_alerts(comp_id, comp_data)
            report["alerts"].extend(alerts)
            
        # 生成汇总
        report["summary"] = self.generate_summary(report)
        
        return report
        
    def monitor_competitor(self, comp_id: str, config: Dict) -> Dict:
        """监控单个竞品"""
        data = {
            "name": config["name"],
            "last_check": datetime.now().isoformat(),
            "news": [],
            "updates": []
        }
        
        # 搜索产品更新
        for keyword in config["keywords"][:3]:
            query = f"{keyword} 发布 更新 2025 2026"
            news = self.search_news(query, count=3)
            data["news"].extend(news)
            
        # 搜索融资新闻
        funding_query = f"{config['name']} 融资 funding investment"
        funding_news = self.search_news(funding_query, count=3)
        data["news"].extend(funding_news)
        
        return data
        
    def check_alerts(self, comp_id: str, data: Dict) -> List[Dict]:
        """检查是否需要触发警报"""
        alerts = []
        
        for news in data.get("news", []):
            title = news.get("title", "").lower()
            snippet = news.get("snippet", "").lower()
            content = title + " " + snippet
            
            # 检查重大更新
            if any(kw in content for kw in ["融资", "funding", "investment", "million", "billion"]):
                alerts.append({
                    "type": "funding",
                    "competitor": data["name"],
                    "title": news.get("title"),
                    "url": news.get("url"),
                    "severity": "high"
                })
                
            # 检查产品发布
            if any(kw in content for kw in ["launch", "release", "发布", "推出"]):
                alerts.append({
                    "type": "product",
                    "competitor": data["name"],
                    "title": news.get("title"),
                    "url": news.get("url"),
                    "severity": "medium"
                })
                
        return alerts
        
    def generate_summary(self, report: Dict) -> Dict:
        """生成监控汇总"""
        total_news = sum(len(c.get("news", [])) for c in report["competitors"].values())
        total_alerts = len(report["alerts"])
        high_priority = len([a for a in report["alerts"] if a.get("severity") == "high"])
        
        return {
            "total_competitors": len(COMPETITORS),
            "total_news": total_news,
            "total_alerts": total_alerts,
            "high_priority_alerts": high_priority,
            "monitor_time": report["timestamp"]
        }
        
    def save_report(self, report: Dict):
        """保存监控报告"""
        filename = f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        filepath = os.path.join(self.memory_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(report, f, ensure_ascii=False, indent=2)
            
        print(f"📄 报告已保存: {filepath}")
        return filepath
        
    def format_report_markdown(self, report: Dict) -> str:
        """格式化为Markdown报告"""
        summary = report["summary"]
        
        md = f"""# 🚨 竞品监控报告

**生成时间**: {report['timestamp']}

## 📊 监控汇总

| 指标 | 数值 |
|------|------|
| 监控竞品数 | {summary['total_competitors']} |
| 发现新闻数 | {summary['total_news']} |
| 触发警报数 | {summary['total_alerts']} |
| 高优先级警报 | {summary['high_priority_alerts']} |

## 🎯 监控竞品列表

"""
        for comp_id, comp in COMPETITORS.items():
            md += f"- **{comp['name']}** ({comp['type']}) - 优先级: {comp['priority']}\n"
            
        md += "\n## 🔔 最新警报\n\n"
        
        if report["alerts"]:
            for alert in report["alerts"][:10]:
                emoji = "🔴" if alert["severity"] == "high" else "🟡"
                md += f"{emoji} **[{alert['competitor']}]** {alert['title']}\n"
                if alert.get("url"):
                    md += f"   - 链接: {alert['url']}\n"
        else:
            md += "暂无新警报\n"
            
        md += "\n## 📰 最新动态\n\n"
        
        for comp_id, comp_data in report["competitors"].items():
            if comp_data.get("news"):
                md += f"### {comp_data['name']}\n"
                for news in comp_data["news"][:3]:
                    md += f"- {news.get('title', 'N/A')}\n"
                    if news.get('url'):
                        md += f"  - {news.get('url')}\n"
                md += "\n"
                
        return md

def main():
    monitor = CompetitorMonitor()
    
    print("=" * 60)
    print("🚀 竞品监控警报器启动")
    print("=" * 60)
    
    # 执行监控
    report = monitor.monitor_all()
    
    # 保存报告
    monitor.save_report(report)
    
    # 输出Markdown报告
    md_report = monitor.format_report_markdown(report)
    
    # 保存Markdown版本
    md_path = os.path.join(monitor.memory_dir, f"report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")
    with open(md_path, 'w', encoding='utf-8') as f:
        f.write(md_report)
    
    print("\n" + "=" * 60)
    print(md_report)
    print("=" * 60)
    
    # 如果有高优先级警报，输出警告
    high_priority = [a for a in report["alerts"] if a.get("severity") == "high"]
    if high_priority:
        print(f"\n⚠️ 发现 {len(high_priority)} 个高优先级警报！")
        for alert in high_priority:
            print(f"  - [{alert['competitor']}] {alert['title']}")
    
    print(f"\n✅ 监控完成！报告已保存至: {md_path}")
    
    return report

if __name__ == "__main__":
    main()
