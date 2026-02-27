---
name: competitor-monitor
description: AI Agent/LLM领域竞品监控系统，自动追踪竞品动态并发送警报
version: 1.0.0
author: KCGS
schedule: 0 * * * *
tags: [competitor, monitoring, alert, ai-agent, llm]
---

# 竞品监控警报器 (Competitor Monitor)

自动监控AI Agent/LLM领域核心竞品动态，包括产品更新、融资新闻、技术发布、GitHub动态等维度。

## 监控竞品列表

| 竞品名称 | 类型 | 监控重点 |
|---------|------|---------|
| OpenAI | LLM/API | GPT系列更新、API发布、融资 |
| Anthropic | LLM/Agent | Claude更新、Agent产品、融资 |
| Google DeepMind | LLM/Research | Gemini更新、研究论文、产品发布 |
| Microsoft Copilot | AI Agent | 产品更新、企业功能、集成动态 |
| Meta AI | LLM/开源 | Llama更新、开源动态、产品发布 |
| AutoGPT | AI Agent框架 | GitHub动态、版本发布、社区活跃度 |
| LangChain | LLM框架 | 版本更新、集成动态、融资 |
| CrewAI | AI Agent框架 | 产品更新、融资、社区动态 |

## 监控维度

1. **产品更新** - 新功能发布、版本更新、产品公告
2. **融资新闻** - 融资轮次、金额、投资方
3. **技术发布** - 论文发布、技术博客、研究突破
4. **GitHub动态** - Star数变化、Release发布、Commit活跃度

## 警报触发条件

- 重大产品发布/更新
- 融资新闻（金额>1000万美元）
- 重要技术突破/论文
- GitHub仓库重大更新（major release）
- 竞争对手战略动向

## 使用方法

```bash
# 手动运行监控
openclaw run competitor-monitor

# 查看监控报告
openclaw run competitor-monitor --report

# 测试警报
openclaw run competitor-monitor --test-alert
```

## 配置

在 `~/.openclaw/config.toml` 中添加：
```toml
[competitor-monitor]
alert_channel = "feishu"  # 或 "email", "slack"
feishu_webhook = "your_webhook_url"
```

## 输出

- 监控报告保存至 `memory/competitor-monitor/`
- 警报通过配置渠道发送
- 每日/每周汇总报告
