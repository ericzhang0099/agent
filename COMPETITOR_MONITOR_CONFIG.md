# 竞品监控配置
# 竞品监控警报器 Skill 配置文件
# 创建时间: 2026-02-27
# 监控周期: 每小时

---

## 监控竞品列表

### 1. OpenAI / ChatGPT
- **类型**: LLM / Chatbot
- **监控维度**: 
  - 产品更新: https://openai.com/blog
  - GitHub: https://github.com/openai
  - API更新: https://platform.openai.com/docs/updates
- **重要性**: ⭐⭐⭐⭐⭐
- **警报阈值**: 任何产品发布、API变更、定价调整

### 2. Anthropic / Claude
- **类型**: LLM / AI Agent
- **监控维度**:
  - 产品更新: https://www.anthropic.com/news
  - 研究博客: https://www.anthropic.com/research
  - GitHub: https://github.com/anthropics
- **重要性**: ⭐⭐⭐⭐⭐
- **警报阈值**: 新模型发布、研究突破、安全相关更新

### 3. Google / Gemini
- **类型**: LLM / AI Platform
- **监控维度**:
  - 产品更新: https://blog.google/technology/ai/
  - 开发者文档: https://ai.google.dev/
- **重要性**: ⭐⭐⭐⭐⭐
- **警报阈值**: 重大产品发布、API更新

### 4. Meta / Llama
- **类型**: Open Source LLM
- **监控维度**:
  - GitHub: https://github.com/meta-llama
  - 研究博客: https://ai.meta.com/blog/
- **重要性**: ⭐⭐⭐⭐
- **警报阈值**: 新版本发布、许可证变更

### 5. AutoGPT
- **类型**: AI Agent Framework
- **监控维度**:
  - GitHub: https://github.com/Significant-Gravitas/AutoGPT
  - 官方文档: https://docs.agpt.co/
- **重要性**: ⭐⭐⭐⭐⭐
- **警报阈值**: 重大版本更新、架构变更

### 6. MetaGPT
- **类型**: Multi-Agent Framework
- **监控维度**:
  - GitHub: https://github.com/geekan/MetaGPT
  - 官方文档: https://docs.metagpt.io/
- **重要性**: ⭐⭐⭐⭐⭐
- **警报阈值**: 新版本发布、多Agent协作功能更新

### 7. CrewAI
- **类型**: Multi-Agent Framework
- **监控维度**:
  - GitHub: https://github.com/joaomdmoura/crewAI
  - 官方文档: https://docs.crewai.com/
- **重要性**: ⭐⭐⭐⭐
- **警报阈值**: 重大功能更新

### 8. Dify
- **类型**: LLM Application Platform
- **监控维度**:
  - GitHub: https://github.com/langgenius/dify
  - 官方文档: https://docs.dify.ai/
- **重要性**: ⭐⭐⭐⭐
- **警报阈值**: 重大版本更新

### 9. LangChain
- **类型**: LLM Framework
- **监控维度**:
  - GitHub: https://github.com/langchain-ai/langchain
  - 官方博客: https://blog.langchain.dev/
- **重要性**: ⭐⭐⭐⭐
- **警报阈值**: 重大架构更新、新版本发布

### 10. OpenClaw / ClawHub
- **类型**: AI Agent Runtime / Skill Market
- **监控维度**:
  - GitHub: https://github.com/openclaw/openclaw
  - Skill市场: https://clawhub.ai/
- **重要性**: ⭐⭐⭐⭐⭐
- **警报阈值**: 新版本发布、重要Skill上架、安全公告

---

## 监控维度配置

### 产品更新
- **监控频率**: 每小时
- **监控来源**: 官方博客、产品更新日志
- **警报条件**: 新功能发布、产品改版、定价调整

### 融资新闻
- **监控频率**: 每4小时
- **监控来源**: TechCrunch、Crunchbase、36氪
- **警报条件**: B轮及以上融资、战略投资、并购消息

### 技术发布
- **监控频率**: 每小时
- **监控来源**: arXiv、GitHub Releases、研究博客
- **警报条件**: 新论文发布、开源项目重大更新、突破性技术

### GitHub动态
- **监控频率**: 每小时
- **监控指标**: Star增长、Release发布、重要PR合并
- **警报条件**: 重大版本发布、架构变更、安全修复

---

## 警报配置

### 警报级别
| 级别 | 条件 | 响应时间 | 通知方式 |
|------|------|---------|---------|
| 🔴 P0 | 竞争对手发布颠覆性产品/技术 | 立即 | 飞书+邮件+短信 |
| 🟠 P1 | 重要功能更新/重大融资 | 15分钟 | 飞书+邮件 |
| 🟡 P2 | 常规更新/技术博客 | 1小时 | 飞书 |
| 🟢 P3 | 社区动态/ minor更新 | 4小时 | 日报汇总 |

### 通知渠道
- **飞书**: 实时推送，@CEO和董事长
- **邮件**: 详细报告，每小时汇总
- **短信**: P0级别紧急通知

---

## 首份监控报告模板

```markdown
# 竞品监控报告 | 2026-02-27 14:30

## 监控概况
- 监控竞品数: 10
- 监控维度: 4
- 新发现: X条
- 警报级别: P0/X条, P1/X条, P2/X条

## 重要发现

### 🔴 P0 警报
1. [竞品名称] - [发现内容]
   - 影响评估: [高/中/低]
   - 建议行动: [立即响应/持续观察]

### 🟠 P1 警报
...

### 🟡 P2 警报
...

## 趋势分析
- [本周/本月重要趋势]
- [对我们团队的启示]

## 下一步行动
1. [行动项1]
2. [行动项2]
```

---

## 立即启动命令

```bash
# 启动竞品监控
python3 /root/.openclaw/skills/竞品监控警报器/竞品监控警报器.py --config /root/.openclaw/workspace/COMPETITOR_MONITOR_CONFIG.md --interval 3600

# 生成首份报告
python3 /root/.openclaw/skills/竞品监控警报器/竞品监控警报器.py --report --output /root/.openclaw/workspace/reports/competitor_report_$(date +%Y%m%d_%H%M).md
```

---

*配置完成时间: 2026-02-27 14:30*  
*配置Agent: CEO Kimi Claw*  
*下次更新: 动态调整*
