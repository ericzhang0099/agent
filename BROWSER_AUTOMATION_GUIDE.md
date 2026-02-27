# CEO Kimi Claw Browser Automation - Implementation Guide
# 浏览器自动化实施指南

## 快速开始 (5分钟)

### 1. 安装依赖

```bash
# 安装Playwright
pip install playwright

# 安装Chromium浏览器
playwright install chromium

# 可选：安装browser-use库
pip install browser-use
```

### 2. 验证安装

```bash
python3 browser_automation_demo.py
```

### 3. 核心模块使用

```python
import asyncio
from browser_manager import BrowserManager

async def quick_start():
    # 启动浏览器
    browser = await BrowserManager(headless=True).start()
    
    # 导航到页面
    await browser.navigate("https://example.com")
    
    # 获取AI Snapshot
    snapshot = await browser.get_snapshot(mode="ai")
    print(f"发现 {len(snapshot['elements'])} 个交互元素")
    
    # 截图
    await browser.screenshot(path="screenshot.png")
    
    # 关闭浏览器
    await browser.close()

asyncio.run(quick_start())
```

## 三种工作模式

### 研究模式 (ResearchMode)

用于自动化信息收集和研究报告生成。

```python
from research_mode import ResearchMode

async def research_example():
    researcher = await ResearchMode().start(headless=True)
    
    report = await researcher.research_topic(
        topic="AI agent browser automation",
        sources=["https://openclaw.ai", "https://docs.openclaw.ai"],
        depth=1
    )
    
    print(report["summary"])
    await researcher.close()
```

### 监控模式 (MonitorMode)

用于定时检测网页变更。

```python
from monitor_mode import MonitorMode, MonitorConfig

async def monitor_example():
    monitor = await MonitorMode().start(headless=True)
    
    # 添加监控任务
    monitor_id = monitor.add_monitor(MonitorConfig(
        url="https://openclaw.ai/blog",
        name="OpenClaw Blog",
        selector="article",
        interval_minutes=30,
        on_change=lambda r: print(f"变更 detected: {r.config.name}")
    ))
    
    # 执行检查
    result = await monitor.check_once(monitor_id)
    print(f"变更: {result.changed}")
    
    await monitor.close()
```

### 执行模式 (ExecuteMode)

用于自动化浏览器操作。

```python
from browser_manager import BrowserManager

async def execute_example():
    browser = await BrowserManager(headless=False).start()
    
    # 导航
    await browser.navigate("https://github.com/login")
    
    # 获取快照
    snapshot = await browser.get_snapshot(mode="ai")
    
    # 根据AI决策执行操作
    # 例如：点击元素 [1]
    await browser.click(1)
    
    # 输入文本到元素 [2]
    await browser.type_text(2, "username", clear=True)
    
    await browser.close()
```

## OpenClaw集成

### 添加到OpenClaw CLI

编辑 `~/.openclaw/skills/browser-automation/SKILL.md`:

```markdown
# Browser Automation Skill

## 研究模式

### browser-research start
启动研究任务

Parameters:
- topic: 研究主题
- sources: 起始URL列表 (逗号分隔)
- depth: 研究深度 (默认: 1)

Example:
```
browser-research start --topic "AI automation" --sources "https://..." --depth 2
```

### browser-research status
查看研究状态

Parameters:
- session-id: 会话ID

### browser-monitor add
添加监控任务

Parameters:
- url: 监控URL
- selector: CSS选择器 (可选)
- interval: 检查间隔(分钟)

## 监控模式

### browser-monitor list
列出所有监控任务

### browser-monitor check
立即执行检查

Parameters:
- monitor-id: 监控ID

### browser-monitor stats
查看监控统计
```

### Python集成

```python
from openclaw_integration import OpenClawBrowserAutomation

# 初始化
automation = OpenClawBrowserAutomation()

# 执行研究任务
report = await automation.execute_task("research", {
    "session_id": "session_001",
    "topic": "AI agent browser automation",
    "sources": ["https://openclaw.ai"],
    "depth": 1
})

# 添加监控
result = await automation.execute_task("monitor_add", {
    "session_id": "monitor_001",
    "config": {
        "url": "https://openclaw.ai/blog",
        "name": "Blog Monitor",
        "interval_minutes": 30
    }
})
```

## 配置选项

### BrowserManager配置

```python
browser = BrowserManager(
    headless=True,           # 无头模式
    profile="openclaw"       # 配置文件: openclaw/chrome/remote
)
```

### 快照模式

```python
# AI Snapshot - 数字引用，适合LLM
snapshot = await browser.get_snapshot(mode="ai")
# 输出: [{"ref": 1, "tag": "button", "text": "Submit"}, ...]

# Role Snapshot - ARIA引用，更稳定
snapshot = await browser.get_snapshot(mode="role")
# 输出: [{"ref": "e12", "role": "button", "name": "Submit"}, ...]
```

## 故障排除

### 常见问题

1. **Chromium未安装**
   ```bash
   playwright install chromium
   ```

2. **权限错误**
   ```python
   browser = BrowserManager(headless=True)
   # 确保使用 --no-sandbox 参数
   ```

3. **页面加载超时**
   ```python
   await browser.navigate(url, wait_until="domcontentloaded")
   ```

### 调试模式

```python
# 非无头模式，可见浏览器
browser = BrowserManager(headless=False)

# 慢速执行，便于观察
await browser.page.slow_mo(1000)
```

## 性能优化

### 并发执行

```python
# 并行处理多个URL
async def process_urls(urls):
    tasks = [process_single(url) for url in urls]
    return await asyncio.gather(*tasks)
```

### 资源管理

```python
# 及时关闭浏览器
await browser.close()

# 限制并发浏览器实例
semaphore = asyncio.Semaphore(5)
```

## 安全注意事项

1. **不要在生产环境使用headless=False**
2. **敏感操作需要人工确认**
3. **遵守网站的robots.txt**
4. **控制请求频率，避免被封禁**

## 参考资源

- [Playwright文档](https://playwright.dev/python/)
- [browser-use GitHub](https://github.com/browser-use/browser-use)
- [OpenClaw Browser文档](https://docs.openclaw.ai/tools/browser)
