#!/usr/bin/env python3
"""
CEO Kimi Claw Browser Automation - Quick Start Demo
快速开始演示脚本

使用方法:
    python3 browser_automation_demo.py

功能:
    1. 验证Playwright环境
    2. 演示基础浏览器操作
    3. 演示AI Snapshot功能
"""

import asyncio
import sys
import json

def check_environment():
    """检查环境依赖"""
    print("🔍 检查环境依赖...")
    
    try:
        import playwright
        print(f"  ✅ Playwright 已安装 (版本: {playwright.__version__})")
    except ImportError:
        print("  ❌ Playwright 未安装")
        print("  💡 安装命令: pip install playwright && playwright install chromium")
        return False
    
    try:
        from playwright.async_api import async_playwright
        print("  ✅ Playwright async API 可用")
    except ImportError as e:
        print(f"  ❌ Playwright API 导入失败: {e}")
        return False
    
    print("✅ 环境检查通过\n")
    return True

async def demo_basic_browser():
    """演示基础浏览器操作"""
    from playwright.async_api import async_playwright
    
    print("🚀 启动浏览器演示...")
    
    async with async_playwright() as p:
        # 启动浏览器
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        
        # 创建新页面
        page = await browser.new_page()
        print("  ✅ 浏览器已启动")
        
        # 导航到示例页面
        await page.goto("https://example.com")
        print(f"  ✅ 已导航到: {page.url}")
        
        # 获取页面标题
        title = await page.title()
        print(f"  📄 页面标题: {title}")
        
        # 截图
        await page.screenshot(path="/tmp/example_screenshot.png")
        print("  📸 截图已保存: /tmp/example_screenshot.png")
        
        # 关闭浏览器
        await browser.close()
        print("  ✅ 浏览器已关闭\n")

async def demo_ai_snapshot():
    """演示AI Snapshot功能"""
    from playwright.async_api import async_playwright
    
    print("🚀 启动AI Snapshot演示...")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        page = await browser.new_page()
        
        # 导航到更复杂的页面
        await page.goto("https://github.com/login")
        print(f"  ✅ 已导航到: {page.url}")
        
        # 获取AI Snapshot
        elements = await page.query_selector_all(
            'button, input, select, textarea, a, [role="button"]'
        )
        
        print(f"  🔍 发现 {len(elements)} 个交互元素:\n")
        
        snapshot = []
        for idx, elem in enumerate(elements[:10], 1):  # 只显示前10个
            try:
                tag = await elem.evaluate('el => el.tagName.toLowerCase()')
                elem_type = await elem.get_attribute('type') or ''
                name = await elem.get_attribute('name') or ''
                placeholder = await elem.get_attribute('placeholder') or ''
                text = await elem.inner_text()
                
                elem_info = {
                    "ref": idx,
                    "tag": tag,
                    "type": elem_type,
                    "name": name,
                    "text": text[:50] if text else '',
                    "placeholder": placeholder
                }
                snapshot.append(elem_info)
                
                # 打印简洁信息
                display_text = text[:30] if text else placeholder[:30] or name
                print(f"    [{idx}] <{tag}> {display_text}")
                
            except Exception as e:
                continue
        
        # 保存完整snapshot
        with open("/tmp/ai_snapshot.json", "w") as f:
            json.dump(snapshot, f, indent=2, ensure_ascii=False)
        print(f"\n  💾 完整snapshot已保存: /tmp/ai_snapshot.json")
        
        await browser.close()
        print("  ✅ 浏览器已关闭\n")

async def demo_research_mode():
    """演示研究模式概念"""
    from playwright.async_api import async_playwright
    
    print("🚀 启动研究模式演示...")
    print("  📚 研究主题: OpenClaw browser automation")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=['--no-sandbox', '--disable-setuid-sandbox']
        )
        page = await browser.new_page()
        
        # 模拟研究流程
        sources = [
            "https://openclaw.ai",
            "https://docs.openclaw.ai"
        ]
        
        findings = []
        for url in sources:
            try:
                await page.goto(url, timeout=10000)
                title = await page.title()
                
                # 提取主要内容
                content_selectors = ['main', 'article', '.content', 'body']
                content = ""
                for selector in content_selectors:
                    try:
                        elem = await page.query_selector(selector)
                        if elem:
                            content = await elem.inner_text()
                            if len(content) > 100:
                                break
                    except:
                        continue
                
                finding = {
                    "url": url,
                    "title": title,
                    "content_preview": content[:300] + "..." if len(content) > 300 else content
                }
                findings.append(finding)
                
                print(f"  ✅ 已收集: {title}")
                
            except Exception as e:
                print(f"  ⚠️ 无法访问 {url}: {e}")
        
        # 生成简单报告
        report = {
            "topic": "OpenClaw browser automation",
            "sources_analyzed": len(findings),
            "findings": findings
        }
        
        with open("/tmp/research_report.json", "w") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"\n  📊 研究报告已生成: /tmp/research_report.json")
        print(f"  📈 分析了 {len(findings)} 个来源")
        
        await browser.close()
        print("  ✅ 浏览器已关闭\n")

async def main():
    """主函数"""
    print("=" * 60)
    print("CEO Kimi Claw - Browser Automation Demo")
    print("浏览器自动化能力快速演示")
    print("=" * 60 + "\n")
    
    # 检查环境
    if not check_environment():
        print("\n❌ 环境检查失败，请先安装依赖:")
        print("   pip install playwright")
        print("   playwright install chromium")
        sys.exit(1)
    
    # 运行演示
    try:
        await demo_basic_browser()
        await demo_ai_snapshot()
        await demo_research_mode()
        
        print("=" * 60)
        print("✅ 所有演示已完成!")
        print("=" * 60)
        print("\n生成的文件:")
        print("  📸 /tmp/example_screenshot.png - 示例页面截图")
        print("  💾 /tmp/ai_snapshot.json - AI Snapshot数据")
        print("  📊 /tmp/research_report.json - 研究报告")
        
    except Exception as e:
        print(f"\n❌ 演示失败: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(main())
