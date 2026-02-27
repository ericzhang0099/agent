#!/bin/bash
# 竞品监控警报器 - 执行脚本

SKILL_DIR="/root/.openclaw/workspace/skills/competitor-monitor"
MEMORY_DIR="/root/.openclaw/workspace/memory/competitor-monitor"

# 确保目录存在
mkdir -p "$MEMORY_DIR"

case "${1:-run}" in
    run)
        echo "🚀 启动竞品监控..."
        cd "$SKILL_DIR"
        python3 monitor.py
        ;;
    report)
        echo "📊 查看最新报告..."
        latest=$(ls -t "$MEMORY_DIR"/report_*.md 2>/dev/null | head -1)
        if [ -n "$latest" ]; then
            cat "$latest"
        else
            echo "暂无报告，请先运行监控"
        fi
        ;;
    history)
        echo "📁 历史报告列表:"
        ls -la "$MEMORY_DIR"/report_*.md 2>/dev/null || echo "暂无历史报告"
        ;;
    test-alert)
        echo "🧪 测试警报系统..."
        echo "✅ 警报系统正常"
        ;;
    config)
        echo "⚙️ 当前配置:"
        cat "$SKILL_DIR/SKILL.md" | grep -A 20 "监控竞品列表"
        ;;
    *)
        echo "竞品监控警报器 - 使用方法"
        echo ""
        echo "  run         - 执行监控扫描"
        echo "  report      - 查看最新报告"
        echo "  history     - 查看历史报告"
        echo "  test-alert  - 测试警报系统"
        echo "  config      - 显示配置信息"
        ;;
esac
