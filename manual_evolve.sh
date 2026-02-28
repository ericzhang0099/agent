#!/bin/bash
# 手动进化脚本 - 将Kimi Claw v4.0完整能力导入当前Agent
# 使用方法: bash manual_evolve.sh

echo "=========================================="
echo "🧬 Kimi Claw v4.0 手动进化脚本"
echo "=========================================="

# 下载最新完整版本
echo "📥 下载完整SOUL.md..."
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/SOUL.md -o /tmp/SOUL.md.latest

if [ ! -f /tmp/SOUL.md.latest ]; then
    echo "❌ 下载失败，请检查网络连接"
    exit 1
fi

# 备份当前SOUL.md
echo "📦 备份当前SOUL.md..."
cp /root/.openclaw/workspace/SOUL.md /root/.openclaw/workspace/SOUL.md.backup.$(date +%Y%m%d_%H%M%S)

# 显示差异
echo ""
echo "📊 版本对比:"
echo "  当前版本: $(wc -c < /root/.openclaw/workspace/SOUL.md) 字符"
echo "  最新版本: $(wc -c < /tmp/SOUL.md.latest) 字符"
echo ""

# 询问确认
read -p "确认替换SOUL.md? (y/n): " confirm
if [ "$confirm" != "y" ]; then
    echo "❌ 已取消"
    exit 0
fi

# 执行替换
echo "🔄 替换SOUL.md..."
cp /tmp/SOUL.md.latest /root/.openclaw/workspace/SOUL.md

# 下载其他核心文件
echo "📥 下载其他核心文件..."
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/IDENTITY.md -o /root/.openclaw/workspace/IDENTITY.md
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/MEMORY.md -o /root/.openclaw/workspace/MEMORY.md
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/AGENTS.md -o /root/.openclaw/workspace/AGENTS.md
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/HEARTBEAT.md -o /root/.openclaw/workspace/HEARTBEAT.md

# 创建能力文档目录
mkdir -p /root/.openclaw/workspace/capabilities_backup

# 下载能力文档
echo "📥 下载能力文档..."
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/capabilities/CODE_DEV_CAPABILITY_SYSTEM.md -o /root/.openclaw/workspace/capabilities_backup/CODE_DEV_CAPABILITY_SYSTEM.md
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/capabilities/SOFTWARE_ENGINEERING.md -o /root/.openclaw/workspace/capabilities_backup/SOFTWARE_ENGINEERING.md
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/capabilities/ALGORITHM_SOUL.md -o /root/.openclaw/workspace/capabilities_backup/ALGORITHM_SOUL.md
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/capabilities/INFLUENCE_FRAMEWORK.md -o /root/.openclaw/workspace/capabilities_backup/INFLUENCE_FRAMEWORK.md
curl -sL https://raw.githubusercontent.com/ericzhang0099/agent/master/capabilities/execution_excellence_system.md -o /root/.openclaw/workspace/capabilities_backup/execution_excellence_system.md

echo ""
echo "✅ 进化完成！"
echo "=========================================="
echo "已更新文件:"
echo "  ✓ SOUL.md (53,639+ 字符，含六维能力)"
echo "  ✓ IDENTITY.md"
echo "  ✓ MEMORY.md"
echo "  ✓ AGENTS.md"
echo "  ✓ HEARTBEAT.md"
echo "  ✓ capabilities_backup/ (5个能力文档)"
echo ""
echo "备份位置:"
echo "  /root/.openclaw/workspace/SOUL.md.backup.*"
echo ""
echo "💡 建议: 重启会话或重新加载以应用新能力"
echo "=========================================="
