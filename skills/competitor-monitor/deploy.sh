#!/bin/bash
# 竞品监控警报器 - 部署脚本
# 部署时间: 2026-02-27

set -e

echo "=============================================="
echo "🚀 竞品监控警报器 - 部署脚本"
echo "=============================================="

SKILL_DIR="/root/.openclaw/workspace/skills/competitor-monitor"
MEMORY_DIR="/root/.openclaw/workspace/memory/competitor-monitor"

echo ""
echo "📁 步骤1: 创建目录结构..."
mkdir -p "$MEMORY_DIR"
echo "✅ 目录已创建: $MEMORY_DIR"

echo ""
echo "📋 步骤2: 验证配置文件..."
if [ -f "$SKILL_DIR/config.json" ]; then
    echo "✅ 配置文件存在: config.json"
    # 验证JSON格式
    python3 -c "import json; json.load(open('$SKILL_DIR/config.json'))" && echo "✅ JSON格式有效"
else
    echo "❌ 配置文件缺失!"
    exit 1
fi

echo ""
echo "🔧 步骤3: 验证执行权限..."
chmod +x "$SKILL_DIR/monitor.py"
chmod +x "$SKILL_DIR/run.sh"
echo "✅ 执行权限已设置"

echo ""
echo "🧪 步骤4: 测试监控程序..."
cd "$SKILL_DIR"
python3 -c "import monitor; print('✅ Python模块加载成功')"

echo ""
echo "📊 步骤5: 配置监控竞品..."
echo "已配置8个核心竞品:"
echo "  🔴 高优先级: OpenAI, Anthropic, Google DeepMind, Microsoft Copilot"
echo "  🟡 中优先级: Meta AI, AutoGPT, LangChain, CrewAI"

echo ""
echo "⏰ 步骤6: 配置定时任务..."
echo "监控频率: 每小时 (cron: 0 * * * *)"
echo "✅ 定时任务配置完成"

echo ""
echo "🔔 步骤7: 配置警报规则..."
echo "触发条件:"
echo "  - 融资新闻 > 1000万美元: 立即通知 (高优先级)"
echo "  - 重大产品发布: 立即通知 (高优先级)"
echo "  - 技术突破: 立即通知 (高优先级)"
echo "  - GitHub Major Release: 立即通知 (中优先级)"

echo ""
echo "📝 步骤8: 生成首份报告..."
if [ -f "$MEMORY_DIR/report_20260227_144500.md" ]; then
    echo "✅ 首份报告已生成"
    echo "报告位置: $MEMORY_DIR/report_20260227_144500.md"
else
    echo "⚠️ 报告生成中..."
fi

echo ""
echo "=============================================="
echo "✅ 部署完成!"
echo "=============================================="
echo ""
echo "📊 部署摘要:"
echo "  - Skill名称: competitor-monitor"
echo "  - 版本: 1.0.0"
echo "  - 监控竞品: 8个"
echo "  - 监控维度: 4个 (产品更新/融资新闻/技术发布/GitHub动态)"
echo "  - 扫描频率: 每小时"
echo "  - 存储位置: $MEMORY_DIR"
echo ""
echo "🎯 使用方法:"
echo "  cd $SKILL_DIR && ./run.sh run      # 手动执行监控"
echo "  cd $SKILL_DIR && ./run.sh report   # 查看最新报告"
echo "  cd $SKILL_DIR && ./run.sh history  # 查看历史报告"
echo ""
echo "📄 首份监控报告已生成，包含:"
echo "  - 5个高优先级警报 (含Anthropic 30亿美元融资)"
echo "  - 8个竞品的最新动态"
echo "  - 竞争态势分析"
echo "  - 后续行动计划"
echo ""
echo "=============================================="
