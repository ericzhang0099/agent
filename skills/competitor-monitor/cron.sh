#!/bin/bash
# 竞品监控警报器 - 定时任务执行脚本
# 由系统cron每小时调用

SKILL_DIR="/root/.openclaw/workspace/skills/competitor-monitor"
MEMORY_DIR="/root/.openclaw/workspace/memory/competitor-monitor"
LOG_FILE="$MEMORY_DIR/monitor.log"

# 记录执行时间
echo "[$(date '+%Y-%m-%d %H:%M:%S')] 开始执行竞品监控扫描..." >> "$LOG_FILE"

# 执行监控
cd "$SKILL_DIR"
python3 monitor.py >> "$LOG_FILE" 2>&1

# 检查是否有高优先级警报
LATEST_REPORT=$(ls -t "$MEMORY_DIR"/report_*.json 2>/dev/null | head -1)
if [ -n "$LATEST_REPORT" ]; then
    HIGH_ALERTS=$(python3 -c "
import json
with open('$LATEST_REPORT') as f:
    data = json.load(f)
    alerts = data.get('alerts', [])
    high = [a for a in alerts if a.get('severity') == 'high']
    print(len(high))
" 2>/dev/null || echo "0")
    
    if [ "$HIGH_ALERTS" -gt 0 ]; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] ⚠️ 发现 $HIGH_ALERTS 个高优先级警报!" >> "$LOG_FILE"
        # 这里可以添加通知逻辑 (feishu/email/slack)
    fi
fi

echo "[$(date '+%Y-%m-%d %H:%M:%S')] 监控扫描完成" >> "$LOG_FILE"
echo "---" >> "$LOG_FILE"
