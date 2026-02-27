#!/bin/bash
#
# 记忆系统自动化脚本
# 用于定期维护和记忆流转
#

MEMORY_BASE="/root/.openclaw/workspace/memory"
DATE=$(date +%Y-%m-%d)
DATETIME=$(date +%Y-%m-%d_%H-%M-%S)

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log() {
    echo -e "${GREEN}[$(date +%H:%M:%S)]${NC} $1"
}

warn() {
    echo -e "${YELLOW}[$(date +%H:%M:%S)] WARNING:${NC} $1"
}

error() {
    echo -e "${RED}[$(date +%H:%M:%S)] ERROR:${NC} $1"
}

# ==================== 每日维护 ====================

daily_maintenance() {
    log "开始每日维护任务..."
    
    # 1. 清理过期短期记忆 (超过24小时)
    log "检查短期记忆..."
    if [ -f "$MEMORY_BASE/short-term/context_stack.json" ]; then
        # 这里可以添加 TTL 检查逻辑
        log "短期记忆检查完成"
    fi
    
    # 2. 创建每日归档目录
    mkdir -p "$MEMORY_BASE/archive/daily"
    
    # 3. 生成每日摘要
    log "生成每日记忆摘要..."
    cat > "$MEMORY_BASE/archive/daily/$DATE.md" << EOF
# 每日记忆摘要 - $DATE

## 生成时间
$(date)

## 活跃项目
$(ls -1 $MEMORY_BASE/mid-term/projects/ 2>/dev/null | sed 's/^/- /')

## 今日归档
- 短期记忆: $MEMORY_BASE/archive/sessions/$(date +%Y)/$(date +%m)/$(date +%d)/

## 系统状态
- 长期记忆条目: $(find $MEMORY_BASE/long-term -name "*.md" | wc -l)
- 活跃项目数: $(ls -1 $MEMORY_BASE/mid-term/projects/ 2>/dev/null | wc -l)

---
*自动生成于 $(date)*
EOF
    
    log "每日维护完成"
}

# ==================== 每周整理 ====================

weekly_cleanup() {
    log "开始每周整理任务..."
    
    # 1. 归档已完成项目
    log "检查已完成项目..."
    for project_dir in "$MEMORY_BASE/mid-term/projects"/*; do
        if [ -d "$project_dir" ]; then
            project_id=$(basename "$project_dir")
            tasks_file="$project_dir/tasks.json"
            
            if [ -f "$tasks_file" ]; then
                # 检查是否所有任务都已完成
                total=$(grep -c '"status"' "$tasks_file" 2>/dev/null || echo 0)
                completed=$(grep -c '"completed"' "$tasks_file" 2>/dev/null || echo 0)
                
                if [ "$total" -gt 0 ] && [ "$total" -eq "$completed" ]; then
                    warn "项目 $project_id 所有任务已完成，建议归档"
                    # 自动归档 (可选)
                    # node "$MEMORY_BASE/scripts/memory_cli.js" promote "$project_id"
                fi
            fi
        fi
    done
    
    # 2. 清理旧归档 (保留最近90天)
    log "清理旧归档..."
    find "$MEMORY_BASE/archive/sessions" -type f -mtime +90 -delete 2>/dev/null || true
    
    log "每周整理完成"
}

# ==================== 会话启动 ====================

session_start() {
    log "会话启动..."
    
    # 更新会话状态
    SESSION_ID="session_${DATETIME}"
    
    cat > "$MEMORY_BASE/short-term/current_session.md" << EOF
# 当前会话状态

## 会话元信息
- **会话ID**: $SESSION_ID
- **开始时间**: $(date '+%Y-%m-%d %H:%M:%S %Z')
- **任务状态**: 🟡 进行中

## 会话上下文栈

### 当前主题
待填写

### 已完成的步骤
1. ✅ 会话启动

## 临时变量

\`\`\`json
{
  "session_id": "$SESSION_ID",
  "started_at": "$(date -Iseconds)"
}
\`\`\`

## 当前待办

- [ ] 加载活跃项目上下文

## 会话笔记

> 会话自动启动

---
*最后更新: $(date)*
EOF
    
    # 更新上下文栈
    cat > "$MEMORY_BASE/short-term/context_stack.json" << EOF
{
  "session_id": "$SESSION_ID",
  "started_at": "$(date -Iseconds)",
  "context_stack": [],
  "active_variables": {},
  "temp_data": {},
  "last_updated": "$(date -Iseconds)"
}
EOF
    
    log "会话 $SESSION_ID 已初始化"
    
    # 显示活跃项目
    echo ""
    echo "📁 活跃项目:"
    ls -1 "$MEMORY_BASE/mid-term/projects/" 2>/dev/null | sed 's/^/   - /' || echo "   (无)"
}

# ==================== 会话结束 ====================

session_end() {
    log "会话结束，保存状态..."
    
    SESSION_ID=$(grep "会话ID" "$MEMORY_BASE/short-term/current_session.md" | cut -d':' -f2 | tr -d ' ' || echo "unknown")
    
    # 使用 Node.js 脚本进行归档
    node "$MEMORY_BASE/scripts/memory_cli.js" save "$SESSION_ID"
    
    log "会话 $SESSION_ID 已归档"
}

# ==================== 主入口 ====================

case "${1:-help}" in
    daily)
        daily_maintenance
        ;;
    weekly)
        weekly_cleanup
        ;;
    session-start)
        session_start
        ;;
    session-end)
        session_end
        ;;
    help|--help|-h)
        echo "记忆系统自动化脚本"
        echo ""
        echo "用法: $0 <命令>"
        echo ""
        echo "命令:"
        echo "  daily          执行每日维护任务"
        echo "  weekly         执行每周整理任务"
        echo "  session-start  会话启动时调用"
        echo "  session-end    会话结束时调用"
        echo "  help           显示此帮助"
        ;;
    *)
        error "未知命令: $1"
        echo "使用 '$0 help' 查看可用命令"
        exit 1
        ;;
esac
