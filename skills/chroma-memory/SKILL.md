# Chroma Memory - 向量记忆系统

## 概述
Chroma Memory 是一个基于 ChromaDB 的向量数据库系统，用于存储和检索语义化的记忆数据。支持向量存储、语义搜索、自动持久化和多集合管理。

## 功能特性
- 🧠 **向量存储**：基于嵌入向量的语义记忆存储
- 🔍 **语义搜索**：支持相似度搜索和语义匹配
- 📦 **自动持久化**：数据自动保存到本地存储
- 🔄 **多集合支持**：可按项目/类型创建不同集合
- 🚀 **API服务**：内置FastAPI REST接口
- 📊 **批量导入**：支持记忆数据的批量向量化迁移
- ⚡ **高性能**：支持并发查询和索引优化

## 安装依赖

```bash
# 基础依赖
pip install chromadb sentence-transformers

# API服务依赖（可选）
pip install fastapi uvicorn

# 完整安装
pip install -r requirements.txt
```

## 快速开始

### 1. 启动ChromaDB服务

```bash
# 使用Python模块启动
python chroma_memory.py

# 或使用Docker（推荐生产环境）
docker run -d -p 8000:8000 chromadb/chroma:latest
```

### 2. Python API 使用

```python
from chroma_memory import ChromaMemory

# 初始化
memory = ChromaMemory(
    persist_dir="./chroma_db",
    collection_name="kimi_claw_memory"
)

# 添加记忆
memory.add(
    text="这是一个重要的项目决策",
    metadata={
        "project": "kimi-claw",
        "type": "decision",
        "priority": "high"
    },
    id="decision_001"
)

# 语义搜索
results = memory.search(
    query="项目决策",
    n_results=5,
    filter={"project": "kimi-claw"}
)

# 批量添加
memory.add_batch([
    {"text": "记忆1", "metadata": {"tag": "a"}},
    {"text": "记忆2", "metadata": {"tag": "b"}}
])
```

### 3. CLI 使用

```bash
# 查看状态
python chroma_memory.py stats

# 添加记忆
python chroma_memory.py add "记忆内容" --metadata '{"key": "value"}'

# 搜索记忆
python chroma_memory.py search "查询内容" --n 5

# 批量导入
python chroma_memory.py import ./memories.json

# 导出数据
python chroma_memory.py export ./backup.json
```

### 4. REST API 使用

```bash
# 启动API服务
python chroma_api.py

# 添加记忆
curl -X POST http://localhost:8000/memory \
  -H "Content-Type: application/json" \
  -d '{
    "text": "项目启动会议记录",
    "metadata": {"type": "meeting", "project": "kimi"}
  }'

# 搜索记忆
curl -X POST http://localhost:8000/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "会议记录",
    "n_results": 5
  }'

# 获取统计
curl http://localhost:8000/stats
```

## 配置选项

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| persist_dir | str | "./chroma_db" | 数据持久化目录 |
| collection_name | str | "kimi_claw_memory" | 集合名称 |
| embedding_model | str | "all-MiniLM-L6-v2" | 嵌入模型 |
| host | str | "0.0.0.0" | API服务主机 |
| port | int | 8000 | API服务端口 |

## 数据迁移

### 从文件迁移到向量数据库

```python
from chroma_memory import migrate_from_files

# 迁移记忆文件
migrate_from_files(
    source_dir="./memory/",
    target_collection="kimi_claw_memory"
)
```

### 批量导入JSON

```bash
python chroma_memory.py import ./memories.json --format json
```

JSON格式示例：
```json
[
  {
    "text": "记忆内容",
    "metadata": {"type": "decision"},
    "id": "mem_001"
  }
]
```

## 部署架构

```
┌─────────────────┐
│   API Gateway   │
└────────┬────────┘
         │
┌────────▼────────┐
│  Chroma Memory  │
│  (FastAPI)      │
└────────┬────────┘
         │
┌────────▼────────┐
│   ChromaDB      │
│  (向量数据库)    │
└────────┬────────┘
         │
┌────────▼────────┐
│  SQLite存储     │
│  (持久化)       │
└─────────────────┘
```

## 性能优化

1. **批量操作**：使用 `add_batch()` 替代多次 `add()`
2. **索引调优**：大集合时调整 `hnsw:space` 参数
3. **连接池**：API服务使用连接池管理
4. **缓存**：对热门查询结果启用缓存

## 监控指标

```bash
# 查看数据库统计
curl http://localhost:8000/stats

# 响应示例
{
  "status": "running",
  "collection": "kimi_claw_memory",
  "count": 1250,
  "persist_dir": "./chroma_db",
  "embedding_model": "all-MiniLM-L6-v2",
  "version": "1.5.1"
}
```

## 故障排除

| 问题 | 解决方案 |
|------|----------|
| 导入错误 | 确保已安装 `chromadb` 和 `sentence-transformers` |
| 权限错误 | 检查数据目录写入权限 `chmod 755 ./chroma_db` |
| 内存不足 | 减少批量操作大小或增加系统内存 |
| 模型下载失败 | 手动下载嵌入模型到本地缓存 |
| API连接失败 | 检查端口占用 `lsof -i :8000` |

## 更新日志

- **2026-02-27**: 初始部署，基础功能实现
  - ✅ ChromaDB 向量数据库部署
  - ✅ Python API 完整实现
  - ✅ CLI 命令行工具
  - ✅ FastAPI REST 接口
  - ✅ 批量数据迁移功能
  - ✅ 语义搜索接口

## 部署状态

- ✅ **已部署**: 2026-02-27 16:14
- 📁 **存储位置**: `/root/.openclaw/workspace/skills/chroma-memory/chroma_db/`
- 🔄 **状态**: 运行中
- 📝 **集合**: `kimi_claw_memory`
- 🔌 **API端口**: 8000
- 📊 **记录数**: 0 (待迁移)

## 文件结构

```
chroma-memory/
├── SKILL.md              # 本文件
├── chroma_memory.py      # 主程序 (Python API + CLI)
├── chroma_api.py         # FastAPI REST服务
├── requirements.txt      # 依赖列表
├── test_chroma.py        # 测试脚本
├── migrate_data.py       # 数据迁移工具
└── chroma_db/            # 数据存储目录
    └── chroma.sqlite3    # SQLite数据库
```

## 许可证

MIT License - 自由使用和修改
