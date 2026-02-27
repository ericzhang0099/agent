from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Dict, Optional, Any
import uvicorn

from chroma_memory import ChromaMemory

app = FastAPI(
    title="Chroma Memory API",
    description="向量记忆系统 REST API",
    version="1.0.0"
)

# 初始化记忆系统
memory = ChromaMemory(persist_dir="./chroma_db", collection_name="kimi_claw_memory")

class MemoryItem(BaseModel):
    text: str
    metadata: Optional[Dict[str, Any]] = {}
    id: Optional[str] = None

class MemoryBatch(BaseModel):
    items: List[MemoryItem]

class SearchQuery(BaseModel):
    query: str
    n_results: int = 5
    filter: Optional[Dict[str, Any]] = None

class SearchResult(BaseModel):
    id: str
    text: str
    metadata: Dict[str, Any]
    distance: Optional[float] = None

@app.get("/")
def root():
    """API根路径"""
    return {
        "service": "Chroma Memory API",
        "version": "1.0.0",
        "status": memory.status
    }

@app.get("/stats")
def get_stats():
    """获取数据库统计信息"""
    stats = memory.get_stats()
    return {
        **stats,
        "embedding_model": "all-MiniLM-L6-v2",
        "version": "1.5.1"
    }

@app.post("/memory")
def add_memory(item: MemoryItem):
    """添加单条记忆"""
    result = memory.add(
        text=item.text,
        metadata=item.metadata,
        id=item.id
    )
    if not result.get('success'):
        raise HTTPException(status_code=500, detail=result.get('error'))
    return result

@app.post("/memory/batch")
def add_batch(batch: MemoryBatch):
    """批量添加记忆"""
    results = []
    for item in batch.items:
        result = memory.add(
            text=item.text,
            metadata=item.metadata,
            id=item.id
        )
        results.append(result)
    return {"success": True, "count": len(results), "results": results}

@app.post("/search", response_model=List[SearchResult])
def search_memory(query: SearchQuery):
    """语义搜索记忆"""
    results = memory.search(
        query=query.query,
        n_results=query.n_results,
        filter=query.filter
    )
    return results

@app.get("/memory/{memory_id}")
def get_memory(memory_id: str):
    """获取指定ID的记忆（通过搜索模拟）"""
    # ChromaDB get_by_id 需要特殊处理，这里用搜索模拟
    results = memory.search(
        query="",  # 空查询配合filter
        n_results=1,
        filter={"id": memory_id}
    )
    if not results:
        raise HTTPException(status_code=404, detail="Memory not found")
    return results[0]

@app.delete("/memory/{memory_id}")
def delete_memory(memory_id: str):
    """删除指定ID的记忆"""
    result = memory.delete(memory_id)
    if not result.get('success'):
        raise HTTPException(status_code=500, detail=result.get('error'))
    return result

@app.get("/health")
def health_check():
    """健康检查"""
    return {
        "status": "healthy" if memory.status == "running" else "degraded",
        "chroma_status": memory.status
    }

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
