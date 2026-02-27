#!/usr/bin/env python3
"""
轻量级记忆检索服务接口 (无torch依赖)
"""

import os
import sys
import time
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chroma_store import ChromaMemoryStore, MemoryMigrator


@dataclass
class SearchResult:
    """检索结果"""
    id: str
    content: str
    source: str
    memory_type: str
    score: float
    metadata: Dict
    
    def to_dict(self) -> Dict:
        return asdict(self)


class MemoryRetrievalService:
    """记忆检索服务"""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.store = ChromaMemoryStore(persist_dir)
        self.search_history = []
    
    def search(self, query: str, mode: str = "hybrid", n_results: int = 5,
               memory_type: Optional[str] = None) -> List[SearchResult]:
        """统一检索接口"""
        start_time = time.time()
        
        if mode == "semantic":
            results = self.store.search_semantic(query, n_results, memory_type)
        elif mode == "keyword":
            results = self.store.search_keyword(query, n_results)
        else:
            results = self.store.search_hybrid(query, n_results, memory_type)
        
        elapsed = time.time() - start_time
        
        self.search_history.append({
            "query": query,
            "mode": mode,
            "n_results": n_results,
            "elapsed_ms": round(elapsed * 1000, 2),
            "timestamp": time.time()
        })
        
        search_results = []
        for r in results:
            metadata = r.get('metadata', {})
            search_results.append(SearchResult(
                id=r['id'],
                content=r.get('content', r.get('document', '')),
                source=metadata.get('source', 'unknown'),
                memory_type=metadata.get('memory_type', 'unknown'),
                score=r.get('fused_score', r.get('score', 1 - r.get('distance', 0))),
                metadata=metadata
            ))
        
        return search_results
    
    def get_context_for_prompt(self, query: str, max_tokens: int = 2000,
                               mode: str = "hybrid") -> str:
        """为LLM prompt获取相关上下文"""
        results = self.search(query, mode=mode, n_results=10)
        
        context_parts = []
        total_length = 0
        
        for result in results:
            content = result.content
            estimated_tokens = len(content) / 2
            
            if total_length + estimated_tokens > max_tokens:
                break
            
            context_parts.append(
                f"[{result.memory_type}] {result.source}\n"
                f"相关度: {result.score:.3f}\n"
                f"{content}\n"
                f"{'='*40}"
            )
            total_length += estimated_tokens
        
        return "\n\n".join(context_parts)
    
    def add_memory(self, content: str, source: str, memory_type: str,
                   metadata: Optional[Dict] = None) -> str:
        """添加新记忆"""
        return self.store.add_memory(content, source, memory_type, metadata)
    
    def get_stats(self) -> Dict:
        """获取服务统计"""
        stats = self.store.get_stats()
        stats["search_history_count"] = len(self.search_history)
        if self.search_history:
            avg_time = sum(h["elapsed_ms"] for h in self.search_history) / len(self.search_history)
            stats["avg_search_time_ms"] = round(avg_time, 2)
        return stats
    
    def benchmark(self, test_queries: List[str]) -> Dict:
        """性能基准测试"""
        results = {
            "semantic": {"times": [], "avg_ms": 0},
            "keyword": {"times": [], "avg_ms": 0},
            "hybrid": {"times": [], "avg_ms": 0}
        }
        
        for query in test_queries:
            # 语义检索测试
            start = time.time()
            self.store.search_semantic(query, 5)
            results["semantic"]["times"].append((time.time() - start) * 1000)
            
            # 关键词检索测试
            start = time.time()
            try:
                self.store.search_keyword(query, 5)
            except:
                pass
            results["keyword"]["times"].append((time.time() - start) * 1000)
            
            # 混合检索测试
            start = time.time()
            self.store.search_hybrid(query, 5)
            results["hybrid"]["times"].append((time.time() - start) * 1000)
        
        for mode in results:
            times = results[mode]["times"]
            if times:
                results[mode]["avg_ms"] = round(sum(times) / len(times), 2)
                results[mode]["min_ms"] = round(min(times), 2)
                results[mode]["max_ms"] = round(max(times), 2)
        
        return results


_service_instance = None


def get_service(persist_dir: str = "./chroma_db") -> MemoryRetrievalService:
    """获取服务单例"""
    global _service_instance
    if _service_instance is None:
        _service_instance = MemoryRetrievalService(persist_dir)
    return _service_instance


def search_memories(query: str, mode: str = "hybrid", n_results: int = 5) -> List[Dict]:
    """便捷函数：搜索记忆"""
    service = get_service()
    results = service.search(query, mode, n_results)
    return [r.to_dict() for r in results]


def get_relevant_context(query: str, max_tokens: int = 2000) -> str:
    """便捷函数：获取相关上下文"""
    service = get_service()
    return service.get_context_for_prompt(query, max_tokens)


if __name__ == "__main__":
    print("=" * 50)
    print("Chroma Memory Retrieval Service Test")
    print("=" * 50)
    
    service = get_service()
    
    print("\n📊 Store Stats:")
    stats = service.get_stats()
    for k, v in stats.items():
        print(f"  {k}: {v}")
    
    print("\n🔍 Test Search:")
    test_queries = [
        "记忆系统架构",
        "Context Engineering",
        "Agent团队",
        "项目进度"
    ]
    
    for query in test_queries:
        print(f"\n  Query: '{query}'")
        results = service.search(query, mode="hybrid", n_results=3)
        for r in results:
            print(f"    - [{r.memory_type}] Score: {r.score:.3f} | {r.source}")
    
    print("\n✅ Test completed!")
