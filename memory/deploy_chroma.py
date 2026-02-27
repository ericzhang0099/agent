#!/usr/bin/env python3
"""
Chroma Memory System 部署和测试脚本 (轻量级版本)
"""

import os
import sys
import time
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chroma_store import ChromaMemoryStore, MemoryMigrator
from retrieval_service import MemoryRetrievalService


def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)


def print_section(text):
    print(f"\n📌 {text}")
    print("-" * 40)


def deploy_chroma():
    """部署Chroma向量数据库"""
    print_header("Chroma Memory System Deployment")
    
    # 1. 初始化存储
    print_section("Step 1: Initializing Chroma Store")
    persist_dir = os.path.join(os.path.dirname(__file__), "chroma_db")
    store = ChromaMemoryStore(persist_dir)
    print(f"✅ Store initialized at: {persist_dir}")
    
    # 显示依赖状态
    stats = store.get_stats()
    print(f"  • ChromaDB: {'✅ Available' if stats['chroma_available'] else '⚠️ Mock Mode'}")
    print(f"  • Embedding: {'✅ Available' if stats['embedding_available'] else '⚠️ Hash-based'}")
    print(f"  • BM25: {'✅ Available' if stats['bm25_available'] else '⚠️ Not Available'}")
    
    # 2. 迁移现有数据
    print_section("Step 2: Migrating Existing Memories")
    memory_dir = os.path.dirname(__file__)
    migrator = MemoryMigrator(store)
    stats = migrator.migrate_from_files(memory_dir)
    
    print(f"  📁 Files processed: {stats['total_files']}")
    print(f"  📝 Memories migrated: {stats['total_memories']}")
    if stats['errors']:
        print(f"  ⚠️  Errors: {len(stats['errors'])}")
        for err in stats['errors'][:3]:
            print(f"     - {err}")
    
    # 3. 显示统计
    print_section("Step 3: Store Statistics")
    store_stats = store.get_stats()
    for k, v in store_stats.items():
        print(f"  • {k}: {v}")
    
    return store


def test_retrieval(store):
    """测试检索功能"""
    print_header("Retrieval Functionality Test")
    
    service = MemoryRetrievalService(store.persist_dir)
    
    # 测试查询
    test_queries = [
        "记忆系统架构",
        "Context Engineering 方法",
        "Agent 团队分工",
        "项目进度跟踪",
        "向量检索",
        "短期记忆",
        "长期记忆存储"
    ]
    
    print_section("Semantic Search Test")
    for query in test_queries[:3]:
        start = time.time()
        results = service.search(query, mode="semantic", n_results=3)
        elapsed = (time.time() - start) * 1000
        print(f"\n  Query: '{query}' ({elapsed:.1f}ms)")
        for r in results[:2]:
            print(f"    → [{r.memory_type}] score={r.score:.3f} | {r.source}")
    
    print_section("Keyword Search Test")
    for query in test_queries[3:5]:
        start = time.time()
        results = service.search(query, mode="keyword", n_results=3)
        elapsed = (time.time() - start) * 1000
        print(f"\n  Query: '{query}' ({elapsed:.1f}ms)")
        for r in results[:2]:
            print(f"    → [{r.memory_type}] score={r.score:.3f} | {r.source}")
    
    print_section("Hybrid Search Test")
    for query in test_queries[5:]:
        start = time.time()
        results = service.search(query, mode="hybrid", n_results=3)
        elapsed = (time.time() - start) * 1000
        print(f"\n  Query: '{query}' ({elapsed:.1f}ms)")
        for r in results[:2]:
            print(f"    → [{r.memory_type}] score={r.score:.3f} | {r.source}")
    
    return service


def benchmark_performance(service):
    """性能基准测试"""
    print_header("Performance Benchmark")
    
    test_queries = [
        "记忆系统",
        "Context Engineering",
        "Agent 团队",
        "项目进度",
        "向量数据库",
        "检索优化",
        "记忆分层",
        "语义搜索"
    ]
    
    print_section("Running Benchmarks")
    results = service.benchmark(test_queries)
    
    print("\n  📊 Results:")
    print(f"  {'Mode':<15} {'Avg (ms)':<12} {'Min (ms)':<12} {'Max (ms)':<12}")
    print("  " + "-" * 51)
    for mode, data in results.items():
        print(f"  {mode:<15} {data['avg_ms']:<12.2f} {data.get('min_ms', 0):<12.2f} {data.get('max_ms', 0):<12.2f}")
    
    return results


def compare_with_file_search():
    """与文件检索对比"""
    print_header("Comparison: Vector Search vs File Search")
    
    print_section("File Search (Baseline)")
    print("  • Method: os.walk + string matching")
    print("  • Time complexity: O(N) where N = total files")
    print("  • No semantic understanding")
    print("  • Exact match only")
    
    print_section("Vector Search (Chroma)")
    print("  • Method: HNSW approximate nearest neighbor")
    print("  • Time complexity: O(log N)")
    print("  • Semantic understanding via embeddings")
    print("  • Fuzzy match supported")
    
    print_section("Hybrid Search (Recommended)")
    print("  • Combines semantic + keyword (BM25)")
    print("  • Best of both worlds")
    print("  • Configurable weights")
    
    # 对比数据
    comparison = {
        "metric": ["检索速度", "语义理解", "精确匹配", "容错性", "可扩展性"],
        "file_search": ["慢 (O(N))", "❌ 无", "✅ 高", "❌ 低", "❌ 差"],
        "vector_search": ["快 (O(log N))", "✅ 强", "⚠️ 中", "✅ 高", "✅ 好"],
        "hybrid_search": ["快 (O(log N))", "✅ 强", "✅ 高", "✅ 高", "✅ 好"]
    }
    
    print("\n  📋 Feature Comparison:")
    print(f"  {'Metric':<15} {'File Search':<15} {'Vector Search':<15} {'Hybrid':<15}")
    print("  " + "-" * 60)
    for i in range(len(comparison["metric"])):
        print(f"  {comparison['metric'][i]:<15} {comparison['file_search'][i]:<15} "
              f"{comparison['vector_search'][i]:<15} {comparison['hybrid_search'][i]:<15}")


def print_usage_guide():
    """打印使用说明"""
    print_header("Usage Guide")
    
    print_section("Quick Start")
    print("""
  1. Import the service:
     from memory.retrieval_service import get_service, search_memories
  
  2. Search memories:
     results = search_memories("your query", mode="hybrid", n_results=5)
  
  3. Get context for LLM:
     from memory.retrieval_service import get_relevant_context
     context = get_relevant_context("query", max_tokens=2000)
    """)
    
    print_section("Search Modes")
    print("""
  • semantic: 语义检索，理解查询意图
  • keyword:  关键词检索，精确匹配
  • hybrid:   混合检索（推荐），结合两者优势
    """)
    
    print_section("API Reference")
    print("""
  MemoryRetrievalService.search(query, mode="hybrid", n_results=5)
    - query: 搜索文本
    - mode: 检索模式 (semantic/keyword/hybrid)
    - n_results: 返回结果数量
    
  MemoryRetrievalService.add_memory(content, source, memory_type, metadata)
    - content: 记忆内容
    - source: 来源文件
    - memory_type: episodic/semantic/mid_term/long_term
    - metadata: 额外元数据
    """)


def main():
    """主函数"""
    print("\n🚀 Chroma Memory System Deployment & Test\n")
    
    try:
        # 部署
        store = deploy_chroma()
        
        # 测试
        service = test_retrieval(store)
        
        # 基准测试
        benchmark_results = benchmark_performance(service)
        
        # 对比
        compare_with_file_search()
        
        # 使用说明
        print_usage_guide()
        
        # 保存结果
        print_header("Saving Results")
        result_file = os.path.join(os.path.dirname(__file__), "chroma_deployment_result.json")
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump({
                "store_stats": store.get_stats(),
                "benchmark": benchmark_results,
                "timestamp": time.time()
            }, f, indent=2, ensure_ascii=False)
        print(f"✅ Results saved to: {result_file}")
        
        print_header("Deployment Complete!")
        print("\n  🎉 Chroma Memory System is ready to use!")
        print(f"  📁 Data directory: {store.persist_dir}")
        print("  📚 Import: from memory.retrieval_service import get_service")
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
