#!/usr/bin/env python3
"""
记忆压缩系统与现有Chroma存储的集成适配器
实现无缝迁移和增强功能
"""

import os
import sys
from typing import List, Dict, Optional, Any
from datetime import datetime

# 导入现有组件
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from chroma_store import ChromaMemoryStore, MemoryRecord as ChromaMemoryRecord
from retrieval_service import MemoryRetrievalService, SearchResult
from memory_compression_system import (
    MemoryCompressionSystem, 
    MemoryRecord as CompressedMemoryRecord,
    MemoryImportanceScorer,
    MemoryCompressor,
    HybridRetriever,
    TextProcessor
)


class EnhancedMemoryStore:
    """
    增强版记忆存储
    整合Chroma向量存储 + 智能压缩系统
    """
    
    def __init__(self, persist_dir: str = "./chroma_db", 
                 compression_enabled: bool = True):
        self.persist_dir = persist_dir
        self.compression_enabled = compression_enabled
        
        # 基础Chroma存储
        self.chroma_store = ChromaMemoryStore(persist_dir)
        
        # 压缩系统
        self.compression_system = MemoryCompressionSystem(
            storage_dir=os.path.join(persist_dir, "compressed"),
            config={
                "hot_storage_days": 7,
                "warm_storage_days": 30,
                "compression_threshold_high": 4.0,
                "compression_threshold_medium": 2.0,
                "compression_threshold_low": 1.0,
            }
        )
        
        # 尝试加载已有数据
        self.compression_system.load()
        
        # 统计信息
        self.stats = {
            "total_added": 0,
            "total_compressed": 0,
            "total_searches": 0,
            "avg_compression_ratio": 1.0
        }
    
    def add_memory(self, content: str, source: str, memory_type: str,
                   metadata: Dict[str, Any] = None) -> str:
        """
        添加记忆（增强版）
        
        流程:
        1. 添加到Chroma向量存储（用于语义检索）
        2. 添加到压缩系统（用于重要性评分和压缩）
        """
        metadata = metadata or {}
        
        # 1. 添加到Chroma存储
        chroma_id = self.chroma_store.add_memory(content, source, memory_type, metadata)
        
        # 2. 添加到压缩系统
        if self.compression_enabled:
            compressed_memory = self.compression_system.add_memory(
                content=content,
                source=source,
                memory_type=memory_type,
                metadata=metadata
            )
            
            # 将压缩信息同步到metadata
            metadata['compressed_id'] = compressed_memory.id
            metadata['importance_score'] = compressed_memory.importance_score
            metadata['compression_level'] = compressed_memory.compression_level
        
        self.stats["total_added"] += 1
        
        return chroma_id
    
    def search(self, query: str, mode: str = "enhanced", n_results: int = 5,
               memory_type: Optional[str] = None) -> List[Dict]:
        """
        增强检索
        
        模式:
        - "semantic": 纯语义检索（Chroma）
        - "keyword": 纯关键词检索（BM25）
        - "hybrid": 混合检索（Chroma原生）
        - "enhanced": 增强混合（语义+关键词+重要性加权）
        """
        self.stats["total_searches"] += 1
        
        if mode == "enhanced" and self.compression_enabled:
            # 使用增强检索（整合压缩系统的重要性评分）
            return self._enhanced_search(query, n_results, memory_type)
        else:
            # 使用原生Chroma检索
            if mode == "semantic":
                return self.chroma_store.search_semantic(query, n_results, memory_type)
            elif mode == "keyword":
                return self.chroma_store.search_keyword(query, n_results)
            else:
                return self.chroma_store.search_hybrid(query, n_results, memory_type)
    
    def _enhanced_search(self, query: str, n_results: int,
                         memory_type: Optional[str] = None) -> List[Dict]:
        """增强检索实现"""
        # 1. 获取Chroma的语义检索结果
        semantic_results = self.chroma_store.search_semantic(
            query, n_results * 2, memory_type
        )
        
        # 2. 获取关键词检索结果
        keyword_results = self.chroma_store.search_keyword(query, n_results * 2)
        
        # 3. 合并并加权
        fused_scores = {}
        
        # 语义结果加权
        for result in semantic_results:
            doc_id = result['id']
            similarity = 1 - result.get('distance', 0)
            fused_scores[doc_id] = {
                'score': similarity * 0.6,  # 语义权重60%
                'data': result
            }
        
        # 关键词结果加权
        for result in keyword_results:
            doc_id = result['id']
            keyword_score = min(result.get('score', 0) / 10, 1.0)
            if doc_id in fused_scores:
                fused_scores[doc_id]['score'] += keyword_score * 0.3  # 关键词权重30%
            else:
                fused_scores[doc_id] = {
                    'score': keyword_score * 0.3,
                    'data': result
                }
        
        # 4. 重要性加权（从压缩系统获取）
        for doc_id in fused_scores:
            # 尝试获取重要性评分
            importance = self._get_importance_for_doc(doc_id)
            if importance > 0:
                # 高重要性记忆获得额外加成（最高10%）
                fused_scores[doc_id]['score'] *= (1 + importance * 0.02)
        
        # 5. 排序并返回
        sorted_results = sorted(
            fused_scores.items(),
            key=lambda x: x[1]['score'],
            reverse=True
        )[:n_results]
        
        return [
            {
                'id': item[0],
                'enhanced_score': item[1]['score'],
                **item[1]['data']
            }
            for item in sorted_results
        ]
    
    def _get_importance_for_doc(self, doc_id: str) -> float:
        """获取文档的重要性评分"""
        # 从metadata中提取
        try:
            if hasattr(self.chroma_store, 'collection'):
                result = self.chroma_store.collection.get(ids=[doc_id])
                if result and result.get('metadatas'):
                    metadata = result['metadatas'][0]
                    return metadata.get('importance_score', 0)
        except:
            pass
        return 0
    
    def compress_all(self) -> Dict:
        """执行全量压缩"""
        if not self.compression_enabled:
            return {"error": "Compression is disabled"}
        
        # 去重
        dedup_result = self.compression_system.deduplicate()
        
        # 压缩
        compress_result = self.compression_system.compress_all()
        
        # 保存
        self.compression_system.save()
        
        return {
            "deduplication": dedup_result,
            "compression": compress_result
        }
    
    def get_memory_stats(self) -> Dict:
        """获取记忆统计信息"""
        chroma_stats = self.chroma_store.get_stats()
        compression_stats = self.compression_system.get_stats()
        
        return {
            "chroma": chroma_stats,
            "compression": compression_stats,
            "operations": self.stats
        }
    
    def migrate_from_chroma(self) -> Dict:
        """从现有Chroma存储迁移数据到压缩系统"""
        if not hasattr(self.chroma_store, 'collection'):
            return {"error": "Chroma collection not available"}
        
        try:
            # 获取所有文档
            all_data = self.chroma_store.collection.get()
            
            if not all_data or not all_data.get('documents'):
                return {"migrated": 0}
            
            migrated = 0
            for i, doc in enumerate(all_data['documents']):
                doc_id = all_data['ids'][i]
                metadata = all_data['metadatas'][i] if all_data.get('metadatas') else {}
                
                # 添加到压缩系统
                self.compression_system.add_memory(
                    content=doc,
                    source=metadata.get('source', 'unknown'),
                    memory_type=metadata.get('memory_type', 'general'),
                    metadata=metadata
                )
                migrated += 1
            
            # 保存
            self.compression_system.save()
            
            return {"migrated": migrated}
            
        except Exception as e:
            return {"error": str(e)}


class MemoryOptimizer:
    """
    记忆优化器
    定期执行压缩、去重、索引优化
    """
    
    def __init__(self, store: EnhancedMemoryStore):
        self.store = store
        self.optimization_log = []
    
    def run_optimization(self) -> Dict:
        """执行完整优化流程"""
        results = {
            "timestamp": datetime.now().isoformat(),
            "steps": []
        }
        
        # Step 1: 去重
        print("🗑️  Step 1: 去重检测...")
        dedup_result = self.store.compression_system.deduplicate()
        results["steps"].append({"name": "deduplication", "result": dedup_result})
        print(f"   发现重复: {dedup_result['duplicates_found']}")
        
        # Step 2: 重新计算重要性
        print("📊 Step 2: 重新计算重要性评分...")
        for memory in self.store.compression_system.memories.values():
            scorer = MemoryImportanceScorer()
            importance, factors = scorer.calculate_importance(memory)
            memory.importance_score = importance
            memory.importance_factors = factors
        print(f"   已更新 {len(self.store.compression_system.memories)} 条记忆的重要性评分")
        
        # Step 3: 压缩
        print("💾 Step 3: 执行压缩...")
        compress_result = self.store.compression_system.compress_all()
        results["steps"].append({"name": "compression", "result": compress_result})
        print(f"   压缩率: {compress_result['compression_ratio']:.2%}")
        
        # Step 4: 保存
        print("💾 Step 4: 保存优化结果...")
        self.store.compression_system.save()
        
        # Step 5: 重建索引
        print("🔍 Step 5: 重建检索索引...")
        memories = list(self.store.compression_system.memories.values())
        self.store.compression_system.retriever.build_index(memories)
        print(f"   已索引 {len(memories)} 条记忆")
        
        self.optimization_log.append(results)
        return results
    
    def get_optimization_report(self) -> str:
        """生成优化报告"""
        if not self.optimization_log:
            return "No optimization history"
        
        latest = self.optimization_log[-1]
        stats = self.store.get_memory_stats()
        
        report = f"""
# 记忆优化报告
生成时间: {latest['timestamp']}

## 存储统计
- 总记忆数: {stats['compression'].get('total_memories', 0)}
- 原始大小: {stats['compression'].get('total_original_bytes', 0)} bytes
- 压缩后大小: {stats['compression'].get('total_compressed_bytes', 0)} bytes
- 压缩率: {stats['compression'].get('overall_compression_ratio', 0):.2%}
- 平均重要性: {stats['compression'].get('avg_importance_score', 0):.2f}

## 优化步骤
"""
        for step in latest['steps']:
            report += f"\n### {step['name']}\n"
            for key, value in step['result'].items():
                report += f"- {key}: {value}\n"
        
        return report


def demo_integration():
    """演示集成系统"""
    print("=" * 60)
    print("增强记忆存储系统 - 集成演示")
    print("=" * 60)
    
    # 创建增强存储
    store = EnhancedMemoryStore(
        persist_dir="./chroma_db_enhanced",
        compression_enabled=True
    )
    
    print("\n📥 添加示例记忆...")
    
    # 添加一些示例记忆
    memories = [
        {
            "content": "今天完成了记忆系统的设计文档，这是一个重要的里程碑。董事长兰山批准了项目计划。",
            "source": "memory/design_doc.md",
            "type": "milestone",
            "metadata": {"user_marked_important": True}
        },
        {
            "content": "日常开发进度更新：完成了ChromaDB的集成测试，修复了3个bug。",
            "source": "dev/daily_update.md",
            "type": "daily",
            "metadata": {}
        },
        {
            "content": "技术调研：对比了Pinecone、Weaviate和ChromaDB三个向量数据库，最终选择ChromaDB因为部署简单。",
            "source": "research/vector_db_comparison.md",
            "type": "research",
            "metadata": {}
        },
        {
            "content": "记忆压缩算法设计：采用分层压缩策略，根据重要性评分决定压缩级别。",
            "source": "design/compression_algorithm.md",
            "type": "design",
            "metadata": {"user_marked_important": True}
        },
        {
            "content": "团队周会记录：讨论了下周的开发计划，分配了任务。",
            "source": "meetings/weekly.md",
            "type": "meeting",
            "metadata": {}
        }
    ]
    
    for mem in memories:
        store.add_memory(
            content=mem["content"],
            source=mem["source"],
            memory_type=mem["type"],
            metadata=mem["metadata"]
        )
    
    print(f"  已添加 {len(memories)} 条记忆")
    
    print("\n📊 当前统计:")
    stats = store.get_memory_stats()
    print(f"  Chroma文档数: {stats['chroma'].get('total_documents', 0)}")
    print(f"  压缩系统记忆数: {stats['compression'].get('total_memories', 0)}")
    print(f"  平均重要性: {stats['compression'].get('avg_importance_score', 0):.2f}")
    
    print("\n🔍 测试检索:")
    queries = ["记忆压缩", "ChromaDB", "重要里程碑"]
    for query in queries:
        print(f"\n  查询: '{query}'")
        results = store.search(query, mode="enhanced", n_results=2)
        for r in results:
            print(f"    → 相关度: {r.get('enhanced_score', 0):.3f} | "
                  f"来源: {r.get('metadata', {}).get('source', 'unknown')}")
    
    print("\n⚡ 执行优化...")
    optimizer = MemoryOptimizer(store)
    opt_result = optimizer.run_optimization()
    
    print("\n📈 优化后统计:")
    final_stats = store.get_memory_stats()
    print(f"  压缩率: {final_stats['compression'].get('overall_compression_ratio', 0):.2%}")
    print(f"  压缩级别分布: {final_stats['compression'].get('compression_level_distribution', {})}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return store, optimizer


if __name__ == "__main__":
    demo_integration()
