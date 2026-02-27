#!/usr/bin/env python3
"""
记忆系统v3.0主入口
整合所有组件：Pinecone + 智能摘要 + 混合检索
"""

import os
import sys
from typing import List, Dict, Optional, Any
from datetime import datetime

# 添加当前目录到路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pinecone_store import PineconeMemoryStore, HybridRetriever, EmbeddingProvider
from smart_summarizer import MemoryCompressionEngine, ImportanceScorer


class MemorySystemV3:
    """
    记忆系统 v3.0
    市场领先级记忆系统实现
    """
    
    def __init__(self, 
                 pinecone_api_key: Optional[str] = None,
                 index_name: str = "kimi-claw-memory",
                 use_compression: bool = True):
        """
        初始化记忆系统v3.0
        
        Args:
            pinecone_api_key: Pinecone API密钥
            index_name: 索引名称
            use_compression: 是否启用智能压缩
        """
        self.use_compression = use_compression
        
        # 初始化向量存储
        try:
            self.vector_store = PineconeMemoryStore(
                api_key=pinecone_api_key,
                index_name=index_name
            )
            self.retriever = HybridRetriever(self.vector_store)
            self.vector_db_available = True
        except Exception as e:
            print(f"⚠️ Pinecone初始化失败: {e}")
            print("   将使用本地模拟模式")
            self.vector_store = None
            self.retriever = None
            self.vector_db_available = False
        
        # 初始化压缩引擎
        if use_compression:
            self.compression_engine = MemoryCompressionEngine()
            self.importance_scorer = ImportanceScorer()
        
        # 统计信息
        self.stats = {
            "total_stored": 0,
            "total_searches": 0,
            "total_compressed": 0
        }
    
    def store(self,
              content: str,
              user_id: str = "default",
              session_id: str = "",
              memory_type: str = "general",
              metadata: Dict[str, Any] = None,
              user_marked: bool = False) -> Dict:
        """
        存储记忆
        
        Args:
            content: 记忆内容
            user_id: 用户ID
            session_id: 会话ID
            memory_type: 记忆类型
            metadata: 额外元数据
            user_marked: 用户是否显式标记为重要
        
        Returns:
            存储结果
        """
        metadata = metadata or {}
        
        # 1. 智能压缩
        if self.use_compression:
            compression_result = self.compression_engine.compress(
                content=content,
                user_marked=user_marked
            )
            
            # 使用压缩后的内容存储
            storage_content = compression_result.get("full_text") or \
                            compression_result.get("summary") or \
                            content[:1000]
            
            # 添加压缩信息到元数据
            metadata.update({
                "importance_score": compression_result["importance_score"],
                "compression_level": compression_result["compression_level"],
                "compression_ratio": compression_result["compression_ratio"],
                "original_length": compression_result["original_length"]
            })
            
            self.stats["total_compressed"] += 1
        else:
            storage_content = content
            compression_result = None
        
        # 2. 存储到向量数据库
        if self.vector_db_available and self.vector_store:
            memory_id = self.vector_store.add_memory(
                content=storage_content,
                metadata=metadata,
                user_id=user_id,
                session_id=session_id,
                memory_type=memory_type
            )
        else:
            # 模拟模式
            memory_id = f"mem_{hash(content) % 1000000:06d}"
        
        self.stats["total_stored"] += 1
        
        return {
            "id": memory_id,
            "compression": compression_result,
            "storage_status": "stored" if self.vector_db_available else "mock"
        }
    
    def retrieve(self,
                 query: str,
                 user_id: Optional[str] = None,
                 memory_type: Optional[str] = None,
                 top_k: int = 5,
                 mode: str = "hybrid") -> List[Dict]:
        """
        检索记忆
        
        Args:
            query: 查询文本
            user_id: 用户ID过滤
            memory_type: 记忆类型过滤
            top_k: 返回结果数
            mode: 检索模式 (semantic/keyword/hybrid)
        
        Returns:
            检索结果列表
        """
        self.stats["total_searches"] += 1
        
        if not self.vector_db_available or not self.vector_store:
            # 模拟检索结果
            return [
                {
                    "id": f"mem_{i:06d}",
                    "score": 0.9 - i * 0.05,
                    "content": f"模拟记忆内容 {i}",
                    "metadata": {"memory_type": memory_type or "general"}
                }
                for i in range(top_k)
            ]
        
        # 使用混合检索
        if mode == "hybrid" and self.retriever:
            results = self.retriever.search(
                query=query,
                user_id=user_id,
                top_k=top_k
            )
        else:
            results = self.vector_store.search(
                query=query,
                user_id=user_id,
                memory_type=memory_type,
                top_k=top_k
            )
        
        return results
    
    def get_stats(self) -> Dict:
        """获取系统统计信息"""
        stats = {
            "operations": self.stats.copy(),
            "compression_enabled": self.use_compression,
            "vector_db_available": self.vector_db_available
        }
        
        if self.vector_db_available and self.vector_store:
            stats["vector_db"] = self.vector_store.get_stats()
        
        return stats
    
    def migrate_from_chroma(self, chroma_dir: str = "./chroma_db") -> Dict:
        """从Chroma迁移数据"""
        if not self.vector_db_available:
            return {"error": "Vector DB not available"}
        
        try:
            from chroma_store import ChromaMemoryStore
            
            chroma_store = ChromaMemoryStore(persist_dir=chroma_dir)
            
            # 获取所有数据
            all_data = chroma_store.collection.get()
            
            if not all_data or not all_data.get('documents'):
                return {"migrated": 0}
            
            migrated = 0
            for i, doc in enumerate(all_data['documents']):
                metadata = all_data['metadatas'][i] if all_data.get('metadatas') else {}
                
                self.store(
                    content=doc,
                    user_id=metadata.get('user_id', 'default'),
                    memory_type=metadata.get('memory_type', 'general'),
                    metadata=metadata
                )
                migrated += 1
            
            return {"migrated": migrated}
            
        except Exception as e:
            return {"error": str(e)}


def demo():
    """演示记忆系统v3.0"""
    print("=" * 70)
    print("Kimi Claw 记忆系统 v3.0 - 演示")
    print("=" * 70)
    
    # 初始化系统
    memory = MemorySystemV3(
        index_name="kimi-claw-memory-demo",
        use_compression=True
    )
    
    print("\n📥 存储示例记忆...")
    
    sample_memories = [
        {
            "content": """
            2026-02-27 重要战略决策
            
            董事长兰山在高管会议上正式宣布：启动KCGS记忆系统v3.0升级项目。
            这是一个具有里程碑意义的决策，将投入500万元预算，组建10人精英团队。
            
            项目目标：
            1. 集成Pinecone向量数据库，实现毫秒级检索
            2. 引入Neo4j知识图谱，支持关系推理
            3. 实现智能摘要，压缩率目标60%
            4. 建立跨会话一致性机制
            
            时间线：3个月完成核心功能，6个月全面上线。
            """,
            "memory_type": "episodic",
            "user_marked": True,
            "metadata": {"decision": True, "budget": 5000000}
        },
        {
            "content": """
            技术调研：向量数据库对比
            
            对比了Pinecone、Weaviate、ChromaDB、Milvus四个主流向量数据库：
            
            Pinecone:
            - 优点：托管服务、低延迟(1-2ms)、易扩展
            - 缺点：成本较高、 vendor lock-in
            
            ChromaDB:
            - 优点：开源、本地部署、易集成
            - 缺点：性能有限、扩展性一般
            
            结论：生产环境使用Pinecone，开发测试使用ChromaDB。
            """,
            "memory_type": "semantic",
            "user_marked": False,
            "metadata": {"topic": "vector_database", "research": True}
        },
        {
            "content": """
            日常开发进度 - 2026-02-27
            
            今日完成：
            - 完成Pinecone集成模块开发
            - 实现智能摘要系统
            - 编写性能基准测试
            
            明日计划：
            - 开始Neo4j知识图谱集成
            - 优化混合检索算法
            
            阻塞问题：无
            """,
            "memory_type": "episodic",
            "user_marked": False,
            "metadata": {"daily_log": True}
        }
    ]
    
    for i, mem in enumerate(sample_memories, 1):
        result = memory.store(
            content=mem["content"],
            memory_type=mem["memory_type"],
            user_marked=mem["user_marked"],
            metadata=mem["metadata"]
        )
        
        compression = result.get("compression", {})
        print(f"\n  [{i}] 存储成功: {result['id'][:20]}...")
        print(f"      重要性: {compression.get('importance_score', 0):.2f}/5.0")
        print(f"      压缩级别: {compression.get('compression_level', 0)}")
        print(f"      压缩率: {compression.get('compression_ratio', 0):.1%}")
    
    print("\n📊 系统统计:")
    stats = memory.get_stats()
    print(f"  已存储: {stats['operations']['total_stored']}")
    print(f"  已压缩: {stats['operations']['total_compressed']}")
    print(f"  向量DB: {'可用' if stats['vector_db_available'] else '模拟模式'}")
    
    print("\n🔍 检索测试:")
    
    test_queries = [
        "记忆系统升级",
        "向量数据库对比",
        "董事长兰山决策",
        "开发进度"
    ]
    
    for query in test_queries:
        print(f"\n  查询: '{query}'")
        results = memory.retrieve(query, top_k=3, mode="hybrid")
        for r in results:
            score = r.get('fused_score', r.get('score', 0))
            content = r.get('content', '')[:50]
            print(f"    → 分数: {score:.3f} | {content}...")
    
    print("\n" + "=" * 70)
    print("演示完成!")
    print("=" * 70)


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Kimi Claw Memory System v3.0')
    parser.add_argument('--demo', action='store_true', help='运行演示')
    parser.add_argument('--benchmark', action='store_true', help='运行基准测试')
    parser.add_argument('--migrate', action='store_true', help='从Chroma迁移数据')
    
    args = parser.parse_args()
    
    if args.benchmark:
        from benchmark import main as benchmark_main
        benchmark_main()
    elif args.migrate:
        memory = MemorySystemV3()
        result = memory.migrate_from_chroma()
        print(f"迁移结果: {result}")
    else:
        demo()


if __name__ == "__main__":
    main()
