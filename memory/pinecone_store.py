#!/usr/bin/env python3
"""
Pinecone向量数据库集成模块
阶段1: 紧急升级 - 向量数据库迁移
"""

import os
import json
import hashlib
from datetime import datetime
from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# 尝试导入Pinecone
try:
    from pinecone import Pinecone, ServerlessSpec
    PINECONE_AVAILABLE = True
except ImportError:
    PINECONE_AVAILABLE = False
    print("⚠️ Pinecone not installed. Run: pip install pinecone-client")

# 尝试导入OpenAI Embeddings
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# 尝试导入sentence-transformers作为fallback
try:
    from sentence_transformers import SentenceTransformer
    ST_AVAILABLE = True
except ImportError:
    ST_AVAILABLE = False


@dataclass
class VectorMemoryRecord:
    """向量记忆记录"""
    id: str
    content: str
    embedding: List[float]
    metadata: Dict[str, Any]
    timestamp: datetime
    user_id: str = "default"
    session_id: str = ""
    importance_score: float = 0.0


class EmbeddingProvider:
    """嵌入向量提供器 - 支持多种后端"""
    
    def __init__(self, provider: str = "auto"):
        self.provider = provider
        self.model = None
        self.dimension = 1536  # 默认OpenAI维度
        
        if provider == "auto":
            # 自动选择最佳可用后端
            if OPENAI_AVAILABLE and os.getenv("OPENAI_API_KEY"):
                self.provider = "openai"
                self.dimension = 1536
            elif ST_AVAILABLE:
                self.provider = "sentence-transformers"
                self.model = SentenceTransformer('all-MiniLM-L6-v2')
                self.dimension = 384
            else:
                self.provider = "hash"
                self.dimension = 384
        
        elif provider == "sentence-transformers" and ST_AVAILABLE:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            self.dimension = 384
    
    def embed(self, text: str) -> List[float]:
        """生成文本嵌入"""
        if self.provider == "openai":
            return self._embed_openai(text)
        elif self.provider == "sentence-transformers" and self.model:
            return self._embed_st(text)
        else:
            return self._embed_hash(text)
    
    def _embed_openai(self, text: str) -> List[float]:
        """使用OpenAI API生成嵌入"""
        try:
            client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
            response = client.embeddings.create(
                model="text-embedding-3-large",
                input=text[:8000]  # 限制长度
            )
            return response.data[0].embedding
        except Exception as e:
            print(f"OpenAI embedding error: {e}")
            return self._embed_hash(text)
    
    def _embed_st(self, text: str) -> List[float]:
        """使用Sentence-Transformers生成嵌入"""
        embedding = self.model.encode(text[:8000])
        return embedding.tolist()
    
    def _embed_hash(self, text: str) -> List[float]:
        """使用哈希生成确定性嵌入 (fallback)"""
        # 基于关键词哈希的简单嵌入
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        vector = np.zeros(self.dimension)
        for word in words[:100]:  # 限制词数
            hash_val = int(hashlib.md5(word.encode()).hexdigest(), 16)
            for i in range(self.dimension):
                if (hash_val >> (i % 32)) & 1:
                    vector[i] += 1
        
        # 归一化
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        
        return vector.tolist()
    
    def embed_batch(self, texts: List[str]) -> List[List[float]]:
        """批量生成嵌入"""
        return [self.embed(text) for text in texts]


class PineconeMemoryStore:
    """
    Pinecone向量数据库存储
    实现语义检索、元数据过滤、用户隔离
    """
    
    def __init__(self, 
                 api_key: Optional[str] = None,
                 index_name: str = "kimi-claw-memory",
                 dimension: int = 1536,
                 metric: str = "cosine",
                 cloud: str = "aws",
                 region: str = "us-east-1"):
        """
        初始化Pinecone存储
        
        Args:
            api_key: Pinecone API密钥 (默认从环境变量PINECONE_API_KEY获取)
            index_name: 索引名称
            dimension: 向量维度 (OpenAI: 1536, MiniLM: 384)
            metric: 相似度度量 (cosine/euclidean/dotproduct)
            cloud: 云服务提供商 (aws/gcp/azure)
            region: 区域
        """
        if not PINECONE_AVAILABLE:
            raise ImportError("Pinecone not installed. Run: pip install pinecone-client")
        
        self.api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self.api_key:
            raise ValueError("Pinecone API key required. Set PINECONE_API_KEY env var.")
        
        self.index_name = index_name
        self.dimension = dimension
        self.metric = metric
        self.cloud = cloud
        self.region = region
        
        # 初始化Pinecone客户端
        self.pc = Pinecone(api_key=self.api_key)
        
        # 初始化嵌入提供器
        self.embedder = EmbeddingProvider()
        self.dimension = self.embedder.dimension
        
        # 获取或创建索引
        self.index = self._get_or_create_index()
    
    def _get_or_create_index(self):
        """获取或创建索引"""
        # 检查索引是否存在
        existing_indexes = self.pc.list_indexes()
        index_names = [idx.name for idx in existing_indexes]
        
        if self.index_name not in index_names:
            print(f"Creating new Pinecone index: {self.index_name}")
            self.pc.create_index(
                name=self.index_name,
                dimension=self.dimension,
                metric=self.metric,
                spec=ServerlessSpec(cloud=self.cloud, region=self.region)
            )
            print(f"✅ Index '{self.index_name}' created successfully")
        else:
            print(f"Using existing index: {self.index_name}")
        
        return self.pc.Index(self.index_name)
    
    def add_memory(self, 
                   content: str,
                   metadata: Dict[str, Any] = None,
                   user_id: str = "default",
                   session_id: str = "",
                   memory_type: str = "general") -> str:
        """
        添加记忆到向量数据库
        
        Args:
            content: 记忆内容
            metadata: 额外元数据
            user_id: 用户ID (用于数据隔离)
            session_id: 会话ID
            memory_type: 记忆类型 (episodic/semantic/mid_term/long_term)
        
        Returns:
            记忆ID
        """
        # 生成唯一ID
        memory_id = f"mem_{hashlib.md5(f'{content}:{user_id}:{datetime.now()}'.encode()).hexdigest()[:16]}"
        
        # 生成嵌入
        embedding = self.embedder.embed(content)
        
        # 构建元数据
        meta = {
            "user_id": user_id,
            "session_id": session_id,
            "memory_type": memory_type,
            "content": content[:1000],  # 限制存储长度
            "timestamp": datetime.now().isoformat(),
            **(metadata or {})
        }
        
        # 上传到Pinecone
        self.index.upsert(
            vectors=[{
                "id": memory_id,
                "values": embedding,
                "metadata": meta
            }]
        )
        
        return memory_id
    
    def add_memories_batch(self, 
                          memories: List[Dict[str, Any]],
                          batch_size: int = 100) -> List[str]:
        """
        批量添加记忆
        
        Args:
            memories: 记忆列表,每项包含content, metadata等
            batch_size: 批量大小
        
        Returns:
            记忆ID列表
        """
        ids = []
        
        for i in range(0, len(memories), batch_size):
            batch = memories[i:i+batch_size]
            
            vectors = []
            for mem in batch:
                memory_id = f"mem_{hashlib.md5(f'{mem['content']}:{mem.get('user_id', 'default')}:{datetime.now()}'.encode()).hexdigest()[:16]}"
                ids.append(memory_id)
                
                embedding = self.embedder.embed(mem['content'])
                
                meta = {
                    "user_id": mem.get("user_id", "default"),
                    "session_id": mem.get("session_id", ""),
                    "memory_type": mem.get("memory_type", "general"),
                    "content": mem['content'][:1000],
                    "timestamp": datetime.now().isoformat(),
                    **mem.get("metadata", {})
                }
                
                vectors.append({
                    "id": memory_id,
                    "values": embedding,
                    "metadata": meta
                })
            
            self.index.upsert(vectors=vectors)
        
        return ids
    
    def search(self, 
               query: str,
               user_id: Optional[str] = None,
               memory_type: Optional[str] = None,
               top_k: int = 5,
               filter_dict: Optional[Dict] = None) -> List[Dict]:
        """
        语义搜索记忆
        
        Args:
            query: 查询文本
            user_id: 用户ID过滤
            memory_type: 记忆类型过滤
            top_k: 返回结果数
            filter_dict: 额外过滤条件
        
        Returns:
            搜索结果列表
        """
        # 生成查询嵌入
        query_embedding = self.embedder.embed(query)
        
        # 构建过滤条件
        filter_conditions = filter_dict or {}
        if user_id:
            filter_conditions["user_id"] = {"$eq": user_id}
        if memory_type:
            filter_conditions["memory_type"] = {"$eq": memory_type}
        
        # 执行查询
        if filter_conditions:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                filter=filter_conditions,
                include_metadata=True
            )
        else:
            results = self.index.query(
                vector=query_embedding,
                top_k=top_k,
                include_metadata=True
            )
        
        # 格式化结果
        formatted_results = []
        for match in results.matches:
            formatted_results.append({
                "id": match.id,
                "score": match.score,
                "content": match.metadata.get("content", ""),
                "metadata": {k: v for k, v in match.metadata.items() if k != "content"}
            })
        
        return formatted_results
    
    def get_memory(self, memory_id: str) -> Optional[Dict]:
        """获取单个记忆"""
        try:
            result = self.index.fetch(ids=[memory_id])
            if result.vectors:
                vector_data = result.vectors[memory_id]
                return {
                    "id": vector_data.id,
                    "metadata": vector_data.metadata,
                    "values": vector_data.values
                }
            return None
        except Exception as e:
            print(f"Error fetching memory: {e}")
            return None
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        try:
            self.index.delete(ids=[memory_id])
            return True
        except Exception as e:
            print(f"Error deleting memory: {e}")
            return False
    
    def delete_user_memories(self, user_id: str) -> bool:
        """删除用户的所有记忆"""
        try:
            self.index.delete(filter={"user_id": {"$eq": user_id}})
            return True
        except Exception as e:
            print(f"Error deleting user memories: {e}")
            return False
    
    def get_stats(self) -> Dict:
        """获取索引统计信息"""
        stats = self.index.describe_index_stats()
        return {
            "total_vectors": stats.total_vector_count,
            "dimension": stats.dimension,
            "index_fullness": getattr(stats, 'index_fullness', 0),
            "namespaces": list(stats.namespaces.keys()) if stats.namespaces else []
        }
    
    def migrate_from_files(self, 
                          memory_dir: str = "./memory",
                          user_id: str = "default") -> Dict:
        """
        从Markdown文件迁移记忆到Pinecone
        
        Args:
            memory_dir: 记忆文件目录
            user_id: 用户ID
        
        Returns:
            迁移统计
        """
        import glob
        import re
        
        memories = []
        
        # 遍历所有Markdown文件
        for filepath in glob.glob(os.path.join(memory_dir, "**/*.md"), recursive=True):
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # 提取文件名作为类型提示
                filename = os.path.basename(filepath)
                
                # 确定记忆类型
                memory_type = "general"
                if "episodic" in filepath:
                    memory_type = "episodic"
                elif "semantic" in filepath:
                    memory_type = "semantic"
                elif "mid-term" in filepath:
                    memory_type = "mid_term"
                elif "long-term" in filepath:
                    memory_type = "long_term"
                
                # 分割长文档
                sections = re.split(r'\n#+ ', content)
                for i, section in enumerate(sections):
                    if len(section.strip()) < 50:
                        continue
                    
                    memories.append({
                        "content": section[:2000],  # 限制长度
                        "metadata": {
                            "source_file": filepath,
                            "section_index": i
                        },
                        "user_id": user_id,
                        "memory_type": memory_type
                    })
                    
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
        
        # 批量上传
        if memories:
            ids = self.add_memories_batch(memories)
            return {
                "migrated_count": len(ids),
                "file_count": len(glob.glob(os.path.join(memory_dir, "**/*.md"), recursive=True)),
                "memory_ids": ids[:5]  # 只显示前5个
            }
        
        return {"migrated_count": 0, "file_count": 0}


class HybridRetriever:
    """
    混合检索器
    结合Pinecone向量检索 + 本地关键词检索
    """
    
    def __init__(self, pinecone_store: PineconeMemoryStore):
        self.vector_store = pinecone_store
        self.keyword_weight = 0.3
        self.vector_weight = 0.7
    
    def search(self, 
               query: str,
               user_id: Optional[str] = None,
               top_k: int = 5) -> List[Dict]:
        """
        混合检索
        
        流程:
        1. 向量检索获取候选集
        2. 计算关键词匹配分数
        3. 融合重排序
        """
        # 1. 向量检索 (扩大候选集)
        vector_results = self.vector_store.search(
            query=query,
            user_id=user_id,
            top_k=top_k * 2
        )
        
        # 2. 计算关键词匹配分数
        query_keywords = set(self._extract_keywords(query))
        
        scored_results = []
        for result in vector_results:
            content = result.get("content", "")
            content_keywords = set(self._extract_keywords(content))
            
            # 关键词匹配分数
            keyword_score = len(query_keywords & content_keywords) / max(len(query_keywords), 1)
            
            # 向量相似度分数 (已归一化)
            vector_score = result.get("score", 0)
            
            # 融合分数
            fused_score = (
                vector_score * self.vector_weight +
                keyword_score * self.keyword_weight
            )
            
            scored_results.append({
                **result,
                "keyword_score": keyword_score,
                "vector_score": vector_score,
                "fused_score": fused_score
            })
        
        # 3. 按融合分数排序
        scored_results.sort(key=lambda x: x["fused_score"], reverse=True)
        
        return scored_results[:top_k]
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        import re
        words = re.findall(r'\b\w+\b', text.lower())
        
        # 停用词过滤
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 
                     '的', '了', '在', '是', '和', '有', '我', '你', '他'}
        
        return [w for w in words if w not in stopwords and len(w) > 1]


def demo():
    """演示Pinecone记忆存储"""
    print("=" * 60)
    print("Pinecone向量数据库演示")
    print("=" * 60)
    
    # 检查API密钥
    if not os.getenv("PINECONE_API_KEY"):
        print("\n⚠️ 未设置PINECONE_API_KEY环境变量")
        print("请设置: export PINECONE_API_KEY='your-api-key'")
        print("\n演示模式: 使用本地模拟...")
        return demo_mock()
    
    # 创建存储实例
    store = PineconeMemoryStore(
        index_name="kimi-claw-memory-demo",
        dimension=384  # 使用hash-based嵌入
    )
    
    print("\n📥 添加示例记忆...")
    
    sample_memories = [
        {
            "content": "董事长兰山批准了记忆系统升级项目，这是一个重要的战略决策。",
            "memory_type": "episodic",
            "metadata": {"importance": "high", "decision": True}
        },
        {
            "content": "Pinecone是一个托管的向量数据库服务，提供低延迟的语义检索。",
            "memory_type": "semantic",
            "metadata": {"topic": "vector_database"}
        },
        {
            "content": "今天完成了ChromaDB的部署，性能测试显示检索延迟约1.6ms。",
            "memory_type": "episodic",
            "metadata": {"project": "memory_system", "milestone": True}
        },
        {
            "content": "向量数据库的选择标准：延迟、可扩展性、成本、易用性。",
            "memory_type": "semantic",
            "metadata": {"topic": "evaluation_criteria"}
        },
        {
            "content": "团队周会讨论了下周的开发计划，重点是Pinecone集成。",
            "memory_type": "episodic",
            "metadata": {"meeting": True}
        }
    ]
    
    ids = []
    for mem in sample_memories:
        mid = store.add_memory(
            content=mem["content"],
            memory_type=mem["memory_type"],
            metadata=mem["metadata"]
        )
        ids.append(mid)
        print(f"  ✅ {mid[:20]}... | {mem['memory_type']}")
    
    print(f"\n📊 索引统计:")
    stats = store.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    print("\n🔍 语义检索测试:")
    
    test_queries = [
        "记忆系统升级",
        "向量数据库选择",
        "团队会议",
        "Pinecone性能"
    ]
    
    for query in test_queries:
        print(f"\n  查询: '{query}'")
        results = store.search(query, top_k=3)
        for r in results:
            print(f"    → 分数: {r['score']:.3f} | 类型: {r['metadata'].get('memory_type', 'unknown')}")
            print(f"      内容: {r['content'][:60]}...")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)
    
    return store


def demo_mock():
    """模拟演示 (无Pinecone时)"""
    print("\n[模拟模式] 展示预期功能...")
    
    # 模拟嵌入提供器
    embedder = EmbeddingProvider()
    
    sample_texts = [
        "董事长兰山批准了记忆系统升级项目",
        "Pinecone是一个托管的向量数据库",
        "ChromaDB部署完成"
    ]
    
    print("\n📐 嵌入向量示例:")
    for text in sample_texts:
        embedding = embedder.embed(text)
        print(f"  '{text[:30]}...' -> 维度: {len(embedding)}, 前5值: {embedding[:5]}")
    
    print("\n✅ 模拟演示完成")
    print("\n要运行完整演示，请:")
    print("1. 注册Pinecone: https://pinecone.io")
    print("2. 获取API密钥")
    print("3. 设置环境变量: export PINECONE_API_KEY='your-key'")
    print("4. 重新运行此脚本")


if __name__ == "__main__":
    demo()
