#!/usr/bin/env python3
"""
轻量级Chroma Memory Vector Store (无torch依赖版本)
使用简单的哈希嵌入作为fallback
"""

import os
import json
import hashlib
import re
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, asdict
import numpy as np

# 尝试导入Chroma，如果失败则使用mock
try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️ ChromaDB not available, using mock implementation")

# 尝试导入sentence-transformers
try:
    from sentence_transformers import SentenceTransformer
    EMBEDDING_AVAILABLE = True
except ImportError:
    EMBEDDING_AVAILABLE = False
    print("⚠️ SentenceTransformer not available, using hash-based embeddings")

try:
    from rank_bm25 import BM25Okapi
    BM25_AVAILABLE = True
except ImportError:
    BM25_AVAILABLE = False


@dataclass
class MemoryRecord:
    """记忆记录数据结构"""
    id: str
    content: str
    source: str
    memory_type: str
    timestamp: str
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict:
        return asdict(self)


class SimpleEmbeddingModel:
    """简单的哈希嵌入模型 (fallback)"""
    
    def __init__(self, dim: int = 384):
        self.dim = dim
        np.random.seed(42)  # 固定随机种子
        self.vocab_hash = {}
    
    def encode(self, texts: List[str]) -> np.ndarray:
        """生成简单的词袋嵌入"""
        embeddings = []
        for text in texts:
            # 分词
            words = re.findall(r'\w+', text.lower())
            # 创建词袋向量
            vec = np.zeros(self.dim)
            for word in words:
                if word not in self.vocab_hash:
                    self.vocab_hash[word] = np.random.randn(self.dim)
                vec += self.vocab_hash[word]
            # 归一化
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec = vec / norm
            embeddings.append(vec)
        return np.array(embeddings)


class ChromaMemoryStore:
    """Chroma向量记忆存储主类"""
    
    def __init__(self, persist_dir: str = "./chroma_db"):
        self.persist_dir = persist_dir
        os.makedirs(persist_dir, exist_ok=True)
        
        # 初始化Chroma客户端
        if CHROMA_AVAILABLE:
            self.client = chromadb.PersistentClient(
                path=persist_dir,
                settings=Settings(anonymized_telemetry=False)
            )
            self.collection = self.client.get_or_create_collection(
                name="memory_store",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            # Mock实现
            self.client = None
            self.collection = MockCollection()
        
        # 加载嵌入模型
        if EMBEDDING_AVAILABLE:
            try:
                self.model = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            except:
                self.model = SimpleEmbeddingModel()
        else:
            self.model = SimpleEmbeddingModel()
        
        # BM25索引缓存
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_ids = []
        
    def _generate_id(self, content: str, source: str) -> str:
        """生成唯一ID"""
        hash_input = f"{content}:{source}:{datetime.now().isoformat()}"
        return hashlib.md5(hash_input.encode()).hexdigest()
    
    def _chunk_text(self, text: str, chunk_size: int = 512, overlap: int = 50) -> List[str]:
        """将长文本分块"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunk = text[start:end]
            chunks.append(chunk)
            start = end - overlap
        return chunks
    
    def add_memory(self, content: str, source: str, memory_type: str, 
                   metadata: Dict[str, Any] = None) -> str:
        """添加记忆到向量存储"""
        if metadata is None:
            metadata = {}
        
        memory_id = self._generate_id(content, source)
        chunks = self._chunk_text(content)
        embeddings = self.model.encode(chunks).tolist()
        
        base_metadata = {
            "source": source,
            "memory_type": memory_type,
            "timestamp": datetime.now().isoformat(),
            "chunk_count": len(chunks),
            **metadata
        }
        
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            chunk_id = f"{memory_id}_{i}"
            chunk_metadata = {
                **base_metadata,
                "chunk_index": i,
                "parent_id": memory_id,
                "content": chunk
            }
            
            if CHROMA_AVAILABLE:
                self.collection.add(
                    ids=[chunk_id],
                    embeddings=[embedding],
                    documents=[chunk],
                    metadatas=[chunk_metadata]
                )
            else:
                self.collection.add(chunk_id, embedding, chunk, chunk_metadata)
        
        self._update_bm25_index()
        return memory_id
    
    def _update_bm25_index(self):
        """更新BM25索引"""
        if not BM25_AVAILABLE:
            return
            
        if CHROMA_AVAILABLE:
            results = self.collection.get()
            if results and results['documents']:
                self.bm25_corpus = []
                self.bm25_ids = []
                for doc, doc_id in zip(results['documents'], results['ids']):
                    tokens = re.findall(r'\w+', doc.lower())
                    self.bm25_corpus.append(tokens)
                    self.bm25_ids.append(doc_id)
                if self.bm25_corpus:
                    self.bm25_index = BM25Okapi(self.bm25_corpus)
        else:
            # Mock实现
            docs = self.collection.get_all_documents()
            if docs:
                self.bm25_corpus = []
                self.bm25_ids = []
                for doc_id, doc in docs.items():
                    tokens = re.findall(r'\w+', doc.lower())
                    self.bm25_corpus.append(tokens)
                    self.bm25_ids.append(doc_id)
                if self.bm25_corpus:
                    self.bm25_index = BM25Okapi(self.bm25_corpus)
    
    def search_semantic(self, query: str, n_results: int = 5, 
                        memory_type: Optional[str] = None) -> List[Dict]:
        """语义检索"""
        query_embedding = self.model.encode([query]).tolist()
        
        where_filter = None
        if memory_type:
            where_filter = {"memory_type": memory_type}
        
        if CHROMA_AVAILABLE:
            results = self.collection.query(
                query_embeddings=query_embedding,
                n_results=n_results,
                where=where_filter
            )
            
            formatted_results = []
            if results and results['ids']:
                for i in range(len(results['ids'][0])):
                    formatted_results.append({
                        'id': results['ids'][0][i],
                        'content': results['documents'][0][i] if results['documents'] else '',
                        'distance': results['distances'][0][i] if results['distances'] else 0,
                        'metadata': results['metadatas'][0][i] if results['metadatas'] else {}
                    })
            return formatted_results
        else:
            # Mock实现 - 简单余弦相似度
            return self.collection.query(query_embedding[0], n_results, where_filter)
    
    def search_keyword(self, query: str, n_results: int = 5) -> List[Dict]:
        """关键词检索 (BM25)"""
        if not BM25_AVAILABLE or self.bm25_index is None:
            self._update_bm25_index()
        
        if not BM25_AVAILABLE or self.bm25_index is None or not self.bm25_corpus:
            return []
        
        query_tokens = re.findall(r'\w+', query.lower())
        scores = self.bm25_index.get_scores(query_tokens)
        top_indices = np.argsort(scores)[::-1][:n_results]
        
        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                doc_id = self.bm25_ids[idx]
                if CHROMA_AVAILABLE:
                    doc_data = self.collection.get(ids=[doc_id])
                    if doc_data and doc_data['documents']:
                        results.append({
                            'id': doc_id,
                            'content': doc_data['documents'][0],
                            'score': float(scores[idx]),
                            'metadata': doc_data['metadatas'][0] if doc_data['metadatas'] else {}
                        })
                else:
                    doc = self.collection.get_document(doc_id)
                    if doc:
                        results.append({
                            'id': doc_id,
                            'content': doc,
                            'score': float(scores[idx]),
                            'metadata': {}
                        })
        return results
    
    def search_hybrid(self, query: str, n_results: int = 5, 
                      memory_type: Optional[str] = None,
                      semantic_weight: float = 0.7) -> List[Dict]:
        """混合检索 (语义 + 关键词)"""
        semantic_results = self.search_semantic(query, n_results * 2, memory_type)
        keyword_results = self.search_keyword(query, n_results * 2)
        
        fused_scores = {}
        
        for result in semantic_results:
            doc_id = result['id']
            similarity = 1 - result['distance']
            fused_scores[doc_id] = {
                'score': semantic_weight * similarity,
                'data': result
            }
        
        keyword_weight = 1 - semantic_weight
        for result in keyword_results:
            doc_id = result['id']
            normalized_score = min(result['score'] / 10, 1.0)
            if doc_id in fused_scores:
                fused_scores[doc_id]['score'] += keyword_weight * normalized_score
            else:
                fused_scores[doc_id] = {
                    'score': keyword_weight * normalized_score,
                    'data': result
                }
        
        sorted_results = sorted(
            fused_scores.items(), 
            key=lambda x: x[1]['score'], 
            reverse=True
        )[:n_results]
        
        return [
            {
                'id': item[0],
                'fused_score': item[1]['score'],
                **item[1]['data']
            }
            for item in sorted_results
        ]
    
    def delete_memory(self, memory_id: str):
        """删除记忆"""
        if CHROMA_AVAILABLE:
            results = self.collection.get(where={"parent_id": memory_id})
            if results and results['ids']:
                self.collection.delete(ids=results['ids'])
                self._update_bm25_index()
        else:
            self.collection.delete_by_parent(memory_id)
    
    def get_stats(self) -> Dict:
        """获取存储统计信息"""
        if CHROMA_AVAILABLE:
            count = self.collection.count()
        else:
            count = self.collection.count()
        
        return {
            "total_documents": count,
            "persist_dir": self.persist_dir,
            "chroma_available": CHROMA_AVAILABLE,
            "embedding_available": EMBEDDING_AVAILABLE,
            "bm25_available": BM25_AVAILABLE,
            "embedding_dim": 384
        }
    
    def clear_all(self):
        """清空所有数据"""
        if CHROMA_AVAILABLE:
            self.client.delete_collection("memory_store")
            self.collection = self.client.get_or_create_collection(
                name="memory_store",
                metadata={"hnsw:space": "cosine"}
            )
        else:
            self.collection.clear()
        self.bm25_index = None
        self.bm25_corpus = []
        self.bm25_ids = []


class MockCollection:
    """Mock集合用于无Chroma环境"""
    
    def __init__(self):
        self.data = {}
        self.embeddings = {}
        self.metadatas = {}
    
    def add(self, id, embedding, document, metadata):
        self.data[id] = document
        self.embeddings[id] = embedding
        self.metadatas[id] = metadata
    
    def query(self, query_embedding, n_results, where_filter):
        # 简单余弦相似度
        results = []
        for doc_id, emb in self.embeddings.items():
            metadata = self.metadatas.get(doc_id, {})
            if where_filter and not self._match_filter(metadata, where_filter):
                continue
            
            sim = np.dot(query_embedding, emb) / (np.linalg.norm(query_embedding) * np.linalg.norm(emb))
            results.append({
                'id': doc_id,
                'content': self.data[doc_id],
                'distance': 1 - sim,
                'metadata': metadata
            })
        
        results.sort(key=lambda x: x['distance'])
        return results[:n_results]
    
    def _match_filter(self, metadata, filter_dict):
        for k, v in filter_dict.items():
            if metadata.get(k) != v:
                return False
        return True
    
    def get(self, ids=None, where=None):
        if ids:
            return {
                'ids': ids,
                'documents': [self.data.get(i) for i in ids],
                'metadatas': [self.metadatas.get(i) for i in ids]
            }
        return {'ids': list(self.data.keys()), 'documents': list(self.data.values())}
    
    def get_all_documents(self):
        return self.data
    
    def get_document(self, doc_id):
        return self.data.get(doc_id)
    
    def delete(self, ids):
        for i in ids:
            self.data.pop(i, None)
            self.embeddings.pop(i, None)
            self.metadatas.pop(i, None)
    
    def delete_by_parent(self, parent_id):
        to_delete = [k for k, v in self.metadatas.items() if v.get('parent_id') == parent_id]
        self.delete(to_delete)
    
    def count(self):
        return len(self.data)
    
    def clear(self):
        self.data.clear()
        self.embeddings.clear()
        self.metadatas.clear()


class MemoryMigrator:
    """记忆数据迁移工具"""
    
    def __init__(self, store: ChromaMemoryStore):
        self.store = store
    
    def migrate_from_files(self, memory_dir: str = "./memory") -> Dict:
        """从文件系统迁移记忆数据"""
        stats = {
            "total_files": 0,
            "total_memories": 0,
            "errors": []
        }
        
        memory_types = ["episodic", "semantic", "mid-term", "long-term"]
        
        for mem_type in memory_types:
            type_dir = os.path.join(memory_dir, mem_type)
            if not os.path.exists(type_dir):
                continue
            
            for root, _, files in os.walk(type_dir):
                for file in files:
                    if file.endswith('.md'):
                        file_path = os.path.join(root, file)
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            
                            title = self._extract_title(content)
                            
                            self.store.add_memory(
                                content=content,
                                source=file_path,
                                memory_type=mem_type,
                                metadata={
                                    "filename": file,
                                    "title": title,
                                    "migrated_at": datetime.now().isoformat()
                                }
                            )
                            
                            stats["total_files"] += 1
                            stats["total_memories"] += 1
                            
                        except Exception as e:
                            stats["errors"].append(f"{file_path}: {str(e)}")
        
        return stats
    
    def _extract_title(self, content: str) -> str:
        """从Markdown内容中提取标题"""
        lines = content.split('\n')
        for line in lines[:10]:
            if line.startswith('# '):
                return line[2:].strip()
            elif line.startswith('## '):
                return line[3:].strip()
        return "Untitled"


def get_memory_store(persist_dir: str = "./chroma_db") -> ChromaMemoryStore:
    """获取记忆存储实例"""
    return ChromaMemoryStore(persist_dir)


def migrate_memories(store: ChromaMemoryStore, memory_dir: str = "./memory") -> Dict:
    """迁移记忆数据"""
    migrator = MemoryMigrator(store)
    return migrator.migrate_from_files(memory_dir)
