#!/usr/bin/env python3
"""
Chroma向量数据库 - 生产部署版
支持持久化存储和语义搜索
"""

import os
import json
from datetime import datetime

try:
    import chromadb
    from chromadb.config import Settings
    CHROMA_AVAILABLE = True
except ImportError:
    CHROMA_AVAILABLE = False
    print("⚠️ 警告: chromadb 未安装，运行: pip install chromadb")

class ChromaMemory:
    """Chroma向量记忆系统 - 生产级实现"""
    
    def __init__(self, persist_dir="./chroma_db", collection_name="kimi_claw_memory"):
        self.persist_dir = persist_dir
        self.collection_name = collection_name
        self.status = "initializing"
        self.client = None
        self.collection = None
        
        # 确保目录存在
        os.makedirs(persist_dir, exist_ok=True)
        
        if CHROMA_AVAILABLE:
            try:
                # 使用持久化客户端
                self.client = chromadb.PersistentClient(
                    path=persist_dir,
                    settings=Settings(
                        anonymized_telemetry=False,
                        allow_reset=True
                    )
                )
                
                # 获取或创建集合
                try:
                    self.collection = self.client.get_collection(name=collection_name)
                    print(f"📂 已加载现有集合: {collection_name}")
                except:
                    self.collection = self.client.create_collection(name=collection_name)
                    print(f"✨ 已创建新集合: {collection_name}")
                
                self.status = "running"
            except Exception as e:
                self.status = f"error: {str(e)}"
                print(f"❌ Chroma初始化失败: {e}")
        else:
            self.status = "fallback_to_file"
            print("📁 使用文件存储模式")
            self._init_file_fallback()
    
    def _init_file_fallback(self):
        """初始化文件存储备用方案"""
        self.file_db_path = os.path.join(self.persist_dir, "memory_fallback.json")
        if os.path.exists(self.file_db_path):
            with open(self.file_db_path, 'r', encoding='utf-8') as f:
                self.file_data = json.load(f)
        else:
            self.file_data = {"memories": [], "metadata": {}}
    
    def _save_file_fallback(self):
        """保存文件备用数据"""
        with open(self.file_db_path, 'w', encoding='utf-8') as f:
            json.dump(self.file_data, f, ensure_ascii=False, indent=2)
    
    def add(self, text, metadata=None, id=None):
        """添加记忆到向量数据库
        
        Args:
            text: 要存储的文本内容
            metadata: 可选的元数据字典
            id: 可选的唯一标识符
        
        Returns:
            dict: 操作结果
        """
        if metadata is None:
            metadata = {}
        
        # 添加时间戳
        metadata['timestamp'] = datetime.now().isoformat()
        
        if self.status == "running" and self.collection:
            try:
                # 生成ID（如果没有提供）
                if id is None:
                    id = f"mem_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{hash(text) % 10000}"
                
                self.collection.add(
                    documents=[text],
                    metadatas=[metadata],
                    ids=[id]
                )
                return {'success': True, 'id': id, 'method': 'chroma'}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        else:
            # 文件备用模式
            memory_entry = {
                'id': id or f"mem_{len(self.file_data['memories'])}",
                'text': text,
                'metadata': metadata
            }
            self.file_data['memories'].append(memory_entry)
            self._save_file_fallback()
            return {'success': True, 'id': memory_entry['id'], 'method': 'file'}
    
    def search(self, query, n_results=5, filter=None):
        """语义搜索记忆
        
        Args:
            query: 搜索查询文本
            n_results: 返回结果数量
            filter: 可选的元数据过滤条件
        
        Returns:
            list: 搜索结果列表
        """
        if self.status == "running" and self.collection:
            try:
                results = self.collection.query(
                    query_texts=[query],
                    n_results=n_results,
                    where=filter
                )
                
                # 格式化结果
                formatted_results = []
                if results['ids'] and len(results['ids'][0]) > 0:
                    for i in range(len(results['ids'][0])):
                        formatted_results.append({
                            'id': results['ids'][0][i],
                            'text': results['documents'][0][i],
                            'metadata': results['metadatas'][0][i] if results['metadatas'] else {},
                            'distance': results['distances'][0][i] if results['distances'] else None
                        })
                return formatted_results
            except Exception as e:
                return [{'error': str(e)}]
        else:
            # 文件备用模式 - 简单文本匹配
            results = []
            query_lower = query.lower()
            for mem in self.file_data['memories']:
                if query_lower in mem['text'].lower():
                    results.append(mem)
                    if len(results) >= n_results:
                        break
            return results
    
    def get_stats(self):
        """获取数据库统计信息"""
        if self.status == "running" and self.collection:
            try:
                count = self.collection.count()
                return {
                    'status': self.status,
                    'collection': self.collection_name,
                    'count': count,
                    'persist_dir': self.persist_dir
                }
            except Exception as e:
                return {'status': 'error', 'error': str(e)}
        else:
            return {
                'status': self.status,
                'count': len(self.file_data.get('memories', [])),
                'persist_dir': self.persist_dir
            }
    
    def delete(self, id):
        """删除指定ID的记忆"""
        if self.status == "running" and self.collection:
            try:
                self.collection.delete(ids=[id])
                return {'success': True}
            except Exception as e:
                return {'success': False, 'error': str(e)}
        else:
            self.file_data['memories'] = [m for m in self.file_data['memories'] if m['id'] != id]
            self._save_file_fallback()
            return {'success': True}

# 全局实例
chroma_memory = ChromaMemory()

def main():
    """主函数 - CLI入口"""
    import sys
    
    if len(sys.argv) < 2:
        # 显示状态
        stats = chroma_memory.get_stats()
        print("=" * 50)
        print("🧠 Chroma向量数据库")
        print("=" * 50)
        print(f"状态: {stats['status']}")
        print(f"集合: {chroma_memory.collection_name}")
        print(f"存储: {chroma_memory.persist_dir}")
        print(f"记录数: {stats.get('count', 0)}")
        print("=" * 50)
        print("\n用法:")
        print("  python chroma_memory.py add '记忆内容' [metadata_json]")
        print("  python chroma_memory.py search '查询内容' [n_results]")
        print("  python chroma_memory.py stats")
        return
    
    command = sys.argv[1]
    
    if command == "add":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供记忆内容")
            return
        text = sys.argv[2]
        metadata = json.loads(sys.argv[3]) if len(sys.argv) > 3 else {}
        result = chroma_memory.add(text, metadata)
        print(f"✅ 已添加: {result}")
    
    elif command == "search":
        if len(sys.argv) < 3:
            print("❌ 错误: 需要提供查询内容")
            return
        query = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 5
        results = chroma_memory.search(query, n_results=n)
        print(f"🔍 搜索结果 ({len(results)} 条):")
        for i, r in enumerate(results, 1):
            print(f"  {i}. {r.get('text', 'N/A')[:100]}...")
    
    elif command == "stats":
        stats = chroma_memory.get_stats()
        print(json.dumps(stats, indent=2, ensure_ascii=False))
    
    else:
        print(f"❌ 未知命令: {command}")

if __name__ == '__main__':
    main()
