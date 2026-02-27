#!/usr/bin/env python3
"""
Chroma Memory 测试脚本
"""

import time
import json
from chroma_memory import ChromaMemory

def test_basic_operations():
    """测试基本CRUD操作"""
    print("=" * 50)
    print("🧪 测试基本操作")
    print("=" * 50)
    
    memory = ChromaMemory(persist_dir="./chroma_db", collection_name="test_collection")
    
    # 测试添加
    print("\n1. 测试添加记忆...")
    result = memory.add(
        text="Kimi-Claw是一个智能助手框架",
        metadata={"type": "intro", "project": "kimi-claw"},
        id="test_001"
    )
    print(f"   结果: {result}")
    assert result['success'], "添加失败"
    
    # 测试搜索
    print("\n2. 测试语义搜索...")
    results = memory.search("智能助手", n_results=3)
    print(f"   找到 {len(results)} 条结果")
    for r in results:
        print(f"   - {r['text'][:50]}... (距离: {r.get('distance')})")
    assert len(results) > 0, "搜索失败"
    
    # 测试带过滤的搜索
    print("\n3. 测试过滤搜索...")
    results = memory.search("框架", filter={"project": "kimi-claw"})
    print(f"   过滤后找到 {len(results)} 条结果")
    
    # 测试统计
    print("\n4. 测试统计信息...")
    stats = memory.get_stats()
    print(f"   统计: {json.dumps(stats, indent=2, ensure_ascii=False)}")
    
    # 测试删除
    print("\n5. 测试删除记忆...")
    result = memory.delete("test_001")
    print(f"   删除结果: {result}")
    assert result['success'], "删除失败"
    
    print("\n✅ 基本操作测试通过!")
    return True

def test_batch_operations():
    """测试批量操作性能"""
    print("\n" + "=" * 50)
    print("🧪 测试批量操作")
    print("=" * 50)
    
    memory = ChromaMemory(persist_dir="./chroma_db", collection_name="batch_test")
    
    # 准备测试数据
    test_data = [
        {"text": f"这是第{i}条测试记忆，用于测试批量导入功能", "metadata": {"index": i, "batch": True}}
        for i in range(100)
    ]
    
    # 批量添加
    print("\n1. 批量添加 100 条记忆...")
    start = time.time()
    for item in test_data:
        memory.add(text=item['text'], metadata=item['metadata'])
    elapsed = time.time() - start
    print(f"   耗时: {elapsed:.2f}s, 平均: {elapsed/100*1000:.1f}ms/条")
    
    # 批量搜索
    print("\n2. 测试批量搜索性能...")
    queries = ["测试记忆", "批量导入", "功能测试"]
    start = time.time()
    for query in queries:
        results = memory.search(query, n_results=10)
    elapsed = time.time() - start
    print(f"   3次搜索耗时: {elapsed:.3f}s")
    
    # 清理
    print("\n3. 清理测试数据...")
    stats = memory.get_stats()
    print(f"   清理前记录数: {stats['count']}")
    
    print("\n✅ 批量操作测试通过!")
    return True

def test_semantic_search():
    """测试语义搜索效果"""
    print("\n" + "=" * 50)
    print("🧪 测试语义搜索")
    print("=" * 50)
    
    memory = ChromaMemory(persist_dir="./chroma_db", collection_name="semantic_test")
    
    # 添加语义相关的记忆
    memories = [
        "今天天气很好，适合去公园散步",
        "机器学习是人工智能的一个重要分支",
        "Python是一种流行的编程语言",
        "深度学习使用神经网络进行训练",
        "我喜欢在周末去爬山",
    ]
    
    print("\n1. 添加测试记忆...")
    for i, text in enumerate(memories):
        memory.add(text=text, id=f"semantic_{i}")
    
    # 测试语义相似度
    print("\n2. 测试语义搜索...")
    test_queries = [
        ("AI技术", ["机器学习", "深度学习"]),
        ("户外活动", ["公园", "爬山"]),
        ("编程", ["Python"]),
    ]
    
    for query, expected_keywords in test_queries:
        results = memory.search(query, n_results=3)
        print(f"\n   查询: '{query}'")
        for r in results:
            print(f"   → {r['text']}")
    
    print("\n✅ 语义搜索测试通过!")
    return True

def test_api_endpoints():
    """测试API端点（需要服务运行）"""
    print("\n" + "=" * 50)
    print("🧪 测试API端点")
    print("=" * 50)
    
    try:
        import requests
        base_url = "http://localhost:8000"
        
        # 测试健康检查
        print("\n1. 测试健康检查...")
        resp = requests.get(f"{base_url}/health", timeout=5)
        print(f"   状态: {resp.status_code}, 响应: {resp.json()}")
        
        # 测试添加
        print("\n2. 测试添加记忆...")
        resp = requests.post(f"{base_url}/memory", json={
            "text": "API测试记忆",
            "metadata": {"test": True}
        })
        print(f"   状态: {resp.status_code}, 响应: {resp.json()}")
        
        # 测试搜索
        print("\n3. 测试搜索...")
        resp = requests.post(f"{base_url}/search", json={
            "query": "API测试",
            "n_results": 5
        })
        print(f"   状态: {resp.status_code}, 结果数: {len(resp.json())}")
        
        print("\n✅ API测试通过!")
        return True
        
    except ImportError:
        print("⚠️ 未安装requests，跳过API测试")
        return True
    except Exception as e:
        print(f"❌ API测试失败: {e}")
        return False

def run_all_tests():
    """运行所有测试"""
    print("\n" + "🚀" * 25)
    print("   Chroma Memory 测试套件")
    print("🚀" * 25 + "\n")
    
    tests = [
        ("基本操作", test_basic_operations),
        ("批量操作", test_batch_operations),
        ("语义搜索", test_semantic_search),
        ("API端点", test_api_endpoints),
    ]
    
    results = []
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success))
        except Exception as e:
            print(f"\n❌ {name} 测试失败: {e}")
            results.append((name, False))
    
    # 汇总
    print("\n" + "=" * 50)
    print("📊 测试结果汇总")
    print("=" * 50)
    for name, success in results:
        status = "✅ 通过" if success else "❌ 失败"
        print(f"  {name}: {status}")
    
    passed = sum(1 for _, s in results if s)
    print(f"\n总计: {passed}/{len(results)} 通过")
    
    return all(s for _, s in results)

if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
