#!/usr/bin/env python3
"""
记忆数据迁移工具 - 从文件迁移到Chroma向量数据库
"""

import os
import json
import glob
from datetime import datetime
from chroma_memory import ChromaMemory

def migrate_from_memory_files(source_dir="../../memory", target_collection="kimi_claw_memory"):
    """从memory目录迁移记忆文件到向量数据库"""
    
    memory = ChromaMemory(
        persist_dir="./chroma_db",
        collection_name=target_collection
    )
    
    # 查找所有记忆文件
    memory_files = glob.glob(os.path.join(source_dir, "*.md"))
    
    if not memory_files:
        print(f"⚠️ 在 {source_dir} 中未找到记忆文件")
        return 0
    
    print(f"📁 找到 {len(memory_files)} 个记忆文件")
    
    migrated_count = 0
    
    for file_path in memory_files:
        filename = os.path.basename(file_path)
        print(f"🔄 正在处理: {filename}")
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # 解析文件名获取日期
            date_str = filename.replace('.md', '')
            
            # 按段落分割内容
            paragraphs = [p.strip() for p in content.split('\n\n') if p.strip()]
            
            for i, paragraph in enumerate(paragraphs):
                if len(paragraph) < 20:  # 跳过太短的段落
                    continue
                
                # 生成元数据
                metadata = {
                    "source_file": filename,
                    "date": date_str,
                    "paragraph_index": i,
                    "type": "memory",
                    "migrated_at": datetime.now().isoformat()
                }
                
                # 生成唯一ID
                memory_id = f"{date_str}_{i:03d}"
                
                # 添加到向量数据库
                result = memory.add(
                    text=paragraph,
                    metadata=metadata,
                    id=memory_id
                )
                
                if result.get('success'):
                    migrated_count += 1
            
            print(f"  ✅ 已迁移 {len(paragraphs)} 个段落")
            
        except Exception as e:
            print(f"  ❌ 处理失败: {e}")
    
    print(f"\n🎉 迁移完成! 共迁移 {migrated_count} 条记忆")
    return migrated_count

def migrate_from_json(json_path, target_collection="kimi_claw_memory"):
    """从JSON文件批量导入记忆"""
    
    memory = ChromaMemory(
        persist_dir="./chroma_db",
        collection_name=target_collection
    )
    
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    if isinstance(data, dict):
        items = data.get('memories', [data])
    else:
        items = data
    
    print(f"📦 准备导入 {len(items)} 条记忆")
    
    migrated_count = 0
    for item in items:
        try:
            text = item.get('text') or item.get('content')
            if not text:
                continue
            
            metadata = item.get('metadata', {})
            metadata['migrated_at'] = datetime.now().isoformat()
            
            result = memory.add(
                text=text,
                metadata=metadata,
                id=item.get('id')
            )
            
            if result.get('success'):
                migrated_count += 1
                
        except Exception as e:
            print(f"❌ 导入失败: {e}")
    
    print(f"🎉 导入完成! 成功 {migrated_count}/{len(items)}")
    return migrated_count

def export_to_json(output_path, collection_name="kimi_claw_memory"):
    """导出向量数据库到JSON"""
    
    memory = ChromaMemory(
        persist_dir="./chroma_db",
        collection_name=collection_name
    )
    
    # 获取所有数据（通过空查询）
    # 注意：ChromaDB的get_all需要特殊处理
    stats = memory.get_stats()
    count = stats.get('count', 0)
    
    print(f"📊 集合中有 {count} 条记录")
    
    # 使用搜索获取所有（通过通用查询）
    all_results = memory.search("the", n_results=min(count, 1000))
    
    export_data = {
        "exported_at": datetime.now().isoformat(),
        "collection": collection_name,
        "count": len(all_results),
        "memories": all_results
    }
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, ensure_ascii=False, indent=2)
    
    print(f"💾 已导出到: {output_path}")
    return export_data

if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("用法:")
        print("  python migrate_data.py migrate --from-dir ../../memory")
        print("  python migrate_data.py import ./data.json")
        print("  python migrate_data.py export ./backup.json")
        sys.exit(1)
    
    command = sys.argv[1]
    
    if command == "migrate":
        source_dir = sys.argv[3] if len(sys.argv) > 3 else "../../memory"
        migrate_from_memory_files(source_dir)
    
    elif command == "import":
        if len(sys.argv) < 3:
            print("❌ 需要指定JSON文件路径")
            sys.exit(1)
        migrate_from_json(sys.argv[2])
    
    elif command == "export":
        if len(sys.argv) < 3:
            print("❌ 需要指定输出文件路径")
            sys.exit(1)
        export_to_json(sys.argv[2])
    
    else:
        print(f"❌ 未知命令: {command}")
