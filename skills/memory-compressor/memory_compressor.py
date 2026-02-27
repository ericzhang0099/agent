#!/usr/bin/env python3
"""
长期记忆压缩优化
提升存储效率和检索速度
"""

import json
from datetime import datetime
from typing import Dict, List

class MemoryCompressor:
    """记忆压缩器"""
    
    def __init__(self):
        self.importance_thresholds = {
            'critical': 0.9,
            'high': 0.7,
            'medium': 0.4,
            'low': 0.0
        }
        
    def calculate_importance(self, memory: Dict) -> float:
        """计算记忆重要性"""
        factors = {
            'user_emphasis': 0.3,      # 用户强调
            'decision_related': 0.25,   # 决策相关
            'emotional_intensity': 0.2, # 情绪强度
            'uniqueness': 0.15,         # 独特性
            'recency': 0.1              # 时效性
        }
        
        score = 0
        # 简化的重要性计算
        if memory.get('user_said_remember', False):
            score += factors['user_emphasis']
        if memory.get('type') == 'decision':
            score += factors['decision_related']
        if memory.get('emotion') in ['excited', 'concerned', 'urgent']:
            score += factors['emotional_intensity']
            
        return min(1.0, score)
    
    def compress_memory(self, memory: Dict) -> Dict:
        """压缩单条记忆"""
        importance = self.calculate_importance(memory)
        
        if importance >= self.importance_thresholds['critical']:
            # 完整保留
            return memory
        elif importance >= self.importance_thresholds['high']:
            # 保留摘要
            return {
                'id': memory.get('id'),
                'timestamp': memory.get('timestamp'),
                'summary': memory.get('content', '')[:200] + '...',
                'importance': importance,
                'compressed': True
            }
        else:
            # 仅保留关键词
            return {
                'id': memory.get('id'),
                'timestamp': memory.get('timestamp'),
                'keywords': self._extract_keywords(memory.get('content', '')),
                'importance': importance,
                'compressed': True
            }
    
    def _extract_keywords(self, text: str) -> List[str]:
        """提取关键词"""
        # 简化版关键词提取
        words = text.lower().split()
        # 过滤常见词，保留重要词
        stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were'}
        keywords = [w for w in words if len(w) > 3 and w not in stopwords]
        return keywords[:10]  # 最多10个关键词

# 全局实例
memory_compressor = MemoryCompressor()

if __name__ == '__main__':
    print("🧠 长期记忆压缩优化系统")
    print("=" * 50)
    
    # 演示
    test_memory = {
        'id': 'mem_001',
        'timestamp': datetime.now().isoformat(),
        'content': '用户强调要记住这个重要的决策：立即启动SOUL.md v3.0重构项目',
        'type': 'decision',
        'emotion': 'urgent',
        'user_said_remember': True
    }
    
    importance = memory_compressor.calculate_importance(test_memory)
    print(f"记忆重要性评分: {importance:.2f}")
    
    compressed = memory_compressor.compress_memory(test_memory)
    print(f"\n压缩后记忆:")
    print(json.dumps(compressed, indent=2, ensure_ascii=False))
