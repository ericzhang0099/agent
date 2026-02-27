#!/usr/bin/env python3
"""
CPT增量更新机制 - Character Persona Training
基于CharacterGPT论文实现
"""

import json
from datetime import datetime
from typing import Dict, List, Optional
from dataclasses import dataclass, asdict

@dataclass
class Epoch:
    """纪元/章节定义"""
    id: str
    name: str
    start_time: str
    end_time: Optional[str]
    personality_snapshot: Dict
    key_events: List[str]
    
@dataclass
class MemoryUpdate:
    """记忆更新记录"""
    timestamp: str
    epoch_id: str
    update_type: str  # 'personality', 'backstory', 'emotion', 'relationship'
    content: str
    importance: float  # 0-1
    
class CPTIncrementalUpdater:
    """CPT增量更新器"""
    
    def __init__(self, persona_name: str = "Kimi Claw"):
        self.persona_name = persona_name
        self.epochs: List[Epoch] = []
        self.current_epoch: Optional[Epoch] = None
        self.memory_updates: List[MemoryUpdate] = []
        self.persona_vector = self._init_persona_vector()
        
    def _init_persona_vector(self) -> Dict:
        """初始化人格向量（8维度）"""
        return {
            'personality': {},
            'physical': {},
            'motivations': {},
            'backstory': {},
            'emotions': {},
            'relationships': {},
            'growth': {},
            'conflict': {}
        }
    
    def start_epoch(self, name: str, initial_snapshot: Dict) -> str:
        """开始新纪元"""
        epoch_id = f"epoch_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # 结束当前纪元
        if self.current_epoch:
            self.current_epoch.end_time = datetime.now().isoformat()
        
        # 创建新纪元
        new_epoch = Epoch(
            id=epoch_id,
            name=name,
            start_time=datetime.now().isoformat(),
            end_time=None,
            personality_snapshot=initial_snapshot,
            key_events=[]
        )
        
        self.epochs.append(new_epoch)
        self.current_epoch = new_epoch
        
        return epoch_id
    
    def update_dimension(self, dimension: str, update_content: str, 
                        importance: float = 0.5) -> bool:
        """
        增量更新单个人格维度
        
        Args:
            dimension: 维度名称（8维度之一）
            update_content: 更新内容
            importance: 重要性（0-1）
        """
        if dimension not in self.persona_vector:
            return False
            
        # 创建记忆更新记录
        update = MemoryUpdate(
            timestamp=datetime.now().isoformat(),
            epoch_id=self.current_epoch.id if self.current_epoch else "default",
            update_type=dimension,
            content=update_content,
            importance=importance
        )
        
        self.memory_updates.append(update)
        
        # 更新人格向量（增量式）
        if 'updates' not in self.persona_vector[dimension]:
            self.persona_vector[dimension]['updates'] = []
            
        self.persona_vector[dimension]['updates'].append({
            'content': update_content,
            'importance': importance,
            'timestamp': update.timestamp
        })
        
        # 添加到当前纪元的关键事件
        if self.current_epoch:
            self.current_epoch.key_events.append({
                'type': dimension,
                'content': update_content[:100] + '...' if len(update_content) > 100 else update_content,
                'timestamp': update.timestamp
            })
        
        return True
    
    def get_persona_at_epoch(self, epoch_id: str) -> Optional[Dict]:
        """获取特定纪元的人格快照"""
        for epoch in self.epochs:
            if epoch.id == epoch_id:
                return epoch.personality_snapshot
        return None
    
    def generate_training_data(self) -> List[Dict]:
        """生成CPT训练数据"""
        training_data = []
        
        for update in self.memory_updates:
            training_data.append({
                'instruction': f"Update {update.update_type} dimension",
                'input': update.content,
                'output': f"Persona updated: {update.content}",
                'metadata': {
                    'timestamp': update.timestamp,
                    'epoch': update.epoch_id,
                    'importance': update.importance
                }
            })
            
        return training_data
    
    def export_persona(self) -> Dict:
        """导出完整人格档案"""
        return {
            'name': self.persona_name,
            'current_vector': self.persona_vector,
            'epochs': [asdict(e) for e in self.epochs],
            'memory_updates': [asdict(m) for m in self.memory_updates],
            'export_time': datetime.now().isoformat()
        }

# 全局实例
cpt_updater = CPTIncrementalUpdater()

if __name__ == '__main__':
    print("🧠 CPT增量更新机制已部署")
    print("支持：章节式记忆更新 + 增量人格训练 + 纪元式角色存储")
    
    # 演示
    epoch_id = cpt_updater.start_epoch("v3.0升级", {
        'version': '3.0',
        'dimensions': 8,
        'emotions': 16
    })
    print(f"\n开始新纪元: {epoch_id}")
    
    cpt_updater.update_dimension('growth', 
        '完成SOUL.md v3.0重构，引入8维度人格模型',
        importance=0.9)
    print("记录重要更新: SOUL.md v3.0重构")
    
    print(f"\n当前记忆更新数: {len(cpt_updater.memory_updates)}")
    print(f"纪元数: {len(cpt_updater.epochs)}")
