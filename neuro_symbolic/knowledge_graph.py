"""
知识图谱基础模块
包含知识图谱表示、实体关系管理和基本操作
"""

from typing import Dict, List, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from collections import defaultdict
import json
import random


@dataclass
class Entity:
    """知识图谱实体"""
    id: str
    name: str
    entity_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Entity):
            return self.id == other.id
        return False
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'type': self.entity_type,
            'attributes': self.attributes
        }


@dataclass
class Relation:
    """知识图谱关系"""
    id: str
    name: str
    head_entity_id: str
    tail_entity_id: str
    relation_type: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    
    def __hash__(self):
        return hash(self.id)
    
    def __eq__(self, other):
        if isinstance(other, Relation):
            return self.id == other.id
        return False
    
    def to_dict(self) -> Dict:
        return {
            'id': self.id,
            'name': self.name,
            'head': self.head_entity_id,
            'tail': self.tail_entity_id,
            'type': self.relation_type,
            'attributes': self.attributes
        }


@dataclass
class Triple:
    """知识图谱三元组 (h, r, t)"""
    head: str
    relation: str
    tail: str
    
    def __hash__(self):
        return hash((self.head, self.relation, self.tail))
    
    def __eq__(self, other):
        if isinstance(other, Triple):
            return (self.head, self.relation, self.tail) == (other.head, other.relation, other.tail)
        return False
    
    def __repr__(self):
        return f"({self.head}, {self.relation}, {self.tail})"


class KnowledgeGraph:
    """
    知识图谱基础类
    
    支持功能：
    - 实体和关系的增删改查
    - 三元组管理
    - 邻居查询
    - 路径搜索
    - 子图提取
    """
    
    def __init__(self, name: str = "KG"):
        self.name = name
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.triples: Set[Triple] = set()
        
        # 索引结构
        self.entity_types: Dict[str, Set[str]] = defaultdict(set)  # type -> entity_ids
        self.relation_types: Dict[str, Set[str]] = defaultdict(set)  # type -> relation_ids
        self.outgoing_edges: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relation_ids
        self.incoming_edges: Dict[str, Set[str]] = defaultdict(set)  # entity_id -> relation_ids
        self.entity_triples: Dict[str, Set[Triple]] = defaultdict(set)  # entity_id -> triples
        
    def add_entity(self, entity: Entity) -> bool:
        """添加实体"""
        if entity.id in self.entities:
            return False
        
        self.entities[entity.id] = entity
        self.entity_types[entity.entity_type].add(entity.id)
        return True
    
    def add_relation(self, relation: Relation) -> bool:
        """添加关系"""
        if relation.id in self.relations:
            return False
        
        # 检查头尾实体是否存在
        if relation.head_entity_id not in self.entities:
            raise ValueError(f"Head entity {relation.head_entity_id} not found")
        if relation.tail_entity_id not in self.entities:
            raise ValueError(f"Tail entity {relation.tail_entity_id} not found")
        
        self.relations[relation.id] = relation
        self.relation_types[relation.relation_type].add(relation.id)
        
        # 更新索引
        self.outgoing_edges[relation.head_entity_id].add(relation.id)
        self.incoming_edges[relation.tail_entity_id].add(relation.id)
        
        # 创建三元组
        triple = Triple(
            head=relation.head_entity_id,
            relation=relation.relation_type,
            tail=relation.tail_entity_id
        )
        self.triples.add(triple)
        self.entity_triples[relation.head_entity_id].add(triple)
        self.entity_triples[relation.tail_entity_id].add(triple)
        
        return True
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """获取实体"""
        return self.entities.get(entity_id)
    
    def get_relation(self, relation_id: str) -> Optional[Relation]:
        """获取关系"""
        return self.relations.get(relation_id)
    
    def remove_entity(self, entity_id: str) -> bool:
        """删除实体及其关联关系"""
        if entity_id not in self.entities:
            return False
        
        # 删除关联的关系
        related_relations = (
            self.outgoing_edges[entity_id] | self.incoming_edges[entity_id]
        )
        for rel_id in list(related_relations):
            self.remove_relation(rel_id)
        
        # 删除实体
        entity = self.entities[entity_id]
        del self.entities[entity_id]
        self.entity_types[entity.entity_type].discard(entity_id)
        
        return True
    
    def remove_relation(self, relation_id: str) -> bool:
        """删除关系"""
        if relation_id not in self.relations:
            return False
        
        relation = self.relations[relation_id]
        
        # 更新索引
        self.outgoing_edges[relation.head_entity_id].discard(relation_id)
        self.incoming_edges[relation.tail_entity_id].discard(relation_id)
        
        # 删除三元组
        triple = Triple(
            head=relation.head_entity_id,
            relation=relation.relation_type,
            tail=relation.tail_entity_id
        )
        self.triples.discard(triple)
        self.entity_triples[relation.head_entity_id].discard(triple)
        self.entity_triples[relation.tail_entity_id].discard(triple)
        
        # 删除关系
        del self.relations[relation_id]
        self.relation_types[relation.relation_type].discard(relation_id)
        
        return True
    
    def get_neighbors(self, entity_id: str, direction: str = "both") -> List[Tuple[str, str, str]]:
        """
        获取实体的邻居
        
        Args:
            entity_id: 实体ID
            direction: "out"(出边), "in"(入边), "both"(双向)
        
        Returns:
            List of (neighbor_id, relation_type, direction)
        """
        neighbors = []
        
        if direction in ("out", "both"):
            for rel_id in self.outgoing_edges[entity_id]:
                relation = self.relations[rel_id]
                neighbors.append((relation.tail_entity_id, relation.relation_type, "out"))
        
        if direction in ("in", "both"):
            for rel_id in self.incoming_edges[entity_id]:
                relation = self.relations[rel_id]
                neighbors.append((relation.head_entity_id, relation.relation_type, "in"))
        
        return neighbors
    
    def get_triples_by_entity(self, entity_id: str) -> Set[Triple]:
        """获取实体相关的所有三元组"""
        return self.entity_triples[entity_id].copy()
    
    def find_paths(self, start_id: str, end_id: str, max_length: int = 3) -> List[List[Triple]]:
        """
        查找两个实体之间的路径
        
        Args:
            start_id: 起始实体ID
            end_id: 目标实体ID
            max_length: 最大路径长度
        
        Returns:
            List of paths, each path is a list of Triples
        """
        if start_id not in self.entities or end_id not in self.entities:
            return []
        
        paths = []
        visited = set()
        
        def dfs(current_id: str, target_id: str, current_path: List[Triple], depth: int):
            if depth > max_length:
                return
            
            if current_id == target_id and len(current_path) > 0:
                paths.append(current_path.copy())
                return
            
            visited.add(current_id)
            
            # 遍历出边
            for rel_id in self.outgoing_edges[current_id]:
                relation = self.relations[rel_id]
                next_id = relation.tail_entity_id
                
                if next_id not in visited:
                    triple = Triple(current_id, relation.relation_type, next_id)
                    current_path.append(triple)
                    dfs(next_id, target_id, current_path, depth + 1)
                    current_path.pop()
            
            visited.remove(current_id)
        
        dfs(start_id, end_id, [], 0)
        return paths
    
    def extract_subgraph(self, entity_ids: Set[str], depth: int = 1) -> 'KnowledgeGraph':
        """
        提取子图
        
        Args:
            entity_ids: 种子实体ID集合
            depth: 扩展深度
        
        Returns:
            子图
        """
        subgraph = KnowledgeGraph(name=f"{self.name}_subgraph")
        
        # BFS扩展
        current_level = entity_ids.copy()
        all_entities = entity_ids.copy()
        
        for _ in range(depth):
            next_level = set()
            for entity_id in current_level:
                if entity_id not in self.entities:
                    continue
                
                # 添加邻居
                for rel_id in self.outgoing_edges[entity_id]:
                    relation = self.relations[rel_id]
                    next_level.add(relation.tail_entity_id)
                
                for rel_id in self.incoming_edges[entity_id]:
                    relation = self.relations[rel_id]
                    next_level.add(relation.head_entity_id)
            
            all_entities.update(next_level)
            current_level = next_level
        
        # 复制实体
        for entity_id in all_entities:
            if entity_id in self.entities:
                subgraph.add_entity(self.entities[entity_id])
        
        # 复制关系
        for relation in self.relations.values():
            if (relation.head_entity_id in all_entities and 
                relation.tail_entity_id in all_entities):
                subgraph.add_relation(relation)
        
        return subgraph
    
    def get_statistics(self) -> Dict:
        """获取图谱统计信息"""
        return {
            'name': self.name,
            'num_entities': len(self.entities),
            'num_relations': len(self.relations),
            'num_triples': len(self.triples),
            'entity_types': {t: len(ids) for t, ids in self.entity_types.items()},
            'relation_types': {t: len(ids) for t, ids in self.relation_types.items()}
        }
    
    def to_dict(self) -> Dict:
        """转换为字典"""
        return {
            'name': self.name,
            'entities': [e.to_dict() for e in self.entities.values()],
            'relations': [r.to_dict() for r in self.relations.values()],
            'triples': [{'head': t.head, 'relation': t.relation, 'tail': t.tail} for t in self.triples]
        }
    
    def save(self, filepath: str):
        """保存到文件"""
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)
    
    @classmethod
    def load(cls, filepath: str) -> 'KnowledgeGraph':
        """从文件加载"""
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        kg = cls(name=data['name'])
        
        # 加载实体
        for e_data in data['entities']:
            entity = Entity(
                id=e_data['id'],
                name=e_data['name'],
                entity_type=e_data['type'],
                attributes=e_data.get('attributes', {})
            )
            kg.add_entity(entity)
        
        # 加载关系
        for r_data in data['relations']:
            relation = Relation(
                id=r_data['id'],
                name=r_data['name'],
                head_entity_id=r_data['head'],
                tail_entity_id=r_data['tail'],
                relation_type=r_data['type'],
                attributes=r_data.get('attributes', {})
            )
            kg.add_relation(relation)
        
        return kg
    
    def __repr__(self):
        stats = self.get_statistics()
        return f"KnowledgeGraph({stats['name']}: {stats['num_entities']} entities, {stats['num_relations']} relations, {stats['num_triples']} triples)"


class KnowledgeGraphBuilder:
    """知识图谱构建工具"""
    
    def __init__(self):
        self.kg = KnowledgeGraph()
        self._entity_counter = 0
        self._relation_counter = 0
    
    def add_entity(self, name: str, entity_type: str, **attributes) -> str:
        """添加实体并返回ID"""
        self._entity_counter += 1
        entity_id = f"E{self._entity_counter}"
        entity = Entity(
            id=entity_id,
            name=name,
            entity_type=entity_type,
            attributes=attributes
        )
        self.kg.add_entity(entity)
        return entity_id
    
    def add_relation(self, head_id: str, tail_id: str, relation_type: str, **attributes) -> str:
        """添加关系并返回ID"""
        self._relation_counter += 1
        relation_id = f"R{self._relation_counter}"
        relation = Relation(
            id=relation_id,
            name=relation_type,
            head_entity_id=head_id,
            tail_entity_id=tail_id,
            relation_type=relation_type,
            attributes=attributes
        )
        self.kg.add_relation(relation)
        return relation_id
    
    def build(self) -> KnowledgeGraph:
        """返回构建好的知识图谱"""
        return self.kg


def create_sample_kg() -> KnowledgeGraph:
    """创建示例知识图谱"""
    builder = KnowledgeGraphBuilder()
    
    # 添加人物实体
    alice = builder.add_entity("Alice", "Person", age=30, occupation="Researcher")
    bob = builder.add_entity("Bob", "Person", age=35, occupation="Engineer")
    charlie = builder.add_entity("Charlie", "Person", age=28, occupation="Student")
    
    # 添加机构实体
    mit = builder.add_entity("MIT", "Organization", location="Boston")
    stanford = builder.add_entity("Stanford", "Organization", location="California")
    
    # 添加论文实体
    paper1 = builder.add_entity("Neural Networks", "Paper", year=2023)
    paper2 = builder.add_entity("Knowledge Graphs", "Paper", year=2024)
    
    # 添加关系
    builder.add_relation(alice, bob, "colleague")
    builder.add_relation(bob, charlie, "mentor")
    builder.add_relation(alice, mit, "works_at")
    builder.add_relation(bob, stanford, "works_at")
    builder.add_relation(charlie, stanford, "studies_at")
    builder.add_relation(alice, paper1, "author_of")
    builder.add_relation(bob, paper2, "author_of")
    builder.add_relation(paper1, paper2, "cites")
    
    return builder.build()


if __name__ == "__main__":
    # 测试
    kg = create_sample_kg()
    print(kg)
    print("\nStatistics:", kg.get_statistics())
    
    # 测试邻居查询
    print("\nNeighbors of Alice:")
    for neighbor, rel_type, direction in kg.get_neighbors("E1"):
        print(f"  -> {neighbor} via {rel_type} ({direction})")
    
    # 测试路径查找
    print("\nPaths from Alice to Charlie:")
    paths = kg.find_paths("E1", "E3", max_length=3)
    for i, path in enumerate(paths):
        print(f"  Path {i+1}: {' -> '.join([str(t) for t in path])}")
