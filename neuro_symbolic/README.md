# 神经符号AI研究 - 第一部分：知识图谱基础

## 概述

本模块实现了知识图谱的基础功能，包括实体关系管理、图嵌入学习（TransE/RotatE/ComplEx）等核心组件。

## 文件结构

```
neuro_symbolic/
├── knowledge_graph.py      # 知识图谱基础类
├── embedding_models.py     # 嵌入模型实现
└── README.md               # 本文档
```

## 知识图谱基础 (knowledge_graph.py)

### 核心类

#### `Entity` - 实体
```python
@dataclass
class Entity:
    id: str           # 唯一标识
    name: str         # 名称
    entity_type: str  # 类型
    attributes: Dict  # 属性
```

#### `Relation` - 关系
```python
@dataclass
class Relation:
    id: str               # 唯一标识
    name: str             # 名称
    head_entity_id: str   # 头实体ID
    tail_entity_id: str   # 尾实体ID
    relation_type: str    # 关系类型
    attributes: Dict      # 属性
```

#### `Triple` - 三元组
```python
@dataclass
class Triple:
    head: str      # 头实体
    relation: str  # 关系
    tail: str      # 尾实体
```

#### `KnowledgeGraph` - 知识图谱

主要功能：
- **实体管理**: `add_entity()`, `get_entity()`, `remove_entity()`
- **关系管理**: `add_relation()`, `get_relation()`, `remove_relation()`
- **邻居查询**: `get_neighbors(entity_id, direction)`
- **路径搜索**: `find_paths(start, end, max_length)`
- **子图提取**: `extract_subgraph(entity_ids, depth)`
- **持久化**: `save()`, `load()`

### 使用示例

```python
from knowledge_graph import KnowledgeGraph, Entity, Relation, KnowledgeGraphBuilder

# 方法1: 使用Builder构建
builder = KnowledgeGraphBuilder()
alice = builder.add_entity("Alice", "Person", age=30)
bob = builder.add_entity("Bob", "Person", age=35)
builder.add_relation(alice, bob, "friend")
kg = builder.build()

# 方法2: 直接操作
kg = KnowledgeGraph()
entity = Entity(id="E1", name="Alice", entity_type="Person")
kg.add_entity(entity)

# 查询邻居
neighbors = kg.get_neighbors("E1", direction="both")

# 查找路径
paths = kg.find_paths("E1", "E2", max_length=3)

# 提取子图
subgraph = kg.extract_subgraph({"E1", "E2"}, depth=2)
```

## 嵌入模型 (embedding_models.py)

### TransE

**论文**: Translating Embeddings for Modeling Multi-relational Data (Bordes et al., 2013)

**核心思想**: 将关系视为实体间的平移变换
- 假设: h + r ≈ t
- 得分函数: ||h + r - t||

**特点**:
- 简单高效
- 适合1对1关系
- 难以处理1对多、多对1、多对多关系

```python
from embedding_models import TransE

transe = TransE(kg, embedding_dim=100)
transe.train(epochs=100, batch_size=32, learning_rate=0.01)

# 预测尾实体
predictions = transe.predict_tail("E1", "friend", top_k=10)
```

### RotatE

**论文**: RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space (Sun et al., 2019)

**核心思想**: 在复数空间中使用旋转操作建模关系
- 假设: t = h ∘ r (逐元素复数乘法)
- 得分函数: ||h ∘ r - t||

**特点**:
- 能够建模对称、反对称、逆、组合等多种关系模式
- 使用复数空间增强表达能力
- 支持自对抗负采样

```python
from embedding_models import RotatE

rotate = RotatE(kg, embedding_dim=100, gamma=12.0)
rotate.train(epochs=100, batch_size=256, learning_rate=0.0001)

# 预测
predictions = rotate.predict_tail("E1", "friend", top_k=10)
```

### ComplEx

**论文**: Complex Embeddings for Simple Link Prediction (Trouillon et al., 2016)

**核心思想**: 使用复数嵌入建模非对称关系
- 得分函数: Re(<h, r, conj(t)>)

**特点**:
- 通过复数共轭处理非对称关系
- 适合处理反对称关系

```python
from embedding_models import ComplEx

complex_model = ComplEx(kg, embedding_dim=100)
complex_model.train(epochs=100, learning_rate=0.01)
```

## 评估指标

```python
from embedding_models import evaluate_model

metrics = evaluate_model(model, test_triples, k=10)
print(f"MR: {metrics['MR']:.2f}")
print(f"MRR: {metrics['MRR']:.4f}")
print(f"Hits@10: {metrics['Hits@10']:.4f}")
```

### 指标说明

- **MR (Mean Rank)**: 平均排名，越低越好
- **MRR (Mean Reciprocal Rank)**: 平均倒数排名，越高越好
- **Hits@K**: 正确答案排在前K的比例，越高越好

## 运行测试

```bash
cd neuro_symbolic

# 测试知识图谱
python knowledge_graph.py

# 测试嵌入模型
python embedding_models.py
```

## 扩展方向

1. **更多模型**: DistMult, ConvE, GCN-based方法
2. **关系抽取**: 从文本中自动抽取实体关系
3. **知识推理**: 基于规则的推理和路径推理
4. **知识融合**: 实体对齐和知识图谱融合
5. **时序建模**: 动态知识图谱

## 参考资料

1. Bordes, A., et al. (2013). Translating embeddings for modeling multi-relational data. NeurIPS.
2. Sun, Z., et al. (2019). RotatE: Knowledge graph embedding by relational rotation in complex space. ICLR.
3. Trouillon, T., et al. (2016). Complex embeddings for simple link prediction. ICML.
4. Wang, Q., et al. (2017). Knowledge graph embedding: A survey of approaches and applications. TKDE.
