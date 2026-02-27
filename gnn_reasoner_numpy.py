"""
GNN Reasoner (NumPy版本) - 图神经网络知识图谱推理器

本模块使用纯NumPy实现了基于图神经网络的知识图谱推理系统，包含：
1. GraphSAGE简化版 - 归纳式图神经网络
2. 知识图谱嵌入 - TransE/RotatE风格的关系学习
3. 图注意力机制 - 多头注意力层
4. 关系预测 - 完整的推理流程

作者: AI Research Agent
日期: 2026-02-27
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import random


# =============================================================================
# 工具函数
# =============================================================================

def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Softmax函数"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)

def leaky_relu(x: np.ndarray, alpha: float = 0.2) -> np.ndarray:
    """LeakyReLU激活函数"""
    return np.where(x > 0, x, alpha * x)

def elu(x: np.ndarray, alpha: float = 1.0) -> np.ndarray:
    """ELU激活函数"""
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def xavier_uniform(shape: Tuple[int, ...]) -> np.ndarray:
    """Xavier均匀初始化"""
    limit = np.sqrt(6.0 / (shape[0] + shape[-1]))
    return np.random.uniform(-limit, limit, size=shape)


# =============================================================================
# 1. GraphSAGE 简化版实现
# =============================================================================

class GraphSAGELayer:
    """
    GraphSAGE层实现 (NumPy版本)
    
    论文: Inductive Representation Learning on Large Graphs (Hamilton et al., 2017)
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 aggregator: str = 'mean', concat: bool = True):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            aggregator: 聚合函数 ('mean', 'max')
            concat: 是否拼接自身特征和邻居聚合特征
        """
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator
        self.concat = concat
        
        # 初始化权重
        if concat:
            self.weight = xavier_uniform((2 * input_dim, output_dim))
        else:
            self.weight = xavier_uniform((input_dim, output_dim))
        
        self.bias = np.zeros(output_dim)
    
    def aggregate_neighbors(self, neighbor_features: np.ndarray) -> np.ndarray:
        """
        聚合邻居特征
        
        Args:
            neighbor_features: [num_nodes, num_neighbors, feature_dim]
        
        Returns:
            aggregated: [num_nodes, feature_dim]
        """
        if self.aggregator == 'mean':
            return np.mean(neighbor_features, axis=1)
        elif self.aggregator == 'max':
            return np.max(neighbor_features, axis=1)
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
    
    def forward(self, node_features: np.ndarray, 
                neighbor_features: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            node_features: [num_nodes, feature_dim]
            neighbor_features: [num_nodes, num_neighbors, feature_dim]
        
        Returns:
            output: [num_nodes, output_dim]
        """
        # 聚合邻居特征
        agg_features = self.aggregate_neighbors(neighbor_features)
        
        # 拼接或相加
        if self.concat:
            combined = np.concatenate([node_features, agg_features], axis=1)
        else:
            combined = node_features + agg_features
        
        # 线性变换
        output = np.dot(combined, self.weight) + self.bias
        
        # L2归一化
        output = output / (np.linalg.norm(output, axis=1, keepdims=True) + 1e-8)
        
        return output


class GraphSAGE:
    """
    完整的GraphSAGE模型 (NumPy版本)
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, aggregator: str = 'mean'):
        self.layers = []
        
        # 第一层
        self.layers.append(GraphSAGELayer(input_dim, hidden_dim, aggregator))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator))
        
        # 输出层
        if num_layers > 1:
            self.layers.append(GraphSAGELayer(hidden_dim, output_dim, aggregator))
    
    def forward(self, node_features: np.ndarray, 
                adjacency_list: Dict[int, List[int]],
                num_samples: int = 10) -> np.ndarray:
        """
        前向传播
        
        Args:
            node_features: [num_nodes, feature_dim]
            adjacency_list: 邻接表 {node_id: [neighbor_ids]}
            num_samples: 邻居采样数量
        
        Returns:
            embeddings: [num_nodes, output_dim]
        """
        h = node_features.copy()
        num_nodes = node_features.shape[0]
        
        for layer in self.layers:
            # 采样邻居并聚合
            neighbor_features_list = []
            
            for node_id in range(num_nodes):
                neighbors = adjacency_list.get(node_id, [])
                
                # 采样邻居
                if len(neighbors) > num_samples:
                    sampled_neighbors = random.sample(neighbors, num_samples)
                else:
                    sampled_neighbors = neighbors + [node_id] * (num_samples - len(neighbors))
                
                # 获取邻居特征
                neighbor_feats = h[sampled_neighbors]
                neighbor_features_list.append(neighbor_feats)
            
            neighbor_features = np.stack(neighbor_features_list)
            
            # 通过GraphSAGE层
            h = layer.forward(h, neighbor_features)
            h = np.maximum(h, 0)  # ReLU
        
        return h


# =============================================================================
# 2. 知识图谱嵌入
# =============================================================================

class KnowledgeGraphEmbedding:
    """
    知识图谱嵌入模型 (NumPy版本)
    
    支持TransE和RotatE两种评分函数
    """
    
    def __init__(self, num_entities: int, num_relations: int, 
                 embedding_dim: int, model_type: str = 'rotate'):
        """
        Args:
            num_entities: 实体数量
            num_relations: 关系数量
            embedding_dim: 嵌入维度
            model_type: 'transe' 或 'rotate'
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        
        # 实体嵌入
        self.entity_embedding = xavier_uniform((num_entities, embedding_dim))
        
        # 关系嵌入
        self.relation_embedding = xavier_uniform((num_relations, embedding_dim))
        
        if model_type == 'rotate':
            # RotatE: 关系模长归一化
            self.relation_embedding = self.relation_embedding / (
                np.linalg.norm(self.relation_embedding, axis=1, keepdims=True) + 1e-8
            )
    
    def trans_e_score(self, heads: np.ndarray, relations: np.ndarray, 
                      tails: np.ndarray) -> np.ndarray:
        """
        TransE评分函数: f(h, r, t) = -||h + r - t||
        """
        h = self.entity_embedding[heads]
        r = self.relation_embedding[relations]
        t = self.entity_embedding[tails]
        
        score = -np.linalg.norm(h + r - t, axis=1)
        return score
    
    def rotate_score(self, heads: np.ndarray, relations: np.ndarray,
                     tails: np.ndarray) -> np.ndarray:
        """
        RotatE评分函数: f(h, r, t) = -||h ∘ r - t||
        其中 ∘ 表示复数乘法（旋转）
        """
        h = self.entity_embedding[heads]
        r = self.relation_embedding[relations]
        t = self.entity_embedding[tails]
        
        # 将嵌入分为实部和虚部
        half_dim = self.embedding_dim // 2
        h_re, h_im = h[:, :half_dim], h[:, half_dim:]
        t_re, t_im = t[:, :half_dim], t[:, half_dim:]
        
        # 关系作为相位（角度）
        r_phase = r[:, :half_dim]
        
        # 计算旋转
        r_cos = np.cos(r_phase)
        r_sin = np.sin(r_phase)
        
        # 复数乘法
        h_rotate_re = h_re * r_cos - h_im * r_sin
        h_rotate_im = h_re * r_sin + h_im * r_cos
        
        # 拼接
        h_rotate = np.concatenate([h_rotate_re, h_rotate_im], axis=1)
        
        # 计算距离
        score = -np.linalg.norm(h_rotate - t, axis=1)
        
        return score
    
    def forward(self, heads: np.ndarray, relations: np.ndarray,
                tails: np.ndarray) -> np.ndarray:
        """前向传播"""
        if self.model_type == 'transe':
            return self.trans_e_score(heads, relations, tails)
        elif self.model_type == 'rotate':
            return self.rotate_score(heads, relations, tails)
    
    def predict_tail(self, heads: np.ndarray, relations: np.ndarray,
                     top_k: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """
        预测尾实体
        
        Returns:
            scores: [batch_size, top_k]
            entities: [batch_size, top_k] 预测的实体索引
        """
        batch_size = heads.shape[0]
        
        all_scores = []
        for i in range(batch_size):
            if self.model_type == 'transe':
                target = self.entity_embedding[heads[i]] + self.relation_embedding[relations[i]]
                scores = -np.linalg.norm(target - self.entity_embedding, axis=1)
            else:  # rotate
                h = self.entity_embedding[heads[i]]
                r = self.relation_embedding[relations[i]]
                
                half_dim = self.embedding_dim // 2
                h_re, h_im = h[:half_dim], h[half_dim:]
                r_phase = r[:half_dim]
                
                r_cos = np.cos(r_phase)
                r_sin = np.sin(r_phase)
                
                h_rotate_re = h_re * r_cos - h_im * r_sin
                h_rotate_im = h_re * r_sin + h_im * r_cos
                h_rotate = np.concatenate([h_rotate_re, h_rotate_im])
                
                scores = -np.linalg.norm(h_rotate - self.entity_embedding, axis=1)
            
            all_scores.append(scores)
        
        all_scores = np.array(all_scores)
        
        # 获取top-k
        top_indices = np.argsort(-all_scores, axis=1)[:, :top_k]
        top_scores = np.take_along_axis(all_scores, top_indices, axis=1)
        
        return top_scores, top_indices


# =============================================================================
# 3. 图注意力机制
# =============================================================================

class GraphAttentionLayer:
    """
    图注意力层 (GAT) - NumPy版本
    
    论文: Graph Attention Networks (Veličković et al., 2018)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 num_heads: int = 1, alpha: float = 0.2):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            num_heads: 注意力头数
            alpha: LeakyReLU负斜率
        """
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        
        # 每个头的权重矩阵
        self.W = xavier_uniform((num_heads, in_features, out_features))
        
        # 注意力参数 a
        self.a = xavier_uniform((num_heads, 2 * out_features, 1))
    
    def compute_attention(self, Wh: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        计算注意力权重
        
        Args:
            Wh: [num_nodes, num_heads, out_features]
            adj: [num_nodes, num_nodes] 邻接矩阵
        
        Returns:
            attention: [num_nodes, num_nodes, num_heads]
        """
        num_nodes = Wh.shape[0]
        
        # 计算 e_ij
        attention_list = []
        for h in range(self.num_heads):
            Wh_h = Wh[:, h, :]  # [N, F]
            
            # 计算所有节点对的注意力分数
            e = np.zeros((num_nodes, num_nodes))
            for i in range(num_nodes):
                for j in range(num_nodes):
                    if adj[i, j] > 0:
                        concat = np.concatenate([Wh_h[i], Wh_h[j]])
                        # self.a[h] 形状是 (2*out_features, 1)，需要flatten
                        e[i, j] = leaky_relu(np.dot(concat, self.a[h].flatten()), self.alpha)
            
            attention_list.append(e)
        
        attention = np.stack(attention_list, axis=-1)  # [N, N, H]
        
        # Mask并Softmax
        for h in range(self.num_heads):
            mask = adj > 0
            masked = attention[:, :, h].copy()
            masked[~mask] = -1e9
            attention[:, :, h] = softmax(masked, axis=1)
        
        return attention
    
    def forward(self, h: np.ndarray, adj: np.ndarray) -> np.ndarray:
        """
        前向传播
        
        Args:
            h: [num_nodes, in_features]
            adj: [num_nodes, num_nodes]
        
        Returns:
            output: [num_nodes, out_features * num_heads]
        """
        num_nodes = h.shape[0]
        
        # 线性变换: Wh
        Wh = np.einsum('nf,hfo->nho', h, self.W)  # [N, H, F]
        
        # 计算注意力权重
        attention = self.compute_attention(Wh, adj)  # [N, N, H]
        
        # 聚合
        h_prime = np.einsum('ijh,jhf->ihf', attention, Wh)  # [N, H, F]
        
        # 拼接多头并应用ELU
        output = h_prime.reshape(num_nodes, -1)
        output = elu(output)
        
        return output


# =============================================================================
# 4. GNN推理器主类
# =============================================================================

class GNNReasoner:
    """
    GNN知识图谱推理器 (NumPy版本)
    """
    
    def __init__(self, num_entities: int, num_relations: int,
                 entity_features: Optional[np.ndarray] = None,
                 embedding_dim: int = 128, hidden_dim: int = 64,
                 model_type: str = 'rotate'):
        """
        Args:
            num_entities: 实体数量
            num_relations: 关系数量
            entity_features: 实体特征矩阵 [num_entities, feature_dim]
            embedding_dim: 嵌入维度
            hidden_dim: 隐藏层维度
            model_type: 知识图谱嵌入类型 ('transe' 或 'rotate')
        """
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        
        # 实体特征
        if entity_features is not None:
            self.entity_features = entity_features.astype(np.float32)
            self.feature_dim = entity_features.shape[1]
        else:
            self.entity_features = np.random.randn(num_entities, embedding_dim).astype(np.float32)
            self.feature_dim = embedding_dim
        
        # 知识图谱嵌入模型
        self.kg_embedding = KnowledgeGraphEmbedding(
            num_entities, num_relations, embedding_dim, model_type
        )
        
        # GraphSAGE模型
        self.graphsage = GraphSAGE(
            self.feature_dim, hidden_dim, embedding_dim, num_layers=2
        )
        
        # 图注意力模型
        self.gat = GraphAttentionLayer(
            self.feature_dim, hidden_dim, num_heads=4
        )
        
        # 图存储
        self.triples: List[Tuple[int, int, int]] = []
        self.adjacency_list: Dict[int, List[int]] = defaultdict(list)
        self.adjacency_matrix: Optional[np.ndarray] = None
    
    def build_graph(self, triples: List[Tuple[int, int, int]]):
        """
        构建图结构
        
        Args:
            triples: 三元组列表 [(head, relation, tail), ...]
        """
        self.triples = triples
        self.adjacency_list = defaultdict(list)
        
        # 构建邻接表
        for h, r, t in triples:
            self.adjacency_list[h].append(t)
            self.adjacency_list[t].append(h)
        
        # 构建邻接矩阵
        adj = np.zeros((self.num_entities, self.num_entities))
        for h, r, t in triples:
            adj[h, t] = 1
            adj[t, h] = 1
        
        self.adjacency_matrix = adj
    
    def get_graph_embeddings(self, method: str = 'graphsage') -> np.ndarray:
        """
        获取图嵌入
        
        Args:
            method: 'graphsage' 或 'gat'
        
        Returns:
            embeddings: [num_entities, embedding_dim]
        """
        if method == 'graphsage':
            return self.graphsage.forward(
                self.entity_features, self.adjacency_list, num_samples=10
            )
        elif method == 'gat':
            return self.gat.forward(self.entity_features, self.adjacency_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def train_kg_embedding(self, epochs: int = 100, batch_size: int = 256,
                          lr: float = 0.001, margin: float = 6.0):
        """
        训练知识图谱嵌入 (使用简单的SGD)
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            margin: margin-based损失的margin值
        """
        for epoch in range(epochs):
            total_loss = 0
            
            # 随机采样批次
            batch_triples = random.sample(self.triples, 
                                          min(batch_size, len(self.triples)))
            
            heads = np.array([t[0] for t in batch_triples])
            relations = np.array([t[1] for t in batch_triples])
            tails = np.array([t[2] for t in batch_triples])
            
            # 生成负样本
            neg_tails = np.random.randint(0, self.num_entities, size=tails.shape)
            
            # 正样本得分
            pos_score = self.kg_embedding.forward(heads, relations, tails)
            
            # 负样本得分
            neg_score = self.kg_embedding.forward(heads, relations, neg_tails)
            
            # Margin-based损失
            loss = np.maximum(0, margin + neg_score - pos_score)
            loss_value = np.mean(loss)
            
            # 简单的梯度下降更新（简化版）
            if loss_value > 0:
                # 计算梯度并更新（简化实现）
                grad_scale = lr * 0.01
                
                # 更新实体嵌入
                for i, (h, r, t, t_neg) in enumerate(zip(heads, relations, tails, neg_tails)):
                    if loss[i] > 0:
                        # 简化的梯度更新
                        self.kg_embedding.entity_embedding[t] += grad_scale * (
                            self.kg_embedding.entity_embedding[h] + 
                            self.kg_embedding.relation_embedding[r] - 
                            self.kg_embedding.entity_embedding[t]
                        )
                        self.kg_embedding.entity_embedding[t_neg] -= grad_scale * (
                            self.kg_embedding.entity_embedding[h] + 
                            self.kg_embedding.relation_embedding[r] - 
                            self.kg_embedding.entity_embedding[t_neg]
                        )
            
            total_loss += loss_value
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss:.4f}")
    
    def predict_relation(self, head: int, tail: int, 
                         top_k: int = 5) -> List[Tuple[int, float]]:
        """
        预测两个实体之间的关系
        
        Args:
            head: 头实体ID
            tail: 尾实体ID
            top_k: 返回前k个关系
        
        Returns:
            predictions: [(relation_id, score), ...]
        """
        head_array = np.array([head])
        tail_array = np.array([tail])
        
        scores = []
        for r in range(self.num_relations):
            relation_array = np.array([r])
            score = self.kg_embedding.forward(head_array, relation_array, tail_array)
            scores.append((r, score[0]))
        
        # 按分数排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        return scores[:top_k]
    
    def predict_tail_entity(self, head: int, relation: int,
                            top_k: int = 10) -> List[Tuple[int, float]]:
        """
        给定头实体和关系，预测尾实体
        
        Args:
            head: 头实体ID
            relation: 关系ID
            top_k: 返回前k个实体
        
        Returns:
            predictions: [(entity_id, score), ...]
        """
        head_array = np.array([head])
        relation_array = np.array([relation])
        
        scores, entities = self.kg_embedding.predict_tail(
            head_array, relation_array, top_k
        )
        
        predictions = [(entities[0, i], scores[0, i]) for i in range(top_k)]
        
        return predictions
    
    def evaluate_link_prediction(self, test_triples: List[Tuple[int, int, int]],
                                  k: int = 10) -> Dict[str, float]:
        """
        评估链接预测性能
        
        Args:
            test_triples: 测试三元组
            k: Hit@k的k值
        
        Returns:
            metrics: {'MR': ..., 'MRR': ..., 'Hits@k': ...}
        """
        ranks = []
        
        for h, r, t in test_triples:
            head_array = np.array([h])
            relation_array = np.array([r])
            
            scores, entities = self.kg_embedding.predict_tail(
                head_array, relation_array, self.num_entities
            )
            
            # 找到真实尾实体的排名
            entity_list = entities[0].tolist()
            if t in entity_list:
                rank = entity_list.index(t) + 1
            else:
                rank = self.num_entities
            
            ranks.append(rank)
        
        ranks = np.array(ranks)
        
        metrics = {
            'MR': np.mean(ranks),
            'MRR': np.mean(1.0 / ranks),
            f'Hits@{k}': np.mean(ranks <= k)
        }
        
        return metrics


# =============================================================================
# 5. 示例用法
# =============================================================================

def create_sample_knowledge_graph():
    """
    创建一个示例知识图谱
    
    实体:
    0: 中国
    1: 北京
    2: 上海
    3: 美国
    4: 华盛顿
    5: 纽约
    6: 日本
    7: 东京
    8: 法国
    9: 巴黎
    
    关系:
    0: 首都
    1: 位于
    2: 邻国
    """
    
    num_entities = 10
    num_relations = 3
    
    # 三元组 (head, relation, tail)
    triples = [
        # 首都关系
        (0, 0, 1),  # 中国 -首都-> 北京
        (3, 0, 4),  # 美国 -首都-> 华盛顿
        (6, 0, 7),  # 日本 -首都-> 东京
        (8, 0, 9),  # 法国 -首都-> 巴黎
        
        # 位于关系
        (1, 1, 0),  # 北京 -位于-> 中国
        (2, 1, 0),  # 上海 -位于-> 中国
        (4, 1, 3),  # 华盛顿 -位于-> 美国
        (5, 1, 3),  # 纽约 -位于-> 美国
        (7, 1, 6),  # 东京 -位于-> 日本
        (9, 1, 8),  # 巴黎 -位于-> 法国
    ]
    
    return num_entities, num_relations, triples


def demo():
    """演示GNN推理器的使用"""
    
    print("=" * 60)
    print("GNN知识图谱推理器演示 (NumPy版本)")
    print("=" * 60)
    
    # 创建示例知识图谱
    num_entities, num_relations, triples = create_sample_knowledge_graph()
    
    print(f"\n实体数量: {num_entities}")
    print(f"关系数量: {num_relations}")
    print(f"三元组数量: {len(triples)}")
    
    # 初始化推理器
    reasoner = GNNReasoner(
        num_entities=num_entities,
        num_relations=num_relations,
        embedding_dim=64,
        hidden_dim=32,
        model_type='rotate'
    )
    
    # 构建图
    reasoner.build_graph(triples)
    print("\n图结构已构建")
    
    # 训练知识图谱嵌入
    print("\n训练知识图谱嵌入...")
    reasoner.train_kg_embedding(epochs=100, batch_size=8, lr=0.01)
    
    # 关系预测示例
    print("\n" + "=" * 60)
    print("关系预测示例")
    print("=" * 60)
    
    entity_names = {
        0: '中国', 1: '北京', 2: '上海', 3: '美国', 4: '华盛顿',
        5: '纽约', 6: '日本', 7: '东京', 8: '法国', 9: '巴黎'
    }
    relation_names = {0: '首都', 1: '位于', 2: '邻国'}
    
    # 预测: 中国 -> ? -> 北京
    predictions = reasoner.predict_relation(head=0, tail=1, top_k=3)
    print(f"\n中国 -> ? -> 北京:")
    for rel_id, score in predictions:
        print(f"  {relation_names.get(rel_id, f'关系{rel_id}')}: {score:.4f}")
    
    # 预测: 美国 -> ? -> 华盛顿
    predictions = reasoner.predict_relation(head=3, tail=4, top_k=3)
    print(f"\n美国 -> ? -> 华盛顿:")
    for rel_id, score in predictions:
        print(f"  {relation_names.get(rel_id, f'关系{rel_id}')}: {score:.4f}")
    
    # 预测: 东京 -> ? -> 日本
    predictions = reasoner.predict_relation(head=7, tail=6, top_k=3)
    print(f"\n东京 -> ? -> 日本:")
    for rel_id, score in predictions:
        print(f"  {relation_names.get(rel_id, f'关系{rel_id}')}: {score:.4f}")
    
    # 尾实体预测示例
    print("\n" + "=" * 60)
    print("尾实体预测示例")
    print("=" * 60)
    
    # 预测: 中国 -首都-> ?
    predictions = reasoner.predict_tail_entity(head=0, relation=0, top_k=5)
    print(f"\n中国 -首都-> ?:")
    for entity_id, score in predictions:
        print(f"  {entity_names.get(entity_id, f'实体{entity_id}')}: {score:.4f}")
    
    # 预测: 华盛顿 -位于-> ?
    predictions = reasoner.predict_tail_entity(head=4, relation=1, top_k=5)
    print(f"\n华盛顿 -位于-> ?:")
    for entity_id, score in predictions:
        print(f"  {entity_names.get(entity_id, f'实体{entity_id}')}: {score:.4f}")
    
    # 预测: 日本 -首都-> ?
    predictions = reasoner.predict_tail_entity(head=6, relation=0, top_k=5)
    print(f"\n日本 -首都-> ?:")
    for entity_id, score in predictions:
        print(f"  {entity_names.get(entity_id, f'实体{entity_id}')}: {score:.4f}")
    
    # 评估链接预测
    print("\n" + "=" * 60)
    print("链接预测评估")
    print("=" * 60)
    
    test_triples = [
        (0, 0, 1),  # 中国 -首都-> 北京
        (3, 0, 4),  # 美国 -首都-> 华盛顿
        (6, 0, 7),  # 日本 -首都-> 东京
    ]
    
    metrics = reasoner.evaluate_link_prediction(test_triples, k=5)
    print(f"\n评估指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    # 图嵌入示例
    print("\n" + "=" * 60)
    print("图嵌入示例")
    print("=" * 60)
    
    print("\n使用GraphSAGE生成图嵌入...")
    embeddings = reasoner.get_graph_embeddings(method='graphsage')
    print(f"嵌入形状: {embeddings.shape}")
    print(f"实体0 (中国) 的嵌入前5维: {embeddings[0, :5]}")
    print(f"实体1 (北京) 的嵌入前5维: {embeddings[1, :5]}")
    
    print("\n使用GAT生成图嵌入...")
    gat_embeddings = reasoner.get_graph_embeddings(method='gat')
    print(f"嵌入形状: {gat_embeddings.shape}")
    print(f"实体0 (中国) 的嵌入前5维: {gat_embeddings[0, :5]}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 设置随机种子
    np.random.seed(42)
    random.seed(42)
    
    # 运行演示
    demo()
