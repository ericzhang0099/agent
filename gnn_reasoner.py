"""
GNN Reasoner - 图神经网络知识图谱推理器

本模块实现了基于图神经网络的知识图谱推理系统，包含：
1. GraphSAGE简化版 - 归纳式图神经网络
2. 知识图谱嵌入 - TransE/RotatE风格的关系学习
3. 图注意力机制 - 多头注意力层
4. 关系预测 - 完整的推理流程

作者: AI Research Agent
日期: 2026-02-27
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Tuple, Dict, Optional, Set
from collections import defaultdict
import random


# =============================================================================
# 1. GraphSAGE 简化版实现
# =============================================================================

class GraphSAGELayer(nn.Module):
    """
    GraphSAGE层实现
    
    论文: Inductive Representation Learning on Large Graphs (Hamilton et al., 2017)
    
    核心思想:
    - 采样邻居节点
    - 聚合邻居特征
    - 更新节点表示
    """
    
    def __init__(self, input_dim: int, output_dim: int, 
                 aggregator: str = 'mean', concat: bool = True):
        """
        Args:
            input_dim: 输入特征维度
            output_dim: 输出特征维度
            aggregator: 聚合函数 ('mean', 'max', 'lstm')
            concat: 是否拼接自身特征和邻居聚合特征
        """
        super(GraphSAGELayer, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.aggregator = aggregator
        self.concat = concat
        
        # 聚合后的维度
        agg_dim = output_dim if concat else input_dim
        
        # 可学习参数
        if concat:
            self.weight = nn.Parameter(torch.FloatTensor(2 * agg_dim, output_dim))
        else:
            self.weight = nn.Parameter(torch.FloatTensor(agg_dim, output_dim))
        
        # LSTM聚合器（可选）
        if aggregator == 'lstm':
            self.lstm = nn.LSTM(input_dim, input_dim, batch_first=True)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.weight)
    
    def aggregate_neighbors(self, neighbor_features: torch.Tensor) -> torch.Tensor:
        """
        聚合邻居特征
        
        Args:
            neighbor_features: [num_nodes, num_neighbors, feature_dim]
        
        Returns:
            aggregated: [num_nodes, feature_dim]
        """
        if self.aggregator == 'mean':
            # Mean aggregator
            return torch.mean(neighbor_features, dim=1)
        
        elif self.aggregator == 'max':
            # Max pooling aggregator
            return torch.max(neighbor_features, dim=1)[0]
        
        elif self.aggregator == 'lstm':
            # LSTM aggregator
            batch_size, num_neighbors, feature_dim = neighbor_features.shape
            neighbor_features = neighbor_features.view(-1, num_neighbors, feature_dim)
            lstm_out, (h_n, c_n) = self.lstm(neighbor_features)
            return h_n.squeeze(0)
        
        else:
            raise ValueError(f"Unknown aggregator: {self.aggregator}")
    
    def forward(self, node_features: torch.Tensor, 
                neighbor_features: torch.Tensor) -> torch.Tensor:
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
            combined = torch.cat([node_features, agg_features], dim=1)
        else:
            combined = node_features + agg_features
        
        # 线性变换
        output = torch.matmul(combined, self.weight)
        output = F.normalize(output, p=2, dim=1)
        
        return output


class GraphSAGE(nn.Module):
    """
    完整的GraphSAGE模型
    """
    
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_layers: int = 2, aggregator: str = 'mean'):
        super(GraphSAGE, self).__init__()
        
        self.layers = nn.ModuleList()
        
        # 第一层
        self.layers.append(GraphSAGELayer(input_dim, hidden_dim, aggregator))
        
        # 隐藏层
        for _ in range(num_layers - 2):
            self.layers.append(GraphSAGELayer(hidden_dim, hidden_dim, aggregator))
        
        # 输出层
        self.layers.append(GraphSAGELayer(hidden_dim, output_dim, aggregator))
    
    def forward(self, node_features: torch.Tensor, 
                adjacency_list: Dict[int, List[int]],
                num_samples: int = 10) -> torch.Tensor:
        """
        前向传播
        
        Args:
            node_features: [num_nodes, feature_dim]
            adjacency_list: 邻接表 {node_id: [neighbor_ids]}
            num_samples: 邻居采样数量
        
        Returns:
            embeddings: [num_nodes, output_dim]
        """
        h = node_features
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
                    sampled_neighbors = neighbors
                    # 填充
                    while len(sampled_neighbors) < num_samples:
                        sampled_neighbors.append(node_id)  # 自环
                
                # 获取邻居特征
                neighbor_feats = h[sampled_neighbors]
                neighbor_features_list.append(neighbor_feats.unsqueeze(0))
            
            neighbor_features = torch.cat(neighbor_features_list, dim=0)
            
            # 通过GraphSAGE层
            h = layer(h, neighbor_features)
            h = F.relu(h)
        
        return h


# =============================================================================
# 2. 知识图谱嵌入
# =============================================================================

class KnowledgeGraphEmbedding(nn.Module):
    """
    知识图谱嵌入模型
    
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
        super(KnowledgeGraphEmbedding, self).__init__()
        
        self.num_entities = num_entities
        self.num_relations = num_relations
        self.embedding_dim = embedding_dim
        self.model_type = model_type
        
        # 实体嵌入
        self.entity_embedding = nn.Embedding(num_entities, embedding_dim)
        
        if model_type == 'transe':
            # 关系嵌入（TransE）
            self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
            
        elif model_type == 'rotate':
            # 关系嵌入（RotatE）- 复数空间
            # 分为实部和虚部
            self.relation_embedding = nn.Embedding(num_relations, embedding_dim)
            
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.entity_embedding.weight)
        nn.init.xavier_uniform_(self.relation_embedding.weight)
        
        if self.model_type == 'rotate':
            # RotatE: 关系模长为1
            with torch.no_grad():
                self.relation_embedding.weight.div_(
                    self.relation_embedding.weight.norm(dim=1, keepdim=True)
                )
    
    def trans_e_score(self, heads: torch.Tensor, relations: torch.Tensor, 
                      tails: torch.Tensor) -> torch.Tensor:
        """
        TransE评分函数
        
        公式: f(h, r, t) = -||h + r - t||
        
        Args:
            heads: [batch_size] 头实体索引
            relations: [batch_size] 关系索引
            tails: [batch_size] 尾实体索引
        
        Returns:
            scores: [batch_size] 评分（越高越好）
        """
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        t = self.entity_embedding(tails)
        
        # 计算 L2 距离
        score = -torch.norm(h + r - t, p=2, dim=1)
        
        return score
    
    def rotate_score(self, heads: torch.Tensor, relations: torch.Tensor,
                     tails: torch.Tensor) -> torch.Tensor:
        """
        RotatE评分函数
        
        公式: f(h, r, t) = -||h ∘ r - t||
        其中 ∘ 表示复数乘法（旋转）
        
        Args:
            heads: [batch_size] 头实体索引
            relations: [batch_size] 关系索引
            tails: [batch_size] 尾实体索引
        
        Returns:
            scores: [batch_size] 评分
        """
        # 获取嵌入
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        t = self.entity_embedding(tails)
        
        # 将嵌入分为实部和虚部
        h_re, h_im = torch.chunk(h, 2, dim=-1)
        t_re, t_im = torch.chunk(t, 2, dim=-1)
        
        # 关系作为相位（角度）
        r_phase = r / (self.embedding_dim // 2)
        
        # 计算旋转
        r_cos = torch.cos(r_phase)
        r_sin = torch.sin(r_phase)
        
        # 复数乘法: h * r = (h_re + i*h_im) * (r_cos + i*r_sin)
        h_rotate_re = h_re * r_cos - h_im * r_sin
        h_rotate_im = h_re * r_sin + h_im * r_cos
        
        # 拼接
        h_rotate = torch.cat([h_rotate_re, h_rotate_im], dim=-1)
        
        # 计算距离
        score = -torch.norm(h_rotate - t, p=2, dim=1)
        
        return score
    
    def forward(self, heads: torch.Tensor, relations: torch.Tensor,
                tails: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        if self.model_type == 'transe':
            return self.trans_e_score(heads, relations, tails)
        elif self.model_type == 'rotate':
            return self.rotate_score(heads, relations, tails)
    
    def predict_tail(self, heads: torch.Tensor, relations: torch.Tensor,
                     top_k: int = 10) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        预测尾实体
        
        Args:
            heads: [batch_size] 头实体索引
            relations: [batch_size] 关系索引
            top_k: 返回前k个预测
        
        Returns:
            scores: [batch_size, top_k]
            entities: [batch_size, top_k] 预测的实体索引
        """
        batch_size = heads.shape[0]
        
        # 获取头实体和关系的嵌入
        h = self.entity_embedding(heads)
        r = self.relation_embedding(relations)
        
        # 计算所有实体作为尾实体的得分
        all_entities = torch.arange(self.num_entities, device=heads.device)
        all_t = self.entity_embedding(all_entities)
        
        scores = []
        for i in range(batch_size):
            if self.model_type == 'transe':
                # h + r ≈ t
                target = h[i] + r[i]
                score = -torch.norm(target.unsqueeze(0) - all_t, p=2, dim=1)
            else:  # rotate
                h_re, h_im = torch.chunk(h[i:i+1], 2, dim=-1)
                r_phase = r[i:i+1] / (self.embedding_dim // 2)
                r_cos = torch.cos(r_phase)
                r_sin = torch.sin(r_phase)
                h_rotate_re = h_re * r_cos - h_im * r_sin
                h_rotate_im = h_re * r_sin + h_im * r_cos
                h_rotate = torch.cat([h_rotate_re, h_rotate_im], dim=-1)
                score = -torch.norm(h_rotate - all_t, p=2, dim=1)
            
            scores.append(score)
        
        scores = torch.stack(scores)
        
        # 获取top-k
        top_scores, top_indices = torch.topk(scores, top_k, dim=1)
        
        return top_scores, top_indices


# =============================================================================
# 3. 图注意力机制
# =============================================================================

class GraphAttentionLayer(nn.Module):
    """
    图注意力层 (GAT)
    
    论文: Graph Attention Networks (Veličković et al., 2018)
    """
    
    def __init__(self, in_features: int, out_features: int, 
                 num_heads: int = 1, dropout: float = 0.0,
                 alpha: float = 0.2, concat: bool = True):
        """
        Args:
            in_features: 输入特征维度
            out_features: 输出特征维度
            num_heads: 注意力头数
            dropout: dropout率
            alpha: LeakyReLU负斜率
            concat: 是否拼接多头结果
        """
        super(GraphAttentionLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.alpha = alpha
        self.concat = concat
        
        # 每个头的权重矩阵
        self.W = nn.Parameter(torch.FloatTensor(num_heads, in_features, out_features))
        
        # 注意力参数 a
        self.a = nn.Parameter(torch.FloatTensor(num_heads, 2 * out_features, 1))
        
        self.dropout = nn.Dropout(dropout)
        self.leakyrelu = nn.LeakyReLU(alpha)
        
        self.reset_parameters()
    
    def reset_parameters(self):
        """初始化参数"""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)
    
    def compute_attention(self, Wh: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        计算注意力权重
        
        Args:
            Wh: [num_nodes, num_heads, out_features]
            adj: [num_nodes, num_nodes] 邻接矩阵
        
        Returns:
            attention: [num_nodes, num_nodes, num_heads]
        """
        num_nodes = Wh.shape[0]
        
        # 计算 e_ij = LeakyReLU(a^T [Wh_i || Wh_j])
        # 扩展维度以便广播
        Wh_i = Wh.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [N, N, H, F]
        Wh_j = Wh.unsqueeze(0).expand(num_nodes, -1, -1, -1)  # [N, N, H, F]
        
        # 拼接
        concat = torch.cat([Wh_i, Wh_j], dim=-1)  # [N, N, H, 2F]
        
        # 计算注意力分数
        e = torch.einsum('ijhd,hdf->ijhf', concat, self.a).squeeze(-1)  # [N, N, H]
        e = self.leakyrelu(e)
        
        # Mask: 只保留邻接矩阵中有边的位置
        mask = adj.unsqueeze(-1).expand(-1, -1, self.num_heads)
        e = e.masked_fill(mask == 0, float('-inf'))
        
        # Softmax归一化
        attention = F.softmax(e, dim=1)
        attention = self.dropout(attention)
        
        return attention
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: [num_nodes, in_features]
            adj: [num_nodes, num_nodes]
        
        Returns:
            output: [num_nodes, out_features * num_heads] 或 [num_nodes, out_features]
        """
        num_nodes = h.shape[0]
        
        # 线性变换: Wh
        Wh = torch.einsum('nf,hfo->nho', h, self.W)  # [N, H, F]
        
        # 计算注意力权重
        attention = self.compute_attention(Wh, adj)  # [N, N, H]
        
        # 聚合: h' = sum_j (alpha_ij * Wh_j)
        h_prime = torch.einsum('ijh,jhf->ihf', attention, Wh)  # [N, H, F]
        
        if self.concat:
            # 拼接多头
            output = h_prime.view(num_nodes, -1)
            return F.elu(output)
        else:
            # 平均多头
            output = h_prime.mean(dim=1)
            return output


class MultiHeadGAT(nn.Module):
    """
    多头图注意力网络
    """
    
    def __init__(self, in_features: int, hidden_features: int, 
                 out_features: int, num_heads: int = 8, dropout: float = 0.6):
        super(MultiHeadGAT, self).__init__()
        
        # 第一层
        self.attention1 = GraphAttentionLayer(
            in_features, hidden_features, num_heads=num_heads, 
            dropout=dropout, concat=True
        )
        
        # 第二层（输出层）
        self.attention2 = GraphAttentionLayer(
            hidden_features * num_heads, out_features, num_heads=1,
            dropout=dropout, concat=False
        )
    
    def forward(self, h: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            h: [num_nodes, in_features]
            adj: [num_nodes, num_nodes]
        
        Returns:
            output: [num_nodes, out_features]
        """
        h = self.attention1(h, adj)
        h = F.dropout(h, p=0.6, training=self.training)
        h = self.attention2(h, adj)
        return h


# =============================================================================
# 4. GNN推理器主类
# =============================================================================

class GNNReasoner:
    """
    GNN知识图谱推理器
    
    整合GraphSAGE、知识图谱嵌入和图注意力机制，
    提供完整的知识图谱推理功能。
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
            self.entity_features = torch.FloatTensor(entity_features)
            self.feature_dim = entity_features.shape[1]
        else:
            # 随机初始化特征
            self.entity_features = torch.randn(num_entities, embedding_dim)
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
        self.gat = MultiHeadGAT(
            self.feature_dim, hidden_dim, embedding_dim, num_heads=4
        )
        
        # 图存储
        self.triples: List[Tuple[int, int, int]] = []
        self.adjacency_list: Dict[int, List[int]] = defaultdict(list)
        self.adjacency_matrix: Optional[torch.Tensor] = None
    
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
            self.adjacency_list[t].append(h)  # 无向图
        
        # 构建邻接矩阵
        adj = torch.zeros(self.num_entities, self.num_entities)
        for h, r, t in triples:
            adj[h, t] = 1
            adj[t, h] = 1
        
        self.adjacency_matrix = adj
    
    def get_graph_embeddings(self, method: str = 'graphsage') -> torch.Tensor:
        """
        获取图嵌入
        
        Args:
            method: 'graphsage' 或 'gat'
        
        Returns:
            embeddings: [num_entities, embedding_dim]
        """
        if method == 'graphsage':
            return self.graphsage(
                self.entity_features, self.adjacency_list, num_samples=10
            )
        elif method == 'gat':
            return self.gat(self.entity_features, self.adjacency_matrix)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def train_kg_embedding(self, epochs: int = 100, batch_size: int = 256,
                          lr: float = 0.001, margin: float = 6.0):
        """
        训练知识图谱嵌入
        
        Args:
            epochs: 训练轮数
            batch_size: 批次大小
            lr: 学习率
            margin: margin-based损失的margin值
        """
        optimizer = torch.optim.Adam(self.kg_embedding.parameters(), lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0
            
            # 随机采样批次
            batch_triples = random.sample(self.triples, 
                                          min(batch_size, len(self.triples)))
            
            heads = torch.LongTensor([t[0] for t in batch_triples])
            relations = torch.LongTensor([t[1] for t in batch_triples])
            tails = torch.LongTensor([t[2] for t in batch_triples])
            
            # 生成负样本（替换尾实体）
            neg_tails = torch.randint(0, self.num_entities, tails.shape)
            
            # 正样本得分
            pos_score = self.kg_embedding(heads, relations, tails)
            
            # 负样本得分
            neg_score = self.kg_embedding(heads, relations, neg_tails)
            
            # Margin-based损失
            loss = F.relu(margin + neg_score - pos_score).mean()
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
            if (epoch + 1) % 10 == 0:
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
        head_tensor = torch.LongTensor([head])
        tail_tensor = torch.LongTensor([tail])
        
        scores = []
        for r in range(self.num_relations):
            relation_tensor = torch.LongTensor([r])
            score = self.kg_embedding(head_tensor, relation_tensor, tail_tensor)
            scores.append((r, score.item()))
        
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
        head_tensor = torch.LongTensor([head])
        relation_tensor = torch.LongTensor([relation])
        
        scores, entities = self.kg_embedding.predict_tail(
            head_tensor, relation_tensor, top_k
        )
        
        predictions = [(entities[0, i].item(), scores[0, i].item()) 
                      for i in range(top_k)]
        
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
            # 计算所有实体作为尾实体的得分
            head_tensor = torch.LongTensor([h])
            relation_tensor = torch.LongTensor([r])
            
            scores, entities = self.kg_embedding.predict_tail(
                head_tensor, relation_tensor, self.num_entities
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
    print("GNN知识图谱推理器演示")
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
    
    # 预测: 中国 -> ? -> 北京
    predictions = reasoner.predict_relation(head=0, tail=1, top_k=3)
    print(f"\n中国 -> ? -> 北京:")
    relation_names = {0: '首都', 1: '位于', 2: '邻国'}
    for rel_id, score in predictions:
        print(f"  {relation_names.get(rel_id, f'关系{rel_id}')}: {score:.4f}")
    
    # 预测: 美国 -> ? -> 华盛顿
    predictions = reasoner.predict_relation(head=3, tail=4, top_k=3)
    print(f"\n美国 -> ? -> 华盛顿:")
    for rel_id, score in predictions:
        print(f"  {relation_names.get(rel_id, f'关系{rel_id}')}: {score:.4f}")
    
    # 尾实体预测示例
    print("\n" + "=" * 60)
    print("尾实体预测示例")
    print("=" * 60)
    
    entity_names = {
        0: '中国', 1: '北京', 2: '上海', 3: '美国', 4: '华盛顿',
        5: '纽约', 6: '日本', 7: '东京', 8: '法国', 9: '巴黎'
    }
    
    # 预测: 中国 -首都-> ?
    predictions = reasoner.predict_tail_entity(head=0, relation=0, top_k=5)
    print(f"\n中国 -首都-> ?:")
    for entity_id, score in predictions:
        print(f"  {entity_names.get(entity_id, f'实体{entity_id}')}: {score:.4f}")
    
    # 预测: 美国 -位于-> ?
    predictions = reasoner.predict_tail_entity(head=4, relation=1, top_k=5)
    print(f"\n华盛顿 -位于-> ?:")
    for entity_id, score in predictions:
        print(f"  {entity_names.get(entity_id, f'实体{entity_id}')}: {score:.4f}")
    
    # 评估链接预测
    print("\n" + "=" * 60)
    print("链接预测评估")
    print("=" * 60)
    
    test_triples = [
        (0, 0, 1),  # 中国 -首都-> 北京
        (3, 0, 4),  # 美国 -首都-> 华盛顿
    ]
    
    metrics = reasoner.evaluate_link_prediction(test_triples, k=5)
    print(f"\n评估指标:")
    for metric, value in metrics.items():
        print(f"  {metric}: {value:.4f}")
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    # 设置随机种子
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
    
    # 运行演示
    demo()
