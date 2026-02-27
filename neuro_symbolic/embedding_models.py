"""
知识图谱嵌入模型
实现TransE、RotatE等经典知识图谱表示学习模型
"""

import numpy as np
from typing import List, Set, Tuple, Dict, Optional
from collections import defaultdict
import random
from knowledge_graph import KnowledgeGraph, Triple


class EmbeddingModel:
    """知识图谱嵌入模型基类"""
    
    def __init__(self, kg: KnowledgeGraph, embedding_dim: int = 100):
        self.kg = kg
        self.embedding_dim = embedding_dim
        
        # 实体和关系的嵌入
        self.entity_embeddings: Dict[str, np.ndarray] = {}
        self.relation_embeddings: Dict[str, np.ndarray] = {}
        
        # 实体和关系ID映射
        self.entity2id: Dict[str, int] = {}
        self.id2entity: Dict[int, str] = {}
        self.relation2id: Dict[str, int] = {}
        self.id2relation: Dict[int, str] = {}
        
        self._build_mappings()
    
    def _build_mappings(self):
        """构建ID映射"""
        for i, entity_id in enumerate(sorted(self.kg.entities.keys())):
            self.entity2id[entity_id] = i
            self.id2entity[i] = entity_id
        
        for i, rel_type in enumerate(sorted(self.kg.relation_types.keys())):
            self.relation2id[rel_type] = i
            self.id2relation[i] = rel_type
    
    def _initialize_embeddings(self):
        """初始化嵌入向量"""
        raise NotImplementedError
    
    def score(self, head: str, relation: str, tail: str) -> float:
        """计算三元组得分（越低越好）"""
        raise NotImplementedError
    
    def train(self, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01):
        """训练模型"""
        raise NotImplementedError
    
    def get_entity_embedding(self, entity_id: str) -> Optional[np.ndarray]:
        """获取实体嵌入"""
        return self.entity_embeddings.get(entity_id)
    
    def get_relation_embedding(self, relation_type: str) -> Optional[np.ndarray]:
        """获取关系嵌入"""
        return self.relation_embeddings.get(relation_type)


class TransE(EmbeddingModel):
    """
    TransE: Translating Embeddings for Modeling Multi-relational Data
    
    核心思想: h + r ≈ t
    得分函数: ||h + r - t||
    """
    
    def __init__(self, kg: KnowledgeGraph, embedding_dim: int = 100, norm: int = 1):
        super().__init__(kg, embedding_dim)
        self.norm = norm  # L1 or L2 norm
        self.margin = 1.0
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """使用Xavier初始化"""
        bound = 6 / np.sqrt(self.embedding_dim)
        
        for entity_id in self.kg.entities:
            self.entity_embeddings[entity_id] = np.random.uniform(
                -bound, bound, self.embedding_dim
            )
        
        for rel_type in self.kg.relation_types:
            self.relation_embeddings[rel_type] = np.random.uniform(
                -bound, bound, self.embedding_dim
            )
            # 归一化关系向量
            self.relation_embeddings[rel_type] = self._normalize(
                self.relation_embeddings[rel_type]
            )
    
    def _normalize(self, vec: np.ndarray) -> np.ndarray:
        """归一化向量"""
        return vec / (np.linalg.norm(vec) + 1e-8)
    
    def score(self, head: str, relation: str, tail: str) -> float:
        """计算三元组得分"""
        h = self.entity_embeddings[head]
        r = self.relation_embeddings[relation]
        t = self.entity_embeddings[tail]
        
        score_vec = h + r - t
        
        if self.norm == 1:
            return np.sum(np.abs(score_vec))
        else:
            return np.sum(score_vec ** 2)
    
    def _negative_sampling(self, triple: Triple) -> Triple:
        """负采样：随机替换头或尾实体"""
        entity_ids = list(self.kg.entities.keys())
        
        if random.random() < 0.5:
            # 替换头实体
            new_head = random.choice(entity_ids)
            return Triple(new_head, triple.relation, triple.tail)
        else:
            # 替换尾实体
            new_tail = random.choice(entity_ids)
            return Triple(triple.head, triple.relation, new_tail)
    
    def train(self, epochs: int = 100, batch_size: int = 32, 
              learning_rate: float = 0.01, negative_samples: int = 1):
        """训练TransE模型"""
        
        triples = list(self.kg.triples)
        num_triples = len(triples)
        
        for epoch in range(epochs):
            # 随机打乱
            random.shuffle(triples)
            
            total_loss = 0
            
            for i in range(0, num_triples, batch_size):
                batch = triples[i:i + batch_size]
                
                for triple in batch:
                    # 正样本
                    h_pos = self.entity_embeddings[triple.head]
                    r = self.relation_embeddings[triple.relation]
                    t_pos = self.entity_embeddings[triple.tail]
                    
                    # 负采样
                    for _ in range(negative_samples):
                        neg_triple = self._negative_sampling(triple)
                        h_neg = self.entity_embeddings[neg_triple.head]
                        t_neg = self.entity_embeddings[neg_triple.tail]
                        
                        # 计算得分
                        pos_score = self._compute_score(h_pos, r, t_pos)
                        neg_score = self._compute_score(h_neg, r, t_neg)
                        
                        # 计算损失
                        loss = max(0, self.margin + pos_score - neg_score)
                        total_loss += loss
                        
                        if loss > 0:
                            # 梯度下降更新
                            grad_pos = self._gradient(h_pos, r, t_pos)
                            grad_neg = self._gradient(h_neg, r, t_neg)
                            
                            # 更新正样本
                            self.entity_embeddings[triple.head] -= learning_rate * grad_pos[0]
                            self.relation_embeddings[triple.relation] -= learning_rate * grad_pos[1]
                            self.entity_embeddings[triple.tail] -= learning_rate * grad_pos[2]
                            
                            # 更新负样本
                            self.entity_embeddings[neg_triple.head] += learning_rate * grad_neg[0]
                            self.entity_embeddings[neg_triple.tail] += learning_rate * grad_neg[2]
                    
                    # 归一化实体嵌入
                    self.entity_embeddings[triple.head] = self._normalize(
                        self.entity_embeddings[triple.head]
                    )
                    self.entity_embeddings[triple.tail] = self._normalize(
                        self.entity_embeddings[triple.tail]
                    )
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / (num_triples * negative_samples)
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _compute_score(self, h: np.ndarray, r: np.ndarray, t: np.ndarray) -> float:
        """计算得分"""
        score_vec = h + r - t
        if self.norm == 1:
            return np.sum(np.abs(score_vec))
        else:
            return np.sum(score_vec ** 2)
    
    def _gradient(self, h: np.ndarray, r: np.ndarray, t: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """计算梯度"""
        diff = h + r - t
        
        if self.norm == 1:
            grad = np.sign(diff)
        else:
            grad = 2 * diff
        
        grad_h = grad
        grad_r = grad
        grad_t = -grad
        
        return grad_h, grad_r, grad_t
    
    def predict_tail(self, head: str, relation: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """预测尾实体"""
        scores = []
        h = self.entity_embeddings[head]
        r = self.relation_embeddings[relation]
        
        for entity_id, emb in self.entity_embeddings.items():
            score = self._compute_score(h, r, emb)
            scores.append((entity_id, score))
        
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]
    
    def predict_head(self, tail: str, relation: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """预测头实体"""
        scores = []
        t = self.entity_embeddings[tail]
        r = self.relation_embeddings[relation]
        
        for entity_id, emb in self.entity_embeddings.items():
            score = self._compute_score(emb, r, t)
            scores.append((entity_id, score))
        
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]


class RotatE(EmbeddingModel):
    """
    RotatE: Knowledge Graph Embedding by Relational Rotation in Complex Space
    
    核心思想: 在复数空间中使用旋转操作建模关系
    t = h ∘ r (逐元素复数乘法)
    得分函数: ||h ∘ r - t||
    """
    
    def __init__(self, kg: KnowledgeGraph, embedding_dim: int = 100, 
                 gamma: float = 12.0):
        super().__init__(kg, embedding_dim // 2)  # 复数需要一半维度
        self.gamma = gamma
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """初始化复数嵌入"""
        bound = 1 / np.sqrt(self.embedding_dim)
        
        # 实体嵌入：复数形式 (实部, 虚部)
        for entity_id in self.kg.entities:
            real = np.random.uniform(-bound, bound, self.embedding_dim)
            imag = np.random.uniform(-bound, bound, self.embedding_dim)
            self.entity_embeddings[entity_id] = np.concatenate([real, imag])
        
        # 关系嵌入：相位角
        for rel_type in self.kg.relation_types:
            phase = np.random.uniform(-np.pi, np.pi, self.embedding_dim)
            self.relation_embeddings[rel_type] = phase
    
    def _get_complex(self, embedding: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """将嵌入分为实部和虚部"""
        d = self.embedding_dim
        real = embedding[:d]
        imag = embedding[d:]
        return real, imag
    
    def _complex_multiply(self, h: np.ndarray, r_phase: np.ndarray) -> np.ndarray:
        """
        复数乘法: h * r
        h = a + bi, r = cos(θ) + i*sin(θ)
        h * r = (a*cos(θ) - b*sin(θ)) + i*(a*sin(θ) + b*cos(θ))
        """
        h_real, h_imag = self._get_complex(h)
        
        r_cos = np.cos(r_phase)
        r_sin = np.sin(r_phase)
        
        result_real = h_real * r_cos - h_imag * r_sin
        result_imag = h_real * r_sin + h_imag * r_cos
        
        return np.concatenate([result_real, result_imag])
    
    def score(self, head: str, relation: str, tail: str) -> float:
        """计算三元组得分"""
        h = self.entity_embeddings[head]
        r_phase = self.relation_embeddings[relation]
        t = self.entity_embeddings[tail]
        
        h_rotated = self._complex_multiply(h, r_phase)
        
        score_vec = h_rotated - t
        return np.sum(score_vec ** 2)
    
    def _negative_sampling(self, triple: Triple) -> Triple:
        """负采样"""
        entity_ids = list(self.kg.entities.keys())
        
        if random.random() < 0.5:
            new_head = random.choice(entity_ids)
            return Triple(new_head, triple.relation, triple.tail)
        else:
            new_tail = random.choice(entity_ids)
            return Triple(triple.head, triple.relation, new_tail)
    
    def train(self, epochs: int = 100, batch_size: int = 256,
              learning_rate: float = 0.0001, negative_samples: int = 256):
        """训练RotatE模型（使用自对抗负采样）"""
        
        triples = list(self.kg.triples)
        num_triples = len(triples)
        
        for epoch in range(epochs):
            random.shuffle(triples)
            total_loss = 0
            
            for i in range(0, num_triples, batch_size):
                batch = triples[i:i + batch_size]
                
                for triple in batch:
                    h = self.entity_embeddings[triple.head]
                    r = self.relation_embeddings[triple.relation]
                    t = self.entity_embeddings[triple.tail]
                    
                    # 正样本得分
                    pos_score = self.score(triple.head, triple.relation, triple.tail)
                    
                    # 负采样
                    neg_scores = []
                    neg_triples = []
                    
                    for _ in range(negative_samples):
                        neg_triple = self._negative_sampling(triple)
                        neg_score = self.score(neg_triple.head, neg_triple.relation, neg_triple.tail)
                        neg_scores.append(neg_score)
                        neg_triples.append(neg_triple)
                    
                    # 计算损失（使用soft margin）
                    pos_loss = -np.log(1 / (1 + np.exp(pos_score)))
                    neg_loss = sum(-np.log(1 / (1 + np.exp(-ns))) for ns in neg_scores)
                    
                    loss = pos_loss + neg_loss / negative_samples
                    total_loss += loss
                    
                    # 简化的梯度更新
                    if epoch < epochs - 10:  # 最后10轮不更新
                        self._update_gradients(triple, neg_triples, learning_rate)
            
            if (epoch + 1) % 10 == 0:
                avg_loss = total_loss / num_triples
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {avg_loss:.4f}")
    
    def _update_gradients(self, pos_triple: Triple, neg_triples: List[Triple], lr: float):
        """简化版梯度更新"""
        # 正样本
        h = self.entity_embeddings[pos_triple.head]
        r = self.relation_embeddings[pos_triple.relation]
        t = self.entity_embeddings[pos_triple.tail]
        
        h_rotated = self._complex_multiply(h, r)
        grad = 2 * (h_rotated - t)
        
        # 更新（简化处理）
        d = self.embedding_dim
        self.entity_embeddings[pos_triple.tail] -= lr * 0.01 * grad
        
        # 约束嵌入范数
        for entity_id in self.entity_embeddings:
            emb = self.entity_embeddings[entity_id]
            norm = np.linalg.norm(emb)
            if norm > 1.0:
                self.entity_embeddings[entity_id] = emb / norm
    
    def predict_tail(self, head: str, relation: str, top_k: int = 10) -> List[Tuple[str, float]]:
        """预测尾实体"""
        scores = []
        
        for entity_id in self.entity_embeddings:
            score = self.score(head, relation, entity_id)
            scores.append((entity_id, score))
        
        scores.sort(key=lambda x: x[1])
        return scores[:top_k]


class ComplEx(EmbeddingModel):
    """
    ComplEx: Complex Embeddings for Simple Link Prediction
    
    使用复数嵌入建模非对称关系
    得分函数: Re(<h, r, conj(t)>)
    """
    
    def __init__(self, kg: KnowledgeGraph, embedding_dim: int = 100):
        super().__init__(kg, embedding_dim // 2)
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """初始化复数嵌入"""
        bound = 1 / np.sqrt(self.embedding_dim)
        
        for entity_id in self.kg.entities:
            real = np.random.uniform(-bound, bound, self.embedding_dim)
            imag = np.random.uniform(-bound, bound, self.embedding_dim)
            self.entity_embeddings[entity_id] = np.concatenate([real, imag])
        
        for rel_type in self.kg.relation_types:
            real = np.random.uniform(-bound, bound, self.embedding_dim)
            imag = np.random.uniform(-bound, bound, self.embedding_dim)
            self.relation_embeddings[rel_type] = np.concatenate([real, imag])
    
    def score(self, head: str, relation: str, tail: str) -> float:
        """计算三元组得分"""
        h = self.entity_embeddings[head]
        r = self.relation_embeddings[relation]
        t = self.entity_embeddings[tail]
        
        d = self.embedding_dim
        
        h_re, h_im = h[:d], h[d:]
        r_re, r_im = r[:d], r[d:]
        t_re, t_im = t[:d], t[d:]
        
        # Re(h * r * conj(t))
        # = (h_re * r_re - h_im * r_im) * t_re + (h_re * r_im + h_im * r_re) * t_im
        score = np.sum(
            (h_re * r_re - h_im * r_im) * t_re +
            (h_re * r_im + h_im * r_re) * t_im
        )
        
        return -score  # 负号因为我们要最小化
    
    def train(self, epochs: int = 100, batch_size: int = 32, learning_rate: float = 0.01):
        """训练ComplEx模型"""
        # 简化的训练流程
        triples = list(self.kg.triples)
        
        for epoch in range(epochs):
            random.shuffle(triples)
            total_loss = 0
            
            for triple in triples:
                score = self.score(triple.head, triple.relation, triple.tail)
                # 逻辑损失
                loss = np.log(1 + np.exp(score))
                total_loss += loss
            
            if (epoch + 1) % 20 == 0:
                print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(triples):.4f}")


def evaluate_model(model: EmbeddingModel, test_triples: List[Triple], 
                   k: int = 10) -> Dict[str, float]:
    """
    评估模型性能
    
    指标：
    - MR (Mean Rank): 平均排名
    - MRR (Mean Reciprocal Rank): 平均倒数排名
    - Hits@K: 前K命中率
    """
    ranks = []
    hits = 0
    
    for triple in test_triples:
        # 尾实体预测
        predictions = model.predict_tail(triple.head, triple.relation, 
                                          top_k=len(model.entity_embeddings))
        
        # 找到真实尾实体的排名
        for rank, (entity_id, _) in enumerate(predictions, 1):
            if entity_id == triple.tail:
                ranks.append(rank)
                if rank <= k:
                    hits += 1
                break
    
    mr = np.mean(ranks) if ranks else 0
    mrr = np.mean([1 / r for r in ranks]) if ranks else 0
    hits_at_k = hits / len(test_triples) if test_triples else 0
    
    return {
        'MR': mr,
        'MRR': mrr,
        f'Hits@{k}': hits_at_k
    }


if __name__ == "__main__":
    from knowledge_graph import create_sample_kg
    
    # 创建示例知识图谱
    kg = create_sample_kg()
    print("Knowledge Graph:", kg)
    print()
    
    # 训练TransE
    print("=" * 50)
    print("Training TransE...")
    print("=" * 50)
    
    transe = TransE(kg, embedding_dim=50)
    transe.train(epochs=50, batch_size=4, learning_rate=0.01)
    
    # 测试预测
    print("\nTransE Predictions:")
    test_triples = list(kg.triples)[:3]
    for triple in test_triples:
        print(f"\nQuery: {triple.head} --{triple.relation}--> ?")
        predictions = transe.predict_tail(triple.head, triple.relation, top_k=3)
        for entity_id, score in predictions:
            entity = kg.get_entity(entity_id)
            print(f"  {entity.name} (score: {score:.4f})")
    
    # 训练RotatE
    print("\n" + "=" * 50)
    print("Training RotatE...")
    print("=" * 50)
    
    rotate = RotatE(kg, embedding_dim=50)
    rotate.train(epochs=30, batch_size=4, learning_rate=0.001)
    
    print("\nRotatE Predictions:")
    for triple in test_triples:
        print(f"\nQuery: {triple.head} --{triple.relation}--> ?")
        predictions = rotate.predict_tail(triple.head, triple.relation, top_k=3)
        for entity_id, score in predictions:
            entity = kg.get_entity(entity_id)
            print(f"  {entity.name} (score: {score:.4f})")
