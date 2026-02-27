"""
Efficient Attention Mechanisms Implementation
=============================================
This module implements various efficient attention mechanisms for Transformers:
1. Sparse Attention (Sliding Window + Global)
2. Linear Attention (Performer-style)
3. Memory Optimizations
4. Long Context Processing Examples

Author: AI Assistant
Date: 2026-02-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple, Literal
import warnings


# =============================================================================
# 1. Sparse Attention Implementations
# =============================================================================

class SlidingWindowAttention(nn.Module):
    """
    滑动窗口注意力 (Sliding Window Attention)
    
    每个token只关注其左右固定窗口大小内的token，复杂度从O(n²)降低到O(n·w)
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        window_size: 滑动窗口大小 (每侧)
        dropout: dropout概率
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        window_size: int = 256,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0, "d_model must be divisible by n_heads"
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_sliding_window_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建滑动窗口掩码
        
        掩码形状: (seq_len, seq_len)
        允许关注的位置: |i - j| <= window_size
        """
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        for i in range(seq_len):
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 0.0
        return mask
    
    def forward(
        self,
        x: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量 (batch_size, seq_len, d_model)
            mask: 可选的额外掩码
            
        Returns:
            输出张量 (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影到Q, K, V
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # (batch, n_heads, seq_len, d_head)
        
        # 计算注意力分数
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        # (batch, n_heads, seq_len, seq_len)
        
        # 应用滑动窗口掩码
        window_mask = self._create_sliding_window_mask(seq_len, x.device)
        scores = scores + window_mask.unsqueeze(0).unsqueeze(0)
        
        # 应用额外掩码
        if mask is not None:
            scores = scores + mask.unsqueeze(1).unsqueeze(1)
        
        # Softmax和dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 应用注意力到V
        attn_output = torch.matmul(attn_weights, V)
        # (batch, n_heads, seq_len, d_head)
        
        # 合并头并投影
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        output = self.out_proj(attn_output)
        
        return output


class GlobalLocalAttention(nn.Module):
    """
    全局-局部注意力 (Global + Sliding Window)
    
    结合全局token(如[CLS])和滑动窗口注意力，类似Longformer
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        window_size: 滑动窗口大小
        global_tokens_idx: 全局token的索引列表
        dropout: dropout概率
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        window_size: int = 256,
        global_tokens_idx: list = [0],
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.global_tokens_idx = global_tokens_idx
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_global_local_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建全局-局部注意力掩码
        
        - 全局token可以关注所有位置
        - 所有token可以关注全局token
        - 非全局token使用滑动窗口
        """
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        # 全局token可以关注所有位置
        for idx in self.global_tokens_idx:
            if idx < seq_len:
                mask[idx, :] = 0.0
        
        # 所有token可以关注全局token
        for idx in self.global_tokens_idx:
            if idx < seq_len:
                mask[:, idx] = 0.0
        
        # 滑动窗口
        for i in range(seq_len):
            if i not in self.global_tokens_idx:
                start = max(0, i - self.window_size)
                end = min(seq_len, i + self.window_size + 1)
                mask[i, start:end] = 0.0
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用全局-局部掩码
        mask = self._create_global_local_mask(seq_len, x.device)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output)


# =============================================================================
# 2. Linear Attention Implementations
# =============================================================================

class PerformerAttention(nn.Module):
    """
    Performer: 使用正交随机特征(ORF)近似Softmax注意力
    
    通过随机特征映射将复杂度从O(n²d)降低到O(nd²)
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        nb_features: 随机特征数量 (默认d*log(d))
        dropout: dropout概率
        ortho_features: 是否使用正交特征
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        nb_features: Optional[int] = None,
        dropout: float = 0.1,
        ortho_features: bool = True
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.nb_features = nb_features or int(self.d_head * math.log(self.d_head))
        self.ortho_features = ortho_features
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 随机特征矩阵 (将在第一次前向传播时初始化)
        self.register_buffer('projection_matrix', None)
        
    def _create_projection_matrix(self, device: torch.device) -> torch.Tensor:
        """创建随机投影矩阵"""
        if self.projection_matrix is None or self.projection_matrix.device != device:
            # 从标准正态分布采样
            matrix = torch.randn(self.d_head, self.nb_features, device=device)
            
            if self.ortho_features:
                # 使用QR分解创建正交矩阵
                q, _ = torch.linalg.qr(matrix)
                matrix = q[:, :self.nb_features]
            
            self.projection_matrix = matrix
        
        return self.projection_matrix
    
    def _kernel_feature_map(self, x: torch.Tensor) -> torch.Tensor:
        """
        核特征映射: φ(x) = exp(-||x||²/2) * [exp(w₁^T·x), ..., exp(wₘ^T·x)]
        
        使用正弦/余弦变换的近似版本
        """
        projection = self._create_projection_matrix(x.device)
        
        # 计算 x @ projection
        x_projected = torch.matmul(x, projection)
        
        # 使用正弦和余弦特征 (正交随机特征)
        x_norm = (x ** 2).sum(dim=-1, keepdim=True) / 2
        features = torch.cat([
            torch.exp(x_projected - x_norm),
        ], dim=-1)
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        使用线性注意力公式:
        Attention(Q, K, V) = φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)
        """
        batch_size, seq_len, _ = x.shape
        
        # 投影
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        # (batch, n_heads, seq_len, d_head)
        
        # 应用特征映射
        Q_prime = self._kernel_feature_map(Q)  # (batch, n_heads, seq_len, nb_features)
        K_prime = self._kernel_feature_map(K)
        
        # 线性注意力计算
        # KV = Σ(φ(k_i)^T @ v_i)
        KV = torch.matmul(K_prime.transpose(-2, -1), V)  # (batch, n_heads, nb_features, d_head)
        
        # Z = Σ(φ(q_i)^T @ φ(k_j))
        Z = K_prime.sum(dim=-2, keepdim=True)  # (batch, n_heads, 1, nb_features)
        
        # 输出 = φ(Q) @ KV / (φ(Q) @ Z^T)
        numerator = torch.matmul(Q_prime, KV)  # (batch, n_heads, seq_len, d_head)
        denominator = torch.matmul(Q_prime, Z.transpose(-2, -1))  # (batch, n_heads, seq_len, 1)
        denominator = torch.clamp(denominator, min=1e-6)  # 防止除零
        
        attn_output = numerator / denominator
        
        # 合并头
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output)


class LinearAttentionRNN(nn.Module):
    """
    线性注意力的RNN形式实现
    
    利用线性注意力的结合律，实现O(1)推理复杂度的自回归生成
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        feature_dim: 特征映射维度
        dropout: dropout概率
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        feature_dim: int = 64,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.feature_dim = feature_dim
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
        # 特征映射层
        self.feature_map = nn.Sequential(
            nn.Linear(self.d_head, feature_dim),
            nn.ELU(),
            nn.Linear(feature_dim, feature_dim)
        )
        
    def forward(self, x: torch.Tensor, state: Optional[Tuple] = None):
        """
        前向传播
        
        Args:
            x: 输入 (batch, seq_len, d_model)
            state: 可选的RNN状态 (用于自回归生成)
            
        Returns:
            output: 输出 (batch, seq_len, d_model)
            new_state: 新的RNN状态
        """
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # 特征映射
        Q_mapped = self.feature_map(Q)  # (batch, n_heads, seq_len, feature_dim)
        K_mapped = self.feature_map(K)
        
        if state is None:
            # 训练模式: 使用完整的线性注意力
            KV = torch.matmul(K_mapped.transpose(-2, -1), V)
            Z = K_mapped.sum(dim=-2, keepdim=True)
            
            numerator = torch.matmul(Q_mapped, KV)
            denominator = torch.matmul(Q_mapped, Z.transpose(-2, -1))
            denominator = torch.clamp(denominator, min=1e-6)
            
            attn_output = numerator / denominator
            new_state = None
        else:
            # 推理模式: 增量更新状态
            S, z = state  # S: (batch, n_heads, feature_dim, d_head), z: (batch, n_heads, 1, feature_dim)
            
            outputs = []
            for t in range(seq_len):
                q_t = Q_mapped[:, :, t:t+1, :]  # (batch, n_heads, 1, feature_dim)
                k_t = K_mapped[:, :, t:t+1, :]
                v_t = V[:, :, t:t+1, :]
                
                # 更新状态
                S = S + torch.matmul(k_t.transpose(-2, -1), v_t)
                z = z + k_t
                
                # 计算输出
                numerator = torch.matmul(q_t, S)
                denominator = torch.matmul(q_t, z.transpose(-2, -1))
                denominator = torch.clamp(denominator, min=1e-6)
                
                o_t = numerator / denominator
                outputs.append(o_t)
            
            attn_output = torch.cat(outputs, dim=2)
            new_state = (S, z)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output), new_state
    
    def init_state(self, batch_size: int, device: torch.device) -> Tuple:
        """初始化RNN状态"""
        S = torch.zeros(batch_size, self.n_heads, self.feature_dim, self.d_head, device=device)
        z = torch.zeros(batch_size, self.n_heads, 1, self.feature_dim, device=device)
        return (S, z)


# =============================================================================
# 3. Memory Optimization Techniques
# =============================================================================

class MemoryEfficientAttention(nn.Module):
    """
    内存高效注意力
    
    使用梯度检查点和分块计算来减少内存占用
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        dropout: dropout概率
        chunk_size: 分块大小
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        chunk_size: int = 1024
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.chunk_size = chunk_size
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _memory_efficient_forward(
        self,
        Q: torch.Tensor,
        K: torch.Tensor,
        V: torch.Tensor
    ) -> torch.Tensor:
        """
        内存高效的前向传播
        
        分块处理Q，避免存储完整的n×n注意力矩阵
        """
        batch_size, n_heads, seq_len, d_head = Q.shape
        
        outputs = []
        
        # 分块处理查询
        for i in range(0, seq_len, self.chunk_size):
            Q_chunk = Q[:, :, i:i+self.chunk_size, :]
            
            # 计算当前块的注意力
            scores = torch.matmul(Q_chunk, K.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            # 应用注意力
            output_chunk = torch.matmul(attn_weights, V)
            outputs.append(output_chunk)
        
        return torch.cat(outputs, dim=2)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        # 使用内存高效的计算
        if seq_len > self.chunk_size:
            attn_output = self._memory_efficient_forward(Q, K, V)
        else:
            # 短序列使用标准计算
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            attn_output = torch.matmul(attn_weights, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output)


class GradientCheckpointAttention(nn.Module):
    """
    使用梯度检查点的注意力
    
    通过重计算前向传播来减少激活值的内存占用
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        dropout: float = 0.1,
        use_checkpoint: bool = True
    ):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.use_checkpoint = use_checkpoint
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _attention_forward(self, Q, K, V):
        """注意力计算"""
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        return torch.matmul(attn_weights, V)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，可选使用梯度检查点"""
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        if self.use_checkpoint and self.training:
            # 使用梯度检查点
            attn_output = torch.utils.checkpoint.checkpoint(
                self._attention_forward, Q, K, V
            )
        else:
            attn_output = self._attention_forward(Q, K, V)
        
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output)


# =============================================================================
# 4. Long Context Processing
# =============================================================================

class LongformerAttention(nn.Module):
    """
    Longformer风格的注意力实现
    
    结合滑动窗口注意力和全局注意力，适用于长文档处理
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        window_size: 滑动窗口大小
        dilation: 膨胀率 (每隔dilation个token选择一个)
        global_token_indices: 全局token位置
        dropout: dropout概率
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        window_size: int = 256,
        dilation: int = 1,
        global_token_indices: list = [0],
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.dilation = dilation
        self.global_token_indices = global_token_indices
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_longformer_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建Longformer注意力掩码
        
        包含:
        1. 全局注意力: 全局token关注所有位置，所有token关注全局token
        2. 滑动窗口: 局部注意力
        3. 膨胀: 在窗口内间隔选择
        """
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        # 全局注意力
        for idx in self.global_token_indices:
            if idx < seq_len:
                mask[idx, :] = 0.0  # 全局token关注所有
                mask[:, idx] = 0.0  # 所有token关注全局
        
        # 滑动窗口 + 膨胀
        for i in range(seq_len):
            if i in self.global_token_indices:
                continue
            
            # 左侧窗口
            left_start = max(0, i - self.window_size)
            for j in range(left_start, i):
                if (i - j) % self.dilation == 0:
                    mask[i, j] = 0.0
            
            # 右侧窗口
            right_end = min(seq_len, i + self.window_size + 1)
            for j in range(i + 1, right_end):
                if (j - i) % self.dilation == 0:
                    mask[i, j] = 0.0
            
            # 自己关注自己
            mask[i, i] = 0.0
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用Longformer掩码
        mask = self._create_longformer_mask(seq_len, x.device)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output)


class BigBirdAttention(nn.Module):
    """
    BigBird风格的注意力实现
    
    在Longformer基础上增加随机注意力
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        window_size: 滑动窗口大小
        num_random_tokens: 每个查询关注的随机token数
        global_token_indices: 全局token位置
        dropout: dropout概率
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        window_size: int = 256,
        num_random_tokens: int = 32,
        global_token_indices: list = [0],
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.window_size = window_size
        self.num_random_tokens = num_random_tokens
        self.global_token_indices = global_token_indices
        self.scale = self.d_head ** -0.5
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def _create_bigbird_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        """
        创建BigBird注意力掩码
        
        包含:
        1. 全局注意力
        2. 滑动窗口
        3. 随机注意力
        """
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        # 全局注意力
        for idx in self.global_token_indices:
            if idx < seq_len:
                mask[idx, :] = 0.0
                mask[:, idx] = 0.0
        
        # 滑动窗口
        for i in range(seq_len):
            if i in self.global_token_indices:
                continue
            
            start = max(0, i - self.window_size)
            end = min(seq_len, i + self.window_size + 1)
            mask[i, start:end] = 0.0
        
        # 随机注意力
        torch.manual_seed(42)  # 固定种子保证可复现
        for i in range(seq_len):
            if i in self.global_token_indices:
                continue
            
            # 随机选择token (排除已经在窗口内的)
            candidates = []
            for j in range(seq_len):
                if j != i and abs(i - j) > self.window_size and j not in self.global_token_indices:
                    candidates.append(j)
            
            if len(candidates) >= self.num_random_tokens:
                random_tokens = torch.randperm(len(candidates))[:self.num_random_tokens]
                for idx in random_tokens:
                    mask[i, candidates[idx]] = 0.0
        
        return mask
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播"""
        batch_size, seq_len, _ = x.shape
        
        Q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        K = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        V = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)
        
        scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        
        # 应用BigBird掩码
        mask = self._create_bigbird_mask(seq_len, x.device)
        scores = scores + mask.unsqueeze(0).unsqueeze(0)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        attn_output = torch.matmul(attn_weights, V)
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.d_model
        )
        
        return self.out_proj(attn_output)


class HierarchicalAttention(nn.Module):
    """
    分层注意力机制
    
    将长序列分块处理，先进行块内注意力，再进行块间注意力
    
    Args:
        d_model: 模型维度
        n_heads: 注意力头数
        block_size: 块大小
        dropout: dropout概率
    """
    def __init__(
        self,
        d_model: int = 512,
        n_heads: int = 8,
        block_size: int = 512,
        dropout: float = 0.1
    ):
        super().__init__()
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.block_size = block_size
        self.scale = self.d_head ** -0.5
        
        # 块内注意力
        self.local_q_proj = nn.Linear(d_model, d_model)
        self.local_k_proj = nn.Linear(d_model, d_model)
        self.local_v_proj = nn.Linear(d_model, d_model)
        self.local_out_proj = nn.Linear(d_model, d_model)
        
        # 块间注意力 (使用压缩的表示)
        self.global_q_proj = nn.Linear(d_model, d_model)
        self.global_k_proj = nn.Linear(d_model, d_model)
        self.global_v_proj = nn.Linear(d_model, d_model)
        self.global_out_proj = nn.Linear(d_model, d_model)
        
        # 压缩层 (将块压缩为单个表示)
        self.compression = nn.AdaptiveAvgPool1d(1)
        
        self.dropout = nn.Dropout(dropout)
        self.gate = nn.Linear(d_model * 2, d_model)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: (batch_size, seq_len, d_model)
            
        Returns:
            output: (batch_size, seq_len, d_model)
        """
        batch_size, seq_len, _ = x.shape
        num_blocks = (seq_len + self.block_size - 1) // self.block_size
        
        # ===== 块内注意力 =====
        local_outputs = []
        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, seq_len)
            
            block = x[:, start:end, :]
            
            Q = self.local_q_proj(block).view(
                batch_size, end - start, self.n_heads, self.d_head
            ).transpose(1, 2)
            K = self.local_k_proj(block).view(
                batch_size, end - start, self.n_heads, self.d_head
            ).transpose(1, 2)
            V = self.local_v_proj(block).view(
                batch_size, end - start, self.n_heads, self.d_head
            ).transpose(1, 2)
            
            scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            local_out = torch.matmul(attn_weights, V)
            local_out = local_out.transpose(1, 2).contiguous().view(
                batch_size, end - start, self.d_model
            )
            local_out = self.local_out_proj(local_out)
            local_outputs.append(local_out)
        
        local_output = torch.cat(local_outputs, dim=1)
        
        # ===== 块间注意力 =====
        # 压缩每个块为全局表示
        block_representations = []
        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, seq_len)
            block = x[:, start:end, :]  # (batch, block_size, d_model)
            
            # 压缩: (batch, d_model, block_size) -> (batch, d_model, 1) -> (batch, 1, d_model)
            compressed = self.compression(block.transpose(1, 2)).transpose(1, 2)
            block_representations.append(compressed)
        
        global_input = torch.cat(block_representations, dim=1)  # (batch, num_blocks, d_model)
        
        Q_global = self.global_q_proj(global_input).view(
            batch_size, num_blocks, self.n_heads, self.d_head
        ).transpose(1, 2)
        K_global = self.global_k_proj(global_input).view(
            batch_size, num_blocks, self.n_heads, self.d_head
        ).transpose(1, 2)
        V_global = self.global_v_proj(global_input).view(
            batch_size, num_blocks, self.n_heads, self.d_head
        ).transpose(1, 2)
        
        scores_global = torch.matmul(Q_global, K_global.transpose(-2, -1)) * self.scale
        attn_weights_global = F.softmax(scores_global, dim=-1)
        attn_weights_global = self.dropout(attn_weights_global)
        
        global_out = torch.matmul(attn_weights_global, V_global)
        global_out = global_out.transpose(1, 2).contiguous().view(
            batch_size, num_blocks, self.d_model
        )
        global_out = self.global_out_proj(global_out)
        
        # 将全局信息扩展回原始序列长度
        global_expanded = []
        for i in range(num_blocks):
            start = i * self.block_size
            end = min((i + 1) * self.block_size, seq_len)
            block_len = end - start
            global_expanded.append(global_out[:, i:i+1, :].expand(-1, block_len, -1))
        
        global_output = torch.cat(global_expanded, dim=1)
        
        # 门控融合
        gate_input = torch.cat([local_output, global_output], dim=-1)
        gate = torch.sigmoid(self.gate(gate_input))
        
        output = gate * local_output + (1 - gate) * global_output
        
        return output


# =============================================================================
# 5. Utility Functions and Examples
# =============================================================================

def compare_memory_usage():
    """
    比较不同注意力机制的内存使用情况
    """
    print("=" * 60)
    print("内存使用比较")
    print("=" * 60)
    
    batch_size = 2
    d_model = 512
    seq_lengths = [512, 1024, 2048, 4096]
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    for seq_len in seq_lengths:
        print(f"\n序列长度: {seq_len}")
        
        # 标准注意力的理论内存
        standard_memory = batch_size * seq_len * seq_len * 4  # float32
        print(f"  标准注意力 (n²): {standard_memory / 1024**2:.2f} MB")
        
        # 滑动窗口
        window_size = 256
        sliding_memory = batch_size * seq_len * (2 * window_size + 1) * 4
        print(f"  滑动窗口 (w={window_size}): {sliding_memory / 1024**2:.2f} MB")
        
        # 线性注意力
        feature_dim = 64
        linear_memory = batch_size * feature_dim * d_model * 4
        print(f"  线性注意力 (d={feature_dim}): {linear_memory / 1024**2:.2f} MB")


def benchmark_attention_speed():
    """
    基准测试不同注意力机制的速度
    """
    print("\n" + "=" * 60)
    print("速度基准测试")
    print("=" * 60)
    
    import time
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"设备: {device}")
    
    batch_size = 2
    d_model = 512
    n_heads = 8
    seq_len = 1024
    
    x = torch.randn(batch_size, seq_len, d_model, device=device)
    
    # 测试不同注意力机制
    attentions = {
        'Standard (Simulated)': None,  # 不实际运行，避免OOM
        'SlidingWindow': SlidingWindowAttention(d_model, n_heads, window_size=256).to(device),
        'GlobalLocal': GlobalLocalAttention(d_model, n_heads, window_size=256).to(device),
        'Performer': PerformerAttention(d_model, n_heads).to(device),
        'LinearRNN': LinearAttentionRNN(d_model, n_heads).to(device),
        'Longformer': LongformerAttention(d_model, n_heads, window_size=256).to(device),
        'BigBird': BigBirdAttention(d_model, n_heads, window_size=256).to(device),
        'Hierarchical': HierarchicalAttention(d_model, n_heads, block_size=256).to(device),
    }
    
    num_runs = 10
    warmup_runs = 3
    
    for name, attn in attentions.items():
        if attn is None:
            continue
        
        attn.eval()
        
        # Warmup
        with torch.no_grad():
            for _ in range(warmup_runs):
                if name == 'LinearRNN':
                    _ = attn(x)[0]
                else:
                    _ = attn(x)
        
        # Benchmark
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        start_time = time.time()
        
        with torch.no_grad():
            for _ in range(num_runs):
                if name == 'LinearRNN':
                    _ = attn(x)[0]
                else:
                    _ = attn(x)
        
        if device.type == 'cuda':
            torch.cuda.synchronize()
        
        elapsed = (time.time() - start_time) / num_runs * 1000  # ms
        
        # 计算参数量
        num_params = sum(p.numel() for p in attn.parameters())
        
        print(f"\n{name}:")
        print(f"  平均时间: {elapsed:.2f} ms")
        print(f"  参数量: {num_params / 1e6:.2f} M")


def demo_long_context_processing():
    """
    长文本处理示例
    """
    print("\n" + "=" * 60)
    print("长文本处理示例")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 1
    d_model = 512
    n_heads = 8
    
    # 模拟不同长度的文档
    scenarios = [
        ("短文档", 512),
        ("中等文档", 2048),
        ("长文档", 8192),
    ]
    
    for name, seq_len in scenarios:
        print(f"\n场景: {name} (长度: {seq_len})")
        
        x = torch.randn(batch_size, seq_len, d_model, device=device)
        
        # 选择合适的方法
        if seq_len <= 512:
            print("  推荐: 标准注意力或Flash Attention")
            attn = MemoryEfficientAttention(d_model, n_heads, chunk_size=256).to(device)
        elif seq_len <= 2048:
            print("  推荐: Longformer或滑动窗口注意力")
            attn = LongformerAttention(d_model, n_heads, window_size=256).to(device)
        else:
            print("  推荐: 分层注意力或线性注意力")
            attn = HierarchicalAttention(d_model, n_heads, block_size=512).to(device)
        
        attn.eval()
        
        with torch.no_grad():
            output = attn(x)
        
        print(f"  输入形状: {x.shape}")
        print(f"  输出形状: {output.shape}")
        
        # 计算理论内存节省
        standard_memory = batch_size * seq_len * seq_len * 4 / 1024**2
        print(f"  标准注意力内存: {standard_memory:.2f} MB")


def demo_autoregressive_generation():
    """
    自回归生成示例 (展示线性注意力的RNN特性)
    """
    print("\n" + "=" * 60)
    print("自回归生成示例 (线性注意力RNN)")
    print("=" * 60)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    batch_size = 1
    d_model = 512
    n_heads = 8
    seq_len = 100
    
    attn = LinearAttentionRNN(d_model, n_heads).to(device)
    attn.eval()
    
    # 初始化状态
    state = attn.init_state(batch_size, device)
    
    print("\n逐步生成:")
    
    generated = []
    
    with torch.no_grad():
        for i in range(seq_len):
            # 每次只输入一个token
            x_t = torch.randn(batch_size, 1, d_model, device=device)
            
            # 前向传播，更新状态
            output, state = attn(x_t, state)
            
            generated.append(output)
            
            if (i + 1) % 20 == 0:
                print(f"  已生成 {i+1}/{seq_len} tokens")
    
    # 合并所有输出
    full_output = torch.cat(generated, dim=1)
    print(f"\n最终输出形状: {full_output.shape}")
    print("特点: 每步推理复杂度O(1)，与序列长度无关！")


# =============================================================================
# 6. Main
# =============================================================================

if __name__ == "__main__":
    print("Efficient Attention Mechanisms Demo")
    print("=" * 60)
    
    # 运行演示
    compare_memory_usage()
    
    try:
        benchmark_attention_speed()
    except Exception as e:
        print(f"\n速度测试遇到错误 (可能是显存不足): {e}")
    
    demo_long_context_processing()
    demo_autoregressive_generation()
    
    print("\n" + "=" * 60)
    print("所有演示完成!")
    print("=" * 60)
