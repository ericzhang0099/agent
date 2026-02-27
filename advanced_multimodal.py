"""
Advanced Multimodal Fusion Framework
=====================================

A comprehensive implementation of advanced multimodal learning techniques including:
- Cross-modal attention mechanisms
- Unified representation learning
- Vision-Language-Action (VLA) modeling
- Multimodal fusion strategies

Author: AI Research Team
Date: 2026-02-27
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, List, Union
from dataclasses import dataclass
import math


# =============================================================================
# Configuration Classes
# =============================================================================

@dataclass
class MultimodalConfig:
    """多模态模型配置"""
    # 视觉配置
    vision_dim: int = 768
    image_size: int = 224
    patch_size: int = 16
    num_image_tokens: int = 197  # (224/16)^2 + 1 (CLS)
    
    # 语言配置
    text_dim: int = 768
    vocab_size: int = 49408
    max_text_length: int = 77
    
    # 动作配置
    action_dim: int = 7  # [x, y, z, roll, pitch, yaw, gripper]
    action_bins: int = 256
    max_action_tokens: int = 11
    
    # 共享配置
    hidden_dim: int = 768
    num_heads: int = 12
    num_layers: int = 12
    dropout: float = 0.1
    
    # 融合配置
    fusion_type: str = "cross_attention"  # ["early", "late", "cross_attention", "hybrid"]
    
    # 训练配置
    temperature: float = 0.07


# =============================================================================
# Utility Functions
# =============================================================================

def sinusoidal_position_embedding(seq_len: int, dim: int, device: torch.device) -> torch.Tensor:
    """正弦位置编码"""
    position = torch.arange(seq_len, dtype=torch.float32, device=device).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32, device=device) * 
                         -(math.log(10000.0) / dim))
    
    pe = torch.zeros(seq_len, dim, device=device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


# =============================================================================
# Core Attention Mechanisms
# =============================================================================

class CrossAttention(nn.Module):
    """
    跨模态注意力机制
    
    允许一个模态的查询(Q)关注另一个模态的键(K)和值(V)
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        assert dim % num_heads == 0, "dim must be divisible by num_heads"
        
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        self.q_proj = nn.Linear(dim, dim)
        self.k_proj = nn.Linear(dim, dim)
        self.v_proj = nn.Linear(dim, dim)
        self.out_proj = nn.Linear(dim, dim)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Args:
            query: [batch_size, query_len, dim]
            key: [batch_size, key_len, dim]
            value: [batch_size, value_len, dim]
            attention_mask: [batch_size, query_len, key_len]
        
        Returns:
            output: [batch_size, query_len, dim]
        """
        batch_size = query.size(0)
        
        # 投影并分头
        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        
        # 注意力计算
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale  # [B, H, Q, K]
        
        if attention_mask is not None:
            attn_scores = attn_scores.masked_fill(
                attention_mask.unsqueeze(1) == 0, float('-inf')
            )
        
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # 加权求和
        attn_output = torch.matmul(attn_weights, V)  # [B, H, Q, D]
        attn_output = attn_output.transpose(1, 2).contiguous().view(
            batch_size, -1, self.dim
        )
        
        output = self.out_proj(attn_output)
        return output


class BidirectionalCrossAttention(nn.Module):
    """
    双向交叉注意力
    
    两个模态相互关注，实现信息的双向流动
    """
    def __init__(self, dim: int, num_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.modality_a_to_b = CrossAttention(dim, num_heads, dropout)
        self.modality_b_to_a = CrossAttention(dim, num_heads, dropout)
        
        self.norm_a = nn.LayerNorm(dim)
        self.norm_b = nn.LayerNorm(dim)
        
    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feat_a: [batch_size, len_a, dim]
            feat_b: [batch_size, len_b, dim]
        
        Returns:
            enhanced_a: [batch_size, len_a, dim]
            enhanced_b: [batch_size, len_b, dim]
        """
        # A关注B
        a_enhanced = self.modality_a_to_b(
            query=feat_a, key=feat_b, value=feat_b
        )
        a_enhanced = self.norm_a(feat_a + a_enhanced)
        
        # B关注A
        b_enhanced = self.modality_b_to_a(
            query=feat_b, key=feat_a, value=feat_a
        )
        b_enhanced = self.norm_b(feat_b + b_enhanced)
        
        return a_enhanced, b_enhanced


class MultiHeadCoAttention(nn.Module):
    """
    多头协同注意力 (Co-Attention)
    
    通过交替更新实现两个模态的深度交互
    """
    def __init__(self, dim: int, num_heads: int = 8, num_iterations: int = 2):
        super().__init__()
        self.num_iterations = num_iterations
        self.cross_attn = CrossAttention(dim, num_heads)
        self.norm = nn.LayerNorm(dim)
        
    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        协同注意力前向传播
        """
        a, b = feat_a, feat_b
        
        for _ in range(self.num_iterations):
            # A关注B
            a_new = self.cross_attn(query=a, key=b, value=b)
            a = self.norm(a + a_new)
            
            # B关注更新后的A
            b_new = self.cross_attn(query=b, key=a, value=a)
            b = self.norm(b + b_new)
        
        return a, b


# =============================================================================
# Modality Encoders
# =============================================================================

class VisionEncoder(nn.Module):
    """
    视觉编码器 (基于ViT)
    """
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        in_channels: int = 3,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch嵌入
        self.patch_embed = nn.Conv2d(
            in_channels, hidden_dim,
            kernel_size=patch_size, stride=patch_size
        )
        
        # CLS token和位置编码
        self.cls_token = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.pos_embed = nn.Parameter(
            torch.zeros(1, self.num_patches + 1, hidden_dim)
        )
        
        # Transformer编码层
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        # 初始化
        nn.init.normal_(self.cls_token, std=0.02)
        nn.init.normal_(self.pos_embed, std=0.02)
        
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        Args:
            images: [batch_size, channels, height, width]
        
        Returns:
            features: [batch_size, num_patches+1, hidden_dim]
        """
        batch_size = images.size(0)
        
        # Patch嵌入
        x = self.patch_embed(images)  # [B, D, H/P, W/P]
        x = x.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 添加CLS token
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        # 添加位置编码
        x = x + self.pos_embed
        x = self.dropout(x)
        
        # Transformer编码
        x = self.transformer(x)
        x = self.norm(x)
        
        return x


class TextEncoder(nn.Module):
    """
    文本编码器 (基于Transformer)
    """
    def __init__(
        self,
        vocab_size: int = 49408,
        max_length: int = 77,
        hidden_dim: int = 768,
        num_layers: int = 12,
        num_heads: int = 12,
        dropout: float = 0.1
    ):
        super().__init__()
        self.max_length = max_length
        
        self.token_embedding = nn.Embedding(vocab_size, hidden_dim)
        self.pos_embedding = nn.Parameter(
            torch.zeros(1, max_length, hidden_dim)
        )
        
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            activation='gelu',
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        self.norm = nn.LayerNorm(hidden_dim)
        self.dropout = nn.Dropout(dropout)
        
        nn.init.normal_(self.pos_embedding, std=0.02)
        
    def forward(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            text_tokens: [batch_size, seq_len]
        
        Returns:
            features: [batch_size, seq_len, hidden_dim]
        """
        # 嵌入
        x = self.token_embedding(text_tokens)
        
        # 添加位置编码
        seq_len = x.size(1)
        x = x + self.pos_embedding[:, :seq_len, :]
        x = self.dropout(x)
        
        # 创建注意力掩码 (处理padding)
        mask = (text_tokens == 0).unsqueeze(1).unsqueeze(2)
        
        # Transformer编码
        x = self.transformer(x, src_key_padding_mask=mask.squeeze(1).squeeze(1))
        x = self.norm(x)
        
        return x


class ActionEncoder(nn.Module):
    """
    动作编码器
    
    将连续动作离散化为token，或编码为连续表示
    """
    def __init__(
        self,
        action_dim: int = 7,
        action_bins: int = 256,
        hidden_dim: int = 768,
        use_discrete: bool = True
    ):
        super().__init__()
        self.action_dim = action_dim
        self.action_bins = action_bins
        self.use_discrete = use_discrete
        
        if use_discrete:
            # 离散动作使用嵌入层
            self.action_embedding = nn.Embedding(action_bins, hidden_dim)
            self.action_proj = nn.Linear(action_dim * hidden_dim, hidden_dim)
        else:
            # 连续动作使用MLP
            self.action_mlp = nn.Sequential(
                nn.Linear(action_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim)
            )
    
    def discretize(self, actions: torch.Tensor) -> torch.Tensor:
        """
        将连续动作离散化为bin索引
        
        Args:
            actions: [batch_size, action_dim] in [-1, 1]
        
        Returns:
            action_tokens: [batch_size, action_dim]
        """
        # 将[-1, 1]映射到[0, bins-1]
        actions = torch.clamp(actions, -1, 1)
        tokens = ((actions + 1) / 2 * (self.action_bins - 1)).long()
        return tokens
    
    def forward(self, actions: torch.Tensor) -> torch.Tensor:
        """
        Args:
            actions: [batch_size, action_dim]
        
        Returns:
            features: [batch_size, hidden_dim]
        """
        if self.use_discrete:
            tokens = self.discretize(actions)
            embeddings = self.action_embedding(tokens)  # [B, A, D]
            embeddings = embeddings.flatten(1)  # [B, A*D]
            features = self.action_proj(embeddings)
        else:
            features = self.action_mlp(actions)
        
        return features


# =============================================================================
# Fusion Modules
# =============================================================================

class EarlyFusion(nn.Module):
    """
    早期融合: 在特征层面直接拼接
    """
    def __init__(self, input_dims: List[int], output_dim: int):
        super().__init__()
        total_dim = sum(input_dims)
        self.fusion_proj = nn.Sequential(
            nn.Linear(total_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(),
            nn.Dropout(0.1)
        )
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 多个模态的特征张量
        
        Returns:
            fused: 融合后的特征
        """
        concat_feat = torch.cat(features, dim=-1)
        fused = self.fusion_proj(concat_feat)
        return fused


class LateFusion(nn.Module):
    """
    晚期融合: 各模态独立预测后融合
    """
    def __init__(
        self,
        input_dim: int,
        num_modalities: int,
        output_dim: int,
        fusion_weights: Optional[List[float]] = None
    ):
        super().__init__()
        self.num_modalities = num_modalities
        
        # 每个模态的预测头
        self.heads = nn.ModuleList([
            nn.Sequential(
                nn.Linear(input_dim, input_dim // 2),
                nn.ReLU(),
                nn.Linear(input_dim // 2, output_dim)
            ) for _ in range(num_modalities)
        ])
        
        # 可学习的融合权重
        if fusion_weights is None:
            self.weights = nn.Parameter(torch.ones(num_modalities))
        else:
            self.register_buffer('weights', torch.tensor(fusion_weights))
    
    def forward(self, *features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: 各模态的特征
        
        Returns:
            fused_output: 融合后的预测结果
        """
        predictions = [head(feat) for head, feat in zip(self.heads, features)]
        
        # 加权融合
        weights = F.softmax(self.weights, dim=0)
        fused = sum(w * pred for w, pred in zip(weights, predictions))
        
        return fused


class CrossModalFusion(nn.Module):
    """
    跨模态注意力融合
    
    使用交叉注意力实现模态间的深度交互
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_layers: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.layers = nn.ModuleList([
            BidirectionalCrossAttention(dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
    
    def forward(
        self,
        feat_a: torch.Tensor,
        feat_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            feat_a: [batch_size, len_a, dim]
            feat_b: [batch_size, len_b, dim]
        
        Returns:
            fused: [batch_size, len_a+len_b, dim]
        """
        # 应用多层双向交叉注意力
        for layer in self.layers:
            feat_a, feat_b = layer(feat_a, feat_b)
        
        # 池化并融合
        pooled_a = feat_a.mean(dim=1)  # [B, D]
        pooled_b = feat_b.mean(dim=1)  # [B, D]
        
        fused = self.fusion_mlp(torch.cat([pooled_a, pooled_b], dim=-1))
        return fused


class HybridFusion(nn.Module):
    """
    混合融合: 结合早期和晚期融合的优点
    """
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        num_fusion_layers: int = 2
    ):
        super().__init__()
        
        # 早期融合层
        self.early_fusion = CrossModalFusion(dim, num_heads, num_fusion_layers)
        
        # 模态特定处理
        self.modality_processors = nn.ModuleList([
            nn.TransformerEncoderLayer(dim, num_heads, dim * 4, batch_first=True)
            for _ in range(2)
        ])
        
        # 晚期融合
        self.late_fusion = nn.Sequential(
            nn.Linear(dim * 2, dim),
            nn.LayerNorm(dim),
            nn.ReLU()
        )
    
    def forward(self, feat_a: torch.Tensor, feat_b: torch.Tensor) -> torch.Tensor:
        """
        混合融合前向传播
        """
        # 早期融合
        early_fused = self.early_fusion(feat_a, feat_b)
        
        # 模态特定处理
        proc_a = self.modality_processors[0](feat_a)
        proc_b = self.modality_processors[1](feat_b)
        
        # 池化
        pooled_a = proc_a.mean(dim=1)
        pooled_b = proc_b.mean(dim=1)
        
        # 晚期融合
        late_fused = self.late_fusion(torch.cat([pooled_a, pooled_b], dim=-1))
        
        # 最终融合
        final = early_fused + late_fused
        return final


# =============================================================================
# Main Multimodal Model
# =============================================================================

class AdvancedMultimodalModel(nn.Module):
    """
    高级多模态融合模型
    
    支持视觉、语言、动作三种模态的统一建模
    """
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # 模态编码器
        self.vision_encoder = VisionEncoder(
            image_size=config.image_size,
            patch_size=config.patch_size,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        self.text_encoder = TextEncoder(
            vocab_size=config.vocab_size,
            max_length=config.max_text_length,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            num_heads=config.num_heads,
            dropout=config.dropout
        )
        
        self.action_encoder = ActionEncoder(
            action_dim=config.action_dim,
            action_bins=config.action_bins,
            hidden_dim=config.hidden_dim
        )
        
        # 模态投影 (统一维度)
        self.vision_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.text_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.action_proj = nn.Linear(config.hidden_dim, config.hidden_dim)
        
        # 融合模块
        if config.fusion_type == "early":
            self.fusion = EarlyFusion([config.hidden_dim] * 3, config.hidden_dim)
        elif config.fusion_type == "late":
            self.fusion = LateFusion(config.hidden_dim, 3, config.hidden_dim)
        elif config.fusion_type == "cross_attention":
            self.fusion = CrossModalFusion(config.hidden_dim, config.num_heads)
        elif config.fusion_type == "hybrid":
            self.fusion = HybridFusion(config.hidden_dim, config.num_heads)
        else:
            raise ValueError(f"Unknown fusion type: {config.fusion_type}")
        
        # 温度参数 (用于对比学习)
        self.logit_scale = nn.Parameter(torch.ones([]) * math.log(1.0 / config.temperature))
        
    def encode_vision(self, images: torch.Tensor) -> torch.Tensor:
        """编码视觉输入"""
        features = self.vision_encoder(images)
        # 使用CLS token作为全局表示
        cls_feat = features[:, 0]
        return F.normalize(self.vision_proj(cls_feat), dim=-1)
    
    def encode_text(self, text_tokens: torch.Tensor) -> torch.Tensor:
        """编码文本输入"""
        features = self.text_encoder(text_tokens)
        # 使用EOS token或平均池化
        pooled = features.mean(dim=1)
        return F.normalize(self.text_proj(pooled), dim=-1)
    
    def encode_action(self, actions: torch.Tensor) -> torch.Tensor:
        """编码动作输入"""
        features = self.action_encoder(actions)
        return F.normalize(self.action_proj(features), dim=-1)
    
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        多模态前向传播
        
        Args:
            images: [batch_size, 3, H, W]
            text_tokens: [batch_size, seq_len]
            actions: [batch_size, action_dim]
        
        Returns:
            outputs: 包含各模态特征和融合结果的字典
        """
        outputs = {}
        features = []
        
        # 编码各模态
        if images is not None:
            vision_feat = self.encode_vision(images)
            outputs['vision'] = vision_feat
            features.append(vision_feat)
        
        if text_tokens is not None:
            text_feat = self.encode_text(text_tokens)
            outputs['text'] = text_feat
            features.append(text_feat)
        
        if actions is not None:
            action_feat = self.encode_action(actions)
            outputs['action'] = action_feat
            features.append(action_feat)
        
        # 融合 (如果有多于一个模态)
        if len(features) >= 2:
            if self.config.fusion_type in ["early", "late"]:
                fused = self.fusion(*features)
            else:
                # 对于需要序列输入的融合方式，扩展维度
                seq_features = [f.unsqueeze(1) for f in features]
                # 两两融合
                fused = self.fusion(seq_features[0].expand(-1, 10, -1), 
                                   seq_features[1].expand(-1, 10, -1))
            outputs['fused'] = fused
        
        outputs['logit_scale'] = self.logit_scale.exp()
        
        return outputs
    
    def compute_contrastive_loss(
        self,
        vision_feat: torch.Tensor,
        text_feat: torch.Tensor
    ) -> torch.Tensor:
        """
        计算视觉-语言对比损失
        
        Args:
            vision_feat: [batch_size, dim]
            text_feat: [batch_size, dim]
        
        Returns:
            loss: 标量
        """
        logit_scale = self.logit_scale.exp()
        
        # 计算相似度矩阵
        logits_per_image = logit_scale * vision_feat @ text_feat.T
        logits_per_text = logits_per_image.T
        
        batch_size = vision_feat.size(0)
        labels = torch.arange(batch_size, device=vision_feat.device)
        
        # 对称损失
        loss_i2t = F.cross_entropy(logits_per_image, labels)
        loss_t2i = F.cross_entropy(logits_per_text, labels)
        loss = (loss_i2t + loss_t2i) / 2
        
        return loss


# =============================================================================
# VLA (Vision-Language-Action) Model
# =============================================================================

class VLATokenizer(nn.Module):
    """
    VLA Tokenizer
    
    将视觉、语言、动作统一token化
    """
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        # 视觉tokenizer (使用VQ-VAE或类似方法)
        self.visual_tokenizer = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(128, config.hidden_dim, 4, 2, 1),
        )
        
        # 动作tokenizer (FAST方法)
        self.action_tokenizer = nn.Linear(config.action_dim, config.hidden_dim)
        
        # 特殊token
        self.boi_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))  # begin of image
        self.eoi_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))  # end of image
        self.boa_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))  # begin of action
        self.eoa_token = nn.Parameter(torch.randn(1, 1, config.hidden_dim))  # end of action
        
    def tokenize_vision(self, images: torch.Tensor) -> torch.Tensor:
        """将图像转换为token序列"""
        B = images.size(0)
        features = self.visual_tokenizer(images)  # [B, D, H', W']
        tokens = features.flatten(2).transpose(1, 2)  # [B, N, D]
        
        # 添加特殊token
        boi = self.boi_token.expand(B, -1, -1)
        eoi = self.eoi_token.expand(B, -1, -1)
        tokens = torch.cat([boi, tokens, eoi], dim=1)
        
        return tokens
    
    def tokenize_action(self, actions: torch.Tensor) -> torch.Tensor:
        """将动作转换为token序列"""
        B = actions.size(0)
        tokens = self.action_tokenizer(actions).unsqueeze(1)  # [B, 1, D]
        
        # 添加特殊token
        boa = self.boa_token.expand(B, -1, -1)
        eoa = self.eoa_token.expand(B, -1, -1)
        tokens = torch.cat([boa, tokens, eoa], dim=1)
        
        return tokens


class VLAModel(nn.Module):
    """
    视觉-语言-动作统一模型
    
    参考UniVLA架构，统一建模三种模态
    """
    def __init__(self, config: MultimodalConfig):
        super().__init__()
        self.config = config
        
        self.tokenizer = VLATokenizer(config)
        
        # 文本嵌入
        self.text_embed = nn.Embedding(config.vocab_size, config.hidden_dim)
        
        # 统一Transformer
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            dim_feedforward=config.hidden_dim * 4,
            dropout=config.dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, config.num_layers)
        
        # 输出头
        self.vision_head = nn.Linear(config.hidden_dim, config.hidden_dim)
        self.text_head = nn.Linear(config.hidden_dim, config.vocab_size)
        self.action_head = nn.Linear(config.hidden_dim, config.action_bins)
        
    def forward(
        self,
        images: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None,
        task: str = "vision_language"  # ["vision_language", "action_prediction", "world_model"]
    ) -> Dict[str, torch.Tensor]:
        """
        VLA统一前向传播
        
        Args:
            images: [B, 3, H, W]
            text_tokens: [B, L]
            actions: [B, action_dim]
            task: 任务类型
        
        Returns:
            outputs: 预测结果
        """
        # 构建交错序列
        sequence_parts = []
        
        if text_tokens is not None:
            text_embeds = self.text_embed(text_tokens)
            sequence_parts.append(text_embeds)
        
        if images is not None:
            vision_tokens = self.tokenizer.tokenize_vision(images)
            sequence_parts.append(vision_tokens)
        
        if actions is not None:
            action_tokens = self.tokenizer.tokenize_action(actions)
            sequence_parts.append(action_tokens)
        
        # 拼接序列
        sequence = torch.cat(sequence_parts, dim=1)
        
        # 创建因果掩码
        seq_len = sequence.size(1)
        causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()
        causal_mask = causal_mask.to(sequence.device)
        
        # Transformer处理
        output = self.transformer(sequence, sequence, tgt_mask=causal_mask)
        
        # 根据任务输出
        results = {}
        if task == "vision_language":
            # 视觉-语言理解任务
            pooled = output.mean(dim=1)
            results['logits'] = self.text_head(pooled)
        elif task == "action_prediction":
            # 动作预测任务
            action_logits = self.action_head(output[:, -1, :])
            results['action_logits'] = action_logits
        elif task == "world_model":
            # 世界模型任务 (预测下一帧)
            vision_pred = self.vision_head(output)
            results['vision_pred'] = vision_pred
        
        return results


# =============================================================================
# Example Usage and Testing
# =============================================================================

def test_multimodal_model():
    """测试多模态模型"""
    print("=" * 60)
    print("Testing Advanced Multimodal Model")
    print("=" * 60)
    
    config = MultimodalConfig(
        hidden_dim=512,
        num_layers=6,
        num_heads=8,
        fusion_type="cross_attention"
    )
    
    model = AdvancedMultimodalModel(config)
    
    # 创建测试输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text_tokens = torch.randint(0, config.vocab_size, (batch_size, config.max_text_length))
    actions = torch.randn(batch_size, config.action_dim)
    
    print(f"\nInput shapes:")
    print(f"  Images: {images.shape}")
    print(f"  Text tokens: {text_tokens.shape}")
    print(f"  Actions: {actions.shape}")
    
    # 前向传播
    outputs = model(images=images, text_tokens=text_tokens, actions=actions)
    
    print(f"\nOutput shapes:")
    for key, value in outputs.items():
        if isinstance(value, torch.Tensor):
            print(f"  {key}: {value.shape}")
    
    # 测试对比损失
    vision_feat = outputs['vision']
    text_feat = outputs['text']
    loss = model.compute_contrastive_loss(vision_feat, text_feat)
    print(f"\nContrastive loss: {loss.item():.4f}")
    
    print("\n" + "=" * 60)
    print("Testing VLA Model")
    print("=" * 60)
    
    vla_model = VLAModel(config)
    vla_outputs = vla_model(
        images=images,
        text_tokens=text_tokens,
        actions=actions,
        task="action_prediction"
    )
    
    print(f"\nVLA output shapes:")
    for key, value in vla_outputs.items():
        print(f"  {key}: {value.shape}")
    
    print("\n✓ All tests passed!")
    return model, vla_model


def demonstrate_fusion_strategies():
    """演示不同的融合策略"""
    print("\n" + "=" * 60)
    print("Demonstrating Fusion Strategies")
    print("=" * 60)
    
    batch_size = 4
    seq_len = 10
    dim = 512
    
    # 模拟两个模态的特征
    feat_a = torch.randn(batch_size, seq_len, dim)
    feat_b = torch.randn(batch_size, seq_len, dim)
    
    # 早期融合
    early_fusion = EarlyFusion([dim, dim], dim)
    pooled_a = feat_a.mean(dim=1)
    pooled_b = feat_b.mean(dim=1)
    early_result = early_fusion(pooled_a, pooled_b)
    print(f"Early fusion output: {early_result.shape}")
    
    # 晚期融合
    late_fusion = LateFusion(dim, 2, 10)
    late_result = late_fusion(pooled_a, pooled_b)
    print(f"Late fusion output: {late_result.shape}")
    
    # 交叉注意力融合
    cross_fusion = CrossModalFusion(dim, num_heads=8)
    cross_result = cross_fusion(feat_a, feat_b)
    print(f"Cross-attention fusion output: {cross_result.shape}")
    
    # 混合融合
    hybrid_fusion = HybridFusion(dim, num_heads=8)
    hybrid_result = hybrid_fusion(feat_a, feat_b)
    print(f"Hybrid fusion output: {hybrid_result.shape}")
    
    print("\n✓ Fusion strategies demonstration complete!")


if __name__ == "__main__":
    # 运行测试
    test_multimodal_model()
    demonstrate_fusion_strategies()
