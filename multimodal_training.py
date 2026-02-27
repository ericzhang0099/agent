"""
Training Utilities for Multimodal Models
=========================================

提供多模态模型的训练工具和辅助函数

包括:
1. 对比学习训练循环
2. 掩码建模训练
3. 多任务学习
4. 评估指标
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from typing import Dict, List, Optional, Callable, Tuple
from tqdm import tqdm
import numpy as np


# =============================================================================
# Loss Functions
# =============================================================================

class ContrastiveLoss(nn.Module):
    """
    对比损失 (InfoNCE)
    
    用于多模态对比学习，如CLIP
    """
    def __init__(self, temperature: float = 0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            features_a: [batch_size, dim]
            features_b: [batch_size, dim]
        
        Returns:
            loss: 标量
        """
        batch_size = features_a.size(0)
        
        # 归一化
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        # 计算相似度矩阵
        logits = torch.matmul(features_a, features_b.T) / self.temperature
        
        # 标签: 对角线元素是正样本对
        labels = torch.arange(batch_size, device=features_a.device)
        
        # 对称损失
        loss_a2b = F.cross_entropy(logits, labels)
        loss_b2a = F.cross_entropy(logits.T, labels)
        loss = (loss_a2b + loss_b2a) / 2
        
        return loss


class TripletLoss(nn.Module):
    """
    三元组损失
    
    用于学习模态间的相对相似度
    """
    def __init__(self, margin: float = 0.2):
        super().__init__()
        self.margin = margin
    
    def forward(
        self,
        anchor: torch.Tensor,
        positive: torch.Tensor,
        negative: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            anchor: [batch_size, dim]
            positive: [batch_size, dim]  正样本
            negative: [batch_size, dim]  负样本
        """
        # 计算距离
        pos_dist = F.pairwise_distance(anchor, positive, p=2)
        neg_dist = F.pairwise_distance(anchor, negative, p=2)
        
        # 三元组损失
        loss = F.relu(pos_dist - neg_dist + self.margin).mean()
        
        return loss


class MaskedMultimodalLoss(nn.Module):
    """
    掩码多模态建模损失
    
    类似BERT的MLM，但扩展到多模态
    """
    def __init__(self, vocab_size: int, mask_token_id: int = 0):
        super().__init__()
        self.vocab_size = vocab_size
        self.mask_token_id = mask_token_id
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Args:
            predictions: [batch_size, seq_len, vocab_size]
            targets: [batch_size, seq_len]
            mask: [batch_size, seq_len] 掩码位置为1
        """
        # 只计算被掩码位置的损失
        predictions = predictions[mask]
        targets = targets[mask]
        
        loss = F.cross_entropy(predictions, targets)
        return loss


class MultimodalAlignmentLoss(nn.Module):
    """
    多模态对齐损失
    
    结合对比损失和特征对齐
    """
    def __init__(
        self,
        temperature: float = 0.07,
        alignment_weight: float = 1.0,
        uniformity_weight: float = 0.1
    ):
        super().__init__()
        self.temperature = temperature
        self.alignment_weight = alignment_weight
        self.uniformity_weight = uniformity_weight
    
    def alignment_loss(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> torch.Tensor:
        """对齐损失: 正样本对应该接近"""
        features_a = F.normalize(features_a, dim=-1)
        features_b = F.normalize(features_b, dim=-1)
        
        # 计算正样本对的距离
        similarity = (features_a * features_b).sum(dim=-1)
        loss = (1 - similarity).mean()
        
        return loss
    
    def uniformity_loss(self, features: torch.Tensor) -> torch.Tensor:
        """均匀性损失: 特征应该均匀分布在超球面上"""
        features = F.normalize(features, dim=-1)
        
        # 计算所有样本对的相似度
        similarity = features @ features.T
        
        # 排除对角线
        mask = torch.eye(features.size(0), device=features.device) == 0
        similarity = similarity[mask]
        
        # 惩罚高相似度 (鼓励分散)
        loss = similarity.pow(2).mean()
        
        return loss
    
    def forward(
        self,
        features_a: torch.Tensor,
        features_b: torch.Tensor
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            features_a: [batch_size, dim]
            features_b: [batch_size, dim]
        
        Returns:
            loss: 总损失
            loss_dict: 各组件损失的字典
        """
        align_loss = self.alignment_loss(features_a, features_b)
        uniform_loss_a = self.uniformity_loss(features_a)
        uniform_loss_b = self.uniformity_loss(features_b)
        
        total_loss = (
            self.alignment_weight * align_loss +
            self.uniformity_weight * (uniform_loss_a + uniform_loss_b)
        )
        
        loss_dict = {
            'alignment': align_loss.item(),
            'uniformity_a': uniform_loss_a.item(),
            'uniformity_b': uniform_loss_b.item(),
            'total': total_loss.item()
        }
        
        return total_loss, loss_dict


# =============================================================================
# Training Loop
# =============================================================================

class MultimodalTrainer:
    """
    多模态模型训练器
    """
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = 'cuda',
        log_interval: int = 100
    ):
        self.model = model
        self.optimizer = optimizer
        self.device = device
        self.log_interval = log_interval
        
        self.model.to(device)
        
        # 损失函数
        self.contrastive_loss = ContrastiveLoss()
        self.alignment_loss = MultimodalAlignmentLoss()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        epoch: int
    ) -> Dict[str, float]:
        """
        训练一个epoch
        
        Args:
            dataloader: 数据加载器
            epoch: 当前epoch
        
        Returns:
            metrics: 训练指标
        """
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(dataloader, desc=f"Epoch {epoch}")
        
        for batch_idx, batch in enumerate(pbar):
            # 将数据移到设备
            images = batch['images'].to(self.device) if 'images' in batch else None
            text_tokens = batch['text_tokens'].to(self.device) if 'text_tokens' in batch else None
            actions = batch['actions'].to(self.device) if 'actions' in batch else None
            
            # 前向传播
            outputs = self.model(
                images=images,
                text_tokens=text_tokens,
                actions=actions
            )
            
            # 计算损失
            loss = 0
            loss_dict = {}
            
            # 视觉-语言对比损失
            if images is not None and text_tokens is not None:
                vl_loss = self.model.compute_contrastive_loss(
                    outputs['vision'],
                    outputs['text']
                )
                loss = loss + vl_loss
                loss_dict['vl_contrastive'] = vl_loss.item()
            
            # 视觉-动作对比损失
            if images is not None and actions is not None:
                va_loss = self.contrastive_loss(
                    outputs['vision'],
                    outputs['action']
                )
                loss = loss + va_loss
                loss_dict['va_contrastive'] = va_loss.item()
            
            # 反向传播
            self.optimizer.zero_grad()
            loss.backward()
            
            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            # 记录
            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % self.log_interval == 0:
                pbar.set_postfix(loss_dict)
        
        avg_loss = total_loss / num_batches
        return {'train_loss': avg_loss}
    
    @torch.no_grad()
    def evaluate(
        self,
        dataloader: DataLoader
    ) -> Dict[str, float]:
        """
        评估模型
        
        Args:
            dataloader: 验证/测试数据加载器
        
        Returns:
            metrics: 评估指标
        """
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        # 用于计算检索指标
        all_vision_features = []
        all_text_features = []
        
        for batch in tqdm(dataloader, desc="Evaluating"):
            images = batch['images'].to(self.device) if 'images' in batch else None
            text_tokens = batch['text_tokens'].to(self.device) if 'text_tokens' in batch else None
            
            outputs = self.model(images=images, text_tokens=text_tokens)
            
            if images is not None and text_tokens is not None:
                loss = self.model.compute_contrastive_loss(
                    outputs['vision'],
                    outputs['text']
                )
                total_loss += loss.item()
                
                # 收集特征用于检索评估
                all_vision_features.append(outputs['vision'].cpu())
                all_text_features.append(outputs['text'].cpu())
            
            num_batches += 1
        
        metrics = {'val_loss': total_loss / num_batches}
        
        # 计算检索指标
        if all_vision_features:
            vision_features = torch.cat(all_vision_features, dim=0)
            text_features = torch.cat(all_text_features, dim=0)
            
            recall_at_k = self.compute_recall_at_k(vision_features, text_features, k=[1, 5, 10])
            metrics.update(recall_at_k)
        
        return metrics
    
    def compute_recall_at_k(
        self,
        vision_features: torch.Tensor,
        text_features: torch.Tensor,
        k: List[int] = [1, 5, 10]
    ) -> Dict[str, float]:
        """
        计算Recall@K指标
        
        Args:
            vision_features: [N, D]
            text_features: [N, D]
            k: 评估的K值列表
        
        Returns:
            metrics: Recall@K指标
        """
        # 归一化
        vision_features = F.normalize(vision_features, dim=-1)
        text_features = F.normalize(text_features, dim=-1)
        
        # 计算相似度矩阵
        similarity = vision_features @ text_features.T  # [N, N]
        
        metrics = {}
        
        # 图像到文本检索
        for ki in k:
            top_k_indices = similarity.topk(ki, dim=1)[1]  # [N, k]
            correct = (top_k_indices == torch.arange(len(similarity)).unsqueeze(1)).any(dim=1)
            recall = correct.float().mean().item()
            metrics[f'i2t_recall@{ki}'] = recall
        
        # 文本到图像检索
        for ki in k:
            top_k_indices = similarity.T.topk(ki, dim=1)[1]  # [N, k]
            correct = (top_k_indices == torch.arange(len(similarity)).unsqueeze(1)).any(dim=1)
            recall = correct.float().mean().item()
            metrics[f't2i_recall@{ki}'] = recall
        
        return metrics


# =============================================================================
# Data Augmentation for Multimodal
# =============================================================================

class MultimodalAugmentation:
    """
    多模态数据增强
    """
    def __init__(
        self,
        image_size: int = 224,
        modality_dropout_prob: float = 0.1
    ):
        self.image_size = image_size
        self.modality_dropout_prob = modality_dropout_prob
    
    def augment_image(self, image: torch.Tensor) -> torch.Tensor:
        """图像增强"""
        # 随机裁剪
        if torch.rand(1).item() < 0.5:
            image = self.random_crop(image)
        
        # 随机水平翻转
        if torch.rand(1).item() < 0.5:
            image = torch.flip(image, dims=[-1])
        
        # 颜色抖动
        if torch.rand(1).item() < 0.3:
            image = self.color_jitter(image)
        
        return image
    
    def random_crop(self, image: torch.Tensor, scale: Tuple[float, float] = (0.8, 1.0)) -> torch.Tensor:
        """随机裁剪"""
        _, h, w = image.shape
        scale_factor = torch.rand(1).item() * (scale[1] - scale[0]) + scale[0]
        new_h, new_w = int(h * scale_factor), int(w * scale_factor)
        
        top = torch.randint(0, h - new_h + 1, (1,)).item()
        left = torch.randint(0, w - new_w + 1, (1,)).item()
        
        cropped = image[:, top:top+new_h, left:left+new_w]
        
        # 调整回原始大小
        cropped = F.interpolate(
            cropped.unsqueeze(0),
            size=(h, w),
            mode='bilinear',
            align_corners=False
        ).squeeze(0)
        
        return cropped
    
    def color_jitter(
        self,
        image: torch.Tensor,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2
    ) -> torch.Tensor:
        """颜色抖动"""
        # 亮度
        if torch.rand(1).item() < 0.5:
            factor = torch.rand(1).item() * 2 * brightness - brightness + 1
            image = image * factor
        
        # 对比度
        if torch.rand(1).item() < 0.5:
            factor = torch.rand(1).item() * 2 * contrast - contrast + 1
            mean = image.mean()
            image = (image - mean) * factor + mean
        
        # 饱和度
        if torch.rand(1).item() < 0.5:
            factor = torch.rand(1).item() * 2 * saturation - saturation + 1
            gray = image.mean(dim=0, keepdim=True)
            image = image * factor + gray * (1 - factor)
        
        return torch.clamp(image, 0, 1)
    
    def modality_dropout(
        self,
        modalities: Dict[str, Optional[torch.Tensor]]
    ) -> Dict[str, Optional[torch.Tensor]]:
        """
        模态丢弃 (Modality Dropout)
        
        随机丢弃某些模态，增强模型对缺失模态的鲁棒性
        """
        result = {}
        for key, value in modalities.items():
            if torch.rand(1).item() < self.modality_dropout_prob:
                result[key] = None
            else:
                result[key] = value
        
        return result


# =============================================================================
# Learning Rate Scheduling
# =============================================================================

class WarmupCosineScheduler:
    """
    带预热的余弦退火学习率调度器
    """
    def __init__(
        self,
        optimizer: torch.optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0
    ):
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']
    
    def step(self, current_step: int):
        """更新学习率"""
        if current_step < self.warmup_steps:
            # 线性预热
            lr = self.base_lr * current_step / self.warmup_steps
        else:
            # 余弦退火
            progress = (current_step - self.warmup_steps) / (self.total_steps - self.warmup_steps)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        
        return lr


# =============================================================================
# Checkpoint Management
# =============================================================================

class CheckpointManager:
    """
    模型检查点管理
    """
    def __init__(
        self,
        save_dir: str = './checkpoints',
        keep_last_n: int = 3,
        keep_best: bool = True
    ):
        self.save_dir = save_dir
        self.keep_last_n = keep_last_n
        self.keep_best = keep_best
        
        self.saved_checkpoints = []
        self.best_metric = float('inf')
        
        import os
        os.makedirs(save_dir, exist_ok=True)
    
    def save(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        epoch: int,
        metrics: Dict[str, float],
        is_best: bool = False
    ):
        """保存检查点"""
        import os
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'metrics': metrics
        }
        
        # 保存最新检查点
        latest_path = os.path.join(self.save_dir, 'latest.pt')
        torch.save(checkpoint, latest_path)
        
        # 保存周期性检查点
        checkpoint_path = os.path.join(self.save_dir, f'checkpoint_epoch_{epoch}.pt')
        torch.save(checkpoint, checkpoint_path)
        self.saved_checkpoints.append(checkpoint_path)
        
        # 清理旧检查点
        if len(self.saved_checkpoints) > self.keep_last_n:
            old_checkpoint = self.saved_checkpoints.pop(0)
            if os.path.exists(old_checkpoint):
                os.remove(old_checkpoint)
        
        # 保存最佳检查点
        if is_best and self.keep_best:
            best_path = os.path.join(self.save_dir, 'best.pt')
            torch.save(checkpoint, best_path)
    
    def load(
        self,
        model: nn.Module,
        optimizer: Optional[torch.optim.Optimizer] = None,
        checkpoint_path: Optional[str] = None
    ) -> Dict:
        """加载检查点"""
        import os
        
        if checkpoint_path is None:
            checkpoint_path = os.path.join(self.save_dir, 'latest.pt')
        
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path)
        
        model.load_state_dict(checkpoint['model_state_dict'])
        
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return checkpoint


# =============================================================================
# Example Usage
# =============================================================================

def example_training_loop():
    """示例训练循环"""
    print("\n" + "=" * 60)
    print("Example: Training Loop")
    print("=" * 60)
    
    from advanced_multimodal import AdvancedMultimodalModel, MultimodalConfig
    
    # 配置
    config = MultimodalConfig(
        hidden_dim=512,
        num_layers=6,
        num_heads=8
    )
    
    # 模型
    model = AdvancedMultimodalModel(config)
    
    # 优化器
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)
    
    # 训练器
    trainer = MultimodalTrainer(model, optimizer, device='cpu')
    
    # 模拟数据
    class DummyDataset:
        def __init__(self, size=100):
            self.size = size
        
        def __len__(self):
            return self.size
        
        def __getitem__(self, idx):
            return {
                'images': torch.randn(3, 224, 224),
                'text_tokens': torch.randint(0, config.vocab_size, (config.max_text_length,)),
                'actions': torch.randn(config.action_dim)
            }
    
    from torch.utils.data import DataLoader
    dataset = DummyDataset()
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)
    
    # 训练一个epoch
    metrics = trainer.train_epoch(dataloader, epoch=1)
    print(f"\nTraining metrics: {metrics}")
    
    # 评估
    val_metrics = trainer.evaluate(dataloader)
    print(f"Validation metrics: {val_metrics}")
    
    print("\n✓ Training example complete!")


def example_loss_functions():
    """示例损失函数"""
    print("\n" + "=" * 60)
    print("Example: Loss Functions")
    print("=" * 60)
    
    batch_size = 32
    dim = 512
    
    # 模拟特征
    features_a = torch.randn(batch_size, dim)
    features_b = torch.randn(batch_size, dim)
    
    # 1. 对比损失
    contrastive_loss = ContrastiveLoss(temperature=0.07)
    loss = contrastive_loss(features_a, features_b)
    print(f"\n1. Contrastive Loss: {loss.item():.4f}")
    
    # 2. 对齐损失
    alignment_loss = MultimodalAlignmentLoss()
    loss, loss_dict = alignment_loss(features_a, features_b)
    print(f"\n2. Alignment Loss: {loss.item():.4f}")
    print(f"   Components: {loss_dict}")
    
    # 3. 三元组损失
    triplet_loss = TripletLoss(margin=0.2)
    anchor = torch.randn(batch_size, dim)
    positive = torch.randn(batch_size, dim)
    negative = torch.randn(batch_size, dim)
    loss = triplet_loss(anchor, positive, negative)
    print(f"\n3. Triplet Loss: {loss.item():.4f}")
    
    print("\n✓ Loss functions example complete!")


def example_data_augmentation():
    """示例数据增强"""
    print("\n" + "=" * 60)
    print("Example: Data Augmentation")
    print("=" * 60)
    
    aug = MultimodalAugmentation()
    
    # 图像增强
    image = torch.randn(3, 224, 224)
    augmented = aug.augment_image(image)
    print(f"\n1. Image augmentation: {image.shape} -> {augmented.shape}")
    
    # 模态丢弃
    modalities = {
        'vision': torch.randn(1, 512),
        'text': torch.randn(1, 512),
        'audio': torch.randn(1, 512)
    }
    dropped = aug.modality_dropout(modalities)
    dropped_modalities = [k for k, v in dropped.items() if v is None]
    print(f"\n2. Modality dropout: dropped {dropped_modalities}")
    
    print("\n✓ Data augmentation example complete!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Multimodal Training Utilities")
    print("=" * 60)
    
    example_loss_functions()
    example_data_augmentation()
    example_training_loop()
    
    print("\n" + "=" * 60)
    print("All examples completed successfully!")
    print("=" * 60)
