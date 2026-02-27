"""
Multimodal Inference Examples
==============================

演示如何使用AdvancedMultimodalModel进行多模态推理任务

包括:
1. 视觉-语言理解 (图像描述、视觉问答)
2. 动作预测 (机器人控制)
3. 跨模态检索 (图像-文本匹配)
4. 多模态融合推理
"""

import torch
import torch.nn.functional as F
from typing import List, Tuple, Optional
import numpy as np

from advanced_multimodal import (
    AdvancedMultimodalModel,
    VLAModel,
    MultimodalConfig,
    CrossAttention,
    BidirectionalCrossAttention
)


# =============================================================================
# Example 1: Vision-Language Understanding
# =============================================================================

class VisionLanguageInference:
    """
    视觉-语言理解推理
    
    任务:
    - 图像描述生成
    - 视觉问答
    - 图像-文本相似度计算
    """
    
    def __init__(self, model: AdvancedMultimodalModel, config: MultimodalConfig):
        self.model = model
        self.config = config
        self.model.eval()
    
    def compute_image_text_similarity(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor
    ) -> torch.Tensor:
        """
        计算图像-文本相似度
        
        Args:
            images: [batch_size, 3, H, W]
            text_tokens: [batch_size, seq_len]
        
        Returns:
            similarity: [batch_size, batch_size] 相似度矩阵
        """
        with torch.no_grad():
            outputs = self.model(images=images, text_tokens=text_tokens)
            
            vision_feat = outputs['vision']  # [B, D]
            text_feat = outputs['text']      # [B, D]
            
            # 计算余弦相似度
            similarity = vision_feat @ text_feat.T
            
        return similarity
    
    def find_best_matching_text(
        self,
        image: torch.Tensor,
        candidate_texts: List[torch.Tensor]
    ) -> int:
        """
        为给定图像找到最匹配的文本描述
        
        Args:
            image: [1, 3, H, W]
            candidate_texts: List of [seq_len] tensors
        
        Returns:
            best_idx: 最佳匹配文本的索引
        """
        # 扩展图像以匹配候选文本数量
        num_candidates = len(candidate_texts)
        images = image.repeat(num_candidates, 1, 1, 1)
        
        # 堆叠候选文本
        text_tokens = torch.stack(candidate_texts, dim=0)
        
        # 计算相似度
        similarity = self.compute_image_text_similarity(images, text_tokens)
        
        # 对角线元素是匹配的相似度
        matched_similarity = torch.diag(similarity)
        
        # 找到最匹配的
        best_idx = matched_similarity.argmax().item()
        
        return best_idx
    
    def zero_shot_classify(
        self,
        image: torch.Tensor,
        class_names: List[str],
        text_tokenizer=None
    ) -> Tuple[int, torch.Tensor]:
        """
        零样本图像分类
        
        Args:
            image: [1, 3, H, W]
            class_names: 类别名称列表
            text_tokenizer: 文本tokenizer函数
        
        Returns:
            predicted_class: 预测的类别索引
            probabilities: 各类别的概率
        """
        # 构建提示模板
        prompts = [f"a photo of a {name}" for name in class_names]
        
        # 这里简化处理，实际应该使用tokenizer
        # 假设prompts已经被tokenized
        if text_tokenizer is None:
            # 使用随机token作为示例
            text_tokens = torch.randint(
                0, self.config.vocab_size, 
                (len(prompts), self.config.max_text_length)
            )
        else:
            text_tokens = text_tokenizer(prompts)
        
        # 扩展图像
        images = image.repeat(len(prompts), 1, 1, 1)
        
        # 计算相似度
        similarity = self.compute_image_text_similarity(images, text_tokens)
        
        # 对角线元素是匹配的相似度
        logits = torch.diag(similarity)
        
        # 计算概率
        probabilities = F.softmax(logits, dim=0)
        predicted_class = probabilities.argmax().item()
        
        return predicted_class, probabilities


# =============================================================================
# Example 2: Action Prediction for Robotics
# =============================================================================

class ActionPredictionInference:
    """
    机器人动作预测推理
    
    基于视觉观察和语言指令预测机器人动作
    """
    
    def __init__(self, vla_model: VLAModel, config: MultimodalConfig):
        self.model = vla_model
        self.config = config
        self.model.eval()
    
    def predict_action(
        self,
        image: torch.Tensor,
        instruction_tokens: torch.Tensor,
        history_images: Optional[List[torch.Tensor]] = None
    ) -> torch.Tensor:
        """
        预测机器人动作
        
        Args:
            image: [1, 3, H, W] 当前视觉观察
            instruction_tokens: [1, seq_len] 语言指令
            history_images: 历史观察图像列表 (可选)
        
        Returns:
            action: [1, action_dim] 预测的动作
        """
        with torch.no_grad():
            # 如果有历史图像，可以构建序列
            if history_images:
                # 这里简化处理，实际应该使用历史信息
                pass
            
            # VLA前向传播
            outputs = self.model(
                images=image,
                text_tokens=instruction_tokens,
                task="action_prediction"
            )
            
            action_logits = outputs['action_logits']  # [1, action_bins]
            
            # 从离散表示恢复连续动作
            # 这里简化处理，实际应该使用反量化
            action_bins = action_logits.argmax(dim=-1).float()
            
            # 将bin索引映射到连续值 [-1, 1]
            action = (action_bins / (self.config.action_bins - 1)) * 2 - 1
            
            # 扩展为完整动作维度
            action = action.unsqueeze(-1).repeat(1, self.config.action_dim)
        
        return action
    
    def predict_action_sequence(
        self,
        image: torch.Tensor,
        instruction_tokens: torch.Tensor,
        horizon: int = 10
    ) -> torch.Tensor:
        """
        预测动作序列 (开环控制)
        
        Args:
            image: [1, 3, H, W]
            instruction_tokens: [1, seq_len]
            horizon: 预测步数
        
        Returns:
            actions: [horizon, action_dim] 动作序列
        """
        actions = []
        current_image = image
        
        for _ in range(horizon):
            action = self.predict_action(current_image, instruction_tokens)
            actions.append(action)
            
            # 在实际应用中，这里应该使用世界模型预测下一状态
            # 这里简化处理，假设图像不变
        
        return torch.cat(actions, dim=0)


# =============================================================================
# Example 3: Cross-Modal Retrieval
# =============================================================================

class CrossModalRetrieval:
    """
    跨模态检索
    
    图像到文本检索、文本到图像检索
    """
    
    def __init__(self, model: AdvancedMultimodalModel):
        self.model = model
        self.model.eval()
        
        # 数据库
        self.image_features = []
        self.text_features = []
        self.image_ids = []
        self.text_ids = []
    
    def add_to_database(
        self,
        images: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        image_ids: Optional[List] = None,
        text_ids: Optional[List] = None
    ):
        """
        添加数据到检索数据库
        
        Args:
            images: [N, 3, H, W]
            text_tokens: [M, seq_len]
            image_ids: 图像ID列表
            text_ids: 文本ID列表
        """
        with torch.no_grad():
            if images is not None:
                outputs = self.model(images=images)
                features = outputs['vision'].cpu()
                self.image_features.append(features)
                if image_ids:
                    self.image_ids.extend(image_ids)
            
            if text_tokens is not None:
                outputs = self.model(text_tokens=text_tokens)
                features = outputs['text'].cpu()
                self.text_features.append(features)
                if text_ids:
                    self.text_ids.extend(text_ids)
    
    def image_to_text_retrieval(
        self,
        query_image: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        图像到文本检索
        
        Args:
            query_image: [1, 3, H, W]
            top_k: 返回最相似的top_k个结果
        
        Returns:
            results: List of (text_id, similarity_score)
        """
        if not self.text_features:
            return []
        
        with torch.no_grad():
            # 编码查询图像
            outputs = self.model(images=query_image)
            query_feat = outputs['vision']  # [1, D]
            
            # 合并所有文本特征
            all_text_features = torch.cat(self.text_features, dim=0)  # [N, D]
            
            # 计算相似度
            similarity = query_feat @ all_text_features.T  # [1, N]
            similarity = similarity.squeeze(0)  # [N]
            
            # 获取top-k
            top_scores, top_indices = similarity.topk(top_k)
            
            results = [
                (self.text_ids[idx], score.item())
                for idx, score in zip(top_indices, top_scores)
            ]
        
        return results
    
    def text_to_image_retrieval(
        self,
        query_text: torch.Tensor,
        top_k: int = 5
    ) -> List[Tuple[int, float]]:
        """
        文本到图像检索
        
        Args:
            query_text: [1, seq_len]
            top_k: 返回最相似的top_k个结果
        >
        Returns:
            results: List of (image_id, similarity_score)
        """
        if not self.image_features:
            return []
        
        with torch.no_grad():
            # 编码查询文本
            outputs = self.model(text_tokens=query_text)
            query_feat = outputs['text']  # [1, D]
            
            # 合并所有图像特征
            all_image_features = torch.cat(self.image_features, dim=0)  # [M, D]
            
            # 计算相似度
            similarity = query_feat @ all_image_features.T  # [1, M]
            similarity = similarity.squeeze(0)  # [M]
            
            # 获取top-k
            top_scores, top_indices = similarity.topk(top_k)
            
            results = [
                (self.image_ids[idx], score.item())
                for idx, score in zip(top_indices, top_scores)
            ]
        
        return results


# =============================================================================
# Example 4: Multimodal Fusion Reasoning
# =============================================================================

class MultimodalFusionReasoning:
    """
    多模态融合推理
    
    结合视觉、语言、动作信息进行复杂推理
    """
    
    def __init__(self, model: AdvancedMultimodalModel):
        self.model = model
        self.model.eval()
    
    def fused_representation_learning(
        self,
        images: torch.Tensor,
        text_tokens: torch.Tensor,
        actions: torch.Tensor
    ) -> torch.Tensor:
        """
        学习统一的多模态表示
        
        Args:
            images: [batch_size, 3, H, W]
            text_tokens: [batch_size, seq_len]
            actions: [batch_size, action_dim]
        
        Returns:
            fused_rep: [batch_size, hidden_dim] 融合表示
        """
        with torch.no_grad():
            outputs = self.model(
                images=images,
                text_tokens=text_tokens,
                actions=actions
            )
            
            fused_rep = outputs['fused']
        
        return fused_rep
    
    def multimodal_similarity(
        self,
        images1: torch.Tensor,
        text1: torch.Tensor,
        actions1: torch.Tensor,
        images2: torch.Tensor,
        text2: torch.Tensor,
        actions2: torch.Tensor
    ) -> torch.Tensor:
        """
        计算两组多模态数据的相似度
        
        用于判断两个场景/任务是否相似
        """
        fused1 = self.fused_representation_learning(images1, text1, actions1)
        fused2 = self.fused_representation_learning(images2, text2, actions2)
        
        # 余弦相似度
        similarity = F.cosine_similarity(fused1, fused2, dim=-1)
        
        return similarity
    
    def missing_modality_inference(
        self,
        images: Optional[torch.Tensor] = None,
        text_tokens: Optional[torch.Tensor] = None,
        actions: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        """
        缺失模态推理
        
        在部分模态缺失的情况下进行推理
        """
        with torch.no_grad():
            outputs = self.model(
                images=images,
                text_tokens=text_tokens,
                actions=actions
            )
        
        return outputs


# =============================================================================
# Demo and Testing
# =============================================================================

def demo_vision_language_tasks():
    """演示视觉-语言任务"""
    print("\n" + "=" * 60)
    print("Demo: Vision-Language Tasks")
    print("=" * 60)
    
    from advanced_multimodal import MultimodalConfig, AdvancedMultimodalModel
    
    config = MultimodalConfig(hidden_dim=512, num_layers=6)
    model = AdvancedMultimodalModel(config)
    
    vl_inference = VisionLanguageInference(model, config)
    
    # 模拟输入
    batch_size = 4
    images = torch.randn(batch_size, 3, 224, 224)
    text_tokens = torch.randint(0, config.vocab_size, (batch_size, config.max_text_length))
    
    # 1. 图像-文本相似度
    similarity = vl_inference.compute_image_text_similarity(images, text_tokens)
    print(f"\n1. Image-Text Similarity Matrix:")
    print(f"   Shape: {similarity.shape}")
    print(f"   Mean similarity: {similarity.mean().item():.4f}")
    
    # 2. 零样本分类
    image = images[0:1]
    class_names = ["cat", "dog", "bird", "car"]
    pred_class, probs = vl_inference.zero_shot_classify(image, class_names)
    print(f"\n2. Zero-shot Classification:")
    print(f"   Predicted class: {class_names[pred_class]}")
    print(f"   Probabilities: {probs.tolist()}")
    
    print("\n✓ Vision-language tasks demo complete!")


def demo_action_prediction():
    """演示动作预测任务"""
    print("\n" + "=" * 60)
    print("Demo: Action Prediction")
    print("=" * 60)
    
    from advanced_multimodal import MultimodalConfig, VLAModel
    
    config = MultimodalConfig(hidden_dim=512, num_layers=6)
    vla_model = VLAModel(config)
    
    action_inference = ActionPredictionInference(vla_model, config)
    
    # 模拟输入
    image = torch.randn(1, 3, 224, 224)
    instruction = torch.randint(0, config.vocab_size, (1, config.max_text_length))
    
    # 预测动作
    action = action_inference.predict_action(image, instruction)
    print(f"\n1. Single Action Prediction:")
    print(f"   Predicted action shape: {action.shape}")
    print(f"   Action values: {action[0].tolist()}")
    
    # 预测动作序列
    action_seq = action_inference.predict_action_sequence(image, instruction, horizon=5)
    print(f"\n2. Action Sequence Prediction:")
    print(f"   Sequence shape: {action_seq.shape}")
    
    print("\n✓ Action prediction demo complete!")


def demo_cross_modal_retrieval():
    """演示跨模态检索"""
    print("\n" + "=" * 60)
    print("Demo: Cross-Modal Retrieval")
    print("=" * 60)
    
    from advanced_multimodal import MultimodalConfig, AdvancedMultimodalModel
    
    config = MultimodalConfig(hidden_dim=512, num_layers=6)
    model = AdvancedMultimodalModel(config)
    
    retrieval = CrossModalRetrieval(model)
    
    # 构建数据库
    num_images = 100
    num_texts = 100
    
    images = torch.randn(num_images, 3, 224, 224)
    text_tokens = torch.randint(0, config.vocab_size, (num_texts, config.max_text_length))
    
    retrieval.add_to_database(
        images=images,
        text_tokens=text_tokens,
        image_ids=list(range(num_images)),
        text_ids=list(range(num_texts))
    )
    
    # 图像到文本检索
    query_image = torch.randn(1, 3, 224, 224)
    text_results = retrieval.image_to_text_retrieval(query_image, top_k=5)
    print(f"\n1. Image-to-Text Retrieval:")
    print(f"   Top-5 matching text IDs: {[r[0] for r in text_results]}")
    print(f"   Similarity scores: {[f'{r[1]:.4f}' for r in text_results]}")
    
    # 文本到图像检索
    query_text = torch.randint(0, config.vocab_size, (1, config.max_text_length))
    image_results = retrieval.text_to_image_retrieval(query_text, top_k=5)
    print(f"\n2. Text-to-Image Retrieval:")
    print(f"   Top-5 matching image IDs: {[r[0] for r in image_results]}")
    print(f"   Similarity scores: {[f'{r[1]:.4f}' for r in image_results]}")
    
    print("\n✓ Cross-modal retrieval demo complete!")


def demo_multimodal_fusion():
    """演示多模态融合"""
    print("\n" + "=" * 60)
    print("Demo: Multimodal Fusion Reasoning")
    print("=" * 60)
    
    from advanced_multimodal import MultimodalConfig, AdvancedMultimodalModel
    
    # 测试不同融合策略
    fusion_types = ["early", "late", "cross_attention", "hybrid"]
    
    for fusion_type in fusion_types:
        print(f"\n--- Testing {fusion_type} fusion ---")
        
        config = MultimodalConfig(
            hidden_dim=512,
            num_layers=6,
            fusion_type=fusion_type
        )
        model = AdvancedMultimodalModel(config)
        
        # 完整模态输入
        batch_size = 4
        images = torch.randn(batch_size, 3, 224, 224)
        text_tokens = torch.randint(0, config.vocab_size, (batch_size, config.max_text_length))
        actions = torch.randn(batch_size, config.action_dim)
        
        outputs = model(images=images, text_tokens=text_tokens, actions=actions)
        print(f"   Fused representation shape: {outputs['fused'].shape}")
        
        # 缺失模态输入
        outputs_missing = model(images=images, text_tokens=text_tokens)
        if 'fused' in outputs_missing:
            print(f"   Fused (missing action) shape: {outputs_missing['fused'].shape}")
    
    print("\n✓ Multimodal fusion demo complete!")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("Advanced Multimodal Inference Examples")
    print("=" * 60)
    
    # 运行所有演示
    demo_vision_language_tasks()
    demo_action_prediction()
    demo_cross_modal_retrieval()
    demo_multimodal_fusion()
    
    print("\n" + "=" * 60)
    print("All demos completed successfully!")
    print("=" * 60)
