#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
XAI Explainer - AI可解释性工具包
================================

提供以下XAI方法的实现：
1. 注意力可视化 (Attention Visualization)
2. LIME (Local Interpretable Model-agnostic Explanations)
3. SHAP (SHapley Additive exPlanations) - 简化版
4. CAV/TCAV (Concept Activation Vectors / Testing with CAV)
5. 反事实解释 (Counterfactual Explanations)

作者: XAI Research Team
日期: 2026-02-27
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple, Callable, Optional, Union
from dataclasses import dataclass
from abc import ABC, abstractmethod
import warnings
warnings.filterwarnings('ignore')

# 尝试导入深度学习库
try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Some features will be disabled.")

try:
    from sklearn.linear_model import Ridge, LogisticRegression
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: scikit-learn not available. Some features will be disabled.")


# =============================================================================
# 配置和工具类
# =============================================================================

@dataclass
class ExplanationResult:
    """解释结果容器"""
    method: str
    feature_importance: Optional[np.ndarray] = None
    concept_importance: Optional[Dict[str, float]] = None
    attention_weights: Optional[np.ndarray] = None
    counterfactual: Optional[np.ndarray] = None
    visualization: Optional[plt.Figure] = None
    metadata: Optional[Dict] = None


class BaseExplainer(ABC):
    """解释器基类"""
    
    @abstractmethod
    def explain(self, model, input_data, **kwargs) -> ExplanationResult:
        """生成解释"""
        pass
    
    @abstractmethod
    def visualize(self, explanation: ExplanationResult, **kwargs) -> plt.Figure:
        """可视化解释"""
        pass


# =============================================================================
# 1. 注意力可视化
# =============================================================================

class AttentionVisualizer(BaseExplainer):
    """
    注意力可视化器
    
    支持Transformer模型的注意力权重提取和可视化
    """
    
    def __init__(self, layer_names: Optional[List[str]] = None):
        self.layer_names = layer_names
        self.attention_weights = []
        self.hooks = []
        
    def _register_hooks(self, model):
        """注册forward hook捕获注意力权重"""
        if not TORCH_AVAILABLE:
            return
        
        self.attention_weights = []
        
        def hook_fn(module, input, output):
            """捕获注意力权重的hook函数"""
            # 假设output是(attention_output, attention_weights)元组
            if isinstance(output, tuple) and len(output) > 1:
                attn_weights = output[1]
                if attn_weights is not None:
                    self.attention_weights.append(attn_weights.detach().cpu())
        
        # 自动查找注意力层
        for name, module in model.named_modules():
            if 'attention' in name.lower() or 'attn' in name.lower():
                if self.layer_names is None or name in self.layer_names:
                    hook = module.register_forward_hook(hook_fn)
                    self.hooks.append(hook)
    
    def _remove_hooks(self):
        """移除所有注册的hooks"""
        for hook in self.hooks:
            hook.remove()
        self.hooks = []
    
    def extract_attention(self, model: nn.Module, input_data: torch.Tensor) -> List[np.ndarray]:
        """
        提取注意力权重
        
        Args:
            model: PyTorch模型
            input_data: 输入张量
            
        Returns:
            注意力权重列表
        """
        if not TORCH_AVAILABLE:
            raise RuntimeError("PyTorch is required for attention visualization")
        
        model.eval()
        self._register_hooks(model)
        
        with torch.no_grad():
            _ = model(input_data)
        
        self._remove_hooks()
        
        return [w.numpy() for w in self.attention_weights]
    
    def explain(self, model, input_data, target_layer: int = 0, **kwargs) -> ExplanationResult:
        """生成注意力解释"""
        attention_weights = self.extract_attention(model, input_data)
        
        if len(attention_weights) == 0:
            raise ValueError("No attention weights captured. Check model architecture.")
        
        # 选择目标层的注意力权重
        target_attention = attention_weights[target_layer]
        
        # 平均所有注意力头
        if target_attention.ndim >= 3:
            target_attention = target_attention.mean(axis=tuple(range(target_attention.ndim - 2)))
        
        return ExplanationResult(
            method="Attention Visualization",
            attention_weights=target_attention,
            metadata={
                "num_layers": len(attention_weights),
                "target_layer": target_layer,
                "attention_shape": target_attention.shape
            }
        )
    
    def visualize(self, explanation: ExplanationResult, 
                  tokens: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (10, 8),
                  cmap: str = 'viridis',
                  **kwargs) -> plt.Figure:
        """
        可视化注意力热图
        
        Args:
            explanation: 解释结果
            tokens: 输入token列表（用于标注）
            figsize: 图像大小
            cmap: 颜色映射
        """
        attn_weights = explanation.attention_weights
        if attn_weights is None:
            raise ValueError("No attention weights in explanation")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 绘制热图
        im = ax.imshow(attn_weights, cmap=cmap, aspect='auto')
        
        # 设置坐标轴
        n_tokens = attn_weights.shape[0]
        if tokens is not None and len(tokens) == n_tokens:
            ax.set_xticks(range(n_tokens))
            ax.set_yticks(range(n_tokens))
            ax.set_xticklabels(tokens, rotation=45, ha='right')
            ax.set_yticklabels(tokens)
        else:
            ax.set_xticks(range(n_tokens))
            ax.set_yticks(range(n_tokens))
        
        ax.set_xlabel('Key Position')
        ax.set_ylabel('Query Position')
        ax.set_title('Attention Heatmap')
        
        # 添加颜色条
        plt.colorbar(im, ax=ax, label='Attention Weight')
        
        plt.tight_layout()
        return fig


# =============================================================================
# 2. LIME 实现
# =============================================================================

class LIMEExplainer(BaseExplainer):
    """
    LIME解释器简化实现
    
    在预测点附近采样，用线性模型近似局部行为
    """
    
    def __init__(self, 
                 kernel_width: float = None,
                 n_samples: int = 1000,
                 feature_selection: str = 'auto',
                 discretize_continuous: bool = True):
        """
        初始化LIME解释器
        
        Args:
            kernel_width: 核函数宽度
            n_samples: 扰动样本数量
            feature_selection: 特征选择方法
            discretize_continuous: 是否离散化连续特征
        """
        self.kernel_width = kernel_width
        self.n_samples = n_samples
        self.feature_selection = feature_selection
        self.discretize_continuous = discretize_continuous
        self.scaler = StandardScaler() if SKLEARN_AVAILABLE else None
        
    def _kernel(self, distances: np.ndarray) -> np.ndarray:
        """指数核函数"""
        if self.kernel_width is None:
            self.kernel_width = np.sqrt(distances.shape[1]) * 0.75
        return np.exp(-distances ** 2 / (2 * self.kernel_width ** 2))
    
    def _generate_perturbations(self, 
                                instance: np.ndarray,
                                feature_stats: Optional[Dict] = None) -> np.ndarray:
        """
        生成扰动样本
        
        Args:
            instance: 原始样本
            feature_stats: 特征统计信息
            
        Returns:
            扰动样本矩阵
        """
        n_features = len(instance)
        
        # 生成二值扰动矩阵
        perturbations = np.random.binomial(1, 0.5, size=(self.n_samples, n_features))
        
        # 将二值扰动转换为实际值
        if feature_stats is not None:
            # 使用特征统计信息生成扰动
            perturbed_data = np.zeros((self.n_samples, n_features))
            for i in range(n_features):
                mean = feature_stats.get('mean', {}).get(i, instance[i])
                std = feature_stats.get('std', {}).get(i, abs(instance[i]) * 0.1)
                perturbed_data[:, i] = np.where(
                    perturbations[:, i],
                    instance[i],
                    np.random.normal(mean, std, self.n_samples)
                )
        else:
            # 简单扰动：在原始值附近添加噪声
            noise = np.random.randn(self.n_samples, n_features) * 0.1
            perturbed_data = instance + noise * perturbations
        
        # 包含原始样本
        perturbed_data[0] = instance
        
        return perturbed_data
    
    def explain(self, 
                model: Callable,
                input_data: np.ndarray,
                feature_names: Optional[List[str]] = None,
                class_names: Optional[List[str]] = None,
                **kwargs) -> ExplanationResult:
        """
        生成LIME解释
        
        Args:
            model: 预测模型（可调用对象）
            input_data: 输入样本
            feature_names: 特征名称列表
            class_names: 类别名称列表
            
        Returns:
            解释结果
        """
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for LIME")
        
        # 确保输入是1D数组
        if input_data.ndim > 1:
            input_data = input_data.flatten()
        
        n_features = len(input_data)
        
        # 生成扰动样本
        perturbed_data = self._generate_perturbations(input_data)
        
        # 获取模型预测
        predictions = model(perturbed_data)
        
        # 如果是分类问题，获取预测概率
        if predictions.ndim > 1:
            predictions = predictions[:, 1] if predictions.shape[1] == 2 else predictions[:, 0]
        
        # 计算与原始样本的距离
        distances = pairwise_distances(
            perturbed_data, 
            input_data.reshape(1, -1),
            metric='euclidean'
        ).flatten()
        
        # 计算样本权重
        weights = self._kernel(distances)
        
        # 训练加权线性模型
        # 标准化特征
        X_scaled = self.scaler.fit_transform(perturbed_data)
        
        # 使用Ridge回归（带正则化的线性回归）
        local_model = Ridge(alpha=1.0)
        local_model.fit(X_scaled, predictions, sample_weight=weights)
        
        # 获取特征重要性（系数）
        feature_importance = local_model.coef_
        
        # 生成特征名称
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        return ExplanationResult(
            method="LIME",
            feature_importance=feature_importance,
            metadata={
                "feature_names": feature_names,
                "class_names": class_names,
                "local_model_score": local_model.score(X_scaled, predictions, sample_weight=weights),
                "intercept": local_model.intercept_
            }
        )
    
    def visualize(self, 
                  explanation: ExplanationResult,
                  top_k: int = 10,
                  figsize: Tuple[int, int] = (10, 6),
                  **kwargs) -> plt.Figure:
        """
        可视化LIME解释
        
        Args:
            explanation: 解释结果
            top_k: 显示前k个重要特征
            figsize: 图像大小
        """
        importance = explanation.feature_importance
        feature_names = explanation.metadata.get("feature_names", [f"F{i}" for i in range(len(importance))])
        
        # 获取绝对重要性最高的特征
        abs_importance = np.abs(importance)
        top_indices = np.argsort(abs_importance)[-top_k:][::-1]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if importance[i] > 0 else 'red' for i in top_indices]
        y_pos = np.arange(len(top_indices))
        
        ax.barh(y_pos, importance[top_indices], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels([feature_names[i] for i in top_indices])
        ax.invert_yaxis()
        ax.set_xlabel('Importance')
        ax.set_title(f'LIME Feature Importance (Top {top_k})')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig


# =============================================================================
# 3. SHAP 简化实现
# =============================================================================

class SHAPExplainer(BaseExplainer):
    """
    SHAP解释器简化实现
    
    基于采样的Shapley值估计
    """
    
    def __init__(self, n_samples: int = 100, background_data: Optional[np.ndarray] = None):
        """
        初始化SHAP解释器
        
        Args:
            n_samples: 采样次数
            background_data: 背景数据集（用于计算期望）
        """
        self.n_samples = n_samples
        self.background_data = background_data
        
    def _compute_shapley_value(self,
                               model: Callable,
                               instance: np.ndarray,
                               feature_idx: int,
                               background: np.ndarray) -> float:
        """
        计算单个特征的Shapley值
        
        Args:
            model: 预测模型
            instance: 目标样本
            feature_idx: 特征索引
            background: 背景样本
            
        Returns:
            Shapley值
        """
        n_features = len(instance)
        shap_value = 0.0
        
        # 蒙特卡洛采样估计
        for _ in range(self.n_samples):
            # 随机选择特征子集
            subset = np.random.choice([0, 1], size=n_features, p=[0.5, 0.5])
            
            # 创建两个样本：包含和不包含目标特征
            sample_with = background.copy()
            sample_without = background.copy()
            
            for j in range(n_features):
                if subset[j] or j == feature_idx:
                    sample_with[j] = instance[j]
                if subset[j] and j != feature_idx:
                    sample_without[j] = instance[j]
            
            # 计算边际贡献
            pred_with = model(sample_with.reshape(1, -1))[0]
            pred_without = model(sample_without.reshape(1, -1))[0]
            
            if isinstance(pred_with, np.ndarray):
                pred_with = pred_with[0]
            if isinstance(pred_without, np.ndarray):
                pred_without = pred_without[0]
            
            shap_value += (pred_with - pred_without)
        
        return shap_value / self.n_samples
    
    def explain(self,
                model: Callable,
                input_data: np.ndarray,
                feature_names: Optional[List[str]] = None,
                **kwargs) -> ExplanationResult:
        """
        生成SHAP解释
        
        Args:
            model: 预测模型
            input_data: 输入样本
            feature_names: 特征名称
            
        Returns:
            解释结果
        """
        if input_data.ndim > 1:
            input_data = input_data.flatten()
        
        n_features = len(input_data)
        
        # 使用背景数据或生成随机背景
        if self.background_data is None:
            background = np.zeros(n_features)
        else:
            background = self.background_data.mean(axis=0) if self.background_data.ndim > 1 else self.background_data
        
        # 计算每个特征的Shapley值
        shap_values = np.zeros(n_features)
        for i in range(n_features):
            shap_values[i] = self._compute_shapley_value(model, input_data, i, background)
        
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        return ExplanationResult(
            method="SHAP",
            feature_importance=shap_values,
            metadata={
                "feature_names": feature_names,
                "base_value": model(background.reshape(1, -1))[0],
                "prediction": model(input_data.reshape(1, -1))[0]
            }
        )
    
    def visualize(self,
                  explanation: ExplanationResult,
                  figsize: Tuple[int, int] = (10, 6),
                  **kwargs) -> plt.Figure:
        """
        可视化SHAP解释（瀑布图风格）
        """
        shap_values = explanation.feature_importance
        feature_names = explanation.metadata.get("feature_names", [f"F{i}" for i in range(len(shap_values))])
        base_value = explanation.metadata.get("base_value", 0)
        
        # 按绝对值排序
        indices = np.argsort(np.abs(shap_values))[::-1]
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # 计算累积值
        cumulative = base_value
        positions = []
        for i in indices:
            positions.append(cumulative)
            cumulative += shap_values[i]
        
        colors = ['green' if shap_values[i] > 0 else 'red' for i in indices]
        
        ax.barh(range(len(indices)), [shap_values[i] for i in indices], 
                left=positions, color=colors, alpha=0.7)
        ax.set_yticks(range(len(indices)))
        ax.set_yticklabels([feature_names[i] for i in indices])
        ax.invert_yaxis()
        ax.set_xlabel('SHAP Value (impact on prediction)')
        ax.set_title('SHAP Feature Importance')
        ax.axvline(x=base_value, color='black', linestyle='--', label='Base Value')
        
        plt.tight_layout()
        return fig


# =============================================================================
# 4. CAV/TCAV 实现
# =============================================================================

class CAVExplainer(BaseExplainer):
    """
    Concept Activation Vector (CAV) 解释器
    
    量化神经网络内部表示与人类概念之间的关系
    """
    
    def __init__(self, 
                 layer_name: str,
                 concept_data: Dict[str, np.ndarray],
                 random_data: np.ndarray,
                 cav_cache: Optional[Dict] = None):
        """
        初始化CAV解释器
        
        Args:
            layer_name: 目标层名称
            concept_data: 概念数据集 {概念名: 样本数组}
            random_data: 随机对照样本
            cav_cache: CAV缓存字典
        """
        self.layer_name = layer_name
        self.concept_data = concept_data
        self.random_data = random_data
        self.cav_cache = cav_cache or {}
        self.cavs = {}
        
    def _extract_activations(self, 
                             model: Callable,
                             data: np.ndarray,
                             layer_name: str) -> np.ndarray:
        """
        提取指定层的激活值
        
        Args:
            model: 模型或激活提取函数
            data: 输入数据
            layer_name: 层名称
            
        Returns:
            激活值数组
        """
        activations = []
        
        if TORCH_AVAILABLE and isinstance(model, nn.Module):
            # PyTorch模型
            activation = {}
            
            def get_activation(name):
                def hook(model, input, output):
                    activation[name] = output.detach()
                return hook
            
            # 注册hook
            for name, module in model.named_modules():
                if name == layer_name:
                    handle = module.register_forward_hook(get_activation(layer_name))
                    break
            
            model.eval()
            with torch.no_grad():
                if isinstance(data, np.ndarray):
                    data = torch.FloatTensor(data)
                _ = model(data)
            
            handle.remove()
            
            # 展平激活值
            acts = activation[layer_name].cpu().numpy()
            activations = acts.reshape(acts.shape[0], -1)
        else:
            # 假设model返回层激活的函数
            activations = model(data, layer=layer_name)
            if activations.ndim > 2:
                activations = activations.reshape(activations.shape[0], -1)
        
        return activations
    
    def learn_cav(self,
                  model: Callable,
                  concept_name: str,
                  use_cache: bool = True) -> np.ndarray:
        """
        学习概念激活向量
        
        Args:
            model: 模型
            concept_name: 概念名称
            use_cache: 是否使用缓存
            
        Returns:
            CAV向量
        """
        if use_cache and concept_name in self.cav_cache:
            return self.cav_cache[concept_name]
        
        if not SKLEARN_AVAILABLE:
            raise RuntimeError("scikit-learn is required for CAV")
        
        # 获取概念样本
        concept_samples = self.concept_data[concept_name]
        
        # 提取激活值
        concept_activations = self._extract_activations(model, concept_samples, self.layer_name)
        random_activations = self._extract_activations(model, self.random_data, self.layer_name)
        
        # 准备训练数据
        X = np.vstack([concept_activations, random_activations])
        y = np.hstack([np.ones(len(concept_activations)), np.zeros(len(random_activations))])
        
        # 训练二分类器
        classifier = LogisticRegression(max_iter=1000, random_state=42)
        classifier.fit(X, y)
        
        # CAV是分类器的系数向量
        cav = classifier.coef_.flatten()
        cav = cav / (np.linalg.norm(cav) + 1e-10)  # 归一化
        
        self.cavs[concept_name] = cav
        self.cav_cache[concept_name] = cav
        
        return cav
    
    def compute_concept_sensitivity(self,
                                    model: Callable,
                                    input_data: np.ndarray,
                                    concept_name: str,
                                    target_class: int = 0) -> float:
        """
        计算概念敏感度（方向导数）
        
        Args:
            model: 模型
            input_data: 输入样本
            concept_name: 概念名称
            target_class: 目标类别
            
        Returns:
            概念敏感度
        """
        if concept_name not in self.cavs:
            self.learn_cav(model, concept_name)
        
        cav = self.cavs[concept_name]
        
        # 计算梯度（简化版：使用数值梯度）
        epsilon = 1e-4
        
        def get_prediction(x):
            pred = model(x.reshape(1, -1))
            if isinstance(pred, np.ndarray) and pred.ndim > 1:
                return pred[0, target_class]
            return pred[0]
        
        # 在CAV方向上的梯度
        grad = np.zeros(len(input_data))
        for i in range(len(input_data)):
            x_plus = input_data.copy()
            x_minus = input_data.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (get_prediction(x_plus) - get_prediction(x_minus)) / (2 * epsilon)
        
        # 概念敏感度 = 梯度 · CAV
        sensitivity = np.dot(grad, cav[:len(grad)])
        
        return sensitivity
    
    def compute_tcav_score(self,
                          model: Callable,
                          test_data: np.ndarray,
                          concept_name: str,
                          target_class: int = 0) -> float:
        """
        计算TCAV分数
        
        TCAV = 有正面概念敏感度的样本比例
        
        Args:
            model: 模型
            test_data: 测试样本集
            concept_name: 概念名称
            target_class: 目标类别
            
        Returns:
            TCAV分数 [0, 1]
        """
        positive_count = 0
        
        for sample in test_data:
            sensitivity = self.compute_concept_sensitivity(
                model, sample, concept_name, target_class
            )
            if sensitivity > 0:
                positive_count += 1
        
        return positive_count / len(test_data)
    
    def explain(self,
                model: Callable,
                input_data: np.ndarray,
                concepts: Optional[List[str]] = None,
                **kwargs) -> ExplanationResult:
        """
        生成CAV解释
        """
        if concepts is None:
            concepts = list(self.concept_data.keys())
        
        concept_importance = {}
        
        for concept in concepts:
            sensitivity = self.compute_concept_sensitivity(
                model, input_data.flatten(), concept
            )
            concept_importance[concept] = sensitivity
        
        return ExplanationResult(
            method="CAV",
            concept_importance=concept_importance,
            metadata={
                "layer_name": self.layer_name,
                "concepts": concepts
            }
        )
    
    def visualize(self,
                  explanation: ExplanationResult,
                  figsize: Tuple[int, int] = (10, 6),
                  **kwargs) -> plt.Figure:
        """
        可视化CAV解释
        """
        concept_importance = explanation.concept_importance
        
        concepts = list(concept_importance.keys())
        values = list(concept_importance.values())
        
        fig, ax = plt.subplots(figsize=figsize)
        
        colors = ['green' if v > 0 else 'red' for v in values]
        y_pos = np.arange(len(concepts))
        
        ax.barh(y_pos, values, color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(concepts)
        ax.invert_yaxis()
        ax.set_xlabel('Concept Sensitivity')
        ax.set_title('CAV Concept Importance')
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig


# =============================================================================
# 5. 反事实解释
# =============================================================================

class CounterfactualExplainer(BaseExplainer):
    """
    反事实解释器
    
    生成使预测改变的最小输入变化
    """
    
    def __init__(self,
                 proximity_weight: float = 1.0,
                 diversity_weight: float = 0.5,
                 max_iterations: int = 1000,
                 learning_rate: float = 0.1):
        """
        初始化反事实解释器
        
        Args:
            proximity_weight: 邻近性权重
            diversity_weight: 多样性权重
            max_iterations: 最大迭代次数
            learning_rate: 学习率
        """
        self.proximity_weight = proximity_weight
        self.diversity_weight = diversity_weight
        self.max_iterations = max_iterations
        self.learning_rate = learning_rate
        
    def _compute_loss(self,
                      x: np.ndarray,
                      x_orig: np.ndarray,
                      target_pred: float,
                      current_pred: float) -> float:
        """
        计算反事实损失函数
        
        L = L_pred + λ·L_proximity
        
        Args:
            x: 当前反事实样本
            x_orig: 原始样本
            target_pred: 目标预测值
            current_pred: 当前预测值
            
        Returns:
            损失值
        """
        # 预测损失
        pred_loss = (current_pred - target_pred) ** 2
        
        # 邻近性损失（L1距离）
        prox_loss = np.sum(np.abs(x - x_orig))
        
        return pred_loss + self.proximity_weight * prox_loss
    
    def generate_counterfactual(self,
                               model: Callable,
                               input_data: np.ndarray,
                               target_class: int,
                               feature_ranges: Optional[Dict[int, Tuple]] = None,
                               immutable_features: Optional[List[int]] = None) -> np.ndarray:
        """
        生成反事实解释
        
        Args:
            model: 预测模型
            input_data: 原始输入
            target_class: 目标类别
            feature_ranges: 特征取值范围 {特征索引: (最小值, 最大值)}
            immutable_features: 不可变特征索引列表
            
        Returns:
            反事实样本
        """
        if input_data.ndim > 1:
            input_data = input_data.flatten()
        
        x_cf = input_data.copy().astype(float)
        immutable_features = immutable_features or []
        
        def get_prediction(x):
            pred = model(x.reshape(1, -1))
            if isinstance(pred, np.ndarray):
                return pred[0] if pred.ndim == 1 else pred[0, target_class]
            return pred
        
        # 目标预测值（假设为1.0表示目标类别）
        target_pred = 1.0
        
        for iteration in range(self.max_iterations):
            current_pred = get_prediction(x_cf)
            
            # 检查是否达到目标
            if current_pred >= 0.5:  # 假设二分类，0.5为阈值
                break
            
            # 数值梯度计算
            grad = np.zeros_like(x_cf)
            epsilon = 1e-4
            
            for i in range(len(x_cf)):
                if i in immutable_features:
                    continue
                
                x_plus = x_cf.copy()
                x_minus = x_cf.copy()
                x_plus[i] += epsilon
                x_minus[i] -= epsilon
                
                grad[i] = (get_prediction(x_plus) - get_prediction(x_minus)) / (2 * epsilon)
            
            # 梯度下降更新
            x_cf = x_cf + self.learning_rate * grad
            
            # 应用特征范围约束
            if feature_ranges:
                for idx, (min_val, max_val) in feature_ranges.items():
                    x_cf[idx] = np.clip(x_cf[idx], min_val, max_val)
            
            # 保持不可变特征
            for idx in immutable_features:
                x_cf[idx] = input_data[idx]
        
        return x_cf
    
    def explain(self,
                model: Callable,
                input_data: np.ndarray,
                target_class: int = 1,
                **kwargs) -> ExplanationResult:
        """
        生成反事实解释
        """
        counterfactual = self.generate_counterfactual(
            model, input_data, target_class, **kwargs
        )
        
        # 计算变化量
        delta = counterfactual - input_data.flatten()
        
        return ExplanationResult(
            method="Counterfactual",
            counterfactual=counterfactual,
            feature_importance=delta,  # 变化量作为特征重要性
            metadata={
                "original": input_data.flatten(),
                "delta": delta,
                "target_class": target_class,
                "changed_features": np.where(np.abs(delta) > 1e-6)[0].tolist()
            }
        )
    
    def visualize(self,
                  explanation: ExplanationResult,
                  feature_names: Optional[List[str]] = None,
                  figsize: Tuple[int, int] = (12, 6),
                  **kwargs) -> plt.Figure:
        """
        可视化反事实解释
        """
        original = explanation.metadata["original"]
        counterfactual = explanation.counterfactual
        delta = explanation.metadata["delta"]
        
        n_features = len(original)
        if feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(n_features)]
        
        fig, axes = plt.subplots(1, 2, figsize=figsize)
        
        # 左图：原始 vs 反事实对比
        x = np.arange(n_features)
        width = 0.35
        
        axes[0].bar(x - width/2, original, width, label='Original', alpha=0.7)
        axes[0].bar(x + width/2, counterfactual, width, label='Counterfactual', alpha=0.7)
        axes[0].set_xlabel('Features')
        axes[0].set_ylabel('Value')
        axes[0].set_title('Original vs Counterfactual')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[0].legend()
        
        # 右图：变化量
        colors = ['green' if d > 0 else 'red' for d in delta]
        axes[1].bar(x, delta, color=colors, alpha=0.7)
        axes[1].set_xlabel('Features')
        axes[1].set_ylabel('Change')
        axes[1].set_title('Required Changes (Counterfactual - Original)')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(feature_names, rotation=45, ha='right')
        axes[1].axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        
        plt.tight_layout()
        return fig


# =============================================================================
# 6. 统一接口和示例
# =============================================================================

class XAIExplainer:
    """
    XAI解释器统一接口
    
    整合所有解释方法，提供统一的API
    """
    
    METHODS = {
        'attention': AttentionVisualizer,
        'lime': LIMEExplainer,
        'shap': SHAPExplainer,
        'cav': CAVExplainer,
        'counterfactual': CounterfactualExplainer
    }
    
    def __init__(self, method: str, **kwargs):
        """
        初始化解释器
        
        Args:
            method: 解释方法名称
            **kwargs: 方法特定参数
        """
        if method not in self.METHODS:
            raise ValueError(f"Unknown method: {method}. Available: {list(self.METHODS.keys())}")
        
        self.method = method
        self.explainer = self.METHODS[method](**kwargs)
    
    def explain(self, model, input_data, **kwargs) -> ExplanationResult:
        """生成解释"""
        return self.explainer.explain(model, input_data, **kwargs)
    
    def visualize(self, explanation: ExplanationResult, **kwargs) -> plt.Figure:
        """可视化解释"""
        return self.explainer.visualize(explanation, **kwargs)


def demo():
    """
    演示所有XAI方法
    """
    print("=" * 60)
    print("XAI Explainer Demo")
    print("=" * 60)
    
    # 创建合成数据和模型
    np.random.seed(42)
    n_samples = 1000
    n_features = 10
    
    # 生成合成数据
    X = np.random.randn(n_samples, n_features)
    # 目标：前3个特征决定输出
    y = (X[:, 0] + X[:, 1] - X[:, 2] > 0).astype(int)
    
    print(f"\nDataset: {n_samples} samples, {n_features} features")
    print(f"Target: y = sign(x0 + x1 - x2)")
    
    # 简单的线性模型作为黑盒
    class SimpleModel:
        def __init__(self, weights):
            self.weights = weights
        
        def __call__(self, X):
            if X.ndim == 1:
                X = X.reshape(1, -1)
            logits = X @ self.weights
            # 返回概率
            return 1 / (1 + np.exp(-logits))
        
        def predict(self, X):
            return (self.__call__(X) > 0.5).astype(int)
    
    # 训练简单权重
    weights = np.array([1.0, 1.0, -1.0] + [0.1] * (n_features - 3))
    model = SimpleModel(weights)
    
    # 选择要解释的样本
    sample = X[0]
    print(f"\nSample to explain: {sample}")
    print(f"Prediction: {model.predict(sample)[0]}")
    
    # 1. LIME解释
    print("\n" + "-" * 40)
    print("1. LIME Explanation")
    print("-" * 40)
    
    lime_explainer = LIMEExplainer(n_samples=500)
    lime_result = lime_explainer.explain(
        model, sample, 
        feature_names=[f"Feature_{i}" for i in range(n_features)]
    )
    print(f"Top 3 important features (LIME):")
    top_indices = np.argsort(np.abs(lime_result.feature_importance))[-3:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {lime_result.feature_importance[idx]:.4f}")
    
    # 2. SHAP解释
    print("\n" + "-" * 40)
    print("2. SHAP Explanation")
    print("-" * 40)
    
    shap_explainer = SHAPExplainer(n_samples=50, background_data=X[:50])
    shap_result = shap_explainer.explain(model, sample)
    print(f"Top 3 important features (SHAP):")
    top_indices = np.argsort(np.abs(shap_result.feature_importance))[-3:][::-1]
    for idx in top_indices:
        print(f"  Feature {idx}: {shap_result.feature_importance[idx]:.4f}")
    
    # 3. 反事实解释
    print("\n" + "-" * 40)
    print("3. Counterfactual Explanation")
    print("-" * 40)
    
    # 找到一个负样本生成反事实
    neg_sample = None
    for x in X:
        if model.predict(x)[0] == 0:
            neg_sample = x
            break
    
    if neg_sample is not None:
        cf_explainer = CounterfactualExplainer(max_iterations=500)
        cf_result = cf_explainer.explain(model, neg_sample, target_class=1)
        print(f"Original prediction: 0")
        print(f"Counterfactual prediction: 1")
        print(f"Changed features: {cf_result.metadata['changed_features']}")
        print(f"Changes: {cf_result.metadata['delta']}")
    
    # 4. CAV解释（简化版）
    print("\n" + "-" * 40)
    print("4. CAV Explanation (Concept-based)")
    print("-" * 40)
    
    # 定义概念数据集
    concept_data = {
        "Positive_Feature_0": X[X[:, 0] > 1.0][:50],
        "Negative_Feature_2": X[X[:, 2] < -1.0][:50]
    }
    random_data = X[np.random.choice(len(X), 100, replace=False)]
    
    # 使用简单激活函数模拟
    def mock_activation_extractor(data, layer=None):
        return data  # 直接使用输入作为"激活"
    
    cav_explainer = CAVExplainer(
        layer_name="input",
        concept_data=concept_data,
        random_data=random_data
    )
    
    cav_result = cav_explainer.explain(
        mock_activation_extractor, 
        sample,
        concepts=["Positive_Feature_0", "Negative_Feature_2"]
    )
    
    print("Concept sensitivities:")
    for concept, sensitivity in cav_result.concept_importance.items():
        print(f"  {concept}: {sensitivity:.4f}")
    
    print("\n" + "=" * 60)
    print("Demo completed!")
    print("=" * 60)
    
    return {
        'lime': lime_result,
        'shap': shap_result,
        'cav': cav_result,
        'counterfactual': cf_result if neg_sample is not None else None
    }


if __name__ == "__main__":
    # 运行演示
    results = demo()
    
    # 保存可视化（可选）
    # for method, result in results.items():
    #     if result is not None:
    #         fig = result.visualization
    #         if fig is not None:
    #             fig.savefig(f"xai_{method}_explanation.png")
    #             plt.close(fig)
