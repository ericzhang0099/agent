"""
Continual Learning Framework - 持续学习框架
==============================================

核心功能：
1. 弹性权重整合 (EWC) - Elastic Weight Consolidation
2. 渐进式神经网络 (PNN) - Progressive Neural Networks
3. 记忆回放机制 - Experience Replay
4. 元学习集成 - Meta-Learning Integration
5. 灾难性遗忘防护 - Catastrophic Forgetting Prevention

Author: AI Research Team
Version: 1.0.0
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable, Any
from collections import deque
import copy
from dataclasses import dataclass, field
from abc import ABC, abstractmethod
import warnings


# =============================================================================
# 配置类
# =============================================================================

@dataclass
class ContinualLearningConfig:
    """持续学习配置类"""
    # EWC 相关参数
    ewc_lambda: float = 100.0  # EWC正则化强度
    ewc_estimate_type: str = 'true'  # 'true' 或 'empirical'
    
    # 记忆缓冲区参数
    memory_buffer_size: int = 1000
    memory_sampling_strategy: str = 'reservoir'  # 'reservoir', 'random', 'herding'
    
    # PNN 参数
    pnn_adapter_hidden_dim: int = 256
    pnn_use_adapters: bool = True
    
    # 元学习参数
    meta_learning_rate: float = 0.001
    meta_inner_steps: int = 5
    
    # 训练参数
    learning_rate: float = 0.001
    batch_size: int = 64
    epochs_per_task: int = 10
    device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # 损失权重
    replay_loss_weight: float = 1.0
    consistency_loss_weight: float = 0.5


# =============================================================================
# 基础神经网络组件
# =============================================================================

class Flatten(nn.Module):
    """展平层"""
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x.view(x.shape[0], -1)


class LinearLayer(nn.Module):
    """带可选BatchNorm的线性层"""
    def __init__(self, input_dim: int, output_dim: int, 
                 act: str = 'relu', use_bn: bool = False):
        super().__init__()
        self.use_bn = use_bn
        self.lin = nn.Linear(input_dim, output_dim)
        
        if act == 'relu':
            self.act = nn.ReLU()
        elif act == 'tanh':
            self.act = nn.Tanh()
        elif act == 'sigmoid':
            self.act = nn.Sigmoid()
        else:
            self.act = nn.Identity()
            
        if use_bn:
            self.bn = nn.BatchNorm1d(output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lin(x)
        x = self.act(x)
        if self.use_bn:
            x = self.bn(x)
        return x


class SimpleMLP(nn.Module):
    """简单的多层感知机"""
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int,
                 num_hidden_layers: int = 2, use_bn: bool = True):
        super().__init__()
        self.flatten = Flatten()
        
        layers = []
        layers.append(LinearLayer(input_dim, hidden_dim, use_bn=use_bn))
        
        for _ in range(num_hidden_layers - 1):
            layers.append(LinearLayer(hidden_dim, hidden_dim, use_bn=use_bn))
        
        self.hidden_layers = nn.Sequential(*layers)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.flatten(x)
        x = self.hidden_layers(x)
        return self.output_layer(x)


class SimpleConvNet(nn.Module):
    """简单的卷积神经网络（用于图像任务）"""
    def __init__(self, input_channels: int = 1, num_classes: int = 10,
                 hidden_dim: int = 128):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Flatten()
        )
        
        # 计算展平后的维度（假设输入为28x28）
        self.feature_dim = 64 * 7 * 7
        
        self.classifier = nn.Sequential(
            nn.Linear(self.feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
    
    def get_features(self, x: torch.Tensor) -> torch.Tensor:
        """获取特征表示"""
        return self.features(x)


# =============================================================================
# EWC (Elastic Weight Consolidation) 实现
# =============================================================================

class EWC:
    """
    弹性权重整合 (Elastic Weight Consolidation)
    
    基于Kirkpatrick et al. 2017的论文实现。
    通过Fisher信息矩阵来识别对先前任务重要的参数，
    并在学习新任务时对这些参数施加正则化约束。
    """
    
    def __init__(self, model: nn.Module, config: ContinualLearningConfig):
        self.model = model
        self.config = config
        self.device = config.device
        
        # 存储每个任务的Fisher信息矩阵和最优参数
        self.fisher_matrices: List[Dict[str, torch.Tensor]] = []
        self.optimal_params: List[Dict[str, torch.Tensor]] = []
        
    def compute_fisher_matrix(self, dataset: Dataset, 
                              num_samples: Optional[int] = None) -> Dict[str, torch.Tensor]:
        """
        计算Fisher信息矩阵的对角近似
        
        Args:
            dataset: 用于计算Fisher矩阵的数据集
            num_samples: 采样的样本数量（None表示使用全部）
            
        Returns:
            Fisher信息矩阵的对角近似
        """
        self.model.eval()
        fisher_matrix = {}
        
        # 初始化Fisher矩阵
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                fisher_matrix[name] = torch.zeros_like(param)
        
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size, 
                               shuffle=True)
        
        num_batches = 0
        max_batches = num_samples // self.config.batch_size if num_samples else float('inf')
        
        for batch_idx, (inputs, targets) in enumerate(dataloader):
            if batch_idx >= max_batches:
                break
                
            inputs = inputs.to(self.device)
            targets = targets.to(self.device)
            
            self.model.zero_grad()
            outputs = self.model(inputs)
            
            # 计算损失
            if self.config.ewc_estimate_type == 'empirical':
                # 使用真实标签
                loss = F.cross_entropy(outputs, targets)
            else:
                # 使用模型预测（真实Fisher）
                predicted = outputs.max(1)[1]
                loss = F.nll_loss(F.log_softmax(outputs, dim=1), predicted)
            
            loss.backward()
            
            # 累积梯度平方
            for name, param in self.model.named_parameters():
                if param.requires_grad and param.grad is not None:
                    fisher_matrix[name] += param.grad.data ** 2
            
            num_batches += 1
        
        # 平均
        for name in fisher_matrix:
            fisher_matrix[name] /= num_batches
            
        return fisher_matrix
    
    def save_task_params(self, fisher_matrix: Dict[str, torch.Tensor]):
        """保存任务的Fisher矩阵和最优参数"""
        optimal_params = {}
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                optimal_params[name] = param.data.clone()
        
        self.fisher_matrices.append(fisher_matrix)
        self.optimal_params.append(optimal_params)
    
    def compute_ewc_loss(self) -> torch.Tensor:
        """
        计算EWC正则化损失
        
        Returns:
            EWC损失值
        """
        if len(self.fisher_matrices) == 0:
            return torch.tensor(0.0, device=self.device)
        
        ewc_loss = 0.0
        
        for fisher_matrix, optimal_params in zip(self.fisher_matrices, 
                                                  self.optimal_params):
            for name, param in self.model.named_parameters():
                if param.requires_grad and name in fisher_matrix:
                    fisher = fisher_matrix[name]
                    optimal = optimal_params[name]
                    ewc_loss += (fisher * (param - optimal) ** 2).sum()
        
        return (self.config.ewc_lambda / 2.0) * ewc_loss
    
    def after_task(self, dataset: Dataset):
        """任务结束后调用，保存当前任务的参数"""
        fisher_matrix = self.compute_fisher_matrix(dataset)
        self.save_task_params(fisher_matrix)


# =============================================================================
# 记忆缓冲区实现
# =============================================================================

class MemoryBuffer:
    """
    经验回放记忆缓冲区
    
    支持多种采样策略：
    - reservoir: 水库采样，每个样本有相同概率被保留
    - random: 随机采样
    - herding: 基于特征中心的选择
    """
    
    def __init__(self, buffer_size: int, sampling_strategy: str = 'reservoir'):
        self.buffer_size = buffer_size
        self.sampling_strategy = sampling_strategy
        self.buffer: List[Tuple[torch.Tensor, int, Optional[torch.Tensor]]] = []
        self.counter = 0  # 用于水库采样
        
    def add(self, sample: torch.Tensor, label: int, 
            logits: Optional[torch.Tensor] = None):
        """
        添加样本到缓冲区
        
        Args:
            sample: 输入样本
            label: 样本标签
            logits: 模型输出的logits（用于知识蒸馏）
        """
        if self.sampling_strategy == 'reservoir':
            self._reservoir_sampling(sample, label, logits)
        elif self.sampling_strategy == 'random':
            self._random_sampling(sample, label, logits)
        else:
            self.buffer.append((sample, label, logits))
            if len(self.buffer) > self.buffer_size:
                self.buffer.pop(0)
    
    def _reservoir_sampling(self, sample: torch.Tensor, label: int,
                           logits: Optional[torch.Tensor]):
        """水库采样算法"""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((sample, label, logits))
        else:
            # 以 buffer_size / counter 的概率替换
            idx = np.random.randint(0, self.counter + 1)
            if idx < self.buffer_size:
                self.buffer[idx] = (sample, label, logits)
        self.counter += 1
    
    def _random_sampling(self, sample: torch.Tensor, label: int,
                        logits: Optional[torch.Tensor]):
        """随机采样"""
        if len(self.buffer) < self.buffer_size:
            self.buffer.append((sample, label, logits))
        else:
            # 随机替换
            idx = np.random.randint(0, self.buffer_size)
            self.buffer[idx] = (sample, label, logits)
    
    def sample(self, batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, 
                                               Optional[torch.Tensor]]:
        """
        从缓冲区采样
        
        Returns:
            samples: 样本张量
            labels: 标签张量
            logits: logits张量（可能为None）
        """
        if len(self.buffer) == 0:
            return None, None, None
        
        batch_size = min(batch_size, len(self.buffer))
        indices = np.random.choice(len(self.buffer), batch_size, replace=False)
        
        samples = []
        labels = []
        logits_list = []
        
        for idx in indices:
            s, l, log = self.buffer[idx]
            samples.append(s)
            labels.append(l)
            logits_list.append(log)
        
        samples = torch.stack(samples)
        labels = torch.tensor(labels)
        
        if logits_list[0] is not None:
            logits = torch.stack(logits_list)
        else:
            logits = None
            
        return samples, labels, logits
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.counter = 0


# =============================================================================
# 渐进式神经网络 (Progressive Neural Networks)
# =============================================================================

class PNNColumn(nn.Module):
    """
    PNN中的单列网络
    
    每列代表一个任务，包含侧向连接到之前所有列
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], output_dim: int,
                 previous_columns: List['PNNColumn'], use_adapters: bool = True,
                 adapter_hidden_dim: int = 256):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.previous_columns = previous_columns
        self.use_adapters = use_adapters
        
        # 构建当前列的层
        self.layers = nn.ModuleList()
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            prev_dim = hidden_dim
        
        self.output_layer = nn.Linear(prev_dim, output_dim)
        
        # 侧向连接（从之前的列到当前列）
        self.lateral_connections = nn.ModuleList()
        for prev_col in previous_columns:
            lateral_layer = nn.ModuleList()
            for i, hidden_dim in enumerate(hidden_dims):
                if use_adapters:
                    # 使用适配器进行降维
                    adapter = nn.Sequential(
                        nn.Linear(prev_col.hidden_dims[i], adapter_hidden_dim),
                        nn.ReLU(),
                        nn.Linear(adapter_hidden_dim, hidden_dim)
                    )
                    lateral_layer.append(adapter)
                else:
                    lateral_layer.append(nn.Linear(prev_col.hidden_dims[i], hidden_dim))
            self.lateral_connections.append(lateral_layer)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """前向传播，包含侧向连接"""
        current = x
        
        for i, layer in enumerate(self.layers):
            # 主连接
            h = layer(current)
            
            # 侧向连接
            for j, prev_col in enumerate(self.previous_columns):
                if i < len(prev_col.layers):
                    with torch.no_grad():
                        prev_h = prev_col.layers[i](current if i == 0 else 
                                                    self._get_prev_activation(prev_col, i-1))
                    lateral_h = self.lateral_connections[j][i](prev_h)
                    h = h + lateral_h
            
            h = F.relu(h)
            current = h
        
        return self.output_layer(current)
    
    def _get_prev_activation(self, prev_col: 'PNNColumn', layer_idx: int) -> torch.Tensor:
        """获取之前列的激活值（用于缓存）"""
        # 简化的实现，实际应用中可能需要缓存机制
        return torch.zeros(1)  # 占位符


class ProgressiveNeuralNetwork(nn.Module):
    """
    渐进式神经网络 (Progressive Neural Networks)
    
    每个新任务添加一个新列，通过侧向连接利用之前任务的知识
    """
    
    def __init__(self, input_dim: int, hidden_dims: List[int], 
                 output_dim: int, config: ContinualLearningConfig):
        super().__init__()
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims
        self.output_dim = output_dim
        self.config = config
        
        self.columns: List[PNNColumn] = nn.ModuleList()
        self.current_task_id = -1
    
    def add_column(self) -> int:
        """
        为新任务添加一列
        
        Returns:
            新列的索引
        """
        self.current_task_id += 1
        
        # 冻结之前所有列的参数
        for col in self.columns:
            for param in col.parameters():
                param.requires_grad = False
        
        # 创建新列
        new_column = PNNColumn(
            self.input_dim,
            self.hidden_dims,
            self.output_dim,
            list(self.columns),  # 传递之前列的副本
            use_adapters=self.config.pnn_use_adapters,
            adapter_hidden_dim=self.config.pnn_adapter_hidden_dim
        )
        
        self.columns.append(new_column)
        return self.current_task_id
    
    def forward(self, x: torch.Tensor, task_id: Optional[int] = None) -> torch.Tensor:
        """
        前向传播
        
        Args:
            x: 输入张量
            task_id: 任务ID（None表示使用当前任务）
        """
        if task_id is None:
            task_id = self.current_task_id
        
        if task_id < 0 or task_id >= len(self.columns):
            raise ValueError(f"Invalid task_id: {task_id}")
        
        return self.columns[task_id](x)
    
    def get_parameters(self, task_id: Optional[int] = None) -> List[nn.Parameter]:
        """获取指定任务列的参数"""
        if task_id is None:
            task_id = self.current_task_id
        
        if task_id < 0 or task_id >= len(self.columns):
            return []
        
        return list(self.columns[task_id].parameters())


# =============================================================================
# 持续学习主类
# =============================================================================

class ContinualLearner:
    """
    持续学习主类
    
    集成多种持续学习策略：
    - EWC (弹性权重整合)
    - Experience Replay (经验回放)
    - Progressive Neural Networks (渐进式神经网络)
    - Meta-Learning (元学习)
    """
    
    def __init__(self, model: nn.Module, config: ContinualLearningConfig,
                 strategy: str = 'ewc'):
        """
        初始化持续学习器
        
        Args:
            model: 基础神经网络模型
            config: 配置对象
            strategy: 学习策略 ('ewc', 'replay', 'pnn', 'meta', 'combined')
        """
        self.model = model.to(config.device)
        self.config = config
        self.strategy = strategy
        self.device = config.device
        
        # 初始化各个组件
        self.ewc = EWC(model, config) if strategy in ['ewc', 'combined'] else None
        self.memory_buffer = MemoryBuffer(
            config.memory_buffer_size, 
            config.memory_sampling_strategy
        ) if strategy in ['replay', 'combined'] else None
        
        self.pnn = None
        if strategy == 'pnn':
            # PNN模式下，model参数会被忽略，使用PNN内部结构
            warnings.warn("PNN strategy uses its own architecture, input model will be ignored")
        
        # 优化器
        self.optimizer = optim.Adam(
            self.model.parameters(), 
            lr=config.learning_rate
        )
        
        # 任务历史
        self.task_history: List[Dict[str, Any]] = []
        self.current_task_id = 0
        
        # 旧模型（用于知识蒸馏）
        self.old_model: Optional[nn.Module] = None
    
    def train_task(self, train_dataset: Dataset, val_dataset: Optional[Dataset] = None,
                   task_name: Optional[str] = None) -> Dict[str, float]:
        """
        训练一个新任务
        
        Args:
            train_dataset: 训练数据集
            val_dataset: 验证数据集
            task_name: 任务名称
            
        Returns:
            训练历史指标
        """
        print(f"\n{'='*60}")
        print(f"Training Task {self.current_task_id}: {task_name or 'Unnamed'}")
        print(f"Strategy: {self.strategy}")
        print(f"{'='*60}")
        
        # 保存旧模型用于知识蒸馏
        if self.current_task_id > 0:
            self.old_model = copy.deepcopy(self.model)
            self.old_model.eval()
            for param in self.old_model.parameters():
                param.requires_grad = False
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=self.config.batch_size,
            shuffle=True
        )
        
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(self.config.epochs_per_task):
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (inputs, targets) in enumerate(train_loader):
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                # 计算损失
                loss, outputs = self._compute_loss(inputs, targets)
                
                # 反向传播
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                
                # 更新记忆缓冲区
                if self.memory_buffer is not None:
                    self._update_memory_buffer(inputs, targets)
                
                # 统计
                epoch_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            if (epoch + 1) % 2 == 0:
                print(f"Epoch {epoch+1}/{self.config.epochs_per_task}: "
                      f"Loss={avg_loss:.4f}, Acc={accuracy:.2f}%")
        
        # 任务结束后处理
        self._after_task(train_dataset)
        
        # 记录任务历史
        self.task_history.append({
            'task_id': self.current_task_id,
            'task_name': task_name,
            'history': history,
            'num_samples': len(train_dataset)
        })
        
        self.current_task_id += 1
        
        return history
    
    def _compute_loss(self, inputs: torch.Tensor, 
                     targets: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        计算损失（根据策略组合不同损失）
        """
        outputs = self.model(inputs)
        
        # 基础分类损失
        loss = F.cross_entropy(outputs, targets)
        
        # EWC损失
        if self.ewc is not None and self.current_task_id > 0:
            ewc_loss = self.ewc.compute_ewc_loss()
            loss = loss + ewc_loss
        
        # 回放损失
        if self.memory_buffer is not None and len(self.memory_buffer) > 0:
            replay_loss = self._compute_replay_loss()
            loss = loss + self.config.replay_loss_weight * replay_loss
        
        # 一致性损失（知识蒸馏）
        if self.old_model is not None and self.current_task_id > 0:
            consistency_loss = self._compute_consistency_loss(inputs)
            loss = loss + self.config.consistency_loss_weight * consistency_loss
        
        return loss, outputs
    
    def _compute_replay_loss(self) -> torch.Tensor:
        """计算回放损失"""
        mem_samples, mem_labels, mem_logits = self.memory_buffer.sample(
            self.config.batch_size
        )
        
        if mem_samples is None:
            return torch.tensor(0.0, device=self.device)
        
        mem_samples = mem_samples.to(self.device)
        mem_labels = mem_labels.to(self.device)
        
        mem_outputs = self.model(mem_samples)
        replay_loss = F.cross_entropy(mem_outputs, mem_labels)
        
        # 如果使用logits进行蒸馏
        if mem_logits is not None:
            mem_logits = mem_logits.to(self.device)
            distill_loss = F.mse_loss(F.softmax(mem_outputs, dim=1),
                                     F.softmax(mem_logits, dim=1))
            replay_loss = replay_loss + distill_loss
        
        return replay_loss
    
    def _compute_consistency_loss(self, inputs: torch.Tensor) -> torch.Tensor:
        """计算一致性损失（前向一致性）"""
        with torch.no_grad():
            old_outputs = self.old_model(inputs)
        
        new_outputs = self.model(inputs)
        
        # MSE损失在logits上
        consistency_loss = F.mse_loss(new_outputs, old_outputs)
        
        return consistency_loss
    
    def _update_memory_buffer(self, inputs: torch.Tensor, targets: torch.Tensor):
        """更新记忆缓冲区"""
        if self.memory_buffer is None:
            return
        
        # 随机选择一些样本加入缓冲区
        batch_size = inputs.size(0)
        num_to_add = max(1, batch_size // 10)  # 添加10%的样本
        
        indices = np.random.choice(batch_size, num_to_add, replace=False)
        
        with torch.no_grad():
            for idx in indices:
                sample = inputs[idx].cpu()
                label = targets[idx].item()
                
                # 获取logits用于蒸馏
                logit = self.model(inputs[idx:idx+1]).squeeze(0).cpu()
                
                self.memory_buffer.add(sample, label, logit)
    
    def _after_task(self, dataset: Dataset):
        """任务结束后的处理"""
        # EWC: 计算Fisher矩阵
        if self.ewc is not None:
            print("Computing Fisher Information Matrix for EWC...")
            self.ewc.after_task(dataset)
        
        # 更新优化器（如果需要）
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.config.learning_rate
        )
    
    def evaluate(self, dataset: Dataset, task_id: Optional[int] = None) -> Dict[str, float]:
        """
        评估模型在数据集上的性能
        
        Args:
            dataset: 评估数据集
            task_id: 任务ID（用于PNN）
            
        Returns:
            评估指标
        """
        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=self.config.batch_size)
        
        correct = 0
        total = 0
        total_loss = 0.0
        
        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(self.device)
                targets = targets.to(self.device)
                
                if self.strategy == 'pnn' and self.pnn is not None:
                    outputs = self.pnn(inputs, task_id)
                else:
                    outputs = self.model(inputs)
                
                loss = F.cross_entropy(outputs, targets)
                total_loss += loss.item()
                
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = total_loss / len(dataloader)
        
        self.model.train()
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def evaluate_all_tasks(self, task_datasets: List[Dataset]) -> Dict[int, Dict[str, float]]:
        """
        评估所有历史任务
        
        Args:
            task_datasets: 每个任务的数据集列表
            
        Returns:
            每个任务的评估结果
        """
        results = {}
        
        print(f"\n{'='*60}")
        print("Evaluating All Tasks")
        print(f"{'='*60}")
        
        for task_id, dataset in enumerate(task_datasets):
            result = self.evaluate(dataset, task_id)
            results[task_id] = result
            
            task_name = (self.task_history[task_id]['task_name'] 
                        if task_id < len(self.task_history) else f"Task {task_id}")
            print(f"{task_name}: Accuracy={result['accuracy']:.2f}%, Loss={result['loss']:.4f}")
        
        # 计算平均准确率
        avg_accuracy = np.mean([r['accuracy'] for r in results.values()])
        print(f"\nAverage Accuracy: {avg_accuracy:.2f}%")
        
        return results
    
    def get_forgetting_rate(self, task_accuracies: List[List[float]]) -> float:
        """
        计算遗忘率
        
        Args:
            task_accuracies: 每个任务在每个训练阶段后的准确率列表
            
        Returns:
            平均遗忘率
        """
        if len(task_accuracies) <= 1:
            return 0.0
        
        forgetting_rates = []
        
        for task_id in range(len(task_accuracies)):
            # 找到训练该任务时的最高准确率
            max_acc = task_accuracies[task_id][task_id]
            
            # 找到训练最后一个任务后的准确率
            final_acc = task_accuracies[task_id][-1]
            
            forgetting = max_acc - final_acc
            forgetting_rates.append(forgetting)
        
        return np.mean(forgetting_rates)


# =============================================================================
# 元学习集成 (Meta-Learning Integration)
# =============================================================================

class MetaLearner:
    """
    元学习器 - 实现MAML风格的元学习
    
    使模型能够快速适应新任务，同时保持对旧任务的记忆
    """
    
    def __init__(self, model: nn.Module, config: ContinualLearningConfig):
        self.model = model
        self.config = config
        self.device = config.device
        
        # 元参数
        self.meta_lr = config.meta_learning_rate
        self.inner_steps = config.meta_inner_steps
        
        # 元优化器
        self.meta_optimizer = optim.Adam(
            self.model.parameters(),
            lr=self.meta_lr
        )
    
    def meta_update(self, support_set: Tuple[torch.Tensor, torch.Tensor],
                   query_set: Tuple[torch.Tensor, torch.Tensor]) -> float:
        """
        执行一次元更新
        
        Args:
            support_set: 支持集 (inputs, targets)
            query_set: 查询集 (inputs, targets)
            
        Returns:
            损失值
        """
        support_inputs, support_targets = support_set
        query_inputs, query_targets = query_set
        
        support_inputs = support_inputs.to(self.device)
        support_targets = support_targets.to(self.device)
        query_inputs = query_inputs.to(self.device)
        query_targets = query_targets.to(self.device)
        
        # 创建内部优化的模型副本
        inner_model = copy.deepcopy(self.model)
        inner_optimizer = optim.SGD(
            inner_model.parameters(),
            lr=self.config.learning_rate
        )
        
        # 内部优化（在支持集上）
        inner_model.train()
        for _ in range(self.inner_steps):
            inner_optimizer.zero_grad()
            outputs = inner_model(support_inputs)
            loss = F.cross_entropy(outputs, support_targets)
            loss.backward()
            inner_optimizer.step()
        
        # 外部优化（在查询集上）
        self.meta_optimizer.zero_grad()
        query_outputs = inner_model(query_inputs)
        query_loss = F.cross_entropy(query_outputs, query_targets)
        
        # 将梯度传播回原始模型
        query_loss.backward()
        self.meta_optimizer.step()
        
        return query_loss.item()


# =============================================================================
# 工具函数
# =============================================================================

def create_permuted_mnist_task(mnist_dataset: Dataset, 
                               permutation: Optional[np.ndarray] = None,
                               seed: Optional[int] = None) -> Dataset:
    """
    创建Permuted MNIST任务
    
    Args:
        mnist_dataset: 原始MNIST数据集
        permutation: 像素排列（None则随机生成）
        seed: 随机种子
        
    Returns:
        变换后的数据集
    """
    if permutation is None:
        if seed is not None:
            np.random.seed(seed)
        permutation = np.random.permutation(784)  # 28*28
    
    class PermutedMNIST(Dataset):
        def __init__(self, base_dataset, perm):
            self.base_dataset = base_dataset
            self.permutation = perm
        
        def __len__(self):
            return len(self.base_dataset)
        
        def __getitem__(self, idx):
            img, label = self.base_dataset[idx]
            # 应用排列
            img_flat = img.view(-1)
            img_permuted = img_flat[self.permutation]
            img_permuted = img_permuted.view(1, 28, 28)
            return img_permuted, label
    
    return PermutedMNIST(mnist_dataset, permutation)


def create_split_mnist_tasks(mnist_dataset: Dataset, 
                            num_tasks: int = 5) -> List[Dataset]:
    """
    创建Split MNIST任务（每个任务包含2个类别）
    
    Args:
        mnist_dataset: 原始MNIST数据集
        num_tasks: 任务数量
        
    Returns:
        任务数据集列表
    """
    classes_per_task = 10 // num_tasks
    tasks = []
    
    # 获取所有数据和标签
    all_data = []
    all_labels = []
    for img, label in mnist_dataset:
        all_data.append(img)
        all_labels.append(label)
    
    all_data = torch.stack(all_data)
    all_labels = torch.tensor(all_labels)
    
    for task_id in range(num_tasks):
        start_class = task_id * classes_per_task
        end_class = start_class + classes_per_task
        
        # 选择属于当前任务的样本
        mask = (all_labels >= start_class) & (all_labels < end_class)
        task_data = all_data[mask]
        task_labels = all_labels[mask]
        
        # 重新映射标签到0-classes_per_task
        task_labels = task_labels - start_class
        
        class TaskDataset(Dataset):
            def __init__(self, data, labels):
                self.data = data
                self.labels = labels
            
            def __len__(self):
                return len(self.data)
            
            def __getitem__(self, idx):
                return self.data[idx], self.labels[idx].item()
        
        tasks.append(TaskDataset(task_data, task_labels))
    
    return tasks


def compute_average_accuracy(results: Dict[int, Dict[str, float]]) -> float:
    """计算平均准确率"""
    accuracies = [r['accuracy'] for r in results.values()]
    return np.mean(accuracies)


def compute_backward_transfer(results: Dict[int, Dict[str, float]],
                              initial_results: Dict[int, Dict[str, float]]) -> float:
    """
    计算后向迁移 (Backward Transfer)
    
    正值表示学习新任务对旧任务有帮助
    """
    bwt = 0.0
    num_tasks = len(results)
    
    for task_id in range(num_tasks - 1):
        final_acc = results[task_id]['accuracy']
        initial_acc = initial_results[task_id]['accuracy']
        bwt += final_acc - initial_acc
    
    return bwt / (num_tasks - 1) if num_tasks > 1 else 0.0


# =============================================================================
# 示例用法和测试
# =============================================================================

def demo_continual_learning():
    """
    演示持续学习的完整流程
    """
    print("="*80)
    print("Continual Learning Framework Demo")
    print("="*80)
    
    # 配置
    config = ContinualLearningConfig(
        ewc_lambda=100.0,
        memory_buffer_size=500,
        learning_rate=0.001,
        epochs_per_task=3,
        batch_size=128
    )
    
    print(f"\nConfiguration:")
    print(f"  Device: {config.device}")
    print(f"  Strategy: EWC + Experience Replay")
    print(f"  EWC Lambda: {config.ewc_lambda}")
    print(f"  Memory Buffer Size: {config.memory_buffer_size}")
    
    # 创建简单模型
    model = SimpleMLP(input_dim=784, hidden_dim=256, output_dim=10, 
                     num_hidden_layers=2)
    
    print(f"\nModel Architecture:")
    print(model)
    
    # 创建持续学习器
    learner = ContinualLearner(model, config, strategy='combined')
    
    print("\n" + "="*80)
    print("Note: This is a demo without actual data.")
    print("To run full experiments, use the example scripts.")
    print("="*80)


if __name__ == "__main__":
    demo_continual_learning()
