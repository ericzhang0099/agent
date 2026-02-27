"""
Unit Tests for Continual Learning Framework
持续学习框架单元测试
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import unittest
import sys
import os

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continual_learner import (
    ContinualLearner, ContinualLearningConfig,
    SimpleMLP, SimpleConvNet, EWC, MemoryBuffer,
    ProgressiveNeuralNetwork, MetaLearner,
    Flatten, LinearLayer
)


class TestBasicComponents(unittest.TestCase):
    """测试基础组件"""
    
    def test_flatten(self):
        """测试展平层"""
        flatten = Flatten()
        x = torch.randn(10, 3, 4, 4)
        out = flatten(x)
        self.assertEqual(out.shape, (10, 48))
    
    def test_linear_layer(self):
        """测试线性层"""
        layer = LinearLayer(10, 20, act='relu', use_bn=True)
        x = torch.randn(5, 10)
        out = layer(x)
        self.assertEqual(out.shape, (5, 20))
    
    def test_simple_mlp(self):
        """测试简单MLP"""
        model = SimpleMLP(input_dim=784, hidden_dim=128, output_dim=10)
        x = torch.randn(16, 784)
        out = model(x)
        self.assertEqual(out.shape, (16, 10))
    
    def test_simple_convnet(self):
        """测试简单卷积网络"""
        model = SimpleConvNet(input_channels=1, num_classes=10)
        x = torch.randn(16, 1, 28, 28)
        out = model(x)
        self.assertEqual(out.shape, (16, 10))


class TestEWC(unittest.TestCase):
    """测试EWC组件"""
    
    def setUp(self):
        self.config = ContinualLearningConfig(device='cpu')
        self.model = SimpleMLP(784, 128, 10)
        self.ewc = EWC(self.model, self.config)
        
        # 创建简单数据集
        X = torch.randn(100, 784)
        y = torch.randint(0, 10, (100,))
        self.dataset = TensorDataset(X, y)
    
    def test_fisher_computation(self):
        """测试Fisher矩阵计算"""
        fisher = self.ewc.compute_fisher_matrix(self.dataset, num_samples=50)
        
        # 检查Fisher矩阵不为空
        self.assertGreater(len(fisher), 0)
        
        # 检查所有参数都有对应的Fisher值
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                self.assertIn(name, fisher)
                self.assertEqual(fisher[name].shape, param.shape)
    
    def test_ewc_loss(self):
        """测试EWC损失计算"""
        # 初始时应该没有EWC损失
        loss = self.ewc.compute_ewc_loss()
        self.assertEqual(loss.item(), 0.0)
        
        # 保存任务参数后
        fisher = self.ewc.compute_fisher_matrix(self.dataset, num_samples=50)
        self.ewc.save_task_params(fisher)
        
        # 修改模型参数
        for param in self.model.parameters():
            param.data += 0.1
        
        # 现在应该有EWC损失
        loss = self.ewc.compute_ewc_loss()
        self.assertGreater(loss.item(), 0.0)
    
    def test_after_task(self):
        """测试任务结束后处理"""
        self.ewc.after_task(self.dataset)
        
        # 检查是否保存了Fisher矩阵和最优参数
        self.assertEqual(len(self.ewc.fisher_matrices), 1)
        self.assertEqual(len(self.ewc.optimal_params), 1)


class TestMemoryBuffer(unittest.TestCase):
    """测试记忆缓冲区"""
    
    def setUp(self):
        self.buffer = MemoryBuffer(buffer_size=100, sampling_strategy='reservoir')
    
    def test_add_and_sample(self):
        """测试添加和采样"""
        # 添加样本
        for i in range(50):
            sample = torch.randn(3, 32, 32)
            label = i % 10
            logits = torch.randn(10)
            self.buffer.add(sample, label, logits)
        
        # 检查缓冲区大小
        self.assertEqual(len(self.buffer), 50)
        
        # 采样
        samples, labels, logits = self.buffer.sample(10)
        self.assertEqual(samples.shape, (10, 3, 32, 32))
        self.assertEqual(labels.shape, (10,))
        self.assertEqual(logits.shape, (10, 10))
    
    def test_reservoir_sampling(self):
        """测试水库采样"""
        buffer_size = 50
        buffer = MemoryBuffer(buffer_size=buffer_size, sampling_strategy='reservoir')
        
        # 添加超过缓冲区大小的样本
        for i in range(200):
            sample = torch.randn(3, 32, 32)
            buffer.add(sample, i, None)
        
        # 检查缓冲区大小不超过限制
        self.assertEqual(len(buffer), buffer_size)
    
    def test_clear(self):
        """测试清空缓冲区"""
        for i in range(10):
            sample = torch.randn(3, 32, 32)
            self.buffer.add(sample, i, None)
        
        self.assertGreater(len(self.buffer), 0)
        self.buffer.clear()
        self.assertEqual(len(self.buffer), 0)


class TestProgressiveNeuralNetwork(unittest.TestCase):
    """测试渐进式神经网络"""
    
    def setUp(self):
        self.config = ContinualLearningConfig(device='cpu')
        self.pnn = ProgressiveNeuralNetwork(
            input_dim=784,
            hidden_dims=[128, 128],
            output_dim=10,
            config=self.config
        )
    
    def test_add_column(self):
        """测试添加列"""
        task_id = self.pnn.add_column()
        self.assertEqual(task_id, 0)
        self.assertEqual(len(self.pnn.columns), 1)
        
        task_id = self.pnn.add_column()
        self.assertEqual(task_id, 1)
        self.assertEqual(len(self.pnn.columns), 2)
    
    def test_forward(self):
        """测试前向传播"""
        # 添加两列
        self.pnn.add_column()
        self.pnn.add_column()
        
        x = torch.randn(16, 784)
        
        # 测试第一列
        out1 = self.pnn(x, task_id=0)
        self.assertEqual(out1.shape, (16, 10))
        
        # 测试第二列
        out2 = self.pnn(x, task_id=1)
        self.assertEqual(out2.shape, (16, 10))
    
    def test_frozen_parameters(self):
        """测试参数冻结"""
        self.pnn.add_column()
        
        # 检查第一列的参数需要梯度
        for param in self.pnn.columns[0].parameters():
            self.assertTrue(param.requires_grad)
        
        # 添加第二列
        self.pnn.add_column()
        
        # 检查第一列的参数被冻结
        for param in self.pnn.columns[0].parameters():
            self.assertFalse(param.requires_grad)
        
        # 检查第二列的参数需要梯度
        for param in self.pnn.columns[1].parameters():
            self.assertTrue(param.requires_grad)


class TestContinualLearner(unittest.TestCase):
    """测试持续学习主类"""
    
    def setUp(self):
        self.config = ContinualLearningConfig(
            device='cpu',
            epochs_per_task=1,
            batch_size=32
        )
        self.model = SimpleMLP(784, 128, 10)
        
        # 创建简单数据集
        X = torch.randn(200, 784)
        y = torch.randint(0, 10, (200,))
        self.dataset = TensorDataset(X, y)
    
    def test_initialization(self):
        """测试初始化"""
        learner = ContinualLearner(self.model, self.config, strategy='ewc')
        self.assertIsNotNone(learner.ewc)
        self.assertIsNone(learner.memory_buffer)
        
        learner2 = ContinualLearner(self.model, self.config, strategy='replay')
        self.assertIsNone(learner2.ewc)
        self.assertIsNotNone(learner2.memory_buffer)
    
    def test_train_task(self):
        """测试任务训练"""
        learner = ContinualLearner(self.model, self.config, strategy='combined')
        
        history = learner.train_task(self.dataset, task_name="Test Task")
        
        # 检查训练历史
        self.assertIn('loss', history)
        self.assertIn('accuracy', history)
        self.assertEqual(len(history['loss']), self.config.epochs_per_task)
    
    def test_evaluate(self):
        """测试评估"""
        learner = ContinualLearner(self.model, self.config, strategy='ewc')
        learner.train_task(self.dataset, task_name="Task 1")
        
        result = learner.evaluate(self.dataset)
        
        self.assertIn('accuracy', result)
        self.assertIn('loss', result)
        self.assertGreaterEqual(result['accuracy'], 0)
        self.assertLessEqual(result['accuracy'], 100)
    
    def test_evaluate_all_tasks(self):
        """测试评估所有任务"""
        learner = ContinualLearner(self.model, self.config, strategy='combined')
        
        # 训练两个任务
        learner.train_task(self.dataset, task_name="Task 1")
        learner.train_task(self.dataset, task_name="Task 2")
        
        # 评估
        datasets = [self.dataset, self.dataset]
        results = learner.evaluate_all_tasks(datasets)
        
        self.assertEqual(len(results), 2)
        self.assertIn(0, results)
        self.assertIn(1, results)


class TestMetaLearner(unittest.TestCase):
    """测试元学习器"""
    
    def setUp(self):
        self.config = ContinualLearningConfig(device='cpu')
        self.model = SimpleMLP(784, 128, 10)
        self.meta_learner = MetaLearner(self.model, self.config)
    
    def test_meta_update(self):
        """测试元更新"""
        # 创建支持集和查询集
        support_x = torch.randn(32, 784)
        support_y = torch.randint(0, 10, (32,))
        query_x = torch.randn(32, 784)
        query_y = torch.randint(0, 10, (32,))
        
        support_set = (support_x, support_y)
        query_set = (query_x, query_y)
        
        loss = self.meta_learner.meta_update(support_set, query_set)
        
        # 检查损失是标量
        self.assertIsInstance(loss, float)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        config = ContinualLearningConfig(
            device='cpu',
            epochs_per_task=1,
            batch_size=32,
            ewc_lambda=10.0,
            memory_buffer_size=50
        )
        
        model = SimpleMLP(784, 64, 10)
        learner = ContinualLearner(model, config, strategy='combined')
        
        # 创建多个任务
        for i in range(3):
            X = torch.randn(100, 784)
            y = torch.randint(0, 10, (100,))
            dataset = TensorDataset(X, y)
            
            history = learner.train_task(dataset, task_name=f"Task {i}")
            self.assertIn('loss', history)
        
        # 评估所有任务
        test_datasets = [dataset] * 3
        results = learner.evaluate_all_tasks(test_datasets)
        
        self.assertEqual(len(results), 3)
        
        # 检查平均准确率
        avg_acc = np.mean([r['accuracy'] for r in results.values()])
        self.assertGreater(avg_acc, 0)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
