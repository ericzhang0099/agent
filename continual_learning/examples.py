"""
Continual Learning Examples - 持续学习示例脚本
===============================================

包含多个经典持续学习基准测试的实现：
1. Permuted MNIST
2. Split MNIST
3. Split CIFAR-10/100

Author: AI Research Team
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Dict, Tuple
import copy
import sys
import os

# 添加父目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from continual_learner import (
    ContinualLearner, ContinualLearningConfig,
    SimpleMLP, SimpleConvNet, create_permuted_mnist_task,
    create_split_mnist_tasks, compute_average_accuracy
)


def get_mnist_datasets(data_dir='./data'):
    """获取MNIST数据集"""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    
    train_dataset = datasets.MNIST(
        data_dir, train=True, download=True, transform=transform
    )
    test_dataset = datasets.MNIST(
        data_dir, train=False, download=True, transform=transform
    )
    
    return train_dataset, test_dataset


def get_cifar10_datasets(data_dir='./data'):
    """获取CIFAR-10数据集"""
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    train_dataset = datasets.CIFAR10(
        data_dir, train=True, download=True, transform=transform_train
    )
    test_dataset = datasets.CIFAR10(
        data_dir, train=False, download=True, transform=transform_test
    )
    
    return train_dataset, test_dataset


class PermutedMNISTExperiment:
    """Permuted MNIST实验"""
    
    def __init__(self, num_tasks=10, strategy='ewc', config=None):
        self.num_tasks = num_tasks
        self.strategy = strategy
        self.config = config or ContinualLearningConfig()
        
        # 获取基础数据集
        self.base_train, self.base_test = get_mnist_datasets()
        
        # 创建任务
        self.train_tasks = []
        self.test_tasks = []
        self.permutations = []
        
        for i in range(num_tasks):
            perm = np.random.permutation(784)
            self.permutations.append(perm)
            
            train_task = create_permuted_mnist_task(self.base_train, perm)
            test_task = create_permuted_mnist_task(self.base_test, perm)
            
            self.train_tasks.append(train_task)
            self.test_tasks.append(test_task)
    
    def run(self):
        """运行实验"""
        print(f"\n{'='*80}")
        print(f"Permuted MNIST Experiment")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Strategy: {self.strategy}")
        print(f"{'='*80}\n")
        
        # 创建模型
        model = SimpleMLP(
            input_dim=784,
            hidden_dim=400,
            output_dim=10,
            num_hidden_layers=2
        )
        
        # 创建持续学习器
        learner = ContinualLearner(model, self.config, strategy=self.strategy)
        
        # 记录每个任务在每个阶段后的准确率
        task_accuracies = [[] for _ in range(self.num_tasks)]
        
        # 顺序训练每个任务
        for task_id in range(self.num_tasks):
            print(f"\n--- Training Task {task_id + 1}/{self.num_tasks} ---")
            
            # 训练当前任务
            history = learner.train_task(
                self.train_tasks[task_id],
                task_name=f"Permutation {task_id + 1}"
            )
            
            # 评估所有已训练任务
            print(f"\nEvaluating all tasks after training Task {task_id + 1}:")
            for eval_task_id in range(task_id + 1):
                result = learner.evaluate(self.test_tasks[eval_task_id])
                task_accuracies[eval_task_id].append(result['accuracy'])
                print(f"  Task {eval_task_id + 1}: {result['accuracy']:.2f}%")
        
        # 最终评估
        print(f"\n{'='*80}")
        print("Final Evaluation")
        print(f"{'='*80}")
        
        final_results = learner.evaluate_all_tasks(self.test_tasks)
        avg_acc = compute_average_accuracy(final_results)
        
        # 计算遗忘率
        forgetting_rates = []
        for task_id in range(self.num_tasks):
            max_acc = max(task_accuracies[task_id])
            final_acc = task_accuracies[task_id][-1]
            forgetting = max_acc - final_acc
            forgetting_rates.append(forgetting)
        
        avg_forgetting = np.mean(forgetting_rates)
        
        print(f"\nAverage Accuracy: {avg_acc:.2f}%")
        print(f"Average Forgetting: {avg_forgetting:.2f}%")
        
        return {
            'final_results': final_results,
            'task_accuracies': task_accuracies,
            'avg_accuracy': avg_acc,
            'avg_forgetting': avg_forgetting
        }


class SplitMNISTExperiment:
    """Split MNIST实验"""
    
    def __init__(self, num_tasks=5, strategy='ewc', config=None):
        self.num_tasks = num_tasks
        self.strategy = strategy
        self.config = config or ContinualLearningConfig()
        
        # 获取基础数据集
        base_train, base_test = get_mnist_datasets()
        
        # 创建任务
        self.train_tasks = create_split_mnist_tasks(base_train, num_tasks)
        self.test_tasks = create_split_mnist_tasks(base_test, num_tasks)
    
    def run(self):
        """运行实验"""
        print(f"\n{'='*80}")
        print(f"Split MNIST Experiment")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Strategy: {self.strategy}")
        print(f"{'='*80}\n")
        
        # 创建模型（输出维度为每任务的类别数）
        classes_per_task = 10 // self.num_tasks
        model = SimpleMLP(
            input_dim=784,
            hidden_dim=256,
            output_dim=classes_per_task,
            num_hidden_layers=2
        )
        
        # 创建持续学习器
        learner = ContinualLearner(model, self.config, strategy=self.strategy)
        
        # 记录每个任务在每个阶段后的准确率
        task_accuracies = [[] for _ in range(self.num_tasks)]
        
        # 顺序训练每个任务
        for task_id in range(self.num_tasks):
            print(f"\n--- Training Task {task_id + 1}/{self.num_tasks} ---")
            print(f"Classes: {task_id * classes_per_task}-{(task_id + 1) * classes_per_task - 1}")
            
            # 训练当前任务
            history = learner.train_task(
                self.train_tasks[task_id],
                task_name=f"Classes {task_id * classes_per_task}-{(task_id + 1) * classes_per_task - 1}"
            )
            
            # 评估所有已训练任务
            print(f"\nEvaluating all tasks after training Task {task_id + 1}:")
            for eval_task_id in range(task_id + 1):
                result = learner.evaluate(self.test_tasks[eval_task_id])
                task_accuracies[eval_task_id].append(result['accuracy'])
                print(f"  Task {eval_task_id + 1}: {result['accuracy']:.2f}%")
        
        # 最终评估
        print(f"\n{'='*80}")
        print("Final Evaluation")
        print(f"{'='*80}")
        
        final_results = learner.evaluate_all_tasks(self.test_tasks)
        avg_acc = compute_average_accuracy(final_results)
        
        # 计算遗忘率
        forgetting_rates = []
        for task_id in range(self.num_tasks):
            max_acc = max(task_accuracies[task_id])
            final_acc = task_accuracies[task_id][-1]
            forgetting = max_acc - final_acc
            forgetting_rates.append(forgetting)
        
        avg_forgetting = np.mean(forgetting_rates)
        
        print(f"\nAverage Accuracy: {avg_acc:.2f}%")
        print(f"Average Forgetting: {avg_forgetting:.2f}%")
        
        return {
            'final_results': final_results,
            'task_accuracies': task_accuracies,
            'avg_accuracy': avg_acc,
            'avg_forgetting': avg_forgetting
        }


class SplitCIFAR10Experiment:
    """Split CIFAR-10实验"""
    
    def __init__(self, num_tasks=5, strategy='ewc', config=None):
        self.num_tasks = num_tasks
        self.strategy = strategy
        self.config = config or ContinualLearningConfig()
        
        # 获取基础数据集
        base_train, base_test = get_cifar10_datasets()
        
        # 创建任务
        self.train_tasks = self._create_split_tasks(base_train)
        self.test_tasks = self._create_split_tasks(base_test)
    
    def _create_split_tasks(self, dataset):
        """创建Split CIFAR-10任务"""
        classes_per_task = 10 // self.num_tasks
        tasks = []
        
        # 获取所有数据和标签
        all_data = []
        all_labels = []
        for img, label in dataset:
            all_data.append(img)
            all_labels.append(label)
        
        all_data = torch.stack(all_data)
        all_labels = torch.tensor(all_labels)
        
        for task_id in range(self.num_tasks):
            start_class = task_id * classes_per_task
            end_class = start_class + classes_per_task
            
            # 选择属于当前任务的样本
            mask = (all_labels >= start_class) & (all_labels < end_class)
            task_data = all_data[mask]
            task_labels = all_labels[mask]
            
            # 重新映射标签
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
    
    def run(self):
        """运行实验"""
        print(f"\n{'='*80}")
        print(f"Split CIFAR-10 Experiment")
        print(f"Number of tasks: {self.num_tasks}")
        print(f"Strategy: {self.strategy}")
        print(f"{'='*80}\n")
        
        # 创建卷积模型
        classes_per_task = 10 // self.num_tasks
        model = SimpleConvNet(
            input_channels=3,
            num_classes=classes_per_task,
            hidden_dim=128
        )
        
        # 创建持续学习器
        learner = ContinualLearner(model, self.config, strategy=self.strategy)
        
        # 记录每个任务在每个阶段后的准确率
        task_accuracies = [[] for _ in range(self.num_tasks)]
        
        # 顺序训练每个任务
        for task_id in range(self.num_tasks):
            print(f"\n--- Training Task {task_id + 1}/{self.num_tasks} ---")
            print(f"Classes: {task_id * classes_per_task}-{(task_id + 1) * classes_per_task - 1}")
            
            # 训练当前任务
            history = learner.train_task(
                self.train_tasks[task_id],
                task_name=f"Classes {task_id * classes_per_task}-{(task_id + 1) * classes_per_task - 1}"
            )
            
            # 评估所有已训练任务
            print(f"\nEvaluating all tasks after training Task {task_id + 1}:")
            for eval_task_id in range(task_id + 1):
                result = learner.evaluate(self.test_tasks[eval_task_id])
                task_accuracies[eval_task_id].append(result['accuracy'])
                print(f"  Task {eval_task_id + 1}: {result['accuracy']:.2f}%")
        
        # 最终评估
        print(f"\n{'='*80}")
        print("Final Evaluation")
        print(f"{'='*80}")
        
        final_results = learner.evaluate_all_tasks(self.test_tasks)
        avg_acc = compute_average_accuracy(final_results)
        
        # 计算遗忘率
        forgetting_rates = []
        for task_id in range(self.num_tasks):
            max_acc = max(task_accuracies[task_id])
            final_acc = task_accuracies[task_id][-1]
            forgetting = max_acc - final_acc
            forgetting_rates.append(forgetting)
        
        avg_forgetting = np.mean(forgetting_rates)
        
        print(f"\nAverage Accuracy: {avg_acc:.2f}%")
        print(f"Average Forgetting: {avg_forgetting:.2f}%")
        
        return {
            'final_results': final_results,
            'task_accuracies': task_accuracies,
            'avg_accuracy': avg_acc,
            'avg_forgetting': avg_forgetting
        }


def compare_strategies():
    """比较不同持续学习策略的性能"""
    
    print("\n" + "="*80)
    print("Comparing Continual Learning Strategies")
    print("="*80)
    
    strategies = ['ewc', 'replay', 'combined']
    num_tasks = 5
    
    results = {}
    
    for strategy in strategies:
        print(f"\n{'='*60}")
        print(f"Running with strategy: {strategy}")
        print(f"{'='*60}")
        
        config = ContinualLearningConfig(
            ewc_lambda=100.0 if strategy in ['ewc', 'combined'] else 0.0,
            memory_buffer_size=500 if strategy in ['replay', 'combined'] else 0,
            learning_rate=0.001,
            epochs_per_task=3,
            batch_size=128
        )
        
        experiment = SplitMNISTExperiment(
            num_tasks=num_tasks,
            strategy=strategy,
            config=config
        )
        
        result = experiment.run()
        results[strategy] = result
    
    # 打印比较结果
    print("\n" + "="*80)
    print("Strategy Comparison Summary")
    print("="*80)
    print(f"{'Strategy':<15} {'Avg Accuracy':<15} {'Avg Forgetting':<15}")
    print("-"*45)
    for strategy, result in results.items():
        print(f"{strategy:<15} {result['avg_accuracy']:<15.2f} {result['avg_forgetting']:<15.2f}")
    
    return results


def plot_results(results: Dict, save_path: str = None):
    """绘制实验结果"""
    
    task_accuracies = results['task_accuracies']
    num_tasks = len(task_accuracies)
    
    plt.figure(figsize=(12, 6))
    
    # 绘制每个任务的准确率变化
    for task_id in range(num_tasks):
        plt.plot(
            range(task_id, num_tasks),
            task_accuracies[task_id],
            marker='o',
            label=f'Task {task_id + 1}'
        )
    
    plt.xlabel('Training Stage')
    plt.ylabel('Accuracy (%)')
    plt.title('Task Accuracies Over Training Stages')
    plt.legend()
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Plot saved to {save_path}")
    else:
        plt.show()


def main():
    """主函数"""
    
    import argparse
    
    parser = argparse.ArgumentParser(description='Continual Learning Experiments')
    parser.add_argument('--experiment', type=str, default='split_mnist',
                       choices=['permuted_mnist', 'split_mnist', 'split_cifar10', 'compare'],
                       help='Experiment to run')
    parser.add_argument('--strategy', type=str, default='ewc',
                       choices=['ewc', 'replay', 'combined', 'pnn'],
                       help='Continual learning strategy')
    parser.add_argument('--num_tasks', type=int, default=5,
                       help='Number of tasks')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Epochs per task')
    parser.add_argument('--batch_size', type=int, default=128,
                       help='Batch size')
    parser.add_argument('--ewc_lambda', type=float, default=100.0,
                       help='EWC regularization strength')
    parser.add_argument('--memory_size', type=int, default=500,
                       help='Memory buffer size')
    
    args = parser.parse_args()
    
    # 创建配置
    config = ContinualLearningConfig(
        ewc_lambda=args.ewc_lambda,
        memory_buffer_size=args.memory_size,
        learning_rate=0.001,
        epochs_per_task=args.epochs,
        batch_size=args.batch_size
    )
    
    # 运行实验
    if args.experiment == 'permuted_mnist':
        experiment = PermutedMNISTExperiment(
            num_tasks=args.num_tasks,
            strategy=args.strategy,
            config=config
        )
        results = experiment.run()
        plot_results(results, save_path='permuted_mnist_results.png')
        
    elif args.experiment == 'split_mnist':
        experiment = SplitMNISTExperiment(
            num_tasks=args.num_tasks,
            strategy=args.strategy,
            config=config
        )
        results = experiment.run()
        plot_results(results, save_path='split_mnist_results.png')
        
    elif args.experiment == 'split_cifar10':
        experiment = SplitCIFAR10Experiment(
            num_tasks=args.num_tasks,
            strategy=args.strategy,
            config=config
        )
        results = experiment.run()
        plot_results(results, save_path='split_cifar10_results.png')
        
    elif args.experiment == 'compare':
        results = compare_strategies()


if __name__ == '__main__':
    main()
