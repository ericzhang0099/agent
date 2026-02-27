# Continual Learning Framework

持续学习（Continual Learning）框架 - 在线进化能力与灾难性遗忘防护

## 项目简介

本项目实现了一个完整的持续学习框架，支持多种先进的持续学习算法：

- **弹性权重整合 (EWC)** - 基于Fisher信息的正则化方法
- **渐进式神经网络 (PNN)** - 零遗忘的架构方法
- **经验回放 (Experience Replay)** - 基于记忆的方法
- **元学习 (Meta-Learning)** - 学习如何学习

## 安装要求

```bash
pip install torch torchvision numpy matplotlib
```

## 快速开始

### 1. 基础使用

```python
from continual_learner import (
    ContinualLearner, ContinualLearningConfig,
    SimpleMLP
)

# 配置
config = ContinualLearningConfig(
    ewc_lambda=100.0,
    memory_buffer_size=500,
    learning_rate=0.001,
    epochs_per_task=5
)

# 创建模型
model = SimpleMLP(input_dim=784, hidden_dim=256, output_dim=10)

# 创建持续学习器
learner = ContinualLearner(model, config, strategy='combined')

# 顺序训练任务
for task_id, (train_data, test_data) in enumerate(tasks):
    learner.train_task(train_data, task_name=f"Task {task_id}")
    
# 评估所有任务
results = learner.evaluate_all_tasks(test_datasets)
```

### 2. 运行示例

```bash
# Permuted MNIST 实验
python examples.py --experiment permuted_mnist --strategy ewc --num_tasks 10

# Split MNIST 实验
python examples.py --experiment split_mnist --strategy combined --num_tasks 5

# 比较不同策略
python examples.py --experiment compare
```

## 支持的策略

| 策略 | 描述 | 命令行参数 |
|------|------|-----------|
| `ewc` | 弹性权重整合 | `--strategy ewc` |
| `replay` | 经验回放 | `--strategy replay` |
| `combined` | EWC + 回放 | `--strategy combined` |
| `pnn` | 渐进式神经网络 | `--strategy pnn` |

## 项目结构

```
continual_learning/
├── continual_learner.py    # 核心框架实现
├── examples.py              # 实验示例脚本
├── research_report.md       # 详细研究报告
└── README.md                # 本文件
```

## 核心组件

### ContinualLearner

主类，集成多种持续学习策略：

```python
learner = ContinualLearner(
    model=model,
    config=config,
    strategy='combined'  # 选择策略
)
```

### EWC

弹性权重整合实现：

```python
from continual_learner import EWC

ewc = EWC(model, config)
ewc.after_task(dataset)  # 任务结束后计算Fisher矩阵
ewc_loss = ewc.compute_ewc_loss()  # 计算EWC损失
```

### MemoryBuffer

经验回放缓冲区：

```python
from continual_learner import MemoryBuffer

buffer = MemoryBuffer(buffer_size=1000, sampling_strategy='reservoir')
buffer.add(sample, label, logits)
samples, labels, logits = buffer.sample(batch_size)
```

### ProgressiveNeuralNetwork

渐进式神经网络：

```python
from continual_learner import ProgressiveNeuralNetwork

pnn = ProgressiveNeuralNetwork(
    input_dim=784,
    hidden_dims=[256, 256],
    output_dim=10,
    config=config
)
task_id = pnn.add_column()  # 添加新任务列
output = pnn(input, task_id)
```

## 配置参数

```python
from continual_learner import ContinualLearningConfig

config = ContinualLearningConfig(
    # EWC参数
    ewc_lambda=100.0,           # EWC正则化强度
    ewc_estimate_type='true',   # Fisher估计类型
    
    # 记忆缓冲区参数
    memory_buffer_size=1000,           # 缓冲区大小
    memory_sampling_strategy='reservoir',  # 采样策略
    
    # PNN参数
    pnn_adapter_hidden_dim=256,  # 适配器隐藏层维度
    
    # 训练参数
    learning_rate=0.001,
    batch_size=128,
    epochs_per_task=10,
    device='cuda'  # 或 'cpu'
)
```

## 基准测试

### Permuted MNIST

每个任务对MNIST像素进行随机排列，测试模型处理输入分布变化的能力。

```bash
python examples.py --experiment permuted_mnist --num_tasks 10 --epochs 5
```

### Split MNIST

将10个类别分成5个任务（每任务2类），测试类别增量学习能力。

```bash
python examples.py --experiment split_mnist --num_tasks 5 --epochs 5
```

### Split CIFAR-10

更复杂的图像分类任务。

```bash
python examples.py --experiment split_cifar10 --num_tasks 5 --epochs 10
```

## 评估指标

- **平均准确率 (Average Accuracy)**：所有任务的平均测试准确率
- **遗忘率 (Forgetting Rate)**：旧任务性能下降的程度
- **后向迁移 (Backward Transfer)**：学习新任务对旧任务的影响

## 预期性能

| 方法 | Permuted MNIST | Split MNIST |
|------|----------------|-------------|
| 基线 (无CL) | ~50% | ~20% |
| EWC | ~85% | ~70% |
| 回放 | ~90% | ~85% |
| EWC+回放 | ~92% | ~88% |
| PNN | ~95% | ~90% |

## 参考文献

1. Kirkpatrick et al. "Overcoming catastrophic forgetting in neural networks" (2017)
2. Rusu et al. "Progressive Neural Networks" (2016)
3. Shin et al. "Continual learning with deep generative replay" (2017)
4. Finn et al. "Model-Agnostic Meta-Learning" (2017)
5. Lopez-Paz & Ranzato "Gradient Episodic Memory for Continual Learning" (2017)

## 许可证

MIT License

## 贡献

欢迎提交Issue和Pull Request！

## 联系方式

如有问题，请通过GitHub Issues联系。
