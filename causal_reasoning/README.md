# 因果推理模块 - 使用说明

## 概述

本模块实现了因果推理的核心功能，基于Judea Pearl的因果推断理论。主要包含以下组件：

1. **因果图模型 (CausalGraph)** - 有向无环图(DAG)表示和操作
2. **Do-Calculus引擎 (DoCalculusEngine)** - Pearl的三条do-calculus规则
3. **反事实引擎 (CounterfactualEngine)** - 反事实推理
4. **因果效应估计器 (CausalEffectEstimator)** - 因果效应估计
5. **因果发现算法 (PC/GES)** - 从数据中发现因果结构

## 安装依赖

```bash
pip install numpy pandas scipy scikit-learn matplotlib
```

## 快速开始

### 1. 基本使用

```python
from causal_reasoner import CausalGraph, CausalEffectEstimator
import pandas as pd
import numpy as np

# 创建因果图
graph = CausalGraph()
graph.add_edge('Z', 'X')
graph.add_edge('X', 'Y')
graph.add_edge('Z', 'Y')

# 生成数据
data = pd.DataFrame({
    'Z': np.random.normal(0, 1, 1000),
    'X': np.random.normal(0, 1, 1000),
    'Y': np.random.normal(0, 1, 1000)
})

# 估计因果效应
estimator = CausalEffectEstimator(graph, data)
ate = estimator.estimate_ate('X', 'Y', method='backdoor')
print(f"估计的ATE: {ate}")
```

### 2. 运行示例

```bash
python causal_reasoner.py
```

或运行详细示例：

```bash
python examples.py
```

## 详细API文档

### CausalGraph 类

#### 方法

- `add_node(node: str)` - 添加节点
- `add_edge(source: str, target: str, edge_type: EdgeType)` - 添加边
- `get_parents(node: str) -> Set[str]` - 获取父节点
- `get_children(node: str) -> Set[str]` - 获取子节点
- `get_ancestors(node: str) -> Set[str]` - 获取祖先节点
- `get_descendants(node: str) -> Set[str]` - 获取后代节点
- `is_d_separated(x: str, y: str, conditioning_set: Set[str]) -> bool` - D-分离测试
- `find_backdoor_adjustment_set(treatment: str, outcome: str) -> Optional[Set[str]]` - 寻找后门调整集

#### 示例

```python
graph = CausalGraph()
graph.add_edge('Z', 'X')
graph.add_edge('X', 'Y')
graph.add_edge('Z', 'Y')

# 检查D-分离
is_sep = graph.is_d_separated('X', 'Y', {'Z'})
print(f"给定Z，X和Y是否d-分离? {is_sep}")

# 寻找后门调整集
adjustment_set = graph.find_backdoor_adjustment_set('X', 'Y')
print(f"后门调整集: {adjustment_set}")
```

### CausalEffectEstimator 类

#### 方法

- `estimate_ate(treatment: str, outcome: str, method: str, adjustment_set: Set[str]) -> float` - 估计ATE

#### 支持的估计方法

- `backdoor` - 后门调整
- `ipw` - 逆概率加权

#### 示例

```python
estimator = CausalEffectEstimator(graph, data)

# 后门调整
ate = estimator.estimate_ate('X', 'Y', method='backdoor')

# IPW
ate_ipw = estimator.estimate_ate('X', 'Y', method='ipw')
```

### DoCalculusEngine 类

#### 方法

- `apply_rule1(y: str, x: Set[str], z: str, w: Set[str]) -> bool` - 应用规则1
- `apply_rule2(y: str, x: Set[str], z: str, w: Set[str]) -> bool` - 应用规则2
- `apply_rule3(y: str, x: Set[str], z: str, w: Set[str]) -> bool` - 应用规则3
- `identify_causal_effect(treatment: str, outcome: str) -> Optional[str]` - 识别因果效应

#### 示例

```python
do_engine = DoCalculusEngine(graph)
identified = do_engine.identify_causal_effect('X', 'Y')
print(f"识别的因果效应: {identified}")
```

### PCAlgorithm 类

#### 方法

- `run() -> CausalGraph` - 运行PC算法

#### 示例

```python
pc = PCAlgorithm(data, alpha=0.05)
discovered_graph = pc.run()
print(discovered_graph)
```

### GESAlgorithm 类

#### 方法

- `run() -> CausalGraph` - 运行GES算法

#### 示例

```python
ges = GESAlgorithm(data, score_type='bic')
discovered_graph = ges.run()
print(discovered_graph)
```

## 核心概念

### 1. 因果层级

因果推理的三个层级：
- **关联 (Association)**: P(Y|X) - 观察到X后对Y的了解
- **干预 (Intervention)**: P(Y|do(X)) - 如果我做X，Y会怎样
- **反事实 (Counterfactuals)**: P(Y_x|X',Y') - 如果我做了X，会怎样

### 2. 后门准则

变量集Z满足后门准则相对于(X,Y)的条件：
1. Z中没有节点是X的后代
2. Z阻塞了所有从X到Y的后门路径

调整公式：
```
P(Y | do(X=x)) = ∑_z P(Y | X=x, Z=z) P(Z=z)
```

### 3. Do-Calculus三条规则

**规则1**（观测的插入/删除）：
```
P(y | do(x), z, w) = P(y | do(x), w) 如果 Y ⟂ Z | X, W 在 G_X̄ 中
```

**规则2**（行动/观测交换）：
```
P(y | do(x), do(z), w) = P(y | do(x), z, w) 如果 Y ⟂ Z | X, W 在 G_X̄,Z̲ 中
```

**规则3**（行动的插入/删除）：
```
P(y | do(x), do(z), w) = P(y | do(x), w) 如果 Y ⟂ Z | X, W 在 G_X̄,Z̄(W) 中
```

### 4. 因果发现算法

**PC算法**：
- 基于约束的方法
- 使用条件独立性测试
- 输出CPDAG（Completed Partially Directed Acyclic Graph）

**GES算法**：
- 基于评分的方法
- 使用BIC等评分函数
- 贪婪搜索最优图结构

## 示例场景

### 场景1: 简单混杂

```
Z -> X -> Y
Z -> Y
```

Z是混杂因素，需要使用后门调整控制Z。

### 场景2: 中介分析

```
X -> M -> Y
X -> Y
```

M是中介变量，可以分解直接效应和间接效应。

### 场景3: 前门准则

```
X -> M -> Y
U -> X, U -> Y (未观测)
```

存在未观测混杂U，使用前门准则识别因果效应。

## 注意事项

1. **样本量**：因果发现算法需要足够大的样本量
2. **忠实性假设**：观测数据的条件独立性应反映真实的图结构
3. **因果充分性**：假设没有未观测的混杂因素（除非使用FCI等算法）
4. **模型正确性**：结构方程模型应正确指定

## 参考文献

1. Pearl, J. (2009). Causality: Models, Reasoning, and Inference.
2. Pearl, J., & Mackenzie, D. (2018). The Book of Why.
3. Peters, J., Janzing, D., & Schölkopf, B. (2017). Elements of Causal Inference.
4. Spirtes, P., Glymour, C., & Scheines, R. (2000). Causation, Prediction, and Search.

## 许可证

MIT License
