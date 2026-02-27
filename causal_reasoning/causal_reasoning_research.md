# 因果推理（Causal Reasoning）深度研究文档

## 目录
1. [引言](#引言)
2. [Judea Pearl因果推断理论](#1-judea-pearl因果推断理论)
3. [因果图模型（Causal Graphs）](#2-因果图模型causal-graphs)
4. [反事实推理（Counterfactuals）](#3-反事实推理counterfactuals)
5. [Do-Calculus计算](#4-do-calculus计算)
6. [因果发现算法（PC/GES）](#5-因果发现算法pcges)
7. [参考文献](#参考文献)

---

## 引言

因果推理是理解干预后果的科学，需要超越纯关联性分析的假设。在现代机器学习和人工智能时代，因果推理能力对于构建不仅具有预测性，而且可信、透明、对分布变化具有鲁棒性的系统至关重要。

因果推理的三个主要框架：
- **潜在结果框架（Potential Outcomes）**：由Neyman和Rubin发展
- **结构方程模型（Structural Equation Models, SEM）**：由Haavelmo和Pearl发展
- **有向无环图（Directed Acyclic Graphs, DAG）**：由Spirtes和Pearl发展

---

## 1. Judea Pearl因果推断理论

### 1.1 因果层级（Ladder of Causation）

Judea Pearl提出了因果推理的三个层级：

| 层级 | 名称 | 问题类型 | 示例 |
|------|------|----------|------|
| 1 | 关联（Association） | 观察到X后，对Y的了解？ | P(Y\|X) |
| 2 | 干预（Intervention） | 如果我做X，Y会怎样？ | P(Y\|do(X)) |
| 3 | 反事实（Counterfactuals） | 如果我做了X，会怎样？ | P(Y_x\|X', Y') |

### 1.2 核心概念

**因果效应（Causal Effect）**：
- 个体因果效应：Y_i(1) - Y_i(0)
- 平均因果效应（ACE）：E[Y(1) - Y(0)]

**基本问题**：
我们无法同时观察到同一个体的两个潜在结果（Fundamental Problem of Causal Inference）。

### 1.3 关键假设

**SUTVA（Stable Unit Treatment Value Assumption）**：
1. 没有多种版本的处理
2. 单位之间没有干扰（no interference）

**一致性假设（Consistency）**：
```
Y = Y(a) if A = a, or equivalently, Y = Y(A)
```

**无未观测混杂（No Unmeasured Confounding）**：
```
A ⟂ Y(a) | L, for a = 0, 1
```

---

## 2. 因果图模型（Causal Graphs）

### 2.1 有向无环图（DAG）

DAG G = (V, E) 包含：
- V = {V₁, ..., Vₚ}：节点集合（随机变量）
- E ⊆ V × V：有向边集合

**基本术语**：
- **父节点（Parents）**：Pa(Vⱼ) = {Vₖ ∈ V : (Vₖ → Vⱼ) ∈ E}
- **祖先（Ancestors）**：An(Vⱼ) = 所有能到达Vⱼ的节点
- **后代（Descendants）**：De(Vⱼ) = 所有Vⱼ能到达的节点

### 2.2 D-分离（D-Separation）

路径被C阻塞的条件：
1. **链（Chain）**：X → M → Y 或 X ← M ← Y，其中 M ∈ C
2. **分叉（Fork）**：X ← M → Y，其中 M ∈ C
3. **碰撞点（Collider）**：X → M ← Y，其中 M ∉ C 且 De(M) ∩ C = ∅

如果所有从A到B的路径都被C阻塞，则A和B被C d-分离。

### 2.3 马尔可夫性质

**马尔可夫分解**：
```
P(V₁, ..., Vₚ) = ∏ⱼ P(Vⱼ | Pa(Vⱼ))
```

**全局马尔可夫性质**：
如果C d-分离A和B，则 A ⟂ B | C。

**忠实性（Faithfulness）**：
A ⟂ B | C ⇔ A和B被C d-分离。

### 2.4 后门准则（Backdoor Criterion）

变量集L满足后门准则相对于(A, Y)的条件：
1. L中没有节点是A的后代
2. L阻塞了所有从A到Y的后门路径

如果L满足后门准则，则：
```
P(Y | do(A=a)) = ∑ₗ P(Y | A=a, L=l) P(L=l)
```

### 2.5 因果DAG模型

**定义**：因果DAG模型由DAG G和干预分布族{P_do(vₛ)}组成，满足：
1. 观测分布P满足马尔可夫分解
2. **模块性（Modularity）**：干预后的分布满足截断分解

**截断分解（Truncated Factorization）**：
```
P_do(vₛ)(V_{[p]\\S}) = ∏_{k∉S} P(Vₖ | V_{Pa(Vₖ)\\S}, V_{Pa(Vₖ)∩S}=v_{Pa(Vₖ)∩S})
```

---

## 3. 反事实推理（Counterfactuals）

### 3.1 反事实定义

反事实是在给定实际发生情况的前提下，询问如果事情不同会怎样。

**结构方程模型中的反事实**：
```
Y(a) = f_Y(a, ε_Y)
```

其中ε_Y是外生噪声，在反事实世界中保持不变。

### 3.2 嵌套反事实

对于中介变量M：
```
Y(a, M(a')) = f_Y(a, f_M(a', ε_M), ε_Y)
```

这表示：如果处理设为a，且中介变量取其在处理a'下的值，结果Y会怎样。

### 3.3 反事实查询类型

1. **治疗对治疗的效应（ETT）**：
   ```
   E[Y(1) - Y(0) | A = 1]
   ```

2. **必要性概率（PN）**：
   ```
   P(Y(0) = 0 | A = 1, Y = 1)
   ```

3. **充分性概率（PS）**：
   ```
   P(Y(1) = 1 | A = 0, Y = 0)
   ```

4. **必要且充分概率（PNS）**：
   ```
   P(Y(1) = 1, Y(0) = 0)
   ```

### 3.4 反事实识别

反事实的识别通常需要比干预更强的假设，因为它依赖于潜在结果之间的依赖关系。

**在NPSEM-IE下的识别**：
如果模型是NPSEM with Independent Errors，则可以通过结构方程计算反事实。

---

## 4. Do-Calculus计算

### 4.1 符号定义

- **G**：因果DAG
- **G_X̄**：从G中移除所有进入X的边后的子图
- **G_X̲**：从G中移除所有从X出去的边后的子图
- **do(x)**：将变量X干预为值x的操作符

### 4.2 三条规则

**规则1：观测的插入/删除**
```
P(y | do(x), z, w) = P(y | do(x), w) 
如果 Y ⟂ Z | X, W 在 G_X̄ 中
```

**规则2：行动/观测交换**
```
P(y | do(x), do(z), w) = P(y | do(x), z, w)
如果 Y ⟂ Z | X, W 在 G_X̄,Z̲ 中
```

简化形式（后门准则）：
```
P(y | do(x), w) = P(y | x, w)
如果 Y ⟂ X | W 在 G_X̲ 中
```

**规则3：行动的插入/删除**
```
P(y | do(x), do(z), w) = P(y | do(x), w)
如果 Y ⟂ Z | X, W 在 G_X̄,Z̄(W) 中
```

其中Z(W)是G_X̄中不是任何W节点祖先的Z节点集合。

### 4.3 后门调整公式的推导

使用do-calculus从P(Y | do(X))推导后门调整公式：

```
P(y | do(x)) = ∑_z P(y | do(x), z) P(z | do(x))    [全概率公式]
             = ∑_z P(y | do(x), z) P(z)             [规则3: Z ⟂ X 在 G_X̄中]
             = ∑_z P(y | x, z) P(z)                 [规则2: Y ⟂ X | Z 在 G_X̲中]
```

### 4.4 前门准则（Front-door Criterion）

当存在未观测混杂时，前门准则允许识别因果效应。

变量集M满足前门准则相对于(X, Y)的条件：
1. M阻塞所有从X到Y的有向路径
2. X到M没有后门路径
3. 所有从M到Y的后门路径被X阻塞

前门调整公式：
```
P(y | do(x)) = ∑_m P(m | x) ∑_{x'} P(y | m, x') P(x')
```

---

## 5. 因果发现算法（PC/GES）

### 5.1 因果发现概述

因果发现是从观测数据中推断因果结构的方法。主要分为两类：
- **基于约束的方法**：使用条件独立性测试
- **基于评分的方法**：使用评分函数优化图结构

### 5.2 PC算法

PC算法是一种基于约束的因果发现算法，由Spirtes和Glymour提出。

**算法步骤**：

1. **骨架学习（Skeleton Learning）**：
   - 从完全无向图开始
   - 对于每对变量(X, Y)，测试条件独立性X ⟂ Y | S
   - 如果独立，移除边X-Y
   - S的大小从0递增到最大度数

2. **v-结构定向（V-structure Orientation）**：
   - 对于每个三元组X-Y-Z，如果Y不在分离集中
   - 定向为X → Y ← Z

3. **边定向传播（Edge Orientation Propagation）**：
   - 应用定向规则避免有向环
   - 尽可能多地定向边

**输出**：CPDAG（Completed Partially Directed Acyclic Graph）

**假设**：
- 因果充分性（无未观测混杂）
- 忠实性
- 正确的条件独立性测试

### 5.3 GES算法

GES（Greedy Equivalence Search）是一种基于评分的因果发现算法，由Chickering提出。

**算法步骤**：

1. **前向阶段（Forward Phase）**：
   - 从空图开始
   - 贪婪地添加能最大程度提高评分函数的边
   - 直到没有边能提高评分

2. **后向阶段（Backward Phase）**：
   - 贪婪地移除能最大程度提高评分函数的边
   - 直到没有边能提高评分

**评分函数**：
- BIC（Bayesian Information Criterion）
- BDeu（Bayesian Dirichlet equivalent uniform）

**输出**：CPDAG

**性质**：
- 在因果充分性和忠实性假设下，GES能一致地识别真实CPDAG
- 在局部最优处停止

### 5.4 算法比较

| 特性 | PC算法 | GES算法 |
|------|--------|---------|
| 类型 | 基于约束 | 基于评分 |
| 计算复杂度 | O(p^q)，q为最大度数 | 多项式时间（启发式） |
| 样本效率 | 需要大样本 | 中等样本即可 |
| 错误传播 | 早期错误会传播 | 评分函数更鲁棒 |
| 输出 | CPDAG | CPDAG |

### 5.5 其他因果发现方法

**LiNGAM（Linear Non-Gaussian Acyclic Model）**：
- 假设：线性关系 + 非高斯噪声
- 利用独立成分分析（ICA）识别因果方向

**FCI（Fast Causal Inference）**：
- 处理存在未观测混杂的情况
- 输出PAG（Partial Ancestral Graph）

**NOTEARS**：
- 使用连续优化替代组合搜索
- 通过约束DAG无环性进行优化

---

## 6. 代码实现架构

### 6.1 核心类设计

```python
# 主要组件：
1. CausalGraph - 因果图表示和操作
2. DoCalculus - do-calculus计算引擎
3. CounterfactualEngine - 反事实推理引擎
4. CausalDiscovery - 因果发现算法（PC/GES）
5. CausalEffectEstimator - 因果效应估计器
```

### 6.2 使用示例

```python
# 创建因果图
graph = CausalGraph()
graph.add_edge('X', 'Y')
graph.add_edge('Z', 'X')
graph.add_edge('Z', 'Y')

# 检查d-分离
is_separated = graph.is_d_separated('X', 'Y', {'Z'})

# 寻找后门调整集
adjustment_set = graph.find_backdoor_adjustment('X', 'Y')

# 估计因果效应
estimator = CausalEffectEstimator(graph, data)
ate = estimator.estimate_ate('X', 'Y', method='backdoor')

# 反事实推理
counterfactual = CounterfactualEngine(model)
result = counterfactual.compute('Y', {'X': 1}, evidence={'X': 0, 'Y': 0})

# 因果发现
pc = PCAlgorithm(data)
discovered_graph = pc.run()
```

---

## 参考文献

1. Pearl, J. (2009). *Causality: Models, Reasoning, and Inference*. Cambridge University Press.

2. Pearl, J., & Mackenzie, D. (2018). *The Book of Why: The New Science of Cause and Effect*. Basic Books.

3. Peters, J., Janzing, D., & Schölkopf, B. (2017). *Elements of Causal Inference: Foundations and Learning Algorithms*. MIT Press.

4. Imbens, G. W., & Rubin, D. B. (2015). *Causal Inference for Statistics, Social, and Biomedical Sciences: An Introduction*. Cambridge University Press.

5. Hernán, M. A., & Robins, J. M. (2025). *Causal Inference: What If*. Chapman & Hall/CRC.

6. Spirtes, P., Glymour, C., & Scheines, R. (2000). *Causation, Prediction, and Search*. MIT Press.

7. Chickering, D. M. (2002). Optimal Structure Identification with Greedy Search. *Journal of Machine Learning Research*, 3, 507-554.

8. Wang, L., Richardson, T. S., & Robins, J. M. (2024). Causal Inference: A Tale of Three Frameworks. *arXiv preprint*.

---

*文档生成时间：2026-02-27*
*版本：v1.0*
