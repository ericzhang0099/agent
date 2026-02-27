"""
因果推理核心模块 - Causal Reasoner

本模块实现了因果推理的核心功能，包括：
1. 因果图模型（Causal Graphs）
2. Do-Calculus计算
3. 反事实推理（Counterfactuals）
4. 因果效应估计
5. 因果发现算法（PC/GES）

作者: Causal Reasoning Research Team
版本: 1.0.0
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Set, Tuple, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
import itertools
from abc import ABC, abstractmethod


# ============================================================================
# 基础数据结构
# ============================================================================

class EdgeType(Enum):
    """边类型枚举"""
    DIRECTED = "->"      # 有向边
    UNDIRECTED = "-"     # 无向边
    BIDIRECTED = "<->"   # 双向边（表示未观测混杂）


@dataclass
class Edge:
    """因果图中的边"""
    source: str
    target: str
    edge_type: EdgeType = EdgeType.DIRECTED
    
    def __hash__(self):
        return hash((self.source, self.target, self.edge_type))
    
    def __eq__(self, other):
        return (self.source, self.target, self.edge_type) == \
               (other.source, other.target, other.edge_type)
    
    def __repr__(self):
        if self.edge_type == EdgeType.DIRECTED:
            return f"{self.source} -> {self.target}"
        elif self.edge_type == EdgeType.UNDIRECTED:
            return f"{self.source} - {self.target}"
        else:
            return f"{self.source} <-> {self.target}"


@dataclass
class CausalModel:
    """结构因果模型（SCM）"""
    variables: List[str]
    equations: Dict[str, Callable] = field(default_factory=dict)
    noise_distributions: Dict[str, Callable] = field(default_factory=dict)
    
    def add_equation(self, var: str, equation: Callable, noise_dist: Callable = None):
        """添加结构方程"""
        self.equations[var] = equation
        if noise_dist is not None:
            self.noise_distributions[var] = noise_dist
    
    def intervene(self, interventions: Dict[str, float]) -> 'CausalModel':
        """
        执行干预 do(X=x)
        
        Args:
            interventions: {变量: 干预值}
        
        Returns:
            干预后的新模型
        """
        new_model = CausalModel(
            variables=self.variables.copy(),
            equations=self.equations.copy(),
            noise_distributions=self.noise_distributions.copy()
        )
        
        # 对被干预变量，替换为常数函数
        for var, value in interventions.items():
            def make_const_func(v):
                return lambda **kwargs: v
            new_model.equations[var] = make_const_func(value)
            
        return new_model
    
    def sample(self, n_samples: int = 1) -> pd.DataFrame:
        """从模型中采样数据"""
        data = {}
        
        # 拓扑排序确保父节点先计算
        ordered_vars = self._topological_sort()
        
        for _ in range(n_samples):
            sample = {}
            for var in ordered_vars:
                if var in self.noise_distributions:
                    noise = self.noise_distributions[var]()
                else:
                    noise = 0
                
                if var in self.equations:
                    sample[var] = self.equations[var](**sample, noise=noise)
                else:
                    sample[var] = noise
            
            for var, val in sample.items():
                if var not in data:
                    data[var] = []
                data[var].append(val)
        
        return pd.DataFrame(data)
    
    def _topological_sort(self) -> List[str]:
        """拓扑排序变量"""
        # 简化的拓扑排序实现
        # 实际实现需要解析方程依赖关系
        return self.variables


# ============================================================================
# 因果图类
# ============================================================================

class CausalGraph:
    """
    因果图模型（DAG）
    
    支持的功能：
    - 图的构建和修改
    - D-分离测试
    - 后门准则
    - 前门准则
    - Do-Calculus规则应用
    """
    
    def __init__(self):
        self.nodes: Set[str] = set()
        self.edges: Set[Edge] = set()
        self.parents: Dict[str, Set[str]] = defaultdict(set)
        self.children: Dict[str, Set[str]] = defaultdict(set)
        self.neighbors: Dict[str, Set[str]] = defaultdict(set)
    
    def add_node(self, node: str):
        """添加节点"""
        self.nodes.add(node)
    
    def add_edge(self, source: str, target: str, edge_type: EdgeType = EdgeType.DIRECTED):
        """添加边"""
        self.add_node(source)
        self.add_node(target)
        
        edge = Edge(source, target, edge_type)
        self.edges.add(edge)
        
        if edge_type == EdgeType.DIRECTED:
            self.parents[target].add(source)
            self.children[source].add(target)
        elif edge_type == EdgeType.UNDIRECTED:
            self.neighbors[source].add(target)
            self.neighbors[target].add(source)
    
    def remove_edge(self, source: str, target: str):
        """移除边"""
        edges_to_remove = [e for e in self.edges if e.source == source and e.target == target]
        for edge in edges_to_remove:
            self.edges.remove(edge)
            if edge.edge_type == EdgeType.DIRECTED:
                self.parents[target].discard(source)
                self.children[source].discard(target)
    
    def get_parents(self, node: str) -> Set[str]:
        """获取父节点"""
        return self.parents[node].copy()
    
    def get_children(self, node: str) -> Set[str]:
        """获取子节点"""
        return self.children[node].copy()
    
    def get_ancestors(self, node: str) -> Set[str]:
        """获取祖先节点"""
        ancestors = set()
        to_visit = [node]
        visited = set()
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for parent in self.get_parents(current):
                if parent != node:
                    ancestors.add(parent)
                    to_visit.append(parent)
        
        return ancestors
    
    def get_descendants(self, node: str) -> Set[str]:
        """获取后代节点"""
        descendants = set()
        to_visit = [node]
        visited = set()
        
        while to_visit:
            current = to_visit.pop(0)
            if current in visited:
                continue
            visited.add(current)
            
            for child in self.get_children(current):
                if child != node:
                    descendants.add(child)
                    to_visit.append(child)
        
        return descendants
    
    def get_neighbors(self, node: str) -> Set[str]:
        """获取相邻节点"""
        neighbors = set()
        neighbors.update(self.parents[node])
        neighbors.update(self.children[node])
        neighbors.update(self.neighbors[node])
        return neighbors
    
    def is_d_separated(self, x: str, y: str, conditioning_set: Set[str] = None) -> bool:
        """
        检查D-分离
        
        Args:
            x: 第一个变量
            y: 第二个变量
            conditioning_set: 条件变量集合
        
        Returns:
            如果x和y被条件集合d-分离则返回True
        """
        if conditioning_set is None:
            conditioning_set = set()
        
        # 使用路径遍历算法检查所有路径是否被阻塞
        visited = set()
        
        def is_path_blocked(path: List[str]) -> bool:
            """检查路径是否被阻塞"""
            if len(path) < 3:
                return False
            
            for i in range(1, len(path) - 1):
                prev_node = path[i - 1]
                curr_node = path[i]
                next_node = path[i + 1]
                
                # 检查是否是碰撞点
                is_collider = (curr_node in self.children.get(prev_node, set()) and 
                              curr_node in self.children.get(next_node, set()))
                
                if is_collider:
                    # 碰撞点：如果不在条件集中且后代也不在条件集中，则阻塞
                    if curr_node not in conditioning_set:
                        descendants = self.get_descendants(curr_node)
                        if not (descendants & conditioning_set):
                            return True
                else:
                    # 非碰撞点：如果在条件集中，则阻塞
                    if curr_node in conditioning_set:
                        return True
            
            return False
        
        # 使用BFS查找所有路径
        from collections import deque
        queue = deque([(x, [x])])
        
        while queue:
            current, path = queue.popleft()
            
            if current == y:
                if not is_path_blocked(path):
                    return False
                continue
            
            if (current, tuple(sorted(path))) in visited:
                continue
            visited.add((current, tuple(sorted(path))))
            
            for neighbor in self.get_neighbors(current):
                if neighbor not in path:
                    queue.append((neighbor, path + [neighbor]))
        
        return True
    
    def find_backdoor_paths(self, treatment: str, outcome: str) -> List[List[str]]:
        """
        查找所有后门路径
        
        后门路径：以箭头指向treatment开始的路径
        """
        paths = []
        
        def find_paths(current: str, target: str, path: List[str], started: bool):
            if len(path) > len(self.nodes) + 1:  # 防止循环
                return
            
            if started and current == target:
                paths.append(path.copy())
                return
            
            for parent in self.get_parents(current):
                if parent not in path:
                    path.append(parent)
                    find_paths(parent, target, path, True)
                    path.pop()
            
            for child in self.get_children(current):
                if child not in path:
                    path.append(child)
                    find_paths(child, target, path, started)
                    path.pop()
        
        # 从treatment开始，找到outcome，路径必须以进入treatment的边开始
        for parent in self.get_parents(treatment):
            find_paths(parent, outcome, [treatment, parent], True)
        
        return paths
    
    def find_backdoor_adjustment_set(self, treatment: str, outcome: str) -> Optional[Set[str]]:
        """
        寻找满足后门准则的调整集合
        
        Returns:
            满足条件的调整集合，如果不存在则返回None
        """
        # 获取所有可能的调整变量（不是treatment的后代）
        descendants = self.get_descendants(treatment)
        candidates = self.nodes - descendants - {treatment, outcome}
        
        # 检查所有可能的子集
        for r in range(len(candidates) + 1):
            for adjustment_set in itertools.combinations(candidates, r):
                adjustment_set = set(adjustment_set)
                
                # 检查是否阻塞所有后门路径
                if self._blocks_all_backdoor_paths(treatment, outcome, adjustment_set):
                    return adjustment_set
        
        return None
    
    def _blocks_all_backdoor_paths(self, treatment: str, outcome: str, 
                                   adjustment_set: Set[str]) -> bool:
        """检查调整集合是否阻塞所有后门路径"""
        # 简化实现：使用d-分离检查
        # 在移除treatment出边的图中检查
        temp_graph = self._get_modified_graph(remove_outgoing={treatment})
        return temp_graph.is_d_separated(treatment, outcome, adjustment_set)
    
    def _get_modified_graph(self, remove_incoming: Set[str] = None, 
                           remove_outgoing: Set[str] = None) -> 'CausalGraph':
        """获取修改后的图（用于do-calculus）"""
        if remove_incoming is None:
            remove_incoming = set()
        if remove_outgoing is None:
            remove_outgoing = set()
        
        new_graph = CausalGraph()
        new_graph.nodes = self.nodes.copy()
        
        for edge in self.edges:
            if edge.edge_type == EdgeType.DIRECTED:
                if edge.target in remove_incoming:
                    continue
                if edge.source in remove_outgoing:
                    continue
            new_graph.edges.add(edge)
            if edge.edge_type == EdgeType.DIRECTED:
                new_graph.parents[edge.target].add(edge.source)
                new_graph.children[edge.source].add(edge.target)
        
        return new_graph
    
    def to_dag(self) -> 'CausalGraph':
        """将CPDAG转换为DAG（如果存在无向边，随机定向）"""
        # 简化的实现
        return self
    
    def __repr__(self):
        edges_str = "\n".join([f"  {e}" for e in self.edges])
        return f"CausalGraph(nodes={self.nodes}, edges=\n{edges_str}\n)"


# ============================================================================
# Do-Calculus引擎
# ============================================================================

class DoCalculusEngine:
    """
    Do-Calculus计算引擎
    
    实现Pearl的三条do-calculus规则
    """
    
    def __init__(self, graph: CausalGraph):
        self.graph = graph
    
    def apply_rule1(self, y: str, x: Set[str], z: str, w: Set[str] = None) -> bool:
        """
        规则1：观测的插入/删除
        
        P(y | do(x), z, w) = P(y | do(x), w)
        如果 Y ⟂ Z | X, W 在 G_X̄ 中
        """
        if w is None:
            w = set()
        
        # 创建移除X入边的图
        modified_graph = self.graph._get_modified_graph(remove_incoming=x)
        
        # 检查条件独立性
        return modified_graph.is_d_separated(y, z, x.union(w))
    
    def apply_rule2(self, y: str, x: Set[str], z: str, w: Set[str] = None) -> bool:
        """
        规则2：行动/观测交换
        
        P(y | do(x), do(z), w) = P(y | do(x), z, w)
        如果 Y ⟂ Z | X, W 在 G_X̄,Z̲ 中
        """
        if w is None:
            w = set()
        
        # 创建移除X入边和Z出边的图
        modified_graph = self.graph._get_modified_graph(
            remove_incoming=x, 
            remove_outgoing={z}
        )
        
        # 检查条件独立性
        return modified_graph.is_d_separated(y, z, x.union(w))
    
    def apply_rule3(self, y: str, x: Set[str], z: str, w: Set[str] = None) -> bool:
        """
        规则3：行动的插入/删除
        
        P(y | do(x), do(z), w) = P(y | do(x), w)
        如果 Y ⟂ Z | X, W 在 G_X̄,Z̄(W) 中
        
        其中Z(W)是不是W祖先的Z节点
        """
        if w is None:
            w = set()
        
        # 找到Z中不是W祖先的节点
        z_set = {z}
        z_w = set()
        for node in z_set:
            ancestors = self.graph.get_ancestors(node)
            if not (ancestors & w):
                z_w.add(node)
        
        # 创建移除X入边和Z(W)入边的图
        modified_graph = self.graph._get_modified_graph(
            remove_incoming=x.union(z_w)
        )
        
        # 检查条件独立性
        return modified_graph.is_d_separated(y, z, x.union(w))
    
    def identify_causal_effect(self, treatment: str, outcome: str) -> Optional[str]:
        """
        识别因果效应 P(y | do(x))
        
        尝试使用do-calculus识别因果效应
        
        Returns:
            可识别的表达式字符串，如果无法识别则返回None
        """
        # 检查是否可以直接应用后门准则
        backdoor_set = self.graph.find_backdoor_adjustment_set(treatment, outcome)
        if backdoor_set is not None:
            if len(backdoor_set) == 0:
                return f"P({outcome} | {treatment})"
            else:
                z_str = ", ".join(sorted(backdoor_set))
                return f"∑_{{{z_str}}} P({outcome} | {treatment}, {z_str}) P({z_str})"
        
        # 尝试前门准则
        frontdoor_set = self._find_frontdoor_adjustment_set(treatment, outcome)
        if frontdoor_set is not None:
            m_str = ", ".join(sorted(frontdoor_set))
            return f"∑_{{{m_str}}} P({m_str} | {treatment}) ∑_{{{treatment}'}} P({outcome} | {m_str}, {treatment}') P({treatment}')"
        
        return None
    
    def _find_frontdoor_adjustment_set(self, treatment: str, outcome: str) -> Optional[Set[str]]:
        """
        寻找满足前门准则的调整集合
        """
        candidates = self.graph.nodes - {treatment, outcome}
        
        for r in range(1, len(candidates) + 1):
            for subset in itertools.combinations(candidates, r):
                subset = set(subset)
                if self._satisfies_frontdoor_criterion(treatment, outcome, subset):
                    return subset
        
        return None
    
    def _satisfies_frontdoor_criterion(self, treatment: str, outcome: str, 
                                       mediator: Set[str]) -> bool:
        """检查是否满足前门准则"""
        # 1. 中介变量阻塞所有从treatment到outcome的有向路径
        # 2. treatment到mediator没有后门路径
        # 3. mediator到outcome的所有后门路径被treatment阻塞
        
        # 简化实现
        return False  # 需要更复杂的实现


# ============================================================================
# 反事实推理引擎
# ============================================================================

class CounterfactualEngine:
    """
    反事实推理引擎
    
    基于结构因果模型计算反事实
    """
    
    def __init__(self, model: CausalModel):
        self.model = model
        self.graph = self._build_graph_from_model()
    
    def _build_graph_from_model(self) -> CausalGraph:
        """从因果模型构建图"""
        graph = CausalGraph()
        for var in self.model.variables:
            graph.add_node(var)
        return graph
    
    def compute(self, target: str, intervention: Dict[str, float], 
                evidence: Dict[str, float]) -> float:
        """
        计算反事实：P(Y_x = y | X = x', Y = y')
        
        Args:
            target: 目标变量
            intervention: 干预 {变量: 值}
            evidence: 观测证据 {变量: 值}
        
        Returns:
            反事实结果
        """
        # 步骤1：外生变量估计（abduction）
        # 使用观测证据估计外生噪声
        noise_estimates = self._estimate_noise(evidence)
        
        # 步骤2：干预（action）
        intervened_model = self.model.intervene(intervention)
        
        # 步骤3：预测（prediction）
        # 使用估计的噪声和干预模型计算结果
        result = self._predict_with_noise(intervened_model, noise_estimates, target)
        
        return result
    
    def _estimate_noise(self, evidence: Dict[str, float]) -> Dict[str, float]:
        """估计外生噪声值"""
        noise_estimates = {}
        
        # 简化的噪声估计
        # 实际实现需要反解结构方程
        for var, value in evidence.items():
            if var in self.model.equations:
                # 这里简化处理，实际应该根据方程反解
                noise_estimates[var] = 0
            else:
                noise_estimates[var] = value
        
        return noise_estimates
    
    def _predict_with_noise(self, model: CausalModel, 
                           noise: Dict[str, float], 
                           target: str) -> float:
        """使用估计的噪声进行预测"""
        # 简化的预测实现
        # 实际实现需要按拓扑顺序计算
        return 0.0
    
    def compute_ate(self, treatment: str, outcome: str) -> float:
        """
        计算平均处理效应（Average Treatment Effect）
        
        ATE = E[Y(1) - Y(0)]
        """
        # 干预为1
        result_1 = self.compute(outcome, {treatment: 1}, {})
        # 干预为0
        result_0 = self.compute(outcome, {treatment: 0}, {})
        
        return result_1 - result_0


# ============================================================================
# 因果效应估计器
# ============================================================================

class CausalEffectEstimator:
    """
    因果效应估计器
    
    使用各种方法从数据中估计因果效应
    """
    
    def __init__(self, graph: CausalGraph, data: pd.DataFrame):
        self.graph = graph
        self.data = data
        self.do_calculus = DoCalculusEngine(graph)
    
    def estimate_ate(self, treatment: str, outcome: str, 
                     method: str = 'backdoor',
                     adjustment_set: Set[str] = None) -> float:
        """
        估计平均处理效应（ATE）
        
        Args:
            treatment: 处理变量
            outcome: 结果变量
            method: 估计方法 ('backdoor', 'frontdoor', 'ipw', 'matching')
            adjustment_set: 调整变量集合（如果为None则自动查找）
        
        Returns:
            ATE估计值
        """
        if method == 'backdoor':
            return self._backdoor_adjustment(treatment, outcome, adjustment_set)
        elif method == 'ipw':
            return self._inverse_probability_weighting(treatment, outcome, adjustment_set)
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _backdoor_adjustment(self, treatment: str, outcome: str,
                            adjustment_set: Set[str] = None) -> float:
        """
        后门调整估计
        
        E[Y | do(X=1)] - E[Y | do(X=0)] = 
        ∑_z [E[Y | X=1, Z=z] - E[Y | X=0, Z=z]] P(Z=z)
        """
        if adjustment_set is None:
            adjustment_set = self.graph.find_backdoor_adjustment_set(treatment, outcome)
            if adjustment_set is None:
                raise ValueError("No valid backdoor adjustment set found")
        
        # 计算条件期望
        ate = 0.0
        
        if len(adjustment_set) == 0:
            # 无调整变量
            treated = self.data[self.data[treatment] == 1][outcome].mean()
            control = self.data[self.data[treatment] == 0][outcome].mean()
            ate = treated - control
        else:
            # 有调整变量
            adjustment_vars = list(adjustment_set)
            
            # 获取所有调整变量的组合
            if len(adjustment_vars) == 1:
                unique_values = self.data[adjustment_vars[0]].unique()
                for z in unique_values:
                    subset = self.data[self.data[adjustment_vars[0]] == z]
                    p_z = len(subset) / len(self.data)
                    
                    treated_subset = subset[subset[treatment] == 1]
                    control_subset = subset[subset[treatment] == 0]
                    
                    if len(treated_subset) > 0 and len(control_subset) > 0:
                        effect = treated_subset[outcome].mean() - control_subset[outcome].mean()
                        ate += effect * p_z
            else:
                # 多变量调整（简化实现）
                grouped = self.data.groupby(adjustment_vars)
                for z_values, subset in grouped:
                    p_z = len(subset) / len(self.data)
                    
                    treated_subset = subset[subset[treatment] == 1]
                    control_subset = subset[subset[treatment] == 0]
                    
                    if len(treated_subset) > 0 and len(control_subset) > 0:
                        effect = treated_subset[outcome].mean() - control_subset[outcome].mean()
                        ate += effect * p_z
        
        return ate
    
    def _inverse_probability_weighting(self, treatment: str, outcome: str,
                                       adjustment_set: Set[str] = None) -> float:
        """
        逆概率加权（IPW）估计
        
        E[Y | do(X=x)] = E[Y * I(X=x) / P(X=x | Z)]
        """
        from sklearn.linear_model import LogisticRegression
        
        if adjustment_set is None:
            adjustment_set = self.graph.find_backdoor_adjustment_set(treatment, outcome)
        
        adjustment_vars = list(adjustment_set) if adjustment_set else []
        
        # 估计倾向得分
        if len(adjustment_vars) > 0:
            X = self.data[adjustment_vars]
            y = self.data[treatment]
            
            model = LogisticRegression()
            model.fit(X, y)
            propensity_scores = model.predict_proba(X)[:, 1]
        else:
            propensity_scores = np.full(len(self.data), 
                                       self.data[treatment].mean())
        
        # 计算IPW估计
        treated = self.data[treatment] == 1
        control = self.data[treatment] == 0
        
        # 避免除零
        ps = np.clip(propensity_scores, 0.01, 0.99)
        
        y1_weighted = self.data.loc[treated, outcome] / ps[treated]
        y0_weighted = self.data.loc[control, outcome] / (1 - ps[control])
        
        ate = y1_weighted.sum() / (1 / ps[treated]).sum() - \
              y0_weighted.sum() / (1 / (1 - ps[control])).sum()
        
        return ate


# ============================================================================
# 因果发现算法
# ============================================================================

class CausalDiscoveryAlgorithm(ABC):
    """因果发现算法基类"""
    
    def __init__(self, data: pd.DataFrame):
        self.data = data
        self.variables = list(data.columns)
    
    @abstractmethod
    def run(self) -> CausalGraph:
        """运行因果发现算法"""
        pass
    
    def _conditional_independence_test(self, x: str, y: str, 
                                       conditioning_set: List[str] = None,
                                       alpha: float = 0.05) -> bool:
        """
        条件独立性测试
        
        Returns:
            如果独立则返回True
        """
        if conditioning_set is None or len(conditioning_set) == 0:
            # 无条件独立性测试（使用相关系数）
            corr = self.data[x].corr(self.data[y])
            n = len(self.data)
            # 简单的t检验
            t_stat = corr * np.sqrt((n - 2) / (1 - corr**2))
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2))
            return p_value > alpha
        else:
            # 偏相关检验（简化实现）
            from sklearn.linear_model import LinearRegression
            
            # 回归X和Y对条件变量
            if len(conditioning_set) == 1:
                z = conditioning_set[0]
                residuals_x = self.data[x] - self.data[x].groupby(self.data[z]).transform('mean')
                residuals_y = self.data[y] - self.data[y].groupby(self.data[z]).transform('mean')
            else:
                X_cond = self.data[conditioning_set]
                reg_x = LinearRegression().fit(X_cond, self.data[x])
                reg_y = LinearRegression().fit(X_cond, self.data[y])
                residuals_x = self.data[x] - reg_x.predict(X_cond)
                residuals_y = self.data[y] - reg_y.predict(X_cond)
            
            corr = np.corrcoef(residuals_x, residuals_y)[0, 1]
            n = len(self.data)
            t_stat = corr * np.sqrt((n - 2 - len(conditioning_set)) / (1 - corr**2))
            from scipy import stats
            p_value = 2 * (1 - stats.t.cdf(abs(t_stat), n - 2 - len(conditioning_set)))
            return p_value > alpha


class PCAlgorithm(CausalDiscoveryAlgorithm):
    """
    PC算法（Peter-Clark算法）
    
    基于约束的因果发现算法
    """
    
    def __init__(self, data: pd.DataFrame, alpha: float = 0.05):
        super().__init__(data)
        self.alpha = alpha
        self.separation_sets = {}
    
    def run(self) -> CausalGraph:
        """
        运行PC算法
        
        Returns:
            发现的因果图（CPDAG）
        """
        # 步骤1：学习骨架
        graph, separation_sets = self._learn_skeleton()
        self.separation_sets = separation_sets
        
        # 步骤2：定向v-结构
        self._orient_v_structures(graph)
        
        # 步骤3：传播定向
        self._propagate_orientations(graph)
        
        return graph
    
    def _learn_skeleton(self) -> Tuple[CausalGraph, Dict]:
        """学习图的骨架"""
        graph = CausalGraph()
        for var in self.variables:
            graph.add_node(var)
        
        # 初始化完全无向图
        for i, var1 in enumerate(self.variables):
            for var2 in self.variables[i+1:]:
                graph.add_edge(var1, var2, EdgeType.UNDIRECTED)
        
        separation_sets = {}
        depth = 0
        
        while True:
            edges_to_remove = []
            
            for edge in list(graph.edges):
                if edge.edge_type != EdgeType.UNDIRECTED:
                    continue
                
                x, y = edge.source, edge.target
                neighbors = list(graph.neighbors[x] | graph.neighbors[y] - {x, y})
                
                if len(neighbors) < depth:
                    continue
                
                # 尝试所有大小为depth的条件集合
                for conditioning_set in itertools.combinations(neighbors, depth):
                    if self._conditional_independence_test(x, y, list(conditioning_set), self.alpha):
                        edges_to_remove.append((x, y))
                        separation_sets[(x, y)] = set(conditioning_set)
                        separation_sets[(y, x)] = set(conditioning_set)
                        break
            
            if not edges_to_remove:
                break
            
            for x, y in edges_to_remove:
                graph.remove_edge(x, y)
                graph.remove_edge(y, x)
            
            depth += 1
        
        return graph, separation_sets
    
    def _orient_v_structures(self, graph: CausalGraph):
        """定向v-结构"""
        # 寻找所有三元组 X - Y - Z
        for y in graph.nodes:
            neighbors = list(graph.neighbors[y])
            for i, x in enumerate(neighbors):
                for z in neighbors[i+1:]:
                    # 检查是否是v-结构候选
                    if (x, z) not in [(e.source, e.target) for e in graph.edges] and \
                       (z, x) not in [(e.source, e.target) for e in graph.edges]:
                        # 检查Y是否在X和Z的分离集中
                        sep_set = self.separation_sets.get((x, z), set())
                        if y not in sep_set:
                            # 定向为 X -> Y <- Z
                            graph.remove_edge(x, y)
                            graph.remove_edge(z, y)
                            graph.add_edge(x, y, EdgeType.DIRECTED)
                            graph.add_edge(z, y, EdgeType.DIRECTED)
    
    def _propagate_orientations(self, graph: CausalGraph):
        """传播边的定向"""
        changed = True
        while changed:
            changed = False
            
            # 规则1：如果 X -> Y - Z，则定向为 X -> Y -> Z
            for edge in list(graph.edges):
                if edge.edge_type == EdgeType.UNDIRECTED:
                    y, z = edge.source, edge.target
                    for x in graph.parents[y]:
                        if graph.neighbors[y] and z in graph.neighbors[y]:
                            if (x, z) not in [(e.source, e.target) for e in graph.edges]:
                                graph.remove_edge(y, z)
                                graph.add_edge(y, z, EdgeType.DIRECTED)
                                changed = True
                                break
            
            # 规则2：避免有向环
            for edge in list(graph.edges):
                if edge.edge_type == EdgeType.UNDIRECTED:
                    x, y = edge.source, edge.target
                    if y in graph.get_ancestors(x):
                        graph.remove_edge(x, y)
                        graph.add_edge(y, x, EdgeType.DIRECTED)
                        changed = True


class GESAlgorithm(CausalDiscoveryAlgorithm):
    """
    GES算法（Greedy Equivalence Search）
    
    基于评分的因果发现算法
    """
    
    def __init__(self, data: pd.DataFrame, score_type: str = 'bic'):
        super().__init__(data)
        self.score_type = score_type
        self.n_samples = len(data)
    
    def run(self) -> CausalGraph:
        """
        运行GES算法
        
        Returns:
            发现的因果图（CPDAG）
        """
        # 前向阶段：贪婪添加边
        graph = self._forward_phase()
        
        # 后向阶段：贪婪删除边
        graph = self._backward_phase(graph)
        
        return graph
    
    def _forward_phase(self) -> CausalGraph:
        """前向阶段：贪婪添加边"""
        graph = CausalGraph()
        for var in self.variables:
            graph.add_node(var)
        
        current_score = self._score_graph(graph)
        improved = True
        
        while improved:
            improved = False
            best_graph = graph
            best_score = current_score
            
            # 尝试添加所有可能的边
            for i, x in enumerate(self.variables):
                for y in self.variables[i+1:]:
                    # 尝试 X -> Y
                    test_graph = self._copy_graph(graph)
                    test_graph.add_edge(x, y, EdgeType.DIRECTED)
                    if self._is_valid_dag(test_graph):
                        score = self._score_graph(test_graph)
                        if score > best_score:
                            best_graph = test_graph
                            best_score = score
                            improved = True
                    
                    # 尝试 Y -> X
                    test_graph = self._copy_graph(graph)
                    test_graph.add_edge(y, x, EdgeType.DIRECTED)
                    if self._is_valid_dag(test_graph):
                        score = self._score_graph(test_graph)
                        if score > best_score:
                            best_graph = test_graph
                            best_score = score
                            improved = True
            
            if improved:
                graph = best_graph
                current_score = best_score
        
        return graph
    
    def _backward_phase(self, graph: CausalGraph) -> CausalGraph:
        """后向阶段：贪婪删除边"""
        current_score = self._score_graph(graph)
        improved = True
        
        while improved:
            improved = False
            best_graph = graph
            best_score = current_score
            
            # 尝试删除所有现有的边
            for edge in list(graph.edges):
                test_graph = self._copy_graph(graph)
                test_graph.remove_edge(edge.source, edge.target)
                
                score = self._score_graph(test_graph)
                if score > best_score:
                    best_graph = test_graph
                    best_score = score
                    improved = True
            
            if improved:
                graph = best_graph
                current_score = best_score
        
        return graph
    
    def _score_graph(self, graph: CausalGraph) -> float:
        """
        计算图的评分
        
        使用BIC评分
        """
        score = 0.0
        
        for node in graph.nodes:
            parents = list(graph.parents[node])
            score += self._local_score(node, parents)
        
        return score
    
    def _local_score(self, node: str, parents: List[str]) -> float:
        """
        计算局部BIC评分
        
        BIC = log P(D | G) - (d/2) * log n
        """
        if len(parents) == 0:
            # 无父节点：使用边际似然
            variance = self.data[node].var()
            if variance == 0:
                variance = 1e-10
            log_likelihood = -0.5 * self.n_samples * np.log(2 * np.pi * variance) - \
                            0.5 * self.n_samples
        else:
            # 有父节点：使用条件似然（线性回归）
            from sklearn.linear_model import LinearRegression
            
            X = self.data[parents].values
            y = self.data[node].values
            
            model = LinearRegression()
            model.fit(X, y)
            
            y_pred = model.predict(X)
            residuals = y - y_pred
            rss = np.sum(residuals**2)
            
            if rss == 0:
                rss = 1e-10
            
            n = self.n_samples
            k = len(parents) + 1  # 参数数量（系数+截距）
            
            log_likelihood = -0.5 * n * np.log(2 * np.pi * rss / n) - 0.5 * n
            
            # BIC惩罚项
            bic_penalty = 0.5 * k * np.log(n)
            return log_likelihood - bic_penalty
        
        # BIC惩罚项（单个参数）
        bic_penalty = 0.5 * np.log(self.n_samples)
        return log_likelihood - bic_penalty
    
    def _is_valid_dag(self, graph: CausalGraph) -> bool:
        """检查图是否为有效的DAG（无有向环）"""
        visited = set()
        rec_stack = set()
        
        def has_cycle(node: str) -> bool:
            visited.add(node)
            rec_stack.add(node)
            
            for child in graph.children[node]:
                if child not in visited:
                    if has_cycle(child):
                        return True
                elif child in rec_stack:
                    return True
            
            rec_stack.remove(node)
            return False
        
        for node in graph.nodes:
            if node not in visited:
                if has_cycle(node):
                    return False
        
        return True
    
    def _copy_graph(self, graph: CausalGraph) -> CausalGraph:
        """复制图"""
        new_graph = CausalGraph()
        new_graph.nodes = graph.nodes.copy()
        new_graph.edges = graph.edges.copy()
        new_graph.parents = defaultdict(set, {k: v.copy() for k, v in graph.parents.items()})
        new_graph.children = defaultdict(set, {k: v.copy() for k, v in graph.children.items()})
        new_graph.neighbors = defaultdict(set, {k: v.copy() for k, v in graph.neighbors.items()})
        return new_graph


# ============================================================================
# 工具函数
# ============================================================================

def create_sample_data(n_samples: int = 1000, seed: int = 42) -> pd.DataFrame:
    """
    创建示例数据
    
    生成一个简单的因果结构：
    Z -> X -> Y
    Z -> Y
    """
    np.random.seed(seed)
    
    # 生成外生变量
    z = np.random.normal(0, 1, n_samples)
    u_x = np.random.normal(0, 0.5, n_samples)
    u_y = np.random.normal(0, 0.5, n_samples)
    
    # 生成内生变量
    x = 0.5 * z + u_x
    y = 0.3 * x + 0.4 * z + u_y
    
    data = pd.DataFrame({
        'Z': z,
        'X': x,
        'Y': y
    })
    
    return data


def demo():
    """
    因果推理演示
    """
    print("=" * 60)
    print("因果推理演示")
    print("=" * 60)
    
    # 1. 创建示例数据
    print("\n1. 创建示例数据")
    data = create_sample_data(n_samples=1000)
    print(data.head())
    print(f"\n数据形状: {data.shape}")
    
    # 2. 构建因果图
    print("\n2. 构建因果图")
    graph = CausalGraph()
    graph.add_edge('Z', 'X')
    graph.add_edge('X', 'Y')
    graph.add_edge('Z', 'Y')
    print(graph)
    
    # 3. 检查D-分离
    print("\n3. D-分离测试")
    print(f"X和Y是否被Z d-分离? {graph.is_d_separated('X', 'Y', {'Z'})}")
    print(f"Z和Y是否被X d-分离? {graph.is_d_separated('Z', 'Y', {'X'})}")
    
    # 4. 寻找后门调整集
    print("\n4. 后门调整集")
    backdoor_set = graph.find_backdoor_adjustment_set('X', 'Y')
    print(f"X -> Y 的后门调整集: {backdoor_set}")
    
    # 5. 因果效应估计
    print("\n5. 因果效应估计")
    estimator = CausalEffectEstimator(graph, data)
    
    # 后门调整
    ate_backdoor = estimator.estimate_ate('X', 'Y', method='backdoor')
    print(f"后门调整估计的ATE: {ate_backdoor:.4f}")
    
    # IPW
    try:
        ate_ipw = estimator.estimate_ate('X', 'Y', method='ipw')
        print(f"IPW估计的ATE: {ate_ipw:.4f}")
    except Exception as e:
        print(f"IPW估计的ATE: 跳过 ({str(e)[:50]}...)")
    
    # 6. Do-Calculus
    print("\n6. Do-Calculus识别")
    do_engine = DoCalculusEngine(graph)
    identified = do_engine.identify_causal_effect('X', 'Y')
    print(f"识别的因果效应表达式: {identified}")
    
    # 7. 因果发现
    print("\n7. 因果发现 - PC算法 (简化版)")
    print("   使用简化数据以加速计算")
    data_small = data.iloc[:200]  # 使用更少样本
    pc = PCAlgorithm(data_small)
    discovered_graph = pc.run()
    print("发现的图结构:")
    print(discovered_graph)
    
    print("\n8. 因果发现 - GES算法 (简化版)")
    ges = GESAlgorithm(data_small)
    discovered_graph_ges = ges.run()
    print("发现的图结构:")
    print(discovered_graph_ges)
    
    print("\n" + "=" * 60)
    print("演示完成!")
    print("=" * 60)


if __name__ == "__main__":
    demo()
