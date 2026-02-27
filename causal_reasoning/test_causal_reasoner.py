"""
因果推理模块测试

测试causal_reasoner.py中的核心功能
"""

import unittest
import numpy as np
import pandas as pd
from causal_reasoner import (
    CausalGraph, CausalModel, CausalEffectEstimator,
    DoCalculusEngine, CounterfactualEngine,
    PCAlgorithm, GESAlgorithm,
    EdgeType, create_sample_data
)


class TestCausalGraph(unittest.TestCase):
    """测试CausalGraph类"""
    
    def setUp(self):
        self.graph = CausalGraph()
        # 构建经典混杂结构
        self.graph.add_edge('Z', 'X')
        self.graph.add_edge('X', 'Y')
        self.graph.add_edge('Z', 'Y')
    
    def test_add_nodes(self):
        """测试添加节点"""
        self.assertIn('X', self.graph.nodes)
        self.assertIn('Y', self.graph.nodes)
        self.assertIn('Z', self.graph.nodes)
    
    def test_add_edges(self):
        """测试添加边"""
        self.assertEqual(len(self.graph.edges), 3)
    
    def test_get_parents(self):
        """测试获取父节点"""
        self.assertEqual(self.graph.get_parents('X'), {'Z'})
        self.assertEqual(self.graph.get_parents('Y'), {'X', 'Z'})
    
    def test_get_children(self):
        """测试获取子节点"""
        self.assertEqual(self.graph.get_children('Z'), {'X', 'Y'})
        self.assertEqual(self.graph.get_children('X'), {'Y'})
    
    def test_get_ancestors(self):
        """测试获取祖先节点"""
        self.assertEqual(self.graph.get_ancestors('Y'), {'X', 'Z'})
        self.assertEqual(self.graph.get_ancestors('X'), {'Z'})
    
    def test_get_descendants(self):
        """测试获取后代节点"""
        self.assertEqual(self.graph.get_descendants('Z'), {'X', 'Y'})
        self.assertEqual(self.graph.get_descendants('X'), {'Y'})
    
    def test_d_separation(self):
        """测试D-分离"""
        # X和Y应该被Z d-分离
        self.assertTrue(self.graph.is_d_separated('X', 'Y', {'Z'}))
        # X和Y无条件时不应被d-分离
        self.assertFalse(self.graph.is_d_separated('X', 'Y'))
    
    def test_backdoor_adjustment_set(self):
        """测试后门调整集"""
        adjustment_set = self.graph.find_backdoor_adjustment_set('X', 'Y')
        self.assertIsNotNone(adjustment_set)
        self.assertIn('Z', adjustment_set)


class TestCausalModel(unittest.TestCase):
    """测试CausalModel类"""
    
    def setUp(self):
        self.model = CausalModel(variables=['X', 'Y'])
        self.model.add_equation('X', lambda **kwargs: kwargs.get('noise', 0))
        self.model.add_equation('Y', lambda X, noise, **kwargs: 2 * X + noise)
        self.model.noise_distributions['X'] = lambda: np.random.normal(0, 1)
        self.model.noise_distributions['Y'] = lambda: np.random.normal(0, 0.5)
    
    def test_intervention(self):
        """测试干预操作"""
        intervened_model = self.model.intervene({'X': 5.0})
        self.assertIn('X', intervened_model.equations)
        # 干预后的X应该总是返回5
        result = intervened_model.equations['X']()
        self.assertEqual(result, 5.0)
    
    def test_sampling(self):
        """测试采样"""
        data = self.model.sample(n_samples=100)
        self.assertEqual(len(data), 100)
        self.assertIn('X', data.columns)
        self.assertIn('Y', data.columns)


class TestCausalEffectEstimator(unittest.TestCase):
    """测试CausalEffectEstimator类"""
    
    def setUp(self):
        # 生成测试数据
        np.random.seed(42)
        n = 1000
        Z = np.random.normal(0, 1, n)
        X = 0.5 * Z + np.random.normal(0, 0.5, n)
        Y = 2.0 * X + 0.3 * Z + np.random.normal(0, 0.5, n)
        
        self.data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
        
        # 构建因果图
        self.graph = CausalGraph()
        self.graph.add_edge('Z', 'X')
        self.graph.add_edge('X', 'Y')
        self.graph.add_edge('Z', 'Y')
        
        self.estimator = CausalEffectEstimator(self.graph, self.data)
    
    def test_backdoor_adjustment(self):
        """测试后门调整"""
        ate = self.estimator.estimate_ate('X', 'Y', method='backdoor')
        # 真实效应约为2.0
        self.assertGreater(ate, 1.5)
        self.assertLess(ate, 2.5)
    
    def test_ipw(self):
        """测试IPW"""
        # 二值化X用于IPW
        data_binary = self.data.copy()
        data_binary['X'] = (data_binary['X'] > data_binary['X'].median()).astype(int)
        estimator_binary = CausalEffectEstimator(self.graph, data_binary)
        
        ate = estimator_binary.estimate_ate('X', 'Y', method='ipw')
        # 应该得到一个合理的估计值
        self.assertIsInstance(ate, float)


class TestDoCalculusEngine(unittest.TestCase):
    """测试DoCalculusEngine类"""
    
    def setUp(self):
        self.graph = CausalGraph()
        self.graph.add_edge('Z', 'X')
        self.graph.add_edge('X', 'Y')
        self.graph.add_edge('Z', 'Y')
        self.engine = DoCalculusEngine(self.graph)
    
    def test_identify_causal_effect(self):
        """测试因果效应识别"""
        identified = self.engine.identify_causal_effect('X', 'Y')
        self.assertIsNotNone(identified)
        # 应该包含后门调整公式
        self.assertIn('Z', identified)
    
    def test_apply_rule2_backdoor(self):
        """测试规则2（后门准则）"""
        # 在G_X̲中，Y应该与X独立给定Z
        result = self.engine.apply_rule2('Y', set(), 'X', {'Z'})
        self.assertTrue(result)


class TestCausalDiscovery(unittest.TestCase):
    """测试因果发现算法"""
    
    def setUp(self):
        # 生成测试数据
        np.random.seed(42)
        n = 1000
        Z = np.random.normal(0, 1, n)
        X = 0.5 * Z + np.random.normal(0, 0.5, n)
        Y = 0.8 * X + 0.3 * Z + np.random.normal(0, 0.5, n)
        
        self.data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
    
    def test_pc_algorithm(self):
        """测试PC算法"""
        pc = PCAlgorithm(self.data, alpha=0.05)
        graph = pc.run()
        
        # 应该发现所有三个变量
        self.assertIn('Z', graph.nodes)
        self.assertIn('X', graph.nodes)
        self.assertIn('Y', graph.nodes)
        
        # 应该发现一些边
        self.assertGreater(len(graph.edges), 0)
    
    def test_ges_algorithm(self):
        """测试GES算法"""
        ges = GESAlgorithm(self.data)
        graph = ges.run()
        
        # 应该发现所有三个变量
        self.assertIn('Z', graph.nodes)
        self.assertIn('X', graph.nodes)
        self.assertIn('Y', graph.nodes)
        
        # 应该发现一些边
        self.assertGreater(len(graph.edges), 0)


class TestIntegration(unittest.TestCase):
    """集成测试"""
    
    def test_full_pipeline(self):
        """测试完整流程"""
        # 1. 生成数据
        data = create_sample_data(n_samples=1000, seed=42)
        
        # 2. 因果发现
        pc = PCAlgorithm(data)
        discovered_graph = pc.run()
        
        # 3. 因果效应估计
        # 使用真实图结构进行估计
        true_graph = CausalGraph()
        true_graph.add_edge('Z', 'X')
        true_graph.add_edge('X', 'Y')
        true_graph.add_edge('Z', 'Y')
        
        estimator = CausalEffectEstimator(true_graph, data)
        ate = estimator.estimate_ate('X', 'Y', method='backdoor')
        
        # 验证结果
        self.assertIsInstance(ate, float)
        self.assertGreater(ate, 0)  # X对Y应该有正向效应
    
    def test_mediated_effect(self):
        """测试中介效应"""
        # 生成中介数据
        np.random.seed(42)
        n = 2000
        X = np.random.binomial(1, 0.5, n)
        M = 0.7 * X + np.random.normal(0, 0.5, n)
        Y = 0.5 * X + 0.4 * M + np.random.normal(0, 0.5, n)
        
        data = pd.DataFrame({'X': X, 'M': M, 'Y': Y})
        
        # 构建图
        graph = CausalGraph()
        graph.add_edge('X', 'M')
        graph.add_edge('M', 'Y')
        graph.add_edge('X', 'Y')
        
        # 估计效应
        estimator = CausalEffectEstimator(graph, data)
        total_effect = estimator.estimate_ate('X', 'Y', method='backdoor')
        
        # 总效应应该约为 0.5 + 0.7*0.4 = 0.78
        self.assertGreater(total_effect, 0.5)
        self.assertLess(total_effect, 1.0)


def run_tests():
    """运行所有测试"""
    unittest.main(argv=[''], verbosity=2, exit=False)


if __name__ == '__main__':
    run_tests()
