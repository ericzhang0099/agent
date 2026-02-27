"""
因果推理示例脚本

本脚本展示如何使用causal_reasoner.py进行各种因果推理任务
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from causal_reasoner import (
    CausalGraph, CausalModel, CausalEffectEstimator,
    DoCalculusEngine, CounterfactualEngine,
    PCAlgorithm, GESAlgorithm,
    EdgeType, create_sample_data
)


def example_1_basic_graph_operations():
    """
    示例1: 基本的因果图操作
    
    展示如何构建因果图、查询图结构、测试D-分离
    """
    print("\n" + "=" * 70)
    print("示例1: 基本的因果图操作")
    print("=" * 70)
    
    # 创建因果图
    graph = CausalGraph()
    
    # 添加边：构建一个经典的混杂结构
    # Z -> X -> Y
    # Z -> Y
    graph.add_edge('Z', 'X')
    graph.add_edge('X', 'Y')
    graph.add_edge('Z', 'Y')
    
    print("\n构建的因果图:")
    print(graph)
    
    # 查询图结构
    print("\n图结构查询:")
    print(f"X的父节点: {graph.get_parents('X')}")
    print(f"X的子节点: {graph.get_children('X')}")
    print(f"X的祖先: {graph.get_ancestors('X')}")
    print(f"X的后代: {graph.get_descendants('X')}")
    
    # D-分离测试
    print("\nD-分离测试:")
    print(f"X和Y是否独立（无条件）? {graph.is_d_separated('X', 'Y')}")
    print(f"给定Z，X和Y是否独立? {graph.is_d_separated('X', 'Y', {'Z'})}")
    print(f"给定X，Z和Y是否独立? {graph.is_d_separated('Z', 'Y', {'X'})}")
    
    return graph


def example_2_backdoor_criterion():
    """
    示例2: 后门准则应用
    
    展示如何寻找后门调整集并进行后门调整
    """
    print("\n" + "=" * 70)
    print("示例2: 后门准则应用")
    print("=" * 70)
    
    # 创建更复杂的因果图
    graph = CausalGraph()
    
    # 构建一个包含多个混杂因素的结构
    # Z1 -> X -> Y
    # Z1 -> Y
    # Z2 -> X
    # Z2 -> Y
    # W -> Y (不是混杂因素)
    graph.add_edge('Z1', 'X')
    graph.add_edge('X', 'Y')
    graph.add_edge('Z1', 'Y')
    graph.add_edge('Z2', 'X')
    graph.add_edge('Z2', 'Y')
    graph.add_edge('W', 'Y')
    
    print("\n因果图结构:")
    print(graph)
    
    # 寻找后门调整集
    print("\n后门调整集分析:")
    backdoor_set = graph.find_backdoor_adjustment_set('X', 'Y')
    print(f"最小后门调整集: {backdoor_set}")
    
    # 解释为什么某些变量应该/不应该包含
    print("\n变量分析:")
    print("- Z1和Z2应该包含：它们是X和Y的共同原因（混杂因素）")
    print("- W不应该包含：它是Y的后代，不是混杂因素")
    
    return graph


def example_3_causal_effect_estimation():
    """
    示例3: 因果效应估计
    
    展示如何使用不同方法估计因果效应
    """
    print("\n" + "=" * 70)
    print("示例3: 因果效应估计")
    print("=" * 70)
    
    # 生成模拟数据
    np.random.seed(42)
    n = 5000
    
    # 真实的因果结构：
    # Z -> X -> Y
    # Z -> Y
    # 真实效应：X -> Y = 2.0
    
    Z = np.random.normal(0, 1, n)
    X = 0.5 * Z + np.random.normal(0, 0.5, n)
    Y = 2.0 * X + 0.3 * Z + np.random.normal(0, 0.5, n)
    
    data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
    
    print(f"\n生成数据: {n} 样本")
    print(f"真实因果效应 (X -> Y): 2.0")
    print("\n数据前5行:")
    print(data.head())
    
    # 构建因果图
    graph = CausalGraph()
    graph.add_edge('Z', 'X')
    graph.add_edge('X', 'Y')
    graph.add_edge('Z', 'Y')
    
    # 创建估计器
    estimator = CausalEffectEstimator(graph, data)
    
    # 方法1: 简单的相关性（有偏）
    print("\n估计结果比较:")
    naive_effect = data['Y'].corr(data['X']) * data['Y'].std() / data['X'].std()
    print(f"1. 简单回归系数（有偏）: {naive_effect:.4f}")
    
    # 方法2: 后门调整
    ate_backdoor = estimator.estimate_ate('X', 'Y', method='backdoor', 
                                          adjustment_set={'Z'})
    print(f"2. 后门调整估计: {ate_backdoor:.4f}")
    
    # 方法3: IPW
    # 二值化处理变量用于IPW
    data_binary = data.copy()
    data_binary['X'] = (data_binary['X'] > data_binary['X'].median()).astype(int)
    estimator_binary = CausalEffectEstimator(graph, data_binary)
    ate_ipw = estimator_binary.estimate_ate('X', 'Y', method='ipw',
                                            adjustment_set={'Z'})
    print(f"3. IPW估计（二值化X）: {ate_ipw:.4f}")
    
    return data, graph


def example_4_do_calculus():
    """
    示例4: Do-Calculus应用
    
    展示如何使用do-calculus识别因果效应
    """
    print("\n" + "=" * 70)
    print("示例4: Do-Calculus应用")
    print("=" * 70)
    
    # 场景1: 简单后门调整
    print("\n场景1: 简单混杂结构")
    graph1 = CausalGraph()
    graph1.add_edge('Z', 'X')
    graph1.add_edge('Z', 'Y')
    graph1.add_edge('X', 'Y')
    
    do_engine1 = DoCalculusEngine(graph1)
    identified1 = do_engine1.identify_causal_effect('X', 'Y')
    print(f"识别的因果效应: {identified1}")
    
    # 场景2: 无混杂
    print("\n场景2: 无混杂结构")
    graph2 = CausalGraph()
    graph2.add_edge('X', 'Y')
    
    do_engine2 = DoCalculusEngine(graph2)
    identified2 = do_engine2.identify_causal_effect('X', 'Y')
    print(f"识别的因果效应: {identified2}")
    
    # 场景3: 复杂结构
    print("\n场景3: 复杂结构（包含中介变量）")
    graph3 = CausalGraph()
    graph3.add_edge('Z', 'X')
    graph3.add_edge('X', 'M')
    graph3.add_edge('M', 'Y')
    graph3.add_edge('X', 'Y')
    graph3.add_edge('Z', 'Y')
    
    print("图结构:")
    print(graph3)
    
    do_engine3 = DoCalculusEngine(graph3)
    identified3 = do_engine3.identify_causal_effect('X', 'Y')
    print(f"识别的因果效应: {identified3}")


def example_5_counterfactuals():
    """
    示例5: 反事实推理
    
    展示如何计算反事实
    """
    print("\n" + "=" * 70)
    print("示例5: 反事实推理")
    print("=" * 70)
    
    # 创建结构因果模型
    model = CausalModel(variables=['X', 'Y'])
    
    # 定义结构方程: Y = 2*X + noise
    model.add_equation('X', lambda **kwargs: np.random.normal(0, 1))
    model.add_equation('Y', lambda X, noise, **kwargs: 2.0 * X + noise)
    model.noise_distributions['Y'] = lambda: np.random.normal(0, 0.5)
    
    print("\n结构因果模型:")
    print("X = ε_X, ε_X ~ N(0, 1)")
    print("Y = 2*X + ε_Y, ε_Y ~ N(0, 0.5)")
    
    # 创建反事实引擎
    engine = CounterfactualEngine(model)
    
    # 计算ATE
    print("\n反事实计算:")
    print("ATE = E[Y(1) - Y(0)] = 2.0")
    
    # 解释反事实
    print("\n反事实解释:")
    print("假设观测到 X=0.5, Y=1.2")
    print("问: 如果X=1.5，Y会是多少？")
    print("答: 保持噪声不变，Y = 2*1.5 + noise = 3.0 + noise")


def example_6_causal_discovery():
    """
    示例6: 因果发现
    
    展示如何使用PC和GES算法从数据中发现因果结构
    """
    print("\n" + "=" * 70)
    print("示例6: 因果发现")
    print("=" * 70)
    
    # 生成数据
    np.random.seed(42)
    n = 1000
    
    # 真实结构: Z -> X -> Y, Z -> Y
    Z = np.random.normal(0, 1, n)
    X = 0.5 * Z + np.random.normal(0, 0.5, n)
    Y = 0.8 * X + 0.3 * Z + np.random.normal(0, 0.5, n)
    
    data = pd.DataFrame({'Z': Z, 'X': X, 'Y': Y})
    
    print(f"\n生成数据: {n} 样本")
    print("真实因果结构: Z -> X -> Y, Z -> Y")
    
    # PC算法
    print("\n" + "-" * 40)
    print("PC算法结果:")
    print("-" * 40)
    pc = PCAlgorithm(data, alpha=0.05)
    graph_pc = pc.run()
    print(graph_pc)
    
    # GES算法
    print("\n" + "-" * 40)
    print("GES算法结果:")
    print("-" * 40)
    ges = GESAlgorithm(data)
    graph_ges = ges.run()
    print(graph_ges)
    
    # 比较
    print("\n" + "-" * 40)
    print("算法比较:")
    print("-" * 40)
    print("PC算法: 基于条件独立性测试，对样本量敏感")
    print("GES算法: 基于评分函数，计算效率较高")
    print("注意: 两种算法都输出CPDAG，可能包含无向边")


def example_7_mediation_analysis():
    """
    示例7: 中介分析
    
    展示如何分析直接效应和间接效应
    """
    print("\n" + "=" * 70)
    print("示例7: 中介分析")
    print("=" * 70)
    
    # 构建中介模型
    graph = CausalGraph()
    graph.add_edge('X', 'M')  # 处理 -> 中介
    graph.add_edge('M', 'Y')  # 中介 -> 结果
    graph.add_edge('X', 'Y')  # 处理 -> 结果（直接效应）
    
    print("\n中介模型结构:")
    print("X -> M -> Y")
    print("X -> Y")
    print("\n其中:")
    print("- X: 处理变量")
    print("- M: 中介变量")
    print("- Y: 结果变量")
    
    print("\n效应分解:")
    print("总效应 = 直接效应 + 间接效应（通过中介）")
    print("自然直接效应: E[Y(1, M(0)) - Y(0, M(0))]")
    print("自然间接效应: E[Y(1, M(1)) - Y(1, M(0))]")
    
    # 生成数据
    np.random.seed(42)
    n = 2000
    
    X = np.random.binomial(1, 0.5, n)
    M = 0.7 * X + np.random.normal(0, 0.5, n)
    Y = 0.5 * X + 0.4 * M + np.random.normal(0, 0.5, n)
    
    data = pd.DataFrame({'X': X, 'M': M, 'Y': Y})
    
    # 估计各种效应
    print("\n效应估计:")
    
    # 总效应
    estimator = CausalEffectEstimator(graph, data)
    total_effect = estimator.estimate_ate('X', 'Y', method='backdoor')
    print(f"总效应: {total_effect:.4f}")
    
    # X对M的效应
    effect_x_m = estimator.estimate_ate('X', 'M', method='backdoor')
    print(f"X对M的效应: {effect_x_m:.4f}")
    
    # M对Y的效应（控制X）
    graph_my = CausalGraph()
    graph_my.add_edge('X', 'M')
    graph_my.add_edge('X', 'Y')
    graph_my.add_edge('M', 'Y')
    estimator_my = CausalEffectEstimator(graph_my, data)
    effect_m_y = estimator_my.estimate_ate('M', 'Y', method='backdoor',
                                           adjustment_set={'X'})
    print(f"M对Y的效应（控制X）: {effect_m_y:.4f}")
    
    # 间接效应近似
    indirect_effect = effect_x_m * effect_m_y
    print(f"间接效应（近似）: {indirect_effect:.4f}")
    
    # 直接效应
    direct_effect = total_effect - indirect_effect
    print(f"直接效应（近似）: {direct_effect:.4f}")


def example_8_front_door_criterion():
    """
    示例8: 前门准则
    
    展示当存在未观测混杂时如何使用前门准则
    """
    print("\n" + "=" * 70)
    print("示例8: 前门准则")
    print("=" * 70)
    
    print("\n场景: 存在未观测混杂U")
    print("结构: X -> M -> Y")
    print("       U -> X, U -> Y (未观测)")
    print("\n后门准则失效（U未观测）")
    print("前门准则适用:")
    print("1. M阻塞所有从X到Y的有向路径")
    print("2. X到M没有后门路径")
    print("3. M到Y的后门路径被X阻塞")
    
    print("\n前门调整公式:")
    print("P(Y | do(X=x)) = ∑_m P(m | x) ∑_{x'} P(y | m, x') P(x')")
    
    # 生成数据（包含未观测混杂）
    np.random.seed(42)
    n = 5000
    
    U = np.random.normal(0, 1, n)  # 未观测
    X = 0.5 * U + np.random.normal(0, 0.5, n)
    M = 0.8 * X + np.random.normal(0, 0.5, n)
    Y = 0.6 * M + 0.4 * U + np.random.normal(0, 0.5, n)
    
    # 观测数据（不含U）
    data = pd.DataFrame({'X': X, 'M': M, 'Y': Y})
    
    print(f"\n生成数据: {n} 样本")
    print("真实因果效应 (X -> M -> Y): 0.8 * 0.6 = 0.48")
    
    # 构建因果图（不含U）
    graph = CausalGraph()
    graph.add_edge('X', 'M')
    graph.add_edge('M', 'Y')
    
    # 前门调整估计
    print("\n前门调整估计:")
    
    # 手动实现前门调整
    # P(Y | do(X)) = ∑_m P(m | X) ∑_{x'} P(Y | m, x') P(x')
    
    # 估计 P(M | X)
    m_given_x = {}
    for x_val in [0, 1]:
        subset = data[data['X'] > data['X'].median() if x_val else data['X'] <= data['X'].median()]
        m_given_x[x_val] = subset['M'].mean()
    
    # 简化的前门估计
    print("前门调整公式应用...")
    print("由于复杂性，这里展示概念性结果")
    print("实际估计需要更复杂的实现")


def run_all_examples():
    """运行所有示例"""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 20 + "因果推理示例脚本" + " " * 28 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)
    
    example_1_basic_graph_operations()
    example_2_backdoor_criterion()
    example_3_causal_effect_estimation()
    example_4_do_calculus()
    example_5_counterfactuals()
    example_6_causal_discovery()
    example_7_mediation_analysis()
    example_8_front_door_criterion()
    
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#" + " " * 25 + "所有示例完成!" + " " * 28 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)


if __name__ == "__main__":
    run_all_examples()
