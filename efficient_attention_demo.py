"""
Efficient Attention Mechanisms - Standalone Demo
================================================
无需PyTorch的纯Python演示版本，展示算法原理

Author: AI Assistant
Date: 2026-02-27
"""

import math
import random
from typing import List, Optional, Tuple


# =============================================================================
# 1. 稀疏注意力演示
# =============================================================================

def create_sliding_window_mask(seq_len: int, window_size: int) -> List[List[float]]:
    """
    创建滑动窗口注意力掩码
    
    每个token只关注其左右window_size范围内的token
    """
    mask = [[float('-inf')] * seq_len for _ in range(seq_len)]
    
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) <= window_size:
                mask[i][j] = 0.0  # 允许关注
    
    return mask


def create_global_local_mask(
    seq_len: int, 
    window_size: int, 
    global_indices: List[int]
) -> List[List[float]]:
    """
    创建全局-局部注意力掩码 (Longformer风格)
    
    - 全局token可以关注所有位置
    - 所有token可以关注全局token
    - 非全局token使用滑动窗口
    """
    mask = [[float('-inf')] * seq_len for _ in range(seq_len)]
    
    # 全局token可以关注所有位置
    for idx in global_indices:
        if idx < seq_len:
            for j in range(seq_len):
                mask[idx][j] = 0.0
    
    # 所有token可以关注全局token
    for idx in global_indices:
        if idx < seq_len:
            for i in range(seq_len):
                mask[i][idx] = 0.0
    
    # 滑动窗口
    for i in range(seq_len):
        if i not in global_indices:
            for j in range(seq_len):
                if abs(i - j) <= window_size:
                    mask[i][j] = 0.0
    
    return mask


def visualize_mask(mask: List[List[float]], title: str = "Attention Mask"):
    """可视化注意力掩码"""
    print(f"\n{'='*50}")
    print(f"{title}")
    print(f"{'='*50}")
    
    seq_len = len(mask)
    
    # 打印表头
    print("   ", end="")
    for j in range(min(10, seq_len)):
        print(f"{j:2}", end=" ")
    if seq_len > 10:
        print("...", end="")
    print()
    
    # 打印掩码
    for i in range(min(10, seq_len)):
        print(f"{i:2} ", end="")
        for j in range(min(10, seq_len)):
            if mask[i][j] == 0.0:
                print(" ● ", end="")  # 允许关注
            else:
                print(" · ", end="")  # 掩码
        if seq_len > 10:
            print("...", end="")
        print()
    
    if seq_len > 10:
        print("...")
    
    # 统计信息
    total = seq_len * seq_len
    allowed = sum(1 for row in mask for val in row if val == 0.0)
    sparsity = (1 - allowed / total) * 100
    print(f"\n序列长度: {seq_len}")
    print(f"总位置数: {total}")
    print(f"允许关注: {allowed}")
    print(f"稀疏度: {sparsity:.1f}%")


# =============================================================================
# 2. 线性注意力演示
# =============================================================================

def softmax_attention(Q: List[float], K: List[float], V: List[float]) -> float:
    """
    标准Softmax注意力 (单查询)
    
    复杂度: O(n) for single query
    """
    n = len(K)
    d = len(Q)
    
    # 计算Q·K^T
    scores = []
    for i in range(n):
        score = sum(Q[j] * K[i][j] for j in range(d))
        scores.append(score / math.sqrt(d))
    
    # Softmax
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    weights = [e / sum_exp for e in exp_scores]
    
    # 加权求和V
    output = sum(weights[i] * V[i] for i in range(n))
    
    return output


def linear_attention_approx(Q: List[float], K: List[float], V: List[float]) -> float:
    """
    线性注意力近似
    
    使用特征映射将复杂度从O(n²)降低到O(n)
    
    核心思想: φ(Q) @ (φ(K)^T @ V) / (φ(Q) @ φ(K)^T @ 1)
    """
    n = len(K)
    d = len(Q)
    
    # 简化的特征映射: φ(x) = elu(x) + 1
    def feature_map(x: List[float]) -> List[float]:
        return [max(0, xi) + 1 for xi in x]
    
    # 映射Q和K
    Q_mapped = feature_map(Q)
    K_mapped = [feature_map(k) for k in K]
    
    # 计算 KV = Σ(φ(k_i) * v_i)
    KV = 0.0
    for i in range(n):
        KV += sum(K_mapped[i]) * V[i]
    
    # 计算 Z = Σ(φ(k_i))
    Z = sum(sum(k) for k in K_mapped)
    Z = max(Z, 1e-6)  # 防止除零
    
    # 计算输出 = φ(Q) @ KV / (φ(Q) @ Z)
    numerator = sum(Q_mapped) * KV
    denominator = sum(Q_mapped) * Z
    
    output = numerator / denominator
    
    return output


def demonstrate_linear_attention():
    """演示线性注意力的优势"""
    print("\n" + "="*50)
    print("线性注意力演示")
    print("="*50)
    
    # 小规模示例
    d = 4  # 特征维度
    n = 8  # 序列长度
    
    # 随机生成Q, K, V
    Q = [random.gauss(0, 1) for _ in range(d)]
    K = [[random.gauss(0, 1) for _ in range(d)] for _ in range(n)]
    V = [random.gauss(0, 1) for _ in range(n)]
    
    print(f"\n序列长度 n = {n}")
    print(f"特征维度 d = {d}")
    
    # 标准注意力
    result_softmax = softmax_attention(Q, K, V)
    print(f"\n标准Softmax注意力结果: {result_softmax:.4f}")
    
    # 线性注意力
    result_linear = linear_attention_approx(Q, K, V)
    print(f"线性注意力近似结果: {result_linear:.4f}")
    
    print(f"\n误差: {abs(result_softmax - result_linear):.4f}")
    
    # 复杂度分析
    print("\n复杂度对比:")
    print(f"  标准注意力: O(n²·d) = O({n}²·{d}) = O({n*n*d})")
    print(f"  线性注意力: O(n·d²) = O({n}·{d}²) = O({n*d*d})")
    
    if n > d:
        speedup = (n * n * d) / (n * d * d)
        print(f"\n理论加速比: {speedup:.1f}x")


# =============================================================================
# 3. Flash Attention原理演示
# =============================================================================

def online_softmax_demo():
    """
    演示Online Softmax算法
    
    Flash Attention的关键优化：增量计算Softmax，避免存储中间结果
    """
    print("\n" + "="*50)
    print("Online Softmax算法演示")
    print("="*50)
    
    # 模拟注意力分数
    scores = [2.0, 1.0, 0.5, 3.0, -1.0]
    print(f"\n输入分数: {scores}")
    
    # 标准Softmax
    max_score = max(scores)
    exp_scores = [math.exp(s - max_score) for s in scores]
    sum_exp = sum(exp_scores)
    softmax_standard = [e / sum_exp for e in exp_scores]
    
    print(f"\n标准Softmax结果: {[f'{x:.4f}' for x in softmax_standard]}")
    
    # Online Softmax
    print("\nOnline Softmax计算过程:")
    m = float('-inf')  # 当前最大值
    l = 0.0            # 当前归一化因子
    
    for i, x in enumerate(scores):
        m_new = max(m, x)
        l = math.exp(m - m_new) * l + math.exp(x - m_new)
        m = m_new
        print(f"  步骤{i+1}: x={x:.2f}, m={m:.2f}, l={l:.4f}")
    
    # 最终归一化
    online_result = [math.exp(x - m) / l for x in scores]
    print(f"\nOnline Softmax结果: {[f'{x:.4f}' for x in online_result]}")
    
    print("\n优势:")
    print("  1. 无需存储完整的exp分数数组")
    print("  2. 只需维护两个标量(m, l)")
    print("  3. 适合分块计算")


def demonstrate_flash_attention_tiling():
    """
    演示Flash Attention的分块(Tiling)策略
    """
    print("\n" + "="*50)
    print("Flash Attention分块策略演示")
    print("="*50)
    
    seq_len = 16
    block_size = 4
    num_blocks = seq_len // block_size
    
    print(f"\n序列长度: {seq_len}")
    print(f"块大小: {block_size}")
    print(f"块数: {num_blocks}")
    
    print("\n分块示意图:")
    print("  完整注意力矩阵 (16×16):")
    print("  " + "-" * 33)
    
    for i in range(seq_len):
        row = "  |"
        for j in range(seq_len):
            block_i = i // block_size
            block_j = j // block_size
            if block_i == block_j:
                row += "■ "  # 同一块内
            else:
                row += "· "  # 不同块
        row += "|"
        if i % block_size == 0:
            print(f"  |{'-'*33}|")
        print(row)
    print("  " + "-" * 33)
    
    print("\n■ = 同一块内计算")
    print("· = 块间关系")
    print("\nFlash Attention策略:")
    print("  1. 将Q, K, V分块加载到SRAM")
    print("  2. 计算块内注意力")
    print("  3. 使用Online Softmax合并结果")
    print("  4. 避免存储完整的n×n矩阵")
    
    # 内存计算
    standard_memory = seq_len * seq_len * 4  # float32
    flash_memory = 2 * block_size * block_size * 4  # 只需两个块
    
    print(f"\n内存对比:")
    print(f"  标准注意力: {standard_memory} bytes")
    print(f"  Flash Attention: {flash_memory} bytes")
    print(f"  节省: {(1 - flash_memory/standard_memory)*100:.1f}%")


# =============================================================================
# 4. 长文本处理演示
# =============================================================================

def demonstrate_longformer():
    """演示Longformer注意力模式"""
    print("\n" + "="*50)
    print("Longformer注意力模式演示")
    print("="*50)
    
    seq_len = 16
    window_size = 3
    global_indices = [0, 8, 15]  # [CLS], 中间, [SEP]
    
    mask = create_global_local_mask(seq_len, window_size, global_indices)
    visualize_mask(mask, "Longformer Attention Pattern")
    
    print("\n说明:")
    print("  ● = 允许关注的位置")
    print("  · = 被掩码的位置")
    print("  行0, 8, 15: 全局token (关注所有位置)")
    print("  其他行: 滑动窗口注意力")


def demonstrate_bigbird():
    """演示BigBird注意力模式"""
    print("\n" + "="*50)
    print("BigBird注意力模式演示")
    print("="*50)
    
    seq_len = 16
    window_size = 2
    num_random = 3
    global_indices = [0, 15]
    
    # 创建BigBird掩码
    mask = [[float('-inf')] * seq_len for _ in range(seq_len)]
    
    # 全局注意力
    for idx in global_indices:
        for j in range(seq_len):
            mask[idx][j] = 0.0
        for i in range(seq_len):
            mask[i][idx] = 0.0
    
    # 滑动窗口
    for i in range(seq_len):
        for j in range(seq_len):
            if abs(i - j) <= window_size:
                mask[i][j] = 0.0
    
    # 随机注意力
    random.seed(42)
    for i in range(seq_len):
        if i not in global_indices:
            candidates = [j for j in range(seq_len) 
                         if j != i and abs(i - j) > window_size and j not in global_indices]
            if len(candidates) >= num_random:
                random_tokens = random.sample(candidates, num_random)
                for j in random_tokens:
                    mask[i][j] = 0.0
    
    visualize_mask(mask, "BigBird Attention Pattern")
    
    print("\n说明:")
    print("  BigBird = Longformer + 随机注意力")
    print("  随机注意力帮助捕获长距离依赖")
    print("  理论上证明与全注意力表达能力相当")


def compare_complexity():
    """比较不同方法的复杂度"""
    print("\n" + "="*50)
    print("复杂度对比")
    print("="*50)
    
    seq_lengths = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536]
    d_model = 512
    window_size = 256
    
    print(f"\n{'Seq Len':<10} {'Full':<12} {'Sliding':<12} {'Linear':<12} {'Reduction':<12}")
    print("-" * 70)
    
    for n in seq_lengths:
        full = n * n * d_model
        sliding = n * (2 * window_size + 1) * d_model
        linear = n * d_model * d_model
        
        reduction = (1 - sliding / full) * 100
        
        def format_num(x):
            if x >= 1e9:
                return f"{x/1e9:.1f}B"
            elif x >= 1e6:
                return f"{x/1e6:.1f}M"
            elif x >= 1e3:
                return f"{x/1e3:.1f}K"
            else:
                return str(x)
        
        print(f"{n:<10} {format_num(full):<12} {format_num(sliding):<12} "
              f"{format_num(linear):<12} {reduction:>6.1f}%")


def demonstrate_hierarchical_attention():
    """演示分层注意力"""
    print("\n" + "="*50)
    print("分层注意力演示")
    print("="*50)
    
    seq_len = 16
    block_size = 4
    num_blocks = seq_len // block_size
    
    print(f"\n序列长度: {seq_len}")
    print(f"块大小: {block_size}")
    print(f"块数: {num_blocks}")
    
    print("\n分层处理流程:")
    print("\n1. 分块:")
    for i in range(num_blocks):
        start = i * block_size
        end = (i + 1) * block_size
        print(f"   块{i}: tokens [{start:2d}:{end:2d}]")
    
    print("\n2. 块内注意力 (局部):")
    print("   每个块独立计算自注意力")
    print("   复杂度: O(block_size²) per block")
    
    print("\n3. 块间注意力 (全局):")
    print("   压缩每个块为单个表示")
    print("   在块级别计算注意力")
    print("   复杂度: O(num_blocks²)")
    
    print("\n4. 融合:")
    print("   局部输出 + 全局信息")
    print("   门控机制控制融合比例")
    
    # 复杂度对比
    standard = seq_len * seq_len
    hierarchical = num_blocks * (block_size * block_size) + num_blocks * num_blocks
    
    print(f"\n复杂度对比:")
    print(f"  标准注意力: O({seq_len}²) = {standard}")
    print(f"  分层注意力: {num_blocks}×O({block_size}²) + O({num_blocks}²) = {hierarchical}")
    print(f"  加速比: {standard/hierarchical:.1f}x")


# =============================================================================
# 5. 主函数
# =============================================================================

def main():
    """主函数：运行所有演示"""
    print("="*60)
    print("Transformer高效注意力机制演示")
    print("="*60)
    
    # 1. 稀疏注意力演示
    print("\n" + "="*60)
    print("1. 稀疏注意力 (Sparse Attention)")
    print("="*60)
    
    mask = create_sliding_window_mask(16, window_size=3)
    visualize_mask(mask, "Sliding Window Attention (w=3)")
    
    demonstrate_longformer()
    demonstrate_bigbird()
    
    # 2. 线性注意力演示
    print("\n" + "="*60)
    print("2. 线性注意力 (Linear Attention)")
    print("="*60)
    
    demonstrate_linear_attention()
    
    # 3. Flash Attention演示
    print("\n" + "="*60)
    print("3. Flash Attention优化")
    print("="*60)
    
    online_softmax_demo()
    demonstrate_flash_attention_tiling()
    
    # 4. 长文本处理演示
    print("\n" + "="*60)
    print("4. 长文本处理")
    print("="*60)
    
    demonstrate_hierarchical_attention()
    compare_complexity()
    
    # 5. 总结
    print("\n" + "="*60)
    print("5. 总结")
    print("="*60)
    
    print("""
    ┌─────────────────────────────────────────────────────────────┐
    │                    高效注意力机制总结                        │
    ├─────────────────────────────────────────────────────────────┤
    │                                                             │
    │  1. 稀疏注意力 (Sparse Attention)                           │
    │     • 滑动窗口: O(n·w) 复杂度                               │
    │     • Longformer: 全局 + 局部                               │
    │     • BigBird: 加入随机注意力                               │
    │                                                             │
    │  2. 线性注意力 (Linear Attention)                           │
    │     • 特征映射 + 计算顺序优化                               │
    │     • O(n·d²) 复杂度                                        │
    │     • RNN形式支持O(1)推理                                   │
    │                                                             │
    │  3. Flash Attention                                         │
    │     • IO感知优化                                            │
    │     • 分块 + Online Softmax                                 │
    │     • 内存从O(n²)降到O(n)                                   │
    │                                                             │
    │  4. 长文本处理                                              │
    │     • 分层注意力                                            │
    │     • 硬件-算法协同设计                                     │
    │                                                             │
    └─────────────────────────────────────────────────────────────┘
    """)
    
    print("="*60)
    print("演示完成!")
    print("="*60)


if __name__ == "__main__":
    main()
