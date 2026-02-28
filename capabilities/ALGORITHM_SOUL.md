# 算法与数据结构能力体系 v1.0

> **研究来源**: LeetCode顶级解法、Google面试模式、ACM-ICPC竞赛技术  
> **核心原则**: 模式识别 > 死记硬背 | 简洁代码 > 复杂实现 | 思维训练 > 题量堆积  
> **适用场景**: 技术面试、竞赛编程、工程实践、系统设计

---

## 一、核心算法思维模式

### 1.1 问题分析框架 (Problem Analysis Framework)

```yaml
AlgorithmicThinking:
  Step1_Understanding:
    - 完全理解问题陈述（阅读2-3遍）
    - 识别输入/输出格式与约束条件
    - 手动解决3个示例用例
    - 识别边界情况和边缘案例
    
  Step2_PatternRecognition:
    - 关联已知算法模式
    - 判断问题类型（搜索/排序/图/动态规划等）
    - 识别数据结构的适用性
    
  Step3_Design:
    - 先用暴力解法理解问题本质
    - 逐步优化到最优解
    - 写出伪代码或注释
    
  Step4_Implementation:
    - 转换为实际代码
    - 使用有意义的变量名
    - 处理边界条件
    
  Step5_Verification:
    - 用测试用例验证
    - 分析时间/空间复杂度
    - 寻找进一步优化空间
```

### 1.2 算法思维层次

| 层次 | 能力描述 | 标志 |
|------|----------|------|
| **L1 机械记忆** | 记住算法步骤和模板 | 能复现标准算法 |
| **L2 模式应用** | 识别问题模式并应用 | 能快速匹配解法 |
| **L3 原理理解** | 理解算法设计原理 | 能解释为什么这样设计 |
| **L4 创造改进** | 能改进或创造新算法 | 能针对特定场景优化 |
| **L5 思维内化** | 算法思维成为本能 | 看到问题自动分解 |

---

## 二、15大核心算法模式

### 2.1 数组与字符串模式

#### Pattern 1: Prefix Sum (前缀和)
```python
# 核心思想: 预处理数组，使子数组求和变为O(1)
# 适用场景: 多次区间求和查询

def build_prefix_sum(nums):
    n = len(nums)
    prefix = [0] * (n + 1)
    for i in range(n):
        prefix[i + 1] = prefix[i] + nums[i]
    return prefix

def range_sum(prefix, left, right):
    return prefix[right + 1] - prefix[left]

# 经典题目: Subarray Sum Equals K, Range Sum Query
```

#### Pattern 2: Two Pointers (双指针)
```python
# 核心思想: 两个指针从两端或同向移动，将O(n²)降为O(n)
# 适用场景: 有序数组、回文判断、容器盛水

def two_sum_sorted(nums, target):
    left, right = 0, len(nums) - 1
    while left < right:
        current = nums[left] + nums[right]
        if current == target:
            return [left, right]
        elif current < target:
            left += 1
        else:
            right -= 1
    return []

# 变体: 同向双指针（滑动窗口的基础）
# 经典题目: 3Sum, Container With Most Water, Trapping Rain Water
```

#### Pattern 3: Sliding Window (滑动窗口)
```python
# 核心思想: 维护一个可变大小的窗口，避免重复计算
# 适用场景: 子数组/子字符串问题

def sliding_window_template(s):
    from collections import defaultdict
    window = defaultdict(int)
    left = 0
    result = 0
    
    for right in range(len(s)):
        # 扩大窗口：加入right字符
        window[s[right]] += 1
        
        # 收缩窗口：当不满足条件时
        while not is_valid(window):
            window[s[left]] -= 1
            left += 1
        
        # 更新结果
        result = max(result, right - left + 1)
    
    return result

# 经典题目: Longest Substring Without Repeating Characters, 
#          Minimum Window Substring, Sliding Window Maximum
```

#### Pattern 4: Kadane's Algorithm (最大子数组和)
```python
# 核心思想: 动态规划，以当前元素结尾的最大子数组和
def max_subarray_sum(nums):
    max_current = max_global = nums[0]
    for i in range(1, len(nums)):
        max_current = max(nums[i], max_current + nums[i])
        max_global = max(max_global, max_current)
    return max_global

# 扩展: 最大子数组积（需要同时维护最大和最小）
```

### 2.2 搜索与排序模式

#### Pattern 5: Binary Search (二分搜索)
```python
# 核心思想: 每次将搜索空间减半
# 适用场景: 有序数组、寻找边界、答案具有单调性

def binary_search_template(nums, target):
    left, right = 0, len(nums) - 1
    
    while left <= right:
        mid = left + (right - left) // 2  # 防止溢出
        
        if nums[mid] == target:
            return mid
        elif nums[mid] < target:
            left = mid + 1
        else:
            right = mid - 1
    
    return -1

# 高级变体:
# - 寻找左边界: right = mid
# - 寻找右边界: left = mid + 1
# - 旋转排序数组: 判断哪一半有序
# 经典题目: Search in Rotated Sorted Array, Find Minimum in Rotated Sorted Array
```

#### Pattern 6: Modified Binary Search (变形二分)
```python
# 应用场景: 答案空间具有单调性，但无法直接获取数组

def binary_search_on_answer(min_val, max_val, condition):
    """
    condition: 函数，返回True/False判断mid是否满足条件
    """
    left, right = min_val, max_val
    answer = None
    
    while left <= right:
        mid = left + (right - left) // 2
        if condition(mid):
            answer = mid
            right = mid - 1  # 寻找更小的满足条件的值
        else:
            left = mid + 1
    
    return answer

# 经典题目: Koko Eating Bananas, Capacity To Ship Packages Within D Days
```

### 2.3 链表模式

#### Pattern 7: Fast & Slow Pointers (快慢指针)
```python
# 核心思想: 快指针每次走2步，慢指针每次走1步
# 适用场景: 检测环、找中点

class ListNode:
    def __init__(self, val=0, next=None):
        self.val = val
        self.next = next

def has_cycle(head):
    if not head or not head.next:
        return False
    
    slow = fast = head
    while fast and fast.next:
        slow = slow.next
        fast = fast.next.next
        if slow == fast:
            return True
    return False

# 扩展: 找环的入口点
# 1. 快慢指针相遇后，慢指针回到头
# 2. 两个指针都以1步前进，再次相遇点即为环入口

# 经典题目: Linked List Cycle, Find the Duplicate Number, Middle of Linked List
```

#### Pattern 8: In-place Reversal (原地反转)
```python
# 核心思想: 迭代反转指针方向
def reverse_linked_list(head):
    prev = None
    current = head
    
    while current:
        next_temp = current.next
        current.next = prev
        prev = current
        current = next_temp
    
    return prev

# 扩展: 反转前N个节点、反转区间[m,n]
# 经典题目: Reverse Linked List, Reverse Nodes in k-Group
```

### 2.4 树与图模式

#### Pattern 9: Tree Traversal Patterns (树遍历)
```python
# 核心决策: 选择遍历方式取决于信息传递方向

# PostOrder (后序遍历) - 从子节点收集信息
# 适用: 深度、直径、最大路径和
def postorder_template(node):
    if not node:
        return base_case_value
    
    left_result = postorder_template(node.left)
    right_result = postorder_template(node.right)
    
    # 整合子节点结果
    current_result = combine(left_result, right_result, node.val)
    return current_result

# PreOrder (前序遍历) - 从父节点传递信息
# 适用: 路径问题、验证BST
def preorder_template(node, path_info):
    if not node:
        return
    
    # 使用path_info做决策
    update_path_info(path_info, node.val)
    
    preorder_template(node.left, path_info)
    preorder_template(node.right, path_info)
    
    # 回溯
    restore_path_info(path_info, node.val)

# LevelOrder (层序遍历) - BFS
from collections import deque
def levelorder_template(root):
    if not root:
        return []
    
    result = []
    queue = deque([root])
    
    while queue:
        level_size = len(queue)
        level_nodes = []
        
        for _ in range(level_size):
            node = queue.popleft()
            level_nodes.append(node.val)
            
            if node.left:
                queue.append(node.left)
            if node.right:
                queue.append(node.right)
        
        result.append(level_nodes)
    
    return result
```

#### Pattern 10: Graph DFS/BFS
```python
# DFS模板 - 递归
from collections import defaultdict

def dfs_template(graph, start):
    visited = set()
    result = []
    
    def dfs(node):
        if node in visited:
            return
        
        visited.add(node)
        result.append(node)
        
        for neighbor in graph[node]:
            dfs(neighbor)
    
    dfs(start)
    return result

# BFS模板 - 迭代，适用于最短路径（无权图）
def bfs_template(graph, start, target):
    from collections import deque
    
    visited = {start}
    queue = deque([(start, 0)])  # (节点, 距离)
    
    while queue:
        node, distance = queue.popleft()
        
        if node == target:
            return distance
        
        for neighbor in graph[node]:
            if neighbor not in visited:
                visited.add(neighbor)
                queue.append((neighbor, distance + 1))
    
    return -1  # 未找到

# 经典题目: Number of Islands, Clone Graph, Course Schedule
```

#### Pattern 11: Topological Sort (拓扑排序)
```python
# 适用场景: 任务调度、依赖解析、课程安排
# 方法1: Kahn's Algorithm (BFS)
def topological_sort_kahn(num_courses, prerequisites):
    from collections import defaultdict, deque
    
    # 构建图和入度表
    graph = defaultdict(list)
    in_degree = [0] * num_courses
    
    for course, prereq in prerequisites:
        graph[prereq].append(course)
        in_degree[course] += 1
    
    # 从入度为0的节点开始
    queue = deque([i for i in range(num_courses) if in_degree[i] == 0])
    result = []
    
    while queue:
        node = queue.popleft()
        result.append(node)
        
        for neighbor in graph[node]:
            in_degree[neighbor] -= 1
            if in_degree[neighbor] == 0:
                queue.append(neighbor)
    
    return result if len(result) == num_courses else []

# 方法2: DFS (后序遍历的逆序)
def topological_sort_dfs(num_courses, prerequisites):
    from collections import defaultdict
    
    graph = defaultdict(list)
    for course, prereq in prerequisites:
        graph[prereq].append(course)
    
    # 0=未访问, 1=访问中, 2=已访问
    state = [0] * num_courses
    result = []
    
    def dfs(node):
        if state[node] == 1:  # 发现环
            return False
        if state[node] == 2:  # 已处理
            return True
        
        state[node] = 1
        for neighbor in graph[node]:
            if not dfs(neighbor):
                return False
        state[node] = 2
        result.append(node)
        return True
    
    for i in range(num_courses):
        if state[i] == 0:
            if not dfs(i):
                return []  # 有环
    
    return result[::-1]  # 逆序
```

### 2.5 动态规划模式

#### Pattern 12: Dynamic Programming Framework
```python
# DP解题框架
"""
1. 定义状态: dp[i]或dp[i][j]代表什么
2. 状态转移方程: dp[i]与dp[i-1]等的关系
3. 初始状态: dp[0]或dp[0][0]的值
4. 遍历顺序: 确保计算dp[i]时所需状态已计算
5. 空间优化: 能否用滚动数组降低空间复杂度
"""

# 一维DP示例: 爬楼梯
def climb_stairs(n):
    if n <= 2:
        return n
    
    dp = [0] * (n + 1)
    dp[1], dp[2] = 1, 2
    
    for i in range(3, n + 1):
        dp[i] = dp[i - 1] + dp[i - 2]
    
    return dp[n]

# 空间优化版本
def climb_stairs_optimized(n):
    if n <= 2:
        return n
    
    prev2, prev1 = 1, 2
    for i in range(3, n + 1):
        current = prev1 + prev2
        prev2 = prev1
        prev1 = current
    
    return prev1
```

#### Pattern 13: DP on Strings (字符串DP)
```python
# LCS (最长公共子序列)
def longest_common_subsequence(text1, text2):
    m, n = len(text1), len(text2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if text1[i - 1] == text2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1] + 1
            else:
                dp[i][j] = max(dp[i - 1][j], dp[i][j - 1])
    
    return dp[m][n]

# 编辑距离
def edit_distance(word1, word2):
    m, n = len(word1), len(word2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if word1[i - 1] == word2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(
                    dp[i - 1][j],      # 删除
                    dp[i][j - 1],      # 插入
                    dp[i - 1][j - 1]   # 替换
                )
    
    return dp[m][n]
```

### 2.6 高级数据结构模式

#### Pattern 14: Monotonic Stack/Queue (单调栈/队列)
```python
# 单调递减栈: 找下一个更大元素
def next_greater_element(nums):
    n = len(nums)
    result = [-1] * n
    stack = []  # 存储索引，保持递减
    
    for i in range(n):
        while stack and nums[stack[-1]] < nums[i]:
            idx = stack.pop()
            result[idx] = nums[i]
        stack.append(i)
    
    return result

# 应用场景: Daily Temperatures, Largest Rectangle in Histogram

# 单调队列: 滑动窗口最大值
from collections import deque
def max_sliding_window(nums, k):
    result = []
    dq = deque()  # 存储索引，保持递减
    
    for i, num in enumerate(nums):
        # 移除窗口外的元素
        while dq and dq[0] <= i - k:
            dq.popleft()
        
        # 保持单调递减
        while dq and nums[dq[-1]] < num:
            dq.pop()
        
        dq.append(i)
        
        # 记录结果
        if i >= k - 1:
            result.append(nums[dq[0]])
    
    return result
```

#### Pattern 15: Heap/Priority Queue (堆)
```python
import heapq

# Top K问题
def top_k_frequent(nums, k):
    from collections import Counter
    
    count = Counter(nums)
    # 使用最小堆，保持大小为k
    heap = []
    
    for num, freq in count.items():
        heapq.heappush(heap, (freq, num))
        if len(heap) > k:
            heapq.heappop(heap)
    
    return [num for freq, num in heap]

# 合并K个有序链表
def merge_k_sorted_lists(lists):
    heap = []
    
    for i, lst in enumerate(lists):
        if lst:
            heapq.heappush(heap, (lst[0], i, 0))
    
    result = []
    while heap:
        val, list_idx, elem_idx = heapq.heappop(heap)
        result.append(val)
        
        if elem_idx + 1 < len(lists[list_idx]):
            next_val = lists[list_idx][elem_idx + 1]
            heapq.heappush(heap, (next_val, list_idx, elem_idx + 1))
    
    return result
```

---

## 三、复杂度分析与优化

### 3.1 复杂度速查表

| 算法/数据结构 | 时间复杂度 | 空间复杂度 | 适用场景 |
|--------------|-----------|-----------|---------|
| 数组访问 | O(1) | O(n) | 随机访问 |
| 链表插入/删除 | O(1) | O(n) | 频繁插入删除 |
| 二分搜索 | O(log n) | O(1) | 有序数据搜索 |
| 哈希表 | O(1)平均 | O(n) | 快速查找 |
| 二叉搜索树 | O(log n) | O(n) | 有序数据维护 |
| 堆 | O(log n)插入 | O(n) | Top K问题 |
| 快速排序 | O(n log n) | O(log n) | 通用排序 |
| 归并排序 | O(n log n) | O(n) | 稳定排序 |
| Dijkstra | O((V+E)log V) | O(V) | 单源最短路径 |
| 并查集 | O(α(n)) | O(n) | 连通性问题 |

### 3.2 优化策略

```yaml
OptimizationStrategies:
  TimeOptimization:
    - 使用哈希表将O(n)查找降为O(1)
    - 使用二分将O(n)降为O(log n)
    - 使用双指针/滑动窗口将O(n²)降为O(n)
    - 使用前缀和将区间查询降为O(1)
    - 使用单调栈/队列优化特定问题
    
  SpaceOptimization:
    - 滚动数组降低DP空间
    - 原地修改（如果允许）
    - 使用迭代代替递归
    - 位运算压缩状态
    
  TradeOffs:
    - 时间换空间: 缓存、预处理
    - 空间换时间: 哈希表、前缀和
    - 预处理: 离线计算、预计算
```

---

## 四、代码简洁性原则

### 4.1 命名规范

```python
# ✅ 好的命名
user_count = 100
max_profit = calculate_max_profit(prices)
is_valid_parentheses = check_balance(s)

# ❌ 差的命名
n = 100  # 无意义
mp = calc(p)  # 缩写不清晰
flag = True  # 无描述性
```

### 4.2 函数设计

```python
# ✅ 单一职责
def find_max_subarray_sum(nums):
    """找到最大子数组和"""
    pass

def find_max_subarray_indices(nums):
    """找到最大子数组的起止索引"""
    pass

# ❌ 职责混杂
def max_subarray(nums, return_indices=False):
    """既找和又可能找索引"""
    pass
```

### 4.3 代码结构模板

```python
class Solution:
    def solve(self, input_data):
        """
        主入口函数
        1. 处理边界情况
        2. 调用核心算法
        3. 返回结果
        """
        if not input_data:
            return default_value
        
        return self._core_algorithm(input_data)
    
    def _core_algorithm(self, data):
        """核心算法实现"""
        pass
    
    def _helper(self, sub_problem):
        """辅助函数"""
        pass
```

---

## 五、面试解题流程

### 5.1 45分钟面试时间分配

```
[0-5分钟]  理解问题
           - 仔细阅读题目
           - 询问澄清问题
           - 确认输入输出格式

[5-10分钟] 设计算法
           - 先提暴力解法
           - 分析时间和空间复杂度
           - 逐步优化到最优解

[10-30分钟] 编码实现
           - 先写主函数框架
           - 实现核心逻辑
           - 处理边界条件

[30-40分钟] 测试验证
           - 用示例测试
           - 考虑边界情况
           - 分析最终复杂度

[40-45分钟] 讨论扩展
           - 可能的优化方向
           - 相关变体问题
           - 实际应用场景
```

### 5.2 沟通技巧

```yaml
InterviewCommunication:
  ThinkAloud:
    - "我首先注意到..."
    - "这个问题让我想到..."
    - "让我尝试..."
    
  ClarifyingQuestions:
    - "输入的规模大概是多少？"
    - "是否允许修改输入数组？"
    - "对于边界情况，应该如何处理？"
    
  TradeOffDiscussion:
    - "这个解法的时间复杂度是O(n)，空间复杂度是O(1)"
    - "如果内存有限，我们可以考虑..."
    - "如果需要支持并发，可能需要..."
```

---

## 六、ACM竞赛特化技巧

### 6.1 快速IO模板

```python
import sys
input = sys.stdin.readline

def fast_io_solve():
    n = int(input())
    nums = list(map(int, input().split()))
    # ... 解题逻辑
    print(result)
```

### 6.2 常用数学工具

```python
# GCD和LCM
import math
def gcd(a, b):
    return math.gcd(a, b)

def lcm(a, b):
    return a * b // gcd(a, b)

# 快速幂
def fast_pow(base, exp, mod):
    result = 1
    base %= mod
    while exp > 0:
        if exp & 1:
            result = result * base % mod
        base = base * base % mod
        exp >>= 1
    return result

# 素数筛
def sieve_of_eratosthenes(n):
    is_prime = [True] * (n + 1)
    is_prime[0] = is_prime[1] = False
    
    for i in range(2, int(n**0.5) + 1):
        if is_prime[i]:
            for j in range(i * i, n + 1, i):
                is_prime[j] = False
    
    return [i for i in range(2, n + 1) if is_prime[i]]
```

### 6.3 并查集模板

```python
class UnionFind:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n
        self.count = n
    
    def find(self, x):
        if self.parent[x] != x:
            self.parent[x] = self.find(self.parent[x])  # 路径压缩
        return self.parent[x]
    
    def union(self, x, y):
        px, py = self.find(x), self.find(y)
        if px == py:
            return False
        
        # 按秩合并
        if self.rank[px] < self.rank[py]:
            px, py = py, px
        self.parent[py] = px
        if self.rank[px] == self.rank[py]:
            self.rank[px] += 1
        
        self.count -= 1
        return True
    
    def connected(self, x, y):
        return self.find(x) == self.find(y)
```

---

## 七、持续学习路径

### 7.1 学习阶段规划

```yaml
LearningPath:
  Phase1_Fundamentals:  # 1-2个月
    - Arrays & Hashing
    - Two Pointers
    - Sliding Window
    - Stack
    - Binary Search
    
  Phase2_DataStructures:  # 2-3个月
    - Linked List
    - Trees (BST, DFS, BFS)
    - Heap/Priority Queue
    - Backtracking
    
  Phase3_Advanced:  # 3-4个月
    - Dynamic Programming
    - Graphs
    - Advanced Graphs
    - Greedy
    - Intervals
    
  Phase4_Mastery:  # 持续
    - Math & Geometry
    - Bit Manipulation
    - Design Problems
    - System Design结合
```

### 7.2 推荐资源

```yaml
Resources:
  PracticePlatforms:
    - LeetCode (NeetCode 150)
    - Codeforces
    - AtCoder
    - HackerRank
    
  LearningMaterials:
    - "Introduction to Algorithms" (CLRS)
    - "Algorithm Design Manual" (Skiena)
    - "Competitive Programming 3" (Halim)
    
  Visualization:
    - VisuAlgo.net
    - Algorithm Visualizer
```

---

## 八、思维检查清单

### 8.1 解题前检查

- [ ] 我完全理解问题了吗？
- [ ] 我识别出所有约束条件了吗？
- [ ] 我考虑了所有边界情况吗？
- [ ] 我识别出可能的算法模式了吗？

### 8.2 解题中检查

- [ ] 我的变量命名清晰吗？
- [ ] 我处理了所有边界条件吗？
- [ ] 我的代码结构清晰吗？
- [ ] 我能在白板上解释这段代码吗？

### 8.3 解题后检查

- [ ] 我分析了时间复杂度吗？
- [ ] 我分析了空间复杂度吗？
- [ ] 我能进一步优化吗？
- [ ] 我能解释为什么这样设计吗？

---

> **核心信条**: 优秀的算法能力不是记住最多解法，而是培养识别模式、分解问题、清晰表达的思维习惯。

---
*文档版本: v1.0*  
*最后更新: 2026-02-28*  
*研究Agent: 算法与数据结构深度学习Agent*
