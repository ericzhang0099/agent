# 神经符号融合模块文档

## 概述

神经符号融合模块实现了神经网络与符号推理系统的深度集成，结合了两者的优势：
- **神经网络**：强大的模式识别和感知能力
- **符号系统**：可解释的逻辑推理和知识表示

## 文件结构

```
workspace/
├── neuro_symbolic_bridge.py    # 神经符号桥梁 (236行)
├── reasoning_engine.py         # 推理引擎 (433行)
└── neuro_symbolic_fusion_documentation.md  # 本文档
```

## 核心组件

### 1. NeuroSymbolicBridge (`neuro_symbolic_bridge.py`)

神经符号桥梁实现双向转换与融合：

- **神经→符号转换**：将神经网络输出（分类、嵌入）转换为符号表示
- **符号→神经转换**：将符号转换为神经嵌入向量
- **融合表示生成**：结合神经输出与符号表示，生成融合结果
- **可解释性生成**：自动生成自然语言解释和推理链

**核心类：**
```python
class NeuroSymbolicBridge:
    def neural_to_symbol(self, neural_output: NeuralOutput) -> List[Symbol]
    def symbol_to_neural(self, symbols: List[Symbol]) -> np.ndarray
    def fuse(self, neural_output, context_symbols=None) -> FusedRepresentation
```

### 2. NeuralSymbolicReasoner (`reasoning_engine.py`)

神经符号推理引擎实现混合推理：

- **知识库管理**：存储符号事实和推理规则
- **多策略推理**：
  - 前向链 (FORWARD_CHAINING)：从事实推导结论
  - 后向链 (BACKWARD_CHAINING)：从目标反向验证
  - 双向推理 (BIDIRECTIONAL)：结合前向和后向
  - 神经引导 (NEURAL_GUIDED)：神经网络指导规则选择
- **不确定性处理**：基于置信度的推理
- **推理链追踪**：记录完整推理过程

**核心类：**
```python
class NeuralSymbolicReasoner:
    def reason(self, query=None, neural_input=None, 
               strategy=InferenceStrategy.BIDIRECTIONAL) -> InferenceResult
```

## 研究内容实现

### 1. 符号-神经混合架构

```
输入 → [神经网络处理] → [NeuroSymbolicBridge融合] → [符号推理] → 可解释输出
         ↑                    ↓                           ↓
    感知层(黑盒)        融合层(灰盒)               认知层(白盒)
```

- 神经网络负责感知和模式识别
- 桥梁层实现双向转换和对齐
- 符号系统负责逻辑推理和解释

### 2. 神经符号推理引擎

支持4种推理策略：

| 策略 | 描述 | 适用场景 |
|------|------|----------|
| FORWARD_CHAINING | 从已知事实推导新结论 | 知识发现 |
| BACKWARD_CHAINING | 从目标反向寻找证据 | 假设检验 |
| BIDIRECTIONAL | 同时从事实和目标出发 | 复杂问题 |
| NEURAL_GUIDED | 神经网络指导规则选择 | 感知-认知融合 |

支持4种规则类型：
- DEDUCTIVE：演绎规则（一般→特殊）
- INDUCTIVE：归纳规则（特殊→一般）
- NEURAL：神经规则
- HYBRID：混合规则

### 3. 可解释性增强

**多层次解释：**
1. **神经层面**：Top-K预测及置信度
2. **符号层面**：推理链和规则应用记录
3. **融合层面**：神经-符号对齐说明

**输出格式：**
```
神经符号融合分析:
神经网络预测: dog(0.85), bird(0.10), cat(0.05)
符号推导: is_dog

推理过程 (2 步):
  步骤 1: 应用规则 'mammal_from_fur' 推导出 is_mammal
  步骤 2: 应用规则 'dog_from_features' 推导出 is_dog

最终结论 (置信度: 0.95):
  - is_dog
```

## 使用示例

### 基础用法

```python
from neuro_symbolic_bridge import (
    NeuroSymbolicBridge, NeuralOutput, NeuralOutputType,
    Symbol, SymbolType, create_default_bridge
)
from reasoning_engine import (
    NeuralSymbolicReasoner, Rule, RuleType,
    InferenceStrategy, create_default_reasoner
)
import numpy as np

# 1. 创建桥梁和推理器
bridge = create_default_bridge()
reasoner = create_default_reasoner()

# 2. 模拟神经网络输出
neural_output = NeuralOutput(
    output_type=NeuralOutputType.CLASSIFICATION,
    values=np.array([0.1, 0.8, 0.1]),
    labels=["cat", "dog", "bird"],
    confidence=0.85
)

# 3. 添加事实
reasoner.kb.add_fact(Symbol("has_fur", SymbolType.PREDICATE, confidence=0.95))
reasoner.kb.add_fact(Symbol("barks", SymbolType.PREDICATE, confidence=0.88))

# 4. 执行推理
result = reasoner.reason(
    query=Symbol("is_dog", SymbolType.PREDICATE),
    neural_input=neural_output,
    strategy=InferenceStrategy.BIDIRECTIONAL
)

print(result.explanation)
```

### 自定义规则

```python
rule = Rule(
    name="my_rule",
    rule_type=RuleType.DEDUCTIVE,
    premises=[Symbol("A", SymbolType.PREDICATE)],
    conclusions=[Symbol("B", SymbolType.PREDICATE)],
    confidence=0.9,
    explanation_template="A蕴含B"
)
reasoner.kb.add_rule(rule)
```

## 核心数据结构

### Symbol (符号)
```python
@dataclass
class Symbol:
    name: str              # 符号名称
    symbol_type: SymbolType # 类型(PREDICATE/CONSTANT/VARIABLE)
    arguments: List[str]   # 参数
    value: Any             # 值
    confidence: float      # 置信度
```

### NeuralOutput (神经输出)
```python
@dataclass
class NeuralOutput:
    output_type: NeuralOutputType
    values: np.ndarray
    labels: Optional[List[str]]
    confidence: float
```

### FusedRepresentation (融合表示)
```python
@dataclass
class FusedRepresentation:
    symbols: List[Symbol]
    neural_output: NeuralOutput
    fusion_score: float
    explanation: str
    reasoning_chain: List[str]
```

### InferenceResult (推理结果)
```python
@dataclass
class InferenceResult:
    conclusions: Set[Symbol]
    reasoning_chain: List[InferenceStep]
    final_confidence: float
    explanation: str
    neural_symbolic_fusion_score: float
```

## 应用场景

1. **智能视觉理解**：图像分类+场景推理
2. **自然语言推理**：语义编码+逻辑推导
3. **决策支持系统**：感知数据+业务规则
4. **异常检测**：模式识别+根因追溯

## 运行结果

```
=== 神经符号融合结果 ===
融合分数: 0.136
解释: 神经网络预测: dog(0.80), bird(0.10), cat(0.10); 符号推导: is_dog; 规则解释: 这是一个犬科动物

推理链:
  1. 神经网络处理输入，输出类型: classification
  2. 转换为符号表示，得到 1 个符号
  3. 符号 'is_dog' (置信度: 0.80)

符号表示:
  - is_dog (置信度: 0.80)

=== 神经符号推理引擎演示 ===
神经符号融合分析:
神经网络预测: dog(0.85), bird(0.10), cat(0.05); 符号推导: is_dog

推理过程 (1 步):
  步骤 1: 应用规则 'mammal_from_fur' 推导出 is_mammal

最终结论 (置信度: 0.90):
  - is_dog
  - is_mammal
```
