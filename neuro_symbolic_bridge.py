"""
神经符号融合模块 - Neuro-Symbolic Bridge
连接神经网络与符号推理系统的桥梁
"""

import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum


class SymbolType(Enum):
    """符号类型"""
    PREDICATE = "predicate"
    CONSTANT = "constant"
    VARIABLE = "variable"


class NeuralOutputType(Enum):
    """神经网络输出类型"""
    CLASSIFICATION = "classification"
    EMBEDDING = "embedding"


@dataclass
class Symbol:
    """符号表示"""
    name: str
    symbol_type: SymbolType
    arguments: List[str] = field(default_factory=list)
    value: Any = None
    confidence: float = 1.0
    
    def __str__(self) -> str:
        if self.arguments:
            return f"{self.name}({', '.join(self.arguments)})"
        return self.name
    
    def __hash__(self) -> int:
        return hash((self.name, self.symbol_type, tuple(self.arguments)))
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, Symbol):
            return False
        return (self.name == other.name and 
                self.symbol_type == other.symbol_type and
                self.arguments == other.arguments)


@dataclass
class NeuralOutput:
    """神经网络输出"""
    output_type: NeuralOutputType
    values: np.ndarray
    labels: Optional[List[str]] = None
    confidence: float = 1.0
    embeddings: Optional[np.ndarray] = None
    
    def get_top_k(self, k: int = 3) -> List[Tuple[str, float]]:
        """获取top-k预测结果"""
        if self.labels is None or len(self.values.shape) == 0:
            return [(str(self.values), 1.0)]
        flat_values = self.values.flatten()
        top_indices = np.argsort(flat_values)[-k:][::-1]
        return [(self.labels[i], float(flat_values[i])) for i in top_indices]


@dataclass
class FusedRepresentation:
    """神经符号融合表示"""
    symbols: List[Symbol]
    neural_output: NeuralOutput
    fusion_score: float
    explanation: str = ""
    reasoning_chain: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "symbols": [str(s) for s in self.symbols],
            "neural_output": {
                "type": self.neural_output.output_type.value,
                "top_predictions": self.neural_output.get_top_k()
            },
            "fusion_score": self.fusion_score,
            "explanation": self.explanation,
            "reasoning_chain": self.reasoning_chain
        }


class NeuroSymbolicBridge:
    """神经符号桥梁 - 实现双向转换与融合"""
    
    def __init__(self, confidence_threshold: float = 0.6):
        self.confidence_threshold = confidence_threshold
        self.label_to_symbol = {}
        self.symbol_embeddings = {}
        self.explanation_rules = {}
        self.fusion_history = []
    
    def register_label_mapping(self, label: str, symbol_name: str):
        """注册标签到符号的映射"""
        self.label_to_symbol[label] = symbol_name
    
    def register_explanation_rule(self, pattern: str, explanation: str):
        """注册解释规则"""
        self.explanation_rules[pattern] = explanation
    
    def neural_to_symbol(self, neural_output: NeuralOutput) -> List[Symbol]:
        """将神经网络输出转换为符号表示"""
        symbols = []
        
        if neural_output.output_type == NeuralOutputType.CLASSIFICATION:
            for label, score in neural_output.get_top_k():
                if score >= self.confidence_threshold:
                    symbol_name = self.label_to_symbol.get(label, f"is_{label}")
                    symbols.append(Symbol(
                        name=symbol_name,
                        symbol_type=SymbolType.PREDICATE,
                        value=score,
                        confidence=score
                    ))
        
        return symbols
    
    def symbol_to_neural(self, symbols: List[Symbol]) -> np.ndarray:
        """将符号表示转换为神经嵌入向量"""
        if not symbols:
            return np.zeros(128)
        
        embeddings = []
        for symbol in symbols:
            if symbol.name in self.symbol_embeddings:
                emb = self.symbol_embeddings[symbol.name] * symbol.confidence
            else:
                emb = np.random.randn(128) * 0.1
            embeddings.append(emb)
        
        return np.mean(embeddings, axis=0) if embeddings else np.zeros(128)
    
    def fuse(self, neural_output: NeuralOutput, 
             context_symbols: Optional[List[Symbol]] = None) -> FusedRepresentation:
        """融合神经输出与符号表示"""
        derived_symbols = self.neural_to_symbol(neural_output)
        all_symbols = derived_symbols + (context_symbols or [])
        
        fusion_score = self._compute_fusion_score(neural_output, all_symbols)
        explanation = self._generate_explanation(all_symbols, neural_output)
        reasoning_chain = self._build_reasoning_chain(all_symbols, neural_output)
        
        fused = FusedRepresentation(
            symbols=all_symbols,
            neural_output=neural_output,
            fusion_score=fusion_score,
            explanation=explanation,
            reasoning_chain=reasoning_chain
        )
        
        self.fusion_history.append(fused)
        return fused
    
    def _compute_fusion_score(self, neural_output: NeuralOutput, 
                              symbols: List[Symbol]) -> float:
        """计算融合分数"""
        if not symbols:
            return neural_output.confidence * 0.5
        
        symbol_confidences = [s.confidence for s in symbols]
        avg_symbol_conf = np.mean(symbol_confidences)
        consistency = min(len(symbols), 5) / 5.0
        
        return min(neural_output.confidence * avg_symbol_conf * consistency, 1.0)
    
    def _generate_explanation(self, symbols: List[Symbol], 
                             neural_output: NeuralOutput) -> str:
        """生成自然语言解释"""
        parts = []
        
        top_preds = neural_output.get_top_k(3)
        if top_preds:
            pred_str = ", ".join([f"{label}({score:.2f})" for label, score in top_preds])
            parts.append(f"神经网络预测: {pred_str}")
        
        if symbols:
            symbol_str = ", ".join([str(s) for s in symbols[:5]])
            parts.append(f"符号推导: {symbol_str}")
        
        for symbol in symbols:
            for pattern, explanation in self.explanation_rules.items():
                if pattern in symbol.name:
                    parts.append(f"规则解释: {explanation}")
        
        return "; ".join(parts)
    
    def _build_reasoning_chain(self, symbols: List[Symbol], 
                               neural_output: NeuralOutput) -> List[str]:
        """构建推理链"""
        chain = [
            f"1. 神经网络处理输入，输出类型: {neural_output.output_type.value}",
            f"2. 转换为符号表示，得到 {len(symbols)} 个符号"
        ]
        for i, symbol in enumerate(symbols[:3], 3):
            chain.append(f"{i}. 符号 '{symbol}' (置信度: {symbol.confidence:.2f})")
        return chain


def create_default_bridge() -> NeuroSymbolicBridge:
    """创建默认的神经符号桥梁"""
    bridge = NeuroSymbolicBridge(confidence_threshold=0.6)
    bridge.register_explanation_rule("dog", "这是一个犬科动物")
    bridge.register_explanation_rule("cat", "这是一个猫科动物")
    return bridge


if __name__ == "__main__":
    # 示例用法
    bridge = create_default_bridge()
    bridge.register_label_mapping("dog", "is_dog")
    
    neural_out = NeuralOutput(
        output_type=NeuralOutputType.CLASSIFICATION,
        values=np.array([0.1, 0.8, 0.1]),
        labels=["cat", "dog", "bird"],
        confidence=0.85
    )
    
    fused = bridge.fuse(neural_out)
    
    print("=== 神经符号融合结果 ===")
    print(f"融合分数: {fused.fusion_score:.3f}")
    print(f"解释: {fused.explanation}")
    print("\n推理链:")
    for step in fused.reasoning_chain:
        print(f"  {step}")
    print("\n符号表示:")
    for symbol in fused.symbols:
        print(f"  - {symbol} (置信度: {symbol.confidence:.2f})")
