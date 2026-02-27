"""
神经符号推理引擎 - Reasoning Engine
实现符号推理与神经推理的融合
"""

import numpy as np
from typing import Dict, List, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum, auto
from collections import defaultdict
import json

from neuro_symbolic_bridge import Symbol, SymbolType, NeuralOutput, NeuralOutputType, FusedRepresentation, NeuroSymbolicBridge


class RuleType(Enum):
    """规则类型"""
    DEDUCTIVE = auto()
    INDUCTIVE = auto()
    NEURAL = auto()
    HYBRID = auto()


class InferenceStrategy(Enum):
    """推理策略"""
    FORWARD_CHAINING = "forward"
    BACKWARD_CHAINING = "backward"
    BIDIRECTIONAL = "bidirectional"
    NEURAL_GUIDED = "neural_guided"


@dataclass
class Rule:
    """推理规则"""
    name: str
    rule_type: RuleType
    premises: List[Symbol]
    conclusions: List[Symbol]
    confidence: float = 1.0
    explanation_template: str = ""
    
    def __str__(self) -> str:
        premises_str = " ∧ ".join([str(p) for p in self.premises])
        conclusions_str = " ∧ ".join([str(c) for c in self.conclusions])
        return f"[{self.name}] {premises_str} → {conclusions_str} (conf: {self.confidence:.2f})"
    
    def apply(self, facts: Set[Symbol]) -> Optional[List[Symbol]]:
        """应用规则"""
        premises_satisfied = all(
            any(p == fact for fact in facts) for p in self.premises
        )
        
        if not premises_satisfied:
            return None
        
        results = []
        for conclusion in self.conclusions:
            adjusted_confidence = conclusion.confidence * self.confidence
            new_conclusion = Symbol(
                name=conclusion.name,
                symbol_type=conclusion.symbol_type,
                arguments=conclusion.arguments,
                value=conclusion.value,
                confidence=adjusted_confidence
            )
            results.append(new_conclusion)
        
        return results


@dataclass
class InferenceStep:
    """推理步骤"""
    step_number: int
    rule_applied: Optional[Rule]
    input_facts: Set[Symbol]
    output_facts: Set[Symbol]
    confidence: float
    explanation: str


@dataclass
class InferenceResult:
    """推理结果"""
    conclusions: Set[Symbol]
    reasoning_chain: List[InferenceStep]
    final_confidence: float
    explanation: str
    inference_steps: int = 0
    neural_symbolic_fusion_score: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "conclusions": [str(c) for c in self.conclusions],
            "final_confidence": self.final_confidence,
            "inference_steps": self.inference_steps,
            "fusion_score": self.neural_symbolic_fusion_score,
            "explanation": self.explanation,
            "reasoning_chain": [
                {"step": step.step_number, "rule": step.rule_applied.name if step.rule_applied else "neural",
                 "explanation": step.explanation}
                for step in self.reasoning_chain
            ]
        }


class KnowledgeBase:
    """知识库 - 存储符号知识和规则"""
    
    def __init__(self):
        self.facts: Set[Symbol] = set()
        self.rules: List[Rule] = []
        self.fact_index: Dict[str, Set[Symbol]] = defaultdict(set)
    
    def add_fact(self, fact: Symbol):
        """添加事实"""
        self.facts.add(fact)
        self.fact_index[fact.name].add(fact)
    
    def add_facts(self, facts: List[Symbol]):
        """批量添加事实"""
        for fact in facts:
            self.add_fact(fact)
    
    def add_rule(self, rule: Rule):
        """添加规则"""
        self.rules.append(rule)
    
    def query(self, pattern: Symbol) -> Set[Symbol]:
        """查询匹配的事实"""
        results = set()
        for fact in self.fact_index.get(pattern.name, set()):
            if self._matches(pattern, fact):
                results.add(fact)
        return results
    
    def _matches(self, pattern: Symbol, fact: Symbol) -> bool:
        """检查模式是否匹配事实"""
        if pattern.name != fact.name:
            return False
        if pattern.symbol_type != fact.symbol_type:
            return False
        if pattern.arguments and fact.arguments:
            return pattern.arguments == fact.arguments
        return True
    
    def get_all_facts(self) -> Set[Symbol]:
        """获取所有事实"""
        return self.facts.copy()


class NeuralSymbolicReasoner:
    """神经符号推理器 - 结合符号推理和神经网络的混合推理系统"""
    
    def __init__(self, bridge: Optional[NeuroSymbolicBridge] = None):
        self.kb = KnowledgeBase()
        self.bridge = bridge or NeuroSymbolicBridge()
        self.max_inference_depth = 10
        self.confidence_threshold = 0.3
    
    def add_knowledge(self, facts: List[Symbol], rules: List[Rule]):
        """添加知识到知识库"""
        self.kb.add_facts(facts)
        for rule in rules:
            self.kb.add_rule(rule)
    
    def reason(self, query: Optional[Symbol] = None,
               neural_input: Optional[NeuralOutput] = None,
               strategy: InferenceStrategy = InferenceStrategy.BIDIRECTIONAL,
               max_depth: int = 10) -> InferenceResult:
        """执行推理"""
        self.max_inference_depth = max_depth
        
        # 神经融合
        fused_representation = None
        if neural_input:
            fused_representation = self.bridge.fuse(neural_input)
            self.kb.add_facts(fused_representation.symbols)
        
        # 根据策略执行推理
        if strategy == InferenceStrategy.FORWARD_CHAINING:
            result = self._forward_chaining(query)
        elif strategy == InferenceStrategy.BACKWARD_CHAINING:
            result = self._backward_chaining(query)
        else:
            result = self._bidirectional_reasoning(query)
        
        # 计算融合分数
        if fused_representation:
            result.neural_symbolic_fusion_score = fused_representation.fusion_score
        
        result.explanation = self._generate_comprehensive_explanation(result, fused_representation)
        return result
    
    def _forward_chaining(self, query: Optional[Symbol] = None) -> InferenceResult:
        """前向链推理"""
        facts = self.kb.get_all_facts()
        new_facts: Set[Symbol] = set()
        reasoning_chain: List[InferenceStep] = []
        step_count = 0
        
        for depth in range(self.max_inference_depth):
            added_in_iteration = False
            
            for rule in self.kb.rules:
                conclusions = rule.apply(facts)
                
                if conclusions:
                    for conclusion in conclusions:
                        if conclusion.confidence >= self.confidence_threshold:
                            if conclusion not in facts and conclusion not in new_facts:
                                new_facts.add(conclusion)
                                added_in_iteration = True
                                
                                step_count += 1
                                step = InferenceStep(
                                    step_number=step_count,
                                    rule_applied=rule,
                                    input_facts=set(rule.premises),
                                    output_facts={conclusion},
                                    confidence=conclusion.confidence,
                                    explanation=f"应用规则 '{rule.name}' 推导出 {conclusion}"
                                )
                                reasoning_chain.append(step)
                                
                                if query and self._matches(query, conclusion):
                                    return InferenceResult(
                                        conclusions={conclusion},
                                        reasoning_chain=reasoning_chain,
                                        final_confidence=conclusion.confidence,
                                        explanation="",
                                        inference_steps=step_count
                                    )
            
            if not added_in_iteration:
                break
            facts.update(new_facts)
        
        return InferenceResult(
            conclusions=new_facts,
            reasoning_chain=reasoning_chain,
            final_confidence=self._aggregate_confidence(new_facts),
            explanation="",
            inference_steps=step_count
        )
    
    def _backward_chaining(self, query: Optional[Symbol] = None) -> InferenceResult:
        """后向链推理"""
        if query is None:
            return self._forward_chaining()
        
        goals = [query]
        proven: Set[Symbol] = set()
        reasoning_chain: List[InferenceStep] = []
        step_count = 0
        
        while goals and step_count < self.max_inference_depth:
            current_goal = goals.pop(0)
            
            matching_facts = self.kb.query(current_goal)
            if matching_facts:
                proven.update(matching_facts)
                continue
            
            for rule in self.kb.rules:
                for conclusion in rule.conclusions:
                    if self._matches(current_goal, conclusion):
                        for premise in rule.premises:
                            if premise not in proven:
                                goals.append(premise)
                        
                        step_count += 1
                        step = InferenceStep(
                            step_number=step_count,
                            rule_applied=rule,
                            input_facts=set(rule.premises),
                            output_facts={conclusion},
                            confidence=rule.confidence,
                            explanation=f"为证明 '{current_goal}'，应用规则 '{rule.name}'"
                        )
                        reasoning_chain.append(step)
        
        return InferenceResult(
            conclusions=proven,
            reasoning_chain=reasoning_chain,
            final_confidence=self._aggregate_confidence(proven),
            explanation="",
            inference_steps=step_count
        )
    
    def _bidirectional_reasoning(self, query: Optional[Symbol] = None) -> InferenceResult:
        """双向推理"""
        forward_result = self._forward_chaining(query)
        
        if query and not any(self._matches(query, c) for c in forward_result.conclusions):
            backward_result = self._backward_chaining(query)
            
            all_conclusions = forward_result.conclusions | backward_result.conclusions
            all_steps = forward_result.reasoning_chain + backward_result.reasoning_chain
            
            return InferenceResult(
                conclusions=all_conclusions,
                reasoning_chain=all_steps,
                final_confidence=max(forward_result.final_confidence, backward_result.final_confidence),
                explanation="",
                inference_steps=len(all_steps)
            )
        
        return forward_result
    
    def _matches(self, pattern: Symbol, fact: Symbol) -> bool:
        """检查模式匹配"""
        return pattern.name == fact.name and pattern.symbol_type == fact.symbol_type
    
    def _aggregate_confidence(self, facts: Set[Symbol]) -> float:
        """聚合置信度"""
        if not facts:
            return 0.0
        confidences = [f.confidence for f in facts]
        return float(np.mean(confidences))
    
    def _generate_comprehensive_explanation(self, result: InferenceResult,
                                            fused: Optional[FusedRepresentation]) -> str:
        """生成综合解释"""
        parts = []
        
        if fused:
            parts.append(f"神经符号融合分析:\n{fused.explanation}")
        
        parts.append(f"\n推理过程 ({result.inference_steps} 步):")
        for step in result.reasoning_chain[:5]:
            parts.append(f"  步骤 {step.step_number}: {step.explanation}")
        
        if len(result.reasoning_chain) > 5:
            parts.append(f"  ... 还有 {len(result.reasoning_chain) - 5} 步")
        
        parts.append(f"\n最终结论 (置信度: {result.final_confidence:.2f}):")
        for conclusion in list(result.conclusions)[:5]:
            parts.append(f"  - {conclusion}")
        
        return "\n".join(parts)


def create_animal_reasoning_rules() -> List[Rule]:
    """创建动物识别推理规则示例"""
    return [
        Rule(
            name="mammal_from_fur",
            rule_type=RuleType.DEDUCTIVE,
            premises=[Symbol("has_fur", SymbolType.PREDICATE)],
            conclusions=[Symbol("is_mammal", SymbolType.PREDICATE)],
            confidence=0.9,
            explanation_template="有毛发的动物是哺乳动物"
        ),
        Rule(
            name="dog_from_features",
            rule_type=RuleType.HYBRID,
            premises=[
                Symbol("is_mammal", SymbolType.PREDICATE),
                Symbol("has_tail", SymbolType.PREDICATE),
                Symbol("barks", SymbolType.PREDICATE)
            ],
            conclusions=[Symbol("is_dog", SymbolType.PREDICATE)],
            confidence=0.95,
            explanation_template="哺乳动物，有尾巴且会吠叫的是狗"
        ),
        Rule(
            name="cat_from_features",
            rule_type=RuleType.HYBRID,
            premises=[
                Symbol("is_mammal", SymbolType.PREDICATE),
                Symbol("has_tail", SymbolType.PREDICATE),
                Symbol("meows", SymbolType.PREDICATE)
            ],
            conclusions=[Symbol("is_cat", SymbolType.PREDICATE)],
            confidence=0.92,
            explanation_template="哺乳动物，有尾巴且会喵喵叫的是猫"
        ),
        Rule(
            name="danger_from_animal",
            rule_type=RuleType.DEDUCTIVE,
            premises=[
                Symbol("is_dog", SymbolType.PREDICATE),
                Symbol("is_aggressive", SymbolType.PREDICATE)
            ],
            conclusions=[
                Symbol("potential_danger", SymbolType.PREDICATE),
                Symbol("keep_distance", SymbolType.PREDICATE)
            ],
            confidence=0.85,
            explanation_template="具有攻击性的狗可能存在危险"
        )
    ]


def create_default_reasoner() -> NeuralSymbolicReasoner:
    """创建默认推理器"""
    reasoner = NeuralSymbolicReasoner()
    reasoner.add_knowledge([], create_animal_reasoning_rules())
    return reasoner


if __name__ == "__main__":
    print("=== 神经符号推理引擎演示 ===\n")
    
    reasoner = create_default_reasoner()
    
    initial_facts = [
        Symbol("has_fur", SymbolType.PREDICATE, confidence=0.95),
        Symbol("has_tail", SymbolType.PREDICATE, confidence=0.9),
        Symbol("barks", SymbolType.PREDICATE, confidence=0.88),
    ]
    reasoner.kb.add_facts(initial_facts)
    
    neural_output = NeuralOutput(
        output_type=NeuralOutputType.CLASSIFICATION,
        values=np.array([0.05, 0.85, 0.1]),
        labels=["cat", "dog", "bird"],
        confidence=0.87
    )
    
    print("执行神经符号推理...\n")
    result = reasoner.reason(
        query=Symbol("is_dog", SymbolType.PREDICATE),
        neural_input=neural_output,
        strategy=InferenceStrategy.BIDIRECTIONAL
    )
    
    print(result.explanation)
    print("\n" + "="*50)
    print("推理结果 (JSON格式):")
    print(json.dumps(result.to_dict(), indent=2, ensure_ascii=False))
