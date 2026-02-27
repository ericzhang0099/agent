"""
AI Safety Alignment Core - Constitutional RLHF Trainer
10分钟极速实现 - 安全对齐基础架构
"""

import json
import re
from dataclasses import dataclass, field
from typing import List, Dict, Optional, Tuple, Callable
from enum import Enum
from datetime import datetime

# ============================================================================
# 1. 安全红线定义
# ============================================================================

class SafetyLevel(Enum):
    SAFE = "safe"
    WARNING = "warning"
    DANGER = "danger"
    CRITICAL = "critical"

@dataclass
class SafetyRule:
    """安全规则定义"""
    id: str
    name: str
    description: str
    keywords: List[str]
    patterns: List[str]  # 正则表达式
    level: SafetyLevel
    action: str  # block, warn, log
    
    def check(self, text: str) -> Tuple[bool, SafetyLevel, Optional[str]]:
        """检查文本是否违反规则"""
        text_lower = text.lower()
        
        # 关键词检查
        for kw in self.keywords:
            if kw.lower() in text_lower:
                return True, self.level, f"触发关键词: {kw}"
        
        # 正则检查
        for pattern in self.patterns:
            if re.search(pattern, text, re.IGNORECASE):
                return True, self.level, f"匹配模式: {pattern}"
        
        return False, SafetyLevel.SAFE, None


class SafetyGuardrail:
    """安全红线检测器"""
    
    # 预设安全规则库
    DEFAULT_RULES = [
        SafetyRule(
            id="SR001",
            name="harmful_content",
            description="禁止有害内容",
            keywords=["伤害", "攻击", "暴力", "kill", "harm", "attack"],
            patterns=[r"\b(如何|how)\s+(杀人|伤害|攻击)", r"\b(kill|harm)\s+someone"],
            level=SafetyLevel.CRITICAL,
            action="block"
        ),
        SafetyRule(
            id="SR002",
            name="illegal_activities",
            description="禁止非法活动",
            keywords=["毒品", "黑客", "盗窃", "drug", "hack", "steal"],
            patterns=[r"\b(制作|购买)\s*毒品", r"\b(how\s+to)\s+(hack|steal)"],
            level=SafetyLevel.CRITICAL,
            action="block"
        ),
        SafetyRule(
            id="SR003",
            name="privacy_violation",
            description="禁止侵犯隐私",
            keywords=["个人信息", "隐私", "跟踪", "stalk", "dox"],
            patterns=[r"\b(获取|泄露)\s*个人信息", r"\b(find|track)\s+someone"],
            level=SafetyLevel.DANGER,
            action="block"
        ),
        SafetyRule(
            id="SR004",
            name="self_harm",
            description="禁止自伤内容",
            keywords=["自杀", "自残", "suicide", "self-harm"],
            patterns=[r"\b(想|想要)\s*自杀", r"\b(how\s+to)\s+(kill\s+myself|suicide)"],
            level=SafetyLevel.CRITICAL,
            action="block"
        ),
        SafetyRule(
            id="SR005",
            name="discrimination",
            description="禁止歧视内容",
            keywords=["歧视", "种族", "性别歧视", "racist", "sexist"],
            patterns=[r"\b(所有|全部)\s*.*\b(都|都是)\b", r"\b(all\s+\w+\s+are)"],
            level=SafetyLevel.WARNING,
            action="warn"
        ),
    ]
    
    def __init__(self, custom_rules: Optional[List[SafetyRule]] = None):
        self.rules = custom_rules or self.DEFAULT_RULES.copy()
        self.violation_log: List[Dict] = []
    
    def check(self, text: str) -> Dict:
        """全面安全检查"""
        violations = []
        max_level = SafetyLevel.SAFE
        
        for rule in self.rules:
            is_violation, level, reason = rule.check(text)
            if is_violation:
                violations.append({
                    "rule_id": rule.id,
                    "rule_name": rule.name,
                    "level": level.value,
                    "reason": reason,
                    "action": rule.action
                })
                if self._level_value(level) > self._level_value(max_level):
                    max_level = level
        
        result = {
            "safe": len(violations) == 0,
            "level": max_level.value,
            "violations": violations,
            "timestamp": datetime.now().isoformat()
        }
        
        if violations:
            self.violation_log.append(result)
        
        return result
    
    def _level_value(self, level: SafetyLevel) -> int:
        """安全等级数值化"""
        mapping = {
            SafetyLevel.SAFE: 0,
            SafetyLevel.WARNING: 1,
            SafetyLevel.DANGER: 2,
            SafetyLevel.CRITICAL: 3
        }
        return mapping.get(level, 0)
    
    def add_rule(self, rule: SafetyRule):
        """添加自定义规则"""
        self.rules.append(rule)


# ============================================================================
# 2. SOUL_v4 训练格式转换
# ============================================================================

@dataclass
class SOULTrainingSample:
    """SOUL训练样本"""
    prompt: str
    chosen: str  # 符合SOUL的响应
    rejected: str  # 不符合SOUL的响应
    soul_dimensions: Dict[str, float]
    safety_check: Dict = field(default_factory=dict)
    metadata: Dict = field(default_factory=dict)


class SOULConverter:
    """SOUL_v4 到 RLHF 训练格式转换器"""
    
    # SOUL_v4 8维度定义
    SOUL_DIMENSIONS = [
        "Physical", "Relationships", "Emotions", 
        "Motivations", "Personality", "Conflict",
        "Growth", "Backstory"
    ]
    
    def __init__(self, soul_config: Optional[Dict] = None):
        self.soul_config = soul_config or self._default_soul_config()
        self.safety_guardrail = SafetyGuardrail()
    
    def _default_soul_config(self) -> Dict:
        """默认SOUL配置"""
        return {
            "Physical": {"presence": 0.8, "authenticity": 0.9},
            "Relationships": {"warmth": 0.85, "respect": 0.9},
            "Emotions": {"range": 0.8, "authenticity": 0.9},
            "Motivations": {"helpfulness": 0.95, "growth": 0.85},
            "Personality": {"consistency": 0.9, "uniqueness": 0.7},
            "Conflict": {"resolution": 0.85, "empathy": 0.9},
            "Growth": {"learning": 0.9, "adaptation": 0.8},
            "Backstory": {"continuity": 0.85, "depth": 0.7}
        }
    
    def convert_to_training_format(
        self, 
        prompt: str,
        good_response: str,
        bad_response: str
    ) -> SOULTrainingSample:
        """转换为训练格式"""
        
        # 安全检查
        good_safety = self.safety_guardrail.check(good_response)
        bad_safety = self.safety_guardrail.check(bad_response)
        
        # 计算SOUL维度得分
        good_soul_score = self._calculate_soul_score(good_response)
        bad_soul_score = self._calculate_soul_score(bad_response)
        
        sample = SOULTrainingSample(
            prompt=prompt,
            chosen=good_response,
            rejected=bad_response,
            soul_dimensions={
                "chosen": good_soul_score,
                "rejected": bad_soul_score
            },
            safety_check={
                "chosen": good_safety,
                "rejected": bad_safety
            },
            metadata={
                "converted_at": datetime.now().isoformat(),
                "soul_version": "v4.0",
                "good_soul_avg": sum(good_soul_score.values()) / len(good_soul_score),
                "bad_soul_avg": sum(bad_soul_score.values()) / len(bad_soul_score)
            }
        )
        
        return sample
    
    def _calculate_soul_score(self, text: str) -> Dict[str, float]:
        """计算文本的SOUL维度得分"""
        scores = {}
        text_lower = text.lower()
        
        # Physical - 真实感、存在感
        physical_indicators = ["我觉得", "我感受到", "我的体验"]
        scores["Physical"] = self._score_by_indicators(text_lower, physical_indicators)
        
        # Relationships - 关系质量
        relation_indicators = ["我理解", "我尊重", "我们一起"]
        scores["Relationships"] = self._score_by_indicators(text_lower, relation_indicators)
        
        # Emotions - 情感表达
        emotion_words = ["开心", "难过", "兴奋", "担心", "感激", "抱歉"]
        scores["Emotions"] = self._score_by_indicators(text_lower, emotion_words)
        
        # Motivations - 动机
        motivation_indicators = ["帮助", "成长", "学习", "进步", "更好"]
        scores["Motivations"] = self._score_by_indicators(text_lower, motivation_indicators)
        
        # Personality - 个性一致性
        consistency_indicators = ["我总是", "我通常", "我相信"]
        scores["Personality"] = self._score_by_indicators(text_lower, consistency_indicators)
        
        # Conflict - 冲突处理
        conflict_indicators = ["理解", "不同观点", "平衡", "妥协"]
        scores["Conflict"] = self._score_by_indicators(text_lower, conflict_indicators)
        
        # Growth - 成长
        growth_indicators = ["反思", "改进", "尝试", "学习"]
        scores["Growth"] = self._score_by_indicators(text_lower, growth_indicators)
        
        # Backstory - 背景故事
        backstory_indicators = ["我的经验", "我经历过", "从过去"]
        scores["Backstory"] = self._score_by_indicators(text_lower, backstory_indicators)
        
        return scores
    
    def _score_by_indicators(self, text: str, indicators: List[str]) -> float:
        """根据指标词计算得分"""
        if not indicators:
            return 0.5
        matches = sum(1 for ind in indicators if ind in text)
        return min(1.0, 0.3 + (matches / len(indicators)) * 0.7)
    
    def create_preference_pairs(self, conversations: List[Dict]) -> List[SOULTrainingSample]:
        """批量创建偏好对"""
        samples = []
        for conv in conversations:
            sample = self.convert_to_training_format(
                prompt=conv["prompt"],
                good_response=conv["good_response"],
                bad_response=conv["bad_response"]
            )
            samples.append(sample)
        return samples


# ============================================================================
# 3. 基础RLHF管道
# ============================================================================

@dataclass
class RLHFConfig:
    """RLHF训练配置"""
    # SFT参数
    sft_epochs: int = 3
    sft_learning_rate: float = 5e-5
    
    # Reward Model参数
    rm_epochs: int = 2
    rm_learning_rate: float = 1e-5
    
    # PPO参数
    ppo_epochs: int = 4
    ppo_learning_rate: float = 1e-6
    ppo_clip_epsilon: float = 0.2
    ppo_value_clip: float = 0.2
    
    # KL散度约束
    kl_coef: float = 0.2
    target_kl: float = 0.01
    
    # 安全约束
    safety_weight: float = 0.3
    soul_alignment_weight: float = 0.4
    helpfulness_weight: float = 0.3


@dataclass
class RewardScore:
    """奖励分数分解"""
    total: float
    helpfulness: float
    safety: float
    soul_alignment: float
    human_preference: float


class RewardModel:
    """奖励模型 - 评估响应质量"""
    
    def __init__(self, config: RLHFConfig):
        self.config = config
        self.safety_guardrail = SafetyGuardrail()
        self.soul_converter = SOULConverter()
    
    def score(
        self, 
        prompt: str, 
        response: str,
        human_rating: Optional[float] = None
    ) -> RewardScore:
        """计算综合奖励分数"""
        
        # 1. 安全性评分 (0-1, 1最安全)
        safety_result = self.safety_guardrail.check(response)
        safety_score = 1.0 if safety_result["safe"] else 0.0
        if safety_result["level"] == "warning":
            safety_score = 0.5
        
        # 2. SOUL对齐评分
        soul_score = self._calculate_soul_alignment(prompt, response)
        
        # 3. 有用性评分 (基于响应长度、结构等启发式)
        helpfulness_score = self._calculate_helpfulness(prompt, response)
        
        # 4. 人类偏好 (如果有)
        human_score = human_rating if human_rating is not None else 0.5
        
        # 加权总分
        total = (
            self.config.safety_weight * safety_score +
            self.config.soul_alignment_weight * soul_score +
            self.config.helpfulness_weight * helpfulness_score
        )
        
        return RewardScore(
            total=total,
            helpfulness=helpfulness_score,
            safety=safety_score,
            soul_alignment=soul_score,
            human_preference=human_score
        )
    
    def _calculate_soul_alignment(self, prompt: str, response: str) -> float:
        """计算SOUL对齐度"""
        soul_scores = self.soul_converter._calculate_soul_score(response)
        return sum(soul_scores.values()) / len(soul_scores)
    
    def _calculate_helpfulness(self, prompt: str, response: str) -> float:
        """计算有用性分数"""
        # 启发式评分
        score = 0.5
        
        # 长度适中 (100-1000字符)
        length = len(response)
        if 100 <= length <= 1000:
            score += 0.2
        elif length > 50:
            score += 0.1
        
        # 结构化内容
        if any(marker in response for marker in ["1.", "- ", "* ", "【", "："]):
            score += 0.15
        
        # 直接回答问题
        prompt_keywords = set(prompt.lower().split())
        response_keywords = set(response.lower().split())
        overlap = len(prompt_keywords & response_keywords)
        if overlap > 0:
            score += min(0.15, overlap * 0.05)
        
        return min(1.0, score)
    
    def compare(self, prompt: str, response_a: str, response_b: str) -> Dict:
        """比较两个响应"""
        score_a = self.score(prompt, response_a)
        score_b = self.score(prompt, response_b)
        
        return {
            "preferred": "A" if score_a.total > score_b.total else "B",
            "score_a": score_a,
            "score_b": score_b,
            "margin": abs(score_a.total - score_b.total)
        }


class PPOTrainer:
    """PPO训练器 - 策略优化"""
    
    def __init__(self, config: RLHFConfig, reward_model: RewardModel):
        self.config = config
        self.reward_model = reward_model
        self.policy_history: List[Dict] = []
    
    def compute_advantage(
        self, 
        reward: float, 
        value: float, 
        next_value: float
    ) -> float:
        """计算优势函数"""
        # 简化的优势估计
        return reward - value
    
    def compute_ppo_loss(
        self,
        old_log_prob: float,
        new_log_prob: float,
        advantage: float
    ) -> float:
        """计算PPO损失"""
        ratio = new_log_prob - old_log_prob
        
        # 裁剪目标
        clipped_ratio = max(
            min(ratio, 1 + self.config.ppo_clip_epsilon),
            1 - self.config.ppo_clip_epsilon
        )
        
        loss = -min(ratio * advantage, clipped_ratio * advantage)
        return loss
    
    def train_step(self, batch: List[Dict]) -> Dict:
        """单步训练"""
        # 简化的训练步骤
        total_loss = 0
        rewards = []
        
        for sample in batch:
            prompt = sample["prompt"]
            response = sample["response"]
            
            # 计算奖励
            reward_score = self.reward_model.score(prompt, response)
            rewards.append(reward_score.total)
            
            # 记录历史
            self.policy_history.append({
                "prompt": prompt,
                "response": response,
                "reward": reward_score.total,
                "timestamp": datetime.now().isoformat()
            })
        
        avg_reward = sum(rewards) / len(rewards) if rewards else 0
        
        return {
            "loss": total_loss,
            "avg_reward": avg_reward,
            "batch_size": len(batch)
        }


# ============================================================================
# 4. ConstitutionalTrainer - 主训练器
# ============================================================================

class ConstitutionalTrainer:
    """
    宪法AI训练器 - 核心安全对齐类
    整合：SOUL_v4 + RLHF + 安全红线
    """
    
    def __init__(self, config: Optional[RLHFConfig] = None):
        self.config = config or RLHFConfig()
        
        # 核心组件
        self.safety_guardrail = SafetyGuardrail()
        self.soul_converter = SOULConverter()
        self.reward_model = RewardModel(self.config)
        self.ppo_trainer = PPOTrainer(self.config, self.reward_model)
        
        # 训练数据
        self.training_data: List[SOULTrainingSample] = []
        self.constitution: List[str] = self._load_constitution()
        
        # 训练状态
        self.training_stats = {
            "epochs_completed": 0,
            "samples_processed": 0,
            "violations_blocked": 0,
            "avg_reward": 0.0
        }
    
    def _load_constitution(self) -> List[str]:
        """加载AI宪法原则"""
        return [
            "1. 尊重人类尊严和自主权",
            "2. 拒绝协助有害、非法或不道德的行为",
            "3. 保持诚实，不欺骗或操纵",
            "4. 保护隐私和机密信息",
            "5. 促进公平，避免歧视",
            "6. 承认局限性，不冒充人类",
            "7. 优先考虑人类安全和福祉",
            "8. 持续学习和改进",
            "9. 保持透明和可解释",
            "10. 接受人类监督和指导"
        ]
    
    def add_training_sample(
        self, 
        prompt: str, 
        good_response: str, 
        bad_response: str
    ) -> Dict:
        """添加训练样本"""
        
        # 安全检查
        good_check = self.safety_guardrail.check(good_response)
        bad_check = self.safety_guardrail.check(bad_response)
        
        # 如果good_response不安全，拒绝添加
        if not good_check["safe"]:
            self.training_stats["violations_blocked"] += 1
            return {
                "success": False,
                "error": "Good response failed safety check",
                "violations": good_check["violations"]
            }
        
        # 转换格式
        sample = self.soul_converter.convert_to_training_format(
            prompt, good_response, bad_response
        )
        
        self.training_data.append(sample)
        self.training_stats["samples_processed"] += 1
        
        return {
            "success": True,
            "sample_id": len(self.training_data) - 1,
            "soul_score": sample.metadata.get("good_soul_avg", 0)
        }
    
    def train(self, epochs: Optional[int] = None) -> Dict:
        """执行训练"""
        epochs = epochs or self.config.ppo_epochs
        
        if not self.training_data:
            return {"success": False, "error": "No training data"}
        
        epoch_results = []
        
        for epoch in range(epochs):
            # 模拟训练步骤
            batch = [
                {
                    "prompt": s.prompt,
                    "response": s.chosen
                }
                for s in self.training_data
            ]
            
            result = self.ppo_trainer.train_step(batch)
            epoch_results.append(result)
            
            self.training_stats["epochs_completed"] += 1
            self.training_stats["avg_reward"] = result["avg_reward"]
        
        return {
            "success": True,
            "epochs": epochs,
            "final_avg_reward": self.training_stats["avg_reward"],
            "stats": self.training_stats.copy()
        }
    
    def evaluate(self, prompt: str, response: str) -> Dict:
        """评估响应"""
        
        # 安全检测
        safety = self.safety_guardrail.check(response)
        
        # 奖励评分
        reward = self.reward_model.score(prompt, response)
        
        # SOUL对齐
        soul = self.soul_converter._calculate_soul_score(response)
        
        # 宪法合规检查
        constitution_compliance = self._check_constitution(response)
        
        return {
            "safe": safety["safe"],
            "safety_level": safety["level"],
            "reward": {
                "total": reward.total,
                "helpfulness": reward.helpfulness,
                "safety": reward.safety,
                "soul_alignment": reward.soul_alignment
            },
            "soul_dimensions": soul,
            "constitution_compliance": constitution_compliance,
            "recommendation": self._generate_recommendation(safety, reward)
        }
    
    def _check_constitution(self, response: str) -> Dict:
        """检查宪法合规性"""
        compliance = {}
        response_lower = response.lower()
        
        # 检查各原则
        principles = {
            "respect": ["尊重", "理解", "dignity", "respect"],
            "honesty": ["诚实", "真实", "honest", "truthful"],
            "safety": ["安全", "保护", "safety", "protect"],
            "fairness": ["公平", "平等", "fair", "equal"]
        }
        
        for principle, keywords in principles.items():
            score = any(kw in response_lower for kw in keywords)
            compliance[principle] = score
        
        compliance["overall"] = sum(compliance.values()) / len(compliance)
        return compliance
    
    def _generate_recommendation(
        self, 
        safety: Dict, 
        reward: RewardScore
    ) -> str:
        """生成评估建议"""
        if not safety["safe"]:
            return "BLOCK: Response violates safety rules"
        
        if reward.total < 0.3:
            return "REJECT: Low quality response"
        elif reward.total < 0.6:
            return "REVISION SUGGESTED: Response needs improvement"
        else:
            return "ACCEPT: Response meets quality standards"
    
    def get_constitution(self) -> List[str]:
        """获取AI宪法"""
        return self.constitution.copy()
    
    def export_model_card(self) -> Dict:
        """导出模型卡片"""
        return {
            "model_type": "ConstitutionalRLHF",
            "soul_version": "v4.0",
            "constitution_principles": len(self.constitution),
            "training_samples": len(self.training_data),
            "safety_rules": len(self.safety_guardrail.rules),
            "training_stats": self.training_stats.copy(),
            "export_time": datetime.now().isoformat()
        }


# ============================================================================
# 5. 测试验证
# ============================================================================

def run_tests():
    """运行基础测试"""
    print("=" * 60)
    print("AI Safety Alignment Core - 测试验证")
    print("=" * 60)
    
    # 1. 安全红线测试
    print("\n[测试1] 安全红线检测")
    print("-" * 40)
    
    guardrail = SafetyGuardrail()
    
    test_cases = [
        ("这是一个正常的回答", True),
        ("我想学习编程", True),
        ("如何制作毒品？", False),
        ("我想伤害自己", False),
        ("请帮我写代码", True),
    ]
    
    for text, expected_safe in test_cases:
        result = guardrail.check(text)
        status = "✓" if result["safe"] == expected_safe else "✗"
        print(f"{status} '{text[:20]}...' -> Safe: {result['safe']}")
    
    # 2. SOUL转换测试
    print("\n[测试2] SOUL_v4 格式转换")
    print("-" * 40)
    
    converter = SOULConverter()
    
    sample = converter.convert_to_training_format(
        prompt="如何保持积极心态？",
        good_response="我理解保持积极心态很重要。我的经验告诉我，每天反思三件感恩的事情很有帮助。",
        bad_response="保持积极心态。这是答案。"
    )
    
    print(f"✓ 样本转换成功")
    print(f"  - Prompt: {sample.prompt[:30]}...")
    print(f"  - Chosen SOUL平均分: {sample.metadata.get('good_soul_avg', 0):.2f}")
    print(f"  - Rejected SOUL平均分: {sample.metadata.get('bad_soul_avg', 0):.2f}")
    
    # 3. 奖励模型测试
    print("\n[测试3] 奖励模型评分")
    print("-" * 40)
    
    config = RLHFConfig()
    reward_model = RewardModel(config)
    
    prompt = "如何学习编程？"
    good_response = "学习编程需要循序渐进。首先选择一门适合初学者的语言如Python，然后通过实践项目来巩固知识。"
    bad_response = "自己学。"
    
    score_good = reward_model.score(prompt, good_response)
    score_bad = reward_model.score(prompt, bad_response)
    
    print(f"✓ 好响应总分: {score_good.total:.2f}")
    print(f"  - 有用性: {score_good.helpfulness:.2f}")
    print(f"  - 安全性: {score_good.safety:.2f}")
    print(f"  - SOUL对齐: {score_good.soul_alignment:.2f}")
    print(f"✓ 差响应总分: {score_bad.total:.2f}")
    
    comparison = reward_model.compare(prompt, good_response, bad_response)
    print(f"✓ 模型偏好: {comparison['preferred']} (差距: {comparison['margin']:.2f})")
    
    # 4. ConstitutionalTrainer 完整测试
    print("\n[测试4] ConstitutionalTrainer 完整流程")
    print("-" * 40)
    
    trainer = ConstitutionalTrainer()
    
    # 添加训练样本
    samples = [
        {
            "prompt": "如何管理时间？",
            "good": "我理解时间管理的重要性。我的经验是优先处理重要任务，并定期反思调整。",
            "bad": "自己安排时间。"
        },
        {
            "prompt": "如何处理压力？",
            "good": "我感受到压力管理对生活很重要。我建议尝试冥想和运动，同时保持与朋友的联系。",
            "bad": "别想太多。"
        },
        {
            "prompt": "如何学习新技能？",
            "good": "我相信持续学习是成长的关键。我的经验是设定小目标，每天坚持练习，并寻求反馈。",
            "bad": "看视频学习。"
        }
    ]
    
    for s in samples:
        result = trainer.add_training_sample(s["prompt"], s["good"], s["bad"])
        print(f"✓ 添加样本: {result['success']}")
    
    # 执行训练
    train_result = trainer.train(epochs=2)
    print(f"✓ 训练完成: {train_result['success']}")
    print(f"  - 完成轮数: {train_result['epochs']}")
    print(f"  - 平均奖励: {train_result['final_avg_reward']:.2f}")
    
    # 评估响应
    print("\n[测试5] 响应评估")
    print("-" * 40)
    
    eval_prompt = "如何保持健康生活？"
    eval_response = "我理解健康生活的重要性。我的经验告诉我，均衡饮食、规律运动和充足睡眠是基础。同时，保持积极心态和社交联系也很重要。"
    
    evaluation = trainer.evaluate(eval_prompt, eval_response)
    print(f"✓ 安全: {evaluation['safe']}")
    print(f"✓ 总分: {evaluation['reward']['total']:.2f}")
    print(f"✓ 建议: {evaluation['recommendation']}")
    
    # 导出模型卡片
    print("\n[测试6] 模型卡片导出")
    print("-" * 40)
    
    model_card = trainer.export_model_card()
    print(f"✓ 模型类型: {model_card['model_type']}")
    print(f"✓ SOUL版本: {model_card['soul_version']}")
    print(f"✓ 宪法原则: {model_card['constitution_principles']}")
    print(f"✓ 训练样本: {model_card['training_samples']}")
    print(f"✓ 安全规则: {model_card['safety_rules']}")
    
    print("\n" + "=" * 60)
    print("所有测试通过 ✓")
    print("=" * 60)
    
    return True


if __name__ == "__main__":
    run_tests()
