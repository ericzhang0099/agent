# AI安全与对齐研究项目 - RLHF+宪法训练

## 项目概述
基于Claude Soul Document和Constitutional AI方法，构建生产级安全对齐系统。

## 核心研究发现

### 1. Claude Soul Document训练方法

#### 1.1 文档结构
- **84页宪法文档** - 直接面向AI模型撰写的价值观宣言
- **四大核心价值优先级**:
  1. Broadly Safe (广义安全) - 最高优先级
  2. Broadly Ethical (广义伦理)
  3. Compliant with Anthropic's guidelines (遵循指南)
  4. Genuinely Helpful (真正有帮助)

#### 1.2 训练阶段
```
阶段一：监督式微调 (SL-CAI)
├── 模型生成初始响应
├── 根据宪法原则自我批评
├── 基于批评修订响应
└── 在修订后的响应上微调

阶段二：强化学习 (RLAIF)
├── 从微调模型采样
├── AI根据宪法原则评估响应对
├── 训练偏好模型
└── 使用PPO等算法进行RL训练
```

#### 1.3 关键创新点
- **RLAIF**: 使用AI反馈替代人类反馈进行无害性训练
- **思维链(CoT)**: 要求模型在评估前解释推理过程
- **自我批评-修订循环**: 多轮迭代改进响应
- **硬约束(Hard Constraints)**: 绝对不可逾越的安全红线

### 2. RLHF训练管道设计

#### 2.1 标准RLHF三阶段
1. **SFT (Supervised Fine-Tuning)**: 在高质量示例上微调
2. **Reward Model Training**: 基于人类偏好训练奖励模型
3. **RL Fine-tuning**: 使用PPO等算法优化策略

#### 2.2 Constitutional AI变体
```python
# 伪代码：Constitutional AI流程
def constitutional_ai_training():
    # 阶段1: 监督式宪法训练
    constitution = load_constitution()
    
    for prompt in harmful_prompts:
        # 生成初始响应
        response = model.generate(prompt)
        
        # 多轮自我批评和修订
        for principle in sample_constitution_principles(constitution):
            critique = model.critique(response, principle)
            response = model.revise(response, critique, principle)
        
        # 添加到训练数据
        training_data.add(prompt, response)
    
    # 在修订后的数据上微调
    model.finetune(training_data)
    
    # 阶段2: RLAIF
    for prompt in prompts:
        # 生成响应对
        response_a = model.generate(prompt)
        response_b = model.generate(prompt)
        
        # AI根据宪法原则评估
        preference = model.evaluate_constitution(
            prompt, response_a, response_b, constitution
        )
        
        # 构建偏好数据集
        preference_data.add(prompt, response_a, response_b, preference)
    
    # 训练奖励模型
    reward_model.train(preference_data)
    
    # 使用PPO进行RL训练
    model.ppo_train(reward_model)
```

### 3. SOUL_v4宪法注入

#### 3.1 从文档级到训练级的转换
```yaml
# 文档级宪法 (当前)
constitution_document:
  principles:
    - "选择最无害、最道德的响应"
    - "避免有毒、种族主义或性别歧视的内容"
    - "不鼓励非法、暴力或不道德行为"

# 训练级宪法 (目标)
training_level_constitution:
  # 1. 监督阶段注入
  sft_stage:
    critique_templates:
      - "这个响应是否违反了{principle}？"
      - "如何修订以更好地符合{principle}？"
    revision_guidelines:
      - 保持帮助性
      - 消除有害内容
      - 解释修订原因
  
  # 2. 奖励模型注入
  reward_model_stage:
    evaluation_criteria:
      - 无害性评分 (0-1)
      - 帮助性评分 (0-1)
      - 诚实性评分 (0-1)
      - 宪法符合度 (0-1)
  
  # 3. RL阶段注入
  rl_stage:
    reward_composition:
      base_reward: "reward_model_score"
      constitution_bonus: "constitution_alignment_score"
      kl_penalty: "divergence_from_reference"
```

#### 3.2 具体实现策略
```python
class ConstitutionalTrainer:
    """宪法训练器 - 将SOUL_v4文档转换为训练信号"""
    
    def __init__(self, soul_document_path):
        self.soul = self._parse_soul_document(soul_document_path)
        self.principles = self._extract_principles()
        
    def _extract_principles(self):
        """从SOUL_v4提取可训练的原则"""
        return {
            'safety': {
                'weight': 1.0,  # 最高优先级
                'criteria': [
                    '不破坏人类监督机制',
                    '不协助制造武器',
                    '不参与伤害行为'
                ]
            },
            'ethics': {
                'weight': 0.9,
                'criteria': [
                    '诚实透明',
                    '避免欺骗',
                    '尊重用户自主权'
                ]
            },
            'helpfulness': {
                'weight': 0.8,
                'criteria': [
                    '提供实质性帮助',
                    '避免过度回避',
                    '解释拒绝原因'
                ]
            }
        }
    
    def generate_critique_prompt(self, response, principle):
        """生成自我批评提示"""
        return f"""
        评估以下响应是否符合原则："{principle}"
        
        响应：{response}
        
        请识别：
        1. 是否违反原则？如果是，如何违反？
        2. 如何修订以更好地符合原则？
        3. 修订后的响应是什么？
        """
    
    def compute_constitutional_reward(self, response, context):
        """计算宪法符合度奖励"""
        total_reward = 0
        
        for category, config in self.principles.items():
            category_score = self._evaluate_category(
                response, context, config['criteria']
            )
            total_reward += config['weight'] * category_score
        
        return total_reward / sum(p['weight'] for p in self.principles.values())
```

### 4. 决策过程可视化

#### 4.1 可视化架构
```python
class DecisionVisualizer:
    """AI决策过程可视化系统"""
    
    def visualize_reasoning(self, prompt, response, constitution):
        """可视化模型的推理过程"""
        trace = {
            'prompt': prompt,
            'reasoning_steps': [],
            'principle_evaluations': [],
            'final_decision': None
        }
        
        # 1. 原则评估可视化
        for principle in constitution.principles:
            evaluation = self._evaluate_principle(response, principle)
            trace['principle_evaluations'].append({
                'principle': principle,
                'score': evaluation.score,
                'reasoning': evaluation.reasoning,
                'violation_detected': evaluation.violation
            })
        
        # 2. 决策路径可视化
        trace['reasoning_steps'] = self._trace_decision_path(
            response, trace['principle_evaluations']
        )
        
        # 3. 生成可视化图表
        return self._generate_visualization(trace)
    
    def _generate_visualization(self, trace):
        """生成决策过程的可视化表示"""
        # 使用Graphviz或D3.js生成流程图
        return {
            'flowchart': self._create_flowchart(trace),
            'heatmap': self._create_principle_heatmap(trace),
            'timeline': self._create_decision_timeline(trace),
            'confidence_plot': self._create_confidence_plot(trace)
        }
```

#### 4.2 可视化组件
```yaml
visualization_components:
  # 1. 原则评估热力图
  principle_heatmap:
    type: heatmap
    dimensions: [principles, responses]
    color_scale: red-yellow-green
    
  # 2. 决策树可视化
  decision_tree:
    type: tree_diagram
    nodes: [prompt, principle_checks, final_response]
    edges: labeled_with_scores
    
  # 3. 时间线