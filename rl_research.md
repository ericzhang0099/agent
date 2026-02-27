# 强化学习 (Reinforcement Learning) - Agent决策优化研究

## 研究概述

本文档深入探讨强化学习中的Agent决策优化，重点研究PPO/RLHF算法原理、奖励模型训练、策略优化以及探索与利用的平衡问题。

---

## 1. PPO/RLHF算法原理

### 1.1 PPO (Proximal Policy Optimization)

PPO由OpenAI于2017年提出，是目前最流行的策略梯度算法之一。

#### 核心思想
通过限制策略更新的幅度来解决传统策略梯度方法的不稳定性问题。

#### 关键公式

**策略比率 (Policy Ratio)**
```
r_t(θ) = π_θ(a_t|s_t) / π_θ_old(a_t|s_t)
```

**裁剪目标函数 (Clipped Surrogate Objective)**
```
L^CLIP(θ) = Ê_t[min(r_t(θ)Â_t, clip(r_t(θ), 1-ε, 1+ε)Â_t)]
```

其中：
- `Â_t`: 优势函数估计
- `ε`: 裁剪参数 (通常0.1-0.2)

**完整PPO目标函数**
```
L^PPO(θ) = L^CLIP(θ) - c_1 * L^VF(θ) + c_2 * S[π_θ](s_t)
```

#### PPO优势
1. **样本效率**: 可重复使用收集的数据进行多轮更新
2. **训练稳定性**: 裁剪机制防止策略剧烈变化
3. **实现简单**: 相比TRPO更容易实现
4. **泛化性好**: 在多种任务上表现稳定

### 1.2 RLHF (Reinforcement Learning from Human Feedback)

RLHF是将人类反馈整合到强化学习中的方法，被广泛用于大语言模型对齐。

#### 三阶段流程

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  阶段1: SFT  │ -> │  阶段2: RM  │ -> │  阶段3: RL  │
│  监督微调    │    │  奖励模型   │    │  策略优化   │
└─────────────┘    └─────────────┘    └─────────────┘
```

**阶段1: 监督微调 (SFT)**
- 使用高质量指令-响应对训练基础模型
- 建立初始策略 π^SFT

**阶段2: 奖励模型训练**
- 收集人类偏好数据 (成对比较)
- 使用Bradley-Terry模型训练奖励模型

偏好概率公式：
```
P(y_0 ≻ y_1|x) = 1 / (1 + exp(R(x,y_1) - R(x,y_0)))
```

**阶段3: RL优化**
- 使用PPO优化策略以最大化奖励模型评分
- 加入KL散度惩罚防止策略偏离太远

PPO+RLHF目标函数：
```
R^PPO(x,y) = R_φ(x,y) - β * log[π_θ(y|x) / π^SFT(y|x)]
```

#### Iterated RLHF (迭代RLHF)

最新研究表明，迭代RLHF可以显著减少奖励模型过优化问题：

1. **多轮迭代**: 重复收集偏好数据、训练奖励模型、优化策略
2. **数据聚合**: 跨迭代连接偏好数据提高奖励模型鲁棒性
3. **策略初始化**: 每轮从SFT模型重新初始化策略最稳健

---

## 2. 奖励模型训练

### 2.1 奖励模型架构

奖励模型通常基于预训练语言模型，添加回归头输出标量奖励值。

```python
class RewardModel(nn.Module):
    def __init__(self, base_model, hidden_size):
        super().__init__()
        self.base = base_model
        self.reward_head = nn.Linear(hidden_size, 1)
    
    def forward(self, input_ids, attention_mask):
        hidden_states = self.base(input_ids, attention_mask).last_hidden_state
        rewards = self.reward_head(hidden_states[:, -1, :])
        return rewards
```

### 2.2 偏好学习

#### Bradley-Terry模型
用于建模成对偏好比较的概率分布。

损失函数：
```
L_RM = -E[(x,y_w,y_l)~D][log σ(R_φ(x,y_w) - R_φ(x,y_l))]
```

其中：
- `y_w`: 人类偏好的响应 (win)
- `y_l`: 人类不喜欢的响应 (loss)
- `σ`: sigmoid函数

### 2.3 奖励模型过优化问题

**问题描述**: 策略可能过度拟合到奖励模型的缺陷上，产生高奖励但低质量的输出。

**缓解策略**:
1. **奖励模型集成**: 使用多个奖励模型取平均
2. **保守优化**: 使用最坏情况优化 (WCO)
3. **权重平均**: 对多个奖励模型参数进行平均
4. **KL惩罚**: 限制策略与参考策略的偏离

---

## 3. 策略优化

### 3.1 策略梯度方法

#### REINFORCE算法
```
∇_θ J(θ) = E[∇_θ log π_θ(a|s) * R]
```

#### Actor-Critic架构
- **Actor**: 策略网络，决定动作
- **Critic**: 价值网络，评估状态价值

### 3.2 优势函数估计

**GAE (Generalized Advantage Estimation)**:
```
Â_t = Σ(γλ)^l * δ_{t+l}

其中: δ_t = r_t + γV(s_{t+1}) - V(s_t)
```

参数：
- `γ`: 折扣因子 (通常0.99)
- `λ`: GAE参数 (通常0.95)

### 3.3 PPO训练技巧

1. **多轮更新**: 对同一批数据进行多次epoch更新
2. **小批量处理**: 使用mini-batch进行梯度下降
3. **梯度裁剪**: 防止梯度爆炸
4. **学习率退火**: 使用余弦退火调度
5. **Early Stopping**: 当KL散度超过阈值时停止更新

---

## 4. 探索vs利用平衡

### 4.1 核心概念

**探索 (Exploration)**: 尝试新的、未知的行为，以发现潜在的更优策略。

**利用 (Exploitation)**: 使用已知的最佳策略，获取当前已知的最大回报。

**平衡挑战**: 
- 过度探索 → 浪费资源在低回报动作上
- 过度利用 → 可能错过更优策略

### 4.2 探索策略

#### ε-贪心 (Epsilon-Greedy)
```python
if random() < ε:
    action = random_action()  # 探索
else:
    action = best_action()    # 利用
```

ε通常随时间衰减：
```
ε = ε_min + (ε_max - ε_min) * exp(-decay_rate * step)
```

#### 玻尔兹曼探索 (Boltzmann Exploration)
```
P(a|s) = exp(Q(s,a)/τ) / Σ exp(Q(s,a')/τ)
```
温度参数τ控制探索程度。

#### 上置信界 (UCB)
```
A_t = argmax_a [Q(a) + c * sqrt(ln(t) / N(a))]
```

#### 熵正则化
在PPO中通过最大化策略熵鼓励探索：
```
S[π_θ](s) = -Σ π_θ(a|s) * log π_θ(a|s)
```

### 4.3 自适应探索

**Curiosity-driven Exploration**:
使用内在奖励鼓励探索新颖状态：
```
r_intrinsic = ||φ(s_{t+1}) - φ(s_t)||^2
```

**计数式探索**:
基于状态访问次数给予奖励：
```
r_bonus = 1 / sqrt(N(s))
```

---

## 5. 代码实现结构

### 核心类说明

| 类名 | 功能 |
|------|------|
| `PPOConfig` | PPO算法配置参数 |
| `RLHFConfig` | RLHF配置参数 |
| `ReplayBuffer` | 经验回放缓冲区 |
| `RolloutBuffer` | PPO专用Rollout缓冲区 |
| `ActorNetwork` | Actor网络 (策略) |
| `CriticNetwork` | Critic网络 (价值) |
| `RewardModel` | 奖励模型 (RLHF) |
| `PPO` | PPO算法实现 |
| `RLAgent` | 强化学习Agent核心类 |

### 快速开始

```python
from rl_agent import RLAgent, PPOConfig

# 配置
config = PPOConfig(
    state_dim=128,
    action_dim=4,
    hidden_dim=256,
    lr=3e-4,
    gamma=0.99,
    epsilon=0.2
)

# 创建Agent
agent = RLAgent(config)

# 训练
for episode in range(1000):
    state = env.reset()
    done = False
    
    while not done:
        action = agent.select_action(state)
        next_state, reward, done, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, done)
        state = next_state
    
    agent.update()
```

---

## 6. 关键研究发现

### 6.1 Iterated RLHF效果

根据最新研究 (Wolf et al., 2025):

1. **过优化减少**: 随着迭代次数增加，奖励模型过优化问题逐渐减少
2. **性能提升**: 奖励模型越来越接近真实人类偏好
3. **收益递减**: 3次迭代后性能提升开始趋于平缓
4. **数据聚合最有效**: 跨迭代连接偏好数据比采样策略效果更好

### 6.2 PPO vs DPO

| 特性 | PPO | DPO |
|------|-----|-----|
| 需要奖励模型 | 是 | 否 |
| 在线学习 | 是 | 否 |
| 实现复杂度 | 高 | 低 |
| 性能上限 | 更高 | 较低 |
| 训练稳定性 | 需调参 | 更稳定 |

---

## 参考文献

1. Schulman et al. "Proximal Policy Optimization Algorithms" (2017)
2. Ziegler et al. "Fine-Tuning Language Models from Human Preferences" (2020)
3. Ouyang et al. "Training Language Models to Follow Instructions with Human Feedback" (2022)
4. Gao et al. "Scaling Laws for Reward Model Overoptimization" (2023)
5. Coste et al. "Reward Model Ensembles Help Mitigate Overoptimization" (2024)
6. Wolf et al. "Reward Model Overoptimisation in Iterated RLHF" (2025)

---

*文档生成时间: 2026-02-27*
