"""
强化学习 Agent - 决策优化核心实现
包含: PPO算法简化版、奖励模型接口、训练循环示例

作者: AI Research Agent
日期: 2026-02-27
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical
import numpy as np
from typing import List, Tuple, Dict, Optional, NamedTuple
from dataclasses import dataclass
from collections import deque
import random


# ==================== 配置类 ====================

@dataclass
class PPOConfig:
    """PPO算法配置参数"""
    state_dim: int = 128          # 状态空间维度
    action_dim: int = 4           # 动作空间维度
    hidden_dim: int = 256         # 隐藏层维度
    
    # 训练参数
    lr: float = 3e-4              # 学习率
    gamma: float = 0.99           # 折扣因子
    gae_lambda: float = 0.95      # GAE参数
    epsilon: float = 0.2          # PPO裁剪参数
    
    # 更新参数
    epochs: int = 4               # 每次数据更新轮数
    batch_size: int = 64          # 批量大小
    buffer_size: int = 2048       # 经验缓冲区大小
    
    # 损失系数
    value_coef: float = 0.5       # 价值函数损失系数
    entropy_coef: float = 0.01    # 熵奖励系数
    max_grad_norm: float = 0.5    # 梯度裁剪阈值
    
    # 探索参数
    initial_eps: float = 1.0      # 初始epsilon
    final_eps: float = 0.05       # 最终epsilon
    eps_decay: float = 0.995      # epsilon衰减率
    
    # 设备
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class RLHFConfig:
    """RLHF配置参数"""
    # 奖励模型参数
    reward_model_path: Optional[str] = None
    reward_coef: float = 1.0      # 奖励系数
    kl_coef: float = 0.01         # KL散度惩罚系数
    
    # 参考策略 (SFT模型)
    reference_policy_path: Optional[str] = None


# ==================== 经验回放缓冲区 ====================

class Experience(NamedTuple):
    """单条经验数据"""
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    log_prob: float
    value: float


class ReplayBuffer:
    """经验回放缓冲区 - 用于存储和采样训练数据"""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.buffer: List[Experience] = []
        self.position = 0
    
    def push(self, exp: Experience):
        """添加经验到缓冲区"""
        if len(self.buffer) < self.capacity:
            self.buffer.append(exp)
        else:
            self.buffer[self.position] = exp
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size: int) -> List[Experience]:
        """随机采样一批经验"""
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))
    
    def get_all(self) -> List[Experience]:
        """获取所有经验 (用于PPO的on-policy更新)"""
        return self.buffer.copy()
    
    def clear(self):
        """清空缓冲区"""
        self.buffer.clear()
        self.position = 0
    
    def __len__(self):
        return len(self.buffer)


class RolloutBuffer:
    """PPO专用Rollout缓冲区 - 存储轨迹数据"""
    
    def __init__(self):
        self.states = []
        self.actions = []
        self.rewards = []
        self.next_states = []
        self.dones = []
        self.log_probs = []
        self.values = []
        self.advantages = []
        self.returns = []
    
    def add(self, state, action, reward, next_state, done, log_prob, value):
        """添加单步数据"""
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.next_states.append(next_state)
        self.dones.append(done)
        self.log_probs.append(log_prob)
        self.values.append(value)
    
    def compute_advantages(self, gamma: float, gae_lambda: float, next_value: float = 0):
        """使用GAE计算优势函数"""
        advantages = []
        gae = 0
        
        # 从后向前计算
        for t in reversed(range(len(self.rewards))):
            if t == len(self.rewards) - 1:
                next_val = next_value
            else:
                next_val = self.values[t + 1]
            
            delta = self.rewards[t] + gamma * next_val * (1 - self.dones[t]) - self.values[t]
            gae = delta + gamma * gae_lambda * (1 - self.dones[t]) * gae
            advantages.insert(0, gae)
        
        self.advantages = advantages
        # 计算回报 (returns = advantages + values)
        self.returns = [adv + val for adv, val in zip(advantages, self.values)]
        
        # 归一化优势
        adv_mean = np.mean(advantages)
        adv_std = np.std(advantages) + 1e-8
        self.advantages = [(adv - adv_mean) / adv_std for adv in advantages]
    
    def get_batch(self, batch_size: int, shuffle: bool = True):
        """生成批次数据"""
        indices = list(range(len(self.states)))
        if shuffle:
            random.shuffle(indices)
        
        for start in range(0, len(indices), batch_size):
            end = min(start + batch_size, len(indices))
            batch_indices = indices[start:end]
            
            yield {
                'states': np.array([self.states[i] for i in batch_indices]),
                'actions': np.array([self.actions[i] for i in batch_indices]),
                'old_log_probs': np.array([self.log_probs[i] for i in batch_indices]),
                'advantages': np.array([self.advantages[i] for i in batch_indices]),
                'returns': np.array([self.returns[i] for i in batch_indices]),
            }
    
    def clear(self):
        """清空缓冲区"""
        self.__init__()
    
    def __len__(self):
        return len(self.states)


# ==================== 神经网络模型 ====================

class ActorNetwork(nn.Module):
    """Actor网络 - 输出动作概率分布"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        logits = self.fc3(x)
        return logits
    
    def get_action(self, state, deterministic=False):
        """采样动作"""
        logits = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = Categorical(probs)
        
        if deterministic:
            action = probs.argmax(dim=-1)
        else:
            action = dist.sample()
        
        log_prob = dist.log_prob(action)
        entropy = dist.entropy()
        
        return action, log_prob, entropy, probs


class CriticNetwork(nn.Module):
    """Critic网络 - 估计状态价值函数"""
    
    def __init__(self, state_dim: int, hidden_dim: int = 256):
        super().__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)
        
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value.squeeze(-1)


class RewardModel(nn.Module):
    """奖励模型 - 用于RLHF"""
    
    def __init__(self, state_dim: int, action_dim: int, hidden_dim: int = 256):
        super().__init__()
        # 将状态和动作编码
        self.state_encoder = nn.Linear(state_dim, hidden_dim // 2)
        self.action_encoder = nn.Linear(action_dim, hidden_dim // 2)
        
        self.fc1 = nn.Linear(hidden_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.reward_head = nn.Linear(hidden_dim // 2, 1)
        
    def forward(self, state, action=None):
        """
        前向传播
        state: [batch, state_dim]
        action: [batch, action_dim] 或 [batch] (离散动作索引)
        """
        state_feat = F.relu(self.state_encoder(state))
        
        if action is not None:
            if len(action.shape) == 1:  # 离散动作索引
                # 转换为one-hot
                action_onehot = F.one_hot(action.long(), num_classes=4).float()
            else:
                action_onehot = action
            action_feat = F.relu(self.action_encoder(action_onehot))
            x = torch.cat([state_feat, action_feat], dim=-1)
        else:
            x = torch.cat([state_feat, state_feat], dim=-1)
        
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        reward = self.reward_head(x)
        return reward.squeeze(-1)


# ==================== PPO算法实现 ====================

class PPO:
    """PPO (Proximal Policy Optimization) 算法实现"""
    
    def __init__(self, config: PPOConfig):
        self.config = config
        self.device = torch.device(config.device)
        
        # 创建网络
        self.actor = ActorNetwork(
            config.state_dim, 
            config.action_dim, 
            config.hidden_dim
        ).to(self.device)
        
        self.critic = CriticNetwork(
            config.state_dim, 
            config.hidden_dim
        ).to(self.device)
        
        # 优化器
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=config.lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=config.lr)
        
        # 学习率调度器
        self.actor_scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.actor_optimizer, T_max=1000, eta_min=config.lr * 0.1
        )
        
        # Rollout缓冲区
        self.rollout_buffer = RolloutBuffer()
        
        # 训练步数
        self.train_step = 0
        
    def select_action(self, state: np.ndarray, deterministic: bool = False) -> Tuple[int, float, float]:
        """
        选择动作
        
        Returns:
            action: 动作索引
            log_prob: 动作的对数概率
            value: 状态价值估计
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, log_prob, _, _ = self.actor.get_action(state_tensor, deterministic)
            value = self.critic(state_tensor)
            
        return action.item(), log_prob.item(), value.item()
    
    def store_transition(self, state, action, reward, next_state, done, log_prob, value):
        """存储转移数据"""
        self.rollout_buffer.add(state, action, reward, next_state, done, log_prob, value)
    
    def compute_gae(self, next_state: np.ndarray, gamma: float = 0.99, gae_lambda: float = 0.95):
        """计算GAE优势估计"""
        with torch.no_grad():
            next_state_tensor = torch.FloatTensor(next_state).unsqueeze(0).to(self.device)
            next_value = self.critic(next_state_tensor).item()
        
        self.rollout_buffer.compute_advantages(gamma, gae_lambda, next_value)
    
    def update(self) -> Dict[str, float]:
        """
        更新策略和价值网络
        
        Returns:
            训练统计信息
        """
        if len(self.rollout_buffer) < self.config.batch_size:
            return {}
        
        # 转换数据为tensor
        states = torch.FloatTensor(np.array(self.rollout_buffer.states)).to(self.device)
        actions = torch.LongTensor(self.rollout_buffer.actions).to(self.device)
        old_log_probs = torch.FloatTensor(self.rollout_buffer.log_probs).to(self.device)
        advantages = torch.FloatTensor(self.rollout_buffer.advantages).to(self.device)
        returns = torch.FloatTensor(self.rollout_buffer.returns).to(self.device)
        
        # 训练统计
        total_actor_loss = 0
        total_critic_loss = 0
        total_entropy = 0
        total_approx_kl = 0
        update_count = 0
        
        # 多轮更新
        for epoch in range(self.config.epochs):
            # 生成批次
            for batch in self.rollout_buffer.get_batch(self.config.batch_size):
                batch_states = torch.FloatTensor(batch['states']).to(self.device)
                batch_actions = torch.LongTensor(batch['actions']).to(self.device)
                batch_old_log_probs = torch.FloatTensor(batch['old_log_probs']).to(self.device)
                batch_advantages = torch.FloatTensor(batch['advantages']).to(self.device)
                batch_returns = torch.FloatTensor(batch['returns']).to(self.device)
                
                # 计算新的动作概率
                _, new_log_probs, entropy, _ = self.actor.get_action(batch_states)
                new_log_probs = new_log_probs.gather(0, batch_actions)
                
                # 计算比率
                ratio = torch.exp(new_log_probs - batch_old_log_probs)
                
                # 裁剪目标
                surr1 = ratio * batch_advantages
                surr2 = torch.clamp(ratio, 1 - self.config.epsilon, 1 + self.config.epsilon) * batch_advantages
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值函数损失
                values = self.critic(batch_states)
                critic_loss = F.mse_loss(values, batch_returns)
                
                # 总损失
                loss = (
                    actor_loss 
                    + self.config.value_coef * critic_loss 
                    - self.config.entropy_coef * entropy.mean()
                )
                
                # 梯度下降
                self.actor_optimizer.zero_grad()
                self.critic_optimizer.zero_grad()
                loss.backward()
                
                # 梯度裁剪
                nn.utils.clip_grad_norm_(self.actor.parameters(), self.config.max_grad_norm)
                nn.utils.clip_grad_norm_(self.critic.parameters(), self.config.max_grad_norm)
                
                self.actor_optimizer.step()
                self.critic_optimizer.step()
                
                # 统计
                with torch.no_grad():
                    approx_kl = ((ratio - 1) - torch.log(ratio)).mean().item()
                
                total_actor_loss += actor_loss.item()
                total_critic_loss += critic_loss.item()
                total_entropy += entropy.mean().item()
                total_approx_kl += approx_kl
                update_count += 1
                
                # Early stopping - 如果KL散度过大
                if approx_kl > 0.015:
                    break
        
        # 更新学习率
        self.actor_scheduler.step()
        
        # 清空缓冲区
        self.rollout_buffer.clear()
        
        self.train_step += 1
        
        # 返回统计信息
        if update_count > 0:
            return {
                'actor_loss': total_actor_loss / update_count,
                'critic_loss': total_critic_loss / update_count,
                'entropy': total_entropy / update_count,
                'approx_kl': total_approx_kl / update_count,
            }
        return {}
    
    def save(self, path: str):
        """保存模型"""
        torch.save({
            'actor': self.actor.state_dict(),
            'critic': self.critic.state_dict(),
            'actor_optimizer': self.actor_optimizer.state_dict(),
            'critic_optimizer': self.critic_optimizer.state_dict(),
        }, path)
    
    def load(self, path: str):
        """加载模型"""
        checkpoint = torch.load(path, map_location=self.device)
        self.actor.load_state_dict(checkpoint['actor'])
        self.critic.load_state_dict(checkpoint['critic'])
        self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer'])
        self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer'])


# ==================== RL Agent核心类 ====================

class RLAgent:
    """
    强化学习Agent核心类
    整合PPO算法、奖励模型、探索策略
    """
    
    def __init__(
        self, 
        ppo_config: PPOConfig,
        rlhf_config: Optional[RLHFConfig] = None
    ):
        self.ppo_config = ppo_config
        self.rlhf_config = rlhf_config or RLHFConfig()
        self.device = torch.device(ppo_config.device)
        
        # PPO算法
        self.ppo = PPO(ppo_config)
        
        # 奖励模型 (用于RLHF)
        self.reward_model: Optional[RewardModel] = None
        if self.rlhf_config.reward_model_path:
            self.load_reward_model(self.rlhf_config.reward_model_path)
        else:
            # 初始化默认奖励模型
            self.reward_model = RewardModel(
                ppo_config.state_dim,
                ppo_config.action_dim,
                ppo_config.hidden_dim
            ).to(self.device)
        
        # 参考策略 (用于KL惩罚)
        self.reference_policy: Optional[ActorNetwork] = None
        if self.rlhf_config.reference_policy_path:
            self.load_reference_policy(self.rlhf_config.reference_policy_path)
        
        # 探索参数
        self.epsilon = ppo_config.initial_eps
        
        # 训练统计
        self.episode_count = 0
        self.total_steps = 0
        self.episode_rewards = deque(maxlen=100)
        
    def load_reward_model(self, path: str):
        """加载预训练奖励模型"""
        self.reward_model = RewardModel(
            self.ppo_config.state_dim,
            self.ppo_config.action_dim,
            self.ppo_config.hidden_dim
        ).to(self.device)
        self.reward_model.load_state_dict(torch.load(path, map_location=self.device))
        self.reward_model.eval()
        
    def load_reference_policy(self, path: str):
        """加载参考策略 (SFT模型)"""
        self.reference_policy = ActorNetwork(
            self.ppo_config.state_dim,
            self.ppo_config.action_dim,
            self.ppo_config.hidden_dim
        ).to(self.device)
        self.reference_policy.load_state_dict(torch.load(path, map_location=self.device))
        self.reference_policy.eval()
    
    def select_action(self, state: np.ndarray, use_exploration: bool = True) -> Tuple[int, Dict]:
        """
        选择动作 - 支持探索vs利用平衡
        
        Args:
            state: 当前状态
            use_exploration: 是否使用探索策略
            
        Returns:
            action: 选择的动作
            info: 额外信息
        """
        info = {}
        
        # ε-贪心探索
        if use_exploration and random.random() < self.epsilon:
            action = random.randint(0, self.ppo_config.action_dim - 1)
            log_prob = np.log(1.0 / self.ppo_config.action_dim)
            value = 0.0
            info['exploration'] = True
        else:
            action, log_prob, value = self.ppo.select_action(state, deterministic=False)
            info['exploration'] = False
        
        info['log_prob'] = log_prob
        info['value'] = value
        info['epsilon'] = self.epsilon
        
        return action, info
    
    def compute_reward(
        self, 
        state: np.ndarray, 
        action: int, 
        env_reward: float,
        next_state: Optional[np.ndarray] = None
    ) -> Tuple[float, Dict]:
        """
        计算奖励 - 支持RLHF奖励模型和KL惩罚
        
        Args:
            state: 当前状态
            action: 执行的动作
            env_reward: 环境原始奖励
            next_state: 下一个状态
            
        Returns:
            total_reward: 总奖励
            reward_info: 奖励分解信息
        """
        reward_info = {
            'env_reward': env_reward,
            'model_reward': 0.0,
            'kl_penalty': 0.0,
        }
        
        total_reward = env_reward
        
        # 奖励模型评分 (RLHF)
        if self.reward_model is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                action_tensor = torch.LongTensor([action]).to(self.device)
                model_reward = self.reward_model(state_tensor, action_tensor).item()
                
            reward_info['model_reward'] = model_reward
            total_reward += self.rlhf_config.reward_coef * model_reward
        
        # KL散度惩罚 (防止策略偏离参考策略太远)
        if self.reference_policy is not None:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
                
                # 当前策略的动作概率
                _, _, _, current_probs = self.ppo.actor.get_action(state_tensor)
                
                # 参考策略的动作概率
                _, _, _, ref_probs = self.reference_policy.get_action(state_tensor)
                
                # 计算KL散度
                kl_div = (current_probs * (torch.log(current_probs + 1e-10) - torch.log(ref_probs + 1e-10))).sum()
                kl_penalty = kl_div.item()
                
            reward_info['kl_penalty'] = -self.rlhf_config.kl_coef * kl_penalty
            total_reward += reward_info['kl_penalty']
        
        reward_info['total_reward'] = total_reward
        return total_reward, reward_info
    
    def store_transition(
        self, 
        state: np.ndarray, 
        action: int, 
        reward: float, 
        next_state: np.ndarray, 
        done: bool,
        info: Optional[Dict] = None
    ):
        """存储转移数据"""
        info = info or {}
        log_prob = info.get('log_prob', 0.0)
        value = info.get('value', 0.0)
        
        self.ppo.store_transition(state, action, reward, next_state, done, log_prob, value)
        self.total_steps += 1
    
    def update(self, next_state: Optional[np.ndarray] = None) -> Dict[str, float]:
        """
        更新Agent
        
        Args:
            next_state: 下一个状态 (用于GAE计算)
            
        Returns:
            训练统计信息
        """
        # 计算GAE
        if next_state is not None:
            self.ppo.compute_gae(next_state, self.ppo_config.gamma, self.ppo_config.gae_lambda)
        
        # 更新PPO
        update_info = self.ppo.update()
        
        # 衰减探索率
        self.epsilon = max(
            self.ppo_config.final_eps,
            self.epsilon * self.ppo_config.eps_decay
        )
        
        return update_info
    
    def train_episode(self, env, max_steps: int = 1000) -> Dict[str, float]:
        """
        训练一个完整的episode
        
        Args:
            env: 环境对象 (需实现reset和step接口)
            max_steps: 最大步数
            
        Returns:
            episode统计信息
        """
        state = env.reset()
        episode_reward = 0
        episode_steps = 0
        
        for step in range(max_steps):
            # 选择动作
            action, info = self.select_action(state, use_exploration=True)
            
            # 执行动作
            next_state, env_reward, done, _ = env.step(action)
            
            # 计算奖励 (支持RLHF)
            total_reward, reward_info = self.compute_reward(state, action, env_reward, next_state)
            
            # 存储转移
            self.store_transition(state, action, total_reward, next_state, done, info)
            
            episode_reward += env_reward
            episode_steps += 1
            
            state = next_state
            
            # 检查是否需要更新
            if len(self.ppo.rollout_buffer) >= self.ppo_config.buffer_size:
                self.update(next_state if not done else None)
            
            if done:
                break
        
        # 如果还有剩余数据，进行更新
        if len(self.ppo.rollout_buffer) > 0:
            self.update(state)
        
        self.episode_count += 1
        self.episode_rewards.append(episode_reward)
        
        return {
            'episode': self.episode_count,
            'reward': episode_reward,
            'steps': episode_steps,
            'avg_reward': np.mean(self.episode_rewards),
            'epsilon': self.epsilon,
        }
    
    def evaluate(self, env, num_episodes: int = 10) -> Dict[str, float]:
        """
        评估Agent性能
        
        Args:
            env: 环境对象
            num_episodes: 评估episode数
            
        Returns:
            评估统计信息
        """
        eval_rewards = []
        
        for _ in range(num_episodes):
            state = env.reset()
            episode_reward = 0
            done = False
            
            while not done:
                action, _ = self.select_action(state, use_exploration=False)
                next_state, reward, done, _ = env.step(action)
                episode_reward += reward
                state = next_state
            
            eval_rewards.append(episode_reward)
        
        return {
            'mean_reward': np.mean(eval_rewards),
            'std_reward': np.std(eval_rewards),
            'min_reward': np.min(eval_rewards),
            'max_reward': np.max(eval_rewards),
        }
    
    def save(self, path: str):
        """保存Agent"""
        self.ppo.save(f"{path}_ppo.pt")
        if self.reward_model is not None:
            torch.save(self.reward_model.state_dict(), f"{path}_reward.pt")
    
    def load(self, path: str):
        """加载Agent"""
        self.ppo.load(f"{path}_ppo.pt")
        if self.reward_model is not None:
            self.reward_model.load_state_dict(torch.load(f"{path}_reward.pt", map_location=self.device))


# ==================== 训练循环示例 ====================

def train_rl_agent(
    env,
    num_episodes: int = 1000,
    eval_interval: int = 100,
    save_interval: int = 500,
    log_interval: int = 10,
):
    """
    完整的RL Agent训练流程
    
    Args:
        env: 训练环境 (需实现reset, step, observation_space, action_space接口)
        num_episodes: 总训练episode数
        eval_interval: 评估间隔
        save_interval: 保存间隔
        log_interval: 日志打印间隔
    """
    # 获取环境维度
    if hasattr(env, 'observation_space'):
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n
    else:
        # 默认维度
        state_dim = 128
        action_dim = 4
    
    # 配置
    ppo_config = PPOConfig(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dim=256,
        lr=3e-4,
        gamma=0.99,
        gae_lambda=0.95,
        epsilon=0.2,
        epochs=4,
        batch_size=64,
        buffer_size=2048,
        value_coef=0.5,
        entropy_coef=0.01,
        initial_eps=1.0,
        final_eps=0.05,
        eps_decay=0.995,
    )
    
    rlhf_config = RLHFConfig(
        reward_coef=1.0,
        kl_coef=0.01,
    )
    
    # 创建Agent
    agent = RLAgent(ppo_config, rlhf_config)
    
    print("=" * 60)
    print("开始训练 RL Agent")
    print(f"State dim: {state_dim}, Action dim: {action_dim}")
    print(f"Device: {ppo_config.device}")
    print("=" * 60)
    
    # 训练循环
    for episode in range(1, num_episodes + 1):
        # 训练一个episode
        episode_info = agent.train_episode(env)
        
        # 日志
        if episode % log_interval == 0:
            print(f"Episode {episode:4d} | "
                  f"Reward: {episode_info['reward']:8.2f} | "
                  f"Avg Reward: {episode_info['avg_reward']:8.2f} | "
                  f"Steps: {episode_info['steps']:4d} | "
                  f"Epsilon: {episode_info['epsilon']:.3f}")
        
        # 评估
        if episode % eval_interval == 0:
            eval_info = agent.evaluate(env, num_episodes=10)
            print(f"\n{'='*60}")
            print(f"评估结果 (Episode {episode}):")
            print(f"  Mean Reward: {eval_info['mean_reward']:.2f} ± {eval_info['std_reward']:.2f}")
            print(f"  Min/Max: {eval_info['min_reward']:.2f} / {eval_info['max_reward']:.2f}")
            print(f"{'='*60}\n")
        
        # 保存
        if episode % save_interval == 0:
            agent.save(f"checkpoint_episode_{episode}")
            print(f"模型已保存: checkpoint_episode_{episode}")
    
    # 最终保存
    agent.save("final_model")
    print("\n训练完成！最终模型已保存")
    
    return agent


# ==================== 模拟环境示例 ====================

class SimpleGridEnv:
    """简单的网格世界环境 - 用于测试"""
    
    def __init__(self, size: int = 5):
        self.size = size
        self.observation_space = type('obj', (object,), {
            'shape': (size * size + 2,)
        })()
        self.action_space = type('obj', (object,), {
            'n': 4
        })()
        
        self.reset()
    
    def reset(self):
        self.agent_pos = [0, 0]
        self.goal_pos = [self.size - 1, self.size - 1]
        self.steps = 0
        self.max_steps = 50
        return self._get_state()
    
    def _get_state(self):
        # 将位置编码为one-hot向量
        state = np.zeros(self.size * self.size + 2, dtype=np.float32)
        idx = self.agent_pos[0] * self.size + self.agent_pos[1]
        state[idx] = 1.0
        state[-2] = self.agent_pos[0] / self.size
        state[-1] = self.agent_pos[1] / self.size
        return state
    
    def step(self, action):
        self.steps += 1
        
        # 执行动作: 0=上, 1=下, 2=左, 3=右
        if action == 0 and self.agent_pos[0] > 0:
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[0] < self.size - 1:
            self.agent_pos[0] += 1
        elif action == 2 and self.agent_pos[1] > 0:
            self.agent_pos[1] -= 1
        elif action == 3 and self.agent_pos[1] < self.size - 1:
            self.agent_pos[1] += 1
        
        # 计算奖励
        if self.agent_pos == self.goal_pos:
            reward = 10.0
            done = True
        else:
            # 距离奖励
            dist = abs(self.agent_pos[0] - self.goal_pos[0]) + abs(self.agent_pos[1] - self.goal_pos[1])
            reward = -0.1 - 0.01 * dist
            done = self.steps >= self.max_steps
        
        return self._get_state(), reward, done, {}


# ==================== 主函数 ====================

if __name__ == "__main__":
    # 创建模拟环境
    env = SimpleGridEnv(size=5)
    
    # 训练Agent
    agent = train_rl_agent(
        env=env,
        num_episodes=500,
        eval_interval=100,
        save_interval=250,
        log_interval=20,
    )
    
    # 最终评估
    print("\n" + "=" * 60)
    print("最终评估:")
    final_eval = agent.evaluate(env, num_episodes=20)
    print(f"  Mean Reward: {final_eval['mean_reward']:.2f}")
    print(f"  Std Reward: {final_eval['std_reward']:.2f}")
    print("=" * 60)
