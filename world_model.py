"""
World Models - 世界模型核心实现
包含: 状态预测网络、RSSM世界模型、动作规划模块、环境交互示例

参考:
- Dreamer: https://github.com/danijar/dreamer
- RSSM: Learning Latent Dynamics for Planning from Pixels
- JEPA: Joint Embedding Predictive Architecture
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as D
from typing import Tuple, Dict, List, Optional, NamedTuple
import numpy as np


# ============================================================================
# 1. 基础组件
# ============================================================================

class ConvEncoder(nn.Module):
    """图像观测编码器 - 将图像编码为特征向量"""
    
    def __init__(self, input_channels: int = 3, depth: int = 32, 
                 activation: nn.Module = nn.ELU):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Conv2d(input_channels, 1*depth, 4, stride=2),
            activation(),
            nn.Conv2d(1*depth, 2*depth, 4, stride=2),
            activation(),
            nn.Conv2d(2*depth, 4*depth, 4, stride=2),
            activation(),
            nn.Conv2d(4*depth, 8*depth, 4, stride=2),
            activation(),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, channels, height, width)
        Returns:
            features: (batch, features)
        """
        x = self.layers(x)
        # Flatten spatial dimensions
        return x.reshape(x.shape[0], -1)


class ConvDecoder(nn.Module):
    """图像解码器 - 从潜在状态重建图像"""
    
    def __init__(self, feature_size: int, output_channels: int = 3,
                 depth: int = 32, activation: nn.Module = nn.ELU,
                 output_size: Tuple[int, int] = (64, 64)):
        super().__init__()
        self.output_size = output_size
        
        # 计算初始特征图大小
        h, w = output_size
        for _ in range(4):
            h = (h - 4) // 2 + 1
            w = (w - 4) // 2 + 1
        
        self.fc = nn.Linear(feature_size, h * w * 8 * depth)
        self.h, self.w = h, w
        self.depth = depth
        
        self.layers = nn.Sequential(
            nn.ConvTranspose2d(8*depth, 4*depth, 5, stride=2),
            activation(),
            nn.ConvTranspose2d(4*depth, 2*depth, 5, stride=2),
            activation(),
            nn.ConvTranspose2d(2*depth, 1*depth, 6, stride=2),
            activation(),
            nn.ConvTranspose2d(1*depth, output_channels, 6, stride=2),
        )
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch, feature_size)
        Returns:
            reconstruction: (batch, channels, height, width)
        """
        x = self.fc(x)
        x = x.reshape(x.shape[0], 8*self.depth, self.h, self.w)
        x = self.layers(x)
        return x


class MLP(nn.Module):
    """多层感知机"""
    
    def __init__(self, input_size: int, hidden_size: int, output_size: int,
                 num_layers: int = 2, activation: nn.Module = nn.ELU,
                 output_activation: Optional[nn.Module] = None):
        super().__init__()
        
        layers = []
        current_size = input_size
        
        for _ in range(num_layers):
            layers.append(nn.Linear(current_size, hidden_size))
            layers.append(activation())
            current_size = hidden_size
        
        layers.append(nn.Linear(current_size, output_size))
        if output_activation is not None:
            layers.append(output_activation())
        
        self.network = nn.Sequential(*layers)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


# ============================================================================
# 2. RSSM (Recurrent State-Space Model) - Dreamer核心
# ============================================================================

class RSSMState(NamedTuple):
    """RSSM状态容器"""
    stochastic: torch.Tensor  # 随机状态 z_t
    deterministic: torch.Tensor  # 确定性状态 h_t
    
    def flatten(self) -> torch.Tensor:
        """将状态展平为向量"""
        return torch.cat([self.stochastic, self.deterministic], dim=-1)
    
    @property
    def shape(self) -> torch.Size:
        return self.stochastic.shape


class RSSM(nn.Module):
    """
    循环状态空间模型 - Dreamer的世界模型核心
    
    包含:
    - 确定性路径: h_t = f(h_{t-1}, z_{t-1}, a_{t-1})
    - 随机状态: z_t ~ p(z_t | h_t)
    
    参考: "Learning Latent Dynamics for Planning from Pixels" (Hafner et al., 2019)
    """
    
    def __init__(
        self,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        action_size: int = 6,
        obs_embed_size: int = 1024,
        hidden_size: int = 200,
        activation: nn.Module = nn.ELU,
        discrete: bool = False,
        num_categories: int = 32
    ):
        super().__init__()
        
        self.stochastic_size = stochastic_size
        self.deterministic_size = deterministic_size
        self.action_size = action_size
        self.obs_embed_size = obs_embed_size
        self.hidden_size = hidden_size
        self.discrete = discrete
        self.num_categories = num_categories
        
        if discrete:
            self.stochastic_flat_size = num_categories * stochastic_size
        else:
            self.stochastic_flat_size = stochastic_size
        
        # 确定性状态更新 (GRU)
        self.gru = nn.GRUCell(
            self.stochastic_flat_size + action_size,
            deterministic_size
        )
        
        # 先验网络: p(z_t | h_t)
        if discrete:
            self.prior_net = nn.Sequential(
                nn.Linear(deterministic_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, stochastic_size * num_categories)
            )
        else:
            self.prior_net = nn.Sequential(
                nn.Linear(deterministic_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, 2 * stochastic_size)  # mean, log_std
            )
        
        # 后验网络: q(z_t | h_t, o_t)
        if discrete:
            self.posterior_net = nn.Sequential(
                nn.Linear(deterministic_size + obs_embed_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, stochastic_size * num_categories)
            )
        else:
            self.posterior_net = nn.Sequential(
                nn.Linear(deterministic_size + obs_embed_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, 2 * stochastic_size)
            )
        
    def initial_state(self, batch_size: int, device: torch.device) -> RSSMState:
        """创建初始状态"""
        return RSSMState(
            stochastic=torch.zeros(batch_size, self.stochastic_flat_size, device=device),
            deterministic=torch.zeros(batch_size, self.deterministic_size, device=device)
        )
    
    def observe_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor,
        obs_embed: torch.Tensor
    ) -> Tuple[RSSMState, Dict]:
        """
        观测步骤: 使用后验分布更新状态
        
        Args:
            prev_state: 前一时刻状态
            action: 执行的动作
            obs_embed: 观测嵌入
            
        Returns:
            state: 新状态
            info: 包含分布信息的字典
        """
        # 确定性状态更新
        x = torch.cat([prev_state.stochastic, action], dim=-1)
        h = self.gru(x, prev_state.deterministic)
        
        # 先验分布 p(z_t | h_t)
        prior_logits = self.prior_net(h)
        prior_dist = self._make_distribution(prior_logits)
        
        # 后验分布 q(z_t | h_t, o_t)
        x = torch.cat([h, obs_embed], dim=-1)
        posterior_logits = self.posterior_net(x)
        posterior_dist = self._make_distribution(posterior_logits)
        
        # 从后验采样
        z = posterior_dist.rsample()
        
        state = RSSMState(stochastic=z, deterministic=h)
        
        info = {
            'prior': prior_dist,
            'posterior': posterior_dist,
            'prior_logits': prior_logits,
            'posterior_logits': posterior_logits
        }
        
        return state, info
    
    def imagine_step(
        self,
        prev_state: RSSMState,
        action: torch.Tensor
    ) -> Tuple[RSSMState, Dict]:
        """
        想象步骤: 仅使用先验分布预测下一状态
        
        Args:
            prev_state: 前一时刻状态
            action: 执行的动作
            
        Returns:
            state: 预测的新状态
            info: 包含分布信息的字典
        """
        # 确定性状态更新
        x = torch.cat([prev_state.stochastic, action], dim=-1)
        h = self.gru(x, prev_state.deterministic)
        
        # 先验分布 p(z_t | h_t)
        prior_logits = self.prior_net(h)
        prior_dist = self._make_distribution(prior_logits)
        
        # 从先验采样
        z = prior_dist.rsample()
        
        state = RSSMState(stochastic=z, deterministic=h)
        
        info = {
            'prior': prior_dist,
            'prior_logits': prior_logits
        }
        
        return state, info
    
    def _make_distribution(self, logits: torch.Tensor) -> D.Distribution:
        """从网络输出创建分布"""
        if self.discrete:
            # 离散: 独立分类分布
            logits = logits.reshape(*logits.shape[:-1], self.stochastic_size, self.num_categories)
            dist = D.Independent(D.OneHotCategoricalStraightThrough(logits=logits), 1)
        else:
            # 连续: 对角高斯分布
            mean, log_std = torch.chunk(logits, 2, dim=-1)
            std = F.softplus(log_std) + 0.1
            dist = D.Independent(D.Normal(mean, std), 1)
        return dist


# ============================================================================
# 3. 世界模型完整实现
# ============================================================================

class WorldModel(nn.Module):
    """
    完整的世界模型
    
    包含:
    - 观测编码器: 将观测转换为嵌入
    - RSSM: 动态模型
    - 观测解码器: 从状态重建观测
    - 奖励预测器: 预测奖励
    - 终止预测器: 预测回合是否结束
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_size: int,
        stochastic_size: int = 32,
        deterministic_size: int = 200,
        hidden_size: int = 200,
        obs_embed_size: int = 1024,
        activation: nn.Module = nn.ELU,
        discrete: bool = False
    ):
        super().__init__()
        
        self.obs_shape = obs_shape
        self.action_size = action_size
        
        # 观测编码器 (假设是图像)
        if len(obs_shape) == 3:
            self.encoder = ConvEncoder(
                input_channels=obs_shape[0],
                depth=32
            )
            # 计算编码器输出大小
            with torch.no_grad():
                dummy = torch.zeros(1, *obs_shape)
                embed_size = self.encoder(dummy).shape[1]
            self.embed_fc = nn.Linear(embed_size, obs_embed_size)
        else:
            # 向量观测
            self.encoder = nn.Sequential(
                nn.Linear(obs_shape[0], hidden_size),
                activation(),
                nn.Linear(hidden_size, obs_embed_size)
            )
            self.embed_fc = nn.Identity()
        
        # RSSM
        self.rssm = RSSM(
            stochastic_size=stochastic_size,
            deterministic_size=deterministic_size,
            action_size=action_size,
            obs_embed_size=obs_embed_size,
            hidden_size=hidden_size,
            activation=activation,
            discrete=discrete
        )
        
        # 特征大小
        self.feature_size = stochastic_size + deterministic_size
        
        # 观测解码器
        if len(obs_shape) == 3:
            self.decoder = ConvDecoder(
                feature_size=self.feature_size,
                output_channels=obs_shape[0]
            )
        else:
            self.decoder = nn.Sequential(
                nn.Linear(self.feature_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, obs_shape[0])
            )
        
        # 奖励预测器
        self.reward_predictor = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1)
        )
        
        # 终止预测器 (折扣因子)
        self.continue_predictor = nn.Sequential(
            nn.Linear(self.feature_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1)
        )
        
    def encode_observation(self, obs: torch.Tensor) -> torch.Tensor:
        """编码观测"""
        embed = self.encoder(obs)
        return self.embed_fc(embed)
    
    def decode_observation(self, state: RSSMState) -> torch.Tensor:
        """从状态解码观测"""
        features = state.flatten()
        return self.decoder(features)
    
    def predict_reward(self, state: RSSMState) -> torch.Tensor:
        """预测奖励"""
        features = state.flatten()
        return self.reward_predictor(features)
    
    def predict_continue(self, state: RSSMState) -> torch.Tensor:
        """预测是否继续 (1 - 终止概率)"""
        features = state.flatten()
        return self.continue_predictor(features)
    
    def imagine_trajectory(
        self,
        initial_state: RSSMState,
        policy: callable,
        horizon: int
    ) -> Tuple[List[RSSMState], List[torch.Tensor], Dict]:
        """
        想象轨迹
        
        Args:
            initial_state: 初始状态
            policy: 策略函数 state -> action
            horizon: 想象步数
            
        Returns:
            states: 状态序列
            actions: 动作序列
            info: 额外信息
        """
        states = [initial_state]
        actions = []
        rewards = []
        
        state = initial_state
        for _ in range(horizon):
            # 策略选择动作
            action = policy(state)
            actions.append(action)
            
            # 想象下一状态
            state, info = self.rssm.imagine_step(state, action)
            states.append(state)
            
            # 预测奖励
            reward = self.predict_reward(state)
            rewards.append(reward)
        
        info = {
            'rewards': torch.stack(rewards, dim=1) if rewards else None
        }
        
        return states, actions, info
    
    def compute_loss(
        self,
        observations: torch.Tensor,
        actions: torch.Tensor,
        rewards: torch.Tensor,
        continues: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        """
        计算世界模型训练损失
        
        Args:
            observations: (batch, seq, channels, height, width)
            actions: (batch, seq, action_size)
            rewards: (batch, seq, 1)
            continues: (batch, seq, 1)
            
        Returns:
            losses: 各组件损失字典
        """
        batch_size, seq_len = observations.shape[:2]
        device = observations.device
        
        # 初始化状态
        state = self.rssm.initial_state(batch_size, device)
        
        # 存储KL散度
        kl_divs = []
        
        # 存储预测
        obs_preds = []
        reward_preds = []
        continue_preds = []
        
        for t in range(seq_len):
            obs_t = observations[:, t]
            action_t = actions[:, t]
            
            # 编码观测
            obs_embed = self.encode_observation(obs_t)
            
            # 观测步骤
            state, info = self.rssm.observe_step(state, action_t, obs_embed)
            
            # 记录KL散度
            prior = info['prior']
            posterior = info['posterior']
            kl_div = D.kl_divergence(posterior, prior).sum(dim=-1)
            kl_divs.append(kl_div)
            
            # 预测
            obs_pred = self.decode_observation(state)
            reward_pred = self.predict_reward(state)
            continue_pred = self.predict_continue(state)
            
            obs_preds.append(obs_pred)
            reward_preds.append(reward_pred)
            continue_preds.append(continue_pred)
        
        # 堆叠预测
        obs_preds = torch.stack(obs_preds, dim=1)
        reward_preds = torch.stack(reward_preds, dim=1)
        continue_preds = torch.stack(continue_preds, dim=1)
        
        # 计算损失
        # 观测重建损失
        obs_loss = F.mse_loss(obs_preds, observations)
        
        # 奖励预测损失
        reward_loss = F.mse_loss(reward_preds, rewards)
        
        # 继续预测损失
        continue_loss = F.binary_cross_entropy_with_logits(continue_preds, continues)
        
        # KL散度损失 (带自由比特)
        kl_divs = torch.stack(kl_divs, dim=1)
        kl_loss = torch.maximum(kl_divs, torch.ones_like(kl_divs) * 1.0).mean()
        
        total_loss = obs_loss + reward_loss + continue_loss + 0.1 * kl_loss
        
        return {
            'total': total_loss,
            'observation': obs_loss,
            'reward': reward_loss,
            'continue': continue_loss,
            'kl': kl_loss
        }


# ============================================================================
# 4. 动作规划模块
# ============================================================================

class ActorCritic(nn.Module):
    """
    Actor-Critic策略网络
    
    在Dreamer中，策略在想象轨迹上训练
    """
    
    def __init__(
        self,
        feature_size: int,
        action_size: int,
        hidden_size: int = 200,
        activation: nn.Module = nn.ELU,
        discrete: bool = False,
        action_bounds: Tuple[float, float] = (-1.0, 1.0)
    ):
        super().__init__()
        
        self.action_size = action_size
        self.discrete = discrete
        self.action_min, self.action_max = action_bounds
        
        # Actor (策略)
        if discrete:
            self.actor = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, action_size)
            )
        else:
            self.actor_mean = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, action_size)
            )
            self.actor_std = nn.Sequential(
                nn.Linear(feature_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, hidden_size),
                activation(),
                nn.Linear(hidden_size, action_size),
                nn.Softplus()
            )
        
        # Critic (价值函数)
        self.critic = nn.Sequential(
            nn.Linear(feature_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, hidden_size),
            activation(),
            nn.Linear(hidden_size, 1)
        )
        
    def forward(self, state: RSSMState) -> Tuple[D.Distribution, torch.Tensor]:
        """
        前向传播
        
        Returns:
            action_dist: 动作分布
            value: 状态价值
        """
        features = state.flatten()
        
        # 价值估计
        value = self.critic(features)
        
        # 动作分布
        if self.discrete:
            logits = self.actor(features)
            action_dist = D.Categorical(logits=logits)
        else:
            mean = self.actor_mean(features)
            std = self.actor_std(features) + 0.1
            
            # 应用动作边界 (tanh压缩)
            action_dist = D.TransformedDistribution(
                D.Normal(mean, std),
                [D.transforms.TanhTransform(cache_size=1)]
            )
        
        return action_dist, value
    
    def get_action(self, state: RSSMState, deterministic: bool = False) -> torch.Tensor:
        """获取动作"""
        action_dist, _ = self.forward(state)
        if deterministic:
            if self.discrete:
                action = action_dist.probs.argmax(dim=-1)
            else:
                action = action_dist.mean
        else:
            action = action_dist.sample()
        return action


class MPCPlanner:
    """
    模型预测控制规划器
    
    使用交叉熵方法(CEM)优化动作序列
    """
    
    def __init__(
        self,
        world_model: WorldModel,
        actor_critic: ActorCritic,
        horizon: int = 12,
        num_samples: int = 1000,
        num_elites: int = 100,
        num_iterations: int = 5,
        temperature: float = 0.5
    ):
        self.world_model = world_model
        self.actor_critic = actor_critic
        self.horizon = horizon
        self.num_samples = num_samples
        self.num_elites = num_elites
        self.num_iterations = num_iterations
        self.temperature = temperature
        
    def plan(
        self,
        state: RSSMState,
        return_trajectory: bool = False
    ) -> torch.Tensor:
        """
        规划最优动作
        
        Args:
            state: 当前状态
            return_trajectory: 是否返回完整轨迹
            
        Returns:
            action: 最优动作
        """
        batch_size = state.shape[0]
        device = state.stochastic.device
        action_size = self.world_model.action_size
        
        # 初始化动作分布
        mean = torch.zeros(self.horizon, action_size, device=device)
        std = torch.ones(self.horizon, action_size, device=device)
        
        for _ in range(self.num_iterations):
            # 采样动作序列
            actions = mean.unsqueeze(0) + std.unsqueeze(0) * torch.randn(
                self.num_samples, self.horizon, action_size, device=device
            )
            actions = torch.clamp(actions, -1, 1)
            
            # 评估每个动作序列
            scores = self._evaluate_actions(state, actions)
            
            # 选择elite
            elite_indices = torch.topk(scores, self.num_elites).indices
            elite_actions = actions[elite_indices]
            
            # 更新分布
            mean = elite_actions.mean(dim=0)
            std = elite_actions.std(dim=0) + 1e-6
            
            # 温度退火
            std = std * self.temperature
        
        # 返回第一个动作
        best_action = mean[0:1].expand(batch_size, -1)
        
        if return_trajectory:
            return best_action, mean
        return best_action
    
    def _evaluate_actions(
        self,
        initial_state: RSSMState,
        action_sequences: torch.Tensor
    ) -> torch.Tensor:
        """
        评估动作序列
        
        Args:
            initial_state: 初始状态 (batch, ...)
            action_sequences: 动作序列 (num_samples, horizon, action_size)
            
        Returns:
            scores: 每个序列的得分
        """
        num_samples = action_sequences.shape[0]
        device = action_sequences.device
        
        # 扩展初始状态
        state = RSSMState(
            stochastic=initial_state.stochastic.expand(num_samples, -1),
            deterministic=initial_state.deterministic.expand(num_samples, -1)
        )
        
        total_rewards = torch.zeros(num_samples, device=device)
        
        for t in range(self.horizon):
            actions = action_sequences[:, t]
            
            # 想象下一状态
            state, _ = self.world_model.rssm.imagine_step(state, actions)
            
            # 预测奖励
            rewards = self.world_model.predict_reward(state)
            total_rewards += rewards.squeeze(-1)
        
        return total_rewards


# ============================================================================
# 5. JEPA风格的世界模型
# ============================================================================

class JEPAWorldModel(nn.Module):
    """
    JEPA风格的世界模型
    
    特点:
    - 非生成式: 不重建观测
    - 在表示空间预测
    - 使用对比学习
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_size: int,
        embed_size: int = 256,
        predictor_hidden: int = 512,
        activation: nn.Module = nn.GELU
    ):
        super().__init__()
        
        # 上下文编码器
        if len(obs_shape) == 3:
            self.context_encoder = nn.Sequential(
                ConvEncoder(obs_shape[0], depth=32),
                nn.Linear(2048, embed_size)  # 假设输出2048
            )
        else:
            self.context_encoder = nn.Sequential(
                nn.Linear(obs_shape[0], 256),
                activation(),
                nn.Linear(256, embed_size)
            )
        
        # 目标编码器 (EMA更新)
        self.target_encoder = nn.Sequential(
            nn.Linear(embed_size, embed_size),
            activation(),
            nn.Linear(embed_size, embed_size)
        )
        
        # 预测器 (带动作条件)
        self.predictor = nn.Sequential(
            nn.Linear(embed_size + action_size, predictor_hidden),
            activation(),
            nn.Linear(predictor_hidden, predictor_hidden),
            activation(),
            nn.Linear(predictor_hidden, embed_size)
        )
        
        self.embed_size = embed_size
        
    def encode_context(self, obs: torch.Tensor) -> torch.Tensor:
        """编码上下文"""
        return self.context_encoder(obs)
    
    def encode_target(self, obs: torch.Tensor) -> torch.Tensor:
        """编码目标"""
        embed = self.context_encoder(obs)
        return self.target_encoder(embed)
    
    def predict_next(self, context_embed: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """预测下一状态的表示"""
        x = torch.cat([context_embed, action], dim=-1)
        return self.predictor(x)
    
    def compute_loss(
        self,
        obs_t: torch.Tensor,
        obs_t1: torch.Tensor,
        action: torch.Tensor
    ) -> torch.Tensor:
        """
        计算JEPA损失
        
        预测下一状态的表示，与目标编码器的输出对比
        """
        # 编码
        context_embed = self.encode_context(obs_t)
        with torch.no_grad():
            target_embed = self.encode_target(obs_t1)
        
        # 预测
        pred_embed = self.predict_next(context_embed, action)
        
        # 归一化
        pred_embed = F.normalize(pred_embed, dim=-1)
        target_embed = F.normalize(target_embed, dim=-1)
        
        # 余弦相似度损失
        loss = 2 - 2 * (pred_embed * target_embed).sum(dim=-1).mean()
        
        return loss


# ============================================================================
# 6. 环境交互示例
# ============================================================================

class SimpleGridWorld:
    """简单网格世界环境 - 用于测试"""
    
    def __init__(self, size: int = 8):
        self.size = size
        self.agent_pos = None
        self.goal_pos = None
        self.reset()
        
    def reset(self):
        """重置环境"""
        self.agent_pos = np.array([0, 0])
        self.goal_pos = np.array([self.size - 1, self.size - 1])
        return self._get_obs()
    
    def step(self, action: int):
        """执行动作"""
        # 动作: 0=上, 1=右, 2=下, 3=左
        moves = [(-1, 0), (0, 1), (1, 0), (0, -1)]
        move = np.array(moves[action])
        
        new_pos = self.agent_pos + move
        new_pos = np.clip(new_pos, 0, self.size - 1)
        self.agent_pos = new_pos
        
        # 计算奖励
        dist = np.linalg.norm(self.agent_pos - self.goal_pos)
        reward = -dist / self.size
        
        # 检查是否到达目标
        done = np.array_equal(self.agent_pos, self.goal_pos)
        if done:
            reward = 10.0
        
        return self._get_obs(), reward, done, {}
    
    def _get_obs(self):
        """获取观测"""
        # 返回agent和goal的位置
        return np.concatenate([self.agent_pos, self.goal_pos]).astype(np.float32)
    
    @property
    def observation_space(self):
        return (4,)
    
    @property
    def action_space(self):
        return 4


def train_world_model_example():
    """训练世界模型的示例"""
    
    # 创建环境
    env = SimpleGridWorld(size=8)
    
    # 配置
    obs_shape = env.observation_space
    action_size = env.action_space
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建世界模型
    world_model = WorldModel(
        obs_shape=obs_shape,
        action_size=action_size,
        stochastic_size=32,
        deterministic_size=200,
        hidden_size=200,
        obs_embed_size=128,
        discrete=False
    ).to(device)
    
    # 创建Actor-Critic
    feature_size = world_model.feature_size
    actor_critic = ActorCritic(
        feature_size=feature_size,
        action_size=action_size,
        hidden_size=200,
        discrete=True
    ).to(device)
    
    # 优化器
    wm_optimizer = torch.optim.Adam(world_model.parameters(), lr=1e-3)
    ac_optimizer = torch.optim.Adam(actor_critic.parameters(), lr=3e-4)
    
    # 回放缓冲区 (简化版)
    replay_buffer = []
    
    # 收集数据
    print("收集初始数据...")
    for episode in range(100):
        obs = env.reset()
        done = False
        episode_data = []
        
        while not done:
            # 随机动作
            action = np.random.randint(action_size)
            next_obs, reward, done, _ = env.step(action)
            
            episode_data.append({
                'obs': obs,
                'action': action,
                'reward': reward,
                'next_obs': next_obs,
                'done': done
            })
            
            obs = next_obs
        
        replay_buffer.extend(episode_data)
    
    print(f"收集了 {len(replay_buffer)} 条经验")
    
    # 训练世界模型
    print("\n训练世界模型...")
    batch_size = 32
    seq_len = 10
    
    for step in range(1000):
        # 采样序列
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_continues = []
        
        for _ in range(batch_size):
            start_idx = np.random.randint(0, len(replay_buffer) - seq_len)
            seq = replay_buffer[start_idx:start_idx + seq_len]
            
            batch_obs.append([s['obs'] for s in seq])
            batch_actions.append([s['action'] for s in seq])
            batch_rewards.append([s['reward'] for s in seq])
            batch_continues.append([0.0 if s['done'] else 1.0 for s in seq])
        
        # 转换为tensor
        obs_tensor = torch.FloatTensor(np.array(batch_obs)).to(device)
        actions_tensor = torch.LongTensor(np.array(batch_actions)).to(device)
        rewards_tensor = torch.FloatTensor(np.array(batch_rewards)).unsqueeze(-1).to(device)
        continues_tensor = torch.FloatTensor(np.array(batch_continues)).unsqueeze(-1).to(device)
        
        # One-hot编码动作
        actions_onehot = F.one_hot(actions_tensor, num_classes=action_size).float()
        
        # 计算损失
        losses = world_model.compute_loss(
            obs_tensor,
            actions_onehot,
            rewards_tensor,
            continues_tensor
        )
        
        # 更新
        wm_optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(world_model.parameters(), 100.0)
        wm_optimizer.step()
        
        if step % 100 == 0:
            print(f"Step {step}: Loss={losses['total'].item():.4f}, "
                  f"Obs={losses['observation'].item():.4f}, "
                  f"Reward={losses['reward'].item():.4f}")
    
    print("\n训练完成!")
    
    # 测试想象能力
    print("\n测试想象能力...")
    with torch.no_grad():
        # 初始化状态
        batch_size = 1
        state = world_model.rssm.initial_state(batch_size, device)
        
        # 编码初始观测
        obs = torch.FloatTensor(env.reset()).unsqueeze(0).to(device)
        obs_embed = world_model.encode_observation(obs)
        
        # 初始动作
        action = torch.zeros(batch_size, action_size).to(device)
        action[0, 0] = 1  # 第一个动作
        
        # 观测步骤
        state, _ = world_model.rssm.observe_step(state, action, obs_embed)
        
        # 想象未来
        print("想象轨迹:")
        imagined_states = [state]
        imagined_rewards = []
        
        for t in range(10):
            # 随机动作
            action_idx = np.random.randint(action_size)
            action = F.one_hot(torch.LongTensor([action_idx]), action_size).float().to(device)
            
            # 想象
            state, _ = world_model.rssm.imagine_step(state, action)
            reward = world_model.predict_reward(state)
            
            imagined_states.append(state)
            imagined_rewards.append(reward.item())
            
            print(f"  Step {t}: Action={action_idx}, Predicted Reward={reward.item():.4f}")
    
    return world_model, actor_critic


def test_jepa_model():
    """测试JEPA模型"""
    
    print("\n" + "="*50)
    print("测试JEPA世界模型")
    print("="*50)
    
    # 创建简单数据
    batch_size = 16
    obs_shape = (4,)  # 4维观测
    action_size = 4
    
    # 创建模型
    model = JEPAWorldModel(
        obs_shape=obs_shape,
        action_size=action_size,
        embed_size=64
    )
    
    # 模拟数据
    obs_t = torch.randn(batch_size, *obs_shape)
    obs_t1 = torch.randn(batch_size, *obs_shape)
    action = torch.randn(batch_size, action_size)
    
    # 计算损失
    loss = model.compute_loss(obs_t, obs_t1, action)
    print(f"JEPA Loss: {loss.item():.4f}")
    
    # 测试预测
    context = model.encode_context(obs_t)
    prediction = model.predict_next(context, action)
    target = model.encode_target(obs_t1)
    
    print(f"Context shape: {context.shape}")
    print(f"Prediction shape: {prediction.shape}")
    print(f"Target shape: {target.shape}")
    
    return model


# ============================================================================
# 7. 主函数
# ============================================================================

if __name__ == "__main__":
    print("="*60)
    print("世界模型 (World Models) - 核心实现")
    print("="*60)
    
    # 测试RSSM
    print("\n1. 测试RSSM...")
    rssm = RSSM(
        stochastic_size=32,
        deterministic_size=200,
        action_size=4,
        obs_embed_size=128
    )
    
    batch_size = 4
    device = torch.device('cpu')
    
    state = rssm.initial_state(batch_size, device)
    action = torch.randn(batch_size, 4)
    obs_embed = torch.randn(batch_size, 128)
    
    # 观测步骤
    next_state, info = rssm.observe_step(state, action, obs_embed)
    print(f"   RSSM State shape: stochastic={next_state.stochastic.shape}, "
          f"deterministic={next_state.deterministic.shape}")
    
    # 想象步骤
    imagined_state, info = rssm.imagine_step(state, action)
    print(f"   Imagined State shape: stochastic={imagined_state.stochastic.shape}")
    
    # 测试JEPA
    print("\n2. 测试JEPA模型...")
    jepa_model = test_jepa_model()
    
    # 训练示例
    print("\n3. 训练世界模型示例...")
    print("-"*60)
    world_model, actor_critic = train_world_model_example()
    
    print("\n" + "="*60)
    print("所有测试完成!")
    print("="*60)
