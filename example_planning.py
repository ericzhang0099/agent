"""
示例: 使用世界模型进行规划和控制
演示如何在实际任务中应用世界模型
"""

import torch
import torch.nn.functional as F
import numpy as np
import gym
from typing import Tuple, List, Optional
from world_model import (
    WorldModel, ActorCritic, MPCPlanner, RSSMState,
    SimpleGridWorld
)


class WorldModelAgent:
    """
    基于世界模型的智能体
    
    包含:
    - 世界模型学习
    - 想象训练
    - MPC规划
    """
    
    def __init__(
        self,
        obs_shape: Tuple[int, ...],
        action_size: int,
        discrete: bool = False,
        device: str = 'cpu'
    ):
        self.obs_shape = obs_shape
        self.action_size = action_size
        self.discrete = discrete
        self.device = torch.device(device)
        
        # 创建世界模型
        self.world_model = WorldModel(
            obs_shape=obs_shape,
            action_size=action_size,
            stochastic_size=32,
            deterministic_size=200,
            hidden_size=200,
            obs_embed_size=128,
            discrete=discrete
        ).to(self.device)
        
        # 创建Actor-Critic
        self.feature_size = self.world_model.feature_size
        self.actor_critic = ActorCritic(
            feature_size=self.feature_size,
            action_size=action_size,
            hidden_size=200,
            discrete=discrete
        ).to(self.device)
        
        # 优化器
        self.wm_optimizer = torch.optim.Adam(
            self.world_model.parameters(), lr=1e-3
        )
        self.ac_optimizer = torch.optim.Adam(
            self.actor_critic.parameters(), lr=3e-4
        )
        
        # 回放缓冲区
        self.replay_buffer = []
        self.buffer_size = 10000
        
        # MPC规划器
        self.mpc_planner = MPCPlanner(
            world_model=self.world_model,
            actor_critic=self.actor_critic,
            horizon=12,
            num_samples=500,
            num_elites=50,
            num_iterations=5
        )
        
    def select_action(
        self,
        obs: np.ndarray,
        use_mpc: bool = False,
        deterministic: bool = False
    ) -> int:
        """
        选择动作
        
        Args:
            obs: 当前观测
            use_mpc: 是否使用MPC规划
            deterministic: 是否确定性选择
            
        Returns:
            action: 选择的动作
        """
        with torch.no_grad():
            obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            
            if use_mpc:
                # 使用MPC规划
                # 首先编码观测
                obs_embed = self.world_model.encode_observation(obs_tensor)
                
                # 初始化状态 (简化: 使用零初始化)
                state = self.world_model.rssm.initial_state(1, self.device)
                action = torch.zeros(1, self.action_size).to(self.device)
                state, _ = self.world_model.rssm.observe_step(state, action, obs_embed)
                
                # MPC规划
                action = self.mpc_planner.plan(state)
                
                if self.discrete:
                    action = action.argmax(dim=-1).item()
                else:
                    action = action.cpu().numpy()[0]
            else:
                # 使用策略网络
                # 编码观测
                obs_embed = self.world_model.encode_observation(obs_tensor)
                state = self.world_model.rssm.initial_state(1, self.device)
                action_zero = torch.zeros(1, self.action_size).to(self.device)
                state, _ = self.world_model.rssm.observe_step(state, action_zero, obs_embed)
                
                # 获取动作
                action = self.actor_critic.get_action(state, deterministic)
                
                if self.discrete:
                    action = action.item()
                else:
                    action = action.cpu().numpy()[0]
        
        return action
    
    def store_transition(self, transition: dict):
        """存储转移"""
        self.replay_buffer.append(transition)
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def train_world_model(self, batch_size: int = 32, seq_len: int = 10) -> dict:
        """
        训练世界模型
        
        Returns:
            losses: 损失字典
        """
        if len(self.replay_buffer) < batch_size * seq_len:
            return {'total': 0.0}
        
        # 采样序列
        batch_obs = []
        batch_actions = []
        batch_rewards = []
        batch_continues = []
        
        for _ in range(batch_size):
            start_idx = np.random.randint(0, max(1, len(self.replay_buffer) - seq_len))
            seq = self.replay_buffer[start_idx:start_idx + seq_len]
            
            batch_obs.append([s['obs'] for s in seq])
            batch_actions.append([s['action'] for s in seq])
            batch_rewards.append([s['reward'] for s in seq])
            batch_continues.append([0.0 if s['done'] else 1.0 for s in seq])
        
        # 转换为tensor
        obs_tensor = torch.FloatTensor(np.array(batch_obs)).to(self.device)
        
        if self.discrete:
            actions_tensor = torch.LongTensor(np.array(batch_actions)).to(self.device)
            actions_tensor = F.one_hot(actions_tensor, self.action_size).float()
        else:
            actions_tensor = torch.FloatTensor(np.array(batch_actions)).to(self.device)
        
        rewards_tensor = torch.FloatTensor(np.array(batch_rewards)).unsqueeze(-1).to(self.device)
        continues_tensor = torch.FloatTensor(np.array(batch_continues)).unsqueeze(-1).to(self.device)
        
        # 计算损失
        losses = self.world_model.compute_loss(
            obs_tensor,
            actions_tensor,
            rewards_tensor,
            continues_tensor
        )
        
        # 更新
        self.wm_optimizer.zero_grad()
        losses['total'].backward()
        torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), 100.0)
        self.wm_optimizer.step()
        
        return {k: v.item() for k, v in losses.items()}
    
    def train_policy(self, batch_size: int = 32, horizon: int = 15) -> dict:
        """
        在想象轨迹上训练策略
        
        Returns:
            losses: 损失字典
        """
        if len(self.replay_buffer) < batch_size:
            return {'actor_loss': 0.0, 'critic_loss': 0.0}
        
        # 采样起始状态
        indices = np.random.choice(len(self.replay_buffer), batch_size)
        
        obs_list = []
        for idx in indices:
            obs_list.append(self.replay_buffer[idx]['obs'])
        
        obs_tensor = torch.FloatTensor(np.array(obs_list)).to(self.device)
        
        # 编码观测
        with torch.no_grad():
            obs_embed = self.world_model.encode_observation(obs_tensor)
            state = self.world_model.rssm.initial_state(batch_size, self.device)
            action_zero = torch.zeros(batch_size, self.action_size).to(self.device)
            initial_state, _ = self.world_model.rssm.observe_step(state, action_zero, obs_embed)
        
        # 想象轨迹
        states = [initial_state]
        actions = []
        log_probs = []
        values = []
        rewards = []
        
        state = initial_state
        for t in range(horizon):
            # 策略选择动作
            action_dist, value = self.actor_critic(state)
            action = action_dist.sample()
            
            if self.discrete:
                action_onehot = F.one_hot(action, self.action_size).float()
            else:
                action_onehot = action
            
            log_prob = action_dist.log_prob(action)
            
            # 想象下一状态
            with torch.no_grad():
                state, _ = self.world_model.rssm.imagine_step(state, action_onehot)
                reward = self.world_model.predict_reward(state)
            
            states.append(state)
            actions.append(action)
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
        
        # 计算回报
        returns = []
        R = torch.zeros(batch_size, 1).to(self.device)
        
        for t in reversed(range(horizon)):
            R = rewards[t] + 0.99 * R
            returns.insert(0, R)
        
        returns = torch.cat(returns, dim=1).unsqueeze(-1)
        values = torch.cat(values, dim=1)
        
        # Actor损失 (策略梯度)
        advantages = (returns - values).detach()
        log_probs = torch.stack(log_probs, dim=1).unsqueeze(-1)
        
        actor_loss = -(log_probs * advantages).mean()
        
        # Critic损失
        critic_loss = F.mse_loss(values, returns)
        
        # 熵奖励
        entropy = action_dist.entropy().mean()
        
        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss - 0.01 * entropy
        
        # 更新
        self.ac_optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor_critic.parameters(), 100.0)
        self.ac_optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }


def run_episode(env, agent: WorldModelAgent, max_steps: int = 200, use_mpc: bool = False) -> dict:
    """
    运行一个回合
    
    Returns:
        info: 回合信息
    """
    obs = env.reset()
    done = False
    total_reward = 0
    steps = 0
    
    while not done and steps < max_steps:
        # 选择动作
        action = agent.select_action(obs, use_mpc=use_mpc)
        
        # 执行动作
        next_obs, reward, done, _ = env.step(action)
        
        # 存储转移
        agent.store_transition({
            'obs': obs,
            'action': action,
            'reward': reward,
            'next_obs': next_obs,
            'done': done
        })
        
        total_reward += reward
        obs = next_obs
        steps += 1
    
    return {
        'total_reward': total_reward,
        'steps': steps
    }


def train_agent():
    """训练智能体示例"""
    
    print("="*60)
    print("训练基于世界模型的智能体")
    print("="*60)
    
    # 创建环境
    env = SimpleGridWorld(size=8)
    
    # 创建智能体
    agent = WorldModelAgent(
        obs_shape=env.observation_space,
        action_size=env.action_space,
        discrete=True,
        device='cpu'
    )
    
    # 训练参数
    num_episodes = 500
    wm_train_freq = 50  # 每50步训练一次世界模型
    policy_train_freq = 50  # 每50步训练一次策略
    
    episode_rewards = []
    
    print("\n开始训练...")
    for episode in range(num_episodes):
        # 运行回合
        info = run_episode(env, agent, max_steps=50, use_mpc=False)
        episode_rewards.append(info['total_reward'])
        
        # 训练世界模型
        if episode % 10 == 0 and len(agent.replay_buffer) > 100:
            wm_losses = agent.train_world_model(batch_size=16, seq_len=10)
        else:
            wm_losses = {'total': 0.0}
        
        # 训练策略
        if episode % 10 == 0 and len(agent.replay_buffer) > 100:
            policy_losses = agent.train_policy(batch_size=16, horizon=10)
        else:
            policy_losses = {'actor_loss': 0.0}
        
        # 打印进度
        if episode % 50 == 0:
            avg_reward = np.mean(episode_rewards[-50:]) if len(episode_rewards) >= 50 else np.mean(episode_rewards)
            print(f"Episode {episode}: "
                  f"Reward={info['total_reward']:.2f}, "
                  f"Avg Reward={avg_reward:.2f}, "
                  f"Buffer Size={len(agent.replay_buffer)}, "
                  f"WM Loss={wm_losses['total']:.4f}")
    
    # 测试阶段
    print("\n" + "="*60)
    print("测试阶段 - 使用训练好的策略")
    print("="*60)
    
    test_rewards = []
    for _ in range(10):
        info = run_episode(env, agent, max_steps=50, use_mpc=False)
        test_rewards.append(info['total_reward'])
    
    print(f"平均测试奖励: {np.mean(test_rewards):.2f} (+/- {np.std(test_rewards):.2f})")
    
    # 测试MPC规划
    print("\n测试MPC规划...")
    mpc_rewards = []
    for _ in range(5):
        info = run_episode(env, agent, max_steps=50, use_mpc=True)
        mpc_rewards.append(info['total_reward'])
    
    print(f"MPC平均奖励: {np.mean(mpc_rewards):.2f} (+/- {np.std(mpc_rewards):.2f})")
    
    return agent, episode_rewards


def visualize_imagination(agent: WorldModelAgent, env: SimpleGridWorld):
    """可视化想象轨迹"""
    
    print("\n" + "="*60)
    print("可视化想象能力")
    print("="*60)
    
    # 重置环境
    obs = env.reset()
    
    # 编码初始观测
    obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(agent.device)
    
    with torch.no_grad():
        obs_embed = agent.world_model.encode_observation(obs_tensor)
        state = agent.world_model.rssm.initial_state(1, agent.device)
        action_zero = torch.zeros(1, agent.action_size).to(agent.device)
        state, _ = agent.world_model.rssm.observe_step(state, action_zero, obs_embed)
        
        print("\n真实轨迹 vs 想象轨迹:")
        print("-"*40)
        
        # 真实轨迹
        real_obs = obs.copy()
        real_states = [real_obs]
        real_rewards = [0.0]
        
        for t in range(10):
            action = np.random.randint(agent.action_size)
            next_obs, reward, done, _ = env.step(action)
            real_states.append(next_obs)
            real_rewards.append(reward)
            real_obs = next_obs
        
        # 想象轨迹 (从相同的初始状态)
        imagined_rewards = []
        imagined_state = state
        
        for t in range(10):
            action_idx = np.random.randint(agent.action_size)
            action = F.one_hot(torch.LongTensor([action_idx]), agent.action_size).float().to(agent.device)
            
            imagined_state, _ = agent.world_model.rssm.imagine_step(imagined_state, action)
            reward = agent.world_model.predict_reward(imagined_state)
            imagined_rewards.append(reward.item())
        
        # 比较
        print("Step | Real Reward | Imagined Reward | Diff")
        print("-"*40)
        for t in range(min(len(real_rewards)-1, len(imagined_rewards))):
            diff = abs(real_rewards[t+1] - imagined_rewards[t])
            print(f"{t:4d} | {real_rewards[t+1]:11.4f} | {imagined_rewards[t]:15.4f} | {diff:.4f}")


def compare_planning_methods():
    """比较不同规划方法"""
    
    print("\n" + "="*60)
    print("比较规划方法: 策略网络 vs MPC")
    print("="*60)
    
    env = SimpleGridWorld(size=8)
    
    # 训练智能体
    agent, rewards = train_agent()
    
    # 可视化想象
    visualize_imagination(agent, env)
    
    print("\n" + "="*60)
    print("实验完成!")
    print("="*60)


if __name__ == "__main__":
    # 运行训练示例
    compare_planning_methods()
