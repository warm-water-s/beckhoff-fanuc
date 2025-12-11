import torch
import torch.distributions as distributions
import numpy as np
from PPO_Model.Networks import ActorSoftmax, Critic, ActorGaussian
from PPO_Model.ReplayBuffer import PGReplay
import torch.nn.functional as F



class Agent:
    def __init__(self, cfg) -> None:
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)

        # ActorSoftmax：为离散动作设计的策略网络
        self.actor = ActorSoftmax(cfg.n_states, cfg.n_actions, hidden_dim=cfg.actor_hidden_dim).to(self.device)
        # Critic：价值网络，用于估计状态值
        self.critic = Critic(cfg.n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(self.device)
        # 优化器
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay() # 经验回放池
        self.k_epochs = cfg.k_epochs  # 更新策略的迭代次数
        self.eps_clip = cfg.eps_clip  # PPO算法的剪切参数
        self.entropy_coef = cfg.entropy_coef  # 熵系数，用于增加探索性
        self.sample_count = 0
        self.update_freq = cfg.update_freq  # 更新频率
        self.batch_size = cfg.batch_size


    def sample_action(self, state):
        self.sample_count += 1
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(dim=0)
        # 计算动作的概率
        probs = self.actor(state)
        # 使用Categorical分布采样动作
        dist = distributions.Categorical(probs)
        # 采样动作
        action = dist.sample()
         # 计算log概率
        log_probs = dist.log_prob(action).detach()
        return (action.detach().cpu().numpy().item(),log_probs.detach().cpu().numpy().item())

    @torch.no_grad()
    def predict_action(self, state):
        state = torch.tensor(state, device=self.device, dtype=torch.float32).unsqueeze(0)
        probs = self.actor(state)
        dist = distributions.Categorical(probs)
        action = dist.sample()
        return action.detach().cpu().numpy().item()

    def update(self):
        # update policy every n steps
        if self.sample_count % self.update_freq != 0:
            return
        # print("update policy")

        # 从memory中采样数据
        old_states, old_actions, old_log_probs, old_rewards, old_dones = self.memory.sample(self.batch_size)

         # 将采样数据转化为tensor
        old_states = torch.tensor(np.array(old_states), device=self.device, dtype=torch.float32)
        old_actions = torch.tensor(np.array(old_actions), device=self.device, dtype=torch.float32)
        old_log_probs = torch.tensor(old_log_probs, device=self.device, dtype=torch.float32)
        
        # 计算蒙特卡洛估计的状态回报
        returns = []
        discounted_sum = 0
        for reward, done in zip(reversed(old_rewards), reversed(old_dones)):
            if done:
                discounted_sum = 0
            discounted_sum = reward + (self.gamma * discounted_sum)
            returns.insert(0, discounted_sum)

         # 对奖励进行归一化
        returns = torch.tensor(returns, device=self.device, dtype=torch.float32)
        returns = (returns - returns.mean()) / (returns.std() + 1e-5) # 防止除零

        # 使用k_epochs来更新多次
        for _ in range(self.k_epochs):
            # compute advantage
            values = self.critic(old_states)  # detach to avoid backprop through the critic
            advantage = returns - values.detach()
            # get action probabilities
            probs = self.actor(old_states)
            dist = distributions.Categorical(probs)
            # get new action probabilities
            new_probs = dist.log_prob(old_actions)
            # compute ratio (pi_theta / pi_theta__old):
            ratio = torch.exp(new_probs - old_log_probs)  # old_log_probs must be detached
            # compute surrogate loss
            surr1 = ratio * advantage
            surr2 = torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantage
            # compute actor loss
            actor_loss = (
                -torch.min(surr1, surr2).mean()
                + self.entropy_coef * dist.entropy().mean()
            )
            # compute critic loss
            critic_loss = (returns - values).pow(2).mean()
            # take gradient step
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()
        self.memory.clear()


class AgentGaussian:
    def __init__(self, cfg) -> None:
        self.gamma = cfg.gamma
        self.device = torch.device(cfg.device)
        self.actor = ActorGaussian(cfg.n_states, cfg.n_actions, hidden_dim=cfg.actor_hidden_dim).to(self.device)
        self.critic = Critic(cfg.n_states, 1, hidden_dim=cfg.critic_hidden_dim).to(self.device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=cfg.actor_lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=cfg.critic_lr)
        self.memory = PGReplay()  # 初始化经验回放池
        self.k_epochs = cfg.k_epochs
        self.eps_clip = cfg.eps_clip
        self.entropy_coef = cfg.entropy_coef
        self.sample_count = 0
        self.update_freq = cfg.update_freq
        self.checkpoint_dir = cfg.checkpoint_dir
        self.lam = cfg.lam
        self.batch_size = cfg.batch_size

    def sample_action(self, state):
        """与环境交互时：采样一个随机动作"""
        self.sample_count += 1
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std = self.actor(state_t)
        # print("mean:", mean, "std:", std)
        dist = distributions.Normal(mean, std)
        action_t = dist.rsample()  # 可微分的采样
        log_prob_t = dist.log_prob(action_t).sum(dim=-1).detach()

        # 转为 numpy
        action = action_t.detach().cpu().numpy().flatten()
        action = np.clip(action, -1, 1)

        return action, log_prob_t.item()

    @torch.no_grad()
    def predict_action(self, state):
        """测试时可用确定性动作( mean )，或者也可继续随机采样"""
        state_t = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
        mean, std = self.actor(state_t)
        action = mean.detach().cpu().numpy().flatten()
        action = np.clip(action, -1, 1)
        return action


    def compute_gae(self, rewards, values, dones):
        """
        计算广义优势估计(GAE)

        参数:
            rewards (list or np.ndarray): 当前批次的环境奖励，形状为 [T](T是轨迹长度)
            values (torch.Tensor): Critic网络输出的状态价值,形状为 [T]
            dones (list or np.ndarray): 是否终止的标记，形状为 [T]
            gamma (float): 折扣因子
            lam (float): GAE的λ参数(权衡偏差和方差)

        返回:
            advantages (torch.Tensor): 计算后的GAE优势函数,形状为 [T]
            targets (torch.Tensor): 计算后的GAE目标值(GAE+V(s)),形状为 [T,1]
        """
        rewards = torch.tensor(rewards, device=self.device, dtype=torch.float32).view(-1, 1)
        dones = torch.tensor(dones, device=self.device, dtype=torch.float32).view(-1, 1)
        values = values.detach()  # 避免计算梯度

        T = len(rewards)
        advantages = torch.zeros_like(rewards)
        last_advantage = 0

        # 反向计算 GAE
        for t in reversed(range(T)):
            if t == T - 1:
                next_non_terminal = 1.0 - dones[t]
                next_value = 0  # 终止状态的后继价值为 0
            else:
                next_non_terminal = 1.0 - dones[t]
                next_value = values[t + 1]

            # TD 误差 δ_t = r_t + γ * V(s_{t+1}) * (1 - done) - V(s_t)
            delta = rewards[t] + self.gamma * next_value * next_non_terminal - values[t]
            advantages[t] = last_advantage = (delta + self.gamma * self.lam * next_non_terminal * last_advantage)

        targets = advantages + values  # GAE + V(s)
        return advantages, targets

    def update(self):
        """PPO核心更新"""
        if self.sample_count % self.update_freq != 0:
            return

        states, actions, old_log_probs, rewards, dones = self.memory.sample(self.batch_size)
        states = torch.tensor(np.array(states), dtype=torch.float32, device=self.device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32, device=self.device)
        old_log_probs = torch.tensor(np.array(old_log_probs), dtype=torch.float32, device=self.device)

        # 使用GAE计算优势函数
        with torch.no_grad():
            values = self.critic(states)
            advantages, targets = self.compute_gae(rewards, values, dones)

        # 标准化优势函数
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(self.k_epochs):
            # 更新Critic
            current_values = self.critic(states)
            critic_loss = F.mse_loss(current_values, targets)

            # 更新Actor
            mean, std = self.actor(states)
            dist = distributions.Normal(mean, std)
            new_log_probs = dist.log_prob(actions).sum(dim=-1)
            entropy = dist.entropy().mean()

            # 计算比率
            log_ratio = new_log_probs - old_log_probs
            ratio = torch.exp(log_ratio.clamp(max=20))
            surr1 = ratio * advantages.squeeze()
            surr2 = (torch.clamp(ratio, 1 - self.eps_clip, 1 + self.eps_clip) * advantages.squeeze())

            actor_loss = -torch.min(surr1, surr2).mean() - self.entropy_coef * entropy

            # 梯度更新
            self.actor_optimizer.zero_grad()
            self.critic_optimizer.zero_grad()
            actor_loss.backward()
            critic_loss.backward()
            self.actor_optimizer.step()
            self.critic_optimizer.step()

        self.memory.clear()

    def save_models(self):
        """
        保存模型

        参数:
        返回:
        - 无返回值，但会将模型的检查点保存到指定的目录中。
        """
        # 保存Actor网络的检查点
        self.actor.save_checkpoint(self.checkpoint_dir + "./Actor/Actor.pth")
        print("Saving actor network successfully!")

        # 保存Critic网络的检查点
        self.critic.save_checkpoint(self.checkpoint_dir + "./Critic/Critic.pth")
        print("Saving critic network successfully!")

    def load_models(self):
        """
        从指定的检查点文件加载模型。

        参数:

        返回:
        - 无返回值，但会加载模型并打印加载成功的消息。
        """
        # 加载Actor网络的检查点
        self.actor.load_checkpoint(self.checkpoint_dir + "./Actor/Actor.pth")
        print("Loading actor network successfully!")

        # 加载Critic网络的检查点
        self.critic.load_checkpoint(self.checkpoint_dir + "./Critic/Critic.pth")
        print("Loading critic1 network successfully!")
