import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np
import torch
import random
from PPO_Model.PPO import Agent, AgentGaussian


from envs.milling_env_ppo import Milling_env

# from envs.env2 import MillingSLDEnv
# from envs.env_three_lobe import MillingEnvLobe3
# from envs.env_lobe_3d_1029 import Milling_env


class Config:
    def __init__(self) -> None:
        self.env_name = "milling-env"  # 环境名字 连续动作
        self.new_step_api = False  # 是否用gym的新api
        self.algo_name = "PPO"  # 算法名字
        self.mode = "train"  # train or test
        self.seed = 1  # 随机种子
        self.device = "cuda" if torch.cuda.is_available() else "cpu"  # device to use
        self.train_eps = 3000  # 训练的回合数
        self.test_eps = 50  # 测试的回合数
        self.max_steps = 500  # 每个回合的最大步数
        self.eval_eps = 5  # 评估的回合数
        self.eval_per_episode = 10  # 评估的频率

        self.gamma = 0.99  # 折扣因子
        self.k_epochs = 10  # 更新策略网络的次数
        self.actor_lr = 3e-4  # actor网络的学习率
        self.critic_lr = 3e-4  # critic网络的学习率
        self.eps_clip = 0.2  # epsilon-clip
        self.entropy_coef = 0.03  # entropy的系数
        self.update_freq = 100  # 更新频率
        self.actor_hidden_dim = 256  # actor网络的隐藏层维度
        self.critic_hidden_dim = 256  # critic网络的隐藏层维度
        self.checkpoint_dir = "./save_path/"
        self.lam = 0.95  # GAE的参数
        self.batch_size = 1024  # 批大小


def env_agent_config(cfg, MAT_FOLDER):
    # 加载不同的铣削环境
    env = Milling_env(MAT_FOLDER)
    # env = MillingSLDEnv() # env2
    # env = MillingEnvLobe3(MAT_FOLDER)
    # env = Milling_env(MAT_FOLDER)

    # all_seed(env, seed=cfg.seed)
    n_states = 4  # 状态空间维度
    n_actions = 1  # 动作空间维度
    print(f"状态空间维度：{n_states}，动作空间维度：{n_actions}")

    # 更新n_states和n_actions到cfg参数中
    setattr(cfg, "n_states", n_states)
    setattr(cfg, "n_actions", n_actions)
    # agent = Agent(cfg) # 离散动作
    agent = AgentGaussian(cfg)  # 连续动作
    return env, agent


def smooth(data, weight=0.9):
    """用于平滑曲线,类似于Tensorboard中的smooth曲线"""
    last = data[0]
    smoothed = []
    for point in data:
        smoothed_val = last * weight + (1 - weight) * point  # 计算平滑值
        smoothed.append(smoothed_val)
        last = smoothed_val
    return smoothed


def plot_rewards(rewards, cfg, tag="train"):
    """画图"""
    sns.set_theme()
    plt.figure()  # 创建一个图形实例，方便同时多画几个图
    plt.title(f"{tag}ing curve on {cfg.device} of {cfg.algo_name} for {cfg.env_name}")
    plt.xlabel("epsiodes")
    plt.plot(rewards, label="rewards")
    plt.plot(smooth(rewards), label="smoothed")
    plt.legend()


def create_directory(path: str, sub_path_list: list):
    """
    创建目录及其子目录。

    参数:
        path (str): 基础路径。
        sub_path_list (list): 要创建的子目录列表。

    返回:
        None
    """
    for sub_path in sub_path_list:
        # 拼接基础路径和子目录路径
        full_path = path + sub_path
        # 检查路径是否存在
        if not os.path.exists(full_path):
            # 创建目录及其子目录
            os.makedirs(full_path, exist_ok=True)
            print("Path: {} create successfully!".format(full_path))
        else:
            print("Path: {} is already existence!".format(full_path))


def all_seed(env, seed=1):
    """万能的seed函数"""
    if seed == 0:  # seed=0时不设置（保留随机性）
        return
    # 1. 设置环境的随机种子
    env.seed(seed)  # 旧版gym API
    env.reset(seed=seed)  # 新版gym API（>=0.26.0）

    # 2. 设置Python和Numpy的随机种子
    np.random.seed(seed)
    random.seed(seed)

    # 3. 设置PyTorch的随机种子（CPU/GPU）
    torch.manual_seed(seed)  # config for CPU
    torch.cuda.manual_seed(seed)  # config for GPU

    # 4. 设置Python哈希种子（影响字典等数据结构的遍历顺序）
    os.environ["PYTHONHASHSEED"] = str(seed)  # config for python scripts

    # 5. 配置cuDNN（PyTorch的GPU后端）
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = False
