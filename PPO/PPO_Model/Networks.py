import torch.nn as nn
import torch.nn.functional as F
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 离散动作网络
class ActorSoftmax(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(ActorSoftmax, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        probs = F.softmax(self.fc3(x), dim=1)
        return probs


class ActorGaussian(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=64):
        super().__init__()
        self.shared = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_dim, hidden_dim),
            torch.nn.ReLU(),
        )
        self.mu = torch.nn.Linear(hidden_dim, output_dim)  # 均值分支
        self.std = torch.nn.Linear(hidden_dim, output_dim)  # 标准差分支

    def forward(self, state):
        features = self.shared(state)
        mu = torch.tanh(self.mu(features))  # 均值范围 [-1, 1]
        std = F.softplus(self.std(features)) + 1e-6  # 标准差 > 0
        return mu, std

    def save_checkpoint(self, checkpoint_file):
        """
        保存模型的检查点到指定文件。

        参数:
        - checkpoint_file (str): 保存检查点的文件名。
        """
        # 使用PyTorch的save函数保存模型的状态字典到指定文件
        # _use_new_zipfile_serialization=False 是为了兼容旧版本的PyTorch
        torch.save(
            self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False
        )

    def load_checkpoint(self, checkpoint_file):
        """
        从指定文件加载模型的检查点。

        参数:
        - checkpoint_file (str): 加载检查点的文件名。
        """
        # 使用PyTorch的load函数加载指定文件中的模型状态字典
        # map_location=device 确保模型被加载到正确的设备上（CPU或GPU）
        self.load_state_dict(torch.load(checkpoint_file, map_location=device))

    # def __init__(self, input_dim, output_dim, hidden_dim=256):
    #     super(ActorGaussian, self).__init__()
    #     self.fc1 = nn.Linear(input_dim, hidden_dim)
    #     self.fc2 = nn.Linear(hidden_dim, hidden_dim)
    #     self.fc_mu = nn.Linear(hidden_dim, output_dim)  # 输出动作的均值
    #     self.fc_std = nn.Linear(hidden_dim, output_dim)  # 输出动作的对数标准差

    # def forward(self, x):
    #     x = F.relu(self.fc1(x))
    #     x = F.relu(self.fc2(x))
    #     # torch.tanh 的输出范围是 [-1, 1]。乘以 2.0 后，mu 的范围是 [-2, 2]。
    #     c = self.fc_mu(x)
    #     # print("c:", c)
    #     mu = torch.tanh(c)
    #     # F.softplus 是一个平滑的激活函数，输出范围是 (0, +∞)。
    #     std = F.softplus(self.fc_std(x)) + 1e-6  # 防止标准差为0
    #     return mu, std


# 对于critic网络而言，输入是状态，输出是状态值函数V(s)或状态-动作值函数Q(s,a)的估计值。
class Critic(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim=256):
        super(Critic, self).__init__()
        assert output_dim == 1  # critic must output a single value
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        value = self.fc3(x)
        return value

    def save_checkpoint(self, checkpoint_file):
        torch.save(
            self.state_dict(), checkpoint_file, _use_new_zipfile_serialization=False
        )

    def load_checkpoint(self, checkpoint_file):
        self.load_state_dict(torch.load(checkpoint_file, map_location=device))
