import numpy as np
import scipy.io
import random
import os
from scipy.interpolate import RegularGridInterpolator 

# 加载StabilityLobeData_3D.mat数据
class StabilityLobeDataLoader:
    def __init__(self, file_path):
        self.data = scipy.io.loadmat(file_path)
        self.X = self.data['X'].flatten()  # 主轴转速（rpm），长度101
        self.Y = self.data['Y'].flatten()  # 固有频率（rad/s），长度160
        self.Z = self.data['Z']            # 最大切深（m），shape=(101, 160)

       # 使用RegularGridInterpolator进行二维插值
        self.interp_func = RegularGridInterpolator(
            (self.X, self.Y),
            self.Z,
            bounds_error=False,
            fill_value=0
        )


    def get_max_depth(self, n, frequency):
        """根据转速n和固有频率返回最大切深"""
        # 使用插值函数计算给定n和频率下的最大切深
        point = np.array([[n, frequency]])
        return self.interp_func(point)[0] * 1000  # 转为mm

# 强化学习环境
class MillingEnvLobe3:
    def __init__(self, mat_file_path):
        self.step_n = 0
        self.loader = StabilityLobeDataLoader(mat_file_path)
        self.n_range = (5000, 8000)  # 主轴转速范围（RPM）
        self.n_min, self.n_max = self.n_range
        self.freqs = self.loader.Y  # 离散频率集合

    def _normalize(self, value, min_val, max_val):
        """将物理状态归一化到[0,1]"""
        return (value - min_val) / (max_val - min_val)

    def _denormalize(self, x_norm, tar_min, tar_max):
        """将归一化状态转回物理值"""
        return x_norm * (tar_max - tar_min) + tar_min

    def reset(self):
        """重置环境"""
        self.step_n = 0
        # 随机选择一个固有频率
        self.freq = random.choice(self.freqs)
        self.freq = self._normalize(self.freq, self.freqs.min(), self.freqs.max()) # 频率的归一化
        # 固定的参数
        ae = 0.5 
        fz = 0.1
        ap = 0.15

        # 随机初始化主轴转速（n）
        n = self.n_min + random.uniform(0.3, 0.6) * (self.n_max - self.n_min)
        n = self._normalize(n, self.n_min, self.n_max)

        self.state = np.array([ap, ae, n, fz,self.optimal_frequency], dtype=np.float32)
        return self.state

    def update_parameters(self, state, action):
        """根据当前参数和动作计算下一时刻的状态"""
        action_value = 0

        n_next = state[2] + action[0] * 0.25  # 修改主轴转速（n）
        if n_next < 0 or n_next > 1:
            action_value -= 10
        n_next = np.clip(n_next, 0, 1)


        f_next = state[3]  # 每齿进给量（fz）保持不变
        done = False

        return np.array([state[0], state[1], n_next, f_next]), action_value, done

    def calculate_reward(self, state, action,action_value):
        """计算当前状态和动作的奖励"""
        ap, ae, n, fz = state
        ap_real = ap * 2
        n_real = int(self._denormalize(n, self.n_min, self.n_max))
        value = action_value
        
        # 计算材料去除率（MRR）
        MRR = ap_real * n_real
        MRR = np.log10(MRR + 1)  # 对数归一化
        value += MRR - 0.01 * np.abs(np.mean(action))

        # 随机选择一个固有频率，进行插值计算最大切深
        # self.step_n += 1
        # if self.step_n % 10 == 0:
        #     self.optimal_frequency = random.choice(self.loader.Y)  # 随机选择一个频率
        # optimal_frequency = random.choice(self.loader.Y)  # 随机选择一个频率
        max_depth = self.loader.get_max_depth(n_real, self.freq)

        # 奖励计算：优先考虑稳定性和最大切深
        reward = MRR  # 先加上MRR
        if ap_real <= max_depth:
            r = np.exp(-0.8 * (max_depth - ap_real))  # 指数衰减
            reward += r * 5
        else:
            reward -= 10  # 超出边界的惩罚

        return reward

    def step(self, state, action):
        """环境执行一步，并返回新的状态、奖励和是否结束"""
        state_next, action_value, done = self.update_parameters(state, action)
        reward = self.calculate_reward(state_next, action,action_value)
        done = False  # 强化学习中的结束条件（可以根据具体需求设定）

        state_next = np.concatenate([state_next[:4], [self.freq]])

        return state_next, reward, done

# # 环境使用示例
# mat_file_path = "path_to_StabilityLobeData_3D.mat"
# env = MillingEnv(mat_file_path)

# state = env.reset()
# print("Initial state:", state)

# action = np.array([0.1, 0.1])  # 假设动作，代表转速n的调整
# state_next, reward, done = env.step(state, action)

# print("Next state:", state_next)
# print("Reward:", reward)
