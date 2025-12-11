"""
1. 状态空间维度: 4
    n_norm: 归一化转速 (0-1)
    ap_norm: 归一化切深目标 (0-1)
    harmonic_ratio_norm: 归一化谐波比 (tanh归一化)
    stiff_norm: 当前的系统刚度倍率 (0-1),Agent 需要学会：刚度低时要保守，刚度高时可以激进。
2. 动作空间维度: 1

3. 奖励函数设计:
    惩罚机制:
        - 如果切削不稳定:
            梯度惩罚: reward = -min(2.0 + over_limit * 5.0, 10.0),超限越多，扣分越多
            死亡惩罚: 如果超限超过1.0mm,则done=True,并额外扣50分
    奖励机制:
        - 如果切削稳定: 当加工稳定时，奖励与转速挂钩，鼓励高效加工：
            指数级奖励: reward = 1.0 + (n_norm ** 2) * 4.0
                低转速 (1000 RPM)：得分约 1.0。
                高转速 (6000 RPM)：得分约 5.0。
        思路：通过平方项(**2),极大地拉开了高转速与低转速的得分差距(5倍之差)。
            这给 Agent 提供了巨大的动力去冒险跨越中速的不稳定区，追求高速区的丰厚回报。
    数值缩放:
        - 最终奖励会乘以0.1进行缩放，保持数值稳定

4. 实现的主要功能与策略
    - 3D 全工况模拟：利用 RegularGridInterpolator 对刚度、频率、转速进行三维插值，模拟了真实且复杂的铣削稳定性边界。
    - 打破局部最优 (Breaking Local Optima)
        三段式空降初始化：在 reset 时,按概率分布(60% 低速,20% 中速,20% 高速)随机初始化转速。
    - 大步长微调：动作幅度设为 0.05(约 250 转),使 Agent 能够一步跨过叶瓣图中狭窄的不稳定波谷。
    - 自适应策略学习：
通过上述设计，训练出的 Agent 能够根据当前的切深要求和机床状态（刚度/频率），智能地判断：
    - 在必死局（切深太大）中，退守低速区止损。
    - 在有希望的局中，大胆穿越不稳定区，锁定 5000-6000 RPM 的高效稳定区。
"""

import numpy as np
import random
import scipy.io
from scipy.interpolate import RegularGridInterpolator


class StabilityChecker3D:
    def __init__(self, mat_file_path):
        try:
            data = scipy.io.loadmat(mat_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {mat_file_path}")

        self.n_list = data["n_list"].flatten()
        self.freq_list = data["freq_list"].flatten()
        self.stiff_list = data["stiffness_list"].flatten()
        self.ap_max_grid = data["ap_max"]

        self.interpolator = RegularGridInterpolator(
            (self.stiff_list, self.freq_list, self.n_list),
            self.ap_max_grid,
            bounds_error=False,
            fill_value=None,
        )

        # 即使这里改了名，只要 env 里用对了就行
        self.n_min_data = self.n_list.min()
        self.n_max_data = self.n_list.max()
        self.freq_min = self.freq_list.min()
        self.freq_max = self.freq_list.max()
        self.stiff_min = self.stiff_list.min()
        self.stiff_max = self.stiff_list.max()

    def get_limit_ap(self, stiffness, frequency, rpm):
        point = np.array([stiffness, frequency, rpm])
        return float(self.interpolator(point))


class Milling_env_1211:
    def __init__(self, mat_file_path, max_steps=200):
        self.checker = StabilityChecker3D(mat_file_path)
        self.teeth = 2
        self.max_steps = max_steps
        self.step_count = 0

        # 转速范围 (1000-6000)
        self.n_min = 1000.0
        self.n_max = 6000.0

        # 目标切深范围
        self.ap_target_min = 0.8
        self.ap_target_max = 4.0

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
        return [seed]

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)

        self.step_count = 0

        # 1. 随机工况 (刚度和频率随机) —— 保持难度
        self.current_stiff = random.uniform(
            self.checker.stiff_min, self.checker.stiff_max
        )
        self.current_freq = random.uniform(self.checker.freq_min, self.checker.freq_max)
        self.target_ap = random.uniform(self.ap_target_min, self.ap_target_max)

        # 生成一个 0-1 的随机数
        rand_p = random.random()

        if rand_p < 0.2:
            # === A. 诱惑区 (20%)：直接空降高转速 ===
            # 范围扩大一点：4500-6000，让它更容易接轨
            self.current_n = random.uniform(4500.0, 6000.0)

        elif rand_p < 0.4:
            # === B. 挑战区 (20%)：填补之前的真空 ===
            # 这里的切削很不稳定，Agent 可能会死很多次
            # 但它必须见识过这里的环境，才能学会怎么跨过去
            self.current_n = random.uniform(2500.0, 4500.0)

        else:
            # === C. 舒适区 (60%)：安全启动 ===
            # 范围扩大一点：1000-2500，让它离中间区域更近一步
            self.current_n = random.uniform(self.n_min, 2500.0)
        return self._get_state()

    def _get_state(self):
        n_norm = (self.current_n - self.n_min) / (self.n_max - self.n_min)
        ap_norm = (self.target_ap - self.ap_target_min) / (
            self.ap_target_max - self.ap_target_min
        )

        if self.current_n > 1e-5:
            harmonic_ratio = (60 * self.current_freq) / (self.current_n * self.teeth)
        else:
            harmonic_ratio = 0.0
        harmonic_ratio_norm = np.tanh(harmonic_ratio * 0.1)

        stiff_norm = (self.current_stiff - self.checker.stiff_min) / (
            self.checker.stiff_max - self.checker.stiff_min
        )

        return np.array(
            [n_norm, ap_norm, harmonic_ratio_norm, stiff_norm], dtype=np.float32
        )

    def step(self, state, action):
        self.step_count += 1

        # 动作步长：2%
        n_range = self.n_max - self.n_min
        delta_n = action[0] * n_range * 0.05

        self.current_n = np.clip(self.current_n + delta_n, self.n_min, self.n_max)

        limit_ap = self.checker.get_limit_ap(
            self.current_stiff, self.current_freq, self.current_n
        )

        is_stable = self.target_ap <= limit_ap
        reward = 0.0
        done = False

        if not is_stable:
            over_limit = self.target_ap - limit_ap

            # 惩罚逻辑
            penalty = 2.0 + (over_limit * 5.0)
            reward = -min(penalty, 10.0)

            # 严重超限惩罚
            if over_limit > 1.0:
                done = True
                reward -= 50.0
        else:
            # ========================================================
            # 🔥 修改点：指数级奖励 🔥
            # ========================================================
            n_norm = (self.current_n - self.n_min) / (self.n_max - self.n_min)

            # 基础分 1.0
            # 效率分改成平方：高转速的分数会变得非常高
            # 例如：
            # n_norm = 0.1 (低速) -> 1.0 + 4 * 0.01 = 1.04 分
            # n_norm = 1.0 (高速) -> 1.0 + 4 * 1.00 = 5.00 分 (是低速的5倍！)
            reward = 1.0 + (n_norm**2) * 4.0

        # 奖励缩放
        reward = reward * 0.1

        if self.step_count >= self.max_steps:
            done = True

        return self._get_state(), reward, done
