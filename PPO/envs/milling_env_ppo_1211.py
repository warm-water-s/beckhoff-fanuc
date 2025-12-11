"""
    1. 引入了 harmonic_ratio(谐波比 = 频率/转速比）。
        作用： 这是这段代码的“灵魂”。稳定性叶瓣图(SLD)虽然随频率变化，但其形状是周期性的。
    通过计算谐波比，你实际上是将不同频率的叶瓣图“对齐”了。神经网络不需要死记硬背具体的转速，
    而是学习“在谐波比为 0.5 的倍数附近通常不稳定”这样的通用物理规律。

    2. reset 函数中随机化 target_ap(切深)并将其放入 State。
        
    3. 有梯度的奖励函数(Shaped Reward):
        reward = -5.0 - (over_limit * 10.0)。

    4. 符合机床特性的动作空间：
        delta_n 限制在 10% 范围内。
        模拟了主轴的物理惯性，防止 Agent 输出类似 1000 -> 9000 -> 2000 
        这种在现实中会损坏电机的抖动指令，生成的策略更平滑、可用。
"""
import numpy as np
import random
import scipy.io
from scipy.interpolate import RegularGridInterpolator

class StabilityChecker3D:
    def __init__(self, mat_file_path):
        """
        加载 3D 叶瓣图数据 (刚度 x 频率 x 转速)
        """
        try:
            data = scipy.io.loadmat(mat_file_path)
        except FileNotFoundError:
            raise FileNotFoundError(f"找不到文件: {mat_file_path}")

        # 1. 提取坐标轴数据
        # flatten() 确保是一维数组
        self.n_list = data['n_list'].flatten()            # RPM 轴 (Z轴)
        self.freq_list = data['freq_list'].flatten()      # 频率 轴 (Y轴)
        self.stiff_list = data['stiffness_list'].flatten() # 刚度 轴 (X轴)
        
        # 2. 提取 3D 数据矩阵
        # 形状应该是 (len(stiff), len(freq), len(n))
        self.ap_max_grid = data['ap_max']
        
        # 3. 创建 3D 插值器 (核心)
        # 这样我们就可以查询任意 (stiffness, freq, rpm) 点的极限切深
        # bounds_error=False, fill_value=None 表示超出范围时使用最近邻或外推
        self.interpolator = RegularGridInterpolator(
            (self.stiff_list, self.freq_list, self.n_list), 
            self.ap_max_grid,
            bounds_error=False, 
            fill_value=None 
        )

        # 记录边界用于归一化
        self.n_min, self.n_max = self.n_list.min(), self.n_list.max()
        self.freq_min, self.freq_max = self.freq_list.min(), self.freq_list.max()
        self.stiff_min, self.stiff_max = self.stiff_list.min(), self.stiff_list.max()
        
        # 记录最大可能的切深用于归一化
        self.global_ap_max = self.ap_max_grid.max()

    def get_limit_ap(self, stiffness, frequency, rpm):
        """
        查询特定工况下的极限切深
        """
        point = np.array([stiffness, frequency, rpm])
        return float(self.interpolator(point))

class Milling_env_1211:
    def __init__(self, mat_file_path):
        # 初始化 3D 检查器
        self.checker = StabilityChecker3D(mat_file_path)
        
        self.teeth = 2  # 刀齿数
        self.max_steps = 100
        self.step_count = 0
        
        # 定义 Context (上下文) 范围
        self.ap_target_min = 0.8
        self.ap_target_max = 6.5 # mm

    def reset(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)
            
        self.step_count = 0
        
        # ===================================================
        # 1. 随机生成环境上下文 (Context)
        # ===================================================
        # 随机选择一个工况（刚度、频率）
        # 这里使用均匀分布，意味着 Agent 会遇到网格之间“未见过”的数据点，测试泛化能力
        self.current_stiff = random.uniform(self.checker.stiff_min, self.checker.stiff_max)
        self.current_freq = random.uniform(self.checker.freq_min, self.checker.freq_max)
        
        # 随机生成目标切深
        self.target_ap = random.uniform(self.ap_target_min, self.ap_target_max)
        
        # ===================================================
        # 2. 随机初始化状态
        # ===================================================
        self.current_n = random.uniform(self.checker.n_min, self.checker.n_max)
        
        return self._get_state()

    def _get_state(self):
        """
        构造 4维 状态向量:
        1. 转速 (Normalized)
        2. 目标切深 (Normalized)
        3. 谐波比率 (Physics Feature)
        4. 刚度倍率 (Normalized) -> 新增！因为刚度影响稳定区的“高度”
        """
        # 1. 转速归一化
        n_norm = (self.current_n - self.checker.n_min) / (self.checker.n_max - self.checker.n_min)
        
        # 2. 目标切深归一化
        ap_norm = (self.target_ap - self.ap_target_min) / (self.ap_target_max - self.ap_target_min)
        
        # 3. 物理特征：谐波比率
        # Ratio = (60 * f) / (n * N)
        if self.current_n > 1e-5:
            harmonic_ratio = (60 * self.current_freq) / (self.current_n * self.teeth)
        else:
            harmonic_ratio = 0.0
        # 简单的缩放防止数值过大
        harmonic_ratio_norm = np.tanh(harmonic_ratio * 0.1) 
        
        # 4. 刚度归一化 (新增)
        stiff_norm = (self.current_stiff - self.checker.stiff_min) / (self.checker.stiff_max - self.checker.stiff_min)
        
        # 返回 4维 状态
        return np.array([n_norm, ap_norm, harmonic_ratio_norm, stiff_norm], dtype=np.float32)


    def step(self, state, action):
        """
        兼容性接口：尽管本环境内部维护了 self.current_n，不需要外部传入 state，
        但为了保持与其他环境接口一致，这里必须接收 state 参数。
        """
        self.step_count += 1
        
        # 1. 执行动作
        # 注意：这里我们依然使用 self.current_n (内部状态)，而不是传入的 state
        # 这样可以保证环境的封闭性和物理连续性
        n_range = self.checker.n_max - self.checker.n_min
        delta_n = action[0] * n_range * 0.1
        
        self.current_n = np.clip(
            self.current_n + delta_n, 
            self.checker.n_min, 
            self.checker.n_max
        )
        
        # 2. 获取当前物理限制
        limit_ap = self.checker.get_limit_ap(
            self.current_stiff, 
            self.current_freq, 
            self.current_n
        )
        
        # 3. 计算奖励
        is_stable = self.target_ap <= limit_ap
        reward = 0.0
        done = False
        
        if not is_stable:
            over_limit = self.target_ap - limit_ap
            reward = -5.0 - (over_limit * 10.0)
        else:
            n_norm = (self.current_n - self.checker.n_min) / (self.checker.n_max - self.checker.n_min)
            reward = 1.0 + n_norm * 2.0

        # 4. 终止条件
        if self.step_count >= self.max_steps:
            done = True
        
        # 保持原来的返回格式
        return self._get_state(), reward, done

    def seed(self, seed=None):
        if seed is not None:
            random.seed(seed)
            np.random.seed(seed)