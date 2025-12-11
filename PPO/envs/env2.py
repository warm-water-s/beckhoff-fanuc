import numpy as np
from scipy.io import loadmat
from scipy.interpolate import interp1d
import random


# === 1. 载入 .mat（含 spindle speed n、极限切深 ap_lim） ===
raw = loadmat("./train_mat_files/stability_boundary_w922.mat")  # 加载文件
# 从加载的数据中提取主轴转速 spindle_speed 和极限切深 depth_of_cut，并将其转换为一维数组。
n_grid = raw["spindle_speed"].ravel()  # 形如 (M,)
ap_lim_g = raw["depth_of_cut"].ravel()  # 形如 (M,)

# 若 n 有重复（多叶瓣），先取每个 n 的最大极限
# 循环遍历唯一的主轴转速，找出每个转速对应的最大极限切深，存储在 uniq_n 和 max_ap 中。
uniq_n, max_ap = [], []
for n in np.unique(n_grid):
    mask = n_grid == n
    uniq_n.append(n)
    max_ap.append(ap_lim_g[mask].max())
uniq_n = np.array(uniq_n)
max_ap = np.array(max_ap)

# === 2. 用样条插值得到 ap_lim(n) 可微函数  ===
# 根据输入的主轴转速 n 计算对应的极限切深
ap_lim_fn = interp1d(
    uniq_n,
    max_ap,
    kind="cubic",  # 三次样条插值
    fill_value="extrapolate",  # 高速区不在原网格也能推
    assume_sorted=True,
)


class MillingSLDEnv():
    """
    observation = [n  , ap]               (rpm, mm)
    action      = [Δn , Δap]   ∈ [-1,1]^2  相对步长
    目标: 选最大稳定 ap
    """

    metadata = {"render_modes": []}

    def __init__(self, n_range=(6500.0, 8500.0), ap_range=(0.0, 2.0)):
        self.n_min, self.n_max = n_range
        self.ap_min, self.ap_max = ap_range
        self.ap_lim_fn = ap_lim_fn

        # 步长按物理范围的10%设置（归一化后约为0.1）
        self.step_scale = np.array([
            0.1 * (self.n_max - self.n_min),  # 转速步长
            0.1 * (self.ap_max - self.ap_min) # 切深步长
        ], dtype=np.float32) # type: ignore

        self.reset()

    def _normalize(self, n, ap):
        """将物理状态归一化到[0,1]"""
        n_norm = (n - self.n_min) / (self.n_max - self.n_min)
        ap_norm = (ap - self.ap_min) / (self.ap_max - self.ap_min)
        return np.array([n_norm, ap_norm], dtype=np.float32)

    def _denormalize(self, n_norm, ap_norm):
        """将归一化状态转回物理值"""
        n = n_norm * (self.n_max - self.n_min) + self.n_min
        ap = ap_norm * (self.ap_max - self.ap_min) + self.ap_min
        return np.array([n, ap], dtype=np.float32)

    # ---------------------------------------------------------
    def reset(self, *, seed=None, options=None):
        # super().reset(seed=seed)
        # 随机初始化物理状态（保守区间）
        self.state = np.array([
            self.n_min + random.uniform(0.05, 0.3) * (self.n_max - self.n_min),
            self.ap_min + random.uniform(0.05, 0.3) * (self.ap_max - self.ap_min)
        ], dtype=np.float32)
        self.steps = 0
        return self._normalize(*self.state)

    # ---------------------------------------------------------
    def step(self, action):
        # 1. 执行动作（在物理空间更新）
        action = np.clip(action, -1, 1)
        self.state += action * self.step_scale
        self.state[0] = np.clip(self.state[0], self.n_min, self.n_max)
        self.state[1] = np.clip(self.state[1], self.ap_min, self.ap_max)

        # 2. 计算稳定性指标
        n, ap = self.state
        ap_lim=float(self.ap_lim_fn(float(n)))
        ap_ratio  = ap / (ap_lim + 1e-6)        # 边界比

        # 3. 设计奖励函数（全部使用归一化值）
        n_norm, ap_norm = self._normalize(n, ap)
        # if ap_ratio <= 1.0:  # 稳定区域
        #     reward = n_norm * 0.6 + ap_norm * 0.4  # 加权和
        #     cost = 0
        # else:  # 颤振区域
        #     reward = - (ap_ratio + 1.0)  # 惩罚随超限程度增加
        #     cost = 1

        # ------- 奖励参数 --------
        # 决定最终想优先拉高 n 还是 ap；β<1 会鼓励先逼近边界再冲转速
        alpha   = 1.0      # 转速权重
        beta    = 0.5      # 切深权重 (→安全裕度)
        # 失稳惩罚：γ 控制斜率、δ 确保一旦颤振奖励一定为负
        gamma   = 2.0      # 颤振斜率
        delta   = 0.2      # 固定惩罚常数 (保证不稳定 reward<0)

        if ap_ratio <= 1.0:                           # 稳定
            reward = (n_norm ** alpha) * (ap_norm ** beta)
            cost   = 0
        else:                                  # 颤振
            reward = -gamma * (ap_ratio - 1.0) - delta
            cost   = 1

        
        # episode 终止：达到上限或最大步长
        self.steps += 1
        terminated = self.steps >= 2000  # 限定步数
        truncated = False
        info = {"ap_lim": ap_lim,
            "stable": ap_ratio <= 1,
            "cost": cost,
            "n_norm": n_norm,
            "ap_norm": ap_norm}
        return self._normalize(*self.state), reward, terminated, truncated, info
