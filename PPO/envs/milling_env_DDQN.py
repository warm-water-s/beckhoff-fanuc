"""铣削过程中的环境定义
    从叶瓣图文件夹中导入指定的mat文件,根据该文件进行训练
    铣削环境随机初始化ap和n,ae和fz固定
    状态空间为ap,ae,n,fz,动作空间为ap、n
    可以准确找到边界的不同位置,不过对于其他叶瓣图不适用,也就是说只学习到了数据而不是边界
"""

import numpy as np
import random
from scipy.interpolate import interp1d
import scipy.io


class StabilityChecker:
    def __init__(self, mat_file_path):
        """
        初始化稳定边界检查器（仅加载一次数据）

        参数:
            mat_file_path (str): .mat文件路径
        """
        # 1. 加载MAT文件数据
        mat_data = scipy.io.loadmat(mat_file_path)
        self.spindle_speed = mat_data["spindle_speed"].flatten()
        self.depth_of_cut = np.nan_to_num(mat_data["depth_of_cut"].flatten() * 1000)

        # 2. 按转速排序数据
        sort_idx = np.argsort(self.spindle_speed)
        self.spindle_speed = self.spindle_speed[sort_idx]
        self.depth_of_cut = self.depth_of_cut[sort_idx]

        # 3. 创建插值函数
        self._interp_func = interp1d(
            self.spindle_speed,
            self.depth_of_cut,
            kind="linear",
            fill_value=0,
            bounds_error=False,
        )

    def get_max_depth(self, spindle_speed):
        """
        查询指定转速下的最大允许切深

        参数:
            spindle_speed (float/np.array): 转速值(rpm)

        返回:
            float/np.array: 最大允许切深(m)
        """
        return self._interp_func(spindle_speed)

    def is_action_stable(self, spindle_speed, depth_of_cut):
        """
        检查给定切削参数是否稳定

        参数:
            spindle_speed (float/np.array): 转速(rpm)
            depth_of_cut (float/np.array): 切深(m)

        返回:
            bool/np.array: 是否稳定
        """
        return depth_of_cut <= self.get_max_depth(spindle_speed)

def linear_denormalize(x_normalized, target_min, target_max):
    """
    将0~1的值线性反归一化到[target_min, target_max]

    参数:
        x_normalized: 归一化后的数据 (0~1)
        target_min: 目标区间最小值
        target_max: 目标区间最大值

    返回:
        反归一化后的数据
    """
    return x_normalized * (target_max - target_min) + target_min


class milling_env:
    def __init__(self, mat_folder_path):
        self.step_n = 0
        self.checker=StabilityChecker(mat_folder_path)
        
    def reset(self):
        # 重置铣削环境
        # 如果随机重置标志为1，则随机生成参数
        self.step_n = 0
        ap = random.random()  # 将切深0~2mm归一化到0~1之间 1mm
        # ap = 0.5
        ae = 0.5  # 将切宽0~4mm归一化到0~1之间  2mm
        n = random.random()  # 将转速1000~10000r/min归一化到0~1之间
        # fz = random.random()  # 每齿进给量归一化到0~1之间,对应的真实值×2
        fz = 0.1
        state = np.array([ap, ae, n, fz])

        return state

    # 计算新的工艺参数
    def update_parameters(self, state, action):
        """
        根据当前参数 (ap、ae n fz) 和动作 action 计算下一时刻参数
        """
        # 动作策略
        action_value = 0
        ap_next = state[0]
        ap_next = state[0] + action[0] * 0.25
        if ap_next < 0 or ap_next > 1:
            action_value -= 100
        ap_next = np.clip(ap_next, 0, 1)

        ae_next = state[1]

        n_next = state[2] + action[1] * 0.25
        if n_next < 0 or n_next > 1:
            action_value -= 100
        n_next = np.clip(n_next, 0, 1)
        done = False

        f_next = state[3]
        return np.array([ap_next, ae_next, n_next, f_next]), action_value, done

    def calculate_boundary_value(self, state, action, action_value):
        # 计算给定动作和状态的价值
        ap, ae, n, fz = state
        # 计算给定动作和状态的奖励
        ap_real = linear_denormalize(ap, 0, 2)
        # ap_real = 1
        ae_real = ae * 4  # 2mm
        value = action_value
        n_real = int(linear_denormalize(n, 6500, 8500))
        fz_real = fz * 2  # 0.2mm
        MRR = ap_real * n_real  # 10~20000范围
        MRR = np.log10(MRR + 1)  # 对数归一化
        value += MRR - 0.01 * np.abs(np.mean(action))
        stable = self.checker.is_action_stable(n_real, ap_real)
        max_depth = self.checker.get_max_depth(n_real)
        if stable == False:  # 如果在叶瓣图之外，则不稳定
            value -= 100
        else:
            if ap_real<=max_depth:
                # 系数小于等于-3时,系统很容易发散;系数为-2时,衰减较强可能导致智能体保守
                # 系数为-1时,能够较好的找到边界
                r = np.exp(-1 * (max_depth - ap_real))  # 指数衰减
                value += r * 5
            else:
                value -= 500
        return value

    def step(self, state, action):
        self.step_n += 1
        state_next, action_value, done = self.update_parameters(state, action)
        value = self.calculate_boundary_value(state_next, action, action_value)
        done = False
        return state_next, value, done

