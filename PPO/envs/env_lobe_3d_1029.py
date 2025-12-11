"""铣削过程中的环境定义 - 使用 3D 叶瓣图，并将 stiffness 和 freq 作为状态输入"""

import numpy as np
import random
from scipy.interpolate import interp1d
import os
import sys


# 获取当前文件所在目录的上一级目录
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# 将上一级目录加入 Python 模块搜索路径
sys.path.append(parent_dir)
# 导入你提供的 3D 叶瓣图查询类
from stability_lookup import StabilityLobeLookup


# 预定义合理的工艺参数组合（可来自工艺手册）
valid_combinations = [
    (0.5, 1.0),  # 精加工
    (1.0, 2.0),  # 半精加工
    (1.5, 3.0),  # 粗加工
    (2.0, 4.0),  # 重切削
]


def linear_denormalize(x_normalized, target_min, target_max):
    """将0~1的值线性反归一化到[target_min, target_max]"""
    return x_normalized * (target_max - target_min) + target_min


def min_max_normalize(data, min_val, max_val):
    """
    将数据归一化到 [0, 1] 范围
    """
    return (data - min_val) / (max_val - min_val)


class Milling_env:
    def __init__(self, mat_file_path):
        """
        初始化铣削环境

        参数:
            mat_file_path (str): 3D 叶瓣图 .mat 文件路径
        """
        self.lobe_lookup = StabilityLobeLookup(mat_file_path)

        # 转速反归一化范围（与原代码一致）
        self.S_min = 2500
        self.S_max = 8500

        # stiffness 和 freq 的归一化范围
        self.stiff_min = self.lobe_lookup.stiffness_list.min()
        self.stiff_max = self.lobe_lookup.stiffness_list.max()
        self.freq_min = self.lobe_lookup.freq_list.min()
        self.freq_max = self.lobe_lookup.freq_list.max()

        self.fz_real_min = 0.01
        self.fz_real_max = 0.2

    def reset(self):
        """重置环境：随机选择当前工件的刚度和频率，并归一化加入状态"""
        # 随机选择当前系统的刚度和频率
        self.current_stiffness = np.random.choice(self.lobe_lookup.stiffness_list)
        self.current_freq = np.random.choice(self.lobe_lookup.freq_list)

        n = random.random()  # 转速归一化（6500~8500 rpm）
        n_real = linear_denormalize(n, self.S_min, self.S_max)

        # 随机选择一组合理的 ap, ae,一个episode中的ap,ae保持不变
        self.ap_real, self.ae_real = random.choice(valid_combinations)
        ap = min_max_normalize(self.ap_real, 0.5, 2.0)
        ae = min_max_normalize(self.ae_real, 1.0, 4.0)

        fz_real_init = random.uniform(self.fz_real_min, self.fz_real_max)
        fz = min_max_normalize(
            fz_real_init, self.fz_real_min, self.fz_real_max
        )  # → [0,1]

        # === 将 stiffness 和 freq 归一化并加入状态 ===
        stiff_norm = (self.current_stiffness - self.stiff_min) / (
            self.stiff_max - self.stiff_min
        )
        freq_norm = (self.current_freq - self.freq_min) / (
            self.freq_max - self.freq_min
        )

        # 状态维度：[ap, ae, n, fz, stiffness_norm, freq_norm]
        state = np.array([ap, ae, n, fz, stiff_norm, freq_norm])
        return state

    def update_parameters(self, state, action):
        """
        根据当前参数和动作更新下一时刻参数
        注意：stiffness 和 freq 保持不变（环境属性）
        """
        ap_next = state[0]
        ae_next = state[1]
        n_next = np.clip(state[2] + action[0] * 0.05, 0.0, 1.0)
        fz_next = np.clip(state[3] + action[1] * 0.05, 0.0, 1.0)
        stiff_norm_next = state[4]  # 不变
        freq_norm_next = state[5]  # 不变

        return np.array(
            [ap_next, ae_next, n_next, fz_next, stiff_norm_next, freq_norm_next]
        )

    def calculate_boundary_value(self, state, action):
        """计算奖励，逻辑与原代码完全一致，仅更换叶瓣图查询方式"""
        ap, ae, n, fz, stiff_norm, freq_norm = state

        # === 反归一化 stiffness 和 freq ===
        stiffness = stiff_norm * (self.stiff_max - self.stiff_min) + self.stiff_min
        freq = freq_norm * (self.freq_max - self.freq_min) + self.freq_min

        # === 其他参数反归一化 ===
        # 使用状态中传入的 ap 还原真实值
        ap_real = linear_denormalize(ap, 0.5, 2.0)  # 或者用你定义的范围
        ae_real = linear_denormalize(ae, 1.0, 4.0)
        n_real = int(linear_denormalize(n, self.S_min, self.S_max))  # 6500~8500 rpm
        fz_real = linear_denormalize(fz, self.fz_real_min, self.fz_real_max)

        # === 材料去除率 MRR（原逻辑）===
        MRR = ap_real * n_real * fz_real * ae_real
        MRR_norm = np.log10(MRR + 1)
        value = MRR_norm - 0.1 * np.abs(np.mean(action))

        # === 稳定性判断（使用 3D 叶瓣图）===
        stable = self.lobe_lookup.is_stable(stiffness, freq, n_real, ap_real)
        max_depth = self.lobe_lookup.get_max_stable_ap(stiffness, freq, n_real)

        if not stable:
            value -= 50
        else:
            r = np.exp(-2 * (max_depth - ap_real))
            value += r * 3

        if np.isnan(value):
            value = -1.0
        done = False
        return value, done

    def step(self, state, action):
        """执行一步"""
        state_next = self.update_parameters(state, action)
        reward, done = self.calculate_boundary_value(state_next, action)
        return state_next, reward, done
