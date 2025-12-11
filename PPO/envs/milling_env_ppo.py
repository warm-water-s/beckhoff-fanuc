"""铣削过程中的环境定义"""

import numpy as np
import random
from scipy.interpolate import interp1d
import scipy.io
import os


class MatFileLoader:
    @staticmethod
    def load_mat_files(folder_path):
        """加载文件夹中所有.mat文件的叶瓣图数据"""
        mat_files = []
        for file in os.listdir(folder_path):
            if file.endswith(".mat"):
                full_path = os.path.join(folder_path, file)
                mat_files.append(full_path)

        if not mat_files:
            raise ValueError(f"No .mat files found in {folder_path}")

        return mat_files


class StabilityChecker:
    def __init__(self, mat_folder_path):
        """
        初始化稳定边界检查器(加载文件夹中所有.mat文件)

        参数:
            mat_folder_path (str): 包含.mat文件的文件夹路径
        """
        mat_files = MatFileLoader.load_mat_files(mat_folder_path)
        self.all_boundaries = []

        for file_path in mat_files:
            try:
                mat_data = scipy.io.loadmat(file_path)
                speed = mat_data["spindle_speed"].flatten()
                depth = np.nan_to_num(
                    mat_data["depth_of_cut"].flatten() * 1000
                )  # 转换为mm
                # 排序后去重
                sorted_idx = np.argsort(speed)
                speed_sorted = speed[sorted_idx]
                depth_sorted = depth[sorted_idx]

                # 去重：保留最后一个出现的值（或第一个，看需求）
                speed_unique, idx = np.unique(speed_sorted, return_index=True)
                depth_unique = depth_sorted[idx]

                self.all_boundaries.append(
                    {
                        "speed": speed_unique,
                        "depth": depth_unique,
                        "file_name": os.path.basename(file_path),
                    }
                )
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        if not self.all_boundaries:
            raise ValueError("No valid boundary data loaded")

        # 初始化当前边界
        self.current_boundary = None
        self.load_random_boundary()

    def load_random_boundary(self):
        """随机加载一个叶瓣图边界"""
        self.current_boundary = random.choice(self.all_boundaries)
        print(f'current_boundary: {self.current_boundary["file_name"]}')
        self._interp_func = interp1d(
            self.current_boundary["speed"],
            self.current_boundary["depth"],
            kind="linear",
            fill_value=0,
            bounds_error=False,
        )

    def get_current_boundary_info(self):
        """获取当前边界信息"""
        return {
            "file_name": self.current_boundary["file_name"],
            "speed_range": (
                min(self.current_boundary["speed"]),
                max(self.current_boundary["speed"]),
            ),
            "max_depth": max(self.current_boundary["depth"]),
        }

    def get_max_depth(self, spindle_speed):
        """获取当前边界下最大允许切深(mm)"""
        return self._interp_func(spindle_speed)

    def is_action_stable(self, spindle_speed, depth_of_cut):
        """检查当前边界下是否稳定"""
        return depth_of_cut <= self.get_max_depth(spindle_speed)


def linear_denormalize(x_normalized, target_min, target_max):
    """将0~1的值线性反归一化到[target_min, target_max]"""
    return x_normalized * (target_max - target_min) + target_min


class Milling_env:
    def __init__(self, mat_folder_path):
        self.checker = StabilityChecker(mat_folder_path)

    def reset(self):
        # 重置铣削环境
        # ap = random.random()  # 将切深0~2mm归一化到0~1之间 1mm
        ap = 0.5
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
        ap_next = state[0]
        ae_next = state[1]
        n_next = state[2] + action[0] * 0.25
        n_next = np.clip(n_next, 0.0, 1.0)
        f_next = state[3]
        return np.array([ap_next, ae_next, n_next, f_next])

    def calculate_boundary_value(self, state, action):
        # 计算给定动作和状态的价值
        ap, ae, n, fz = state
        # 计算给定动作和状态的奖励
        # ap_real = linear_denormalize(ap, 0, 2)
        ap_real = 1
        ae_real = ae * 4  # 2mm
        value = 0
        n_real = int(linear_denormalize(n, 6500, 8500))
        # n_real = 5500
        fz_real = fz * 2  # 0.2mm
        done = False
        MRR = ap_real * n_real  # 10~20000范围
        MRR = np.log10(MRR + 1)  # 对数归一化
        value += MRR - 0.1 * np.abs(np.mean(action))
        stable = self.checker.is_action_stable(n_real, ap_real)
        max_depth = self.checker.get_max_depth(n_real)
        if stable == False:  # 如果在叶瓣图之外，则不稳定
            value -= 100
            done = True
        else:
            r = np.exp(-2 * (max_depth - ap_real))  # 指数衰减，d=0->r=1;d=2->r=0.018
            # r = 1
            value += r * 3
        return value

    def step(self, state, action):
        state_next = self.update_parameters(state, action)
        value = self.calculate_boundary_value(state_next, action)
        done = False
        return state_next, value, done
