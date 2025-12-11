"""
    铣削过程中的环境定义
        读取一个文件夹中的mat文件,每个文件代表不同的频率
            每个回合随机选择一个mat文件,在固定ap、ae和fz的情况下寻找最优的n
            状态参数为: ap ae n fz freq
            动作参数为: n

"""

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
    def extract_frequency(self, mat_data, file_path):
        """从文件名中提取频率参数"""
        file_name = os.path.basename(file_path)
        
        # 使用正则表达式（更健壮）
        import re
        match = re.search(r'w(\d+)\.mat', file_name)
        if match:
            frequency = float(match.group(1))
            print(f"从文件名 {file_name} 中提取频率: {frequency} Hz")
            return frequency
        
        # 如果无法提取，使用默认值或从mat数据中读取
        print(f"警告: 无法从文件名 {file_name} 中提取频率，使用默认值")
        return 500  # 默认频率

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

                frequency = self.extract_frequency(mat_data, file_path)

                self.all_boundaries.append(
                    {
                        "speed": speed_unique,
                        "depth": depth_unique,
                        "frequency": frequency,
                        "file_name": os.path.basename(file_path),
                    }
                )
            except Exception as e:
                print(f"Error loading {file_path}: {str(e)}")

        if not self.all_boundaries:
            raise ValueError("No valid boundary data loaded")

        # 获取所有频率用于归一化
        self.all_frequencies = [boundary["frequency"] for boundary in self.all_boundaries]
        self.freq_min = min(self.all_frequencies)
        self.freq_max = max(self.all_frequencies)
        
        print(f"频率范围: {self.freq_min} - {self.freq_max} Hz")

        # 初始化当前边界
        self.current_boundary = None
        self.current_frequency = None
        self.load_random_boundary()

    def load_random_boundary(self):
        """随机加载一个叶瓣图边界"""
        self.current_boundary = random.choice(self.all_boundaries)
        self.current_frequency = self.current_boundary["frequency"]  # 保存当前频率
        print(f'current_boundary: {self.current_boundary["file_name"]},频率: {self.current_frequency}Hz')
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
            "file_name": self.current_boundary["file_name"], # type: ignore
            "frequency": self.current_boundary["frequency"], # type: ignore
            "speed_range": (
                min(self.current_boundary["speed"]), # type: ignore
                max(self.current_boundary["speed"]), # type: ignore
            ),
            "max_depth": max(self.current_boundary["depth"]), # type: ignore
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


def linear_normalize(x, original_min, original_max):
    """将原始值线性归一化到0~1"""
    return (x - original_min) / (original_max - original_min)

class Milling_env_1125:
    def __init__(self, mat_folder_path):
        self.checker = StabilityChecker(mat_folder_path)
        # 使用checker中的频率范围,用于频率归一化
        self.freq_min = self.checker.freq_min
        self.freq_max = self.checker.freq_max
        # 记录上一次状态的稳定性
        self.last_stable = True  # 初始化
        self.last_n_real = None

    def reset(self):
        # 重置铣削环境
        self.checker.load_random_boundary()

        ap = 0.5
        ae = 0.5  # 将切宽0~4mm归一化到0~1之间  2mm
        n = random.random()  # 将转速1000~10000r/min归一化到0~1之间
        fz = 0.1

        current_freq = self.checker.current_frequency
        freq_normalized = linear_normalize(current_freq, self.freq_min, self.freq_max)

        state = np.array([ap, ae, n, fz,freq_normalized])
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
        return np.array([ap_next, ae_next, n_next, f_next,state[4]])

    def calculate_boundary_value(self,state,action):
        """
            计算执行该动作的奖励，并决定是否真正执行（通过返回 next_state）
            返回: (next_state, reward)
        """
        # 1. 计算 tentative 下一状态
        tentative_next_state = self.update_parameters(state, action)

        # 2. 反归一化用于稳定性检查
        ap_real = 1 # 固定
        ae_real = tentative_next_state[1] * 4.0  # 2mm
        n_real = int(linear_denormalize(tentative_next_state[2], 6500, 8500))
        fz_real = tentative_next_state[3] * 0.2 # 假设 [0,1] -> [0,0.2]


        # 3. 检查稳定性（注意：checker 应支持输入真实物理值）
        is_stable = self.checker.is_action_stable(n_real, ap_real)

        # 4. 获取上一次的稳定性（用于判断是否刚越界）
        was_stable = self.last_stable

        reward = 0.0

        if not is_stable:
            # 不稳定：强惩罚
            reward = -100.0
            next_state = tentative_next_state  # 允许进入，但惩罚

        else:
             # 稳定：给予基础奖励
            MRR = ap_real * ae_real * n_real * fz_real  # 更合理的材料去除率公式
            MRR_norm = np.log10(MRR + 1)
            reward = MRR_norm - 0.1 * np.abs(np.mean(action))

            # === 关键：如果上一步不稳定，现在稳定了 → 说明退回边界内，奖励不高
            #     如果上一步稳定，现在也稳定 → 看是否在“试探边界”
            
            # 如果之前是稳定的，且这次动作很小（精细调整），说明在边界附近徘徊
            if was_stable:
                action_magnitude = np.abs(action).mean()
                if action_magnitude < 0.05:  # 小动作阈值（对应 Δn < ~100 rpm）
                    reward += 2.0  # 鼓励在边界附近微调

            # 可选：如果刚刚从不稳定恢复，轻微惩罚（避免震荡）
            if not was_stable:
                reward -= 1.0

            next_state = tentative_next_state

        # 更新历史信息
        self.last_stable = is_stable
        self.last_n_real = n_real

        return next_state, reward

    def step(self, state, action):
        state_next,value = self.calculate_boundary_value(state,action)
        done = False
        return state_next, value, done
