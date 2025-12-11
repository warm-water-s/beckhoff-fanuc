# stability_lookup.py
# 让 SAC 智能体在调整进给 F 和主轴转速 S 时，能通过查询 .mat 文件中的稳定性叶瓣图
# 在当前刚度和频率下，以当前 S 运行，我能安全使用的最大切深 ap 是多少？
# 如果我当前的切深超过了这个值 → 就是不稳定（颤振）→ 应该惩罚

import numpy as np
from scipy.io import loadmat
from scipy.interpolate import RegularGridInterpolator


# 加载 .mat 文件并封装查询接口
class StabilityLobeLookup:
    def __init__(self, mat_file="leafmap_3D.mat"):
        data = loadmat(mat_file)
        self.n_list = data["n_list"].flatten()  # rpm
        self.freq_list = data["freq_list"].flatten()  # Hz
        self.stiffness_list = data["stiffness_list"].flatten()
        self.ap_max = data["ap_max"]  # shape: (K, M, N)

        # 构建三维插值器：支持连续输入 (stiffness, freq, rpm)
        self.interp = RegularGridInterpolator(
            (self.stiffness_list, self.freq_list, self.n_list),
            self.ap_max,
            method="linear",
            bounds_error=False,
            fill_value=None,  # 超出范围时使用最近值
        )

    def get_max_stable_ap(self, stiffness, frequency, rpm):
        """
        查询在给定工况下的最大稳定切深
        """
        point = np.array([[stiffness, frequency, rpm]])
        return float(self.interp(point))

    def is_stable(self, stiffness, frequency, rpm, ap_cut):
        """
        判断当前切深是否稳定
        """
        max_ap = self.get_max_stable_ap(stiffness, frequency, rpm)
        return ap_cut <= max_ap

    def get_safety_margin(self, stiffness, frequency, rpm, ap_cut):
        """
        返回安全裕度：>0 表示安全，<0 表示失稳
        """
        max_ap = self.get_max_stable_ap(stiffness, frequency, rpm)
        return max_ap - ap_cut
