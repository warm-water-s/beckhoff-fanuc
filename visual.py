# -*- coding:utf-8 -*-
"""
读取PLC采集的振动和电流数据，绘制三向振动波形（x、y、z方向）和三相电流波形
- 振动：各时刻10通道数据连续拼接为单条时序曲线（1-10→x，11-20→y，21-30→z）
- 电流：31-33通道分3个子图展示（A/B/C相）
已修复：实例属性调用遗漏self的问题，确保x_vibration_seq、current_A等变量正常定义
"""
import numpy as np
import matplotlib.pyplot as plt
import re
from datetime import datetime
import os

# 设置中文字体（兼容Windows/macOS）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

START = 80000
END = 81000

class DataPlotter:
    def __init__(self, file_path):
        self.file_path = file_path
        # 原始数据存储
        self.raw_data = []  # 每行=1个采样时刻的33通道数据（numpy数组）
        self.timestamps = []  # 采集时间戳列表
        # 处理后的数据（实例属性，必须通过self访问）
        self.x_vibration_seq = []  # x向振动：1-10通道连续拼接
        self.y_vibration_seq = []  # y向振动：11-20通道连续拼接
        self.z_vibration_seq = []  # z向振动：21-30通道连续拼接
        self.current_A = []       # A相电流：31通道（索引30）
        self.current_B = []       # B相电流：32通道（索引31）
        self.current_C = []       # C相电流：33通道（索引32）
    
    def read_data(self):
        """读取txt文件，提取时间戳和前33通道有效数据"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"数据文件不存在：{self.file_path}")
        
        with open(self.file_path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
        
        current_batch = []  # 临时存储一批数据（一个时间戳下的所有采样点）
        for line in lines:
            line = line.strip()
            # 1. 匹配时间戳行（格式：=== 采集时间: 2025-12-03 11:34:35 ===）
            timestamp_match = re.match(r'=== 采集时间: (.*) ===', line)
            if timestamp_match:
                # 若有未保存的批量数据，先存入raw_data
                if current_batch:
                    self.raw_data.extend(current_batch)
                    current_batch = []
                # 解析时间戳（兼容格式异常）
                try:
                    ts = datetime.strptime(timestamp_match.group(1), '%Y-%m-%d %H:%M:%S')
                    self.timestamps.append(ts)
                except:
                    self.timestamps.append(f"格式异常：{timestamp_match.group(1)}")
                continue
            
            # 2. 匹配数据行（制表符分隔的数字，取前33列有效数据）
            if line and not line.startswith('='):  # 跳过分隔线（===...===）
                try:
                    data = list(map(int, line.split('\t')[:33]))  # 只保留前33通道
                    if len(data) == 33:  # 确保数据完整性
                        current_batch.append(data)
                except:
                    continue  # 跳过格式异常的数据行
        
        # 保存最后一批未处理的数据
        if current_batch:
            self.raw_data.extend(current_batch)
        
        # 转换为numpy数组，方便后续处理
        self.raw_data = np.array(self.raw_data)
        print(f"✅ 数据读取完成")
        print(f"   - 总采样时刻数：{len(self.raw_data)}")
        print(f"   - 采集时间戳数：{len(self.timestamps)}")
        print(f"   - 单时刻数据维度：{self.raw_data.shape[1]} 通道")
    
    def process_data(self):
        """数据预处理：振动时序拼接 + 电流提取（核心修复self调用）"""
        if self.raw_data.size == 0:
            raise ValueError("❌ 请先调用read_data()读取数据，再执行数据处理")
        
        # 1. 振动信号：按时间顺序拼接每个时刻的10个通道数据
        for time_point in self.raw_data:
            self.x_vibration_seq.extend(time_point[0:10])   # x向：1-10通道（索引0-9）
            self.y_vibration_seq.extend(time_point[10:20])  # y向：11-20通道（索引10-19）
            self.z_vibration_seq.extend(time_point[20:30])  # z向：21-30通道（索引20-29）
        
        # 2. 电流信号：按采样时刻提取单个通道数据
        self.current_A = self.raw_data[:, 30]  # 31通道（索引30）
        self.current_B = self.raw_data[:, 31]  # 32通道（索引31）
        self.current_C = self.raw_data[:, 32]  # 33通道（索引32）
        
        # 修复：通过self访问实例属性，使用len()函数（更规范）
        print(f"\n✅ 数据处理完成")
        print(f"   - 振动时序长度：x={len(self.x_vibration_seq)}, y={len(self.y_vibration_seq)}, z={len(self.z_vibration_seq)}")
        print(f"   - 电流序列长度：A={len(self.current_A)}, B={len(self.current_B)}, C={len(self.current_C)}")
    
    def plot_vibration_waveforms(self):
        """绘制三向振动波形（连续时序，3个子图）"""
        if not self.x_vibration_seq:
            raise ValueError("❌ 请先调用process_data()处理数据，再绘制振动图")
        
        self.x_vibration_seq = self.x_vibration_seq[START:END]
        self.y_vibration_seq = self.y_vibration_seq[START:END]
        self.z_vibration_seq = self.z_vibration_seq[START:END]

        self.current_A = self.current_A[START:END]
        self.current_B = self.current_B[START:END]
        self.current_C = self.current_C[START:END]
        
        # 创建3行1列的子图布局，figsize控制画布大小
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=False)
        fig.suptitle('三向振动波形图（连续时序）', fontsize=18, fontweight='bold', y=0.98)
        
        # 振动时序轴：每个点对应1个采样值（总长度=采样时刻数×10）
        vibration_time = np.arange(len(self.x_vibration_seq))
        
        # —— X方向振动 ——
        axes[0].set_title('X方向振动（1-10通道连续拼接）', fontsize=14, pad=20)
        axes[0].plot(vibration_time, self.x_vibration_seq, color='#ff6b6b', linewidth=1.2, alpha=0.8)
        axes[0].set_ylabel('振幅', fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')  # 虚线网格，提升可读性
        axes[0].set_facecolor('#f8f9fa')  # 浅灰背景，减少视觉疲劳
        
        # —— Y方向振动 ——
        axes[1].set_title('Y方向振动（11-20通道连续拼接）', fontsize=14, pad=20)
        axes[1].plot(vibration_time, self.y_vibration_seq, color='#4ecdc4', linewidth=1.2, alpha=0.8)
        axes[1].set_ylabel('振幅', fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_facecolor('#f8f9fa')
        
        # —— Z方向振动 ——
        axes[2].set_title('Z方向振动（21-30通道连续拼接）', fontsize=14, pad=20)
        axes[2].plot(vibration_time, self.z_vibration_seq, color='#45b7d1', linewidth=1.2, alpha=0.8)
        axes[2].set_xlabel('振动采样点序号', fontsize=12)
        axes[2].set_ylabel('振幅', fontsize=12)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].set_facecolor('#f8f9fa')
        
        # 调整子图间距，避免标题/标签重叠
        plt.tight_layout()
        # 保存高清图片（300dpi，兼容印刷/报告）
        plt.savefig('三向振动波形图_连续时序.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"\n✅ 振动图已保存：三向振动波形图_连续时序.png")
    
    def plot_current_waveforms(self):
        """绘制三相电流波形（3个子图，分相展示）"""
        if len(self.current_A) == 0:
            raise ValueError("❌ 请先调用process_data()处理数据，再绘制电流图")
        
        # 创建3行1列的子图布局，与振动图风格统一
        fig, axes = plt.subplots(3, 1, figsize=(16, 12), sharex=True)
        fig.suptitle('三相电流波形图（采样时刻时序）', fontsize=18, fontweight='bold', y=0.98)
        
        # 电流时序轴：每个点对应1个采样时刻
        current_time = np.arange(len(self.current_A))
        
        # —— A相电流（31通道） ——
        axes[0].set_title('A相电流（31通道）', fontsize=14, pad=20)
        axes[0].plot(current_time, self.current_A, color='#ff9999', linewidth=1.2, alpha=0.8)
        axes[0].set_ylabel('电流值', fontsize=12)
        axes[0].grid(True, alpha=0.3, linestyle='--')
        axes[0].set_facecolor('#f8f9fa')
        
        # —— B相电流（32通道） ——
        axes[1].set_title('B相电流（32通道）', fontsize=14, pad=20)
        axes[1].plot(current_time, self.current_B, color='#66b3ff', linewidth=1.2, alpha=0.8)
        axes[1].set_ylabel('电流值', fontsize=12)
        axes[1].grid(True, alpha=0.3, linestyle='--')
        axes[1].set_facecolor('#f8f9fa')
        
        # —— C相电流（33通道） ——
        axes[2].set_title('C相电流（33通道）', fontsize=14, pad=20)
        axes[2].plot(current_time, self.current_C, color='#99ff99', linewidth=1.2, alpha=0.8)
        axes[2].set_xlabel('采样时刻序号', fontsize=12)
        axes[2].set_ylabel('电流值', fontsize=12)
        axes[2].grid(True, alpha=0.3, linestyle='--')
        axes[2].set_facecolor('#f8f9fa')
        
        plt.tight_layout()
        plt.savefig('三相电流波形图_分图.png', dpi=300, bbox_inches='tight', facecolor='white')
        plt.show()
        print(f"✅ 电流图已保存：三相电流波形图_分图.png")
    
    def run_all(self):
        """一键执行：读取→处理→绘图（简化用户操作）"""
        try:
            self.read_data()
            self.process_data()
            self.plot_vibration_waveforms()
            self.plot_current_waveforms()
            print(f"\n🎉 所有任务完成！图片已保存到当前目录")
        except Exception as e:
            print(f"\n❌ 程序出错：{str(e)}")

def main():
    # --------------------------
    # 关键：替换为你的数据文件路径
    # --------------------------
    file_path = "gvl_buffer_data1.txt"  # 若文件在子文件夹，需写完整路径（如"数据/gvl_buffer_data.txt"）
    
    # 初始化并执行
    plotter = DataPlotter(file_path)
    plotter.run_all()

if __name__ == "__main__":
    main()