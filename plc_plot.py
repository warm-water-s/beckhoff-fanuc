# -*- coding:utf-8 -*-
"""
    基于 tkinter 和 pyads 库的 实时数据采集、处理和可视化系统,用于监测来自倍福PLC的振动和电流数据
    1、系统架构与连接(ADS通讯)
        GUI 界面： 使用 tkinter 构建了一个图形用户界面，包含连接配置、数据采集控制、系统日志和两个实时绘图区域。
        PLC 连接： 使用 pyads 库通过 ADS 协议 连接到 PLC (Plc_port_open 方法)，读取 PLC 内存中预定的数据块。
            读取的数据包括：
                GvlBuffer:原始数据缓冲区(80*100 = 8000个INT)。
                GvlIndexBuffer:每个通道的写入指针/索引(80个INT)。
            read_data_atomic 方法确保了数据和索引是在一次操作中安全读取，保持数据一致性。
    2、核心数据处理(process_data)——负责将原始的 PLC 内存数据转换为可用的时序数据
        环形缓冲区重组 (时序恢复)：由于 PLC 将数据写入一个环形缓冲区，数据在内存中的物理顺序是错乱的。
        代码利用每个通道的写入指针 (write_ptr)，将数据块重新拼接 (part1 + part2),确保了33个有效通道的时序连续性。
        数据分离与整合:将重组后的连续数据分成两大部分：
            振动数据 (Vibration Data):前30个通道 (VIBRATION_CHANNELS)。进一步将30个通道的数据按每10个通道进行
                扁平化 (flatten) 拼接，形成 X、Y、Z 三个方向的完整振动波形数据 (每个方向10*100 = 1000个点)。
            电流数据 (Current Data):后3个通道 (CURRENT_CHANNELS)。
    3. 数据存储与缓存
        振动数据缓存： 振动数据是瞬时波形,最新的1000个点直接存入 self.vib_data,并计算对应的时间轴。
        电流数据缓存： 电流数据用于绘制趋势图,新的100个点追加到历史数据列表 (self.current_x_data 和 self.current_y_data) 中。
        通过限制数据长度 (`MAX_POINTS_TO_SHOW = 50*100点)，实现了滑动窗口/历史趋势效果。
        文件保存 (save_data_to_file):将每次采集到的数据（包括振动和电流）以文本格式追加保存到文件中，方便后续分析。
    4. 实时可视化
        实时监测循环:realtime_monitor_loop 通过 tkinter.after 实现定时循环读取、处理和更新图表。
        振动波形图 (update_vibration_plot)：在同一张图上绘制 X、Y、Z 三个方向的瞬时波形 (最新1000个点)。
            实现了动态 Y 轴调整:根据当前数据的最大值和最小值,自动添加15%的裕量 (PLOT_Y_MARGIN) 来设置 Y 轴范围，防止波形被截断。
        电流趋势图 (update_current_plot)：绘制 A、B、C 三相电流的历史趋势。同样实现了动态 Y 轴调整。
"""
import pyads
import tkinter
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.animation import FuncAnimation

# 日志默认条数
LOG_LINE_NUM = 0
# 全局PLC连接对象
Plc = None

# ========== 通道配置 ==========
TOTAL_CHANNELS = 33              # 总有效通道数 (实际使用)
VIBRATION_CHANNELS = 30          # 振动通道数（前30个）
CURRENT_CHANNELS = 3             # 电流通道数（后3个）
VIBRATION_GROUP_SIZE = 10        # 每10个通道合成一个振动方向 (X, Y, Z)
SAMPLE_COUNT = 100               # 每个通道采样点数
SAMPLING_FREQUENCY = 10000       # 采样频率10000Hz

# GVL Buffer 配置 (%MB0: ARRAY[1..8000] OF INT)
FULL_CHANNELS = 80               # PLC实际分配的通道数
FULL_BUFFER_LENGTH = FULL_CHANNELS * SAMPLE_COUNT # 80x100 = 8000个INT
GVL_BUFFER_DATATYPE = pyads.PLCTYPE_INT # 数组元素类型为INT
GVL_BUFFER_GROUP = 0x4020        # %MB对应的indexgroup
GVL_BUFFER_OFFSET = 0x0          # %MB0 偏移量

# GVL Index Buffer 配置 (%MB16000: ARRAY[1..80] OF INT)
INDEX_BUFFER_OFFSET = 16000      
INDEX_BUFFER_LENGTH = FULL_CHANNELS # 80个通道的索引
INDEX_BUFFER_DATATYPE = pyads.PLCTYPE_INT

# 默认连接参数
DEFAULT_AMS_NETID = "5.136.192.215.1.1"
DEFAULT_PORT = "851"
DEFAULT_INTERVAL_MS = "10" 

# 绘图配置
PLOT_HISTORY_LENGTH = 50         # 绘图显示的历史数据周期数（50个采样周期）
VIB_PLOT_POINTS = 1000           # 振动波形显示点数（10通道×100点）

# 绘图纵坐标裕量 (防止波形贴边或被截断)
PLOT_Y_MARGIN = 0.15 # 15% 裕量

class GUI():
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value="gvl_buffer_data.txt")
        
        # 绘图数据缓存
        self.current_x_data = []                            
        self.current_y_data = [[] for _ in range(CURRENT_CHANNELS)]
        
        self.vib_data = {
            'X': np.array([]),
            'Y': np.array([]),
            'Z': np.array([]),
            'time': np.array([])
        }
        
        self.sample_index = 0
        self.ani = None
        self.is_realtime_running = False

    def set_init_window(self):
        # 窗口基础设置
        self.init_windows_name.title('ADS 通讯 - 振动电流监测系统 (Frame 隔离布局优化)')
        self.init_windows_name.geometry('1400x800+30+30')
        self.init_windows_name.attributes('-alpha', 0.95)
        
        # ***布局优化：调整 grid 权重***
        self.init_windows_name.grid_columnconfigure(2, weight=1) # 右侧绘图区可拉伸
        
        # 右侧：Row 0-3 均分空间给两个波形图
        self.init_windows_name.grid_rowconfigure(0, weight=1)   
        self.init_windows_name.grid_rowconfigure(1, weight=1)   
        self.init_windows_name.grid_rowconfigure(2, weight=1)   
        self.init_windows_name.grid_rowconfigure(3, weight=1) 
        
        # 1. ========== 左侧：主容器 Frame (隔离行高影响) ==========
        self.left_frame = tkinter.Frame(self.init_windows_name)
        # 让 left_frame 跨越主窗口的 Row 0 到 Row 3，占据整个左侧垂直空间
        # 这样，左侧组件的布局将只由 left_frame 内部的权重控制
        self.left_frame.grid(row=0, column=0, columnspan=2, rowspan=4, padx=10, pady=5, sticky="nsew")
        
        # 配置 left_frame 内部的行权重
        # Row 4 (日志文本框行) 吸收所有剩余空间，将上方的组件推到顶部
        self.left_frame.grid_rowconfigure(0, weight=0) # ADS 连接
        self.left_frame.grid_rowconfigure(1, weight=0) # 数据采集
        self.left_frame.grid_rowconfigure(2, weight=0) # 文件/维护
        self.left_frame.grid_rowconfigure(3, weight=0) # Log Label
        self.left_frame.grid_rowconfigure(4, weight=1) # Log Text，**核心：吸收所有垂直剩余空间**
        
        # 2. ========== 左侧：操作控制栏 (紧凑排列，父级: left_frame) ==========
        
        # 2.1. ADS 连接配置组 (Row 0)
        frame_conn = tkinter.LabelFrame(self.left_frame, text="ADS 连接配置", padx=5, pady=5)
        frame_conn.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")
        
        tkinter.Label(frame_conn, text='AmsNetID').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.netID_text = tkinter.Text(frame_conn, width=20, height=1)
        self.netID_text.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.netID_text.insert(tkinter.END, DEFAULT_AMS_NETID)
        
        tkinter.Label(frame_conn, text='Port').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.port_text = tkinter.Text(frame_conn, width=20, height=1)
        self.port_text.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        self.port_text.insert(tkinter.END, DEFAULT_PORT)
        
        self.open_port_button = tkinter.Button(frame_conn, text='打开端口', command=self.Plc_port_open)
        self.open_port_button.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        
        # 2.2. 数据采集控制组 (Row 1) 
        frame_data = tkinter.LabelFrame(self.left_frame, text="数据采集控制", padx=5, pady=5)
        frame_data.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew") 
        
        tkinter.Label(frame_data, text='读取间隔(ms)').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.interval_text = tkinter.Text(frame_data, width=15, height=1)
        self.interval_text.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        self.interval_text.insert(tkinter.END, DEFAULT_INTERVAL_MS)
        
        self.read_data_button = tkinter.Button(frame_data, text='读取数据 (单次)', command=self.read_data_once)
        self.read_data_button.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")
        
        self.realtime_read_button = tkinter.Button(frame_data, text='开始实时监测', command=self.start_realtime_monitor)
        self.realtime_read_button.grid(row=2, column=0, pady=5, sticky="ew")
        
        self.stop_read_button = tkinter.Button(frame_data, text='停止实时监测', command=self.stop_realtime_monitor, state=tkinter.DISABLED)
        self.stop_read_button.grid(row=2, column=1, pady=5, sticky="ew")

        # 2.3. 文件/维护操作组 (Row 2) 
        frame_file = tkinter.LabelFrame(self.left_frame, text="文件/维护", padx=5, pady=5)
        frame_file.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew") 
        
        tkinter.Label(frame_file, text='保存路径').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.save_path_entry = tkinter.Entry(frame_file, textvariable=self.save_path, width=25)
        self.save_path_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        self.delete_log_button = tkinter.Button(frame_file, text='清空日志', command=self.delete_log)
        self.delete_log_button.grid(row=1, column=0, pady=5, sticky="ew")
        
        self.delete_all_button = tkinter.Button(frame_file, text='清空参数', command=self.delete_all_parameter)
        self.delete_all_button.grid(row=1, column=1, pady=5, sticky="ew")

        # 2.4. ***系统日志区 (Row 3, 4)***
        self.ads_communication_log = tkinter.Label(self.left_frame, text='系统日志')
        # 占据 Row 3
        self.ads_communication_log.grid(row=3, column=0, columnspan=2, pady=(5, 0), sticky="sw")
        
        self.log_text = tkinter.Text(self.left_frame, width=35, height=10) 
        # 占据 Row 4，并拉伸
        self.log_text.grid(row=4, column=0, columnspan=2, pady=5, sticky="nsew")


        # ========== 右侧：绘图展示主区 (Row 0 - Row 3) ==========
        
        # 振动波形图 (Row 0 - Row 1)
        self.fig_vib, self.ax_vib = plt.subplots(figsize=(10, 4), dpi=100)
        self.canvas_vib = FigureCanvasTkAgg(self.fig_vib, master=self.init_windows_name)
        # 上移到 Row 0，占据两行
        self.canvas_vib.get_tk_widget().grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky="nsew") 
        
        # 电流趋势图 (Row 2 - Row 3)
        self.fig_current, self.ax_current = plt.subplots(figsize=(10, 4), dpi=100)
        self.canvas_current = FigureCanvasTkAgg(self.fig_current, master=self.init_windows_name)
        # 紧随振动图下方，占据 Row 2，占据两行
        self.canvas_current.get_tk_widget().grid(row=2, column=2, rowspan=2, padx=10, pady=5, sticky="nsew") 
        
        # 初始化绘图
        self.init_plots()

    def init_plots(self):
        """初始化绘图样式"""
        # 振动波形图初始化 (单张图)
        self.ax_vib.clear()
        self.ax_vib.set_title('三方向振动瞬时波形 (X/Y/Z)', fontsize=12)
        self.ax_vib.set_xlabel('时间 (秒)', fontsize=10)
        self.ax_vib.set_ylabel('振幅 (INT)', fontsize=10)
        self.ax_vib.grid(True, alpha=0.3)
        self.fig_vib.tight_layout()
        
        # 电流趋势图初始化
        self.ax_current.clear()
        self.ax_current.set_title('三相电流实时波形趋势', fontsize=12)
        self.ax_current.set_xlabel(f'采样序号 (每周期增加{SAMPLE_COUNT}点)', fontsize=10)
        self.ax_current.set_ylabel('电流值 (INT)', fontsize=10)
        self.ax_current.grid(True, alpha=0.3)
        self.fig_current.tight_layout()

    def update_vibration_plot(self):
        """更新振动波形图，合并 X/Y/Z 到同一张图，并动态调整纵坐标"""
        if len(self.vib_data['time']) == 0:
            return
        
        plot_points = min(VIB_PLOT_POINTS, len(self.vib_data['time']))
        time_slice = self.vib_data['time'][-plot_points:]
        
        x_slice = self.vib_data['X'][-plot_points:] if len(self.vib_data['X']) >= plot_points else self.vib_data['X']
        y_slice = self.vib_data['Y'][-plot_points:] if len(self.vib_data['Y']) >= plot_points else self.vib_data['Y']
        z_slice = self.vib_data['Z'][-plot_points:] if len(self.vib_data['Z']) >= plot_points else self.vib_data['Z']
        
        # ***动态调整纵坐标逻辑***
        all_vib_data = np.concatenate((x_slice, y_slice, z_slice))
        if len(all_vib_data) > 0:
            min_val = np.min(all_vib_data)
            max_val = np.max(all_vib_data)
            
            # 计算裕量
            range_val = max_val - min_val
            margin = range_val * PLOT_Y_MARGIN if range_val > 0 else 10 # 最小裕量为10
            
            y_min = min_val - margin
            y_max = max_val + margin
            
            # 避免范围过小，例如数据全部为0的情况
            if y_min == y_max:
                 y_min -= 50
                 y_max += 50
        else:
            y_min, y_max = -100, 100 # 默认范围
            
        self.ax_vib.clear()
        
        # 绘制 X/Y/Z 波形
        self.ax_vib.plot(time_slice, x_slice, color='red', label='X方向', linewidth=1)
        self.ax_vib.plot(time_slice, y_slice, color='green', label='Y方向', linewidth=1)
        self.ax_vib.plot(time_slice, z_slice, color='blue', label='Z方向', linewidth=1)
        
        self.ax_vib.set_title('三方向振动瞬时波形 (X/Y/Z)', fontsize=12)
        self.ax_vib.set_xlabel('时间 (秒)', fontsize=10)
        self.ax_vib.set_ylabel('振幅 (INT)', fontsize=10)
        self.ax_vib.grid(True, alpha=0.3)
        self.ax_vib.legend(loc='upper right', fontsize=8) 
        
        self.ax_vib.set_ylim(y_min, y_max) # 应用动态纵坐标
        
        self.fig_vib.tight_layout()
        self.canvas_vib.draw()

    def update_current_plot(self):
        """更新电流波形趋势图，并动态调整纵坐标"""
        if len(self.current_x_data) == 0:
            return
        
        MAX_POINTS_TO_SHOW = PLOT_HISTORY_LENGTH * SAMPLE_COUNT
        plot_length = min(MAX_POINTS_TO_SHOW, len(self.current_x_data))
        x_data = self.current_x_data[-plot_length:]
        
        # ***动态调整纵坐标逻辑***
        all_current_data = []
        
        colors = ['red', 'green', 'blue']
        labels = ['A相电流', 'B相电流', 'C相电流']
        
        self.ax_current.clear()
        
        for i in range(CURRENT_CHANNELS):
            if len(self.current_y_data[i]) >= plot_length:
                y_data = self.current_y_data[i][-plot_length:]
            else:
                y_data = self.current_y_data[i]
            
            all_current_data.extend(y_data)
            self.ax_current.plot(x_data, y_data, color=colors[i], label=labels[i], linewidth=1) 
            
        if len(all_current_data) > 0:
            min_val = np.min(all_current_data)
            max_val = np.max(all_current_data)
            
            range_val = max_val - min_val
            margin = range_val * PLOT_Y_MARGIN if range_val > 0 else 1 # 最小裕量为1
            
            y_min = min_val - margin
            y_max = max_val + margin
            
            # 避免范围过小
            if y_min == y_max:
                 y_min -= 1
                 y_max += 1
        else:
            y_min, y_max = 0, 1000 # 默认范围
            
        self.ax_current.set_title('三相电流实时波形趋势', fontsize=12)
        self.ax_current.set_xlabel(f'采样序号 (每周期增加{SAMPLE_COUNT}点)', fontsize=10)
        self.ax_current.set_ylabel('电流值 (INT)', fontsize=10)
        self.ax_current.grid(True, alpha=0.3)
        self.ax_current.legend(loc='upper right', fontsize=10)
        
        self.ax_current.set_ylim(y_min, y_max) # 应用动态纵坐标
        
        self.fig_current.tight_layout()
        self.canvas_current.draw()
        
    def process_data(self, raw_data, index_data):
        """
        数据处理：
        1. 使用 index_data 重组 GvlBuffer，保证时序连续性。
        2. 分离振动和电流数据。
        """
        if raw_data is None or index_data is None:
            self.write_log_to_text("数据或索引读取失败，跳过处理。")
            return

        # 1. ========== 环形缓冲区重组 ==========
        try:
            raw_matrix = np.array(raw_data, dtype=np.int16).reshape(FULL_CHANNELS, SAMPLE_COUNT)
            index_array = np.array(index_data, dtype=np.int16)
        except ValueError as e:
            self.write_log_to_text(f"数据重塑或索引转换错误: {e}")
            return
        
        continuous_data = np.zeros((TOTAL_CHANNELS, SAMPLE_COUNT), dtype=np.int16)
        
        for i in range(TOTAL_CHANNELS): # 只处理前 33 个通道
            write_ptr = index_array[i] # 写指针标记了缓冲区最新的数据块开头
            channel_raw = raw_matrix[i, :]
            
            # 时序重组：[P, P+1, ..., 100] + [1, 2, ..., P-1]
            part1 = channel_raw[write_ptr - 1:] 
            part2 = channel_raw[:write_ptr - 1]
            continuous_data[i, :] = np.concatenate((part1, part2))

        # 2. ========== 分离和处理振动数据 ==========
        
        vib_channels_data = continuous_data[0:VIBRATION_CHANNELS, :]
        
        # X/Y/Z 方向是 10 个通道的拼接
        self.vib_data['X'] = vib_channels_data[0:10, :].flatten()
        self.vib_data['Y'] = vib_channels_data[10:20, :].flatten()
        self.vib_data['Z'] = vib_channels_data[20:30, :].flatten()

        # 计算时间轴 (只计算一次)
        if len(self.vib_data['time']) != len(self.vib_data['X']):
            total_vib_points = VIBRATION_GROUP_SIZE * SAMPLE_COUNT
            time_axis = np.arange(total_vib_points) / SAMPLING_FREQUENCY
            self.vib_data['time'] = time_axis
        
        # 3. ========== 处理电流数据（将 100 个点追加）==========
        
        current_channels_data = continuous_data[VIBRATION_CHANNELS:, :]

        # 3.1 更新 X 轴数据 (每次增加 100 个点)
        start_index = self.sample_index + 1
        end_index = self.sample_index + SAMPLE_COUNT
        new_x_data = np.arange(start_index, end_index + 1) 
        self.current_x_data.extend(new_x_data.tolist())
        self.sample_index = end_index

        # 3.2 更新 Y 轴数据 (将 100 个点追加到每个通道的列表中)
        for i in range(CURRENT_CHANNELS):
            channel_data = current_channels_data[i, :]
            self.current_y_data[i].extend(channel_data.tolist())
        
        # 3.3 限制数据长度
        MAX_POINTS_TO_SHOW = PLOT_HISTORY_LENGTH * SAMPLE_COUNT 
        
        if len(self.current_x_data) > MAX_POINTS_TO_SHOW:
            self.current_x_data = self.current_x_data[-MAX_POINTS_TO_SHOW:]
            for i in range(CURRENT_CHANNELS):
                self.current_y_data[i] = self.current_y_data[i][-MAX_POINTS_TO_SHOW:]

    def delete_all_parameter(self):
        try:
            self.netID_text.delete(1.0, tkinter.END)
            self.netID_text.insert(tkinter.END, DEFAULT_AMS_NETID)
            
            self.port_text.delete(1.0, tkinter.END)
            self.port_text.insert(tkinter.END, DEFAULT_PORT)
            
            self.interval_text.delete(1.0, tkinter.END)
            self.interval_text.insert(tkinter.END, DEFAULT_INTERVAL_MS)
            
            self.write_log_to_text('清空参数并恢复默认值')
        except:
            self.write_log_to_text('清空参数错误')

    def delete_log(self):
        global LOG_LINE_NUM
        self.log_text.delete(1.0, tkinter.END)
        LOG_LINE_NUM = 0
        self.write_log_to_text('日志已清空')

    def Plc_port_open(self):
        global Plc
        AmsNetID = self.netID_text.get(1.0, tkinter.END).strip()
        port = self.port_text.get(1.0, tkinter.END).strip()
        try:
            pyads.open_port() 
            Plc = pyads.Connection(AmsNetID, int(port))
            Plc.open()
            self.write_log_to_text(f'成功连接PLC: {AmsNetID}:{port}')
        except Exception as e:
            self.write_log_to_text(f'连接失败: {str(e)}')
            print(f"连接错误: {e}")

    def get_current_time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    def write_log_to_text(self, logmsg):
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = f"{current_time} {logmsg}\n"
        
        if LOG_LINE_NUM <= 30:
            self.log_text.insert(tkinter.END, logmsg_in)
            LOG_LINE_NUM += 1
        else:
            self.log_text.delete(1.0, 2.0)
            self.log_text.insert(tkinter.END, logmsg_in)
        
        self.log_text.see(tkinter.END)
        self.log_text.update()

    def read_data_atomic(self):
        """
        安全的单次原子读取：
        同时读取 GvlBuffer (数据, 8000 INT) 和 GvlIndexBuffer (写指针, 80 INT)
        """
        global Plc
        if not Plc or not Plc.is_open:
            return None, None

        try:
            # 1. 读取整个数据缓冲区 (8000个INT)
            raw_data = Plc.read(
                GVL_BUFFER_GROUP, 
                GVL_BUFFER_OFFSET, 
                GVL_BUFFER_DATATYPE * FULL_BUFFER_LENGTH
            )
            
            # 2. 读取索引缓冲区 (80个INT)
            index_data = Plc.read(
                GVL_BUFFER_GROUP,
                INDEX_BUFFER_OFFSET,
                INDEX_BUFFER_DATATYPE * INDEX_BUFFER_LENGTH
            )
            
            return raw_data, index_data
            
        except Exception as e:
            self.write_log_to_text(f'原子读取失败: {str(e)}')
            return None, None

    def read_data_once(self):
        """单次读取数据"""
        try:
            self.write_log_to_text('开始读取振动电流数据...')
            
            raw_data, index_data = self.read_data_atomic()
            if raw_data is None:
                return
            
            self.process_data(raw_data, index_data)
            
            self.update_vibration_plot()
            self.update_current_plot()
            
            self.save_data_to_file(raw_data)
            
            self.write_log_to_text(f'成功读取{len(raw_data)}个数据点 (已使用索引重组)')
            self.write_log_to_text('数据已更新到图表并保存')
            
        except Exception as e:
            self.write_log_to_text(f'单次读取失败: {str(e)}')
            print(f"读取错误: {e}")

    def save_data_to_file(self, raw_data):
        """保存数据到文件（保存原始PLC内存快照，不进行重组）"""
        filepath = self.save_path.get()
        try:
            # 裁剪到实际使用的 3300 个 INT
            used_raw_data = raw_data[:TOTAL_CHANNELS * SAMPLE_COUNT]
            
            vib_raw = used_raw_data[:VIBRATION_CHANNELS * SAMPLE_COUNT]
            curr_raw = used_raw_data[VIBRATION_CHANNELS * SAMPLE_COUNT:]

            # 振动数据重组 
            vib_matrix = np.array(vib_raw, dtype=np.int16).reshape(VIBRATION_CHANNELS, SAMPLE_COUNT)
            vib_channels_data = vib_matrix
            vib_x = vib_channels_data[0:10, :].flatten()
            vib_y = vib_channels_data[10:20, :].flatten()
            vib_z = vib_channels_data[20:30, :].flatten()
            
            # 电流数据重组
            current_matrix = np.array(curr_raw, dtype=np.int16).reshape(CURRENT_CHANNELS, SAMPLE_COUNT)
            current_matrix = current_matrix.T
            
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 采集时间: {self.get_current_time()} ===\n")
                
                # 写入振动数据
                # f.write("=== 振动数据（X/Y/Z方向，10通道时序合成，**原始物理存储顺序**）===\n")
                # f.write("时间(s)\tX振动\tY振动\tZ振动\n")
                time_axis = np.arange(len(vib_x)) / SAMPLING_FREQUENCY
                for i in range(len(time_axis)):
                    f.write(f"{time_axis[i]:.4f}\t{vib_x[i]}\t{vib_y[i]}\t{vib_z[i]}\n")
                
                # 写入电流波形数据 (所有100个点)
                # f.write("\n=== 三相电流数据（100个采样点波形，**原始物理存储顺序**）===\n")
                # f.write(f"采样序号\tA相电流\tB相电流\tC相电流\n")
                for j in range(SAMPLE_COUNT):
                    f.write(f"{j+1}\t{current_matrix[j, 0]}\t{current_matrix[j, 1]}\t{current_matrix[j, 2]}\n")

                f.write('='*80 + '\n')
                
        except Exception as e:
            self.write_log_to_text(f'文件保存失败: {str(e)}')


    def start_realtime_monitor(self):
        if not Plc or not Plc.is_open:
            self.write_log_to_text('请先打开PLC端口')
            return

        self.is_realtime_running = True
        self.realtime_read_button.config(state=tkinter.DISABLED)
        self.stop_read_button.config(state=tkinter.NORMAL)
        self.write_log_to_text('开始实时振动电流监测...')
        
        self.realtime_monitor_loop()

    def stop_realtime_monitor(self):
        self.is_realtime_running = False
        self.realtime_read_button.config(state=tkinter.NORMAL)
        self.stop_read_button.config(state=tkinter.DISABLED)
        self.write_log_to_text('已停止实时监测')

    def realtime_monitor_loop(self):
        """实时监测循环"""
        if not self.is_realtime_running:
            return

        try:
            interval = int(self.interval_text.get(1.0, tkinter.END).strip())
            
            raw_data, index_data = self.read_data_atomic()
            if raw_data is None:
                self.stop_realtime_monitor() 
                return
            
            self.process_data(raw_data, index_data)
            
            self.update_vibration_plot()
            self.update_current_plot()
            
            self.save_data_to_file(raw_data)
            
            self.write_log_to_text(f'实时监测数据更新完成 (累计采样点: {self.sample_index})')
            
        except Exception as e:
            self.write_log_to_text(f'实时监测错误: {str(e)}')
        
        if self.is_realtime_running:
            self.init_windows_name.after(interval, self.realtime_monitor_loop)

# 主程序
def Gui_Start():
    plt.rcParams['font.sans-serif'] = ['SimHei']  # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False
    
    init_window = tkinter.Tk()
    MAIN_Window = GUI(init_window)
    MAIN_Window.set_init_window()
    init_window.mainloop()

if __name__ == "__main__":
    Gui_Start()