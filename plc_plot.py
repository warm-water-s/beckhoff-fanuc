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

# -*- coding:utf-8 -*-
"""
    ADS 数据采集、动态增量提取与高效保存系统
    功能：
    1. 通过 pyads 连接 PLC,读取环形缓冲区数据。
    2. 实现 PLC 基础 1kHz 采样率,振动 10 个通道交错实现 10kHz 等效采样率的逻辑。
    3. 动态提取 T_interval (用户设定)时长内的增量数据。
    4. 实时更新振动和电流的趋势图(使用增量数据更新历史缓存，并绘制滑动窗口)。
    5. 仅保存增量数据到文件。
    
    主要优化点:
    - 历史缓存点数计算逻辑修正: 确保 MAX_HISTORY_POINTS 对应 PLOT_HISTORY_LENGTH * (每周期实际采集点数)。
    - 代码结构优化: 将 MAX_HISTORY_POINTS 等基于配置的参数在 __init__ 中计算。
"""
import pyads
import tkinter
import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 日志默认条数
LOG_LINE_NUM = 0

# ========== 通道与采样配置 (常量) ==========
TOTAL_CHANNELS = 33 # 总有效通道数 (实际使用)
VIBRATION_CHANNELS = 30 # 振动通道数（前30个）
CURRENT_CHANNELS = 3 # 电流通道数（后3个）
VIBRATION_GROUP_SIZE = 10 # 每10个通道合成一个振动方向 (X, Y, Z)

# 采样率配置 (基于 1kHz 基础采样率和通道组合)
BASE_SAMPLING_FREQUENCY = 1000 # PLC基础采样频率 1000Hz
SAMPLE_COUNT = 100 # PLC缓冲区每通道存储的点数 (100ms 窗口)

# 等效采样率
VIB_SAMPLING_FREQUENCY = BASE_SAMPLING_FREQUENCY * VIBRATION_GROUP_SIZE # 1000 * 10 = 10000 Hz
CURR_SAMPLING_FREQUENCY = BASE_SAMPLING_FREQUENCY # 1000 Hz

# ADS 配置 (PLC内存配置)
FULL_CHANNELS = 80 # PLC实际分配的通道数
FULL_BUFFER_LENGTH = FULL_CHANNELS * SAMPLE_COUNT 
GVL_BUFFER_DATATYPE = pyads.PLCTYPE_INT 
GVL_BUFFER_GROUP = 0x4020 
GVL_BUFFER_OFFSET = 0x0 
INDEX_BUFFER_OFFSET = 16000 
INDEX_BUFFER_LENGTH = FULL_CHANNELS 
INDEX_BUFFER_DATATYPE = pyads.PLCTYPE_INT

# 默认连接参数
DEFAULT_AMS_NETID = "5.136.192.215.1.1"
DEFAULT_PORT = "851"
DEFAULT_INTERVAL_MS = "10" # 采集周期/增量时间 T_interval

# 绘图配置 (历史点数, 对应 1s 窗口, 历史缓存长度)
# 注意: PLOT_HISTORY_LENGTH 决定了历史缓存的总长度（非显示长度）
VIB_PLOT_POINTS = VIB_SAMPLING_FREQUENCY * 1 # 10000 Hz * 1s = 10000 点 (振动波形显示窗口宽度 1s)
CURR_PLOT_POINTS = CURR_SAMPLING_FREQUENCY * 1 # 1000 Hz * 1s = 1000 点 (电流趋势图显示窗口宽度 1s)
PLOT_HISTORY_LENGTH = 110 # 历史缓存周期数 (110 * 100ms = 11s)
PLOT_Y_MARGIN = 0.15 # 15% 绘图纵坐标裕量


class GUI():
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value="processed_log.txt")
        self.plc_conn = None
        self.is_realtime_running = False
        self.sample_index = 0 # 用于电流趋势图的 x 轴点数计数 (累计)
        
        # 修正历史缓存点数计算 (核心修正)
        # 历史缓存点数 = PLOT_HISTORY_LENGTH * (每周期实际采集点数)
        
        # 振动历史数据缓存: 每周期 (100ms) 采集 10 个 1kHz 通道 = 1000 点
        # MAX_VIB_HISTORY_POINTS = 周期数 * 1000 点
        self.MAX_VIB_HISTORY_POINTS = PLOT_HISTORY_LENGTH * VIBRATION_GROUP_SIZE * SAMPLE_COUNT 
        self.vib_x_history = [] 
        self.vib_y_history = []
        self.vib_z_history = []
        
        # 电流历史数据缓存: 每周期 (100ms) 采集 1 个 1kHz 通道 = 100 点
        # MAX_CURR_HISTORY_POINTS = 周期数 * 100 点
        self.MAX_CURR_HISTORY_POINTS = PLOT_HISTORY_LENGTH * SAMPLE_COUNT 
        self.current_x_history = [] 
        self.current_y_history = [[] for _ in range(CURRENT_CHANNELS)]
        
        self.set_init_window()

    def set_init_window(self):
        """初始化基础UI界面和布局"""
        self.init_windows_name.title('ADS 通讯 - 振动(10kHz)电流(1kHz)监测系统')
        self.init_windows_name.geometry('1400x800+30+30')
        self.init_windows_name.attributes('-alpha', 0.95)
        
        # ***布局优化：调整 grid 权重***
        self.init_windows_name.grid_columnconfigure(2, weight=1) 
        for i in range(4): self.init_windows_name.grid_rowconfigure(i, weight=1) 
        
        # 1. ========== 左侧：主容器 Frame ==========
        self.left_frame = tkinter.Frame(self.init_windows_name)
        self.left_frame.grid(row=0, column=0, columnspan=2, rowspan=4, padx=10, pady=5, sticky="nsew")
        self.left_frame.grid_rowconfigure(4, weight=1) 
        
        # 2. ========== 左侧：操作控制栏 (简化结构) ==========
        
        # 2.1. ADS 连接配置组 (Row 0)
        frame_conn = tkinter.LabelFrame(self.left_frame, text="ADS 连接配置", padx=5, pady=5)
        frame_conn.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")
        
        # 使用 Entry 代替 Text for single line input
        tkinter.Label(frame_conn, text='AmsNetID').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.netID_var = tkinter.StringVar(value=DEFAULT_AMS_NETID)
        tkinter.Entry(frame_conn, textvariable=self.netID_var, width=25).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        tkinter.Label(frame_conn, text='Port').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.port_var = tkinter.StringVar(value=DEFAULT_PORT)
        tkinter.Entry(frame_conn, textvariable=self.port_var, width=25).grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        self.open_port_button = tkinter.Button(frame_conn, text='打开端口', command=self.plc_port_open)
        self.open_port_button.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        
        # 2.2. 数据采集控制组 (Row 1) 
        frame_data = tkinter.LabelFrame(self.left_frame, text="数据采集控制", padx=5, pady=5)
        frame_data.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew") 
        
        tkinter.Label(frame_data, text='读取间隔(ms)').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.interval_var = tkinter.StringVar(value=DEFAULT_INTERVAL_MS)
        tkinter.Entry(frame_data, textvariable=self.interval_var, width=25).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
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
        
        self.delete_all_button = tkinter.Button(frame_file, text='重置连接参数', command=self.reset_parameters)
        self.delete_all_button.grid(row=1, column=1, pady=5, sticky="ew")

        # 2.4. ***系统日志区 (Row 3, 4)***
        tkinter.Label(self.left_frame, text='系统日志').grid(row=3, column=0, columnspan=2, pady=(5, 0), sticky="sw")
        self.log_text = tkinter.Text(self.left_frame, width=35, height=10) 
        self.log_text.grid(row=4, column=0, columnspan=2, pady=5, sticky="nsew")


        # 3. ========== 右侧：绘图展示主区 ==========
        
        # 振动趋势图
        self.fig_vib, self.ax_vib = plt.subplots(figsize=(10, 4), dpi=100)
        self.canvas_vib = FigureCanvasTkAgg(self.fig_vib, master=self.init_windows_name)
        self.canvas_vib.get_tk_widget().grid(row=0, column=2, rowspan=2, padx=10, pady=5, sticky="nsew") 
        
        # 电流趋势图
        self.fig_current, self.ax_current = plt.subplots(figsize=(10, 4), dpi=100)
        self.canvas_current = FigureCanvasTkAgg(self.fig_current, master=self.init_windows_name)
        self.canvas_current.get_tk_widget().grid(row=2, column=2, rowspan=2, padx=10, pady=5, sticky="nsew") 
        
        self.init_plots()


    # --- UI & Log Functions ---
    def get_current_time(self):
        return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(time.time()))

    def write_log_to_text(self, logmsg):
        """将日志写入UI文本框"""
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = f"[{current_time}] {logmsg}\n"
        
        if LOG_LINE_NUM <= 30:
            self.log_text.insert(tkinter.END, logmsg_in)
            LOG_LINE_NUM += 1
        else:
            self.log_text.delete(1.0, 2.0)
            self.log_text.insert(tkinter.END, logmsg_in)
        
        self.log_text.see(tkinter.END)
        self.log_text.update()

    def reset_parameters(self):
        """重置连接和采集参数"""
        try:
            self.netID_var.set(DEFAULT_AMS_NETID)
            self.port_var.set(DEFAULT_PORT)
            self.interval_var.set(DEFAULT_INTERVAL_MS)
            self.write_log_to_text('连接参数已重置为默认值')
        except Exception as e:
            self.write_log_to_text(f'重置参数错误: {e}')

    def delete_log(self):
        """清空日志"""
        global LOG_LINE_NUM
        self.log_text.delete(1.0, tkinter.END)
        LOG_LINE_NUM = 0
        self.write_log_to_text('日志已清空')

    # --- PLC Connection & Read ---

    def plc_port_open(self):
        """打开ADS端口并连接到PLC"""
        AmsNetID = self.netID_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self.write_log_to_text('端口号无效，请检查输入。')
            return
        
        if self.plc_conn and self.plc_conn.is_open:
            self.write_log_to_text('端口已连接，请勿重复操作。')
            return

        try:
            pyads.open_port() 
            self.plc_conn = pyads.Connection(AmsNetID, port)
            self.plc_conn.open()
            self.write_log_to_text(f'成功连接PLC: {AmsNetID}:{port}')
        except Exception as e:
            self.write_log_to_text(f'连接失败: {str(e)}')
            self.plc_conn = None

    def _read_data_atomic(self):
        """原子读取数据和索引"""
        if not self.plc_conn or not self.plc_conn.is_open:
            return None, None

        try:
            raw_data = self.plc_conn.read(
                GVL_BUFFER_GROUP, GVL_BUFFER_OFFSET, GVL_BUFFER_DATATYPE * FULL_BUFFER_LENGTH
            )
            index_data = self.plc_conn.read(
                GVL_BUFFER_GROUP, INDEX_BUFFER_OFFSET, INDEX_BUFFER_DATATYPE * INDEX_BUFFER_LENGTH
            )
            return raw_data, index_data
        except Exception as e:
            self.write_log_to_text(f'原子读取失败: {str(e)}')
            return None, None

    # --- Data Processing (核心逻辑) ---

    def process_data(self, raw_data, index_data):
        """
        数据处理：
        1. 环形缓冲区重组，获取 100ms 连续波形。
        2. 振动通道交错重排 (10kHz)。
        3. 提取增量数据，并更新**绘图历史缓存**。
        4. 返回增量数据字典供保存。
        """
        if raw_data is None or index_data is None:
            return None, None

        # 获取 T_interval (ms)
        try:
            T_interval_ms = float(self.interval_var.get().strip())
        except ValueError:
            T_interval_ms = float(DEFAULT_INTERVAL_MS)
            
        # 1. 计算增量点数 (确保增量点数是整数)
        N_inc_vib_points = int(round(T_interval_ms * (VIB_SAMPLING_FREQUENCY / 1000))) 
        N_inc_curr_points = int(round(T_interval_ms * (CURR_SAMPLING_FREQUENCY / 1000)))
        
        # 边界检查: 确保采集点数不大于 100ms 的 PLC 周期点数
        N_inc_vib_points = min(N_inc_vib_points, VIBRATION_GROUP_SIZE * SAMPLE_COUNT) # 1000点
        N_inc_curr_points = min(N_inc_curr_points, SAMPLE_COUNT) # 100点

        # --- 环形缓冲区重组 ---
        try:
            raw_matrix = np.array(raw_data, dtype=np.int16).reshape(FULL_CHANNELS, SAMPLE_COUNT)
            index_array = np.array(index_data, dtype=np.int16)
        except ValueError as e:
            self.write_log_to_text(f"数据重塑或索引转换错误: {e}")
            return None, None
        
        continuous_data_100ms = np.zeros((TOTAL_CHANNELS, SAMPLE_COUNT), dtype=np.int16)
        
        for i in range(TOTAL_CHANNELS):
            write_ptr = index_array[i] 
            channel_raw = raw_matrix[i, :]
            # 确保指针在合法范围内
            if write_ptr >= SAMPLE_COUNT: write_ptr = SAMPLE_COUNT - 1
            if write_ptr < 0: write_ptr = 0
            
            # 环形缓冲区重组: [P, P+1, ..., 99] + [0, 1, ..., P-1] (确保时序连续)
            continuous_data_100ms[i, :] = np.concatenate((channel_raw[write_ptr:], channel_raw[:write_ptr]))


        # 2. ========== 振动数据交错重排 (10kHz 连续波形) ==========
        vib_channels_100ms = continuous_data_100ms[0:VIBRATION_CHANNELS, :]
        
        def interleave_vibration(channels_data):
            """将 10 个 1kHz 通道交错重排成 10kHz 连续波形 (1000点)"""
            # (10通道, 100点) -> (100点, 10通道) -> 扁平化 (1000点)
            return channels_data.T.flatten()

        vib_x_10ch = vib_channels_100ms[0:10, :] 
        vib_y_10ch = vib_channels_100ms[10:20, :] 
        vib_z_10ch = vib_channels_100ms[20:30, :] 
        
        vib_x_100ms = interleave_vibration(vib_x_10ch)
        vib_y_100ms = interleave_vibration(vib_y_10ch)
        vib_z_100ms = interleave_vibration(vib_z_10ch)

        # 3. ========== 提取和缓存振动增量 ==========
        
        # 提取振动增量 (位于 100ms 波形的末尾)
        vib_x_inc = vib_x_100ms[-N_inc_vib_points:]
        vib_y_inc = vib_y_100ms[-N_inc_vib_points:]
        vib_z_inc = vib_z_100ms[-N_inc_vib_points:]

        # ***更新绘图历史缓存***
        self.vib_x_history.extend(vib_x_inc.tolist())
        self.vib_y_history.extend(vib_y_inc.tolist())
        self.vib_z_history.extend(vib_z_inc.tolist())
        
        # 限制历史缓存的长度
        if len(self.vib_x_history) > self.MAX_VIB_HISTORY_POINTS:
            self.vib_x_history = self.vib_x_history[-self.MAX_VIB_HISTORY_POINTS:]
            self.vib_y_history = self.vib_y_history[-self.MAX_VIB_HISTORY_POINTS:]
            self.vib_z_history = self.vib_z_history[-self.MAX_VIB_HISTORY_POINTS:]


        # 4. ========== 提取和缓存电流增量 ==========
        current_channels_100ms = continuous_data_100ms[VIBRATION_CHANNELS:, :]
        current_channels_inc = current_channels_100ms[:, -N_inc_curr_points:] 
        
        # 更新 X 轴数据 (每次增加 N_inc_curr_points 个点)
        new_x_data = np.arange(self.sample_index + 1, self.sample_index + N_inc_curr_points + 1) 
        self.current_x_history.extend(new_x_data.tolist())
        self.sample_index = self.sample_index + N_inc_curr_points # 累计采样点数更新

        # 更新 Y 轴数据
        for i in range(CURRENT_CHANNELS):
            channel_data = current_channels_inc[i, :]
            self.current_y_history[i].extend(channel_data.tolist())
        
        # 限制电流历史数据长度
        if len(self.current_x_history) > self.MAX_CURR_HISTORY_POINTS:
            self.current_x_history = self.current_x_history[-self.MAX_CURR_HISTORY_POINTS:]
            for i in range(CURRENT_CHANNELS):
                self.current_y_history[i] = self.current_y_history[i][-self.MAX_CURR_HISTORY_POINTS:]

        # 5. ========== 返回增量数据字典 (用于保存文件) ==========
        incremental_data = {
            'Vibration': {
                'X': vib_x_inc,
                'Y': vib_y_inc,
                'Z': vib_z_inc,
            },
            'Current': {
                'A': current_channels_inc[0, :], 
                'B': current_channels_inc[1, :],
                'C': current_channels_inc[2, :],
            },
            'T_interval_ms': T_interval_ms,
            'N_inc_vib_points': len(vib_x_inc),
            'N_inc_curr_points': N_inc_curr_points,
        }

        # process_data 返回处理结果和增量数据, 这里的 None 保持原意, 增量数据用于文件保存
        return None, incremental_data 

    # --- Data Save ---

    def save_data_to_file(self, incremental_data):
        """只保存增量数据 (T_interval 时长)"""
        filepath = self.save_path.get()
        timestamp = self.get_current_time()
        
        # ... (数据提取和计算部分不变, 逻辑清晰)
        vib_x = incremental_data['Vibration']['X'] 
        vib_y = incremental_data['Vibration']['Y']
        vib_z = incremental_data['Vibration']['Z']
        curr_a = incremental_data['Current']['A'] 
        curr_b = incremental_data['Current']['B']
        curr_c = incremental_data['Current']['C']
        
        T_interval_ms = incremental_data['T_interval_ms'] 
        NUM_VIB_POINTS = len(vib_x) 
        NUM_CURR_POINTS = len(curr_a) 
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 采集时间: {timestamp} (增量周期: {T_interval_ms}ms) ===\n")
                
                # 写入振动数据
                f.write(f"振动增量数据 [等效{VIB_SAMPLING_FREQUENCY}Hz, {NUM_VIB_POINTS}点]\n")
                f.write("时序序号\t时间(s)\tX振动(INT)\tY振动(INT)\tZ振动(INT)\n")
                
                time_axis_vib = np.arange(NUM_VIB_POINTS) / VIB_SAMPLING_FREQUENCY
                for i in range(NUM_VIB_POINTS):
                    f.write(f"{i+1}\t{time_axis_vib[i]:.4f}\t{vib_x[i]}\t{vib_y[i]}\t{vib_z[i]}\n")
                
                # 写入电流波形数据
                f.write(f"\n电流增量数据 [{CURR_SAMPLING_FREQUENCY}Hz, {NUM_CURR_POINTS}点]\n")
                f.write(f"采样序号\t时间(s)\tA相电流(INT)\tB相电流(INT)\tC相电流(INT)\n")
                
                time_axis_curr = np.arange(NUM_CURR_POINTS) / CURR_SAMPLING_FREQUENCY
                for j in range(NUM_CURR_POINTS):
                    f.write(f"{j+1}\t{time_axis_curr[j]:.3f}\t{curr_a[j]}\t{curr_b[j]}\t{curr_c[j]}\n")
                        
                f.write('='*80 + '\n')
            
        except Exception as e:
            self.write_log_to_text(f'文件保存失败: {str(e)}')


    # --- Plotting Functions ---
    def init_plots(self):
        """初始化绘图样式"""
        # 振动趋势图初始化
        self.ax_vib.clear()
        self.ax_vib.set_title(f'三方向振动波形 (滑动窗口 {VIB_PLOT_POINTS}点/1s) [{VIB_SAMPLING_FREQUENCY}Hz]', fontsize=12)
        self.ax_vib.set_xlabel('时间 (秒)', fontsize=10)
        self.ax_vib.set_ylabel('振幅 (INT)', fontsize=10)
        self.ax_vib.grid(True, alpha=0.3)
        self.fig_vib.tight_layout()
        
        # 电流趋势图初始化
        self.ax_current.clear()
        
        try:
            T_interval_ms = float(self.interval_var.get().strip())
        except ValueError:
            T_interval_ms = float(DEFAULT_INTERVAL_MS)
        
        N_inc_points = int(round(T_interval_ms * (CURR_SAMPLING_FREQUENCY / 1000))) 
        self.ax_current.set_title(f'三相电流实时趋势 (滑动窗口 {CURR_PLOT_POINTS}点/1s) [{CURR_SAMPLING_FREQUENCY}Hz]', fontsize=12)
        self.ax_current.set_xlabel(f'采样序号 (每次更新{N_inc_points}点)', fontsize=10)
        self.ax_current.set_ylabel('电流值 (INT)', fontsize=10)
        self.ax_current.grid(True, alpha=0.3)
        self.fig_current.tight_layout()

    def update_vibration_plot(self):
        """更新振动波形趋势图，绘制最新的 VIB_PLOT_POINTS 点"""
        if len(self.vib_x_history) == 0:
            return
        
        plot_length = min(VIB_PLOT_POINTS, len(self.vib_x_history))
        
        # 提取滑动窗口数据
        x_slice = np.array(self.vib_x_history[-plot_length:])
        y_slice = np.array(self.vib_y_history[-plot_length:])
        z_slice = np.array(self.vib_z_history[-plot_length:])
        
        # 根据采样率创建时间轴
        time_slice = np.arange(len(x_slice)) / VIB_SAMPLING_FREQUENCY
        
        # ... (动态 Y 轴调整逻辑不变)
        all_vib_data = np.concatenate((x_slice, y_slice, z_slice))
        if len(all_vib_data) > 0:
            min_val = np.min(all_vib_data); max_val = np.max(all_vib_data)
            range_val = max_val - min_val
            margin = range_val * PLOT_Y_MARGIN if range_val > 0 else 10
            y_min = min_val - margin; y_max = max_val + margin
            if y_min == y_max: y_min -= 50; y_max += 50
        else:
            y_min, y_max = -100, 100
            
        self.ax_vib.clear()
        
        # 绘制 X/Y/Z 波形
        self.ax_vib.plot(time_slice, x_slice, color='red', label='X方向', linewidth=1)
        self.ax_vib.plot(time_slice, y_slice, color='green', label='Y方向', linewidth=1)
        self.ax_vib.plot(time_slice, z_slice, color='blue', label='Z方向', linewidth=1)
        
        self.ax_vib.set_title(f'三方向振动波形 (滑动窗口 {plot_length}点) [{VIB_SAMPLING_FREQUENCY}Hz]', fontsize=12)
        self.ax_vib.set_xlabel('时间 (秒)', fontsize=10)
        self.ax_vib.set_ylabel('振幅 (INT)', fontsize=10)
        self.ax_vib.grid(True, alpha=0.3)
        self.ax_vib.legend(loc='upper right', fontsize=8) 
        
        self.ax_vib.set_ylim(y_min, y_max)
        self.fig_vib.tight_layout()
        self.canvas_vib.draw()

    def update_current_plot(self):
        """更新电流波形趋势图，绘制最新的 CURR_PLOT_POINTS 点"""
        if len(self.current_x_history) == 0:
            return
        
        plot_length = min(CURR_PLOT_POINTS, len(self.current_x_history))
        
        # 提取滑动窗口数据
        x_data = self.current_x_history[-plot_length:]
        
        # 更新标题中的增量点数
        try:
            T_interval_ms = float(self.interval_var.get().strip())
        except ValueError:
            T_interval_ms = float(DEFAULT_INTERVAL_MS)
        
        N_inc_points = int(round(T_interval_ms * (CURR_SAMPLING_FREQUENCY / 1000))) 
        
        all_current_data = []
        colors = ['red', 'green', 'blue']
        labels = ['A相电流', 'B相电流', 'C相电流']
        
        self.ax_current.clear()
        
        for i in range(CURRENT_CHANNELS):
            y_data = self.current_y_history[i][-plot_length:]
            all_current_data.extend(y_data)
            self.ax_current.plot(x_data, y_data, color=colors[i], label=labels[i], linewidth=1) 
            
        # ... (动态 Y 轴调整逻辑不变)
        if len(all_current_data) > 0:
            min_val = np.min(all_current_data); max_val = np.max(all_current_data)
            range_val = max_val - min_val
            margin = range_val * PLOT_Y_MARGIN if range_val > 0 else 1
            y_min = min_val - margin; y_max = max_val + margin
            if y_min == y_max: y_min -= 1; y_max += 1
        else:
            y_min, y_max = 0, 1000
            
        self.ax_current.set_title(f'三相电流实时趋势 (滑动窗口 {plot_length}点) [{CURR_SAMPLING_FREQUENCY}Hz]', fontsize=12)
        self.ax_current.set_xlabel(f'采样序号 (每次更新{N_inc_points}点)', fontsize=10)
        self.ax_current.set_ylabel('电流值 (INT)', fontsize=10)
        self.ax_current.grid(True, alpha=0.3)
        self.ax_current.legend(loc='upper right', fontsize=10)
        
        self.ax_current.set_ylim(y_min, y_max)
        self.fig_current.tight_layout()
        self.canvas_current.draw()

    # --- Realtime Control ---

    def read_data_once(self):
        """单次读取数据并更新图表和文件"""
        try:
            self.write_log_to_text('开始读取振动电流数据...')
            
            raw_data, index_data = self._read_data_atomic()
            if raw_data is None: return
            
            # 使用增量处理 (同时更新内部历史缓存)
            _, incremental_data = self.process_data(raw_data, index_data)
            
            if incremental_data is None: return

            self.update_vibration_plot()
            self.update_current_plot()
            
            self.save_data_to_file(incremental_data)
            
            self.write_log_to_text(f'成功读取数据点 (振动增量: {incremental_data["N_inc_vib_points"]}点, 电流增量: {incremental_data["N_inc_curr_points"]}点)')
            self.write_log_to_text('数据已更新到图表并保存')
            
        except Exception as e:
            self.write_log_to_text(f'单次读取失败: {str(e)}')

    def start_realtime_monitor(self):
        """启动实时采集循环"""
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text('请先打开PLC端口')
            return

        self.is_realtime_running = True
        self.realtime_read_button.config(state=tkinter.DISABLED)
        self.stop_read_button.config(state=tkinter.NORMAL)
        self.write_log_to_text('开始实时振动电流监测...')
        
        self.realtime_monitor_loop()

    def stop_realtime_monitor(self):
        """停止实时采集循环"""
        self.is_realtime_running = False
        self.realtime_read_button.config(state=tkinter.NORMAL)
        self.stop_read_button.config(state=tkinter.DISABLED)
        self.write_log_to_text('已停止实时监测')

    def realtime_monitor_loop(self):
        """实时采集循环：读取、处理、更新图表、保存"""
        if not self.is_realtime_running:
            return

        try:
            # 确保 interval 是一个有效的整数
            try:
                interval = int(self.interval_var.get().strip())
                if interval <= 0:
                    interval = int(DEFAULT_INTERVAL_MS)
                    self.write_log_to_text('读取间隔必须大于0，已恢复默认值。')
            except ValueError:
                interval = int(DEFAULT_INTERVAL_MS)
                self.write_log_to_text('读取间隔输入无效，已恢复默认值。')
            
            raw_data, index_data = self._read_data_atomic()
            if raw_data is None:
                self.stop_realtime_monitor() 
                return
            
            # 核心：使用增量处理和缓存
            _, incremental_data = self.process_data(raw_data, index_data)
            
            if incremental_data is None:
                return
            
            # 更新绘图
            self.update_vibration_plot()
            self.update_current_plot()
            
            self.save_data_to_file(incremental_data)
            
            # 优化日志输出频率 (避免日志过多影响性能)
            # 仅在需要时输出，或者限制输出频率
            if self.sample_index % (10 * incremental_data["N_inc_curr_points"]) == 0: # 比如每 10 次增量更新打印一次
                 self.write_log_to_text(f'实时监测数据更新完成 (振动增量: {incremental_data["N_inc_vib_points"]}点, 电流增量: {incremental_data["N_inc_curr_points"]}点, 累计采样点: {self.sample_index})')
            
        except Exception as e:
            self.write_log_to_text(f'实时监测循环发生错误: {str(e)}')
            self.stop_realtime_monitor() # 发生错误时停止循环
        
        # 安排下一次运行
        if self.is_realtime_running:
            self.init_windows_name.after(interval, self.realtime_monitor_loop)

# 主程序
def Gui_Start():
    plt.rcParams['font.sans-serif'] = ['SimHei'] # 支持中文显示
    plt.rcParams['axes.unicode_minus'] = False
    
    init_window = tkinter.Tk()
    MAIN_Window = GUI(init_window)
    init_window.mainloop()

if __name__ == "__main__":
    Gui_Start()