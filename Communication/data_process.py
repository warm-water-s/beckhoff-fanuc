# -*- coding:utf-8 -*-
"""
ADS 数据采集、动态增量提取与高效保存系统

功能：
1. 实时读取 PLC 环形缓冲区数据。
2. 将 100ms 完整波形重组 (可用于特征计算)。
3. 根据用户设置的采集间隔 (T_interval) 动态提取最新的增量数据。
4. 仅保存 T_interval 增量数据，实现高效存储。

数据文件(processed_sensor_log.txt)结构 (只保存增量数据)
    === 采集时间: 2025-12-03 21:00:00 (周期序号: 1) ===
    振动增量数据 (10ms, 100点)
    时序序号 X振动(INT) Y振动(INT) Z振动(INT)
    ... (共 10 * N_inc_points 行振动数据)

    电流增量数据 (10ms, 10个采样点)
    采样序号 A相电流(INT) B相电流(INT) C相电流(INT)
    ... (共 N_inc_points 行电流数据)
    ===================================================
"""
import pyads
import tkinter
import time
import numpy as np

# ========== 通道配置 (保持不变) ==========
TOTAL_CHANNELS = 33 # 总有效通道数 (实际使用)
VIBRATION_CHANNELS = 30 # 振动通道数（前30个）
CURRENT_CHANNELS = 3 # 电流通道数（后3个）
VIBRATION_GROUP_SIZE = 10 # 每10个通道合成一个振动方向 (X, Y, Z)
SAMPLE_COUNT = 100 # 每个通道采样点数 (代表 100ms 历史波形)
SAMPLING_FREQUENCY = 10000 # 振动/高速采样频率 10000Hz (0.1ms/点)

# ADS 配置 (保持不变)
FULL_CHANNELS = 80
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
DEFAULT_INTERVAL_MS = "10" # 采集周期/请求间隔 10ms
DEFAULT_SAVE_PATH = "processed_sensor_log.txt"
LOG_LINE_NUM = 0

# ============================================

class DataLoggerApp:
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value=DEFAULT_SAVE_PATH)
        self.plc_conn = None
        self.is_realtime_running = False
        self.sample_index = 0 # 用于记录总采样周期次数
        
        # 实时数据缓存
        self.latest_processed_data_100ms = None 
        
        self.set_init_window()

    # --- UI 初始化与日志功能 (保持不变) ---
    def set_init_window(self):
        """初始化基础UI界面"""
        self.init_windows_name.title('ADS 数据采集与保存系统')
        self.init_windows_name.geometry('600x450+100+100')
        self.init_windows_name.grid_columnconfigure(0, weight=1)
        self.init_windows_name.grid_rowconfigure(4, weight=1)

        # 1. ADS 连接配置组 
        frame_conn = tkinter.LabelFrame(self.init_windows_name, text="ADS 连接配置", padx=5, pady=5)
        frame_conn.grid(row=0, column=0, pady=5, padx=10, sticky="ew")
        
        tkinter.Label(frame_conn, text='AmsNetID').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.netID_text = self._create_text_widget(frame_conn, DEFAULT_AMS_NETID, row=0, column=1)
        
        tkinter.Label(frame_conn, text='Port').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.port_text = self._create_text_widget(frame_conn, DEFAULT_PORT, row=1, column=1)
        
        self.open_port_button = tkinter.Button(frame_conn, text='打开端口', command=self.plc_port_open)
        self.open_port_button.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        
        frame_conn.grid_columnconfigure(1, weight=1)


        # 2. 数据采集控制组
        frame_data = tkinter.LabelFrame(self.init_windows_name, text="数据采集与保存控制", padx=5, pady=5)
        frame_data.grid(row=1, column=0, pady=5, padx=10, sticky="ew")
        
        tkinter.Label(frame_data, text='采集间隔(ms)').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.interval_text = self._create_text_widget(frame_data, DEFAULT_INTERVAL_MS, width=15, row=0, column=1)
        
        tkinter.Label(frame_data, text='保存路径').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.save_path_entry = tkinter.Entry(frame_data, textvariable=self.save_path, width=25)
        self.save_path_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        self.realtime_read_button = tkinter.Button(frame_data, text='开始实时采集并保存', command=self.start_realtime_monitor)
        self.realtime_read_button.grid(row=2, column=0, pady=5, sticky="ew")
        
        self.stop_read_button = tkinter.Button(frame_data, text='停止采集', command=self.stop_realtime_monitor, state=tkinter.DISABLED)
        self.stop_read_button.grid(row=2, column=1, pady=5, sticky="ew")
        
        frame_data.grid_columnconfigure(1, weight=1)

        # 3. 系统日志区
        tkinter.Label(self.init_windows_name, text='系统日志').grid(row=3, column=0, pady=(5, 0), padx=10, sticky="sw")
        self.log_text = tkinter.Text(self.init_windows_name, width=60, height=10) 
        self.log_text.grid(row=4, column=0, pady=5, padx=10, sticky="nsew")

    def _create_text_widget(self, parent, default_value, row, column, width=20):
        text_widget = tkinter.Text(parent, width=width, height=1)
        text_widget.grid(row=row, column=column, padx=5, pady=2, sticky="ew")
        text_widget.insert(tkinter.END, default_value)
        return text_widget
        
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
            # 滚动日志
            self.log_text.delete(1.0, 2.0)
            self.log_text.insert(tkinter.END, logmsg_in)
        
        self.log_text.see(tkinter.END)
        self.log_text.update()
        
    # --- ADS 连接与读写逻辑 (保持不变) ---
    
    def plc_port_open(self):
        """打开ADS端口并连接到PLC"""
        AmsNetID = self.netID_text.get(1.0, tkinter.END).strip()
        port = self.port_text.get(1.0, tkinter.END).strip()
        
        if self.plc_conn and self.plc_conn.is_open:
            self.write_log_to_text('端口已连接，请勿重复操作。')
            return

        try:
            pyads.open_port() 
            self.plc_conn = pyads.Connection(AmsNetID, int(port))
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

    def process_data(self, raw_data, index_data):
        """
        数据处理：执行环形缓冲区重组，并分离振动/电流数据。
        返回：
        1. 完整波形 (100ms) - 可用于特征计算 (processed_data_100ms)
        2. 增量波形 (T_interval) - 用于高效保存 (incremental_data_dict)
        """
        if raw_data is None or index_data is None:
            return None, None

        # --- 1. 环形缓冲区重组，获取完整的 100ms 连续波形 ---
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
            
            # 时序重组：[P-1...100] + [0...P-2]
            # 注意：如果 write_ptr=1，则 [0:] + [:0]，即原样
            # 如果 write_ptr=100，则 [99:] + [:99]
            # 由于 PLC 写入的是 P-1，所以 P-1 才是最新的，需要放在末尾
            part1 = channel_raw[write_ptr:] # 从 write_ptr 开始到末尾 (旧数据)
            part2 = channel_raw[:write_ptr] # 从开头到 write_ptr (新数据，但时序是 P-1 -> 0)
            
            # 正确的重组逻辑应该是让最新写入的点 (index_array[i]-1) 位于末尾，
            # 确保 continuous_data_100ms[i, :] 是时序连续的 [最旧 ... 最旧-1, ..., 最最新]
            
            # 假设 index_array[i] 指向下一个写入位置 (P)，则 P-1 是最新写入的点。
            # 最新数据是 part2，part1 是旧数据。
            # continuous_data_100ms = np.concatenate((part2, part1)) 是错误的 (时序不连续)
            
            # 根据常见的环形缓冲实现，如果 index_array[i] 是下一个写入位置 P，
            # 那么当前缓冲区内的连续数据应该是：[P, P+1, ..., 99, 0, 1, ..., P-1]
            continuous_data_100ms[i, :] = np.concatenate((channel_raw[write_ptr:], channel_raw[:write_ptr]))

        # --- 2. 100ms 完整波形字典 (用于可能需要特征计算) ---
        vib_channels_100ms = continuous_data_100ms[0:VIBRATION_CHANNELS, :]
        current_channels_100ms = continuous_data_100ms[VIBRATION_CHANNELS:, :]
        
        processed_data_100ms = {
            'Vibration': {
                'X': vib_channels_100ms[0:10, :].flatten(), # 1000 points
                'Y': vib_channels_100ms[10:20, :].flatten(),
                'Z': vib_channels_100ms[20:30, :].flatten(),
            },
            'Current': {
                'A': current_channels_100ms[0, :], # 100 points
                'B': current_channels_100ms[1, :],
                'C': current_channels_100ms[2, :],
            },
        }

        # --- 3. 动态提取增量数据 (T_interval) ---
        
        try:
            # 获取用户配置的采集间隔 (ms)
            T_interval_ms = float(self.interval_text.get(1.0, tkinter.END).strip())
        except ValueError:
             # 如果配置出错，退回到 10ms 默认值
            T_interval_ms = float(DEFAULT_INTERVAL_MS)
        
        # N_inc_points: T_interval_ms 内，每个通道更新的点数 (电流和振动单通道都是 1点/ms)
        N_inc_points = int(T_interval_ms) 
        
        # N_inc_vib_total: 振动总点数（3个方向 x 10个通道/方向 x N_inc_points/通道）
        # 实际是 3个方向 * (10通道/方向 * 1点/ms) * T_interval_ms，但为了对齐日志结构：
        N_inc_vib_total = 3 * VIBRATION_GROUP_SIZE * N_inc_points # 30 * N_inc_points
        
        # 安全检查，确保 N_inc_points 不超过 100
        N_inc_points = min(N_inc_points, SAMPLE_COUNT) 
        
        # 提取增量数据 (位于 continuous_data_100ms 的末尾，末尾是最新数据)
        current_channels_inc = current_channels_100ms[:, -N_inc_points:] 
        vib_channels_inc = vib_channels_100ms[:, -N_inc_points:] 
        
        incremental_data_dict = {
            'Vibration': {
                # X振动: 10通道 * N_inc_points 点
                'X': vib_channels_inc[0:10, :].flatten(), 
                'Y': vib_channels_inc[10:20, :].flatten(),
                'Z': vib_channels_inc[20:30, :].flatten(),
            },
            'Current': {
                'A': current_channels_inc[0, :], 
                'B': current_channels_inc[1, :],
                'C': current_channels_inc[2, :],
            },
            'T_interval_ms': T_interval_ms, # 用于文件保存的描述
            'N_inc_vib_total': N_inc_vib_total # 用于文件保存时振动点数的描述
        }
        
        return processed_data_100ms, incremental_data_dict

    # --- 数据保存逻辑 (只保存增量数据) ---

    def save_processed_data_to_file(self, incremental_data):
        """只保存增量数据 (T_interval 时长)"""
        filepath = self.save_path.get()
        timestamp = self.get_current_time()
        
        vib_x = incremental_data['Vibration']['X'] 
        vib_y = incremental_data['Vibration']['Y']
        vib_z = incremental_data['Vibration']['Z']
        curr_a = incremental_data['Current']['A'] 
        
        T_interval_ms = incremental_data['T_interval_ms'] 
        
        # 根据增量数据字典中携带的信息获取点数
        NUM_VIB_POINTS = len(vib_x) 
        NUM_CURR_POINTS = len(curr_a) 
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 采集时间: {timestamp} (周期序号: {self.sample_index}) ===\n")
                
                # 写入振动数据
                f.write(f"振动增量数据 ({T_interval_ms}ms, {NUM_VIB_POINTS}点)\n")
                f.write("时序序号\tX振动(INT)\tY振动(INT)\tZ振动(INT)\n")
                
                # 振动采样间隔 0.1ms
                time_axis_vib = np.arange(NUM_VIB_POINTS) * (1 / SAMPLING_FREQUENCY)
                for i in range(NUM_VIB_POINTS):
                    f.write(f"{time_axis_vib[i]:.4f}\t{vib_x[i]}\t{vib_y[i]}\t{vib_z[i]}\n")
                
                # 写入电流波形数据
                f.write(f"\n电流增量数据 ({T_interval_ms}ms, {NUM_CURR_POINTS}点)\n")
                f.write(f"采样序号\tA相电流(INT)\tB相电流(INT)\tC相电流(INT)\n")
                for j in range(NUM_CURR_POINTS):
                    f.write(f"{j+1}\t{curr_a[j]}\t{incremental_data['Current']['B'][j]}\t{incremental_data['Current']['C'][j]}\n")
                        
                f.write('='*80 + '\n')
            
            return True
        except Exception as e:
            self.write_log_to_text(f'文件保存失败: {str(e)}')
            return False

    # --- 实时监测控制 ---
    def start_realtime_monitor(self):
        """启动实时采集循环"""
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text('请先打开PLC端口')
            return

        self.is_realtime_running = True
        self.realtime_read_button.config(state=tkinter.DISABLED)
        self.stop_read_button.config(state=tkinter.NORMAL)
        self.write_log_to_text('开始实时采集并保存增量数据...')
        
        self.realtime_monitor_loop()

    def stop_realtime_monitor(self):
        """停止实时采集循环"""
        self.is_realtime_running = False
        self.realtime_read_button.config(state=tkinter.NORMAL)
        self.stop_read_button.config(state=tkinter.DISABLED)
        self.write_log_to_text('已停止实时采集')

    def realtime_monitor_loop(self):
        """实时采集循环：读取、处理、保存"""
        if not self.is_realtime_running:
            return

        try:
            interval = int(self.interval_text.get(1.0, tkinter.END).strip())
            
            # 1. 读取数据
            raw_data, index_data = self._read_data_atomic()
            if raw_data is None:
                self.stop_realtime_monitor() 
                return
            
            # 2. 处理数据: 获取 100ms 完整波形 和 T_interval 增量波形
            processed_data_100ms, incremental_data_dict = self.process_data(raw_data, index_data)
            
            if processed_data_100ms and incremental_data_dict:
                self.sample_index += 1
                self.latest_processed_data_100ms = processed_data_100ms # 缓存最新的完整波形
                
                # 3. 核心：只保存增量数据
                saved = self.save_processed_data_to_file(incremental_data_dict)
                
                if saved:
                    log_msg = f'数据更新完成 (周期 {self.sample_index}). **[增量数据已保存 ({incremental_data_dict["T_interval_ms"]}ms)]**'
                else:
                    log_msg = f'数据处理完成，但文件保存失败 (周期 {self.sample_index})'
                    
                self.write_log_to_text(log_msg)
            
        except ValueError:
            self.write_log_to_text('错误：采集间隔输入无效，请检查。')
        except Exception as e:
            self.write_log_to_text(f'实时监测循环发生错误: {str(e)}')
        
        # 安排下一次运行
        if self.is_realtime_running:
            self.init_windows_name.after(interval, self.realtime_monitor_loop)

# 主程序
def Gui_Start():
    init_window = tkinter.Tk()
    app = DataLoggerApp(init_window)
    init_window.mainloop()

if __name__ == "__main__":
    Gui_Start()