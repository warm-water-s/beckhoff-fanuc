# -*- coding:utf-8 -*-
"""
ADS 数据采集、双特征工况识别与增量式保存系统

功能：
1. 实时读取 PLC 环形缓冲区数据。
2. 使用 100ms 完整波形计算 RMS 特征,判断当前工况:STOP, IDLE, CUTTING。
3. 使用状态机平滑切换，过滤进刀/退刀的过渡数据。
4. 仅在 CUTTING 状态下，提取最新的 10ms 增量数据进行高效存储。

数据文件(processed_sensor_log.txt)结构 (现在只保存 10ms 增量数据)
    === 采集时间: 2025-12-03 21:00:00 (周期序号: 1) ===
    振动增量数据 (10ms, 100点)
    时序序号 X振动(INT) Y振动(INT) Z振动(INT)
    ... (共 100 行振动数据)

    电流增量数据 (10ms, 10个采样点)
    采样序号 A相电流(INT) B相电流(INT) C相电流(INT)
    ... (共 10 行电流数据)
    ===================================================
"""

"""
    待修改功能:
        1、日志区对于周期与状态的更新太频繁了,最好一段时间更新一次
        2、对于日志部分可以频繁的可以更新到文件中,显示部分仅显示当前状态，如果状态变化则改变，或者根据1修改

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

# ========== 工况识别配置 ==========
DEFAULT_IDLE_THRESHOLD = "500"     # 电流低阈值：区分停转和运行 (原始INT RMS)
DEFAULT_VIB_THRESHOLD = "320"    # 振动高阈值：区分空转和切削 (原始INT RMS)
STABILITY_CHECK_COUNT = 5         # 连续多少个 10ms 周期判断为稳定状态切换 (50ms 延迟)
# ============================================

class DataLoggerApp:
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value=DEFAULT_SAVE_PATH)
        self.plc_conn = None
        self.is_realtime_running = False
        self.sample_index = 0 
        
        # 实时数据缓存
        self.latest_processed_data = None 
        
        # 工况识别状态和阈值
        self.cutting_state = 'STOP'     # 状态: 'STOP', 'IDLE', 'CUTTING'
        self.state_history = []         # 状态历史记录，用于平滑判断
        self.stability_check_count = STABILITY_CHECK_COUNT
        self.idle_threshold = tkinter.StringVar(value=DEFAULT_IDLE_THRESHOLD)
        self.vib_threshold = tkinter.StringVar(value=DEFAULT_VIB_THRESHOLD)
        

        self.set_init_window()

    # --- UI 初始化与日志功能 ---
    def set_init_window(self):
        """初始化基础UI界面"""
        self.init_windows_name.title('ADS 数据采集与工况识别系统')
        self.init_windows_name.geometry('600x600+100+100') 
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
        
        self.realtime_read_button = tkinter.Button(frame_data, text='开始实时采集并识别', command=self.start_realtime_monitor)
        self.realtime_read_button.grid(row=2, column=0, pady=5, sticky="ew")
        
        self.stop_read_button = tkinter.Button(frame_data, text='停止采集', command=self.stop_realtime_monitor, state=tkinter.DISABLED)
        self.stop_read_button.grid(row=2, column=1, pady=5, sticky="ew")
        
        frame_data.grid_columnconfigure(1, weight=1)
        
        # 3. 工况识别配置组 
        frame_threshold = tkinter.LabelFrame(self.init_windows_name, text="工况识别配置 (双特征)", padx=5, pady=5)
        frame_threshold.grid(row=2, column=0, pady=5, padx=10, sticky="ew")
        
        # 电流低阈值：用于区分停转和运行 (基于电流RMS)
        tkinter.Label(frame_threshold, text='电流停转阈值(低)').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        tkinter.Entry(frame_threshold, textvariable=self.idle_threshold, width=15).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        # 振动高阈值：用于区分空转和切削 (基于振动Z轴RMS)
        tkinter.Label(frame_threshold, text='振动切削阈值(高)').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        tkinter.Entry(frame_threshold, textvariable=self.vib_threshold, width=15).grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        
        frame_threshold.grid_columnconfigure(1, weight=1)

        # 4. 系统日志区
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
            self.log_text.delete(1.0, 2.0)
            self.log_text.insert(tkinter.END, logmsg_in)
        
        self.log_text.see(tkinter.END)
        self.log_text.update()
        
    # --- ADS 连接与读写逻辑 (保持不变) ---
    def plc_port_open(self):
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
        1. 完整波形 (100ms) - 用于特征计算
        2. 增量波形 (T_interval) - 用于高效保存
        """
        if raw_data is None or index_data is None:
            return None, None

        # --- 1. 环形缓冲区重组，获取完整的 100ms 连续波形 (保持不变) ---
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
            part1 = channel_raw[write_ptr - 1:] 
            part2 = channel_raw[:write_ptr - 1]
            continuous_data_100ms[i, :] = np.concatenate((part1, part2))

        # 2. 从完整的 100ms 数据中分离振动和电流 (用于 RMS 特征计算)
        vib_channels_100ms = continuous_data_100ms[0:VIBRATION_CHANNELS, :]
        current_channels_100ms = continuous_data_100ms[VIBRATION_CHANNELS:, :]
        
        # processed_data_100ms (用于 RMS 计算) 保持不变
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

        # --- 3. 动态提取增量数据 (N_inc) ---
        
        try:
            # 获取用户配置的采集间隔 (ms)
            T_interval_ms = float(self.interval_text.get(1.0, tkinter.END).strip())
        except ValueError:
             # 如果配置出错，退回到 10ms 默认值
            T_interval_ms = float(DEFAULT_INTERVAL_MS)
        
        # 电流采样频率 1000Hz (1点/ms)
        N_inc_curr = int(T_interval_ms) 
        
        # 安全检查，确保 N_inc 不超过 100
        N_inc_curr = min(N_inc_curr, SAMPLE_COUNT) 
        
        # 提取增量数据 (位于 continuous_data_100ms 的末尾)
        # 提取电流（3相 x N_inc_curr 点）
        current_channels_inc = current_channels_100ms[:, -N_inc_curr:] 
        # 提取振动（30通道 x N_inc_curr 点）
        # 注意：这里 vib_channels_100ms 是 (30 x 100) 矩阵
        vib_channels_inc = vib_channels_100ms[:, -N_inc_curr:] 
        
        incremental_data_dict = {
            'Vibration': {
                # X振动: 10通道 * N_inc_curr 点 = N_inc_vib 点
                'X': vib_channels_inc[0:10, :].flatten(), 
                'Y': vib_channels_inc[10:20, :].flatten(),
                'Z': vib_channels_inc[20:30, :].flatten(),
            },
            'Current': {
                'A': current_channels_inc[0, :], 
                'B': current_channels_inc[1, :],
                'C': current_channels_inc[2, :],
            },
            'T_interval_ms': T_interval_ms # 将增量时长带出，用于文件保存的描述
        }
        
        return processed_data_100ms, incremental_data_dict

    # --- 数据特征计算 (新增) ---
    def calculate_current_feature(self, current_data):
        """计算三相电流的平均均方根 (RMS)"""
        curr_a = current_data['A']
        curr_b = current_data['B']
        curr_c = current_data['C']
        
        rms_sq_a = np.mean(curr_a.astype(np.float64)**2)
        rms_sq_b = np.mean(curr_b.astype(np.float64)**2)
        rms_sq_c = np.mean(curr_c.astype(np.float64)**2)
        
        avg_rms = (np.sqrt(rms_sq_a) + np.sqrt(rms_sq_b) + np.sqrt(rms_sq_c)) / 3
        return avg_rms

    def calculate_vibration_feature(self, vib_data):
        """计算 Z 轴振动信号的均方根 (RMS) 作为切削特征"""
        vib_z = vib_data['Z']
        rms_vib_z = np.sqrt(np.mean(vib_z.astype(np.float64)**2))
        return rms_vib_z
    
    def classify_cutting_state(self, processed_data):
        """双特征工况识别，使用状态机平滑过渡。"""
        current_rms_value = self.calculate_current_feature(processed_data['Current'])
        vib_rms_value = self.calculate_vibration_feature(processed_data['Vibration'])

        try:
            idle_thresh = float(self.idle_threshold.get())
            vib_thresh = float(self.vib_threshold.get())
        except ValueError:
            self.write_log_to_text('警告: 阈值输入无效，使用默认值。')
            idle_thresh = float(DEFAULT_IDLE_THRESHOLD)
            vib_thresh = float(DEFAULT_VIB_THRESHOLD)
            
        # 瞬时状态判断 (用于填入历史记录)
        current_instant_state = 'STOP'
        if current_rms_value >= idle_thresh:
            if vib_rms_value >= vib_thresh:
                current_instant_state = 'CUTTING'
            else:
                current_instant_state = 'IDLE'

        # 历史记录和状态平滑 (FSM)
        self.state_history.append(current_instant_state)
        if len(self.state_history) > self.stability_check_count:
            self.state_history.pop(0) 
            
        state_counts = {state: self.state_history.count(state) for state in ['STOP', 'IDLE', 'CUTTING']}
        majority_count = self.stability_check_count
        
        # 状态切换逻辑 (要求连续 N 个周期一致)
        if self.cutting_state != 'CUTTING' and state_counts['CUTTING'] >= majority_count:
            self.cutting_state = 'CUTTING'
            self.write_log_to_text(f'>>> ⚠️ **工况切换: CUTTING** ⚠️ (Vib RMS: {vib_rms_value:.2f})')
        elif self.cutting_state == 'CUTTING' and state_counts['IDLE'] >= majority_count:
            self.cutting_state = 'IDLE'
            self.write_log_to_text(f'>>> ✅ **工况切换: IDLE** ✅ (Vib RMS: {vib_rms_value:.2f})')
        elif state_counts['STOP'] >= majority_count:
            if self.cutting_state != 'STOP':
                self.cutting_state = 'STOP'
                self.write_log_to_text(f'>>> 🛑 **工况切换: STOP** 🛑 (Curr RMS: {current_rms_value:.2f})')
        elif state_counts['IDLE'] >= majority_count:
            self.cutting_state = 'IDLE'
            
        return self.cutting_state 

    def send_data_to_model(self, processed_data):
        """占位函数：在这里将 processed_data 传入您的后续模型 (建议传入 100ms 完整波形)"""
        # TODO: 请根据您的模型接口修改此函数。
        pass 
        
    # --- 数据保存逻辑 (只保存增量数据) ---
    def save_processed_data_to_file(self, incremental_data):
        """保存增量数据 (T_interval 时长)"""
        filepath = self.save_path.get()
        timestamp = self.get_current_time()
        
        vib_x = incremental_data['Vibration']['X'] 
        vib_y = incremental_data['Vibration']['Y']
        vib_z = incremental_data['Vibration']['Z']
        curr_a = incremental_data['Current']['A'] 
        
        T_interval_ms = incremental_data['T_interval_ms'] # 获取增量时长
        
        # 根据增量数据计算点数
        NUM_VIB_POINTS = len(vib_x) # 10 * N_inc_curr
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
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text('请先打开PLC端口')
            return

        self.is_realtime_running = True
        self.realtime_read_button.config(state=tkinter.DISABLED)
        self.stop_read_button.config(state=tkinter.NORMAL)
        self.write_log_to_text('开始实时采集并识别工况...')
        
        self.realtime_monitor_loop()

    def stop_realtime_monitor(self):
        self.is_realtime_running = False
        self.realtime_read_button.config(state=tkinter.NORMAL)
        self.stop_read_button.config(state=tkinter.DISABLED)
        self.write_log_to_text('已停止实时采集')

    def realtime_monitor_loop(self):
        """实时采集循环：读取、处理、识别、保存"""
        if not self.is_realtime_running:
            return

        try:
            # 1. 获取采集间隔
            interval = int(self.interval_text.get(1.0, tkinter.END).strip())
            
            # 2. 读取数据
            raw_data, index_data = self._read_data_atomic()
            if raw_data is None:
                # 读取失败，停止循环
                self.stop_realtime_monitor() 
                return
            
            # 3. 处理数据 (返回 100ms 完整波形用于特征计算, 和 T_interval 增量波形用于保存)
            # incremental_data_dict 现在包含 T_interval_ms，用于动态调整增量大小
            processed_data_100ms, incremental_data_dict = self.process_data(raw_data, index_data)
            
            if processed_data_100ms and incremental_data_dict:
                self.sample_index += 1
                self.latest_processed_data = processed_data_100ms
                
                # 4. 工况识别 (基于 100ms 完整波形计算的特征)
                current_state = self.classify_cutting_state(processed_data_100ms)
                
                log_msg = f'数据更新完成 (周期 {self.sample_index}). 状态: **{current_state}**'
                
                # 5. 核心数据划分逻辑：只保存 CUTTING 状态的增量数据
                if current_state == 'CUTTING':
                    # 使用动态提取的增量数据进行保存
                    saved = self.save_processed_data_to_file(incremental_data_dict)
                    
                    # 将 100ms 完整波形数据传入模型 (如果需要)
                    self.send_data_to_model(processed_data_100ms) 
                    
                    if saved:
                        log_msg += ' **[增量切削数据已保存并送入模型]**'
                    else:
                        log_msg += ' **[增量切削数据保存失败]**'
                else:
                    log_msg += ' [非切削数据已跳过处理]'

                self.write_log_to_text(log_msg)
            
        except ValueError:
            self.write_log_to_text('错误：采集间隔或阈值输入无效，请检查。')
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