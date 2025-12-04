# -*- coding:utf-8 -*-
"""
数据文件(processed_sensor_log.txt)结构
    === 采集时间: 2025-12-03 21:00:00 (样本序号: 1) ===
    时间(s) X振动(INT)  Y振动(INT)  Z振动(INT)
    0.0000  [振动X点1]  [振动Y点1]  [振动Z点1]
    0.0001  [振动X点2]  [振动Y点2]  [振动Z点2]
    ... (共 1000 行振动数据)
    0.0999  [振动X点1000] [振动Y点1000] [振动Z点1000]

    电流数据(A/B/C相,100个采样点,时序连续)
    采样序号 A相电流(INT) B相电流(INT) C相电流(INT)
    1           [电流A点1] [电流B点1] [电流C点1]
    2           [电流A点2] [电流B点2] [电流C点2]
    ... (共 100 行电流数据)
    100         [电流A点100] [电流B点100] [电流C点100]
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
SAMPLE_COUNT = 100 # 每个通道采样点数
SAMPLING_FREQUENCY = 10000 # 采样频率10000Hz

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
DEFAULT_INTERVAL_MS = "10" # 采集周期
DEFAULT_SAVE_PATH = "processed_sensor_log.txt"
LOG_LINE_NUM = 0

class DataLoggerApp:
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value=DEFAULT_SAVE_PATH)
        self.plc_conn = None
        self.is_realtime_running = False
        self.sample_index = 0 # 用于记录总采样次数/点数
        
        # 实时数据缓存 (可选，但保留了计数和日志功能)
        self.latest_processed_data = None 
        self.set_init_window()

    # --- UI 初始化与日志功能 (简化) ---
    
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
        
    # --- ADS 连接与读写逻辑 ---
    
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
        返回已处理的振动和电流数据的字典。
        """
        if raw_data is None or index_data is None:
            return None

        # 1. 环形缓冲区重组 (与您原始代码逻辑一致)
        try:
            raw_matrix = np.array(raw_data, dtype=np.int16).reshape(FULL_CHANNELS, SAMPLE_COUNT)
            index_array = np.array(index_data, dtype=np.int16)
        except ValueError as e:
            self.write_log_to_text(f"数据重塑或索引转换错误: {e}")
            return None
        
        continuous_data = np.zeros((TOTAL_CHANNELS, SAMPLE_COUNT), dtype=np.int16)
        
        for i in range(TOTAL_CHANNELS):
            write_ptr = index_array[i] 
            channel_raw = raw_matrix[i, :]
            
            # 时序重组：[P-1...100] + [0...P-2]
            part1 = channel_raw[write_ptr - 1:] 
            part2 = channel_raw[:write_ptr - 1]
            continuous_data[i, :] = np.concatenate((part1, part2))

        # 2. 分离和处理振动数据
        vib_channels_data = continuous_data[0:VIBRATION_CHANNELS, :]
        vib_data = {
            'X': vib_channels_data[0:10, :].flatten(),
            'Y': vib_channels_data[10:20, :].flatten(),
            'Z': vib_channels_data[20:30, :].flatten(),
        }

        # 3. 处理电流数据
        current_channels_data = continuous_data[VIBRATION_CHANNELS:, :]
        current_data = {
            'A': current_channels_data[0, :],
            'B': current_channels_data[1, :],
            'C': current_channels_data[2, :],
        }
        """
            {
                'Vibration': {
                    'X': numpy.array([1000 points]), 
                    'Y': numpy.array([1000 points]), 
                    'Z': numpy.array([1000 points])
                },
                'Current': {
                    'A': numpy.array([100 points]), 
                    'B': numpy.array([100 points]), 
                    'C': numpy.array([100 points])
                },
            }
        """
        return {
            'Vibration': vib_data,
            'Current': current_data
        }

    # --- 数据保存逻辑 (保存处理后的数据) ---

    def save_processed_data_to_file(self, processed_data):
        """保存经过环形缓冲区重组后的数据"""
        filepath = self.save_path.get()
        timestamp = self.get_current_time()
        
        vib_x = processed_data['Vibration']['X']
        vib_y = processed_data['Vibration']['Y']
        vib_z = processed_data['Vibration']['Z']
        curr_a = processed_data['Current']['A']
        curr_b = processed_data['Current']['B']
        curr_c = processed_data['Current']['C']

        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 采集时间: {timestamp} (样本序号: {self.sample_index}) ===\n")
                
                # 写入振动数据（已重组，1000个点）
                f.write("时间(s)\tX振动(INT)\tY振动(INT)\tZ振动(INT)\n")
                time_axis = np.arange(len(vib_x)) / SAMPLING_FREQUENCY
                for i in range(len(time_axis)):
                    f.write(f"{time_axis[i]:.4f}\t{vib_x[i]}\t{vib_y[i]}\t{vib_z[i]}\n")
                
                # 写入电流波形数据（已重组，100个点）
                f.write("\n电流数据（A/B/C相，100个采样点，时序连续）\n")
                f.write(f"采样序号\tA相电流(INT)\tB相电流(INT)\tC相电流(INT)\n")
                for j in range(SAMPLE_COUNT):
                    f.write(f"{j+1}\t{curr_a[j]}\t{curr_b[j]}\t{curr_c[j]}\n")
                    
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
        self.write_log_to_text('开始实时采集并保存数据...')
        
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
            
            # 2. 处理数据 (核心逻辑)
            processed_data = self.process_data(raw_data, index_data)
            
            if processed_data:
                self.sample_index += 1
                self.latest_processed_data = processed_data # 缓存最新的处理结果
                
                # 3. 保存处理后的数据
                saved = self.save_processed_data_to_file(processed_data)
                
                if saved:
                    log_msg = f'数据更新并保存完成 (周期 {self.sample_index})'
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