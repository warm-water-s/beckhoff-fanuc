# -*- coding:utf-8 -*-
"""
ADS 数据采集、双特征工况识别、增量保存 与 机床倍率闭环控制系统

功能整合：
1. 实时读取 PLC 环形缓冲区数据 (振动+电流)。
2. 双特征(RMS)工况识别 (STOP/IDLE/CUTTING) + 状态机平滑。
3. 仅在 CUTTING 状态下保存 10ms 增量数据。
4. [新增] 机床进给/主轴倍率的读取、写入与使能控制。
5. [优化] 日志输出频率降低，防止刷屏。

"""
import pyads
import tkinter
from tkinter import messagebox
import time
import numpy as np

# ========== 通道配置 ==========
TOTAL_CHANNELS = 33 
VIBRATION_CHANNELS = 30 
CURRENT_CHANNELS = 3 
VIBRATION_GROUP_SIZE = 10 
SAMPLE_COUNT = 100 
SAMPLING_FREQUENCY = 10000 

# ADS 配置 (PLC内存)
FULL_CHANNELS = 80
FULL_BUFFER_LENGTH = FULL_CHANNELS * SAMPLE_COUNT
GVL_BUFFER_DATATYPE = pyads.PLCTYPE_INT
GVL_BUFFER_GROUP = 0x4020 
GVL_BUFFER_OFFSET = 0x0
INDEX_BUFFER_OFFSET = 16000
INDEX_BUFFER_LENGTH = FULL_CHANNELS 
INDEX_BUFFER_DATATYPE = pyads.PLCTYPE_INT

# ADS 配置 (倍率控制变量)
VAR_CMD_FEED = "GVL.Gvl_Cmd_FeedRate_Set"         # WORD
VAR_CMD_SPINDLE = "GVL.Gvl_Cmd_Spindle_Set"       # WORD
VAR_CMD_ENABLE = "GVL.Gvl_Cmd_Enable_Override"    # BOOL
VAR_ACT_FEED = "GVL.Gvl_Act_FeedRate_Real"        # WORD

# 默认参数
DEFAULT_AMS_NETID = "5.136.192.215.1.1"
DEFAULT_PORT = "851"
DEFAULT_INTERVAL_MS = "10" 
DEFAULT_SAVE_PATH = "processed_sensor_log.txt"
LOG_LINE_NUM = 0

# ========== 工况识别配置 ==========
DEFAULT_IDLE_THRESHOLD = "500"     # 电流低阈值
DEFAULT_VIB_THRESHOLD = "320"      # 振动高阈值
STABILITY_CHECK_COUNT = 5          # 状态机稳定性计数

class DataLoggerApp:
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value=DEFAULT_SAVE_PATH)
        self.plc_conn = None
        self.is_realtime_running = False
        self.sample_index = 0 
        
        # 实时数据缓存
        self.latest_processed_data = None 
        
        # 工况识别状态
        self.cutting_state = 'STOP' 
        self.state_history = [] 
        self.stability_check_count = STABILITY_CHECK_COUNT
        self.idle_threshold = tkinter.StringVar(value=DEFAULT_IDLE_THRESHOLD)
        self.vib_threshold = tkinter.StringVar(value=DEFAULT_VIB_THRESHOLD)
        
        # --- 倍率控制相关变量 ---
        self.var_set_feed = tkinter.StringVar(value="100")      # 设定进给输入
        self.var_set_spindle = tkinter.StringVar(value="100")   # 设定主轴输入
        self.var_act_feed_display = tkinter.StringVar(value="---") # 实际进给显示
        self.var_cmd_feed_display = tkinter.StringVar(value="---") # PLC内设定进给显示
        self.var_cmd_spindle_display = tkinter.StringVar(value="---") # PLC内设定主轴显示
        self.is_override_enabled = False # 本地标记使能状态

        self.set_init_window()

    def set_init_window(self):
        """初始化基础UI界面"""
        self.init_windows_name.title('ADS 数据采集 & 工况识别 & 倍率控制系统')
        self.init_windows_name.geometry('650x750+100+50') # 增加高度以容纳新面板
        self.init_windows_name.grid_columnconfigure(0, weight=1)
        # 调整权重，确保日志在底部拉伸
        for i in range(5): self.init_windows_name.grid_rowconfigure(i, weight=0)
        self.init_windows_name.grid_rowconfigure(5, weight=1) 

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
        
        self.realtime_read_button = tkinter.Button(frame_data, text='开始实时采集并识别', command=self.start_realtime_monitor, bg="#d0f0c0")
        self.realtime_read_button.grid(row=2, column=0, pady=5, sticky="ew")
        
        self.stop_read_button = tkinter.Button(frame_data, text='停止采集', command=self.stop_realtime_monitor, state=tkinter.DISABLED, bg="#f0d0d0")
        self.stop_read_button.grid(row=2, column=1, pady=5, sticky="ew")
        frame_data.grid_columnconfigure(1, weight=1)
        
        # 3. 工况识别配置组 
        frame_threshold = tkinter.LabelFrame(self.init_windows_name, text="工况识别配置 (RMS双特征)", padx=5, pady=5)
        frame_threshold.grid(row=2, column=0, pady=5, padx=10, sticky="ew")
        
        tkinter.Label(frame_threshold, text='电流停转阈值(低)').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        tkinter.Entry(frame_threshold, textvariable=self.idle_threshold, width=15).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        
        tkinter.Label(frame_threshold, text='振动切削阈值(高)').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        tkinter.Entry(frame_threshold, textvariable=self.vib_threshold, width=15).grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        frame_threshold.grid_columnconfigure(1, weight=1)

        # 4. [新增] 机床倍率控制组
        frame_override = tkinter.LabelFrame(self.init_windows_name, text="机床倍率控制 (闭环读写)", padx=5, pady=5, fg="blue")
        frame_override.grid(row=3, column=0, pady=5, padx=10, sticky="ew")
        
        # 4.1 进给倍率
        tkinter.Label(frame_override, text="设定进给:").grid(row=0, column=0, sticky="w")
        tkinter.Entry(frame_override, textvariable=self.var_set_feed, width=6).grid(row=0, column=1, padx=2)
        tkinter.Button(frame_override, text="写入", command=self.write_feed_override, width=5).grid(row=0, column=2, padx=5)
        tkinter.Label(frame_override, text="PLC当前:").grid(row=0, column=3, padx=(10,0))
        tkinter.Label(frame_override, textvariable=self.var_cmd_feed_display, fg="blue", font=("Arial", 10, "bold")).grid(row=0, column=4, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=0, column=5)

        # 4.2 主轴倍率
        tkinter.Label(frame_override, text="设定主轴:").grid(row=1, column=0, sticky="w")
        tkinter.Entry(frame_override, textvariable=self.var_set_spindle, width=6).grid(row=1, column=1, padx=2)
        tkinter.Button(frame_override, text="写入", command=self.write_spindle_override, width=5).grid(row=1, column=2, padx=5)
        tkinter.Label(frame_override, text="PLC当前:").grid(row=1, column=3, padx=(10,0))
        tkinter.Label(frame_override, textvariable=self.var_cmd_spindle_display, fg="blue", font=("Arial", 10, "bold")).grid(row=1, column=4, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=1, column=5)

        # 4.3 实际反馈
        tkinter.Label(frame_override, text="--------------------------------------------------").grid(row=2, column=0, columnspan=6)
        tkinter.Label(frame_override, text="机床实际执行进给:").grid(row=3, column=0, columnspan=2, sticky="e")
        tkinter.Label(frame_override, textvariable=self.var_act_feed_display, fg="red", font=("Arial", 12, "bold")).grid(row=3, column=2, columnspan=2, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=3, column=4, sticky="w")

        # 4.4 使能控制
        tkinter.Label(frame_override, text="控制权限:").grid(row=4, column=0, sticky="w", pady=5)
        self.btn_enable_override = tkinter.Button(frame_override, text="OFF (面板控制)", bg="gray", command=self.toggle_override_enable, width=20)
        self.btn_enable_override.grid(row=4, column=1, columnspan=4, pady=5)


        # 5. 系统日志区
        tkinter.Label(self.init_windows_name, text='系统日志').grid(row=4, column=0, pady=(5, 0), padx=10, sticky="sw")
        self.log_text = tkinter.Text(self.init_windows_name, width=60, height=10) 
        self.log_text.grid(row=5, column=0, pady=5, padx=10, sticky="nsew")

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
        
    # --- ADS 连接与读写逻辑 ---
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
            self.open_port_button.config(bg="#a0ffa0")
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


    # ============================================================
    # --- [核心接口] 供外部模型/算法调用的 API ---
    # ============================================================
    def api_set_feed_override(self, value: int):
        """
        [外部接口] 设置进给倍率
        :param value: 整数, 范围 0-150
        :return: Boolean (是否写入成功)
        """
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text("API调用失败: PLC未连接")
            return False
            
        # 1. 范围限制
        safe_value = max(0, min(150, int(value)))
        
        try:
            # 2. 写入 PLC
            self.plc_conn.write_by_name(VAR_CMD_FEED, safe_value, pyads.PLCTYPE_WORD)
            
            # 3. 同步更新 UI 输入框 (让操作员也能看到模型改了什么)
            self.var_set_feed.set(str(safe_value))
            
            self.write_log_to_text(f"[API] 外部指令执行: 设定进给 -> {safe_value}%")
            return True
        except Exception as e:
            self.write_log_to_text(f"[API] 进给写入异常: {e}")
            return False

    def api_set_spindle_override(self, value: int):
        """
        [外部接口] 设置主轴倍率
        :param value: 整数, 范围 50-120
        :return: Boolean
        """
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text("API调用失败: PLC未连接")
            return False
            
        safe_value = max(50, min(120, int(value)))
        
        try:
            self.plc_conn.write_by_name(VAR_CMD_SPINDLE, safe_value, pyads.PLCTYPE_WORD)
            
            # 同步更新 UI
            self.var_set_spindle.set(str(safe_value))
            
            self.write_log_to_text(f"[API] 外部指令执行: 设定主轴 -> {safe_value}%")
            return True
        except Exception as e:
            self.write_log_to_text(f"[API] 主轴写入异常: {e}")
            return False

    def api_set_control_enable(self, enable: bool):
        """
        [外部接口] 设置控制权限 (True=HMI接管, False=面板控制)
        :param enable: Boolean
        :return: Boolean
        """
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text("API调用失败: PLC未连接")
            return False
            
        try:
            # 1. 写入 PLC
            self.plc_conn.write_by_name(VAR_CMD_ENABLE, enable, pyads.PLCTYPE_BOOL)
            
            # 2. 更新内部状态和 UI 按钮样式
            self.is_override_enabled = enable
            if self.is_override_enabled:
                self.btn_enable_override.config(text="ON (HMI 接管)", bg="#00ff00")
                self.write_log_to_text("[API] 外部指令: 权限已切换至 HMI")
            else:
                self.btn_enable_override.config(text="OFF (面板控制)", bg="gray")
                self.write_log_to_text("[API] 外部指令: 权限已释放给机床面板")
            return True
        except Exception as e:
            self.write_log_to_text(f"[API] 权限切换异常: {e}")
            return False


    # ============================================================
    # --- [UI 回调] 界面按钮点击事件 (内部调用 API) ---
    # ============================================================
    def write_feed_override(self):
        """UI按钮: 写入进给倍率"""
        if not self.plc_conn or not self.plc_conn.is_open:
            messagebox.showwarning("警告", "请先连接PLC")
            return
        try:
            # 获取输入框的值
            val = int(self.var_set_feed.get())
            # 调用 API
            self.api_set_feed_override(val)
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数")

    def write_spindle_override(self):
        """UI按钮: 写入主轴倍率"""
        if not self.plc_conn or not self.plc_conn.is_open:
            messagebox.showwarning("警告", "请先连接PLC")
            return
        try:
            val = int(self.var_set_spindle.get())
            # 调用 API
            self.api_set_spindle_override(val)
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数")

    def toggle_override_enable(self):
        """UI按钮: 切换使能状态"""
        if not self.plc_conn or not self.plc_conn.is_open:
            messagebox.showwarning("警告", "请先连接PLC")
            return
        
        # 翻转当前状态
        target_state = not self.is_override_enabled
        # 调用 API
        self.api_set_control_enable(target_state)

    def update_override_status(self):
        """同步 PLC 当前的所有状态 (Act 和 Cmd)"""
        if not self.plc_conn or not self.plc_conn.is_open:
            return
        try:
            # 读取实际反馈 (Actual)
            act_feed = self.plc_conn.read_by_name(VAR_ACT_FEED, pyads.PLCTYPE_WORD)
            self.var_act_feed_display.set(str(act_feed))

            # 读取当前设定 (Command)
            cmd_feed = self.plc_conn.read_by_name(VAR_CMD_FEED, pyads.PLCTYPE_WORD)
            self.var_cmd_feed_display.set(str(cmd_feed))

            cmd_spindle = self.plc_conn.read_by_name(VAR_CMD_SPINDLE, pyads.PLCTYPE_WORD)
            self.var_cmd_spindle_display.set(str(cmd_spindle))
            
            # 可以在这里同步 enable 按钮状态，如果需要双向同步
        except Exception:
            pass 

    # --- 数据处理与特征 ---
    def process_data(self, raw_data, index_data):
        """数据处理：重组并提取增量"""
        if raw_data is None or index_data is None:
            return None, None

        # 1. 重组环形缓冲区
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
            
            part1 = channel_raw[write_ptr - 1:] 
            part2 = channel_raw[:write_ptr - 1]
            continuous_data_100ms[i, :] = np.concatenate((part1, part2))

        # 2. 分离振动和电流 (完整波形)
        vib_channels_100ms = continuous_data_100ms[0:VIBRATION_CHANNELS, :]
        current_channels_100ms = continuous_data_100ms[VIBRATION_CHANNELS:, :]
        
        processed_data_100ms = {
            'Vibration': {
                'X': vib_channels_100ms[0:10, :].flatten(), 
                'Y': vib_channels_100ms[10:20, :].flatten(),
                'Z': vib_channels_100ms[20:30, :].flatten(),
            },
            'Current': {
                'A': current_channels_100ms[0, :], 
                'B': current_channels_100ms[1, :],
                'C': current_channels_100ms[2, :],
            },
        }

        # 3. 动态提取增量数据
        try:
            T_interval_ms = float(self.interval_text.get(1.0, tkinter.END).strip())
        except ValueError:
            T_interval_ms = float(DEFAULT_INTERVAL_MS)
        
        N_inc_curr = int(T_interval_ms) 
        N_inc_curr = min(N_inc_curr, SAMPLE_COUNT) 
        
        current_channels_inc = current_channels_100ms[:, -N_inc_curr:] 
        vib_channels_inc = vib_channels_100ms[:, -N_inc_curr:] 
        
        incremental_data_dict = {
            'Vibration': {
                'X': vib_channels_inc[0:10, :].flatten(), 
                'Y': vib_channels_inc[10:20, :].flatten(),
                'Z': vib_channels_inc[20:30, :].flatten(),
            },
            'Current': {
                'A': current_channels_inc[0, :], 
                'B': current_channels_inc[1, :],
                'C': current_channels_inc[2, :],
            },
            'T_interval_ms': T_interval_ms 
        }
        
        return processed_data_100ms, incremental_data_dict

    def calculate_current_feature(self, current_data):
        """计算电流 RMS"""
        curr_a = current_data['A']
        curr_b = current_data['B']
        curr_c = current_data['C']
        
        rms_sq_a = np.mean(curr_a.astype(np.float64)**2)
        rms_sq_b = np.mean(curr_b.astype(np.float64)**2)
        rms_sq_c = np.mean(curr_c.astype(np.float64)**2)
        
        avg_rms = (np.sqrt(rms_sq_a) + np.sqrt(rms_sq_b) + np.sqrt(rms_sq_c)) / 3
        return avg_rms

    def calculate_vibration_feature(self, vib_data):
        """计算振动 Z 轴 RMS"""
        vib_z = vib_data['Z']
        rms_vib_z = np.sqrt(np.mean(vib_z.astype(np.float64)**2))
        return rms_vib_z
    
    def classify_cutting_state(self, processed_data):
        """双特征工况识别 + 状态机"""
        current_rms_value = self.calculate_current_feature(processed_data['Current'])
        vib_rms_value = self.calculate_vibration_feature(processed_data['Vibration'])

        try:
            idle_thresh = float(self.idle_threshold.get())
            vib_thresh = float(self.vib_threshold.get())
        except ValueError:
            idle_thresh = float(DEFAULT_IDLE_THRESHOLD)
            vib_thresh = float(DEFAULT_VIB_THRESHOLD)
            
        current_instant_state = 'STOP'
        if current_rms_value >= idle_thresh:
            if vib_rms_value >= vib_thresh:
                current_instant_state = 'CUTTING'
            else:
                current_instant_state = 'IDLE'

        # 状态机平滑
        self.state_history.append(current_instant_state)
        if len(self.state_history) > self.stability_check_count:
            self.state_history.pop(0) 
            
        state_counts = {state: self.state_history.count(state) for state in ['STOP', 'IDLE', 'CUTTING']}
        majority_count = self.stability_check_count
        
        prev_state = self.cutting_state
        
        if self.cutting_state != 'CUTTING' and state_counts['CUTTING'] >= majority_count:
            self.cutting_state = 'CUTTING'
        elif self.cutting_state == 'CUTTING' and state_counts['IDLE'] >= majority_count:
            self.cutting_state = 'IDLE'
        elif state_counts['STOP'] >= majority_count:
            if self.cutting_state != 'STOP':
                self.cutting_state = 'STOP'
        elif state_counts['IDLE'] >= majority_count:
            self.cutting_state = 'IDLE'
        
        # 仅在状态发生改变时打印日志
        if prev_state != self.cutting_state:
            self.write_log_to_text(f'>>> ⚠️ 状态切换: {prev_state} -> {self.cutting_state} (Vib:{vib_rms_value:.1f}, Curr:{current_rms_value:.1f})')
            
        return self.cutting_state 

    def send_data_to_model(self, processed_data):
        """
            占位：模型接口
                # 假设这是你的算法返回的结果
                suggested_feed_rate = 120 
                should_stop_optimization = False
                
                # === 示例：直接调用刚才写的 API ===
                
                # 1. 确保开启 HMI 控制权限
                if not self.is_override_enabled:
                        self.api_set_control_enable(True)
                        
                # 2. 写入优化后的倍率
                self.api_set_feed_override(suggested_feed_rate)
                
                # 3. 如果需要，也可以控制主轴
                # self.api_set_spindle_override(100)
        """
        pass 
    
        
    def save_processed_data_to_file(self, incremental_data):
        """保存增量数据"""
        filepath = self.save_path.get()
        timestamp = self.get_current_time()
        
        vib_x = incremental_data['Vibration']['X'] 
        vib_y = incremental_data['Vibration']['Y']
        vib_z = incremental_data['Vibration']['Z']
        curr_a = incremental_data['Current']['A'] 
        
        T_interval_ms = incremental_data['T_interval_ms']
        NUM_VIB_POINTS = len(vib_x) 
        NUM_CURR_POINTS = len(curr_a) 
        
        try:
            with open(filepath, 'a', encoding='utf-8') as f:
                f.write(f"\n=== 采集时间: {timestamp} (周期序号: {self.sample_index}) ===\n")
                
                f.write(f"振动增量数据 ({T_interval_ms}ms, {NUM_VIB_POINTS}点)\n")
                f.write("时序序号\tX振动(INT)\tY振动(INT)\tZ振动(INT)\n")
                
                time_axis_vib = np.arange(NUM_VIB_POINTS) * (1 / SAMPLING_FREQUENCY)
                for i in range(NUM_VIB_POINTS):
                    f.write(f"{time_axis_vib[i]:.4f}\t{vib_x[i]}\t{vib_y[i]}\t{vib_z[i]}\n")
                
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
        """实时采集循环"""
        if not self.is_realtime_running:
            return

        try:
            interval = int(self.interval_text.get(1.0, tkinter.END).strip())
            
            # 1. [新增] 每一帧都同步倍率状态 (不打印日志，静默更新UI)
            self.update_override_status()

            # 2. 读取数据
            raw_data, index_data = self._read_data_atomic()
            if raw_data is None:
                self.stop_realtime_monitor() 
                return
            
            processed_data_100ms, incremental_data_dict = self.process_data(raw_data, index_data)
            
            if processed_data_100ms and incremental_data_dict:
                self.sample_index += 1
                self.latest_processed_data = processed_data_100ms
                
                # 3. 工况识别
                current_state = self.classify_cutting_state(processed_data_100ms)
                
                # 4. 数据保存逻辑
                save_msg = ""
                if current_state == 'CUTTING':
                    saved = self.save_processed_data_to_file(incremental_data_dict)
                    self.send_data_to_model(processed_data_100ms) 
                    
                    if saved:
                        save_msg = " [已保存]"
                    else:
                        save_msg = " [保存失败]"
                
                # 5. [优化] 日志输出控制：降低频率
                # 策略：每 50 个周期(约0.5s)输出一次心跳日志，或者当有数据保存时输出
                if self.sample_index % 50 == 0 or (current_state == 'CUTTING' and self.sample_index % 10 == 0):
                     self.write_log_to_text(f'周期 {self.sample_index}: 状态={current_state}{save_msg}')
            
        except ValueError:
            self.write_log_to_text('错误：采集间隔或阈值输入无效。')
        except Exception as e:
            self.write_log_to_text(f'监测循环错误: {str(e)}')
            self.stop_realtime_monitor()
        
        if self.is_realtime_running:
            self.init_windows_name.after(interval, self.realtime_monitor_loop)


# 主程序
def Gui_Start():
    init_window = tkinter.Tk()
    app = DataLoggerApp(init_window)
    init_window.mainloop()

if __name__ == "__main__":
    Gui_Start()