# -*- coding:utf-8 -*-
"""
ADS 数据采集、双特征工况识别、增量保存(双CSV文件) 与 机床倍率闭环控制系统

功能整合：
1. 实时读取 PLC 环形缓冲区数据 (振动+电流)。
2. 双特征(RMS)工况识别 (STOP/IDLE/CUTTING) + 状态机平滑。
3. 仅在 CUTTING 状态下，将数据分别保存为 _Vib.csv 和 _Curr.csv，自动创建目录。
4. 机床进给/主轴倍率的读取、写入与使能控制。
5. 日志输出频率优化，实时显示 RMS 值。
6. 增加 Cmd (设定值) 的回读显示。
7. 优雅退出机制，防止 TclError。
"""
import pyads
import tkinter
from tkinter import messagebox
import time
import numpy as np
import os
import threading

# ========== 通道配置 ==========
TOTAL_CHANNELS = 33
VIBRATION_CHANNELS = 30
CURRENT_CHANNELS = 3
VIBRATION_GROUP_SIZE = 10
SAMPLE_COUNT = 100
SAMPLING_FREQUENCY = 10000
CURRENT_FREQUENCY = 1000

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
# 默认路径改为 文件夹/文件名前缀 的格式
DEFAULT_SAVE_PATH = "data_record/test_01"
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

        # 定时器标记
        self.is_realtime_running = False
        self.is_status_polling = False  # 状态轮询标记
        self.is_app_closing = False     # 程序退出标记

        self.set_init_window()

        # 绑定窗口关闭事件
        self.init_windows_name.protocol("WM_DELETE_WINDOW", self.on_closing)

    def set_init_window(self):
        """初始化基础UI界面"""
        self.init_windows_name.title('ADS 数据采集 & 工况识别 & 倍率控制系统')
        self.init_windows_name.geometry('650x780+100+50') # 略微增加高度
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
        
        tkinter.Label(frame_data, text='保存路径(目录/文件名)').grid(row=1, column=0, padx=5, pady=2, sticky="w")
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

        # 4. 机床倍率控制组
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
        # 安全检查：如果程序正在关闭，不再尝试写入UI
        if self.is_app_closing:
            return
            
        global LOG_LINE_NUM
        current_time = self.get_current_time()
        logmsg_in = f"[{current_time}] {logmsg}\n"
        
        try:
            if LOG_LINE_NUM <= 30:
                self.log_text.insert(tkinter.END, logmsg_in)
                LOG_LINE_NUM += 1
            else:
                self.log_text.delete(1.0, 2.0)
                self.log_text.insert(tkinter.END, logmsg_in)
            
            self.log_text.see(tkinter.END)
        except Exception:
            pass

    # ============ [核心] 优雅退出 ============
    def on_closing(self):
        """停止循环并断开连接"""
        self.is_app_closing = True       # 1. 锁死所有UI更新操作
        self.is_realtime_running = False # 2. 停止采集循环
        self.is_status_polling = False   # 3. 停止状态轮询
        
        # 4. 安全断开 PLC
        if self.plc_conn and self.plc_conn.is_open:
            try:
                self.plc_conn.close()
                print("ADS Connection Closed.")
            except:
                pass
        
        # 5. 销毁窗口
        self.init_windows_name.destroy()

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

            # 连接成功后，开启独立的后台状态轮询
            if not self.is_status_polling:
                self.is_status_polling = True
                self.status_polling_loop()

        except Exception as e:
            self.write_log_to_text(f'连接失败: {str(e)}')
            self.plc_conn = None

    def status_polling_loop(self):
        """独立的后台状态轮询"""
        if self.is_app_closing or not self.is_status_polling:
            return

        if self.plc_conn and self.plc_conn.is_open:
            self.update_override_status()
            
        # 每 500ms 刷新一次
        if not self.is_app_closing:
            self.init_windows_name.after(500, self.status_polling_loop)

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
            self.write_log_to_text(f'读取失败: {str(e)}')
            return None, None

    # ============================================================
    # --- [核心接口] 供外部模型/算法调用的 API ---
    # ============================================================
    def api_set_feed_override(self, value: int):
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text("API调用失败: PLC未连接")
            return False
            
        safe_value = max(0, min(150, int(value)))
        try:
            self.plc_conn.write_by_name(VAR_CMD_FEED, safe_value, pyads.PLCTYPE_WORD)
            self.var_set_feed.set(str(safe_value))
            self.write_log_to_text(f"[API] 设定进给 -> {safe_value}%")
            return True
        except Exception as e:
            self.write_log_to_text(f"[API] 进给写入异常: {e}")
            return False

    def api_set_spindle_override(self, value: int):
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text("API调用失败: PLC未连接")
            return False
            
        safe_value = max(50, min(120, int(value)))
        try:
            self.plc_conn.write_by_name(VAR_CMD_SPINDLE, safe_value, pyads.PLCTYPE_WORD)
            self.var_set_spindle.set(str(safe_value))
            self.write_log_to_text(f"[API] 设定主轴 -> {safe_value}%")
            return True
        except Exception as e:
            self.write_log_to_text(f"[API] 主轴写入异常: {e}")
            return False

    def api_set_control_enable(self, enable: bool):
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text("API调用失败: PLC未连接")
            return False
            
        try:
            self.plc_conn.write_by_name(VAR_CMD_ENABLE, enable, pyads.PLCTYPE_BOOL)
            self.is_override_enabled = enable
            if self.is_override_enabled:
                self.btn_enable_override.config(text="ON (HMI 接管)", bg="#00ff00")
                self.write_log_to_text("[API] 权限已切换至 HMI")
            else:
                self.btn_enable_override.config(text="OFF (面板控制)", bg="gray")
                self.write_log_to_text("[API] 权限已释放给机床面板")
            return True
        except Exception as e:
            self.write_log_to_text(f"[API] 权限切换异常: {e}")
            return False

    # ============================================================
    # --- [UI 回调] 界面按钮点击事件 ---
    # ============================================================
    def write_feed_override(self):
        if not self.plc_conn or not self.plc_conn.is_open:
            messagebox.showwarning("警告", "请先连接PLC")
            return
        try:
            val = int(self.var_set_feed.get())
            self.api_set_feed_override(val)
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数")

    def write_spindle_override(self):
        if not self.plc_conn or not self.plc_conn.is_open:
            messagebox.showwarning("警告", "请先连接PLC")
            return
        try:
            val = int(self.var_set_spindle.get())
            self.api_set_spindle_override(val)
        except ValueError:
            messagebox.showerror("错误", "请输入有效的整数")

    def toggle_override_enable(self):
        if not self.plc_conn or not self.plc_conn.is_open:
            messagebox.showwarning("警告", "请先连接PLC")
            return
        target_state = not self.is_override_enabled
        self.api_set_control_enable(target_state)

    def update_override_status(self):
        """同步 PLC 当前的所有状态"""
        if self.is_app_closing or not self.plc_conn or not self.plc_conn.is_open:
            return
        try:
            act_feed = self.plc_conn.read_by_name(VAR_ACT_FEED, pyads.PLCTYPE_WORD)
            self.var_act_feed_display.set(str(act_feed))

            cmd_feed = self.plc_conn.read_by_name(VAR_CMD_FEED, pyads.PLCTYPE_WORD)
            self.var_cmd_feed_display.set(str(cmd_feed))

            cmd_spindle = self.plc_conn.read_by_name(VAR_CMD_SPINDLE, pyads.PLCTYPE_WORD)
            self.var_cmd_spindle_display.set(str(cmd_spindle))
        except Exception:
            pass

    # --- 数据处理与特征 ---
    def process_data(self, raw_data, index_data):
        if raw_data is None or index_data is None:
            return None, None

        try:
            raw_matrix = np.array(raw_data, dtype=np.int16).reshape(FULL_CHANNELS, SAMPLE_COUNT)
            index_array = np.array(index_data, dtype=np.int16)
        except ValueError as e:
            self.write_log_to_text(f"数据重塑错误: {e}")
            return None, None
        
        continuous_data_100ms = np.zeros((TOTAL_CHANNELS, SAMPLE_COUNT), dtype=np.int16)
        
        for i in range(TOTAL_CHANNELS):
            write_ptr = index_array[i]
            channel_raw = raw_matrix[i, :]
            
            part1 = channel_raw[write_ptr - 1:]
            part2 = channel_raw[:write_ptr - 1]
            continuous_data_100ms[i, :] = np.concatenate((part1, part2))

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

        # 动态提取增量数据
        try:
            T_interval_ms = float(self.interval_text.get(1.0, tkinter.END).strip())
        except ValueError:
            T_interval_ms = float(DEFAULT_INTERVAL_MS)
        
        T_valid_ms = min(T_interval_ms, 10.0)
        n_vib = int(T_valid_ms * 10) # 10kHz
        n_curr = int(T_valid_ms * 1) # 1kHz
        
        if n_vib <= 0: n_vib = 1
        if n_curr <= 0: n_curr = 1

        current_channels_inc = current_channels_100ms[:, -n_curr:]
        vib_channels_inc = vib_channels_100ms[:, -n_vib:]
        
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
            'T_interval_ms': T_valid_ms
        }
        
        return processed_data_100ms, incremental_data_dict

    def calculate_current_feature(self, current_data):
        curr_a = current_data['A']
        curr_b = current_data['B']
        curr_c = current_data['C']
        
        rms_sq_a = np.mean(curr_a.astype(np.float64)**2)
        rms_sq_b = np.mean(curr_b.astype(np.float64)**2)
        rms_sq_c = np.mean(curr_c.astype(np.float64)**2)
        
        avg_rms = (np.sqrt(rms_sq_a) + np.sqrt(rms_sq_b) + np.sqrt(rms_sq_c)) / 3
        return avg_rms

    def calculate_vibration_feature(self, vib_data):
        vib_z = vib_data['Z']
        rms_vib_z = np.sqrt(np.mean(vib_z.astype(np.float64)**2))
        return rms_vib_z
    
    def classify_cutting_state(self, processed_data):
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
        
        if prev_state != self.cutting_state:
            self.write_log_to_text(f'>>> ⚠️ 状态切换: {prev_state} -> {self.cutting_state} (Vib:{vib_rms_value:.1f}, Curr:{current_rms_value:.1f})')
            
        return self.cutting_state, current_rms_value, vib_rms_value

    def send_data_to_model(self, processed_data):
        """占位：模型接口"""
        pass
    
    def save_processed_data_to_file(self, incremental_data):
        raw_input_path = self.save_path.get().strip()
        if not raw_input_path:
            return False

        # 1. 路径处理
        dir_name = os.path.dirname(raw_input_path)
        file_base = os.path.basename(raw_input_path)
        
        if dir_name and not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except Exception as e:
                self.write_log_to_text(f"创建文件夹失败: {e}")
                return False

        path_vib = os.path.join(dir_name, f"{file_base}_Vib.csv")
        path_curr = os.path.join(dir_name, f"{file_base}_Curr.csv")

        # 2. 提取数据
        vib_x = incremental_data['Vibration']['X']
        vib_y = incremental_data['Vibration']['Y']
        vib_z = incremental_data['Vibration']['Z']
        
        curr_a = incremental_data['Current']['A']
        curr_b = incremental_data['Current']['B']
        curr_c = incremental_data['Current']['C']
        
        T_interval_ms = incremental_data['T_interval_ms']
        
        # 3. 计算全局时间轴
        chunk_start_time = (self.sample_index - 1) * (T_interval_ms / 1000.0)

        # 4. 保存振动数据 (10kHz)
        num_vib_points = len(vib_x)
        try:
            file_exists = os.path.exists(path_vib) and os.path.getsize(path_vib) > 0
            with open(path_vib, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write("Time_Sec,Cycle_Index,Vib_X,Vib_Y,Vib_Z\n")
                
                for i in range(num_vib_points):
                    t_point = chunk_start_time + i * (1.0 / SAMPLING_FREQUENCY)
                    f.write(f"{t_point:.5f},{self.sample_index},{vib_x[i]},{vib_y[i]},{vib_z[i]}\n")
        except Exception as e:
            self.write_log_to_text(f"振动文件保存失败: {e}")
            return False

        # 5. 保存电流数据 (1kHz)
        num_curr_points = len(curr_a)
        try:
            file_exists = os.path.exists(path_curr) and os.path.getsize(path_curr) > 0
            with open(path_curr, 'a', encoding='utf-8') as f:
                if not file_exists:
                    f.write("Time_Sec,Cycle_Index,Curr_A,Curr_B,Curr_C\n")
                
                for j in range(num_curr_points):
                    t_point = chunk_start_time + j * (1.0 / CURRENT_FREQUENCY)
                    f.write(f"{t_point:.5f},{self.sample_index},{curr_a[j]},{curr_b[j]},{curr_c[j]}\n")
        except Exception as e:
            self.write_log_to_text(f"电流文件保存失败: {e}")
            return False
            
        return True

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
        if self.is_app_closing or not self.is_realtime_running:
            return

        try:
            interval = int(self.interval_text.get(1.0, tkinter.END).strip())

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
                current_state, curr_rms, vib_rms = self.classify_cutting_state(processed_data_100ms)
                
                # 4. 数据保存
                save_msg = ""
                if current_state == 'CUTTING':
                    saved = self.save_processed_data_to_file(incremental_data_dict)
                    self.send_data_to_model(processed_data_100ms)
                    save_msg = " [已保存]" if saved else " [保存失败]"
                
                # 5. 日志输出 (每20个周期刷新一次)
                if self.sample_index % 20 == 0 or (current_state == 'CUTTING' and self.sample_index % 20 == 0):
                     log_content = f'周期 {self.sample_index}: {current_state} | ⚡Curr:{curr_rms:.1f} | 〰️Vib:{vib_rms:.1f}{save_msg}'
                     self.write_log_to_text(log_content)
            
        except ValueError:
            self.write_log_to_text('错误：采集间隔或阈值输入无效。')
        except Exception as e:
            self.write_log_to_text(f'监测循环错误: {str(e)}')
            self.stop_realtime_monitor()
        
        # 递归调用 (如果程序未关闭)
        if self.is_realtime_running and not self.is_app_closing:
            self.init_windows_name.after(interval, self.realtime_monitor_loop)

# 主程序
def Gui_Start():
    init_window = tkinter.Tk()
    app = DataLoggerApp(init_window)
    init_window.mainloop()

if __name__ == "__main__":
    Gui_Start()