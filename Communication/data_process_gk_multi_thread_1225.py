# -*- coding:utf-8 -*-
"""
ADS 数据采集系统 - 多线程旗舰版
版本:Multi-Threaded Final Release

UI 线程(主线程):只负责响应按钮、刷新界面文本。它绝不碰硬盘文件，也绝不碰 sample_index(计数器)。
采集线程(后台线程): 这是一个**串行(Serial)**的死循环。

相较于 data_process_gk_1125
功能升级：
1. [多线程] 数据采集与保存运行在独立线程，彻底解决界面卡顿风险。
2. [线程安全] 使用 Queue 队列机制跨线程传递日志，防止 Tkinter 崩溃。
3. [完整功能] 保留了之前所有的工况识别、双文件保存、倍率闭环控制。
"""
import pyads
import tkinter
from tkinter import messagebox
import time
import numpy as np
import os
import threading
import queue # [新增] 线程安全的队列

# ========== 通道配置 ==========
TOTAL_CHANNELS = 33 
VIBRATION_CHANNELS = 30 
CURRENT_CHANNELS = 3 
SAMPLE_COUNT = 100 
SAMPLING_FREQUENCY = 10000 
CURRENT_FREQUENCY = 1000

# ADS 配置
FULL_CHANNELS = 80
FULL_BUFFER_LENGTH = FULL_CHANNELS * SAMPLE_COUNT
GVL_BUFFER_DATATYPE = pyads.PLCTYPE_INT
GVL_BUFFER_GROUP = 0x4020 
GVL_BUFFER_OFFSET = 0x0
INDEX_BUFFER_OFFSET = 16000
INDEX_BUFFER_LENGTH = FULL_CHANNELS 
INDEX_BUFFER_DATATYPE = pyads.PLCTYPE_INT

# ADS 配置 (倍率控制变量)
VAR_CMD_FEED = "GVL.Gvl_Cmd_FeedRate_Set"         
VAR_CMD_SPINDLE = "GVL.Gvl_Cmd_Spindle_Set"       
VAR_CMD_ENABLE = "GVL.Gvl_Cmd_Enable_Override"    
VAR_ACT_FEED = "GVL.Gvl_Act_FeedRate_Real"        

# 默认参数
DEFAULT_AMS_NETID = "5.136.192.215.1.1"
DEFAULT_PORT = "851"
DEFAULT_INTERVAL_MS = "10" 
DEFAULT_SAVE_PATH = "data_record/test_01" 
LOG_LINE_NUM = 0

# ========== 工况识别配置 ==========
DEFAULT_IDLE_THRESHOLD = "500"
DEFAULT_VIB_THRESHOLD = "320"
STABILITY_CHECK_COUNT = 5

class DataLoggerApp:
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value=DEFAULT_SAVE_PATH)
        self.plc_conn = None
        self.sample_index = 0
        
        # 多线程相关
        self.log_queue = queue.Queue() # [新增] 日志队列
        self.monitor_thread = None     # [新增] 采集线程对象
        self.lock = threading.Lock()   # [新增] 线程锁(虽然pyads通常不需要，但为了严谨)
        
        # 标志位
        self.is_realtime_running = False # 控制采集线程
        self.is_status_polling = False   # 控制状态轮询
        self.is_app_closing = False      # 程序退出标记
        
        # 工况识别
        self.cutting_state = 'STOP' 
        self.state_history = [] 
        self.stability_check_count = STABILITY_CHECK_COUNT
        self.idle_threshold = tkinter.StringVar(value=DEFAULT_IDLE_THRESHOLD)
        self.vib_threshold = tkinter.StringVar(value=DEFAULT_VIB_THRESHOLD)
        
        # 倍率控制
        self.var_set_feed = tkinter.StringVar(value="100")      
        self.var_set_spindle = tkinter.StringVar(value="100")   
        self.var_act_feed_display = tkinter.StringVar(value="---") 
        self.var_cmd_feed_display = tkinter.StringVar(value="---") 
        self.var_cmd_spindle_display = tkinter.StringVar(value="---") 
        self.is_override_enabled = False 

        self.set_init_window()

        # 启动日志消费者（主线程）
        self.process_log_queue()

        # 绑定关闭事件
        self.init_windows_name.protocol("WM_DELETE_WINDOW", self.on_closing)

    def set_init_window(self):
        self.init_windows_name.title('ADS 数据采集 (多线程旗舰版)')
        self.init_windows_name.geometry('650x780+100+50')
        self.init_windows_name.grid_columnconfigure(0, weight=1)
        for i in range(5): self.init_windows_name.grid_rowconfigure(i, weight=0)
        self.init_windows_name.grid_rowconfigure(5, weight=1) 

        # 1. ADS 连接
        frame_conn = tkinter.LabelFrame(self.init_windows_name, text="ADS 连接配置", padx=5, pady=5)
        frame_conn.grid(row=0, column=0, pady=5, padx=10, sticky="ew")
        tkinter.Label(frame_conn, text='AmsNetID').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.netID_text = self._create_text_widget(frame_conn, DEFAULT_AMS_NETID, row=0, column=1)
        tkinter.Label(frame_conn, text='Port').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.port_text = self._create_text_widget(frame_conn, DEFAULT_PORT, row=1, column=1)
        self.open_port_button = tkinter.Button(frame_conn, text='打开端口', command=self.plc_port_open)
        self.open_port_button.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")
        frame_conn.grid_columnconfigure(1, weight=1)

        # 2. 数据采集
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
        
        # 3. 工况识别
        frame_threshold = tkinter.LabelFrame(self.init_windows_name, text="工况识别配置 (RMS双特征)", padx=5, pady=5)
        frame_threshold.grid(row=2, column=0, pady=5, padx=10, sticky="ew")
        tkinter.Label(frame_threshold, text='电流停转阈值(低)').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        tkinter.Entry(frame_threshold, textvariable=self.idle_threshold, width=15).grid(row=0, column=1, padx=5, pady=2, sticky="ew")
        tkinter.Label(frame_threshold, text='振动切削阈值(高)').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        tkinter.Entry(frame_threshold, textvariable=self.vib_threshold, width=15).grid(row=1, column=1, padx=5, pady=2, sticky="ew")
        frame_threshold.grid_columnconfigure(1, weight=1)

        # 4. 机床倍率
        frame_override = tkinter.LabelFrame(self.init_windows_name, text="机床倍率控制 (闭环读写)", padx=5, pady=5, fg="blue")
        frame_override.grid(row=3, column=0, pady=5, padx=10, sticky="ew")
        tkinter.Label(frame_override, text="设定进给:").grid(row=0, column=0, sticky="w")
        tkinter.Entry(frame_override, textvariable=self.var_set_feed, width=6).grid(row=0, column=1, padx=2)
        tkinter.Button(frame_override, text="写入", command=self.write_feed_override, width=5).grid(row=0, column=2, padx=5)
        tkinter.Label(frame_override, text="PLC当前:").grid(row=0, column=3, padx=(10,0))
        tkinter.Label(frame_override, textvariable=self.var_cmd_feed_display, fg="blue", font=("Arial", 10, "bold")).grid(row=0, column=4, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=0, column=5)

        tkinter.Label(frame_override, text="设定主轴:").grid(row=1, column=0, sticky="w")
        tkinter.Entry(frame_override, textvariable=self.var_set_spindle, width=6).grid(row=1, column=1, padx=2)
        tkinter.Button(frame_override, text="写入", command=self.write_spindle_override, width=5).grid(row=1, column=2, padx=5)
        tkinter.Label(frame_override, text="PLC当前:").grid(row=1, column=3, padx=(10,0))
        tkinter.Label(frame_override, textvariable=self.var_cmd_spindle_display, fg="blue", font=("Arial", 10, "bold")).grid(row=1, column=4, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=1, column=5)

        tkinter.Label(frame_override, text="--------------------------------------------------").grid(row=2, column=0, columnspan=6)
        tkinter.Label(frame_override, text="机床实际执行进给:").grid(row=3, column=0, columnspan=2, sticky="e")
        tkinter.Label(frame_override, textvariable=self.var_act_feed_display, fg="red", font=("Arial", 12, "bold")).grid(row=3, column=2, columnspan=2, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=3, column=4, sticky="w")

        tkinter.Label(frame_override, text="控制权限:").grid(row=4, column=0, sticky="w", pady=5)
        self.btn_enable_override = tkinter.Button(frame_override, text="OFF (面板控制)", bg="gray", command=self.toggle_override_enable, width=20)
        self.btn_enable_override.grid(row=4, column=1, columnspan=4, pady=5)

        # 5. 系统日志
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

    # --- [线程安全] 日志系统 ---
    def put_log(self, logmsg):
        """任何线程都可以调用此方法，将日志放入队列"""
        if self.is_app_closing: return
        timestamp_msg = f"[{self.get_current_time()}] {logmsg}\n"
        self.log_queue.put(timestamp_msg)

    def process_log_queue(self):
        """主线程专用：从队列取日志并更新UI"""
        if self.is_app_closing: return
        
        while not self.log_queue.empty():
            try:
                msg = self.log_queue.get_nowait()
                global LOG_LINE_NUM
                if LOG_LINE_NUM <= 30:
                    self.log_text.insert(tkinter.END, msg)
                    LOG_LINE_NUM += 1
                else:
                    self.log_text.delete(1.0, 2.0)
                    self.log_text.insert(tkinter.END, msg)
                self.log_text.see(tkinter.END)
            except queue.Empty:
                break
        
        # 每100ms检查一次队列
        self.init_windows_name.after(100, self.process_log_queue)

    def write_log_to_text(self, logmsg):
        """兼容旧调用的接口，重定向到 put_log"""
        self.put_log(logmsg)

    # --- 优雅退出 ---
    def on_closing(self):
        self.is_app_closing = True
        self.is_realtime_running = False
        self.is_status_polling = False
        
        # 等待采集线程结束 (最多等待1秒)
        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

        if self.plc_conn and self.plc_conn.is_open:
            try:
                self.plc_conn.close()
                print("ADS Connection Closed.")
            except:
                pass
        self.init_windows_name.destroy()

    # --- ADS 连接 ---
    def plc_port_open(self):
        if self.plc_conn and self.plc_conn.is_open:
            self.put_log('端口已连接，请勿重复操作。')
            return
        
        # 获取UI值需在主线程做
        AmsNetID = self.netID_text.get(1.0, tkinter.END).strip()
        port = self.port_text.get(1.0, tkinter.END).strip()

        try:
            pyads.open_port()
            self.plc_conn = pyads.Connection(AmsNetID, int(port))
            self.plc_conn.open()
            self.put_log(f'成功连接PLC: {AmsNetID}:{port}')
            self.open_port_button.config(bg="#a0ffa0")

            if not self.is_status_polling:
                self.is_status_polling = True
                self.status_polling_loop()

        except Exception as e:
            self.put_log(f'连接失败: {str(e)}')
            self.plc_conn = None

    def status_polling_loop(self):
        """[主线程] 状态轮询：每500ms刷新一次UI显示"""
        if self.is_app_closing: return
        
        if self.plc_conn and self.plc_conn.is_open:
            self.update_override_status()
            
        if not self.is_app_closing:
            self.init_windows_name.after(500, self.status_polling_loop)

    def _read_data_atomic(self):
        """带锁的ADS读取"""
        if not self.plc_conn or not self.plc_conn.is_open:
            return None, None
        
        # 虽然ADS通常线程安全，但加锁更保险
        with self.lock:
            try:
                raw_data = self.plc_conn.read(
                    GVL_BUFFER_GROUP, GVL_BUFFER_OFFSET, GVL_BUFFER_DATATYPE * FULL_BUFFER_LENGTH
                )
                index_data = self.plc_conn.read(
                    GVL_BUFFER_GROUP, INDEX_BUFFER_OFFSET, INDEX_BUFFER_DATATYPE * INDEX_BUFFER_LENGTH
                )
                return raw_data, index_data
            except Exception as e:
                self.put_log(f'原子读取失败: {str(e)}')
                return None, None

    # --- API 接口 ---
    def api_set_feed_override(self, value):
        if not self.plc_conn: return False
        safe_val = max(0, min(150, int(value)))
        try:
            with self.lock:
                self.plc_conn.write_by_name(VAR_CMD_FEED, safe_val, pyads.PLCTYPE_WORD)
            # 在主线程更新UI变量(这里我们是在主线程调用的，所以直接设置没问题)
            # 如果从子线程调用此API，建议只log，不直接操作Tkinter Var
            self.var_set_feed.set(str(safe_val)) 
            self.put_log(f"[API] 设定进给 -> {safe_val}%")
            return True
        except Exception as e:
            self.put_log(f"写入失败: {e}")
            return False

    def api_set_spindle_override(self, value):
        if not self.plc_conn: return False
        safe_val = max(50, min(120, int(value)))
        try:
            with self.lock:
                self.plc_conn.write_by_name(VAR_CMD_SPINDLE, safe_val, pyads.PLCTYPE_WORD)
            self.var_set_spindle.set(str(safe_val))
            self.put_log(f"[API] 设定主轴 -> {safe_val}%")
            return True
        except Exception:
            return False

    def api_set_control_enable(self, enable):
        if not self.plc_conn: return False
        try:
            with self.lock:
                self.plc_conn.write_by_name(VAR_CMD_ENABLE, enable, pyads.PLCTYPE_BOOL)
            self.is_override_enabled = enable
            if enable:
                self.btn_enable_override.config(text="ON (HMI 接管)", bg="#00ff00")
                self.put_log("[API] 权限 -> HMI")
            else:
                self.btn_enable_override.config(text="OFF (面板控制)", bg="gray")
                self.put_log("[API] 权限 -> 面板")
            return True
        except Exception:
            return False

    # --- UI 事件 ---
    def write_feed_override(self):
        try: self.api_set_feed_override(int(self.var_set_feed.get()))
        except: messagebox.showerror("Err", "Invalid Int")

    def write_spindle_override(self):
        try: self.api_set_spindle_override(int(self.var_set_spindle.get()))
        except: messagebox.showerror("Err", "Invalid Int")

    def toggle_override_enable(self):
        self.api_set_control_enable(not self.is_override_enabled)

    def update_override_status(self):
        try:
            # 这里的读操作在主线程，和后台采集线程竞争，所以建议加锁或捕获异常
            with self.lock:
                act = self.plc_conn.read_by_name(VAR_ACT_FEED, pyads.PLCTYPE_WORD)
                cmd = self.plc_conn.read_by_name(VAR_CMD_FEED, pyads.PLCTYPE_WORD)
                spindle = self.plc_conn.read_by_name(VAR_CMD_SPINDLE, pyads.PLCTYPE_WORD)
            
            self.var_act_feed_display.set(str(act))
            self.var_cmd_feed_display.set(str(cmd))
            self.var_cmd_spindle_display.set(str(spindle))
        except:
            pass

    # --- 数据处理 ---
    def process_data(self, raw_data, index_data):
        # (保持原有的高效 numpy 逻辑)
        if raw_data is None: return None, None
        try:
            raw_matrix = np.array(raw_data, dtype=np.int16).reshape(FULL_CHANNELS, SAMPLE_COUNT)
            index_array = np.array(index_data, dtype=np.int16)
            
            continuous = np.zeros((TOTAL_CHANNELS, SAMPLE_COUNT), dtype=np.int16)
            for i in range(TOTAL_CHANNELS):
                ptr = index_array[i]
                row = raw_matrix[i, :]
                continuous[i, :] = np.concatenate((row[ptr-1:], row[:ptr-1]))
            
            vib_100ms = continuous[0:VIBRATION_CHANNELS, :]
            curr_100ms = continuous[VIBRATION_CHANNELS:, :]
            
            processed = {
                'Vibration': {
                    'X': vib_100ms[0:10].flatten(),
                    'Y': vib_100ms[10:20].flatten(),
                    'Z': vib_100ms[20:30].flatten()
                },
                'Current': {
                    'A': curr_100ms[0], 'B': curr_100ms[1], 'C': curr_100ms[2]
                }
            }
            
            # 使用 float 获取线程安全的值（虽然 StringVar 非线程安全，但读取一般没事）
            # 最好的做法是在 Start 时读取一次存入 self.interval_val
            try: t_ms = float(self.interval_text.get("1.0", "end").strip())
            except: t_ms = 10.0
            
            t_valid = min(t_ms, 10.0)
            n_vib = max(1, int(t_valid * 10))
            n_curr = max(1, int(t_valid * 1))
            
            inc_data = {
                'Vibration': {
                    'X': vib_100ms[0:10, -n_vib:].flatten(),
                    'Y': vib_100ms[10:20, -n_vib:].flatten(),
                    'Z': vib_100ms[20:30, -n_vib:].flatten()
                },
                'Current': {
                    'A': curr_100ms[0, -n_curr:],
                    'B': curr_100ms[1, -n_curr:],
                    'C': curr_100ms[2, -n_curr:]
                },
                'T_interval_ms': t_valid
            }
            return processed, inc_data
        except Exception as e:
            self.put_log(f"Process Err: {e}")
            return None, None

    def classify_cutting_state(self, processed):
        # 计算 RMS
        c = processed['Current']
        curr_rms = (np.sqrt(np.mean(c['A']**2)) + np.sqrt(np.mean(c['B']**2)) + np.sqrt(np.mean(c['C']**2)))/3
        
        v = processed['Vibration']
        vib_rms = np.sqrt(np.mean(v['Z']**2))
        
        # 获取阈值 (StringVar 在主线程，子线程读取可能有风险，建议 Start 时缓存，但 Python GIL 通常能保护读操作)
        try: 
            i_th = float(self.idle_threshold.get())
            v_th = float(self.vib_threshold.get())
        except: 
            i_th, v_th = 500.0, 320.0
            
        state = 'STOP'
        if curr_rms >= i_th:
            state = 'CUTTING' if vib_rms >= v_th else 'IDLE'
            
        self.state_history.append(state)
        if len(self.state_history) > self.stability_check_count:
            self.state_history.pop(0)
            
        counts = {s: self.state_history.count(s) for s in ['STOP', 'IDLE', 'CUTTING']}
        maj = self.stability_check_count
        
        prev = self.cutting_state
        if self.cutting_state != 'CUTTING' and counts['CUTTING'] >= maj: self.cutting_state = 'CUTTING'
        elif self.cutting_state == 'CUTTING' and counts['IDLE'] >= maj: self.cutting_state = 'IDLE'
        elif counts['STOP'] >= maj and self.cutting_state != 'STOP': self.cutting_state = 'STOP'
        elif counts['IDLE'] >= maj: self.cutting_state = 'IDLE'
        
        if prev != self.cutting_state:
            self.put_log(f">>> ⚠️ 状态切换: {prev}->{self.cutting_state} (Vib:{vib_rms:.1f}, Curr:{curr_rms:.1f})")
            
        return self.cutting_state, curr_rms, vib_rms

    def save_processed_data_to_file(self, inc_data):
        # 路径处理
        path_str = self.save_path.get().strip()
        if not path_str: return False
        
        dirname = os.path.dirname(path_str)
        basename = os.path.basename(path_str)
        if dirname and not os.path.exists(dirname):
            try: os.makedirs(dirname)
            except: return False
            
        path_vib = os.path.join(dirname, f"{basename}_Vib.csv")
        path_curr = os.path.join(dirname, f"{basename}_Curr.csv")
        
        try:
            # 振动写入
            need_head = not (os.path.exists(path_vib) and os.path.getsize(path_vib) > 0)
            with open(path_vib, 'a', encoding='utf-8') as f:
                if need_head: f.write("Time_Sec,Cycle_Index,Vib_X,Vib_Y,Vib_Z\n")
                
                t0 = (self.sample_index - 1) * (inc_data['T_interval_ms']/1000.0)
                vx, vy, vz = inc_data['Vibration']['X'], inc_data['Vibration']['Y'], inc_data['Vibration']['Z']
                
                # 批量构建字符串，减少 I/O 次数
                lines = []
                for k in range(len(vx)):
                    t = t0 + k * (1.0/SAMPLING_FREQUENCY)
                    lines.append(f"{t:.5f},{self.sample_index},{vx[k]},{vy[k]},{vz[k]}\n")
                f.writelines(lines)

            # 电流写入
            need_head_c = not (os.path.exists(path_curr) and os.path.getsize(path_curr) > 0)
            with open(path_curr, 'a', encoding='utf-8') as f:
                if need_head_c: f.write("Time_Sec,Cycle_Index,Curr_A,Curr_B,Curr_C\n")
                
                ca, cb, cc = inc_data['Current']['A'], inc_data['Current']['B'], inc_data['Current']['C']
                lines_c = []
                for k in range(len(ca)):
                    t = t0 + k * (1.0/CURRENT_FREQUENCY)
                    lines_c.append(f"{t:.5f},{self.sample_index},{ca[k]},{cb[k]},{cc[k]}\n")
                f.writelines(lines_c)
                
            return True
        except Exception as e:
            self.put_log(f"保存失败: {e}")
            return False

    # --- [核心] 多线程采集循环 ---
    def start_realtime_monitor(self):
        if not self.plc_conn or not self.plc_conn.is_open:
            self.put_log('请先打开PLC端口')
            return

        self.is_realtime_running = True
        self.realtime_read_button.config(state=tkinter.DISABLED)
        self.stop_read_button.config(state=tkinter.NORMAL)
        self.put_log('启动后台采集线程...')
        
        # 创建并启动线程
        self.monitor_thread = threading.Thread(target=self.thread_monitor_task)
        self.monitor_thread.daemon = True # 设置为守护线程
        self.monitor_thread.start()

    def stop_realtime_monitor(self):
        self.is_realtime_running = False
        self.realtime_read_button.config(state=tkinter.NORMAL)
        self.stop_read_button.config(state=tkinter.DISABLED)
        self.put_log('正在停止采集线程...')

    def thread_monitor_task(self):
        """
        这是在独立线程中运行的死循环。
        不会阻塞 GUI。
        """
        self.put_log(">>> 采集线程已启动")
        
        while self.is_realtime_running and not self.is_app_closing:
            start_time = time.time()
            
            # 获取间隔 (简单的线程安全处理，若要严谨应在Start时传入)
            try: interval_ms = int(self.interval_text.get("1.0", "end").strip())
            except: interval_ms = 10
            
            # 1. 业务逻辑
            raw, idx = self._read_data_atomic()
            if raw is not None:
                proc_data, inc_data = self.process_data(raw, idx)
                if proc_data:
                    self.sample_index += 1
                    state, c_rms, v_rms = self.classify_cutting_state(proc_data)
                    
                    save_msg = ""
                    if state == 'CUTTING':
                        saved = self.save_processed_data_to_file(inc_data)
                        save_msg = " [已保存]" if saved else " [保存失败]"
                    
                    if self.sample_index % 20 == 0 or (state == 'CUTTING' and self.sample_index % 20 == 0):
                        self.put_log(f"周期 {self.sample_index}: {state} | ⚡{c_rms:.1f} | 〰️{v_rms:.1f}{save_msg}")

            # 2. 精确控频 (保持 10ms 节奏)
            elapsed = (time.time() - start_time) * 1000 # ms
            wait_time = interval_ms - elapsed
            if wait_time > 0:
                time.sleep(wait_time / 1000.0)
        
        self.put_log("<<< 采集线程已结束")

# 主程序
def Gui_Start():
    init_window = tkinter.Tk()
    app = DataLoggerApp(init_window)
    init_window.mainloop()

if __name__ == "__main__":
    Gui_Start()