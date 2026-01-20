# -*- coding:utf-8 -*-
"""
ADS 数据采集系统 - 多线程旗舰版（仅保存/单CSV/真实时序/六列输出/自动复位UI）
版本: Multi-Threaded Final Release (Save Only + Single CSV + True Time Order + Auto UI Reset)

数据结构（按你的最终说明实现）：
- TOTAL_CHANNELS = 60
- 前 30 个通道为振动，后 30 个通道为电流
- 每个方向由 10 个“子通道”组成（SUBCH_PER_AXIS=10）
- PLC 扫描周期 1ms；PC 端每 interval_ms 读取一次（默认10ms）
- 在每个 PLC 周期内：10 个子通道各写 1 个点 => 该方向 1ms 内有 10 个连续点
- PC 每次读取完整环形数组（每通道长度 SAMPLE_COUNT=100），每次新增 n_cycles 点（10ms->10点/子通道）
- 使用 index_data 对齐环形缓冲，并提取每次新增的 n_cycles 点，再按真实时间顺序交织成 N=n_cycles*10 点
- 输出到一个 CSV：六列数据（前三列电流ABC，后三列振动XYZ），严格按真实时间顺序
- 不做切削状态识别：每次采集都直接保存数据
- 读取失败：自动停止采集线程，UI 自动复位为“开始可用/停止不可用”，不再刷屏

CSV列：
Time_Sec,Cycle_Index,Curr_A,Curr_B,Curr_C,Vib_X,Vib_Y,Vib_Z
"""

import pyads
import tkinter
from tkinter import messagebox
import time
import numpy as np
import os
import threading
import queue  # 线程安全队列


# ========== 通道配置 ==========
TOTAL_CHANNELS = 60

# 每方向子通道数：10个子通道 = 1ms内10个连续点
SUBCH_PER_AXIS = 10

# PLC扫描周期（毫秒）
PLC_CYCLE_MS = 1.0

# ADS缓冲配置（PLC侧完整数组长度）
SAMPLE_COUNT = 100

# ADS 配置
FULL_CHANNELS = 80
FULL_BUFFER_LENGTH = FULL_CHANNELS * SAMPLE_COUNT
GVL_BUFFER_DATATYPE = pyads.PLCTYPE_INT
GVL_BUFFER_GROUP = 0x4020
GVL_BUFFER_OFFSET = 0x0

# index缓冲：每通道一个指针
INDEX_BUFFER_OFFSET = 16000
INDEX_BUFFER_LENGTH = FULL_CHANNELS
INDEX_BUFFER_DATATYPE = pyads.PLCTYPE_INT

# ADS 配置 (倍率控制变量) —— 保留，不影响采集保存
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


class DataLoggerApp:
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value=DEFAULT_SAVE_PATH)
        self.plc_conn = None
        self.sample_index = 0

        # 多线程相关
        self.log_queue = queue.Queue()
        self.monitor_thread = None
        self.lock = threading.Lock()

        # 标志位
        self.is_realtime_running = False
        self.is_status_polling = False
        self.is_app_closing = False

        # 倍率控制（保留）
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

    # ---------------- UI：采集按钮状态控制 ----------------
    def _set_collect_ui_running(self, running: bool):
        """只在主线程调用：更新开始/停止按钮状态"""
        if running:
            self.realtime_read_button.config(state=tkinter.DISABLED)
            self.stop_read_button.config(state=tkinter.NORMAL)
        else:
            self.realtime_read_button.config(state=tkinter.NORMAL)
            self.stop_read_button.config(state=tkinter.DISABLED)

    def _reset_collect_ui_safe(self):
        """可被 after 调用：主线程复位按钮"""
        try:
            self._set_collect_ui_running(False)
        except:
            pass

    def set_init_window(self):
        self.init_windows_name.title('ADS 数据采集 (多线程旗舰版-仅保存/单CSV/真实时序/自动复位UI)')
        self.init_windows_name.geometry('650x720+100+50')
        self.init_windows_name.grid_columnconfigure(0, weight=1)
        for i in range(5):
            self.init_windows_name.grid_rowconfigure(i, weight=0)
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
        frame_data = tkinter.LabelFrame(self.init_windows_name, text="数据采集与保存控制（单CSV）", padx=5, pady=5)
        frame_data.grid(row=1, column=0, pady=5, padx=10, sticky="ew")
        tkinter.Label(frame_data, text='采集间隔(ms)').grid(row=0, column=0, padx=5, pady=2, sticky="w")
        self.interval_text = self._create_text_widget(frame_data, DEFAULT_INTERVAL_MS, width=15, row=0, column=1)
        tkinter.Label(frame_data, text='保存路径(目录/文件名)').grid(row=1, column=0, padx=5, pady=2, sticky="w")
        self.save_path_entry = tkinter.Entry(frame_data, textvariable=self.save_path, width=25)
        self.save_path_entry.grid(row=1, column=1, padx=5, pady=2, sticky="ew")

        self.realtime_read_button = tkinter.Button(
            frame_data, text='开始实时采集并保存', command=self.start_realtime_monitor, bg="#d0f0c0"
        )
        self.realtime_read_button.grid(row=2, column=0, pady=5, sticky="ew")
        self.stop_read_button = tkinter.Button(
            frame_data, text='停止采集', command=self.stop_realtime_monitor, state=tkinter.DISABLED, bg="#f0d0d0"
        )
        self.stop_read_button.grid(row=2, column=1, pady=5, sticky="ew")
        frame_data.grid_columnconfigure(1, weight=1)

        # 3. 机床倍率（保留）
        frame_override = tkinter.LabelFrame(self.init_windows_name, text="机床倍率控制 (闭环读写)", padx=5, pady=5, fg="blue")
        frame_override.grid(row=2, column=0, pady=5, padx=10, sticky="ew")
        tkinter.Label(frame_override, text="设定进给:").grid(row=0, column=0, sticky="w")
        tkinter.Entry(frame_override, textvariable=self.var_set_feed, width=6).grid(row=0, column=1, padx=2)
        tkinter.Button(frame_override, text="写入", command=self.write_feed_override, width=5).grid(row=0, column=2, padx=5)
        tkinter.Label(frame_override, text="PLC当前:").grid(row=0, column=3, padx=(10, 0))
        tkinter.Label(frame_override, textvariable=self.var_cmd_feed_display, fg="blue", font=("Arial", 10, "bold")).grid(row=0, column=4, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=0, column=5)

        tkinter.Label(frame_override, text="设定主轴:").grid(row=1, column=0, sticky="w")
        tkinter.Entry(frame_override, textvariable=self.var_set_spindle, width=6).grid(row=1, column=1, padx=2)
        tkinter.Button(frame_override, text="写入", command=self.write_spindle_override, width=5).grid(row=1, column=2, padx=5)
        tkinter.Label(frame_override, text="PLC当前:").grid(row=1, column=3, padx=(10, 0))
        tkinter.Label(frame_override, textvariable=self.var_cmd_spindle_display, fg="blue", font=("Arial", 10, "bold")).grid(row=1, column=4, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=1, column=5)

        tkinter.Label(frame_override, text="--------------------------------------------------").grid(row=2, column=0, columnspan=6)
        tkinter.Label(frame_override, text="机床实际执行进给:").grid(row=3, column=0, columnspan=2, sticky="e")
        tkinter.Label(frame_override, textvariable=self.var_act_feed_display, fg="red", font=("Arial", 12, "bold")).grid(row=3, column=2, columnspan=2, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=3, column=4, sticky="w")

        tkinter.Label(frame_override, text="控制权限:").grid(row=4, column=0, sticky="w", pady=5)
        self.btn_enable_override = tkinter.Button(
            frame_override, text="OFF (面板控制)", bg="gray", command=self.toggle_override_enable, width=20
        )
        self.btn_enable_override.grid(row=4, column=1, columnspan=4, pady=5)

        # 4. 系统日志
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

    # ---------------- 日志（线程安全） ----------------
    def put_log(self, logmsg):
        if self.is_app_closing:
            return
        timestamp_msg = f"[{self.get_current_time()}] {logmsg}\n"
        self.log_queue.put(timestamp_msg)

    def process_log_queue(self):
        if self.is_app_closing:
            return

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

        self.init_windows_name.after(100, self.process_log_queue)

    # ---------------- 优雅退出 ----------------
    def on_closing(self):
        self.is_app_closing = True
        self.is_realtime_running = False
        self.is_status_polling = False

        if self.monitor_thread and self.monitor_thread.is_alive():
            self.monitor_thread.join(timeout=1.0)

        if self.plc_conn and self.plc_conn.is_open:
            try:
                self.plc_conn.close()
                print("ADS Connection Closed.")
            except:
                pass
        self.init_windows_name.destroy()

    # ---------------- ADS连接与状态轮询 ----------------
    def plc_port_open(self):
        if self.plc_conn and self.plc_conn.is_open:
            self.put_log('端口已连接，请勿重复操作。')
            return

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
        if self.is_app_closing:
            return

        if self.plc_conn and self.plc_conn.is_open:
            self.update_override_status()

        if not self.is_app_closing:
            self.init_windows_name.after(500, self.status_polling_loop)

    def _read_data_atomic(self):
        """
        带锁的ADS读取：raw + index
        失败不刷屏，只返回(None, None, errstr)，由采集线程统一处理并自动停止。
        """
        if not self.plc_conn or not self.plc_conn.is_open:
            return None, None, "PLC未连接或端口未打开"

        with self.lock:
            try:
                raw_data = self.plc_conn.read(
                    GVL_BUFFER_GROUP, GVL_BUFFER_OFFSET, GVL_BUFFER_DATATYPE * FULL_BUFFER_LENGTH
                )
                index_data = self.plc_conn.read(
                    GVL_BUFFER_GROUP, INDEX_BUFFER_OFFSET, INDEX_BUFFER_DATATYPE * INDEX_BUFFER_LENGTH
                )
                return raw_data, index_data, None
            except Exception as e:
                return None, None, f"{type(e).__name__}: {e}"

    # ---------------- 倍率控制 API（保留） ----------------
    def api_set_feed_override(self, value):
        if not self.plc_conn:
            return False
        safe_val = max(0, min(150, int(value)))
        try:
            with self.lock:
                self.plc_conn.write_by_name(VAR_CMD_FEED, safe_val, pyads.PLCTYPE_WORD)
            self.var_set_feed.set(str(safe_val))
            self.put_log(f"[API] 设定进给 -> {safe_val}%")
            return True
        except Exception as e:
            self.put_log(f"写入失败: {e}")
            return False

    def api_set_spindle_override(self, value):
        if not self.plc_conn:
            return False
        safe_val = max(50, min(120, int(value)))
        try:
            with self.lock:
                self.plc_conn.write_by_name(VAR_CMD_SPINDLE, safe_val, pyads.PLCTYPE_WORD)
            self.var_set_spindle.set(str(safe_val))
            self.put_log(f"[API] 设定主轴 -> {safe_val}%")
            return True
        except Exception as e:
            self.put_log(f"写入失败: {e}")
            return False

    def api_set_control_enable(self, enable):
        if not self.plc_conn:
            return False
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
        except Exception as e:
            self.put_log(f"写入失败: {e}")
            return False

    # ---------------- UI事件 ----------------
    def write_feed_override(self):
        try:
            self.api_set_feed_override(int(self.var_set_feed.get()))
        except:
            messagebox.showerror("Err", "Invalid Int")

    def write_spindle_override(self):
        try:
            self.api_set_spindle_override(int(self.var_set_spindle.get()))
        except:
            messagebox.showerror("Err", "Invalid Int")

    def toggle_override_enable(self):
        self.api_set_control_enable(not self.is_override_enabled)

    def update_override_status(self):
        try:
            with self.lock:
                act = self.plc_conn.read_by_name(VAR_ACT_FEED, pyads.PLCTYPE_WORD)
                cmd = self.plc_conn.read_by_name(VAR_CMD_FEED, pyads.PLCTYPE_WORD)
                spindle = self.plc_conn.read_by_name(VAR_CMD_SPINDLE, pyads.PLCTYPE_WORD)

            self.var_act_feed_display.set(str(act))
            self.var_cmd_feed_display.set(str(cmd))
            self.var_cmd_spindle_display.set(str(spindle))
        except:
            pass

    # ---------------- 数据处理：按真实时间顺序输出6列 ----------------
    @staticmethod
    def _interleave_10xn(mat_10xn):
        """
        mat_10xn shape: (10, n_cycles)
        输出 shape: (n_cycles*10,)
        顺序：每个PLC周期 j，依次输出子通道0..9（真实时间顺序）
        """
        subch, n_cycles = mat_10xn.shape
        n_total = subch * n_cycles
        out = np.empty((n_total,), dtype=np.int16)
        idx = 0
        for j in range(n_cycles):
            out[idx:idx + subch] = mat_10xn[:, j]
            idx += subch
        return out

    def process_data_to_pack(self, raw_data, index_data, interval_ms):
        """
        生成 pack:
          data: (N,6)  N = n_cycles*10 (10ms->100)
          dt:   真实采样间隔 = 1ms/10 = 0.0001s
          t0:   当前块起始时间（按 sample_index 与 interval 推算）
        """
        if raw_data is None or index_data is None:
            return None

        raw_matrix = np.array(raw_data, dtype=np.int16).reshape(FULL_CHANNELS, SAMPLE_COUNT)
        index_array = np.array(index_data, dtype=np.int16)

        # 1) 对齐环形缓冲：对前60通道
        continuous = np.zeros((TOTAL_CHANNELS, SAMPLE_COUNT), dtype=np.int16)
        for i in range(TOTAL_CHANNELS):
            ptr = int(index_array[i])
            if ptr <= 0 or ptr > SAMPLE_COUNT:
                ptr = 1
            row = raw_matrix[i, :]
            continuous[i, :] = np.concatenate((row[ptr - 1:], row[:ptr - 1]))

        # 2) interval_ms -> PLC周期数 n_cycles（10ms->10）
        interval_ms = max(1.0, float(interval_ms))
        n_cycles = int(round(interval_ms / PLC_CYCLE_MS))
        n_cycles = max(1, min(n_cycles, SAMPLE_COUNT))

        # 3) 提取每个子通道新增的 n_cycles 点（对齐后末尾 n_cycles）
        vib_x_10xn = continuous[0:10,  -n_cycles:]
        vib_y_10xn = continuous[10:20, -n_cycles:]
        vib_z_10xn = continuous[20:30, -n_cycles:]

        cur_a_10xn = continuous[30:40, -n_cycles:]
        cur_b_10xn = continuous[40:50, -n_cycles:]
        cur_c_10xn = continuous[50:60, -n_cycles:]

        # 4) 交织为真实时间顺序（长度 N）
        vx = self._interleave_10xn(vib_x_10xn)
        vy = self._interleave_10xn(vib_y_10xn)
        vz = self._interleave_10xn(vib_z_10xn)

        ca = self._interleave_10xn(cur_a_10xn)
        cb = self._interleave_10xn(cur_b_10xn)
        cc = self._interleave_10xn(cur_c_10xn)

        data = np.column_stack([ca, cb, cc, vx, vy, vz]).astype(np.int16)

        # 5) 时间：1ms内10点 => dt=0.001/10=0.0001s
        dt = (PLC_CYCLE_MS / 1000.0) / SUBCH_PER_AXIS

        # 块起始时间（按采集周期累积）
        t0 = (self.sample_index - 1) * (interval_ms / 1000.0)

        return {"t0": t0, "dt": dt, "data": data, "interval_ms": interval_ms}

    # ---------------- 保存：单CSV，六列按真实时间顺序 ----------------
    def save_pack_to_single_csv(self, pack):
        path_str = self.save_path.get().strip()
        if not path_str:
            return False

        dirname = os.path.dirname(path_str)
        basename = os.path.basename(path_str)
        if dirname and not os.path.exists(dirname):
            try:
                os.makedirs(dirname)
            except:
                return False

        path_all = os.path.join(dirname, f"{basename}_All.csv")

        try:
            need_head = not (os.path.exists(path_all) and os.path.getsize(path_all) > 0)
            with open(path_all, "a", encoding="utf-8") as f:
                if need_head:
                    f.write("Time_Sec,Cycle_Index,Curr_A,Curr_B,Curr_C,Vib_X,Vib_Y,Vib_Z\n")

                t0 = pack["t0"]
                dt = pack["dt"]
                data = pack["data"]  # (N,6)

                lines = []
                for k in range(data.shape[0]):
                    t = t0 + k * dt
                    ca, cb, cc, vx, vy, vz = data[k, :]
                    lines.append(
                        f"{t:.6f},{self.sample_index},{int(ca)},{int(cb)},{int(cc)},{int(vx)},{int(vy)},{int(vz)}\n"
                    )
                f.writelines(lines)

            return True
        except Exception as e:
            self.put_log(f"保存失败: {e}")
            return False

    # ---------------- 采集控制（开始/停止） ----------------
    def start_realtime_monitor(self):
        if not self.plc_conn or not self.plc_conn.is_open:
            self.put_log('请先打开PLC端口')
            return

        if self.is_realtime_running:
            self.put_log("采集已在运行中，无需重复启动。")
            return

        self.is_realtime_running = True
        self.init_windows_name.after(0, lambda: self._set_collect_ui_running(True))
        self.put_log('启动后台采集线程...')

        self.monitor_thread = threading.Thread(target=self.thread_monitor_task)
        self.monitor_thread.daemon = True
        self.monitor_thread.start()

    def stop_realtime_monitor(self):
        # 如果线程已经结束/未运行：直接确保UI复位，不再提示“正在停止...”
        if not self.is_realtime_running:
            self.init_windows_name.after(0, self._reset_collect_ui_safe)
            return

        self.is_realtime_running = False
        self.init_windows_name.after(0, self._reset_collect_ui_safe)
        self.put_log('正在停止采集线程...')

    # ---------------- 采集线程 ----------------
    def thread_monitor_task(self):
        self.put_log(">>> 采集线程已启动（仅保存/单CSV/真实时序/六列输出）")

        try:
            while self.is_realtime_running and not self.is_app_closing:
                start_time = time.time()

                # interval_ms
                try:
                    interval_ms = float(self.interval_text.get("1.0", "end").strip())
                except:
                    interval_ms = 10.0
                interval_ms = max(1.0, interval_ms)

                # ADS read
                raw, idx, err = self._read_data_atomic()
                if err is not None or raw is None or idx is None:
                    self.put_log(f"采集已自动停止：{err if err else '读取失败'}")
                    break

                self.sample_index += 1

                pack = self.process_data_to_pack(raw, idx, interval_ms)
                if pack:
                    saved = self.save_pack_to_single_csv(pack)
                    if not saved:
                        self.put_log(f"周期 {self.sample_index}: 保存失败")
                    elif self.sample_index % 20 == 0:
                        self.put_log(f"周期 {self.sample_index}: 已保存 {pack['data'].shape[0]} 点")

                # pacing
                elapsed = (time.time() - start_time) * 1000.0
                wait_time = interval_ms - elapsed
                if wait_time > 0:
                    time.sleep(wait_time / 1000.0)

        except Exception as e:
            self.put_log(f"采集异常，已自动停止：{type(e).__name__}: {e}")

        finally:
            # 线程退出：强制复位状态与UI（避免你截图里那种“线程结束但UI还在采集中”）
            self.is_realtime_running = False
            try:
                self.init_windows_name.after(0, self._reset_collect_ui_safe)
            except:
                pass
            self.put_log("<<< 采集线程已结束")


# 主程序
def Gui_Start():
    init_window = tkinter.Tk()
    app = DataLoggerApp(init_window)
    init_window.mainloop()


if __name__ == "__main__":
    Gui_Start()
