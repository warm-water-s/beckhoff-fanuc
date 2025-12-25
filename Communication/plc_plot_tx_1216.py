# -*- coding:utf-8 -*-
"""
主要优化点:
- 历史缓存点数计算逻辑修正: 确保 MAX_HISTORY_POINTS 对应 PLOT_HISTORY_LENGTH * (每周期实际采集点数)。
- 代码结构优化: 将 MAX_HISTORY_POINTS 等基于配置的参数在 __init__ 中计算。
1. 增加 Cmd (设定值) 的回读显示。
2. 无论倍率是由 HMI 手动写入，还是由外部模块修改，界面均能实时同步显示 PLC 当前的设定值。
3. 增加清楚绘图区案件，清除当前图形中的波形,以便于重新一次的采集
4. 将历史数据绘制的波形图保存成图片
5. 将采集的数据保存到指定文件目录下的指定文件中，振动与电流分开存放，中间没有时间戳
"""

import pyads
import tkinter
from tkinter import messagebox
import time
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# 日志默认条数
LOG_LINE_NUM = 0

# ========== 通道与采样配置 (常量) ==========
TOTAL_CHANNELS = 33  # 总有效通道数 (实际使用)
VIBRATION_CHANNELS = 30  # 振动通道数（前30个）
CURRENT_CHANNELS = 3  # 电流通道数（后3个）
VIBRATION_GROUP_SIZE = 10  # 每10个通道合成一个振动方向 (X, Y, Z)

# 采样率配置 (基于 1kHz 基础采样率和通道组合)
BASE_SAMPLING_FREQUENCY = 1000  # PLC基础采样频率 1000Hz
SAMPLE_COUNT = 100  # PLC缓冲区每通道存储的点数 (100ms 窗口)

SAVE_PATH = "save_file/file_name"

# 等效采样率
VIB_SAMPLING_FREQUENCY = (
    BASE_SAMPLING_FREQUENCY * VIBRATION_GROUP_SIZE
)  # 1000 * 10 = 10000 Hz
CURR_SAMPLING_FREQUENCY = BASE_SAMPLING_FREQUENCY  # 1000 Hz

# ADS 配置 (PLC内存配置)
FULL_CHANNELS = 80  # PLC实际分配的通道数
FULL_BUFFER_LENGTH = FULL_CHANNELS * SAMPLE_COUNT
GVL_BUFFER_DATATYPE = pyads.PLCTYPE_INT
GVL_BUFFER_GROUP = 0x4020
GVL_BUFFER_OFFSET = 0x0
INDEX_BUFFER_OFFSET = 16000
INDEX_BUFFER_LENGTH = FULL_CHANNELS
INDEX_BUFFER_DATATYPE = pyads.PLCTYPE_INT

# ADS 配置 (倍率控制变量名 - 对应 PLC GVL)
VAR_CMD_FEED = "GVL.Gvl_Cmd_FeedRate_Set"  # WORD
VAR_CMD_SPINDLE = "GVL.Gvl_Cmd_Spindle_Set"  # WORD
VAR_CMD_ENABLE = "GVL.Gvl_Cmd_Enable_Override"  # BOOL
VAR_ACT_FEED = "GVL.Gvl_Act_FeedRate_Real"  # WORD

# 默认连接参数
DEFAULT_AMS_NETID = "5.136.192.215.1.1"
DEFAULT_PORT = "851"
DEFAULT_INTERVAL_MS = "10"  # 采集周期/增量时间 T_interval

# 绘图配置 (历史点数, 对应 1s 窗口, 历史缓存长度)
# 注意: PLOT_HISTORY_LENGTH 决定了历史缓存的总长度（非显示长度）
VIB_PLOT_POINTS = (
    VIB_SAMPLING_FREQUENCY * 1
)  # 10000 Hz * 1s = 10000 点 (振动波形显示窗口宽度 1s)
CURR_PLOT_POINTS = (
    CURR_SAMPLING_FREQUENCY * 1
)  # 1000 Hz * 1s = 1000 点 (电流趋势图显示窗口宽度 1s)
PLOT_HISTORY_LENGTH = 110  # 历史缓存周期数 (110 * 100ms = 11s)
PLOT_Y_MARGIN = 0.15  # 15% 绘图纵坐标裕量


class GUI:
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value=SAVE_PATH)
        self.plc_conn = None
        self.is_realtime_running = False
        self.sample_index = 0  # 用于电流趋势图的 x 轴点数计数 (累计)

        # 倍率控制相关变量
        self.var_set_feed = tkinter.StringVar(value="100")
        self.var_set_spindle = tkinter.StringVar(value="100")

        # --- 显示变量 (从PLC回读) ---
        self.var_act_feed_display = tkinter.StringVar(value="---")  # 实际进给(反馈)
        self.var_cmd_feed_display = tkinter.StringVar(value="---")  # 当前设定进给(回读)
        self.var_cmd_spindle_display = tkinter.StringVar(
            value="---"
        )  # 当前设定主轴(回读)

        self.is_override_enabled = False  # 本地标记使能状态

        # 振动历史数据缓存: 每周期 (100ms) 采集 10 个 1kHz 通道 = 1000 点
        self.MAX_VIB_HISTORY_POINTS = (
            PLOT_HISTORY_LENGTH * VIBRATION_GROUP_SIZE * SAMPLE_COUNT
        )
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
        self.init_windows_name.title("ADS 通讯 - 振动(10kHz)电流(1kHz)监测系统")
        self.init_windows_name.geometry("1400x800+30+30")
        self.init_windows_name.attributes("-alpha", 0.95)

        # ***布局优化：调整 grid 权重***
        self.init_windows_name.grid_columnconfigure(2, weight=1)
        for i in range(5):
            self.init_windows_name.grid_rowconfigure(i, weight=1)

        # 1. ========== 左侧：主容器 Frame ==========
        self.left_frame = tkinter.Frame(self.init_windows_name)
        self.left_frame.grid(
            row=0, column=0, columnspan=2, rowspan=5, padx=10, pady=5, sticky="nsew"
        )
        self.left_frame.grid_rowconfigure(5, weight=1)

        # 2. ========== 左侧：操作控制栏 (简化结构) ==========
        # 2.1. ADS 连接配置组 (Row 0)
        frame_conn = tkinter.LabelFrame(
            self.left_frame, text="ADS 连接配置", padx=5, pady=5
        )
        frame_conn.grid(row=0, column=0, columnspan=2, pady=5, sticky="ew")

        # 使用 Entry 代替 Text for single line input
        tkinter.Label(frame_conn, text="AmsNetID").grid(
            row=0, column=0, padx=5, pady=2, sticky="w"
        )
        self.netID_var = tkinter.StringVar(value=DEFAULT_AMS_NETID)
        tkinter.Entry(frame_conn, textvariable=self.netID_var, width=25).grid(
            row=0, column=1, padx=5, pady=2, sticky="ew"
        )

        tkinter.Label(frame_conn, text="Port").grid(
            row=1, column=0, padx=5, pady=2, sticky="w"
        )
        self.port_var = tkinter.StringVar(value=DEFAULT_PORT)
        tkinter.Entry(frame_conn, textvariable=self.port_var, width=25).grid(
            row=1, column=1, padx=5, pady=2, sticky="ew"
        )

        self.open_port_button = tkinter.Button(
            frame_conn, text="打开端口", command=self.plc_port_open
        )
        self.open_port_button.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # 2.2. 数据采集控制组 (Row 1)
        frame_data = tkinter.LabelFrame(
            self.left_frame, text="数据采集控制", padx=5, pady=5
        )
        frame_data.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        tkinter.Label(frame_data, text="读取间隔(ms)").grid(
            row=0, column=0, padx=5, pady=2, sticky="w"
        )
        self.interval_var = tkinter.StringVar(value=DEFAULT_INTERVAL_MS)
        tkinter.Entry(frame_data, textvariable=self.interval_var, width=25).grid(
            row=0, column=1, padx=5, pady=2, sticky="ew"
        )

        self.read_data_button = tkinter.Button(
            frame_data, text="读取数据 (单次)", command=self.read_data_once
        )
        self.read_data_button.grid(row=1, column=0, columnspan=2, pady=5, sticky="ew")

        self.realtime_read_button = tkinter.Button(
            frame_data, text="开始实时监测", command=self.start_realtime_monitor
        )
        self.realtime_read_button.grid(row=2, column=0, pady=5, sticky="ew")

        self.stop_read_button = tkinter.Button(
            frame_data,
            text="停止实时监测",
            command=self.stop_realtime_monitor,
            state=tkinter.DISABLED,
        )
        self.stop_read_button.grid(row=2, column=1, pady=5, sticky="ew")

        # 2.3. *** 新增：机床倍率控制组 (Row 2) ***
        frame_override = tkinter.LabelFrame(
            self.left_frame, text="机床倍率控制", padx=5, pady=5, fg="blue"
        )
        frame_override.grid(row=2, column=0, columnspan=2, pady=5, sticky="ew")

        # --- 进给倍率 ---
        tkinter.Label(frame_override, text="设定进给:").grid(
            row=0, column=0, sticky="w", pady=2
        )
        tkinter.Entry(frame_override, textvariable=self.var_set_feed, width=6).grid(
            row=0, column=1, pady=2
        )
        tkinter.Button(
            frame_override, text="写入", command=self.write_feed_override, width=4
        ).grid(row=0, column=2, padx=2)
        # 回读显示
        tkinter.Label(frame_override, text="PLC当前:").grid(
            row=0, column=3, padx=(10, 0)
        )
        tkinter.Label(
            frame_override,
            textvariable=self.var_cmd_feed_display,
            fg="blue",
            font=("Arial", 10, "bold"),
        ).grid(row=0, column=4, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=0, column=5)

        # --- 主轴倍率 ---
        tkinter.Label(frame_override, text="设定主轴:").grid(
            row=1, column=0, sticky="w", pady=2
        )
        tkinter.Entry(frame_override, textvariable=self.var_set_spindle, width=6).grid(
            row=1, column=1, pady=2
        )
        tkinter.Button(
            frame_override, text="写入", command=self.write_spindle_override, width=4
        ).grid(row=1, column=2, padx=2)
        # 回读显示
        tkinter.Label(frame_override, text="PLC当前:").grid(
            row=1, column=3, padx=(10, 0)
        )
        tkinter.Label(
            frame_override,
            textvariable=self.var_cmd_spindle_display,
            fg="blue",
            font=("Arial", 10, "bold"),
        ).grid(row=1, column=4, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=1, column=5)

        # --- 实际反馈 (只读) ---
        tkinter.Label(frame_override, text="--------------------------------").grid(
            row=2, column=0, columnspan=6
        )
        tkinter.Label(frame_override, text="机床实际执行进给:").grid(
            row=3, column=0, columnspan=3, sticky="w"
        )
        self.lbl_act_feed = tkinter.Label(
            frame_override,
            textvariable=self.var_act_feed_display,
            fg="red",
            font=("Arial", 12, "bold"),
        )
        self.lbl_act_feed.grid(row=3, column=3, columnspan=2, sticky="w")
        tkinter.Label(frame_override, text="%").grid(row=3, column=5, sticky="w")

        # --- 使能控制 ---
        tkinter.Label(frame_override, text="控制权限:").grid(
            row=4, column=0, sticky="w", pady=5
        )
        self.btn_enable_override = tkinter.Button(
            frame_override,
            text="OFF (面板控制)",
            bg="gray",
            command=self.toggle_override_enable,
            width=15,
        )
        self.btn_enable_override.grid(row=4, column=1, columnspan=4, pady=5)

        # 2.4. 文件/维护操作组 (Row 3)
        frame_file = tkinter.LabelFrame(
            self.left_frame, text="文件/维护", padx=5, pady=5
        )
        frame_file.grid(row=3, column=0, columnspan=2, pady=5, sticky="ew")

        # Row 0: 保存路径
        tkinter.Label(frame_file, text="保存路径").grid(
            row=0, column=0, padx=5, pady=2, sticky="w"
        )
        self.save_path_entry = tkinter.Entry(
            frame_file, textvariable=self.save_path, width=25
        )
        self.save_path_entry.grid(row=0, column=1, padx=5, pady=2, sticky="ew")

        # Row 1: 清空日志 | 重置参数
        self.delete_log_button = tkinter.Button(
            frame_file, text="清空日志", command=self.delete_log
        )
        self.delete_log_button.grid(row=1, column=0, pady=2, sticky="ew", padx=2)

        self.delete_all_button = tkinter.Button(
            frame_file, text="重置连接参数", command=self.reset_parameters
        )
        self.delete_all_button.grid(row=1, column=1, pady=2, sticky="ew", padx=2)

        # === 【修改】Row 2: 保存截图按钮 (不再占用两列，只占左侧) ===
        self.btn_save_images = tkinter.Button(
            frame_file,
            text="保存截图",
            command=self.save_figures_to_image,
            bg="#ADD8E6", # 淡蓝色
        )
        self.btn_save_images.grid(row=2, column=0, pady=5, sticky="ew", padx=2)

        # === 【新增】Row 2: 清除绘图按钮 (占用右侧) ===
        # 这个按钮调用你现有的 reset_and_clear_plots 方法
        self.btn_clear_plots = tkinter.Button(
            frame_file,
            text="清除绘图区",
            command=self.reset_and_clear_plots,
            bg="#FFB6C1", # 淡粉色 (提示该操作具有破坏性)
        )
        self.btn_clear_plots.grid(row=2, column=1, pady=5, sticky="ew", padx=2)

        # 2.5. ***系统日志区 (Row 4, 5)***
        tkinter.Label(self.left_frame, text="系统日志").grid(
            row=3, column=0, columnspan=2, pady=(5, 0), sticky="sw"
        )
        self.log_text = tkinter.Text(self.left_frame, width=35, height=10)
        self.log_text.grid(row=5, column=0, columnspan=2, pady=5, sticky="nsew")

        # 3. ========== 右侧：绘图展示主区 ==========

        # 振动趋势图
        self.fig_vib, self.ax_vib = plt.subplots(figsize=(10, 4), dpi=100)
        self.canvas_vib = FigureCanvasTkAgg(self.fig_vib, master=self.init_windows_name)
        self.canvas_vib.get_tk_widget().grid(
            row=0, column=2, rowspan=2, padx=10, pady=5, sticky="nsew"
        )

        # 电流趋势图
        self.fig_current, self.ax_current = plt.subplots(figsize=(10, 4), dpi=100)
        self.canvas_current = FigureCanvasTkAgg(
            self.fig_current, master=self.init_windows_name
        )
        self.canvas_current.get_tk_widget().grid(
            row=2, column=2, rowspan=2, padx=10, pady=5, sticky="nsew"
        )

        self.init_plots()

    # --- UI & Log Functions ---
    def get_current_time(self):
        return time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(time.time()))

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
            self.write_log_to_text("连接参数已重置为默认值")
        except Exception as e:
            self.write_log_to_text(f"重置参数错误: {e}")

    def delete_log(self):
        """清空日志"""
        global LOG_LINE_NUM
        self.log_text.delete(1.0, tkinter.END)
        LOG_LINE_NUM = 0
        self.write_log_to_text("日志已清空")

    # --- PLC Connection & Read ---
    def plc_port_open(self):
        """打开ADS端口并连接到PLC"""
        AmsNetID = self.netID_var.get().strip()
        try:
            port = int(self.port_var.get().strip())
        except ValueError:
            self.write_log_to_text("端口号无效，请检查输入。")
            return

        if self.plc_conn and self.plc_conn.is_open:
            self.write_log_to_text("端口已连接，请勿重复操作。")
            return

        try:
            pyads.open_port()
            self.plc_conn = pyads.Connection(AmsNetID, port)
            self.plc_conn.open()
            self.write_log_to_text(f"成功连接PLC: {AmsNetID}:{port}")
        except Exception as e:
            self.write_log_to_text(f"连接失败: {str(e)}")
            self.plc_conn = None

    def _read_data_atomic(self):
        """原子读取数据和索引"""
        if not self.plc_conn or not self.plc_conn.is_open:
            return None, None

        try:
            raw_data = self.plc_conn.read(
                GVL_BUFFER_GROUP,
                GVL_BUFFER_OFFSET,
                GVL_BUFFER_DATATYPE * FULL_BUFFER_LENGTH,
            )
            index_data = self.plc_conn.read(
                GVL_BUFFER_GROUP,
                INDEX_BUFFER_OFFSET,
                INDEX_BUFFER_DATATYPE * INDEX_BUFFER_LENGTH,
            )
            return raw_data, index_data
        except Exception as e:
            self.write_log_to_text(f"原子读取失败: {str(e)}")
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
        :param value: 整数, 范围 0-150
        :return: Boolean
        """
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text("API调用失败: PLC未连接")
            return False

        safe_value = max(0, min(150, int(value)))

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

    # --- Override Control Functions---
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
            # 1. 读取实际反馈 (Actual)
            act_feed = self.plc_conn.read_by_name(VAR_ACT_FEED, pyads.PLCTYPE_WORD)
            self.var_act_feed_display.set(str(act_feed))

            # 2. 读取当前设定 (Command) - 这实现了"实时显示外部传入的倍率"
            cmd_feed = self.plc_conn.read_by_name(VAR_CMD_FEED, pyads.PLCTYPE_WORD)
            self.var_cmd_feed_display.set(str(cmd_feed))

            cmd_spindle = self.plc_conn.read_by_name(
                VAR_CMD_SPINDLE, pyads.PLCTYPE_WORD
            )
            self.var_cmd_spindle_display.set(str(cmd_spindle))

            # 3. 同步使能状态: 如果你也希望外部能修改 Enable 位，这里也可以增加读取 Logic
            curr_enable = self.plc_conn.read_by_name(VAR_CMD_ENABLE, pyads.PLCTYPE_BOOL)

        except Exception:
            pass

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
        N_inc_vib_points = min(
            N_inc_vib_points, VIBRATION_GROUP_SIZE * SAMPLE_COUNT
        )  # 1000点
        N_inc_curr_points = min(N_inc_curr_points, SAMPLE_COUNT)  # 100点

        # --- 环形缓冲区重组 ---
        try:
            raw_matrix = np.array(raw_data, dtype=np.int16).reshape(
                FULL_CHANNELS, SAMPLE_COUNT
            )
            index_array = np.array(index_data, dtype=np.int16)
        except ValueError as e:
            self.write_log_to_text(f"数据重塑或索引转换错误: {e}")
            return None, None

        continuous_data_100ms = np.zeros((TOTAL_CHANNELS, SAMPLE_COUNT), dtype=np.int16)

        for i in range(TOTAL_CHANNELS):
            write_ptr = index_array[i]
            channel_raw = raw_matrix[i, :]
            # 确保指针在合法范围内
            if write_ptr >= SAMPLE_COUNT:
                write_ptr = SAMPLE_COUNT - 1
            if write_ptr < 0:
                write_ptr = 0

            # 环形缓冲区重组: [P, P+1, ..., 99] + [0, 1, ..., P-1] (确保时序连续)
            continuous_data_100ms[i, :] = np.concatenate(
                (channel_raw[write_ptr:], channel_raw[:write_ptr])
            )

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
            self.vib_x_history = self.vib_x_history[-self.MAX_VIB_HISTORY_POINTS :]
            self.vib_y_history = self.vib_y_history[-self.MAX_VIB_HISTORY_POINTS :]
            self.vib_z_history = self.vib_z_history[-self.MAX_VIB_HISTORY_POINTS :]

        # 4. ========== 提取和缓存电流增量 ==========
        current_channels_100ms = continuous_data_100ms[VIBRATION_CHANNELS:, :]
        current_channels_inc = current_channels_100ms[:, -N_inc_curr_points:]

        # 更新 X 轴数据 (每次增加 N_inc_curr_points 个点)
        new_x_data = np.arange(
            self.sample_index + 1, self.sample_index + N_inc_curr_points + 1
        )
        self.current_x_history.extend(new_x_data.tolist())
        self.sample_index = self.sample_index + N_inc_curr_points  # 累计采样点数更新

        # 更新 Y 轴数据
        for i in range(CURRENT_CHANNELS):
            channel_data = current_channels_inc[i, :]
            self.current_y_history[i].extend(channel_data.tolist())

        # 限制电流历史数据长度
        if len(self.current_x_history) > self.MAX_CURR_HISTORY_POINTS:
            self.current_x_history = self.current_x_history[
                -self.MAX_CURR_HISTORY_POINTS :
            ]
            for i in range(CURRENT_CHANNELS):
                self.current_y_history[i] = self.current_y_history[i][
                    -self.MAX_CURR_HISTORY_POINTS :
                ]

        # 5. ========== 返回增量数据字典 (用于保存文件) ==========
        incremental_data = {
            "Vibration": {
                "X": vib_x_inc,
                "Y": vib_y_inc,
                "Z": vib_z_inc,
            },
            "Current": {
                "A": current_channels_inc[0, :],
                "B": current_channels_inc[1, :],
                "C": current_channels_inc[2, :],
            },
            "T_interval_ms": T_interval_ms,
            "N_inc_vib_points": len(vib_x_inc),
            "N_inc_curr_points": N_inc_curr_points,
        }

        # process_data 返回处理结果和增量数据, 这里的 None 保持原意, 增量数据用于文件保存
        return None, incremental_data

    # --- Data Save ---
    def _init_recording_files(self):
        """
        在开始监测前，根据用户指定的路径生成两个文件，并写入表头。
        例如用户输入路径: D:/Data/test.txt
        生成: D:/Data/test_Vib.csv 和 D:/Data/test_Curr.csv
        """
        user_path = self.save_path.get().strip()
        if not user_path:
            user_path = "data_record"  # 默认名

        # 分离路径、文件名和扩展名
        dir_name = os.path.dirname(user_path)
        base_name = os.path.basename(user_path)
        file_name_no_ext, _ = os.path.splitext(base_name)

        # 如果目录为空，默认为当前目录
        if not dir_name:
            dir_name = os.getcwd()

        # 确保目录存在
        if not os.path.exists(dir_name):
            try:
                os.makedirs(dir_name)
            except Exception as e:
                self.write_log_to_text(f"创建目录失败: {e}")
                return False

        # 构造两个文件的完整路径
        self.vib_filepath = os.path.join(dir_name, f"{file_name_no_ext}_Vib.csv")
        self.curr_filepath = os.path.join(dir_name, f"{file_name_no_ext}_Curr.csv")

        try:
            # 1. 初始化振动文件 (覆盖写模式 'w')
            with open(self.vib_filepath, "w", encoding="utf-8", newline="") as f:
                # 写入表头：时间, X, Y, Z
                f.write("Time(s),Vib_X,Vib_Y,Vib_Z\n")

            # 2. 初始化电流文件 (覆盖写模式 'w')
            with open(self.curr_filepath, "w", encoding="utf-8", newline="") as f:
                # 写入表头：时间, A, B, C
                f.write("Time(s),Curr_A,Curr_B,Curr_C\n")

            self.write_log_to_text(
                f"文件初始化成功:\n  振动: {self.vib_filepath}\n  电流: {self.curr_filepath}"
            )
            return True

        except Exception as e:
            self.write_log_to_text(f"文件初始化失败 (请检查路径或权限): {e}")
            return False

    def save_data_to_file(self, incremental_data):
        """
        将增量数据追加保存到各自的 CSV 文件中，不包含多余的时间戳标记。
        数据按时间顺序纯净排列。
        """
        # 数据提取
        vib_x = incremental_data["Vibration"]["X"]
        vib_y = incremental_data["Vibration"]["Y"]
        vib_z = incremental_data["Vibration"]["Z"]

        curr_a = incremental_data["Current"]["A"]
        curr_b = incremental_data["Current"]["B"]
        curr_c = incremental_data["Current"]["C"]

        num_vib = len(vib_x)
        num_curr = len(curr_a)

        try:
            # --- 1. 保存振动数据 (Append 模式) ---
            if num_vib > 0 and hasattr(self, "vib_filepath"):
                with open(self.vib_filepath, "a", encoding="utf-8") as f:
                    # 计算这批数据的连续时间轴
                    # start_index 是之前保存的总点数
                    start_idx = self.total_vib_saved

                    # 构建写入字符串 buffer，减少 I/O 次数提高性能
                    lines = []
                    for i in range(num_vib):
                        # 绝对时间 = (历史总点数 + 当前索引) / 采样率
                        t = (start_idx + i) / VIB_SAMPLING_FREQUENCY
                        lines.append(f"{t:.5f},{vib_x[i]},{vib_y[i]},{vib_z[i]}\n")

                    f.writelines(lines)

                # 更新全局计数器
                self.total_vib_saved += num_vib

            # --- 2. 保存电流数据 (Append 模式) ---
            if num_curr > 0 and hasattr(self, "curr_filepath"):
                with open(self.curr_filepath, "a", encoding="utf-8") as f:
                    start_idx = self.total_curr_saved

                    lines = []
                    for i in range(num_curr):
                        # 绝对时间 = (历史总点数 + 当前索引) / 采样率
                        t = (start_idx + i) / CURR_SAMPLING_FREQUENCY
                        lines.append(f"{t:.4f},{curr_a[i]},{curr_b[i]},{curr_c[i]}\n")

                    f.writelines(lines)

                # 更新全局计数器
                self.total_curr_saved += num_curr

        except Exception as e:
            self.write_log_to_text(f"文件追加写入失败: {str(e)}")

    # --- 【新增2】辅助功能：生成文件名时间戳、保存图片、重置绘图 ---
    def get_timestamp_for_filename(self):
        """生成用于文件名的纯数字时间戳 (如: 20231211_103000)"""
        return time.strftime("%Y%m%d_%H%M%S", time.localtime(time.time()))

    def save_figures_to_image(self):
        """将当前两个绘图区域保存为PNG图片"""
        # === 空数据检查 ===
        if not self.vib_x_history:
            messagebox.showwarning("提示", "当前没有波形数据，无法保存图片！")
            return

        save_dir = "saved_images"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)  # 如果文件夹不存在则创建

        timestamp = self.get_timestamp_for_filename()

        # 保存振动图
        vib_filename = os.path.join(save_dir, f"Vib_Waveform_{timestamp}.png")
        try:
            self.fig_vib.savefig(vib_filename)
        except Exception as e:
            self.write_log_to_text(f"振动图保存失败: {e}")

        # 保存电流图
        curr_filename = os.path.join(save_dir, f"Curr_Trend_{timestamp}.png")
        try:
            self.fig_current.savefig(curr_filename)
        except Exception as e:
            self.write_log_to_text(f"电流图保存失败: {e}")

        self.write_log_to_text(f"监测结束，波形图已自动保存至 '{save_dir}' 文件夹")

    def reset_and_clear_plots(self):
        """重置数据缓存、清空绘图区、重置保存计数器"""
        # 1. 清空绘图缓存
        self.vib_x_history = []
        self.vib_y_history = []
        self.vib_z_history = []
        self.current_x_history = []
        self.current_y_history = [[] for _ in range(CURRENT_CHANNELS)]
        self.sample_index = 0

        # 2. 【新增】重置文件保存的全局计数器 (用于计算连续时间轴)
        self.total_vib_saved = 0
        self.total_curr_saved = 0

        # 3. 重新初始化绘图区
        self.init_plots()
        self.canvas_vib.draw()
        self.canvas_current.draw()

        self.write_log_to_text("历史数据已清除，计数器已重置...")

    # --- Plotting Functions ---
    def init_plots(self):
        """初始化绘图样式"""
        # 振动趋势图初始化
        self.ax_vib.clear()
        self.ax_vib.set_title(
            f"三方向振动波形 (滑动窗口 {VIB_PLOT_POINTS}点/1s) [{VIB_SAMPLING_FREQUENCY}Hz]",
            fontsize=12,
        )
        self.ax_vib.set_xlabel("时间 (秒)", fontsize=10)
        self.ax_vib.set_ylabel("振幅 (INT)", fontsize=10)
        self.ax_vib.grid(True, alpha=0.3)
        self.fig_vib.tight_layout()

        # 电流趋势图初始化
        self.ax_current.clear()

        try:
            T_interval_ms = float(self.interval_var.get().strip())
        except ValueError:
            T_interval_ms = float(DEFAULT_INTERVAL_MS)

        N_inc_points = int(round(T_interval_ms * (CURR_SAMPLING_FREQUENCY / 1000)))
        self.ax_current.set_title(
            f"三相电流实时趋势 (滑动窗口 {CURR_PLOT_POINTS}点/1s) [{CURR_SAMPLING_FREQUENCY}Hz]",
            fontsize=12,
        )
        self.ax_current.set_xlabel(f"采样序号 (每次更新{N_inc_points}点)", fontsize=10)
        self.ax_current.set_ylabel("电流值 (INT)", fontsize=10)
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
            min_val = np.min(all_vib_data)
            max_val = np.max(all_vib_data)
            range_val = max_val - min_val
            margin = range_val * PLOT_Y_MARGIN if range_val > 0 else 10
            y_min = min_val - margin
            y_max = max_val + margin
            if y_min == y_max:
                y_min -= 50
                y_max += 50
        else:
            y_min, y_max = -100, 100

        self.ax_vib.clear()

        # 绘制 X/Y/Z 波形
        self.ax_vib.plot(time_slice, x_slice, color="red", label="X方向", linewidth=1)
        self.ax_vib.plot(time_slice, y_slice, color="green", label="Y方向", linewidth=1)
        self.ax_vib.plot(time_slice, z_slice, color="blue", label="Z方向", linewidth=1)

        self.ax_vib.set_title(
            f"三方向振动波形 (滑动窗口 {plot_length}点) [{VIB_SAMPLING_FREQUENCY}Hz]",
            fontsize=12,
        )
        self.ax_vib.set_xlabel("时间 (秒)", fontsize=10)
        self.ax_vib.set_ylabel("振幅 (INT)", fontsize=10)
        self.ax_vib.grid(True, alpha=0.3)
        self.ax_vib.legend(loc="upper right", fontsize=8)

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
        colors = ["red", "green", "blue"]
        labels = ["A相电流", "B相电流", "C相电流"]

        self.ax_current.clear()

        for i in range(CURRENT_CHANNELS):
            y_data = self.current_y_history[i][-plot_length:]
            all_current_data.extend(y_data)
            self.ax_current.plot(
                x_data, y_data, color=colors[i], label=labels[i], linewidth=1
            )

        # ... (动态 Y 轴调整逻辑不变)
        if len(all_current_data) > 0:
            min_val = np.min(all_current_data)
            max_val = np.max(all_current_data)
            range_val = max_val - min_val
            margin = range_val * PLOT_Y_MARGIN if range_val > 0 else 1
            y_min = min_val - margin
            y_max = max_val + margin
            if y_min == y_max:
                y_min -= 1
                y_max += 1
        else:
            y_min, y_max = 0, 1000

        self.ax_current.set_title(
            f"三相电流实时趋势 (滑动窗口 {plot_length}点) [{CURR_SAMPLING_FREQUENCY}Hz]",
            fontsize=12,
        )
        self.ax_current.set_xlabel(f"采样序号 (每次更新{N_inc_points}点)", fontsize=10)
        self.ax_current.set_ylabel("电流值 (INT)", fontsize=10)
        self.ax_current.grid(True, alpha=0.3)
        self.ax_current.legend(loc="upper right", fontsize=10)

        self.ax_current.set_ylim(y_min, y_max)
        self.fig_current.tight_layout()
        self.canvas_current.draw()

    # --- Realtime Control ---
    def read_data_once(self):
        """单次读取数据并更新图表和文件"""
        try:
            self.write_log_to_text("开始读取振动电流数据...")

            raw_data, index_data = self._read_data_atomic()
            if raw_data is None:
                return

            # 使用增量处理 (同时更新内部历史缓存)
            _, incremental_data = self.process_data(raw_data, index_data)

            if incremental_data is None:
                return

            self.update_vibration_plot()
            self.update_current_plot()

            self.save_data_to_file(incremental_data)

            self.write_log_to_text(
                f'成功读取数据点 (振动增量: {incremental_data["N_inc_vib_points"]}点, 电流增量: {incremental_data["N_inc_curr_points"]}点)'
            )
            self.write_log_to_text("数据已更新到图表并保存")

        except Exception as e:
            self.write_log_to_text(f"单次读取失败: {str(e)}")

    def start_realtime_monitor(self):
        """启动实时采集循环"""
        if not self.plc_conn or not self.plc_conn.is_open:
            self.write_log_to_text("请先打开PLC端口")
            return

        # 1. 重置绘图和计数器
        self.reset_and_clear_plots()

        # 2. 【新增】初始化保存文件 (创建文件头)
        if not self._init_recording_files():
            self.write_log_to_text("无法创建保存文件，采集取消。")
            return

        self.is_realtime_running = True
        self.realtime_read_button.config(state=tkinter.DISABLED)
        self.stop_read_button.config(state=tkinter.NORMAL)
        self.write_log_to_text("开始实时采集，数据正写入CSV文件...")

        self.realtime_monitor_loop()

    # --- 【修改】停止实时监测 ---
    def stop_realtime_monitor(self):
        """停止实时采集循环"""
        self.is_realtime_running = False
        self.realtime_read_button.config(state=tkinter.NORMAL)
        self.stop_read_button.config(state=tkinter.DISABLED)

        # 可以在停止时提示最终保存的点数
        msg = f"已停止监测。\n振动数据共保存: {self.total_vib_saved} 点\n电流数据共保存: {self.total_curr_saved} 点"
        self.write_log_to_text(msg)

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
                    self.write_log_to_text("读取间隔必须大于0，已恢复默认值。")
            except ValueError:
                interval = int(DEFAULT_INTERVAL_MS)
                self.write_log_to_text("读取间隔输入无效，已恢复默认值。")

            # 确保在实时监测时，界面上的倍率回读值也能实时跳变
            self.update_override_status()

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
            if (
                self.sample_index % (10 * incremental_data["N_inc_curr_points"]) == 0
            ):  # 比如每 10 次增量更新打印一次
                self.write_log_to_text(
                    f'实时监测数据更新完成 (振动增量: {incremental_data["N_inc_vib_points"]}点, 电流增量: {incremental_data["N_inc_curr_points"]}点, 累计采样点: {self.sample_index})'
                )

        except Exception as e:
            self.write_log_to_text(f"实时监测循环发生错误: {str(e)}")
            self.stop_realtime_monitor()  # 发生错误时停止循环

        # 安排下一次运行
        if self.is_realtime_running:
            self.init_windows_name.after(interval, self.realtime_monitor_loop)


# 主程序
def Gui_Start():
    plt.rcParams["font.sans-serif"] = ["SimHei"]  # 支持中文显示
    plt.rcParams["axes.unicode_minus"] = False

    init_window = tkinter.Tk()
    MAIN_Window = GUI(init_window)
    init_window.mainloop()


if __name__ == "__main__":
    Gui_Start()
