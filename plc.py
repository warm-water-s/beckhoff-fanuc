# -*- coding:utf-8 -*-
"""
    通过ui界面来打开端口,并通过按钮来实现实时读取/停止的操作,读取的数据会存放在txt文件中
    已经通过现场测试证明可以采集出数据，但数据的有效性值得进一步验证
"""
import pyads
import tkinter
import time
import os

# 日志默认条数
LOG_LINE_NUM = 0
# 全局PLC连接对象
Plc = None
# 数组配置（根据PLC定义）
CHANNEL_COUNT = 33        # 通道数量
SAMPLE_COUNT = 100        # 每个通道采样点数
GVL_BUFFER_LENGTH = CHANNEL_COUNT * SAMPLE_COUNT  # GvlBuffer数组长度

GVL_BUFFER_DATATYPE = pyads.PLCTYPE_INT  # 数组元素类型为INT
GVL_BUFFER_GROUP = 0x4020  # %MB对应的indexgroup
GVL_BUFFER_OFFSET = 0x0    # 偏移量MB0

# 默认连接参数
DEFAULT_AMS_NETID = "5.136.192.215.1.1"
DEFAULT_PORT = "851"

class GUI():
    def __init__(self, init_windows_name):
        self.init_windows_name = init_windows_name
        self.save_path = tkinter.StringVar(value="gvl_buffer_data.txt")  # 默认保存路径

    def set_init_window(self):
        # 窗口基础设置
        self.init_windows_name.title('ADS 通讯 - GvlBuffer读取')
        self.init_windows_name.geometry('750x500+30+30')
        self.init_windows_name.attributes('-alpha', 0.95)

        # ========== 原有组件保持不变 ==========
        # 标签部分
        self.ads_communication_title = tkinter.Label(self.init_windows_name, text='通讯参数配置')
        self.ads_communication_title.grid(row=0, column=0)
        self.ads_communication_log = tkinter.Label(self.init_windows_name, text='通讯日志')
        self.ads_communication_log.grid(row=0, column=12)
        
        self.ads_communication_netID = tkinter.Label(self.init_windows_name, text='AmsNetID')
        self.ads_communication_netID.grid(row=1, column=0)
        self.ads_communication_port = tkinter.Label(self.init_windows_name, text='Port')
        self.ads_communication_port.grid(row=2, column=0)
        
        # 文本框部分
        self.log_text = tkinter.Text(self.init_windows_name, width=50, height=25)
        self.log_text.grid(row=1, column=12, rowspan=15)
        
        # 设置默认AmsNetID
        self.netID_text = tkinter.Text(self.init_windows_name, width=20, height=1)
        self.netID_text.grid(row=1, column=1)
        self.netID_text.insert(tkinter.END, DEFAULT_AMS_NETID)
        
        # 设置默认Port
        self.port_text = tkinter.Text(self.init_windows_name, width=20, height=1)
        self.port_text.grid(row=2, column=1)
        self.port_text.insert(tkinter.END, DEFAULT_PORT)

        # ========== 新增GvlBuffer相关组件 ==========
        self.save_path_label = tkinter.Label(self.init_windows_name, text='保存路径')
        self.save_path_label.grid(row=8, column=0)
        
        self.save_path_entry = tkinter.Entry(self.init_windows_name, textvariable=self.save_path, width=25)
        self.save_path_entry.grid(row=8, column=1)
        
        self.read_interval_label = tkinter.Label(self.init_windows_name, text='读取间隔(ms)')
        self.read_interval_label.grid(row=9, column=0)
        
        self.interval_text = tkinter.Text(self.init_windows_name, width=20, height=1)
        self.interval_text.grid(row=9, column=1)
        self.interval_text.insert(tkinter.END, "10")  # 默认10ms
        
        # 按钮部分
        self.delete_all_button = tkinter.Button(self.init_windows_name, text='清空参数', bg='lightblue', width=10,
                                                command=self.delete_all_parameter)
        self.delete_all_button.grid(row=0, column=1)
        
        self.delete_log_button = tkinter.Button(self.init_windows_name, text='清空日志', bg='lightblue', width=10,
                                                command=self.delete_log)
        self.delete_log_button.grid(row=0, column=2)
        
        self.open_port_button = tkinter.Button(self.init_windows_name, text='打开端口', width=10,
                                               command=self.Plc_port_open)
        self.open_port_button.grid(row=1, column=2)
        
        # ========== 新增按钮 ==========
        self.read_gvlbuffer_button = tkinter.Button(self.init_windows_name, text='读取GvlBuffer', width=15,
                                                    command=self.read_gvlbuffer_once)
        self.read_gvlbuffer_button.grid(row=4, column=2)
        
        self.realtime_read_button = tkinter.Button(self.init_windows_name, text='开始实时读取', width=15,
                                                   command=self.start_realtime_read)
        self.realtime_read_button.grid(row=5, column=2)
        
        self.stop_read_button = tkinter.Button(self.init_windows_name, text='停止实时读取', width=15,
                                               command=self.stop_realtime_read, state=tkinter.DISABLED)
        self.stop_read_button.grid(row=6, column=2)

        # 实时读取状态标志
        self.is_realtime_running = False

    # ========== 原有功能函数保持不变 ==========
    def delete_all_parameter(self):
        try:
            self.netID_text.delete(1.0, tkinter.END)
            self.port_text.delete(1.0, tkinter.END)
            self.interval_text.delete(1.0, tkinter.END)
            self.interval_text.insert(tkinter.END, "1000")
            self.write_log_to_text('清空所有参数')
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
        
        self.log_text.see(tkinter.END)  # 自动滚动到最后
        self.log_text.update()

    # ========== 修复后的GvlBuffer读取功能 ==========
    def read_gvlbuffer_once(self):
        """单次读取GvlBuffer数组并保存（使用地址方式）"""
        global Plc
        if not Plc or not Plc.is_open:
            self.write_log_to_text('请先打开PLC端口')
            return

        try:
            self.write_log_to_text('开始读取GvlBuffer数组...')
            
            # 使用地址方式读取（修复后的关键部分）
            gvl_buffer_data = []
            
            # 分块读取（每次读取1000个INT，避免单次读取过大）
            chunk_size = 1000
            for i in range(0, GVL_BUFFER_LENGTH, chunk_size):
                # 计算当前块的偏移量（每个INT占2字节）
                current_offset = GVL_BUFFER_OFFSET + i * 2
                # 计算当前块的长度
                current_length = min(chunk_size, GVL_BUFFER_LENGTH - i)
                
                # 读取当前块数据
                chunk_data = Plc.read(GVL_BUFFER_GROUP, current_offset, 
                                     pyads.PLCTYPE_INT * current_length)
                
                gvl_buffer_data.extend(chunk_data)
            
            # 保存到文件（按通道列排列）
            self.save_data_to_file(gvl_buffer_data)
            
            self.write_log_to_text(f'成功读取{len(gvl_buffer_data)}个数据')
            self.write_log_to_text(f'数据已保存到: {self.save_path.get()}')
            
        except Exception as e:
            self.write_log_to_text(f'读取失败: {str(e)}')
            print(f"读取错误: {e}")

    def save_data_to_file(self, data):
        """
        按通道列保存数据：
        - 每行：1个采样时刻的80个通道数据
        - 每列：1个通道的100个采样点数据
        """
        filepath = self.save_path.get()
        try:
            # 重新组织数据：按通道拆分后转置
            # 1. 将8000个数据分成80个通道，每个通道100个采样点
            channels_data = []
            for channel in range(CHANNEL_COUNT):
                # 每个通道的起始索引：channel * SAMPLE_COUNT
                start_idx = channel * SAMPLE_COUNT
                # 每个通道的结束索引：(channel + 1) * SAMPLE_COUNT
                end_idx = start_idx + SAMPLE_COUNT
                # 提取该通道的所有采样点
                channel_samples = data[start_idx:end_idx]
                channels_data.append(channel_samples)
            
            # 2. 转置数据：从(80通道×100点)转为(100点×80通道)
            transposed_data = list(zip(*channels_data))
            
            with open(filepath, 'a', encoding='utf-8') as f:
                # 写入时间戳和说明
                f.write(f"=== 采集时间: {self.get_current_time()} ===\n")
                
                # 写入数据（每行一个采样点，每列一个通道）
                for sample_idx, sample_data in enumerate(transposed_data):
                    # 行格式：采样点序号 + 80个通道数据（制表符分隔）
                    row =[str(val) for val in sample_data]
                    f.write('\t'.join(row) + '\n')
                
                # 写入分隔线
                # f.write('='*100 + '\n\n')
                
        except Exception as e:
            self.write_log_to_text(f'文件保存失败: {str(e)}')

    def start_realtime_read(self):
        """开始实时读取"""
        if not Plc or not Plc.is_open:
            self.write_log_to_text('请先打开PLC端口')
            return

        self.is_realtime_running = True
        self.realtime_read_button.config(state=tkinter.DISABLED)
        self.stop_read_button.config(state=tkinter.NORMAL)
        self.write_log_to_text('开始实时读取GvlBuffer...')
        
        self.realtime_read_loop()

    def stop_realtime_read(self):
        """停止实时读取"""
        self.is_realtime_running = False
        self.realtime_read_button.config(state=tkinter.NORMAL)
        self.stop_read_button.config(state=tkinter.DISABLED)
        self.write_log_to_text('已停止实时读取')

    def realtime_read_loop(self):
        """实时读取循环"""
        if not self.is_realtime_running:
            return

        try:
            # 读取间隔（毫秒转秒）
            interval = int(self.interval_text.get(1.0, tkinter.END).strip()) / 1000
            
            # 读取数据（使用修复后的地址读取方式）
            gvl_buffer_data = []
            chunk_size = 1000
            for i in range(0, GVL_BUFFER_LENGTH, chunk_size):
                current_offset = GVL_BUFFER_OFFSET + i * 2
                current_length = min(chunk_size, GVL_BUFFER_LENGTH - i)
                chunk_data = Plc.read(GVL_BUFFER_GROUP, current_offset, 
                                     pyads.PLCTYPE_INT * current_length)
                gvl_buffer_data.extend(chunk_data)
            
            # 保存数据（按通道列排列）
            self.save_data_to_file(gvl_buffer_data)
            
            self.write_log_to_text(f'实时读取完成，下次读取将在{interval}秒后')
            
        except Exception as e:
            self.write_log_to_text(f'实时读取错误: {str(e)}')
        
        # 定时执行下一次读取
        if self.is_realtime_running:
            self.init_windows_name.after(int(interval*1000), self.realtime_read_loop)

# 主程序
def Gui_Start():
    init_window = tkinter.Tk()
    MAIN_Window = GUI(init_window)
    MAIN_Window.set_init_window()
    init_window.mainloop()

if __name__ == "__main__":
    Gui_Start()