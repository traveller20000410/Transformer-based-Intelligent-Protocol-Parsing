import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime
import os
import pandas as pd
from rs232_labeler import RS232Labeler


class RealisticRS232SignalGenerator:
    def __init__(self, baud_rate=9600):
        self.sample_rate_dict = {
            1: 100000,
            2: 200000,
        }
        self.signal_rate_dict = {
            1: 9600,
            2: 19200,
        }

        # self.sample_rate = self.sample_rate_dict[np.random.choice(list(self.sample_rate_dict.keys()))]
        self.signal_rate = self.signal_rate_dict[np.random.choice(list(self.signal_rate_dict.keys()))]
        #self.samples_per_symbol = int(self.sample_rate / self.signal_rate)  # 每个符号的采样点数，可以根据需要调整
        self.samples_per_bit = 10

        # 非理想特性参数
        self.noise_level = 0.03
        self.jitter_std = 0.05
        self.rise_time = 0.2
        self.fall_time = 0.2
        self.voltage_high = 3.3
        self.voltage_low = 0.0
        self.voltage_noise_std = 0.03

    def generate_multiple_frames(self, total_samples=None):
        """生成多个 RS232 帧，直到达到指定的总采样点数"""
        useless_bit_beforestart=np.random.randint(1, 50)*self.samples_per_bit   #随机生成第一个帧前的无用位
        tx_total = np.ones(useless_bit_beforestart) * self.voltage_high
        frame_starts = []  # 记录每个帧的起始位置

        while len(tx_total) < total_samples:
            # 随机生成 1-10 个字节的数据
            num_bytes = np.random.randint(1, 11)
            data_bytes = [np.random.randint(0, 256) for _ in range(num_bytes)]
            # 记录当前帧的起始位置
            frame_starts.append(len(tx_total))
            # 生成单个帧
            tx_frame = self.generate_rs232_frame(data_bytes)
            # 添加帧间隔
            idle_samples = np.random.randint(10, 50)*self.samples_per_bit  # 随机帧间隔
            idle_tx = np.ones(idle_samples) * self.voltage_high
            # 拼接信号
            tx_total = np.append(tx_total, np.append(tx_frame, idle_tx))
        # 裁剪到指定长度
        tx_total = tx_total[:total_samples]
        tx_total = self.add_noise(tx_total)

        return tx_total, frame_starts, useless_bit_beforestart

    def add_noise(self, signal):
        noise = np.random.normal(0, self.voltage_noise_std, len(signal))
        return signal + noise

    def add_jitter(self, signal, edge_indices):
        jittered_signal = signal.copy()
        for idx in edge_indices:
            jitter_samples = int(np.random.normal(0, self.jitter_std * self.samples_per_bit))
            if idx + jitter_samples < len(signal) and idx + jitter_samples >= 0:
                if idx > 0:
                    jittered_signal[idx] = signal[idx + jitter_samples]
        return jittered_signal

    def add_transition_time(self, signal):
        result = signal.copy()
        rise_samples = int(self.rise_time * self.samples_per_bit)
        fall_samples = int(self.fall_time * self.samples_per_bit)

        edges = np.where(np.diff(signal)!= 0)[0]

        for edge in edges:
            if edge + 1 < len(signal):
                if signal[edge] < signal[edge + 1]:
                    if edge + rise_samples < len(signal):
                        transition = np.linspace(self.voltage_low, self.voltage_high, rise_samples)
                        result[edge:edge + rise_samples] = transition
                else:
                    if edge + fall_samples < len(signal):
                        transition = np.linspace(self.voltage_high, self.voltage_low, fall_samples)
                        result[edge:edge + fall_samples] = transition

        return result

    def find_edges(self, signal):
        return np.where(np.abs(np.diff(signal)) > 0.5)[0]

    def generate_clock(self, num_bits):
        samples_total = num_bits * self.samples_per_bit
        clk = np.zeros(samples_total)

        for i in range(num_bits):
            start_idx = i * self.samples_per_bit
            mid_idx = start_idx + self.samples_per_bit // 2
            clk[start_idx:mid_idx] = self.voltage_high

        edges = self.find_edges(clk)
        clk = self.add_transition_time(clk)
        clk = self.add_jitter(clk, edges)

        return clk

    def generate_data_bit(self, bit_value):
        tx = np.ones(self.samples_per_bit) * (self.voltage_high if bit_value else self.voltage_low)
        # edges = self.find_edges(tx)
        # tx = self.add_transition_time(tx)
        # tx = self.add_jitter(tx, edges)

        return tx

    def generate_rs232_frame(self, data_bytes):
        """生成多字节数据的 RS232 帧"""
        if not isinstance(data_bytes, list):
            data_bytes = [data_bytes]
        tx=np.array([])
        tx_data=np.array([])

        tx_start = np.zeros(self.samples_per_bit) * self.voltage_low
        tx_stop = np.ones(self.samples_per_bit) * self.voltage_high

        # 生成数据位
        for data_byte in data_bytes:

            # 生成数据位
            for i in range(7, -1, -1):
                bit = (data_byte >> i) & 1
                tx_data = np.append(tx_data,self.generate_data_bit(bit))

            # 组合完整帧
            data_frame = np.concatenate([tx_start, tx_data, tx_stop])
            tx_data=np.array([])
            tx=np.append(tx,data_frame)

        return tx

    def plot_signals(self, tx, title="Realistic RS232 Signal", plot_samples=None):
        """绘制 RS232 信号波形，可选择绘制的采样点数"""
        if plot_samples is None:
            # 默认显示前 20 个位时间的数据
            plot_samples = self.samples_per_bit * 20

        plot_samples = min(plot_samples, len(tx))
        # time = np.arange(plot_samples) / self.sampling_rate * 1e6  # 转换为微秒
        time = np.arange(plot_samples) / 100000 * 1e6  # 转换为微秒

        fig = plt.figure(figsize=(15, 10))

        # 完整信号视图
        plt.plot(time, tx[:plot_samples], 'b-', label='TX', linewidth=1)
        plt.grid(True)
        plt.ylabel('Voltage (V)')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def save_to_csv(self, tx, filename=None):
        """保存信号数据到 CSV 文件，使用整数索引"""
        current_file_path = os.path.abspath(__file__)
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        target_dir = os.path.join(parent_dir, 'RS232_Data_Set')
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        file_path = os.path.join(target_dir, filename)

        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'rs232_data_{timestamp}.csv'

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            # writer.writerow(['Index', 'TX (V)'])
            for i in range(len(tx)):
                writer.writerow([round(tx[i], 6)])
        print(f"Data saved to {filename}")
        print(f"Total samples: {len(tx)}")

    def save_labeled_data(self,voltages, labeled_data, output_file_name):
        current_file_path = os.path.abspath(__file__)
        parent_dir = os.path.dirname(os.path.dirname(os.path.dirname(current_file_path)))
        target_dir = os.path.join(parent_dir, 'RS232_Data_Set')   #设定保存的目标文件夹
        if not os.path.exists(target_dir):
            os.makedirs(target_dir)
        file_path = os.path.join(target_dir, output_file_name)

        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['无效位是',useless_bit_beforestart])
            writer.writerow([frame_starts[:]])
            for i in range(len(voltages)):
                writer.writerow([round(voltages[i], 6), labeled_data[i]])

        print(f"Data saved to {output_file_name}")
        print(f"Total samples: {len(tx)}")

    def generate_rs232_datasets(self,num_datasets=100, samples_per_dataset=2000):
        protocols_dataset = []
        protocol_labels = []
        for i in range(num_datasets):
            sda, _ ,_ = self.generate_multiple_frames(samples_per_dataset)
            labeler = RS232Labeler()
            labels= labeler.label_sequence(sda,self.samples_per_bit)
            protocols_dataset.append(sda)
            protocol_labels.append(labels)

        return protocols_dataset, protocol_labels

    def plot_labeled_data(self, labeled_data, voltages, title="Labeled RS232 Data", plot_samples=None):
        if plot_samples is None:
            plot_samples = len(labeled_data)
        plot_samples = min(plot_samples, len(labeled_data), len(voltages))
        time = np.arange(plot_samples) / 100000 * 1e6  # 转换为微秒

        fig = plt.figure(figsize=(15, 10))
        # 绘制 voltages 数据，颜色为蓝色
        plt.plot(time, voltages[:plot_samples], 'b-', label='Voltages', linewidth=1)
        # 绘制 labeled_data 数据，颜色为红色
        plt.plot(time, labeled_data[:plot_samples], 'r-', label='Labeled Data', linewidth=1)
        plt.grid(True)
        plt.ylabel('Value')
        plt.title(title)
        plt.legend()
        plt.tight_layout()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建信号生成器实例，初始化波特率为 9600，采样率为 1000000
    gen = RealisticRS232SignalGenerator(baud_rate=9600)

    for i in range(100):
        tx, frame_starts,useless_bit_beforestart = gen.generate_multiple_frames(5000)
        # # 保存到 CSV 文件
        filename = f'rs232_samples{i + 1}.csv'
        gen.save_to_csv(tx, filename)
        print(f"Frame start positions: {frame_starts[:]}BY GENERATOR") #打印帧的起始位置，用来测试后面标记函数的正确性
        voltages = tx
        # 创建标注器实例
        labeler = RS232Labeler()
        # 标注整个电压序列
        labeled_data = labeler.label_sequence(voltages,gen.samples_per_bit)
        # 保存标注后的数据，可自定义输出文件名格式
        output_file_name = f'labeled_rs232_samples{i + 1}.csv'
        gen.save_labeled_data(voltages, labeled_data, output_file_name)
        gen.plot_labeled_data(labeled_data, voltages, title="Labeled RS232 Data", plot_samples=3000)


    # # 打印信号信息
    # print(f"Generated signal info:")
    # print(f"Total samples: {len(tx)}")
    # print(f"Signal duration: {len(tx) / gen.sampling_rate * 1000:.2f} ms")
    #
    # # 打印统计信息
    # print(f"Total number of frames: {len(frame_starts)}")
    #print(f"Frame start positions: {frame_starts[:]}")
    #
    # 只绘制前 2000 个点的可视化图
    # gen.plot_signals(tx[:3000], "RS232 Signal (First 30k samples)", plot_samples=3000)