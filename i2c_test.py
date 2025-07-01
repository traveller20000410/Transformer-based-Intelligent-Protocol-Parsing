import numpy as np
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import csv
from datetime import datetime


class RealisticI2CSignalGenerator:
    def __init__(self, sampling_rate=1000000):
        # I2C常见信号速率
        # 标准模式（Standard Mode）‌：速率约为100kbps（100kbit / s）‌12。
        # 快速模式（FastMode）‌：速率约为400kbps（400kbit / s）‌
        # 快速 + 模式（Fast - PlusMode）‌：速率约为1Mbps（1Mbit / s）‌
        # 高速模式（High - SpeedMode）‌：速率约为3.4Mbps（3.4Mbit / s）‌
        # 超高速模式（Ultra - FastMode）‌：速率约为5Mbps（5Mbit / s，单向传输）‌
        self.sampling_rate = sampling_rate
        self.scl_freq = 100000  # 100kHz SCL频率
        self.samples_per_bit = int(sampling_rate / self.scl_freq)
        # self.filter_cutoff_normalized = 0.8  # 直接使用归一化频率

        # 非理想特性参数
        self.noise_level = 0.03
        self.jitter_std = 0.04
        self.rise_time = 0.2
        self.fall_time = 0.2
        self.voltage_high = 3.3
        self.voltage_low = 0.0
        self.voltage_noise_std = 0.03

        # # 信号滤波器参数
        # self.filter_cutoff = 1e6
        # self.filter_order = 4

    def generate_multiple_frames(self, total_samples=10000):
        """生成多个I2C帧，直到达到指定的总采样点数"""
        scl_total = np.array([])
        sda_total = np.array([])
        frame_starts = []  # 记录每个帧的起始位置

        while len(scl_total) < total_samples:
            # 固定从机地址（这里示例设置为0x30，可根据需求修改），只取低7位作为地址位
            slave_addr = 0x30
            # 固定数据字节数为5，且固定数据内容
            data_bytes = [0x1, 0x2, 0x3, 0x4, 0x5]

            # 记录当前帧的起始位置
            frame_starts.append(len(scl_total))

            # 生成单个帧
            scl_frame, sda_frame = self.generate_i2c_frame(slave_addr, data_bytes)

            # 添加帧间隔
            # idle_samples = np.random.randint(100, 1000)  # 随机帧间隔
            idle_scl = np.ones(50) * self.voltage_high
            idle_sda = np.ones(50) * self.voltage_high
            # 添加噪声
            idle_scl = self.add_noise(idle_scl)
            idle_sda = self.add_noise(idle_sda)

            # 拼接信号
            scl_total = np.append(scl_total, np.append(scl_frame, idle_scl))
            sda_total = np.append(sda_total, np.append(sda_frame, idle_sda))

        # 裁剪到指定长度
        scl_total = scl_total[:total_samples]
        sda_total = sda_total[:total_samples]

        return scl_total, sda_total, frame_starts

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

    # def apply_lowpass_filter(self, signal):
    #     b, a = butter(self.filter_order, self.filter_cutoff_normalized, btype='low')
    #     return lfilter(b, a, signal)

    def find_edges(self, signal):
        return np.where(np.abs(np.diff(signal)) > 0.5)[0]

    def generate_clock(self, num_bits):
        samples_total = num_bits * self.samples_per_bit
        scl = np.zeros(samples_total)

        for i in range(num_bits):
            start_idx = i * self.samples_per_bit
            mid_idx = start_idx + self.samples_per_bit // 2
            scl[start_idx:mid_idx] = self.voltage_high

        edges = self.find_edges(scl)
        scl = self.add_transition_time(scl)
        scl = self.add_jitter(scl, edges)
        scl = self.add_noise(scl)
        # scl = self.apply_lowpass_filter(scl)

        return scl

    def generate_data_bit(self, bit_value):
        sda = np.ones(self.samples_per_bit) * (self.voltage_high if bit_value else self.voltage_low)

        edges = self.find_edges(sda)
        sda = self.add_transition_time(sda)
        sda = self.add_jitter(sda, edges)
        sda = self.add_noise(sda)
        # sda = self.apply_lowpass_filter(sda)

        return sda

    def generate_i2c_frame(self, slave_addr, data_bytes):
        """生成多字节数据的I2C帧"""
        # 确保传入的数据字节为列表形式
        if not isinstance(data_bytes, list):
            data_bytes = [data_bytes]

        # 生成起始条件
        start_samples = self.samples_per_bit
        sda_start = np.ones(start_samples) * self.voltage_high
        sda_start[start_samples // 2:] = self.voltage_low
        sda_start = self.add_noise(sda_start)
        scl_start = np.ones(start_samples) * self.voltage_high
        scl_start = self.add_noise(scl_start)

        # 生成地址
        scl_addr = self.generate_clock(9)  # 8位地址+读写位+ACK
        sda_addr = np.array([])
        addr_with_rw = (slave_addr << 1) & 0xFE

        for i in range(7, -1, -1):
            bit = (addr_with_rw >> i) & 1
            sda_addr = np.append(sda_addr, self.generate_data_bit(bit))

        # ACK位
        sda_addr = np.append(sda_addr, self.generate_data_bit(0))

        # 生成多个数据字节
        scl_data = np.array([])
        sda_data = np.array([])

        for data_byte in data_bytes:
            # 每个字节的时钟
            scl_data = np.append(scl_data, self.generate_clock(9))  # 8位数据+ACK

            # 数据位
            for i in range(7, -1, -1):
                bit = (data_byte >> i) & 1
                sda_data = np.append(sda_data, self.generate_data_bit(bit))

            # 每个字节后的ACK
            sda_data = np.append(sda_data, self.generate_data_bit(0))

        # 停止条件
        stop_samples = self.samples_per_bit
        sda_stop = np.zeros(stop_samples)
        sda_stop[stop_samples // 2:] = self.voltage_high
        sda_stop = self.add_noise(sda_stop)
        scl_stop = np.ones(stop_samples) * self.voltage_high
        scl_stop = self.add_noise(scl_stop)

        # 组合完整帧
        scl = np.concatenate([scl_start, scl_addr, scl_data, scl_stop])
        sda = np.concatenate([sda_start, sda_addr, sda_data, sda_stop])
        return scl, sda

    def plot_signals(self, scl, sda, title="Realistic I2C Signal", plot_samples=None):
        """绘制I2C信号波形，可选择绘制的采样点数"""
        if plot_samples is None:
            # 默认显示前20个位时间的数据
            plot_samples = self.samples_per_bit * 20

        plot_samples = min(plot_samples, len(scl))
        time = np.arange(plot_samples) / self.sampling_rate * 1e6  # 转换为微秒

        fig = plt.figure(figsize=(15, 10))

        # 完整信号视图
        # plt.subplot(3, 1, 1)
        plt.plot(time, scl[:plot_samples], 'b-', label='SCL', linewidth=1)
        plt.plot(time, sda[:plot_samples], 'r-', label='SDA', linewidth=1)
        plt.grid(True)
        plt.ylabel('Voltage (V)')
        plt.title(title)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_to_csv(self, scl, sda, filename=None):
        """保存信号数据到CSV文件，使用整数索引"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'i2c_data_{timestamp}.csv'

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'SCL (V)', 'SDA (V)'])
            for i in range(len(scl)):
                writer.writerow([i, round(scl[i], 6), round(sda[i], 6)])

        print(f"Data saved to {filename}")
        print(f"Total samples: {len(scl)}")

    def generate_random_data(self, num_bytes):
        """生成指定数量的随机数据字节"""
        return [np.random.randint(0, 256) for _ in range(num_bytes)]

    # def plot_fft(self, signal, title="FFT Spectrum"):
    #     """绘制信号的FFT频谱图"""
    #     N = len(signal)
    #     yf = fft(signal)
    #     xf = fftfreq(N, 1 / self.sampling_rate)[:N // 2]
    #
    #     plt.figure(figsize=(12, 6))
    #     plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    #     plt.xlabel('Frequency (Hz)')
    #     plt.ylabel('Amplitude')
    #     plt.title(title)
    #     plt.grid()
    #     plt.show()
    #     print("fftok")


# 使用示例
if __name__ == "__main__":
    # 创建信号生成器实例,初始化采样率为1000000
    gen = RealisticI2CSignalGenerator(sampling_rate=1000000)

    # 生成10万个采样点的多个I2C帧
    scl, sda, frame_starts = gen.generate_multiple_frames(10000)
    scl_binary = np.where(scl > 1.65, 1, 0)
    sda_binary = np.where(sda > 1.65, 1, 0)
    # 保存到CSV文件
    gen.save_to_csv(scl, sda, 'i2c_samples.csv')
    gen.save_to_csv(scl_binary, sda_binary, 'i2c_samples_binary.csv')

    # # 绘制FFT频谱图
    # gen.plot_fft(scl, title="FFT Spectrum of SCL Signal")

    # 打印信号信息
    print(f"Generated signal info:")
    print(f"Total samples: {len(scl)}")
    print(f"Signal duration: {len(scl) / gen.sampling_rate * 1000:.2f} ms")

    # 打印统计信息
    print(f"Total number of frames: {len(frame_starts)}")
    print(f"Frame start positions: {frame_starts[:10]}... (showing first 10)")

    # 只绘制前2000个点的可视化图
    gen.plot_signals(scl[:700], sda[:700], "I2C Signal (First 20k samples)", plot_samples=700)
