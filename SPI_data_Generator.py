import numpy as np
from scipy.signal import butter, lfilter
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
import csv
from datetime import datetime


class RealisticSPISignalGenerator:
    def __init__(self, sampling_rate=1000000, spi_clock_freq=100000):
        """
        初始化SPI信号生成器

        参数:
        sampling_rate: 采样率，默认1000000 Hz
        spi_clock_freq: SPI时钟频率，默认100000 Hz  SPI协议的信号速率比较宽泛，从几KHZ到10Mhz都可以，根据实际情况调整
        """
        self.sampling_rate = sampling_rate
        self.spi_clock_freq = spi_clock_freq
        self.samples_per_clock_cycle = int(sampling_rate / spi_clock_freq)

        # 非理想特性参数
        self.noise_level = 0.03
        self.jitter_std = 0.02
        self.rise_time = 0.2
        self.fall_time = 0.2
        self.voltage_high = 3.3
        self.voltage_low = 0.0
        self.voltage_noise_std = 0.03

        # # 信号滤波器参数
        # self.filter_cutoff = 1e6
        # self.filter_order = 4
        # self.filter_cutoff_normalized = 0.8

    def generate_spi_frame(self, data, cs_active_low=True):
        """
        生成一个SPI帧

        参数:
        data: 要发送的数据，是一个整数列表，每个整数代表一个字节数据
        cs_active_low: 片选信号是否低电平有效，默认为True

        返回:
        sck: SPI时钟信号数组
        mosi: 主设备输出从设备输入信号数组
        miso: 主设备输入从设备输出信号数组（这里简单模拟为固定值，可根据实际情况修改）
        cs: 片选信号数组
        """
        num_bits = len(data) * 8
        sck = np.zeros(num_bits * self.samples_per_clock_cycle)
        mosi = np.zeros(num_bits * self.samples_per_clock_cycle)
        miso = np.ones(num_bits * self.samples_per_clock_cycle) * self.voltage_high  # 简单模拟固定返回高电平，可按需改
        cs_samples = self.samples_per_clock_cycle * 16  # 片选信号持续时间示例

        if cs_active_low:
            cs = np.ones(cs_samples) * self.voltage_high
            cs[cs_samples // 2:] = self.voltage_low
        else:
            cs = np.zeros(cs_samples)
            cs[cs_samples // 2:] = self.voltage_high

        for byte_idx, byte_data in enumerate(data):
            byte_start = byte_idx * 8 * self.samples_per_clock_cycle
            for bit_idx in range(7, -1, -1):
                bit = (byte_data >> bit_idx) & 1
                bit_start = byte_start + (7 - bit_idx) * self.samples_per_clock_cycle
                mid_bit_idx = bit_start + self.samples_per_clock_cycle // 2
                sck[bit_start:mid_bit_idx] = self.voltage_high
                mosi[bit_start:mid_bit_idx] = self.voltage_high if bit else self.voltage_low

        # 添加非理想特性
        sck = self.add_noise(sck)
        sck = self.add_jitter(sck, self.find_edges(sck))
        sck = self.add_transition_time(sck)
        # sck = self.apply_lowpass_filter(sck)

        mosi = self.add_noise(mosi)
        mosi = self.add_jitter(mosi, self.find_edges(mosi))
        mosi = self.add_transition_time(mosi)
        # mosi = self.apply_lowpass_filter(mosi)

        miso = self.add_noise(miso)
        miso = self.add_jitter(miso, self.find_edges(miso))
        miso = self.add_transition_time(miso)
        # miso = self.apply_lowpass_filter(miso)

        cs = self.add_noise(cs)
        cs = self.add_jitter(cs, self.find_edges(cs))
        cs = self.add_transition_time(cs)
        # cs = self.apply_lowpass_filter(cs)

        return sck, mosi, miso, cs

    def add_noise(self, signal):
        """给信号添加噪声"""
        noise = np.random.normal(0, self.voltage_noise_std, len(signal))
        return signal + noise

    def add_jitter(self, signal, edge_indices):
        """给信号添加抖动"""
        jittered_signal = signal.copy()
        for idx in edge_indices:
            jitter_samples = int(np.random.normal(0, self.jitter_std * self.samples_per_clock_cycle))
            if idx + jitter_samples < len(signal) and idx + jitter_samples >= 0:
                if idx > 0:
                    jittered_signal[idx] = signal[idx + jitter_samples]
        return jittered_signal

    def add_transition_time(self, signal):
        """添加信号的上升下降沿过渡时间"""
        result = signal.copy()
        rise_samples = int(self.rise_time * self.samples_per_clock_cycle)
        fall_samples = int(self.fall_time * self.samples_per_clock_cycle)

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
    #     """应用低通滤波器到信号上"""
    #     b, a = butter(self.filter_order, self.filter_cutoff_normalized, btype='low')
    #     return lfilter(b, a, signal)

    def find_edges(self, signal):
        """查找信号的边沿位置（上升沿或下降沿）"""
        return np.where(np.abs(np.diff(signal)) > 0.5)[0]

    def generate_multiple_frames(self, data_list, total_samples=1000000):
        """
        生成多个SPI帧，直到达到指定的总采样点数

        参数:
        data_list: 包含多个要发送的数据列表的列表，每个子列表代表一个SPI帧的数据
        total_samples: 要生成的总采样点数，默认1000000

        返回:
        sck_total: 拼接后的SPI时钟信号总数组
        mosi_total: 拼接后的主设备输出从设备输入信号总数组
        miso_total: 拼接后的主设备输入从设备输出信号总数组
        cs_total: 拼接后的片选信号总数组
        """
        sck_total = np.array([])
        mosi_total = np.array([])
        miso_total = np.array([])
        cs_total = np.array([])

        current_samples = 0
        for data in data_list:
            sck, mosi, miso, cs = self.generate_spi_frame(data)
            frame_samples = len(sck)
            if current_samples + frame_samples > total_samples:
                break
            sck_total = np.append(sck_total, sck)
            mosi_total = np.append(mosi_total, mosi)
            miso_total = np.append(miso_total, miso)
            cs_total = np.append(cs_total, cs)
            current_samples += frame_samples

            # 添加帧间间隔（这里简单示例为固定间隔的高电平信号，可按需调整）
            idle_samples = self.samples_per_clock_cycle * 10
            idle_sck = np.ones(idle_samples) * self.voltage_high
            idle_mosi = np.ones(idle_samples) * self.voltage_high
            idle_miso = np.ones(idle_samples) * self.voltage_high
            idle_cs = np.ones(idle_samples) * self.voltage_high

            sck_total = np.append(sck_total, idle_sck)
            mosi_total = np.append(mosi_total, idle_mosi)
            miso_total = np.append(miso_total, idle_miso)
            cs_total = np.append(cs_total, idle_cs)
            current_samples += idle_samples

        # 裁剪到指定长度
        sck_total = sck_total[:total_samples]
        mosi_total = mosi_total[:total_samples]
        miso_total = miso_total[:total_samples]
        cs_total = cs_total[:total_samples]

        return sck_total, mosi_total, miso_total, cs_total

    def plot_signals(self, sck, mosi, miso, cs, title="Realistic SPI Signal", plot_samples=None):
        """
        绘制SPI信号波形，可选择绘制的采样点数

        参数:
        sck: SPI时钟信号数组
        mosi: 主设备输出从设备输入信号数组
        miso: 主设备输入从设备输出信号数组
        cs: 片选信号数组
        title: 图像标题，默认为"Realistic SPI Signal"
        plot_samples: 要绘制的采样点数，默认为None（绘制合适默认长度）
        """
        if plot_samples is None:
            # 默认显示前若干个时钟周期的数据，可根据实际调整
            plot_samples = self.samples_per_clock_cycle * 10

        plot_samples = min(plot_samples, len(sck))
        time = np.arange(plot_samples) / self.sampling_rate * 1e6  # 转换为微秒

        fig = plt.figure(figsize=(15, 10))

        plt.plot(time, sck[:plot_samples], 'b-', label='SCK', linewidth=1)
        # plt.plot(time, mosi[:plot_samples], 'r-', label='MOSI', linewidth=1)
        # plt.plot(time, miso[:plot_samples], 'g-', label='MISO', linewidth=1)
        # plt.plot(time, cs[:plot_samples], 'k-', label='CS', linewidth=1)
        plt.grid(True)
        plt.ylabel('Voltage (V)')
        plt.title(title)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_to_csv(self, sck, mosi, miso, cs, filename=None):
        """
        保存信号数据到CSV文件，使用整数索引

        参数:
        sck: SPI时钟信号数组
        mosi: 主设备输出从设备输入信号数组
        miso: 主设备输入从设备输出信号数组
        cs: 片选信号数组
        filename: 保存的文件名，默认为None（自动生成带时间戳的文件名）
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'spi_data_{timestamp}.csv'

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'SCK (V)', 'MOSI (V)', 'MISO (V)', 'CS (V)'])
            print("sck长度:", len(sck))
            print("mosi长度:", len(mosi))
            print("miso长度:", len(miso))
            print("cs长度:", len(cs))
            for i in range(len(sck)):
                writer.writerow([i, round(sck[i], 6), round(mosi[i], 6), round(miso[i], 6), round(cs[i], 6)])

        print(f"Data saved to {filename}")
        print(f"Total samples: {len(sck)}")

    def plot_fft(self, signal, title="FFT Spectrum"):
        """
        绘制信号的FFT频谱图

        参数:
        signal: 要绘制频谱的信号数组
        title: 频谱图标题，默认为"FFT Spectrum"
        """
        N = len(signal)
        yf = fft(signal)
        xf = fftfreq(N, 1 / self.sampling_rate)[:N // 2]

        plt.figure(figsize=(12, 6))
        plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
        plt.xlabel('Frequency (Hz)')
        plt.ylabel('Amplitude')
        plt.title(title)
        plt.grid()
        plt.show()

# 使用示例
if __name__ == "__main__":
    # 创建信号生成器实例
    gen = RealisticSPISignalGenerator(sampling_rate=1000000)

    # 生成一些示例数据，这里生成3个SPI帧的数据，每个帧包含2个字节数据（可按需调整）
    data_list = [
        [np.random.randint(0, 256), np.random.randint(0, 256)],
        [np.random.randint(0, 256), np.random.randint(0, 256)],
        [np.random.randint(0, 256), np.random.randint(0, 256)]
    ]

    # 生成多个SPI帧的信号
    sck, mosi, miso, cs = gen.generate_multiple_frames(data_list, 100000)

    # 保存到CSV文件
    gen.save_to_csv(sck, mosi, miso, cs, 'spi_million_samples.csv')

    # 绘制信号波形
    gen.plot_signals(sck[:500], mosi[:500], miso[:500], cs[:500], "SPI Signal (First 10k samples)", plot_samples=50)

    # # 绘制信号的FFT频谱图
    # gen.plot_fft(sck[:10000], "FFT Spectrum of SCK Signal")
    # gen.plot_fft(mosi[:10000], "FFT Spectrum of MOSI Signal")
    # gen.plot_fft(miso[:10000], "FFT Spectrum of MISO Signal")
    # gen.plot_fft(cs[:10000], "FFT Spectrum of CS Signal")