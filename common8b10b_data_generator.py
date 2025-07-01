import numpy as np
import matplotlib.pyplot as plt
import csv
from datetime import datetime


class RealisticCommon8b10bSignalGenerator:
    def __init__(self, symbol_rate=2500000000):
        self.sample_rate_dict = {
            1: 5000000000,
            2: 10000000000,
            3: 20000000000,
        }
        self.signal_rate_dict ={
            1:2500000000, #PCIE1信号速率
            2:5000000000, #PCIE2信号速率
            }

        self.sample_rate = self.sample_rate_dict[np.random.choice(list(self.sample_rate_dict.keys()))]
        self.signal_rate = self.signal_rate_dict[np.random.choice(list(self.signal_rate_dict.keys()))]
        self.samples_per_symbol = self.sample_rate/self.signal_rate  # 每个符号的采样点数，可以根据需要调整
        # 非理想特性参数
        self.noise_level = 0.03
        self.jitter_std = 0.04
        self.rise_time = 0.2
        self.fall_time = 0.2
        self.voltage_high = 3.3
        self.voltage_low = 0.0
        self.voltage_noise_std = 0.03

    def generate_multiple_frames(self, total_samples=5000):
        """生成多个 8b10b 帧，直到达到指定的总采样点数"""
        data_total = np.array([])
        frame_starts = []  # 记录每个帧的起始位置

        while len(data_total) < total_samples:
            # 随机生成 1-10 个 8b10b 符号的数据
            num_symbols = np.random.randint(1, 11)
            data_symbols = [self.generate_random_8b10b_symbol() for _ in range(num_symbols)]

            # 记录当前帧的起始位置
            frame_starts.append(len(data_total))

            # 生成单个帧
            data_frame = self.generate_8b10b_frame(data_symbols)

            # 添加帧间隔
            idle_samples = np.random.randint(100, 1000)  # 随机帧间隔
            idle_data = np.ones(idle_samples) * self.voltage_high
            # 添加噪声
            idle_data = self.add_noise(idle_data)

            # 拼接信号
            data_total = np.append(data_total, np.append(data_frame, idle_data))

        # 裁剪到指定长度
        data_total = data_total[:total_samples]

        return data_total, frame_starts

    def add_noise(self, signal):
        noise = np.random.normal(0, self.voltage_noise_std, len(signal))
        return signal + noise

    def add_jitter(self, signal, edge_indices):
        jittered_signal = signal.copy()
        for idx in edge_indices:
            jitter_samples = int(np.random.normal(0, self.jitter_std * self.samples_per_symbol))
            if idx + jitter_samples < len(signal) and idx + jitter_samples >= 0:
                if idx > 0:
                    jittered_signal[idx] = signal[idx + jitter_samples]
        return jittered_signal

    def add_transition_time(self, signal):
        result = signal.copy()
        rise_samples = int(self.rise_time * self.samples_per_symbol)
        fall_samples = int(self.fall_time * self.samples_per_symbol)

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

    def generate_random_8b10b_symbol(self):
        """生成一个随机的 8 位数据字节，范围是 0 到 255"""
        return np.random.randint(0, 256)

    def encode_8b10b(self, byte):
        """8b10b 编码函数，根据 8b10b 编码规则进行编码"""
        disparity = 0  # 初始的极性偏差
        running_disparity = 0  # 运行时的极性偏差
        def running_disparity_check(encoded, running_disparity):
            ones_count = bin(encoded).count('1')
            if (ones_count % 2) == 0:
                return encoded, running_disparity
            elif running_disparity == 0:
                encoded = encoded | 0x01  # 增加最低位
                running_disparity = 1
            else:
                encoded = encoded & 0x3FF  # 清除最低位
                running_disparity = 0
            return encoded, running_disparity

        def encode_5b6b(data):
            """将 5 位数据编码为 6 位 8b10b 编码"""
            encoding_5b6b = {
                0: 0b110001, 1: 0b110010, 2: 0b100011, 3: 0b100101, 4: 0b100110, 5: 0b101001, 6: 0b101010, 7: 0b101100,
                8: 0b010011, 9: 0b010101, 10: 0b010110, 11: 0b011001, 12: 0b011010, 13: 0b011100, 14: 0b001011, 15: 0b001101,
                16: 0b001110, 17: 0b110100, 18: 0b110110, 19: 0b111000, 20: 0b111010, 21: 0b111100, 22: 0b101101, 23: 0b101110,
                24: 0b101011, 25: 0b100111, 26: 0b100001, 27: 0b100010, 28: 0b110000, 29: 0b010001, 30: 0b010010, 31: 0b011000
            }
            return encoding_5b6b[data]

        def encode_3b4b(data):
            """将 3 位数据编码为 4 位 8b10b 编码"""
            encoding_3b4b = {
                0: 0b1011, 1: 0b1001, 2: 0b1010, 3: 0b1100, 4: 0b1101, 5: 0b1110, 6: 0b0101, 7: 0b0110
            }
            return encoding_3b4b[data]

        five_bit = (byte >> 3) & 0x1F  # 高 5 位
        three_bit = byte & 0x07  # 低 3 位
        encoded_5b6b = encode_5b6b(five_bit)
        encoded_3b4b = encode_3b4b(three_bit)
        encoded = (encoded_5b6b << 4) | encoded_3b4b
        encoded, running_disparity = running_disparity_check(encoded, running_disparity)
        return encoded

    def generate_8b10b_frame(self, data_symbols):
        """生成 8b10b 帧"""
        data = np.array([])
        for symbol in data_symbols:
            encoded_symbol = self.encode_8b10b(symbol)
            symbol_data = self.generate_symbol_data(encoded_symbol)
            data = np.append(data, symbol_data)
        return data

    def generate_symbol_data(self, symbol):
        """将 8b10b 符号转换为电压信号"""
        symbol_bits = [(symbol >> i) & 1 for i in range(9, -1, -1)]
        symbol_data = np.array([])
        for bit in symbol_bits:
            bit_value = self.voltage_high if bit else self.voltage_low
            bit_signal = np.ones(self.samples_per_symbol) * bit_value
            symbol_data = np.append(symbol_data, bit_signal)

        edges = self.find_edges(symbol_data)
        symbol_data = self.add_transition_time(symbol_data)
        symbol_data = self.add_jitter(symbol_data, edges)
        symbol_data = self.add_noise(symbol_data)
        return symbol_data

    def plot_signals(self, data, title="Realistic 8b10b Signal", plot_samples=None):
        """绘制 8b10b 信号波形，可选择绘制的采样点数"""
        if plot_samples is None:
            # 默认显示前 20 个符号的数据
            plot_samples = self.samples_per_symbol * 20

        plot_samples = min(plot_samples, len(data))
        time = np.arange(plot_samples) / (self.symbol_rate * self.samples_per_symbol) * 1e6  # 转换为微秒

        fig = plt.figure(figsize=(15, 10))

        # 完整信号视图
        plt.plot(time, data[:plot_samples], 'b-', label='Data', linewidth=1)
        plt.grid(True)
        plt.ylabel('Voltage (V)')
        plt.title(title)
        plt.legend()

        plt.tight_layout()
        plt.show()

    def save_to_csv(self, data, filename=None):
        """保存信号数据到 CSV 文件，使用整数索引"""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f'8b10b_data_{timestamp}.csv'

        with open(filename, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Index', 'Data (V)'])
            for i in range(len(data)):
                writer.writerow([i, round(data[i], 6)])

        print(f"Data saved to {filename}")
        print(f"Total samples: {len(data)}")

    def generate_common8b10b_datasets(self, num_datasets=100, samples_per_dataset=5000):
        protocols_dataset = []
        protocol_names = []
        for i in range(num_datasets):
            sda, _ = self.generate_multiple_frames(samples_per_dataset)
            protocols_dataset.append(sda)
            protocol_names.append('common8b10b')

        return protocols_dataset, protocol_names

# 使用示例
if __name__ == "__main__":
    # 创建信号生成器实例，初始化符号速率为 1000000
    gen = RealisticCommon8b10bSignalGenerator(symbol_rate=2500000000)

    # 生成 5000个采样点的多个 8b10b 帧
    data, frame_starts = gen.generate_multiple_frames(5000)

    # 保存到 CSV 文件
    gen.save_to_csv(data, '8b10b_million_samples.csv')

    # 打印信号信息
    print(f"Generated signal info:")
    print(f"Total samples: {len(data)}")
    print(f"Signal duration: {len(data) / (gen.symbol_rate * gen.samples_per_symbol) * 1000:.2f} ms")

    # 打印统计信息
    print(f"Total number of frames: {len(frame_starts)}")
    print(f"Frame start positions: {frame_starts[:10]}... (showing first 10)")

    # 只绘制前 2000 个点的可视化图
    gen.plot_signals(data[:5000], "8b10b Signal (First 20k samples)", plot_samples=5000)