import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# --- UART 标签定义 (新增异常标签) ---
LABEL_MAP = {
    "IDLE": 0,
    "START": 1,
    "STOP": 2,
    "DATA_0": 3,
    "DATA_1": 4,
    "PARITY_0": 5,
    "PARITY_1": 6,
    "GLITCH": 7,  # 噪声/毛刺
    "PARITY_ERROR": 8,  # 校验位错误
    "FRAMING_ERROR": 9,  # 帧错误（停止位缺失）
    "BREAK": 10  # Break 信号
}


def get_label_name(label_id):
    for name, id_ in LABEL_MAP.items():
        if id_ == label_id:
            return name
    return "UNKNOWN"


# --- 默认配置 (增加了异常概率) ---
DEFAULT_UART_CONFIG = {
    'voltage_high': 3.3,
    'voltage_low': 0.0,
    'voltage_noise_std': 0.05,  # 增加一点底噪
    'jitter_std_factor': 0.03,  # 增加一点时钟抖动
    'rise_time_factor': 0.05,
    'fall_time_factor': 0.05,

    # 基础协议参数
    'prob_parity_none': 0.3, 'prob_parity_odd': 0.35, 'prob_parity_even': 0.35,  # 增加校验位出现的概率
    'prob_stop_1': 0.8, 'prob_stop_2': 0.2,
    'data_bits': 8,

    # --- 异常通信概率分布 (模拟现实中的恶劣环境) ---
    'prob_parity_error': 0.00,  # 5% 的概率发生校验错误
    'prob_framing_error': 0.00,  # 5% 的概率发生帧错误
    'prob_glitch': 0.00,  # 10% 的概率出现随机毛刺
    'prob_break': 0.00,  # 2% 的概率出现 Break 信号
    'prob_baud_drift': 0.00,  # 10% 的概率出现波特率严重漂移

    'idle_before_min': 1, 'idle_before_max': 10,
    'inter_byte_idle_min': 0, 'inter_byte_idle_max': 5,
    'min_bytes': 2, 'max_bytes': 15,  # 增加每帧字节数，增加出错机会

    # 波特率选项
    'baud_rate_options': {
        9600: 1, 
        #19200: 0.1, 38400: 0.1, 57600: 0.1,
        #115200: 0.4, 230400: 0.1, 921600: 0.1
    },

    # 采样率
    'sampling_rate_options': [
        #100e3, 200e3, 500e3,
        1e6,
        #2e6, 5e6, 10e6, 20e6, 50e6, 100e6
    ]
}


class RealisticUARTSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.voltage_high = config['voltage_high']
        self.voltage_low = config['voltage_low']
        self.voltage_noise_std = config.get('voltage_noise_std', 0.03)
        self.jitter_std_factor = config.get('jitter_std_factor', 0.02)

        self.baud_rate = None
        self.sampling_rate = None
        self.samples_per_bit = None
        self.parity = 'NONE'
        self.stop_bits = 1
        self.data_bits = 8

    def _select_parameters(self):
        # 1. 选择波特率
        br_opts = list(self.config['baud_rate_options'].keys())
        br_probs = list(self.config['baud_rate_options'].values())
        self.baud_rate = np.random.choice(br_opts, p=br_probs)

        min_sr = 40 * self.baud_rate
        max_sr = 200 * self.baud_rate
        # 2. 选择采样率
        valid_srs = [
            sr for sr in self.config['sampling_rate_options'] 
            if min_sr <= sr <= max_sr
        ]

        if not valid_srs:
            # 兜底逻辑：如果区间内没有选项
            # 优先找最接近下限的（保证清晰度），如果没有大的就找最接近上限的
            all_srs = sorted(self.config['sampling_rate_options'])
            # 找到第一个大于 min_sr 的
            candidates = [sr for sr in all_srs if sr >= min_sr]
            if candidates:
                self.sampling_rate = candidates[0] # 选最小的满足清晰度的
            else:
                self.sampling_rate = all_srs[-1] # 实在没办法，选最大的
                
            # print(f"Warning: No optimal sampling rate for baud {self.baud_rate}. Fallback to {self.sampling_rate}")
        else:
            # 在甜蜜点范围内随机选一个，增加数据多样性
            self.sampling_rate = np.random.choice(valid_srs)

        self.samples_per_bit = int(self.sampling_rate / self.baud_rate)

        # 3. 选择协议格式
        p_rand = np.random.rand()
        if p_rand < self.config['prob_parity_none']:
            self.parity = 'NONE'
        elif p_rand < self.config['prob_parity_none'] + self.config['prob_parity_odd']:
            self.parity = 'ODD'
        else:
            self.parity = 'EVEN'

        self.stop_bits = 2 if np.random.rand() < self.config['prob_stop_2'] else 1
        self.data_bits = self.config.get('data_bits', 8)

    def _get_samples_for_current_bit(self, drift_factor=1.0):
        # 模拟时钟抖动 + 系统性漂移
        # drift_factor != 1.0 时模拟波特率偏差
        factor = drift_factor + np.random.normal(0, self.jitter_std_factor)
        return max(1, int(self.samples_per_bit * factor))

    def add_noise(self, signal):
        return signal + np.random.normal(0, self.voltage_noise_std, len(signal))

    def add_transition_time(self, signal):
        # 简单的上升沿/下降沿模拟
        result = signal.copy()
        rise = int(self.config['rise_time_factor'] * self.samples_per_bit)
        fall = int(self.config['fall_time_factor'] * self.samples_per_bit)
        edges = np.where(np.diff(signal) != 0)[0]
        for e in edges:
            if e + 1 < len(signal):
                if signal[e] < signal[e + 1] and rise > 0:  # Rise
                    end = min(e + rise, len(signal) - 1)
                    if end > e: result[e:end] = np.linspace(signal[e], signal[end], end - e)
                elif signal[e] > signal[e + 1] and fall > 0:  # Fall
                    end = min(e + fall, len(signal) - 1)
                    if end > e: result[e:end] = np.linspace(signal[e], signal[end], end - e)
        return result

    def _generate_bit_waveform(self, level, label_id, drift=1.0):
        n = self._get_samples_for_current_bit(drift)
        voltage = self.voltage_high if level else self.voltage_low
        return np.full(n, voltage), np.full(n, label_id)

    # --- 生成毛刺 (Glitch) ---
    def _generate_glitch(self):
        # 毛刺是一个极短的脉冲，通常小于 1/4 个 bit
        glitch_len = max(1, int(self.samples_per_bit * np.random.uniform(0.05, 0.2)))
        # 随机决定是向上还是向下的毛刺
        level = np.random.choice([0, 1])
        voltage = self.voltage_high if level else self.voltage_low
        return np.full(glitch_len, voltage), np.full(glitch_len, LABEL_MAP['GLITCH'])

    # --- 生成 Break 信号 ---
    def _generate_break(self):
        # Break 是指总线被拉低超过一帧的时间（Start + Data + Parity + Stop）
        # 这里模拟 12 到 20 个 bit 长度的低电平
        break_bits = np.random.randint(12, 20)
        total_len = int(break_bits * self.samples_per_bit)
        return np.full(total_len, self.voltage_low), np.full(total_len, LABEL_MAP['BREAK'])

    def _generate_uart_byte(self, value):
        seq_wave = []
        seq_label = []
        byte_events = []

        # 模拟波特率漂移：如果是漂移模式，这一字节的所有位都会略长或略短
        drift = 1.0
        if np.random.rand() < self.config['prob_baud_drift']:
            drift = np.random.uniform(0.9, 1.1)  # +/- 10% 的严重漂移

        # 1. Start Bit
        w, l = self._generate_bit_waveform(0, LABEL_MAP['START'], drift)
        seq_wave.append(w);
        seq_label.append(l)

        # 2. Data Bits
        ones_count = 0
        for i in range(self.data_bits):
            bit = (value >> i) & 1
            if bit: ones_count += 1
            lbl = LABEL_MAP['DATA_1'] if bit else LABEL_MAP['DATA_0']
            w, l = self._generate_bit_waveform(bit, lbl, drift)
            seq_wave.append(w);
            seq_label.append(l)

        # 3. Parity Bit (含异常注入)
        if self.parity != 'NONE':
            if self.parity == 'ODD':
                p_bit = 1 if (ones_count % 2 == 0) else 0
            else:  # EVEN
                p_bit = 1 if (ones_count % 2 != 0) else 0

            # --- 异常 1: 校验错误 ---
            is_parity_error = False
            if np.random.rand() < self.config['prob_parity_error']:
                p_bit = 1 - p_bit  # 翻转校验位
                is_parity_error = True
                byte_events.append({'type': 'ERROR', 'subtype': 'PARITY_ERROR', 'expected': 1 - p_bit, 'actual': p_bit})

            lbl = LABEL_MAP['PARITY_ERROR'] if is_parity_error else (
                LABEL_MAP['PARITY_1'] if p_bit else LABEL_MAP['PARITY_0'])
            w, l = self._generate_bit_waveform(p_bit, lbl, drift)
            seq_wave.append(w);
            seq_label.append(l)

        # 4. Stop Bit (含异常注入)
        stop_count = int(self.stop_bits)

        # --- 异常 2: 帧错误 (Framing Error) ---
        # 帧错误通常指应该为高电平的停止位变成了低电平
        is_framing_error = False
        if np.random.rand() < self.config['prob_framing_error']:
            is_framing_error = True
            byte_events.append({'type': 'ERROR', 'subtype': 'FRAMING_ERROR'})

        for _ in range(stop_count):
            if is_framing_error:
                # 强制拉低停止位
                w, l = self._generate_bit_waveform(0, LABEL_MAP['FRAMING_ERROR'], drift)
            else:
                # 正常停止位
                w, l = self._generate_bit_waveform(1, LABEL_MAP['STOP'], drift)
            seq_wave.append(w);
            seq_label.append(l)

        return np.concatenate(seq_wave), np.concatenate(seq_label), byte_events

    def generate_uart_transaction(self):
        self._select_parameters()

        wave_seq = []
        label_seq = []
        events = []

        events.append({
            'type': 'CONFIG',
            'baud': self.baud_rate,
            'parity': self.parity,
            'stop': self.stop_bits
        })

        # 1. 前置空闲
        idle_bits = np.random.randint(self.config['idle_before_min'], self.config['idle_before_max'])
        w_idle, l_idle = self._generate_bit_waveform(1, LABEL_MAP['IDLE'])
        for _ in range(idle_bits):
            wave_seq.append(w_idle);
            label_seq.append(l_idle)

        # 2. 生成传输内容
        num_bytes = np.random.randint(self.config['min_bytes'], self.config['max_bytes'] + 1)

        for i in range(num_bytes):
            # --- 异常 3: Break 信号 ---
            if np.random.rand() < self.config['prob_break']:
                w_bk, l_bk = self._generate_break()
                wave_seq.append(w_bk)
                label_seq.append(l_bk)
                events.append({'type': 'BREAK'})
                # Break 之后通常需要一段空闲恢复
                wave_seq.append(w_idle);
                label_seq.append(l_idle)
                continue  # 跳过这次字节生成

            # --- 异常 4: 随机毛刺 (Glitch) ---
            if np.random.rand() < self.config['prob_glitch']:
                w_gl, l_gl = self._generate_glitch()
                wave_seq.append(w_gl)
                label_seq.append(l_gl)
                # 不记录 Glitch 事件，因为它是噪声，模型最好能学会忽略它或标记为 GLITCH

            # 生成正常(或包含Parity/Frame错误)的字节
            val = np.random.randint(0, 2 ** self.data_bits)
            w_byte, l_byte, byte_evs = self._generate_uart_byte(val)

            # 记录数据事件
            if not byte_evs:  # 没有错误
                events.append({'type': 'DATA', 'value': val})
            else:
                for err in byte_evs: events.append(err)  # 记录具体错误
                events.append({'type': 'DATA_WITH_ERROR', 'value': val})

            wave_seq.append(w_byte)
            label_seq.append(l_byte)

            # 字节间空闲
            if i < num_bytes - 1:
                gaps = np.random.randint(self.config['inter_byte_idle_min'], self.config['inter_byte_idle_max'] + 1)
                for _ in range(gaps):
                    wave_seq.append(w_idle);
                    label_seq.append(l_idle)

        # 3. 后置空闲
        for _ in range(3):
            wave_seq.append(w_idle);
            label_seq.append(l_idle)

        raw_wave = np.concatenate(wave_seq)
        final_wave = self.add_noise(self.add_transition_time(raw_wave))
        final_labels = np.concatenate(label_seq)

        return final_wave, final_labels, events

    def generate_uart_datasets(self, num_datasets=100, samples_per_dataset=10000):
        # 4通道格式 (TX在通道0)
        all_datasets_np = np.zeros((num_datasets, samples_per_dataset, 4), dtype=np.float32)
        all_labels_np = np.zeros((num_datasets, samples_per_dataset), dtype=np.int64)
        all_events = []
        all_channel_maps = []

        for i in range(num_datasets):
            tx_raw, labels_raw, events = self.generate_uart_transaction()

            tx_ch = 0
            all_channel_maps.append((tx_ch,))

            # 背景噪声
            final_waveform_4ch = np.full((samples_per_dataset, 4), self.voltage_high, dtype=np.float32)
            final_waveform_4ch += np.random.normal(0, self.voltage_noise_std / 2, final_waveform_4ch.shape)

            L = min(len(tx_raw), samples_per_dataset)
            final_waveform_4ch[:L, tx_ch] = tx_raw[:L]

            final_labels = np.full(samples_per_dataset, LABEL_MAP['IDLE'], dtype=np.int64)
            final_labels[:L] = labels_raw[:L]

            all_datasets_np[i] = final_waveform_4ch
            all_labels_np[i] = final_labels
            all_events.append(events)

        return all_datasets_np, all_labels_np, all_events, all_channel_maps

    # (plot_signals 和 save_dataset 函数与之前相同，此处省略以节省空间，可直接复用)
    # ... 请确保保留之前代码中的 save_dataset 和 plot_signals ...
    def save_dataset(self, wave, labels, events, base_dir, prefix="uart"):
        os.makedirs(base_dir, exist_ok=True)
        # 文件编号逻辑
        existing_files = [f for f in os.listdir(base_dir) if f.startswith(f"{prefix}-") and f.endswith(".csv")]
        next_num = 1
        if existing_files:
            nums = [int(f.split('-')[-1].split('.')[0]) for f in existing_files if
                    f.split('-')[-1].split('.')[0].isdigit()]
            if nums: next_num = max(nums) + 1

        base_filename = f"{prefix}-{next_num:04d}"
        csv_path = os.path.join(base_dir, f"{base_filename}.csv")
        txt_path = os.path.join(base_dir, f"{base_filename}.txt")

        # 保存 CSV (Time, TX, Label)
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time_Index', 'TX', 'Label'])
            for i in range(len(wave)):
                writer.writerow([i, wave[i], int(labels[i])])
        print(f"Saved waveform to: {csv_path}")

        # 保存事件 TXT
        with open(txt_path, 'w') as txtfile:
            for event in events:
                if event['type'] == 'CONFIG':
                    txtfile.write(f"CONFIG: Baud={event['baud']}, Parity={event['parity']}, Stop={event['stop']}\n")
                elif event['type'] == 'DATA':
                    txtfile.write(f"DATA: 0x{event['value']:02X}\n")
                elif event['type'] == 'ERROR':
                    txtfile.write(f"ERROR: {event['subtype']}\n")
                else:
                    txtfile.write(f"{event['type']}\n")
        print(f"Saved events to: {txt_path}")

    def plot_signals(self, wave, labels=None, title="UART Signal", plot_samples=None):
        if plot_samples is None: plot_samples = len(wave)
        plot_samples = min(plot_samples, len(wave))
        t = np.arange(plot_samples) / self.sampling_rate * 1e6

        fig, ax = plt.subplots(figsize=(20, 6))
        ax.plot(t, wave[:plot_samples], 'b-', label='TX', alpha=0.8)

        if labels is not None:
            colors = plt.get_cmap('tab20', len(LABEL_MAP))  # 使用tab20以支持更多颜色
            unique_labels = sorted(list(set(labels[:plot_samples])))
            for lbl_id in unique_labels:
                if lbl_id == LABEL_MAP['IDLE']: continue
                ax.plot([], [], color=colors(lbl_id), label=f"{int(lbl_id)}:{get_label_name(lbl_id)}", linewidth=5,
                        alpha=0.5)

            last_lbl = -1
            start_idx = 0
            for i in range(plot_samples):
                curr = labels[i]
                if curr != last_lbl:
                    if last_lbl != -1 and last_lbl != LABEL_MAP['IDLE']:
                        ax.axvspan(start_idx / self.sampling_rate * 1e6, i / self.sampling_rate * 1e6,
                                   color=colors(last_lbl), alpha=0.3)
                    last_lbl = curr
                    start_idx = i

        ax.set_xlabel('Time (µs)')
        ax.set_ylabel('Voltage (V)')
        ax.set_title(f"{title} (Baud: {self.baud_rate})")
        ax.legend(loc='upper right')
        ax.grid(True, linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    output_dir = "../dataset/uart_generated"
    gen = RealisticUARTSignalGenerator(config=DEFAULT_UART_CONFIG)

    print("Generating UART dataset with Anomalies...")
    # 生成带异常的数据
    all_data, all_labels, all_events, all_maps = gen.generate_uart_datasets(num_datasets=5, samples_per_dataset=10000)

    idx = 0
    tx_ch = all_maps[idx][0]

    gen.plot_signals(all_data[idx][:, tx_ch], all_labels[idx], title="Generated UART (with Error Injection)")




