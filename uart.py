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
    # 'voltage_noise_std': 0.05,  # 增加一点底噪
    # 'jitter_std_factor': 0.03,  # 增加一点时钟抖动
    # 'rise_time_factor': 0.05,
    # 'fall_time_factor': 0.05,
    'voltage_noise_std': 0.00,  # 增加一点底噪
    'jitter_std_factor': 0.00,  # 增加一点时钟抖动
    'rise_time_factor': 0.00,
    'fall_time_factor': 0.00,

    # 基础协议参数
    'prob_parity_none': 0.3, 'prob_parity_odd': 0.35, 'prob_parity_even': 0.35,  # 增加校验位出现的概率
    'prob_stop_1': 0.8, 'prob_stop_2': 0.2,
    'data_bits': 8,

    # --- 异常通信概率分布 (模拟现实中的恶劣环境) ---
    # 'prob_parity_error': 0.03,  # 5% 的概率发生校验错误
    # 'prob_framing_error': 0.03,  # 5% 的概率发生帧错误
    # 'prob_glitch': 0.05,  # 10% 的概率出现随机毛刺
    # 'prob_break': 0.01,  # 2% 的概率出现 Break 信号
    # 'prob_baud_drift': 0.05,  # 10% 的概率出现波特率严重漂移
    'prob_parity_error': 0.00,  # 5% 的概率发生校验错误
    'prob_framing_error': 0.00,  # 5% 的概率发生帧错误
    'prob_glitch': 0.00,  # 10% 的概率出现随机毛刺
    'prob_break': 0.00,  # 2% 的概率出现 Break 信号
    'prob_baud_drift': 0.00,  # 10% 的概率出现波特率严重漂移

    'idle_before_min': 1, 'idle_before_max': 10,
    'inter_byte_idle_min': 0, 'inter_byte_idle_max': 5,
    'min_bytes': 2, 'max_bytes': 15,  # 增加每帧字节数，增加出错机会
    #过采样率范围
    'min_oversampling': 16,
    'max_oversampling': 64,

    # 波特率选项
    'baud_rate_options': {
        9600: 0.1, 19200: 0.1, 38400: 0.1, 57600: 0.1,
        115200: 0.4, 230400: 0.1, 921600: 0.1
    },

    # 采样率
    'sampling_rate_options': [
        100e3, 200e3, 500e3,
        1e6, 2e6, 5e6, 10e6, 20e6, 50e6, 100e6
    ]
}


class RealisticUARTSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.voltage_high = config['voltage_high']
        self.voltage_low = config['voltage_low']
        # self.voltage_noise_std = config.get('voltage_noise_std', 0.03)
        # self.jitter_std_factor = config.get('jitter_std_factor', 0.02)
        self.voltage_noise_std = config.get('voltage_noise_std', 0.00)
        self.jitter_std_factor = config.get('jitter_std_factor', 0.00)

        self.baud_rate = None
        self.sampling_rate = None
        self.samples_per_bit = None
        self.parity = 'NONE'
        self.stop_bits = 1
        self.data_bits = 8

    def _select_parameters(self, max_samples=None):
        # ---------- 1) 先选波特率 ----------
        br_opts = list(self.config['baud_rate_options'].keys())
        br_probs = list(self.config['baud_rate_options'].values())
        self.baud_rate = int(np.random.choice(br_opts, p=br_probs))

        # ---------- 2) 选校验 ----------
        p_rand = np.random.rand()
        if p_rand < self.config['prob_parity_none']:
            self.parity = 'NONE'
        elif p_rand < self.config['prob_parity_none'] + self.config['prob_parity_odd']:
            self.parity = 'ODD'
        else:
            self.parity = 'EVEN'

        # ---------- 3) 选停止位 ----------
        self.stop_bits = 2 if np.random.rand() < self.config['prob_stop_2'] else 1

        # ---------- 4) 数据位 ----------
        self.data_bits = int(self.config.get('data_bits', 8))

        # ---------- 5) 计算一个字节最少占多少 bit ----------
        parity_bits = 0 if self.parity == 'NONE' else 1
        bits_per_byte = 1 + self.data_bits + parity_bits + int(self.stop_bits)
        #                起始位 + 数据位 + 校验位 + 停止位

        # ---------- 6) 先按过采样倍数筛选采样率 ----------
        min_oversampling = int(self.config.get('min_oversampling', 16))
        max_oversampling = int(self.config.get('max_oversampling', 64))

        min_sr = min_oversampling * self.baud_rate
        max_sr = max_oversampling * self.baud_rate

        sr_candidates = [
            int(sr) for sr in self.config['sampling_rate_options']
            if min_sr <= sr <= max_sr
        ]

        # ---------- 7) 再按 max_samples 做预算约束 ----------
        if max_samples is not None:
            min_bytes = int(self.config.get('min_bytes', 1))
            idle_before_min = int(self.config.get('idle_before_min', 1))
            inter_byte_idle_min = int(self.config.get('inter_byte_idle_min', 0))
            tail_idle_bits = 3

            min_total_bits = (
                    idle_before_min
                    + min_bytes * bits_per_byte
                    + max(0, min_bytes - 1) * inter_byte_idle_min
                    + tail_idle_bits
            )

            max_sr_by_budget = int((max_samples * self.baud_rate) / min_total_bits)

            sr_candidates = [sr for sr in sr_candidates if sr <= max_sr_by_budget]

        # ---------- 8) 没有合法采样率就报错 ----------
        if not sr_candidates:
            raise ValueError(
                f"没有合法的采样率候选。"
                f" baud={self.baud_rate}, parity={self.parity}, stop={self.stop_bits}, "
                f"data_bits={self.data_bits}, max_samples={max_samples}, "
                f"oversampling=[{min_oversampling}, {max_oversampling}]"
            )
        # ---------- 9) 从合法候选里选一个 ----------
        self.sampling_rate = int(np.random.choice(sr_candidates))
        # samples_per_bit 至少保证为 1
        self.samples_per_bit = max(1, int(self.sampling_rate / self.baud_rate))

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

    def _bits_per_byte(self):
        parity_bits = 0 if self.parity == 'NONE' else 1
        return 1 + self.data_bits + parity_bits + int(self.stop_bits)

    def generate_uart_transaction(self, max_samples=None):
        self._select_parameters(max_samples=max_samples)

        wave_seq = []
        label_seq = []
        events = []
        curr_len = 0

        def can_fit(n_more):
            return (max_samples is None) or (curr_len + n_more <= max_samples)

        def append_seg(w, l):
            nonlocal curr_len
            if not can_fit(len(w)):
                return False
            wave_seq.append(w)
            label_seq.append(l)
            curr_len += len(w)
            return True

        events.append({
            'type': 'CONFIG',
            'baud': self.baud_rate,
            'sampling_rate': self.sampling_rate,
            'parity': self.parity,
            'stop': self.stop_bits,
            'data_bits': self.data_bits,
        })

        # 前置空闲
        idle_bits = np.random.randint(self.config['idle_before_min'], self.config['idle_before_max'])
        w_idle, l_idle = self._generate_bit_waveform(1, LABEL_MAP['IDLE'])

        # 预留尾部 3bit 空闲
        tail_reserve = 3 * len(w_idle)

        for _ in range(idle_bits):
            if max_samples is not None and curr_len + len(w_idle) + tail_reserve > max_samples:
                break
            append_seg(w_idle, l_idle)

        num_bytes_target = np.random.randint(self.config['min_bytes'], self.config['max_bytes'] + 1)
        bytes_generated = 0

        while bytes_generated < num_bytes_target:
            # 先尝试生成一个字节
            val = np.random.randint(0, 2 ** self.data_bits)
            w_byte, l_byte, byte_evs = self._generate_uart_byte(val)

            # 估算最小附加成本：当前字节 + 尾部空闲
            need = len(w_byte) + tail_reserve

            # 如果不是最后一个字节，至少还可能有 gap
            if bytes_generated < num_bytes_target - 1:
                gap_bits = np.random.randint(
                    self.config['inter_byte_idle_min'],
                    self.config['inter_byte_idle_max'] + 1
                )
                need += gap_bits * len(w_idle)
            else:
                gap_bits = 0

            if max_samples is not None and curr_len + need > max_samples:
                break

            if not byte_evs:
                events.append({'type': 'DATA', 'value': val})
            else:
                for err in byte_evs:
                    events.append(err)
                events.append({'type': 'DATA_WITH_ERROR', 'value': val})

            append_seg(w_byte, l_byte)
            bytes_generated += 1

            for _ in range(gap_bits):
                append_seg(w_idle, l_idle)

        # 如果连 min_bytes 都放不下，直接报错，说明参数组合不合法
        if bytes_generated < self.config['min_bytes']:
            raise ValueError(
                f"当前参数组合在 max_samples={max_samples} 下无法容纳最少 {self.config['min_bytes']} 个字节"
            )

        # 后置空闲
        for _ in range(3):
            if not append_seg(w_idle, l_idle):
                break

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
            tx_raw, labels_raw, events = self.generate_uart_transaction(max_samples=samples_per_dataset)

            tx_ch = 0
            all_channel_maps.append((tx_ch,))

            # 背景噪声
            final_waveform_4ch = np.full((samples_per_dataset, 4), self.voltage_high, dtype=np.float32)
            final_waveform_4ch += np.random.normal(0, self.voltage_noise_std / 2, final_waveform_4ch.shape)

            L = len(tx_raw)
            if L > samples_per_dataset:
                raise RuntimeError(f"生成结果长度 {L} 超过预算 {samples_per_dataset}")
            final_waveform_4ch[:L, tx_ch] = tx_raw

            final_labels = np.full(samples_per_dataset, LABEL_MAP['IDLE'], dtype=np.int64)
            final_labels[:L] = labels_raw

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
            writer.writerow(['Time_s', 'TX', 'Label'])
            for i in range(len(wave)):
                writer.writerow([i / self.sampling_rate, wave[i], int(labels[i])])
        print(f"Saved waveform to: {csv_path}")

        # 保存事件 TXT
        with open(txt_path, 'w', encoding='utf-8') as txtfile:
            config_event = next((e for e in events if e['type'] == 'CONFIG'), None)

            if config_event is not None:
                txtfile.write(
                    f"CONFIG: Baud={config_event['baud']}, "
                    f"SamplingRate={config_event['sampling_rate']}, "
                    f"DataBits={config_event['data_bits']}, "
                    f"Parity={config_event['parity']}, "
                    f"Stop={config_event['stop']}\n"
                )

            for event in events:
                if event['type'] == 'DATA':
                    txtfile.write(f"DATA: 0x{event['value']:02X}\n")
                elif event['type'] == 'DATA_WITH_ERROR':
                    txtfile.write(f"DATA_WITH_ERROR: 0x{event['value']:02X}\n")
                elif event['type'] == 'BREAK':
                    txtfile.write("BREAK\n")
                elif event['type'] == 'ERROR':
                    txtfile.write(f"ERROR: {event['error_type']}\n")

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

    # ========== 新增：保存数据到指定文件夹 ==========
    # 遍历生成的5组数据，逐个保存
    for idx in range(len(all_data)):
        tx_ch = all_maps[idx][0]
        # 提取当前组的TX波形和标签
        tx_wave = all_data[idx][:, tx_ch]
        tx_labels = all_labels[idx]
        tx_events = all_events[idx]

        # 调用save_dataset保存（会自动创建output_dir文件夹）
        gen.save_dataset(
            wave=tx_wave,
            labels=tx_labels,
            events=tx_events,
            base_dir=output_dir,  # 保存到指定目录
            prefix="uart"  # 文件名前缀，可选
        )
    # ==============================================
    idx = 0
    tx_ch = all_maps[idx][0]
    gen.plot_signals(all_data[idx][:, tx_ch], all_labels[idx], title="Generated UART (with Error Injection)")

