import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# 定义标签字典 (无变化)
LABEL_MAP = {
    "IDLE": 0, "START": 1, "STOP": 2, "REPEATED_START": 3,
    "ADDR_7_BIT_0": 4, "ADDR_7_BIT_1": 5, "RW_BIT_READ": 6, "RW_BIT_WRITE": 7,
    "ADDR_10_HEAD_0": 8, "ADDR_10_HEAD_1": 9, "ADDR_10_LOW_0": 10, "ADDR_10_LOW_1": 11,
    "DATA_BIT_0": 12, "DATA_BIT_1": 13, "ACK": 14, "NACK": 15,
}

# I2C常见信号速率
# 标准模式（Standard Mode）‌：速率约为100kbps（100kbit / s）‌12。
# 快速模式（FastMode）‌：速率约为400kbps（400kbit / s）‌
# 快速 + 模式（Fast - PlusMode）‌：速率约为1Mbps（1Mbit / s）‌
# 高速模式（High - SpeedMode）‌：速率约为3.4Mbps（3.4Mbit / s）‌
# 超高速模式（Ultra - FastMode）‌：速率约为5Mbps（5Mbit / s，单向传输）‌

# 默认配置 (无变化)
DEFAULT_I2C_CONFIG = {
    'voltage_high': 3.3, 'voltage_low': 0.0,
    'voltage_noise_std': 0.03, 'jitter_std_factor': 0.02,
    'rise_time_factor': 0.05, 'fall_time_factor': 0.05,
    'prob_write': 0.4, 'prob_read': 0.4, 'prob_write_read': 0.2,
    'prob_10bit_addr': 0.2, 'prob_addr_nack': 0.1, 'prob_data_nack': 0.1,
    'length_jitter_prob': 0.05, 'length_jitter_range': 0.1,
    'idle_bits_min': 1, 'idle_bits_max': 10,'swap_channels_prob': 0.5,
    'max_write_bytes': 9, 'max_read_bytes': 9,
    'scl_freq_options': {
        100e3: 0.4,  # 标准模式 100kbps，40%概率
        400e3: 0.2,  # 快速模式 400kbps，30%概率
        1e6: 0.1,  # 快速+模式 1Mbps，15%概率
        3.4e6: 0.2,  # 高速模式 3.4Mbps，10%概率
        5e6: 0.1,  # 超高速模式 5Mbps，5%概率
    },
    # 添加可用的采样率选项
    'sampling_rate_options': [
        # 10e3, 20e3, 50e3, 100e3, 200e3, 500e3,1e6,
        2e6, 5e6, 10e6, 20e6, 50e6,100e6, 200e6, 500e6, 1e9]
}


def get_label_name(label_id):
    for name, id_ in LABEL_MAP.items():
        if id_ == label_id:
            return name
    return "UNKNOWN"


class RealisticI2CSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.idle_bits_min = config.get('idle_bits_min', 1)
        self.idle_bits_max = config.get('idle_bits_max', 10)
        self.voltage_high = config['voltage_high']
        self.voltage_low = config['voltage_low']
        self.voltage_noise_std = config.get('voltage_noise_std', 0.03)
        self.jitter_std_factor = config.get('jitter_std_factor', 0.04)
        self.rise_time_factor = config.get('rise_time_factor', 0.2)
        self.fall_time_factor = config.get('fall_time_factor', 0.2)
        self.length_jitter_prob = config.get('length_jitter_prob', 0.1)
        self.length_jitter_range = config.get('length_jitter_range', 0.1)

        self.sampling_rate = None
        self.scl_freq = None
        self.base_samples_per_bit = None

    def _select_frequencies(self):
        scl_options = list(self.config['scl_freq_options'].keys())
        scl_probs = list(self.config['scl_freq_options'].values())
        self.scl_freq = np.random.choice(scl_options, p=scl_probs)
        # 选择满足条件的最小采样率（采样率/SCL频率 > 50）
        min_rate = 50 * self.scl_freq
        valid_rates = [r for r in self.config['sampling_rate_options'] if r >= min_rate]
        if not valid_rates:
            # 如果没有满足条件的采样率，则选择可用的最大采样率
            self.sampling_rate = max(self.config['sampling_rate_options'])
            print(f"Warning: No valid sampling rate for SCL={self.scl_freq / 1e3:.1f}kHz. "
                  f"Using max available: {self.sampling_rate / 1e6:.1f}MHz")
        else:
            # 选择最小的满足条件的采样率（减少数据量）
            self.sampling_rate = min(valid_rates)
        # 更新每个bit的基础样本数
        self.base_samples_per_bit = int(self.sampling_rate / self.scl_freq)
        return self.scl_freq, self.sampling_rate


    def _get_samples_per_bit(self):
        n = self.base_samples_per_bit
        if np.random.rand() < self.length_jitter_prob:
            factor = 1 + np.random.uniform(-self.length_jitter_range, self.length_jitter_range)
            return max(1, int(n * factor))
        return n

    def add_noise(self, signal):
        return signal + np.random.normal(0, self.voltage_noise_std, len(signal))

    def add_transition_time(self, signal):
        result = signal.copy()
        rise = int(self.rise_time_factor * self.base_samples_per_bit)
        fall = int(self.fall_time_factor * self.base_samples_per_bit)
        edges = np.where(np.diff(signal) != 0)[0]
        for e in edges:
            if e + 1 < len(signal):
                if signal[e] < signal[e + 1] and rise > 0:
                    end = min(e + rise, len(signal) - 1)
                    if end > e: result[e:end] = np.linspace(signal[e], signal[end], end - e)
                elif signal[e] > signal[e + 1] and fall > 0:
                    end = min(e + fall, len(signal) - 1)
                    if end > e: result[e:end] = np.linspace(signal[e], signal[end], end - e)
        return result

    def _generate_bit(self, bit_value, prev_sda_level, lbl0, lbl1):
        spb = self._get_samples_per_bit()
        scl = np.ones(spb) * self.voltage_low
        scl[int(spb / 2):] = self.voltage_high
        current_sda_level = self.voltage_high if bit_value else self.voltage_low
        sda = np.full(spb, current_sda_level)
        if current_sda_level != prev_sda_level:
            transition_point = int(spb * 0.25)
            sda[:transition_point] = prev_sda_level
            sda[transition_point:] = current_sda_level
        label = lbl1 if bit_value else lbl0
        labels = np.full(spb, label)
        return scl, sda, labels, current_sda_level

    # --- MODIFIED: 底层函数现在返回事件信息 ---
    def _generate_byte(self, val, prev_sda_level, lbl0, lbl1, rw_bit_label=None):
        S, D, L = [], [], []
        current_sda_level = prev_sda_level
        for i in range(7, -1, -1):
            bit = (val >> i) & 1
            if i == 0 and rw_bit_label is not None:
                b0 = b1 = rw_bit_label
            else:
                b0, b1 = lbl0, lbl1
            s, d, l, current_sda_level = self._generate_bit(bit, current_sda_level, b0, b1)
            S.append(s);
            D.append(d);
            L.append(l)

        # 返回事件信息：字节的值
        event = {'value': val}
        return np.concatenate(S), np.concatenate(D), np.concatenate(L), current_sda_level, event

    def _generate_ack_nack(self, prev_sda_level, simulate_nack=False):
        bit = 1 if simulate_nack else 0
        s, d, l, level = self._generate_bit(bit, prev_sda_level, LABEL_MAP['ACK'], LABEL_MAP['NACK'])
        # 返回事件信息：是ACK还是NACK
        event = {'type': 'NACK'} if simulate_nack else {'type': 'ACK'}
        return s, d, l, level, event

    def _generate_start(self, rep=False):
        spb = self._get_samples_per_bit()
        scl = np.ones(spb) * self.voltage_high
        mid = int(spb / 2 + np.random.normal(0, self.jitter_std_factor * spb))
        mid = np.clip(mid, 0, spb)
        sda = np.ones(spb) * self.voltage_high
        sda[mid:] = self.voltage_low
        lbl_name = 'REPEATED_START' if rep else 'START'
        lbl_id = LABEL_MAP[lbl_name]
        event = {'type': lbl_name}
        return scl, sda, np.full(spb, lbl_id), self.voltage_low, event

    def _generate_stop(self):
        spb = self._get_samples_per_bit()
        scl = np.ones(spb) * self.voltage_high
        mid = int(spb / 2 + np.random.normal(0, self.jitter_std_factor * spb))
        mid = np.clip(mid, 0, spb)
        sda = np.full(spb, self.voltage_low)
        sda[mid:] = self.voltage_high
        event = {'type': 'STOP'}
        return scl, sda, np.full(spb, LABEL_MAP['STOP']), self.voltage_high, event

    def _generate_idle_frames(self):
        scl_seq, sda_seq, lbl_seq = [], [], []
        idle_bits = np.random.randint(self.idle_bits_min, self.idle_bits_max + 1)
        for _ in range(idle_bits):
            spb = self._get_samples_per_bit()
            scl_seq.append(np.full(spb, self.voltage_high))
            sda_seq.append(np.full(spb, self.voltage_high))
            lbl_seq.append(np.full(spb, LABEL_MAP['IDLE']))
        # IDLE帧不产生事件
        return np.concatenate(scl_seq), np.concatenate(sda_seq), np.concatenate(lbl_seq)

    # --- MODIFIED: generate_i2c_transaction 现在是总指挥 ---
    def generate_i2c_transaction(self):
        scl, sr = self._select_frequencies()
        print(f"Selected SCL={scl/1e3:.1f}kHz, sampling_rate={sr/1e6:.1f}MHz,base_samples_per_bit={sr/scl:.1f}")
        scl_seq, sda_seq, lbl_seq, events = [], [], [], []

        op_rand = np.random.rand()
        if op_rand < self.config['prob_write']:
            op_type = 'WRITE'
        elif op_rand < self.config['prob_write'] + self.config['prob_read']:
            op_type = 'READ'
        else:
            op_type = 'WRITE_READ'
        addr_bits = 10 if np.random.rand() < self.config['prob_10bit_addr'] else 7
        num_write = np.random.randint(1, self.config.get('max_write_bytes', 10) + 1)
        num_read = np.random.randint(1, self.config.get('max_read_bytes', 10) + 1)

        # 如果是混合读写，就把写和读都限制到最多 5 字节
        if op_type == 'WRITE_READ':
            num_write = min(num_write, 4)
            num_read = min(num_read, 4)

        current_sda_level = self.voltage_high

        s, d, l = self._generate_idle_frames()
        scl_seq.append(s);
        sda_seq.append(d);
        lbl_seq.append(l)

        s, d, l, current_sda_level, ev = self._generate_start()
        scl_seq.append(s);
        sda_seq.append(d);
        lbl_seq.append(l);
        events.append(ev)

        addr_nacked = False
        slave_addr_for_read = 0

        if addr_bits == 7:
            slave = np.random.randint(0x08, 0x77)
            slave_addr_for_read = slave
            rw = 0 if op_type in ['WRITE', 'WRITE_READ'] else 1
            addr_byte = (slave << 1) | rw

            # --- NEW: 分开记录地址和读写操作 ---
            events.append({'type': 'ADDRESS_7B', 'value': slave})
            events.append({'type': 'WRITE' if rw == 0 else 'READ'})
            # --- END NEW ---

            s, d, l, current_sda_level, _ = self._generate_byte(addr_byte, current_sda_level, LABEL_MAP['ADDR_7_BIT_0'],
                                                                LABEL_MAP['ADDR_7_BIT_1'],
                                                                rw_bit_label=LABEL_MAP['RW_BIT_WRITE'] if rw == 0 else
                                                                LABEL_MAP['RW_BIT_READ'])
            scl_seq.append(s);
            sda_seq.append(d);
            lbl_seq.append(l)

            simulate_addr_nack = np.random.rand() < self.config['prob_addr_nack']
            s, d, l, current_sda_level, ev = self._generate_ack_nack(current_sda_level, simulate_addr_nack)
            scl_seq.append(s);
            sda_seq.append(d);
            lbl_seq.append(l);
            events.append(ev)
            if simulate_addr_nack: addr_nacked = True

        else:  # --- NEW: 10-BIT ADDRESS LOGIC ---
            slave = np.random.randint(0, 0x400)  # 10-bit address range 0 to 1023
            # Logical Event Logging First
            events.append({'type': 'ADDRESS_10B', 'value': slave})
            # The first part of a 10-bit transaction is always a write to set up the address
            events.append({'type': 'WRITE'})

            # --- Physical Transmission ---  # 1. First frame: Header (11110xx0) + W=0
            head_w = 0b11110000 | ((slave >> 8) & 0b11) << 1
            s, d, l, current_sda_level, _ = self._generate_byte(head_w, current_sda_level, LABEL_MAP['ADDR_10_HEAD_0'],
                                                                LABEL_MAP['ADDR_10_HEAD_1'],
                                                                rw_bit_label=LABEL_MAP['RW_BIT_WRITE'])
            scl_seq.append(s);
            sda_seq.append(d);
            lbl_seq.append(l)

            # 2. First ACK/NACK
            simulate_addr_nack1 = np.random.rand() < self.config['prob_addr_nack']
            s, d, l, current_sda_level, ev = self._generate_ack_nack(current_sda_level, simulate_addr_nack1)
            scl_seq.append(s);
            sda_seq.append(d);
            lbl_seq.append(l);
            events.append(ev)

            if not simulate_addr_nack1:
                # 3. Second frame: Lower 8 bits of the address
                low_byte = slave & 0xFF
                s, d, l, current_sda_level, _ = self._generate_byte(low_byte, current_sda_level,
                                                                    LABEL_MAP['ADDR_10_LOW_0'],
                                                                    LABEL_MAP['ADDR_10_LOW_1'])
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l)

                # 4. Second ACK/NACK
                simulate_addr_nack2 = np.random.rand() < self.config['prob_addr_nack']
                s, d, l, current_sda_level, ev = self._generate_ack_nack(current_sda_level, simulate_addr_nack2)
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l);
                events.append(ev)
                if simulate_addr_nack2: addr_nacked = True
            else:
                addr_nacked = True

            # 5. For READ operations, a Repeated Start is required
            if not addr_nacked and op_type in ['READ', 'WRITE_READ']:
                s, d, l, current_sda_level, ev = self._generate_start(rep=True)
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l);
                events.append(ev)

                # Logical event for the read part
                events.append({'type': 'READ'})

                # 6. Third frame: Header (11110xx1) + R=1
                head_r = head_w | 1
                s, d, l, current_sda_level, _ = self._generate_byte(head_r, current_sda_level,
                                                                    LABEL_MAP['ADDR_10_HEAD_0'],
                                                                    LABEL_MAP['ADDR_10_HEAD_1'],
                                                                    rw_bit_label=LABEL_MAP['RW_BIT_READ'])
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l)

                # 7. Third ACK/NACK (should be ACK if slave is present)
                s, d, l, current_sda_level, ev = self._generate_ack_nack(current_sda_level, False)
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l);
                events.append(ev)

        if not addr_nacked and op_type in ['WRITE', 'WRITE_READ']:
            for _ in range(num_write):
                data = np.random.randint(0, 256)
                s, d, l, current_sda_level, ev = self._generate_byte(data, current_sda_level, LABEL_MAP['DATA_BIT_0'],
                                                                     LABEL_MAP['DATA_BIT_1'])
                events.append({'type': 'DATA', 'value': data})  # 将ev中的value赋给新字典
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l)

                nack = np.random.rand() < self.config['prob_data_nack']
                s, d, l, current_sda_level, ev = self._generate_ack_nack(current_sda_level, nack)
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l);
                events.append(ev)
                if nack: break

        if not addr_nacked and addr_bits == 7 and op_type == 'WRITE_READ':
            s, d, l, current_sda_level, ev = self._generate_start(rep=True)
            scl_seq.append(s);
            sda_seq.append(d);
            lbl_seq.append(l);
            events.append(ev)

            # --- NEW: 同样分开记录 ---
            events.append({'type': 'ADDRESS_7B', 'value': slave_addr_for_read})
            events.append({'type': 'READ'})
            # --- END NEW ---

            addr_byte_r = (slave_addr_for_read << 1) | 1
            s, d, l, current_sda_level, _ = self._generate_byte(addr_byte_r, current_sda_level,
                                                                LABEL_MAP['ADDR_7_BIT_0'], LABEL_MAP['ADDR_7_BIT_1'],
                                                                rw_bit_label=LABEL_MAP['RW_BIT_READ'])
            scl_seq.append(s);
            sda_seq.append(d);
            lbl_seq.append(l)

            s, d, l, current_sda_level, ev = self._generate_ack_nack(current_sda_level, False)
            scl_seq.append(s);
            sda_seq.append(d);
            lbl_seq.append(l);
            events.append(ev)

        if not addr_nacked and op_type in ['READ', 'WRITE_READ']:
            for i in range(num_read):
                data = np.random.randint(0, 256)
                s, d, l, current_sda_level, ev = self._generate_byte(data, current_sda_level, LABEL_MAP['DATA_BIT_0'],
                                                                     LABEL_MAP['DATA_BIT_1'])
                events.append({'type': 'DATA', 'value': data})
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l)

                last_byte = (i == num_read - 1)
                s, d, l, current_sda_level, ev = self._generate_ack_nack(current_sda_level, last_byte)
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l);
                events.append(ev)

        s, d, l, current_sda_level, ev = self._generate_stop()
        scl_seq.append(s);
        sda_seq.append(d);
        lbl_seq.append(l);
        events.append(ev)

        final_scl = self.add_noise(self.add_transition_time(np.concatenate(scl_seq)))
        final_sda = self.add_noise(self.add_transition_time(np.concatenate(sda_seq)))
        final_lbl = np.concatenate(lbl_seq)

        return final_scl, final_sda, final_lbl, events


    def generate_i2c_datasets(self, num_datasets=100, samples_per_dataset=10000):
        all_datasets_np = np.zeros((num_datasets, samples_per_dataset, 4), dtype=np.float32)
        all_labels_np = np.zeros((num_datasets, samples_per_dataset), dtype=np.int64)
        all_events = []  # events 仍然是Python列表
        all_channel_maps = []  #保存每个样本的 SCL/SDA 通道映射

        for i in range(num_datasets):
            scl_raw, sda_raw, labels_raw, events_raw = self.generate_i2c_transaction()
            # 随机映射到 4 通道
            ch_indices = np.random.choice(4, 2, replace=False)
            scl_ch, sda_ch = int(ch_indices[0]), int(ch_indices[1])
            all_channel_maps.append((scl_ch, sda_ch))
            # 构建 4 通道波形，并添加微小噪声
            final_waveform_4ch = np.full((samples_per_dataset, 4),self.voltage_high, dtype=np.float32)
            final_waveform_4ch += np.random.normal(0, self.voltage_noise_std / 2, final_waveform_4ch.shape)

            current_len = len(scl_raw)
            # 截断或填充原始波形
            current_len = len(scl_raw)
            L = min(current_len, samples_per_dataset)
            final_waveform_4ch[:L, scl_ch] = scl_raw[:L]
            final_waveform_4ch[:L, sda_ch] = sda_raw[:L]

            # 标签同样截断
            final_labels = np.full(samples_per_dataset, LABEL_MAP['IDLE'], dtype=np.int64)
            final_labels[:L] = labels_raw[:L]

            all_datasets_np[i] = final_waveform_4ch
            all_labels_np[i] = final_labels
            all_events.append(events_raw)

        return all_datasets_np, all_labels_np, all_events, all_channel_maps

    # --- NEW: 完全重写的 save_dataset 函数 ---
    def save_dataset(self, scl, sda, labels, events, base_dir, prefix="i2c"):
        os.makedirs(base_dir, exist_ok=True)
        # 查找下一个可用的文件序号
        existing_files = [f for f in os.listdir(base_dir) if f.startswith(f"{prefix}-") and f.endswith(".csv")]
        next_num = 1
        if existing_files:
            nums = [int(f.split('-')[-1].split('.')[0]) for f in existing_files if
                    f.split('-')[-1].split('.')[0].isdigit()]
            if nums: next_num = max(nums) + 1

        # 定义CSV和TXT文件名
        base_filename = f"{prefix}-{next_num:04d}"
        csv_path = os.path.join(base_dir, f"{base_filename}.csv")
        txt_path = os.path.join(base_dir, f"{base_filename}.txt")

        # 1. 写入波形和标签到CSV
        with open(csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['Time_Index', 'SCL', 'SDA', 'Label'])
            for i in range(len(scl)):
                writer.writerow([i, scl[i], sda[i], int(labels[i])])
        print(f"Saved waveform data to: {csv_path}")

        # 2. 写入事件序列到TXT
        with open(txt_path, 'w') as txtfile:
            for event in events:
                event_type = event['type']
                if 'value' in event:
                    # 对于有数值的事件，格式化为十六进制
                    value_hex = f"0x{event['value']:02X}"
                    txtfile.write(f"{event_type}:{value_hex}\n")
                else:
                    # 对于没有数值的事件，直接写入类型
                    txtfile.write(f"{event_type}\n")
        print(f"Saved event ground truth to: {txt_path}")

    # plot_signals 函数保持不变
    def plot_signals(self, scl, sda, labels=None, title="Realistic I2C Signal", plot_samples=None):
        if plot_samples is None: plot_samples = len(scl)
        plot_samples = min(plot_samples, len(scl))
        t = np.arange(plot_samples) / self.sampling_rate * 1e6
        fig, ax = plt.subplots(figsize=(20, 8))
        ax.plot(t, scl[:plot_samples], 'b-', label='SCL', alpha=0.7)
        ax.plot(t, sda[:plot_samples], 'r-', label='SDA', alpha=0.7)
        if labels is not None:
            colors = plt.get_cmap('tab20', len(LABEL_MAP))
            patches = []
            last_lbl, start_idx = -1, 0
            unique_labels_in_view = sorted(list(set(labels[:plot_samples])))
            for i in range(plot_samples):
                current_lbl = int(labels[i])
                if current_lbl != last_lbl:
                    if last_lbl not in (LABEL_MAP['IDLE'], -1):
                        start_time = start_idx / self.sampling_rate * 1e6
                        end_time = i / self.sampling_rate * 1e6
                        ax.axvspan(start_time, end_time, color=colors(last_lbl), alpha=0.3)
                    start_idx, last_lbl = i, current_lbl
            if last_lbl != LABEL_MAP['IDLE']:
                ax.axvspan(start_idx / self.sampling_rate * 1e6, plot_samples / self.sampling_rate * 1e6,
                           color=colors(last_lbl), alpha=0.3)

            for lbl_id in unique_labels_in_view:
                if lbl_id != LABEL_MAP['IDLE']:
                    patches.append(plt.Rectangle((0, 0), 1, 1, color=colors(lbl_id), alpha=0.3,
                                                 label=f"{int(lbl_id)}:{get_label_name(lbl_id)}"))

            handles, legend_labels = ax.get_legend_handles_labels()
            ax.legend(handles=handles + patches, bbox_to_anchor=(1.02, 1), loc='upper left')
        ax.set_xlabel('Time (µs)');
        ax.set_ylabel('Voltage (V)');
        ax.set_title(title)
        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0, 0, 0.82, 1])
        plt.show()


# --- MODIFIED: __main__ 部分以适应新的函数签名 ---
if __name__ == '__main__':
    # 定义输出目录
    output_dir = "../dataset/i2c_generated"

    gen = RealisticI2CSignalGenerator(config=DEFAULT_I2C_CONFIG)
    print("Generating dataset...")
    # generate_i2c_datasets 现在返回3个值
    all_data, all_labels, all_events, all_maps = gen.generate_i2c_datasets(num_datasets=20, samples_per_dataset=10000)

    # 循环保存每个生成的数据集
    for i in range(len(all_data)):
        data_sample = all_data[i]
        label_sample = all_labels[i]
        event_sample = all_events[i]
        scl_ch, sda_ch = all_maps[i]

        # 从堆叠的数组中分离出SCL和SDA，以便保存
        scl = data_sample[:, scl_ch]
        sda = data_sample[:, sda_ch]

        # 调用保存函数
        gen.save_dataset(scl, sda, label_sample, event_sample, base_dir=output_dir, prefix="i2c")

    print("\nDone.")

    # 可选：画出最后一个生成的数据图以供检查
    # gen.plot_signals(all_data[-1][:, 0], all_data[-1][:, 1], all_labels[-1], title="Last Generated I2C Signal")