import numpy as np
import matplotlib.pyplot as plt
import csv
import os

# 定义标签字典
LABEL_MAP = {
    "IDLE": 0, "START": 1, "STOP": 2, "REPEATED_START": 3,
    "ADDR_7_BIT_0": 4, "ADDR_7_BIT_1": 5, "RW_BIT_READ": 6, "RW_BIT_WRITE": 7,
    "ADDR_10_HEAD_0": 8, "ADDR_10_HEAD_1": 9, "ADDR_10_LOW_0": 10, "ADDR_10_LOW_1": 11,
    "DATA_BIT_0": 12, "DATA_BIT_1": 13, "ACK": 14, "NACK": 15,
}
DEFAULT_I2C_CONFIG = {
        'sampling_rate':8e6, 'scl_freq':1e5,
        'voltage_high':3.3, 'voltage_low':0.0,
        'voltage_noise_std':0.03,
        'jitter_std_factor':0.02,
        'rise_time_factor':0.05, 'fall_time_factor':0.05,
        'prob_write':0.4, 'prob_read':0.4, 'prob_write_read':0.2,
        'prob_10bit_addr':0.05, 'prob_addr_nack':0.05, 'prob_data_nack':0.05,
        'length_jitter_prob':0.05, 'length_jitter_range':0.1,
        'idle_bits_min': 1, 'idle_bits_max': 10,
}

def get_label_name(label_id):
    for name, id_ in LABEL_MAP.items():
        if id_ == label_id:
            return name
    return "UNKNOWN"

class RealisticI2CSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.sampling_rate = config['sampling_rate']
        self.scl_freq = config['scl_freq']
        self.base_samples_per_bit = int(self.sampling_rate / self.scl_freq)

        #START前的IDLE帧
        self.idle_bits_min = config.get('idle_bits_min', 1)
        self.idle_bits_max = config.get('idle_bits_max', 10)
        # 非理想参数
        self.voltage_high = config['voltage_high']
        self.voltage_low = config['voltage_low']
        self.voltage_noise_std = config.get('voltage_noise_std', 0.03)
        self.jitter_std_factor = config.get('jitter_std_factor', 0.04)
        self.rise_time_factor = config.get('rise_time_factor', 0.2)
        self.fall_time_factor = config.get('fall_time_factor', 0.2)
        # 随机长度抖动参数
        self.length_jitter_prob = config.get('length_jitter_prob', 0.1)
        self.length_jitter_range = config.get('length_jitter_range', 0.1)

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
                if signal[e] < signal[e+1] and rise > 0:
                    end = min(e + rise, len(signal)-1)
                    result[e:end] = np.linspace(signal[e], signal[end], end-e)
                elif fall > 0:
                    end = min(e + fall, len(signal)-1)
                    result[e:end] = np.linspace(signal[e], signal[end], end-e)
        return result

    def _generate_bit(self, bit_value, prev_sda_level, lbl0, lbl1):
        spb = self._get_samples_per_bit()

        # SCL 逻辑不变: 前半周期低, 后半周期高
        scl = np.ones(spb) * self.voltage_low
        scl[int(spb / 2):] = self.voltage_high

        # NEW: 核心SDA时序修正逻辑
        current_sda_level = self.voltage_high if bit_value else self.voltage_low
        sda = np.full(spb, current_sda_level)

        if current_sda_level != prev_sda_level:
            # 如果SDA电平需要变化, 就在SCL为低的期间完成
            # 我们选择在SCL低电平周期的中间点进行翻转 (即整个比特周期的1/4处)
            transition_point = int(spb * 0.25)
            sda[:transition_point] = prev_sda_level  # 保持上一比特的电平
            sda[transition_point:] = current_sda_level  # 然后翻转到新电平

        label = lbl1 if bit_value else lbl0
        labels = np.full(spb, label)

        # MODIFIED: 返回当前SDA的最终电平, 作为下一个bit的prev_sda_level
        return scl, sda, labels, current_sda_level

    def _generate_byte(self, val, prev_sda_level, lbl0, lbl1, rw_bit_label=None):
        S, D, L = [], [], []
        current_sda_level = prev_sda_level
        for i in range(7, -1, -1):
            bit = (val >> i) & 1
            if i == 0 and rw_bit_label is not None:
                b0 = b1 = rw_bit_label
            else:
                b0, b1 = lbl0, lbl1

            # MODIFIED: 传入当前的SDA电平, 并接收新的SDA电平
            s, d, l, current_sda_level = self._generate_bit(bit, current_sda_level, b0, b1)
            S.append(s);
            D.append(d);
            L.append(l)

        # MODIFIED: 返回最终的SDA电平
        return np.concatenate(S), np.concatenate(D), np.concatenate(L), current_sda_level

    def _generate_ack_nack(self, prev_sda_level, simulate_nack=False):
        bit = 1 if simulate_nack else 0
        return self._generate_bit(bit, prev_sda_level, LABEL_MAP['ACK'], LABEL_MAP['NACK'])

    def _generate_start(self, rep=False):
        spb = self._get_samples_per_bit()
        scl = np.ones(spb) * self.voltage_high
        mid = int(spb / 2 + np.random.normal(0, self.jitter_std_factor * spb))
        mid = np.clip(mid, 0, spb)
        sda = np.ones(spb) * self.voltage_high
        sda[mid:] = self.voltage_low
        lbl = LABEL_MAP['REPEATED_START'] if rep else LABEL_MAP['START']
        return scl, sda, np.full(spb, lbl), self.voltage_low  # Start后SDA为低

    def _generate_stop(self):
        spb = self._get_samples_per_bit()
        scl = np.ones(spb) * self.voltage_high

        mid = int(spb / 2 + np.random.normal(0, self.jitter_std_factor * spb))
        mid = np.clip(mid, 0, spb)
        # 为了产生STOP(SDA由低到高)，SDA必须从低电平开始
        sda = np.full(spb, self.voltage_low)
        sda[mid:] = self.voltage_high
        labels = np.full(spb, LABEL_MAP['STOP'])
        # STOP条件结束后，SDA保持高电平
        return scl, sda, labels, self.voltage_high

    def generate_i2c_transaction(self):
        scl_seq, sda_seq, lbl_seq ,events = [], [], [] ,[]

        op_rand = np.random.rand()
        if op_rand < self.config['prob_write']:
            op_type = 'WRITE'
        elif op_rand < self.config['prob_write'] + self.config['prob_read']:
            op_type = 'READ'
        else:
            op_type = 'WRITE_READ'
        addr_bits = 10 if np.random.rand() < self.config['prob_10bit_addr'] else 7
        simulate_addr_nack = np.random.rand() < self.config['prob_addr_nack']
        num_write = np.random.randint(1, self.config.get('max_write_bytes', 10) + 1)
        num_read = np.random.randint(1, self.config.get('max_read_bytes', 10) + 1)

        # --- 生成 ---   # NEW: 初始化SDA电平状态。事务开始前, SDA为高电平(IDLE)
        current_sda_level = self.voltage_high

        # —— 生成START前的IDLE函数 ——
        idle_s, idle_d, idle_l = self._generate_idle_frames_before_start()
        scl_seq.extend(idle_s)
        sda_seq.extend(idle_d)
        lbl_seq.extend(idle_l)

        # Start
        events.append({'type': 'START'}) #直接加入事件列表
        s, d, l, current_sda_level = self._generate_start()
        scl_seq.append(s);
        sda_seq.append(d);
        lbl_seq.append(l)
        addr_nacked = False

        # --- 地址阶段 ---
        if addr_bits == 7:
            slave = np.random.randint(0x08, 0x77)
            rw = 0 if op_type in ['WRITE', 'WRITE_READ'] else 1
            addr = (slave << 1) | rw
            s, d, l, current_sda_level = self._generate_byte(addr, current_sda_level, LABEL_MAP['ADDR_7_BIT_0'],
                                                             LABEL_MAP['ADDR_7_BIT_1'],
                                                             rw_bit_label=LABEL_MAP['RW_BIT_WRITE'] if rw == 0 else
                                                             LABEL_MAP['RW_BIT_READ'])
            scl_seq.append(s);          sda_seq.append(d);          lbl_seq.append(l);
            events.append({'type': 'ADDRESS_7bit', 'value': addr})
            # ACK/NACK
            s, d, l, current_sda_level = self._generate_ack_nack(current_sda_level, simulate_addr_nack)
            scl_seq.append(s);          sda_seq.append(d);          lbl_seq.append(l);
            events.append({'type': 'NACK' if simulate_addr_nack else 'ACK'})
            if simulate_addr_nack: addr_nacked = True

        else:  # 10位地址
            slave = np.random.randint(0, 0x400)
            head = 0b11110000 | (((slave >> 8) & 3) << 1)
            s, d, l, current_sda_level = self._generate_byte(head, current_sda_level, LABEL_MAP['ADDR_10_HEAD_0'],
                                                             LABEL_MAP['ADDR_10_HEAD_1'],
                                                             rw_bit_label=LABEL_MAP['RW_BIT_WRITE'])
            scl_seq.append(s);          sda_seq.append(d);          lbl_seq.append(l);
            events.append({'type': 'ADDRESS_10bit_high', 'value': head})
            # ACK/NACK
            s, d, l, current_sda_level = self._generate_ack_nack(current_sda_level, simulate_addr_nack)
            scl_seq.append(s);          sda_seq.append(d);          lbl_seq.append(l)
            events.append({'type': 'NACK' if simulate_addr_nack else 'ACK'})

            if not simulate_addr_nack:
                low = slave & 0xFF
                s, d, l, current_sda_level = self._generate_byte(low, current_sda_level, LABEL_MAP['ADDR_10_LOW_0'],LABEL_MAP['ADDR_10_LOW_1'])
                scl_seq.append(s);      sda_seq.append(d);          lbl_seq.append(l)
                events.append({'type': 'ADDRESS_10bit_low', 'value': low})

                s, d, l, current_sda_level = self._generate_ack_nack(current_sda_level,simulate_addr_nack)  # Can also NACK here
                scl_seq.append(s);      sda_seq.append(d);          lbl_seq.append(l)
                events.append({'type': 'NACK' if simulate_addr_nack else 'ACK'})
                if simulate_addr_nack: addr_nacked = True
            else:
                addr_nacked = True

            if not addr_nacked and op_type in ['READ', 'WRITE_READ']:
                s, d, l, current_sda_level = self._generate_start(rep=True)
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l)

                head_r = head | 1
                s, d, l, current_sda_level = self._generate_byte(head_r, current_sda_level, LABEL_MAP['ADDR_10_HEAD_0'],
                                                                 LABEL_MAP['ADDR_10_HEAD_1'],
                                                                 rw_bit_label=LABEL_MAP['RW_BIT_READ'])
                scl_seq.append(s);  sda_seq.append(d);  lbl_seq.append(l)

                s, d, l, current_sda_level = self._generate_ack_nack(current_sda_level, False)
                scl_seq.append(s);  sda_seq.append(d);  lbl_seq.append(l)

        # --- 数据传输 ---
        if not addr_nacked:
            if op_type in ['WRITE', 'WRITE_READ']:
                for _ in range(num_write):
                    data = np.random.randint(0, 256)
                    nack = np.random.rand() < self.config['prob_data_nack']
                    s, d, l, current_sda_level = self._generate_byte(data, current_sda_level, LABEL_MAP['DATA_BIT_0'],
                                                                     LABEL_MAP['DATA_BIT_1'])
                    scl_seq.append(s);          sda_seq.append(d);          lbl_seq.append(l)
                    events.append({'type': 'DATA', 'value': data})

                    s, d, l, current_sda_level = self._generate_ack_nack(current_sda_level, nack)
                    scl_seq.append(s);          sda_seq.append(d);          lbl_seq.append(l)
                    events.append({'type': 'NACK' if nack else 'ACK'})

                    if nack: break

            if addr_bits == 7 and op_type == 'WRITE_READ':
                s, d, l, current_sda_level = self._generate_start(rep=True)
                scl_seq.append(s);          sda_seq.append(d);          lbl_seq.append(l)

                rw_addr = (slave << 1) | 1
                s, d, l, current_sda_level = self._generate_byte(rw_addr, current_sda_level, LABEL_MAP['ADDR_7_BIT_0'],
                                                                 LABEL_MAP['ADDR_7_BIT_1'],
                                                                 rw_bit_label=LABEL_MAP['RW_BIT_READ'])
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l)

                s, d, l, current_sda_level = self._generate_ack_nack(current_sda_level, False)
                scl_seq.append(s);
                sda_seq.append(d);
                lbl_seq.append(l)

            if op_type in ['READ', 'WRITE_READ']:
                for i in range(num_read):
                    data = np.random.randint(0, 256)  # In a real read, this would come from the slave
                    last = (i == num_read - 1)
                    s, d, l, current_sda_level = self._generate_byte(data, current_sda_level, LABEL_MAP['DATA_BIT_0'],
                                                                     LABEL_MAP['DATA_BIT_1'])
                    scl_seq.append(s);
                    sda_seq.append(d);
                    lbl_seq.append(l)

                    # Master sends ACK/NACK
                    s, d, l, current_sda_level = self._generate_ack_nack(current_sda_level, last)
                    scl_seq.append(s);
                    sda_seq.append(d);
                    lbl_seq.append(l)

        # --- 结束 ---
        s, d, l, current_sda_level = self._generate_stop()
        scl_seq.append(s);          sda_seq.append(d);          lbl_seq.append(l)
        events.append({'type': 'STOP'})

        # 最终处理 (无变化)
        final_scl = self.add_noise(self.add_transition_time(np.concatenate(scl_seq)))
        final_sda = self.add_noise(self.add_transition_time(np.concatenate(sda_seq)))
        final_lbl = np.concatenate(lbl_seq)
        return final_scl, final_sda, final_lbl, events

    def generate_i2c_datasets(self, num_datasets=100, samples_per_dataset=10000):
        all_datasets, all_labels, all_events = [], [], []

        for _ in range(num_datasets):
            # 1. 生成一个完整的、长度可变的I2C事务
            s, d, l, ev = self.generate_i2c_transaction()

            # 2. 检查生成的事务是否过长
            current_len = len(s)
            if current_len > samples_per_dataset:
                # 如果单个事务就超长了，这是一个警告。# 我们可以选择截断它，或者在日志中提示并跳过。# 这里我们选择截断并打印警告。
                print(f"Warning: A single I2C transaction ({current_len} samples) "
                      f"exceeded the target length ({samples_per_dataset}). Truncating.")
                s = s[:samples_per_dataset]
                d = d[:samples_per_dataset]
                l = l[:samples_per_dataset]
                final_scl = s
                final_sda = d
                final_labels = l
            else:
                # 3. 如果长度足够，用IDLE状态进行填充
                padding_len = samples_per_dataset - current_len

                # 创建填充部分
                scl_padding = np.full(padding_len, self.voltage_high)
                sda_padding = np.full(padding_len, self.voltage_high)
                label_padding = np.full(padding_len, LABEL_MAP['IDLE'])

                # 将事务和填充部分拼接起来
                final_scl = np.concatenate([s, scl_padding])
                final_sda = np.concatenate([d, sda_padding])
                final_labels = np.concatenate([l, label_padding])

            all_datasets.append(np.stack([final_scl, final_sda], axis=-1))
            all_labels.append(final_labels)
            all_events.append(ev)

        # 直接返回我们需要的格式
        return np.array(all_datasets), all_labels, all_events

    def _generate_idle_frames_before_start(self):
        """生成随机 1~N 个 IDLE 比特周期的 SCL/SDA/Label 序列"""
        scl_seq, sda_seq, lbl_seq = [], [], []
        idle_bits = np.random.randint(self.idle_bits_min, self.idle_bits_max + 1)
        for _ in range(idle_bits):
            spb = self._get_samples_per_bit()
            idle_scl = np.full(spb, self.voltage_high)
            idle_sda = np.full(spb, self.voltage_high)
            idle_label = np.full(spb, LABEL_MAP['IDLE'])
            scl_seq.append(idle_scl)
            sda_seq.append(idle_sda)
            lbl_seq.append(idle_label)
        return scl_seq, sda_seq, lbl_seq

    def plot_signals(self, scl, sda, labels=None, title="Realistic I2C Signal", plot_samples=None):
        if plot_samples is None:
            plot_samples = len(scl)
        plot_samples = min(plot_samples, len(scl))
        t = np.arange(plot_samples)/self.sampling_rate*1e6
        fig, ax = plt.subplots(figsize=(20,8))
        ax.plot(t, scl[:plot_samples], 'b-', label='SCL')
        ax.plot(t, sda[:plot_samples], 'r-', label='SDA')
        if labels is not None:
            colors = plt.get_cmap('tab20', len(LABEL_MAP))
            patches = []
            last, start = -1, 0
            for i in range(plot_samples):
                cur = labels[i]
                if cur != last:
                    if last not in (LABEL_MAP['IDLE'], -1):
                        st = start/self.sampling_rate*1e6
                        ed = (i-1)/self.sampling_rate*1e6
                        ax.axvspan(st, ed, color=colors(last), alpha=0.3)
                    start, last = i, cur
            if last != LABEL_MAP['IDLE']:
                st = start/self.sampling_rate*1e6
                ed = (plot_samples-1)/self.sampling_rate*1e6
                ax.axvspan(st, ed, color=colors(last), alpha=0.3)
            for lbl in set(labels[:plot_samples]):
                if lbl != LABEL_MAP['IDLE']:
                    patches.append(plt.Rectangle((0,0),1,1, color=colors(lbl), alpha=0.3,
                                                 label=f"{lbl}:{get_label_name(lbl)}"))
            h, l = ax.get_legend_handles_labels()
            ax.legend(handles=h+patches, bbox_to_anchor=(1.02,1), loc='upper left')
        ax.set_xlabel('Time (µs)'); ax.set_ylabel('Voltage (V)'); ax.set_title(title)
        ax.grid(True, linestyle='--', linewidth=0.5)
        plt.tight_layout(rect=[0,0,0.85,1])
        plt.show()

    def save_dataset(self, scl, sda, labels, events, base_dir="../dataset/i2c_dataset", prefix="i2c"):
        os.makedirs(base_dir, exist_ok=True)
        existing = [f for f in os.listdir(base_dir) if f.startswith(prefix+"-") and f.endswith(".csv")]
        nums = []
        for fn in existing:
            try:
                nums.append(int(fn[len(prefix)+1:fn.rfind(".csv")]))
            except: pass
        nxt = max(nums)+1 if nums else 1

        # CSV 文件路径
        csv_fn = f"{prefix}-{nxt:02d}.csv"
        csv_path = os.path.join(base_dir, csv_fn)
        # 写入 CSV
        with open(csv_path, 'w', newline='') as csvfile:
            w = csv.writer(csvfile)
            w.writerow(['Time', 'SCL', 'SDA', 'Label'])
            for i in range(len(scl)):
                w.writerow([i, scl[i], sda[i], labels[i]])
        print(f"Saved CSV: {csv_path}")

        # 导出到同名 TXT
        txt_fn = csv_fn.replace('.csv', '.txt')
        txt_path = os.path.join(base_dir, txt_fn)
        with open(txt_path, 'w') as f:
            for ev in events:
                if 'value' in ev:
                    f.write(f"{ev['type']}:0x{ev['value']:02X}\n")
                else:
                    f.write(f"{ev['type']}\n")
        print(f"Saved events TXT: {txt_path}")

    def export_frames_to_txt(self,frames, txt_path):
        """
        把 parse_i2c_frames 得到的帧列表写到 txt，格式：
            START;start=100;len=10
            ADDRESS:0x3A;start=110;len=80
            ACK;start=190;len=10
            DATA:0xF1;start=200;len=80
            NACK;start=290;len=10
            STOP;start=300;len=10
        """
        with open(txt_path, 'w') as f:
            for fr in frames:
                parts = [fr['type']]
                if 'value' in fr:
                    parts[0] += f":0x{fr['value']:02X}"
                parts.append(f"start={fr['start']}")
                parts.append(f"len={fr['length']}")
                f.write(";".join(parts) + "\n")
        print(f"Parsed TXT saved: {txt_path}")


if __name__ == '__main__':

    gen = RealisticI2CSignalGenerator(config=DEFAULT_I2C_CONFIG)
    print("Generating dataset...")
    all_data, all_lbl, events = gen.generate_i2c_datasets(num_datasets=10, samples_per_dataset=10000)
    for i, (data, lbl, events) in enumerate(zip(all_data, all_lbl,events)):
        # data 的形状是 (samples_per_dataset, 2)，第 0 维是 SCL，第 1 维是 SDA
        scl = data[:, 0]
        sda = data[:, 1]
        gen.save_dataset(scl, sda, lbl, events, prefix=f"i2c")
    print("Done.")

    #画出最后生成的一个I2C协议数据图
    # gen.plot_signals(scl,sda,lbl,title="Generated I2C Signal with Labels",plot_samples=10000)