import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# --- SPI 标签定义 ---
LABEL_MAP = {
    "IDLE": 0,       # CS High, Bus Floating
    "CS_ACTIVE": 1,  # CS Low, Setup/Hold time
    "DATA_0": 2,     # Logic 0
    "DATA_1": 3,     # Logic 1
    "GLITCH": 4      # Noise
}

def get_label_name(label_id):
    for name, id_ in LABEL_MAP.items():
        if id_ == label_id: return name
    return "UNKNOWN"

# --- 增强版配置 ---
DEFAULT_SPI_CONFIG = {
    'voltage_high': 3.3,
    'voltage_low': 0.0,
    'voltage_noise_std': 0.03,
    'jitter_std_factor': 0.02,
    
    'clk_freq_options': [100e3, 500e3, 1e6, 5e6],
    'sampling_rate_options': [10e6, 20e6, 50e6, 100e6, 200e6],

    # SPI 模式概率 [Mode0, Mode1, Mode2, Mode3]
    'prob_modes': [0.4, 0.2, 0.2, 0.2], 

    # [新增] 突发传输配置
    'burst_prob': 0.5,        # 50% 概率连续发多字节
    'min_burst_len': 2,
    'max_burst_len': 10,      # 一次最多发 10 个词

    # [新增] 位宽多样性
    'prob_8bit': 0.6,         # 60% 是标准 8-bit
    'prob_16bit': 0.2,        # 20% 是 16-bit
    'prob_custom_bit': 0.2,   # 20% 是奇怪的位宽 (9, 10, 12...)

    # [新增] 大小端模式
    'prob_lsb_first': 0.2,    # 20% 概率低位先行

    # [新增] 高阻态模拟
    'high_z_noise_std': 0.1,  # 悬空时的噪声更大
    'crosstalk_factor': 0.15
}

class RealisticSPISignalGenerator:
    def __init__(self, config):
        self.config = config
        self.v_h = config['voltage_high']
        self.v_l = config['voltage_low']
        
        self.sclk_freq = None
        self.fs = None
        self.spb = None 
        
        self.cpol = 0
        self.cpha = 0
        self.bits_per_word = 8
        self.lsb_first = False

    def _select_parameters(self):
        # 基础参数
        self.sclk_freq = np.random.choice(self.config['clk_freq_options'])
        valid_fs = [fs for fs in self.config['sampling_rate_options'] if fs >= 10 * self.sclk_freq]
        self.fs = np.random.choice(valid_fs) if valid_fs else max(self.config['sampling_rate_options'])
        self.spb = int(self.fs / (2 * self.sclk_freq))
        
        # 模式选择
        mode = np.random.choice([0, 1, 2, 3], p=self.config['prob_modes'])
        self.cpol = (mode >> 1) & 1
        self.cpha = mode & 1

        # [新增] 位宽选择
        r = np.random.rand()
        if r < self.config['prob_8bit']: self.bits_per_word = 8
        elif r < self.config['prob_8bit'] + self.config['prob_16bit']: self.bits_per_word = 16
        else: self.bits_per_word = np.random.choice([9, 10, 12, 14]) # 模拟 ADC/Screen

        # [新增] 大小端
        self.lsb_first = True if np.random.rand() < self.config['prob_lsb_first'] else False

    def _generate_high_z_noise(self, length):
        """模拟高阻态(High-Z)：悬空线会随环境漂移"""
        # 随机游走 + 大底噪
        walk = np.cumsum(np.random.normal(0, 0.01, length))
        noise = np.random.normal(0, self.config['high_z_noise_std'], length)
        # 限制在 -0.5 ~ 3.8 之间
        floating_level = 1.0 + walk + noise # 悬空电压通常在中间某处飘荡，不一定是0
        return floating_level

    def generate_transaction(self):
        self._select_parameters()
        
        cs_seq, sclk_seq, mosi_seq, miso_seq, label_seq = [], [], [], [], []
        events = []
        
        # --- 1. IDLE 阶段 (高阻态模拟) ---
        idle_len = int(self.spb * np.random.randint(5, 10))
        
        cs_seq.append(np.full(idle_len, self.v_h)) # CS High
        sclk_seq.append(np.full(idle_len, self.v_h if self.cpol else self.v_l))
        
        # 关键：IDLE 时 MISO/MOSI 可能是悬空的 (High-Z)
        # 这里我们让 MOSI 为 0 (Master 驱动)，MISO 为悬空 (Slave High-Z)
        mosi_seq.append(np.full(idle_len, self.v_l)) 
        miso_seq.append(self._generate_high_z_noise(idle_len)) 
        label_seq.append(np.full(idle_len, LABEL_MAP['IDLE']))

        # --- 2. CS Active ---
        cs_setup = int(self.spb * 2)
        cs_seq.append(np.full(cs_setup, self.v_l))
        sclk_seq.append(np.full(cs_setup, self.v_h if self.cpol else self.v_l))
        mosi_seq.append(np.full(cs_setup, self.v_l))
        miso_seq.append(self._generate_high_z_noise(cs_setup)) # CS拉低瞬间 Slave 可能还没反应过来
        label_seq.append(np.full(cs_setup, LABEL_MAP['CS_ACTIVE']))

        # --- 3. 突发数据传输 ---
        if np.random.rand() < self.config['burst_prob']:
            word_count = np.random.randint(self.config['min_burst_len'], self.config['max_burst_len'])
        else:
            word_count = 1

        events.append({
            'type': 'CONFIG', 
            'mode': f"CPOL={self.cpol},CPHA={self.cpha}", 
            'bits': self.bits_per_word,
            'lsb': self.lsb_first
        })

        for _ in range(word_count):
            val_mosi = np.random.randint(0, 2**self.bits_per_word)
            events.append({'type': 'DATA', 'value': val_mosi})
            
            # 生成比特序列
            range_bits = range(self.bits_per_word) if self.lsb_first else range(self.bits_per_word-1, -1, -1)
            
            for i in range_bits:
                bit_mosi = (val_mosi >> i) & 1
                bit_miso = np.random.randint(0, 2) # MISO 随机数据
                
                # 半周期长度 (加 jitter)
                t_1 = int(self.spb * np.random.uniform(0.95, 1.05))
                t_2 = int(self.spb * np.random.uniform(0.95, 1.05))
                
                # 电平定义
                clk_idle = self.v_h if self.cpol else self.v_l
                clk_active = self.v_l if self.cpol else self.v_h
                v_mosi = self.v_h if bit_mosi else self.v_l
                v_miso = self.v_h if bit_miso else self.v_l
                l_mosi = LABEL_MAP['DATA_1'] if bit_mosi else LABEL_MAP['DATA_0']

                # 根据 CPHA 生成时序
                if self.cpha == 0:
                    # Sample @ Edge 1, Change @ Edge 2
                    # Phase 1: Edge 1 occurs. Data Valid.
                    sclk_bit = np.concatenate([np.full(t_1, clk_active), np.full(t_2, clk_idle)])
                    # Data stable for whole period (simplified)
                    mosi_bit = np.full(t_1 + t_2, v_mosi)
                    miso_bit = np.full(t_1 + t_2, v_miso)
                else:
                    # Change @ Edge 1, Sample @ Edge 2
                    sclk_bit = np.concatenate([np.full(t_1, clk_active), np.full(t_2, clk_idle)])
                    mosi_bit = np.full(t_1 + t_2, v_mosi)
                    miso_bit = np.full(t_1 + t_2, v_miso)

                sclk_seq.append(sclk_bit)
                mosi_seq.append(mosi_bit)
                miso_seq.append(miso_bit)
                cs_seq.append(np.full(len(sclk_bit), self.v_l))
                label_seq.append(np.full(len(sclk_bit), l_mosi))

        # --- 4. End Frame ---
        cs_hold = int(self.spb * 2)
        cs_seq.append(np.full(cs_hold, self.v_h))
        sclk_seq.append(np.full(cs_hold, self.v_h if self.cpol else self.v_l))
        mosi_seq.append(np.full(cs_hold, self.v_l))
        miso_seq.append(self._generate_high_z_noise(cs_hold)) # 回到 High-Z
        label_seq.append(np.full(cs_hold, LABEL_MAP['IDLE']))

        # 合并
        sclk = np.concatenate(sclk_seq)
        mosi = np.concatenate(mosi_seq)
        miso = np.concatenate(miso_seq)
        cs = np.concatenate(cs_seq)
        labels = np.concatenate(label_seq)

        # 添加物理损伤
        # 1. 串扰
        diff = np.diff(sclk, prepend=sclk[0]) * self.config['crosstalk_factor']
        mosi += diff
        miso += diff
        
        # 2. 高斯底噪
        noise_std = self.config['voltage_noise_std']
        sclk += np.random.normal(0, noise_std, len(sclk))
        mosi += np.random.normal(0, noise_std, len(mosi))
        miso += np.random.normal(0, noise_std, len(miso))
        cs += np.random.normal(0, noise_std, len(cs))

        return sclk, mosi, miso, cs, labels, events

    def generate_spi_datasets(self, num_datasets=100, samples_per_dataset=10000):
        all_datasets = np.zeros((num_datasets, samples_per_dataset, 4), dtype=np.float32)
        all_labels = np.zeros((num_datasets, samples_per_dataset), dtype=np.int64)
        all_events = []
        all_maps = []

        for i in range(num_datasets):
            sclk, mosi, miso, cs, lbl, evt = self.generate_transaction()
            L = min(len(sclk), samples_per_dataset)
            
            # 随机打乱通道映射 (增强泛化性)
            # 默认: 0:CS, 1:SCLK, 2:MOSI, 3:MISO
            ch_indices = [0, 1, 2, 3]
            # np.random.shuffle(ch_indices) # 如果想让模型自己猜通道，可以开启
            
            all_datasets[i, :L, ch_indices[0]] = cs[:L]
            all_datasets[i, :L, ch_indices[1]] = sclk[:L]
            all_datasets[i, :L, ch_indices[2]] = mosi[:L]
            all_datasets[i, :L, ch_indices[3]] = miso[:L]
            
            # Padding
            if L < samples_per_dataset:
                # CS Idle High
                all_datasets[i, L:, ch_indices[0]] = self.v_h
                # SCLK Idle (CPOL dependent, but random noise here is safer)
                all_datasets[i, L:, ch_indices[1]] = self.v_h if self.cpol else self.v_l
                
            all_labels[i, :L] = lbl[:L]
            all_events.append(evt)
            all_maps.append(tuple(ch_indices))

        return all_datasets, all_labels, all_events, all_maps

# 测试生成
if __name__ == '__main__':
    gen = RealisticSPISignalGenerator(DEFAULT_SPI_CONFIG)
    data, labels, events, maps = gen.generate_spi_datasets(5, 1250) # 短一点方便看
    
    # 画图验证
    plt.figure(figsize=(12, 8))
    t = np.arange(1250)
    plt.subplot(4,1,1); plt.plot(t, data[0,:,0]); plt.ylabel('CS'); plt.title(f"SPI Mode: CPOL={gen.cpol}, CPHA={gen.cpha}")
    plt.subplot(4,1,2); plt.plot(t, data[0,:,1]); plt.ylabel('SCLK')
    plt.subplot(4,1,3); plt.plot(t, data[0,:,2]); plt.ylabel('MOSI')
    plt.subplot(4,1,4); plt.plot(t, data[0,:,3]); plt.ylabel('MISO')
    plt.tight_layout()
    plt.show()
