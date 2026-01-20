import numpy as np
import matplotlib.pyplot as plt
import os
import csv

# --- SPI 标签定义 ---
# SPI 物理层很简单，没有 ACK/NACK，也没有强制的 ADDR
# 我们主要关注 DATA 和 CS 状态
LABEL_MAP = {
    "IDLE": 0,       # CS 为高，或 CS 为低但无时钟
    "CS_ACTIVE": 1,  # CS 下降沿/上升沿的过渡区
    "DATA_0": 2,     # 逻辑 0
    "DATA_1": 3,     # 逻辑 1
    "GLITCH": 4      # 噪声/串扰
}

def get_label_name(label_id):
    for name, id_ in LABEL_MAP.items():
        if id_ == label_id: return name
    return "UNKNOWN"

# --- 默认配置 ---
DEFAULT_SPI_CONFIG = {
    'voltage_high': 3.3,
    'voltage_low': 0.0,
    'voltage_noise_std': 0.03,
    'jitter_std_factor': 0.02,
    
    # 时钟频率选项 (SPI 可以跑很快，从几百K到几十M)
    'clk_freq_options': [100e3, 500e3, 1e6, 5e6, 10e6, 20e6],
    
    # 采样率 (通常要求采样率 > 4 * 时钟频率)
    'sampling_rate_options': [2e6, 10e6, 20e6, 50e6, 100e6, 200e6, 500e6],

    # SPI 模式概率 (Mode 0 ~ 3)
    # Mode 0: CPOL=0, CPHA=0 (最常见)
    # Mode 1: CPOL=0, CPHA=1
    # Mode 2: CPOL=1, CPHA=0
    # Mode 3: CPOL=1, CPHA=1
    'prob_modes': [0.4, 0.2, 0.2, 0.2], 

    # 传输参数
    'min_bytes': 1, 
    'max_bytes': 8,
    
    # 损伤参数
    'crosstalk_factor': 0.15, # 串扰强度 (SCLK 对 MISO/MOSI 的干扰)
    'prob_glitch': 0.1        # 随机毛刺概率
}

class RealisticSPISignalGenerator:
    def __init__(self, config):
        self.config = config
        self.v_h = config['voltage_high']
        self.v_l = config['voltage_low']
        
        self.sclk_freq = None
        self.fs = None
        self.spb = None # samples per bit (half cycle actually)
        
        self.cpol = 0
        self.cpha = 0

    def _select_parameters(self):
        # 1. 随机选择频率
        self.sclk_freq = np.random.choice(self.config['clk_freq_options'])
        
        # 2. 选择合适的采样率 (至少 10 倍过采样以保证波形质量)
        valid_fs = [fs for fs in self.config['sampling_rate_options'] if fs >= 10 * self.sclk_freq]
        self.fs = np.random.choice(valid_fs) if valid_fs else max(self.config['sampling_rate_options'])
        
        # 一个时钟周期 = 2 个 bit 宽度 (High + Low)
        # 这里计算半周期的采样点数
        self.spb = int(self.fs / (2 * self.sclk_freq))
        
        # 3. 随机选择 SPI 模式
        mode = np.random.choice([0, 1, 2, 3], p=self.config['prob_modes'])
        self.cpol = (mode >> 1) & 1
        self.cpha = mode & 1

    def _add_noise(self, signal):
        return signal + np.random.normal(0, self.config['voltage_noise_std'], len(signal))

    def _add_crosstalk(self, victim, aggressor):
        """
        模拟线间串扰 (Crosstalk): Aggressor (如 SCLK) 的跳变会在 Victim 上产生尖峰
        V_noise = k * dV/dt
        """
        diff = np.diff(aggressor, prepend=aggressor[0])
        # 简单的微分模拟：跳变处产生尖峰
        crosstalk = diff * self.config['crosstalk_factor']
        return victim + crosstalk

    def _generate_byte_waveform(self, val_mosi, val_miso):
        """
        生成 8-bit 数据的 SCLK, MOSI, MISO 波形
        注意：SPI 的数据采样取决于 CPHA
        """
        sclk_seq = []
        mosi_seq = []
        miso_seq = []
        labels_mosi = [] # 仅以 MOSI 为例生成标签
        
        # 初始时钟电平
        idle_clk = self.v_h if self.cpol else self.v_l
        active_clk = self.v_l if self.cpol else self.v_h
        
        # 数据位 (MSB First)
        for i in range(7, -1, -1):
            bit_mosi = (val_mosi >> i) & 1
            bit_miso = (val_miso >> i) & 1
            
            # 生成半周期采样点 (加抖动)
            len_ph1 = int(self.spb * np.random.uniform(0.95, 1.05))
            len_ph2 = int(self.spb * np.random.uniform(0.95, 1.05))
            
            # 这里的逻辑比较绕，取决于 CPHA
            # CPHA=0: Data sample at 1st edge, change at trailing edge
            # CPHA=1: Data change at 1st edge, sample at 2nd edge
            
            # Phase 1
            p1_clk = active_clk if self.cpha else idle_clk # CPHA=0时, P1是Idle->Active边沿前
            # 实际上：
            # CPHA=0: P1 是 SCLK 的前半段 (Idle -> Active 边沿在这里发生)
            # CPHA=1: P1 是 SCLK 的前半段 (Idle -> Active 边沿在这里发生)
            
            # 为了简化，我们按 "Edge 1" 和 "Edge 2" 生成
            
            # 构建标准的 2-phase 时钟周期
            if self.cpha == 0:
                # Mode 0/2: Sample @ 1st Edge (Middle of Phase 1 is tricky, usually Sample occurs at Edge)
                # Data must be stable BEFORE 1st edge.
                # P1: Active Edge occurs. Data Valid.
                # P2: Trailing Edge. Data Changes.
                
                # SCLK: Idle -> Active -> Idle
                # CPOL=0: Low -> High -> Low
                w_clk_p1 = np.full(len_ph1, active_clk) 
                w_clk_p2 = np.full(len_ph2, idle_clk)
                
            else: # CPHA=1
                # Mode 1/3: Sample @ 2nd Edge.
                # P1: Leading Edge. Data Changes.
                # P2: Trailing Edge. Data Sampled.
                
                # SCLK: Idle -> Active -> Idle
                w_clk_p1 = np.full(len_ph1, active_clk)
                w_clk_p2 = np.full(len_ph2, idle_clk)

            # 修正 SCLK 时序：
            # 实际上代码写反了，重写生成逻辑：
            # 我们直接生成 SCLK 的脉冲，然后对齐 Data
            
            pass # 重新设计逻辑见下文
            
        return sclk_seq, mosi_seq, miso_seq, labels_mosi

    def generate_transaction(self):
        self._select_parameters()
        
        # 1. 初始化空闲线
        # CS 默认为高 (无效)
        # SCLK 默认为 CPOL
        # MOSI/MISO 默认为 0 或 1 (随机)
        
        cs_seq, sclk_seq, mosi_seq, miso_seq, label_seq = [], [], [], [], []
        events = []
        
        idle_len = int(self.spb * 5)
        cs_seq.append(np.full(idle_len, self.v_h))
        sclk_seq.append(np.full(idle_len, self.v_h if self.cpol else self.v_l))
        mosi_seq.append(np.full(idle_len, self.v_l))
        miso_seq.append(np.full(idle_len, self.v_l))
        label_seq.append(np.full(idle_len, LABEL_MAP['IDLE']))
        
        # 2. CS 拉低 (Start Frame)
        cs_setup_len = int(self.spb * 2)
        cs_seq.append(np.full(cs_setup_len, self.v_l)) # CS Active
        sclk_seq.append(np.full(cs_setup_len, self.v_h if self.cpol else self.v_l)) # SCLK Idle
        
        # CPHA=0 时，Data 在 CS 下降沿之后立刻 Setup
        mosi_val = 0 # 初始值
        miso_val = 0
        mosi_seq.append(np.full(cs_setup_len, self.v_l))
        miso_seq.append(np.full(cs_setup_len, self.v_l))
        label_seq.append(np.full(cs_setup_len, LABEL_MAP['CS_ACTIVE']))
        
        # 3. 数据传输
        num_bytes = np.random.randint(self.config['min_bytes'], self.config['max_bytes'] + 1)
        
        for b_idx in range(num_bytes):
            val_mosi = np.random.randint(0, 256)
            val_miso = np.random.randint(0, 256)
            events.append({'type': 'MOSI', 'value': val_mosi})
            
            for i in range(7, -1, -1):
                bit_mosi = (val_mosi >> i) & 1
                bit_miso = (val_miso >> i) & 1
                
                # 模拟时钟抖动
                t_ph1 = int(self.spb * np.random.uniform(0.9, 1.1))
                t_ph2 = int(self.spb * np.random.uniform(0.9, 1.1))
                
                # 构造电平
                v_clk_idle = self.v_h if self.cpol else self.v_l
                v_clk_active = self.v_l if self.cpol else self.v_h
                
                v_mosi = self.v_h if bit_mosi else self.v_l
                v_miso = self.v_h if bit_miso else self.v_l
                
                l_mosi = LABEL_MAP['DATA_1'] if bit_mosi else LABEL_MAP['DATA_0']
                
                # 关键：根据 CPHA 生成时序
                if self.cpha == 0:
                    # CPHA=0: Sample @ 1st Edge (Leading), Change @ 2nd Edge (Trailing)
                    # Data must be ready BEFORE Leading Edge
                    
                    # Phase 1: Leading Edge occurs (Sample happens here)
                    # SCLK goes Active
                    sclk_bit = np.concatenate([np.full(t_ph1, v_clk_active), np.full(t_ph2, v_clk_idle)])
                    
                    # Data is stable throughout Phase 1, changes at start of Phase 2 (Trailing Edge)
                    # 但对于下一个 bit，setup 是在 phase 2 做的。
                    # 这里简化：整个周期保持数据稳定，或者在中间切换
                    # CPHA=0 意味着数据在 SCLK 变高之前就已经准备好了
                    mosi_bit = np.full(t_ph1 + t_ph2, v_mosi)
                    miso_bit = np.full(t_ph1 + t_ph2, v_miso)
                    
                else: # CPHA=1
                    # CPHA=1: Change @ 1st Edge (Leading), Sample @ 2nd Edge (Trailing)
                    
                    # Phase 1: Leading Edge (Data Changes)
                    # SCLK goes Active
                    sclk_bit = np.concatenate([np.full(t_ph1, v_clk_active), np.full(t_ph2, v_clk_idle)])
                    
                    # Data changes at the beginning of Phase 1
                    mosi_bit = np.full(t_ph1 + t_ph2, v_mosi)
                    miso_bit = np.full(t_ph1 + t_ph2, v_miso)
                
                # 拼接
                sclk_seq.append(sclk_bit)
                mosi_seq.append(mosi_bit)
                miso_seq.append(miso_bit)
                cs_seq.append(np.full(len(sclk_bit), self.v_l))
                label_seq.append(np.full(len(sclk_bit), l_mosi))

        # 4. CS 拉高 (End Frame)
        cs_hold = int(self.spb * 2)
        cs_seq.append(np.full(cs_hold, self.v_h))
        sclk_seq.append(np.full(cs_hold, self.v_h if self.cpol else self.v_l))
        mosi_seq.append(np.full(cs_hold, self.v_l))
        miso_seq.append(np.full(cs_hold, self.v_l))
        label_seq.append(np.full(cs_hold, LABEL_MAP['IDLE']))
        
        # 合并
        sclk = np.concatenate(sclk_seq)
        mosi = np.concatenate(mosi_seq)
        miso = np.concatenate(miso_seq)
        cs = np.concatenate(cs_seq)
        labels = np.concatenate(label_seq)
        
        # 添加损伤
        # 1. 串扰 (SCLK -> MOSI/MISO)
        mosi = self._add_crosstalk(mosi, sclk)
        miso = self._add_crosstalk(miso, sclk)
        
        # 2. 高斯噪声
        sclk = self._add_noise(sclk)
        mosi = self._add_noise(mosi)
        miso = self._add_noise(miso)
        cs = self._add_noise(cs)
        
        return sclk, mosi, miso, cs, labels, events

    def generate_spi_datasets(self, num_datasets=100, samples_per_dataset=10000):
        # 4通道: CH0=CS, CH1=SCLK, CH2=MOSI, CH3=MISO (默认映射)
        all_datasets = np.zeros((num_datasets, samples_per_dataset, 4), dtype=np.float32)
        all_labels = np.zeros((num_datasets, samples_per_dataset), dtype=np.int64)
        all_events = []
        all_maps = []
        
        for i in range(num_datasets):
            sclk, mosi, miso, cs, lbl, evt = self.generate_transaction()
            
            # 截断或填充
            L = min(len(sclk), samples_per_dataset)
            
            # 映射到 4 通道 (这里可以随机打乱通道顺序，增强泛化)
            # 假设标准顺序: CS, SCLK, MOSI, MISO
            all_datasets[i, :L, 0] = cs[:L]
            all_datasets[i, :L, 1] = sclk[:L]
            all_datasets[i, :L, 2] = mosi[:L]
            all_datasets[i, :L, 3] = miso[:L]
            
            # 空白处填充 IDLE 电平
            if L < samples_per_dataset:
                all_datasets[i, L:, 0] = self.v_h # CS Idle High
                all_datasets[i, L:, 1] = self.v_h if self.cpol else self.v_l
                
            all_labels[i, :L] = lbl[:L]
            all_events.append(evt)
            all_maps.append((0,1,2,3)) # 记录通道映射
            
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
