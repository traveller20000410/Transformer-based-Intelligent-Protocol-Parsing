import numpy as np
import pandas as pd


class RS232Labeler:
    def __init__(self, voltage_threshold=1.6):
        """
        初始化RS232标注器
        Parameters:
        voltage_threshold: 高低电平判定阈值
        """
        self.voltage_threshold = voltage_threshold
        self.labels = {
            'NO_SIGNAL': 0,  # 非有效信号位
            'START': 1,  # 起始位
            'DATA_0': 2,  # 数据位0
            'DATA_1': 3,  # 数据位1
            'STOP': 4  # 停止位
        }

    def detect_frame_boundaries(self, voltages, samples_per_symbol):
        # 将电压值转换为二进制序列（高电平 = 1，低电平 = 0）
        binary = np.array(voltages) > self.voltage_threshold
        binary = binary.astype(int)

        # 检测下降沿（可能的起始位开始）
        edges = np.where(np.diff(binary) < 0)[0]

        frames = []
        prev_frame_end = 0
        for edge in edges:
            if edge < prev_frame_end :
                continue
            # 根据实际每个符号采样点数判断是否有足够的采样点来包含一个完整的帧
            # 这里假设起始位1个，数据位是8位，停止位1个，共10个符号
            if edge + samples_per_symbol * 10 <= len(voltages)-10*samples_per_symbol:  #留出末尾的一部分作为无信号位，方便测试使用
                frame_start = edge
                frame_end_pointer  = edge+10*samples_per_symbol   #帧位置结束指针
                while voltages[frame_end_pointer+1] == 0:
                    frame_end_pointer += 10*samples_per_symbol

                frames.append((frame_start, frame_end_pointer,'data'))
                # 标记帧间空闲位
                if prev_frame_end < frame_start:
                    frames.append((prev_frame_end, frame_start,'no_signal'))
                prev_frame_end = frame_end_pointer

        # 标记最后的无信号位
        if prev_frame_end < len(voltages):
            frames.append((prev_frame_end, len(voltages),'no_signal'))
        return frames

    def label_frame(self, voltages, frame_start, samples_per_symbol):
        # 对单个RS232帧进行标注
        frame_labels = np.zeros(len(voltages))
        # 标注起始位
        start_bit = slice(frame_start, frame_start + samples_per_symbol)
        frame_labels[start_bit] = self.labels['START']

        # 标注8个数据位
        for i in range(8):
            bit_start = frame_start + (i + 1) * samples_per_symbol
            bit_end = bit_start + samples_per_symbol
            bit_slice = slice(bit_start, bit_end)

            # 通过采样中点的电平值判断数据位的值
            mid_point = (bit_start + bit_end) // 2
            if mid_point < len(voltages):
                is_high = voltages[mid_point] > self.voltage_threshold
                frame_labels[bit_slice] = self.labels['DATA_1'] if is_high else self.labels['DATA_0']

        # 标注停止位
        stop_bit_start = frame_start + 9 * samples_per_symbol
        stop_bit_end = stop_bit_start + samples_per_symbol
        if stop_bit_end <= len(voltages):
            frame_labels[slice(stop_bit_start, stop_bit_end)] = self.labels['STOP']

        return frame_labels

    def label_sequence(self, voltages, samples_per_symbol):
        """
        标注整个电压序列
        """
        # 初始化标签序列
        sequence_labels = np.zeros(len(voltages))

        # 检测所有帧的边界
        frames = self.detect_frame_boundaries(voltages, samples_per_symbol)

        # 标注每一帧
        for frame_start, frame_end, hint_label in frames:
            if hint_label=='data':
                # 正常帧的标注
                frame_labels = self.label_frame(voltages, frame_start, samples_per_symbol)
                sequence_labels[frame_start:frame_end] = frame_labels[frame_start:frame_end]
            elif hint_label =='no_signal':
                # 帧间空闲位或无信号位的标注
                sequence_labels[frame_start:frame_end] = self.labels['NO_SIGNAL']

        return sequence_labels

    def create_training_data(self, voltages, labels):
        """
        创建用于Transformer训练的数据集
        """
        return {
            'voltage': voltages,
            'labels': labels,
            'label_map': {v: k for k, v in self.labels.items()}
        }