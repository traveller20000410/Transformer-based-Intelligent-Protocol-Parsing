import numpy as np
import matplotlib.pyplot as plt
from scipy.fft import fft, fftfreq

# 生成时钟信号数据
def generate_clock_signal(length):
    clock_signal = np.zeros(length)
    period = 100  # 假设时钟周期为100个数据点，可根据需要调整
    for i in range(length):
        if i % period < period // 2:
            clock_signal[i] = 1
    return clock_signal

# 绘制FFT频谱图
def plot_fft(signal, sampling_rate):
    N = len(signal)
    yf = fft(signal)
    xf = fftfreq(N, 1 / sampling_rate)[:N // 2]

    plt.figure(figsize=(12, 6))
    plt.plot(xf, 2.0 / N * np.abs(yf[0:N // 2]))
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Amplitude')
    plt.title('FFT Spectrum of Clock Signal')
    plt.grid()
    plt.show()

if __name__ == "__main__":
    sampling_rate = 1000  # 假设采样率为1000Hz，可根据需要调整
    clock_signal_data = generate_clock_signal(10000)
    plot_fft(clock_signal_data, sampling_rate)