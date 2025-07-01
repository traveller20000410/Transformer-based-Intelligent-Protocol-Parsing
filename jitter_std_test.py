import numpy as np

# 设置抖动标准差和每个比特的采样点数
jitter_std = 0.04
samples_per_bit = 10

# 初始化计数器
count_0 = 0
count_1 = 0
count_neg_1 = 0

# 循环测试1000次
for _ in range(10000):
    # 调用代码生成jitter_samples
    jitter_samples = int(np.random.normal(0, jitter_std * samples_per_bit))

    # 统计每个数字的出现次数
    if jitter_samples == 0:
        count_0 += 1
    elif jitter_samples == 1:
        count_1 += 1
    elif jitter_samples == -1:
        count_neg_1 += 1

# 打印统计结果
print(f"Number of 0s: {count_0}")
print(f"Number of 1s: {count_1}")
print(f"Number of -1s: {count_neg_1}")
