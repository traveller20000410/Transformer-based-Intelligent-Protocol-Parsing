import i2c_data_gen_one_frame as I2C_data_generator;            #import I2C_data_generator_mutliframe;
import torch;   import os;                                      #import RS232_data_generator;
#import common8b10b_data_generator;                             #import matplotlib.pyplot as plt
from scipy import stats                                         # from transformer_MLA import train_model as MLA_train_model,predict_protocol,load_model
from transformer_GQA_Teacher import train_model as GQA_train_model,predict_protocol,load_model
# from transformer_GQA_Student import trained_student_model
import numpy as np;                                             from universal_function import save_downsampled_csv
import pandas as pd;                                            from scipy import stats
from joblib import load;                                        from scipy.signal import resample

#定义
RESUME_TRAINING = False
DATA_CACHE_PATH = "cached_data.npz"

#生成协议数据
def generate_protocols_dataset(num_datasets=None):
    # #生成I2C协议数据与标签
    gen = I2C_data_generator.RealisticI2CSignalGenerator(config=I2C_data_generator.DEFAULT_I2C_CONFIG)
    protocols_dataset0, protocol_labels0,_,channel_maps = gen.generate_i2c_datasets(num_datasets)
    #生成RS232协议数据与标签
    # gen = RS232_data_generator.RealisticRS232SignalGenerator()
    # protocols_dataset1, protocol_labels1 = gen.generate_rs232_datasets(num_datasets)
    # #生成common8b10b协议数据与标签
    # gen = common8b10b_data_generator.RealisticCommon8b10bSignalGenerator()
    # protocols_dataset2, protocol_labels2 = gen.generate_common8b10b_datasets(num_datasets)
    # 合并数据和标签
    # protocols_dataset = protocols_dataset0 + protocols_dataset1+protocols_dataset2
    # protocol_labels = protocol_labels0 + protocol_labels1+protocol_labels2

    # # 创建索引数组
    # indices = np.arange(len(protocols_dataset))
    # np.random.shuffle(indices)
    #
    # # 创建新的空列表以存储打散后的数据和标签
    # shuffled_protocols_dataset = [None] * len(protocols_dataset)
    # shuffled_protocol_labels = [None] * len(protocol_labels)

    # 根据打散后的索引重新排列数据和标签
    # for i in range(len(protocols_dataset)):
    #     shuffled_protocols_dataset[i] = protocols_dataset[indices[i]]
    #     shuffled_protocol_labels[i] = protocol_labels[indices[i]]

    # return shuffled_protocols_dataset, shuffled_protocol_labels
    return protocols_dataset0, protocol_labels0,channel_maps


def train_transformer_model(num_datasets=None):
    if RESUME_TRAINING and os.path.exists(DATA_CACHE_PATH):
        print("[main.py] 加载之前缓存的数据...")
        cached = np.load(DATA_CACHE_PATH, allow_pickle=False)
        processed_dataset = cached["data"]
        processed_labels = cached["labels"]
    else:
        print("[main.py] 重新生成并处理数据...")
        # 生成协议数据与标签
        protocols_dataset, protocol_labels,channel_maps = generate_protocols_dataset(num_datasets)
        print(f"[main.py] Shape after generation: {protocols_dataset.shape}")  # 应该输出 [64, 10000, 2]
        #预处理
        processed_dataset,processed_labels=preprocess_dataset(protocols_dataset, protocol_labels)
        print(f"[main.py] Shape after downsampling: {processed_dataset.shape}")  # 应该输出 [64, 1250, 2]
        np.savez(DATA_CACHE_PATH, data=processed_dataset, labels=processed_labels)

        # 3) 导出下采样后的 SCL/SDA
        #export_scl_sda_from_4ch(data_4ch=processed_dataset,labels=processed_labels,maps=channel_maps,base_dir="downsampled_scl_sda",sampling_rate=processed_dataset.shape[1])

    #启动训练
    # MLA_train_model(processed_dataset, processed_labels)
    GQA_train_model(processed_dataset, processed_labels)


# def export_downsampled_waveforms(down_data, down_labels, base_dir="downsampled_i2c", sampling_rate=None):
#
#     os.makedirs(base_dir, exist_ok=True)
#     num_ds, L, _ = down_data.shape
#     # 如果你想用通用函数直接保存
#     try:
#         for i in range(num_ds):
#             # save_downsampled_csv 会把 (L,2) 的数据写成 CSV
#             save_downsampled_csv(
#                 down_data[i],                         # 波形 (SCL/SDA)
#                 down_labels[i],                       # 对应标签
#                 os.path.join(base_dir, f"ds_{i:03d}.csv"),
#                 fs=sampling_rate                     # 可选：传给它采样率
#             )
#         print(">>> downsampled CSVs saved via save_downsampled_csv()")
#         return
#     except NameError:
#         # 如果没有这个函数，再走下面的 pandas 路径
#         pass
#     # pandas 版本
#     for i in range(num_ds):
#         df = pd.DataFrame({
#             'Time_us': np.arange(L) / sampling_rate * 1e6 if sampling_rate else np.arange(L),
#             'SCL':      down_data[i, :, 0],
#             'SDA':      down_data[i, :, 1],
#             'Label':    down_labels[i]
#         })
#         path = os.path.join(base_dir, f"down_i2c_{i:03d}.csv")
#         df.to_csv(path, index=False)
#     print(f">>> downsampled CSVs saved under {base_dir}/")

def export_scl_sda_from_4ch(data_4ch: np.ndarray, labels: np.ndarray, maps: list[tuple[int,int]],  base_dir: str = "scl_sda_export",sampling_rate: float = None):
    os.makedirs(base_dir, exist_ok=True)
    N, L, C = data_4ch.shape
    assert C == 4, "输入必须是 4 通道"
    for i, (scl_ch, sda_ch) in enumerate(maps):
        scl = data_4ch[i, :, scl_ch]
        sda = data_4ch[i, :, sda_ch]
        lab = labels[i]
        # 可选地，生成 时间 列
        if sampling_rate:
            time_us = np.arange(L) / sampling_rate * 1e6
            df = pd.DataFrame({
                "Time_us": time_us,
                "SCL":      scl,
                "SDA":      sda,
                "Label":    lab,
            })
        else:
            df = pd.DataFrame({
                "SCL":   scl,
                "SDA":   sda,
                "Label": lab,
            })
        path = os.path.join(base_dir, f"ds_{i:03d}.csv")
        df.to_csv(path, index=False)
    print(f">>> 已导出 {N} 条仅含 SCL/SDA 的 CSV 到：{base_dir}/")


def preprocess_dataset(dataset, labels, target_length=1250):
    original_data = dataset
    original_labels = labels

    num_datasets = original_data.shape[0]
    original_length = original_data.shape[1]

    if original_length <= target_length:
        print(f"Warning: Original length {original_length} is <= target {target_length}. Skipping resampling.")
        return original_data.astype(np.float32), original_labels

    resampled_data = resample(original_data, target_length, axis=1)
    factor = original_length // target_length
    trimmed_labels = original_labels[:, :target_length * factor]
    reshaped_labels = trimmed_labels.reshape(num_datasets, target_length, factor)
    resampled_labels, _ = stats.mode(reshaped_labels, axis=2, keepdims=False)

    return resampled_data.astype(np.float32), resampled_labels.astype(np.int64)

# def generate_test_protocols_dataset(num_datasets=None):
#     # 生成测试数据以预测标签
#     gen = I2C_data_generator.RealisticI2CSignalGenerator(sampling_rate=1000000)
#     protocols_dataset1, protocol_labels0 = gen.generate_i2c_datasets(num_datasets=10)
#     gen = common8b10b_data_generator.RealisticCommon8b10bSignalGenerator()
#     protocols_dataset2, protocol_labels3 = gen.generate_common8b10b_datasets(num_datasets=10)
#     gen=RS232_data_generator.RealisticRS232SignalGenerator(sampling_rate=1000000)
#     protocols_dataset3, protocol_labels2 = gen.generate_rs232_datasets(num_datasets=10)
#     # 合并数据
#     protocols_dataset_fortest = protocols_dataset1 + protocols_dataset2+protocols_dataset3
#     return protocols_dataset_fortest

def test_model(flag=None,num_datasets=None):
    input_dim = 21
    output_dim = 5  # 假设输出维度为 5，根据你的具体情况修改
    d_model = 64
    num_heads = 4
    num_layers = 4
    dropout = 0.2
    num_groups = 2
    max_length = 1024
    # 加载模型
    model = load_model(input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout)
    label_encoder = load('label_encoder.joblib')
    # 调用测试函数
    predicted_protocol,original_protocol_label = test_with_sequence(model, label_encoder)
    original_protocol_label = original_protocol_label[0].tolist()
    # test_protocol_plt(original_protocol_label,predicted_protocol)
    print("Predicted protocol:", predicted_protocol)
    # model=load_model(input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout)
    # # print(model)
    # # 生成测试数据以预测标签
    # protocols_dataset_from_generator = generate_test_protocols_dataset()
    # protocols_dataset_from_Tek = import_tek_data()

    # if flag == "tek":
    #     protocols_dataset_fortest=protocols_dataset_from_Tek
    # else:
    #     protocols_dataset_fortest = protocols_dataset_from_generator
    # 输入数据以进行预测
    # predicted_protocols = []
    # for data in protocols_dataset_fortest:
    #     protocol = predict_protocol(model, data)
    #     predicted_protocols.append(protocol)
    #
    # # 打印预测的协议名称标签
    # for i, protocol in enumerate(predicted_protocols):
    #     print(f"Test data {i + 1}: Predicted protocol: {protocol}")


#导入外部数据
# def import_tek_data():
#     # 读取CSV文件
#     df1 = pd.read_csv('8b10b_data_ALL.csv', header=None)
#     all_data1 = df1.iloc[:, 0].values
#     data_list1 = [all_data1[i * 5000:(i + 1) * 5000] for i in range(len(all_data1) // 5000)]
#
#     df2 = pd.read_csv('RS232_data_ALL.csv', header=None)
#     all_data2 = df2.iloc[:, 0].values
#     data_list2 = [all_data2[i * 5000:(i + 1) * 5000] for i in range(len(all_data2) // 5000)]
#
#     data_list=data_list1+data_list2
#     # 对每个子数组进行归一化处理
#     normalized_data_list = []
#     for sub_array in data_list:
#         normalized_sub_array = normalize_data(sub_array)
#         normalized_data_list.append(normalized_sub_array)
#     return normalized_data_list

def test_with_sequence(model, label_encoder, sequence_length=1024):
    device = next(model.parameters()).device  # 获取模型所在的设备
    # 生成一个长度为 1024 的随机数据序列，假设数据为整数
    data,protocol_label=generate_protocols_dataset(num_datasets=1)
    processed_dataset, protocol_label = preprocess_dataset(data, protocol_label)
    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        emissions, mask = model(processed_dataset)
        predicted = model.crf.decode(emissions, mask=mask)[0]
    return predicted,protocol_label

# def test_protocol_plt(original_protocol_label,predicted_protocol):
#     if len(original_protocol_label) != len(predicted_protocol):
#         raise ValueError("original_protocol_label and predicted_protocol must have the same length.")
#     # 创建一个包含两个子图的图形，共享x轴
#     fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True, figsize=(10, 8))
#
#     # 绘制第一列数据（original_protocol_label）的图形，在上方的子图
#     ax1.plot(original_protocol_label, label='Original Protocol Label', color='orange')
#     ax1.set_xlabel('Index')
#     ax1.set_ylabel('Original Label Value')
#     ax1.set_title('Predicted Protocol vs Original Protocol Label')
#     ax1.legend()
#
#     # 绘制第二列数据（predicted_protocol）的图形，在下方的子图
#     ax2.plot(predicted_protocol, label='Predicted Protocol')
#     ax2.set_xlabel('Index')
#     ax2.set_ylabel('Predicted Label Value')
#     ax2.legend()
#     # 调整子图之间的间距等布局设置
#     plt.tight_layout()
#     # 显示图形
#     plt.show()

def main():
    train_transformer_model(num_datasets=1000)
    # test_model(flag="te",num_datasets=None)

if __name__ == "__main__":
    main()
