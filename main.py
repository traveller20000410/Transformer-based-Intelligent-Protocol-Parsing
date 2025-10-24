import i2c_data_gen_one_frame as I2C_data_generator;            #import I2C_data_generator_mutliframe;
import torch;   import os;                                      import json;
from scipy import stats                                         # from transformer_MLA import train_model as MLA_train_model,predict_protocol,load_model
import numpy as np;                                             from universal_function import save_downsampled_csv
import pandas as pd;                                            from scipy import stats
from joblib import load;                                        from scipy.signal import resample

# --- 导入教师和学生模型 ---
# 教师模型用于加载预训练权重
from transformer_GQA_Teacher import TransformerModel as TeacherTransformerModel
from transformer_GQA_Teacher import D_MODEL as TEACHER_D_MODEL
from transformer_GQA_Teacher import NUM_HEADS as TEACHER_NUM_HEADS
from transformer_GQA_Teacher import NUM_LAYERS as TEACHER_NUM_LAYERS
from transformer_GQA_Teacher import DROPOUT as TEACHER_DROPOUT
from transformer_GQA_Teacher import MAX_LENGTH as TEACHER_MAX_LENGTH
from transformer_GQA_Teacher import NUM_GROUPS as TEACHER_NUM_GROUPS
from transformer_GQA_Teacher import train_model as GQA_train_model # 导入教师模型的 *训练* 函数
# 导入学生模型的 *蒸馏训练* 函数
from transformer_GQA_Student import train_model_distill as GQA_train_model_distill

#定义
RESUME_TRAINING = False
DATA_CACHE_PATH = "cached_data.npz"

#生成协议数据
def generate_protocols_dataset(num_datasets=None):
    # #生成I2C协议数据与标签
    gen = I2C_data_generator.RealisticI2CSignalGenerator(config=I2C_data_generator.DEFAULT_I2C_CONFIG)
    protocols_dataset0, protocol_labels0,_,channel_maps = gen.generate_i2c_datasets(num_datasets)
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


def train_transformer_model(mode='teacher', num_datasets=70):
    if RESUME_TRAINING and os.path.exists(DATA_CACHE_PATH):
        print("[main.py] 加载之前缓存的数据...")
        cached = np.load(DATA_CACHE_PATH, allow_pickle=False)
        processed_dataset = cached["data"]
        processed_labels = cached["labels"]
    else:
        print("[main.py] 重新生成并处理数据...")
        # 生成协议数据与标签
        protocols_dataset, protocol_labels,channel_maps = generate_protocols_dataset(num_datasets)
        print(f"[main.py] Shape after generation: {protocols_dataset.shape}")
        #预处理
        processed_dataset,processed_labels=preprocess_dataset(protocols_dataset, protocol_labels)
        print(f"[main.py] Shape after downsampling: {processed_dataset.shape}")

        # 导出给 C# ONNX 推理使用：每条样本一个 (L,4) float32 .bin
        save_for_csharp_onnx(processed_dataset,out_dir="csharp_onnx_data",prefix="i2c",norm_meta={"type": "minmax", "per_channel": True, "range": [0.0, 1.0]})
        np.savez(DATA_CACHE_PATH, data=processed_dataset, labels=processed_labels)

        # 3) 导出下采样后的 SCL/SDA
        # export_scl_sda_from_4ch(data_4ch=processed_dataset,labels=processed_labels,maps=channel_maps,base_dir="downsampled_scl_sda",sampling_rate=processed_dataset.shape[1])

    # --- 流程控制 ---
    if mode == 'teacher':
        print("\n" + "#" * 30)
        print("###   开始训练【教师】模型   ###")
        print("#" * 30 + "\n")
        # 启动教师模型的训练
        GQA_train_model(processed_dataset, processed_labels)

    elif mode == 'student':
        print("\n" + "#" * 30)
        print("###  开始蒸馏【学生】模型  ###")
        print("#" * 30 + "\n")

        # 1. 定义教师模型架构
        print("[main.py] 正在加载教师模型...")
        # (确保这里的 16 是您教师模型的 output_dim)
        output_dim = len(I2C_data_generator.LABEL_MAP)  # 应该是 16

        teacher_model = TeacherTransformerModel(
            output_dim, TEACHER_MAX_LENGTH, TEACHER_D_MODEL, TEACHER_NUM_HEADS,
            TEACHER_NUM_LAYERS, TEACHER_DROPOUT, TEACHER_NUM_GROUPS
        )

        # 2. 加载训练好的教师模型权重
        teacher_checkpoint_path = 'best_transformer_model.pth'  # 假设这是您最好的教师模型
        if not os.path.exists(teacher_checkpoint_path):
            print(f"[ERROR] 教师模型权重 {teacher_checkpoint_path} 不存在!")
            print("请先运行 'teacher' 模式进行训练。")
            return

        # ------------------- [新的修复代码块] -------------------
        # 加载 state_dict 到 CPU
        state_dict = torch.load(teacher_checkpoint_path, map_location='cpu')
        
        # 自动检测并清理 '_orig_mod.' 前缀
        # 检查是否有任何键以此前缀开头
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("[main.py] 检测到 'torch.compile' 权重，正在清理 '_orig_mod.' 前缀...")
            # 使用字典推导式 (dictionary comprehension) 来创建新的 state_dict
            # 它会遍历所有键值对，如果键以 '_orig_mod.' 开头，就去掉这个前缀 (长度为 10)
            new_state_dict = {k[len('_orig_mod.'):]: v for k, v in state_dict.items() if k.startswith('_orig_mod.')}
        else:
            print("[main.py] 未检测到 'torch.compile' 权重，正常加载。")
            new_state_dict = state_dict # 保持原样

        # 加载清理后的 state_dict
        # 我们使用 strict=True，因为 best_transformer_model.pth 应该只包含模型权重
        try:
            teacher_model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            # 如果因为某种原因 (例如文件损坏或误用) 导致严格加载失败，我们尝试非严格加载
            print(f"严格加载 (strict=True) 失败: {e}")
            print("尝试非严格加载 (strict=False)...")
            teacher_model.load_state_dict(new_state_dict, strict=False)
        # ------------------- [修复代码块结束] -------------------

        print("[main.py] 教师模型加载成功。")

        # 3. 启动学生模型的蒸馏训练 将 *教师模型实例* 和数据一起传递给蒸馏函数
        GQA_train_model_distill(teacher_model, processed_dataset, processed_labels)


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

def save_for_csharp_onnx(data_4ch_norm: np.ndarray,out_dir: str = "csharp_onnx_data",prefix: str = "i2c",norm_meta: dict | None = None):
    os.makedirs(out_dir, exist_ok=True)
    N, L, C = data_4ch_norm.shape
    assert C == 4, f"expect 4 channels, got {C}"

    class_map = {int(v): k for k, v in I2C_data_generator.LABEL_MAP.items()}  # :contentReference[oaicite:3]{index=3}

    manifest = {
        "version": 1,
        "num_samples": int(N),
        "length": int(L),
        "num_channels": int(C),
        "dtype": "float32",
        "layout": "LxC row-major",
        "class_map": class_map,
        "norm": norm_meta or {"type":"minmax","per_channel":True,"range":[0.0,1.0]},
        "files": []
    }

    for i in range(N):
        x = np.ascontiguousarray(data_4ch_norm[i].astype(np.float32))  # (L,4)
        x_path = os.path.join(out_dir, f"{prefix}_{i:04d}.bin")
        x.tofile(x_path)
        manifest["files"].append({"x": os.path.basename(x_path), "shape": [int(L), int(C)]})

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f">>> Saved {N} samples to '{out_dir}' as float32 [0,1].")


def preprocess_dataset(dataset, labels, target_length=1250, normalize=True,
                       v_low=0.0, v_high=3.3):
    original_data = dataset
    original_labels = labels
    N, L, C = original_data.shape  # 例如 C=4

    # 1) 重采样 + 标签重采样
    if L <= target_length:
        print(f"Warning: Original length {L} is <= target {target_length}. Skipping resampling.")
        resampled_data = original_data.astype(np.float32)
        resampled_labels = original_labels
    else:
        from scipy.signal import resample
        resampled_data = resample(original_data, target_length, axis=1)
        factor = L // target_length
        trimmed_labels = original_labels[:, :target_length * factor]
        reshaped_labels = trimmed_labels.reshape(N, target_length, factor)
        from scipy import stats
        resampled_labels, _ = stats.mode(reshaped_labels, axis=2, keepdims=False)

    # 2) 固定范围归一化到 [0,1]（按通道广播）
    if normalize:
        import numpy as np
        v_low_arr  = np.array(v_low,  dtype=np.float32).reshape(1, 1, -1) if np.ndim(v_low)  else np.full((1,1,C), v_low,  dtype=np.float32)
        v_high_arr = np.array(v_high, dtype=np.float32).reshape(1, 1, -1) if np.ndim(v_high) else np.full((1,1,C), v_high, dtype=np.float32)
        # 防止除零
        denom = (v_high_arr - v_low_arr)
        denom[denom == 0] = 1e-6

        resampled_data = (resampled_data - v_low_arr) / denom
        resampled_data = np.clip(resampled_data, 0.0, 1.0)

    return resampled_data.astype(np.float32), resampled_labels.astype(np.int64)


# def test_model(flag=None,num_datasets=None):
#     input_dim = 21
#     output_dim = 5  # 假设输出维度为 5，根据你的具体情况修改
#     d_model = 64
#     num_heads = 4
#     num_layers = 4
#     dropout = 0.2
#     num_groups = 2
#     max_length = 1024
#     # 加载模型
#     model = load_model(input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout)
#     label_encoder = load('label_encoder.joblib')
#     # 调用测试函数
#     predicted_protocol,original_protocol_label = test_with_sequence(model, label_encoder)
#     original_protocol_label = original_protocol_label[0].tolist()
#     # test_protocol_plt(original_protocol_label,predicted_protocol)
#     print("Predicted protocol:", predicted_protocol)

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

# def test_with_sequence(model, label_encoder, sequence_length=1024):
#     device = next(model.parameters()).device  # 获取模型所在的设备
#     # 生成一个长度为 1024 的随机数据序列，假设数据为整数
#     data,protocol_label=generate_protocols_dataset(num_datasets=1)
#     processed_dataset, protocol_label = preprocess_dataset(data, protocol_label)
#     model.eval()  # 将模型设置为评估模式
#     with torch.no_grad():
#         emissions, mask = model(processed_dataset)
#         predicted = model.crf.decode(emissions, mask=mask)[0]
#     return predicted,protocol_label


def main():
    # 步骤 1: 训练教师模型
    train_transformer_model(mode='teacher', num_datasets=1000)

    # 步骤 2: 训练好教师后，注释掉上面一行，运行下面一行来蒸馏学生模型
    # train_transformer_model(mode='student', num_datasets=1000)

if __name__ == "__main__":
    main()
