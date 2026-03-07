import i2c_data_gen_one_frame as I2C_data_generator;            import uart as UART_data_generator;
import spi_data_gen as SPI_data_generator;
import torch;   import os;                                      import json;
from scipy import stats;                                        import torch.nn.functional as F
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
CURRENT_PROTOCOL = 'uart' 
RESUME_TRAINING = False
DATA_CACHE_PATH = f"cached_data_{CURRENT_PROTOCOL}.npz" # 缓存文件区分协议

#生成协议数据
def generate_protocols_dataset(protocol_type, num_datasets=None):
    if protocol_type == 'uart':
        print(f"Generating {num_datasets} UART samples...")
        gen = UART_data_generator.RealisticUARTSignalGenerator(config=UART_data_generator.DEFAULT_UART_CONFIG)
        # UART 生成器返回: data, labels, events, maps
        return gen.generate_uart_datasets(num_datasets)
        
    elif protocol_type == 'i2c':
        print(f"Generating {num_datasets} I2C samples...")
        gen = I2C_data_generator.RealisticI2CSignalGenerator(config=I2C_data_generator.DEFAULT_I2C_CONFIG)
        # I2C 生成器返回: data, labels, events, maps
        return gen.generate_i2c_datasets(num_datasets)
        
    elif protocol_type == 'spi':
        print(f"Generating {num_datasets} SPI samples...")
        gen = SPI_data_generator.RealisticSPISignalGenerator(config=SPI_data_generator.DEFAULT_SPI_CONFIG)
        # SPI 生成器返回: data, labels, events, maps
        return gen.generate_spi_datasets(num_datasets)
    
    else:
        raise ValueError(f"Unknown protocol type: {protocol_type}")
        
def get_label_map(protocol_type):
    if protocol_type == 'uart':
        return UART_data_generator.LABEL_MAP
    elif protocol_type == 'i2c':
        return I2C_data_generator.LABEL_MAP
    elif protocol_type == 'spi':
        return SPI_data_generator.LABEL_MAP
    else:
        raise ValueError("Unknown protocol")

def train_transformer_model(mode='teacher', num_datasets=3000):
    current_label_map = get_label_map(CURRENT_PROTOCOL)
    output_dim = len(current_label_map)
    if RESUME_TRAINING and os.path.exists(DATA_CACHE_PATH):
        print("[main.py] 加载之前缓存的数据...")
        cached = np.load(DATA_CACHE_PATH, allow_pickle=False)
        processed_dataset = cached["data"]
        processed_labels = cached["labels"]
    else:
        print("[main.py] 重新生成并处理数据...")
        # 生成协议数据与标签
        protocols_dataset, protocol_labels, events, channel_maps = generate_protocols_dataset(CURRENT_PROTOCOL, num_datasets)
        print(f"[main.py] Shape after generation: {protocols_dataset.shape}")
        #预处理
        processed_dataset,processed_labels=preprocess_dataset(protocols_dataset, protocol_labels)
        print(f"[main.py] Shape after downsampling: {processed_dataset.shape}")

        # 导出给 C# ONNX 推理使用：每条样本一个 (L,4) float32 .bin
        (processed_dataset,out_dir=f"csharp_onnx_data_{CURRENT_PROTOCOL}", prefix=CURRENT_PROTOCOL, label_map=current_label_map,norm_meta={"type": "minmax", "per_channel": True, "range": [-0.2, 1.5]})
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
        output_dim = len(current_label_map)

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

        state_dict = torch.load(teacher_checkpoint_path, map_location='cpu')
        if any(k.startswith('_orig_mod.') for k in state_dict.keys()):
            print("[main.py] 检测到 'torch.compile' 权重，正在清理 '_orig_mod.' 前缀...")
            new_state_dict = {k[len('_orig_mod.'):]: v for k, v in state_dict.items() if k.startswith('_orig_mod.')}
        else:
            print("[main.py] 未检测到 'torch.compile' 权重，正常加载。")
            new_state_dict = state_dict # 保持原样

        # 加载清理后的 state_dict # 我们使用 strict=True，因为 best_transformer_model.pth 应该只包含模型权重
        try:
            teacher_model.load_state_dict(new_state_dict, strict=True)
        except RuntimeError as e:
            print(f"严格加载 (strict=True) 失败: {e}")
            print("尝试非严格加载 (strict=False)...")
            teacher_model.load_state_dict(new_state_dict, strict=False)

        print("[main.py] 教师模型加载成功。")

        #  启动学生模型的蒸馏训练 将 *教师模型实例* 和数据一起传递给蒸馏函数
        GQA_train_model_distill(teacher_model, processed_dataset, processed_labels)

def save_for_csharp_onnx(data_4ch_norm: np.ndarray, out_dir: str, prefix: str, label_map: dict, norm_meta: dict | None = None):
    os.makedirs(out_dir, exist_ok=True)
    N, L, C = data_4ch_norm.shape
    assert C == 4, f"expect 4 channels, got {C}"

    # 反转字典: {0: "IDLE", 1: "START", ...}
    class_map = {int(v): k for k, v in label_map.items()}

    manifest = {
        "version": 1,
        "num_samples": int(N),
        "length": int(L),
        "num_channels": int(C),
        "dtype": "float32",
        "layout": "LxC row-major",
        "class_map": class_map,
        "protocol": prefix, # 记录协议类型
        "norm": norm_meta or {"type":"minmax","per_channel":True,"range":[0.0,1.0]},
        "files": []
    }

    for i in range(min(N, 20)): # 只保存前20个做演示，别存几千个
        x = np.ascontiguousarray(data_4ch_norm[i].astype(np.float32))
        x_path = os.path.join(out_dir, f"{prefix}_{i:04d}.bin")
        x.tofile(x_path)
        manifest["files"].append({"x": os.path.basename(x_path), "shape": [int(L), int(C)]})

    with open(os.path.join(out_dir, "manifest.json"), "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f">>> Saved {min(N, 20)} samples to '{out_dir}' with {len(class_map)} classes.")

def preprocess_dataset(dataset, labels, target_length=1250, normalize=True,v_low=0.0, v_high=3.3):
    data_tensor = torch.from_numpy(dataset).float().permute(0, 2, 1)
    
    labels_tensor = torch.from_numpy(labels).float().unsqueeze(1)
    resampled_data = F.interpolate(data_tensor, size=target_length, mode='linear', align_corners=False)

    resampled_labels = F.interpolate(labels_tensor, size=target_length, mode='nearest')
    resampled_data = resampled_data.permute(0, 2, 1).numpy()
    resampled_labels = resampled_labels.squeeze(1).long().numpy()

    if normalize:
        v_low_arr  = np.array(v_low,  dtype=np.float32).reshape(1, 1, -1) if np.ndim(v_low)  else np.full((1,1,dataset.shape[2]), v_low,  dtype=np.float32)
        v_high_arr = np.array(v_high, dtype=np.float32).reshape(1, 1, -1) if np.ndim(v_high) else np.full((1,1,dataset.shape[2]), v_high, dtype=np.float32)
        denom = (v_high_arr - v_low_arr)
        denom[denom == 0] = 1e-6
        resampled_data = (resampled_data - v_low_arr) / denom
        #resampled_data = np.clip(resampled_data, 0.0, 1.0)
        resampled_data = np.clip(resampled_data, -0.2, 1.5)

    return resampled_data.astype(np.float32), resampled_labels.astype(np.int64)


def main():
    # 训练教师模型
    train_transformer_model(mode='teacher', num_datasets=3000)

    # 蒸馏学生模型
    # train_transformer_model(mode='student', num_datasets=3000)

if __name__ == "__main__":
    main()
