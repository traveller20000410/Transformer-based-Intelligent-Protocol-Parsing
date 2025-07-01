import torch
import torch.nn.functional as F
import os
from sklearn.preprocessing import LabelEncoder
import numpy as np
from sklearn.model_selection import train_test_split
from joblib import dump # 用来保存label_encoder

def weighted_cross_entropy_loss(output, target):
    weight = torch.tensor([1.0, 2.0, 1.0, 1.0, 2.0]).to(output.device)
    loss = F.cross_entropy(output, target, weight=weight)
    return loss

def loss_result_plt(train_losses_per_epoch, val_losses_per_epoch,test_losses_per_epoch):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses_per_epoch, label='Train Loss')
    plt.plot(val_losses_per_epoch, label='Val Loss')
    plt.plot(test_losses_per_epoch, label='Test Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Loss Curves')
    plt.legend()
    plt.grid(True)
    plt.show()

def check_pth_is_accessible(checkpoint_path,model,optimizer,scaler,device):
    # 检查是否存在检查点
    start_epoch = 0
    best_val_loss = float('inf')
    if os.path.exists(checkpoint_path):
        print("加载检查点...")
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        # scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_val_loss = checkpoint['best_val_loss']
        # 确保模型和优化器状态转移到 GPU
        model.to(device)  # 将模型转移到正确的设备
        for state in optimizer.state.values():
            if isinstance(state, dict):  # 如果优化器的状态是字典，检查并转移其中的张量
                for k, v in state.items():
                    if isinstance(v, torch.Tensor):
                        state[k] = v.to(device)
        print(f"从 epoch {start_epoch} 开始继续训练。")
    return model,optimizer,scaler,start_epoch,best_val_loss

# 将模型转换为 ONNX 格式
def export_to_onnx(device,model, d_model, max_length, output_filename):
    # if not os.path.exists(output_dir):
    #     os.makedirs(output_dir)
    model.eval() ;                      model.to("cpu")
    # 创建一个虚拟输入，假设输入形状为 (1, max_length) 并且数据类型为 torch.long
    dummy_input = torch.zeros(1, max_length,dtype=torch.long)
    dummy_input = dummy_input.to("cpu")
    torch.onnx.export(model, dummy_input, output_filename, verbose=True,
                    input_names=['input'], output_names=['emissions','mask'],opset_version=11 )
    print(f"Model has been exported to {output_filename}")


def preprocess_data(protocols_dataset, protocol_labels, max_len=1000, test_size=0.1, random_state=42):
    print(f"[preprocess_data] Input shape: Data {protocols_dataset.shape}, Labels {protocol_labels.shape}")

    # 直接使用传入的数据和标签，不再做任何长度对齐 ---
    aligned_data = protocols_dataset
    aligned_labels_raw = protocol_labels

    # --- 2. 对齐整的标签进行编码 ---
    label_encoder = LabelEncoder()
    label_encoder.fit(aligned_labels_raw.ravel())
    print(f"[preprocess_data] Learned label classes: {label_encoder.classes_}")
    dump(label_encoder, 'label_encoder.joblib')

    aligned_labels_encoded = np.apply_along_axis(label_encoder.transform, 1, aligned_labels_raw)

    print("[preprocess_data] Label encoding complete.")

    # --- 3. 划分数据集 ---
    x_train, x_test, y_train, y_test = train_test_split(
        aligned_data,
        aligned_labels_encoded,
        test_size=test_size,
        random_state=random_state
    )
    print("[preprocess_data] Dataset splitting complete.")

    return x_train, x_test, y_train, y_test, label_encoder