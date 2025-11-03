# 文件名: transformer_GQA_Student.py

import torch, torchcrf, torch.onnx, torch.nn as nn, torch.nn.functional as F, os
from torch.utils.data import Dataset, DataLoader;
from sklearn.model_selection import train_test_split
import numpy as np;
from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import time;
import math;
from transformer_component import weighted_cross_entropy_loss, check_pth_is_accessible, export_to_onnx, loss_result_plt, \
    preprocess_data
from joblib import load;
# 移除了 checkpoint
from sklearn.metrics import classification_report;
from sklearn.utils.class_weight import compute_class_weight
from torch.cuda.amp import autocast, GradScaler;
# 移除了 xformers 和 dynamo
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

# --- 学生模型的超参数 ---
# 定义超参数，包括批量大小、训练轮次、学习率等
BATCH_SIZE =        256;                        EPOCHS =        500   # 蒸馏可能需要更多/更少时间，取决于收敛情况
LEARNING_RATE =     0.0002;                     D_MODEL =       64
NUM_HEADS =         4;                          NUM_LAYERS =    6
DROPOUT =           0.1;                        MAX_LENGTH =    1250
NUM_GROUPS =        2 ;                         PATIENCE=       20;   # 蒸馏训练可能需要更多耐心
INITIAL_ALPHA =     0.5;                        FINAL_ALPHA =   0.1
ALPHA_DECAY_EPOCHS = EPOCHS * 0.7               #SCHEDULER_PATIENCE=15;

# --- 知识蒸馏超参数 ---
KD_ALPHA = 0.7;  # 知识蒸馏损失的权重 (70% 损失来自蒸馏)
TEMPERATURE = 2.0;  # 蒸馏温度，用于平滑教师的输出


# --- 从教师模型复制的组件 (无修改) ---

class ProtocolTensorDataset(Dataset):
    def __init__(self, data_tensor, labels_tensor):
        assert data_tensor.size(0) == labels_tensor.size(0)
        self.data_tensor = data_tensor
        self.labels_tensor = labels_tensor

    def __getitem__(self, index):
        return self.data_tensor[index], self.labels_tensor[index]

    def __len__(self):
        return self.data_tensor.size(0)


class RotaryEmbedding(nn.Module):
    def __init__(self, dim, max_position_embeddings=MAX_LENGTH):
        super().__init__()
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        positions = torch.arange(max_position_embeddings, dtype=torch.float32)
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)
        self.register_buffer("cos_cached", torch.cos(sinusoid_inp).half())
        self.register_buffer("sin_cached", torch.sin(sinusoid_inp).half())

    def forward(self, seq_len):
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin


def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, L, Dh]
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    q_rot = torch.cat([q1 * cos - q2 * sin,
                       q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin,
                       k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


class FeedForward(nn.Module):
    def __init__(self, d_model, d_ff_multiplier=4, dropout=DROPOUT):
        super(FeedForward, self).__init__()
        d_ff = d_model * d_ff_multiplier
        self.linear1 = nn.Linear(d_model, d_ff)
        self.activation = FastGELU()
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.activation(self.linear1(x))
        x = self.dropout(x)
        x = self.linear2(x)
        return x


# --- 修改后的学生模型组件 ---

class GroupedQueryAttention(nn.Module):
    """
    学生版的 GQA，使用 PyTorch 内置的 SDPA 替换 xformers。
    这对于 CPU 推理至关重要。
    """

    def __init__(self, d_model, num_heads, num_groups, dropout=DROPOUT,
                 max_position_embeddings=MAX_LENGTH):
        super().__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        kv_dim = num_groups * self.head_dim

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, kv_dim)
        self.W_v = nn.Linear(d_model, kv_dim)
        self.W_o = nn.Linear(d_model, d_model)

        self.rotary_emb = RotaryEmbedding(self.head_dim, max_position_embeddings)
        self.attn_dropout = dropout

    def forward(self, query, key, value, mask=None):
        B, L, _ = query.size()

        # 1. 独立投影 Q, K, V
        q = self.W_q(query).view(B, L, self.num_heads, self.head_dim)
        k = self.W_k(key).view(B, L, self.num_groups, self.head_dim)
        v = self.W_v(value).view(B, L, self.num_groups, self.head_dim)

        # 2. GQA的核心：为K和V复制头数
        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=2)

        # 3. 调整形状以适应 SDPA (B, H, L, Dh)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)  # 教师代码中这里是 reshape

        # 4. 应用旋转位置编码 (RoPE)
        cos, sin = self.rotary_emb(L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)  # q, k 仍是 (B, H, L, Dh)

        # 5. **核心修改**: 调用 PyTorch 2.0+ 的 scaled_dot_product_attention (CPU / GPU 通用)
        out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=None,  # 我们的序列是定长的，不需要 mask
            dropout_p=self.attn_dropout if self.training else 0.0
        )

        # 6. reshape和输出投影
        out = out.transpose(1, 2).reshape(B, L, self.d_model)  # (B, H, L, Dh) -> (B, L, H, Dh) -> (B, L, D_MODEL)
        return self.W_o(out)

class RMSNormQwen3(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))  # ✅ 改为ones
        self.eps = eps

    def forward(self, x: torch.Tensor):
        rms = torch.sqrt(torch.mean(x.float() ** 2, dim=-1, keepdim=True) + self.eps)
        return (x / rms * self.weight).type_as(x)  # ✅ 直接乘weight

class FastGELU(nn.Module):
    def __init__(self):
        super().__init__()
    def forward(self, x):
        return 0.5 * x * (1.0 + torch.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))
        
class TransformerEncoderLayer(nn.Module):
    """
    学生版的 EncoderLayer，移除了梯度检查点。
    """
    def __init__(self, d_model, num_heads, dropout, num_groups):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = GroupedQueryAttention(d_model, num_heads, num_groups, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.norm1 = RMSNormQwen3(d_model)
        self.norm2 = RMSNormQwen3(d_model)

    def forward(self, x, mask=None):
        # **核心修改**: 移除 checkpoint(custom_forward, ...)
        # 直接执行前向传播
        norm_x = self.norm1(x)
        attn_output = self.attention(norm_x, norm_x, norm_x, mask)
        x = x + attn_output  # 第一个残差连接

        norm_x2 = self.norm2(x)
        ffn_output = self.ffn(norm_x2)
        x = x + ffn_output  # 第二个残差连接
        return x


# --- 从教师模型复制的模型主干 (无修改) ---

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout, num_groups):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(d_model, num_heads, dropout, num_groups) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:  x = layer(x, mask)
        return x


class TransformerModel(nn.Module):
    def __init__(self, output_dim, max_length, d_model, num_heads, num_layers, dropout, num_groups):
        super(TransformerModel, self).__init__()
        self.input_projection = nn.Linear(4, d_model)  # 假设输入还是 4 通道
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dropout, num_groups)
        self.fc = nn.Linear(d_model, output_dim)
        self.crf = torchcrf.CRF(output_dim, batch_first=True)

    def forward(self, x):
        x_features = self.input_projection(x)
        mask = torch.ones(x_features.shape[0], x_features.shape[1], dtype=torch.bool, device=x.device)
        encoded_output = self.encoder(x_features, mask=mask)
        emissions = self.fc(encoded_output)
        return emissions, mask


# --- 新的知识蒸馏训练函数 ---

def train_distill(model, teacher_model, train_loader, optimizer, device, scaler,
                  weight_tensor=None, alpha=0.5, kd_alpha=0.5, temperature=2.0, do_profiling=False):
    """
    知识蒸馏的内部训练循环
    """
    model.train()  # 学生模型设为训练模式
    teacher_model.eval()  # 教师模型设为评估模式

    running_loss = 0.0
    running_hard_loss = 0.0
    running_kd_loss = 0.0
    total_correct_gpu = torch.tensor(0.0, device=device)
    total_samples_gpu = torch.tensor(0.0, device=device)

    cross_entropy_func = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)

    # 定义 KL 散度损失 (蒸馏损失)
    distill_loss_func = nn.KLDivLoss(reduction='batchmean')

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()

        # 1. 获取教师模型的输出 (不需要梯度)
        with torch.no_grad():
            teacher_emissions, teacher_mask = teacher_model(data)

        # 2. 正常执行学生模型的前向传播
        with torch.amp.autocast('cuda'):
            student_emissions, mask = model(data)

            # --- 3. 计算 "Hard Loss" (学生 vs 真实标签) ---
            loss_crf = -model.crf(student_emissions, target, mask=mask)

            emissions_flat = student_emissions.view(-1, student_emissions.shape[-1])
            target_flat = target.view(-1)
            active_loss_mask = mask.view(-1) == 1

            active_student_emissions = emissions_flat[active_loss_mask]
            active_targets = target_flat[active_loss_mask]

            loss_ce = cross_entropy_func(active_student_emissions, active_targets)
            loss_hard = loss_crf + alpha * loss_ce

            # --- 4. 计算 "Soft Loss" (学生 vs 教师) ---
            # 找到教师输出的对应部分
            active_teacher_emissions = teacher_emissions.view(-1, teacher_emissions.shape[-1])[active_loss_mask]

            # 使用温度 T 平滑输出
            soft_student_logits = F.log_softmax(active_student_emissions / temperature, dim=-1)
            soft_teacher_probs = F.softmax(active_teacher_emissions / temperature, dim=-1)

            # T^2 是为了让损失的梯度尺度与 hard loss 匹配
            loss_kd = (temperature ** 2) * distill_loss_func(soft_student_logits, soft_teacher_probs)

            # --- 5. 组合损失 ---
            # (1-kd_alpha) * HardLoss + kd_alpha * SoftLoss
            loss = (1.0 - kd_alpha) * loss_hard + kd_alpha * loss_kd

        # 反向传播
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
        running_hard_loss += loss_hard.item()
        running_kd_loss += loss_kd.item()

        # 计算精度 (基于学生的 "硬" 预测)
        with torch.no_grad():
            proxy_predicted = torch.argmax(student_emissions, dim=-1)
            active_predictions = proxy_predicted[mask]
            total_correct_gpu += (active_predictions == active_targets).sum()
            total_samples_gpu += active_targets.numel()

    final_accuracy = (total_correct_gpu / total_samples_gpu).item() if total_samples_gpu > 0 else 0.0

    return (running_loss / len(train_loader),
            running_hard_loss / len(train_loader),
            running_kd_loss / len(train_loader),
            final_accuracy)


# --- 验证函数 (从教师模型复制过来，用于评估) ---
def test(model, test_loader, device, scaler, weight_tensor=None, alpha=0.5):
    model.eval()
    running_loss = 0.0
    total_correct_gpu = torch.tensor(0.0, device=device)
    total_samples_gpu = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            with torch.amp.autocast('cuda'):
                emissions, mask = model(data)

                # 在验证时，我们只关心 "Hard Loss"
                loss_crf = -model.crf(emissions, target, mask=mask)
                emissions_flat = emissions.view(-1, emissions.shape[-1])
                target_flat = target.view(-1)
                active_loss_mask = mask.view(-1) == 1
                active_emissions = emissions_flat[active_loss_mask]
                active_targets = target_flat[active_loss_mask]

                cross_entropy_func = nn.CrossEntropyLoss(weight=weight_tensor, label_smoothing=0.1)
                loss_ce = cross_entropy_func(active_emissions, active_targets)
                loss = loss_crf + alpha * loss_ce

                running_loss += loss.item()

                # 计算精度
                predicted = model.crf.decode(emissions, mask=mask)
                predicted_flat_cpu = [p for sublist in predicted for p in sublist]
                if not predicted_flat_cpu:  continue
                predicted_flat_gpu = torch.tensor(predicted_flat_cpu, device=device)

                total_correct_gpu += (predicted_flat_gpu == active_targets).sum()
                total_samples_gpu += active_targets.numel()

    final_accuracy = (total_correct_gpu / total_samples_gpu).item() if total_samples_gpu > 0 else 0.0
    return running_loss / len(test_loader), final_accuracy


# --- 新的知识蒸馏训练主函数 ---

def train_model_distill(teacher_model, protocols_dataset, protocol_labels):
    """
    知识蒸馏的外部训练循环
    """
    print("Converting list of arrays to a single large NumPy array...")
    all_data_np = np.array(protocols_dataset)
    all_labels_np = np.array(protocol_labels)

    print("Splitting dataset into training, validation, and test sets...")
    indices = np.arange(all_data_np.shape[0])
    train_val_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=42, shuffle=True)

    x_train_np, y_train_np = all_data_np[train_indices], all_labels_np[train_indices]
    x_val_np, y_val_np = all_data_np[val_indices], all_labels_np[val_indices]
    x_test_np, y_test_np = all_data_np[test_indices], all_labels_np[test_indices]

    del all_data_np, all_labels_np, protocols_dataset, protocol_labels
    import gc
    gc.collect()

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    all_train_labels_flat = y_train_np.flatten()
    label_encoder.fit(all_train_labels_flat)
    class_weights = compute_class_weight('balanced', classes=np.unique(all_train_labels_flat), y=all_train_labels_flat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Converting NumPy arrays to PyTorch Tensors and pinning memory...")
    x_train_tensor = torch.from_numpy(x_train_np).float().pin_memory()
    y_train_tensor = torch.from_numpy(y_train_np).long().pin_memory()
    x_val_tensor = torch.from_numpy(x_val_np).float().pin_memory()
    y_val_tensor = torch.from_numpy(y_val_np).long().pin_memory()
    x_test_tensor = torch.from_numpy(x_test_np).float().pin_memory()
    y_test_tensor = torch.from_numpy(y_test_np).long().pin_memory()

    del x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np
    gc.collect()

    train_dataset = ProtocolTensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = ProtocolTensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = ProtocolTensorDataset(x_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                             persistent_workers=True)

    # --- 1. 冻结教师模型 ---
    print(f"Freezing Teacher model and moving to {device}...")
    teacher_model.to(device)
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False
    print("Teacher model frozen.")

    # --- 2. 初始化学生模型 ---
    output_dim = len(label_encoder.classes_)
    # 使用学生模型的超参数
    model = TransformerModel(output_dim, MAX_LENGTH, D_MODEL, NUM_HEADS, NUM_LAYERS, DROPOUT, NUM_GROUPS)
    # model = torch.compile(model) # torch.compile() 在学生模型上也是一个好主意
    model = model.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = CosineAnnealingLR(optimizer, T_max=EPOCHS)

    checkpoint_path = 'student_checkpoint.pth'
    best_model_path = 'best_student_model.pth'
    best_val_loss = float('inf');
    start_epoch = 0
    counter = 0
    train_losses_per_epoch = [];
    val_losses_per_epoch = [];

    # (您可以添加与教师模型类似的检查点加载逻辑)
    if os.path.exists(checkpoint_path):
        print(f"--- Resuming student training from checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, weights_only=False, map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])
        model = model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        train_losses_per_epoch = checkpoint.get('train_losses', [])
        val_losses_per_epoch = checkpoint.get('val_losses', [])
        print(f"--- Resumed student from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f} ---")
    else:
        print("--- No student checkpoint found, starting distillation from scratch. ---")

    print("=" * 60)
    print(" " * 15 + "Starting Student Distillation")
    print("=" * 60)
    print(f"{'--- Student Architecture ---':^60}")
    print(f"{'D_MODEL':<25}: {D_MODEL}")
    print(f"{'NUM_LAYERS':<25}: {NUM_LAYERS}")
    print(f"{'NUM_HEADS':<25}: {NUM_HEADS}")
    print(f"{'NUM_GROUPS (for GQA)':<25}: {NUM_GROUPS}")
    print(f"\n{'--- Distillation Params ---':^60}")
    print(f"{'KD_ALPHA (Distill %)':<25}: {KD_ALPHA}")
    print(f"{'TEMPERATURE':<25}: {TEMPERATURE}")
    print(f"{'HARD_LOSS_ALPHA (CE %)':<25}: {INITIAL_ALPHA} -> {FINAL_ALPHA}")
    print("=" * 60 + "\n")

    start_time = time.time()

    for epoch in range(start_epoch, EPOCHS):
        if epoch < ALPHA_DECAY_EPOCHS:
            current_alpha = INITIAL_ALPHA - (INITIAL_ALPHA - FINAL_ALPHA) * (epoch / ALPHA_DECAY_EPOCHS)
        else:
            current_alpha = FINAL_ALPHA

        train_loss, hard_loss, kd_loss, train_acc = train_distill(
            model, teacher_model, train_loader, optimizer, device, scaler,
            class_weights_tensor, alpha=current_alpha, kd_alpha=KD_ALPHA, temperature=TEMPERATURE
        )

        val_loss, val_acc = test(model, val_loader, device, scaler,
                                 weight_tensor=class_weights_tensor, alpha=current_alpha)

        elapsed_time = time.time() - start_time
        scheduler.step()

        print(
            f"/**** Epoch {epoch + 1}, LR: {optimizer.param_groups[0]['lr']:.2e}, Alpha(Hard): {current_alpha:.3f} ****/")
        print(f"Train Loss: {train_loss:.4f} (Hard: {hard_loss:.4f}, KD: {kd_loss:.4f})")
        print(f"Train Acc: {train_acc:.4f}, Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        print(f"Time elapsed: {elapsed_time:.2f} seconds")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            torch.save(model.state_dict(), best_model_path)
            print(f"New best val_loss: {best_val_loss:.4f}, saved {best_model_path}")
        else:
            counter += 1
            if counter >= PATIENCE:
                print("Early stopping")
                break

        if epoch % 5 == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),
                'best_val_loss': best_val_loss,
                'label_encoder': label_encoder
            }, checkpoint_path)
            print(f"已保存学生模型检查点到 {checkpoint_path}")

    print("Student distillation training complete.")
    return model, label_encoder
