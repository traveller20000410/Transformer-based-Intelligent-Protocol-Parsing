import torch,   torchcrf,   torch.onnx,    torch.nn as nn,      torch.nn.functional as F   ,os
from torch.utils.data import Dataset, DataLoader;               from sklearn.model_selection import train_test_split
import numpy as np;     from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import time;      import math;                                  from transformer_component import weighted_cross_entropy_loss,check_pth_is_accessible,export_to_onnx,loss_result_plt,preprocess_data
from joblib import load;                                        from torch.utils.checkpoint import checkpoint
from sklearn.metrics import classification_report;              from sklearn.utils.class_weight import compute_class_weight # 导入一个方便的工具
from torch.cuda.amp import autocast, GradScaler;                import xformers.ops as xops
from xformers.ops.fmha import attn_bias
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

# 定义超参数，包括批量大小、训练轮次、学习率等
BATCH_SIZE =        128;                        EPOCHS =        500
LEARNING_RATE =     0.0001;                     D_MODEL =       128
NUM_HEADS =         8;                          NUM_LAYERS =    8
DROPOUT =           0.1;                        MAX_LENGTH =    1250
NUM_GROUPS =        2 ;                         PATIENCE=       20;
INITIAL_ALPHA =     1.0;                        FINAL_ALPHA =   0.2
ALPHA_DECAY_EPOCHS = EPOCHS * 0.7               #SCHEDULER_PATIENCE=15;

#缓存数据类
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
        sinusoid_inp = torch.einsum("i,j->ij", positions, inv_freq)  # [L, d/2]
        self.register_buffer("cos_cached", torch.cos(sinusoid_inp).half()) # 直接用 half 存
        self.register_buffer("sin_cached", torch.sin(sinusoid_inp).half())

    def forward(self, seq_len):
        # 返回 [1, 1, L, d/2] 形式的 cos/sin
        cos = self.cos_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        sin = self.sin_cached[:seq_len].unsqueeze(0).unsqueeze(0)
        return cos, sin

def apply_rotary_pos_emb(q, k, cos, sin):
    # q, k: [B, H, L, Dh], cos/sin: [1,1,L, Dh/ head?] 注意Dh=head_dim
    # 先把最后一维一分为二
    q1, q2 = q[..., ::2], q[..., 1::2]
    k1, k2 = k[..., ::2], k[..., 1::2]
    # 交替旋转
    q_rot = torch.cat([q1 * cos - q2 * sin,
                       q1 * sin + q2 * cos], dim=-1)
    k_rot = torch.cat([k1 * cos - k2 * sin,
                       k1 * sin + k2 * cos], dim=-1)
    return q_rot, k_rot


class GroupedQueryAttention(nn.Module):
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

    def forward(self, query, key, value, mask=None):  # mask参数现在实际上没用了，但保留以兼容接口
        B, L, _ = query.size()

        # 1. 独立投影 Q, K, V
        q = self.W_q(query).view(B, L, self.num_heads, self.head_dim)
        k = self.W_k(key).view(B, L, self.num_groups, self.head_dim)
        v = self.W_v(value).view(B, L, self.num_groups, self.head_dim)

        # 2. GQA的核心：为K和V复制头数
        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=2)
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=2)

        # 3. 应用旋转位置编码 (RoPE)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)
        cos, sin = self.rotary_emb(L)
        q, k = apply_rotary_pos_emb(q, k, cos, sin)
        q = q.transpose(1, 2)
        k = k.transpose(1, 2)

        # v也需要调整形状以匹配xformers的输入
        v = v.reshape(B, L, self.num_heads, self.head_dim)

        # 4. **核心修复**: 对于无padding的定长序列，我们不需要attn_bias    # 直接调用xformers，它会执行全局注意力
        out = xops.memory_efficient_attention(
            q, k, v,
            p=self.attn_dropout if self.training else 0.0,
            attn_bias=None,  # <--- 明确地传入None
        )

        # 5. reshape和输出投影
        out = out.reshape(B, L, self.d_model)
        return self.W_o(out)


# 前馈网络类
class FeedForward(nn.Module):
        def __init__(self, d_model, d_ff_multiplier=4, dropout=DROPOUT):
            super(FeedForward, self).__init__()
            d_ff = d_model * d_ff_multiplier  #transformer里一般FFN取D_MODEL的4倍
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)
            self.dropout = nn.Dropout(dropout)

        def forward(self, x):
            x = F.gelu(self.linear1(x))
            x = self.dropout(x)
            x = self.linear2(x)
            return x

class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, num_groups):
        super(TransformerEncoderLayer, self).__init__()
        # 将dropout率传递给Attention层
        self.attention = GroupedQueryAttention(d_model, num_heads, num_groups, dropout=dropout)
        self.ffn = FeedForward(d_model, dropout=dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, mask=None):
        def custom_forward(x):
            norm_x = self.norm1(x)
            attn_output = self.attention(norm_x, norm_x, norm_x, mask)
            # MODIFIED: 直接进行残差连接, 因为dropout已在attention内部处理
            x = x + attn_output

            norm_x2 = self.norm2(x)
            ffn_output = self.ffn(norm_x2)
            x = x + ffn_output
            return x
        x = checkpoint(custom_forward, x, use_reentrant=False)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout, num_groups):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dropout, num_groups) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:  x = layer(x, mask)
        return x

class TransformerModel(nn.Module):
    def __init__(self, output_dim, max_length, d_model, num_heads, num_layers, dropout, num_groups):
        super(TransformerModel, self).__init__()

        self.input_projection = nn.Linear(4, d_model)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dropout, num_groups)
        self.fc = nn.Linear(d_model, output_dim)
        self.crf = torchcrf.CRF(output_dim, batch_first=True)  # 添加 CRF 层

    def forward(self, x):

        x_features = self.input_projection(x)
        mask = torch.ones(x_features.shape[0], x_features.shape[1], dtype=torch.bool, device=x.device)
        encoded_output = self.encoder(x_features, mask=mask)
        emissions = self.fc(encoded_output)

        return emissions, mask

# 训练函数，包括前向传播、计算损失、反向传播和参数更新
def train(model, train_loader, optimizer, device,scaler,scheduler,weight_tensor=None, alpha=0.5, do_profiling=False):
    model.train()
    running_loss = 0.0
    #  使用GPU进行精度计算
    total_correct_gpu = torch.tensor(0.0, device=device)
    total_samples_gpu = torch.tensor(0.0, device=device)

    #可选，性能检测
    if do_profiling:
        prof = profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=schedule(wait=1, warmup=1, active=1),
            on_trace_ready=tensorboard_trace_handler('./log'),
            record_shapes=True,
            with_stack=True
        )
        prof.__enter__()

    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        # 混合精度训练
        with torch.amp.autocast('cuda'):
            emissions,mask = model(data)  # (batch_size, seq_len, num_classes)
            #计算组合损失
            loss_crf = -model.crf(emissions, target, mask=mask)
            emissions_flat = emissions.view(-1, emissions.shape[-1])
            target_flat = target.view(-1)
            active_loss_mask = mask.view(-1) == 1
            active_emissions = emissions_flat[active_loss_mask]
            active_targets = target_flat[active_loss_mask]
            cross_entropy_func = nn.CrossEntropyLoss(weight=weight_tensor)
            loss_ce = cross_entropy_func(active_emissions, active_targets)
            loss = loss_crf + alpha * loss_ce

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        #性能分析
        if do_profiling:
            prof.step()

        running_loss += loss.item()

        with torch.no_grad():
            predicted = model.crf.decode(emissions, mask=mask)
            predicted_flat_cpu = [p for sublist in predicted for p in sublist]
            if not predicted_flat_cpu:    continue
            predicted_flat_gpu = torch.tensor(predicted_flat_cpu, device=device)
            total_correct_gpu += (predicted_flat_gpu == active_targets).sum()
            total_samples_gpu += active_targets.numel()

    final_accuracy = (total_correct_gpu / total_samples_gpu).item() if total_samples_gpu > 0 else 0.0

    if do_profiling:
        prof.__exit__(None, None, None)
        # 输出 profiling 结果
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return running_loss / len(train_loader), final_accuracy


def train_model(protocols_dataset, protocol_labels):
    print("Converting list of arrays to a single large NumPy array...")
    try:
        all_data_np = np.array(protocols_dataset)  # 预期形状: [N, 10000, 2]
        all_labels_np = np.array(protocol_labels)  # 预期形状: [N, 1250]
        assert all_data_np.ndim == 3, "Data should be a 3D array."
        assert all_labels_np.ndim == 2, "Labels should be a 2D array."
        print(f"Conversion successful. Data shape: {all_data_np.shape}, Labels shape: {all_labels_np.shape}")
    except ValueError as e:
        print(
            "\n[ERROR] Failed to convert list to NumPy array. This usually happens if the arrays in the list have different shapes.")
        print("Please ensure all generated data samples have the same length.")
        print(f"Original error: {e}")
        # 找出第一个形状不匹配的样本
        first_shape_data = protocols_dataset[0].shape
        for i, arr in enumerate(protocols_dataset):
            if arr.shape != first_shape_data:
                print(f"Shape mismatch found at index {i}: expected {first_shape_data}, got {arr.shape}")
                break
        return None, None  # 提前退出

    # 直接在NumPy数组上进行高效的 train/val/test 分割
    print("Splitting dataset into training, validation, and test sets...")
    indices = np.arange(all_data_np.shape[0])
    train_val_indices, test_indices = train_test_split(indices, test_size=0.15, random_state=42, shuffle=True)
    train_indices, val_indices = train_test_split(train_val_indices, test_size=0.1, random_state=42, shuffle=True)

    # 用索引来获取分割后的数据，这比直接分割数据本身更高效
    x_train_np, y_train_np = all_data_np[train_indices], all_labels_np[train_indices]
    x_val_np, y_val_np = all_data_np[val_indices], all_labels_np[val_indices]
    x_test_np, y_test_np = all_data_np[test_indices], all_labels_np[test_indices]

    # 释放原始大数组的内存
    del all_data_np, all_labels_np, protocols_dataset, protocol_labels
    import gc
    gc.collect()
    print("Splitting complete and original data memory released.")

    from sklearn.preprocessing import LabelEncoder
    label_encoder = LabelEncoder()
    # 这里的 all_train_labels 来自 y_train_np，而不是原始的 y_train 列表
    all_train_labels_flat = y_train_np.flatten()
    label_encoder.fit(all_train_labels_flat)  # 确保编码器被fit

    class_weights = compute_class_weight('balanced', classes=np.unique(all_train_labels_flat), y=all_train_labels_flat)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)

    print("Converting NumPy arrays to PyTorch Tensors...")
    # 1. 一次性将所有分割好的数据转换为PyTorch张量
    x_train_tensor = torch.from_numpy(x_train_np).float()
    y_train_tensor = torch.from_numpy(y_train_np).long()

    x_val_tensor = torch.from_numpy(x_val_np).float()
    y_val_tensor = torch.from_numpy(y_val_np).long()

    x_test_tensor = torch.from_numpy(x_test_np).float()
    y_test_tensor = torch.from_numpy(y_test_np).long()

    # 2. 释放不再需要的NumPy数组内存，以防万一
    del x_train_np, y_train_np, x_val_np, y_val_np, x_test_np, y_test_np
    import gc
    gc.collect()
    print("Tensors created and NumPy memory released.")

    # 3. 使用新的高效Dataset类来创建数据集实例
    train_dataset = ProtocolTensorDataset(x_train_tensor, y_train_tensor)
    val_dataset = ProtocolTensorDataset(x_val_tensor, y_val_tensor)
    test_dataset = ProtocolTensorDataset(x_test_tensor, y_test_tensor)
    print("High-performance TensorDatasets created.")

    # 4. DataLoader的定义保持不变，它现在包裹的是我们高效的Dataset
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True,
                              persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                            persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True,
                             persistent_workers=True)

    # 初始化模型、损失函数和优化器
    output_dim = len(label_encoder.classes_)  #16
    model = TransformerModel(output_dim, MAX_LENGTH, D_MODEL, NUM_HEADS, NUM_LAYERS, DROPOUT,NUM_GROUPS)
    # #多卡并行计算
    # if torch.cuda.device_count() > 1:
    #     print(f"Let's use {torch.cuda.device_count()} GPUs!")
    #     model = torch.nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-5)
    scaler = torch.amp.GradScaler('cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

    checkpoint_path = 'checkpoint.pth' # 中断时保存的检查点路径
    best_model_path = 'best_transformer_model.pth'
    best_val_loss = float('inf');       start_epoch = 0
    counter = 0  # 记录验证集损失不下降的次数
    train_losses_per_epoch = [];        val_losses_per_epoch = [];     test_losses_per_epoch = []
    # model,optimizer,scaler,start_epoch,best_val_loss=check_pth_is_accessible(checkpoint_path,model, optimizer, scaler,device)  #加载保存的模型

    #检查是否有已经存在的模型存档
    if os.path.exists(checkpoint_path):
        print(f"--- Resuming training from checkpoint: {checkpoint_path} ---")
        checkpoint = torch.load(checkpoint_path, weights_only=False,map_location='cpu')
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scaler.load_state_dict(checkpoint['scaler_state_dict'])

        # 确保所有状态移动到当前设备
        model = model.to(device)
        for state in optimizer.state.values():
            for k, v in state.items():
                if isinstance(v, torch.Tensor):
                    state[k] = v.to(device)

        # 恢复之前的学习率
        if 'scheduler_state_dict' in checkpoint:
            scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint.get('epoch', -1) + 1  # 使用.get()更安全
        best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        # 恢复历史损失记录，以便绘图
        train_losses_per_epoch = checkpoint.get('train_losses', [])
        val_losses_per_epoch = checkpoint.get('val_losses', [])
        test_losses_per_epoch = checkpoint.get('test_losses', [])

        print(f"--- Resumed from epoch {start_epoch}, best_val_loss: {best_val_loss:.4f} ---")
    else:
        print("--- No checkpoint found, starting training from scratch. ---")

    # 训练和测试模型
    model.to(device);
    start_time = time.time()
    torch.backends.cudnn.benchmark = True
    try:
        for epoch in range(start_epoch, EPOCHS):
            #do_profile = (epoch == start_epoch)
            do_profile = False
            # 计算当前epoch的alpha值
            if epoch < ALPHA_DECAY_EPOCHS:
                current_alpha = INITIAL_ALPHA - (INITIAL_ALPHA - FINAL_ALPHA) * (epoch / ALPHA_DECAY_EPOCHS)
            else:
                current_alpha = FINAL_ALPHA
            train_loss, train_acc = train(model, train_loader, optimizer, device, scaler,scheduler,class_weights_tensor,alpha=current_alpha, do_profiling=do_profile)
            val_loss, val_acc = test(model, val_loader, device, scaler,weight_tensor=class_weights_tensor, alpha=current_alpha)  # 验证集评估
            test_loss, test_acc = test(model, test_loader, device, scaler)
            elapsed_time = time.time() - start_time
            train_losses_per_epoch.append(train_loss); val_losses_per_epoch.append(val_loss); test_losses_per_epoch.append(test_loss)
            scheduler.step()  # 更新学习率
            print(f'/****Epoch {epoch + 1}, Learning Rate: {optimizer.param_groups[0]["lr"]}****/, Alpha: {current_alpha:.3f}****/ ')
            print(f'Epoch {epoch + 1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print(f"Time elapsed: {elapsed_time:.2f} seconds")

            # —— 每 20 轮打印一次分类报告 —— #
            if (epoch + 1) % 20 == 0:
                # 在验证集上跑一次完整预测
                all_preds, all_trues = [], []
                model.eval()
                with torch.no_grad():
                    for data, target in test_loader:
                        data, target = data.to(device), target.to(device)
                        emissions, mask = model(data)
                        preds = model.crf.decode(emissions, mask=mask)
                        # 展平所有非 pad 标签
                        for p_seq, t_seq, m in zip(preds, target, mask):
                            length = m.sum().item()
                            all_preds.extend(p_seq[:length])
                            all_trues.extend(t_seq[:length].cpu().tolist())
                # 打印分类报告
                str_names = label_encoder.classes_.astype(str).tolist()
                print(f"\n=== Classification Report at Epoch {epoch + 1} ===")
                print(classification_report(all_trues, all_preds, target_names=str_names,zero_division=0))
                print("=== End Report ===\n")
                model.train()

            # 早停机制与学习率衰减机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss;      counter = 0;     plateau_counter=0;
                torch.save(model.state_dict(), 'best_transformer_model.pth')
                print(f"New best val_loss: {best_val_loss:.4f}, saved best_transformer.pth")
            else:
                counter += 1;   #plateau_counter+=1;
                #学习率衰减
                # if plateau_counter >= SCHEDULER_PATIENCE:
                #     for group in optimizer.param_groups:
                #         group['lr'] *= decay_factor
                #     scheduler.base_lrs = [base_lr * decay_factor for base_lr in scheduler.base_lrs]
                #     plateau_counter = 0
                #     print(f"Plateau! lr 和 base_lrs 均乘以 {decay_factor}")
                #早停
                if counter >= PATIENCE:
                    print("Early stopping")
                    break

            # 保存检查点
            if epoch % 5 == 0:  # 每5个epoch保存一次
                torch.save({'epoch': epoch,                 'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(), 'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),       'best_val_loss': best_val_loss,
                'label_encoder': label_encoder,                 'train_losses': train_losses_per_epoch,
                'val_losses': val_losses_per_epoch,             'test_losses': test_losses_per_epoch
                }, checkpoint_path);                             print(f"已保存检查点到 {checkpoint_path}")

    except KeyboardInterrupt:
        print("训练中断，保存当前检查点...")
        # torch.save({'epoch': epoch,                     'model_state_dict': model.state_dict(),
        # 'optimizer_state_dict': optimizer.state_dict(),   # 'scheduler_state_dict': scheduler.state_dict(),
        # 'scaler_state_dict': scaler.state_dict(),            'best_val_loss': best_val_loss
        # }, checkpoint_path)
        # print(f"已保存检查点到 {checkpoint_path}");            print("退出训练。")

        return model, label_encoder  # 或者您可以选择重新抛出异常

    # 在训练集上评估
    train_predictions = [];             train_true_labels = []
    model.to(device)
    with torch.no_grad():
        for data, target in train_loader:
            data, target = data.to(device), target.to(device)
            emissions, mask = model(data)
            predicted = model.crf.decode(emissions, mask=mask)
            for pred_seq, tgt_seq, m in zip(predicted, target, mask):
                # 仅考虑非填充部分
                seq_len = m.sum().item()
                train_predictions.extend(pred_seq[:seq_len])
                train_true_labels.extend(tgt_seq[:seq_len].cpu().numpy())
    print("Training set classification report:")
    print(classification_report(train_true_labels, train_predictions,target_names=label_encoder.classes_.astype(str).tolist(),zero_division=0))
    # 在测试集上评估
    test_predictions = [];              test_true_labels = []
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            emissions, mask = model(data)
            predicted = model.crf.decode(emissions, mask=mask)
            for pred_seq, tgt_seq, m in zip(predicted, target, mask):
                seq_len = m.sum().item()
                test_predictions.extend(pred_seq[:seq_len])
                test_true_labels.extend(tgt_seq[:seq_len].cpu().numpy())
    print("Test set classification report:")
    str_names = [str(c) for c in label_encoder.classes_]
    print(classification_report(test_true_labels,test_predictions,target_names=str_names,zero_division=0))
    loss_result_plt(train_losses_per_epoch, val_losses_per_epoch, test_losses_per_epoch) #画出损失函数曲线

    return model, label_encoder

# 测试函数，计算损失和准确率
def test(model, test_loader, device, scaler, weight_tensor=None, alpha=0.5):
    model.eval()
    running_loss = 0.0
    # --- 初始化用于GPU计数的张量 ---
    total_correct_gpu = torch.tensor(0.0, device=device)
    total_samples_gpu = torch.tensor(0.0, device=device)

    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            with torch.amp.autocast('cuda'):
                emissions, mask = model(data)

                # --- 为了和训练loss可比，这里也计算组合损失 ---
                loss_crf = -model.crf(emissions, target, mask=mask)
                emissions_flat = emissions.view(-1, emissions.shape[-1])
                target_flat = target.view(-1)
                active_loss_mask = mask.view(-1) == 1
                active_emissions = emissions_flat[active_loss_mask]
                active_targets = target_flat[active_loss_mask]

                cross_entropy_func = nn.CrossEntropyLoss(weight=weight_tensor)
                loss_ce = cross_entropy_func(active_emissions, active_targets) if weight_tensor is not None else 0
                loss = loss_crf + alpha * loss_ce if weight_tensor is not None else loss_crf
                # --- 结束组合损失计算 ---

                running_loss += loss.item()
                #GPU精度计算
                predicted = model.crf.decode(emissions, mask=mask)
                predicted_flat_cpu = [p for sublist in predicted for p in sublist]
                if not predicted_flat_cpu:  continue
                predicted_flat_gpu = torch.tensor(predicted_flat_cpu, device=device)
                # `active_targets` 已经存在于GPU上
                total_correct_gpu += (predicted_flat_gpu == active_targets).sum()
                total_samples_gpu += active_targets.numel()

    final_accuracy = (total_correct_gpu / total_samples_gpu).item() if total_samples_gpu > 0 else 0.0

    return running_loss / len(test_loader), final_accuracy

def predict_protocol(model, data):
    device = next(model.parameters()).device
    model.eval()
    if isinstance(data, np.ndarray):
        data = torch.from_numpy(data)
    data = data.long().unsqueeze(0).to(device)
    with torch.no_grad():
        emissions, mask = model(data)
        predicted = model.crf.decode(emissions, mask=mask)[0]
        label_encoder = load('label_encoder.joblib')
        predicted_protocol = label_encoder.inverse_transform(predicted)
    return predicted_protocol[:len(data[0])]


def predict_with_model(model, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_protocol = predict_protocol(model, test_data)
    return predicted_protocol

def load_model( output_dim, max_length, d_model, num_heads, num_layers, dropout):
    model = TransformerModel(output_dim, max_length, d_model, num_heads, num_layers, dropout,num_groups=NUM_GROUPS)
    # model.load_state_dict(torch.load('checkpoint.pth'))
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])  # 仅加载模型权重
    model.eval()
    return model
