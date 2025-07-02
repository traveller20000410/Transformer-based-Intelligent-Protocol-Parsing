import torch,   torchcrf,   torch.onnx,    torch.nn as nn,      torch.nn.functional as F   ,os
from torch.utils.data import Dataset, DataLoader;               from sklearn.model_selection import train_test_split
import numpy as np;     from torch.profiler import profile, ProfilerActivity, schedule, tensorboard_trace_handler
import time;                                                    from transformer_component import weighted_cross_entropy_loss,check_pth_is_accessible,export_to_onnx,loss_result_plt,preprocess_data
from joblib import load;                                        from torch.utils.checkpoint import checkpoint
from sklearn.metrics import classification_report;              from sklearn.utils.class_weight import compute_class_weight # 导入一个方便的工具
from torch.cuda.amp import autocast, GradScaler
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

# 定义超参数，包括批量大小、训练轮次、学习率等
BATCH_SIZE =        24;                         EPOCHS =        600
LEARNING_RATE =     0.0001;                     D_MODEL =       32
NUM_HEADS =         4;                          NUM_LAYERS =    4
DROPOUT =           0.1 ;                       MAX_LENGTH =    1250
NUM_GROUPS =        2 ;                         PATIENCE=       30;
SCHEDULER_PATIENCE=15;

#缓存数据类
class CachedDataset(Dataset):
    def __init__(self, cached_data):
        self.cached_data = cached_data

    def __len__(self):
        return len(self.cached_data)

    def __getitem__(self, idx):
        return self.cached_data[idx]

class RelativePositionBias(nn.Module):
    def __init__(self, num_heads, max_distance=128):
        super(RelativePositionBias, self).__init__()
        self.num_heads = num_heads
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(2 * self.max_distance + 1, self.num_heads)

    def forward(self, seq_length):
        # seq_length 是序列的长度 (L)     # 创建位置索引 [0, 1, ..., L-1]
        q_pos = torch.arange(seq_length, dtype=torch.long)
        k_pos = torch.arange(seq_length, dtype=torch.long)
        relative_position = k_pos[None, :] - q_pos[:, None]
        # 将相对位置裁剪到 [-max_distance, max_distance] 范围内
        clipped_relative_position = torch.clamp(relative_position, -self.max_distance, self.max_distance)
        # 将裁剪后的位置映射到嵌入表的正索引 [0, 2 * max_distance]
        indices = clipped_relative_position + self.max_distance
        bias = self.relative_attention_bias(indices.to(next(self.parameters()).device))
        bias = bias.permute(2, 0, 1).unsqueeze(0)
        return bias

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        # 确保分组数量能够整除注意力头数
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        kv_dim = self.num_groups * self.head_dim
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, kv_dim)  # 键变换（分组共享）
        self.W_v = nn.Linear(d_model, kv_dim)  # 值变换（分组共享）
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

    def forward(self, query, key, value, mask=None,pos_bias=None):
        batch_size = query.size(0)
        seq_len = query.size(1)

        # 线性变换
        q = self.W_q(query)  # (batch_size, seq_len, d_model)
        k = self.W_k(key)    # (batch_size, seq_len, d_model // num_groups)
        v = self.W_v(value)  # (batch_size, seq_len, d_model // num_groups)
        # 重塑为多头形式
        q = q.view(batch_size,seq_len, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size,seq_len, self.num_groups, self.head_dim).transpose(1, 2)  # (batch_size, num_groups, seq_len, head_dim)
        v = v.view(batch_size,seq_len, self.num_groups, self.head_dim).transpose(1, 2)  # (batch_size, num_groups, seq_len, head_dim)
        # 广播键和值到每个查询组
        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=1)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=1)  # (batch_size, num_heads, seq_len, head_dim)
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))

        # 将相对位置偏置项直接加到Attention分数上
        if pos_bias is not None:
            scores += pos_bias

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)  # 扩展 mask 维度
            scores = scores.masked_fill(mask.to(torch.bool) == 0, -1e4)  # 应用 mask
        # 计算注意力权重
        attn_weights = F.softmax(scores, dim=-1)
        # 计算注意力输出
        attn_output = torch.matmul(attn_weights, v)  # (batch_size, num_heads, seq_len, head_dim)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)  # (batch_size, seq_len, d_model)
        # 输出变换
        output = self.W_o(attn_output)
        return output, attn_weights

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

# Transformer 编码器层类
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, num_groups,relative_position_bias):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = GroupedQueryAttention(d_model, num_heads, num_groups)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model);     self.norm2 = nn.LayerNorm(d_model)
        self.dropout_attn = nn.Dropout(dropout)
        self.relative_position_bias = relative_position_bias

    def forward(self, x, mask=None):
        def custom_forward(x):
            # 先归一化，再进Attention
            norm_x = self.norm1(x)
            pos_bias = self.relative_position_bias(norm_x.size(1))
            attn_output, _ = self.attention(norm_x, norm_x, norm_x, mask, pos_bias=pos_bias)
            x = x + self.dropout_attn(attn_output)
            norm_x2 = self.norm2(x)
            ffn_output = self.ffn(norm_x2)
            x = x + ffn_output
            return x

        x = checkpoint(custom_forward, x, use_reentrant=False)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout, num_groups, relative_position_bias):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dropout, num_groups, relative_position_bias) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:  x = layer(x, mask)
        return x

class TransformerModel(nn.Module):
    def __init__(self, output_dim, max_length, d_model, num_heads, num_layers, dropout, num_groups):
        super(TransformerModel, self).__init__()

        # CNN Frontend
        self.cnn_frontend = nn.Sequential(
            # 输入: [B, 2, 10000]
            nn.Conv1d(in_channels=2, out_channels=d_model // 2, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm1d(d_model // 2),  # BatchNorm有助于CNN稳定训练
            nn.GELU(),
            # 输出: [B, d_model/2, 5000]
            nn.Conv1d(in_channels=d_model // 2, out_channels=d_model, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(d_model),
            nn.GELU(),
            # 输出: [B, d_model, 2500]
            nn.Conv1d(in_channels=d_model, out_channels=d_model, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(d_model),
            nn.GELU()   # 最终输出: [B, d_model, 1250]
        )

        self.relative_position_bias = RelativePositionBias(num_heads=num_heads)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dropout, num_groups, self.relative_position_bias)
        self.fc = nn.Linear(d_model, output_dim)
        self.crf = torchcrf.CRF(output_dim, batch_first=True)  # 添加 CRF 层

    def forward(self, x):
        x_cnn_in = x.permute(0, 2, 1)
        x_features = self.cnn_frontend(x_cnn_in)  # Shape: [B, d_model, 1250]
        # print(">>> after cnn_frontend:", x_features.shape)

        x_transformer_in = x_features.permute(0, 2, 1)  # Shape: [B, 1250, d_model]
        mask = torch.ones(x_transformer_in.shape[0], x_transformer_in.shape[1], dtype=torch.bool, device=x.device)
        encoded_output = self.encoder(x_transformer_in, mask=mask)
        emissions = self.fc(encoded_output)
        return emissions, mask

# 训练函数，包括前向传播、计算损失、反向传播和参数更新
def train(model, train_loader, optimizer, device,scaler,scheduler,weight_tensor=None, alpha=0.5, do_profiling=False):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
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
        predicted = model.crf.decode(emissions, mask)  # 使用 CRF 解码得到预测结果
        predicted = [item for sublist in predicted for item in sublist]
        target = target[mask].view(-1).tolist()
        total += len(target)
        correct += sum(p == t for p, t in zip(predicted, target))

    if do_profiling:
        prof.__exit__(None, None, None)
        # 输出 profiling 结果
        print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))

    return running_loss / len(train_loader), correct / total


def train_model(protocols_dataset, protocol_labels):
    # 数据预处理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_test, y_train, y_test,label_encoder = preprocess_data(protocols_dataset, protocol_labels)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)  # 划分验证集

    #数据缓存
    print("开始预加载并缓存训练数据到内存中...")
    train_data_cached = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))for x, y in zip(x_train, y_train)]
    print("训练数据缓存完成。");         print("开始缓存验证数据...")
    val_data_cached = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))for x, y in zip(x_val, y_val)]
    print("验证数据缓存完成。");         print("开始缓存测试数据...")
    test_data_cached = [(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))for x, y in zip(x_test, y_test)]
    print("测试数据缓存完成。")

    #计算类别权重
    all_train_labels = np.concatenate([y.flatten() for y in y_train])
    class_weights = compute_class_weight('balanced', classes=np.unique(all_train_labels), y=all_train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Computed Class Weights:")
    print(", ".join(f"{label_encoder.classes_[i]}: {w:.4f}"  for i, w in enumerate(class_weights)))

    # 创建数据加载器
    train_dataset = CachedDataset(train_data_cached);val_dataset = CachedDataset(val_data_cached);test_dataset = CachedDataset(test_data_cached)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True,num_workers=8, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True, persistent_workers=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False,num_workers=4, pin_memory=True, persistent_workers=True)

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
    decay_factor = 0.8  ;    plateau_counter = 0;
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
            train_loss, train_acc = train(model, train_loader, optimizer, device, scaler,scheduler,class_weights_tensor,alpha=0.5, do_profiling=do_profile)
            val_loss, val_acc = test(model, val_loader, device, scaler,weight_tensor=class_weights_tensor, alpha=0.5)  # 验证集评估
            test_loss, test_acc = test(model, test_loader, device, scaler)
            elapsed_time = time.time() - start_time
            train_losses_per_epoch.append(train_loss); val_losses_per_epoch.append(val_loss); test_losses_per_epoch.append(test_loss)
            scheduler.step()  # 更新学习率
            print(f'/****Epoch {epoch + 1}, Learning Rate: {optimizer.param_groups[0]["lr"]}****/')
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
            else:
                counter += 1;   plateau_counter+=1;
                #学习率衰减
                if plateau_counter >= SCHEDULER_PATIENCE:
                    for group in optimizer.param_groups:
                        group['lr'] *= decay_factor
                    scheduler.base_lrs = [base_lr * decay_factor for base_lr in scheduler.base_lrs]
                    plateau_counter = 0
                    print(f"Plateau! lr 和 base_lrs 均乘以 {decay_factor}")
                #早停
                if counter >= PATIENCE:
                    print("Early stopping")
                    break
            # # 更新学习率调度器
            # scheduler.step(val_loss)

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
    correct = 0
    total = 0
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
                # 使用 CRF 解码得到预测结果
                predicted = model.crf.decode(emissions, mask=mask)
                # 准确率计算
                predicted = [item for sublist in predicted for item in sublist]
                target = target[mask].view(-1).tolist()
                total += len(target)
                correct += sum(p == t for p, t in zip(predicted, target))
    return running_loss / len(test_loader), correct / total

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
