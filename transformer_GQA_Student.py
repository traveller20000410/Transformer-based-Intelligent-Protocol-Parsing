import torch,   torchcrf,   torch.onnx,    torch.nn as nn,   torch.nn.functional as F   ,os
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import time;                                                    import torch.profiler
from joblib import load
from sklearn.metrics import classification_report;              from sklearn.utils.class_weight import compute_class_weight # 导入一个方便的工具
from torch.utils.checkpoint import checkpoint
from torch.cuda.amp import autocast, GradScaler
from transformer_component import weighted_cross_entropy_loss,check_pth_is_accessible,export_to_onnx,loss_result_plt,preprocess_data
from torch.optim.lr_scheduler import StepLR, MultiStepLR, ReduceLROnPlateau, CosineAnnealingLR

# 定义超参数，包括批量大小、训练轮次、学习率等
BATCH_SIZE =        32;                         EPOCHS =        600
LEARNING_RATE =     0.0001;                     D_MODEL =       30
NUM_HEADS =         6;                          NUM_LAYERS =    4
DROPOUT =           0.1 ;                       MAX_LENGTH =    1000
NUM_GROUPS = 2 ;                                PATIENCE=       30;
# 自定义数据集类，用于存储协议数据和标签
class ProtocolDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data ;           self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # 将数据和标签转换为张量
        return torch.tensor(self.data[idx], dtype=torch.long), torch.tensor(self.labels[idx], dtype=torch.long)

# 位置编码函数，为输入添加位置信息
def positional_encoding(max_length, d_model):
    pe = torch.zeros(max_length, d_model, dtype=torch.float32)
    position = torch.arange(0, max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term);       pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # (1, max_length, d_model)
    return pe

class GroupedQueryAttention(nn.Module):
    def __init__(self, d_model, num_heads, num_groups):
        super(GroupedQueryAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads
        self.num_groups = num_groups
        self.head_dim = d_model // num_heads
        # 确保分组数量能够整除注意力头数
        assert num_heads % num_groups == 0, "num_heads must be divisible by num_groups"
        # 线性变换层
        self.W_q = nn.Linear(d_model, d_model)  # 查询变换
        self.W_k = nn.Linear(d_model, d_model // num_groups)  # 键变换（分组共享）
        self.W_v = nn.Linear(d_model, d_model // num_groups)  # 值变换（分组共享）
        self.W_o = nn.Linear(d_model, d_model)  # 输出变换

    def forward(self, query, key, value, mask=None):
        batch_size = query.size(0)

        # 线性变换
        q = self.W_q(query)  # (batch_size, seq_len, d_model)
        k = self.W_k(key)    # (batch_size, seq_len, d_model // num_groups)
        v = self.W_v(value)  # (batch_size, seq_len, d_model // num_groups)
        # 重塑为多头形式
        q = q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)  # (batch_size, num_heads, seq_len, head_dim)
        k = k.view(batch_size, -1, self.num_groups, self.head_dim).transpose(1, 2)  # (batch_size, num_groups, seq_len, head_dim)
        v = v.view(batch_size, -1, self.num_groups, self.head_dim).transpose(1, 2)  # (batch_size, num_groups, seq_len, head_dim)
        # 广播键和值到每个查询组
        k = k.repeat_interleave(self.num_heads // self.num_groups, dim=1)  # (batch_size, num_heads, seq_len, head_dim)
        v = v.repeat_interleave(self.num_heads // self.num_groups, dim=1)  # (batch_size, num_heads, seq_len, head_dim)
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))  # (batch_size, num_heads, seq_len, seq_len)
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
        def __init__(self, d_model, d_ff=512):
            super(FeedForward, self).__init__()
            self.linear1 = nn.Linear(d_model, d_ff)
            self.linear2 = nn.Linear(d_ff, d_model)

        def forward(self, x):
            x = F.gelu(self.linear1(x))
            x = self.linear2(x)
            return x

# Transformer 编码器层类
class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dropout, num_groups):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = GroupedQueryAttention(d_model, num_heads, num_groups)
        self.ffn = FeedForward(d_model)
        self.norm1 = nn.LayerNorm(d_model);     self.norm2 = nn.LayerNorm(d_model)
        self.dropout1 = nn.Dropout(dropout);    self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        def custom_forward(x):
            attn_output, _ = self.attention(x, x, x, mask)
            # 调整 attn_output 的维度，确保与 x 维度匹配
            if attn_output.size(1) != x.size(1):
                attn_output = attn_output[:, :x.size(1), :]
            x = self.norm1(x + self.dropout1(attn_output))
            ffn_output = self.ffn(x)
            x = self.norm2(x + self.dropout2(ffn_output));
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
    def __init__(self, input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout, num_groups):
        super(TransformerModel, self).__init__()
        self.scl_embedding = nn.Embedding(input_dim, d_model // 2, padding_idx=0)
        self.sda_embedding = nn.Embedding(input_dim, d_model // 2, padding_idx=0)
        pe = positional_encoding(max_length, d_model)
        self.register_buffer('pos_encoding', pe)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dropout, num_groups)
        self.fc = nn.Linear(d_model, output_dim)
        self.crf = torchcrf.CRF(output_dim, batch_first=True)  # 添加 CRF 层

    def forward(self, x):
        mask = (x[..., 0] != 0).to(x.device)

        scl_tokens = x[..., 0].long()
        sda_tokens = x[..., 1].long()

        scl_embedded = self.scl_embedding(scl_tokens)
        sda_embedded = self.sda_embedding(sda_tokens)

        x_embedded = torch.cat([scl_embedded, sda_embedded], dim=-1)
        x = x_embedded + self.pos_encoding[:, :x_embedded.size(1), :].to(x.device)
        x = self.encoder(x, mask=mask)
        emissions = self.fc(x)

        return emissions, mask

# 训练函数，包括前向传播、计算损失、反向传播和参数更新
def train(model, train_loader, optimizer, device,scaler,scheduler,weight_tensor=None, alpha=0.5):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.profiler.profile(
            schedule=torch.profiler.schedule(wait=1, warmup=1, active=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler('./log'),
            record_shapes=True,
            with_stack=True
    ) as prof:
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

            running_loss += loss.item()
            predicted = model.crf.decode(emissions, mask)  # 使用 CRF 解码得到预测结果
            predicted = [item for sublist in predicted for item in sublist]
            target = target[mask].view(-1).tolist()
            total += len(target)
            correct += sum(p == t for p, t in zip(predicted, target))

            if prof.step_num >= 2:  # 仅在 warmup 阶段之后开始打印
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=10))
    return running_loss / len(train_loader), correct / total

def train_model(protocols_dataset, protocol_labels,num_groups=D_MODEL):
    # 数据预处理
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    x_train, x_test, y_train, y_test,label_encoder = preprocess_data(protocols_dataset, protocol_labels)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)  # 划分验证集

    #计算类别权重
    all_train_labels = np.concatenate([y.flatten() for y in y_train])
    class_weights = compute_class_weight('balanced', classes=np.unique(all_train_labels), y=all_train_labels)
    class_weights_tensor = torch.tensor(class_weights, dtype=torch.float32).to(device)
    print("Computed Class Weights:")
    print(", ".join(f"{label_encoder.classes_[i]}: {w:.4f}"  for i, w in enumerate(class_weights)))

    # 创建数据加载器
    train_dataset = ProtocolDataset(x_train, y_train)
    val_dataset = ProtocolDataset(x_val, y_val) ; test_dataset = ProtocolDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,num_workers=2, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=2, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=2, pin_memory=True)

    # 初始化模型、损失函数和优化器
    input_dim = 21 ; output_dim = len(label_encoder.classes_)  #16
    model = TransformerModel(input_dim, output_dim, MAX_LENGTH, D_MODEL, NUM_HEADS, NUM_LAYERS, DROPOUT,NUM_GROUPS)
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

        # # 只有当检查点里有scheduler状态时才加载
        # if 'scheduler_state_dict' in checkpoint:
        #     scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
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
    try:
        for epoch in range(start_epoch, EPOCHS):
            train_loss, train_acc = train(model, train_loader, optimizer, device, scaler,scheduler,class_weights_tensor)
            val_loss, val_acc = test(model, val_loader, device, scaler)  # 验证集评估
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
                print(classification_report(all_trues, all_preds, target_names=str_names))
                print("=== End Report ===\n")
                model.train()

            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss;                    counter = 0
                torch.save(model.state_dict(), 'best_transformer_model.pth')
            else:
                counter += 1
                if counter >= PATIENCE:
                    print("Early stopping")
                    break
            # # 更新学习率调度器
            # scheduler.step(val_loss)

            # 保存检查点
            if epoch>=1:
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
    print(classification_report(train_true_labels, train_predictions,target_names=label_encoder.classes_))
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
    print(classification_report(test_true_labels,test_predictions,target_names=str_names))
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
    device = next(model.parameters()).device  # 自动获取模型所在的设备
    model.eval()
    data = torch.tensor(np.array(data), dtype=torch.long).unsqueeze(0).to(device)
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

def load_model(input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout):
    model = TransformerModel(input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout,num_groups=NUM_GROUPS)
    # model.load_state_dict(torch.load('checkpoint.pth'))
    checkpoint = torch.load('checkpoint.pth')
    model.load_state_dict(checkpoint['model_state_dict'])  # 仅加载模型权重
    model.eval()
    return model
