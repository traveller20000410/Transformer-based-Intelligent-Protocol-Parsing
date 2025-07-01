import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx
import torchcrf
from joblib import load
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from torch.cuda.amp import autocast, GradScaler
from torch.utils.checkpoint import checkpoint
from torch.utils.data import Dataset, DataLoader
from transformer_component import export_to_onnx, loss_result_plt, preprocess_data

# 定义超参数，包括批量大小、训练轮次、学习率等
BATCH_SIZE =        8;                  EPOCHS =        100
LEARNING_RATE =     0.001;               D_MODEL =       16
NUM_HEADS =         4;                   NUM_LAYERS =    6
DROPOUT =           0.1 ;                MAX_LENGTH =    2000
GRADIENT_ACCUMULATION_STEPS = 2  # 梯度累积的批次数

# 自定义数据集类，用于存储协议数据和标签
class ProtocolDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data ;           self.labels = labels
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        # 将数据和标签转换为张量
        sequence = torch.tensor(self.data[idx], dtype=torch.long)
        labels = torch.tensor(self.labels[idx], dtype=torch.long)
        return sequence, labels

# 位置编码函数，为输入添加位置信息
def positional_encoding(max_length, d_model):
    pe = torch.zeros(max_length, d_model, dtype=torch.float32)
    position = torch.arange(0, max_length).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term);       pe[:, 1::2] = torch.cos(position * div_term)
    pe = pe.unsqueeze(0)  # (1, max_length, d_model)
    return pe

class MultiHeadLatentAttention(nn.Module):
    def __init__(self, d_model, num_heads, latent_dim):
        super(MultiHeadLatentAttention, self).__init__()
        self.d_model = d_model
        self.num_heads = num_heads;                     self.latent_dim = latent_dim
        self.W_q = nn.Linear(d_model, d_model);         self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model);         self.W_o = nn.Linear(d_model, d_model)
        self.latent_generator = nn.Sequential(nn.Linear(d_model, latent_dim),nn.ReLU())
        self.latent_modulation_q = nn.Linear(latent_dim, d_model)
        self.latent_modulation_k = nn.Linear(latent_dim, d_model)

    def forward(self, query, key, value, mask=None):
        latent = self.latent_generator(query)    # Generate latent variables
        modulation_q = torch.sigmoid(self.latent_modulation_q(latent))
        modulation_k = torch.sigmoid(self.latent_modulation_k(latent))

        q = self.W_q(query) * modulation_q      # Modify queries and keys with modulation
        k = self.W_k(key) * modulation_k
        v = self.W_v(value)

        batch_size = query.size(0)
        q = q.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        k = k.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)
        v = v.view(batch_size, -1, self.num_heads, self.d_model // self.num_heads)

        q = q.transpose(1, 2)                  # Transpose to be compatible with attention heads
        k = k.transpose(1, 2)
        v = v.transpose(1, 2)
        # Compute scaled dot-product attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / torch.sqrt(
            torch.tensor(self.d_model // self.num_heads, dtype=torch.float32))
        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(-1)      # 将 mask 扩展为 [8, 1, 1024, 1]
            # print(f"mask shape: {mask.shape}")        # print(f"scores shape: {scores.shape}")
            scores = scores.masked_fill(mask.unsqueeze(1).to(torch.bool) == 0, -1e4)
        attn_weights = F.softmax(scores, dim=-1)
        attn_output = torch.matmul(attn_weights, v)
        # Transpose back and concatenate heads
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.d_model)
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
    def __init__(self, d_model, num_heads, dropout, latent_dim):
        super(TransformerEncoderLayer, self).__init__()
        self.attention = MultiHeadLatentAttention(d_model, num_heads, latent_dim)
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
            x = self.norm2(x + self.dropout2(ffn_output));  return x
        x = checkpoint(custom_forward, x)
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, num_layers, d_model, num_heads, dropout, latent_dim):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([TransformerEncoderLayer(d_model, num_heads, dropout, latent_dim) for _ in range(num_layers)])

    def forward(self, x, mask=None):
        for layer in self.layers:  x = layer(x, mask)
        return x

class TransformerModel(nn.Module):
    def __init__(self, input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout, latent_dim):
        super(TransformerModel, self).__init__()
        # 我们将每条协议线的token映射到 d_model/2 的维度
        self.scl_embedding = nn.Embedding(input_dim, d_model // 2, padding_idx=0)
        self.sda_embedding = nn.Embedding(input_dim, d_model // 2, padding_idx=0)
        pe = positional_encoding(max_length, d_model)

        self.register_buffer('pos_encoding', pe)
        self.encoder = TransformerEncoder(num_layers, d_model, num_heads, dropout, latent_dim)
        self.fc = nn.Linear(d_model, output_dim)
        self.crf = torchcrf.CRF(output_dim, batch_first=True)  # 添加 CRF 层

    def forward(self, x):
        print(f"[Model forward] Shape of input 'x': {x.shape}")
        # x 的输入形状现在是: [batch_size, seq_len, 2]
        mask = (x[..., 0] != 0).to(x.device)  # Shape: [batch_size, seq_len]
        scl_tokens = x[..., 0].long()  # Shape: [batch_size, seq_len]
        sda_tokens = x[..., 1].long()  # Shape: [batch_size, seq_len]

        # 分别进行Embedding
        scl_embedded = self.scl_embedding(scl_tokens)  # Shape: [batch_size, seq_len, d_model/2]
        sda_embedded = self.sda_embedding(sda_tokens)  # Shape: [batch_size, seq_len, d_model/2]

        # NEW: 将两个Embedding向量在最后一个维度上拼接起来
        x_embedded = torch.cat([scl_embedded, sda_embedded], dim=-1)  # Shape: [batch_size, seq_len, d_model]

        # 现在 x_embedded 的形状是正确的 [B, L, D_model]，可以和位置编码相加了
        x = x_embedded + self.pos_encoding[:, :x_embedded.size(1), :].to(x.device)

        x = self.encoder(x, mask=mask)
        emissions = self.fc(x)
        return emissions, mask

# 训练函数，包括前向传播、计算损失、反向传播和参数更新
def train(model, train_loader, optimizer, device,scaler):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        # data.requires_grad_(True);                           target.requires_grad_()
        optimizer.zero_grad()
        # 混合精度训练
        with autocast():  # 使用混合精度自动转换
            emissions,mask = model(data)  # (batch_size, seq_len, num_classes)
            # print(f"emissions device: {emissions.device}, target device: {target.device}")
            # print("Mask:", mask)
            # print("Mask first timestep:", mask[:, 0])  # 检查第一个时间步
            loss = -model.crf(emissions, target, mask=mask)  # 使用 CRF 计算损失
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        running_loss += loss.item()
        predicted = model.crf.decode(emissions, mask)  # 使用 CRF 解码得到预测结果
        predicted = [item for sublist in predicted for item in sublist]
        target = target[mask].view(-1).tolist()
        total += len(target)
        correct += sum(p == t for p, t in zip(predicted, target))
    return running_loss / len(train_loader), correct / total

def train_model(protocols_dataset, protocol_labels,latent_dim=D_MODEL):
    # 数据预处理
    x_train, x_test, y_train, y_test,label_encoder = preprocess_data(protocols_dataset, protocol_labels)
    x_train, x_val, y_train, y_val = train_test_split(x_train, y_train, test_size=0.1, random_state=42)  # 划分验证集
    # 创建数据加载器
    train_dataset = ProtocolDataset(x_train, y_train)
    val_dataset = ProtocolDataset(x_val, y_val) ; test_dataset = ProtocolDataset(x_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False,num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False,num_workers=4, pin_memory=True)

    # 初始化模型、损失函数和优化器
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    input_dim = 21 ; output_dim = len(label_encoder.classes_)  #16
    model = TransformerModel(input_dim, output_dim, MAX_LENGTH, D_MODEL, NUM_HEADS, NUM_LAYERS, DROPOUT,latent_dim)
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE,weight_decay=1e-5)
    scaler = GradScaler()   # 创建 GradScaler 实例
    checkpoint_path = 'checkpoint.pth' # 中断时保存的检查点路径
    best_val_loss = float('inf');       patience = 10 ;                start_epoch = 0
    counter = 0  # 记录验证集损失不下降的次数
    train_losses_per_epoch = [];        val_losses_per_epoch = [];     test_losses_per_epoch = []
    # model,optimizer,scaler,start_epoch,best_val_loss=check_pth_is_accessible(checkpoint_path,model, optimizer, scaler,device)  #加载保存的模型

    # 训练和测试模型
    model.to(device);
    # 启用编译优化
    # if hasattr(torch, 'compile'):
    #     model = torch.compile(model, mode="max-autotune")  # PyTorch 2.0+ 编译优化
    start_time = time.time()
    try:
        for epoch in range(start_epoch, EPOCHS):
            train_loss, train_acc = train(model, train_loader, optimizer, device, scaler)
            val_loss, val_acc = test(model, val_loader, device, scaler)  # 验证集评估
            test_loss, test_acc = test(model, test_loader, device, scaler)
            elapsed_time = time.time() - start_time
            train_losses_per_epoch.append(train_loss); val_losses_per_epoch.append(val_loss); test_losses_per_epoch.append(test_loss)

            print(f'Epoch {epoch + 1}/{EPOCHS}: Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, '
                  f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, Test Loss: {test_loss:.4f}, Test Acc: {test_acc:.4f}')
            print(f"Time elapsed: {elapsed_time:.2f} seconds")

            # 早停机制
            if val_loss < best_val_loss:
                best_val_loss = val_loss;                    counter = 0
                torch.save(model.state_dict(), 'best_transformer_model.pth')
            else:
                counter += 1
                if counter >= patience:
                    print("Early stopping")
                    break
            # # 更新学习率调度器
            # scheduler.step(val_loss)
            torch.save({'epoch': epoch,                 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),# 'scheduler_state_dict': scheduler.state_dict(),
            'scaler_state_dict': scaler.state_dict(),       'best_val_loss': best_val_loss
            }, checkpoint_path);                             print(f"已保存检查点到 {checkpoint_path}")

    except KeyboardInterrupt:
        print("训练中断，保存当前检查点...")
        torch.save({'epoch': epoch,                     'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),   # 'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict(),            'best_val_loss': best_val_loss
        }, checkpoint_path)
        print(f"已保存检查点到 {checkpoint_path}");            print("退出训练。")

        return model, label_encoder  # 或者您可以选择重新抛出异常

    # # 检查输入到 embedding 层的数据范围
    # with torch.no_grad():
    #     for data, target in train_loader:a
    #         data, target = data.to(device), target.to(device)
    #         print(f"Max index in input: {data.max().item()}, Min index in input: {data.min().item()}")
    #         print(f"Embedding layer vocab size: {model.embedding.num_embeddings}")
    export_to_onnx(device,model, d_model=D_MODEL, max_length=MAX_LENGTH, output_filename='transformer_model.onnx')

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
    print(classification_report(test_true_labels, test_predictions, target_names=label_encoder.classes_))
    loss_result_plt(train_losses_per_epoch, val_losses_per_epoch, test_losses_per_epoch) #画出损失函数曲线

    return model, label_encoder

# 测试函数，计算损失和准确率
def test(model, test_loader, device,scaler):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (data, target) in enumerate(test_loader):
            data, target = data.to(device), target.to(device)
            with autocast():  # 使用混合精度自动转换
                emissions, mask = model(data)
                loss = -model.crf(emissions, target, mask=mask)
                running_loss += loss.item()
                # 使用 CRF 解码得到预测结果
                predicted = model.crf.decode(emissions, mask=mask)
                predicted = [item for sublist in predicted for item in sublist]
                target = target[mask].view(-1).tolist()
                total += len(target)
                correct += sum(p == t for p, t in zip(predicted, target))
    return running_loss / len(test_loader), correct / total

def predict_protocol(model, data):  # 接收 label_encoder 作为参数
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    data = np.array(data)
    data = torch.tensor(data, dtype=torch.float32).unsqueeze(0).unsqueeze(-1)
    # # 将数据移到与模型相同的设备上
    data = data.to(device)
    model.to(device)  # 将模型也移动到设备上
    label_encoder = load('label_encoder.joblib')
    # print("Training label encoder classes:", label_encoder.classes_)
    with torch.no_grad():
        emissions, mask = model(data)
        predicted = model.crf.decode(emissions, mask=mask)  # List of lists
        predicted = predicted[0]  # 取第一个（唯一的）序列
        predicted_protocol = label_encoder.inverse_transform(predicted)
        # 根据原始长度裁剪
    original_length = min(len(data[0]), MAX_LENGTH)
    return predicted_protocol[:len(data[0])]  # 根据原始长度裁剪


def predict_with_model(model, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    predicted_protocol = predict_protocol(model, test_data)
    return predicted_protocol

def load_model(input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout):
    model = TransformerModel(input_dim, output_dim, max_length, d_model, num_heads, num_layers, dropout,latent_dim=D_MODEL)
    model.load_state_dict(torch.load('transformer_model.pth'))
    model.eval()
    return model
