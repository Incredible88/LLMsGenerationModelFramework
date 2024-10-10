import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader, random_split
from Pretraining.transformer import InfiniTransformer
from Pretraining.embedding_esm3 import Esm3Embedding
from sklearn.preprocessing import LabelEncoder
import torch.optim as optim
import os
from Bio import SeqIO
import pickle
import wandb
from tqdm import tqdm
import wandb

wandb.login(key="5f5f94d3de9157cf146ad88ecc4e0518a7a7549e")
# 初始化 wandb
wandb.init(project="protein-function-prediction", config={
    "learning_rate": 1e-4,
    "epochs": 10,
    "batch_size": 1,
    "model": "ProteinFunctionPredictor"
})

sequence_list = []

# 读取所有PFAM家族的文件
source_directory = '/home/share/huadjyin/home/yinpeng/czw/test_pfam'
for pfam_id in os.listdir(source_directory):
    pfam_path = os.path.join(source_directory, pfam_id)

    # 遍历家族中的所有.pkl文件
    for filename in os.listdir(pfam_path):
        if filename.endswith('.pkl'):
            file_path = os.path.join(pfam_path, filename)
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                seq = data['seq']
                start = data['start']
                end = data['end']
                token_labels = data['token_label']
                # 使用pfam_id作为label
                label = pfam_id  # 假设pfam_id就是文件夹的名字
                sequence_list.append({
                    'seq': seq,
                    'label': label,
                    'start': start,
                    'end': end,
                    'token_label':token_labels
                })

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=150):
        super(PositionalEncoding, self).__init__()
        # 创建一个矩阵来存储位置编码
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        # 正弦和余弦的计算方式
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return x


class EmbeddingDataset(Dataset):
    def __init__(self, sequence_list, esm3_embedder, max_len, embedding_dim):
        self.sequence_list = sequence_list
        self.esm3_embedder = esm3_embedder
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.label_to_idx = generate_label_to_idx(sequence_list)  # 动态生成 label_to_idx
        self.data = self.prepare_data()

    def prepare_data(self):
        all_data = []

        # 使用 tqdm 包裹数据循环，添加进度条显示
        for entry in tqdm(self.sequence_list, desc="Generating Embeddings"):
            # 处理每个序列的 embedding
            sequence = entry['seq']
            label = entry['label']  # 标签
            start = int(entry['start'])  # 起始位置
            end = int(entry['end'])  # 结束位置

            # 使用 ESM3 模型生成 embedding
            batch = {"sequence": [sequence]}
            embedding = self.esm3_embedder.get_embedding(batch)

            # 填充 embedding 到固定长度
            padded_embedding = torch.zeros(self.max_len, self.embedding_dim)
            seq_len = min(embedding.shape[1], self.max_len)
            padded_embedding[:seq_len, :] = embedding.squeeze(0)[:seq_len, :]

            # 将标签转换为索引
            label_idx = self.label_to_idx[label]

            # 存储 embedding、标签、起始和结束位置
            all_data.append((padded_embedding, label_idx, start, end))

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # 返回 embedding、标签、起始位置、结束位置
        embedding, label_idx, start, end = self.data[idx]
        return embedding, torch.tensor(label_idx), torch.tensor(start), torch.tensor(end)


classification_loss_fn = nn.CrossEntropyLoss()  # 标签分类损失
regression_loss_fn = nn.SmoothL1Loss()  # 使用更稳健的 SmoothL1Loss


# 归一化起始和结束位置
def normalize_positions(start_positions, end_positions, max_len):
    start_positions_normalized = start_positions / max_len
    end_positions_normalized = end_positions / max_len
    return start_positions_normalized, end_positions_normalized


# 计算损失函数，带回归损失的权重
def compute_loss(label_logits, positions, labels, start_positions, end_positions, max_len, regression_weight=0.1):
    # 标签分类损失
    classification_loss = classification_loss_fn(label_logits, labels)

    # 归一化起始和结束位置
    start_positions_normalized, end_positions_normalized = normalize_positions(start_positions, end_positions, max_len)

    # 起始和结束位置的回归损失
    start_loss = regression_loss_fn(positions[:, 0], start_positions_normalized.float())
    end_loss = regression_loss_fn(positions[:, 1], end_positions_normalized.float())

    # 总损失：分类损失 + 回归损失（乘以权重）
    total_loss = classification_loss + regression_weight * (start_loss + end_loss)
    return total_loss


# 计算分类准确率
def compute_accuracy(label_logits, labels):
    _, predicted_labels = torch.max(label_logits, 1)
    correct = (predicted_labels == labels).float().sum()
    accuracy = correct / labels.size(0)
    return accuracy


# 计算起始和结束位置的 MAE（平均绝对误差）
def compute_position_error(positions, start_positions, end_positions, max_len):
    start_positions_normalized, end_positions_normalized = normalize_positions(start_positions, end_positions, max_len)
    start_error = torch.abs(positions[:, 0] - start_positions_normalized.float()).mean()
    end_error = torch.abs(positions[:, 1] - end_positions_normalized.float()).mean()
    return start_error, end_error


class ProteinFunctionPredictor(nn.Module):
    def __init__(self, embedding_dim, num_labels, max_len=150):
        super(ProteinFunctionPredictor, self).__init__()

        # 位置编码层
        self.positional_encoding = PositionalEncoding(embedding_dim, max_len)

        # 定义6层decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

        # 标签的分类头部
        self.label_head = nn.Linear(embedding_dim, num_labels)

        # 起始和结束位置的回归头部
        self.position_head = nn.Linear(embedding_dim, 2)  # 回归两个值：start 和 end

    def forward(self, embeddings):
        # 添加位置编码
        embeddings = self.positional_encoding(embeddings)

        # 通过 transformer decoder
        transformer_output = self.transformer_decoder(embeddings, embeddings)  # (batch_size, sequence_length, embedding_dim)
        sequence_rep = transformer_output[:, -1, :]  # (batch_size, embedding_dim)

        # 标签预测
        label_logits = self.label_head(sequence_rep)  # (batch_size, num_labels)

        # 起始和结束位置预测
        positions = self.position_head(sequence_rep)  # (batch_size, 2)，即 start 和 end

        return label_logits, positions


# 训练和验证函数
def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=10, max_len=150, device="cpu",
                       regression_weight=0.1):
    for epoch in range(num_epochs):
        # 训练模式
        model.train()
        total_train_loss = 0

        for embeddings, labels, start_positions, end_positions in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            start_positions, end_positions = start_positions.to(device), end_positions.to(device)

            # 前向传播
            label_logits, positions = model(embeddings)

            # 计算损失
            loss = compute_loss(label_logits, positions, labels, start_positions, end_positions, max_len,
                                regression_weight)

            # 反向传播和优化
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")

        # 验证模式
        model.eval()
        total_val_loss = 0
        total_accuracy = 0
        total_start_error = 0
        total_end_error = 0

        with torch.no_grad():
            for embeddings, labels, start_positions, end_positions in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                start_positions, end_positions = start_positions.to(device), end_positions.to(device)

                # 前向传播
                label_logits, positions = model(embeddings)

                # 计算损失
                val_loss = compute_loss(label_logits, positions, labels, start_positions, end_positions, max_len,
                                        regression_weight)
                total_val_loss += val_loss.item()

                # 计算分类准确率和位置误差
                accuracy = compute_accuracy(label_logits, labels)
                start_error, end_error = compute_position_error(positions, start_positions, end_positions, max_len)

                total_accuracy += accuracy.item()
                total_start_error += start_error.item()
                total_end_error += end_error.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        avg_start_error = total_start_error / len(val_loader)
        avg_end_error = total_end_error / len(val_loader)

        print(
            f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Accuracy: {avg_accuracy * 100:.2f}%, Start Error: {avg_start_error:.4f}, End Error: {avg_end_error:.4f}")

        wandb.log({
            "Validation Loss": avg_val_loss,
            "Validation Accuracy": avg_accuracy * 100,  # 转换为百分比
            "Start Error": avg_start_error,
            "End Error": avg_end_error,
            "Epoch": epoch + 1
        })

def generate_label_to_idx(sequence_list):
    # 提取所有唯一的标签
    labels = set([entry['label'] for entry in sequence_list])

    # 生成标签到索引的映射
    label_to_idx = {label: idx for idx, label in enumerate(labels)}

    return label_to_idx

esm3_embedder = Esm3Embedding(pooling="mean")
# 设备选择
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# 创建训练和验证集
dataset = EmbeddingDataset(sequence_list, esm3_embedder, max_len=150, embedding_dim=1536)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# 使用 DataLoader 加载数据
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 初始化模型和优化器
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# 初始化模型和优化器
model = ProteinFunctionPredictor(embedding_dim=1536, num_labels=len(generate_label_to_idx(sequence_list)), max_len=150).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# 开始训练和验证
train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=10, max_len=150, device=device, regression_weight=0.05)
wandb.finish()