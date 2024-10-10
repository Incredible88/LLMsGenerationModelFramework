import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader, random_split
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

import math


class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=512):
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

from tqdm import tqdm

class PreSavedEmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, label_mapping, max_files=2000):
        self.embedding_dir = embedding_dir
        self.files = [f for f in os.listdir(embedding_dir) if f.endswith('.pkl')]
        self.label_mapping = label_mapping

        # 限制读取的文件数量，默认最多读取 max_files 个文件
        self.files = self.files[:max_files]
        # 创建一个 tqdm 进度条
        self.progress_bar = tqdm(total=len(self.files), desc="Loading Embeddings")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.embedding_dir, file_name)

        with open(file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            embedding = torch.tensor(data['embedding'], dtype=torch.float32)

            # 获取真实的标签，并将其映射为索引
            label_str = data['label']
            label = self.label_mapping[label_str]  # 将标签映射到整数索引

            start = int(data['start'])
            end = int(data['end'])

        # 更新进度条
        self.progress_bar.update(1)

        return embedding, torch.tensor(label, dtype=torch.long), torch.tensor(start), torch.tensor(end)

    def __del__(self):
        # 确保在完成时关闭进度条
        self.progress_bar.close()
class FocalLoss(nn.Module):
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = self.alpha * (1 - pt) ** self.gamma * BCE_loss
        return torch.mean(F_loss)
# 损失计算
classification_loss_fn = FocalLoss(alpha=1, gamma=2)  # 改为Focal Loss
regression_loss_fn = nn.SmoothL1Loss()

def compute_loss(label_logits, predicted_positions, true_labels, start_positions, end_positions, max_len, regression_weight=0.1):
    # 标签分类损失
    label_loss = classification_loss_fn(label_logits, true_labels)

    # 计算位置的回归损失
    start_loss = regression_loss_fn(predicted_positions[:, 0], start_positions.float() / max_len)
    end_loss = regression_loss_fn(predicted_positions[:, 1], end_positions.float() / max_len)

    # 总损失 = 分类损失 + 回归位置的损失
    total_loss = label_loss + regression_weight * (start_loss + end_loss)

    return total_loss


def normalize_positions(start_positions, end_positions, max_len):
    start_positions_normalized = start_positions / max_len
    end_positions_normalized = end_positions / max_len
    return start_positions_normalized, end_positions_normalized


# 计算分类准确率
def compute_accuracy(label_logits, true_labels):
    _, predicted_labels = torch.max(label_logits, 1)
    correct_predictions = (predicted_labels == true_labels).float().sum()
    accuracy = correct_predictions / true_labels.size(0)
    return accuracy, predicted_labels



# 计算起始和结束位置的 MAE（平均绝对误差）
def compute_position_error(positions, start_positions, end_positions, max_len):
    start_positions_normalized, end_positions_normalized = normalize_positions(start_positions, end_positions, max_len)
    start_error = torch.abs(positions[:, 0] - start_positions_normalized.float()).mean()
    end_error = torch.abs(positions[:, 1] - end_positions_normalized.float()).mean()
    return start_error, end_error


# Protein Function Prediction Model
import torch.nn.functional as F



class ProteinFunctionPredictorEncoderOnly(nn.Module):
    def __init__(self, original_embedding_dim, reduced_embedding_dim, num_labels, max_len=512):
        super(ProteinFunctionPredictorEncoderOnly, self).__init__()

        # 位置编码层
        self.positional_encoding = PositionalEncoding(original_embedding_dim, max_len)

        # 添加线性层用于降维
        self.fc_reduce = nn.Linear(original_embedding_dim, reduced_embedding_dim)

        # 使用Transformer Encoder
        encoder_layer = nn.TransformerEncoderLayer(d_model=reduced_embedding_dim, nhead=8)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=1)

        # 归一化层
        self.layer_norm = nn.LayerNorm(reduced_embedding_dim)
        self.dropout = nn.Dropout(p=0.3)  # 加入Dropout

        # 标签的分类头部，改为全连接层直接预测 num_labels
        self.fc_label = nn.Linear(reduced_embedding_dim, num_labels)

        # 起始和结束位置的回归头部
        self.fc_position = nn.Linear(reduced_embedding_dim, 2)  # 回归两个值：start 和 end

        # 初始化权重
        self._init_weights()

    def _init_weights(self):
        # 使用 Xavier 均匀分布初始化权重
        nn.init.xavier_uniform_(self.fc_label.weight)
        nn.init.xavier_uniform_(self.fc_position.weight)
        nn.init.zeros_(self.fc_label.bias)
        nn.init.zeros_(self.fc_position.bias)

    def forward(self, embeddings):
        # 添加位置编码
        embeddings = self.positional_encoding(embeddings)

        # 通过线性层进行降维
        embeddings = self.fc_reduce(embeddings)

        # 通过 transformer encoder
        transformer_output = self.transformer_encoder(embeddings)  # (batch_size, sequence_length, reduced_embedding_dim)

        # 通过归一化层
        sequence_rep = self.layer_norm(transformer_output[:, -1, :])  # (batch_size, reduced_embedding_dim)
        sequence_rep = self.dropout(sequence_rep)  # 加入Dropout

        # 直接通过全连接层预测标签类别
        label_logits = self.fc_label(sequence_rep)  # (batch_size, num_labels)

        # 起始和结束位置预测
        positions = self.fc_position(sequence_rep)  # (batch_size, 2)，即 start 和 end

        # 使用 softmax 将 label_logits 转化为概率分布
        label_probs = F.softmax(label_logits, dim=-1)

        return label_probs, positions

# 标签最近邻映射
def postprocess_label(predicted_labels, label_embeddings):
    closest_labels = []
    for pred in predicted_labels:
        distances = torch.cdist(pred.unsqueeze(0), label_embeddings.unsqueeze(0)).squeeze(0)
        closest_label_idx = distances.argmin().item()
        closest_labels.append(closest_label_idx)
    return torch.tensor(closest_labels, dtype=torch.long)

# 训练和验证函数
def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=10, max_len=512, device="cpu", regression_weight=0.1):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        all_predicted_labels = []  # 存储所有预测标签
        all_true_labels = []  # 存储所有真实标签

        for embeddings, labels, start_positions, end_positions in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            start_positions, end_positions = start_positions.to(device), end_positions.to(device)

            predicted_label_logits, predicted_positions = model(embeddings)

            loss = compute_loss(predicted_label_logits, predicted_positions, labels, start_positions, end_positions, max_len, regression_weight)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_accuracy = 0
        total_start_error = 0
        total_end_error = 0

        with torch.no_grad():
            for embeddings, labels, start_positions, end_positions in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                start_positions, end_positions = start_positions.to(device), end_positions.to(device)

                predicted_label_logits, predicted_positions = model(embeddings)

                val_loss = compute_loss(predicted_label_logits, predicted_positions, labels, start_positions, end_positions, max_len, regression_weight)
                total_val_loss += val_loss.item()

                accuracy, predicted_labels = compute_accuracy(predicted_label_logits, labels)
                total_accuracy += accuracy.item()

                start_error, end_error = compute_position_error(predicted_positions, start_positions, end_positions, max_len)
                total_start_error += start_error.item()
                total_end_error += end_error.item()

                # 将当前 batch 的预测标签和真实标签添加到列表
                all_predicted_labels.extend(predicted_labels.cpu().numpy().tolist())
                all_true_labels.extend(labels.cpu().numpy().tolist())

        avg_val_loss = total_val_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)
        avg_start_error = total_start_error / len(val_loader)
        avg_end_error = total_end_error / len(val_loader)

        print(
            f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Accuracy: {avg_accuracy * 100:.2f}%, Start Error: {avg_start_error:.4f}, End Error: {avg_end_error:.4f}")

        # 保存当前 epoch 的预测和真实标签
        true_label_file_path = f'/home/share/huadjyin/home/yinpeng/czw/code/output/true_labels_epoch_{epoch + 1}.txt'
        predicted_label_file_path = f'/home/share/huadjyin/home/yinpeng/czw/code/output/predicted_labels_epoch_{epoch + 1}.txt'

        with open(true_label_file_path, 'w') as true_file, open(predicted_label_file_path, 'w') as predicted_file:
            for true_label, predicted_label in zip(all_true_labels, all_predicted_labels):
                true_file.write(f"{true_label}\n")
                predicted_file.write(f"{predicted_label}\n")


def generate_label_embeddings(num_labels, label_embedding_dim, device):
    label_embeddings = nn.Embedding(num_labels, label_embedding_dim).to(device)

    # 使用 Xavier 均匀分布初始化嵌入向量
    nn.init.xavier_uniform_(label_embeddings.weight)

    # 你也可以使用标准正态分布来初始化
    # nn.init.normal_(label_embeddings.weight, mean=0, std=1)

    return label_embeddings


def generate_label_to_idx(embedding_dir, max_files=1000):
    labels = set()
    all_files = [f for f in os.listdir(embedding_dir) if f.endswith('.pkl')][:max_files]
    for file_name in tqdm(all_files, desc="Extracting labels"):
        file_path = os.path.join(embedding_dir, file_name)
        with open(file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            label = data['label']
            labels.add(label)
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    return label_to_idx

def check_label_distribution(train_dataset, val_dataset):
    # 用来存储训练集和验证集中的所有类别
    train_labels = set()
    val_labels = set()

    # 统计训练集中的所有类别
    print("Checking train dataset...")
    for _, label, _, _ in tqdm(train_dataset, desc="Train Labels"):
        train_labels.add(label.item())

    # 统计验证集中的所有类别
    print("Checking validation dataset...")
    for _, label, _, _ in tqdm(val_dataset, desc="Validation Labels"):
        val_labels.add(label.item())

    # 打印不同数据集中的类别数量
    print(f"Number of unique labels in train dataset: {len(train_labels)}")
    print(f"Number of unique labels in validation dataset: {len(val_labels)}")

    # 检查验证集中是否有在训练集中不存在的类别
    missing_in_train = val_labels - train_labels
    if missing_in_train:
        print(f"Labels in validation set but not in train set: {missing_in_train}")
    else:
        print("All labels in validation set are present in the train set.")


# 设备选择，确保在 GPU 0 上
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.cuda.set_device(device)

# 标签到索引生成
embedding_dir = '/home/share/huadjyin/home/yinpeng/czw/code/Embedding_500_4'
label_to_idx = generate_label_to_idx(embedding_dir, max_files=2000)
num_labels = len(label_to_idx)
print(f"Number of labels: {num_labels}")

# 更新 wandb 配置中的 num_labels
wandb.config.update({"num_labels": num_labels})

# 数据集52083
dataset = PreSavedEmbeddingDataset(embedding_dir, label_to_idx, max_files=2000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

# DataLoader
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 检查标签分布
check_label_distribution(train_dataset, val_dataset)

# 初始化模型和优化器
embedding_dim = 512
reduced_dim = 64
model = ProteinFunctionPredictorEncoderOnly(original_embedding_dim=embedding_dim, reduced_embedding_dim=reduced_dim, num_labels=num_labels, max_len=512).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-5)

# 打印模型参数数量
def count_model_parameters(model):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")

count_model_parameters(model)

# 开始训练和验证
train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=50, max_len=512, device=device, regression_weight=0.05)

# 结束 wandb 记录
wandb.finish()
