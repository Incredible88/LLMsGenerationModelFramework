import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader, random_split
from Pretraining.transformer import InfiniTransformer
from Pretraining.embedding_esm3 import Esm3Embedding
import torch.optim as optim
import os
from Bio import SeqIO
import pickle
import wandb
from tqdm import tqdm
import math

wandb.login(key="5f5f94d3de9157cf146ad88ecc4e0518a7a7549e")
# 初始化 wandb
wandb.init(project="protein-function-prediction", config={
    "learning_rate": 1e-4,
    "epochs": 10,
    "batch_size": 1,
    "model": "ProteinFunctionPredictor"
})

class PreSavedEmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, label_mapping, max_files=50000):
        self.embedding_dir = embedding_dir
        self.files = [f for f in os.listdir(embedding_dir) if f.endswith('.pkl')]
        self.label_mapping = label_mapping

        # 限制读取的文件数量，默认最多读取 max_files 个文件
        self.files = self.files[:max_files]

        # 预加载数据并添加进度条
        self.data = []
        for file_name in tqdm(self.files, desc="Loading Embeddings"):
            file_path = os.path.join(self.embedding_dir, file_name)

            with open(file_path, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
                embedding = torch.tensor(data['embedding'], dtype=torch.float32)

                # 获取真实的标签，并将其映射为索引
                label_str = data['label']
                label = self.label_mapping[label_str]  # 将标签映射到整数索引

                start = int(data['start'])
                end = int(data['end'])

                self.data.append((embedding, label, start, end))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding, label, start, end = self.data[idx]
        return embedding, torch.tensor(label, dtype=torch.long), torch.tensor(start), torch.tensor(end)


def build_label_mapping(embedding_dir, max_files=50000):
    label_mapping = {}
    index = 0
    processed_files = 0

    # 遍历所有文件以提取唯一的标签，最多处理 max_files 个文件
    for file_name in os.listdir(embedding_dir):
        if file_name.endswith('.pkl'):
            file_path = os.path.join(embedding_dir, file_name)

            with open(file_path, 'rb') as pkl_file:
                data = pickle.load(pkl_file)
                label_str = data['label']

                # 如果标签不在映射中，添加新的映射
                if label_str not in label_mapping:
                    label_mapping[label_str] = index
                    index += 1

            # 增加处理的文件计数
            processed_files += 1

            # 如果达到 max_files，提前结束
            if processed_files >= max_files:
                break

    return label_mapping

# Dataset class
class EmbeddingDataset(Dataset):
    def __init__(self, sequence_list, esm3_embedder, max_len, embedding_dim):
        self.sequence_list = sequence_list
        self.esm3_embedder = esm3_embedder
        self.max_len = max_len
        self.embedding_dim = embedding_dim
        self.data = self.prepare_data()

    def prepare_data(self):
        all_data = []
        for entry in tqdm(self.sequence_list, desc="Generating Embeddings"):
            sequence = entry['seq']
            label = string_to_float_hash(entry['label'])
            start = int(entry['start'])
            end = int(entry['end'])

            # 使用 ESM3 模型生成 embedding
            batch = {"sequence": [sequence]}
            embedding = self.esm3_embedder.get_embedding(batch)

            padded_embedding = torch.zeros(self.max_len, self.embedding_dim)
            seq_len = min(embedding.shape[1], self.max_len)
            padded_embedding[:seq_len, :] = embedding.squeeze(0)[:seq_len, :]

            all_data.append((padded_embedding, label, start, end))

        return all_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        embedding, label, start, end = self.data[idx]
        return embedding, torch.tensor(label), torch.tensor(start), torch.tensor(end)

# Loss functions
regression_loss_fn = nn.SmoothL1Loss()

# Normalize positions
def normalize_positions(start_positions, end_positions, max_len):
    start_positions_normalized = start_positions / max_len
    end_positions_normalized = end_positions / max_len
    return start_positions_normalized, end_positions_normalized


# Compute loss
def compute_loss(predicted_label_embedding, predicted_start, predicted_end, true_labels, start_positions, end_positions, max_len, regression_weight=0.2):
    # 标签的回归损失
    true_label_embeddings = model.label_embeddings(true_labels)
    label_loss = torch.nn.functional.mse_loss(predicted_label_embedding, true_label_embeddings)

    # 归一化位置
    start_positions_normalized, end_positions_normalized = normalize_positions(start_positions, end_positions, max_len)

    # 位置的回归损失
    start_loss = regression_loss_fn(predicted_start.float(), start_positions_normalized.float())
    end_loss = regression_loss_fn(predicted_end.float(), end_positions_normalized.float())

    # 总损失
    total_loss = label_loss + regression_weight * (start_loss + end_loss)
    return total_loss



# Compute label error (MAE)
def compute_label_error(predicted_labels, true_labels):
    label_error = torch.abs(predicted_labels - true_labels).mean()
    return label_error

# Compute position error
def compute_position_error(positions, start_positions, end_positions, max_len):
    start_positions_normalized, end_positions_normalized = normalize_positions(start_positions, end_positions, max_len)
    start_error = torch.abs(positions[:, 0] - start_positions_normalized.float()).mean()
    end_error = torch.abs(positions[:, 1] - end_positions_normalized.float()).mean()
    return start_error, end_error


def postprocess_label_embedding(predicted_label_embedding, label_embeddings):
    # 计算模型输出的嵌入与真实标签嵌入之间的距离，找到最近的标签
    distances = torch.cdist(predicted_label_embedding.unsqueeze(0), label_embeddings.weight.unsqueeze(0)).squeeze(0)
    closest_label_indices = torch.argmin(distances, dim=1)
    return closest_label_indices




def compute_label_accuracy(predicted_labels, true_labels, label_mapping, device):
    # 通过标签嵌入的最近邻找到最近的标签
    processed_labels = postprocess_label_embedding(predicted_labels, model.label_embeddings)

    # 计算标签的准确率
    correct = (processed_labels == true_labels).float().sum()
    accuracy = correct / len(true_labels)
    return accuracy.item()



# Model definition
class ProteinFunctionPredictor(nn.Module):
    def __init__(self, embedding_dim, label_dim, num_labels, max_len=150):
        super(ProteinFunctionPredictor, self).__init__()
        self.decoder_layer = nn.TransformerDecoderLayer(d_model=embedding_dim, nhead=8)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layer, num_layers=6)

        self.label_dim = label_dim
        self.label_embeddings = nn.Embedding(num_labels, label_dim)
        self.regression_head = nn.Linear(embedding_dim, label_dim + 2)

    def rope(self, embeddings):
        seq_len = embeddings.size(0)
        dim = embeddings.size(1)

        position = torch.arange(seq_len).unsqueeze(1)  # (seq_len, 1)
        div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))  # (dim // 2)

        sinusoids = torch.empty(seq_len, dim).to(embeddings.device)
        sinusoids[:, 0::2] = torch.sin(position * div_term)  # 偶数维
        sinusoids[:, 1::2] = torch.cos(position * div_term)  # 奇数维

        return embeddings * sinusoids

    def forward(self, embeddings):
        # 应用ROPE
        embeddings = self.rope(embeddings)

        seq_len = embeddings.size(0)
        batch_size = embeddings.size(1)

        tgt_key_padding_mask = torch.zeros((batch_size, seq_len), dtype=torch.bool).to(embeddings.device)
        causal_mask = torch.triu(torch.ones((seq_len, seq_len)), diagonal=1).to(embeddings.device).bool()

        transformer_output = self.transformer_decoder(
            embeddings,
            memory=embeddings,
            tgt_mask=causal_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )

        sequence_rep = transformer_output[:, -1, :]
        outputs = self.regression_head(sequence_rep)

        predicted_label_embedding = outputs[:, :self.label_dim]
        predicted_start = outputs[:, self.label_dim]
        predicted_end = outputs[:, self.label_dim + 1]

        return predicted_label_embedding, predicted_start, predicted_end


# Training and evaluation function
def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=10, max_len=150, device="cpu", regression_weight=0.1, label_mapping=None):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        total_label_error = 0
        total_label_accuracy = 0

        for embeddings, labels, start_positions, end_positions in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)
            start_positions, end_positions = start_positions.to(device), end_positions.to(device)

            predicted_labels, predicted_start, predicted_end = model(embeddings)

            loss = compute_loss(predicted_labels, predicted_start, predicted_end, labels, start_positions, end_positions, max_len, regression_weight)
            label_error = compute_label_error(predicted_labels, labels)

            # 计算标签准确率
            label_accuracy = compute_label_accuracy(predicted_labels, labels, label_mapping, device)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()
            total_label_error += label_error.item()
            total_label_accuracy += label_accuracy

        avg_train_loss = total_train_loss / len(train_loader)
        avg_label_error = total_label_error / len(train_loader)
        avg_label_accuracy = total_label_accuracy / len(train_loader)

        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}, Label Error: {avg_label_error}, Label Accuracy: {avg_label_accuracy * 100:.2f}%")

        model.eval()
        total_val_loss = 0
        total_start_error = 0
        total_end_error = 0
        total_label_error = 0
        total_label_accuracy = 0

        with torch.no_grad():
            for embeddings, labels, start_positions, end_positions in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)
                start_positions, end_positions = start_positions.to(device), end_positions.to(device)

                predicted_labels, predicted_start, predicted_end = model(embeddings)

                val_loss = compute_loss(predicted_labels, predicted_start, predicted_end, labels, start_positions, end_positions, max_len, regression_weight)
                label_error = compute_label_error(predicted_labels, labels)
                label_accuracy = compute_label_accuracy(predicted_labels, labels, label_mapping, device)

                total_val_loss += val_loss.item()
                total_start_error += torch.abs(predicted_start - start_positions).mean().item()
                total_end_error += torch.abs(predicted_end - end_positions).mean().item()
                total_label_error += label_error.item()
                total_label_accuracy += label_accuracy

        avg_val_loss = total_val_loss / len(val_loader)
        avg_start_error = total_start_error / len(val_loader)
        avg_end_error = total_end_error / len(val_loader)
        avg_label_error = total_label_error / len(val_loader)
        avg_label_accuracy = total_label_accuracy / len(val_loader)
        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Start Error: {avg_start_error:.4f}, End Error: {avg_end_error:.4f}, Label Error: {avg_label_error:.4f}, Label Accuracy: {avg_label_accuracy * 100:.2f}%")

        scheduler.step(avg_val_loss)
        # Log to wandb
        wandb.log({
            "Validation Loss": avg_val_loss,
            "Start Error": avg_start_error,
            "End Error": avg_end_error,
            "Label Error": avg_label_error,
            "Label Accuracy": avg_label_accuracy * 100,  # 记录标签准确率
            "Epoch": epoch + 1
        })
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


torch.cuda.set_device(0)

# Initialize ESM3 embedder
esm3_embedder = Esm3Embedding(pooling="mean")

# Set up device
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

embedding_dir = '/home/share/huadjyin/home/yinpeng/czw/code/Embedding_500_4'
max_len = 512
embedding_dim = 512

# Create the dataset and dataloader
label_mapping = build_label_mapping(embedding_dir)
print(f"Total unique labels: {len(label_mapping)}")

# 创建数据集
dataset = PreSavedEmbeddingDataset(embedding_dir=embedding_dir, label_mapping=label_mapping)
# Prepare dataset and dataloadersA
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

# 调用函数来检查训练集和验证集中的类别分布
check_label_distribution(train_dataset, val_dataset)
# Initialize model and optimizer
model = ProteinFunctionPredictor(embedding_dim=embedding_dim, label_dim=16, num_labels=len(label_mapping), max_len=max_len).to(device)

# 初始化模型、优化器和学习率调度器
learning_rate = 1e-5  # 可以先设置较大的学习率
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=3, factor=0.5, verbose=True)

# 开始训练和验证
train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=50, max_len=max_len, device=device, regression_weight=0.05, label_mapping=label_mapping)

# Finish wandb logging
wandb.finish()

