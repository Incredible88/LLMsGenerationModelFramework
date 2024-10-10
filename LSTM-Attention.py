import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
import pickle
import os
from tqdm import tqdm
import wandb

# Initialize wandb
wandb.login(key="5f5f94d3de9157cf146ad88ecc4e0518a7a7549e")
wandb.init(project="protein-function-transformer-lstm", config={
    "learning_rate": 1e-4,
    "epochs": 20,
    "batch_size": 4,
    "model": "TransformerLSTMClassifier"
})

class PreSavedEmbeddingDataset(Dataset):
    def __init__(self, embedding_dir, label_mapping, max_files=2000):
        self.embedding_dir = embedding_dir
        self.files = [f for f in os.listdir(embedding_dir) if f.endswith('.pkl')]
        self.label_mapping = label_mapping
        self.files = self.files[:max_files]
        self.progress_bar = tqdm(total=len(self.files), desc="Loading Embeddings")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        file_name = self.files[idx]
        file_path = os.path.join(self.embedding_dir, file_name)
        with open(file_path, 'rb') as pkl_file:
            data = pickle.load(pkl_file)
            embedding = torch.tensor(data['embedding'], dtype=torch.float32)
            label_str = data['label']
            label = self.label_mapping[label_str]
            start = int(data['start'])
            end = int(data['end'])
        self.progress_bar.update(1)
        return embedding, torch.tensor(label, dtype=torch.long)

    def __del__(self):
        self.progress_bar.close()

# LSTM with Attention
class AttentionLSTMClassifier(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, num_layers, num_labels):
        super(AttentionLSTMClassifier, self).__init__()
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, num_labels)
        self.label_fc = nn.Linear(num_labels, 16)

    def attention(self, lstm_out):
        attn_weights = torch.bmm(lstm_out, lstm_out.transpose(1, 2))
        attn_weights = torch.softmax(attn_weights, dim=-1)
        attn_applied = torch.bmm(attn_weights, lstm_out)
        return attn_applied

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attn_out = self.attention(lstm_out)
        lstm_out = attn_out[:, -1, :]
        logits = self.fc(lstm_out)
        return logits

# Loss function and accuracy calculation
classification_loss_fn = nn.CrossEntropyLoss()

def compute_loss(logits, true_labels):
    return classification_loss_fn(logits, true_labels)

def compute_accuracy(logits, true_labels):
    _, predicted_labels = torch.max(logits, 1)
    correct_predictions = (predicted_labels == true_labels).float().sum()
    accuracy = correct_predictions / true_labels.size(0)
    return accuracy, predicted_labels

# Training and evaluation loop
def train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=10, device="cpu"):
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0

        for embeddings, labels in train_loader:
            embeddings, labels = embeddings.to(device), labels.to(device)

            optimizer.zero_grad()

            # 模型的前向传播
            predicted_logits = model(embeddings)

            loss = compute_loss(predicted_logits, labels)
            loss.backward()
            optimizer.step()

            total_train_loss += loss.item()

        avg_train_loss = total_train_loss / len(train_loader)
        print(f"Epoch {epoch + 1}/{num_epochs}, Training Loss: {avg_train_loss}")

        # Validation phase
        model.eval()
        total_val_loss = 0
        total_accuracy = 0

        with torch.no_grad():
            for embeddings, labels in val_loader:
                embeddings, labels = embeddings.to(device), labels.to(device)

                predicted_logits = model(embeddings)

                val_loss = compute_loss(predicted_logits, labels)
                total_val_loss += val_loss.item()

                accuracy, _ = compute_accuracy(predicted_logits, labels)
                total_accuracy += accuracy.item()

        avg_val_loss = total_val_loss / len(val_loader)
        avg_accuracy = total_accuracy / len(val_loader)

        print(f"Epoch {epoch + 1}, Validation Loss: {avg_val_loss}, Accuracy: {avg_accuracy * 100:.2f}%")

        # Log metrics to wandb
        wandb.log({
            "epoch": epoch + 1,
            "avg_train_loss": avg_train_loss,
            "avg_val_loss": avg_val_loss,
            "avg_accuracy": avg_accuracy * 100
        })

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

# Load data and setup the model
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
embedding_dir = '/home/share/huadjyin/home/yinpeng/czw/code/Embedding_500'
label_to_idx = generate_label_to_idx(embedding_dir, max_files=50000)
num_labels = len(label_to_idx)

dataset = PreSavedEmbeddingDataset(embedding_dir, label_to_idx, max_files=50000)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=4, shuffle=False)

# Initialize the LSTM model with attention
embedding_dim = 512
hidden_dim = 256
num_layers = 4
model = AttentionLSTMClassifier(embedding_dim, hidden_dim, num_layers, num_labels).to(device)

# Optimizer
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Start training and evaluating
train_and_evaluate(model, train_loader, val_loader, optimizer, num_epochs=20, device=device)

# End wandb run
wandb.finish()
