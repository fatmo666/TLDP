import torch
import torch.nn as nn
import torch.optim as optim
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
import torchtext

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 定义字段处理
tokenizer = get_tokenizer('spacy', language='en')


# 使用 torchtext 0.9+ 新 API，手动处理文本和标签
def process_data(dataset, tokenizer):
    texts, labels = [], []
    for label, text in dataset:
        text = tokenizer(text)
        texts.append(text)
        # 根据数据集中的标签定义进行处理
        if label == 1:  # 负面评论
            labels.append(1)
        elif label == 2:  # 正面评论
            labels.append(0)
    return texts, labels


# 指定IMDB数据集的根目录
root_dir = '/home/huangyu/lab/MobileNet/IMDB'

# 加载IMDB数据集，指定root目录
train_data, test_data = IMDB(split='train', root=root_dir), IMDB(split='test', root=root_dir)

from collections import Counter

# 打印一些标签样本以检查标签的实际格式
def print_sample_labels(dataset, dataset_name):
    print(f"Sample labels from {dataset_name}:")
    for i, (label, text) in enumerate(dataset):
        if i < 3:  # 只打印前5个标签样本
            print(f"Label {i+1}: {label} - {text}")
        else:
            break
    print("-" * 40)

# 打印训练集和测试集的标签样本
print_sample_labels(train_data, "Train Data")
print_sample_labels(test_data, "Test Data")


# 计算训练集和测试集的类别分布
def print_class_distribution(dataset, dataset_name):
    # 提取标签（已是数字格式 0 或 1）
    labels = [label for label, text in dataset]

    # 计算每个类别的频数
    label_counts = Counter(labels)

    # 打印类别分布
    print(f"{dataset_name} - Class Distribution:")
    print(f"Negative samples (1): {label_counts.get(1, 0)}")
    print(f"Positive samples (2): {label_counts.get(2, 0)}")
    print("-" * 40)


# 计算并打印训练集和测试集的类别分布
print_class_distribution(train_data, "Train Data")
print_class_distribution(test_data, "Test Data")

# 处理训练数据
train_texts, train_labels = process_data(train_data, tokenizer)
test_texts, test_labels = process_data(test_data, tokenizer)

# 使用GloVe预训练词向量
glove = GloVe(name='6B', dim=100)

# 添加 <unk> 标记到词汇表
vocab = glove.stoi
vocab['<unk>'] = len(vocab)  # 如果 <unk> 不在词汇表中，添加它


# 文本转为索引
def text_to_index(texts, vocab):
    return [[vocab.get(word, vocab['<unk>']) for word in text] for text in texts]


train_texts = text_to_index(train_texts, vocab)
test_texts = text_to_index(test_texts, vocab)

# 最大序列长度，用于填充
MAX_LENGTH = 500  # 你可以根据需要调整最大长度


# 填充序列到相同长度
def pad_sequence(sequences, max_length=MAX_LENGTH):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [vocab['<unk>']] * (max_length - len(seq))  # 填充至最大长度
        else:
            padded_seq = seq[:max_length]  # 截断至最大长度
        padded_sequences.append(padded_seq)
    return padded_sequences


# 填充训练和测试数据
train_texts = pad_sequence(train_texts)
test_texts = pad_sequence(test_texts)


# 转换为PyTorch数据集
class IMDBDataset(torch.utils.data.Dataset):
    def __init__(self, texts, labels):
        self.texts = texts
        self.labels = labels

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        return torch.tensor(self.texts[idx]), torch.tensor(self.labels[idx])


# 创建DataLoader
train_dataset = IMDBDataset(train_texts, train_labels)
test_dataset = IMDBDataset(test_texts, test_labels)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)



# 定义模型（RNN）
class RNNModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.rnn = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, text):
        embedded = self.embedding(text)
        rnn_out, (hidden, cell) = self.rnn(embedded)
        hidden = self.dropout(hidden[-1])
        return self.fc(hidden)


# 定义模型参数
vocab_size = len(vocab)
emb_dim = 100  # 使用100维的GloVe词向量
hidden_dim = 256
output_dim = 2  # 二分类
n_layers = 2
dropout = 0.5

# 初始化模型
model = RNNModel(vocab_size, emb_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())


# 训练模型
def train(model, iterator, optimizer, criterion):
    model.train()
    epoch_loss = 0
    epoch_acc = 0

    for batch in iterator:
        text, labels = batch
        text, labels = text.to(device), labels.to(device)

        optimizer.zero_grad()
        predictions = model(text)

        loss = criterion(predictions, labels)
        loss.backward()

        optimizer.step()

        epoch_loss += loss.item()
        _, predicted = torch.max(predictions, 1)
        epoch_acc += (predicted == labels).sum().item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset)


# 测试模型
def evaluate(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    epoch_acc = 0

    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)

            predictions = model(text)

            loss = criterion(predictions, labels)
            epoch_loss += loss.item()
            _, predicted = torch.max(predictions, 1)
            epoch_acc += (predicted == labels).sum().item()

    return epoch_loss / len(iterator), epoch_acc / len(iterator.dataset)


# 训练和评估
epochs = 100
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    valid_loss, valid_acc = evaluate(model, test_loader, criterion)

    print(f'Epoch {epoch + 1}/{epochs}')
    print(f'Training Loss: {train_loss:.3f} | Training Accuracy: {train_acc * 100:.2f}%')
    print(f'Validation Loss: {valid_loss:.3f} | Validation Accuracy: {valid_acc * 100:.2f}%')
