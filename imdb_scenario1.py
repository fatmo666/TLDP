import torch
import torch.nn as nn
import torch.optim as optim
from torch.fx.traceback import set_grad_fn_seq_nr
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader
from torchtext.vocab import GloVe
from collections import Counter
from utils.privacy_utils_new import *
from test import compute_metrics

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 定义字段处理
tokenizer = get_tokenizer('spacy', language='en')

config = {
    "epsilon": 1,
    "sensitivity": 2,
    # "noise_type": "laplace",
    # "noise_type": "gaussian",
    # "noise_type": "add_optimized_tensor_noise_Yang",
    # "noise_type": "rrldp_laplace",
    "noise_type": "mvg_mechanism",
    # "noise_type": "rrldp_gaussian",
    "embedding_dim": 100,
    "hidden_dim": 256,
    "output_dim": 2,
    "n_layers": 2,
    "bidirectional": True,
    "dropout": 0.5,
    "learning_rate": 0.001,
    "batch_size": 64,
    "loss_function": "cross_entropy"
}

# 使用 BiLSTM 进行文本分类
class BiLSTMModel(nn.Module):
    def __init__(self, vocab_size, emb_dim, hidden_dim, output_dim, n_layers, dropout):
        super().__init__()

        # 嵌入层
        self.embedding = nn.Embedding(vocab_size, emb_dim)

        # 生成噪声并保存
        self.noise = self.generate_noise(vocab_size, emb_dim)

        # 双向LSTM层
        self.bilstm = nn.LSTM(emb_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True, bidirectional=True)

        # 全连接层
        self.fc = nn.Linear(hidden_dim * 2, output_dim)  # 双向LSTM的隐藏层数需要乘以2
        self.dropout = nn.Dropout(dropout)

    def generate_noise(self, vocab_size, emb_dim):
        """生成用于Embedding的噪声"""
        if config["noise_type"] == "rrldp_laplace":
            noise = add_noise_rrldp(
                torch.zeros((vocab_size, emb_dim)).to(device),
                epsilon=config["epsilon"],
                sensitivity=config["sensitivity"],
                noise_type='laplace'
            )
        elif config["noise_type"] == "rrldp_gaussian":
            noise = add_noise_rrldp(
                torch.zeros((vocab_size, emb_dim)).to(device),
                epsilon=config["epsilon"],
                sensitivity=config["sensitivity"],
                noise_type='gaussian'
            )
        elif config["noise_type"] == "laplace":
            import math
            # sensitivity = 2 * math.sqrt(torch.numel(sample))
            scale = (torch.numel(torch.zeros((vocab_size, emb_dim))) * config["sensitivity"]) / config["epsilon"]
            noise = add_laplace_noise(torch.zeros((vocab_size, emb_dim)).to(device), scale=scale)
        elif config["noise_type"] == "gaussian":
            import math
            std = (torch.numel(torch.zeros((vocab_size, emb_dim))) * config["sensitivity"] ** 2) / (2 * config["epsilon"])
            # sensitivity = 2 * math.sqrt(torch.numel(sample))
            # std = (sensitivity * torch.log(1 / torch.tensor(1e-5))) / config["epsilon"]
            noise = add_gaussian_noise(torch.zeros((vocab_size, emb_dim)).to(device), mean=0.0, std=std)
        elif config["noise_type"] == "add_optimized_tensor_noise_Yang":
            import math
            sensitivity = 2 * math.sqrt(torch.numel(torch.zeros((vocab_size, emb_dim))))
            noise = add_optimized_tensor_noise_Yang(torch.zeros((vocab_size, emb_dim)).to(device), epsilon=config["epsilon"],
                                                           sigma=10 ** (-5), sensitivity=sensitivity)
        return noise

    def forward(self, text):
        # embedded = self.embedding(text)
        # embedded = add_noise_rrldp(embedded.to(device), epsilon=config["epsilon"], sensitivity=config["sensitivity"], noise_type='laplace')

        if self.training:
            # 使用嵌入层并添加预生成的噪声
            embedded = self.embedding(text) + self.noise[text]
        else:
            embedded = self.embedding(text)

        # 双向LSTM
        lstm_out, (hidden, cell) = self.bilstm(embedded)

        # 取双向LSTM的最后一层的隐状态
        hidden = self.dropout(torch.cat((hidden[-2], hidden[-1]), dim=1))  # 拼接正向和反向的最后一层隐藏状态

        return self.fc(hidden)


# 数据预处理函数
def process_data(dataset, tokenizer):
    texts, labels = [], []
    for label, text in dataset:
        text = tokenizer(text)  # 使用 tokenizer 处理文本
        texts.append(text)

        if label == 1:  # 负面评论
            labels.append(0)
        elif label == 2:  # 正面评论
            labels.append(1)

    return texts, labels


# 计算训练集和测试集的类别分布
def print_class_distribution(dataset, dataset_name):
    labels = [label for _, label in dataset]
    label_counts = Counter(labels)
    print(f"{dataset_name} - Class Distribution:")
    print(f"Negative samples (1): {label_counts.get(1, 0)}")
    print(f"Positive samples (2): {label_counts.get(2, 0)}")
    print("-" * 40)


# 设置路径
root_dir = '/home/huangyu/lab/MobileNet/IMDB'

# 加载IMDB数据集，指定root目录
train_data, test_data = IMDB(split='train', root=root_dir), IMDB(split='test', root=root_dir)

# 处理训练数据
train_texts, train_labels = process_data(train_data, tokenizer)
test_texts, test_labels = process_data(test_data, tokenizer)

# 使用GloVe预训练词向量
glove = GloVe(name='6B', dim=100)
vocab = glove.stoi
vocab['<unk>'] = len(vocab)  # 如果 <unk> 不在词汇表中，添加它


# 文本转为索引
def text_to_index(texts, vocab):
    return [[vocab.get(word, vocab['<unk>']) for word in text] for text in texts]


train_texts = text_to_index(train_texts, vocab)
test_texts = text_to_index(test_texts, vocab)

# 填充序列到相同长度
MAX_LENGTH = 500  # 设置最大序列长度


def pad_sequence(sequences, max_length=MAX_LENGTH):
    padded_sequences = []
    for seq in sequences:
        if len(seq) < max_length:
            padded_seq = seq + [vocab['<unk>']] * (max_length - len(seq))  # 填充至最大长度
        else:
            padded_seq = seq[:max_length]  # 截断至最大长度
        padded_sequences.append(padded_seq)
    return padded_sequences


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

# 定义模型参数
vocab_size = len(vocab)
emb_dim = 100  # 使用100维的GloVe词向量
hidden_dim = 256
output_dim = 2  # 二分类
n_layers = 2
dropout = 0.5

# 初始化BiLSTM模型
model = BiLSTMModel(vocab_size, emb_dim, hidden_dim, output_dim, n_layers, dropout).to(device)

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


# 修改 evaluate 函数
def evaluate_with_metrics(model, iterator, criterion):
    model.eval()
    epoch_loss = 0
    all_targets = []
    all_predictions = []
    all_probabilities = []

    with torch.no_grad():
        for batch in iterator:
            text, labels = batch
            text, labels = text.to(device), labels.to(device)

            outputs = model(text)  # 模型输出 logits

            # probabilities = torch.softmax(outputs, dim=1) if outputs.shape[1] > 1 else torch.sigmoid(outputs)
            probabilities = torch.softmax(outputs, dim=1)  # 转换为概率

            # # 二分类时调整概率形状
            # if outputs.shape[1] == 1:
            #     probabilities = probabilities.squeeze(1)  # (batch_size, 1) -> (batch_size,)

            # 获取预测类别
            predictions = torch.argmax(probabilities, dim=1)  # 获取预测类别
            # predictions = torch.argmax(probabilities, dim=1) if outputs.shape[1] > 1 else (probabilities > 0.5).long()

            # 累积损失
            loss = criterion(outputs, labels)
            epoch_loss += loss.item()

            # 收集目标、预测和概率
            all_targets.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())

    # 计算指标
    metrics = compute_metrics(
        targets=np.array(all_targets),
        predictions=np.array(all_predictions),
        probabilities=np.array(all_probabilities)
    )
    # metrics = compute_metrics(
    #     targets=all_targets,
    #     predictions=all_predictions,
    #     probabilities=all_probabilities
    # )

    return epoch_loss / len(iterator), metrics

# 训练和评估
epochs = 50
for epoch in range(epochs):
    train_loss, train_acc = train(model, train_loader, optimizer, criterion)
    # valid_loss, valid_acc = evaluate(model, test_loader, criterion)
    valid_loss, valid_metrics = evaluate_with_metrics(model, test_loader, criterion)

    print(f'Validation Loss: {valid_loss:.3f}')
    print("Validation Metrics:")
    for key, value in valid_metrics.items():
        if key != "confusion_matrix":  # 排除混淆矩阵打印
            print(f"{key}: {value}")
    print("Confusion Matrix:")
    print(valid_metrics["confusion_matrix"])
