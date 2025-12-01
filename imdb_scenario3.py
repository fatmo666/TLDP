import torch
import torch.nn as nn
import torch.optim as optim
from torch.fx.traceback import set_grad_fn_seq_nr
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Subset
from torchtext.vocab import GloVe
from collections import Counter
from utils.privacy_utils_new import *
from utils.privacy_utils_new import _update_hessian_and_threshold_func_targeted
from test import compute_metrics
import math

# 设置设备
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')

# 定义字段处理
tokenizer = get_tokenizer('spacy', language='en')

config = {
    "epsilon": 160,
    "sensitivity": 2,
    "noise_type": "privacy",
    # "privacy_function": "laplace",
    # "privacy_function": "gaussian",
    # "privacy_function": "add_optimized_tensor_noise_Yang",
    # "privacy_function": "mvg_mechanism",
    # "privacy_function": "rrldp_laplace",
    # "privacy_function": "rrldp_gaussian",
    # "privacy_function": "perturb_tensor_adaptive_iterative",
    "privacy_function": "adaptive_clipping_weight_perturb",
    "num_epochs": 10,
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
        if config["privacy_function"] == "rrldp_laplace":
            noise = add_noise_rrldp(
                torch.zeros((vocab_size, emb_dim)).to(device),
                epsilon=config["epsilon"],
                sensitivity=config["sensitivity"],
                noise_type='laplace'
            )
        elif config["privacy_function"] == "rrldp_gaussian":
            noise = add_noise_rrldp(
                torch.zeros((vocab_size, emb_dim)).to(device),
                epsilon=config["epsilon"],
                sensitivity=config["sensitivity"],
                noise_type='gaussian'
            )
        elif config["privacy_function"] == "laplace":
            import math
            # sensitivity = 2 * math.sqrt(torch.numel(sample))
            scale = (torch.numel(torch.zeros((vocab_size, emb_dim))) * config["sensitivity"]) / config["epsilon"]
            noise = add_laplace_noise(torch.zeros((vocab_size, emb_dim)).to(device), scale=scale)
        elif config["privacy_function"] == "gaussian":
            import math
            std = (torch.numel(torch.zeros((vocab_size, emb_dim))) * config["sensitivity"] ** 2) / (2 * config["epsilon"])
            # sensitivity = 2 * math.sqrt(torch.numel(sample))
            # std = (sensitivity * torch.log(1 / torch.tensor(1e-5))) / config["epsilon"]
            noise = add_gaussian_noise(torch.zeros((vocab_size, emb_dim)).to(device), mean=0.0, std=std)
        elif config["privacy_function"] == "add_optimized_tensor_noise_Yang":
            import math
            sensitivity = 2 * math.sqrt(torch.numel(torch.zeros((vocab_size, emb_dim))))
            noise = add_optimized_tensor_noise_Yang(torch.zeros((vocab_size, emb_dim)).to(device), epsilon=config["epsilon"],
                                                           sigma=10 ** (-5), sensitivity=sensitivity)
        elif config["privacy_function"] == "mvg_mechanism":
            # noise = mvg_mechanism(X=torch.zeros((vocab_size, emb_dim)).to(device), epsilon=config["epsilon"], delta=10 ** (-5), theta=[0.55 / 2, 0.55 / 2] + [0.45 / (torch.zeros((vocab_size, emb_dim)).shape[0] - 2)] * (torch.zeros((vocab_size, emb_dim)).shape[0] - 2))
            noise = 0
        elif config["privacy_function"] == "perturb_tensor_adaptive_iterative":
            # noise = mvg_mechanism(X=torch.zeros((vocab_size, emb_dim)).to(device), epsilon=config["epsilon"], delta=10 ** (-5), theta=[0.55 / 2, 0.55 / 2] + [0.45 / (torch.zeros((vocab_size, emb_dim)).shape[0] - 2)] * (torch.zeros((vocab_size, emb_dim)).shape[0] - 2))
            noise = 0
        elif config["privacy_function"] == "adaptive_clipping_weight_perturb":
            # noise = mvg_mechanism(X=torch.zeros((vocab_size, emb_dim)).to(device), epsilon=config["epsilon"], delta=10 ** (-5), theta=[0.55 / 2, 0.55 / 2] + [0.45 / (torch.zeros((vocab_size, emb_dim)).shape[0] - 2)] * (torch.zeros((vocab_size, emb_dim)).shape[0] - 2))
            noise = 0

        return noise

    def forward(self, text):
        # embedded = self.embedding(text)
        # embedded = add_noise_rrldp(embedded.to(device), epsilon=config["epsilon"], sensitivity=config["sensitivity"], noise_type='laplace')

        if self.training:
            # 使用嵌入层并添加预生成的噪声
            # embedded = self.embedding(text) + self.noise[text]
            embedded = self.embedding(text)
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


# 定义模型参数
vocab_size = len(vocab)
emb_dim = 100  # 使用100维的GloVe词向量
hidden_dim = 256
output_dim = 2  # 二分类
n_layers = 2
dropout = 0.5

# all_embeddings = []
# model = BiLSTMModel(vocab_size, emb_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
# # 遍历训练数据集，收集所有嵌入向量
# for text, _ in train_dataset:
#     text = text.to(device)
#     embedded = model.embedding(text)  # 获取嵌入向量
#     all_embeddings.append(embedded.cpu().detach().numpy())  # 将结果转移到CPU并转为numpy数组
#
# # 将所有嵌入向量合并成一个大的数组（可以按批次合并）
# all_embeddings = np.concatenate(all_embeddings, axis=0)
#
# # 计算整个嵌入向量的最小值和最大值
# embedded_min = np.min(all_embeddings)
# embedded_max = np.max(all_embeddings)
#
# print(f"整个数据集嵌入向量的取值范围：{embedded_min} 到 {embedded_max}")

# 将训练数据集分割成 num_splits 份
num_splits = 3
dataset_length = len(train_dataset)
indices = np.arange(dataset_length)
np.random.shuffle(indices)

split_size = dataset_length // num_splits
split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

if dataset_length % num_splits != 0:
    # 如果无法平均分割，将多余样本分配到最后一个分割
    split_indices[-1] = np.concatenate([split_indices[-1], indices[num_splits * split_size:]])

# 为每个分割创建 DataLoader
train_loaders = [DataLoader(Subset(train_dataset, split_indices[i]), batch_size=64, shuffle=True)
                 for i in range(num_splits)]
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# 定义模型参数
vocab_size = len(vocab)
emb_dim = 100  # 使用100维的GloVe词向量
hidden_dim = 256
output_dim = 2  # 二分类
n_layers = 2
dropout = 0.5

# 初始化BiLSTM模型
global_model = BiLSTMModel(vocab_size, emb_dim, hidden_dim, output_dim, n_layers, dropout).to(device)
local_models = [BiLSTMModel(vocab_size, emb_dim, hidden_dim, output_dim, n_layers, dropout).to(device) for _ in range(3)]
optimizers = [optim.Adam(local_model.parameters(), lr=0.001) for local_model in local_models]
criterion = nn.CrossEntropyLoss()


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

        # 梯度裁剪，限制梯度的L2范数
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

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

def aggregate_models(global_model, local_models, device, config, total_rounds, current_round):
    """
    聚合多个local_models为一个global_model。

    参数：
    - global_model: 全局模型，结构与local_models相同。
    - local_models: 多个局部模型列表，每个模型是一个与global_model结构相同的PyTorch模型。
    - device: 模型所在设备 (例如 "cpu" 或 "cuda")。

    返回：
    - 聚合后的global_model。
    """
    weight_keys = [key for key in global_model.state_dict().keys() if 'weight' in key]
    if not weight_keys:
        # 如果模型中没有权重层，则设置为空
        first_layer_name, last_layer_name = None, None
    else:
        first_layer_name = weight_keys[0]
        last_layer_name = weight_keys[-1]

    # 将模型权重初始化为零
    global_dict = global_model.state_dict()

    # 对所有局部模型的权重进行平均聚合
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key], device=device, dtype=torch.float)

    # 遍历每个局部模型
    for local_model in local_models:
        # 获取局部模型的参数
        local_dict = local_model.state_dict()

        # 对每个参数进行加权聚合
        for key in global_dict:
            # if "weight" in key:
            if "weight" in key or "bias" in key:

                # clip_value = 1
                clip_value = 1
                # clip_value = 10e-1
                sensitivity = config["sensitivity"] * clip_value

                # global_dict[key] += local_dict[key].to(device)  # 加上局部模型的权重
                if config["noise_type"] == "none":
                    noisy_param = local_dict[key].to(device)  # 加上局部模型的权重
                elif config["noise_type"] == "privacy":
                    if config["privacy_function"] == "rrldp_laplace":
                        noisy_param = add_noise_rrldp(local_dict[key].to(device), epsilon=config["epsilon"],
                                                         sensitivity=sensitivity, noise_type='laplace')
                    elif config["privacy_function"] == "rrldp_gaussian":
                        noisy_param = add_noise_rrldp(local_dict[key].to(device), epsilon=config["epsilon"],
                                                         sensitivity=config["sensitivity"], noise_type='gaussian')
                    elif config["privacy_function"] == "add_optimized_tensor_noise_Yang":
                        import math
                        sensitivity = 2 * math.sqrt(torch.numel(local_dict[key])) * clip_value
                        noisy_param = add_optimized_tensor_noise_Yang(local_dict[key].to(device), epsilon=config["epsilon"],
                                                                         sigma=10 ** (-5), sensitivity=sensitivity)
                    elif config["privacy_function"] == "laplace":
                        import math
                        # sensitivity = 2 * math.sqrt(torch.numel(sample))
                        scale = (torch.numel(local_dict[key]) * config["sensitivity"]) / config["epsilon"]
                        noisy_param = add_laplace_noise(local_dict[key].to(device), scale=scale)
                    elif config["privacy_function"] == "gaussian":
                        import math
                        std = (torch.numel(local_dict[key]) * config["sensitivity"] ** 2) / (2 * config["epsilon"])
                        # sensitivity = 2 * math.sqrt(torch.numel(sample))
                        # std = (sensitivity * torch.log(1 / torch.tensor(1e-5))) / config["epsilon"]
                        noisy_param = add_gaussian_noise(local_dict[key].to(device), mean=0.0, std=std)
                    elif config["privacy_function"] == "mvg_mechanism":
                        if local_dict[key].shape[0] < 10000:
                            if len(local_dict[key].shape) == 1:
                                if local_dict[key].shape[0] == 2:
                                    noisy_param = mvg_mechanism_2d(X=local_dict[key].unsqueeze(0).to(device), epsilon=config["epsilon"], delta=10 ** (-5),
                                                     theta=[0.55 / 2, 0.55 / 2])
                                else:
                                    noisy_param = mvg_mechanism_2d(X=local_dict[key].unsqueeze(0).to(device),
                                                                   epsilon=config["epsilon"], delta=10 ** (-5),
                                                                   theta=[0.55 / 2, 0.55 / 2] + [0.45 / (local_dict[key].shape[0] - 2)] * (local_dict[key].shape[0] - 2))
                            else:
                                if local_dict[key].view(-1, local_dict[key].size(1)).shape[0] == 2:
                                    noisy_param = mvg_mechanism(X=local_dict[key].to(device), epsilon=config["epsilon"],
                                                                   delta=10 ** (-5),
                                                                   theta=[0.55 / 2, 0.55 / 2])
                                else:
                                    noisy_param = mvg_mechanism(X=local_dict[key].to(device), epsilon=config["epsilon"],
                                                                delta=10 ** (-5),
                                                                theta=[0.55 / 2, 0.55 / 2] + [0.45 / (
                                                                            local_dict[key].view(-1, local_dict[key].size(
                                                                                1)).shape[0] - 2)] * (local_dict[key].view(-1, local_dict[key].size(1)).shape[0] - 2))
                        else:
                            noisy_param = torch.zeros(local_dict[key].shape).to(device)
                    elif config["privacy_function"] == "perturb_tensor_adaptive_iterative":
                        if key == first_layer_name:
                            noisy_param = perturb_tensor_adaptive_iterative(tensor=local_dict[key].to(device), total_epsilon=config["epsilon"], sensitivity=sensitivity,
                                                                            total_rounds=total_rounds, current_round=current_round, gamma=(1 / (2 * total_rounds)), is_first_layer=True, N=2)
                        elif key == last_layer_name:
                            noisy_param = perturb_tensor_adaptive_iterative(tensor=local_dict[key].to(device),
                                                                            total_epsilon=config["epsilon"],
                                                                            sensitivity=sensitivity,
                                                                            total_rounds=total_rounds,
                                                                            current_round=current_round,
                                                                            gamma=(1 / (2 * total_rounds)),
                                                                            is_first_layer=False, N=2)
                        else:
                            noisy_param = local_dict[key].to(device)


                global_dict[key] += noisy_param
            else:
                global_dict[key] += local_dict[key].to(device)


    # 计算平均
    num_models = len(local_models)
    for key in global_dict:
        global_dict[key] /= num_models  # 对权重取平均

    # 裁剪权重
    # for key in global_dict:
        # global_dict[key].clamp_(-10e-4, 10e-4) # 永远低结果
        global_dict[key].clamp_(-clip_value, clip_value) # 稍微好一点

    # 更新全局模型的权重
    global_model.load_state_dict(global_dict)

    return global_model

def aggregate_models_adaptive_clipping(global_model, local_models, local_data_loaders,
                                       adaptive_clipping_states, criterion, device, config, current_round):
    global_dict = global_model.state_dict()
    for key in global_dict:
        global_dict[key] = torch.zeros_like(global_dict[key], device=device, dtype=torch.float)

    next_adaptive_clipping_states = [{} for _ in range(len(local_models))]

    for i, local_model in enumerate(local_models):
        client_state = adaptive_clipping_states[i]
        client_loader = local_data_loaders[i]

        num_data_samples = 64
        num_total_samples = len(client_loader.dataset)
        if num_total_samples == 0: raise ValueError("Dataset is empty.")
        # 创建所有样本的索引
        indices = list(range(num_total_samples))
        random.shuffle(indices)  # 打乱索引
        # 选择子集的索引
        subset_indices = indices[:num_data_samples]
        # 创建一个只从该子集采样的采样器和加载器
        subset_sampler = SubsetRandomSampler(subset_indices)
        temp_loader = torch.utils.data.DataLoader(
            client_loader.dataset,
            batch_size=len(subset_indices),
            sampler=subset_sampler
        )

        # 从这个代表“小数据库”的加载器中获取唯一的一批数据
        inputs, targets = next(iter(temp_loader))
        inputs, targets = inputs.to(device), targets.to(device)

        try:
            with torch.no_grad():
                current_loss = criterion(local_model(inputs.to(device)), targets.to(device)).item()
        except StopIteration:
            current_loss = 0.0

        # --- 1. 为当前客户端计算一次 C_adaptive ---
        hessian_update_counter = client_state.get('hessian_update_counter', 0)
        interval_phi = client_state.get('interval_phi', config.get('adaptive_clip_interval', 5))
        C_adaptive = client_state.get('adaptive_clipping_threshold', None)
        loss_list = client_state.get('loss_list', [])
        pre_mean_loss = client_state.get('pre_mean_loss', float('inf'))
        loss_list.append(current_loss)

        if hessian_update_counter % interval_phi == 0:
            new_C_adaptive = _update_hessian_and_threshold_func_targeted(
                model=local_model,
                loss_fn=criterion,
                inputs=inputs,
                targets=targets,
                device=device,
                # hessian_approx_d=config.get('hessian_approx_d', 1e-2),
                hessian_approx_d=config.get('hessian_approx_d', 100),
            )

            if new_C_adaptive is not None:
                C_adaptive = new_C_adaptive
                print(f"[Client {i}] New adaptive clipping threshold C_adaptive: {C_adaptive:.4f}")

            if len(loss_list) == interval_phi:
                mean_loss = np.mean(loss_list)
                a = config.get('adaptive_clip_a', 1)
                b = config.get('adaptive_clip_b', 1)
                if mean_loss < pre_mean_loss:
                    interval_phi += a
                else:
                    interval_phi -= b
                if interval_phi <= 0: interval_phi = 1

                print(f"[Client {i}] Mean loss: {mean_loss:.4f}. New interval: {interval_phi}")
                pre_mean_loss = mean_loss
                loss_list = []
                hessian_update_counter = 0

        if C_adaptive is None: C_adaptive = torch.tensor(1.0, device=device)

        # --- 2. 遍历该客户端的所有层 ---
        local_dict = local_model.state_dict()
        for key in local_dict:

            # <<< 只对权重和偏置进行裁剪和加噪 >>>
            if "weight" in key or "bias" in key:
                tensor = local_dict[key].to(device)

                # a. 自适应裁剪
                tensor_norm = torch.norm(tensor, p=2)
                clip_coef = C_adaptive / (tensor_norm + 1e-6)
                clipped_tensor = tensor * clip_coef if clip_coef < 1.0 else tensor

                # b. 添加噪音
                if torch.isnan(clipped_tensor.cpu()).all().item() != True:
                    sensitivity = clipped_tensor.cpu().max().item()
                    epsilon = config["epsilon"]
                    if math.isnan(sensitivity):
                        sensitivity = clipped_tensor[~torch.isnan(clipped_tensor)].max().item()
                    std = epsilon * abs(sensitivity)
                    noise = torch.distributions.Normal(0, std).sample(clipped_tensor.shape).to(device)
                    noisy_param = clipped_tensor + noise
                else:
                    noisy_param = clipped_tensor

                global_dict[key] += noisy_param
            else:
                # 对于非权重/偏置的参数（如BN层的running_mean），执行标准聚合
                global_dict[key] += local_dict[key].to(device)

        # --- 4. 保存该客户端的下一轮状态 ---
        next_adaptive_clipping_states[i] = {
            'hessian_update_counter': hessian_update_counter + 1,
            'interval_phi': interval_phi,
            'adaptive_clipping_threshold': C_adaptive,
            'loss_list': loss_list,
            'pre_mean_loss': pre_mean_loss
        }

    # --- 聚合收尾 ---
    num_models = len(local_models)
    for key in global_dict:
        global_dict[key] /= num_models

    global_model.load_state_dict(global_dict)

    return global_model, next_adaptive_clipping_states

adaptive_clipping_states = [{} for _ in range(len(local_models))]
for epoch in range(1, config['num_epochs'] + 1):
    print(f'Starting epoch {epoch}')
    # 使用分割后的加载器进行训练
    for i, train_loader in enumerate(train_loaders):
        train_loss, train_acc = train(local_models[i], train_loader, optimizers[i], criterion)

    # 每轮后聚合模型
    # global_model = aggregate_models(global_model, local_models, device, config=config, total_rounds=config['num_epochs'] + 1, current_round=epoch)
    privacy_function = config.get("privacy_function", "none")
    if privacy_function == "adaptive_clipping_weight_perturb":
        # 调用专用的、有状态的聚合函数
        global_model, adaptive_clipping_states = aggregate_models_adaptive_clipping(
            global_model,
            local_models,
            train_loaders,  # 传入数据加载器
            adaptive_clipping_states,  # 传入并接收状态
            criterion,  # 传入损失函数
            device, config, epoch
        )
    else:
        # 调用原来的、通用的、无状态的聚合函数
        global_model = aggregate_models(
            global_model, local_models, device, config,
            total_rounds=config['num_epochs'] + 1, current_round=epoch
        )

    # 将全局模型更新到所有本地模型
    global_state_dict = global_model.state_dict()
    for local_model in local_models:
        local_model.load_state_dict(global_state_dict)

    print(f'Epoch {epoch} aggregation and update completed')

    # 将全局模型更新到所有本地模型
    global_state_dict = global_model.state_dict()
    for local_model in local_models:
        local_model.load_state_dict(global_state_dict)

    print(f'Epoch {epoch} aggregation and update completed')
    # test_loss, metrics, predictions, targets = test(global_model, device, test_loader, criterion, target_float=config["target_float"])

valid_loss, valid_metrics = evaluate_with_metrics(global_model, test_loader, criterion)

print(f'Validation Loss: {valid_loss:.3f}')
print("Validation Metrics:")
for key, value in valid_metrics.items():
    if key != "confusion_matrix":  # 排除混淆矩阵打印
        print(f"{key}: {value}")
print("Confusion Matrix:")
print(valid_metrics["confusion_matrix"])

# # 训练和评估
# epochs = 50
# for epoch in range(epochs):
#     train_loss, train_acc = train(model, train_loader, optimizer, criterion)
#     # valid_loss, valid_acc = evaluate(model, test_loader, criterion)
#     valid_loss, valid_metrics = evaluate_with_metrics(model, test_loader, criterion)
#
#     print(f'Validation Loss: {valid_loss:.3f}')
#     print("Validation Metrics:")
#     for key, value in valid_metrics.items():
#         if key != "confusion_matrix":  # 排除混淆矩阵打印
#             print(f"{key}: {value}")
#     print("Confusion Matrix:")
#     print(valid_metrics["confusion_matrix"])
