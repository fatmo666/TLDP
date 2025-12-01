from models.MLP import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, \
    average_precision_score, roc_auc_score
from utils.privacy_utils_new import *
from models.feature_model import MLPClassifier, MLPFeatureExtractor
from utils.feature_utils import extract_features

config = {
    "noise_type": "none",
    # "noise_type": "privacy",
    # "privacy_function": "laplace",
    "privacy_function": "rrldp_laplace",
    "epsilon": 10,
    "sensitivity": 2,
}

# 导入并预处理DarkNet
def loadDarkNet(file_path, split_num=100, test_start_index=0.8):
    # Load the CSV file
    df = pd.read_csv(file_path)
    print("Loaded CIC Darknet data successfully.")

    # 将字符串标签替换为数值标签
    df['Label'] = df['Label'].apply(lambda x: 0 if x in ['Non-Tor', 'NonVPN'] else 1)

    # 替换无穷大和极大值
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 对TimeStamp列进行排序
    df['Timestamp'] = pd.to_datetime(df['Timestamp'])
    df = df.sort_values(by='Timestamp')

    # 分离正常样本和异常样本
    normal_df = df[df['Label'] == 0]
    abnormal_df = df[df['Label'] == 1]

    # 将异常样本分成10组
    abnormal_groups = np.array_split(abnormal_df, split_num)

    # 初始化一个列表，用来存储间隔插入后的样本
    new_df = []

    # 计算每组正常样本之间的间隔大小
    normal_interval = len(normal_df) // split_num

    for i in range(split_num):
        # 取出一部分正常样本
        normal_part = normal_df.iloc[i * normal_interval: (i + 1) * normal_interval]
        # 添加正常样本到新数据集中
        new_df.append(normal_part)
        # 添加一组异常样本到新数据集中
        new_df.append(abnormal_groups[i])

    # 将剩余的正常样本添加到新数据集中
    remaining_normal = normal_df.iloc[split_num * normal_interval:]
    new_df.append(remaining_normal)

    # 将所有部分合并成一个新的 DataFrame
    new_df = pd.concat(new_df).reset_index(drop=True)

    # 计算数据集总样本数
    total_samples = len(new_df)

    # Step 1: 根据 test_start_index 划分测试集和训练集
    test_start = int(test_start_index * total_samples)
    train_df = new_df.iloc[:test_start]
    test_df = new_df.iloc[test_start:]

    # 去掉不必要的字符串列
    train_df = train_df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label.1'])
    test_df = test_df.drop(columns=['Flow ID', 'Src IP', 'Dst IP', 'Timestamp', 'Label.1'])

    # 数据标准化
    scaler = MinMaxScaler()

    # 处理训练集数据
    train_labels = train_df['Label'].values
    train_features = train_df.drop(columns=['Label']).values
    train_features_scaled = scaler.fit_transform(train_features)


    # 处理测试集数据
    test_labels = test_df['Label'].values
    test_features = test_df.drop(columns=['Label']).values
    test_features_scaled = scaler.fit_transform(test_features)

    # 转换为PyTorch张量
    train_data = torch.tensor(train_features_scaled, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_features_scaled, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 创建 TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    # 创建 DataLoader
    enumerate_batch_size = 640
    train_batch_size = 64
    train_loader = DataLoader(train_dataset, batch_size=enumerate_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False)

    # 保存 DataLoader
    torch.save(train_loader, 'DataLoader/DarkNet_train_loader_fcn.pth')
    torch.save(test_loader, 'DataLoader/DarkNet_test_loader_fcn.pth')

    # 保存 input_size 和 feature_names
    input_size = train_data.shape[1]
    with open(f'DataLoader/DarkNet_input_size.txt', 'w') as f:
        f.write(str(input_size))

    feature_names = (train_df.drop(columns=['Label']).columns.tolist())
    with open(f'DataLoader/DarkNet_feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    return train_loader, test_loader, input_size, feature_names

def train_model(model, train_loader, criterion, optimizer, num_epochs=10, patience=5, device="cuda"):
    model.train()

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for epoch in range(num_epochs):
        running_loss = 0.0
        for data, labels in train_loader:
            data, labels = data.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, labels.long())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        # # Validation phase
        # val_loss = 0.0
        # model.eval()  # Set model to evaluation mode for validation
        # with torch.no_grad():
        #     for val_data, val_labels in val_loader:
        #         val_data, val_labels = val_data.to(device), val_labels.to(device)
        #         val_outputs = model(val_data)
        #         val_loss += criterion(val_outputs, val_labels.long()).item()
        #
        # val_loss /= len(val_loader)  # Average validation loss

        # print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}, Val Loss: {val_loss:.4f}")
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss:.4f}")

        # # Early stopping logic
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     epochs_no_improve = 0
        #     torch.save(model.state_dict(), 'best_model.pth')  # Save the best model
        # else:
        #     epochs_no_improve += 1

        # if epochs_no_improve >= patience:
        #     print(f"Early stopping at epoch {epoch + 1}")
        #     break

        model.train()  # Set model back to training mode after validation

    print("Training complete.")

def evaluate_model(model, data_loader, device="cuda"):
    all_labels = []
    all_predictions = []

    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # 计算混淆矩阵
    cm = confusion_matrix(all_labels, all_predictions)

    # 从混淆矩阵计算 FPR 和 TPR
    tn, fp, fn, tp = cm.ravel()  # 提取混淆矩阵中的元素
    # # 如果标签反了，交换 tn 和 tp，fp 和 fn
    # tn, fp, fn, tp = tp, fn, fp, tn

    # 计算指标
    TPR = tp / (tp + fn)  # True Positive Rate, 召回率
    TNR = tn / (tn + fp)  # True Negative Rate, 特异性
    PPV = tp / (tp + fp)  # Positive Predictive Value, 精确度
    NPV = tn / (tn + fn)  # Negative Predictive Value
    FPR = fp / (fp + tn)  # False Positive Rate
    FNR = fn / (fn + tp)  # False Negative Rate
    FDR = fp / (fp + tp)  # False Discovery Rate
    ACC = (tp + tn) / (tp + tn + fp + fn)  # Accuracy

    AUC = roc_auc_score(all_labels, all_predictions)
    max_fpr = 0.05  # DarkNet
    # 计算在max_fpr下的原始部分AUC (pAUC)
    pAUC = roc_auc_score(all_labels, all_predictions, max_fpr=max_fpr)

    metrics = {
        'accuracy': accuracy_score(all_labels, all_predictions),
        'f1_score_macro': f1_score(all_labels, all_predictions, average='macro'),
        'f1_score_micro': f1_score(all_labels, all_predictions, average='micro'),
        'f1_score_weighted': f1_score(all_labels, all_predictions, average='weighted'),
        'f1_score_binary': f1_score(all_labels, all_predictions, average='binary'),
        'recall_macro': recall_score(all_labels, all_predictions, average='macro'),
        'recall_micro': recall_score(all_labels, all_predictions, average='micro'),
        'recall_weighted': recall_score(all_labels, all_predictions, average='weighted'),
        'recall_binary': recall_score(all_labels, all_predictions, average='binary'),
        'precision_macro': precision_score(all_labels, all_predictions, average='macro'),
        'precision_micro': precision_score(all_labels, all_predictions, average='micro'),
        'precision_weighted': precision_score(all_labels, all_predictions, average='weighted'),
        'precision_binary': precision_score(all_labels, all_predictions, average='binary'),
        'average_precision_macro': average_precision_score(all_labels, all_predictions, average='macro'),
        'average_precision_micro': average_precision_score(all_labels, all_predictions, average='micro'),
        'average_precision_weighted': average_precision_score(all_labels, all_predictions, average='weighted'),
        'confusion_matrix': confusion_matrix(all_labels, all_predictions).tolist(),  # Convert to list for JSON serialization
        'TPR': TPR,
        'TNR': TNR,
        'PPV': PPV,
        'NPV': NPV,
        'FPR': FPR,
        'FNR': FNR,
        'FDR': FDR,
        'AUC': AUC,
        'pAUC': pAUC,
    }

    return metrics, all_labels, all_predictions


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Select Device: ", device)

# 导入DarkNet
train_loader, test_loader, input_size, feature_names = loadDarkNet('DarkNet/Darknet.CSV')

# Deep Learning Model Training and Testing
hidden_size = 256
num_layers = 3
output_size = 128
num_classes = 2
deep_model = MLPClassifier(output_size=output_size, num_classes=num_classes)
deep_model = deep_model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(deep_model.parameters(), lr=0.001)

feature_extractor = MLPFeatureExtractor(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, num_classes=num_classes, dropout_rate=0.3, use_batch_norm=False).to(device)

all_features = []
all_targets = []
features, targets = extract_features(feature_extractor, device, train_loader)

# no noise ver.
if config["noise_type"] == "none":
    noisy_features = features
elif config["noise_type"] == "privacy":
    if config["privacy_function"] == "rrldp_laplace":
        noisy_features = add_noise_rrldp(features.to(device), epsilon=config["epsilon"],
                                       sensitivity=config["sensitivity"], noise_type='laplace')
    elif config["privacy_function"] == "rrldp_gaussian":
        noisy_features = add_noise_rrldp(features.to(device), epsilon=config["epsilon"],
                                       sensitivity=config["sensitivity"], noise_type='gaussian')
    elif config["privacy_function"] == "add_optimized_tensor_noise_Yang":
        import math
        sensitivity = 2 * math.sqrt(torch.numel(features))
        noisy_features = add_optimized_tensor_noise_Yang(features.to(device), epsilon=config["epsilon"],
                                                       sigma=10 ** (-5), sensitivity=sensitivity)
    elif config["privacy_function"] == "laplace":
        import math
        # sensitivity = 2 * math.sqrt(torch.numel(sample))
        scale = (torch.numel(features) * config["sensitivity"]) / config["epsilon"]
        noisy_features = add_laplace_noise(features.to(device), scale=scale)
    elif config["privacy_function"] == "gaussian":
        import math
        std = (torch.numel(features) * config["sensitivity"] ** 2) / (2 * config["epsilon"])
        # sensitivity = 2 * math.sqrt(torch.numel(sample))
        # std = (sensitivity * torch.log(1 / torch.tensor(1e-5))) / config["epsilon"]
        noisy_features = add_gaussian_noise(features.to(device), mean=0.0, std=std)

all_features = []
all_targets = []
all_features.append(noisy_features)
all_targets.append(targets)
all_features = torch.cat(all_features).to(device)
all_targets = torch.cat(all_targets).to(device)

dataset = torch.utils.data.TensorDataset(all_features, all_targets)
train_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

all_test_features = []
all_test_targets = []
test_features, test_targets = extract_features(feature_extractor, device, test_loader)
# noisy_features = add_laplace_noise(test_features, noise_scale=0.1)
all_test_features.append(test_features)
all_test_targets.append(test_targets)

all_test_features = torch.cat(all_test_features).to(device)
all_test_targets = torch.cat(all_test_targets).to(device)

dataset = torch.utils.data.TensorDataset(all_test_features, all_test_targets)
test_loader = torch.utils.data.DataLoader(dataset, batch_size=64, shuffle=True)

train_model(deep_model, train_loader, criterion, optimizer, num_epochs=20, patience=3, device=device)

metrics, labels_before, predictions_before = evaluate_model(deep_model, test_loader, device)
print(f"Deep Model After Training: ")
print(metrics)