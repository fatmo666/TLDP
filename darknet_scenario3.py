from models.MLP import *
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import torch
from torch.utils.data import TensorDataset, DataLoader, Subset
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, confusion_matrix, roc_curve, \
    average_precision_score, roc_auc_score
from utils.privacy_utils_new import *
from utils.privacy_utils_new import _update_hessian_and_threshold_func_targeted
import ipaddress

config = {
    # "noise_type": "none",
    "noise_type": "privacy",
    # "privacy_function": "mvg_mechanism",
    # "privacy_function": "laplace",
    # "privacy_function": "gaussian",
    # "privacy_function": "add_optimized_tensor_noise_Yang",
    # "privacy_function": "rrldp_laplace",
    # "privacy_function": "rrldp_gaussian",
    # "privacy_function": "perturb_tensor_adaptive_iterative",
    "privacy_function": "adaptive_clipping_weight_perturb",
    "num_epochs": 10,
    "epsilon": 160,
    "sensitivity": 2,
}

def ip_to_int(ip_str):
    """将IP地址字符串转换为整数表示"""
    return int(ipaddress.ip_address(ip_str))

def loadTor(file_path, test_start_index=0.80, split_num=200, train_batch_size=64, enumerate_batch_size=6400):
    # Load the CSV file
    df = pd.read_csv(file_path)
    print("Loaded CIC Tor data successfully.")

    # 将字符串标签替换为数值标签
    # df['label'] = df['label'].apply(lambda x: 0 if x in ['nonTOR'] else 1)
    df['label'] = df['label'].apply(lambda x: 1 if x in ['nonTOR'] else 0)

    # 替换无穷大和极大值
    df = df.replace([np.inf, -np.inf], np.nan).dropna()

    # 转换 Src IP 和 Dst IP 为整数
    df['Source IP'] = df['Source IP'].apply(ip_to_int)
    df[' Destination IP'] = df[' Destination IP'].apply(ip_to_int)

    # 分离正常样本和异常样本
    normal_df = df[df['label'] == 0]
    abnormal_df = df[df['label'] == 1]

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

    # 数据标准化
    scaler = MinMaxScaler()

    # 处理训练集数据
    train_labels = train_df['label'].values
    train_features = train_df.drop(columns=['label']).values
    train_features_scaled = scaler.fit_transform(train_features)

    # 处理测试集数据
    test_labels = test_df['label'].values
    test_features = test_df.drop(columns=['label']).values
    test_features_scaled = scaler.fit_transform(test_features)

    # 转换为PyTorch张量
    train_data = torch.tensor(train_features_scaled, dtype=torch.float32)
    train_labels = torch.tensor(train_labels, dtype=torch.long)
    test_data = torch.tensor(test_features_scaled, dtype=torch.float32)
    test_labels = torch.tensor(test_labels, dtype=torch.long)

    # 创建 TensorDataset
    train_dataset = TensorDataset(train_data, train_labels)
    test_dataset = TensorDataset(test_data, test_labels)

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    # 创建 DataLoader
    train_loader = DataLoader(train_dataset, batch_size=enumerate_batch_size, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=train_batch_size, shuffle=False)

    # 保存 DataLoader
    torch.save(train_loader, 'DataLoader/Tor_train_loader_fcn.pth')
    torch.save(test_loader, 'DataLoader/Tor_test_loader_fcn.pth')

    # 保存 input_size 和 feature_names
    input_size = train_data.shape[1]
    with open(f'DataLoader/Tort_input_size.txt', 'w') as f:
        f.write(str(input_size))

    feature_names = train_df.drop(columns=['label']).columns.tolist()
    with open(f'DataLoader/Tor_feature_names.txt', 'w') as f:
        for feature in feature_names:
            f.write(f"{feature}\n")

    return train_loader, test_loader, input_size, feature_names

# 导入并预处理DarkNet
def loadDarkNet(file_path, split_num=200, test_start_index=0.8):
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

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    # 创建 DataLoader
    batch_size = 64
    # 分割训练数据集
    dataset_length = len(train_dataset)
    indices = np.arange(dataset_length)
    np.random.shuffle(indices)

    num_splits = 3
    split_size = dataset_length // num_splits
    split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

    if dataset_length % num_splits != 0:
        # 如果无法平均分割，将多余样本分配到最后一个分割
        split_indices[-1] = np.concatenate([split_indices[-1], indices[num_splits * split_size:]])

    # 为每个分割创建 DataLoader
    train_loaders = [DataLoader(Subset(train_dataset, split_indices[i]), batch_size=batch_size, shuffle=True)
                     for i in range(num_splits)]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    input_size = train_data.shape[1]

    feature_names = (train_df.drop(columns=['Label']).columns.tolist())

    return train_loaders, test_loader, input_size, feature_names


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

def train_local(model, train_loader, criterion, optimizer, num_epochs=10, patience=5, device="cuda"):
    model.train()

    for data, labels in train_loader:
        data, labels = data.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(data)
        loss = criterion(outputs, labels.long())
        loss.backward()
        optimizer.step()

    print("Training complete.")

def evaluate_model(model, data_loader, device="cuda"):
    all_labels = []
    all_predictions = []
    all_probabilities = []

    model.eval()
    with torch.no_grad():
        for data, labels in data_loader:
            data, labels = data.to(device), labels.to(device)
            outputs = model(data)
            _, predicted = torch.max(outputs.data, 1)
            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())
            # probabilities = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            probabilities = torch.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            all_probabilities.extend(probabilities)

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
        'average_precision_macro': average_precision_score(all_labels, all_probabilities, average='macro'),
        'average_precision_micro': average_precision_score(all_labels, all_probabilities, average='micro'),
        'average_precision_weighted': average_precision_score(all_labels, all_probabilities, average='weighted'),
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
                sensitivity = clipped_tensor.cpu().max().item()
                epsilon = config["epsilon"]
                std = epsilon * abs(sensitivity)
                # std = epsilon * sensitivity
                noise = torch.distributions.Normal(0, std).sample(clipped_tensor.shape).to(device)
                noisy_param = clipped_tensor + noise

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
            # if "weight" in key or "bias" in key:
            if "weight" in key or "bias" in key:

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


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
print("Select Device: ", device)

# 导入DarkNet
train_loaders, test_loader, input_size, feature_names = loadDarkNet('DarkNet/Darknet.CSV')
# train_loader, test_loader, input_size, feature_names = loadTor('Tor-Scenario-A/Scenario-A-merged_5s.csv')

# Deep Learning Model Training and Testing
hidden_size = 256
num_layers = 3
output_size = 128
num_classes = 2

# 初始化BiLSTM模型
global_model = FCNClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, num_classes=num_classes, dropout_rate=0.5, use_batch_norm=False).to(device)
local_models = [FCNClassifier(input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, output_size=output_size, num_classes=num_classes, dropout_rate=0.5, use_batch_norm=False).to(device) for _ in range(3)]
optimizers = [optim.Adam(local_model.parameters(), lr=0.001) for local_model in local_models]
criterion = nn.CrossEntropyLoss()

# # no noise ver.
# if config["noise_type"] == "none":
#     noisy_data = features
# elif config["noise_type"] == "privacy":
#     if config["privacy_function"] == "rrldp_laplace":
#         noisy_data = add_noise_rrldp(features.to(device), epsilon=config["epsilon"],
#                                        sensitivity=config["sensitivity"], noise_type='laplace')
#     elif config["privacy_function"] == "rrldp_gaussian":
#         noisy_data = add_noise_rrldp(features.to(device), epsilon=config["epsilon"],
#                                        sensitivity=config["sensitivity"], noise_type='gaussian')
#     elif config["privacy_function"] == "add_optimized_tensor_noise_Yang":
#         import math
#         sensitivity = 2 * math.sqrt(torch.numel(features))
#         noisy_data = add_optimized_tensor_noise_Yang(features.to(device), epsilon=config["epsilon"],
#                                                        sigma=10 ** (-5), sensitivity=sensitivity)
#     elif config["privacy_function"] == "laplace":
#         import math
#         # sensitivity = 2 * math.sqrt(torch.numel(sample))
#         scale = (torch.numel(features) * config["sensitivity"]) / config["epsilon"]
#         noisy_data = add_laplace_noise(features.to(device), scale=scale)
#     elif config["privacy_function"] == "gaussian":
#         import math
#         std = (torch.numel(features) * config["sensitivity"] ** 2) / (2 * config["epsilon"])
#         # sensitivity = 2 * math.sqrt(torch.numel(sample))
#         # std = (sensitivity * torch.log(1 / torch.tensor(1e-5))) / config["epsilon"]
#         noisy_data = add_gaussian_noise(features.to(device), mean=0.0, std=std)


adaptive_clipping_states = [{} for _ in range(len(local_models))]
for epoch in range(1, config['num_epochs'] + 1):
    print(f'Starting epoch {epoch}')
    # 使用分割后的加载器进行训练
    for i, train_loader in enumerate(train_loaders):
        train_local(local_models[i], train_loader, criterion, optimizers[i], num_epochs=25, patience=3, device=device)

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

metrics, labels_before, predictions_before = evaluate_model(global_model, test_loader, device)
print(f"Deep Model After Training: ")
print(metrics)