import os
import torch
import torch.optim as optim
import torch.nn as nn
from models.mobilenet import *
from models.BiLSTM import *
from utils.data_utils import *
from test import *
from utils.privacy_utils_new import *
from utils.privacy_utils_new import _update_hessian_and_threshold_func_targeted
import yaml
import argparse
import logging
import json
import time

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_experiment_info(experiment_dir, info):
    with open(os.path.join(experiment_dir, 'experiment_info.json'), 'w') as f:
        json.dump(info, f, indent=4)

def save_predictions(experiment_dir, predictions, targets):
    torch.save({'predictions': predictions, 'targets': targets}, os.path.join(experiment_dir, 'predictions.pt'))


# ==========================================
# 打印模型详情的函数
# ==========================================
def print_model_details(model, config, device):
    """
    不使用 if 判断，直接依次打印 MNIST(3通道)、CIFAR-10 和 SVHN 的模型结构 summary。
    """
    try:
        from torchinfo import summary
        import traceback

        # 定义要打印的三个数据集场景
        # 格式: (数据集名称, 输入尺寸)
        scenarios = [
            ("MNIST (Force 3-Channel)", (1, 3, 28, 28)),
            ("CIFAR-10", (1, 3, 32, 32)),
            ("SVHN", (1, 3, 32, 32))
        ]

        # 临时切换到评估模式 (避免 BatchNorm 在 batch_size=1 时报错)
        original_mode = model.training
        model.eval()

        logger.info("=" * 60)
        logger.info(f"Model Architecture Check: {model.__class__.__name__}")
        logger.info("=" * 60)

        for name, input_size in scenarios:
            try:
                logger.info(f"\n>>> Generating Summary for Dataset: {name} | Input Size: {input_size}")

                # 生成报告
                model_stats = summary(
                    model,
                    input_size=input_size,
                    col_names=["input_size", "output_size", "num_params", "mult_adds"],
                    verbose=0,
                    device=device
                )
                logger.info("\n" + str(model_stats))

            except RuntimeError as e:
                # 捕获尺寸不匹配导致的错误 (比如 Linear 层固定了输入维度)
                logger.error(f"Failed to summarize for {name}. It might be due to resolution mismatch.")
                logger.error(f"Error: {e}")

        logger.info("=" * 60)

        # 恢复原始模式
        model.train(original_mode)

    except ImportError:
        logger.warning("Package 'torchinfo' not found. Please `pip install torchinfo`.")
    except Exception:
        logger.error("Unexpected error in print_model_details:")
        logger.error(traceback.format_exc())

def train_local(model, device, train_loader, optimizer, criterion, epoch, target_float):
    # print(f"Type of train_loader: {type(train_loader)}")  # 调试信息
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)

        if target_float == True:
            target = target.float()

        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 100 == 0:
            print(f'Train Epoch: {epoch} [{batch_idx * len(data)}/{len(train_loader.dataset)} ({100. * batch_idx / len(train_loader):.0f}%)]\tLoss: {loss.item():.6f}')

def train_local_text(model, device, train_loader, optimizer, criterion, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):  # 直接解包数据和目标
        logger.info(f"Batch {batch_idx + 1}: Data type before model: {data.dtype}")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data).squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 1000 == 0:
            print(f'Train Epoch: {epoch} [Batch {batch_idx + 1}]\tLoss: {loss.item():.6f}')

def aggregate_models_old(global_model, local_models, device, noise_level=1.0):
    global_dict = global_model.state_dict()
    for k in global_dict.keys():
        local_grads = []
        for i in range(len(local_models)):
            local_grad = local_models[i].state_dict()[k].float().to(device)
            # local_grad_noisy = add_laplace_noise_device(local_grad, noise_scale=noise_level)
            local_grad_noisy = local_grad
            local_grads.append(local_grad_noisy)
        global_dict[k] = torch.stack(local_grads, 0).mean(0)
    global_model.load_state_dict(global_dict)
    return global_model


def print_weight_differences(local_models):
    """
    打印第一个局部模型的前三个权重，并根据命名规则提取其他模型的权重进行比较。

    参数：
    - local_models: 包含多个局部模型的列表
    """
    # 获取第一个局部模型的 state_dict
    first_model_state_dict = local_models[0].state_dict()

    # 获取第一个模型的前3个权重层名称
    layer_names = list(first_model_state_dict.keys())[:3]  # 获取前三个层的名字

    print(f"Comparing first three layers' weights across the local models:")

    for i, layer_name in enumerate(layer_names):
        print(f"Layer {i + 1} - {layer_name}:")

        # 提取并打印第一个模型对应层的权重
        first_weight = first_model_state_dict[layer_name]
        print(f"Local Model 1 - {layer_name} weight shape: {first_weight.shape}")
        print(
            f"Local Model 1 - {layer_name} weight mean: {first_weight.mean().item()}, std: {first_weight.std().item()}")
        print(f"Local Model 1 - {layer_name} weight values (first few):")
        print(first_weight.view(-1)[:10].cpu().numpy())  # 展平并显示前10个权重值
        print("-" * 50)

        # 比较其他模型中相同层的权重差异
        for j in range(1, len(local_models)):
            other_weight = local_models[j].state_dict()[layer_name]

            # 打印其他模型对应层的权重
            print(f"Local Model {j + 1} - {layer_name} weight shape: {other_weight.shape}")
            print(
                f"Local Model {j + 1} - {layer_name} weight mean: {other_weight.mean().item()}, std: {other_weight.std().item()}")
            print(f"Local Model {j + 1} - {layer_name} weight values (first few):")
            print(other_weight.view(-1)[:10].cpu().numpy())  # 展平并显示前10个权重值
            print("-" * 50)

            # 比较当前模型与第一个模型的权重差异
            diff = torch.abs(first_weight - other_weight)
            print(f"Difference between Local Model 1 and Local Model {j + 1} for {layer_name}:")
            print(f"Max difference: {diff.max().item()}, Mean difference: {diff.mean().item()}")
            print("=" * 50)

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
                std = epsilon * sensitivity
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
                        # noisy_param = add_noise_rrldp(local_dict[key].to(device), epsilon=config["epsilon"],
                        #                                  sensitivity=config["sensitivity"], noise_type='gaussian')
                        noisy_param = add_noise_rrldp(local_dict[key].to(device), epsilon=config["epsilon"],
                                                      sensitivity=sensitivity, noise_type='gaussian')
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
                            noisy_param = mvg_mechanism_2d(X=local_dict[key].unsqueeze(0).to(device), epsilon=config["epsilon"], delta=10 ** (-5),
                                             theta=[0.55 / 2, 0.55 / 2] + [0.45 / (local_dict[key].shape[0] - 2)] * (local_dict[key].shape[0] - 2))
                        else:
                            noisy_param = mvg_mechanism(X=local_dict[key].to(device), epsilon=config["epsilon"],
                                                           delta=10 ** (-5),
                                                           theta=[0.55 / 2, 0.55 / 2] + [0.45 / (local_dict[key].view(-1, local_dict[key].size(1)).shape[0] - 2)] * (local_dict[key].view(-1, local_dict[key].size(1)).shape[0] - 2))
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


def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    if config["dataset"] == "mnist":
        # train_dataset, test_dataset = get_mnist_loaders(batch_size=config['batch_size'])
        # test_loader = test_dataset
        train_loaders, test_loader = get_mnist_split_loaders(batch_size=config['batch_size'])
    elif config["dataset"] == "cifar10":
        # train_dataset, test_dataset = get_cifar10_loaders(batch_size=config['batch_size'])
        train_loaders, test_loader = get_cifar10_split_loaders(batch_size=config['batch_size'])
    elif config["dataset"] == "svhn":
        # train_dataset, test_dataset = get_svhn_loaders(batch_size=config['batch_size'])
        train_loaders, test_loader = get_svhn_split_loaders(batch_size=config['batch_size'])
    elif config["dataset"] == "celeba":
        train_dataset, val_dataset, test_dataset = get_celeba_loaders(batch_size=config['batch_size'])
        test_loader = test_dataset
    elif config["dataset"] == "imdb":
        train_dataset, test_dataset, vocab = get_imdb_loaders(batch_size=config['batch_size'])
        test_loader = test_dataset
    else:
        logger.error("Unknown dataset: ", config["dataset"])
        return

    # # no noise ver.
    # if config["noise_type"] == "none":
    #     train_loader = train_dataset
    # elif config["noise_type"] == "privacy":
    #     # train_loader = get_data_loaders_with_noise(train_dataset, device, noise_level=0.1)
    #     train_loader = get_data_loaders_with_noise_nostack(train_dataset, device, config=config)

    # local_train_loaders = get_local_data_loaders(train_dataset)
    # local_loader1, local_loader2, local_loader3 = get_local_data_loaders(train_dataset)
    # local_loader1, local_loader2, local_loader3 = manual_split_data_loaders(train_dataset)

    logger.info(f'Initializing model: {config["model"]}')
    if config["model"] == "mobilenet":
        # model = get_mobilenet_model(num_classes=config['num_classes']).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        global_model = get_mobilenet_model(num_classes=config['num_classes']).to(device)
        local_models = [get_mobilenet_model(num_classes=config['num_classes']).to(device) for _ in range(3)]
        optimizers = [optim.Adam(local_model.parameters(), lr=0.001) for local_model in local_models]
        criterion = nn.CrossEntropyLoss()
    elif config["model"] == "vgg16":
        # model = get_vgg16_model(num_classes=config['num_classes']).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        global_model = get_vgg16_model(num_classes=config['num_classes']).to(device)
        local_models = [get_vgg16_model(num_classes=config['num_classes']).to(device) for _ in range(3)]
        optimizers = [optim.Adam(local_model.parameters(), lr=0.001) for local_model in local_models]
        criterion = nn.CrossEntropyLoss()
    elif config["model"] == "bilstm":
        # 假设您已经将词汇表大小 (vocab_size) 和其他参数存储在配置文件中
        # model = get_bilstm_model(
        #     vocab_size=len(vocab),
        #     embedding_dim=config['embedding_dim'],
        #     hidden_dim=config['hidden_dim'],
        #     output_dim=config['output_dim'],
        #     n_layers=config['n_layers'],
        #     bidirectional=config['bidirectional'],
        #     dropout=config['dropout']
        # ).to(device)
        # optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])

        global_model = get_bilstm_model(
            vocab_size=len(vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            n_layers=config['n_layers'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout']
        ).to(device)
        local_models = [get_bilstm_model(
            vocab_size=len(vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            n_layers=config['n_layers'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout']
        ).to(device) for _ in range(3)]
        optimizers = [optim.Adam(local_model.parameters(), lr=0.001) for local_model in local_models]
        criterion = nn.CrossEntropyLoss()
    else:
        logger.error("Unkown Model: ", config["model"])
        return

    print_model_details(global_model, config, device)
    # Create a directory for saving experiment results
    experiment_dir = os.path.join('experiments/scenario1', time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(experiment_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    experiment_info = {
        'metrics': [],
    }

    adaptive_clipping_states = [{} for _ in range(len(local_models))]
    for epoch in range(1, config['num_epochs'] + 1):
        logger.info(f'Starting epoch {epoch}')
        if config["dataset_type"] == "image":
            # 使用分割后的加载器进行训练
            for i, train_loader in enumerate(train_loaders):
                train_local(local_models[i], device, train_loader, optimizers[i], criterion, epoch, target_float=config["target_float"])
        elif config["dataset_type"] == "text":
            # train_local_text(local_models[i], device, local_train_loaders[i], optimizers[i], criterion, epoch)
            pass
        else:
            logger.error("Unkown dataset type: ", config["dataset_type"])

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

        logger.info(f'Epoch {epoch} aggregation and update completed')
        # test_loss, metrics, predictions, targets = test(global_model, device, test_loader, criterion, target_float=config["target_float"])

    # 假设 local_models 是一个包含多个局部模型的列表
    # 调用该函数查看模型权重差异
    print_weight_differences(local_models)

    # 聚合模型
    # global_model = aggregate_models(global_model, local_models, device, noise_level=0.1)
    # global_model = aggregate_models(global_model, local_models[:2], device, noise_level=0.1)

    model_path = os.path.join(experiment_dir, 'model.pth')
    torch.save(global_model.state_dict(), model_path)
    logger.info(f'Model saved to {model_path}')

    if config["multi_targets"] == False:
        if config["dataset_type"] == "image":
            test_loss, metrics, predictions, targets = test(global_model, device, test_loader, criterion,
                                                            target_float=config["target_float"])
        elif config["dataset_type"] == "text":
            test_loss, metrics, predictions, targets = test_text(global_model, device, test_loader, criterion)
    else:
        test_loss, metrics, predictions, targets = test_multi(global_model, device, test_loader, criterion,
                                                              target_float=config["target_float"])
    logger.info('Testing completed')

    experiment_info['metrics'].append(metrics)
    save_experiment_info(experiment_dir, experiment_info)
    save_predictions(experiment_dir, predictions, targets)

    # # 测试每个本地模型
    # for i, local_model in enumerate(local_models):
    #     print(f"Testing Local Model {i + 1}:")
    #     test(local_model, device, test_loader, criterion)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model based on YAML configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    main(args.config)