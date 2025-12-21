import argparse
import json
import os
import time

import torch
import torch.optim as optim
import torch.nn as nn
import yaml
import logging

from models.mobilenet import *
from models.BiLSTM import *
from models.feature_model import ResNetFeatureExtractor, MobileNetFeatureExtractor, MobileNetClassifier
from utils.data_utils import *
from utils.privacy_utils_new import *
from test import *
from utils.feature_utils import extract_features

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_experiment_info(experiment_dir, info):
    with open(os.path.join(experiment_dir, 'experiment_info.json'), 'w') as f:
        json.dump(info, f, indent=4)

def save_predictions(experiment_dir, predictions, targets):
    torch.save({'predictions': predictions, 'targets': targets}, os.path.join(experiment_dir, 'predictions.pt'))

def reshape_features(features):
    # Assuming a square input feature map for simplicity
    batch_size, feature_size = features.size()
    side_length = int(feature_size ** 0.5)
    reshaped_features = features.view(batch_size, 1, side_length, side_length)
    return reshaped_features

def train(model, device, train_loader, optimizer, criterion, epoch, target_float):
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

def train_text(model, device, train_loader, optimizer, criterion, epoch):
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


def main(config_file):
    with open(config_file, 'r') as file:
        config = yaml.safe_load(file)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # device = "cpu"

    if config["dataset"] == "mnist":
        train_dataset, test_dataset = get_mnist_loaders(batch_size=config['batch_size'])
        test_loader = test_dataset
    elif config["dataset"] == "cifar10":
        train_dataset, test_dataset = get_cifar10_loaders(batch_size=config['batch_size'])
        test_loader = test_dataset
    elif config["dataset"] == "svhn":
        train_dataset, test_dataset = get_svhn_loaders(batch_size=config['batch_size'])
        test_loader = test_dataset
    elif config["dataset"] == "celeba":
        train_dataset, val_dataset, test_dataset = get_celeba_loaders(batch_size=config['batch_size'])
        test_loader = test_dataset
    elif config["dataset"] == "imdb":
        train_dataset, test_dataset, vocab = get_imdb_loaders(batch_size=config['batch_size'])
        test_loader = test_dataset
    else:
        logger.error("Unknown dataset: ", config["dataset"])
        return

    feature_extractor = MobileNetFeatureExtractor().to(device)
    # feature_extractor = ResNetFeatureExtractor().to(device)

    print("feature_extractor: ", feature_extractor)

    # no noise ver.
    train_loader = train_dataset

    all_features = []
    all_targets = []
    features, targets = extract_features(feature_extractor, device, train_loader)
    # noisy_features = add_laplace_noise(features, noise_scale=0.1)

    # noisy_features = features
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
        elif config["privacy_function"] == "mvg_mechanism":
            if len(features.shape) == 1:
                noisy_features = mvg_mechanism_2d(X=features.unsqueeze(0).to(device), epsilon=config["epsilon"],
                                               delta=10 ** (-5),
                                               theta=[0.55 / 2, 0.55 / 2] + [0.45 / (features.shape[0] - 2)] * (
                                                           features.shape[0] - 2))
            else:
                noisy_features = mvg_mechanism(X=features.to(device), epsilon=config["epsilon"],
                                            delta=10 ** (-5),
                                            theta=[0.55 / 2, 0.55 / 2] + [0.45 / (features.view(-1, features.size(1)).shape[0] - 2)] * (features.view(-1, features.size(1)).shape[0] - 2))

    all_features.append(noisy_features)
    all_targets.append(targets)

    # for local_loader in train_loader:
    #     features, targets = extract_features(feature_extractor, device, local_loader)
    #     noisy_features = add_laplace_noise(features, noise_scale=0.1)
    #     all_features.append(noisy_features)
    #     all_targets.append(targets)

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

    logger.info(f'Initializing model: {config["model"]}')
    if config["model"] == "mobilenet":
        # model = get_mobilenet_model(num_classes=config['num_classes']).to(device)
        # print(model)
        ############
        model = MobileNetClassifier(num_classes=config['num_classes']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config["model"] == "vgg16":
        model = get_vgg16_model(num_classes=config['num_classes']).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    elif config["model"] == "bilstm":
        # 假设您已经将词汇表大小 (vocab_size) 和其他参数存储在配置文件中
        model = get_bilstm_model(
            vocab_size=len(vocab),
            embedding_dim=config['embedding_dim'],
            hidden_dim=config['hidden_dim'],
            output_dim=config['output_dim'],
            n_layers=config['n_layers'],
            bidirectional=config['bidirectional'],
            dropout=config['dropout']
        ).to(device)
        optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    else:
        logger.error("Unkown Model: ", config["model"])

    if config['loss_function'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss_function'] == 'bce_with_logits':
        criterion = nn.BCEWithLogitsLoss()

    print("MobileNetClassifier: ", model)

    # Create a directory for saving experiment results
    experiment_dir = os.path.join('experiments/scenario1', time.strftime('%Y%m%d-%H%M%S'))
    os.makedirs(experiment_dir, exist_ok=True)

    # Save configuration
    with open(os.path.join(experiment_dir, 'config.yaml'), 'w') as file:
        yaml.dump(config, file)

    experiment_info = {
        'metrics': [],
    }

    for epoch in range(1, config['num_epochs'] + 1):
        logger.info(f'Starting epoch {epoch}')
        if config["dataset_type"] == "image":
            train(model, device, train_loader, optimizer, criterion, epoch, target_float=config["target_float"])
        elif config["dataset_type"] == "text":
            train_text(model, device, train_loader, optimizer, criterion, epoch)
        else:
            logger.error("Unkown dataset type: ", config["dataset_type"])

    model_path = os.path.join(experiment_dir, 'model.pth')
    torch.save(model.state_dict(), model_path)
    logger.info(f'Model saved to {model_path}')

    if config["multi_targets"] == False:
        if config["dataset_type"] == "image":
            test_loss, metrics, predictions, targets = test(model, device, test_loader, criterion,
                                                            target_float=config["target_float"])
        elif config["dataset_type"] == "text":
            test_loss, metrics, predictions, targets = test_text(model, device, test_loader, criterion)
    else:
        test_loss, metrics, predictions, targets = test_multi(model, device, test_loader, criterion,
                                                              target_float=config["target_float"])
    logger.info('Testing completed')

    experiment_info['metrics'].append(metrics)
    save_experiment_info(experiment_dir, experiment_info)
    save_predictions(experiment_dir, predictions, targets)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model based on YAML configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    main(args.config)