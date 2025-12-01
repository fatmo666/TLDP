import os
import torch
import torch.optim as optim
import torch.nn as nn
from models.mobilenet import *
from models.BiLSTM import *
from utils.data_utils import *
from test import *
import yaml
import argparse
import logging
import json
import time
from captum.attr import DeepLift, IntegratedGradients

"""
场景一： 三个终端用户，差分隐私处理后上传数据集至服务器
"""

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def save_experiment_info(experiment_dir, info):
    with open(os.path.join(experiment_dir, 'experiment_info.json'), 'w') as f:
        json.dump(info, f, indent=4)

def save_predictions(experiment_dir, predictions, targets):
    torch.save({'predictions': predictions, 'targets': targets}, os.path.join(experiment_dir, 'predictions.pt'))


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
        # logger.info(f"Batch {batch_idx + 1}: Data type before model: {data.dtype}")
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        # 如果是二分类任务
        if output.size(1) == 1:
            output = torch.sigmoid(output).squeeze(1)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

        if batch_idx % 10 == 0:
            print(f'Train Epoch: {epoch} [Batch {batch_idx + 1}]\tLoss: {loss.item():.6f}')


def check_label_distribution(train_loader):
    label_counts = {0: 0, 1: 0}  # 假设是二分类任务
    for _, target in train_loader:
        for label in target:
            label_counts[label.item()] += 1

    total_samples = sum(label_counts.values())
    print(f"Label distribution - 0: {label_counts[0] / total_samples:.2f}, 1: {label_counts[1] / total_samples:.2f}")

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

    # check_label_distribution(train_dataset)
    # check_label_distribution(test_dataset)

    logger.info(f'Initializing model: {config["model"]}')
    if config["model"] == "mobilenet":
        model = get_mobilenet_model(num_classes=config['num_classes']).to(device)
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

    deeplift = DeepLift(model)

    # no noise ver.
    if config["noise_type"] == "none":
        train_loader = train_dataset
    elif config["noise_type"] == "privacy":
        # train_loader = get_data_loaders_with_noise(train_dataset, device, noise_level=0.1)
        train_loader = get_data_loaders_with_noise_nostack(train_dataset, device, config=config)

    if config['loss_function'] == 'cross_entropy':
        criterion = nn.CrossEntropyLoss()
    elif config['loss_function'] == 'bce_with_logits':
        criterion = nn.BCEWithLogitsLoss()

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
            test_loss, metrics, predictions, targets = test(model, device, test_loader, criterion, target_float=config["target_float"])
        elif config["dataset_type"] == "text":
            test_loss, metrics, predictions, targets = test_text(model, device, test_loader, criterion)
    else:
        test_loss, metrics, predictions, targets = test_multi(model, device, test_loader, criterion, target_float=config["target_float"])
    logger.info('Testing completed')

    experiment_info['metrics'].append(metrics)
    save_experiment_info(experiment_dir, experiment_info)
    save_predictions(experiment_dir, predictions, targets)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a model based on YAML configuration')
    parser.add_argument('--config', type=str, required=True, help='Path to the YAML configuration file')
    args = parser.parse_args()
    main(args.config)
