import torch
import torchvision
from torchvision import datasets, transforms
from torchtext.datasets import IMDB
from torchtext.data.utils import get_tokenizer
from torchtext.vocab import build_vocab_from_iterator
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
import numpy as np
from torch.utils.data import Subset
import torch.nn as nn

import torch.multiprocessing as mp
import torchtext

from utils.privacy_utils_new import mvg_mechanism

torchtext.disable_torchtext_deprecation_warning()

# 更改共享策略
mp.set_sharing_strategy('file_system')

# def get_adult_loaders(batch_size):
#     from ucimlrepo import fetch_ucirepo
#
#     adult = fetch_ucirepo(id=2)
#     X = adult.data.features
#     y = adult.data.targets


# class CTGDataset(Dataset):
#     def __init__(self, csv_file, tranform=None):
#

def yield_tokens(data_iter):
    tokenizer = get_tokenizer('spacy', 'en_core_web_sm')
    for _, line in data_iter:
        yield tokenizer(line)

def get_imdb_loaders(batch_size, embedding_dim=100):
    # 加载数据集
    train_iter, test_iter = IMDB(root='/home/huangyu/lab/MobileNet/IMDB', split=('train', 'test'))

    # 将 train_iter 和 test_iter 转换为列表
    train_data = list(train_iter)
    test_data = list(test_iter)

    def check_label_distribution(data_iter):
        label_counts = {}
        for label, _ in data_iter:
            # 打印标签内容
            # print(f"Label: {label}")
            if label not in label_counts:
                label_counts[label] = 1
            else:
                label_counts[label] += 1
        total_samples = sum(label_counts.values())
        for label, count in label_counts.items():
            print(f"Label {label}: {count / total_samples:.2f} ({count} samples)")

    # 检查训练集标签分布
    print("Train dataset label distribution:")
    check_label_distribution(train_iter)

    # 检查测试集标签分布
    print("Test dataset label distribution:")
    check_label_distribution(test_iter)

    # 构建词汇表
    vocab = build_vocab_from_iterator(yield_tokens(train_iter), specials=["<unk>"])
    vocab.set_default_index(vocab["<unk>"])

    tokenizer = get_tokenizer('spacy', 'en_core_web_sm')
    text_pipeline = lambda x: vocab(tokenizer(x))  # 确保将文本转换为词汇索引

    label_pipeline = lambda x: 1 if x == 1 else 0

    # 初始化嵌入层（根据词汇表大小和嵌入维度）
    embedding = nn.Embedding(num_embeddings=len(vocab), embedding_dim=embedding_dim, padding_idx=vocab['<pad>'])

    def collate_batch(batch):
        label_list, text_list = [], []
        for _label, _text in batch:
            label_list.append(label_pipeline(_label))
            processed_text = torch.tensor(text_pipeline(_text), dtype=torch.long)
            text_list.append(processed_text)

        # 填充序列并生成嵌入向量
        padded_text = pad_sequence(text_list, padding_value=vocab['<pad>'], batch_first=True)
        padded_text = padded_text.to(torch.long)  # 确保数据类型是 Long
        embedded_text = embedding(padded_text)  # 将索引转换为嵌入向量
        labels = torch.tensor(label_list, dtype=torch.float)

        return embedded_text, labels

    # 创建 DataLoader
    train_loader = DataLoader(train_iter, batch_size=batch_size, shuffle=True, collate_fn=collate_batch)
    test_loader = DataLoader(test_iter, batch_size=batch_size, shuffle=False, collate_fn=collate_batch)

    return train_loader, test_loader, vocab

def get_mnist_loaders(batch_size):
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./DataSet/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./DataSet/MNIST', train=False, download=True, transform=transform)

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # 获取一个批次的训练数据并打印出来
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)  # 使用 next() 函数来获取数据
    #
    # print("Images batch shape:", images.shape)
    # print("Labels batch shape:", labels.shape)
    # print("Min pixel value:", torch.min(images).item())
    # print("Max pixel value:", torch.max(images).item())

    return train_loader, test_loader

def get_mnist_loaders_resize(batch_size):
    transform = transforms.Compose([
        transforms.Resize((28, 28)),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    train_dataset = datasets.MNIST(root='./DataSet/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./DataSet/MNIST', train=False, download=True, transform=transform)

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 获取一个批次的训练数据并打印出来
    data_iter = iter(train_loader)
    images, labels = next(data_iter)  # 使用 next() 函数来获取数据

    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)
    print("Min pixel value:", torch.min(images).item())
    print("Max pixel value:", torch.max(images).item())

    return train_loader, test_loader

def get_mnist_split_loaders(batch_size=64, num_splits=3):
    # 定义数据转换
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))
    ])

    # 加载完整的 MNIST 数据集
    train_dataset = datasets.MNIST(root='./DataSet/MNIST', train=True, download=True, transform=transform)
    test_dataset = datasets.MNIST(root='./DataSet/MNIST', train=False, download=True, transform=transform)

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    # 分割训练数据集
    dataset_length = len(train_dataset)
    indices = np.arange(dataset_length)
    np.random.shuffle(indices)

    split_size = dataset_length // num_splits
    split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

    if dataset_length % num_splits != 0:
        split_indices[-1] = np.concatenate([split_indices[-1], indices[num_splits * split_size:]])

    # 为每个分割创建 DataLoader
    train_loaders = [DataLoader(Subset(train_dataset, split_indices[i]), batch_size=batch_size, shuffle=True)
                     for i in range(num_splits)]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader

def get_cifar10_loaders(batch_size):
    # 原始尺寸
    # 3*32*32
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./DataSet/CIFAR10', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./DataSet/CIFAR10', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # # 获取一个批次的训练数据并打印出来
    # data_iter = iter(train_loader)
    # images, labels = next(data_iter)  # 使用 next() 函数来获取数据
    #
    # print("Images batch shape:", images.shape)
    # print("Labels batch shape:", labels.shape)
    # print("Min pixel value:", torch.min(images).item())
    # print("Max pixel value:", torch.max(images).item())

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    return train_loader, test_loader

def get_cifar10_loaders_resize(batch_size):
    # 原始尺寸
    # 3*32*32
    transform = transforms.Compose([
        transforms.Resize((64, 64)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        # transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

    train_dataset = torchvision.datasets.CIFAR10(root='./DataSet/CIFAR10', train=True, download=True, transform=transform)
    test_dataset = torchvision.datasets.CIFAR10(root='./DataSet/CIFAR10', train=False, download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # 获取一个批次的训练数据并打印出来
    data_iter = iter(train_loader)
    images, labels = next(data_iter)  # 使用 next() 函数来获取数据

    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)
    print("Min pixel value:", torch.min(images).item())
    print("Max pixel value:", torch.max(images).item())

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    return train_loader, test_loader

def get_cifar10_split_loaders(batch_size=64, num_splits=3):
    from torchvision import transforms, datasets
    from torch.utils.data import DataLoader, Subset
    import numpy as np

    # 定义数据转换
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    # 加载完整的 CIFAR-10 数据集
    train_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=True, download=True, transform=transform)
    test_dataset = datasets.CIFAR10(root='./DataSet/CIFAR10', train=False, download=True, transform=transform)

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    # 分割训练数据集
    dataset_length = len(train_dataset)
    indices = np.arange(dataset_length)
    np.random.shuffle(indices)

    split_size = dataset_length // num_splits
    split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

    if dataset_length % num_splits != 0:
        # 如果无法平均分割，将多余样本分配到最后一个分割
        split_indices[-1] = np.concatenate([split_indices[-1], indices[num_splits * split_size:]])

    # 为每个分割创建 DataLoader
    train_loaders = [DataLoader(Subset(train_dataset, split_indices[i]), batch_size=batch_size, shuffle=True)
                     for i in range(num_splits)]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader

def get_imagenet_loaders(data_dir):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),  # 转换为张量
        # 范围不是-1到1
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # 归一化到 [-1, 1] 范围
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.ImageFolder(root=f'{data_dir}/train', transform=transform)
    val_dataset = datasets.ImageFolder(root=f'{data_dir}/val', transform=transform)

    return train_dataset, val_dataset

def get_svhn_loaders(batch_size=64, data_dir='./DataSet/SVHN'):
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    return train_loader, test_loader

def get_svhn_loaders_resize(batch_size=64, data_dir='./DataSet/SVHN'):
    transform = transforms.Compose([
        # transforms.Resize((128, 128)),
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=1)

    # 获取一个批次的训练数据并打印出来
    data_iter = iter(train_loader)
    images, labels = next(data_iter)  # 使用 next() 函数来获取数据

    print("Images batch shape:", images.shape)
    print("Labels batch shape:", labels.shape)
    print("Min pixel value:", torch.min(images).item())
    print("Max pixel value:", torch.max(images).item())

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    return train_loader, test_loader

def get_svhn_split_loaders(batch_size=64, data_dir='./DataSet/SVHN', num_splits=3):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),  # 调整图像大小
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.4377, 0.4438, 0.4728], std=[0.1980, 0.2010, 0.1970])
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    ])

    train_dataset = datasets.SVHN(root=data_dir, split='train', download=True, transform=transform)
    test_dataset = datasets.SVHN(root=data_dir, split='test', download=True, transform=transform)

    print("Number of training samples:", len(train_dataset))
    print("Number of testing samples:", len(test_dataset))

    # 分割训练数据集
    dataset_length = len(train_dataset)
    indices = np.arange(dataset_length)
    np.random.shuffle(indices)

    split_size = dataset_length // num_splits
    split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_splits)]

    if dataset_length % num_splits != 0:
        # 如果无法平均分割，将多余样本分配到最后一个分割
        split_indices[-1] = np.concatenate([split_indices[-1], indices[num_splits * split_size:]])

    # 为每个分割创建 DataLoader
    train_loaders = [DataLoader(Subset(train_dataset, split_indices[i]), batch_size=batch_size, shuffle=True)
                     for i in range(num_splits)]
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loaders, test_loader

def get_celeba_loaders(batch_size=64, data_dir='./DataSet/CelebA/'):
    transform = transforms.Compose([
        transforms.CenterCrop(178),  # 裁剪中心区域，178x178
        transforms.Resize((224, 224)),  # 调整图像大小为 224x224
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1] 范围
    ])

    train_dataset = datasets.CelebA(root=data_dir, split='train', download=False, transform=transform)
    val_dataset = datasets.CelebA(root=data_dir, split='valid', download=False, transform=transform)
    test_dataset = datasets.CelebA(root=data_dir, split='test', download=False, transform=transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    return train_loader, val_loader, test_loader

def get_voc2012_loaders(batch_size):
    # 定义数据变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # 转换为张量
        # transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化到 [-1, 1] 范围
    ])

    # 加载训练集
    train_dataset = datasets.VOCDetection(
        root='./DataSet/VOC2012',
        year='2012',
        image_set='train',
        download=True,
        transform=transform
    )

    # 加载验证集
    val_dataset = datasets.VOCDetection(
        root='./DataSet/VOC2012',
        year='2012',
        image_set='val',
        download=True,
        transform=transform
    )

    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader

def get_data_loaders_with_noise_nostack(train_dataset, device, config):
    from .privacy_utils_new import add_laplace_noise, add_gaussian_noise, add_noise_rrldp, add_weighted_noise_rrldp, perturb_tensor_krr, perturb_tensor_krr_new, add_optimized_tensor_noise_Yang

    # 创建一个新的带噪声的训练数据集
    noisy_train_data = []
    train_targets = []

    # 假设文本词汇表大小为 121066，用于限制噪声后索引范围
    vocab_size = 121066

    print("privacy function: ", config["privacy_function"])

    for data, target in train_dataset:
        for sample, target in zip(data, target):
            if config["privacy_function"] == "rrldp_laplace":
                noisy_sample = add_noise_rrldp(sample.to(device), epsilon=config["epsilon"], sensitivity=config["sensitivity"], noise_type='laplace')
            elif config["privacy_function"] == "rrldp_gaussian":
                noisy_sample = add_noise_rrldp(sample.to(device), epsilon=config["epsilon"], sensitivity=config["sensitivity"], noise_type='gaussian')
            elif config["privacy_function"] == "laplace":
                import math
                # sensitivity = 2 * math.sqrt(torch.numel(sample))
                scale = (torch.numel(sample) * config["sensitivity"]) / config["epsilon"]
                noisy_sample = add_laplace_noise(sample.to(device), scale=scale)
            elif config["privacy_function"] == "gaussian":
                import math
                std = (torch.numel(sample) * config["sensitivity"] ** 2) / (2 * config["epsilon"])
                # sensitivity = 2 * math.sqrt(torch.numel(sample))
                # std = (sensitivity * torch.log(1 / torch.tensor(1e-5))) / config["epsilon"]
                noisy_sample = add_gaussian_noise(sample.to(device), mean=0.0, std=std)
            elif config["privacy_function"] == "perturb_tensor_krr":
                # noisy_sample = perturb_tensor_krr(sample.to(device), epsilon=config["epsilon"])
                noisy_sample = perturb_tensor_krr_new(sample.to(device), epsilon=config["epsilon"])
            elif config["privacy_function"] == "add_optimized_tensor_noise_Yang":
                import math
                sensitivity = 2 * math.sqrt(torch.numel(sample))
                noisy_sample = add_optimized_tensor_noise_Yang(sample.to(device), epsilon=config["epsilon"], sigma=10**(-5), sensitivity=sensitivity)
            elif config["privacy_function"] == "mvg_mechanism":
                noisy_sample = mvg_mechanism(X=sample.to(device), epsilon=config["epsilon"], delta=10**(-5), theta=[0.55/2, 0.55/2] + [0.45 / (sample.shape[0] * sample.shape[1] - 2)] * (sample.shape[0] * sample.shape[1] - 2))
            elif config["privacy_function"] == "add_weighted_noise_rrldp_laplace":
                # # mnist
                # first_data = sample.cpu().numpy()[0]
                # weighted_matrix = (first_data + 1) / 2

                import cv2
                import numpy as np

                # 1. 获取数据并反归一化 (通用步骤)
                # tensor shape: [C, H, W] -> numpy
                img_float = sample.cpu().numpy()

                # 假设均值0.5，方差0.5的反归一化
                # 结果形状依然是 [C, H, W]
                img_uint8_all = ((img_float * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

                # debug_img = img_uint8_all.transpose(1, 2, 0)
                # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f"check_color.png", debug_img)

                # 2. 准备灰度图供 Otsu 使用 (分情况处理)
                if config["dataset"] == "mnist":
                    # MNIST (1, 28, 28) 或 (3, 28, 28)
                    # 直接取第一个通道即可
                    img_gray = img_uint8_all[0]

                elif config["dataset"] in ["cifar10", "svhn"]:
                    # CIFAR/SVHN (3, 32, 32)
                    # 需要转置为 (32, 32, 3) 才能被 cv2 处理
                    img_hwc = img_uint8_all.transpose(1, 2, 0)
                    # 必须转换为灰度图才能做 Otsu
                    img_gray = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2GRAY)
                else:
                    # 默认处理
                    img_gray = img_uint8_all[0]

                # 3. Otsu 阈值处理 (针对灰度图)
                _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 4. 形态学处理
                kernel = np.ones((3, 3), np.uint8)
                processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                # 5. 生成权重矩阵 (0.0 ~ 1.0)
                # 形状为 (H, W)，例如 (28,28) 或 (32,32)
                weighted_matrix = (processed / 255.0).astype(np.float32)

                # ######## Debug
                # debug_mask_vis = (weighted_matrix * 255).astype(np.uint8)
                # # --- 彩色情况 (CIFAR/SVHN) ---
                # # 1. 转置: (3,H,W) -> (H,W,3)
                # temp_img = img_uint8_all.transpose(1, 2, 0)
                # # 2. 颜色转换: RGB -> BGR (OpenCV默认)
                # temp_img_bgr = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
                #
                # # 3. 核心步骤: 将 Mask 作用在原图上
                # # Mask 为白色(255)的地方保留原图，黑色(0)的地方变黑
                # masked_result = cv2.bitwise_and(temp_img_bgr, temp_img_bgr, mask=debug_mask_vis)
                #
                # # 保存三张图：原图、Mask、效果图
                # cv2.imwrite(f"check_origin.png", temp_img_bgr)
                # cv2.imwrite(f"check_mask.png", debug_mask_vis)
                # cv2.imwrite(f"check_applied.png", masked_result)  # <--- 看这张

                noisy_sample = add_weighted_noise_rrldp(tensor=sample.to(device), epsilon=config["epsilon"], sensitivity=config["sensitivity"], weight_matrix=weighted_matrix, noise_type='laplace')
            elif config["privacy_function"] == "add_weighted_noise_rrldp_gaussian":
                import cv2
                import numpy as np

                # 1. 获取数据并反归一化 (通用步骤)
                # tensor shape: [C, H, W] -> numpy
                img_float = sample.cpu().numpy()

                # 假设均值0.5，方差0.5的反归一化
                # 结果形状依然是 [C, H, W]
                img_uint8_all = ((img_float * 0.5 + 0.5) * 255).clip(0, 255).astype(np.uint8)

                # debug_img = img_uint8_all.transpose(1, 2, 0)
                # debug_img = cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR)
                # cv2.imwrite(f"check_color.png", debug_img)

                # 2. 准备灰度图供 Otsu 使用 (分情况处理)
                if config["dataset"] == "mnist":
                    # MNIST (1, 28, 28) 或 (3, 28, 28)
                    # 直接取第一个通道即可
                    img_gray = img_uint8_all[0]

                elif config["dataset"] in ["cifar10", "svhn"]:
                    # CIFAR/SVHN (3, 32, 32)
                    # 需要转置为 (32, 32, 3) 才能被 cv2 处理
                    img_hwc = img_uint8_all.transpose(1, 2, 0)
                    # 必须转换为灰度图才能做 Otsu
                    img_gray = cv2.cvtColor(img_hwc, cv2.COLOR_RGB2GRAY)
                else:
                    # 默认处理
                    img_gray = img_uint8_all[0]

                # 3. Otsu 阈值处理 (针对灰度图)
                _, binary = cv2.threshold(img_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                # 4. 形态学处理
                kernel = np.ones((3, 3), np.uint8)
                processed = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

                # 5. 生成权重矩阵 (0.0 ~ 1.0)
                # 形状为 (H, W)，例如 (28,28) 或 (32,32)
                weighted_matrix = (processed / 255.0).astype(np.float32)

                # ######## Debug
                # debug_mask_vis = (weighted_matrix * 255).astype(np.uint8)
                # # --- 彩色情况 (CIFAR/SVHN) ---
                # # 1. 转置: (3,H,W) -> (H,W,3)
                # temp_img = img_uint8_all.transpose(1, 2, 0)
                # # 2. 颜色转换: RGB -> BGR (OpenCV默认)
                # temp_img_bgr = cv2.cvtColor(temp_img, cv2.COLOR_RGB2BGR)
                #
                # # 3. 核心步骤: 将 Mask 作用在原图上
                # # Mask 为白色(255)的地方保留原图，黑色(0)的地方变黑
                # masked_result = cv2.bitwise_and(temp_img_bgr, temp_img_bgr, mask=debug_mask_vis)
                #
                # # 保存三张图：原图、Mask、效果图
                # cv2.imwrite(f"check_origin.png", temp_img_bgr)
                # cv2.imwrite(f"check_mask.png", debug_mask_vis)
                # cv2.imwrite(f"check_applied.png", masked_result)  # <--- 看这张


                noisy_sample = add_weighted_noise_rrldp(tensor=sample.to(device), epsilon=config["epsilon"], sensitivity=config["sensitivity"], weight_matrix=weighted_matrix, noise_type='gaussian')
            else:
                exit(1)
            # noisy_sample = add_noise_rrldp(sample.to(device), 1.0, 1.0, noise_type='laplace')
            if config["dataset_type"] == "text":
                noisy_sample = noisy_sample.to(torch.long)
                # noisy_sample = torch.clamp(noisy_sample, 0, vocab_size - 1)
                # noisy_sample = torch.clamp(noisy_sample, 0, 121066 - 1)

            noisy_train_data.append(noisy_sample.cpu())
            train_targets.append(target)

    if config["dataset_type"] == "text":
        # 确保所有数据张量长度一致
        noisy_train_data = pad_sequence(noisy_train_data, batch_first=True, padding_value=0)
    else:
        noisy_train_data = torch.stack(noisy_train_data)
    if config["multi_targets"] == False:
        train_targets = torch.tensor(train_targets)
    else:
        train_targets = torch.stack(train_targets)

    noisy_train_dataset = torch.utils.data.TensorDataset(noisy_train_data, train_targets)
    train_loader = torch.utils.data.DataLoader(noisy_train_dataset, batch_size=config["batch_size"], shuffle=True)
    return train_loader


def get_data_loaders_with_noise(train_dataset, device, noise_level=1.0, batch_size=64):
    from .privacy_utils_new import add_laplace_noise, add_gaussian_noise, add_noise_rrldp, add_weighted_noise_rrldp, perturb_tensor

    # 添加噪声到训练数据
    train_data_noisy = []
    train_targets = []
    for data, target in train_dataset:
        noisy_data = add_laplace_noise(data.to(device), scale=10.0)
        # noisy_data = add_gaussian_noise(data.to(device))
        # noisy_data = add_noise_rrldp(data.to(device), 1.0, 1.0, noise_type='laplace')
        # noisy_data = perturb_tensor(data.to(device), p=0.5)
        train_data_noisy.append(noisy_data.cpu())
        train_targets.append(target)

    train_data_noisy = torch.stack(train_data_noisy)
    train_targets = torch.tensor(train_targets)

    # 创建一个新的带噪声的训练数据集
    noisy_train_dataset = torch.utils.data.TensorDataset(train_data_noisy, train_targets)
    train_loader = torch.utils.data.DataLoader(noisy_train_dataset, batch_size=batch_size, shuffle=True)
    return train_loader


def get_local_data_loaders_old(train_dataset, batch_size=64):
    # 分割数据集，假设有三个用户
    dataset_split = torch.utils.data.random_split(train_dataset, [20000, 20000, 20000])

    local_loaders = []
    for local_dataset in dataset_split:
        local_loader = torch.utils.data.DataLoader(local_dataset, batch_size=batch_size, shuffle=True)
        local_loaders.append(local_loader)

    return local_loaders

def get_local_data_loaders(train_dataset, batch_size=64):
    # 计算数据集的总长度
    dataset_length = len(train_dataset)

    # 确定分割大小
    split_lengths = [dataset_length // 3] * 2 + [dataset_length - 2 * (dataset_length // 3)]

    # 分割数据集
    dataset_split = torch.utils.data.random_split(train_dataset, split_lengths)

    # 为每个分割创建数据加载器
    loader1 = torch.utils.data.DataLoader(dataset_split[0], batch_size=batch_size, shuffle=True)
    loader2 = torch.utils.data.DataLoader(dataset_split[1], batch_size=batch_size, shuffle=True)
    loader3 = torch.utils.data.DataLoader(dataset_split[2], batch_size=batch_size, shuffle=True)

    return loader1, loader2, loader3

def manual_split_data_loaders(train_dataset, batch_size=64, num_splits=3):
    import numpy as np
    from torch.utils.data import Subset

    dataset_length = len(train_dataset)
    indices = np.arange(dataset_length)
    np.random.shuffle(indices)  # 打乱索引顺序以实现随机分割

    split_size = dataset_length // num_splits
    split_indices = [indices[i * split_size:(i + 1) * split_size] for i in range(num_splits)]
    if dataset_length % num_splits != 0:
        # 将剩余的样本分配到最后一个分割中
        split_indices[-1] = np.concatenate([split_indices[-1], indices[num_splits * split_size:]])

    loaders = [DataLoader(Subset(train_dataset, split_indices[i]), batch_size=batch_size, shuffle=True)
               for i in range(num_splits)]

    return loaders
