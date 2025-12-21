from torchvision import models
import torch.nn as nn

class ResNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(ResNetFeatureExtractor, self).__init__()
        resnet = models.resnet18(pretrained=True)
        self.features = nn.Sequential(*list(resnet.children())[:-1])
        self.fc = nn.Linear(resnet.fc.in_features, 1280)

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# class MobileNetFeatureExtractor(nn.Module):
#     def __init__(self):
#         super(MobileNetFeatureExtractor, self).__init__()
#         mobilenet = models.mobilenet_v2(pretrained=True)
#         self.features = mobilenet.features
#         self.pool = nn.AdaptiveAvgPool2d(1)
#         self.fc = nn.Linear(mobilenet.last_channel, 1280)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.pool(x)
#         x = x.view(x.size(0), -1)
#         x = self.fc(x)
#         return x

# class MobileNetFeatureExtractor(nn.Module):
#     def __init__(self, out_channels=3):
#         super(MobileNetFeatureExtractor, self).__init__()
#         self.feature_extractor = models.mobilenet_v2(pretrained=True).features
#         self.conv = nn.Conv2d(1280, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         x = self.feature_extractor(x)
#         x = self.conv(x)
#         return x

# class MobileNetFeatureExtractor(nn.Module):
#     def __init__(self, out_channels=3):
#         super(MobileNetFeatureExtractor, self).__init__()
#         self.mobilenet = models.mobilenet_v2(pretrained=True)
#         self.features = self.mobilenet.features
#         self.conv1x1 = nn.Conv2d(1280, out_channels, kernel_size=1)
#
#     def forward(self, x):
#         x = self.features(x)
#         x = self.conv1x1(x)
#         return x

class MobileNetFeatureExtractorSingle(nn.Module):
    def __init__(self, out_channels=3):
        super(MobileNetFeatureExtractor, self).__init__()
        # 载入 MobileNetV2 模型
        mobilenet = models.mobilenet_v2(pretrained=True)
        # 保留到某个中间层，而不是整个特征提取器
        # self.features = nn.Sequential(*list(mobilenet.features.children())[:-6])  # 保留更多层，避免将空间分辨率降为 (1,1)
        # self.conv1x1 = nn.Conv2d(96, out_channels, kernel_size=1)  # 调整通道数，以适应减去的层后的输出通道数

        self.features = nn.Sequential(*list(mobilenet.features.children())[:-8])  # 保留更多层，避免将空间分辨率降为 (1,1)
        self.conv1x1 = nn.Conv2d(64, out_channels, kernel_size=1)  # 调整通道数，以适应减去的层后的输出通道数

    def forward(self, x):
        x = self.features(x)
        x = self.conv1x1(x)
        return x


# 特征提取器
class MobileNetFeatureExtractor(nn.Module):
    def __init__(self):
        super(MobileNetFeatureExtractor, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)

        self.features = mobilenet.features
        self.replace_relu6_with_tanh(self.features)

        ###############
        self.pool = nn.AdaptiveAvgPool2d((1, 1))  # 添加自适应池化层
        self.tanh = nn.Tanh()  # 添加 Tanh 激活函数 1280是MobileNetV2的最后特征图通道数

    def replace_relu6_with_tanh(self, module):
        """
        替换模型中的所有 ReLU6 激活函数为 Tanh。
        """
        for name, child in module.named_children():
            if isinstance(child, nn.ReLU6):  # 如果是 ReLU6 层，替换为 Tanh
                setattr(module, name, nn.Tanh())
            else:  # 如果是子模块，递归替换
                self.replace_relu6_with_tanh(child)

    def forward(self, x):
        x = self.features(x)
        ##########################
        x = self.pool(x)  # 确保输出形状为 (batch_size, 1280, 1, 1)
        # x = self.tanh(x)  # 使用 Tanh 将输出限制到 [-1, 1] 范围
        return x

# 分类模型
class MobileNetClassifier(nn.Module):
    def __init__(self, num_classes=10):  # 假设有10个类别
        super(MobileNetClassifier, self).__init__()
        mobilenet = models.mobilenet_v2(pretrained=True)
        self.classifier = mobilenet.classifier  # 后半部分
        self.classifier[1] = nn.Linear(self.classifier[1].in_features, num_classes)  # 修改分类层以匹配类别数

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平操作
        x = self.classifier(x)
        return x

class MLPFeatureExtractor(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_classes, dropout_rate=0.5, use_batch_norm=False):
        super(MLPFeatureExtractor, self).__init__()
        layers = []
        in_features = input_size
        for _ in range(num_layers):
            layers.append(nn.Linear(in_features, hidden_size))
            if use_batch_norm:
                layers.append(nn.BatchNorm1d(hidden_size))  # 如果use_batch_norm为True，添加批归一化层
            layers.append(nn.ReLU())
            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))  # 如果dropout_rate > 0，添加Dropout层
            in_features = hidden_size

        # 添加一个输出层以获取特定大小的特征
        layers.append(nn.Linear(hidden_size, output_size))
        if use_batch_norm:
            layers.append(nn.BatchNorm1d(output_size))  # 如果use_batch_norm为True，添加批归一化层
        layers.append(nn.ReLU())  # 添加ReLU激活函数
        if dropout_rate > 0:
            layers.append(nn.Dropout(dropout_rate))  # 如果dropout_rate > 0，添加Dropout层
        self.fc = nn.Sequential(*layers)

        # 最终分类层
        self.out = nn.Linear(output_size, num_classes)

        # # 初始化每个层的参数
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         # 使用 Kaiming 正态初始化线性层权重
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #         nn.init.constant_(m.bias, 0.0)
        #     elif isinstance(m, nn.BatchNorm1d):
        #         # 将批归一化层的权重初始化为 1，偏置初始化为 0
        #         nn.init.constant_(m.weight, 1.0)
        #         nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        """
                提取特征的方法
                """
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        return features

# MLP的特征提取器合分类模型
class MLPClassifier(nn.Module):
    def __init__(self, output_size, num_classes):
        super(MLPClassifier, self).__init__()
        self.out = nn.Linear(output_size, num_classes)

    def forward(self, x):
        x = self.out(x)
        return x