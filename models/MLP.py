import torch
import torch.nn as nn

class FCNClassifier(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, num_classes, dropout_rate=0.5, use_batch_norm=False):
        super(FCNClassifier, self).__init__()
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
        # Flatten the input
        x = x.view(x.size(0), -1)

        # Forward propagate FCN
        features = self.fc(x)

        # Classify
        out = self.out(features)
        return out

    def extract_features(self, x):
        """
        提取特征的方法
        """
        x = x.view(x.size(0), -1)
        features = self.fc(x)
        return features