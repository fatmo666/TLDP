import torch.nn as nn
from torchvision import models

def get_mobilenet_model(num_classes=10):
    # model = models.mobilenet_v2(pretrained=True)
    model = models.mobilenet_v2(pretrained=True)
    model.classifier[1] = nn.Linear(model.last_channel, num_classes)
    return model


def get_vgg16_model(num_classes=10):
    # 加载预定义的 VGG16 模型，不使用预训练权重
    model = models.vgg16(pretrained=False)

    # 替换最后的全连接层以适应新的类别数量
    model.classifier[6] = nn.Linear(4096, num_classes)

    return model