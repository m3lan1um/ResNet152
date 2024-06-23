import torch
from torch import nn
from torch import optim
from torch.utils.data import DataLoader

import torchvision
from torchvision import datasets, transforms

class Bottleneck(nn.Module):
    def __init__(self, in_channel, inner_channel, stride=1, W_s=None):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channel, inner_channel, kernel_size=1, stride=stride, padding=0)
        self.bn1 = nn.BatchNorm2d(inner_channel)
        self.conv2 = nn.Conv2d(inner_channel, inner_channel, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(inner_channel)
        self.conv3 = nn.Conv2d(inner_channel, inner_channel * 4, kernel_size=1, stride=1, padding=0)
        self.bn3 = nn.BatchNorm2d(inner_channel * 4)
        self.relu = nn.ReLU()

        self.W_s = W_s

    def forward(self, x):
        id = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.W_s is not None:
            id = self.W_s(id)

        out = out + id
        out = self.relu(out)

        return out
    
class ResNet(nn.Module):
    def __init__(self, ResBlock, num_layers, num_classes=10):
        super().__init__()

        self.in_channel = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3)   # different from original version
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block=ResBlock, inner_channel=64, num_blocks=num_layers[0], stride=1)
        self.layer2 = self._make_layer(block=ResBlock, inner_channel=128, num_blocks=num_layers[1], stride=2)
        self.layer3 = self._make_layer(block=ResBlock, inner_channel=256, num_blocks=num_layers[2], stride=2)
        self.layer4 = self._make_layer(block=ResBlock, inner_channel=512, num_blocks=num_layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(2048, num_classes)

    def _make_layer(self, block, inner_channel, num_blocks, stride):
        layer = []
        W_s = None

        if stride != 1 or self.in_channel != inner_channel * 4:
            W_s = nn.Sequential(
                nn.Conv2d(self.in_channel, inner_channel * 4, kernel_size=1, stride=stride),
                nn.BatchNorm2d(inner_channel * 4)
            )
        layer.append(block(self.in_channel, inner_channel, stride, W_s))

        self.in_channel = inner_channel * 4
        for _ in range(1, num_blocks):
            layer.append(block(self.in_channel, inner_channel))

        return nn.Sequential(*layer)

    def forward(self, x):
        out = x

        # layer name: conv1
        out = self.conv1(out)
        out = self.bn(out)
        out = self.relu(out)

        # layer name: conv2_x
        out = self.maxpool(out)
        out = self.layer1(out)

        # layer name: conv3_x
        out = self.layer2(out)

        # layer name: conv4_x
        out = self.layer3(out)

        # layer name: conv5_x
        out = self.layer4(out)

        # fully connected
        out = self.avgpool(out)
        out = self.flatten(out)
        out = self.fc(out)

        return out
