#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# models/__init__.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# VGG-like model
class VGG_like_model(nn.Module):
    def __init__(self):
        super(VGG_like_model, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(128)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(256)
        self.conv4 = nn.Conv2d(256, 512, kernel_size=3, padding=0, bias=False)
        self.bn4 = nn.BatchNorm2d(512)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(0.4)
        self.fc1 = nn.Linear(512 * 3 * 3, 128)
        self.fc2 = nn.Linear(128, 1)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(F.relu(self.bn2(self.conv2(x))))
        x = F.relu(self.bn3(self.conv3(x)))
        x = self.pool2(F.relu(self.bn4(self.conv4(x))))
        x = x.view(x.size(0), -1)
        x = self.dropout(F.relu(self.fc1(x)))
        return torch.sigmoid(self.fc2(x))

# ResNet-like model
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        identity = self.downsample(x) if self.downsample else x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + identity)

class ResNet_like_model(nn.Module):
    def __init__(self, block, layers, num_classes=1):
        super(ResNet_like_model, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        layers = [block(self.in_channels, out_channels, stride, downsample)]
        self.in_channels = out_channels
        layers += [block(out_channels, out_channels) for _ in range(1, blocks)]
        return nn.Sequential(*layers)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        return torch.sigmoid(self.fc(x))

# InceptionNet-like model
class InceptionBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InceptionBlock, self).__init__()
        self.branch1x1 = nn.Conv2d(in_channels, out_channels // 4, 1, bias=False)
        self.branch5x5 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 8, 1, bias=False),
            nn.Conv2d(out_channels // 8, out_channels // 4, 5, padding=2, bias=False),
        )
        self.branch3x3dbl = nn.Sequential(
            nn.Conv2d(in_channels, out_channels // 8, 1, bias=False),
            nn.Conv2d(out_channels // 8, out_channels // 4, 3, padding=1, bias=False),
            nn.Conv2d(out_channels // 4, out_channels // 4, 3, padding=1, bias=False),
        )
        self.branch_pool = nn.Sequential(
            nn.AvgPool2d(3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_channels // 4, 1, bias=False)
        )
        self.bn = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        outputs = [
            self.branch1x1(x),
            self.branch5x5(x),
            self.branch3x3dbl(x),
            self.branch_pool(x)
        ]
        return self.bn(torch.cat(outputs, 1))

class InceptionNet_like_model(nn.Module):
    def __init__(self):
        super(InceptionNet_like_model, self).__init__()
        self.initial = nn.Sequential(
            nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1)
        )
        self.inception3a = InceptionBlock(64, 64)
        self.inception3b = InceptionBlock(64, 128)
        self.pool3 = nn.MaxPool2d(3, stride=2, padding=1)
        self.inception4a = InceptionBlock(128, 256)
        self.inception4b = InceptionBlock(256, 512)
        self.pool4 = nn.MaxPool2d(3, stride=2, padding=1)
        self.head = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Dropout(0.4),
            nn.Linear(512, 1),
        )

    def forward(self, x):
        x = self.initial(x)
        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.pool3(x)
        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.pool4(x)
        return torch.sigmoid(self.head(x))

