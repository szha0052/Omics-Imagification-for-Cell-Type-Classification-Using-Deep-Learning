import torch
import torch.nn as nn

# Basic Convolutional Block
class ConvBNReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding='same'):
        super().__init__()
        if padding == 'same':
            pad = kernel_size // 2
        else:
            pad = 0
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=pad),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.block(x)

# Residual Block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.conv1 = ConvBNReLU(in_channels, out_channels, 3, stride)
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if in_channels != out_channels or stride != 1:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        identity = self.shortcut(x)
        out = self.conv1(x)
        out = self.conv2(out)
        out += identity
        return self.relu(out)

# Residual Stage
class ResidualStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block1 = ResidualBlock(in_channels, out_channels, stride)
        self.block2 = ResidualBlock(out_channels, out_channels, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        return x

# Main Network: Vec2ImageNet
class Vec2ImageNet(nn.Module):
    def __init__(self, input_channels, num_classes, filter_size=3, initial_filters=16):
        super().__init__()
        f = initial_filters

        # Main branch
        self.stem = ConvBNReLU(input_channels, f, kernel_size=filter_size)
        self.stage1 = ResidualStage(f, f, stride=1)
        self.stage2 = ResidualStage(f, f * 2, stride=2)
        self.stage3 = ResidualStage(f * 2, f * 4, stride=2)

        # Auxiliary branch
        self.stem2 = ConvBNReLU(input_channels, f, kernel_size=filter_size)
        self.l2_stage1 = ResidualStage(f, f, stride=1)
        self.l2_stage2 = ResidualStage(f, f * 2, stride=2)
        self.l2_stage3 = ResidualStage(f * 2, f * 4, stride=2)

        # Merge branches and classifier
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.classifier = nn.Linear(f * 4, num_classes)

    def forward(self, x):
        # Main branch
        x1 = self.stem(x)
        x1 = self.stage1(x1)
        x1 = self.stage2(x1)
        x1 = self.stage3(x1)

        # Auxiliary branch
        x2 = self.stem2(x)
        x2 = self.l2_stage1(x2)
        x2 = self.l2_stage2(x2)
        x2 = self.l2_stage3(x2)

        # Merge branches
        out = x1 + x2
        out = self.global_pool(out)
        out = out.view(out.size(0), -1)
        out = self.classifier(out)
        return out
