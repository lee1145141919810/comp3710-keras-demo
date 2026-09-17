"""ResNet-18 for 32x32 CIFAR-10 images (He et al., 2016), written from scratch.

Differences from the ImageNet ResNet-18 in torchvision:
    * the stem is a single 3x3 convolution with stride 1 and there is **no** max-pool, so the first
      stage works at full 32x32 resolution (an ImageNet stem would shrink 32x32 down to 8x8 before
      the residual blocks even start);
    * the classifier is a global average pool followed by one linear layer.

Layout:  conv3x3(64) -> [BasicBlock x2]x4 with 64/128/256/512 channels -> avgpool -> fc(10)
Total 11.2 M parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Two 3x3 convolutions with a residual (identity or 1x1-projection) shortcut."""

    expansion = 1

    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes),
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class ResNet(nn.Module):
    def __init__(self, block: type[BasicBlock], num_blocks: list[int], num_classes: int = 10, width: int = 64, zero_init_residual: bool = True):
        super().__init__()
        self.in_planes = width
        self.conv1 = nn.Conv2d(3, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(width)
        self.layer1 = self._make_layer(block, width, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, width * 2, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, width * 4, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, width * 8, num_blocks[3], stride=2)
        self.fc = nn.Linear(width * 8 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
        if zero_init_residual:
            # Start each residual branch as the identity: makes the very high learning rates used
            # for fast training noticeably more stable (Goyal et al., 2017).
            for m in self.modules():
                if isinstance(m, BasicBlock):
                    nn.init.zeros_(m.bn2.weight)

    def _make_layer(self, block: type[BasicBlock], planes: int, num_blocks: int, stride: int) -> nn.Sequential:
        layers = []
        for s in [stride] + [1] * (num_blocks - 1):
            layers.append(block(self.in_planes, planes, s))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.layer4(self.layer3(self.layer2(self.layer1(out))))
        out = F.adaptive_avg_pool2d(out, 1).flatten(1)
        return self.fc(out)


def resnet18(num_classes: int = 10, width: int = 64) -> ResNet:
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, width=width)


if __name__ == "__main__":
    model = resnet18()
    print(model)
    print(f"parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("output:", model(torch.randn(2, 3, 32, 32)).shape)
