"""
WideResNet-28-10 for CIFAR-10 Uncertainty Quantification.

Architecture: depth=28, widen_factor=10
- Initial 3×3 convolution (no max-pooling)
- 3 wide residual groups (BasicBlock with BN-ReLU-Conv pattern)
- Downsampling via stride-2 convolutions
- Global average pooling → 10-class linear head
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    """Wide residual block: BN-ReLU-Conv-BN-ReLU-Conv with optional shortcut."""

    def __init__(self, in_planes, out_planes, stride, dropout_rate=0.0):
        super().__init__()
        self.bn1 = nn.BatchNorm2d(in_planes)
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.conv2 = nn.Conv2d(out_planes, out_planes, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.dropout_rate = dropout_rate

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != out_planes:
            self.shortcut = nn.Conv2d(in_planes, out_planes, kernel_size=1,
                                      stride=stride, bias=False)

    def forward(self, x):
        out = self.conv1(F.relu(self.bn1(x)))
        if self.dropout_rate > 0:
            out = F.dropout(out, p=self.dropout_rate, training=self.training)
        out = self.conv2(F.relu(self.bn2(out)))
        out += self.shortcut(x)
        return out


class WideResNetGroup(nn.Module):
    """A group of N stacked BasicBlocks."""

    def __init__(self, num_blocks, in_planes, out_planes, stride, dropout_rate=0.0):
        super().__init__()
        layers = []
        for i in range(num_blocks):
            s = stride if i == 0 else 1
            inp = in_planes if i == 0 else out_planes
            layers.append(BasicBlock(inp, out_planes, s, dropout_rate))
        self.blocks = nn.Sequential(*layers)

    def forward(self, x):
        return self.blocks(x)


class WideResNet(nn.Module):
    """
    WideResNet-28-10 for CIFAR.

    Parameters
    ----------
    depth : int
        Network depth (28 for WRN-28-10). Must satisfy (depth - 4) % 6 == 0.
    widen_factor : int
        Width multiplier (10 for WRN-28-10).
    num_classes : int
        Number of output classes (10 for CIFAR-10).
    dropout_rate : float
        Dropout rate inside residual blocks.
    """

    def __init__(self, depth=28, widen_factor=10, num_classes=10, dropout_rate=0.0):
        super().__init__()
        assert (depth - 4) % 6 == 0, "Depth must satisfy (depth - 4) % 6 == 0"
        n = (depth - 4) // 6  # number of blocks per group

        channels = [16, 16 * widen_factor, 32 * widen_factor, 64 * widen_factor]

        # Initial convolution — no max-pooling
        self.conv1 = nn.Conv2d(3, channels[0], kernel_size=3, stride=1,
                               padding=1, bias=False)

        # Three wide residual groups
        self.group1 = WideResNetGroup(n, channels[0], channels[1], stride=1,
                                       dropout_rate=dropout_rate)
        self.group2 = WideResNetGroup(n, channels[1], channels[2], stride=2,
                                       dropout_rate=dropout_rate)
        self.group3 = WideResNetGroup(n, channels[2], channels[3], stride=2,
                                       dropout_rate=dropout_rate)

        # Final BN-ReLU before pooling
        self.bn = nn.BatchNorm2d(channels[3])

        # Classification head
        self.fc = nn.Linear(channels[3], num_classes)

        # Weight initialization
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x):
        out = self.conv1(x)
        out = self.group1(out)
        out = self.group2(out)
        out = self.group3(out)
        out = F.relu(self.bn(out))
        out = F.adaptive_avg_pool2d(out, 1)
        out = out.view(out.size(0), -1)
        out = self.fc(out)
        return out


def get_model(num_classes=10, depth=28, widen_factor=10, dropout_rate=0.0):
    """Factory function for WideResNet."""
    return WideResNet(
        depth=depth,
        widen_factor=widen_factor,
        num_classes=num_classes,
        dropout_rate=dropout_rate,
    )


if __name__ == "__main__":
    # Quick sanity check
    model = get_model()
    x = torch.randn(2, 3, 32, 32)
    out = model(x)
    print(f"Model output shape: {out.shape}")
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params:,}")
