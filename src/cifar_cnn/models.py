import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_ch: int, out_ch: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_ch)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_ch)

        # When size or channels change, project x so it can be added to F(x)
        self.shortcut = nn.Identity()
        if stride != 1 or in_ch != out_ch:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_ch),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)), inplace=True)
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x), inplace=True)


class CifarResNet(nn.Module):
    def __init__(self, blocks_per_stage=(2, 2, 2, 2), widths=(64, 128, 256, 512),
                 num_classes: int = 10, dropout: float = 0.0):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(3, widths[0], 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(widths[0]),
            nn.ReLU(inplace=True),
        )
        layers, in_ch = [], widths[0]
        for i, (n, w) in enumerate(zip(blocks_per_stage, widths, strict=True)):
            stride = 1 if i == 0 else 2
            for j in range(n):
                layers.append(BasicBlock(in_ch, w, stride if j == 0 else 1))
                in_ch = w
        self.features = nn.Sequential(*layers)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(in_ch, num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
        # Zero-init the last BN of each block: every block starts as identity,
        # which makes early training more stable (+0.2-0.5% accuracy)
        for m in self.modules():
            if isinstance(m, BasicBlock):
                nn.init.zeros_(m.bn2.weight)

    def forward(self, x):
        x = self.stem(x)
        x = self.features(x)
        x = torch.flatten(self.pool(x), 1)
        return self.fc(self.dropout(x))


MODELS = {
    "resnet_small": lambda nc, d: CifarResNet((1, 1, 1), (32, 64, 128), nc, d),  # ~0.31M params, CPU-friendly
    "resnet18":     lambda nc, d: CifarResNet((2, 2, 2, 2), (64, 128, 256, 512), nc, d),
    "resnet34":     lambda nc, d: CifarResNet((3, 4, 6, 3), (64, 128, 256, 512), nc, d),
}


def build_model(name: str, num_classes: int = 10, dropout: float = 0.0) -> nn.Module:
    if name not in MODELS:
        raise ValueError(f"Unknown model {name!r}. Choose from {list(MODELS)}")
    return MODELS[name](num_classes, dropout)