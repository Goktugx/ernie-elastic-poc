"""
Baseline CNN Model

Vanilla CNN without MoE for comparison against elastic training.
Available in small, medium, and large variants.
"""

import torch
import torch.nn as nn


class BaselineCNN(nn.Module):
    """Simple CNN model. Size parameter controls capacity."""

    def __init__(self, num_classes=10, size="large"):
        super().__init__()
        configs = {
            "large": {"channels": [64, 128, 256], "fc_dim": 512, "num_fc": 3},
            "medium": {"channels": [48, 96, 192], "fc_dim": 384, "num_fc": 2},
            "small": {"channels": [32, 64, 128], "fc_dim": 256, "num_fc": 1},
        }
        cfg = configs[size]
        self.size = size

        # Conv layers
        layers = []
        in_channels = 3
        for out_channels in cfg["channels"]:
            layers.extend([
                nn.Conv2d(in_channels, out_channels, 3, padding=1),
                nn.BatchNorm2d(out_channels),
                nn.GELU(),
                nn.MaxPool2d(2),
            ])
            in_channels = out_channels
        self.features = nn.Sequential(*layers)

        # CIFAR-10: 32x32 -> 3 pools -> 4x4
        feature_size = cfg["channels"][-1] * 4 * 4

        # FC layers
        fc_layers = []
        fc_in = feature_size
        for _ in range(cfg["num_fc"]):
            fc_layers.extend([
                nn.Linear(fc_in, cfg["fc_dim"]),
                nn.GELU(),
                nn.Dropout(0.3),
            ])
            fc_in = cfg["fc_dim"]
        fc_layers.append(nn.Linear(fc_in, num_classes))
        self.classifier = nn.Sequential(*fc_layers)

    def forward(self, x):
        h = self.features(x)
        h = h.view(h.size(0), -1)
        return self.classifier(h)
