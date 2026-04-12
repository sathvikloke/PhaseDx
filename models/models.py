"""
models.py — PhaseAwareClassifier with frozen backbone for small datasets.
Only layer4 and fc are trained — prevents overfitting on ~85 training exams.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights


class PhaseAwareClassifier(nn.Module):
    def __init__(self, in_channels=2, num_classes=2, pretrained=True, dropout_p=0.5):
        super().__init__()
        assert in_channels in (1, 2)
        self.in_channels = in_channels

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt first conv for 1 or 2 channels
        orig = backbone.conv1
        new_conv = nn.Conv2d(in_channels, orig.out_channels,
                             orig.kernel_size, orig.stride, orig.padding, bias=False)
        if pretrained:
            with torch.no_grad():
                avg = orig.weight.mean(dim=1, keepdim=True)
                new_conv.weight.copy_(avg.repeat(1, in_channels, 1, 1))
        backbone.conv1 = new_conv

        # Freeze everything except layer4 and fc
        for name, param in backbone.named_parameters():
            if not any(x in name for x in ["layer4", "fc"]):
                param.requires_grad = False

        # Replace FC with stronger classifier head
        in_features = backbone.fc.in_features
        fc = nn.Linear(in_features, num_classes)
        nn.init.xavier_uniform_(fc.weight, gain=0.01)
        nn.init.zeros_(fc.bias)
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            fc,
        )
        self.backbone = backbone

    def forward(self, x):
        return self.backbone(x)


def build_model(mode, num_classes=2, pretrained=True, dropout_p=0.5):
    in_channels = 2 if mode == "both" else 1
    return PhaseAwareClassifier(in_channels, num_classes, pretrained, dropout_p)
