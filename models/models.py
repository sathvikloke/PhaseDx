"""
models.py
---------
Defines the classifier used for all three conditions and all three organs.
Uses a pretrained ResNet-18, adapted for 1 or 2 input channels.
"""

import torch
import torch.nn as nn
from torchvision.models import resnet18, ResNet18_Weights
from typing import Optional


class PhaseAwareClassifier(nn.Module):
    """
    ResNet-18 adapted for variable input channels (1 or 2).

    in_channels:
        1 -> magnitude only (Condition A) or phase only (Condition B)
        2 -> magnitude + phase stacked (Condition C)

    The first conv layer is re-initialized to handle in_channels != 3.
    Pretrained ImageNet weights are used for all layers except the
    first conv (which we adapt) and the final FC layer.
    """

    def __init__(
        self,
        in_channels: int = 2,
        num_classes: int = 2,
        pretrained: bool = True,
        dropout_p: float = 0.3,
    ):
        super().__init__()
        assert in_channels in (1, 2), \
            f"in_channels must be 1 or 2, got {in_channels}"

        self.in_channels = in_channels

        # Load pretrained ResNet-18
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        # Adapt first conv layer for in_channels != 3
        original_conv = backbone.conv1
        new_conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=original_conv.out_channels,
            kernel_size=original_conv.kernel_size,
            stride=original_conv.stride,
            padding=original_conv.padding,
            bias=False,
        )

        if pretrained and in_channels != 3:
            # Initialize new conv by averaging pretrained weights across channel dim
            with torch.no_grad():
                pretrained_weight = original_conv.weight  # (64, 3, 7, 7)
                # Average across input channels, then repeat for in_channels
                avg_weight = pretrained_weight.mean(dim=1, keepdim=True)  # (64,1,7,7)
                new_conv.weight.copy_(avg_weight.repeat(1, in_channels, 1, 1))

        backbone.conv1 = new_conv

        # Replace final FC layer
        in_features = backbone.fc.in_features
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout_p),
            nn.Linear(in_features, num_classes),
        )

        self.backbone = backbone

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def build_model(
    mode: str,
    num_classes: int = 2,
    pretrained: bool = True,
    dropout_p: float = 0.3,
) -> PhaseAwareClassifier:
    """
    Convenience factory.
    mode: 'magnitude', 'phase', or 'both'
    """
    in_channels = 2 if mode == "both" else 1
    return PhaseAwareClassifier(
        in_channels=in_channels,
        num_classes=num_classes,
        pretrained=pretrained,
        dropout_p=dropout_p,
    )
