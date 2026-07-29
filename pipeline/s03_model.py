"""
s03_model.py
------------
Stage 3a of the PhaseDx pipeline: the classifier.

One architecture (ImageNet-pretrained ResNet-18) is used for all three input
conditions so that any difference in performance between magnitude, phase, and
both is attributable to the *input*, not to the model. That is the entire point
of the study, and it is why the channel adaptation below has to be done
carefully rather than by just re-initialising conv1.

Input conditions and channel counts
    'magnitude' -> 1 channel   z-scored magnitude
    'phase'     -> 2 channels  (sin, cos) of the phase in radians
    'both'      -> 3 channels  (magnitude, sin, cos)

Why sin/cos and not the raw angle
    Phase is circular. -pi and +pi are the same angle but are maximally far
    apart as floating point numbers, so a 7x7 convolution over raw wrapped
    phase sees a huge artificial edge along every wrap fringe. Those fringes
    are dense in real MRI (B0 inhomogeneity alone produces many), so a network
    fed raw angle spends its capacity modelling wrap artefacts. common.
    phase_channels() does the encoding; this module only needs to know that the
    'phase' condition arrives with 2 channels.

Why conv1 is averaged-and-repeated rather than re-initialised
    The pretrained conv1 is (64, 3, 7, 7) and computes, for each output filter,
    sum_c W_c * x_c. On a grayscale image replicated across RGB that equals
    (sum_c W_c) * x = 3 * mean_c(W_c) * x. So the correct grayscale-equivalent
    kernel is 3 * mean over the RGB axis. Generalising to in_ch input channels,
    we set every input channel's kernel to mean_c(W_c) and multiply by
    (3 / in_ch). The pre-activation statistics entering bn1 then match what the
    pretrained batch-norm running stats expect, for any in_ch. Re-initialising
    conv1 randomly instead throws away the Gabor-like edge detectors that are
    the main thing ImageNet pretraining buys on small medical datasets, and it
    also breaks bn1's running statistics.

    Note the scale factor is exactly right only if the input channels are
    similarly scaled. They are: magnitude is z-scored per slice and sin/cos are
    both bounded in [-1, 1].

Optional freezing
    With ~600 positive slices in the prostate cohort, a fully trainable
    11.2M-parameter ResNet-18 can memorise the training set. --freeze-backbone
    freezes conv1/bn1/layer1/layer2/layer3 and trains only layer4 + the head
    (~8.4M params), which is the standard low-data transfer recipe. Frozen
    batch-norm layers are additionally held in eval mode so their running
    statistics do not drift toward the new input distribution.

Usage:
    python pipeline/s03_model.py          # prints the parameter table
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402

logger = logging.getLogger(__name__)

# Channel count per input condition. Keep in sync with the Dataset in
# s03_train.py -- these two are the only places that encode it.
CONDITION_CHANNELS = {
    "magnitude": 1,
    "phase": 2,
    "both": 3,
}

# The blocks frozen by --freeze-backbone, in forward order.
FROZEN_BLOCK_NAMES = ("conv1", "bn1", "layer1", "layer2", "layer3")


def input_channels_for(condition: str) -> int:
    """Number of input channels required by a condition."""
    try:
        return CONDITION_CHANNELS[condition]
    except KeyError:
        raise ValueError(
            f"unknown condition {condition!r}; expected one of {sorted(CONDITION_CHANNELS)}"
        ) from None


def adapt_conv1(conv1: nn.Conv2d, in_channels: int) -> nn.Conv2d:
    """
    Replace a pretrained 3-channel conv1 with an in_channels-channel one.

    weight_new[:, i] = mean over RGB of weight_old, times 3 / in_channels.

    The mean makes each new channel a grayscale-equivalent filter; the 3/in_ch
    factor keeps sum-over-channels (and therefore the variance entering bn1)
    the same as it was for a replicated-grayscale RGB input. See the module
    docstring for the derivation.
    """
    if conv1.in_channels != 3:
        raise ValueError(f"expected a 3-channel conv1 to adapt, got {conv1.in_channels}")

    new = nn.Conv2d(
        in_channels,
        conv1.out_channels,
        kernel_size=conv1.kernel_size,
        stride=conv1.stride,
        padding=conv1.padding,
        bias=conv1.bias is not None,
    )
    with torch.no_grad():
        # (64, 3, 7, 7) -> (64, 1, 7, 7) -> (64, in_ch, 7, 7)
        mean_rgb = conv1.weight.mean(dim=1, keepdim=True)
        new.weight.copy_(mean_rgb.repeat(1, in_channels, 1, 1) * (3.0 / in_channels))
        if conv1.bias is not None:
            new.bias.copy_(conv1.bias)
    return new


class PhaseDxResNet18(nn.Module):
    """
    ResNet-18 with a condition-dependent input stem and a 2-class dropout head.

    Attributes worth reading back out at logging time:
        condition, in_channels, freeze_backbone, pretrained
    """

    def __init__(
        self,
        condition: str,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3,
        num_classes: int = 2,
    ):
        super().__init__()
        from torchvision import models

        self.condition = condition
        self.in_channels = input_channels_for(condition)
        self.freeze_backbone = bool(freeze_backbone)
        self.num_classes = num_classes

        weights = None
        if pretrained:
            try:
                weights = models.ResNet18_Weights.IMAGENET1K_V1
            except Exception as exc:  # noqa: BLE001
                logger.warning("Could not resolve pretrained weights enum: %s", exc)

        try:
            backbone = models.resnet18(weights=weights)
            self.pretrained = weights is not None
        except Exception as exc:  # noqa: BLE001
            # No network and no cached checkpoint. Fall back rather than crash,
            # but make it extremely visible -- comparing a pretrained run
            # against a randomly-initialised one would silently invalidate the
            # whole magnitude-vs-phase comparison.
            logger.error("PRETRAINED WEIGHTS UNAVAILABLE (%s); falling back to random init", exc)
            backbone = models.resnet18(weights=None)
            self.pretrained = False

        backbone.conv1 = adapt_conv1(backbone.conv1, self.in_channels)

        feat_dim = backbone.fc.in_features  # 512 for resnet18
        backbone.fc = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(feat_dim, num_classes),
        )
        self.backbone = backbone
        self.feature_dim = feat_dim

        self._frozen_modules = []
        if self.freeze_backbone:
            for name in FROZEN_BLOCK_NAMES:
                module = getattr(self.backbone, name)
                for p in module.parameters():
                    p.requires_grad_(False)
                self._frozen_modules.append(module)

    def train(self, mode: bool = True):
        """
        Standard train() plus: frozen blocks stay in eval mode.

        Without this, the BatchNorm layers inside a "frozen" block still update
        their running mean/var on every forward pass, so the block is not
        actually frozen -- its output drifts across training even though no
        gradient touches it. That silently changes the features layer4 sees.
        """
        super().train(mode)
        if mode:
            for module in self._frozen_modules:
                module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


def count_parameters(model: nn.Module) -> dict:
    """Total / trainable / frozen parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total": total, "trainable": trainable, "frozen": total - trainable}


def build_model(
    condition: str,
    pretrained: bool = True,
    freeze_backbone: bool = False,
    dropout: float = 0.3,
    num_classes: int = 2,
    verbose: bool = True,
) -> PhaseDxResNet18:
    """Construct the model for a condition and log its parameter budget."""
    model = PhaseDxResNet18(
        condition=condition,
        pretrained=pretrained,
        freeze_backbone=freeze_backbone,
        dropout=dropout,
        num_classes=num_classes,
    )
    counts = count_parameters(model)
    if verbose:
        logger.info(
            "model[%s] in_ch=%d pretrained=%s frozen_backbone=%s | "
            "trainable %s / %s params (%.1f%%)",
            condition,
            model.in_channels,
            model.pretrained,
            model.freeze_backbone,
            f"{counts['trainable']:,}",
            f"{counts['total']:,}",
            100.0 * counts["trainable"] / max(1, counts["total"]),
        )
    return model


def _self_check():
    """
    Print the parameter table and verify the conv1 scaling claim numerically.

    The check: push the same grayscale image through the original 3-channel
    conv1 (replicated to RGB) and through each adapted conv1 (replicated to
    in_ch). The pre-activations must agree to floating point tolerance,
    otherwise the "activation magnitude is preserved" claim is false and bn1's
    running statistics are wrong for the new stem.
    """
    from torchvision import models

    logging.basicConfig(level=logging.INFO, format="%(message)s")

    print("=" * 78)
    print("PARAMETER BUDGET")
    print("=" * 78)
    print(f"{'condition':<12}{'in_ch':>6}{'freeze':>8}{'trainable':>14}{'total':>14}{'%':>8}")
    for condition in common.CONDITIONS:
        for freeze in (False, True):
            m = build_model(condition, pretrained=True, freeze_backbone=freeze, verbose=False)
            c = count_parameters(m)
            print(
                f"{condition:<12}{m.in_channels:>6}{str(freeze):>8}"
                f"{c['trainable']:>14,}{c['total']:>14,}"
                f"{100.0 * c['trainable'] / c['total']:>7.1f}%"
            )

    print()
    print("=" * 78)
    print("CONV1 ADAPTATION CHECK (pre-activation must match replicated-RGB baseline)")
    print("=" * 78)
    ref = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).conv1
    torch.manual_seed(0)
    gray = torch.randn(2, 1, 64, 64)
    with torch.no_grad():
        baseline = ref(gray.repeat(1, 3, 1, 1))
        for condition in common.CONDITIONS:
            in_ch = input_channels_for(condition)
            adapted = adapt_conv1(ref, in_ch)
            out = adapted(gray.repeat(1, in_ch, 1, 1))
            err = (out - baseline).abs().max().item()
            scale = out.std().item() / baseline.std().item()
            print(
                f"  {condition:<12} in_ch={in_ch}  max|diff|={err:.3e}  "
                f"std_ratio={scale:.6f}  {'OK' if err < 1e-4 else 'FAIL'}"
            )

    print()
    print("=" * 78)
    print("FORWARD SHAPE CHECK")
    print("=" * 78)
    for condition in common.CONDITIONS:
        m = build_model(condition, pretrained=False, verbose=False).eval()
        x = torch.randn(3, input_channels_for(condition), *common.TARGET_HW)
        with torch.no_grad():
            y = m(x)
        print(f"  {condition:<12} {tuple(x.shape)} -> {tuple(y.shape)}")

    print()
    print("=" * 78)
    print("FROZEN-BN CHECK (running stats must not move in frozen blocks)")
    print("=" * 78)
    m = build_model("both", pretrained=False, freeze_backbone=True, verbose=False)
    m.train()
    before_frozen = m.backbone.bn1.running_mean.clone()
    before_free = m.backbone.layer4[0].bn1.running_mean.clone()
    for _ in range(3):
        m(torch.randn(4, 3, 64, 64))
    d_frozen = (m.backbone.bn1.running_mean - before_frozen).abs().max().item()
    d_free = (m.backbone.layer4[0].bn1.running_mean - before_free).abs().max().item()
    print(f"  bn1 (frozen)          drift={d_frozen:.3e}  {'OK' if d_frozen == 0 else 'FAIL'}")
    print(f"  layer4.0.bn1 (train)  drift={d_free:.3e}  {'OK' if d_free > 0 else 'FAIL'}")


if __name__ == "__main__":
    _self_check()
