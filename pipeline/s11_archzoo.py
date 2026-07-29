#!/usr/bin/env python3
"""
s11_archzoo.py
==============
Stage 11: the architecture zoo. Answers ONE reviewer objection, the one a
hostile audit ranked most likely to sink the paper:

    "YOUR MODEL WAS BAD, NOT THE PHASE."

Every headline number in this study came from a single architecture -- an
ImageNet-pretrained ResNet-18, 20 epochs, 2 seeds. A null produced by one weak
model is a statement about that model, not about phase. This module makes the
architecture a swept variable so the null can be re-read as "six architectures,
including one that represents phase natively, all fail to beat chance" rather
than "our one CNN failed".

WHAT THIS IS NOT
----------------
It is not a second training loop. Everything here plugs into the EXISTING path:
``s03_train.run_one`` does the training, the splits, the sampler, the early
stopping, the JSON schema. This module only supplies a different ``nn.Module``
and refiles the output. The mechanism is the same one ``s05_controls`` already
uses for its dataset transforms -- ``run_one`` resolves ``build_model`` from its
own module globals at call time, so temporarily rebinding
``s03_train.build_model`` is enough:

    with architecture("convnext_tiny"):
        s03_train.run_one(cohort, condition, seed, index, device, args, ...)

s03_train is NOT modified. No ``--arch`` flag was added to it, so every existing
result stays reproducible byte-for-byte by the command line that produced it.
The one thing ``run_one`` hard-codes is ``payload["model"]["arch"] = "resnet18"``;
rather than patch that, ``run_tagged()`` here overwrites the field after the
fact (exactly as s05 rewrites the payload it gets back), so a swept JSON can
never mislabel which network produced it.

THE ARCHITECTURES
-----------------
    resnet18       the incumbent. Literally s03_model.PhaseDxResNet18 -- this
                   entry delegates, it does not reimplement, so the sweep's
                   resnet18 arm is bit-identical to the published runs. Proved,
                   not asserted: see --regression.
    resnet50       capacity control. 4x the parameters, same inductive bias.
                   If the null is "too small a model", this moves it.
    densenet121    dense connectivity, ~8M params. Different feature reuse
                   pattern at similar capacity to resnet18.
    vit_b_16       attention instead of convolution. No locality prior at all,
                   which matters because phase structure in raw k-space
                   reconstructions is global (B0 shim, coil sensitivity), not
                   local. Requires exactly 224x224 -- which is TARGET_HW.
    convnext_tiny  modern conv baseline; rules out "your CNN is from 2015".
    complex_small  a complex-valued CNN that consumes the complex image
                   directly. See below.

CHANNEL ADAPTATION
------------------
Identical rule for all five real-valued nets, and it is the SAME CODE, not a
re-derivation: ``s03_model.adapt_conv1`` is imported and applied to whatever the
first convolution happens to be (``conv1`` for ResNet, ``features.conv0`` for
DenseNet, ``conv_proj`` -- the patch embedding -- for ViT, ``features.0.0``
for ConvNeXt). All four are plain ``Conv2d`` with groups=1, dilation=1,
padding_mode='zeros', so the adapter transfers exactly:

    W_new[:, i] = mean_RGB(W_old) * (3 / in_channels)

The mean gives each input channel a grayscale-equivalent filter; the 3/in_ch
factor preserves the sum over channels and therefore the activation magnitude
entering the first normalisation layer. --self-test verifies this numerically
per architecture (pre-activations must match the replicated-RGB baseline).

THE COMPLEX-VALUED MODEL, AND WHY IT IS THE POINT
-------------------------------------------------
The honest form of the reviewer's objection is not "ResNet-18 is small". It is:

    feeding phase to a real-valued CNN as two independent channels (sin, cos)
    cannot represent phase relationships natively. A complex multiplication --
    the operation that actually mixes phase -- is not in the hypothesis class.
    Of course it found nothing.

That criticism is correct about the mechanism, so it is answered with a model
that does have complex multiplication in its hypothesis class.

``ComplexSmallCNN`` (design after Trabelsi et al., "Deep Complex Networks",
ICLR 2018, arXiv:1705.09792; modReLU from Arjovsky, Shah & Bengio, "Unitary
Evolution Recurrent Neural Networks", ICML 2016, arXiv:1511.06464; MRI
motivation from Virtue, Yu & Lustig, ICIP 2017, and Cole et al.,
arXiv:2004.01738):

  * ComplexConv2d implements (W_r + iW_i)(x_r + ix_i) as two real convolutions,
    so a learned filter can rotate phase, not merely reweight sin and cos.
  * modReLU(z) = z * ReLU(|z| + b) / |z|, i.e. a magnitude threshold that
    leaves the phase untouched. CReLU (independent ReLU on real and imaginary
    parts) is also implemented and selectable.
  * Normalisation: two options. ``modulus`` divides by sqrt(E|z|^2) per channel
    with a real gain and no additive bias. ``whiten`` is Trabelsi's full 2x2
    covariance whitening. Modulus is the default -- it is the one that keeps
    the invariance property below under training.
  * The head takes |z| (magnitude), global-average-pools it, then Dropout +
    Linear, matching s03_model's head exactly.

Input adaptation is exact and lossless -- the complex net sees precisely the
information the real nets see, re-assembled:

    magnitude (1 ch)  z            -> (z, 0i)              purely real
    phase     (2 ch)  (sin, cos)   -> (cos, i*sin)         unit modulus, e^{i phi}
    both      (3 ch)  (z,sin,cos)  -> (z cos, i z sin)     = z * e^{i phi}

so the ``both`` condition literally reconstructs the complex image. No extra
information is smuggled in, which is what makes the comparison fair.

PROPERTY WORTH HAVING, AND TESTED: with complex convolutions carrying no
additive bias, modulus normalisation, modReLU, and a magnitude head, the
network is EXACTLY invariant to a global phase rotation of the input,
f(e^{i theta} z) = f(z). That is the physically right prior -- the absolute
phase reference of a receive chain is arbitrary, only phase DIFFERENCES carry
information -- and it is asserted numerically in --self-test, in both train and
eval mode.

The self-test does not stop at asserting the property, because an invariance
test that passes for everything proves nothing. It also establishes where the
property comes from and where it fails:

  * modulus + CReLU is measurably NOT invariant (drift ~1e-1). CReLU is defined
    on the real/imaginary decomposition, so it carves the complex plane into
    quadrants and cannot be equivariant. This is the negative control.
  * whiten + modReLU IS invariant at initialisation -- a fact that is easy to
    get wrong. Whitening by the SYMMETRIC inverse square root commutes with
    rotation, since (R V R^T)^{-1/2} = R V^{-1/2} R^T, and at initialisation
    Trabelsi's learnable Gamma is isotropic (gamma_rr = gamma_ii, gamma_ri = 0)
    with beta = 0. Perturb Gamma off isotropy -- i.e. train it -- and the drift
    jumps to ~0.4. So whitening does not break the property by construction;
    it stops guaranteeing it once the layer has learned. That is why modulus is
    the default and why the claim is stated as a property of the default
    configuration, not of complex networks in general.

Complex nets have no ImageNet weights. ``complex_small`` therefore reports
pretrained=False always, and the fair comparison for it is the ``--scratch``
arm of the real-valued nets, which is why --scratch exists.

--scratch
---------
ImageNet-pretrained RGB features are themselves a confound worth ruling out: a
network whose first layers are edge detectors tuned on photographs may simply
be a poor phase reader. --scratch trains the same architectures from random
initialisation.

CAVEATS THIS MODULE WILL NOT PAPER OVER
---------------------------------------
* A sweep that finds nothing is evidence about the hypothesis class actually
  searched. It cannot exclude an architecture nobody ran. State it that way.
* Sweeping architectures is a multiplicity: six architectures x three
  conditions is eighteen looks per cohort, and the maximum of eighteen noisy
  AUCs is biased upward. Read the sweep as "does ANY architecture clear
  chance", and if one does, its nominal CI is not a p-value. s04's
  subject-clustered bootstrap is still the interval to quote.
* Hyperparameters were tuned (loosely) for resnet18. ViT in particular usually
  wants a different LR. A ViT arm that underperforms is therefore weak evidence
  on its own; it is included for the inductive-bias contrast, not as a
  well-tuned competitor. --lr is passed through for exactly this reason.

USAGE
-----
    # every architecture at 1/2/3 channels: forward pass, shapes, finiteness
    python pipeline/s11_archzoo.py --self-test

    # prove the resnet18 arm is bit-identical to the published path
    python pipeline/s11_archzoo.py --regression

    # capacity table
    python pipeline/s11_archzoo.py --params

    # end-to-end plumbing check on the fabricated cache (no drive access)
    python pipeline/s11_archzoo.py --smoke --archs resnet18,complex_small

    # a real sweep
    python pipeline/s11_archzoo.py --cohort prostate_t2 --split-col cv0_split \\
        --archs all --conditions all --seeds 42,123 --epochs 20
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import hashlib
import json
import logging
import math
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import s03_model  # noqa: E402
import s03_train  # noqa: E402
from s03_model import adapt_conv1, count_parameters, input_channels_for  # noqa: E402

logger = logging.getLogger("s11_archzoo")

RESULTS_SUBDIR = "archzoo"
SCHEMA = "phasedx.s11.archzoo.v1"

# modReLU with b = 0 is the identity (|z| >= 0 always), which would make the
# whole network linear at initialisation. Initialise the bias negative so the
# layer starts non-linear; b is learnable and free to move back.
MODRELU_BIAS_INIT = -0.2


# ===========================================================================
# Complex-valued building blocks   (Trabelsi et al. 2018 unless noted)
# ===========================================================================

def complex_weight_init(re: nn.Conv2d, im: nn.Conv2d, generator=None) -> None:
    """
    Trabelsi complex initialisation, He variant (arXiv:1705.09792 sec. 3.6).

    Independently initialising the real and imaginary convolutions with the
    usual real-valued scheme gets the variance of the resulting COMPLEX weight
    wrong by a factor of two and destroys the uniform-phase prior. The correct
    construction draws a magnitude from a Rayleigh distribution with
    sigma = 1/sqrt(fan_in) and a phase uniform on (-pi, pi], then sets
    W_r = |W| cos(theta), W_i = |W| sin(theta). Then Var(|W|^2) = 2/fan_in,
    matching He initialisation for a ReLU-family non-linearity, and no phase is
    privileged at initialisation -- which matters here, because a phase prior
    baked in at init is precisely the thing under test.
    """
    fan_in = re.in_channels * re.kernel_size[0] * re.kernel_size[1]
    sigma = 1.0 / math.sqrt(max(1, fan_in))
    with torch.no_grad():
        shape = re.weight.shape
        # Rayleigh(sigma) via inverse CDF: |W| = sigma * sqrt(-2 ln U)
        u = torch.rand(shape, generator=generator).clamp_min(1e-12)
        magnitude = sigma * torch.sqrt(-2.0 * torch.log(u))
        theta = (torch.rand(shape, generator=generator) * 2.0 - 1.0) * math.pi
        re.weight.copy_(magnitude * torch.cos(theta))
        im.weight.copy_(magnitude * torch.sin(theta))
        if re.bias is not None:
            re.bias.zero_()
        if im.bias is not None:
            im.bias.zero_()


class ComplexConv2d(nn.Module):
    """
    (W_r + i W_i) * (x_r + i x_i) = (W_r x_r - W_i x_i) + i (W_r x_i + W_i x_r).

    This is the operation the (sin, cos) real-valued encoding cannot express:
    a single complex filter tap multiplies the input by a learned phasor, so
    the layer can compute phase DIFFERENCES between locations. Two real
    channels convolved by an ordinary Conv2d can only take linear combinations
    of sin and cos with fixed coefficients per tap.

    bias defaults to False. A complex additive bias would break the global
    phase equivariance the rest of the block set is built to preserve.
    """

    def __init__(self, in_ch: int, out_ch: int, kernel_size: int, stride: int = 1,
                 padding: int = 0, bias: bool = False):
        super().__init__()
        self.in_channels = in_ch
        self.out_channels = out_ch
        self.conv_re = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        self.conv_im = nn.Conv2d(in_ch, out_ch, kernel_size, stride, padding, bias=bias)
        complex_weight_init(self.conv_re, self.conv_im)

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        return (self.conv_re(xr) - self.conv_im(xi),
                self.conv_re(xi) + self.conv_im(xr))


class ModulusBatchNorm2d(nn.Module):
    """
    Phase-preserving normalisation: z <- gamma * z / sqrt(E[|z|^2] + eps).

    The statistic is the mean squared modulus per channel, which is invariant
    to a global phase rotation, and the correction is a real scalar, so the
    layer commutes with rotation: N(e^{i th} z) = e^{i th} N(z). No additive
    bias, for the same reason.

    This is a deliberate simplification of Trabelsi's complex batch norm. It
    normalises power without decorrelating the real and imaginary parts, and it
    is what makes the global-phase-invariance property of the whole network
    hold exactly (see the module docstring). Use ComplexBatchNorm2d if you want
    the published whitening version instead.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.register_buffer("running_power", torch.ones(num_features))

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        if self.training:
            power = (xr * xr + xi * xi).mean(dim=(0, 2, 3))
            with torch.no_grad():
                self.running_power.mul_(1 - self.momentum).add_(self.momentum * power.detach())
        else:
            power = self.running_power
        scale = (self.gamma / torch.sqrt(power + self.eps)).view(1, -1, 1, 1)
        return xr * scale, xi * scale


class ComplexBatchNorm2d(nn.Module):
    """
    Trabelsi et al. 2018, sec. 3.5: whiten the 2x2 real/imaginary covariance.

    Scaling real and imaginary parts independently (i.e. two ordinary
    BatchNorms) does not decorrelate them and can leave the distribution
    elliptical, which biases the network toward whichever phase the ellipse
    points at. The published fix computes V = [[Vrr, Vri], [Vri, Vii]] and
    applies V^{-1/2}, which for a 2x2 matrix has the closed form used below
    (s = sqrt(det V), t = sqrt(tr V + 2s)).

    Phase equivariance: the whitening step alone IS equivariant, because the
    symmetric inverse square root commutes with rotation --
    (R V R^T)^{-1/2} = R V^{-1/2} R^T -- so at initialisation, where Gamma is
    isotropic (gamma_rr = gamma_ii = 1/sqrt2, gamma_ri = 0) and beta = 0, this
    layer behaves like a scalar gain and the network built from it is
    phase-invariant. It stops being so as soon as Gamma learns an anisotropy or
    beta learns an offset. --self-test measures both states rather than
    assuming either.

    det V is clamped at eps^2 before the square root: for an input whose
    imaginary part is identically zero (the 'magnitude' condition at the very
    first layer) the covariance is singular and an unclamped s*t denominator
    goes to zero.
    """

    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        inv_sqrt2 = 1.0 / math.sqrt(2.0)
        self.gamma_rr = nn.Parameter(torch.full((num_features,), inv_sqrt2))
        self.gamma_ii = nn.Parameter(torch.full((num_features,), inv_sqrt2))
        self.gamma_ri = nn.Parameter(torch.zeros(num_features))
        self.beta_r = nn.Parameter(torch.zeros(num_features))
        self.beta_i = nn.Parameter(torch.zeros(num_features))
        self.register_buffer("running_mean_r", torch.zeros(num_features))
        self.register_buffer("running_mean_i", torch.zeros(num_features))
        self.register_buffer("running_Vrr", torch.full((num_features,), inv_sqrt2))
        self.register_buffer("running_Vii", torch.full((num_features,), inv_sqrt2))
        self.register_buffer("running_Vri", torch.zeros(num_features))

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        shape = (1, -1, 1, 1)
        if self.training:
            mr = xr.mean(dim=(0, 2, 3))
            mi = xi.mean(dim=(0, 2, 3))
            cr = xr - mr.view(shape)
            ci = xi - mi.view(shape)
            Vrr = (cr * cr).mean(dim=(0, 2, 3)) + self.eps
            Vii = (ci * ci).mean(dim=(0, 2, 3)) + self.eps
            Vri = (cr * ci).mean(dim=(0, 2, 3))
            with torch.no_grad():
                m = self.momentum
                self.running_mean_r.mul_(1 - m).add_(m * mr.detach())
                self.running_mean_i.mul_(1 - m).add_(m * mi.detach())
                self.running_Vrr.mul_(1 - m).add_(m * Vrr.detach())
                self.running_Vii.mul_(1 - m).add_(m * Vii.detach())
                self.running_Vri.mul_(1 - m).add_(m * Vri.detach())
        else:
            cr = xr - self.running_mean_r.view(shape)
            ci = xi - self.running_mean_i.view(shape)
            Vrr, Vii, Vri = self.running_Vrr, self.running_Vii, self.running_Vri

        det = (Vrr * Vii - Vri * Vri).clamp_min(self.eps * self.eps)
        s = torch.sqrt(det)
        t = torch.sqrt(Vrr + Vii + 2.0 * s)
        inv_st = 1.0 / (s * t)
        Wrr = ((Vii + s) * inv_st).view(shape)
        Wii = ((Vrr + s) * inv_st).view(shape)
        Wri = (-Vri * inv_st).view(shape)

        zr = Wrr * cr + Wri * ci
        zi = Wri * cr + Wii * ci
        out_r = self.gamma_rr.view(shape) * zr + self.gamma_ri.view(shape) * zi + self.beta_r.view(shape)
        out_i = self.gamma_ri.view(shape) * zr + self.gamma_ii.view(shape) * zi + self.beta_i.view(shape)
        return out_r, out_i


class ModReLU(nn.Module):
    """
    modReLU (Arjovsky, Shah & Bengio 2016): z * ReLU(|z| + b) / |z|.

    A magnitude threshold with a learnable per-channel real bias. The phase of
    every surviving unit is passed through untouched, so the layer is exactly
    equivariant to a global phase rotation. That is the property that lets the
    network treat phase as an angle rather than as two arbitrary reals.
    """

    def __init__(self, num_features: int, eps: float = 1e-6):
        super().__init__()
        self.bias = nn.Parameter(torch.full((num_features,), float(MODRELU_BIAS_INIT)))
        self.eps = eps

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        mag = torch.sqrt(xr * xr + xi * xi + self.eps)
        scale = F.relu(mag + self.bias.view(1, -1, 1, 1)) / mag
        return xr * scale, xi * scale


class CReLU(nn.Module):
    """
    CReLU (Trabelsi et al. 2018): independent ReLU on real and imaginary parts.

    Cheaper and often trains better than modReLU, but it is defined in terms of
    the real/imaginary decomposition and therefore NOT phase-equivariant: it
    carves the complex plane into the first quadrant. Offered for completeness
    and because the published complex nets mostly use it.
    """

    def forward(self, xr: torch.Tensor, xi: torch.Tensor):
        return F.relu(xr), F.relu(xi)


class ComplexBlock(nn.Module):
    """conv -> norm -> activation, twice, with the second conv doing the stride."""

    def __init__(self, in_ch: int, out_ch: int, stride: int, norm: str, act: str):
        super().__init__()
        self.conv1 = ComplexConv2d(in_ch, out_ch, 3, stride=1, padding=1, bias=False)
        self.norm1 = _complex_norm(norm, out_ch)
        self.act1 = _complex_act(act, out_ch)
        self.conv2 = ComplexConv2d(out_ch, out_ch, 3, stride=stride, padding=1, bias=False)
        self.norm2 = _complex_norm(norm, out_ch)
        self.act2 = _complex_act(act, out_ch)

    def forward(self, xr, xi):
        xr, xi = self.conv1(xr, xi)
        xr, xi = self.norm1(xr, xi)
        xr, xi = self.act1(xr, xi)
        xr, xi = self.conv2(xr, xi)
        xr, xi = self.norm2(xr, xi)
        xr, xi = self.act2(xr, xi)
        return xr, xi


def _complex_norm(kind: str, ch: int) -> nn.Module:
    if kind == "modulus":
        return ModulusBatchNorm2d(ch)
    if kind == "whiten":
        return ComplexBatchNorm2d(ch)
    raise ValueError(f"unknown complex norm {kind!r}; expected 'modulus' or 'whiten'")


def _complex_act(kind: str, ch: int) -> nn.Module:
    if kind == "modrelu":
        return ModReLU(ch)
    if kind == "crelu":
        return CReLU()
    raise ValueError(f"unknown complex activation {kind!r}; expected 'modrelu' or 'crelu'")


def real_to_complex(x: torch.Tensor, in_channels: int):
    """
    Rebuild the complex image from the real channel stack s03_train produces.

    The dataset emits, per condition (channel order fixed by
    common.phase_channels, which returns [sin, cos]):

        1 ch : [zscore(mag)]                -> (z, 0i)
        2 ch : [sin, cos]                   -> cos + i sin  = e^{i phi}
        3 ch : [zscore(mag), sin, cos]      -> z cos + i z sin = z e^{i phi}

    All three are exact re-encodings: nothing is added and nothing is dropped
    relative to what the real-valued networks receive. That is what makes the
    complex arm a controlled comparison rather than a different experiment.
    """
    if in_channels == 1:
        return x[:, 0:1], torch.zeros_like(x[:, 0:1])
    if in_channels == 2:
        return x[:, 1:2], x[:, 0:1]
    if in_channels == 3:
        mag = x[:, 0:1]
        return mag * x[:, 2:3], mag * x[:, 1:2]
    raise ValueError(f"cannot build a complex image from {in_channels} channels")


class ComplexSmallCNN(nn.Module):
    """
    Small complex-valued CNN over the complex image, with a magnitude head.

    Stem 7x7/2 then three stride-2 blocks: 224 -> 112 -> 56 -> 28 -> 14, widths
    (32, 64, 128, 256) COMPLEX channels. Kept small on purpose -- the clinical
    cohorts have 45-70 subjects, and an 11M-parameter model is already at the
    edge of memorising them; the interesting question is whether a phase-native
    inductive bias finds signal, not whether more parameters do. resnet50 is
    the capacity arm.

    Everything downstream of the last block is real: |z| = sqrt(re^2 + im^2),
    global average pool, Dropout, Linear -- the same head recipe as
    s03_model.PhaseDxResNet18, so head capacity is not a confound between arms.
    """

    def __init__(self, condition: str, dropout: float = 0.3, num_classes: int = 2,
                 widths=(32, 64, 128, 256), norm: str = "modulus", act: str = "modrelu",
                 freeze_backbone: bool = False):
        super().__init__()
        self.arch = "complex_small"
        self.condition = condition
        self.in_channels = input_channels_for(condition)
        self.num_classes = num_classes
        self.norm_kind = norm
        self.act_kind = act
        # No ImageNet weights exist for complex convolutions. Say so rather
        # than inheriting a True from the caller: the sweep's JSON is read by
        # s04/s06 and a mislabelled pretrained flag would make the complex arm
        # look like a pretrained model that failed.
        self.pretrained = False
        self.freeze_backbone = bool(freeze_backbone)

        w0, w1, w2, w3 = widths
        self.stem_conv = ComplexConv2d(1, w0, 7, stride=2, padding=3, bias=False)
        self.stem_norm = _complex_norm(norm, w0)
        self.stem_act = _complex_act(act, w0)
        self.block1 = ComplexBlock(w0, w1, stride=2, norm=norm, act=act)
        self.block2 = ComplexBlock(w1, w2, stride=2, norm=norm, act=act)
        self.block3 = ComplexBlock(w2, w3, stride=2, norm=norm, act=act)
        self.feature_dim = w3
        self.head = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(w3, num_classes))

        self._frozen_modules = []
        if self.freeze_backbone:
            # Mirrors s03_model: freeze everything but the last stage + head.
            for module in (self.stem_conv, self.stem_norm, self.stem_act,
                           self.block1, self.block2):
                for p in module.parameters():
                    p.requires_grad_(False)
                self._frozen_modules.append(module)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            for module in self._frozen_modules:
                module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xr, xi = real_to_complex(x, self.in_channels)
        xr, xi = self.stem_conv(xr, xi)
        xr, xi = self.stem_norm(xr, xi)
        xr, xi = self.stem_act(xr, xi)
        xr, xi = self.block1(xr, xi)
        xr, xi = self.block2(xr, xi)
        xr, xi = self.block3(xr, xi)
        mag = torch.sqrt(xr * xr + xi * xi + 1e-12)
        pooled = mag.mean(dim=(2, 3))
        return self.head(pooled)


# ===========================================================================
# Real-valued architectures: one generic wrapper, per-family surgery
# ===========================================================================

def _get_module(root: nn.Module, path: str) -> nn.Module:
    obj = root
    for part in path.split("."):
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    return obj


def _set_module(root: nn.Module, path: str, value: nn.Module) -> None:
    parts = path.split(".")
    obj = root
    for part in parts[:-1]:
        obj = obj[int(part)] if part.isdigit() else getattr(obj, part)
    last = parts[-1]
    if last.isdigit():
        obj[int(last)] = value
    else:
        setattr(obj, last, value)


def _head_linear(backbone: nn.Module, path: str, dropout: float, num_classes: int) -> int:
    """Replace a bare Linear head with Dropout + Linear. Returns the feature dim."""
    old = _get_module(backbone, path)
    feat = old.in_features
    _set_module(backbone, path, nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feat, num_classes)))
    return feat


def _head_resnet(backbone, dropout, num_classes):
    return _head_linear(backbone, "fc", dropout, num_classes)


def _head_densenet(backbone, dropout, num_classes):
    return _head_linear(backbone, "classifier", dropout, num_classes)


def _head_vit(backbone, dropout, num_classes):
    # heads is Sequential(head=Linear); replace the whole thing.
    feat = backbone.heads.head.in_features
    backbone.heads = nn.Sequential(nn.Dropout(p=dropout), nn.Linear(feat, num_classes))
    return feat


def _head_convnext(backbone, dropout, num_classes):
    # classifier is Sequential(LayerNorm2d, Flatten, Linear). The first two are
    # part of the pooled feature path, not the head -- keep them.
    feat = backbone.classifier[2].in_features
    backbone.classifier = nn.Sequential(
        backbone.classifier[0],
        backbone.classifier[1],
        nn.Dropout(p=dropout),
        nn.Linear(feat, num_classes),
    )
    return feat


class ArchSpec:
    """How to build, adapt, re-head and freeze one torchvision architecture."""

    def __init__(self, name, ctor_name, weights_enum, first_conv, head_fn,
                 freeze_prefixes, notes=""):
        self.name = name
        self.ctor_name = ctor_name
        self.weights_enum = weights_enum
        self.first_conv = first_conv
        self.head_fn = head_fn
        self.freeze_prefixes = tuple(freeze_prefixes)
        self.notes = notes


# Freeze specs follow s03_model's rule -- everything up to and including the
# penultimate stage is frozen, the last stage and the head stay trainable --
# translated into each family's module names.
REAL_SPECS = {
    "resnet18": ArchSpec(
        "resnet18", "resnet18", "ResNet18_Weights", "conv1", _head_resnet,
        s03_model.FROZEN_BLOCK_NAMES,
        notes="the incumbent; this entry delegates to s03_model.PhaseDxResNet18",
    ),
    "resnet50": ArchSpec(
        "resnet50", "resnet50", "ResNet50_Weights", "conv1", _head_resnet,
        s03_model.FROZEN_BLOCK_NAMES,
        notes="capacity control: same inductive bias, ~4x parameters",
    ),
    "densenet121": ArchSpec(
        "densenet121", "densenet121", "DenseNet121_Weights", "features.conv0",
        _head_densenet,
        ("features.conv0", "features.norm0", "features.denseblock1", "features.transition1",
         "features.denseblock2", "features.transition2", "features.denseblock3",
         "features.transition3"),
        notes="dense connectivity; different feature reuse at ~8M params",
    ),
    "vit_b_16": ArchSpec(
        "vit_b_16", "vit_b_16", "ViT_B_16_Weights", "conv_proj", _head_vit,
        ("conv_proj", "class_token", "encoder.pos_embedding") +
        tuple(f"encoder.layers.encoder_layer_{i}" for i in range(10)),
        notes="attention, no locality prior; requires exactly 224x224 input",
    ),
    "convnext_tiny": ArchSpec(
        "convnext_tiny", "convnext_tiny", "ConvNeXt_Tiny_Weights", "features.0.0",
        _head_convnext,
        tuple(f"features.{i}" for i in range(6)),
        notes="modern conv baseline",
    ),
}

ARCH_NAMES = ("resnet18", "resnet50", "densenet121", "vit_b_16", "convnext_tiny",
              "complex_small")


def weights_are_cached(weights) -> bool:
    """True if the checkpoint for a torchvision weights enum is already on disk."""
    if weights is None:
        return False
    try:
        fname = weights.url.rsplit("/", 1)[-1]
    except Exception:  # noqa: BLE001
        return False
    return (Path(torch.hub.get_dir()) / "checkpoints" / fname).exists()


class PhaseDxNet(nn.Module):
    """
    Generic counterpart of s03_model.PhaseDxResNet18 for the other backbones.

    Exposes exactly the attribute surface s03_train reads back out
    (condition, in_channels, freeze_backbone, pretrained, num_classes) plus
    .arch, and keeps the weights under .backbone so checkpoints from the sweep
    have the same key layout as the headline ones.
    """

    def __init__(self, spec: ArchSpec, condition: str, pretrained: bool = True,
                 freeze_backbone: bool = False, dropout: float = 0.3,
                 num_classes: int = 2, allow_download: bool = False):
        super().__init__()
        from torchvision import models

        self.arch = spec.name
        self.spec = spec
        self.condition = condition
        self.in_channels = input_channels_for(condition)
        self.freeze_backbone = bool(freeze_backbone)
        self.num_classes = num_classes
        self.weights_unavailable = False

        weights = None
        if pretrained:
            try:
                weights = getattr(models, spec.weights_enum).IMAGENET1K_V1
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not resolve %s: %s", spec.weights_enum, exc)
            if weights is not None and not weights_are_cached(weights) and not allow_download:
                # Silently downloading 300 MB mid-sweep is bad manners; silently
                # falling back to random init without saying so is worse,
                # because a random-init arm compared against pretrained ones
                # invalidates the whole comparison. Be loud, and record it.
                logger.error(
                    "PRETRAINED WEIGHTS FOR %s ARE NOT IN THE LOCAL TORCH HUB CACHE "
                    "(%s). Falling back to RANDOM INIT for this build. Re-run with "
                    "--allow-download to fetch them.", spec.name, weights.url,
                )
                weights = None
                self.weights_unavailable = True

        try:
            backbone = getattr(models, spec.ctor_name)(weights=weights)
            self.pretrained = weights is not None
        except Exception as exc:  # noqa: BLE001
            logger.error("could not load %s with weights=%s (%s); random init",
                         spec.name, weights, exc)
            backbone = getattr(models, spec.ctor_name)(weights=None)
            self.pretrained = False
            self.weights_unavailable = True

        first = _get_module(backbone, spec.first_conv)
        _set_module(backbone, spec.first_conv, adapt_conv1(first, self.in_channels))
        self.feature_dim = spec.head_fn(backbone, dropout, num_classes)
        self.backbone = backbone

        self._frozen_modules = []
        if self.freeze_backbone:
            frozen_names = set()
            for pname, p in self.backbone.named_parameters():
                if any(pname == pre or pname.startswith(pre + ".") for pre in spec.freeze_prefixes):
                    p.requires_grad_(False)
                    frozen_names.add(pname)
            if not frozen_names:
                raise RuntimeError(
                    f"--freeze-backbone matched no parameters for {spec.name}; the freeze "
                    "spec is stale relative to this torchvision version"
                )
            for mname, module in self.backbone.named_modules():
                if any(mname == pre or mname.startswith(pre + ".")
                       for pre in spec.freeze_prefixes):
                    self._frozen_modules.append(module)

    def train(self, mode: bool = True):
        """Frozen blocks stay in eval mode; see s03_model.PhaseDxResNet18.train."""
        super().train(mode)
        if mode:
            for module in self._frozen_modules:
                module.eval()
        return self

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# ===========================================================================
# The zoo's public constructor
# ===========================================================================

def build_arch(arch: str, condition: str, pretrained: bool = True,
               freeze_backbone: bool = False, dropout: float = 0.3,
               num_classes: int = 2, allow_download: bool = False,
               complex_norm: str = "modulus", complex_act: str = "modrelu",
               verbose: bool = True) -> nn.Module:
    """
    Build any architecture in the zoo, with s03_model.build_model's signature.

    resnet18 DELEGATES to s03_model.build_model. It is not reimplemented and
    not re-adapted: the incumbent arm of the sweep is the incumbent code, so
    "the sweep reproduces the published numbers" is true by construction rather
    than by careful copying. --regression proves it empirically anyway.
    """
    if arch not in ARCH_NAMES:
        raise ValueError(f"unknown arch {arch!r}; expected one of {list(ARCH_NAMES)}")

    if arch == "resnet18":
        model = s03_model.build_model(
            condition=condition, pretrained=pretrained, freeze_backbone=freeze_backbone,
            dropout=dropout, num_classes=num_classes, verbose=False,
        )
        model.arch = "resnet18"
        model.weights_unavailable = not model.pretrained and pretrained
    elif arch == "complex_small":
        if pretrained:
            logger.info("complex_small has no ImageNet weights; training from scratch "
                        "(compare it against the --scratch arm of the real-valued nets)")
        model = ComplexSmallCNN(
            condition=condition, dropout=dropout, num_classes=num_classes,
            norm=complex_norm, act=complex_act, freeze_backbone=freeze_backbone,
        )
        model.weights_unavailable = False
    else:
        model = PhaseDxNet(
            REAL_SPECS[arch], condition=condition, pretrained=pretrained,
            freeze_backbone=freeze_backbone, dropout=dropout, num_classes=num_classes,
            allow_download=allow_download,
        )

    if verbose:
        counts = count_parameters(model)
        logger.info(
            "model[%s/%s] in_ch=%d pretrained=%s frozen_backbone=%s | "
            "trainable %s / %s params (%.1f%%)",
            arch, condition, model.in_channels, model.pretrained, model.freeze_backbone,
            f"{counts['trainable']:,}", f"{counts['total']:,}",
            100.0 * counts["trainable"] / max(1, counts["total"]),
        )
    return model


def make_build_model(arch: str, allow_download: bool = False,
                     complex_norm: str = "modulus", complex_act: str = "modrelu"):
    """
    A drop-in replacement for s03_model.build_model bound to one architecture.

    The returned callable accepts exactly the arguments s03_train.run_one
    passes (condition, pretrained=, freeze_backbone=, dropout=) so the training
    code needs no knowledge that anything was swapped.
    """

    def _build(condition, pretrained=True, freeze_backbone=False, dropout=0.3,
               num_classes=2, verbose=True):
        return build_arch(
            arch, condition, pretrained=pretrained, freeze_backbone=freeze_backbone,
            dropout=dropout, num_classes=num_classes, allow_download=allow_download,
            complex_norm=complex_norm, complex_act=complex_act, verbose=verbose,
        )

    _build.arch = arch
    return _build


@contextlib.contextmanager
def architecture(arch: str, allow_download: bool = False,
                 complex_norm: str = "modulus", complex_act: str = "modrelu"):
    """
    Temporarily route s03_train.run_one's model construction through the zoo.

    run_one resolves build_model from s03_train's module globals at call time,
    so rebinding the module attribute is enough and no training code has to be
    duplicated or parameterised -- the same trick s05_controls uses for its
    dataset transforms. The swap is scoped and always restored, including on
    exception, so an aborted sweep cannot leave the headline path pointing at
    a ViT.

    architecture("resnet18") is a no-op by construction: the zoo's resnet18
    entry calls s03_model.build_model, which is what was there before.
    """
    original = s03_train.build_model
    s03_train.build_model = make_build_model(
        arch, allow_download=allow_download,
        complex_norm=complex_norm, complex_act=complex_act,
    )
    try:
        yield
    finally:
        s03_train.build_model = original


# ===========================================================================
# Capacity table
# ===========================================================================

def parameter_rows(archs=ARCH_NAMES, conditions=common.CONDITIONS, pretrained: bool = False,
                   allow_download: bool = False, freeze: bool = False) -> list:
    """
    Parameter counts per (arch, condition).

    Built with pretrained=False by default: the parameter COUNT does not depend
    on the initialisation, and this way the table can be produced with no
    network access and no 300 MB download.
    """
    rows = []
    for arch in archs:
        for condition in conditions:
            model = build_arch(arch, condition, pretrained=pretrained,
                               freeze_backbone=freeze, allow_download=allow_download,
                               verbose=False)
            counts = count_parameters(model)
            rows.append({
                "arch": arch,
                "condition": condition,
                "in_channels": model.in_channels,
                "freeze_backbone": bool(freeze),
                "params_total": counts["total"],
                "params_trainable": counts["trainable"],
                "params_frozen": counts["frozen"],
                "feature_dim": int(getattr(model, "feature_dim", -1)),
            })
            del model
    return rows


def print_parameter_table(rows: list) -> None:
    print("=" * 88)
    print("PARAMETER BUDGET  (counts are initialisation-independent)")
    print("=" * 88)
    print(f"{'arch':<16}{'condition':<12}{'in_ch':>6}{'feat':>7}"
          f"{'trainable':>14}{'total':>14}{'%':>8}")
    for r in rows:
        print(f"{r['arch']:<16}{r['condition']:<12}{r['in_channels']:>6}"
              f"{r['feature_dim']:>7}{r['params_trainable']:>14,}{r['params_total']:>14,}"
              f"{100.0 * r['params_trainable'] / max(1, r['params_total']):>7.1f}%")
    print()
    print(f"{'arch':<16}{'total params (both)':>22}{'x resnet18':>12}")
    base = next((r["params_total"] for r in rows
                 if r["arch"] == "resnet18" and r["condition"] == "both"), None)
    for arch in dict.fromkeys(r["arch"] for r in rows):
        tot = next((r["params_total"] for r in rows
                    if r["arch"] == arch and r["condition"] == "both"), None)
        if tot is None or base is None:
            continue
        print(f"{arch:<16}{tot:>22,}{tot / base:>11.2f}x")


# ===========================================================================
# Self-test
# ===========================================================================

def _phase_stack(phase: torch.Tensor, condition: str, mag: torch.Tensor) -> torch.Tensor:
    """Build the dataset's channel stack for a given phase map, on the fly."""
    if condition == "magnitude":
        return mag
    sin, cos = torch.sin(phase), torch.cos(phase)
    if condition == "phase":
        return torch.cat([sin, cos], dim=1)
    return torch.cat([mag, sin, cos], dim=1)


def self_test(allow_download: bool = False, batch: int = 2) -> int:
    """
    Instantiate every architecture at 1, 2 and 3 channels; forward; assert.

    Checks, in order:
      1. forward pass produces exactly (batch, 2) and is finite, per arch x condition
      2. the channel adaptation preserves activation magnitude for every
         real-valued backbone (the s03_model claim, re-verified per family)
      3. real_to_complex is an exact re-encoding of the dataset's channel stack
      4. the complex net is exactly invariant to a global phase rotation under
         (modulus norm + modReLU), and is NOT invariant under whitening norm
      5. --freeze-backbone freezes something in every arch, and frozen BN
         running statistics do not drift
      6. the s03_train swap installs and, crucially, restores
    """
    failures = []
    checks = 0

    def check(ok: bool, label: str, detail: str = "") -> None:
        nonlocal checks
        checks += 1
        if not ok:
            failures.append(label)
        print(f"  [{'OK  ' if ok else 'FAIL'}] {label}{('  ' + detail) if detail else ''}")

    hw = common.TARGET_HW
    print("=" * 88)
    print(f"s11_archzoo self-test   torch {torch.__version__}   input {batch}x{{1,2,3}}x"
          f"{hw[0]}x{hw[1]}")
    print("=" * 88)

    print("\n1. FORWARD PASS: every architecture at 1, 2 and 3 channels")
    for arch in ARCH_NAMES:
        for condition in common.CONDITIONS:
            in_ch = input_channels_for(condition)
            torch.manual_seed(0)
            model = build_arch(arch, condition, pretrained=False, verbose=False).eval()
            x = torch.randn(batch, in_ch, *hw)
            with torch.no_grad():
                y = model(x)
            counts = count_parameters(model)
            ok = (tuple(y.shape) == (batch, 2)) and bool(torch.isfinite(y).all())
            check(ok, f"{arch:<14} {condition:<10} {tuple(x.shape)} -> {tuple(y.shape)}",
                  f"finite={bool(torch.isfinite(y).all())} params={counts['total']:,}")
            del model

    print("\n2. CHANNEL ADAPTATION preserves activation magnitude (per backbone family)")
    from torchvision import models as tvm
    for arch, spec in REAL_SPECS.items():
        ref_backbone = getattr(tvm, spec.ctor_name)(weights=None)
        ref = _get_module(ref_backbone, spec.first_conv)
        torch.manual_seed(0)
        gray = torch.randn(2, 1, 64, 64)
        with torch.no_grad():
            baseline = ref(gray.repeat(1, 3, 1, 1))
            worst = 0.0
            for condition in common.CONDITIONS:
                in_ch = input_channels_for(condition)
                out = adapt_conv1(ref, in_ch)(gray.repeat(1, in_ch, 1, 1))
                worst = max(worst, (out - baseline).abs().max().item())
        check(worst < 1e-4, f"{arch:<14} first conv = {spec.first_conv}",
              f"max|pre-activation diff| = {worst:.3e}")
        del ref_backbone

    print("\n3. real_to_complex is an exact re-encoding of the dataset's channels")
    torch.manual_seed(1)
    mag = torch.rand(2, 1, 8, 8) * 2.0
    phase = (torch.rand(2, 1, 8, 8) * 2.0 - 1.0) * math.pi
    for condition, expect in (("magnitude", "z + 0i"), ("phase", "e^{i phi}"),
                              ("both", "z e^{i phi}")):
        x = _phase_stack(phase, condition, mag)
        re, im = real_to_complex(x, input_channels_for(condition))
        if condition == "magnitude":
            want_re, want_im = mag, torch.zeros_like(mag)
        elif condition == "phase":
            want_re, want_im = torch.cos(phase), torch.sin(phase)
        else:
            want_re, want_im = mag * torch.cos(phase), mag * torch.sin(phase)
        err = max((re - want_re).abs().max().item(), (im - want_im).abs().max().item())
        check(err == 0.0, f"{condition:<10} -> {expect}", f"max|diff| = {err:.3e}")

    print("\n4. GLOBAL PHASE INVARIANCE of the complex net (the phase-native claim)")
    theta = 0.7231
    # Full-size inputs, not the 8x8 tensors used above. At 8x8 the stem and
    # three stride-2 blocks collapse the map to 1x1 and the network becomes
    # structurally blind -- its output does not change when the phase MAP
    # changes -- at which point every invariance check below passes vacuously.
    # That is a real trap and it caught this test during development, so each
    # configuration is first required to be phase-SENSITIVE (it must respond to
    # a rearranged phase map) before it is allowed to be phase-INVARIANT.
    torch.manual_seed(2)
    mag_f = torch.rand(2, 1, *hw) * 2.0
    phase_f = (torch.rand(2, 1, *hw) * 2.0 - 1.0) * math.pi

    def _drift(model, x0, x1, train_mode=False) -> float:
        model.train(train_mode)
        with torch.no_grad():
            # Same dropout mask on both sides: we are measuring the network's
            # response to a phase rotation, not the RNG.
            torch.manual_seed(5)
            y0 = model(x0)
            torch.manual_seed(5)
            y1 = model(x1)
        return (y0 - y1).abs().max().item()

    for condition in ("phase", "both"):
        x = _phase_stack(phase_f, condition, mag_f)
        x_rot = _phase_stack(phase_f + theta, condition, mag_f)
        # Same phase values, rearranged in space: a global rotation must not
        # move the output, but this must.
        x_struct = _phase_stack(phase_f.flip(-1), condition, mag_f)

        # (a) the claim: default configuration is exactly phase-invariant
        torch.manual_seed(0)
        m = build_arch("complex_small", condition, pretrained=False,
                       complex_norm="modulus", complex_act="modrelu", verbose=False)
        d_struct = _drift(m, x, x_struct, train_mode=False)
        check(d_struct > 1e-3, f"modulus+modrelu {condition:<6} responds to phase "
                               f"STRUCTURE (non-vacuity)", f"max|drift| = {d_struct:.3e}")
        d_eval = _drift(m, x, x_rot, train_mode=False)
        d_train = _drift(m, x, x_rot, train_mode=True)
        check(d_eval < 1e-4, f"modulus+modrelu {condition:<6} f(e^i0.72 z) == f(z)  [eval]",
              f"max|drift| = {d_eval:.3e}")
        check(d_train < 1e-4, f"modulus+modrelu {condition:<6} holds on batch stats [train]",
              f"max|drift| = {d_train:.3e}")
        del m

        # (b) negative control: CReLU is defined on re/im, so it cannot be
        #     equivariant. If this ever passes, check (a) is vacuous.
        torch.manual_seed(0)
        m = build_arch("complex_small", condition, pretrained=False,
                       complex_norm="modulus", complex_act="crelu", verbose=False)
        d_crelu = _drift(m, x, x_rot, train_mode=False)
        check(d_crelu > 1e-3, f"modulus+crelu   {condition:<6} NOT invariant (control)",
              f"max|drift| = {d_crelu:.3e}")
        del m

        # (c) whitening: equivariant at init because V^{-1/2} commutes with
        #     rotation and Gamma starts isotropic; not once Gamma/beta train.
        torch.manual_seed(0)
        m = build_arch("complex_small", condition, pretrained=False,
                       complex_norm="whiten", complex_act="modrelu", verbose=False)
        d_struct_w = _drift(m, x, x_struct, train_mode=False)
        check(d_struct_w > 1e-3, f"whiten+modrelu  {condition:<6} responds to phase "
                                 f"STRUCTURE (non-vacuity)", f"max|drift| = {d_struct_w:.3e}")
        d_init = _drift(m, x, x_rot, train_mode=False)
        with torch.no_grad():
            for mod in m.modules():
                if isinstance(mod, ComplexBatchNorm2d):
                    mod.gamma_ri.add_(0.3)   # anisotropic Gamma
                    mod.beta_i.add_(0.1)     # complex offset
        d_trained = _drift(m, x, x_rot, train_mode=False)
        check(d_init < 1e-4, f"whiten+modrelu  {condition:<6} invariant at init "
                             f"(V^-1/2 commutes with R)", f"max|drift| = {d_init:.3e}")
        check(d_trained > 1e-3, f"whiten+modrelu  {condition:<6} invariance LOST once "
                                f"Gamma/beta leave isotropy", f"max|drift| = {d_trained:.3e}")
        del m

    print("\n5. --freeze-backbone actually freezes, and frozen norm stats do not drift")
    for arch in ARCH_NAMES:
        torch.manual_seed(0)
        model = build_arch(arch, "both", pretrained=False, freeze_backbone=True,
                           verbose=False)
        counts = count_parameters(model)
        check(counts["frozen"] > 0 and counts["trainable"] > 0,
              f"{arch:<14} frozen/trainable split",
              f"{counts['trainable']:,} trainable / {counts['frozen']:,} frozen")
        model.train()
        buffers_before = {k: v.clone() for k, v in model.named_buffers()
                          if v.dtype.is_floating_point}
        with torch.no_grad():
            for _ in range(2):
                model(torch.randn(4, 3, *hw))
        drifted_frozen, n_checked = [], 0
        frozen_ids = {id(m) for m in model._frozen_modules}
        for mod_name, module in model.named_modules():
            if id(module) not in frozen_ids:
                continue
            for buf_name, buf in module.named_buffers(prefix=mod_name):
                if buf_name in buffers_before and buf.dtype.is_floating_point:
                    n_checked += 1
                    if not torch.equal(buf, buffers_before[buf_name]):
                        drifted_frozen.append(buf_name)
        # vit_b_16 and convnext_tiny normalise with LayerNorm, which carries no
        # running statistics, so zero buffers checked is correct for them and a
        # bug for anything with BatchNorm. n_checked is printed so a silently
        # vacuous pass is visible rather than indistinguishable from a real one.
        expect_buffers = arch not in ("vit_b_16", "convnext_tiny")
        check(not drifted_frozen and (n_checked > 0) == expect_buffers,
              f"{arch:<14} frozen buffers held",
              f"{n_checked} float buffers checked, {len(drifted_frozen)} drifted")
        del model

    print("\n6. s03_train swap installs and restores")
    original = s03_train.build_model
    with architecture("convnext_tiny"):
        swapped = s03_train.build_model
        check(getattr(swapped, "arch", None) == "convnext_tiny",
              "inside context: s03_train.build_model is the zoo's builder")
    check(s03_train.build_model is original,
          "after context: s03_train.build_model restored to s03_model.build_model")
    try:
        with architecture("resnet50"):
            raise RuntimeError("simulated failure")
    except RuntimeError:
        pass
    check(s03_train.build_model is original, "restored even when the body raises")

    print()
    print("=" * 88)
    print(f"s11_archzoo self-test: {checks - len(failures)}/{checks} checks passed")
    if failures:
        for f in failures:
            print(f"  FAILED: {f}")
    print("=" * 88)
    return 1 if failures else 0


# ===========================================================================
# Regression check: resnet18 through the zoo == resnet18 through s03_model
# ===========================================================================

def _state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for k, v in sorted(model.state_dict().items()):
        h.update(k.encode())
        h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()


def regression_check(seed: int = 20240517, pretrained: bool = True, batch: int = 4) -> int:
    """
    Prove the zoo's resnet18 is BIT-IDENTICAL to s03_model's, not merely close.

    Three things must match, and all three matter:

      state_dict   every tensor bitwise equal (torch.equal, not allclose). If
                   the weights differ at all, the sweep's resnet18 arm is a
                   different model and cannot be compared to the published
                   numbers.
      forward      logits bitwise equal on a fixed input in eval mode.
      RNG state    the global torch RNG must be in the same state AFTER
                   construction. run_one calls set_seed(seed) and then builds
                   the model; if the zoo consumed a different number of random
                   draws, every subsequent stochastic decision (dropout masks,
                   sampler order) would diverge even from identical weights,
                   and training would not reproduce.

    If any of these fails, the sweep is NOT comparable to the existing results
    and this function says so in those words.
    """
    print("=" * 88)
    print("REGRESSION CHECK: zoo resnet18 vs s03_model.PhaseDxResNet18")
    print(f"seed={seed}  pretrained={pretrained}  torch {torch.__version__}")
    print("=" * 88)

    failures = []
    for condition in common.CONDITIONS:
        in_ch = input_channels_for(condition)
        x = torch.randn(batch, in_ch, *common.TARGET_HW,
                        generator=torch.Generator().manual_seed(99))

        s03_train.set_seed(seed)
        ref = s03_model.build_model(condition, pretrained=pretrained, verbose=False)
        rng_ref = torch.get_rng_state()

        s03_train.set_seed(seed)
        zoo = build_arch("resnet18", condition, pretrained=pretrained, verbose=False)
        rng_zoo = torch.get_rng_state()

        # ...and through the actual production entry point, under the swap.
        s03_train.set_seed(seed)
        with architecture("resnet18"):
            via_train = s03_train.build_model(condition, pretrained=pretrained,
                                              freeze_backbone=False, dropout=0.3)
        rng_train = torch.get_rng_state()

        ref.eval(), zoo.eval(), via_train.eval()
        with torch.no_grad():
            y_ref, y_zoo, y_train = ref(x), zoo(x), via_train(x)

        keys_ok = sorted(ref.state_dict()) == sorted(zoo.state_dict())
        weights_ok = keys_ok and all(
            torch.equal(v, zoo.state_dict()[k]) for k, v in ref.state_dict().items()
        )
        swap_ok = keys_ok and all(
            torch.equal(v, via_train.state_dict()[k]) for k, v in ref.state_dict().items()
        )
        fwd_ok = torch.equal(y_ref, y_zoo) and torch.equal(y_ref, y_train)
        rng_ok = torch.equal(rng_ref, rng_zoo) and torch.equal(rng_ref, rng_train)
        h_ref, h_zoo = _state_hash(ref), _state_hash(zoo)

        print(f"\ncondition = {condition}  (in_ch={in_ch}, "
              f"{count_parameters(ref)['total']:,} params, pretrained={ref.pretrained})")
        print(f"  state_dict keys identical      : {keys_ok}")
        print(f"  every tensor torch.equal       : {weights_ok}")
        print(f"  same via s03_train swap        : {swap_ok}")
        print(f"  sha256(state_dict) direct      : {h_ref}")
        print(f"  sha256(state_dict) via zoo     : {h_zoo}")
        print(f"  logits bitwise equal           : {fwd_ok}"
              f"   max|diff| = {(y_ref - y_zoo).abs().max().item():.1e}")
        print(f"  logits[0] = {[round(float(v), 6) for v in y_ref[0]]}")
        print(f"  torch RNG state after build    : {rng_ok}")

        if not (keys_ok and weights_ok and swap_ok and fwd_ok and rng_ok and h_ref == h_zoo):
            failures.append(condition)

    print()
    print("=" * 88)
    if failures:
        print("REGRESSION CHECK FAILED for: " + ", ".join(failures))
        print("The zoo's resnet18 is NOT bit-identical to the published implementation.")
        print("THE SWEEP IS THEREFORE NOT COMPARABLE TO THE EXISTING RESULTS -- any")
        print("difference between architectures in the sweep is confounded with a")
        print("difference in the code path, and no architecture claim may be made")
        print("until this passes.")
    else:
        print("REGRESSION CHECK PASSED for all three conditions.")
        print("The zoo's resnet18 arm is the published implementation (it delegates to")
        print("s03_model.build_model), so sweep results are directly comparable to the")
        print("existing magnitude/phase/both numbers.")
    print("=" * 88)
    return 1 if failures else 0


# ===========================================================================
# Sweep: run the zoo through s03_train.run_one
# ===========================================================================

def build_train_args(args) -> argparse.Namespace:
    """
    Start from s03_train's own defaults, then apply only what this CLI set.

    Same reasoning as s05_controls.build_train_args: deriving the namespace
    from s03_train.parse_args([]) means a sweep cannot silently drift away from
    the headline runs' hyperparameters when s03_train changes.
    """
    base = s03_train.parse_args([])
    owned_here = {"cohort", "conditions", "seeds", "dry_run", "results_dir", "ckpt_dir",
                  "archs", "arch", "scratch"}
    for k, v in vars(args).items():
        if v is not None and hasattr(base, k) and k not in owned_here:
            setattr(base, k, v)
    base.dry_run = False
    base.no_pretrained = bool(args.scratch)
    return base


def _stem(cohort, arch, condition, seed, init: str) -> str:
    return f"{cohort}__{arch}__{init}__{condition}__seed{seed}"


def run_tagged(cohort, arch, condition, seed, index, device, train_args, *,
               results_root: Path, h5_path=None, arrays=None, scratch=False,
               allow_download=False, complex_norm="modulus", complex_act="modrelu",
               keep_checkpoints=False) -> dict:
    """
    One sweep run: s03_train.run_one under the architecture swap, then refiled.

    The payload is byte-for-byte s03_train's schema with two corrections and one
    addition:

        model.arch      overwritten with the architecture that actually ran.
                        run_one hard-codes "resnet18"; leaving that in place
                        would produce a results tree in which every JSON claims
                        to be a ResNet-18, which is the single worst failure
                        mode available to this module.
        model.pretrained/params_* already come from the live model, so they are
                        correct without intervention.
        archzoo         the sweep's own bookkeeping.
    """
    run_args = copy.deepcopy(train_args)
    run_args.arch = arch                  # lands in the checkpoint's saved args
    run_args.no_pretrained = bool(scratch)

    scratch_dir = Path(results_root) / "_scratch"
    if scratch_dir.exists():
        shutil.rmtree(scratch_dir)
    scratch_dir.mkdir(parents=True, exist_ok=True)
    run_args.results_dir = str(scratch_dir)
    run_args.ckpt_dir = str(scratch_dir)

    t0 = time.time()
    with architecture(arch, allow_download=allow_download,
                      complex_norm=complex_norm, complex_act=complex_act):
        res = s03_train.run_one(cohort, condition, seed, index, device, run_args,
                                h5_path=h5_path, arrays=arrays)

    payload = json.loads(Path(res["results_json"]).read_text())
    payload["model"]["arch"] = arch
    # The init tag comes from what the model ACTUALLY got, not from the flag
    # that was requested. complex_small has no ImageNet weights, and a
    # torchvision checkpoint can be missing from the hub cache, so a tag taken
    # from --scratch alone would file a randomly-initialised run under
    # "imagenet" and make an unpretrained arm look like a pretrained failure.
    init = "imagenet" if payload["model"]["pretrained"] else "scratch"
    payload["archzoo"] = {
        "schema": SCHEMA,
        "arch": arch,
        "init": init,
        "init_requested": "scratch" if scratch else "imagenet",
        "split_col": getattr(train_args, "split_col", "official_split"),
        "complex_norm": complex_norm if arch == "complex_small" else None,
        "complex_act": complex_act if arch == "complex_small" else None,
        "wall_seconds": time.time() - t0,
    }

    stem = _stem(cohort, arch, condition, seed, init)
    results_root = Path(results_root)
    results_root.mkdir(parents=True, exist_ok=True)
    if keep_checkpoints:
        dst = results_root / "checkpoints" / f"{stem}.pt"
        dst.parent.mkdir(parents=True, exist_ok=True)
        src = Path(payload["checkpoint"])
        if src.exists():
            shutil.move(str(src), dst)
            payload["checkpoint"] = str(dst)
    else:
        payload["checkpoint"] = None

    out_path = results_root / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    shutil.rmtree(scratch_dir, ignore_errors=True)

    logger.info("  -> %s  test_auc=%.4f", out_path.name, res["test_auc"])
    return {
        "path": str(out_path), "arch": arch, "condition": condition, "seed": seed,
        "init": init,
        "test_auc": res["test_auc"], "test_ap": res["test_ap"],
        "best_epoch": res["best_epoch"],
        "params_total": payload["model"]["params_total"],
        "params_trainable": payload["model"]["params_trainable"],
        "pretrained": payload["model"]["pretrained"],
        "wall_seconds": payload["archzoo"]["wall_seconds"],
    }


def run_sweep(args) -> int:
    archs = (list(ARCH_NAMES) if args.archs == "all"
             else [a.strip() for a in args.archs.split(",") if a.strip()])
    for a in archs:
        if a not in ARCH_NAMES:
            print(f"ERROR: unknown arch {a!r}; expected any of {list(ARCH_NAMES)}",
                  file=sys.stderr)
            return 2
    conditions = (list(common.CONDITIONS) if args.conditions == "all"
                  else [c.strip() for c in args.conditions.split(",") if c.strip()])
    for c in conditions:
        input_channels_for(c)
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]

    train_args = build_train_args(args)
    device = s03_train.pick_device(args.device)
    results_root = Path(args.results_dir or (common.OUT_ROOT / RESULTS_SUBDIR / "results"))

    if args.smoke:
        cohort = "dryrun"
        train_args.epochs = min(train_args.epochs, 2)
        train_args.warmup_epochs = 1
        train_args.batch_size = min(train_args.batch_size, 8)
        train_args.workers = 0
        seeds = seeds[:1]
        # n_patients must stay >= 16: make_dry_run_cache assigns splits by
        # patient index (p<12 train, p<15 val, else test), so a smaller cohort
        # silently produces an empty test split. hw is forced to TARGET_HW
        # because vit_b_16 accepts exactly 224x224 and the default 64x64 dry-run
        # cache would also collapse complex_small's feature map to 1x1.
        arrays, raw_index = s03_train.make_dry_run_cache(
            n_patients=18, slices_per_patient=4, hw=common.TARGET_HW,
        )
        index = s03_train.prepare_index(raw_index, train_args.val_frac,
                                        train_args.val_split_seed)
        h5_path = None
        results_root = Path(args.results_dir or (common.OUT_ROOT / RESULTS_SUBDIR / "smoke"))
        print("=" * 88)
        print("SMOKE: fabricated 224x224 cache, no drive access. This proves the "
              "PLUMBING, not the science.")
        print("=" * 88)
    else:
        if not args.cohort:
            print("ERROR: --cohort is required (or use --smoke)", file=sys.stderr)
            return 2
        cohort = args.cohort
        cache_dir = Path(train_args.cache_dir)
        h5_path = cache_dir / f"{cohort}.h5"
        idx_path = cache_dir / f"{cohort}_index.csv"
        for pth in (h5_path, idx_path):
            if not pth.exists():
                print(f"ERROR: missing {pth}. Run stage 2 first.", file=sys.stderr)
                return 2
        raw_index = pd.read_csv(idx_path)
        index = s03_train.prepare_index(raw_index, train_args.val_frac,
                                        train_args.val_split_seed,
                                        split_col=train_args.split_col)
        arrays = None

    rows = []
    for arch in archs:
        for condition in conditions:
            for seed in seeds:
                logger.info("=" * 78)
                logger.info("ARCHZOO %s / %s / seed %d  (init=%s)", arch, condition, seed,
                            "scratch" if args.scratch else "imagenet")
                logger.info("=" * 78)
                try:
                    rows.append(run_tagged(
                        cohort, arch, condition, seed, index, device, train_args,
                        results_root=results_root, h5_path=h5_path, arrays=arrays,
                        scratch=args.scratch, allow_download=args.allow_download,
                        complex_norm=args.complex_norm, complex_act=args.complex_act,
                        keep_checkpoints=args.keep_checkpoints,
                    ))
                except Exception as exc:  # noqa: BLE001
                    # One architecture dying must not cost the rest of the sweep,
                    # but a missing arm must be visible as a FAILURE rather than
                    # as an absence.
                    logger.error("ARCH %s / %s / seed %d FAILED: %s", arch, condition,
                                 seed, exc)
                    rows.append({"arch": arch, "condition": condition, "seed": seed,
                                 "init": "scratch" if args.scratch else "imagenet",
                                 "test_auc": float("nan"), "test_ap": float("nan"),
                                 "error": str(exc)})

    results_root.mkdir(parents=True, exist_ok=True)
    summary_path = results_root / f"archzoo_summary_{cohort}.json"
    summary_path.write_text(json.dumps({
        "schema": SCHEMA, "cohort": cohort,
        "split_col": getattr(train_args, "split_col", "official_split"),
        "region": train_args.region, "epochs": train_args.epochs, "lr": train_args.lr,
        "init": "scratch" if args.scratch else "imagenet",
        "device": str(device), "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "runs": rows,
    }, indent=2))

    print()
    print("=" * 88)
    print(f"ARCHZOO SUMMARY  cohort={cohort}  init="
          f"{'scratch' if args.scratch else 'imagenet'}  device={device}")
    print("=" * 88)
    print(f"{'arch':<16}{'condition':<12}{'seed':>6}{'init':>10}{'test_auc':>10}"
          f"{'test_ap':>10}{'params':>14}{'sec':>8}")
    for r in rows:
        print(f"{r['arch']:<16}{r['condition']:<12}{r['seed']:>6}{r.get('init', '?'):>10}"
              f"{r.get('test_auc', float('nan')):>10.4f}"
              f"{r.get('test_ap', float('nan')):>10.4f}"
              f"{r.get('params_total', 0):>14,}{r.get('wall_seconds', 0.0):>8.1f}")
    print()
    if any(r.get("init") == "scratch" for r in rows) and not args.scratch:
        print("NOTE: some arms report init=scratch despite a pretrained sweep --")
        print("complex_small has no ImageNet weights, and any torchvision checkpoint")
        print("missing from the local hub cache falls back to random init. Compare")
        print("those arms against a --scratch sweep, not against the pretrained ones.")
        print()
    print("A single-fold test AUC here is NOT the headline estimate. Pool the JSONs")
    print("through s04_stats.py for the out-of-fold, subject-clustered interval, and")
    print("remember that the maximum over architectures is biased upward.")
    print(f"\nwrote {summary_path}")
    return 0


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PhaseDx stage 11: architecture zoo over the existing training path",
    )
    p.add_argument("--self-test", action="store_true",
                   help="instantiate every arch at 1/2/3 channels, forward, assert")
    p.add_argument("--regression", action="store_true",
                   help="prove the zoo's resnet18 is bit-identical to s03_model's")
    p.add_argument("--params", action="store_true", help="print the parameter table and exit")
    p.add_argument("--smoke", action="store_true",
                   help="run the full training path on a fabricated cache (no drive)")

    p.add_argument("--cohort", choices=list(common.COHORTS))
    p.add_argument("--archs", default="all",
                   help=f"comma-separated subset of {','.join(ARCH_NAMES)} (or 'all')")
    p.add_argument("--conditions", default="all")
    p.add_argument("--seeds", default="42,123")
    p.add_argument("--split-col", default=None,
                   help="official_split (default) or cv0_split..cv4_split")
    p.add_argument("--region", default=None, choices=("full", "body", "background"))
    p.add_argument("--scratch", action="store_true",
                   help="no ImageNet pretraining; pretrained RGB features are "
                        "themselves a confound worth ruling out")
    p.add_argument("--freeze-backbone", action="store_true", default=None)
    p.add_argument("--allow-download", action="store_true",
                   help="permit torchvision to fetch weights that are not already "
                        "in the local hub cache (vit_b_16 ~330MB, convnext_tiny ~110MB)")
    p.add_argument("--complex-norm", default="modulus", choices=("modulus", "whiten"),
                   help="modulus keeps global-phase equivariance; whiten is Trabelsi's")
    p.add_argument("--complex-act", default="modrelu", choices=("modrelu", "crelu"))

    p.add_argument("--epochs", type=int, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--cache-dir", default=None)
    p.add_argument("--results-dir", default=None)
    p.add_argument("--keep-checkpoints", action="store_true")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    rc = 0
    did_something = False
    if args.params:
        print_parameter_table(parameter_rows(allow_download=args.allow_download))
        did_something = True
    if args.self_test:
        rc |= self_test(allow_download=args.allow_download)
        did_something = True
    if args.regression:
        rc |= regression_check()
        did_something = True
    if did_something and not (args.cohort or args.smoke):
        return rc
    if rc:
        print("refusing to sweep: a self-test or regression check failed", file=sys.stderr)
        return rc
    return run_sweep(args)


if __name__ == "__main__":
    raise SystemExit(main())
