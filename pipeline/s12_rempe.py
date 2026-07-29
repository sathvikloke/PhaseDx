"""
s12_rempe.py
------------
Stage 12: a faithful-as-documented reproduction of the protocol of

    Moritz Rempe, Fabian Hoerst, Constantin Seibold, Boris Hadaschik, Marco
    Schlimbach, Jan Egger, Kevin Kroeninger, Felix Breuer, Martin Blaimer,
    Jens Kleesiek. "Tumor likelihood estimation on MRI prostate data by
    utilizing k-Space information." arXiv:2407.06165v2 [cs.CV], 14 Apr 2025.
    Code: https://github.com/TIO-IKIM/tumor-prediction-on-undersampled-MRI-kSpace

and then that protocol's exposure to the PhaseDx falsification suite.

This module exists because the rest of the study establishes a null ("we find
no phase signal") which is a statement about OUR pipeline. The stronger and
more useful claim is a statement about the published positive result. To be
allowed to make any such claim we have to (a) implement their protocol as
documented rather than a strawman of it, (b) say exactly which parts of it we
can and cannot instantiate on the data we hold, and (c) separate "we could not
reproduce it" from "we reproduced it and it fails controls". Those are
different claims and this module keeps them apart.

=============================================================================
                     PART 1 -- THE PROTOCOL, AS DOCUMENTED
=============================================================================
Every line below is sourced. [P Sx] = paper, section x. [C <file>] = their
published code at the commit fetched 2026-07-28. [UNSPECIFIED] marks a choice
the paper does not pin down; those are our degrees of freedom and each one is
declared at the point where we make it (see PART 2).

--- 1.1 Dataset ------------------------------------------------------------
  Source        fastMRI Prostate (Tibrewala et al., arXiv:2304.09254).  [P IIIC]
  Modality      DWI raw k-space ONLY. T2 is present in the dataset but they
                state "we only work on the DWI data".                   [P IIIC]
  Size          312 patients, 9508 slices (their own label file, fetched from
                their repo, contains 9490 slice rows / 312 patients).
                                                       [P Abstract, C dwi_slice_level_labels.csv]
  Acquisition   2 x 3T MAGNETOM Vida; 14-30 coils; 24-38 slices;
                2.0 x 2.0 mm in-plane; DWI matrix 100 x 100; FOV 200 mm. [P IIIC]
  Diffusion     3 directions x 4 averages at b=50; 3 x 12 at b=1000; 2 averages
                at b=0. b=1500 and ADC are NOT in the raw data.          [P IIIC]

--- 1.2 Label --------------------------------------------------------------
  PI-RADS, assigned PER SLICE by a radiologist on the fully reconstructed
  magnitude data. Binarised as PI-RADS <= 2 -> 0, PI-RADS > 2 -> 1.      [P IIIC]
  Their code: `y = csv["PIRADS"][idx] - 1; y = int(y > 1)`, i.e. PI-RADS > 2.
                                                        [C datasets/dwi_dataset.py]

  VERIFIED AGAINST OUR CACHE. We merged their published per-slice label file
  onto our prostate_dwi index on (file, slice) with their 1-based slice
  numbering mapped to our 0-based:
      1359 / 1359 of our rows matched
      PI-RADS agreement            1.0000
      train/val/test split agreement 1.0000
  Our labels and our official split ARE their labels and their split. This is
  the one part of the reproduction that carries no uncertainty at all.

--- 1.3 Split --------------------------------------------------------------
  "The authors of the FastMRI Prostate dataset give a predefined 70% - 15% -
  15% datasplit for training, validation and test dataset, which was also used
  in this work."                                                        [P IIIC]
  Their label file resolves this to 218 / 48 / 46 PATIENTS and
  6637 / 1458 / 1395 slices.               [C dwi_slice_level_labels.csv]

  NOTE, AND THIS MATTERS FOR HOW WE ARE ALLOWED TO CRITICISE THEM: their split
  is PATIENT-LEVEL. No patient appears in two splits. There is no patient
  leakage in their protocol and we must not imply there is. The statistical
  problem lies elsewhere -- see 1.6.

--- 1.4 Preprocessing / input construction ---------------------------------
  raw k-space [Average, Coil, Depth, Width, Height]
    -> regridding (trapezoidal)                                     [P IIID]
    -> sum over ALL averages into one channel ("instead of splitting the DWI
       data into b50, b1000 and adc channel ... we sum all averages")   [P IIID]
    -> coil compression: PCA in k-space, FIRST principal component only.
       Explicitly NOT RSS and NOT sensitivity maps. Alternative arm: GRAPPA
       then PCA. Both arms stay complex and stay in k-space.             [P IIID]
    -> per-slice, in their Dataset, in this exact order:      [C dwi_dataset.py]
         1. train only: horizontal flip, p=0.5, applied to the K-SPACE tensor
         2. train only: undersample(x, random.randint(0, 8))
            eval:       undersample(x, sampling_factor) if one is given
            undersample keeps every factor-th line outward from the midline
            and zeroes the rest; factor <= 1 is a no-op
         3. center_crop(x, (224, 224)) -- performed IN K-SPACE
         4. channels:
              "magnitude"        -> [ |ifft(k)| ]                    1 channel
              "magnitude_phase"  -> [ |ifft(k)|, angle(ifft(k)) ]    2 channels
              "magnitude_kspace" -> [ |ifft(k)|, k.real, k.imag ]    3 channels
            angle() is the RAW WRAPPED phase on [-pi, pi]. Not sin/cos.
         5. per-channel min-max rescale to [0, 1]
         6. per-channel z-score
  The paper says normalisation is "batch wise and over each channel
  separately" [P IVA]; the code does it per sample per channel. We follow the
  code and say so.

--- 1.5 Network and training ----------------------------------------------
  Architecture  ConvNeXt, "roughly 3 million parameters", timm, ImageNet
                pretrained.                                            [P IIID]
                Code pins this to `convnext_atto`. We measured 3.37 M
                parameters, consistent with "roughly 3 million".        [C model.py]
  Optimiser     Adam, lr 1e-4. Paper adds beta 0.99 and eps 1e-8 [P IVA]; the
                code uses `optim.Adam(params, lr=self.lr)`, i.e. library
                defaults.                                              [C train_dwi.py]
  Schedule      cosine annealing [P IVA]; code CosineAnnealingLR(T_max=3).
  Loss          CrossEntropy with class weight 17:1.        [P IVA, C train_dwi.py]
  Early stop    Paper: patience 10 on validation loss [P IVA]. Code: patience
                50 on validation AUROC, maximising.                    [C train_dwi.py]
  Selection     Checkpoint written whenever validation AUROC improves; that
                checkpoint is what gets tested.                        [C train_dwi.py]
  Augmentation  random horizontal flip; simulated undersampling up to x8. [P IVA]
  Batch size    Paper: 128 [P IVA]. Config file in the repo: 1.  [C configs/train_dwi.yaml]
  Epochs        argparse default 100.                                  [C train_dwi.py]

--- 1.6 Evaluation ---------------------------------------------------------
  "the ConvNeXts then predict if the slice should be classified with a
  Pi-RADS score larger than two"                                    [P Fig. 1]
  The prediction unit, the label unit and therefore the AUROC unit are all the
  SLICE. There is no patient-level aggregation anywhere in the paper or the
  code.
  "To gather confidence intervals, the bootstrapping method with 1000
  iterations has been applied on the test dataset."                     [P IVB]
  [UNSPECIFIED] the bootstrap resampling unit. Nothing in the paper or code
  indicates clustering by patient, and the reported half-widths (+-1.3 to
  +-3.6 points on 1395 slices) are consistent with resampling SLICES. We
  therefore reproduce their interval as an unclustered slice bootstrap, and
  report a subject-clustered interval beside it rather than instead of it.

--- 1.7 The numbers they report --------------------------------------------
  Table I -- "gold standard": GRAPPA-reconstructed, coil-combined ADC +
  Trace50 + Trace1000 magnitude maps in the image domain, optionally with
  k-space fed to a SECOND network whose logits are averaged with the first.
      Image domain (ADC, Trace50, Trace1000)   x0  85.7 +- 1.6
                                               x2  85.1 +- 1.9
      + k-Space                                x0  84.0 +- 1.9
                                               x2  86.1 +- 1.8   <-- the headline
      Image domain (Trace50, Trace1000)        x0  83.5 +- 1.6
      + k-Space                                x0  85.3 +- 1.5

  Table II -- the PCA / GRAPPA pipeline, which is the arm whose INPUTS we can
  actually instantiate, at their native x2 acceleration:
                              GRAPPA        PCA
      Magnitude (i)         80.7 +- 1.9   81.3 +- 2.2
      + Phase (i)           80.7 +- 1.8   80.9 +- 2.1
      + Real & Imaginary(k) 78.3 +- 2.1   81.1 +- 1.9

  READ TABLE II AGAIN. At their native sampling rate the phase model does not
  beat the magnitude model in either column (80.7 vs 80.7; 80.9 vs 81.3), and
  the k-space model loses in the GRAPPA column. And in Table I the headline
  86.1 sits against a magnitude-only 85.7. The paper's own claim of a phase /
  k-space benefit is explicitly a claim about HIGH SIMULATED UNDERSAMPLING
  (x16 and beyond), where the magnitude model degrades faster. Anyone citing
  "86.1% AUC for phase-informed prostate classification" is citing a number
  that is (a) mostly magnitude, three image-domain contrast maps deep, and
  (b) not separated from its magnitude-only control by its own error bars.
  That is a reading of the published table, not a reproduction result, and it
  is the first thing this module reports.

=============================================================================
             PART 2 -- WHAT WE CAN AND CANNOT DO ON THE DATA WE HOLD
=============================================================================
DECLARED LIMITATION 1 -- COHORT.
  They used 312 patients / 9490 labelled slices. Our validated prostate_dwi
  cache holds 45 patients / 1359 slices, 14.4% of their patients. Under their
  own official split that leaves us 33 train / 8 val / 4 test patients and a
  122-slice test set with 11 positives. A single 4-patient test fold cannot
  estimate an AUROC to +-2 points. Every number produced under the
  official-split arm is therefore reported but is NOT the number we reason
  from. The arm we reason from is 5-fold subject-level cross-validation with
  pooled out-of-fold predictions, which uses all 45 patients as test patients
  exactly once (verified: the five cv*_split test sets partition all 1359
  rows). That is a deviation from their protocol and it is declared here.

DECLARED LIMITATION 2 -- RECONSTRUCTION.
  We do not hold their PCA-compressed single-virtual-coil k-space. Our cache
  holds an image-domain magnitude (RSS across coils) and an image-domain phase
  (sensitivity-map combined; RSS destroys phase, so it cannot be used for it)
  resampled to 224 x 224. We reconstitute a k-space as
      k = fft2c( mag * exp(i * phase) )
  This is a HYBRID: its modulus follows RSS while its argument follows the
  sensitivity-map combination, whereas theirs are both the first PCA
  component. Consequences, stated plainly:
    * "magnitude"        arm -- faithful. |ifft(k)| is exactly our cached mag.
    * "magnitude_phase"  arm -- faithful in content. Both channels are the
      image-domain quantities their code feeds; only the coil-combination
      recipe behind them differs.
    * "magnitude_kspace" arm -- an APPROXIMATION. k.real / k.imag depend on
      the hybrid above. Results from this arm are reported but are the weakest
      link in the chain and are labelled as such.
  Their center_crop to 224 x 224 in k-space is a no-op for us because our
  cached grid is already 224 x 224. That removes a degree of freedom rather
  than adding one.

DECLARED LIMITATION 3 -- THE "GOLD STANDARD" ARM IS OUT OF REACH.
  The 86.1% headline needs ADC + Trace50 + Trace1000 maps and a dual-encoder
  whose logits are averaged. ADC is not in the raw data and our cache does not
  carry the vendor trace maps as model inputs. We cannot reproduce 86.1 and we
  do not claim to. The arm we CAN instantiate is Table II, whose native-rate
  entries are 80.7-81.3.

DECLARED LIMITATION 4 -- THEIR PUBLISHED CODE DOES NOT RUN AS PUBLISHED.
  Three defects in the repository as fetched:
    (i)   `dwi_dataset.py` calls `KSpace_Dataset.stack_complex(x)` through the
          class, so the tensor binds to `self` and the `x` parameter is never
          supplied. Every non-"magnitude" data_type raises TypeError.
    (ii)  `get_test_loader` constructs the test Dataset without passing
          `data_type`, so it keeps the default `Literal[...]` sentinel and
          falls through to the 3-channel magnitude+real+imag branch whatever
          was configured.
    (iii) `train_dwi.py` calls `ConvNext(in_channels=..., num_classes=...)`
          but `ConvNext.__init__` requires `split`; and with `split: True` and
          `in_channel: 3` the config routes 2 channels into a 1-channel
          encoder.
  We are NOT claiming these defects produced their numbers. Published code
  routinely drifts from the code that made the tables. We record them because
  they mean the repository cannot arbitrate the [UNSPECIFIED] items for us, so
  those remain genuine degrees of freedom rather than things we could have
  looked up.

DEGREES OF FREEDOM WE CHOSE (all overridable from the CLI, all echoed into
every emitted JSON so no reader has to trust this docstring):
  batch size 32       paper says 128, repo config says 1. Our training split
                      is 6.7x smaller than theirs; 128 would give 8 optimiser
                      steps per epoch.
  epochs 30           their argparse default is 100; with patience 10 the
                      runs stop well before 30 on this cohort.
  patience 10         paper's number, on validation AUROC, which is the
                      quantity their code actually monitors (it uses 50).
  Adam defaults       betas (0.9, 0.999), eps 1e-8, following their code
                      rather than the paper's beta 0.99.
  phase-encode axis   their undersample() indexes axis -2 of a (1, H, W)
                      tensor. We do the same. Which physical axis that is on
                      our resampled grid is not something we can verify, so
                      it is a declared assumption.
  normalisation       per sample per channel, following their code, not the
                      paper's "batch wise".

=============================================================================
                      PART 3 -- THE CONTROLS APPLIED TO IT
=============================================================================
Each rung changes exactly ONE thing relative to the rung above, so a reader
can attribute the movement.

  W0  their reported number                     (read from the paper)
  W1  zero-image positional baseline            (no network at all)
  W2  this protocol, their evaluation level     (slice-level AUROC, their
                                                 unclustered slice bootstrap)
  W3  W2 predictions, subject-clustered CI      (only the interval changes)
  W4  W2 predictions, patient-level AUROC       (only the unit changes)
  W5  W2 with anatomy deleted                   (--region background)
  W6  W2 with labels permuted within subject    (the null)
  W7  W2 with the split stratified on acquisition

W1 deserves its own sentence. It is a 20-bin estimate of P(PI-RADS > 2 |
relative slice position) fitted on the TRAINING slices only and applied to the
test slices. It sees no pixels. It exists because per-slice PI-RADS labels on
a 30-slice stack are not positionally uniform -- the prostate, and any lesion
in it, occupies the middle of the stack -- so "how central is this slice"
is a strong slice-level predictor that carries no diagnostic information about
the patient and vanishes the moment you aggregate to patients. If W1 lands
near the published number then the published number is largely a statement
about slice geometry, and that conclusion needs no images and no reproduction
at all. W1 is computed BOTH on our 45-patient cache and directly on their own
published 312-patient label file, so it is not contingent on our subset.

=============================================================================
                        PART 4 -- WHAT ACTUALLY HAPPENED
=============================================================================
Run 2026-07-28, prostate_dwi cache, convnext_atto, seed 42, MPS.
Full payload: pipeline_out/rempe/waterfall_magnitude_phase.json

  rung  what changed                                  AUROC   95% CI          unit
  W0    reported (Table II, PCA x2, +Phase)           0.809   [0.788, 0.830]  slice
  W0h   their headline (gold standard, not reprod.)   0.861   [0.843, 0.879]  slice
  W1    ZERO-IMAGE positional baseline, THEIR labels  0.851   [0.821, 0.880]  slice
  W1p     the same baseline, patient-level            0.424   [0.298, 0.547]  patient
  W2    this protocol, their evaluation level         0.616   [0.559, 0.672]  slice
  W3      same predictions, subject-clustered CI      0.616   [0.528, 0.691]  slice
  W4      same predictions, patient-level             0.528   [0.356, 0.696]  patient
  W4s     same predictions, position held fixed       0.562        --         slice
  W5    + anatomy deleted (background only)           0.474   [0.370, 0.603]  slice
  W6    + labels permuted within subject (null)       0.667   [0.536, 0.779]  slice
  W7    + acquisition-stratified split (A2B / B2A)  0.582 / 0.523             slice
  Rempe-protocol magnitude arm, same ladder: 0.574 slice / 0.524 patient.

TWO CONCLUSIONS, AND THEY ARE NOT THE SAME CLAIM.

CONCLUSION 1 -- WE DID NOT REPRODUCE THEIR NUMBER. W2 = 0.616 against a
reported 0.809. We therefore have NO standing to say "we reproduced their
result and it failed our controls." The gap is fully consistent with causes
that are ours, not theirs: we train on 33 patients where they train on 218,
our k-space is the reconstituted hybrid of DECLARED LIMITATION 2, and our
coil combination is not their PCA. That W6, the permuted-label null, sits at
0.667 -- ABOVE our own W2 -- says our network arm carries no usable signal at
all and should be read as a failed reproduction, not as evidence about them.
Everything downstream of W2 (W5, W6, W7) is therefore a statement about a
protocol-as-described on 14% of the data, and nothing more.

CONCLUSION 2 -- THE PART THAT DOES NOT DEPEND ON OUR REPRODUCTION. W1 needs
none of our pixels, none of our reconstruction, none of our network and none
of our cohort. It is fitted on their own published per-slice label file, on
their own 218 training patients, and scored on their own 46 test patients. A
model whose ONLY input is "how far down the stack is this slice" reaches a
slice-level AUROC of 0.851 -- above every entry in their Table II
(0.783-0.813) and statistically indistinguishable from their 0.861 headline.
It is not an artefact of the estimator: 0.834 / 0.842 / 0.851 / 0.852 / 0.841
at 5 / 10 / 20 / 30 / 50 bins, and 0.841 with no fitting whatsoever, using
-(|relative position - 0.5|) directly.

  The same predictor scores 0.424 at patient level and 0.539 once slice
  position is held fixed.

The mechanism is not subtle. Per-slice PI-RADS labels on a ~30-slice stack are
not positionally uniform: the prostate, and any lesion in it, sits in the
middle of the volume. Under a slice-level AUROC every one of those slices is
an independent evaluation point, so "predict the middle slices" is a winning
strategy that encodes no information about which PATIENT has cancer. It is
worth 0.85 at their evaluation level and it is worth nothing at the level a
clinician cares about.

HOW THIS MAY AND MAY NOT BE WORDED. The supportable claim is: "a slice-level
AUROC on this dataset is largely determined by slice position; on the authors'
own published labels and split, a zero-image positional predictor attains
0.851, exceeding the values reported for every input condition in their
Table II, while attaining 0.424 at the patient level." The claim that is NOT
supportable, and must not be written, is "their result is wrong" or "their
result is confounded". We did not reproduce their pipeline, we do not have
their PCA k-space, and a model can perfectly well use slice position AND
genuine pathology at the same time. What the evidence supports is that the
reported metric does not, on its own, establish patient-level diagnostic
performance, and that a positional baseline is the control the comparison
needs.

=============================================================================
Usage
    python pipeline/s12_rempe.py --self-test
    python pipeline/s12_rempe.py --positional-baseline
    python pipeline/s12_rempe.py --run --arm cv --data-type magnitude_phase
    python pipeline/s12_rempe.py --waterfall --data-type magnitude_phase
    python pipeline/s12_rempe.py --report
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import s03_train  # noqa: E402
import s04_stats  # noqa: E402
import s05_controls  # noqa: E402
import s14_trivialbaselines as s14  # noqa: E402

logger = logging.getLogger(__name__)

COHORT = "prostate_dwi"
OUT_DIR = common.OUT_ROOT / "rempe"

DATA_TYPES = ("magnitude", "magnitude_phase", "magnitude_kspace")
DATA_TYPE_CHANNELS = {"magnitude": 1, "magnitude_phase": 2, "magnitude_kspace": 3}


# --- Part 1.7 transcribed, so the comparison target is in the code ----------
#
# Every entry is (auroc_percent, halfwidth_percent). "level" records that the
# number is a slice-level AUROC with an unclustered bootstrap interval, which
# is the only way the paper could have produced half-widths this narrow.
REPORTED = {
    "headline_goldstandard_x2_image_plus_kspace": (86.1, 1.8),
    "goldstandard_x0_image_only": (85.7, 1.6),
    "goldstandard_x2_image_only": (85.1, 1.9),
    "pca_x2_magnitude": (81.3, 2.2),
    "pca_x2_magnitude_phase": (80.9, 2.1),
    "pca_x2_magnitude_kspace": (81.1, 1.9),
    "grappa_x2_magnitude": (80.7, 1.9),
    "grappa_x2_magnitude_phase": (80.7, 1.8),
    "grappa_x2_magnitude_kspace": (78.3, 2.1),
}
REPORTED_LEVEL = "slice-level AUROC, unclustered bootstrap (1000 iterations)"

# The arm of Table II our inputs correspond to. Used as the reproduction
# target, NOT the 86.1 headline, which needs ADC/trace maps we do not hold.
REPRODUCTION_TARGET = {
    "magnitude": "pca_x2_magnitude",
    "magnitude_phase": "pca_x2_magnitude_phase",
    "magnitude_kspace": "pca_x2_magnitude_kspace",
}


# ---------------------------------------------------------------------------
# Their undersampling, ported verbatim in behaviour
# ---------------------------------------------------------------------------
def undersample_kspace(kspace: torch.Tensor, factor: int) -> torch.Tensor:
    """
    Port of `undersample` in their datasets/dwi_dataset.py.

    Keeps every `factor`-th line counting outward from the midline in both
    directions and zeroes everything else. `factor <= 1` is a no-op, which is
    why their `random.randint(0, 8)` augmentation leaves ~2/9 of samples fully
    sampled.

    Their code indexes `mask[0, midline::factor]` on a (1, H, W) tensor, so the
    lines are rows -- axis -2. We keep that indexing; which physical gradient
    axis it corresponds to on our resampled grid is a declared assumption, not
    something we can verify (see PART 2).
    """
    if factor <= 1:
        return kspace
    out = kspace.clone()
    keep = torch.zeros(out.shape[-2], dtype=torch.bool)
    midline = out.shape[-2] // 2
    keep[midline::factor] = True
    # midline::-factor walks backwards to index 0 inclusive.
    keep[torch.arange(midline, -1, -factor)] = True
    out[..., ~keep, :] = 0
    return out


def minmax_per_channel(x: torch.Tensor) -> torch.Tensor:
    """Their `normalization`: per channel, rescale to [0, 1]."""
    out = x.clone()
    for c in range(out.shape[0]):
        lo = out[c].min()
        hi = out[c].max()
        rng = hi - lo
        out[c] = (out[c] - lo) / rng if rng > 0 else torch.zeros_like(out[c])
    return out


def zscore_per_channel(x: torch.Tensor) -> torch.Tensor:
    """Their `standardization`: per channel, subtract mean, divide by sd."""
    out = x.clone()
    for c in range(out.shape[0]):
        sd = out[c].std()
        out[c] = (out[c] - out[c].mean()) / sd if sd > 0 else torch.zeros_like(out[c])
    return out


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class RempeDataset(Dataset):
    """
    Our cache, presented through their preprocessing chain.

    Their chain starts from a complex k-space slice. Ours starts from a cached
    image-domain (magnitude, phase) pair, so we re-form the complex image and
    transform it to k-space first; see DECLARED LIMITATION 2. From that point
    the operation order is theirs exactly:

        hflip (train, p=0.5, on the k-space tensor, as their code does)
        undersample(factor)
        center-crop to 224x224   [no-op for us; our grid is already 224x224]
        split into channels for the requested data_type
        per-channel min-max to [0, 1]
        per-channel z-score

    `region` is ours, not theirs: it is the falsification control. The mask is
    applied to the complex IMAGE before the forward FFT, so deleting anatomy
    survives into k-space instead of being smeared back across it.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        data_type: str,
        h5_path: Path | None = None,
        arrays: dict | None = None,
        train: bool = False,
        region: str = "full",
        sampling_factor: int | None = None,
        seed: int = 0,
    ):
        if (h5_path is None) == (arrays is None):
            raise ValueError("provide exactly one of h5_path or arrays")
        if data_type not in DATA_TYPES:
            raise ValueError(f"unknown data_type {data_type!r}; expected {DATA_TYPES}")
        if region not in ("full", "body", "background"):
            raise ValueError(f"unknown region {region!r}")

        self.rows = rows.reset_index(drop=True)
        self.data_type = data_type
        self.in_channels = DATA_TYPE_CHANNELS[data_type]
        self.h5_path = Path(h5_path) if h5_path is not None else None
        self.arrays = arrays
        self.train = train
        self.region = region
        self.sampling_factor = sampling_factor
        self.seed = seed

        self.indices = self.rows["idx"].to_numpy(dtype=np.int64)
        self.labels = self.rows["label"].to_numpy(dtype=np.int64)
        self.subject_ids = self.rows["subject_id"].astype(str).to_numpy()
        self._h5 = None

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, k: int):
        if self.arrays is not None:
            mag, phase, mask = (self.arrays[n][k] for n in ("mag", "phase", "mask"))
        else:
            if self._h5 is None:
                import h5py

                self._h5 = h5py.File(self.h5_path, "r")
            mag, phase, mask = (self._h5[n][k] for n in ("mag", "phase", "mask"))
        return (
            np.asarray(mag, dtype=np.float32),
            np.asarray(phase, dtype=np.float32),
            np.asarray(mask, dtype=bool),
        )

    def __getitem__(self, i: int):
        k = int(self.indices[i])
        mag, phase, mask = self._read(k)

        # Our control, applied in the image domain before the FFT.
        if self.region == "body":
            keep = mask
        elif self.region == "background":
            keep = ~mask
        else:
            keep = None
        if keep is not None:
            mag = mag * keep.astype(np.float32)
            phase = phase * keep.astype(np.float32)

        # DECLARED LIMITATION 2: reconstituted, hybrid k-space.
        image = torch.from_numpy(mag).to(torch.complex64) * torch.exp(
            1j * torch.from_numpy(phase).to(torch.complex64)
        )
        kspace = torch.from_numpy(
            common.fft2c(image.numpy())
        ).to(torch.complex64).unsqueeze(0)

        # Deterministic per-(epoch-free) sample randomness so a run is
        # reproducible from (seed, cache idx) alone.
        rng = np.random.default_rng((self.seed * 1_000_003 + k) % (2**32))

        if self.train:
            if rng.random() < 0.5:
                kspace = torch.flip(kspace, dims=[-1])
            kspace = undersample_kspace(kspace, int(rng.integers(0, 9)))
        elif self.sampling_factor:
            kspace = undersample_kspace(kspace, int(self.sampling_factor))

        # Their center_crop(224, 224) -- a no-op on our grid, kept for clarity.
        if kspace.shape[-2:] != (224, 224):
            raise RuntimeError(f"expected a 224x224 cache grid, got {tuple(kspace.shape[-2:])}")

        img = torch.fft.fftshift(
            torch.fft.ifft2(torch.fft.ifftshift(kspace, dim=(-2, -1)), norm="ortho"),
            dim=(-2, -1),
        )
        if self.data_type == "magnitude":
            x = img.abs()
        elif self.data_type == "magnitude_phase":
            x = torch.cat([img.abs(), img.angle()], dim=0)
        else:
            x = torch.cat([img.abs(), kspace.real, kspace.imag], dim=0)

        x = zscore_per_channel(minmax_per_channel(x.float()))
        x = torch.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)
        return x, int(self.labels[i]), i


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------
def build_model(in_channels: int, pretrained: bool = True) -> nn.Module:
    """
    Their model.py, non-split branch: timm convnext_atto, ImageNet pretrained.

    The split (dual-encoder) branch belongs to the Table I "gold standard"
    experiment, which we cannot instantiate (DECLARED LIMITATION 3), so it is
    deliberately absent rather than approximated.
    """
    import timm

    return timm.create_model(
        "convnext_atto", pretrained=pretrained, in_chans=in_channels, num_classes=2
    )


# ---------------------------------------------------------------------------
# Training, following their train_dwi.py
# ---------------------------------------------------------------------------
@dataclass
class TrainConfig:
    data_type: str = "magnitude_phase"
    lr: float = 1e-4
    batch_size: int = 32          # declared deviation; paper 128, repo config 1
    epochs: int = 30              # declared deviation; their argparse default 100
    patience: int = 10            # paper's value; their code uses 50
    class_weight: float = 17.0    # theirs
    t_max: int = 3                # their CosineAnnealingLR(T_max=3)
    pretrained: bool = True
    region: str = "full"
    seed: int = 42
    device: str = "auto"
    workers: int = 0
    paper_batch_size: int = 128
    repo_batch_size: int = 1
    repo_patience: int = 50
    monitor: str = "val_auroc"
    normalisation: str = "per-sample-per-channel (their code; paper says batch-wise)"
    optimiser: str = "Adam, library-default betas/eps (their code; paper says beta 0.99)"


def _epoch(model, loader, device, criterion, optimizer=None):
    """
    One pass over a loader.

    Two things here are deliberate and must not be "tidied" back:

    1. The labels reported out of this function are the CPU-side labels
       captured BEFORE the device transfer, never a copy read back off the
       accelerator. On MPS a device->host read of an int64 tensor can race a
       still-pending host->device write and return uninitialised memory; that
       silently turns the target vector into ~1e18 garbage, trains the network
       against nonsense and yields a NaN AUROC that looks like a failed
       reproduction rather than a bug. There is no reason to ask the GPU for
       numbers we already have.
    2. non_blocking is off. Without pinned memory it buys nothing, and it is
       what opens the race above.

    _assert_label_integrity() below is the standing guard against this whole
    class of failure.
    """
    train = optimizer is not None
    model.train(train)
    total, n = 0.0, 0
    probs, labels, idxs = [], [], []
    for x, y, i in loader:
        y_cpu = y.detach().clone().numpy()      # authoritative copy, host side
        idx_cpu = i.detach().clone().numpy()
        x = x.to(device)
        y = y.to(device)
        with torch.set_grad_enabled(train):
            logits = model(x)
            loss = criterion(logits, y)
        if train:
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        total += float(loss.item()) * len(y_cpu)
        n += len(y_cpu)
        probs.append(torch.softmax(logits.detach().float(), dim=1)[:, 1].cpu().numpy())
        labels.append(y_cpu)
        idxs.append(idx_cpu)
    out = {
        "loss": total / max(n, 1),
        "probs": np.concatenate(probs) if probs else np.array([]),
        "labels": np.concatenate(labels) if labels else np.array([], dtype=int),
        "order": np.concatenate(idxs) if idxs else np.array([], dtype=int),
    }
    _assert_label_integrity(out["labels"], loader.dataset)
    return out


def _assert_label_integrity(labels: np.ndarray, dataset) -> None:
    """
    Fail loudly if the labels that came back are not the labels we put in.

    A corrupted target vector produces a plausible-looking bad AUROC, which is
    the single most dangerous failure mode in a study whose headline claim is a
    null: it manufactures evidence for the conclusion we already expect.
    """
    if labels.size == 0:
        raise RuntimeError("epoch produced no labels")
    bad = np.setdiff1d(np.unique(labels), np.array([0, 1]))
    if bad.size:
        raise RuntimeError(
            f"label corruption: values outside {{0,1}} came back from the epoch: {bad[:5]}"
        )
    expected = np.bincount(np.asarray(dataset.labels, dtype=int), minlength=2)
    got = np.bincount(labels.astype(int), minlength=2)
    if not np.array_equal(expected, got):
        raise RuntimeError(
            f"label corruption: class counts {got.tolist()} != dataset counts {expected.tolist()}"
        )


def train_arm(index: pd.DataFrame, cfg: TrainConfig, h5_path: Path,
              arrays: dict | None = None) -> dict:
    """
    Train one configuration and return out-of-sample test predictions.

    Model selection is theirs: keep the weights from the epoch with the best
    validation AUROC, stop when it has not improved for `patience` epochs.
    """
    s03_train.set_seed(cfg.seed)
    device = s03_train.pick_device(cfg.device)
    index = s03_train.normalize_splits(index.copy())
    s03_train.assert_no_patient_leakage(index, split_col="official_split")

    subsets = {s: index[index["official_split"] == s] for s in ("training", "validation", "test")}
    for name, rows in subsets.items():
        if len(rows) == 0:
            raise RuntimeError(f"split {name!r} is empty")
    if subsets["training"]["label"].nunique() < 2:
        raise RuntimeError("training split is single-class")

    def loader(split, train):
        ds = RempeDataset(
            subsets[split], cfg.data_type, h5_path=h5_path, arrays=arrays,
            train=train, region=cfg.region, seed=cfg.seed,
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=train,
                          num_workers=cfg.workers, drop_last=False)

    train_loader = loader("training", True)
    val_loader = loader("validation", False)
    test_loader = loader("test", False)

    model = build_model(DATA_TYPE_CHANNELS[cfg.data_type], pretrained=cfg.pretrained).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.t_max, eta_min=0)
    criterion = nn.CrossEntropyLoss(weight=torch.tensor([1.0, cfg.class_weight], device=device))

    best, best_epoch, best_state, since = -np.inf, -1, None, 0
    history = []
    t0 = time.time()
    for epoch in range(cfg.epochs):
        tr = _epoch(model, train_loader, device, criterion, optimizer)
        scheduler.step()
        with torch.no_grad():
            va = _epoch(model, val_loader, device, criterion)
        va_auc = s04_stats.auc_midrank(va["labels"], va["probs"])
        history.append({"epoch": epoch, "train_loss": tr["loss"],
                        "val_loss": va["loss"], "val_auc": va_auc})
        logger.info("  epoch %2d  train_loss %.4f  val_loss %.4f  val_auc %.4f",
                    epoch, tr["loss"], va["loss"], va_auc)
        if np.isfinite(va_auc) and va_auc > best:
            best, best_epoch, since = va_auc, epoch, 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            since += 1
            if since >= cfg.patience:
                logger.info("  early stop at epoch %d (best epoch %d)", epoch, best_epoch)
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    with torch.no_grad():
        te = _epoch(model, test_loader, device, criterion)

    rows = subsets["test"].reset_index(drop=True)
    order = te["order"]
    return {
        "labels": te["labels"].tolist(),
        "probs": te["probs"].tolist(),
        "subject_ids": rows["subject_id"].astype(str).to_numpy()[order].tolist(),
        "cache_idx": rows["idx"].to_numpy()[order].tolist(),
        "best_epoch": int(best_epoch),
        "best_val_auc": float(best),
        "epochs_run": len(history),
        "history": history,
        "wall_seconds": time.time() - t0,
        "device": str(device),
        "n_train": int(len(subsets["training"])),
        "n_val": int(len(subsets["validation"])),
        "n_test": int(len(subsets["test"])),
        "n_train_subjects": int(subsets["training"]["subject_id"].nunique()),
        "n_test_subjects": int(subsets["test"]["subject_id"].nunique()),
    }


# ---------------------------------------------------------------------------
# Evaluation at every level, from ONE set of predictions
# ---------------------------------------------------------------------------
def stratified_auc(labels, scores, strata) -> dict:
    """
    AUROC computed WITHIN strata and pooled, i.e. a stratified Mann-Whitney.

    Only positive/negative pairs that sit in the same stratum contribute, so
    stratifying on slice position removes exactly the part of a slice-level
    AUROC that comes from "positives sit nearer the middle of the stack". A
    predictor whose whole signal is positional falls to 0.5 here; a predictor
    that actually reads the patient does not.

    Pooling weights each stratum by its number of comparable pairs, which is
    the weighting that makes the pooled statistic equal the overall AUROC when
    there is only one stratum.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    strata = np.asarray(strata)
    num = den = 0.0
    used = 0
    for s in np.unique(strata):
        m = strata == s
        y, sc = labels[m], scores[m]
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos == 0 or nneg == 0:
            continue
        a = s04_stats.auc_midrank(y, sc)
        if not np.isfinite(a):
            continue
        w = npos * nneg
        num += a * w
        den += w
        used += 1
    return {
        "auc": float(num / den) if den > 0 else float("nan"),
        "n_strata_used": used,
        "n_pairs": int(den),
    }


def position_strata(relpos, n_strata: int = 10) -> np.ndarray:
    """Equal-width bins of relative slice position, as stratification labels."""
    edges = np.linspace(0.0, 1.0, n_strata + 1)
    return np.clip(np.digitize(np.asarray(relpos, dtype=float), edges) - 1, 0, n_strata - 1)


def evaluate_predictions(labels, probs, subject_ids, n_boot: int = 2000,
                         seed: int = 0, relpos=None) -> dict:
    """
    The same predictions read four ways.

    W2/W3/W4 of the waterfall differ ONLY in which of these keys you read, so
    nothing between those rungs can be attributed to a different model, a
    different split or a different seed.
    """
    labels = np.asarray(labels, dtype=int)
    probs = np.asarray(probs, dtype=float)
    subject_ids = np.asarray(subject_ids, dtype=object)

    slice_auc = s04_stats.auc_midrank(labels, probs)
    naive = s04_stats.naive_slice_bootstrap_auc(labels, probs, n_boot=n_boot, seed=seed)
    clustered = s04_stats.cluster_bootstrap_auc(labels, probs, subject_ids,
                                                n_boot=n_boot, seed=seed)
    agg = s04_stats.aggregate_by_cluster(labels, probs, subject_ids, how="mean")
    pat_auc = s04_stats.auc_midrank(np.asarray(agg["labels"]), np.asarray(agg["scores"]))
    pat_ci = s04_stats.cluster_bootstrap_auc(
        np.asarray(agg["labels"]), np.asarray(agg["scores"]),
        np.asarray(agg["cluster_ids"], dtype=object), n_boot=n_boot, seed=seed,
    )
    out = {
        "n_slices": int(len(labels)),
        "n_pos_slices": int(labels.sum()),
        "n_subjects": int(len(set(subject_ids.tolist()))),
        # W2 -- their level, their interval
        "slice_auc": float(slice_auc),
        "slice_ap": float(s04_stats.average_precision(labels, probs)),
        "slice_ci_naive": [naive.get("ci_lo"), naive.get("ci_hi")],
        # W3 -- same predictions, honest interval
        "slice_ci_clustered": [clustered.get("ci_lo"), clustered.get("ci_hi")],
        # W4 -- same predictions, patient unit
        "patient_auc": float(pat_auc),
        "patient_ci_clustered": [pat_ci.get("ci_lo"), pat_ci.get("ci_hi")],
        "n_patients": int(agg["n_clusters"]),
        "n_pos_patients": int(agg["n_pos_clusters"]),
    }
    if relpos is not None:
        strat = stratified_auc(labels, probs, position_strata(relpos))
        out["slice_auc_position_stratified"] = strat["auc"]
        out["position_strata_used"] = strat["n_strata_used"]
    return out


# ---------------------------------------------------------------------------
# W1 -- the zero-image positional baseline
# ---------------------------------------------------------------------------
def positional_scores(train_rows: pd.DataFrame, test_rows: pd.DataFrame,
                      n_bins: int = 20, pos_col: str = "relpos") -> np.ndarray:
    """
    P(label = 1 | relative slice position), binned, fitted on TRAINING rows and
    applied to TEST rows. Empty bins fall back to the training prevalence.

    This model has no access to pixels, to k-space, to phase, or to anything
    else about the patient. It knows only where a slice sits in its own stack.

    The implementation lives in s14_trivialbaselines, which generalises it to
    arbitrary column names and wraps it in the fit/score contract used by the
    whole zero-image baseline family. This wrapper is kept so the s12 call
    sites, and the numbers already reported from them, are unchanged.
    """
    return s14.positional_scores(train_rows, test_rows, n_bins=n_bins,
                                 pos_col=pos_col, label_col="label")


def add_relpos(df: pd.DataFrame, subject_col: str = "subject_id",
               slice_col: str = "slice") -> pd.DataFrame:
    """Relative position of each slice within its own stack, in [0, 1].

    Thin wrapper over s14_trivialbaselines.add_relative_position, which takes
    the grouping column explicitly (a subject can hold several stacks) instead
    of assuming subject_id.
    """
    return s14.add_relative_position(df, volume_col=subject_col,
                                     slice_col=slice_col, out_col="relpos")


def run_positional_baseline(index: pd.DataFrame, n_boot: int = 2000) -> dict:
    """W1 on our cache, under the official split and pooled over the CV folds."""
    idx = add_relpos(index)
    out = {}
    for name, col in [("official", "official_split")] + [(f"cv{k}", f"cv{k}_split") for k in range(5)]:
        tr = idx[idx[col] == "training"]
        te = idx[idx[col] == "test"]
        if len(tr) == 0 or len(te) == 0 or te["label"].nunique() < 2:
            continue
        out[name] = {
            "scores": positional_scores(tr, te),
            "labels": te["label"].to_numpy(),
            "subjects": te["subject_id"].astype(str).to_numpy(),
            "relpos": te["relpos"].to_numpy(),
        }
    res = {}
    if "official" in out:
        o = out["official"]
        res["official_split"] = evaluate_predictions(o["labels"], o["scores"],
                                                     o["subjects"], n_boot=n_boot,
                                                     relpos=o["relpos"])
    folds = [out[f"cv{k}"] for k in range(5) if f"cv{k}" in out]
    if folds:
        res["cv_pooled_oof"] = evaluate_predictions(
            np.concatenate([f["labels"] for f in folds]),
            np.concatenate([f["scores"] for f in folds]),
            np.concatenate([f["subjects"] for f in folds]),
            n_boot=n_boot,
            relpos=np.concatenate([f["relpos"] for f in folds]),
        )
    return res


def run_positional_baseline_on_their_labels(labels_csv: Path, n_boot: int = 2000) -> dict:
    """
    W1 computed on THEIR published 312-patient label file, under THEIR split.

    This is the version that does not depend on our 45-patient subset, our
    reconstruction, our normalisation, our network or our compute. It needs
    only the label CSV they published.
    """
    r = pd.read_csv(labels_csv)
    r = r.rename(columns={"fastmri_pt_id": "subject_id"})
    r["label"] = (r["PIRADS"] > 2).astype(int)
    r = add_relpos(r)
    tr = r[r["data_split"] == "training"]
    te = r[r["data_split"] == "test"]
    scores = positional_scores(tr, te)
    res = evaluate_predictions(te["label"].to_numpy(), scores,
                               te["subject_id"].astype(str).to_numpy(), n_boot=n_boot,
                               relpos=te["relpos"].to_numpy())
    res["n_train_slices"] = int(len(tr))
    res["n_train_subjects"] = int(tr["subject_id"].nunique())
    res["source"] = str(labels_csv)

    # The 20-bin histogram is an arbitrary estimator, so record that the result
    # does not depend on it. `centrality_no_fit` uses no training data at all:
    # it is -(|relpos - 0.5|), i.e. "how close to the middle of the stack".
    y_te = te["label"].to_numpy()
    res["bin_sweep"] = {
        str(nb): float(s04_stats.auc_midrank(y_te, positional_scores(tr, te, n_bins=nb)))
        for nb in (5, 10, 20, 30, 50)
    }
    res["centrality_no_fit"] = float(
        s04_stats.auc_midrank(y_te, -np.abs(te["relpos"].to_numpy() - 0.5))
    )
    return res


# ---------------------------------------------------------------------------
# Arms
# ---------------------------------------------------------------------------
def relpos_for(index: pd.DataFrame, cache_idx) -> np.ndarray:
    """Relative slice position for a list of cache row ids, in input order."""
    lut = add_relpos(index).set_index("idx")["relpos"]
    return lut.reindex(np.asarray(cache_idx)).to_numpy(dtype=float)


def evaluate_run(res: dict, index: pd.DataFrame, n_boot: int) -> dict:
    """evaluate_predictions for one trained run, with position stratification."""
    return evaluate_predictions(res["labels"], res["probs"], res["subject_ids"],
                                n_boot=n_boot,
                                relpos=relpos_for(index, res["cache_idx"]))


def load_index(cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{COHORT}_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"missing cache index {path}")
    return pd.read_csv(path)


def index_for_split_col(index: pd.DataFrame, split_col: str) -> pd.DataFrame:
    """Promote one of the cv*_split columns into official_split."""
    if split_col not in index.columns:
        raise KeyError(f"{split_col!r} not in cache index")
    out = index.copy()
    out["official_split"] = out[split_col]
    return s03_train.normalize_splits(out)


def acquisition_split_indices(index: pd.DataFrame, key: str, val_frac: float = 0.2,
                              seed: int = 0) -> list[tuple[str, pd.DataFrame]]:
    """
    Rebuild the split so an acquisition property is HELD OUT.

    Reuses the s05 primitives so this control is the same control the rest of
    the study runs, not a second implementation of it.
    """
    df = index[index[key].notna()].copy()
    gkey = s05_controls.group_level_value(df, "subject_id", key)
    values = gkey.value_counts()
    if len(values) < 2:
        raise RuntimeError(f"{key!r} takes one value in this cohort")
    a, b = s05_controls.balanced_bipartition(values)
    arms = {"A": a, "B": b}
    out = []
    for train_arm, test_arm in (("A", "B"), ("B", "A")):
        train_subj = set(gkey[gkey.isin(arms[train_arm])].index)
        d = df.copy()
        d["official_split"] = np.where(d["subject_id"].isin(train_subj), "training", "test")
        d = s05_controls.carve_validation_by_group(d, "subject_id", val_frac, seed)
        try:
            s05_controls.assert_trainable(d, f"acquisition_split {train_arm}2{test_arm}")
        except RuntimeError as exc:
            logger.warning("  skipping direction %s2%s: %s", train_arm, test_arm, exc)
            continue
        out.append((f"{train_arm}2{test_arm}", d))
    if not out:
        raise RuntimeError(f"no trainable direction for acquisition key {key!r}")
    return out


def run_cv_arm(index: pd.DataFrame, cfg: TrainConfig, h5_path: Path,
               permute: bool = False, n_boot: int = 2000) -> dict:
    """
    Five subject-level folds, pooled out-of-fold, so every one of the 45
    patients is a test patient exactly once.
    """
    folds = []
    for k in range(5):
        idx = index_for_split_col(index, f"cv{k}_split")
        detail = None
        if permute:
            idx, detail = s05_controls.permute_labels_by_subject(
                idx, "subject_id", ("training", "validation", "test"), seed=cfg.seed + k
            )
        logger.info("fold cv%d  data_type=%s region=%s permute=%s", k, cfg.data_type,
                    cfg.region, permute)
        res = train_arm(idx, cfg, h5_path)
        res["fold"] = k
        res["permutation_detail"] = detail
        folds.append(res)
    pooled = evaluate_predictions(
        np.concatenate([f["labels"] for f in folds]),
        np.concatenate([f["probs"] for f in folds]),
        np.concatenate([f["subject_ids"] for f in folds]),
        n_boot=n_boot,
        relpos=relpos_for(index, np.concatenate([f["cache_idx"] for f in folds])),
    )
    return {"pooled": pooled, "folds": folds}


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------
def _fmt_ci(pair) -> str:
    lo, hi = pair
    if lo is None or hi is None or not np.isfinite(lo) or not np.isfinite(hi):
        return "     n/a      "
    return f"[{lo:.3f}, {hi:.3f}]"


def print_waterfall(rows: list[dict]) -> None:
    print()
    print("=" * 104)
    print("WATERFALL -- Rempe et al. (arXiv:2407.06165) protocol, on the PhaseDx prostate_dwi cache")
    print("=" * 104)
    print(f"{'rung':<5}{'what changed':<46}{'AUROC':>8}  {'95% CI':<18}{'unit':<12}")
    print("-" * 104)
    for r in rows:
        print(f"{r['rung']:<5}{r['change']:<46}{r['auc']:>8}  {r['ci']:<18}{r['unit']:<12}")
    print("-" * 104)
    for note in (n for r in rows for n in [r.get("note")] if n):
        print(f"  * {note}")
    print()


def build_waterfall(payload: dict) -> list[dict]:
    """
    Turn the saved run payload into the ordered waterfall rows.

    Each rung changes exactly one thing relative to the one above it. W2, W3,
    W4 and W4s are all the SAME predictions from the SAME model -- only the
    interval, the unit of analysis, or the conditioning changes -- so nothing
    that moves between them can be blamed on a different model or split.
    """
    rows: list[dict] = []
    arms = payload.get("arms", {})
    dt = payload["config"]["data_type"]
    key = REPRODUCTION_TARGET[dt]

    auc, hw = REPORTED[key]
    rows.append({
        "rung": "W0", "change": f"reported by Rempe et al. ({key})",
        "auc": f"{auc / 100:.3f}", "ci": _fmt_ci([(auc - hw) / 100, (auc + hw) / 100]),
        "unit": "slice",
        "note": ("W0 is transcribed from their Table II, not recomputed; its interval is the "
                 "reported +/- half-width."),
    })
    hl, hlw = REPORTED["headline_goldstandard_x2_image_plus_kspace"]
    mg = REPORTED["goldstandard_x0_image_only"][0]
    rows.append({
        "rung": "W0h", "change": "  their 0.861 headline, for reference (NOT reproduced)",
        "auc": f"{hl / 100:.3f}", "ci": _fmt_ci([(hl - hlw) / 100, (hl + hlw) / 100]),
        "unit": "slice",
        "note": (f"The headline is the gold-standard ADC+trace dual-network arm, which needs "
                 f"maps we do not hold. Their own magnitude-only control for it is "
                 f"{mg / 100:.3f}."),
    })

    pb = payload.get("positional_baseline_their_labels")
    if pb:
        rows.append({
            "rung": "W1", "change": "zero-image positional baseline, THEIR 312-pt labels",
            "auc": f"{pb['slice_auc']:.3f}", "ci": _fmt_ci(pb["slice_ci_naive"]),
            "unit": "slice",
            "note": ("W1 uses no pixels, no phase and no k-space. It is P(PI-RADS>2 | relative "
                     "slice position), fitted on their 218 training patients and scored on "
                     "their 46 test patients, from their own published label CSV."),
        })
        rows.append({
            "rung": "W1p", "change": "  the same zero-image baseline, patient-level",
            "auc": f"{pb['patient_auc']:.3f}", "ci": _fmt_ci(pb["patient_ci_clustered"]),
            "unit": "patient", "note": None,
        })

    spec = [
        ("real", "W2", "this protocol, at THEIR evaluation level",
         "slice_auc", "slice_ci_naive", "slice"),
        ("real", "W3", "  same predictions, subject-clustered CI",
         "slice_auc", "slice_ci_clustered", "slice"),
        ("real", "W4", "  same predictions, patient-level AUROC",
         "patient_auc", "patient_ci_clustered", "patient"),
        ("real", "W4s", "  same predictions, slice position held fixed",
         "slice_auc_position_stratified", None, "slice"),
        ("official", "W2o", "  their exact 70/15/15 split (4 test patients)",
         "slice_auc", "slice_ci_clustered", "slice"),
        ("background", "W5", "+ anatomy deleted (background only)",
         "slice_auc", "slice_ci_clustered", "slice"),
        ("permuted", "W6", "+ labels permuted within subject (the null)",
         "slice_auc", "slice_ci_clustered", "slice"),
    ]
    for tag, rung, label, auc_key, ci_key, unit in spec:
        blk = arms.get(tag)
        if not blk or auc_key not in blk["pooled"]:
            continue
        p = blk["pooled"]
        rows.append({
            "rung": rung, "change": label, "auc": f"{p[auc_key]:.3f}",
            "ci": _fmt_ci(p[ci_key]) if ci_key else "      n/a       ",
            "unit": unit,
            "note": (("W4s is a stratified Mann-Whitney over 10 relative-position bins: only "
                      "same-position slice pairs contribute, so whatever the slice-level "
                      "AUROC owed to stack geometry is removed.") if rung == "W4s" else
                     ("W2o is reported for completeness only. A 4-patient test fold cannot "
                      "estimate an AUROC and no conclusion rests on it.") if rung == "W2o"
                     else None),
        })

    for name, blk in sorted(arms.items()):
        if not name.startswith("acq_"):
            continue
        p = blk["pooled"]
        rows.append({
            "rung": "W7", "change": f"+ acquisition-stratified split ({name[4:]})",
            "auc": f"{p['slice_auc']:.3f}", "ci": _fmt_ci(p["slice_ci_clustered"]),
            "unit": "slice", "note": None,
        })
    return rows


# ---------------------------------------------------------------------------
# Self-test
# ---------------------------------------------------------------------------
def self_test() -> int:
    failures = []

    def check(name, cond, detail=""):
        if cond:
            print(f"  PASS  {name}")
        else:
            print(f"  FAIL  {name}  {detail}")
            failures.append(name)

    print("s12_rempe self-test")

    # --- undersampling -----------------------------------------------------
    k = torch.ones(1, 16, 8, dtype=torch.complex64)
    check("undersample factor<=1 is a no-op", torch.equal(undersample_kspace(k, 1), k))
    check("undersample factor 0 is a no-op", torch.equal(undersample_kspace(k, 0), k))
    u4 = undersample_kspace(k, 4)
    nz = (u4.abs().sum(dim=-1)[0] > 0).nonzero().flatten().tolist()
    check("undersample x4 keeps every 4th line from midline",
          nz == [0, 4, 8, 12], f"kept rows {nz}")
    check("undersample x4 keeps ~1/4 of lines", len(nz) == 4, f"{len(nz)} of 16")
    u2 = undersample_kspace(k, 2)
    check("undersample x2 keeps half the lines",
          int((u2.abs().sum(dim=-1)[0] > 0).sum()) == 8)
    check("undersample keeps the midline",
          bool(u4.abs().sum(dim=-1)[0][8] > 0))
    check("undersample does not mutate its input", torch.equal(k, torch.ones_like(k)))

    # --- normalisation -----------------------------------------------------
    x = torch.tensor([[[1.0, 2.0], [3.0, 4.0]], [[-5.0, 0.0], [5.0, 10.0]]])
    mm = minmax_per_channel(x)
    check("min-max maps each channel to [0,1]",
          all(abs(float(mm[c].min())) < 1e-6 and abs(float(mm[c].max()) - 1) < 1e-6
              for c in range(2)))
    zz = zscore_per_channel(mm)
    check("z-score gives each channel mean 0",
          all(abs(float(zz[c].mean())) < 1e-5 for c in range(2)))
    const = minmax_per_channel(torch.ones(1, 4, 4))
    check("constant channel does not produce NaN", bool(torch.isfinite(const).all()))
    check("constant channel z-score does not produce NaN",
          bool(torch.isfinite(zscore_per_channel(torch.ones(1, 4, 4))).all()))

    # --- FFT round trip ----------------------------------------------------
    rng = np.random.default_rng(0)
    img = (rng.standard_normal((32, 32)) + 1j * rng.standard_normal((32, 32))).astype(np.complex64)
    back = common.ifft2c(common.fft2c(img))
    check("fft2c/ifft2c round trip", float(np.abs(back - img).max()) < 1e-4,
          f"max err {float(np.abs(back - img).max()):.2e}")

    # --- dataset channel construction --------------------------------------
    n = 12
    arrays = {
        "mag": rng.random((n, 224, 224)).astype(np.float32),
        "phase": (rng.random((n, 224, 224)).astype(np.float32) * 2 - 1) * np.pi,
        "mask": np.zeros((n, 224, 224), dtype=bool),
    }
    arrays["mask"][:, 60:160, 60:160] = True
    rows = pd.DataFrame({
        "idx": np.arange(n), "label": [0, 1] * (n // 2),
        "subject_id": [f"s{i//3}" for i in range(n)], "slice": list(range(n)),
    })
    for dt, want in DATA_TYPE_CHANNELS.items():
        ds = RempeDataset(rows, dt, arrays=arrays, train=False)
        x, y, i = ds[0]
        check(f"{dt} yields {want} channels", tuple(x.shape) == (want, 224, 224),
              f"got {tuple(x.shape)}")
        check(f"{dt} output is finite", bool(torch.isfinite(x).all()))

    ds_m = RempeDataset(rows, "magnitude", arrays=arrays, train=False)
    x_m, _, _ = ds_m[0]
    ref = zscore_per_channel(minmax_per_channel(
        torch.from_numpy(arrays["mag"][0]).unsqueeze(0)))
    check("magnitude channel reproduces the cached magnitude through their chain",
          float((x_m - ref).abs().max()) < 1e-3, f"max diff {float((x_m - ref).abs().max()):.2e}")

    ds_p = RempeDataset(rows, "magnitude_phase", arrays=arrays, train=False)
    x_p, _, _ = ds_p[0]
    check("magnitude_phase channel 0 equals the magnitude arm",
          float((x_p[0] - x_m[0]).abs().max()) < 1e-3)

    # The background control must actually delete the anatomy. The right
    # assertion is that the body box becomes CONSTANT, not that it becomes
    # small: min-max followed by z-scoring maps a deleted region to a single
    # large-magnitude negative value, so an energy comparison would fail on a
    # correctly masked input.
    ds_bg = RempeDataset(rows, "magnitude", arrays=arrays, train=False, region="background")
    x_bg, _, _ = ds_bg[0]
    body_sd_bg = float(x_bg[0][70:150, 70:150].std())
    edge_sd_bg = float(x_bg[0][:40, :40].std())
    body_sd_full, _, _ = RempeDataset(rows, "magnitude", arrays=arrays, train=False)[0]
    body_sd_full = float(body_sd_full[0][70:150, 70:150].std())
    check("background region flattens the body box to a constant",
          body_sd_bg < 1e-4, f"body sd {body_sd_bg:.2e}")
    check("background region leaves the air outside intact",
          edge_sd_bg > 0.1, f"edge sd {edge_sd_bg:.3f}")
    check("the same box is NOT flat without the control",
          body_sd_full > 0.1, f"full-region body sd {body_sd_full:.3f}")

    # eval is deterministic, train is stochastic
    a, _, _ = RempeDataset(rows, "magnitude", arrays=arrays, train=False)[3]
    b, _, _ = RempeDataset(rows, "magnitude", arrays=arrays, train=False)[3]
    check("eval-mode sampling is deterministic", torch.allclose(a, b))
    t1, _, _ = RempeDataset(rows, "magnitude", arrays=arrays, train=True, seed=1)[3]
    t2, _, _ = RempeDataset(rows, "magnitude", arrays=arrays, train=True, seed=2)[3]
    check("train-mode augmentation varies with the seed", not torch.allclose(t1, t2))

    # --- label integrity through a real epoch ------------------------------
    # Regression test for the MPS async-copy race described in _epoch: an
    # epoch must hand back exactly the labels the dataset holds, on whatever
    # device this machine actually uses.
    from torch.utils.data import DataLoader

    ds_lab = RempeDataset(rows, "magnitude", arrays=arrays, train=False)
    dl_lab = DataLoader(ds_lab, batch_size=4, shuffle=False)
    dev = s03_train.pick_device("auto")
    mdl = build_model(1, pretrained=False).to(dev)
    crit = nn.CrossEntropyLoss(weight=torch.tensor([1.0, 17.0], device=dev))
    with torch.no_grad():
        ep = _epoch(mdl, dl_lab, dev, crit)
    check(f"epoch returns uncorrupted labels on {dev.type}",
          np.array_equal(np.sort(ep["labels"]), np.sort(ds_lab.labels)),
          f"got {np.unique(ep['labels'])[:5]}")
    check("epoch returns each row exactly once",
          sorted(ep["order"].tolist()) == list(range(len(ds_lab))))
    check("epoch probabilities are finite and in [0,1]",
          bool(np.isfinite(ep["probs"]).all()) and float(ep["probs"].min()) >= 0
          and float(ep["probs"].max()) <= 1)
    check("epoch loss is finite", np.isfinite(ep["loss"]))
    try:
        _assert_label_integrity(np.array([0, 1, 7]), ds_lab)
        check("integrity guard rejects out-of-range labels", False, "no exception")
    except RuntimeError:
        check("integrity guard rejects out-of-range labels", True)
    try:
        _assert_label_integrity(np.zeros(len(ds_lab), dtype=int), ds_lab)
        check("integrity guard rejects a wrong class balance", False, "no exception")
    except RuntimeError:
        check("integrity guard rejects a wrong class balance", True)

    # --- positional baseline ----------------------------------------------
    m = 40
    pos = pd.DataFrame({
        "subject_id": np.repeat([f"p{i}" for i in range(m)], 20),
        "slice": np.tile(np.arange(20), m),
    })
    pos["label"] = ((pos["slice"] >= 8) & (pos["slice"] <= 11)).astype(int)
    pos = add_relpos(pos)
    check("relpos spans [0,1]",
          abs(pos["relpos"].min()) < 1e-9 and abs(pos["relpos"].max() - 1) < 1e-9)
    tr = pos[pos.subject_id.isin([f"p{i}" for i in range(30)])]
    te = pos[~pos.subject_id.isin([f"p{i}" for i in range(30)])]
    sc = positional_scores(tr, te)
    auc = s04_stats.auc_midrank(te["label"].to_numpy(), sc)
    check("positional baseline recovers a purely positional label", auc > 0.99,
          f"auc {auc:.3f}")
    sweep = [s04_stats.auc_midrank(te["label"].to_numpy(), positional_scores(tr, te, n_bins=nb))
             for nb in (5, 10, 20, 30, 50)]
    check("positional baseline is stable across bin counts",
          max(sweep) - min(sweep) < 0.05, f"spread {max(sweep) - min(sweep):.3f}")
    check("positional baseline never peeks at the test labels",
          not te.index.isin(tr.index).any())
    agg = s04_stats.aggregate_by_cluster(te["label"].to_numpy(), sc,
                                         te["subject_id"].to_numpy(), how="mean")
    check("that same signal is undefined at patient level (every patient positive)",
          agg["n_pos_clusters"] == agg["n_clusters"])

    # --- position-stratified AUROC ----------------------------------------
    # A purely positional predictor must collapse to chance once position is
    # held fixed; a genuinely informative one must not.
    sc_pos = positional_scores(tr, te)
    st = stratified_auc(te["label"].to_numpy(), sc_pos, position_strata(te["relpos"], 10))
    check("stratifying on position collapses a positional predictor",
          not np.isfinite(st["auc"]) or abs(st["auc"] - 0.5) < 0.05,
          f"stratified auc {st['auc']:.3f}")
    # A predictor that tracks the label independently of position survives.
    rng2 = np.random.default_rng(3)
    lab2 = rng2.integers(0, 2, size=len(te))
    sc2 = lab2 + rng2.normal(0, 0.3, size=len(te))
    st2 = stratified_auc(lab2, sc2, position_strata(te["relpos"], 10))
    check("stratifying on position preserves a non-positional predictor",
          st2["auc"] > 0.9, f"stratified auc {st2['auc']:.3f}")
    st_one = stratified_auc(lab2, sc2, np.zeros(len(te), dtype=int))
    check("with one stratum the stratified AUROC is the plain AUROC",
          abs(st_one["auc"] - s04_stats.auc_midrank(lab2, sc2)) < 1e-9)
    check("single-class strata are skipped rather than counted",
          stratified_auc(np.zeros(10, dtype=int), np.arange(10.0),
                         np.zeros(10, dtype=int))["n_strata_used"] == 0)
    check("position_strata stays inside its bin range",
          set(np.unique(position_strata([0.0, 0.5, 1.0], 10)).tolist()) <= set(range(10)))

    # --- evaluate_predictions ---------------------------------------------
    lab = np.array([0, 0, 1, 1, 0, 1, 0, 1] * 5)
    prb = np.linspace(0, 1, len(lab))
    subj = np.repeat([f"s{i}" for i in range(10)], 4)
    ev = evaluate_predictions(lab, prb, subj, n_boot=200)
    check("evaluate_predictions returns every level",
          all(k in ev for k in ("slice_auc", "slice_ci_naive", "slice_ci_clustered",
                                "patient_auc", "patient_ci_clustered")))

    # The claim "the naive interval is too narrow" is not a theorem about
    # arbitrary data -- it holds when slices within a patient are positively
    # correlated, which is the actual situation. So the test has to generate
    # that situation rather than assert it universally. s04_stats._sim_clustered
    # is the study's exchangeable random-effects generator.
    ys, ss, pids = s04_stats._sim_clustered(np.random.default_rng(7),
                                            n_patients=40, slices=30, mu=1.0)
    ec = evaluate_predictions(ys, ss, pids.astype(str), n_boot=1000)
    nw = ec["slice_ci_naive"][1] - ec["slice_ci_naive"][0]
    cw = ec["slice_ci_clustered"][1] - ec["slice_ci_clustered"][0]
    check("on genuinely clustered data the naive slice CI is too narrow",
          cw > nw, f"clustered {cw:.3f} vs naive {nw:.3f}")
    check("patient-level AUROC is recovered on clustered data",
          abs(ec["patient_auc"] - ec["slice_auc"]) < 0.15,
          f"patient {ec['patient_auc']:.3f} vs slice {ec['slice_auc']:.3f}")

    # --- reported table ----------------------------------------------------
    check("every data_type has a reproduction target",
          all(REPRODUCTION_TARGET[d] in REPORTED for d in DATA_TYPES))
    check("headline is recorded as the gold-standard arm",
          REPORTED["headline_goldstandard_x2_image_plus_kspace"] == (86.1, 1.8))
    check("magnitude-only gold standard is recorded beside it",
          REPORTED["goldstandard_x0_image_only"][0] == 85.7)

    print(f"\n{'ALL PASS' if not failures else str(len(failures)) + ' FAILURE(S): ' + ', '.join(failures)}")
    return 1 if failures else 0


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__.split("Usage")[0],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--positional-baseline", action="store_true")
    p.add_argument("--run", action="store_true", help="train one arm")
    p.add_argument("--waterfall", action="store_true", help="run the full control ladder")
    p.add_argument("--report", action="store_true", help="print the waterfall from saved runs")
    p.add_argument("--arm", default="cv", choices=("cv", "official", "background",
                                                   "permuted", "acquisition"))
    p.add_argument("--data-type", default="magnitude_phase", choices=list(DATA_TYPES))
    p.add_argument("--acq-key", default="institution")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--workers", type=int, default=0)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--cache-dir", default=str(common.CACHE_DIR))
    p.add_argument("--out-dir", default=str(OUT_DIR))
    p.add_argument("--their-labels", default="",
                   help="path to their published dwi_slice_level_labels.csv")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if args.self_test:
        return self_test()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = Path(args.cache_dir)
    h5_path = cache_dir / f"{COHORT}.h5"
    index = load_index(cache_dir)

    cfg = TrainConfig(data_type=args.data_type, lr=args.lr, batch_size=args.batch_size,
                      epochs=args.epochs, patience=args.patience, seed=args.seed,
                      device=args.device, workers=args.workers)

    if args.positional_baseline:
        res = {"cache": run_positional_baseline(index, n_boot=args.n_boot)}
        if args.their_labels:
            res["their_labels"] = run_positional_baseline_on_their_labels(
                Path(args.their_labels), n_boot=args.n_boot)
        print(json.dumps(res, indent=2, default=float))
        (out_dir / "positional_baseline.json").write_text(json.dumps(res, indent=2, default=float))
        return 0

    if args.report:
        payload = json.loads((out_dir / f"waterfall_{args.data_type}.json").read_text())
        print_waterfall(build_waterfall(payload))
        return 0

    arms: dict = {}
    if args.run:
        if args.arm == "cv":
            arms["real"] = run_cv_arm(index, cfg, h5_path, n_boot=args.n_boot)
        elif args.arm == "official":
            res = train_arm(s03_train.normalize_splits(index.copy()), cfg, h5_path)
            arms["official"] = {"pooled": evaluate_run(res, index, args.n_boot), "folds": [res]}
        elif args.arm == "background":
            cfg.region = "background"
            arms["background"] = run_cv_arm(index, cfg, h5_path, n_boot=args.n_boot)
        elif args.arm == "permuted":
            arms["permuted"] = run_cv_arm(index, cfg, h5_path, permute=True, n_boot=args.n_boot)
        elif args.arm == "acquisition":
            key = s05_controls.resolve_confound_column(index, args.acq_key)
            for variant, idx in acquisition_split_indices(index, key):
                res = train_arm(idx, cfg, h5_path)
                arms[f"acq_{variant}"] = {"pooled": evaluate_run(res, index, args.n_boot),
                                          "folds": [res]}

    if args.waterfall:
        arms["real"] = run_cv_arm(index, cfg, h5_path, n_boot=args.n_boot)
        res = train_arm(s03_train.normalize_splits(index.copy()), cfg, h5_path)
        arms["official"] = {"pooled": evaluate_run(res, index, args.n_boot), "folds": [res]}
        bg = TrainConfig(**{**asdict(cfg), "region": "background"})
        arms["background"] = run_cv_arm(index, bg, h5_path, n_boot=args.n_boot)
        arms["permuted"] = run_cv_arm(index, cfg, h5_path, permute=True, n_boot=args.n_boot)
        try:
            key = s05_controls.resolve_confound_column(index, args.acq_key)
            for variant, idx in acquisition_split_indices(index, key):
                r = train_arm(idx, cfg, h5_path)
                arms[f"acq_{variant}"] = {"pooled": evaluate_run(r, index, args.n_boot),
                                          "folds": [r]}
        except RuntimeError as exc:
            logger.warning("acquisition split unavailable: %s", exc)

    payload = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohort": COHORT,
        "config": asdict(cfg),
        "reported": REPORTED,
        "reported_level": REPORTED_LEVEL,
        "reproduction_target": REPRODUCTION_TARGET[args.data_type],
        "arms": arms,
        "positional_baseline_cache": run_positional_baseline(index, n_boot=args.n_boot),
    }
    if args.their_labels:
        payload["positional_baseline_their_labels"] = run_positional_baseline_on_their_labels(
            Path(args.their_labels), n_boot=args.n_boot)

    name = "waterfall" if args.waterfall else f"arm_{args.arm}"
    path = out_dir / f"{name}_{args.data_type}.json"
    path.write_text(json.dumps(payload, indent=2, default=float))
    logger.info("wrote %s", path)
    print_waterfall(build_waterfall(payload))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
