"""
s02_prostate.py
---------------
Stage 2a of the PhaseDx pipeline: turn the fastMRI **prostate** raw k-space
(AXDIFF diffusion and AXT2 turbo-spin-echo) into the magnitude / phase / mask
cache defined by the CACHE CONTRACT.

Everything below was established empirically by reading the real files on
/Volumes/Research/fastmridatasets and validating every reconstruction step
against the vendor images that ship inside the same HDF5 files. The numbers
quoted in the comments are measured, not assumed.


WHAT THE RAW DATA ACTUALLY IS
=============================

AXDIFF (50 files, 45 readable)
------------------------------
  kspace            (50, S, C, 200, 150) complex64  (diffusion volume, slice, coil, kx, ky)
  coil_sens_maps    (S, C, 200, 150)     complex128
  calibration_data  (S, C, 200, 32)      complex64   <- 32 fully sampled ACS lines
  phase_correction  (50, S, C, 200, 2)   complex64   <- EPI navigator echoes
  adc_map, b50{x,y,z}, b1000{x,y,z}, b1500, trace_b50, trace_b1000  (S, 100, 100)

  *** THE KSPACE IS 2x UNDERSAMPLED AND THE ZEROS ARE STORED. ***
  Only the odd ky indices (1, 3, ... 149) carry data; 75 of the 150 lines are
  exactly zero (verified: `(|k|.sum(kx,coil) == 0).sum() == 75`). The ISMRMRD
  header confirms it: parallelImaging/accelerationFactor/kspace_encoding_step_1
  = 2, calibrationMode "other", with the ACS shipped separately as
  `calibration_data`. A plain ifft2 of that array produces an image with TWO
  COPIES of the pelvis folded on top of each other. Any pipeline that skips
  GRAPPA is training on aliased anatomy. We therefore run a real GRAPPA kernel
  (see grappa_weights / grappa_apply below) before the inverse FFT.

  EPI Nyquist-ghost correction: `phase_correction` holds two navigator echoes
  per (volume, slice, coil), stored in k-space along the readout. The
  echo-to-echo phase difference measured from them is already small
  (~0.05-0.2 rad, smooth), and applying the standard hybrid-space correction
  changes the correlation against the vendor images by <0.004 in either
  direction (measured, both polarity conventions, coil-combined and per-coil).
  Conclusion: the shipped k-space is ALREADY ghost-corrected. We do not apply
  it again; `--epi-phase-correction` exists so the decision can be re-tested.

  Diffusion structure: the 50 volumes are NOT 50 averages of one thing. Mean
  intensity and correlation against the shipped vendor images show a clean
  low-b / high-b split, e.g. for file 001 (correlation vs trace_b50):

      volumes  0- 4  r=0.74   low b  (b=50 / b=0 reference)
      volumes  5- 7  r=0.19   high b (b=1000)
      volumes  8-10  r=0.75   low b
      volumes 11-13  r=0.17   high b
      volumes 14-16  r=0.73   low b
      volumes 17-19  r=0.14   high b
      volumes 20-22  r=0.72   low b
      volumes 23-49  r<0.25   high b

  i.e. 14 low-b and 36 high-b volumes, interleaved in triplets (the three
  diffusion directions) for the first half of the scan and then a long block
  of extra b=1000 averages. Grouping the b=1000 volumes by their position in
  each triplet and averaging reproduces the direction ordering x, y, z
  (dir1 correlates best with b1000y, dir2 best with b1000z).

  HOW WE COLLAPSE THE 50 VOLUMES: we do not. We take exactly ONE low-b volume
  and derive both the magnitude and the phase from that single complex image.
  Reasons:
    * Complex-averaging across diffusion DIRECTIONS is invalid: every direction
      has its own eddy-current phase, so the mean is a phase-cancelled mush.
    * Complex-averaging across AVERAGES of the same direction is also invalid
      for DWI: each average carries a different bulk-motion-induced phase.
      This is exactly why clinical DWI is magnitude-averaged, never complex.
    * A single b=1000 average is unusable - it is visually pure noise at this
      matrix size (see the correlations above, r < 0.25). b=50 is the only
      diffusion contrast in this dataset with usable single-shot SNR.
    * Using one volume for BOTH channels guarantees the magnitude arm and the
      phase arm of the study see literally the same acquisition, which is the
      whole point of the comparison.
  The low-b set is re-detected per file (correlation against the shipped
  trace_b50 vs trace_b1000) so a file with a different volume ordering cannot
  silently poison the cache. The chosen index is recorded per slice in the
  index CSV as `dwi_volume`.

AXT2 (71 files, 67 readable)
----------------------------
  kspace             (3, S, C, 640, 451) complex64  (average, slice, coil, kx, ky)
  calibration_data   (S, C, 640, 32)     complex64
  reconstruction_rss (S, 320, 320)       float32

  The three "averages" are an INTERLEAVED acceleration, not three repeats:
      average 0 -> odd ky  (1, 3, ... 449)   225 lines
      average 1 -> even ky (2, 4, ... 450)   225 lines
      average 2 -> odd ky  (same lines as average 0)
  (header: accelerationFactor 2, calibrationMode "interleaved",
  interleavingDimension "average"). So no GRAPPA is needed for T2 - averages
  0 and 1 together tile the full 451-line grid.

  Are they phase coherent? Averages 0 and 2 occupy the SAME k-space lines, so
  they can be compared directly. Measured complex correlation over whole
  slices, several files: |r| = 0.94-0.97 with a residual global phase of
  |dphi| < 0.06 rad. They ARE coherent, so complex averaging is safe.
  (Averages 0 and 1 are trivially orthogonal - disjoint k-space support - so
  correlating them says nothing.)

  We use  k = (avg0 + avg2)/2 + avg1  and NOT  k.mean(axis=0). The naive mean
  puts twice as much amplitude on the odd lines as on the even lines, which is
  a Nyquist modulation and produces a FOV/2 ghost. Measured against the shipped
  reconstruction_rss: (avg0+avg2)/2+avg1 -> r = 0.998-0.999, avg0+avg1+avg2 ->
  r = 0.959-0.990, avg0 alone (aliased) -> r = 0.61-0.80.


GEOMETRY (calibrated against the vendor images, not guessed)
===========================================================
The ISMRMRD fieldOfView numbers in these files are not self-consistent, so the
image-domain mapping was solved by a scale + flip search against the shipped
vendor reconstructions and then verified on further files and slices.

  AXDIFF: image = ifft2c(GRAPPA(k))            -> (200, 150)
          resample readout 200 -> 172 (x0.86)  -> (172, 150), isotropic 2.33 mm
          flip the readout axis (np.flipud)
          the vendor 100x100 image is the CENTRE CROP of that.
          Verified: mean of the 12 low-b volumes, centre-cropped to 100x100,
          vs trace_b50: r = 0.946 - 0.997 over 4 files x 3 slices, and the
          optimal scale is 0.86 for every single one of them.

  AXT2:   zero-pad ky 451 -> 640 about the k-space centre (index 225 -> 320),
          ifft2c                                -> (640, 640), isotropic 0.5625 mm
          flip the readout axis
          the vendor 320x320 image is the CENTRE CROP of that.
          Verified vs reconstruction_rss: r = 0.998 - 0.9996 over 6 files.

  The flip is the same for HFS and FFS patients (checked both), so patient
  position does not need special handling - but it IS recorded as a confound.

  WE CACHE THE FULL FOV, NOT THE VENDOR CROP. The vendor FOV is all body and
  contains essentially no air (measured foreground fraction 0.69 for AXDIFF
  and 0.41 for AXT2, and the "mask" is then just bright-tissue-vs-dark-tissue,
  which is meaningless). The full FOV gives a real body mask (0.62 and 0.60,
  correct body outline) and is what makes the background-only falsification
  control possible at all. The vendor crop is still computed per slice, purely
  to record the validation correlation in the index CSV as `recon_ncc`.
  Price paid: the gland occupies fewer pixels than it would in the vendor FOV.

  The AXT2 640 -> 224 step is a 2.9x downsample and is done by CROPPING THE
  K-SPACE, not by interpolating the image: a k-space crop is an ideal
  anti-aliasing filter and is exact for phase. Interpolating a 640x640 image
  down to 224 with a linear kernel folds high spatial frequencies - including
  exactly the coil/noise texture a phase network would latch onto - back into
  the passband. AXDIFF needs no such care because 172x150 -> 224x224 is an
  upsample.


HOW TO READ THE TWO AXDIFF VALIDATION NUMBERS
=============================================
The vendor's trace_b50 is the geometric mean of three directions, each already
averaged - roughly 14 acquisitions. We cache ONE. Correlating one noisy average
against a 14-average reference cannot reach 1.0 no matter how correct the
reconstruction is, so both numbers are reported:

    per cached slice, single low-b volume vs trace_b50        mean r = 0.89
    per file, the same low-b volumes magnitude-averaged the
    way the vendor does, vs trace_b50                         mean r = 0.97

Measured slice by slice on file 001 (slices 2..29): single 0.806-0.969,
averaged 0.942-0.992. The second number is what says the GRAPPA kernel, the
geometry and the coil combination are right; the first is the SNR cost of the
deliberate decision not to complex-average diffusion data. If the second number
drops below 0.90 the run prints a capitalised warning, because then the
reconstruction really is broken.


COIL COMBINATION
================
  magnitude: ifft2c per coil -> common.rss                       (phase-blind, clinical)
  phase    : AXDIFF -> common.combine_with_sensitivities using the shipped
                       coil_sens_maps (optimal, removes per-coil phase offsets)
             AXT2   -> common.adaptive_virtual_coil (no maps are shipped)

Phase is kept in radians on [-pi, pi] and is only ever resampled through
sin/cos (common.resample_phase). It is never min-max scaled.


USAGE
=====
    python pipeline/s02_prostate.py --limit 2                 # quick smoke test
    python pipeline/s02_prostate.py --cohort prostate_t2
    python pipeline/s02_prostate.py                           # both cohorts, all files
    python pipeline/s02_prostate.py --cohort-csv pipeline_out/cohorts/prostate_dwi.csv
"""

from __future__ import annotations

import argparse
import logging
import re
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import (  # noqa: E402
    CACHE_DIR,
    COHORT_DIR,
    DATA_ROOT,
    OUT_ROOT,
    PROSTATE_LABEL_DIR,
    SUBJECT_COL,
    TARGET_HW,
    adaptive_virtual_coil,
    backfill_cache_subject_ids,
    body_mask,
    combine_with_sensitivities,
    ifft2c,
    normalize_magnitude,
    resample_phase,
    resample_to,
    rss,
    subject_id_lut,
)

logger = logging.getLogger("s02_prostate")

# --------------------------------------------------------------------------
# Calibrated geometry constants (see module docstring for how these were found)
# --------------------------------------------------------------------------

DWI_READOUT_SCALE = 0.86      # 200 readout px -> 172 px, gives isotropic 2.33 mm
DWI_VENDOR_HW = 100           # shipped b50x / trace_b1000 etc. are 100 x 100
T2_KY_PADDED = 640            # zero-pad 451 acquired ky lines onto a 640 grid
T2_VENDOR_HW = 320            # shipped reconstruction_rss is 320 x 320

# GRAPPA kernel geometry for R = 2: four source lines (grid offsets -3,-1,+1,+3)
# and five readout points predict the missing line at offset 0.
GRAPPA_KX_OFF = np.array([-2, -1, 0, 1, 2])
GRAPPA_KY_OFF = np.array([-3, -1, 1, 3])


# --------------------------------------------------------------------------
# GRAPPA
# --------------------------------------------------------------------------

def grappa_weights(acs: np.ndarray, lamda: float = 1e-4):
    """
    Fit GRAPPA interpolation weights from a fully sampled ACS block.

    acs : (C, X, Yacs) complex, fully sampled (fastMRI prostate ships 32 lines).

    Returns (W, rel_residual) where W has shape (C * 4 * 5, C): stacking the
    4x5 neighbourhood of every coil and multiplying by W predicts the missing
    line for every coil at once.

    The ACS is on the same k-space grid as the undersampled data, so the source
    offsets in grid units are identical in both, which is what makes the fitted
    kernel transferable.
    """
    acs = np.asarray(acs, dtype=np.complex128)
    n_coils, n_x, n_y = acs.shape
    ys = np.arange(-GRAPPA_KY_OFF.min(), n_y - GRAPPA_KY_OFF.max())
    xs = np.arange(-GRAPPA_KX_OFF.min(), n_x - GRAPPA_KX_OFF.max())
    if len(ys) < 4 or len(xs) < 4:
        raise ValueError(f"ACS block too small to calibrate GRAPPA: {acs.shape}")

    src = np.stack(
        [np.stack([acs[:, xs[None, :] + dx, ys[:, None] + dy] for dx in GRAPPA_KX_OFF], axis=1)
         for dy in GRAPPA_KY_OFF],
        axis=1,
    )
    a = src.reshape(n_coils * len(GRAPPA_KY_OFF) * len(GRAPPA_KX_OFF), -1).T
    tgt = acs[:, xs[None, :], ys[:, None]].reshape(n_coils, -1).T

    aha = a.conj().T @ a
    # Tikhonov regularisation scaled to the problem so it behaves the same for
    # 10-coil and 30-coil exams (the coil count varies from 10 to 30 here).
    ridge = lamda * np.trace(aha).real / aha.shape[0]
    w = np.linalg.solve(aha + ridge * np.eye(aha.shape[0]), a.conj().T @ tgt)
    resid = float(np.linalg.norm(a @ w - tgt) / (np.linalg.norm(tgt) + 1e-30))
    return w.astype(np.complex64), resid


def grappa_apply(kspace: np.ndarray, weights: np.ndarray) -> np.ndarray:
    """
    Fill the zeroed lines of an R=2 undersampled k-space with fitted weights.

    kspace : (C, X, Y) complex, missing lines are exactly zero.
    Returns the same shape with every line populated.
    """
    ks = np.asarray(kspace, dtype=np.complex64)
    n_coils, n_x, n_y = ks.shape

    line_energy = np.abs(ks).sum(axis=(0, 1))
    acquired = np.nonzero(line_energy)[0]
    missing = np.nonzero(line_energy == 0)[0]
    if missing.size == 0:
        return ks
    step = np.unique(np.diff(acquired))
    if step.size != 1 or step[0] != 2:
        raise ValueError(f"expected R=2 uniform sampling, got line spacing {step}")

    pad = np.zeros((n_coils, n_x + 4, n_y + 6), np.complex64)
    pad[:, 2:2 + n_x, 3:3 + n_y] = ks

    xs = np.arange(n_x)
    src = np.stack(
        [np.stack([pad[:, xs[None, :] + 2 + dx, missing[:, None] + 3 + dy]
                   for dx in GRAPPA_KX_OFF], axis=1)
         for dy in GRAPPA_KY_OFF],
        axis=1,
    )
    src = src.reshape(n_coils * len(GRAPPA_KY_OFF) * len(GRAPPA_KX_OFF), -1).T
    filled = (src @ weights).T.reshape(n_coils, len(missing), n_x).transpose(0, 2, 1)

    out = ks.copy()
    out[:, :, missing] = filled
    return out


# --------------------------------------------------------------------------
# EPI ghost correction (measured to be a no-op on this data; kept for audit)
# --------------------------------------------------------------------------

def _ifft1c(a, axis):
    return np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(a, axes=axis), axis=axis, norm="ortho"),
                           axes=axis)


def _fft1c(a, axis):
    return np.fft.fftshift(np.fft.fft(np.fft.ifftshift(a, axes=axis), axis=axis, norm="ortho"),
                           axes=axis)


def epi_ghost_correct(kspace: np.ndarray, navigators: np.ndarray, sign: int = 1) -> np.ndarray:
    """
    Classic navigator-based EPI Nyquist ghost correction.

    kspace     : (C, X, Y) with zeros on unacquired lines
    navigators : (C, X, 2), two navigator echoes stored in k-space along X

    The two navigators are acquired with opposite readout gradient polarity;
    their phase difference in hybrid (x, ky) space is the odd/even echo phase
    error, applied as +/- half to alternating acquired lines.

    On the fastMRI prostate data this changes the correlation with the vendor
    images by less than 0.004, i.e. the shipped k-space is already corrected.
    """
    nav = _ifft1c(np.asarray(navigators, np.complex128), 1)
    dphi = np.angle((nav[..., 0] * np.conj(nav[..., 1])).sum(axis=0))[None, :, None]

    hyb = _ifft1c(np.asarray(kspace, np.complex128), 1)
    acquired = np.nonzero(np.abs(kspace).sum(axis=(0, 1)))[0]
    polarity = np.zeros(kspace.shape[2])
    polarity[acquired[0::2]] = 1.0
    polarity[acquired[1::2]] = -1.0
    hyb = hyb * np.exp(1j * sign * 0.5 * dphi * polarity[None, None, :])
    return _fft1c(hyb, 1).astype(np.complex64)


# --------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------

def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two same-shape real images."""
    a = np.asarray(a, np.float64).ravel() - np.mean(a)
    b = np.asarray(b, np.float64).ravel() - np.mean(b)
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def center_crop2d(arr: np.ndarray, size) -> np.ndarray:
    """
    Centre crop the last two axes. `size` is an int or an (h, w) pair.

    The start offset uses floor((n - m) / 2), which keeps the DC sample of an
    even-length k-space axis at index m // 2 - exactly where ifftshift expects
    it - so this is safe to use on k-space as well as on images.
    """
    hw = (size, size) if isinstance(size, int) else tuple(size)
    for axis, want in zip((-2, -1), hw):
        have = arr.shape[axis]
        start = (have - want) // 2
        arr = np.take(arr, range(start, start + want), axis=axis)
    return arr


def pad_kspace_ky(kspace: np.ndarray, n: int) -> np.ndarray:
    """
    Zero-pad the last (ky) axis to n samples keeping the k-space centre aligned.

    Zero-padding in k-space is exact sinc interpolation in image space, which
    is why we do the T2 ky upsampling here rather than interpolating the image
    (interpolating a wrapped phase map is what we are trying to avoid).
    """
    n_y = kspace.shape[-1]
    left = n // 2 - n_y // 2
    right = n - n_y - left
    if left < 0 or right < 0:
        raise ValueError(f"cannot pad ky {n_y} -> {n}")
    return np.pad(kspace, [(0, 0)] * (kspace.ndim - 1) + [(left, right)])


def header_fields(h5file) -> dict:
    """Pull the acquisition metadata that the confound analysis needs."""
    out = {}
    try:
        raw = h5file["ismrmrd_header"][()]
    except Exception:  # noqa: BLE001
        return out
    raw = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)
    tags = ["systemModel", "institutionName", "deviceID", "systemFieldStrength_T",
            "receiverChannels", "protocolName", "patientPosition", "TR", "TE",
            "echo_spacing", "measurementID"]
    for tag in tags:
        m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.S)
        if m:
            out[tag] = m.group(1).strip()
    return out


def open_h5(path: Path):
    """
    Open and *probe* an HDF5 file.

    Several prostate files on this drive are corrupt in a way that lets
    h5py.File() succeed and then raises KeyError on the first dataset access
    ("free block size is zero?"). 5 of 50 AXDIFF and 4 of 71 AXT2 files are
    like this. Probing here means the caller sees a clean failure.
    """
    f = h5py.File(path, "r")
    try:
        _ = f["kspace"].shape
        _ = f["ismrmrd_header"][()]
    except Exception:
        f.close()
        raise
    return f


# --------------------------------------------------------------------------
# reconstruction: prostate DWI
# --------------------------------------------------------------------------

def dwi_slice_recon(f, slice_idx: int, volume: int, weights, sens=None,
                    epi_correct: bool = False, mag_only: bool = False):
    """
    Reconstruct one AXDIFF slice of one diffusion volume.

    Returns (magnitude_full_fov, phase_full_fov) both (172, 150) in the display
    orientation (readout axis flipped). With mag_only=True the phase is None
    and the sensitivity combination is skipped - used by the low-b detector,
    which only needs magnitudes and would otherwise pay for 50 phase resamples.

    Both the 200 -> 172 readout resample and the later 172x150 -> 224x224 step
    are net upsamples or a mild 0.86 shrink, so ordinary interpolation is safe
    here; phase still goes through sin/cos (common.resample_phase) so a wrap
    never gets averaged with its neighbour.
    """
    ks = f["kspace"][volume, slice_idx].astype(np.complex64)
    if epi_correct:
        ks = epi_ghost_correct(ks, f["phase_correction"][volume, slice_idx], sign=1)
    coil_images = ifft2c(grappa_apply(ks, weights))

    mag = rss(coil_images, coil_axis=0)
    n_x = mag.shape[0]
    target = (int(round(n_x * DWI_READOUT_SCALE)), mag.shape[1])
    mag = np.flipud(resample_to(mag, target, order=1))
    if mag_only:
        return mag, None

    complex_image = combine_with_sensitivities(coil_images, sens, coil_axis=0)
    phase = np.flipud(resample_phase(np.angle(complex_image), target))
    return mag, phase


def dwi_detect_lowb(f, weights, slice_idx: int, margin: float = 0.15):
    """
    Work out which of the 50 diffusion volumes are the low-b ones.

    Every volume is reconstructed at one slice, centre-cropped to the vendor
    100x100 FOV, and correlated against the shipped trace_b50 and trace_b1000
    images. Low-b volumes correlate ~0.7-0.8 with trace_b50; b=1000 volumes are
    single-average noise and correlate <0.3 with anything.

    Also returns the correlation obtained when the detected low-b volumes are
    MAGNITUDE-averaged the way the vendor's trace_b50 is. That number is the
    honest "is the reconstruction correct" check: it removes the SNR penalty
    of using a single average and isolates GRAPPA / geometry / coil combination.
    Measured on this data it is ~0.97, versus ~0.89 for one volume, so a 0.89
    per-slice value means "one average is noisy", not "the recon is wrong".

    Returns (low_b_indices, r_vs_b50, r_vs_b1000, r_lowb_mean_vs_b50).
    """
    ref50 = np.asarray(f["trace_b50"][slice_idx], np.float64)
    ref1k = np.asarray(f["trace_b1000"][slice_idx], np.float64)
    n_vol = f["kspace"].shape[0]

    low_b, r50, r1k = [], [], []
    accum = None
    for v in range(n_vol):
        mag, _ = dwi_slice_recon(f, slice_idx, v, weights, mag_only=True)
        crop = center_crop2d(mag, DWI_VENDOR_HW)
        a, b = ncc(crop, ref50), ncc(crop, ref1k)
        r50.append(a)
        r1k.append(b)
        if a > b + margin and a > 0.5:
            low_b.append(v)
            accum = crop.copy() if accum is None else accum + crop
    r_mean = ncc(accum / len(low_b), ref50) if low_b else float("nan")
    return low_b, r50, r1k, r_mean


# --------------------------------------------------------------------------
# reconstruction: prostate T2
# --------------------------------------------------------------------------

def t2_combine_averages(k: np.ndarray, coherent: bool) -> np.ndarray:
    """
    Merge the three interleaved T2 'averages' into one fully sampled k-space.

    avg0 and avg2 sample the same (odd) lines, avg1 samples the complementary
    (even) lines. (avg0 + avg2)/2 + avg1 keeps every line at the same amplitude;
    plain k.mean(0) does not and rings a Nyquist ghost.

    If the coherence check failed we fall back to avg0 + avg1, which is still a
    fully sampled, phase-consistent k-space - it just throws away avg2's SNR.
    """
    if k.shape[0] < 3:
        return k.sum(axis=0)
    if coherent:
        return (k[0] + k[2]) / 2.0 + k[1]
    return k[0] + k[1]


def t2_average_coherence(f, slice_idx: int) -> tuple:
    """
    Test whether averages 0 and 2 (which share k-space lines) are phase coherent.

    Returns (|r|, mean phase offset in radians). Measured on this dataset:
    |r| = 0.94-0.97, |dphi| < 0.06 rad, i.e. coherent.
    """
    if f["kspace"].shape[0] < 3:
        # Nothing to compare: only the interleaved pair is present.
        return 0.0, 0.0
    k0 = f["kspace"][0, slice_idx].astype(np.complex64)
    k2 = f["kspace"][2, slice_idx].astype(np.complex64)
    i0, i2 = ifft2c(k0).ravel(), ifft2c(k2).ravel()
    denom = np.linalg.norm(i0) * np.linalg.norm(i2)
    if denom == 0:
        return 0.0, 0.0
    r = np.vdot(i2, i0) / denom
    return float(abs(r)), float(np.angle(r))


def t2_slice_recon(f, slice_idx: int, coherent: bool, target_hw):
    """
    Reconstruct one AXT2 slice.

    Returns (magnitude_at_native_640, magnitude_at_target, phase_at_target),
    all in display orientation. The 640x640 magnitude exists only so the
    vendor-crop validation correlation can be computed at native resolution.

    The 640 -> 224 step is a 2.9x DOWNSAMPLE, so it is done by cropping the
    k-space rather than by interpolating the image: a k-space crop is an ideal
    anti-aliasing low-pass and it is exact for phase (no interpolation of a
    wrapped quantity at all). Interpolating a 640x640 image straight down to
    224 with a linear kernel would fold high spatial frequencies - including
    the coil/noise texture that a phase network is most likely to latch onto -
    back into the passband.
    """
    k = f["kspace"][:, slice_idx].astype(np.complex64)
    merged = pad_kspace_ky(t2_combine_averages(k, coherent), T2_KY_PADDED)

    mag_native = np.flipud(rss(ifft2c(merged), coil_axis=0))

    coil_small = ifft2c(center_crop2d(merged, tuple(target_hw)))
    mag = np.flipud(rss(coil_small, coil_axis=0))
    # No sensitivity maps ship with AXT2, so the phase must come from a
    # sensitivity-free complex combination.
    phase = np.flipud(np.angle(adaptive_virtual_coil(coil_small, coil_axis=0)))
    return mag_native, mag, phase


# --------------------------------------------------------------------------
# labels
# --------------------------------------------------------------------------

def load_slice_labels(cohort: str) -> pd.DataFrame:
    name = "dwi_slice_level_labels.csv" if cohort == "prostate_dwi" else "t2_slice_level_labels.csv"
    path = PROSTATE_LABEL_DIR / name
    df = pd.read_csv(path)
    df = df.rename(columns={"Unnamed: 0": "_row"})
    # `slice` in the CSV is 1-based; the HDF5 slice axis is 0-based.
    df["slice0"] = df["slice"].astype(int) - 1
    return df


def cross_check_labels(cohort_csv, labels: pd.DataFrame, threshold: int) -> None:
    """
    If stage 1 already binarised the labels, make sure we agree with it.

    Two stages independently turning PI-RADS into 0/1 with different cut-offs
    would be a silent, untraceable disagreement between the cache and whatever
    stage 1 handed downstream, so we compare and shout rather than assume.
    """
    if cohort_csv is None or not Path(cohort_csv).exists():
        return
    try:
        df = pd.read_csv(cohort_csv)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not cross-check labels against %s: %s", cohort_csv, exc)
        return
    if not {"file", "slice", "label", "raw_label"}.issubset(df.columns):
        return
    theirs = df.dropna(subset=["raw_label"])
    mismatch = theirs[theirs["label"].astype(int)
                      != (theirs["raw_label"].astype(int) >= threshold).astype(int)]
    if len(mismatch):
        logger.error("LABEL DISAGREEMENT with %s: %d/%d rows. Stage 1 used a different "
                     "PI-RADS cut-off than --pirads-threshold=%d. Fix one of them before "
                     "trusting any downstream result.",
                     Path(cohort_csv).name, len(mismatch), len(theirs), threshold)
    else:
        logger.info("label binarisation agrees with %s (PI-RADS >= %d)",
                    Path(cohort_csv).name, threshold)


def discover_files(cohort: str, cohort_csv: Path | None) -> list:
    """
    Build the file list. Prefer a stage-1 cohort CSV when one exists, otherwise
    enumerate the drive directly so this stage can run before stage 1.
    """
    root = DATA_ROOT / "prostate"
    pattern = ("fastMRI_prostate_DIFF_IDS_*/file_prostate_AXDIFF_*.h5"
               if cohort == "prostate_dwi" else
               "fastMRI_prostate_T2_IDS_*/file_prostate_AXT2_*.h5")
    # AppleDouble sidecars (._*.h5) are not HDF5 and must never be opened.
    on_disk = sorted(p for p in root.glob(pattern) if not p.name.startswith("._"))

    if cohort_csv is None or not cohort_csv.exists():
        return on_disk

    df = pd.read_csv(cohort_csv)
    col = next((c for c in ("path", "file_path", "filepath", "file") if c in df.columns), None)
    if col is None:
        raise SystemExit(f"--cohort-csv {cohort_csv} has no path/file/filepath column")
    by_name = {p.name: p for p in on_disk}
    paths, missing = [], []
    for value in df[col].astype(str):
        p = Path(value)
        if p.is_absolute() and p.exists():
            paths.append(p)
        elif p.name in by_name:
            paths.append(by_name[p.name])
        else:
            missing.append(value)
    if missing:
        logger.warning("cohort CSV lists %d rows whose file is not on disk, e.g. %s",
                       len(missing), sorted(set(missing))[:3])
    # Stage-1 cohort CSVs are slice-level, so many rows point at the same file.
    unique = sorted(set(paths))
    logger.info("cohort CSV %s: %d rows -> %d distinct files on disk",
                cohort_csv.name, len(df), len(unique))
    return unique


# --------------------------------------------------------------------------
# incremental cache writer
# --------------------------------------------------------------------------

class CacheWriter:
    """
    Appends slices to <cohort>.h5 and rewrites <cohort>_index.csv after every
    file, so a crash (or an unplugged drive) costs at most one exam.

    The index carries `subject_id`, joined from the stage-1 cohort CSV on file
    basename. That column is the unit stage 3 enforces splits on, and it is NOT
    derivable from the cache alone -- for breast it collapses repeat-scan groups
    that live only in the label sheet. Prostate has one exam per subject so the
    join is an identity here, but stage 3 requires the column from every cohort:
    a contract that only some caches honour is a contract nothing can rely on.
    """

    def __init__(self, cache_dir: Path, cohort: str, target_hw=TARGET_HW,
                 cohort_dir=None):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cohort = cohort
        self.h5_path = cache_dir / f"{cohort}.h5"
        self.csv_path = cache_dir / f"{cohort}_index.csv"
        # Fail now, not after hours of reconstruction: if the stage-1 table is
        # missing there is no sanctioned source of subject_id and the cache we
        # are about to build would be unusable by stage 3.
        self.subject_lut = subject_id_lut(cohort, cohort_dir)
        self.rows = []
        self.f = h5py.File(self.h5_path, "w")
        h, w = target_hw
        opts = dict(maxshape=(None, h, w), chunks=(1, h, w), compression="lzf")
        self.f.create_dataset("mag", (0, h, w), dtype="float16", **opts)
        self.f.create_dataset("phase", (0, h, w), dtype="float16", **opts)
        self.f.create_dataset("mask", (0, h, w), dtype=np.bool_, **opts)

    def append(self, mag, phase, mask, row: dict):
        n = self.f["mag"].shape[0]
        for key, value, dtype in (("mag", mag, np.float16),
                                  ("phase", phase, np.float16),
                                  ("mask", mask, np.bool_)):
            ds = self.f[key]
            ds.resize(n + 1, axis=0)
            ds[n] = value.astype(dtype)
        row = dict(row)
        row["idx"] = n
        self.rows.append(row)

    def subject_id_for(self, file_name: str) -> str:
        """Stage-1 subject_id for a cached file, or raise."""
        bn = Path(str(file_name)).name
        try:
            return self.subject_lut[bn]
        except KeyError:
            raise KeyError(
                f"[{self.cohort}] {bn} has no row in the stage-1 cohort CSV, so it has "
                "no subject_id and stage 3 could not enforce splits on it. Re-run "
                "stage 1 over the same file set before caching."
            ) from None

    def flush(self):
        self.f.flush()
        if self.rows:
            cols = ["idx", "patient_id", SUBJECT_COL, "file", "slice", "label",
                    "raw_label", "official_split", "folder", "acq"]
            df = pd.DataFrame(self.rows)
            df[SUBJECT_COL] = [self.subject_id_for(v) for v in df["file"]]
            ordered = cols + [c for c in df.columns if c not in cols]
            df[ordered].to_csv(self.csv_path, index=False)

    def close(self):
        self.flush()
        self.f.close()


# --------------------------------------------------------------------------
# per-cohort drivers
# --------------------------------------------------------------------------

def finalize(mag_full, phase_full, target_hw):
    """
    Normalise, resample to the network grid, and build the body mask.

    resample_to / resample_phase are no-ops when the input is already at
    target_hw (the T2 path arrives pre-sized via a k-space crop).
    """
    mag = resample_to(normalize_magnitude(mag_full), target_hw, order=1)
    mag = np.clip(mag, 0.0, 1.0)
    phase = resample_phase(phase_full, target_hw)
    return mag, phase, body_mask(mag)


def process_dwi(paths, labels, writer, args):
    stats = {"files": 0, "slices": 0, "failed": [], "ncc": [], "ncc_lowb_mean": []}
    for path in paths:
        try:
            f = open_h5(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("UNREADABLE %s (%s: %s)", path.name, type(exc).__name__, exc)
            stats["failed"].append((path.name, f"{type(exc).__name__}"))
            continue

        t0 = time.time()
        with f:
            n_vol, n_slices, n_coils = f["kspace"].shape[:3]
            meta = header_fields(f)
            mid = n_slices // 2

            # GRAPPA weights come from the ACS and depend only on the coils, so
            # they are fitted once per slice and reused for every volume.
            weights_mid, resid = grappa_weights(f["calibration_data"][mid])

            if args.dwi_volume == "auto":
                low_b, r50, _, r_lowb_mean = dwi_detect_lowb(f, weights_mid, mid)
                if not low_b:
                    logger.error("%s: no low-b volume detected (max r vs trace_b50 = %.3f) - SKIPPED",
                                 path.name, max(r50))
                    stats["failed"].append((path.name, "no_lowb_detected"))
                    continue
                # Prefer a FIXED volume index across the whole cohort so that
                # "which diffusion volume" cannot become a per-patient variable
                # (and therefore a confound). Fall back to the first detected
                # low-b volume only when the preferred one is not low-b in this
                # exam, and shout about it when that happens.
                if args.dwi_preferred_volume in low_b:
                    volume = args.dwi_preferred_volume
                else:
                    volume = low_b[0]
                    logger.warning("%s: preferred volume %d is not low-b here "
                                   "(r vs trace_b50 = %.3f); falling back to volume %d",
                                   path.name, args.dwi_preferred_volume,
                                   r50[args.dwi_preferred_volume], volume)
                logger.info("%s: %d/%d volumes classified low-b %s -> using volume %d "
                            "(low-b MEAN vs trace_b50 r=%.3f: recon fidelity without the "
                            "single-average noise penalty)",
                            path.name, len(low_b), n_vol, low_b[:16], volume, r_lowb_mean)
                stats["ncc_lowb_mean"].append(r_lowb_mean)
            else:
                volume = int(args.dwi_volume)
                low_b = [volume]
                r_lowb_mean = float("nan")

            file_rows = labels[labels["fastmri_rawfile"] == path.name]
            by_slice = {int(r.slice0): r for r in file_rows.itertuples()}

            for s in range(n_slices):
                label_row = by_slice.get(s)
                if label_row is None:
                    logger.warning("%s slice %d has no label row - skipped", path.name, s)
                    continue
                weights, _ = (grappa_weights(f["calibration_data"][s])
                              if s != mid else (weights_mid, resid))
                sens = f["coil_sens_maps"][s]
                mag_full, phase_full = dwi_slice_recon(
                    f, s, volume, weights, sens, epi_correct=args.epi_phase_correction)

                r = ncc(center_crop2d(mag_full, DWI_VENDOR_HW), f["trace_b50"][s])
                stats["ncc"].append(r)

                mag, phase, mask = finalize(mag_full, phase_full, tuple(args.target_hw))
                pirads = int(label_row.PIRADS)
                writer.append(mag, phase, mask, dict(
                    patient_id=int(label_row.fastmri_pt_id),
                    file=path.name,
                    slice=s,
                    label=int(pirads >= args.pirads_threshold),
                    raw_label=pirads,
                    official_split=str(label_row.data_split),
                    folder=str(label_row.folder),
                    acq="AXDIFF",
                    dwi_volume=volume,
                    n_lowb_volumes=len(low_b),
                    recon_ncc=round(r, 4),
                    recon_ncc_lowb_mean=round(r_lowb_mean, 4),
                    grappa_acs_resid=round(resid, 4),
                    n_coils=int(n_coils),
                    n_slices=int(n_slices),
                    kx=int(f["kspace"].shape[3]),
                    ky=int(f["kspace"].shape[4]),
                    scanner=meta.get("systemModel", ""),
                    device_id=meta.get("deviceID", ""),
                    institution=meta.get("institutionName", ""),
                    field_strength=meta.get("systemFieldStrength_T", ""),
                    protocol=meta.get("protocolName", ""),
                    patient_position=meta.get("patientPosition", ""),
                    tr=meta.get("TR", ""),
                    te=meta.get("TE", ""),
                ))
                stats["slices"] += 1

        writer.flush()
        stats["files"] += 1
        recent = stats["ncc"][-n_slices:]
        logger.info("%s done: %d slices in %.1fs, recon_ncc vs trace_b50 mean=%.3f min=%.3f",
                    path.name, len(recent), time.time() - t0,
                    float(np.mean(recent)) if recent else float("nan"),
                    float(np.min(recent)) if recent else float("nan"))
    return stats


def process_t2(paths, labels, writer, args):
    stats = {"files": 0, "slices": 0, "failed": [], "ncc": [], "coherence": []}
    for path in paths:
        try:
            f = open_h5(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("UNREADABLE %s (%s: %s)", path.name, type(exc).__name__, exc)
            stats["failed"].append((path.name, f"{type(exc).__name__}"))
            continue

        t0 = time.time()
        with f:
            n_avg, n_slices, n_coils = f["kspace"].shape[:3]
            meta = header_fields(f)
            mid = n_slices // 2

            coh_r, coh_dphi = t2_average_coherence(f, mid)
            coherent = (coh_r >= args.coherence_threshold) and (abs(coh_dphi) < 0.3)
            stats["coherence"].append((path.name, coh_r, coh_dphi, coherent))
            logger.info("%s: average0 vs average2 complex |r|=%.3f dphi=%+.3f rad -> %s",
                        path.name, coh_r, coh_dphi,
                        "coherent, complex-averaging avg0/avg2"
                        if coherent else "INCOHERENT, using avg0+avg1 only")

            file_rows = labels[labels["fastmri_rawfile"] == path.name]
            by_slice = {int(r.slice0): r for r in file_rows.itertuples()}

            for s in range(n_slices):
                label_row = by_slice.get(s)
                if label_row is None:
                    logger.warning("%s slice %d has no label row - skipped", path.name, s)
                    continue
                mag_native, mag_small, phase_small = t2_slice_recon(
                    f, s, coherent, tuple(args.target_hw))

                r = ncc(center_crop2d(mag_native, T2_VENDOR_HW), f["reconstruction_rss"][s])
                stats["ncc"].append(r)

                mag, phase, mask = finalize(mag_small, phase_small, tuple(args.target_hw))
                pirads = int(label_row.PIRADS)
                writer.append(mag, phase, mask, dict(
                    patient_id=int(label_row.fastmri_pt_id),
                    file=path.name,
                    slice=s,
                    label=int(pirads >= args.pirads_threshold),
                    raw_label=pirads,
                    official_split=str(label_row.data_split),
                    folder=str(label_row.folder),
                    acq="AXT2",
                    recon_ncc=round(r, 4),
                    avg_coherence=round(coh_r, 4),
                    avg_phase_offset=round(coh_dphi, 4),
                    averages_used="(a0+a2)/2+a1" if coherent else "a0+a1",
                    n_coils=int(n_coils),
                    n_slices=int(n_slices),
                    kx=int(f["kspace"].shape[3]),
                    ky=int(f["kspace"].shape[4]),
                    scanner=meta.get("systemModel", ""),
                    device_id=meta.get("deviceID", ""),
                    institution=meta.get("institutionName", ""),
                    field_strength=meta.get("systemFieldStrength_T", ""),
                    protocol=meta.get("protocolName", ""),
                    patient_position=meta.get("patientPosition", ""),
                    tr=meta.get("TR", ""),
                    te=meta.get("TE", ""),
                ))
                stats["slices"] += 1

        writer.flush()
        stats["files"] += 1
        recent = stats["ncc"][-n_slices:]
        logger.info("%s done: %d slices in %.1fs, recon_ncc vs reconstruction_rss mean=%.4f min=%.4f",
                    path.name, len(recent), time.time() - t0,
                    float(np.mean(recent)) if recent else float("nan"),
                    float(np.min(recent)) if recent else float("nan"))
    return stats


# --------------------------------------------------------------------------

def report(cohort: str, stats: dict, writer: CacheWriter, threshold: float):
    print("\n" + "=" * 72)
    print(f"COHORT {cohort}")
    print("=" * 72)
    print(f"  files processed : {stats['files']}")
    print(f"  slices cached   : {stats['slices']}")
    if stats["failed"]:
        print(f"  files skipped   : {len(stats['failed'])}")
        for name, why in stats["failed"]:
            print(f"      {name}: {why}")
    if stats["ncc"]:
        arr = np.array(stats["ncc"])
        ref = "trace_b50" if cohort == "prostate_dwi" else "reconstruction_rss"
        print(f"  RECONSTRUCTION VALIDATION (correlation vs the vendor {ref} in the same file)")
        print(f"    per cached slice (what we actually store):")
        print(f"      n={arr.size}  mean={arr.mean():.4f}  median={np.median(arr):.4f} "
              f"min={arr.min():.4f}  max={arr.max():.4f}")
        bad = int((arr < threshold).sum())
        if bad:
            print(f"      {bad}/{arr.size} slices below {threshold:.2f}")
        else:
            print(f"      all slices >= {threshold:.2f}")
    lowb = np.array([v for v in stats.get("ncc_lowb_mean", []) if np.isfinite(v)])
    if lowb.size:
        print(f"    per file, low-b volumes magnitude-averaged as the vendor does")
        print(f"    (isolates GRAPPA + geometry + coil combination from single-average noise):")
        print(f"      n={lowb.size}  mean={lowb.mean():.4f}  median={np.median(lowb):.4f} "
              f"min={lowb.min():.4f}  max={lowb.max():.4f}")
        if lowb.mean() < 0.90:
            print(f"      *** MEAN BELOW 0.90 -- the reconstruction itself is suspect, not just SNR ***")
        else:
            print(f"      reconstruction validated (>=0.90 mean)")
    if writer.rows:
        df = pd.DataFrame(writer.rows)
        print(f"  patients        : {df.patient_id.nunique()}")
        print(f"  label balance   : {df.label.value_counts().to_dict()}")
        print(f"  raw PIRADS      : {df.raw_label.value_counts().sort_index().to_dict()}")
        print(f"  official split  : {df.official_split.value_counts().to_dict()}")
        print(f"  cache           : {writer.h5_path}")
        print(f"  index           : {writer.csv_path}")


def main():
    ap = argparse.ArgumentParser(description="PhaseDx stage 2: prostate DWI + T2 preprocessing")
    ap.add_argument("--cohort", choices=["prostate_dwi", "prostate_t2", "both"], default="both")
    ap.add_argument("--limit", type=int, default=None, help="only process the first N files per cohort")
    ap.add_argument("--cohort-csv", type=Path, default=None,
                    help="stage-1 cohort CSV; defaults to pipeline_out/cohorts/<cohort>.csv if present")
    ap.add_argument("--out", type=Path, default=CACHE_DIR)
    ap.add_argument("--pirads-threshold", type=int, default=3,
                    help="PIRADS >= this becomes label 1 (raw PIRADS is always kept in raw_label)")
    ap.add_argument("--dwi-volume", default="auto",
                    help="'auto' (detect the low-b volumes per file) or an explicit volume index")
    ap.add_argument("--dwi-preferred-volume", type=int, default=0,
                    help="which low-b volume to take when --dwi-volume auto; kept fixed across "
                         "the cohort so volume choice cannot become a per-patient confound")
    ap.add_argument("--epi-phase-correction", action="store_true",
                    help="re-apply navigator EPI ghost correction (measured to be a no-op here)")
    ap.add_argument("--coherence-threshold", type=float, default=0.80,
                    help="min |complex correlation| between T2 averages 0 and 2 to allow complex averaging")
    ap.add_argument("--ncc-warn", type=float, default=0.80,
                    help="flag slices whose validation correlation falls below this")
    ap.add_argument("--target-hw", type=int, nargs=2, default=list(TARGET_HW))
    ap.add_argument("--cohort-dir", type=Path, default=COHORT_DIR,
                    help="stage-1 output directory; the source of subject_id")
    ap.add_argument("--backfill-subject-ids", action="store_true",
                    help="add subject_id to an EXISTING <cohort>_index.csv and exit. "
                         "Touches only the CSV -- the validated HDF5 caches (hours of "
                         "reconstruction, r=0.998/0.984 vs vendor) are never reopened. "
                         "Idempotent: re-running a backfilled index is a no-op.")
    ap.add_argument("--log-level", default="INFO")
    args = ap.parse_args()

    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    cohorts = ["prostate_dwi", "prostate_t2"] if args.cohort == "both" else [args.cohort]

    if args.backfill_subject_ids:
        # Deliberately before the DATA_ROOT check: the backfill reads two CSVs
        # and never touches the drive, so it must work with the drive unplugged.
        for cohort in cohorts:
            rep = backfill_cache_subject_ids(cohort, Path(args.out), args.cohort_dir)
            print(f"[{cohort}] {rep['status']}: {rep['rows']} rows, "
                  f"{rep['n_patients']} patient_id -> {rep['n_subjects']} subject_id "
                  f"({rep['path']})")
            if rep["subjects_spanning_splits"]:
                print(f"[{cohort}] WARNING: subjects spanning splits: "
                      f"{rep['subjects_spanning_splits']}")
        return

    if not DATA_ROOT.exists():
        raise SystemExit(f"data root not mounted: {DATA_ROOT}")

    for cohort in cohorts:
        csv = args.cohort_csv
        if csv is None:
            # Stage 1 writes <cohort>_cohort.csv; accept <cohort>.csv too so a
            # rename upstream does not silently fall back to a raw drive scan.
            for candidate in (OUT_ROOT / "cohorts" / f"{cohort}_cohort.csv",
                              OUT_ROOT / "cohorts" / f"{cohort}.csv"):
                if candidate.exists():
                    csv = candidate
                    break
        paths = discover_files(cohort, csv)
        if args.limit:
            paths = paths[: args.limit]
        logger.info("%s: %d candidate files", cohort, len(paths))

        labels = load_slice_labels(cohort)
        cross_check_labels(csv, labels, args.pirads_threshold)
        writer = CacheWriter(Path(args.out), cohort, tuple(args.target_hw),
                             cohort_dir=args.cohort_dir)
        try:
            fn = process_dwi if cohort == "prostate_dwi" else process_t2
            stats = fn(paths, labels, writer, args)
        finally:
            writer.close()
        report(cohort, stats, writer, args.ncc_warn)


if __name__ == "__main__":
    main()
