"""
s02_breast.py
=============
fastMRI **breast** stack-of-stars golden-angle radial DCE -> PhaseDx slice cache.

This is the hardest reader in the study because the HDF5 files ship no ISMRMRD
header and the k-space array is a bare ``(2, 288, 640, 16, N)`` float64 block.
Everything below was established empirically before a single slice was cached;
the evidence is recorded here so the reconstruction can be audited rather than
trusted.


--------------------------------------------------------------------------
1. AXIS SEMANTICS  ->  kspace = (ri, spoke, readout, coil, partition)
--------------------------------------------------------------------------
Conclusion::

    kspace[2, 288, 640, 16, N]
      axis 0 = 2   : [real, imag]              (NOT [imag, real])
      axis 1 = 288 : radial spokes / views, stored in ACQUISITION ORDER
      axis 2 = 640 : radial readout, 2x oversampled, k=0 at sample index 320
      axis 3 = 16  : receive coils
      axis 4 = N   : partitions (kz), N in {83, 90, 98}, patient dependent

    temptv[192, 4, 320, 320] = (slice, dynamic_frame, y, x)
      = the vendor's temporal-TV-regularised reference reconstruction.

Evidence, in the order it was obtained:

(a) *Dataset-level HDF5 attributes* -- these exist and were the decisive clue.
    ``kspace.attrs = {centerPartition: '32', imagesPerSlab: '192',
    partitions: '83'}`` and ``temptv.attrs = {frameNumber: 4,
    images_per_slab: 192, spokes_per_frame: 72}``.  ``4 x 72 = 288`` names
    axis 1 as spokes; ``partitions`` names axis 4; ``imagesPerSlab = 192``
    matches ``temptv.shape[0]``.  (The *values* of ``partitions`` and
    ``centerPartition`` are stale for the N != 83 files -- see (f) -- so the
    attributes were used only as hypotheses and every one was re-tested.)

(b) *Readout axis.*  Coil/spoke/partition-averaged |k| along axis 2 is a
    single smooth peak at index **320** exactly (values at 318/319/320/321/322
    = 1.617/1.674/1.697/1.660/1.627 e-4), i.e. every spoke passes through the
    k-space origin at sample 320 of 640 -- the signature of a full-diameter
    radial readout.  640 / 2 = 320 = the temptv matrix, confirming 2x readout
    oversampling.

(c) *Partition axis.*  |k| along axis 4 is a sharply peaked k-space-like decay
    (peak 9.6e-5 at index 31, falling to ~6e-6 at the ends) -- a 1D k-space,
    not an image.  Taking the DC sample of every spoke, zero-filling the 83
    partitions into a 192-point grid centred on index 31 and inverse-FFTing
    reproduces the temptv slice-intensity profile with **r = 0.984**.  So axis
    4 is kz and the 192 temptv slices are its zero-filled reconstruction.

(d) *Spoke axis + golden-angle ordering.*  A 1D FFT along the readout of the
    centre partition turns each spoke into a Radon projection.  Sorting the 288
    projections by a candidate angle rule and measuring the mean correlation of
    angularly-adjacent projections cleanly separates the hypotheses::

        golden angle 111.246 deg, acquisition order   r = 0.9945   <-- WINNER
        linear 0..360 deg                             r = 0.6939
        tiny golden angle 23.628 deg                  r = 0.6789
        random angles (baseline)                      r = 0.4992
        frame-interleaved golden angle                r = 0.5164
        linear 0..180 deg                             r = 0.3941

    So spokes are stored in acquisition order with
    ``theta_s = s * 111.24611796 deg`` (= 180/phi), zero offset.  Independently,
    randomly permuting the spoke->angle assignment collapses the temptv
    correlation from 0.878 to **0.109**.

(e) *[real, imag] vs [imag, real].*  Swapping the two halves replaces the image
    by ``i * conj(image(-r))``, i.e. a 180 deg rotation plus conjugation.  Run
    as a control it drops the temptv correlation of the un-transformed image
    from **0.878 to 0.046**, and its best match over the 8 dihedral transforms
    is exactly transform #3 (180 deg rotation) at 0.709 -- precisely the
    predicted signature.  ``[real, imag]`` is correct, and the *identity*
    transform is the correct orientation (the recon matches temptv pixel for
    pixel with no flip, rotation or transpose).

(f) *N is the partition count and it really does vary.*  10 of 140 files have
    N = 90 or 98 while their ``partitions`` attribute still says '83'.  The
    extra partitions carry real, smoothly decaying signal (not padding) and the
    kz centre moves with them (N=90 -> centre 34).  The centre partition is
    therefore detected **per file** from the data, not read from the attribute,
    and all N partitions are used.  A N=90 file validates at r = 0.876.

(g) *Slice ordering.*  Reconstructing slices [40,64,96,128,152] and
    cross-correlating against the same temptv slices gives a clean diagonal
    (mean 0.845) for kz phase sign -1 and a clean **anti**-diagonal (0.416 on
    the diagonal) for +1.  Sign -1 reproduces temptv's slice order.


--------------------------------------------------------------------------
2. RECONSTRUCTION
--------------------------------------------------------------------------
Per file, for the selected slices:

  1. **kz**: direct DFT of the N acquired partitions onto the 192-slice grid,
     ``img[z] = (1/sqrt(192)) sum_p k[p] exp(-2i.pi (p-p0)(z-96)/192)``, with
     ``p0`` the per-file centre partition.  Equivalent to zero-fill + centred
     IFFT but evaluated only at the wanted slices, which keeps the working set
     under ~1 GB.  Acquisition is partial-Fourier in kz (p0 = 31 of 83, so
     kz spans -31..+51); no PF phase correction is applied -- see caveats.

  2. **in-plane**: density-compensated Kaiser-Bessel gridding (W=4,
     Beatty beta, grid oversampling 1.25 -> 800x800), centred IFFT, numeric
     deapodisation, crop 800 -> 640, then crop 640 -> 320 to undo the 2x
     readout FOV oversampling.  Adjoint NUFFT, not a zero-filled Cartesian
     IFFT: zero-filling radial spokes onto a Cartesian grid (what
     ``make_real_figure5.py`` does) is not a reconstruction.

  3. **density compensation** (mandatory -- without it the image is severely
     low-pass and the phase is wrong).  Two options, both radial-correct:

       ``ramlak``  (default) w = |k_r| * pi/Nspokes, w(0) = pi/(4 Nspokes)
       ``voronoi``           w = |k_r| * dtheta_s, with dtheta_s the angular
                             Voronoi width of spoke s among the golden-angle
                             set (sum dtheta = pi)

     Measured against temptv over 5 slices: ramlak 0.868, voronoi 0.845,
     **no DCF at all 0.716**.  Ram-Lak wins -- angular Voronoi weighting
     up-weights isolated golden-angle spokes and amplifies their streaks --
     so it is the default.  Grid oversampling 1.10 / 1.25 / 1.60 all give
     identical correlations, so 1.25 is used.

  4. **coil combination**.  Magnitude uses ``common.rss`` (phase-blind, and
     exactly what a radiologist sees).  Phase uses
     ``common.adaptive_virtual_coil``, since RSS destroys phase.  Note the
     AVC *magnitude* correlates only ~0.55-0.66 with temptv because it retains
     coil-sensitivity shading, which is why magnitude is taken from RSS.


--------------------------------------------------------------------------
3. VALIDATION  (Pearson r of reconstructed |image| vs temptv)
--------------------------------------------------------------------------
Every cached slice stores its own ``temptv_r`` in the index CSV, so QC is not a
one-off claim.  Reference numbers from the forensics:

    spokes used     r vs temptv        phase split-half cos-sim
    72  (1 frame)      0.854 - 0.871           0.50
    144 (2 frames)     0.934                   0.78
    288 (all)          0.973 - 0.976           (>0.78, not directly measurable)

  * Cross-patient, 288 spokes, 3 patients incl. one N=90 file: 0.85 - 0.88 at
    72 spokes, 0.97 at 288.
  * Orientation is the identity transform -- no flip/rotation needed.

**Confidence statement.**  The magnitude channel is validated (r ~ 0.97).  The
phase channel is *indirectly* validated: it comes from the same complex image
whose magnitude matches temptv, and temptv is magnitude-only so the phase
cannot be checked against a reference at all.  What *was* measured is phase
**self-consistency**: reconstructing two disjoint halves of the spokes and
comparing their phase inside the body gives cos-similarity 0.50 at 72 spokes
and 0.78 at 144 spokes.  That is the honest bound -- a 72-spoke (single dynamic
frame) phase map is substantially streak-contaminated and should not be
trusted; 288 spokes is materially better but still not a clean field map.
Downstream analysis must treat breast phase as the noisiest of the three
cohorts, and the background-only control matters most here.


--------------------------------------------------------------------------
4. DYNAMIC FRAME CHOICE
--------------------------------------------------------------------------
Default is ``--frame all``: all 288 spokes, i.e. the *entire* acquisition, for
every patient.  Reasons, in order of weight:

  1. It removes DCE timing as a confound *completely* -- every patient
     contributes an identical temporal footprint (the whole scan), rather than
     a 72-spoke window whose contrast state depends on that patient's
     injection timing.
  2. There is almost no dynamic signal to preserve anyway: across 5 patients
     the temptv enhancement curve over the 4 frames, measured on the top 1% of
     frame-0 voxels, is flat to within 3% ([1.0, 0.99, 0.98, 0.99] etc.).  The
     4 frames sit in the same contrast state.
  3. 72 spokes is ~7x below the radial Nyquist rate for a 320 matrix
     (pi/2 * 320 ~ 503 spokes).  The resulting streaks wreck the phase channel
     (split-half cos-sim 0.50).  288 spokes cuts that 4-fold.

``--frame 0|1|2|3`` remains available and is honoured consistently for the
whole cohort; the choice is recorded in the ``frame`` column of the index CSV.


--------------------------------------------------------------------------
5. KNOWN LIMITATIONS  (read these before believing any breast result)
--------------------------------------------------------------------------
  * **Class balance is broken.**  Only 70 of the 300 labelled patients are on
    disk, and they are not a random subset: 54 malignant / 9 benign / 7
    negative.  The official test split contains 14 malignant and 2
    non-malignant patients.  No trustworthy AUC can come from 2 negative test
    patients; the breast cohort should be treated as a secondary/nested-CV
    analysis at best.
  * No B0 field map or coil sensitivity maps ship with the data, so the phase
    is the AVC phase, which mixes tissue susceptibility, B0 inhomogeneity,
    partial-Fourier-in-kz phase, and gridding streaks.
  * No gradient-delay / eddy-current trajectory correction is applied.  The
    coil-averaged readout peak sits at sample 320.0 so the bulk delay is below
    one sample, but per-spoke delays are not corrected.
  * ``n_partitions`` (83/90/98) is a hard protocol fingerprint and is written
    to the index CSV as a confound column precisely so the fingerprint
    hypothesis can be tested rather than assumed away.
  * One file, ``fastMRI_breast_272_1.h5``, is corrupt ("bad global heap
    collection signature") and is skipped.
"""

from __future__ import annotations

import argparse
import csv
import logging
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import h5py
import numpy as np
import scipy.sparse as sp
from scipy.fft import fftshift, ifft2, ifftshift
from scipy.ndimage import gaussian_filter
from scipy.special import i0

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import (  # noqa: E402
    BREAST_LABEL_XLSX,
    CACHE_DIR,
    COHORT_DIR,
    DATA_ROOT,
    SUBJECT_COL,
    TARGET_HW,
    adaptive_virtual_coil,
    backfill_cache_subject_ids,
    body_mask,
    iter_h5,
    normalize_magnitude,
    resample_phase,
    resample_to,
    rss,
    subject_id_lut,
)

logger = logging.getLogger("s02_breast")

COHORT = "breast"
BREAST_ROOT = DATA_ROOT / "breast_updated" / "breast"

# --- Acquisition constants, all established in the module docstring ----------
N_SPOKES_TOTAL = 288
N_READOUT = 640          # 2x oversampled radial readout
KSPACE_CENTER_RO = 320   # index of k_r = 0 along the readout
N_COILS = 16
N_SLICES_OUT = 192       # imagesPerSlab
SPOKES_PER_FRAME = 72
N_FRAMES = 4
MATRIX_OUT = 320         # temptv in-plane matrix after undoing 2x readout OS
GOLDEN_ANGLE_DEG = 111.24611796498215  # 180 / phi
KZ_PHASE_SIGN = -1       # established in docstring section 1(g)

FNAME_RE = re.compile(r"^fastMRI_breast_(\d+)_(\d+)\.h5$")

LABEL_COL_PID = "Patient Coded Name"
LABEL_COL_REPEAT = "Repeated scans (0 = not repeat, 1:16 = repeated)"
LABEL_COL_SPLIT = "Data split (0=training, 1=testing)"
LABEL_COL_LESION = "Lesion status (0 = negative, 1= malignancy, 2= benign)"
LABEL_COL_AGE = "Age"
LABEL_COL_MENO = "Menopause (0= pre, 1 = post, 2 = unknown)"
LABEL_COL_REASON = (
    "Reason (0= increased risk screening, 1= short-term f/u, 2= cancer workup, "
    "3=problem-solving, 4=neoadjuvant therapy response"
)
LABEL_COL_LATERALITY = "Laterality (1=right, 2=left)"
LABEL_COL_DETECTION = "Detection type (1= mammo, 2= US, 3 = MRI)"
LABEL_COL_GRADE = "Grade (1-3)"
LABEL_COL_ER = "ER+ (yes=1, no=0)"
LABEL_COL_PR = "PR+ (yes=1, no=0)"
LABEL_COL_HER2 = "HER2+ (yes=1, no=0)"

INDEX_COLUMNS = [
    # contract columns
    #
    # subject_id is the split-enforcement unit and is NOT redundant with
    # patient_id here: the label sheet's 'Repeated scan' column links two
    # different coded patient names to one physical woman, and stage 1 collapses
    # them into a single subject_id. A split enforced on patient_id can put her
    # repeat scan in test and her index scan in training. It is joined from the
    # stage-1 cohort CSV rather than recomputed from `repeated_scan` below,
    # because the cohort table is the one that was actually subject-checked by
    # s01_labels.assert_no_subject_spans_splits -- deriving it a second way here
    # would create two definitions that can silently disagree.
    "idx", "patient_id", SUBJECT_COL, "file", "slice", "label", "raw_label",
    "official_split", "folder", "acq",
    # cohort-specific confounds / QC
    "frame", "n_spokes", "n_partitions", "center_partition", "dcf",
    "temptv_r", "qc", "mask_frac",
    "repeated_scan", "age", "menopause", "reason", "laterality",
    "detection_type", "grade", "er", "pr", "her2",
]


# =============================================================================
# Trajectory, density compensation and gridding
# =============================================================================

def spoke_angles(spoke_ids: np.ndarray) -> np.ndarray:
    """
    Golden-angle radial ordering, established empirically (docstring 1d).

    ASSUMPTION, documented and tested: spokes are stored in acquisition order
    and the angle of spoke s is ``s * 180/phi`` degrees with zero offset and
    positive handedness.  Both the offset (0 vs 45 vs 90 deg) and the
    handedness were swept against the temptv reference; only (offset 0,
    sign +1) reproduces temptv under the identity transform.
    """
    return np.deg2rad(GOLDEN_ANGLE_DEG) * spoke_ids.astype(np.float64)


def _angular_voronoi(theta: np.ndarray) -> np.ndarray:
    """
    Angular Voronoi width per spoke.

    Spokes are full diameters, so the angle lives on a circle of period pi and
    the widths sum to pi.  Golden-angle sets are only quasi-uniform, so this is
    the "correct" area weight -- but see the docstring: it is empirically worse
    than plain Ram-Lak because it amplifies isolated spokes.
    """
    t = np.mod(theta, np.pi)
    order = np.argsort(t)
    ts = t[order]
    nxt = np.roll(ts, -1).copy()
    nxt[-1] += np.pi
    prv = np.roll(ts, 1).copy()
    prv[0] -= np.pi
    widths = (nxt - prv) / 2.0
    out = np.empty_like(widths)
    out[order] = widths
    return out


def density_compensation(theta: np.ndarray, mode: str = "ramlak") -> np.ndarray:
    """
    Radial density compensation, shape (n_spokes, N_READOUT).

    The polar area element is ``r dr dtheta``, hence the |k_r| (Ram-Lak) ramp.
    The DC sample is shared by every spoke; its Voronoi cell is the disc of
    radius dr/2, area ``pi/4`` in sample units, split over the spokes ->
    ``pi / (4 * n_spokes)``.  Omitting the ramp leaves the reconstruction
    dominated by the massively oversampled centre of k-space (r drops 0.868 ->
    0.716 against temptv), which both blurs the magnitude and, worse, drags the
    phase toward the centre-of-k-space phase.
    """
    n_spokes = theta.size
    r = np.abs(np.arange(N_READOUT) - KSPACE_CENTER_RO).astype(np.float64)

    if mode == "ramlak":
        ang = np.full(n_spokes, np.pi / n_spokes)
    elif mode == "voronoi":
        ang = _angular_voronoi(theta)
    else:
        raise ValueError(f"unknown dcf mode {mode!r}")

    w = r[None, :] * ang[:, None]
    w[:, KSPACE_CENTER_RO] = np.pi / (4.0 * n_spokes)
    return w


def _kaiser_bessel(width: float, oversamp: float, n: int = 2048):
    """Kaiser-Bessel gridding kernel, Beatty et al. optimal beta."""
    beta = np.pi * np.sqrt(width ** 2 / oversamp ** 2 * (oversamp - 0.5) ** 2 - 0.8)
    u = np.linspace(0.0, width / 2.0, n)
    v = i0(beta * np.sqrt(np.maximum(0.0, 1.0 - (2.0 * u / width) ** 2)))
    return u, v / v[0]


class Gridder:
    """
    Adjoint NUFFT for a fixed radial trajectory.

    Builds the sparse (grid x sample) Kaiser-Bessel interpolation operator once
    and reuses it for every coil, slice, and file -- the trajectory is identical
    across the whole cohort, so this is built exactly once per run.
    """

    def __init__(self, theta: np.ndarray, oversamp: float = 1.25,
                 width: int = 4, matrix: int = N_READOUT):
        self.matrix = matrix
        self.grid = int(round(matrix * oversamp / 2.0) * 2)
        self.n_spokes = theta.size

        kr = (np.arange(N_READOUT) - KSPACE_CENTER_RO) / float(N_READOUT)  # [-0.5, 0.5)
        kx = (np.cos(theta)[:, None] * kr[None, :]).ravel()
        ky = (np.sin(theta)[:, None] * kr[None, :]).ravel()

        u, v = _kaiser_bessel(float(width), oversamp)
        g = self.grid
        gx = g / 2.0 + kx * g
        gy = g / 2.0 + ky * g
        half = width // 2
        rows, cols, vals = [], [], []
        sidx = np.arange(kx.size)
        for dx in range(-half, half + 1):
            ix = np.floor(gx).astype(np.int64) + dx
            wx = np.interp(np.abs(gx - ix), u, v, right=0.0)
            for dy in range(-half, half + 1):
                iy = np.floor(gy).astype(np.int64) + dy
                wy = np.interp(np.abs(gy - iy), u, v, right=0.0)
                w = wx * wy
                ok = (ix >= 0) & (ix < g) & (iy >= 0) & (iy < g) & (w > 0)
                rows.append(ix[ok] * g + iy[ok])
                cols.append(sidx[ok])
                vals.append(w[ok])
        self.M = sp.csr_matrix(
            (np.concatenate(vals).astype(np.float32),
             (np.concatenate(rows), np.concatenate(cols))),
            shape=(g * g, kx.size),
        )

        # Numeric (exact, separable) deapodisation for this kernel.
        c1 = np.zeros(g)
        for d in range(-half, half + 1):
            c1[g // 2 + d] = np.interp(abs(d), u, v, right=0.0)
        d1 = np.real(np.fft.fftshift(np.fft.ifft(np.fft.ifftshift(c1)))) * g
        lo, hi = g // 2 - matrix // 2, g // 2 + matrix // 2
        self.deapod = np.outer(d1, d1)[lo:hi, lo:hi].astype(np.float32)
        self._lo, self._hi = lo, hi

    def __call__(self, samples: np.ndarray) -> np.ndarray:
        """
        samples: (n_spokes * N_READOUT, n_coils) complex, already DCF-weighted.
        returns: (matrix, matrix, n_coils) complex coil images.
        """
        g = self.grid
        grid = np.asarray(self.M @ samples).reshape(g, g, -1)
        img = fftshift(
            ifft2(ifftshift(grid, axes=(0, 1)), axes=(0, 1), norm="ortho"),
            axes=(0, 1),
        )
        img = img[self._lo:self._hi, self._lo:self._hi] / self.deapod[:, :, None]
        return img


def kz_dft_matrix(n_part: int, center_part: int,
                  slices: Sequence[int], n_out: int = N_SLICES_OUT) -> np.ndarray:
    """
    Partition -> slice transform, evaluated only at the requested slices.

    Identical to zero-filling the ``n_part`` acquired partitions into an
    ``n_out``-point grid centred on ``center_part`` and taking a centred FFT,
    but without materialising the zero-filled array.  The sign is
    ``KZ_PHASE_SIGN``, fixed by the cross-correlation test in docstring 1(g).
    """
    z = np.asarray(slices, dtype=np.float64)[:, None] - n_out // 2
    p = np.arange(n_part, dtype=np.float64)[None, :] - center_part
    return (np.exp(KZ_PHASE_SIGN * 2j * np.pi * p * z / n_out)
            / np.sqrt(n_out)).astype(np.complex64)


# =============================================================================
# Masking
# =============================================================================

def _multiotsu_low_threshold(x: np.ndarray, nbins: int = 128) -> float:
    """
    Lower threshold of a 3-class Otsu split -- i.e. the air|tissue boundary.

    Two-class Otsu is the wrong model here.  A breast slice has three
    populations, not two: air (plus radial streaks), dark fat / muscle, and
    bright fibroglandular tissue and vessels.  Two-class Otsu puts its single
    threshold between fat and fibroglandular tissue, so it segments *bright
    tissue*, not *body*, and throws away most of the breast.  The lower of the
    two 3-class thresholds is the one that separates air from patient.
    """
    hist, edges = np.histogram(x, bins=nbins, range=(0.0, 1.0))
    hist = hist.astype(np.float64)
    hist /= max(hist.sum(), 1e-12)
    mids = 0.5 * (edges[:-1] + edges[1:])
    w = np.cumsum(hist)
    m = np.cumsum(hist * mids)
    mt = m[-1]

    a = np.arange(nbins)[:, None]
    b = np.arange(nbins)[None, :]
    w0, w1, w2 = w[a], w[b] - w[a], 1.0 - w[b]
    with np.errstate(divide="ignore", invalid="ignore"):
        m0 = m[a] / w0
        m1 = (m[b] - m[a]) / w1
        m2 = (mt - m[b]) / w2
        var = w0 * (m0 - mt) ** 2 + w1 * (m1 - mt) ** 2 + w2 * (m2 - mt) ** 2
    var = np.broadcast_to(var, (nbins, nbins)).copy()
    var[~np.isfinite(var)] = -1.0
    var[np.broadcast_to(np.minimum(np.minimum(w0, w1), w2),
                        (nbins, nbins)) < 1e-9] = -1.0
    var[np.broadcast_to(b <= a, (nbins, nbins))] = -1.0
    ia = int(np.unravel_index(int(np.argmax(var)), var.shape)[0])
    return float(mids[ia])


def breast_body_mask(mag: np.ndarray, sigma: float = 3.0,
                     min_component_frac: float = 0.01) -> np.ndarray:
    """
    Body mask for a breast slice.

    Three things go wrong if this is done naively, and all three were observed:

    1. ``common.body_mask``'s 60th-percentile default is calibrated for cohorts
       where the body fills the FOV.  Axial breast does not, and the radial
       streaks lift the background, so the 60th percentile lands *in the air*.
       Measured mask fraction was 0.56-0.63 -- the "body" mask was mostly
       background, which would silently destroy the background-only
       falsification control, the single most important experiment here.
    2. Two-class Otsu goes the other way and segments only bright
       fibroglandular tissue (fraction 0.13, 79-85% of image energy), dropping
       the fatty breast.  Hence ``_multiotsu_low_threshold``.
    3. Thresholding the *unsmoothed* image leaves thin radial streaks that
       reach the image border and bridge the body to it; ``binary_fill_holes``
       then floods the enclosed air (e.g. the gap between the two breasts) into
       the mask.  Hence the Gaussian pre-smooth and the opening + connected
       component filter before the final hole fill.

    So: smooth (sigma=3 px, streaks average out, anatomy does not) -> 3-class
    Otsu lower threshold -> ``common.body_mask`` for the shared closing +
    hole-filling morphology -> opening + drop components below
    ``min_component_frac`` of the image -> fill holes.

    Measured on validated slices: fraction 0.20-0.46, 82-95% of image energy
    captured, mean in-mask / out-of-mask magnitude ratio 4-8.
    """
    from scipy.ndimage import binary_fill_holes, binary_opening, label

    sm = gaussian_filter(np.asarray(mag, dtype=np.float64), sigma)
    thr = _multiotsu_low_threshold(sm)
    pct = float(np.clip((sm < thr).mean() * 100.0, 1.0, 99.0))

    m = body_mask(sm, thresh_pct=pct)
    m = binary_opening(m, structure=np.ones((5, 5)))
    lab, n = label(m)
    if n:
        sizes = np.bincount(lab.ravel())
        sizes[0] = 0
        keep = np.where(sizes > min_component_frac * m.size)[0]
        m = np.isin(lab, keep) if keep.size else m
    return binary_fill_holes(m).astype(bool)


# =============================================================================
# File discovery and labels
# =============================================================================

def discover_files() -> List[Tuple[Path, int, int]]:
    """Return (path, patient_id, acquisition_number), AppleDouble-filtered."""
    out = []
    for p in iter_h5(BREAST_ROOT):
        m = FNAME_RE.match(p.name)
        if not m:
            logger.warning("skipping unrecognised filename %s", p.name)
            continue
        out.append((p, int(m.group(1)), int(m.group(2))))
    return sorted(out, key=lambda t: (t[1], t[2]))


def load_labels() -> Dict[int, dict]:
    """
    Patient-level labels keyed by integer patient id.

    Binary label = 1 iff 'Lesion status' == 1 (malignancy); benign (2) and
    negative (0) both map to 0.  Official split 0 -> training, 1 -> test (the
    breast release has no validation split, per the cache contract mapping).
    """
    import pandas as pd

    df = pd.read_excel(BREAST_LABEL_XLSX)
    df["pid"] = df[LABEL_COL_PID].astype(str).str.extract(r"(\d+)\s*$").astype(int)

    def g(row, col):
        if col not in row:
            return ""
        v = row[col]
        return "" if (v is None or (isinstance(v, float) and np.isnan(v))) else v

    out: Dict[int, dict] = {}
    for _, row in df.iterrows():
        raw = row[LABEL_COL_LESION]
        out[int(row["pid"])] = dict(
            label=int(raw == 1),
            raw_label=int(raw),
            official_split="test" if int(row[LABEL_COL_SPLIT]) == 1 else "training",
            repeated_scan=g(row, LABEL_COL_REPEAT),
            age=g(row, LABEL_COL_AGE),
            menopause=g(row, LABEL_COL_MENO),
            reason=g(row, LABEL_COL_REASON),
            laterality=g(row, LABEL_COL_LATERALITY),
            detection_type=g(row, LABEL_COL_DETECTION),
            grade=g(row, LABEL_COL_GRADE),
            er=g(row, LABEL_COL_ER),
            pr=g(row, LABEL_COL_PR),
            her2=g(row, LABEL_COL_HER2),
        )
    return out


# =============================================================================
# Per-file reconstruction
# =============================================================================

def probe_volume(kspace: h5py.Dataset, n_probe_spokes: int = 8
                 ) -> Tuple[int, int, np.ndarray]:
    """
    Cheap pre-read: centre partition and the slab intensity profile.

    Reads only ``n_probe_spokes`` spokes (~1/36 of the file) and uses their DC
    samples.  Every spoke passes through kx=ky=0, so averaging the DC samples
    over spokes gives a clean 1D kz signal per coil; transforming it to the 192
    slice grid gives the slab profile, which is what slice selection needs.

    Returns (n_partitions, center_partition, slab_profile[192] normalised).
    """
    n_part = kspace.shape[-1]
    lo, hi = KSPACE_CENTER_RO - 2, KSPACE_CENTER_RO + 3
    blk = kspace[:, :n_probe_spokes, lo:hi, :, :]
    z = (blk[0] + 1j * blk[1])[:, KSPACE_CENTER_RO - lo]        # (spokes, coils, part)

    center_part = int(np.abs(z).mean(axis=(0, 1)).argmax())
    E = kz_dft_matrix(n_part, center_part, np.arange(N_SLICES_OUT))
    prof_c = z.mean(axis=0).astype(np.complex64) @ E.T          # (coils, 192)
    prof = np.sqrt((np.abs(prof_c) ** 2).sum(axis=0))
    return n_part, center_part, prof / max(prof.max(), 1e-30)


def select_slices(profile: np.ndarray, n_slices: int, frac: float = 0.4) -> List[int]:
    """
    Pick ``n_slices`` evenly spaced slices inside the covered slab.

    Slab coverage varies a lot between patients -- measured across 10 patients,
    the >50%-of-peak range runs from 7..87 for one and 56..151 for another.  A
    fixed slice list would therefore sample completely different anatomy in
    different patients, so selection is adaptive: take the contiguous span where
    the slab profile exceeds ``frac`` of its peak, drop the outer 10% at each
    end (partial slices / slab edges), and sample it uniformly.
    """
    idx = np.where(profile > frac)[0]
    if idx.size < n_slices:
        idx = np.where(profile > 0.5 * frac)[0]
    if idx.size < n_slices:
        idx = np.arange(N_SLICES_OUT // 4, 3 * N_SLICES_OUT // 4)
    lo, hi = int(idx.min()), int(idx.max())
    pad = int(0.10 * (hi - lo))
    lo, hi = lo + pad, hi - pad
    if hi - lo + 1 < n_slices:
        lo, hi = int(idx.min()), int(idx.max())
    # dedupe: a very thin slab can be narrower than n_slices, and caching the
    # same slice twice would silently double-weight it in training.
    return sorted({int(round(v)) for v in np.linspace(lo, hi, n_slices)})


def reconstruct_file(path: Path, gridder: Gridder, dcf_w: np.ndarray,
                     spoke_ids: np.ndarray, n_slices: int,
                     read_chunk: int = 36
                     ) -> Tuple[List[int], np.ndarray, np.ndarray, int, int]:
    """
    Reconstruct ``n_slices`` slices of one breast volume.

    Returns (slice_indices, magnitude[nsl,320,320], complex_avc[nsl,320,320],
             n_partitions, center_partition).
    """
    with h5py.File(path, "r") as f:
        ks = f["kspace"]
        n_part, center_part, prof = probe_volume(ks)
        slices = select_slices(prof, n_slices)
        E = kz_dft_matrix(n_part, center_part, slices)          # (nsl, n_part)

        n_sp = spoke_ids.size
        acc = np.zeros((len(slices), n_sp, N_READOUT, N_COILS), np.complex64)
        s0, s1 = int(spoke_ids[0]), int(spoke_ids[-1]) + 1
        for a in range(s0, s1, read_chunk):
            b = min(a + read_chunk, s1)
            blk = ks[:, a:b, :, :, :]
            z = (blk[0] + 1j * blk[1]).astype(np.complex64)
            del blk
            acc[:, a - s0:b - s0] = np.einsum("zp,srcp->zsrc", E, z, optimize=True)
            del z

    w = dcf_w.reshape(-1, 1).astype(np.float32)
    mags = np.empty((len(slices), MATRIX_OUT, MATRIX_OUT), np.float32)
    cplx = np.empty((len(slices), MATRIX_OUT, MATRIX_OUT), np.complex64)
    crop = (N_READOUT - MATRIX_OUT) // 2
    for zi in range(len(slices)):
        coil_img = gridder(acc[zi].reshape(-1, N_COILS) * w)
        coil_img = coil_img[crop:crop + MATRIX_OUT, crop:crop + MATRIX_OUT]
        mags[zi] = rss(coil_img, coil_axis=-1).astype(np.float32)
        cplx[zi] = adaptive_virtual_coil(coil_img, coil_axis=-1).astype(np.complex64)
    return slices, mags, cplx, n_part, center_part


def temptv_correlation(path: Path, slices: Sequence[int], mags: np.ndarray,
                       frame: Optional[int]) -> List[float]:
    """
    Pearson r of each reconstructed magnitude slice against the vendor temptv
    reference.  This is the gate for the whole cohort and is recorded per slice.

    When all 288 spokes are used the comparison is against the mean of the 4
    temptv dynamic frames, which is the matching temporal footprint.

    NaN means "could not be validated", not "validated as bad".  Exactly one
    file in the release (``fastMRI_breast_272_1.h5``) has an all-zero temptv
    dataset -- its k-space is fine, its reference is simply absent -- so these
    slices are cached but flagged ``qc='unvalidated'``.
    """
    out = []
    with h5py.File(path, "r") as f:
        tv = f["temptv"]
        for zi, z in enumerate(slices):
            ref = np.asarray(tv[z, frame]) if frame is not None \
                else np.asarray(tv[z]).mean(axis=0)
            a = mags[zi].ravel().astype(np.float64)
            b = ref.ravel().astype(np.float64)
            if a.std() < 1e-20 or b.std() < 1e-20:
                out.append(float("nan"))
            else:
                out.append(float(np.corrcoef(a, b)[0, 1]))
    return out


# =============================================================================
# Cache writing
# =============================================================================

class CacheWriter:
    """Incremental, crash-tolerant writer for the cache contract."""

    def __init__(self, h5_path: Path, csv_path: Path, resume: bool, cohort_dir=None):
        self.h5_path, self.csv_path = h5_path, csv_path
        h5_path.parent.mkdir(parents=True, exist_ok=True)
        self.done_files = set()
        # Fail before any reconstruction runs: without the stage-1 table there is
        # no sanctioned source of subject_id, and an index without it is one that
        # stage 3 will refuse to train on.
        self.subject_lut = subject_id_lut(COHORT, cohort_dir)
        mode = "a" if (resume and h5_path.exists() and csv_path.exists()) else "w"
        if mode == "a":
            with csv_path.open() as fh:
                reader = csv.DictReader(fh)
                header = list(reader.fieldnames or [])
                # Appending new-schema rows under an old header would shift every
                # column silently from the resume point onward.
                if header != INDEX_COLUMNS:
                    extra = [c for c in header if c not in INDEX_COLUMNS]
                    missing = [c for c in INDEX_COLUMNS if c not in header]
                    raise SystemExit(
                        f"cannot resume: {csv_path} was written with a different index "
                        f"schema (missing {missing}, unexpected {extra}). Backfill it "
                        "with --backfill-subject-ids, or pass --no-resume to rebuild."
                    )
                for row in reader:
                    self.done_files.add(row["file"])
            logger.info("resuming: %d files already cached", len(self.done_files))
        self.h5 = h5py.File(h5_path, "a" if mode == "a" else "w")
        for name, dtype in (("mag", np.float16), ("phase", np.float16), ("mask", np.bool_)):
            if name not in self.h5:
                self.h5.create_dataset(
                    name, shape=(0, *TARGET_HW), maxshape=(None, *TARGET_HW),
                    dtype=dtype, chunks=(8, *TARGET_HW), compression="lzf",
                )
        self.n = self.h5["mag"].shape[0]
        new = mode == "w"
        self.csv = csv_path.open("w" if new else "a", newline="")
        self.writer = csv.DictWriter(self.csv, fieldnames=INDEX_COLUMNS)
        if new:
            self.writer.writeheader()
            self.csv.flush()

    def subject_id_for(self, file_name: str) -> str:
        """Stage-1 subject_id for a cached file, or raise."""
        bn = Path(str(file_name)).name
        try:
            return self.subject_lut[bn]
        except KeyError:
            raise KeyError(
                f"[{COHORT}] {bn} has no row in the stage-1 cohort CSV, so it has no "
                "subject_id and stage 3 could not enforce splits on it. Re-run stage 1 "
                "over the same file set before caching."
            ) from None

    def append(self, mag: np.ndarray, phase: np.ndarray, mask: np.ndarray,
               rows: List[dict]) -> None:
        k = mag.shape[0]
        for name, arr in (("mag", mag), ("phase", phase), ("mask", mask)):
            d = self.h5[name]
            d.resize(self.n + k, axis=0)
            d[self.n:self.n + k] = arr
        for i, r in enumerate(rows):
            r["idx"] = self.n + i
            # Set explicitly rather than letting the .get(c, "") default fill it:
            # an empty-string subject_id is NOT null, so it would sail through
            # every notna() check downstream and then group every row of the
            # cohort into one "subject".
            r[SUBJECT_COL] = self.subject_id_for(r["file"])
            self.writer.writerow({c: r.get(c, "") for c in INDEX_COLUMNS})
        self.n += k
        self.h5.flush()
        self.csv.flush()

    def close(self):
        self.h5.close()
        self.csv.close()


# =============================================================================
# Main
# =============================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[3],
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cohort-csv", type=Path, default=None,
                   help="restrict to files listed in this CSV; recognises a "
                        "'file' (basename or path) or 'patient_id' column")
    p.add_argument("--limit", type=int, default=None,
                   help="process at most this many files")
    p.add_argument("--frame", default="all", choices=["0", "1", "2", "3", "all"],
                   help="dynamic frame to reconstruct; 'all' uses all 288 "
                        "spokes (default, see module docstring section 4)")
    p.add_argument("--slices-per-volume", type=int, default=16)
    p.add_argument("--dcf", default="ramlak", choices=["ramlak", "voronoi"])
    p.add_argument("--grid-oversamp", type=float, default=1.25)
    p.add_argument("--out-h5", type=Path, default=CACHE_DIR / f"{COHORT}.h5")
    p.add_argument("--out-csv", type=Path, default=CACHE_DIR / f"{COHORT}_index.csv")
    p.add_argument("--cohort-dir", type=Path, default=COHORT_DIR,
                   help="stage-1 output directory; the source of subject_id")
    p.add_argument("--backfill-subject-ids", action="store_true",
                   help="add subject_id to an EXISTING breast_index.csv and exit. "
                        "Touches only the CSV -- the validated HDF5 cache (hours of "
                        "gridding, r=0.97 vs the vendor temptv reference) is never "
                        "reopened. Idempotent: re-running a backfilled index is a no-op.")
    p.add_argument("--no-resume", action="store_true")
    p.add_argument("--min-temptv-r", type=float, default=0.90,
                   help="slices below this correlation with the temptv "
                        "reference are flagged qc='low_r' (and dropped if "
                        "--drop-failed is set)")
    p.add_argument("--drop-failed", action="store_true",
                   help="exclude slices whose qc is not 'ok' from the cache "
                        "instead of caching them with a flag")
    return p


def load_cohort_filter(path: Optional[Path]) -> Optional[set]:
    if path is None:
        return None
    import pandas as pd
    df = pd.read_csv(path)
    if "file" in df.columns:
        return {Path(str(v)).name for v in df["file"].dropna()}
    if "patient_id" in df.columns:
        return {f"pid:{int(v)}" for v in df["patient_id"].dropna()}
    raise ValueError(f"{path} has neither a 'file' nor a 'patient_id' column")


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    if args.backfill_subject_ids:
        # Reads two CSVs and never touches the drive, so it works unmounted.
        rep = backfill_cache_subject_ids(
            COHORT, args.out_csv.parent, args.cohort_dir
        )
        print(f"[{COHORT}] {rep['status']}: {rep['rows']} rows, "
              f"{rep['n_patients']} patient_id -> {rep['n_subjects']} subject_id "
              f"({rep['path']})")
        if rep["subjects_spanning_splits"]:
            print(f"[{COHORT}] WARNING: subjects spanning splits: "
                  f"{rep['subjects_spanning_splits']}")
        return 0

    frame = None if args.frame == "all" else int(args.frame)
    if frame is None:
        spoke_ids = np.arange(N_SPOKES_TOTAL)
    else:
        spoke_ids = np.arange(frame * SPOKES_PER_FRAME,
                              (frame + 1) * SPOKES_PER_FRAME)

    theta = spoke_angles(spoke_ids)
    dcf_w = density_compensation(theta, args.dcf)
    t0 = time.time()
    gridder = Gridder(theta, oversamp=args.grid_oversamp)
    logger.info("gridder: %d spokes, %d x %d grid, %d nnz (%.1fs)",
                spoke_ids.size, gridder.grid, gridder.grid,
                gridder.M.nnz, time.time() - t0)

    labels = load_labels()
    files = discover_files()
    keep = load_cohort_filter(args.cohort_csv)
    if keep is not None:
        files = [(p, pid, acq) for (p, pid, acq) in files
                 if p.name in keep or f"pid:{pid}" in keep]
    logger.info("%d candidate files", len(files))

    writer = CacheWriter(args.out_h5, args.out_csv, resume=not args.no_resume,
                         cohort_dir=args.cohort_dir)
    n_done = n_fail = 0
    n_qc: Counter = Counter()
    all_r: List[float] = []
    try:
        for path, pid, acq in files:
            if args.limit is not None and n_done >= args.limit:
                break
            if path.name in writer.done_files:
                logger.info("skip (cached) %s", path.name)
                continue
            if pid not in labels:
                logger.warning("no label row for patient %d (%s) -- skipping",
                               pid, path.name)
                n_fail += 1
                continue
            t_file = time.time()
            try:
                slices, mags, cplx, n_part, center_part = reconstruct_file(
                    path, gridder, dcf_w, spoke_ids, args.slices_per_volume)
                rs = temptv_correlation(path, slices, mags, frame)
            except Exception as exc:  # noqa: BLE001
                logger.error("FAILED %s: %s", path.name, exc)
                n_fail += 1
                continue

            mags_l, ph_l, mask_l, rows = [], [], [], []
            meta = labels[pid]
            for zi in range(len(slices)):
                r = rs[zi]
                qc = ("unvalidated" if not np.isfinite(r)
                      else "ok" if r >= args.min_temptv_r else "low_r")
                if args.drop_failed and qc != "ok":
                    continue
                m = normalize_magnitude(resample_to(mags[zi].astype(np.float64)))
                # phase resampled through sin/cos, kept in radians on [-pi, pi]
                ph = resample_phase(np.angle(cplx[zi]).astype(np.float64))
                msk = breast_body_mask(m)
                mags_l.append(m.astype(np.float16))
                ph_l.append(ph.astype(np.float16))
                mask_l.append(msk)
                rows.append(dict(
                    patient_id=pid, file=path.name, slice=slices[zi],
                    folder=path.parent.name, acq=acq,
                    frame=args.frame, n_spokes=int(spoke_ids.size),
                    n_partitions=n_part, center_partition=center_part,
                    dcf=args.dcf,
                    temptv_r="" if not np.isfinite(r) else round(r, 4),
                    qc=qc, mask_frac=round(float(msk.mean()), 4), **meta,
                ))
            if rows:
                writer.append(np.stack(mags_l), np.stack(ph_l),
                              np.stack(mask_l), rows)
            all_r.extend(rs)
            n_done += 1
            for r_ in rows:
                n_qc[r_["qc"]] += 1
            bad = [f"{s}:{v:.2f}" if np.isfinite(v) else f"{s}:nan"
                   for s, v in zip(slices, rs)
                   if not (np.isfinite(v) and v >= args.min_temptv_r)]
            finite = [v for v in rs if np.isfinite(v)]
            logger.log(
                logging.WARNING if bad else logging.INFO,
                "[%d/%d] %s pid=%d acq=%d label=%d N=%d p0=%d slices=%d "
                "temptv_r mean=%s min=%s (%.0fs)%s",
                n_done, len(files), path.name, pid, acq, meta["label"],
                n_part, center_part, len(rows),
                f"{np.mean(finite):.3f}" if finite else "n/a",
                f"{np.min(finite):.3f}" if finite else "n/a",
                time.time() - t_file,
                ("  NOT-OK: " + ",".join(bad)) if bad else "")
    finally:
        writer.close()

    a = np.asarray([v for v in all_r if np.isfinite(v)], dtype=float)
    if a.size:
        logger.info("VALIDATION: %d/%d slices validated against temptv; "
                    "Pearson r mean=%.4f median=%.4f p05=%.4f min=%.4f",
                    a.size, len(all_r), a.mean(), np.median(a),
                    np.percentile(a, 5), a.min())
    if n_qc:
        logger.info("QC breakdown: %s", dict(n_qc))
    logger.info("done: %d files cached, %d failed, %d slices total -> %s",
                n_done, n_fail, writer.n, args.out_h5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
