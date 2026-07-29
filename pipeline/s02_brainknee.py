"""
s02_brainknee.py
================
Stage 2c of the PhaseDx pipeline: the fastMRI **brain** and **knee** readers.

These two cohorts exist for one purpose and it is not diagnosis. Neither
release ships a tumour label, a lesion mask, or any clinical annotation
whatsoever, and nothing downstream may present them as if it did. They are the
CONFOUND cohorts: the targets are receive-coil count, pulse-sequence contrast
and matrix size -- pure acquisition properties with no pathological meaning. A
network that reads those off the phase channel is fingerprinting the scanner,
which is the finding this study now reports.

The two cohorts are a matched pair:

  brain (454 files)  receive-coil count VARIES from 4 to 20 across ten distinct
                     values. If phase predicts coil count here, phase carries
                     receive-array identity.
  knee  (199 files)  receive-coil count is FIXED at 15 for every single file --
                     one TxRx_15Ch_Knee array, no exceptions. Hardware is
                     therefore held constant by construction, and the question
                     becomes whether phase can still separate CORPD_FBK from
                     CORPDFS_FBK, i.e. the pulse sequence rather than the coil.
                     This is the control on the control.


--------------------------------------------------------------------------
1. WHAT THE RAW DATA ACTUALLY IS  (measured, not assumed)
--------------------------------------------------------------------------
Both releases have the identical, boringly simple layout::

    kspace              (slices, coils, kx, ky)  complex64, Cartesian
    reconstruction_rss  (slices, rx, ry)         float32, the vendor image
    ismrmrd_header                               XML
    attrs               acquisition, max, norm, patient_id

FULLY SAMPLED. accelerationFactor/kspace_encoding_step_1 = 1 in every header,
there are no zeroed ky lines, no calibration_data, no coil_sens_maps and no
navigator echoes. That makes these by far the easiest readers in the study:
no GRAPPA, no interleave merging, no ghost correction, no gridding. Just
ifft2c, coil-combine, resample.

brain/val, 455 .h5 files present, 454 readable
    acquisition   AXT2 264, AXT1POST 97, AXFLAIR 32, AXT1 32, AXT1PRE 29
    n_coils        4:130   5:4   6:5   8:1  10:1  12:27  14:27  16:141  18:1  20:117
    matrix (kx,ky) (640,320):244  (768,396):140  (640,322):37  (640,272):12
                   (768,392):12  (640,264):3  (512,276):2  (640,260):2
                   (512,213):1  (768,324):1
    slices/file   16 x417, 14 x31, 12 x5, 10 x1
    patient_id    454 distinct values for 454 files -- one exam per patient
    unreadable    file_brain_AXT2_201_2010147.h5 opens, and its kspace shape
                  reads, but reconstruction_rss raises "bad object header
                  version number". It is excluded by open_h5's probe, which is
                  why the cohort is 454 and not 455.

knee/val, 199 genuine knee files
    acquisition   CORPD_FBK 100, CORPDFS_FBK 99
    n_coils       15 for all 199. This uniformity is the scientific point.
    matrix        (640,368):97 (640,372):89 (640,320):4 (640,356):2 (640,386):2
                  (640,338):2 (640,454):1 (640,400):1 (640,644):1
    slices/file   30 to 46
    patient_id    ***96 distinct values for 199 files*** -- see section 3.
    unreadable    none


--------------------------------------------------------------------------
2. THE DEDUPLICATION HAZARD  (this one silently destroys the experiment)
--------------------------------------------------------------------------
``knee/val`` contains a COMPLETE DUPLICATE COPY of the brain release. Measured
by direct enumeration of both folders:

    brain/val   455 files, every one named file_brain_*.h5
    knee/val    659 files = 460 named file_brain_*.h5 + 199 named file<digits>.h5

All 455 brain/val basenames appear again in knee/val, byte-for-byte the same
exams (identical patient_id attribute and identical kspace shape for all 455).
A naive ``rglob('*.h5')`` over both folders therefore yields every brain patient
TWICE. That is not merely double-counting: two copies of one patient under one
basename land on opposite sides of any split that is not basename-aware, so the
test set contains exams the model trained on and every reported number is
inflated. It would look like a clean, well-powered result.

Defence, applied unconditionally and asserted rather than trusted:

  * cohort membership is decided by BASENAME, not by directory:
        brain  <- ^file_brain_.+\\.h5$
        knee   <- ^file\\d+\\.h5$
  * every rejected file is counted and logged by reason.
  * after discovery we ASSERT no basename occurs twice within a cohort, and
    that the brain and knee basename sets are disjoint.
  * a second, independent net keyed on the ISMRMRD **measurementID**, which
    identifies an acquisition uniquely (455/455 brain and 199/199 knee files
    have distinct, non-empty ones). A repeat there is one acquisition stored
    twice, and only then is a file dropped.

    The brief proposed (acquisition, patient_id, kspace shape) for this net.
    That key is WRONG on this data and would delete real exams: it collides on
    7 knee (patient, acquisition) pairs, and every one is a genuine repeat scan
    rather than a copy. file1001916.h5 and file1000229.h5 share a patient, the
    CORPD contrast and a (35, 15, 640, 372) k-space, but their measurementIDs
    differ (41194_54605398_54605407_4396 vs 41194_54605398_54605426_4418), their
    studyTimes are 25 minutes apart, their `max`/`norm` attributes differ, and
    their mid-slice k-space differs by up to 1.5e-3. Repeats are logged and
    kept; subject_id groups them onto one patient so they cannot straddle a
    split.

Note the brief's "genuine knee = file1000*.h5" is too narrow: only 87 of the
199 knee files begin with ``file1000``, the rest run up to ``file1002570.h5``.
The correct discriminator is the fastMRI knee convention ``file<digits>.h5``,
which matches exactly 199 files and no brain file.

Also measured, and deliberately NOT used: knee/val holds 5 brain exams that do
NOT exist in brain/val (AXFLAIR_201_6002988, AXT1POST_201_6002672,
AXT1POST_208_2080169, AXT2_206_2060018, AXT2_207_2070075; all readable, all
with patient_ids absent from brain/val). ``--brain-root`` defaults to brain/val
because that is the canonical brain release and a reproducible cohort is worth
more than 5 extra files; pass ``--brain-root .../knee/val`` to include them and
the dedup logic will still refuse to cache any exam twice.

s00_inventory.guess_organ used to substring-match organ names over the whole
path, so those 460 brain files inside knee/val were attributed to 'brain' while
the directory they live in is 'knee'. It now derives organ from the path
component after the data root and records the filename-implied organ
separately, which makes the duplication visible in the stage-0 manifest instead
of quietly doubling the brain file count.


--------------------------------------------------------------------------
3. subject_id IS NOT patient_id-PER-FILE HERE, AND FOR KNEE IT MATTERS
--------------------------------------------------------------------------
Brain: 454 files, 454 distinct ``patient_id`` attributes. One exam per patient,
so the mapping is an identity.

Knee: 199 files, **96 distinct patient_id attributes**. 92 patients contribute
2 files, 3 contribute 4, 1 contributes 3 (the extras are repeat acquisitions of
a contrast the patient already has -- 7 (patient, acquisition) pairs in total).
And every one of the 96 patients contributes BOTH contrasts -- there is not a
single patient with only CORPD_FBK or only CORPDFS_FBK. So:

  * splitting the knee cohort by FILE puts the same knee in train and test, and
    since the two files of a patient differ precisely in the contrast we are
    trying to predict, the leak flatters exactly the number we care about.
    subject_id is the patient hash, and stage 3 enforces splits on it.
  * the flip side is that this design is unusually clean: because contrast is
    balanced WITHIN patient, a patient-grouped split is automatically balanced,
    and anatomy is held constant across the two classes. If phase predicts
    fat-suppression here it cannot be predicting the knee.

The brief stated "subject_id = patient_id (one exam per patient)". That holds
for brain and is false for knee; the code derives it from the data.

``patient_id`` in the index is a short readable id (``brain_0001``,
``knee_0001``) assigned by sorting the raw hashes, with the full 64-char
fastMRI hash kept in ``patient_hash``. ``subject_id`` is set to the same value,
because here the person and the split-enforcement unit coincide; the column
exists as a first-class column because stage 3 refuses to train without it.
Unlike the prostate and breast cohorts it is NOT joined from a stage-1 cohort
CSV, for the plain reason that no such table exists: stage 1 builds clinical
label tables, and these cohorts have no clinical labels. The authoritative
source is the ``patient_id`` attribute inside every HDF5 file.


--------------------------------------------------------------------------
4. RECONSTRUCTION AND ITS VALIDATION  (r = 1.000000)
--------------------------------------------------------------------------
Because the data is fully sampled Cartesian and every file ships
``reconstruction_rss``, we get per-slice ground truth for free and there is no
excuse for not using it.

    coil_images = common.ifft2c(kspace)                    (C, kx, ky)
    magnitude   = common.rss(coil_images)                  vendor-identical
    phase       = angle(common.adaptive_virtual_coil(...))  no sens maps ship

Validation: the full-resolution magnitude, centre-cropped to the shape of
``reconstruction_rss``, is correlated against ``reconstruction_rss`` for every
cached slice. Measured::

    file_brain_AXT2_200_2000022.h5   (768,396)->(384,384)  r = 1.000000000
    file_brain_AXFLAIR_200_6002471.h5 (640,320)->(320,320) r = 1.000000000
    file_brain_AXFLAIR_203_6000906.h5 (512,213)->(213,213) r = 1.000000000
    file1000000.h5   CORPDFS_FBK     (640,368)->(320,320)  r = 1.000000000
    file1001077.h5   CORPD_FBK       (640,368)->(320,320)  r = 1.000000000

Exactly 1.0, not 0.99: ifft2c of a fully sampled k-space followed by RSS *is*
the fastMRI reference reconstruction, so any deviation would be a bug rather
than noise. Orientation is the identity -- flipping up/down or left/right drops
r to 0.25-0.92, so no flip is applied (contrast the prostate readers, which
need ``np.flipud``). The run aborts the cohort if the mean drops below
``--rss-min`` (default 0.99); the prostate T2 reader reaches 0.998 on
equivalent Cartesian data, so anything lower here means the reader is wrong.

CROP CONVENTION. The centre crop uses fastMRI's ``(n - t) // 2`` offset, which
matters exactly once: brain file 512x213 -> 213x213 is an even source with an
ODD target, and there the two candidate conventions differ by one pixel. The
offset scan is unambiguous -- offset 149 = (512-213)//2 gives r = 1.000000,
offset 150 (the "keep DC at t//2" rule) gives r = 0.959. For an EVEN target the
two conventions are provably identical, which is why the same helper is safe to
use on k-space below: ``floor((n-t)/2) == n//2 - t//2`` whenever t is even.


--------------------------------------------------------------------------
5. GEOMETRY: WHAT WE CACHE AND WHY
--------------------------------------------------------------------------
Both releases oversample the readout 2x (brain AXT2: encoded FOV 440 x 226.8 mm
on a 768 x 396 matrix, recon FOV 220 x 220 on 384 x 384; knee: 280 x 161.4 mm
on 640 x 368, recon 140 x 140 on 320 x 320). We therefore

    1. undo the readout oversampling by cropping the IMAGE along kx to
       ``reconstruction_rss.shape[-2]`` -- the vendor's own recon width, read
       per file rather than assumed to be kx/2 (it is 213 of 512 for one brain
       file, not 256);
    2. keep the phase-encode axis at its FULL acquired extent, unlike the
       vendor, which crops it too (368 -> 320 for knee, 396 -> 384 for brain);
    3. resample that to 224 x 224.

Step 2 is the same decision the prostate reader documents: the vendor FOV is
mostly body, and a body mask over it degenerates. Measured foreground fraction
of ``common.body_mask``:

    brain, readout-cropped as above      0.40 - 0.45   (correct head outline)
    brain, no readout crop at all        0.68 - 0.73   (mask floods into air:
                                         with that much air the 60th percentile
                                         threshold lands inside the noise)
    knee,  readout-cropped as above      0.50 - 0.65
    knee,  no readout crop at all        0.42 - 0.47

compared with 0.60-0.62 for prostate. Cropping the readout and keeping ky is
the only variant that produces a usable mask on BOTH cohorts, and the mask is
what makes the background-only falsification control possible.

Step 3 is a DOWNSAMPLE (384 or 320 -> 224) and is done by CROPPING K-SPACE, not
by interpolating the image, for the reason spelled out in s02_prostate: a
k-space crop is an ideal anti-aliasing filter and is exact for phase, whereas a
linear image-domain kernel folds high spatial frequencies -- precisely the
coil/noise texture a phase network would latch onto -- back into the passband.
Here that choice is deliberately CONSERVATIVE: it removes high-frequency
information from the phase channel, so a positive fingerprinting result
survives it rather than being manufactured by it. Axes shorter than 224
(ky = 213 on one brain file) are zero-padded in k-space instead, which is exact
sinc interpolation.


--------------------------------------------------------------------------
6. SLICE SELECTION: WHY CENTRAL
--------------------------------------------------------------------------
``--slices-per-file`` (default 5) slices are taken from the central
``--central-frac`` (default 0.5) of each volume, evenly spaced.

  * Central, because the outermost slices of both releases are near-empty or
    anatomically degenerate -- the first and last brain slices sit at the vertex
    and skull base, the outer knee slices clip the joint -- and an empty slice
    contributes a body mask of nothing, a meaningless phase map and a free
    training example of "this file has few coils".
  * A FIXED count per file, because file lengths differ by 4x (brain 10-16,
    knee 30-46 slices). Taking every slice would weight a 46-slice knee 1.5x
    over a 30-slice one, and since slice count correlates with protocol it
    would import the confound we are trying to measure into the sampling.
  * Evenly spaced across the central band rather than 5 CONTIGUOUS central
    slices: adjacent slices are near-duplicates, so contiguous sampling gives
    5 examples' worth of compute for ~1 example's worth of information.

5 slices x 454 brain files = 2270 slices, x 199 knee files = 995, comparable to
the prostate_dwi (1359) and breast (2240) caches.


--------------------------------------------------------------------------
7. THERE IS NO TUMOUR LABEL
--------------------------------------------------------------------------
``label`` holds the CONFOUND target chosen by ``--target``:

    brain default  n_coils_bucket   1 iff n_coils >= --coil-split (16)
                                    -> 259 files positive, 195 negative
    knee  default  acquisition      1 iff CORPDFS_FBK (fat-suppressed)
                                    -> 99 files positive, 100 negative

Other targets: ``matrix`` (is this the cohort's modal matrix size),
``scanner`` (is this the modal systemModel). Every target is binary because
stage 3 asserts ``label`` is 0/1. The RAW values are all kept in their own
columns -- ``n_coils``, ``acquisition``, ``matrix_kx``, ``matrix_ky``,
``scanner``, ``field_strength``, ``device_id``, ``coil_array``, ``tr``, ``te``,
``sequence`` and the rest -- so a different target can be selected by rewriting
the ``label`` column alone, with no re-caching. ``label_target`` records which
rule produced the label and ``label_name`` the human-readable class.

``official_split``: neither release ships one (both folders are the fastMRI
*validation* split of a challenge whose test labels were never public). We
assign training/test deterministically, GROUPED BY subject and STRATIFIED by
the subject-level target, from a SHA-1 of the subject id -- no RNG, so the same
files always land on the same side regardless of order, --limit or platform.
No validation split is emitted; stage 3 carves one out of training itself,
subject-grouped, with a seed fixed independently of the run seed.


USAGE
=====
    python pipeline/s02_brainknee.py --limit 3               # smoke test, both
    python pipeline/s02_brainknee.py --cohort knee
    python pipeline/s02_brainknee.py --cohort brain --target matrix
    python pipeline/s02_brainknee.py --self-test             # no drive needed
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import re
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import (  # noqa: E402
    CACHE_DIR,
    DATA_ROOT,
    SUBJECT_COL,
    TARGET_HW,
    adaptive_virtual_coil,
    body_mask,
    fft2c,
    ifft2c,
    normalize_magnitude,
    resample_phase,
    resample_to,
    rss,
)

logger = logging.getLogger("s02_brainknee")

COHORTS = ("brain", "knee")

DEFAULT_ROOTS = {
    "brain": DATA_ROOT / "brain" / "val",
    "knee": DATA_ROOT / "knee" / "val",
}

# Cohort membership is decided by BASENAME. See docstring section 2: knee/val
# holds a full duplicate copy of the brain release, so the directory a file
# sits in says nothing about which cohort it belongs to.
#
# The knee pattern is `file<digits>.h5`, NOT `file1000*.h5`: the 199 genuine
# knee files run from file1000000.h5 to file1002570.h5 and only 87 of them
# start with "file1000".
COHORT_FNAME_RE = {
    "brain": re.compile(r"^file_brain_.+\.h5$"),
    "knee": re.compile(r"^file\d+\.h5$"),
}

# ISMRMRD tags worth carrying into the index. Every one is a candidate
# acquisition fingerprint, which is the whole point of these two cohorts.
# Which side of a two-class acquisition target is class 1. The knee question is
# "can phase see the fat-suppression pulse", so the fat-suppressed sequence is
# the positive class. Stated explicitly because ASCII sorts 'CORPDFS_FBK'
# before 'CORPD_FBK' and an alphabetical rule would silently invert it.
ACQ_POSITIVE_CLASSES = {"CORPDFS_FBK"}

HEADER_TAGS = {
    "vendor": "systemVendor",
    "scanner": "systemModel",
    "field_strength": "systemFieldStrength_T",
    "institution": "institutionName",
    "receiver_channels": "receiverChannels",
    "noise_bandwidth": "relativeReceiverNoiseBandwidth",
    "measurement_id": "measurementID",
    "protocol": "protocolName",
    "patient_position": "patientPosition",
    "frame_of_reference": "frameOfReferenceUID",
    "h1_freq": "H1resonanceFrequency_Hz",
    "trajectory": "trajectory",
    "tr": "TR",
    "te": "TE",
    "ti": "TI",
    "flip_angle": "flipAngle_deg",
    "sequence": "sequence_type",
    "echo_spacing": "echo_spacing",
}

INDEX_COLUMNS = [
    # --- cache contract ----------------------------------------------------
    # subject_id is the split-enforcement unit. For knee it genuinely collapses
    # 199 files onto 96 patients (docstring section 3); for brain it is 1:1.
    # It is written here rather than joined from a stage-1 cohort CSV because
    # these cohorts have no clinical labels and therefore no stage-1 table --
    # the authoritative source is the patient_id attribute in each HDF5 file.
    "idx", "patient_id", SUBJECT_COL, "file", "slice", "label", "raw_label",
    "official_split", "folder", "acq",
    # --- what the label actually is ----------------------------------------
    "label_target", "label_name",
    # --- confound targets, kept raw so other targets need no re-caching -----
    "n_coils", "matrix_kx", "matrix_ky", "matrix", "n_slices",
    "recon_x", "recon_y",
    # --- QC ----------------------------------------------------------------
    "recon_ncc", "mask_frac", "phase_std", "qc",
    # --- provenance / hardware fingerprints --------------------------------
    "cohort", "patient_hash", "device_id", "coil_array", "source_dir",
    "vendor", "scanner", "field_strength", "institution", "receiver_channels",
    "noise_bandwidth", "measurement_id", "protocol", "patient_position",
    "frame_of_reference", "h1_freq", "trajectory",
    "tr", "te", "ti", "flip_angle", "sequence", "echo_spacing",
    "enc_matrix_x", "enc_matrix_y", "enc_fov_x", "enc_fov_y",
    "recon_matrix_x", "recon_matrix_y", "recon_fov_x", "recon_fov_y",
    "accel_1", "calibration_mode",
]


# ==========================================================================
# geometry helpers
# ==========================================================================

def center_crop_or_pad2d(arr: np.ndarray, hw) -> np.ndarray:
    """
    Centre crop OR zero-pad the last two axes to `hw`, one axis at a time.

    Offsets use fastMRI's ``(n - t) // 2``. Two properties earn this its own
    function instead of an inline slice:

    * On an IMAGE it reproduces the vendor crop exactly, including the awkward
      even-source/odd-target case (512 -> 213, offset 149). Using the
      DC-alignment rule ``n//2 - t//2`` there gives offset 150 and drops the
      correlation against reconstruction_rss from 1.000000 to 0.959.
    * On K-SPACE it keeps the DC sample where ifftshift expects it (index
      t//2) for every EVEN target, because floor((n-t)/2) == n//2 - t//2
      whenever t is even -- for odd n that identity is
      (n-1)/2 - t/2 == n//2 - t//2. Our k-space target is always 224.

    Padding is the mirror image: a negative offset becomes a left pad of
    -offset, which is exact sinc interpolation when applied to k-space.
    """
    out = arr
    for axis, target in zip((-2, -1), hw):
        n = out.shape[axis]
        off = (n - target) // 2
        if off > 0:
            out = np.take(out, np.arange(off, off + target), axis=axis)
        elif off < 0:
            pad = [(0, 0)] * out.ndim
            pad[axis % out.ndim] = (-off, target - n + off)
            out = np.pad(out, pad)
    return out


def resize_kspace(coil_images: np.ndarray, target_hw) -> np.ndarray:
    """
    Resample coil images to `target_hw` through k-space.

    fft2c -> centre crop/pad -> ifft2c. Cropping is an ideal anti-aliasing
    low-pass (and exact for phase, since nothing interpolates a wrapped
    quantity); padding is exact sinc interpolation. See docstring section 5.
    """
    if tuple(coil_images.shape[-2:]) == tuple(target_hw):
        return coil_images
    return ifft2c(center_crop_or_pad2d(fft2c(coil_images), target_hw))


def ncc(a: np.ndarray, b: np.ndarray) -> float:
    """Normalised cross-correlation of two same-shape real images."""
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    a = a - a.mean()
    b = b - b.mean()
    denom = np.linalg.norm(a) * np.linalg.norm(b)
    return float(a @ b / denom) if denom > 0 else 0.0


def select_central_slices(n_slices: int, want: int, central_frac: float = 0.5) -> list:
    """
    `want` evenly spaced slice indices from the central `central_frac` of a volume.

    Central because the outer slices of both releases are near-empty; evenly
    spaced rather than contiguous because adjacent slices are near-duplicates.
    See docstring section 6. Returns sorted, de-duplicated indices; a volume
    shorter than `want` returns all of its slices.
    """
    if n_slices <= want:
        return list(range(n_slices))
    band = max(want, int(round(central_frac * n_slices)))
    lo = (n_slices - band) // 2
    hi = lo + band - 1
    return sorted({int(round(float(v))) for v in np.linspace(lo, hi, want)})


# ==========================================================================
# file discovery and deduplication
# ==========================================================================

def _iter_h5(root: Path):
    """Real .h5 files directly under `root`, AppleDouble sidecars excluded."""
    return sorted(p for p in root.glob("*.h5") if not p.name.startswith("._"))


def discover_files(cohort: str, roots, other_roots=()) -> tuple:
    """
    Files belonging to `cohort`, deduplicated by basename.

    Returns (paths, rejects) where `rejects` is a Counter keyed by reason. The
    caller logs it -- "duplicates rejected: 460" is evidence that the guard in
    docstring section 2 fired, and a sudden 0 would mean the layout changed
    under us.

    `other_roots` are scanned only so that cross-cohort basename collisions can
    be asserted against; nothing from them is ever returned.
    """
    rejects: Counter = Counter()
    seen: dict = {}
    paths = []
    pattern = COHORT_FNAME_RE[cohort]

    for root in roots:
        root = Path(root)
        if not root.exists():
            logger.warning("[%s] root does not exist, skipping: %s", cohort, root)
            continue
        for p in _iter_h5(root):
            if not pattern.match(p.name):
                # The overwhelming majority of these, for cohort='knee', are the
                # 460 file_brain_*.h5 copies sitting in knee/val.
                other = next((c for c, rx in COHORT_FNAME_RE.items()
                              if c != cohort and rx.match(p.name)), None)
                rejects[f"not_{cohort}" + (f"_is_{other}" if other else "_unrecognised")] += 1
                continue
            if p.name in seen:
                rejects["duplicate_basename"] += 1
                logger.debug("[%s] duplicate basename %s (keeping %s, dropping %s)",
                             cohort, p.name, seen[p.name], p)
                continue
            seen[p.name] = p
            paths.append(p)

    # Cross-cohort collision check. Cheap, and it is the assertion that would
    # fire first if fastMRI ever renamed a release.
    foreign = set()
    for root in other_roots:
        root = Path(root)
        if root.exists():
            foreign.update(p.name for p in _iter_h5(root) if pattern.match(p.name))
    overlap = foreign - set(seen)
    if overlap:
        logger.info("[%s] %d %s-named file(s) also exist outside --%s-root "
                    "(not used): e.g. %s", cohort, len(overlap), cohort, cohort,
                    sorted(overlap)[:5])

    names = [p.name for p in paths]
    assert len(names) == len(set(names)), (
        f"[{cohort}] duplicate basenames survived discovery: "
        f"{[n for n, c in Counter(names).items() if c > 1][:5]}"
    )
    return paths, rejects


def assert_cohorts_disjoint(by_cohort: dict) -> None:
    """No basename may appear in two cohorts. Fails loudly rather than warns."""
    cohorts = sorted(by_cohort)
    for i, a in enumerate(cohorts):
        for b in cohorts[i + 1:]:
            shared = {p.name for p in by_cohort[a]} & {p.name for p in by_cohort[b]}
            if shared:
                raise AssertionError(
                    f"cohorts {a!r} and {b!r} share {len(shared)} basename(s), "
                    f"e.g. {sorted(shared)[:5]} -- a file cannot be in two cohorts"
                )


# ==========================================================================
# metadata pass
# ==========================================================================

def open_h5(path: Path):
    """
    Open and *probe* an HDF5 file.

    file_brain_AXT2_201_2010147.h5 opens cleanly and reports its kspace shape,
    then raises "bad object header version number" on reconstruction_rss. The
    probe turns that into one clean exclusion at discovery time instead of a
    crash 400 files into a run.
    """
    f = h5py.File(path, "r")
    try:
        _ = f["kspace"].shape
        _ = f["reconstruction_rss"].shape
        _ = f["ismrmrd_header"][()]
    except Exception:
        f.close()
        raise
    return f


def _tag(raw: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", raw, re.S)
    return m.group(1).strip() if m else ""


def header_fields(h5file) -> dict:
    """
    Pull every acquisition descriptor the ISMRMRD header yields.

    Beyond the flat tags this recovers three derived fingerprints:

      device_id   the leading field of measurementID, which is the scanner
                  serial (brain '45219_...' = Skyra, knee '67041_...' =
                  Prisma_fit). fastMRI brain/knee headers carry no <deviceID>
                  element, unlike prostate.
      coil_array  the array name shared by every <coilName>, e.g.
                  'HeadNeck_20' from 'HeadNeck_20:1:H12' or 'TxRx_15Ch_Knee'.
                  This is the hardware the coil-count target is a proxy for.
      encoded/recon matrix and FOV, from the nested <encoding> block.
    """
    out = {k: "" for k in HEADER_TAGS}
    out.update(dict(device_id="", coil_array="", accel_1="", calibration_mode="",
                    enc_matrix_x="", enc_matrix_y="", enc_fov_x="", enc_fov_y="",
                    recon_matrix_x="", recon_matrix_y="", recon_fov_x="",
                    recon_fov_y=""))
    try:
        raw = h5file["ismrmrd_header"][()]
    except Exception:  # noqa: BLE001
        return out
    raw = raw.decode("utf-8", "replace") if isinstance(raw, bytes) else str(raw)

    for key, tag in HEADER_TAGS.items():
        out[key] = _tag(raw, tag)

    if out["measurement_id"]:
        out["device_id"] = out["measurement_id"].split("_")[0]

    coils = re.findall(r"<coilName>(.*?)</coilName>", raw, re.S)
    arrays = {c.split(":")[0].strip() for c in coils}
    out["coil_array"] = sorted(arrays)[0] if len(arrays) == 1 else "|".join(sorted(arrays))

    for space, prefix in (("encodedSpace", "enc"), ("reconSpace", "recon")):
        block = re.search(rf"<{space}>(.*?)</{space}>", raw, re.S)
        if not block:
            continue
        body = block.group(1)
        mat = re.search(r"<matrixSize>(.*?)</matrixSize>", body, re.S)
        fov = re.search(r"<fieldOfView_mm>(.*?)</fieldOfView_mm>", body, re.S)
        if mat:
            out[f"{prefix}_matrix_x"] = _tag(mat.group(1), "x")
            out[f"{prefix}_matrix_y"] = _tag(mat.group(1), "y")
        if fov:
            out[f"{prefix}_fov_x"] = _tag(fov.group(1), "x")
            out[f"{prefix}_fov_y"] = _tag(fov.group(1), "y")

    accel = re.search(r"<accelerationFactor>(.*?)</accelerationFactor>", raw, re.S)
    if accel:
        out["accel_1"] = _tag(accel.group(1), "kspace_encoding_step_1")
    out["calibration_mode"] = _tag(raw, "calibrationMode")
    return out


def scan_metadata(cohort: str, paths) -> tuple:
    """
    First pass: attributes, shapes and header for every file. No k-space read.

    Cheap (measured: 654 files in 25 s over USB) and it is what makes the
    cohort-level decisions -- subject numbering, the modal matrix, the
    stratified split -- consistent across the whole run instead of dependent on
    the order files happen to be reconstructed in.

    Returns (records, failures). A record is a plain dict; failures carry the
    exception type so the report can name them.
    """
    records, failures = [], []
    for path in paths:
        try:
            f = open_h5(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] UNREADABLE %s (%s: %s)",
                           cohort, path.name, type(exc).__name__, exc)
            failures.append((path.name, type(exc).__name__))
            continue
        with f:
            attrs = dict(f.attrs)
            n_slices, n_coils, kx, ky = (int(v) for v in f["kspace"].shape)
            rec = dict(
                path=path,
                file=path.name,
                folder=path.parent.name,
                source_dir=str(path.parent),
                cohort=cohort,
                acq=str(attrs.get("acquisition", "")),
                patient_hash=str(attrs.get("patient_id", "")),
                n_slices=n_slices,
                n_coils=n_coils,
                matrix_kx=kx,
                matrix_ky=ky,
                matrix=f"{kx}x{ky}",
                recon_x=int(f["reconstruction_rss"].shape[-2]),
                recon_y=int(f["reconstruction_rss"].shape[-1]),
            )
            rec.update(header_fields(f))
        records.append(rec)

    kept = dedup_by_exam(cohort, records)
    return kept, failures


def dedup_by_exam(cohort: str, records) -> list:
    """
    Second dedup net, independent of basename: drop files that are the SAME
    ACQUISITION stored twice.

    Exam identity is the ISMRMRD measurementID -- scanner serial plus the
    measurement counters, unique per acquisition. Verified on this drive:
    455/455 brain and 199/199 knee files have distinct, non-empty
    measurementIDs. A repeated one therefore means one acquisition has been
    stored under two names, which the basename net cannot see.

    (acquisition, patient_id, kspace shape) is deliberately NOT the identity,
    although the brief suggested it. It fires on 7 knee (patient, acquisition)
    pairs and every one of them is a REAL repeat scan, not a copy:
    file1001916.h5 and file1000229.h5 share a patient, the CORPD contrast and a
    (35, 15, 640, 372) k-space, but carry different measurementIDs
    (41194_54605398_54605407_4396 vs 41194_54605398_54605426_4418), studyTimes
    25 minutes apart, different `max`/`norm` attributes, and mid-slice k-space
    that differs by up to 1.5e-3. Keying on shape would silently delete one
    half of every repeat pair -- real data, thrown away to fix a duplication
    problem it is not part of.

    Repeats are reported instead, because a repeat acquisition is a confound in
    its own right and because subject_id has to collapse them onto one patient
    (it does; that is what stops them straddling a split).
    """
    keyed = defaultdict(list)
    for r in records:
        keyed[r.get("measurement_id") or f"__no_id__{r['file']}"].append(r)
    kept, dropped = [], []
    for group in keyed.values():
        kept.append(group[0])
        dropped.extend((extra["file"], group[0]["file"]) for extra in group[1:])
    if dropped:
        logger.error("[%s] %d file(s) share a measurementID with an already "
                     "accepted file and were dropped as true duplicates: %s",
                     cohort, len(dropped), dropped[:5])

    repeats = Counter((r["patient_hash"], r["acq"]) for r in kept)
    n_repeat_files = sum(v - 1 for v in repeats.values() if v > 1)
    if n_repeat_files:
        logger.info("[%s] %d (patient, acquisition) pair(s) hold more than one "
                    "acquisition (%d extra file(s)). Distinct measurementIDs, "
                    "i.e. genuine repeat scans -- kept, and grouped onto one "
                    "subject_id so they cannot straddle a split.",
                    cohort, sum(1 for v in repeats.values() if v > 1), n_repeat_files)
    kept.sort(key=lambda r: r["file"])
    return kept


def assign_subject_ids(cohort: str, records) -> None:
    """
    Give every record a short readable patient_id/subject_id in place.

    Sorted by the raw fastMRI hash so the numbering is deterministic and
    independent of file order, --limit, or filesystem enumeration.

    subject_id == patient_id here because the person IS the split unit: one
    physical patient, one or more files. For knee that is a real 199 -> 96
    collapse (docstring section 3), not a formality.
    """
    hashes = sorted({r["patient_hash"] for r in records})
    lut = {h: f"{cohort}_{i + 1:04d}" for i, h in enumerate(hashes)}
    for r in records:
        r["patient_id"] = lut[r["patient_hash"]]
        r[SUBJECT_COL] = lut[r["patient_hash"]]


# ==========================================================================
# confound targets
# ==========================================================================

def _binary_from_modal(records, field: str, label_target: str):
    """label = 1 iff `field` equals the cohort's modal value. raw = the value."""
    modal = Counter(r[field] for r in records).most_common(1)[0][0]
    for r in records:
        r["label"] = int(r[field] == modal)
        r["raw_label"] = r[field]
        r["label_target"] = label_target
        r["label_name"] = f"{field}={modal}" if r["label"] else f"{field}!={modal}"
    return f"{label_target}: 1 = {field} == {modal!r} (modal), 0 = anything else"


def apply_target(cohort: str, records, target: str, coil_split: int) -> str:
    """
    Set label / raw_label / label_target / label_name on every record.

    Every target is BINARY because stage 3 asserts ``label`` is 0/1. The raw
    driving values stay in their own index columns, so switching target later
    is a rewrite of one column and never a re-cache.

    Returns a one-line human description of the rule, which is logged and
    printed in the report so no result can be read without knowing what the
    positive class was.
    """
    if target == "auto":
        target = "n_coils_bucket" if cohort == "brain" else "acquisition"

    if target == "n_coils_bucket":
        for r in records:
            r["label"] = int(r["n_coils"] >= coil_split)
            r["raw_label"] = r["n_coils"]
            r["label_target"] = f"n_coils>={coil_split}"
            r["label_name"] = f">={coil_split}" if r["label"] else f"<{coil_split}"
        return (f"n_coils_bucket: 1 = receive-coil count >= {coil_split}, "
                f"0 = < {coil_split} (a pure hardware property)")

    if target == "acquisition":
        classes = sorted({r["acq"] for r in records})
        if len(classes) != 2:
            raise SystemExit(
                f"--target acquisition needs exactly 2 acquisition values in "
                f"cohort {cohort!r}, found {len(classes)}: {classes}. The brain "
                "cohort has 5 (AXT2/AXT1POST/AXFLAIR/AXT1/AXT1PRE) and stage 3 "
                "only accepts a binary label; use --target n_coils_bucket, or "
                "recode the acq column downstream for a multi-class head."
            )
        # Do NOT let alphabetical order decide the positive class. ASCII sorts
        # 'CORPDFS_FBK' BEFORE 'CORPD_FBK' ('F' < '_'), so sorted()[-1] would
        # quietly make the non-fat-suppressed sequence positive and every
        # downstream sentence about "the fat-suppressed class" would be
        # inverted. Named classes win; anything else is chosen by sort order
        # and logged so the choice is never implicit.
        named = [c for c in classes if c in ACQ_POSITIVE_CLASSES]
        if len(named) == 1:
            pos = named[0]
        else:
            pos = classes[-1]
            logger.warning("[%s] neither of %s is a known positive class; "
                           "defaulting to %r as the positive class",
                           cohort, classes, pos)
        neg = next(c for c in classes if c != pos)
        for r in records:
            r["label"] = int(r["acq"] == pos)
            r["raw_label"] = r["acq"]
            r["label_target"] = f"acquisition:{pos}_vs_{neg}"
            r["label_name"] = r["acq"]
        return f"acquisition: 1 = {pos}, 0 = {neg} (pulse sequence, not pathology)"

    if target == "matrix":
        return _binary_from_modal(records, "matrix", "matrix")
    if target == "scanner":
        return _binary_from_modal(records, "scanner", "scanner")

    raise SystemExit(f"unknown --target {target!r}")


# ==========================================================================
# splits
# ==========================================================================

def _subject_hash(subject_id: str, salt: str) -> str:
    return hashlib.sha1(f"{salt}|{subject_id}".encode()).hexdigest()


def assign_splits(records, test_frac: float, salt: str) -> None:
    """
    Deterministic subject-grouped, target-stratified training/test assignment.

    Neither release ships an official split, so one has to be invented -- and
    an invented split is only defensible if it is (a) reproducible bit for bit,
    (b) grouped so no subject straddles it, and (c) stratified so neither side
    comes out single-class.

    (a) comes from ranking subjects by SHA-1(salt|subject_id) rather than
        drawing from an RNG: the assignment does not depend on file order,
        --limit, dict iteration or platform.
    (b) is enforced by assigning whole subjects. This is not cosmetic for knee,
        where the two files of a patient differ exactly in the contrast we are
        predicting; a file-level split would leak the answer.
    (c) is done inside each subject-level class. Knee is stratified for free
        (every patient contributes both contrasts, so every subject is class
        "mixed"); brain is not, since coil count is a per-exam property.

    A subject whose files disagree on the target is put in its own stratum
    keyed on the sorted set of its labels, so mixed subjects are split evenly
    too rather than piling onto one side.

    No validation split is emitted: stage 3 carves one from training itself,
    subject-grouped, with a seed fixed independently of the run seed so that
    the magnitude / phase / both arms all select epochs on identical data.
    """
    by_subject = defaultdict(list)
    for r in records:
        by_subject[r[SUBJECT_COL]].append(r)

    strata = defaultdict(list)
    for sid, group in by_subject.items():
        strata[tuple(sorted({int(g["label"]) for g in group}))].append(sid)

    test_subjects = set()
    for _, sids in sorted(strata.items()):
        ordered = sorted(sids, key=lambda s: _subject_hash(s, salt))
        n_test = int(round(test_frac * len(ordered)))
        if len(ordered) > 1:
            n_test = min(max(n_test, 1), len(ordered) - 1)
        else:
            n_test = 0
        test_subjects.update(ordered[:n_test])

    for r in records:
        r["official_split"] = "test" if r[SUBJECT_COL] in test_subjects else "training"


# ==========================================================================
# reconstruction
# ==========================================================================

def slice_recon(f, s: int, target_hw):
    """
    Reconstruct one slice.

    Returns (mag_vendor_fov, mag_target, phase_target):

      mag_vendor_fov  full-resolution RSS centre-cropped to the shape of
                      reconstruction_rss. Exists only to be correlated against
                      the vendor image -- it is the validation, not the cache.
      mag_target      the cached magnitude, readout oversampling undone and
                      resampled to target_hw through k-space.
      phase_target    the cached phase in radians on [-pi, pi], from the
                      adaptive virtual coil combination of the SAME complex
                      image. No sensitivity maps ship with either release, so
                      a sensitivity-free combination is the only option; RSS
                      cannot be used because it discards phase entirely.

    Orientation is the identity: no flip is applied, and flipping either axis
    drops the vendor correlation from 1.000000 to 0.25-0.92 (docstring 4).
    """
    k = f["kspace"][s].astype(np.complex64)          # (C, kx, ky)
    coil_images = ifft2c(k)

    recon_hw = f["reconstruction_rss"].shape[-2:]
    mag_vendor = center_crop_or_pad2d(rss(coil_images, coil_axis=0), recon_hw)

    # Undo the 2x readout oversampling using the vendor's own recon width --
    # read per file, because it is 213 of 512 for one brain exam, not kx/2 --
    # and keep the phase-encode axis at full extent so the body mask has air
    # to work with (docstring section 5).
    coil_images = center_crop_or_pad2d(coil_images, (recon_hw[0], k.shape[-1]))
    coil_small = resize_kspace(coil_images, tuple(target_hw))

    mag = rss(coil_small, coil_axis=0)
    phase = np.angle(adaptive_virtual_coil(coil_small, coil_axis=0))
    return mag_vendor, mag, phase


def finalize(mag, phase, target_hw):
    """
    Normalise, resample to the network grid, and build the body mask.

    resample_to / resample_phase are no-ops here because slice_recon already
    delivers target_hw via the k-space crop; they stay in the path so this
    reader cannot silently diverge from the prostate and breast readers if the
    geometry ever changes.
    """
    m = resample_to(normalize_magnitude(mag), target_hw, order=1)
    m = np.clip(m, 0.0, 1.0)
    p = resample_phase(phase, target_hw)
    return m, p, body_mask(m)


# ==========================================================================
# cache writer
# ==========================================================================

class CacheWriter:
    """
    Appends slices to <cohort>.h5 and rewrites <cohort>_index.csv after every
    file, so a crash (or an unplugged drive) costs at most one exam.

    Unlike the prostate and breast writers this one does NOT join subject_id
    from a stage-1 cohort CSV: no such table exists for brain or knee, because
    stage 1 builds clinical label tables and these cohorts carry no clinical
    labels at all. subject_id comes from the fastMRI patient_id attribute
    inside each file (see assign_subject_ids) and is asserted non-empty before
    any row is written -- an empty-string subject_id is not null, so it would
    sail through every notna() check downstream and then group the entire
    cohort into a single "subject".
    """

    def __init__(self, cache_dir: Path, cohort: str, target_hw=TARGET_HW):
        cache_dir.mkdir(parents=True, exist_ok=True)
        self.cohort = cohort
        self.h5_path = cache_dir / f"{cohort}.h5"
        self.csv_path = cache_dir / f"{cohort}_index.csv"
        self.rows = []
        self.f = h5py.File(self.h5_path, "w")
        h, w = target_hw
        opts = dict(maxshape=(None, h, w), chunks=(1, h, w), compression="lzf")
        self.f.create_dataset("mag", (0, h, w), dtype="float16", **opts)
        self.f.create_dataset("phase", (0, h, w), dtype="float16", **opts)
        self.f.create_dataset("mask", (0, h, w), dtype=np.bool_, **opts)

    def append(self, mag, phase, mask, row: dict):
        sid = str(row.get(SUBJECT_COL, "")).strip()
        if not sid:
            raise ValueError(
                f"[{self.cohort}] {row.get('file')} has no {SUBJECT_COL}; refusing "
                "to write a row stage 3 could not enforce a split on"
            )
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

    def flush(self):
        self.f.flush()
        if self.rows:
            df = pd.DataFrame(self.rows)
            ordered = INDEX_COLUMNS + [c for c in df.columns if c not in INDEX_COLUMNS]
            for col in INDEX_COLUMNS:
                if col not in df.columns:
                    df[col] = ""
            df[ordered].to_csv(self.csv_path, index=False)

    def close(self):
        self.flush()
        self.f.close()


# ==========================================================================
# per-cohort driver
# ==========================================================================

def slice_qc(mask_frac: float, lo: float, hi: float) -> str:
    """
    Flag slices whose body mask has degenerated.

    common.body_mask thresholds at a fixed percentile, so on a slice with very
    little anatomy (the vertex slices of a brain volume) the threshold lands
    inside the noise floor and morphological closing then floods the mask over
    almost the whole frame. Measured on brain: most slices sit at 0.40-0.55 but
    the outermost cached slice of one AXFLAIR volume reaches 0.955.

    A mask that covers 95% of the image leaves no background, so those slices
    cannot support the background-only falsification control. They are cached
    with a flag rather than dropped -- dropping them would make slice count
    depend on anatomy, which is its own confound -- and the report counts them.
    """
    if mask_frac > hi:
        return "mask_flooded"
    if mask_frac < lo:
        return "mask_empty"
    return "ok"


def process(cohort: str, records, writer: CacheWriter, args) -> dict:
    stats = {"files": 0, "slices": 0, "failed": [], "ncc": [], "per_file_ncc": [],
             "qc": Counter()}
    meta_cols = [c for c in INDEX_COLUMNS
                 if c not in ("idx", "slice", "recon_ncc", "mask_frac",
                              "phase_std", "qc")]

    for rec in records:
        path = rec["path"]
        try:
            f = open_h5(path)
        except Exception as exc:  # noqa: BLE001
            logger.warning("[%s] UNREADABLE %s (%s: %s)",
                           cohort, path.name, type(exc).__name__, exc)
            stats["failed"].append((path.name, type(exc).__name__))
            continue

        t0 = time.time()
        with f:
            slices = select_central_slices(rec["n_slices"], args.slices_per_file,
                                           args.central_frac)
            file_ncc = []
            for s in slices:
                mag_vendor, mag_small, phase_small = slice_recon(
                    f, s, tuple(args.target_hw))
                r = ncc(mag_vendor, f["reconstruction_rss"][s])
                file_ncc.append(r)
                stats["ncc"].append(r)

                mag, phase, mask = finalize(mag_small, phase_small,
                                            tuple(args.target_hw))
                mask_frac = float(mask.mean())
                qc = slice_qc(mask_frac, args.mask_frac_min, args.mask_frac_max)
                stats["qc"][qc] += 1
                row = {c: rec.get(c, "") for c in meta_cols}
                row.update(slice=s,
                           recon_ncc=round(r, 6),
                           mask_frac=round(mask_frac, 4),
                           phase_std=round(float(np.std(phase)), 4),
                           qc=qc)
                writer.append(mag, phase, mask, row)
                stats["slices"] += 1

        writer.flush()
        stats["files"] += 1
        stats["per_file_ncc"].append((path.name, float(np.mean(file_ncc))))
        logger.info("[%s] %s: %d slices %s in %.1fs | n_coils=%d %s %dx%d | "
                    "recon_ncc vs reconstruction_rss mean=%.6f min=%.6f",
                    cohort, path.name, len(slices), slices, time.time() - t0,
                    rec["n_coils"], rec["acq"], rec["matrix_kx"], rec["matrix_ky"],
                    float(np.mean(file_ncc)), float(np.min(file_ncc)))
    return stats


# ==========================================================================
# report
# ==========================================================================

def report(cohort: str, stats: dict, writer: CacheWriter, rejects: Counter,
           meta_failures, target_desc: str, rss_min: float) -> bool:
    """Print the cohort summary. Returns False if the validation gate failed."""
    ok = True
    print("\n" + "=" * 76)
    print(f"COHORT {cohort}   (CONFOUND cohort -- no tumour label exists for this data)")
    print("=" * 76)
    print(f"  target          : {target_desc}")
    print(f"  files processed : {stats['files']}")
    print(f"  slices cached   : {stats['slices']}")
    if rejects:
        print("  DEDUP -- files rejected at discovery:")
        for reason, n in rejects.most_common():
            print(f"      {reason:<34} {n}")
    if meta_failures:
        print(f"  unreadable files: {len(meta_failures)}")
        for name, why in meta_failures:
            print(f"      {name}: {why}")
    if stats["failed"]:
        print(f"  failed in recon : {len(stats['failed'])}")
        for name, why in stats["failed"]:
            print(f"      {name}: {why}")

    if stats["ncc"]:
        arr = np.array(stats["ncc"])
        per_file = np.array([v for _, v in stats["per_file_ncc"]])
        print("  RECONSTRUCTION VALIDATION vs the vendor reconstruction_rss "
              "shipped in the same file")
        print(f"    per cached slice: n={arr.size}  mean={arr.mean():.6f}  "
              f"median={np.median(arr):.6f}  min={arr.min():.6f}  max={arr.max():.6f}")
        print(f"    per file        : n={per_file.size}  mean={per_file.mean():.6f}  "
              f"min={per_file.min():.6f}  max={per_file.max():.6f}")
        bad = np.array(stats["per_file_ncc"], dtype=object)[per_file < rss_min]
        if len(bad):
            ok = False
            print(f"    *** {len(bad)} FILE(S) BELOW {rss_min:.3f} -- the reader is "
                  f"WRONG, not merely noisy: this data is fully sampled Cartesian "
                  f"and ifft2c+rss IS the reference reconstruction ***")
            for name, v in bad[:10]:
                print(f"        {name}: {v:.6f}")
        else:
            print(f"    all {per_file.size} file(s) >= {rss_min:.3f} -- reconstruction "
                  "validated against per-slice ground truth")

    if writer.rows:
        df = pd.DataFrame(writer.rows)
        print(f"  patients        : {df.patient_id.nunique()}")
        print(f"  subjects        : {df[SUBJECT_COL].nunique()}  "
              f"(split-enforcement unit)")
        print(f"  label balance   : {df.label.value_counts().to_dict()}  "
              f"({df.label_target.iloc[0]})")
        print(f"  official_split  : {df.official_split.value_counts().to_dict()}")
        span = df.groupby(SUBJECT_COL)["official_split"].nunique()
        straddling = sorted(span[span > 1].index)
        print(f"  subjects spanning splits: {len(straddling)}"
              + (f" {straddling[:5]} <-- LEAK" if straddling else " (none)"))
        if straddling:
            ok = False
        for col in ("n_coils", "acq", "matrix", "scanner", "coil_array", "device_id"):
            if col in df.columns:
                vc = df[col].value_counts().sort_index()
                shown = dict(list(vc.items())[:12])
                print(f"  {col:<15} : {shown}"
                      + (f" (+{len(vc) - 12} more)" if len(vc) > 12 else ""))
        print(f"  mask_frac       : mean={df.mask_frac.mean():.3f} "
              f"min={df.mask_frac.min():.3f} max={df.mask_frac.max():.3f}")
        print(f"  slice qc        : {dict(stats['qc'])}")
        n_bad = sum(v for k, v in stats["qc"].items() if k != "ok")
        if n_bad:
            print(f"      {n_bad}/{stats['slices']} slice(s) have a degenerate body "
                  "mask and cannot support the background-only control; they are "
                  "cached with qc != 'ok' so downstream can exclude them")
        print(f"  cache           : {writer.h5_path}")
        print(f"  index           : {writer.csv_path}")
    return ok


# ==========================================================================
# self-test  (no drive required)
# ==========================================================================

def self_test() -> int:
    checks, failed = 0, []

    def check(name, cond):
        nonlocal checks
        checks += 1
        if not cond:
            failed.append(name)
            print(f"  FAIL  {name}")
        else:
            print(f"  ok    {name}")

    print("s02_brainknee self-test")

    # --- 1. crop/pad conventions -----------------------------------------
    a = np.arange(512 * 213, dtype=np.float64).reshape(512, 213)
    check("image crop 512->213 uses the fastMRI offset 149 (measured r=1.0; "
          "offset 150 gives 0.959)",
          np.array_equal(center_crop_or_pad2d(a, (213, 213)), a[149:149 + 213, :]))
    for n in (213, 264, 320, 368, 396, 640, 768):
        k = np.zeros(n, np.complex128)
        k[n // 2] = 1.0                       # DC of a centred k-space
        out = center_crop_or_pad2d(k[None, :].repeat(2, 0), (2, 224))[0]
        check(f"k-space resize {n}->224 keeps DC at index 112",
              int(np.argmax(np.abs(out))) == 112 and abs(out[112] - 1.0) < 1e-12)
    rng = np.random.default_rng(0)
    img = np.zeros((2, 64, 64), np.complex128)
    img[:, 26:38, 26:38] = rng.normal(size=(2, 12, 12)) + 1j * rng.normal(size=(2, 12, 12))
    up = resize_kspace(img, (128, 128))
    l2 = lambda a: float(np.linalg.norm(a))  # noqa: E731
    check("k-space zero-pad (upsample) is norm preserving, i.e. exact sinc "
          "interpolation and not a rescale",
          abs(l2(up) - l2(img)) / l2(img) < 1e-9)
    check("upsample then downsample round-trips to the original",
          np.allclose(resize_kspace(up, (64, 64)), img, atol=1e-9))
    check("k-space crop (downsample) removes energy rather than folding it "
          "back in (that is the anti-aliasing property)",
          l2(resize_kspace(img, (32, 32))) < l2(img))
    check("resize_kspace is a no-op at matching size",
          resize_kspace(img, (64, 64)) is img)
    ramp = np.exp(2j * np.pi * np.add.outer(np.arange(64) * 3.0 / 64,
                                            np.arange(64) * 2.0 / 64))[None]
    small = resize_kspace(ramp, (32, 32))
    check("a smooth phase ramp survives the resize unwrapped (phase is never "
          "interpolated as a wrapped quantity)",
          np.abs(np.abs(small) - np.abs(small).mean()).max() < 1e-6)

    # --- 2. cohort membership / dedup ------------------------------------
    check("brain regex matches file_brain_AXT2_200_2000019.h5",
          bool(COHORT_FNAME_RE["brain"].match("file_brain_AXT2_200_2000019.h5")))
    check("knee regex matches file1002570.h5 (NOT just file1000*)",
          bool(COHORT_FNAME_RE["knee"].match("file1002570.h5")))
    check("knee regex rejects file_brain_*.h5 (the 460 copies in knee/val)",
          not COHORT_FNAME_RE["knee"].match("file_brain_AXT2_200_2000019.h5"))
    check("brain regex rejects file1000000.h5",
          not COHORT_FNAME_RE["brain"].match("file1000000.h5"))

    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        bdir, kdir = td / "brain" / "val", td / "knee" / "val"
        bdir.mkdir(parents=True)
        kdir.mkdir(parents=True)
        brain_names = [f"file_brain_AXT2_200_200{i:04d}.h5" for i in range(7)]
        knee_names = [f"file100{i:04d}.h5" for i in range(4)]
        for n in brain_names:
            (bdir / n).touch()
            (kdir / n).touch()          # the duplicate copy
        for n in knee_names:
            (kdir / n).touch()
        (kdir / "._file1000000.h5").touch()   # AppleDouble sidecar
        bpaths, brej = discover_files("brain", [bdir], other_roots=[kdir])
        kpaths, krej = discover_files("knee", [kdir], other_roots=[bdir])
        check("brain discovery finds all 7 brain files", len(bpaths) == 7)
        check("knee discovery finds 4 knee files, not 11", len(kpaths) == 4)
        check("knee discovery rejects the 7 brain copies by name",
              krej["not_knee_is_brain"] == 7)
        check("AppleDouble sidecars never reach discovery",
              all(not p.name.startswith("._") for p in kpaths))
        assert_cohorts_disjoint({"brain": bpaths, "knee": kpaths})
        check("cohorts are basename-disjoint", True)
        both, rej2 = discover_files("brain", [bdir, kdir])
        check("scanning brain/val AND knee/val still yields 7 brain files, "
              "not 14 (duplicate_basename fires)",
              len(both) == 7 and rej2["duplicate_basename"] == 7)
        try:
            assert_cohorts_disjoint({"a": [bdir / brain_names[0]],
                                     "b": [kdir / brain_names[0]]})
            check("assert_cohorts_disjoint raises on a shared basename", False)
        except AssertionError:
            check("assert_cohorts_disjoint raises on a shared basename", True)

    # The measured knee repeat pair: same patient, same contrast, same shape,
    # DIFFERENT measurementID. Both files must survive.
    recs = [dict(file="file1000229.h5", patient_hash="p1", acq="CORPD_FBK",
                 matrix="640x372", measurement_id="41194_54605398_54605407_4396"),
            dict(file="file1001916.h5", patient_hash="p1", acq="CORPD_FBK",
                 matrix="640x372", measurement_id="41194_54605398_54605426_4418"),
            dict(file="copy_of_229.h5", patient_hash="p1", acq="CORPD_FBK",
                 matrix="640x372", measurement_id="41194_54605398_54605407_4396")]
    kept = dedup_by_exam("knee", recs)
    check("a true duplicate (same measurementID) is dropped",
          "copy_of_229.h5" not in {r["file"] for r in kept})
    check("a genuine repeat scan (same patient+contrast+shape, different "
          "measurementID) is KEPT -- 7 such pairs exist in knee/val",
          {r["file"] for r in kept} == {"file1000229.h5", "file1001916.h5"})
    check("dedup output is sorted by filename",
          [r["file"] for r in kept] == sorted(r["file"] for r in kept))

    # --- 3. slice selection ----------------------------------------------
    for n, want in ((16, 5), (35, 5), (46, 5), (10, 5), (30, 5)):
        sel = select_central_slices(n, want, 0.5)
        check(f"central slices n={n} want={want} -> {sel}: count, uniqueness, "
              "no edge slice",
              len(sel) == want and len(set(sel)) == want
              and 0 < min(sel) and max(sel) < n - 1)
    check("a volume shorter than the request returns all of its slices",
          select_central_slices(3, 5) == [0, 1, 2])
    check("selection is sorted and reproducible",
          select_central_slices(35, 5) == sorted(select_central_slices(35, 5)))
    check("degenerate body masks are flagged, not silently cached",
          (slice_qc(0.45, 0.05, 0.90), slice_qc(0.955, 0.05, 0.90),
           slice_qc(0.0, 0.05, 0.90)) == ("ok", "mask_flooded", "mask_empty"))

    # --- 4. targets -------------------------------------------------------
    recs = [dict(n_coils=c, acq=a, matrix=m, scanner="Skyra")
            for c, a, m in [(4, "AXT2", "640x320"), (20, "AXT2", "768x396"),
                            (16, "AXT1", "640x320"), (12, "AXFLAIR", "640x320")]]
    desc = apply_target("brain", recs, "auto", 16)
    check("brain default target is the coil-count bucket", "n_coils" in desc)
    check("coil bucket labels are binary and correct",
          [r["label"] for r in recs] == [0, 1, 1, 0])
    check("raw coil count is preserved in raw_label",
          [r["raw_label"] for r in recs] == [4, 20, 16, 12])
    knee = [dict(n_coils=15, acq=a, matrix="640x368", scanner="Prisma_fit")
            for a in ("CORPD_FBK", "CORPDFS_FBK", "CORPD_FBK")]
    desc = apply_target("knee", knee, "auto", 16)
    check("knee default target is acquisition contrast", "CORPDFS_FBK" in desc)
    check("CORPDFS_FBK is the positive class",
          [r["label"] for r in knee] == [0, 1, 0])
    try:
        apply_target("brain", recs, "acquisition", 16)
        check("acquisition target refuses a 5-class cohort", False)
    except SystemExit:
        check("acquisition target refuses a 5-class cohort", True)
    apply_target("brain", recs, "matrix", 16)
    check("matrix target is binary against the modal matrix",
          sorted(r["label"] for r in recs) == [0, 1, 1, 1])

    # --- 5. splits --------------------------------------------------------
    many = []
    for i in range(96):
        sid = f"knee_{i + 1:04d}"
        for lab in (0, 1):              # every knee patient has both contrasts
            many.append({SUBJECT_COL: sid, "label": lab})
    assign_splits(many, 0.30, "phasedx")
    by_sub = defaultdict(set)
    for r in many:
        by_sub[r[SUBJECT_COL]].add(r["official_split"])
    check("no subject spans splits (knee: 2 files/patient)",
          all(len(v) == 1 for v in by_sub.values()))
    n_test = len({s for s, v in by_sub.items() if v == {"test"}})
    check(f"test fraction honoured ({n_test}/96 subjects)", 25 <= n_test <= 35)
    tr = [r["label"] for r in many if r["official_split"] == "training"]
    te = [r["label"] for r in many if r["official_split"] == "test"]
    check("both classes present on both sides",
          set(tr) == {0, 1} and set(te) == {0, 1})
    again = [dict(r) for r in many]
    for r in again:
        r.pop("official_split")
    assign_splits(again, 0.30, "phasedx")
    check("split is deterministic across calls",
          [r["official_split"] for r in again] == [r["official_split"] for r in many])
    shuffled = [dict(r) for r in reversed(again)]
    for r in shuffled:
        r.pop("official_split")
    assign_splits(shuffled, 0.30, "phasedx")
    order = {r[SUBJECT_COL]: r["official_split"] for r in shuffled}
    check("split does not depend on file order",
          all(order[r[SUBJECT_COL]] == r["official_split"] for r in many))
    imbal = [{SUBJECT_COL: f"brain_{i:04d}", "label": int(i < 60)} for i in range(100)]
    assign_splits(imbal, 0.30, "phasedx")
    for cls in (0, 1):
        sides = {r["official_split"] for r in imbal if r["label"] == cls}
        check(f"stratified: class {cls} appears in both splits", sides == {"training", "test"})

    # --- 6. subject ids ---------------------------------------------------
    recs = [dict(patient_hash=h, cohort="knee")
            for h in ["ccc", "aaa", "bbb", "aaa"]]
    assign_subject_ids("knee", recs)
    check("one short id per distinct patient hash",
          [r["patient_id"] for r in recs] == ["knee_0003", "knee_0001",
                                              "knee_0002", "knee_0001"])
    check("subject_id == patient_id (person is the split unit)",
          all(r[SUBJECT_COL] == r["patient_id"] for r in recs))
    check("two files of one patient share a subject_id",
          recs[1][SUBJECT_COL] == recs[3][SUBJECT_COL])

    # --- 7. contract columns ---------------------------------------------
    for col in ("idx", "patient_id", SUBJECT_COL, "file", "slice", "label",
                "raw_label", "official_split", "n_coils", "acq",
                "matrix_kx", "matrix_ky"):
        check(f"index schema carries {col!r}", col in INDEX_COLUMNS)
    check("index schema has no duplicate columns",
          len(INDEX_COLUMNS) == len(set(INDEX_COLUMNS)))

    print(f"\n{checks - len(failed)}/{checks} checks passed")
    if failed:
        print("FAILED: " + ", ".join(failed))
    return 1 if failed else 0


# ==========================================================================
# main
# ==========================================================================

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PhaseDx stage 2c: fastMRI brain + knee confound cohorts")
    p.add_argument("--cohort", choices=["brain", "knee", "both"], default="both")
    p.add_argument("--limit", type=int, default=None,
                   help="cache only N files per cohort, spread evenly across the "
                        "cohort rather than taken from the front. Subject ids, "
                        "labels and splits are still computed over the FULL "
                        "cohort, so a --limit run agrees with the full run row "
                        "for row.")
    p.add_argument("--brain-root", type=Path, default=DEFAULT_ROOTS["brain"])
    p.add_argument("--knee-root", type=Path, default=DEFAULT_ROOTS["knee"])
    p.add_argument("--out", type=Path, default=CACHE_DIR)
    p.add_argument("--slices-per-file", type=int, default=5,
                   help="central slices cached per file (default 5)")
    p.add_argument("--central-frac", type=float, default=0.5,
                   help="fraction of the volume, centred, that slices are drawn "
                        "from; edge slices are near-empty")
    p.add_argument("--target", default="auto",
                   choices=["auto", "n_coils_bucket", "acquisition", "matrix", "scanner"],
                   help="binary confound target for the 'label' column. "
                        "auto = n_coils_bucket for brain, acquisition for knee. "
                        "Raw values are always kept in their own columns.")
    p.add_argument("--coil-split", type=int, default=16,
                   help="n_coils >= this is the positive class for "
                        "--target n_coils_bucket (16 gives 259/195 on brain)")
    p.add_argument("--test-frac", type=float, default=0.30)
    p.add_argument("--split-salt", default="phasedx",
                   help="salt for the deterministic subject hash; changing it "
                        "reshuffles the split reproducibly")
    p.add_argument("--mask-frac-min", type=float, default=0.05,
                   help="body-mask fraction below which a slice is flagged "
                        "qc='mask_empty'")
    p.add_argument("--mask-frac-max", type=float, default=0.90,
                   help="body-mask fraction above which a slice is flagged "
                        "qc='mask_flooded' (no background left for the "
                        "background-only control)")
    p.add_argument("--rss-min", type=float, default=0.99,
                   help="abort the cohort if the mean per-file correlation with "
                        "reconstruction_rss falls below this")
    p.add_argument("--target-hw", type=int, nargs=2, default=list(TARGET_HW))
    p.add_argument("--self-test", action="store_true",
                   help="run the offline self-test and exit (no drive needed)")
    p.add_argument("--log-level", default="INFO")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=getattr(logging, args.log_level.upper()),
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    if args.self_test:
        return self_test()

    roots = {"brain": args.brain_root, "knee": args.knee_root}
    cohorts = list(COHORTS) if args.cohort == "both" else [args.cohort]
    if not DATA_ROOT.exists():
        raise SystemExit(f"data root not mounted: {DATA_ROOT}")

    # Discover both cohorts first so the cross-cohort disjointness assertion
    # runs before a single slice is reconstructed.
    discovered, rejects = {}, {}
    for cohort in cohorts:
        others = [roots[c] for c in COHORTS if c != cohort]
        paths, rej = discover_files(cohort, [roots[cohort]], other_roots=others)
        discovered[cohort], rejects[cohort] = paths, rej
        logger.info("[%s] %d file(s) after dedup; rejected %s",
                    cohort, len(paths), dict(rej) or "{}")
    assert_cohorts_disjoint(discovered)

    exit_code = 0
    for cohort in cohorts:
        paths = discovered[cohort]
        logger.info("[%s] scanning metadata for %d file(s)", cohort, len(paths))
        records, meta_failures = scan_metadata(cohort, paths)
        if not records:
            logger.error("[%s] no readable files, skipping cohort", cohort)
            exit_code = 1
            continue

        # Cohort-level decisions are made over the WHOLE cohort, before --limit
        # is applied. That costs one cheap metadata pass on a smoke test and
        # buys two things: a --limit run assigns exactly the subject ids,
        # labels and splits the full run would, so the two are comparable; and
        # the target is validated against the real class structure rather than
        # against whichever three files sort first (the first three knee files
        # are all CORPDFS_FBK, which would otherwise abort a binary target).
        assign_subject_ids(cohort, records)
        target_desc = apply_target(cohort, records, args.target, args.coil_split)
        assign_splits(records, args.test_frac, args.split_salt)
        logger.info("[%s] target -> %s", cohort, target_desc)
        balance = Counter(r["label"] for r in records)
        logger.info("[%s] %d file(s), %d subject(s), file-level label balance %s",
                    cohort, len(records),
                    len({r[SUBJECT_COL] for r in records}), dict(balance))
        if len(balance) < 2:
            logger.warning("[%s] the chosen target is SINGLE-CLASS over this "
                           "cohort (%s) and nothing can be trained on it.",
                           cohort, dict(balance))

        if args.limit and args.limit < len(records):
            # Spread the subsample over the cohort instead of taking the first
            # N: files are sorted by name and name correlates with protocol, so
            # the first N are a single acquisition from a single scanner.
            picks = sorted({int(round(v)) for v in
                            np.linspace(0, len(records) - 1, args.limit)})
            records = [records[i] for i in picks]
            logger.info("[%s] --limit %d -> caching %d file(s) spread across the "
                        "cohort: %s", cohort, args.limit, len(records),
                        [r["file"] for r in records])

        writer = CacheWriter(Path(args.out), cohort, tuple(args.target_hw))
        try:
            stats = process(cohort, records, writer, args)
        finally:
            writer.close()
        if not report(cohort, stats, writer, rejects.get(cohort, Counter()),
                      meta_failures, target_desc, args.rss_min):
            exit_code = 1
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
