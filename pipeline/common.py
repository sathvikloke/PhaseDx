"""
common.py
---------
Shared foundation for the PhaseDx pipeline: paths, MRI reconstruction
primitives, and normalization. Every stage builds on this module so that
magnitude and phase are computed exactly one way across the whole study.

Reconstruction correctness notes (these matter, and the previous code got
them wrong):

* Centered inverse FFT is ifftshift -> ifft2 -> fftshift. The old
  utils/data_utils.py used ifftshift on both sides, which is only equivalent
  for even-length axes and silently mis-centers odd ones.

* Phase must never be min-max normalized. Phase is a circular quantity on
  [-pi, pi]; rescaling it by per-slice extrema destroys the physical units and
  makes the value at a pixel depend on the most extreme pixel elsewhere in the
  slice. We keep phase in radians and feed sin/cos or the raw angle.

* Coil combination for phase cannot use root-sum-of-squares: RSS discards
  phase entirely. We use either the provided coil sensitivity maps (optimal,
  SENSE-style) or an adaptive virtual-coil combination. RSS is used for
  magnitude only.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import h5py
import numpy as np

logger = logging.getLogger(__name__)

# --- Paths ------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = Path("/Volumes/Research/fastmridatasets")
OUT_ROOT = PROJECT_ROOT / "pipeline_out"
CACHE_DIR = OUT_ROOT / "cache"
COHORT_DIR = OUT_ROOT / "cohorts"
RESULTS_DIR = OUT_ROOT / "results"
CKPT_DIR = OUT_ROOT / "checkpoints"

PROSTATE_LABEL_DIR = DATA_ROOT / "prostate" / "labels"
BREAST_LABEL_XLSX = DATA_ROOT / "breast_updated" / "breast" / "fastMRI_breast_labels.xlsx"

# Cohort identifiers used as cache keys throughout the pipeline.
#
# CLINICAL cohorts carry a real diagnostic label (PI-RADS, malignancy) and are
# where the phase-vs-magnitude question is asked.
#
# CONFOUND cohorts have NO tumour label and never will. Their label is an
# acquisition property -- receive-coil count for brain, pulse sequence for knee
# -- and they exist to measure how much acquisition identity the phase channel
# carries on its own. They must never be presented as diagnostic results.
CLINICAL_COHORTS = ("prostate_dwi", "prostate_t2", "breast")
CONFOUND_COHORTS = ("brain", "knee")
COHORTS = CLINICAL_COHORTS + CONFOUND_COHORTS
CONDITIONS = ("magnitude", "phase", "both")

# Network input resolution. 224 matches the ImageNet pretraining of ResNet-18.
TARGET_HW = (224, 224)


def iter_h5(root: Path):
    """
    Yield real .h5 files under root.

    macOS writes AppleDouble sidecars (._name.h5) onto exFAT drives. There are
    1425 of them on this drive and they are not HDF5 files; a naive rglob picks
    them up and every open fails. Filter them out here, once, for everyone.
    """
    for p in sorted(root.rglob("*.h5")):
        if p.name.startswith("._"):
            continue
        yield p


# --- FFT / reconstruction primitives ----------------------------------------

def ifft2c(kspace: np.ndarray, axes=(-2, -1)) -> np.ndarray:
    """
    Centered 2D inverse FFT. Operates on the last two axes by default and
    preserves all leading batch axes.

    The orthonormal norm keeps magnitudes comparable across matrix sizes, which
    matters because prostate DWI is 200x150 while T2 is 640x451.
    """
    return np.fft.fftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def fft2c(image: np.ndarray, axes=(-2, -1)) -> np.ndarray:
    """Centered 2D forward FFT (inverse of ifft2c)."""
    return np.fft.fftshift(
        np.fft.fft2(np.fft.ifftshift(image, axes=axes), axes=axes, norm="ortho"),
        axes=axes,
    )


def rss(coil_images: np.ndarray, coil_axis: int = 0) -> np.ndarray:
    """
    Root-sum-of-squares coil combination -> magnitude only.

    This is the standard clinical magnitude reconstruction and is what the
    'magnitude' condition must use, because it is exactly what a radiologist
    would see. It is phase-blind by construction.
    """
    return np.sqrt(np.sum(np.abs(coil_images) ** 2, axis=coil_axis))


def combine_with_sensitivities(
    coil_images: np.ndarray,
    sens_maps: np.ndarray,
    coil_axis: int = 0,
    eps: float = 1e-9,
) -> np.ndarray:
    """
    Optimal (matched-filter / SENSE) coil combination using known coil
    sensitivity maps:

        m = sum_c conj(S_c) * I_c / (sum_c |S_c|^2)

    This is the correct way to get a complex image whose phase is physically
    meaningful, because the sensitivity maps carry each coil's own phase offset
    and dividing them out removes the arbitrary per-coil phase. FastMRI
    prostate DWI ships coil_sens_maps, so we use this when available.

    Returns a complex image with the coil axis removed.
    """
    num = np.sum(np.conj(sens_maps) * coil_images, axis=coil_axis)
    den = np.sum(np.abs(sens_maps) ** 2, axis=coil_axis)
    return num / (den + eps)


def adaptive_virtual_coil(coil_images: np.ndarray, coil_axis: int = 0) -> np.ndarray:
    """
    Sensitivity-free complex coil combination via the dominant left singular
    vector of the coil x pixel matrix (Walsh-style adaptive combine).

    Used when coil sensitivity maps are not distributed with the data (prostate
    T2, breast). The leading singular vector is the maximum-SNR complex weight
    set; projecting onto it yields a single complex virtual coil whose phase is
    coherent across the slice.

    Note the global phase of a singular vector is arbitrary. We fix it by
    rotating so the image-domain center-of-mass phase is zero, which makes the
    output deterministic across runs and across slices. Without this the sign
    of the phase map flips arbitrarily and the network sees pure noise.
    """
    imgs = np.moveaxis(coil_images, coil_axis, 0)
    n_coils = imgs.shape[0]
    spatial_shape = imgs.shape[1:]
    flat = imgs.reshape(n_coils, -1)

    # Economy SVD on (coils x pixels); coils is small (16-32) so this is cheap.
    u, _, _ = np.linalg.svd(flat, full_matrices=False)
    weights = np.conj(u[:, 0])

    combined = (weights[:, None] * flat).sum(axis=0).reshape(spatial_shape)

    # Deterministic global phase: null out the intensity-weighted mean phase.
    mag = np.abs(combined)
    if mag.sum() > 0:
        ref = np.sum(combined * mag) / (mag.sum() + 1e-12)
        if np.abs(ref) > 0:
            combined = combined * np.conj(ref / np.abs(ref))
    return combined


# --- Normalization ----------------------------------------------------------

def normalize_magnitude(mag: np.ndarray, lo_pct: float = 1.0, hi_pct: float = 99.0) -> np.ndarray:
    """
    Robust percentile normalization of a magnitude image to roughly [0, 1].

    Percentiles rather than min/max because a single hot pixel (common near
    coils) otherwise compresses the whole anatomy into a narrow band.
    """
    lo, hi = np.percentile(mag, [lo_pct, hi_pct])
    return np.clip((mag - lo) / (hi - lo + 1e-8), 0.0, 1.0)


def phase_channels(phase: np.ndarray) -> np.ndarray:
    """
    Encode a phase map in radians as a 2-channel (sin, cos) representation.

    Phase wraps: -pi and +pi are the same angle but maximally distant as raw
    numbers. A convolution over raw wrapped phase therefore sees huge false
    edges at every wrap boundary. sin/cos is continuous across the wrap and is
    the standard fix. Returns shape (2, H, W) in [-1, 1].
    """
    return np.stack([np.sin(phase), np.cos(phase)], axis=0)


def center_crop_or_pad(arr: np.ndarray, target_hw=TARGET_HW) -> np.ndarray:
    """Center crop or zero-pad a 2D array to target_hw."""
    h, w = arr.shape[-2:]
    th, tw = target_hw

    if h >= th:
        top = (h - th) // 2
        arr = arr[..., top: top + th, :]
    else:
        pad = th - h
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(pad // 2, pad - pad // 2), (0, 0)])

    if w >= tw:
        left = (w - tw) // 2
        arr = arr[..., left: left + tw]
    else:
        pad = tw - w
        arr = np.pad(arr, [(0, 0)] * (arr.ndim - 2) + [(0, 0), (pad // 2, pad - pad // 2)])

    return arr


def resample_to(arr: np.ndarray, target_hw=TARGET_HW, order: int = 1) -> np.ndarray:
    """
    Resample a 2D array to target_hw by interpolation (not cropping).

    Use this when the field of view must be preserved -- prostate DWI is
    200x150 and cropping to 224 would pad more than it crops, wasting
    resolution. For phase, resample sin/cos separately rather than the angle,
    since interpolating across a wrap gives meaningless intermediate values.
    """
    from scipy.ndimage import zoom

    h, w = arr.shape[-2:]
    th, tw = target_hw
    if (h, w) == (th, tw):
        return arr
    return zoom(arr, (th / h, tw / w), order=order)


def resample_phase(phase: np.ndarray, target_hw=TARGET_HW) -> np.ndarray:
    """Resample a wrapped phase map safely by going through sin/cos."""
    s = resample_to(np.sin(phase), target_hw, order=1)
    c = resample_to(np.cos(phase), target_hw, order=1)
    return np.arctan2(s, c)


def zscore(arr: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
    """Per-slice z-score. If a mask is given, statistics come from inside it."""
    ref = arr[mask] if mask is not None else arr
    mu, sd = float(np.mean(ref)), float(np.std(ref))
    return (arr - mu) / (sd + 1e-8)


def body_mask(mag: np.ndarray, thresh_pct: float = 60.0) -> np.ndarray:
    """
    Crude foreground mask from the magnitude image via Otsu-like percentile
    thresholding plus morphological cleanup.

    Needed for the background-only control: to prove the classifier is reading
    anatomy rather than the scanner's coil/shim signature, we train it on the
    complement of this mask and check that performance collapses.
    """
    from scipy.ndimage import binary_closing, binary_fill_holes

    thresh = np.percentile(mag, thresh_pct)
    m = mag > thresh
    m = binary_closing(m, structure=np.ones((5, 5)))
    m = binary_fill_holes(m)
    return m.astype(bool)


def safe_open(path) -> Optional[h5py.File]:
    """Open an h5 file, returning None (with a warning) instead of raising."""
    try:
        return h5py.File(path, "r")
    except Exception as exc:  # noqa: BLE001
        logger.warning("Could not open %s: %s", path, exc)
        return None


# --- The split-enforcement unit: subject_id ---------------------------------
#
# `patient_id` is NOT the unit a train/test split may be enforced on.
#
# The fastMRI breast release codes some women twice: a repeat scan of the same
# physical person is issued a *different* coded patient name, and the label
# sheet links the two only through its 'Repeated scan' group column. Stage 1
# resolves that and emits `subject_id`, which collapses (a) the two
# acquisitions of one exam and (b) the repeat-scan groups spanning two coded
# names. Splitting on patient_id therefore lets the same woman appear in both
# training and test under two names -- a leak that inflates the headline AUC
# and is invisible to a patient_id-keyed check.
#
# Stage 2 originally wrote no subject_id into <cohort>_index.csv at all, so
# every consumer of the cache index (which is the path that produces the
# published number) silently fell back to patient_id. These helpers are the one
# place that derives subject_id, and the derivation is always the stage-1 cohort
# CSV -- the table that was actually subject-checked by
# s01_labels.assert_no_subject_spans_splits. Nothing here invents an id.
#
# The two tables agree on file BASENAME and nothing else: the cohort CSV stores
# an absolute /Volumes path, and their patient_id columns live in different
# namespaces (cohort says 'prostate_0001', the prostate cache index says 1), so
# joining on patient_id would silently produce garbage.

SUBJECT_COL = "subject_id"


def basenames(paths) -> "list[str]":
    """File basenames for a sequence of path-ish values (absolute or bare)."""
    return [Path(str(p)).name for p in paths]


def cohort_csv_path(cohort: str, cohort_dir=None) -> Path:
    """Location of the stage-1 cohort table for `cohort`."""
    return Path(cohort_dir if cohort_dir is not None else COHORT_DIR) / f"{cohort}_cohort.csv"


def subject_id_lut(cohort: str, cohort_dir=None) -> dict:
    """
    Build {file basename -> subject_id} from the stage-1 cohort CSV.

    Raises rather than returning a partial map: a half-populated subject column
    is worse than none, because it looks authoritative to every downstream
    groupby while quietly degrading to patient_id for the rows it missed.
    """
    import pandas as pd

    path = cohort_csv_path(cohort, cohort_dir)
    if not path.exists():
        raise FileNotFoundError(
            f"stage-1 cohort CSV {path} not found; it is the only sanctioned "
            f"source of subject_id. Run s01_labels.py for cohort {cohort!r} first."
        )
    coh = pd.read_csv(path, low_memory=False)
    for col in ("file", SUBJECT_COL):
        if col not in coh.columns:
            raise ValueError(f"{path} has no {col!r} column; cannot derive subject_id")
    if coh[SUBJECT_COL].isna().any():
        raise ValueError(f"{path} has {int(coh[SUBJECT_COL].isna().sum())} null subject_id value(s)")

    coh = coh.assign(_bn=basenames(coh["file"]))
    # One basename must resolve to exactly one subject. If stage 1 ever emitted
    # two subjects for one file the join below would pick an arbitrary winner,
    # so make that impossible instead of probable.
    spread = coh.groupby("_bn")[SUBJECT_COL].nunique()
    bad = spread[spread > 1]
    if len(bad):
        raise ValueError(
            f"{path}: {len(bad)} file(s) map to more than one subject_id, "
            f"e.g. {list(bad.index)[:5]}; refusing to build an ambiguous LUT"
        )
    lut = coh.drop_duplicates("_bn").set_index("_bn")[SUBJECT_COL].astype(str)
    return dict(lut)


def attach_subject_ids(df, cohort: str, cohort_dir=None, file_col: str = "file"):
    """
    Return `df` with a `subject_id` column joined from the stage-1 cohort CSV.

    Placed immediately after patient_id so the CSV reads sensibly. Raises if any
    row fails to match -- see subject_id_lut on why partial maps are refused.
    """
    import pandas as pd

    if file_col not in df.columns:
        raise ValueError(f"cannot attach subject_id: no {file_col!r} column")
    lut = subject_id_lut(cohort, cohort_dir)
    bn = pd.Series(basenames(df[file_col]), index=df.index)
    sid = bn.map(lut)
    missing = sorted(set(bn[sid.isna()]))
    if missing:
        raise ValueError(
            f"[{cohort}] {len(missing)} cached file(s) have no row in the stage-1 "
            f"cohort CSV and therefore no subject_id, e.g. {missing[:5]}. "
            "Re-run stage 1 over the same file set before caching."
        )

    out = df.copy()
    out[SUBJECT_COL] = sid.astype(str).to_numpy()
    if "patient_id" in out.columns:
        cols = [c for c in out.columns if c != SUBJECT_COL]
        at = cols.index("patient_id") + 1
        out = out[cols[:at] + [SUBJECT_COL] + cols[at:]]
    return out


def backfill_cache_subject_ids(cohort: str, cache_dir=None, cohort_dir=None,
                               dry_run: bool = False) -> dict:
    """
    Add `subject_id` to an existing <cohort>_index.csv, in place and idempotently.

    The stage-2 caches cost hours of reconstruction and are validated against
    the vendor images (prostate_t2 r=0.998, prostate_dwi r=0.984, breast r=0.97),
    so the fix for a missing *index column* must not be "re-preprocess". Only the
    CSV is touched; the HDF5 arrays are never opened, and row order and `idx` are
    preserved exactly, so the CSV stays row-aligned with the cache datasets.

    Idempotent: if subject_id is already present and agrees with the stage-1 LUT
    the file is left byte-identical. If it is present but DISAGREES the function
    raises rather than silently rewriting -- a changed subject mapping means the
    cohort definition moved under the cache and a human has to look.

    Returns a small report dict. Writes are atomic (temp file + os.replace) so an
    interrupted backfill cannot leave a truncated index.
    """
    import os
    import tempfile

    import pandas as pd

    cache_dir = Path(cache_dir if cache_dir is not None else CACHE_DIR)
    path = cache_dir / f"{cohort}_index.csv"
    if not path.exists():
        raise FileNotFoundError(f"no cache index at {path}")

    df = pd.read_csv(path, low_memory=False)
    n_before = len(df)
    filled = attach_subject_ids(df, cohort, cohort_dir)

    already = SUBJECT_COL in df.columns and not df[SUBJECT_COL].isna().any()
    if already:
        lhs = df[SUBJECT_COL].astype(str).to_numpy()
        rhs = filled[SUBJECT_COL].astype(str).to_numpy()
        if (lhs != rhs).any():
            n_diff = int((lhs != rhs).sum())
            raise ValueError(
                f"[{cohort}] {path} already has subject_id but {n_diff} row(s) "
                "disagree with the stage-1 cohort CSV. Refusing to overwrite: "
                "resolve which cohort definition is correct first."
            )
        status = "already-present"
    else:
        status = "backfilled"
        if not dry_run:
            fd, tmp = tempfile.mkstemp(dir=str(cache_dir), suffix=".csv.tmp")
            os.close(fd)
            filled.to_csv(tmp, index=False)
            os.replace(tmp, path)

    assert len(filled) == n_before, "backfill changed the row count"
    span = (filled[filled["official_split"].astype(str) != ""]
            .groupby(SUBJECT_COL)["official_split"].nunique())
    straddling = sorted(span[span > 1].index)
    report = {
        "cohort": cohort, "path": str(path), "status": status, "rows": n_before,
        "n_patients": int(df["patient_id"].nunique()) if "patient_id" in df else -1,
        "n_subjects": int(filled[SUBJECT_COL].nunique()),
        "subjects_spanning_splits": straddling,
        "dry_run": bool(dry_run),
    }
    if straddling:
        logger.error(
            "[%s] %d subject(s) span more than one split in %s: %s -- stage 3 will "
            "refuse to train on this index, which is the point.",
            cohort, len(straddling), path.name, straddling[:10],
        )
    return report
