"""
s10_reconfidelity.py
====================
Stage 10 of the PhaseDx pipeline: turn "our reconstruction is faithful" from a
CODE COMMENT into a persisted, re-checkable dataset.

WHY THIS MODULE EXISTS
----------------------
The study's whole premise -- that a network reading raw-k-space PHASE is
reading acquisition identity rather than pathology -- only means anything if
the images it reads are actually images of the anatomy. Up to now the evidence
for that was:

  * ``pipeline/s02_prostate.py`` docstring: "Verified vs reconstruction_rss:
    r = 0.998 - 0.9996 over 6 files."
  * ``pipeline/s02_breast.py`` argparse help: "r=0.97 vs the vendor temptv
    reference".
  * ``pipeline/s02_brainknee.py`` docstring: "r = 1.000000".

Those numbers WERE measured -- they were computed while the cache was being
built and printed to a run log -- and the per-slice value is even carried in
the stage-2 index CSVs (``recon_ncc`` for prostate/brain/knee, ``temptv_r`` for
breast). But a reviewer cannot check a prose claim, the run logs are not
release artifacts, and a QC column buried in a 50-column index is not evidence
anybody will find. A reviewer who says "reconstruction fidelity is unevidenced"
is making a fair procedural point even when the underlying number is right.

So this module recomputes the correlation FROM SCRATCH, for EVERY cached slice
of EVERY cohort, against the vendor reference that ships inside the same HDF5
file, and writes it out as data:

    pipeline_out/recon_fidelity/<cohort>.csv           per cached slice
    pipeline_out/recon_fidelity/<cohort>_perfile.csv   per file
    pipeline_out/recon_fidelity/<cohort>.json          per-cohort summary
    pipeline_out/recon_fidelity/recon_fidelity_summary.json
    pipeline_out/recon_fidelity/fig_recon_fidelity.{png,pdf}

It imports the reconstruction functions from the stage-2 readers rather than
reimplementing them. That is deliberate: a fidelity check written against a
second, independent implementation of the reconstruction would measure the
agreement of two pieces of code, not the fidelity of the cache that stage 3
actually trained on. What is independent here is the MEASUREMENT and its
PERSISTENCE, not the reconstruction. The per-slice value recorded at caching
time is carried alongside as ``r_cached`` and differenced, so a drift between
the cache on disk and the code that claims to have produced it shows up as a
non-zero ``r_delta`` rather than being silently absorbed.

WHAT IS COMPARED, PER COHORT
----------------------------
  prostate_t2  our RSS magnitude, native 640x640, centre-cropped to 320x320
               vs ``reconstruction_rss``           -- fully sampled after the
               interleaved averages are merged, so this is a strong check.
  brain, knee  our RSS magnitude centre-cropped to the vendor recon shape
               vs ``reconstruction_rss``           -- fully sampled Cartesian;
               ifft2c + RSS *is* the fastMRI reference recon, so r must be 1.
  prostate_dwi our single low-b volume, centre-cropped to 100x100,
               vs ``trace_b50``                    -- WEAKENED BY DESIGN (see
               below); the per-file low-b-averaged number is the real check.
  breast       our gridded radial magnitude, 320x320,
               vs ``temptv`` (mean over the 4 dynamic frames)
                                                   -- WEAKEST comparison in
               the study (see below).

TWO PLACES WHERE r IS NOT A FIDELITY NUMBER, AND WE SAY SO
----------------------------------------------------------
1. prostate_dwi. ``trace_b50`` is the geometric mean of three diffusion
   directions, each already averaged -- roughly 14 acquisitions. The cache
   deliberately stores ONE low-b volume, because complex-averaging across
   diffusion directions or averages is invalid (each carries its own
   eddy-current / bulk-motion phase, which is exactly the channel under study).
   Correlating one noisy average against a ~14-average reference CANNOT reach
   1.0 however correct the reconstruction is. So this module also computes, per
   file, the comparison in which the detected low-b volumes are
   MAGNITUDE-averaged the way the vendor's trace_b50 is. That second number --
   ``r_lowb_avg``, reported as its own distribution -- is the one that isolates
   GRAPPA + geometry + coil combination from single-average SNR. A low per-slice
   value with a high per-file low-b-averaged value means "one average is noisy",
   not "the recon is wrong". Both are persisted; neither is allowed to stand in
   for the other.

2. breast. ``temptv`` IS NOT GROUND TRUTH. It is the vendor's own
   temporal-TV-REGULARISED reconstruction of the same radial k-space: a
   different estimator, with its own smoothing prior, its own streak
   suppression and its own temporal coupling across the 4 dynamic frames. Our
   reconstruction is an unregularised density-compensated adjoint NUFFT. So
   r(ours, temptv) is agreement between two reconstructions, and the residual
   is not decomposable into "our error" and "their error". It bounds fidelity
   from neither side cleanly:
     * it can be pessimistic -- the TV prior removes noise we keep, so even a
       perfect adjoint recon would not reach 1.0;
     * it can be optimistic -- both estimators are driven by the same k-space,
       so a trajectory error (wrong angle rule, wrong readout centre, wrong kz
       sign) that corrupts both would partially cancel.
   There is no unregularised vendor image in the breast release to compare
   against, so this is the best available check and it is weaker than the
   prostate/brain/knee ones. That sentence belongs in the manuscript, not in a
   footnote. Independently, the breast PHASE channel cannot be validated at all
   -- temptv is magnitude-only -- and s02_breast records phase split-half
   cosine similarity (0.50 at 72 spokes, 0.78 at 144) as the honest bound.

THE ANATOMY-SUPPORT NULL
------------------------
A Pearson r between two MRI magnitude images of the same slice is inflated by
something that has nothing to do with reconstruction correctness: both images
are mostly "bright body inside a dark air background", so any two images of the
same body part correlate. To keep r honest this module also correlates our
slice z against the VENDOR reference at slice z +/- ``--null-shift`` (default
5) of the SAME volume -- a different slice of the same patient, same contrast,
same FOV, same intensity scale. That null is the floor r would sit at from
shared support and gross anatomy alone. The reported margin

    r_margin = r - r_null_shift

is how much of the correlation is actually slice-specific agreement. It is
reported per cohort and drawn in panel (b) of the figure. It is a context
statistic, not the headline metric: the headline metric stays exactly the
correlation the stage-2 readers computed, so this file can be diffed against
their index CSVs.

WHAT IS *NOT* CLAIMED HERE
--------------------------
Nothing about phase. Every vendor reference in every one of these releases is a
magnitude image, so every number in this module validates the MAGNITUDE
reconstruction: the FFT centring, the GRAPPA kernel, the average merging, the
radial trajectory, the density compensation, the geometry, the crops and the
coil-combination-for-magnitude. Phase inherits credibility from that (it is
derived from the same complex image) but is never directly validated, and no
result in this study should be read as if it were. This module makes the
magnitude claim checkable; it does not upgrade the phase claim.

Nothing about the study's verdicts either. Reconstruction fidelity is a
precondition for the analysis being about anatomy at all. It is not evidence
for or against the phase-vs-magnitude question, and a good number here does not
turn any NOT SUPPORTED verdict into a supported one.

USAGE
-----
    python pipeline/s10_reconfidelity.py --self-test          # no drive needed
    python pipeline/s10_reconfidelity.py --cohort prostate_t2 --limit 3
    python pipeline/s10_reconfidelity.py --cohort breast
    python pipeline/s10_reconfidelity.py                      # all five cohorts
    python pipeline/s10_reconfidelity.py --figure-only        # redraw from CSVs

The five cohorts are independent: each writes its own CSV/JSON, so they can be
run as separate processes and the figure assembled afterwards with
``--figure-only``. Results are checkpointed after every file and a re-run
resumes by default (``--no-resume`` starts over); a full pass is ~4 hours of
reading the raw drive and the first attempt at it was killed at the one-hour
mark, which is why that is not optional.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import CACHE_DIR, DATA_ROOT, OUT_ROOT  # noqa: E402

logger = logging.getLogger("s10_reconfidelity")

OUT_DIR_DEFAULT = OUT_ROOT / "recon_fidelity"

COHORT_ORDER = ("prostate_t2", "prostate_dwi", "breast", "brain", "knee")

# Thresholds the summary counts slices below. 0.99 is the "this is a fully
# sampled Cartesian recon and should be essentially exact" bar, 0.95 the
# "clearly the same image" bar, 0.90 the bar s02_breast and s02_prostate
# already gate on.
THRESHOLDS = (0.99, 0.95, 0.90)


# ==========================================================================
# what each cohort's stage-2 reader documented, so a drift is visible
# ==========================================================================
#
# `claim` is the number the stage-2 docstring/help text asserts; `statistic`
# names which distribution that claim is about. `tol` is how far below the
# claim the observed value may sit before this module calls it a discrepancy --
# generous enough that the difference between "6 files" and "every file" does
# not fire on its own, tight enough that a real regression does.
CLAIMS = {
    "prostate_t2": dict(
        reference="reconstruction_rss",
        statistic="per_slice",
        claim=0.998,
        tol=0.010,
        source="s02_prostate.py docstring GEOMETRY: 'Verified vs "
               "reconstruction_rss: r = 0.998 - 0.9996 over 6 files.'",
        strength="strong",
        note="Fully sampled once the three interleaved averages are merged; "
             "the vendor image is an unregularised RSS reconstruction of the "
             "same k-space, so this is a like-for-like comparison.",
    ),
    "prostate_dwi": dict(
        reference="trace_b50",
        statistic="per_file_lowb_averaged",
        claim=0.97,
        tol=0.010,
        source="s02_prostate.py docstring HOW TO READ THE TWO AXDIFF "
               "VALIDATION NUMBERS: single volume mean r = 0.89, low-b "
               "magnitude-averaged mean r = 0.97.",
        strength="indirect",
        note="The vendor trace_b50 averages ~14 acquisitions and we cache one, "
             "so the per-slice number carries a deliberate SNR penalty. The "
             "per-file low-b-averaged number is the reconstruction check.",
        secondary=dict(statistic="per_slice", claim=0.89, tol=0.020),
    ),
    "breast": dict(
        reference="temptv",
        statistic="per_slice",
        claim=0.97,
        tol=0.010,
        source="s02_breast.py docstring section 3 (0.973-0.976 at 288 spokes) "
               "and the --frame argparse help ('r=0.97 vs the vendor temptv "
               "reference').",
        strength="WEAK",
        note="temptv is the vendor's temporal-TV-REGULARISED reconstruction of "
             "the same radial k-space, not ground truth. This is agreement "
             "between two estimators with different priors, and is the weakest "
             "fidelity comparison in the study. See module docstring.",
    ),
    "brain": dict(
        reference="reconstruction_rss",
        statistic="per_slice",
        claim=1.0,
        tol=0.001,
        source="s02_brainknee.py docstring section 4: 'r = 1.000000'.",
        strength="strong",
        note="Fully sampled Cartesian; ifft2c + RSS IS the fastMRI reference "
             "reconstruction, so anything below 1 is a bug, not noise.",
    ),
    "knee": dict(
        reference="reconstruction_rss",
        statistic="per_slice",
        claim=1.0,
        tol=0.001,
        source="s02_brainknee.py docstring section 4: 'r = 1.000000'.",
        strength="strong",
        note="Fully sampled Cartesian; ifft2c + RSS IS the fastMRI reference "
             "reconstruction, so anything below 1 is a bug, not noise.",
    ),
}

# Column in the stage-2 index CSV that already holds the caching-time value.
CACHED_COL = {
    "prostate_t2": "recon_ncc",
    "prostate_dwi": "recon_ncc",
    "breast": "temptv_r",
    "brain": "recon_ncc",
    "knee": "recon_ncc",
}

PER_SLICE_COLUMNS = [
    "cohort", "file", "folder", "slice", "cache_idx", "reference",
    "r", "r_cached", "r_delta", "r_null_shift", "r_margin", "extra",
]


# ==========================================================================
# metric
# ==========================================================================

def pearson(a: np.ndarray, b: np.ndarray) -> float:
    """
    Pearson correlation of two same-shape real images, in float64.

    This is deliberately the same estimator the stage-2 readers use
    (``s02_prostate.ncc`` / ``s02_brainknee.ncc`` are the mean-centred
    normalised dot product; ``s02_breast`` calls ``np.corrcoef``), so the
    numbers in this file are directly comparable with the ``recon_ncc`` /
    ``temptv_r`` columns of the caches. The self-test asserts that agreement
    rather than assuming it.

    Returns NaN -- not 0 -- when either input is constant, because "could not
    be measured" and "measured as uncorrelated" are different facts and only
    one of them should be averaged into a summary.
    """
    a = np.asarray(a, np.float64).ravel()
    b = np.asarray(b, np.float64).ravel()
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if not np.isfinite(denom) or denom <= 0.0:
        return float("nan")
    return float(a @ b / denom)


def null_slice(s: int, n: int, shift: int) -> int:
    """
    Pick the slice of the vendor volume to use as the anatomy-support null.

    Prefer s + shift, fall back to s - shift at the far edge of the volume, and
    return -1 when the volume is too thin for either (which is recorded as NaN
    rather than silently correlating a slice against itself, which would return
    1.0 and make the null look like perfect agreement).
    """
    if n <= 1:
        return -1
    if s + shift < n:
        return s + shift
    if s - shift >= 0:
        return s - shift
    return n - 1 if s != n - 1 else 0


# ==========================================================================
# locating the raw files behind the cache
# ==========================================================================

def raw_path(cohort: str, row, data_root: Path) -> Path:
    """
    Map one stage-2 index row back to the HDF5 it was reconstructed from.

    brain/knee index rows carry an absolute ``source_dir``; prostate and breast
    carry a ``folder`` relative to their release root. Both are honoured rather
    than globbing the drive, so a file that moved is a hard error here instead
    of quietly matching a different exam with the same basename.
    """
    name = str(row["file"])
    if cohort in ("brain", "knee"):
        return Path(str(row["source_dir"])) / name
    if cohort == "breast":
        return data_root / "breast_updated" / "breast" / str(row["folder"]) / name
    return data_root / "prostate" / str(row["folder"]) / name


def load_index(cohort: str, cache_dir: Path) -> pd.DataFrame:
    path = cache_dir / f"{cohort}_index.csv"
    if not path.exists():
        raise SystemExit(
            f"[{cohort}] no stage-2 index at {path}. This module measures the "
            "fidelity of a cache that already exists; it does not build one."
        )
    df = pd.read_csv(path)
    df["slice"] = df["slice"].astype(int)
    return df


# ==========================================================================
# per-file checkpointing
# ==========================================================================

class Sink:
    """
    Append-after-every-file writer, so a killed run costs at most one exam.

    A full pass over all five caches is a couple of hours of reading an
    external drive; the stage-2 cache writers checkpoint per file for exactly
    this reason and so does this one.

    The two CSVs are written in the order slices-then-file, so a process killed
    between the two writes leaves slice rows whose file has no per-file record.
    On resume those orphans are dropped and the file is measured again --
    resuming must never be able to double-count a slice, because every summary
    in this module is an average over slices.
    """

    def __init__(self, out_dir: Path, cohort: str, resume: bool = True):
        out_dir.mkdir(parents=True, exist_ok=True)
        self.cohort = cohort
        self.slice_path = out_dir / f"{cohort}.csv"
        self.file_path = out_dir / f"{cohort}_perfile.csv"
        self.fail_path = out_dir / f"{cohort}_failures.json"
        self.done: set[str] = set()
        self.failures: list[dict] = []
        self._file_cols: list[str] | None = None

        if not resume:
            for p in (self.slice_path, self.file_path, self.fail_path):
                p.unlink(missing_ok=True)
            return

        if self.file_path.exists() and self.slice_path.exists():
            fdf = pd.read_csv(self.file_path)
            fdf = fdf[fdf["file"].notna()]
            self._file_cols = list(fdf.columns)
            self.done = set(fdf["file"].astype(str))
            sdf = pd.read_csv(self.slice_path)
            keep = sdf[sdf["file"].astype(str).isin(self.done)]
            if len(keep) != len(sdf):
                logger.warning("[%s] resume: dropping %d orphan slice row(s) "
                               "from a file that was interrupted mid-write",
                               cohort, len(sdf) - len(keep))
                keep.to_csv(self.slice_path, index=False, float_format="%.12g")
            fdf.to_csv(self.file_path, index=False, float_format="%.12g")
            logger.info("[%s] resume: %d file(s) already measured, %d slice rows",
                        cohort, len(self.done), len(keep))
        if self.fail_path.exists():
            self.failures = json.loads(self.fail_path.read_text())

    def add(self, slice_rows: list, file_row: dict) -> None:
        sdf = pd.DataFrame(slice_rows)
        sdf = sdf[[c for c in PER_SLICE_COLUMNS if c in sdf.columns]]
        sdf.to_csv(self.slice_path, mode="a", index=False,
                   header=not self.slice_path.exists(), float_format="%.12g")
        fdf = pd.DataFrame([file_row])
        if self._file_cols is None:
            self._file_cols = list(fdf.columns)
        fdf = fdf.reindex(columns=self._file_cols)
        fdf.to_csv(self.file_path, mode="a", index=False,
                   header=not self.file_path.exists(), float_format="%.12g")
        self.done.add(str(file_row["file"]))

    def fail(self, name: str, exc: BaseException) -> None:
        self.failures.append(dict(cohort=self.cohort, file=name,
                                  error=f"{type(exc).__name__}: {exc}"))
        self.fail_path.write_text(json.dumps(self.failures, indent=2))
        self.done.add(name)

    def load(self) -> tuple[pd.DataFrame, pd.DataFrame]:
        return pd.read_csv(self.slice_path), pd.read_csv(self.file_path)


# ==========================================================================
# per-cohort measurement
# ==========================================================================

def _finish_file(cohort: str, name: str, rows: list, extra: dict | None = None) -> dict:
    """Collapse the per-slice rows of one file into its per-file record."""
    vals = np.array([r["r"] for r in rows], float)
    vals = vals[np.isfinite(vals)]
    nulls = np.array([r["r_null_shift"] for r in rows], float)
    nulls = nulls[np.isfinite(nulls)]
    rec = dict(
        cohort=cohort, file=name, n_slices=len(rows), n_finite=int(vals.size),
        r_mean=float(vals.mean()) if vals.size else float("nan"),
        r_median=float(np.median(vals)) if vals.size else float("nan"),
        r_min=float(vals.min()) if vals.size else float("nan"),
        r_max=float(vals.max()) if vals.size else float("nan"),
        r_null_mean=float(nulls.mean()) if nulls.size else float("nan"),
    )
    if extra:
        rec.update(extra)
    return rec


def measure_prostate_t2(index: pd.DataFrame, args, sink: Sink) -> None:
    from pipeline import s02_prostate as P

    groups = list(index.groupby("file", sort=True))
    if args.limit:
        groups = groups[: args.limit]

    for i, (name, grp) in enumerate(groups, 1):
        if name in sink.done:
            continue
        path = raw_path("prostate_t2", grp.iloc[0], args.data_root)
        t0 = time.time()
        try:
            f = P.open_h5(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("[prostate_t2] UNREADABLE %s (%s)", path.name, exc)
            sink.fail(name, exc)
            continue

        rows = []
        with f:
            n_slices_vol = int(f["reconstruction_rss"].shape[0])
            # The average-merge decision is a per-file property recorded in the
            # cache; re-read it rather than re-deciding, so this measures the
            # image that was actually cached.
            used = str(grp.iloc[0].get("averages_used", "(a0+a2)/2+a1"))
            coherent = used.startswith("(a0+a2)")
            for _, row in grp.iterrows():
                s = int(row["slice"])
                mag_native, _, _ = P.t2_slice_recon(f, s, coherent,
                                                    tuple(args.target_hw))
                ours = P.center_crop2d(mag_native, P.T2_VENDOR_HW)
                r = pearson(ours, f["reconstruction_rss"][s])
                ns = null_slice(s, n_slices_vol, args.null_shift)
                rn = (pearson(ours, f["reconstruction_rss"][ns])
                      if ns >= 0 else float("nan"))
                rows.append(_slice_row("prostate_t2", name, row, s, r, rn,
                                       "reconstruction_rss",
                                       extra=f"averages_used={used}"))
        frec = _finish_file("prostate_t2", name, rows, dict(averages_used=used))
        sink.add(rows, frec)
        logger.info("[prostate_t2] %3d/%d %s: %d slices  r mean=%.6f min=%.6f "
                    "(%.1fs)", i, len(groups), name, len(rows),
                    frec["r_mean"], frec["r_min"], time.time() - t0)


def measure_prostate_dwi(index: pd.DataFrame, args, sink: Sink) -> None:
    from pipeline import s02_prostate as P

    groups = list(index.groupby("file", sort=True))
    if args.limit:
        groups = groups[: args.limit]

    for i, (name, grp) in enumerate(groups, 1):
        if name in sink.done:
            continue
        path = raw_path("prostate_dwi", grp.iloc[0], args.data_root)
        t0 = time.time()
        try:
            f = P.open_h5(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("[prostate_dwi] UNREADABLE %s (%s)", path.name, exc)
            sink.fail(name, exc)
            continue

        rows = []
        with f:
            n_slices_vol = int(f["trace_b50"].shape[0])
            mid = int(f["kspace"].shape[1]) // 2

            # --- the number that actually tests the reconstruction ---------
            # Re-detect the low-b volume set and correlate their MAGNITUDE
            # average -- the vendor's own way of forming trace_b50 -- against
            # trace_b50. This removes the single-average SNR penalty and is
            # what isolates GRAPPA + geometry + coil combination.
            w_mid, resid = P.grappa_weights(f["calibration_data"][mid])
            lowb_rs, lowb_slices, low_b = [], [], []
            for k in range(args.dwi_lowb_slices):
                sl = mid + (k - args.dwi_lowb_slices // 2) * 3
                if not 0 <= sl < n_slices_vol:
                    continue
                w = w_mid if sl == mid else P.grappa_weights(
                    f["calibration_data"][sl])[0]
                lb, _, _, r_avg = P.dwi_detect_lowb(f, w, sl)
                if lb:
                    lowb_rs.append(r_avg)
                    lowb_slices.append(sl)
                    low_b = lb
            r_lowb_avg = float(np.mean(lowb_rs)) if lowb_rs else float("nan")

            cached_vol = grp["dwi_volume"].astype(int)
            vol_in_lowb = bool(low_b) and bool(set(cached_vol) <= set(low_b))

            for _, row in grp.iterrows():
                s = int(row["slice"])
                vol = int(row["dwi_volume"])
                w = w_mid if s == mid else P.grappa_weights(
                    f["calibration_data"][s])[0]
                mag_full, _ = P.dwi_slice_recon(f, s, vol, w, mag_only=True)
                ours = P.center_crop2d(mag_full, P.DWI_VENDOR_HW)
                r = pearson(ours, f["trace_b50"][s])
                ns = null_slice(s, n_slices_vol, args.null_shift)
                rn = (pearson(ours, f["trace_b50"][ns])
                      if ns >= 0 else float("nan"))
                rows.append(_slice_row("prostate_dwi", name, row, s, r, rn,
                                       "trace_b50",
                                       extra=f"dwi_volume={vol}"))

        frec = _finish_file(
            "prostate_dwi", name, rows,
            dict(r_lowb_avg=r_lowb_avg,
                 lowb_slices=";".join(str(v) for v in lowb_slices),
                 n_lowb_volumes=len(low_b),
                 cached_volume_is_lowb=vol_in_lowb,
                 grappa_acs_resid=float(resid)))
        sink.add(rows, frec)
        logger.info("[prostate_dwi] %3d/%d %s: %d slices  per-slice r mean=%.4f "
                    "min=%.4f | low-b-averaged r=%.4f (n_lowb=%d) (%.1fs)",
                    i, len(groups), name, len(rows), frec["r_mean"],
                    frec["r_min"], r_lowb_avg, len(low_b), time.time() - t0)


def measure_brainknee(cohort: str, index: pd.DataFrame, args, sink: Sink) -> None:
    from pipeline import s02_brainknee as BK

    groups = list(index.groupby("file", sort=True))
    if args.limit:
        groups = groups[: args.limit]

    for i, (name, grp) in enumerate(groups, 1):
        if name in sink.done:
            continue
        path = raw_path(cohort, grp.iloc[0], args.data_root)
        t0 = time.time()
        try:
            f = BK.open_h5(path)
        except Exception as exc:  # noqa: BLE001
            logger.error("[%s] UNREADABLE %s (%s)", cohort, path.name, exc)
            sink.fail(name, exc)
            continue

        rows = []
        with f:
            ref = f["reconstruction_rss"]
            n_slices_vol = int(ref.shape[0])
            for _, row in grp.iterrows():
                s = int(row["slice"])
                mag_vendor_fov, _, _ = BK.slice_recon(f, s, tuple(args.target_hw))
                r = pearson(mag_vendor_fov, ref[s])
                ns = null_slice(s, n_slices_vol, args.null_shift)
                rn = (pearson(mag_vendor_fov, ref[ns])
                      if ns >= 0 else float("nan"))
                rows.append(_slice_row(cohort, name, row, s, r, rn,
                                       "reconstruction_rss",
                                       extra=f"acq={row.get('acq', '')}"))
        frec = _finish_file(cohort, name, rows,
                            dict(acq=str(grp.iloc[0].get("acq", ""))))
        sink.add(rows, frec)
        if i % 25 == 0 or i == len(groups):
            logger.info("[%s] %3d/%d %s: r mean=%.8f min=%.8f (%.1fs)",
                        cohort, i, len(groups), name, frec["r_mean"],
                        frec["r_min"], time.time() - t0)


def measure_breast(index: pd.DataFrame, args, sink: Sink) -> None:
    from pipeline import s02_breast as B

    frames = set(str(v) for v in index["frame"].unique())
    if frames != {"all"}:
        raise SystemExit(
            f"[breast] the cache mixes dynamic frames {sorted(frames)}; this "
            "module assumes the '--frame all' cache the study uses and would "
            "otherwise correlate against the wrong temptv reference"
        )
    dcf = str(index["dcf"].iloc[0])
    spoke_ids = np.arange(B.N_SPOKES_TOTAL)
    theta = B.spoke_angles(spoke_ids)
    dcf_w = B.density_compensation(theta, dcf)
    t0 = time.time()
    gridder = B.Gridder(theta, oversamp=args.grid_oversamp)
    logger.info("[breast] gridder: %d spokes, %dx%d grid, %d nnz (%.1fs)",
                spoke_ids.size, gridder.grid, gridder.grid, gridder.M.nnz,
                time.time() - t0)

    groups = list(index.groupby("file", sort=True))
    if args.limit:
        groups = groups[: args.limit]

    for i, (name, grp) in enumerate(groups, 1):
        if name in sink.done:
            continue
        path = raw_path("breast", grp.iloc[0], args.data_root)
        want = sorted(int(v) for v in grp["slice"])
        t0 = time.time()
        try:
            slices, mags, _cplx, n_part, center_part = B.reconstruct_file(
                path, gridder, dcf_w, spoke_ids, len(want))
        except Exception as exc:  # noqa: BLE001
            logger.error("[breast] FAILED %s (%s)", path.name, exc)
            sink.fail(name, exc)
            continue

        # reconstruct_file re-derives its own slice list from the slab profile.
        # It is deterministic, so it should reproduce the cached list exactly;
        # if it does not, that is itself a finding and must not be papered over.
        matched = list(slices) == want
        if not matched:
            logger.warning("[breast] %s: recomputed slices %s != cached %s; "
                           "measuring the recomputed set and flagging the row",
                           name, list(slices), want)

        by_slice = {int(r["slice"]): r for _, r in grp.iterrows()}
        rows = []
        with h5py.File(path, "r") as f:
            tv = f["temptv"]
            n_tv = int(tv.shape[0])
            for zi, z in enumerate(slices):
                ref = np.asarray(tv[z]).mean(axis=0)
                r = pearson(mags[zi], ref)
                ns = null_slice(int(z), n_tv, args.null_shift)
                rn = (pearson(mags[zi], np.asarray(tv[ns]).mean(axis=0))
                      if ns >= 0 else float("nan"))
                src = by_slice.get(int(z), grp.iloc[0])
                rows.append(_slice_row(
                    "breast", name, src, int(z), r, rn, "temptv",
                    extra=f"n_partitions={n_part};center_partition={center_part};"
                          f"dcf={dcf};slice_list_matches_cache={matched}"))
        frec = _finish_file(
            "breast", name, rows,
            dict(n_partitions=int(n_part), center_partition=int(center_part),
                 dcf=dcf, slice_list_matches_cache=matched))
        sink.add(rows, frec)
        logger.info("[breast] %3d/%d %s: %d slices  r mean=%.4f min=%.4f "
                    "N=%d p0=%d (%.1fs)", i, len(groups), name, len(rows),
                    frec["r_mean"], frec["r_min"], n_part, center_part,
                    time.time() - t0)


def _slice_row(cohort, name, src_row, s, r, r_null, reference, extra=""):
    """Build one per-slice record, including the caching-time cross-check."""
    col = CACHED_COL[cohort]
    cached = src_row.get(col, "") if hasattr(src_row, "get") else ""
    try:
        cached = float(cached)
    except (TypeError, ValueError):
        cached = float("nan")
    return dict(
        cohort=cohort,
        file=name,
        folder=str(src_row.get("folder", "")) if hasattr(src_row, "get") else "",
        slice=int(s),
        cache_idx=int(src_row["idx"]) if "idx" in src_row else -1,
        reference=reference,
        r=float(r),
        r_cached=cached,
        r_delta=float(r - cached) if np.isfinite(cached) and np.isfinite(r)
        else float("nan"),
        r_null_shift=float(r_null),
        r_margin=float(r - r_null) if np.isfinite(r_null) and np.isfinite(r)
        else float("nan"),
        extra=extra,
    )


MEASURERS = {
    "prostate_t2": measure_prostate_t2,
    "prostate_dwi": measure_prostate_dwi,
    "breast": measure_breast,
    "brain": lambda idx, a, sink: measure_brainknee("brain", idx, a, sink),
    "knee": lambda idx, a, sink: measure_brainknee("knee", idx, a, sink),
}


# ==========================================================================
# summarising
# ==========================================================================

def describe(values, thresholds=THRESHOLDS) -> dict:
    """
    n / mean / median / min / max plus the below-threshold counts.

    ``n`` is the number of FINITE values actually summarised and ``n_missing``
    the number that could not be measured, kept separate so a cohort cannot
    look complete by silently dropping the slices that failed.
    """
    a = np.asarray(list(values), dtype=float)
    finite = a[np.isfinite(a)]
    out = dict(n=int(finite.size), n_missing=int(a.size - finite.size))
    if finite.size == 0:
        out.update(mean=None, median=None, min=None, max=None, std=None, p05=None)
        for t in thresholds:
            out[f"n_below_{t:g}"] = 0
            out[f"frac_below_{t:g}"] = None
        return out
    out.update(
        mean=float(finite.mean()),
        median=float(np.median(finite)),
        min=float(finite.min()),
        max=float(finite.max()),
        std=float(finite.std(ddof=1)) if finite.size > 1 else 0.0,
        p05=float(np.percentile(finite, 5)),
    )
    for t in thresholds:
        n_bad = int((finite < t).sum())
        out[f"n_below_{t:g}"] = n_bad
        out[f"frac_below_{t:g}"] = float(n_bad / finite.size)
    return out


def verdict_for(cohort: str, per_slice_stats: dict, extra_stats: dict) -> dict:
    """
    Compare the observed distribution with what the stage-2 reader documented.

    Returns a verdict dict; ``status`` is one of

      "matches_documented_claim"  observed mean >= claim - tol
      "BELOW_DOCUMENTED_CLAIM"    observed mean is materially worse
      "not_measured"              nothing finite to compare

    A discrepancy is a finding about the study, not about this module, and is
    printed and persisted as such.
    """
    spec = CLAIMS[cohort]
    stats_by_name = {"per_slice": per_slice_stats}
    stats_by_name.update(extra_stats)
    out = dict(reference=spec["reference"], claim=spec["claim"],
               claim_statistic=spec["statistic"], tolerance=spec["tol"],
               claim_source=spec["source"], comparison_strength=spec["strength"],
               note=spec["note"])
    st = stats_by_name.get(spec["statistic"])
    if not st or st.get("mean") is None:
        out.update(status="not_measured", observed_mean=None, delta=None)
        return out
    obs = st["mean"]
    out.update(observed_mean=obs, observed_median=st["median"],
               observed_min=st["min"], delta=obs - spec["claim"],
               status=("matches_documented_claim"
                       if obs >= spec["claim"] - spec["tol"]
                       else "BELOW_DOCUMENTED_CLAIM"))
    sec = spec.get("secondary")
    if sec:
        sst = stats_by_name.get(sec["statistic"])
        if sst and sst.get("mean") is not None:
            out["secondary"] = dict(
                statistic=sec["statistic"], claim=sec["claim"],
                observed_mean=sst["mean"], delta=sst["mean"] - sec["claim"],
                status=("matches_documented_claim"
                        if sst["mean"] >= sec["claim"] - sec["tol"]
                        else "BELOW_DOCUMENTED_CLAIM"))
    return out


def summarise_cohort(cohort: str, slices: pd.DataFrame, files: pd.DataFrame,
                     failures: list) -> dict:
    per_slice = describe(slices["r"])
    per_file = describe(files["r_mean"])
    extra = {}
    if cohort == "prostate_dwi" and "r_lowb_avg" in files.columns:
        extra["per_file_lowb_averaged"] = describe(files["r_lowb_avg"])

    delta = slices["r_delta"].to_numpy(dtype=float)
    delta = delta[np.isfinite(delta)]
    cross = dict(
        n_compared=int(delta.size),
        n_cached_missing=int(len(slices) - delta.size),
        max_abs_delta=float(np.abs(delta).max()) if delta.size else None,
        mean_abs_delta=float(np.abs(delta).mean()) if delta.size else None,
        note="r_cached is the value stage 2 wrote into "
             f"pipeline_out/cache/{cohort}_index.csv column "
             f"'{CACHED_COL[cohort]}' at caching time; it is rounded there "
             "(4 or 6 dp), so a delta of that order is rounding, not drift.",
    )

    out = dict(
        cohort=cohort,
        reference=CLAIMS[cohort]["reference"],
        reference_is_ground_truth=(cohort != "breast"),
        n_files=int(files.shape[0]),
        n_slices=int(slices.shape[0]),
        per_slice=per_slice,
        per_file=per_file,
        anatomy_support_null=dict(
            r_null_shift=describe(slices["r_null_shift"]),
            r_margin=describe(slices["r_margin"], thresholds=()),
            note="r_null_shift correlates our slice against the VENDOR "
                 "reference at a different slice of the same volume. It is the "
                 "floor r sits at from shared body support alone; r_margin = "
                 "r - r_null_shift is the slice-specific part.",
        ),
        cross_check_vs_cache=cross,
        failures=failures,
    )
    out.update(extra)
    out["verdict"] = verdict_for(cohort, per_slice, extra)
    if cohort == "breast":
        out["caveat"] = (
            "temptv is the vendor's TEMPORAL-TV-REGULARISED reconstruction of "
            "the same radial k-space, NOT an independent ground truth. r here "
            "measures agreement between two estimators with different priors "
            "(ours: unregularised density-compensated adjoint NUFFT), so it "
            "neither upper- nor lower-bounds fidelity cleanly: the TV prior "
            "removes noise we keep (pessimistic), while a shared trajectory "
            "error would corrupt both and partially cancel (optimistic). This "
            "is the weakest of the five comparisons and must be reported as "
            "such. The breast PHASE channel is not validated at all, because "
            "temptv is magnitude-only."
        )
    if cohort == "prostate_dwi":
        out["caveat"] = (
            "The per-slice number is deliberately depressed: trace_b50 averages "
            "~14 acquisitions and the cache stores ONE low-b volume, because "
            "complex-averaging diffusion data is invalid. "
            "per_file_lowb_averaged is the statistic that isolates "
            "reconstruction correctness from single-average noise."
        )
    return out


# ==========================================================================
# figure
# ==========================================================================

FIG_ROWS = [
    ("prostate_t2", "prostate_t2  (PRIMARY)\nvs reconstruction_rss", "per_slice"),
    ("prostate_dwi", "prostate_dwi\nvs trace_b50, single low-b", "per_slice"),
    ("prostate_dwi", "prostate_dwi\nvs trace_b50, low-b averaged", "lowb"),
    ("breast", "breast\nvs temptv  (regularised ref)", "per_slice"),
    ("brain", "brain\nvs reconstruction_rss", "per_slice"),
    ("knee", "knee\nvs reconstruction_rss", "per_slice"),
]


def _load_for_figure(out_dir: Path):
    data = {}
    for cohort in COHORT_ORDER:
        sp = out_dir / f"{cohort}.csv"
        fp = out_dir / f"{cohort}_perfile.csv"
        if sp.exists():
            data[cohort] = dict(slices=pd.read_csv(sp),
                                files=pd.read_csv(fp) if fp.exists() else None)
    return data


def make_figure(out_dir: Path, dpi: int = 300) -> list[Path]:
    """
    Publication figure: per-cohort distribution of the fidelity metric.

    (a) the headline distribution, one row per cohort, violin + box + the
        below-threshold gridlines the summary counts against;
    (b) the same rows as the ANATOMY-SUPPORT MARGIN r - r_null_shift, which is
        what stops panel (a) being read as "0.97 is close to 1 so it is nearly
        perfect" when a wrong-but-plausible image of the same breast would
        already score highly.

    prostate_dwi appears twice on purpose: the single-average row is what the
    cache stores, the low-b-averaged row is what tests the reconstruction, and
    hiding either would misrepresent the cohort.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    data = _load_for_figure(out_dir)
    if not data:
        raise SystemExit(f"no per-cohort CSVs in {out_dir}; run the measurement first")

    rows = []
    for cohort, label, kind in FIG_ROWS:
        if cohort not in data:
            continue
        if kind == "lowb":
            f = data[cohort]["files"]
            if f is None or "r_lowb_avg" not in f.columns:
                continue
            v = f["r_lowb_avg"].to_numpy(float)
            # The low-b-averaged number is a per-FILE statistic evaluated at one
            # slice, so there is no matching per-slice null to subtract; panel
            # (b) is left blank for this row rather than filled with a number
            # that would not mean the same thing as its neighbours.
            m = np.array([], float)
            unit = "per file"
        else:
            s = data[cohort]["slices"]
            v = s["r"].to_numpy(float)
            m = s["r_margin"].to_numpy(float)
            unit = "per slice"
        v = v[np.isfinite(v)]
        m = m[np.isfinite(m)]
        if v.size == 0:
            continue
        rows.append(dict(cohort=cohort, label=label, vals=v, margin=m, unit=unit,
                         weak=(cohort == "breast")))

    n = len(rows)
    fig, (ax, axm, axt) = plt.subplots(
        1, 3, figsize=(13.6, 0.95 * n + 3.0), sharey=True,
        gridspec_kw=dict(width_ratios=[1.75, 0.95, 0.80], wspace=0.07))

    strong = "#1f4e79"
    weak = "#b3541e"
    ypos = np.arange(n)[::-1]

    # The x axis stops at 0.80. prostate_dwi has a tail reaching 0.089 (one
    # exam whose single low-b averages are almost pure noise) and letting it
    # set the limit squashes every other cohort into the last 2% of the panel.
    # Nothing is hidden: values below the axis are drawn as a triangle at the
    # boundary WITH THEIR COUNT, and the true minimum of every row is printed
    # in panel (c).
    lo = min(float(np.min(r["vals"])) for r in rows)
    xlo = 0.80 if lo < 0.80 else math.floor(lo * 100) / 100 - 0.01

    def band(a, y, vals, colour):
        if vals.size > 1 and vals.std() > 1e-12:
            parts = a.violinplot([vals], positions=[y], vert=False, widths=0.68,
                                 showextrema=False, showmedians=False)
            for b in parts["bodies"]:
                b.set_facecolor(colour)
                b.set_alpha(0.28)
                b.set_edgecolor("none")
        a.boxplot([vals], positions=[y], vert=False, widths=0.18,
                  showfliers=False, patch_artist=True, whis=(5, 95),
                  medianprops=dict(color="white", lw=1.5),
                  boxprops=dict(facecolor=colour, edgecolor=colour, lw=0.8),
                  whiskerprops=dict(color=colour, lw=1.0),
                  capprops=dict(color=colour, lw=1.0))

    for y, rec in zip(ypos, rows):
        c = weak if rec["weak"] else strong
        v = rec["vals"]
        band(ax, y, v, c)
        # The minimum matters more than the box here -- a single badly
        # reconstructed slice is a finding -- so it gets its own tick.
        n_off = int((v < xlo).sum())
        if n_off:
            ax.plot([xlo], [y], marker="<", ms=6, color=c, lw=0,
                    clip_on=False, zorder=5)
            ax.text(xlo + 0.004, y - 0.30,
                    f"{n_off} below axis (min {v.min():.3f})", fontsize=6.8,
                    color=c, ha="left", va="center")
        else:
            ax.plot([v.min()], [y], marker="|", ms=10, color=c, lw=0)
        if rec["margin"].size:
            band(axm, y, rec["margin"], c)
        else:
            axm.text(0.5, y, "n/a  (per-file statistic)", va="center",
                     ha="center", fontsize=7.2, color="#999999",
                     transform=axm.get_yaxis_transform())

    for t, style in zip(THRESHOLDS, ("-", "--", ":")):
        ax.axvline(t, color="#999999", lw=0.8, ls=style, zorder=0)
        ax.text(t, 1.004, f"{t:g}", ha="center", va="bottom", fontsize=7,
                color="#777777", transform=ax.get_xaxis_transform())
    ax.axvline(1.0, color="#444444", lw=0.9, zorder=0)

    ax.set_yticks(ypos)
    ax.set_yticklabels([r["label"] for r in rows], fontsize=8.6)
    ax.set_ylim(-0.62, n - 0.38)
    ax.set_xlim(xlo, 1.0)
    ax.set_xlabel("Pearson r,  our reconstructed magnitude vs the vendor "
                  "reference in the same HDF5", fontsize=9)
    ax.set_title("(a)  reconstruction fidelity, every cached slice",
                 fontsize=10, loc="left", pad=16)

    axm.axvline(0.0, color="#444444", lw=0.9, zorder=0)
    axm.set_xlabel("r  -  r(our slice, vendor ref. at slice ± shift)", fontsize=9)
    axm.set_title("(b)  margin over the anatomy-support null", fontsize=10,
                  loc="left", pad=16)

    for a in (ax, axm):
        a.grid(axis="x", color="#eeeeee", lw=0.6, zorder=0)
        a.set_axisbelow(True)
        for side in ("top", "right"):
            a.spines[side].set_visible(False)

    # (c) is a table, not a plot: the r values in (a) differ in the 4th decimal
    # and no axis can show that honestly at this scale.
    axt.set_axis_off()
    cols = ((0.02, "left"), (0.44, "right"), (0.50, "left"), (0.78, "left"))
    for (x, align), head in zip(cols, ("unit", "n", "mean r", "min r")):
        axt.text(x, 1.004, head, fontsize=7.5, color="#555555",
                 transform=axt.transAxes, ha=align, va="bottom")
    axt.set_title("(c)  values", fontsize=10, loc="left", pad=16)
    for y, rec in zip(ypos, rows):
        v = rec["vals"]
        cells = (rec["unit"], f"{v.size}", f"{v.mean():.5f}", f"{v.min():.5f}")
        for (x, align), txt in zip(cols, cells):
            axt.text(x, y, txt, fontsize=7.8, color="#222222", ha=align,
                     va="center", transform=axt.get_yaxis_transform(),
                     family="DejaVu Sans Mono")

    handles = [Line2D([], [], color=strong, lw=6, alpha=0.5,
                      label="vendor reference is an unregularised reconstruction "
                            "of the same k-space"),
               Line2D([], [], color=weak, lw=6, alpha=0.5,
                      label="vendor reference is itself REGULARISED (temporal "
                            "TV) — the weakest comparison in the study")]
    fig.legend(handles=handles, loc="lower left", ncol=1, fontsize=8.2,
               frameon=False, bbox_to_anchor=(0.055, 0.005))
    fig.suptitle("Reconstruction fidelity of the PhaseDx k-space → image cache",
                 fontsize=12, x=0.008, ha="left", y=0.985)
    fig.text(0.008, 0.075,
             "prostate_dwi appears twice: the cache stores ONE low-b volume "
             "while the vendor trace_b50 averages ~14 acquisitions, so the "
             "per-slice row carries a deliberate SNR penalty;\nthe low-b-averaged "
             "row is the statistic that tests the reconstruction itself. Every "
             "vendor reference is a MAGNITUDE image — none of this validates the "
             "phase channel.",
             fontsize=7.8, color="#444444", ha="left", va="bottom")
    fig.subplots_adjust(left=0.145, right=0.995, top=0.86, bottom=0.20)

    paths = []
    for ext, kw in (("png", dict(dpi=dpi)), ("pdf", {})):
        p = out_dir / f"fig_recon_fidelity.{ext}"
        fig.savefig(p, bbox_inches="tight", facecolor="white", **kw)
        paths.append(p)
    plt.close(fig)
    return paths


# ==========================================================================
# printing
# ==========================================================================

def print_cohort(summary: dict) -> None:
    c = summary["cohort"]
    ps = summary["per_slice"]
    pf = summary["per_file"]
    v = summary["verdict"]
    print("\n" + "=" * 78)
    print(f"COHORT {c}   reference = {summary['reference']}"
          f"   ({'ground-truth-like' if summary['reference_is_ground_truth'] else 'REGULARISED, not ground truth'})")
    print("=" * 78)
    print(f"  files measured : {summary['n_files']}"
          + (f" of {summary['n_files_expected']} in the cache"
             if "n_files_expected" in summary else "")
          + ("" if summary.get("complete", True) else "   *** PARTIAL RUN ***"))
    print(f"  slices measured: {summary['n_slices']}")
    if ps["n"]:
        print("  PER CACHED SLICE")
        print(f"    n={ps['n']}  mean={ps['mean']:.6f}  median={ps['median']:.6f}  "
              f"min={ps['min']:.6f}  max={ps['max']:.6f}  p05={ps['p05']:.6f}")
        print("    below thresholds: " + "  ".join(
            f"<{t:g}: {ps[f'n_below_{t:g}']} ({100 * ps[f'frac_below_{t:g}']:.1f}%)"
            for t in THRESHOLDS))
    if pf["n"]:
        print("  PER FILE (mean of its slices)")
        print(f"    n={pf['n']}  mean={pf['mean']:.6f}  median={pf['median']:.6f}  "
              f"min={pf['min']:.6f}  max={pf['max']:.6f}")
        print("    below thresholds: " + "  ".join(
            f"<{t:g}: {pf[f'n_below_{t:g}']}" for t in THRESHOLDS))
    lb = summary.get("per_file_lowb_averaged")
    if lb and lb["n"]:
        print("  PER FILE, LOW-B VOLUMES MAGNITUDE-AVERAGED AS THE VENDOR DOES")
        print("    (this is the number that isolates GRAPPA + geometry + coil "
              "combination")
        print("     from the SNR cost of caching a single average)")
        print(f"    n={lb['n']}  mean={lb['mean']:.6f}  median={lb['median']:.6f}  "
              f"min={lb['min']:.6f}  max={lb['max']:.6f}")
        print("    below thresholds: " + "  ".join(
            f"<{t:g}: {lb[f'n_below_{t:g}']}" for t in THRESHOLDS))
    nul = summary["anatomy_support_null"]["r_null_shift"]
    mar = summary["anatomy_support_null"]["r_margin"]
    if nul["n"] and mar["n"]:
        print("  ANATOMY-SUPPORT NULL (our slice vs the vendor reference at "
              "another slice)")
        print(f"    r_null mean={nul['mean']:.6f}   margin (r - r_null) "
              f"mean={mar['mean']:.6f} median={mar['median']:.6f} "
              f"min={mar['min']:.6f}")
    cc = summary["cross_check_vs_cache"]
    if cc["n_compared"]:
        print("  CROSS-CHECK vs the value stage 2 recorded at caching time")
        print(f"    n={cc['n_compared']}  max|delta|={cc['max_abs_delta']:.6f}  "
              f"mean|delta|={cc['mean_abs_delta']:.6f}")
    if summary["failures"]:
        print(f"  FILES THAT COULD NOT BE MEASURED: {len(summary['failures'])}")
        for fl in summary["failures"][:10]:
            print(f"    {fl['file']}: {fl['error']}")
    print(f"  DOCUMENTED CLAIM: {v['claim']:g} on {v['claim_statistic']} "
          f"({v['comparison_strength']} comparison)")
    print(f"    source: {v['claim_source']}")
    if v["observed_mean"] is not None:
        print(f"    observed {v['observed_mean']:.6f}  delta {v['delta']:+.6f}  "
              f"-> {v['status']}")
    if "secondary" in v:
        s = v["secondary"]
        print(f"    secondary claim {s['claim']:g} on {s['statistic']}: observed "
              f"{s['observed_mean']:.6f}  delta {s['delta']:+.6f} -> {s['status']}")
    if summary.get("caveat"):
        print("  CAVEAT")
        for line in _wrap(summary["caveat"], 72):
            print(f"    {line}")


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


# ==========================================================================
# self-test  (no drive required)
# ==========================================================================

def self_test() -> int:
    checks, failed = 0, []

    def check(name, cond):
        nonlocal checks
        checks += 1
        if cond:
            print(f"  ok    {name}")
        else:
            failed.append(name)
            print(f"  FAIL  {name}")

    print("s10_reconfidelity self-test")
    rng = np.random.default_rng(11)

    # --- 1. the metric ---------------------------------------------------
    img = rng.normal(size=(64, 64))
    check("pearson(x, x) == 1", abs(pearson(img, img) - 1.0) < 1e-12)
    check("pearson is invariant to positive affine rescaling of either input "
          "(so a global intensity-scale difference between our recon and the "
          "vendor image cannot be mistaken for infidelity)",
          abs(pearson(img, 3.7 * img + 12.0) - 1.0) < 1e-12)
    check("pearson(x, -x) == -1", abs(pearson(img, -img) + 1.0) < 1e-12)
    a, b = rng.normal(size=4096), rng.normal(size=4096)
    check("pearson == np.corrcoef on the same pair",
          abs(pearson(a, b) - np.corrcoef(a, b)[0, 1]) < 1e-12)
    check("pearson returns NaN (not 0) on a constant input, so 'unmeasurable' "
          "is never averaged in as 'uncorrelated'",
          math.isnan(pearson(np.ones(100), rng.normal(size=100))))
    try:
        pearson(np.ones((4, 4)), np.ones((5, 5)))
        check("pearson raises on a shape mismatch", False)
    except ValueError:
        check("pearson raises on a shape mismatch", True)

    # --- 2. agreement with the stage-2 estimators ------------------------
    # This is the load-bearing one: the numbers this module writes must be the
    # same statistic the caches recorded, or r_delta is meaningless.
    try:
        from pipeline import s02_prostate as P
        from pipeline import s02_brainknee as BK
        x = rng.gamma(2.0, size=(80, 80))
        y = 0.8 * x + rng.gamma(2.0, size=(80, 80))
        check("pearson == s02_prostate.ncc", abs(pearson(x, y) - P.ncc(x, y)) < 1e-12)
        check("pearson == s02_brainknee.ncc", abs(pearson(x, y) - BK.ncc(x, y)) < 1e-12)
        check("pearson == the np.corrcoef s02_breast uses",
              abs(pearson(x, y) - float(np.corrcoef(x.ravel(), y.ravel())[0, 1])) < 1e-12)
    except Exception as exc:  # noqa: BLE001
        check(f"stage-2 estimators importable for comparison ({exc})", False)

    # --- 3. the null-slice picker ----------------------------------------
    check("null slice prefers s+shift inside the volume", null_slice(4, 30, 5) == 9)
    check("null slice falls back to s-shift at the top of the volume",
          null_slice(29, 30, 5) == 24)
    check("null slice never returns the slice itself (which would score 1.0 "
          "and make the null look like perfect agreement)",
          all(null_slice(s, 8, 5) != s for s in range(8)))
    check("null slice is undefined for a single-slice volume",
          null_slice(0, 1, 5) == -1)

    # --- 4. reconstruction round-trip on synthetic k-space ---------------
    # A phantom pushed through the same primitives the readers use must come
    # back correlating 1.0 with itself; this catches an FFT-centring or crop
    # regression in common.py without needing the drive.
    from pipeline.common import fft2c, ifft2c, rss
    yy, xx = np.mgrid[0:64, 0:64]
    phantom = ((yy - 32) ** 2 + (xx - 32) ** 2 < 20 ** 2).astype(float)
    phantom += 0.4 * ((yy - 26) ** 2 + (xx - 38) ** 2 < 6 ** 2)
    coils = np.stack([phantom * (0.5 + 0.5 * np.exp(-((xx - c) ** 2) / 900.0))
                      for c in (5, 30, 58)])
    k = fft2c(coils.astype(np.complex128))
    back = rss(ifft2c(k), coil_axis=0)
    ref = rss(coils.astype(np.complex128), coil_axis=0)
    check("fft2c -> ifft2c -> rss round-trips to r = 1 on a phantom",
          abs(pearson(back, ref) - 1.0) < 1e-10)
    shifted = np.roll(ref, 7, axis=0)
    check("a 7-pixel misregistration is detected (r drops well below 0.99), "
          "i.e. the metric can actually fail",
          pearson(shifted, ref) < 0.99)

    # --- 5. describe() ----------------------------------------------------
    vals = [1.0, 0.995, 0.98, 0.94, 0.88, float("nan")]
    d = describe(vals)
    check("describe counts finite values only", d["n"] == 5 and d["n_missing"] == 1)
    check("describe median is right", abs(d["median"] - 0.98) < 1e-12)
    check("describe counts below 0.99 correctly", d["n_below_0.99"] == 3)
    check("describe counts below 0.95 correctly", d["n_below_0.95"] == 2)
    check("describe counts below 0.9 correctly", d["n_below_0.9"] == 1)
    check("describe on an all-NaN input reports n=0 rather than crashing",
          describe([float("nan")] * 3)["n"] == 0)

    # --- 6. the verdict logic actually fires -----------------------------
    good = verdict_for("prostate_t2", describe([0.9985] * 10), {})
    check("a matching prostate_t2 mean is reported as matching",
          good["status"] == "matches_documented_claim")
    bad = verdict_for("prostate_t2", describe([0.93] * 10), {})
    check("a materially worse prostate_t2 mean is reported as BELOW, not "
          "rounded up",
          bad["status"] == "BELOW_DOCUMENTED_CLAIM" and bad["delta"] < 0)
    dwi = verdict_for("prostate_dwi", describe([0.89] * 10),
                      {"per_file_lowb_averaged": describe([0.80] * 10)})
    check("prostate_dwi is judged on the low-b-AVERAGED statistic, so a good "
          "per-slice value cannot rescue a bad reconstruction",
          dwi["status"] == "BELOW_DOCUMENTED_CLAIM")
    dwi2 = verdict_for("prostate_dwi", describe([0.60] * 10),
                       {"per_file_lowb_averaged": describe([0.975] * 10)})
    check("a low per-slice DWI value alone does not condemn the reconstruction "
          "(that is the single-average SNR penalty, documented in s02)",
          dwi2["status"] == "matches_documented_claim"
          and dwi2["secondary"]["status"] == "BELOW_DOCUMENTED_CLAIM")
    check("brain/knee are held to 1.0 within 0.001, not to 0.99",
          verdict_for("brain", describe([0.995] * 5), {})["status"]
          == "BELOW_DOCUMENTED_CLAIM")

    # --- 7. breast is flagged as the weak comparison everywhere ----------
    check("breast is declared the WEAK comparison in CLAIMS",
          CLAIMS["breast"]["strength"] == "WEAK")
    fake_slices = pd.DataFrame(dict(r=[0.97, 0.98], r_cached=[0.97, 0.98],
                                    r_delta=[0.0, 0.0],
                                    r_null_shift=[0.90, 0.91],
                                    r_margin=[0.07, 0.07]))
    fake_files = pd.DataFrame(dict(r_mean=[0.975]))
    s = summarise_cohort("breast", fake_slices, fake_files, [])
    check("the breast summary JSON carries the temptv-is-not-ground-truth "
          "caveat as data, not as a comment",
          "REGULARISED" in s["caveat"] and s["reference_is_ground_truth"] is False)
    check("non-breast cohorts are marked reference_is_ground_truth",
          summarise_cohort("brain", fake_slices, fake_files,
                           [])["reference_is_ground_truth"] is True)

    # --- 8. path resolution ----------------------------------------------
    row = dict(file="f.h5", folder="fastMRI_prostate_T2_IDS_001_020")
    check("prostate raw path is built from the release root + folder",
          raw_path("prostate_t2", row, Path("/D")) ==
          Path("/D/prostate/fastMRI_prostate_T2_IDS_001_020/f.h5"))
    check("breast raw path uses breast_updated/breast",
          raw_path("breast", dict(file="b.h5", folder="IDS"), Path("/D")) ==
          Path("/D/breast_updated/breast/IDS/b.h5"))
    check("brain/knee raw path uses the absolute source_dir recorded in the "
          "index, not a glob (a moved file must be an error, not a silent "
          "match on basename)",
          raw_path("brain", dict(file="x.h5", source_dir="/D/brain/val"),
                   Path("/other")) == Path("/D/brain/val/x.h5"))

    # --- 9. the resume checkpoint cannot double-count a slice ------------
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        sink = Sink(td, "knee")
        for fn in ("a.h5", "b.h5"):
            rws = [dict(cohort="knee", file=fn, folder="", slice=s,
                        cache_idx=s, reference="reconstruction_rss", r=1.0,
                        r_cached=1.0, r_delta=0.0, r_null_shift=0.4,
                        r_margin=0.6, extra="") for s in range(3)]
            sink.add(rws, _finish_file("knee", fn, rws))
        # simulate a kill between the slice write and the per-file write
        with open(td / "knee.csv", "a") as fh:
            fh.write("knee,c.h5,,0,0,reconstruction_rss,1,1,0,0.4,0.6,\n")
        resumed = Sink(td, "knee")
        s2, f2 = resumed.load()
        check("resume skips files that are already measured",
              resumed.done == {"a.h5", "b.h5"})
        check("resume drops the orphan slice rows of a file interrupted "
              "mid-write, so re-measuring it cannot double-count",
              len(s2) == 6 and "c.h5" not in set(s2["file"]))
        check("resume keeps one per-file row per file", len(f2) == 2)
        fresh = Sink(td, "knee", resume=False)
        check("--no-resume clears the checkpoint",
              not fresh.slice_path.exists() and fresh.done == set())

    # --- 10. figure builds from CSVs alone -------------------------------
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        for cohort, centre in (("prostate_t2", 0.999), ("prostate_dwi", 0.89),
                               ("breast", 0.97), ("brain", 1.0), ("knee", 1.0)):
            k = 60
            r = np.clip(centre - np.abs(rng.normal(0, 0.006, k)), 0, 1)
            pd.DataFrame(dict(cohort=cohort, file="f", folder="", slice=range(k),
                              cache_idx=range(k), reference="ref", r=r,
                              r_cached=r, r_delta=0.0,
                              r_null_shift=r - 0.08, r_margin=0.08,
                              extra="")).to_csv(td / f"{cohort}.csv", index=False)
            pf = dict(cohort=cohort, file=["f"], n_slices=[k], n_finite=[k],
                      r_mean=[float(r.mean())], r_median=[float(np.median(r))],
                      r_min=[float(r.min())], r_max=[float(r.max())],
                      r_null_mean=[float(r.mean() - 0.08)])
            if cohort == "prostate_dwi":
                pf["r_lowb_avg"] = [0.972]
            pd.DataFrame(pf).to_csv(td / f"{cohort}_perfile.csv", index=False)
        try:
            paths = make_figure(td, dpi=72)
            check("the figure builds from the per-cohort CSVs alone "
                  "(so --figure-only works after parallel cohort runs)",
                  all(p.exists() and p.stat().st_size > 5000 for p in paths))
        except Exception as exc:  # noqa: BLE001
            check(f"the figure builds from the per-cohort CSVs alone ({exc})", False)

    # --- 11. real data, only if the drive happens to be mounted ----------
    idx = CACHE_DIR / "prostate_t2_index.csv"
    if idx.exists() and DATA_ROOT.exists():
        try:
            from pipeline import s02_prostate as P
            df = pd.read_csv(idx)
            row0 = df.iloc[0]
            p = raw_path("prostate_t2", row0, DATA_ROOT)
            with P.open_h5(p) as f:
                s = int(row0["slice"])
                mag_native, _, _ = P.t2_slice_recon(f, s, True, (224, 224))
                r = pearson(P.center_crop2d(mag_native, P.T2_VENDOR_HW),
                            f["reconstruction_rss"][s])
            check(f"live check: recomputed r={r:.6f} reproduces the cached "
                  f"recon_ncc={float(row0['recon_ncc']):.6f} for "
                  f"{row0['file']} slice {s}",
                  abs(r - float(row0["recon_ncc"])) < 5e-4)
        except Exception as exc:  # noqa: BLE001
            print(f"  skip  live drive check ({type(exc).__name__}: {exc})")
    else:
        print("  skip  live drive check (cache index or data root not present)")

    print(f"\ns10_reconfidelity self-test: {checks - len(failed)}/{checks} checks passed")
    for name in failed:
        print(f"  FAILED: {name}")
    return 0 if not failed else 1


# ==========================================================================
# driver
# ==========================================================================

def run_cohort(cohort: str, args) -> dict:
    index = load_index(cohort, args.cache_dir)
    expected = int(index["file"].nunique())
    logger.info("[%s] cache index: %d slices over %d files", cohort, len(index),
                expected)
    t0 = time.time()
    # The per-slice/per-file CSVs are written by the Sink as the run proceeds,
    # at 12 significant figures rather than the 4 the stage-2 index CSVs use.
    # brain and knee land within ~1e-14 of 1 -- double-precision round-off --
    # so their `r` column does print as "1"; the residual is not lost, because
    # `r_delta` is formed in memory at full precision against the caching-time
    # value and is what the summary's cross_check_vs_cache block reports.
    sink = Sink(args.out, cohort, resume=not args.no_resume)
    MEASURERS[cohort](index, args, sink)
    if not sink.slice_path.exists():
        raise SystemExit(f"[{cohort}] nothing measured")

    sdf, fdf = sink.load()
    summary = summarise_cohort(cohort, sdf, fdf, sink.failures)
    summary["elapsed_s"] = round(time.time() - t0, 1)
    summary["n_files_expected"] = expected
    # A run that was cut short leaves valid CSVs for the files it did reach.
    # Say so in the JSON rather than letting a partial pass be read as a
    # complete one.
    summary["complete"] = bool(
        args.limit is not None
        or summary["n_files"] + len(sink.failures) >= expected)
    summary["per_slice_csv"] = str(sink.slice_path)
    summary["per_file_csv"] = str(sink.file_path)
    with open(args.out / f"{cohort}.json", "w") as fh:
        json.dump(summary, fh, indent=2)
    print_cohort(summary)
    return summary


def aggregate(args) -> dict:
    """Merge whatever per-cohort JSONs exist into one summary + the figure."""
    combined = dict(
        generated_by="pipeline/s10_reconfidelity.py",
        what="Pearson correlation between the PhaseDx reconstructed magnitude "
             "and the vendor reference image shipped in the same HDF5, "
             "recomputed for every cached slice.",
        null_shift=args.null_shift,
        thresholds=list(THRESHOLDS),
        scope_note="Every vendor reference in these releases is a MAGNITUDE "
                   "image, so these numbers validate the magnitude "
                   "reconstruction only. The phase channel is never directly "
                   "validated against any reference and inherits credibility "
                   "only through sharing the same complex image.",
        cohorts={},
    )
    for cohort in COHORT_ORDER:
        p = args.out / f"{cohort}.json"
        if p.exists():
            combined["cohorts"][cohort] = json.loads(p.read_text())

    discrepancies = []
    for cohort, s in combined["cohorts"].items():
        v = s["verdict"]
        if v["status"] == "BELOW_DOCUMENTED_CLAIM":
            discrepancies.append(
                f"{cohort}: documented {v['claim']:g} on {v['claim_statistic']}, "
                f"observed {v['observed_mean']:.6f} (delta {v['delta']:+.6f})")
        sec = v.get("secondary")
        if sec and sec["status"] == "BELOW_DOCUMENTED_CLAIM":
            discrepancies.append(
                f"{cohort} [secondary]: documented {sec['claim']:g} on "
                f"{sec['statistic']}, observed {sec['observed_mean']:.6f} "
                f"(delta {sec['delta']:+.6f})")
    combined["discrepancies_vs_documented_claims"] = discrepancies

    with open(args.out / "recon_fidelity_summary.json", "w") as fh:
        json.dump(combined, fh, indent=2)

    print("\n" + "=" * 78)
    print("SUMMARY TABLE  (per cached slice unless marked otherwise)")
    print("=" * 78)
    hdr = (f"{'cohort':<14}{'reference':<20}{'n':>6}{'mean':>10}{'median':>10}"
           f"{'min':>10}{'<0.99':>7}{'<0.95':>7}{'<0.90':>7}{'null':>9}{'margin':>9}")
    print(hdr)
    print("-" * len(hdr))
    for cohort, s in combined["cohorts"].items():
        ps = s["per_slice"]
        if not ps["n"]:
            continue
        nul = s["anatomy_support_null"]["r_null_shift"]["mean"]
        mar = s["anatomy_support_null"]["r_margin"]["mean"]
        print(f"{cohort:<14}{s['reference']:<20}{ps['n']:>6}{ps['mean']:>10.6f}"
              f"{ps['median']:>10.6f}{ps['min']:>10.6f}"
              f"{ps['n_below_0.99']:>7}{ps['n_below_0.95']:>7}{ps['n_below_0.9']:>7}"
              + (f"{nul:>9.4f}{mar:>9.4f}" if nul is not None else " " * 18))
        lb = s.get("per_file_lowb_averaged")
        if lb and lb["n"]:
            print(f"{'  (per file,':<14}{'low-b averaged)':<20}{lb['n']:>6}"
                  f"{lb['mean']:>10.6f}{lb['median']:>10.6f}{lb['min']:>10.6f}"
                  f"{lb['n_below_0.99']:>7}{lb['n_below_0.95']:>7}"
                  f"{lb['n_below_0.9']:>7}")
    print("\n'null' is the mean r of our slice against the VENDOR reference at "
          "another slice of the\nsame volume, i.e. the floor r sits at from "
          "shared body support alone; 'margin' is r - null.")
    print()
    if discrepancies:
        print("DISCREPANCIES vs the claims in the stage-2 docstrings "
              "(reported, not smoothed):")
        for d in discrepancies:
            print(f"  * {d}")
    else:
        print("No cohort's fidelity is materially below what its stage-2 "
              "reader documents.")
    print("\nRead the breast row with the caveat attached: temptv is the "
          "vendor's temporal-TV-\nREGULARISED reconstruction of the same "
          "radial k-space, not an independent ground\ntruth, so that r is "
          "agreement between two estimators and is the weakest of the\nfive "
          "comparisons. It also has by far the highest null (0.80): a breast "
          "slice already\ncorrelates 0.80 with a DIFFERENT slice of the same "
          "breast, so its 0.977 buys the least\nslice-specific evidence of any "
          "cohort. The breast phase channel is not validated at all.")

    if not args.no_figure:
        try:
            for p in make_figure(args.out, dpi=args.dpi):
                print(f"\nfigure -> {p}")
        except SystemExit as exc:
            logger.warning("figure skipped: %s", exc)
    print(f"summary -> {args.out / 'recon_fidelity_summary.json'}")
    return combined


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="PhaseDx stage 10: persist reconstruction fidelity as data",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--cohort", default="all",
                   help="comma-separated subset of " + ",".join(COHORT_ORDER)
                        + ", or 'all'")
    p.add_argument("--limit", type=int, default=None,
                   help="measure only the first N files per cohort")
    p.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    p.add_argument("--data-root", type=Path, default=DATA_ROOT)
    p.add_argument("--out", type=Path, default=OUT_DIR_DEFAULT)
    p.add_argument("--null-shift", type=int, default=5,
                   help="slice offset used for the anatomy-support null")
    p.add_argument("--dwi-lowb-slices", type=int, default=1,
                   help="slices per AXDIFF file at which the low-b-averaged "
                        "comparison is evaluated (each costs ~50 GRAPPA recons)")
    p.add_argument("--target-hw", type=int, nargs=2, default=(224, 224),
                   help="network grid the stage-2 readers resample to; only "
                        "affects the code path, not the validation crop")
    p.add_argument("--grid-oversamp", type=float, default=1.25,
                   help="breast gridder oversampling (must match s02_breast)")
    p.add_argument("--no-resume", action="store_true",
                   help="discard any existing per-cohort CSV and re-measure "
                        "from scratch (default is to resume: results are "
                        "checkpointed after every file)")
    p.add_argument("--figure-only", action="store_true",
                   help="rebuild the summary + figure from existing CSVs")
    p.add_argument("--no-figure", action="store_true")
    p.add_argument("--dpi", type=int, default=300)
    p.add_argument("--self-test", action="store_true",
                   help="run the offline self-test and exit")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        return self_test()

    if args.figure_only:
        aggregate(args)
        return 0

    cohorts = (list(COHORT_ORDER) if args.cohort == "all"
               else [c.strip() for c in args.cohort.split(",") if c.strip()])
    unknown = [c for c in cohorts if c not in MEASURERS]
    if unknown:
        raise SystemExit(f"unknown cohort(s) {unknown}; choose from {COHORT_ORDER}")

    for cohort in cohorts:
        run_cohort(cohort, args)
    aggregate(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
