"""
s03_train.py
------------
Stage 3b of the PhaseDx pipeline: training and test-set inference.

Trains the ResNet-18 from s03_model.py on the stage-2 cache, once per
(cohort, condition, seed). Reading from the cache rather than the raw HDF5 on
the external drive is the whole reason this stage takes minutes instead of
days: the expensive part (coil combination, FFTs, resampling) already happened
once in stage 2 and the result is a contiguous (N, 224, 224) float16 array.

Non-negotiables enforced here, because they are the ways this study dies:

* Subject-level split integrity. If a single subject contributes slices to two
  splits, adjacent slices of the same anatomy end up on both sides of the
  train/test boundary and the reported AUC is meaningless. We assert
  disjointness of subject_id across splits and raise rather than warn. The
  splits themselves come from the dataset's own 'official_split' column, not
  from a random resplit, so results are comparable to published FastMRI work.

  The unit is subject_id, NOT patient_id. The fastMRI breast release issues a
  repeat scan of the same woman a different coded patient name and links the
  two only through the label sheet's 'Repeated scan' group; a patient_id-keyed
  check cannot see that and accepts the leak. Stage 1 resolves the groups and
  emits subject_id, stage 2 now writes it into the cache index, and this stage
  REFUSES TO RUN if it is absent -- silently degrading to patient_id here is
  exactly how the audited version let a leaking breast index through.

* Phase is fed as (sin, cos) in radians, never min-max scaled. See common.py.

* Class imbalance is handled by resampling, not by silently reporting accuracy
  on a 94%-negative test set. Prostate slice-level positives (PIRADS>=3) are
  608 / 9490 = 6.4%; a constant "negative" predictor scores 93.6% accuracy and
  0.5 AUC, so only AUC/AP are reported.

* Every test prediction is saved together with its patient_id. Stage 4 needs
  those to do patient-level aggregation and a patient-clustered bootstrap;
  a slice-level bootstrap that ignores patient clustering produces confidence
  intervals that are far too narrow.

Usage:
    # smoke test, no drive access at all
    python pipeline/s03_train.py --dry-run

    # real run: all three conditions, three seeds
    python pipeline/s03_train.py --cohort prostate_dwi --seeds 42,123,7

    # single condition, frozen backbone
    python pipeline/s03_train.py --cohort breast --conditions phase --freeze-backbone

    # background-only falsification control
    python pipeline/s03_train.py --cohort prostate_dwi --region background
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
from s03_model import build_model, count_parameters, input_channels_for  # noqa: E402

logger = logging.getLogger("s03_train")

SPLITS = ("training", "validation", "test")

# The cache index may carry split labels in several dialects depending on which
# label file it came from (prostate CSVs use words, the breast XLSX uses 0/1).
_SPLIT_ALIASES = {
    "training": "training", "train": "training", "0": "training", "0.0": "training",
    "validation": "validation", "val": "validation", "valid": "validation",
    "test": "test", "testing": "test", "1": "test", "1.0": "test",
}


# --------------------------------------------------------------------------
# Environment
# --------------------------------------------------------------------------

def pick_device(requested: str = "auto") -> torch.device:
    """CUDA > MPS > CPU. MPS is what this machine actually has."""
    if requested != "auto":
        return torch.device(requested)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def to_device(t: torch.Tensor, device: torch.device) -> torch.Tensor:
    """
    Host-to-device copy, with non_blocking used only where it is safe.

    Measured on this machine (torch 2.11, MPS): copying an unpinned CPU tensor
    with non_blocking=True returns a corrupted tensor 20 times out of 20 --
    the copy is queued asynchronously and the source buffer is recycled before
    it completes. Training silently produced NaN losses. non_blocking is only
    sound when the source is pinned, which we only do for CUDA, so gate it on
    device type rather than passing it unconditionally.
    """
    device = torch.device(device)
    return t.to(device, non_blocking=(device.type == "cuda"))


def set_seed(seed: int) -> None:
    """Seed every RNG that can affect a run."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


# --------------------------------------------------------------------------
# Splits
# --------------------------------------------------------------------------

def normalize_splits(df: pd.DataFrame) -> pd.DataFrame:
    """Map whatever dialect the index CSV used onto training/validation/test."""
    raw = df["official_split"].astype(str).str.strip().str.lower()
    mapped = raw.map(_SPLIT_ALIASES)
    unknown = sorted(set(raw[mapped.isna()]))
    if unknown:
        raise ValueError(
            f"unrecognised official_split values {unknown}; "
            f"expected one of {sorted(set(_SPLIT_ALIASES))}"
        )
    out = df.copy()
    out["official_split"] = mapped.to_numpy()
    return out


SUBJECT_COL = common.SUBJECT_COL          # "subject_id"


def require_subject_column(df: pd.DataFrame) -> str:
    """
    Return the split-enforcement column, or raise.

    This mirrors s05_controls.resolve_subject_col's *resolution rule* --
    subject_id when present and complete -- but not its fallback. s05 warns and
    degrades to patient_id; this stage refuses, because this is the path that
    produces the published number. patient_id does not collapse the breast
    repeat-scan groups, so a patient_id-keyed split can put the same physical
    woman on both sides of the train/test line under two different coded names.
    Stage 1 already catches that on its own table; the audited failure was that
    the same leak, expressed through the *cache index*, was accepted silently
    because the cache index carried no subject_id at all.

    Recovering the column costs one join and no reconstruction:
        python pipeline/s02_breast.py --backfill-subject-ids
        python pipeline/s02_prostate.py --backfill-subject-ids
    """
    if SUBJECT_COL not in df.columns:
        raise RuntimeError(
            f"SUBJECT ID MISSING: the cache index has no {SUBJECT_COL!r} column, so "
            "splits could only be enforced on patient_id -- which does NOT collapse "
            "the breast repeat-scan groups and therefore cannot detect the same "
            "subject appearing in two splits under two coded names.\n"
            "Refusing to train. Backfill it from the stage-1 cohort CSV with:\n"
            "    python pipeline/s02_breast.py   --backfill-subject-ids\n"
            "    python pipeline/s02_prostate.py --backfill-subject-ids"
        )
    n_null = int(df[SUBJECT_COL].isna().sum())
    if n_null:
        raise RuntimeError(
            f"SUBJECT ID INCOMPLETE: {n_null}/{len(df)} rows have a null "
            f"{SUBJECT_COL!r}. A partially populated subject column silently "
            "degrades to per-row groups for the missing rows, which defeats the "
            "check entirely. Refusing to train."
        )
    return SUBJECT_COL


def assert_no_patient_leakage(df: pd.DataFrame, split_col: str = "official_split") -> None:
    """
    Hard failure if any subject appears in more than one split.

    This is deliberately an exception and not a warning. Subject leakage is the
    single most common reason a medical-imaging result fails to replicate, and
    a warning in a log nobody reads is not a control.

    Both units are checked, subject_id first: patient_id must be a refinement of
    subject_id, so a patient-level leak is also a subject-level leak, but
    reporting both makes the diagnosis unambiguous when only one fires. The
    patient_id pass is retained purely as a redundant check -- it is strictly
    weaker and must never be the only one that runs. The function keeps its
    historical name so existing callers and self-tests keep working.
    """
    subject_col = require_subject_column(df)
    for group_col, kind in ((subject_col, "SUBJECT"), ("patient_id", "PATIENT")):
        if group_col not in df.columns:
            continue
        per_group = df.groupby(group_col)[split_col].nunique()
        offenders = per_group[per_group > 1]
        if len(offenders):
            detail = []
            for gid in list(offenders.index)[:10]:
                splits = sorted(df.loc[df[group_col] == gid, split_col].unique())
                # Name the coded patients inside a leaking subject: for breast
                # they are the two different names the same woman was given,
                # which is the whole point of the subject unit.
                members = sorted(
                    str(v) for v in df.loc[df[group_col] == gid, "patient_id"].unique()
                ) if "patient_id" in df.columns else []
                extra = f"  (patient_id: {members})" if group_col != "patient_id" else ""
                detail.append(f"    {gid!r}: {splits}{extra}")
            raise RuntimeError(
                f"{kind} LEAKAGE: {len(offenders)} {group_col}(s) appear in more "
                "than one split.\n"
                + "\n".join(detail)
                + ("\n    ..." if len(offenders) > 10 else "")
                + "\nRefusing to train. Fix the cache index before proceeding."
            )
    logger.info(
        "leakage check passed: %d subjects (%d patients), disjoint across %d splits",
        df[subject_col].nunique(),
        df["patient_id"].nunique() if "patient_id" in df.columns else -1,
        df[split_col].nunique(),
    )


def carve_validation_from_training(
    df: pd.DataFrame, val_frac: float, seed: int
) -> pd.DataFrame:
    """
    Split a subject-grouped validation set out of training.

    The breast cohort's label file only defines training/testing, so there is
    no official validation split to select the early-stopping epoch on. We
    carve one out by *subject*, never by slice and never by patient_id, and
    with a seed that is fixed independently of the run seed so that
    magnitude/phase/both runs all see the identical validation set. Without
    that, the three conditions would be compared on different data and the
    comparison would be confounded.

    Grouping on subject_id rather than patient_id is what stops the carve from
    manufacturing a leak of its own: the breast repeat-scan groups are two coded
    patient names for one woman, so a patient-grouped draw can put one of her
    scans in training and the other in validation, and then epoch selection is
    made on data the model has memorised.
    """
    subject_col = require_subject_column(df)
    train_mask = df["official_split"] == "training"
    subjects = df.loc[train_mask, subject_col].drop_duplicates().sort_values().to_numpy()

    # Stratify the subject draw by subject-level positivity so a small
    # validation set does not come out single-class (AUC would be undefined).
    sub_label = (
        df.loc[train_mask].groupby(subject_col)["label"].max().reindex(subjects).to_numpy()
    )
    rng = np.random.default_rng(seed)
    chosen = []
    for cls in (0, 1):
        pool = subjects[sub_label == cls]
        if len(pool) == 0:
            continue
        n_val = max(1, int(round(val_frac * len(pool)))) if len(pool) > 1 else 0
        chosen.extend(rng.permutation(pool)[:n_val].tolist())

    out = df.copy()
    # Restrict to rows that are CURRENTLY training. Without the train_mask this
    # assignment relabels every row of a chosen subject, including any that were
    # in test -- which does not just move data, it launders a leak: a subject
    # straddling training/test gets rewritten to sit entirely in validation and
    # the downstream leakage assertion then passes. Found by the defect-4
    # regression test; s01_labels.carve_validation already masks this way.
    out.loc[train_mask & out[subject_col].isin(chosen), "official_split"] = "validation"
    logger.info(
        "carved validation from training: %d/%d subjects, %d slices (val_split_seed=%d)",
        len(chosen), len(subjects), int((out["official_split"] == "validation").sum()), seed,
    )
    return out


def prepare_index(
    df: pd.DataFrame, val_frac: float, val_split_seed: int,
    split_col: str = "official_split",
) -> pd.DataFrame:
    """Normalize dtypes/splits, ensure a validation split exists, assert integrity."""
    df = df.copy()

    # Cross-validation folds. Stage 1 emits one complete training/validation/test
    # column per fold ('cv0_split' .. 'cv{K-1}_split'), each already built at
    # subject level with a nested inner validation fold. Aliasing the requested
    # column onto 'official_split' HERE, at the single entry point, means every
    # split decision, leakage assertion and validation carve downstream is
    # untouched and cannot disagree about which fold it is looking at.
    #
    # This is needed because the official folds are too small to support a
    # confirmatory claim: prostate_t2's official test split holds 7 subjects and
    # prostate_dwi's holds 4, against s06's floor of 10 clusters / 5 per class.
    # Under 5-fold CV both prostate_t2 and breast clear that floor on 5/5 folds.
    if split_col != "official_split":
        if split_col not in df.columns:
            raise ValueError(
                f"requested --split-col {split_col!r} is not in the cache index; "
                f"available split columns: "
                f"{sorted(c for c in df.columns if c.endswith('_split'))}. "
                f"Run s01_labels.py to regenerate the cohort, then backfill the "
                f"cache index."
            )
        # Rows outside every fold (corrupt files, rows stage 1 could not place)
        # carry a null here. Dropping them is correct -- they have no fold to be
        # trained or tested in -- but it must be visible, not silent.
        n_unassigned = int(df[split_col].isna().sum())
        if n_unassigned:
            logger.info(
                "%s: dropping %d row(s) with no assignment in %s",
                split_col, n_unassigned, split_col,
            )
            df = df[df[split_col].notna()].copy()
        if df.empty:
            raise ValueError(f"no rows carry an assignment in {split_col!r}")
        df["official_split"] = df[split_col]

    # subject_id is required, not optional: it is the unit every split decision
    # below is made on. Checking it here (rather than only inside the leakage
    # assertion) means an index without it can never reach carve_validation,
    # which would otherwise have drawn a patient-grouped validation set.
    require_subject_column(df)
    required = {"idx", "patient_id", SUBJECT_COL, "label", "official_split"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"cache index is missing required columns: {sorted(missing)}")

    # Nulls here would become garbage integers under .astype(int) rather than
    # raising, so reject them before any coercion.
    for col in sorted(required):
        n_null = int(df[col].isna().sum())
        if n_null:
            raise ValueError(f"cache index column {col!r} has {n_null} null value(s)")

    df["patient_id"] = df["patient_id"].astype(str)
    df[SUBJECT_COL] = df[SUBJECT_COL].astype(str)
    # patient_id must be a refinement of subject_id (many patients -> one
    # subject, never the reverse). If one coded patient claimed two subjects the
    # grouping would be incoherent and neither unit would mean anything.
    spread = df.groupby("patient_id")[SUBJECT_COL].nunique()
    bad = spread[spread > 1]
    if len(bad):
        raise ValueError(
            f"cache index is incoherent: {len(bad)} patient_id(s) map to more than "
            f"one {SUBJECT_COL}, e.g. {list(bad.index)[:5]}"
        )
    df["idx"] = df["idx"].astype(int)
    df["label"] = df["label"].astype(float).round().astype(int)
    bad = sorted(set(df["label"]) - {0, 1})
    if bad:
        raise ValueError(f"'label' must be binary 0/1, found {bad}")
    if df["idx"].duplicated().any():
        raise ValueError("cache index has duplicate 'idx' values; each row must be one slice")

    df = normalize_splits(df)

    # Assert BEFORE the carve as well as after. The carve only ever relabels
    # training rows, so it cannot create a leak -- but asserting first means a
    # leak already present in the cache index is attributed to the cache index,
    # and no future change to the carve can mask one.
    assert_no_patient_leakage(df)
    if (df["official_split"] == "validation").sum() == 0:
        df = carve_validation_from_training(df, val_frac, val_split_seed)
        assert_no_patient_leakage(df)
    return df


# --------------------------------------------------------------------------
# Dataset
# --------------------------------------------------------------------------

class CacheSliceDataset(Dataset):
    """
    One cached slice -> one (C, H, W) float32 tensor.

    Backed either by the stage-2 HDF5 cache (real runs) or by in-memory arrays
    (--dry-run). The HDF5 handle is opened lazily inside __getitem__ so that
    the dataset survives being forked into DataLoader workers; h5py handles are
    not fork-safe if opened in the parent.

    Channel construction, per condition:
        magnitude : [ zscore(mag) ]
        phase     : [ sin(phase), cos(phase) ]
        both      : [ zscore(mag), sin(phase), cos(phase) ]

    The magnitude channel is z-scored per slice so the network cannot use the
    global brightness of a slice -- which tracks receive gain and therefore
    scanner/session -- as a shortcut. sin/cos are already bounded in [-1, 1]
    and are deliberately NOT rescaled: any per-slice rescaling of phase would
    destroy its physical units and make each pixel depend on the most extreme
    pixel elsewhere in the slice.
    """

    def __init__(
        self,
        rows: pd.DataFrame,
        condition: str,
        h5_path: Path | None = None,
        arrays: dict | None = None,
        region: str = "full",
    ):
        if (h5_path is None) == (arrays is None):
            raise ValueError("provide exactly one of h5_path or arrays")
        if region not in ("full", "body", "background"):
            raise ValueError(f"unknown region {region!r}")

        self.rows = rows.reset_index(drop=True)
        self.condition = condition
        self.in_channels = input_channels_for(condition)
        self.h5_path = Path(h5_path) if h5_path is not None else None
        self.arrays = arrays
        self.region = region

        self.indices = self.rows["idx"].to_numpy(dtype=np.int64)
        self.labels = self.rows["label"].to_numpy(dtype=np.int64)
        self.patient_ids = self.rows["patient_id"].astype(str).to_numpy()
        self._h5 = None

    def __len__(self) -> int:
        return len(self.rows)

    def _read(self, k: int):
        """Fetch (mag, phase, mask) for cache row k as float32/bool numpy."""
        if self.arrays is not None:
            mag = self.arrays["mag"][k]
            phase = self.arrays["phase"][k]
            mask = self.arrays["mask"][k]
        else:
            if self._h5 is None:
                import h5py

                self._h5 = h5py.File(self.h5_path, "r")
            mag = self._h5["mag"][k]
            phase = self._h5["phase"][k]
            mask = self._h5["mask"][k]
        return (
            np.asarray(mag, dtype=np.float32),
            np.asarray(phase, dtype=np.float32),
            np.asarray(mask, dtype=bool),
        )

    def __getitem__(self, i: int):
        k = int(self.indices[i])
        mag, phase, mask = self._read(k)

        # region: 'body' keeps anatomy, 'background' keeps only air. The
        # background-only run is the falsification control -- if a phase model
        # still scores well on pure background, it is reading the scanner's
        # shim/coil signature, not the tumour.
        if self.region == "body":
            keep = mask
        elif self.region == "background":
            keep = ~mask
        else:
            keep = None

        chans = []
        if self.condition in ("magnitude", "both"):
            stats_mask = keep if (keep is not None and keep.sum() > 16) else None
            chans.append(common.zscore(mag, mask=stats_mask))
        if self.condition in ("phase", "both"):
            chans.extend(list(common.phase_channels(phase)))

        x = np.stack(chans, axis=0).astype(np.float32)
        if keep is not None:
            x = x * keep[None, :, :].astype(np.float32)
        x = np.nan_to_num(x, nan=0.0, posinf=0.0, neginf=0.0)

        return torch.from_numpy(x), int(self.labels[i]), i


def make_sampler(labels: np.ndarray, seed: int) -> WeightedRandomSampler:
    """
    Inverse-frequency WeightedRandomSampler.

    At 6% positives an unweighted epoch shows the network ~2 positives per
    batch of 32, and with label smoothing plus cross entropy it converges to
    predicting the prior. Sampling each class with equal expected frequency
    fixes that without discarding any negatives (unlike undersampling).
    """
    counts = np.bincount(labels, minlength=2).astype(np.float64)
    if (counts == 0).any():
        raise RuntimeError(f"training split is single-class: counts={counts.tolist()}")
    per_class = 1.0 / counts
    weights = per_class[labels]
    gen = torch.Generator()
    gen.manual_seed(seed)
    return WeightedRandomSampler(
        weights=torch.as_tensor(weights, dtype=torch.double),
        num_samples=len(labels),
        replacement=True,
        generator=gen,
    )


# --------------------------------------------------------------------------
# Optimisation
# --------------------------------------------------------------------------

def make_scheduler(optimizer, steps_per_epoch: int, epochs: int, warmup_epochs: int = 2):
    """Linear warmup for warmup_epochs, then cosine decay to zero. Steps per batch."""
    total = max(1, steps_per_epoch * epochs)
    warmup = min(total - 1, steps_per_epoch * warmup_epochs)

    def lr_lambda(step: int) -> float:
        if warmup > 0 and step < warmup:
            return float(step + 1) / float(warmup)
        progress = (step - warmup) / max(1, total - warmup)
        progress = min(1.0, max(0.0, progress))
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def binary_metrics(labels: np.ndarray, probs: np.ndarray) -> dict:
    """AUC and average precision, NaN-safe when a split is single-class."""
    from sklearn.metrics import average_precision_score, roc_auc_score

    out = {"n": int(len(labels)), "n_pos": int(labels.sum())}
    if len(np.unique(labels)) < 2:
        out["auc"] = float("nan")
        out["ap"] = float("nan")
        return out
    out["auc"] = float(roc_auc_score(labels, probs))
    out["ap"] = float(average_precision_score(labels, probs))
    return out


@torch.no_grad()
def evaluate(model, loader, device, criterion) -> dict:
    """Forward the whole loader; return loss, metrics, and per-sample outputs."""
    model.eval()
    losses, probs, labels, order = [], [], [], []
    for x, y, i in loader:
        x = to_device(x, device)
        y = to_device(y, device)
        logits = model(x)
        losses.append(criterion(logits, y).item() * len(y))
        probs.append(torch.softmax(logits.float(), dim=1)[:, 1].cpu().numpy())
        labels.append(y.cpu().numpy())
        order.append(i.numpy())

    probs = np.concatenate(probs) if probs else np.zeros(0)
    labels = np.concatenate(labels) if labels else np.zeros(0, dtype=int)
    order = np.concatenate(order) if len(order) else np.zeros(0, dtype=int)
    n = max(1, len(labels))
    result = binary_metrics(labels, probs)
    result["loss"] = float(sum(losses) / n)
    result["probs"] = probs
    result["labels"] = labels
    result["row_order"] = order
    return result


def train_one_epoch(model, loader, device, criterion, optimizer, scheduler, grad_clip) -> dict:
    model.train()
    total_loss, total_n = 0.0, 0
    for x, y, _ in loader:
        x = to_device(x, device)
        y = to_device(y, device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(x)
        loss = criterion(logits, y)
        if not torch.isfinite(loss):
            # A non-finite loss poisons every BatchNorm running stat in one
            # step, after which the run reports NaN AUC rather than failing.
            # Stop at the source so the cause is still visible.
            raise RuntimeError(
                f"non-finite training loss ({loss.item()}) at step {total_n // max(1, len(y))}; "
                "refusing to continue -- inspect the cache for NaN/inf slices"
            )
        loss.backward()
        if grad_clip and grad_clip > 0:
            nn.utils.clip_grad_norm_(
                [p for p in model.parameters() if p.requires_grad], grad_clip
            )
        optimizer.step()
        scheduler.step()
        total_loss += loss.item() * len(y)
        total_n += len(y)
    return {
        "loss": total_loss / max(1, total_n),
        "lr": float(optimizer.param_groups[0]["lr"]),
    }


# --------------------------------------------------------------------------
# One run = (cohort, condition, seed)
# --------------------------------------------------------------------------

def run_one(
    cohort: str,
    condition: str,
    seed: int,
    index: pd.DataFrame,
    device: torch.device,
    args,
    h5_path: Path | None = None,
    arrays: dict | None = None,
) -> dict:
    set_seed(seed)
    tag = f"{cohort}_{condition}_seed{seed}"
    logger.info("=" * 72)
    logger.info("RUN %s  device=%s  region=%s", tag, device, args.region)
    logger.info("=" * 72)

    subsets = {s: index[index["official_split"] == s] for s in SPLITS}
    for s in SPLITS:
        sub = subsets[s]
        logger.info(
            "  %-10s %5d slices  %4d patients  %4d pos (%.1f%%)",
            s, len(sub), sub["patient_id"].nunique(), int(sub["label"].sum()),
            100.0 * sub["label"].mean() if len(sub) else 0.0,
        )
    for s in SPLITS:
        if len(subsets[s]) == 0:
            # An empty validation split silently disables epoch selection and
            # early stopping, so the run would report whatever the last epoch
            # happened to give. Fail instead.
            raise RuntimeError(
                f"the {s!r} split is empty; cannot train/select/evaluate. "
                "Check official_split in the cache index (and --val-frac if the "
                "cohort has no official validation split)."
            )

    def make_loader(split, shuffle=False, sampler=None):
        ds = CacheSliceDataset(
            subsets[split], condition, h5_path=h5_path, arrays=arrays, region=args.region
        )
        gen = torch.Generator()
        gen.manual_seed(seed)
        return ds, DataLoader(
            ds,
            batch_size=args.batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=args.workers,
            pin_memory=(device.type == "cuda"),
            drop_last=False,
            generator=gen if (shuffle or sampler is not None) else None,
        )

    train_ds = CacheSliceDataset(
        subsets["training"], condition, h5_path=h5_path, arrays=arrays, region=args.region
    )
    sampler = make_sampler(train_ds.labels, seed)
    train_gen = torch.Generator()
    train_gen.manual_seed(seed)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        sampler=sampler,
        num_workers=args.workers,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        generator=train_gen,
    )
    val_ds, val_loader = make_loader("validation")
    test_ds, test_loader = make_loader("test")

    model = build_model(
        condition,
        pretrained=not args.no_pretrained,
        freeze_backbone=args.freeze_backbone,
        dropout=args.dropout,
    ).to(device)
    counts = count_parameters(model)

    criterion = nn.CrossEntropyLoss(label_smoothing=args.label_smoothing)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.wd
    )
    scheduler = make_scheduler(
        optimizer, max(1, len(train_loader)), args.epochs, args.warmup_epochs
    )

    history = []
    best = {"score": -np.inf, "epoch": -1, "state": None}
    since_best = 0
    t0 = time.time()

    for epoch in range(args.epochs):
        tr = train_one_epoch(
            model, train_loader, device, criterion, optimizer, scheduler, args.grad_clip
        )
        va = evaluate(model, val_loader, device, criterion) if len(val_ds) else {
            "loss": float("nan"), "auc": float("nan"), "ap": float("nan"),
            "n": 0, "n_pos": 0,
        }
        row = {
            "epoch": epoch,
            "train_loss": tr["loss"],
            "lr": tr["lr"],
            "val_loss": va["loss"],
            "val_auc": va["auc"],
            "val_ap": va["ap"],
            "seconds": time.time() - t0,
        }
        history.append(row)
        logger.info(
            "  epoch %2d/%d  train_loss=%.4f  val_loss=%.4f  val_auc=%.4f  val_ap=%.4f  lr=%.2e",
            epoch + 1, args.epochs, row["train_loss"], row["val_loss"],
            row["val_auc"], row["val_ap"], row["lr"],
        )

        # Early stopping on val AUC. If the validation split is single-class
        # (AUC undefined) we fall back to val loss so the run still selects an
        # epoch instead of keeping epoch 0 forever.
        score = row["val_auc"] if not math.isnan(row["val_auc"]) else -row["val_loss"]
        if score > best["score"] + args.min_delta:
            best = {
                "score": score,
                "epoch": epoch,
                "state": {k: v.detach().cpu().clone() for k, v in model.state_dict().items()},
            }
            since_best = 0
        else:
            since_best += 1
            if since_best >= args.patience:
                logger.info("  early stop at epoch %d (no val gain for %d epochs)",
                            epoch + 1, args.patience)
                break

    if best["state"] is not None:
        model.load_state_dict(best["state"])
    logger.info("  best epoch %d, selection score %.4f", best["epoch"] + 1, best["score"])

    val_out = evaluate(model, val_loader, device, criterion) if len(val_ds) else None
    test_out = evaluate(model, test_loader, device, criterion)
    logger.info(
        "  TEST auc=%.4f ap=%.4f  (n=%d, pos=%d)",
        test_out["auc"], test_out["ap"], test_out["n"], test_out["n_pos"],
    )

    results_dir = Path(args.results_dir)
    ckpt_dir = Path(args.ckpt_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    ckpt_path = ckpt_dir / f"{tag}.pt"
    torch.save(
        {
            "state_dict": best["state"] if best["state"] is not None else model.state_dict(),
            "cohort": cohort, "condition": condition, "seed": seed,
            "in_channels": model.in_channels, "freeze_backbone": model.freeze_backbone,
            "pretrained": model.pretrained, "best_epoch": best["epoch"],
            "best_selection_score": float(best["score"]),
            "args": vars(args),
        },
        ckpt_path,
    )

    def pack(split_out, ds):
        if split_out is None:
            return None
        order = split_out["row_order"].astype(int)
        return {
            "probs": [float(v) for v in split_out["probs"]],
            "labels": [int(v) for v in split_out["labels"]],
            "patient_ids": [str(p) for p in ds.patient_ids[order]],
            "cache_idx": [int(v) for v in ds.indices[order]],
            "auc": split_out["auc"], "ap": split_out["ap"],
            "loss": split_out["loss"], "n": split_out["n"], "n_pos": split_out["n_pos"],
        }

    payload = {
        "cohort": cohort,
        "condition": condition,
        "seed": seed,
        "region": args.region,
        "device": str(device),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": time.time() - t0,
        "model": {
            "arch": "resnet18",
            "in_channels": model.in_channels,
            "pretrained": model.pretrained,
            "freeze_backbone": model.freeze_backbone,
            "params_total": counts["total"],
            "params_trainable": counts["trainable"],
        },
        "hyperparams": {
            "lr": args.lr, "weight_decay": args.wd, "batch_size": args.batch_size,
            "epochs": args.epochs, "warmup_epochs": args.warmup_epochs,
            "label_smoothing": args.label_smoothing, "grad_clip": args.grad_clip,
            "dropout": args.dropout, "patience": args.patience,
        },
        "splits": {
            s: {
                "n_slices": int(len(subsets[s])),
                "n_patients": int(subsets[s]["patient_id"].nunique()),
                "n_pos": int(subsets[s]["label"].sum()),
            }
            for s in SPLITS
        },
        "best_epoch": best["epoch"],
        "best_selection_score": float(best["score"]),
        "history": history,
        "val": pack(val_out, val_ds),
        "test": pack(test_out, test_ds),
        "checkpoint": str(ckpt_path),
    }
    out_path = results_dir / f"{tag}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    logger.info("  wrote %s", out_path)
    logger.info("  wrote %s", ckpt_path)

    return {
        "tag": tag, "cohort": cohort, "condition": condition, "seed": seed,
        "test_auc": test_out["auc"], "test_ap": test_out["ap"],
        "best_epoch": best["epoch"], "results_json": str(out_path),
    }


# --------------------------------------------------------------------------
# Dry run: a fabricated cache, no drive access
# --------------------------------------------------------------------------

def make_dry_run_cache(n_patients: int = 18, slices_per_patient: int = 8,
                       hw=(64, 64), seed: int = 0):
    """
    Fabricate a tiny in-memory cache honouring the stage-2 contract.

    Positives get a Gaussian blob added to the phase map (strong) and to the
    magnitude (weak), so all three conditions are learnable and the smoke test
    exercises a non-degenerate optimisation. Splits are assigned per patient so
    the leakage assertion sees a legitimate index.

    subject_id is emitted alongside patient_id because the cache contract now
    requires it. Here the two are 1:1 (no fabricated repeat-scan groups); the
    breast-style many-patients-to-one-subject case is exercised by
    make_repeat_group_index() and check_leakage_guard() below.
    """
    rng = np.random.default_rng(seed)
    h, w = hw
    yy, xx = np.mgrid[0:h, 0:w]

    mags, phases, masks, rows = [], [], [], []
    for p in range(n_patients):
        pid = f"DRY{p:03d}"
        split = "training" if p < 12 else ("validation" if p < 15 else "test")
        positive = (p % 3 == 0)
        for s in range(slices_per_patient):
            label = int(positive and 2 <= s < 6)

            mag = 0.35 + 0.15 * rng.standard_normal((h, w)).astype(np.float32)
            body = ((yy - h / 2) ** 2 + (xx - w / 2) ** 2) < (0.35 * h) ** 2
            mag = mag + 0.5 * body

            phase = 0.4 * rng.standard_normal((h, w)).astype(np.float32)
            # Smooth B0-like background so phase is not pure noise.
            phase = phase + 1.5 * np.sin(2 * np.pi * xx / w) * np.cos(np.pi * yy / h)

            if label:
                cy = h / 2 + rng.normal(0, 3)
                cx = w / 2 + rng.normal(0, 3)
                blob = np.exp(-(((yy - cy) ** 2 + (xx - cx) ** 2) / (2 * (0.08 * h) ** 2)))
                phase = phase + 2.0 * blob
                mag = mag + 0.15 * blob

            phase = np.arctan2(np.sin(phase), np.cos(phase))  # wrap to [-pi, pi]
            mags.append(common.normalize_magnitude(mag).astype(np.float16))
            phases.append(phase.astype(np.float16))
            masks.append(body)
            rows.append({
                "idx": len(rows), "patient_id": pid, SUBJECT_COL: pid,
                "file": f"{pid}.h5", "slice": s,
                "label": label, "raw_label": label, "official_split": split,
                "folder": "dry", "acq": "DRY",
            })

    arrays = {
        "mag": np.stack(mags), "phase": np.stack(phases), "mask": np.stack(masks),
    }
    return arrays, pd.DataFrame(rows)


def make_repeat_group_index(n_patients: int = 8, slices_per_patient: int = 4) -> pd.DataFrame:
    """
    A cache index in the shape of the breast cohort's repeat-scan groups.

    Two DIFFERENT coded patient names (BR000 / BR001) belong to one physical
    woman and therefore share a subject_id, exactly as stage 1 emits for the
    fastMRI breast release, where the label sheet's 'Repeated scan' column is
    the only thing linking the two codes. Every other patient is its own
    subject. All rows start in 'training'; the tests below move one code into
    'test' to create the straddle.
    """
    rows = []
    for p in range(n_patients):
        pid = f"BR{p:03d}"
        # patients 0 and 1 are the same woman scanned twice
        sid = "breast_repeat_group_1" if p < 2 else pid
        for s in range(slices_per_patient):
            rows.append({
                "idx": len(rows), "patient_id": pid, SUBJECT_COL: sid,
                "file": f"{pid}_{s}.h5", "slice": s,
                "label": int(p % 2 == 0), "raw_label": int(p % 2 == 0),
                "official_split": "training", "folder": "rg", "acq": 1,
            })
    return pd.DataFrame(rows)


def _legacy_patient_only_leakage(df: pd.DataFrame) -> bool:
    """
    The pre-fix check, kept ONLY so the regression tests can demonstrate what it
    missed. Returns True if a patient_id spans more than one split.

    Do not call this from the pipeline. It exists to make the defect's failure
    mode reproducible rather than merely asserted in a comment.
    """
    per_patient = df.groupby("patient_id")["official_split"].nunique()
    return bool((per_patient > 1).any())


def check_leakage_guard() -> None:
    """
    Prove every leakage assertion actually fires.

    A guard that has never been observed to trigger is not a guard. Four
    scenarios, each of which reached a trained model in the audited version or
    would have under a weaker check:

      1. slice-level leak    -- one slice of a training patient moved to test.
      2. SUBJECT-level leak  -- two coded patient names of one woman split
                                across training/test. This is the audit's
                                scenario (B). The old patient_id-keyed check
                                CANNOT see it; the assertion below proves the
                                old check stays silent and the new one raises.
      3. missing subject_id  -- an index with no subject_id column at all, i.e.
                                every stage-2 cache as shipped before this fix.
                                Must refuse rather than fall back to patient_id.
      4. null subject_id     -- a partially populated column, which would
                                degrade to per-row groups for the missing rows.
    """
    # 1. slice-level leak
    _, df = make_dry_run_cache(n_patients=6, slices_per_patient=4)
    df.loc[0, "official_split"] = "test"
    try:
        prepare_index(df, val_frac=0.2, val_split_seed=0)
        raise AssertionError("LEAKAGE GUARD 1/4: FAILED -- a leaking index was accepted")
    except RuntimeError as exc:
        print(f"LEAKAGE GUARD 1/4 (slice): OK -- {str(exc).splitlines()[0]}", flush=True)

    # 2. subject-level leak across two coded patient names (audit scenario B)
    rg = make_repeat_group_index()
    rg.loc[rg["patient_id"] == "BR001", "official_split"] = "test"
    if _legacy_patient_only_leakage(rg):
        raise AssertionError(
            "LEAKAGE GUARD 2/4: test is invalid -- the patient_id-only check fired, "
            "so this scenario does not isolate the subject-level defect"
        )
    try:
        prepare_index(rg, val_frac=0.2, val_split_seed=0)
        raise AssertionError(
            "LEAKAGE GUARD 2/4: FAILED -- a repeat-scan group straddling the split "
            "was accepted (defect 4 is still open)"
        )
    except RuntimeError as exc:
        first = str(exc).splitlines()[0]
        if "SUBJECT LEAKAGE" not in first:
            raise AssertionError(f"LEAKAGE GUARD 2/4: wrong error -- {first}") from exc
        print(f"LEAKAGE GUARD 2/4 (subject, patient-check blind): OK -- {first}", flush=True)

    # 3. subject_id column absent entirely
    _, df3 = make_dry_run_cache(n_patients=6, slices_per_patient=4)
    df3 = df3.drop(columns=[SUBJECT_COL])
    try:
        prepare_index(df3, val_frac=0.2, val_split_seed=0)
        raise AssertionError(
            "LEAKAGE GUARD 3/4: FAILED -- an index with no subject_id was accepted"
        )
    except RuntimeError as exc:
        first = str(exc).splitlines()[0]
        if "SUBJECT ID MISSING" not in first:
            raise AssertionError(f"LEAKAGE GUARD 3/4: wrong error -- {first}") from exc
        print(f"LEAKAGE GUARD 3/4 (no subject_id): OK -- {first}", flush=True)

    # 4. subject_id present but partially null
    _, df4 = make_dry_run_cache(n_patients=6, slices_per_patient=4)
    df4.loc[0, SUBJECT_COL] = np.nan
    try:
        prepare_index(df4, val_frac=0.2, val_split_seed=0)
        raise AssertionError(
            "LEAKAGE GUARD 4/4: FAILED -- a partially null subject_id was accepted"
        )
    except RuntimeError as exc:
        first = str(exc).splitlines()[0]
        if "SUBJECT ID INCOMPLETE" not in first:
            raise AssertionError(f"LEAKAGE GUARD 4/4: wrong error -- {first}") from exc
        print(f"LEAKAGE GUARD 4/4 (null subject_id): OK -- {first}", flush=True)


def check_validation_carve_guard() -> None:
    """
    Prove carve_validation_from_training groups on subject_id, not patient_id.

    The carve is the second half of defect 4: even on an index whose official
    split is clean, a patient-grouped draw can put one scan of a repeat-scan
    woman in training and her other scan in validation, and epoch selection is
    then made on data the model has memorised. Sweep every seed in 0..49 and
    require that no subject is ever split -- one counterexample is enough.
    """
    rg = make_repeat_group_index(n_patients=10, slices_per_patient=4)
    split_seeds = []
    prev_level = logger.level
    logger.setLevel(logging.WARNING)          # 50 carves would emit 50 INFO lines
    try:
        for seed in range(50):
            out = carve_validation_from_training(rg, val_frac=0.4, seed=seed)
            per = out.groupby(SUBJECT_COL)["official_split"].nunique()
            if (per > 1).any():
                split_seeds.append(seed)
    finally:
        logger.setLevel(prev_level)
    if split_seeds:
        raise AssertionError(
            f"VALIDATION CARVE GUARD: FAILED -- subject split across training/validation "
            f"at seed(s) {split_seeds[:5]}"
        )
    # ...and prove the scenario is a real hazard, i.e. a patient-grouped carve
    # would have split that subject for at least one of those seeds. Without
    # this the test above could pass vacuously.
    hazard = 0
    for seed in range(50):
        pats = rg["patient_id"].drop_duplicates().sort_values().to_numpy()
        pat_label = rg.groupby("patient_id")["label"].max().reindex(pats).to_numpy()
        rng = np.random.default_rng(seed)
        chosen = []
        for cls in (0, 1):
            pool = pats[pat_label == cls]
            if len(pool) == 0:
                continue
            n_val = max(1, int(round(0.4 * len(pool)))) if len(pool) > 1 else 0
            chosen.extend(rng.permutation(pool)[:n_val].tolist())
        tmp = rg.copy()
        tmp.loc[tmp["patient_id"].isin(chosen), "official_split"] = "validation"
        if (tmp.groupby(SUBJECT_COL)["official_split"].nunique() > 1).any():
            hazard += 1
    if hazard == 0:
        raise AssertionError(
            "VALIDATION CARVE GUARD: test is vacuous -- the patient-grouped carve never "
            "split the repeat-scan subject, so this fixture proves nothing"
        )
    print(
        f"VALIDATION CARVE GUARD: OK -- subject-grouped carve never split a subject over "
        f"50 seeds; the old patient-grouped carve would have, at {hazard}/50 seeds",
        flush=True,
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PhaseDx stage 3: train magnitude/phase/both models")
    p.add_argument("--cohort", choices=list(common.COHORTS),
                   help="which cached cohort to train on (not needed with --dry-run)")
    p.add_argument("--conditions", default="all",
                   help="comma-separated subset of magnitude,phase,both (or 'all')")
    p.add_argument("--seeds", default="42,123,7", help="comma-separated seeds")
    p.add_argument("--region", choices=("full", "body", "background"), default="full",
                   help="'background' is the falsification control: anatomy removed")
    p.add_argument("--split-col", default="official_split",
                   help="which split column to train on. 'official_split' is the "
                        "dataset's own fold; 'cv0_split'..'cv4_split' are the "
                        "subject-level cross-validation folds from stage 1. The "
                        "official folds are too small for a confirmatory claim "
                        "(prostate_t2: 7 test subjects, prostate_dwi: 4), so CV "
                        "is the primary analysis and the official split is "
                        "reported as a secondary one")

    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--warmup-epochs", type=int, default=2)
    p.add_argument("--label-smoothing", type=float, default=0.05)
    p.add_argument("--grad-clip", type=float, default=1.0)
    p.add_argument("--dropout", type=float, default=0.3)
    p.add_argument("--patience", type=int, default=5, help="early-stopping patience on val AUC")
    p.add_argument("--min-delta", type=float, default=1e-4)
    p.add_argument("--freeze-backbone", action="store_true",
                   help="freeze conv1..layer3; train layer4 + head only")
    p.add_argument("--no-pretrained", action="store_true")

    p.add_argument("--val-frac", type=float, default=0.2,
                   help="fraction of training patients used for validation when the "
                        "cohort has no official validation split (breast)")
    p.add_argument("--val-split-seed", type=int, default=0,
                   help="fixed, independent of --seeds, so every condition sees the "
                        "same validation set")

    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))
    p.add_argument("--workers", type=int, default=0,
                   help="DataLoader workers; 0 is safest with MPS and h5py")
    p.add_argument("--cache-dir", default=str(common.CACHE_DIR))
    p.add_argument("--results-dir", default=str(common.RESULTS_DIR))
    p.add_argument("--ckpt-dir", default=str(common.CKPT_DIR))
    p.add_argument("--dry-run", action="store_true",
                   help="fabricate a tiny in-memory cache and run the full path")
    return p.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    conditions = (
        list(common.CONDITIONS) if args.conditions == "all"
        else [c.strip() for c in args.conditions.split(",") if c.strip()]
    )
    for c in conditions:
        input_channels_for(c)  # validates
    seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]

    device = pick_device(args.device)

    if args.dry_run:
        cohort = "dryrun"
        # Keep fabricated output out of the real results tree unless the user
        # explicitly pointed --results-dir/--ckpt-dir somewhere.
        if args.results_dir == str(common.RESULTS_DIR):
            args.results_dir = str(common.OUT_ROOT / "dryrun" / "results")
        if args.ckpt_dir == str(common.CKPT_DIR):
            args.ckpt_dir = str(common.OUT_ROOT / "dryrun" / "checkpoints")
        args.epochs = min(args.epochs, 3)
        args.warmup_epochs = min(args.warmup_epochs, 1)
        args.batch_size = min(args.batch_size, 16)
        seeds = seeds[:1]

        print("=" * 72, flush=True)
        print("DRY RUN -- fabricated in-memory cache, no drive access", flush=True)
        print("=" * 72, flush=True)
        check_leakage_guard()
        check_validation_carve_guard()
        arrays, raw_index = make_dry_run_cache()
        print(f"fabricated cache: mag{arrays['mag'].shape} {arrays['mag'].dtype}, "
              f"phase{arrays['phase'].shape} {arrays['phase'].dtype}, "
              f"mask{arrays['mask'].shape} {arrays['mask'].dtype}", flush=True)
        print(f"phase range: [{arrays['phase'].astype(np.float32).min():.3f}, "
              f"{arrays['phase'].astype(np.float32).max():.3f}] rad "
              f"(must lie inside [-pi, pi])", flush=True)
        index = prepare_index(raw_index, args.val_frac, args.val_split_seed)
        h5_path = None
    else:
        if not args.cohort:
            print("ERROR: --cohort is required (or use --dry-run)", file=sys.stderr)
            return 2
        cohort = args.cohort
        cache_dir = Path(args.cache_dir)
        h5_path = cache_dir / f"{cohort}.h5"
        idx_path = cache_dir / f"{cohort}_index.csv"
        for pth in (h5_path, idx_path):
            if not pth.exists():
                print(f"ERROR: missing {pth}. Run stage 2 first.", file=sys.stderr)
                return 2
        raw_index = pd.read_csv(idx_path)
        logger.info("loaded %s: %d rows", idx_path, len(raw_index))
        index = prepare_index(
            raw_index, args.val_frac, args.val_split_seed, split_col=args.split_col
        )
        arrays = None

    summary = []
    for condition in conditions:
        for seed in seeds:
            summary.append(
                run_one(cohort, condition, seed, index, device, args,
                        h5_path=h5_path, arrays=arrays)
            )

    print(flush=True)
    print("=" * 72, flush=True)
    print(f"SUMMARY  cohort={cohort}  region={args.region}  device={device}", flush=True)
    print("=" * 72, flush=True)
    print(f"{'condition':<12}{'seed':>6}{'best_ep':>9}{'test_auc':>10}{'test_ap':>10}", flush=True)
    for r in summary:
        print(f"{r['condition']:<12}{r['seed']:>6}{r['best_epoch'] + 1:>9}"
              f"{r['test_auc']:>10.4f}{r['test_ap']:>10.4f}")
    print(flush=True)
    print(f"results -> {args.results_dir}", flush=True)
    print(f"checkpoints -> {args.ckpt_dir}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
