"""
s05_controls.py
---------------
Stage 5 of the PhaseDx pipeline: the falsification suite.

Every other stage in this pipeline is trying to measure a phase effect. This
stage is trying to kill it. That asymmetry is deliberate. A prior manuscript
claimed AUC 0.85/0.97 for phase-only tumour classification; no code in this
repository has ever produced those numbers, and the single most plausible
explanation for a large phase effect is not biology but instrumentation --
MRI phase is an outstanding fingerprint of the coil, the shim, the receive
chain and the protocol. Prostate DWI here spans two institutions, two device
ids and six distinct receiver-channel counts (14/16/20/24/26/30). If patients
from one scanner are more likely to be positive, a phase model can score
beautifully while learning nothing about cancer.

So this module implements five controls, each of which is a way for the phase
result to die:

  1. label_permutation
        Shuffle the training labels at SUBJECT level, preserving class
        balance, retrain, evaluate on the real test labels. Repeated over many
        permutation seeds to build a null distribution of test AUC. If the
        headline AUC sits inside that null, there is no result. Permuting at
        slice level instead of subject level would be a broken control: two
        adjacent slices of the same prostate would get opposite labels, the
        task would become impossible for reasons that have nothing to do with
        the null hypothesis, and the null distribution would be biased low --
        which makes the headline look better than it is.

  2. background_only          <-- THE IMPORTANT ONE
        Zero everything INSIDE the body mask and train on air alone. Air has
        no anatomy. It does have coil sensitivity roll-off, B0 shim
        structure, Gibbs ringing, EPI ghosts and receiver noise colour -- i.e.
        the entire scanner signature and nothing else. If a phase model still
        classifies tumour from pure background, the model is reading the
        machine, and the headline number is an artefact regardless of what
        every other control says.

  3. phase_scramble
        Randomly permute phase pixels WITHIN the body mask. The marginal
        distribution of phase values inside the body is preserved exactly;
        all spatial structure is destroyed. A real anatomical phase effect
        must collapse. An effect that survives was never spatial -- it was a
        summary statistic of the phase histogram, which is exactly what a
        per-scanner offset looks like.

  4. acquisition_split
        Re-split so that acquisition metadata is HELD OUT rather than shared:
        train on one institution / receiver-channel group, test on another.
        Under the official split, scanner identity is shared across train and
        test, so a scanner-reading model is rewarded. Under this split it is
        punished. The AUC drop is the size of the confound.

  5. confound_predictability
        Train the identical network to predict a NON-diagnostic target from
        phase alone -- institution, receiver-channel bucket, source folder.
        This does not test the phase result at all; it measures the ceiling of
        the artefact. If phase predicts the scanner at AUC 0.99, then any
        phase-based claim in this dataset needs to clear a very high bar, and
        reviewers will (correctly) ask for exactly this number.

Design rules followed here:

  * Training is NOT reimplemented. Every control calls s03_train.run_one(),
    the same entry point the headline runs use, with the same model, the same
    optimiser, the same early stopping and the same leakage assertions. A
    control that trained differently from the headline would not be a control.

  * Every control emits the exact JSON schema s03_train.run_one() writes, plus
    a top-level "control" field naming it (and "control_detail" with the
    control's own bookkeeping). s04_stats.py can therefore consume control
    output unchanged and filter on payload.get("control", "none").

  * Splits are enforced on subject_id, not patient_id. Breast contributes two
    acquisitions per patient AND has repeated-scan groups that link different
    coded patient names; stage 1 emits subject_id as the unit that collapses
    both. The cache index does not always carry it, so it is joined back in
    from the stage-1 cohort CSV here.

  * No point estimate is ever printed bare. The prostate DWI test fold has
    four patients, three of them positive. Every AUC in the verdict table
    carries a SUBJECT-clustered percentile bootstrap interval -- clustered on
    the same unit the splits are enforced on, never on patient_id, because
    clustering on the finer unit returns an interval that is too narrow in
    exactly the direction that makes a control look like it survived -- and
    when that interval is uninformative the table says so instead of hiding it.

  * A control that could not run is recorded, not just logged. _guarded()
    swallows exceptions so one broken control cannot abort the other four;
    every swallowed failure is written to control_failures.json in the results
    tree (see FailureLedger). Stage 6 reads that file and reports the matching
    criterion MISSING. Without it, a control that half-ran is indistinguishable
    on disk from one that ran clean, and the surviving arm gets scored.

  * Verdicts are three-valued: SURVIVES / FAILS / INCONCLUSIVE. With test
    folds this small, INCONCLUSIVE is the honest answer most of the time and
    must never be rounded up to SURVIVES.

Usage:
    # the audit regression suite (selection, provenance, clustering), ~1 second
    python pipeline/s05_controls.py --self-test

    # full smoke test of every control, no drive access, ~1 minute
    python pipeline/s05_controls.py --dry-run

    # real suite for one cohort
    python pipeline/s05_controls.py --cohort prostate_dwi --controls all

    # just the one that matters, on both cohorts
    python pipeline/s05_controls.py --cohort breast --controls background_only

    # re-render the verdict table from JSON already on disk
    python pipeline/s05_controls.py --cohort prostate_dwi --summary-only
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import json
import logging
import math
import os
import shutil
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import common  # noqa: E402
import s03_train  # noqa: E402

logger = logging.getLogger("s05_controls")

CONTROLS = (
    "label_permutation",
    "background_only",
    "phase_scramble",
    "acquisition_split",
    "confound_predictability",
)

# Which input conditions each control is worth running by default.
#
# 'phase' is the claim under test, so it is in every list. 'magnitude' is
# included where it is a meaningful comparator: background-only magnitude says
# how much scanner signature the magnitude channel carries too, and a
# confound-predictability run on magnitude is the baseline against which the
# phase number must be read. 'magnitude' is deliberately absent from
# phase_scramble, where it is a literal no-op.
DEFAULT_CONDITIONS = {
    "label_permutation": ("phase",),
    "background_only": ("magnitude", "phase"),
    "phase_scramble": ("phase", "both"),
    "acquisition_split": ("magnitude", "phase"),
    "confound_predictability": ("magnitude", "phase"),
}

# Logical confound name -> candidate column names in the cache index, in
# preference order. The cache index and the stage-1 cohort CSV do not use the
# same spellings (the cohort CSV says receiver_channels, the prostate cache
# index says n_coils), so resolve rather than hardcode.
CONFOUND_ALIASES = {
    "institution": ("institution", "device_id"),
    "receiver_channels": ("receiver_channels", "n_coils"),
    "device_id": ("device_id",),
    "folder": ("folder",),
    "source_dir": ("source_dir", "folder"),
    "protocol": ("protocol_name", "protocol"),
    "patient_position": ("patient_position",),
    "field_strength": ("field_strength_T", "field_strength"),
    "scanner_model": ("scanner_model", "scanner"),
    "tr": ("TR", "tr"),
    "te": ("TE", "te"),
    "reason": ("reason",),
    "repeated_scan": ("repeated_scan", "repeat_group"),
    "acq": ("acq",),
    "n_spokes": ("n_spokes",),
    "dcf": ("dcf",),
}

# Per-cohort defaults. Chosen from what stage 1 actually measured on this
# drive: prostate DWI spans 2 institutions x 6 channel counts; breast is split
# across two directory trees and stage 1 already flagged 'reason' and
# 'source_dir' as label-associated at patient level (p=1.4e-17 and p=8.9e-06).
# Per-cohort defaults, chosen from what stage 1 and stage 2 actually put on
# disk rather than from what the datasets nominally contain:
#
#   prostate: 'institution' is a clean two-arm split (38 vs 7 subjects) and is
#       associated with the label -- Bay Ridge is 8.4% positive slices, Langone
#       2.7% -- so it is exactly the confound that can manufacture an AUC.
#       'receiver_channels' takes six values across the same 45 subjects.
#   breast: 'source_dir' is NOT usable even though stage 1 flagged it, because
#       stage 2 cached only the breast_updated tree, so the column is constant
#       in the cache; 'folder' (batches of ten patients) is the surviving
#       acquisition-batch proxy. 'acq' is the cleanest scanner-identity target
#       in the whole study: it is the 1st vs 2nd acquisition of the SAME woman,
#       perfectly balanced, and carries no diagnostic information whatsoever by
#       construction, so any AUC above 0.5 there is pure acquisition signature.
#       'reason' is deliberately NOT a default: it is 100% negative for values
#       0/1 and 96% positive for value 2, so predicting it is nearly the same
#       task as predicting the diagnosis and the number would not be
#       interpretable as scanner identity. It remains available via --confounds.
DEFAULT_STRAT_KEY = {
    "prostate_dwi": "institution",
    "prostate_t2": "institution",
    "breast": "folder",
    "dryrun": "institution",
}
DEFAULT_CONFOUNDS = {
    "prostate_dwi": ("institution", "receiver_channels"),
    "prostate_t2": ("institution", "receiver_channels"),
    "breast": ("folder", "acq"),
    "dryrun": ("institution",),
}

# Pre-registered decision rules. These are conventions, not laws; they are
# printed with the verdict table so a reader can disagree with them explicitly
# rather than having to reverse-engineer them from the numbers.
DECISION = {
    # Background-only: FAIL if the lower bound of the patient-clustered CI is
    # above chance, i.e. the model classifies tumour from air at a level the
    # data can actually distinguish from 0.5.
    "background_auc_alarm": 0.50,
    # Confound predictability: this is a severity scale, not a test. A lower
    # CI bound above this means phase encodes scanner identity strongly enough
    # that any phase-based diagnostic claim on this dataset is confounded
    # until proven otherwise.
    "confound_auc_alarm": 0.70,
    # Permutation test significance level.
    "perm_alpha": 0.05,
}

_EXCLUDE_FROM_JOIN = {
    "idx", "slice", "slice_1based", "label", "raw_label", "official_split",
    "patient_id", "file", "cohort",
}


# ===========================================================================
# Index loading, subject resolution, metadata enrichment
# ===========================================================================

def _basename_series(s: pd.Series) -> pd.Series:
    return s.astype(str).map(lambda p: os.path.basename(p.strip()))


def enrich_index(index: pd.DataFrame, cohort: str, cohort_dir: Path) -> pd.DataFrame:
    """
    Join stage-1 cohort metadata onto the stage-2 cache index by file basename.

    The cache index is deliberately lean -- stage 2 writes only what the
    reconstruction needed. The confound columns this stage lives on
    (subject_id, institution, receiver_channels, source_dir) live in the
    stage-1 cohort CSV. The two tables agree on file basename and nothing else:
    the cohort CSV stores an absolute /Volumes path, and their patient_id
    columns are in different namespaces (cohort says 'prostate_0001', the
    prostate cache index says 1), so joining on patient_id would silently
    produce garbage. Basename it is.

    Only columns absent from the index are brought over, and row-level columns
    (label, slice, official_split, ...) are never overwritten even if missing,
    because the cache index is authoritative for those.
    """
    out = index.copy()
    path = Path(cohort_dir) / f"{cohort}_cohort.csv"
    if not path.exists():
        logger.warning("no stage-1 cohort CSV at %s; confound columns limited to the cache index", path)
        return out

    coh = pd.read_csv(path, low_memory=False)
    if "file" not in coh.columns:
        logger.warning("%s has no 'file' column; skipping enrichment", path)
        return out

    coh = coh.copy()
    coh["_bn"] = _basename_series(coh["file"])
    bring = [
        c for c in coh.columns
        if c not in out.columns and c not in _EXCLUDE_FROM_JOIN and not c.startswith("_")
    ]
    if not bring:
        return out

    # File-level metadata only: collapse the per-slice cohort rows to one row
    # per file. Columns that vary within a file would be meaningless here.
    per_file = coh.drop_duplicates("_bn")[["_bn"] + bring]
    out["_bn"] = _basename_series(out["file"])
    merged = out.merge(per_file, on="_bn", how="left", validate="many_to_one")
    n_unmatched = int(merged["_bn"].isin(set(per_file["_bn"])).eq(False).sum())
    merged = merged.drop(columns=["_bn"])
    logger.info(
        "enriched cache index with %d stage-1 column(s) (%d/%d rows matched a cohort file)",
        len(bring), len(merged) - n_unmatched, len(merged),
    )
    return merged


def resolve_subject_col(index: pd.DataFrame) -> str:
    """
    Return the column to use as the split-enforcement unit.

    subject_id is the correct unit and is what stage 1 emits: for breast it
    collapses the two acquisitions of a patient AND the repeated-scan groups
    that link different coded patient names into one entity. patient_id does
    not do the second of those, so a breast split enforced on patient_id can
    put the same physical woman on both sides of the train/test line under two
    different codes. If subject_id is genuinely unavailable we fall back, but
    loudly -- a silent fallback here is a published-and-retracted-later bug.
    """
    if "subject_id" in index.columns and index["subject_id"].notna().all():
        return "subject_id"
    logger.warning(
        "SUBJECT ID UNAVAILABLE: falling back to patient_id as the split unit. "
        "For breast this does NOT collapse repeated-scan groups across coded "
        "names; verify manually before trusting any split-based control."
    )
    return "patient_id"


def assert_no_group_leakage(df: pd.DataFrame, group_col: str, what: str = "index") -> None:
    """Hard failure if any group spans more than one split. Mirrors s03_train."""
    per = df.groupby(group_col)["official_split"].nunique()
    bad = per[per > 1]
    if len(bad):
        detail = "\n".join(
            f"    {g!r}: {sorted(df.loc[df[group_col] == g, 'official_split'].unique())}"
            for g in list(bad.index)[:10]
        )
        raise RuntimeError(
            f"{what.upper()} LEAKAGE: {len(bad)} {group_col}(s) appear in more than one split.\n"
            + detail + "\nRefusing to run this control."
        )
    logger.info("  %s leakage check on %s: OK (%d groups)", what, group_col, df[group_col].nunique())


def load_cohort(cohort: str, args) -> tuple[pd.DataFrame, Path, str]:
    """Read the stage-2 cache index, enrich it, prepare splits, return (index, h5, subject_col)."""
    cache_dir = Path(args.cache_dir)
    h5_path = cache_dir / f"{cohort}.h5"
    idx_path = cache_dir / f"{cohort}_index.csv"
    for p in (h5_path, idx_path):
        if not p.exists():
            raise FileNotFoundError(f"missing {p}; run stage 2 for cohort {cohort!r} first")

    raw = pd.read_csv(idx_path, low_memory=False)
    logger.info("loaded %s: %d rows", idx_path, len(raw))
    raw = enrich_index(raw, cohort, Path(args.cohort_dir))
    # The split column is a passthrough to s03_train.prepare_index, which
    # aliases `cv<k>_split` onto `official_split` at the single entry point.
    # Without it every control here would be measured on the OFFICIAL split
    # while the headline runs it is meant to falsify were measured on the
    # stage-1 CV folds -- a control evaluated on a different test set than the
    # claim is not a control, it is an unrelated experiment.
    split_col = getattr(args, "split_col", "official_split") or "official_split"
    logger.info("split column: %s", split_col)
    index = s03_train.prepare_index(raw, args.val_frac, args.val_split_seed,
                                    split_col=split_col)
    index = index.reset_index(drop=True)

    subject_col = resolve_subject_col(index)
    index[subject_col] = index[subject_col].astype(str)
    assert_no_group_leakage(index, subject_col, "subject")
    return index, h5_path, subject_col


def resolve_confound_column(index: pd.DataFrame, name: str) -> str:
    """Map a logical confound name onto a column that actually exists."""
    for cand in CONFOUND_ALIASES.get(name, (name,)):
        if cand in index.columns:
            return cand
    available = sorted(c for c in index.columns if c not in _EXCLUDE_FROM_JOIN)
    raise KeyError(
        f"confound {name!r} not available; tried {CONFOUND_ALIASES.get(name, (name,))}. "
        f"Columns present: {available}"
    )


# ===========================================================================
# Statistics: patient-clustered bootstrap
# ===========================================================================

def cluster_bootstrap_auc(labels, probs, groups, n_boot: int = 2000,
                          seed: int = 0, alpha: float = 0.05,
                          unit: str = "patient") -> dict:
    """
    AUC with a percentile bootstrap that resamples CLUSTERS, not slices.

    Slices from one patient are near-duplicates of each other. A slice-level
    bootstrap treats 122 correlated slices as 122 independent observations and
    returns an interval several times too narrow -- on the prostate DWI test
    fold, which is four patients, the effective sample size is four, not 122.
    Resampling whole clusters with replacement is the standard fix.

    `groups` must be the SPLIT-ENFORCEMENT unit (subject_id where available),
    not patient_id; `unit` only names it in the reported method string, so
    callers must not assume passing unit= changes the clustering. See
    subject_groups_for_split() for how that vector is built.

    With four clusters, three of them positive, many resamples come out
    single-class and have no defined AUC. Those are dropped and counted; the
    count is reported so the reader can see how thin the evidence is rather
    than being handed a confident-looking interval derived from 40 usable
    resamples.
    """
    from sklearn.metrics import roc_auc_score

    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    groups = np.asarray(groups).astype(str)
    out = {
        "auc": float("nan"), "lo": float("nan"), "hi": float("nan"),
        "n": int(len(labels)), "n_pos": int(labels.sum()),
        "n_clusters": int(len(np.unique(groups))) if len(groups) else 0,
        "n_boot_ok": 0, "n_boot_degenerate": 0,
        "method": f"{unit}-clustered percentile bootstrap, B={n_boot}, alpha={alpha}",
    }
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        return out

    out["auc"] = float(roc_auc_score(labels, probs))
    uniq, inv = np.unique(groups, return_inverse=True)
    by_group = [np.flatnonzero(inv == g) for g in range(len(uniq))]
    rng = np.random.default_rng(seed)

    boots = []
    degenerate = 0
    for _ in range(n_boot):
        pick = rng.integers(0, len(uniq), len(uniq))
        sel = np.concatenate([by_group[g] for g in pick])
        yl = labels[sel]
        if len(np.unique(yl)) < 2:
            degenerate += 1
            continue
        boots.append(roc_auc_score(yl, probs[sel]))

    out["n_boot_ok"] = len(boots)
    out["n_boot_degenerate"] = degenerate
    if len(boots) >= 20:
        lo, hi = np.percentile(boots, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out["lo"], out["hi"] = float(lo), float(hi)
    return out


def subject_groups_for_split(index: pd.DataFrame, subject_col: str,
                             split_block: dict, what: str = "split") -> np.ndarray:
    """
    The cluster vector for `split_block`, resolved to SUBJECTS, not patients.

    stage 3 packs `patient_ids` and `cache_idx` alongside every split's
    predictions. `patient_ids` is the wrong resampling unit for this module:
    resolve_subject_col() exists precisely because breast repeat-scan groups
    give one physical woman two coded patient names, and clustering on
    patient_id then treats two scans of the same woman as two independent
    draws. That narrows every control's interval, which is the direction that
    makes a control look like it survived.

    `cache_idx` is the cache index's own `idx` column (CacheSliceDataset sets
    self.indices = rows["idx"]), so it joins back to the cohort index
    unambiguously. If the join cannot be completed we RAISE rather than fall
    back to patient_ids: a silently patient-clustered interval in a file whose
    control_detail says subject_col=subject_id is exactly the kind of quiet
    mislabelling this module exists to prevent. The raise is caught by
    _guarded(), recorded in the failure ledger, and read downstream as MISSING.
    """
    idx = split_block.get("cache_idx")
    if idx is None:
        raise RuntimeError(
            f"{what}: stage-3 payload carries no 'cache_idx'; cannot resolve the "
            f"{subject_col!r} cluster for the bootstrap. Refusing to fall back to "
            "patient_id (see subject_groups_for_split)."
        )
    if subject_col not in index.columns:
        raise RuntimeError(
            f"{what}: cohort index has no {subject_col!r} column; cannot cluster the "
            "bootstrap on the split-enforcement unit."
        )
    if "idx" not in index.columns:
        raise RuntimeError(f"{what}: cohort index has no 'idx' column to join cache_idx on")

    lut = index.drop_duplicates("idx").set_index("idx")[subject_col]
    want = pd.Index([int(v) for v in idx])
    groups = lut.reindex(want)
    if groups.isna().any():
        missing = [int(v) for v in want[groups.isna().to_numpy()][:5]]
        raise RuntimeError(
            f"{what}: {int(groups.isna().sum())} of {len(want)} predicted row(s) have no "
            f"{subject_col!r} in the cohort index (cache_idx e.g. {missing}); refusing to "
            "cluster the bootstrap on an incomplete map."
        )
    return groups.astype(str).to_numpy()


def extra_test_metrics(labels, probs) -> dict:
    """Accuracy, balanced accuracy and the majority-class baseline at threshold 0.5."""
    labels = np.asarray(labels).astype(int)
    probs = np.asarray(probs, dtype=float)
    if len(labels) == 0:
        return {}
    pred = (probs >= 0.5).astype(int)
    per_class = []
    for c in (0, 1):
        m = labels == c
        per_class.append(float((pred[m] == c).mean()) if m.any() else float("nan"))
    maj = float(max(np.mean(labels == 0), np.mean(labels == 1)))
    return {
        "accuracy": float((pred == labels).mean()),
        "balanced_accuracy": float(np.nanmean(per_class)),
        "majority_baseline_accuracy": maj,
        "recall_neg": per_class[0],
        "recall_pos": per_class[1],
    }


def fmt_ci(auc, lo, hi, width: int = 22) -> str:
    """Never print a point estimate bare -- if the CI is undefined, say so."""
    if auc is None or (isinstance(auc, float) and math.isnan(auc)):
        return "n/a".rjust(width)
    if lo is None or (isinstance(lo, float) and math.isnan(lo)):
        return f"{auc:.3f} [CI undef]".rjust(width)
    return f"{auc:.3f} [{lo:.2f}-{hi:.2f}]".rjust(width)


# ===========================================================================
# Split construction helpers (subject-grouped, stratified)
# ===========================================================================

def balanced_bipartition(counts: pd.Series) -> tuple[list, list]:
    """
    Split a set of categorical values into two size-balanced arms.

    Largest value first, each goes to whichever arm is currently smaller. Used
    both to build acquisition arms and to binarise a many-valued confound.
    Balance matters in both places: a 1-vs-6 folder split gives an 86% majority
    baseline, against which almost any AUC looks impressive and almost no
    accuracy is interpretable.
    """
    a, b = [], []
    na = nb = 0
    for val, n in counts.sort_values(ascending=False).items():
        if na <= nb:
            a.append(val)
            na += int(n)
        else:
            b.append(val)
            nb += int(n)
    return a, b


def group_level_value(df: pd.DataFrame, group_col: str, value_col: str) -> pd.Series:
    """Collapse a column to one value per group (modal value; ties -> first)."""
    def _mode(s):
        m = s.dropna()
        if len(m) == 0:
            return np.nan
        vc = m.value_counts()
        return vc.index[0]
    return df.groupby(group_col)[value_col].agg(_mode)


def carve_validation_by_group(df: pd.DataFrame, group_col: str, val_frac: float,
                              seed: int, strat_col: str = "label") -> pd.DataFrame:
    """
    Move a stratified fraction of training GROUPS into validation.

    s03_train has an equivalent that keys on patient_id; this one keys on the
    subject column, which is the unit we actually enforce. The stratification
    is on group-level positivity so that a small validation set does not come
    out single-class, which would make val AUC undefined and silently disable
    early stopping.
    """
    train = df["official_split"] == "training"
    groups = df.loc[train, group_col].drop_duplicates().sort_values().to_numpy()
    glabel = group_level_value(df.loc[train], group_col, strat_col).reindex(groups)
    glabel = glabel.fillna(0).to_numpy()

    rng = np.random.default_rng(seed)
    chosen = []
    for cls in np.unique(glabel):
        pool = groups[glabel == cls]
        if len(pool) <= 1:
            continue
        n_val = max(1, int(round(val_frac * len(pool))))
        n_val = min(n_val, len(pool) - 1)  # never empty the training pool
        chosen.extend(rng.permutation(pool)[:n_val].tolist())

    out = df.copy()
    out.loc[out[group_col].isin(chosen), "official_split"] = "validation"
    return out


def stratified_group_split(df: pd.DataFrame, group_col: str, strat_col: str,
                           fracs=(0.6, 0.2, 0.2), seed: int = 0) -> pd.DataFrame:
    """
    Fresh subject-grouped train/val/test split, stratified on a group-level column.

    Used by the confound-predictability control, where the official split is
    not usable: it was built to balance the DIAGNOSTIC label, so there is no
    guarantee that both institutions appear in the four-patient test fold, and
    a single-institution test fold makes the confound AUC undefined. Stratify
    on the confound instead, keep the grouping at subject level, and say so in
    the emitted JSON.
    """
    groups = df[group_col].drop_duplicates().sort_values().to_numpy()
    gval = group_level_value(df, group_col, strat_col).reindex(groups)
    rng = np.random.default_rng(seed)

    assign = {}
    for cls in pd.unique(gval.dropna()):
        pool = groups[(gval == cls).to_numpy()]
        pool = rng.permutation(pool)
        n = len(pool)
        n_tr = max(1, int(round(fracs[0] * n)))
        n_va = max(1, int(round(fracs[1] * n))) if n >= 3 else 0
        n_tr = min(n_tr, max(1, n - n_va - (1 if n >= 3 else 0)))
        for g in pool[:n_tr]:
            assign[g] = "training"
        for g in pool[n_tr:n_tr + n_va]:
            assign[g] = "validation"
        for g in pool[n_tr + n_va:]:
            assign[g] = "test"

    out = df.copy()
    out["official_split"] = out[group_col].map(assign)
    missing = int(out["official_split"].isna().sum())
    if missing:
        raise RuntimeError(f"stratified split left {missing} rows unassigned (null {strat_col}?)")
    return out


def describe_splits(df: pd.DataFrame, group_col: str) -> dict:
    return {
        s: {
            "n_slices": int((df["official_split"] == s).sum()),
            "n_groups": int(df.loc[df["official_split"] == s, group_col].nunique()),
            "n_pos": int(df.loc[df["official_split"] == s, "label"].sum()),
        }
        for s in s03_train.SPLITS
    }


def assert_trainable(df: pd.DataFrame, what: str) -> None:
    """Every split must be non-empty, and train/test must both be two-class."""
    for s in s03_train.SPLITS:
        sub = df[df["official_split"] == s]
        if len(sub) == 0:
            raise RuntimeError(f"{what}: the {s!r} split is empty")
    for s in ("training", "test"):
        sub = df[df["official_split"] == s]
        if sub["label"].nunique() < 2:
            raise RuntimeError(
                f"{what}: the {s!r} split is single-class "
                f"(n={len(sub)}, n_pos={int(sub['label'].sum())}); AUC is undefined"
            )


# ===========================================================================
# Control 1 primitive: subject-level label permutation
# ===========================================================================

def permute_labels_by_subject(df: pd.DataFrame, subject_col: str, splits, seed: int,
                              max_tries: int = 500) -> tuple[pd.DataFrame, dict]:
    """
    Permute labels among SUBJECTS within the given splits, preserving balance.

    The permutation is over whole label BLOCKS. Each subject hands its ordered
    vector of slice labels to another subject; where slice counts differ, the
    donor vector is index-mapped onto the recipient's slices. Consequences:

      * subject-level positive count is preserved exactly (a permutation
        cannot change a multiset), so the null has the same subject-level
        prevalence as the real data;
      * slice-level prevalence is preserved exactly when subjects have equal
        slice counts and to within the block-size ratio otherwise -- the
        realised value is reported in control_detail so it is auditable;
      * slices of one subject never receive conflicting labels, which is the
        failure mode of naive slice-level shuffling;
      * the association between image content and label is destroyed, which is
        the entire point.

    One consequence is deliberate and must be understood before reading the
    null. Because the donor's vector is index-mapped onto the recipient's
    slices, the WITHIN-SUBJECT position of the positive block is preserved:
    positives stay mid-gland. Measured on prostate DWI, the mean normalised
    distance of a positive slice from mid-stack is 0.088 in the real labels and
    0.089 after permutation. So "predict the middle slices" remains a winning
    strategy under the null, and the null does NOT centre on 0.5 -- on the
    four-patient prostate test fold a pure slice-position prior scores 0.86 on
    its own. That makes this a STRICTER null than a position-destroying
    shuffle would be, which is the right choice: a headline that only
    rediscovers "tumours are in the middle of the stack" should not be scored
    as a phase result. The realised null mean is reported next to the headline
    so the actual bar is visible rather than assumed to be 0.5.

    Requires at least two subjects of differing class and rejects the identity
    permutation -- a "null" run that happened to reproduce the true labels
    would contaminate the null distribution with a real-signal draw.
    """
    out = df.copy()
    rng = np.random.default_rng(seed)
    detail = {}

    for split in splits:
        mask = out["official_split"] == split
        sub = out.loc[mask].sort_values("idx")
        if len(sub) == 0:
            raise RuntimeError(f"cannot permute {split!r}: split is empty")

        subjects = list(pd.unique(sub[subject_col].astype(str)))
        vectors = {
            s: sub.loc[sub[subject_col].astype(str) == s, "label"].to_numpy(dtype=int)
            for s in subjects
        }
        subj_pos = {s: int(v.max()) for s, v in vectors.items()}
        if len(set(subj_pos.values())) < 2:
            raise RuntimeError(
                f"cannot permute {split!r}: all {len(subjects)} subjects are the same class"
            )

        newmap = None
        for _ in range(max_tries):
            perm = rng.permutation(len(subjects))
            cand = {subjects[i]: subjects[perm[i]] for i in range(len(subjects))}
            if any(subj_pos[cand[s]] != subj_pos[s] for s in subjects):
                newmap = cand
                break
        if newmap is None:
            raise RuntimeError(f"could not draw a non-identity permutation for {split!r}")

        new_col = out["label"].copy()
        for s in subjects:
            donor = vectors[newmap[s]]
            n_r = len(vectors[s])
            idx_map = np.minimum((np.arange(n_r) * len(donor)) // n_r, len(donor) - 1)
            rows = sub.index[sub[subject_col].astype(str) == s]
            new_col.loc[rows] = donor[idx_map]
        before = out.loc[mask, "label"].to_numpy(dtype=int)
        out.loc[mask, "label"] = new_col.loc[mask].to_numpy(dtype=int)
        after = out.loc[mask, "label"].to_numpy(dtype=int)

        detail[split] = {
            "n_subjects": len(subjects),
            "n_subjects_class_changed": int(sum(subj_pos[newmap[s]] != subj_pos[s] for s in subjects)),
            "subject_pos_before": int(sum(subj_pos.values())),
            "subject_pos_after": int(sum(subj_pos[newmap[s]] for s in subjects)),
            "slice_pos_frac_before": float(before.mean()),
            "slice_pos_frac_after": float(after.mean()),
            "frac_slice_labels_changed": float((before != after).mean()),
        }

    return out, detail


# ===========================================================================
# Control 3 primitive: within-mask phase scrambling
# ===========================================================================

class PhaseScramble:
    """
    Deterministic per-slice permutation of phase pixels inside the body mask.

    Deterministic matters. If the scramble were redrawn every epoch it would
    be a stochastic augmentation, and the network could average over draws to
    recover mask-conditional statistics -- a weaker, different control. Seeding
    on (base_seed, cache_row) means slice k has exactly one scramble for the
    whole run, so the model sees a fixed dataset in which phase has the same
    within-body marginal distribution as the real data and no spatial
    structure whatsoever.

    Background phase is left untouched: this control targets anatomy. The
    background is what control 2 interrogates.
    """

    def __init__(self, base_seed: int):
        self.base_seed = int(base_seed)

    def __call__(self, mag, phase, mask, k):
        sel = np.flatnonzero(np.asarray(mask).ravel())
        if sel.size < 2:
            return mag, phase, mask
        rng = np.random.default_rng([self.base_seed, int(k)])
        flat = np.asarray(phase, dtype=np.float32).ravel().copy()
        flat[sel] = flat[rng.permutation(sel)]
        return mag, flat.reshape(np.asarray(phase).shape), mask


class TransformedCacheDataset(s03_train.CacheSliceDataset):
    """s03_train's dataset with a hook between the cache read and the channel build."""

    TRANSFORM = None

    def _read(self, k):
        mag, phase, mask = super()._read(k)
        if self.TRANSFORM is not None:
            mag, phase, mask = self.TRANSFORM(mag, phase, mask, k)
        return (
            np.asarray(mag, dtype=np.float32),
            np.asarray(phase, dtype=np.float32),
            np.asarray(mask, dtype=bool),
        )


@contextlib.contextmanager
def dataset_transform(fn):
    """
    Temporarily route s03_train.run_one's dataset construction through a transform.

    run_one() resolves CacheSliceDataset from its own module globals at call
    time, so swapping the module attribute is enough and no training code has
    to be duplicated or parameterised. The swap is scoped and always restored.

    DataLoader workers must be 0 while this is active: the transform is held on
    the class, class attributes are not pickled with instances, and a forked
    worker would re-import the module and quietly get TRANSFORM=None -- i.e.
    the control would silently not happen. Callers force workers=0.
    """
    if fn is None:
        yield
        return
    original = s03_train.CacheSliceDataset
    TransformedCacheDataset.TRANSFORM = fn
    s03_train.CacheSliceDataset = TransformedCacheDataset
    try:
        yield
    finally:
        s03_train.CacheSliceDataset = original
        TransformedCacheDataset.TRANSFORM = None


# ===========================================================================
# Provenance: which controls were attempted, and which of them failed
# ===========================================================================

FAILURES_FILENAME = "control_failures.json"
FAILURES_SCHEMA = "phasedx.s05.control_failures.v1"


class FailureLedger:
    """
    Durable record of every control stage 5 ATTEMPTED and every one that broke.

    _guarded() deliberately does not re-raise: one control blowing up must not
    take the other four with it. The consequence, before this ledger existed,
    was that a control which failed and a control which was never requested
    looked identical on disk -- an empty result set -- so stage 6 scored
    whatever partial output survived. That is how an acquisition-split arm
    dying single-class turned a NOT SUPPORTED into a SUPPORTED: the surviving
    direction was the only one stage 6 could see.

    The file is rewritten after every mutation rather than once at exit, so a
    killed or OOM-ed run still leaves the record behind. Entries are keyed by
    (cohort, control): re-running one control with --controls resets only that
    pair, so a genuine later success clears an earlier failure instead of
    poisoning the tree forever, while pairs not attempted this time keep the
    state their own run left.
    """

    def __init__(self, path: Path):
        self.path = Path(path)
        self.entries: dict = {}
        prior = {}
        if self.path.exists():
            try:
                obj = json.loads(self.path.read_text())
                for e in (obj.get("entries") or []):
                    prior[(str(e.get("cohort")), str(e.get("control")))] = e
            except (OSError, ValueError) as exc:  # noqa: PERF203
                logger.warning("could not read existing %s (%s); starting a fresh ledger",
                               self.path, exc)
        self.entries = prior

    @staticmethod
    def _key(cohort: str, control: str) -> tuple:
        return (str(cohort), str(control))

    def begin(self, cohort: str, control: str) -> None:
        """Mark a (cohort, control) pair as attempted NOW, discarding prior state."""
        self.entries[self._key(cohort, control)] = {
            "cohort": str(cohort), "control": str(control),
            "attempted": True, "failed": False, "errors": [],
            "updated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        self.flush()

    def note(self, cohort: str, control: str, reason: str, **extra) -> None:
        """Record a deliberate, non-fatal skip. Does NOT set failed."""
        e = self.entries.get(self._key(cohort, control))
        if e is None:
            self.begin(cohort, control)
            e = self.entries[self._key(cohort, control)]
        rec = {"reason": str(reason)}
        rec.update({k: (str(v) if v is not None else None) for k, v in extra.items()})
        e.setdefault("skips", []).append(rec)
        e["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        self.flush()

    def fail(self, cohort: str, control: str, error: str, **extra) -> dict:
        e = self.entries.get(self._key(cohort, control))
        if e is None:
            self.begin(cohort, control)
            e = self.entries[self._key(cohort, control)]
        e["failed"] = True
        e["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        rec = {"error": str(error)}
        rec.update({k: (str(v) if v is not None else None) for k, v in extra.items()})
        e["errors"].append(rec)
        self.flush()
        return rec

    def flush(self) -> Path:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema": FAILURES_SCHEMA,
            "written": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "note": ("Controls stage 5 attempted. Any entry with failed=true did NOT "
                     "produce trustworthy output; stage 6 must report the corresponding "
                     "criterion MISSING rather than scoring whatever partial runs "
                     "landed on disk."),
            "entries": [self.entries[k] for k in sorted(self.entries)],
        }
        tmp = self.path.with_suffix(".json.tmp")
        tmp.write_text(json.dumps(payload, indent=2))
        tmp.replace(self.path)
        return self.path

    @property
    def failed(self) -> list:
        return [e for e in self.entries.values() if e.get("failed")]


# ===========================================================================
# Run wrapper: call s03_train.run_one, tag the JSON, file it
# ===========================================================================

@dataclass
class ControlContext:
    cohort: str
    index: pd.DataFrame
    subject_col: str
    device: torch.device
    args: argparse.Namespace
    train_args: argparse.Namespace
    results_root: Path
    ckpt_root: Path
    h5_path: Path | None = None
    arrays: dict | None = None
    # Resolved per cohort, not on args: a multi-cohort run must not inherit the
    # previous cohort's acquisition key (prostate has no 'source_dir', breast
    # has no 'institution').
    strat_key: str = "institution"
    confounds: tuple = ("institution",)
    failures: list = field(default_factory=list)
    # Set by main(); every append to .failures also lands here, on disk, where
    # stage 6 can see it. Optional so unit tests can build a bare context.
    ledger: FailureLedger | None = None

    def record_failure(self, control: str, error, **extra) -> dict:
        """
        Record a swallowed failure BOTH in memory and on disk.

        Every ctx.failures.append in this module goes through here. An
        in-memory-only list dies with the process and is invisible to stage 6,
        which is the whole of defect [3].
        """
        rec = {"cohort": self.cohort, "control": str(control), "error": str(error)}
        rec.update(extra)
        self.failures.append(rec)
        if self.ledger is not None:
            self.ledger.fail(self.cohort, control, error, **extra)
        return rec


def split_bootstrap(ctx: ControlContext, blk: dict, what: str = "split") -> dict:
    """
    The clustered bootstrap block s06 reads for C3/C4/C5/C7.

    Separated from run_tagged so the clustering unit can be regression-tested
    without a training run: this is the one place a control's interval is
    computed, and it clusters on ctx.subject_col, NOT on the payload's
    patient_ids. patient_id does not collapse breast repeat-scan groups, so
    clustering on it treats two scans of one woman as two independent draws and
    returns an interval that is too narrow -- exactly the direction that makes
    a falsification control look like it survived.

    `n_patient_clusters` is kept alongside so a reader can see how much the two
    units differ for this fold without re-deriving it.
    """
    groups = subject_groups_for_split(ctx.index, ctx.subject_col, blk, what=what)
    boot = cluster_bootstrap_auc(
        blk["labels"], blk["probs"], groups,
        n_boot=ctx.args.n_boot, seed=ctx.args.bootstrap_seed, unit=ctx.subject_col,
    )
    boot["cluster_unit"] = ctx.subject_col
    boot["n_patient_clusters"] = int(len({str(p) for p in blk["patient_ids"]}))
    # Persist the resolved vector, aligned with the split's own row order, so
    # anything reading these JSONs later (this module's own summary table,
    # stage 6, a reviewer) can re-cluster on the right unit without needing the
    # cohort index in scope. Without it the only cluster id in the file is
    # patient_id, and the correct unit is unrecoverable from the file alone.
    boot["cluster_ids"] = [str(g) for g in groups]
    return boot


def _stem(cohort, control, variant, condition, seed) -> str:
    parts = [str(cohort), str(control)]
    if variant:
        parts.append(str(variant))
    parts += [str(condition), f"seed{seed}"]
    return "__".join(parts)


def run_tagged(ctx: ControlContext, *, control: str, condition: str, seed: int,
               index: pd.DataFrame, variant: str = "", region: str = "full",
               detail: dict | None = None, transform=None,
               label_semantics: str = "diagnosis") -> dict:
    """
    Run one control training run and emit s03_train's JSON schema plus 'control'.

    The output file is renamed to {cohort}__{control}__{variant}__{condition}__
    seed{N}.json so that controls sharing a (cohort, condition, seed) triple do
    not overwrite each other, but the CONTENTS are byte-for-byte the schema
    s03_train.run_one writes, with two keys added at the top level:

        control        : the control name, or "none" for a headline run
        control_detail : bookkeeping specific to this control, including the
                         patient-clustered bootstrap CI for the test AUC and
                         (where the target is not the diagnosis) an explicit
                         label_semantics string so nobody reads a scanner-ID
                         AUC as a cancer AUC.

    s04_stats.py can therefore glob both trees and branch on
    payload.get("control", "none") with no other changes.
    """
    run_args = copy.deepcopy(ctx.train_args)
    run_args.region = region
    run_args.control = control  # lands in the checkpoint's saved args
    if transform is not None and run_args.workers:
        logger.warning("forcing --workers 0: %s uses an in-process cache transform", control)
        run_args.workers = 0

    scratch = ctx.results_root / "_scratch"
    if scratch.exists():
        shutil.rmtree(scratch)
    scratch.mkdir(parents=True, exist_ok=True)
    run_args.results_dir = str(scratch)
    run_args.ckpt_dir = str(scratch)

    t0 = time.time()
    with dataset_transform(transform):
        res = s03_train.run_one(
            ctx.cohort, condition, seed, index, ctx.device, run_args,
            h5_path=ctx.h5_path, arrays=ctx.arrays,
        )

    payload = json.loads(Path(res["results_json"]).read_text())
    stem = _stem(ctx.cohort, control, variant, condition, seed)

    det = dict(detail or {})
    det.update({
        "variant": variant or None,
        "label_semantics": label_semantics,
        "subject_col": ctx.subject_col,
        # Which split the headline this control falsifies was measured on.
        # Recorded per run so a controls tree cannot be silently compared
        # against a headline computed on a different test set.
        "split_col": getattr(ctx.train_args, "split_col", "official_split"),
        "transform": type(transform).__name__ if transform is not None else None,
        "wall_seconds": time.time() - t0,
    })
    for split in ("val", "test"):
        blk = payload.get(split)
        if not blk:
            continue
        det[f"{split}_auc_ci95"] = split_bootstrap(ctx, blk, what=f"{control} {split}")
        det[f"{split}_extra"] = extra_test_metrics(blk["labels"], blk["probs"])

    payload["control"] = control
    payload["control_detail"] = det

    if ctx.args.keep_checkpoints:
        ctx.ckpt_root.mkdir(parents=True, exist_ok=True)
        dst = ctx.ckpt_root / f"{stem}.pt"
        src = Path(payload["checkpoint"])
        if src.exists():
            shutil.move(str(src), dst)
            payload["checkpoint"] = str(dst)
    else:
        # 45 MB of ResNet-18 weights per control run adds up to gigabytes and
        # is never needed downstream; null it out rather than leave a dangling path.
        payload["checkpoint"] = None

    ctx.results_root.mkdir(parents=True, exist_ok=True)
    out_path = ctx.results_root / f"{stem}.json"
    out_path.write_text(json.dumps(payload, indent=2))
    shutil.rmtree(scratch, ignore_errors=True)

    ci = det.get("test_auc_ci95", {})
    logger.info(
        "  -> %s  test_auc=%s  %s",
        out_path.name, fmt_ci(ci.get("auc"), ci.get("lo"), ci.get("hi")).strip(), control,
    )
    return {
        "path": str(out_path), "control": control, "variant": variant,
        "condition": condition, "seed": seed,
        "test_auc": payload["test"]["auc"] if payload.get("test") else float("nan"),
    }


def _guarded(ctx: ControlContext, control: str, fn):
    """Run a control, recording (not raising) operational failures."""
    logger.info("=" * 78)
    logger.info("CONTROL: %s   cohort=%s", control.upper(), ctx.cohort)
    logger.info("=" * 78)
    try:
        return fn()
    except Exception as exc:  # noqa: BLE001
        logger.error("control %s FAILED TO RUN on %s: %s", control, ctx.cohort, exc)
        ctx.record_failure(control, exc)
        return []


# ===========================================================================
# The five controls
# ===========================================================================

def control_label_permutation(ctx: ControlContext, conditions) -> list:
    """
    Retrain on subject-level permuted training labels; build a null of test AUC.

    Test labels are never touched -- the null distribution is of the statistic
    we actually report. By default only the training split is permuted, which
    is what the classical permutation test does; epoch selection then still
    uses real validation labels, so best-epoch choice can pick up validation
    noise. That does not bias the TEST AUC (test is independent of the
    selection), but --permute-splits training,validation is available for
    anyone who wants the whole model-building procedure under H0.
    """
    splits = [s.strip() for s in ctx.args.permute_splits.split(",") if s.strip()]
    out = []
    for condition in conditions:
        aucs = []
        for rep in range(ctx.args.n_permutations):
            perm_seed = ctx.args.permutation_seed0 + rep
            permuted, detail = permute_labels_by_subject(
                ctx.index, ctx.subject_col, splits, perm_seed
            )
            assert_trainable(permuted, f"label_permutation rep{rep}")
            logger.info(
                "  perm rep %d/%d (seed %d): training slice prevalence %.3f -> %.3f, "
                "%.0f%% of slice labels moved",
                rep + 1, ctx.args.n_permutations, perm_seed,
                detail["training"]["slice_pos_frac_before"],
                detail["training"]["slice_pos_frac_after"],
                100 * detail["training"]["frac_slice_labels_changed"],
            )
            r = run_tagged(
                ctx, control="label_permutation", condition=condition,
                seed=ctx.args.seeds[0] + rep, index=permuted,
                variant=f"perm{rep:02d}",
                detail={
                    "permutation_seed": perm_seed,
                    "permuted_splits": splits,
                    "permutation_unit": ctx.subject_col,
                    "per_split": detail,
                },
            )
            aucs.append(r["test_auc"])
            out.append(r)
        finite = [a for a in aucs if not math.isnan(a)]
        if finite:
            logger.info(
                "  NULL DISTRIBUTION [%s]: n=%d  mean=%.3f  sd=%.3f  min=%.3f  max=%.3f",
                condition, len(finite), float(np.mean(finite)), float(np.std(finite, ddof=1))
                if len(finite) > 1 else 0.0, float(np.min(finite)), float(np.max(finite)),
            )
    return out


def control_background_only(ctx: ControlContext, conditions) -> list:
    """
    Train on air. Everything inside the body mask is zeroed.

    s03_train's dataset already implements region='background' (keep = ~mask),
    so this control is the headline pipeline with one argument changed -- which
    is exactly the property a control should have. What this module adds is the
    check that the mask is not degenerate: if the body mask covers essentially
    the whole frame there is no background left and a null result here would be
    vacuous rather than informative.
    """
    frac = None
    if "mask_frac" in ctx.index.columns:
        frac = float(pd.to_numeric(ctx.index["mask_frac"], errors="coerce").mean())
        logger.info("  mean body-mask fraction %.3f -> background is %.1f%% of each frame",
                    frac, 100 * (1 - frac))
        if frac > 0.95:
            raise RuntimeError(
                f"body mask covers {100*frac:.1f}% of the frame; there is no background "
                "to train on and this control would be vacuous"
            )
    detail = {"mean_mask_frac": frac,
              "note": "all pixels inside the body mask are zeroed; only air is seen"}
    return [
        run_tagged(ctx, control="background_only", condition=c, seed=s,
                   index=ctx.index, region="background", detail=detail)
        for c in conditions for s in ctx.args.seeds
    ]


def control_phase_scramble(ctx: ControlContext, conditions) -> list:
    """Permute phase pixels within the body mask; magnitude and background untouched."""
    conds = [c for c in conditions if c != "magnitude"]
    skipped = sorted(set(conditions) - set(conds))
    if skipped:
        logger.info("  skipping %s: scrambling phase is a no-op for a magnitude-only model",
                    ",".join(skipped))
    out = []
    for c in conds:
        for s in ctx.args.seeds:
            out.append(run_tagged(
                ctx, control="phase_scramble", condition=c, seed=s, index=ctx.index,
                transform=PhaseScramble(ctx.args.scramble_seed),
                detail={
                    "scramble_seed": ctx.args.scramble_seed,
                    "scope": "within body mask only",
                    "note": "within-mask phase marginal preserved exactly; "
                            "spatial structure destroyed; deterministic per cache row",
                },
            ))
    return out


def control_acquisition_split(ctx: ControlContext, conditions) -> list:
    """
    Re-split so acquisition metadata is held out instead of shared.

    Subjects are bipartitioned by the stratification key (institution by
    default for prostate, source tree for breast) into two arms; one arm
    trains, the other tests, validation is carved from the training arm by
    subject. Both directions are run by default, because an AUC drop seen in
    only one direction is usually an artefact of arm size or arm prevalence
    rather than evidence about transfer.

    An arm that is single-class on the diagnostic label makes the AUC
    undefined, so that direction is skipped with an explicit message instead of
    producing a NaN that later looks like a missing run.
    """
    key = resolve_confound_column(ctx.index, ctx.strat_key)
    df = ctx.index[ctx.index[key].notna()].copy()
    dropped = len(ctx.index) - len(df)
    if dropped:
        logger.info("  dropped %d row(s) with null %s", dropped, key)

    gkey = group_level_value(df, ctx.subject_col, key)
    values = gkey.value_counts()
    logger.info("  %s at subject level: %s", key, values.to_dict())
    if len(values) < 2:
        raise RuntimeError(f"{key!r} takes only one value in this cohort; no arms to split on")

    a, b = balanced_bipartition(values)
    arms = {"A": a, "B": b}
    logger.info("  arm A = %s (%d subjects) | arm B = %s (%d subjects)",
                arms["A"], int(values[a].sum()), arms["B"], int(values[b].sum()))

    out = []
    directions = [("A", "B")] + ([("B", "A")] if ctx.args.both_directions else [])
    for train_arm, test_arm in directions:
        train_subj = set(gkey[gkey.isin(arms[train_arm])].index)
        test_subj = set(gkey[gkey.isin(arms[test_arm])].index)
        d = df.copy()
        d["official_split"] = np.where(d[ctx.subject_col].isin(train_subj), "training", "test")
        d = carve_validation_by_group(d, ctx.subject_col, ctx.train_args.val_frac,
                                      ctx.train_args.val_split_seed)
        variant = f"{train_arm}2{test_arm}"
        try:
            assert_trainable(d, f"acquisition_split {variant}")
        except RuntimeError as exc:
            # A skipped direction is NOT a benign log line: the criterion is
            # defined as the worse of both directions, so losing one silently
            # promotes the surviving arm to the whole answer. Record it where
            # stage 6 can read it and downgrade the criterion to MISSING.
            logger.warning("  skipping direction %s: %s", variant, exc)
            ctx.record_failure("acquisition_split", f"{variant}: {exc}", variant=variant)
            continue
        assert_no_group_leakage(d, ctx.subject_col, f"acquisition_split {variant}")

        detail = {
            "strat_key": key,
            "train_arm_values": [str(v) for v in arms[train_arm]],
            "test_arm_values": [str(v) for v in arms[test_arm]],
            "direction": variant,
            "n_subjects_train_arm": len(train_subj),
            "n_subjects_test_arm": len(test_subj),
            "splits_by_subject": describe_splits(d, ctx.subject_col),
            "rows_dropped_null_key": dropped,
        }
        logger.info("  direction %s: %s", variant, detail["splits_by_subject"])
        for c in conditions:
            for s in ctx.args.seeds:
                out.append(run_tagged(ctx, control="acquisition_split", condition=c,
                                      seed=s, index=d, variant=variant, detail=detail))
    return out


def control_confound_predictability(ctx: ControlContext, conditions) -> list:
    """
    How much scanner identity does the phase channel carry?

    The identical network is trained to predict a NON-diagnostic target. The
    diagnostic label is replaced wholesale, which is why the emitted JSON
    carries label_semantics="confound:<name>": every 'label', 'n_pos' and
    'auc' in that file refers to the confound, not to cancer, and reading them
    as cancer numbers would be a serious misinterpretation.

    Multi-valued targets are binarised because the shared head is 2-class:
    numeric columns split at the median, categorical columns become
    most-frequent-value versus the rest. The mapping and the resulting class
    balance are both recorded, and the majority-class baseline accuracy is
    computed so the AUC can be read against something.

    The split is rebuilt and stratified on the CONFOUND, not on the diagnosis.
    The official split was constructed for the diagnostic task and gives no
    guarantee that both institutions appear in a four-patient test fold; a
    single-institution test fold would make this AUC undefined.
    """
    out = []
    for name in ctx.confounds:
        col = resolve_confound_column(ctx.index, name)
        df = ctx.index[ctx.index[col].notna()].copy()
        if len(df) == 0:
            raise RuntimeError(f"confound {name!r} (column {col!r}) is entirely null")

        raw = df[col]
        numeric = pd.to_numeric(raw, errors="coerce")
        n_unique = raw.nunique()
        if n_unique < 2:
            # Benign, not a failure: a constant confound has nothing to predict.
            # Recorded as a skip so the absence of this target is explicable
            # later, but it does not mark the control failed.
            logger.warning("  confound %s is constant (%s); nothing to predict, skipping",
                           name, raw.iloc[0])
            if ctx.ledger is not None:
                ctx.ledger.note(ctx.cohort, "confound_predictability",
                                f"target {name!r} is constant ({raw.iloc[0]!r})",
                                variant=name)
            continue
        if n_unique == 2:
            pos_value = sorted(map(str, raw.unique()))[-1]
            new_label = (raw.astype(str) == pos_value).astype(int)
            rule = f"{col} == {pos_value!r}"
        elif numeric.notna().all():
            # Balance-optimal cut, not the median. receiver_channels is a
            # skewed discrete variable (752 slices at exactly 20 channels), so
            # '>= median' puts 93% of rows in one class and the resulting
            # accuracy is uninterpretable. Choose the threshold among the
            # observed values whose split is closest to 50/50 and name it.
            cuts = np.sort(numeric.unique())[1:]
            thr = float(min(cuts, key=lambda t: abs(float((numeric >= t).mean()) - 0.5)))
            new_label = (numeric >= thr).astype(int)
            rule = f"{col} >= {thr:g}  (balance-optimal cut of {n_unique} distinct values)"
        else:
            hi, _lo = balanced_bipartition(raw.astype(str).value_counts())
            new_label = raw.astype(str).isin(hi).astype(int)
            rule = f"{col} in {sorted(hi)} vs rest  (size-balanced bipartition)"

        diag_prev = float(df["label"].mean())
        df["label"] = new_label.to_numpy()
        balance = df["label"].value_counts().to_dict()
        logger.info("  target %s -> binary rule [%s]; slice balance %s (diagnostic "
                    "prevalence in the same rows was %.3f)", name, rule, balance, diag_prev)

        if ctx.args.confound_use_official_split:
            d = df
            split_note = "official split (unchanged)"
        else:
            d = stratified_group_split(df, ctx.subject_col, "label",
                                       fracs=tuple(ctx.args.confound_fracs),
                                       seed=ctx.args.confound_split_seed)
            split_note = (f"fresh subject-grouped split stratified on the confound, "
                          f"fracs={tuple(ctx.args.confound_fracs)}, "
                          f"seed={ctx.args.confound_split_seed}")
        assert_trainable(d, f"confound_predictability {name}")
        assert_no_group_leakage(d, ctx.subject_col, f"confound {name}")

        detail = {
            "target": name,
            "column": col,
            "binarisation_rule": rule,
            "n_distinct_values": int(n_unique),
            "value_counts": {str(k): int(v) for k, v in raw.value_counts().items()},
            "class_balance": {str(k): int(v) for k, v in balance.items()},
            "diagnostic_prevalence_in_same_rows": diag_prev,
            "split_note": split_note,
            "splits_by_subject": describe_splits(d, ctx.subject_col),
        }
        for c in conditions:
            for s in ctx.args.seeds:
                out.append(run_tagged(
                    ctx, control="confound_predictability", condition=c, seed=s,
                    index=d, variant=name, detail=detail,
                    label_semantics=f"confound:{name}",
                ))
    return out


CONTROL_FUNCS = {
    "label_permutation": control_label_permutation,
    "background_only": control_background_only,
    "phase_scramble": control_phase_scramble,
    "acquisition_split": control_acquisition_split,
    "confound_predictability": control_confound_predictability,
}


# ===========================================================================
# Summary / verdict table
# ===========================================================================

def load_results(dirs) -> list:
    """
    Load every run JSON under the given directories, control or headline.

    Deduplicated by resolved path. The control results dir is legitimately also
    a headline dir (--with-headline writes there), so the same file arrives
    twice; loading it twice would double every permutation replicate and make
    the null distribution look better sampled than it is.
    """
    payloads = []
    seen = set()
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.json")):
            rp = str(p.resolve())
            if p.parent.name == "_scratch" or rp in seen:
                continue
            seen.add(rp)
            try:
                obj = json.loads(p.read_text())
            except Exception:  # noqa: BLE001
                continue
            if "test" not in obj or "condition" not in obj:
                continue
            obj.setdefault("control", "none")
            obj["_path"] = rp
            payloads.append(obj)
    return payloads


def pool_predictions(payloads) -> pd.DataFrame:
    """
    Average each test slice's predicted probability across runs.

    Runs that differ only by training seed describe the same experiment, and
    averaging their per-slice probabilities is what a reader means by "the"
    result for a condition. Slices are keyed by (patient, cache row) so that
    runs testing on DISJOINT subjects -- the two directions of the
    acquisition-stratified split -- concatenate instead of colliding. Runs that
    differ by permutation replicate are never pooled here; those are the null
    distribution and are summarised separately.

    The `cluster` column is the SUBJECT-level vector run_tagged resolved and
    stored in control_detail (see split_bootstrap), not patient_id. It falls
    back to patient_ids only for payloads written before that field existed,
    and records which unit it ended up with in the frame's .attrs so the caller
    reports the truth rather than assuming.
    """
    frames = []
    fell_back = False
    for pl in payloads:
        t = pl["test"]
        ids = (((pl.get("control_detail") or {}).get("test_auc_ci95") or {})
               .get("cluster_ids"))
        if not ids or len(ids) != len(t["patient_ids"]):
            ids = t["patient_ids"]
            fell_back = True
        frames.append(pd.DataFrame({
            "key": [f"{a}|{b}" for a, b in zip(t["patient_ids"], t["cache_idx"])],
            "patient": t["patient_ids"], "cluster": [str(v) for v in ids],
            "label": t["labels"], "prob": t["probs"],
        }))
    allf = pd.concat(frames, ignore_index=True)
    out = allf.groupby("key").agg(patient=("patient", "first"),
                                  cluster=("cluster", "first"),
                                  label=("label", "first"),
                                  prob=("prob", "mean")).reset_index()
    out.attrs["cluster_unit"] = "patient_id (fallback)" if fell_back else "subject_id"
    return out


def _pooled_ci(payloads, n_boot, seed) -> dict:
    """Subject-clustered bootstrap CI over the pooled test predictions."""
    if not payloads:
        return {}
    agg = pool_predictions(payloads)
    unit = str(agg.attrs.get("cluster_unit", "patient_id (fallback)"))
    boot = cluster_bootstrap_auc(agg["label"], agg["prob"], agg["cluster"],
                                 n_boot=n_boot, seed=seed, unit=unit)
    boot["cluster_unit"] = unit
    boot["n_runs_pooled"] = len(payloads)
    boot["extra"] = extra_test_metrics(agg["label"], agg["prob"])
    return boot


def _mark(state: str) -> str:
    return {"survives": "ok", "fails": "FAIL", "inconclusive": "?",
            "flagged": "HIGH", "absent": "-"}[state]


def summarize(results_dirs, headline_dirs, n_boot: int = 2000, seed: int = 0,
              cohorts=None) -> str:
    """
    Render the verdict table: per cohort and condition, does the headline
    phase result survive every control?

    Verdicts are three-valued on purpose.

        SURVIVES      every control that ran came back clean
        FALSIFIED     at least one control fired
        NOT ESTABLISHED
                      no control fired, but at least one was too underpowered
                      to fire even if it should have. This is the expected
                      answer for a four-patient test fold and is NOT a
                      positive result.

    Decision rules are printed above the table so a reader can substitute
    their own.
    """
    payloads = load_results(list(results_dirs) + list(headline_dirs))
    if not payloads:
        return "no result JSON found; nothing to summarise\n"

    lines = []
    W = 100

    def head(title):
        lines.append("=" * W)
        lines.append(title)
        lines.append("=" * W)

    head("PhaseDx STAGE 5 -- FALSIFICATION SUITE")
    lines.append(f"runs loaded            : {len(payloads)}")
    by_control = pd.Series([p["control"] for p in payloads]).value_counts().to_dict()
    lines.append(f"runs by control        : {by_control}")
    lines.append(f"AUC intervals          : patient-clustered percentile bootstrap, B={n_boot}")
    lines.append("decision rules (pre-registered, editable):")
    lines.append(f"  background-only FAILS if lower CI > {DECISION['background_auc_alarm']:.2f}"
                 "   (model classifies tumour from air alone)")
    lines.append(f"  permutation     FAILS if headline <= null mean, or if p > "
                 f"{DECISION['perm_alpha']:.2f} with >=20 replicates")
    lines.append("  phase-scramble  FAILS if the scrambled CI overlaps the headline CI upward")
    lines.append("  acq-split       FAILS if cross-acquisition CI includes 0.5 while the "
                 "standard split's excludes it")
    lines.append(f"  confound        HIGH  if lower CI > {DECISION['confound_auc_alarm']:.2f}"
                 "   (phase encodes scanner identity; annotation, not a test)")
    lines.append("  anything the data cannot separate is reported as '?', never as 'ok'")
    lines.append("")

    cohort_list = cohorts or sorted({p["cohort"] for p in payloads})
    rows = []
    bg_rows = []
    conf_rows = []

    for cohort in cohort_list:
        pc = [p for p in payloads if p["cohort"] == cohort]
        conditions = sorted({p["condition"] for p in pc if p["control"] == "none"}) or \
            sorted({p["condition"] for p in pc})
        for condition in conditions:
            sel = [p for p in pc if p["condition"] == condition]
            headline = [p for p in sel if p["control"] == "none"]
            hb = _pooled_ci(headline, n_boot, seed) if headline else {}

            state = {}

            # --- 1. label permutation -> null distribution + p-value ---------
            perm = [p for p in sel if p["control"] == "label_permutation"]
            perm_aucs = [p["test"]["auc"] for p in perm
                         if p["test"] and not math.isnan(p["test"]["auc"])]
            perm_txt, perm_p = "-", None
            if perm_aucs:
                mu, sd = float(np.mean(perm_aucs)), (
                    float(np.std(perm_aucs, ddof=1)) if len(perm_aucs) > 1 else float("nan"))
                perm_txt = (f"null {mu:.3f}+-{sd:.3f} "
                            f"[{min(perm_aucs):.2f}-{max(perm_aucs):.2f}] n={len(perm_aucs)}")
                if hb.get("auc") is not None and not math.isnan(hb.get("auc", float("nan"))):
                    ge = sum(1 for a in perm_aucs if a >= hb["auc"])
                    perm_p = (1 + ge) / (1 + len(perm_aucs))
                    perm_txt += f"  p={perm_p:.3f}"
                    if hb["auc"] <= mu:
                        # The headline does not even reach the centre of its
                        # own null. No number of extra replicates rescues that,
                        # so this is a FAIL regardless of how few were run.
                        state["perm"] = "fails"
                        perm_txt += "  <- HEADLINE IS AT OR BELOW THE NULL MEAN"
                    elif len(perm_aucs) < 20:
                        # A null built from <20 replicates has a p-value floor
                        # of 1/(1+n) > 0.05 and cannot establish significance.
                        state["perm"] = "inconclusive"
                        perm_txt += " (n<20: p floor too high to establish)"
                    else:
                        state["perm"] = "survives" if perm_p <= DECISION["perm_alpha"] else "fails"
                else:
                    state["perm"] = "inconclusive"
                if abs(mu - 0.5) > 0.1:
                    # A null that is not centred on chance is informative, not
                    # broken: it says the test fold can be partly predicted
                    # without ever using the training labels, and it tells the
                    # reader what bar the headline actually has to clear.
                    perm_txt += (
                        f"\n      !! the null is centred at {mu:.3f}, not 0.5. The headline must "
                        f"beat {mu:.3f}, not chance.\n"
                        "         (a) the block permutation preserves each subject's WITHIN-SUBJECT\n"
                        "             position of positive slices, so 'predict the mid-gland slices'\n"
                        "             survives into the null. That is deliberate: it makes the null\n"
                        "             strict. On prostate DWI a pure slice-position prior already\n"
                        "             scores ~0.86 on the four-patient test fold.\n"
                        "         (b) epoch selection on real validation labels can also leak; rule\n"
                        "             it out with --permute-splits training,validation")
            else:
                state["perm"] = "absent"

            # --- 2. background only -----------------------------------------
            bg = [p for p in sel if p["control"] == "background_only"]
            bgb = _pooled_ci(bg, n_boot, seed) if bg else {}
            if bgb.get("n_boot_ok", 0) >= 20 and not math.isnan(bgb.get("lo", float("nan"))):
                state["bg"] = "fails" if bgb["lo"] > DECISION["background_auc_alarm"] else "survives"
            elif bg:
                state["bg"] = "inconclusive"
            else:
                state["bg"] = "absent"
            if bg:
                bg_rows.append((cohort, condition, bgb, state["bg"]))

            # --- 3. phase scramble ------------------------------------------
            scr = [p for p in sel if p["control"] == "phase_scramble"]
            scb = _pooled_ci(scr, n_boot, seed) if scr else {}
            if scr and hb and not math.isnan(hb.get("lo", float("nan"))) \
                    and not math.isnan(scb.get("hi", float("nan"))):
                if scb["hi"] < hb["lo"]:
                    # Destroying spatial structure destroyed the result: the
                    # effect really was spatial.
                    state["scr"] = "survives"
                elif scb["lo"] >= hb["auc"]:
                    # The scrambled interval sits entirely at or above the
                    # headline point estimate. Spatial phase structure is
                    # contributing nothing; whatever the model uses is a
                    # summary statistic of the phase histogram, which is what a
                    # per-scanner offset looks like.
                    state["scr"] = "fails"
                else:
                    state["scr"] = "inconclusive"
            elif scr:
                state["scr"] = "inconclusive"
            else:
                state["scr"] = "absent"

            # --- 4. acquisition-stratified split ----------------------------
            acq = [p for p in sel if p["control"] == "acquisition_split"]
            acb = _pooled_ci(acq, n_boot, seed) if acq else {}
            if acq and not math.isnan(acb.get("lo", float("nan"))):
                if acb["lo"] > 0.5:
                    state["acq"] = "survives"
                elif hb and not math.isnan(hb.get("lo", float("nan"))) and hb["lo"] > 0.5:
                    state["acq"] = "fails"
                else:
                    state["acq"] = "inconclusive"
            elif acq:
                state["acq"] = "inconclusive"
            else:
                state["acq"] = "absent"

            # --- 5. confound predictability ---------------------------------
            conf = [p for p in sel if p["control"] == "confound_predictability"]
            conf_state = "absent"
            for target in sorted({p["control_detail"].get("target", "?") for p in conf}):
                tp = [p for p in conf if p["control_detail"].get("target") == target]
                cb = _pooled_ci(tp, n_boot, seed)
                extra = cb.get("extra", {})
                conf_rows.append((cohort, condition, target, cb, extra,
                                  tp[0]["control_detail"].get("binarisation_rule", "")))
                if not math.isnan(cb.get("lo", float("nan"))):
                    if cb["lo"] > DECISION["confound_auc_alarm"]:
                        conf_state = "flagged"
                    elif conf_state != "flagged":
                        conf_state = "survives"
                elif conf_state == "absent":
                    conf_state = "inconclusive"
            state["conf"] = conf_state

            # Confound predictability is a severity annotation, not a
            # falsification test. Phase predicting the scanner at AUC 0.99 does
            # not by itself prove the diagnostic result is an artefact -- it
            # proves the artefact is AVAILABLE. What decides the question is
            # whether the result survives holding that acquisition variable
            # out (control 4). So it is reported alongside the verdict, and
            # only the four genuine tests can produce FALSIFIED.
            testable = {k: v for k, v in state.items() if k != "conf"}
            ran = [v for v in testable.values() if v != "absent"]
            if not ran:
                verdict = "NO CONTROLS RUN"
            elif "fails" in ran:
                verdict = "FALSIFIED (" + ",".join(
                    k for k, v in testable.items() if v == "fails") + ")"
            elif "inconclusive" in ran:
                verdict = "NOT ESTABLISHED"
            else:
                verdict = "SURVIVES"
            if state["conf"] == "flagged":
                verdict += " +CONFOUNDED"

            rows.append({
                "cohort": cohort, "condition": condition,
                "headline": fmt_ci(hb.get("auc"), hb.get("lo"), hb.get("hi")),
                "state": state, "verdict": verdict, "perm_txt": perm_txt,
                "bg": bgb, "scr": scb, "acq": acb,
            })

    # ---- the background-only control gets its own prominent block ----------
    head("CONTROL 2 -- BACKGROUND ONLY   (anatomy deleted; only air, coil and shim remain)")
    lines.append("If a model still classifies tumour here, it is reading the scanner.")
    lines.append("")
    lines.append(f"{'cohort':<16}{'condition':<12}{'n_test':>8}{'clusters':>10}"
                 f"{'background AUC [95% CI]':>26}{'usable boots':>14}   verdict")
    lines.append("-" * W)
    if not bg_rows:
        lines.append("  (not run)")
    for cohort, condition, b, st in bg_rows:
        lines.append(
            f"{cohort:<16}{condition:<12}{b.get('n', 0):>8}{b.get('n_clusters', 0):>10}"
            f"{fmt_ci(b.get('auc'), b.get('lo'), b.get('hi'), 26)}"
            f"{b.get('n_boot_ok', 0):>14}   "
            + ("READS THE SCANNER" if st == "fails"
               else "clean" if st == "survives" else "underpowered")
        )
    lines.append("")

    # ---- confound predictability -------------------------------------------
    head("CONTROL 5 -- CONFOUND PREDICTABILITY   (can this input identify the scanner?)")
    lines.append("Labels here are NOT diagnostic. AUC is against the acquisition target named.")
    lines.append("")
    lines.append(f"{'cohort':<16}{'input':<11}{'target':<20}"
                 f"{'AUC [95% CI]':>24}{'bal.acc':>9}{'majority':>10}  rule")
    lines.append("-" * W)
    if not conf_rows:
        lines.append("  (not run)")
    for cohort, condition, target, cb, extra, rule in conf_rows:
        lines.append(
            f"{cohort:<16}{condition:<11}{target:<20}"
            f"{fmt_ci(cb.get('auc'), cb.get('lo'), cb.get('hi'), 24)}"
            f"{extra.get('balanced_accuracy', float('nan')):>9.3f}"
            f"{extra.get('majority_baseline_accuracy', float('nan')):>10.3f}  {rule}"
        )
    lines.append("")

    # ---- the verdict table --------------------------------------------------
    head("VERDICT TABLE")
    lines.append(f"{'cohort':<16}{'input':<11}{'headline AUC [95% CI]':>24}"
                 f"{'perm':>6}{'bkgd':>6}{'scram':>7}{'acq':>6}{'conf':>6}   VERDICT")
    lines.append("-" * W)
    for r in rows:
        s = r["state"]
        lines.append(
            f"{r['cohort']:<16}{r['condition']:<11}{r['headline']:>24}"
            f"{_mark(s['perm']):>6}{_mark(s['bg']):>6}{_mark(s['scr']):>7}"
            f"{_mark(s['acq']):>6}{_mark(s['conf']):>6}   {r['verdict']}"
        )
    lines.append("")
    lines.append("per-condition detail:")
    for r in rows:
        lines.append(f"  {r['cohort']}/{r['condition']}:")
        lines.append(f"    permutation null : {r['perm_txt']}")
        for nm, b in (("background      ", r["bg"]), ("phase-scramble  ", r["scr"]),
                      ("acquisition-split", r["acq"])):
            if b:
                lines.append(f"    {nm} : {fmt_ci(b.get('auc'), b.get('lo'), b.get('hi')).strip()}"
                             f"  (n={b.get('n')}, clusters={b.get('n_clusters')}, "
                             f"usable boots={b.get('n_boot_ok')})")
    lines.append("")
    lines.append("'ok'   = the control did not falsify the result. It is not evidence FOR it.")
    lines.append("'HIGH' = this input predicts the scanner well. '+CONFOUNDED' means the artefact "
                 "is available;")
    lines.append("         control 4 (acquisition-held-out split) is what decides whether it was used.")
    lines.append("'?'    = the test fold is too small to distinguish the control from the "
                 "headline. Report as")
    lines.append("         underpowered, never as support.")
    lines.append("=" * W)
    return "\n".join(lines) + "\n"


# ===========================================================================
# Dry run
# ===========================================================================

def make_dry_run_cache(n_patients: int = 24, slices_per_patient: int = 6, seed: int = 0):
    """
    s03_train's fabricated cache plus a synthetic SCANNER CONFOUND.

    The base cache already puts a Gaussian phase blob on positive slices, which
    is the "anatomical" signal. On top of that, every patient is assigned an
    institution, a receiver-channel count and a source folder, and patients at
    institution B get a smooth linear phase ramp plus a channel-count-dependent
    quadratic term added across the WHOLE frame -- body and air alike, which is
    how a shim/coil signature actually behaves.

    That makes the dry run a real test of the suite and not just of its
    plumbing:
      * institution is assigned independently of the label, so background-only
        should come back at chance -- the control passing is meaningful;
      * confound_predictability should come back high, proving the machinery
        can detect a scanner signature when one is present. A dry run where
        every control returns 0.5 would not distinguish "controls work" from
        "controls are broken and always return 0.5".
    """
    arrays, index = s03_train.make_dry_run_cache(
        n_patients=n_patients, slices_per_patient=slices_per_patient, seed=seed
    )
    rng = np.random.default_rng(seed + 991)
    h, w = arrays["phase"].shape[-2:]
    yy, xx = np.mgrid[0:h, 0:w]
    ramp = (xx / w - 0.5).astype(np.float32)
    quad = (((yy / h - 0.5) ** 2 + (xx / w - 0.5) ** 2) - 0.17).astype(np.float32)

    pids = sorted(index["patient_id"].unique())
    meta = {}
    for i, pid in enumerate(pids):
        inst = "INST_A" if i % 2 == 0 else "INST_B"
        meta[pid] = {
            "subject_id": pid,
            "institution": inst,
            "n_coils": int([14, 16, 20, 24, 26, 30][i % 6]),
            "source_dir": "tree_1" if i % 2 == 0 else "tree_2",
            "mask_frac": float(arrays["mask"].mean()),
        }

    phase = arrays["phase"].astype(np.float32)
    for k, row in index.iterrows():
        m = meta[row["patient_id"]]
        shift = 0.0
        if m["institution"] == "INST_B":
            shift = shift + 1.2 * ramp
        shift = shift + 0.05 * (m["n_coils"] - 20) * quad
        shift = shift + 0.02 * rng.standard_normal((h, w)).astype(np.float32)
        p = phase[k] + shift
        phase[k] = np.arctan2(np.sin(p), np.cos(p))
    arrays["phase"] = phase.astype(np.float16)

    for col in ("subject_id", "institution", "n_coils", "source_dir", "mask_frac"):
        index[col] = index["patient_id"].map(lambda p, c=col: meta[p][c])
    return arrays, index


def check_background_masking(arrays, index) -> None:
    """
    Prove region='background' actually deletes the anatomy.

    The background-only run is the control that decides this paper. If the
    masking silently did nothing, the control would report the headline AUC and
    look like a catastrophic failure of the phase hypothesis -- or, worse, the
    masking could be inverted and the control would look clean while training
    on anatomy. Assert both directions on a real tensor.
    """
    ds = s03_train.CacheSliceDataset(index.head(4), "both", arrays=arrays, region="background")
    x, _, _ = ds[0]
    mask = np.asarray(arrays["mask"][int(index["idx"].iloc[0])], dtype=bool)
    inside = x.numpy()[:, mask]
    outside = x.numpy()[:, ~mask]
    ok_in = float(np.abs(inside).max()) == 0.0
    ok_out = float(np.abs(outside).max()) > 0.0
    print(f"BACKGROUND MASK GUARD: inside-body max|x|={np.abs(inside).max():.3e} (must be 0), "
          f"air max|x|={np.abs(outside).max():.3e} (must be >0) -- "
          f"{'OK' if (ok_in and ok_out) else 'FAILED'}", flush=True)
    if not (ok_in and ok_out):
        raise AssertionError("background masking is not doing what the control claims")


def check_scramble(arrays) -> None:
    """
    Prove the scramble preserves the within-mask marginal and destroys structure.

    Structure is measured with the circular dissimilarity 1 - cos(a - b) of
    horizontally adjacent pixels, NOT |a - b|. Phase wraps, so a raw absolute
    difference reports a huge distance between -3.14 and +3.14 rad, which are
    the same angle; on a wrapped map that metric saturates and cannot tell an
    ordered image from a scrambled one. The reference value is the same
    statistic over random within-mask pixel PAIRS, i.e. what perfect
    structurelessness looks like for this exact marginal. A correct scramble
    lands on that reference; the original must sit well below it.
    """
    scr = PhaseScramble(1234)
    k = 0
    mag = np.asarray(arrays["mag"][k], dtype=np.float32)
    phase = np.asarray(arrays["phase"][k], dtype=np.float32)
    mask = np.asarray(arrays["mask"][k], dtype=bool)
    _, sc, _ = scr(mag, phase, mask, k)

    same_marginal = np.allclose(np.sort(phase[mask]), np.sort(sc[mask]))
    untouched_bg = np.array_equal(phase[~mask], sc[~mask])

    def neighbour_dissim(a, m):
        mm = m[:, 1:] & m[:, :-1]
        return float((1.0 - np.cos(a[:, 1:] - a[:, :-1]))[mm].mean())

    rng = np.random.default_rng(0)
    vals = phase[mask]
    pairs = 1.0 - np.cos(rng.permutation(vals) - rng.permutation(vals))
    reference = float(pairs.mean())  # fully structureless, same marginal

    before, after = neighbour_dissim(phase, mask), neighbour_dissim(sc, mask)
    _, sc2, _ = scr(mag, phase, mask, k)
    deterministic = np.array_equal(sc, sc2)

    destroyed = abs(after - reference) < 0.15 * reference
    had_structure = before < 0.85 * reference
    ok = same_marginal and untouched_bg and deterministic and destroyed and had_structure
    print(f"PHASE SCRAMBLE GUARD: marginal preserved={same_marginal}  background "
          f"untouched={untouched_bg}  deterministic={deterministic}  "
          f"neighbour dissimilarity {before:.3f} -> {after:.3f} "
          f"(structureless reference {reference:.3f})  -- {'OK' if ok else 'FAILED'}",
          flush=True)
    if not ok:
        raise AssertionError("phase scramble does not satisfy its own contract")


def check_permutation(index, subject_col) -> None:
    """Prove the permutation preserves subject-level balance and moves labels."""
    prepared = s03_train.prepare_index(index, 0.2, 0)
    perm, detail = permute_labels_by_subject(prepared, subject_col, ["training"], seed=7)
    d = detail["training"]
    before = prepared[prepared["official_split"] == "training"].groupby(subject_col)["label"].max()
    after = perm[perm["official_split"] == "training"].groupby(subject_col)["label"].max()
    ok = (int(before.sum()) == int(after.sum())
          and d["frac_slice_labels_changed"] > 0
          and d["n_subjects_class_changed"] > 0)
    print(f"LABEL PERMUTATION GUARD: positive subjects {int(before.sum())} -> {int(after.sum())} "
          f"(must match), {100*d['frac_slice_labels_changed']:.0f}% of slice labels moved, "
          f"{d['n_subjects_class_changed']} subjects changed class -- "
          f"{'OK' if ok else 'FAILED'}", flush=True)
    if not ok:
        raise AssertionError("label permutation does not preserve subject-level balance")


# ===========================================================================
# Regression suite for the control-selection / provenance / clustering audit
#
# Each check below reproduces one CONFIRMED defect from the adversarial
# biostatistics audit and asserts it is closed. They run in about a second and
# touch no data, so they are cheap enough to gate every dry run as well as
# being available on their own via --self-test.
# ===========================================================================

def _clustering_fixture(n_subjects: int = 8, scans_per_subject: int = 2,
                        slices_per_scan: int = 8, seed: int = 0):
    """
    A test fold where each subject was scanned twice under a DIFFERENT coded
    patient name -- the breast repeated_scan situation that resolve_subject_col
    exists to handle -- with a subject-level score offset so the two scans of
    one subject are correlated. Returns (index, split_block).
    """
    rng = np.random.default_rng(seed)
    rows, labels, probs, pids, cidx = [], [], [], [], []
    k = 0
    for s in range(n_subjects):
        y = int(s % 2 == 0)
        offset = float(rng.normal(0, 1.5))     # shared by both scans of this subject
        for rep in range(scans_per_subject):
            pid = f"P{s}_{rep}"
            for _ in range(slices_per_scan):
                rows.append({"idx": k, "patient_id": pid, "subject_id": f"S{s}",
                             "label": y, "official_split": "test"})
                labels.append(y)
                probs.append(float(1 / (1 + np.exp(
                    -(offset + 0.8 * (y - 0.5) + 0.05 * rng.normal())))))
                pids.append(pid)
                cidx.append(k)
                k += 1
    index = pd.DataFrame(rows)
    blk = {"labels": labels, "probs": probs, "patient_ids": pids, "cache_idx": cidx}
    return index, blk


def check_defect8_subject_clustering() -> list:
    """
    [8] Control bootstrap CIs must cluster on ctx.subject_col, joined on cache_idx.

    Before: run_tagged passed blk["patient_ids"] straight to
    cluster_bootstrap_auc, so two scans of the same woman counted as two
    independent clusters and every control's interval came out too narrow --
    the direction that makes a falsification control look like it survived.
    """
    problems = []
    index, blk = _clustering_fixture()

    groups = subject_groups_for_split(index, "subject_id", blk, what="regression")
    by_subject = cluster_bootstrap_auc(blk["labels"], blk["probs"], groups,
                                       n_boot=2000, seed=0, unit="subject_id")
    by_patient = cluster_bootstrap_auc(blk["labels"], blk["probs"], blk["patient_ids"],
                                       n_boot=2000, seed=0)
    w_sub = by_subject["hi"] - by_subject["lo"]
    w_pat = by_patient["hi"] - by_patient["lo"]

    if by_subject["n_clusters"] != 8:
        problems.append(f"subject clustering found {by_subject['n_clusters']} clusters, want 8")
    if by_patient["n_clusters"] != 16:
        problems.append(f"patient clustering found {by_patient['n_clusters']} clusters, want 16")
    if not (w_sub > w_pat):
        problems.append(f"subject-clustered CI ({w_sub:.3f}) is not wider than the "
                        f"patient-clustered one ({w_pat:.3f}); the fixture is not "
                        "exercising the defect")
    if "subject_id" not in by_subject["method"]:
        problems.append(f"method string {by_subject['method']!r} does not name the unit")

    # The join must REFUSE rather than fall back when it cannot be completed.
    for what, mutate in (
        ("no cache_idx", lambda i, b: (i, {k: v for k, v in b.items() if k != "cache_idx"})),
        ("no subject_id column", lambda i, b: (i.drop(columns=["subject_id"]), b)),
        ("cache_idx not in the index", lambda i, b: (i.iloc[:4], b)),
    ):
        i2, b2 = mutate(index.copy(), dict(blk))
        try:
            subject_groups_for_split(i2, "subject_id", b2, what="regression")
            problems.append(f"subject_groups_for_split accepted a broken join ({what}) "
                            "instead of refusing")
        except RuntimeError:
            pass

    # ...and the block s06 actually reads must be the subject-clustered one.
    # Testing the helper alone would pass even if run_tagged still handed
    # patient_ids to the bootstrap, which is precisely what the defect was, so
    # go through the real call path with a stand-in context.
    ctx = ControlContext(
        cohort="regression", index=index, subject_col="subject_id", device=None,
        args=argparse.Namespace(n_boot=2000, bootstrap_seed=0), train_args=None,
        results_root=Path("."), ckpt_root=Path("."),
    )
    emitted = split_bootstrap(ctx, blk, what="regression")
    if emitted.get("cluster_unit") != "subject_id":
        problems.append(f"the emitted CI block says cluster_unit="
                        f"{emitted.get('cluster_unit')!r}, want 'subject_id'")
    if emitted.get("n_clusters") != 8:
        problems.append(f"the emitted CI block clustered on {emitted.get('n_clusters')} "
                        "units; with 8 subjects scanned twice each, 16 means it is still "
                        "clustering on patient_id")
    if emitted.get("n_patient_clusters") != 16:
        problems.append("the emitted CI block does not record the patient-level count "
                        "for comparison")
    if abs((emitted["hi"] - emitted["lo"]) - w_sub) > 1e-9:
        problems.append("the emitted interval does not match the subject-clustered one")

    print(f"  [8] subject-clustered control CI: n_clusters {by_patient['n_clusters']} "
          f"(patient) -> {emitted.get('n_clusters')} (subject), CI width "
          f"{w_pat:.3f} -> {emitted['hi'] - emitted['lo']:.3f}  -- "
          f"{'OK' if not problems else 'FAILED'}", flush=True)
    return problems


def check_defect3_failure_ledger(tmp_root: Path) -> list:
    """
    [3] Swallowed failures must be PERSISTED where stage 6 can see them.

    Before: ctx.failures lived in memory, was printed once, and died with the
    process. A single-class acquisition arm was therefore indistinguishable on
    disk from a control that ran cleanly in one direction.
    """
    problems = []
    path = tmp_root / FAILURES_FILENAME
    led = FailureLedger(path)
    led.begin("prostate_dwi", "acquisition_split")
    led.begin("prostate_dwi", "background_only")
    led.fail("prostate_dwi", "acquisition_split",
             "B2A: test split is single-class on the diagnostic label", variant="B2A")

    if not path.exists():
        problems.append("ledger did not write a file")
        return problems
    obj = json.loads(path.read_text())
    if obj.get("schema") != FAILURES_SCHEMA:
        problems.append(f"schema is {obj.get('schema')!r}, want {FAILURES_SCHEMA!r}")
    by = {(e["cohort"], e["control"]): e for e in obj["entries"]}
    acq = by.get(("prostate_dwi", "acquisition_split"))
    bg = by.get(("prostate_dwi", "background_only"))
    if not (acq and acq.get("failed")):
        problems.append("the failed acquisition control is not marked failed on disk")
    if not (bg and bg.get("attempted") and not bg.get("failed")):
        problems.append("a control that ran clean is not recorded as attempted-and-ok")
    if acq and "B2A" not in json.dumps(acq):
        problems.append("the failing direction is not named in the record")

    # A later successful re-run of the same control must CLEAR the failure --
    # a ledger that can only accumulate would make the tree permanently
    # unusable after one transient error.
    led2 = FailureLedger(path)
    led2.begin("prostate_dwi", "acquisition_split")
    obj2 = json.loads(path.read_text())
    by2 = {(e["cohort"], e["control"]): e for e in obj2["entries"]}
    if by2[("prostate_dwi", "acquisition_split")].get("failed"):
        problems.append("a clean re-run did not clear the previous failure")
    if not by2[("prostate_dwi", "background_only")].get("attempted"):
        problems.append("re-running one control dropped another control's record")

    print(f"  [3] failure ledger persisted to {path.name}: "
          f"{len(obj['entries'])} entries, {len(led.failed)} failed, "
          f"re-run clears state  -- {'OK' if not problems else 'FAILED'}", flush=True)
    return problems


def check_defects_2_3_in_s06(tmp_root: Path) -> list:
    """
    [2] and [3] as stage 6 sees them: the s05 -> s06 contract.

    Reproduces the audit's two SUPPORTED-by-accident scenarios end to end
    against the real s06 Controls class:

      [2] a background-only control run ONLY on the magnitude model must not be
          credited to phase (before: select() fell back to any condition);
      [3] an acquisition control whose B2A arm died must not be scored on A2B
          alone (before: min() over whatever was on disk).
    """
    problems = []
    try:
        import s06_report as s06
    except Exception as exc:  # noqa: BLE001
        print(f"  [2/3] s06 contract: CANNOT IMPORT s06_report ({exc}) -- FAILED", flush=True)
        return [f"s06_report is not importable: {exc}"]

    def payload(control, auc, condition="phase", variant=None, semantics="diagnosis"):
        n, npos = 40, 11
        return {
            "cohort": "prostate_dwi", "condition": condition, "seed": 42, "region": "full",
            "test": {"probs": [0.9] * npos + [0.1] * (n - npos),
                     "labels": [1] * npos + [0] * (n - npos),
                     "patient_ids": [f"S{i % 4}" for i in range(n)],
                     "cache_idx": list(range(n)), "auc": auc, "n": n, "n_pos": npos},
            "control": control,
            "control_detail": {
                "variant": variant, "label_semantics": semantics,
                "subject_col": "subject_id",
                "test_auc_ci95": {"auc": auc, "lo": max(0.0, auc - 0.13),
                                  "hi": min(1.0, auc + 0.13), "n": n, "n_pos": npos,
                                  "n_clusters": 4, "n_boot_ok": 1180,
                                  "cluster_unit": "subject_id"},
            },
        }

    # --- [2] cross-condition fallback ---------------------------------------
    only_magnitude = [payload("background_only", 0.55, condition="magnitude")]
    c = s06.Controls(only_magnitude, [tmp_root / "empty"])
    if c.select("prostate_dwi", "background", "phase"):
        problems.append("[2] select() still returns magnitude runs for a phase request")
    if c.estimate("prostate_dwi", "background", "phase").ok:
        problems.append("[2] a magnitude-only background control still yields a phase estimate")
    if not c.estimate("prostate_dwi", "background", "magnitude").ok:
        problems.append("[2] the magnitude control is no longer readable as magnitude")

    # --- [3a] one surviving acquisition direction ---------------------------
    both = [payload("acquisition_split", 0.85, variant="A2B"),
            payload("acquisition_split", 0.55, variant="B2A")]
    c_both = s06.Controls(both, [tmp_root / "empty"])
    e_both = c_both.estimate("prostate_dwi", "acquisition")
    if not e_both.ok or abs(e_both.point - 0.55) > 1e-9:
        problems.append(f"[3] with both directions the WORSE one must be scored, got "
                        f"{e_both.point!r} from {e_both.source}")
    c_one = s06.Controls(both[:1], [tmp_root / "empty"])
    e_one = c_one.estimate("prostate_dwi", "acquisition")
    if e_one.ok:
        problems.append(f"[3] a single split direction still produced an estimate "
                        f"({e_one.point:.3f}); C5 would be scored on the surviving arm")

    # --- [3b] the ledger reaches s06 ----------------------------------------
    tree = tmp_root / "controls_tree"
    tree.mkdir(parents=True, exist_ok=True)
    led = FailureLedger(tree / FAILURES_FILENAME)
    for name in ("acquisition_split", "background_only", "phase_scramble",
                 "label_permutation", "confound_predictability"):
        led.begin("prostate_dwi", name)
    led.fail("prostate_dwi", "background_only", "cache read error on 3 of 40 test rows")
    led.fail("prostate_dwi", "confound_predictability", "institution column is all null")

    full = both + [payload("background_only", 0.55),
                   payload("phase_scramble", 0.55),
                   payload("label_permutation", 0.50, variant="perm00"),
                   payload("confound_predictability", 0.60, variant="institution",
                           semantics="confound:institution")]
    c_led = s06.Controls(full, [tree])
    if not c_led.failures:
        problems.append("[3] s06 did not read the failure ledger at all")
    if c_led.estimate("prostate_dwi", "background").ok:
        problems.append("[3] a control stage 5 recorded as FAILED is still being scored")
    if c_led.confound_targets("prostate_dwi"):
        problems.append("[3] a failed confound control still yields targets to max() over")
    if not c_led.estimate("prostate_dwi", "scramble").ok:
        problems.append("[3] a control with no recorded failure was wrongly suppressed")

    # No ledger on disk must not mean "everything failed": trees predating the
    # ledger are legitimate and are still readable.
    if not s06.Controls(full, [tmp_root / "no_such_dir"]).estimate(
            "prostate_dwi", "background").ok:
        problems.append("[3] absence of a ledger was misread as a failure")

    print(f"  [2] cross-condition fallback removed; [3] single-direction and "
          f"ledger-failed controls report MISSING  -- "
          f"{'OK' if not problems else 'FAILED'}", flush=True)
    return problems


def run_regression_checks() -> int:
    """The audit regression suite. Returns the number of FAILED checks."""
    print("=" * 82, flush=True)
    print("s05 REGRESSION SUITE -- audit defects [2] control selection, "
          "[3] provenance, [8] clustering", flush=True)
    print("=" * 82, flush=True)
    tmp_root = Path(tempfile.mkdtemp(prefix="s05_regression_"))
    checks = []
    try:
        checks.append(("[8] subject-level clustering", check_defect8_subject_clustering()))
        checks.append(("[3] failure ledger on disk",
                       check_defect3_failure_ledger(tmp_root / "ledger")))
        checks.append(("[2]+[3] s05 -> s06 contract",
                       check_defects_2_3_in_s06(tmp_root / "s06")))
    finally:
        shutil.rmtree(tmp_root, ignore_errors=True)
    failed = 0
    for name, problems in checks:
        if problems:
            failed += 1
            print(f"  {name}: FAILED", flush=True)
            for p in problems:
                print(f"      - {p}", flush=True)
    print("=" * 82, flush=True)
    print(f"regression suite: {len(checks) - failed}/{len(checks)} checks passed", flush=True)
    print("=" * 82, flush=True)
    return failed


# ===========================================================================
# CLI
# ===========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PhaseDx stage 5: the falsification suite",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--cohort", help="cohort to falsify (repeatable via comma), or --dry-run")
    p.add_argument("--controls", default="all",
                   help=f"comma-separated subset of {','.join(CONTROLS)} (or 'all')")
    p.add_argument("--conditions", default=None,
                   help="override the per-control default input conditions")
    p.add_argument("--seeds", default="42", help="training seeds per control run")
    p.add_argument("--with-headline", action="store_true",
                   help="also run the uncontrolled headline so the table has a comparator")
    p.add_argument("--summary-only", action="store_true",
                   help="render the verdict table from JSON already on disk")
    p.add_argument("--split-col", default="official_split",
                   help="split column to falsify on; must be the SAME column the "
                        "headline runs used (run_full.sh passes --split-col cv<k>_split "
                        "for the cross-validated clinical cohorts)")

    # control 1
    p.add_argument("--n-permutations", type=int, default=10,
                   help="permutation replicates; <20 cannot resolve p=0.05")
    p.add_argument("--permutation-seed0", type=int, default=1000)
    p.add_argument("--permute-splits", default="training",
                   help="splits whose labels are permuted; 'training,validation' puts "
                        "epoch selection under H0 too")
    # control 3
    p.add_argument("--scramble-seed", type=int, default=20240517)
    # control 4
    p.add_argument("--strat-key", default=None,
                   help="acquisition variable to hold out (default: per cohort)")
    p.add_argument("--both-directions", dest="both_directions", action="store_true", default=True)
    p.add_argument("--one-direction", dest="both_directions", action="store_false")
    # control 5
    p.add_argument("--confounds", default=None,
                   help="comma-separated non-diagnostic targets (default: per cohort)")
    p.add_argument("--confound-split-seed", type=int, default=13)
    p.add_argument("--confound-fracs", default="0.6,0.2,0.2")
    p.add_argument("--confound-use-official-split", action="store_true",
                   help="keep the diagnostic split instead of stratifying on the confound "
                        "(often leaves a single-class test fold)")

    # statistics
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--bootstrap-seed", type=int, default=0)

    # training passthrough: default None so unset flags keep s03_train's defaults
    for name, kind in (("--epochs", int), ("--batch-size", int), ("--lr", float),
                       ("--wd", float), ("--dropout", float), ("--patience", int),
                       ("--workers", int), ("--label-smoothing", float),
                       ("--grad-clip", float), ("--warmup-epochs", int),
                       ("--min-delta", float), ("--val-frac", float),
                       ("--val-split-seed", int)):
        p.add_argument(name, type=kind, default=None)
    p.add_argument("--freeze-backbone", action="store_true", default=None)
    p.add_argument("--no-pretrained", action="store_true", default=None)
    p.add_argument("--device", default="auto", choices=("auto", "cuda", "mps", "cpu"))

    p.add_argument("--cache-dir", default=str(common.CACHE_DIR))
    p.add_argument("--cohort-dir", default=str(common.OUT_ROOT / "cohorts"))
    p.add_argument("--results-dir", default=str(common.OUT_ROOT / "controls" / "results"))
    p.add_argument("--ckpt-dir", default=str(common.OUT_ROOT / "controls" / "checkpoints"))
    p.add_argument("--headline-dir", default=str(common.RESULTS_DIR))
    p.add_argument("--keep-checkpoints", action="store_true",
                   help="retain the 45 MB ResNet-18 weights of every control run")
    p.add_argument("--dry-run", action="store_true",
                   help="fabricated in-memory cache with a synthetic scanner confound")
    p.add_argument("--self-test", dest="self_test", action="store_true",
                   help="run the audit regression suite (control selection, failure "
                        "provenance, subject-level clustering) and exit; no data needed")
    return p.parse_args(argv)


def build_train_args(args) -> argparse.Namespace:
    """
    Start from s03_train's own defaults, then apply only what the user set.

    Deriving the namespace from s03_train.parse_args([]) rather than
    reconstructing it here means the control runs cannot silently drift away
    from the headline runs' hyperparameters when s03_train changes.
    """
    base = s03_train.parse_args([])
    # results_dir/ckpt_dir are owned by run_tagged (every run is redirected
    # through a scratch dir and refiled), so they must not be copied over.
    owned_by_s05 = {"cohort", "conditions", "seeds", "dry_run", "results_dir", "ckpt_dir"}
    for k, v in vars(args).items():
        if v is not None and hasattr(base, k) and k not in owned_by_s05:
            setattr(base, k, v)
    base.dry_run = False
    # Not part of s03_train's CLI, but load_cohort needs it.
    base.cohort_dir = args.cohort_dir
    return base


def main(argv=None):
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")

    if args.self_test:
        return 1 if run_regression_checks() else 0

    args.seeds = [int(s) for s in str(args.seeds).split(",") if str(s).strip()]
    args.confound_fracs = [float(x) for x in str(args.confound_fracs).split(",")]
    controls = list(CONTROLS) if args.controls == "all" else [
        c.strip() for c in args.controls.split(",") if c.strip()]
    for c in controls:
        if c not in CONTROLS:
            print(f"ERROR: unknown control {c!r}; choose from {list(CONTROLS)}", file=sys.stderr)
            return 2

    results_root = Path(args.results_dir)
    ckpt_root = Path(args.ckpt_dir)
    headline_dirs = [Path(args.headline_dir), results_root]

    if args.dry_run:
        # Keep fabricated output out of the real results tree -- but only when
        # the user did not explicitly point somewhere, exactly as s03_train
        # does. Silently ignoring an explicit --results-dir is the kind of
        # thing that makes someone spend an hour looking for missing files.
        defaults = parse_args([])
        if args.results_dir == defaults.results_dir:
            results_root = Path(common.OUT_ROOT / "dryrun" / "controls" / "results")
        if args.ckpt_dir == defaults.ckpt_dir:
            ckpt_root = Path(common.OUT_ROOT / "dryrun" / "controls" / "checkpoints")
        headline_dirs = [results_root]
        args.with_headline = True
        args.n_permutations = min(args.n_permutations, 4)
        args.n_boot = min(args.n_boot, 500)
        cohorts = ["dryrun"]
    else:
        if not args.cohort:
            print("ERROR: --cohort is required (or use --dry-run)", file=sys.stderr)
            return 2
        cohorts = [c.strip() for c in args.cohort.split(",") if c.strip()]

    if args.summary_only:
        print(summarize([results_root], headline_dirs, args.n_boot,
                        args.bootstrap_seed, cohorts), flush=True)
        return 0

    train_args = build_train_args(args)
    if args.dry_run:
        train_args.epochs = args.epochs if args.epochs is not None else 6
        train_args.warmup_epochs = 1
        train_args.batch_size = args.batch_size if args.batch_size is not None else 16
        train_args.workers = 0
    device = s03_train.pick_device(args.device)

    all_failures = []
    ledger = FailureLedger(results_root / FAILURES_FILENAME)
    for cohort in cohorts:
        print("#" * 100, flush=True)
        print(f"# COHORT {cohort}", flush=True)
        print("#" * 100, flush=True)

        if args.dry_run:
            print("DRY RUN -- fabricated cache with a synthetic scanner confound, "
                  "no drive access", flush=True)
            # The audit regression suite runs first and hard-fails the dry run:
            # a dry run that exercises the controls while control SELECTION is
            # broken proves nothing about the numbers it prints.
            if run_regression_checks():
                raise AssertionError(
                    "the control selection/provenance/clustering regression suite "
                    "failed; refusing to run controls")
            arrays, raw = make_dry_run_cache()
            s03_train.check_leakage_guard()
            check_scramble(arrays)
            check_permutation(raw.copy(), "subject_id")
            index = s03_train.prepare_index(raw, train_args.val_frac,
                                            train_args.val_split_seed).reset_index(drop=True)
            check_background_masking(arrays, index)
            subject_col = resolve_subject_col(index)
            assert_no_group_leakage(index, subject_col, "subject")
            h5_path = None
            print(flush=True)
        else:
            try:
                index, h5_path, subject_col = load_cohort(cohort, train_args)
            except Exception as exc:  # noqa: BLE001
                logger.error("cannot load cohort %s: %s", cohort, exc)
                # A cohort that will not load has no evaluable control at all;
                # record every requested one as failed so stage 6 cannot score
                # leftovers from a previous run against this cohort.
                for name in controls:
                    ledger.fail(cohort, name, f"cohort failed to load: {exc}")
                all_failures.append({"cohort": cohort, "control": "load", "error": str(exc)})
                continue
            arrays = None

        ctx = ControlContext(
            cohort=cohort, index=index, subject_col=subject_col, device=device,
            args=args, train_args=train_args, results_root=results_root,
            ckpt_root=ckpt_root, h5_path=h5_path, arrays=arrays,
            strat_key=args.strat_key or DEFAULT_STRAT_KEY.get(cohort, "institution"),
            confounds=tuple(
                [c.strip() for c in args.confounds.split(",")] if args.confounds
                else DEFAULT_CONFOUNDS.get(cohort, ("institution",))
            ),
            ledger=ledger,
        )

        if args.with_headline:
            ledger.begin(cohort, "none")
            _guarded(ctx, "none", lambda: [
                run_tagged(ctx, control="none", condition=c, seed=s, index=ctx.index)
                for c in (args.conditions.split(",") if args.conditions else common.CONDITIONS)
                for s in args.seeds
            ])

        for name in controls:
            conds = ([c.strip() for c in args.conditions.split(",")] if args.conditions
                     else list(DEFAULT_CONDITIONS[name]))
            # begin() before the run so a hard kill mid-control leaves an
            # attempted-but-unfinished entry rather than nothing.
            ledger.begin(cohort, name)
            _guarded(ctx, name, lambda n=name, cd=conds: CONTROL_FUNCS[n](ctx, cd))
        all_failures.extend(ctx.failures)

    ledger_path = ledger.flush()
    print(flush=True)
    print(summarize([results_root], headline_dirs, args.n_boot,
                    args.bootstrap_seed, cohorts), flush=True)
    if all_failures:
        print("CONTROLS THAT COULD NOT RUN (operational, not scientific):", flush=True)
        for f in all_failures:
            print(f"  {f['cohort']}/{f['control']}: {f['error']}", flush=True)
        print("These are recorded in the failure ledger; stage 6 reports the "
              "matching criteria MISSING rather than scoring the surviving runs.",
              flush=True)
        print(flush=True)
    print(f"control results  -> {results_root}", flush=True)
    print(f"failure ledger   -> {ledger_path} "
          f"({len(ledger.failed)} failed / {len(ledger.entries)} attempted)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
