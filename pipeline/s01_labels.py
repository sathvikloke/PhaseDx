"""
s01_labels.py
-------------
Stage 1 of the PhaseDx pipeline: build the cohort tables.

One tidy table per cohort, written to pipeline_out/cohorts/<cohort>_cohort.csv:

    prostate_dwi   one row per SLICE   (slice-level PI-RADS)
    prostate_t2    one row per SLICE   (slice-level PI-RADS)
    breast         one row per ACQUISITION (patient-level lesion status)
    brain          one row per FILE    (CONFOUND target: receive-coil count)
    knee           one row per FILE    (CONFOUND target: contrast / fat suppression)

Everything downstream (caching, training, confound analysis) reads these tables
and nothing else. They are the single source of truth for "which slice of which
file has which label and which split".

THE BRAIN AND KNEE COHORTS HAVE NO TUMOUR LABELS AND NEVER WILL
===============================================================
The fastMRI brain and knee releases ship no pathology annotation of any kind.
Their `label` column is a HARDWARE / PROTOCOL property -- how many receive coils
the array had, or whether fat suppression was on -- and every row carries
`label_kind='confound_target'` and `has_tumour_label=0` so that no downstream
stage, figure, or sentence can mistake them for diagnosis. They exist to answer
one question: how much of what a phase-only network learns is the scanner rather
than the patient.

    brain   coil count VARIES (4..20 channels). If phase predicts it, phase is
            carrying an acquisition fingerprint, because coil count has no
            diagnostic meaning whatsoever.
    knee    coil count is FIXED at 15 for every file. With the hardware held
            constant, can phase still separate CORPD_FBK from CORPDFS_FBK? This
            is the control on the control: it distinguishes "phase encodes the
            coil array" from "phase encodes the pulse sequence".

DEDUPLICATION IS A CORRECTNESS REQUIREMENT, NOT HYGIENE
=======================================================
knee/val on this drive contains a complete duplicate copy of the brain release
(and five brain files that are NOT in brain/val at all). A glob over both trees
double-counts every brain patient and, because one patient then owns two rows
with two paths, can place the same patient on both sides of a split. Files are
therefore attributed to a cohort by BASENAME pattern, deduplicated by basename
with the canonical organ tree winning, checked a second time against
(acquisition, patient_id, kspace shape), and asserted unique before any split is
drawn.

Design decisions that matter
============================

1. Join is against files ACTUALLY PRESENT on disk (common.iter_h5, which filters
   the 1425 macOS ._* AppleDouble sidecars). Both directions of the join failure
   are reported and written to sidecar CSVs -- nothing is dropped silently.

2. Prostate label is SLICE-level PI-RADS >= 3. The fastMRI prostate release ships
   per-slice PI-RADS, which is a far stronger target than the exam-level grade the
   old draft used. exam_level / t2_volume_level / dwi_volume_level are carried
   along as extra columns for a volume-level analysis, not as the primary label.

3. Slice indexing. The label CSVs number slices 1..n. This stage verifies, for
   every readable file, that n equals the k-space slice dimension and that the
   labelled indices are exactly 1..n (it holds for all 112 readable prostate
   files). Two columns are emitted:
       slice        0-based index into the h5 slice axis  <-- USE THIS to index
       slice_1based the raw value from the label CSV
   Downstream code must index kspace with `slice`, not `slice_1based`.

4. Breast leakage. Breast has TWO acquisitions per coded patient, and the
   'Repeated scans' column links pairs of DIFFERENT coded names that are the SAME
   physical person. 10 of the 16 repeat pairs straddle the official train/test
   split in the full 300-patient label sheet. So the split-enforcement unit is
   `subject_id` (repeat-group-collapsed), not `patient_id`. Both are emitted and
   the no-subject-spans-splits property is asserted, not assumed.

5. Corrupt files. 9 of the 121 prostate .h5 files on this drive cannot be opened
   at all ("bad symbol table node signature"). They are kept in the table with
   h5_ok=0 so the loss is visible and auditable; every consumer must filter on
   h5_ok. Class balance is reported both with and without them.

6. Scanner confounds. The whole point of this study is to rule out "phase is just
   a scanner fingerprint", so this stage extracts vendor / model / field strength
   / institution / device id / protocol / TR / TE / sequence / matrix / FOV /
   receiver channels from each prostate file's ismrmrd_header, plus array shapes
   for every cohort. Those columns are what stage 4 conditions on.

7. Patient-level cross-validation. The official test folds are too small to
   support a confirmatory conclusion: prostate_t2's holds 7 subjects (4 positive,
   3 negative) against s06_report's gate of >= 10 clusters with >= 5 per class,
   and prostate_dwi's holds 4 (3 positive, 1 negative). The confirmatory path is
   blocked by fold SIZE, not by any finding, which is not a defensible reason to
   report a negative result. So this stage also emits a seeded, deterministic,
   stratified K-fold partition at SUBJECT level (`cv_fold` plus one
   `cv<k>_split` column per fold, each with its own nested inner validation
   split carved from that fold's training subjects for model selection and
   thresholding). The official-split columns are kept untouched so the official
   split remains reportable as a secondary analysis.

Usage:
    python pipeline/s01_labels.py
    python pipeline/s01_labels.py --cohorts breast --no-probe
    python pipeline/s01_labels.py --cohorts brain knee
    python pipeline/s01_labels.py --seed 1234 --val-frac 0.2 --test-frac 0.2
    python pipeline/s01_labels.py --cv-folds 4 --inner-val-frac 0.2
    python pipeline/s01_labels.py --self-test
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from pipeline.common import (  # noqa: E402
    BREAST_LABEL_XLSX,
    DATA_ROOT,
    OUT_ROOT,
    PROSTATE_LABEL_DIR,
    iter_h5,
)

COHORT_DIR = OUT_ROOT / "cohorts"

# Breast raw files live in two sibling trees on this drive: the original
# 20-patient release and the 70-patient "breast_updated" release. The patient id
# ranges are disjoint (001-020 vs 131-160/261-300), so both are usable; the tree
# is recorded in `source_dir` because it is a candidate batch confound.
BREAST_DIRS = ("breast_updated", "breast")

BREAST_COLMAP = {
    "Patient Coded Name": "patient_coded_name",
    "Repeated scans (0 = not repeat, 1:16 = repeated)": "repeat_group",
    "Data split (0=training, 1=testing)": "official_split_raw",
    "Age": "age",
    "Menopause (0= pre, 1 = post, 2 = unknown)": "menopause",
    "Reason (0= increased risk screening, 1= short-term f/u, 2= cancer workup, "
    "3=problem-solving, 4=neoadjuvant therapy response": "reason",
    "Lesion status (0 = negative, 1= malignancy, 2= benign)": "lesion_status",
    "Malignant lesion type": "malignant_lesion_type",
    "Laterality (1=right, 2=left)": "laterality",
    "Detection type (1= mammo, 2= US, 3 = MRI)": "detection_type",
    "Biomarkers (Raw Text)": "biomarkers_raw",
    "Grade (1-3)": "grade",
    "ER+ (yes=1, no=0)": "er_pos",
    "PR+ (yes=1, no=0)": "pr_pos",
    "HER2+ (yes=1, no=0)": "her2_pos",
}

BREAST_FILE_RE = re.compile(r"^fastMRI_breast_(\d+)_(\d+)\.h5$")

# --------------------------------------------------------------------------
# Confound cohorts (brain, knee)
# --------------------------------------------------------------------------

CONFOUND_COHORTS = ("brain", "knee")
CLINICAL_COHORTS = ("prostate_dwi", "prostate_t2", "breast")
ALL_COHORTS = CLINICAL_COHORTS + CONFOUND_COHORTS

# Membership is decided by BASENAME, not by which folder the file is in, because
# on this drive those two disagree for 460 files. Brain val files are
# file_brain_<ACQ>_<batch>_<id>.h5; knee val files are file<digits>.h5.
CONFOUND_FILE_RE = {
    "brain": re.compile(r"^file_brain_(?P<acq>[A-Za-z0-9]+)_(?P<batch>\d+)_(?P<fid>\d+)\.h5$"),
    "knee": re.compile(r"^file(?P<fid>\d+)\.h5$"),
}

# Trees searched for each confound cohort. Both are searched for both cohorts
# precisely so the cross-contamination is FOUND and reported rather than assumed
# away; the canonical tree then wins the dedup.
CONFOUND_SEARCH_DIRS = ("brain", "knee")

# Which confound target is the primary `label` for each cohort, and why.
CONFOUND_TARGETS = {
    # Coil count is pure hardware. It cannot be a disease correlate, so any
    # accuracy above the class prior is fingerprinting by definition.
    "brain": "n_coils",
    # Coil count is constant (15) across the whole knee release, so a model that
    # separates fat-suppressed from non-fat-suppressed cannot be reading the
    # coil array. It is reading the pulse sequence.
    "knee": "acquisition",
}

# Secondary targets emitted as extra columns for each confound cohort.
CONFOUND_SECONDARY_TARGETS = {
    "brain": ("acquisition", "matrix"),
    "knee": ("n_coils", "matrix"),
}

# Extra candidate confounds that only exist on the confound cohorts. Kept out of
# ACQUISITION_CONFOUNDS so the prostate/breast screens report exactly the same
# set of tests they reported before this stage learned about brain and knee.
CONFOUND_COHORT_EXTRA_CONFOUNDS = (
    "n_coils", "n_slices", "matrix_kx", "matrix_ky", "matrix",
    "acquisition", "dir_organ", "attr_max", "attr_norm", "recon_rss_shape",
)

# Columns that are the target itself under another name. Testing the target
# against the target returns p ~ 0 and would swamp the screen.
TARGET_ALIASES = {
    "n_coils": ("n_coils", "receiver_channels", "kspace_shape"),
    "acquisition": ("acquisition", "attr_acquisition", "protocol_name"),
    "matrix": ("matrix", "matrix_kx", "matrix_ky", "enc_matrix_x", "enc_matrix_y",
               "kspace_shape"),
}

# Candidate confounds: things fixed by how/where/when the scan was acquired, or
# by the patient, all knowable without knowing the outcome. These are what the
# stage-1 screen tests against the label.
ACQUISITION_CONFOUNDS = (
    "vendor", "scanner_model", "field_strength_T", "institution", "device_id",
    "protocol_name", "sequence_type", "patient_position", "TR", "TE", "TI",
    "flip_angle_deg", "echo_spacing", "receiver_channels", "enc_matrix_x",
    "enc_matrix_y", "recon_matrix_x", "fov_x_mm", "fov_y_mm", "kspace_shape",
    "kspace_last_dim", "file_size_bytes", "source_dir", "folder",
    "age", "menopause", "reason", "repeat_group",
)

# Columns derived from the outcome. Kept in the table for descriptive analysis,
# but excluded from the confound screen because their association with the label
# is tautological.
LABEL_DERIVED_COLS = (
    "grade", "er_pos", "pr_pos", "her2_pos", "malignant_lesion_type",
    "laterality", "detection_type", "exam_level", "t2_volume_level",
    "dwi_volume_level",
)

# Columns every cohort table starts with, in this order. `cv_fold` sits with the
# other split columns; the per-fold cv<k>_split columns are appended after it by
# add_cv_folds and are not listed here because their number depends on K.
LEAD_COLS = [
    "idx", "cohort", "patient_id", "subject_id", "file", "slice", "slice_1based",
    "label", "raw_label", "label_kind", "has_tumour_label", "official_split",
    "fallback_split", "split", "split_with_val", "cv_fold", "folder", "acq",
    "h5_ok",
]

# The s06_report gate that the official folds currently fail. Imported rather
# than re-typed so the two files cannot drift apart; the fallback exists only so
# that stage 1 still runs if stage 6 is being edited.
try:  # pragma: no cover - exercised implicitly by every real run
    from pipeline.s06_report import (  # noqa: E402
        MIN_CLASS_CLUSTERS_C1 as _S06_MIN_PER_CLASS,
        MIN_CLUSTERS_C1 as _S06_MIN_CLUSTERS,
    )
    GATE_SOURCE = "pipeline.s06_report"
except Exception:  # noqa: BLE001
    _S06_MIN_CLUSTERS, _S06_MIN_PER_CLASS = 10, 5
    GATE_SOURCE = "hard-coded fallback (could not import pipeline.s06_report)"

MIN_CLUSTERS_C1 = _S06_MIN_CLUSTERS
MIN_CLASS_CLUSTERS_C1 = _S06_MIN_PER_CLASS


# --------------------------------------------------------------------------
# Deterministic, subject-level, stratified splitting
# --------------------------------------------------------------------------

def hash_unit(key: str, seed: int) -> float:
    """
    Map a subject id to a stable pseudo-random number in [0, 1).

    Hashing rather than an RNG shuffle so that a subject's draw depends only on
    its own id and the seed -- adding or removing other subjects (e.g. when more
    of the drive gets copied over) does not reshuffle everyone else.
    """
    digest = hashlib.sha1(f"{seed}|{key}".encode("utf-8")).hexdigest()
    return int(digest[:16], 16) / float(1 << 64)


def stratified_subject_split(subject_label: pd.Series, seed: int,
                             val_frac: float, test_frac: float) -> dict:
    """
    Assign each subject to training/validation/test.

    subject_label: index = subject_id, value = subject-level binary label
    (max over that subject's rows -- a patient with any positive slice counts as
    positive for stratification).

    Stratifying by label is not cosmetic here: prostate DWI has 25 positive
    patients out of 50 but only ~7% positive slices, and breast has 20 negative
    patients out of 90. An unstratified random split can easily produce a test
    fold with zero positives, which makes AUC undefined.
    """
    assignment = {}
    for lab, group in subject_label.groupby(subject_label):
        subjects = sorted(group.index.astype(str))
        ranked = sorted(subjects, key=lambda s: hash_unit(s, seed))
        n = len(ranked)
        n_test = int(round(n * test_frac))
        n_val = int(round(n * val_frac))
        # Never starve training; with 1-2 subjects in a stratum, training wins.
        n_test = min(n_test, max(0, n - 1))
        n_val = min(n_val, max(0, n - n_test - 1))
        for i, s in enumerate(ranked):
            if i < n_test:
                assignment[s] = "test"
            elif i < n_test + n_val:
                assignment[s] = "validation"
            else:
                assignment[s] = "training"
        del lab
    return assignment


def carve_validation(df: pd.DataFrame, seed: int, val_frac: float) -> pd.Series:
    """
    Produce `split_with_val`: identical to `split`, except that when a cohort's
    official split has no validation fold (breast is train/test only), a
    deterministic subject-level slice of training becomes validation.

    Model selection has to happen somewhere. Doing it on the official test fold
    is how you manufacture an AUC of 0.97, so we carve validation out of
    training instead, at subject level, with the same stratified hash.
    """
    out = df["split"].copy()
    if (df["split"] == "validation").any():
        return out
    train_mask = df["split"] == "training"
    if not train_mask.any():
        return out
    subj_lab = df.loc[train_mask].groupby("subject_id")["label"].max()
    picks = set()
    for lab, group in subj_lab.groupby(subj_lab):
        subjects = sorted(group.index.astype(str))
        ranked = sorted(subjects, key=lambda s: hash_unit(s, seed + 977))
        n_val = min(int(round(len(ranked) * val_frac)), max(0, len(ranked) - 1))
        picks.update(ranked[:n_val])
        del lab
    out.loc[train_mask & df["subject_id"].astype(str).isin(picks)] = "validation"
    return out


# --------------------------------------------------------------------------
# Patient-level cross-validation
# --------------------------------------------------------------------------
#
# Why this exists at all. The official fastMRI/prostate test folds are not
# merely small, they are below the threshold at which the report is allowed to
# read anything off them:
#
#     prostate_t2    7 test subjects (4 positive, 3 negative)
#     prostate_dwi   4 test subjects (3 positive, 1 negative)
#
# against s06_report's MIN_CLUSTERS_C1 = 10 and MIN_CLASS_CLUSTERS_C1 = 5. On
# folds that small a percentile bootstrap over clusters fires at 3-25% under a
# complete null against a nominal 2.5%, so C1 comes back MISSING -- neither PASS
# nor FAIL. Every subject in the cohort is already labelled and already usable;
# they are simply sitting in the training fold. K-fold recovers them as test
# subjects without ever letting a subject score itself.
#
# Three properties are enforced rather than hoped for:
#   * folds are drawn on subject_id, which is the leak-proof unit (see the note
#     in common.py: breast codes some women twice under different patient_ids);
#   * every eligible subject is a test subject in exactly one fold, and appears
#     in no other fold's test set;
#   * each fold's training subjects are further split into an inner training and
#     an inner VALIDATION set, so early stopping, model selection and threshold
#     choice never touch that fold's test subjects. A K-fold without a nested
#     inner split is a slow way to tune on the test set.


def stratified_subject_kfold(subject_label: pd.Series, k: int, seed: int) -> dict:
    """
    Partition subjects into k folds, stratified on the subject-level label.

    Returns {subject_id: fold_index}. Deterministic given (subjects, k, seed).

    Within each label stratum, subjects are ranked by the same hash used for the
    holdout split -- so a subject's position depends only on its own id and the
    seed, and adding subjects later does not reshuffle the ones already there --
    and then dealt round-robin into folds. Dealing (rather than slicing into k
    contiguous blocks) keeps each stratum spread as evenly as the arithmetic
    allows: with 31 positives and 5 folds every fold gets 6 or 7, never 5 or 8.

    Each stratum starts at its own rotating offset, derived from the seed and the
    class value. Without it every stratum would start dealing at fold 0, so with
    several small strata fold 0 quietly accumulates all of them.
    """
    if k < 2:
        raise ValueError(f"cross-validation needs k >= 2, got {k}")
    assignment: dict = {}
    for lab, group in subject_label.groupby(subject_label):
        ranked = sorted(sorted(group.index.astype(str)),
                        key=lambda s: hash_unit(s, seed))
        offset = int(hashlib.sha1(f"{seed}|stratum|{lab}".encode("utf-8")).hexdigest()[:8], 16) % k
        for i, s in enumerate(ranked):
            assignment[s] = (i + offset) % k
    return assignment


def carve_inner_validation(subject_label: pd.Series, seed: int, fold: int,
                           inner_val_frac: float) -> set:
    """
    Choose the inner validation subjects for one outer fold's training set.

    Stratified and deterministic, with the fold index folded into the seed so
    that different outer folds do not select the same subjects (they would if the
    seed were shared, because the ranking is a pure function of subject id and
    seed). Never takes the last subject of a stratum, so a stratum can never be
    emptied out of inner-training.
    """
    picks: set = set()
    for lab, group in subject_label.groupby(subject_label):
        ranked = sorted(sorted(group.index.astype(str)),
                        key=lambda s: hash_unit(s, seed + 10_007 * (fold + 1)))
        n_val = min(int(round(len(ranked) * inner_val_frac)), max(0, len(ranked) - 1))
        picks.update(ranked[:n_val])
        del lab
    return picks


def cv_eligible_mask(df: pd.DataFrame) -> pd.Series:
    """
    Rows a fold may legitimately contain: readable file, real label.

    The cohort tables deliberately keep corrupt files (h5_ok=0) and on-disk
    files with no label (label=-1) so the loss stays visible. Neither can be
    scored, so folding them in would inflate the fold sizes this stage is
    supposed to be reporting honestly.
    """
    ok = df["h5_ok"].astype(int) == 1
    lab = df["label"].astype(int) >= 0
    return ok & lab


def add_cv_folds(df: pd.DataFrame, cohort: str, k: int, seed: int,
                 inner_val_frac: float) -> pd.DataFrame:
    """
    Add `cv_fold` and one `cv<j>_split` column per fold.

    cv_fold      the fold in whose TEST set this row's subject sits, or -1 for a
                 subject with no scorable row (corrupt h5 / unlabelled).
    cv<j>_split  'training' | 'validation' | 'test' for fold j, or '' for an
                 ineligible row.

    Fold membership is a property of the SUBJECT and is written onto every row
    that subject owns, which is what makes "no subject spans two folds" checkable
    at row level. The official-split columns are not touched.
    """
    out = df.copy()
    elig = cv_eligible_mask(out)
    out["cv_fold"] = -1
    for j in range(k):
        out[f"cv{j}_split"] = ""
    out["cv_k"] = int(k)
    out["cv_seed"] = int(seed)
    out["cv_inner_val_frac"] = float(inner_val_frac)

    if not elig.any():
        return out

    sub_lab = out.loc[elig].groupby("subject_id")["label"].max()
    folds = stratified_subject_kfold(sub_lab, k, seed)
    subj = out["subject_id"].astype(str)

    # -1 stays on rows whose subject has nothing scorable at all; a subject with
    # a MIX of usable and unusable rows keeps its fold on every row, because the
    # fold is a property of the subject and downstream code filters h5_ok/label
    # exactly as it already does for the holdout splits.
    out.loc[subj.isin(folds), "cv_fold"] = subj[subj.isin(folds)].map(folds).astype(int)

    for j in range(k):
        test_subjects = {s for s, f in folds.items() if f == j}
        train_subjects = {s for s, f in folds.items() if f != j}
        inner_val = carve_inner_validation(
            sub_lab.loc[sorted(train_subjects)], seed, j, inner_val_frac)
        assign = {}
        for s in test_subjects:
            assign[s] = "test"
        for s in train_subjects:
            assign[s] = "validation" if s in inner_val else "training"
        col = f"cv{j}_split"
        sel = elig & subj.isin(assign)
        out.loc[sel, col] = subj[sel].map(assign)
    return out


def assert_cv_partition(df: pd.DataFrame, cohort: str, k: int) -> None:
    """
    Hard failure if the K-fold partition is not a partition.

    Checks, in order:
      1. a subject never carries two different cv_fold values;
      2. per fold, the training / validation / test subject sets are pairwise
         disjoint and together cover every eligible subject;
      3. every eligible subject is 'test' in exactly ONE fold;
      4. cv_fold agrees with the fold whose column says 'test';
      5. patient_id never spans folds either (subject_id is a coarsening of
         patient_id, so this must follow -- if it does not, the subject mapping
         itself is broken).
    """
    if "cv_fold" not in df.columns:
        raise AssertionError(f"[{cohort}] add_cv_folds did not run")
    elig = cv_eligible_mask(df)
    sub = df.loc[elig]
    if sub.empty:
        return

    spanning = sub.groupby("subject_id")["cv_fold"].nunique()
    bad = spanning[spanning > 1]
    if len(bad):
        raise AssertionError(
            f"[{cohort}] CV LEAKAGE: {len(bad)} subject(s) carry more than one "
            f"cv_fold: {list(bad.index)[:10]}")

    all_subjects = set(sub["subject_id"].astype(str))
    test_counts = {s: 0 for s in all_subjects}
    for j in range(k):
        col = f"cv{j}_split"
        if col not in sub.columns:
            raise AssertionError(f"[{cohort}] missing fold column {col}")
        sets = {}
        for role in ("training", "validation", "test"):
            sets[role] = set(sub.loc[sub[col] == role, "subject_id"].astype(str))
        for a, b in (("training", "validation"), ("training", "test"),
                     ("validation", "test")):
            overlap = sets[a] & sets[b]
            if overlap:
                raise AssertionError(
                    f"[{cohort}] CV LEAKAGE in fold {j}: {len(overlap)} subject(s) "
                    f"in both {a} and {b}: {sorted(overlap)[:10]}")
        covered = sets["training"] | sets["validation"] | sets["test"]
        if covered != all_subjects:
            missing = sorted(all_subjects - covered)
            raise AssertionError(
                f"[{cohort}] fold {j} does not cover every eligible subject; "
                f"{len(missing)} unassigned, e.g. {missing[:10]}")
        for s in sets["test"]:
            test_counts[s] += 1
        # cv_fold must be the fold that calls this subject 'test'.
        wrong = sub.loc[(sub[col] == "test") & (sub["cv_fold"].astype(int) != j)]
        if len(wrong):
            raise AssertionError(
                f"[{cohort}] fold {j}: {len(wrong)} row(s) marked test but carry "
                f"cv_fold={sorted(set(wrong['cv_fold']))}")

    not_once = {s: c for s, c in test_counts.items() if c != 1}
    if not_once:
        raise AssertionError(
            f"[{cohort}] {len(not_once)} subject(s) are not a test subject in "
            f"exactly one fold: {dict(list(not_once.items())[:10])}")

    for j in range(k):
        pspan = sub.groupby("patient_id")[f"cv{j}_split"].nunique()
        pbad = pspan[pspan > 1]
        if len(pbad):
            raise AssertionError(
                f"[{cohort}] CV LEAKAGE: {len(pbad)} patient(s) span folds of "
                f"cv{j}_split: {list(pbad.index)[:10]}")


def design_kind(df: pd.DataFrame) -> dict:
    """
    Is the target a BETWEEN-subject or a WITHIN-subject (paired) variable?

    This is not a detail. s06's C1 counts positive clusters as subjects whose max
    label is 1, and negative clusters as the remainder -- i.e. subjects with no
    positive row at all. That arithmetic assumes "a subject with none of the
    positive class" exists. It does for prostate and breast (36 of 67 prostate_t2
    subjects have no positive slice) and for brain (one file per patient). It
    does NOT for knee: every one of the 96 knee patients was scanned both with
    and without fat suppression, so every subject's max is 1, the negative
    cluster count is 0 by construction, and C1 returns MISSING for every K.
    Reporting that as "the folds are too small" would be false.
    """
    elig = cv_eligible_mask(df)
    sub = df.loc[elig]
    if sub.empty:
        return {"paired": False, "n_subjects": 0, "n_mixed_subjects": 0,
                "n_homogeneous_subjects": 0}
    per_subject = sub.groupby("subject_id")["label"].nunique()
    n_mixed = int((per_subject > 1).sum())
    n_sub = int(len(per_subject))
    return {
        "paired": bool(n_sub > 0 and n_mixed == n_sub),
        "n_subjects": n_sub,
        "n_mixed_subjects": n_mixed,
        "n_homogeneous_subjects": n_sub - n_mixed,
    }


def fold_cluster_counts(df: pd.DataFrame, k: int) -> list:
    """
    Per-fold test-set composition, in the units s04/s06 actually gate on.

    A 'cluster' in s06 is one independent test SUBJECT (s04 resamples subjects,
    not slices), and its class is the MAX over that subject's scorable rows --
    the same rule cluster_bootstrap_auc uses. That arithmetic is reproduced here
    exactly, including its lossiness, so what is printed is what the gate will
    see rather than a friendlier version of it.

    `clusters_containing_class` is reported alongside because for a paired target
    the max-collapse throws away the answer: it counts the subjects that actually
    supply at least one row of each class, which is what a paired analysis uses
    and what makes the knee fold sizes interpretable.
    """
    elig = cv_eligible_mask(df)
    sub = df.loc[elig]
    row_classes = sorted(int(c) for c in sub["label"].unique()) if len(sub) else []
    binary = set(row_classes) <= {0, 1}
    design = design_kind(df)

    rows = []
    for j in range(k):
        col = f"cv{j}_split"
        te = sub[sub[col] == "test"]
        tr = sub[sub[col] == "training"]
        va = sub[sub[col] == "validation"]
        smax = te.groupby("subject_id")["label"].max()
        n_cl = int(len(smax))

        if binary:
            # Exactly s06's arithmetic: pos = subjects with any positive row,
            # neg = every other subject (NOT "subjects with a negative row").
            pos = int((smax == 1).sum())
            cluster_classes = {"0": n_cl - pos, "1": pos}
        else:
            vc = smax.value_counts().to_dict()
            cluster_classes = {str(c): int(vc.get(c, 0)) for c in row_classes}
        min_class = int(min(cluster_classes.values())) if cluster_classes else 0
        missing = [c for c, n in cluster_classes.items() if n == 0]

        rows.append({
            "fold": j,
            "n_test_clusters": n_cl,
            "n_test_rows": int(len(te)),
            "n_train_clusters": int(tr["subject_id"].nunique()),
            "n_val_clusters": int(va["subject_id"].nunique()),
            "n_train_rows": int(len(tr)),
            "n_val_rows": int(len(va)),
            # s06's view: one class per subject, taken as the max.
            "class_clusters": cluster_classes,
            # The paired view: subjects supplying at least one row of each class.
            "clusters_containing_class": {
                str(c): int(te.loc[te["label"] == c, "subject_id"].nunique())
                for c in row_classes},
            "class_rows": {str(c): int((te["label"] == c).sum()) for c in row_classes},
            "min_class_clusters": min_class,
            "missing_classes": missing,
            "paired_design": design["paired"],
            "clears_s06_gate": bool(n_cl >= MIN_CLUSTERS_C1
                                    and min_class >= MIN_CLASS_CLUSTERS_C1),
        })
    return rows


def viable_class_restriction(df: pd.DataFrame, k: int, seed: int,
                             inner_val_frac: float) -> dict:
    """
    Which target classes are large enough to survive a K-fold gate, and what does
    the cohort look like if the analysis is restricted to them?

    For a multi-class target, a class needs at least K * MIN_CLASS_CLUSTERS_C1
    subjects cohort-wide before ANY K-fold can put MIN_CLASS_CLUSTERS_C1 of them
    in every test fold. Brain has ten coil counts, four of which have 1, 1, 1 and
    4 subjects; no split and no extra fold count can rescue those, so "5-fold
    fails" needs the follow-up "and here is what does". This computes the
    restricted cohort and re-runs the gate on it, so the recommendation is
    measured rather than asserted.
    """
    elig = cv_eligible_mask(df)
    sub = df.loc[elig]
    per_class = sub.groupby("label")["subject_id"].nunique()
    need = k * MIN_CLASS_CLUSTERS_C1
    keep = sorted(int(c) for c in per_class[per_class >= need].index)
    drop = sorted(int(c) for c in per_class[per_class < need].index)
    out = {
        "k": k, "min_subjects_per_class_needed": int(need),
        "classes_kept": keep, "classes_dropped": drop,
        "subjects_per_class": {str(c): int(n) for c, n in per_class.items()},
        "restricted_rows": 0, "restricted_subjects": 0,
        "restricted_fraction_of_rows": 0.0, "restricted_clears_gate": False,
    }
    if not keep or not drop:
        return out
    rest = sub[sub["label"].isin(keep)].copy()
    if rest.empty or rest["label"].nunique() < 2:
        return out
    rest = add_cv_folds(rest, "restricted", k, seed, inner_val_frac)
    counts = fold_cluster_counts(rest, k)
    out.update({
        "restricted_rows": int(len(rest)),
        "restricted_subjects": int(rest["subject_id"].nunique()),
        "restricted_fraction_of_rows": round(len(rest) / max(1, len(sub)), 4),
        "restricted_clears_gate": all(c["clears_s06_gate"] for c in counts),
        "restricted_min_test_clusters": min(c["n_test_clusters"] for c in counts),
        "restricted_min_class_clusters": min(c["min_class_clusters"] for c in counts),
    })
    return out


def largest_k_clearing_gate(df: pd.DataFrame, seed: int, inner_val_frac: float,
                            k_max: int = 12) -> dict:
    """
    For each candidate K, does every fold clear the s06 gate?

    Reported so that "5-fold does not clear it" is never the end of the sentence.
    Smaller K means bigger test folds, so the set of K that clears is downward
    closed and the largest one is the most efficient choice. Returns the per-K
    verdicts plus the recommendation.
    """
    elig = cv_eligible_mask(df)
    sub = df.loc[elig]
    n_subjects = int(sub["subject_id"].nunique())
    per_k = {}
    for k in range(2, max(3, k_max + 1)):
        if k > n_subjects:
            break
        tmp = add_cv_folds(sub, "probe", k, seed, inner_val_frac)
        counts = fold_cluster_counts(tmp, k)
        per_k[k] = {
            "all_folds_clear": all(c["clears_s06_gate"] for c in counts),
            "min_test_clusters": min(c["n_test_clusters"] for c in counts),
            "min_class_clusters": min(c["min_class_clusters"] for c in counts),
        }
    clearing = [k for k, v in per_k.items() if v["all_folds_clear"]]
    return {"per_k": per_k, "clearing": clearing,
            "largest_clearing": max(clearing) if clearing else None,
            "n_subjects": n_subjects}


def assert_no_subject_spans_splits(df: pd.DataFrame, cohort: str) -> None:
    """Hard failure if any subject appears in more than one fold of any split column."""
    for col in ("official_split", "fallback_split", "split", "split_with_val"):
        if col not in df.columns:
            continue
        sub = df[[c for c in ("subject_id", col)]].dropna()
        sub = sub[sub[col].astype(str) != ""]
        if sub.empty:
            continue
        spanning = sub.groupby("subject_id")[col].nunique()
        bad = spanning[spanning > 1]
        if len(bad):
            raise AssertionError(
                f"[{cohort}] LEAKAGE: {len(bad)} subject(s) span multiple folds of "
                f"'{col}': {list(bad.index)[:10]}"
            )
    # Same check on patient_id, which must be a subset relationship of subject_id.
    for col in ("split", "split_with_val"):
        spanning = df.groupby("patient_id")[col].nunique()
        bad = spanning[spanning > 1]
        if len(bad):
            raise AssertionError(
                f"[{cohort}] LEAKAGE: {len(bad)} patient(s) span multiple folds of "
                f"'{col}': {list(bad.index)[:10]}"
            )


# --------------------------------------------------------------------------
# h5 probing: array shapes + scanner metadata
# --------------------------------------------------------------------------

def _strip_ns(tag: str) -> str:
    return tag.split("}")[-1]


def parse_ismrmrd(xml_text: str) -> dict:
    """
    Pull the acquisition-context fields out of an ISMRMRD header.

    These are exactly the variables that could let a phase-only classifier cheat:
    if malignant scans came off a different scanner, a different coil count, or a
    different protocol, phase encodes that directly.
    """
    out = {}
    try:
        root = ET.fromstring(xml_text)
    except ET.ParseError:
        return out

    def text(path_tags):
        node = root
        for want in path_tags:
            nxt = None
            for child in node:
                if _strip_ns(child.tag) == want:
                    nxt = child
                    break
            if nxt is None:
                return None
            node = nxt
        return (node.text or "").strip()

    fields = {
        "vendor": ["acquisitionSystemInformation", "systemVendor"],
        "scanner_model": ["acquisitionSystemInformation", "systemModel"],
        "field_strength_T": ["acquisitionSystemInformation", "systemFieldStrength_T"],
        "receiver_channels": ["acquisitionSystemInformation", "receiverChannels"],
        "institution": ["acquisitionSystemInformation", "institutionName"],
        "device_id": ["acquisitionSystemInformation", "deviceID"],
        "protocol_name": ["measurementInformation", "protocolName"],
        "patient_position": ["measurementInformation", "patientPosition"],
        "study_time": ["studyInformation", "studyTime"],
        "TR": ["sequenceParameters", "TR"],
        "TE": ["sequenceParameters", "TE"],
        "TI": ["sequenceParameters", "TI"],
        "flip_angle_deg": ["sequenceParameters", "flipAngle_deg"],
        "sequence_type": ["sequenceParameters", "sequence_type"],
        "echo_spacing": ["sequenceParameters", "echo_spacing"],
        "enc_matrix_x": ["encoding", "encodedSpace", "matrixSize", "x"],
        "enc_matrix_y": ["encoding", "encodedSpace", "matrixSize", "y"],
        "enc_matrix_z": ["encoding", "encodedSpace", "matrixSize", "z"],
        "recon_matrix_x": ["encoding", "reconSpace", "matrixSize", "x"],
        "recon_matrix_y": ["encoding", "reconSpace", "matrixSize", "y"],
        "fov_x_mm": ["encoding", "encodedSpace", "fieldOfView_mm", "x"],
        "fov_y_mm": ["encoding", "encodedSpace", "fieldOfView_mm", "y"],
        "fov_z_mm": ["encoding", "encodedSpace", "fieldOfView_mm", "z"],
    }
    for name, path in fields.items():
        val = text(path)
        if val is None:
            continue
        try:
            out[name] = float(val) if re.fullmatch(r"-?\d+(\.\d+)?", val) else val
        except ValueError:
            out[name] = val
    return out


def probe_file(path: Path) -> dict:
    """
    Open one h5 read-only and record shapes, attrs and scanner metadata.

    Returns a dict that always contains h5_ok; on failure it also carries
    h5_error. Nine prostate files on this drive fail here and that is a real,
    reportable data loss -- not something to swallow.
    """
    rec = {"h5_ok": 1, "h5_error": "", "file_size_bytes": None}
    try:
        rec["file_size_bytes"] = path.stat().st_size
    except OSError:
        pass
    try:
        with h5py.File(path, "r") as f:
            keys = list(f.keys())
            rec["h5_keys"] = "|".join(keys)
            for key in keys:
                try:
                    shape = f[key].shape
                except Exception:  # noqa: BLE001
                    continue
                if key == "kspace":
                    rec["kspace_shape"] = str(tuple(shape))
                elif key == "temptv":
                    rec["temptv_shape"] = str(tuple(shape))
                elif key == "coil_sens_maps":
                    rec["has_sens_maps"] = 1
                    rec["sens_shape"] = str(tuple(shape))
                elif key == "reconstruction_rss":
                    # Present on all 454+ brain and all 199 knee files. It is the
                    # vendor's own magnitude image, which is what makes the
                    # reconstruction check possible for these cohorts.
                    rec["has_recon_rss"] = 1
                    rec["recon_rss_shape"] = str(tuple(shape))
            rec.setdefault("has_recon_rss", 0)
            rec.setdefault("has_sens_maps", 0)

            for k, v in f.attrs.items():
                if isinstance(v, bytes):
                    v = v.decode("utf-8", errors="replace")
                if isinstance(v, np.generic):
                    v = v.item()
                rec[f"attr_{k}"] = v

            if "ismrmrd_header" in keys:
                raw = f["ismrmrd_header"][()]
                xml_text = raw.decode("utf-8", errors="replace") if isinstance(raw, bytes) else str(raw)
                rec.update(parse_ismrmrd(xml_text))
    except Exception as exc:  # noqa: BLE001
        rec["h5_ok"] = 0
        rec["h5_error"] = f"{type(exc).__name__}: {exc}"[:200]
    return rec


def probe_many(paths, label: str, enabled: bool = True) -> pd.DataFrame:
    rows = []
    n = len(paths)
    for i, p in enumerate(paths, 1):
        rec = {"file": str(p)}
        if enabled:
            rec.update(probe_file(Path(p)))
        else:
            rec.update({"h5_ok": 1, "h5_error": "", "probed": 0})
        rows.append(rec)
        if enabled and (i % 25 == 0 or i == n):
            print(f"    probed {i}/{n} {label} files")
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------
# Prostate
# --------------------------------------------------------------------------

def build_prostate(kind: str, probe: bool, seed: int, val_frac: float,
                   test_frac: float, cv_folds: int = 5, cv_seed: int = None,
                   inner_val_frac: float = 0.2) -> tuple:
    """
    kind: 'dwi' (AXDIFF) or 't2' (AXT2). Returns (cohort_df, accounting dict).
    """
    assert kind in ("dwi", "t2")
    acq_tag = "AXDIFF" if kind == "dwi" else "AXT2"
    csv_name = "dwi_slice_level_labels.csv" if kind == "dwi" else "t2_slice_level_labels.csv"
    cohort = f"prostate_{kind}"

    labels = pd.read_csv(PROSTATE_LABEL_DIR / csv_name)
    labels = labels.drop(columns=[c for c in labels.columns if c.startswith("Unnamed")])
    vol = pd.read_csv(PROSTATE_LABEL_DIR / "volume_exam_labels.csv")
    vol = vol.drop(columns=[c for c in vol.columns if c.startswith("Unnamed")])

    disk = {p.name: p for p in iter_h5(DATA_ROOT / "prostate") if acq_tag in p.name}
    label_files = set(labels["fastmri_rawfile"].unique())
    disk_files = set(disk)

    missing_files = sorted(label_files - disk_files)      # labelled, not on disk
    unlabelled_files = sorted(disk_files - label_files)   # on disk, not labelled

    df = labels[labels["fastmri_rawfile"].isin(disk_files)].copy()
    df = df.merge(vol, on="fastmri_pt_id", how="left")

    df["cohort"] = cohort
    df["patient_id"] = "prostate_" + df["fastmri_pt_id"].astype(int).map("{:04d}".format)
    df["subject_id"] = df["patient_id"]           # one exam per prostate patient
    df["file"] = df["fastmri_rawfile"].map(lambda n: str(disk[n]))
    df["slice_1based"] = df["slice"].astype(int)
    df["slice"] = df["slice_1based"] - 1          # 0-based index into the h5 slice axis
    df["raw_label"] = df["PIRADS"].astype(int)
    df["label"] = (df["raw_label"] >= 3).astype(int)
    df["official_split"] = df["data_split"].astype(str)
    df["acq"] = acq_tag
    # Stamped so the brain/knee confound tables, which carry
    # label_kind='confound_target', are distinguishable from a real diagnosis by
    # a column rather than by the reader remembering which cohort is which.
    df["label_kind"] = "diagnosis"
    df["has_tumour_label"] = 1
    df = df.rename(columns={"PIRADS": "pirads", "fastmri_pt_id": "fastmri_pt_id"})
    df = df.drop(columns=["data_split", "fastmri_rawfile"])

    # Rows on disk with no label at all: emitted so the loss is visible.
    for fname in unlabelled_files:
        df = pd.concat([df, pd.DataFrame([{
            "cohort": cohort, "patient_id": f"UNLABELLED_{fname}",
            "subject_id": f"UNLABELLED_{fname}", "file": str(disk[fname]),
            "slice": -1, "slice_1based": -1, "label": -1, "raw_label": -1,
            "official_split": "", "folder": "", "acq": acq_tag,
            "label_kind": "diagnosis", "has_tumour_label": 1,
        }])], ignore_index=True)

    probes = probe_many(sorted(df["file"].unique()), cohort, enabled=probe)
    df = df.merge(probes, on="file", how="left")
    if "h5_ok" not in df:
        df["h5_ok"] = 1
    df["h5_ok"] = df["h5_ok"].fillna(0).astype(int)

    # Verify the slice-index assumption against the real array (readable files).
    slice_mismatch = []
    if probe and "kspace_shape" in df.columns:
        for fname, g in df[df["h5_ok"] == 1].groupby("file"):
            shape_str = g["kspace_shape"].iloc[0]
            if not isinstance(shape_str, str):
                continue
            n_h5 = int(shape_str.strip("()").split(",")[1])
            n_lab = int(g["slice_1based"].max())
            if n_h5 != n_lab or int(g["slice_1based"].min()) != 1 or len(g) != n_lab:
                slice_mismatch.append({"file": fname, "kspace_slices": n_h5,
                                       "labelled_slices": n_lab, "rows": len(g)})

    df = finish_cohort(df, cohort, seed, val_frac, test_frac,
                       cv_folds=cv_folds, cv_seed=cv_seed,
                       inner_val_frac=inner_val_frac)

    acct = {
        "cohort": cohort,
        "label_rows_total": int(len(labels)),
        "label_files_total": len(label_files),
        "disk_files_total": len(disk_files),
        "labelled_files_without_data": len(missing_files),
        "labelled_rows_without_data": int(labels[~labels["fastmri_rawfile"].isin(disk_files)].shape[0]),
        "disk_files_without_label": len(unlabelled_files),
        "missing_files_examples": missing_files[:10],
        "unlabelled_files": unlabelled_files,
        "slice_index_mismatches": slice_mismatch,
    }
    return df, acct


# --------------------------------------------------------------------------
# Breast
# --------------------------------------------------------------------------

def build_breast(probe: bool, seed: int, val_frac: float, test_frac: float,
                 dirs=BREAST_DIRS, cv_folds: int = 5, cv_seed: int = None,
                 inner_val_frac: float = 0.2) -> tuple:
    cohort = "breast"
    labels = pd.read_excel(BREAST_LABEL_XLSX)
    unknown = [c for c in labels.columns if c not in BREAST_COLMAP]
    if unknown:
        print(f"  WARNING: unmapped breast label columns kept verbatim: {unknown}")
    labels = labels.rename(columns=BREAST_COLMAP)
    labels["patient_id"] = labels["patient_coded_name"].astype(str).str.strip()

    rows = []
    for d in dirs:
        root = DATA_ROOT / d
        if not root.exists():
            print(f"  NOTE: breast tree {root} not present, skipping")
            continue
        for p in iter_h5(root):
            m = BREAST_FILE_RE.match(p.name)
            if not m:
                print(f"  WARNING: unparseable breast filename, skipped: {p}")
                continue
            rows.append({
                "patient_id": f"fastMRI_breast_{m.group(1)}",
                "acq": int(m.group(2)),
                "file": str(p),
                "folder": p.parent.name,
                "source_dir": d,
            })
    disk = pd.DataFrame(rows)
    if disk.empty:
        raise RuntimeError("no breast h5 files found on disk")

    dup = disk.duplicated(subset=["patient_id", "acq"]).sum()
    if dup:
        raise AssertionError(f"[breast] {dup} duplicate (patient_id, acq) pairs across {dirs}")

    label_pts = set(labels["patient_id"])
    disk_pts = set(disk["patient_id"])
    missing_pts = sorted(label_pts - disk_pts)
    unlabelled_pts = sorted(disk_pts - label_pts)

    df = disk.merge(labels, on="patient_id", how="left")
    df["cohort"] = cohort
    df["slice"] = -1          # breast rows are whole acquisitions, not slices
    df["slice_1based"] = -1
    df["raw_label"] = df["lesion_status"]
    df["label"] = (df["lesion_status"] == 1).astype(int)
    df.loc[df["lesion_status"].isna(), "label"] = -1
    df["label"] = df["label"].astype(int)
    df["label_kind"] = "diagnosis"
    df["has_tumour_label"] = 1
    df["official_split"] = df["official_split_raw"].map({0: "training", 1: "test"}).fillna("")

    # subject_id collapses the repeat-scan pairs: two different coded names with
    # the same nonzero repeat_group are the SAME physical person and must never
    # land on opposite sides of a split.
    df["subject_id"] = np.where(
        df["repeat_group"].fillna(0).astype(int) > 0,
        "breast_repeat_group_" + df["repeat_group"].fillna(0).astype(int).astype(str),
        df["patient_id"],
    )

    probes = probe_many(sorted(df["file"].unique()), cohort, enabled=probe)
    df = df.merge(probes, on="file", how="left")
    df["h5_ok"] = df.get("h5_ok", pd.Series(1, index=df.index)).fillna(0).astype(int)

    # Radial spoke/partition count differs across files (83 / 90 / 98) -- a
    # per-file acquisition difference, therefore a candidate confound.
    if "kspace_shape" in df.columns:
        df["kspace_last_dim"] = df["kspace_shape"].map(
            lambda s: int(s.strip("()").split(",")[-1]) if isinstance(s, str) else np.nan
        )

    # Repeat groups whose members straddle the official split (a real hazard in
    # the full 300-patient sheet) are reported even when only one member is here.
    span = (df[df["official_split"] != ""]
            .groupby("subject_id")["official_split"].nunique())
    official_span = sorted(span[span > 1].index)

    df = finish_cohort(df, cohort, seed, val_frac, test_frac,
                       cv_folds=cv_folds, cv_seed=cv_seed,
                       inner_val_frac=inner_val_frac)

    acct = {
        "cohort": cohort,
        "label_rows_total": int(len(labels)),
        "label_patients_total": len(label_pts),
        "disk_acquisitions_total": int(len(disk)),
        "disk_patients_total": len(disk_pts),
        "labelled_patients_without_data": len(missing_pts),
        "disk_patients_without_label": len(unlabelled_pts),
        "unlabelled_patients": unlabelled_pts,
        "missing_patients_examples": missing_pts[:10],
        "subjects_spanning_official_split": official_span,
        "source_dirs": list(dirs),
    }
    return df, acct


# --------------------------------------------------------------------------
# Confound cohorts: brain and knee
# --------------------------------------------------------------------------

def _shape_tuple(shape_str):
    """Parse the '(a, b, c, d)' string probe_file records into a tuple of ints."""
    if not isinstance(shape_str, str) or not shape_str.strip():
        return ()
    try:
        return tuple(int(float(x)) for x in shape_str.strip("() ").split(",") if x.strip())
    except ValueError:
        return ()


def discover_confound_files(organ: str, dirs=CONFOUND_SEARCH_DIRS) -> pd.DataFrame:
    """
    Find every file BELONGING TO `organ` anywhere under the searched trees.

    Membership is by basename pattern, not by directory, because the two
    disagree: knee/val holds a copy of the whole brain release. Every discovered
    path is returned -- including the duplicates -- so that dedup is an explicit,
    reported step rather than an invisible side effect of which glob ran first.
    """
    rx = CONFOUND_FILE_RE[organ]
    rows = []
    for d in dirs:
        root = DATA_ROOT / d
        if not root.exists():
            print(f"  NOTE: tree {root} not present, skipping")
            continue
        for p in iter_h5(root):
            m = rx.match(p.name)
            if not m:
                continue
            rows.append({
                "file": str(p),
                "basename": p.name,
                "dir_organ": d,
                "folder": p.parent.name,
                "source_dir": d,
                "canonical_tree": int(d == organ),
                **{f"name_{k}": v for k, v in (m.groupdict() or {}).items()},
            })
    return pd.DataFrame(rows)


def dedup_by_basename(disc: pd.DataFrame, organ: str) -> tuple:
    """
    Collapse duplicate copies of one file to a single row, canonical tree wins.

    Returns (kept_df, dropped_records). Sorting by (-canonical_tree, file) makes
    the winner deterministic: the copy under the organ's own tree if there is
    one, otherwise the lexicographically first path. This is the guard that stops
    the same brain patient being counted twice and landing in two folds.
    """
    if disc.empty:
        return disc, []
    ranked = disc.sort_values(["basename", "canonical_tree", "file"],
                              ascending=[True, False, True], kind="mergesort")
    kept = ranked.drop_duplicates("basename", keep="first").reset_index(drop=True)
    dropped = ranked[ranked.duplicated("basename", keep="first")]
    records = [{"basename": r.basename, "dropped_path": r.file,
                "kept_path": kept.loc[kept["basename"] == r.basename, "file"].iloc[0]}
               for r in dropped.itertuples()]
    dup_names = sorted(set(kept["basename"]).intersection(dropped["basename"]))
    print(f"   dedup pass 1 (basename): {len(disc)} discovered -> {len(kept)} kept, "
          f"{len(records)} duplicate copies dropped across {len(dup_names)} basename(s)")
    if kept["basename"].duplicated().any():
        raise AssertionError(
            f"[{organ}] basename dedup failed: "
            f"{sorted(kept.loc[kept['basename'].duplicated(), 'basename'])[:10]}")
    return kept, records


def kspace_fingerprint(path: str) -> str:
    """
    Cheap, deterministic content hash of one k-space array.

    Reads the middle slice of the first coil only -- a few MB, not the whole
    file -- and hashes its bytes. Enough to tell a byte-identical copy from a
    genuinely different scan, which is the only distinction dedup pass 2 needs.
    """
    try:
        with h5py.File(path, "r") as f:
            arr = f["kspace"]
            mid = arr.shape[0] // 2
            sl = np.asarray(arr[mid, 0] if arr.ndim >= 3 else arr[mid])
        return hashlib.sha1(np.ascontiguousarray(sl).tobytes()).hexdigest()
    except Exception as exc:  # noqa: BLE001
        return f"UNREADABLE:{type(exc).__name__}"


def dedup_by_identity(df: pd.DataFrame, organ: str) -> tuple:
    """
    Second safety net: (acquisition, patient_id, kspace shape) collisions.

    Basename dedup only catches copies that kept their name. Two files with
    different names but the same acquisition, the same anonymised patient hash
    and the same k-space shape are CANDIDATE duplicates -- but on this drive that
    key is not sufficient on its own, and dropping on it alone destroys real
    data. One knee patient (ec29c3b8...) owns four files: two CORPD_FBK and two
    CORPDFS_FBK, all (35, 15, 640, 372). The colliding pairs are not copies --
    their k-space differs (max|dk| = 1.5e-3 and 3.2e-3) and their reconstructions
    are uncorrelated at the same slice index (r = -0.02, -0.13) -- so they are a
    repeat exam of one knee, and both scans are usable data.

    So a collision triggers a CONTENT check, and only byte-identical arrays are
    dropped. Distinct scans are kept and reported; they are safe because the
    split unit is the patient, so a repeat exam travels with its original into
    whichever fold that patient lands in and cannot leak across the split.
    """
    if df.empty:
        return df, []
    key = ["acquisition", "attr_patient_id", "kspace_shape"]
    have = [c for c in key if c in df.columns]
    if len(have) < len(key):
        print(f"   dedup pass 2 SKIPPED: missing column(s) "
              f"{sorted(set(key) - set(have))} (was --no-probe used?)")
        return df, []

    ranked = df.sort_values(have + ["file"], kind="mergesort").reset_index(drop=True)
    collide = ranked.duplicated(subset=have, keep=False)
    n_groups = int(ranked.loc[collide].groupby(have, dropna=False).ngroups) if collide.any() else 0
    print(f"   dedup pass 2 (acquisition, patient_id, kspace shape): "
          f"{int(collide.sum())} file(s) in {n_groups} colliding group(s)")

    drop_idx, records, kept_repeats = [], [], []
    if collide.any():
        fps = {i: kspace_fingerprint(ranked.at[i, "file"])
               for i in ranked.index[collide]}
        for _, grp in ranked.loc[collide].groupby(have, dropna=False, sort=False):
            seen: dict = {}
            for i in grp.index:
                fp = fps[i]
                if fp in seen:
                    drop_idx.append(i)
                    records.append({"basename": ranked.at[i, "basename"],
                                    "dropped_path": ranked.at[i, "file"],
                                    "identical_to": ranked.at[seen[fp], "file"],
                                    "acquisition": ranked.at[i, "acquisition"],
                                    "kspace_shape": ranked.at[i, "kspace_shape"],
                                    "reason": "byte-identical k-space"})
                else:
                    if seen:
                        kept_repeats.append({
                            "basename": ranked.at[i, "basename"],
                            "patient_id_attr": str(ranked.at[i, "attr_patient_id"])[:16],
                            "acquisition": ranked.at[i, "acquisition"],
                            "kspace_shape": ranked.at[i, "kspace_shape"]})
                    seen[fp] = i

    kept = (ranked.drop(index=drop_idx)
            .sort_values("basename", kind="mergesort").reset_index(drop=True))
    print(f"      {len(records)} byte-identical copy/copies dropped, "
          f"{len(kept_repeats)} distinct repeat scan(s) KEPT "
          f"({len(df)} -> {len(kept)} files)")
    for rec in records[:10]:
        print(f"      IDENTICAL COPY dropped: {rec['basename']} == "
              f"{Path(rec['identical_to']).name}")
    for rec in kept_repeats[:10]:
        print(f"      REPEAT SCAN kept: {rec['basename']} "
              f"(patient {rec['patient_id_attr']}, {rec['acquisition']}, "
              f"{rec['kspace_shape']}) -- different k-space, travels with its "
              f"patient across splits")
    return kept, records, kept_repeats


def build_confound(organ: str, probe: bool, seed: int, val_frac: float,
                   test_frac: float, cv_folds: int, cv_seed: int,
                   inner_val_frac: float, dirs=CONFOUND_SEARCH_DIRS) -> tuple:
    """
    Build a confound cohort: one row per FILE, label = an acquisition property.

    There is no tumour label here and no way to fabricate one. `label` encodes
    the primary confound target (coil count for brain, contrast for knee) and
    every row is stamped label_kind='confound_target', has_tumour_label=0.
    """
    if organ not in CONFOUND_COHORTS:
        raise ValueError(f"{organ!r} is not a confound cohort")
    if not probe:
        raise ValueError(
            f"[{organ}] --no-probe is not supported for confound cohorts: the "
            "label IS the acquisition metadata, so a table built without opening "
            "the files would have no label at all")

    disc = discover_confound_files(organ, dirs)
    if disc.empty:
        raise RuntimeError(f"no {organ} files found under {[str(DATA_ROOT / d) for d in dirs]}")

    per_dir = disc.groupby("dir_organ").size().to_dict()
    print(f"   discovered {len(disc)} {organ}-named file(s) by tree: {per_dir}")
    misfiled = int((disc["canonical_tree"] == 0).sum())
    if misfiled:
        print(f"   !! {misfiled} {organ}-named file(s) sit OUTSIDE the {organ}/ tree")

    kept, dup_basename_records = dedup_by_basename(disc, organ)

    probes = probe_many(sorted(kept["file"].unique()), organ, enabled=True)
    df = kept.merge(probes, on="file", how="left")
    df["h5_ok"] = df.get("h5_ok", pd.Series(1, index=df.index)).fillna(0).astype(int)

    # ---- acquisition-derived fields ------------------------------------
    df["acquisition"] = df.get("attr_acquisition", pd.Series("", index=df.index)) \
        .fillna("").astype(str)
    shapes = df["kspace_shape"].map(_shape_tuple)
    # fastMRI brain/knee k-space is (slices, coils, kx, ky).
    df["n_slices"] = shapes.map(lambda t: t[0] if len(t) == 4 else np.nan)
    df["n_coils"] = shapes.map(lambda t: t[1] if len(t) == 4 else np.nan)
    df["matrix_kx"] = shapes.map(lambda t: t[2] if len(t) == 4 else np.nan)
    df["matrix_ky"] = shapes.map(lambda t: t[3] if len(t) == 4 else np.nan)
    df["matrix"] = df.apply(
        lambda r: "" if pd.isna(r["matrix_kx"]) or pd.isna(r["matrix_ky"])
        else f"{int(r['matrix_kx'])}x{int(r['matrix_ky'])}", axis=1)

    bad_shape = df[df["h5_ok"] == 1]["kspace_shape"].map(lambda s: len(_shape_tuple(s)) != 4)
    if bad_shape.any():
        raise AssertionError(
            f"[{organ}] {int(bad_shape.sum())} readable file(s) do not have a "
            f"4-D (slices, coils, kx, ky) k-space; the coil axis cannot be "
            f"identified: "
            f"{df.loc[bad_shape[bad_shape].index, 'basename'].tolist()[:5]}")

    # Coil count from the array must agree with the header's receiverChannels.
    # If it does not, one of them is describing a different scan.
    if "receiver_channels" in df.columns:
        rc = pd.to_numeric(df["receiver_channels"], errors="coerce")
        mism = df[(df["h5_ok"] == 1) & rc.notna() & (rc != df["n_coils"])]
        if len(mism):
            print(f"   !! {len(mism)} file(s) where ismrmrd receiverChannels != "
                  f"k-space coil axis, e.g. "
                  f"{mism[['basename', 'receiver_channels', 'n_coils']].head(3).to_dict('records')}")

    df, dup_identity_records, kept_repeat_scans = dedup_by_identity(df, organ)

    # ---- THE dedup assertion -------------------------------------------
    if df["basename"].duplicated().any():
        dups = sorted(df.loc[df["basename"].duplicated(), "basename"])
        raise AssertionError(
            f"[{organ}] {len(dups)} basename(s) appear twice in the finished "
            f"cohort: {dups[:10]}. knee/val holds a duplicate copy of the whole "
            f"brain release; a cohort containing both copies puts one patient in "
            f"two folds.")

    # ---- identity ------------------------------------------------------
    pid = df.get("attr_patient_id", pd.Series("", index=df.index)).fillna("").astype(str)
    # Fall back to the basename only where the anonymised hash is missing, and
    # say so, rather than quietly inventing a patient.
    no_pid = pid.str.len() == 0
    if no_pid.any():
        print(f"   !! {int(no_pid.sum())} file(s) have no patient_id attribute; "
              f"using basename as the patient id for those")
    df["patient_id"] = np.where(no_pid, organ + "_file_" + df["basename"],
                                organ + "_" + pid.str.slice(0, 16))
    # No repeat-scan structure is documented for these releases, and the
    # anonymised hash is per-exam, so subject == patient here. Emitted anyway so
    # every cohort table has the same split-enforcement column.
    df["subject_id"] = df["patient_id"]

    n_pt = df["patient_id"].nunique()
    if n_pt != len(df):
        rep = df["patient_id"].value_counts()
        multi = rep[rep > 1]
        print(f"   NOTE: {len(multi)} patient_id(s) own more than one file "
              f"({len(df)} files, {n_pt} patients); folds are drawn on the patient, "
              f"so those files always travel together")

    # ---- the confound target(s) ----------------------------------------
    target = CONFOUND_TARGETS[organ]
    tgt_raw = df[target]
    if target == "n_coils":
        tgt_str = tgt_raw.map(lambda v: "" if pd.isna(v) else str(int(v)))
        classes = sorted({int(v) for v in tgt_raw.dropna()})
        code_of = {str(c): i for i, c in enumerate(classes)}
    else:
        tgt_str = tgt_raw.astype(str)
        classes = sorted(set(tgt_str))
        code_of = {c: i for i, c in enumerate(classes)}

    df["target_name"] = target
    df["target_value"] = tgt_str
    df["label_name"] = tgt_str
    df["label"] = tgt_str.map(code_of).fillna(-1).astype(int)
    df.loc[df["h5_ok"] != 1, "label"] = -1
    df["raw_label"] = tgt_str
    df["n_classes"] = len(classes)
    # Stamped on every row so no downstream stage, table or figure can present
    # these cohorts as if they carried a diagnosis.
    df["label_kind"] = "confound_target"
    df["has_tumour_label"] = 0
    df["label_semantics"] = f"{target} (acquisition/hardware property, NOT pathology)"

    for sec in CONFOUND_SECONDARY_TARGETS[organ]:
        vals = df[sec]
        s = (vals.map(lambda v: "" if pd.isna(v) else str(int(v)))
             if sec == "n_coils" else vals.astype(str))
        codes = {c: i for i, c in enumerate(sorted(set(s)))}
        df[f"target_{sec}"] = s
        df[f"target_{sec}_code"] = s.map(codes).astype(int)

    # ---- shared cohort scaffolding -------------------------------------
    df["cohort"] = organ
    df["slice"] = -1            # rows are whole files, not slices
    df["slice_1based"] = -1
    df["acq"] = df["acquisition"]
    # fastMRI ships these as its own validation fold. That is where they came
    # from, not a train/test split for THIS experiment, so it is recorded as
    # provenance and `official_split` is left empty -- which routes `split` to
    # the deterministic subject-level fallback instead.
    df["fastmri_official_fold"] = "val"
    df["official_split"] = ""

    df = finish_cohort(df, organ, seed, val_frac, test_frac,
                       cv_folds=cv_folds, cv_seed=cv_seed,
                       inner_val_frac=inner_val_frac)

    class_counts = (df[df["label"] >= 0]
                    .groupby("label_name").size().sort_index().to_dict())
    acct = {
        "cohort": organ,
        "target": target,
        "target_is_pathology": False,
        "searched_trees": [str(DATA_ROOT / d) for d in dirs],
        "discovered_paths": int(len(disc)),
        "discovered_by_tree": {k: int(v) for k, v in per_dir.items()},
        "files_outside_canonical_tree": misfiled,
        "duplicate_basename_copies_dropped": len(dup_basename_records),
        "duplicate_basename_examples": dup_basename_records[:5],
        "identity_duplicates_dropped": len(dup_identity_records),
        "identity_duplicate_examples": dup_identity_records[:5],
        "repeat_scans_kept": len(kept_repeat_scans),
        "repeat_scan_examples": kept_repeat_scans[:5],
        "files_after_dedup": int(len(df)),
        "n_classes": len(classes),
        "class_counts": {str(k): int(v) for k, v in class_counts.items()},
        "coil_counts": {str(int(k)): int(v) for k, v in
                        df["n_coils"].dropna().value_counts().sort_index().items()},
        "acquisition_counts": {str(k): int(v) for k, v in
                               df["acquisition"].value_counts().items()},
        "matrix_counts": {str(k): int(v) for k, v in
                          df["matrix"].value_counts().items()},
    }
    return df, acct


# --------------------------------------------------------------------------
# Shared finishing: splits, ordering, assertions
# --------------------------------------------------------------------------

def finish_cohort(df: pd.DataFrame, cohort: str, seed: int, val_frac: float,
                  test_frac: float, cv_folds: int = 5, cv_seed: int = None,
                  inner_val_frac: float = 0.2) -> pd.DataFrame:
    df = df.reset_index(drop=True)
    df["official_split"] = df["official_split"].fillna("").astype(str)

    subj_lab = df.groupby("subject_id")["label"].max().clip(lower=0)
    fallback = stratified_subject_split(subj_lab, seed, val_frac, test_frac)
    df["fallback_split"] = df["subject_id"].map(fallback)

    has_official = df["official_split"].str.len() > 0
    df["split"] = np.where(has_official, df["official_split"], df["fallback_split"])
    df["split_source"] = np.where(has_official, "official", "fallback")
    df["split_with_val"] = carve_validation(df, seed, val_frac)

    assert_no_subject_spans_splits(df, cohort)

    # Cross-validation is ADDITIVE. Nothing above this line changes, so the
    # official split stays reportable as a secondary analysis exactly as before.
    df = add_cv_folds(df, cohort, cv_folds,
                      seed if cv_seed is None else cv_seed, inner_val_frac)
    assert_cv_partition(df, cohort, cv_folds)

    df["idx"] = np.arange(len(df))
    lead = [c for c in LEAD_COLS if c in df.columns]
    fold_cols = [f"cv{j}_split" for j in range(cv_folds) if f"cv{j}_split" in df.columns]
    if "cv_fold" in lead:
        at = lead.index("cv_fold") + 1
        lead = lead[:at] + fold_cols + lead[at:]
    else:
        lead = lead + fold_cols
    rest = [c for c in df.columns if c not in lead]
    return df[lead + rest]


# --------------------------------------------------------------------------
# Reporting
# --------------------------------------------------------------------------

def _bal(sub: pd.DataFrame) -> str:
    n = len(sub)
    pos = int((sub["label"] == 1).sum())
    pct = 100.0 * pos / n if n else 0.0
    return f"{pos}/{n} pos ({pct:.1f}%)"


def _is_multiclass(df: pd.DataFrame) -> bool:
    """True when `label` has more than two realised classes (brain coil count)."""
    return int(df.loc[df["label"] >= 0, "label"].nunique()) > 2


def _class_summary(sub: pd.DataFrame, multiclass: bool, name_col: str = "label_name") -> str:
    """
    'pos/total' for a binary cohort, a class histogram for a multi-class one.

    Printing "0/130 pos" for a 10-class coil-count target would be true and
    useless, so the multi-class branch prints the whole histogram instead.
    """
    if not multiclass:
        return _bal(sub)
    n = len(sub)
    key = name_col if name_col in sub.columns else "label"
    counts = sub.loc[sub["label"] >= 0, key].astype(str).value_counts()
    counts = counts.reindex(sorted(counts.index, key=lambda x: (len(x), x)))
    body = " ".join(f"{k}:{int(v)}" for k, v in counts.items())
    return f"n={n} [{body}]"


def confound_screen(usable: pd.DataFrame, confounds: list, unit: str = "patient") -> list:
    """
    Association test between each candidate confound and the label.

    This belongs in stage 1, not stage 4, because if a scanner/site/batch variable
    predicts the label at the patient level then a phase-only network can score a
    high AUC without ever looking at pathology -- and we want to know that BEFORE
    spending GPU hours. Categorical -> chi-square, continuous -> Mann-Whitney.
    Returns a list sorted by p-value, most suspicious first.

    Multi-class targets (the brain confound cohort's 10 coil counts, its 5
    contrasts) take the Kruskal-Wallis branch for continuous confounds instead of
    Mann-Whitney. The old code compared label==1 against label==0 and would have
    silently discarded the other eight classes -- reporting a p-value computed on
    a fifth of the cohort as if it described the whole screen. The binary path is
    unchanged, so the prostate and breast screens return exactly what they did
    before.

    `unit` selects the row on which the test is run:

      'patient'  one row per patient, label = max, confound = first. This is the
                 default and the right unit when the label is a property OF the
                 patient, because slices from one patient are not independent
                 observations of a scanner variable.
      'file'     one row per file. Required when the target varies WITHIN a
                 patient. Every knee patient has both a fat-suppressed and a
                 non-fat-suppressed scan, so collapsing by patient sets every
                 patient's max to 1, leaves one class, and makes the whole screen
                 return "nothing testable" -- which reads as "no confounding
                 found" and is the opposite of the truth. At file level the
                 screen actually asks whether contrast is collinear with scanner
                 model, field strength or matrix size, which is the question the
                 knee experiment depends on.
    """
    from scipy import stats

    if unit == "file":
        cols = ["label"] + [c for c in confounds if c in usable.columns]
        pt = usable[cols].copy()
    else:
        pt = usable.groupby("patient_id").agg(
            label=("label", "max"),
            **{c: (c, "first") for c in confounds},
        )
    pt = pt[pt["label"] >= 0]
    n_classes = int(pt["label"].nunique())
    multiclass = n_classes > 2
    results = []
    for c in confounds:
        v = pt[c]
        if v.dropna().nunique() < 2 or n_classes < 2:
            continue
        try:
            if v.dtype.kind in "if" and v.dropna().nunique() > 6:
                groups = [v[pt["label"] == lab].dropna()
                          for lab in sorted(pt["label"].unique())]
                groups = [g for g in groups if len(g) >= 2]
                if len(groups) < 2:
                    continue
                if multiclass:
                    p = float(stats.kruskal(*groups).pvalue)
                    test = "kruskal_wallis"
                    detail = "medians by class: " + ", ".join(
                        f"{lab}={v[pt['label'] == lab].dropna().median():.4g}"
                        for lab in sorted(pt["label"].unique())
                        if len(v[pt["label"] == lab].dropna()) >= 2)
                else:
                    a = v[pt["label"] == 1].dropna()
                    b = v[pt["label"] == 0].dropna()
                    if len(a) < 2 or len(b) < 2:
                        continue
                    p = float(stats.mannwhitneyu(a, b).pvalue)
                    test = "mann_whitney_u"
                    detail = f"median pos={a.median():.4g} neg={b.median():.4g}"
            else:
                ct = pd.crosstab(v.astype(str).fillna("NA"), pt["label"])
                if ct.shape[0] < 2 or ct.shape[1] < 2:
                    continue
                p = float(stats.chi2_contingency(ct)[1])
                test = "chi2_contingency"
                detail = ct.to_dict()
        except Exception:  # noqa: BLE001
            continue
        results.append({"confound": c, "p_value": p, "test": test, "detail": detail})
    return sorted(results, key=lambda r: r["p_value"])


def report_cv(df: pd.DataFrame, cohort: str, k: int, unit: str,
              seed: int, inner_val_frac: float, multiclass: bool) -> dict:
    """
    Print and return the per-fold composition, and the s06 gate verdict per fold.

    The gate is the reason this section exists, so it is stated per fold and not
    only in aggregate. When 5-fold does not clear it, the K sweep says which K
    would -- "the fold is too small" is only useful with the number attached.
    """
    counts = fold_cluster_counts(df, k)
    design = design_kind(df)
    print(f"-- cross-validation: {k}-fold, stratified, subject level "
          f"(seed {seed}, inner val {inner_val_frac:.0%} of each training split) --")
    print(f"   gate ({GATE_SOURCE}): >= {MIN_CLUSTERS_C1} test clusters "
          f"AND >= {MIN_CLASS_CLUSTERS_C1} per class, where a cluster's class is "
          f"the MAX over its rows")
    print(f"   design: {'WITHIN-SUBJECT (paired)' if design['paired'] else 'between-subject'}"
          f"  -- {design['n_mixed_subjects']}/{design['n_subjects']} subjects carry "
          f"more than one class")
    for c in counts:
        verdict = "CLEARS GATE" if c["clears_s06_gate"] else "BELOW GATE"
        cls = " ".join(f"{a}:{b}" for a, b in c["class_clusters"].items())
        held = " ".join(f"{a}:{b}" for a, b in c["clusters_containing_class"].items())
        print(f"   fold {c['fold']}  test clusters {c['n_test_clusters']:>4} "
              f"| s06 class-of-subject [{cls}] min {c['min_class_clusters']:>3} "
              f"| subjects supplying each class [{held}] "
              f"| test {unit} {c['n_test_rows']:>6} "
              f"| train/val clusters {c['n_train_clusters']:>4}/{c['n_val_clusters']:<4} "
              f"-> {verdict}")
        if c["missing_classes"]:
            print(f"           !! class(es) with ZERO clusters under the s06 "
                  f"max-collapse: {c['missing_classes']}")
    n_clear = sum(1 for c in counts if c["clears_s06_gate"])
    all_clear = n_clear == len(counts)
    print(f"   {n_clear}/{len(counts)} folds clear the s06 gate"
          f"{'  -- ALL FOLDS CLEAR' if all_clear else ''}")

    sweep = largest_k_clearing_gate(df, seed, inner_val_frac)
    blocked_by_design = design["paired"] and not all_clear
    restriction = (viable_class_restriction(df, k, seed, inner_val_frac)
                   if multiclass and not all_clear else None)
    if blocked_by_design:
        # Saying "increase K" here would be wrong: no K can help.
        print(f"   !! {cohort} is a PAIRED cohort -- every subject supplies every "
              f"class -- so s06's cluster class (the max over a subject's rows) "
              f"is the same for all {design['n_subjects']} subjects and the "
              f"negative-cluster count is 0 BY CONSTRUCTION, not by fold size.")
        print(f"      NO value of K fixes this, and neither does more data. Each "
              f"fold already holds "
              f"{min(c['n_test_clusters'] for c in counts)}-"
              f"{max(c['n_test_clusters'] for c in counts)} independent test "
              f"subjects, each contributing both classes "
              f"({min(min(c['class_rows'].values()) for c in counts)}+ rows per "
              f"class per fold), which is a well-powered paired design.")
        print(f"      What has to change is the GATE, not the split: C1 needs a "
              f"paired branch that counts subjects SUPPLYING each class "
              f"(clusters_containing_class above) instead of subjects whose max "
              f"IS each class. Until s06 has that branch, this cohort must be "
              f"reported as descriptive and cannot carry a confirmatory claim.")
    elif not all_clear:
        if sweep["largest_clearing"] is not None:
            print(f"   !! {k}-fold does NOT clear the gate for {cohort}. "
                  f"K in {sweep['clearing']} does; the largest is K="
                  f"{sweep['largest_clearing']}.")
        else:
            print(f"   !! {k}-fold does NOT clear the gate for {cohort}, and "
                  f"NO K in 2..12 does either: with {sweep['n_subjects']} "
                  f"eligible subjects the binding constraint is the cohort, not "
                  f"the fold count.")
        if restriction and restriction["classes_dropped"]:
            print(f"      Cause: {len(restriction['classes_dropped'])} class(es) "
                  f"have fewer than the {restriction['min_subjects_per_class_needed']} "
                  f"subjects that K={k} needs -- "
                  f"{ {str(c): restriction['subjects_per_class'][str(c)] for c in restriction['classes_dropped']} }.")
            if restriction["restricted_clears_gate"]:
                print(f"      RESTRICTING to the {len(restriction['classes_kept'])} "
                      f"well-populated classes {restriction['classes_kept']} keeps "
                      f"{restriction['restricted_rows']}/{sweep['n_subjects']} "
                      f"rows ({restriction['restricted_fraction_of_rows']:.1%}) and "
                      f"CLEARS every fold at K={k} "
                      f"(min {restriction['restricted_min_test_clusters']} test "
                      f"clusters, min {restriction['restricted_min_class_clusters']} "
                      f"per class). That is the recommended primary analysis; the "
                      f"dropped classes stay in the table and are reported "
                      f"descriptively.")
            else:
                print(f"      Restricting to the well-populated classes "
                      f"{restriction['classes_kept']} does NOT clear the gate "
                      f"either.")
        print(f"      Alternatively, pooling the out-of-fold predictions instead "
              f"of scoring each fold separately gives ONE evaluation on all "
              f"{sweep['n_subjects']} subjects (every subject scored by a model "
              f"that never saw it), which clears the cluster count for any K. "
              f"Per-fold numbers then describe stability rather than carrying the "
              f"confirmatory test.")
    for kk, v in sorted(sweep["per_k"].items()):
        print(f"      K={kk:<3} min test clusters {v['min_test_clusters']:>4}  "
              f"min per-class {v['min_class_clusters']:>4}  "
              f"{'clears' if v['all_folds_clear'] else 'below gate'}")

    return {
        "k": k, "seed": seed, "inner_val_frac": inner_val_frac,
        "multiclass": multiclass,
        "design": design,
        "gate_blocked_by_paired_design": blocked_by_design,
        "gate": {"source": GATE_SOURCE, "min_clusters": MIN_CLUSTERS_C1,
                 "min_per_class": MIN_CLASS_CLUSTERS_C1},
        "folds": counts,
        "folds_clearing_gate": n_clear,
        "all_folds_clear_gate": all_clear,
        "k_sweep": {str(a): b for a, b in sweep["per_k"].items()},
        "k_values_clearing_gate": sweep["clearing"],
        "largest_k_clearing_gate": sweep["largest_clearing"],
        "eligible_subjects": sweep["n_subjects"],
        "class_restriction": restriction,
    }


def report_cohort(df: pd.DataFrame, acct: dict) -> dict:
    cohort = acct["cohort"]
    usable = df[df["h5_ok"] == 1]
    unit = ("files" if cohort in CONFOUND_COHORTS
            else "acquisitions" if cohort == "breast" else "slices")
    multiclass = _is_multiclass(usable)

    print()
    print("=" * 78)
    print(f"COHORT: {cohort}")
    if cohort in CONFOUND_COHORTS:
        print(f"  *** CONFOUND COHORT -- NO TUMOUR LABELS EXIST FOR {cohort.upper()}. ***")
        print(f"  *** `label` is {acct['target']}, an acquisition/hardware property. ***")
        print("  *** Any accuracy reported here is evidence about the SCANNER, ***")
        print("  *** never about pathology.                                     ***")
    print("=" * 78)

    print("-- join accounting --")
    for k, v in acct.items():
        if k == "cohort" or isinstance(v, (list, dict)):
            continue
        print(f"   {k:<38} {v}")
    if acct.get("unlabelled_files") or acct.get("unlabelled_patients"):
        print(f"   !! ON DISK WITH NO LABEL: "
              f"{acct.get('unlabelled_files') or acct.get('unlabelled_patients')}")
    if "slice_index_mismatches" in acct:
        if acct["slice_index_mismatches"]:
            print(f"   !! SLICE-INDEX MISMATCHES: {acct['slice_index_mismatches']}")
        else:
            print("   slice index 1..n verified against k-space slice axis   OK")
    else:
        print("   (no slice axis: rows are whole acquisitions)")
    if acct.get("subjects_spanning_official_split"):
        print("   !! subjects straddling the official split (present here): "
              f"{acct['subjects_spanning_official_split']}")

    n_bad = int((df["h5_ok"] == 0).sum())
    print("-- cohort size --")
    print(f"   rows ({unit}):        {len(df)}   (usable, h5_ok=1: {len(usable)})")
    print(f"   unreadable h5 rows:   {n_bad} across "
          f"{df.loc[df['h5_ok'] == 0, 'file'].nunique()} file(s)")
    if n_bad:
        for f in sorted(df.loc[df["h5_ok"] == 0, "file"].unique()):
            print(f"      CORRUPT: {Path(f).name}")
    print(f"   files:                {df['file'].nunique()}   "
          f"(usable {usable['file'].nunique()})")
    print(f"   patients:             {df['patient_id'].nunique()}   "
          f"(usable {usable['patient_id'].nunique()})")
    print(f"   subjects (split unit):{df['subject_id'].nunique()}   "
          f"(usable {usable['subject_id'].nunique()})")
    print(f"   class balance all:    {_class_summary(df, multiclass)}")
    print(f"   class balance usable: {_class_summary(usable, multiclass)}")
    if "raw_label" in df:
        counts = usable["raw_label"].astype(str).value_counts().sort_index().to_dict()
        print(f"   raw_label counts:     {str(counts)[:300]}")
    pl = usable.groupby("patient_id")["label"].max()
    if multiclass:
        print(f"   patient-level classes: "
              f"{pl.value_counts().sort_index().to_dict()}")
    else:
        print(f"   positive patients:    {int((pl == 1).sum())}/{len(pl)}")

    print("-- splits (usable rows only) --")
    split_tables = {}
    for col in ("official_split", "split", "split_with_val", "fallback_split"):
        if col not in usable.columns:
            continue
        print(f"   [{col}]")
        tab = {}
        for name, g in usable.groupby(usable[col].replace("", "(none)")):
            pl_g = g.groupby("patient_id")["label"].max()
            print(f"      {str(name):<12} {unit:<13} {len(g):>6}   "
                  f"{_class_summary(g, multiclass):<44} "
                  f"patients {g['patient_id'].nunique():>4}  "
                  f"pos-patients {int((pl_g == 1).sum()):>4}")
            tab[str(name)] = {
                "rows": int(len(g)),
                "positives": int((g["label"] == 1).sum()),
                "patients": int(g["patient_id"].nunique()),
                "positive_patients": int((pl_g == 1).sum()),
                "class_patients": {str(k): int(v)
                                   for k, v in pl_g.value_counts().sort_index().items()},
            }
        split_tables[col] = tab
        if not multiclass:
            empties = [k for k, v in tab.items() if v["positives"] == 0 and k != "(none)"]
            if empties:
                print(f"      !! WARNING: folds with ZERO positives: {empties} "
                      f"-- AUC is undefined there")

    # Loud degeneracy warnings.
    warnings = []
    if not multiclass:
        test_rows = usable[usable["split"] == "test"]
        if len(test_rows):
            n_test_pos_pt = int(test_rows.groupby("patient_id")["label"].max().sum())
            if n_test_pos_pt < 10:
                warnings.append(
                    f"official test fold has only {n_test_pos_pt} positive patient(s) "
                    f"({int((test_rows['label'] == 1).sum())} positive {unit}); any AUC "
                    f"from it will have an enormous confidence interval")
        if len(usable) and (usable["label"] == 1).mean() < 0.10:
            warnings.append(
                f"severe class imbalance: only "
                f"{100 * (usable['label'] == 1).mean():.1f}% positive {unit}")
    else:
        rare = {str(k): int(v) for k, v in pl.value_counts().items()
                if v < MIN_CLASS_CLUSTERS_C1}
        if rare:
            warnings.append(
                f"{len(rare)} target class(es) have fewer than "
                f"{MIN_CLASS_CLUSTERS_C1} subjects in the WHOLE cohort "
                f"(class code -> subjects: {rare}); no K-fold can put "
                f"{MIN_CLASS_CLUSTERS_C1} of them in every test fold, so those "
                f"classes must be pooled, dropped, or reported as descriptive")
    if usable["patient_id"].nunique() < 100:
        warnings.append(
            f"only {usable['patient_id'].nunique()} patients on disk -- this is a "
            f"small-sample study and per-patient bootstrap CIs are mandatory")
    is_confound_cohort = cohort in CONFOUND_COHORTS
    extra = CONFOUND_COHORT_EXTRA_CONFOUNDS if is_confound_cohort else ()
    confounds = [c for c in ACQUISITION_CONFOUNDS + LABEL_DERIVED_COLS + extra
                 if c in usable.columns]
    print("-- confound columns present --")
    for c in confounds:
        vals = usable[c].dropna()
        nun = vals.nunique()
        if nun == 0:
            desc = "(all missing)"
        elif nun <= 6:
            # astype(str) because some label-sheet columns (e.g. breast
            # 'laterality') mix ints and strings, which makes sort_index raise.
            desc = str(vals.astype(str).value_counts().sort_index().to_dict())
        else:
            desc = f"{nun} distinct values"
        print(f"   {c:<20} {desc[:150]}")

    # Only ACQUISITION_CONFOUNDS are screened. Columns like grade/ER/PR/HER2 and
    # exam_level are derived from the outcome itself, so "associated with the
    # label" is tautological for them and would drown the real signals.
    #
    # On a confound cohort the TARGET is itself an acquisition variable, so its
    # own aliases have to come out too: screening n_coils against a label that
    # IS n_coils returns p = 0 and tells us nothing. What the screen answers here
    # is the question the knee experiment lives or dies on -- is the target
    # collinear with some OTHER acquisition variable, so that "phase predicts
    # contrast" could really be "phase predicts the scanner that happened to run
    # the fat-suppressed protocol"?
    target = acct.get("target")
    excluded = set(TARGET_ALIASES.get(target, ())) if target else set()
    screen_cols = [c for c in ACQUISITION_CONFOUNDS + extra
                   if c in usable.columns and c not in excluded]
    # A paired target is invisible after a patient-level collapse, so the screen
    # has to run per file there or it silently tests nothing at all.
    screen_unit = "file" if design_kind(df)["paired"] else "patient"
    screen = confound_screen(usable, screen_cols, unit=screen_unit)
    print(f"-- acquisition confound vs label, {screen_unit} level "
          "(small p = the confound alone could explain a high AUC) --")
    if screen_unit == "file":
        print("   run per FILE, not per patient: the target varies WITHIN every "
              "subject, so a patient-level collapse leaves one class and tests "
              "nothing")
    if is_confound_cohort:
        print(f"   target = {target}; excluded from the screen as tautological: "
              f"{sorted(excluded & set(usable.columns))}")
        print("   a SUSPECT row here means the target is collinear with another "
              "acquisition variable, so a model could be reading that instead")
    if not screen:
        print("   (nothing testable)")
    for r in screen[:10]:
        flag = "  <<< SUSPECT" if r["p_value"] < 0.05 else ""
        print(f"   {r['confound']:<20} p={r['p_value']:.4g}  "
              f"[{r.get('test', 'n/a')}]{flag}")
        if r["p_value"] < 0.05:
            print(f"       {str(r['detail'])[:220]}")
            if is_confound_cohort:
                warnings.append(
                    f"confound '{r['confound']}' is associated with the TARGET "
                    f"'{target}' at {screen_unit} level (p={r['p_value']:.4g}); a model "
                    f"that predicts the target may be reading '{r['confound']}' "
                    f"instead, so the two cannot be separated in this cohort")
            else:
                warnings.append(
                    f"confound '{r['confound']}' is associated with the label at "
                    f"patient level (p={r['p_value']:.4g}); a phase-only model could "
                    f"score high by reading it instead of pathology")

    # --- cross-validation ------------------------------------------------
    k = int(df["cv_k"].iloc[0]) if "cv_k" in df.columns and len(df) else 0
    cv_report = None
    if k:
        cv_report = report_cv(df, cohort, k, unit,
                              int(df["cv_seed"].iloc[0]),
                              float(df["cv_inner_val_frac"].iloc[0]),
                              multiclass)
        if cv_report["gate_blocked_by_paired_design"]:
            warnings.append(
                f"{cohort} is a PAIRED cohort: all "
                f"{cv_report['design']['n_subjects']} subjects supply every class, "
                f"so s06's C1 sees 0 negative clusters for any K and returns "
                f"MISSING. The folds are NOT too small -- each has "
                f"~{cv_report['folds'][0]['n_test_clusters']} independent subjects "
                f"contributing both classes. s06's C1 needs a paired branch "
                f"counting subjects that SUPPLY each class; until then this "
                f"cohort is descriptive only")
        elif not cv_report["all_folds_clear_gate"]:
            n_bad_folds = k - cv_report["folds_clearing_gate"]
            best = cv_report["largest_k_clearing_gate"]
            restr = cv_report.get("class_restriction")
            if best is not None:
                fix = (f"K={best} clears every fold and should be used for the "
                       f"confirmatory test on this cohort")
            elif restr and restr["restricted_clears_gate"]:
                fix = (f"no K in 2..12 clears every fold because classes "
                       f"{restr['classes_dropped']} are too rare; restricting to "
                       f"classes {restr['classes_kept']} keeps "
                       f"{restr['restricted_fraction_of_rows']:.1%} of the rows "
                       f"and clears every fold at K={k}")
            else:
                fix = ("no K in 2..12 clears every fold, so the confirmatory "
                       "readout must be a single pooled out-of-fold evaluation "
                       f"over all {cv_report['eligible_subjects']} subjects")
            warnings.append(
                f"{n_bad_folds}/{k} CV fold(s) are below the s06 gate "
                f"({MIN_CLUSTERS_C1} clusters, {MIN_CLASS_CLUSTERS_C1} per class); "
                + fix)

    if warnings:
        print("-- WARNINGS --")
        for w in warnings:
            print(f"   !! {w}")

    return {
        "accounting": acct,
        "rows": int(len(df)),
        "rows_usable": int(len(usable)),
        "files": int(df["file"].nunique()),
        "patients": int(df["patient_id"].nunique()),
        "subjects": int(df["subject_id"].nunique()),
        "multiclass_target": bool(multiclass),
        "label_kind": str(df["label_kind"].iloc[0]) if "label_kind" in df.columns else "diagnosis",
        "positives_usable": int((usable["label"] == 1).sum()),
        "positive_patients_usable": int(pl.sum()),
        "unreadable_rows": n_bad,
        "unreadable_files": sorted(Path(f).name for f in df.loc[df["h5_ok"] == 0, "file"].unique()),
        "splits": split_tables,
        "cross_validation": cv_report,
        "warnings": warnings,
        "confound_columns": confounds,
        "confound_screen": screen,
        "confound_screen_unit": screen_unit,
    }


# --------------------------------------------------------------------------
# Self-test
# --------------------------------------------------------------------------

def _toy_cohort(n_subjects: int, pos_frac: float, rows_per_subject: int = 3,
                n_classes: int = 2) -> pd.DataFrame:
    """A minimal cohort table with the columns the CV code touches."""
    rows = []
    for i in range(n_subjects):
        lab = (i % n_classes) if n_classes > 2 else int(i < round(n_subjects * pos_frac))
        for r in range(rows_per_subject):
            rows.append({"patient_id": f"p{i:03d}", "subject_id": f"p{i:03d}",
                         "file": f"/data/p{i:03d}.h5", "label": lab,
                         "label_name": str(lab), "h5_ok": 1,
                         "official_split": ""})
    return pd.DataFrame(rows)


def self_test() -> bool:
    """
    Assertions on the CV partition and the dedup guard. No drive access.

    Every check here is a property the real run silently depends on, written so
    that breaking it fails fast in CI rather than three stages downstream.
    """
    checks, failures = 0, []

    def ck(name, cond):
        nonlocal checks
        checks += 1
        if not cond:
            failures.append(name)

    # --- partition properties -------------------------------------------
    for k in (2, 3, 5, 7):
        df = _toy_cohort(43, 0.4)
        df = add_cv_folds(df, "toy", k, seed=7, inner_val_frac=0.2)
        assert_cv_partition(df, "toy", k)          # raises on failure
        ck(f"k={k} every subject has one fold",
           (df.groupby("subject_id")["cv_fold"].nunique() == 1).all())
        ck(f"k={k} folds cover 0..k-1", set(df["cv_fold"]) == set(range(k)))
        test_once = {}
        for j in range(k):
            for s in set(df.loc[df[f"cv{j}_split"] == "test", "subject_id"]):
                test_once[s] = test_once.get(s, 0) + 1
        ck(f"k={k} each subject tested exactly once",
           set(test_once.values()) == {1} and len(test_once) == 43)
        ck(f"k={k} every fold has a nonempty inner validation split",
           all((df[f"cv{j}_split"] == "validation").any() for j in range(k)))
        ck(f"k={k} inner validation disjoint from test",
           all(not (set(df.loc[df[f"cv{j}_split"] == "validation", "subject_id"])
                    & set(df.loc[df[f"cv{j}_split"] == "test", "subject_id"]))
               for j in range(k)))
        # Round-robin balances each STRATUM to within one subject, so the total
        # fold size is balanced to within (number of strata). Asserting <= 1 on
        # the total would be asserting something the algorithm does not promise
        # and does not need: what matters is that no class is starved.
        per_class = (df.drop_duplicates("subject_id")
                     .groupby("label")["cv_fold"].value_counts().unstack(fill_value=0))
        ck(f"k={k} every class is balanced across folds to within one subject",
           bool(((per_class.max(axis=1) - per_class.min(axis=1)) <= 1).all()))
        sizes = df.drop_duplicates("subject_id")["cv_fold"].value_counts()
        ck(f"k={k} fold sizes differ by at most the number of classes",
           int(sizes.max() - sizes.min()) <= int(df["label"].nunique()))

    # --- determinism and seed sensitivity --------------------------------
    a = add_cv_folds(_toy_cohort(31, 0.5), "toy", 5, 11, 0.2)["cv_fold"].tolist()
    b = add_cv_folds(_toy_cohort(31, 0.5), "toy", 5, 11, 0.2)["cv_fold"].tolist()
    c = add_cv_folds(_toy_cohort(31, 0.5), "toy", 5, 12, 0.2)["cv_fold"].tolist()
    ck("same seed -> identical partition", a == b)
    ck("different seed -> different partition", a != c)

    # Inner validation must differ between outer folds; sharing the seed across
    # folds would make every fold select the same validation subjects.
    df = add_cv_folds(_toy_cohort(40, 0.5), "toy", 5, 3, 0.2)
    vsets = [frozenset(df.loc[df[f"cv{j}_split"] == "validation", "subject_id"])
             for j in range(5)]
    ck("inner validation differs across outer folds", len(set(vsets)) > 1)

    # --- stratification ---------------------------------------------------
    df = add_cv_folds(_toy_cohort(50, 0.4), "toy", 5, 5, 0.2)
    per_fold_pos = [int(df[df[f"cv{j}_split"] == "test"]
                        .groupby("subject_id")["label"].max().sum()) for j in range(5)]
    ck("stratified: positives spread evenly",
       max(per_fold_pos) - min(per_fold_pos) <= 1)
    ck("stratified: no fold has zero positives", min(per_fold_pos) > 0)

    # --- ineligible rows are excluded, not folded in ---------------------
    df = _toy_cohort(20, 0.5)
    df.loc[df["subject_id"] == "p000", "h5_ok"] = 0
    df.loc[df["subject_id"] == "p019", "label"] = -1
    df = add_cv_folds(df, "toy", 5, 1, 0.2)
    assert_cv_partition(df, "toy", 5)
    ck("corrupt subject gets no fold",
       (df.loc[df["subject_id"] == "p000", "cv_fold"] == -1).all())
    ck("unlabelled subject gets no fold",
       (df.loc[df["subject_id"] == "p019", "cv_fold"] == -1).all())
    ck("ineligible rows carry an empty fold column",
       (df.loc[df["subject_id"].isin(["p000", "p019"]), "cv0_split"] == "").all())

    # --- the assertion actually fires ------------------------------------
    df = add_cv_folds(_toy_cohort(20, 0.5), "toy", 4, 2, 0.2)
    broken = df.copy()
    # Put one subject in two folds' test sets: exactly the leak this guards.
    broken.loc[broken["subject_id"] == "p001", "cv0_split"] = "test"
    try:
        assert_cv_partition(broken, "toy", 4)
        ck("assert_cv_partition rejects a double-tested subject", False)
    except AssertionError:
        ck("assert_cv_partition rejects a double-tested subject", True)

    broken = df.copy()
    broken.loc[broken.index[0], "cv_fold"] = 99
    try:
        assert_cv_partition(broken, "toy", 4)
        ck("assert_cv_partition rejects a subject with two cv_fold values", False)
    except AssertionError:
        ck("assert_cv_partition rejects a subject with two cv_fold values", True)

    # --- gate arithmetic --------------------------------------------------
    df = add_cv_folds(_toy_cohort(67, 31 / 67), "toy", 5, 9, 0.2)
    counts = fold_cluster_counts(df, 5)
    ck("gate: 67 subjects / 5 folds clears",
       all(c["clears_s06_gate"] for c in counts))
    df = add_cv_folds(_toy_cohort(45, 24 / 45), "toy", 5, 9, 0.2)
    ck("gate: 45 subjects / 5 folds does NOT clear",
       not all(c["clears_s06_gate"] for c in fold_cluster_counts(df, 5)))
    ck("gate: 45 subjects / 4 folds does clear",
       all(c["clears_s06_gate"]
           for c in fold_cluster_counts(add_cv_folds(_toy_cohort(45, 24 / 45),
                                                     "toy", 4, 9, 0.2), 4)))
    ck("gate thresholds match s06", (MIN_CLUSTERS_C1, MIN_CLASS_CLUSTERS_C1) == (10, 5))

    # A class too rare to ever satisfy the per-class floor must be reported as
    # missing rather than quietly counted as a pass.
    df = _toy_cohort(30, 0.5, n_classes=3)
    df.loc[df["subject_id"].isin([f"p{i:03d}" for i in range(3, 30)]), "label"] = 0
    df = add_cv_folds(df, "toy", 5, 4, 0.2)
    counts = fold_cluster_counts(df, 5)
    ck("rare class -> no fold clears the gate",
       not any(c["clears_s06_gate"] for c in counts))
    ck("rare class -> absent classes are named",
       any(c["missing_classes"] for c in counts))
    restr = viable_class_restriction(df, 5, 4, 0.2)
    ck("rare class -> the rare class is named as the blocker",
       restr["classes_dropped"] == [1, 2] and restr["classes_kept"] == [0])

    # Restricting to well-populated classes must actually be checked, not
    # asserted: 3 classes of 30 subjects each clears at K=5, and the helper has
    # to say so only when it is true.
    df = _toy_cohort(93, 0.5, n_classes=3)
    df.loc[df["subject_id"].isin([f"p{i:03d}" for i in (91, 92)]), "label"] = 7
    df = add_cv_folds(df, "toy", 5, 4, 0.2)
    restr = viable_class_restriction(df, 5, 4, 0.2)
    ck("restriction drops only the starved class", restr["classes_dropped"] == [7])
    ck("restriction clears the gate when the rest are big enough",
       restr["restricted_clears_gate"])
    ck("restriction reports how much data it keeps",
       0.9 < restr["restricted_fraction_of_rows"] <= 1.0)

    # --- dedup guard ------------------------------------------------------
    disc = pd.DataFrame([
        {"file": "/d/brain/val/file_brain_AXT2_1_2.h5", "basename": "file_brain_AXT2_1_2.h5",
         "dir_organ": "brain", "canonical_tree": 1},
        {"file": "/d/knee/val/file_brain_AXT2_1_2.h5", "basename": "file_brain_AXT2_1_2.h5",
         "dir_organ": "knee", "canonical_tree": 0},
        {"file": "/d/knee/val/file_brain_AXT1_3_4.h5", "basename": "file_brain_AXT1_3_4.h5",
         "dir_organ": "knee", "canonical_tree": 0},
    ])
    kept, dropped = dedup_by_basename(disc, "brain")
    ck("dedup keeps one row per basename", len(kept) == 2)
    ck("dedup prefers the canonical tree",
       kept.loc[kept["basename"] == "file_brain_AXT2_1_2.h5", "file"].iloc[0]
       == "/d/brain/val/file_brain_AXT2_1_2.h5")
    ck("dedup reports what it dropped",
       len(dropped) == 1 and dropped[0]["dropped_path"] == "/d/knee/val/file_brain_AXT2_1_2.h5")
    ck("dedup keeps a file that exists only outside the canonical tree",
       "file_brain_AXT1_3_4.h5" in set(kept["basename"]))

    # Identity dedup must distinguish a byte-identical copy from a genuine
    # repeat exam. Written to a temp dir so the content check is exercised for
    # real rather than mocked.
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rng = np.random.default_rng(0)
        same = rng.normal(size=(4, 8, 8)).astype(np.complex64)
        other = rng.normal(size=(4, 8, 8)).astype(np.complex64)
        for nm, arr in (("a.h5", same), ("b.h5", same), ("c.h5", other)):
            with h5py.File(f"{td}/{nm}", "w") as f:
                f.create_dataset("kspace", data=arr)
        idf = pd.DataFrame([
            {"basename": nm, "file": f"{td}/{nm}", "acquisition": "AXT2",
             "attr_patient_id": "h1", "kspace_shape": "(4, 8, 8)"}
            for nm in ("a.h5", "b.h5", "c.h5")])
        kept, dropped, repeats = dedup_by_identity(idf, "brain")
        ck("identity dedup drops the byte-identical copy", len(dropped) == 1)
        ck("identity dedup names the file it matched",
           dropped[0]["identical_to"].endswith("a.h5"))
        ck("identity dedup KEEPS a distinct repeat scan", len(kept) == 2)
        ck("identity dedup reports the kept repeat scan", len(repeats) == 1)
        ck("identity dedup keeps the different-content file",
           "c.h5" in set(kept["basename"]))

    # --- paired design detection ------------------------------------------
    between = add_cv_folds(_toy_cohort(60, 0.5), "toy", 5, 6, 0.2)
    ck("between-subject design is not flagged as paired",
       not design_kind(between)["paired"])
    ck("between-subject folds clear the gate (12 clusters, 6 per class)",
       all(c["clears_s06_gate"] for c in fold_cluster_counts(between, 5)))

    paired = pd.DataFrame([
        {"patient_id": f"p{i:03d}", "subject_id": f"p{i:03d}",
         "file": f"/d/p{i}_{lab}.h5", "label": lab, "label_name": str(lab),
         "h5_ok": 1, "official_split": ""}
        for i in range(40) for lab in (0, 1)])
    paired = add_cv_folds(paired, "toy", 5, 6, 0.2)
    assert_cv_partition(paired, "toy", 5)
    dk = design_kind(paired)
    ck("paired design is detected", dk["paired"] and dk["n_mixed_subjects"] == 40)
    pc = fold_cluster_counts(paired, 5)
    ck("paired: s06 max-collapse yields zero negative clusters",
       all(c["class_clusters"]["0"] == 0 for c in pc))
    ck("paired: the gate is reported as NOT cleared",
       not any(c["clears_s06_gate"] for c in pc))
    ck("paired: every subject still supplies both classes",
       all(c["clusters_containing_class"]["0"] == c["n_test_clusters"]
           and c["clusters_containing_class"]["1"] == c["n_test_clusters"]
           for c in pc))
    ck("paired: no K rescues the gate",
       largest_k_clearing_gate(paired, 6, 0.2)["largest_clearing"] is None)

    # A paired target is invisible to a patient-level screen and must be tested
    # per file, or the screen reports "nothing testable" and reads as "clean".
    scr = paired.copy()
    scr["site"] = np.where(scr["label"] == 1, "A", "B")
    ck("patient-level screen sees nothing on a paired target",
       confound_screen(scr, ["site"], unit="patient") == [])
    file_scr = confound_screen(scr, ["site"], unit="file")
    ck("file-level screen catches a perfectly collinear confound",
       len(file_scr) == 1 and file_scr[0]["p_value"] < 0.05)

    # --- filename routing -------------------------------------------------
    ck("brain regex matches a brain file",
       bool(CONFOUND_FILE_RE["brain"].match("file_brain_AXT2_200_2000019.h5")))
    ck("brain regex rejects a knee file",
       not CONFOUND_FILE_RE["brain"].match("file1000000.h5"))
    ck("knee regex matches a knee file",
       bool(CONFOUND_FILE_RE["knee"].match("file1001031.h5")))
    ck("knee regex rejects a brain file",
       not CONFOUND_FILE_RE["knee"].match("file_brain_AXT2_200_2000019.h5"))

    # --- organ attribution (the s00 bug) ---------------------------------
    from pipeline.s00_inventory import organ_from_filename, organ_from_path
    ck("brain copy in knee/ is attributed to the knee DIRECTORY",
       organ_from_path("/Volumes/Research/fastmridatasets/knee/val/file_brain_AXT2_1_2.h5")
       == "knee")
    ck("brain copy in knee/ is attributed to brain by FILENAME",
       organ_from_filename("/Volumes/Research/fastmridatasets/knee/val/file_brain_AXT2_1_2.h5")
       == "brain")
    ck("a mount point named after an organ does not relabel files",
       organ_from_path("/Volumes/brain-backup/fastmridatasets/prostate/x/file_prostate_AXT2_001.h5")
       == "prostate")
    ck("breast_updated resolves to breast",
       organ_from_path("/Volumes/Research/fastmridatasets/breast_updated/breast/x.h5")
       == "breast")
    ck("unknown tree is not guessed",
       organ_from_path("/Volumes/Research/fastmridatasets/misc/x.h5") == "unknown")

    print(f"s01 self-test: {checks - len(failures)}/{checks} checks passed")
    for f in failures:
        print(f"   FAILED: {f}")
    return not failures


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="PhaseDx stage 1: build cohort tables")
    ap.add_argument("--cohorts", nargs="+", default=list(ALL_COHORTS),
                    choices=list(ALL_COHORTS))
    ap.add_argument("--out", default=str(COHORT_DIR))
    ap.add_argument("--seed", type=int, default=20240517,
                    help="seed for the deterministic patient-level fallback split")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--cv-folds", type=int, default=5,
                    help="K for the subject-level stratified cross-validation")
    ap.add_argument("--cv-seed", type=int, default=None,
                    help="seed for the CV partition (defaults to --seed)")
    ap.add_argument("--inner-val-frac", type=float, default=0.2,
                    help="fraction of each CV training split held out as the "
                         "nested inner validation fold for model selection")
    ap.add_argument("--no-probe", action="store_true",
                    help="skip opening every h5 (faster, but no scanner confounds "
                         "and no corrupt-file detection). Not permitted for the "
                         "brain/knee confound cohorts, whose label IS the metadata")
    ap.add_argument("--breast-dirs", nargs="+", default=list(BREAST_DIRS),
                    help="which breast trees under DATA_ROOT to include")
    ap.add_argument("--confound-dirs", nargs="+", default=list(CONFOUND_SEARCH_DIRS),
                    help="trees searched for brain/knee files; both are searched "
                         "for both cohorts so the duplicate copy is found, not "
                         "assumed away")
    ap.add_argument("--self-test", action="store_true",
                    help="run the cross-validation and dedup self-tests and exit")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if self_test() else 1)

    if args.cv_folds < 2:
        print("ERROR: --cv-folds must be >= 2", file=sys.stderr)
        sys.exit(2)

    if not DATA_ROOT.exists():
        print(f"ERROR: DATA_ROOT does not exist: {DATA_ROOT}", file=sys.stderr)
        print("Is the drive mounted? Check `ls /Volumes`.", file=sys.stderr)
        sys.exit(2)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    probe = not args.no_probe

    cv_seed = args.seed if args.cv_seed is None else args.cv_seed
    summary = {"seed": args.seed, "val_frac": args.val_frac,
               "test_frac": args.test_frac, "probed": probe,
               "cv_folds": args.cv_folds, "cv_seed": cv_seed,
               "inner_val_frac": args.inner_val_frac,
               "s06_gate": {"source": GATE_SOURCE,
                            "min_clusters": MIN_CLUSTERS_C1,
                            "min_per_class": MIN_CLASS_CLUSTERS_C1},
               "cohorts": {}}
    cv_kw = dict(cv_folds=args.cv_folds, cv_seed=cv_seed,
                 inner_val_frac=args.inner_val_frac)

    for cohort in args.cohorts:
        print(f"\n>>> building {cohort} ...")
        if cohort in CONFOUND_COHORTS:
            df, acct = build_confound(cohort, probe, args.seed, args.val_frac,
                                      args.test_frac,
                                      dirs=tuple(args.confound_dirs), **cv_kw)
        elif cohort == "breast":
            df, acct = build_breast(probe, args.seed, args.val_frac,
                                    args.test_frac, dirs=tuple(args.breast_dirs),
                                    **cv_kw)
        else:
            df, acct = build_prostate(cohort.split("_", 1)[1], probe, args.seed,
                                      args.val_frac, args.test_frac, **cv_kw)

        path = out_dir / f"{cohort}_cohort.csv"
        df.to_csv(path, index=False)
        summary["cohorts"][cohort] = report_cohort(df, acct)
        summary["cohorts"][cohort]["path"] = str(path)
        print(f"   -> wrote {path}  ({len(df)} rows x {df.shape[1]} cols)")

        # Sidecars so nothing is silently dropped.
        if cohort in CONFOUND_COHORTS:
            drops = (acct.get("duplicate_basename_examples", [])
                     + acct.get("identity_duplicate_examples", []))
            miss = pd.DataFrame(drops) if drops else pd.DataFrame()
            if len(miss):
                miss.to_csv(out_dir / f"{cohort}_duplicate_copies_dropped.csv",
                            index=False)
            miss = pd.DataFrame()
        elif cohort == "breast":
            miss = pd.DataFrame({"patient_id": acct.get("missing_patients_examples", [])})
        else:
            miss = pd.DataFrame({"file": acct.get("missing_files_examples", [])})
        if len(miss):
            mp = out_dir / f"{cohort}_labelled_without_data_sample.csv"
            miss.to_csv(mp, index=False)

    (out_dir / "s01_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print("\n" + "=" * 78)
    print("STAGE 1 TOTALS")
    print("=" * 78)
    for cohort, s in summary["cohorts"].items():
        unit = ("files" if cohort in CONFOUND_COHORTS
                else "acq" if cohort == "breast" else "slices")
        tail = (f"target={s['accounting'].get('target')} "
                f"({s['accounting'].get('n_classes')} classes, NO TUMOUR LABEL)"
                if cohort in CONFOUND_COHORTS
                else f"{s['positives_usable']:>5} positive")
        print(f"  {cohort:<14} {s['rows_usable']:>6} usable {unit:<7} "
              f"{s['patients']:>4} patients  {tail}  "
              f"({s['unreadable_rows']} rows lost to corrupt h5)")

    print("\n" + "=" * 78)
    print(f"CROSS-VALIDATION vs THE s06 GATE "
          f"(>= {MIN_CLUSTERS_C1} clusters, >= {MIN_CLASS_CLUSTERS_C1} per class)")
    print("=" * 78)
    for cohort, s in summary["cohorts"].items():
        cv = s.get("cross_validation")
        if not cv:
            continue
        off = s["splits"].get("official_split", {}).get("test")
        off_txt = (f"official test fold: {off['patients']} patients, "
                   f"{off['positive_patients']} positive"
                   if off else "no official test fold")
        best = cv["largest_k_clearing_gate"]
        print(f"  {cohort:<14} K={cv['k']}  "
              f"{cv['folds_clearing_gate']}/{cv['k']} folds clear   "
              f"| largest K that clears: {best if best is not None else 'none in 2..12'}"
              f"   | {off_txt}")
    print(f"\nWrote {out_dir/'s01_summary.json'}")


if __name__ == "__main__":
    main()
