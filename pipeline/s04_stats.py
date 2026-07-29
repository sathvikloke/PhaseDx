"""
s04_stats.py
------------
Stage 4 of the PhaseDx pipeline: the statistics layer.

Consumes the per-run JSONs written by s03_train.py (one per cohort x condition
x seed, each carrying the full vector of test probabilities together with the
patient id and cache index of every slice) and turns them into numbers a
reviewer will accept.

The statistical hazards this file exists to avoid, in order of how badly each
one would sink the study:

1. SLICE-LEVEL BOOTSTRAP. ~30 slices of one prostate are not 30 independent
   observations; they are one patient measured 30 times. A bootstrap that
   resamples slices treats them as independent, shrinks the standard error by
   roughly sqrt(slices-per-patient), and produces a confidence interval that is
   two to three times too narrow. Every interval here resamples PATIENTS
   (subjects, strictly -- see the cluster-unit note below) with replacement and
   rebuilds the slice set from whoever was drawn. The self-test measures the
   coverage of both and prints them side by side; the naive one undercovers
   badly, which is exactly the reviewer's objection made quantitative.

2. A BOOTSTRAP DIFFERENCE TEST WEARING DELONG'S NAME. DeLong's test for two
   correlated ROC curves is a specific covariance estimator built from
   placement values (Sun & Xu's O(n log n) midrank form). It is implemented
   here, from midranks, and validated in the self-test against a large-sample
   empirical variance and against the empirical type-I error rate. A clustered
   bootstrap difference test is ALSO provided, because DeLong's variance
   assumes independent observations and slices are not independent -- but it is
   named `cluster_bootstrap_diff` and is never called DeLong.

3. THRESHOLD CHOSEN ON THE TEST SET. The Youden threshold is picked on
   VALIDATION and applied unchanged to test. Picking it on test inflates
   sensitivity and specificity by an amount nobody can bound.

4. DEGENERATE FOLDS SILENTLY BECOMING NUMBERS. The prostate DWI test fold has
   4 patients, 3 of them positive. Single-class bootstrap replicates,
   single-class folds, and NaN probabilities are all expected. Every such case
   returns an explicit null with a `reason` string. Nothing here ever invents a
   number to fill a hole.

5. A NULL RESULT ROUNDED UP. If phase does not beat magnitude, that is the
   finding. Nothing in this file is one-sided, nothing reports a bare point
   estimate, and multiplicity across the condition-pair comparisons is
   Holm-adjusted.

6. K-FOLD CV READ AS K INDEPENDENT EXPERIMENTS. The clinical cohorts are run
   over the stage-1 subject-level CV folds, one results SUBDIRECTORY per fold
   (`<cohort>_cv<k>/`). Treating those five directories as five experiments
   would give five underpowered estimates of one quantity AND multiply the Holm
   family by five. The correct reading is OUT-OF-FOLD POOLING: every subject
   sits in exactly one test fold, so concatenating the five test blocks yields
   one prediction per subject over the whole cohort -- full power, one estimate,
   one comparison. `pool_folds` builds that vector and REFUSES if the
   every-subject-exactly-once property does not hold, because a duplicated
   subject would be double-weighted and would narrow every interval downstream.
   Per-fold estimates are still computed and printed, but purely as a dispersion
   diagnostic; they never enter the comparison family.

Two results layouts are supported and auto-detected:

    results/confound_brain/brain_phase_seed42.json          single split, no folds
    results/prostate_t2_cv3/prostate_t2_phase_seed42.json   fold 3 of a 5-fold CV

The fold index comes from the DIRECTORY name; cohort, condition and seed come
from the payload. Filenames are never parsed for identity -- s03 writes the same
filename into every fold directory, so the filename alone cannot tell two folds
apart, and a non-recursive glob cannot see them at all.

Cluster unit. Stage 1 emits `subject_id`, which is NOT `patient_id` for breast:
repeated scans of one woman appear under different coded names and are joined
into `breast_repeat_group_*`. The stage-3 JSONs carry `patient_id` only, so
this module re-resolves the cluster unit by joining `cache_idx` back through
the stage-2 cache index and the stage-1 cohort CSV. If that join cannot be
made it falls back to patient_id and says so, loudly, in both the printed
tables and the output JSON.

Usage:
    python pipeline/s04_stats.py                       # reads pipeline_out/results
    python pipeline/s04_stats.py --results-dir pipeline_out/dryrun/results
    python pipeline/s04_stats.py --self-test           # no data needed
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import re
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import common  # noqa: E402
    _DEFAULT_RESULTS_DIR = common.RESULTS_DIR
    _DEFAULT_CACHE_DIR = common.CACHE_DIR
    _DEFAULT_COHORT_DIR = common.OUT_ROOT / "cohorts"
except Exception:  # pragma: no cover - stats layer must run standalone
    _ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_RESULTS_DIR = _ROOT / "pipeline_out" / "results"
    _DEFAULT_CACHE_DIR = _ROOT / "pipeline_out" / "cache"
    _DEFAULT_COHORT_DIR = _ROOT / "pipeline_out" / "cohorts"

logger = logging.getLogger("s04_stats")

# Normal quantile for a two-sided 95% interval. scipy is installed here, but
# this module deliberately depends on numpy + the stdlib only so the statistics
# can be rerun anywhere; the normal tail comes from math.erfc, which is exact
# to machine precision, and this constant is Phi^-1(0.975) to 16 digits.
Z_975 = 1.959963984540054


# ==========================================================================
# Small numerical helpers (numpy-only; no scipy dependency)
# ==========================================================================

def _norm_sf(z: float) -> float:
    """Upper tail of the standard normal, via erfc. |error| < 1e-15."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_sided_normal_p(z: float) -> float:
    """Two-sided p-value for a standard normal test statistic."""
    if not np.isfinite(z):
        return float("nan")
    return float(min(1.0, 2.0 * _norm_sf(abs(z))))


def wilson_interval(k: int, n: int, z: float = Z_975) -> tuple[float | None, float | None]:
    """
    Wilson score interval for a binomial proportion.

    Used instead of the Wald interval because the counts here are tiny (3
    positive patients) and Wald gives [1.0, 1.0] for 3/3, which is a lie.
    Wilson gives [0.44, 1.00] for the same data.
    """
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def compute_midrank(x: np.ndarray) -> np.ndarray:
    """
    Midranks (average ranks for ties), 1-based, in O(n log n).

    This is the primitive behind both the AUC and the DeLong covariance; ties
    must get the average rank or a model that emits many identical probabilities
    (a saturated sigmoid, which happens) gets a silently wrong AUC.
    """
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    z = x[order]
    n = len(x)
    t = np.zeros(n, dtype=float)
    i = 0
    while i < n:
        j = i
        while j < n and z[j] == z[i]:
            j += 1
        t[i:j] = 0.5 * (i + j - 1) + 1.0   # mean of the 1-based ranks i+1..j
        i = j
    out = np.empty(n, dtype=float)
    out[order] = t
    return out


def auc_midrank(labels: np.ndarray, scores: np.ndarray) -> float:
    """
    AUC == the Mann-Whitney U statistic, computed from midranks so that ties
    count as half a concordance. Returns NaN if either class is missing.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    m = int((labels == 1).sum())
    n = int((labels == 0).sum())
    if m == 0 or n == 0:
        return float("nan")
    r = compute_midrank(scores)
    return float((r[labels == 1].sum() - m * (m + 1) / 2.0) / (m * n))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    """AP = sum_k (R_k - R_{k-1}) * P_k, the sklearn (non-interpolated) form."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    n_pos = int((labels == 1).sum())
    if n_pos == 0 or len(labels) == 0:
        return float("nan")
    order = np.argsort(-scores, kind="mergesort")
    y = labels[order]
    s = scores[order]
    tp = np.cumsum(y == 1)
    fp = np.cumsum(y == 0)
    # Collapse tied scores to their last index: a threshold cannot split a tie.
    keep = np.r_[s[1:] != s[:-1], True]
    tp, fp = tp[keep], fp[keep]
    precision = tp / np.maximum(tp + fp, 1)
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


# ==========================================================================
# DeLong's test for two correlated ROC curves
# ==========================================================================

def fast_delong(preds_sorted: np.ndarray, m: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Sun & Xu (2014) O(n log n) DeLong.

    preds_sorted : (k, m+n) score matrix, k models, columns ordered so that the
                   first `m` columns are the positives.
    Returns (aucs, cov) where cov is the k x k covariance matrix of the AUCs.

    The structure is the classic one: V01 are the positives' placement values
    against the negatives, V10 the negatives' against the positives, and
    cov = S01/m + S10/n with S the sample covariance (ddof=1) of the placements.
    """
    preds_sorted = np.atleast_2d(np.asarray(preds_sorted, dtype=float))
    k, total = preds_sorted.shape
    m = int(m)
    n = total - m
    if m < 1 or n < 1:
        raise ValueError("fast_delong needs at least one positive and one negative")

    positive = preds_sorted[:, :m]
    negative = preds_sorted[:, m:]

    tx = np.empty((k, m))
    ty = np.empty((k, n))
    tz = np.empty((k, total))
    for r in range(k):
        tx[r] = compute_midrank(positive[r])
        ty[r] = compute_midrank(negative[r])
        tz[r] = compute_midrank(preds_sorted[r])

    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n            # placement value of each positive
    v10 = 1.0 - (tz[:, m:] - ty) / m      # placement value of each negative
    s01 = np.atleast_2d(np.cov(v01))      # ddof=1, as DeLong's S estimator
    s10 = np.atleast_2d(np.cov(v10))
    cov = s01 / m + s10 / n
    return aucs, cov


def delong_test(labels: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray) -> dict:
    """
    DeLong's two-sided test that two ROC curves computed on the SAME cases have
    equal area. Returns explicit nulls (with a reason) for degenerate inputs.

    NOTE ON INDEPENDENCE: DeLong's variance treats the cases as independent.
    At slice level in this study they are not (many slices per patient), so the
    slice-level p-value is anti-conservative and is reported with that caveat
    attached. The patient-level comparison, where each patient contributes one
    aggregated score, is the version that satisfies the assumption.
    """
    labels = np.asarray(labels)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    out = {
        "test": "delong_correlated_roc",
        "auc_a": None, "auc_b": None, "diff": None,
        "var_a": None, "var_b": None, "cov_ab": None,
        "se_diff": None, "z": None, "p": None,
        "ci_lo_diff": None, "ci_hi_diff": None,
        "n": int(len(labels)), "n_pos": int((labels == 1).sum()),
        "reason": None,
    }
    if len(labels) != len(a) or len(labels) != len(b):
        out["reason"] = "score vectors and labels have different lengths"
        return out
    if len(labels) == 0:
        out["reason"] = "empty case set"
        return out
    if not (np.isfinite(a).all() and np.isfinite(b).all()):
        out["reason"] = "non-finite scores"
        return out
    m = int((labels == 1).sum())
    n = int((labels == 0).sum())
    if m < 1 or n < 1:
        out["reason"] = f"single-class case set (n_pos={m}, n_neg={n}); AUC undefined"
        return out
    # The point estimates are always computable; only the variance is fragile.
    # Report them even when the test itself has to be refused, so the reader
    # sees the observed difference next to the refusal.
    out["auc_a"] = float(auc_midrank(labels, a))
    out["auc_b"] = float(auc_midrank(labels, b))
    out["diff"] = out["auc_a"] - out["auc_b"]

    if m < 2 or n < 2:
        # The placement-value covariance is a sample covariance with ddof=1; with
        # one positive or one negative case it is 0/0. Caught here rather than
        # letting numpy emit a NaN and a RuntimeWarning. This fires for real on
        # the prostate DWI patient-level fold (3 positive patients, 1 negative).
        out["reason"] = (
            f"DeLong variance needs >=2 positive and >=2 negative cases "
            f"(have n_pos={m}, n_neg={n}); use the clustered bootstrap difference instead"
        )
        return out

    order = np.argsort(-labels, kind="mergesort")   # positives first, stable
    preds = np.vstack([a[order], b[order]])
    aucs, cov = fast_delong(preds, m)
    var_diff = float(cov[0, 0] + cov[1, 1] - 2 * cov[0, 1])
    # fast_delong recomputes the AUCs from the same midranks; they must agree
    # with the direct computation or the column ordering is wrong.
    assert abs(aucs[0] - out["auc_a"]) < 1e-9 and abs(aucs[1] - out["auc_b"]) < 1e-9
    out["var_a"] = float(cov[0, 0])
    out["var_b"] = float(cov[1, 1])
    out["cov_ab"] = float(cov[0, 1])

    if not np.isfinite(var_diff) or var_diff <= 0:
        # Happens when the two score vectors induce identical rankings, or when
        # m == 1 or n == 1 so the ddof=1 covariance is degenerate.
        out["reason"] = (
            f"degenerate DeLong variance (var_diff={var_diff:.3g}); "
            f"identical rankings or too few cases (n_pos={m}, n_neg={n})"
        )
        return out

    se = math.sqrt(var_diff)
    z = out["diff"] / se
    out["se_diff"] = float(se)
    out["z"] = float(z)
    out["p"] = two_sided_normal_p(z)
    out["ci_lo_diff"] = float(out["diff"] - Z_975 * se)
    out["ci_hi_diff"] = float(out["diff"] + Z_975 * se)
    return out


# ==========================================================================
# Cluster-aware resampling
# ==========================================================================

def _clean(labels, scores, clusters):
    """
    Drop rows with non-finite scores or non-binary labels.

    NaN probabilities happen (a diverged run, an all-zero slice). They must be
    removed explicitly and counted, not silently propagated into a NaN AUC that
    later gets formatted as '-'.
    """
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    clusters = np.asarray(clusters, dtype=object)
    ok = np.isfinite(scores) & np.isin(labels, (0, 1))
    n_dropped = int((~ok).sum())
    return labels[ok].astype(int), scores[ok], clusters[ok], n_dropped


def _cluster_index(clusters: np.ndarray):
    """cluster id -> array of row positions, in first-appearance order."""
    uniq, inverse = np.unique(clusters, return_inverse=True)
    groups = [np.flatnonzero(inverse == i) for i in range(len(uniq))]
    return uniq, groups


def cluster_bootstrap_auc(
    labels, scores, clusters, n_boot: int = 2000, seed: int = 0,
    alpha: float = 0.05, statistic=None,
) -> dict:
    """
    AUC with a percentile confidence interval from a bootstrap CLUSTERED ON
    PATIENT/SUBJECT: resample the cluster ids with replacement, rebuild the slice
    set from whoever was drawn (a patient drawn twice contributes all of their
    slices twice), recompute the statistic.

    Replicates that come out single-class, or that draw fewer than 2 distinct
    clusters, are skipped and counted -- with 3 positive patients out of 4 in
    the prostate DWI test fold, a large fraction of replicates are degenerate
    and pretending otherwise would be dishonest.
    """
    labels, scores, clusters, n_dropped = _clean(labels, scores, clusters)
    stat = statistic or auc_midrank
    out = {
        "auc": None, "ci_lo": None, "ci_hi": None,
        "ci_method": f"cluster_bootstrap_percentile_{int((1-alpha)*100)}",
        "cluster_unit": "cluster",
        "n_slices": int(len(labels)), "n_pos_slices": int((labels == 1).sum()),
        "n_clusters": 0, "n_pos_clusters": 0,
        "n_boot_requested": int(n_boot), "n_boot_used": 0,
        "n_skipped_single_class": 0, "n_skipped_single_cluster": 0,
        "n_dropped_nonfinite": n_dropped,
        "boot_mean": None, "boot_sd": None,
        "reason": None,
    }
    if len(labels) == 0:
        out["reason"] = "no usable rows (all dropped as non-finite or non-binary)"
        return out

    uniq, groups = _cluster_index(clusters)
    out["n_clusters"] = int(len(uniq))
    out["n_pos_clusters"] = int(sum(1 for g in groups if labels[g].max() == 1))

    if len(np.unique(labels)) < 2:
        out["reason"] = (
            f"single-class fold: {out['n_pos_slices']}/{out['n_slices']} positive; "
            "AUC is undefined"
        )
        return out
    if len(uniq) < 2:
        # AUC is computable but every observation comes from one patient, so it
        # measures within-patient slice ordering, not between-patient discrimination.
        out["auc"] = float(stat(labels, scores))
        out["reason"] = (
            f"only {len(uniq)} cluster(s) in this fold; no between-patient "
            "resampling is possible, so no interval is reported"
        )
        return out

    out["auc"] = float(stat(labels, scores))
    rng = np.random.default_rng(seed)
    k = len(uniq)
    vals = []
    n_single_class = 0
    n_single_cluster = 0
    for _ in range(int(n_boot)):
        draw = rng.integers(0, k, size=k)
        if len(np.unique(draw)) < 2:
            n_single_cluster += 1
            continue
        rows = np.concatenate([groups[d] for d in draw])
        yl = labels[rows]
        if yl.min() == yl.max():
            n_single_class += 1
            continue
        v = stat(yl, scores[rows])
        if np.isfinite(v):
            vals.append(v)
        else:
            n_single_class += 1

    out["n_boot_used"] = len(vals)
    out["n_skipped_single_class"] = int(n_single_class)
    out["n_skipped_single_cluster"] = int(n_single_cluster)
    if len(vals) < 20:
        out["reason"] = (
            f"only {len(vals)}/{n_boot} bootstrap replicates were evaluable "
            f"({n_single_class} single-class, {n_single_cluster} single-cluster); "
            "no interval is reported"
        )
        return out
    vals = np.asarray(vals)
    out["ci_lo"] = float(np.quantile(vals, alpha / 2))
    out["ci_hi"] = float(np.quantile(vals, 1 - alpha / 2))
    out["boot_mean"] = float(vals.mean())
    out["boot_sd"] = float(vals.std(ddof=1))
    return out


def naive_slice_bootstrap_auc(labels, scores, n_boot: int = 2000, seed: int = 0,
                              alpha: float = 0.05) -> dict:
    """
    Slice-level bootstrap that ignores clustering. Computed ONLY so the report
    can show, per fold, how much narrower the wrong interval would have been.
    It is never used as the headline interval.
    """
    labels, scores, clusters, _ = _clean(labels, scores, np.arange(len(labels)))
    out = {"ci_lo": None, "ci_hi": None, "width": None, "reason": None}
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        out["reason"] = "single-class or empty fold"
        return out
    rng = np.random.default_rng(seed)
    n = len(labels)
    vals = []
    for _ in range(int(n_boot)):
        rows = rng.integers(0, n, size=n)
        yl = labels[rows]
        if yl.min() == yl.max():
            continue
        v = auc_midrank(yl, scores[rows])
        if np.isfinite(v):
            vals.append(v)
    if len(vals) < 20:
        out["reason"] = "too few evaluable replicates"
        return out
    vals = np.asarray(vals)
    out["ci_lo"] = float(np.quantile(vals, alpha / 2))
    out["ci_hi"] = float(np.quantile(vals, 1 - alpha / 2))
    out["width"] = out["ci_hi"] - out["ci_lo"]
    return out


def cluster_bootstrap_diff(labels, scores_a, scores_b, clusters,
                           n_boot: int = 2000, seed: int = 0,
                           alpha: float = 0.05) -> dict:
    """
    Clustered bootstrap for the AUC DIFFERENCE between two models scored on the
    same cases. This is NOT DeLong's test -- it is reported next to DeLong
    because DeLong's variance assumes independent cases and slices are not.

    The p-value is the usual two-sided bootstrap p: twice the smaller tail mass
    on the wrong side of zero, floored at 1/n_boot_used (a bootstrap cannot
    resolve p below its own resolution and must not claim to).
    """
    labels = np.asarray(labels)
    a = np.asarray(scores_a, dtype=float)
    b = np.asarray(scores_b, dtype=float)
    out = {
        "test": "cluster_bootstrap_auc_difference",
        "diff": None, "ci_lo": None, "ci_hi": None, "p": None,
        "n_boot_used": 0, "n_skipped_single_class": 0, "n_skipped_single_cluster": 0,
        "reason": None,
    }
    ok = np.isfinite(a) & np.isfinite(b) & np.isin(labels, (0, 1))
    labels, a, b = labels[ok].astype(int), a[ok], b[ok]
    clusters = np.asarray(clusters, dtype=object)[ok]
    if len(labels) == 0 or len(np.unique(labels)) < 2:
        out["reason"] = "single-class or empty fold"
        return out
    uniq, groups = _cluster_index(clusters)
    if len(uniq) < 2:
        out["reason"] = f"only {len(uniq)} cluster(s); no between-patient resampling possible"
        return out

    out["diff"] = float(auc_midrank(labels, a) - auc_midrank(labels, b))
    rng = np.random.default_rng(seed)
    k = len(uniq)
    vals = []
    n_sc, n_su = 0, 0
    for _ in range(int(n_boot)):
        draw = rng.integers(0, k, size=k)
        if len(np.unique(draw)) < 2:
            n_su += 1
            continue
        rows = np.concatenate([groups[d] for d in draw])
        yl = labels[rows]
        if yl.min() == yl.max():
            n_sc += 1
            continue
        d = auc_midrank(yl, a[rows]) - auc_midrank(yl, b[rows])
        if np.isfinite(d):
            vals.append(d)
        else:
            n_sc += 1
    out["n_boot_used"] = len(vals)
    out["n_skipped_single_class"] = n_sc
    out["n_skipped_single_cluster"] = n_su
    if len(vals) < 20:
        out["reason"] = f"only {len(vals)}/{n_boot} evaluable replicates"
        return out
    vals = np.asarray(vals)
    out["ci_lo"] = float(np.quantile(vals, alpha / 2))
    out["ci_hi"] = float(np.quantile(vals, 1 - alpha / 2))
    tail = min((vals <= 0).mean(), (vals >= 0).mean())
    out["p"] = float(min(1.0, max(2.0 * tail, 1.0 / len(vals))))
    return out


# ==========================================================================
# Patient-level aggregation
# ==========================================================================

def aggregate_by_cluster(labels, scores, clusters, how: str = "mean") -> dict:
    """
    Collapse slice predictions to one score per patient/subject.

    how='mean' is the stable summary; how='max' is the "does any slice look like
    tumour" reading, which is what a radiologist-facing triage tool would use and
    which is far more sensitive to a single confident false positive. Both are
    reported because they can disagree and picking the better one post hoc is
    cherry-picking.

    The patient label is max(slice labels): a patient with any positive slice is
    a positive patient. Patients with mixed slice labels are counted and returned
    so the report can say how many there were.
    """
    if how not in ("mean", "max"):
        raise ValueError(f"unknown aggregation {how!r}")
    labels, scores, clusters, n_dropped = _clean(labels, scores, clusters)
    uniq, groups = _cluster_index(clusters)
    agg_scores, agg_labels, n_mixed = [], [], 0
    for g in groups:
        yl = labels[g]
        if yl.min() != yl.max():
            n_mixed += 1
        agg_labels.append(int(yl.max()))
        agg_scores.append(float(scores[g].mean() if how == "mean" else scores[g].max()))
    return {
        "cluster_ids": [str(u) for u in uniq],
        "labels": np.asarray(agg_labels, dtype=int),
        "scores": np.asarray(agg_scores, dtype=float),
        "n_clusters": int(len(uniq)),
        "n_pos_clusters": int(sum(agg_labels)),
        "n_mixed_label_clusters": int(n_mixed),
        "n_dropped_nonfinite": n_dropped,
        "how": how,
    }


# ==========================================================================
# Operating point: Youden on validation, applied to test
# ==========================================================================

def youden_threshold(labels, scores) -> dict:
    """
    Threshold maximising Youden's J = sensitivity + specificity - 1.

    Candidates are midpoints between adjacent distinct scores (plus one below
    the minimum), so the threshold never sits exactly on an observed value where
    a floating-point tie would flip the decision. Ties in J are broken by taking
    the MEDIAN of the tied candidates -- deterministic, and it does not drift to
    an extreme of the score range the way 'first maximum' does.
    """
    labels, scores, _, n_dropped = _clean(labels, scores, np.arange(len(np.asarray(labels))))
    out = {"threshold": None, "youden_j": None, "sens": None, "spec": None,
           "n": int(len(labels)), "n_pos": int((labels == 1).sum()),
           "n_dropped_nonfinite": n_dropped, "reason": None}
    if len(labels) == 0:
        out["reason"] = "no usable rows"
        return out
    if len(np.unique(labels)) < 2:
        out["reason"] = (
            f"single-class set (n_pos={out['n_pos']}/{out['n']}); "
            "Youden's J is undefined"
        )
        return out
    u = np.unique(scores)
    cands = np.r_[u[0] - 1e-9, (u[:-1] + u[1:]) / 2.0] if len(u) > 1 else np.r_[u[0] - 1e-9]
    pos, neg = scores[labels == 1], scores[labels == 0]
    sens = np.array([(pos >= t).mean() for t in cands])
    spec = np.array([(neg < t).mean() for t in cands])
    j = sens + spec - 1.0
    best = np.flatnonzero(j >= j.max() - 1e-12)
    pick = int(best[len(best) // 2])
    out["threshold"] = float(cands[pick])
    out["youden_j"] = float(j[pick])
    out["sens"] = float(sens[pick])
    out["spec"] = float(spec[pick])
    out["n_tied_candidates"] = int(len(best))
    return out


def metrics_at_threshold(labels, scores, threshold: float | None) -> dict:
    """Sens/spec/PPV/NPV at a FIXED threshold, with Wilson intervals."""
    out = {"threshold": threshold, "tp": None, "fp": None, "tn": None, "fn": None,
           "sens": None, "sens_ci": [None, None], "spec": None, "spec_ci": [None, None],
           "ppv": None, "npv": None, "accuracy": None, "n": None, "n_pos": None,
           "reason": None}
    if threshold is None:
        out["reason"] = "no threshold available (see the validation-fold reason)"
        return out
    labels, scores, _, n_dropped = _clean(labels, scores, np.arange(len(np.asarray(labels))))
    out["n"] = int(len(labels))
    out["n_pos"] = int((labels == 1).sum())
    out["n_dropped_nonfinite"] = n_dropped
    if len(labels) == 0:
        out["reason"] = "no usable rows"
        return out
    pred = scores >= threshold
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    tn = int(((pred == 0) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    out.update(tp=tp, fp=fp, tn=tn, fn=fn)
    n_p, n_n = tp + fn, tn + fp
    if n_p > 0:
        out["sens"] = tp / n_p
        out["sens_ci"] = list(wilson_interval(tp, n_p))
    if n_n > 0:
        out["spec"] = tn / n_n
        out["spec_ci"] = list(wilson_interval(tn, n_n))
    if tp + fp > 0:
        out["ppv"] = tp / (tp + fp)
    if tn + fn > 0:
        out["npv"] = tn / (tn + fn)
    out["accuracy"] = (tp + tn) / len(labels)
    if n_p == 0 or n_n == 0:
        out["reason"] = (
            f"single-class evaluation fold (n_pos={n_p}, n_neg={n_n}); "
            "sensitivity or specificity is undefined"
        )
    return out


# ==========================================================================
# Multiplicity
# ==========================================================================

def holm_adjust(pvals) -> list[float | None]:
    """
    Holm-Bonferroni step-down adjusted p-values, monotonicity enforced.

    Entries that are None or NaN (a comparison that could not be computed) are
    passed through as None and do NOT count towards the family size -- an
    uncomputable comparison is not a test that was performed.
    """
    pvals = list(pvals)
    idx = [i for i, p in enumerate(pvals) if p is not None and np.isfinite(p)]
    out: list[float | None] = [None] * len(pvals)
    m = len(idx)
    if m == 0:
        return out
    order = sorted(idx, key=lambda i: pvals[i])
    running = 0.0
    for rank, i in enumerate(order):
        adj = (m - rank) * pvals[i]
        running = max(running, adj)          # enforce monotone non-decreasing
        out[i] = float(min(1.0, running))
    return out


# ==========================================================================
# Loading stage-3 run JSONs and resolving the cluster unit
# ==========================================================================

_CV_DIR_RE = re.compile(r"^(?P<base>.+)_cv(?P<k>\d+)$")


def parse_fold_dir(dirname: str) -> tuple[str, int | None]:
    """
    Split a results subdirectory name into (base, fold).

        'prostate_t2_cv3' -> ('prostate_t2', 3)
        'confound_brain'  -> ('confound_brain', None)
        ''                -> ('', None)

    This is the ONLY place a fold index is derived, and it reads the DIRECTORY,
    never the filename: s03 writes `<cohort>_<condition>_seed<seed>.json` with no
    fold component, so all five folds share one filename.
    """
    m = _CV_DIR_RE.match(dirname or "")
    if not m:
        return (dirname or ""), None
    return m.group("base"), int(m.group("k"))


def load_runs(results_dir: Path, recurse: bool = True) -> list[dict]:
    """
    Load every stage-3 run payload under results_dir, recording where it came from.

    Recursion is required: the cross-validated cohorts live one directory per
    fold. It also drags two hazards in with it, both guarded here:

      * stage-5 CONTROL payloads. A control payload is a stage-3 payload plus a
        `control` field, and stage 5's default output tree can sit inside the
        results tree. Pooling a phase-scramble or confound-predictability run
        into the headline would mix the model under test with the controls that
        exist to falsify it. Anything with a `control` other than "none" is
        skipped by shape, and the skip is logged.
      * FILENAME COLLISIONS. Five folds share one filename, so `_tag` is
        disambiguated with the fold and `_path` keeps the full provenance.

    Provenance recorded on every run:
        _path          absolute path
        _subdir        directory relative to results_dir ('' at the top level)
        _fold          int fold index, or None for a single-split layout
        _fold_base     the `<base>` of `<base>_cv<k>`, or None
        _fold_source   human-readable statement of how _fold was decided
        _split_family  'cv' for fold-tagged runs, else the subdirectory name.
                       Runs in different families are DIFFERENT experiments and
                       are never pooled with one another.
    """
    results_dir = Path(results_dir)
    runs = []
    if not results_dir.is_dir():
        return runs
    paths = sorted(results_dir.rglob("*.json") if recurse else results_dir.glob("*.json"))
    for p in paths:
        if p.name == "statistics.json":
            continue
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            logger.warning("skipping %s: unreadable JSON (%s)", p.name, exc)
            continue
        if not isinstance(d, dict) or "test" not in d or "condition" not in d:
            logger.warning("skipping %s: not a stage-3 run payload", p.name)
            continue
        ctrl = d.get("control")
        if ctrl is not None and str(ctrl) != "none":
            logger.info("skipping %s: stage-5 control payload (control=%s)",
                        p.relative_to(results_dir) if p.is_relative_to(results_dir) else p, ctrl)
            continue
        try:
            rel = p.parent.relative_to(results_dir)
            subdir = "" if str(rel) == "." else str(rel)
        except ValueError:                       # pragma: no cover - defensive
            subdir = str(p.parent)
        base, fold = parse_fold_dir(Path(subdir).name if subdir else "")
        d["_path"] = str(p)
        d["_subdir"] = subdir
        d["_fold"] = fold
        d["_fold_base"] = base if fold is not None else None
        d["_fold_source"] = (
            f"directory {subdir!r} matches <base>_cv<k> -> fold {fold}"
            if fold is not None else
            f"directory {subdir or '<results root>'!r} carries no _cv<k> suffix -> single split"
        )
        d["_stem"] = p.stem
        d["_tag"] = p.stem if fold is None else f"{p.stem}@cv{fold}"
        d["_split_family"] = "cv" if fold is not None else (subdir or ".")
        runs.append(d)
    return runs


def unit_key(run: dict) -> tuple:
    """
    The identity of one ESTIMATE: (cohort, region, split family, condition, seed).

    Every run sharing this key is a fold of the same cross-validated estimate.
    Cohort/condition/seed come from the payload; only the split family comes from
    the directory. `region` is part of the key because a region-restricted run is
    a different question asked of the same subjects.
    """
    return (str(run.get("cohort")), str(run.get("region", "full")),
            str(run.get("_split_family", ".")),
            str(run.get("condition")), run.get("seed"))


def experiment_key(run: dict) -> tuple:
    """(cohort, region, split family) -- the scope within which conditions compare."""
    return (str(run.get("cohort")), str(run.get("region", "full")),
            str(run.get("_split_family", ".")))


def _read_csv_dicts(path: Path) -> list[dict]:
    import csv
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_cluster_map(cohort: str, cache_dir: Path, cohort_dir: Path) -> dict:
    """
    Map stage-2 cache_idx -> stage-1 subject_id.

    Why this exists: stage-3 JSONs carry patient_id, but for breast the split
    -enforcement unit is subject_id, which merges repeated scans of the same
    woman that appear under different coded names (fastMRI_breast_139 and its
    repeat live in breast_repeat_group_1). Bootstrapping on patient_id there
    would treat two scans of one woman as two independent patients.

    Join path: cache index (idx -> file basename) -> cohort CSV (file basename
    -> subject_id). Returns {} plus a reason when the join is not possible.
    """
    info = {"map": {}, "source": None, "reason": None}
    idx_path = Path(cache_dir) / f"{cohort}_index.csv"
    coh_path = Path(cohort_dir) / f"{cohort}_cohort.csv"
    if not idx_path.exists():
        info["reason"] = f"no cache index at {idx_path}"
        return info
    rows = _read_csv_dicts(idx_path)
    if rows and "subject_id" in rows[0]:
        info["map"] = {int(r["idx"]): str(r["subject_id"]) for r in rows}
        info["source"] = f"{idx_path.name}:subject_id"
        return info
    if not coh_path.exists():
        info["reason"] = f"cache index has no subject_id and no cohort CSV at {coh_path}"
        return info
    coh = _read_csv_dicts(coh_path)
    if not coh or "subject_id" not in coh[0] or "file" not in coh[0]:
        info["reason"] = f"{coh_path.name} has no subject_id/file columns"
        return info
    by_file: dict[str, str] = {}
    for r in coh:
        by_file[Path(r["file"]).name] = str(r["subject_id"])
    mapping, missing = {}, 0
    for r in rows:
        base = Path(r.get("file", "")).name
        sid = by_file.get(base)
        if sid is None:
            missing += 1
            continue
        mapping[int(r["idx"])] = sid
    if missing:
        info["reason"] = (
            f"{missing}/{len(rows)} cache rows could not be joined to a subject_id "
            f"via {coh_path.name}"
        )
        return info
    info["map"] = mapping
    info["source"] = f"{idx_path.name}+{coh_path.name}:file->subject_id"
    return info


def check_cluster_maps(cache_dir: Path, cohort_dir: Path, cohorts=None) -> int:
    """
    Print, per cohort, whether the cache_idx -> subject_id join works and how
    many patient_ids collapse into a single subject_id.

    Run this before trusting any interval: if breast's repeat groups are not
    being collapsed, two scans of one woman are being counted as two
    independent patients in every bootstrap in this file.
    """
    # Discover cohorts from the cache rather than hard-coding them: a literal
    # tuple here silently skipped `brain` and `knee` once the confound cohorts
    # were added, which is the exact blind spot this check exists to close.
    if cohorts is None:
        found = sorted(p.name[: -len("_index.csv")]
                       for p in Path(cache_dir).glob("*_index.csv"))
        cohorts = found or ("prostate_dwi", "prostate_t2", "breast")
    print(_rule(100, "="))
    print("cluster-unit resolution check")
    print(_rule(100, "="))
    rc = 0
    for cohort in cohorts:
        info = build_cluster_map(cohort, cache_dir, cohort_dir)
        idx_path = Path(cache_dir) / f"{cohort}_index.csv"
        if not idx_path.exists():
            print(f"{cohort:<14} no cache yet ({idx_path.name} absent) -- skipped")
            continue
        if info["reason"]:
            print(f"{cohort:<14} FALLBACK to patient_id: {info['reason']}")
            rc = 1
            continue
        rows = _read_csv_dicts(idx_path)
        pid = {r["patient_id"] for r in rows}
        sid = set(info["map"].values())
        pairs = {(r["patient_id"], info["map"][int(r["idx"])]) for r in rows}
        collapsed = len(pairs) - len(sid)
        print(f"{cohort:<14} OK via {info['source']}")
        print(f"{'':<14} {len(rows)} cached slices, {len(pid)} patient_id, "
              f"{len(sid)} subject_id  ({collapsed} patient_id(s) merged by repeat grouping)")
        if collapsed:
            for p, s in sorted(pairs):
                if sum(1 for _, s2 in pairs if s2 == s) > 1:
                    print(f"{'':<14}   {p} -> {s}")
    print(_rule(100, "="))
    return rc


def resolve_clusters(split_payload: dict, cluster_map: dict | None) -> tuple[np.ndarray, str, str]:
    """
    Return (cluster ids, unit name, source description) for one split payload.

    Preference order: subject_ids written directly by stage 3 (if a future
    version writes them) > subject_id joined through cache_idx > patient_id
    straight from the run JSON, with the downgrade recorded.
    """
    if "subject_ids" in split_payload and split_payload["subject_ids"]:
        return (np.asarray([str(s) for s in split_payload["subject_ids"]], dtype=object),
                "subject_id", "run_json:subject_ids")
    pids = np.asarray([str(p) for p in split_payload["patient_ids"]], dtype=object)
    if cluster_map:
        cidx = split_payload.get("cache_idx")
        if cidx is not None and all(int(i) in cluster_map for i in cidx):
            return (np.asarray([cluster_map[int(i)] for i in cidx], dtype=object),
                    "subject_id", "cache_idx->subject_id join")
    return pids, "patient_id", "run_json:patient_ids (subject_id join unavailable)"


# ==========================================================================
# Per-run analysis
# ==========================================================================

def analyse_run(run: dict, cluster_map: dict | None, n_boot: int, seed: int,
                alpha: float) -> dict:
    """Everything computable from a single stage-3 run JSON."""
    tag = run.get("_tag", f"{run.get('cohort')}_{run.get('condition')}_seed{run.get('seed')}")
    res = {
        "tag": tag,
        "cohort": run.get("cohort"),
        "condition": run.get("condition"),
        "seed": run.get("seed"),
        "region": run.get("region", "full"),
        "path": run.get("_path"),
        # provenance: which layout this estimate came from
        "pooled": bool(run.get("pooled")),
        "folds": run.get("folds"),
        "n_folds": run.get("n_folds"),
        "split_family": run.get("_split_family"),
        "subdir": run.get("_subdir"),
        "fold_source": run.get("_fold_source"),
        "reported_test_auc": (run.get("test") or {}).get("auc"),
        "cluster_unit": None, "cluster_source": None,
        "slice_level": None,
        "patient_level_mean": None,
        "patient_level_max": None,
        "operating_point": None,
        "warnings": [],
    }
    test = run.get("test")
    if not test or not test.get("probs"):
        res["warnings"].append("run has no test predictions; nothing to analyse")
        return res

    y = np.asarray(test["labels"], dtype=float)
    p = np.asarray(test["probs"], dtype=float)
    clusters, unit, source = resolve_clusters(test, cluster_map)
    res["cluster_unit"], res["cluster_source"] = unit, source
    if unit != "subject_id":
        res["warnings"].append(
            "clustering on patient_id, not subject_id; for breast this can split "
            "repeated scans of one woman into two 'patients'"
        )
    n_nonfinite = int((~np.isfinite(p)).sum())
    if n_nonfinite:
        res["warnings"].append(f"{n_nonfinite}/{len(p)} test probabilities are non-finite; dropped")

    # ---- slice level
    sl = cluster_bootstrap_auc(y, p, clusters, n_boot=n_boot, seed=seed, alpha=alpha)
    sl["cluster_unit"] = unit
    cy, cp, _, _ = _clean(y, p, clusters)
    sl["ap"] = float(average_precision(cy, cp)) if len(cy) and len(np.unique(cy)) > 1 else None
    sl["prevalence"] = float(cy.mean()) if len(cy) else None
    sl["naive_slice_bootstrap_ci"] = naive_slice_bootstrap_auc(
        y, p, n_boot=n_boot, seed=seed + 1, alpha=alpha)
    res["slice_level"] = sl

    # Cross-check against the AUC stage 3 computed with sklearn. A mismatch means
    # the probability/label vectors are being read in a different order than the
    # one they were scored in, which would silently invalidate everything below.
    rep = res["reported_test_auc"]
    if rep is not None and np.isfinite(rep) and sl.get("auc") is not None and not n_nonfinite:
        if abs(float(rep) - sl["auc"]) > 1e-6:
            res["warnings"].append(
                f"recomputed test AUC {sl['auc']:.6f} disagrees with the value stage 3 "
                f"reported ({float(rep):.6f}); the prediction vectors may be misaligned"
            )
        else:
            res["auc_matches_stage3"] = True

    # ---- patient level (mean and max aggregation)
    for how, key in (("mean", "patient_level_mean"), ("max", "patient_level_max")):
        agg = aggregate_by_cluster(y, p, clusters, how=how)
        one_per = np.arange(agg["n_clusters"])   # patients are their own clusters
        r = cluster_bootstrap_auc(agg["labels"], agg["scores"], one_per,
                                  n_boot=n_boot, seed=seed + 2, alpha=alpha)
        r["aggregation"] = how
        r["cluster_unit"] = unit
        r["n_mixed_label_clusters"] = agg["n_mixed_label_clusters"]
        r["ap"] = (float(average_precision(agg["labels"], agg["scores"]))
                   if len(np.unique(agg["labels"])) > 1 else None)
        res[key] = r
        if agg["n_mixed_label_clusters"]:
            msg = (f"{agg['n_mixed_label_clusters']} {unit}(s) have both positive and "
                   "negative slices; patient label taken as max(slice labels)")
            if msg not in res["warnings"]:
                res["warnings"].append(msg)

    # ---- operating point: Youden picked on VALIDATION, applied to TEST
    val = run.get("val")
    op = {"chosen_on": "validation", "slice": None, "patient_mean": None,
          "validation": None, "reason": None}
    if not val or not val.get("probs"):
        op["reason"] = "run has no validation predictions; no threshold can be chosen off-test"
    else:
        vy = np.asarray(val["labels"], dtype=float)
        vp = np.asarray(val["probs"], dtype=float)
        vcl, _, _ = resolve_clusters(val, cluster_map)
        vth = youden_threshold(vy, vp)
        op["validation"] = {"slice": vth}
        op["slice"] = metrics_at_threshold(y, p, vth["threshold"])
        if vth["reason"]:
            op["slice"]["reason"] = (op["slice"].get("reason") or "") + \
                f" | validation threshold unavailable: {vth['reason']}"

        vagg = aggregate_by_cluster(vy, vp, vcl, how="mean")
        vth_pat = youden_threshold(vagg["labels"], vagg["scores"])
        op["validation"]["patient_mean"] = vth_pat
        tagg = aggregate_by_cluster(y, p, clusters, how="mean")
        op["patient_mean"] = metrics_at_threshold(tagg["labels"], tagg["scores"],
                                                  vth_pat["threshold"])
        if vth_pat["reason"]:
            op["patient_mean"]["reason"] = (op["patient_mean"].get("reason") or "") + \
                f" | validation threshold unavailable: {vth_pat['reason']}"
    res["operating_point"] = op
    return res


# ==========================================================================
# Out-of-fold pooling for cross-validated cohorts
# ==========================================================================

def pool_folds(fold_runs: list[dict], cluster_map: dict | None,
               expected_folds=None) -> tuple[dict | None, dict]:
    """
    Concatenate the TEST blocks of the folds of ONE (cohort, condition, seed)
    into a single out-of-fold prediction vector.

    K-fold CV over subjects partitions the cohort: every subject is in the test
    block of exactly one fold. Concatenating those test blocks therefore gives
    one prediction per subject over the FULL cohort. That is the estimate with
    the power, and it is one estimate rather than five, so it costs one entry in
    the multiplicity family rather than five.

    The whole construction rests on the partition property, so this function
    REFUSES rather than pools when it cannot verify it:

      * two runs claiming the same fold index (two sweeps written to one tree);
      * the same cache_idx appearing in two folds;
      * the same SUBJECT appearing in two folds.

    A silently duplicated subject would be weighted twice in the point estimate
    and, far worse, would be resampled as two independent clusters by the
    bootstrap -- narrowing every interval by pretending the cohort is larger
    than it is. There is no safe way to "fix that up", so the estimate is
    withheld and the reason is reported.

    Degeneracies that are NOT refusals, because the pooled estimate is still the
    best available reading and the caller is told exactly what it covers:
    a fold missing from the tree, a fold whose test block is empty, a fold whose
    test block is single-class, and non-finite probabilities (dropped and
    counted downstream by `_clean`, as everywhere else).

    Returns (pooled_run | None, info).
    """
    info = {
        "ok": False, "reason": None,
        "cohort": None, "condition": None, "seed": None, "region": None,
        "folds": [], "n_folds": 0,
        "expected_folds": sorted(int(k) for k in expected_folds) if expected_folds else None,
        "missing_folds": [],
        "n_slices": 0, "n_pos_slices": 0, "n_subjects": 0, "n_pos_subjects": 0,
        "cluster_unit": None, "cluster_source": None,
        "subjects_per_fold": {}, "slices_per_fold": {},
        "duplicate_subjects": [], "duplicate_cache_idx": [],
        "single_class_folds": [], "empty_folds": [],
        "per_fold_reported_auc": {}, "auc_recheck_failures": [],
        "cohort_coverage": None,
        "warnings": [], "paths": [],
    }
    if not fold_runs:
        info["reason"] = "no folds supplied"
        return None, info

    r0 = fold_runs[0]
    info["cohort"] = str(r0.get("cohort"))
    info["condition"] = str(r0.get("condition"))
    info["seed"] = r0.get("seed")
    info["region"] = str(r0.get("region", "full"))

    # ---- fold indices must be unique ------------------------------------
    seen: dict[int, str] = {}
    for r in fold_runs:
        k = r.get("_fold")
        if k is None:
            info["reason"] = (
                f"{r.get('_path')} has no parseable fold index but was handed to the "
                "pooler; refusing to pool runs whose fold membership is unknown"
            )
            return None, info
        if int(k) in seen:
            info["reason"] = (
                f"fold {int(k)} appears twice ({seen[int(k)]} and {r.get('_path')}); "
                "refusing to pool -- one of these would double-weight its subjects"
            )
            return None, info
        seen[int(k)] = str(r.get("_path"))

    ordered = sorted(fold_runs, key=lambda r: int(r["_fold"]))
    info["paths"] = [str(r.get("_path")) for r in ordered]

    # ---- concatenate the test blocks ------------------------------------
    probs, labels, pids, cidx, fold_of_row = [], [], [], [], []
    used_folds: list[int] = []
    for r in ordered:
        k = int(r["_fold"])
        t = r.get("test") or {}
        if not t.get("probs"):
            info["empty_folds"].append(k)
            info["warnings"].append(
                f"fold {k} has no test predictions; its subjects are absent from the "
                "pooled vector and the pooled estimate covers less than the full cohort"
            )
            continue
        n = len(t["probs"])
        if not (len(t.get("labels", [])) == n and len(t.get("patient_ids", [])) == n
                and len(t.get("cache_idx", [])) == n):
            info["reason"] = (
                f"fold {k} ({r.get('_path')}) has mismatched test vector lengths "
                f"(probs={n}, labels={len(t.get('labels', []))}, "
                f"patient_ids={len(t.get('patient_ids', []))}, "
                f"cache_idx={len(t.get('cache_idx', []))}); refusing to pool"
            )
            return None, info
        probs.extend(float(v) for v in t["probs"])
        labels.extend(t["labels"])
        pids.extend(str(v) for v in t["patient_ids"])
        cidx.extend(int(v) for v in t["cache_idx"])
        fold_of_row.extend([k] * n)
        used_folds.append(k)
        info["slices_per_fold"][k] = n
        info["per_fold_reported_auc"][k] = t.get("auc")
        yk = np.asarray(t["labels"], dtype=float)
        if len(np.unique(yk[np.isin(yk, (0, 1))])) < 2:
            info["single_class_folds"].append(k)
            info["warnings"].append(
                f"fold {k} has a single-class test block "
                f"({int((yk == 1).sum())}/{n} positive); it contributes cases to the "
                "pooled vector but has no fold-level AUC of its own"
            )

    if not probs:
        info["reason"] = "every fold's test block was empty; nothing to pool"
        return None, info

    info["folds"] = used_folds
    info["n_folds"] = len(used_folds)
    info["single_fold"] = len(used_folds) == 1
    if len(used_folds) == 1:
        info["warnings"].append(
            f"only fold {used_folds[0]} is on disk, so this 'pooled' estimate is a "
            "SINGLE FOLD under another name -- it has one fold's power and covers one "
            "fold's subjects, and must not be read as an out-of-fold estimate over the "
            "cohort (expected while a sweep is still running)"
        )
    if info["expected_folds"]:
        info["missing_folds"] = [k for k in info["expected_folds"] if k not in used_folds]
        if info["missing_folds"]:
            info["warnings"].append(
                f"folds {info['missing_folds']} are missing from this "
                f"(cohort, condition, seed); the pooled vector covers "
                f"{len(used_folds)}/{len(info['expected_folds'])} folds and is NOT the "
                "full cohort -- it must not be compared against a condition pooled over "
                "a different set of folds"
            )

    probs_a = np.asarray(probs, dtype=float)
    labels_a = np.asarray(labels)
    cidx_a = np.asarray(cidx, dtype=int)
    fold_a = np.asarray(fold_of_row, dtype=int)

    pooled_test = {
        "probs": probs_a.tolist(),
        "labels": [int(v) for v in labels_a],
        "patient_ids": list(pids),
        "cache_idx": cidx_a.tolist(),
        "fold_of_row": fold_a.tolist(),
        # No stage-3 AUC exists for a vector stage 3 never saw. Leaving this
        # None keeps analyse_run's stage-3 cross-check from comparing the pooled
        # AUC against a number that means something else; the per-fold recheck
        # below is the stronger guard it is replaced by.
        "auc": None, "ap": None, "loss": None,
        "n": int(len(probs_a)), "n_pos": int((labels_a == 1).sum()),
    }
    info["n_slices"] = int(len(probs_a))
    info["n_pos_slices"] = int((labels_a == 1).sum())

    # ---- resolve the cluster unit ONCE, on the pooled block --------------
    clusters, unit, source = resolve_clusters(pooled_test, cluster_map)
    info["cluster_unit"], info["cluster_source"] = unit, source

    # ---- assertion 1: no cache_idx in two folds -------------------------
    uidx, counts = np.unique(cidx_a, return_counts=True)
    dup_idx = uidx[counts > 1]
    if len(dup_idx):
        info["duplicate_cache_idx"] = [int(v) for v in dup_idx[:20]]
        info["reason"] = (
            f"{len(dup_idx)} cache_idx value(s) appear in more than one fold "
            f"(e.g. {[int(v) for v in dup_idx[:5]]}); the folds are not a partition, "
            "so pooling them would score the same slice twice. Refusing to pool"
        )
        return None, info

    # ---- assertion 2: every subject in exactly one fold ------------------
    # Within a fold a subject appears in many rows -- that is what clustering is
    # for. The property being asserted is across folds: subject -> {folds} must
    # be a singleton for every subject.
    folds_by_subject: dict[str, set] = {}
    for s, k in zip(clusters, fold_a):
        folds_by_subject.setdefault(str(s), set()).add(int(k))
    straddlers = sorted((s, sorted(f)) for s, f in folds_by_subject.items() if len(f) > 1)
    if straddlers:
        info["duplicate_subjects"] = [
            {"subject": s, "folds": f} for s, f in straddlers[:20]
        ]
        shown = ", ".join(f"{s} in folds {f}" for s, f in straddlers[:5])
        info["reason"] = (
            f"{len(straddlers)} {unit}(s) appear in more than one test fold ({shown}"
            f"{', ...' if len(straddlers) > 5 else ''}). Out-of-fold pooling assumes "
            "every subject is tested exactly once; a subject counted twice is "
            "double-weighted in the point estimate and resampled as two independent "
            "clusters by the bootstrap, which narrows every interval downstream. "
            "Refusing to pool"
        )
        return None, info

    info["n_subjects"] = len(folds_by_subject)
    for s, fs in folds_by_subject.items():
        k = next(iter(fs))
        info["subjects_per_fold"][k] = info["subjects_per_fold"].get(k, 0) + 1
    pos_subj = set()
    for s, y in zip(clusters, labels_a):
        if int(y) == 1:
            pos_subj.add(str(s))
    info["n_pos_subjects"] = len(pos_subj)

    # ---- coverage of the cohort (diagnostic, never a refusal) ------------
    if cluster_map and unit == "subject_id":
        all_subjects = set(str(v) for v in cluster_map.values())
        covered = set(folds_by_subject)
        info["cohort_coverage"] = {
            "n_pooled": len(covered),
            "n_in_cache": len(all_subjects),
            "n_uncovered": len(all_subjects - covered),
            "uncovered_examples": sorted(all_subjects - covered)[:10],
        }
        if all_subjects - covered:
            info["warnings"].append(
                f"{len(all_subjects - covered)}/{len(all_subjects)} cached {unit}(s) "
                "appear in no test fold; the pooled vector is out-of-fold over the "
                "subjects the CV actually held out, not over every cached subject "
                "(expected when a region filter or a missing fold removes subjects)"
            )

    # ---- per-fold AUC recheck against what stage 3 reported --------------
    # analyse_run's usual cross-check cannot fire on a pooled vector, so the
    # equivalent guard is applied per fold: if the probability/label vectors were
    # read in a different order than they were scored in, this catches it.
    for k in used_folds:
        rep = info["per_fold_reported_auc"].get(k)
        if rep is None or not np.isfinite(float(rep)):
            continue
        sel = fold_a == k
        yk, pk = labels_a[sel].astype(float), probs_a[sel]
        ok = np.isfinite(pk) & np.isin(yk, (0, 1))
        if ok.sum() != sel.sum() or len(np.unique(yk[ok])) < 2:
            continue
        got = float(auc_midrank(yk[ok].astype(int), pk[ok]))
        if abs(got - float(rep)) > 1e-6:
            info["auc_recheck_failures"].append(
                {"fold": k, "recomputed": got, "reported": float(rep)})
    if info["auc_recheck_failures"]:
        info["warnings"].append(
            "the AUC recomputed from the pooled rows of "
            f"{len(info['auc_recheck_failures'])} fold(s) disagrees with what stage 3 "
            "reported for that fold; the prediction vectors may be misaligned"
        )

    folds_txt = ",".join(f"cv{k}" for k in used_folds)
    pooled = {
        "cohort": info["cohort"], "condition": info["condition"],
        "seed": info["seed"], "region": info["region"],
        "val": None,          # thresholds are fit PER FOLD -- see pooled_operating_point
        "test": pooled_test,
        "pooled": True,
        "folds": used_folds,
        "n_folds": len(used_folds),
        "_path": " + ".join(info["paths"]),
        "_subdir": f"(out-of-fold pool of {folds_txt})",
        "_stem": f"{info['cohort']}_{info['condition']}_seed{info['seed']}",
        "_tag": f"{info['cohort']}_{info['condition']}_seed{info['seed']}~pooled",
        "_fold": None,
        "_fold_base": None,
        "_fold_source": f"pooled out-of-fold over {folds_txt}",
        "_split_family": "cv",
    }
    info["ok"] = True
    return pooled, info


def pooled_operating_point(fold_runs: list[dict], cluster_map: dict | None) -> dict:
    """
    Operating point for a pooled cross-validated estimate.

    The Youden threshold is fit on EACH FOLD'S OWN validation set and applied to
    THAT fold's test slices; the confusion counts are then summed across folds
    and sensitivity/specificity recomputed from the totals.

    Fitting a single threshold on the pooled validation predictions would be
    wrong twice over. The folds' validation carves are drawn from their own
    training portions and can share subjects, so the pooled validation set is not
    a set of distinct subjects; and each fold's test rows were scored by a model
    that never saw the other folds' validation data, so there is no single model
    whose threshold this would be. Neither error touches test data, but both
    would make the reported operating point uninterpretable.

    Test data is never used to choose a threshold, here or anywhere else.
    """
    op = {
        "chosen_on": "validation, per fold (thresholds summed at the count level)",
        "slice": None, "patient_mean": None, "validation": None,
        "per_fold": [], "reason": None,
    }
    tot = {lvl: {"tp": 0, "fp": 0, "tn": 0, "fn": 0, "n": 0, "n_folds": 0}
           for lvl in ("slice", "patient_mean")}
    no_threshold: list[int] = []
    for r in sorted(fold_runs, key=lambda r: int(r["_fold"])):
        k = int(r["_fold"])
        rec = {"fold": k, "slice_threshold": None, "patient_mean_threshold": None,
               "reason": None}
        val, test = r.get("val"), r.get("test")
        if not val or not val.get("probs") or not test or not test.get("probs"):
            rec["reason"] = "fold has no validation and/or test predictions"
            no_threshold.append(k)
            op["per_fold"].append(rec)
            continue
        vy = np.asarray(val["labels"], dtype=float)
        vp = np.asarray(val["probs"], dtype=float)
        vcl, _, _ = resolve_clusters(val, cluster_map)
        y = np.asarray(test["labels"], dtype=float)
        p = np.asarray(test["probs"], dtype=float)
        cl, _, _ = resolve_clusters(test, cluster_map)

        vth = youden_threshold(vy, vp)
        rec["slice_threshold"] = vth["threshold"]
        rec["slice_threshold_reason"] = vth["reason"]
        if vth["threshold"] is None:
            no_threshold.append(k)
        else:
            m = metrics_at_threshold(y, p, vth["threshold"])
            if m.get("tp") is not None:
                for f in ("tp", "fp", "tn", "fn"):
                    tot["slice"][f] += m[f]
                tot["slice"]["n"] += m["n"]
                tot["slice"]["n_folds"] += 1
                rec["slice_counts"] = {f: m[f] for f in ("tp", "fp", "tn", "fn")}

        vagg = aggregate_by_cluster(vy, vp, vcl, how="mean")
        vth_pat = youden_threshold(vagg["labels"], vagg["scores"])
        rec["patient_mean_threshold"] = vth_pat["threshold"]
        rec["patient_mean_threshold_reason"] = vth_pat["reason"]
        if vth_pat["threshold"] is not None:
            tagg = aggregate_by_cluster(y, p, cl, how="mean")
            m = metrics_at_threshold(tagg["labels"], tagg["scores"], vth_pat["threshold"])
            if m.get("tp") is not None:
                for f in ("tp", "fp", "tn", "fn"):
                    tot["patient_mean"][f] += m[f]
                tot["patient_mean"]["n"] += m["n"]
                tot["patient_mean"]["n_folds"] += 1
                rec["patient_mean_counts"] = {f: m[f] for f in ("tp", "fp", "tn", "fn")}
        op["per_fold"].append(rec)

    for lvl in ("slice", "patient_mean"):
        t = tot[lvl]
        thr_list = [r.get(f"{lvl}_threshold") for r in op["per_fold"]
                    if r.get(f"{lvl}_threshold") is not None]
        out = {
            "threshold": None,
            "threshold_note": (
                f"{len(thr_list)} per-fold validation threshold(s), "
                f"range [{min(thr_list):.3f}, {max(thr_list):.3f}]"
                if thr_list else "no fold produced a validation threshold"),
            "per_fold_thresholds": [float(v) for v in thr_list],
            "tp": t["tp"], "fp": t["fp"], "tn": t["tn"], "fn": t["fn"],
            "sens": None, "sens_ci": [None, None],
            "spec": None, "spec_ci": [None, None],
            "ppv": None, "npv": None, "accuracy": None,
            "n": t["n"], "n_pos": t["tp"] + t["fn"], "n_folds_contributing": t["n_folds"],
            "reason": None,
        }
        if thr_list:
            out["threshold"] = float(np.median(thr_list))
            out["threshold_is_summary"] = True
        if t["n_folds"] == 0:
            out["reason"] = ("no fold produced a validation threshold; no operating "
                             "point can be reported off-test")
            op[lvl] = out
            continue
        n_p, n_n = t["tp"] + t["fn"], t["tn"] + t["fp"]
        if n_p > 0:
            out["sens"] = t["tp"] / n_p
            out["sens_ci"] = list(wilson_interval(t["tp"], n_p))
        if n_n > 0:
            out["spec"] = t["tn"] / n_n
            out["spec_ci"] = list(wilson_interval(t["tn"], n_n))
        if t["tp"] + t["fp"] > 0:
            out["ppv"] = t["tp"] / (t["tp"] + t["fp"])
        if t["tn"] + t["fn"] > 0:
            out["npv"] = t["tn"] / (t["tn"] + t["fn"])
        if t["n"] > 0:
            out["accuracy"] = (t["tp"] + t["tn"]) / t["n"]
        if n_p == 0 or n_n == 0:
            out["reason"] = (f"pooled evaluation set is single-class (n_pos={n_p}, "
                             f"n_neg={n_n}); sensitivity or specificity is undefined")
        if no_threshold:
            out["reason"] = ((out["reason"] or "") +
                             f" | folds {sorted(set(no_threshold))} contributed no "
                             "threshold and are excluded from these counts").strip(" |")
        op[lvl] = out
    if tot["slice"]["n_folds"] == 0 and tot["patient_mean"]["n_folds"] == 0:
        op["reason"] = ("no fold produced a usable validation threshold; no operating "
                        "point can be chosen off-test")
    return op


def fold_diagnostic(run: dict, cluster_map: dict | None, n_boot: int, seed: int,
                    alpha: float) -> dict:
    """
    Per-fold AUC + clustered CI, computed ONLY as a dispersion diagnostic.

    A fold-to-fold spread far wider than the pooled CI is itself a finding: it
    says the estimate is unstable across which subjects happen to be held out,
    which the pooled interval alone would not show. These numbers are printed and
    stored, and they are deliberately kept out of `runs`, out of `across_seeds`
    and out of the comparison family -- five folds of one cohort are one
    experiment, and letting them in would multiply the Holm family by five.
    """
    out = {
        "cohort": run.get("cohort"), "condition": run.get("condition"),
        "seed": run.get("seed"), "region": run.get("region", "full"),
        "fold": run.get("_fold"), "tag": run.get("_tag"), "path": run.get("_path"),
        "fold_source": run.get("_fold_source"),
        "reported_test_auc": (run.get("test") or {}).get("auc"),
        "slice_level": None, "patient_level_mean": None,
        "cluster_unit": None, "reason": None,
    }
    test = run.get("test")
    if not test or not test.get("probs"):
        out["reason"] = "fold has no test predictions"
        return out
    y = np.asarray(test["labels"], dtype=float)
    p = np.asarray(test["probs"], dtype=float)
    clusters, unit, _ = resolve_clusters(test, cluster_map)
    out["cluster_unit"] = unit
    sl = cluster_bootstrap_auc(y, p, clusters, n_boot=n_boot, seed=seed, alpha=alpha)
    sl["cluster_unit"] = unit
    out["slice_level"] = sl
    agg = aggregate_by_cluster(y, p, clusters, how="mean")
    pm = cluster_bootstrap_auc(agg["labels"], agg["scores"], np.arange(agg["n_clusters"]),
                               n_boot=n_boot, seed=seed + 2, alpha=alpha)
    pm["cluster_unit"] = unit
    out["patient_level_mean"] = pm
    return out


def fold_dispersion(per_fold: list[dict], per_run: list[dict]) -> list[dict]:
    """
    Fold-to-fold spread of the per-fold AUCs, next to the pooled CI width.

    `spread_vs_pooled_ci` > 1 means the folds disagree with each other by more
    than the pooled interval admits -- i.e. the pooled CI, which is honest about
    subject sampling, is NOT honest about which-subjects-were-held-out
    variability, and the reader should be told so.
    """
    groups: dict[tuple, list[dict]] = {}
    for f in per_fold:
        groups.setdefault((f["cohort"], f["condition"], f["seed"], f["region"]), []).append(f)
    pooled_by_key = {
        (r["cohort"], r["condition"], r["seed"], r["region"]): r
        for r in per_run if r.get("pooled")
    }
    out = []
    for key, fs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        for level in ("slice_level", "patient_level_mean"):
            vals = [(f["fold"], (f.get(level) or {}).get("auc")) for f in fs]
            ok = [(k, v) for k, v in vals if v is not None and np.isfinite(v)]
            pooled = pooled_by_key.get(key)
            pooled_blk = (pooled or {}).get(level) or {}
            pooled_w = (None if pooled_blk.get("ci_lo") is None
                        else float(pooled_blk["ci_hi"] - pooled_blk["ci_lo"]))
            rec = {
                "cohort": key[0], "condition": key[1], "seed": key[2], "region": key[3],
                "level": level, "n_folds": len(fs), "n_evaluable_folds": len(ok),
                "folds": [k for k, _ in ok],
                "fold_aucs": [float(v) for _, v in ok],
                "fold_auc_min": float(min(v for _, v in ok)) if ok else None,
                "fold_auc_max": float(max(v for _, v in ok)) if ok else None,
                "fold_auc_sd": (float(np.std([v for _, v in ok], ddof=1))
                                if len(ok) > 1 else None),
                "fold_auc_range": (float(max(v for _, v in ok) - min(v for _, v in ok))
                                   if ok else None),
                "pooled_auc": pooled_blk.get("auc"),
                "pooled_ci_width": pooled_w,
                "spread_vs_pooled_ci": None,
                "note": None,
            }
            if rec["fold_auc_range"] is not None and pooled_w:
                rec["spread_vs_pooled_ci"] = rec["fold_auc_range"] / pooled_w
                if rec["spread_vs_pooled_ci"] > 1.0:
                    rec["note"] = (
                        "fold-to-fold range exceeds the pooled 95% CI width: the estimate "
                        "moves more with WHICH subjects are held out than the pooled "
                        "interval alone would suggest"
                    )
            if len(ok) < len(fs):
                rec["note"] = ((rec["note"] or "") +
                               f" | {len(fs) - len(ok)} fold(s) had no evaluable AUC").strip(" |")
            out.append(rec)
    return out


# ==========================================================================
# Comparisons between conditions
# ==========================================================================

def _align_on_cache_idx(run_a: dict, run_b: dict, split: str = "test"):
    """
    Put two runs' predictions in the same case order, keyed on cache_idx.

    DeLong requires the SAME cases in both vectors. Two runs of the same cohort
    and seed should already agree, but a silent mismatch (different val carve,
    different region filter) would otherwise produce a confident comparison of
    two different test sets.
    """
    a, b = run_a.get(split), run_b.get(split)
    if not a or not b:
        return None, f"one of the runs has no {split} predictions"
    ia = np.asarray(a.get("cache_idx", []), dtype=int)
    ib = np.asarray(b.get("cache_idx", []), dtype=int)
    if len(ia) == 0 or len(ib) == 0:
        return None, "cache_idx missing; cannot verify the two runs scored the same cases"
    if set(ia.tolist()) != set(ib.tolist()):
        return None, (f"the two runs scored different {split} sets "
                      f"({len(set(ia.tolist()) ^ set(ib.tolist()))} non-shared cases)")
    oa, ob = np.argsort(ia), np.argsort(ib)
    ya = np.asarray(a["labels"], dtype=float)[oa]
    yb = np.asarray(b["labels"], dtype=float)[ob]
    if not np.array_equal(ya, yb):
        return None, "labels disagree between the two runs for the same cache_idx"
    return {
        "labels": ya,
        "scores_a": np.asarray(a["probs"], dtype=float)[oa],
        "scores_b": np.asarray(b["probs"], dtype=float)[ob],
        "patient_ids_a": np.asarray([str(x) for x in a["patient_ids"]], dtype=object)[oa],
        "cache_idx": ia[oa],
    }, None


def compare_runs(run_a: dict, run_b: dict, cluster_map: dict | None,
                 n_boot: int, seed: int, alpha: float) -> list[dict]:
    """
    DeLong (+ a clustered bootstrap difference) at slice level and at patient
    level, for one pair of conditions on one cohort/seed/region.
    """
    base = {
        "cohort": run_a.get("cohort"), "seed": run_a.get("seed"),
        "region": run_a.get("region", "full"),
        "model_a": run_a.get("condition"), "model_b": run_b.get("condition"),
        "tag_a": run_a.get("_tag"), "tag_b": run_b.get("_tag"),
        # Provenance. `fold` is always None: a comparison is between two POOLED
        # (or two single-split) estimates, never between two folds. The Holm
        # family-size check downstream asserts exactly that.
        "split_family": run_a.get("_split_family"),
        "pooled": bool(run_a.get("pooled")) and bool(run_b.get("pooled")),
        "folds_a": run_a.get("folds"), "folds_b": run_b.get("folds"),
        "fold": None,
    }
    aligned, err = _align_on_cache_idx(run_a, run_b, "test")
    if aligned is None:
        if base["pooled"] and run_a.get("folds") != run_b.get("folds"):
            err = (f"{err}; the two conditions were pooled over different folds "
                   f"({run_a.get('condition')}: {run_a.get('folds')}, "
                   f"{run_b.get('condition')}: {run_b.get('folds')}), so they were not "
                   "scored on the same out-of-fold set and a paired test would be "
                   "comparing two different cohorts")
        return [dict(base, level="slice", delong={"reason": err, "p": None},
                     cluster_bootstrap_diff={"reason": err, "p": None},
                     p_raw=None, reason=err)]

    y = aligned["labels"]
    a, b = aligned["scores_a"], aligned["scores_b"]
    # Rebuild clusters from the aligned cache_idx so both levels use one unit.
    fake_split = {"patient_ids": aligned["patient_ids_a"],
                  "cache_idx": aligned["cache_idx"].tolist()}
    clusters, unit, source = resolve_clusters(fake_split, cluster_map)

    # Complete cases only: a paired comparison must score both models on exactly
    # the same slices, so a NaN in either model's output removes that slice from
    # BOTH. Dropping per-model would compare model A on 120 slices to model B on
    # 119 and call the difference an effect.
    keep = np.isfinite(a) & np.isfinite(b) & np.isin(y, (0, 1))
    n_dropped = int((~keep).sum())
    y, a, b, clusters = y[keep], a[keep], b[keep], clusters[keep]
    base["n_dropped_nonfinite"] = n_dropped
    if n_dropped:
        base["note"] = (f"{n_dropped} case(s) dropped from BOTH models "
                        "(non-finite probability in at least one)")
    if len(y) == 0 or len(np.unique(y)) < 2:
        err = f"no usable paired cases after dropping {n_dropped} non-finite/invalid rows"
        return [dict(base, level="slice", delong={"reason": err, "p": None},
                     cluster_bootstrap_diff={"reason": err, "p": None},
                     p_raw=None, reason=err)]

    out = []

    slice_delong = delong_test(y, a, b)
    slice_delong["caveat"] = (
        "slices within a patient are correlated; DeLong's variance assumes "
        "independent cases, so this p-value is anti-conservative. The "
        "patient-level row is the one that satisfies the assumption."
    )
    out.append(dict(
        base, level="slice", cluster_unit=unit, cluster_source=source,
        n_cases=int(len(y)), n_pos=int((y == 1).sum()),
        n_clusters=int(len(np.unique(clusters))),
        delong=slice_delong,
        cluster_bootstrap_diff=cluster_bootstrap_diff(
            y, a, b, clusters, n_boot=n_boot, seed=seed, alpha=alpha),
        p_raw=slice_delong["p"],
        preferred=False,
    ))

    for how in ("mean", "max"):
        agg_a = aggregate_by_cluster(y, a, clusters, how=how)
        agg_b = aggregate_by_cluster(y, b, clusters, how=how)
        if agg_a["cluster_ids"] != agg_b["cluster_ids"]:
            out.append(dict(base, level=f"patient_{how}",
                            delong={"reason": "cluster ordering mismatch", "p": None},
                            p_raw=None))
            continue
        d = delong_test(agg_a["labels"], agg_a["scores"], agg_b["scores"])
        d["caveat"] = "one observation per patient; DeLong's independence assumption holds"
        out.append(dict(
            base, level=f"patient_{how}", cluster_unit=unit, cluster_source=source,
            n_cases=agg_a["n_clusters"], n_pos=agg_a["n_pos_clusters"],
            n_clusters=agg_a["n_clusters"],
            delong=d,
            cluster_bootstrap_diff=cluster_bootstrap_diff(
                agg_a["labels"], agg_a["scores"], agg_b["scores"],
                np.arange(agg_a["n_clusters"]), n_boot=n_boot, seed=seed, alpha=alpha),
            p_raw=d["p"],
            preferred=(how == "mean"),
        ))
    return out


# ==========================================================================
# Across-seed aggregation
# ==========================================================================

def _mean_sd(values) -> dict:
    v = [x for x in values if x is not None and np.isfinite(x)]
    out = {"n": len(v), "mean": None, "sd": None, "min": None, "max": None,
           "values": [float(x) for x in v], "reason": None}
    if not v:
        out["reason"] = "no evaluable seeds"
        return out
    out["mean"] = float(np.mean(v))
    out["min"] = float(np.min(v))
    out["max"] = float(np.max(v))
    if len(v) < 2:
        out["reason"] = "only one evaluable seed; SD is undefined"
    else:
        out["sd"] = float(np.std(v, ddof=1))
    return out


def aggregate_across_seeds(per_run: list[dict]) -> list[dict]:
    """
    mean +/- SD per (cohort, condition, region) across seeds.

    Caveat carried into the output: these seeds share ONE test fold. The SD
    measures training stochasticity, not sampling uncertainty about the
    population AUC. The clustered bootstrap CI is the sampling-uncertainty
    number and is always wider.
    """
    groups: dict[tuple, list[dict]] = {}
    for r in per_run:
        groups.setdefault((r["cohort"], r["condition"], r["region"]), []).append(r)
    out = []
    for (cohort, condition, region), rs in sorted(groups.items(), key=lambda kv: str(kv[0])):
        def pull(key, field="auc"):
            return [(r.get(key) or {}).get(field) for r in rs]
        fams = sorted({str(r.get("split_family")) for r in rs})
        out.append({
            "cohort": cohort, "condition": condition, "region": region,
            "n_runs": len(rs),
            "seeds": sorted(r["seed"] for r in rs if r["seed"] is not None),
            "split_families": fams,
            "n_pooled": sum(1 for r in rs if r.get("pooled")),
            "mixes_split_families": len(fams) > 1,
            "slice_auc": _mean_sd(pull("slice_level")),
            "patient_mean_auc": _mean_sd(pull("patient_level_mean")),
            "patient_max_auc": _mean_sd(pull("patient_level_max")),
            "slice_sens_at_val_youden": _mean_sd(
                [((r.get("operating_point") or {}).get("slice") or {}).get("sens") for r in rs]),
            "slice_spec_at_val_youden": _mean_sd(
                [((r.get("operating_point") or {}).get("slice") or {}).get("spec") for r in rs]),
            "caveat": (
                "seeds share one test set; this SD is training stochasticity, not "
                "sampling uncertainty -- use the clustered bootstrap CI for that"
                + (". WARNING: this row averages estimates from DIFFERENT split "
                   f"families {fams} (a cross-validated pool and a single split are "
                   "not seeds of one experiment); move the stale layout out of the "
                   "results tree before reading it" if len(fams) > 1 else "")),
        })
    return out


# ==========================================================================
# Printing
# ==========================================================================

def _fmt(x, nd=3):
    if x is None or (isinstance(x, float) and not np.isfinite(x)):
        return "  -  "
    return f"{x:.{nd}f}"


def _fmt_ci(d: dict, nd=3) -> str:
    if d is None:
        return "n/a"
    if d.get("auc") is None:
        return "null"
    if d.get("ci_lo") is None:
        return f"{d['auc']:.{nd}f} [no CI]"
    return f"{d['auc']:.{nd}f} [{d['ci_lo']:.{nd}f}, {d['ci_hi']:.{nd}f}]"


def _rule(width=100, ch="-"):
    return ch * width


def _print_pooling(stats: dict) -> None:
    """Provenance for every out-of-fold pool, including the refusals."""
    pooling = ((stats.get("cv") or {}).get("pooling")) or []
    if not pooling:
        return
    print()
    print("OUT-OF-FOLD POOLING  (K-fold CV partitions the cohort by subject, so the K")
    print("test blocks concatenate into ONE prediction per subject over the whole")
    print("cohort. 'exactly-once' is the assertion that no subject sits in two test")
    print("folds -- a duplicate would be double-weighted and would narrow every")
    print("interval below, so a failure REFUSES the estimate rather than reporting it.)")
    print(_rule(108))
    print(f"{'cohort/condition/seed':<34} {'folds':<14} {'slices':>7} {'subj':>6} "
          f"{'pos subj':>8} {'unit':<11} {'exactly-once':<13}")
    print(_rule(108))
    for p in pooling:
        who = f"{p['cohort']}/{p['condition']}/s{p['seed']}"
        folds = ",".join(str(k) for k in p["folds"]) if p["folds"] else "-"
        if not p["ok"]:
            verdict = "REFUSED"
        elif p.get("single_fold"):
            verdict = "OK (1 fold!)"
        else:
            verdict = "OK"
        print(f"{who:<34} {folds:<14} {p['n_slices']:>7} {p['n_subjects']:>6} "
              f"{p['n_pos_subjects']:>8} {str(p['cluster_unit'] or '-'):<11} {verdict:<13}")
        if not p["ok"]:
            print(f"  ! {who}: {p['reason']}")
        for w in p["warnings"]:
            print(f"  ! {who}: {w}")
        for f in p.get("auc_recheck_failures") or []:
            print(f"  ! {who}: fold {f['fold']} AUC recomputed {f['recomputed']:.6f} vs "
                  f"stage-3 reported {f['reported']:.6f}")
    print(_rule(108))


def _print_fold_dispersion(stats: dict) -> None:
    """
    Per-fold estimates as a dispersion diagnostic.

    These are NOT the headline and are not in the comparison family. They answer
    a question the pooled interval cannot: how much does the estimate move with
    WHICH subjects were held out? A fold-to-fold range wider than the pooled CI
    says the pooled interval understates the real uncertainty.
    """
    disp = ((stats.get("cv") or {}).get("fold_dispersion")) or []
    if not disp:
        return
    rows = [d for d in disp if d["level"] == "patient_level_mean"] or disp
    print()
    print("PER-FOLD DISPERSION  (DIAGNOSTIC ONLY -- these five numbers are five")
    print("underpowered estimates of ONE quantity. The pooled row is the headline;")
    print("these are here because a fold-to-fold spread wider than the pooled CI is")
    print("itself a finding. They never enter the comparison family.)")
    print(_rule(108))
    print(f"{'cohort/condition/seed':<30} {'level':<14} {'k':>2} "
          f"{'per-fold AUCs':<34} {'range':>7} {'pooled CI w':>12} {'ratio':>7}")
    print(_rule(108))
    for d in rows:
        who = f"{d['cohort']}/{d['condition']}/s{d['seed']}"
        aucs = " ".join(f"{v:.3f}" for v in d["fold_aucs"]) or "-"
        if len(aucs) > 33:
            aucs = aucs[:30] + "..."
        rng = _fmt(d["fold_auc_range"])
        pw = _fmt(d["pooled_ci_width"])
        ratio = ("-" if d["spread_vs_pooled_ci"] is None
                 else f"{d['spread_vs_pooled_ci']:.2f}x")
        print(f"{who:<30} {d['level'].replace('_level', ''):<14} {d['n_evaluable_folds']:>2} "
              f"{aucs:<34} {rng:>7} {pw:>12} {ratio:>7}")
    wide = [d for d in rows if (d["spread_vs_pooled_ci"] or 0) > 1.0]
    if wide:
        print(_rule(108))
        for d in wide:
            print(f"  ! {d['cohort']}/{d['condition']}/s{d['seed']} [{d['level']}]: {d['note']}")
    print(_rule(108))


def _print_family(stats: dict) -> None:
    """The multiplicity family, itemised, so its size can be checked by eye."""
    h = stats.get("holm") or {}
    per_pair = h.get("comparisons_per_cohort_pair") or {}
    if not per_pair:
        return
    print()
    print("MULTIPLICITY FAMILY  (one comparison per cohort x condition-pair x seed x")
    print("level. Folds are pooled BEFORE comparison, so a 5-fold cohort costs one")
    print("comparison, not five.)")
    print(_rule(108))
    print(f"{'cohort :: condition pair':<52} {'emitted':>9} {'evaluable':>10}")
    print(_rule(108))
    ev = h.get("evaluable_per_cohort_pair") or {}
    for pair, n in per_pair.items():
        print(f"{pair:<52} {n:>9} {ev.get(pair, 0):>10}")
    print(_rule(108))
    fm = h.get("fold_multiplicity") or {}
    print(f"cohort x condition-pair combinations : {h.get('n_cohort_condition_pairs')}")
    print(f"comparisons emitted                  : {h.get('n_comparisons_emitted')}")
    print(f"HOLM FAMILY SIZE (evaluable)         : {h.get('family_size')}")
    if fm:
        print(f"had each fold been an experiment     : {fm.get('n_if_each_fold_were_independent')} "
              f"comparisons ({fm.get('inflation_avoided')} avoided by pooling)")
    print(f"fold dimension present in family     : "
          f"{'YES -- THIS IS A BUG' if h.get('fold_dimension_in_family') else 'no'}")
    print(_rule(108))


def print_report(stats: dict) -> None:
    cfg = stats["config"]
    print()
    print(_rule(108, "="))
    print("PhaseDx stage 4 -- statistics")
    print(_rule(108, "="))
    cv = stats.get("cv") or {}
    print(f"results dir : {cfg['results_dir']}")
    print(f"estimates   : {len(stats['runs'])} "
          f"({cv.get('n_pooled', 0)} out-of-fold pooled, "
          f"{sum(1 for r in stats['runs'] if not r.get('pooled'))} single-split"
          + (f", {cv.get('n_refused', 0)} REFUSED" if cv.get("n_refused") else "") + ")")
    print(f"fold files  : {len(cv.get('per_fold') or [])} "
          "(diagnostic only; not in the comparison family)")
    print(f"bootstrap   : {cfg['n_boot']} replicates, clustered on the unit shown per row, "
          f"{int((1-cfg['alpha'])*100)}% percentile CI")
    print(f"generated   : {stats['generated']}")
    if not stats["runs"]:
        print("\nNo stage-3 run JSONs found. Nothing to report.")
        return

    _print_pooling(stats)

    print()
    print("PER-ESTIMATE TEST PERFORMANCE  (every AUC carries its clustered 95% CI; a")
    print("bare point estimate on a 4-patient fold is not interpretable. Rows tagged")
    print("'~pooled' are one out-of-fold prediction per subject over the whole cohort,")
    print("not one fold.)")
    print(_rule(108))
    hdr = (f"{'run':<36} {'unit':<10} {'pat':>4} {'pos':>4} "
           f"{'slice AUC [95% CI]':<26} {'patient-mean AUC [95% CI]':<27} {'pat-max AUC':<12}")
    print(hdr)
    print(_rule(108))
    for r in stats["runs"]:
        sl, pm, px = r["slice_level"], r["patient_level_mean"], r["patient_level_max"]
        npat = (sl or {}).get("n_clusters", 0)
        npos = (sl or {}).get("n_pos_clusters", 0)
        print(f"{r['tag']:<36} {str(r['cluster_unit']):<10} {npat:>4} {npos:>4} "
              f"{_fmt_ci(sl):<26} {_fmt_ci(pm):<27} {_fmt_ci(px):<12}")
    print(_rule(108))
    for r in stats["runs"]:
        for key, lab in (("slice_level", "slice"), ("patient_level_mean", "patient-mean"),
                         ("patient_level_max", "patient-max")):
            d = r.get(key) or {}
            if d.get("reason"):
                print(f"  ! {r['tag']} [{lab}]: {d['reason']}")
        for w in r["warnings"]:
            print(f"  ! {r['tag']}: {w}")

    rows = []
    for r in stats["runs"]:
        for key, lab in (("slice_level", "slice"), ("patient_level_mean", "patient-mean"),
                         ("patient_level_max", "patient-max")):
            d = r.get(key) or {}
            skipped = (d.get("n_skipped_single_class") or 0) + (d.get("n_skipped_single_cluster") or 0)
            if skipped:
                rows.append((r["tag"], lab, d, skipped / max(d.get("n_boot_requested") or 1, 1)))
    if rows:
        print()
        print("BOOTSTRAP REPLICATE ACCOUNTING  (a replicate is skipped when the drawn")
        print("patients are all one class, or when fewer than 2 distinct patients are")
        print("drawn. A large skip fraction means the fold is too small for the")
        print("interval to be taken at face value -- it is conditioned on the")
        print("replicates that happened to be evaluable.)")
        print(_rule(108))
        print(f"{'run':<32} {'level':<13} {'used/req':>12} {'1-class':>9} {'1-cluster':>10} {'skipped':>8}")
        print(_rule(108))
        ordered = sorted(rows, key=lambda t: -t[3])
        for tag, lab, d, frac in ordered[:24]:
            print(f"{tag:<32} {lab:<13} "
                  f"{str(d['n_boot_used']) + '/' + str(d['n_boot_requested']):>12} "
                  f"{d['n_skipped_single_class']:>9} {d['n_skipped_single_cluster']:>10} "
                  f"{frac:>7.0%}")
        if len(ordered) > 24:
            print(f"... and {len(ordered) - 24} more (run/level, all with smaller skip "
                  "fractions); the full accounting is in statistics.json")

    print()
    print("CLUSTERING DIAGNOSTIC  (the headline interval is always the clustered one;")
    print("the naive slice-level interval is shown only for contrast. Expect the")
    print("clustered interval to be 2-3x WIDER on realistic folds. It can invert on")
    print("folds with very few patients, where the clustered bootstrap has almost no")
    print("distinct patient draws left after degenerate replicates are skipped -- that")
    print("is a sign the fold is too small to bootstrap, not that clustering is optional.)")
    print(_rule(108))
    print(f"{'run':<34} {'clustered CI width':>20} {'naive slice CI width':>22} {'ratio':>8}")
    print(_rule(108))
    for r in stats["runs"]:
        sl = r["slice_level"] or {}
        nv = sl.get("naive_slice_bootstrap_ci") or {}
        if sl.get("ci_lo") is None or nv.get("width") is None:
            print(f"{r['tag']:<34} {'-':>20} {'-':>22} {'-':>8}")
            continue
        w_c = sl["ci_hi"] - sl["ci_lo"]
        w_n = nv["width"]
        ratio = w_c / w_n if w_n > 0 else float("nan")
        print(f"{r['tag']:<34} {w_c:>20.3f} {w_n:>22.3f} {ratio:>8.2f}x")

    _print_fold_dispersion(stats)

    print()
    print("OPERATING POINT  (Youden threshold chosen on VALIDATION, applied unchanged to")
    print("TEST. For a pooled cross-validated estimate the threshold is fit on EACH")
    print("FOLD'S OWN validation set and applied to that fold's test slices; the counts")
    print("are then summed, and the threshold column shows the median of the per-fold")
    print("thresholds purely as a summary.)")
    print(_rule(108))
    print(f"{'run':<36} {'thr(val)':>9} {'level':<13} {'sens [95% CI]':<24} {'spec [95% CI]':<24} {'n':>5}")
    print(_rule(108))
    for r in stats["runs"]:
        op = r.get("operating_point") or {}
        if op.get("reason"):
            print(f"{r['tag']:<36} {'-':>9} {'-':<13} {op['reason']}")
            continue
        for level, key, thrkey in (("slice", "slice", "slice"),
                                   ("patient-mean", "patient_mean", "patient_mean")):
            m = op.get(key) or {}
            thr = ((op.get("validation") or {}).get(thrkey) or {}).get("threshold")
            if thr is None:
                thr = m.get("threshold")
            s_ci = m.get("sens_ci") or [None, None]
            p_ci = m.get("spec_ci") or [None, None]
            sens = (f"{_fmt(m.get('sens'))} [{_fmt(s_ci[0])}, {_fmt(s_ci[1])}]"
                    if m.get("sens") is not None else "  -  ")
            spec = (f"{_fmt(m.get('spec'))} [{_fmt(p_ci[0])}, {_fmt(p_ci[1])}]"
                    if m.get("spec") is not None else "  -  ")
            print(f"{r['tag']:<34} {_fmt(thr):>9} {level:<13} {sens:<24} {spec:<24} "
                  f"{m.get('n') if m.get('n') is not None else '-':>5}")
            if m.get("reason"):
                print(f"  ! {r['tag']} [{level}]: {m['reason']}")

    print()
    print("ACROSS SEEDS  (mean +/- SD over seeds sharing ONE test fold: training")
    print("stochasticity, NOT sampling uncertainty)")
    print(_rule(108))
    print(f"{'cohort':<14} {'condition':<11} {'region':<11} {'n':>2} "
          f"{'slice AUC':<20} {'patient-mean AUC':<20} {'patient-max AUC':<20}")
    print(_rule(108))
    for g in stats["across_seeds"]:
        def cell(d):
            if d["mean"] is None:
                return "   -   "
            if d["sd"] is None:
                return f"{d['mean']:.3f} (n=1, no SD)"
            return f"{d['mean']:.3f} +/- {d['sd']:.3f}"
        print(f"{str(g['cohort']):<14} {str(g['condition']):<11} {str(g['region']):<11} "
              f"{g['n_runs']:>2} {cell(g['slice_auc']):<20} "
              f"{cell(g['patient_mean_auc']):<20} {cell(g['patient_max_auc']):<20}")

    print()
    print("PAIRWISE COMPARISONS  (DeLong for correlated ROC curves; "
          "clustered bootstrap difference alongside)")
    print(_rule(108))
    print(f"{'cohort/seed':<22} {'A vs B':<22} {'level':<14} {'dAUC':>7} "
          f"{'DeLong p':>9} {'Holm p':>8} {'boot dAUC 95% CI':<22}")
    print(_rule(108))
    for c in stats["comparisons"]:
        d = c.get("delong") or {}
        cb = c.get("cluster_bootstrap_diff") or {}
        boot = ("-" if cb.get("ci_lo") is None
                else f"[{cb['ci_lo']:+.3f}, {cb['ci_hi']:+.3f}]")
        who = f"{c['model_a']} vs {c['model_b']}"
        key = f"{c['cohort']}/s{c['seed']}" + ("~pooled" if c.get("pooled") else "")
        diff = d.get("diff")
        print(f"{key:<22} {who:<22} {c['level']:<14} "
              f"{('%+.3f' % diff) if diff is not None else '   -   ':>7} "
              f"{_fmt(d.get('p'), 4):>9} {_fmt(c.get('p_holm'), 4):>8} {boot:<22}")
        if d.get("reason"):
            print(f"  ! {key} {who} [{c['level']}]: {d['reason']}")
    print(_rule(108))
    print("Holm adjustment is over all evaluable comparisons in this file "
          f"(family size {stats['holm']['family_size']}).")
    print("Slice-level DeLong p-values assume independent cases and are")
    print("anti-conservative under within-patient correlation; prefer the patient_mean rows.")

    _print_family(stats)

    if stats["warnings"]:
        print()
        print("GLOBAL WARNINGS")
        print(_rule(108))
        for w in stats["warnings"]:
            print(f"  ! {w}")
    print()


# ==========================================================================
# Driver
# ==========================================================================

def run_statistics(results_dir: Path, cache_dir: Path, cohort_dir: Path,
                   n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                   cluster_unit: str = "auto") -> dict:
    runs = load_runs(results_dir)
    warnings: list[str] = []
    if not runs:
        warnings.append(f"no stage-3 run JSONs found in {results_dir}")

    cohorts = sorted({r.get("cohort") for r in runs if r.get("cohort")})
    cluster_maps: dict[str, dict] = {}
    for c in cohorts:
        if cluster_unit == "patient":
            cluster_maps[c] = {}
            warnings.append(f"{c}: --cluster-unit patient forces clustering on patient_id")
            continue
        info = build_cluster_map(c, cache_dir, cohort_dir)
        cluster_maps[c] = info["map"]
        if info["reason"]:
            msg = f"{c}: falling back to patient_id as the cluster unit ({info['reason']})"
            if cluster_unit == "subject":
                raise RuntimeError(msg + " -- but --cluster-unit subject was requested")
            warnings.append(msg)
        else:
            logger.info("%s: cluster unit subject_id via %s", c, info["source"])

    # ---------------------------------------------------------------------
    # Layout detection and out-of-fold pooling
    # ---------------------------------------------------------------------
    # One ESTIMATE per (cohort, region, split family, condition, seed). Runs
    # sharing that key that carry a fold index are the folds of one
    # cross-validated estimate and get pooled; a key with a single non-fold run
    # is the single-split layout and passes through untouched.
    units: dict[tuple, list[dict]] = {}
    for r in runs:
        units.setdefault(unit_key(r), []).append(r)

    # Expected fold set per (cohort, region): the union of folds seen anywhere in
    # that cohort's CV family. A condition/seed missing one of them is pooling
    # over less than the cohort and must say so.
    expected_folds: dict[tuple, set] = {}
    for r in runs:
        if r.get("_fold") is not None:
            expected_folds.setdefault(
                (str(r.get("cohort")), str(r.get("region", "full"))), set()
            ).add(int(r["_fold"]))

    analysis_runs: list[dict] = []      # exactly one run object per estimate
    per_fold_runs: list[dict] = []      # folds, for the dispersion diagnostic only
    pooling: list[dict] = []
    refused: list[dict] = []

    for key, rs in sorted(units.items(), key=lambda kv: str(kv[0])):
        cohort, region, family, condition, sd = key
        cmap = cluster_maps.get(cohort)
        if family == "cv":
            per_fold_runs.extend(rs)
            pooled, info = pool_folds(
                rs, cmap, expected_folds.get((cohort, region)))
            info["unit_key"] = list(key)
            pooling.append(info)
            if pooled is None:
                refused.append(info)
                warnings.append(
                    f"{cohort}/{condition}/seed{sd}: REFUSED to pool "
                    f"{len(rs)} fold(s) -- {info['reason']}. No pooled estimate is "
                    "reported for this condition, and every comparison involving it "
                    "will be null."
                )
                continue
            for w in info["warnings"]:
                warnings.append(f"{cohort}/{condition}/seed{sd} [pooling]: {w}")
            analysis_runs.append(pooled)
            continue
        # Single-split layout: exactly as before.
        if len(rs) > 1:
            paths = ", ".join(str(r.get("_path")) for r in rs)
            warnings.append(
                f"{cohort}/{condition}/seed{sd} in split family {family!r}: "
                f"{len(rs)} run files share one identity and none carries a fold "
                f"index ({paths}); using the first and ignoring the rest, because "
                "pooling runs that may have scored the same subjects would "
                "double-weight them"
            )
        analysis_runs.append(rs[0])

    # A cohort present in BOTH layouts is two experiments, not one. That is not
    # automatically wrong -- a stale flat sweep can legitimately sit next to a CV
    # sweep -- but stage 6 keys its headline on (cohort, condition) and would
    # average the two, so it is called out loudly rather than silently merged.
    fams_by_cohort: dict[str, set] = {}
    for r in analysis_runs:
        fams_by_cohort.setdefault(str(r.get("cohort")), set()).add(str(r.get("_split_family")))
    for cohort, fams in sorted(fams_by_cohort.items()):
        if len(fams) > 1 and "cv" in fams:
            others = sorted(f for f in fams if f != "cv")
            stale = sorted({str(r.get("_path")) for r in analysis_runs
                            if str(r.get("cohort")) == cohort
                            and str(r.get("_split_family")) != "cv"})
            warnings.append(
                f"{cohort}: results exist in BOTH the cross-validated layout and the "
                f"single-split layout {others}. These are different experiments on "
                "different test sets; they are kept separate here (the pooled rows are "
                "tagged '~pooled') and both enter the Holm family, which is the "
                "conservative direction. Stage 6 keys on (cohort, condition) and will "
                "average them unless the single-split files are moved out: "
                + "; ".join(stale)
            )

    per_run = [analyse_run(r, cluster_maps.get(r.get("cohort")), n_boot, seed, alpha)
               for r in analysis_runs]

    # The pooled operating point cannot come from analyse_run: it needs each
    # fold's OWN validation set, which the pooled payload deliberately does not
    # carry (see pooled_operating_point).
    fold_runs_by_key: dict[tuple, list[dict]] = {}
    for r in per_fold_runs:
        fold_runs_by_key.setdefault(unit_key(r), []).append(r)
    for res, r in zip(per_run, analysis_runs):
        if r.get("pooled"):
            res["operating_point"] = pooled_operating_point(
                fold_runs_by_key.get(unit_key(r), []), cluster_maps.get(r.get("cohort")))

    # ---- per-fold dispersion diagnostic (never enters runs or comparisons) ---
    per_fold = [fold_diagnostic(r, cluster_maps.get(r.get("cohort")), n_boot, seed, alpha)
                for r in per_fold_runs]
    dispersion = fold_dispersion(per_fold, per_run)

    # ---------------------------------------------------------------------
    # Pairwise comparisons: same cohort, region, SPLIT FAMILY and seed,
    # different conditions -- one comparison per pair, never one per fold.
    # ---------------------------------------------------------------------
    by_experiment: dict[tuple, list[dict]] = {}
    for r in analysis_runs:
        by_experiment.setdefault(
            (r.get("cohort"), r.get("region", "full"),
             r.get("_split_family"), r.get("seed")), []).append(r)
    comparisons: list[dict] = []
    for key, rs in sorted(by_experiment.items(), key=lambda kv: str(kv[0])):
        rs = sorted(rs, key=lambda r: str(r.get("condition")))
        for ra, rb in combinations(rs, 2):
            comparisons.extend(compare_runs(
                ra, rb, cluster_maps.get(ra.get("cohort")), n_boot, seed, alpha))

    # Holm across every evaluable comparison in the file, plus a per-family view.
    holm_global = holm_adjust([c.get("p_raw") for c in comparisons])
    for c, ph in zip(comparisons, holm_global):
        c["p_holm"] = ph
    fam: dict[tuple, list[int]] = {}
    for i, c in enumerate(comparisons):
        fam.setdefault((c.get("cohort"), c.get("level")), []).append(i)
    for fkey, idxs in fam.items():
        adj = holm_adjust([comparisons[i].get("p_raw") for i in idxs])
        for i, a in zip(idxs, adj):
            comparisons[i]["p_holm_within_cohort_level"] = a
            comparisons[i]["holm_family"] = f"{fkey[0]}/{fkey[1]}"

    n_evaluable = sum(1 for c in comparisons
                      if c.get("p_raw") is not None and np.isfinite(c["p_raw"]))
    holm_info = _verify_family(comparisons, n_evaluable, per_fold_runs)
    warnings.extend(holm_info.pop("_warnings"))

    stats = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "config": {
            "results_dir": str(results_dir),
            "cache_dir": str(cache_dir),
            "cohort_dir": str(cohort_dir),
            "n_boot": int(n_boot),
            "bootstrap_seed": int(seed),
            "alpha": float(alpha),
            "cluster_unit_requested": cluster_unit,
            "ci_method": "percentile bootstrap, resampling clusters (patients/subjects)",
        },
        "methods_note": (
            "AUC CIs come from a bootstrap that resamples PATIENTS/SUBJECTS with "
            "replacement and rebuilds the slice set from the drawn patients; a "
            "slice-level bootstrap would be roughly sqrt(slices-per-patient) too "
            "narrow. Replicates that are single-class or draw <2 distinct clusters "
            "are skipped and counted; note that skipping conditions the bootstrap "
            "distribution on evaluable replicates, which on a fold with a handful "
            "of patients can make the interval narrower than it should be -- read "
            "the skip counts before trusting any interval where they are large. "
            "Thresholds are chosen on validation only. "
            "DeLong's test is the Sun & Xu midrank form; its slice-level p-values "
            "assume independent cases, which within-patient correlation violates, "
            "so patient-level comparisons are the primary inference."
        ),
        "runs": per_run,
        "across_seeds": aggregate_across_seeds(per_run),
        "comparisons": comparisons,
        "holm": holm_info,
        "cv": {
            "layout_note": (
                "Estimates from a `<cohort>_cv<k>/` layout are OUT-OF-FOLD POOLED: the "
                "test blocks of the folds are concatenated into one prediction per "
                "subject over the whole cohort, after asserting that every subject "
                "appears in exactly one fold. Single-directory cohorts are unchanged. "
                "Per-fold numbers below are a dispersion diagnostic only and are NOT "
                "in `runs`, `across_seeds` or `comparisons`."
            ),
            "pooling": pooling,
            "n_pooled": sum(1 for p in pooling if p["ok"]),
            "n_refused": len(refused),
            "per_fold": per_fold,
            "fold_dispersion": dispersion,
        },
        "warnings": warnings,
    }
    return stats


def _verify_family(comparisons: list[dict], n_evaluable: int,
                   per_fold_runs: list[dict]) -> dict:
    """
    Describe the multiplicity family and CHECK that folds are not in it.

    The check that matters: a cross-validated cohort must contribute one
    comparison per (condition-pair, seed, level), not five. `fold_multiplicity`
    reports how many comparisons the naive recursive reading would have produced,
    so the inflation avoided is a number in the output rather than a claim.
    """
    ws: list[str] = []
    per_pair: dict[str, int] = {}
    per_pair_eval: dict[str, int] = {}
    for c in comparisons:
        pair = f"{c.get('cohort')}::{c.get('model_a')}-vs-{c.get('model_b')}"
        per_pair[pair] = per_pair.get(pair, 0) + 1
        if c.get("p_raw") is not None and np.isfinite(c["p_raw"]):
            per_pair_eval[pair] = per_pair_eval.get(pair, 0) + 1

    stray = [c.get("tag_a") for c in comparisons if c.get("fold") is not None]
    if stray:
        ws.append(
            f"{len(stray)} comparison(s) carry a fold index; folds must be pooled "
            "before comparison, never compared fold by fold. This is a bug -- the "
            "reported Holm family is inflated by the fold dimension"
        )

    # What the naive per-fold reading would have cost: within each fold of each
    # (cohort, region, seed), every condition pair at every level.
    by_cvexp: dict[tuple, dict[int, set]] = {}
    for r in per_fold_runs:
        exp = (str(r.get("cohort")), str(r.get("region", "full")), r.get("seed"))
        by_cvexp.setdefault(exp, {}).setdefault(int(r["_fold"]), set()).add(str(r.get("condition")))
    n_levels = 3        # slice, patient_mean, patient_max
    naive = 0
    for folds in by_cvexp.values():
        for conds in folds.values():
            n = len(conds)
            naive += (n * (n - 1) // 2) * n_levels
    pooled_cmps = sum(1 for c in comparisons if c.get("pooled"))

    return {
        "method": "holm-bonferroni step-down, monotonicity enforced",
        "family_size": n_evaluable,
        "family_definition": (
            "all evaluable comparisons in this file. A cross-validated cohort is "
            "pooled out-of-fold FIRST, so it contributes one comparison per "
            "(cohort, condition-pair, seed, level) -- not one per fold."
        ),
        "n_comparisons_emitted": len(comparisons),
        "n_cohort_condition_pairs": len(per_pair),
        "comparisons_per_cohort_pair": dict(sorted(per_pair.items())),
        "evaluable_per_cohort_pair": dict(sorted(per_pair_eval.items())),
        "levels_per_comparison": n_levels,
        "fold_multiplicity": {
            "n_pooled_comparisons_emitted": pooled_cmps,
            "n_if_each_fold_were_independent": naive,
            "inflation_avoided": naive - pooled_cmps,
            "note": ("the family that a naive recursive glob would have built, versus "
                     "the one built here; the difference is the multiplicity the "
                     "pooling removes"),
        },
        "fold_dimension_in_family": bool(stray),
        "_warnings": ws,
    }


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        v = float(o)
        return v if np.isfinite(v) else None
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return str(o)


# ==========================================================================
# Self-test
# ==========================================================================

def _sim_clustered(rng, n_patients=20, slices=15, mu=1.0, su=1.0, se=1.0):
    """
    Exchangeable random-effects generator with PATIENT-level labels.

    score = mu*y + u_patient + e_slice,  u ~ N(0, su^2), e ~ N(0, se^2)

    Because positives and negatives are different patients, the population AUC
    is exactly Phi(mu / sqrt(2*su^2 + 2*se^2)) -- an analytic truth to check
    bootstrap coverage against, with no Monte Carlo error in the target.
    """
    y_pat = rng.integers(0, 2, size=n_patients)
    pid = np.repeat(np.arange(n_patients), slices)
    u = np.repeat(rng.normal(0, su, size=n_patients), slices)
    y = np.repeat(y_pat, slices)
    s = mu * y + u + rng.normal(0, se, size=n_patients * slices)
    return y, s, pid


def _true_auc(mu, su, se):
    return float(0.5 * math.erfc(-(mu / math.sqrt(2 * su * su + 2 * se * se)) / math.sqrt(2)))


class _Check:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def ok(self, cond, label, detail=""):
        mark = "PASS" if cond else "FAIL"
        if cond:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{mark}] {label}" + (f"  {detail}" if detail else ""))
        return bool(cond)


def self_test(quick: bool = False) -> int:
    print(_rule(100, "="))
    print("s04_stats self-test")
    print(_rule(100, "="))
    c = _Check()
    rng = np.random.default_rng(20260727)

    # ---------------------------------------------------------------- midrank
    print("\n[1] midrank primitive")
    x = np.array([3.0, 1.0, 1.0, 2.0, 3.0, 3.0])
    got = compute_midrank(x)
    want = np.array([5.0, 1.5, 1.5, 3.0, 5.0, 5.0])   # hand-computed
    c.ok(np.allclose(got, want), "midranks with ties match hand computation",
         f"{got.tolist()}")
    try:
        from scipy.stats import rankdata
        big = rng.integers(0, 5, size=500).astype(float)
        c.ok(np.allclose(compute_midrank(big), rankdata(big, method="average")),
             "midranks match scipy.stats.rankdata(average) on 500 tied values")
    except ImportError:
        print("  [SKIP] scipy not available")

    # -------------------------------------------------------------------- AUC
    print("\n[2] AUC against an independent implementation")
    try:
        from sklearn.metrics import average_precision_score, roc_auc_score
        for name, (yy, ss) in {
            "continuous": (rng.integers(0, 2, 400), rng.normal(size=400)),
            "heavily tied": (rng.integers(0, 2, 400),
                             rng.integers(0, 4, 400).astype(float)),
            "all-tied scores": (rng.integers(0, 2, 200), np.zeros(200)),
        }.items():
            mine, theirs = auc_midrank(yy, ss), roc_auc_score(yy, ss)
            c.ok(abs(mine - theirs) < 1e-12, f"AUC matches sklearn ({name})",
                 f"mine={mine:.12f} sklearn={theirs:.12f}")
        yy = rng.integers(0, 2, 300)
        ss = rng.normal(size=300)
        c.ok(abs(average_precision(yy, ss) - average_precision_score(yy, ss)) < 1e-12,
             "average precision matches sklearn")
    except ImportError:
        print("  [SKIP] sklearn not available")

    # ----------------------------------------------------------------- DeLong
    print("\n[3] DeLong variance vs a large-sample EMPIRICAL variance")
    print("    (the point of this check: the analytic covariance estimator is")
    print("     validated against brute-force resimulation, not against itself)")
    n_sims = 400 if quick else 3000
    n_case = 60
    mu_a, mu_b = 1.0, 0.6
    aucs_a, aucs_b, var_a_hat, var_b_hat, var_d_hat = [], [], [], [], []
    for _ in range(n_sims):
        yy = np.r_[np.ones(n_case // 2), np.zeros(n_case // 2)].astype(int)
        latent = rng.normal(size=n_case)
        sa = mu_a * yy + latent + 0.7 * rng.normal(size=n_case)
        sb = mu_b * yy + latent + 0.7 * rng.normal(size=n_case)   # correlated with A
        d = delong_test(yy, sa, sb)
        aucs_a.append(d["auc_a"])
        aucs_b.append(d["auc_b"])
        var_a_hat.append(d["var_a"])
        var_b_hat.append(d["var_b"])
        var_d_hat.append(d["se_diff"] ** 2)
    emp_a = float(np.var(aucs_a, ddof=1))
    emp_b = float(np.var(aucs_b, ddof=1))
    emp_d = float(np.var(np.array(aucs_a) - np.array(aucs_b), ddof=1))
    mean_a, mean_b, mean_d = map(lambda v: float(np.mean(v)),
                                 (var_a_hat, var_b_hat, var_d_hat))
    print(f"    model A : DeLong var {mean_a:.6f}  empirical var {emp_a:.6f}  "
          f"ratio {mean_a/emp_a:.3f}")
    print(f"    model B : DeLong var {mean_b:.6f}  empirical var {emp_b:.6f}  "
          f"ratio {mean_b/emp_b:.3f}")
    print(f"    A - B   : DeLong var {mean_d:.6f}  empirical var {emp_d:.6f}  "
          f"ratio {mean_d/emp_d:.3f}")
    tol = 0.25 if quick else 0.12
    c.ok(abs(mean_a / emp_a - 1) < tol, "DeLong var(AUC_A) agrees with empirical variance",
         f"ratio {mean_a/emp_a:.3f}")
    c.ok(abs(mean_b / emp_b - 1) < tol, "DeLong var(AUC_B) agrees with empirical variance",
         f"ratio {mean_b/emp_b:.3f}")
    c.ok(abs(mean_d / emp_d - 1) < tol,
         "DeLong var(AUC_A - AUC_B) agrees (covariance term is right)",
         f"ratio {mean_d/emp_d:.3f}")

    print("\n[4] DeLong type-I error under a true null (equal AUCs, correlated models)")
    n_null = 300 if quick else 1500
    rejects = 0
    for _ in range(n_null):
        yy = np.r_[np.ones(40), np.zeros(40)].astype(int)
        latent = rng.normal(size=80)
        sa = 1.0 * yy + latent + 0.8 * rng.normal(size=80)
        sb = 1.0 * yy + latent + 0.8 * rng.normal(size=80)
        d = delong_test(yy, sa, sb)
        if d["p"] is not None and np.isfinite(d["p"]) and d["p"] < 0.05:
            rejects += 1
    rate = rejects / n_null
    se_rate = math.sqrt(0.05 * 0.95 / n_null)
    print(f"    nominal 0.05, empirical {rate:.4f} (MC se {se_rate:.4f}, {n_null} sims)")
    c.ok(abs(rate - 0.05) < max(0.02, 3 * se_rate),
         "DeLong rejection rate is near nominal under the null", f"{rate:.4f}")

    print("\n[5] DeLong degenerate-input guards")
    yy = np.r_[np.ones(5), np.zeros(5)].astype(int)
    s1 = rng.normal(size=10)
    c.ok(delong_test(np.zeros(10, int), s1, s1)["p"] is None
         and "single-class" in delong_test(np.zeros(10, int), s1, s1)["reason"],
         "single-class fold -> null with reason")
    same = delong_test(yy, s1, s1.copy())
    c.ok(same["p"] is None and "degenerate" in same["reason"],
         "identical rankings -> null with reason, not p=1.0 or a crash")
    bad = delong_test(yy, np.r_[np.nan, s1[1:]], s1)
    c.ok(bad["p"] is None and "non-finite" in bad["reason"], "NaN score -> null with reason")
    # The real prostate DWI patient-level fold: 3 positive patients, 1 negative.
    thin = delong_test(np.r_[1, 1, 1, 0], np.r_[.8, .6, .4, .3], np.r_[.7, .5, .2, .4])
    c.ok(thin["p"] is None and ">=2 positive and >=2 negative" in thin["reason"],
         "1 negative case -> null with reason, no NaN and no numpy warning",
         thin["reason"][:52])
    import warnings as _w
    with _w.catch_warnings():
        _w.simplefilter("error")     # any RuntimeWarning becomes an exception
        delong_test(np.r_[1, 1, 1, 0], np.r_[.8, .6, .4, .3], np.r_[.7, .5, .2, .4])
        delong_test(np.r_[1, 1, 0, 0], np.r_[.8, .6, .4, .3], np.r_[.7, .5, .2, .4])
    c.ok(True, "no numpy RuntimeWarning is emitted on thin folds")

    # ------------------------------------------------- clustered vs naive boot
    print("\n[6] clustered bootstrap: coverage of the TRUE AUC vs the naive")
    print("    slice-level bootstrap (this is the reviewer's objection, measured)")
    mu, su, se = 1.0, 1.2, 0.8
    truth = _true_auc(mu, su, se)
    n_cov = 60 if quick else 200
    nb = 300 if quick else 500
    cov_cluster, cov_naive, w_cluster, w_naive, n_used = 0, 0, [], [], 0
    for i in range(n_cov):
        y, s, pid = _sim_clustered(rng, n_patients=20, slices=15, mu=mu, su=su, se=se)
        if len(np.unique(y)) < 2:
            continue
        n_used += 1
        cb = cluster_bootstrap_auc(y, s, pid, n_boot=nb, seed=int(rng.integers(1e9)))
        nv = naive_slice_bootstrap_auc(y, s, n_boot=nb, seed=int(rng.integers(1e9)))
        if cb["ci_lo"] is not None:
            cov_cluster += int(cb["ci_lo"] <= truth <= cb["ci_hi"])
            w_cluster.append(cb["ci_hi"] - cb["ci_lo"])
        if nv["ci_lo"] is not None:
            cov_naive += int(nv["ci_lo"] <= truth <= nv["ci_hi"])
            w_naive.append(nv["width"])
    print(f"    true AUC = Phi(mu/sqrt(2su^2+2se^2)) = {truth:.4f}   ({n_used} datasets, "
          f"20 patients x 15 slices)")
    print(f"    clustered-on-patient bootstrap : coverage {cov_cluster/n_used:.3f}   "
          f"mean width {np.mean(w_cluster):.3f}")
    print(f"    naive slice-level bootstrap    : coverage {cov_naive/n_used:.3f}   "
          f"mean width {np.mean(w_naive):.3f}")
    c.ok(np.mean(w_cluster) > 1.5 * np.mean(w_naive),
         "clustered CI is substantially wider than the naive one",
         f"{np.mean(w_cluster)/np.mean(w_naive):.2f}x")
    c.ok(cov_cluster / n_used > cov_naive / n_used,
         "clustered CI covers the truth more often than the naive one",
         f"{cov_cluster/n_used:.3f} vs {cov_naive/n_used:.3f}")
    c.ok(cov_cluster / n_used >= 0.85,
         "clustered coverage is in the right neighbourhood of 0.95",
         f"{cov_cluster/n_used:.3f}")

    print("\n[7] clustered bootstrap bookkeeping and guards")
    y, s, pid = _sim_clustered(rng, n_patients=8, slices=10)
    r = cluster_bootstrap_auc(y, s, pid, n_boot=200, seed=1)
    c.ok(abs(r["auc"] - auc_midrank(y, s)) < 1e-12,
         "point estimate equals the full-data AUC")
    c.ok(r["n_boot_used"] + r["n_skipped_single_class"] + r["n_skipped_single_cluster"] == 200,
         "used + skipped replicates account for every requested replicate",
         f"{r['n_boot_used']}+{r['n_skipped_single_class']}+{r['n_skipped_single_cluster']}")
    # 4 patients, 3 positive: the real prostate DWI test fold shape. One
    # positive patient (P2) is scored below the only negative, so the true
    # fold AUC is 2/3 and the interval has something to be wide about.
    y4 = np.repeat([1, 1, 1, 0], 30)
    p4 = np.repeat([0.7, 0.2, 0.6, 0.3], 30) + rng.normal(0, .01, 120)
    pid4 = np.repeat(["P1", "P2", "P3", "P4"], 30)
    r4 = cluster_bootstrap_auc(y4, p4, pid4, n_boot=500, seed=2)
    print(f"    4-patient/3-positive fold: AUC {r4['auc']:.3f} "
          f"CI [{r4['ci_lo']:.3f}, {r4['ci_hi']:.3f}], "
          f"{r4['n_boot_used']}/500 replicates used, "
          f"{r4['n_skipped_single_class']} single-class skipped")
    c.ok(abs(r4["auc"] - 2 / 3) < 0.02, "fold AUC is the expected 2/3", f"{r4['auc']:.3f}")
    c.ok(r4["n_skipped_single_class"] > 50,
         "a 3-positive/1-negative fold really does produce many single-class replicates",
         f"{r4['n_skipped_single_class']}/500")
    c.ok(r4["ci_hi"] - r4["ci_lo"] > 0.3,
         "and the resulting interval is honestly wide",
         f"width {r4['ci_hi']-r4['ci_lo']:.3f}")
    single = cluster_bootstrap_auc(np.r_[1, 1, 0, 0], np.r_[.9, .8, .2, .1],
                                   np.array(["A"] * 4, dtype=object), n_boot=100)
    c.ok(single["ci_lo"] is None and "cluster" in single["reason"],
         "<2 clusters -> AUC reported, interval refused, reason given")
    sc = cluster_bootstrap_auc(np.zeros(10, int), rng.normal(size=10),
                               np.repeat(["A", "B"], 5), n_boot=100)
    c.ok(sc["auc"] is None and "single-class" in sc["reason"],
         "single-class fold -> null AUC with reason")
    nanr = cluster_bootstrap_auc(np.r_[1, 1, 0, 0], np.r_[np.nan, .8, .2, .1],
                                 np.array(["A", "B", "C", "D"], dtype=object), n_boot=200)
    c.ok(nanr["n_dropped_nonfinite"] == 1 and nanr["auc"] is not None,
         "NaN probability is dropped and counted, not propagated")
    allnan = cluster_bootstrap_auc(np.r_[1, 0], np.r_[np.nan, np.nan],
                                   np.array(["A", "B"], dtype=object), n_boot=10)
    c.ok(allnan["auc"] is None and "no usable rows" in allnan["reason"],
         "all-NaN probabilities -> null with reason")

    # ------------------------------------------------------------ aggregation
    print("\n[8] patient-level aggregation")
    yy = np.array([1, 1, 0, 0, 0, 0])
    ss = np.array([0.9, 0.1, 0.2, 0.2, 0.8, 0.0])
    pp = np.array(["A", "A", "B", "B", "C", "C"], dtype=object)
    am = aggregate_by_cluster(yy, ss, pp, "mean")
    ax = aggregate_by_cluster(yy, ss, pp, "max")
    c.ok(np.allclose(am["scores"], [0.5, 0.2, 0.4]), "mean aggregation is right",
         f"{am['scores'].tolist()}")
    c.ok(np.allclose(ax["scores"], [0.9, 0.2, 0.8]), "max aggregation is right",
         f"{ax['scores'].tolist()}")
    c.ok(am["labels"].tolist() == [1, 0, 0], "patient label is max(slice labels)")
    mixed = aggregate_by_cluster(np.array([1, 0]), np.array([.9, .1]),
                                 np.array(["A", "A"], dtype=object), "mean")
    c.ok(mixed["n_mixed_label_clusters"] == 1, "mixed-label patients are counted")
    # A positive patient with two lukewarm slices vs a negative patient with one
    # very confident false positive: mean ranks them right, max ranks them wrong.
    dy = np.array([1, 1, 0, 0])
    ds = np.array([0.50, 0.50, 0.90, 0.00])
    dp = np.array(["A", "A", "B", "B"], dtype=object)
    dm = aggregate_by_cluster(dy, ds, dp, "mean")
    dx = aggregate_by_cluster(dy, ds, dp, "max")
    auc_m = auc_midrank(dm["labels"], dm["scores"])
    auc_x = auc_midrank(dx["labels"], dx["scores"])
    c.ok(auc_m == 1.0 and auc_x == 0.0,
         "mean and max aggregation can disagree, so both are always reported",
         f"mean AUC {auc_m:.3f} vs max AUC {auc_x:.3f}")

    # ------------------------------------------------------- operating points
    print("\n[9] Youden threshold on validation, applied to test")
    vy = np.array([0, 0, 0, 1, 1, 1])
    vs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
    th = youden_threshold(vy, vs)
    c.ok(0.3 < th["threshold"] < 0.7 and th["youden_j"] == 1.0,
         "separable validation set -> threshold in the gap, J = 1",
         f"thr={th['threshold']:.3f}")
    ty = np.array([0, 0, 1, 1])
    ts = np.array([0.05, 0.65, 0.45, 0.95])
    m = metrics_at_threshold(ty, ts, th["threshold"])
    c.ok(m["tp"] == 1 and m["fn"] == 1 and m["fp"] == 1 and m["tn"] == 1,
         "test metrics use the validation threshold verbatim (no re-tuning)",
         f"tp={m['tp']} fp={m['fp']} tn={m['tn']} fn={m['fn']}")
    c.ok(m["sens"] == 0.5 and m["spec"] == 0.5, "sens/spec computed correctly")
    lo, hi = wilson_interval(3, 3)
    c.ok(abs(lo - 0.4385) < 1e-3 and hi == 1.0,
         "Wilson interval for 3/3 is [0.439, 1.000], not [1.000, 1.000]",
         f"[{lo:.4f}, {hi:.4f}]")
    c.ok(youden_threshold(np.zeros(5, int), rng.normal(size=5))["threshold"] is None,
         "single-class validation fold -> no threshold, with reason")
    c.ok(metrics_at_threshold(ty, ts, None)["reason"] is not None,
         "missing threshold -> null metrics with reason")
    # A threshold picked on test would flatter the model; show the gap.
    th_test = youden_threshold(ty, ts)
    m_cheat = metrics_at_threshold(ty, ts, th_test["threshold"])
    print(f"    threshold from validation -> test sens {m['sens']:.2f} spec {m['spec']:.2f}; "
          f"if it had been picked ON test -> sens {m_cheat['sens']:.2f} "
          f"spec {m_cheat['spec']:.2f} (this is the inflation we refuse)")

    # -------------------------------------------------------------- Holm
    print("\n[10] Holm adjustment")
    adj = holm_adjust([0.01, 0.04, 0.03])
    c.ok(np.allclose(adj, [0.03, 0.06, 0.06]),
         "Holm matches the hand-computed reference [0.03, 0.06, 0.06]", f"{adj}")
    adj2 = holm_adjust([0.01, None, float("nan"), 0.04])
    c.ok(adj2[1] is None and adj2[2] is None and abs(adj2[0] - 0.02) < 1e-12,
         "uncomputable comparisons pass through as null and shrink the family",
         f"{adj2}")
    c.ok(all(a <= 1.0 for a in holm_adjust([0.9, 0.95, 0.99])),
         "adjusted p-values are capped at 1.0")
    big = sorted(rng.uniform(size=30))
    c.ok(all(x <= y + 1e-15 for x, y in zip(holm_adjust(big), holm_adjust(big)[1:])),
         "adjusted p-values are monotone in the raw ones")

    # ------------------------------------------------- cluster bootstrap diff
    print("\n[11] clustered bootstrap difference test (NOT DeLong)")
    y, s1, pid = _sim_clustered(rng, n_patients=24, slices=12, mu=1.2)
    s2 = s1 + rng.normal(0, 0.05, size=len(s1))     # nearly identical model
    d_null = cluster_bootstrap_diff(y, s1, s2, pid, n_boot=400, seed=3)
    c.ok(d_null["p"] > 0.05 and d_null["ci_lo"] < 0 < d_null["ci_hi"],
         "near-identical models -> CI straddles 0, p not significant",
         f"p={d_null['p']:.3f} CI [{d_null['ci_lo']:+.3f}, {d_null['ci_hi']:+.3f}]")
    s3 = np.asarray(rng.normal(size=len(s1)))       # pure noise model
    d_alt = cluster_bootstrap_diff(y, s1, s3, pid, n_boot=400, seed=4)
    c.ok(d_alt["p"] < 0.05 and d_alt["ci_lo"] > 0,
         "signal vs noise -> CI excludes 0",
         f"p={d_alt['p']:.4f} CI [{d_alt['ci_lo']:+.3f}, {d_alt['ci_hi']:+.3f}]")
    c.ok(d_alt["p"] >= 1.0 / max(d_alt["n_boot_used"], 1),
         "bootstrap p is floored at its own resolution, never reported as 0")

    # ---------------------------------------------------------- end-to-end
    print("\n[12] end-to-end on synthetic stage-3 run JSONs (incl. a degenerate fold)")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "results"
        rd.mkdir()
        # One test fold per seed, scored by three "conditions" -- exactly the
        # stage-3 layout, where DeLong is only legal because the cases are shared.
        for sd in (42, 123):
            y, base_s, pid = _sim_clustered(rng, n_patients=10, slices=12, mu=1.0)
            vy, vbase, vpid = _sim_clustered(rng, n_patients=6, slices=12, mu=1.0)
            shared = rng.normal(size=len(y))
            vshared = rng.normal(size=len(vy))
            for cond, w in (("magnitude", 1.0), ("phase", 0.15), ("both", 1.1)):
                s = w * y + shared + 0.5 * rng.normal(size=len(y))
                vs = w * vy + vshared + 0.5 * rng.normal(size=len(vy))
                prob = 1 / (1 + np.exp(-s))
                vprob = 1 / (1 + np.exp(-vs))
                if cond == "phase" and sd == 123:
                    prob[0] = float("nan")          # a real hazard: NaN probability
                payload = {
                    "cohort": "synth", "condition": cond, "seed": sd, "region": "full",
                    "val": {"probs": vprob.tolist(), "labels": vy.tolist(),
                            "patient_ids": [f"V{p}" for p in vpid],
                            "cache_idx": list(range(1000, 1000 + len(vy)))},
                    "test": {"probs": prob.tolist(), "labels": y.tolist(),
                             "patient_ids": [f"P{p}" for p in pid],
                             "cache_idx": list(range(len(y)))},
                }
                (rd / f"synth_{cond}_seed{sd}.json").write_text(json.dumps(payload))
        # single-class test fold, 1 patient -- every guard at once
        (rd / "degen_magnitude_seed42.json").write_text(json.dumps({
            "cohort": "degen", "condition": "magnitude", "seed": 42, "region": "full",
            "val": {"probs": [0.1, 0.9], "labels": [0, 1], "patient_ids": ["A", "B"],
                    "cache_idx": [0, 1]},
            "test": {"probs": [0.2, 0.3, 0.4], "labels": [1, 1, 1],
                     "patient_ids": ["Z", "Z", "Z"], "cache_idx": [2, 3, 4]},
        }))
        st = run_statistics(rd, Path(td) / "nocache", Path(td) / "nocohorts",
                            n_boot=300, seed=7, alpha=0.05)
        c.ok(len(st["runs"]) == 7, "all 7 synthetic runs loaded", f"{len(st['runs'])}")
        degen = [r for r in st["runs"] if r["cohort"] == "degen"][0]
        c.ok(degen["slice_level"]["auc"] is None and degen["slice_level"]["reason"],
             "single-class test fold -> null AUC with reason",
             degen["slice_level"]["reason"][:60])
        c.ok(degen["operating_point"]["slice"]["sens"] is not None
             and degen["operating_point"]["slice"]["spec"] is None
             and "single-class" in degen["operating_point"]["slice"]["reason"],
             "single-class fold -> sensitivity defined, specificity explicitly null",
             f"sens={degen['operating_point']['slice']['sens']}")
        nanrun = [r for r in st["runs"] if r["tag"] == "synth_phase_seed123"][0]
        c.ok(any("non-finite" in w for w in nanrun["warnings"]),
             "NaN probability surfaced as a run warning")
        nan_cmps = [cc for cc in st["comparisons"]
                    if cc["seed"] == 123 and "phase" in (cc["model_a"], cc["model_b"])]
        c.ok(all(cc.get("n_dropped_nonfinite") == 1 for cc in nan_cmps),
             "a NaN in one model drops that case from BOTH sides of the paired test",
             f"{len(nan_cmps)} comparisons")
        c.ok(all(r["cluster_unit"] == "patient_id" for r in st["runs"])
             and any("falling back to patient_id" in w for w in st["warnings"]),
             "missing cohort CSVs -> patient_id fallback announced, not silent")
        n_cmp = len(st["comparisons"])
        c.ok(n_cmp == 18, "3 condition pairs x 2 seeds x 3 levels = 18 comparisons",
             f"{n_cmp}")
        c.ok(all(("p_holm" in cc) for cc in st["comparisons"]),
             "every comparison carries a Holm-adjusted p-value")
        evalu = [cc for cc in st["comparisons"] if cc["p_raw"] is not None]
        c.ok(all(cc["p_holm"] >= cc["p_raw"] - 1e-12 for cc in evalu),
             "Holm p >= raw p for every evaluable comparison")
        seedrow = [g for g in st["across_seeds"] if g["condition"] == "magnitude"
                   and g["cohort"] == "synth"][0]
        c.ok(seedrow["slice_auc"]["n"] == 2 and seedrow["slice_auc"]["sd"] is not None,
             "across-seed mean +/- SD computed for 2 seeds")
        onerow = [g for g in st["across_seeds"] if g["cohort"] == "degen"][0]
        c.ok(onerow["slice_auc"]["sd"] is None and onerow["slice_auc"]["mean"] is None
             and "no evaluable seeds" in (onerow["slice_auc"]["reason"] or ""),
             "group whose only seed was degenerate -> mean and SD null with reason")
        c.ok(_mean_sd([0.71])["sd"] is None
             and "one evaluable seed" in _mean_sd([0.71])["reason"],
             "single evaluable seed -> mean reported, SD null with reason")
        blob = json.dumps(st, default=_json_default)
        c.ok("NaN" not in blob and "Infinity" not in blob,
             "output JSON contains no bare NaN/Infinity tokens (strict-JSON parseable)")
        json.loads(blob)
        # mismatched test sets must refuse to compare
        p1 = json.loads((rd / "synth_magnitude_seed42.json").read_text())
        p2 = json.loads((rd / "synth_phase_seed42.json").read_text())
        p2["test"]["cache_idx"] = [i + 10000 for i in p2["test"]["cache_idx"]]
        bad = compare_runs(p1, p2, None, 100, 0, 0.05)
        c.ok(bad[0]["p_raw"] is None and "different test sets" in bad[0]["reason"],
             "two runs scored on different test sets -> refused, with reason")

    # -------------------------------------------------- cross-validation
    print("\n[13] cross-validated layout: out-of-fold pooling")
    c.ok(parse_fold_dir("prostate_t2_cv3") == ("prostate_t2", 3)
         and parse_fold_dir("confound_brain") == ("confound_brain", None)
         and parse_fold_dir("") == ("", None)
         and parse_fold_dir("breast_cv12") == ("breast", 12),
         "fold index is parsed from the DIRECTORY name, not the filename",
         f"{parse_fold_dir('prostate_t2_cv3')}")

    K, N_SUB, SLICES = 5, 10, 12
    folds_struct = {}
    for k in range(K):
        y_sub = np.array([1] * 5 + [0] * 5)
        subs = [f"S{k}_{i}" for i in range(N_SUB)]
        vsubs = [f"V{k}_{i}" for i in range(6)]
        folds_struct[k] = {
            "pid": np.repeat(subs, SLICES),
            "y": np.repeat(y_sub, SLICES),
            "u": np.repeat(rng.normal(0, 1.0, N_SUB), SLICES),
            "cidx": list(range(k * 10000, k * 10000 + N_SUB * SLICES)),
            "vpid": np.repeat(vsubs, SLICES),
            "vy": np.repeat(np.array([1] * 3 + [0] * 3), SLICES),
            "vu": np.repeat(rng.normal(0, 1.0, 6), SLICES),
            "vcidx": list(range(500000 + k * 10000, 500000 + k * 10000 + 6 * SLICES)),
        }

    def _cv_payload(cond, sd, k, w):
        f = folds_struct[k]
        s = w * f["y"] + f["u"] + rng.normal(0, 0.8, len(f["y"]))
        vs = w * f["vy"] + f["vu"] + rng.normal(0, 0.8, len(f["vy"]))
        prob = 1 / (1 + np.exp(-s))
        vprob = 1 / (1 + np.exp(-vs))
        return {
            "cohort": "cvcoh", "condition": cond, "seed": sd, "region": "full",
            "val": {"probs": vprob.tolist(), "labels": [int(v) for v in f["vy"]],
                    "patient_ids": [str(p) for p in f["vpid"]],
                    "cache_idx": list(f["vcidx"])},
            "test": {"probs": prob.tolist(), "labels": [int(v) for v in f["y"]],
                     "patient_ids": [str(p) for p in f["pid"]],
                     "cache_idx": list(f["cidx"]),
                     "auc": float(auc_midrank(f["y"].astype(int), prob))},
        }

    with tempfile.TemporaryDirectory() as td:
        rd = Path(td) / "results"
        rd.mkdir()
        for k in range(K):
            (rd / f"cvcoh_cv{k}").mkdir()
        # a single-split cohort in its own directory, exactly the confound layout
        (rd / "confound_flat").mkdir()
        for sd in (42, 123):
            for cond, w in (("magnitude", 1.0), ("phase", 0.35), ("both", 1.1)):
                for k in range(K):
                    (rd / f"cvcoh_cv{k}" / f"cvcoh_{cond}_seed{sd}.json").write_text(
                        json.dumps(_cv_payload(cond, sd, k, w)))
                flat = _cv_payload(cond, sd, 0, w)
                flat["cohort"] = "flatcoh"
                (rd / "confound_flat" / f"flatcoh_{cond}_seed{sd}.json").write_text(
                    json.dumps(flat))
        # a stage-5 control payload sitting inside the tree: must be skipped, or
        # the control that exists to falsify the headline joins the headline
        ctrl = _cv_payload("phase", 42, 0, 0.0)
        ctrl["cohort"] = "flatcoh"
        ctrl["control"] = "phase_scramble"
        (rd / "confound_flat" / "flatcoh_phase_seed42_scramble.json").write_text(
            json.dumps(ctrl))

        loaded = load_runs(rd)
        c.ok(len(loaded) == K * 6 + 6,
             "recursive discovery finds every fold subdirectory",
             f"{len(loaded)} runs (expected {K*6+6}; a non-recursive glob finds 0)")
        c.ok(not any(r.get("control") for r in loaded),
             "a stage-5 control payload inside the results tree is skipped, not pooled "
             "into the headline")
        cvruns = [r for r in loaded if r["_split_family"] == "cv"]
        c.ok(sorted({r["_fold"] for r in cvruns}) == list(range(K))
             and all(r["_fold"] is None for r in loaded if r["_split_family"] != "cv"),
             "fold provenance recorded on CV runs, absent on single-split runs")
        c.ok(len({r["_tag"] for r in cvruns}) == len(cvruns),
             "five folds share one FILENAME but get distinct tags",
             f"{len({r['_tag'] for r in cvruns})}/{len(cvruns)} distinct")

        cmap = None
        one = [r for r in loaded
               if r["_split_family"] == "cv" and r["condition"] == "phase" and r["seed"] == 42]
        pooled, pinfo = pool_folds(one, cmap, expected_folds=set(range(K)))
        c.ok(pooled is not None and pinfo["ok"], "five folds pool without complaint")
        c.ok(pinfo["n_slices"] == K * N_SUB * SLICES
             and pinfo["n_subjects"] == K * N_SUB,
             "pooled vector covers every subject of the cohort exactly once",
             f"{pinfo['n_slices']} slices / {pinfo['n_subjects']} subjects")
        want_probs = np.concatenate([
            np.asarray(json.loads((rd / f"cvcoh_cv{k}" / "cvcoh_phase_seed42.json")
                                  .read_text())["test"]["probs"]) for k in range(K)])
        c.ok(np.allclose(np.asarray(pooled["test"]["probs"]), want_probs),
             "pooled probabilities are exactly the fold test blocks concatenated in "
             "fold order")
        c.ok(sorted(pooled["test"]["cache_idx"]) == sorted(
                 sum((folds_struct[k]["cidx"] for k in range(K)), [])),
             "pooled cache_idx is the union of the folds', with no gaps or repeats")
        pooled_auc = auc_midrank(np.asarray(pooled["test"]["labels"]),
                                 np.asarray(pooled["test"]["probs"]))
        c.ok(all(abs(pinfo["per_fold_reported_auc"][k]
                     - auc_midrank(folds_struct[k]["y"].astype(int),
                                   want_probs[k * 120:(k + 1) * 120])) < 1e-12
                 for k in range(K)) and not pinfo["auc_recheck_failures"],
             "each fold's stage-3 AUC is re-derived from the pooled rows and matches",
             f"pooled AUC {pooled_auc:.4f}")

        # ---- the exactly-once assertion must FIRE on a duplicate ----------
        dup = [json.loads(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}))
               for r in sorted(one, key=lambda r: r["_fold"])]
        for i, r in enumerate(dup):
            r["_fold"] = i
            r["_path"] = f"fold{i}.json"
        # one subject of fold 0 also appears in fold 1 (cache_idx still unique,
        # so ONLY the subject check can catch this)
        dup[1]["test"]["patient_ids"] = (
            [dup[0]["test"]["patient_ids"][0]] * SLICES
            + dup[1]["test"]["patient_ids"][SLICES:])
        bad_pool, bad_info = pool_folds(dup, None, expected_folds=set(range(K)))
        c.ok(bad_pool is None and "more than one test fold" in (bad_info["reason"] or ""),
             "a subject appearing in two folds REFUSES the pool, it is not averaged away",
             (bad_info["reason"] or "")[:64])
        c.ok(bad_info["duplicate_subjects"]
             and bad_info["duplicate_subjects"][0]["folds"] == [0, 1],
             "the refusal names the straddling subject and the folds it straddles",
             str(bad_info["duplicate_subjects"][:1]))

        dup2 = [json.loads(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}))
                for r in sorted(one, key=lambda r: r["_fold"])]
        for i, r in enumerate(dup2):
            r["_fold"] = i
            r["_path"] = f"fold{i}.json"
        dup2[2]["test"]["cache_idx"] = list(dup2[0]["test"]["cache_idx"])
        p2_, i2_ = pool_folds(dup2, None)
        c.ok(p2_ is None and "more than one fold" in (i2_["reason"] or "")
             and i2_["duplicate_cache_idx"],
             "the same cache_idx in two folds REFUSES the pool too",
             (i2_["reason"] or "")[:56])

        same_fold = [dict(one[0], _fold=1, _path="a.json"), dict(one[1], _fold=1, _path="b.json")]
        p3_, i3_ = pool_folds(same_fold, None)
        c.ok(p3_ is None and "appears twice" in (i3_["reason"] or ""),
             "two runs claiming the same fold index -> refused, not silently stacked")

        # ---- missing fold: pool anyway, but say what it covers ------------
        short, sinfo = pool_folds([r for r in one if r["_fold"] != 4], None,
                                  expected_folds=set(range(K)))
        c.ok(short is not None and sinfo["missing_folds"] == [4]
             and any("missing" in w for w in sinfo["warnings"]),
             "a missing fold pools the rest and states the reduced coverage",
             f"covers folds {sinfo['folds']}")
        lone, linfo = pool_folds([r for r in one if r["_fold"] == 0], None,
                                 expected_folds=set(range(K)))
        c.ok(lone is not None and linfo["single_fold"]
             and any("SINGLE FOLD under another name" in w for w in linfo["warnings"]),
             "one fold on disk is labelled a single fold, not sold as an out-of-fold "
             "estimate over the cohort")
        cmp_short = compare_runs(pooled, short, None, 100, 0, 0.05)
        c.ok(cmp_short[0]["p_raw"] is None
             and "pooled over different folds" in (cmp_short[0]["reason"] or ""),
             "two conditions pooled over DIFFERENT folds -> paired test refused")

        # ---- single-class fold and NaN probabilities ----------------------
        degen = [json.loads(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}))
                 for r in sorted(one, key=lambda r: r["_fold"])]
        for i, r in enumerate(degen):
            r["_fold"] = i
            r["_path"] = f"fold{i}.json"
        degen[3]["test"]["labels"] = [1] * len(degen[3]["test"]["labels"])
        degen[3]["test"]["auc"] = None
        degen[2]["test"]["probs"][0] = float("nan")
        dpool, dinfo = pool_folds(degen, None, expected_folds=set(range(K)))
        c.ok(dpool is not None and dinfo["single_class_folds"] == [3]
             and any("single-class" in w for w in dinfo["warnings"]),
             "a single-class fold still contributes its cases, and is flagged")
        dres = analyse_run(dpool, None, 200, 0, 0.05)
        c.ok(dres["slice_level"]["auc"] is not None
             and dres["slice_level"]["n_dropped_nonfinite"] == 1,
             "the pooled estimate survives a NaN probability, dropped and counted")

        # ---- end to end -----------------------------------------------------
        st = run_statistics(rd, Path(td) / "nocache", Path(td) / "nocohorts",
                            n_boot=300, seed=7, alpha=0.05)
        cvrows = [r for r in st["runs"] if r["cohort"] == "cvcoh"]
        flatrows = [r for r in st["runs"] if r["cohort"] == "flatcoh"]
        c.ok(len(cvrows) == 6 and all(r["pooled"] for r in cvrows),
             "a 5-fold x 3-condition x 2-seed sweep yields 6 estimates, not 30",
             f"{len(cvrows)} pooled estimates")
        c.ok(len(flatrows) == 6 and not any(r["pooled"] for r in flatrows)
             and all(r["folds"] is None for r in flatrows),
             "the single-directory layout is untouched: 6 runs, none pooled",
             f"{len(flatrows)}")
        c.ok(all(r["n_folds"] == K and r["folds"] == list(range(K)) for r in cvrows),
             "every pooled estimate records which folds it pooled")
        c.ok(len(st["cv"]["per_fold"]) == K * 6,
             "per-fold diagnostics are computed for all 30 fold files",
             f"{len(st['cv']['per_fold'])}")
        c.ok(not any(r.get("fold") is not None for r in st["runs"])
             and not any(cc.get("fold") is not None for cc in st["comparisons"]),
             "no per-fold estimate leaks into `runs` or into a comparison")

        # ---- power: pooled vs per fold --------------------------------------
        pooled_row = [r for r in cvrows if r["condition"] == "phase" and r["seed"] == 42][0]
        fold_rows = [f for f in st["cv"]["per_fold"]
                     if f["condition"] == "phase" and f["seed"] == 42]
        pw = pooled_row["slice_level"]["ci_hi"] - pooled_row["slice_level"]["ci_lo"]
        fws = [f["slice_level"]["ci_hi"] - f["slice_level"]["ci_lo"]
               for f in fold_rows if f["slice_level"]["ci_lo"] is not None]
        print(f"    pooled 95% CI width {pw:.3f} over {pooled_row['slice_level']['n_clusters']} "
              f"subjects; per-fold widths {[round(w, 3) for w in fws]} "
              f"(mean {np.mean(fws):.3f}) over {N_SUB} subjects each")
        c.ok(len(fws) == K and pw < 0.6 * float(np.mean(fws)),
             "the pooled CI is markedly narrower than the per-fold CIs -- this is the "
             "power that reading five folds as five experiments throws away",
             f"{pw:.3f} vs mean {np.mean(fws):.3f} ({np.mean(fws)/pw:.2f}x)")
        c.ok(pooled_row["slice_level"]["n_clusters"] == K * N_SUB,
             "the pooled bootstrap resamples all 50 subjects, not 10",
             f"{pooled_row['slice_level']['n_clusters']}")
        disp = [d for d in st["cv"]["fold_dispersion"]
                if d["condition"] == "phase" and d["seed"] == 42
                and d["level"] == "slice_level"][0]
        c.ok(disp["n_evaluable_folds"] == K and disp["fold_auc_range"] is not None
             and disp["pooled_ci_width"] is not None,
             "fold-to-fold dispersion is reported next to the pooled CI width",
             f"range {disp['fold_auc_range']:.3f} vs pooled CI width "
             f"{disp['pooled_ci_width']:.3f}")

        # ---- family size ----------------------------------------------------
        cv_cmps = [cc for cc in st["comparisons"] if cc["cohort"] == "cvcoh"]
        c.ok(len(cv_cmps) == 3 * 2 * 3,
             "3 condition pairs x 2 seeds x 3 levels = 18 comparisons for the CV "
             "cohort -- NOT 90, which is what five independent folds would have cost",
             f"{len(cv_cmps)}")
        h = st["holm"]
        c.ok(h["fold_multiplicity"]["n_if_each_fold_were_independent"] == 90
             and h["fold_multiplicity"]["inflation_avoided"] == 72,
             "the fold multiplicity that pooling removes is reported, not assumed",
             f"{h['fold_multiplicity']}")
        c.ok(h["n_cohort_condition_pairs"] == 6
             and all(v == 6 for v in h["comparisons_per_cohort_pair"].values()),
             "each (cohort, condition-pair) contributes 6 comparisons (2 seeds x 3 "
             "levels) and no fold dimension",
             f"{h['comparisons_per_cohort_pair']}")
        c.ok(h["fold_dimension_in_family"] is False,
             "the family carries no fold dimension")
        c.ok(h["family_size"] == sum(1 for cc in st["comparisons"]
                                     if cc["p_raw"] is not None and np.isfinite(cc["p_raw"]))
             and h["family_size"] <= h["n_comparisons_emitted"],
             "Holm family size counts evaluable comparisons only",
             f"family_size={h['family_size']} of {h['n_comparisons_emitted']} emitted")
        c.ok(all(cc["pooled"] and cc.get("n_cases") in (None, K * N_SUB * SLICES, K * N_SUB)
                 for cc in cv_cmps),
             "every CV comparison is between two POOLED vectors covering the full cohort")

        # ---- operating point stays off test ---------------------------------
        op = pooled_row["operating_point"]
        c.ok("per fold" in op["chosen_on"] and len(op["per_fold"]) == K
             and len(op["slice"]["per_fold_thresholds"]) == K,
             "the pooled operating point uses one validation-fit threshold PER FOLD",
             op["chosen_on"])
        c.ok(op["slice"]["tp"] + op["slice"]["fn"] == pooled_row["slice_level"]["n_pos_slices"]
             and op["slice"]["n"] == K * N_SUB * SLICES,
             "per-fold confusion counts sum to the pooled totals",
             f"n={op['slice']['n']} pos={op['slice']['tp'] + op['slice']['fn']}")
        blob = json.dumps(st, default=_json_default)
        c.ok("NaN" not in blob and "Infinity" not in blob, "pooled output is strict JSON")
        json.loads(blob)

    print()
    print(_rule(100, "="))
    print(f"self-test: {c.passed} passed, {c.failed} failed")
    print(_rule(100, "="))
    return 0 if c.failed == 0 else 1


# ==========================================================================
# CLI
# ==========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description="PhaseDx stage 4: statistics")
    p.add_argument("--results-dir", default=str(_DEFAULT_RESULTS_DIR),
                   help="directory of stage-3 per-run JSONs")
    p.add_argument("--cache-dir", default=str(_DEFAULT_CACHE_DIR),
                   help="stage-2 cache dir (for the cache_idx -> subject_id join)")
    p.add_argument("--cohort-dir", default=str(_DEFAULT_COHORT_DIR),
                   help="stage-1 cohort CSV dir (for subject_id)")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: <results-dir>/statistics.json)")
    p.add_argument("--n-boot", type=int, default=2000,
                   help="bootstrap replicates (clustered on patient/subject)")
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    p.add_argument("--alpha", type=float, default=0.05, help="1-alpha CI level")
    p.add_argument("--cluster-unit", choices=("auto", "subject", "patient"), default="auto",
                   help="auto: subject_id when the join works, else patient_id with a warning; "
                        "subject: fail if the join is unavailable; patient: force patient_id")
    p.add_argument("--check-cluster-map", action="store_true",
                   help="report the cache_idx -> subject_id join per cohort and exit")
    p.add_argument("--self-test", action="store_true", help="run the unit tests and exit")
    p.add_argument("--quick", action="store_true", help="smaller simulations in --self-test")
    p.add_argument("--quiet", action="store_true", help="suppress the printed tables")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if args.self_test:
        return self_test(quick=args.quick)
    if args.check_cluster_map:
        return check_cluster_maps(Path(args.cache_dir), Path(args.cohort_dir))

    results_dir = Path(args.results_dir)
    stats = run_statistics(
        results_dir=results_dir,
        cache_dir=Path(args.cache_dir),
        cohort_dir=Path(args.cohort_dir),
        n_boot=args.n_boot, seed=args.seed, alpha=args.alpha,
        cluster_unit=args.cluster_unit,
    )
    out_path = Path(args.out) if args.out else results_dir / "statistics.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(stats, indent=2, default=_json_default))
    if not args.quiet:
        print_report(stats)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
