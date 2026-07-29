"""
trivialbaselines.stats
----------------------
The statistics the zero-image baselines need, and nothing else.

VENDORED VERBATIM from ``pipeline/s04_stats.py`` of the PhaseDx study
(https://github.com/sathvikloke/PhaseDx). The function bodies below are byte-identical
to the ones every number in the paper was computed with; they are copied rather than
imported so that this package installs with numpy + pandas alone and can be checked
by a reviewer without cloning the study. Regenerate with ``tools/sync_from_pipeline.py``.

numpy only. No scipy, no scikit-learn, no torch. That is deliberate: the claim this
tool exists to support is that a published slice-level benchmark can be audited with
no images and no GPU, and a dependency on a deep-learning stack would undercut it.

Three of these deserve a note, because they are where naive implementations go wrong:

``auc_midrank``      ties get the AVERAGE rank. A pixel-blind baseline emits the same
                     score for every slice in a position bin, so ties are the common
                     case here, not the exception. Ranking them arbitrarily inflates
                     or deflates the AUROC depending on the input order.

``cluster_bootstrap_auc``
                     resamples SUBJECTS, not slices. The slice-level bootstrap is the
                     reason published intervals are too narrow: in simulation
                     (20 subjects x 15 slices, 200 datasets) the naive slice interval
                     covered the true AUC 46.5% of the time at a nominal 95%, against
                     91.5% for this one, and it was 3.2x narrower.

``naive_slice_bootstrap_auc``
                     the wrong interval, kept ONLY so a report can print how much
                     narrower the wrong interval would have been. Never a headline.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "compute_midrank",
    "auc_midrank",
    "average_precision",
    "aggregate_by_cluster",
    "cluster_bootstrap_auc",
    "naive_slice_bootstrap_auc",
]


# ========================================================================
# Rank statistics
# ========================================================================


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


# ========================================================================
# Cluster-aware resampling
# ========================================================================


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
