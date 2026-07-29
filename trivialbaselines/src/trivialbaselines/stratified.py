"""
trivialbaselines.stratified
---------------------------
The position-stratified AUROC: the REMEDY metric that goes with the positional baseline.

A slice-level AUROC counts every positive/negative slice pair, including pairs drawn
from different parts of the stack. If positives sit nearer the middle of the organ --
and in a lesion-detection benchmark they usually do -- a large share of those pairs are
won by geometry rather than by pathology. Stratifying the Mann-Whitney statistic on
relative slice position lets ONLY same-position pairs contribute, so exactly that share
is removed and nothing else is.

Measured on Rempe et al. (2024)'s own published prostate DWI label file and split, all
three numbers from the same score vector:

    zero-image positional baseline, slice-level AUROC        0.851
    the same scores, patient-level AUROC                     0.424
    the same scores, position-stratified slice-level AUROC   0.539  (6 strata)

and on the PhaseDx reimplementation of their protocol, again one score vector:

    slice-level AUROC                                        0.574
    position-stratified slice-level AUROC                    0.467

This is not part of ``audit()``, which reads a label file and never sees your model's
predictions. Call it directly on your own test-set scores::

    from trivialbaselines import position_strata, stratified_auc
    rel = (slice_idx - slice_idx_min_in_volume) / (slice_idx_max - slice_idx_min)
    print(stratified_auc(labels, scores, position_strata(rel, n_strata=10)))

VENDORED VERBATIM from ``pipeline/s12_rempe.py`` of the PhaseDx study, with the
statistics import repointed at the vendored copy in ``trivialbaselines.stats``.
Regenerate with ``tools/sync_from_pipeline.py``.
"""

from __future__ import annotations

import numpy as np

from . import stats as s04_stats

__all__ = ["stratified_auc", "position_strata"]


def stratified_auc(labels, scores, strata) -> dict:
    """
    AUROC computed WITHIN strata and pooled, i.e. a stratified Mann-Whitney.

    Only positive/negative pairs that sit in the same stratum contribute, so
    stratifying on slice position removes exactly the part of a slice-level
    AUROC that comes from "positives sit nearer the middle of the stack". A
    predictor whose whole signal is positional falls to 0.5 here; a predictor
    that actually reads the patient does not.

    Pooling weights each stratum by its number of comparable pairs, which is
    the weighting that makes the pooled statistic equal the overall AUROC when
    there is only one stratum.
    """
    labels = np.asarray(labels, dtype=int)
    scores = np.asarray(scores, dtype=float)
    strata = np.asarray(strata)
    num = den = 0.0
    used = 0
    for s in np.unique(strata):
        m = strata == s
        y, sc = labels[m], scores[m]
        npos, nneg = int(y.sum()), int((1 - y).sum())
        if npos == 0 or nneg == 0:
            continue
        a = s04_stats.auc_midrank(y, sc)
        if not np.isfinite(a):
            continue
        w = npos * nneg
        num += a * w
        den += w
        used += 1
    return {
        "auc": float(num / den) if den > 0 else float("nan"),
        "n_strata_used": used,
        "n_pairs": int(den),
    }


def position_strata(relpos, n_strata: int = 10) -> np.ndarray:
    """Equal-width bins of relative slice position, as stratification labels."""
    edges = np.linspace(0.0, 1.0, n_strata + 1)
    return np.clip(np.digitize(np.asarray(relpos, dtype=float), edges) - 1, 0, n_strata - 1)
