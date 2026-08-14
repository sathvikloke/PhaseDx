"""Weighted / stratified midrank AUROC machinery.

Written from scratch for the frozen-holdout re-analysis. Shares no code with
pipeline/s14_trivialbaselines.py or pipeline/audit_prep/rsna_ich_unit_collapse.py;
point estimates are cross-checked against sklearn.roc_auc_score before any resampling.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

TIE_SNAP = 12


def snap(x):
    return np.round(np.asarray(x, dtype=float), TIE_SNAP)


# --------------------------------------------------------------------------
# count-matrix route: fast when the score takes few distinct values
# --------------------------------------------------------------------------

def auc_from_counts(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC from per-distinct-value positive/negative counts in ascending order."""
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return float("nan")
    below = np.concatenate(([0.0], np.cumsum(neg)[:-1]))
    return float((pos * (below + neg / 2.0)).sum() / (P * N))


def cluster_count_matrix(y, s, cluster_code, k):
    """(k clusters x v distinct values) positive and negative count matrices."""
    vals, inv = np.unique(snap(s), return_inverse=True)
    v = len(vals)
    flat = cluster_code.astype(np.int64) * v + inv
    Cp = np.bincount(flat, weights=(y == 1).astype(float), minlength=k * v).reshape(k, v)
    Cn = np.bincount(flat, weights=(y == 0).astype(float), minlength=k * v).reshape(k, v)
    return Cp, Cn


def boot_ci_counts(Cp, Cn, rng, n_boot=2000, alpha=0.05):
    k = Cp.shape[0]
    out = np.empty(n_boot)
    out[:] = np.nan
    for b in range(n_boot):
        m = np.bincount(rng.integers(0, k, size=k), minlength=k).astype(float)
        out[b] = auc_from_counts(m @ Cp, m @ Cn)
    good = out[np.isfinite(out)]
    if len(good) < 2:
        return float("nan"), float("nan"), 0
    return (float(np.percentile(good, 100 * alpha / 2)),
            float(np.percentile(good, 100 * (1 - alpha / 2))), int(len(good)))


# --------------------------------------------------------------------------
# stratified route: strata-wise concordance, ties counted as half
# --------------------------------------------------------------------------

class StratifiedAUC:
    """Pooled within-stratum concordance with ties at half weight.

    One unit = one row (here, one patient). Set stratum to a constant for the
    ordinary unstratified AUROC; the two share the identical code path so a
    difference between them cannot be an implementation difference.
    """

    def __init__(self, y, score, stratum):
        y = np.asarray(y, dtype=float)
        score = snap(score)
        stratum = np.asarray(stratum)
        order = np.lexsort((score, stratum))
        self.order = order
        ys, ss, st = y[order], score[order], stratum[order]
        # group id per (stratum, score) run; strata are contiguous after the lexsort
        key_new = np.ones(len(ys), dtype=bool)
        if len(ys) > 1:
            key_new[1:] = (st[1:] != st[:-1]) | (ss[1:] != ss[:-1])
        self.g = np.cumsum(key_new) - 1
        self.G = int(self.g[-1]) + 1 if len(ys) else 0
        st_new = np.ones(len(ys), dtype=bool)
        if len(ys) > 1:
            st_new[1:] = st[1:] != st[:-1]
        s_code = np.cumsum(st_new) - 1
        self.S = int(s_code[-1]) + 1 if len(ys) else 0
        # stratum index of each group
        self.sgrp = np.zeros(self.G, dtype=np.int64)
        self.sgrp[self.g] = s_code
        self.ys = ys
        self.n = len(ys)

    def value(self, w=None):
        if w is None:
            w = np.ones(self.n)
        else:
            w = np.asarray(w, dtype=float)[self.order]
        wpos = w * self.ys
        wneg = w * (1.0 - self.ys)
        gpos = np.bincount(self.g, weights=wpos, minlength=self.G)
        gneg = np.bincount(self.g, weights=wneg, minlength=self.G)
        cum = np.cumsum(gneg)
        below_global = cum - gneg
        Ns = np.bincount(self.sgrp, weights=gneg, minlength=self.S)
        Ps = np.bincount(self.sgrp, weights=gpos, minlength=self.S)
        base = np.concatenate(([0.0], np.cumsum(Ns)[:-1]))
        below = below_global - base[self.sgrp]
        denom = float((Ps * Ns).sum())
        if denom <= 0:
            return float("nan"), 0.0
        conc = float((gpos * (below + gneg / 2.0)).sum())
        return conc / denom, denom

    def pairs(self):
        return self.value()[1]

    def boot_ci(self, rng, n_boot=2000, alpha=0.05):
        vals = np.empty(n_boot)
        vals[:] = np.nan
        for b in range(n_boot):
            m = np.bincount(rng.integers(0, self.n, size=self.n),
                            minlength=self.n).astype(float)
            vals[b] = self.value(m)[0]
        good = vals[np.isfinite(vals)]
        if len(good) < 2:
            return float("nan"), float("nan"), 0
        return (float(np.percentile(good, 100 * alpha / 2)),
                float(np.percentile(good, 100 * (1 - alpha / 2))), int(len(good)))


# --------------------------------------------------------------------------
# slice-level clustered bootstrap wrapper
# --------------------------------------------------------------------------

def slice_auc_ci(y, s, cluster_code, k, rng, n_boot=2000):
    Cp, Cn = cluster_count_matrix(np.asarray(y, dtype=int), s, cluster_code, k)
    point = auc_from_counts(Cp.sum(0), Cn.sum(0))
    lo, hi, used = boot_ci_counts(Cp, Cn, rng, n_boot=n_boot)
    return point, lo, hi, used


# --------------------------------------------------------------------------
# the locked primary baseline: 20-bin relative-position histogram
# --------------------------------------------------------------------------

def positional_fit_score(y_train, bins_train, bins_test, n_bins=20):
    num = np.bincount(bins_train, weights=y_train.astype(float), minlength=n_bins)
    den = np.bincount(bins_train, minlength=n_bins).astype(float)
    prior = float(y_train.mean())
    rate = np.where(den > 0, num / np.maximum(den, 1.0), prior)
    return rate[bins_test], rate, prior


def relpos_bins(slice_idx, vol_code, n_bins=20):
    slc = np.asarray(slice_idx, dtype=float)
    lo = pd.Series(slc).groupby(vol_code).transform("min").to_numpy()
    hi = pd.Series(slc).groupby(vol_code).transform("max").to_numpy()
    span = np.where(hi > lo, hi - lo, 1.0)
    rel = (slc - lo) / span
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    edges[-1] += 1e-9
    return rel, np.clip(np.digitize(rel, edges) - 1, 0, n_bins - 1)
