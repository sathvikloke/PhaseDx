"""Shared synthetic-experiment code (frozen at LOCK25). AUROC: rank-based, ties 1/2, NaN when a class is empty (never 0.5)."""
import numpy as np
from scipy.stats import rankdata, norm
def auroc(y, s):
    y = np.asarray(y).astype(bool); s = np.asarray(s, float)
    n1 = int(y.sum()); n0 = int((~y).sum())
    if n1 == 0 or n0 == 0: return float('nan'), n1, n0
    r = rankdata(s)
    return float((r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)), n1, n0
def auroc_within(y, s, strata):
    """pooled over strata: only pairs with equal stratum; returns (auc, n_pairs)."""
    num = 0.0; den = 0
    for g in np.unique(strata):
        m = strata == g; a, n1, n0 = auroc(y[m], s[m])
        if np.isfinite(a): num += a * n1 * n0; den += n1 * n0
    return (num / den if den else float('nan')), den
def sigmoid(x): return 1.0 / (1.0 + np.exp(-x))
