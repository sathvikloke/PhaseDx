"""Lane B (Batch 03) estimators: exact AUROC with ties, multiplicity-weighted AUROC over many bootstrap
draws (vectorised), patient-cluster draw generation, interval summaries. No NaN ever leaves this module
in JSON-bound objects: undefined values are None with a reason string."""
from __future__ import annotations
from fractions import Fraction
import numpy as np
from scipy import sparse
from scipy.stats import rankdata
UNSTABLE_FRAC = 0.05

def dense_rank(values):
    """Exact dense ranks (0..G-1) for arbitrary orderable values (Fractions, ints, floats)."""
    u = sorted(set(values))
    pos = {v: i for i, v in enumerate(u)}
    return np.array([pos[v] for v in values], dtype=np.int64)

def auroc(y, s):
    """Mann-Whitney AUROC, ties = 1/2. Returns dict(auc|None, reason, n_pos, n_neg, U)."""
    y = np.asarray(y).astype(bool)
    s = np.asarray(s)
    npos, nneg = (int(y.sum()), int((~y).sum()))
    if npos == 0 or nneg == 0:
        return {'auc': None, 'reason': 'no positive units' if npos == 0 else 'no negative units', 'n_pos': npos, 'n_neg': nneg}
    r = rankdata(s, method='average')
    U = float(r[y].sum()) - npos * (npos + 1) / 2.0
    return {'auc': U / (npos * nneg), 'reason': None, 'n_pos': npos, 'n_neg': nneg, 'U': U}

def auroc_exact_fraction(y, s):
    """Exact rational AUROC via pair enumeration (for verification; O(n_pos*n_neg))."""
    y = np.asarray(y).astype(bool)
    pos = [v for v, t in zip(s, y) if t]
    neg = [v for v, t in zip(s, y) if not t]
    if not pos or not neg:
        return None
    num = Fraction(0)
    for p in pos:
        for n in neg:
            num += 1 if p > n else Fraction(1, 2) if p == n else 0
    return num / (len(pos) * len(neg))

def weighted_auroc_draws(y, s, W, chunk=250):
    """W: (n_draws, n_units) non-negative weights (cluster multiplicities mapped to units).
    Returns (auc array with NaN where undefined, num, den). Pair weight = w_i * w_j; ties = 1/2."""
    y = np.asarray(y).astype(bool)
    _, g = np.unique(np.asarray(s), return_inverse=True)
    G = int(g.max()) + 1 if len(g) else 0
    n = len(y)
    oh = sparse.csr_matrix((np.ones(n), (np.arange(n), g)), shape=(n, G))
    ohp, ohn = (oh[np.flatnonzero(y)], oh[np.flatnonzero(~y)])
    out_num = np.empty(W.shape[0])
    out_den = np.empty(W.shape[0])
    for a in range(0, W.shape[0], chunk):
        Wc = np.asarray(W[a:a + chunk], dtype=np.float64)
        P = np.asarray((ohp.T @ Wc[:, y].T).T)
        N = np.asarray((ohn.T @ Wc[:, ~y].T).T)
        below = np.cumsum(N, axis=1) - N
        out_num[a:a + chunk] = (P * (below + 0.5 * N)).sum(1)
        out_den[a:a + chunk] = P.sum(1) * N.sum(1)
    with np.errstate(invalid='ignore', divide='ignore'):
        auc = np.where(out_den > 0, out_num / np.where(out_den > 0, out_den, 1), np.nan)
    return (auc, out_num, out_den)

def cluster_draws(rng, n_clusters, n_draws):
    """Frozen procedure: draw b in order, idx = rng.integers(0, n, size=n); m = bincount(idx, minlength=n)."""
    M = np.zeros((n_draws, n_clusters), dtype=np.int64)
    for b in range(n_draws):
        idx = rng.integers(0, n_clusters, size=n_clusters)
        M[b] = np.bincount(idx, minlength=n_clusters)
    return M

def summarise(est, reps, label, extra=None, withhold_frac=UNSTABLE_FRAC):
    v = np.asarray(reps, dtype=float)
    ok = np.isfinite(v)
    nv, ni = (int(ok.sum()), int((~ok).sum()))
    unstable = ni > withhold_frac * len(v)
    lo = hi = None
    if nv:
        lo, hi = (float(x) for x in np.quantile(v[ok], [0.025, 0.975], method='linear'))
    row = {'quantity': label, 'estimate': est, 'n_draws': int(len(v)), 'n_valid': nv, 'n_undefined': ni, 'undefined_reason': 'draw(s) with no positive or no negative weighted unit in an arm' if ni else None, 'withheld_gt5pct_undefined': bool(unstable), 'ci95_low': None if unstable or lo is None else lo, 'ci95_high': None if unstable or hi is None else hi, 'ci95_low_valid_draws_diagnostic': lo, 'ci95_high_valid_draws_diagnostic': hi, 'replicate_mean': float(v[ok].mean()) if nv else None, 'replicate_sd': float(v[ok].std(ddof=1)) if nv > 1 else None, 'interval_method': 'percentile 95% numpy.quantile(linear) over valid draws; paired patient-cluster bootstrap'}
    if extra:
        row.update(extra)
    return row

def select_with_ties(values, order, tol=1e-12):
    """values: dict name->float|None; returns earliest name in `order` within tol of the max (None if all undefined)."""
    defined = [(k, values[k]) for k in order if values.get(k) is not None]
    if not defined:
        return None
    best = max((v for _, v in defined))
    for k, v in defined:
        if v >= best - tol:
            return k
    return None
