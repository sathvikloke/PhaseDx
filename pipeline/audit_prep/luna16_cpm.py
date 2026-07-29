"""LUNA16 on its OWN metric: CPM for the zero-image positional baseline.

AUROC is not the LUNA16 scale. The false-positive-reduction track is scored by the
competition performance metric (CPM): the mean sensitivity at 1/8, 1/4, 1/2, 1, 2, 4 and
8 false positives per scan. Comparing a positional AUROC against a published CPM would
be exactly the kind of incomparable comparison this audit is supposed to refuse, so the
baseline is scored on the published scale instead.

Scores come from the same 20-bin P(nodule | relative z within scan) estimator used
everywhere else, fitted out-of-fold on a scan-disjoint 5-fold split.
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path("/Users/sathvikloke/Downloads/PhaseDx/pipeline")))
import s14_trivialbaselines as tb  # noqa: E402

src = sys.argv[1]
N_BINS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
K = 5
FP_POINTS = (0.125, 0.25, 0.5, 1, 2, 4, 8)

d = pd.read_csv(src)
d = d.rename(columns={"slice": "z"})
z = d.z.to_numpy(float)
g = d.series_id
lo = g.map(d.groupby("series_id").z.min())
hi = g.map(d.groupby("series_id").z.max())
span = (hi - lo).replace(0, 1)
d["relpos"] = (z - lo) / span
d["label"] = d.label.astype(int)

scans = np.array(sorted(d.series_id.unique()))
rng = np.random.default_rng(0)
fold = {s: i % K for i, s in enumerate(rng.permutation(scans))}
d["_fold"] = d.series_id.map(fold)

scores = np.empty(len(d))
for k in range(K):
    tr = d[d._fold != k]
    te = d[d._fold == k]
    scores[te.index.to_numpy()] = tb.positional_scores(
        tr, te, n_bins=N_BINS, pos_col="relpos", label_col="label")

n_scans = len(scans)
y = d.label.to_numpy()
order = np.argsort(-scores, kind="mergesort")
ys = y[order]
tp = np.cumsum(ys)
fp = np.cumsum(1 - ys)
n_pos = int(y.sum())
sens = []
for f in FP_POINTS:
    budget = f * n_scans
    i = np.searchsorted(fp, budget, side="right") - 1
    sens.append(float(tp[i] / n_pos) if i >= 0 else 0.0)
cpm = float(np.mean(sens))

print(f"scans {n_scans}   candidates {len(d)}   true nodules {n_pos}")
print("zero-image positional baseline, LUNA16 FP-reduction scale:")
for f, s in zip(FP_POINTS, sens):
    print(f"   sensitivity at {f:>5} FP/scan : {s:.4f}")
print(f"   CPM (mean of the seven)        : {cpm:.4f}")
print()
print("chance reference: a random score has sensitivity == FP fraction, so its CPM is")
fpr_ref = [min(1.0, f * n_scans / (len(d) - n_pos)) for f in FP_POINTS]
print(f"   {np.mean(fpr_ref):.4f}  (mean of {['%.4f' % v for v in fpr_ref]})")
