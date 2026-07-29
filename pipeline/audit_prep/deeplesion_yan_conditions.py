"""Reconstruct Yan et al.'s (CVPR 2018, arXiv:1711.10535) evaluation conditions.

Their Table 1 is NOT computed on DeepLesion's released Train_Val_Test field. Verbatim:
"Among the labeled samples, we randomly select 25% as training seeds to predict
pseudo-labels, 25% as the validation set, and the other 50% as the test set. There is
no patient-level overlap between all subsets." Their reported test set is 4,927
samples, which coincidentally has almost exactly the size of the official test split
(4,927) but is a different partition of the 9,816 type-labelled rows.

So to compare a zero-image baseline with their 90.5% and with their own image-derived
"Location feature" baseline of 59.7%, their partition has to be rebuilt: a random
patient-disjoint 25/25/50 split of the labelled rows, fitting on the 25% seed set only.
Repeated over many random draws, because the partition is random and a single draw
would make the comparison hostage to a seed.
"""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1]
N_BINS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_REP = int(sys.argv[3]) if len(sys.argv) > 3 else 200

d = pd.read_csv(src)
classes = np.sort(d.lesion_type.unique())
pats = d.patient_id.unique()
edges = np.linspace(0.0, 1.0, N_BINS + 1)
edges[-1] += 1e-9
d["_bin"] = np.clip(np.digitize(d.norm_z, edges) - 1, 0, N_BINS - 1)
y = np.searchsorted(classes, d.lesion_type.to_numpy())

accs, maj, ntest, nseed = [], [], [], []
for rep in range(N_REP):
    rng = np.random.default_rng(rep)
    perm = rng.permutation(pats)
    n = len(perm)
    seed_p = set(perm[:int(0.25 * n)])
    test_p = set(perm[int(0.50 * n):])
    is_seed = d.patient_id.isin(seed_p).to_numpy()
    is_test = d.patient_id.isin(test_p).to_numpy()

    counts = np.zeros((N_BINS, len(classes)))
    np.add.at(counts, (d._bin.to_numpy()[is_seed], y[is_seed]), 1.0)
    prior = np.bincount(y[is_seed], minlength=len(classes)).astype(float)
    prior /= prior.sum()
    probs = counts + 1e-9 * prior
    probs /= probs.sum(axis=1, keepdims=True)
    pred = np.argmax(probs[d._bin.to_numpy()[is_test]], axis=1)
    accs.append(float((pred == y[is_test]).mean()))
    maj.append(float((y[is_test] == int(np.argmax(prior))).mean()))
    ntest.append(int(is_test.sum()))
    nseed.append(int(is_seed.sum()))

accs = np.array(accs)
print(f"replicates: {N_REP} random patient-disjoint 25/25/50 partitions of the "
      f"{len(d)} type-labelled rows")
print(f"seed (training) rows  mean {np.mean(nseed):.0f}   "
      f"test rows mean {np.mean(ntest):.0f}  (Yan et al. report 4,927 test samples)")
print(f"zero-image {N_BINS}-bin P(type | published normalised z):  "
      f"accuracy {accs.mean():.4f}  sd {accs.std(ddof=1):.4f}  "
      f"[{np.percentile(accs, 2.5):.4f}, {np.percentile(accs, 97.5):.4f}] over partitions")
print(f"majority class under the same partitions:              "
      f"accuracy {np.mean(maj):.4f}")
print()
print("published on this partition scheme (Yan et al. Table 1):")
print("  Triplet with type + location + size (their best)      0.905 +/- 0.002")
print("  Baseline: Multi-scale ImageNet feature                0.862")
print("  Baseline: Location feature (image-derived x, y, SSBR z) 0.597")
