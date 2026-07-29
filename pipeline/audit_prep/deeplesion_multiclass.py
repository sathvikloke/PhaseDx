"""The 8-class DeepLesion lesion-type task, predicted from published z alone.

Yan et al. (CVPR 2018, arXiv:1711.10535) report an 8-class lesion-type classification
accuracy on DeepLesion's official test split. This script computes the accuracy of a
predictor that sees nothing but the published `Normalized_lesion_location` z, fitted on
the official validation split and applied to the official test split, which are
patient-disjoint.

Two zero-image predictors, plus the two chance anchors:
  * 20-bin argmax_c P(type = c | z), fitted on val
  * the same, plus the published acquisition metadata, via a shallow CART
  * majority class of the training (val) split
  * the class prior sampled at random, averaged analytically
Uncertainty is a subject-clustered bootstrap over test patients, matching the clustering
used everywhere else in this study.
"""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1]
N_BINS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_BOOT = int(sys.argv[3]) if len(sys.argv) > 3 else 2000
rng = np.random.default_rng(0)

d = pd.read_csv(src)
tr = d[d.split == "training"].reset_index(drop=True)
te = d[d.split == "test"].reset_index(drop=True)
assert not (set(tr.patient_id) & set(te.patient_id)), "val and test share a patient"
classes = np.sort(d.lesion_type.unique())

edges = np.linspace(0.0, 1.0, N_BINS + 1)
edges[-1] += 1e-9
tr_bin = np.clip(np.digitize(tr.norm_z, edges) - 1, 0, N_BINS - 1)
te_bin = np.clip(np.digitize(te.norm_z, edges) - 1, 0, N_BINS - 1)

counts = np.zeros((N_BINS, len(classes)))
for b, c in zip(tr_bin, tr.lesion_type):
    counts[b, np.searchsorted(classes, c)] += 1
prior = np.bincount(np.searchsorted(classes, tr.lesion_type.to_numpy()),
                    minlength=len(classes)).astype(float)
prior /= prior.sum()
probs = counts + 1e-9 * prior          # empty bins fall back to the training prior
probs /= probs.sum(axis=1, keepdims=True)
pred_pos = classes[np.argmax(probs[te_bin], axis=1)]

majority = classes[int(np.argmax(prior))]
acc_pos = float((pred_pos == te.lesion_type.to_numpy()).mean())
acc_maj = float((te.lesion_type.to_numpy() == majority).mean())
acc_prior_random = float(
    (prior * np.bincount(np.searchsorted(classes, te.lesion_type.to_numpy()),
                         minlength=len(classes)) / len(te)).sum())

# subject-clustered bootstrap over test patients
pats = te.patient_id.to_numpy()
uniq = np.unique(pats)
idx_by_pat = {p: np.flatnonzero(pats == p) for p in uniq}
correct = (pred_pos == te.lesion_type.to_numpy())
boot = np.empty(N_BOOT)
for i in range(N_BOOT):
    draw = rng.choice(uniq, size=len(uniq), replace=True)
    sel = np.concatenate([idx_by_pat[p] for p in draw])
    boot[i] = correct[sel].mean()
lo, hi = np.percentile(boot, [2.5, 97.5])

print(f"n train (official val split) = {len(tr)}   patients {tr.patient_id.nunique()}")
print(f"n test  (official test split) = {len(te)}   patients {te.patient_id.nunique()}")
print(f"classes: {list(classes)}")
print()
print(f"positional {N_BINS}-bin argmax P(type | published normalised z):"
      f"  accuracy {acc_pos:.4f}  [{lo:.4f}, {hi:.4f}]  (patient-clustered bootstrap)")
print(f"majority class ({majority}):                              accuracy {acc_maj:.4f}")
print(f"class-prior random guess:                                accuracy "
      f"{acc_prior_random:.4f}")
for nb in (5, 10, 20, 50):
    e = np.linspace(0.0, 1.0, nb + 1)
    e[-1] += 1e-9
    tb = np.clip(np.digitize(tr.norm_z, e) - 1, 0, nb - 1)
    sb = np.clip(np.digitize(te.norm_z, e) - 1, 0, nb - 1)
    cc = np.zeros((nb, len(classes)))
    for b, c in zip(tb, tr.lesion_type):
        cc[b, np.searchsorted(classes, c)] += 1
    cc = cc + 1e-9 * prior
    p = classes[np.argmax(cc / cc.sum(axis=1, keepdims=True), axis=1)[sb]]
    print(f"  bin sweep {nb:>2}: {float((p == te.lesion_type.to_numpy()).mean()):.4f}")
