"""Reconstruct Burduja, Ionescu & Verga's (Sensors 2020;20(19):5611) evaluation conditions.

Their Table 3 reports a SLICE-LEVEL ROC AUC of 0.9843 for the label 'any' -- the same
metric and the same evaluation unit the positional null produces -- on a held-out split
of the RSNA 2019 Intracranial Haemorrhage OFFICIAL TRAINING SET. Verbatim:

    "The dataset is split into a training set of 752,803 slices and a test set of
     121,232 slices... We further split the official training data into a training set
     of 728,513 slices and a validation set 24,290 slices... In total, the training set
     contains 21,000 CT scans, the validation set contains 744 CT scans and the test
     set contains 3528 CT scans."

So their reported number is on 744 scans drawn from the same 21,744-scan file we hold.
It is NOT on the competition's hidden stage-2 test set, whose labels were never
released. That is what makes the comparison possible at all.

WHAT IS AND IS NOT REPRODUCIBLE HERE
    Reproducible: the cohort (identical file), the label ('any'), the metric (slice
    ROC AUC), the evaluation unit (the slice), and the split GEOMETRY -- 744 held-out
    scans, ~24,290 slices, grouped by scan.
    Not reproducible: their particular random draw, which they do not publish. So this
    script repeats the draw N times and reports the distribution, exactly as
    deeplesion_yan_conditions.py does, so the comparison is not hostage to one seed.
    The comparison is therefore APPROXIMATE, and the audit records it as such.

A NOTE ON THEIR SPLIT THAT CUTS AGAINST US, NOT FOR US
    Their split is by SCAN. The 21,744 scans come from only 18,938 patients, so a
    scan-level split lets the same patient appear in both arms. The zero-image null
    benefits from that leak, so this script ALSO reports the patient-disjoint variant,
    which is the honest number and the one the audit quotes. Both are printed.
"""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1]
N_BINS = int(sys.argv[2]) if len(sys.argv) > 2 else 20
N_REP = int(sys.argv[3]) if len(sys.argv) > 3 else 200
N_HELD_OUT_SCANS = 744  # their validation set size, verbatim

# Burduja et al. Table 3, slice-level ROC AUC, per label. Their column order is
# ours: 'any' plus the five official subtypes.
PUBLISHED = {  # label column -> (their BiLSTM slice AUC, their plain ResNeXt slice AUC)
    "label": ("Any", 0.9843, 0.9752),
    "epidural": ("Epidural (EPH)", 0.9851, 0.9703),
    "intraparenchymal": ("Intraparenchymal (IPH)", 0.9927, 0.9883),
    "intraventricular": ("Intraventricular (IVH)", 0.9970, 0.9953),
    "subarachnoid": ("Subarachnoid (SAH)", 0.9821, 0.9644),
    "subdural": ("Subdural (SDH)", 0.9682, 0.9576),
}

d = pd.read_csv(src, usecols=["patient_id", "series_id", "slice"] + list(PUBLISHED))

# relative position within the series, the harness's definition
g = d.groupby("series_id")["slice"]
lo, hi = g.transform("min").to_numpy(), g.transform("max").to_numpy()
span = np.where(hi > lo, hi - lo, 1.0)
relpos = (d.slice.to_numpy() - lo) / span
edges = np.linspace(0.0, 1.0, N_BINS + 1)
edges[-1] += 1e-9
bins = np.clip(np.digitize(relpos, edges) - 1, 0, N_BINS - 1)
series = d.series_id.to_numpy()
patient = d.patient_id.to_numpy()


def auc(yt, sc):
    yt = np.asarray(yt)
    if yt.min() == yt.max():
        return np.nan
    r = pd.Series(sc).rank().to_numpy()
    n1 = yt.sum()
    n0 = len(yt) - n1
    return float((r[yt == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def run(y, group_by, n_held_out):
    keys = np.unique(group_by)
    aucs, nsl = [], []
    for rep in range(N_REP):
        rng = np.random.default_rng(rep)
        held = set(rng.permutation(keys)[:n_held_out].tolist())
        is_test = np.fromiter((k in held for k in group_by), bool, len(group_by))
        is_train = ~is_test
        # P(label | relative position bin), fitted on train slices only
        num = np.bincount(bins[is_train], weights=y[is_train], minlength=N_BINS)
        den = np.bincount(bins[is_train], minlength=N_BINS)
        prior = y[is_train].mean()
        rate = np.where(den > 0, num / np.maximum(den, 1), prior)
        aucs.append(auc(y[is_test], rate[bins[is_test]]))
        nsl.append(int(is_test.sum()))
    return np.array(aucs, float), float(np.mean(nsl))


n_scans, n_pats = pd.Series(series).nunique(), pd.Series(patient).nunique()
n_pat_held = int(round(N_HELD_OUT_SCANS / n_scans * n_pats))
print(f"file: {len(d):,} slices | {n_scans:,} scans | {n_pats:,} patients")
print(f"replicates: {N_REP} random draws per label, {N_BINS}-bin positional baseline")
print(f"held out: {N_HELD_OUT_SCANS} scans (their protocol) / {n_pat_held} patients "
      f"(patient-disjoint variant); they report 24,290 held-out slices\n")

hdr = (f"{'label':<24}{'prev':>8}{'zero-image slice AUROC':>26}"
       f"{'  (patient-disjoint)':>21}{'published':>11}{'trivial frac':>14}")
print(hdr)
print("-" * len(hdr))
for col, (name, pub_lstm, pub_plain) in PUBLISHED.items():
    y = d[col].to_numpy().astype(float)
    a_scan, nsl = run(y, series, N_HELD_OUT_SCANS)
    a_pat, _ = run(y, patient, n_pat_held)
    tf = (a_scan.mean() - 0.5) / (pub_lstm - 0.5)
    print(f"{name:<24}{y.mean():>8.4f}"
          f"{a_scan.mean():>14.4f} [{np.percentile(a_scan, 2.5):.3f},"
          f"{np.percentile(a_scan, 97.5):.3f}]"
          f"{a_pat.mean():>21.4f}{pub_lstm:>11.4f}{tf:>14.4f}")

print()
print("published = Burduja, Ionescu & Verga, Sensors 2020;20(19):5611, Table 3,")
print("            ResNeXt-101 32x8d + bidirectional LSTM, slice-level ROC AUC.")
print("            Their plain ResNeXt-101 column, same metric and unit, runs")
print("            0.9752 / 0.9703 / 0.9883 / 0.9953 / 0.9644 / 0.9576 in the same order.")
print("trivial frac = (zero-image - 0.5) / (published - 0.5), their-protocol split.")
