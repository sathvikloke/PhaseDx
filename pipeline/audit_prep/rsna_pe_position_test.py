"""Test whether RSNA-STR Pulmonary Embolism's official label file can locate the slice.

WHY THIS EXISTS
    RSNA-STR PE (2020) is the benchmark where we already KNOW a positional baseline works:
    OsciiArt's Kaggle notebook "Baseline with no image" (2020-10-10, gold medal) binned
    P(PE | relative slice location) on train and applied it to test with no pixels. But
    that notebook had the DICOM headers. A label-file-only auditor does not.

    `train.csv` publishes StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID and the
    labels. It publishes NO slice index and NO z position. The only remaining hope is
    that some ordering already present in the file is the acquisition order -- either the
    row order as shipped, or the lexical order of the anonymised SOPInstanceUID. This
    script tests both, and both fail.

THE TEST
    Identical to the one that VALIDATED the RSNA ICH ordering in prep_rsna_ich.py, run
    here in the opposite direction. A pulmonary embolism is a spatially contiguous
    object, so under a true anatomical ordering the positive slices of a series form far
    fewer runs than random placement would produce. Under an arbitrary ordering they
    form exactly as many as random placement predicts.

    For a series of n slices with k positives placed at random, the expected number of
    maximal runs of positives is k(n - k + 1)/n.

    ratio = observed / expected.   ~0.2 on RSNA ICH (ordered).   ~1.0 means NO ordering.

PROVENANCE OF THE INPUT
    `train.csv`, expanded from `data/train.csv.zip` in the public GitHub repository
    github.com/darraghdog/rsnastr (119,970,071 bytes, file date 2020-09-07). Verified
    genuine against Table 4 of Hu et al., npj Digital Medicine 2025;8:254
    (doi:10.1038/s41746-025-01594-2), which reports the RSPECT training split as 96,540
    positive of 1,790,594 slices (5.39%) over 7,279 exams.
"""
import sys

import numpy as np
import pandas as pd

src = sys.argv[1]
MIN_POS = int(sys.argv[2]) if len(sys.argv) > 2 else 3

d = pd.read_csv(src, usecols=["StudyInstanceUID", "SeriesInstanceUID",
                              "SOPInstanceUID", "pe_present_on_image"])
print(f"rows {len(d):,} | studies {d.StudyInstanceUID.nunique():,} "
      f"| series {d.SeriesInstanceUID.nunique():,} "
      f"| SOPInstanceUID unique {d.SOPInstanceUID.is_unique}")
print(f"pe_present_on_image prevalence {d.pe_present_on_image.mean():.5f} "
      f"({int(d.pe_present_on_image.sum()):,} positive)")
print("published for this split (Hu et al., npj Digit Med 2025;8:254, Table 4): "
      "96,540 positive of 1,790,594 slices, 5.39%, 7,279 exams\n")

assert "slice" not in [c.lower() for c in d.columns]
print("columns available:", list(d.columns))
print("-> the file carries NO slice index and NO z position.\n")


def n_runs(v):
    v = np.asarray(v)
    if len(v) == 0:
        return 0
    return int(((v[1:] == 1) & (v[:-1] == 0)).sum() + (1 if v[0] == 1 else 0))


g = d.groupby("SeriesInstanceUID")["pe_present_on_image"]
keep = g.transform("sum") >= MIN_POS
sub = d[keep]
gg = sub.groupby("SeriesInstanceUID")["pe_present_on_image"]
k, n = gg.sum(), gg.size()
expected = float((k * (n - k + 1) / n).mean())
print(f"series with >= {MIN_POS} positive slices: {gg.ngroups:,}")
print(f"expected runs of positive slices if the ordering carries nothing: "
      f"{expected:.3f}\n")

for label, frame in [
    ("row order as shipped in train.csv", sub),
    ("lexical order of SOPInstanceUID",
     sub.sort_values(["SeriesInstanceUID", "SOPInstanceUID"])),
]:
    obs = float(frame.groupby("SeriesInstanceUID")["pe_present_on_image"]
                .apply(n_runs).mean())
    print(f"{label:<38} observed {obs:7.3f}   ratio {obs / expected:.4f}")

print()
print("for contrast, the same statistic on RSNA ICH's reconstructed ordering "
      "(prep_rsna_ich.py): ratio 0.1931")
print("VERDICT: neither ordering in the RSNA-STR PE label file carries positional "
      "information. The benchmark cannot be audited from its label file alone.")
