"""Build the LUNA16 false-positive-reduction candidate table.

`candidates_V2.csv` publishes, per candidate, the series id, the candidate's world
coordinates in mm and a binary nodule/not-nodule class. 754,975 candidates over 888
scans, 1,557 of them true nodules. No pixels.

The world z coordinate is not comparable across scans because it carries the scanner
table offset, so relative position within the scan is used: (z - min z in this scan) /
(max - min), computed over that scan's own ~850 candidates. With that many candidates
per scan the endpoints are well determined, which is exactly the condition that fails
on DeepLesion and is why this one needs no supplied position column.

LUNA16's own metric is FROC / CPM on candidates, not AUROC, so the AUROC computed here
is a reference level and NOT directly comparable to a published CPM. The sensitivity at
fixed false positives per scan is computed separately by audit_luna16_froc.py, which is
the number that IS on the published scale.
"""
import sys

import pandas as pd

src, out = sys.argv[1], sys.argv[2]
c = pd.read_csv(src)
c = c.rename(columns={"seriesuid": "series_id", "coordZ": "z_mm", "class": "label"})
c["subject_id"] = c.series_id
c["slice"] = c.z_mm
n = c.groupby("series_id").size()
c["n_candidates_in_scan"] = c.series_id.map(n)
c[["subject_id", "series_id", "slice", "z_mm", "label", "coordX", "coordY",
   "n_candidates_in_scan"]].to_csv(out, index=False)
print("wrote", out, c.shape, "| scans", c.series_id.nunique(),
      "| positives", int(c.label.sum()), "| prevalence", round(c.label.mean(), 6))
