"""Build the Duke Breast Cancer MRI slice-level table under the DATA OWNERS' definition.

Mazurowski lab tutorial, verbatim: "take all 2D slices that contain a tumor bounding box
to be positive, (labeled with "1"), and all other slices at least five slices away from
positive slices to be negative ("0")". Slices within five of the box but outside it are
neither, and are dropped here exactly as the tutorial drops them.

Two tabular sources, no pixels:
  * Annotation_Boxes.csv    -- per-patient inclusive tumour slice range (922 rows)
  * TCIA getSeries metadata -- per-series ImageCount, manufacturer, model, software

Slice count per patient is the modal ImageCount over that patient's non-derived MR
series. Validated: for all 922 patients the annotated End Slice is strictly less than
that count, and the modal and maximum counts agree, so the two files are on the same
slice indexing.

Read this result knowing that EVERY patient in this cohort has cancer. The slice task is
within-patient localisation, not diagnosis, and it is positional by construction.
"""
import sys

import pandas as pd

boxes_path, series_path, out = sys.argv[1], sys.argv[2], sys.argv[3]
GAP = 5

b = pd.read_csv(boxes_path).rename(columns={
    "Patient ID": "patient_id", "Start Slice": "start_slice", "End Slice": "end_slice"})
s = pd.read_csv(series_path)
s = s[(s.Modality == "MR")
      & (~s.SeriesDescription.astype(str).str.contains("Segmentation", case=False,
                                                       na=False))]
agg = s.groupby("PatientID").agg(
    n_slices=("ImageCount", lambda x: int(x.mode().iloc[0])),
    manufacturer=("Manufacturer", lambda x: x.mode().iloc[0]),
    scanner_model=("ManufacturerModelName", lambda x: x.mode().iloc[0]),
    software_version=("SoftwareVersions", lambda x: str(x.mode().iloc[0])),
    n_series=("SeriesInstanceUID", "size"),
)
b = b.merge(agg, left_on="patient_id", right_index=True, how="inner")
assert (b["end_slice"] < b["n_slices"]).all(), "box exceeds the series slice count"

rows = []
for r in b.itertuples(index=False):
    lo, hi = int(r.start_slice), int(r.end_slice)
    for z in range(int(r.n_slices)):
        if lo <= z <= hi:
            y = 1
        elif z < lo - GAP or z > hi + GAP:
            y = 0
        else:
            continue  # the tutorial's exclusion band
        rows.append(dict(patient_id=r.patient_id, slice=z, label=y,
                         manufacturer=r.manufacturer, scanner_model=r.scanner_model,
                         software_version=r.software_version, n_series=r.n_series))
d = pd.DataFrame(rows)
d.to_csv(out, index=False)
print("wrote", out, d.shape, "| patients", d.patient_id.nunique(),
      "| slice prevalence", round(d.label.mean(), 4),
      "| patient prevalence", round(d.groupby("patient_id").label.max().mean(), 4))
