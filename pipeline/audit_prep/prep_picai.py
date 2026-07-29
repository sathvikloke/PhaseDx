"""Build a CASE-level zero-image table for PI-CAI from the public marksheet.

PI-CAI's evaluation unit is the case (patient/examination), not the slice, and the
public release publishes NO slice-level metric. The positional null is therefore not
applicable to PI-CAI as released: per-slice positivity would require downloading the
1,295 human-expert lesion delineation VOLUMES from picai_labels, which is a real
download and a NIfTI reader, not one curl of a CSV.

What can be audited from the marksheet alone is the METADATA null at the benchmark's
own evaluation unit.

Columns dropped, and why, stated before the numbers so the reader can check the call:
  case_ISUP, lesion_ISUP, lesion_GS, lesion_PIRADS, histopath_type
        outcome-derived. case_csPCa is defined as ISUP >= 2, so these are the label.
  prostate_volume, psad
        MEASURED FROM THE MRI. Including either would break the zero-image guarantee
        that is the whole premise of this tool. psad = psa / prostate_volume, so it
        inherits the image dependence.
Retained as genuinely pixel-free: patient_age, psa (a blood test), center, and the
year of acquisition.

The split column encodes the OFFICIAL picai_baseline cross-validation fold 0 as the
test arm and folds 1-4 as the training arm, so that the reported official-split row is
on the same partition every challenge submission was trained under. The harness also
reports its own 5-fold subject-level CV over all 1500 cases in the same payload.
"""
import glob
import json
import sys

import pandas as pd

marksheet, splits_dir, out = sys.argv[1], sys.argv[2], sys.argv[3]

m = pd.read_csv(marksheet)
m["case_key"] = m.patient_id.astype(str) + "_" + m.study_id.astype(str)
fold0 = set(json.load(open(f"{splits_dir}/valid-fold-0.json"))["subject_list"])
covered = set()
for f in sorted(glob.glob(f"{splits_dir}/valid-fold-*.json")):
    covered |= set(json.load(open(f))["subject_list"])
assert covered >= set(m.case_key), "official folds do not cover every marksheet case"

m["split"] = m.case_key.map(lambda k: "test" if k in fold0 else "training")
m["mri_year"] = pd.to_datetime(m.mri_date, errors="coerce").dt.year
m["slice"] = 0          # PI-CAI publishes no slice index; recorded, not invented
m["label"] = (m.case_csPCa.str.upper() == "YES").astype(int)

keep = ["patient_id", "slice", "label", "split", "patient_age", "psa", "center",
        "mri_year"]
m[keep].to_csv(out, index=False)
print("wrote", out, m[keep].shape,
      "| case prevalence", round(m.label.mean(), 4),
      "| test arm", int((m.split == 'test').sum()),
      "| test prevalence", round(m.loc[m.split == 'test', 'label'].mean(), 4))
