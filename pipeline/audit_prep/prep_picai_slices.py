#!/usr/bin/env python3
"""Build a SLICE-level label table for PI-CAI from the human-expert delineations.

    venv/bin/python pipeline/audit_prep/prep_picai_slices.py DELINEATION_DIR OUT.csv

Why this exists
---------------
An earlier round reported PI-CAI at case level only and recorded the locked
positional baseline as NOT ESTIMABLE there, because the public marksheet carries
no slice index. That was accurate about the marksheet and wrong about the
benchmark: PI-CAI publishes 1,295 human-expert csPCa lesion delineations as
volumes, which are pixel-free label data. The blocker was a NIfTI reader, not
access. `prep_picai.py` says so in its own docstring.

This reads those volumes and derives, per slice, whether that slice intersects a
delineated clinically significant lesion. It reads NO image intensity: the only
files opened are the label volumes, which are integer lesion masks.

The delineations ship in two grids, `original` and `resampled`, which differ in
in-plane resolution only -- slice counts are identical, because resampling is
in-plane. `original` is used, being the acquisition grid on which "where in the
stack" is defined.

A case with no clinically significant cancer has an all-zero volume and
contributes all-negative slices. That is correct and is not a missing value.

THE ANALYSIS PROTOCOL IS NOT SET HERE. This script only builds the table. It is
scored by frozen_arm_holdout.py under the parameters already locked for the
flagship -- 20 bins, 30% subject-disjoint holdout, seed 20260813, mean
aggregation, 24 draws -- with nothing re-tuned for this arm.
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

try:
    import nibabel as nib
except ImportError:
    sys.exit("needs nibabel:  venv/bin/python -m pip install nibabel")

SLICE_AXIS = 2


def main() -> None:
    src = Path(sys.argv[1])
    out = Path(sys.argv[2])
    files = sorted(src.glob("*.nii.gz"))
    if not files:
        sys.exit(f"no .nii.gz under {src}")

    t0 = time.time()
    rows = []
    n_pos_cases = 0
    for i, f in enumerate(files):
        stem = f.name.replace(".nii.gz", "")
        patient_id, _, study_id = stem.partition("_")
        vol = np.asanyarray(nib.load(f).dataobj)
        if vol.ndim != 3:
            sys.exit(f"unexpected shape {vol.shape} in {f.name}")
        n_sl = vol.shape[SLICE_AXIS]
        # a slice is positive iff it intersects any delineated lesion voxel
        pos = (vol > 0).any(axis=tuple(a for a in range(3) if a != SLICE_AXIS))
        n_pos_cases += int(pos.any())
        for s in range(n_sl):
            rows.append((patient_id, study_id, stem, s, int(pos[s]), n_sl))
        if (i + 1) % 250 == 0:
            print(f"  {i+1}/{len(files)} volumes")

    d = pd.DataFrame(rows, columns=["patient_id", "study_id", "case_id",
                                    "slice", "label", "n_slices_in_volume"])
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    print(f"\nwrote {out}  {d.shape}")
    print(f"  cases              : {d.case_id.nunique()}  "
          f"(patients {d.patient_id.nunique()})")
    print(f"  cases with a lesion: {n_pos_cases}  "
          f"({n_pos_cases / d.case_id.nunique():.3f})")
    print(f"  slices             : {len(d)}  "
          f"(median per volume {int(d.groupby('case_id').size().median())})")
    print(f"  slice prevalence   : {d.label.mean():.5f}")
    print(f"  ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
