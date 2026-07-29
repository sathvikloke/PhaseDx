"""Build a slice-level label table for fastMRI+ knee.

fastMRI+ publishes POSITIVE annotations only: one row per bounding box, carrying the
volume id and the slice index. Negative slices are implicit, so the table cannot be
built from the annotation file alone -- the number of slices in each volume is needed,
and that lives in the fastMRI HDF5 headers.

We therefore read the DATASET SHAPE of each volume's kspace/reconstruction (an HDF5
header read; no array is materialised and no pixel is decoded) and, for the volumes we
hold locally, expand the annotation file into a full slice roster. This is the reason
fastMRI+ is NOT a label-file-only target and must not be described as one.

Acquisition metadata is taken from the fastMRI HDF5 attributes that are header fields
(acquisition protocol, matrix size, number of coils) -- provenance, not pixels.
"""
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ann_path, file_list_path, knee_dir, out = sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4]
target_label = sys.argv[5] if len(sys.argv) > 5 else "Meniscus Tear"

ann = pd.read_csv(ann_path)
roster = [l.strip() for l in open(file_list_path) if l.strip()]
roster_set = set(roster)

rows = []
meta = {}
for p in sorted(Path(knee_dir).glob("*.h5")):
    stem = p.stem
    if stem not in roster_set:
        continue
    with h5py.File(p, "r") as f:
        if "kspace" not in f:
            continue
        n_sl, n_coil = f["kspace"].shape[0], f["kspace"].shape[1]
        ncols = f["kspace"].shape[-1]
        acq = f.attrs.get("acquisition", "")
        acq = acq.decode() if isinstance(acq, bytes) else str(acq)
    meta[stem] = dict(n_slices=int(n_sl), n_coils=int(n_coil),
                      readout_cols=int(ncols), acquisition=acq)

print(f"volumes on the fastMRI+ roster that we hold locally: {len(meta)} of {len(roster)}")

ann = ann[ann.file.isin(meta)]
pos = set(zip(ann.loc[ann.label == target_label, "file"],
              ann.loc[ann.label == target_label, "slice"]))
any_pos = set(zip(ann.file, ann.slice))

for vol, m in meta.items():
    for s in range(m["n_slices"]):
        rows.append(dict(volume_id=vol, slice=s,
                         label=int((vol, s) in pos),
                         any_finding=int((vol, s) in any_pos),
                         n_coils=m["n_coils"], readout_cols=m["readout_cols"],
                         acquisition=m["acquisition"]))
d = pd.DataFrame(rows)
# fastMRI knee volumes are one subject each; the file id is the subject id.
d["subject_id"] = d.volume_id
d.to_csv(out, index=False)
print("wrote", out, d.shape,
      "| slice prevalence", round(d.label.mean(), 4),
      "| volumes with >=1 positive slice", d.groupby("volume_id").label.max().sum(),
      "| any-finding prevalence", round(d.any_finding.mean(), 4))
