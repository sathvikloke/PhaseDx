"""Build the RSNA 2019 Intracranial Haemorrhage slice-level table. No pixels.

WHY THIS TARGET NEEDED A DETOUR
    RSNA ICH is the benchmark whose OFFICIAL METRIC IS PER-SLICE (multi-label weighted
    log loss over images) and whose released label file makes the slice unlocatable:
    `stage_2_train.csv` is keyed by `ID_<SOPInstanceUID>_<subtype>` and carries only the
    label -- no patient id, no study id, no slice index. The organisers stated on the
    record that "the available fields do not contain information that can determine if
    an image contains intracranial hemorrhage", and the fields they were describing
    include ImagePositionPatient. That claim is the reason this row exists.

    The join is recoverable from a public third-party DICOM-header extract. This script
    uses `ianpan/rsna-intracranial-hemorrhage-16bit-png` on HuggingFace (MIT-licensed
    derived dataset, mirror of the Kaggle release), which publishes two TABULAR files
    next to its image tar:

        slice_labels.csv     patient_ID, study_ID, series_ID, filename, and the six
                             official labels. The filename is `IM%06d_<image_ID>.png`,
                             where the IM counter is the slice's ORDER WITHIN ITS SERIES
                             and <image_ID> is the original Kaggle image id.
        rescale_values.csv   per-series rescale slope/intercept and imaging plane --
                             DICOM header fields, not pixels.

    NEITHER FILE CONTAINS PIXEL DATA. `images.tar` (144 GB) was not downloaded.

THE ORDERING CLAIM IS TESTED, NOT ASSUMED
    Everything here rests on IM%06d being the anatomical z-order rather than an
    arbitrary enumeration. That is checked falsifiably by `verify_ordering()` below: a
    haemorrhage is a spatially contiguous object, so if the index is z-ordered the
    positive slices of a series form very few runs, and if it is arbitrary they form as
    many runs as random placement predicts. Observed 1.384 runs per series against
    7.167 expected under random placement; shuffling the labels within each series
    returns 7.170, recovering the random expectation exactly. The index is z-ordered.
    The test is re-run on every invocation and the script aborts if it stops holding.

    Note this fixes ORDER, not orientation: whether IM000001 is the vertex or the skull
    base is not determined by the test. It does not need to be -- the positional
    baseline is invariant to reversing the axis, and it is applied per series.

WHAT THE LABEL IS
    `any` -- the official binary "this slice contains any intracranial haemorrhage",
    which is the label whose slice-level ROC AUC the peer-reviewed comparator reports.
    The five subtypes are carried through so the per-subtype arms can be run too.

COLUMN DISCIPLINE
    Kept as metadata: imaging plane, per-series rescale slope and intercept (acquisition
    header fields), and the number of slices in the series. Excluded: nothing
    image-derived, because nothing image-derived is present. The identifiers are opaque
    hashes and are used only for grouping.
"""
import sys

import numpy as np
import pandas as pd

labels_path, rescale_path, out = sys.argv[1], sys.argv[2], sys.argv[3]

d = pd.read_csv(labels_path)
d["slice"] = d.filename.str.extract(r"^IM(\d+)_").astype(int)
d["image_id"] = d.filename.str.extract(r"_(ID_[0-9a-f]+)\.png$")
assert d.image_id.notna().all(), "filename did not carry an image id"
assert d.image_id.is_unique, "image ids are not unique"


def verify_ordering(df, min_pos=3, seed=0):
    """Falsifiable test that the IM counter is the anatomical z-order.

    Returns (observed runs, runs expected under random placement, runs after shuffling
    within series). A z-ordered index makes haemorrhage slices contiguous, so observed
    << expected; an arbitrary index makes observed == expected.
    """
    def n_runs(v):
        v = np.asarray(v)
        if len(v) == 0:
            return 0
        return int(((v[1:] == 1) & (v[:-1] == 0)).sum() + (1 if v[0] == 1 else 0))

    s = df.sort_values(["series_ID", "slice"])
    keep = s.groupby("series_ID")["any"].transform("sum") >= min_pos
    s = s[keep]
    g = s.groupby("series_ID")["any"]
    obs = g.apply(n_runs).mean()
    npos, n = g.sum(), g.size()
    exp = (npos * (n - npos + 1) / n).mean()
    rng = np.random.default_rng(seed)
    shuf = g.apply(lambda v: n_runs(rng.permutation(v.values))).mean()
    return obs, exp, shuf


obs, exp, shuf = verify_ordering(d)
print(f"ordering test | observed runs {obs:.3f} | expected-if-random {exp:.3f} "
      f"| after within-series shuffle {shuf:.3f} | ratio {obs / exp:.4f}")
assert obs / exp < 0.5, "IM index does not behave like an anatomical z-order -- ABORT"
assert abs(shuf / exp - 1) < 0.1, "the random-placement expectation is miscalibrated"

r = pd.read_csv(rescale_path)
d = d.merge(r[["series_id", "plane", "rescale_slope", "rescale_intercept"]],
            left_on="series_ID", right_on="series_id", how="left").drop(
    columns=["series_id"])
d["n_slices_in_series"] = d.series_ID.map(d.groupby("series_ID").size())

out_cols = ["patient_ID", "study_ID", "series_ID", "image_id", "slice",
            "any", "epidural", "intraparenchymal", "intraventricular",
            "subarachnoid", "subdural",
            "plane", "rescale_slope", "rescale_intercept", "n_slices_in_series"]
d = d[out_cols].rename(columns={"patient_ID": "patient_id", "study_ID": "study_id",
                                "series_ID": "series_id", "any": "label"})
d.to_csv(out, index=False)
print("wrote", out, d.shape,
      "| patients", d.patient_id.nunique(), "| series", d.series_id.nunique(),
      "| slice prevalence", round(d.label.mean(), 5),
      "| series prevalence", round(d.groupby("series_id").label.max().mean(), 5),
      "| patient prevalence", round(d.groupby("patient_id").label.max().mean(), 5))
