"""Build a tidy zero-image label table for DeepLesion from DL_info.csv.

No pixels are read. Every column written here comes from the released label file.

Positional feature. DeepLesion publishes `Normalized_lesion_location`, whose third
component is the lesion centroid's normalised z within the volume -- i.e. exactly the
"where in the stack is this slice" quantity the positional null uses. Because a series
contributes only ~2 annotated lesions on average, a within-series min/max normalisation
would be degenerate, so the whole cohort is treated as ONE pseudo-volume and the slice
index is a monotone integer rescaling of the published normalised z. Relative position
inside that pseudo-volume is therefore the published normalised z itself.

Consequence, recorded honestly: the volume-size baseline is degenerate under this
construction (one volume) and is instead supplied as an explicit metadata column
(`n_lesions_in_series`, `slice_range_span`).
"""
import sys
import pandas as pd

src, out = sys.argv[1], sys.argv[2]
d = pd.read_csv(src)

loc = d["Normalized_lesion_location"].str.split(",", expand=True).astype(float)
d["norm_z"] = loc[2]
sr = d["Slice_range"].str.split(",", expand=True).astype(float)
d["slice_range_span"] = sr[1] - sr[0] + 1
sp = d["Spacing_mm_px_"].str.split(",", expand=True).astype(float)
d["slice_thickness_mm"] = sp[2]
d["inplane_spacing_mm"] = sp[0].round(4)
d["series_key"] = (d.Patient_index.astype(str) + "_" + d.Study_index.astype(str)
                   + "_" + d.Series_ID.astype(str))
d["n_lesions_in_series"] = d.groupby("series_key")["series_key"].transform("size")

# The 8-class type field exists only in val (2) and test (3); train (1) is -1.
d = d[d.Coarse_lesion_type > 0].copy()
d["split"] = d.Train_Val_Test.map({2: "training", 3: "test"})

# Monotone integer rescaling of the published normalised z, used as the slice index.
d["z_index"] = (d["norm_z"] * 10000).round().astype(int)
d["pseudo_volume"] = "all"

keep = ["Patient_index", "z_index", "norm_z", "Coarse_lesion_type", "split", "pseudo_volume",
        "slice_thickness_mm", "inplane_spacing_mm", "DICOM_windows", "Image_size",
        "Patient_gender", "Patient_age", "slice_range_span", "n_lesions_in_series"]
d[keep].rename(columns={"Patient_index": "patient_id",
                        "Coarse_lesion_type": "lesion_type"}).to_csv(out, index=False)
print("wrote", out, d.shape, "| split counts:", d.split.value_counts().to_dict())
