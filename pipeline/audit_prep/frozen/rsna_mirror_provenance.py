"""Provenance validation for the third-party RSNA 2019 ICH metadata mirror.

WHY
    The flagship arm needs three things the official Kaggle release does not carry:
    a patient identifier, a series identifier, and a slice ORDER. All three come from
    one third-party pixel-free mirror (`ianpan/rsna-intracranial-hemorrhage-16bit-png`,
    MIT). `prep_rsna_ich.py::verify_ordering` already tests the slice order with a
    run-length statistic. That test is a point estimate with no stated power, it tests
    only the `any` label, and it says nothing at all about the patient identifier.

    This script does four things the existing test does not:
      1. Calibrates the run-length statistic against its own randomised-ordering null
         (mean, sd, range over R draws) so the observed value can be quoted as a
         separation rather than as a bare ratio.
      2. Measures the test's SENSITIVITY: how much of the true ordering has to be
         destroyed before the statistic stops firing.
      3. Repeats the test on the five subtype labels, which were not used to build or
         to check the ordering.
      4. Corroborates the PATIENT identifier -- which the run-length test cannot touch
         -- by asking whether slices that the mirror assigns to one patient share
         acquisition parameters and labels more than slices assigned at random.

WHAT IT CANNOT DO, STATED HERE SO IT IS NOT OVERCLAIMED LATER
    No DICOM header was read. Nothing here verifies the mirror against the original
    release: `stage_2_train.csv` was never obtained, so the image-id set is unchecked
    against the official file, and no ImagePositionPatient value is available to
    confirm the z-order directly. Every statement below is internal-consistency
    evidence, which can only falsify a bad mapping, never certify a good one.

Usage:
    python rsna_mirror_provenance.py <slices.csv> <slice_labels.csv> <rescale.csv> <out.json>
"""
import hashlib
import json
import os
import sys
import time

import numpy as np
import pandas as pd

SUBTYPES = ["epidural", "intraparenchymal", "intraventricular", "subarachnoid", "subdural"]


def sha256(path, cap=None):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def n_runs(v):
    """Number of maximal runs of 1s in a 0/1 vector."""
    v = np.asarray(v)
    if v.size == 0:
        return 0
    return int(((v[1:] == 1) & (v[:-1] == 0)).sum() + (1 if v[0] == 1 else 0))


def runs_stat(order_idx, series_start, series_len, lab, min_pos=3):
    """Mean observed runs and mean expected-if-random runs, over series with >= min_pos
    positives. `order_idx` is a permutation of row positions defining the within-series
    ordering to score. Series are contiguous blocks [start, start+len).
    """
    lab_o = lab[order_idx]
    obs, exp = [], []
    for s, n in zip(series_start, series_len):
        v = lab_o[s:s + n]
        npos = int(v.sum())
        if npos < min_pos:
            continue
        obs.append(n_runs(v))
        exp.append(npos * (n - npos + 1) / n)
    return float(np.mean(obs)), float(np.mean(exp)), len(obs)


def build_blocks(df):
    """Sort by (series, slice) and return the block layout plus the sorted frame."""
    s = df.sort_values(["series_id", "slice"], kind="mergesort").reset_index(drop=True)
    sizes = s.groupby("series_id", sort=False).size().to_numpy()
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]])
    return s, starts, sizes


def shuffled_order(starts, sizes, rng):
    """A within-series random permutation of row positions."""
    out = np.empty(int(sizes.sum()), dtype=np.int64)
    for st, n in zip(starts, sizes):
        out[st:st + n] = st + rng.permutation(n)
    return out


def corrupted_order(starts, sizes, frac, rng):
    """Keep the true order, then randomly relocate a fraction `frac` of each series'
    slices to random positions within that series. frac=0 is the true order, frac=1 is
    a full within-series shuffle.
    """
    out = np.empty(int(sizes.sum()), dtype=np.int64)
    for st, n in zip(starts, sizes):
        idx = np.arange(n)
        k = int(round(frac * n))
        if k >= 2:
            pick = rng.choice(n, size=k, replace=False)
            idx[np.sort(pick)] = pick[rng.permutation(k)]
        out[st:st + n] = st + idx
    return out


def main():
    slices_path, labels_path, rescale_path, out_path = sys.argv[1:5]
    t0 = time.time()
    rng = np.random.default_rng(20260813)
    R = {"tool": "rsna_mirror_provenance", "version": "1.0",
         "generated_utc": time.strftime("%Y-%m-%dT%H:%M:%S+00:00", time.gmtime()),
         "NOT_FOR_SUBMISSION": "contains absolute paths; strip before any anonymised upload",
         "inputs": {}}
    for name, p in [("tidied_slices", slices_path), ("mirror_slice_labels", labels_path),
                    ("mirror_rescale_values", rescale_path)]:
        R["inputs"][name] = {"path": os.path.abspath(p), "bytes": os.path.getsize(p),
                             "sha256": sha256(p)}

    d = pd.read_csv(slices_path)
    raw = pd.read_csv(labels_path)
    resc = pd.read_csv(rescale_path)

    # ---------------------------------------------------------------- structure
    st = {}
    st["n_rows"] = int(len(d))
    st["n_patients"] = int(d.patient_id.nunique())
    st["n_studies"] = int(d.study_id.nunique())
    st["n_series"] = int(d.series_id.nunique())
    st["n_image_ids"] = int(d.image_id.nunique())
    st["image_ids_unique"] = bool(d.image_id.is_unique)
    st["slice_prevalence"] = float(d.label.mean())
    # strict nesting: a series belongs to exactly one study, a study to exactly one patient
    st["series_with_multiple_studies"] = int(
        (d.groupby("series_id").study_id.nunique() > 1).sum())
    st["studies_with_multiple_patients"] = int(
        (d.groupby("study_id").patient_id.nunique() > 1).sum())
    st["patients_with_multiple_studies"] = int(
        (d.groupby("patient_id").study_id.nunique() > 1).sum())
    st["patients_with_multiple_series"] = int(
        (d.groupby("patient_id").series_id.nunique() > 1).sum())
    # IM counter contiguity: is the counter exactly 1..n within every series?
    g = d.groupby("series_id")["slice"]
    mn, mx, cnt, nun = g.min(), g.max(), g.size(), g.nunique()
    contiguous = (mn == 1) & (mx == cnt) & (nun == cnt)
    st["series_with_contiguous_IM_1_to_n"] = int(contiguous.sum())
    st["series_with_gapped_IM"] = int((~contiguous).sum())
    bad = contiguous[~contiguous]
    st["gapped_series_detail"] = [
        {"series_id": s, "n_rows": int(cnt[s]), "im_min": int(mn[s]), "im_max": int(mx[s])}
        for s in bad.index[:10]]
    st["implied_missing_rows_in_gapped_series"] = int((mx[~contiguous] - cnt[~contiguous]).sum())
    # rescale coverage
    st["series_in_labels_missing_from_rescale"] = int(
        (~d.series_id.isin(set(resc.series_id))).groupby(d.series_id).first().sum())
    st["rescale_rows"] = int(len(resc))
    st["rescale_series_unique"] = bool(resc.series_id.is_unique)
    # raw mirror file cross-check
    st["raw_rows"] = int(len(raw))
    st["raw_rows_lost_in_tidy"] = int(len(raw) - len(d))
    R["structure"] = st

    # ------------------------------------------------------- run-length: any label
    s, starts, sizes = build_blocks(d)
    true_order = np.arange(len(s), dtype=np.int64)
    rl = {}
    for lab_name in ["label"] + SUBTYPES:
        lab = s[lab_name].to_numpy(np.int8)
        obs, exp, n_ser = runs_stat(true_order, starts, sizes, lab)
        entry = {"label": "any" if lab_name == "label" else lab_name,
                 "n_series_with_ge3_positives": n_ser,
                 "observed_runs_per_series": round(obs, 4),
                 "expected_if_random": round(exp, 4),
                 "ratio": round(obs / exp, 4)}
        # calibrate against the randomised-ordering null
        if lab_name in ("label", "epidural", "subdural"):
            draws = []
            for _ in range(20):
                o2, e2, _ = runs_stat(shuffled_order(starts, sizes, rng), starts, sizes, lab)
                draws.append(o2 / e2)
            draws = np.array(draws)
            entry["null_ratio_mean"] = round(float(draws.mean()), 4)
            entry["null_ratio_sd"] = round(float(draws.std(ddof=1)), 4)
            entry["null_ratio_range"] = [round(float(draws.min()), 4), round(float(draws.max()), 4)]
            entry["z_vs_null"] = round(float((entry["ratio"] - draws.mean()) / draws.std(ddof=1)), 1)
            entry["null_draws"] = 20
        rl[entry["label"]] = entry
    R["run_length"] = rl

    # ------------------------------------------------------------ sensitivity curve
    lab = s["label"].to_numpy(np.int8)
    curve = []
    for frac in [0.0, 0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90, 1.0]:
        rr = []
        for _ in range(3):
            o2, e2, _ = runs_stat(corrupted_order(starts, sizes, frac, rng), starts, sizes, lab)
            rr.append(o2 / e2)
        curve.append({"fraction_of_slices_relocated": frac,
                      "ratio_mean": round(float(np.mean(rr)), 4),
                      "ratio_min": round(float(np.min(rr)), 4),
                      "ratio_max": round(float(np.max(rr)), 4),
                      "n_draws": 3})
    R["sensitivity_curve"] = {
        "description": ("true ordering degraded by relocating a random fraction of each "
                        "series' slices to random within-series positions; ratio is "
                        "observed runs / expected-if-random on the `any` label"),
        "abort_threshold_in_prep_script": 0.5,
        "curve": curve}

    # ------------------------------------- per-series contiguity, not just the mean
    per = []
    for st_, n in zip(starts, sizes):
        v = lab[st_:st_ + n]
        npos = int(v.sum())
        if npos < 3:
            continue
        per.append((n_runs(v), npos * (n - npos + 1) / n))
    per = np.array(per)
    R["per_series"] = {
        "n_series_scored": int(len(per)),
        "frac_series_with_obs_below_expected": round(float((per[:, 0] < per[:, 1]).mean()), 4),
        "frac_series_forming_a_single_run": round(float((per[:, 0] == 1).mean()), 4),
        "median_observed_runs": float(np.median(per[:, 0])),
        "median_expected_runs": round(float(np.median(per[:, 1])), 4)}

    # ---------------------------------------------- orientation consistency (new test)
    # The run-length test fixes ORDER but not ORIENTATION. If the IM counter ran
    # head-first in some series and foot-first in others, the POOLED distribution of
    # positive-slice relative position would be forced symmetric about 0.5. Measured
    # asymmetry therefore lower-bounds orientation consistency across series.
    relpos = ((s["slice"] - 1) / np.maximum(s["n_slices_in_series"] - 1, 1)).to_numpy()
    pos_rp = relpos[lab == 1]
    obs_mean = float(pos_rp.mean())
    # what does the same statistic give if a random half of the series are reversed?
    ser_id = np.repeat(np.arange(len(sizes)), sizes)
    mixed = []
    for _ in range(20):
        flip = rng.random(len(sizes)) < 0.5
        rp2 = np.where(flip[ser_id], 1.0 - relpos, relpos)
        mixed.append(float(rp2[lab == 1].mean()))
    mixed = np.array(mixed)
    R["orientation"] = {
        "description": ("mean relative position of positive slices; a cohort whose series "
                        "disagree on orientation is pushed toward 0.5"),
        "observed_mean_relpos_of_positive_slices": round(obs_mean, 4),
        "mean_relpos_of_all_slices": round(float(relpos.mean()), 4),
        "randomly_reversing_half_the_series": {
            "mean": round(float(mixed.mean()), 4),
            "sd": round(float(mixed.std(ddof=1)), 4),
            "range": [round(float(mixed.min()), 4), round(float(mixed.max()), 4)],
            "n_draws": 20},
        "z_vs_orientation_mixed_null": round(
            float((obs_mean - mixed.mean()) / mixed.std(ddof=1)), 1),
        "one_sided": ("asymmetry implies consistency; symmetry would have implied nothing, "
                      "so a null result here would not have been evidence against the mirror"),
        "note": ("this establishes that the orientation is CONSISTENT across series, not "
                 "which end of the counter is the vertex. The positional baseline is "
                 "invariant to reversal and is fitted per series, so orientation is not "
                 "needed for the result; it is reported because the mirror's ordering "
                 "claim is stronger if orientation is coherent too.")}

    # ------------------------------------------------- patient identifier corroboration
    # The run-length test says nothing about patient_id. If the mirror's patient_id were
    # fabricated or mis-joined, slices grouped under one patient would agree on
    # acquisition parameters and on the study-level label no more than random groups of
    # the same sizes do.
    pid = {}
    ser = d.groupby("series_id").agg(patient_id=("patient_id", "first"),
                                     study_id=("study_id", "first"),
                                     plane=("plane", "first"),
                                     slope=("rescale_slope", "first"),
                                     intercept=("rescale_intercept", "first"),
                                     nsl=("n_slices_in_series", "first"),
                                     lab=("label", "max")).reset_index()
    multi = ser.groupby("patient_id").filter(lambda x: len(x) >= 2)
    pid["n_patients_with_ge2_series"] = int(multi.patient_id.nunique())
    pid["n_series_in_those_patients"] = int(len(multi))

    def concord(frame, key, col):
        """Fraction of same-group series pairs that agree on `col`, over all groups."""
        agree = tot = 0
        for _, grp in frame.groupby(key, sort=False):
            v = grp[col].to_numpy()
            n = len(v)
            if n < 2:
                continue
            # pairwise agreement via value counts
            _, c = np.unique(v, return_counts=True)
            agree += int((c * (c - 1) // 2).sum())
            tot += n * (n - 1) // 2
        return agree / tot if tot else float("nan")

    for col in ["intercept", "nsl", "lab", "plane", "slope"]:
        obs_c = concord(multi, "patient_id", col)
        draws = []
        for _ in range(20):
            perm = multi.copy()
            perm["patient_id"] = rng.permutation(perm["patient_id"].to_numpy())
            draws.append(concord(perm, "patient_id", col))
        draws = np.array(draws)
        pid[col] = {"within_patient_pair_agreement": round(float(obs_c), 4),
                    "shuffled_patient_id_mean": round(float(draws.mean()), 4),
                    "shuffled_sd": round(float(draws.std(ddof=1)), 4),
                    "z": (round(float((obs_c - draws.mean()) / draws.std(ddof=1)), 1)
                          if draws.std(ddof=1) > 0 else None),
                    "n_draws": 20}
    # study level: recorded because it turns out to be degenerate, and that fact matters
    multi_st = ser.groupby("study_id").filter(lambda x: len(x) >= 2)
    pid["n_studies_with_ge2_series"] = int(multi_st.study_id.nunique())
    pid["study_series_is_one_to_one"] = bool(len(multi_st) == 0)
    pid["study_level_note"] = (
        "study_id and series_id are 1:1 in this release, so every same-patient series "
        "pair below comes from a DIFFERENT study. The agreement is across separate "
        "examinations, not within one acquisition.")

    # CONFOUNDER CONTROL. A mirror that invented patient_id by clustering series on
    # acquisition parameters would reproduce the rescale_intercept agreement for free.
    # Restrict to pairs that ALREADY agree on rescale_intercept and re-ask the label
    # question; if patient_id is a real person-level grouping the label agreement must
    # survive, and if it is an acquisition cluster it must collapse to the base rate.
    def concord_matched(frame, key, col, match_col):
        agree = tot = 0
        for _, grp in frame.groupby(key, sort=False):
            v = grp[col].to_numpy()
            m = grp[match_col].to_numpy()
            n = len(v)
            for i in range(n):
                for j in range(i + 1, n):
                    if m[i] != m[j]:
                        continue
                    tot += 1
                    agree += int(v[i] == v[j])
        return (agree / tot if tot else float("nan")), tot

    obs_c, n_pairs = concord_matched(multi, "patient_id", "lab", "intercept")
    draws = []
    for _ in range(20):
        perm = multi.copy()
        perm["patient_id"] = rng.permutation(perm["patient_id"].to_numpy())
        draws.append(concord_matched(perm, "patient_id", "lab", "intercept")[0])
    draws = np.array(draws)
    pid["label_agreement_matched_on_intercept"] = {
        "within_patient_pair_agreement": round(float(obs_c), 4),
        "n_pairs": int(n_pairs),
        "shuffled_patient_id_mean": round(float(draws.mean()), 4),
        "shuffled_sd": round(float(draws.std(ddof=1)), 4),
        "z": round(float((obs_c - draws.mean()) / draws.std(ddof=1)), 1),
        "n_draws": 20,
        "rationale": ("controls for the possibility that patient_id was reconstructed by "
                      "clustering series on acquisition parameters; the label is not an "
                      "acquisition parameter, so agreement surviving this restriction is "
                      "evidence of a genuine person-level grouping")}
    R["patient_identifier"] = pid

    R["cannot_establish"] = [
        "No DICOM header was read. ImagePositionPatient was never observed, so the "
        "z-order is inferred from label contiguity, not confirmed directly.",
        "stage_2_train.csv (the official Kaggle label file) was not obtained, so the "
        "mirror's image-id set and its labels were never checked row-for-row against "
        "the official release.",
        "Nothing here identifies which end of the IM counter is the vertex.",
        "Every test is internal-consistency evidence. A mapping that is wrong in a way "
        "that preserves within-series label contiguity and within-patient acquisition "
        "homogeneity would pass all of them.",
    ]
    R["elapsed_s"] = round(time.time() - t0, 1)
    with open(out_path, "w") as fh:
        json.dump(R, fh, indent=1, allow_nan=False, default=lambda o: None)
    print(json.dumps({k: v for k, v in R.items()
                      if k in ("structure", "run_length", "sensitivity_curve",
                               "per_series", "orientation", "patient_identifier")},
                     indent=1))
    print("wrote", out_path, "in", R["elapsed_s"], "s")


if __name__ == "__main__":
    main()
