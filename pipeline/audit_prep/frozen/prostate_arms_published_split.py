"""fastMRI Prostate T2 and DWI: independent re-evaluation on the benchmark's own
published train/test split, under the same discipline as the RSNA frozen holdout.

No pooling anywhere: one fit on the published training arm, scored once on the
published test arm. Validation rows are dropped, matching the recorded run.
The locked primary baseline is the 20-bin positional histogram; the other four
zero-image baselines are reported as secondary.
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, "/Users/sathvikloke/Downloads/PhaseDx/pipeline")
from aucutil import (StratifiedAUC, auc_from_counts, boot_ci_counts,
                     cluster_count_matrix, positional_fit_score, relpos_bins, snap)
import s14_trivialbaselines as s14  # noqa: E402

BASE = Path("/Volumes/Research/fastmridatasets/prostate/labels")
ARMS = {
    "fastmri_prostate_t2": dict(csv=BASE / "t2_slice_level_labels.csv", published=0.861),
    "fastmri_prostate_dwi": dict(csv=BASE / "dwi_slice_level_labels.csv", published=0.861),
}
N_BINS, N_BOOT, N_PERM = 20, 2000, 20
SEED = 20260813
OUT = Path("/Users/sathvikloke/Downloads/PhaseDx/pipeline_out/trivial_baselines") / "prostate_arms_published_split.json"


def patient_aggregates(pcode, scores, k):
    order = np.lexsort((snap(scores), pcode))
    p, s = pcode[order], snap(scores)[order]
    counts = np.bincount(p, minlength=k)
    offs = np.concatenate(([0], np.cumsum(counts)))
    start, end, n = offs[:-1], offs[1:], counts.astype(float)
    out = {"mean": np.bincount(p, weights=s, minlength=k) / np.maximum(n, 1),
           "max": s[np.maximum(end - 1, start)]}
    for kt in (1, 3, 5):
        kk = np.minimum(counts, kt)
        acc = np.zeros(k)
        for j in range(kt):
            valid = j < counts
            acc[valid] += s[(end - 1 - j)[valid]]
        out[f"topk{kt}_mean"] = acc / np.maximum(kk, 1)
    for q in (0.75, 0.90):
        pos = (n - 1) * q
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, counts - 1)
        v0, v1 = s[start + i0], s[start + i1]
        out[f"p{int(q * 100)}"] = v0 + (pos - i0) * (v1 - v0)
    return {a: snap(v) for a, v in out.items()}


def run(name, cfg):
    src = cfg["csv"]
    h = hashlib.sha256()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()
    df = pd.read_csv(src)
    y_all = (df.PIRADS.to_numpy() > 2).astype(int)
    split = df.data_split.astype(str).str.lower().to_numpy()
    vol_code, vol_uni = pd.factorize(df.fastmri_rawfile.to_numpy())
    pcode_all, puni = pd.factorize(df.fastmri_pt_id.to_numpy())
    relpos, bins = relpos_bins(df.slice.to_numpy(), vol_code, N_BINS)

    tr = split == "training"
    te = split == "test"
    dropped = int((~tr & ~te).sum())

    # patient depth = slices per volume, truncated (same definition as RSNA)
    tot = np.bincount(pcode_all, minlength=len(puni))
    nvol = pd.DataFrame({"p": pcode_all, "v": vol_code}).groupby("p").v.nunique(
    ).reindex(range(len(puni))).to_numpy()
    depth = (tot / nvol).astype(int)

    s_te, rate, prior = positional_fit_score(y_all[tr], bins[tr], bins[te], N_BINS)
    c_te = np.full(int(te.sum()), prior)
    y_te = y_all[te]
    tp_uni, tp_code = np.unique(pcode_all[te], return_inverse=True)
    kt = len(tp_uni)
    ypat_te = (np.bincount(tp_code, weights=y_te.astype(float), minlength=kt) > 0
               ).astype(int)
    depth_te = depth[tp_uni]

    res = {"arm": name, "source_file": str(src), "source_sha256": sha,
           "estimator": "benchmark's own published train/test split, single fit, "
                        "no pooling",
           "label_rule": "PIRADS > 2", "n_bins": N_BINS, "n_boot": N_BOOT,
           "n_rows": int(len(df)), "n_subjects": int(len(puni)),
           "n_volumes": int(len(vol_uni)),
           "n_train_slices": int(tr.sum()), "n_test_slices": int(te.sum()),
           "n_rows_dropped_validation_or_other": dropped,
           "n_test_subjects": int(kt),
           "train_prevalence": float(prior),
           "test_slice_prevalence": float(y_te.mean()),
           "test_patient_prevalence": float(ypat_te.mean()),
           "n_pos_test_slices": int(y_te.sum()),
           "n_pos_test_patients": int(ypat_te.sum())}

    Cp, Cn = cluster_count_matrix(y_te, s_te, tp_code, kt)
    slice_auc = auc_from_counts(Cp.sum(0), Cn.sum(0))
    assert abs(slice_auc - roc_auc_score(y_te, s_te)) < 1e-9
    lo, hi, used = boot_ci_counts(Cp, Cn, np.random.default_rng(SEED + 1), N_BOOT)
    res.update(slice_auc=float(slice_auc), slice_ci=[lo, hi], slice_ci_reps=used)

    aggs = patient_aggregates(tp_code, s_te, kt)
    sa_flat = StratifiedAUC(ypat_te, aggs["mean"], np.zeros(kt, dtype=int))
    res["patient_auc_mean"] = float(sa_flat.value()[0])
    l2, h2, _ = sa_flat.boot_ci(np.random.default_rng(SEED + 2), N_BOOT)
    res["patient_auc_mean_ci"] = [l2, h2]
    sa_dep = StratifiedAUC(ypat_te, aggs["mean"], depth_te)
    res["patient_auc_mean_depthfixed"] = float(sa_dep.value()[0])
    l3, h3, _ = sa_dep.boot_ci(np.random.default_rng(SEED + 3), N_BOOT)
    res["patient_auc_mean_depthfixed_ci"] = [l3, h3]
    res["pairs_unstratified"] = float(sa_flat.pairs())
    res["pairs_depthfixed"] = float(sa_dep.pairs())

    res["aggregation"], res["aggregation_ci"], res["n_distinct_patient_scores"] = {}, {}, {}
    for i, (a, v) in enumerate(aggs.items()):
        sa = StratifiedAUC(ypat_te, v, np.zeros(kt, dtype=int))
        res["aggregation"][a] = float(sa.value()[0])
        la, ha, _ = sa.boot_ci(np.random.default_rng(SEED + 10 + i), N_BOOT)
        res["aggregation_ci"][a] = [la, ha]
        res["n_distinct_patient_scores"][a] = int(len(np.unique(v)))

    # constant predictor, identical path
    Ccp, Ccn = cluster_count_matrix(y_te, c_te, tp_code, kt)
    res["constant_slice_auc"] = float(auc_from_counts(Ccp.sum(0), Ccn.sum(0)))
    cagg = patient_aggregates(tp_code, c_te, kt)
    res["constant_patient_auc"] = float(
        StratifiedAUC(ypat_te, cagg["mean"], np.zeros(kt, dtype=int)).value()[0])

    # within-volume label permutation null
    nrow = len(df)
    base = np.lexsort((np.arange(nrow), vol_code))
    ps, pp = [], []
    for r in range(N_PERM):
        prg = np.random.default_rng(SEED + 100 + r)
        shuf = np.lexsort((prg.random(nrow), vol_code))
        yperm = np.empty(nrow, dtype=int)
        yperm[base] = y_all[shuf]
        sp, _, _ = positional_fit_score(yperm[tr], bins[tr], bins[te], N_BINS)
        yp = yperm[te]
        ppat = (np.bincount(tp_code, weights=yp.astype(float), minlength=kt) > 0
                ).astype(int)
        if yp.min() == yp.max() or ppat.min() == ppat.max():
            continue
        ps.append(float(roc_auc_score(yp, sp)))
        pp.append(float(roc_auc_score(ppat, patient_aggregates(tp_code, sp, kt)["mean"])))
    res["permutation_null"] = {
        "n_perm": len(ps), "slice_mean": float(np.mean(ps)),
        "slice_range": [float(np.min(ps)), float(np.max(ps))],
        "patient_mean": float(np.mean(pp)),
        "patient_range": [float(np.min(pp)), float(np.max(pp))],
        "slice_excess_over_null": float(slice_auc - np.mean(ps))}

    # secondary baselines, definitions imported from the audit tool
    vol_size = pd.Series(np.ones(nrow)).groupby(vol_code).transform("size").to_numpy(float)
    frame = pd.DataFrame({s14.C_SUBJ: pcode_all, s14.C_LABEL: y_all,
                          s14.C_RELPOS: relpos, s14.C_NSLICES: vol_size,
                          "folder": df.folder.astype(str).to_numpy()})
    bls = [s14.ColumnBaseline(s14.C_NSLICES),
           s14.TreeBaseline(["folder"], name="metadata_tree", max_depth=3, max_levels=30),
           s14.TreeBaseline(["folder"], name="combined_position_metadata",
                            use_relpos=True, max_depth=3, max_levels=30)]
    bls[0].name = "volume_size"
    res["secondary_baselines"] = {}
    for bl in bls:
        s_x = bl.fit(frame[tr]).score(frame[te])
        C1, C2 = cluster_count_matrix(y_te, s_x, tp_code, kt)
        res["secondary_baselines"][bl.name] = {
            "slice_auc": float(auc_from_counts(C1.sum(0), C2.sum(0))),
            "patient_auc_mean": float(StratifiedAUC(
                ypat_te, patient_aggregates(tp_code, s_x, kt)["mean"],
                np.zeros(kt, dtype=int)).value()[0])}

    pub = cfg["published"]
    res["published_slice_auc"] = pub
    res["trivial_fraction_slice_locked"] = float((slice_auc - 0.5) / (pub - 0.5))
    res["trivial_fraction_slice_locked_ci"] = [float((lo - 0.5) / (pub - 0.5)),
                                               float((hi - 0.5) / (pub - 0.5))]
    return res


def main():
    out = {"tool": "prostate_arms_published_split", "version": "1.0",
           "seed": SEED, "arms": {}}
    for name, cfg in ARMS.items():
        r = run(name, cfg)
        out["arms"][name] = r
        print(f"{name:24s} slice {r['slice_auc']:.4f} "
              f"[{r['slice_ci'][0]:.4f},{r['slice_ci'][1]:.4f}] | "
              f"pat {r['patient_auc_mean']:.4f} "
              f"[{r['patient_auc_mean_ci'][0]:.4f},{r['patient_auc_mean_ci'][1]:.4f}] | "
              f"depthfix {r['patient_auc_mean_depthfixed']:.4f} | "
              f"const {r['constant_slice_auc']:.4f}/{r['constant_patient_auc']:.4f} | "
              f"n_test_pat {r['n_test_subjects']}")
    OUT.write_text(json.dumps(out, indent=1))
    print("wrote", OUT)


if __name__ == "__main__":
    main()
