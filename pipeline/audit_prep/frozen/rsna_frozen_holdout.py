"""RSNA 2019 ICH: frozen single holdout as the primary estimator.

PRE-SPECIFIED BEFORE ANY HELD-OUT NUMBER WAS LOOKED AT
    baseline        20-bin relative-position histogram, P(label | position bin),
                    fitted on training slices only.  This is the LOCKED primary
                    baseline; nothing is selected on the test data.
    holdout         30% of PATIENTS, drawn uniformly at random without stratification,
                    single fit on the remaining 70%, no pooling of any kind.
    primary seed    20260813 (draw index 0 of the family below)
    family          24 independent draws, seeds 20260813 + i, i = 0..23
    aggregation     mean is the pre-specified primary patient operator;
                    max / top-k / percentile are sensitivity analyses.
    interval        95% percentile bootstrap resampling PATIENTS, 2000 replicates.
    depth           a patient's slices per volume, int(total slices / n volumes) --
                    the definition that reproduces the manuscript's printed pair
                    counts (10,065,308 within exact depth over 36 strata).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
from aucutil import (StratifiedAUC, auc_from_counts, cluster_count_matrix,
                     boot_ci_counts, positional_fit_score, relpos_bins, snap)

SRC = Path("/Users/sathvikloke/Downloads/PhaseDx/pipeline_out/audit_data/rsna_ich_slices.csv")
OUT = Path("/Users/sathvikloke/Downloads/PhaseDx/pipeline_out/trivial_baselines") / "rsna_frozen_holdout.json"

LABELS = ["label", "epidural", "intraparenchymal", "intraventricular",
          "subarachnoid", "subdural"]
PRETTY = {"label": "any"}
PUBLISHED = {"label": 0.9843, "epidural": 0.9851, "intraparenchymal": 0.9927,
             "intraventricular": 0.9970, "subarachnoid": 0.9821, "subdural": 0.9682}

N_BINS = 20
HOLDOUT_FRAC = 0.30
PRIMARY_SEED = 20260813
N_DRAWS = 24
N_BOOT = 2000
N_PERM = 20


# ---------------------------------------------------------------- aggregation
def patient_aggregates(pcode_test, scores_test, k_patients):
    """All patient operators in one pass over score-sorted groups."""
    order = np.lexsort((snap(scores_test), pcode_test))
    p = pcode_test[order]
    s = snap(scores_test)[order]
    counts = np.bincount(p, minlength=k_patients)
    offs = np.concatenate(([0], np.cumsum(counts)))
    cs = np.concatenate(([0.0], np.cumsum(s)))
    start, end, n = offs[:-1], offs[1:], counts.astype(float)
    out = {}
    # per-patient sum by bincount, NOT by differencing a global cumsum. Differencing a
    # cumsum whose running total reaches ~3e4 leaves 1e-13 residue in the per-patient
    # mean, which survives snapping at 1e-12 for some patients and not others and hands
    # the CONSTANT predictor a spurious ordering (measured: patient AUROC 0.498 instead
    # of 0.500 on the 'any' label). bincount sums each patient's ~40 terms alone.
    out["mean"] = np.bincount(p, weights=s, minlength=k_patients) / np.maximum(n, 1)
    out["max"] = s[np.maximum(end - 1, start)]
    for kt in (1, 3, 5):
        kk = np.minimum(counts, kt)
        acc = np.zeros(k_patients)
        for j in range(kt):
            valid = j < counts
            acc[valid] += s[(end - 1 - j)[valid]]
        out[f"topk{kt}_mean"] = acc / np.maximum(kk, 1)
    for q in (0.75, 0.90):
        pos = (n - 1) * q
        i0 = np.floor(pos).astype(int)
        i1 = np.minimum(i0 + 1, counts - 1)
        frac = pos - i0
        v0 = s[start + i0]
        v1 = s[start + i1]
        out[f"p{int(q * 100)}"] = v0 + frac * (v1 - v0)
    return {k: snap(v) for k, v in out.items()}


def main():
    t0 = time.time()
    h = hashlib.sha256()
    with open(SRC, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()

    d = pd.read_csv(SRC, usecols=["patient_id", "series_id", "slice",
                                  "n_slices_in_series"] + LABELS)
    n = len(d)
    ser_code, ser_uni = pd.factorize(d.series_id.to_numpy())
    pcode, puni = pd.factorize(d.patient_id.to_numpy())
    k = len(puni)
    relpos, bins = relpos_bins(d.slice.to_numpy(), ser_code, N_BINS)

    # patient-level depth = slices per volume, truncated
    tot = np.bincount(pcode, minlength=k)
    nvol = pd.DataFrame({"p": pcode, "v": ser_code}).groupby("p").v.nunique().reindex(
        range(k)).to_numpy()
    depth = (tot / nvol).astype(int)

    Y = {c: d[c].to_numpy().astype(int) for c in LABELS}
    PY = {c: np.bincount(pcode, weights=Y[c].astype(float), minlength=k) > 0
          for c in LABELS}
    PY = {c: v.astype(int) for c, v in PY.items()}

    report = {
        "tool": "rsna_frozen_holdout",
        "version": "1.0",
        "source_file": str(SRC),
        "source_sha256": sha,
        "n_slices": int(n), "n_series": int(len(ser_uni)), "n_patients": int(k),
        "n_bins": N_BINS, "holdout_fraction": HOLDOUT_FRAC,
        "primary_seed": PRIMARY_SEED, "n_draws": N_DRAWS,
        "n_boot": N_BOOT, "n_perm": N_PERM,
        "depth_definition": "int(total slices for the patient / number of that "
                            "patient's series); reproduces 10,065,308 within-exact-"
                            "depth pairs over 36 informative strata on the full cohort",
        "estimator": "single frozen patient-disjoint holdout, one fit, no pooling",
    }

    # ------------------------------------------------------------------ draws
    def draw(seed):
        rng = np.random.default_rng(seed)
        perm = rng.permutation(k)
        n_test = int(round(HOLDOUT_FRAC * k))
        test_p = np.zeros(k, dtype=bool)
        test_p[perm[:n_test]] = True
        return test_p

    def evaluate(seed, y, ypat, full=False, published=None):
        test_p = draw(seed)
        te = test_p[pcode]
        tr = ~te
        s_te, rate, prior = positional_fit_score(y[tr], bins[tr], bins[te], N_BINS)
        c_te = np.full(te.sum(), prior)          # constant predictor, train prevalence

        y_te = y[te]
        p_te = pcode[te]
        # compact patient codes on the holdout
        tp_uni, tp_code = np.unique(p_te, return_inverse=True)
        kt = len(tp_uni)
        ypat_te = ypat[tp_uni]
        depth_te = depth[tp_uni]

        res = {"seed": int(seed),
               "n_test_patients": int(kt), "n_test_slices": int(te.sum()),
               "n_train_patients": int(k - kt), "n_train_slices": int(tr.sum()),
               "train_prevalence": float(prior),
               "test_slice_prevalence": float(y_te.mean()),
               "test_patient_prevalence": float(ypat_te.mean()),
               "n_pos_test_slices": int(y_te.sum()),
               "n_pos_test_patients": int(ypat_te.sum())}

        # slice level
        Cp, Cn = cluster_count_matrix(y_te, s_te, tp_code, kt)
        slice_auc = auc_from_counts(Cp.sum(0), Cn.sum(0))
        sk = float(roc_auc_score(y_te, s_te))
        assert abs(slice_auc - sk) < 1e-9, (slice_auc, sk)
        res["slice_auc"] = float(slice_auc)

        # patient level, all operators
        aggs = patient_aggregates(tp_code, s_te, kt)
        res["patient_auc_mean"] = float(roc_auc_score(ypat_te, aggs["mean"]))
        res["aggregation"] = {a: float(roc_auc_score(ypat_te, v))
                              for a, v in aggs.items()}

        # patient level, stack depth held fixed
        sa_flat = StratifiedAUC(ypat_te, aggs["mean"], np.zeros(kt, dtype=int))
        sa_dep = StratifiedAUC(ypat_te, aggs["mean"], depth_te)
        sa_d5 = StratifiedAUC(ypat_te, aggs["mean"], depth_te // 5)
        v_flat, pr_flat = sa_flat.value()
        v_dep, pr_dep = sa_dep.value()
        v_d5, pr_d5 = sa_d5.value()
        assert abs(v_flat - res["patient_auc_mean"]) < 1e-9
        res["patient_auc_mean_depthfixed"] = float(v_dep)
        res["patient_auc_mean_depth5"] = float(v_d5)
        res["pairs_unstratified"] = float(pr_flat)
        res["pairs_depthfixed"] = float(pr_dep)
        res["pairs_depth5"] = float(pr_d5)
        res["n_depth_strata_informative"] = int(
            len({dd for dd in np.unique(depth_te)
                 if ypat_te[depth_te == dd].sum() not in (0, (depth_te == dd).sum())}))

        # depth alone, as a raw patient score
        res["depth_alone_patient_auc"] = float(roc_auc_score(ypat_te, depth_te))

        # constant predictor floor, identical path
        Ccp, Ccn = cluster_count_matrix(y_te, c_te, tp_code, kt)
        res["constant_slice_auc"] = float(auc_from_counts(Ccp.sum(0), Ccn.sum(0)))
        cagg = patient_aggregates(tp_code, c_te, kt)
        res["constant_patient_auc"] = float(
            StratifiedAUC(ypat_te, cagg["mean"], np.zeros(kt, dtype=int)).value()[0])

        if not full:
            return res

        # ------------- intervals, primary draw only -------------
        rng_b = np.random.default_rng(seed + 1)
        lo, hi, used = boot_ci_counts(Cp, Cn, rng_b, n_boot=N_BOOT)
        res["slice_ci"] = [lo, hi]
        res["slice_ci_reps"] = used

        rng_b = np.random.default_rng(seed + 2)
        lo, hi, used = sa_flat.boot_ci(rng_b, n_boot=N_BOOT)
        res["patient_auc_mean_ci"] = [lo, hi]
        res["patient_ci_reps"] = used

        rng_b = np.random.default_rng(seed + 3)
        lo, hi, _ = sa_dep.boot_ci(rng_b, n_boot=N_BOOT)
        res["patient_auc_mean_depthfixed_ci"] = [lo, hi]
        rng_b = np.random.default_rng(seed + 4)
        lo, hi, _ = sa_d5.boot_ci(rng_b, n_boot=N_BOOT)
        res["patient_auc_mean_depth5_ci"] = [lo, hi]
        rng_b = np.random.default_rng(seed + 5)
        lo, hi, _ = StratifiedAUC(ypat_te, depth_te.astype(float),
                                  np.zeros(kt, dtype=int)).boot_ci(rng_b, n_boot=N_BOOT)
        res["depth_alone_patient_ci"] = [lo, hi]

        res["aggregation_ci"] = {}
        for i, (a, v) in enumerate(aggs.items()):
            rb = np.random.default_rng(seed + 10 + i)
            sa = StratifiedAUC(ypat_te, v, np.zeros(kt, dtype=int))
            l2, h2, _ = sa.boot_ci(rb, n_boot=N_BOOT)
            res["aggregation_ci"][a] = [l2, h2]
        res["n_distinct_patient_scores"] = {
            a: int(len(np.unique(v))) for a, v in aggs.items()}

        # constant predictor interval (degenerate but reported, not assumed)
        rb = np.random.default_rng(seed + 40)
        l3, h3, _ = boot_ci_counts(Ccp, Ccn, rb, n_boot=N_BOOT)
        res["constant_slice_ci"] = [l3, h3]

        # ------------- within-series label permutation null -------------
        base = np.lexsort((np.arange(n), ser_code))
        ps, pp, pdep = [], [], []
        for r in range(N_PERM):
            prg = np.random.default_rng(seed + 100 + r)
            shuf = np.lexsort((prg.random(n), ser_code))
            yperm = np.empty(n, dtype=int)
            yperm[base] = y[shuf]
            assert yperm.sum() == y.sum()
            sp, _, _ = positional_fit_score(yperm[tr], bins[tr], bins[te], N_BINS)
            yp_te = yperm[te]
            ppat = (np.bincount(tp_code, weights=yp_te.astype(float),
                                minlength=kt) > 0).astype(int)
            if yp_te.min() == yp_te.max() or ppat.min() == ppat.max():
                continue
            ps.append(float(roc_auc_score(yp_te, sp)))
            ag = patient_aggregates(tp_code, sp, kt)
            pp.append(float(roc_auc_score(ppat, ag["mean"])))
            pdep.append(float(StratifiedAUC(ppat, ag["mean"], depth_te).value()[0]))
        res["permutation_null"] = {
            "n_perm": len(ps),
            "slice_mean": float(np.mean(ps)), "slice_range": [float(np.min(ps)), float(np.max(ps))],
            "patient_mean": float(np.mean(pp)), "patient_range": [float(np.min(pp)), float(np.max(pp))],
            "patient_depthfixed_mean": float(np.mean(pdep)),
            "patient_depthfixed_range": [float(np.min(pdep)), float(np.max(pdep))],
            "slice_excess_over_null": float(slice_auc - np.mean(ps)),
        }

        # ------------- secondary zero-image baselines (reported, not selected) ---
        res["secondary_baselines"] = secondary(tr, te, y, y_te, tp_code, kt,
                                               ypat_te, seed)

        if published is not None:
            res["published_slice_auc"] = published
            res["trivial_fraction_slice_locked"] = float(
                (slice_auc - 0.5) / (published - 0.5))
            res["trivial_fraction_slice_locked_ci"] = [
                float((res["slice_ci"][0] - 0.5) / (published - 0.5)),
                float((res["slice_ci"][1] - 0.5) / (published - 0.5))]
        return res

    # ------------------------------------------------- secondary baselines
    # The other four of the five zero-image baselines, taken from the audit tool's own
    # class definitions so they mean exactly what the manuscript says they mean
    # (depth-3 CART, smoothed target encoding), rather than a surrogate of my own.
    sys.path.insert(0, "/Users/sathvikloke/Downloads/PhaseDx/pipeline")
    import s14_trivialbaselines as s14                      # noqa: E402

    meta_cols = ["plane", "rescale_slope", "rescale_intercept"]
    dmeta = pd.read_csv(SRC, usecols=meta_cols)
    vol_size = pd.Series(np.ones(n)).groupby(ser_code).transform("size").to_numpy(float)

    def s14_frame(y):
        return pd.DataFrame({
            s14.C_SUBJ: pcode, s14.C_LABEL: y, s14.C_RELPOS: relpos,
            s14.C_NSLICES: vol_size,
            "plane": dmeta.plane.to_numpy(),
            "rescale_slope": dmeta.rescale_slope.to_numpy(),
            "rescale_intercept": dmeta.rescale_intercept.to_numpy(),
        })

    def secondary(tr, te, y, y_te, tp_code, kt, ypat_te, seed):
        frame = s14_frame(y)
        trf, tef = frame[tr], frame[te]
        out = {}
        bls = [s14.ColumnBaseline(s14.C_NSLICES),
               s14.TreeBaseline(meta_cols, name="metadata_tree", max_depth=3,
                                max_levels=30),
               s14.TreeBaseline(meta_cols, name="combined_position_metadata",
                                use_relpos=True, max_depth=3, max_levels=30)]
        bls[0].name = "volume_size"
        for c in meta_cols:
            cb = s14.ColumnBaseline(c)
            cb.name = f"column[{c}]"
            bls.append(cb)
        for bl in bls:
            s_x = bl.fit(trf).score(tef)
            Cp2, Cn2 = cluster_count_matrix(y_te, s_x, tp_code, kt)
            sl = auc_from_counts(Cp2.sum(0), Cn2.sum(0))
            ag = patient_aggregates(tp_code, s_x, kt)
            pa = StratifiedAUC(ypat_te, ag["mean"], np.zeros(kt, dtype=int)).value()[0]
            out[bl.name] = {"slice_auc": float(sl), "patient_auc_mean": float(pa),
                            "n_distinct_slice_scores": int(len(np.unique(snap(s_x))))}
        out["_note"] = ("secondary, reported for completeness; the primary baseline "
                        "was locked to positional_20bin before the holdout was touched. "
                        "Definitions imported from the audit tool: depth-3 CART for the "
                        "tree arms, smoothed target encoding for the column arms.")
        return out

    # ---------------------------------------------------------- run the family
    print(f"{n:,} slices | {len(ser_uni):,} series | {k:,} patients | sha {sha[:16]}")
    print(f"holdout {HOLDOUT_FRAC:.0%} of patients, {N_DRAWS} draws, "
          f"primary seed {PRIMARY_SEED}\n")

    report["labels"] = {}
    for col in LABELS:
        nm = PRETTY.get(col, col)
        y, ypat = Y[col], PY[col]
        draws = [evaluate(PRIMARY_SEED + i, y, ypat) for i in range(N_DRAWS)]
        primary = evaluate(PRIMARY_SEED, y, ypat, full=True, published=PUBLISHED[col])

        def spread(key):
            v = np.array([dd[key] for dd in draws], dtype=float)
            return {"mean": float(v.mean()), "sd": float(v.std(ddof=1)),
                    "min": float(v.min()), "max": float(v.max()),
                    "median": float(np.median(v))}

        report["labels"][nm] = {
            "primary_frozen_holdout": primary,
            "across_holdout_spread": {
                "n_draws": N_DRAWS,
                "seeds": [PRIMARY_SEED + i for i in range(N_DRAWS)],
                "slice_auc": spread("slice_auc"),
                "patient_auc_mean": spread("patient_auc_mean"),
                "patient_auc_mean_depthfixed": spread("patient_auc_mean_depthfixed"),
                "constant_slice_auc": spread("constant_slice_auc"),
                "depth_alone_patient_auc": spread("depth_alone_patient_auc"),
                "n_depth_strata_informative": spread("n_depth_strata_informative"),
                "constant_patient_auc": spread("constant_patient_auc"),
                "per_draw": [{kk: dd[kk] for kk in
                              ("seed", "slice_auc", "patient_auc_mean",
                               "patient_auc_mean_depthfixed", "constant_slice_auc",
                               "constant_patient_auc", "depth_alone_patient_auc",
                               "n_test_patients", "n_pos_test_patients")} for dd in draws],
            },
        }
        p = primary
        print(f"{nm:<18} slice {p['slice_auc']:.4f} "
              f"[{p['slice_ci'][0]:.4f},{p['slice_ci'][1]:.4f}] | "
              f"pat {p['patient_auc_mean']:.4f} "
              f"[{p['patient_auc_mean_ci'][0]:.4f},{p['patient_auc_mean_ci'][1]:.4f}] | "
              f"depthfix {p['patient_auc_mean_depthfixed']:.4f} | "
              f"const {p['constant_slice_auc']:.4f}/{p['constant_patient_auc']:.4f}")

    report["elapsed_s"] = round(time.time() - t0, 1)
    OUT.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT}  ({report['elapsed_s']}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
