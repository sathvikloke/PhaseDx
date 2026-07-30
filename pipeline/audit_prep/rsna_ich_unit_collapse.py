"""Independent recomputation of the RSNA ICH slice-to-patient collapse.

WHY THIS FILE EXISTS
    The paper's flagship number -- a zero-image positional baseline that reaches a high
    SLICE-level AUROC and falls to or below chance at PATIENT level on the same score
    vector -- was first produced by pipeline/s14_trivialbaselines.py on a seeded
    1,500-patient subsample. A flagship number should not rest on one implementation or
    one subsample. This script recomputes it on ALL 18,938 patients with an
    implementation that shares no code with s14: its own fold assignment, its own
    binning, its own AUROC, its own clustered bootstrap. If the two disagree, one of
    them is wrong and the paper should say so.

WHAT IS COMPUTED
    One score vector per label, produced by a model that never sees a pixel:
        P(label | 20-bin relative position within the series),
    fitted on the training slices of a subject-disjoint 5-fold split and applied to the
    held-out slices. That one vector is then read at TWO evaluation units:
        slice   -- every slice is an independent observation (the unit RSNA ICH's own
                   official metric uses, and the unit the peer-reviewed comparator
                   reports)
        patient -- slice scores averaged within patient, one observation per patient
                   (the unit a deployed triage tool would be judged at)
    The gap between the two is the finding. It needs no published comparator: it is a
    statement about what the same predictions are worth under two reporting conventions.

    The max-aggregated patient reading is reported alongside the mean, because "take the
    most suspicious slice" is what a real triage system would do, and it is a different
    -- and for a positional baseline, near-degenerate -- number.

ESTIMATORS
    AUROC          sklearn.metrics.roc_auc_score, i.e. NOT the midrank implementation in
                   pipeline/s04_stats.py that s14 calls. Agreement between the two is
                   part of the check.
    interval       95% percentile bootstrap resampling PATIENTS with replacement, 2,000
                   replicates. A patient drawn twice contributes all of their slices
                   twice. Resampling slices would be wrong: slices within a patient are
                   not independent, and the naive interval is reported too so the size
                   of that error is visible rather than asserted.
    fast path      inside the bootstrap the AUROC is evaluated from per-patient count
                   vectors over the distinct score values rather than by re-ranking
                   750k rows. That is exact midrank AUROC, not an approximation; it is
                   asserted equal to the sklearn value on the observed data before any
                   resampling is done.

NULL AND PROTOCOL CHECKS
    constant predictor   scored through the identical path. Its AUROC is 0.5 by
                         arithmetic within one test set; pooled across folds whose
                         training prevalence differs it need not be, and the deviation
                         is reported rather than assumed away.
    label permutation    labels shuffled WITHIN each series, which destroys the
                         position-label association and preserves prevalence, subject
                         clustering, series depth and everything else. This is the
                         positional baseline's own null. It is not automatically 0.5.

Usage:
    python pipeline/audit_prep/rsna_ich_unit_collapse.py \
        pipeline_out/audit_data/rsna_ich_slices.csv \
        [--out pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json]
"""
from __future__ import annotations

import argparse
import datetime as _dt
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

LABELS = ["label", "epidural", "intraparenchymal", "intraventricular",
          "subarachnoid", "subdural"]
PRETTY = {"label": "any", "epidural": "epidural",
          "intraparenchymal": "intraparenchymal",
          "intraventricular": "intraventricular",
          "subarachnoid": "subarachnoid", "subdural": "subdural"}
# Burduja, Ionescu & Verga, Sensors 2020;20(19):5611, Table 3, ResNeXt-101 32x8d +
# bidirectional LSTM, slice-level ROC AUC. Peer reviewed.
PUBLISHED = {"label": 0.9843, "epidural": 0.9851, "intraparenchymal": 0.9927,
             "intraventricular": 0.9970, "subarachnoid": 0.9821, "subdural": 0.9682}

N_BINS = 20
N_FOLDS = 5
N_BOOT = 2000
SEED = 20260729          # deliberately not s14's seed=0
TIE_SNAP = 12            # decimals; ties must compare equal before ranking


# --------------------------------------------------------------------------
# AUROC from per-group counts. Exact midrank, no re-ranking.
# --------------------------------------------------------------------------

def auc_from_counts(pos: np.ndarray, neg: np.ndarray) -> float:
    """AUROC given, for each distinct score value in ASCENDING order, the number of
    positives and negatives holding it. Ties get half credit -- the midrank rule.

        AUC = sum_v pos_v * (neg_below_v + neg_v / 2) / (P * N)
    """
    P, N = pos.sum(), neg.sum()
    if P <= 0 or N <= 0:
        return float("nan")
    below = np.concatenate(([0.0], np.cumsum(neg)[:-1]))
    return float((pos * (below + neg / 2.0)).sum() / (P * N))


def group_counts(y: np.ndarray, s: np.ndarray, subj_code: np.ndarray, k: int):
    """Per-subject count vectors over the distinct values of s.

    Returns (C_pos, C_neg, n_values) with C_* of shape (k subjects, n_values), so a
    bootstrap replicate is one matrix-vector product instead of a 750k-row sort.
    """
    vals, inv = np.unique(np.round(s, TIE_SNAP), return_inverse=True)
    v = len(vals)
    flat = subj_code.astype(np.int64) * v + inv
    C_pos = np.bincount(flat, weights=(y == 1).astype(float),
                        minlength=k * v).reshape(k, v)
    C_neg = np.bincount(flat, weights=(y == 0).astype(float),
                        minlength=k * v).reshape(k, v)
    return C_pos, C_neg, v


def clustered_ci(C_pos, C_neg, rng, n_boot=N_BOOT, alpha=0.05):
    """95% percentile interval, resampling SUBJECTS with replacement."""
    k = C_pos.shape[0]
    vals = np.empty(n_boot)
    vals[:] = np.nan
    for b in range(n_boot):
        m = np.bincount(rng.integers(0, k, size=k), minlength=k).astype(float)
        vals[b] = auc_from_counts(m @ C_pos, m @ C_neg)
    good = vals[np.isfinite(vals)]
    if len(good) < 2:
        return (float("nan"), float("nan"), 0)
    return (float(np.percentile(good, 100 * alpha / 2)),
            float(np.percentile(good, 100 * (1 - alpha / 2))), int(len(good)))


def naive_ci(y, s, rng, n_boot=500, alpha=0.05):
    """The WRONG interval: resample slices as if they were independent. Reported so the
    understatement is visible."""
    n = len(y)
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        yy = y[idx]
        if yy.min() == yy.max():
            continue
        vals.append(roc_auc_score(yy, s[idx]))
    if len(vals) < 2:
        return (float("nan"), float("nan"))
    return (float(np.percentile(vals, 100 * alpha / 2)),
            float(np.percentile(vals, 100 * (1 - alpha / 2))))


# --------------------------------------------------------------------------
# The zero-image positional model
# --------------------------------------------------------------------------

def positional_oof(y: np.ndarray, bins: np.ndarray, fold: np.ndarray,
                   n_bins: int = N_BINS) -> np.ndarray:
    """Out-of-fold P(label | position bin). Fitted on train slices, applied to test."""
    out = np.empty(len(y), dtype=float)
    for f in range(fold.max() + 1):
        tr, te = fold != f, fold == f
        num = np.bincount(bins[tr], weights=y[tr], minlength=n_bins)
        den = np.bincount(bins[tr], minlength=n_bins)
        prior = y[tr].mean()
        rate = np.where(den > 0, num / np.maximum(den, 1), prior)
        out[te] = rate[bins[te]]
    return out


def constant_oof(y: np.ndarray, fold: np.ndarray) -> np.ndarray:
    """The constant predictor: each fold emits its own TRAINING prevalence. Pooled, that
    makes fold identity rankable, which is the protocol check."""
    out = np.empty(len(y), dtype=float)
    for f in range(fold.max() + 1):
        tr, te = fold != f, fold == f
        out[te] = y[tr].mean()
    return out


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("Usage:")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("labels")
    ap.add_argument("--out", default=None)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    ap.add_argument("--bins", type=int, default=N_BINS)
    ap.add_argument("--folds", type=int, default=N_FOLDS)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--n-perm", type=int, default=20)
    a = ap.parse_args(argv)

    src = Path(a.labels)
    h = hashlib.sha256()
    with open(src, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()[:16]

    d = pd.read_csv(src, usecols=["patient_id", "series_id", "slice"] + LABELS)
    n = len(d)

    # relative position within the series, derived here rather than imported
    ser = d.series_id.to_numpy()
    ser_code, ser_uni = pd.factorize(ser)
    slc = d.slice.to_numpy(float)
    lo = pd.Series(slc).groupby(ser_code).transform("min").to_numpy()
    hi = pd.Series(slc).groupby(ser_code).transform("max").to_numpy()
    span = np.where(hi > lo, hi - lo, 1.0)
    relpos = (slc - lo) / span
    edges = np.linspace(0.0, 1.0, a.bins + 1)
    edges[-1] += 1e-9
    bins = np.clip(np.digitize(relpos, edges) - 1, 0, a.bins - 1)

    subj = d.patient_id.to_numpy()
    subj_code, subj_uni = pd.factorize(subj)
    k = len(subj_uni)

    # subject-level folds, balanced on the patient-level 'any' label. Assignment scheme
    # and seed both differ from s14's; agreement must not depend on reusing its split.
    rng = np.random.default_rng(a.seed)
    pat_any = pd.Series(d.label.to_numpy()).groupby(subj_code).max().to_numpy()
    fold_of_subject = np.empty(k, dtype=int)
    for cls in (0, 1):
        idx = np.flatnonzero(pat_any == cls)
        idx = rng.permutation(idx)
        fold_of_subject[idx] = np.arange(len(idx)) % a.folds
    fold = fold_of_subject[subj_code]

    report = {
        "tool": "rsna_ich_unit_collapse",
        "version": "1.0",
        "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "labels_file": str(src),
        "labels_sha256": sha,
        "command": " ".join(sys.argv),
        "n_slices": int(n),
        "n_patients": int(k),
        "n_series": int(len(ser_uni)),
        "n_bins": int(a.bins),
        "n_folds": int(a.folds),
        "n_boot": int(a.n_boot),
        "seed": int(a.seed),
        "auroc_implementation": "sklearn.metrics.roc_auc_score (point estimate); "
                                "exact midrank from per-patient count vectors (bootstrap)",
        "interval": "95% percentile bootstrap over PATIENTS; a patient drawn twice "
                    "contributes all of their slices twice",
        "labels": {},
    }

    print(f"file: {n:,} slices | {len(ser_uni):,} series | {k:,} patients "
          f"| sha256 {sha}")
    print(f"{a.folds}-fold subject-disjoint CV, pooled out of fold | "
          f"{a.bins}-bin positional baseline | {a.n_boot} clustered bootstrap reps\n")

    hdr = (f"{'label':<18}{'slice prev':>11}{'pat prev':>10}"
           f"{'slice AUROC':>14}{'95% CI':>18}"
           f"{'patient AUROC':>15}{'95% CI':>18}{'collapse':>10}")
    print(hdr)
    print("-" * len(hdr))

    for col in LABELS:
        y = d[col].to_numpy().astype(int)
        s = positional_oof(y.astype(float), bins, fold, a.bins)
        c = constant_oof(y.astype(float), fold)

        # ---- slice level -------------------------------------------------
        slice_auc_sk = float(roc_auc_score(y, s))
        C_pos, C_neg, nv = group_counts(y, s, subj_code, k)
        slice_auc_ct = auc_from_counts(C_pos.sum(0), C_neg.sum(0))
        assert abs(slice_auc_sk - slice_auc_ct) < 1e-9, (
            f"{col}: sklearn {slice_auc_sk} != count-based {slice_auc_ct}")
        brng = np.random.default_rng(a.seed + 1)
        s_lo, s_hi, s_used = clustered_ci(C_pos, C_neg, brng, a.n_boot)
        nrng = np.random.default_rng(a.seed + 2)
        n_lo, n_hi = naive_ci(y, s, nrng, n_boot=min(500, a.n_boot))

        # ---- patient level, SAME score vector -----------------------------
        agg = pd.DataFrame({"p": subj_code, "y": y,
                            "s": np.round(s, TIE_SNAP)}).groupby("p").agg(
            y=("y", "max"), mean=("s", "mean"), max=("s", "max"))
        py = agg.y.to_numpy().astype(int)
        pm = np.round(agg["mean"].to_numpy(), TIE_SNAP)
        px = np.round(agg["max"].to_numpy(), TIE_SNAP)
        pat_auc_sk = float(roc_auc_score(py, pm))
        pat_max_sk = float(roc_auc_score(py, px))
        # one row per patient, so the clustered bootstrap IS the ordinary one
        Pp, Pn, _ = group_counts(py, pm, np.arange(len(py)), len(py))
        assert abs(auc_from_counts(Pp.sum(0), Pn.sum(0)) - pat_auc_sk) < 1e-9
        prng = np.random.default_rng(a.seed + 3)
        p_lo, p_hi, _ = clustered_ci(Pp, Pn, prng, a.n_boot)

        # ---- constant predictor, identical path ---------------------------
        const_slice = float(roc_auc_score(y, c))
        cagg = pd.DataFrame({"p": subj_code, "y": y,
                             "s": np.round(c, TIE_SNAP)}).groupby("p").agg(
            y=("y", "max"), mean=("s", "mean"))
        const_pat = float(roc_auc_score(cagg.y.to_numpy(), cagg["mean"].to_numpy()))

        # ---- the positional baseline's own null ---------------------------
        # base: rows ordered by series, original order within series.
        # shuf: rows ordered by series, RANDOM order within series.
        # yperm[base] = y[shuf] hands each slot in a series the label of a randomly
        # chosen slice of the SAME series. Prevalence, clustering, depth all survive.
        base = np.lexsort((np.arange(n), ser_code))
        perm_slice, perm_pat = [], []
        for r in range(a.n_perm):
            prg = np.random.default_rng(a.seed + 100 + r)
            shuf = np.lexsort((prg.random(n), ser_code))
            yperm = np.empty(n, dtype=int)
            yperm[base] = y[shuf]
            assert yperm.sum() == y.sum()
            sp = positional_oof(yperm.astype(float), bins, fold, a.bins)
            perm_slice.append(float(roc_auc_score(yperm, sp)))
            pa = pd.DataFrame({"p": subj_code, "y": yperm,
                               "s": np.round(sp, TIE_SNAP)}).groupby("p").agg(
                y=("y", "max"), mean=("s", "mean"))
            perm_pat.append(float(roc_auc_score(pa.y.to_numpy(),
                                                pa["mean"].to_numpy())))

        pub = PUBLISHED[col]
        tf = (slice_auc_sk - 0.5) / (pub - 0.5)
        tf_ci = [(s_lo - 0.5) / (pub - 0.5), (s_hi - 0.5) / (pub - 0.5)]
        tf_pat = (pat_auc_sk - 0.5) / (pub - 0.5)

        report["labels"][PRETTY[col]] = {
            "slice_prevalence": float(y.mean()),
            "patient_prevalence": float(py.mean()),
            "n_pos_slices": int(y.sum()), "n_pos_patients": int(py.sum()),
            "slice_auc": slice_auc_sk,
            "slice_ci_clustered": [s_lo, s_hi],
            "slice_ci_naive_WRONG": [n_lo, n_hi],
            "n_boot_used": s_used,
            "patient_auc_mean_agg": pat_auc_sk,
            "patient_ci_clustered": [p_lo, p_hi],
            "patient_auc_max_agg": pat_max_sk,
            "collapse_slice_minus_patient": slice_auc_sk - pat_auc_sk,
            "constant_predictor_slice_auc": const_slice,
            "constant_predictor_patient_auc": const_pat,
            "within_series_permutation_null": {
                "n_perm": int(a.n_perm),
                "slice_mean": float(np.mean(perm_slice)),
                "slice_range": [float(np.min(perm_slice)), float(np.max(perm_slice))],
                "patient_mean": float(np.mean(perm_pat)),
                "slice_excess_over_null": slice_auc_sk - float(np.mean(perm_slice)),
            },
            "published_slice_auc": pub,
            "published_source": "Burduja, Ionescu & Verga, Sensors 2020;20(19):5611, "
                                "Table 3, ResNeXt-101 32x8d + BiLSTM, slice ROC AUC",
            "trivial_fraction_slice": tf,
            "trivial_fraction_slice_ci": tf_ci,
            "trivial_fraction_patient": tf_pat,
        }

        print(f"{PRETTY[col]:<18}{y.mean():>11.4f}{py.mean():>10.4f}"
              f"{slice_auc_sk:>14.4f}  [{s_lo:.3f}, {s_hi:.3f}]"
              f"{pat_auc_sk:>15.4f}  [{p_lo:.3f}, {p_hi:.3f}]"
              f"{slice_auc_sk - pat_auc_sk:>10.3f}")

    print()
    print("collapse = slice AUROC - patient AUROC, ONE score vector read at two units.")
    print("naive (slice-resampled) intervals, which this file reports but does not use:")
    for nm, r in report["labels"].items():
        w_cl = r["slice_ci_clustered"][1] - r["slice_ci_clustered"][0]
        w_nv = r["slice_ci_naive_WRONG"][1] - r["slice_ci_naive_WRONG"][0]
        print(f"  {nm:<18} clustered width {w_cl:.4f} | naive width {w_nv:.4f} "
              f"| naive is {w_cl / w_nv:.2f}x too narrow")

    if a.out:
        outp = Path(a.out)
        outp.parent.mkdir(parents=True, exist_ok=True)
        outp.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {outp}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
