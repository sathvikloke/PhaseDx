"""
s08_belowchance.py
------------------
Why is the prostate_t2 phase classifier BELOW chance?

Pooled out-of-fold, patient-level, the phase model scores AUC ~0.38 on prostate_t2 --
systematically below 0.500 across both seeds, while magnitude sits at ~0.48. An AUC
reliably below chance is not "no signal": a coin lands on 0.5. It means the model's score
runs OPPOSITE to the label, which happens when the network locks onto something that is
genuinely learnable and happens to be anti-correlated with the label in this cohort.

The confound thesis predicts exactly that: the network learns an acquisition property
(coil count, scanner, protocol) because that is what phase encodes most strongly, and in
this cohort that property runs against PI-RADS. This module tests the prediction directly
rather than leaving it as an assertion in the discussion.

Three questions, in order:

  Q1  Is any acquisition property associated with the LABEL at subject level?
      If nothing is, the confound explanation is dead and the below-chance result needs a
      different account (overfitting to the validation carve, say).

  Q2  Do the model's PREDICTIONS track acquisition properties better than they track the
      label? This is the load-bearing test. If phase predictions are more predictable from
      the scanner than from the diagnosis, the network is a scanner detector.

  Q3  Does conditioning on the confound explain the below-chance AUC away -- i.e. is the
      AUC computed WITHIN a confound stratum closer to 0.5 than the pooled AUC?
      Simpson's-paradox logic: a within-stratum null plus a between-stratum association
      running the wrong way produces a pooled AUC below chance.

This is an explanatory analysis of an already-observed result, not a fishing expedition
for a positive finding. It cannot rescue the phase hypothesis and is not intended to: the
best possible outcome here is a clean mechanism for a negative result.

Usage:
    python pipeline/s08_belowchance.py --cohort prostate_t2 --condition phase
    python pipeline/s08_belowchance.py --cohort prostate_t2 --all-conditions
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from pipeline import common  # noqa: E402

# Columns that describe HOW the scan was acquired rather than what is in it. These are the
# candidate confounds. Outcome-derived columns (PIRADS, exam_level, grade, receptor status)
# are deliberately excluded -- their association with the label is tautological.
# eta^2 needs several subjects per level to mean anything; see assoc_with_prediction.
MIN_SUBJECTS_PER_LEVEL = 4

ACQ_COLS = [
    "scanner_model", "institution", "device_id", "receiver_channels", "n_coils",
    "protocol_name", "sequence_type", "patient_position", "TR", "TE", "flip_angle_deg",
    "echo_spacing", "kspace_shape", "matrix", "matrix_kx", "matrix_ky", "folder",
    "field_strength_T", "coil_array", "n_slices", "file_size_bytes", "attr_max", "attr_norm",
]


def auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Mann-Whitney AUC with correct mid-rank handling of ties."""
    labels = np.asarray(labels)
    scores = np.asarray(scores, dtype=float)
    pos, neg = scores[labels == 1], scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=float)
    ranks[order] = np.arange(1, len(scores) + 1)
    # Average ranks within tie groups, otherwise ties bias the statistic.
    s_sorted = scores[order]
    i = 0
    while i < len(s_sorted):
        j = i
        while j + 1 < len(s_sorted) and s_sorted[j + 1] == s_sorted[i]:
            j += 1
        if j > i:
            ranks[order[i:j + 1]] = np.mean(ranks[order[i:j + 1]])
        i = j + 1
    r_pos = ranks[labels == 1].sum()
    n_p, n_n = len(pos), len(neg)
    return float((r_pos - n_p * (n_p + 1) / 2) / (n_p * n_n))


def load_pooled(results_root: Path, cohort: str, condition: str, seed: int):
    """
    Concatenate the out-of-fold test blocks for one (cohort, condition, seed).

    Mirrors stage 4's pooling: every subject appears in exactly one test fold, so the
    concatenation is one prediction per subject over the whole cohort. Asserts that
    property rather than assuming it -- a duplicated subject would silently double-weight.
    """
    rows = []
    for d in sorted(results_root.glob(f"{cohort}_cv*")):
        f = d / f"{cohort}_{condition}_seed{seed}.json"
        if not f.exists():
            continue
        blk = json.load(open(f)).get("test", {})
        fold = int(d.name.rsplit("cv", 1)[1])
        for p, l, ci in zip(blk["probs"], blk["labels"], blk["cache_idx"]):
            rows.append({"prob": p, "label": l, "cache_idx": ci, "fold": fold})
    if not rows:
        return None
    df = pd.DataFrame(rows)
    if df["cache_idx"].duplicated().any():
        raise ValueError(
            f"{cohort}/{condition}/seed{seed}: a slice appears in more than one test "
            f"fold; the pool would double-weight it"
        )
    return df


def subject_level(df: pd.DataFrame, index: pd.DataFrame) -> pd.DataFrame:
    """Collapse slice predictions to one row per subject (mean score, max label)."""
    m = df.merge(index, left_on="cache_idx", right_on="idx", how="left",
                 suffixes=("", "_idx"))
    lab = "label" if "label" in m.columns else "label_idx"
    g = m.groupby("subject_id").agg(score=("prob", "mean"), label=(lab, "max"))
    return g.reset_index()


def assoc_with_label(sub: pd.DataFrame, meta: pd.DataFrame, col: str):
    """
    Association between one acquisition column and the LABEL, at subject level.

    Reported as an AUC so it is directly comparable with the classifier's own AUC: "this
    scanner property alone separates the classes this well".
    """
    d = sub.merge(meta[["subject_id", col]].drop_duplicates("subject_id"),
                  on="subject_id", how="left").dropna(subset=[col])
    if d[col].nunique() < 2 or d["label"].nunique() < 2:
        return None
    # Test for numeric explicitly rather than comparing dtype to object: pandas 3
    # stores strings in a dedicated 'str' dtype, so `dtype == object` is False for
    # text columns and the astype(float) below would raise on 'NYU Bay Ridge Radiology'.
    if pd.api.types.is_numeric_dtype(d[col]):
        codes = d[col].astype(float).to_numpy()
    else:
        codes = pd.factorize(d[col])[0].astype(float)
    a = auc(d["label"].to_numpy(), codes)
    # A categorical coded arbitrarily has no meaningful direction, so fold to >= 0.5 and
    # report the strength of association rather than a signed effect.
    return {"col": col, "n": int(len(d)), "n_levels": int(d[col].nunique()),
            "auc_vs_label": max(a, 1 - a)}


def assoc_with_prediction(sub: pd.DataFrame, meta: pd.DataFrame, col: str):
    """
    How well does an acquisition column explain the model's SCORE?

    Uses eta-squared: the fraction of score variance lying between levels of the column.
    Compared against the same quantity computed for the true label, this answers the
    question the paper actually needs: is the network tracking the scanner or the disease?
    """
    d = sub.merge(meta[["subject_id", col]].drop_duplicates("subject_id"),
                  on="subject_id", how="left").dropna(subset=[col])
    if d[col].nunique() < 2 or len(d) < 4:
        return None

    # eta^2 is meaningless once a column has nearly as many levels as there are
    # subjects: with 67 levels over 67 subjects every subject is its own stratum, the
    # between-group sum of squares equals the total, and eta^2 is 1.000 by arithmetic
    # rather than by association. attr_max and attr_norm (per-file floats) do exactly
    # this. Require several subjects per level so the statistic measures something.
    n_levels = int(d[col].nunique())
    if n_levels > max(2, len(d) // MIN_SUBJECTS_PER_LEVEL):
        return {"col": col, "n": int(len(d)), "n_levels": n_levels,
                "eta2_score": float("nan"),
                "skipped": f"{n_levels} levels over {len(d)} subjects is too granular "
                           f"for eta^2 (needs >= {MIN_SUBJECTS_PER_LEVEL}/level)"}

    grand = d["score"].mean()
    ss_tot = float(((d["score"] - grand) ** 2).sum())
    if ss_tot <= 0:
        return None
    ss_between = float(sum(
        len(g) * (g["score"].mean() - grand) ** 2 for _, g in d.groupby(col)
    ))
    return {"col": col, "n": int(len(d)), "n_levels": int(d[col].nunique()),
            "eta2_score": ss_between / ss_tot}


def within_stratum_auc(sub: pd.DataFrame, meta: pd.DataFrame, col: str):
    """
    Pooled AUC vs the sample-size-weighted mean of AUCs computed WITHIN each level.

    If the within-stratum mean sits near 0.5 while the pooled AUC is well below it, the
    below-chance result is a between-stratum artefact: the confound, not the anatomy, is
    driving the ranking.
    """
    d = sub.merge(meta[["subject_id", col]].drop_duplicates("subject_id"),
                  on="subject_id", how="left").dropna(subset=[col])
    if d[col].nunique() < 2:
        return None
    parts, weights = [], []
    for _, g in d.groupby(col):
        if g["label"].nunique() < 2 or len(g) < 4:
            continue
        parts.append(auc(g["label"].to_numpy(), g["score"].to_numpy()))
        weights.append(len(g))
    if not parts:
        return None
    return {"col": col, "n_strata_used": len(parts), "n_subjects": int(sum(weights)),
            "within_auc": float(np.average(parts, weights=weights)),
            "per_stratum": [round(p, 3) for p in parts]}


def analyse(cohort: str, condition: str, seeds, results_root: Path, top: int):
    cache_idx = pd.read_csv(common.CACHE_DIR / f"{cohort}_index.csv")
    cohort_csv = common.COHORT_DIR / f"{cohort}_cohort.csv"
    meta = pd.read_csv(cohort_csv) if cohort_csv.exists() else cache_idx
    if "subject_id" not in meta.columns:
        meta = cache_idx
    present = [c for c in ACQ_COLS if c in meta.columns
               and meta[c].notna().any() and meta[c].nunique(dropna=True) > 1]

    print("=" * 78)
    print(f"BELOW-CHANCE DIAGNOSIS  cohort={cohort}  condition={condition}")
    print("=" * 78)
    print(f"acquisition columns available with >1 level: {len(present)}")
    if not present:
        print("  none -- the confound explanation cannot be tested in this cohort")
        return

    for seed in seeds:
        pooled = load_pooled(results_root, cohort, condition, seed)
        if pooled is None:
            print(f"\nseed {seed}: no pooled folds on disk, skipping")
            continue
        sub = subject_level(pooled, cache_idx)
        pooled_auc = auc(sub["label"].to_numpy(), sub["score"].to_numpy())
        print(f"\n--- seed {seed}: {len(sub)} subjects, "
              f"{int(sub.label.sum())} positive, patient-level AUC = {pooled_auc:.3f} ---")

        # Q1
        rows = [r for r in (assoc_with_label(sub, meta, c) for c in present) if r]
        rows.sort(key=lambda r: -r["auc_vs_label"])
        print("\n  Q1. acquisition property vs LABEL (AUC, 0.5 = unrelated)")
        for r in rows[:top]:
            flag = "  <<< as strong as the classifier" if r["auc_vs_label"] >= abs(pooled_auc - 0.5) + 0.5 else ""
            print(f"      {r['col']:<22} AUC={r['auc_vs_label']:.3f}  "
                  f"({r['n_levels']} levels, n={r['n']}){flag}")

        # Q2 -- the load-bearing comparison
        allp = [r for r in (assoc_with_prediction(sub, meta, c) for c in present) if r]
        preds = [r for r in allp if not r.get("skipped")]
        skipped = [r for r in allp if r.get("skipped")]
        preds.sort(key=lambda r: -r["eta2_score"])
        # Baseline: how much of the score's variance the TRUE LABEL explains. Computed
        # inline rather than through assoc_with_prediction, which merges a metadata frame
        # onto `sub` and would collide on a column already present there.
        _grand = sub["score"].mean()
        _sst = float(((sub["score"] - _grand) ** 2).sum())
        lab_eta = None
        if _sst > 0:
            _ssb = float(sum(len(g) * (g["score"].mean() - _grand) ** 2
                             for _, g in sub.groupby("label")))
            lab_eta = {"eta2_score": _ssb / _sst}
        print("\n  Q2. what explains the MODEL'S SCORE? (eta^2, variance explained)")
        if lab_eta:
            print(f"      {'TRUE LABEL':<22} eta2={lab_eta['eta2_score']:.3f}   <-- the target")
        for r in preds[:top]:
            beat = "  <<< BEATS THE LABEL" if lab_eta and r["eta2_score"] > lab_eta["eta2_score"] else ""
            print(f"      {r['col']:<22} eta2={r['eta2_score']:.3f}  "
                  f"({r['n_levels']} levels){beat}")
        for r in skipped:
            print(f"      {r['col']:<22} skipped -- {r['skipped']}")

        # Q3
        print("\n  Q3. AUC WITHIN a confound stratum vs pooled "
              f"(pooled = {pooled_auc:.3f})")
        strat = [r for r in (within_stratum_auc(sub, meta, c) for c in present[:top]) if r]
        strat.sort(key=lambda r: -abs(r["within_auc"] - pooled_auc))
        for r in strat[:top]:
            moved = r["within_auc"] - pooled_auc
            note = "  <<< confound explains the below-chance result" if (
                pooled_auc < 0.5 and r["within_auc"] > pooled_auc + 0.05) else ""
            print(f"      within {r['col']:<16} AUC={r['within_auc']:.3f} "
                  f"({moved:+.3f}, {r['n_strata_used']} strata, "
                  f"n={r['n_subjects']}){note}")


def main():
    ap = argparse.ArgumentParser(
        description="Explain a below-chance classifier via acquisition confounds")
    ap.add_argument("--cohort", default="prostate_t2")
    ap.add_argument("--condition", default="phase")
    ap.add_argument("--all-conditions", action="store_true")
    ap.add_argument("--seeds", default="42,123")
    ap.add_argument("--results-dir", default=str(common.RESULTS_DIR))
    ap.add_argument("--top", type=int, default=8)
    args = ap.parse_args()

    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    conds = list(common.CONDITIONS) if args.all_conditions else [args.condition]
    for c in conds:
        analyse(args.cohort, c, seeds, Path(args.results_dir), args.top)
        print()


if __name__ == "__main__":
    main()
