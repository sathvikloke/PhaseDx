#!/usr/bin/env python3
"""
Bin count x patient-aggregation grid on the frozen holdout.

    venv/bin/python pipeline/audit_prep/frozen/rsna_bin_agg_grid.py

Why this exists
---------------
The manuscript pre-specifies 20 bins and mean aggregation. Two reviewers
independently observed that the SLICE-level reading is nearly invariant to the
bin count while the PATIENT-level reading is not, and that the patient reading
also depends on the aggregation operator. Reporting either sensitivity alone
understates the joint dependence, so this builds the full grid and the figure
that goes with it.

It imports the primary script's own helpers rather than reimplementing them.
That matters: an independent reimplementation of the split disagreed with the
recorded 20-bin patient value by 0.016, because this pipeline factorizes
patient identifiers in FILE ORDER while the obvious reimplementation sorts
them, so the two draw different holdouts from the same seed. The grid must sit
on the same holdout as the headline or its 20-bin/mean cell will not reproduce
the number the abstract prints.

Emits pipeline_out/trivial_baselines/rsna_bin_agg_grid.json and
paper/tex/rsna/figures/fig4_bin_agg_grid.pdf.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from rsna_bin_sweep import (  # noqa: E402
    SRC, LABEL, PRIMARY_SEED, HOLDOUT_FRAC,
    relpos_bins, positional_fit_score, patient_mean,
)
from aucutil import cluster_count_matrix, auc_from_counts  # noqa: E402

REPO = HERE.parents[2]
OUT_JSON = REPO / "pipeline_out" / "trivial_baselines" / "rsna_bin_agg_grid.json"
OUT_PDF = REPO / "paper" / "tex" / "rsna" / "figures" / "fig4_bin_agg_grid.pdf"

BINS = [5, 10, 20, 30, 50]
AGGS = ["mean", "max", "top-3 mean", "top-5 mean", "75th pct", "90th pct"]


def aggregate(pcode: np.ndarray, scores: np.ndarray, k: int, how: str) -> np.ndarray:
    """Per-patient aggregation. `mean` uses the primary script's own bincount
    implementation so the grid's mean row is bit-identical to the headline."""
    if how == "mean":
        return patient_mean(pcode, scores, k)
    out = np.empty(k, dtype=float)
    order = np.argsort(pcode, kind="mergesort")
    pc, sc = pcode[order], scores[order]
    bounds = np.searchsorted(pc, np.arange(k + 1))
    for i in range(k):
        v = np.sort(sc[bounds[i]:bounds[i + 1]])
        if how == "max":
            out[i] = v[-1]
        elif how == "top-3 mean":
            out[i] = v[-3:].mean()
        elif how == "top-5 mean":
            out[i] = v[-5:].mean()
        elif how == "75th pct":
            out[i] = np.quantile(v, 0.75)
        elif how == "90th pct":
            out[i] = np.quantile(v, 0.90)
        else:
            raise ValueError(how)
    return out


def plain_auc(y: np.ndarray, s: np.ndarray) -> float:
    C = cluster_count_matrix(y.astype(int), s, np.arange(len(y)), len(y))
    return float(auc_from_counts(C[0].sum(0), C[1].sum(0)))


def main() -> None:
    t0 = time.time()
    d = pd.read_csv(SRC, usecols=["patient_id", "series_id", "slice", LABEL])
    ser_code, _ = pd.factorize(d.series_id.to_numpy())
    pcode, puni = pd.factorize(d.patient_id.to_numpy())      # FILE ORDER, not sorted
    k = len(puni)
    y = d[LABEL].to_numpy().astype(int)
    ypat = (np.bincount(pcode, weights=y.astype(float), minlength=k) > 0).astype(int)

    rng = np.random.default_rng(PRIMARY_SEED)
    perm = rng.permutation(k)
    test_p = np.zeros(k, dtype=bool)
    test_p[perm[: int(round(HOLDOUT_FRAC * k))]] = True
    te, tr = test_p[pcode], ~test_p[pcode]
    tp_uni, tp_code = np.unique(pcode[te], return_inverse=True)
    kt = len(tp_uni)
    ypat_te = ypat[tp_uni]

    grid: dict[str, float] = {}
    slice_auc: dict[str, float] = {}
    distinct: dict[str, int] = {}

    for nb in BINS:
        _, bins = relpos_bins(d.slice.to_numpy(), ser_code, nb)
        s_te, _, _ = positional_fit_score(y[tr], bins[tr], bins[te], nb)
        slice_auc[str(nb)] = plain_auc(y[te], s_te)
        for a in AGGS:
            v = aggregate(tp_code, s_te, kt, a)
            grid[f"{nb}|{a}"] = plain_auc(ypat_te, v)
            distinct[f"{nb}|{a}"] = int(len(np.unique(v)))
        print(f"  {nb:>2d} bins  slice {slice_auc[str(nb)]:.4f}  " +
              "  ".join(f"{a} {grid[f'{nb}|{a}']:.4f}" for a in AGGS))

    report = {
        "tool": "rsna_bin_agg_grid",
        "version": "1.0",
        "source_file": str(SRC),
        "estimator": "single frozen patient-disjoint holdout, one fit, no pooling",
        "primary_seed": PRIMARY_SEED,
        "holdout_fraction": HOLDOUT_FRAC,
        "n_test_patients": int(kt),
        "pre_specified_cell": "20 bins, mean",
        "NOT_FOR_SUBMISSION": "working artefact; contains absolute local paths",
        "bins": BINS, "aggregations": AGGS,
        "slice_auc": slice_auc, "patient_auc": grid,
        "distinct_patient_scores": distinct,
    }
    OUT_JSON.write_text(json.dumps(report, indent=1))
    print(f"\nwrote {OUT_JSON}")

    # ---------------- figure ----------------
    import matplotlib
    matplotlib.use("Agg")
    matplotlib.rcParams.update({
        "pdf.fonttype": 42, "ps.fonttype": 42,
        "font.family": "DejaVu Sans", "font.size": 7.4,
    })
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    M = np.array([[grid[f"{nb}|{a}"] for a in AGGS] for nb in BINS])
    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(7.1, 2.65), gridspec_kw={"width_ratios": [1, 3.05]})

    axL.plot(range(len(BINS)), [slice_auc[str(b)] for b in BINS],
             "o-", color="#0072B2", lw=1.4, ms=4.2)
    axL.axhline(0.5, color="0.45", lw=0.8, ls=":")
    axL.set_xticks(range(len(BINS))); axL.set_xticklabels(BINS)
    axL.set_ylim(0.42, 0.78); axL.set_xlabel("bins"); axL.set_ylabel("slice-level AUC")
    axL.set_title("A   slice unit: flat", loc="left", fontsize=8)
    for sp in ("top", "right"): axL.spines[sp].set_visible(False)

    norm = TwoSlopeNorm(vmin=min(0.42, M.min()), vcenter=0.5, vmax=max(0.66, M.max()))
    im = axR.imshow(M, cmap="RdBu_r", norm=norm, aspect="auto")
    axR.set_xticks(range(len(AGGS))); axR.set_xticklabels(AGGS, rotation=20, ha="right")
    axR.set_yticks(range(len(BINS))); axR.set_yticklabels(BINS)
    axR.set_ylabel("bins"); axR.set_xlabel("patient aggregation operator")
    axR.set_title("B   patient unit: not flat", loc="left", fontsize=8)
    for i in range(len(BINS)):
        for j in range(len(AGGS)):
            v = M[i, j]
            axR.text(j, i, f"{v:.3f}", ha="center", va="center", fontsize=6.8,
                     color="white" if abs(v - 0.5) > 0.085 else "black")
    pre_i, pre_j = BINS.index(20), AGGS.index("mean")
    axR.add_patch(plt.Rectangle((pre_j - .5, pre_i - .5), 1, 1,
                                fill=False, ec="black", lw=2.0))
    cb = fig.colorbar(im, ax=axR, fraction=0.035, pad=0.02)
    cb.set_label("patient-level AUC", fontsize=7)
    cb.ax.axhline(0.5, color="0.2", lw=0.9)
    fig.tight_layout()
    fig.savefig(OUT_PDF)
    print(f"wrote {OUT_PDF}   ({time.time()-t0:.1f}s)")


if __name__ == "__main__":
    main()
