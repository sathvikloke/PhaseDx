#!/usr/bin/env python3
"""Bin-robustness and fit-freeness checks for the frozen RSNA ICH holdout.

The manuscript's three "further checks" (bin sweep, fit-free variant, apparent
vs held-out) were originally measured under the pooled out-of-fold estimator
that the revision abandons. This recomputes them on the SAME frozen holdout the
revised primary numbers use -- seed 20260813, 30% of patients, one fit -- so
that nothing in the manuscript is carried over from the superseded estimator.

Shares its split, its binning and its AUROC code with rsna_frozen_holdout.py by
importing them, so a difference between this output and the primary numbers
cannot be a difference of implementation.

    venv/bin/python pipeline/audit_prep/frozen/rsna_bin_sweep.py
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aucutil import (auc_from_counts, cluster_count_matrix, positional_fit_score,  # noqa: E402
                     relpos_bins, slice_auc_ci, snap)

REPO = Path(__file__).resolve().parents[3]
SRC = REPO / "pipeline_out" / "audit_data" / "rsna_ich_slices.csv"
OUT = REPO / "pipeline_out" / "trivial_baselines" / "rsna_bin_sweep.json"

LABEL = "label"          # the 'any hemorrhage' column
HOLDOUT_FRAC = 0.30
PRIMARY_SEED = 20260813
SWEEP = [5, 10, 20, 50]
N_BOOT = 2000


def patient_mean(pcode, scores, k):
    n = np.bincount(pcode, minlength=k)
    s = np.bincount(pcode, weights=scores, minlength=k)
    return s / np.maximum(n, 1)


def main():
    t0 = time.time()
    h = hashlib.sha256()
    with open(SRC, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    sha = h.hexdigest()

    d = pd.read_csv(SRC, usecols=["patient_id", "series_id", "slice", LABEL])
    ser_code, ser_uni = pd.factorize(d.series_id.to_numpy())
    pcode, puni = pd.factorize(d.patient_id.to_numpy())
    k = len(puni)
    y = d[LABEL].to_numpy().astype(int)
    ypat = (np.bincount(pcode, weights=y.astype(float), minlength=k) > 0).astype(int)

    rng = np.random.default_rng(PRIMARY_SEED)
    perm = rng.permutation(k)
    test_p = np.zeros(k, dtype=bool)
    test_p[perm[: int(round(HOLDOUT_FRAC * k))]] = True
    te = test_p[pcode]
    tr = ~te
    y_te = y[te]
    tp_uni, tp_code = np.unique(pcode[te], return_inverse=True)
    kt = len(tp_uni)
    ypat_te = ypat[tp_uni]

    report = {
        "tool": "rsna_bin_sweep",
        "version": "1.0",
        "source_file": str(SRC),
        "source_sha256": sha,
        "estimator": "single frozen patient-disjoint holdout, one fit, no pooling",
        "primary_seed": PRIMARY_SEED,
        "holdout_fraction": HOLDOUT_FRAC,
        "label": "any hemorrhage",
        "n_test_patients": int(kt),
        "n_test_slices": int(te.sum()),
        "n_boot": N_BOOT,
        "NOT_FOR_SUBMISSION": "working artefact; contains absolute local paths",
        "sweep": {},
    }

    for nb in SWEEP:
        relpos, bins = relpos_bins(d.slice.to_numpy(), ser_code, nb)
        s_te, _, _ = positional_fit_score(y[tr], bins[tr], bins[te], nb)
        s_tr, _, _ = positional_fit_score(y[tr], bins[tr], bins[tr], nb)
        r = np.random.default_rng(PRIMARY_SEED + nb)
        pt, lo, hi, _ = slice_auc_ci(y_te, s_te, tp_code, kt, r, n_boot=N_BOOT)
        # apparent (training) slice AUC of the same fit
        trp_uni, trp_code = np.unique(pcode[tr], return_inverse=True)
        Cp, Cn = cluster_count_matrix(y[tr], s_tr, trp_code, len(trp_uni))
        apparent = auc_from_counts(Cp.sum(0), Cn.sum(0))
        pm = patient_mean(tp_code, s_te, kt)
        r2 = np.random.default_rng(PRIMARY_SEED + 1000 + nb)
        ppt, plo, phi, _ = slice_auc_ci(ypat_te, pm, np.arange(kt), kt, r2,
                                        n_boot=N_BOOT)
        report["sweep"][str(nb)] = {
            "slice_auc": float(pt), "slice_ci": [float(lo), float(hi)],
            "slice_auc_apparent_on_training_rows": float(apparent),
            "patient_auc_mean": float(ppt), "patient_ci": [float(plo), float(phi)],
        }
        print(f"  {nb:>2d} bins   slice {pt:.4f} [{lo:.4f}, {hi:.4f}]   "
              f"apparent {apparent:.4f}   patient {ppt:.4f} [{plo:.4f}, {phi:.4f}]")

    # fit-free centrality score: uses no training data at all
    relpos, _ = relpos_bins(d.slice.to_numpy(), ser_code, 20)
    cen = -np.abs(relpos - 0.5)
    r = np.random.default_rng(PRIMARY_SEED + 7)
    pt, lo, hi, _ = slice_auc_ci(y_te, cen[te], tp_code, kt, r, n_boot=N_BOOT)
    pm = patient_mean(tp_code, cen[te], kt)
    r2 = np.random.default_rng(PRIMARY_SEED + 8)
    ppt, plo, phi, _ = slice_auc_ci(ypat_te, pm, np.arange(kt), kt, r2, n_boot=N_BOOT)
    report["fit_free_centrality"] = {
        "definition": "-|relative position - 0.5|; uses no training data",
        "slice_auc": float(pt), "slice_ci": [float(lo), float(hi)],
        "patient_auc_mean": float(ppt), "patient_ci": [float(plo), float(phi)],
    }
    print(f"  fit-free  slice {pt:.4f} [{lo:.4f}, {hi:.4f}]   "
          f"patient {ppt:.4f} [{plo:.4f}, {phi:.4f}]")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    with OUT.open("w") as fh:
        json.dump(report, fh, indent=1)
    print(f"  wrote {OUT}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
