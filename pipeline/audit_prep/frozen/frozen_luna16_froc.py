#!/usr/bin/env python3
"""LUNA16 on its OWN metric, re-scored on the frozen holdout.

    venv/bin/python pipeline/audit_prep/frozen/frozen_luna16_froc.py LABELS.csv

AUROC is not the LUNA16 scale. The false-positive-reduction track is scored by
the competition performance metric (CPM): mean sensitivity at 1/8, 1/4, 1/2, 1,
2, 4 and 8 false positives per scan. The manuscript's cross-study comparison uses
sensitivity at 1 FP/scan, which is the point on that curve the published
comparator reports.

The original run computed this pooled out-of-fold over 5 scan-disjoint folds --
the estimator the revision abandons. This recomputes it under the frozen
protocol: one 30% scan-disjoint holdout at seed 20260813, a single 20-bin fit on
the remaining 70%, no pooling. It shares its binning and its fit with
frozen_arm_holdout.py by importing them, so a difference between this output and
that one cannot be a difference of implementation.

The chance reference is the random-score value of the same metric on the same
held-out scans: a random ranking has sensitivity equal to the false-positive
fraction it has spent, so its sensitivity at f FP/scan is f * n_scans / n_neg.
That reference replaces 0.5, which is the AUROC anchor and not this metric's.
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
from aucutil import positional_fit_score, relpos_bins  # noqa: E402

N_BINS = 20
HOLDOUT_FRAC = 0.30
PRIMARY_SEED = 20260813
N_DRAWS = 24
FP_POINTS = (0.125, 0.25, 0.5, 1, 2, 4, 8)
PUBLISHED_SENS_AT_1FP = 0.95          # Setio et al, Med Image Anal 2017;42:1-13

REPO = Path(__file__).resolve().parents[3]
OUT = REPO / "pipeline_out" / "trivial_baselines" / "frozen_luna16_froc.json"


def froc(y: np.ndarray, s: np.ndarray, n_scans: int):
    """Sensitivity at each FP/scan operating point, plus the CPM."""
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    n_pos = int(y.sum())
    sens = []
    for f in FP_POINTS:
        i = np.searchsorted(fp, f * n_scans, side="right") - 1
        sens.append(float(tp[i] / n_pos) if i >= 0 and n_pos else 0.0)
    return sens, float(np.mean(sens))


def chance_froc(n_scans: int, n_neg: int):
    """Random-score value of the same metric on the same held-out scans."""
    ref = [min(1.0, f * n_scans / max(n_neg, 1)) for f in FP_POINTS]
    return ref, float(np.mean(ref))


def one_draw(seed, y, bins, scan_code, k):
    rng = np.random.default_rng(seed)
    perm = rng.permutation(k)
    test_s = np.zeros(k, dtype=bool)
    test_s[perm[: int(round(HOLDOUT_FRAC * k))]] = True
    te, tr = test_s[scan_code], ~test_s[scan_code]
    s_te, _, _ = positional_fit_score(y[tr], bins[tr], bins[te], N_BINS)
    y_te = y[te]
    n_scans_te = int(test_s.sum())
    sens, cpm = froc(y_te, s_te, n_scans_te)
    ref, ref_cpm = chance_froc(n_scans_te, int((y_te == 0).sum()))
    at1 = sens[FP_POINTS.index(1)]
    ref1 = ref[FP_POINTS.index(1)]
    frac = (at1 - ref1) / (PUBLISHED_SENS_AT_1FP - ref1)
    return dict(seed=int(seed), n_test_scans=n_scans_te,
                n_test_candidates=int(te.sum()), n_test_nodules=int(y_te.sum()),
                sensitivity=sens, cpm=cpm,
                chance_sensitivity=ref, chance_cpm=ref_cpm,
                sensitivity_at_1fp=at1, chance_at_1fp=ref1,
                margin_fraction_at_1fp=frac)


def main() -> None:
    t0 = time.time()
    src = Path(sys.argv[1])
    h = hashlib.sha256()
    with open(src, "rb") as f:
        for c in iter(lambda: f.read(1 << 20), b""):
            h.update(c)

    d = pd.read_csv(src, usecols=["series_id", "z_mm", "label"])
    scan_code, scans = pd.factorize(d.series_id.to_numpy())
    k = len(scans)
    y = d.label.to_numpy().astype(int)
    _, bins = relpos_bins(d.z_mm.to_numpy(), scan_code, N_BINS)

    primary = one_draw(PRIMARY_SEED, y, bins, scan_code, k)
    fam = [one_draw(PRIMARY_SEED + i, y, bins, scan_code, k) for i in range(N_DRAWS)]
    f1 = np.array([f["sensitivity_at_1fp"] for f in fam])
    fr = np.array([f["margin_fraction_at_1fp"] for f in fam])

    report = {
        "tool": "frozen_luna16_froc",
        "version": "1.0",
        "arm": "LUNA16 false-positive-reduction track",
        "metric": "sensitivity at 1 false positive per scan",
        "source_file": src.name,
        "source_sha256": h.hexdigest(),
        "estimator": "single frozen scan-disjoint holdout, one fit, no pooling",
        "supersedes": "pooled out-of-fold over 5 scan-disjoint folds",
        "published_comparator": "Setio et al, Med Image Anal 2017;42:1-13",
        "published_sensitivity_at_1fp": PUBLISHED_SENS_AT_1FP,
        "n_scans_total": int(k), "n_candidates_total": int(len(d)),
        "n_nodules_total": int(y.sum()),
        "NOT_FOR_SUBMISSION": "working artefact; contains absolute local paths",
        "frozen_holdout": primary,
        "family": {
            "n": N_DRAWS,
            "sensitivity_at_1fp_mean": float(f1.mean()),
            "sensitivity_at_1fp_range": [float(f1.min()), float(f1.max())],
            "margin_fraction_mean": float(fr.mean()),
            "margin_fraction_range": [float(fr.min()), float(fr.max())],
            "draws": fam,
        },
    }
    OUT.write_text(json.dumps(report, indent=1))
    print(f"  frozen holdout: {primary['n_test_scans']} scans, "
          f"{primary['n_test_candidates']} candidates, "
          f"{primary['n_test_nodules']} nodules")
    for f, s, r in zip(FP_POINTS, primary["sensitivity"], primary["chance_sensitivity"]):
        print(f"    sens @ {f:>5} FP/scan : {s:.4f}   (chance {r:.4f})")
    print(f"    CPM                   : {primary['cpm']:.4f} "
          f"(chance {primary['chance_cpm']:.4f})")
    print(f"    sens @ 1 FP/scan      : {primary['sensitivity_at_1fp']:.4f}, "
          f"chance {primary['chance_at_1fp']:.4f}, "
          f"margin fraction {primary['margin_fraction_at_1fp']:+.4f}")
    print(f"    family mean fraction  : {fr.mean():+.4f} "
          f"[{fr.min():+.4f}, {fr.max():+.4f}]")
    print(f"  wrote {OUT}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
