#!/usr/bin/env python3
"""Frozen patient-disjoint holdout for a NON-flagship audited arm.

    venv/bin/python pipeline/audit_prep/frozen/frozen_arm_holdout.py \
        --src LABELS.csv --name luna16 --subject-col subject_id \
        --volume-col series_id --pos-col slice --label-col label

Why this exists
---------------
The revision replaced the pooled out-of-fold estimator with a single frozen
patient-disjoint holdout, because the pooled estimator gave a constant predictor
0.455-0.483 instead of 0.500 -- a ranking artefact of fold identity, not a
property of the data. Four arms were left on the superseded estimator because
their prepared label tables had been lost.

Two of those four were recovered: their source files are public and their
prepared tables were rebuilt and verified byte-identical against the SHA-256
recorded in the original run's output JSON. This script re-scores them under the
IDENTICAL protocol the flagship uses, so no arm in the primary analysis is
carried over from the superseded estimator.

Protocol, identical to rsna_frozen_holdout.py and not re-specified here:
    baseline     20-bin relative-position histogram P(label | position bin),
                 fitted on TRAINING subjects only
    holdout      30% of subjects, drawn uniformly, single fit, no pooling
    primary seed 20260813; family of 24 draws, seeds 20260813 + i
    aggregation  mean (pre-specified primary patient operator)
    interval     95% percentile bootstrap resampling SUBJECTS, 2000 replicates
    control      constant predictor, which must read exactly 0.500 at both units

An arm whose subject-level label is constant (Duke Breast: all 922 patients are
positive) has no defined patient-level AUC. That is reported as undefined, never
as a number, and never as 0.5.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from aucutil import (auc_from_counts, cluster_count_matrix,  # noqa: E402
                     positional_fit_score, relpos_bins, slice_auc_ci)

N_BINS = 20
HOLDOUT_FRAC = 0.30
PRIMARY_SEED = 20260813
N_DRAWS = 24
N_BOOT = 2000

REPO = Path(__file__).resolve().parents[3]
OUTDIR = REPO / "pipeline_out" / "trivial_baselines"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def subject_mean(scode: np.ndarray, scores: np.ndarray, k: int) -> np.ndarray:
    n = np.bincount(scode, minlength=k)
    s = np.bincount(scode, weights=scores, minlength=k)
    return s / np.maximum(n, 1)


def plain_auc(y: np.ndarray, s: np.ndarray) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    C = cluster_count_matrix(np.asarray(y, dtype=int), s, np.arange(len(y)), len(y))
    return float(auc_from_counts(C[0].sum(0), C[1].sum(0)))


def one_draw(seed, y, bins, scode, vol_code, k, y_subj, n_boot):
    """One frozen holdout. Returns slice/subject readings and both controls."""
    rng = np.random.default_rng(seed)
    perm = rng.permutation(k)
    test_s = np.zeros(k, dtype=bool)
    test_s[perm[: int(round(HOLDOUT_FRAC * k))]] = True
    te, tr = test_s[scode], ~test_s[scode]

    s_te, _, _ = positional_fit_score(y[tr], bins[tr], bins[te], N_BINS)
    ts_uni, ts_code = np.unique(scode[te], return_inverse=True)
    kt = len(ts_uni)
    y_subj_te = y_subj[ts_uni]

    r = np.random.default_rng(seed + 1)
    sl_pt, sl_lo, sl_hi, _ = slice_auc_ci(y[te], s_te, ts_code, kt, r, n_boot=n_boot)

    agg = subject_mean(ts_code, s_te, kt)
    if len(np.unique(y_subj_te)) < 2:
        su_pt = su_lo = su_hi = float("nan")
    else:
        r2 = np.random.default_rng(seed + 2)
        su_pt, su_lo, su_hi, _ = slice_auc_ci(y_subj_te, agg, np.arange(kt), kt, r2,
                                              n_boot=n_boot)

    const = np.zeros(int(te.sum()), dtype=float)
    c_slice = plain_auc(y[te], const)
    c_subj = plain_auc(y_subj_te, np.zeros(kt)) if len(np.unique(y_subj_te)) > 1 \
        else float("nan")

    return dict(seed=int(seed), n_test_subjects=int(kt), n_test_rows=int(te.sum()),
                slice_auc=sl_pt, slice_ci=[sl_lo, sl_hi],
                subject_auc_mean=su_pt, subject_ci=[su_lo, su_hi],
                constant_slice=c_slice, constant_subject=c_subj)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", required=True)
    ap.add_argument("--name", required=True)
    ap.add_argument("--subject-col", required=True)
    ap.add_argument("--volume-col", required=True)
    ap.add_argument("--pos-col", required=True)
    ap.add_argument("--label-col", default="label")
    ap.add_argument("--expect-sha16", default=None,
                    help="16-hex prefix recorded by the original run; verified, not trusted")
    a = ap.parse_args()

    t0 = time.time()
    src = Path(a.src)
    full_sha = sha256(src)
    sha_ok = None
    if a.expect_sha16:
        sha_ok = full_sha.startswith(a.expect_sha16.lower())
        print(f"  source sha256 {full_sha[:16]}  expected {a.expect_sha16}  "
              f"{'MATCH' if sha_ok else 'MISMATCH'}")
        if not sha_ok:
            raise SystemExit("refusing to score an arm whose label table does not "
                             "reproduce the recorded hash")

    cols = {a.subject_col, a.volume_col, a.pos_col, a.label_col}
    d = pd.read_csv(src, usecols=sorted(cols))
    scode, suni = pd.factorize(d[a.subject_col].to_numpy())
    vol_code, _ = pd.factorize(d[a.volume_col].to_numpy())
    k = len(suni)
    y = d[a.label_col].to_numpy().astype(int)
    y_subj = (np.bincount(scode, weights=y.astype(float), minlength=k) > 0).astype(int)
    subject_label_constant = len(np.unique(y_subj)) < 2

    _, bins = relpos_bins(d[a.pos_col].to_numpy(), vol_code, N_BINS)

    primary = one_draw(PRIMARY_SEED, y, bins, scode, vol_code, k, y_subj, N_BOOT)
    print(f"  frozen  slice {primary['slice_auc']:.4f} "
          f"[{primary['slice_ci'][0]:.4f}, {primary['slice_ci'][1]:.4f}]   "
          f"subject {primary['subject_auc_mean']:.4f}   "
          f"const {primary['constant_slice']:.4f}")

    fam = [one_draw(PRIMARY_SEED + i, y, bins, scode, vol_code, k, y_subj, 200)
           for i in range(N_DRAWS)]
    sl = np.array([f["slice_auc"] for f in fam], dtype=float)
    su = np.array([f["subject_auc_mean"] for f in fam], dtype=float)
    cs = np.array([f["constant_slice"] for f in fam], dtype=float)

    report = {
        "tool": "frozen_arm_holdout",
        "version": "1.0",
        "arm": a.name,
        "source_file": src.name,
        "source_sha256": full_sha,
        "source_sha256_matches_original_run": sha_ok,
        "estimator": "single frozen subject-disjoint holdout, one fit, no pooling",
        "supersedes": "pooled out-of-fold, whose constant predictor was not 0.500",
        "n_bins": N_BINS, "holdout_fraction": HOLDOUT_FRAC,
        "primary_seed": PRIMARY_SEED, "n_draws": N_DRAWS, "n_boot": N_BOOT,
        "n_subjects": int(k), "n_rows": int(len(d)),
        "slice_prevalence": float(y.mean()),
        "subject_label_constant": bool(subject_label_constant),
        "subject_auc_status": ("undefined: every subject carries the same label"
                               if subject_label_constant else "defined"),
        "NOT_FOR_SUBMISSION": "working artefact; contains absolute local paths",
        "frozen_holdout": primary,
        "family": {
            "n": N_DRAWS,
            "slice_mean": float(sl.mean()), "slice_sd": float(sl.std(ddof=1)),
            "slice_range": [float(sl.min()), float(sl.max())],
            "subject_mean": (None if subject_label_constant else float(np.nanmean(su))),
            "subject_sd": (None if subject_label_constant
                           else float(np.nanstd(su, ddof=1))),
            "subject_range": (None if subject_label_constant
                              else [float(np.nanmin(su)), float(np.nanmax(su))]),
            "constant_slice_all_exactly_half": bool(np.allclose(cs, 0.5, atol=1e-12)),
            "constant_slice_max_dev": float(np.abs(cs - 0.5).max()),
            "draws": fam,
        },
    }
    out = OUTDIR / f"frozen_{a.name}.json"
    out.write_text(json.dumps(report, indent=1))
    print(f"  family  slice {sl.mean():.4f} (SD {sl.std(ddof=1):.4f}, "
          f"range {sl.min():.4f}-{sl.max():.4f})")
    print(f"  control constant predictor max |dev| from 0.500 = "
          f"{np.abs(cs - 0.5).max():.2e}")
    print(f"  wrote {out}  ({time.time() - t0:.1f}s)")


if __name__ == "__main__":
    main()
