#!/usr/bin/env python3
"""
s09_robustness.py
=================
Two hostile-reviewer attacks on the confound paper, answered with numbers from
data that is already on this drive. Neither analysis can rescue a null and
neither is intended to: the best available outcome for A is "the null survives
every aggregation we tried", and the best available outcome for B is "the
mechanism claim survives stratification". Both are reported as they land.

--------------------------------------------------------------------------
A. DOES PATIENT AGGREGATION DILUTE FOCAL LESIONS?
--------------------------------------------------------------------------
C1 -- the criterion that decides every verdict -- is read at `patient_mean`:
one score per subject, obtained by AVERAGING that subject's slice scores. On
prostate_t2 roughly 7.5% of slices are positive, so a subject with ~30 slices
of which 2 contain tumour has 28 slices of pure noise voting against the 2 that
carry the lesion. If the classifier were finding focal disease, the mean would
bury it and `max` would not.

That is a real objection and it is answered here by recomputing the SAME
pooled out-of-fold, subject-clustered estimate under eight aggregation rules:

    mean, max, top-1 mean, top-2 mean, top-3 mean, top-5 mean, q75, q90

(top-1 mean IS max -- it is kept because the reviewer asked for k=1, it is
proved identical in --self-test, and it is excluded from the multiplicity count
so the "number of looks" is not inflated by an alias.)

Three things make this a test rather than a fishing trip:

  1. EVERY scheme is reported, not the best one. Picking the aggregation that
     helps after seeing the numbers is exactly the practice this paper exists
     to criticise.
  2. All eight schemes share ONE set of bootstrap resamples. The patient labels
     do not depend on the aggregation, so a replicate that is single-class is
     single-class for all schemes; the intervals are therefore directly
     comparable rather than eight independently-noisy things.
  3. A SELECTION-AWARE ENVELOPE is computed: in each replicate take the best
     AUC across schemes, then take the 2.5th percentile of that. Because the
     per-replicate max dominates every individual scheme replicate-by-replicate,
     its lower quantile is >= every individual scheme's lower bound. So if the
     envelope's lower bound sits below 0.500, then NO aggregation -- including
     one chosen after the fact -- can put a lower bound above chance. That is
     the statement a reviewer actually needs, and it is stronger than eight
     separate intervals.

--------------------------------------------------------------------------
B. IS "COIL COUNT" REALLY COIL COUNT, OR IS IT SITE?
--------------------------------------------------------------------------
The mechanism claim rests on the brain confound cohort: a network reading only
phase predicts receive-coil-count >= 16 at patient-level AUC ~0.92. If coil
count were simply a relabelling of site, the honest claim would be weaker --
"phase encodes site", not "phase encodes hardware".

The brain cohort's `institution` column has eight levels ('NYU',
'NYU LANGONE CBI', 'NYU LANGONE 32ND ST', 'TH', 'TH RADIOLOGY', ...). Whether
those are eight sites or eight spellings is not a judgement call: the cache
index also carries `device_id`, the scanner's serial. If one physical magnet is
recorded under two institution strings, those strings are the same site,
full stop. So sites are recovered as CONNECTED COMPONENTS of the
institution <-> device_id bipartite graph rather than by string heuristics, and
a first-token prefix rule is reported alongside it so the conclusion can be
checked against a normalisation that used no data at all.

Then, over the 454 cached brain subjects:
  * contingency of coil bucket against site, raw institution, scanner model,
    device id, field strength and coil array;
  * Cramer's V (raw and Bergsma bias-corrected) and mutual information, each
    with a PERMUTATION null, because V and MI are both biased upward on sparse
    tables and an uncorrected V of 0.9 on a 6x2 table proves nothing on its own;
  * separability: does any stratum contain BOTH coil buckets with enough
    subjects in each to test coil count inside it?
  * where the answer is yes, the within-stratum AUC of the actual phase model,
    on its actual held-out test subjects, with a subject bootstrap -- plus a
    pair-weighted stratified estimate that conditions on the stratum entirely.

Usage
-----
    python pipeline/s09_robustness.py --self-test
    python pipeline/s09_robustness.py                       # both analyses
    python pipeline/s09_robustness.py --only aggregation
    python pipeline/s09_robustness.py --only coil-vs-site
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s04_stats as s04  # noqa: E402  -- pooling, cluster maps and AUC live there

try:
    import common  # noqa: E402
    _DEFAULT_RESULTS_DIR = common.RESULTS_DIR
    _DEFAULT_CACHE_DIR = common.CACHE_DIR
    _DEFAULT_COHORT_DIR = common.OUT_ROOT / "cohorts"
    _DEFAULT_OUT_DIR = common.OUT_ROOT / "robustness"
except Exception:                                    # pragma: no cover
    _ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_RESULTS_DIR = _ROOT / "pipeline_out" / "results"
    _DEFAULT_CACHE_DIR = _ROOT / "pipeline_out" / "cache"
    _DEFAULT_COHORT_DIR = _ROOT / "pipeline_out" / "cohorts"
    _DEFAULT_OUT_DIR = _ROOT / "pipeline_out" / "robustness"

CHANCE = 0.500


# ==========================================================================
# Part A primitives: aggregation schemes
# ==========================================================================

# name -> (kind, param, counts_toward_multiplicity)
#
# `counts_toward_multiplicity` is False only for exact aliases of another
# scheme. top1_mean is the mean of the single largest score, which is the
# maximum; reporting it as an independent "look" would overstate how many
# chances the null was given to fail.
AGG_SPECS: "OrderedDict[str, tuple]" = OrderedDict([
    ("mean",      ("mean",  None, True)),
    ("max",       ("max",   None, True)),
    ("top1_mean", ("topk",  1,    False)),   # == max, kept because k=1 was asked for
    ("top2_mean", ("topk",  2,    True)),
    ("top3_mean", ("topk",  3,    True)),
    ("top5_mean", ("topk",  5,    True)),
    ("q75",       ("quant", 0.75, True)),
    ("q90",       ("quant", 0.90, True)),
])

AGG_NAMES = list(AGG_SPECS)
AGG_NAMES_DISTINCT = [n for n in AGG_NAMES if AGG_SPECS[n][2]]


def apply_aggregation(scores: np.ndarray, name: str) -> float:
    """
    Collapse one subject's slice scores to a single number.

    top-k with k > (slices this subject has) falls back to the mean of what
    exists rather than erroring or padding: padding with zeros would drag a
    short series down and padding with the max would drag it up, and both would
    make the score depend on the subject's slice count. How many subjects this
    affects is counted and reported, not hidden.
    """
    kind, param, _ = AGG_SPECS[name]
    s = np.asarray(scores, dtype=float)
    if s.size == 0:
        return float("nan")
    if kind == "mean":
        return float(s.mean())
    if kind == "max":
        return float(s.max())
    if kind == "topk":
        k = int(param)
        if k >= s.size:
            return float(s.mean())
        part = np.partition(s, s.size - k)[s.size - k:]
        return float(part.mean())
    if kind == "quant":
        return float(np.quantile(s, float(param)))
    raise ValueError(f"unknown aggregation kind {kind!r}")


def aggregate_matrix(labels, scores, clusters, names=None) -> dict:
    """
    One row per subject, one column per aggregation scheme.

    The subject label is max(slice labels) -- a subject with any positive slice
    is a positive subject -- which is the same convention stage 4 uses, so these
    rows are comparable to `patient_level_mean` / `patient_level_max` there.

    Because aggregation happens WITHIN a subject, resampling subjects later is
    just resampling rows of this matrix: the aggregation never has to be redone
    inside the bootstrap.
    """
    names = list(names or AGG_NAMES)
    labels, scores, clusters, n_dropped = s04._clean(labels, scores, clusters)
    uniq, groups = s04._cluster_index(clusters)
    n = len(uniq)
    S = np.empty((n, len(names)), dtype=float)
    y = np.empty(n, dtype=int)
    sizes = np.empty(n, dtype=int)
    n_mixed = 0
    n_pos_slices_per = np.empty(n, dtype=int)
    for i, g in enumerate(groups):
        yl = labels[g]
        if yl.min() != yl.max():
            n_mixed += 1
        y[i] = int(yl.max())
        sizes[i] = len(g)
        n_pos_slices_per[i] = int((yl == 1).sum())
        sg = scores[g]
        for j, nm in enumerate(names):
            S[i, j] = apply_aggregation(sg, nm)
    short = {nm: int((sizes < AGG_SPECS[nm][1]).sum())
             for nm in names if AGG_SPECS[nm][0] == "topk"}
    return {
        "cluster_ids": [str(u) for u in uniq],
        "labels": y,
        "scores": S,
        "names": names,
        "slices_per_cluster": sizes,
        "pos_slices_per_cluster": n_pos_slices_per,
        "n_clusters": int(n),
        "n_pos_clusters": int(y.sum()),
        "n_mixed_label_clusters": int(n_mixed),
        "n_dropped_nonfinite": int(n_dropped),
        "n_clusters_shorter_than_k": short,
    }


# ==========================================================================
# Part A primitives: many-column AUC and one shared bootstrap
# ==========================================================================

def auc_columns(labels: np.ndarray, S: np.ndarray) -> np.ndarray:
    """
    Mann-Whitney AUC of every column of S against one label vector, with
    mid-rank tie handling, computed in one pass.

    Vectorised because the shared-resample bootstrap needs 8 AUCs per replicate
    and 2000 replicates per estimate; the loop version is the whole runtime.
    Validated column-by-column against s04.auc_midrank in --self-test.
    """
    labels = np.asarray(labels).astype(int)
    S = np.asarray(S, dtype=float)
    if S.ndim == 1:
        S = S[:, None]
    n, k = S.shape
    npos = int((labels == 1).sum())
    nneg = int((labels == 0).sum())
    if npos == 0 or nneg == 0:
        return np.full(k, np.nan)
    order = np.argsort(S, axis=0, kind="mergesort")
    Ss = np.take_along_axis(S, order, axis=0)
    same = np.zeros((n, k), dtype=bool)
    if n > 1:
        same[1:, :] = Ss[1:, :] == Ss[:-1, :]
    gid = np.cumsum(~same, axis=0) - 1                     # 0-based group id per column
    flat = (gid + np.arange(k)[None, :] * n).ravel()       # disjoint ids across columns
    pos = np.broadcast_to(np.arange(1, n + 1)[:, None], (n, k)).ravel().astype(float)
    cnt = np.bincount(flat, minlength=k * n)
    tot = np.bincount(flat, weights=pos, minlength=k * n)
    mean_rank = tot / np.maximum(cnt, 1)
    ranks_sorted = mean_rank[flat].reshape(n, k)
    ranks = np.empty((n, k), dtype=float)
    np.put_along_axis(ranks, order, ranks_sorted, axis=0)
    r_pos = ranks[labels == 1, :].sum(axis=0)
    return (r_pos - npos * (npos + 1) / 2.0) / (npos * nneg)


def shared_cluster_bootstrap(labels, S, names, n_boot=2000, seed=0, alpha=0.05,
                             envelope_names=None) -> dict:
    """
    Percentile CIs for every aggregation scheme from ONE set of subject
    resamples, plus a selection-aware envelope.

    Rows here are already one-per-subject, so resampling rows IS the
    subject-clustered bootstrap that stage 4 performs (stage 4 calls
    cluster_bootstrap_auc with each patient as its own cluster).

    Sharing the resamples matters twice over. The obvious reason is that the
    schemes become comparable. The load-bearing reason is the envelope: within
    a replicate, max_j AUC_j >= AUC_j for every j, so quantile_q(max_j AUC_j)
    >= quantile_q(AUC_j) for every j and every q. The envelope's lower bound is
    therefore an upper bound on what ANY aggregation -- including one picked
    after seeing the data -- could have delivered.

    Degenerate replicates (single-class, or fewer than 2 distinct subjects) are
    skipped and counted, exactly as in stage 4. The skip rule depends only on
    the labels, which do not depend on the aggregation, so all schemes keep the
    same replicate set.
    """
    labels = np.asarray(labels).astype(int)
    S = np.asarray(S, dtype=float)
    names = list(names)
    env_names = list(envelope_names or names)
    env_cols = [names.index(nm) for nm in env_names]
    out = {
        "names": names,
        "envelope_names": env_names,
        "n_clusters": int(len(labels)),
        "n_pos_clusters": int((labels == 1).sum()),
        "n_boot_requested": int(n_boot),
        "n_boot_used": 0,
        "n_skipped_single_class": 0,
        "n_skipped_single_cluster": 0,
        "ci_method": f"cluster_bootstrap_percentile_{int((1 - alpha) * 100)}",
        "per_scheme": {},
        "envelope": None,
        "reason": None,
    }
    point = auc_columns(labels, S)
    for j, nm in enumerate(names):
        out["per_scheme"][nm] = {
            "auc": (float(point[j]) if np.isfinite(point[j]) else None),
            "ci_lo": None, "ci_hi": None, "boot_mean": None, "boot_sd": None,
            "excludes_chance_above": None,
        }
    if len(np.unique(labels)) < 2:
        out["reason"] = (f"single-class: {int((labels == 1).sum())}/{len(labels)} "
                         "positive subjects; AUC is undefined")
        return out
    if len(labels) < 2:
        out["reason"] = "fewer than 2 subjects; no resampling possible"
        return out

    rng = np.random.default_rng(seed)
    n = len(labels)
    vals, n_sc, n_su = [], 0, 0
    for _ in range(int(n_boot)):
        rows = rng.integers(0, n, size=n)
        if len(np.unique(rows)) < 2:
            n_su += 1
            continue
        yb = labels[rows]
        if yb.min() == yb.max():
            n_sc += 1
            continue
        v = auc_columns(yb, S[rows, :])
        if np.all(np.isfinite(v)):
            vals.append(v)
        else:
            n_sc += 1
    out["n_boot_used"] = len(vals)
    out["n_skipped_single_class"] = int(n_sc)
    out["n_skipped_single_cluster"] = int(n_su)
    if len(vals) < 20:
        out["reason"] = (f"only {len(vals)}/{n_boot} bootstrap replicates were "
                         f"evaluable ({n_sc} single-class, {n_su} single-subject); "
                         "no interval is reported")
        return out
    V = np.vstack(vals)                                   # (B, k)
    lo = np.quantile(V, alpha / 2, axis=0)
    hi = np.quantile(V, 1 - alpha / 2, axis=0)
    for j, nm in enumerate(names):
        d = out["per_scheme"][nm]
        d["ci_lo"] = float(lo[j])
        d["ci_hi"] = float(hi[j])
        d["boot_mean"] = float(V[:, j].mean())
        d["boot_sd"] = float(V[:, j].std(ddof=1))
        d["excludes_chance_above"] = bool(lo[j] > CHANCE)
    env = V[:, env_cols].max(axis=1)
    best_j = int(np.nanargmax([out["per_scheme"][nm]["auc"] or -np.inf
                               for nm in env_names]))
    out["envelope"] = {
        "definition": ("per replicate, the best AUC across the distinct aggregation "
                       "schemes; its lower quantile is >= every individual scheme's "
                       "lower bound, so it bounds what post-hoc scheme selection "
                       "could achieve"),
        "n_schemes": len(env_names),
        "best_scheme_on_point_estimate": env_names[best_j],
        "auc_best_observed": float(max(out["per_scheme"][nm]["auc"] for nm in env_names)),
        "ci_lo": float(np.quantile(env, alpha / 2)),
        "ci_hi": float(np.quantile(env, 1 - alpha / 2)),
        "excludes_chance_above": bool(np.quantile(env, alpha / 2) > CHANCE),
    }
    return out


# ==========================================================================
# Part A driver
# ==========================================================================

def build_estimates(results_dir: Path, cache_dir: Path, cohort_dir: Path) -> tuple:
    """
    Rebuild stage 4's list of ESTIMATES -- one per (cohort, region, split
    family, condition, seed) -- reusing stage 4's own pooler.

    This deliberately does not reimplement pooling. pool_folds refuses to
    concatenate folds whose subjects overlap, and that refusal is the only
    reason a pooled interval can be trusted; a second, looser copy of that logic
    living in this file would be the obvious place for the two to drift apart.
    """
    runs = s04.load_runs(results_dir)
    cohorts = sorted({r.get("cohort") for r in runs if r.get("cohort")})
    cluster_maps, cmap_notes = {}, {}
    for c in cohorts:
        info = s04.build_cluster_map(c, cache_dir, cohort_dir)
        cluster_maps[c] = info["map"]
        cmap_notes[c] = info["reason"] or f"subject_id via {info['source']}"

    units = {}
    for r in runs:
        units.setdefault(s04.unit_key(r), []).append(r)
    expected = {}
    for r in runs:
        if r.get("_fold") is not None:
            expected.setdefault((str(r.get("cohort")), str(r.get("region", "full"))),
                                set()).add(int(r["_fold"]))

    estimates, refusals = [], []
    for key, rs in sorted(units.items(), key=lambda kv: str(kv[0])):
        cohort, region, family, condition, sd = key
        cmap = cluster_maps.get(cohort)
        if family == "cv":
            pooled, info = s04.pool_folds(rs, cmap, expected.get((cohort, region)))
            if pooled is None:
                refusals.append({"unit": list(key), "reason": info["reason"]})
                continue
            run = pooled
        else:
            run = rs[0]
        test = run.get("test") or {}
        if not test.get("probs"):
            continue
        clusters, unit, source = s04.resolve_clusters(test, cmap)
        estimates.append({
            "cohort": cohort, "region": region, "split_family": family,
            "condition": condition, "seed": sd,
            "pooled": bool(run.get("pooled")),
            "folds": run.get("folds"),
            "cluster_unit": unit, "cluster_source": source,
            "labels": np.asarray(test["labels"], dtype=float),
            "probs": np.asarray(test["probs"], dtype=float),
            "clusters": clusters,
        })
    return estimates, refusals, cmap_notes


def dilution_profile(agg: dict) -> dict:
    """
    Is the reviewer's premise even true for this cohort?

    Reports the positive-slice rate, and for POSITIVE subjects the distribution
    of how many of their slices are positive. If positive subjects are mostly
    positive throughout, averaging cannot be diluting anything and the objection
    dies here; if a positive subject typically has 2 positive slices out of 30,
    the objection is live and has to be answered with the aggregation sweep.
    """
    y = agg["labels"]
    sizes = agg["slices_per_cluster"].astype(float)
    npos = agg["pos_slices_per_cluster"].astype(float)
    pos = y == 1
    prof = {
        "n_subjects": int(len(y)),
        "n_pos_subjects": int(pos.sum()),
        "n_slices": int(sizes.sum()),
        "n_pos_slices": int(npos.sum()),
        "positive_slice_rate": float(npos.sum() / sizes.sum()) if sizes.sum() else None,
        "slices_per_subject_median": float(np.median(sizes)) if len(sizes) else None,
    }
    if pos.any():
        frac = npos[pos] / sizes[pos]
        prof.update({
            "pos_subject_pos_slices_median": float(np.median(npos[pos])),
            "pos_subject_pos_slices_min": float(npos[pos].min()),
            "pos_subject_pos_slices_max": float(npos[pos].max()),
            "pos_subject_pos_fraction_q25": float(np.quantile(frac, 0.25)),
            "pos_subject_pos_fraction_median": float(np.median(frac)),
            "pos_subject_pos_fraction_q75": float(np.quantile(frac, 0.75)),
            "n_pos_subjects_with_one_pos_slice": int((npos[pos] == 1).sum()),
        })
    return prof


def crosscheck_against_stage4(rows: list, stats_json: Path) -> dict:
    """
    The mean and max columns here must reproduce stage 4's
    patient_level_mean / patient_level_max point estimates exactly.

    They are computed by different code from the same JSONs, so an exact match
    is evidence that this module is pooling and clustering the way stage 4 does
    and that any difference in the other six columns is the aggregation and
    nothing else. A mismatch is reported as a failure, loudly, rather than
    quietly tolerated.
    """
    out = {"stats_json": str(stats_json), "checked": 0, "mismatches": [],
           "reason": None}
    if not Path(stats_json).exists():
        out["reason"] = f"{stats_json} not found; cross-check skipped"
        return out
    st = json.loads(Path(stats_json).read_text())
    by_tag = {}
    for r in st.get("runs", []):
        by_tag[(str(r.get("cohort")), str(r.get("condition")), r.get("seed"))] = r
    for row in rows:
        key = (row["cohort"], row["condition"], row["seed"])
        ref = by_tag.get(key)
        if not ref:
            continue
        for nm, refkey in (("mean", "patient_level_mean"), ("max", "patient_level_max")):
            want = (ref.get(refkey) or {}).get("auc")
            got = (row["bootstrap"]["per_scheme"].get(nm) or {}).get("auc")
            if want is None or got is None:
                continue
            out["checked"] += 1
            if abs(float(want) - float(got)) > 1e-9:
                out["mismatches"].append({
                    "cohort": row["cohort"], "condition": row["condition"],
                    "seed": row["seed"], "scheme": nm,
                    "s04": float(want), "s09": float(got),
                })
    return out


try:
    CLINICAL_COHORTS = tuple(common.CLINICAL_COHORTS)
    CONFOUND_COHORTS = tuple(common.CONFOUND_COHORTS)
except Exception:                                    # pragma: no cover
    CLINICAL_COHORTS = ("prostate_dwi", "prostate_t2", "breast")
    CONFOUND_COHORTS = ("brain", "knee")


def _aggregation_verdict(rows: list) -> dict:
    """
    Decide, from the sweep, whether the null survives the choice of aggregation.

    Two scoping rules, both of which change the answer if you get them wrong:

    1. ONLY THE CLINICAL COHORTS ARE ON TRIAL. brain and knee are confound
       cohorts whose label is receive-coil count, not pathology. Phase scoring
       far above chance there is the paper's FINDING, not a threat to it, and
       sweeping it into "a phase result went above chance" would invert the
       meaning of this analysis. Those cohorts are reported separately as a
       POSITIVE CONTROL: they are the evidence that this sweep can see a real
       effect when one is present, which is what makes a null from the same
       sweep on the clinical cohorts worth anything.

    2. THE DECISION IS MADE THE WAY C1 MAKES IT. C1 envelopes seeds with a
       min-lo / max-hi hull, so a cohort clears chance only if the LOWEST lower
       bound across seeds is above 0.500. A single seed clearing 0.500 under a
       single scheme is a seed-and-scheme coincidence, and it is reported as
       such -- named, not buried -- rather than either ignored or promoted.
    """
    def scan(cohorts):
        hits = []
        for r in rows:
            if r["condition"] != "phase" or r["cohort"] not in cohorts:
                continue
            for nm in AGG_NAMES:
                d = r["bootstrap"]["per_scheme"].get(nm) or {}
                if d.get("ci_lo") is not None and d["ci_lo"] > CHANCE:
                    hits.append({"cohort": r["cohort"], "seed": r["seed"], "scheme": nm,
                                 "auc": d["auc"], "ci_lo": d["ci_lo"],
                                 "ci_hi": d["ci_hi"]})
        return hits

    clinical_hits = scan(CLINICAL_COHORTS)
    control_hits = scan(CONFOUND_COHORTS)

    # C1's own rule: envelope over seeds, min lower bound.
    seed_env, env_lo = {}, {}
    for r in rows:
        if r["condition"] != "phase" or r["cohort"] not in CLINICAL_COHORTS:
            continue
        for nm in AGG_NAMES:
            d = r["bootstrap"]["per_scheme"].get(nm) or {}
            if d.get("ci_lo") is not None:
                seed_env.setdefault(r["cohort"], {}).setdefault(nm, []).append(d["ci_lo"])
        e = r["bootstrap"].get("envelope") or {}
        if e.get("ci_lo") is not None:
            env_lo.setdefault(r["cohort"], []).append(float(e["ci_lo"]))
    c1_style = {}
    for coh, per in seed_env.items():
        c1_style[coh] = {nm: {"min_ci_lo_over_seeds": float(min(v)),
                              "n_seeds": len(v),
                              "clears_chance": bool(min(v) > CHANCE)}
                         for nm, v in per.items()}
    c1_lifts = [{"cohort": coh, "scheme": nm, **d}
                for coh, per in c1_style.items() for nm, d in per.items()
                if d["clears_chance"]]

    worst_env = [v for vs in env_lo.values() for v in vs]
    best_env = max(worst_env) if worst_env else None
    verdict = {
        "question": ("does ANY patient aggregation lift a CLINICAL phase result's "
                     "95% CI lower bound above 0.500?"),
        "clinical_cohorts": list(CLINICAL_COHORTS),
        "confound_cohorts_excluded_from_the_decision": list(CONFOUND_COHORTS),
        "n_distinct_schemes": len(AGG_NAMES_DISTINCT),
        "n_schemes_reported": len(AGG_NAMES),
        "alias_schemes": [n for n in AGG_NAMES if not AGG_SPECS[n][2]],
        "per_seed_scheme_hits": clinical_hits,
        "any_per_seed_hit": bool(clinical_hits),
        "c1_rule": "envelope over seeds; cohort clears chance only if min(ci_lo) > 0.500",
        "c1_style_min_ci_lo_over_seeds": c1_style,
        "c1_style_lifts": c1_lifts,
        "any_cohort_clears_under_c1_rule": bool(c1_lifts),
        "envelope_ci_lo_by_clinical_cohort": {k: vs for k, vs in env_lo.items()},
        "selection_aware_envelope_max_ci_lo": best_env,
        "positive_control_hits_on_confound_cohorts": control_hits,
        "positive_control_note": (
            f"{len(control_hits)} confound-cohort phase result(s) DO clear chance under "
            "these same schemes; the sweep is therefore capable of detecting an effect "
            "and a null from it on the clinical cohorts is informative rather than "
            "merely underpowered machinery"),
        "statement": None,
    }
    if c1_lifts:
        verdict["statement"] = (
            "AT LEAST ONE aggregation clears chance under C1's own cross-seed rule. "
            "The null is NOT robust to the choice of aggregation and the affected "
            "cohort's verdict must be re-examined. See `c1_style_lifts`.")
    else:
        head = (f"No aggregation lifts any clinical phase result above {CHANCE:.3f} "
                "under C1's cross-seed rule. ")
        if clinical_hits:
            named = "; ".join(f"{h['cohort']} seed{h['seed']} {h['scheme']} "
                              f"{h['auc']:.3f} [{h['ci_lo']:.3f}, {h['ci_hi']:.3f}]"
                              for h in clinical_hits)
            head += (f"{len(clinical_hits)} single-seed/single-scheme cell(s) do clear "
                     f"0.500 on their own ({named}), but the other seed of the same "
                     "cohort does not, so the min-lo envelope C1 uses stays below "
                     "chance -- that is a seed coincidence, not an aggregation effect. ")
        if best_env is not None:
            head += (
                f"The selection-aware envelope -- the 2.5th percentile of the best-of-"
                f"{len(AGG_NAMES_DISTINCT)} AUC within each subject resample, which "
                "dominates every individual scheme's lower bound -- tops out at "
                f"{best_env:.3f} across the clinical cohorts, so no aggregation chosen "
                "after the fact could have produced a lower bound above chance either. ")
        head += "The null is a property of the data, not an artefact of averaging."
        verdict["statement"] = head
    return verdict


def run_aggregation_sensitivity(results_dir: Path, cache_dir: Path, cohort_dir: Path,
                                n_boot: int, seed: int, alpha: float,
                                stats_json: Path | None = None) -> dict:
    estimates, refusals, cmap_notes = build_estimates(results_dir, cache_dir, cohort_dir)
    rows = []
    for est in estimates:
        agg = aggregate_matrix(est["labels"], est["probs"], est["clusters"])
        boot = shared_cluster_bootstrap(
            agg["labels"], agg["scores"], agg["names"],
            n_boot=n_boot, seed=seed, alpha=alpha,
            envelope_names=AGG_NAMES_DISTINCT)
        rows.append({
            "cohort": est["cohort"], "condition": est["condition"], "seed": est["seed"],
            "region": est["region"], "split_family": est["split_family"],
            "pooled": est["pooled"], "folds": est["folds"],
            "cluster_unit": est["cluster_unit"], "cluster_source": est["cluster_source"],
            "n_subjects": agg["n_clusters"], "n_pos_subjects": agg["n_pos_clusters"],
            "n_mixed_label_subjects": agg["n_mixed_label_clusters"],
            "n_subjects_shorter_than_k": agg["n_clusters_shorter_than_k"],
            "dilution": dilution_profile(agg),
            "bootstrap": boot,
        })

    verdict = _aggregation_verdict(rows)
    return {
        "config": {"n_boot": n_boot, "bootstrap_seed": seed, "alpha": alpha,
                   "results_dir": str(results_dir),
                   "schemes": {k: {"kind": v[0], "param": v[1], "distinct": v[2]}
                               for k, v in AGG_SPECS.items()}},
        "cluster_map_notes": cmap_notes,
        "pooling_refusals": refusals,
        "rows": rows,
        "crosscheck_vs_stage4": (crosscheck_against_stage4(rows, stats_json)
                                 if stats_json else
                                 {"reason": "no --stats-json given"}),
        "verdict": verdict,
    }


# ==========================================================================
# Part B primitives: association measures
# ==========================================================================

def contingency(a, b) -> tuple:
    """Counts table for two categorical vectors, with their level orders."""
    a = np.asarray([str(v) for v in a], dtype=object)
    b = np.asarray([str(v) for v in b], dtype=object)
    la = sorted(set(a.tolist()))
    lb = sorted(set(b.tolist()))
    ia = {v: i for i, v in enumerate(la)}
    ib = {v: i for i, v in enumerate(lb)}
    T = np.zeros((len(la), len(lb)), dtype=float)
    for x, y in zip(a, b):
        T[ia[x], ib[y]] += 1
    return T, la, lb


def chi2_stat(T: np.ndarray) -> float:
    r = T.sum(axis=1, keepdims=True)
    c = T.sum(axis=0, keepdims=True)
    n = T.sum()
    if n <= 0:
        return float("nan")
    E = r @ c / n
    with np.errstate(divide="ignore", invalid="ignore"):
        term = np.where(E > 0, (T - E) ** 2 / np.maximum(E, 1e-300), 0.0)
    return float(term.sum())


def cramers_v(T: np.ndarray) -> dict:
    """
    Cramer's V, raw and Bergsma (2013) bias-corrected.

    The correction is not decoration. V is biased upward when the table is
    sparse relative to n, and 'coil bucket vs scanner model' is a 2x6 table on
    454 subjects with several near-empty cells -- exactly where an uncorrected V
    flatters the association. Both are reported so a reader can see the size of
    the correction rather than take one number on trust.
    """
    n = float(T.sum())
    r, c = T.shape
    out = {"n": n, "r": int(r), "c": int(c), "chi2": None,
           "v": None, "v_bias_corrected": None}
    if n <= 0 or min(r, c) < 2:
        return out
    chi2 = chi2_stat(T)
    out["chi2"] = chi2
    phi2 = chi2 / n
    out["v"] = float(math.sqrt(max(phi2, 0.0) / (min(r, c) - 1)))
    if n > 1:
        phi2c = max(0.0, phi2 - (r - 1) * (c - 1) / (n - 1))
        rc = r - (r - 1) ** 2 / (n - 1)
        cc = c - (c - 1) ** 2 / (n - 1)
        denom = min(rc, cc) - 1
        out["v_bias_corrected"] = (float(math.sqrt(phi2c / denom)) if denom > 0 else None)
    return out


def mutual_information(T: np.ndarray) -> dict:
    """
    Mutual information between the two table margins, in nats and bits, plus
    the uncertainty coefficient U(col|row) = I / H(col).

    U is the number that answers the question in plain language: what FRACTION
    of the uncertainty about coil bucket is removed by being told the site (or
    the scanner)? A V of 0.9 is hard to interpret; "knowing the scanner model
    removes 78% of the uncertainty about coil count" is not.
    """
    n = float(T.sum())
    out = {"mi_nats": None, "mi_bits": None, "h_row_nats": None, "h_col_nats": None,
           "nmi_sqrt": None, "u_col_given_row": None, "cond_entropy_col_given_row": None}
    if n <= 0:
        return out
    P = T / n
    pr = P.sum(axis=1)
    pc = P.sum(axis=0)

    def H(p):
        p = p[p > 0]
        return float(-(p * np.log(p)).sum())

    hr, hc = H(pr), H(pc)
    outer = np.outer(pr, pc)
    mask = (P > 0) & (outer > 0)
    mi = float((P[mask] * np.log(P[mask] / outer[mask])).sum())
    mi = max(mi, 0.0)
    out.update({
        "mi_nats": mi, "mi_bits": mi / math.log(2),
        "h_row_nats": hr, "h_col_nats": hc,
        "nmi_sqrt": (float(mi / math.sqrt(hr * hc)) if hr > 0 and hc > 0 else None),
        "u_col_given_row": (float(mi / hc) if hc > 0 else None),
        "cond_entropy_col_given_row": float(hc - mi),
    })
    return out


def association(a, b, n_perm: int = 2000, seed: int = 0) -> dict:
    """
    V and MI with a permutation null.

    Both statistics are non-negative and both are biased upward on sparse
    tables, so "V = 0.31" means nothing until it is compared against what V
    would have been for these same margins under independence. The permutation
    shuffles one variable, holding both margins' shapes, and reports the null
    mean plus a p-value floored at 1/(n_perm+1) -- a permutation test cannot
    resolve below its own resolution and must not claim to.
    """
    T, la, lb = contingency(a, b)
    res = {
        "levels_row": la, "levels_col": lb,
        "table": T.astype(int).tolist(),
        "cramers_v": cramers_v(T),
        "mutual_information": mutual_information(T),
        "permutation": None,
    }
    if n_perm and min(T.shape) >= 2 and T.sum() > 1:
        rng = np.random.default_rng(seed)
        a_arr = np.asarray([str(v) for v in a], dtype=object)
        b_arr = np.asarray([str(v) for v in b], dtype=object)
        v_obs = res["cramers_v"]["v"]
        mi_obs = res["mutual_information"]["mi_nats"]
        vs, mis = [], []
        for _ in range(int(n_perm)):
            Tp, _, _ = contingency(a_arr, rng.permutation(b_arr))
            vs.append(cramers_v(Tp)["v"])
            mis.append(mutual_information(Tp)["mi_nats"])
        vs = np.asarray(vs, dtype=float)
        mis = np.asarray(mis, dtype=float)
        res["permutation"] = {
            "n_perm": int(n_perm),
            "v_null_mean": float(vs.mean()), "v_null_q95": float(np.quantile(vs, 0.95)),
            "v_p": float(max((vs >= v_obs).sum() + 1, 1) / (n_perm + 1)),
            "mi_null_mean_nats": float(mis.mean()),
            "mi_null_q95_nats": float(np.quantile(mis, 0.95)),
            "mi_p": float(max((mis >= mi_obs).sum() + 1, 1) / (n_perm + 1)),
            "mi_excess_over_null_nats": float(mi_obs - mis.mean()),
        }
    return res


# ==========================================================================
# Part B primitives: site normalisation
# ==========================================================================

def canon_institution(s) -> str:
    """
    Canonical form of an institution string: upper case, punctuation to space,
    runs of whitespace collapsed.

    This is deliberately conservative. It merges spelling and spacing variants
    and NOTHING else -- 'NYU' and 'NYU LANGONE CBI' survive as two levels. Any
    merging beyond this has to be justified by evidence, which is what
    `sites_from_devices` does.
    """
    if s is None:
        return "UNKNOWN"
    t = str(s).strip()
    if t == "" or t.lower() in ("nan", "none", "unknown"):
        return "UNKNOWN"
    keep = []
    for ch in t.upper():
        keep.append(ch if (ch.isalnum() or ch.isspace()) else " ")
    return " ".join("".join(keep).split())


def site_prefix(s) -> str:
    """First token of the canonical string. A normalisation that used no data."""
    c = canon_institution(s)
    return c.split(" ")[0] if c else "UNKNOWN"


def sites_from_devices(institutions, devices) -> dict:
    """
    Recover true sites as connected components of the institution <-> device_id
    bipartite graph.

    The argument: a scanner is bolted to a floor. If serial number 45219 is
    recorded as 'NYU' on some exams and 'NYU LANGONE CBI' on others, those two
    strings name the same place and treating them as two sites would
    manufacture site-level structure that does not exist. Conversely two strings
    that never share a magnet are left apart, however similar they look --
    'TH' and 'TH RADIOLOGY' are merged because they share magnets 41964 and
    45774, not because both start with 'TH'.

    Returns the mapping, the components, and the shared-device evidence for
    every merge, so each merge can be audited individually.
    """
    inst = [canon_institution(v) for v in institutions]
    dev = [("DEV:" + str(v)) for v in devices]
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for i, d in zip(inst, dev):
        union("INST:" + i, d)

    comps = {}
    for i in set(inst):
        comps.setdefault(find("INST:" + i), []).append(i)

    counts = {}
    for i in inst:
        counts[i] = counts.get(i, 0) + 1

    mapping, components, evidence = {}, [], []
    for root, members in comps.items():
        members = sorted(members)
        # name the site after its most common member string, shortest as tiebreak
        name = sorted(members, key=lambda m: (-counts.get(m, 0), len(m), m))[0]
        for m in members:
            mapping[m] = name
        components.append({"site": name, "institution_strings": members,
                           "n_strings": len(members),
                           "n_rows": int(sum(counts.get(m, 0) for m in members))})
        if len(members) > 1:
            shared = {}
            for i, d in zip(inst, dev):
                if i in members:
                    shared.setdefault(d.replace("DEV:", ""), set()).add(i)
            evidence.append({
                "site": name,
                "devices_shared_by_more_than_one_string": {
                    k: sorted(v) for k, v in sorted(shared.items()) if len(v) > 1},
            })
    components.sort(key=lambda c: -c["n_rows"])
    return {"map": mapping, "components": components, "merge_evidence": evidence,
            "n_input_strings": len(set(inst)), "n_sites": len(components)}


# ==========================================================================
# Part B primitives: separability and within-stratum AUC
# ==========================================================================

def separability(strata, labels, min_per_class: int = 10) -> dict:
    """
    Which strata contain BOTH classes with enough subjects in each to test the
    label inside them?

    `min_per_class` is a power floor, not a significance rule. A stratum with 1
    positive subject produces an AUC that is one subject's score against the
    negatives; reporting it next to a 98-subject stratum as if they were the
    same kind of evidence is how underdetermined claims get made.
    """
    strata = np.asarray([str(v) for v in strata], dtype=object)
    labels = np.asarray(labels).astype(int)
    rows = []
    for lv in sorted(set(strata.tolist())):
        sel = strata == lv
        npos = int((labels[sel] == 1).sum())
        nneg = int((labels[sel] == 0).sum())
        rows.append({"stratum": lv, "n": int(sel.sum()), "n_pos": npos, "n_neg": nneg,
                     "spans_both": bool(npos > 0 and nneg > 0),
                     "meets_floor": bool(npos >= min_per_class and nneg >= min_per_class),
                     "discordant_pairs": int(npos * nneg)})
    tot_pairs = int((labels == 1).sum()) * int((labels == 0).sum())
    usable = sum(r["discordant_pairs"] for r in rows if r["spans_both"])
    return {
        "min_per_class": int(min_per_class),
        "n_strata": len(rows),
        "n_strata_spanning_both": sum(1 for r in rows if r["spans_both"]),
        "n_strata_meeting_floor": sum(1 for r in rows if r["meets_floor"]),
        "strata": rows,
        "discordant_pairs_total": tot_pairs,
        "discordant_pairs_within_strata": usable,
        "fraction_of_pairs_retained": (float(usable / tot_pairs) if tot_pairs else None),
    }


def _stratified_auc(labels, scores, strata, floor: int) -> tuple:
    """Pair-weighted average of within-stratum AUCs; weights are n_pos*n_neg."""
    num, den = 0.0, 0.0
    per = {}
    for lv in sorted(set(strata.tolist())):
        sel = strata == lv
        y, s = labels[sel], scores[sel]
        npos, nneg = int((y == 1).sum()), int((y == 0).sum())
        if npos == 0 or nneg == 0:
            continue
        a = float(auc_columns(y, s[:, None])[0])
        per[lv] = (a, npos, nneg)
        if npos >= floor and nneg >= floor:
            num += npos * nneg * a
            den += npos * nneg
    return (num / den if den > 0 else float("nan")), per


def within_stratum_auc(labels, scores, strata, n_boot=2000, seed=0, alpha=0.05,
                       floor: int = 10) -> dict:
    """
    AUC computed INSIDE each stratum, plus a pair-weighted pooled estimate over
    the strata that clear the power floor.

    The pooled number is the conditional statistic: it asks only about pairs of
    subjects who share a stratum, so between-stratum differences -- the very
    thing that would make "coil count" a relabelling of "site" -- cannot
    contribute to it. It is bootstrapped by resampling subjects WITHIN each
    stratum, which preserves the stratum sizes; resampling across strata would
    let a stratum vanish and change what the statistic is estimating.
    """
    labels = np.asarray(labels).astype(int)
    scores = np.asarray(scores, dtype=float)
    strata = np.asarray([str(v) for v in strata], dtype=object)
    point, per = _stratified_auc(labels, scores, strata, floor)
    unstrat = float(auc_columns(labels, scores[:, None])[0])
    out = {
        "unstratified_auc": (unstrat if np.isfinite(unstrat) else None),
        "stratified_auc": (float(point) if np.isfinite(point) else None),
        "stratified_ci_lo": None, "stratified_ci_hi": None,
        "floor_per_class": int(floor),
        "per_stratum": {},
        "n_boot_used": 0,
        "reason": None,
    }
    idx_by = {lv: np.flatnonzero(strata == lv) for lv in sorted(set(strata.tolist()))}
    for lv, (a, npos, nneg) in per.items():
        rows = idx_by[lv]
        d = {"auc": a, "n_pos": npos, "n_neg": nneg,
             "meets_floor": bool(npos >= floor and nneg >= floor),
             "ci_lo": None, "ci_hi": None}
        if npos >= 3 and nneg >= 3:
            rng = np.random.default_rng(seed + abs(hash(lv)) % 10007)
            vals = []
            for _ in range(int(n_boot)):
                draw = rng.integers(0, len(rows), size=len(rows))
                yb = labels[rows][draw]
                if yb.min() == yb.max():
                    continue
                v = float(auc_columns(yb, scores[rows][draw][:, None])[0])
                if np.isfinite(v):
                    vals.append(v)
            if len(vals) >= 20:
                d["ci_lo"] = float(np.quantile(vals, alpha / 2))
                d["ci_hi"] = float(np.quantile(vals, 1 - alpha / 2))
                d["n_boot_used"] = len(vals)
        out["per_stratum"][lv] = d

    used = [lv for lv, (a, npos, nneg) in per.items() if npos >= floor and nneg >= floor]
    if not used:
        out["reason"] = (f"no stratum has >= {floor} subjects in BOTH classes; the "
                         "within-stratum estimate is not computable at this floor")
        # Compute it anyway with the floor dropped to 1 and label it as
        # underpowered. Refusing outright invites "you just declined to look";
        # reporting the number next to the fraction of discordant pairs it rests
        # on lets a reader see for themselves that it rests on almost nothing.
        loose, _p = _stratified_auc(labels, scores, strata, 1)
        pairs_all = int((labels == 1).sum()) * int((labels == 0).sum())
        pairs_in = sum(npos * nneg for _lv, (_a, npos, nneg) in per.items())
        out["underpowered_all_spanning_strata"] = {
            "stratified_auc": (float(loose) if np.isfinite(loose) else None),
            "n_strata_used": len(per),
            "discordant_pairs_used": int(pairs_in),
            "discordant_pairs_total": int(pairs_all),
            "fraction_of_pairs_retained": (float(pairs_in / pairs_all)
                                           if pairs_all else None),
            "warning": ("floor dropped to 1 subject per class purely so the number "
                        "exists; it is NOT evidence and must not be quoted as the "
                        "within-stratum result"),
        }
        return out
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_boot)):
        rows = np.concatenate([idx_by[lv][rng.integers(0, len(idx_by[lv]),
                                                       size=len(idx_by[lv]))]
                               for lv in used])
        v, _p = _stratified_auc(labels[rows], scores[rows], strata[rows], floor)
        if np.isfinite(v):
            vals.append(v)
    out["n_boot_used"] = len(vals)
    if len(vals) >= 20:
        out["stratified_ci_lo"] = float(np.quantile(vals, alpha / 2))
        out["stratified_ci_hi"] = float(np.quantile(vals, 1 - alpha / 2))
    else:
        out["reason"] = f"only {len(vals)}/{n_boot} evaluable replicates"
    out["strata_used"] = used
    return out


# ==========================================================================
# Part B driver
# ==========================================================================

_STRAT_COLS = OrderedDict([
    ("site", "site recovered from institution<->device_id components"),
    ("institution_canon", "canonicalised institution string, no merging"),
    ("site_prefix", "first token of the institution string (no data used)"),
    ("scanner", "scanner model"),
    ("device_id", "individual magnet serial"),
    ("field_strength", "nominal field strength"),
    ("coil_array", "coil array name"),
])


def _read_cache_index(cache_dir: Path, cohort: str) -> list:
    import csv
    p = Path(cache_dir) / f"{cohort}_index.csv"
    with open(p, newline="") as fh:
        return list(csv.DictReader(fh))


def run_coil_vs_site(cache_dir: Path, results_dir: Path, cohort: str = "brain",
                     seeds=(42, 123), condition: str = "phase",
                     n_boot: int = 2000, n_perm: int = 2000, seed: int = 0,
                     alpha: float = 0.05, floor: int = 10) -> dict:
    rows = _read_cache_index(cache_dir, cohort)
    if not rows:
        return {"reason": f"no cache index for {cohort}"}

    # one record per subject -- association is a property of subjects, not slices
    by_subj = {}
    for r in rows:
        by_subj.setdefault(str(r["subject_id"]), r)
    subs = [by_subj[k] for k in sorted(by_subj)]
    inst = [r.get("institution") for r in subs]
    dev = [r.get("device_id") for r in subs]
    sites = sites_from_devices(inst, dev)

    def col(r, name):
        if name == "site":
            return sites["map"].get(canon_institution(r.get("institution")), "UNKNOWN")
        if name == "institution_canon":
            return canon_institution(r.get("institution"))
        if name == "site_prefix":
            return site_prefix(r.get("institution"))
        return str(r.get(name, "") or "UNKNOWN")

    y = np.asarray([int(r["label"]) for r in subs], dtype=int)
    coil = [str(r.get("n_coils")) for r in subs]
    label_name = subs[0].get("label_target") or "label"

    out = {
        "cohort": cohort,
        "label_definition": label_name,
        "n_subjects": len(subs),
        "n_pos_subjects": int(y.sum()),
        "coil_counts_by_bucket": {},
        "site_normalisation": sites,
        "prefix_vs_device_normalisation": None,
        "associations": {},
        "pairwise_context_associations": {},
        "separability": {},
        "model_within_stratum": {},
        "verdict": {},
    }
    for b in (0, 1):
        vals = {}
        for c, yy in zip(coil, y):
            if yy == b:
                vals[c] = vals.get(c, 0) + 1
        out["coil_counts_by_bucket"][int(b)] = dict(sorted(
            vals.items(), key=lambda kv: float(kv[0])))

    # do the two normalisations agree?
    pref = [col(r, "site_prefix") for r in subs]
    site = [col(r, "site") for r in subs]
    Tps, lp, ls = contingency(pref, site)
    out["prefix_vs_device_normalisation"] = {
        "prefix_levels": lp, "device_site_levels": ls,
        "table": Tps.astype(int).tolist(),
        "identical_partition": bool(
            all((Tps[i] > 0).sum() == 1 for i in range(Tps.shape[0]))
            and all((Tps[:, j] > 0).sum() == 1 for j in range(Tps.shape[1]))),
        "note": ("if these two partitions differ, the device-based one is preferred: "
                 "it merges strings only where a physical magnet is shared"),
    }

    # ---- association of every acquisition-context variable with coil bucket
    for name, desc in _STRAT_COLS.items():
        vals = [col(r, name) for r in subs]
        a = association(vals, y.tolist(), n_perm=n_perm, seed=seed)
        a["description"] = desc
        a["n_levels"] = len(set(vals))
        out["associations"][name] = a

    # ---- how collinear are the context variables with EACH OTHER?
    keys = ["site", "scanner", "device_id", "field_strength", "coil_array"]
    for i, ka in enumerate(keys):
        for kb in keys[i + 1:]:
            va = [col(r, ka) for r in subs]
            vb = [col(r, kb) for r in subs]
            T, _, _ = contingency(va, vb)
            out["pairwise_context_associations"][f"{ka}|{kb}"] = {
                "cramers_v": cramers_v(T)["v"],
                "v_bias_corrected": cramers_v(T)["v_bias_corrected"],
                "u_col_given_row": mutual_information(T)["u_col_given_row"],
            }

    # ---- separability over the WHOLE cohort
    for name in _STRAT_COLS:
        vals = [col(r, name) for r in subs]
        out["separability"][name] = separability(vals, y, min_per_class=floor)

    # ---- the model's own held-out predictions, stratified
    idx_map = {int(r["idx"]): r for r in rows}
    for sd in seeds:
        f = Path(results_dir) / f"confound_{cohort}" / f"{cohort}_{condition}_seed{sd}.json"
        if not f.exists():
            out["model_within_stratum"][str(sd)] = {"reason": f"{f} not found"}
            continue
        payload = json.loads(f.read_text())
        test = payload.get("test") or {}
        cidx = [int(v) for v in test["cache_idx"]]
        recs = [idx_map[i] for i in cidx]
        clusters = np.asarray([str(r["subject_id"]) for r in recs], dtype=object)
        agg = aggregate_matrix(np.asarray(test["labels"], dtype=float),
                               np.asarray(test["probs"], dtype=float),
                               clusters, names=["mean", "max"])
        sub_of_row = {str(r["subject_id"]): r for r in recs}
        order = agg["cluster_ids"]
        per_seed = {
            "path": str(f), "condition": condition, "seed": sd,
            "n_test_subjects": agg["n_clusters"],
            "n_pos_test_subjects": agg["n_pos_clusters"],
            "test_separability": {},
            "strata": {},
        }
        for name in _STRAT_COLS:
            vals = np.asarray([col(sub_of_row[s], name) for s in order], dtype=object)
            per_seed["test_separability"][name] = separability(
                vals, agg["labels"], min_per_class=floor)
            per_seed["strata"][name] = within_stratum_auc(
                agg["labels"], agg["scores"][:, 0], vals,
                n_boot=n_boot, seed=seed, alpha=alpha, floor=floor)
            per_seed["strata"][name]["aggregation"] = "mean"
        out["model_within_stratum"][str(sd)] = per_seed

    out["verdict"] = _coil_vs_site_verdict(out, floor)
    return out


def _coil_vs_site_verdict(out: dict, floor: int) -> dict:
    """
    Turn the tables into the sentence the paper is allowed to write.

    Three outcomes are possible and all three are writable:
      SEPARABLE          -- a stratum spans both coil buckets with power, and the
                            within-stratum AUC holds up. The hardware claim stands
                            in its clean form.
      NOT SEPARABLE      -- no stratum can test coil count. The claim must be
                            demoted to "acquisition context (site/scanner/coil
                            jointly)", which is still sufficient for the paper.
      PARTIALLY SEPARABLE -- separable from one context variable but not another.
                            The claim has to name which.
    """
    v = {"floor_per_class": floor, "separable_from": [], "not_separable_from": [],
         "within_stratum": {}, "claim": None, "reasoning": []}
    seeds = [k for k in out["model_within_stratum"]
             if isinstance(out["model_within_stratum"][k], dict)
             and "strata" in out["model_within_stratum"][k]]
    for name in _STRAT_COLS:
        coh = out["separability"].get(name, {})
        ok_cohort = coh.get("n_strata_meeting_floor", 0) > 0
        ok_test, aucs = False, []
        for sd in seeds:
            st = out["model_within_stratum"][sd]["strata"].get(name, {})
            if st.get("stratified_auc") is not None:
                ok_test = True
                aucs.append((sd, st["stratified_auc"], st.get("stratified_ci_lo"),
                             st.get("stratified_ci_hi"),
                             out["model_within_stratum"][sd]["strata"][name].get(
                                 "unstratified_auc")))
        if ok_cohort and ok_test:
            v["separable_from"].append(name)
            v["within_stratum"][name] = [
                {"seed": sd, "stratified_auc": a, "ci_lo": lo, "ci_hi": hi,
                 "unstratified_auc": un} for sd, a, lo, hi, un in aucs]
        else:
            v["not_separable_from"].append({
                "variable": name,
                "n_strata_meeting_floor_in_cohort": coh.get("n_strata_meeting_floor"),
                "testable_on_test_fold": ok_test,
            })
    # ---- how much of the within-stratum claim rests on ONE stratum? -------
    # "Coil count is testable within site" is worth much less if one site
    # carries the whole estimate and the other disagrees. That is the next
    # objection after this one, so it is measured here rather than waited for.
    for name in list(v["within_stratum"]):
        rows = []
        for sd in seeds:
            st = out["model_within_stratum"][sd]["strata"].get(name, {})
            used = set(st.get("strata_used") or [])
            for lv, d in (st.get("per_stratum") or {}).items():
                rows.append({"seed": sd, "stratum": lv, "auc": d["auc"],
                             "n_pos": d["n_pos"], "n_neg": d["n_neg"],
                             "counted": lv in used})
        counted = [r for r in rows if r["counted"]]
        dropped = [r for r in rows if not r["counted"]]
        note = None
        if dropped and counted:
            cmin = min(r["auc"] for r in counted)
            dmax = max(r["auc"] for r in dropped)
            if dmax < cmin - 0.10:
                note = (f"the estimate is carried by "
                        f"{sorted({r['stratum'] for r in counted})}; the "
                        f"{sorted({r['stratum'] for r in dropped})} stratum/a fall "
                        f"below the power floor and sit as low as {dmax:.3f}, so the "
                        "within-stratum claim rests on the large stratum and must be "
                        "written that way")
        v["within_stratum_heterogeneity"] = v.get("within_stratum_heterogeneity", {})
        v["within_stratum_heterogeneity"][name] = {
            "strata_counted": sorted({r["stratum"] for r in counted}),
            "strata_below_floor": sorted({r["stratum"] for r in dropped}),
            "per_stratum": rows,
            "note": note,
        }

    sep = set(v["separable_from"])
    if "site" in sep and "scanner" in sep:
        v["claim"] = ("SEPARABLE: coil count can be tested within both site and "
                      "scanner model. 'Phase encodes receive-coil hardware' stands "
                      "in its strong form.")
    elif "site" in sep:
        v["claim"] = (
            "PARTIALLY SEPARABLE: coil count is separable from SITE -- there are "
            "sites containing both coil buckets with enough subjects to test coil "
            "count inside them, and the within-site result is reported above. It is "
            "NOT separable from SCANNER MODEL: no scanner model carries enough "
            "subjects in both coil buckets. So the paper may say 'phase encodes "
            "hardware, not merely site', but may NOT decompose that hardware into "
            "coil count as distinct from the scanner it is attached to. The exact "
            "claim the data supports is: phase encodes the scanner-and-coil "
            "configuration, demonstrated within site.")
    elif sep:
        v["claim"] = (
            "PARTIALLY SEPARABLE: coil count is testable only within "
            f"{sorted(sep)}. Every other context variable is collinear with it. "
            "The honest claim is 'phase encodes acquisition context "
            "(site/scanner/coil jointly)'.")
    else:
        v["claim"] = (
            "NOT SEPARABLE: no context stratum contains both coil buckets with "
            "enough subjects to test coil count inside it. The claim must be stated "
            "as 'phase encodes acquisition context (site/scanner/coil jointly)', "
            "which is sufficient for the paper's argument -- a diagnostic score that "
            "inherits acquisition identity is confounded whether that identity is "
            "the coil, the magnet or the building -- but it is NOT a pure hardware "
            "claim and must not be written as one.")
    return v


# ==========================================================================
# Printing
# ==========================================================================

def _f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _rule(w=104, ch="-"):
    return ch * w


def print_aggregation(res: dict) -> None:
    print(_rule(104, "="))
    print("A. PATIENT AGGREGATION SENSITIVITY  --  does averaging bury a focal lesion?")
    print(_rule(104, "="))
    cc = res["crosscheck_vs_stage4"]
    if cc.get("reason"):
        print(f"stage-4 cross-check: {cc['reason']}")
    else:
        print(f"stage-4 cross-check: {cc['checked']} mean/max point estimates compared, "
              f"{len(cc['mismatches'])} mismatch(es)"
              + ("" if not cc["mismatches"] else "  <<< INVESTIGATE"))
        for m in cc["mismatches"]:
            print(f"   MISMATCH {m['cohort']}/{m['condition']}/seed{m['seed']}/{m['scheme']}: "
                  f"s04={m['s04']:.6f} s09={m['s09']:.6f}")
    if res["pooling_refusals"]:
        for r in res["pooling_refusals"]:
            print(f"POOLING REFUSED for {r['unit']}: {r['reason']}")

    print("\nslice-level dilution profile (is the reviewer's premise true here?)")
    print(f"{'cohort':<14} {'subj':>5} {'pos':>4} {'slices':>7} {'pos%':>6} "
          f"{'slices/subj':>11} {'pos slices in a +subj: med [min,max]':>38}")
    seen = set()
    for r in res["rows"]:
        if r["cohort"] in seen or r["condition"] != "phase":
            continue
        seen.add(r["cohort"])
        d = r["dilution"]
        med = d.get("pos_subject_pos_slices_median")
        lo, hi = d.get("pos_subject_pos_slices_min"), d.get("pos_subject_pos_slices_max")
        print(f"{r['cohort']:<14} {d['n_subjects']:>5} {d['n_pos_subjects']:>4} "
              f"{d['n_slices']:>7} {100 * (d['positive_slice_rate'] or 0):>5.1f}% "
              f"{d['slices_per_subject_median']:>11.0f} "
              f"{'' if med is None else f'{med:>10.0f} [{lo:.0f}, {hi:.0f}]':>38}")

    for cond in ("phase", "magnitude", "both"):
        rows = [r for r in res["rows"] if r["condition"] == cond]
        if not rows:
            continue
        print(f"\npatient-level AUC [95% subject-clustered CI] -- condition = {cond}")
        hdr = f"{'cohort':<14}{'seed':>5}{'n':>5}{'pos':>5}  "
        hdr += "".join(f"{nm:>20}" for nm in AGG_NAMES[:4])
        print(hdr)
        print(f"{'':<29}  " + "".join(f"{nm:>20}" for nm in AGG_NAMES[4:])
              + f"{'ENVELOPE':>22}")
        print(_rule(104))
        for r in sorted(rows, key=lambda r: (r["cohort"], r["seed"])):
            ps = r["bootstrap"]["per_scheme"]

            def cell(nm):
                d = ps.get(nm) or {}
                if d.get("auc") is None:
                    return f"{'-':>20}"
                star = "*" if d.get("excludes_chance_above") else " "
                return f"{_f(d['auc'])} [{_f(d['ci_lo'],2)},{_f(d['ci_hi'],2)}]{star:>1}"[-20:].rjust(20)

            print(f"{r['cohort']:<14}{r['seed']:>5}{r['n_subjects']:>5}"
                  f"{r['n_pos_subjects']:>5}  " + "".join(cell(nm) for nm in AGG_NAMES[:4]))
            env = r["bootstrap"].get("envelope")
            envtxt = ("-" if not env else
                      f"{_f(env['auc_best_observed'])} [{_f(env['ci_lo'],2)},{_f(env['ci_hi'],2)}]")
            print(f"{'':<29}  " + "".join(cell(nm) for nm in AGG_NAMES[4:])
                  + f"{envtxt:>22}")
            if r["bootstrap"].get("reason"):
                print(f"{'':<29}  ({r['bootstrap']['reason']})")

    v = res["verdict"]
    print("\n" + _rule(104, "="))
    print(f"VERDICT (A): {v['question']}")
    print(_rule(104, "="))
    print(f"  distinct schemes tried: {v['n_distinct_schemes']} "
          f"({v['n_schemes_reported']} reported; "
          f"{v['alias_schemes']} is an exact alias of max and is not counted)")
    print(f"  cohorts on trial: {v['clinical_cohorts']}   "
          f"reported but NOT part of the decision: "
          f"{v['confound_cohorts_excluded_from_the_decision']}")
    print(f"  positive control: {len(v['positive_control_hits_on_confound_cohorts'])} "
          "confound-cohort phase cell(s) clear chance under these schemes -- the sweep "
          "can see an effect when there is one")
    print(f"\n  single seed x single scheme cells above {CHANCE:.3f} "
          f"(clinical only): {len(v['per_seed_scheme_hits'])}")
    for L in v["per_seed_scheme_hits"]:
        print(f"    {L['cohort']} seed{L['seed']} {L['scheme']}: "
              f"{_f(L['auc'])} [{_f(L['ci_lo'])}, {_f(L['ci_hi'])}]")
    print(f"\n  under C1's own cross-seed rule ({v['c1_rule']}):")
    print(f"{'cohort':<16}" + "".join(f"{nm:>12}" for nm in AGG_NAMES))
    for coh in sorted(v["c1_style_min_ci_lo_over_seeds"]):
        per = v["c1_style_min_ci_lo_over_seeds"][coh]
        print(f"{coh:<16}" + "".join(
            f"{_f((per.get(nm) or {}).get('min_ci_lo_over_seeds')):>12}"
            for nm in AGG_NAMES))
    print(f"  any cohort clears under the C1 rule: "
          f"{'YES' if v['any_cohort_clears_under_c1_rule'] else 'NO'}")
    print("\n  selection-aware envelope lower bound, clinical cohorts (per seed):")
    for k, vs in sorted(v["envelope_ci_lo_by_clinical_cohort"].items()):
        print(f"    {k:<14} " + ", ".join(_f(x) for x in vs))
    print("\n  " + (v["statement"] or ""))


def print_coil(res: dict) -> None:
    print(_rule(104, "="))
    print("B. IS 'COIL COUNT' REALLY COIL COUNT, OR IS IT SITE?")
    print(_rule(104, "="))
    print(f"cohort={res['cohort']}  label={res['label_definition']}  "
          f"subjects={res['n_subjects']}  positive(>=16ch)={res['n_pos_subjects']}")
    print(f"coil counts, bucket 0 (<16): {res['coil_counts_by_bucket'][0]}")
    print(f"coil counts, bucket 1 (>=16): {res['coil_counts_by_bucket'][1]}")

    sn = res["site_normalisation"]
    print(f"\nsite normalisation: {sn['n_input_strings']} institution string(s) -> "
          f"{sn['n_sites']} site(s), by shared device_id")
    for c in sn["components"]:
        print(f"  {c['site']:<28} n={c['n_rows']:<5} <- {c['institution_strings']}")
    for e in sn["merge_evidence"]:
        for devid, strings in e["devices_shared_by_more_than_one_string"].items():
            print(f"    evidence: magnet {devid} recorded as {strings}")
    pv = res["prefix_vs_device_normalisation"]
    print(f"  prefix-rule partition identical to device-based partition: "
          f"{pv['identical_partition']}  "
          f"(prefix levels {pv['prefix_levels']} -> sites {pv['device_site_levels']})")

    print("\nassociation with the coil-count bucket "
          "(U = fraction of coil-count entropy removed by knowing the variable)")
    print(f"{'variable':<20}{'lv':>4}{'V':>7}{'V_corr':>8}{'V_null':>8}{'p':>8}"
          f"{'MI(bits)':>10}{'MI_null':>9}{'U':>7}")
    for name, a in res["associations"].items():
        cv, mi, pm = a["cramers_v"], a["mutual_information"], (a["permutation"] or {})
        print(f"{name:<20}{a['n_levels']:>4}{_f(cv['v']):>7}{_f(cv['v_bias_corrected']):>8}"
              f"{_f(pm.get('v_null_mean')):>8}{_f(pm.get('v_p')):>8}"
              f"{_f(mi['mi_bits']):>10}"
              f"{_f((pm.get('mi_null_mean_nats') or 0) / math.log(2)):>9}"
              f"{_f(mi['u_col_given_row']):>7}")

    print("\ncollinearity AMONG the context variables (Cramer's V)")
    for k, d in res["pairwise_context_associations"].items():
        print(f"  {k:<34} V={_f(d['cramers_v'])}  V_corr={_f(d['v_bias_corrected'])}")

    print(f"\nseparability over all {res['n_subjects']} subjects "
          f"(floor: >= {res['verdict']['floor_per_class']} subjects in EACH coil bucket)")
    print(f"{'variable':<20}{'strata':>7}{'span both':>11}{'meet floor':>12}"
          f"{'pairs kept':>12}")
    for name, s in res["separability"].items():
        print(f"{name:<20}{s['n_strata']:>7}{s['n_strata_spanning_both']:>11}"
              f"{s['n_strata_meeting_floor']:>12}"
              f"{_f(s['fraction_of_pairs_retained']):>12}")

    for sd, blk in sorted(res["model_within_stratum"].items(), key=lambda kv: int(kv[0])):
        if "strata" not in blk:
            print(f"\nseed {sd}: {blk.get('reason')}")
            continue
        print(f"\nphase model, seed {sd}: within-stratum AUC for coil count "
              f"({blk['n_test_subjects']} held-out subjects, "
              f"{blk['n_pos_test_subjects']} positive; patient-mean aggregation)")
        for name, st in blk["strata"].items():
            head = (f"  within {name:<19} unstratified={_f(st['unstratified_auc'])}  ")
            if st["stratified_auc"] is None:
                print(head + "stratified=NOT COMPUTABLE at the power floor")
                up = st.get("underpowered_all_spanning_strata") or {}
                if up.get("stratified_auc") is not None:
                    print(f"       (floor dropped to 1: {_f(up['stratified_auc'])} over "
                          f"{up['n_strata_used']} stratum/a, resting on "
                          f"{up['discordant_pairs_used']}/{up['discordant_pairs_total']} "
                          f"= {100 * up['fraction_of_pairs_retained']:.1f}% of the "
                          "discordant pairs -- NOT evidence)")
            else:
                print(head + f"stratified={_f(st['stratified_auc'])} "
                             f"[{_f(st['stratified_ci_lo'])}, {_f(st['stratified_ci_hi'])}] "
                             f"over {st.get('strata_used')}")
            # Always print the strata, including the ones that disqualify the
            # variable: "no scanner spans both coil buckets" is a claim, and a
            # claim has to be checkable at the point it is made.
            for lv, d in sorted(st["per_stratum"].items()):
                mark = "" if d["meets_floor"] else "   (below power floor)"
                print(f"       {lv:<26} AUC={_f(d['auc'])} "
                      f"[{_f(d['ci_lo'])}, {_f(d['ci_hi'])}]  "
                      f"n+={d['n_pos']:<4} n-={d['n_neg']:<4}{mark}")
            missing = [r for r in res["separability"][name]["strata"]
                       if not r["spans_both"]]
            if missing:
                txt = ", ".join(f"{r['stratum']}({r['n_pos']}+/{r['n_neg']}-)"
                                for r in missing)
                print(f"       [cohort-wide, spans only ONE coil bucket: {txt}]")

    v = res["verdict"]
    print("\n" + _rule(104, "="))
    print("VERDICT (B)")
    print(_rule(104, "="))
    print(f"  separable from:     {v['separable_from']}")
    print("  NOT separable from: "
          + ", ".join(f"{d['variable']}" for d in v["not_separable_from"]))
    for name, h in (v.get("within_stratum_heterogeneity") or {}).items():
        if h.get("note"):
            print(f"  caveat [{name}]: {h['note']}")
    print("\n  " + v["claim"])


# ==========================================================================
# Self-test
# ==========================================================================

class _Check:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def ok(self, cond, label, detail=""):
        if cond:
            self.passed += 1
        else:
            self.failed += 1
        print(f"  [{'PASS' if cond else 'FAIL'}] {label}" + (f"  {detail}" if detail else ""))
        return bool(cond)


def self_test(quick: bool = False) -> int:
    print(_rule(104, "="))
    print("s09_robustness self-test")
    print(_rule(104, "="))
    c = _Check()
    rng = np.random.default_rng(20260728)
    B = 300 if quick else 1500

    # ------------------------------------------------------- [1] aggregation
    print("\n[1] aggregation schemes")
    s = np.array([0.1, 0.2, 0.3, 0.9])
    c.ok(apply_aggregation(s, "mean") == 0.375, "mean is the arithmetic mean")
    c.ok(apply_aggregation(s, "max") == 0.9, "max is the maximum")
    c.ok(apply_aggregation(s, "top1_mean") == apply_aggregation(s, "max"),
         "top-1 mean IS max (so it is excluded from the multiplicity count)")
    c.ok(abs(apply_aggregation(s, "top2_mean") - 0.6) < 1e-12,
         "top-2 mean = (0.9+0.3)/2", f"{apply_aggregation(s, 'top2_mean')}")
    c.ok(apply_aggregation(s, "top5_mean") == apply_aggregation(s, "mean"),
         "top-k with k >= n falls back to the mean of what exists")
    c.ok(abs(apply_aggregation(s, "q75") - float(np.quantile(s, 0.75))) < 1e-12,
         "q75 is numpy's linear-interpolation quantile")
    one = np.array([0.42])
    c.ok(all(abs(apply_aggregation(one, nm) - 0.42) < 1e-12 for nm in AGG_NAMES),
         "every scheme returns the value itself for a one-slice subject")
    mono = np.array([0.1, 0.5, 0.7])
    c.ok(all(apply_aggregation(mono, nm) <= apply_aggregation(mono, "max") + 1e-12
             for nm in AGG_NAMES),
         "no scheme can exceed the maximum")

    # --------------------------------------------- [2] aggregate_matrix vs s04
    print("\n[2] aggregate_matrix agrees with stage 4")
    y = np.array([0, 0, 1, 1, 1, 0])
    p = np.array([.1, .9, .2, .8, .4, .3])
    cl = np.array(["a", "a", "b", "b", "c", "c"], dtype=object)
    A = aggregate_matrix(y, p, cl)
    m4 = s04.aggregate_by_cluster(y, p, cl, "mean")
    x4 = s04.aggregate_by_cluster(y, p, cl, "max")
    j = A["names"].index("mean")
    k = A["names"].index("max")
    c.ok(np.allclose(A["scores"][:, j], m4["scores"]),
         "mean column reproduces s04.aggregate_by_cluster(mean)")
    c.ok(np.allclose(A["scores"][:, k], x4["scores"]),
         "max column reproduces s04.aggregate_by_cluster(max)")
    c.ok(np.array_equal(A["labels"], m4["labels"]),
         "subject label is max(slice labels), as in stage 4")
    c.ok(A["n_mixed_label_clusters"] == m4["n_mixed_label_clusters"],
         "mixed-label subject count matches stage 4",
         f"{A['n_mixed_label_clusters']}")

    # ------------------------------------------------------ [3] auc_columns
    print("\n[3] vectorised many-column AUC")
    n = 200
    yy = rng.integers(0, 2, size=n)
    if len(np.unique(yy)) < 2:
        yy[0], yy[1] = 0, 1
    S = rng.normal(size=(n, 5)) + yy[:, None] * 0.4
    got = auc_columns(yy, S)
    want = np.array([s04.auc_midrank(yy, S[:, j]) for j in range(S.shape[1])])
    c.ok(np.allclose(got, want), "matches s04.auc_midrank column by column",
         f"max|diff|={np.abs(got - want).max():.2e}")
    tied = np.repeat([0.5, 0.5, 0.7, 0.7], 5)[:, None]
    ytie = np.array([0, 1] * 10)
    c.ok(abs(float(auc_columns(ytie, tied)[0]) - s04.auc_midrank(ytie, tied[:, 0])) < 1e-12,
         "mid-rank tie handling matches on an all-ties vector")
    c.ok(np.all(np.isnan(auc_columns(np.zeros(10, dtype=int), rng.normal(size=(10, 2))))),
         "single-class input returns NaN rather than a number")

    # ------------------------------------------------- [4] shared bootstrap
    print("\n[4] shared-resample bootstrap and the selection envelope")
    n = 120
    yb = np.array([0] * 60 + [1] * 60)
    Sb = np.column_stack([rng.normal(size=n) + yb * d for d in (0.0, 0.3, 0.9)])
    nb = ["mean", "max", "q75"]
    bs = shared_cluster_bootstrap(yb, Sb, nb, n_boot=B, seed=1, envelope_names=nb)
    lows = [bs["per_scheme"][nm]["ci_lo"] for nm in nb]
    c.ok(bs["envelope"]["ci_lo"] >= max(lows) - 1e-12,
         "envelope lower bound dominates every scheme's lower bound",
         f"env={bs['envelope']['ci_lo']:.3f} vs max={max(lows):.3f}")
    c.ok(all(bs["per_scheme"][nm]["ci_lo"] <= bs["per_scheme"][nm]["auc"]
             <= bs["per_scheme"][nm]["ci_hi"] for nm in nb),
         "each point estimate lies inside its own interval")
    c.ok(bs["per_scheme"]["mean"]["excludes_chance_above"] is False,
         "a genuinely null column does not exclude chance")
    c.ok(bs["per_scheme"]["q75"]["excludes_chance_above"] is True,
         "a genuinely strong column does exclude chance")
    bs2 = shared_cluster_bootstrap(yb, Sb, nb, n_boot=B, seed=1, envelope_names=nb)
    c.ok(bs["per_scheme"]["mean"]["ci_lo"] == bs2["per_scheme"]["mean"]["ci_lo"],
         "the bootstrap is deterministic given the seed")
    deg = shared_cluster_bootstrap(np.zeros(20, dtype=int), rng.normal(size=(20, 3)),
                                   nb, n_boot=B, seed=1)
    c.ok(deg["per_scheme"]["mean"]["auc"] is None and deg["reason"],
         "single-class input yields a reason, not a number", deg["reason"][:44])

    # a max aggregation really can beat mean when the signal is focal
    print("\n[5] the reviewer's premise is reproducible in simulation")
    npat, nsl = 60, 30
    ys, ps, cs = [], [], []
    for i in range(npat):
        pos = i >= npat // 2
        base = rng.normal(0, 1, size=nsl)
        lab = np.zeros(nsl, dtype=int)
        if pos:                                   # 2 focal slices out of 30
            lab[:2] = 1
            base[:2] += 3.0
        ys.append(lab)
        ps.append(base)
        cs.append([f"p{i}"] * nsl)
    A = aggregate_matrix(np.concatenate(ys), np.concatenate(ps),
                         np.asarray(np.concatenate(cs), dtype=object))
    a = auc_columns(A["labels"], A["scores"])
    im, ix = A["names"].index("mean"), A["names"].index("max")
    c.ok(a[ix] > a[im] + 0.05,
         "with 2 focal slices in 30, max beats mean by a wide margin -- so the "
         "sweep can detect dilution if it is there",
         f"mean={a[im]:.3f} max={a[ix]:.3f}")

    # ------------------------------------------------------ [6] association
    print("\n[6] Cramer's V and mutual information")
    perfect = np.array([[50.0, 0.0], [0.0, 50.0]])
    indep = np.array([[25.0, 25.0], [25.0, 25.0]])
    c.ok(abs(cramers_v(perfect)["v"] - 1.0) < 1e-12, "V = 1 for a perfect 2x2")
    c.ok(abs(cramers_v(indep)["v"]) < 1e-12, "V = 0 under exact independence")
    c.ok(abs(mutual_information(indep)["mi_nats"]) < 1e-12,
         "MI = 0 under exact independence")
    c.ok(abs(mutual_information(perfect)["u_col_given_row"] - 1.0) < 1e-12,
         "U(col|row) = 1 when row determines col")
    c.ok(abs(mutual_information(perfect)["mi_bits"] - 1.0) < 1e-12,
         "MI = 1 bit for a perfectly-associated balanced 2x2")
    c.ok(cramers_v(perfect)["v_bias_corrected"] <= cramers_v(perfect)["v"] + 1e-12,
         "the bias-corrected V never exceeds the raw V")
    xs = rng.integers(0, 4, size=300)
    zs = rng.integers(0, 2, size=300)
    a_ind = association(xs, zs, n_perm=200, seed=3)
    c.ok(a_ind["permutation"]["v_p"] > 0.05,
         "permutation test does not reject independence on independent data",
         f"p={a_ind['permutation']['v_p']:.3f}")
    c.ok(a_ind["permutation"]["v_null_mean"] > 0.0,
         "the null mean of V is strictly positive -- which is exactly why a raw V "
         "must never be read without it",
         f"V_null={a_ind['permutation']['v_null_mean']:.3f}")
    a_dep = association(zs, zs, n_perm=200, seed=3)
    c.ok(a_dep["permutation"]["v_p"] <= 1.0 / 201 + 1e-12,
         "a deterministic relationship hits the permutation resolution floor")

    # ------------------------------------------------ [7] site normalisation
    print("\n[7] institution -> site normalisation")
    c.ok(canon_institution(" nyu  langone,  cbi ") == "NYU LANGONE CBI",
         "canonicalisation folds case, punctuation and whitespace")
    c.ok(canon_institution(None) == "UNKNOWN" and canon_institution("") == "UNKNOWN",
         "missing institution becomes UNKNOWN rather than an empty level")
    inst = ["NYU", "NYU LANGONE CBI", "NYU", "TH", "TH RADIOLOGY", "SOLO"]
    dev = ["45219", "45219", "25077", "41964", "41964", "999"]
    sn = sites_from_devices(inst, dev)
    c.ok(sn["n_sites"] == 3, "three sites recovered from six string/device pairs",
         str([cc["site"] for cc in sn["components"]]))
    c.ok(sn["map"]["NYU LANGONE CBI"] == sn["map"]["NYU"],
         "two strings sharing magnet 45219 become one site")
    c.ok(sn["map"]["TH RADIOLOGY"] == sn["map"]["TH"],
         "two strings sharing magnet 41964 become one site")
    c.ok(sn["map"]["SOLO"] != sn["map"]["NYU"],
         "a string that shares no magnet is NOT merged, however it is spelled")
    sn2 = sites_from_devices(["A", "B"], ["1", "2"])
    c.ok(sn2["n_sites"] == 2, "distinct magnets keep distinct institutions apart")

    # ----------------------------------------------- [8] within-stratum AUC
    print("\n[8] separability and within-stratum AUC (Simpson's paradox)")
    sep = separability(["s1"] * 20 + ["s2"] * 20,
                       np.array([1] * 15 + [0] * 5 + [1] * 2 + [0] * 18),
                       min_per_class=5)
    c.ok(sep["n_strata_spanning_both"] == 2, "both strata span both classes")
    c.ok(sep["n_strata_meeting_floor"] == 1,
         "only the stratum with >= 5 per class clears the floor")
    # A perfectly null score inside each stratum, but stratum membership is
    # associated with the label AND shifts the score -- pooled AUC must move
    # away from 0.5 while the stratified estimate stays at it.
    nA, nB = 200, 200
    yA = np.array([1] * 100 + [0] * 100)
    yB = np.array([1] * 100 + [0] * 100)
    sA = rng.normal(0.0, 1.0, size=nA)
    sB = rng.normal(3.0, 1.0, size=nB)
    lab = np.concatenate([yA, yB])
    sc = np.concatenate([sA, sB])
    st = np.array(["A"] * nA + ["B"] * nB, dtype=object)
    # make stratum B mostly negative so the offset runs against the label
    keep = np.ones(nA + nB, dtype=bool)
    keep[np.flatnonzero((st == "B") & (lab == 1))[:80]] = False
    keep[np.flatnonzero((st == "A") & (lab == 0))[:80]] = False
    ws = within_stratum_auc(lab[keep], sc[keep], st[keep], n_boot=B, seed=5, floor=10)
    c.ok(ws["unstratified_auc"] < 0.45,
         "pooled AUC is dragged below chance by the between-stratum offset",
         f"{ws['unstratified_auc']:.3f}")
    c.ok(abs(ws["stratified_auc"] - 0.5) < 0.08,
         "the within-stratum estimate returns to chance, as it must",
         f"{ws['stratified_auc']:.3f} [{ws['stratified_ci_lo']:.3f}, "
         f"{ws['stratified_ci_hi']:.3f}]")
    c.ok(ws["stratified_ci_lo"] < 0.5 < ws["stratified_ci_hi"],
         "and its interval covers chance")
    none_ok = within_stratum_auc(np.array([1, 0, 1, 0]), np.array([.1, .2, .3, .4]),
                                 np.array(["a", "a", "b", "b"], dtype=object),
                                 n_boot=50, seed=1, floor=10)
    c.ok(none_ok["stratified_auc"] is None and "floor" in (none_ok["reason"] or ""),
         "an underpowered stratification refuses rather than inventing a number")

    print("\n" + _rule(104, "="))
    print(f"{c.passed} passed, {c.failed} failed")
    print(_rule(104, "="))
    return 0 if c.failed == 0 else 1


# ==========================================================================
# CLI
# ==========================================================================

def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PhaseDx stage 9: aggregation robustness and coil-vs-site separability")
    p.add_argument("--results-dir", default=str(_DEFAULT_RESULTS_DIR))
    p.add_argument("--cache-dir", default=str(_DEFAULT_CACHE_DIR))
    p.add_argument("--cohort-dir", default=str(_DEFAULT_COHORT_DIR))
    p.add_argument("--out-dir", default=str(_DEFAULT_OUT_DIR))
    p.add_argument("--stats-json", default=None,
                   help="stage-4 statistics.json for the mean/max cross-check "
                        "(default: <results-dir>/statistics.json)")
    p.add_argument("--only", choices=("both", "aggregation", "coil-vs-site"),
                   default="both")
    p.add_argument("--confound-cohort", default="brain")
    p.add_argument("--seeds", default="42,123")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--n-perm", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--min-per-class", type=int, default=10,
                   help="power floor: subjects required in EACH coil bucket for a "
                        "stratum to be usable")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--quiet", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test(quick=args.quick)

    results_dir = Path(args.results_dir)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stats_json = Path(args.stats_json) if args.stats_json else results_dir / "statistics.json"
    payload = {"generated": __import__("time").strftime("%Y-%m-%dT%H:%M:%S"),
               "module": "s09_robustness"}

    if args.only in ("both", "aggregation"):
        agg = run_aggregation_sensitivity(
            results_dir, Path(args.cache_dir), Path(args.cohort_dir),
            n_boot=args.n_boot, seed=args.seed, alpha=args.alpha,
            stats_json=stats_json)
        payload["aggregation_sensitivity"] = agg
        if not args.quiet:
            print_aggregation(agg)
            print()

    if args.only in ("both", "coil-vs-site"):
        seeds = tuple(int(s) for s in str(args.seeds).split(",") if s.strip())
        coil = run_coil_vs_site(
            Path(args.cache_dir), results_dir, cohort=args.confound_cohort,
            seeds=seeds, n_boot=args.n_boot, n_perm=args.n_perm,
            seed=args.seed, alpha=args.alpha, floor=args.min_per_class)
        payload["coil_vs_site"] = coil
        if not args.quiet:
            print_coil(coil)

    out = out_dir / "s09_robustness.json"
    out.write_text(json.dumps(payload, indent=2, default=_json_default))
    print(f"\nwrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
