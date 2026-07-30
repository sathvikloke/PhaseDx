#!/usr/bin/env python3
"""
s16_rankinversion.py
--------------------
Does the CHOICE OF EVALUATION UNIT change WHICH METHOD you would conclude is best?

The paper already shows that slice-level numbers are inflated relative to
patient-level numbers. Inflation is a caveat. The claim with consequences is
that the unit changes the ORDERING: if a slice-level benchmark ranks method A
above method B and a patient-level benchmark ranks B above A, then a slice-level
benchmark is not merely optimistic, it selects the wrong method, and every paper
that used one to justify an architectural choice inherits the error.

THE TRAP THIS FILE EXISTS TO AVOID
----------------------------------
A ranking computed on noisy estimates will differ between two units BY CHANCE.
Showing slice-rank != patient-rank proves nothing on its own: with 21 methods
whose AUCs sit inside one another's confidence intervals, ANY two re-estimates
disagree. The analysis therefore has to separate

    "the unit changes the ranking"   from   "the rankings are too noisy to be
                                              stable at either unit"

and the only honest way to do that is to measure the WITHIN-unit rank
variability on the same data and compare the between-unit disagreement against
it. That is what `stability_comparison` does, and if between-unit disagreement
does not exceed within-unit noise, this file says so in those words. A chance
reordering is not an inversion and is never reported as one.

CONSTRUCTION
------------
1. A METHOD is one (architecture, initialisation, condition) cell. Its estimate
   is the pooled out-of-fold prediction vector built by s04_stats.pool_folds --
   every subject tested exactly once, refusing to pool if that property fails.

2. All methods in a cohort are ALIGNED on cache_idx, so every method is scored
   on exactly the same slices of exactly the same subjects, and any slice whose
   score is non-finite for ANY method is dropped from ALL methods. Comparing
   rankings computed over different case sets would be meaningless.

3. ONE JOINT CLUSTERED BOOTSTRAP drives everything. Each replicate resamples
   SUBJECTS with replacement (the cluster unit -- 30 slices of one prostate are
   one patient measured 30 times, not 30 observations) and, from that single
   draw, recomputes the slice-level AUC and the patient-level AUC of EVERY
   method. Because the draw is shared, the two units and all methods move
   together exactly as they do in the real dependence structure, and every
   between-method / between-unit comparison below is properly paired.

   The estimator is s04_stats': `auc_midrank` for the AUC, `_cluster_index` for
   the subject grouping, `aggregate_by_cluster` for the patient-level collapse.
   Only the SHARING of one draw across methods and units is new, and the
   self-test checks the resulting intervals against s04's own
   `cluster_bootstrap_auc` and `cluster_bootstrap_diff`.

WHAT IS REPORTED, per cohort
----------------------------
  * the slice-level and patient-level ranking of every method;
  * Spearman rho and Kendall tau between the two rankings, with subject-resampled
    intervals; top-1 agreement and top-3 overlap;
  * the DECISIVE COMPARISON: the distribution of between-unit rank disagreement
    against the distribution of within-unit rank disagreement produced by
    resampling subjects alone, and a plain statement of whether between exceeds
    within;
  * named INVERSION PAIRS -- A > B at slice level, B > A at patient level, with
    BOTH orderings individually supported by the clustered bootstrap after Holm
    adjustment over every pair examined. If none survives, that is stated.

Usage:
    python pipeline/s16_rankinversion.py --self-test
    python pipeline/s16_rankinversion.py                       # every cohort
    python pipeline/s16_rankinversion.py --cohorts brain knee
"""
from __future__ import annotations

import argparse
import json
import logging
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s04_stats as S  # noqa: E402  -- the estimator layer; never reimplemented here

try:
    import common  # noqa: E402
    _OUT = common.OUT_ROOT
except Exception:  # pragma: no cover
    _OUT = Path(__file__).resolve().parent.parent / "pipeline_out"

_DEFAULT_ARCH_DIR = _OUT / "results_arch"
_DEFAULT_RESULTS_DIR = _OUT / "results"
_DEFAULT_CACHE_DIR = _OUT / "cache"
_DEFAULT_COHORT_DIR = _OUT / "cohorts"

logger = logging.getLogger("s16_rankinversion")

# Cohorts, in the order the paper discusses them. brain and knee are the
# confound cohorts where methods genuinely separate; the clinical cohorts sit at
# chance at patient level, which is exactly why an apparent inversion there
# deserves more suspicion, not less.
ALL_COHORTS = ["brain", "knee", "prostate_t2", "prostate_dwi", "breast"]

# A ranking is called stable enough to be worth interpreting only if resampling
# subjects reproduces it. These are the thresholds, stated up front rather than
# chosen after seeing the numbers.
STABLE_TAU = 0.50        # median Kendall tau between a resampled and the point ranking
STABLE_TOP1 = 0.50       # P(the resampled best method is the point best method)

# The five verdicts `stability_comparison` can return.
#
#   UNIT_CHANGES_RANK   between-unit rank disagreement exceeds the within-unit
#                       resampling noise floor at BOTH units. The evaluation unit,
#                       not sampling noise, is moving the ranking.
#   CANNOT_RANK         the whole order fails to reproduce under subject
#                       resampling, so no ranking claim holds at either unit.
#   CANNOT_NAME_WINNER  the broad order DOES reproduce, but the identity of the
#                       best method does not. Strong methods separate from weak
#                       ones; no winner can be named.
#   INCONCLUSIVE        the rankings are stable and the units agree within noise.
#   NO_ESTIMATE         no evaluable replicate, or every ranking fully tied.
#
# The first is the only one that supports the headline claim; the middle two are
# refusals, and a refusal is a finding to be reported, not a gap to be filled.
REFUSAL_VERDICTS = ("CANNOT_RANK", "CANNOT_NAME_WINNER")


# ==========================================================================
# Rank primitives
# ==========================================================================

def ranks_desc(values) -> np.ndarray:
    """
    Competition-free ranking with rank 1 = LARGEST value, ties averaged.

    Midranks come from s04's `compute_midrank` (ascending), applied to the
    negated values. Averaging ties matters: two methods with identical AUC must
    not be assigned an arbitrary order, or a coin flip inside the sort would be
    read downstream as a ranking disagreement.
    """
    v = np.asarray(values, dtype=float)
    return S.compute_midrank(-v)


def spearman_rho(a, b) -> float | None:
    """
    Spearman's rho between two RANK vectors: the Pearson correlation of the
    midranks. Computing it from midranks (rather than the 1 - 6*sum(d^2)/n(n^2-1)
    shortcut) is what makes it correct in the presence of ties, and ties are
    guaranteed here -- a ceilinged confound cohort produces identical AUCs.

    Returns None when either ranking is entirely tied, because rho is then 0/0
    and reporting it as 0.0 would claim disagreement that was never measured.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    if len(x) != len(y) or len(x) < 2:
        return None
    sx, sy = x.std(), y.std()
    if sx <= 0 or sy <= 0:
        return None
    return float(((x - x.mean()) * (y - y.mean())).mean() / (sx * sy))


def kendall_tau_b(a, b) -> float | None:
    """
    Kendall's tau-b between two rank vectors, with the standard tie correction

        tau_b = (C - D) / sqrt((n0 - n1) * (n0 - n2))

    where n0 = n(n-1)/2 and n1, n2 are the tie-pair counts in each vector.
    O(n^2), which for <= 21 methods is nothing, and unlike an O(n log n)
    merge-sort implementation there is no place for an off-by-one to hide.

    tau is the primary agreement measure in this file because it has a direct
    reading as a disagreement COUNT: (1 - tau)/2 is the fraction of method pairs
    ordered differently by the two rankings.
    """
    x = np.asarray(a, dtype=float)
    y = np.asarray(b, dtype=float)
    n = len(x)
    if len(y) != n or n < 2:
        return None
    conc = disc = n1 = n2 = 0
    for i in range(n):
        for j in range(i + 1, n):
            dx = x[i] - x[j]
            dy = y[i] - y[j]
            if dx == 0:
                n1 += 1
            if dy == 0:
                n2 += 1
            s = dx * dy
            if s > 0:
                conc += 1
            elif s < 0:
                disc += 1
    n0 = n * (n - 1) // 2
    denom = (n0 - n1) * (n0 - n2)
    if denom <= 0:
        return None
    return float((conc - disc) / np.sqrt(denom))


def top_k_set(values, k: int) -> set[int]:
    """
    Indices of the k largest values, ties broken by INCLUDING every tied member.

    A hard slice of argsort would silently pick one of two exactly-equal methods
    as 'in the top 3', which is the kind of arbitrary choice this whole file
    exists to expose rather than commit.
    """
    v = np.asarray(values, dtype=float)
    if len(v) <= k:
        return set(range(len(v)))
    cutoff = np.sort(v)[::-1][k - 1]
    return set(int(i) for i in np.flatnonzero(v >= cutoff))


def argmax_set(values) -> set[int]:
    """Indices attaining the maximum (a set, because ties are real)."""
    v = np.asarray(values, dtype=float)
    return set(int(i) for i in np.flatnonzero(v >= v.max()))


# ==========================================================================
# Method assembly
# ==========================================================================

def _init_of(run: dict) -> str:
    """imagenet vs scratch, from the archzoo block when present, else the model block."""
    az = run.get("archzoo") or {}
    if az.get("init"):
        return str(az["init"])
    return "imagenet" if (run.get("model") or {}).get("pretrained") else "scratch"


def _arch_of(run: dict) -> str:
    az = run.get("archzoo") or {}
    if az.get("arch"):
        return str(az["arch"])
    return str((run.get("model") or {}).get("arch", "?"))


def collect_methods(cohort: str, source_dirs, cache_dir: Path, cohort_dir: Path,
                    seeds=(42,)) -> tuple[list[dict], list[str], dict]:
    """
    Every (architecture, init, condition) cell available for `cohort`, as one
    pooled out-of-fold estimate each. Returns (methods, notes, cluster_map).

    Source trees are read IN ORDER and de-duplicated on
    (arch, init, condition, seed): the architecture zoo and the main sweep both
    contain resnet18/imagenet, and the two files are the same run written twice.
    Counting it as two methods would put a duplicated point into the ranking and
    guarantee a tie that means nothing. First tree wins; the duplicate is logged.

    Pooling is s04's `pool_folds`, which REFUSES when the every-subject-exactly-
    once property cannot be verified. A refusal drops the method and is recorded
    in the returned notes rather than being papered over.
    """
    notes: list[str] = []
    cmap = S.build_cluster_map(cohort, cache_dir, cohort_dir)
    if cmap.get("reason"):
        notes.append(f"cluster map: falling back to patient_id ({cmap['reason']})")
    cluster_map = cmap["map"]

    seen: dict[tuple, str] = {}
    methods: list[dict] = []
    for src in source_dirs:
        src = Path(src)
        if not src.is_dir():
            continue
        runs = [r for r in S.load_runs(src) if str(r.get("cohort")) == cohort]
        if not runs:
            continue
        # exclude run_tagged's per-run temp tree, which can hold an in-flight payload
        runs = [r for r in runs if "/_scratch/" not in str(r.get("_path", ""))]
        # s04's `unit_key` is (cohort, region, split family, condition, seed) and
        # carries NO architecture, because in the main sweep there is only one. Here
        # there are seven, so arch/init are appended: without them two architectures
        # sharing a directory would have their folds pooled into one another, and
        # `pool_folds` would either refuse (fold 0 twice) or, if the trees were laid
        # out differently, quietly build a chimaera. The current layout is one
        # architecture per directory, so this changes nothing today; it is here so
        # that a future tree cannot make it change silently.
        units: dict[tuple, list[dict]] = {}
        for r in runs:
            units.setdefault(S.unit_key(r) + (_arch_of(r), _init_of(r)), []).append(r)
        expected: dict[tuple, set] = {}
        for r in runs:
            if r.get("_fold") is not None:
                expected.setdefault((str(r.get("cohort")), str(r.get("region", "full"))),
                                    set()).add(int(r["_fold"]))
        for key, rs in sorted(units.items(), key=lambda kv: str(kv[0])):
            _, region, family, condition, sd, arch, init = key
            if seeds is not None and sd not in seeds:
                continue
            ident = (arch, init, condition, sd)
            if ident in seen:
                notes.append(
                    f"duplicate {arch}/{init}/{condition}/seed{sd} in {src.name} "
                    f"already taken from {seen[ident]}; ignored")
                continue
            if family == "cv":
                pooled, info = S.pool_folds(rs, cluster_map,
                                            expected.get((cohort, region)))
                if pooled is None:
                    notes.append(f"{arch}/{init}/{condition}: REFUSED to pool -- "
                                 f"{info['reason']}")
                    continue
                if info.get("missing_folds"):
                    notes.append(f"{arch}/{init}/{condition}: pooled over "
                                 f"{info['n_folds']} folds, missing "
                                 f"{info['missing_folds']}; dropped (not the full cohort)")
                    continue
                run = pooled
                n_folds = info["n_folds"]
            else:
                if len(rs) > 1:
                    notes.append(f"{arch}/{init}/{condition}: {len(rs)} files share one "
                                 "identity in a single-split layout; using the first")
                run = rs[0]
                n_folds = None
            seen[ident] = src.name
            methods.append({
                "name": f"{arch}/{init}/{condition}",
                "arch": arch, "init": init, "condition": condition, "seed": sd,
                "source": src.name, "n_folds": n_folds,
                "run": run,
            })
    return methods, notes, cluster_map


def build_scan_map(cohort: str, cache_dir: Path) -> dict:
    """
    cache_idx -> SCAN id (subject + acquisition file), for cohorts whose design is
    MATCHED PAIRS WITHIN SUBJECT.

    The knee confound is such a design: each of the 96 subjects contributes one
    CORPDFS_FBK acquisition (label 1) and one CORPD_FBK acquisition (label 0),
    five slices each. Collapsing that to one score per SUBJECT is not a
    patient-level evaluation, it is a destroyed one -- every subject then has
    label max(0, 1) = 1, the aggregated vector is single-class, and the AUC does
    not exist. The unit that carries the label is the SCAN; the subject remains
    the unit of dependence and therefore of resampling.

    Returns {"map": {idx: scan_id}, "reason": str | None}.
    """
    out = {"map": {}, "reason": None, "source": None}
    idx_path = Path(cache_dir) / f"{cohort}_index.csv"
    if not idx_path.exists():
        out["reason"] = f"no cache index at {idx_path}"
        return out
    rows = S._read_csv_dicts(idx_path)
    if not rows or "file" not in rows[0]:
        out["reason"] = f"{idx_path.name} has no file column"
        return out
    subj_col = "subject_id" if "subject_id" in rows[0] else "patient_id"
    out["map"] = {int(r["idx"]): f"{r[subj_col]}|{Path(r['file']).name}" for r in rows}
    out["source"] = f"{idx_path.name}:{subj_col}+file"
    return out


def align_methods(methods: list[dict], cluster_map: dict | None,
                  scan_map: dict | None = None) -> dict:
    """
    Put every method on the SAME slices, in the same order, with one subject id
    per slice.

    Two guards, both of which have to be refusals rather than repairs:
      * methods that scored different case sets cannot be ranked against one
        another at all, so the intersection is taken and the shortfall reported;
        a method missing more than nothing is dropped, not silently truncated.
      * labels must agree between methods for the same cache_idx. If they do
        not, the vectors are misaligned and every number downstream is fiction.

    Slices whose probability is non-finite for ANY method are dropped from ALL
    methods, so the ranking is over one common case set rather than a per-method
    one. The count is returned.
    """
    out = {"ok": False, "reason": None, "notes": [], "dropped_nonfinite": 0}
    if len(methods) < 2:
        out["reason"] = f"only {len(methods)} method(s); a ranking needs at least 2"
        return out

    idx_sets = []
    for m in methods:
        t = m["run"].get("test") or {}
        ci = t.get("cache_idx")
        if not ci:
            out["reason"] = f"{m['name']} has no cache_idx; cannot verify case identity"
            return out
        idx_sets.append(set(int(v) for v in ci))
    common = set.intersection(*idx_sets)
    if not common:
        out["reason"] = "the methods share no cases"
        return out

    keep_methods = []
    for m, s in zip(methods, idx_sets):
        if s != common:
            out["notes"].append(
                f"{m['name']} scored {len(s)} cases against a common set of "
                f"{len(common)} ({len(s ^ common)} non-shared); dropped rather than "
                "compared on a different test set")
            continue
        keep_methods.append(m)
    if len(keep_methods) < 2:
        out["reason"] = "fewer than 2 methods share a common case set"
        return out

    order = np.array(sorted(common), dtype=int)
    pos = {int(v): i for i, v in enumerate(order)}

    labels = None
    scores = np.empty((len(keep_methods), len(order)), dtype=float)
    pids = None
    for mi, m in enumerate(keep_methods):
        t = m["run"]["test"]
        ci = np.asarray(t["cache_idx"], dtype=int)
        perm = np.array([pos[int(v)] for v in ci], dtype=int)
        y = np.full(len(order), np.nan)
        p = np.full(len(order), np.nan)
        pid = np.empty(len(order), dtype=object)
        y[perm] = np.asarray(t["labels"], dtype=float)
        p[perm] = np.asarray(t["probs"], dtype=float)
        pid[perm] = np.asarray([str(v) for v in t["patient_ids"]], dtype=object)
        if labels is None:
            labels, pids = y, pid
        elif not np.array_equal(labels, y):
            out["reason"] = (f"labels disagree between {keep_methods[0]['name']} and "
                             f"{m['name']} for the same cache_idx; the prediction "
                             "vectors are misaligned")
            return out
        scores[mi] = p

    ok = np.isfinite(scores).all(axis=0) & np.isin(labels, (0.0, 1.0))
    out["dropped_nonfinite"] = int((~ok).sum())
    if out["dropped_nonfinite"]:
        out["notes"].append(
            f"{out['dropped_nonfinite']}/{len(order)} slices dropped from EVERY method "
            "(non-finite probability somewhere, or a non-binary label), so all methods "
            "are ranked over one common case set")
    order, labels, scores, pids = order[ok], labels[ok], scores[:, ok], pids[ok]
    if len(order) == 0 or len(np.unique(labels)) < 2:
        out["reason"] = "no usable rows, or a single-class case set; AUC is undefined"
        return out

    clusters, unit, source = S.resolve_clusters(
        {"patient_ids": list(pids), "cache_idx": order.tolist()}, cluster_map)
    uniq, groups = S._cluster_index(clusters)
    if len(uniq) < 2:
        out["reason"] = f"only {len(uniq)} {unit}(s); no between-subject resampling possible"
        return out

    # ---- the aggregated ("patient-level") unit ---------------------------
    # Default: one case per SUBJECT, collapsed with s04's own `aggregate_by_cluster`
    # so the numbers here are byte-for-byte the ones the headline tables use (mean
    # over a subject's slices; subject label = max over its slice labels).
    #
    # Fallback: a MATCHED-PAIRS cohort, where every subject carries both labels and
    # the subject-level collapse is single-class by construction. There the unit
    # that carries the label is the SCAN, and the subject stays the unit of
    # dependence. Falling back is a documented downgrade, never silent.
    def _collapse(case_ids):
        cs = np.empty((len(keep_methods), len(np.unique(case_ids))), dtype=float)
        cl = None
        nmix = 0
        for mi in range(len(keep_methods)):
            agg = S.aggregate_by_cluster(labels.astype(int), scores[mi], case_ids, how="mean")
            cs[mi] = agg["scores"]
            nmix = agg["n_mixed_label_clusters"]
            if cl is None:
                cl = agg["labels"]
            elif not np.array_equal(cl, agg["labels"]):  # pragma: no cover
                return None, None, None, "aggregated labels differ between methods"
        return cs, cl, nmix, None

    case_ids = clusters
    case_unit = unit
    case_scores, case_labels, n_mixed, err = _collapse(case_ids)
    if err:
        out["reason"] = err
        return out

    if len(np.unique(case_labels)) < 2:
        scans = None
        if scan_map:
            try:
                scans = np.asarray([scan_map[int(i)] for i in order], dtype=object)
            except KeyError:
                scans = None
        if scans is None:
            out["reason"] = (
                f"every {unit} has the same aggregated label, so a {unit}-level AUC is "
                "undefined, and no scan identifier is available to fall back on. This is "
                "a matched-pairs design: the label belongs to the scan, not the subject")
            return out
        cs, cl, nmix, err = _collapse(scans)
        if err or cl is None or len(np.unique(cl)) < 2:
            out["reason"] = (f"every {unit} AND every scan has the same aggregated label; "
                             "no aggregated AUC exists")
            return out
        per_subj = len(np.unique(scans)) / len(uniq)
        out["notes"].append(
            f"MATCHED-PAIRS design: every {unit} carries both labels, so collapsing to one "
            f"score per {unit} gives a single-class vector and no {unit}-level AUC exists. "
            f"The aggregated unit is therefore the SCAN ({len(np.unique(scans))} scans, "
            f"{per_subj:.1f} per {unit}); the {unit} remains the bootstrap cluster, so the "
            "two scans of one subject are never resampled as independent cases")
        case_ids, case_unit = scans, "scan"
        case_scores, case_labels, n_mixed = cs, cl, nmix

    # Which aggregated cases belong to which resampling cluster. For the default
    # subject-level unit this is one case per cluster; for the paired design it is
    # the two scans of that subject, which must move together in every replicate.
    case_uniq = np.unique(case_ids)
    case_pos = {v: i for i, v in enumerate(case_uniq)}
    cluster_of_case = np.empty(len(case_uniq), dtype=object)
    for ci, ci_id in zip(case_ids, clusters):
        cluster_of_case[case_pos[ci]] = ci_id
    case_of_cluster = [
        np.array(sorted({case_pos[c] for c in case_ids[g]}), dtype=int) for g in groups]

    out.update({
        "ok": True,
        "methods": keep_methods,
        "names": [m["name"] for m in keep_methods],
        "cache_idx": order,
        "labels": labels.astype(int),
        "scores": scores,
        "clusters": clusters,
        "cluster_unit": unit,
        "cluster_source": source,
        "subjects": uniq,
        "groups": groups,
        "case_unit": case_unit,
        "case_ids": case_uniq,
        "case_clusters": cluster_of_case,
        "case_of_cluster": case_of_cluster,
        "subj_labels": case_labels,
        "subj_scores": case_scores,
        "n_slices": int(len(order)),
        "n_pos_slices": int((labels == 1).sum()),
        "n_subjects": int(len(uniq)),
        "n_cases": int(len(case_uniq)),
        "n_pos_cases": int(case_labels.sum()),
        "n_pos_subjects": int(case_labels.sum()) if case_unit == unit else None,
        "n_mixed_label_subjects": int(n_mixed),
    })
    return out


# ==========================================================================
# The joint clustered bootstrap
# ==========================================================================

def joint_bootstrap(al: dict, n_boot: int = 2000, seed: int = 0) -> dict:
    """
    ONE subject-resampled draw per replicate, shared by every method and BOTH
    units.

    Why shared: the question is whether the unit changes the ranking. If the
    slice-level and patient-level replicates came from different draws, part of
    the between-unit disagreement would be nothing but two independent samples,
    and the comparison against within-unit noise would be rigged in favour of
    finding an inversion. Sharing the draw makes every between-unit and
    between-method contrast PAIRED on the same subjects.

    A patient drawn twice contributes all of their slices twice at slice level
    and appears twice in the patient-level vector -- which is the point: that is
    what the dependence structure does to both estimates simultaneously.

    Replicates that are degenerate at EITHER unit (fewer than 2 distinct
    subjects drawn, or a single-class label vector) are skipped WHOLE, so the
    slice and patient columns always refer to the same set of draws and can be
    differenced. They are counted, not hidden.
    """
    labels, scores = al["labels"], al["scores"]
    groups, subj_labels, subj_scores = al["groups"], al["subj_labels"], al["subj_scores"]
    # Which aggregated cases a drawn cluster brings with it. One case per cluster
    # in the usual layout; both scans of a subject in a matched-pairs cohort, where
    # they must move together or the pairing would be broken by the resampling.
    case_of_cluster = al.get("case_of_cluster") or [np.array([i]) for i in range(len(groups))]
    n_meth, k = scores.shape[0], len(groups)
    rng = np.random.default_rng(seed)

    sl_rows, pat_rows = [], []
    n_single_cluster = n_single_class_slice = n_single_class_pat = n_nonfinite = 0
    for _ in range(int(n_boot)):
        draw = rng.integers(0, k, size=k)
        if len(np.unique(draw)) < 2:
            n_single_cluster += 1
            continue
        cases = np.concatenate([case_of_cluster[d] for d in draw])
        yp = subj_labels[cases]
        if yp.min() == yp.max():
            n_single_class_pat += 1
            continue
        rows = np.concatenate([groups[d] for d in draw])
        ys = labels[rows]
        if ys.min() == ys.max():
            n_single_class_slice += 1
            continue
        a_s = np.array([S.auc_midrank(ys, scores[m][rows]) for m in range(n_meth)])
        a_p = np.array([S.auc_midrank(yp, subj_scores[m][cases]) for m in range(n_meth)])
        if not (np.isfinite(a_s).all() and np.isfinite(a_p).all()):
            n_nonfinite += 1
            continue
        sl_rows.append(a_s)
        pat_rows.append(a_p)

    return {
        "auc_slice": np.asarray(sl_rows) if sl_rows else np.zeros((0, n_meth)),
        "auc_patient": np.asarray(pat_rows) if pat_rows else np.zeros((0, n_meth)),
        "n_boot_requested": int(n_boot),
        "n_boot_used": len(sl_rows),
        "n_skipped_single_cluster": n_single_cluster,
        "n_skipped_single_class_slice": n_single_class_slice,
        "n_skipped_single_class_patient": n_single_class_pat,
        "n_skipped_nonfinite": n_nonfinite,
        "seed": int(seed),
    }


def _pct(v, q):
    return float(np.quantile(np.asarray(v, dtype=float), q)) if len(v) else None


def split_half_stability(al: dict, n_rep: int = 200, seed: int = 0,
                         alpha: float = 0.05) -> dict:
    """
    Would a DIFFERENT cohort of this kind reproduce the ranking?

    Split the subjects into two disjoint halves, rank the methods at one unit
    inside each half, and measure the Kendall tau between the two rankings.
    Repeat over random splits.

    This is a second, sharper reference than the bootstrap-vs-point stability
    reported by `stability_comparison`, and it exists because that one has a
    known blind spot: a bootstrap resample shares roughly 63% of its subjects
    with the sample the point ranking was computed on, so it partly reproduces
    the point ranking even when the methods are IDENTICAL and the ordering is
    pure noise. Two disjoint halves share nothing, so a tau near zero here means
    the ordering carries no information -- which is precisely the claim the paper
    needs to be able to make, and to make honestly.

    The cost is that each half has half the subjects, so this number is
    PESSIMISTIC as an estimate of the full-cohort ranking's reproducibility. It
    is reported as a bound, not as the headline, and both are printed.

    Splits in which either half is single-class at either unit are skipped and
    counted; with 45 subjects and 24 positives that is rare, but it is not
    impossible and it must not be silently averaged over.
    """
    labels, scores = al["labels"], al["scores"]
    groups, case_labels, case_scores = al["groups"], al["subj_labels"], al["subj_scores"]
    case_of_cluster = al.get("case_of_cluster") or [np.array([i]) for i in range(len(groups))]
    n_meth, k = scores.shape[0], len(groups)
    rng = np.random.default_rng(seed + 4242)
    out = {"n_rep_requested": int(n_rep), "n_rep_used": 0, "n_skipped": 0,
           "tau_slice": [], "tau_agg": []}
    if k < 4:
        out["reason"] = f"only {k} clusters; a split-half needs at least 4"
        return out

    tau_s, tau_a = [], []
    for _ in range(int(n_rep)):
        perm = rng.permutation(k)
        halves = (perm[: k // 2], perm[k // 2:])
        ok = True
        ranks_s, ranks_a = [], []
        for h in halves:
            rows = np.concatenate([groups[d] for d in h])
            cases = np.concatenate([case_of_cluster[d] for d in h])
            ys, ya = labels[rows], case_labels[cases]
            if ys.min() == ys.max() or ya.min() == ya.max():
                ok = False
                break
            ranks_s.append(ranks_desc(
                [S.auc_midrank(ys, scores[m][rows]) for m in range(n_meth)]))
            ranks_a.append(ranks_desc(
                [S.auc_midrank(ya, case_scores[m][cases]) for m in range(n_meth)]))
        if not ok:
            out["n_skipped"] += 1
            continue
        t_s = kendall_tau_b(ranks_s[0], ranks_s[1])
        t_a = kendall_tau_b(ranks_a[0], ranks_a[1])
        if t_s is not None:
            tau_s.append(t_s)
        if t_a is not None:
            tau_a.append(t_a)
    out["n_rep_used"] = len(tau_s)
    for key, v in (("slice", tau_s), ("agg", tau_a)):
        out[f"tau_{key}_median"] = _pct(v, 0.5)
        out[f"tau_{key}_lo"] = _pct(v, alpha / 2)
        out[f"tau_{key}_hi"] = _pct(v, 1 - alpha / 2)
    out.pop("tau_slice", None)
    out.pop("tau_agg", None)
    return out


def point_estimates(al: dict) -> dict:
    """Full-data pooled AUC of every method at each unit, plus the two rankings."""
    n = al["scores"].shape[0]
    auc_s = np.array([S.auc_midrank(al["labels"], al["scores"][m]) for m in range(n)])
    auc_p = np.array([S.auc_midrank(al["subj_labels"], al["subj_scores"][m])
                      for m in range(n)])
    return {
        "auc_slice": auc_s, "auc_patient": auc_p,
        "rank_slice": ranks_desc(auc_s), "rank_patient": ranks_desc(auc_p),
    }


# ==========================================================================
# Rank agreement between the two units
# ==========================================================================

def rank_agreement(pt: dict, bs: dict, alpha: float = 0.05) -> dict:
    """
    Spearman rho and Kendall tau between the slice-level and the patient-level
    ranking, with percentile intervals from the SUBJECT-resampled replicates, so
    the interval reflects the actual clustering rather than pretending slices
    are independent.

    Also top-1 agreement and top-3 overlap, each with the fraction of replicates
    in which it holds -- the point estimate of 'the best method is the same
    under both units' is a single bit, and a single bit with no stability
    attached is not evidence.
    """
    rs_obs, rp_obs = pt["rank_slice"], pt["rank_patient"]
    out = {
        "rho_obs": spearman_rho(rs_obs, rp_obs),
        "tau_obs": kendall_tau_b(rs_obs, rp_obs),
        "top1_slice": sorted(argmax_set(pt["auc_slice"])),
        "top1_patient": sorted(argmax_set(pt["auc_patient"])),
        "top3_slice": sorted(top_k_set(pt["auc_slice"], 3)),
        "top3_patient": sorted(top_k_set(pt["auc_patient"], 3)),
        "n_boot_used": bs["n_boot_used"],
    }
    out["top1_same_obs"] = bool(set(out["top1_slice"]) & set(out["top1_patient"]))
    out["top3_overlap_obs"] = len(set(out["top3_slice"]) & set(out["top3_patient"]))

    A, B = bs["auc_slice"], bs["auc_patient"]
    rhos, taus, t1, t3 = [], [], [], []
    for b in range(len(A)):
        r_s, r_p = ranks_desc(A[b]), ranks_desc(B[b])
        rr, tt = spearman_rho(r_s, r_p), kendall_tau_b(r_s, r_p)
        if rr is not None:
            rhos.append(rr)
        if tt is not None:
            taus.append(tt)
        t1.append(bool(argmax_set(A[b]) & argmax_set(B[b])))
        t3.append(len(top_k_set(A[b], 3) & top_k_set(B[b], 3)))
    out["rho_ci"] = [_pct(rhos, alpha / 2), _pct(rhos, 1 - alpha / 2)]
    out["rho_median"] = _pct(rhos, 0.5)
    out["tau_ci"] = [_pct(taus, alpha / 2), _pct(taus, 1 - alpha / 2)]
    out["tau_median"] = _pct(taus, 0.5)
    out["top1_same_rate"] = float(np.mean(t1)) if t1 else None
    out["top3_overlap_mean"] = float(np.mean(t3)) if t3 else None
    out["top3_overlap_ci"] = [_pct(t3, alpha / 2), _pct(t3, 1 - alpha / 2)]
    return out


# ==========================================================================
# THE DECISIVE COMPARISON: between-unit disagreement vs within-unit noise
# ==========================================================================

def stability_comparison(pt: dict, bs: dict, alpha: float = 0.05,
                         stable_tau: float = STABLE_TAU,
                         stable_top1: float = STABLE_TOP1,
                         split_half: dict | None = None) -> dict:
    """
    Is the between-unit rank disagreement LARGER than the disagreement noise
    alone produces?

    Disagreement is measured as D = 1 - tau_b, so D = 0 is an identical ordering
    and D = 2 is a perfect reversal; (D/2) is the fraction of method pairs
    ordered differently.

    Three quantities, from the SAME replicates:

      D_between(b) = 1 - tau(rank_slice(b), rank_patient(b))
          how far apart the two units put the methods, in replicate b.

      D_within_slice(b)   = 1 - tau(rank_slice(b),   rank_slice(point))
      D_within_patient(b) = 1 - tau(rank_patient(b), rank_patient(point))
          how far the ranking moves when the UNIT IS HELD FIXED and only the
          sampled subjects change. This is the noise floor: the amount of
          reordering you get for free.

    The headline is the PAIRED difference

      delta_u(b) = D_between(b) - D_within_u(b)

    computed inside each replicate, so the two terms share a draw and the
    interval is a paired one. The unit is declared to change the ranking only if
    the interval for delta lies entirely above zero against BOTH references. The
    exceedance probability P(D_within >= D_between_obs) is reported alongside:
    it is the fraction of noise-only perturbations that move the ranking at least
    as far as switching the unit does, and if it is large the honest reading is
    that this benchmark cannot rank methods at either unit.

    That last reading is not a failure of the analysis. It is a finding, and the
    verdict string says so rather than reaching for the more quotable answer.
    """
    A, B = bs["auc_slice"], bs["auc_patient"]
    rs_obs, rp_obs = pt["rank_slice"], pt["rank_patient"]
    tau_obs = kendall_tau_b(rs_obs, rp_obs)
    out = {
        "d_between_obs": None if tau_obs is None else 1.0 - tau_obs,
        "n_boot_used": int(len(A)),
        "stable_tau_threshold": stable_tau,
        "stable_top1_threshold": stable_top1,
    }
    if len(A) == 0:
        out["verdict"] = "NO_ESTIMATE"
        out["verdict_text"] = "no evaluable bootstrap replicates; nothing can be said"
        return out

    d_between, d_ws, d_wp, tau_ws, tau_wp = [], [], [], [], []
    t1_stab_s, t1_stab_p = [], []
    top1_s_obs, top1_p_obs = argmax_set(pt["auc_slice"]), argmax_set(pt["auc_patient"])
    for b in range(len(A)):
        r_s, r_p = ranks_desc(A[b]), ranks_desc(B[b])
        t_bt = kendall_tau_b(r_s, r_p)
        t_ws = kendall_tau_b(r_s, rs_obs)
        t_wp = kendall_tau_b(r_p, rp_obs)
        if None in (t_bt, t_ws, t_wp):
            continue
        d_between.append(1.0 - t_bt)
        d_ws.append(1.0 - t_ws)
        d_wp.append(1.0 - t_wp)
        tau_ws.append(t_ws)
        tau_wp.append(t_wp)
        t1_stab_s.append(bool(argmax_set(A[b]) & top1_s_obs))
        t1_stab_p.append(bool(argmax_set(B[b]) & top1_p_obs))
    if not d_between:
        out["verdict"] = "NO_ESTIMATE"
        out["verdict_text"] = ("every replicate produced a fully tied ranking; tau is "
                               "undefined and no comparison is possible")
        return out

    d_between = np.asarray(d_between)
    d_ws = np.asarray(d_ws)
    d_wp = np.asarray(d_wp)
    delta_s = d_between - d_ws
    delta_p = d_between - d_wp

    def summ(v):
        return {"median": _pct(v, 0.5), "lo": _pct(v, alpha / 2), "hi": _pct(v, 1 - alpha / 2),
                "mean": float(np.mean(v))}

    out["d_between"] = summ(d_between)
    out["d_within_slice"] = summ(d_ws)
    out["d_within_patient"] = summ(d_wp)
    def _boot_p(d):
        """
        Two-sided bootstrap p for 'delta differs from zero', in s04's convention:
        twice the smaller tail mass, floored at the bootstrap's own resolution.
        Replicates landing exactly on zero are counted in BOTH tails, which is
        the conservative direction -- delta = 0 means the unit moved the ranking
        exactly as far as noise did, and that is not evidence for the unit. This
        matters because tau over a handful of methods is a discrete statistic
        with real mass at zero, so an interval-endpoint test is brittle here.
        """
        tail = min(float((d <= 0).mean()), float((d >= 0).mean()))
        return float(min(1.0, max(2.0 * tail, 1.0 / len(d))))

    out["delta_vs_slice"] = summ(delta_s) | {"p_gt0": float(np.mean(delta_s > 0)),
                                             "p": _boot_p(delta_s)}
    out["delta_vs_patient"] = summ(delta_p) | {"p_gt0": float(np.mean(delta_p > 0)),
                                               "p": _boot_p(delta_p)}
    out["tau_within_slice_median"] = _pct(tau_ws, 0.5)
    out["tau_within_slice_lo"] = _pct(tau_ws, alpha / 2)
    out["tau_within_patient_median"] = _pct(tau_wp, 0.5)
    out["tau_within_patient_lo"] = _pct(tau_wp, alpha / 2)
    out["top1_stability_slice"] = float(np.mean(t1_stab_s))
    out["top1_stability_patient"] = float(np.mean(t1_stab_p))

    dbo = out["d_between_obs"]
    if dbo is not None:
        out["p_exceed_within_slice"] = float(np.mean(d_ws >= dbo))
        out["p_exceed_within_patient"] = float(np.mean(d_wp >= dbo))
        # the conservative one: between must beat the LARGER noise floor
        out["p_exceed"] = max(out["p_exceed_within_slice"], out["p_exceed_within_patient"])
    else:
        out["p_exceed_within_slice"] = out["p_exceed_within_patient"] = out["p_exceed"] = None

    # ---- verdict ---------------------------------------------------------
    # Two kinds of instability, kept apart because they mean different things.
    # A WHOLE-ORDER failure (low tau to the point ranking) means the benchmark
    # cannot separate methods at all. A TOP-1 failure with a reproducible whole
    # order means it can tell strong methods from weak ones but cannot name a
    # winner -- which is the situation on the confound cohorts, and reporting it
    # as if the ranking were pure noise would overstate the finding.
    order_flags, top1_flags = [], []
    if (out["tau_within_slice_median"] or 0) < stable_tau:
        order_flags.append(f"slice (median tau to its own point ranking "
                           f"{out['tau_within_slice_median']:.2f} < {stable_tau})")
    if (out["tau_within_patient_median"] or 0) < stable_tau:
        order_flags.append(f"aggregated (median tau "
                           f"{out['tau_within_patient_median']:.2f} < {stable_tau})")
    # The split-half reference, when available. It is the decisive one for
    # "would another cohort rank these the same way": disjoint halves share no
    # subjects, so unlike the bootstrap-vs-point tau it cannot be propped up by
    # sample overlap when the methods are in truth indistinguishable.
    sh = split_half or {}
    out["split_half"] = sh or None
    for key, label in (("slice", "slice"), ("agg", "aggregated")):
        v = sh.get(f"tau_{key}_median")
        if v is not None and v < stable_tau:
            order_flags.append(
                f"{label} split-half tau {v:.2f} < {stable_tau} (two disjoint halves of "
                "the subjects do not rank the methods the same way)")
    if out["top1_stability_slice"] < stable_top1:
        top1_flags.append(f"slice top-1 reproduces in only "
                          f"{out['top1_stability_slice']*100:.0f}% of resamples")
    if out["top1_stability_patient"] < stable_top1:
        top1_flags.append(f"aggregated top-1 reproduces in only "
                          f"{out['top1_stability_patient']*100:.0f}% of resamples")
    unstable = order_flags + top1_flags
    out["instability_flags"] = unstable
    out["order_instability_flags"] = order_flags
    out["top1_instability_flags"] = top1_flags

    # Which noise floor the between-unit disagreement clears. Both must be cleared
    # before the unit can be blamed: clearing only one is the signature of a
    # ranking that is noise AT that one unit, which is a different finding and
    # must not be dressed up as an inversion.
    exc_s = bool(out["delta_vs_slice"]["median"] > 0
                 and out["delta_vs_slice"]["p"] < alpha)
    exc_p = bool(out["delta_vs_patient"]["median"] > 0
                 and out["delta_vs_patient"]["p"] < alpha)
    out["exceeds_slice_noise_floor"] = exc_s
    out["exceeds_agg_noise_floor"] = exc_p
    between_exceeds = exc_s and exc_p
    out["between_exceeds_within"] = bool(between_exceeds)

    if unstable and not between_exceeds:
        if order_flags:
            out["verdict"] = "CANNOT_RANK"
            head = ("the ranking does not reproduce under subject resampling: "
                    + "; ".join(unstable) + ". ")
            tail_claim = ("this benchmark cannot rank these methods at either unit")
        else:
            out["verdict"] = "CANNOT_NAME_WINNER"
            head = (
                "the broad ordering DOES reproduce under subject resampling (median tau "
                f"to its own point ranking: slice {out['tau_within_slice_median']:.2f}, "
                f"aggregated {out['tau_within_patient_median']:.2f}), but the identity of "
                "the BEST method does not -- " + "; ".join(top1_flags) + ". The benchmark "
                "can separate strong methods from weak ones without being able to name a "
                "winner at either unit. ")
            tail_claim = ("the reordering between units is not attributable to the unit")
        if not exc_s and not exc_p:
            body = (
                "Between-unit disagreement does not exceed the disagreement that "
                "resampling subjects produces on its own at EITHER unit "
                f"(P(within >= observed between) = {out['p_exceed']:.3f}), so "
                + tail_claim + ".")
        else:
            one, other = ("slice", "aggregated") if exc_s else ("aggregated", "slice")
            weak_tau = (out["tau_within_patient_median"] if exc_s
                        else out["tau_within_slice_median"])
            weak_t1 = (out["top1_stability_patient"] if exc_s
                       else out["top1_stability_slice"])
            body = (
                f"Between-unit disagreement (D = {out['d_between_obs']:.3f}) does exceed "
                f"the {one}-level noise floor, but NOT the {other}-level one, whose paired "
                "delta spans zero. That asymmetry is what a ranking with no ordering "
                f"information looks like: at the {other} unit the ranking reproduces "
                f"itself at only tau = {weak_tau:.2f} under subject resampling and its "
                f"top-1 method reproduces in {weak_t1*100:.0f}% of resamples. The two "
                "units disagree because one of them is noise, not because they encode "
                "two different orderings, so no ranking claim is supported at either "
                "unit.")
        out["verdict_text"] = head + body + (
            " Any apparent slice-vs-aggregated reordering here is a chance reordering "
            "and must not be reported as an inversion.")
    elif between_exceeds:
        out["verdict"] = "UNIT_CHANGES_RANK"
        out["verdict_text"] = (
            "between-unit rank disagreement exceeds within-unit resampling noise "
            f"(paired delta vs slice {out['delta_vs_slice']['median']:.3f} "
            f"[{out['delta_vs_slice']['lo']:.3f}, {out['delta_vs_slice']['hi']:.3f}], "
            f"vs patient {out['delta_vs_patient']['median']:.3f} "
            f"[{out['delta_vs_patient']['lo']:.3f}, {out['delta_vs_patient']['hi']:.3f}]"
            "): the evaluation unit, not sampling noise, is moving the ranking."
            + (" NOTE: the within-unit rankings are themselves unstable ("
               + "; ".join(unstable) + "), so the ordering at each unit is poorly "
               "determined even though the two units differ." if unstable else ""))
    else:
        out["verdict"] = "INCONCLUSIVE"
        out["verdict_text"] = (
            "the within-unit rankings are stable, but between-unit disagreement is "
            "not distinguishable from within-unit resampling noise "
            f"(P(within >= between_obs) = {out['p_exceed']:.3f}); no claim that the "
            "unit changes the ranking is supported.")
    return out


# ==========================================================================
# Inversion pairs
# ==========================================================================

def _boot_diff_ci(col_a, col_b, alpha):
    """
    Percentile interval and two-sided p for a paired difference read off the
    joint bootstrap columns. Same construction as s04's `cluster_bootstrap_diff`
    -- same clustered resample, same statistic, same p-value floor at the
    bootstrap's own resolution -- differing only in that the draw is shared
    across all methods so every pair is paired on identical subjects.
    """
    d = np.asarray(col_a) - np.asarray(col_b)
    if len(d) < 20:
        return {"ci_lo": None, "ci_hi": None, "p": None, "n": int(len(d))}
    tail = min(float((d <= 0).mean()), float((d >= 0).mean()))
    return {"ci_lo": _pct(d, alpha / 2), "ci_hi": _pct(d, 1 - alpha / 2),
            "p": float(min(1.0, max(2.0 * tail, 1.0 / len(d)))), "n": int(len(d))}


def inversion_pairs(al: dict, pt: dict, bs: dict, alpha: float = 0.05,
                    verify: bool = True, n_boot_verify: int = 2000,
                    seed: int = 0) -> dict:
    """
    Named pairs (A, B) with A > B at slice level and B > A at patient level,
    where BOTH orderings are individually well supported.

    The bar, in order:
      1. the point estimates disagree in sign between the two units;
      2. the paired clustered-bootstrap interval for the difference excludes
         zero at slice level AND at patient level;
      3. after HOLM adjustment over EVERY pair examined at that unit -- not just
         the ones that flipped -- both p-values stay below alpha.

    Step 3 is what stops this from being a fishing expedition. With 21 methods
    there are 210 pairs and two units; at alpha = 0.05 roughly 21 of them would
    clear step 2 in each direction by chance alone. The Holm family is therefore
    every pair, and p-values for every pair are computed, not just for the
    candidates.

    `verify` re-runs the surviving and candidate pairs through s04's own
    `cluster_bootstrap_diff` with an independent RNG stream. That is a check on
    this file, not a second estimator: the two are the same construction, and a
    disagreement beyond Monte Carlo error means something here is wrong.
    """
    names = al["names"]
    n = len(names)
    A, B = bs["auc_slice"], bs["auc_patient"]
    out = {"n_methods": n, "n_pairs_examined": n * (n - 1) // 2,
           "candidates": [], "survivors": [], "verified": [], "notes": []}
    if len(A) < 20:
        out["notes"].append(f"only {len(A)} evaluable replicates; no pair test is reported")
        return out

    pairs = list(combinations(range(n), 2))
    ds, dp = {}, {}
    for (i, j) in pairs:
        ds[(i, j)] = _boot_diff_ci(A[:, i], A[:, j], alpha)
        dp[(i, j)] = _boot_diff_ci(B[:, i], B[:, j], alpha)

    # Holm over the WHOLE family of pairs, per unit.
    holm_s = dict(zip(pairs, S.holm_adjust([ds[p]["p"] for p in pairs])))
    holm_p = dict(zip(pairs, S.holm_adjust([dp[p]["p"] for p in pairs])))

    auc_s, auc_p = pt["auc_slice"], pt["auc_patient"]
    for (i, j) in pairs:
        d_s = float(auc_s[i] - auc_s[j])
        d_p = float(auc_p[i] - auc_p[j])
        if d_s == 0 or d_p == 0 or (d_s > 0) == (d_p > 0):
            continue
        # orient so that A is the slice-level winner
        a, b = (i, j) if d_s > 0 else (j, i)
        sgn = 1.0 if d_s > 0 else -1.0
        rec = {
            "A": names[a], "B": names[b],
            "auc_slice_A": float(auc_s[a]), "auc_slice_B": float(auc_s[b]),
            "auc_patient_A": float(auc_p[a]), "auc_patient_B": float(auc_p[b]),
            "slice_diff": abs(d_s), "patient_diff": abs(d_p),
            "slice": {k: (v if k not in ("ci_lo", "ci_hi") else
                          (None if v is None else sgn * v))
                      for k, v in ds[(i, j)].items()},
            "patient": {k: (v if k not in ("ci_lo", "ci_hi") else
                            (None if v is None else sgn * v))
                        for k, v in dp[(i, j)].items()},
            "holm_p_slice": holm_s[(i, j)],
            "holm_p_patient": holm_p[(i, j)],
            "_idx": (a, b),
        }
        for side in ("slice", "patient"):
            lo, hi = rec[side]["ci_lo"], rec[side]["ci_hi"]
            if lo is not None and hi is not None and lo > hi:
                rec[side]["ci_lo"], rec[side]["ci_hi"] = hi, lo
        rec["ci_excludes_zero_slice"] = bool(
            rec["slice"]["ci_lo"] is not None and rec["slice"]["ci_lo"] > 0)
        rec["ci_excludes_zero_patient"] = bool(
            rec["patient"]["ci_hi"] is not None and rec["patient"]["ci_hi"] < 0)
        rec["survives"] = bool(
            rec["ci_excludes_zero_slice"] and rec["ci_excludes_zero_patient"]
            and (rec["holm_p_slice"] or 1) < alpha
            and (rec["holm_p_patient"] or 1) < alpha)
        out["candidates"].append(rec)

    out["candidates"].sort(key=lambda r: -(r["slice_diff"] + r["patient_diff"]))
    out["survivors"] = [r for r in out["candidates"] if r["survives"]]

    if verify and out["candidates"]:
        # independent RNG stream, s04's estimator, on the strongest few
        check = (out["survivors"] or out["candidates"])[:5]
        for rec in check:
            a, b = rec["_idx"]
            v_s = S.cluster_bootstrap_diff(al["labels"], al["scores"][a], al["scores"][b],
                                           al["clusters"], n_boot=n_boot_verify,
                                           seed=seed + 991, alpha=alpha)
            # The aggregated vector is clustered on the SUBJECT, which is one case
            # per cluster in the usual layout and two in a matched-pairs cohort.
            case_cl = al.get("case_clusters")
            if case_cl is None:
                case_cl = np.arange(len(al["subj_labels"]))
            v_p = S.cluster_bootstrap_diff(al["subj_labels"], al["subj_scores"][a],
                                           al["subj_scores"][b], case_cl,
                                           n_boot=n_boot_verify, seed=seed + 992,
                                           alpha=alpha)
            out["verified"].append({
                "A": rec["A"], "B": rec["B"], "survives": rec["survives"],
                "s04_slice": {k: v_s[k] for k in ("diff", "ci_lo", "ci_hi", "p")},
                "s04_patient": {k: v_p[k] for k in ("diff", "ci_lo", "ci_hi", "p")},
                "joint_slice": {k: rec["slice"][k] for k in ("ci_lo", "ci_hi", "p")},
                "joint_patient": {k: rec["patient"][k] for k in ("ci_lo", "ci_hi", "p")},
            })
    for rec in out["candidates"]:
        rec.pop("_idx", None)
    return out


# ==========================================================================
# Does the unit change the answer to the PAPER'S comparison?
# ==========================================================================

def _model_condition(al: dict, i: int) -> tuple[str, str] | None:
    """(architecture+init, condition) of method i, from metadata or from its name."""
    ms = al.get("methods")
    if ms and isinstance(ms[i], dict) and ms[i].get("condition"):
        return f"{ms[i]['arch']}/{ms[i]['init']}", str(ms[i]["condition"])
    parts = str(al["names"][i]).split("/")
    return (("/".join(parts[:-1]), parts[-1]) if len(parts) >= 2 else None)


def condition_interaction(al: dict, pt: dict, bs: dict, alpha: float = 0.05,
                          contrast: tuple[str, str] = ("magnitude", "phase")) -> dict:
    """
    The unit x condition INTERACTION, for the one contrast this paper exists to
    make: magnitude versus phase.

    Why this and not only the pairwise inversion test above. `inversion_pairs`
    asks whether the unit reorders two ARBITRARY methods, which over 210 pairs is
    a multiplicity problem and, on these cohorts, has no power. But the question
    a reader actually asks is not "is method 37 above method 12" -- it is "does
    phase beat magnitude?", with the architecture HELD FIXED. That contrast is
    paired, there is one of it per architecture rather than 210, and it is the
    decision a downstream paper would make.

    With architecture `a` and conditions (c1, c2):

        d_slice(a) = AUC_slice(c1, a) - AUC_slice(c2, a)
        d_agg(a)   = AUC_agg  (c1, a) - AUC_agg  (c2, a)
        I(a)       = d_agg(a) - d_slice(a)

    I(a) is how much the answer MOVES when the evaluation unit changes, with
    everything else held fixed. I(a) > 0 means aggregating to the patient unit
    shifts the verdict towards c1. A SIGN FLIP -- d_slice and d_agg of opposite
    sign -- means the two units return literally opposite answers to the paper's
    question for that architecture.

    Every quantity is read off the SAME joint bootstrap columns as everything
    else, so each I(a) is a paired clustered-bootstrap statistic on identical
    subjects, and Holm runs over the architectures examined. The mean over
    architectures is reported too, but as a description of these architectures,
    not as an inference to architectures in general -- they are not a sample of
    anything.

    This contrast is fixed by the paper's subject, not chosen after inspecting
    the table; the alternative contrasts (`both` vs either) are reported in the
    same table so the choice cannot be hidden.
    """
    A, B = bs["auc_slice"], bs["auc_patient"]
    out = {"contrast": list(contrast), "rows": [], "notes": [], "n_models": 0,
           "n_sign_flips": 0, "n_supported": 0, "mean": None}
    if len(A) < 20:
        out["notes"].append(f"only {len(A)} evaluable replicates; no interaction reported")
        return out

    by_model: dict[str, dict[str, int]] = {}
    for i in range(len(al["names"])):
        mc = _model_condition(al, i)
        if mc:
            by_model.setdefault(mc[0], {})[mc[1]] = i
    c1, c2 = contrast
    models = sorted(m for m, cs in by_model.items() if c1 in cs and c2 in cs)
    if not models:
        out["notes"].append(
            f"no architecture carries both '{c1}' and '{c2}'; interaction not defined")
        return out

    inter_cols, rows = [], []
    for m in models:
        i, j = by_model[m][c1], by_model[m][c2]
        d_sl = A[:, i] - A[:, j]
        d_ag = B[:, i] - B[:, j]
        inter = d_ag - d_sl
        inter_cols.append(inter)
        tail = min(float((inter <= 0).mean()), float((inter >= 0).mean()))
        obs_sl = float(pt["auc_slice"][i] - pt["auc_slice"][j])
        obs_ag = float(pt["auc_patient"][i] - pt["auc_patient"][j])
        # A CEILING makes the interaction real and meaningless at once. If both
        # conditions already saturate the aggregated unit, d_agg is pinned at 0 by
        # arithmetic, so I = -d_slice with almost no variance and the bootstrap will
        # happily call it significant. What has been detected there is the ceiling,
        # not a unit effect, so it is excluded from the count and labelled.
        ceil = bool(min(pt["auc_patient"][i], pt["auc_patient"][j]) >= 0.995)
        rows.append({
            "model": m,
            "d_slice": obs_sl, "d_agg": obs_ag,
            "interaction": obs_ag - obs_sl,
            "ci_lo": _pct(inter, alpha / 2), "ci_hi": _pct(inter, 1 - alpha / 2),
            "p": float(min(1.0, max(2.0 * tail, 1.0 / len(inter)))),
            "sign_flip": bool(obs_sl != 0 and obs_ag != 0 and (obs_sl > 0) != (obs_ag > 0)),
            "ceilinged": ceil,
        })
        if ceil:
            out["notes"].append(
                f"{m}: both conditions reach >= 0.995 at the aggregated unit, so the "
                "interaction is pinned by the ceiling rather than by the unit; it is "
                "reported but not counted as supported")
    for row, hp in zip(rows, S.holm_adjust([r["p"] for r in rows])):
        row["holm_p"] = hp
        row["supported"] = bool(row["ci_lo"] is not None and row["ci_hi"] is not None
                                and (row["ci_lo"] > 0 or row["ci_hi"] < 0)
                                and (hp or 1) < alpha and not row["ceilinged"])

    mean_col = np.mean(np.vstack(inter_cols), axis=0)
    tail = min(float((mean_col <= 0).mean()), float((mean_col >= 0).mean()))
    out["mean"] = {
        "interaction": float(np.mean([r["interaction"] for r in rows])),
        "ci_lo": _pct(mean_col, alpha / 2), "ci_hi": _pct(mean_col, 1 - alpha / 2),
        "p": float(min(1.0, max(2.0 * tail, 1.0 / len(mean_col)))),
        "n_positive": int(sum(r["interaction"] > 0 for r in rows)),
    }
    out["rows"] = rows
    out["n_models"] = len(rows)
    out["n_sign_flips"] = int(sum(r["sign_flip"] for r in rows))
    out["n_supported"] = int(sum(r["supported"] for r in rows))
    return out


# ==========================================================================
# Per-cohort driver
# ==========================================================================

def analyse_cohort(cohort: str, source_dirs, cache_dir: Path, cohort_dir: Path,
                   n_boot: int = 2000, seed: int = 0, alpha: float = 0.05,
                   seeds=(42,), verify: bool = True,
                   n_split_half: int = 200) -> dict:
    res = {"cohort": cohort, "ok": False, "reason": None, "notes": []}
    methods, notes, cluster_map = collect_methods(
        cohort, source_dirs, cache_dir, cohort_dir, seeds=seeds)
    res["notes"].extend(notes)
    if len(methods) < 2:
        res["reason"] = f"only {len(methods)} method(s) found for {cohort}"
        return res

    al = align_methods(methods, cluster_map, build_scan_map(cohort, cache_dir)["map"])
    res["notes"].extend(al.get("notes", []))
    if not al["ok"]:
        res["reason"] = al["reason"]
        return res

    pt = point_estimates(al)
    bs = joint_bootstrap(al, n_boot=n_boot, seed=seed)

    res.update({
        "ok": True,
        "n_methods": len(al["names"]),
        "method_names": al["names"],
        "cluster_unit": al["cluster_unit"],
        "cluster_source": al["cluster_source"],
        "case_unit": al["case_unit"],
        "n_slices": al["n_slices"], "n_pos_slices": al["n_pos_slices"],
        "n_subjects": al["n_subjects"],
        "n_cases": al["n_cases"], "n_pos_cases": al["n_pos_cases"],
        "n_pos_subjects": al["n_pos_subjects"],
        "n_mixed_label_subjects": al["n_mixed_label_subjects"],
        "bootstrap": {k: v for k, v in bs.items() if not isinstance(v, np.ndarray)},
        "table": [
            {"method": al["names"][m],
             "auc_slice": float(pt["auc_slice"][m]),
             "auc_patient": float(pt["auc_patient"][m]),
             "rank_slice": float(pt["rank_slice"][m]),
             "rank_patient": float(pt["rank_patient"][m]),
             "rank_shift": float(pt["rank_patient"][m] - pt["rank_slice"][m]),
             "slice_minus_patient": float(pt["auc_slice"][m] - pt["auc_patient"][m]),
             "auc_slice_ci": [_pct(bs["auc_slice"][:, m], alpha / 2),
                              _pct(bs["auc_slice"][:, m], 1 - alpha / 2)],
             "auc_patient_ci": [_pct(bs["auc_patient"][:, m], alpha / 2),
                                _pct(bs["auc_patient"][:, m], 1 - alpha / 2)]}
            for m in range(len(al["names"]))
        ],
        # How much of the ranking is even distinguishable from chance. If nearly
        # every method's interval covers 0.5 at a unit, the ordering at that unit
        # is an ordering of noise, and that is the reason any reordering against
        # the other unit must not be read as an inversion.
        "n_slice_ci_covers_half": None,
        "n_agg_ci_covers_half": None,
        "agreement": rank_agreement(pt, bs, alpha=alpha),
        "stability": stability_comparison(
            pt, bs, alpha=alpha,
            split_half=split_half_stability(al, n_rep=n_split_half, seed=seed,
                                            alpha=alpha)),
        "inversions": inversion_pairs(al, pt, bs, alpha=alpha, verify=verify, seed=seed),
        "interaction": condition_interaction(al, pt, bs, alpha=alpha),
    })

    def _covers_half(key):
        n = 0
        for row in res["table"]:
            lo, hi = row[key]
            if lo is not None and hi is not None and lo <= 0.5 <= hi:
                n += 1
        return n
    res["n_slice_ci_covers_half"] = _covers_half("auc_slice_ci")
    res["n_agg_ci_covers_half"] = _covers_half("auc_patient_ci")
    return res


# ==========================================================================
# Printing
# ==========================================================================

def _f(x, nd=3):
    return "-" if x is None or (isinstance(x, float) and not np.isfinite(x)) else f"{x:.{nd}f}"


def _ci(pair, nd=3):
    if not pair or pair[0] is None:
        return "-"
    return f"[{pair[0]:.{nd}f}, {pair[1]:.{nd}f}]"


def print_cohort(res: dict) -> None:
    print(S._rule(108, "="))
    print(f"COHORT: {res['cohort']}")
    print(S._rule(108, "="))
    for n in res["notes"]:
        print(f"  note: {n}")
    if not res["ok"]:
        print(f"  NO ANALYSIS: {res['reason']}")
        print()
        return
    print(f"  {res['n_methods']} methods | {res['n_slices']} slices "
          f"({res['n_pos_slices']} pos) | {res['n_subjects']} {res['cluster_unit']}s "
          f"| aggregated unit: {res['n_cases']} {res['case_unit']}s "
          f"({res['n_pos_cases']} pos) | cluster source: {res['cluster_source']}")
    b = res["bootstrap"]
    print(f"  bootstrap: {b['n_boot_used']}/{b['n_boot_requested']} replicates evaluable "
          f"(skipped {b['n_skipped_single_cluster']} single-cluster, "
          f"{b['n_skipped_single_class_slice']} single-class-slice, "
          f"{b['n_skipped_single_class_patient']} single-class-patient)")

    # 'subject_id' is the cluster-map's name for the unit; the tables read better
    # with the bare noun, and the columns are sized for it.
    cu = res.get("case_unit", "subject").replace("_id", "")
    print(f"\n  [1] RANKING BY UNIT   (rank 1 = best; shift = {cu} rank - slice rank)")
    hdr = (f"      {'method':<34}{'slice AUC':>10} {'95% CI':<18}"
           f"{cu[:3] + ' AUC':>9} {'95% CI':<18}{'r_sl':>6}{'r_agg':>7}{'shift':>7}")
    print(hdr)
    print("      " + "-" * (len(hdr) - 6))
    for row in sorted(res["table"], key=lambda r: r["rank_slice"]):
        print(f"      {row['method']:<34}{row['auc_slice']:>10.3f} "
              f"{_ci(row['auc_slice_ci']):<18}{row['auc_patient']:>9.3f} "
              f"{_ci(row['auc_patient_ci']):<18}"
              f"{row['rank_slice']:>6.1f}{row['rank_patient']:>7.1f}"
              f"{row['rank_shift']:>+7.1f}")
    m = res["n_methods"]
    print(f"      -> {res['n_slice_ci_covers_half']}/{m} methods have a slice-level 95% CI "
          f"covering 0.500; {res['n_agg_ci_covers_half']}/{m} at {cu} level. A unit at "
          "which")
    print(f"         every interval covers chance is ordering noise, and a reordering "
          "against it is not an inversion.")
    # The opposite failure. A cohort every method solves perfectly has no ordering to
    # invert either, and saying so is not the same as saying the units agree.
    ceil = min(r["auc_patient"] for r in res["table"])
    if ceil >= 0.995:
        print(f"      -> CEILING: the WORST method reaches {ceil:.3f} at {cu} level. There is "
              "no ordering left to")
        print("         invert; this cohort can neither support nor refute a unit effect.")

    ag = res["agreement"]
    names = res["method_names"]
    print("\n  [2] RANK AGREEMENT BETWEEN UNITS  (intervals from resampling subjects)")
    print(f"      Spearman rho  {_f(ag['rho_obs'])}   bootstrap median {_f(ag['rho_median'])} "
          f"{_ci(ag['rho_ci'])}")
    print(f"      Kendall tau   {_f(ag['tau_obs'])}   bootstrap median {_f(ag['tau_median'])} "
          f"{_ci(ag['tau_ci'])}")
    print(f"      top-1 slice    {', '.join(names[i] for i in ag['top1_slice'])}")
    print(f"      top-1 {cu:<9}{', '.join(names[i] for i in ag['top1_patient'])}")
    print(f"      top-1 agrees  {ag['top1_same_obs']}   "
          f"(agrees in {ag['top1_same_rate']*100:.1f}% of subject resamples)")
    print(f"      top-3 overlap {ag['top3_overlap_obs']}/3   "
          f"(resample mean {_f(ag['top3_overlap_mean'], 2)} {_ci(ag['top3_overlap_ci'], 1)})")

    st = res["stability"]
    print("\n  [3] DECISIVE COMPARISON: between-unit disagreement vs within-unit noise")
    print("      D = 1 - Kendall tau  (0 = identical ordering, 2 = full reversal;")
    print("                            D/2 = fraction of method pairs ordered differently)")
    if st["verdict"] == "NO_ESTIMATE":
        print(f"      {st['verdict_text']}")
    else:
        print(f"      D_between  (slice ranking vs {cu} ranking, same subjects)")
        print(f"          observed {_f(st['d_between_obs'])}   resampled median "
              f"{_f(st['d_between']['median'])} "
              f"[{_f(st['d_between']['lo'])}, {_f(st['d_between']['hi'])}]")
        print(f"      D_within   (same unit, subjects resampled -- the noise floor)")
        print(f"          slice   median {_f(st['d_within_slice']['median'])} "
              f"[{_f(st['d_within_slice']['lo'])}, {_f(st['d_within_slice']['hi'])}]")
        print(f"          {cu:<7} median {_f(st['d_within_patient']['median'])} "
              f"[{_f(st['d_within_patient']['lo'])}, {_f(st['d_within_patient']['hi'])}]")
        print(f"      paired delta = D_between - D_within  (>0 means the unit moves the "
              f"ranking more than noise does)")
        print(f"          vs slice   {_f(st['delta_vs_slice']['median'])} "
              f"[{_f(st['delta_vs_slice']['lo'])}, {_f(st['delta_vs_slice']['hi'])}]"
              f"   P(delta>0) = {_f(st['delta_vs_slice']['p_gt0'])}")
        print(f"          vs {cu:<8}{_f(st['delta_vs_patient']['median'])} "
              f"[{_f(st['delta_vs_patient']['lo'])}, {_f(st['delta_vs_patient']['hi'])}]"
              f"   P(delta>0) = {_f(st['delta_vs_patient']['p_gt0'])}")
        print(f"      P(within-unit noise >= observed between-unit disagreement) = "
              f"{_f(st['p_exceed'])}")
        print(f"      within-unit rank stability: median tau to own point ranking "
              f"slice {_f(st['tau_within_slice_median'], 2)}, "
              f"{cu} {_f(st['tau_within_patient_median'], 2)}")
        sh = st.get("split_half") or {}
        if sh.get("tau_slice_median") is not None:
            print(f"      SPLIT-HALF reproducibility (two DISJOINT halves of the subjects, "
                  f"{sh['n_rep_used']} splits;")
            print(f"         half the subjects each, so this is a lower bound): "
                  f"slice tau {_f(sh['tau_slice_median'], 2)} "
                  f"[{_f(sh['tau_slice_lo'], 2)}, {_f(sh['tau_slice_hi'], 2)}], "
                  f"{cu} tau {_f(sh['tau_agg_median'], 2)} "
                  f"[{_f(sh['tau_agg_lo'], 2)}, {_f(sh['tau_agg_hi'], 2)}]")
        print(f"      top-1 reproducibility under resampling: "
              f"slice {st['top1_stability_slice']*100:.1f}%, "
              f"{cu} {st['top1_stability_patient']*100:.1f}%")
        print(f"      VERDICT [{st['verdict']}]")
        for line in _wrap(st["verdict_text"], 96):
            print(f"          {line}")

    iv = res["inversions"]
    print(f"\n  [4] INVERSION PAIRS  ({iv['n_pairs_examined']} pairs examined, "
          f"Holm-adjusted over all of them at each unit)")
    if iv.get("notes"):
        for n in iv["notes"]:
            print(f"      note: {n}")
    if not iv["candidates"]:
        print("      no pair even flips sign between the two units.")
    else:
        print(f"      {len(iv['candidates'])} pair(s) flip sign; "
              f"{len(iv['survivors'])} survive the support bar.")
        hdr = (f"      {'A (slice winner)':<30}{'B (' + cu + ' winner)':<30}"
               f"{'slice diff [CI]':<26}{cu + ' diff [CI]':<26}"
               f"{'holm_s':>8}{'holm_a':>8}  ok")
        print(hdr)
        print("      " + "-" * (len(hdr) - 6))
        for r in iv["candidates"][:12]:
            sd = f"+{r['slice_diff']:.3f} {_ci([r['slice']['ci_lo'], r['slice']['ci_hi']])}"
            pd_ = (f"-{r['patient_diff']:.3f} "
                   f"{_ci([r['patient']['ci_lo'], r['patient']['ci_hi']])}")
            print(f"      {r['A']:<30}{r['B']:<30}{sd:<26}{pd_:<26}"
                  f"{_f(r['holm_p_slice']):>8}{_f(r['holm_p_patient']):>8}"
                  f"  {'YES' if r['survives'] else 'no'}")
        if len(iv["candidates"]) > 12:
            print(f"      ... {len(iv['candidates']) - 12} further sign-flipping pairs "
                  "not shown (none survives; see the JSON)")
    if not iv["survivors"]:
        print("      NO inversion pair clears the bar: no pair is BOTH ordered one way at "
              "slice level")
        print(f"      and the other way at {cu} level with each ordering individually "
              "supported.")
    if iv.get("verified"):
        print("      cross-check against s04_stats.cluster_bootstrap_diff "
              "(independent RNG stream):")
        for v in iv["verified"]:
            print(f"        {v['A']} vs {v['B']}")
            print(f"          slice   s04 {_f(v['s04_slice']['diff'])} "
                  f"{_ci([v['s04_slice']['ci_lo'], v['s04_slice']['ci_hi']])} "
                  f"p={_f(v['s04_slice']['p'])}   | joint "
                  f"{_ci([v['joint_slice']['ci_lo'], v['joint_slice']['ci_hi']])} "
                  f"p={_f(v['joint_slice']['p'])}")
            print(f"          patient s04 {_f(v['s04_patient']['diff'])} "
                  f"{_ci([v['s04_patient']['ci_lo'], v['s04_patient']['ci_hi']])} "
                  f"p={_f(v['s04_patient']['p'])}   | joint "
                  f"{_ci([v['joint_patient']['ci_lo'], v['joint_patient']['ci_hi']])} "
                  f"p={_f(v['joint_patient']['p'])}")

    it = res.get("interaction") or {}
    c1, c2 = (it.get("contrast") or ["magnitude", "phase"])
    print(f"\n  [5] DOES THE UNIT CHANGE THE ANSWER TO '{c1.upper()} vs {c2.upper()}'?")
    print(f"      architecture held fixed; d = AUC({c1}) - AUC({c2}) at each unit;")
    print(f"      interaction I = d_{cu} - d_slice  (>0: aggregating favours {c1})")
    for n in it.get("notes", []):
        print(f"      note: {n}")
    if it.get("rows"):
        hdr = (f"      {'architecture':<26}{'d_slice':>9}{'d_' + cu[:3]:>9}{'I':>9}"
               f"  {'95% CI':<20}{'holm p':>8}  flip  ok")
        print(hdr)
        print("      " + "-" * (len(hdr) - 6))
        for r in it["rows"]:
            print(f"      {r['model']:<26}{r['d_slice']:>+9.3f}{r['d_agg']:>+9.3f}"
                  f"{r['interaction']:>+9.3f}  {_ci([r['ci_lo'], r['ci_hi']]):<20}"
                  f"{_f(r['holm_p']):>8}  {'YES' if r['sign_flip'] else ' - ':<5}"
                  f" {'YES' if r['supported'] else ('CEIL' if r['ceilinged'] else 'no')}")
        mn = it["mean"]
        print(f"      {'MEAN over architectures':<26}{'':>9}{'':>9}"
              f"{mn['interaction']:>+9.3f}  {_ci([mn['ci_lo'], mn['ci_hi']]):<20}"
              f"{_f(mn['p']):>8}")
        print(f"      -> I > 0 for {mn['n_positive']}/{it['n_models']} architectures; "
              f"{it['n_sign_flips']} architecture(s) get the OPPOSITE answer at the two "
              "units;")
        print(f"         {it['n_supported']}/{it['n_models']} interactions are individually "
              "supported after Holm.")
    print()


def _wrap(text: str, width: int) -> list[str]:
    words, lines, cur = text.split(), [], ""
    for w in words:
        if len(cur) + len(w) + 1 > width:
            lines.append(cur)
            cur = w
        else:
            cur = f"{cur} {w}".strip()
    if cur:
        lines.append(cur)
    return lines


def print_summary(results: list[dict]) -> None:
    print(S._rule(108, "="))
    print("SUMMARY -- does the evaluation unit change which method wins?")
    print(S._rule(108, "="))
    hdr = (f"  {'cohort':<14}{'M':>4}{'subj':>6}{'rho':>7}{'tau':>7}"
           f"{'top1?':>7}{'D_btw':>7}{'D_win_s':>9}{'P(noise>=)':>12}"
           f"{'sh_sl':>7}{'sh_agg':>7}{'inv':>5}{'m-vs-p':>8}  verdict")
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for r in results:
        if not r["ok"]:
            print(f"  {r['cohort']:<14}{'-':>4}{'-':>6}{'-':>7}{'-':>7}{'-':>7}"
                  f"{'-':>7}{'-':>9}{'-':>12}{'-':>7}{'-':>7}{'-':>5}{'-':>8}"
                  f"  NO ANALYSIS ({r['reason']})")
            continue
        ag, st, iv = r["agreement"], r["stability"], r["inversions"]
        sh = st.get("split_half") or {}
        it = r.get("interaction") or {}
        mvp = (f"{it['n_supported']}/{it['n_models']}" if it.get("n_models") else "-")
        print(f"  {r['cohort']:<14}{r['n_methods']:>4}{r['n_subjects']:>6}"
              f"{_f(ag['rho_obs'], 2):>7}{_f(ag['tau_obs'], 2):>7}"
              f"{('yes' if ag['top1_same_obs'] else 'NO'):>7}"
              f"{_f(st.get('d_between_obs'), 2):>7}"
              f"{_f((st.get('d_within_slice') or {}).get('median'), 2):>9}"
              f"{_f(st.get('p_exceed'), 3):>12}"
              f"{_f(sh.get('tau_slice_median'), 2):>7}"
              f"{_f(sh.get('tau_agg_median'), 2):>7}"
              f"{len(iv['survivors']):>5}{mvp:>8}  {st['verdict']}")
    print("  sh_sl / sh_agg = split-half Kendall tau: do two DISJOINT halves of the "
          "subjects rank the")
    print("  methods the same way, at slice and at the aggregated unit? "
          "(1 = identical, 0 = unrelated)")
    print("  inv = inversion pairs surviving the support bar. m-vs-p = architectures whose "
          "magnitude-vs-phase")
    print("  verdict moves with the unit by more than noise (Holm-supported interaction) "
          "/ architectures tested.")
    print()


# ==========================================================================
# Self-test
# ==========================================================================

def _sim_methods(rng, specs, n_patients=200, slices=10):
    """
    Clustered simulator with an ANALYTIC truth at both units.

    Each patient carries a label y and a patient effect u ~ N(0,1); each slice
    carries e ~ N(0,1). A method with spec (a, b, c) scores

        s = a*y + b*u + c*e

    so, comparing a slice of a positive patient with a slice of a negative one,
    the score difference is a + b*(u1-u2) + c*(e1-e2) and the population AUCs are

        slice-level    Phi( a / sqrt(2b^2 + 2c^2) )
        patient-level  Phi( a / sqrt(2b^2 + 2c^2/n_slices) )   [mean aggregation]

    Both are closed form, so the TRUE ranking at each unit is known exactly and
    the test does not depend on the very machinery it is testing. Choosing b and
    c independently is what lets a genuine inversion be constructed: raising c
    (within-patient noise) hurts the slice-level AUC and is averaged away at
    patient level, while raising b (between-patient noise) hurts both.
    """
    from math import erfc, sqrt
    phi = lambda z: 0.5 * erfc(-z / sqrt(2))
    y_pat = rng.integers(0, 2, size=n_patients)
    pid = np.repeat(np.arange(n_patients), slices)
    y = np.repeat(y_pat, slices)
    u = np.repeat(rng.normal(0, 1, size=n_patients), slices)
    scores, truth_s, truth_p = [], [], []
    for (a, b, c) in specs:
        e = rng.normal(0, 1, size=n_patients * slices)
        scores.append(a * y + b * u + c * e)
        truth_s.append(phi(a / sqrt(2 * b * b + 2 * c * c)))
        truth_p.append(phi(a / sqrt(2 * b * b + 2 * c * c / slices)))
    return (y.astype(int), np.asarray(scores), pid.astype(str),
            np.asarray(truth_s), np.asarray(truth_p))


def _al_from_sim(y, scores, pid, names):
    """Wrap simulated arrays in the structure align_methods produces."""
    clusters = np.asarray([str(p) for p in pid], dtype=object)
    uniq, groups = S._cluster_index(clusters)
    subj_scores = np.empty((scores.shape[0], len(uniq)))
    subj_labels = None
    for m in range(scores.shape[0]):
        agg = S.aggregate_by_cluster(y, scores[m], clusters, how="mean")
        subj_scores[m] = agg["scores"]
        subj_labels = agg["labels"]
    return {
        "ok": True, "names": names, "labels": y, "scores": scores,
        "clusters": clusters, "cluster_unit": "patient_id", "cluster_source": "sim",
        "subjects": uniq, "groups": groups,
        "subj_labels": subj_labels, "subj_scores": subj_scores,
        "n_slices": len(y), "n_pos_slices": int(y.sum()),
        "n_subjects": len(uniq), "n_pos_subjects": int(subj_labels.sum()),
        "n_mixed_label_subjects": 0,
    }


def self_test(quick: bool = False, sim_seed: int = 20260729) -> int:
    print(S._rule(100, "="))
    print(f"s16_rankinversion self-test (simulation seed {sim_seed})")
    print(S._rule(100, "="))
    c = S._Check()
    rng = np.random.default_rng(sim_seed)
    nb = 300 if quick else 1500

    # ------------------------------------------------------------ primitives
    print("\n[1] rank primitives")
    c.ok(np.allclose(ranks_desc([0.9, 0.7, 0.8]), [1, 3, 2]),
         "ranks_desc puts the largest value at rank 1")
    c.ok(np.allclose(ranks_desc([0.5, 0.5, 0.1]), [1.5, 1.5, 3]),
         "ranks_desc averages ties")
    c.ok(abs(kendall_tau_b([1, 2, 3, 4], [1, 2, 3, 4]) - 1.0) < 1e-12,
         "tau of a ranking with itself is 1")
    c.ok(abs(kendall_tau_b([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12,
         "tau of a ranking with its reversal is -1")
    c.ok(abs(spearman_rho([1, 2, 3, 4], [4, 3, 2, 1]) + 1.0) < 1e-12,
         "rho of a ranking with its reversal is -1")
    c.ok(kendall_tau_b([1, 1, 1], [1, 2, 3]) is None,
         "tau is None (not 0) when one ranking is fully tied")
    c.ok(spearman_rho([1, 1, 1], [1, 2, 3]) is None,
         "rho is None (not 0) when one ranking is fully tied")
    # (1 - tau)/2 must equal the fraction of discordant pairs
    a, b = [1, 2, 3, 4, 5], [1, 2, 3, 5, 4]
    c.ok(abs((1 - kendall_tau_b(a, b)) / 2 - 1 / 10) < 1e-12,
         "(1-tau)/2 is the fraction of pairs ordered differently", "1 of 10 pairs")
    try:
        from scipy.stats import kendalltau, spearmanr
        x = rng.normal(size=15).round(1)
        z = rng.normal(size=15).round(1)
        c.ok(abs(kendall_tau_b(ranks_desc(x), ranks_desc(z))
                 - kendalltau(-x, -z).statistic) < 1e-10,
             "kendall_tau_b matches scipy.stats.kendalltau (with ties)")
        c.ok(abs(spearman_rho(ranks_desc(x), ranks_desc(z))
                 - spearmanr(-x, -z).statistic) < 1e-10,
             "spearman_rho matches scipy.stats.spearmanr (with ties)")
    except ImportError:
        print("  [SKIP] scipy not installed; rank primitives not cross-checked")
    c.ok(top_k_set([0.9, 0.9, 0.5, 0.4], 1) == {0, 1},
         "top_k_set returns BOTH members of a tie rather than picking one")

    # ------------------------------------------------- joint bootstrap vs s04
    print("\n[2] the joint bootstrap reproduces s04's own estimator")
    y, sc, pid, ts, tp = _sim_methods(rng, [(1.0, 0.5, 1.0), (0.8, 0.5, 1.0)],
                                      n_patients=120, slices=8)
    al = _al_from_sim(y, sc, pid, ["m0", "m1"])
    bs = joint_bootstrap(al, n_boot=nb, seed=7)
    pt = point_estimates(al)
    ref = S.cluster_bootstrap_auc(al["labels"], al["scores"][0], al["clusters"],
                                  n_boot=nb, seed=11)
    c.ok(abs(pt["auc_slice"][0] - ref["auc"]) < 1e-12,
         "point slice AUC identical to s04.cluster_bootstrap_auc's",
         f"{pt['auc_slice'][0]:.6f}")
    lo, hi = _pct(bs["auc_slice"][:, 0], 0.025), _pct(bs["auc_slice"][:, 0], 0.975)
    c.ok(abs(lo - ref["ci_lo"]) < 0.03 and abs(hi - ref["ci_hi"]) < 0.03,
         "joint-bootstrap slice CI matches s04's within Monte Carlo error",
         f"joint [{lo:.3f}, {hi:.3f}] vs s04 [{ref['ci_lo']:.3f}, {ref['ci_hi']:.3f}]")
    ref_d = S.cluster_bootstrap_diff(al["labels"], al["scores"][0], al["scores"][1],
                                     al["clusters"], n_boot=nb, seed=13)
    got_d = _boot_diff_ci(bs["auc_slice"][:, 0], bs["auc_slice"][:, 1], 0.05)
    c.ok(abs(got_d["ci_lo"] - ref_d["ci_lo"]) < 0.03
         and abs(got_d["ci_hi"] - ref_d["ci_hi"]) < 0.03,
         "joint-bootstrap paired difference CI matches s04.cluster_bootstrap_diff",
         f"joint [{got_d['ci_lo']:.3f}, {got_d['ci_hi']:.3f}] vs "
         f"s04 [{ref_d['ci_lo']:.3f}, {ref_d['ci_hi']:.3f}]")
    ref_p = S.cluster_bootstrap_auc(al["subj_labels"], al["subj_scores"][0],
                                    np.arange(al["n_subjects"]), n_boot=nb, seed=17)
    c.ok(abs(pt["auc_patient"][0] - ref_p["auc"]) < 1e-12,
         "point patient AUC identical to the s04 aggregate + bootstrap path",
         f"{pt['auc_patient'][0]:.6f}")

    # ------------------------------------------------- CASE A: no inversion
    print("\n[3] CASE A -- true ordering IDENTICAL at both units (must find no inversion)")
    # separation varies only through `a`; b and c are held fixed, so both closed-form
    # AUCs are monotone in `a` and the true ranking is the same at both units.
    specs = [(1.6, 0.6, 1.0), (1.2, 0.6, 1.0), (0.9, 0.6, 1.0), (0.6, 0.6, 1.0)]
    y, sc, pid, ts, tp = _sim_methods(rng, specs, n_patients=400, slices=10)
    c.ok(np.array_equal(ranks_desc(ts), ranks_desc(tp)),
         "the SIMULATION's true rankings agree at both units, by construction",
         f"slice {ranks_desc(ts).tolist()} patient {ranks_desc(tp).tolist()}")
    al = _al_from_sim(y, sc, pid, [f"m{i}" for i in range(len(specs))])
    pt = point_estimates(al)
    bs = joint_bootstrap(al, n_boot=nb, seed=3)
    ag = rank_agreement(pt, bs)
    st = stability_comparison(pt, bs)
    iv = inversion_pairs(al, pt, bs, verify=False)
    c.ok(np.array_equal(pt["rank_slice"], ranks_desc(ts)),
         "estimated slice ranking recovers the true slice ranking")
    c.ok(np.array_equal(pt["rank_patient"], ranks_desc(tp)),
         "estimated patient ranking recovers the true patient ranking")
    c.ok(ag["tau_obs"] > 0.95, "Kendall tau between units is ~1", f"tau={ag['tau_obs']:.3f}")
    c.ok(ag["top1_same_obs"], "top-1 method is the same at both units")
    c.ok(len(iv["survivors"]) == 0, "NO inversion pair survives",
         f"{len(iv['candidates'])} candidates, 0 survivors")
    c.ok(st["verdict"] != "UNIT_CHANGES_RANK",
         "verdict does not claim the unit changes the ranking", st["verdict"])

    # ------------------------------------------------- CASE B: true inversion
    print("\n[4] CASE B -- ordering GENUINELY REVERSES between units (must detect it)")
    # mA*: weak between-patient noise, huge within-patient noise -> poor at slice
    #      level, excellent once the slice noise is averaged away.
    # mB*: strong within-slice signal but large between-patient noise -> good at
    #      slice level, and averaging buys them nothing.
    # Two of each, so four method PAIRS genuinely invert, plus filler_top, which
    # wins at both units by construction. A top-1 check ALONE would therefore
    # report perfect agreement while four real inversions sit below it -- exactly
    # the failure mode the pairwise analysis exists to catch.
    specs = [(1.50, 0.05, 3.5), (1.35, 0.05, 3.5),
             (1.15, 1.20, 0.2), (1.05, 1.20, 0.2),
             (1.70, 0.50, 1.0)]
    names = ["mA1_patientwise", "mA2_patientwise",
             "mB1_slicewise", "mB2_slicewise", "filler_top"]
    y, sc, pid, ts, tp = _sim_methods(rng, specs, n_patients=400, slices=16)
    true_tau = kendall_tau_b(ranks_desc(ts), ranks_desc(tp))
    planted_pairs = {frozenset((a, b)) for a in (0, 1) for b in (2, 3)}
    c.ok(all(ts[a] < ts[b] and tp[a] > tp[b] for a in (0, 1) for b in (2, 3)),
         "all four planted pairs really do reverse between units in the SIMULATION",
         f"true slice {np.round(ts, 3).tolist()} | patient {np.round(tp, 3).tolist()}")
    c.ok(-1.0 < true_tau < 1.0,
         "the true between-unit tau is strictly inside (-1, 1), so the test is not vacuous",
         f"true tau = {true_tau:.3f}")
    c.ok(int(np.argmax(ts)) == int(np.argmax(tp)) == 4,
         "by construction the TRUE top-1 method is the same at both units",
         "so top-1 agreement alone cannot detect the planted inversions")
    al = _al_from_sim(y, sc, pid, names)
    pt = point_estimates(al)
    bs = joint_bootstrap(al, n_boot=nb, seed=5)
    ag = rank_agreement(pt, bs)
    st = stability_comparison(
        pt, bs, split_half=split_half_stability(al, n_rep=60, seed=5))
    iv = inversion_pairs(al, pt, bs, verify=False)
    c.ok(ag["tau_obs"] < 1.0, "the estimated rankings disagree between units",
         f"tau={ag['tau_obs']:.3f}")
    c.ok(ag["top1_same_obs"] and len(iv["survivors"]) >= 1,
         "top-1 agrees YET supported inversions are found -- the pairwise test sees "
         "what a top-1 check misses",
         f"{len(iv['survivors'])} survivor(s) of {len(iv['candidates'])} candidates")
    found = {frozenset((names.index(s["A"]), names.index(s["B"])))
             for s in iv["survivors"]}
    c.ok(found and found <= planted_pairs,
         "every survivor is one of the four PLANTED pairs -- no spurious inversion",
         f"{len(found)}/4 planted pairs recovered, 0 unplanted")
    c.ok(all(s["A"].startswith("mB") and s["B"].startswith("mA")
             for s in iv["survivors"]),
         "each survivor is named in the right direction (slice winner -> aggregated loser)",
         "; ".join(f"{s['A']} > {s['B']}" for s in iv["survivors"][:2]))
    c.ok(st["verdict"] == "UNIT_CHANGES_RANK",
         "verdict is UNIT_CHANGES_RANK", st["verdict"])
    c.ok(st["between_exceeds_within"],
         "between-unit disagreement exceeds within-unit resampling noise",
         f"delta vs slice {st['delta_vs_slice']['median']:.3f} "
         f"(p={st['delta_vs_slice']['p']:.3f}), vs aggregated "
         f"{st['delta_vs_patient']['median']:.3f} "
         f"(p={st['delta_vs_patient']['p']:.3f})")

    print("\n[4b] CASE B2 -- the inversion is at the TOP (top-1 must differ between units)")
    # Deliberately only two methods. tau is then forced to +-1 and carries no
    # information, which is why CASE B above exists; B2's job is narrower -- that
    # a reversal at the TOP of the table is reported as a top-1 disagreement and
    # that the pair clears the support bar.
    specs = [(1.5, 0.05, 3.5), (1.15, 1.2, 0.2)]
    y, sc, pid, ts, tp = _sim_methods(rng, specs, n_patients=300, slices=16)
    c.ok(int(np.argmax(ts)) == 1 and int(np.argmax(tp)) == 0,
         "by construction the TRUE best method differs between units",
         f"slice best mB ({ts[1]:.3f} vs {ts[0]:.3f}), "
         f"aggregated best mA ({tp[0]:.3f} vs {tp[1]:.3f})")
    al = _al_from_sim(y, sc, pid, ["mA_patientwise", "mB_slicewise"])
    pt = point_estimates(al)
    bs = joint_bootstrap(al, n_boot=nb, seed=6)
    ag = rank_agreement(pt, bs)
    iv = inversion_pairs(al, pt, bs, verify=False)
    c.ok(not ag["top1_same_obs"], "top-1 method DIFFERS between units",
         f"slice {ag['top1_slice']} aggregated {ag['top1_patient']}")
    c.ok(ag["top1_same_rate"] < 0.25,
         "and it differs in the large majority of subject resamples, not just this one",
         f"top-1 agrees in {ag['top1_same_rate']*100:.1f}% of resamples")
    c.ok(any({s["A"], s["B"]} == {"mA_patientwise", "mB_slicewise"}
             for s in iv["survivors"]),
         "the top-of-table inversion survives the bar",
         f"{len(iv['survivors'])}/{len(iv['candidates'])}")

    # ------------------------------------------------- CASE C: pure noise
    print("\n[5] CASE C -- methods IDENTICAL in truth (a chance reordering must NOT be "
          "called an inversion)")
    # Eight methods, identical population AUCs, on a deliberately small cohort --
    # the shape of the clinical cohorts in this study. Every reordering between
    # the two units is chance and nothing else.
    specs = [(1.0, 0.6, 1.0)] * 8
    y, sc, pid, ts, tp = _sim_methods(rng, specs, n_patients=20, slices=8)
    c.ok(float(np.ptp(ts)) < 1e-12 and float(np.ptp(tp)) < 1e-12,
         "the SIMULATION's methods have identical true AUCs at both units",
         f"slice {ts[0]:.4f}, patient {tp[0]:.4f}")
    al = _al_from_sim(y, sc, pid, [f"m{i}" for i in range(len(specs))])
    pt = point_estimates(al)
    bs = joint_bootstrap(al, n_boot=nb, seed=9)
    sh = split_half_stability(al, n_rep=200, seed=9)
    ag = rank_agreement(pt, bs)
    st = stability_comparison(pt, bs, split_half=sh)
    iv = inversion_pairs(al, pt, bs, verify=False)
    c.ok(len(iv["candidates"]) > 0,
         "the two units DO reorder methods by chance (this is the trap)",
         f"{len(iv['candidates'])} sign-flipping pairs")
    c.ok(sh["tau_slice_median"] < STABLE_TAU and sh["tau_agg_median"] < STABLE_TAU,
         "the SPLIT-HALF reference reports the ranking as NOT reproducible, as it must "
         "when the methods are identical in truth",
         f"split-half tau slice {sh['tau_slice_median']:.2f}, "
         f"aggregated {sh['tau_agg_median']:.2f} (threshold {STABLE_TAU})")
    c.ok(sh["tau_slice_median"] <= st["tau_within_slice_median"] + 0.05
         and sh["tau_agg_median"] <= st["tau_within_patient_median"] + 0.05,
         "and it is the STRICTER of the two references -- the bootstrap-vs-point tau is "
         "propped up by the ~63% of subjects a resample shares with the point sample, "
         "which is exactly the blind spot the split-half exists to close",
         f"split-half slice {sh['tau_slice_median']:.2f} vs bootstrap-vs-point "
         f"{st['tau_within_slice_median']:.2f}")
    c.ok(len(iv["survivors"]) == 0,
         "but NO chance reordering is reported as an inversion",
         f"{len(iv['survivors'])} survivors")
    c.ok(st["verdict"] != "UNIT_CHANGES_RANK",
         "verdict never claims the unit changes the ranking", st["verdict"])
    c.ok(not st["between_exceeds_within"],
         "between-unit disagreement does NOT exceed within-unit noise",
         f"P(noise >= observed) = {st['p_exceed']:.3f}")
    c.ok(st["verdict"] in REFUSAL_VERDICTS,
         "and the verdict is the honest one: this benchmark cannot rank at either unit",
         st["verdict"] + " -- " + "; ".join(st["instability_flags"])[:70])
    # The verdict label is a threshold crossing on a noisy statistic, so what is
    # checked here is that the LOGIC is consistent with the flags it reports --
    # a property of the code, true for every sample, not of this one draw.
    c.ok((st["verdict"] in REFUSAL_VERDICTS)
         == bool(st["instability_flags"] and not st["between_exceeds_within"]),
         "a refusal verdict fires exactly when the stability flags say it should")
    c.ok((st["verdict"] == "CANNOT_RANK") == bool(st["order_instability_flags"]),
         "CANNOT_RANK is used only when the WHOLE ORDER fails to reproduce, "
         "CANNOT_NAME_WINNER when only the top-1 does", st["verdict"])

    # ------------------------------------- CASE D: stable but no unit effect
    print("\n[6] CASE D -- methods well separated and units agree (must be INCONCLUSIVE, "
          "not an inversion)")
    specs = [(2.2, 0.6, 1.0), (1.4, 0.6, 1.0), (0.7, 0.6, 1.0)]
    y, sc, pid, ts, tp = _sim_methods(rng, specs, n_patients=500, slices=10)
    al = _al_from_sim(y, sc, pid, ["big", "mid", "small"])
    pt = point_estimates(al)
    bs = joint_bootstrap(al, n_boot=nb, seed=21)
    sh = split_half_stability(al, n_rep=100, seed=21)
    st = stability_comparison(pt, bs, split_half=sh)
    iv = inversion_pairs(al, pt, bs, verify=False)
    c.ok(st["tau_within_slice_median"] >= STABLE_TAU
         and st["top1_stability_slice"] >= STABLE_TOP1,
         "the within-unit ranking IS stable here",
         f"tau {st['tau_within_slice_median']:.2f}, "
         f"top1 {st['top1_stability_slice']*100:.0f}%")
    c.ok(sh["tau_slice_median"] >= STABLE_TAU and sh["tau_agg_median"] >= STABLE_TAU,
         "and two DISJOINT halves of the subjects rank them the same way -- the "
         "split-half reference does not cry wolf on a genuinely separable benchmark",
         f"split-half tau slice {sh['tau_slice_median']:.2f}, "
         f"aggregated {sh['tau_agg_median']:.2f}")
    c.ok(st["verdict"] == "INCONCLUSIVE",
         "verdict is INCONCLUSIVE (stable ranking, no unit effect)", st["verdict"])
    c.ok(len(iv["survivors"]) == 0, "no inversion pair survives")

    # ------------------------------- CASE E: the unit x condition interaction
    print("\n[7] CASE E -- unit x condition interaction, against a known truth")
    from math import erfc, sqrt
    phi = lambda z: 0.5 * erfc(-z / sqrt(2))
    nsl = 12

    def _truth_I(spec1, spec2):
        """d_agg - d_slice for one architecture, in closed form."""
        def au(spec, n):
            a, b, cc = spec
            return phi(a / sqrt(2 * b * b + 2 * cc * cc / n))
        return ((au(spec1, nsl) - au(spec2, nsl)) - (au(spec1, 1) - au(spec2, 1)))

    # E1 -- NULL. Within each architecture the two conditions are the same spec, so
    # the interaction is EXACTLY zero however the architectures differ from one
    # another. Nothing may be declared supported.
    null_specs = [(1.3, 0.6, 1.2), (1.3, 0.6, 1.2),      # archX mag / phase
                  (0.8, 0.9, 0.7), (0.8, 0.9, 0.7)]      # archY mag / phase
    names_e = ["archX/imagenet/magnitude", "archX/imagenet/phase",
               "archY/scratch/magnitude", "archY/scratch/phase"]
    y, sc, pid, ts, tp = _sim_methods(rng, null_specs, n_patients=400, slices=nsl)
    al = _al_from_sim(y, sc, pid, names_e)
    pt, bs = point_estimates(al), joint_bootstrap(al, n_boot=nb, seed=31)
    it = condition_interaction(al, pt, bs)
    c.ok(it["n_models"] == 2,
         "both architectures are picked up as carrying magnitude AND phase",
         ", ".join(r["model"] for r in it["rows"]))
    c.ok(all(abs(_truth_I(null_specs[2 * k], null_specs[2 * k + 1])) < 1e-12
             for k in range(2)),
         "the SIMULATION's true interaction is exactly zero in E1, by construction")
    c.ok(it["n_supported"] == 0,
         "no architecture's interaction is declared supported when the truth is zero",
         f"{it['n_supported']}/2 supported")
    c.ok(not any(r["ceilinged"] for r in it["rows"]),
         "and nothing is mislabelled as ceilinged when the AUCs are nowhere near 1")
    c.ok(it["mean"]["ci_lo"] <= 0 <= it["mean"]["ci_hi"],
         "and the mean interaction's interval covers zero",
         f"{it['mean']['interaction']:+.4f} "
         f"[{it['mean']['ci_lo']:+.4f}, {it['mean']['ci_hi']:+.4f}]")

    # E2 -- PLANTED. Magnitude carries most of its noise WITHIN patient (large c),
    # so aggregation averages it away and helps magnitude; phase carries its noise
    # BETWEEN patients (large b), which averaging cannot touch. The interaction is
    # positive by construction, and the specs are tuned so that d_slice < 0 while
    # d_agg > 0 -- the two units return literally opposite answers.
    inv_specs = [(1.45, 0.05, 3.2), (1.05, 1.05, 0.25),   # archX mag / phase
                 (1.40, 0.05, 3.2), (1.02, 1.05, 0.25)]   # archY mag / phase
    truth_I = [_truth_I(inv_specs[0], inv_specs[1]), _truth_I(inv_specs[2], inv_specs[3])]
    c.ok(all(t > 0.1 for t in truth_I),
         "the SIMULATION's true interaction is clearly positive in E2",
         f"true I = {np.round(truth_I, 3).tolist()}")
    y, sc, pid, ts, tp = _sim_methods(rng, inv_specs, n_patients=400, slices=nsl)
    c.ok(all(ts[2 * k] < ts[2 * k + 1] and tp[2 * k] > tp[2 * k + 1] for k in range(2)),
         "and the TRUE magnitude-vs-phase answer is opposite at the two units",
         f"true slice {np.round(ts, 3).tolist()} | agg {np.round(tp, 3).tolist()}")
    al = _al_from_sim(y, sc, pid, names_e)
    pt, bs = point_estimates(al), joint_bootstrap(al, n_boot=nb, seed=33)
    it = condition_interaction(al, pt, bs)
    c.ok(it["n_supported"] == 2,
         "both planted interactions are recovered and survive Holm",
         "; ".join(f"{r['model']} I={r['interaction']:+.3f} "
                   f"[{r['ci_lo']:+.3f}, {r['ci_hi']:+.3f}]" for r in it["rows"]))
    c.ok(it["n_sign_flips"] == 2,
         "and both are flagged as SIGN FLIPS: the units give opposite answers",
         "; ".join(f"{r['model']} d_slice={r['d_slice']:+.3f} d_agg={r['d_agg']:+.3f}"
                   for r in it["rows"]))
    c.ok(it["mean"]["ci_lo"] > 0,
         "the mean interaction excludes zero", f"{it['mean']['interaction']:+.3f} "
         f"[{it['mean']['ci_lo']:+.3f}, {it['mean']['ci_hi']:+.3f}]")
    c.ok(all(min(abs(r["interaction"] - t) for t in truth_I) < 0.06 for r in it["rows"]),
         "the estimated interactions land on the closed-form truth",
         f"estimated {[round(r['interaction'], 3) for r in it['rows']]} vs true "
         f"{np.round(truth_I, 3).tolist()}")
    it_swapped = condition_interaction(al, pt, bs, contrast=("phase", "magnitude"))
    c.ok(all(abs(a["interaction"] + b_["interaction"]) < 1e-12
             for a, b_ in zip(it["rows"], it_swapped["rows"])),
         "swapping the contrast negates the interaction exactly -- no orientation bug")

    # E3 -- CEILING. Both conditions saturate the aggregated unit, so d_agg is pinned
    # at 0 and I = -d_slice with almost no spread. The bootstrap will call that
    # significant; it must not be counted as a unit effect.
    ceil_specs = [(6.0, 0.05, 1.4), (6.0, 0.05, 0.9)]
    y, sc, pid, ts, tp = _sim_methods(rng, ceil_specs, n_patients=200, slices=nsl)
    al = _al_from_sim(y, sc, pid, ["archZ/imagenet/magnitude", "archZ/imagenet/phase"])
    pt, bs = point_estimates(al), joint_bootstrap(al, n_boot=nb, seed=35)
    it = condition_interaction(al, pt, bs)
    c.ok(min(pt["auc_patient"]) >= 0.995,
         "the SIMULATION saturates the aggregated unit, as this case needs",
         f"aggregated AUCs {np.round(pt['auc_patient'], 4).tolist()}")
    c.ok(it["rows"][0]["ceilinged"] and it["n_supported"] == 0,
         "a ceilinged interaction is labelled and NOT counted as supported, however "
         "tight its interval",
         f"I = {it['rows'][0]['interaction']:+.4f} "
         f"[{it['rows'][0]['ci_lo']:+.4f}, {it['rows'][0]['ci_hi']:+.4f}], "
         f"p = {it['rows'][0]['p']:.4f}")

    # ------------------------------------------------- alignment guards
    print("\n[8] alignment refuses rather than repairs")
    r1 = {"test": {"probs": [0.1, 0.2], "labels": [0, 1], "patient_ids": ["a", "b"],
                   "cache_idx": [0, 1]}}
    r2 = {"test": {"probs": [0.3, 0.4], "labels": [1, 0], "patient_ids": ["a", "b"],
                   "cache_idx": [0, 1]}}
    al_bad = align_methods([{"name": "x", "run": r1}, {"name": "y", "run": r2}], None)
    c.ok(not al_bad["ok"] and "labels disagree" in (al_bad["reason"] or ""),
         "labels disagreeing for one cache_idx is a refusal", al_bad["reason"])
    r3 = {"test": {"probs": [0.3], "labels": [0], "patient_ids": ["a"], "cache_idx": [0]}}
    al_part = align_methods([{"name": "x", "run": r1}, {"name": "y", "run": r3}], None)
    c.ok(not al_part["ok"],
         "a method scored on a different case set does not silently enter the ranking",
         al_part["reason"])

    print("\n" + S._rule(100, "="))
    print(f"self-test: {c.passed} passed, {c.failed} failed")
    print(S._rule(100, "="))
    return 1 if c.failed else 0


# ==========================================================================
# CLI
# ==========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description="PhaseDx stage 16: does the evaluation unit change the ranking?")
    p.add_argument("--arch-dir", default=str(_DEFAULT_ARCH_DIR),
                   help="architecture-zoo tree (one subdirectory per architecture)")
    p.add_argument("--results-dir", default=str(_DEFAULT_RESULTS_DIR),
                   help="main sweep tree")
    p.add_argument("--cache-dir", default=str(_DEFAULT_CACHE_DIR))
    p.add_argument("--cohort-dir", default=str(_DEFAULT_COHORT_DIR))
    p.add_argument("--cohorts", nargs="*", default=None,
                   help=f"default: {' '.join(ALL_COHORTS)}")
    p.add_argument("--seeds", nargs="*", type=int, default=[42],
                   help="training seeds admitted as methods (default 42, the seed the "
                        "architecture zoo was run at); pass more to treat each "
                        "(arch, condition, seed) as its own method")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0, help="bootstrap RNG seed")
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--no-verify", action="store_true",
                   help="skip the s04_stats.cluster_bootstrap_diff cross-check")
    p.add_argument("--out", default=None, help="output JSON path")
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--sim-seed", type=int, default=20260729,
                   help="RNG seed for the --self-test simulations; the checks are "
                        "designed to hold for any seed, so varying it is a real test")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s %(message)s")
    if args.self_test:
        return self_test(quick=args.quick, sim_seed=args.sim_seed)

    sources = [Path(args.arch_dir) / d.name
               for d in sorted(Path(args.arch_dir).iterdir())
               if d.is_dir()] if Path(args.arch_dir).is_dir() else []
    sources.append(Path(args.results_dir))

    cohorts = args.cohorts or ALL_COHORTS
    results = []
    for cohort in cohorts:
        res = analyse_cohort(
            cohort, sources, Path(args.cache_dir), Path(args.cohort_dir),
            n_boot=args.n_boot, seed=args.seed, alpha=args.alpha,
            seeds=tuple(args.seeds) if args.seeds else None,
            verify=not args.no_verify)
        results.append(res)
        print_cohort(res)
    print_summary(results)

    out = Path(args.out) if args.out else Path(args.results_dir).parent / "rankinversion.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"schema": "phasedx.s16.rankinversion.v1",
         "n_boot": args.n_boot, "seed": args.seed, "alpha": args.alpha,
         "seeds_admitted": args.seeds,
         "sources": [str(s) for s in sources],
         "cohorts": results},
        indent=2, default=S._json_default))
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
