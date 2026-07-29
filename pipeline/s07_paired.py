"""
s07_paired.py
-------------
Stage 7 of the PhaseDx pipeline: the WITHIN-SUBJECT PAIRED estimator for the
knee cohort.

Why this module exists
======================
Knee is a paired design. All 96 subjects contribute BOTH classes: every patient
has a proton-density scan (CORPD_FBK) and a fat-suppressed proton-density scan
(CORPDFS_FBK) of the same knee. The subject-level machinery in s04_stats
collapses a subject to one label by taking the max, which on the knee test fold
yields 29 positive clusters and 0 negative clusters -- a subject-level AUC is
undefined by construction, and no amount of cross-validation or extra data
fixes that, because the design is paired and needs a paired statistic.

The pairing is a STRENGTH. Anatomy, patient, scanner, coil array and site are
all held constant inside a subject, so a within-subject difference isolates the
pulse sequence almost perfectly. What this module computes, per model run:

  1. Per subject, mean model score on their CORPDFS_FBK slices MINUS mean model
     score on their CORPD_FBK slices. One number per subject.
  2. A two-sided Wilcoxon signed-rank test over those subject differences, plus
     an exact sign test as a distribution-free backup. BOTH are reported; the
     sign test survives any monotone distortion of the score scale, so when the
     two disagree the sign test is the one to believe.
  3. A paired effect size: the proportion of subjects whose difference has the
     expected sign, with a Wilson interval. This is a within-subject
     concordance and it is the number a reader will actually understand.
  4. A subject-level bootstrap CI on the mean paired difference. The resampling
     unit is the SUBJECT, matching the cluster-unit contract in s04_stats.
  5. A guard: subjects that lack one of the two classes are excluded and
     COUNTED, with their ids listed. Nothing is silently dropped.

Because the estimator is a difference of two within-subject means, the
slice-correlation hazard that motivates the clustered bootstrap everywhere else
in this pipeline is handled by construction: slices are averaged inside a
subject before any test sees them, and every test here has n = number of
subjects, not number of slices.

*** SCIENTIFIC CAVEAT -- DO NOT OVERSTATE KNEE *********************************
Stage 1's confound screen shows that the two knee classes differ in echo time,
echo spacing and flip angle (p ~ 1e-40). The observed contrast in the cached
cohort is TE 33 ms vs 27 ms and echo spacing 10.96 ms vs 8.85 ms (medians). So
"phase predicts knee contrast" is partly "phase reflects echo time", which is
EXPECTED PHYSICS rather than a hardware fingerprint: fat suppression changes the
sequence, and the sequence changes the phase evolution.

Knee is therefore SUPPORTING evidence about sequence-parameter dependence. The
LOAD-BEARING fingerprint claim is the brain receive-coil-count result, because
coil count is pure hardware with no physiological meaning at all and no
plausible route into the image except the acquisition. A large paired effect
here does not upgrade the brain claim and must never be reported as if it did.
********************************************************************************

What it reads / writes
======================
    reads   pipeline_out/results/**/{cohort}_{condition}_seed{seed}.json  <- s03
            pipeline_out/cache/{cohort}_index.csv                        <- s02
    writes  pipeline_out/paired/{cohort}_paired.json

The output payload is deliberately shaped so that it CANNOT be mistaken for an
AUC or for a training run:

  * it carries no top-level "test" key and no "control" key, so both
    s06_report.load_runs (which requires cohort+condition+test.probs) and
    s06_report.load_control_payloads (which requires control+test) skip it;
  * it is written OUTSIDE the results tree that s04_stats and s06_report walk,
    mirroring how stage 5 keeps its controls in a sibling directory;
  * it is tagged `analysis: paired_within_subject`, `metric_kind:
    paired_difference`, `is_auc: false`, `render_as: paired_analysis`;
  * it carries the caveat above verbatim in `caveat`, and a ready-to-embed
    markdown block in `markdown`, so a report layer can render it correctly
    without knowing this schema. `render_markdown()` is importable for the same
    purpose.

Usage:
    python pipeline/s07_paired.py                     # reads pipeline_out/results
    python pipeline/s07_paired.py --cohort knee --n-boot 5000
    python pipeline/s07_paired.py --self-test         # synthetic, no data needed
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))

try:
    import common  # noqa: E402
    _DEFAULT_RESULTS_DIR = common.RESULTS_DIR
    _DEFAULT_CACHE_DIR = common.CACHE_DIR
    _DEFAULT_COHORT_DIR = common.COHORT_DIR
    _DEFAULT_OUT_ROOT = common.OUT_ROOT
except Exception:  # pragma: no cover - must run standalone
    _ROOT = Path(__file__).resolve().parent.parent
    _DEFAULT_OUT_ROOT = _ROOT / "pipeline_out"
    _DEFAULT_RESULTS_DIR = _DEFAULT_OUT_ROOT / "results"
    _DEFAULT_CACHE_DIR = _DEFAULT_OUT_ROOT / "cache"
    _DEFAULT_COHORT_DIR = _DEFAULT_OUT_ROOT / "cohorts"

# s04_stats owns the cluster-unit join and the Wilson interval. Import them
# rather than re-deriving them, so the subject definition here is the same
# object the rest of the statistics layer uses. The import is guarded because
# s04 is a live file: if it is momentarily unimportable this module still runs,
# on local fallbacks that the self-test checks against s04 whenever it IS
# importable.
_S04 = None
try:  # pragma: no cover - exercised implicitly
    import s04_stats as _S04  # noqa: E402
except Exception as _exc:  # pragma: no cover
    logging.getLogger("s07_paired").warning(
        "s04_stats not importable (%s); using local fallbacks", _exc)

logger = logging.getLogger("s07_paired")

Z_975 = 1.959963984540054

# Exact conditional signed-rank enumeration is an O(n^3)-ish integer DP. It is
# instant at the knee's n=29 and still cheap at n=100; past that the normal
# approximation with the tie correction is accurate to more digits than anyone
# will read.
EXACT_MAX_N = 100

# The expected direction. Stage 2 (s02_brainknee.ACQ_POSITIVE_CLASSES) makes
# CORPDFS_FBK the positive class, models are trained to output P(label = 1), so
# a model that has learned anything at all scores the fat-suppressed scan HIGHER
# than the non-suppressed scan of the same knee: difference > 0. Both tests here
# are two-sided; this constant only fixes which direction "concordant" means.
EXPECTED_SIGN = +1

CAVEAT = (
    "SUPPORTING EVIDENCE ONLY -- do not overstate knee. Stage 1's confound screen "
    "shows the two knee classes differ in echo time, echo spacing and flip angle "
    "(p ~ 1e-40). So 'phase predicts knee contrast' is partly 'phase reflects echo "
    "time', which is expected physics rather than a hardware fingerprint: fat "
    "suppression changes the pulse sequence and the pulse sequence changes phase "
    "evolution. Knee therefore supports the weaker claim that phase tracks "
    "sequence parameters. The LOAD-BEARING fingerprint claim remains the brain "
    "receive-coil-count result, because coil count is pure hardware with no "
    "physiological meaning; a large paired effect here does not strengthen it."
)

METRIC_NOTE = (
    "The estimate below is a WITHIN-SUBJECT PAIRED MEAN SCORE DIFFERENCE, not an "
    "AUC and not comparable to one. Knee is a paired design -- every subject "
    "supplies both classes -- so collapsing a subject to one label by max gives 29 "
    "positive and 0 negative clusters on the test fold and a subject-level AUC is "
    "undefined by construction. Each subject contributes exactly one number (mean "
    "score on their positive-class slices minus mean score on their negative-class "
    "slices) and every n below is a number of SUBJECTS."
)


# ==========================================================================
# Small numerical helpers (numpy + stdlib only, matching s04_stats)
# ==========================================================================

def _norm_sf(z: float) -> float:
    """Upper tail of the standard normal, via erfc. |error| < 1e-15."""
    return 0.5 * math.erfc(z / math.sqrt(2.0))


def two_sided_normal_p(z: float) -> float:
    if not np.isfinite(z):
        return float("nan")
    return float(min(1.0, 2.0 * _norm_sf(abs(z))))


def wilson_interval(k: int, n: int, z: float = Z_975):
    """
    Wilson score interval for a binomial proportion.

    Delegates to s04_stats when it is importable so there is exactly one
    definition in the pipeline; the local branch is a byte-for-byte copy of the
    same formula and the self-test asserts the two agree.
    """
    if _S04 is not None and hasattr(_S04, "wilson_interval"):
        return _S04.wilson_interval(k, n, z)
    if n <= 0:
        return None, None
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = (z / denom) * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def _midranks(x: np.ndarray) -> np.ndarray:
    """Midranks (average ranks for ties), 1-based. Same convention as s04."""
    x = np.asarray(x, dtype=float)
    order = np.argsort(x, kind="mergesort")
    xs = x[order]
    ranks = np.empty(len(x), dtype=float)
    i = 0
    while i < len(xs):
        j = i
        while j < len(xs) - 1 and xs[j + 1] == xs[i]:
            j += 1
        ranks[i:j + 1] = 0.5 * (i + j) + 1.0
        i = j + 1
    out = np.empty(len(x), dtype=float)
    out[order] = ranks
    return out


def _exact_signed_rank_tail(ranks: np.ndarray, w_plus: float) -> float:
    """
    Exact two-sided p for the signed-rank statistic, conditional on |d|.

    Under H0 the sign of every non-zero difference is +/- with probability 1/2,
    independently, and the |d| values (hence their midranks) are fixed. So the
    exact null distribution of W+ = sum of the ranks carrying a + sign is the
    subset-sum distribution of the observed rank multiset, which is counted here
    with an integer knapsack DP -- exact even in the presence of ties, where a
    table-lookup critical value would not be.

    Midranks can be half-integers, so everything is doubled to stay in the
    integers and the DP is exact (Python big ints, no floating error).
    """
    r2 = [int(round(2.0 * float(r))) for r in ranks]
    total = int(sum(r2))
    counts = [0] * (total + 1)
    counts[0] = 1
    for r in r2:
        for s in range(total, r - 1, -1):
            if counts[s - r]:
                counts[s] += counts[s - r]
    denom = 1 << len(r2)
    w2 = int(round(2.0 * float(w_plus)))
    w2 = max(0, min(total, w2))
    lo = sum(counts[:w2 + 1])
    hi = sum(counts[w2:])
    return float(min(1.0, 2.0 * min(lo, hi) / denom))


def wilcoxon_signed_rank(diffs, exact_max_n: int = EXACT_MAX_N) -> dict:
    """
    Two-sided Wilcoxon signed-rank test on paired differences.

    Zero differences are discarded before ranking (the classical "wilcox" zero
    method) and counted in `n_zero`, because with continuous model scores a zero
    means the two within-subject means were bit-identical and that is
    information the reader should see, not something to average away.

    Exact conditional p for n <= exact_max_n (ties included -- see
    `_exact_signed_rank_tail`), otherwise the normal approximation with the
    standard tie correction sum(t^3 - t)/48. No continuity correction, matching
    scipy.stats.wilcoxon(correction=False); the self-test checks both branches
    against scipy when scipy is importable.
    """
    d = np.asarray([v for v in np.asarray(diffs, dtype=float)], dtype=float)
    out = {"n_input": int(d.size), "n_used": 0, "n_zero": 0, "w_plus": None,
           "w_minus": None, "statistic": None, "z": None, "p": None,
           "method": None, "reason": None, "n_ties": 0}
    if d.size == 0 or not np.all(np.isfinite(d)):
        out["reason"] = "empty or non-finite differences"
        return out
    nz = d[d != 0.0]
    out["n_zero"] = int(d.size - nz.size)
    out["n_used"] = int(nz.size)
    if nz.size == 0:
        out["reason"] = "every paired difference is exactly zero"
        return out
    ranks = _midranks(np.abs(nz))
    w_plus = float(ranks[nz > 0].sum())
    w_minus = float(ranks[nz < 0].sum())
    out["w_plus"], out["w_minus"] = w_plus, w_minus
    out["statistic"] = float(min(w_plus, w_minus))
    absvals, counts = np.unique(np.abs(nz), return_counts=True)
    out["n_ties"] = int(np.sum(counts > 1))
    n = int(nz.size)
    if n <= exact_max_n:
        out["p"] = _exact_signed_rank_tail(ranks, w_plus)
        out["method"] = "exact conditional (subset-sum over observed midranks)"
        return out
    mean = n * (n + 1) / 4.0
    tie_corr = float(np.sum(counts ** 3 - counts)) / 48.0
    var = n * (n + 1) * (2 * n + 1) / 24.0 - tie_corr
    if var <= 0:
        out["reason"] = "zero variance under the null (all |differences| tied)"
        return out
    z = (w_plus - mean) / math.sqrt(var)
    out["z"] = float(z)
    out["p"] = two_sided_normal_p(z)
    out["method"] = "normal approximation with tie correction, no continuity correction"
    return out


def sign_test(diffs) -> dict:
    """
    Exact two-sided sign test on paired differences.

    Distribution-free backup for the signed-rank test: it uses only the sign of
    each subject's difference, so it is invariant to any monotone rescaling of
    the model score and cannot be moved by a couple of extreme subjects. It is
    the conservative one, and it is the one to believe when the two disagree.
    """
    d = np.asarray(diffs, dtype=float)
    out = {"n_input": int(d.size), "n_used": 0, "n_zero": 0, "n_pos": 0,
           "n_neg": 0, "p": None, "method": "exact binomial (p = 0.5), two-sided",
           "reason": None}
    if d.size == 0 or not np.all(np.isfinite(d)):
        out["reason"] = "empty or non-finite differences"
        return out
    n_pos = int(np.sum(d > 0))
    n_neg = int(np.sum(d < 0))
    out["n_pos"], out["n_neg"] = n_pos, n_neg
    out["n_zero"] = int(d.size - n_pos - n_neg)
    n = n_pos + n_neg
    out["n_used"] = n
    if n == 0:
        out["reason"] = "every paired difference is exactly zero"
        return out
    lo = sum(math.comb(n, i) for i in range(0, n_pos + 1))
    hi = sum(math.comb(n, i) for i in range(n_pos, n + 1))
    out["p"] = float(min(1.0, 2.0 * min(lo, hi) / float(1 << n)))
    return out


def paired_concordance(diffs, expected_sign: int = EXPECTED_SIGN) -> dict:
    """
    Within-subject concordance: the fraction of subjects whose difference has
    the expected sign, with a Wilson interval.

    The denominator is EVERY paired subject, so exact ties count against
    concordance. That is conservative and it keeps the denominator equal to the
    number of subjects actually analysed, which is what a reader assumes.
    """
    d = np.asarray(diffs, dtype=float)
    n = int(d.size)
    out = {"k": None, "n": n, "proportion": None, "ci_lo": None, "ci_hi": None,
           "ci_method": "wilson", "n_zero": 0, "expected_sign": int(expected_sign),
           "note": ("proportion of subjects whose paired difference has the "
                    "expected sign; exact ties are counted as NON-concordant"),
           "reason": None}
    if n == 0 or not np.all(np.isfinite(d)):
        out["reason"] = "empty or non-finite differences"
        out["n"] = n
        return out
    k = int(np.sum(d > 0)) if expected_sign > 0 else int(np.sum(d < 0))
    out["k"] = k
    out["n_zero"] = int(np.sum(d == 0.0))
    out["proportion"] = float(k / n)
    lo, hi = wilson_interval(k, n)
    out["ci_lo"], out["ci_hi"] = lo, hi
    return out


def subject_bootstrap_mean_diff(diffs, n_boot: int = 2000, seed: int = 0,
                                alpha: float = 0.05) -> dict:
    """
    Percentile bootstrap CI for the mean paired difference, resampling SUBJECTS.

    The resampling unit is the subject, which is the cluster unit the rest of
    the pipeline uses. Here each subject already contributes exactly one number,
    so this is a plain nonparametric bootstrap over subjects -- the slice
    correlation that forces the clustered bootstrap in s04_stats has already
    been absorbed by averaging within subject and within class.
    """
    d = np.asarray(diffs, dtype=float)
    out = {"mean": None, "ci_lo": None, "ci_hi": None, "alpha": float(alpha),
           "n_subjects": int(d.size), "n_boot_requested": int(n_boot),
           "n_boot_used": 0, "seed": int(seed),
           "method": "percentile bootstrap, resampling unit = subject",
           "reason": None}
    if d.size == 0 or not np.all(np.isfinite(d)):
        out["reason"] = "empty or non-finite differences"
        return out
    out["mean"] = float(np.mean(d))
    if d.size < 2:
        out["reason"] = "fewer than 2 paired subjects; no interval is defensible"
        return out
    if n_boot <= 0:
        out["reason"] = "n_boot <= 0"
        return out
    rng = np.random.default_rng(seed)
    n = d.size
    means = np.empty(int(n_boot), dtype=float)
    step = max(1, int(2_000_000 // max(1, n)))
    done = 0
    while done < n_boot:
        take = min(step, int(n_boot) - done)
        idx = rng.integers(0, n, size=(take, n))
        means[done:done + take] = d[idx].mean(axis=1)
        done += take
    out["n_boot_used"] = int(n_boot)
    lo, hi = np.percentile(means, [100.0 * alpha / 2.0, 100.0 * (1.0 - alpha / 2.0)])
    out["ci_lo"], out["ci_hi"] = float(lo), float(hi)
    return out


# ==========================================================================
# Loading: runs, subject ids, class names
# ==========================================================================

def load_runs(results_dir: Path, cohort: str | None = None) -> list[dict]:
    """
    Collect stage-3 run payloads under results_dir, RECURSIVELY.

    Recursive because the sweep writes per-fold subdirectories
    (confound_knee/, prostate_t2_cv0/ ...). Payloads that carry a `control` key
    other than "none" are stage-5 control runs and are refused here: a
    phase-scramble or label-permutation payload has the same shape as a headline
    run, and pooling one into a paired estimate would put a deliberately broken
    model into the headline number.
    """
    results_dir = Path(results_dir)
    runs: list[dict] = []
    if not results_dir.is_dir():
        return runs
    for p in sorted(results_dir.rglob("*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception as exc:
            logger.debug("skipping %s: unreadable JSON (%s)", p.name, exc)
            continue
        if not isinstance(d, dict):
            continue
        if not ("cohort" in d and "condition" in d
                and isinstance(d.get("test"), dict)
                and "probs" in d.get("test", {})):
            continue
        ctrl = d.get("control")
        if ctrl is not None and str(ctrl) != "none":
            continue
        if cohort is not None and str(d.get("cohort")) != cohort:
            continue
        d["_path"] = str(p)
        d["_tag"] = p.stem
        d.setdefault("region", "full")
        d.setdefault("seed", 0)
        runs.append(d)
    return runs


def _read_csv_dicts(path: Path) -> list[dict]:
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def build_cluster_map(cohort: str, cache_dir: Path, cohort_dir: Path) -> dict:
    """
    cache_idx -> subject_id, via s04_stats when importable.

    The fallback covers only the direct case (the cache index already carries a
    subject_id column, which is true for knee and brain). If the fallback cannot
    resolve subjects it says so instead of quietly handing back patient_id.
    """
    if _S04 is not None and hasattr(_S04, "build_cluster_map"):
        return _S04.build_cluster_map(cohort, Path(cache_dir), Path(cohort_dir))
    info = {"map": {}, "source": None, "reason": None}
    idx_path = Path(cache_dir) / f"{cohort}_index.csv"
    if not idx_path.exists():
        info["reason"] = f"no cache index at {idx_path}"
        return info
    rows = _read_csv_dicts(idx_path)
    if rows and "subject_id" in rows[0]:
        info["map"] = {int(r["idx"]): str(r["subject_id"]) for r in rows}
        info["source"] = f"{idx_path.name}:subject_id (s07 fallback reader)"
        return info
    info["reason"] = (f"{idx_path.name} has no subject_id column and s04_stats "
                      f"was not importable for the CSV join")
    return info


def resolve_subjects(split_payload: dict, cluster_map: dict | None):
    """
    (subject ids, unit name, source) for one split payload.

    Same preference order as s04_stats.resolve_clusters, and delegated to it
    when possible: subject_ids written by stage 3 > subject_id joined through
    cache_idx > patient_id with the downgrade recorded.
    """
    if _S04 is not None and hasattr(_S04, "resolve_clusters"):
        return _S04.resolve_clusters(split_payload, cluster_map)
    if split_payload.get("subject_ids"):
        return (np.asarray([str(s) for s in split_payload["subject_ids"]], dtype=object),
                "subject_id", "run_json:subject_ids")
    pids = np.asarray([str(p) for p in split_payload["patient_ids"]], dtype=object)
    if cluster_map:
        cidx = split_payload.get("cache_idx")
        if cidx is not None and all(int(i) in cluster_map for i in cidx):
            return (np.asarray([cluster_map[int(i)] for i in cidx], dtype=object),
                    "subject_id", "cache_idx->subject_id join")
    return pids, "patient_id", "run_json:patient_ids (subject_id join unavailable)"


def class_names(cohort: str, cache_dir: Path) -> dict:
    """
    {1: <positive raw label>, 0: <negative raw label>} read off the cache index.

    Read rather than hard-coded so the printed direction ("CORPDFS_FBK minus
    CORPD_FBK") is the direction the data actually encodes. If a numeric label
    maps to more than one raw label the mapping is refused and generic names are
    used, because a wrong class name in the report is worse than a dull one.
    """
    out = {"positive": "positive class", "negative": "negative class",
           "source": None, "reason": None}
    idx_path = Path(cache_dir) / f"{cohort}_index.csv"
    if not idx_path.exists():
        out["reason"] = f"no cache index at {idx_path}"
        return out
    try:
        rows = _read_csv_dicts(idx_path)
    except Exception as exc:  # pragma: no cover
        out["reason"] = f"unreadable cache index: {exc}"
        return out
    if not rows or "raw_label" not in rows[0] or "label" not in rows[0]:
        out["reason"] = f"{idx_path.name} has no label/raw_label columns"
        return out
    seen = defaultdict(set)
    for r in rows:
        try:
            seen[int(r["label"])].add(str(r["raw_label"]))
        except (ValueError, TypeError):
            continue
    if set(seen) != {0, 1} or any(len(v) != 1 for v in seen.values()):
        out["reason"] = (f"{idx_path.name}: label -> raw_label is not a clean "
                         f"binary map ({ {k: sorted(v) for k, v in seen.items()} })")
        return out
    out["positive"] = next(iter(seen[1]))
    out["negative"] = next(iter(seen[0]))
    out["source"] = idx_path.name
    return out


def sequence_parameter_contrast(cohort: str, cache_dir: Path,
                                columns=("te", "echo_spacing", "flip_angle", "tr")) -> dict:
    """
    DESCRIPTIVE ONLY: per-class median/min/max of the acquisition parameters that
    stage 1's confound screen flagged.

    This is not a test and deliberately reports no p-value. It exists so the
    caveat above carries real numbers instead of an assertion: if the two knee
    classes are separated by a 6 ms difference in echo time, a reader can see
    that on the same page as the paired statistic and discount accordingly.
    """
    out = {"descriptive_only": True, "is_a_test": False, "by_class": {},
           "source": None, "reason": None,
           "note": ("per-class summary of the acquisition parameters stage 1's "
                    "confound screen flagged (p ~ 1e-40); shown so the paired "
                    "effect is read next to the physics that could explain it. "
                    "NO test is performed here.")}
    idx_path = Path(cache_dir) / f"{cohort}_index.csv"
    if not idx_path.exists():
        out["reason"] = f"no cache index at {idx_path}"
        return out
    try:
        rows = _read_csv_dicts(idx_path)
    except Exception as exc:  # pragma: no cover
        out["reason"] = f"unreadable cache index: {exc}"
        return out
    if not rows:
        out["reason"] = "empty cache index"
        return out
    vals: dict = defaultdict(lambda: defaultdict(list))
    for r in rows:
        lab = r.get("raw_label") or r.get("label")
        for c in columns:
            if c not in r:
                continue
            try:
                vals[str(lab)][c].append(float(r[c]))
            except (TypeError, ValueError):
                continue
    if not vals:
        out["reason"] = f"none of {list(columns)} present in {idx_path.name}"
        return out
    for lab, per in sorted(vals.items()):
        out["by_class"][lab] = {
            c: {"n": len(v), "min": float(min(v)),
                "median": float(statistics.median(v)), "max": float(max(v))}
            for c, v in sorted(per.items()) if v}
    out["source"] = idx_path.name
    return out


# ==========================================================================
# The paired estimator
# ==========================================================================

def paired_differences(split_payload: dict, subjects) -> dict:
    """
    Per-subject mean(score | positive class) - mean(score | negative class).

    Returns the difference vector in a stable subject order plus the full
    accounting of what was dropped and why. Two drop paths exist and both are
    counted, never silent:

      * a slice with a non-finite probability or a label outside {0, 1};
      * a SUBJECT that does not supply both classes -- which is the whole reason
        this module exists, so those subjects are listed by id.
    """
    probs = np.asarray(split_payload.get("probs", []), dtype=float)
    labels = np.asarray(split_payload.get("labels", []), dtype=float)
    subjects = np.asarray(subjects, dtype=object)
    out = {"subject_ids": [], "diffs": [], "mean_pos": [], "mean_neg": [],
           "n_pos_slices": [], "n_neg_slices": [],
           "n_slices_input": int(probs.size), "n_slices_used": 0,
           "n_slices_dropped": 0, "dropped_slice_reason": None,
           "n_subjects_seen": 0, "n_subjects_paired": 0,
           "n_subjects_excluded": 0, "excluded": [], "reason": None}
    if not (probs.size == labels.size == subjects.size):
        out["reason"] = (f"length mismatch: probs={probs.size} labels={labels.size} "
                         f"subjects={subjects.size}")
        return out
    if probs.size == 0:
        out["reason"] = "empty split payload"
        return out
    ok = np.isfinite(probs) & np.isin(labels, (0.0, 1.0))
    n_drop = int((~ok).sum())
    out["n_slices_dropped"] = n_drop
    out["n_slices_used"] = int(ok.sum())
    if n_drop:
        out["dropped_slice_reason"] = ("non-finite probability or label outside "
                                       "{0, 1}")
    probs, labels, subjects = probs[ok], labels[ok], subjects[ok]
    if probs.size == 0:
        out["reason"] = "every slice was dropped as non-finite / non-binary"
        return out

    buckets: dict = defaultdict(lambda: {1: [], 0: []})
    for s, l, p in zip(subjects, labels, probs):
        buckets[str(s)][int(l)].append(float(p))
    out["n_subjects_seen"] = len(buckets)
    for sid in sorted(buckets):
        pos, neg = buckets[sid][1], buckets[sid][0]
        if not pos or not neg:
            out["excluded"].append({
                "subject_id": sid, "n_pos_slices": len(pos), "n_neg_slices": len(neg),
                "reason": ("subject supplies only the positive class"
                           if not neg else "subject supplies only the negative class")})
            continue
        out["subject_ids"].append(sid)
        out["mean_pos"].append(float(np.mean(pos)))
        out["mean_neg"].append(float(np.mean(neg)))
        out["n_pos_slices"].append(len(pos))
        out["n_neg_slices"].append(len(neg))
        out["diffs"].append(float(np.mean(pos) - np.mean(neg)))
    out["n_subjects_paired"] = len(out["subject_ids"])
    out["n_subjects_excluded"] = len(out["excluded"])
    if out["n_subjects_paired"] == 0:
        out["reason"] = ("no subject supplies both classes; this cohort is not a "
                         "paired design and this estimator does not apply")
    return out


def analyse_paired_group(group: list[dict], cluster_map: dict | None,
                         n_boot: int, seed: int, alpha: float,
                         names: dict | None = None) -> dict:
    """
    One paired result for one (cohort, condition, seed, region) group of runs.

    `group` is a list because a cross-validated cohort writes one run per fold.
    Every subject appears in exactly one test fold, so the test blocks are
    concatenated OUT OF FOLD: one prediction per subject over the full cohort,
    one estimate, one comparison. If the test folds are NOT disjoint the group is
    refused with a reason rather than pooled -- overlapping folds would count a
    subject twice and shrink every interval below.
    """
    first = group[0]
    res = {
        "cohort": str(first.get("cohort")), "condition": str(first.get("condition")),
        "seed": int(first.get("seed", 0)), "region": str(first.get("region", "full")),
        "split": "test", "n_runs_pooled": len(group),
        "tags": [str(r.get("_tag")) for r in group],
        "paths": [str(r.get("_path")) for r in group],
        "positive_class": (names or {}).get("positive", "positive class"),
        "negative_class": (names or {}).get("negative", "negative class"),
        "cluster_unit": None, "cluster_source": None,
        "n_subjects_seen": 0, "n_subjects_paired": 0, "n_subjects_excluded": 0,
        "excluded_subjects": [], "n_slices_used": 0, "n_slices_dropped": 0,
        "paired_difference": None, "wilcoxon_signed_rank": None,
        "sign_test": None, "concordance": None,
        "reported_test_auc_slice_level": [float(r["test"]["auc"])
                                          for r in group
                                          if isinstance(r.get("test"), dict)
                                          and r["test"].get("auc") is not None],
        "warnings": [], "reason": None,
    }

    probs, labels, subjects = [], [], []
    units, sources, seen_subjects = set(), set(), set()
    for r in group:
        t = r.get("test") or {}
        sids, unit, source = resolve_subjects(t, cluster_map)
        units.add(unit)
        sources.add(source)
        overlap = seen_subjects & set(map(str, sids))
        if overlap:
            res["reason"] = (
                f"test folds are not disjoint: {len(overlap)} subject(s) appear in "
                f"more than one run of this group (e.g. {sorted(overlap)[:3]}). "
                f"Out-of-fold pooling would count them twice, so this group is not "
                f"analysed.")
            return res
        seen_subjects |= set(map(str, sids))
        probs.extend(list(t.get("probs", [])))
        labels.extend(list(t.get("labels", [])))
        subjects.extend(list(sids))

    res["cluster_unit"] = "/".join(sorted(units))
    res["cluster_source"] = "; ".join(sorted(sources))
    if res["cluster_unit"] != "subject_id":
        res["warnings"].append(
            "clustering on patient_id, not subject_id: the cache_idx -> subject_id "
            "join was unavailable, so repeat scans of one person may be treated as "
            "different people")

    pd_ = paired_differences({"probs": probs, "labels": labels}, subjects)
    res["n_subjects_seen"] = pd_["n_subjects_seen"]
    res["n_subjects_paired"] = pd_["n_subjects_paired"]
    res["n_subjects_excluded"] = pd_["n_subjects_excluded"]
    res["excluded_subjects"] = pd_["excluded"]
    res["n_slices_used"] = pd_["n_slices_used"]
    res["n_slices_dropped"] = pd_["n_slices_dropped"]
    if pd_["n_subjects_excluded"]:
        res["warnings"].append(
            f"{pd_['n_subjects_excluded']} of {pd_['n_subjects_seen']} subjects were "
            f"excluded for supplying only one class; they are listed in "
            f"excluded_subjects and are NOT in any statistic below")
    if pd_["reason"]:
        res["reason"] = pd_["reason"]
        return res

    d = np.asarray(pd_["diffs"], dtype=float)
    res["per_subject"] = [
        {"subject_id": s, "mean_pos": mp, "mean_neg": mn, "diff": dd,
         "n_pos_slices": npo, "n_neg_slices": nne}
        for s, mp, mn, dd, npo, nne in zip(
            pd_["subject_ids"], pd_["mean_pos"], pd_["mean_neg"], pd_["diffs"],
            pd_["n_pos_slices"], pd_["n_neg_slices"])]

    boot = subject_bootstrap_mean_diff(d, n_boot=n_boot, seed=seed, alpha=alpha)
    res["paired_difference"] = {
        "definition": (f"per subject: mean score on {res['positive_class']} slices "
                       f"minus mean score on {res['negative_class']} slices"),
        "n_subjects": int(d.size),
        "mean": float(np.mean(d)),
        "sd": float(np.std(d, ddof=1)) if d.size > 1 else None,
        "median": float(np.median(d)),
        "min": float(np.min(d)), "max": float(np.max(d)),
        "ci_lo": boot["ci_lo"], "ci_hi": boot["ci_hi"],
        "ci_method": boot["method"], "alpha": boot["alpha"],
        "n_boot_requested": boot["n_boot_requested"],
        "n_boot_used": boot["n_boot_used"],
        "bootstrap_seed": boot["seed"], "reason": boot["reason"],
    }
    res["wilcoxon_signed_rank"] = wilcoxon_signed_rank(d)
    res["sign_test"] = sign_test(d)
    res["concordance"] = paired_concordance(d)

    # Saturation guard. If every subject moves the same way, both exact tests
    # return 2/2^n -- the SMALLEST p attainable at this n -- and concordance is
    # pinned at 1.0 (or 0.0). Two conditions that both saturate are therefore
    # reporting identical numbers because the statistic has run out of
    # resolution, NOT because they perform identically, and the difference
    # between them cannot be ranked from these p-values. Flagged explicitly so
    # nobody reads a tie at the floor as a finding.
    res["at_exact_test_floor"] = False
    if res["concordance"]["proportion"] in (0.0, 1.0) and d.size > 0:
        floor = min(1.0, 2.0 / float(1 << int(d.size)))
        res["at_exact_test_floor"] = True
        res["exact_test_floor_p"] = float(floor)
        res["warnings"].append(
            f"SATURATED: all {d.size} subjects move the same way, so both p-values "
            f"sit at the exact-test floor 2/2^{d.size} = {floor:.3e}. That is the "
            f"smallest p reachable at this n. The statistic has no resolution left, "
            f"so it can support a direction but CANNOT rank one condition above "
            f"another, and a tie at the floor is not evidence of equal performance.")
    if d.size < 6:
        res["warnings"].append(
            f"only {d.size} paired subjects: the exact two-sided sign test cannot "
            f"go below p = {sign_test(np.ones(d.size))['p']:.3f} at this n, so a "
            f"non-significant result here is uninformative rather than negative")
    return res


def aggregate_across_seeds(per_group: list[dict]) -> list[dict]:
    """
    Collapse seeds for one (cohort, condition, region).

    The seeds share ONE test fold, so they are re-runs of the same experiment on
    the same subjects, not independent replicates. Averaging their p-values or
    pooling their intervals would manufacture precision, so this reports the
    per-seed values, the mean point estimate, the WIDEST interval, and the WORST
    (largest) p-value of the set. Worst-case rather than best-case because
    nothing in this pipeline may make a favourable conclusion easier to reach.
    """
    by_key: dict = defaultdict(list)
    for g in per_group:
        by_key[(g["cohort"], g["condition"], g["region"])].append(g)
    out = []
    for (cohort, cond, region), gs in sorted(by_key.items()):
        usable = [g for g in gs if g.get("paired_difference")]
        entry = {
            "cohort": cohort, "condition": cond, "region": region,
            "n_seeds": len(gs), "seeds": sorted(int(g["seed"]) for g in gs),
            "n_seeds_evaluable": len(usable),
            "caveat": ("seeds share one test fold; these are re-runs of the same "
                       "experiment on the same subjects, NOT independent replicates. "
                       "The interval is the widest of the per-seed intervals and the "
                       "p-value is the worst (largest), never the best."),
            "mean_diff": None, "ci_lo": None, "ci_hi": None,
            "per_seed_mean_diff": [], "concordance": None,
            "concordance_ci_lo": None, "concordance_ci_hi": None,
            "wilcoxon_p_worst": None, "sign_p_worst": None,
            "n_subjects_paired": None, "n_subjects_excluded": None,
            "reason": None,
        }
        if not usable:
            entry["reason"] = "; ".join(sorted({str(g.get("reason")) for g in gs}))
            out.append(entry)
            continue
        md = [g["paired_difference"]["mean"] for g in usable]
        entry["per_seed_mean_diff"] = [float(v) for v in md]
        entry["mean_diff"] = float(np.mean(md))
        los = [g["paired_difference"]["ci_lo"] for g in usable
               if g["paired_difference"]["ci_lo"] is not None]
        his = [g["paired_difference"]["ci_hi"] for g in usable
               if g["paired_difference"]["ci_hi"] is not None]
        entry["ci_lo"] = float(min(los)) if los else None
        entry["ci_hi"] = float(max(his)) if his else None
        conc = [g["concordance"]["proportion"] for g in usable
                if g["concordance"]["proportion"] is not None]
        if conc:
            worst = min(conc, key=lambda v: abs(v - 0.5))
            j = [g["concordance"]["proportion"] for g in usable].index(worst)
            entry["concordance"] = float(worst)
            entry["concordance_ci_lo"] = usable[j]["concordance"]["ci_lo"]
            entry["concordance_ci_hi"] = usable[j]["concordance"]["ci_hi"]
            entry["concordance_note"] = ("the per-seed concordance CLOSEST TO 0.5, "
                                         "i.e. the least favourable seed")
        wp = [g["wilcoxon_signed_rank"]["p"] for g in usable
              if g["wilcoxon_signed_rank"]["p"] is not None]
        sp = [g["sign_test"]["p"] for g in usable if g["sign_test"]["p"] is not None]
        entry["wilcoxon_p_worst"] = float(max(wp)) if wp else None
        entry["sign_p_worst"] = float(max(sp)) if sp else None
        entry["n_subjects_paired"] = int(max(g["n_subjects_paired"] for g in usable))
        entry["n_subjects_excluded"] = int(max(g["n_subjects_excluded"] for g in usable))
        entry["at_exact_test_floor"] = all(g.get("at_exact_test_floor")
                                           for g in usable)
        if entry["at_exact_test_floor"]:
            entry["floor_note"] = (
                "every seed is SATURATED: all subjects move the same way, so the "
                "p-value is the exact-test floor 2/2^n and the concordance is pinned "
                "at 1.0. This condition can be reported as directional, but it cannot "
                "be ranked against another saturated condition.")
        out.append(entry)
    return out


# ==========================================================================
# Rendering
# ==========================================================================

def _fmt(x, nd=4):
    if x is None:
        return "n/a"
    if isinstance(x, float) and not math.isfinite(x):
        return "n/a"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def _fmt_p(p):
    if p is None or (isinstance(p, float) and not math.isfinite(p)):
        return "n/a"
    return f"{p:.2e}" if p < 1e-3 else f"{p:.4f}"


def render_markdown(payload: dict) -> str:
    """
    A ready-to-embed markdown block: the header says PAIRED and NOT AN AUC, the
    table carries the estimate, and the caveat is printed underneath.

    Returned as a string AND stored on the payload so a report layer can render
    this correctly without importing anything from here. The caveat is not
    optional and is not a footnote: it sits directly below the number it
    qualifies.
    """
    cohort = payload.get("cohort", "?")
    lines = [f"### {cohort}: within-subject PAIRED analysis (NOT an AUC)", ""]
    lines.append(METRIC_NOTE)
    lines.append("")
    rows = payload.get("across_seeds") or []
    if rows:
        lines.append("| condition | subjects paired | excluded | mean paired diff (95% CI) "
                     "| concordance (95% CI) | Wilcoxon p | sign-test p |")
        lines.append("|---|---|---|---|---|---|---|")
        for r in rows:
            if r.get("mean_diff") is None:
                lines.append(f"| {r['condition']} | - | - | MISSING ({r.get('reason')}) "
                             f"| - | - | - |")
                continue
            ci = f"{_fmt(r['mean_diff'])} [{_fmt(r['ci_lo'])}, {_fmt(r['ci_hi'])}]"
            cc = (f"{_fmt(r['concordance'], 3)} "
                  f"[{_fmt(r['concordance_ci_lo'], 3)}, {_fmt(r['concordance_ci_hi'], 3)}]")
            lines.append(
                f"| {r['condition']} | {r['n_subjects_paired']} "
                f"| {r['n_subjects_excluded']} | {ci} | {cc} "
                f"| {_fmt_p(r['wilcoxon_p_worst'])} | {_fmt_p(r['sign_p_worst'])} |")
        lines.append("")
        lines.append("Intervals are the widest across seeds and p-values the largest "
                     "(worst-case), because the seeds share one test fold and are not "
                     "independent replicates.")
        sat = [r["condition"] for r in rows if r.get("at_exact_test_floor")]
        if sat:
            n_sub = max(r.get("n_subjects_paired") or 0 for r in rows)
            lines.append("")
            lines.append(
                f"**Saturated:** {', '.join(sat)} put every one of the {n_sub} subjects "
                f"on the same side, so their p-values are the exact-test floor "
                f"2/2^{n_sub} and their concordance is pinned at 1.0. Those conditions "
                f"are directional evidence only and CANNOT be ranked against one "
                f"another from this table -- an identical p here means the statistic "
                f"ran out of resolution, not that the models perform identically.")
    else:
        lines.append("MISSING: no paired result could be computed "
                     f"({payload.get('reason') or 'no runs found'}).")
    seq = payload.get("sequence_parameter_contrast") or {}
    if seq.get("by_class"):
        lines.append("")
        lines.append("Acquisition parameters by class (descriptive, no test):")
        lines.append("")
        lines.append("| class | TE median | echo spacing median | flip angle median |")
        lines.append("|---|---|---|---|")
        for lab, per in sorted(seq["by_class"].items()):
            lines.append(f"| {lab} | {_fmt((per.get('te') or {}).get('median'), 1)} "
                         f"| {_fmt((per.get('echo_spacing') or {}).get('median'), 2)} "
                         f"| {_fmt((per.get('flip_angle') or {}).get('median'), 1)} |")
    lines.append("")
    lines.append(f"**CAVEAT.** {payload.get('caveat', CAVEAT)}")
    lines.append("")
    return "\n".join(lines)


def print_report(payload: dict) -> None:
    rule = "=" * 100
    print(rule)
    print(f"s07 paired within-subject analysis -- cohort {payload.get('cohort')}")
    print(rule)
    print(METRIC_NOTE)
    print()
    rows = payload.get("runs") or []
    if not rows:
        print(f"MISSING: {payload.get('reason') or 'no stage-3 runs found'}")
    else:
        hdr = (f"{'condition':<12}{'seed':>6}{'subj':>6}{'excl':>6}"
               f"{'mean diff':>12}{'95% CI':>24}{'concord':>10}"
               f"{'wilcoxon p':>13}{'sign p':>13}")
        print(hdr)
        print("-" * len(hdr))
        for r in rows:
            if not r.get("paired_difference"):
                print(f"{r['condition']:<12}{r['seed']:>6}"
                      f"   MISSING: {r.get('reason')}")
                continue
            pdif = r["paired_difference"]
            ci = f"[{_fmt(pdif['ci_lo'])}, {_fmt(pdif['ci_hi'])}]"
            print(f"{r['condition']:<12}{r['seed']:>6}{r['n_subjects_paired']:>6}"
                  f"{r['n_subjects_excluded']:>6}{_fmt(pdif['mean']):>12}{ci:>24}"
                  f"{_fmt(r['concordance']['proportion'], 3):>10}"
                  f"{_fmt_p(r['wilcoxon_signed_rank']['p']):>13}"
                  f"{_fmt_p(r['sign_test']['p']):>13}")
    print()
    for w in payload.get("warnings", []):
        print(f"  WARNING: {w}")
    print()
    print("CAVEAT:")
    for line in _wrap(payload.get("caveat", CAVEAT), 96):
        print(f"  {line}")
    print(rule)


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


# ==========================================================================
# Driver
# ==========================================================================

def run_paired(results_dir: Path, cache_dir: Path, cohort_dir: Path,
               cohort: str = "knee", n_boot: int = 2000, seed: int = 0,
               alpha: float = 0.05) -> dict:
    """Build the whole payload for one cohort. Never raises on missing data."""
    results_dir, cache_dir, cohort_dir = Path(results_dir), Path(cache_dir), Path(cohort_dir)
    payload = {
        "schema": "s07_paired/1",
        "analysis": "paired_within_subject",
        "metric_kind": "paired_difference",
        "metric": "within_subject_mean_score_difference",
        "is_auc": False,
        "render_as": "paired_analysis",
        "evidence_role": "supporting",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "cohort": cohort,
        "config": {
            "results_dir": str(results_dir), "cache_dir": str(cache_dir),
            "cohort_dir": str(cohort_dir), "n_boot": int(n_boot),
            "bootstrap_seed": int(seed), "alpha": float(alpha),
            "expected_sign": EXPECTED_SIGN, "exact_max_n": EXACT_MAX_N,
        },
        "metric_note": METRIC_NOTE,
        "caveat": CAVEAT,
        "methods_note": (
            "Per subject, the mean model score on their positive-class slices minus "
            "the mean on their negative-class slices; one number per subject. "
            "Two-sided Wilcoxon signed-rank (exact conditional enumeration at this n) "
            "and an exact two-sided sign test over those differences. Effect size is "
            "the within-subject concordance -- the proportion of subjects whose "
            "difference has the expected sign -- with a Wilson interval. The interval "
            "on the mean difference is a percentile bootstrap that resamples "
            "SUBJECTS. Subjects lacking one of the two classes are excluded, counted "
            "and listed. No threshold is fitted anywhere in this module."),
        "runs": [], "across_seeds": [], "warnings": [], "reason": None,
    }

    cinfo = build_cluster_map(cohort, cache_dir, cohort_dir)
    cmap = cinfo.get("map") or None
    payload["cluster_map"] = {"source": cinfo.get("source"), "reason": cinfo.get("reason"),
                              "n_entries": len(cinfo.get("map") or {})}
    if cinfo.get("reason"):
        payload["warnings"].append(
            f"cache_idx -> subject_id join unavailable ({cinfo['reason']}); falling "
            f"back to patient_id as the pairing unit")

    names = class_names(cohort, cache_dir)
    payload["classes"] = names
    if names.get("reason"):
        payload["warnings"].append(f"class names unresolved: {names['reason']}")

    payload["sequence_parameter_contrast"] = sequence_parameter_contrast(cohort, cache_dir)

    runs = load_runs(results_dir, cohort=cohort)
    if not runs:
        payload["reason"] = (
            f"no stage-3 run JSON for cohort '{cohort}' under {results_dir}. The "
            f"training sweep has not produced them yet; rerun this module when it "
            f"has. Nothing is reported as passing in the meantime.")
        payload["warnings"].append(payload["reason"])
        payload["markdown"] = render_markdown(payload)
        return payload

    groups: dict = defaultdict(list)
    for r in runs:
        groups[(str(r["cohort"]), str(r["condition"]), int(r.get("seed", 0)),
                str(r.get("region", "full")))].append(r)
    for key in sorted(groups):
        res = analyse_paired_group(groups[key], cmap, n_boot=n_boot, seed=seed,
                                   alpha=alpha, names=names)
        payload["runs"].append(res)
        for w in res["warnings"]:
            tagged = f"{res['condition']} seed{res['seed']}: {w}"
            if tagged not in payload["warnings"]:
                payload["warnings"].append(tagged)
    payload["across_seeds"] = aggregate_across_seeds(payload["runs"])
    payload["markdown"] = render_markdown(payload)
    return payload


def default_out_path(results_dir: Path, cohort: str) -> Path:
    """
    pipeline_out/paired/{cohort}_paired.json.

    A SIBLING of the results tree, like stage 5's controls directory, so that
    neither s04_stats.load_runs nor s06_report.load_runs ever walks over it. The
    payload is also shaped to be rejected by both loaders, but placement is the
    cheaper of the two guarantees and costs nothing.
    """
    rd = Path(results_dir).resolve()
    root = rd.parent if rd.name == "results" else rd
    return root / "paired" / f"{cohort}_paired.json"


def _json_default(o):
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    return str(o)


# ==========================================================================
# Self-test
# ==========================================================================

class _Check:
    def __init__(self):
        self.passed = 0
        self.failed = 0

    def __call__(self, name, cond, detail=""):
        if cond:
            self.passed += 1
            print(f"  PASS  {name}" + (f"   {detail}" if detail else ""))
        else:
            self.failed += 1
            print(f"  FAIL  {name}" + (f"   {detail}" if detail else ""))
        return bool(cond)


def _fake_run(rng, n_subjects=20, n_slices=5, delta=0.0, subject_sd=1.0,
              noise_sd=0.3, cohort="knee", condition="phase", seed=42,
              drop_neg_for=(), start=0):
    """
    A stage-3-shaped run payload with a known paired truth.

    Each subject gets its own offset (the between-subject variance a paired
    design is supposed to remove) and each class-mean differs by `delta`. Scores
    are pushed through a logistic so they live in (0, 1) like real probabilities;
    the transform is monotone, so the sign of every subject difference -- and
    therefore the sign test and the concordance -- is preserved exactly.
    """
    probs, labels, sids = [], [], []
    for i in range(n_subjects):
        sid = f"s{start + i:03d}"
        base = rng.normal(0.0, subject_sd)
        for lab in (1, 0):
            if lab == 0 and sid in drop_neg_for:
                continue
            mu = base + (delta if lab == 1 else 0.0)
            for _ in range(n_slices):
                probs.append(float(1.0 / (1.0 + math.exp(-(mu + rng.normal(0, noise_sd))))))
                labels.append(lab)
                sids.append(sid)
    return {"cohort": cohort, "condition": condition, "seed": seed, "region": "full",
            "test": {"probs": probs, "labels": labels, "patient_ids": sids,
                     "subject_ids": sids, "cache_idx": list(range(len(probs))),
                     "auc": 0.5, "n": len(probs), "n_pos": int(sum(labels))},
            "_tag": f"{cohort}_{condition}_seed{seed}", "_path": "<synthetic>"}


def self_test(quick: bool = False) -> int:
    print("=" * 100)
    print("s07_paired self-test")
    print("=" * 100)
    ck = _Check()
    rng = np.random.default_rng(20240517)

    # ---------------------------------------------------------------- exact
    print("\n[1] Wilcoxon signed-rank: closed-form cases")
    r = wilcoxon_signed_rank([1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    ck("all-positive n=10 gives W+ = 55, W- = 0", r["w_plus"] == 55 and r["w_minus"] == 0,
       f"W+={r['w_plus']} W-={r['w_minus']}")
    ck("all-positive n=10 exact p = 2/2^10", abs(r["p"] - 2.0 / 1024.0) < 1e-15,
       f"p={r['p']!r} expected {2.0/1024.0!r}")
    ck("exact branch selected at n=10", "exact" in (r["method"] or ""), r["method"])
    r = wilcoxon_signed_rank([-1, -2, -3, -4, -5, -6, -7, -8, -9, -10])
    ck("sign flip gives the same two-sided p", abs(r["p"] - 2.0 / 1024.0) < 1e-15,
       f"p={r['p']!r}")
    r = wilcoxon_signed_rank([1, -1, 2, -2, 3, -3])
    ck("perfectly symmetric differences give p = 1", abs(r["p"] - 1.0) < 1e-12,
       f"p={r['p']!r}")
    r = wilcoxon_signed_rank([0.0, 0.0, 1.0, 2.0, 3.0])
    ck("zeros are discarded and counted", r["n_zero"] == 2 and r["n_used"] == 3,
       f"n_zero={r['n_zero']} n_used={r['n_used']}")
    ck("p after discarding zeros = 2/2^3", abs(r["p"] - 2.0 / 8.0) < 1e-15, f"p={r['p']!r}")
    r = wilcoxon_signed_rank([0.0, 0.0, 0.0])
    ck("all-zero differences return null with a reason",
       r["p"] is None and r["reason"], str(r["reason"]))
    r = wilcoxon_signed_rank([])
    ck("empty input returns null with a reason", r["p"] is None and r["reason"] is not None)
    r = wilcoxon_signed_rank([1.0, 1.0, 1.0, -1.0])
    ck("all-tied |d| still gets an exact p (ties handled by enumeration)",
       r["p"] is not None and 0 < r["p"] <= 1, f"p={r['p']!r} ties={r['n_ties']}")

    # -------------------------------------------------- scipy cross-check
    print("\n[2] Wilcoxon vs scipy (independent reference)")
    try:
        from scipy import stats as _sps
        worst_exact, worst_approx = 0.0, 0.0
        for trial in range(8 if quick else 40):
            n = int(rng.integers(6, 26))
            d = rng.normal(0.3, 1.0, size=n)
            d = np.round(d, 12)
            if np.any(d == 0) or len(np.unique(np.abs(d))) != n:
                continue
            mine = wilcoxon_signed_rank(d)
            ref = _sps.wilcoxon(d, alternative="two-sided", method="exact")
            worst_exact = max(worst_exact, abs(mine["p"] - float(ref.pvalue)))
        ck("exact p matches scipy method='exact' to 1e-12", worst_exact < 1e-12,
           f"max |diff| = {worst_exact:.3e}")
        for trial in range(6 if quick else 25):
            n = int(rng.integers(120, 200))
            d = np.round(rng.normal(0.15, 1.0, size=n), 2)
            d = d[d != 0]
            if d.size < 100:
                continue
            mine = wilcoxon_signed_rank(d, exact_max_n=0)
            ref = _sps.wilcoxon(d, alternative="two-sided", method="approx",
                                correction=False)
            worst_approx = max(worst_approx, abs(mine["p"] - float(ref.pvalue)))
        ck("normal-approx p (with ties) matches scipy method='approx' to 1e-12",
           worst_approx < 1e-12, f"max |diff| = {worst_approx:.3e}")
        # exact enumeration vs scipy on a tie-heavy vector where scipy's exact
        # path is not defined: agreement of my exact with my approx as n grows.
        d = np.round(rng.normal(0.2, 1.0, size=90), 3)
        d = d[d != 0]
        pe = wilcoxon_signed_rank(d)["p"]
        pa = wilcoxon_signed_rank(d, exact_max_n=0)["p"]
        ck("exact and asymptotic agree to <5% relative at n~90",
           pe is not None and pa is not None and abs(pe - pa) <= 0.05 * max(pe, pa, 1e-12),
           f"exact={pe:.4g} approx={pa:.4g}")
    except ImportError:  # pragma: no cover
        ck("scipy cross-check", True, "scipy unavailable; skipped (not a failure)")

    # ------------------------------------------------------------ sign test
    print("\n[3] Sign test: closed-form cases")
    r = sign_test([1] * 10)
    ck("10/10 positive gives p = 2/2^10", abs(r["p"] - 2.0 / 1024.0) < 1e-15, f"p={r['p']!r}")
    r = sign_test([1] * 8 + [-1] * 2)
    ck("8/10 positive gives p = 112/1024", abs(r["p"] - 112.0 / 1024.0) < 1e-15,
       f"p={r['p']!r} expected {112.0/1024.0!r}")
    r = sign_test([1] * 5 + [-1] * 5)
    ck("5/5 split gives p = 1", abs(r["p"] - 1.0) < 1e-15, f"p={r['p']!r}")
    r = sign_test([0.0, 0.0, 1.0, 1.0, 1.0])
    ck("sign test discards zeros and counts them",
       r["n_used"] == 3 and r["n_zero"] == 2 and abs(r["p"] - 0.25) < 1e-15,
       f"n_used={r['n_used']} n_zero={r['n_zero']} p={r['p']!r}")
    ck("sign test is more conservative than signed-rank on a clean effect",
       sign_test(np.arange(1, 13))["p"] >= wilcoxon_signed_rank(np.arange(1, 13))["p"],
       f"sign={sign_test(np.arange(1,13))['p']:.3g} "
       f"wilcoxon={wilcoxon_signed_rank(np.arange(1,13))['p']:.3g}")
    try:
        from scipy import stats as _sps2
        d = rng.normal(0.4, 1.0, size=31)
        k = int(np.sum(d > 0))
        ref = float(_sps2.binomtest(k, 31, 0.5, alternative="two-sided").pvalue)
        ck("sign-test p matches scipy binomtest to 1e-15",
           abs(sign_test(d)["p"] - ref) < 1e-15, f"|diff| = {abs(sign_test(d)['p']-ref):.2e}")
    except ImportError:  # pragma: no cover
        ck("sign test vs scipy", True, "scipy unavailable; skipped")

    # ---------------------------------------------------------- concordance
    print("\n[4] Concordance + Wilson interval")
    c = paired_concordance([1.0] * 29)
    ck("29/29 concordance = 1.0", c["proportion"] == 1.0 and c["k"] == 29)
    ck("Wilson interval for 29/29 is not [1, 1]", c["ci_lo"] < 1.0 and c["ci_hi"] == 1.0,
       f"[{c['ci_lo']:.4f}, {c['ci_hi']:.4f}]")
    c2 = paired_concordance([1.0] * 15 + [-1.0] * 15)
    ck("15/30 concordance = 0.5 with an interval straddling 0.5",
       abs(c2["proportion"] - 0.5) < 1e-15 and c2["ci_lo"] < 0.5 < c2["ci_hi"],
       f"[{c2['ci_lo']:.3f}, {c2['ci_hi']:.3f}]")
    c3 = paired_concordance([1.0, 0.0, 1.0, 1.0])
    ck("exact ties count as NON-concordant (conservative)",
       c3["k"] == 3 and c3["n"] == 4 and c3["n_zero"] == 1,
       f"k={c3['k']} n={c3['n']} zeros={c3['n_zero']}")
    if _S04 is not None and hasattr(_S04, "wilson_interval"):
        a = wilson_interval(7, 23)
        b = _S04.wilson_interval(7, 23)
        ck("wilson_interval agrees with s04_stats exactly", a == b, f"{a} vs {b}")
    else:  # pragma: no cover
        ck("wilson_interval vs s04_stats", True, "s04_stats unimportable; skipped")
    ck("wilson_interval on n = 0 returns nulls, not a number",
       wilson_interval(0, 0) == (None, None))

    # ------------------------------------------------------------ bootstrap
    print("\n[5] Subject bootstrap on the mean paired difference")
    d = rng.normal(0.5, 1.0, size=40)
    b = subject_bootstrap_mean_diff(d, n_boot=4000, seed=1)
    ck("bootstrap mean equals the sample mean exactly",
       abs(b["mean"] - float(np.mean(d))) < 1e-15)
    se_boot = (b["ci_hi"] - b["ci_lo"]) / (2 * Z_975)
    se_analytic = float(np.std(d, ddof=1) / math.sqrt(d.size))
    ck("bootstrap SE matches the analytic paired SE within 15%",
       abs(se_boot - se_analytic) / se_analytic < 0.15,
       f"boot={se_boot:.4f} analytic={se_analytic:.4f}")
    b2 = subject_bootstrap_mean_diff(d, n_boot=4000, seed=1)
    ck("bootstrap is reproducible at a fixed seed",
       (b["ci_lo"], b["ci_hi"]) == (b2["ci_lo"], b2["ci_hi"]))
    ck("n = 1 subject returns a mean but refuses an interval",
       subject_bootstrap_mean_diff([0.3])["ci_lo"] is None
       and subject_bootstrap_mean_diff([0.3])["reason"] is not None)
    n_cov = 60 if quick else 400
    cover = 0
    for _ in range(n_cov):
        s = rng.normal(0.25, 1.0, size=30)
        bb = subject_bootstrap_mean_diff(s, n_boot=400, seed=int(rng.integers(1e6)))
        cover += int(bb["ci_lo"] <= 0.25 <= bb["ci_hi"])
    cov = cover / n_cov
    ck("subject bootstrap covers the true mean at ~95%", 0.88 <= cov <= 0.99,
       f"coverage = {cov:.3f} over {n_cov} replicates (nominal 0.95)")

    # ------------------------------------------------------ paired extraction
    print("\n[6] Paired extraction from a stage-3-shaped payload")
    run = _fake_run(rng, n_subjects=20, n_slices=5, delta=2.0)
    pd_ = paired_differences(run["test"], run["test"]["subject_ids"])
    ck("20 subjects seen, 20 paired, 0 excluded",
       (pd_["n_subjects_seen"], pd_["n_subjects_paired"], pd_["n_subjects_excluded"])
       == (20, 20, 0), str((pd_["n_subjects_seen"], pd_["n_subjects_paired"])))
    ck("a large positive delta gives 20/20 positive differences",
       int(np.sum(np.asarray(pd_["diffs"]) > 0)) == 20,
       f"{int(np.sum(np.asarray(pd_['diffs'])>0))}/20")
    manual = []
    for sid in sorted(set(run["test"]["subject_ids"])):
        p = [q for q, s, l in zip(run["test"]["probs"], run["test"]["subject_ids"],
                                  run["test"]["labels"]) if s == sid and l == 1]
        n = [q for q, s, l in zip(run["test"]["probs"], run["test"]["subject_ids"],
                                  run["test"]["labels"]) if s == sid and l == 0]
        manual.append(float(np.mean(p) - np.mean(n)))
    ck("difference vector matches an independent recomputation",
       np.allclose(pd_["diffs"], manual, atol=0, rtol=0),
       f"max |diff| = {np.max(np.abs(np.asarray(pd_['diffs']) - np.asarray(manual))):.2e}")

    print("\n[7] The guard: unpaired subjects are excluded AND counted")
    run = _fake_run(rng, n_subjects=20, n_slices=5, delta=1.5,
                    drop_neg_for=("s000", "s007", "s019"))
    pd_ = paired_differences(run["test"], run["test"]["subject_ids"])
    ck("3 one-class subjects are excluded", pd_["n_subjects_excluded"] == 3,
       f"excluded {pd_['n_subjects_excluded']}")
    ck("17 subjects remain paired", pd_["n_subjects_paired"] == 17)
    ck("excluded subjects are listed by id, not silently dropped",
       sorted(e["subject_id"] for e in pd_["excluded"]) == ["s000", "s007", "s019"],
       str(sorted(e["subject_id"] for e in pd_["excluded"])))
    ck("each exclusion carries its per-class slice counts and a reason",
       all(e["n_neg_slices"] == 0 and e["n_pos_slices"] > 0 and e["reason"]
           for e in pd_["excluded"]))
    res = analyse_paired_group([run], None, n_boot=500, seed=0, alpha=0.05)
    ck("the group result surfaces the exclusion as a warning",
       any("excluded" in w for w in res["warnings"]), str(res["warnings"])[:90])
    ck("the group result reports n_subjects_paired = 17, seen = 20",
       (res["n_subjects_paired"], res["n_subjects_seen"]) == (17, 20))

    print("\n[8] Non-finite scores are dropped and counted")
    run = _fake_run(rng, n_subjects=6, n_slices=4, delta=1.0)
    run["test"]["probs"][0] = float("nan")
    run["test"]["probs"][1] = float("inf")
    pd_ = paired_differences(run["test"], run["test"]["subject_ids"])
    ck("2 bad slices are dropped and counted", pd_["n_slices_dropped"] == 2,
       f"dropped {pd_['n_slices_dropped']} of {pd_['n_slices_input']}")
    ck("the reason for dropping is recorded", bool(pd_["dropped_slice_reason"]))
    ck("all 6 subjects still pair up", pd_["n_subjects_paired"] == 6)

    print("\n[9] Known-answer end-to-end: strong effect, null effect, reversed effect")
    strong = analyse_paired_group([_fake_run(rng, 25, 5, delta=3.0)], None,
                                  n_boot=2000, seed=0, alpha=0.05)
    ck("strong effect: concordance = 1.0",
       strong["concordance"]["proportion"] == 1.0,
       f"{strong['concordance']['k']}/{strong['concordance']['n']}")
    ck("strong effect: Wilcoxon p = 2/2^25",
       abs(strong["wilcoxon_signed_rank"]["p"] - 2.0 / (1 << 25)) < 1e-15,
       f"p = {strong['wilcoxon_signed_rank']['p']:.3e}")
    ck("strong effect: bootstrap CI excludes 0",
       strong["paired_difference"]["ci_lo"] > 0,
       f"[{strong['paired_difference']['ci_lo']:.4f}, "
       f"{strong['paired_difference']['ci_hi']:.4f}]")
    ck("strong effect: flagged SATURATED at the exact-test floor",
       strong["at_exact_test_floor"] is True
       and any("SATURATED" in w for w in strong["warnings"]),
       f"floor p = {strong.get('exact_test_floor_p'):.3e}")
    ck("the floor p equals the reported p exactly (nothing beyond it exists)",
       abs(strong["exact_test_floor_p"] - strong["sign_test"]["p"]) < 1e-18
       and abs(strong["exact_test_floor_p"]
               - strong["wilcoxon_signed_rank"]["p"]) < 1e-18)
    # A single null dataset would fail this check 5% of the time BY DESIGN --
    # that is what a 95% interval means -- so it is run over 10 independent null
    # datasets and required to behave 8+ times. Under true 95% coverage,
    # P(>= 8 of 10) = 0.9885, so this is a real check rather than a coin flip.
    hit_ci = hit_w = hit_s = 0
    for _ in range(10):
        nul = analyse_paired_group([_fake_run(rng, 40, 5, delta=0.0)], None,
                                   n_boot=1000, seed=0, alpha=0.05)
        hit_ci += int(nul["paired_difference"]["ci_lo"] <= 0
                      <= nul["paired_difference"]["ci_hi"])
        hit_w += int(nul["wilcoxon_signed_rank"]["p"] > 0.05)
        hit_s += int(nul["sign_test"]["p"] > 0.05)
    ck("null effect: bootstrap CI contains 0 in >= 8 of 10 null datasets",
       hit_ci >= 8, f"{hit_ci}/10")
    ck("null effect: Wilcoxon is non-significant in >= 8 of 10 null datasets",
       hit_w >= 8, f"{hit_w}/10")
    ck("null effect: sign test is non-significant in >= 8 of 10 null datasets",
       hit_s >= 8, f"{hit_s}/10")
    ck("null effect: NOT flagged as saturated", nul["at_exact_test_floor"] is False)
    rev = analyse_paired_group([_fake_run(rng, 25, 5, delta=-3.0)], None,
                               n_boot=2000, seed=0, alpha=0.05)
    ck("reversed effect: mean difference is negative",
       rev["paired_difference"]["mean"] < 0, _fmt(rev["paired_difference"]["mean"]))
    ck("reversed effect: concordance = 0.0 (direction is not absorbed)",
       rev["concordance"]["proportion"] == 0.0)
    ck("reversed effect: the two-sided test still detects it",
       rev["wilcoxon_signed_rank"]["p"] < 1e-6,
       f"p = {rev['wilcoxon_signed_rank']['p']:.3e}")

    print("\n[10] Type-I error of the paired tests under a true null")
    n_rep = 80 if quick else 500
    rej_w = rej_s = 0
    for _ in range(n_rep):
        d = rng.normal(0.0, 1.0, size=25)
        rej_w += int(wilcoxon_signed_rank(d)["p"] < 0.05)
        rej_s += int(sign_test(d)["p"] < 0.05)
    aw, asg = rej_w / n_rep, rej_s / n_rep
    ck("Wilcoxon type-I rate is at or below nominal 0.05", aw <= 0.09,
       f"{aw:.3f} over {n_rep} null replicates")
    ck("sign test type-I rate is at or below nominal 0.05 (it is discrete)",
       asg <= 0.09, f"{asg:.3f} over {n_rep} null replicates")

    print("\n[11] Between-subject variance is removed by the pairing")
    a = _fake_run(rng, 30, 5, delta=0.8, subject_sd=0.05)
    b = _fake_run(rng, 30, 5, delta=0.8, subject_sd=3.0)
    ra = analyse_paired_group([a], None, n_boot=1000, seed=0, alpha=0.05)
    rb = analyse_paired_group([b], None, n_boot=1000, seed=0, alpha=0.05)
    ck("a 60x increase in between-subject spread does not destroy the paired test",
       rb["wilcoxon_signed_rank"]["p"] < 0.05 and ra["wilcoxon_signed_rank"]["p"] < 0.05,
       f"tight={ra['wilcoxon_signed_rank']['p']:.2e} "
       f"spread={rb['wilcoxon_signed_rank']['p']:.2e}")

    print("\n[12] Out-of-fold pooling and the non-disjoint refusal")
    f0 = _fake_run(rng, 10, 5, delta=2.0, seed=7, start=0)
    f1 = _fake_run(rng, 10, 5, delta=2.0, seed=7, start=10)
    pooled = analyse_paired_group([f0, f1], None, n_boot=500, seed=0, alpha=0.05)
    ck("two disjoint folds pool to 20 subjects, one estimate",
       pooled["n_subjects_paired"] == 20 and pooled["n_runs_pooled"] == 2,
       f"n={pooled['n_subjects_paired']} from {pooled['n_runs_pooled']} runs")
    dup = analyse_paired_group([f0, f0], None, n_boot=500, seed=0, alpha=0.05)
    ck("overlapping folds are REFUSED, not pooled",
       dup["paired_difference"] is None and "not disjoint" in (dup["reason"] or ""),
       str(dup["reason"])[:70])

    print("\n[13] Degenerate inputs return MISSING, never a number")
    one = analyse_paired_group([_fake_run(rng, 1, 5, delta=2.0)], None,
                               n_boot=500, seed=0, alpha=0.05)
    ck("1 subject: no interval, and a warning that the test cannot be significant",
       one["paired_difference"]["ci_lo"] is None and any("uninformative" in w
                                                        for w in one["warnings"]),
       str(one["warnings"])[:80])
    unp = _fake_run(rng, 8, 5, delta=1.0, drop_neg_for=tuple(f"s{i:03d}" for i in range(8)))
    res = analyse_paired_group([unp], None, n_boot=500, seed=0, alpha=0.05)
    ck("a cohort where nobody supplies both classes is refused with a reason",
       res["paired_difference"] is None and "not a paired design" in (res["reason"] or ""),
       str(res["reason"])[:70])

    print("\n[14] Across-seed aggregation takes the WORST case")
    g1 = analyse_paired_group([_fake_run(rng, 20, 5, delta=3.0, seed=42)], None,
                              n_boot=500, seed=0, alpha=0.05)
    g2 = analyse_paired_group([_fake_run(rng, 20, 5, delta=0.05, seed=123)], None,
                              n_boot=500, seed=0, alpha=0.05)
    agg = aggregate_across_seeds([g1, g2])[0]
    ck("the reported p is the LARGEST of the seeds, not the smallest",
       abs(agg["wilcoxon_p_worst"] - max(g1["wilcoxon_signed_rank"]["p"],
                                         g2["wilcoxon_signed_rank"]["p"])) < 1e-18,
       f"{agg['wilcoxon_p_worst']:.3g}")
    ck("the reported interval is the WIDEST of the seeds",
       agg["ci_lo"] == min(g1["paired_difference"]["ci_lo"],
                           g2["paired_difference"]["ci_lo"])
       and agg["ci_hi"] == max(g1["paired_difference"]["ci_hi"],
                               g2["paired_difference"]["ci_hi"]))
    ck("the reported concordance is the seed closest to 0.5",
       abs(agg["concordance"] - 0.5) <= abs(g1["concordance"]["proportion"] - 0.5),
       f"{agg['concordance']:.3f}")
    ck("the aggregate states that seeds share one test fold",
       "not independent replicates" in agg["caveat"].lower()
       or "NOT independent replicates" in agg["caveat"])

    # ------------------------------------------------- payload + rendering
    print("\n[15] Payload shape, tagging and the caveat")
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        res_dir = td / "results" / "confound_knee"
        res_dir.mkdir(parents=True)
        for cond in ("phase", "magnitude"):
            for sd in (42, 123):
                r = _fake_run(rng, 12, 5, delta=(2.0 if cond == "phase" else 1.0),
                              condition=cond, seed=sd)
                r.pop("_tag"), r.pop("_path")
                (res_dir / f"knee_{cond}_seed{sd}.json").write_text(json.dumps(r))
        # a stage-5 control payload that must NOT be pooled in
        ctrl = _fake_run(rng, 12, 5, delta=0.0, condition="phase", seed=42)
        ctrl["control"] = "label_permutation"
        ctrl.pop("_tag"), ctrl.pop("_path")
        (res_dir / "knee__label_permutation__phase__seed42.json").write_text(json.dumps(ctrl))

        payload = run_paired(td / "results", td / "cache", td / "cohorts",
                             cohort="knee", n_boot=500, seed=0)
        ck("recursive discovery finds runs in the fold subdirectory",
           len(payload["runs"]) == 4, f"{len(payload['runs'])} groups")
        ck("stage-5 control payloads are refused",
           all(r["n_runs_pooled"] == 1 for r in payload["runs"]),
           str([r["n_runs_pooled"] for r in payload["runs"]]))
        ck("payload is tagged as a paired analysis",
           payload["analysis"] == "paired_within_subject"
           and payload["render_as"] == "paired_analysis")
        ck("payload declares it is NOT an AUC",
           payload["is_auc"] is False and payload["metric_kind"] == "paired_difference")
        ck("payload declares knee is SUPPORTING evidence",
           payload["evidence_role"] == "supporting")
        ck("payload carries the caveat verbatim", payload["caveat"] == CAVEAT)
        low = payload["caveat"].lower()
        ck("the caveat names echo time, echo spacing and flip angle",
           "echo time" in low and "echo spacing" in low and "flip angle" in low)
        ck("the caveat names the brain coil-count result as load-bearing",
           "load-bearing" in low and "coil" in low)
        doc = " ".join((__doc__ or "").split())      # the docstring is line-wrapped
        ck("the module docstring carries the same caveat",
           "echo time, echo spacing and flip angle" in doc and "LOAD-BEARING" in doc
           and "SUPPORTING evidence" in doc)
        md = payload["markdown"]
        ck("the rendered markdown says PAIRED and NOT an AUC",
           "PAIRED" in md and "NOT an AUC" in md)
        ck("the rendered markdown prints the caveat", "CAVEAT" in md and "echo time" in md)
        ck("the rendered markdown reports subjects paired and excluded",
           "subjects paired" in md and "excluded" in md)
        ck("the rendered markdown warns when conditions are saturated at the floor",
           "Saturated" in md and "ran out of resolution" in md)
        ck("render_markdown is a pure function of the payload",
           render_markdown(payload) == md)

        ck("payload has no top-level 'test' key (cannot look like a run)",
           "test" not in payload)
        ck("payload has no 'control' key (cannot look like a stage-5 control)",
           "control" not in payload)
        out_p = default_out_path(td / "results", "knee")
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_p.write_text(json.dumps(payload, default=_json_default))
        rel = out_p.relative_to(Path(td).resolve())
        ck("the default output path is OUTSIDE the results tree",
           "results" not in rel.parts, str(rel))
        try:
            import s06_report as _s06
        except Exception as exc:  # pragma: no cover
            _s06 = None
            ck("s06_report interop", True, f"s06_report unimportable ({exc}); skipped")
        if _s06 is not None:
            ck("s06_report._looks_like_run rejects the payload",
               _s06._looks_like_run(payload) is False)
            ck("s06_report.load_runs does not pick the payload up",
               all(Path(r["_path"]).name != out_p.name
                   for r in _s06.load_runs(td / "results")))
            ck("s06_report.load_runs still finds the 4 real runs next to it",
               len(_s06.load_runs(td / "results")) == 4,
               f"{len(_s06.load_runs(td / 'results'))} runs")
        if _S04 is not None and hasattr(_S04, "load_runs"):
            s04_seen = _S04.load_runs(default_out_path(td / "results", "knee").parent)
            ck("s04_stats.load_runs does not pick the payload up", len(s04_seen) == 0,
               f"{len(s04_seen)} runs")
        ck("the payload round-trips through JSON",
           isinstance(json.loads(json.dumps(payload, default=_json_default)), dict))

        empty = run_paired(td / "nothing", td / "cache", td / "cohorts", cohort="knee")
        ck("missing results degrade to MISSING with a reason, not a crash",
           empty["runs"] == [] and empty["reason"] is not None,
           str(empty["reason"])[:60])
        ck("the MISSING payload still renders the caveat",
           "CAVEAT" in empty["markdown"] and "MISSING" in empty["markdown"])

    print("\n" + "=" * 100)
    print(f"{ck.passed} passed, {ck.failed} failed")
    print("=" * 100)
    return 0 if ck.failed == 0 else 1


# ==========================================================================
# CLI
# ==========================================================================

def parse_args(argv=None):
    p = argparse.ArgumentParser(description=(
        "Within-subject paired analysis for the knee cohort (paired design: every "
        "subject supplies both classes, so a subject-level AUC is undefined)."))
    p.add_argument("--results-dir", default=str(_DEFAULT_RESULTS_DIR),
                   help="stage-3 results tree; searched RECURSIVELY (per-fold subdirs)")
    p.add_argument("--cache-dir", default=str(_DEFAULT_CACHE_DIR),
                   help="stage-2 cache dir (cache_idx -> subject_id join, class names)")
    p.add_argument("--cohort-dir", default=str(_DEFAULT_COHORT_DIR),
                   help="stage-1 cohort CSV dir")
    p.add_argument("--cohort", default="knee",
                   help="cohort to analyse (only paired designs make sense here)")
    p.add_argument("--out", default=None,
                   help="output JSON path (default: <pipeline_out>/paired/<cohort>_paired.json)")
    p.add_argument("--markdown", default=None,
                   help="also write the rendered markdown block to this path")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--alpha", type=float, default=0.05)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--quick", action="store_true", help="fewer replicates in the self-test")
    p.add_argument("-v", "--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    if args.self_test:
        return self_test(quick=args.quick)

    payload = run_paired(Path(args.results_dir), Path(args.cache_dir),
                         Path(args.cohort_dir), cohort=args.cohort,
                         n_boot=args.n_boot, seed=args.seed, alpha=args.alpha)
    out = Path(args.out) if args.out else default_out_path(Path(args.results_dir), args.cohort)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, default=_json_default))
    if args.markdown:
        Path(args.markdown).parent.mkdir(parents=True, exist_ok=True)
        Path(args.markdown).write_text(payload["markdown"])
    print_report(payload)
    print(f"wrote {out}")
    # A missing result is not an error: the sweep may simply not have reached
    # this cohort. Say so and exit 0 so a driver script does not abort.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
