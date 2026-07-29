"""
s06_report.py
-------------
Stage 6 of the PhaseDx pipeline: figures + RESULTS.md + the verdict.

This stage decides, in public and in writing, whether "MRI phase carries tumour
signal beyond magnitude" is SUPPORTED, NOT SUPPORTED, or INCONCLUSIVE. It is
deliberately the least clever stage in the pipeline: it computes almost nothing
of its own, it copies numbers that stages 3-5 produced, and it applies a fixed
rule set that is printed at the top of the report before any number appears.

Why it is built this way
========================
The failure mode this whole study exists to avoid is a positive-sounding
headline that survives because nobody re-checked the controls. So:

* The verdict function is pure and total. It takes the criteria, returns one of
  three strings, and the string SUPPORTED is unreachable unless every criterion
  is present and passing. There is a runtime assertion to that effect
  (`_assert_verdict_consistent`) that fires before anything is written.
* A missing control is not a passing control. If stage 5 has not run, the
  verdict is INCONCLUSIVE and the missing criterion is named. Nor is an
  UNVERIFIABLE control a passing one: a destroy-the-signal control that arrives
  without a bootstrap interval FAILS its criterion rather than being credited
  with the collapse nobody measured.
* Every number printed carries its sample size inline. prostate_dwi's official
  test fold is 4 patients, 3 of them positive. A bare "AUC 0.85" from 4 people
  is not a result, and this report never prints one.
* "Above chance" is decided on independent units and against a MEASURED null.
  C1 reads the cluster-level interval (one score per subject), not the
  slice-level one, and refuses to be read at all on a fold too small for a
  cluster bootstrap; C8 compares the headline to the label-permutation null
  stage 5 actually computed rather than to a hard-coded 0.500.
* Exactly one cohort per report is confirmatory. `PRIMARY_COHORT` is
  pre-registered on cohort size and reconstruction fidelity; every other cohort
  is labelled exploratory in the verdict table and cannot reach SUPPORTED,
  because three cohorts tested at one alpha is a ~25% family-wise error rate.

What it reads
=============
    pipeline_out/results/{cohort}_{condition}_seed{seed}.json   <- s03_train.py
    pipeline_out/results/statistics.json                        <- s04_stats.py
    pipeline_out/controls/results/*.json                        <- s05_controls.py
    pipeline_out/cache/{cohort}.h5 + {cohort}_index.csv         <- s02
    pipeline_out/cohorts/{cohort}_cohort.csv + s01_summary.json <- s01

Everything degrades. If statistics.json is absent, s06 still draws every figure
it can from the raw stage-3 run JSONs, recomputes AUC intervals itself with a
subject-clustered bootstrap, labels those numbers `s06-fallback` everywhere they
appear, and returns INCONCLUSIVE -- because DeLong and the controls have no
fallback and must not be invented.

The formats below are the ones stages 4 and 5 actually emit; they were read off
real output, not agreed in advance.

statistics.json (s04_stats.py)
------------------------------
    {"generated", "config", "methods_note", "holm", "warnings",
     "runs": [ {"tag","cohort","condition","seed","region",
                "reported_test_auc","cluster_unit","cluster_source",
                "auc_matches_stage3",
                "slice_level":        {"auc","ci_lo","ci_hi","n_slices","n_pos_slices",
                                       "n_clusters","n_pos_clusters","n_boot_used",
                                       "n_skipped_single_class", ...},
                "patient_level_mean": {...same shape...},
                "patient_level_max":  {...},
                "operating_point":    {...}, "warnings": [...] } ],
     "across_seeds": [ {"cohort","condition","region","n_runs","seeds",
                        "slice_auc": {"n","mean","sd","min","max","values"}, ...,
                        "caveat"} ],
     "comparisons": [ {"cohort","seed","region","model_a","model_b","level",
                       "n_clusters","n_cases","n_pos",
                       "delong": {"auc_a","auc_b","diff","p","ci_lo_diff",
                                  "ci_hi_diff","reason","caveat"},
                       "cluster_bootstrap_diff": {"diff","ci_lo","ci_hi","p", ...},
                       "p_raw","p_holm","preferred","holm_family"} ] }

Two things about that format drive the logic here:

* `comparisons` are per SEED and per LEVEL (`slice`, `patient_mean`,
  `patient_max`), and s04 marks `preferred: true` on the patient-level rows
  because its own DeLong `caveat` says the slice-level p-value is
  anti-conservative (slices within a patient are correlated). s06 therefore
  uses the preferred level to decide C2. If only the slice level is evaluable,
  an anti-conservative p can justify a FAIL but never a PASS -- see
  `evaluate_cohort`.
* Intervals exist per seed, and the seeds share one test fold. Averaging them
  would narrow the interval for no statistical reason, so s06 takes the
  ENVELOPE: mean point estimate, widest interval, thinnest bootstrap. Stated on
  every table and figure that uses it.

s05_controls.py
---------------
Stage 5 writes no aggregate file. It writes one JSON per control run into
`pipeline_out/controls/results/`, named
`{cohort}__{control}__{variant}__{condition}__seed{N}.json`, each in exactly
s03_train's schema plus:

    "control"        : one of label_permutation, background_only, phase_scramble,
                       acquisition_split, confound_predictability (or "none" for
                       a re-run headline)
    "control_detail" : {"variant", "label_semantics", "subject_col",
                        "test_auc_ci95": {"auc","lo","hi","n","n_pos",
                                          "n_clusters","n_boot_ok",
                                          "n_boot_degenerate","method"},
                        ...control-specific keys}

`label_semantics` matters: for confound_predictability it is `confound:<target>`,
meaning every label/AUC in that file refers to scanner identity, not cancer.
s06 keeps those numbers strictly separated from diagnostic AUCs.

A control name s06 does not recognise is reported as UNRECOGNISED in RESULTS.md
rather than silently dropped -- and it is never counted as passing.

Usage
=====
    python pipeline/s06_report.py
    python pipeline/s06_report.py --results-dir pipeline_out/results --out pipeline_out/report
    python pipeline/s06_report.py --self-test      # synthetic end-to-end, no data needed
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import re
import shutil
import sys
import tempfile
import textwrap
import time
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
try:
    import common  # noqa: E402
except Exception:  # pragma: no cover - common.py is always present in-repo
    common = None

logger = logging.getLogger("s06_report")

# --------------------------------------------------------------------------
# Verdict thresholds. These are the whole argument of the paper; they live here
# as named constants so a reviewer can find and challenge each one.
# --------------------------------------------------------------------------

ALPHA = 0.05               # significance level for the Holm-adjusted DeLong test
PERM_NULL_TOL = 0.10       # |permutation AUC - 0.5| must be <= this
BACKGROUND_MARGIN = 0.10   # headline AUC must beat background-only by >= this
ACQ_EROSION_MAX = 0.10     # acquisition-stratified AUC may drop at most this much
MIN_CLUSTERS_RELIABLE = 10  # fewer independent test subjects than this => unreliable CI
DEFAULT_BOOTSTRAP = 2000

# -- C4 / C7, the destroy-the-signal controls ------------------------------
# The old rule was "the control's CI lower bound must be <= 0.60", plus a free
# pass (`not est.has_ci`) whenever stage 5 emitted no bootstrap block at all.
# Together those let a control sitting at AUC 0.79 -- a scanner fingerprint,
# which is precisely the alternative explanation this study exists to exclude
# -- be recorded as a PASS. The replacement is the direct statement of what
# "the control collapsed" means: its own interval must contain chance.
#
# Both directions are required, and that is deliberate:
#   lo > 0.5  the destroyed input still predicts the label   -> did not collapse
#   hi < 0.5  the destroyed input predicts the label INVERSELY, which is still
#             label information flowing through an input that was supposed to
#             carry none                                     -> did not collapse
CONTROL_CHANCE = 0.50      # a collapsed control's 95% CI must contain this

# -- The power floor every control-based criterion is held to ---------------
# C1 was given a power floor (>= 10 clusters, >= 5 per class) because a
# percentile bootstrap on a handful of subjects fires far above its nominal
# rate. C3/C4/C5/C6/C7 were given none -- and for them the asymmetry is worse
# than for C1, because their test is "the interval CONTAINS 0.500". C1 asks an
# interval to EXCLUDE something, so noise costs it a pass; C4/C7 ask an interval
# to INCLUDE something, so noise BUYS them a pass. A control scored on two
# subjects, or whose bootstrap returned [0.00, 1.00], or [-inf, +inf], or the
# same number 2000 times, satisfies "contains 0.500" without having measured
# anything at all.
#
# Measured on the synthetic tree (`self_test`, scenarios A2/A2b/A9/A13/DG_*):
# a background-only control sitting at AUC 0.790 -- a scanner fingerprint, the
# alternative explanation this study exists to exclude -- was recorded as a PASS
# on C4 by five separate degenerate intervals.
#
# So: a control estimate that is too noisy to discriminate cannot satisfy a
# criterion. It cannot FAIL one either on the strength of its interval; it is
# MISSING, which is what "we could not measure this" means. The fail paths that
# do not depend on interval width (the control is too close to the headline; the
# control has no interval at all) are untouched and still FAIL.
MIN_CLUSTERS_CONTROL = MIN_CLUSTERS_RELIABLE   # independent subjects behind a control CI
# 0.40 is the width at which the interval stops discriminating between the two
# hypotheses the criterion is deciding between. C4 asks whether a control
# collapsed to 0.500 or stayed up at the headline; the headline must be >= 0.10
# above the control (BACKGROUND_MARGIN), so a control at 0.60 with a symmetric
# +/-0.20 interval covers both 0.500 and 0.790 simultaneously. Any interval
# wider than that cannot separate "collapsed" from "did not collapse", whatever
# its midpoint is.
MAX_CONTROL_CI_WIDTH = 0.40
# Below this the interval is a point: every bootstrap resample returned the same
# AUC (all-tied predictions, or one distinct cluster resampled), so the interval
# is an artefact of the degeneracy rather than a measurement of spread.
MIN_CONTROL_CI_WIDTH = 1e-6
# s05 reports how many resamples produced a defined AUC (`n_boot_ok`). Zero of
# them means there is no bootstrap behind the numbers in the block at all.
MIN_BOOT_VALID_CONTROL = 200
# Runs enveloped together are meant to be repeats of ONE quantity differing only
# by training seed. If their point estimates disagree by more than the margin
# the criterion itself is measured at, they are not repeats and the min-lo /
# max-hi hull over them is not an interval for anything.
MAX_ENVELOPE_SPREAD = BACKGROUND_MARGIN

# -- C6, confound predictability -------------------------------------------
# Was 0.80. Under the standard equal-variance normal mapping AUC = Phi(d/sqrt 2)
# (Rice & Harris 2005), Cohen's small/medium/large d = 0.2/0.5/0.8 correspond to
# AUC 0.56 / 0.64 / 0.71. A confound predictable at 0.79 is therefore a LARGER
# than "large" effect: the input demonstrably encodes scanner identity more
# strongly than most real biological effects, and no downstream diagnostic
# number can be separated from it. 0.80 was above every one of those benchmarks
# and so excluded nothing. The ceiling is set just above Cohen's medium (0.64),
# i.e. a confound may be detectable but must be at most a small-to-medium
# effect.
CONFOUND_AUC_MAX = 0.65
# ... and, independently of the absolute ceiling, the confound must be clearly
# LESS predictable than the diagnosis itself. Same margin the destroy-controls
# are held to (BACKGROUND_MARGIN): if scanner identity is readable from this
# input to within 0.10 AUC of the tumour label, "it is reading the scanner" has
# not been excluded, whatever the absolute numbers are.
CONFOUND_HEADLINE_MARGIN = BACKGROUND_MARGIN

# -- C1, phase above chance ------------------------------------------------
# C1 is decided at the level stage 4 marks `preferred` (one score per subject),
# never at the slice level: slices within a subject are correlated, so the
# slice-level percentile interval undercovers badly. Measured on the real
# prostate_dwi test fold (122 slices / 4 patients / 11 positive slices /
# 3 positive patients): slice AUC 0.918 CI [0.849, 1.000] but patient_mean
# AUC 0.778 CI [0.000, 1.000].
C1_LEVEL = "patient_mean"          # fallback if stage 4 marks nothing `preferred`
#
# Even at the cluster level, a percentile bootstrap on a handful of clusters
# cannot be rescued by more replicates. Simulated null false-positive rate of
# the C1 rule itself (score independent of label; K clusters, P positive;
# 3000 trials x 500 resamples each; nominal one-sided rate 2.5%):
#
#     K   P   min(P, K-P)   P(CI lower bound > 0.5) under a complete null
#     4   3        1                24.9%
#     4   2        2                16.9%
#     6   2        2                 6.7%
#     6   3        3                 4.8%
#     8   3        3                 4.3%
#    10   3        3                 4.5%
#     8   4        4                 3.3%
#    12   4        4                 3.5%
#    10   5        5                 2.9%
#    16   5        5                 2.9%
#    20   6        6                 3.0%
#    30  12       12                 2.4%
#    40  20       20                 2.6%
#
# The driver is min(positive clusters, negative clusters), not the total: 10
# clusters split 3/7 still fires at 4.5%, while 10 split 5/5 fires at nominal.
# Re-run at 10000 trials to separate the two candidate floors:
#
#    (K,P) = (12,4) (16,4)  -> min 4 -> 3.61%, 4.21%
#    (K,P) = (10,5) (11,5) (12,5) (14,5) -> min 5 -> 3.12%, 3.01%, 3.14%, 3.19%
#
# so C1 is reported MISSING -- not PASS and not FAIL -- unless there are at
# least MIN_CLUSTERS_C1 clusters AND at least MIN_CLASS_CLUSTERS_C1 in each
# class. At that floor the rule still runs ~0.6 points hot (3.1% vs 2.5%), so
# it is a floor and not a certificate; the report says so.
# (Reproduce with `self_test`, scenario `O_c1_too_few_clusters`.)
MIN_CLUSTERS_C1 = MIN_CLUSTERS_RELIABLE   # 10 independent test subjects
MIN_CLASS_CLUSTERS_C1 = 5                 # ... at least 5 positive and 5 negative

# -- C8, headline vs the EMPIRICAL permutation null -------------------------
# C1 tests the headline against a hard-coded 0.500. That is only the chance
# level if the whole pipeline is unbiased, which is exactly what the label
# permutation control exists to measure and what C3 (a loose +/- 0.10 sanity
# band) does not establish. C8 uses the null stage 5 actually computed as the
# reference distribution, with the Phipson & Smyth (2010) estimator
#     p = (1 + #{null AUC >= observed}) / (1 + n_replicates)
# which is unbiased for a Monte-Carlo permutation test and never returns 0.
# Consequence: with r = 0 exceedances, p < ALPHA needs 1/(1+n) < 0.05, i.e.
# n >= 20 replicates. Below that C8 is MISSING (the test cannot reach the
# threshold), never PASS.
MIN_PERM_REPLICATES_FOR_P = int(math.ceil(1.0 / ALPHA))   # = 20 at ALPHA = 0.05

# -- C0, the pre-registered primary cohort ---------------------------------
# One RESULTS.md reports three cohorts. Applying C1 independently to each with
# no family-wise control gives a ~25% chance that at least one cohort clears it
# under a complete null (measured per-cohort null pass rates 7.2%, 5.2%,
# 15.2%). Rather than adjust C1 -- which would require pooling three bootstraps
# built on different folds -- one cohort is pre-registered as primary and the
# others are labelled exploratory in the verdict table, so there is exactly one
# confirmatory test in the family.
#
# prostate_t2 is primary because it is the largest cohort (67 patients) and the
# only one whose reconstruction is validated against the vendor's own images
# (r = 0.998). Neither fact depends on any outcome, which is what makes this a
# pre-registration rather than a selection.
PRIMARY_COHORT = "prostate_t2"

HEADLINE_CONDITION = "phase"
REFERENCE_CONDITION = "magnitude"

CONDITION_ORDER = ("magnitude", "phase", "both")
CONDITION_COLOR = {
    "magnitude": "#4C72B0",
    "phase": "#C44E52",
    "both": "#55A868",
}
STATUS_COLOR = {"pass": "#2E7D32", "fail": "#B71C1C", "missing": "#F9A825"}

MIN_PERMUTATION_REPLICATES = 5   # below this the null is not characterised at all


# ==========================================================================
# Confound cohorts: cohorts whose label is NOT a diagnosis
# ==========================================================================
# brain and knee carry no tumour annotation and never will. Their label is an
# ACQUISITION PROPERTY -- how many receive coils were used, which pulse sequence
# was played -- so every number computed on them is an answer to "how well can
# this input channel identify the hardware/protocol?", not to "how well can it
# find cancer".
#
# Two consequences are enforced in code rather than left to the caption:
#
#  1. A confound cohort never enters `evaluate_cohort`, never appears in the
#     diagnostic verdict table, and can never carry a verdict on the phase
#     hypothesis. The criteria are phrased in terms of tumour signal; applying
#     them to a coil-count label would be a category error that reads as a
#     result.
#  2. The INTERPRETATION IS INVERTED. For a clinical cohort a high phase AUC is
#     (weak, control-gated) evidence FOR the hypothesis. Here a high phase AUC
#     is evidence AGAINST it: it says the channel encodes the scanner, which is
#     the alternative explanation the whole study exists to exclude. Every
#     rendering of these numbers states that inversion next to the number.
#
# Excluding a cohort from the diagnostic table can only REDUCE the set of
# cohorts able to reach SUPPORTED, so this classification cannot manufacture a
# positive result. It is also belt-and-braces: PRIMARY_COHORT is prostate_t2, so
# neither brain nor knee could have been confirmatory anyway.

@dataclass(frozen=True)
class ConfoundCohortSpec:
    """What a confound cohort's label actually is, in the words of the paper."""
    cohort: str
    label_short: str          # goes in headings: never the word "tumour"
    label_long: str           # the full definition, positive class first
    positive_name: str        # what label == 1 means
    negative_name: str        # what label == 0 means
    paired: bool              # do subjects supply both classes?
    design_note: str
    why_it_matters: str
    # Does this cohort's label answer the question C6 asks -- "is scanner / coil /
    # site IDENTITY readable from this input?" Only a hardware-identity label
    # does. A protocol label such as fat suppression is plainly visible in the
    # magnitude image to a human radiologist; it is a real acquisition property
    # and belongs in this section, but letting it decide C6 would fail the
    # criterion on a contrast that was never hidden in the first place.
    feeds_c6: bool = False
    c6_note: str = ""


CONFOUND_COHORTS: Dict[str, ConfoundCohortSpec] = {
    "brain": ConfoundCohortSpec(
        cohort="brain",
        label_short="receive-coil count",
        label_long="receive-coil count >= 16 channels (pure hardware; there is no "
                   "pathology of any kind in this label)",
        positive_name=">= 16 receive coils",
        negative_name="< 16 receive coils",
        paired=False,
        design_note="UNPAIRED by construction: each subject was scanned on one coil "
                    "array and therefore supplies exactly one class. A subject-clustered "
                    "interval is the only honest one, and the split is enforced on "
                    "subject so no subject appears on both sides.",
        why_it_matters="This is the mechanism claim of the paper measured directly. It "
                       "replicates the prostate DWI confound result (phase -> "
                       "receiver_channels) on a cohort roughly three times larger and "
                       "with no paired structure to explain the result away.",
        feeds_c6=True,
        c6_note="Receive-coil count is literally one of the quantities C6 names "
                "(scanner / coil / site identity), it is invisible to a reader of the "
                "image, and it is measured here on 136 independent test subjects "
                "against a label with no pathology in it. Where this measurement "
                "exists it is the best-powered available answer to C6's question.",
    ),
    "knee": ConfoundCohortSpec(
        cohort="knee",
        label_short="pulse sequence",
        label_long="pulse sequence: CORPDFS_FBK (fat-suppressed proton density) vs "
                   "CORPD_FBK (proton density), i.e. protocol identity, not pathology",
        positive_name="CORPDFS_FBK (fat-suppressed)",
        negative_name="CORPD_FBK",
        paired=True,
        design_note="PAIRED: every subject supplies BOTH classes, and the receive-coil "
                    "count is uniform at 15 across the whole cohort. Subject identity and "
                    "coil count are therefore held constant across the label, so a "
                    "classifier cannot win here by recognising the patient or the "
                    "hardware -- only by recognising the sequence.",
        why_it_matters="The paired design removes the two confounds that could otherwise "
                       "explain the brain result: subject identity and coil count are "
                       "held constant across the label. Whatever separates the classes "
                       "here is a property of the acquisition itself. Note what that "
                       "does and does not establish -- stage 1's confound screen shows "
                       "the two classes also differ in echo time, echo spacing and flip "
                       "angle, so 'phase predicts knee contrast' is partly 'phase "
                       "reflects echo time', which is EXPECTED PHYSICS rather than a "
                       "hardware fingerprint. Knee is supporting evidence about "
                       "sequence-parameter dependence; the load-bearing fingerprint "
                       "claim is the brain coil-count result, and a large effect here "
                       "does not upgrade it.",
        feeds_c6=False,
        c6_note="This cohort does NOT feed C6. Fat suppression is a protocol choice that "
                "is plainly visible in the magnitude image -- the magnitude AUC here is "
                "close to 1.0 for exactly that reason -- so it is not the hidden "
                "scanner-identity confound C6's ceiling was calibrated against. Letting "
                "it decide C6 would fail the criterion on a contrast that was never "
                "concealed. It is reported here because the paired design is what rules "
                "out subject identity and coil count as explanations of the brain "
                "result.",
    ),
}

# The label the CLINICAL cohorts carry, for the columns that name every cohort's
# label side by side.
DIAGNOSTIC_LABEL_SHORT = "tumour present"
DIAGNOSTIC_LABEL_LONG = "clinically annotated tumour present on this slice"

# -- C6, wired to the direct measurement ------------------------------------
# C6 asks whether scanner/coil/site identity is predictable from the same input
# the diagnostic claim rests on. It used to be answerable only from the stage-5
# confound control on the clinical cohorts, whose test folds hold 4-7 subjects.
# The brain cohort measures the same quantity on 136 independent test subjects
# with a label that is nothing but hardware, so where that measurement exists it
# is the better-powered answer to the same question and C6 reads it.
#
# It is wired in ONE DIRECTION ONLY. An external confound measured at or above
# CONFOUND_AUC_MAX can FAIL C6 for a clinical cohort; it can never satisfy C6,
# never substitute for a missing stage-5 confound control, and never turn a
# MISSING into a PASS. The reason is asymmetry of evidence: "this input encodes
# the scanner" transfers across cohorts because it is a statement about the
# input channel and the physics, while "this input does not encode the scanner
# HERE" does not transfer, because a different cohort has different scanners.
# An external measurement may only be cited as decisive if it clears the same
# power floor every other control-based criterion is held to.
MIN_CLUSTERS_EXTERNAL_CONFOUND = MIN_CLUSTERS_RELIABLE


def confound_spec(cohort: str) -> Optional[ConfoundCohortSpec]:
    """The registry entry for a cohort, or None if it is a diagnostic cohort."""
    for key, spec in CONFOUND_COHORTS.items():
        if _ident(key) == _ident(cohort):
            return spec
    return None


def is_confound_cohort(cohort: str) -> bool:
    return confound_spec(cohort) is not None


# ==========================================================================
# Cross-validation
# ==========================================================================
# run_full.sh writes each stage-1 CV fold to its own results subdirectory,
# `<cohort>_cv<k>`, because s03 names its output `<cohort>_<condition>_seed<N>.json`
# with no fold component and five folds written to one directory would overwrite
# each other. The fold index therefore lives in the DIRECTORY NAME and nowhere
# else, which is why it is parsed here rather than read from the payload.
#
# The statistically correct treatment is OUT-OF-FOLD POOLING: every subject is in
# exactly one test fold, so concatenating the per-fold test predictions gives one
# prediction per subject over the whole cohort. That is one estimate at full
# power and one entry in the comparison family, instead of five underpowered
# estimates and a five-fold inflation of the family.
#
# The property that makes it valid -- each subject tested exactly once -- is
# CHECKED, not assumed (`pool_folds_oof`). If two folds share a subject, pooling
# would double-count them and narrow every interval, so pooling is refused and
# the report says why.
CV_DIR_RE = re.compile(r"^(?P<cohort>.+)_cv(?P<fold>\d+)$", re.IGNORECASE)
OFFICIAL_SCHEME = "official split"
OOF_SCHEME = "pooled out-of-fold (cross-validated)"

# The SPLIT FAMILY of a run: which experiment it belongs to. s04_stats writes
# exactly this string into every record of statistics.json (`split_family`), and
# s06 recomputes it the same way from the directory layout, so the two stages
# agree on which records describe which experiment. A pooled out-of-fold
# estimate and a single-split estimate are different experiments on different
# test sets: they may be reported side by side and must never be averaged.
CV_FAMILY = "cv"

# The columns stage 1/2 write for the k-fold sweep: cv0_split .. cv{K-1}_split,
# each holding training/validation/test per row. They are the DECLARED design of
# the cross-validation, written before any model ran, which is what makes them
# usable as the expected fold set: a fold that is missing from the results tree
# is then a fold that DIED, not a fold that was never planned.
CV_SPLIT_COL_RE = re.compile(r"^cv(?P<fold>\d+)_split$", re.IGNORECASE)

# The control names s05_controls.py actually writes, mapped to the canonical
# names used by the criteria. Anything outside this map is UNRECOGNISED.
S05_TO_CANONICAL: Dict[str, str] = {
    "label_permutation": "permutation",
    "background_only": "background",
    "phase_scramble": "scramble",
    "acquisition_split": "acquisition",
    "confound_predictability": "confound",
}
CANONICAL_LABEL = {
    "permutation": "label permutation",
    "background": "background only (anatomy removed)",
    "scramble": "phase scramble (spatial structure destroyed)",
    "acquisition": "acquisition-stratified split",
    "confound": "confound predictability",
}

CRITERIA_ORDER = (
    # C0 is not an experimental result: it records which cohort the report is
    # allowed to draw a confirmatory conclusion from. It is first because it
    # qualifies every criterion below it.
    ("C0", "preregistered_primary",
     f"Cohort is the pre-registered primary cohort ({PRIMARY_COHORT}); every "
     f"other cohort in the same report is exploratory"),
    ("C1", "phase_above_chance",
     f"Phase AUC 95% CI lower bound > 0.50 at the cluster level stage 4 marks "
     f"`preferred`, with >= {MIN_CLUSTERS_C1} test clusters and "
     f">= {MIN_CLASS_CLUSTERS_C1} in each class"),
    ("C2", "phase_beats_magnitude",
     f"Phase > magnitude by DeLong, Holm-adjusted p < {ALPHA}"),
    ("C3", "permutation_null",
     f"Label-permutation control AUC within 0.50 +/- {PERM_NULL_TOL:.2f} and range covers "
     f"0.50, over >= {MIN_PERMUTATION_REPLICATES} DISTINCT replicates spanning "
     f"<= {MAX_CONTROL_CI_WIDTH:.2f} AUC"),
    ("C4", "background_collapses",
     f"Background-only control at least {BACKGROUND_MARGIN:.2f} AUC below the headline, "
     f"and its own 95% CI contains {CONTROL_CHANCE:.2f} and is discriminating "
     f"(>= {MIN_CLUSTERS_CONTROL} clusters, width <= {MAX_CONTROL_CI_WIDTH:.2f}, finite, "
     f"non-degenerate)"),
    ("C5", "acquisition_stratified_holds",
     f"Acquisition-stratified split keeps CI lower bound > 0.50 and loses at most "
     f"{ACQ_EROSION_MAX:.2f} AUC, in BOTH split directions, on a discriminating interval"),
    ("C6", "confound_not_explanatory",
     f"Scanner/coil/site predictability from the same input < {CONFOUND_AUC_MAX:.2f} AUC, "
     f"and at least {CONFOUND_HEADLINE_MARGIN:.2f} AUC below the headline, with every "
     f"confound run stage 5 wrote actually scored -- and no directly measured confound "
     f"cohort at or above {CONFOUND_AUC_MAX:.2f} (a direct measurement can fail this "
     f"criterion but can never satisfy it)"),
    # C7 is not in the original six. Stage 5 implements a fifth falsification
    # control (phase scramble), and the rule "no SUPPORTED while a control
    # fails" has to cover it too, so it is gated rather than merely displayed.
    ("C7", "phase_scramble_collapses",
     f"Phase-scramble control at least {BACKGROUND_MARGIN:.2f} AUC below the headline "
     f"and its own 95% CI contains {CONTROL_CHANCE:.2f} and is discriminating "
     f"(>= {MIN_CLUSTERS_CONTROL} clusters, width <= {MAX_CONTROL_CI_WIDTH:.2f}, finite, "
     f"non-degenerate) "
     f"(a spatial effect must not survive destroying spatial structure)"),
    # C8 is the empirical counterpart of C1. C1 asks whether the headline beats
    # a hard-coded 0.500; C8 asks whether it beats the null stage 5 measured.
    ("C8", "beats_permutation_null",
     f"P(permuted-label AUC >= headline) < {ALPHA} over the label-permutation "
     f"replicates (needs >= {MIN_PERM_REPLICATES_FOR_P} of them to be reachable)"),
)


# ==========================================================================
# Small data holders
# ==========================================================================

@dataclass
class Estimate:
    """A point estimate with an interval and the provenance of both."""
    point: float = float("nan")
    lo: float = float("nan")
    hi: float = float("nan")
    n: Optional[int] = None
    n_pos: Optional[int] = None
    n_clusters: Optional[int] = None
    n_boot_valid: Optional[int] = None
    source: str = "unknown"
    per_seed: List[float] = field(default_factory=list)
    note: str = ""
    detail: dict = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return not math.isnan(self.point)

    @property
    def has_ci(self) -> bool:
        """An interval was reported at all. Says nothing about whether it is usable."""
        return not (math.isnan(self.lo) or math.isnan(self.hi))

    @property
    def ci_finite(self) -> bool:
        """
        ... and both of its bounds are real numbers.

        +/-inf reaches here from a JSON `Infinity` (Python's json module both
        writes and reads it) or from a bootstrap that divided by zero. It is not
        NaN, so `has_ci` is True and `lo <= 0.5 <= hi` is trivially satisfied --
        an interval that contains every possible AUC is not evidence that a
        control collapsed to one particular one. Everything that formats or
        plots an interval checks this first.
        """
        return self.has_ci and math.isfinite(self.lo) and math.isfinite(self.hi)

    @property
    def ci_width(self) -> float:
        return (self.hi - self.lo) if self.ci_finite else float("nan")


@dataclass(frozen=True)
class Headline:
    """
    THE headline number for one (cohort, condition, level), with its provenance.

    Resolved once, in `build_report`, and handed to `evaluate_cohort` as an
    object rather than as a bare AUC. Every criterion that quotes the headline
    quotes this, and says so on the Criterion it emits, which is what makes the
    two invariants below checkable at all:

      * a point estimate may never fall outside its own interval;
      * no two criteria may quote different headlines for the same
        (cohort, condition, level).

    Both were free gifts from a real report. `Stats.estimate` scoped RUN
    SELECTION to one split family and then overwrote the point estimate from an
    across-seeds row that spanned two, so RESULTS.md printed
    "headline 0.794 [0.600, 0.780]" -- a number outside its own interval -- in
    the C4 evidence, while C8, which reads the per-run blocks directly, quoted
    0.691 for the same cohort and condition in the same file. Neither criterion
    could see the disagreement because neither knew what the other had read.

    `n` / `n_clusters` / `folds` / `split_family` are the test-set fingerprint.
    Any criterion that DIFFERENCES something against this headline has to show
    that the other side was scored on the same test set, using
    `_test_set_mismatch`, which is the check C8 has always applied.
    """
    cohort: str
    condition: str
    level: str                       # "slice", "patient_mean", ...
    est: Estimate
    split_family: Optional[str] = None
    folds: Tuple[int, ...] = ()
    scheme: str = ""

    @property
    def key(self) -> str:
        return f"{self.cohort}/{self.condition}@{self.level}"

    @property
    def fingerprint(self) -> str:
        """What test set this number was computed on, in one printable string."""
        return (f"{self.est.n if self.est.n is not None else '?'} slices / "
                f"{self.est.n_clusters if self.est.n_clusters is not None else '?'} "
                f"subjects"
                + (f", folds {list(self.folds)}" if self.folds else "")
                + (f", split family {self.split_family!r}"
                   if self.split_family else ""))

    # -- delegations, so a criterion reads the headline the way it always did --
    @property
    def ok(self) -> bool:
        return self.est.ok

    @property
    def point(self) -> float:
        return self.est.point

    @property
    def lo(self) -> float:
        return self.est.lo

    @property
    def hi(self) -> float:
        return self.est.hi

    @property
    def has_ci(self) -> bool:
        return self.est.has_ci

    @property
    def per_seed(self) -> List[float]:
        return self.est.per_seed

    @property
    def n(self) -> Optional[int]:
        return self.est.n

    @property
    def n_clusters(self) -> Optional[int]:
        return self.est.n_clusters

    @property
    def source(self) -> str:
        return self.est.source

    @property
    def note(self) -> str:
        return self.est.note


def no_headline(cohort: str, condition: str = "", level: str = "slice") -> Headline:
    """A headline that does not exist. Every criterion reading it goes MISSING."""
    return Headline(cohort=cohort, condition=condition or HEADLINE_CONDITION,
                    level=level, est=Estimate(source="unavailable"))


@dataclass
class Criterion:
    key: str
    code: str
    rule: str
    status: str          # "pass" | "fail" | "missing"
    detail: str
    evidence: str = ""
    # -- which headline this criterion quoted, and what it did with it. Filled
    # in by `evaluate_cohort.add`; read by `_assert_verdict_consistent`.
    headline_key: str = ""
    headline_point: float = float("nan")
    # True when the criterion SUBTRACTED something from that headline (C4, C5,
    # C6, C7, C8). Such a criterion may only PASS when it has also shown that
    # both sides were scored on the same test set.
    differenced: bool = False
    test_set_verified: bool = False


@dataclass
class CohortVerdict:
    cohort: str
    verdict: str
    criteria: List[Criterion]
    reason: str
    # Default False so that a caller that forgets to say which cohort is the
    # pre-registered primary gets the conservative answer, not the flattering
    # one: `_assert_verdict_consistent` refuses SUPPORTED on a non-primary
    # cohort.
    is_primary: bool = False
    # key -> (point, lo, hi) for every headline any criterion quoted.
    headlines: Dict[str, Tuple[float, float, float]] = field(default_factory=dict)
    # One record per AGGREGATE any criterion was decided on -- an envelope over
    # seeds, a permutation null pool, a control pool over folds -- as
    # {"what", "reported", "members"}. `_assert_verdict_consistent` re-checks
    # every one of them against the invariant that an aggregate may never report
    # a test-set fingerprint its members do not all share.
    aggregates: List[dict] = field(default_factory=list)

    @property
    def role(self) -> str:
        return "PRIMARY (pre-registered)" if self.is_primary else "exploratory"

    @property
    def failing(self) -> List[Criterion]:
        return [c for c in self.criteria if c.status == "fail"]

    @property
    def missing(self) -> List[Criterion]:
        return [c for c in self.criteria if c.status == "missing"]


# ==========================================================================
# Generic loading helpers
# ==========================================================================

def _slug(s: Any) -> str:
    """
    Aggressive normalisation, for DICTIONARY KEY lookup only (see `_first`).

    It deletes every separator, so `ci_lo`, `ci-lo` and `cilo` all collide --
    which is what is wanted when guessing which key of a payload holds a lower
    bound, and emphatically NOT what is wanted when deciding whether two records
    describe the same cohort. Use `_ident` for that.
    """
    return re.sub(r"[^a-z0-9]", "", str(s).lower())


_IDENT_SEP = re.compile(r"[\s\-_.]+")


def _ident(s: Any) -> str:
    """
    Normalise an IDENTIFIER (cohort, condition, region, model name) for matching.

    Tolerant of case and of separator style -- `prostate-t2`, `Prostate_T2` and
    `prostate t2` are the same cohort -- but NOT separator-blind: `prostatet2`
    is a different string and is treated as a different cohort.

    That distinction is the whole point. `_slug` was being used to decide which
    stage-5 control payloads belong to the cohort under evaluation, and it maps
    every one of `prostate_t2`, `prostatet2`, `p.r.o.s.t.a.t.e.t.2` and
    `PROSTATE  T2` onto one key. A controls tree written for a neighbouring
    cohort -- or a payload whose `cohort` field was typed by hand -- was
    therefore credited to the primary cohort's criteria, which is a silent way
    of scoring a control that never ran on this data. A near-miss now yields no
    runs, hence a MISSING criterion, which is the honest answer.
    """
    t = _IDENT_SEP.sub("_", str(s).strip().lower())
    return t.strip("_")


def _first(d: Optional[dict], *names, default=None):
    """Return the first present key among names (case/punctuation tolerant)."""
    if not isinstance(d, dict):
        return default
    norm = {_slug(k): v for k, v in d.items()}
    for n in names:
        if n in d:
            return d[n]
        if _slug(n) in norm:
            return norm[_slug(n)]
    return default


def _num(x) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return float("nan")
    return v


def _as_estimate(obj: Any, source: str = "s04") -> Estimate:
    """
    Normalize the many shapes an interval estimate can arrive in.

    Accepts a bare float, [lo, hi]-style pairs, {"point","lo","hi"},
    {"auc","ci":[lo,hi]}, {"mean","ci_low","ci_high"}, and so on. Anything it
    cannot parse comes back as an empty Estimate (point = NaN), which the
    verdict engine treats as MISSING rather than as a pass.
    """
    est = Estimate(source=source)
    if obj is None:
        return est
    if isinstance(obj, (int, float)) and not isinstance(obj, bool):
        est.point = _num(obj)
        return est
    if isinstance(obj, (list, tuple)):
        vals = [_num(v) for v in obj]
        if len(vals) == 2:
            est.lo, est.hi = vals
        elif len(vals) == 3:
            est.lo, est.point, est.hi = sorted(vals)[0], vals[0], vals[-1]
        return est
    if not isinstance(obj, dict):
        return est

    point = _first(obj, "point", "point_estimate", "estimate", "value", "auc",
                   "mean", "median", "auroc", "score")
    # A nested {"auc": {...}} needs one more hop.
    if isinstance(point, (dict, list, tuple)):
        return _as_estimate(point, source=source)
    est.point = _num(point)

    ci = _first(obj, "ci", "ci95", "interval", "conf_int", "confidence_interval")
    if isinstance(ci, (list, tuple)) and len(ci) >= 2:
        est.lo, est.hi = _num(ci[0]), _num(ci[-1])
    else:
        est.lo = _num(_first(obj, "lo", "low", "lower", "ci_lo", "ci_low",
                             "ci_lower", "lcl", "q025", "p2_5", default=float("nan")))
        est.hi = _num(_first(obj, "hi", "high", "upper", "ci_hi", "ci_high",
                             "ci_upper", "ucl", "q975", "p97_5", default=float("nan")))

    for attr, names in (
        ("n", ("n", "n_slices", "n_test", "n_samples")),
        ("n_pos", ("n_pos", "n_positive", "positives")),
        ("n_clusters", ("n_clusters", "n_subjects", "n_patients", "clusters")),
        ("n_boot_valid", ("n_boot_valid", "n_resamples_valid", "n_boot")),
    ):
        v = _first(obj, *names)
        if v is not None:
            try:
                setattr(est, attr, int(v))
            except (TypeError, ValueError):
                pass

    ps = _first(obj, "per_seed", "auc_per_seed", "seeds", "values")
    if isinstance(ps, (list, tuple)):
        est.per_seed = [_num(v) for v in ps if not math.isnan(_num(v))]
    return est


def load_json(path: Path) -> Optional[dict]:
    try:
        with open(path, "r") as fh:
            return json.load(fh)
    except FileNotFoundError:
        return None
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not parse %s: %s", path, exc)
        return None


def _looks_like_run(d: Any) -> bool:
    """
    True for a stage-3 training run, False for a stage-5 control run.

    The `control` check is not cosmetic. A stage-5 payload is a stage-3 payload
    plus two keys, so it satisfies every structural test above, and one of the
    directories s06 searches for controls (`<results-dir>/controls`) is INSIDE
    the tree `load_runs` walks. Without this guard, pointing --results-dir at a
    layout that keeps controls under results/ pools the control predictions into
    the headline: the phase-scramble and acquisition-split payloads carry
    region="full", and the confound payloads carry SCANNER labels, so the
    "headline" would be a mixture of the model under test and the controls that
    are supposed to falsify it.

    `control: "none"` is kept: that is how stage 5 tags a re-run of the ordinary
    headline, which is a genuine stage-3 run.
    """
    if not (
        isinstance(d, dict)
        and "cohort" in d and "condition" in d
        and isinstance(d.get("test"), dict)
        and "probs" in d.get("test", {})
    ):
        return False
    ctrl = d.get("control")
    return ctrl is None or str(ctrl) == "none"


def _fold_of(path: Path, cohort: str) -> Tuple[Optional[int], List[str]]:
    """
    The CV fold a run belongs to, parsed from its results SUBDIRECTORY name.

    s03 writes `<cohort>_<condition>_seed<N>.json` with no fold component, so
    run_full.sh gives each fold its own directory (`prostate_t2_cv3`). The
    directory name is therefore the only record of which fold a file is, and it
    is a filesystem convention rather than data, so it is verified: the cohort
    prefix of the directory must be the cohort inside the payload. A directory
    called `breast_cv2` holding a prostate run is a mis-filed sweep, and pooling
    it as prostate fold 2 would silently mix cohorts.

    Returns (fold or None for the official split, defects).
    """
    m = CV_DIR_RE.match(path.parent.name)
    if not m:
        return None, []
    if _ident(m.group("cohort")) != _ident(cohort):
        return None, [
            f"{path}: lives in results subdirectory {path.parent.name!r}, which names "
            f"cohort {m.group('cohort')!r}, but the payload says cohort {cohort!r}. "
            f"Treated as an official-split run rather than as a fold of either cohort."
        ]
    return int(m.group("fold")), []


def load_runs(results_dir: Path, recurse: bool = True,
              defects: Optional[List[str]] = None) -> List[dict]:
    """
    Collect every stage-3 per-run JSON under results_dir.

    Non-run JSONs (statistics.json, controls.json, anything else) are skipped by
    shape, not by filename, so a renamed file still works.

    The walk is RECURSIVE because the cross-validation sweep writes one
    subdirectory per fold. Each run is tagged with `_fold` (None = the official
    split) so that downstream code can pool the folds out-of-fold instead of
    treating five folds of one cohort as five independent experiments.

    It is also tagged with `_split_family`, ported from s04_stats.load_runs and
    computed identically: `CV_FAMILY` for a fold-tagged run, otherwise the
    subdirectory name (`.` at the top level). Runs in different families are
    DIFFERENT EXPERIMENTS ON DIFFERENT TEST SETS -- a five-fold cross-validation
    and a leftover official-split sweep are not two seeds of one thing -- and
    nothing in this module may average across the boundary.

    ... and a fold-tagged run keeps the directory it was found UNDER, because
    `CV_FAMILY` alone collapsed every fold-tagged run into one family regardless
    of where it lived. `results/sweepA/<cohort>_cv0` and
    `results/sweepB/<cohort>_cv0` are both "fold 0 of a five-fold sweep" -- of
    two DIFFERENT five-fold sweeps -- and both were labelled `cv`, so the pooler
    put them in the same fold bucket and averaged their probability vectors.
    Measured: an honest sweep at AUC 0.63 averaged with an optimistic one at
    0.995 produced a single "pooled out-of-fold" headline of 0.938 over the same
    70 subjects, and the destroy-controls that FAIL against 0.63 PASS against
    0.938. The family of a fold run is therefore `cv` at the results root and
    `cv@<subdirectory>` below it; stage 4, which writes plain `cv`, then has no
    rows for a nested sweep's family, so C1/C2 go MISSING rather than being
    decided on a blend. Nothing about the ordinary one-sweep layout changes.
    """
    if not results_dir.exists():
        return []
    results_dir = Path(results_dir)
    paths = sorted(results_dir.rglob("*.json") if recurse else results_dir.glob("*.json"))
    runs = []
    for p in paths:
        d = load_json(p)
        if _looks_like_run(d):
            d["_path"] = str(p)
            d.setdefault("region", "full")
            d.setdefault("seed", 0)
            fold, defs = _fold_of(p, str(d.get("cohort")))
            d["_fold"] = fold
            d["_scheme"] = OFFICIAL_SCHEME if fold is None else OOF_SCHEME
            try:
                rel = p.parent.relative_to(results_dir)
                subdir = "" if str(rel) == "." else str(rel)
            except ValueError:                    # pragma: no cover - defensive
                subdir = str(p.parent)
            d["_subdir"] = subdir
            if fold is None:
                d["_split_family"] = subdir or "."
            else:
                # The directory holding the `<cohort>_cv<k>` directory. Empty at
                # the results root, which is the layout run_full.sh writes and
                # the one stage 4 assumes, so the family is the bare `cv` there.
                above = str(Path(subdir).parent) if subdir else "."
                d["_split_family"] = (CV_FAMILY if above in (".", "")
                                      else f"{CV_FAMILY}@{above}")
            if defs and defects is not None:
                defects.extend(defs)
            for msg in defs:
                logger.warning("%s", msg)
            runs.append(d)
    return runs


def OUT_ROOT_GUESS(results_dir: Path) -> Path:
    """
    The pipeline_out root, inferred from the results directory.

    Stage 5's default output tree is a SIBLING of the results tree
    (pipeline_out/controls/results next to pipeline_out/results), so s06 has to
    walk up rather than search underneath.
    """
    rd = Path(results_dir).resolve()
    return rd.parent if rd.name == "results" else rd


def find_first_existing(candidates: Iterable[Path]) -> Optional[Path]:
    for c in candidates:
        if c and Path(c).exists():
            return Path(c)
    return None


# ==========================================================================
# Clustering unit: subject_id, per the stage-1 contract
# ==========================================================================

def build_cluster_map(cohort: str, cache_dir: Path, cohorts_dir: Path) -> Tuple[Dict[int, str], str]:
    """
    Map cache row index -> subject_id.

    Stage 1 emits subject_id as the split-enforcement unit because breast has
    two acquisitions per patient and repeated-scan groups that link different
    coded names. The stage-3 run JSONs only carry patient_id, so we re-derive
    the subject through the cache index.

    RESOLUTION ORDER, ported verbatim from s04_stats.build_cluster_map so the
    two stages cannot disagree about what a cluster is:

      1. `{cohort}_index.csv:subject_id`. Stage 2 writes this column, and it is
         stage 4's FIRST choice. s06 used to skip it entirely and demand a join
         through the stage-1 cohort CSV, so a single unmatched basename -- a
         renamed file, a path rewritten between stages -- made `notna().all()`
         false and silently dropped the WHOLE cohort to patient_id. On breast
         that does not collapse the repeat-scan groups, so two acquisitions of
         one woman were bootstrapped as two independent patients and every
         interval in the report came out too narrow. The column that answers the
         question directly is read first.
      2. the cache index joined to the cohort CSV on basename(file).
      3. patient_id, named as such.

    Returns ({cache_idx: subject_id}, unit_name). Falls back to patient_id and
    says so, so the report can print which unit the intervals actually used.
    """
    idx_csv = cache_dir / f"{cohort}_index.csv"
    if not idx_csv.exists():
        return {}, "patient_id (cache index missing)"
    try:
        ci = pd.read_csv(idx_csv, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("could not read %s: %s", idx_csv, exc)
        return {}, "patient_id (cache index unreadable)"

    # -- 1. the column stage 2 writes, which is what stage 4 reads ----------
    if "subject_id" in ci.columns and "idx" in ci.columns:
        sid = ci["subject_id"]
        if len(sid) and sid.notna().all() and (sid.astype(str).str.strip() != "").all():
            logger.info("%s: cluster unit subject_id via %s:subject_id",
                        cohort, idx_csv.name)
            return dict(zip(ci["idx"].astype(int), sid.astype(str))), "subject_id"
        logger.warning(
            "%s: %s has a subject_id column but %d/%d rows are blank; trying the "
            "cohort-CSV join", cohort, idx_csv.name,
            int((sid.isna() | (sid.astype(str).str.strip() == "")).sum()), len(sid))

    co_csv = cohorts_dir / f"{cohort}_cohort.csv"
    if co_csv.exists():
        try:
            co = pd.read_csv(co_csv, low_memory=False)
            if "subject_id" in co.columns and "file" in co.columns:
                co = co.copy()
                co["_base"] = co["file"].map(lambda x: os.path.basename(str(x)))
                lut = co.drop_duplicates("_base").set_index("_base")["subject_id"].astype(str)
                base = ci["file"].map(lambda x: os.path.basename(str(x)))
                sid = base.map(lut)
                if sid.notna().all():
                    return dict(zip(ci["idx"].astype(int), sid.astype(str))), "subject_id"
                logger.warning(
                    "%s: %d/%d cache rows have no subject_id; falling back to patient_id",
                    cohort, int(sid.isna().sum()), len(sid),
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("subject_id join failed for %s: %s", cohort, exc)

    if "patient_id" in ci.columns:
        return dict(zip(ci["idx"].astype(int), ci["patient_id"].astype(str))), \
            "patient_id (no subject_id available)"
    return {}, "patient_id (fallback)"


# ==========================================================================
# What the cross-validation was DESIGNED to cover
# ==========================================================================

@dataclass
class CVExpectation:
    """
    The fold set and per-fold test subjects the cross-validation declared.

    Read from the `cv<k>_split` columns stage 1/2 write into the cache index (or
    the stage-1 cohort table), which were fixed before any model ran. That is
    what makes them an EXPECTATION rather than a description: a fold present in
    the design and absent from the results tree is a fold that DIED, and a fold
    dies for data-dependent reasons -- a single-class test block, an OOM on the
    largest fold. Missingness that depends on the data is not ignorable, so a
    pooled vector that quietly omits such a fold is not "the cohort minus a bit";
    it is a subset selected by the failure.

    `folds` may be empty: not every tree has the columns (a synthetic tree, an
    older cache). The caller then falls back to the union of folds observed
    across the cohort's own conditions, which still catches the case that
    matters most -- one condition pooling over fewer folds than its neighbours.
    """
    folds: List[int] = field(default_factory=list)
    subjects_by_fold: Dict[int, set] = field(default_factory=dict)
    source: str = ""
    reason: str = ""

    @property
    def available(self) -> bool:
        return bool(self.folds)

    @property
    def all_subjects(self) -> set:
        out: set = set()
        for s in self.subjects_by_fold.values():
            out |= s
        return out


def cv_expectation(cohort: str, cache_dir: Path, cohorts_dir: Path) -> CVExpectation:
    """
    Expected fold set (and expected test subjects per fold) for one cohort.

    Tries the stage-2 cache index first because that is the table whose rows the
    runs are indexed by, then the stage-1 cohort table. Returns an empty
    expectation with a reason rather than raising: an absent design table must
    degrade the guard to the weaker cross-condition check, not disable the
    report.
    """
    for path, label in ((Path(cache_dir) / f"{cohort}_index.csv", "cache index"),
                        (Path(cohorts_dir) / f"{cohort}_cohort.csv", "stage-1 cohort table")):
        if not path.exists():
            continue
        try:
            df = pd.read_csv(path, low_memory=False)
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read %s: %s", path, exc)
            continue
        cols = {}
        for c in df.columns:
            m = CV_SPLIT_COL_RE.match(str(c))
            if m:
                cols[int(m.group("fold"))] = c
        if not cols:
            continue
        unit = "subject_id" if "subject_id" in df.columns else (
            "patient_id" if "patient_id" in df.columns else None)
        subj: Dict[int, set] = {}
        if unit:
            for k, c in cols.items():
                sel = df[c].astype(str).str.strip().str.lower() == "test"
                subj[k] = {str(v) for v in df.loc[sel, unit].dropna().unique()}
        return CVExpectation(
            folds=sorted(cols),
            subjects_by_fold=subj,
            source=f"{path.name}:{', '.join(cols[k] for k in sorted(cols))}"
                   + (f" (test {unit}s)" if unit else " (fold set only)"),
        )
    return CVExpectation(
        reason=f"no cv<k>_split columns in {cohort}_index.csv or {cohort}_cohort.csv")


# ==========================================================================
# Pooling stage-3 runs and the fallback bootstrap
# ==========================================================================

@dataclass
class PooledPredictions:
    cohort: str
    condition: str
    region: str
    seeds: List[int]
    labels: np.ndarray
    probs: np.ndarray
    cache_idx: np.ndarray
    clusters: np.ndarray
    cluster_unit: str
    per_seed_auc: List[float]
    n_runs: int
    history: List[List[dict]]
    # -- the SAME rows, kept per training seed instead of averaged over them.
    # `probs` is the seed-MEAN probability, which is a 2-model ENSEMBLE and
    # scores higher than either constituent seed (prostate_t2 phase: seeds
    # 0.622/0.636, ensemble 0.650, s04's mean-over-seeds 0.629). The ensemble is
    # the right vector to bootstrap -- concatenating seeds would fake
    # independence and shrink every interval by sqrt(N) -- but it is the WRONG
    # curve to draw under a table that reports the mean of per-seed AUCs, and it
    # is the wrong basis for a matched comparison against a SINGLE-seed control.
    # Keyed by seed, each array aligned with `labels`/`probs`/`cache_idx`. Left
    # EMPTY whenever the seeds do not all cover exactly the same rows, so every
    # consumer must handle absence and none can silently report a partial seed.
    per_seed_probs: Dict[int, np.ndarray] = field(default_factory=dict)
    # -- cross-validation provenance. Defaulted so that every existing caller
    # keeps describing what it always described: a single official-split fold.
    scheme: str = OFFICIAL_SCHEME
    folds: List[int] = field(default_factory=list)
    per_fold: List[dict] = field(default_factory=list)
    # -- which experiment this is. Never averaged across; see CV_FAMILY.
    split_family: str = "."
    # -- coverage of the design. `expected_folds` is what the cross-validation
    # declared (cv<k>_split columns) or, failing that, the union of folds seen
    # across this cohort's own conditions. `coverage_defect` is non-empty
    # exactly when the pooled vector does NOT cover it, and every claim of the
    # form "every subject is tested exactly once" is gated on it being empty.
    expected_folds: List[int] = field(default_factory=list)
    missing_folds: List[int] = field(default_factory=list)
    expected_subjects: Optional[int] = None
    uncovered_subjects: List[str] = field(default_factory=list)
    coverage_defect: str = ""
    coverage_source: str = ""

    @property
    def is_pooled_oof(self) -> bool:
        return self.scheme == OOF_SCHEME

    @property
    def coverage_complete(self) -> bool:
        """
        True only when the pooled vector covers the declared design.

        Defaults to True for a non-CV estimate: an official split makes no
        coverage claim to violate.
        """
        return not self.coverage_defect

    @property
    def coverage_text(self) -> str:
        """One phrase stating what the estimate ACTUALLY covers."""
        if not self.is_pooled_oof:
            return f"{self.n_clusters} test subjects"
        folds = f"{len(self.folds)}"
        if self.expected_folds:
            folds += f" of {len(self.expected_folds)}"
        subj = f"{self.n_clusters}"
        if self.expected_subjects is not None:
            subj += f" of {self.expected_subjects}"
        return f"{folds} folds, {subj} subjects"

    @property
    def scheme_label(self) -> str:
        """
        How the estimate is described everywhere it is printed.

        The claim "every subject is tested exactly once" is made ONLY when the
        pooled fold set matches the declared one. When a fold is missing the
        label says so instead, because a fold dies for data-dependent reasons
        and the surviving subset is not a random sample of the cohort.
        """
        if not self.is_pooled_oof:
            return OFFICIAL_SCHEME
        if self.coverage_complete:
            return (f"{OOF_SCHEME}, {len(self.folds)} folds; every subject is tested "
                    f"exactly once ({self.n_clusters} subjects)")
        return (f"{OOF_SCHEME} over folds {self.folds} -- INCOMPLETE: "
                f"{self.coverage_text}; {self.coverage_defect}")

    @property
    def n(self) -> int:
        return int(len(self.labels))

    @property
    def n_pos(self) -> int:
        return int(self.labels.sum())

    @property
    def n_clusters(self) -> int:
        return int(len(np.unique(self.clusters)))

    @property
    def n_pos_clusters(self) -> int:
        pos = self.clusters[self.labels == 1]
        return int(len(np.unique(pos)))


def pool_runs(runs: Sequence[dict], cluster_map: Dict[int, str],
              cluster_unit: str, split: str = "test") -> Optional[PooledPredictions]:
    """
    Average per-slice probabilities over seeds for one (cohort, condition, region).

    Seeds are averaged rather than concatenated: concatenating N seeds' worth of
    predictions over the SAME 122 slices would make the bootstrap think it has
    N x 122 independent observations and shrink every interval by sqrt(N).
    Per-seed AUCs are kept separately so the report can show seed spread.

    All the runs handed here must belong to ONE split family. Averaging a fold
    of a cross-validation with a run from a leftover single-split sweep would
    intersect two disjoint test sets and average whatever happened to survive,
    which is not an estimate of anything. Callers group by family; this refuses
    if they did not.
    """
    runs = [r for r in runs if isinstance(r.get(split), dict) and r[split].get("probs")]
    if not runs:
        return None
    fams = sorted({str(r.get("_split_family", ".")) for r in runs})
    if len(fams) > 1:
        logger.error("refusing to pool runs from split families %s (%s/%s): these are "
                     "different experiments on different test sets",
                     fams, runs[0].get("cohort"), runs[0].get("condition"))
        return None

    by_idx: Dict[int, List[float]] = {}
    by_seed: Dict[int, Dict[int, float]] = {}
    labels: Dict[int, int] = {}
    pids: Dict[int, str] = {}
    per_seed_auc: List[float] = []
    seeds: List[int] = []
    history: List[List[dict]] = []

    common_idx: Optional[set] = None
    for r in runs:
        blk = r[split]
        idxs = [int(v) for v in blk.get("cache_idx", range(len(blk["probs"])))]
        common_idx = set(idxs) if common_idx is None else (common_idx & set(idxs))
    if not common_idx:
        return None

    for r in runs:
        blk = r[split]
        idxs = [int(v) for v in blk.get("cache_idx", range(len(blk["probs"])))]
        pid_list = [str(p) for p in blk.get("patient_ids", ["?"] * len(idxs))]
        seed_of_run = int(r.get("seed", 0))
        for k, prob, lab, pid in zip(idxs, blk["probs"], blk["labels"], pid_list):
            if k not in common_idx:
                continue
            by_idx.setdefault(k, []).append(float(prob))
            # Two runs claiming the same seed AND the same row are not two
            # measurements to average -- one of them is a re-run this function
            # cannot tell apart. Recorded as a collision, which empties the
            # per-seed view below rather than silently keeping the last writer.
            slot = by_seed.setdefault(seed_of_run, {})
            if k in slot:
                slot[k] = float("nan")
            else:
                slot[k] = float(prob)
            labels[k] = int(lab)
            pids[k] = pid
        auc = _num(blk.get("auc"))
        if not math.isnan(auc):
            per_seed_auc.append(auc)
        seeds.append(int(r.get("seed", 0)))
        if isinstance(r.get("history"), list):
            history.append(r["history"])

    order = sorted(by_idx)
    probs = np.array([float(np.mean(by_idx[k])) for k in order])
    labs = np.array([labels[k] for k in order], dtype=int)
    # A per-seed view only where EVERY seed scored EVERY pooled row exactly once.
    # A seed missing rows is not a curve over this test set, and reporting it
    # against the pooled n would be the size-borrowing this file refuses
    # everywhere else. Fails closed to {} -- consumers then fall back to the
    # ensemble and SAY they did.
    per_seed_probs: Dict[int, np.ndarray] = {}
    if len(by_seed) > 1 and all(
            len(slot) == len(order) and not any(math.isnan(v) for v in slot.values())
            for slot in by_seed.values()):
        per_seed_probs = {s: np.array([by_seed[s][k] for k in order], dtype=float)
                          for s in sorted(by_seed)}
    cidx = np.array(order, dtype=int)
    clusters = np.array(
        [cluster_map.get(int(k), pids.get(int(k), f"row{k}")) for k in order], dtype=object
    )
    unit = cluster_unit if cluster_map else "patient_id (from run JSON)"

    return PooledPredictions(
        cohort=str(runs[0].get("cohort")),
        condition=str(runs[0].get("condition")),
        region=str(runs[0].get("region", "full")),
        seeds=sorted(set(seeds)),
        labels=labs, probs=probs, cache_idx=cidx,
        clusters=clusters, cluster_unit=unit,
        per_seed_auc=per_seed_auc, n_runs=len(runs), history=history,
        per_seed_probs=per_seed_probs,
        split_family=fams[0],
    )


def _safe_auc(labels: np.ndarray, probs: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, probs))


def seed_mean_auc(p: "PooledPredictions",
                  sel: Optional[Sequence[int]] = None) -> Optional[Tuple[float, int]]:
    """
    Mean over seeds of the per-seed AUC -- the quantity stage 4 reports.

    THREE different numbers can be computed from the same runs, and this file
    used two of them interchangeably:

      (a) mean over seeds of each seed's AUC          <- stage 4, and the
                                                         pre-registered estimand
      (b) AUC of the seed-AVERAGED probability vector <- an ENSEMBLE of N models
      (c) AUC of the seeds CONCATENATED               <- fakes N x independence

    `PooledPredictions.probs` is (b), correctly: it is the vector the clustered
    bootstrap must run on, because (c) would shrink every interval by sqrt(N).
    But (b) is a DIFFERENT MODEL from the one the paper claims to evaluate --
    averaging two networks' outputs is variance reduction, and it scores above
    both constituents. Measured on prostate_t2: magnitude 0.695/0.671 -> (a)
    0.683, (b) 0.715; phase 0.622/0.636 -> (a) 0.629, (b) 0.650; both
    0.590/0.599 -> (a) 0.594, (b) 0.627. Drawing (b) under a table that reports
    (a) is what put every clinical cohort's ROC figure 0.02-0.03 above its own
    headline and sent the report DEGRADED.

    It also matters against the CONTROLS, which stage 5 runs at ONE seed: the
    background-only and scramble controls are (a) by construction, so
    differencing them against (b) compares a 2-model ensemble with a 1-model
    control and widens every gap in the direction of "there is signal".

    `sel` restricts to a subset of pooled rows (the matched-comparison case).
    Returns `(mean_auc, n_seeds)`, or None where there is no usable per-seed
    view -- one seed, seeds covering different rows, or a subset that leaves a
    single-class block. None means "say you are showing the ensemble", never
    "quietly show the ensemble".
    """
    if not p.per_seed_probs or len(p.per_seed_probs) < 2:
        return None
    labs = p.labels if sel is None else p.labels[list(sel)]
    if len(np.unique(labs)) < 2:
        return None
    aucs: List[float] = []
    for s in sorted(p.per_seed_probs):
        pr = p.per_seed_probs[s]
        a = _safe_auc(labs, pr if sel is None else pr[list(sel)])
        if math.isnan(a):
            return None
        aucs.append(a)
    return float(np.mean(aucs)), len(aucs)


def seed_mean_roc(p: "PooledPredictions", grid: np.ndarray
                  ) -> Optional[np.ndarray]:
    """
    The VERTICALLY AVERAGED ROC over seeds, interpolated onto `grid`.

    Area under this curve is the mean of the per-seed AUCs by linearity of the
    integral -- exactly `seed_mean_auc` -- so the curve a reader sees and the
    number printed beside it are the same quantity, which is the whole point.
    Returns None when there is no usable per-seed view.
    """
    from sklearn.metrics import roc_curve
    if not p.per_seed_probs or len(p.per_seed_probs) < 2:
        return None
    if len(np.unique(p.labels)) < 2:
        return None
    curves = []
    for s in sorted(p.per_seed_probs):
        fpr, tpr, _ = roc_curve(p.labels, p.per_seed_probs[s])
        curves.append(np.interp(grid, fpr, tpr))
    return np.mean(curves, axis=0)


def pool_folds_oof(per_fold: Dict[int, PooledPredictions],
                   expected: Optional[CVExpectation] = None,
                   expected_folds: Optional[Sequence[int]] = None,
                   ) -> Tuple[Optional[PooledPredictions], List[str]]:
    """
    Concatenate per-fold TEST predictions into one out-of-fold prediction vector.

    Why concatenate here when `pool_runs` deliberately AVERAGES over seeds:
    seeds share one test fold, so concatenating them would tell the bootstrap it
    had N x the observations it really has. Folds are the opposite case -- their
    test sets are DISJOINT by construction, each subject appearing in exactly
    one -- so concatenating them yields one prediction per subject over the full
    cohort. That is the whole point of doing the cross-validation: full power,
    ONE estimate, ONE entry in the comparison family, instead of five
    underpowered estimates and a five-fold inflation of the family.

    "Disjoint by construction" is the assumption the power gain rests on, so it
    is CHECKED rather than trusted. A subject appearing in two folds' test sets
    would be counted twice and would narrow every interval computed downstream;
    a repeated cache row likewise. Either one refuses the pooling and returns
    the reason, because an inflated-power estimate is worse than no estimate.

    COMPLETENESS is checked as well, and for the same reason. It used to refuse
    only the degenerate case of exactly one fold on disk, so a sweep in which
    fold 3 died -- run_full.sh continues past a failed fold and logs it to
    run_full_failures.txt, which no stage reads -- pooled four folds, called
    itself "every subject tested exactly once", and decided C1 on the survivors.
    A fold dies for DATA-DEPENDENT reasons (a single-class test block, an OOM on
    the largest fold), so the surviving subjects are not a random subsample and
    the missingness is not ignorable. The pooled vector is still returned,
    because the per-fold numbers alone would tell the reader less, but it is
    stamped with `coverage_defect`, every label it appears under states what it
    actually covers, and the caller caps the criteria decided on it at MISSING.

    `expected` is the DECLARED design (cv<k>_split columns); `expected_folds` is
    the weaker fallback -- the union of folds seen across the cohort's own
    conditions -- used when the design table has no cv columns.

    Returns (pooled or None, defects).
    """
    defects: List[str] = []
    folds = sorted(per_fold)
    if not folds:
        return None, defects

    # -- what the sweep was supposed to cover -------------------------------
    exp_folds: List[int] = []
    exp_source = ""
    if expected is not None and expected.available:
        exp_folds = list(expected.folds)
        exp_source = expected.source
    elif expected_folds:
        exp_folds = sorted({int(f) for f in expected_folds})
        exp_source = ("the union of folds present for this cohort's other conditions "
                      "(no cv<k>_split columns to read the design from)")
    missing = [f for f in exp_folds if f not in per_fold]

    if len(folds) == 1:
        defects.append(
            f"only fold {folds[0]} is on disk"
            + (f" of an expected {exp_folds}" if exp_folds else "")
            + ", so 'pooled out-of-fold' would be a "
              "single fold under another name; it is reported as one fold, not pooled")
        return None, defects

    seen_idx: Dict[int, int] = {}
    seen_cluster: Dict[Any, int] = {}
    overlaps_idx: List[str] = []
    overlaps_cluster: List[str] = []
    for f in folds:
        p = per_fold[f]
        for k in p.cache_idx.tolist():
            if k in seen_idx and len(overlaps_idx) < 5:
                overlaps_idx.append(f"cache row {k} is in the test set of folds "
                                    f"{seen_idx[k]} and {f}")
            seen_idx.setdefault(int(k), f)
        for c in np.unique(p.clusters).tolist():
            if c in seen_cluster and len(overlaps_cluster) < 5:
                overlaps_cluster.append(f"subject {c!r} is tested in folds "
                                        f"{seen_cluster[c]} and {f}")
            seen_cluster.setdefault(c, f)
    if overlaps_idx or overlaps_cluster:
        defects.append(
            "the CV folds are NOT disjoint, so concatenating them would test some "
            "subjects more than once and narrow every interval computed from the "
            "result: " + "; ".join((overlaps_cluster + overlaps_idx)[:5])
            + ". Out-of-fold pooling is refused; per-fold estimates are reported "
              "instead.")
        return None, defects

    units = {p.cluster_unit for p in per_fold.values()}
    if len(units) > 1:
        defects.append(
            "the folds were clustered on different units (" + ", ".join(sorted(units))
            + "), so a pooled interval would mix clustering levels; pooling refused.")
        return None, defects

    labels = np.concatenate([per_fold[f].labels for f in folds])
    probs = np.concatenate([per_fold[f].probs for f in folds])
    cidx = np.concatenate([per_fold[f].cache_idx for f in folds])
    clusters = np.concatenate([per_fold[f].clusters for f in folds])
    # The per-seed view survives pooling only if EVERY fold has one and they
    # all ran the SAME seeds. Folds trained on different seed sets do not
    # concatenate into "seed s across the cohort" -- that vector would be seed s
    # here and seed t there. Fails closed to {}, in the same fold order as
    # `probs` so the two stay row-aligned.
    seed_sets = [set(per_fold[f].per_seed_probs) for f in folds]
    per_seed_probs: Dict[int, np.ndarray] = {}
    if all(seed_sets) and all(s == seed_sets[0] for s in seed_sets):
        per_seed_probs = {
            s: np.concatenate([per_fold[f].per_seed_probs[s] for f in folds])
            for s in sorted(seed_sets[0])}

    rows = []
    for f in folds:
        p = per_fold[f]
        rows.append({
            "fold": f, "auc": _safe_auc(p.labels, p.probs),
            "n": p.n, "n_pos": p.n_pos,
            "n_clusters": p.n_clusters, "n_pos_clusters": p.n_pos_clusters,
            "seeds": list(p.seeds), "per_seed_auc": list(p.per_seed_auc),
        })

    # -- coverage of the declared design ------------------------------------
    # Two independent readings, because they fail independently: a fold absent
    # from the tree (the sweep died) and a subject the design assigned to a test
    # fold that is nevertheless absent from the pooled vector (the fold ran but
    # dropped rows).
    covered = {str(c) for c in np.unique(clusters).tolist()}
    exp_subjects: Optional[int] = None
    uncovered: List[str] = []
    if expected is not None and expected.available and expected.all_subjects:
        design = expected.all_subjects
        exp_subjects = len(design)
        uncovered = sorted(design - covered)

    reasons: List[str] = []
    if missing:
        reasons.append(
            f"fold(s) {missing} of the declared set {exp_folds} produced no usable "
            f"predictions, so {len(folds)} of {len(exp_folds)} folds were pooled")
    if uncovered:
        reasons.append(
            f"{len(uncovered)} of {exp_subjects} subjects the design assigns to a test "
            f"fold are absent from the pooled vector (e.g. "
            + ", ".join(repr(s) for s in uncovered[:3])
            + (", ..." if len(uncovered) > 3 else "") + ")")
    coverage_defect = "; ".join(reasons)
    if coverage_defect:
        defects.append(
            f"the pooled out-of-fold estimate for {per_fold[folds[0]].condition} does "
            f"NOT cover the cross-validation design: {coverage_defect}"
            + (f" [design read from {exp_source}]" if exp_source else "")
            + ". A fold fails for data-dependent reasons, so the subjects that "
              "survive are not a random subsample of the cohort; the estimate is "
              "reported over what it actually covers and the criteria decided on it "
              "are capped at MISSING.")

    any_p = per_fold[folds[0]]
    pooled = PooledPredictions(
        cohort=any_p.cohort, condition=any_p.condition, region=any_p.region,
        seeds=sorted({s for f in folds for s in per_fold[f].seeds}),
        labels=labels, probs=probs, cache_idx=cidx, clusters=clusters,
        cluster_unit=any_p.cluster_unit,
        # Per-FOLD AUCs, not per-seed: the seed spread was already collapsed
        # inside each fold by pool_runs. Reusing the field keeps the dispersion
        # dots on the AUC figure meaningful, and every label calls them folds.
        per_seed_auc=[r["auc"] for r in rows if not math.isnan(r["auc"])],
        n_runs=sum(per_fold[f].n_runs for f in folds),
        history=[h for f in folds for h in per_fold[f].history],
        per_seed_probs=per_seed_probs,
        scheme=OOF_SCHEME, folds=folds, per_fold=rows,
        split_family=any_p.split_family,
        expected_folds=exp_folds, missing_folds=missing,
        expected_subjects=exp_subjects, uncovered_subjects=uncovered[:20],
        coverage_defect=coverage_defect, coverage_source=exp_source,
    )
    return pooled, defects


def cluster_bootstrap(labels: np.ndarray, probs: np.ndarray, clusters: np.ndarray,
                      n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0,
                      fpr_grid: Optional[np.ndarray] = None,
                      level: float = 0.95) -> Dict[str, Any]:
    """
    Cluster (subject) bootstrap for AUC and, optionally, the ROC band.

    Resampling subjects with replacement -- not slices -- is the only honest
    option here: adjacent slices of the same prostate are near-duplicates, and a
    slice-level bootstrap on 122 correlated slices reports an interval roughly
    sqrt(30) times too narrow.

    Resamples that end up single-class contribute nothing and are counted; with
    3 positive subjects out of 4 that happens often, and the count is reported
    so the reader can see how thin the resampling actually was.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    out: Dict[str, Any] = {
        "point": float("nan"), "lo": float("nan"), "hi": float("nan"),
        "n": int(len(labels)), "n_pos": int(np.sum(labels)),
        "n_clusters": int(len(np.unique(clusters))),
        "n_boot_valid": 0, "tpr_lo": None, "tpr_hi": None,
    }
    if len(np.unique(labels)) < 2:
        return out
    out["point"] = float(roc_auc_score(labels, probs))

    uniq = np.unique(clusters)
    groups = [np.where(clusters == c)[0] for c in uniq]
    rng = np.random.default_rng(seed)
    aucs: List[float] = []
    tprs: List[np.ndarray] = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([groups[i] for i in pick])
        y, p = labels[sel], probs[sel]
        if len(np.unique(y)) < 2:
            continue
        aucs.append(float(roc_auc_score(y, p)))
        if fpr_grid is not None:
            f, t, _ = roc_curve(y, p)
            tprs.append(np.interp(fpr_grid, f, t))
    out["n_boot_valid"] = len(aucs)
    if aucs:
        a = float((1.0 - level) / 2.0 * 100.0)
        out["lo"] = float(np.percentile(aucs, a))
        out["hi"] = float(np.percentile(aucs, 100.0 - a))
    if fpr_grid is not None and tprs:
        arr = np.vstack(tprs)
        a = float((1.0 - level) / 2.0 * 100.0)
        out["tpr_lo"] = np.percentile(arr, a, axis=0)
        out["tpr_hi"] = np.percentile(arr, 100.0 - a, axis=0)
    return out


def cluster_bootstrap_delta(p_head: "PooledPredictions", p_ref: "PooledPredictions",
                            n_boot: int = DEFAULT_BOOTSTRAP, seed: int = 0,
                            level: float = 0.95) -> Dict[str, Any]:
    """
    Subject-clustered bootstrap interval for AUC(head) - AUC(ref), PAIRED.

    This exists to gate a sentence, not to replace a test. The confound
    paragraph used to call phase "the BETTER predictor of the acquisition
    property" on the bare sign of the point difference, so a +0.007 gap whose
    interval spans zero -- and whose sign flips between seeds -- rendered as a
    finding. That is the one direction this file is not allowed to lean.

    Both conditions are scored on the SAME subjects, so the resample draws
    subjects once and scores both vectors on that draw; the difference is
    therefore paired and its interval is far narrower than the overlap of the
    two marginal intervals would suggest. Rows are aligned on cache_idx, so a
    condition that scored a different set of rows contributes nothing rather
    than being silently compared against a different test set.

    Returns `ok=False` when the two vectors cannot be aligned or resampled;
    every caller treats that as "cannot claim an ordering", never as one.
    """
    from sklearn.metrics import roc_auc_score

    out: Dict[str, Any] = {"ok": False, "point": float("nan"),
                           "lo": float("nan"), "hi": float("nan"),
                           "n_clusters": 0, "n_boot_valid": 0,
                           "excludes_zero": False, "reason": ""}
    if p_head is None or p_ref is None:
        out["reason"] = "a condition is missing"
        return out

    common = np.intersect1d(p_head.cache_idx, p_ref.cache_idx)
    if len(common) == 0:
        out["reason"] = "the two conditions scored no rows in common"
        return out
    hsel = np.searchsorted(p_head.cache_idx, common,
                           sorter=np.argsort(p_head.cache_idx))
    hord = np.argsort(p_head.cache_idx)[hsel]
    rord = np.argsort(p_ref.cache_idx)[
        np.searchsorted(p_ref.cache_idx, common, sorter=np.argsort(p_ref.cache_idx))]

    y = np.asarray(p_head.labels)[hord]
    if not np.array_equal(y, np.asarray(p_ref.labels)[rord]):
        out["reason"] = "the two conditions disagree on the labels of shared rows"
        return out
    if len(np.unique(y)) < 2:
        out["reason"] = "shared rows are single-class"
        return out

    ph = np.asarray(p_head.probs)[hord]
    pr = np.asarray(p_ref.probs)[rord]
    clusters = np.asarray(p_head.clusters, dtype=object)[hord]

    out["point"] = float(roc_auc_score(y, ph)) - float(roc_auc_score(y, pr))
    uniq = np.unique(clusters)
    out["n_clusters"] = int(len(uniq))
    if len(uniq) < 2:
        out["reason"] = "fewer than 2 clusters; no interval is defensible"
        return out

    groups = [np.where(clusters == c)[0] for c in uniq]
    rng = np.random.default_rng(seed)
    deltas: List[float] = []
    for _ in range(int(n_boot)):
        pick = rng.integers(0, len(groups), len(groups))
        sel = np.concatenate([groups[i] for i in pick])
        yb = y[sel]
        if len(np.unique(yb)) < 2:
            continue
        deltas.append(float(roc_auc_score(yb, ph[sel]))
                      - float(roc_auc_score(yb, pr[sel])))
    out["n_boot_valid"] = len(deltas)
    if not deltas:
        out["reason"] = "every resample was single-class"
        return out
    a = float((1.0 - level) / 2.0 * 100.0)
    out["lo"] = float(np.percentile(deltas, a))
    out["hi"] = float(np.percentile(deltas, 100.0 - a))
    out["ok"] = True
    out["excludes_zero"] = bool(out["lo"] > 0.0 or out["hi"] < 0.0)
    return out


# ==========================================================================
# Statistics / controls access layer
# ==========================================================================

class Stats:
    """
    Read-only view over the statistics.json that s04_stats.py actually writes.

    That file is a flat list of per-run records plus a flat list of per-seed,
    per-level comparisons -- not a cohort-keyed tree. Two aggregation decisions
    are made here and stated wherever their output is printed:

    * ENVELOPE OVER SEEDS. Seeds share one test fold, so the spread across seeds
      is training stochasticity, not sampling uncertainty (s04 says exactly this
      in its own `caveat` field). Averaging the per-seed intervals would narrow
      them for no statistical reason. The point estimate is the mean over seeds;
      the interval is the widest seen; the bootstrap count reported is the
      thinnest seen.

    * PREFERRED COMPARISON LEVEL. s04 marks `preferred: true` on the
      patient-level comparisons because its own DeLong caveat says the
      slice-level p-value is anti-conservative. s06 follows that mark.
    """

    LEVEL_KEY = {
        "slice": "slice_level",
        "patient_mean": "patient_level_mean",
        "patient_max": "patient_level_max",
    }

    def __init__(self, raw: Optional[dict], path: Optional[Path]):
        self.raw = raw or {}
        self.path = path
        self.available = bool(raw)
        self.runs = [r for r in self.raw.get("runs", []) if isinstance(r, dict)]
        self.across = [a for a in self.raw.get("across_seeds", []) if isinstance(a, dict)]
        self.comparisons = [c for c in self.raw.get("comparisons", [])
                            if isinstance(c, dict)]
        self.warnings = [str(w) for w in (self.raw.get("warnings") or [])]
        self.config = self.raw.get("config") or {}

    # -- provenance ------------------------------------------------------
    def bootstrap_note(self) -> str:
        if not self.available:
            return "unavailable"
        n = _first(self.config, "n_boot", default="?")
        alpha = _num(_first(self.config, "alpha", default=0.05))
        unit = self.cluster_unit() or "?"
        method = _first(self.config, "ci_method", default="percentile bootstrap")
        lvl = "" if math.isnan(alpha) else f", {(1 - alpha):.2f} interval"
        return (f"{n} resamples, {method}, clustered on {unit}{lvl}; across seeds "
                f"s06 reports the mean point estimate and the WIDEST interval "
                f"(seeds share one test fold, so averaging intervals would "
                f"narrow them for no statistical reason)")

    def cluster_unit(self) -> str:
        units = {str(r.get("cluster_unit")) for r in self.runs if r.get("cluster_unit")}
        return "/".join(sorted(units)) if units else ""

    def stage3_mismatches(self) -> List[str]:
        """Runs where s04's recomputed AUC disagrees with what stage 3 reported."""
        out = []
        for r in self.runs:
            if r.get("auc_matches_stage3") is False:
                out.append(f"{r.get('tag')}: stage-3 reported "
                           f"{r.get('reported_test_auc')}, stage 4 recomputed "
                           f"{(r.get('slice_level') or {}).get('auc')}")
        return out

    def cohorts(self) -> List[str]:
        return sorted({str(r.get("cohort")) for r in self.runs if r.get("cohort")})

    # -- split family ----------------------------------------------------
    @staticmethod
    def family_of(rec: dict) -> str:
        """
        The split family of one stage-4 record, however old the file is.

        s04 writes `split_family` on every run and every comparison. A file
        written before that field existed is classified from the two fields it
        does carry -- `pooled` / `folds` -- so an out-of-fold pool is never
        mistaken for a single split.
        """
        fam = rec.get("split_family")
        if fam not in (None, ""):
            return str(fam)
        return CV_FAMILY if (rec.get("pooled") or rec.get("folds")) else "."

    def families_for(self, cohort: str, condition: str,
                     region: str = "full") -> List[str]:
        """Every split family statistics.json holds for this (cohort, condition)."""
        return sorted({self.family_of(r)
                       for r in self.runs_for(cohort, condition, region, family=None)})

    # -- estimates -------------------------------------------------------
    def runs_for(self, cohort: str, condition: str, region: str = "full",
                 family: Optional[str] = None) -> List[dict]:
        out = [r for r in self.runs
               if _ident(r.get("cohort")) == _ident(cohort)
               and _ident(r.get("condition")) == _ident(condition)
               and _ident(r.get("region", "full")) == _ident(region)]
        if family is None:
            return out
        return [r for r in out if _ident(self.family_of(r)) == _ident(family)]

    def estimate(self, cohort: str, condition: str, level: str = "slice",
                 region: str = "full", family: Optional[str] = None) -> Estimate:
        """
        The stage-4 estimate for one (cohort, condition, level) in ONE split family.

        `family` is not optional in spirit. This used to key on
        (cohort, condition, region) alone, so a pooled out-of-fold row and any
        leftover single-split row for the same condition were averaged into one
        number and the result was labelled "N seed(s)". They are not seeds of one
        experiment: they are two experiments on two different test sets, and the
        smaller one is typically a handful of subjects whose AUC can be 1.000.
        One leftover 4-subject file moved a real 70-subject headline from 0.691
        to 0.794 -- which is more than the whole margin C4 and C7 are asking
        about -- while the report went on describing the number as the pooled
        out-of-fold estimate over 70 subjects.

        Callers pass the family of the predictions they are actually headlining.
        With `family=None` and more than one family present, no estimate is
        returned at all: MISSING is the honest answer, and it is the safe one.
        """
        key = self.LEVEL_KEY.get(level, "slice_level")
        fams = self.families_for(cohort, condition, region)
        if family is None and len(fams) > 1:
            est = Estimate(source="s04-ambiguous-split-family")
            est.note = (f"statistics.json holds {len(fams)} split families for "
                        f"{cohort}/{condition} ({', '.join(fams)}); they are different "
                        f"experiments on different test sets and are never averaged")
            est.detail = {"split_families": fams}
            return est
        runs = self.runs_for(cohort, condition, region, family=family)
        pairs = [(r, r.get(key)) for r in runs if isinstance(r.get(key), dict)]
        blocks = [b for _, b in pairs]
        pts = [_num(b.get("auc")) for b in blocks]
        pts = [v for v in pts if not math.isnan(v)]
        if not pts:
            est = Estimate(source="s04-missing")
            if family is not None and fams:
                est.note = (f"statistics.json has no {family!r} rows for "
                            f"{cohort}/{condition}; it holds {', '.join(fams)}. Re-run "
                            f"stage 4 against the predictions this report headlines")
            return est

        los = [_num(b.get("ci_lo")) for b in blocks]
        his = [_num(b.get("ci_hi")) for b in blocks]
        los = [v for v in los if not math.isnan(v)]
        his = [v for v in his if not math.isnan(v)]
        boots = [int(b.get("n_boot_used", 0) or 0) for b in blocks]
        b0 = blocks[0]

        # s04's own across-seed mean, when it PROVES it agrees on scope, is
        # authoritative. `aggregate_across_seeds` groups by
        # (cohort, condition, region) ONLY, so its mean spans split families
        # whenever more than one is on disk.
        #
        # The declaration is now REQUIRED rather than merely respected when
        # present. `family is None or not acr_fams` granted the override to any
        # across row that simply did not say what it covered -- which re-opened
        # the whole defect this method exists to close, because run selection is
        # scoped to a family and then the point estimate was overwritten from an
        # unscoped row. Measured on a 70-subject five-fold tree with one leftover
        # 4-subject single-split file at AUC 1.000 and an across row carrying
        # neither `split_families` nor `mixes_split_families`: the headline moved
        # 0.6909 -> 0.7939, the report printed "headline 0.794 [0.600, 0.780]" --
        # a point estimate outside its own interval -- while C8, which reads the
        # per-run blocks, went on quoting 0.691 two sections later.
        #
        # An undeclared scope is not evidence of a matching scope. Where the row
        # does not say, the mean over the per-run blocks OF THE REQUESTED FAMILY
        # is used instead: the same computation, done here, over records whose
        # family is known.
        point = float(np.mean(pts))
        acr = self._across(cohort, condition, region)
        acr_key = {"slice": "slice_auc", "patient_mean": "patient_mean_auc",
                   "patient_max": "patient_max_auc"}.get(level)
        acr_fams = [str(f) for f in ((acr or {}).get("split_families") or [])]
        # With family=None there is exactly one family on disk (more than one
        # returned above), and that single family is the scope being asked for.
        scope = family if family is not None else (fams[0] if len(fams) == 1 else None)
        acr_scoped = (bool(acr)
                      and not bool(acr.get("mixes_split_families"))
                      and bool(acr_fams)          # it must SAY what it covers ...
                      and scope is not None
                      and {_ident(f) for f in acr_fams} == {_ident(scope)})
        if acr and acr_scoped and acr_key and isinstance(acr.get(acr_key), dict):
            m = _num(acr[acr_key].get("mean"))
            if not math.isnan(m):
                point = m

        n_seeds = len(pts)
        # Every block's sample size is kept, not just blocks[0]'s. The guard that
        # asks "does stage 4 describe the predictions this report headlines?"
        # reads these, and reading one arbitrary block made the answer depend on
        # the order records happened to be written in.
        n_set = sorted({int(b.get("n_slices", 0) or 0) for b in blocks
                        if b.get("n_slices")})
        ncl_set = sorted({int(b.get("n_clusters", 0) or 0) for b in blocks
                          if b.get("n_clusters")})
        est = Estimate(
            point=point,
            lo=min(los) if los else float("nan"),
            hi=max(his) if his else float("nan"),
            n_boot_valid=min(boots) if boots else None,
            source=f"s04 {level}-level",
            per_seed=[round(v, 4) for v in pts],
        )
        est.note = (f"{n_seeds} seed(s); interval is the envelope over seeds"
                    if n_seeds > 1 else "single seed")
        if family is not None:
            est.note += f"; split family {family!r}"
        est.detail = {"cluster_unit": b0.get("cluster_unit"),
                      "n_pos_clusters": b0.get("n_pos_clusters"),
                      "n_skipped_single_class": b0.get("n_skipped_single_class"),
                      "split_family": family, "split_families_present": fams,
                      "n_set": n_set, "n_clusters_set": ncl_set,
                      "folds": sorted({tuple(r.get("folds") or ()) for r in runs})}
        # THE ROW'S SAMPLE SIZE IS THE ONE EVERY BLOCK DECLARES. `b0` is
        # `blocks[0]`, i.e. whichever seed statistics.json happened to list
        # first; the sizes it carried were then handed to the headline and, from
        # there, to `_test_set_mismatch` as the test set every control had to
        # match. `n_set` / `n_clusters_set` (kept above, and read by the stage-4
        # provenance guard) record the whole spread; these two fields must not
        # claim a single value the blocks do not agree on.
        stat_conflicts = _agree_or_forget(
            est,
            [{"split_family": Stats.family_of(r),
              "folds": r.get("folds"),
              "n_clusters": b.get("n_clusters") or None,
              "n": b.get("n_slices") or None,
              "n_pos": b.get("n_pos_slices") or None,
              "subjects": None}
             for r, b in pairs],
            f"stage 4's {len(pairs)} {level}-level row(s) for "
            f"{cohort}/{condition}")
        if stat_conflicts:
            est.note += ("; stage 4's own rows for this condition do not describe one "
                         "test set (" + "; ".join(stat_conflicts) + "), so this row "
                         "carries no sample size")
        return est

    def _across(self, cohort: str, condition: str, region: str = "full") -> Optional[dict]:
        for a in self.across:
            if (_ident(a.get("cohort")) == _ident(cohort)
                    and _ident(a.get("condition")) == _ident(condition)
                    and _ident(a.get("region", "full")) == _ident(region)):
                return a
        return None

    def preferred_level(self, cohort: str, region: str = "full",
                        family: Optional[str] = None) -> str:
        """
        The level stage 4 marks `preferred` for this cohort, never "slice".

        s04_stats writes `preferred=(how == "mean")` on the per-cluster
        comparison rows and `preferred=False` on the slice row, because its own
        DeLong caveat says the slice-level test is anti-conservative: slices
        within a subject are correlated, so the slice row violates the
        independence assumption. The same correlation is what makes the
        slice-level bootstrap interval undercover, so C1 is read at this level
        too.

        "slice" is refused even if some future stage 4 marks it preferred. The
        whole point of the criterion is that it is decided on independent
        units; a mark in an input file cannot license reading it on units that
        are not independent. Falls back to C1_LEVEL when nothing is marked
        (e.g. stage 4 ran but emitted no comparisons), which is the same
        cluster level s04 would have chosen.
        """
        marked = []
        for c in self.comparisons:
            if (_ident(c.get("cohort")) == _ident(cohort)
                    and _ident(c.get("region", "full")) == _ident(region)
                    and (family is None
                         or _ident(self.family_of(c)) == _ident(family))
                    and bool(c.get("preferred"))):
                lvl = str(c.get("level", ""))
                if lvl and lvl != "slice" and lvl not in marked:
                    marked.append(lvl)
        if not marked:
            return C1_LEVEL
        if len(marked) > 1:
            # More than one cluster-level row marked preferred: take the one
            # whose key is alphabetically first only after logging, so the
            # choice is visible rather than an accident of dict ordering.
            logger.warning("%s: stage 4 marks %d levels preferred (%s); using %s",
                           cohort, len(marked), ", ".join(sorted(marked)),
                           sorted(marked)[0])
            return sorted(marked)[0]
        return marked[0]

    def seed_spread(self, cohort: str, condition: str, region: str = "full",
                    family: Optional[str] = None) -> str:
        a = self._across(cohort, condition, region)
        if not a or not isinstance(a.get("slice_auc"), dict):
            return ""
        # s04's across-seed rows are grouped by (cohort, condition, region) only,
        # so one of them can span split families. A spread that mixes a
        # cross-validated pool with a leftover single split is not a measure of
        # seed stochasticity, so it is withheld rather than printed.
        #
        # Same fail-closed rule as `estimate`: the row must SAY which families it
        # covers. `row_fams and ...` printed the spread whenever the row was
        # silent, which is the case where it is least known to be a spread over
        # one experiment.
        row_fams = [str(f) for f in (a.get("split_families") or [])]
        if a.get("mixes_split_families") or (
                family is not None
                and (not row_fams
                     or {_ident(f) for f in row_fams} != {_ident(family)})):
            return (f"not reported: stage 4's across-seed row for this condition covers "
                    f"split families {', '.join(row_fams) or '(unstated)'}, which are "
                    f"different experiments"
                    + (f"; this report headlines {family!r}" if family else ""))
        s = a["slice_auc"]
        sd = _num(s.get("sd"))
        txt = f"{int(s.get('n', 0))} seed(s), mean {_num(s.get('mean')):.3f}"
        if not math.isnan(sd):
            txt += f", SD {sd:.3f}"
        if s.get("reason"):
            txt += f" ({s['reason']})"
        return txt

    # -- comparisons -----------------------------------------------------
    def comparison_levels(self, cohort: str, a: str, b: str,
                          region: str = "full",
                          family: Optional[str] = None) -> Dict[str, dict]:
        """
        Aggregate the per-seed comparison records of `a` vs `b` by level.

        Signs are normalised so `delta` always means (a - b). Aggregation over
        seeds is conservative: the reported Holm p is the WORST (largest) across
        seeds and the reported delta range spans them, because a criterion that
        only holds for the luckiest seed has not been demonstrated.

        `family` scopes the records to one experiment, for the same reason
        `estimate` does: s04 only ever compares two conditions WITHIN a split
        family, so records from another family describe a different test set and
        cannot be pooled with these. With no family requested and records from
        more than one on disk, nothing is returned -- C2 is then MISSING, which
        is the honest reading of "the file describes two experiments and the
        report did not say which one it is claiming".

        The family is necessary and NOT sufficient. Two conditions of one
        cross-validated family can still have been pooled over different FOLD
        SETS -- a fold that died for magnitude and not for phase -- and s04 says
        so on each comparison record, in `folds_a` / `folds_b`. Those two fields
        were being carried through the file and read by nobody. They are surfaced
        here (`folds_a`, `folds_b`, `declares_folds`) so the caller can check
        them against the folds it is actually headlining; `declares_folds` is
        False when any record in the group is silent, because a record that does
        not say which folds it covers cannot demonstrate it covers these.
        """
        want = {_ident(a), _ident(b)}
        rows = [c for c in self.comparisons
                if _ident(c.get("cohort")) == _ident(cohort)
                and _ident(c.get("region", "full")) == _ident(region)
                and {_ident(c.get("model_a")), _ident(c.get("model_b"))} == want]
        fams = sorted({self.family_of(c) for c in rows})
        if family is not None:
            rows = [c for c in rows if _ident(self.family_of(c)) == _ident(family)]
        elif len(fams) > 1:
            logger.warning("%s: %s-vs-%s comparisons span split families %s; refusing "
                           "to aggregate across them", cohort, a, b, fams)
            return {}
        by_level: Dict[str, List[dict]] = {}
        for r in rows:
            by_level.setdefault(str(r.get("level", "slice")), []).append(r)

        def _folds_of(rec: dict, side: str) -> Optional[Tuple[int, ...]]:
            """The fold set one comparison record declares for one side."""
            v = rec.get(f"folds_{side}")
            if v is None:
                return None
            try:
                return tuple(sorted(int(f) for f in v))
            except (TypeError, ValueError):          # pragma: no cover - defensive
                return None

        out: Dict[str, dict] = {}
        for level, group in by_level.items():
            deltas, p_holms, p_raws, reasons = [], [], [], []
            for c in group:
                dl = c.get("delong") or {}
                d = _num(dl.get("diff"))
                if math.isnan(d):
                    d = _num(dl.get("auc_a")) - _num(dl.get("auc_b"))
                if _ident(c.get("model_a")) != _ident(a):
                    d = -d
                if not math.isnan(d):
                    deltas.append(d)
                ph, pr = _num(c.get("p_holm")), _num(c.get("p_raw"))
                if not math.isnan(ph):
                    p_holms.append(ph)
                if not math.isnan(pr):
                    p_raws.append(pr)
                if dl.get("reason"):
                    reasons.append(str(dl["reason"]))
            # `model_a` is not necessarily `a`: the sign of every delta above is
            # normalised, and so is the side each fold set belongs to.
            fa = {(_folds_of(c, "a") if _ident(c.get("model_a")) == _ident(a)
                   else _folds_of(c, "b")) for c in group}
            fb = {(_folds_of(c, "b") if _ident(c.get("model_a")) == _ident(a)
                   else _folds_of(c, "a")) for c in group}
            # A LEVEL IS A GROUP OF PER-SEED RECORDS, AND ITS PROVENANCE IS
            # THEIRS. `n_clusters` and `n_cases` used to be `group[0]`'s -- one
            # arbitrary seed's -- so a group in which one record described a
            # different test set reported the first record's, and the caller
            # checked that. Each side of the comparison gets its own consensus,
            # because a comparison has two sides and they can differ.
            def _side(rec: dict, side: str) -> Dict[str, Any]:
                flip = _ident(rec.get("model_a")) != _ident(a)
                want = ("b" if flip else "a") if side == "a" else ("a" if flip else "b")
                return {"split_family": self.family_of(rec),
                        "folds": rec.get(f"folds_{want}"),
                        "n_clusters": _first(rec, "n_clusters"),
                        "n": _first(rec, "n_cases"),
                        "subjects": None}
            fp_a, conf_a = _consensus_fingerprint(
                [_side(c, "a") for c in group],
                f"stage 4's {len(group)} {level}-level {a} comparison record(s)")
            fp_b, conf_b = _consensus_fingerprint(
                [_side(c, "b") for c in group],
                f"stage 4's {len(group)} {level}-level {b} comparison record(s)")
            out[level] = {
                "level": level,
                "n_seeds": len(group),
                "n_evaluable": len(p_holms),
                # sorted() over a set containing None is not orderable, so the
                # None (= "this record did not say") is carried separately.
                "folds_a": sorted(f for f in fa if f is not None),
                "folds_b": sorted(f for f in fb if f is not None),
                "declares_folds": (None not in fa) and (None not in fb),
                "preferred": any(bool(c.get("preferred")) for c in group),
                "delta_mean": float(np.mean(deltas)) if deltas else float("nan"),
                "delta_min": min(deltas) if deltas else float("nan"),
                "delta_max": max(deltas) if deltas else float("nan"),
                "p_holm_worst": max(p_holms) if p_holms else float("nan"),
                "p_raw_worst": max(p_raws) if p_raws else float("nan"),
                # None where the records disagree: the group then has no sample
                # size, which is what "no fingerprint at all" means here.
                "n_clusters": fp_a.get("n_clusters"),
                "n_cases": fp_a.get("n"),
                "fingerprint_a": dict(fp_a),
                "fingerprint_b": dict(fp_b),
                "provenance_conflicts": list(conf_a) + list(conf_b),
                "caveat": (group[0].get("delong") or {}).get("caveat", ""),
                "reasons": sorted(set(reasons)),
                "holm_family": group[0].get("holm_family"),
            }
        return out


# --------------------------------------------------------------------------
# Stage-5 controls: a directory of per-run JSONs, not an aggregate file
# --------------------------------------------------------------------------

def _control_subject_ids(payload: dict) -> Optional[List[str]]:
    """
    The subjects one control run was scored on, where the payload says.

    s05 writes them twice -- `test.patient_ids`, one per slice, and
    `control_detail.test_auc_ci95.cluster_ids`, the vector the cluster bootstrap
    resampled. Either identifies the test set far more precisely than a count
    does: two folds of a five-fold split can hold 14 subjects each and share
    none of them.
    """
    for block, key in ((payload.get("test"), "patient_ids"),
                       ((payload.get("control_detail") or {}).get("test_auc_ci95"),
                        "cluster_ids")):
        if isinstance(block, dict):
            v = block.get(key)
            if isinstance(v, (list, tuple)) and v:
                return [str(x) for x in v]
    return None


def _control_cluster_ids(payload: dict,
                         cluster_map: Optional[Dict[int, str]]) -> Optional[List[str]]:
    """
    The control's test subjects IN THE HEADLINE'S NAMESPACE, or None.

    `_control_subject_ids` returns whatever identifier the payload happens to
    carry, which for every stage-5 tree on disk is `test.patient_ids` -- the raw
    per-slice patient id. The headline is clustered on `subject_id` (the stage-1
    split-enforcement unit; breast has two acquisitions per woman and repeated-
    scan groups that link different coded names), so the two are DIFFERENT
    IDENTIFIERS FOR THE SAME PEOPLE. Comparing one against the other would
    report a mismatch for every control ever written, which is not a check, it
    is noise.

    So the subject set is declared in the headline's namespace whenever it can
    be: the control's own `test.cache_idx` mapped through the SAME cache-row ->
    subject_id table `pool_runs` uses for the headline. Where no such table was
    resolved (no stage-2 cache index), the payload's own ids are returned and
    the headline declares no subject set at all, so the component simply adds no
    constraint -- exactly the state every component was in before.

    Returns None when a table exists but does not cover every row this control
    was scored on: a partially-mapped set is not this control's subject set, and
    "unknown" is the honest declaration.
    """
    if cluster_map:
        idx = (payload.get("test") or {}).get("cache_idx")
        if isinstance(idx, (list, tuple)) and idx:
            try:
                mapped = [cluster_map.get(int(k)) for k in idx]
            except (TypeError, ValueError):
                mapped = [None]
            if mapped and all(m is not None for m in mapped):
                return sorted({str(m) for m in mapped})
            return None
    return _control_subject_ids(payload)


def _control_test_set(payload: dict,
                      cluster_map: Optional[Dict[int, str]] = None) -> Dict[str, Any]:
    """
    What ONE stage-5 payload declares about the test set it was scored on.

    The unit `_consensus_fingerprint` works in. `split_family` and `folds` are
    read from the payload where stage 5 writes them -- and, failing that, from
    the fold identity `Controls` parses off the results subdirectory the run was
    found in (`<cohort>_cv<k>`), which is the same and only place `load_runs`
    reads the fold of a stage-3 run from. A payload that declares neither simply
    adds no constraint; a component declared by SOME members of a pool and not
    others is a conflict, which is what makes this safe.
    """
    ci = ((payload.get("control_detail") or {}).get("test_auc_ci95")) or {}
    folds = payload.get("folds")
    if folds is None and payload.get("_fold") is not None:
        folds = [int(payload["_fold"])]
    return {"split_family": payload.get("split_family") or payload.get("_split_family"),
            "folds": folds,
            "n_clusters": _first(ci, "n_clusters"),
            "n": _first(ci, "n"),
            "n_pos": _first(ci, "n_pos"),
            "subjects": _control_cluster_ids(payload, cluster_map)}


def _pooled_test_set(p: "PooledPredictions") -> Dict[str, Any]:
    """
    The same declaration, for an out-of-fold pool of control runs.

    Built from the pooled vector itself rather than from any one member, so the
    fingerprint an out-of-fold control carries is a property of the predictions
    that were actually scored -- and is stated in exactly the terms the headline
    states its own, which is what makes the two differenceable.
    """
    return {"split_family": p.split_family,
            "folds": list(p.folds),
            "n_clusters": p.n_clusters,
            "n": p.n,
            "n_pos": p.n_pos,
            "subjects": sorted({str(c) for c in np.unique(p.clusters).tolist()})}


def _control_estimate(payload: dict,
                      cluster_map: Optional[Dict[int, str]] = None) -> Estimate:
    """Estimate from one s05 run payload's patient-clustered bootstrap block."""
    det = payload.get("control_detail") or {}
    ci = det.get("test_auc_ci95") or {}
    est = Estimate(
        point=_num(ci.get("auc")), lo=_num(ci.get("lo")), hi=_num(ci.get("hi")),
        n=_first(ci, "n"), n_pos=_first(ci, "n_pos"), n_clusters=_first(ci, "n_clusters"),
        n_boot_valid=_first(ci, "n_boot_ok"), source="s05",
    )
    if not est.ok:
        est.point = _num((payload.get("test") or {}).get("auc"))
        est.source = "s05 (point only; no bootstrap block)"
    est.detail = {
        "variant": det.get("variant"),
        # s05 writes the split direction in two places (the filename variant and
        # `control_detail.direction`). Both are carried so that C5 can decide
        # which DIRECTION a run is, rather than counting distinct variant
        # STRINGS -- two strings naming the same direction are not two
        # directions.
        "direction": det.get("direction"),
        "strat_key": det.get("strat_key"),
        "label_semantics": det.get("label_semantics", "diagnosis"),
        "condition": payload.get("condition"),
        "seed": payload.get("seed"),
        "n_boot_degenerate": ci.get("n_boot_degenerate"),
        "path": payload.get("_path"),
        # This run's OWN declaration of the test set it was scored on. Every
        # aggregate built out of these estimates reduces them with
        # `_consensus_fingerprint` rather than picking one.
        "test_set": _control_test_set(payload, cluster_map),
        "fold": payload.get("_fold"),
    }
    return est


def _control_power_reasons(est: Estimate) -> List[str]:
    """
    Why this control estimate cannot decide a criterion. Empty list = it can.

    A criterion that asks whether an interval CONTAINS chance is satisfied by
    any interval wide enough, so every such criterion is gated on the interval
    being able to discriminate in the first place. This is the C1 power floor
    generalised: C1 refuses to be read on fewer than MIN_CLUSTERS_C1 subjects,
    and so, now, does every control-based criterion.

    Callers use this ONLY to withhold a PASS. It never manufactures a FAIL: an
    interval nobody can read is a measurement that was not made, and that is
    MISSING. The fail paths that do not depend on the interval (the control sits
    too close to the headline; the control reports no interval at all) are
    decided before this is consulted.
    """
    why: List[str] = []
    if not est.has_ci:
        return ["it reports no bootstrap interval at all"]
    if not est.ci_finite:
        # Nothing below is meaningful once a bound is +/-inf, so stop here.
        return [f"its interval [{est.lo}, {est.hi}] is not finite, so it contains "
                f"chance no matter what the control did"]
    if est.hi < est.lo:
        return [f"its interval [{est.lo:.3f}, {est.hi:.3f}] is inverted "
                f"(upper bound below lower bound), so it is not an interval"]
    width = est.ci_width
    if width < MIN_CONTROL_CI_WIDTH:
        why.append(f"its 95% CI is degenerate (width {width:.3g}): every bootstrap "
                   f"resample returned the same AUC, so the interval records a tie "
                   f"rather than a spread")
    elif width > MAX_CONTROL_CI_WIDTH:
        why.append(f"its 95% CI spans {width:.3f} AUC "
                   f"(> {MAX_CONTROL_CI_WIDTH:.2f}), which is wide enough to cover "
                   f"both chance and the headline at once -- it cannot distinguish "
                   f"a control that collapsed from one that did not")
    conflicts = (est.detail or {}).get("fingerprint_conflicts") or []
    if conflicts:
        # An aggregate whose members do not agree on one test set. Named before
        # the size checks below, which would otherwise report the CLEARED count
        # as "stage 5 recorded no cluster count" -- true of the pool, and
        # misleading about the runs, every one of which recorded one.
        why.append("the runs pooled into it were not scored on one test set, so it "
                   "is not an estimate of a single quantity: " + "; ".join(conflicts))
    elif est.n_clusters is None:
        why.append("stage 5 recorded no cluster count for it, so the interval "
                   "cannot be checked against the minimum fold size "
                   f"({MIN_CLUSTERS_CONTROL} independent subjects)")
    elif int(est.n_clusters) < MIN_CLUSTERS_CONTROL:
        why.append(f"it was scored on {int(est.n_clusters)} independent cluster(s), "
                   f"below the {MIN_CLUSTERS_CONTROL} a cluster bootstrap needs "
                   f"before its interval means anything")
    if est.n_boot_valid is None:
        why.append("it carries no record of how many bootstrap resamples were "
                   "valid, so the interval cannot be audited")
    elif int(est.n_boot_valid) < MIN_BOOT_VALID_CONTROL:
        why.append(f"only {int(est.n_boot_valid)} bootstrap resample(s) produced a "
                   f"defined AUC (< {MIN_BOOT_VALID_CONTROL}); the percentiles are "
                   f"taken over almost nothing")
    why.extend(_envelope_defects(est))
    return why


def _envelope_defects(est: Estimate) -> List[str]:
    """Reasons an envelope's hull is not an interval for a single quantity."""
    d = est.detail or {}
    if not d.get("is_envelope"):
        return []
    out: List[str] = []
    n_mem = int(d.get("n_members") or 0)
    if d.get("members_without_ci"):
        out.append(f"{d['members_without_ci']} of the {n_mem} run(s) enveloped here "
                   f"reported no interval, so the hull is not a bound on all of them")
    spread = d.get("point_spread")
    if spread is not None and float(spread) > MAX_ENVELOPE_SPREAD:
        out.append(f"the {n_mem} run(s) enveloped here disagree by {float(spread):.3f} "
                   f"AUC (> {MAX_ENVELOPE_SPREAD:.2f}); they are not repeats of one "
                   f"quantity, so the min-lo/max-hi hull over them is not an interval "
                   f"for anything -- points {d.get('member_points')}")
    if n_mem > 1:
        # With a single member the envelope IS that member, and its defects are
        # already reported directly against the envelope's own bounds.
        for reason in (d.get("member_defects") or []):
            out.append("one of the runs enveloped here is itself unreadable: " + reason)
    return out


def _envelope(estimates: Sequence[Estimate], source: str, note: str) -> Estimate:
    """
    Mean point, widest interval, thinnest bootstrap -- over runs of ONE quantity.

    The hull is only conservative when its members are repeats differing by
    training seed, which is the case it was written for. It is not conservative
    when they are not: min(lo)/max(hi) over heterogeneous estimates launders one
    junk member into the whole group. Measured (`self_test`, A1_envelope_launder)
    -- a background-only control at AUC 0.790 with a tight interval [0.660,
    0.920], enveloped with one extra run at 0.790 whose bootstrap returned
    [0.000, 1.000], produced the hull [0.000, 0.920]. That hull contains 0.500,
    so C4 recorded the scanner fingerprint as a PASSED falsification control.

    The hull is still computed and still displayed -- suppressing it would hide
    the disagreement rather than report it -- but the envelope now carries, in
    `detail`, everything a criterion needs to refuse to be decided on it:
    how many runs went in, how far apart their point estimates are, how many
    arrived without an interval, and which of them individually fail the power
    floor. `_control_power_reasons` reads those and withholds the PASS.

    ... and it does NOT report a test set it cannot show all of its members
    share. `n = good[0].n, n_pos = good[0].n_pos` took the FIRST member's slice
    count, and `n_clusters = min(clusters)` took the smallest, so an honest
    background control on the pooled 560-slice / 70-subject out-of-fold set,
    enveloped with a second run scored on 40 slices of 200 other subjects,
    produced a hull reporting 560 slices / 70 subjects. `_test_set_mismatch` then
    confirmed the hull was on the headline's own test set and C4 and C7 PASSED
    (`AG1`); with the second run on the same 70 subjects but a different slice
    selection, `min()` over cluster counts could not see it at all (`AG2`). The
    fingerprint now comes from `_agree_or_forget`: the value every member
    declares, or nothing.
    """
    good = [e for e in estimates if e.ok]
    if not good:
        return Estimate(source=source + "-missing")
    pts = [e.point for e in good]
    los = [e.lo for e in good if not math.isnan(e.lo)]
    his = [e.hi for e in good if not math.isnan(e.hi)]
    boots = [e.n_boot_valid for e in good if e.n_boot_valid is not None]
    member_defects: List[str] = []
    for e in good:
        for r in _control_power_reasons(e):
            if r not in member_defects:
                member_defects.append(r)
    out = Estimate(
        point=float(np.mean(pts)),
        lo=min(los) if los else float("nan"),
        hi=max(his) if his else float("nan"),
        # `n_boot_valid` stays the THINNEST member's: it is a statement about how
        # well the hull's own bounds are resolved, not about which subjects were
        # scored, and the hull does inherit the noise of its worst constituent.
        n_boot_valid=min(boots) if boots else None,
        source=source,
        per_seed=[round(p, 4) for p in pts],
    )
    out.note = note
    out.detail = {
        "is_envelope": True,
        "n_members": len(good),
        "member_points": [round(p, 4) for p in pts],
        "point_spread": float(max(pts) - min(pts)),
        "members_without_ci": sum(1 for e in good if not e.has_ci),
        "member_defects": member_defects,
        "members": [dict(e.detail or {}) for e in good],
    }
    _agree_or_forget(out, [_member_test_set(e) for e in good],
                     f"the {len(good)} {source} run(s) enveloped here")
    return out


# Fields that legitimately differ between two records of the SAME computation,
# or that record where a file happens to sit. Excluded from the content
# fingerprint below so that a replicate copied to a second filename, or re-saved
# with a bumped seed, is still recognised as the one replicate it is.
_REPLICATE_VOLATILE_KEYS = frozenset({
    "_path", "_canonical", "timestamp", "wall_seconds", "checkpoint", "seed",
    "device", "best_selection_score",
    # Everything s06 itself derives from WHERE the file sits: the search
    # directory it was reached through, and the fold/family parsed off its
    # parent directory name. Including any of them would make one replicate
    # copied into a second directory look like two draws from the null, which is
    # exactly what this hash exists to prevent. The identity of a replicate is
    # decided by `(permutation index, fold)` and by its predictions, in
    # `permutation_null`, where a fold difference IS counted.
    "_control_root", "_fold", "_split_family",
})
_REPLICATE_VOLATILE_DETAIL_KEYS = frozenset({
    "permutation_seed", "wall_seconds", "variant",
})


def _replicate_fingerprint(payload: dict) -> int:
    """
    Content hash of a control run, ignoring where it was stored and its seed.

    Two permutation replicates are two DRAWS FROM THE NULL. Two files holding
    the same predictions are one draw, however they are named and whatever seed
    field they carry: a permutation that produced identical per-slice
    probabilities to another is the same permutation, not an independent
    replicate of it.
    """
    body = {k: v for k, v in payload.items() if k not in _REPLICATE_VOLATILE_KEYS}
    det = dict(body.get("control_detail") or {})
    for k in _REPLICATE_VOLATILE_DETAIL_KEYS:
        det.pop(k, None)
    body["control_detail"] = det
    try:
        blob = json.dumps(body, sort_keys=True, default=str)
    except (TypeError, ValueError):     # pragma: no cover - defensive
        blob = repr(sorted(body.items(), key=lambda kv: kv[0]))
    return zlib.crc32(blob.encode())


def _prediction_fingerprint(payload: dict) -> Optional[int]:
    """
    Content hash of the PREDICTIONS one control run produced. None if it has none.

    `_replicate_fingerprint` above hashes the whole payload minus a hand-written
    denylist of volatile keys, which is a guard that fails OPEN: a field the
    denylist has never heard of counts as substance. One permutation replicate
    copied to 20 filenames was correctly collapsed to ONE draw from the null --
    and the same 20 copies with a per-epoch wall clock inside `history` varied
    counted as 20, giving C3 a characterised null and C8 p = 1/21 = 0.048 out of
    a single permutation (`AG5`). The training clock is not a permutation.

    So the count of DISTINCT draws -- which is the whole of what makes C3 and C8
    evaluable -- is established from an allowlist of the only thing that can make
    two replicates two: the per-slice probabilities they produced, with the
    labels and cache rows they were scored against. `permutation_null` treats a
    match on EITHER hash as a duplicate, so this can only ever lower the count,
    which moves C3 and C8 towards MISSING and never away from it. Returning None
    where a payload carries no predictions leaves the existing body hash in sole
    charge, exactly as before.
    """
    test = payload.get("test")
    if not isinstance(test, dict):
        return None
    probs = test.get("probs")
    if not isinstance(probs, (list, tuple)) or not probs:
        return None
    try:
        blob = json.dumps(
            [[float(p) for p in probs],
             [str(v) for v in (test.get("labels") or [])],
             [str(v) for v in (test.get("cache_idx") or [])]],
            sort_keys=True)
    except (TypeError, ValueError):     # pragma: no cover - defensive
        return None
    return zlib.crc32(blob.encode())


def _acq_direction(*candidates: Any) -> Optional[Tuple[str, str]]:
    """
    Parse a split direction into an ORDERED (train_arm, test_arm) pair.

    s05_controls writes `variant = f"{train_arm}2{test_arm}"` -- `A2B`, `B2A` --
    and repeats it in `control_detail.direction`. C5 is defined as the WORSE of
    the two directions of one bipartition, and "both directions are present" was
    being established by counting distinct variant STRINGS. Two strings are not
    two directions: `A2B` and `A_to_B` are the same experiment written twice,
    and a run duplicated under a second variant name satisfied the both-
    directions requirement while only one direction had ever been executed
    (`self_test`, A4_one_direction_twice). Comparing parsed pairs makes the
    check what it claims to be.

    Returns None when no candidate parses, which C5 treats as "the direction of
    this run cannot be established" -- MISSING, not a free second direction.
    """
    pats = (r"^(.+?)_to_(.+?)$", r"^(.+?)_2_(.+?)$", r"^(.+?)_?->_?(.+?)$",
            r"^([a-z0-9]+?)2([a-z0-9]+?)$")
    for raw in candidates:
        if raw is None:
            continue
        t = _ident(raw)
        if not t:
            continue
        for pat in pats:
            m = re.match(pat, t)
            if m:
                a, b = m.group(1).strip("_"), m.group(2).strip("_")
                if a and b:
                    return (a, b)
    return None


def _dir_label(key: Any) -> str:
    """Printable name for an `_acq_direction` key (or an unparsed variant)."""
    if isinstance(key, tuple) and len(key) == 2:
        if key[0] == "?unparsed":
            return f"{key[1]} (direction not identifiable)"
        return f"{key[0]}2{key[1]}"
    return str(key)


# The semantics every criterion in this file except C6 is about. Stage 5 writes
# `confound:<target>` on the confound-predictability runs and `diagnosis` on
# every other control; a payload that declares nothing is read as diagnostic,
# exactly as `_control_estimate` already reads it.
DIAGNOSTIC_SEMANTICS = "diagnosis"


def _control_fold_identity(payload: dict) -> Tuple[Optional[int], Optional[str], List[str]]:
    """
    Which CV fold ONE stage-5 control run belongs to, and in which sweep.

    Returns `(fold, split_family, defects)`; `fold is None` means "this run
    cannot be shown to be a fold of a cross-validation", which is the state
    every non-CV tree is in and which makes every caller below a no-op.

    TWO INDEPENDENT SOURCES, AND THEY MUST AGREE.

      * the results subdirectory, `<cohort>_cv<k>` -- the same and only place
        `load_runs` reads the fold of a stage-3 run from, because s03 and s05
        both name their output with no fold component and five folds written to
        one directory would overwrite each other;
      * the payload's own `control_detail.split_col`, `cv<k>_split`, which is the
        design column stage 5 actually read the split from.

    A directory that says fold 3 holding a run that split on `cv1_split` is a
    mis-filed sweep, and pooling it as fold 3 would put two folds' predictions
    in one bucket. Disagreement therefore yields NO fold at all rather than
    either candidate, which costs the pooling and leaves the per-run estimates
    exactly as they were.

    The family follows `load_runs` verbatim: `cv` for a fold directory sitting at
    the root of the controls tree, `cv@<subdir>` below it. Two five-fold control
    sweeps in two subdirectories are two experiments, and nothing may merge them.
    """
    defects: List[str] = []
    cohort = str(payload.get("cohort", ""))
    path = payload.get("_path")
    dir_fold: Optional[int] = None
    subdir_above = "."
    if path:
        p = Path(str(path))
        m = CV_DIR_RE.match(p.parent.name)
        if m:
            if _ident(m.group("cohort")) != _ident(cohort):
                defects.append(
                    f"{path}: lives in controls subdirectory {p.parent.name!r}, which "
                    f"names cohort {m.group('cohort')!r}, but the payload says cohort "
                    f"{cohort!r}. Not treated as a fold of either cohort.")
            else:
                dir_fold = int(m.group("fold"))
                root = payload.get("_control_root")
                if root:
                    try:
                        rel = p.parent.parent.relative_to(Path(str(root)))
                        subdir_above = "." if str(rel) == "." else str(rel)
                    except ValueError:            # pragma: no cover - defensive
                        subdir_above = str(p.parent.parent)
    col = (payload.get("control_detail") or {}).get("split_col")
    col_fold: Optional[int] = None
    if col is not None:
        mc = CV_SPLIT_COL_RE.match(str(col).strip())
        if mc:
            col_fold = int(mc.group("fold"))
    if dir_fold is not None and col_fold is not None and dir_fold != col_fold:
        defects.append(
            f"{path}: the controls subdirectory says fold {dir_fold} but the payload "
            f"split on {str(col)!r} (fold {col_fold}). The two disagree, so this run "
            f"carries no fold identity and is not pooled out-of-fold.")
        return None, None, defects
    fold = dir_fold if dir_fold is not None else col_fold
    if fold is None:
        return None, None, defects
    family = CV_FAMILY if subdir_above in (".", "") else f"{CV_FAMILY}@{subdir_above}"
    return fold, family, defects


class Controls:
    """
    Read-only view over the stage-5 control tree.

    Stage 5 emits one JSON per control run. This class groups them by
    (cohort, canonical control, condition) and produces one conservative
    estimate per group. Four rules that matter:

    * A control is evidence about the INPUT CHANNEL it was run on. select()
      never widens a condition-specific request to whatever else is on disk: a
      background-only control run on the magnitude model says nothing about
      phase, and crediting it to phase is how the phase falsification controls
      came to be scored without ever having run.
    * The acquisition-stratified control runs in two directions (train on arm A
      test on arm B, and the reverse). The WORSE direction is what the criterion
      sees, and BOTH must be present -- a single surviving direction is not
      "the worse of two", it is the only one that did not break.
    * Any control stage 5 recorded as FAILED in its failure ledger is treated
      as absent, however many of its runs landed on disk. Scoring the surviving
      arm of a half-failed control is scoring a survivorship-biased sample.
    * confound_predictability runs carry `label_semantics = "confound:<target>"`.
      Their AUCs are never mixed with diagnostic AUCs anywhere in this file.
      That invariant is now ENFORCED by select() rather than stated: it filters
      on `label_semantics` alongside cohort/control/condition, so a scanner-
      identity AUC cannot reach C3/C4/C5/C7/C8 however it is filed. Only
      `confound_targets`/`confound_defects` ask for the confound semantics, and
      only they ask for every semantics at once -- because C6 has to SEE a
      mislabelled confound run in order to refuse to pass while it is unscored.

    * A control run per CV fold is FIVE FIFTHS OF ONE MEASUREMENT, not five
      measurements. The headline is a pooled out-of-fold vector -- every subject
      predicted exactly once, one AUC over the whole cohort -- so a control
      reduced to the MEAN OF ITS PER-FOLD AUCs is not the same quantity, and the
      difference C4/C5/C7 take between them is part control effect and part
      aggregation artefact. Where the runs of one control are folds of one
      sweep they are therefore pooled out-of-fold through the SAME code path the
      headline uses (`pool_runs` + `pool_folds_oof`), and the estimate is a
      subject-clustered bootstrap over the pooled vector. Where they are not
      (no fold identity, one fold only, two sweeps, overlapping folds) the
      envelope over runs is kept exactly as before.
    """

    def __init__(self, payloads: Sequence[dict], dirs: Sequence[Path],
                 harvested: Optional[dict] = None,
                 failures: Optional[dict] = None,
                 cluster_maps: Optional[Dict[str, Tuple[Dict[int, str], str]]] = None,
                 cv_expect: Optional[Dict[str, "CVExpectation"]] = None,
                 bootstrap: int = DEFAULT_BOOTSTRAP,
                 seed: int = 0):
        self.dirs = [str(d) for d in dirs]
        self.harvested = harvested or {}
        self.unrecognised: Dict[str, List[str]] = {}
        self.payloads: List[dict] = []
        # The cache-row -> subject_id tables the HEADLINE is clustered on, so a
        # pooled control is clustered on the same unit and declares its subject
        # set in the same namespace. Absent (ad-hoc construction, a tree with no
        # stage-2 cache index) everything below degrades to the per-run
        # behaviour that predates it.
        self.cluster_maps: Dict[str, Tuple[Dict[int, str], str]] = dict(cluster_maps or {})
        self.cv_expect: Dict[str, Any] = dict(cv_expect or {})
        self.bootstrap = int(bootstrap)
        self.seed = int(seed)
        self.fold_defects: List[str] = []
        for pl in payloads:
            name = str(pl.get("control", "none"))
            if name == "none":
                continue          # a re-run headline, not a control
            canon = S05_TO_CANONICAL.get(name)
            if canon is None:
                self.unrecognised.setdefault(str(pl.get("cohort")), [])
                if name not in self.unrecognised[str(pl.get("cohort"))]:
                    self.unrecognised[str(pl.get("cohort"))].append(name)
                continue
            pl = dict(pl)
            pl["_canonical"] = canon
            fold, family, defs = _control_fold_identity(pl)
            pl["_fold"] = fold
            if family is not None:
                pl["_split_family"] = family
            for d in defs:
                if d not in self.fold_defects:
                    self.fold_defects.append(d)
                logger.warning("%s", d)
            self.payloads.append(pl)
        # Loaded here rather than at the call site so that every construction of
        # Controls -- report, self-test, ad-hoc -- reads the same provenance.
        self.failures: Dict[Tuple[str, str], dict] = (
            failures if failures is not None else load_control_failures(dirs))
        self.available = bool(self.payloads) or bool(self.harvested)

    def cohorts(self) -> List[str]:
        return sorted({str(p.get("cohort")) for p in self.payloads})

    def failure(self, cohort: str, canonical: str) -> Optional[dict]:
        """The stage-5 failure record for this control, if it recorded one."""
        return self.failures.get((_ident(cohort), canonical))

    def cluster_map_for(self, cohort: str) -> Tuple[Dict[int, str], str]:
        """The (cache row -> subject) table and unit name this cohort is clustered on."""
        for key, val in self.cluster_maps.items():
            if _ident(key) == _ident(cohort):
                return val
        return {}, "patient_id (from run JSON)"

    def select(self, cohort: str, canonical: str,
               condition: Optional[str] = None,
               semantics: Optional[str] = DIAGNOSTIC_SEMANTICS) -> List[dict]:
        """
        Runs for exactly this (cohort, control, condition, semantics). No widening.

        The previous behaviour fell back to `out` -- every condition -- when the
        requested condition had no runs. That silently credited a control run on
        one input channel to a different one: with the background-only control
        run only on the magnitude model, C4 was scored PASS for phase although
        the phase falsification control had never been executed. An empty list
        here becomes a MISSING criterion downstream, which is the honest answer.

        `semantics` is the same rule applied to the LABEL. The class docstring
        has always claimed that confound-cohort AUCs are never mixed with
        diagnostic ones, and nothing enforced it: a payload carrying
        `label_semantics="confound:receiver_channels"` filed under
        `label_permutation` or `background_only` was selected, averaged into the
        control estimate, and differenced against a cancer AUC. A scanner-
        identity number and a cancer number are not two measurements of one
        quantity, whatever the filename says.

        The default is therefore DIAGNOSTIC_SEMANTICS: only runs whose declared
        semantics is diagnostic (or which declare none, exactly as
        `_control_estimate` reads them) are returned. `semantics=None` asks for
        everything and is used in ONE place -- `_confound_groups`, which has to
        see a mislabelled confound run in order to report it, because C6 is a
        maximum over targets and a dropped target can only lower it.

        Filtering can only REMOVE runs, so every criterion it touches moves
        towards MISSING and none towards PASS.
        """
        out = [p for p in self.payloads
               if p["_canonical"] == canonical
               and _ident(p.get("cohort")) == _ident(cohort)]
        if condition is not None:
            out = [p for p in out if _ident(p.get("condition")) == _ident(condition)]
        if semantics is not None:
            out = [p for p in out
                   if _ident((p.get("control_detail") or {}).get(
                       "label_semantics", DIAGNOSTIC_SEMANTICS)) == _ident(semantics)]
        return out

    def names_present(self, cohort: str) -> List[str]:
        names = sorted({p["_canonical"] for p in self.payloads
                        if _ident(p.get("cohort")) == _ident(cohort)})
        for k in self.harvested.get(cohort, {}):
            if k not in names:
                names.append(k)
        return names

    # -- out-of-fold pooling, the same one the headline gets ---------------
    def pool_out_of_fold(self, cohort: str, runs: Sequence[dict], what: str
                         ) -> Tuple[Optional["PooledPredictions"], List[str]]:
        """
        Concatenate the per-fold TEST blocks of one control into ONE vector.

        The headline is a pooled out-of-fold estimate: every subject predicted
        exactly once, one AUC over the whole cohort. A control that ran per fold
        and is reduced to the mean of its five per-fold AUCs is a different
        estimator of a different quantity, and the difference C4/C5/C7 take
        between the two is part control effect and part aggregation artefact.
        Measured on the real prostate_t2 tree: the background-only control's
        per-fold mean is 0.5956 and its pooled out-of-fold AUC is 0.6038, so the
        gap C4 reports moves 0.033 -> 0.025 on the same five files.

        Reuses `pool_runs` + `pool_folds_oof` rather than reimplementing them, so
        a control pool inherits every refusal the headline's pooling makes:
        folds that share a subject or a cache row (which would double-count and
        narrow the interval), folds clustered on different units, two sweeps in
        two subdirectories, and the coverage check against the declared
        `cv<k>_split` design.

        Returns `(pooled or None, defects)`. None means "these runs are not the
        folds of one sweep" -- no fold identity, one fold, more than one family,
        or a refusal -- and every caller then keeps the per-run envelope it
        always built. Pooling is never forced: an unpoolable control is reported
        exactly as before, not dropped.
        """
        defects: List[str] = []
        fams: Dict[str, Dict[int, List[dict]]] = {}
        n_unfolded = 0
        for p in runs:
            fold = p.get("_fold")
            if fold is None:
                n_unfolded += 1
                continue
            fams.setdefault(str(p.get("_split_family", CV_FAMILY)), {}) \
                .setdefault(int(fold), []).append(p)
        if not fams:
            return None, defects
        if n_unfolded:
            defects.append(
                f"{n_unfolded} of the {len(runs)} {what} run(s) carry no fold identity "
                f"while the others do, so they are not the folds of one sweep; pooled "
                f"out-of-fold estimation is refused and the per-run envelope is used")
            return None, defects
        if len(fams) > 1:
            defects.append(
                f"the {what} runs are folds of {len(fams)} different sweeps "
                f"({', '.join(sorted(fams))}); those are different experiments on "
                f"different test sets, so they are not pooled")
            return None, defects
        by_fold = next(iter(fams.values()))
        if len(by_fold) < 2:
            return None, defects           # one fold is not an out-of-fold pool
        cmap, unit = self.cluster_map_for(cohort)
        per_fold: Dict[int, PooledPredictions] = {}
        for f, rs in sorted(by_fold.items()):
            pf = pool_runs(rs, cmap, unit)
            if pf is not None:
                per_fold[f] = pf
        if len(per_fold) < 2:
            return None, defects
        pooled, defs = pool_folds_oof(per_fold, expected=self.cv_expect.get(cohort))
        defects.extend(f"{what}: {d}" for d in defs)
        return pooled, defects

    def pooled_estimate(self, pooled: "PooledPredictions", source: str, note: str
                        ) -> Estimate:
        """
        A subject-clustered bootstrap over a pooled out-of-fold control vector.

        The same bootstrap the headline's own fallback estimate uses, on the same
        clustering unit, so the two numbers are comparable by construction. The
        test-set fingerprint is written by `_agree_or_forget` like every other
        aggregate in this file -- never assembled by hand -- so it is checked
        against the pool's members at the moment it is built.
        """
        bs = cluster_bootstrap(pooled.labels, pooled.probs, pooled.clusters,
                               self.bootstrap, self.seed)
        est = Estimate(point=bs["point"], lo=bs["lo"], hi=bs["hi"],
                       n_boot_valid=bs["n_boot_valid"], source=source,
                       per_seed=[round(r["auc"], 4) for r in pooled.per_fold
                                 if not math.isnan(_num(r.get("auc")))])
        est.note = note
        est.detail = {"is_oof_pool": True, "folds": list(pooled.folds),
                      "scheme": pooled.scheme,
                      "coverage_defect": pooled.coverage_defect,
                      "per_fold": list(pooled.per_fold)}
        _agree_or_forget(est, [_pooled_test_set(pooled)],
                         f"the {len(pooled.folds)}-fold out-of-fold pool of {source}")
        return est

    def group_estimate(self, cohort: str, runs: Sequence[dict], source: str,
                       what: str) -> Estimate:
        """
        ONE estimate for a group of control runs: pooled out-of-fold if they are
        the folds of one sweep, the envelope over runs otherwise.

        This is the single place the choice is made, so background, scramble,
        every acquisition DIRECTION and every future control get the same
        treatment and none of them can drift into differencing a per-fold mean
        against a pooled headline.
        """
        cmap, _unit = self.cluster_map_for(cohort)
        pooled, defs = self.pool_out_of_fold(cohort, runs, what)
        if pooled is not None:
            est = self.pooled_estimate(
                pooled, source,
                f"pooled out-of-fold over {len(pooled.folds)} fold(s) of "
                f"{len(runs)} run(s), the same pooling the headline uses; the "
                f"interval is a subject-clustered bootstrap over the pooled vector")
            if pooled.coverage_defect:
                est.note += ("; the pool does NOT cover the cross-validation design ("
                             + pooled.coverage_defect + ")")
            for d in defs:
                est.note += "; " + d
            return est
        est = _envelope([_control_estimate(p, cmap) for p in runs], source,
                        f"envelope over {len(runs)} run(s)")
        for d in defs:
            est.note += "; " + d
        return est

    # -- individual controls ---------------------------------------------
    def permutation_null(self, cohort: str, condition: str = HEADLINE_CONDITION) -> dict:
        """
        The null distribution of test AUC over DISTINCT permutation replicates.

        Both criteria that read this are counts-driven: C3 needs
        MIN_PERMUTATION_REPLICATES before the null is characterised at all, and
        C8's smallest reachable p-value is 1/(n+1), so n is what decides whether
        C8 can be evaluated. `load_control_payloads` de-duplicates by resolved
        path, which stops the same FILE being counted twice through two search
        directories but does nothing about the same REPLICATE stored under two
        names. Copying one replicate to 20 filenames therefore turned an
        unevaluable C8 into p = 1/21 = 0.048 < 0.05, i.e. a PASS manufactured
        out of a single permutation (`self_test`, A3_dup_perm_reps).

        Duplicates are counted and reported rather than silently dropped: a
        controls tree that contains them is a tree whose provenance the reader
        needs to know about.

        The seed check is the other half of that, and it FAILS CLOSED for the
        same reason C8's fold check now does. `pseed is not None and pseed in
        seen_perm_seed` only compared seeds where the payload happened to carry
        one, so a file that declared no `permutation_seed` was counted as a
        distinct draw from the null on the strength of its predictions being
        byte-different -- which two runs of the SAME label permutation with two
        training seeds also are. A replicate that cannot say which permutation
        it is cannot be shown to be a second one, so it is counted separately
        (`n_unidentified`) and reported, exactly like a duplicate. Every stage-5
        payload on the real tree carries the field; this only bites a tree whose
        provenance is already broken, and it can only LOWER n, which moves C3 and
        C8 towards MISSING.

        A REPLICATE IS A PERMUTATION, NOT A PERMUTATION-AND-A-FOLD -- and the
        de-duplication key has to say so. `pseed in seen_perm_seed` treated the
        permutation index ALONE as the identity of a replicate, and stage 5
        assigns `permutation_seed = 1000 + i` by replicate INDEX, independent of
        which cross-validation fold the run belongs to. On the real tree that is
        100 genuinely distinct payloads per clinical cohort -- 20 indices x 5
        folds, five disjoint subject sets, 100 distinct probability vectors --
        of which this method kept the 20 that `sorted(rglob)` happened to reach
        first (all of fold 0) and reported the other 80 as duplicates. Measured
        on prostate_t2: the reported null was cv0 alone at mean 0.622 [0.503,
        0.719], the five per-fold means are 0.622 / 0.484 / 0.695 / 0.605 /
        0.678, and had cv1 sorted first C3 would have PASSED instead of FAILED.
        A criterion outcome decided by directory sort order is not a criterion.

        So the folds of one permutation index are POOLED OUT-OF-FOLD FIRST --
        through `pool_runs` + `pool_folds_oof`, the same code path the headline
        uses -- and each index contributes ONE replicate: an out-of-fold AUC over
        the whole cohort, on the headline's own test set, which is what makes the
        null the null distribution OF THE HEADLINE rather than of a fifth of it.
        On prostate_t2 that null is mean 0.595, range [0.548, 0.645] over 2039
        slices / 67 subjects, and it is what C3 and C8 now read.

        Where an index has runs in only ONE fold bucket -- every tree without a
        `<cohort>_cv<k>` layout, including every synthetic tree in `self_test` --
        there is nothing to pool and the run's own `test.auc` and fingerprint are
        used exactly as before.

        Duplicates are still caught, and the key is now the whole identity: the
        same index AND the same fold, or byte-identical predictions, or an
        identical payload body. Two files that differ only in which fold they
        scored are two thirds of nothing in common and are never collapsed.
        """
        runs = self.select(cohort, "permutation", condition)
        cmap, _unit = self.cluster_map_for(cohort)
        # (permutation index, fold) -> the runs claiming to be that experiment.
        by_key: Dict[Tuple[Any, Any], List[dict]] = {}
        order: List[Tuple[Any, Any]] = []
        seen_content: set = set()
        seen_predictions: set = set()
        n_dupes = 0
        n_unidentified = 0
        for p in runs:
            if math.isnan(_num((p.get("test") or {}).get("auc"))):
                continue
            pseed = (p.get("control_detail") or {}).get("permutation_seed")
            if pseed is None:
                n_unidentified += 1
                continue
            fp = _replicate_fingerprint(p)
            pfp = _prediction_fingerprint(p)
            key = (pseed, p.get("_fold"))
            if (key in by_key or fp in seen_content
                    or (pfp is not None and pfp in seen_predictions)):
                n_dupes += 1
                continue
            seen_content.add(fp)
            if pfp is not None:
                seen_predictions.add(pfp)
            by_key[key] = [p]
            order.append(key)

        by_index: Dict[Any, List[dict]] = {}
        index_order: List[Any] = []
        for key in order:
            idx = key[0]
            if idx not in by_index:
                by_index[idx] = []
                index_order.append(idx)
            by_index[idx].extend(by_key[key])

        aucs: List[float] = []
        members: List[Dict[str, Any]] = []
        pool_defects: List[str] = []
        n_pooled = 0
        for idx in index_order:
            group = by_index[idx]
            pooled, defs = self.pool_out_of_fold(
                cohort, group, f"label-permutation replicate {idx}")
            for d in defs:
                if d not in pool_defects:
                    pool_defects.append(d)
            if pooled is not None:
                a = _safe_auc(pooled.labels, pooled.probs)
                if math.isnan(a):
                    continue
                aucs.append(float(a))
                members.append(_pooled_test_set(pooled))
                n_pooled += 1
                continue
            if len(group) > 1:
                # More than one fold's worth of files for this index and no
                # pooled vector to show for it. Counting them as separate draws
                # from the null would be the defect this method exists to close,
                # and picking one would be the sort-order dependency it exists to
                # close; the index contributes nothing.
                if not defs:
                    pool_defects.append(
                        f"label-permutation replicate {idx} has {len(group)} run(s) "
                        f"that could not be pooled into one out-of-fold vector; it is "
                        f"not counted as a draw from the null")
                continue
            p = group[0]
            aucs.append(_num((p.get("test") or {}).get("auc")))
            members.append(_control_test_set(p, cmap))
        if not aucs:
            return {}
        sd = float(np.std(aucs, ddof=1)) if len(aucs) > 1 else float("nan")
        return {"aucs": aucs, "n": len(aucs), "mean": float(np.mean(aucs)),
                "sd": sd, "min": min(aucs), "max": max(aucs),
                "n_files": len(runs), "n_duplicates": n_dupes,
                "n_unidentified": n_unidentified, "condition": condition,
                "n_pooled": n_pooled, "pool_defects": pool_defects,
                # The replicates that were actually counted. The pool's test-set
                # fingerprint is the consensus of THESE, not of whichever one
                # `runs[0]` happens to be; see `Controls.estimate`.
                "members": members}

    def _failed_estimate(self, cohort: str, canonical: str) -> Optional[Estimate]:
        """
        A control stage 5 recorded as failed yields no estimate at all.

        Checked before the runs and before the harvested fallback, because the
        failure mode being closed here is precisely "some runs did land on
        disk". s05's _guarded() swallows an exception so one broken control
        cannot abort the other four; that is right, but it means a partial
        control looks exactly like a complete one from the file tree alone. The
        ledger is the only place the difference is recorded, so it outranks the
        files. NaN point => the criterion reports MISSING, never PASS.
        """
        rec = self.failure(cohort, canonical)
        if rec is None:
            return None
        errs = [str(e.get("error", "")) for e in (rec.get("errors") or [])]
        est = Estimate(source="s05-failed")
        est.note = (f"stage 5 recorded {len(errs) or 1} failure(s) running this control"
                    + (": " + "; ".join(errs[:3]) if errs else "")
                    + " -- reported MISSING rather than scored on whatever runs "
                      "survived, which would be a survivorship-biased sample")
        est.detail = {"s05_failure": rec}
        return est

    def estimate(self, cohort: str, canonical: str,
                 condition: Optional[str] = HEADLINE_CONDITION) -> Estimate:
        """
        One conservative estimate for a control.

        failed      -> nothing at all, however many runs landed on disk
        (see _failed_estimate).
        permutation -> the null distribution's mean, with a range rather than a
        bootstrap interval (each replicate is a draw from the null).
        acquisition -> the worse of the two split directions, and only if BOTH
        directions are there.
        others      -> envelope over seeds.
        """
        if canonical == "confound":
            raise ValueError("use confound_targets() for confound predictability")

        failed = self._failed_estimate(cohort, canonical)
        if failed is not None:
            return failed

        runs = self.select(cohort, canonical, condition)
        if not runs:
            blk = self.harvested.get(cohort, {}).get(canonical)
            # The harvested fallback is condition-scoped for the same reason
            # select() is: a background-region run on one input channel is not
            # evidence about another. harvest_controls_from_runs only keeps
            # HEADLINE_CONDITION today, so this is belt-and-braces -- but it is
            # the exact substitution that produced a PASS on a control that had
            # never been run on the channel being reported.
            if blk and (condition is None
                        or _slug(blk.get("condition")) == _slug(condition)):
                e = _as_estimate(blk.get("auc"), source=str(blk.get("source", "s06-fallback")))
                e.note = "reconstructed by s06 from stage-3 region runs"
                return e
            return Estimate(source="s05-missing")

        if canonical == "permutation":
            null = self.permutation_null(cohort, condition or HEADLINE_CONDITION)
            if not null:
                est = Estimate(source="s05-missing")
                if runs:
                    # Files exist but none of them could be counted as a draw
                    # from the null. Said out loud, because "stage 5 has not
                    # produced one" would be a different and untrue statement.
                    est.note = (f"{len(runs)} label-permutation file(s) are on disk but "
                                f"none could be counted as a distinct replicate: a "
                                f"payload that declares no `permutation_seed` cannot be "
                                f"shown to be a second draw from the null")
                return est
            est = Estimate(point=null["mean"], lo=null["min"], hi=null["max"],
                           source="s05 permutation null",
                           per_seed=[round(v, 4) for v in null["aucs"]])
            # THE NULL'S TEST SET IS THE ONE ALL ITS REPLICATES SHARE.
            # `base = _control_estimate(runs[0])` handed the pool whatever
            # fingerprint the first file on disk carried: 19 replicates scored on
            # one 4-subject fold plus one scored on the headline's own 70-subject
            # out-of-fold set reported 560 slices / 70 subjects, C8's fold check
            # was satisfied by it, and p = 1/21 passed on a null measured almost
            # entirely somewhere else (`AG3`).
            perm_conflicts = _agree_or_forget(
                est, null.get("members") or [],
                f"the {null['n']} label-permutation replicate(s) pooled into this null")
            est.note = (f"{null['n']} distinct permutation replicate(s); the interval "
                        f"shown is the observed range of the null, not a bootstrap CI"
                        + (f"; {null['n_duplicates']} duplicate file(s) of an existing "
                           f"replicate were not counted"
                           if null.get("n_duplicates") else "")
                        + (f"; {null['n_unidentified']} file(s) declare no "
                           f"`permutation_seed` and were not counted as replicates"
                           if null.get("n_unidentified") else "")
                        + (f"; each replicate is one permutation index pooled "
                           f"out-of-fold across its folds, exactly as the headline is "
                           f"pooled ({null['n_pooled']} of {null['n']} were pooled "
                           f"this way)" if null.get("n_pooled") else "")
                        + ("; " + "; ".join(null["pool_defects"])
                           if null.get("pool_defects") else ""))
            if perm_conflicts:
                est.note += ("; the replicates were NOT all scored on one test set ("
                             + "; ".join(perm_conflicts)
                             + "), so this pool is not the null distribution of any "
                               "single number and carries no test-set fingerprint")
            # Merged, not replaced: `_agree_or_forget` has already written the
            # consensus fingerprint and the conflicts into `detail`, and dropping
            # them here would be the same borrow by another route.
            est.detail = dict(est.detail or {}, **{
                "n_replicates": null["n"], "sd": null["sd"],
                "n_files": null.get("n_files"),
                "n_duplicates": null.get("n_duplicates", 0),
                "n_unidentified": null.get("n_unidentified", 0),
                "n_pooled": null.get("n_pooled", 0)})
            return est

        cmap, _unit = self.cluster_map_for(cohort)

        if canonical == "acquisition":
            # Group by the PARSED direction, not by the variant string. Runs
            # whose direction cannot be parsed are kept in their own buckets and
            # counted as unidentified: they are shown, but they cannot be the
            # second direction.
            #
            # THE TWO DIRECTIONS ARE NEVER MERGED, and that is not an oversight
            # in the pooling above: A2B is tested on protocol arm B and B2A on
            # protocol arm A, so their test sets are disjoint BY DESIGN and their
            # union is the whole cohort. Concatenating them would manufacture an
            # "out-of-fold" vector out of two different models trained on two
            # different halves -- one number that is an estimate of nothing. The
            # folds WITHIN one direction are pooled, because those are folds of
            # one experiment; the directions are two experiments.
            by_dir: Dict[Any, List[dict]] = {}
            unparsed: List[str] = []
            for p in runs:
                det = p.get("control_detail") or {}
                key = _acq_direction(det.get("direction"), det.get("variant"))
                if key is None:
                    label = str(det.get("variant") or det.get("direction") or "-")
                    if label not in unparsed:
                        unparsed.append(label)
                    key = ("?unparsed", label)
                by_dir.setdefault(key, []).append(p)
            per_dir = {k: self.group_estimate(cohort, ps, "s05",
                                              f"acquisition direction {_dir_label(k)}")
                       for k, ps in by_dir.items()}
            for k, e in per_dir.items():
                e.note = f"direction {_dir_label(k)}" + (f"; {e.note}" if e.note else "")
            usable = {k: e for k, e in per_dir.items() if e.ok}
            if not usable:
                return Estimate(source="s05-missing")
            # BOTH directions or nothing. The control is defined as the worse of
            # train-A/test-B and train-B/test-A; min() over however many happen
            # to be on disk silently redefines it as "the best available", which
            # is the opposite of what the criterion claims to measure. Measured
            # on the real tree: with A2B=0.85 and B2A=0.55 the criterion FAILS;
            # with B2A absent the same code PASSES on 0.85 alone. Two is the
            # threshold because two is how many directions a bipartition has --
            # it is not a tunable knob.
            #
            # ... and the two must be REVERSES of each other. Counting variant
            # strings let one direction stored under two names ("A2B" and
            # "A_to_B") satisfy the requirement while the reverse split had
            # never been run. A bipartition has exactly one reverse, so the
            # check is for the pair {(a,b), (b,a)}, not for a cardinality.
            parsed = [k for k in usable if not (isinstance(k, tuple) and k[0] == "?unparsed")]
            pairs = [(a, b) for (a, b) in parsed if (b, a) in set(parsed)]
            if not pairs:
                est = Estimate(source="s05-incomplete")
                shown = ", ".join(f"{_dir_label(k)}={usable[k].point:.3f}"
                                  for k in sorted(usable, key=_dir_label))
                est.note = (
                    f"{len(usable)} acquisition-split run group(s) present ({shown}), but "
                    f"they do not include a direction and its reverse: distinct directions "
                    f"found = {sorted(_dir_label(k) for k in parsed) or 'none'}"
                    + (f", unidentifiable variant(s) = {sorted(unparsed)}" if unparsed else "")
                    + ". This control is the WORSE of train-A/test-B and train-B/test-A, so "
                      "one direction -- however many files it was written to -- cannot "
                      "establish it. Reported MISSING rather than scored on the surviving "
                      "arm (run stage 5 with --both-directions).")
                est.detail = {"directions_present": sorted(_dir_label(k) for k in usable),
                              "distinct_directions": sorted(_dir_label(k) for k in parsed),
                              "unidentified_variants": sorted(unparsed),
                              "n_directions": len(parsed)}
                return est
            worst_k = min(usable, key=lambda k: (usable[k].lo if usable[k].has_ci
                                                 else usable[k].point))
            est = usable[worst_k]
            est.source = "s05 acquisition_split"
            est.note = (f"worse of {len(parsed)} distinct split direction(s) "
                        f"({_dir_label(worst_k)}); "
                        + "; ".join(f"{_dir_label(k)}={e.point:.3f}"
                                    for k, e in sorted(usable.items(), key=lambda kv: _dir_label(kv[0])))
                        + (f"; {len(unparsed)} run group(s) with an unidentifiable "
                           f"direction were shown but not counted as a direction"
                           if unparsed else ""))
            # THE COMPARISON BASIS, said out loud on the estimate itself so C5
            # cannot print an erosion without printing what it is an erosion
            # between. Every other control in this file is pooled onto the
            # headline's own out-of-fold test set and is differenced against it
            # like with like; this one cannot be, and the reader has to be told
            # which of the two situations they are looking at.
            est.detail = dict(est.detail or {}, **{
                "comparison_basis": (
                    f"the acquisition-stratified arms train on one protocol group and "
                    f"test on the other, so each arm's test set is a protocol subgroup "
                    f"(" + ", ".join(
                        f"{_dir_label(k)}: {e.n if e.n is not None else '?'} slices / "
                        f"{e.n_clusters if e.n_clusters is not None else '?'} subjects"
                        for k, e in sorted(usable.items(), key=lambda kv: _dir_label(kv[0])))
                    + ") and NOT the pooled out-of-fold set the headline is computed on. "
                      "The arms are disjoint by design and are never merged, so the "
                      "figure below is a difference ACROSS test sets, not a matched "
                      "difference"),
                "arm_rows": sorted({int(k) for p in by_dir.get(worst_k, [])
                                    for k in ((p.get("test") or {}).get("cache_idx") or [])}),
                "arm_label": _dir_label(worst_k)})
            return est

        return self.group_estimate(cohort, runs, f"s05 {canonical}",
                                   CANONICAL_LABEL.get(canonical, canonical))

    def confound_targets(self, cohort: str,
                         condition: str = HEADLINE_CONDITION) -> List[Tuple[str, Estimate]]:
        """
        (target, estimate) for every confound the network was asked to predict.

        These are scanner-identity AUCs. `label_semantics` on each payload says
        so explicitly, and it is checked here rather than assumed: a payload
        claiming diagnostic semantics under this control would be a bug worth
        surfacing, not averaging in.

        A confound run stage 5 recorded as failed returns nothing at all: the
        criterion asks whether the MOST predictable confound stays low, so
        dropping the target that crashed and taking the max over the survivors
        is exactly the wrong direction to be wrong in.

        The same asymmetry applies to the semantics check, which is why the
        payloads it rejects are reported by `confound_defects` instead of only
        being logged. Dropping a payload here removes a candidate for "most
        predictable confound", so a malformed high-AUC run silently lowers the
        maximum the criterion is testing.
        """
        if self.failure(cohort, "confound") is not None:
            return []
        by_target, _ = self._confound_groups(cohort, condition)
        out = [(t, _envelope(es, "s05 confound_predictability",
                             f"envelope over {len(es)} run(s)"))
               for t, es in by_target.items()]
        return [(t, e) for t, e in out if e.ok]

    def _confound_groups(self, cohort: str, condition: str
                         ) -> Tuple[Dict[str, List[Estimate]], List[str]]:
        # THE ONE CALLER THAT ASKS FOR EVERY SEMANTICS. C6 is a maximum over
        # targets, so a confound run dropped before it gets here can only lower
        # the number the criterion is testing; the rejection has to happen where
        # it can be REPORTED (`defects` below, read by `confound_defects`), not
        # in the selector.
        runs = self.select(cohort, "confound", condition, semantics=None)
        cmap, _unit = self.cluster_map_for(cohort)
        by_target: Dict[str, List[Estimate]] = {}
        defects: List[str] = []
        for p in runs:
            e = _control_estimate(p, cmap)
            sem = str(e.detail.get("label_semantics") or "")
            if not sem.startswith("confound:"):
                name = str(e.detail.get("variant") or "unnamed target")
                logger.warning("confound run %s has label_semantics=%r; not scored",
                               p.get("_path"), sem)
                defects.append(
                    f"a confound_predictability run ({name}, AUC "
                    f"{e.point:.3f}) carries label_semantics={sem!r} instead of "
                    f"'confound:<target>', so it cannot be read as a "
                    f"scanner-identity AUC and was not scored")
                continue
            target = str(e.detail.get("variant") or sem.split(":", 1)[1] or "confound")
            by_target.setdefault(target, []).append(e)
        return by_target, defects

    def confound_defects(self, cohort: str,
                         condition: str = HEADLINE_CONDITION) -> List[str]:
        """
        Confound runs stage 5 wrote that `confound_targets` could not score.

        C6 takes the MAXIMUM over targets, so any target dropped from the set
        can only lower it. A payload whose `label_semantics` is not
        `confound:<target>` was previously dropped with a log line and the
        criterion was then decided on the survivors -- so a
        receiver-coil-count run at AUC 0.97 written with the wrong semantics
        field left C6 PASSING on a second target at 0.45 (`self_test`,
        A5_confound_semantics_drop). C6 now refuses to pass while any confound
        run stage 5 produced is unaccounted for.
        """
        if self.failure(cohort, "confound") is not None:
            return []
        return self._confound_groups(cohort, condition)[1]


def load_control_payloads(dirs: Sequence[Path]) -> List[dict]:
    """
    Load stage-5 control run JSONs.

    Mirrors s05's own loader: skip `_scratch`, de-duplicate by resolved path
    (the same tree can be reachable through two of the search paths, and
    double-counting permutation replicates would make the null look better
    sampled than it is).
    """
    seen, out = set(), []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for p in sorted(d.rglob("*.json")):
            rp = str(p.resolve())
            if p.parent.name == "_scratch" or rp in seen:
                continue
            seen.add(rp)
            obj = load_json(p)
            if not isinstance(obj, dict) or "control" not in obj or "test" not in obj:
                continue
            obj["_path"] = rp
            # The search directory this run was found under, so that a fold
            # directory can be located RELATIVE to the controls tree root and
            # two five-fold control sweeps in two subdirectories come out as two
            # split families rather than one (`_control_fold_identity`).
            obj["_control_root"] = str(d.resolve()) if d.exists() else str(d)
            out.append(obj)
    return out


# s05_controls.FAILURES_FILENAME. Duplicated as a literal rather than imported
# so that s06 can read a controls tree produced by any s05 version, and so the
# report never fails to build because stage 5 is not importable.
CONTROL_FAILURES_FILENAME = "control_failures.json"


def load_control_failures(dirs: Sequence[Path]) -> Dict[Tuple[str, str], dict]:
    """
    Read stage 5's failure ledger: which controls it attempted and which broke.

    Returns {(cohort_slug, canonical_control): entry} for FAILED entries only.
    s05's _guarded() swallows exceptions so one broken control cannot abort the
    rest; the ledger is where it writes down that it did so. Without reading it,
    a control that half-ran (one acquisition direction dead single-class, say)
    is indistinguishable on disk from one that ran clean, and the surviving arm
    gets scored as though it were the whole control.

    A missing ledger is NOT treated as "everything failed" -- controls trees
    predating the ledger are legitimate and simply carry no provenance. It is
    also not treated as "everything is fine": the entries that exist are
    believed, and the rest is decided by the run files as before.
    """
    out: Dict[Tuple[str, str], dict] = {}
    seen: set = set()
    candidates: List[Path] = []
    for d in dirs:
        d = Path(d)
        candidates.append(d / CONTROL_FAILURES_FILENAME)
        # s05 writes the ledger at the root of --results-dir; a caller may point
        # s06 at a subdirectory of it, so look one level up as well.
        candidates.append(d.parent / CONTROL_FAILURES_FILENAME)
        if d.exists():
            candidates.extend(sorted(d.rglob(CONTROL_FAILURES_FILENAME)))
    for path in candidates:
        try:
            rp = str(path.resolve())
        except OSError:                      # pragma: no cover - unreadable path
            continue
        if rp in seen or not path.exists():
            continue
        seen.add(rp)
        obj = load_json(path)
        if not isinstance(obj, dict):
            continue
        for entry in (obj.get("entries") or []):
            if not isinstance(entry, dict) or not entry.get("failed"):
                continue
            canon = S05_TO_CANONICAL.get(str(entry.get("control")))
            if canon is None:
                continue                     # "none" (a headline re-run) or unknown
            key = (_ident(entry.get("cohort")), canon)
            rec = dict(entry)
            rec["_path"] = rp
            prev = out.get(key)
            if prev is not None:
                # Two trees disagree: keep both error lists. A failure recorded
                # anywhere is a failure -- there is no quorum rule that could
                # make it safe to ignore one.
                rec["errors"] = list(prev.get("errors") or []) + list(rec.get("errors") or [])
            out[key] = rec
    return out


def harvest_controls_from_runs(runs: Sequence[dict], cluster_maps: Dict[str, Tuple[dict, str]],
                               n_boot: int, seed: int) -> Dict[str, Dict[str, dict]]:
    """
    Reconstruct the background-only control from plain stage-3 run JSONs.

    `s03_train.py --region background` is already the falsification control and
    writes an ordinary run JSON. If stage 5 has not run but those runs exist, C4
    can still be populated from them -- tagged `s06-fallback` so it is never
    mistaken for a stage-5 result.
    """
    out: Dict[str, Dict[str, dict]] = {}
    by_key: Dict[Tuple[str, str, str], List[dict]] = {}
    for r in runs:
        region = str(r.get("region", "full"))
        if region == "full" or r.get("control") not in (None, "none"):
            continue
        by_key.setdefault((str(r["cohort"]), str(r["condition"]), region), []).append(r)

    for (cohort, condition, region), group in by_key.items():
        if region != "background" or condition != HEADLINE_CONDITION:
            continue
        cmap, unit = cluster_maps.get(cohort, ({}, "patient_id"))
        pooled = pool_runs(group, cmap, unit)
        if pooled is None:
            continue
        bs = cluster_bootstrap(pooled.labels, pooled.probs, pooled.clusters,
                               n_boot=n_boot, seed=seed)
        out.setdefault(cohort, {})["background"] = {
            "condition": condition,
            # n_boot_valid travels with the interval: the power floor every
            # control-based criterion is now held to asks how many resamples
            # actually produced a defined AUC, and an interval that arrives
            # without that count cannot be audited (so it cannot pass).
            "auc": {"point": bs["point"], "lo": bs["lo"], "hi": bs["hi"],
                    "n": bs["n"], "n_pos": bs["n_pos"], "n_clusters": bs["n_clusters"],
                    "n_boot_valid": bs["n_boot_valid"]},
            "source": "s06-fallback (region=background stage-3 runs)",
            "n_runs": pooled.n_runs,
        }
    return out


# ==========================================================================
# Verdict engine -- the only place a verdict is decided
# ==========================================================================

def _fmt_est_inline(est: Estimate) -> str:
    if not est.ok:
        return "n/a"
    if est.has_ci:
        return f"{est.point:.3f} [{est.lo:.3f}, {est.hi:.3f}]"
    return f"{est.point:.3f} (no CI)"


# --------------------------------------------------------------------------
# Aggregate provenance: a CONSENSUS of the members, never a loan from one
# --------------------------------------------------------------------------
#
# `_test_set_mismatch` below asks ONE object for its test-set fingerprint. Every
# object it is asked about on the criterion side is an AGGREGATE -- an envelope
# over seeds, a permutation null pool, a control pool over folds, a
# comparison-level group, the row stage 4's per-seed blocks are reduced to --
# and each of those used to synthesise its fingerprint out of whichever member
# happened to have one: `n = good[0].n`, `base = _control_estimate(runs[0])`,
# `n_clusters = _first(group[0], ...)`. `_test_set_mismatch` fails closed on a
# missing declaration but has no way to notice a declaration that is true of one
# member and false of the group, so a fifth adversarial round reached SUPPORTED
# four ways one level below the guard:
#
#   * a second background/scramble run scored on 40 slices / 200 subjects,
#     enveloped with the honest one; the hull reported the honest run's 560/70
#     and C4 and C7 passed on it (`AG1`, and `AG2` with the same subjects but a
#     different slice selection, which `min()` over cluster counts cannot see);
#   * 20 permutation replicates of which ONE was scored on the headline's fold;
#     `runs[0]` was that one, so the null pool reported the headline's fingerprint
#     and C8 passed on a null measured on 4 subjects (`AG3`);
#   * a phase-vs-magnitude comparison whose reference condition has no
#     predictions in this report at all, so the fold check that would have
#     compared the two sides never ran (`AG4`).
#
# The rule below is applied ONCE and every aggregation site is routed through it:
#
#   * the aggregate's fingerprint is the value ALL of its members declare;
#   * if the members DISAGREE on any component, the aggregate carries NO
#     fingerprint -- disagreement is strictly worse than absence, and the
#     missing-declaration case already withholds the PASS;
#   * if SOME members declare a component and others do not, that is a
#     disagreement too: a silent member cannot be shown to match a loud one;
#   * if NO member declares a component, it is simply unknown -- which for the
#     sizes `_test_set_mismatch` reads is already a refusal, and for the rest is
#     the state every tree on disk today is in.
#
# It can only move a criterion PASS -> MISSING. Every FAIL path in
# `evaluate_cohort` is decided before the fingerprint is consulted, because a
# wrong provenance is not a reason to upgrade a criterion.

# The components of a test-set fingerprint, in reporting order.
_FP_ORDER = ("split_family", "folds", "n_clusters", "n", "subjects")
_FP_LABEL = {
    "split_family": "split family",
    "folds": "fold set",
    "n_clusters": "independent subject count",
    "n": "slice count",
    "subjects": "subject-id set",
}


def _fp_value(component: str, raw: Any) -> Any:
    """
    One member's declaration for one component, normalised for comparison.

    Anything unparseable becomes None, i.e. "this member did not declare it",
    which is the conservative reading: a fold list of `["a", "b"]` is not a
    fold set that can be shown to match anything.
    """
    if raw is None:
        return None
    if component in ("n", "n_clusters"):
        try:
            return int(raw)
        except (TypeError, ValueError):
            return None
    if component == "folds":
        try:
            return tuple(sorted(int(f) for f in raw))
        except (TypeError, ValueError):
            return None
    if component == "subjects":
        try:
            ids = frozenset(str(s) for s in raw)
        except TypeError:
            return None
        return ids or None
    text = str(raw)
    return text or None


def _fp_show(component: str, value: Any) -> str:
    """A printable form of one fingerprint component."""
    if value is None:
        return "(not declared)"
    if component == "subjects":
        shown = sorted(value)[:3]
        return (f"{len(value)} subject(s) [" + ", ".join(shown)
                + (", ..." if len(value) > 3 else "") + "]")
    if component == "folds":
        return str(list(value))
    return str(value)


def _consensus_fingerprint(members: Sequence[Mapping[str, Any]],
                           what: str = "this aggregate"
                           ) -> Tuple[Dict[str, Any], List[str]]:
    """
    The test-set fingerprint EVERY member of an aggregate declares.

    `members` is one mapping per member, each giving that member's declaration
    for the components in `_FP_ORDER` (or None / absent where it declares
    nothing).

    Returns `(fingerprint, conflicts)`. `conflicts` is empty iff the members
    agree on every component any of them declares; when it is not empty the
    fingerprint is `{}` -- an aggregate that cannot show its members describe
    one test set carries NO fingerprint at all, rather than the intersection or
    the first member's or the smallest.
    """
    mems = [dict(m or {}) for m in members]
    if not mems:
        return {}, [f"{what} pools no runs at all, so it describes no test set"]
    fp: Dict[str, Any] = {}
    conflicts: List[str] = []
    for comp in _FP_ORDER:
        vals = [_fp_value(comp, m.get(comp)) for m in mems]
        declared = [v for v in vals if v is not None]
        if not declared:
            # Nobody declares it. Unknown, not disputed -- and for `n` and
            # `n_clusters` the missing-declaration rule in `_test_set_mismatch`
            # already refuses to difference against it.
            continue
        silent = len(vals) - len(declared)
        if silent:
            conflicts.append(
                f"{silent} of {what} record no {_FP_LABEL[comp]} while the other "
                f"{len(declared)} record {_fp_show(comp, declared[0])}, and a run "
                f"that does not say cannot be shown to match one that does")
            continue
        uniq: List[Any] = []
        for v in declared:
            if v not in uniq:
                uniq.append(v)
        if len(uniq) > 1:
            conflicts.append(
                f"{what} disagree on {_FP_LABEL[comp]}: "
                + " vs ".join(_fp_show(comp, v) for v in uniq[:4])
                + (f" (+{len(uniq) - 4} more)" if len(uniq) > 4 else ""))
            continue
        fp[comp] = uniq[0]
    if conflicts:
        return {}, conflicts
    return fp, []


def _fingerprint_violations(reported: Mapping[str, Any],
                            members: Sequence[Mapping[str, Any]],
                            what: str) -> List[str]:
    """Components `reported` claims that some member does not share. Empty = legal."""
    out: List[str] = []
    mems = [dict(m or {}) for m in members]
    for comp in _FP_ORDER:
        val = _fp_value(comp, (reported or {}).get(comp))
        if val is None:
            continue
        for i, m in enumerate(mems):
            got = _fp_value(comp, m.get(comp))
            if got != val:
                out.append(
                    f"a pool of {what} reports {_FP_LABEL[comp]} "
                    f"{_fp_show(comp, val)} while the run at position {i} declares "
                    f"{_fp_show(comp, got)}")
    return out


def _assert_aggregate_fingerprint(record: Mapping[str, Any]) -> None:
    """
    The invariant that would have caught this whole class of defect.

    An aggregate may NEVER report a test-set fingerprint that is not shared by
    every member it aggregates. Not the first member's, not the smallest, not
    whichever member happens to have declared one: every route to SUPPORTED in
    the fifth adversarial round was one number differenced against another after
    an aggregate had quietly adopted one member's provenance as the group's.

    Raising here stops the report being written at all, which is the right
    outcome: a fingerprint the members do not share is a wiring bug in the
    aggregation layer, not a result.
    """
    bad = _fingerprint_violations(record.get("reported") or {},
                                  record.get("members") or [],
                                  str(record.get("what") or "an aggregate"))
    if bad:
        raise AssertionError(
            "aggregate provenance violated: " + "; ".join(bad[:4])
            + (f" (+{len(bad) - 4} more)" if len(bad) > 4 else "")
            + " -- an aggregate may only report a test-set fingerprint that every "
              "member it pools shares, or none at all")


def _member_test_set(est: Estimate) -> Dict[str, Any]:
    """What ONE estimate declares about the test set it was scored on."""
    ts = (est.detail or {}).get("test_set")
    if isinstance(ts, dict):
        return dict(ts)
    return {"n_clusters": est.n_clusters, "n": est.n}


def _agree_or_forget(out: Estimate, members: Sequence[Mapping[str, Any]],
                     what: str) -> List[str]:
    """
    Give `out` the fingerprint its members AGREE on, and nothing else.

    THE routing point. Every aggregation site in this module calls this and
    nothing else writes an aggregate's `n` / `n_clusters` / `test_set`. On any
    conflict the sizes are cleared, which makes `_test_set_mismatch` refuse to
    difference the aggregate against the headline exactly as it already refuses
    an undeclared size -- so a criterion that would have PASSED goes MISSING,
    and one that was going to FAIL is untouched.

    Returns the conflicts, so the caller can say them out loud.
    """
    fp, conflicts = _consensus_fingerprint(members, what)
    out.n = fp.get("n")
    out.n_clusters = fp.get("n_clusters")
    pos = [m.get("n_pos") for m in members]
    out.n_pos = (int(pos[0]) if (not conflicts and pos and all(
        p is not None and int(p) == int(pos[0]) for p in pos)) else None)
    detail = dict(out.detail or {})
    detail["test_set"] = dict(fp)
    detail["fingerprint_conflicts"] = list(conflicts)
    detail["aggregate_provenance"] = {
        "what": what, "reported": dict(fp),
        "members": [dict(m or {}) for m in members]}
    out.detail = detail
    # Checked at the moment the aggregate is built, not only at verdict time, so
    # a site that is added later and forgets the rule cannot reach a report.
    _assert_aggregate_fingerprint(detail["aggregate_provenance"])
    return list(conflicts)


def _test_set_mismatch(head: Estimate, other: Estimate,
                       other_label: str = "this control") -> List[str]:
    """
    Why `other` is NOT demonstrably on the same test set as the headline.

    Empty list = both sides declare their sample size and the sizes agree.
    Anything else -- a disagreement, or a side that declares nothing -- is a
    reason, and it withholds a PASS.

    ONE function, used by every criterion that differences something against the
    headline. C8 has always asked this question ("a null computed on a different,
    say easier or smaller, test set is not the null distribution of THIS
    headline"); C4, C5, C6 and C7 subtracted a control AUC from the headline AUC
    with no check at all, so one RESULTS.md could refuse the comparison as
    meaningless for C8 and pass the other four on exactly it. Live on the real
    tree: the only prostate_t2 background control sits in
    controls/results/prostate_t2_cv0, was scored on that fold's 15 subjects, and
    was differenced against the 67-subject pooled out-of-fold headline.

    MISSING PROVENANCE FAILS CLOSED. The old C8 form was

        if a is not None and b is not None and int(a) != int(b): refuse

    which is defeated by DELETING information: the same null on the same wrong
    fold was REFUSED when its payload declared `n_clusters` and ACCEPTED when it
    did not. An undeclared size is not evidence of agreement, so it is a reason
    here. This can only move a criterion from PASS to MISSING; the fail paths
    that do not depend on the headline are decided before it is consulted.

    AN AGGREGATE WITH NO CONSENSUS HAS NO FINGERPRINT. Either side may be a pool
    -- an envelope over seeds, a null over replicates, a control pool over folds
    -- and `_agree_or_forget` clears the sizes of any pool whose own members do
    not agree on them, which lands in the undeclared-size branches below. The
    reason is said in its own words first: "these runs disagree" is a different
    and more serious statement than "this run did not say", and a reader told
    the second when the first is true goes looking for a field that is not
    missing.

    ALL FIVE COMPONENTS OF `_FP_ORDER`, NOT TWO. This compared `n` and
    `n_clusters` and stopped there, while `split_family`, `folds` and `subjects`
    are first-class members of the fingerprint, are read out of every control
    payload by `_control_test_set`, are written onto every aggregate by
    `_agree_or_forget`, and were carried the whole way here to be ignored. Two
    test sets can agree exactly on both counts and share not one subject: fold 1
    and fold 3 of a five-fold split hold 14 subjects and 448 slices each, a
    stale sweep in a second directory has the identical shape to the live one,
    and a control pooled over folds [0,1,2,3] matches a headline pooled over
    [0,1,2,4] on every count there is. Counting is the weakest evidence of
    identity available, and it was the only evidence being asked for.

    The two size components keep their fail-closed rule, because every stage-4
    row and every stage-5 payload records them: silence there is a refusal. The
    other three are compared WHERE BOTH SIDES DECLARE THEM -- a disagreement is a
    reason; a side that declares nothing adds no constraint, exactly as
    `_consensus_fingerprint` treats a component no member declares. Widening can
    only ADD reasons, so it can only move a criterion from PASS to MISSING.

    What that means on today's trees, stated so nobody reads more into it than
    is there: `split_family` and `folds` are declared by BOTH sides -- stage 4
    writes them on every row and `Controls` parses them off the controls tree --
    so those two are live, and they are what refuses a control from a second
    cross-validation sweep whose sizes happen to match. `subjects` is declared by
    the control side only: stage 4 records no subject-id set for the headline, so
    the component is carried, compared, and inert until some stage writes one.
    It is compared rather than skipped because the moment either side gains a
    subject set it is the strongest evidence of identity available, and because
    the two sides are already in ONE NAMESPACE -- stage 5's `patient_ids` are a
    different identifier from the `subject_id` the headline clusters on, so
    `Controls` translates a control's subject set through the headline's own
    cache-row table before declaring it, and declares nothing when it cannot
    (`_control_cluster_ids`).
    """
    reasons: List[str] = []
    disputed = set()
    for side, est in (("head", head), ("other", other)):
        conflicts = (est.detail or {}).get("fingerprint_conflicts") or []
        if not conflicts:
            continue
        disputed.add(side)
        name = "the headline" if side == "head" else other_label
        reasons.append(f"{name} is a pool whose own runs do not describe one test set "
                       f"({'; '.join(conflicts[:2])}), so it has no fingerprint to be "
                       f"matched")
    if disputed:
        # A side whose conflict has just been named carries None for every size
        # BECAUSE of that conflict. Repeating it below as "did not record how
        # many subjects" would send the reader looking for a field that is
        # present in every run of the pool. The refusal is already in `reasons`.
        return reasons
    for label, a, b in (("independent subject", head.n_clusters, other.n_clusters),
                        ("slice", head.n, other.n)):
        if a is None and b is None:
            reasons.append(f"neither the headline nor {other_label} records how many "
                           f"{label}s it was scored on")
        elif a is None:
            reasons.append(f"the headline does not record how many {label}s it "
                           f"was scored on")
        elif b is None:
            reasons.append(f"{other_label} does not record how many {label}s it was "
                           f"scored on, so it cannot be shown to be the same test set")
        elif int(a) != int(b):
            reasons.append(f"{label} count {int(a)} (headline) vs {int(b)} "
                           f"({other_label})")
    head_fp = (head.detail or {}).get("test_set") or {}
    other_fp = (other.detail or {}).get("test_set") or {}
    for comp in _FP_ORDER:
        if comp in ("n", "n_clusters"):
            continue                       # decided above, with the fail-closed rule
        a = _fp_value(comp, head_fp.get(comp))
        b = _fp_value(comp, other_fp.get(comp))
        if a is None or b is None:
            # Neither the headline nor the control is REQUIRED to record these:
            # no stage writes a subject-id set into statistics.json, and a
            # non-cross-validated tree has no fold set to record. An undeclared
            # component adds no constraint here rather than refusing everything,
            # which is the same reading `_consensus_fingerprint` gives it.
            continue
        if a != b:
            reasons.append(f"{_FP_LABEL[comp]} {_fp_show(comp, a)} (headline) vs "
                           f"{_fp_show(comp, b)} ({other_label})")
    return reasons


def _matched_headline(pred: Optional["PooledPredictions"],
                      rows: Optional[Sequence[int]]) -> Optional[Tuple[float, int]]:
    """
    The headline AUC recomputed on EXACTLY the cache rows a control was scored on.

    Reporting only, and only where the control's rows are a SUBSET of the
    headline's out-of-fold vector, i.e. where the two really are the same
    subjects and the same slices scored by two models. That is the one case in
    which a like-with-like difference exists for a control whose test set is a
    designed subgroup rather than the whole cohort, and it is the number the
    acquisition-stratified erosion should be read against.

    The headline side is the MEAN OVER SEEDS of the per-seed AUC on those rows,
    not the AUC of the seed-averaged vector, because the control it is about to
    be differenced against is a SINGLE stage-5 run. Matching a 2-seed ensemble
    against a 1-seed control is not like with like: the ensemble scores above
    both of its own constituents (prostate_t2 phase 0.622/0.636 -> 0.650), so
    the erosion printed beside it would be inflated by an aggregation artefact
    in the direction of "the control destroyed the signal". Falls back to the
    ensemble only where no per-seed view exists, and says which it used.

    Returns `(auc, n_rows, n_seeds)` or None -- None whenever the rows are
    absent, are not all present in the pooled vector, or leave a single-class
    block, because a partial match is not a match. `n_seeds` is 0 when the
    number is the ensemble's.
    """
    if pred is None or not rows:
        return None
    try:
        want = [int(r) for r in rows]
    except (TypeError, ValueError):           # pragma: no cover - defensive
        return None
    pos = {int(k): i for i, k in enumerate(pred.cache_idx.tolist())}
    if any(r not in pos for r in want):
        return None
    sel = [pos[r] for r in want]
    sm = seed_mean_auc(pred, sel)
    if sm is not None:
        return float(sm[0]), len(sel), sm[1]
    auc = _safe_auc(pred.labels[sel], pred.probs[sel])
    if math.isnan(auc):
        return None
    return float(auc), len(sel), 0


def _assert_estimate_within_ci(est: Estimate, what: str) -> None:
    """
    A point estimate outside its own interval is a wiring bug, not a result.

    It happens when a point and an interval are read from two different places
    -- which is exactly what the defect this guard was added for did: the point
    came from an across-seeds row spanning two split families and the interval
    from the per-run blocks of one of them, giving "0.794 [0.600, 0.780]" in
    print. Raising here stops the report being written at all.
    """
    if not est.ok or not est.ci_finite:
        return
    if not (est.lo - 1e-9) <= est.point <= (est.hi + 1e-9):
        raise AssertionError(
            f"{what}: point estimate {est.point:.6g} lies OUTSIDE its own 95% "
            f"interval [{est.lo:.6g}, {est.hi:.6g}] (source {est.source!r}, "
            f"{est.note!r}). A point and an interval read from different scopes "
            f"are not an estimate of anything.")


def evaluate_cohort(cohort: str,
                    headline: Headline,
                    reference: Estimate,
                    comparison_levels: Dict[str, dict],
                    controls: Controls,
                    *,
                    cluster_headline: Headline,
                    cluster_level: str,
                    is_primary: bool,
                    external_confounds: Sequence[Tuple[str, Estimate, str]] = (),
                    stats_scheme_mismatch: str = "",
                    coverage_defect: str = "",
                    comparison_mismatch: str = "",
                    headline_predictions: Optional["PooledPredictions"] = None,
                    ) -> CohortVerdict:
    """
    Apply the criteria. Pure function of its arguments.

    Status semantics, printed verbatim in the report:
        pass    - the criterion was evaluated and met
        fail    - the criterion was evaluated and NOT met
        missing - the criterion could not be evaluated (input absent or
                  degenerate)

    SUPPORTED requires every criterion to be `pass`. A single `fail` gives NOT
    SUPPORTED and names the criterion. Any `missing` with no `fail` gives
    INCONCLUSIVE. `_assert_verdict_consistent` re-checks that combination before
    anything is written.

    Two different views of the same quantity are passed in on purpose, and
    mixing them up is the bug this signature exists to prevent. Both are
    `Headline` objects -- resolved ONCE by the caller, carrying their own
    provenance -- and no criterion may compute a headline of its own:

    `cluster_headline`  the headline AUC at `cluster_level` -- one score per
                        subject, which is the only level whose interval has an
                        independence assumption that holds. C1 is decided here
                        and nowhere else.
    `headline`          the headline AUC at the SLICE level. Stage 5 scores
                        every control at the slice level (`test.auc` and
                        `control_detail.test_auc_ci95` in each control payload),
                        so C4/C5/C7/C8 -- which are all comparisons AGAINST a
                        control -- have to read the headline at the same level
                        or the difference is an artefact of the aggregation, not
                        of the control.

    Every criterion that quotes one of them records WHICH one and WHAT VALUE on
    the Criterion it emits, and every criterion that DIFFERENCES a control
    against one of them records whether it verified the two share a test set
    (`_test_set_mismatch`, the check C8 has always applied). Those three fields
    are what `_assert_verdict_consistent` re-checks: no point estimate outside
    its own interval, no two criteria quoting different headlines for one
    (cohort, condition, level), and no PASS on an unverified difference.

    `is_primary` says whether this cohort is the pre-registered primary one. It
    decides C0 and nothing else: an exploratory cohort's criteria are still
    evaluated and printed, they just cannot add up to a confirmatory SUPPORTED.

    `external_confounds` are (target, estimate, provenance) triples measured on a
    DIFFERENT cohort -- in practice the brain confound cohort, where the label is
    receive-coil count and nothing else, and the test fold holds 136 independent
    subjects instead of the 4-7 a clinical official split holds. They feed C6 in
    ONE DIRECTION ONLY: an external confound at or above CONFOUND_AUC_MAX FAILS
    C6, and nothing about them can make C6 pass, can supply a missing stage-5
    confound control, or can turn a MISSING into a PASS. "This input channel
    encodes the scanner" is a claim about the channel and transfers between
    cohorts; "it does not encode the scanner here" is a claim about one set of
    scanners and does not.

    `stats_scheme_mismatch`, when non-empty, says that the stage-4 records this
    cohort's C1/C2 would be read from do not describe the predictions the report
    is headlining (typically: stage 4 summarised the 7-subject official split
    while the headline is the pooled out-of-fold cross-validation). Those two
    criteria are then capped at MISSING with the reason named. It can only move a
    criterion from PASS to MISSING, never the other way.

    `coverage_defect`, when non-empty, says that the headline predictions do not
    cover the cross-validation the report claims to be reporting: a fold died, or
    subjects the design assigned to a test fold never reached the pooled vector.
    A fold dies for data-dependent reasons -- a single-class test block, an OOM
    on the largest fold -- so the surviving subjects are a sample selected by the
    failure, not a random subsample of the cohort, and a criterion decided on
    them is not a criterion decided on the cohort. Capped the same way, and with
    the same one-way property: PASS -> MISSING only.

    `comparison_mismatch`, when non-empty, says the same thing about the OTHER
    side of C2: the stage-4 comparison records, or the reference condition's own
    stage-4 rows, do not describe the fold set this report headlines. Caps C2
    alone -- C1 does not involve the reference condition -- and PASS -> MISSING
    only.
    """
    crit: List[Criterion] = []
    rules = {key: (code, rule) for code, key, rule in CRITERIA_ORDER}
    # Every headline any criterion quoted, so the invariant checker can see the
    # whole set at once rather than one criterion at a time.
    quoted: Dict[str, Tuple[float, float, float]] = {}
    # ... and every AGGREGATE any criterion was decided on, for the same reason:
    # the invariant "an aggregate may not report a fingerprint its members do not
    # share" has to be checkable over the whole verdict, not one pool at a time.
    aggregates: List[dict] = []

    def watch(est: Estimate) -> Estimate:
        """Record an aggregate's declared provenance on the way past."""
        prov = (est.detail or {}).get("aggregate_provenance")
        if prov and prov not in aggregates:
            aggregates.append(prov)
        return est

    def add(key: str, status: str, detail: str, evidence: str = "",
            quotes: Optional[Headline] = None,
            differenced: bool = False, test_set_verified: bool = False):
        code, rule = rules[key]
        c = Criterion(key=key, code=code, rule=rule, status=status,
                      detail=detail, evidence=evidence,
                      differenced=differenced,
                      test_set_verified=test_set_verified)
        if quotes is not None and quotes.ok:
            c.headline_key = quotes.key
            c.headline_point = quotes.point
            quoted.setdefault(quotes.key, (quotes.point, quotes.lo, quotes.hi))
        crit.append(c)

    # --- C0: is this the pre-registered primary cohort? --------------------
    # Three cohorts in one RESULTS.md, each tested at the same alpha, is a
    # family of three. With the measured per-cohort null pass rates for C1
    # (7.2% / 5.2% / 15.2%) the chance that at least one clears it under a
    # complete null is ~25%. Naming the primary in advance -- on cohort size and
    # reconstruction fidelity, neither of which depends on an outcome -- leaves
    # exactly one confirmatory test in the family.
    if is_primary:
        add("preregistered_primary", "pass",
            f"{cohort} is the pre-registered primary cohort",
            "chosen on size (67 patients) and on reconstruction validated at "
            "r=0.998 against the vendor images; both fixed before any verdict")
    else:
        add("preregistered_primary", "missing",
            f"{cohort} is EXPLORATORY: the pre-registered primary cohort is "
            f"{PRIMARY_COHORT}. Its criteria below are reported in full, but a "
            f"second cohort tested at the same alpha inflates the family-wise "
            f"error rate (~25% for three cohorts under a complete null), so no "
            f"confirmatory verdict is drawn from it",
            "reported as INCONCLUSIVE by construction; this is a design "
            "decision, not a property of the data")

    # --- C1: phase above chance -------------------------------------------
    # Decided on `cluster_headline`, NOT on `headline`: see the docstring. The
    # slice-level interval is the one that undercovers, and it is also the one
    # that looks best, which is exactly why it is not the one used.
    est_c1 = cluster_headline.est
    n_cl = est_c1.n_clusters
    n_pos_cl = _first(est_c1.detail or {}, "n_pos_clusters")
    try:
        n_pos_cl = None if n_pos_cl is None else int(n_pos_cl)
    except (TypeError, ValueError):
        n_pos_cl = None
    n_neg_cl = None if (n_cl is None or n_pos_cl is None) else int(n_cl) - n_pos_cl
    c1_ev = (_fmt_est_inline(est_c1) + f" [{est_c1.source}] {est_c1.note}"
             if est_c1.ok else "")
    if est_c1.ok and n_cl is not None:
        c1_ev += f"; {n_cl} clusters"
        if n_pos_cl is not None:
            c1_ev += f" ({n_pos_cl} positive, {n_neg_cl} negative)"
    if not est_c1.ok:
        add("phase_above_chance", "missing",
            f"no {HEADLINE_CONDITION} AUC for {cohort} at the {cluster_level} "
            f"level; the slice-level estimate is not a substitute because "
            f"slices within a subject are correlated", quotes=cluster_headline)
    elif not est_c1.has_ci:
        add("phase_above_chance", "missing",
            f"{HEADLINE_CONDITION} {cluster_level} AUC = {est_c1.point:.3f} has "
            "no confidence interval; a point estimate alone cannot clear this "
            "criterion", c1_ev, quotes=cluster_headline)
    elif n_cl is None or n_pos_cl is None:
        add("phase_above_chance", "missing",
            f"stage 4 reported no cluster counts for the {cluster_level} level, "
            f"so the interval cannot be checked against the minimum "
            f"({MIN_CLUSTERS_C1} clusters, {MIN_CLASS_CLUSTERS_C1} per class)",
            c1_ev, quotes=cluster_headline)
    elif int(n_cl) < MIN_CLUSTERS_C1 or min(n_pos_cl, n_neg_cl) < MIN_CLASS_CLUSTERS_C1:
        add("phase_above_chance", "missing",
            f"the test fold has {n_cl} independent cluster(s) ({n_pos_cl} "
            f"positive, {n_neg_cl} negative); below {MIN_CLUSTERS_C1} clusters "
            f"or {MIN_CLASS_CLUSTERS_C1} per class a percentile bootstrap on "
            f"clusters fires at 3-25% under a complete null against a nominal "
            f"2.5%, so neither PASS nor FAIL can be read off it. More bootstrap "
            f"replicates do not fix this -- more subjects do",
            c1_ev, quotes=cluster_headline)
    elif est_c1.lo > 0.5:
        add("phase_above_chance", "pass",
            f"{cluster_level} CI lower bound {est_c1.lo:.3f} > 0.500 on {n_cl} "
            f"clusters ({n_pos_cl} positive, {n_neg_cl} negative)", c1_ev,
            quotes=cluster_headline)
    else:
        add("phase_above_chance", "fail",
            f"{cluster_level} CI lower bound {est_c1.lo:.3f} does not exclude "
            f"chance (0.500)", c1_ev, quotes=cluster_headline)

    # --- C2: phase beats magnitude, DeLong, Holm-adjusted ------------------
    # s04 emits this comparison at three levels and marks the patient level
    # `preferred`, because its own DeLong caveat says the slice-level p-value is
    # anti-conservative (slices within a patient are correlated). So:
    #   * a preferred-level p decides the criterion either way;
    #   * a slice-level p may only produce FAIL. If an anti-conservative test
    #     cannot reach significance, the correct test certainly cannot, so FAIL
    #     is safe -- but a slice-level PASS would be exactly the kind of number
    #     this pipeline exists to refuse.
    # If stage 4 ever marks more than one level `preferred`, take the WORST
    # (largest p), not whichever landed first in the dict. `next(...)` made the
    # criterion depend on the order records happened to appear in
    # statistics.json, which is not a statistical property of anything.
    _prefs = [v for v in comparison_levels.values()
              if v.get("preferred")
              and not math.isnan(v.get("p_holm_worst", float("nan")))]
    pref = max(_prefs, key=lambda v: v["p_holm_worst"]) if _prefs else None
    slice_lvl = comparison_levels.get("slice")

    def _c2_evidence(blk: dict) -> str:
        if not blk:
            return ""
        return (f"level={blk['level']}, seeds={blk['n_seeds']}, "
                f"delta AUC {blk['delta_mean']:+.3f} "
                f"[{blk['delta_min']:+.3f}, {blk['delta_max']:+.3f}], "
                f"worst Holm p="
                f"{'n/a' if math.isnan(blk['p_holm_worst']) else format(blk['p_holm_worst'], '.4g')}, "
                f"clusters={blk.get('n_clusters')}")

    if not comparison_levels:
        add("phase_beats_magnitude", "missing",
            f"statistics.json has no {HEADLINE_CONDITION}-vs-{REFERENCE_CONDITION} "
            "comparison for this cohort")
    elif pref is not None:
        p = pref["p_holm_worst"]
        d = pref["delta_mean"]
        if p < ALPHA and pref["delta_min"] > 0:
            add("phase_beats_magnitude", "pass",
                f"phase exceeds magnitude at every seed (delta {d:+.3f}), "
                f"worst Holm-adjusted p={p:.4g}", _c2_evidence(pref))
        elif pref["delta_max"] <= 0:
            add("phase_beats_magnitude", "fail",
                f"phase does not exceed magnitude (delta {d:+.3f})", _c2_evidence(pref))
        else:
            add("phase_beats_magnitude", "fail",
                f"worst Holm-adjusted p={p:.4g} is not below {ALPHA}"
                + ("" if pref["delta_min"] > 0 else "; the sign of the effect also "
                                                    "changes across seeds"),
                _c2_evidence(pref))
    elif slice_lvl is not None and not math.isnan(slice_lvl.get("p_holm_worst", float("nan"))):
        p = slice_lvl["p_holm_worst"]
        if p >= ALPHA or slice_lvl["delta_max"] <= 0:
            add("phase_beats_magnitude", "fail",
                f"the patient-level comparison is not evaluable, and even the "
                f"anti-conservative slice-level test fails (worst Holm p={p:.4g}, "
                f"delta {slice_lvl['delta_mean']:+.3f})", _c2_evidence(slice_lvl))
        else:
            add("phase_beats_magnitude", "missing",
                "only the slice-level DeLong test is evaluable, and stage 4 states "
                "it is anti-conservative because slices within a patient are "
                "correlated; a significant result at that level is not evidence "
                "the criterion is met", _c2_evidence(slice_lvl))
    else:
        reasons = sorted({r for v in comparison_levels.values() for r in v.get("reasons", [])})
        add("phase_beats_magnitude", "missing",
            "no evaluable Holm-adjusted p-value at any level"
            + (f" ({'; '.join(reasons)})" if reasons else ""),
            _c2_evidence(slice_lvl) if slice_lvl else "")

    # --- C3: label-permutation control -------------------------------------
    perm = watch(controls.estimate(cohort, "permutation"))
    perm_pool_conflicts = list(perm.detail.get("fingerprint_conflicts") or [])
    n_rep = int(perm.detail.get("n_replicates", 0) or 0)
    n_dupe = int(perm.detail.get("n_duplicates", 0) or 0)
    n_unid = int(perm.detail.get("n_unidentified", 0) or 0)
    # The permutation "interval" is the observed range of the replicates, not a
    # bootstrap CI, so the power floor that applies to it is a different one:
    # a null whose replicates span most of the AUC axis covers 0.500 whatever
    # the pipeline does, and "the range covers chance" then says nothing.
    perm_range = perm.ci_width
    if not perm.ok:
        add("permutation_null", "missing",
            ("no label-permutation control (stage 5 has not produced one)"
             if not perm.note else perm.note))
    elif n_rep and n_rep < MIN_PERMUTATION_REPLICATES:
        add("permutation_null", "missing",
            f"only {n_rep} DISTINCT permutation replicate(s)"
            + (f" ({n_dupe} further file(s) were duplicates of one already counted, "
               f"and a copied replicate is not a second draw from the null)"
               if n_dupe else "")
            + (f" ({n_unid} further file(s) declare no `permutation_seed`, so they "
               f"cannot be shown to be further draws from the null)"
               if n_unid else "")
            + f"; the null is not characterised "
              f"(need at least {MIN_PERMUTATION_REPLICATES})",
            _fmt_est_inline(perm) + f" [{perm.source}]")
    elif perm.has_ci and (not perm.ci_finite or perm_range > MAX_CONTROL_CI_WIDTH):
        add("permutation_null", "missing",
            f"the {n_rep} permutation replicate(s) span "
            + ("a non-finite range" if not perm.ci_finite else f"{perm_range:.3f} AUC")
            + f" (> {MAX_CONTROL_CI_WIDTH:.2f}): a null that wide contains 0.500 "
              f"however biased the pipeline is, so 'the range covers chance' "
              f"establishes nothing",
            _fmt_est_inline(perm) + f" [{perm.source}] {perm.note}")
    else:
        near = abs(perm.point - 0.5) <= PERM_NULL_TOL
        # `covers` used to be `(not perm.has_ci) or ...`, i.e. a control payload
        # with no interval was granted the property the criterion is supposed to
        # verify. For the permutation control the "interval" is just the
        # observed min/max of the replicates, so its absence means the null was
        # not characterised at all -- which is a reason to withhold the pass,
        # not to grant it.
        covers = perm.has_ci and (perm.lo <= 0.5 <= perm.hi)
        ev = _fmt_est_inline(perm) + f" [{perm.source}] {perm.note}"
        if not perm.has_ci:
            add("permutation_null", "fail",
                f"the permutation control reports a point AUC {perm.point:.3f} "
                f"with no range over replicates, so 'the null sits at chance' "
                f"cannot be checked; an unverified control is not a passed one",
                ev)
        elif near and covers and perm_pool_conflicts:
            # The replicates are not all draws from ONE null: they were scored on
            # different test sets, so their mean and range are a mixture and "the
            # null sits at chance" is a statement about no particular experiment.
            # Placed AFTER the fail branches on purpose -- a pool that scores
            # above chance still FAILS however it was assembled, because a wrong
            # provenance never upgrades a criterion.
            add("permutation_null", "missing",
                f"the {n_rep} permutation replicate(s) average {perm.point:.3f}, but "
                f"they were not all scored on one test set: "
                + "; ".join(perm_pool_conflicts)
                + ". The mean and range over them are a mixture of experiments, not "
                  "the null distribution of any one of them, so this is reported as "
                  "unmeasured rather than passed", ev)
        elif near and covers:
            add("permutation_null", "pass",
                f"permuted-label AUC {perm.point:.3f} sits at chance over "
                f"{n_rep or len(perm.per_seed)} replicate(s)", ev)
        else:
            why = []
            if not near:
                why.append(f"null mean {perm.point:.3f} is {abs(perm.point - 0.5):.3f} "
                           f"away from 0.500")
            if not covers:
                why.append(f"observed null range [{perm.lo:.3f}, {perm.hi:.3f}] "
                           "does not contain 0.500")
            add("permutation_null", "fail",
                "the permutation control is not at chance: " + "; ".join(why) +
                " -- the pipeline scores above chance on scrambled labels, so "
                "nothing downstream of it can be believed", ev)

    # --- C4 / C7: destroy-the-signal controls ------------------------------
    for canon, key, what in (
        ("background", "background_collapses",
         "removing the anatomy (training on air alone)"),
        ("scramble", "phase_scramble_collapses",
         "destroying spatial structure while preserving the within-mask phase "
         "histogram"),
    ):
        est = watch(controls.estimate(cohort, canon))
        if not est.ok:
            add(key, "missing",
                f"no {CANONICAL_LABEL[canon]} control (stage 5 has not produced one)",
                quotes=headline, differenced=True)
            continue
        if not headline.ok:
            add(key, "missing", "no headline AUC to compare this control against",
                differenced=True)
            continue
        drop = headline.point - est.point
        # Same test set on both sides, or the difference is not a difference:
        # this is the check C8 applies, applied here too. It withholds a PASS
        # and never manufactures a FAIL -- a control sitting too close to the
        # headline still FAILS whatever fold it ran on, because a wrong
        # provenance is not a reason to upgrade a criterion.
        fold_mismatch = _test_set_mismatch(
            headline.est, est, f"the {CANONICAL_LABEL[canon]} control")
        ev = (f"{CANONICAL_LABEL[canon]} {_fmt_est_inline(est)} vs headline "
              f"{_fmt_est_inline(headline.est)} [{est.source}]"
              + (f"; TEST SETS DIFFER: {'; '.join(fold_mismatch)}"
                 if fold_mismatch else
                 f"; both scored on {headline.fingerprint}"))
        # Three independent requirements, and ALL are about the control itself:
        #   1. it must be far enough below the headline (BACKGROUND_MARGIN),
        #   2. its own 95% CI must contain chance, and
        #   3. that CI must be able to discriminate at all.
        # (2) replaces the old `est.lo <= 0.60`, which tolerated a control that
        # was significantly above chance, and its `not est.has_ci` escape, which
        # awarded the pass to any control whose bootstrap block was absent. A
        # control with no interval FAILS: it ran, it scored above chance, and
        # nothing on disk shows it collapsed. That is deliberately stronger than
        # MISSING -- MISSING would read as "we did not check", when what
        # happened is "we checked and could not confirm it".
        #
        # (3) exists because (2) alone REWARDS noise: "the interval contains
        # 0.500" is satisfied by [0.00, 1.00], by [-inf, +inf], by a two-subject
        # bootstrap, and by a bootstrap in which every resample tied. Those are
        # not observations of a collapse, so they are neither PASS nor FAIL:
        # they are MISSING, and the reason is named. The order below is
        # deliberate -- evidence that the control did NOT collapse is decided
        # first and still FAILS, so a wide interval can never launder a control
        # that is demonstrably alive.
        why = []
        if drop < BACKGROUND_MARGIN:
            why.append(f"AUC {est.point:.3f} is only {drop:.3f} below the headline "
                       f"(need >= {BACKGROUND_MARGIN:.2f})")
        if not est.has_ci:
            why.append(f"it has no confidence interval, so its collapse to "
                       f"{CONTROL_CHANCE:.2f} cannot be verified")
        elif est.ci_finite and est.lo > CONTROL_CHANCE:
            why.append(f"its 95% CI [{est.lo:.3f}, {est.hi:.3f}] lies entirely "
                       f"above chance ({CONTROL_CHANCE:.2f}) -- the control is "
                       f"still predicting the label")
        elif est.ci_finite and est.hi < CONTROL_CHANCE:
            why.append(f"its 95% CI [{est.lo:.3f}, {est.hi:.3f}] lies entirely "
                       f"BELOW chance ({CONTROL_CHANCE:.2f}) -- the control "
                       f"predicts the label inversely, which is still label "
                       f"information flowing through an input that should carry "
                       f"none")
        underpowered = [] if why else _control_power_reasons(est)
        if why:
            add(key, "fail",
                f"the signal survives {what}: " + "; ".join(why) +
                (" -- consistent with the model reading the scanner's shim/coil "
                 "signature rather than the tumour" if canon == "background" else
                 " -- an effect that survives scrambling was never spatial, which is "
                 "what a per-scanner phase offset looks like"),
                ev, quotes=headline, differenced=True,
                test_set_verified=not fold_mismatch)
        elif fold_mismatch:
            add(key, "missing",
                f"the {CANONICAL_LABEL[canon]} control ran (AUC {est.point:.3f}) but it "
                f"was not scored on the test set this report headlines "
                f"({headline.fingerprint}): " + "; ".join(fold_mismatch)
                + f". Subtracting it from the headline would measure the difference "
                  f"between two test sets, not the effect of {what} -- the same "
                  f"comparison C8 refuses. Re-run stage 5 against the predictions "
                  f"this report headlines",
                ev, quotes=headline, differenced=True)
        elif underpowered:
            add(key, "missing",
                f"the {CANONICAL_LABEL[canon]} control ran (AUC {est.point:.3f}) but "
                f"its interval cannot decide whether it collapsed: "
                + "; ".join(underpowered)
                + f". An interval that covers {CONTROL_CHANCE:.2f} because it covers "
                  f"everything is not evidence of a collapse, so this is reported as "
                  f"unmeasured rather than passed",
                ev, quotes=headline, differenced=True, test_set_verified=True)
        else:
            add(key, "pass",
                f"{what} costs {drop:.3f} AUC and leaves the control's own CI "
                f"[{est.lo:.3f}, {est.hi:.3f}] (width {est.ci_width:.3f}, "
                f"{est.n_clusters} clusters) covering chance", ev,
                quotes=headline, differenced=True, test_set_verified=True)

    # --- C5: acquisition-stratified split ----------------------------------
    acq = watch(controls.estimate(cohort, "acquisition"))
    if not acq.ok:
        add("acquisition_stratified_holds", "missing",
            "no acquisition-stratified split control (stage 5 has not produced one)",
            quotes=headline, differenced=True)
    elif not headline.ok:
        add("acquisition_stratified_holds", "missing",
            "no headline AUC to compare the stratified split against",
            differenced=True)
    elif not acq.has_ci:
        add("acquisition_stratified_holds", "missing",
            f"stratified AUC {acq.point:.3f} has no interval",
            f"[{acq.source}] {acq.note}", quotes=headline, differenced=True)
    else:
        erosion = headline.point - acq.point
        # Same shared-test-set requirement as C4/C7 and C8. C5's erosion is a
        # difference against the headline, so it means nothing across two test
        # sets; the arm of C5 that reads only the control's OWN interval is
        # unaffected and can still FAIL.
        acq_mismatch = _test_set_mismatch(
            headline.est, acq, "the acquisition-stratified control")
        # THE COMPARISON BASIS, printed with the erosion rather than left for
        # the reader to infer. Every other control this file differences against
        # the headline is pooled onto the headline's own out-of-fold test set
        # first, so the subtraction is like with like; this one CANNOT be,
        # because the two arms are trained on one protocol group and tested on
        # the other and their union is not an experiment. The number below is
        # therefore a difference across test sets, and where the arm's own rows
        # are a subset of the headline's out-of-fold vector, the MATCHED figure
        # -- the headline recomputed on exactly the rows this arm was scored on
        # -- is quoted next to it so the two can be told apart.
        basis = str((acq.detail or {}).get("comparison_basis") or "")
        matched = _matched_headline(headline_predictions,
                                    (acq.detail or {}).get("arm_rows"))
        ev = (f"stratified {_fmt_est_inline(acq)} vs headline "
              f"{_fmt_est_inline(headline.est)} [{acq.source}] {acq.note}"
              + (f"; COMPARISON BASIS: {basis}" if basis else "")
              + (f"; matched basis: on the {matched[1]} row(s) this arm "
                 f"({(acq.detail or {}).get('arm_label', 'the worse arm')}) was actually "
                 f"scored on, the headline's own out-of-fold predictions give "
                 f"{matched[0]:.3f} ("
                 + (f"mean over {matched[2]} seeds, the same one-model basis as this "
                    f"single-seed control" if matched[2]
                    else "the seed-averaged ENSEMBLE -- no per-seed view was available, "
                         "so this is NOT the same one-model basis as this single-seed "
                         "control and reads high")
                 + f"), i.e. a matched drop of "
                 f"{matched[0] - acq.point:+.3f} against this control's "
                 f"{acq.point:.3f}" if matched else "")
              + (f"; TEST SETS DIFFER: {'; '.join(acq_mismatch)}"
                 if acq_mismatch else f"; both scored on {headline.fingerprint}"))
        why = []
        if not acq.ci_finite or acq.lo <= 0.5:
            why.append(f"CI lower bound {acq.lo:.3f} no longer excludes chance")
        if erosion > ACQ_EROSION_MAX:
            why.append(f"AUC falls by {erosion:.3f} (> {ACQ_EROSION_MAX:.2f})")
        # Same asymmetry as C4/C7, mirrored: C5 asks the interval to EXCLUDE
        # chance, so a wide interval costs it a pass rather than buying one --
        # but a NARROW interval on two subjects buys one, which is the same
        # defect from the other side (it is exactly the failure C1's power floor
        # was added for). C5 may therefore still FAIL on any interval, and may
        # only PASS on one that could have failed.
        underpowered = [] if why else _control_power_reasons(acq)
        if why:
            add("acquisition_stratified_holds", "fail",
                "holding acquisition out across the split erases the effect: "
                + "; ".join(why), ev, quotes=headline, differenced=True,
                test_set_verified=not acq_mismatch)
        elif acq_mismatch:
            add("acquisition_stratified_holds", "missing",
                f"the stratified split scored AUC {acq.point:.3f}, but not on the test "
                f"set this report headlines ({headline.fingerprint}): "
                + "; ".join(acq_mismatch)
                + ". The erosion against the headline would then be a difference "
                  "between two test sets rather than the cost of stratifying, which "
                  "is the comparison C8 refuses; reported as unevaluated",
                ev, quotes=headline, differenced=True)
        elif underpowered:
            add("acquisition_stratified_holds", "missing",
                f"the stratified split scored AUC {acq.point:.3f}, but its interval "
                f"cannot support the claim that the effect survived: "
                + "; ".join(underpowered), ev, quotes=headline, differenced=True,
                test_set_verified=True)
        else:
            add("acquisition_stratified_holds", "pass",
                f"effect survives stratification (loses {erosion:.3f} AUC)", ev,
                quotes=headline, differenced=True, test_set_verified=True)

    # --- C6: confound predictability ---------------------------------------
    targets = [(t, watch(e)) for t, e in controls.confound_targets(cohort)]
    confound_defects = controls.confound_defects(cohort)

    # Direct measurements of the same question from a cohort whose label is
    # nothing but hardware. Only those that clear the power floor may be cited
    # as decisive; the rest are shown as context and decide nothing.
    ext_decisive: List[Tuple[str, Estimate, str]] = []
    ext_context: List[str] = []
    for name, e, prov in external_confounds:
        if not e.ok:
            continue
        if (e.n_clusters or 0) < MIN_CLUSTERS_EXTERNAL_CONFOUND:
            ext_context.append(f"{name} {_fmt_est_inline(e)} ({prov}; only "
                               f"{e.n_clusters} test subjects, below the "
                               f"{MIN_CLUSTERS_EXTERNAL_CONFOUND}-subject floor, so it is "
                               f"shown but decides nothing)")
            continue
        ext_decisive.append((name, e, prov))
    ext_fail = [(n, e, prov) for n, e, prov in ext_decisive
                if e.point >= CONFOUND_AUC_MAX]
    ext_why = [
        f"the same input channel ({HEADLINE_CONDITION}) predicts {n} at AUC "
        f"{e.point:.3f}"
        + (f" [{e.lo:.3f}, {e.hi:.3f}]" if e.ci_finite else "")
        + f" on {e.n_clusters} independent test subjects ({prov}) -- at or above "
          f"{CONFOUND_AUC_MAX:.2f}, i.e. a larger-than-Cohen's-medium effect for a "
          f"label that contains no pathology at all"
        for n, e, prov in ext_fail
    ]
    ext_ev = "; ".join(f"{n}={_fmt_est_inline(e)} [{prov}]"
                       for n, e, prov in ext_decisive)

    if ext_why:
        # Decided before the stage-5 branch, and independently of it. This is a
        # direct, well-powered measurement of the exact quantity C6 names, so it
        # settles the criterion on its own -- in the FAILING direction only.
        detail = "; ".join(ext_why) + (
            " -- measured directly rather than inferred from a clinical cohort's "
            "4-7 subject fold. Phase is functioning as an acquisition fingerprint, "
            "and no diagnostic number computed from the same channel can be "
            "separated from it")
        if targets:
            worst_name, worst = max(targets, key=lambda kv: kv[1].point)
            detail += (f". The stage-5 confound control on {cohort} itself puts "
                       f"{worst_name} at AUC {worst.point:.3f}")
        add("confound_not_explanatory", "fail", detail,
            (ext_ev + ("; " + "; ".join(ext_context) if ext_context else ""))
            + "  (these are acquisition-identity AUCs, not cancer AUCs)")
    elif not targets:
        add("confound_not_explanatory", "missing",
            ("no confound-predictability control (stage 5 has not produced one); "
             "receiver_channels alone takes six distinct values across prostate DWI "
             "patients, so this cannot be assumed benign")
            + ("; " + "; ".join(confound_defects) if confound_defects else "")
            + ("; the external measurement(s) available (" + "; ".join(
                f"{n} AUC {e.point:.3f}" for n, e, _ in ext_decisive)
               + ") are below the ceiling, and a low external number cannot stand in "
                 "for a control that was never run on this cohort"
               if ext_decisive else ""),
            ext_ev)
    else:
        worst_name, worst = max(targets, key=lambda kv: kv[1].point)
        ev = "; ".join(f"{n}={_fmt_est_inline(e)}" for n, e in
                       sorted(targets, key=lambda kv: -kv[1].point))
        if ext_ev or ext_context:
            # Shown even when it is below the ceiling, so the reader sees the
            # direct measurement next to the small-fold one it is meant to
            # replace. It is evidence on the record; it decides nothing here.
            ev += ("; external (does not decide this criterion unless it exceeds "
                   f"{CONFOUND_AUC_MAX:.2f}): "
                   + "; ".join(x for x in (ext_ev, "; ".join(ext_context)) if x))
        ev += "  (these are scanner-identity AUCs, not cancer AUCs)"
        # Absolute ceiling AND a margin below the headline. The ceiling alone
        # let a confound at 0.79 pass while the headline was 0.90; the margin
        # alone would let a confound at 0.60 pass while the headline was 0.62.
        # Either pattern is the scanner-fingerprint explanation, so both are
        # refused.
        why = []
        if worst.point >= CONFOUND_AUC_MAX:
            why.append(f"the same input predicts {worst_name} at AUC "
                       f"{worst.point:.3f} (>= {CONFOUND_AUC_MAX:.2f}, which is "
                       f"Cohen's medium effect on the AUC scale)")
        if headline.ok and (headline.point - worst.point) < CONFOUND_HEADLINE_MARGIN:
            why.append(f"{worst_name} (AUC {worst.point:.3f}) is only "
                       f"{headline.point - worst.point:+.3f} below the headline "
                       f"({headline.point:.3f}), less than the "
                       f"{CONFOUND_HEADLINE_MARGIN:.2f} margin required to say the "
                       f"diagnosis is the better-supported reading of the input")
        # C6's second arm differences the confound against the headline, so it
        # carries the same shared-test-set requirement as C4/C5/C7/C8. It is a
        # gate on the PASS only: the ceiling arm needs no headline at all, and a
        # confound above it still FAILS however either side was scored.
        conf_mismatch = ([] if not headline.ok else
                         _test_set_mismatch(headline.est, worst,
                                            f"the {worst_name} confound control"))
        if conf_mismatch:
            ev += f"; TEST SETS DIFFER: {'; '.join(conf_mismatch)}"
        # C6 is a MAXIMUM over targets, so anything that removes a target from
        # the set, or that makes the surviving maximum unreadable, can only
        # lower it. Both are therefore gates on the pass and not on the fail:
        # a confound that is demonstrably high still FAILS however it was
        # measured, but "the worst confound is low" requires that every confound
        # run stage 5 wrote was actually scored, and that the winner's own
        # interval could have shown otherwise.
        underpowered = [] if why else _control_power_reasons(worst)
        if why:
            add("confound_not_explanatory", "fail",
                "; ".join(why) + " -- phase is functioning as a scanner "
                "fingerprint and any tumour signal is not separable from it", ev,
                quotes=headline, differenced=True,
                test_set_verified=not conf_mismatch)
        elif confound_defects:
            add("confound_not_explanatory", "missing",
                f"the highest confound that could be SCORED is {worst_name} at AUC "
                f"{worst.point:.3f}, but stage 5 also wrote confound run(s) this "
                f"report could not read, and C6 is a maximum over targets -- an "
                f"unscored target can only lower it: " + "; ".join(confound_defects),
                ev, quotes=headline, differenced=True)
        elif conf_mismatch:
            add("confound_not_explanatory", "missing",
                f"the most predictable confound is {worst_name} at AUC "
                f"{worst.point:.3f}, but it was not scored on the test set this "
                f"report headlines ({headline.fingerprint}): "
                + "; ".join(conf_mismatch)
                + ". The margin below the headline is then a difference between two "
                  "test sets, so the criterion cannot be met on it",
                ev, quotes=headline, differenced=True)
        elif underpowered:
            add("confound_not_explanatory", "missing",
                f"the most predictable confound is {worst_name} at AUC "
                f"{worst.point:.3f}, but that estimate cannot establish it is "
                f"genuinely low: " + "; ".join(underpowered), ev,
                quotes=headline, differenced=True, test_set_verified=True)
        else:
            add("confound_not_explanatory", "pass",
                f"the most predictable confound is {worst_name} at AUC "
                f"{worst.point:.3f}, below {CONFOUND_AUC_MAX:.2f}"
                + (f" and {headline.point - worst.point:.3f} below the headline"
                   if headline.ok else ""), ev,
                quotes=headline, differenced=True, test_set_verified=True)

    # --- C8: headline vs the EMPIRICAL permutation null ---------------------
    # C1 compares the headline to a hard-coded 0.500. That is the chance level
    # only if the pipeline itself is unbiased -- which is a claim about this
    # code, not about the data, and is exactly what the label-permutation
    # control measures. Where stage 5 produced that null, it is the reference
    # distribution the headline must beat.
    #
    # Same scoring level on both sides: `headline` is the slice-level AUC, and
    # each permutation replicate's `test.auc` is the slice-level AUC of a model
    # trained on permuted labels. Across seeds the WORST (lowest) headline is
    # used, matching the rest of this module: a result that only holds for the
    # luckiest seed has not been demonstrated.
    null = controls.permutation_null(cohort, HEADLINE_CONDITION)
    if not null:
        add("beats_permutation_null", "missing",
            "no label-permutation control, so there is no empirical null to "
            "compare the headline against; 0.500 is an assumption, not a "
            "measurement", quotes=headline, differenced=True)
    elif not headline.ok:
        add("beats_permutation_null", "missing",
            f"no {HEADLINE_CONDITION} AUC to compare against the permutation null",
            differenced=True)
    else:
        n_rep = int(null["n"])
        obs = min(headline.per_seed) if headline.per_seed else headline.point
        n_ge = sum(1 for a in null["aucs"] if a >= obs - 1e-12)
        # Phipson & Smyth (2010): (r + 1) / (n + 1) is the unbiased estimator of
        # a Monte-Carlo permutation p-value. r/n would report p = 0 from a
        # finite number of replicates, which is never true.
        p_emp = (1.0 + n_ge) / (1.0 + n_rep)
        ev = (f"headline {obs:.3f} ("
              + ("worst of seeds " + ", ".join(f"{v:.3f}" for v in headline.per_seed)
                 if headline.per_seed else "single estimate")
              + f") vs null n={n_rep} distinct replicate(s)"
              + (f" (of {null.get('n_files')} file(s); {null['n_duplicates']} were "
                 f"copies of a replicate already counted)"
                 if null.get("n_duplicates") else "")
              + f", mean {null['mean']:.3f}, range "
                f"[{null['min']:.3f}, {null['max']:.3f}]; {n_ge} replicate(s) "
                f"reach or exceed it")
        # Same fold on both sides, or the comparison is meaningless: a null
        # computed on a different (say, easier or smaller) test set is not the
        # null distribution of THIS headline.
        #
        # This used to read `if a is not None and b is not None and a != b`,
        # i.e. it was checked only where both sides happened to declare a count.
        # That is a guard defeated by DELETING information: the identical null,
        # computed on the identical wrong fold, was REFUSED when the payload
        # carried `n_clusters` and ACCEPTED when the key was absent. It now
        # shares `_test_set_mismatch` with C4/C5/C6/C7, which treats an
        # undeclared size as a reason rather than as consent.
        fold_mismatch = _test_set_mismatch(headline.est, perm,
                                           "the permutation null")
        if fold_mismatch:
            add("beats_permutation_null", "missing",
                "the permutation null cannot be shown to have been computed on the "
                "test set this report headlines (" + "; ".join(fold_mismatch)
                + "), so it is not the null distribution of this number and the "
                  "comparison would be meaningless", ev,
                quotes=headline, differenced=True)
        elif n_rep < MIN_PERM_REPLICATES_FOR_P:
            add("beats_permutation_null", "missing",
                f"only {n_rep} DISTINCT permutation replicate(s)"
                + (f" ({null['n_duplicates']} further file(s) held a replicate "
                   f"already counted; a copied replicate is not a second draw from "
                   f"the null and does not shrink (r+1)/(n+1))"
                   if null.get("n_duplicates") else "")
                + f": the smallest p-value "
                  f"reachable is 1/{n_rep + 1} = {1.0 / (n_rep + 1):.3f}, which "
                  f"cannot be below {ALPHA}. At least "
                  f"{MIN_PERM_REPLICATES_FOR_P} replicates are needed before this "
                  f"criterion can be evaluated at all", ev,
                quotes=headline, differenced=True, test_set_verified=True)
        elif p_emp < ALPHA:
            add("beats_permutation_null", "pass",
                f"P(permuted-label AUC >= headline) = ({n_ge}+1)/({n_rep}+1) = "
                f"{p_emp:.4g} < {ALPHA}", ev,
                quotes=headline, differenced=True, test_set_verified=True)
        else:
            add("beats_permutation_null", "fail",
                f"P(permuted-label AUC >= headline) = ({n_ge}+1)/({n_rep}+1) = "
                f"{p_emp:.4g}, not below {ALPHA}: the headline is inside the "
                f"distribution this pipeline produces from labels that carry no "
                f"information, whatever its interval says about 0.500", ev,
                quotes=headline, differenced=True, test_set_verified=True)

    # --- Provenance gate on the stage-4 criteria ----------------------------
    # C1 and C2 are the only criteria read out of statistics.json. If stage 4
    # summarised a different set of predictions from the ones this report
    # headlines -- in practice: stage 4 describes the 7-subject official split
    # while the headline is the pooled out-of-fold cross-validation -- then a
    # PASS on either would be a pass on a fold nobody is claiming a result for.
    # Cap them at MISSING and name the reason. FAIL is left alone: a criterion
    # that could not be met even on the more flattering fold is not rescued by
    # the provenance being wrong.
    if stats_scheme_mismatch:
        for c in crit:
            if c.key not in ("phase_above_chance", "phase_beats_magnitude"):
                continue
            if c.status == "pass":
                c.status = "missing"
                c.detail = (
                    f"{c.detail} -- but this was computed on a different set of "
                    f"predictions from the one this report headlines: "
                    f"{stats_scheme_mismatch}. A criterion met on a fold the report "
                    f"does not headline is reported as unevaluated, not as met. "
                    f"Re-run stage 4 against the pooled out-of-fold predictions.")
            elif c.status == "fail":
                # Left FAILING -- provenance being wrong is not a reason to
                # upgrade a criterion -- but the reader is told which fold the
                # number came from, so a FAIL on 7 subjects is not read as a
                # FAIL on the 67 the headline covers.
                c.detail = (
                    f"{c.detail} (computed on the official split, not on the pooled "
                    f"out-of-fold predictions headlined here: {stats_scheme_mismatch}. "
                    f"The criterion is left FAILING because a wrong provenance is not a "
                    f"reason to upgrade it; re-run stage 4 on the pooled predictions to "
                    f"decide it on the fold this report claims.)")

    # --- Coverage gate on the cross-validated criteria ----------------------
    # Same shape, different question. Above: does stage 4 describe the SAME
    # predictions? Here: do those predictions cover the cohort the report says
    # they cover? A cross-validated headline missing a fold is an estimate over
    # the subjects whose fold survived, and folds fail for data-dependent
    # reasons, so PASS is not available -- but FAIL still is, because losing
    # coverage cannot manufacture a failure that a complete sweep would not have
    # had, and refusing to report it would hide a negative result.
    if coverage_defect:
        for c in crit:
            if c.key not in ("phase_above_chance", "phase_beats_magnitude"):
                continue
            if c.status == "pass":
                c.status = "missing"
                c.detail = (
                    f"{c.detail} -- but the predictions it was decided on do not cover "
                    f"the cross-validation this report claims: {coverage_defect}. "
                    f"Folds fail for data-dependent reasons, so the subjects that "
                    f"survived are not a random subsample of the cohort and this is "
                    f"reported as unevaluated, not as met. Re-run the missing fold(s).")
            elif c.status == "fail":
                c.detail = (
                    f"{c.detail} (decided on an INCOMPLETE cross-validation: "
                    f"{coverage_defect}. Left FAILING, because incomplete coverage is "
                    f"not a reason to upgrade a criterion.)")

    # --- Coverage gate on the OTHER side of the comparison -------------------
    # Both gates above ask about the headline condition. C2 is a comparison, and
    # a comparison has two sides: stage 4's `folds_a` / `folds_b` say which fold
    # set each condition was pooled over, and the size guard upstream only ever
    # ran on HEADLINE_CONDITION, so the reference condition's coverage was
    # checked by nothing at all. phase over five folds versus magnitude over
    # four is a comparison between two test sets whatever its p-value says.
    # Caps C2 alone -- C1 does not involve the reference condition -- and, like
    # the other two gates, PASS -> MISSING only.
    if comparison_mismatch:
        for c in crit:
            if c.key != "phase_beats_magnitude":
                continue
            if c.status == "pass":
                c.status = "missing"
                c.detail = (
                    f"{c.detail} -- but the two conditions were not compared on one "
                    f"test set: {comparison_mismatch}. A difference between two "
                    f"conditions scored on two different sets of subjects is partly a "
                    f"difference of test set, so this is reported as unevaluated, not "
                    f"as met.")
            elif c.status == "fail":
                c.detail = (
                    f"{c.detail} (and the two conditions were not compared on one test "
                    f"set: {comparison_mismatch}. Left FAILING, because a wrong "
                    f"provenance is not a reason to upgrade a criterion.)")

    # --- Combine ------------------------------------------------------------
    order = {key: i for i, (_, key, _) in enumerate(CRITERIA_ORDER)}
    crit.sort(key=lambda c: order.get(c.key, 99))
    fails = [c for c in crit if c.status == "fail"]
    missing = [c for c in crit if c.status == "missing"]
    if fails:
        verdict = "NOT SUPPORTED"
        reason = "failed criteria: " + ", ".join(f"{c.code} ({c.key})" for c in fails)
    elif missing:
        verdict = "INCONCLUSIVE"
        reason = "could not evaluate: " + ", ".join(f"{c.code} ({c.key})" for c in missing)
    else:
        verdict = "SUPPORTED"
        reason = f"all {len(crit)} criteria met"

    cv = CohortVerdict(cohort=cohort, verdict=verdict, criteria=crit, reason=reason,
                       is_primary=bool(is_primary), headlines=dict(quoted),
                       aggregates=list(aggregates))
    _assert_verdict_consistent(cv)
    return cv


def _assert_verdict_consistent(cv: CohortVerdict) -> None:
    """
    Guard the one property the whole report rests on.

    If this ever raises, the report is not written at all. A crash is a far
    better outcome than a RESULTS.md that says SUPPORTED next to a failed
    control.
    """
    if cv.verdict == "SUPPORTED" and (cv.failing or cv.missing):
        raise AssertionError(
            f"verdict logic violated for {cv.cohort}: SUPPORTED with "
            f"{len(cv.failing)} failing and {len(cv.missing)} unevaluated criteria"
        )
    if cv.failing and cv.verdict != "NOT SUPPORTED":
        raise AssertionError(
            f"verdict logic violated for {cv.cohort}: {len(cv.failing)} failing "
            f"criteria but verdict is {cv.verdict}"
        )
    # Family-wise: only the pre-registered primary cohort can carry a
    # confirmatory verdict. C0 already forces this by being MISSING everywhere
    # else, so reaching here means C0 was dropped or mis-set -- a code bug that
    # must not be allowed to print a positive headline.
    if cv.verdict == "SUPPORTED" and not cv.is_primary:
        raise AssertionError(
            f"verdict logic violated for {cv.cohort}: SUPPORTED on a cohort that "
            f"is not the pre-registered primary ({PRIMARY_COHORT})"
        )
    codes = {c.code for c in cv.criteria}
    required = {code for code, _, _ in CRITERIA_ORDER}
    if cv.verdict == "SUPPORTED" and codes != required:
        raise AssertionError(
            f"verdict logic violated for {cv.cohort}: SUPPORTED while criteria "
            f"{sorted(required - codes)} were never evaluated at all"
        )
    if cv.verdict not in {"SUPPORTED", "NOT SUPPORTED", "INCONCLUSIVE"}:
        raise AssertionError(f"unknown verdict {cv.verdict!r}")

    # ---- ONE HEADLINE PER (COHORT, CONDITION, LEVEL) ---------------------
    # Three invariants an adversarial round handed us for free, from a single
    # real report in which `Stats.estimate` scoped run selection to one split
    # family and then overwrote the point estimate from an across-seeds row that
    # spanned two. All three are checked here, on every verdict, for every
    # cohort -- not only on SUPPORTED ones, because a report that quotes two
    # different headlines for one cohort is broken whatever it concludes.

    # (1) A point estimate may never fall outside its own interval. That
    #     combination can only arise from reading the point and the bounds out
    #     of different scopes; the report printed "headline 0.794 [0.600,
    #     0.780]" in the C4 evidence line before this existed.
    for key, (point, lo, hi) in sorted(cv.headlines.items()):
        if math.isnan(point) or math.isnan(lo) or math.isnan(hi):
            continue
        if not math.isfinite(lo) or not math.isfinite(hi):
            continue
        if not (lo - 1e-9) <= point <= (hi + 1e-9):
            raise AssertionError(
                f"verdict logic violated for {cv.cohort}: the headline {key} is "
                f"{point:.6g} with interval [{lo:.6g}, {hi:.6g}] -- the point "
                f"estimate lies outside its own interval, so the two were read "
                f"from different scopes")

    # (2) No two criteria may quote different headlines for the same
    #     (cohort, condition, level). C4 quoted 0.794 while C8 quoted 0.691 in
    #     the same file, and neither could see the other.
    for c in cv.criteria:
        if not c.headline_key:
            continue
        if c.headline_key not in cv.headlines:
            raise AssertionError(
                f"verdict logic violated for {cv.cohort}: {c.code} quotes headline "
                f"{c.headline_key!r}, which the verdict does not record at all")
        want = cv.headlines[c.headline_key][0]
        if math.isnan(want) != math.isnan(c.headline_point) or (
                not math.isnan(want) and abs(want - c.headline_point) > 1e-9):
            raise AssertionError(
                f"verdict logic violated for {cv.cohort}: {c.code} quotes "
                f"{c.headline_key} = {c.headline_point:.6g} while another criterion "
                f"quotes {want:.6g} for the same cohort, condition and level")

    # (3) Any criterion that DIFFERENCES a control against the headline must
    #     have verified that the two share a test set before it may PASS. This
    #     is the check C8 already applied and C4/C5/C6/C7 did not, so one report
    #     could refuse a comparison as meaningless for one criterion and pass
    #     four others on exactly it.
    for c in cv.criteria:
        if c.status == "pass" and c.differenced and not c.test_set_verified:
            raise AssertionError(
                f"verdict logic violated for {cv.cohort}: {c.code} PASSES on a "
                f"difference against the headline without checking that the two "
                f"were scored on the same test set")

    # (4) NO AGGREGATE MAY REPORT A FINGERPRINT ITS MEMBERS DO NOT SHARE. The
    #     fifth adversarial round lived entirely one level below (3): the
    #     criteria dutifully asked their control for its test-set fingerprint,
    #     and the control was a POOL that had adopted one member's. An envelope
    #     over two background runs, one on 560 slices / 70 subjects and one on
    #     40 slices / 200 subjects, answered "560 slices / 70 subjects" and C4
    #     and C7 passed on it; a permutation null of which one replicate in
    #     twenty was on the headline's fold answered with that replicate's, and
    #     C8 passed on it. Checked here for every aggregate any criterion was
    #     decided on, on every verdict, whatever it concluded.
    for record in cv.aggregates:
        _assert_aggregate_fingerprint(record)


# ==========================================================================
# Sample-size caveats -- attached to every number, never footnoted
# ==========================================================================

def caveat_text(n_clusters: Optional[int], n_pos_clusters: Optional[int],
                n: Optional[int], n_pos: Optional[int]) -> str:
    """A compact, unavoidable statement of how thin the evidence is."""
    bits = []
    if n_clusters is not None:
        s = f"{n_clusters} test subject" + ("s" if n_clusters != 1 else "")
        if n_pos_clusters is not None:
            s += f", {n_pos_clusters} positive"
        bits.append(s)
    if n is not None:
        s = f"{n} slices"
        if n_pos is not None:
            s += f" ({n_pos} positive)"
        bits.append(s)
    core = "; ".join(bits) if bits else "sample size unknown"
    if n_clusters is not None and n_clusters < MIN_CLUSTERS_RELIABLE:
        core += (f" -- interval rests on {n_clusters} independent units, "
                 "hypothesis-generating only")
    return core


def fmt_estimate_cell(est: Estimate, pooled: Optional[PooledPredictions] = None) -> str:
    """Markdown cell: value, interval, provenance, and the caveat, together."""
    if not est.ok:
        return "not available"
    txt = _fmt_est_inline(est)
    nc = est.n_clusters if est.n_clusters is not None else (pooled.n_clusters if pooled else None)
    npc = pooled.n_pos_clusters if pooled else None
    n = est.n if est.n is not None else (pooled.n if pooled else None)
    npos = est.n_pos if est.n_pos is not None else (pooled.n_pos if pooled else None)
    txt += f"<br><sub>{caveat_text(nc, npc, n, npos)}</sub>"
    if est.per_seed:
        seeds = ", ".join(f"{v:.3f}" for v in est.per_seed)
        txt += f"<br><sub>per-seed: {seeds}</sub>"
    if est.n_boot_valid is not None and est.n_boot_valid > 0:
        txt += f"<br><sub>{est.n_boot_valid} usable bootstrap resamples</sub>"
    txt += f"<br><sub>source: {est.source}"
    if est.note:
        txt += f"; {est.note}"
    txt += "</sub>"
    return txt


# ==========================================================================
# Figures
# ==========================================================================

def _save(fig, path: Path, dpi: int = 300) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    logger.info("figure -> %s", path)
    return path


AUC_MISMATCH_TOL = 0.02


def multi_condition_size_note(pooled_by_cond: Dict[str, PooledPredictions]) -> str:
    """
    The SAMPLE SIZE annotation for a figure that overlays several conditions.

    One string per figure was the defect: the figure was labelled with the size
    and evaluation scheme of whichever condition came first, and every other
    curve or bar in it silently inherited that label. When one condition lost a
    fold -- which happened on the real tree mid-sweep, magnitude over 5 folds and
    70 subjects sharing an axis with phase over 4 folds and 58 -- the annotation
    was numerically wrong for every curve but one.

    So: one line per condition whenever the conditions disagree about ANY of
    (fold set, subject count, slice count, scheme), and the shared one-liner only
    when they genuinely agree.
    """
    items = [(c, p) for c, p in pooled_by_cond.items() if p is not None]
    if not items:
        return "sample size unknown"
    items.sort(key=lambda cp: CONDITION_ORDER.index(cp[0])
               if cp[0] in CONDITION_ORDER else 99)
    keys = {(tuple(p.folds), p.n_clusters, p.n_pos_clusters, p.n, p.n_pos,
             p.scheme, p.coverage_complete) for _, p in items}
    if len(keys) == 1:
        _, p = items[0]
        note = "SAMPLE SIZE: " + caveat_text(p.n_clusters, p.n_pos_clusters, p.n, p.n_pos)
        if p.is_pooled_oof and not p.coverage_complete:
            note += f"\nINCOMPLETE CROSS-VALIDATION: {p.coverage_text}"
        return note
    lines = ["SAMPLE SIZE DIFFERS BY CONDITION -- these curves are not on the same "
             "test set:"]
    for cond, p in items:
        extra = f", folds {list(p.folds)}" if p.is_pooled_oof else ""
        if p.is_pooled_oof and not p.coverage_complete:
            extra += " (INCOMPLETE)"
        lines.append(f"  {cond}: "
                     + caveat_text(p.n_clusters, p.n_pos_clusters, p.n, p.n_pos)
                     + extra)
    return "\n".join(lines)


def fig_roc(cohort: str, pooled_by_cond: Dict[str, PooledPredictions],
            bands: Dict[str, dict], out_dir: Path, dpi: int,
            mismatches: Optional[List[str]] = None,
            notes: Optional[List[str]] = None) -> Optional[Path]:
    """
    ROC per cohort, conditions overlaid, with cluster-bootstrap CI bands.

    The curve is always drawn from the stage-3 predictions; the AUC printed in
    the legend prefers stage 4's number. Those two must agree, and if they do
    not -- different pooling, a stale statistics.json, a different split -- the
    figure would quietly show a curve that contradicts its own label. So the
    disagreement is measured, printed on the figure, and returned for the report
    rather than left for a reader to notice.

    Every curve carries its OWN subject count in the legend, and the annotation
    below the axes states each condition's separately whenever they differ. The
    figure used to be annotated once, with the size and scheme of whichever
    condition sorted first; when the conditions covered different fold sets that
    label was numerically wrong for every other curve on the plot.

    `notes`, when passed, receives the annotation text -- so a test can assert
    what the figure actually says rather than re-deriving it.
    """
    from sklearn.metrics import roc_auc_score, roc_curve

    usable = {c: p for c, p in pooled_by_cond.items()
              if p is not None and len(np.unique(p.labels)) > 1}
    if not usable:
        return None

    fig, ax = plt.subplots(figsize=(6.6, 6.4))
    grid = np.linspace(0, 1, 201)
    any_pooled = next(iter(usable.values()))
    schemes = sorted({p.scheme_label for p in usable.values()})
    local_mismatch: List[str] = []
    for cond in CONDITION_ORDER:
        p = usable.get(cond)
        if p is None:
            continue
        color = CONDITION_COLOR.get(cond, "#666666")
        # THE CURVE IS THE ESTIMAND THE TABLE REPORTS. `p.probs` is the
        # seed-averaged (ensemble) vector; its ROC sits above every seed that
        # went into it, so drawing it under a stage-4 point that is the MEAN of
        # the per-seed AUCs put the two 0.02-0.03 apart and tripped the
        # disagreement guard below on every clinical cohort. Where a per-seed
        # view exists, the curve drawn is the vertical average over seeds --
        # whose area IS that mean -- and the ensemble is not drawn at all.
        # Where it does not, the ensemble is drawn and the legend says so.
        sm = seed_mean_auc(p)
        avg_tpr = seed_mean_roc(p, grid) if sm else None
        if sm is not None and avg_tpr is not None:
            fpr, tpr = grid, avg_tpr
            empirical, n_seeds = sm
        else:
            fpr, tpr, _ = roc_curve(p.labels, p.probs)
            empirical = float(roc_auc_score(p.labels, p.probs))
            n_seeds = 0
        b = bands.get(cond, {})
        auc_txt = f"{empirical:.3f} (drawn)"
        pt = _num(b.get("point"))
        if not math.isnan(pt):
            auc_txt = f"{pt:.3f}"
            if not math.isnan(_num(b.get("lo"))):
                auc_txt += f" [{b['lo']:.3f}, {b['hi']:.3f}]"
            if abs(pt - empirical) > AUC_MISMATCH_TOL and b.get("_from_s04"):
                local_mismatch.append(
                    f"{cond}: curve drawn from stage-3 predictions has AUC "
                    f"{empirical:.3f}, statistics.json reports {pt:.3f}")
        # n on the curve's own label, so a legend entry cannot be read against
        # the wrong sample size even at a glance -- and WHICH curve it is, so a
        # seed-averaged ROC and a single-model one are never mistaken for each
        # other at a glance either.
        ax.plot(fpr, tpr, color=color, lw=2.0,
                label=f"{cond}  AUC {auc_txt}  (n={p.n_clusters} subj"
                      + (f", {len(p.folds)} folds" if p.is_pooled_oof else "")
                      + (f", mean of {n_seeds} seeds" if n_seeds
                         else (f", {len(p.seeds)}-seed ENSEMBLE"
                               if len(p.seeds) > 1 else ""))
                      + ")")
        if b.get("tpr_lo") is not None:
            ax.fill_between(grid, b["tpr_lo"], b["tpr_hi"], color=color, alpha=0.15, lw=0)

    ax.plot([0, 1], [0, 1], ls="--", lw=1.0, color="#999999", label="chance")
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.set_title(f"{cohort} -- test-set ROC\n"
                 + (schemes[0] if len(schemes) == 1
                    else "CONDITIONS USE DIFFERENT SCHEMES: " + " | ".join(schemes))
                 + f"\nshaded: 95% band, {any_pooled.cluster_unit}-clustered bootstrap",
                 fontsize=9 if len(schemes) > 1 else 11)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.grid(alpha=0.25, lw=0.5)
    note = multi_condition_size_note(usable)
    if local_mismatch:
        note += "\nINCONSISTENT: " + "; ".join(local_mismatch)
        if mismatches is not None:
            mismatches.extend(f"{cohort} ROC -- {m}" for m in local_mismatch)
    if notes is not None:
        notes.append(note)
    ax.text(0.0, -0.11, note, transform=ax.transAxes, fontsize=8,
            color="#B71C1C", va="top")
    return _save(fig, out_dir / f"fig1_roc_{cohort}.png", dpi)


def fig_auc_bars(rows: List[dict], out_dir: Path, dpi: int,
                 titles: Optional[List[str]] = None) -> Optional[Path]:
    """
    AUC bars with subject-clustered intervals, one group per cohort.

    Each panel used to be titled with `sub[0]["caveat"]` -- the sample size of
    whichever condition sorted first -- and the other bars in the panel inherited
    it. Where the conditions differ, each bar now carries its own count under its
    own tick label and the panel title says the sizes differ instead of asserting
    one of them.
    """
    rows = [r for r in rows if r["est"].ok]
    if not rows:
        return None
    cohorts = sorted({r["cohort"] for r in rows})
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.3 * len(cohorts), 5.0),
                             squeeze=False, sharey=True)
    for ax, cohort in zip(axes[0], cohorts):
        sub = [r for r in rows if r["cohort"] == cohort]
        sub.sort(key=lambda r: CONDITION_ORDER.index(r["condition"])
                 if r["condition"] in CONDITION_ORDER else 99)
        x = np.arange(len(sub))
        pts = [r["est"].point for r in sub]
        lo = [max(0.0, r["est"].point - r["est"].lo) if r["est"].has_ci else 0.0 for r in sub]
        hi = [max(0.0, r["est"].hi - r["est"].point) if r["est"].has_ci else 0.0 for r in sub]
        colors = [CONDITION_COLOR.get(r["condition"], "#777777") for r in sub]
        ax.bar(x, pts, color=colors, alpha=0.85, width=0.62)
        ax.errorbar(x, pts, yerr=[lo, hi], fmt="none", ecolor="#222222",
                    elinewidth=1.4, capsize=6)
        for xi, r in zip(x, sub):
            if r["est"].per_seed:
                ax.scatter([xi] * len(r["est"].per_seed), r["est"].per_seed,
                           s=14, color="#111111", zorder=5, alpha=0.7)
        ax.axhline(0.5, ls="--", lw=1.0, color="#999999")
        ax.set_xticks(x)
        caveats = [str(r.get("caveat", "")) for r in sub]
        shared = len(set(caveats)) == 1
        # When the bars in a panel rest on different test sets, each tick label
        # carries its own count. A single panel-level caveat would be true of one
        # bar and false of the rest.
        ax.set_xticklabels(
            [r["condition"] if shared
             else f"{r['condition']}\nn={(r.get('pooled').n_clusters if r.get('pooled') else r['est'].n_clusters)}"
             for r in sub],
            fontsize=9 if shared else 8)
        if shared:
            cav = caveats[0]
        else:
            cav = ("SAMPLE SIZE DIFFERS BY CONDITION -- the bars are not on the same "
                   "test set; each bar's n is under its own label")
        # wrap: the caveat carries the n-subjects/n-positives warning and is long
        # enough to overrun its own axes and collide with the neighbouring panel.
        cav = "\n".join(textwrap.wrap(cav, width=46)) if cav else ""
        title = f"{cohort}\n{cav}"
        ax.set_title(title, fontsize=9)
        if titles is not None:
            titles.append(title)
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("Test AUC (95% CI, subject-clustered)")
    fig.suptitle("Test AUC by cohort and condition -- dots are individual seeds",
                 fontsize=12, y=1.10)
    fig.text(0.01, -0.04,
             "Every interval is clustered on the split-enforcement subject id. "
             "Bars whose interval crosses 0.50 are indistinguishable from chance.",
             fontsize=8, color="#B71C1C")
    return _save(fig, out_dir / "fig2_auc_bars.png", dpi)


def fig_training_curves(runs: Sequence[dict], out_dir: Path, dpi: int) -> Optional[Path]:
    full = [r for r in runs if str(r.get("region", "full")) == "full" and r.get("history")]
    if not full:
        return None
    cohorts = sorted({str(r["cohort"]) for r in full})
    fig, axes = plt.subplots(2, len(cohorts), figsize=(4.4 * len(cohorts), 7.0),
                             squeeze=False)
    for j, cohort in enumerate(cohorts):
        ax_loss, ax_auc = axes[0][j], axes[1][j]
        for cond in CONDITION_ORDER:
            group = [r for r in full if str(r["cohort"]) == cohort
                     and str(r["condition"]) == cond]
            if not group:
                continue
            color = CONDITION_COLOR.get(cond, "#777777")
            for r in group:
                ep = [h["epoch"] + 1 for h in r["history"]]
                ax_loss.plot(ep, [h.get("train_loss", np.nan) for h in r["history"]],
                             color=color, alpha=0.75, lw=1.3)
                ax_loss.plot(ep, [h.get("val_loss", np.nan) for h in r["history"]],
                             color=color, alpha=0.75, lw=1.3, ls="--")
                ax_auc.plot(ep, [h.get("val_auc", np.nan) for h in r["history"]],
                            color=color, alpha=0.75, lw=1.3)
                be = int(r.get("best_epoch", -1))
                if 0 <= be < len(r["history"]):
                    ax_auc.scatter([be + 1], [r["history"][be].get("val_auc", np.nan)],
                                   color=color, s=26, zorder=5, edgecolor="white", lw=0.6)
        ax_loss.set_title(f"{cohort} -- loss (solid train, dashed val)", fontsize=10)
        ax_loss.set_xlabel("epoch")
        ax_loss.set_ylabel("loss")
        ax_loss.grid(alpha=0.25, lw=0.5)
        ax_auc.axhline(0.5, ls="--", lw=1.0, color="#999999")
        ax_auc.set_title(f"{cohort} -- validation AUC (dot = selected epoch)", fontsize=10)
        ax_auc.set_xlabel("epoch")
        ax_auc.set_ylabel("val AUC")
        ax_auc.set_ylim(0.0, 1.0)
        ax_auc.grid(alpha=0.25, lw=0.5)
    handles = [Line2D([0], [0], color=CONDITION_COLOR[c], lw=2, label=c)
               for c in CONDITION_ORDER]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    return _save(fig, out_dir / "fig3_training_curves.png", dpi)


def fig_controls_panel(cohort: str, headline: Estimate, controls: Controls,
                       verdict: CohortVerdict, out_dir: Path, dpi: int,
                       pooled: Optional[PooledPredictions],
                       cluster_headline: Optional[Estimate] = None,
                       cluster_level: str = "") -> Optional[Path]:
    """
    The headline next to every falsification control, on one axis.

    This is the figure a sceptical reviewer looks at first, so the criteria and
    their pass/fail state are printed on it rather than left to the caption.

    BOTH headline levels are drawn when the cluster-level one exists. The
    slice-level bar is the one the controls are compared against (stage 5 scores
    them at the slice level); the cluster-level bar is the one C1 is decided on.
    Showing only the first is how a slice interval that excludes chance ends up
    read as the criterion having been met.
    """
    entries: List[Tuple[str, Estimate]] = [
        (f"HEADLINE\n{HEADLINE_CONDITION}, real labels, full image\n"
         f"(slice level -- what the controls are compared against)", headline),
    ]
    if cluster_headline is not None and cluster_headline.ok:
        entries.append(
            (f"HEADLINE, {cluster_level or 'cluster'} level\n"
             f"one score per subject -- THIS is what C1 reads", cluster_headline))
    for canon, label in (("permutation", "CONTROL\nlabels permuted (null range)"),
                         ("background", "CONTROL\nbackground only (no anatomy)"),
                         ("scramble", "CONTROL\nphase scrambled within mask"),
                         ("acquisition", "CONTROL\nacquisition held out (worse direction)")):
        entries.append((label, controls.estimate(cohort, canon)))

    for name, e in sorted(controls.confound_targets(cohort),
                          key=lambda kv: -kv[1].point)[:4]:
        entries.append((f"CONFOUND (not a cancer AUC)\npredict {name} from the same input", e))

    # Height has to cover BOTH columns: the bars on the left and the criteria
    # text on the right, which has its own line count. Sizing off the bars
    # alone is what made the last criterion collide with the footer.
    text_lines = sum(2 + len(_wrap(c.rule, 58).split("\n"))
                     + len(_wrap(c.detail, 62).split("\n"))
                     for c in verdict.criteria)
    height = max(1.05 * len(entries) + 3.6, 0.30 * text_lines + 3.0)
    fig, (ax, ax_txt) = plt.subplots(
        1, 2, figsize=(14.5, height),
        gridspec_kw={"width_ratios": [1.35, 1.0]},
    )
    y = np.arange(len(entries))[::-1]
    for yi, (label, est) in zip(y, entries):
        if not est.ok:
            ax.text(0.02, yi, "NOT AVAILABLE -- criterion cannot pass",
                    va="center", fontsize=9, color=STATUS_COLOR["missing"], weight="bold")
            continue
        is_head = label.startswith("HEADLINE")
        color = "#C44E52" if is_head else "#4C72B0"
        ax.barh(yi, est.point, color=color, alpha=0.85, height=0.55)
        if est.ci_finite:
            ax.errorbar(est.point, yi, xerr=[[max(0.0, est.point - est.lo)],
                                             [max(0.0, est.hi - est.point)]],
                        fmt="none", ecolor="#222222", elinewidth=1.4, capsize=5)
            ax.text(min(1.0, max(est.hi, est.point) + 0.02), yi,
                    f"{est.point:.3f} [{est.lo:.3f}, {est.hi:.3f}]",
                    va="center", fontsize=8.5)
        elif est.has_ci:
            # A non-finite bound cannot be drawn and must not be silently
            # dropped either: an unbounded interval is the shape that used to
            # satisfy "the CI contains chance" for free.
            ax.text(min(1.0, est.point + 0.02), yi,
                    f"{est.point:.3f} [{est.lo}, {est.hi}] -- interval not finite",
                    va="center", fontsize=8.5, color=STATUS_COLOR["missing"])
        else:
            ax.text(min(1.0, est.point + 0.02), yi, f"{est.point:.3f} (no CI)",
                    va="center", fontsize=8.5)

    ax.axvline(0.5, ls="--", lw=1.2, color="#444444")
    ax.set_ylim(-0.9, len(entries) - 0.25)
    ax.text(0.5, -0.75, "chance", fontsize=8, color="#444444", ha="center")
    ax.set_yticks(y)
    ax.set_yticklabels([e[0] for e in entries], fontsize=8.5)
    ax.set_xlim(0.0, 1.15)
    ax.set_xticks(np.arange(0, 1.01, 0.1))
    ax.set_xlabel("AUC")
    ax.grid(axis="x", alpha=0.25, lw=0.5)
    ax.set_title(f"{cohort} -- headline vs falsification controls", fontsize=12)

    ax_txt.axis("off")
    vcolor = {"SUPPORTED": STATUS_COLOR["pass"], "NOT SUPPORTED": STATUS_COLOR["fail"],
              "INCONCLUSIVE": STATUS_COLOR["missing"]}[verdict.verdict]
    ax_txt.text(0.0, 1.0, f"VERDICT: {verdict.verdict}", fontsize=14,
                weight="bold", color=vcolor, va="top")
    ytxt = 0.945
    ax_txt.text(0.0, ytxt, _wrap(verdict.reason, 74), fontsize=9, va="top")
    ytxt -= 0.030 * (_wrap(verdict.reason, 74).count("\n") + 1) + 0.015
    if pooled is not None:
        # Sample size goes ABOVE the criteria, not in a footer: it qualifies
        # every number on this figure, so it is read before them, not after.
        cav = _wrap("SAMPLE SIZE: " + caveat_text(pooled.n_clusters, pooled.n_pos_clusters,
                                                  pooled.n, pooled.n_pos), 70)
        ax_txt.text(0.0, ytxt, cav, fontsize=8.5, color="#B71C1C",
                    va="top", weight="bold")
        ytxt -= 0.030 * (cav.count("\n") + 1) + 0.020
    for c in verdict.criteria:
        mark = {"pass": "PASS", "fail": "FAIL", "missing": "MISSING"}[c.status]
        rule = _wrap(c.rule, 58)
        detail = _wrap(c.detail, 62)
        ax_txt.text(0.0, ytxt, f"{c.code} {mark}", fontsize=9, weight="bold",
                    color=STATUS_COLOR[c.status], va="top")
        ax_txt.text(0.19, ytxt, rule, fontsize=8, va="top")
        ytxt -= 0.030 + 0.024 * rule.count("\n")
        ax_txt.text(0.19, ytxt, detail, fontsize=7.5,
                    va="top", color="#444444", style="italic")
        ytxt -= 0.040 + 0.022 * (detail.count("\n") + 1)
    fig.tight_layout()
    return _save(fig, out_dir / f"fig4_controls_{cohort}.png", dpi)


def _wrap(text: str, width: int) -> str:
    import textwrap
    return "\n".join(textwrap.wrap(str(text), width)) or ""


def class_names(cohort: str, index: Optional[pd.DataFrame] = None,
                sep: str = "\n") -> Tuple[str, str]:
    """
    (name of label 1, name of label 0) for this cohort -- never assumed.

    A confound cohort's label is receive-coil count or pulse sequence, so
    captioning its panels "tumour-positive" would be a false statement printed
    on a figure. The registry supplies the wording; the stage-2 index's own
    `label_name` column is quoted alongside it when present, because it is what
    stage 2 actually wrote next to the rows, and a disagreement between the two
    is logged rather than silently resolved.

    `sep` joins the registry wording to the stage-2 wording: a newline on a
    figure, a space in a markdown table cell (where a newline would end the row).
    """
    spec = confound_spec(cohort)
    pos, neg = ((spec.positive_name, spec.negative_name) if spec else
                (f"{DIAGNOSTIC_LABEL_SHORT} (label 1)", "no tumour (label 0)"))
    if index is not None and {"label", "label_name"} <= set(index.columns):
        try:
            got = {int(k): sorted({str(x) for x in v.dropna().unique()})
                   for k, v in index.groupby("label")["label_name"]}
            fpos = "/".join(got.get(1, [])) or None
            fneg = "/".join(got.get(0, [])) or None
            # Compare with whitespace removed: ">=16" and ">= 16 receive coils"
            # are the same statement written two ways, and warning about that
            # would train the reader to ignore the warning.
            def _agrees(short: Optional[str], long: str) -> bool:
                return bool(short) and _slug(short) in _slug(long)
            if spec and fpos and fneg and not (_agrees(fpos, pos) and _agrees(fneg, neg)):
                logger.warning(
                    "%s: stage-2 index names the classes %r/%r; the s06 registry says "
                    "%r/%r. Both are shown wherever the classes are named.",
                    cohort, fpos, fneg, pos, neg)
            if fpos:
                pos = f"{pos}{sep}[stage-2 index: {fpos}]" if spec else fpos
            if fneg:
                neg = f"{neg}{sep}[stage-2 index: {fneg}]" if spec else fneg
        except Exception as exc:  # noqa: BLE001
            logger.warning("could not read class names for %s: %s", cohort, exc)
    return pos, neg


def fig_qualitative(cohort: str, cache_dir: Path, out_dir: Path, dpi: int,
                    pooled: Optional[PooledPredictions] = None) -> Optional[Path]:
    """
    Magnitude vs phase, drawn from the cache exactly as the network saw it.

    This replaces the old manuscript Figure 5, which was produced by a separate
    script (`make_real_figure5.py`) that transposed the prostate k-space axes
    before reconstructing -- so the panel captioned "slice 30" was in fact slice
    0 of diffusion volume 30 -- and that treated the radial breast acquisition
    as if it were Cartesian. Here we read pipeline_out/cache/{cohort}.h5
    directly, with no transpose, no re-reconstruction, and no re-normalization
    beyond the exact channel construction in s03_train.CacheSliceDataset:

        magnitude channel : zscore(mag)
        phase channels    : sin(phase), cos(phase)

    The cache row index of each panel is printed on the figure so any claim
    about it can be checked against the index CSV in one command. The row labels
    name the cohort's ACTUAL classes: for brain and knee that is coil count and
    pulse sequence, not tumour status.
    """
    import h5py

    h5_path = cache_dir / f"{cohort}.h5"
    idx_path = cache_dir / f"{cohort}_index.csv"
    if not h5_path.exists() or not idx_path.exists():
        logger.warning("qualitative panel skipped for %s: cache not on disk (%s)",
                       cohort, h5_path)
        return None
    try:
        index = pd.read_csv(idx_path, low_memory=False)
    except Exception as exc:  # noqa: BLE001
        logger.warning("qualitative panel skipped for %s: %s", cohort, exc)
        return None

    pool = index[index.get("official_split", "").astype(str) == "test"] \
        if "official_split" in index.columns else index
    if pool.empty:
        pool = index
    if pooled is not None and len(pooled.cache_idx):
        keep = pool["idx"].isin(set(int(v) for v in pooled.cache_idx))
        if keep.any():
            pool = pool[keep]

    pos_name, neg_name = class_names(cohort, index)
    picks: List[Tuple[str, int]] = []
    for lab, name in ((1, pos_name), (0, neg_name)):
        sub = pool[pool["label"] == lab]
        if len(sub):
            picks.append((name, int(sub.iloc[len(sub) // 2]["idx"])))
    if not picks:
        logger.warning("qualitative panel skipped for %s: no labelled test rows", cohort)
        return None

    try:
        fh = h5py.File(h5_path, "r")
    except Exception as exc:  # noqa: BLE001
        # The stage-2 writer holds an exclusive lock while the cache is being
        # built; that is expected, not an error worth aborting the report for.
        logger.warning("qualitative panel skipped for %s: cannot open %s (%s)",
                       cohort, h5_path, exc)
        return None

    spec = confound_spec(cohort)
    header = [
        f"{cohort} -- magnitude vs phase, read directly from "
        f"pipeline_out/cache/{cohort}.h5",
        "arrays are drawn in stored order with no transpose and no re-reconstruction: "
        "this is exactly the network's input",
        (f"ROW LABEL IS {spec.label_long.upper()} -- THERE IS NO TUMOUR ANNOTATION IN "
         f"THIS COHORT" if spec else f"row label is {DIAGNOSTIC_LABEL_LONG}"),
        "(cyan contour = stage-2 body mask; the background-only control trains on its "
        "complement)",
    ]
    header_lines = sum(len(_wrap(h, 118).split("\n")) for h in header)

    with fh:
        # Explicit inch arithmetic instead of tight_layout(). The panels are
        # square images with aspect="equal", and tight_layout resolves the
        # conflict between that constraint and a four-line suptitle by shrinking
        # the axes -- which is how this figure came to render 224x224 images at
        # thumbnail size inside a mostly empty canvas.
        nrows, ncols = len(picks), 4
        panel_in = 2.45                       # per image, square
        left_in, right_in = 1.55, 0.35        # row labels / trailing colorbar
        head_in = 0.20 * header_lines + 0.35
        row_extra_in = 0.62                   # provenance line under each row
        fig_w = left_in + ncols * (panel_in + 0.45) + right_in
        fig_h = head_in + nrows * (panel_in + row_extra_in) + 0.25
        fig, axes = plt.subplots(nrows, ncols, figsize=(fig_w, fig_h), squeeze=False)
        fig.subplots_adjust(
            left=left_in / fig_w, right=1.0 - right_in / fig_w,
            top=1.0 - head_in / fig_h, bottom=0.22 / fig_h,
            wspace=0.42, hspace=row_extra_in / panel_in + 0.10,
        )
        for r, (name, k) in enumerate(picks):
            mag = np.asarray(fh["mag"][k], dtype=np.float32)
            phase = np.asarray(fh["phase"][k], dtype=np.float32)
            mask = np.asarray(fh["mask"][k], dtype=bool)
            mag_z = (mag - mag.mean()) / (mag.std() + 1e-8)
            row = index[index["idx"] == k]
            meta = row.iloc[0].to_dict() if len(row) else {}
            prov = (f"cache row {k}  |  {meta.get('file', '?')}  slice "
                    f"{meta.get('slice', '?')}  |  patient {meta.get('patient_id', '?')}  "
                    f"|  label={meta.get('label', '?')}")

            panels = [
                (mag_z, "gray", None, "magnitude channel\nzscore(mag), as fed to the net"),
                (np.sin(phase), "twilight_shifted", (-1, 1),
                 "phase channel 1\nsin(phase)"),
                (np.cos(phase), "twilight_shifted", (-1, 1),
                 "phase channel 2\ncos(phase)"),
                (phase, "twilight", (-np.pi, np.pi),
                 "raw phase (radians)\nnever min-max scaled"),
            ]
            for c, (img, cmap, lims, title) in enumerate(panels):
                ax = axes[r][c]
                kw = {"cmap": cmap, "interpolation": "nearest", "aspect": "equal"}
                if lims:
                    kw["vmin"], kw["vmax"] = lims
                else:
                    v = float(np.percentile(np.abs(img), 99)) or 1.0
                    kw["vmin"], kw["vmax"] = -v, v
                im = ax.imshow(img, **kw)          # no transpose: array as stored
                ax.contour(mask.astype(float), levels=[0.5], colors="#00E5FF",
                           linewidths=0.6, alpha=0.8)
                ax.set_xticks([])
                ax.set_yticks([])
                if r == 0:
                    ax.set_title(title, fontsize=9)
                fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02).ax.tick_params(labelsize=7)
            axes[r][0].set_ylabel(_wrap(name, 22), fontsize=9, weight="bold")
            axes[r][0].text(0.0, -0.045, prov, transform=axes[r][0].transAxes,
                            fontsize=7.5, color="#333333", va="top")
        fig.suptitle("\n".join(_wrap(h, 118) for h in header),
                     fontsize=10.5, y=1.0, va="top")
        return _save(fig, out_dir / f"fig5_qualitative_{cohort}.png", dpi)


def fig_confound_predictability(rows: List[dict], out_dir: Path,
                                dpi: int) -> Optional[Path]:
    """
    Phase vs magnitude at predicting the ACQUISITION PROPERTY, with
    subject-clustered intervals.

    Everything about this figure is arranged so that a reader who sees only the
    figure cannot read a high bar as a good result. The y-axis is labelled with
    what the label actually is, the title says the interpretation is inverted,
    the region above CONFOUND_AUC_MAX is shaded as the disqualifying zone, and
    the footer states the conclusion in words.

    `rows` are dicts: cohort, condition, est (Estimate), spec (ConfoundCohortSpec
    or None), n_text.
    """
    rows = [r for r in rows if r["est"].ok]
    if not rows:
        return None
    cohorts = sorted({r["cohort"] for r in rows})
    fig, axes = plt.subplots(1, len(cohorts), figsize=(5.0 * len(cohorts), 5.8),
                             squeeze=False, sharey=True)
    for ax, cohort in zip(axes[0], cohorts):
        sub = [r for r in rows if r["cohort"] == cohort]
        sub.sort(key=lambda r: CONDITION_ORDER.index(r["condition"])
                 if r["condition"] in CONDITION_ORDER else 99)
        x = np.arange(len(sub))
        pts = [r["est"].point for r in sub]
        lo = [max(0.0, r["est"].point - r["est"].lo) if r["est"].ci_finite else 0.0
              for r in sub]
        hi = [max(0.0, r["est"].hi - r["est"].point) if r["est"].ci_finite else 0.0
              for r in sub]
        colors = [CONDITION_COLOR.get(r["condition"], "#777777") for r in sub]
        # The disqualifying zone, drawn before the bars so the bars sit on it.
        ax.axhspan(CONFOUND_AUC_MAX, 1.0, color="#B71C1C", alpha=0.07, lw=0)
        ax.axhline(CONFOUND_AUC_MAX, ls=":", lw=1.2, color="#B71C1C")
        ax.text(0.01, CONFOUND_AUC_MAX - 0.012,
                f"C6 ceiling {CONFOUND_AUC_MAX:.2f} -- above this line the input is "
                f"an acquisition fingerprint",
                transform=ax.get_yaxis_transform(),
                fontsize=7.5, color="#B71C1C", va="top")
        ax.bar(x, pts, color=colors, alpha=0.9, width=0.6)
        ax.errorbar(x, pts, yerr=[lo, hi], fmt="none", ecolor="#222222",
                    elinewidth=1.4, capsize=6)
        for xi, r in zip(x, sub):
            # Labels sit above the interval, in the headroom the 1.14 ylim
            # exists to provide, so a bar at 1.000 does not have its own number
            # drawn through it.
            top = max(r["est"].hi, r["est"].point) if r["est"].ci_finite else r["est"].point
            if r["est"].ci_finite:
                ax.text(xi, top + 0.025,
                        f"{r['est'].point:.3f}\n[{r['est'].lo:.3f}, {r['est'].hi:.3f}]",
                        ha="center", va="bottom", fontsize=8)
            else:
                ax.text(xi, top + 0.025, f"{r['est'].point:.3f}",
                        ha="center", va="bottom", fontsize=8)
        ax.axhline(0.5, ls="--", lw=1.0, color="#999999")
        ax.set_xticks(x)
        ax.set_xticklabels([r["condition"] for r in sub], fontsize=10)
        ax.set_xlim(-0.6, len(sub) - 0.4)
        spec = sub[0].get("spec")
        title = f"{cohort}\nlabel = {spec.label_short if spec else 'unknown'}"
        ax.set_title(title + "\n" + "\n".join(textwrap.wrap(sub[0].get("n_text", ""), 44)),
                     fontsize=9)
        ax.set_ylim(0.0, 1.14)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("AUC at predicting the ACQUISITION PROPERTY\n"
                          "(95% CI, subject-clustered)", fontsize=10)
    fig.suptitle(
        "What phase actually predicts: acquisition identity, not pathology\n"
        "READ THIS FIGURE BACKWARDS -- a HIGH bar is evidence AGAINST "
        "'phase carries tumour signal'",
        fontsize=12.5, y=1.06,
    )
    fig.text(
        0.0, -0.10,
        "These cohorts carry NO tumour annotation. The label is the acquisition "
        "property named under each panel, so an AUC here measures how well the input "
        "channel identifies the hardware or the protocol.\n"
        "For a clinical cohort a high phase AUC would be (control-gated) evidence FOR "
        "the hypothesis. Here the interpretation is inverted: the higher the bar, the "
        "more of the phase channel is\nexplained by acquisition identity, and the less "
        "of any clinical number computed from it can be attributed to disease. "
        "Intervals are clustered on subject.",
        fontsize=8.5, color="#B71C1C")
    return _save(fig, out_dir / "fig6_confound_predictability.png", dpi)


def fig_cv_folds(rows: List[dict], out_dir: Path, dpi: int,
                 titles: Optional[List[str]] = None) -> Optional[Path]:
    """
    Per-fold AUC as a dispersion diagnostic, against the pooled out-of-fold
    estimate that is the actual headline.

    The per-fold points are drawn small and grey and the pooled estimate is
    drawn as the interval, because that is their relative status: the folds show
    how much the estimate moves when the test subjects change, and the pooled
    number -- one prediction per subject, every subject tested once -- is the
    estimate being claimed.

    `rows`: cohort, condition, pooled (PooledPredictions), est (Estimate).
    """
    rows = [r for r in rows if r.get("pooled") is not None
            and r["pooled"].is_pooled_oof and r["pooled"].per_fold]
    if not rows:
        return None
    cohorts = sorted({r["cohort"] for r in rows})
    fig, axes = plt.subplots(1, len(cohorts), figsize=(4.6 * len(cohorts), 5.2),
                             squeeze=False, sharey=True)
    for ax, cohort in zip(axes[0], cohorts):
        sub = [r for r in rows if r["cohort"] == cohort]
        sub.sort(key=lambda r: CONDITION_ORDER.index(r["condition"])
                 if r["condition"] in CONDITION_ORDER else 99)
        for xi, r in enumerate(sub):
            p, est = r["pooled"], r["est"]
            color = CONDITION_COLOR.get(r["condition"], "#777777")
            fold_aucs = [f["auc"] for f in p.per_fold if not math.isnan(f["auc"])]
            jitter = np.linspace(-0.16, 0.16, max(len(fold_aucs), 1))
            ax.scatter([xi + j for j in jitter], fold_aucs, s=26, color="#666666",
                       zorder=4, alpha=0.85,
                       label="per-fold (dispersion diagnostic)" if xi == 0 else None)
            if est.ok:
                yerr = ([[max(0.0, est.point - est.lo)], [max(0.0, est.hi - est.point)]]
                        if est.ci_finite else None)
                ax.errorbar([xi], [est.point], yerr=yerr, fmt="o", color=color,
                            markersize=9, elinewidth=2.0, capsize=7, zorder=5,
                            label="pooled out-of-fold (the estimate)" if xi == 0 else None)
                lab = f"{est.point:.3f}"
                top = est.hi if est.ci_finite else est.point
                if est.ci_finite:
                    lab += f"\n[{est.lo:.3f}, {est.hi:.3f}]"
                # Above the interval and centred, not beside it: a label placed
                # to the right lands on the neighbouring condition's fold dots.
                ax.text(xi, min(0.97, top) + 0.02, lab, fontsize=8,
                        ha="center", va="bottom")
        ax.axhline(0.5, ls="--", lw=1.0, color="#999999")
        ax.set_xticks(range(len(sub)))
        keys = {(tuple(r["pooled"].folds), r["pooled"].n_clusters,
                 r["pooled"].n_pos_clusters, r["pooled"].coverage_complete)
                for r in sub}
        same = len(keys) == 1
        ax.set_xticklabels(
            [r["condition"] if same
             else f"{r['condition']}\n{len(r['pooled'].folds)}f/"
                  f"{r['pooled'].n_clusters}subj" for r in sub],
            fontsize=10 if same else 8)
        ax.set_xlim(-0.6, len(sub) - 0.4)
        # The title states the fold set only when every condition in the panel
        # shares it, and claims "each tested exactly once" only when the pooled
        # vectors actually cover the declared design.
        if same:
            p0 = sub[0]["pooled"]
            sub_title = (f"{len(p0.folds)} folds, {p0.n_clusters} subjects "
                         f"({p0.n_pos_clusters} positive), "
                         + ("each tested exactly once" if p0.coverage_complete
                            else f"INCOMPLETE: {p0.coverage_text}"))
        else:
            sub_title = ("conditions cover DIFFERENT fold sets -- "
                         + "; ".join(f"{r['condition']} {list(r['pooled'].folds)}"
                                     for r in sub))
        ax.set_title(f"{cohort}\n{sub_title}", fontsize=9 if same else 8)
        if titles is not None:
            titles.append(f"{cohort}\n{sub_title}")
        ax.set_ylim(0.0, 1.0)
        ax.grid(axis="y", alpha=0.25, lw=0.5)
    axes[0][0].set_ylabel("Test AUC (95% CI, subject-clustered)")
    axes[0][0].legend(loc="lower left", fontsize=8, frameon=True)
    fig.suptitle("Cross-validated clinical results: pooled out-of-fold estimate, "
                 "with the folds it was pooled from", fontsize=12, y=1.04)
    fig.text(0.0, -0.05,
             "The pooled estimate is not the mean of the fold AUCs. It is one AUC over "
             "one prediction per subject, obtained by concatenating the test-block "
             "predictions of folds whose test sets are\ndisjoint -- so it has the power "
             "of the whole cohort and contributes ONE entry to the comparison family. "
             "The fold points show how far the estimate moves when the test subjects "
             "change.",
             fontsize=8, color="#B71C1C")
    return _save(fig, out_dir / "fig7_cv_folds.png", dpi)


# ==========================================================================
# RESULTS.md
# ==========================================================================

CONFOUND_COHORT_LIST_MD = ", ".join(f"`{c}` (label: {s.label_short})"
                                    for c, s in sorted(CONFOUND_COHORTS.items()))

VERDICT_RULES_MD = f"""\
## 1. How the verdict is decided (rules stated before any number)

"Phase carries tumour signal beyond magnitude" is recorded as **SUPPORTED** for
a cohort only if **every one** of the following holds. Each is evaluated
independently and reported with its own pass/fail/missing state.

| # | Criterion | Threshold |
|---|-----------|-----------|
| C0 | Cohort is the pre-registered primary one | `{PRIMARY_COHORT}`. Every other cohort in this report is EXPLORATORY and cannot reach SUPPORTED |
| C1 | Phase AUC is above chance | 95% CI lower bound > 0.500 at the CLUSTER level stage 4 marks `preferred` (one score per subject), on at least {MIN_CLUSTERS_C1} clusters with at least {MIN_CLASS_CLUSTERS_C1} in each class |
| C2 | Phase beats magnitude | DeLong test, Holm-adjusted across the comparison family, p < {ALPHA}, at the level stage 4 marks `preferred` |
| C3 | Label-permutation control is null | AUC within 0.500 +/- {PERM_NULL_TOL:.2f} over at least {MIN_PERMUTATION_REPLICATES} DISTINCT replicates spanning at most {MAX_CONTROL_CI_WIDTH:.2f} AUC, and its range covers 0.500 |
| C4 | Background-only control collapses | at least {BACKGROUND_MARGIN:.2f} AUC below the headline, AND its own 95% CI contains {CONTROL_CHANCE:.2f}, AND that CI is discriminating (see the power floor below) |
| C5 | Acquisition-stratified split preserves the effect | CI lower bound still > 0.500 and at most {ACQ_EROSION_MAX:.2f} AUC lost, in the WORSE of the two split directions, both of which must have been run |
| C6 | Confounds do not explain it | scanner / coil / site predictability from the same input < {CONFOUND_AUC_MAX:.2f} AUC, and at least {CONFOUND_HEADLINE_MARGIN:.2f} AUC below the headline, with every confound run stage 5 wrote actually scored -- and no DIRECTLY MEASURED confound cohort at or above {CONFOUND_AUC_MAX:.2f} on the same input channel |
| C7 | Phase-scramble control collapses | at least {BACKGROUND_MARGIN:.2f} AUC below the headline AND its own 95% CI contains {CONTROL_CHANCE:.2f} AND that CI is discriminating (an effect that survives destroying spatial structure was never spatial) |
| C8 | Headline beats the EMPIRICAL null | P(permuted-label AUC >= headline) < {ALPHA} over the DISTINCT label-permutation replicates, by (r+1)/(n+1); needs at least {MIN_PERM_REPLICATES_FOR_P} of them to be reachable at all |

C1-C6 are the six pre-registered criteria. C7 is added because stage 5
implements a fifth falsification control, and the rule "no SUPPORTED while a
control fails" has to cover every control that ran, not just the ones named in
advance. C0 and C8 are added for the reasons set out immediately below; both
make SUPPORTED harder to reach and neither can make it easier.

Resolution rules, applied in this order:

1. **Any criterion FAILS -> NOT SUPPORTED**, and the failing criteria are named.
2. **Otherwise any criterion is MISSING -> INCONCLUSIVE**, and the missing
   criteria are named. A control that was never run is not a control that
   passed.
3. **Only if every criterion PASSES -> SUPPORTED.**

### Where each number is measured, and why it matters

* **C1 is read at the cluster level, never the slice level.** Slices within a
  subject are correlated, so a slice-level percentile interval is far too
  narrow. On the real prostate DWI test fold (122 slices / 4 patients / 11
  positive slices / 3 positive patients) the slice-level phase AUC is 0.918
  with CI [0.849, 1.000] -- which clears C1 -- while the patient-mean AUC is
  0.778 with CI [0.000, 1.000], which does not. The second number is the honest
  one, and it is the one C1 uses.
* **C1 is withheld entirely below a minimum fold size.** Simulated under a
  complete null, the rule "cluster-bootstrap CI lower bound > 0.500" fires 24.9%
  of the time with 4 clusters (3 positive) against a nominal 2.5%. It reaches
  nominal only once there are at least {MIN_CLASS_CLUSTERS_C1} clusters in each
  class. Below {MIN_CLUSTERS_C1} clusters, or {MIN_CLASS_CLUSTERS_C1} per class,
  C1 is reported MISSING -- not PASS and not FAIL, because neither can be read
  off an interval that wide. **More bootstrap replicates cannot fix this. More
  subjects can.** A cohort can therefore be permanently INCONCLUSIVE on fold
  size alone; that is the correct answer for a fold of four people.
* **C4/C5/C7/C8 read the headline at the SLICE level**, because that is the
  level stage 5 scores every control at. Comparing a patient-level headline
  against a slice-level control would measure the aggregation, not the control.
* **C8 replaces 0.500 as the operative chance level.** 0.500 is the chance level
  only if the whole pipeline is unbiased, which is a claim about this code
  rather than about the data. C3 checks the permutation null is not obviously
  broken (a loose +/-{PERM_NULL_TOL:.2f} band); C8 does the actual test, using
  that null as the reference distribution.

### Which cohorts the criteria apply to at all

Two of the cohorts in this pipeline -- {CONFOUND_COHORT_LIST_MD} --
carry **no diagnostic label**. Their label is an acquisition property (receive-coil
count; pulse sequence), so "phase carries tumour signal" is not a proposition that can
be evaluated on them at all. They are excluded from the verdict table, from the family
of cohorts C0 controls, and from the diagnostic statistics table, and reported in their
own section, where the interpretation is stated to be inverted: a HIGH AUC there is
evidence AGAINST the hypothesis, because it says the input channel encodes the scanner.

Excluding them can only shrink the set of cohorts that could reach SUPPORTED.

### C6 is wired to the direct measurement of the confound

C6 asks whether scanner / coil / site identity is predictable from the same input the
diagnostic claim rests on. Until now it could only be answered from the stage-5 confound
control on a clinical cohort, whose official test fold is 4-7 subjects. The brain
confound cohort measures the same quantity against a label that is nothing but hardware,
on 136 independent test subjects. Where that measurement exists, C6 reads it.

**It is wired in one direction only.** An external confound measured at or above
{CONFOUND_AUC_MAX:.2f} AUC FAILS C6 for a clinical cohort. Nothing about an external
measurement can make C6 pass, stand in for a stage-5 confound control that was never
run, or turn a MISSING into a PASS. The asymmetry is not a convention -- it follows from
what transfers between cohorts. "This input channel encodes acquisition identity" is a
claim about the channel and the physics and does transfer. "This input channel does not
encode acquisition identity here" is a claim about one set of scanners and does not. An
external measurement is also only cited as decisive if it clears the same
{MIN_CLUSTERS_EXTERNAL_CONFOUND}-subject power floor as every other control-based
criterion; below it, the number is displayed and decides nothing.

### Cross-validation: one pooled out-of-fold estimate per cohort

Where a cohort was run over the stage-1 subject-level CV folds (one results
subdirectory per fold), the reported estimate is the **pooled out-of-fold** one:
each subject is in exactly one fold's test set, so concatenating the folds' test
predictions gives one prediction per subject over the whole cohort. That vector is what
the subject-clustered bootstrap and any downstream test operate on.

Five folds are NOT five results. Reporting them separately would be five underpowered
estimates and a five-fold inflation of the comparison family; pooling leaves one
estimate and one entry in that family, at the power of the whole cohort. The property
this rests on -- each subject tested exactly once -- is verified before pooling, at both
the cache-row and the subject level, and pooling is refused with a stated reason if it
does not hold.

The official split remains as a clearly labelled **secondary** analysis with its subject
count printed beside it (7 subjects for prostate_t2, 4 for prostate_dwi), and it cannot
be read as confirming the pooled result. If stage 4's records describe a different set
of predictions from the pooled ones -- which is what happens when stage 4 has not been
re-run after the sweep -- then C1 and C2 are **capped at MISSING** for that cohort, and
the mismatch is named. A criterion met on a fold the report does not headline is
reported as unevaluated, not as met.

### Family-wise error control (C0)

This file reports more than one cohort. Testing each at the same alpha with no
adjustment gives roughly a **25% chance that at least one cohort clears C1 under
a complete null** (measured per-cohort null pass rates: 7.2%, 5.2%, 15.2%).

Rather than adjust C1 across folds that share nothing, **one cohort is
pre-registered as primary: `{PRIMARY_COHORT}`.** It is chosen because it is the
largest cohort (67 patients) and the only one whose reconstruction is validated
against the vendor's own images (r = 0.998). Both facts are fixed independently
of any result, which is what makes this a pre-registration rather than a
selection after the fact.

Every other cohort is **exploratory**: its criteria are computed and printed in
full, and it is labelled in the verdict table below, but it cannot produce a
confirmatory SUPPORTED. If `{PRIMARY_COHORT}` is absent from a run, no other
cohort is promoted in its place and no cohort can be SUPPORTED.

Three asymmetries are deliberate:

* An **anti-conservative test can fail a criterion but never pass one.** Stage 4
  states that its slice-level DeLong p-value is anti-conservative because slices
  within a patient are correlated. If only that level is evaluable, a
  non-significant result still fails C2 (the correct test would be even less
  significant), but a significant one leaves C2 MISSING.
* **Multi-seed and multi-direction results are read at their worst.** The Holm
  p-value used is the largest across seeds, the headline compared against the
  permutation null is the smallest across seeds, and the acquisition-split
  direction used is the weaker of the two. A criterion that holds only for the
  luckiest configuration has not been demonstrated.
* **A control with no interval fails, it does not pass.** C4 and C7 assert a
  positive property of a control -- that it collapsed to chance. Earlier
  versions of this file granted that property to any control whose bootstrap
  block was missing from the stage-5 payload. It is now a FAIL, and the cell
  says so: the control ran, it scored above chance, and nothing on disk shows it
  collapsed.

### The power floor on every control-based criterion

C1 has always refused to be read on a fold too small for a cluster bootstrap.
C3-C7 had no equivalent, and for them the asymmetry runs the wrong way. C1 asks
an interval to EXCLUDE 0.500, so noise costs it a pass. C4 and C7 ask an
interval to CONTAIN 0.500, so **noise buys them one**: `[0.00, 1.00]`,
`[-inf, +inf]`, a bootstrap on two subjects, a bootstrap in which every resample
returned the same number, and a bootstrap block with zero valid resamples all
satisfy "contains 0.500" without having measured anything.

A control estimate must therefore be able to DISCRIMINATE before it can satisfy
any criterion:

* at least **{MIN_CLUSTERS_CONTROL} independent clusters** behind the interval;
* interval width at most **{MAX_CONTROL_CI_WIDTH:.2f} AUC** -- wider than that
  and it covers chance and the headline simultaneously, so it cannot tell them
  apart -- and strictly greater than zero;
* both bounds finite;
* at least **{MIN_BOOT_VALID_CONTROL} resamples** that produced a defined AUC;
* if several runs were pooled, their point estimates must agree to within
  **{MAX_ENVELOPE_SPREAD:.2f} AUC** and each must clear the floor on its own.

An estimate that fails any of these is reported **MISSING with the reason
named** -- not PASS, and not FAIL either, because "we could not measure this" is
neither. The fail paths that do NOT depend on interval width are evaluated
first and are untouched: a control sitting within {BACKGROUND_MARGIN:.2f} AUC of
the headline still FAILS, a control with no interval at all still FAILS, and a
control whose interval lies wholly above or below chance still FAILS. A wide
interval can never launder a control that is demonstrably still predicting.

### Aggregation rules that decide what the thresholds are applied to

Thresholds are only as good as the numbers fed to them. Four aggregation steps
could previously only move a verdict toward SUPPORTED, and each is now
constrained:

* **Envelopes are hulls over repeats, not over anything.** `min(lo)`/`max(hi)`
  across seeds is conservative when the members are the same experiment run
  again. It is not conservative when they are not: one member whose bootstrap
  returned `[0.00, 1.00]` widens the hull until it covers chance. Members must
  agree to within {MAX_ENVELOPE_SPREAD:.2f} AUC and each must clear the power
  floor, or the envelope cannot decide a criterion.
* **Permutation replicates are counted after de-duplication.** C3's minimum
  replicate count and C8's smallest reachable p-value 1/(n+1) are both counts,
  so copying one replicate to twenty filenames turned an unevaluable C8 into
  p = 0.048. Replicates are fingerprinted on their contents with the seed and
  the file path excluded: two files holding the same predictions are one draw
  from the null.
* **"Both split directions" means a direction and its reverse.** It was a count
  of distinct variant STRINGS, so `A2B` and `A_to_B` -- one experiment written
  twice -- satisfied it while the reverse split had never been run. Directions
  are now parsed into ordered (train, test) pairs and the pair must be present.
* **Cohort matching tolerates separator style, not near-misses.** Matching on a
  key with every separator deleted made `prostate_t2` and `prostatet2` the same
  cohort, so a neighbouring cohort's controls could be scored against this one's
  headline. `prostate-t2` and `Prostate T2` still match; `prostatet2` does not.

Every one of these is verified by a named scenario in
`python pipeline/s06_report.py --self-test`, and each scenario is an input that
previously printed SUPPORTED.

There is no code path that can print SUPPORTED while any criterion is failing or
unevaluated: `s06_report._assert_verdict_consistent` re-checks the combination
and raises before the report is written.

A null result is an acceptable and publishable outcome of this study. Nothing in
this report is tuned to avoid one.
"""


def _split_cell(d: Optional[dict]) -> str:
    if not isinstance(d, dict):
        return "n/a"
    return (f"{d.get('n_slices', '?')} slices ({d.get('n_pos', '?')} positive), "
            f"{d.get('n_patients', '?')} patients")


def cohort_table_md(cohort_stats: Dict[str, dict]) -> str:
    if not cohort_stats:
        return "_Stage 1 summary not found; cohort table unavailable._\n"
    lines = [
        "| Cohort | Files | Patients | Subjects | Slices (usable) | Positive slices | "
        "Test fold: subjects / positive subjects / slices / positive slices |",
        "|---|---|---|---|---|---|---|",
    ]
    for cohort, s in cohort_stats.items():
        splits = (s.get("splits", {}) or {}).get("official_split", {}) or {}
        t = splits.get("test", {}) or {}
        rows_usable = s.get("rows_usable", s.get("rows"))
        pos = s.get("positives_usable")
        pct = ""
        if isinstance(pos, (int, float)) and rows_usable:
            pct = f" ({100.0 * pos / rows_usable:.1f}%)"
        test_txt = (f"{t.get('patients', '?')} / {t.get('positive_patients', '?')} / "
                    f"{t.get('rows', '?')} / {t.get('positives', '?')}")
        if isinstance(t.get("patients"), int) and t["patients"] < MIN_CLUSTERS_RELIABLE:
            test_txt += f" **<- {t['patients']} independent units**"
        lines.append(
            f"| {cohort} | {s.get('files', '?')} | {s.get('patients', '?')} | "
            f"{s.get('subjects', '?')} | {rows_usable} | {pos}{pct} | {test_txt} |"
        )
    return "\n".join(lines) + "\n"


def _est_md(est: Estimate) -> str:
    """AUC with its interval, or an explicit statement that there is none."""
    if not est.ok:
        return "not available"
    txt = f"**{est.point:.3f}**"
    if est.ci_finite:
        txt += f" [{est.lo:.3f}, {est.hi:.3f}]"
    elif est.has_ci:
        txt += f" [{est.lo}, {est.hi}] -- interval not finite"
    else:
        txt += " (no interval)"
    return txt


def _paired_block(cohort: str, results_dir: Path) -> dict:
    """
    The stage-7 within-subject paired analysis for this cohort, if it exists.

    s07_paired.py is optional and is owned by another stage. s06 therefore reads
    exactly one field -- the markdown block s07 renders for itself -- and asserts
    nothing about the rest of the payload. If s07 has not run, or its file cannot
    be parsed, the section says so and the report is otherwise unchanged.

    Only the paired cohorts need this: on knee the subject-level machinery in
    stage 4 collapses to 29 positive and 0 negative clusters, so a subject-level
    AUC is undefined by construction and the paired statistic is the only correct
    one.
    """
    root = OUT_ROOT_GUESS(results_dir)
    path = find_first_existing([
        root / "paired" / f"{cohort}_paired.json",
        results_dir / "paired" / f"{cohort}_paired.json",
        results_dir.parent / "paired" / f"{cohort}_paired.json",
    ])
    if path is None:
        return {"paired_markdown": "", "paired_path": ""}
    payload = load_json(path)
    md = (payload or {}).get("markdown")
    if not isinstance(md, str) or not md.strip():
        logger.warning("%s: %s has no `markdown` block; paired analysis not rendered",
                       cohort, path)
        return {"paired_markdown": "", "paired_path": str(path)}
    return {"paired_markdown": md, "paired_path": str(path)}


def _delta_ci_text(dci: Optional[dict]) -> str:
    """The paired interval, rendered inline next to the point difference."""
    if not dci or not dci.get("ok"):
        return ""
    return f", 95% CI [{dci['lo']:+.3f}, {dci['hi']:+.3f}]"


def _confound_ordering_claim(d: float, dci: Optional[dict]) -> str:
    """
    The sentence that reads the phase-vs-magnitude ordering on a confound cohort.

    Gated on the PAIRED interval, not on the sign of the point estimate. The
    absolute level is the finding on these cohorts and is asserted either way;
    the ORDERING is only asserted when the data separate it from zero. A tie is
    reported as a tie, because on this cohort "phase beats magnitude" is the
    claim the paper's mechanism argument rests on and it must not be produced by
    rounding.
    """
    if math.isnan(d):
        return ("The phase-versus-magnitude ordering could not be computed here; "
                "only the absolute level above is being claimed.")
    if dci and dci.get("ok") and not dci.get("excludes_zero"):
        return (f"The phase-minus-magnitude difference is {d:+.3f} with a paired "
                f"subject-clustered 95% CI of [{dci['lo']:+.3f}, {dci['hi']:+.3f}], "
                f"which INCLUDES ZERO: on this cohort the two channels predict the "
                f"acquisition property about equally well, and no ordering between "
                f"them is claimed. The finding here is the LEVEL, not the ranking -- "
                f"phase alone identifies the acquisition far above chance, and that "
                f"is what contaminates any diagnostic score computed from it.")
    if d > 0:
        return ("Phase is the BETTER predictor of the acquisition property, which is "
                "the finding: the information phase carries beyond magnitude is at "
                "least in part the identity of the acquisition.")
    return ("Phase does not beat magnitude here; the acquisition signal is present "
            "in both channels.")


def _confound_fold_text(p: Optional[PooledPredictions],
                        spec: ConfoundCohortSpec) -> str:
    """
    Test-fold size, phrased correctly for the cohort's design.

    `caveat_text` counts "subjects with at least one positive slice", which on an
    UNPAIRED cohort is the number of positive subjects and on a PAIRED one is
    every subject in the fold. Printing "29 test subjects, 29 positive" for the
    paired cohort would read as a fold with no negative subjects, which is the
    opposite of what a paired design means.
    """
    if p is None:
        return "unknown"
    if not spec.paired:
        return caveat_text(p.n_clusters, p.n_pos_clusters, p.n, p.n_pos)
    return (f"{p.n_clusters} test subjects, each supplying BOTH classes; "
            f"{p.n} slices ({p.n_pos} positive)")


def _ceiling_note(per_cond: Dict[str, Estimate]) -> str:
    """A near-perfect AUC is a fact about the task, and is said out loud."""
    at_ceiling = sorted(c for c, e in per_cond.items() if e.ok and e.point >= 0.99)
    if not at_ceiling:
        return ""
    return (f"The {', '.join(at_ceiling)} channel(s) separate the classes essentially "
            f"perfectly (AUC >= 0.99). At that ceiling the phase-vs-magnitude "
            f"difference is not interpretable -- both channels have saturated -- so the "
            f"comparison of interest here is not which is larger but that the property "
            f"is trivially readable at all.")


def write_confound_section(A, ctx: dict) -> None:
    """
    Section 4: the cohorts whose label is an acquisition property.

    Written as its own section, with its own interpretation rule stated before
    the first number, because these AUCs read the opposite way round from every
    other AUC in the file. A reader who lands here from the table of contents
    must not be able to reach a number before reaching the inversion.
    """
    A("## 4. Confound cohorts -- what phase predicts when the label is the scanner")
    A("")
    cohorts = ctx.get("confound_cohorts") or []
    if not cohorts:
        A("_No confound cohort was run. The mechanism claim -- that phase encodes "
          "acquisition identity -- is then supported only by the stage-5 confound "
          "control on the clinical cohorts, whose test folds hold 4-7 subjects._")
        A("")
        return

    A("### 4a. How to read this section (the interpretation is INVERTED)")
    A("")
    A("**These cohorts have no tumour label, and they never will.** The label is an "
      "acquisition property -- how many receive coils were used, which pulse sequence "
      "was played. There is no pathology anywhere in the target variable.")
    A("")
    A("That inverts the reading of every number below:")
    A("")
    A("| | A clinical cohort (sections 2, 5, 7) | A confound cohort (this section) |")
    A("|---|---|---|")
    A("| What the label is | tumour present on this slice | receive-coil count; pulse "
      "sequence |")
    A("| A HIGH phase AUC means | the channel may carry disease signal -- **if and only "
      "if** every falsification control also passes | the channel carries **acquisition "
      "identity** |")
    A("| A HIGH phase AUC is therefore | weak, control-gated evidence **FOR** the "
      "hypothesis | direct evidence **AGAINST** it |")
    A("| A LOW phase AUC would be | evidence against the hypothesis | reassuring -- the "
      "channel would not be a scanner fingerprint |")
    A("")
    A(f"**So a phase AUC of, say, 0.89 in the table below is not a good result. It is "
      f"the bad result.** It says that a network reading the phase channel can identify "
      f"the hardware that produced the image, with no access to any clinical "
      f"information whatsoever. Every diagnostic AUC computed from the same channel is "
      f"then contaminated by a variable that is at least this predictable, and the "
      f"burden of separating the two falls on the controls -- which is exactly what "
      f"criterion C6 is, and why the ceiling it enforces is "
      f"{CONFOUND_AUC_MAX:.2f}.")
    A("")
    A("The comparison that matters here is **phase vs magnitude at predicting the "
      "acquisition property**. If phase does this better than magnitude does, then the "
      "extra information phase carries over magnitude -- the very thing the study "
      "hypothesised was disease signal -- is at least partly acquisition identity.")
    A("")

    A("### 4b. Phase vs magnitude at predicting the acquisition property")
    A("")
    A("| Cohort | What the label actually is | Condition | AUC at predicting THAT label "
      "(95% CI, subject-clustered) | Test fold | Design |")
    A("|---|---|---|---|---|---|")
    for cohort in cohorts:
        c = (ctx.get("confound_context") or {}).get(cohort)
        if not c:
            continue
        spec = c["spec"]
        p = c["pooled"]
        fold = _confound_fold_text(p, spec)
        design = "PAIRED -- every subject supplies both classes" if spec.paired \
            else "UNPAIRED -- each subject supplies exactly one class"
        first = True
        for cond in CONDITION_ORDER:
            est = c["per_cond"].get(cond)
            if est is None or not est.ok:
                continue
            A(f"| {cohort if first else ''} | "
              f"{spec.label_long if first else ''} | {cond} | {_est_md(est)} | "
              f"{fold if first else ''} | {design if first else ''} |")
            first = False
    A("")
    A("Every interval above is a subject-clustered bootstrap over the pooled test "
      "predictions, computed by s06 with the same machinery as every other interval in "
      "this file. **None of these is a diagnostic AUC.** They are not comparable with "
      "any number in sections 2, 5 or 7 and they are deliberately never drawn on the "
      "same axis as one.")
    A("")

    for cohort in cohorts:
        c = (ctx.get("confound_context") or {}).get(cohort)
        if not c:
            continue
        spec = c["spec"]
        A(f"#### {cohort}: label = {spec.label_long}")
        A("")
        if c.get("label_target"):
            A(f"Stage 2 wrote `label_target = {c['label_target']}` next to every row of "
              f"this cohort, and the positive class is `{c['positive_name']}` against "
              f"`{c['negative_name']}`. The heading above is that column, not a "
              f"description of it.")
            A("")
        A(spec.design_note)
        A("")
        head = c["per_cond"].get(HEADLINE_CONDITION)
        ref = c["per_cond"].get(REFERENCE_CONDITION)
        d = c.get("delta", float("nan"))
        if head is not None and head.ok:
            A(f"**{HEADLINE_CONDITION} predicts {spec.label_short} at AUC "
              f"{head.point:.3f}"
              + (f" [{head.lo:.3f}, {head.hi:.3f}]" if head.ci_finite else "")
              + (f", against {ref.point:.3f} for {REFERENCE_CONDITION}"
                 if (ref is not None and ref.ok) else "")
              + (f" (difference {d:+.3f}{_delta_ci_text(c.get('delta_ci'))})"
                 if not math.isnan(d) else "")
              + ".** "
              + _confound_ordering_claim(d, c.get("delta_ci")))
            A("")
        ceiling = _ceiling_note(c["per_cond"])
        if ceiling:
            A(ceiling)
            A("")
        A(spec.why_it_matters)
        A("")
        A(("**Feeds criterion C6.** " if spec.feeds_c6
           else "**Does not feed criterion C6.** ") + spec.c6_note)
        A("")
        if spec.paired:
            A(f"##### {cohort}: within-subject paired analysis (stage 7)")
            A("")
            if c.get("paired_markdown"):
                A(f"_Rendered by `pipeline/s07_paired.py` from "
                  f"`{c.get('paired_path')}`; s06 reproduces its markdown block "
                  f"verbatim and computes none of it._")
                A("")
                A(c["paired_markdown"])
                A("")
            else:
                A("_Stage 7 has not produced a paired analysis for this cohort. That "
                  "matters here specifically: on a paired design the subject-level "
                  "machinery used everywhere else collapses every test subject into "
                  "the positive class, so a subject-level AUC is undefined by "
                  "construction and the AUCs above are slice-level only. Run "
                  "`pipeline/s07_paired.py` for the within-subject statistic._")
                A("")
    A("These cohorts appear in no verdict table, produce no verdict.json `verdict` "
      "field, and are excluded from the family of cohorts criterion C0 controls. They "
      "answer a different question from the one the criteria are written for.")
    A("")


def write_cv_section(A, ctx: dict) -> None:
    """
    Section 5: cross-validated clinical results.

    Headline = the pooled out-of-fold estimate. Per-fold estimates are a
    subordinate dispersion diagnostic, and the official split is a clearly
    labelled secondary analysis carrying its own subject count.
    """
    A("## 5. Cross-validated clinical results")
    A("")
    cv_rows = ctx.get("cv_rows") or []
    fold_rows = ctx.get("fold_rows") or []
    if not cv_rows and not fold_rows:
        A("_No cross-validation folds were found on disk (the sweep writes one "
          "results subdirectory per fold, `<cohort>_cv<k>`). Clinical numbers in this "
          "report therefore come from the official split, whose test folds hold 7 "
          "subjects (prostate_t2) and 4 subjects (prostate_dwi) -- below the "
          f"{MIN_CLUSTERS_C1}-subject floor at which criterion C1 can be read at all._")
        A("")
        return

    A("### 5a. Headline: the pooled out-of-fold estimate")
    A("")
    if not cv_rows:
        A("**No cohort could be pooled out-of-fold.** Folds are present on disk but "
          "pooling was refused for every one of them; the reason is in the degraded "
          "banner at the top of this file. The per-fold numbers in 5b are a diagnostic "
          "and are not a result: each rests on a fraction of the cohort, and none of "
          "them is used to decide any criterion.")
        A("")
    # The design claim and the coverage reality are stated as two separate
    # sentences, and the second one is computed. Section 5a used to assert "one
    # prediction per subject over the whole cohort" unconditionally, while the
    # table beneath it could be showing a condition that lost a fold.
    incomplete = [r for r in cv_rows if not r["pooled"].coverage_complete]
    A("Every subject appears in the test fold of exactly one CV fold, so concatenating "
      "the folds' test-block predictions gives **one prediction per subject over the "
      "whole cohort**. That single vector is what is scored below, and it is what the "
      "subject-clustered bootstrap runs on.")
    A("")
    if incomplete:
        A("> **That is the design. It is NOT what is on disk for every row below.** "
          "The following conditions were pooled over fewer folds than the "
          "cross-validation declares, so for them the sentence above is false and the "
          "estimate covers only the subjects whose folds survived. A fold fails for "
          "data-dependent reasons -- a single-class test block, an out-of-memory on "
          "the largest fold -- so the survivors are not a random subsample, and every "
          "criterion decided on these rows is capped at MISSING:")
        A(">")
        for r in incomplete:
            p = r["pooled"]
            A(f"> - **{r['cohort']} / {r['condition']}**: pooled folds {p.folds} of "
              f"{p.expected_folds or 'an unknown declared set'} -- {p.coverage_text}. "
              f"{p.coverage_defect}."
              + (f" (design read from {p.coverage_source})" if p.coverage_source else ""))
        A(">")
        A("")
    else:
        A(f"_Checked, not assumed: every pooled row below covers the fold set the "
          f"cross-validation declares._")
        A("")
    mm = ctx.get("fold_set_mismatch") or {}
    if mm:
        A("> **The conditions in this table were not scored on the same test set.** "
          "Stage 4 refuses a paired test between two conditions pooled over different "
          "fold sets, because the difference between them is then partly a difference "
          "of test set rather than of model. The same applies to reading the rows below "
          "against one another, and to every figure that overlays them:")
        A(">")
        for cohort, why in sorted(mm.items()):
            A(f"> - **{cohort}**: {why}.")
        A(">")
        A("> Criterion C2 is capped at MISSING for those cohorts.")
        A("")
    A("This is not the mean of the per-fold AUCs, and the distinction is the point of "
      "doing it this way:")
    A("")
    A("- **Full power.** The estimate rests on every subject in the cohort, not on a "
      "fifth of them.")
    A("- **One estimate, one test.** Reporting five folds as five results would be five "
      "underpowered estimates AND a five-fold inflation of the comparison family. "
      "Pooling leaves exactly one entry per cohort in that family.")
    A("- **Each subject counted once.** s06 verifies that the folds' test sets are "
      "disjoint at both the cache-row and the subject level before pooling; if they "
      "are not, pooling is refused and the reason appears in the degraded banner, "
      "because a subject counted twice narrows every interval below.")
    A("")
    A("**One caveat, stated because it cuts against the headline rather than for it.** "
      "Each fold's predictions come from a separately trained model, and those models "
      "are not calibrated to one another. AUC is rank-based, so pooling scores from "
      "different models can move the pooled number relative to the fold-wise mean in "
      "either direction -- and where a fold's scores are shifted relative to another's, "
      "the pooled AUC is typically the LOWER of the two. So a pooled estimate below the "
      "average fold AUC (compare 5a with 5b) is partly a calibration artefact and not "
      "necessarily a worse model. It is reported this way regardless, because the "
      "alternative -- averaging five fold AUCs -- has no interval anybody can defend "
      "and no single population it is an estimate of. Where the fold-wise mean is the "
      "HIGHER number, that difference is not evidence of anything.")
    A("")
    if cv_rows:
        A("| Cohort | Condition | Folds pooled / declared | Pooled out-of-fold AUC "
          "(95% CI, subject-clustered) | Pooled test set | Coverage |")
        A("|---|---|---|---|---|---|")
        for r in cv_rows:
            p, est = r["pooled"], r["est"]
            declared = (f" of {len(p.expected_folds)}" if p.expected_folds else "")
            cover = ("complete" if p.coverage_complete
                     else f"**INCOMPLETE -- {p.coverage_text}**")
            A(f"| {r['cohort']} | {r['condition']} | {len(p.folds)}{declared} "
              f"({', '.join(str(f) for f in p.folds)}) | {_est_md(est)} | "
              f"{caveat_text(p.n_clusters, p.n_pos_clusters, p.n, p.n_pos)} | "
              f"{cover} |")
        A("")

    A("### 5b. Per-fold estimates -- a dispersion diagnostic, not five results")
    A("")
    A("These are shown so the reader can see how far the estimate moves when the test "
      "subjects change. They are **not** independent findings and no criterion is "
      "evaluated on any of them: each rests on roughly a fifth of the cohort, and "
      "treating them as several experiments is precisely the family-wise inflation "
      "section 5a exists to avoid. They are printed even when pooling was refused, "
      "because in that case they are the only numbers there are -- and they are still "
      "not a result.")
    A("")
    A("| Cohort | Condition | Fold | Fold test AUC | Fold test set | Seeds |")
    A("|---|---|---|---|---|---|")
    for r in fold_rows:
        auc = "n/a (single-class fold)" if math.isnan(r["auc"]) else f"{r['auc']:.3f}"
        A(f"| {r['cohort']} | {r['condition']} | {r['fold']} | {auc} | "
          f"{caveat_text(r['n_clusters'], r['n_pos_clusters'], r['n'], r['n_pos'])} | "
          f"{r['seeds']} |")
    A("")

    A("### 5c. Official split -- SECONDARY analysis, not confirmatory")
    A("")
    official = ctx.get("official_rows") or []
    if not official:
        A("_The official-split runs are not present in this results tree, so there is "
          "no secondary analysis to report._")
        A("")
    else:
        A("The official split is retained because it is the split the source datasets "
          "ship with. It is **not** the headline and cannot be read as confirming it: "
          "its test fold is a handful of subjects, and its subject count is printed "
          "next to every number below for that reason.")
        A("")
        A("| Cohort | Condition | Official-split AUC (95% CI) | Official test fold | "
          "Status |")
        A("|---|---|---|---|---|")
        for r in official:
            p, est = r["pooled"], r["est"]
            flag = ("**SECONDARY -- below the "
                    f"{MIN_CLUSTERS_C1}-subject floor for any confirmatory reading**"
                    if p.n_clusters < MIN_CLUSTERS_C1 else "SECONDARY")
            A(f"| {r['cohort']} | {r['condition']} | {_est_md(est)} | "
              f"{caveat_text(p.n_clusters, p.n_pos_clusters, p.n, p.n_pos)} | {flag} |")
        A("")

    mismatch = ctx.get("scheme_mismatch") or {}
    if mismatch:
        A("> **Stage 4 has not been re-run against the pooled predictions.** For the "
          "cohorts below, `statistics.json` summarises a different set of predictions "
          "from the one headlined in 5a, so criteria C1 and C2 are capped at MISSING "
          "for them: a criterion met on a fold this report does not headline is "
          "reported as unevaluated, not as met.")
        A(">")
        for cohort, why in sorted(mismatch.items()):
            A(f"> - **{cohort}**: {why}.")
        A(">")
        A("> Re-run `pipeline/s04_stats.py` against the pooled out-of-fold predictions "
          "to make C1 and C2 evaluable. DeLong has no fallback in s06 and is not "
          "invented here.")
        A("")


def write_results_md(path: Path, ctx: dict) -> Path:
    L: List[str] = []
    A = L.append

    A("# PhaseDx -- RESULTS")
    A("")
    A(f"_Generated {time.strftime('%Y-%m-%d %H:%M:%S')} by `pipeline/s06_report.py`._")
    A("")
    A("**Question under test:** does MRI phase carry tumour signal beyond what the "
      "magnitude image already provides?")
    A("")
    A("**Dominant failure mode this report is built to catch:** phase is an "
      "excellent scanner / coil / protocol fingerprint. Receiver-channel count "
      "alone takes six distinct values (14/16/20/24/26/30) across prostate DWI "
      "patients, and it is spread across two institutions. A classifier that "
      "reads phase can therefore score well by identifying the scanner rather "
      "than the tumour. The controls below exist to separate those two "
      "explanations, and the verdict is withheld unless they do.")
    A("")

    # --- Provenance -------------------------------------------------------
    A("### Inputs used")
    A("")
    A("| Input | Path | State |")
    A("|---|---|---|")
    for label, p, ok, note in ctx["inputs"]:
        A(f"| {label} | `{p}` | {'present' if ok else '**MISSING** -- ' + note} |")
    A("")

    if ctx["degraded"]:
        A("> **DEGRADED REPORT.** " + " ".join(ctx["degraded"]))
        A("")

    A(VERDICT_RULES_MD)
    A("")

    # --- Verdict ----------------------------------------------------------
    A("## 2. Verdict")
    A("")
    confounds = ctx.get("confound_cohorts") or []
    if confounds:
        # Said before the table, not after it. The table below is the diagnostic
        # verdict; a reader must know what is NOT in it before reading it.
        A(f"> **The table below covers the cohorts that have a diagnostic label, and "
          f"only those.** "
          + ", ".join(f"`{c}`" for c in confounds)
          + (" carries" if len(confounds) == 1 else " carry")
          + " no tumour annotation -- the label is an acquisition property -- "
            "so no criterion here applies"
          + (" to it" if len(confounds) == 1 else " to them")
          + " and no verdict on the phase hypothesis is drawn from"
          + (" it" if len(confounds) == 1 else " them")
          + ". They are reported in section 4, where a HIGH AUC means the opposite of "
            "what it would mean here.")
        A("")
    if not ctx["verdicts"]:
        # Distinguish "nothing ran" from "everything that ran was a confound
        # cohort". The second is a legitimate state of this pipeline -- the
        # mechanism cohorts run first -- and telling the reader to go run
        # run_all.py would be wrong.
        if confounds:
            A(f"**No cohort with a diagnostic label was found.** The only stage-3 "
              f"results present are for "
              + ", ".join(f"`{c}`" for c in confounds)
              + ", which carry no tumour annotation. That is not a null result on the "
                "phase hypothesis -- the hypothesis was not tested. Section 4 reports "
                "what these cohorts do measure.")
        else:
            A("**No cohort could be evaluated.** No stage-3 run JSONs were found, so "
              "there is nothing to report. Run `pipeline/run_all.py` first.")
        A("")
    if ctx["verdicts"]:
        # The role column is here, in the verdict table itself, rather than only
        # in the prose: a reader who skips section 1 must still be unable to
        # mistake an exploratory cohort's result for a confirmatory one.
        A("| Cohort | Role | Verdict | Decided on |")
        A("|---|---|---|---|")
        for cv in ctx["verdicts"]:
            role = ("**PRIMARY** (pre-registered)" if cv.is_primary
                    else "exploratory -- cannot be SUPPORTED")
            A(f"| {cv.cohort} | {role} | **{cv.verdict}** | {cv.reason} |")
        A("")
        if not ctx.get("primary_present", True):
            A(f"> The pre-registered primary cohort (`{ctx['primary_cohort']}`) is "
              f"not present in this run. No other cohort is promoted in its "
              f"place, so every cohort above is exploratory and C0 is MISSING "
              f"throughout.")
            A("")
    for cv in ctx["verdicts"]:
        A(f"### {cv.cohort}: **{cv.verdict}**")
        A("")
        A(f"Role: **{cv.role}**. {cv.reason}")
        A("")
        lvl = (ctx.get("c1_levels") or {}).get(cv.cohort)
        if lvl:
            A(f"C1 is decided on the `{lvl}` estimate (one score per subject). "
              f"The slice-level AUC is reported in section 7 and is NOT what the "
              f"criterion reads, because slices within a subject are correlated.")
            A("")
        A("| # | Criterion | State | What was actually observed |")
        A("|---|---|---|---|")
        for c in cv.criteria:
            state = {"pass": "PASS", "fail": "**FAIL**", "missing": "_MISSING_"}[c.status]
            ev = f"<br><sub>{c.evidence}</sub>" if c.evidence else ""
            A(f"| {c.code} | {c.rule} | {state} | {c.detail}{ev} |")
        A("")
        pooled = ctx["pooled"].get((cv.cohort, HEADLINE_CONDITION))
        if pooled is not None:
            A(f"> **Sample size for every number in this section:** "
              f"{caveat_text(pooled.n_clusters, pooled.n_pos_clusters, pooled.n, pooled.n_pos)}. "
              f"Intervals are clustered on `{pooled.cluster_unit}`.")
            A("")

    # --- Plain language ---------------------------------------------------
    A("## 3. Plain-language reading")
    A("")
    for para in ctx["plain_language"]:
        A(para)
        A("")

    # --- Confound cohorts --------------------------------------------------
    write_confound_section(A, ctx)

    # --- Cross-validation --------------------------------------------------
    write_cv_section(A, ctx)

    # --- Cohorts ----------------------------------------------------------
    A("## 6. Cohorts")
    A("")
    A("### 6a. As catalogued by stage 1")
    A("")
    A("Row counts are in each cohort's stage-1 unit: one row per slice for the "
      "prostate cohorts, one row per acquisition for breast. They are reproduced "
      "here exactly as stage 1 emitted them and are not the number of images "
      "trained on -- see 6b for that.")
    A("")
    A(cohort_table_md(ctx["cohort_stats"]))
    A("")
    A("### 6b. As actually trained and tested")
    A("")
    A("Taken from the stage-3 run JSONs and the pooled test predictions, so this "
      "is what the reported numbers were computed on. **Every row names its own "
      "label**: the confound cohorts are in this table because it describes what "
      "ran, and they are marked so that it cannot be read as several cohorts "
      "answering one question.")
    A("")
    if ctx["trained_rows"]:
        A("| Cohort | Label | Evaluation scheme | Conditions | Seeds | Training | "
          "Validation | Test |")
        A("|---|---|---|---|---|---|---|---|")
        for r in ctx["trained_rows"]:
            A(f"| {r['cohort']} | {r['label']} | {r.get('scheme', '')} | "
              f"{r['conditions']} | {r['seeds']} | {r['train']} | "
              f"{r['val']} | {r['test']} |")
    else:
        A("_No stage-3 runs to summarise._")
    A("")
    if ctx["cohort_warnings"]:
        A("Stage-1 warnings carried forward verbatim:")
        A("")
        for cohort, warns in ctx["cohort_warnings"].items():
            for w in warns:
                A(f"- **{cohort}**: {w}")
        A("")

    # --- Statistics -------------------------------------------------------
    A("## 7. Statistics")
    A("")
    A(f"Interval construction: {ctx['bootstrap_note']}")
    A("")
    A("Every cell states its own sample size. There are no bare point estimates "
      "in this report by design. **This table covers the cohorts with a "
      "diagnostic label only**; the acquisition-identity AUCs are in section 4 "
      "and are never placed on the same axis as these.")
    A("")
    A("| Cohort | Condition | Evaluation scheme | "
      "Slice-level test AUC (95% CI, subject-clustered) | "
      "Patient-level test AUC (95% CI) | Across seeds |")
    A("|---|---|---|---|---|---|")
    for row in ctx["stat_rows"]:
        A(f"| {row['cohort']} | {row['condition']} | {row.get('scheme', '')} | "
          f"{row['auc_cell']} | {row['patient_cell']} | {row['spread']} |")
    A("")
    A("The patient-level column aggregates each subject's slice probabilities "
      "before scoring. It has fewer observations but satisfies the independence "
      "assumption that the slice-level column does not.")
    A("")
    A("### Phase vs magnitude (DeLong, Holm-adjusted)")
    A("")
    if ctx["comparison_rows"]:
        A("Stage 4 emits this comparison per seed at three levels and marks the "
          "patient level `preferred`, because its own DeLong caveat states the "
          "slice-level p-value is anti-conservative (slices within a patient are "
          "correlated). The p-values below are the WORST across seeds: a result "
          "that only holds for the luckiest seed has not been demonstrated.")
        A("")
        A("| Cohort | Level | Contrast | Delta AUC | DeLong p (worst seed) | "
          "Holm-adjusted p (worst seed) | Sample size | Note |")
        A("|---|---|---|---|---|---|---|---|")
        for r in ctx["comparison_rows"]:
            A(f"| {r['cohort']} | {r['level']} | {r['contrast']} | {r['delta']} | "
              f"{r['p']} | {r['p_holm']} | {r['n']} | <sub>{r['note']}</sub> |")
        A("")
    else:
        A("_No DeLong comparisons available (stage 4 has not produced them). "
          "Criterion C2 is therefore MISSING for every cohort, and no cohort can "
          "be SUPPORTED._")
        A("")
    if ctx["stats_warnings"]:
        A("Stage-4 warnings carried forward verbatim:")
        A("")
        for w in ctx["stats_warnings"]:
            A(f"- {w}")
        A("")

    # --- Controls ---------------------------------------------------------
    A("## 8. Controls verdict table")
    A("")
    A("| Cohort | Control | Result | Expected if the effect is real | State |")
    A("|---|---|---|---|---|")
    for r in ctx["control_rows"]:
        A(f"| {r['cohort']} | {r['control']} | {r['result']} | {r['expected']} | {r['state']} |")
    A("")
    if ctx["unrecognised_controls"]:
        A("> **Unrecognised control entries** (present in the stage-5 output but not "
          "matched to any criterion; they were NOT counted as passing):")
        for cohort, names in ctx["unrecognised_controls"].items():
            A(f"> - {cohort}: `{'`, `'.join(names)}`")
        A("")

    # --- Figures ----------------------------------------------------------
    A("## 9. Figures")
    A("")
    if ctx["figures"]:
        for label, p in ctx["figures"]:
            A(f"- **{label}** -- `{p}`")
            A("")
            A(f"  ![{label}]({os.path.relpath(p, path.parent)})")
            A("")
    else:
        A("_No figures could be produced._")
        A("")
    for note in ctx["figure_notes"]:
        A(f"- _{note}_")
    A("")

    # --- Reproduction -----------------------------------------------------
    A("## 10. Reproducing this report")
    A("")
    A("```bash")
    A("python pipeline/run_all.py            # full pipeline, resumable")
    A("python pipeline/s06_report.py         # regenerate this file only")
    A("```")
    A("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(L))
    logger.info("report -> %s", path)
    return path


def plain_language(verdicts: List[CohortVerdict], ctx_pooled: dict,
                   confound_context: Optional[dict] = None) -> List[str]:
    out: List[str] = []
    # The confound paragraph goes FIRST when there is one, because on this data
    # it is the finding: the diagnostic verdicts below are all negative or
    # unevaluable, and the reason they are is measured here.
    for cohort, c in sorted((confound_context or {}).items()):
        spec = c["spec"]
        head = c["per_cond"].get(HEADLINE_CONDITION)
        ref = c["per_cond"].get(REFERENCE_CONDITION)
        if head is None or not head.ok:
            continue
        d = c.get("delta", float("nan"))
        n_sub = head.n_clusters
        out.append(
            f"**{cohort}: phase identifies the {spec.label_short}, not a disease.** "
            f"This cohort has no tumour label of any kind -- the target is "
            f"{spec.label_long}. Asked to predict it, a network reading only the phase "
            f"channel scores AUC {head.point:.3f}"
            + (f" [{head.lo:.3f}, {head.hi:.3f}]" if head.ci_finite else "")
            + (f" on {n_sub} independent test subjects" if n_sub else "")
            + (f", against {ref.point:.3f} for magnitude" if (ref and ref.ok) else "")
            + ". "
            + (f"Phase is the better predictor, by {d:+.3f} AUC. "
               if (not math.isnan(d) and d > 0
                   and (c.get("delta_ci") or {}).get("excludes_zero")) else "")
            + (f"Phase and magnitude predict it about equally well "
               f"({d:+.3f}, paired 95% CI "
               f"[{c['delta_ci']['lo']:+.3f}, {c['delta_ci']['hi']:+.3f}] -- includes "
               f"zero), so no ordering between the channels is claimed; the level is "
               f"the point. "
               if ((c.get("delta_ci") or {}).get("ok")
                   and not c["delta_ci"]["excludes_zero"]) else "")
            + "**Read that backwards.** A high number here is not a success; it is the "
              "alternative explanation for the whole study, measured directly. Whatever "
              "phase carries beyond magnitude, a large part of it is the identity of "
              "the acquisition -- and any diagnostic score computed from the same "
              "channel inherits that."
        )
    if not verdicts:
        out.append("No cohort with a diagnostic label could be evaluated. This is not "
                   "a null result on the phase hypothesis; it is an unfinished run.")
        return out
    for cv in verdicts:
        pooled = ctx_pooled.get((cv.cohort, HEADLINE_CONDITION))
        size = ""
        if pooled is not None:
            size = (f" The test fold for this cohort is "
                    f"{caveat_text(pooled.n_clusters, pooled.n_pos_clusters, pooled.n, pooled.n_pos)}.")
        if cv.verdict == "SUPPORTED":
            out.append(
                f"**{cv.cohort}: supported.** Phase beat magnitude, and every "
                f"falsification control behaved as it must if the signal is "
                f"anatomical rather than instrumental.{size} Even so, an effect "
                f"demonstrated on this few independent subjects is a hypothesis "
                f"for a prospective, multi-scanner replication, not a finding "
                f"ready for clinical claims."
            )
        elif cv.verdict == "NOT SUPPORTED":
            names = "; ".join(f"{c.code} -- {c.detail}" for c in cv.failing)
            out.append(
                f"**{cv.cohort}: not supported.** At least one criterion failed: "
                f"{names}.{size} The honest reading is that this cohort provides no "
                f"evidence that phase adds tumour information beyond magnitude. "
                f"A prior manuscript reported AUC 0.85/0.97 for phase; no code in "
                f"this repository has ever reproduced those numbers, and this "
                f"analysis does not support them."
            )
        else:
            names = "; ".join(f"{c.code} -- {c.detail}" for c in cv.missing)
            role_note = ""
            if not cv.is_primary:
                # Say it in the plain-language section too. A reader who takes
                # only this paragraph away must not come away thinking an
                # exploratory cohort was merely unlucky with its controls.
                role_note = (
                    f" This cohort is EXPLORATORY: the pre-registered primary "
                    f"cohort is {PRIMARY_COHORT}, and testing a second cohort at "
                    f"the same alpha would inflate the family-wise error rate. "
                    f"Its criteria are reported for information; whatever they "
                    f"say, no confirmatory conclusion is drawn from them."
                )
            out.append(
                f"**{cv.cohort}: inconclusive.** Nothing failed outright, but the "
                f"evidence needed to rule out the scanner-fingerprint explanation "
                f"is not present: {names}.{size}{role_note} Inconclusive is "
                f"reported here rather than positive, because a control that has "
                f"not been run is not a control that passed."
            )
    return out


# ==========================================================================
# Orchestration
# ==========================================================================

def build_report(args) -> Tuple[int, dict]:
    results_dir = Path(args.results_dir)
    cache_dir = Path(args.cache_dir)
    cohorts_dir = Path(args.cohorts_dir)
    out_dir = Path(args.out)
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs: List[Tuple[str, str, bool, str]] = []
    degraded: List[str] = []

    # --- stage 3 runs -----------------------------------------------------
    fold_defects: List[str] = []
    runs = load_runs(results_dir, defects=fold_defects)
    inputs.append(("stage-3 run JSONs", str(results_dir), bool(runs),
                   "no per-run JSON found; run s03_train.py"))
    if not runs:
        degraded.append("No stage-3 training results were found, so no cohort could "
                        "be evaluated at all.")
    degraded.extend(fold_defects)
    full_runs = [r for r in runs if str(r.get("region", "full")) == "full"]
    all_cohorts = sorted({str(r["cohort"]) for r in full_runs})

    # --- confound cohorts are separated here, before anything is evaluated ---
    # brain and knee have no tumour label. They must never reach evaluate_cohort
    # (whose criteria are all phrased about tumour signal), never appear in the
    # diagnostic verdict table, and never be counted in the family of cohorts C0
    # controls. `cohorts` from here on means DIAGNOSTIC cohorts only.
    confound_cohorts = [c for c in all_cohorts if is_confound_cohort(c)]
    cohorts = [c for c in all_cohorts if not is_confound_cohort(c)]
    if confound_cohorts:
        logger.info("confound cohorts held out of the diagnostic verdict: %s",
                    ", ".join(confound_cohorts))

    # --- stage 4 ----------------------------------------------------------
    stats_path = find_first_existing([
        Path(args.stats) if args.stats else None,
        results_dir / "statistics.json",
        results_dir.parent / "statistics.json",
        results_dir / "s04_statistics.json",
    ])
    stats = Stats(load_json(stats_path) if stats_path else None, stats_path)
    inputs.append(("stage-4 statistics.json",
                   str(stats_path or results_dir / "statistics.json"),
                   stats.available,
                   "stage 4 has not run; AUC intervals fall back to an s06-computed "
                   "cluster bootstrap and DeLong comparisons are unavailable"))
    if not stats.available:
        degraded.append(
            "statistics.json is absent: AUC intervals shown here were recomputed by "
            "s06 with a subject-clustered bootstrap and are labelled `s06-fallback`; "
            "DeLong comparisons have no fallback, so criterion C2 is MISSING "
            "everywhere and no cohort can be SUPPORTED.")

    # --- cluster maps + pooled predictions --------------------------------
    # Three separate pooled views per (cohort, condition), because they answer
    # three different questions and conflating them is the defect this section
    # exists to close:
    #
    #   pooled_by_fold  one entry per CV fold; the dispersion diagnostic.
    #   pooled_oof      the folds concatenated -- one prediction per subject over
    #                   the whole cohort. This is the HEADLINE for any cohort
    #                   that has a cross-validation sweep on disk.
    #   pooled_official the official-split runs only. Secondary analysis, printed
    #                   with its subject count attached so nobody reads a 7- or
    #                   4-subject fold as confirmatory.
    #
    # `pooled` is the one everything downstream uses: OOF where it exists,
    # official split otherwise. Note that running pool_runs over ALL runs at once
    # would not merely be wrong, it would silently do something arbitrary: it
    # intersects cache indices across runs, and the folds' test sets are
    # disjoint, so the intersection is whatever the official split happens to
    # share with them.
    cluster_maps = {c: build_cluster_map(c, cache_dir, cohorts_dir) for c in all_cohorts}
    # What the cross-validation DECLARED, read from the cv<k>_split columns
    # stage 1/2 wrote before any model ran. `pool_folds_oof` compares the folds
    # actually on disk against this; without it, a fold that died just shrinks
    # the pooled set and nothing says so.
    cv_expect = {c: cv_expectation(c, cache_dir, cohorts_dir) for c in all_cohorts}
    for c, exp in sorted(cv_expect.items()):
        if exp.available:
            logger.info("%s: cross-validation design = folds %s (%s)",
                        c, exp.folds, exp.source)
    # Fallback expectation: the union of folds present anywhere in this cohort's
    # CV family, exactly as s04_stats derives `expected_folds`. Weaker than the
    # design table -- it cannot see a fold that failed for EVERY condition -- but
    # it catches the case that produced a side-by-side 5-fold and 4-fold headline
    # on the real tree.
    folds_seen: Dict[str, set] = {}
    for r in full_runs:
        if r.get("_fold") is not None:
            folds_seen.setdefault(str(r["cohort"]), set()).add(int(r["_fold"]))
    pooled: Dict[Tuple[str, str], PooledPredictions] = {}
    pooled_official: Dict[Tuple[str, str], PooledPredictions] = {}
    pooled_by_fold: Dict[Tuple[str, str], Dict[int, PooledPredictions]] = {}
    cv_defects: List[str] = []
    for cohort in all_cohorts:
        cmap, unit = cluster_maps[cohort]
        for cond in CONDITION_ORDER:
            group = [r for r in full_runs if str(r["cohort"]) == cohort
                     and str(r["condition"]) == cond]
            if not group:
                continue
            official = [r for r in group if r.get("_fold") is None]
            # Fold runs are bucketed by (SPLIT FAMILY, fold), not by fold alone.
            # Grouping on the index alone put `sweepA/<cohort>_cv0` and
            # `sweepB/<cohort>_cv0` -- fold 0 of two DIFFERENT five-fold sweeps
            # over the same subjects -- into one bucket, where pool_runs averaged
            # their probability vectors because they share cache rows. Measured:
            # an honest sweep at 0.63 blended with an optimistic one at 0.995
            # gave a single "pooled out-of-fold" headline of 0.938 over the same
            # 70 subjects, with the destroy-controls flipping FAIL -> PASS.
            cv_by_family: Dict[str, Dict[int, List[dict]]] = {}
            for r in group:
                if r.get("_fold") is not None:
                    fam = str(r.get("_split_family", CV_FAMILY))
                    cv_by_family.setdefault(fam, {}).setdefault(
                        int(r["_fold"]), []).append(r)
            by_fold: Dict[int, List[dict]] = {}
            if len(cv_by_family) > 1:
                # Same refusal the single-split branch below already makes, for
                # the same reason. None of them is promoted: choosing one would
                # be choosing an experiment on an outcome.
                cv_defects.append(
                    f"{cohort}: cross-validation folds for {cond} exist in "
                    f"{len(cv_by_family)} different results subdirectories "
                    f"({', '.join(sorted(cv_by_family))}). Those are different "
                    f"cross-validation sweeps on different models, so their folds are "
                    f"NOT pooled together and none of them is used as a headline; move "
                    f"the stale sweep out of the results tree.")
            elif cv_by_family:
                by_fold = next(iter(cv_by_family.values()))

            # Single-split runs are grouped by split family before pooling. Two
            # sweeps written to two subdirectories are two experiments on two
            # test sets; averaging their probability vectors (pool_runs
            # INTERSECTS cache indices) would produce a number that is an
            # estimate of nothing. If more than one is present, none is promoted.
            off_by_family: Dict[str, List[dict]] = {}
            for r in official:
                off_by_family.setdefault(str(r.get("_split_family", ".")), []).append(r)
            po = None
            if len(off_by_family) > 1:
                cv_defects.append(
                    f"{cohort}: single-split runs for {cond} exist in "
                    f"{len(off_by_family)} different results subdirectories "
                    f"({', '.join(sorted(off_by_family))}). Those are different "
                    f"experiments on different test sets, so none of them is used as "
                    f"a headline or a secondary analysis; move the stale sweep out of "
                    f"the results tree.")
            elif off_by_family:
                fam_runs = next(iter(off_by_family.values()))
                po = pool_runs(fam_runs, cmap, unit)
            if po is not None:
                pooled_official[(cohort, cond)] = po

            folds: Dict[int, PooledPredictions] = {}
            for f, rs in sorted(by_fold.items()):
                pf = pool_runs(rs, cmap, unit)
                if pf is not None:
                    folds[f] = pf
            if folds:
                pooled_by_fold[(cohort, cond)] = folds

            oof, defs = pool_folds_oof(
                folds, expected=cv_expect.get(cohort),
                expected_folds=sorted(folds_seen.get(cohort, set())),
            ) if folds else (None, [])
            for d in defs:
                cv_defects.append(f"{cohort}: {d}")
            # Out-of-fold if it could be built, else the official split. NEVER a
            # single arbitrary fold: promoting whichever fold sorted first would
            # headline a fifth of the cohort as if it were the cohort, which is
            # the failure the disjointness check exists to catch. A cohort whose
            # folds could not be pooled and that has no official-split run has no
            # headline, so every criterion for it is MISSING -- the honest
            # answer, and the conservative one.
            chosen = oof or po
            if chosen is not None:
                pooled[(cohort, cond)] = chosen
            elif folds:
                cv_defects.append(
                    f"{cohort}: the CV folds could not be pooled and there is no "
                    f"official-split run either, so this cohort has NO headline "
                    f"estimate and every criterion for it is reported MISSING. The "
                    f"per-fold numbers are still shown, as a diagnostic only.")

    # --- do the conditions being compared cover the SAME folds? -------------
    # s04_stats already refuses a paired test between two conditions pooled over
    # different fold sets ("they were not scored on the same out-of-fold set and
    # a paired test would be comparing two different cohorts"). s06 puts those
    # same two numbers side by side in the 5a headline table and in every
    # overlaid figure, and had no such check. This is that guard, ported: the
    # mismatch is named, printed in 5a, and caps C2 -- the criterion that IS the
    # comparison -- at MISSING.
    #
    # EVERY condition present is compared, including one whose headline is not a
    # pooled out-of-fold vector at all. Restricting the comparison to
    # `is_pooled_oof` rows meant that a cohort with phase cross-validated over
    # five folds and magnitude on a leftover official split showed exactly one
    # entry here and was recorded as consistent -- the largest possible
    # difference of test set, invisible because only one side declared a fold
    # set. The signature is therefore (scheme, folds), and a condition with no
    # folds contributes its scheme.
    fold_set_mismatch: Dict[str, str] = {}
    for cohort in cohorts:
        sets = {cond: (p.scheme, tuple(p.folds)) for cond in CONDITION_ORDER
                for p in [pooled.get((cohort, cond))] if p is not None}
        if len(set(sets.values())) > 1:
            shown = "; ".join(
                f"{cond}: {sch}"
                + (f" folds {list(fs)}" if fs else " (no folds)")
                + f" ({pooled[(cohort, cond)].n_clusters} subjects)"
                for cond, (sch, fs) in sorted(sets.items()))
            fold_set_mismatch[cohort] = (
                f"the conditions were pooled over DIFFERENT test sets ({shown}), so "
                f"they were not scored on the same out-of-fold set and the difference "
                f"between them is partly a difference of test set")
            cv_defects.append(f"{cohort}: {fold_set_mismatch[cohort]}. Criterion C2 is "
                              f"capped at MISSING and every table and figure that puts "
                              f"these numbers side by side states the discrepancy.")

    # The same structural problem shows up once per condition; report it once.
    seen_cv: set = set()
    for d in cv_defects:
        if d not in seen_cv:
            seen_cv.add(d)
            degraded.append(d)

    # --- stage 5 ----------------------------------------------------------
    # s05 writes no aggregate file: it writes one JSON per control run into
    # pipeline_out/controls/results. Search the places it can put them.
    control_dirs = []
    for d in (Path(args.controls) if args.controls else None,
              OUT_ROOT_GUESS(results_dir) / "controls" / "results",
              results_dir.parent / "controls" / "results",
              results_dir / "controls"):
        if d is not None and d not in control_dirs:
            control_dirs.append(d)
    control_payloads = load_control_payloads(control_dirs)
    harvested = harvest_controls_from_runs(runs, cluster_maps, args.bootstrap, args.seed)
    # The cluster maps and the declared CV design are handed in so that a
    # control which ran per fold is pooled out-of-fold on the SAME clustering
    # unit, against the SAME declared fold set, and with the same coverage
    # refusals as the headline it is going to be differenced against.
    controls = Controls(control_payloads, control_dirs, harvested=harvested,
                        cluster_maps=cluster_maps, cv_expect=cv_expect,
                        bootstrap=args.bootstrap, seed=args.seed)
    for d in controls.fold_defects:
        if d not in degraded:
            degraded.append(d)
    inputs.append(("stage-5 controls",
                   " ; ".join(str(d) for d in control_dirs),
                   controls.available,
                   "stage 5 has not run; only controls recoverable from stage-3 "
                   "region runs are available"))
    if controls.failures:
        # Provenance, not a verdict: name the controls stage 5 recorded as
        # broken, so a MISSING criterion below cannot be misread as "stage 5
        # simply has not run this one yet".
        for (cslug, canon), rec in sorted(controls.failures.items()):
            errs = "; ".join(str(e.get("error", "")) for e in (rec.get("errors") or []))
            degraded.append(
                f"{rec.get('cohort', cslug)}: stage 5 recorded a FAILURE running the "
                f"{CANONICAL_LABEL.get(canon, canon)} control"
                + (f" ({errs})" if errs else "")
                + ". That criterion is reported MISSING; any run files this control "
                  "did leave behind are a survivorship-biased sample and are not scored.")
    if not control_payloads:
        msg = ("no stage-5 control runs were found: the permutation, phase-scramble, "
               "acquisition-stratified and confound controls could not be evaluated.")
        if harvested:
            msg += (" A background-only control was reconstructed from "
                    "`--region background` stage-3 runs and is labelled `s06-fallback`.")
        degraded.append(msg)
    else:
        found = {c: controls.names_present(c) for c in cohorts}
        wanted = set(S05_TO_CANONICAL.values())
        for cohort, names in found.items():
            gap = sorted(wanted - set(names))
            if gap:
                degraded.append(
                    f"{cohort}: stage 5 produced no {', '.join(gap)} control; those "
                    f"criteria are reported MISSING, never as passing.")

    # --- stage 1 cohort summary ------------------------------------------
    s01_path = cohorts_dir / "s01_summary.json"
    s01 = load_json(s01_path) or {}
    inputs.append(("stage-1 cohort summary", str(s01_path), bool(s01),
                   "cohort table unavailable"))
    cohort_stats = (s01.get("cohorts") or {}) if isinstance(s01, dict) else {}
    cohort_warnings = {c: (s.get("warnings") or []) for c, s in cohort_stats.items()
                       if isinstance(s, dict) and s.get("warnings")}

    cache_present = (any((cache_dir / f"{c}.h5").exists() for c in all_cohorts)
                     if all_cohorts
                     else bool(list(cache_dir.glob("*.h5"))) if cache_dir.exists() else False)
    inputs.append(("stage-2 cache", str(cache_dir), cache_present,
                   "qualitative magnitude-vs-phase panel cannot be drawn"))

    # --- statistics rows + estimates --------------------------------------
    def headline_family(cohort: str, cond: str) -> Optional[str]:
        """
        The split family of the predictions this report is headlining.

        Every stage-4 lookup is scoped to it. Without that scoping, a leftover
        single-split row for the same condition was averaged into the pooled
        out-of-fold number and the blend was reported as an extra "seed"; see
        Stats.estimate.
        """
        p = pooled.get((cohort, cond))
        return None if p is None else p.split_family

    def estimate_for(cohort: str, cond: str, level: str = "slice") -> Estimate:
        est = stats.estimate(cohort, cond, level,
                             family=headline_family(cohort, cond)) \
            if stats.available else Estimate(source="s04-missing")
        p = pooled.get((cohort, cond))
        if est.ok:
            # The back-fill below lends the pooled vector's sample size to a
            # stage-4 row that did not state one. It must NOT be lent to a row
            # whose own per-seed blocks DISAGREE about the size: that is not a
            # row missing a declaration, it is a row with two, and giving it the
            # pooled vector's is the same borrow this round exists to close --
            # the headline would then present a fingerprint stage 4 never had.
            conflicted = bool((est.detail or {}).get("fingerprint_conflicts"))
            if not conflicted:
                if est.n_clusters is None and p is not None:
                    est.n_clusters = p.n_clusters
                if est.n is None and p is not None:
                    est.n, est.n_pos = p.n, p.n_pos
            if not est.per_seed and p is not None:
                est.per_seed = p.per_seed_auc
            return est
        if p is None or level != "slice":
            return Estimate(source="unavailable")
        bs = _bootstrap_cache(p, args.bootstrap, args.seed)
        e = Estimate(point=bs["point"], lo=bs["lo"], hi=bs["hi"], n=bs["n"],
                     n_pos=bs["n_pos"], n_clusters=bs["n_clusters"],
                     n_boot_valid=bs["n_boot_valid"], source="s06-fallback",
                     per_seed=p.per_seed_auc)
        e.note = "computed by s06 because stage 4 has not run"
        return e

    headline_cache: Dict[Tuple[str, str, str], Headline] = {}

    def headline_for(cohort: str, cond: str, level: str = "slice") -> Headline:
        """
        THE headline for one (cohort, condition, level), resolved exactly once.

        Memoised, so every criterion, every figure and every table that asks for
        this cohort's headline gets the SAME object with the same provenance
        attached -- which is the only way the "no two criteria may quote
        different headlines" invariant can mean anything. It also validates the
        estimate on the way out: a point estimate outside its own interval is a
        wiring bug (a point and an interval read from two different scopes) and
        stops the report being written at all rather than being printed.
        """
        key = (cohort, cond, level)
        if key not in headline_cache:
            p = pooled.get((cohort, cond))
            if p is None:
                # No prediction vector was promoted for this condition -- the
                # folds overlapped, or two sweeps were on disk, or nothing ran.
                # Stage 4 may still hold rows, but there is then nothing on disk
                # that says WHICH predictions they describe, and the size guard
                # that would have asked cannot run without a pooled vector to
                # compare against. Measured: two five-fold sweeps in two
                # subdirectories were correctly refused a pooled headline, and
                # every criterion went on being decided on statistics.json --
                # C4 and C7 PASSED against a number for predictions the report
                # had just declined to build. This cohort/condition has NO
                # headline, which is what the pooling refusal already says.
                headline_cache[key] = no_headline(cohort, cond, level)
                return headline_cache[key]
            est = estimate_for(cohort, cond, level)
            _assert_estimate_within_ci(est, f"{cohort}/{cond} at the {level} level")
            headline_cache[key] = Headline(
                cohort=cohort, condition=cond, level=level, est=est,
                split_family=None if p is None else p.split_family,
                folds=tuple(p.folds) if p is not None else (),
                scheme="" if p is None else p.scheme)
        return headline_cache[key]

    def pooled_estimate(p: Optional[PooledPredictions], why: str) -> Estimate:
        """A subject-clustered bootstrap over an explicit prediction vector."""
        if p is None or len(np.unique(p.labels)) < 2:
            return Estimate(source="unavailable")
        bs = _bootstrap_cache(p, args.bootstrap, args.seed)
        e = Estimate(point=bs["point"], lo=bs["lo"], hi=bs["hi"], n=bs["n"],
                     n_pos=bs["n_pos"], n_clusters=bs["n_clusters"],
                     n_boot_valid=bs["n_boot_valid"], source="s06 cluster bootstrap",
                     per_seed=[round(v, 4) for v in p.per_seed_auc])
        e.note = why
        e.detail = {"n_pos_clusters": p.n_pos_clusters, "scheme": p.scheme}
        return e

    # --- does stage 4 describe the predictions this report headlines? -------
    # Only asked where the headline is the pooled out-of-fold vector. If stage 4
    # summarised a different fold, its C1/C2 are capped at MISSING; see
    # evaluate_cohort(stats_scheme_mismatch=...). Decided on the sample sizes
    # both sides already publish rather than on a schema field s06 would have to
    # invent, so it works whatever stage 4 emits.
    #
    # The sizes are read from EVERY stage-4 block in the family, not from
    # whichever one happened to be written first. Reading blocks[0] made this
    # guard depend on the alphabetical order of the split-family key -- '.' and
    # 'confound_x' sort before 'cv', 'official' and 'zz_stale' sort after -- so a
    # stale single-split sweep evaded it entirely if its directory name sorted
    # the right way. Family scoping (above) removes the mixture; checking every
    # block removes the ordering dependence that remains within a family.
    def _stage4_describes(cohort: str, cond: str) -> str:
        """Why stage 4's rows for (cohort, cond) are not the pooled vector, or ''."""
        p = pooled.get((cohort, cond))
        if p is None or not p.is_pooled_oof or not stats.available:
            return ""
        s4 = stats.estimate(cohort, cond, "slice", family=p.split_family)
        if not s4.ok:
            fams = stats.families_for(cohort, cond)
            if fams:
                return (f"statistics.json holds no {cond} rows for the split family "
                        f"this report headlines ({p.split_family!r}); it holds "
                        f"{', '.join(fams)}")
            return ""
        n_set = list(s4.detail.get("n_set") or ([] if s4.n is None else [int(s4.n)]))
        ncl_set = list(s4.detail.get("n_clusters_set")
                       or ([] if s4.n_clusters is None else [int(s4.n_clusters)]))
        if not n_set and not ncl_set:
            return (f"stage 4 reports no sample size for {cond}, so there is no way to "
                    f"tell whether it summarised the pooled out-of-fold predictions "
                    f"({p.n} slices / {p.n_clusters} subjects) or a single fold")
        if any(v != p.n for v in n_set) or any(v != p.n_clusters for v in ncl_set):
            return (f"stage 4 summarised {cond} over {n_set or '?'} slices / "
                    f"{ncl_set or '?'} subjects, while the headline here is the pooled "
                    f"out-of-fold vector over {p.n} slices / {p.n_clusters} subjects "
                    f"across {len(p.folds)} folds")
        return ""

    scheme_mismatch = {c: why for c in cohorts
                       for why in [_stage4_describes(c, HEADLINE_CONDITION)] if why}
    for cohort, why in sorted(scheme_mismatch.items()):
        degraded.append(
            f"{cohort}: statistics.json does not describe the predictions this report "
            f"headlines ({why}). C1 and C2 are capped at MISSING for this cohort and the "
            f"stage-4 numbers are shown as the official-split secondary analysis.")

    # --- ... and does it describe the OTHER side of the comparison? ---------
    # The guard above runs on HEADLINE_CONDITION only, which is right for C1 --
    # C1 is a statement about phase alone. C2 is a COMPARISON, and nothing
    # checked the reference condition at all: stage 4 could describe magnitude
    # over four folds and 56 subjects while the report pooled magnitude over five
    # and 70, and the Holm-adjusted p-value comparing them was read as if both
    # sides covered the same people.
    #
    # The comparison records themselves carry `folds_a` / `folds_b`, which s04
    # writes and which this module has been carrying through and reading nowhere.
    # They are checked here against the folds actually headlined, and -- like
    # every other provenance check in this file -- an absent declaration is a
    # reason to refuse, not a licence to proceed: a record that does not say
    # which folds it covers cannot demonstrate it covers these.
    #
    # Scoped to the family being headlined, for the same reason the estimates
    # are: a comparison record from a leftover single-split sweep is a comparison
    # on a different test set. Computed once, here, and reused for the section-5b
    # table below -- one object, one provenance check.
    comparisons: Dict[str, Dict[str, dict]] = {
        cohort: (stats.comparison_levels(
            cohort, HEADLINE_CONDITION, REFERENCE_CONDITION,
            family=headline_family(cohort, HEADLINE_CONDITION))
            if stats.available else {})
        for cohort in cohorts}
    comparison_mismatch: Dict[str, str] = {}
    for cohort in cohorts:
        why_ref = _stage4_describes(cohort, REFERENCE_CONDITION)
        reasons = [why_ref] if why_ref else []
        ph = pooled.get((cohort, HEADLINE_CONDITION))
        rf = pooled.get((cohort, REFERENCE_CONDITION))
        levels = comparisons.get(cohort) or {}
        # (a) The comparison GROUP is an aggregate of per-seed records, and its
        # own records must agree on one test set before the group can be said to
        # describe one. `n_clusters` / `n_cases` used to be `group[0]`'s.
        for lvl, blk in sorted(levels.items()):
            for conflict in (blk.get("provenance_conflicts") or []):
                reasons.append(
                    f"stage 4's {lvl}-level comparison records do not describe one "
                    f"test set: {conflict}")
            if reasons:
                break
        # (b) ... and both sides of it must be checkable against what this report
        # actually headlines. The fold check below only ran when BOTH conditions
        # had a pooled prediction vector, so a report that holds no
        # {REFERENCE_CONDITION} predictions at all -- none on disk, or two sweeps
        # the pooler refused to merge -- skipped it entirely and C2 was decided
        # on a stage-4 p-value whose reference side this report cannot identify.
        # An unverifiable side is a reason to refuse, exactly like an undeclared
        # one.
        if levels and not reasons and (ph is None or rf is None):
            absent = [name for name, p_ in ((HEADLINE_CONDITION, ph),
                                            (REFERENCE_CONDITION, rf)) if p_ is None]
            reasons.append(
                f"stage 4 holds a {HEADLINE_CONDITION}-vs-{REFERENCE_CONDITION} "
                f"comparison, but this report has no pooled predictions for "
                f"{' or '.join(absent)} (none on disk, or more than one sweep was "
                f"found and they were not merged), so there is nothing to check the "
                f"comparison's test set against; the fold sets it declares cannot be "
                f"shown to be the ones headlined here")
        if levels and not reasons and ph is not None and rf is not None and (
                ph.is_pooled_oof or rf.is_pooled_oof):
            want_a, want_b = tuple(ph.folds), tuple(rf.folds)
            for lvl, blk in sorted(levels.items()):
                if not blk.get("declares_folds"):
                    reasons.append(
                        f"stage 4's {lvl}-level comparison records do not say which "
                        f"folds each condition was scored on, so they cannot be shown "
                        f"to describe the fold sets this report headlines "
                        f"({HEADLINE_CONDITION} {list(want_a)}, "
                        f"{REFERENCE_CONDITION} {list(want_b)})")
                    break
                got_a = {tuple(f) for f in blk.get("folds_a") or []}
                got_b = {tuple(f) for f in blk.get("folds_b") or []}
                if got_a != {want_a} or got_b != {want_b}:
                    reasons.append(
                        f"stage 4's {lvl}-level comparison was computed over folds "
                        f"{sorted(got_a)} ({HEADLINE_CONDITION}) vs {sorted(got_b)} "
                        f"({REFERENCE_CONDITION}), while this report headlines "
                        f"{list(want_a)} vs {list(want_b)}")
                    break
                if want_a != want_b:
                    reasons.append(
                        f"stage 4's {lvl}-level comparison puts {HEADLINE_CONDITION} "
                        f"over folds {list(want_a)} against {REFERENCE_CONDITION} over "
                        f"folds {list(want_b)}: two different test sets")
                    break
        if reasons:
            comparison_mismatch[cohort] = "; ".join(reasons)
            degraded.append(
                f"{cohort}: the {HEADLINE_CONDITION}-vs-{REFERENCE_CONDITION} "
                f"comparison is not between two numbers on one test set "
                f"({comparison_mismatch[cohort]}). Criterion C2 is capped at MISSING.")

    # --- more than one experiment for one condition ------------------------
    # Reported once per cohort, whether or not it changed a number: a reader has
    # to know that statistics.json describes two experiments before reading any
    # row of it, and s06 only ever shows the one it headlines.
    for cohort in cohorts:
        for cond in CONDITION_ORDER:
            if not stats.available or (cohort, cond) not in pooled:
                continue
            fams = stats.families_for(cohort, cond)
            if len(fams) > 1:
                degraded.append(
                    f"{cohort}/{cond}: statistics.json holds {len(fams)} split families "
                    f"({', '.join(fams)}). A pooled out-of-fold estimate and a "
                    f"single-split estimate are different experiments on different test "
                    f"sets and are NEVER averaged: this report reads only the "
                    f"{headline_family(cohort, cond)!r} family, and the others are "
                    f"reported separately (section 5c) or not at all. Move the stale "
                    f"sweep out of the results tree and re-run stage 4.")

    stat_rows = []
    bar_rows = []
    cv_rows = []
    fold_rows = []
    official_rows = []
    for cohort in cohorts:
        for cond in CONDITION_ORDER:
            # Per-fold dispersion is emitted whenever folds exist, INDEPENDENTLY
            # of whether they could be pooled: if pooling was refused, the fold
            # numbers are the only thing there is to show, and hiding them would
            # leave the reader with nothing but the refusal.
            for f, pf in sorted((pooled_by_fold.get((cohort, cond)) or {}).items()):
                fold_rows.append({
                    "cohort": cohort, "condition": cond, "fold": f,
                    "auc": _safe_auc(pf.labels, pf.probs),
                    "n": pf.n, "n_pos": pf.n_pos,
                    "n_clusters": pf.n_clusters, "n_pos_clusters": pf.n_pos_clusters,
                    "seeds": ", ".join(str(s) for s in pf.seeds),
                })
            if (cohort, cond) not in pooled:
                continue
            p = pooled[(cohort, cond)]
            # Through the same resolver the criteria use: section 5a and the
            # verdict must not be able to print two different numbers.
            auc = headline_for(cohort, cond, "slice").est
            pat = headline_for(cohort, cond, "patient_mean").est if stats.available \
                else Estimate(source="s04-missing")
            stat_rows.append({
                "cohort": cohort, "condition": cond,
                "scheme": p.scheme_label,
                "auc_cell": fmt_estimate_cell(auc, p),
                "patient_cell": fmt_estimate_cell(pat, p) if pat.ok else "not available",
                "spread": stats.seed_spread(cohort, cond,
                                            family=headline_family(cohort, cond))
                          if stats.available else "",
            })
            bar_rows.append({"cohort": cohort, "condition": cond, "est": auc,
                             "caveat": caveat_text(p.n_clusters, p.n_pos_clusters,
                                                   p.n, p.n_pos),
                             "pooled": p})

            # -- cross-validation views ------------------------------------
            if p.is_pooled_oof:
                # The "every subject tested exactly once" sentence is written
                # only when it is true of THIS vector; otherwise the note says
                # what was actually covered.
                oof_est = pooled_estimate(
                    p, (f"pooled out-of-fold over {len(p.folds)} folds; one prediction "
                        f"per subject, every subject tested exactly once")
                    if p.coverage_complete else
                    (f"pooled over folds {p.folds} only -- INCOMPLETE cross-validation "
                     f"({p.coverage_text}): {p.coverage_defect}"))
                cv_rows.append({"cohort": cohort, "condition": cond,
                                "pooled": p, "est": oof_est})
                po = pooled_official.get((cohort, cond))
                if po is not None:
                    official_rows.append({
                        "cohort": cohort, "condition": cond, "pooled": po,
                        "est": pooled_estimate(
                            po, "official split only -- SECONDARY analysis"),
                    })

    # --- what was actually trained on, as opposed to what stage 1 catalogued -
    # Confound cohorts are INCLUDED here, because this table is a description of
    # what ran rather than a result -- but every row states what its label is, so
    # the table cannot be read as five cohorts answering one question.
    trained_rows = []
    for cohort in all_cohorts:
        p = pooled.get((cohort, HEADLINE_CONDITION)) or \
            next((v for (c, _), v in pooled.items() if c == cohort), None)
        if p is None:
            continue
        splits = {}
        run0 = next((r for r in full_runs if str(r["cohort"]) == cohort), None)
        if run0 and isinstance(run0.get("splits"), dict):
            splits = run0["splits"]
        spec = confound_spec(cohort)
        trained_rows.append({
            "cohort": cohort,
            "label": (f"**{spec.label_long}** -- NOT a diagnosis" if spec
                      else DIAGNOSTIC_LABEL_LONG),
            "scheme": p.scheme_label,
            "conditions": ", ".join(sorted({str(r["condition"]) for r in full_runs
                                            if str(r["cohort"]) == cohort})),
            "seeds": ", ".join(str(s) for s in p.seeds),
            "train": _split_cell(splits.get("training")),
            "val": _split_cell(splits.get("validation")),
            "test": (f"{p.n} slices ({p.n_pos} positive), {p.n_clusters} "
                     f"{p.cluster_unit} units, {p.n_pos_clusters} positive"),
        })

    # --- confound cohorts: phase vs magnitude at predicting the acquisition --
    # These are the numbers the paper's mechanism claim rests on, and they are
    # computed here with the same subject-clustered bootstrap as everything else.
    # They are NOT diagnostic AUCs and never enter a verdict.
    confound_rows: List[dict] = []
    confound_context: Dict[str, dict] = {}
    external_confounds: List[Tuple[str, Estimate, str]] = []
    for cohort in confound_cohorts:
        spec = confound_spec(cohort)
        index = None
        idx_csv = cache_dir / f"{cohort}_index.csv"
        if idx_csv.exists():
            try:
                index = pd.read_csv(idx_csv, low_memory=False)
            except Exception as exc:  # noqa: BLE001
                logger.warning("could not read %s: %s", idx_csv, exc)
        # sep=" ": these strings land in markdown table cells and prose, where a
        # newline would terminate the row.
        pos_name, neg_name = class_names(cohort, index, sep=" ")
        # The label_target column is what stage 2 wrote next to every row. It is
        # quoted verbatim so the heading cannot drift from the data.
        label_target = ""
        if index is not None and "label_target" in index.columns:
            vals = sorted({str(v) for v in index["label_target"].dropna().unique()})
            label_target = ", ".join(vals)
        per_cond: Dict[str, Estimate] = {}
        for cond in CONDITION_ORDER:
            p = pooled.get((cohort, cond))
            if p is None:
                continue
            est = pooled_estimate(
                p, f"AUC at predicting {spec.label_short}, not at detecting disease")
            per_cond[cond] = est
            confound_rows.append({"cohort": cohort, "condition": cond, "est": est,
                                  "spec": spec, "pooled": p,
                                  "n_text": _confound_fold_text(p, spec)})
        head = per_cond.get(HEADLINE_CONDITION)
        ref = per_cond.get(REFERENCE_CONDITION)
        confound_context[cohort] = {
            "spec": spec, "per_cond": per_cond,
            "positive_name": pos_name, "negative_name": neg_name,
            "label_target": label_target,
            "pooled": pooled.get((cohort, HEADLINE_CONDITION)),
            "delta": (head.point - ref.point) if (head and ref and head.ok and ref.ok)
                     else float("nan"),
            # Paired subject-clustered interval on that difference. The prose
            # may only call phase "the better predictor" when this excludes
            # zero; on a tie it says so. Computed here so both the section-4
            # paragraph and the plain-language summary read the same evidence.
            "delta_ci": cluster_bootstrap_delta(
                pooled.get((cohort, HEADLINE_CONDITION)),
                pooled.get((cohort, REFERENCE_CONDITION)),
                n_boot=args.bootstrap, seed=args.seed),
            # Stage 7, if it has run. Read-only and entirely optional: s06 takes
            # the markdown block s07 renders and nothing else, so a change to
            # s07's numeric schema cannot break this report, and its absence
            # costs a paragraph rather than a section.
            **_paired_block(cohort, results_dir),
        }
        # Wire C6 to the direct measurement -- but only from a cohort whose label
        # is a HARDWARE identity, which is the quantity C6 names. See
        # ConfoundCohortSpec.feeds_c6.
        if head is not None and head.ok and spec.feeds_c6:
            external_confounds.append((
                spec.label_short, head,
                f"measured directly on the {cohort} confound cohort, "
                f"{HEADLINE_CONDITION} channel, {head.n_clusters} independent test "
                f"subjects, label = {spec.label_long}"))
    # Deterministic and worst-first, so the criterion's evidence string does not
    # depend on dict ordering.
    external_confounds.sort(key=lambda t: -t[1].point)

    # --- comparisons ------------------------------------------------------
    # `comparisons` was resolved with the provenance checks above; this only
    # renders it. Re-deriving it here would be a second lookup that no guard had
    # seen, which is the pattern this round of fixes exists to remove.
    comparison_rows = []
    for cohort in cohorts:
        levels = comparisons.get(cohort) or {}
        p = pooled.get((cohort, HEADLINE_CONDITION))
        for level in ("patient_mean", "patient_max", "slice"):
            blk = levels.get(level)
            if not blk:
                continue
            d = blk["delta_mean"]
            dtxt = "n/a" if math.isnan(d) else \
                f"{d:+.3f} [seeds {blk['delta_min']:+.3f} to {blk['delta_max']:+.3f}]"
            comparison_rows.append({
                "cohort": cohort,
                "level": level + (" (preferred)" if blk["preferred"] else ""),
                "contrast": f"{HEADLINE_CONDITION} - {REFERENCE_CONDITION}",
                "delta": dtxt,
                "p": "n/a" if math.isnan(blk["p_raw_worst"]) else f"{blk['p_raw_worst']:.4g}",
                "p_holm": "n/a" if math.isnan(blk["p_holm_worst"]) else
                          f"{blk['p_holm_worst']:.4g}",
                "n": (caveat_text(p.n_clusters, p.n_pos_clusters, p.n, p.n_pos)
                      if p else "unknown"),
                "note": "; ".join(blk.get("reasons", [])) or blk.get("caveat", ""),
            })

    # --- verdicts ---------------------------------------------------------
    # The pre-registered primary cohort is the only one that can carry a
    # confirmatory verdict (see PRIMARY_COHORT). If it is not in this report at
    # all, NO cohort is promoted in its place: promoting whichever cohort
    # happens to be present is selection, which is the thing pre-registration
    # exists to prevent. Every cohort is then exploratory and the report says
    # so instead of quietly re-designating one.
    primary_present = any(_ident(c) == _ident(PRIMARY_COHORT) for c in cohorts)
    if cohorts and not primary_present:
        degraded.append(
            f"The pre-registered primary cohort ({PRIMARY_COHORT}) is not in this "
            f"report, so every cohort here is exploratory and none can reach "
            f"SUPPORTED. Criterion C0 is MISSING throughout. This is a "
            f"consequence of what was run, not of what was found.")
    c1_levels: Dict[str, str] = {}
    verdicts: List[CohortVerdict] = []
    for cohort in cohorts:
        # Slice level: the level stage 5 scores every control at, so it is the
        # level the control comparisons (C4/C5/C7/C8) must read the headline at.
        headline = headline_for(cohort, HEADLINE_CONDITION)
        reference = estimate_for(cohort, REFERENCE_CONDITION)
        # Cluster level: one score per subject. C1 is decided here, because the
        # slice-level interval assumes independence that slices do not have.
        c1_level = stats.preferred_level(
            cohort, family=headline_family(cohort, HEADLINE_CONDITION)) \
            if stats.available else C1_LEVEL
        c1_levels[cohort] = c1_level
        cluster_headline = headline_for(cohort, HEADLINE_CONDITION, c1_level)
        # Two independent ways the headline can fail to be what the report says
        # it is, combined into the one cap: the pooled vector does not cover the
        # declared cross-validation, or two conditions were pooled over
        # different fold sets so the comparison is between two test sets.
        ph = pooled.get((cohort, HEADLINE_CONDITION))
        cov = [ph.coverage_defect] if (ph is not None and ph.coverage_defect) else []
        if cohort in fold_set_mismatch:
            cov.append(fold_set_mismatch[cohort])
        verdicts.append(evaluate_cohort(
            cohort, headline, reference, comparisons.get(cohort, {}), controls,
            cluster_headline=cluster_headline, cluster_level=c1_level,
            is_primary=_ident(cohort) == _ident(PRIMARY_COHORT),
            external_confounds=external_confounds,
            stats_scheme_mismatch=scheme_mismatch.get(cohort, ""),
            coverage_defect="; ".join(cov),
            comparison_mismatch=comparison_mismatch.get(cohort, ""),
            # Reporting only: lets C5 quote the headline recomputed on exactly
            # the rows the acquisition arm was scored on, which is the matched
            # comparison a designed subgroup admits. It decides nothing.
            headline_predictions=ph))

    # --- control rows -----------------------------------------------------
    CONTROL_EXPECT = {
        "permutation": ("permutation_null",
                        "AUC ~ 0.500 (the pipeline must learn nothing from "
                        "scrambled labels)"),
        # The permutation control answers two separate questions, so it gets two
        # rows: is the null at chance (C3), and does the headline beat that null
        # empirically (C8)? The second is the one that matters; the first only
        # says the null is not obviously broken.
        "permutation_empirical": ("beats_permutation_null",
                                  f"P(permuted-label AUC >= headline) < {ALPHA} "
                                  f"over the replicates"),
        "background": ("background_collapses",
                       f"at least {BACKGROUND_MARGIN:.2f} AUC below the headline, "
                       f"and its own 95% CI covers {CONTROL_CHANCE:.2f}"),
        "scramble": ("phase_scramble_collapses",
                     f"at least {BACKGROUND_MARGIN:.2f} AUC below the headline "
                     f"and its own 95% CI covers {CONTROL_CHANCE:.2f} "
                     f"(a spatial effect cannot survive scrambling)"),
        "acquisition": ("acquisition_stratified_holds",
                        "effect survives holding acquisition out across the split, "
                        "in BOTH directions"),
        "confound": ("confound_not_explanatory",
                     f"scanner / coil / site predictable at < {CONFOUND_AUC_MAX:.2f} AUC "
                     f"and >= {CONFOUND_HEADLINE_MARGIN:.2f} below the headline"),
    }
    CONTROL_ROW_LABEL = dict(CANONICAL_LABEL)
    CONTROL_ROW_LABEL["permutation_empirical"] = \
        "label permutation, used as the empirical null for the headline"
    control_rows = []
    for cv in verdicts:
        by_key = {c.key: c for c in cv.criteria}
        for canon, (crit_key, expected) in CONTROL_EXPECT.items():
            crit = by_key[crit_key]
            if canon == "permutation_empirical":
                control_rows.append({
                    "cohort": cv.cohort, "control": CONTROL_ROW_LABEL[canon],
                    "result": crit.evidence or "not available",
                    "expected": expected,
                    "state": {"pass": "PASS", "fail": "**FAIL**",
                              "missing": "_NOT EVALUABLE_"}[crit.status]
                             + (f"<br><sub>{crit.detail}</sub>"
                                if crit.status != "pass" else ""),
                })
                continue
            if canon == "confound":
                result = crit.evidence or "not available"
            else:
                est = controls.estimate(cv.cohort, canon)
                result = (_fmt_est_inline(est) +
                          f"<br><sub>{est.source}; {est.note}</sub>") \
                    if est.ok else "not available"
            # "missing" covers both "never run" and "run but not evaluable"
            # (an underpowered permutation null, for instance). Saying NOT RUN
            # for the latter would misdescribe it, so the reason is carried
            # into the cell instead of being flattened into a label.
            state = {"pass": "PASS", "fail": "**FAIL**",
                     "missing": "_NOT EVALUABLE_"}[crit.status]
            if crit.status != "pass":
                state += f"<br><sub>{crit.detail}</sub>"
            control_rows.append({"cohort": cv.cohort,
                                 "control": CONTROL_ROW_LABEL[canon],
                                 "result": result, "expected": expected, "state": state})

    # --- figures ----------------------------------------------------------
    figures: List[Tuple[str, str]] = []
    figure_notes: List[str] = []
    auc_mismatches: List[str] = []
    # What the figures actually assert about sample size, captured so the
    # self-test can read it back rather than re-deriving it.
    fig_labels: Dict[str, List[str]] = {"roc": [], "auc_bars": [], "cv_folds": []}
    for cohort in cohorts:
        by_cond = {c: pooled.get((cohort, c)) for c in CONDITION_ORDER}
        grid = np.linspace(0, 1, 201)
        bands = {}
        for cond, p in by_cond.items():
            if p is None:
                continue
            bs = _bootstrap_cache(p, args.bootstrap, args.seed, fpr_grid=grid)
            s04 = stats.estimate(cohort, cond, "slice",
                                 family=headline_family(cohort, cond)) \
                if stats.available else Estimate()
            # Stage 4's number is preferred -- unless it demonstrably describes a
            # different set of predictions from the one the curve is drawn from.
            # Printing a 7-subject official-split AUC in the legend of a curve
            # over 67 pooled out-of-fold subjects would put the wrong number on
            # the right picture, which is worse than either alone.
            if s04.ok and cohort not in scheme_mismatch:
                bs = dict(bs)
                bs["point"], bs["lo"], bs["hi"] = s04.point, s04.lo, s04.hi
                bs["_from_s04"] = True
            bands[cond] = bs
        pth = fig_roc(cohort, {k: v for k, v in by_cond.items() if v is not None},
                      bands, fig_dir, args.dpi, mismatches=auc_mismatches,
                      notes=fig_labels["roc"])
        if pth:
            figures.append((f"ROC, {cohort}", str(pth)))

    pth = fig_auc_bars(bar_rows, fig_dir, args.dpi, titles=fig_labels["auc_bars"])
    if pth:
        figures.append(("AUC by cohort and condition", str(pth)))
    pth = fig_training_curves(full_runs, fig_dir, args.dpi)
    if pth:
        figures.append(("Training curves", str(pth)))
    for cv in verdicts:
        _lvl = c1_levels.get(cv.cohort, C1_LEVEL)
        # The SAME headline object the criteria were decided on, so the panel
        # cannot draw a bar the verdict never saw.
        pth = fig_controls_panel(cv.cohort,
                                 headline_for(cv.cohort, HEADLINE_CONDITION).est,
                                 controls, cv, fig_dir, args.dpi,
                                 pooled.get((cv.cohort, HEADLINE_CONDITION)),
                                 cluster_headline=headline_for(
                                     cv.cohort, HEADLINE_CONDITION, _lvl).est,
                                 cluster_level=_lvl)
        if pth:
            figures.append((f"Controls panel, {cv.cohort}", str(pth)))
    # The confound cohorts get their OWN AUC figure, never a shared axis with a
    # diagnostic cohort: putting brain's 0.893 next to prostate_t2's number on
    # one chart invites exactly the comparison the paper says cannot be made.
    pth = fig_confound_predictability(confound_rows, fig_dir, args.dpi)
    if pth:
        figures.append(("Confound predictability (brain / knee): phase vs magnitude at "
                        "predicting the ACQUISITION property -- high is BAD", str(pth)))
    elif confound_cohorts:
        figure_notes.append(
            "Confound-predictability figure was skipped: no scorable predictions for "
            + ", ".join(confound_cohorts) + ".")
    pth = fig_cv_folds(cv_rows, fig_dir, args.dpi, titles=fig_labels["cv_folds"])
    if pth:
        figures.append(("Cross-validated clinical results: pooled out-of-fold estimate "
                        "with per-fold dispersion", str(pth)))

    # Qualitative panels for EVERY cohort, confound cohorts included: the point
    # of the panel is to show what the network saw, and that is as relevant for
    # a coil-count label as for a tumour one. `fig_qualitative` names each row
    # with the cohort's real classes.
    for cohort in (all_cohorts or []):
        pth = fig_qualitative(cohort, cache_dir, fig_dir, args.dpi,
                              pooled.get((cohort, HEADLINE_CONDITION)))
        if pth:
            spec = confound_spec(cohort)
            suffix = (f" (rows are {spec.label_short}, NOT tumour status)" if spec else "")
            figures.append((f"Qualitative magnitude vs phase, {cohort}{suffix}", str(pth)))
        else:
            figure_notes.append(
                f"Qualitative panel for {cohort} was skipped: "
                f"pipeline_out/cache/{cohort}.h5 is missing or locked by a running "
                f"stage 2. Re-run s06 once the cache is complete.")

    ctx = {
        "inputs": [(a, b, c, d) for a, b, c, d in inputs],
        "degraded": degraded,
        "verdicts": verdicts,
        "pooled": pooled,
        "cohort_stats": cohort_stats,
        "cohort_warnings": cohort_warnings,
        "bootstrap_note": (stats.bootstrap_note() if stats.available else
                           f"{args.bootstrap} resamples clustered on the stage-1 "
                           f"subject id, 0.95 percentile interval (computed by s06, "
                           f"because stage 4 has not run)"),
        "stat_rows": stat_rows,
        "comparison_rows": comparison_rows,
        "control_rows": control_rows,
        # The Controls view itself, so the self-test can interrogate the pooled
        # null and the pooled control estimates directly instead of parsing them
        # back out of a formatted evidence string.
        "controls": controls,
        "unrecognised_controls": controls.unrecognised,
        "stats_warnings": (stats.warnings + stats.stage3_mismatches()) if stats.available else [],
        "figures": figures,
        "figure_notes": figure_notes,
        "auc_mismatches": auc_mismatches,
        "trained_rows": trained_rows,
        "plain_language": plain_language(verdicts, pooled, confound_context),
        "c1_levels": c1_levels,
        "primary_cohort": PRIMARY_COHORT,
        "primary_present": primary_present,
        # -- confound cohorts (no diagnostic label, ever) -------------------
        "confound_cohorts": confound_cohorts,
        "confound_context": confound_context,
        "confound_rows": confound_rows,
        "external_confounds": external_confounds,
        # -- cross-validation ----------------------------------------------
        "cv_rows": cv_rows,
        "fold_rows": fold_rows,
        "official_rows": official_rows,
        "scheme_mismatch": scheme_mismatch,
        "fold_set_mismatch": fold_set_mismatch,
        "cv_expectation": cv_expect,
        "fig_labels": fig_labels,
        "pooled_official": pooled_official,
        "pooled_by_fold": pooled_by_fold,
    }
    if auc_mismatches:
        degraded.append(
            "statistics.json disagrees with the stage-3 predictions it is supposed "
            "to summarise: " + "; ".join(auc_mismatches) +
            ". Re-run stage 4 against the current results directory before using "
            "any number in this report.")
    write_results_md(out_dir / "RESULTS.md", ctx)

    machine = {
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "degraded": degraded,
        "primary_cohort": PRIMARY_COHORT,
        "primary_cohort_present": primary_present,
        "cohorts": {
            cv.cohort: {
                "verdict": cv.verdict,
                "reason": cv.reason,
                "role": "primary" if cv.is_primary else "exploratory",
                "c1_level": c1_levels.get(cv.cohort),
                "criteria": [
                    {"code": c.code, "key": c.key, "status": c.status,
                     "rule": c.rule, "detail": c.detail, "evidence": c.evidence,
                     # Which headline this criterion quoted, and whether it
                     # showed the two sides of any difference share a test set.
                     # Written so the invariants are auditable from the file and
                     # not only from the process that produced it.
                     "headline_key": c.headline_key or None,
                     "headline_point": (None if math.isnan(c.headline_point)
                                        else c.headline_point),
                     "differenced_against_headline": c.differenced,
                     "same_test_set_verified": c.test_set_verified}
                    for c in cv.criteria
                ],
                "headlines": {k: {"point": pt, "lo": lo, "hi": hi}
                              for k, (pt, lo, hi) in sorted(cv.headlines.items())},
            } for cv in verdicts
        },
        # Confound cohorts are recorded, but under a different key and with no
        # `verdict` field at all, so a machine reader cannot iterate "cohorts"
        # and find a coil-count result sitting among the diagnostic ones.
        "confound_cohorts": {
            cohort: {
                "label": ctxc["spec"].label_long,
                "label_target_from_cache": ctxc["label_target"],
                "paired": ctxc["spec"].paired,
                "interpretation": "HIGH AUC HERE IS EVIDENCE AGAINST THE PHASE "
                                  "HYPOTHESIS; this cohort has no diagnostic label and "
                                  "carries no verdict",
                "n_test_subjects": (ctxc["pooled"].n_clusters if ctxc["pooled"] else None),
                "auc": {cond: {"point": e.point, "lo": e.lo, "hi": e.hi,
                               "n": e.n, "n_clusters": e.n_clusters}
                        for cond, e in ctxc["per_cond"].items() if e.ok},
            } for cohort, ctxc in sorted(confound_context.items())
        },
        "cross_validation": {
            cohort: {
                "scheme": p.scheme,
                "n_folds": len(p.folds),
                "folds": p.folds,
                "n_slices": p.n, "n_pos_slices": p.n_pos,
                "n_subjects": p.n_clusters, "n_pos_subjects": p.n_pos_clusters,
                # Recomputed, not asserted: the sum of the folds' subject counts
                # equals the pooled subject count exactly when no subject is in
                # two folds. `pool_folds_oof` refuses to build this object
                # otherwise, so this is a second, independent statement of the
                # same property rather than a claim taken on trust.
                # "tested once" and "all tested" are different claims and are
                # recorded separately. The first is about double-counting within
                # what was pooled; the second is about whether what was pooled is
                # the cohort at all.
                "every_subject_tested_once": bool(
                    sum(int(f["n_clusters"]) for f in p.per_fold) == p.n_clusters),
                "coverage_complete": bool(p.coverage_complete),
                "expected_folds": p.expected_folds or None,
                "missing_folds": p.missing_folds or None,
                "expected_subjects": p.expected_subjects,
                "uncovered_subjects": p.uncovered_subjects or None,
                "coverage_defect": p.coverage_defect or None,
                "coverage_source": p.coverage_source or None,
                "conditions_share_fold_set": cohort not in fold_set_mismatch,
                "fold_set_mismatch": fold_set_mismatch.get(cohort) or None,
                "split_family": p.split_family,
                "per_fold": p.per_fold,
                "official_split_secondary": (
                    {"n_slices": pooled_official[(cohort, HEADLINE_CONDITION)].n,
                     "n_subjects": pooled_official[(cohort, HEADLINE_CONDITION)].n_clusters}
                    if (cohort, HEADLINE_CONDITION) in pooled_official else None),
                "stage4_scheme_mismatch": scheme_mismatch.get(cohort) or None,
            }
            for cohort in cohorts
            for p in [pooled.get((cohort, HEADLINE_CONDITION))]
            if p is not None and p.is_pooled_oof
        },
        "figures": {label: p for label, p in figures},
    }
    (out_dir / "verdict.json").write_text(json.dumps(machine, indent=2))
    logger.info("verdict -> %s", out_dir / "verdict.json")
    return 0, ctx


_BS_CACHE: Dict[Tuple[int, int, int, bool], dict] = {}


def _bootstrap_cache(p: PooledPredictions, n_boot: int, seed: int,
                     fpr_grid: Optional[np.ndarray] = None) -> dict:
    """Memoized cluster bootstrap; the ROC and bar figures ask for the same one."""
    key = (id(p), int(n_boot), int(seed), fpr_grid is not None)
    if key not in _BS_CACHE:
        _BS_CACHE[key] = cluster_bootstrap(p.labels, p.probs, p.clusters,
                                           n_boot=n_boot, seed=seed, fpr_grid=fpr_grid)
    return _BS_CACHE[key]


# ==========================================================================
# Self-test: synthetic statistics + runs, exercising every verdict branch
# ==========================================================================

def _d_for_auc(auc: float) -> float:
    """
    Separation d giving AUC under the equal-variance normal model (Rice & Harris
    2005): AUC = Phi(d / sqrt 2), so d = sqrt(2) * Phi^-1(AUC).
    """
    a = min(max(float(auc), 1e-6), 1 - 1e-6)
    # Acklam's rational approximation to the normal quantile is overkill here;
    # the bisection is exact to 1e-12 and needs no extra dependency.
    lo, hi = -12.0, 12.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if 0.5 * (1.0 + math.erf(mid / math.sqrt(2.0))) < a:
            lo = mid
        else:
            hi = mid
    return math.sqrt(2.0) * 0.5 * (lo + hi)


def _synth_run(cohort: str, condition: str, seed: int, auc_target: float,
               region: str = "full", n_subj: int = 4, n_pos_subj: int = 3,
               slices: int = 30, subj_offset: int = 0, idx_offset: int = 0,
               pos_rate: float = 0.25, auc_exact: Optional[float] = None) -> dict:
    """
    One synthetic stage-3 run.

    `subj_offset` / `idx_offset` exist so that several runs can be made to look
    like DISJOINT cross-validation folds of one cohort (distinct subjects,
    distinct cache rows) -- or, by leaving them equal, like folds that overlap,
    which is the input the out-of-fold pooler has to refuse.

    `pos_rate` is the fraction of a positive subject's slices that are positive.
    1.0 gives the confound-cohort shape: the label is a property of the whole
    acquisition, so every slice of a subject carries the same class.

    `auc_exact` requests predictions whose expected AUC is that number. Use it
    whenever a test asserts something ABOUT the AUC of the synthetic run.
    `auc_target` cannot be used for that: it is a separation knob, and for every
    value the original scenarios pass it produces near-complete separation.
    """
    # zlib.crc32, not hash(): Python randomises string hashing per process, and a
    # self-test whose synthetic data changes between runs cannot assert anything.
    rng = np.random.default_rng(
        zlib.crc32(f"{cohort}|{condition}|{seed}|{region}|{subj_offset}".encode())
        % (2 ** 31))
    probs, labels, pids, cidx = [], [], [], []
    k = idx_offset
    for s in range(subj_offset, subj_offset + n_subj):
        pid = f"{cohort[:4].upper()}{s:03d}"
        pos_subject = (s - subj_offset) < n_pos_subj
        for _ in range(slices):
            lab = int(pos_subject and rng.random() < pos_rate)
            if auc_exact is None:
                # Original generator, kept byte-for-byte so every pre-existing
                # scenario draws exactly the numbers it always drew.
                base = rng.random()
                probs.append(float(np.clip(base * (1 - auc_target) + lab * auc_target
                                           + rng.normal(0, 0.05), 0, 1)))
            else:
                # Scores whose expected AUC is `auc_exact`. Under the standard
                # equal-variance normal model AUC = Phi(d / sqrt 2), so
                # d = sqrt(2) * Phi^-1(AUC); the logistic squash into (0, 1) is
                # monotone and therefore leaves the AUC unchanged.
                # `auc_target` is NOT an AUC -- it is a separation knob that
                # happens to give near-perfect separation for every value the
                # existing scenarios pass -- so a test that needs a control to
                # sit at 0.52 rather than 1.00 has to say so explicitly.
                probs.append(float(1.0 / (1.0 + math.exp(
                    -(rng.normal(_d_for_auc(auc_exact) * lab, 1.0))))))
            labels.append(lab)
            pids.append(pid)
            cidx.append(k)
            k += 1
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, pr = np.array(labels), np.array(probs)
    auc = float(roc_auc_score(y, pr)) if len(np.unique(y)) > 1 else float("nan")
    ap = float(average_precision_score(y, pr)) if len(np.unique(y)) > 1 else float("nan")
    hist = [{"epoch": e, "train_loss": 0.7 - 0.05 * e, "lr": 1e-4,
             "val_loss": 0.7 - 0.03 * e, "val_auc": 0.5 + 0.03 * e,
             "val_ap": 0.2 + 0.01 * e, "seconds": 1.0 * e} for e in range(6)]
    return {
        "cohort": cohort, "condition": condition, "seed": seed, "region": region,
        "device": "cpu", "timestamp": "2026-01-01T00:00:00", "wall_seconds": 1.0,
        "model": {"arch": "resnet18", "in_channels": 2},
        "hyperparams": {"lr": 1e-4, "epochs": 6},
        "splits": {"training": {"n_slices": 100, "n_patients": 10, "n_pos": 20},
                   "validation": {"n_slices": 40, "n_patients": 4, "n_pos": 8},
                   "test": {"n_slices": len(labels), "n_patients": n_subj,
                            "n_pos": int(y.sum())}},
        "best_epoch": 3, "best_selection_score": 0.62, "history": hist,
        "val": {"probs": probs, "labels": labels, "patient_ids": pids,
                "cache_idx": cidx, "auc": auc, "ap": ap, "loss": 0.6,
                "n": len(labels), "n_pos": int(y.sum())},
        "test": {"probs": probs, "labels": labels, "patient_ids": pids,
                 "cache_idx": cidx, "auc": auc, "ap": ap, "loss": 0.6,
                 "n": len(labels), "n_pos": int(y.sum())},
        "checkpoint": "/dev/null",
    }


def _synth_stats(cohort: str, phase_lo: float, p_holm: float,
                 seeds=(42, 123, 7), preferred_evaluable: bool = True,
                 n_clusters: int = 30, n_pos_clusters: int = 12,
                 phase_slice: Optional[Tuple[float, float, float]] = None,
                 phase_auc: float = 0.78,
                 n_slices: int = 122, n_pos_slices: int = 11,
                 split_family: Optional[str] = None,
                 folds: Optional[Sequence[int]] = None,
                 per_cond_size: Optional[Dict[str, Tuple[int, int, Sequence[int]]]] = None,
                 ) -> dict:
    """
    Synthesise statistics.json in the exact shape s04_stats.py emits.

    Shape verified against real s04 output: a flat `runs` list, a flat
    `across_seeds` list, and per-seed per-level `comparisons` where the
    patient-level rows carry `preferred: true`.

    `n_clusters` / `n_pos_clusters` are the test-fold cluster counts C1's
    minimum-size gate reads. The default (30 / 12) is a fold big enough for the
    gate to let C1 be decided at all; scenario O_* drops it to the real
    prostate_dwi fold (4 / 3) to check the gate fires.

    `phase_slice` overrides the SLICE-level phase block only, leaving the
    patient-level block at (phase_auc, phase_lo, ...). That is the shape of
    defect [1]: a slice interval that excludes chance sitting on top of a
    patient interval that does not.

    `split_family` / `folds` / `per_cond_size` describe a CROSS-VALIDATED sweep
    as s04 records one: every run record carries `split_family`, `pooled` and
    `folds`, and `per_cond_size` = {condition: (n_slices, n_clusters, folds)}
    lets one condition describe a smaller pool than its neighbours -- which is
    exactly what stage 4 writes when a fold died. Defaults leave the file
    byte-identical to what the pre-existing scenarios always got.
    """
    def level_block(auc, lo, hi, n=None, ncl=None, cluster_level=False):
        n_sl = n_slices if n is None else n
        n_cl = n_clusters if ncl is None else ncl
        return {"auc": auc, "ci_lo": lo, "ci_hi": hi,
                "ci_method": "cluster_bootstrap_percentile_95",
                "cluster_unit": "subject_id",
                "n_slices": n_cl if cluster_level else n_sl,
                "n_pos_slices": n_pos_slices,
                "n_clusters": n_cl, "n_pos_clusters": n_pos_clusters,
                "n_boot_requested": 2000, "n_boot_used": 1204,
                "n_skipped_single_class": 796, "n_skipped_single_cluster": 0,
                "n_dropped_nonfinite": 0}

    def size_of(cond):
        """(n_slices, n_clusters, folds) for one condition."""
        if per_cond_size and cond in per_cond_size:
            n, ncl, fs = per_cond_size[cond]
            return int(n), int(ncl), list(fs)
        return n_slices, n_clusters, (list(folds) if folds else None)

    aucs = {"magnitude": (0.60, 0.44, 0.75), "phase": (phase_auc, phase_lo, 0.91),
            "both": (0.80, 0.63, 0.92)}
    # The slice-level block is what C4/C5/C7/C8 compare controls against; the
    # patient block is what C1 reads. Keeping them separately addressable is the
    # whole point of the parameter.
    slice_override = {"phase": phase_slice} if phase_slice else {}
    runs, across, comparisons = [], [], []
    for cond, (auc, lo, hi) in aucs.items():
        sl = slice_override.get(cond) or (auc, lo, hi)
        c_n, c_ncl, c_folds = size_of(cond)
        cv_keys = ({"split_family": split_family, "pooled": c_folds is not None,
                    "folds": c_folds, "n_folds": len(c_folds) if c_folds else None}
                   if split_family is not None else {})
        for sd in seeds:
            runs.append({
                "tag": f"{cohort}_{cond}_seed{sd}", "cohort": cohort,
                "condition": cond, "seed": sd, "region": "full",
                "reported_test_auc": sl[0], "cluster_unit": "subject_id",
                "cluster_source": "cache_index+cohort_csv", "auc_matches_stage3": True,
                "slice_level": level_block(*sl, n=c_n, ncl=c_ncl),
                "patient_level_mean": level_block(auc + 0.02, lo + 0.02, hi + 0.02,
                                                  n=c_n, ncl=c_ncl, cluster_level=True),
                "patient_level_max": level_block(auc - 0.01, lo - 0.01, hi - 0.01,
                                                 n=c_n, ncl=c_ncl, cluster_level=True),
                "operating_point": {"chosen_on": "validation", "reason": None},
                "warnings": [],
                **cv_keys,
            })
        across.append({
            "cohort": cohort, "condition": cond, "region": "full",
            "n_runs": len(seeds), "seeds": list(seeds),
            "slice_auc": {"n": len(seeds), "mean": sl[0], "sd": 0.01,
                          "min": sl[0] - 0.01, "max": sl[0] + 0.01,
                          "values": [sl[0]] * len(seeds), "reason": None},
            "patient_mean_auc": {"n": len(seeds), "mean": auc + 0.02, "sd": 0.01,
                                 "min": auc + 0.01, "max": auc + 0.03,
                                 "values": [auc + 0.02] * len(seeds), "reason": None},
            "caveat": "seeds share one test fold; this SD is training stochasticity",
            **({"split_families": [split_family], "mixes_split_families": False}
               if split_family is not None else {}),
        })

    delta = aucs["phase"][0] - aucs["magnitude"][0]
    for level in ("slice", "patient_mean", "patient_max"):
        preferred = (level == "patient_mean")
        evaluable = (not preferred) or preferred_evaluable
        for sd in seeds:
            comparisons.append({
                "cohort": cohort, "seed": sd, "region": "full",
                "model_a": "phase", "model_b": "magnitude",
                "tag_a": f"{cohort}_phase_seed{sd}",
                "tag_b": f"{cohort}_magnitude_seed{sd}",
                "level": level, "cluster_unit": "subject_id",
                "n_cases": n_slices, "n_pos": n_pos_slices,
                "n_clusters": size_of("phase")[1],
                **({"split_family": split_family,
                    "pooled": size_of("phase")[2] is not None,
                    "folds_a": size_of("phase")[2],
                    "folds_b": size_of("magnitude")[2]}
                   if split_family is not None else {}),
                "delong": {"test": "delong_correlated_roc",
                           "auc_a": aucs["phase"][0], "auc_b": aucs["magnitude"][0],
                           "diff": delta,
                           "p": (p_holm / 3.0) if evaluable else None,
                           "ci_lo_diff": delta - 0.15, "ci_hi_diff": delta + 0.15,
                           "reason": None if evaluable else
                                     "too few positive clusters for a defined variance",
                           "caveat": "slices within a patient are correlated"},
                "cluster_bootstrap_diff": {"diff": delta, "ci_lo": delta - 0.12,
                                           "ci_hi": delta + 0.12, "p": 0.02,
                                           "reason": None},
                "p_raw": (p_holm / 3.0) if evaluable else None,
                "preferred": preferred,
                "p_holm": p_holm if evaluable else None,
                "holm_family": f"{cohort}/{level}",
            })

    return {
        "generated": "2026-01-01T00:00:00",
        "config": {"n_boot": 2000, "bootstrap_seed": 0, "alpha": 0.05,
                   "cluster_unit_requested": "auto",
                   "ci_method": "percentile bootstrap, resampling clusters"},
        "methods_note": "AUC CIs resample subjects, not slices.",
        "runs": runs, "across_seeds": across, "comparisons": comparisons,
        "holm": {"method": "holm-bonferroni step-down", "family_size": 3},
        "warnings": [],
    }


def _synth_control_payloads(cohort: str, perm: float, bg: float, scramble: float,
                            acq: float, confound: float,
                            n_permutations: int = 20,
                            control_ci: bool = True,
                            perm_aucs: Optional[Sequence[float]] = None,
                            n_clusters: int = 30,
                            destroy_ci: Optional[dict] = None,
                            bg_extra: Optional[dict] = None,
                            perm_duplicate: bool = False,
                            acq_variants: Optional[Sequence[Tuple[str, float]]] = None,
                            confound_specs: Optional[Sequence[Tuple[str, float, str]]] = None,
                            ci_n: int = 122, ci_n_pos: int = 11,
                            ) -> List[dict]:
    """
    Synthesise the per-run control JSONs s05_controls.py writes.

    One file per run, s03 schema plus `control` and `control_detail`; the
    filename carries {cohort}__{control}__{variant}__{condition}__seed{N}.

    `control_ci=False` omits `control_detail.test_auc_ci95` from the
    background/scramble/acquisition payloads, which is how a stage-5 run whose
    bootstrap degenerated actually looks on disk. That is the shape that used to
    be awarded a free PASS on C4/C7 (scenario Q_*).

    `perm_aucs`, when given, replaces the jittered null with an explicit list of
    permutation-replicate AUCs, so a null that the headline does NOT beat can be
    written down exactly (scenario P_*).

    `n_clusters` / `ci_n` / `ci_n_pos` must track the values handed to
    `_synth_stats`: stage 5 scores its controls on the same test fold as the
    headline, and C8 refuses the comparison when the two disagree. They are
    parameters rather than constants so that a CROSS-VALIDATED tree -- whose
    pooled fold is hundreds of slices, not 122 -- can be written down.

    The remaining arguments exist for the aggregation-layer scenarios, and each
    one writes a tree that is individually plausible on disk:

    `destroy_ci`      keys merged into the background/scramble bootstrap block
                      only -- a wide interval, [0, 1], [-inf, +inf], a two-cluster
                      fold, a tied (zero-width) interval, or n_boot_ok = 0
                      (A2/A2b/A9/A13/DG_*). Scoped to the destroy-controls so
                      the permutation fold still matches the headline and C8 is
                      not perturbed.
    `bg_extra`        a SECOND background-only run with its own AUC and bootstrap
                      block, so the two are enveloped together (A1).
    `perm_duplicate`  write ONE permutation replicate, copied to
                      `n_permutations` filenames (A3).
    `acq_variants`    explicit (variant, auc) pairs for acquisition_split, so two
                      spellings of one direction can be written (A4).
    `confound_specs`  explicit (target, auc, label_semantics) triples, so a
                      high-AUC confound run can carry the wrong semantics and be
                      dropped from the maximum (A5).
    """
    def ci(auc, half=0.13, override=None):
        blk = {"auc": auc, "lo": max(0.0, auc - half), "hi": min(1.0, auc + half),
               "n": ci_n, "n_pos": ci_n_pos, "n_clusters": n_clusters,
               "n_boot_ok": 1180, "n_boot_degenerate": 820,
               "method": "patient-clustered percentile bootstrap, B=2000"}
        blk.update(override or {})
        return blk

    def payload(control, auc, condition="phase", seed=42, variant=None,
                label_semantics="diagnosis", detail=None, region="full",
                with_ci=True, ci_override=None):
        n = 40
        labels = [1] * 11 + [0] * (n - 11)
        # Distinct per run, as real stage-5 predictions are. Two control runs
        # that produced byte-identical predictions are one run stored twice,
        # which is exactly what `_replicate_fingerprint` has to be able to see;
        # a generator that emits the same probability vector for every replicate
        # would make every replicate look like a duplicate of every other.
        rngp = np.random.default_rng(
            zlib.crc32(f"{cohort}|{control}|{variant}|{seed}|{auc:.6f}".encode())
            % (2 ** 31))
        probs = ([float(np.clip(0.75 + 0.25 * rngp.random(), 0, 1)) for _ in range(11)]
                 + [float(np.clip(0.25 * rngp.random(), 0, 1)) for _ in range(n - 11)])
        det = {"variant": variant, "label_semantics": label_semantics,
               "subject_col": "subject_id", "transform": None, "wall_seconds": 1.0}
        if with_ci:
            det["test_auc_ci95"] = ci(auc, override=ci_override)
        det.update(detail or {})
        return {
            "cohort": cohort, "condition": condition, "seed": seed, "region": region,
            "device": "cpu", "timestamp": "2026-01-01T00:00:00", "wall_seconds": 1.0,
            "model": {"arch": "resnet18", "in_channels": 2},
            "hyperparams": {"lr": 1e-4, "epochs": 6},
            "splits": {"training": {"n_slices": 995, "n_patients": 33, "n_pos": 48},
                       "validation": {"n_slices": 242, "n_patients": 8, "n_pos": 43},
                       "test": {"n_slices": n, "n_patients": 4, "n_pos": 11}},
            "best_epoch": 3, "best_selection_score": auc,
            "history": [{"epoch": e, "train_loss": 0.7 - 0.05 * e, "lr": 1e-4,
                         "val_loss": 0.7, "val_auc": auc, "val_ap": 0.3,
                         "seconds": float(e)} for e in range(4)],
            "val": None,
            "test": {"probs": probs, "labels": labels,
                     "patient_ids": [f"S{i % 4}" for i in range(n)],
                     "cache_idx": list(range(n)), "auc": auc, "ap": 0.5,
                     "loss": 0.6, "n": n, "n_pos": 11},
            "checkpoint": None,
            "control": control, "control_detail": det,
        }

    out: List[dict] = []
    rng = np.random.default_rng(7)
    if perm_duplicate:
        # One replicate, stored n_permutations times. Every file is a complete,
        # well-formed stage-5 permutation payload; nothing about any single one
        # of them is wrong. Only the COUNT is a lie.
        one = payload("label_permutation", float(perm), seed=1000, variant="perm00",
                      detail={"permutation_seed": 1000,
                              "permutation_unit": "subject_id"})
        out.extend(json.loads(json.dumps(one)) for _ in range(n_permutations))
    else:
        if perm_aucs is None:
            perm_aucs = [float(np.clip(perm + float(rng.normal(0, 0.04)), 0, 1))
                         for _ in range(n_permutations)]
        for i, a in enumerate(perm_aucs):
            out.append(payload("label_permutation", float(a), seed=1000 + i,
                               variant=f"perm{i:02d}",
                               detail={"permutation_seed": 1000 + i,
                                       "permutation_unit": "subject_id"}))
    out.append(payload("background_only", bg, region="background", with_ci=control_ci,
                       ci_override=destroy_ci))
    if bg_extra is not None:
        out.append(payload("background_only", float(bg_extra["auc"]), region="background",
                           seed=int(bg_extra.get("seed", 43)),
                           ci_override=bg_extra.get("ci_override")))
    out.append(payload("phase_scramble", scramble, with_ci=control_ci,
                       ci_override=destroy_ci,
                       detail={"scramble_seed": 20240517, "scope": "within body mask only"}))
    # s05_controls._acquisition_split writes variant = f"{train_arm}2{test_arm}",
    # i.e. "A2B" / "B2A"; the synthetic tree uses the same spelling so the
    # direction parser is exercised on the real one.
    for direction, val in (acq_variants or (("A2B", acq), ("B2A", acq + 0.06))):
        out.append(payload("acquisition_split", float(val), variant=direction,
                           detail={"strat_key": "institution", "direction": direction}))
    for target, val, sem in (confound_specs or
                             (("receiver_channels", confound, f"confound:receiver_channels"),
                              ("institution", confound - 0.1, "confound:institution"))):
        out.append(payload("confound_predictability", float(val), variant=target,
                           label_semantics=sem,
                           detail={"target": target, "column": target}))
    return out


def _write_control_payloads(dirpath: Path, payloads: Sequence[dict]) -> None:
    """
    Write payloads under s05's real naming scheme.

    Two payloads can legitimately collide on that name -- a replicate copied to
    a second location is the whole point of A3_dup_perm_reps -- so a collision
    gets a `__copyN` suffix rather than overwriting. That is what a duplicated
    controls tree looks like on disk: N distinct filenames, one experiment.
    """
    dirpath.mkdir(parents=True, exist_ok=True)
    for pl in payloads:
        det = pl["control_detail"]
        parts = [str(pl["cohort"]), str(pl["control"])]
        if det.get("variant"):
            parts.append(str(det["variant"]))
        parts += [str(pl["condition"]), f"seed{pl['seed']}"]
        stem = "__".join(parts)
        path = dirpath / (stem + ".json")
        k = 1
        while path.exists():
            path = dirpath / f"{stem}__copy{k}.json"
            k += 1
        path.write_text(json.dumps(pl))


_MD_VERDICT_RE = re.compile(r"^### (?P<cohort>\S+): \*\*(?P<verdict>[A-Z ]+)\*\*$", re.M)


def _verdicts_claimed_in_markdown(md: str) -> Dict[str, str]:
    """Read back the per-cohort verdict headings actually written to RESULTS.md."""
    return {m.group("cohort"): m.group("verdict") for m in _MD_VERDICT_RE.finditer(md)}


def self_test(dpi: int = 120) -> int:
    """
    Build synthetic stage-3/4/5 outputs -- in the exact formats those stages
    emit -- and run the whole report path on them.

    Each scenario asserts, independently:
      * the named criteria came out with the expected status (pass/fail/missing);
      * the verdict follows the documented resolution order;
      * RESULTS.md and verdict.json on disk carry that same verdict, not a
        nicer one.

    Scenarios A-M cover the original rule set. Scenarios N-S are regression
    tests for four confirmed defects in the verdict logic. Scenarios A1-A13,
    DG_* and A_slug_* are regression tests for seven defects in the AGGREGATION
    layer -- the steps that build the numbers the thresholds are applied to.
    Every one of them is a constructed input that used to print SUPPORTED.

    Rules that are easy to get wrong, and the scenario that pins each:

    D_no_stage4  statistics.json absent. s06 falls back to its own
                 subject-clustered bootstrap for the headline AUC, which is
                 slice-level only, so C1 is MISSING (the cluster-level estimate
                 does not exist) and C2 is MISSING (DeLong has no fallback).
                 The assertion is "anything but SUPPORTED", because which of the
                 remaining criteria pass depends on the fallback point estimate
                 rather than on the rule being tested.

    K_slice_only Only the anti-conservative slice-level DeLong test is
                 evaluable and it IS significant. C2 must come out MISSING,
                 not PASS: significance under a test stage 4 itself labels
                 anti-conservative is not evidence the criterion was met.

    N_c1_...     DEFECT 1, part 1. The slice-level phase interval excludes
                 chance ([0.849, 1.000]) while the patient-level interval does
                 not ([0.000, 1.000]) -- the real prostate_dwi numbers. C1 was
                 decided on the slice CI and PASSED. It must now FAIL.

    O_c1_...     DEFECT 1, part 2. The patient-level interval excludes chance,
                 but the fold is 4 clusters / 3 positive. At that size the C1
                 rule fires 24.9% of the time under a complete null, so C1 must
                 be MISSING -- not PASS, and not FAIL either.

    P_headline_  DEFECT 5. The headline (0.62) sits inside the permutation null
                 stage 5 measured (mean 0.590; 6 of 20 replicates reach it).
                 C1 and C3 both pass -- C1 because it compares to a hard-coded
                 0.500, C3 because the null mean is within its loose band. C8
                 must FAIL: the empirical p is 7/21 = 0.333.

    Q_scanner_   DEFECT 6. A pure scanner fingerprint: headline 0.90,
                 background 0.79, scramble 0.79, confound 0.79, and the
                 destroy-controls carry no bootstrap block. Every one of
                 C4/C6/C7 used to PASS. All three must now FAIL.

    R_exploratory DEFECT 7. Two cohorts, both clean. Only the pre-registered
                 primary may be SUPPORTED; the other must be INCONCLUSIVE with
                 C0 MISSING, and RESULTS.md must say so in the verdict table.

    S_perm_...   DEFECT 5, boundary. 19 replicates cannot produce p < 0.05 even
                 with zero exceedances, so C8 must be MISSING rather than PASS.

    The aggregation-layer scenarios all share one shape: the per-control
    thresholds are satisfied, and the number the threshold was applied to is not
    a measurement. Nine of the eleven put a background/scramble control at AUC
    0.790 under a headline of 0.900 -- a scanner fingerprint, the alternative
    explanation this study exists to exclude -- and none of them touches a
    threshold constant.

    A1_envelope_ Two background-only runs at 0.790: one with a tight interval
                 [0.660, 0.920], one whose bootstrap returned [0.000, 1.000].
                 min(lo)/max(hi) hulls them into [0.000, 0.920], which contains
                 0.500, so C4 read the fingerprint as a collapse. C4 must be
                 MISSING: the two runs are not repeats of one quantity.

    A2 / A2b /   One control, one interval, too noisy to decide anything:
    A9 / A13 /   [0.39, 1.00]; [0.00, 1.00]; [-inf, +inf] (JSON round-trips
    DG_tied /    Infinity, and inf is not NaN, so `has_ci` was True); a
    DG_empty     bootstrap on 2 subjects; a zero-width interval sitting exactly
                 on 0.500; and a plausible interval with n_boot_ok = 0. All six
                 satisfy "the CI contains chance". All six must be MISSING.

    A3_dup_perm  One permutation replicate stored under 20 filenames. Every
                 file is a well-formed stage-5 payload; only the count is
                 false, and the count is what makes C8 evaluable
                 ((0+1)/(20+1) = 0.048 < 0.05). C3 and C8 must be MISSING.

    A4_one_dir   One acquisition direction written as both `A2B` and `A_to_B`.
                 "Both directions present" was a count of distinct variant
                 strings, so the reverse split never had to be run. C5 MISSING.

    A5_confound  The coil-count confound run -- the measured mechanism of this
                 study, at AUC 0.97 -- carries `label_semantics="diagnosis"`
                 and is dropped from the target set. C6 is a MAXIMUM over
                 targets, so the drop lowers it to the survivor at 0.45 and C6
                 passed. It must be MISSING while any confound run is unscored.

    A_slug_...   The controls tree names its cohort `prostatet2`; the results
                 tree names it `prostate_t2`. Separator-blind matching credited
                 them to each other. Every control criterion must be MISSING.

    The SF_*, CVM/CVC/CVD, FL and CU checks at the end are regression tests for
    six further defects, all in the layer that decides WHICH predictions a number
    describes. Each is a constructed tree that reached a flattering answer, and
    each has been mutation-checked: with its guard removed, its check fails.

    SF_split_family_*  Three trees with byte-identical CV folds and byte-identical
                 stage-5 controls. Two of them also hold ONE leftover 4-subject
                 official-split file at AUC 1.000 -- in `results/official/` and at
                 the results root. Stats.estimate keyed on (cohort, condition,
                 region), so it averaged the pooled out-of-fold row with the
                 leftover and labelled the blend "3 seed(s)": the 70-subject
                 headline moved 0.691 -> 0.794, more than the entire margin C4
                 and C7 test, and NOT SUPPORTED became SUPPORTED. Separately,
                 whether the sample-size guard noticed depended on whether the
                 stale directory name sorted before or after 'cv'. All three
                 trees must now give the same criteria and the same verdict.

    CVM_missing_fold  phase loses fold 4; magnitude and both keep all five, and
                 stage 4 honestly describes 4 folds / 56 subjects, so nothing
                 else flags it. pool_folds_oof refused only when EXACTLY ONE fold
                 was on disk, so this pooled four, called itself "every subject
                 tested exactly once", and decided C1 on 56 of 70 subjects.
                 C1/C2 must be MISSING and the shortfall must appear in the
                 banner, in 5a and in verdict.json.

    CVC_complete_folds  The same tree with all five folds. Must still reach
                 SUPPORTED: a guard that refuses unconditionally is not a check.

    CVD_design_table  Three folds on disk for EVERY condition, and the
                 `cv<k>_split` columns stage 1/2 wrote declare five. The
                 cross-condition fallback cannot see this -- every condition
                 agrees, because the same two folds died for all of them. Only
                 the declared design knows 28 of 70 subjects were never tested.

    FL_figures   Reusing CVM: the overlaid ROC, the AUC bars and the per-fold
                 panel were annotated with the sample size and scheme of the
                 FIRST usable condition, so the label was numerically wrong for
                 every other curve or bar. Asserted in BOTH directions -- the
                 shared one-liner must come back on CVC, because a label that
                 always hedges carries no information either.

    CU_cluster_unit  The cache index carries subject_id (stage 4's first choice)
                 and the stage-1 cohort CSV has one basename that does not join.
                 s06 demanded the join and dropped the WHOLE cohort to
                 patient_id on that single miss, which on breast does not
                 collapse the repeat-scan groups.

    The last five checks are the fourth adversarial round. Every one of them is
    the same species of bug -- one number compared against another without
    establishing that the two describe the same experiment -- and every one of
    them printed SUPPORTED on the honest five-fold tree used above (headline
    0.6909 over 70 subjects, destroy-controls 0.061 below it, correctly NOT
    SUPPORTED with C4 and C7 failing).

    headline_not_overwritten_after_scoping  [R1] `Stats.estimate` scoped RUN
                 SELECTION to one split family and then took s04's across-seeds
                 mean whenever the across row did not declare its own families.
                 With one leftover 4-subject file at AUC 1.000 on disk the
                 70-subject headline went 0.691 -> 0.794, C4 and C7 flipped
                 FAIL -> PASS, and RESULTS.md printed "headline 0.794 [0.600,
                 0.780]" -- a point outside its own interval -- in the C4
                 evidence while C8 quoted 0.691 two sections later. The scenario
                 asserts the headline, both those invariants, and the verdict.

    missing_size_fails_closed  [R2b] C8's fold guard was `a is not None and b is
                 not None and a != b`, so the SAME null on the SAME wrong fold
                 was refused when the payload declared `n_clusters` and accepted
                 when the key was absent -- a guard defeated by DELETING
                 information. Both trees are asserted, plus the third instance of
                 the shape (permutation replicates that declare no
                 `permutation_seed`), plus the negative control: a null on the
                 headline's own fold is still decided.

    control_headline_share_a_test_set  [R3] C4/C5/C6/C7 subtracted a control AUC
                 from the headline with no test-set check at all, so one report
                 could refuse the comparison as meaningless for C8 and pass the
                 other four on exactly it. Live on the real tree: the only
                 prostate_t2 background control was scored on one fold's subjects
                 and differenced against the 67-subject pooled headline. Asserted
                 in all three directions -- refused across folds, decided on the
                 same fold, and a control that demonstrably SURVIVED still FAILS
                 whatever fold it ran on.

    comparison_covers_one_test_set  [R4] `comparison_levels` carried s04's
                 `folds_a` / `folds_b` through the file and read them nowhere,
                 and the stage-4 size guard ran on HEADLINE_CONDITION only, so
                 nothing checked the reference condition. C2 was a Holm-adjusted
                 p-value between phase on 70 subjects and magnitude on 56. Both
                 the mismatch and the undeclared case are asserted.

    The AG_* checks at the very end are the FIFTH adversarial round, and they sit
    one level below R2b/R3/R4: those guards ask an object for its test-set
    fingerprint, and every object they ask is a POOL that used to synthesise one
    out of whichever member happened to have it. All four reached SUPPORTED on
    the same honest five-fold tree (0.6909 over 70 subjects, destroy-controls at
    0.6299, i.e. NOT SUPPORTED with C4 and C7 failing), and each is asserted in
    both directions -- a consensus rule that refuses everything is not a check.

    envelope_fingerprint_is_consensus  [AG1/AG2] `_envelope` reported
                 `n = good[0].n` and `n_clusters = min(clusters)`. An honest
                 background/scramble control at 0.6299 on 560 slices / 70
                 subjects, enveloped with a second run at 0.5300 scored on 40
                 slices of 200 OTHER subjects, hulled to 0.580 and reported the
                 headline's own "560 slices / 70 subjects": C4 and C7 flipped
                 FAIL -> PASS on a 0.111 drop that is half a different test set.
                 AG2 is the version `min()` cannot see (same 70 subjects, 200 of
                 the 560 slices) and AG2b the silent-member version. All three
                 must be MISSING; the same second run on the SAME test set must
                 still PASS, and the tree without it must still FAIL.

    null_pool_fingerprint_is_consensus  [AG3] `base = _control_estimate(runs[0])`
                 handed the permutation null whatever the first file on disk
                 declared. Nineteen replicates on a 4-subject fold plus ONE on
                 the headline's 70-subject out-of-fold set satisfied C8's fold
                 check and gave p = (0+1)/(20+1) = 0.048. C8 must be MISSING in
                 both the all-wrong and the one-right tree, and C3 -- whose mean
                 and range are then a mixture of experiments -- must be MISSING
                 in the one-right tree too.

    comparison_group_has_two_checkable_sides  [AG4] the `folds_a` / `folds_b`
                 check ran only when BOTH conditions had a pooled prediction
                 vector, so a report holding no magnitude predictions at all --
                 deleted, or split over two sweeps the pooler refused to merge --
                 skipped it and decided C2 on a Holm p-value whose reference side
                 the report cannot identify. C2 must be MISSING in both, and the
                 group's own per-seed records must agree with each other before
                 the group reports a size at all.

    replicate_count_from_predictions  [AG5] `_replicate_fingerprint` hashes the
                 whole payload minus a hand-written denylist of volatile keys, so
                 a field the list has never heard of counts as substance. Twenty
                 copies of ONE replicate collapse to one draw; the same twenty
                 with a per-epoch wall clock varied counted as twenty and C8
                 passed at 1/21 out of a single permutation. The count now comes
                 from the predictions, and 20 real permutations must still be 20.

    cv_sweeps_never_merged  [R5] `_split_family` collapsed every fold-tagged run
                 into `cv` whatever directory it lived in, and the pooler grouped
                 folds by index alone, so `sweepA/<cohort>_cv0` and
                 `sweepB/<cohort>_cv0` were averaged together. An honest sweep at
                 0.63 blended with an optimistic one at 0.995 gave one "pooled
                 out-of-fold" headline of 0.938 over the same 70 subjects. The
                 ordinary root layout must be untouched, which is asserted too.
    """
    logging.basicConfig(level=logging.WARNING, format="%(levelname)-7s %(message)s")
    cohort = PRIMARY_COHORT
    other = "prostate_dwi"
    root = Path(tempfile.mkdtemp(prefix="s06_selftest_"))
    failures = 0

    base = dict(stats=True, controls=True, phase_lo=0.62, p_holm=0.01, perm=0.50,
                bg=0.55, scramble=0.55, acq=0.74, confound=0.55,
                n_permutations=20, preferred_evaluable=True,
                n_clusters=30, n_pos_clusters=12, phase_slice=None,
                phase_auc=0.78, control_ci=True, perm_aucs=None,
                cohorts=(PRIMARY_COHORT,),
                # aggregation-layer knobs (scenarios A1-A13, DG_*)
                destroy_ci=None, bg_extra=None, perm_duplicate=False,
                acq_variants=None, confound_specs=None,
                controls_cohort_alias=None)

    def cfg(**over):
        d = dict(base)
        d.update(over)
        return d

    # The audit's measured null for defect 5, padded to 20 replicates. Six of
    # them reach the 0.62 headline, so p = (6+1)/(20+1) = 0.333.
    D5_NULL = [0.44, 0.52, 0.58, 0.61, 0.63, 0.66, 0.55, 0.73,
               0.44, 0.52, 0.58, 0.61, 0.63, 0.66, 0.55, 0.73,
               0.44, 0.52, 0.58, 0.61]

    # (name, config, expected verdict or None = "anything but SUPPORTED",
    #  {code: expected status})  -- verdict/status may be dicts keyed by cohort
    scenarios = [
        ("A_all_pass", cfg(),
         "SUPPORTED", {"C0": "pass", "C1": "pass", "C2": "pass", "C3": "pass",
                       "C4": "pass", "C5": "pass", "C6": "pass", "C7": "pass",
                       "C8": "pass"}),
        ("B_permutation_fails", cfg(perm=0.72),
         "NOT SUPPORTED", {"C3": "fail"}),
        ("C_confound_fails", cfg(confound=0.95),
         "NOT SUPPORTED", {"C6": "fail"}),
        ("D_no_stage4", cfg(stats=False),
         None, {"C1": "missing", "C2": "missing"}),
        ("E_no_stage5", cfg(controls=False),
         "INCONCLUSIVE", {"C0": "pass", "C1": "pass", "C2": "pass",
                          "C3": "missing", "C4": "missing", "C5": "missing",
                          "C6": "missing", "C7": "missing", "C8": "missing"}),
        ("F_ci_touches_chance", cfg(phase_lo=0.46),
         "NOT SUPPORTED", {"C1": "fail"}),
        ("G_background_survives", cfg(bg=0.76),
         "NOT SUPPORTED", {"C4": "fail"}),
        ("H_holm_not_significant", cfg(p_holm=0.31),
         "NOT SUPPORTED", {"C2": "fail"}),
        ("I_acq_erases_effect", cfg(acq=0.49),
         "NOT SUPPORTED", {"C5": "fail"}),
        ("J_nothing_but_stage3", cfg(stats=False, controls=False),
         "INCONCLUSIVE", {"C1": "missing", "C2": "missing", "C3": "missing",
                          "C4": "missing", "C5": "missing", "C6": "missing",
                          "C7": "missing", "C8": "missing"}),
        ("K_slice_only_delong", cfg(preferred_evaluable=False),
         "INCONCLUSIVE", {"C2": "missing"}),
        ("L_scramble_survives", cfg(scramble=0.77),
         "NOT SUPPORTED", {"C7": "fail"}),
        ("M_too_few_permutations", cfg(n_permutations=2),
         "INCONCLUSIVE", {"C3": "missing", "C8": "missing"}),

        # ---- regression tests for the four confirmed verdict-logic defects ----
        # [1] C1 was read off the slice-level CI, which undercovers.
        ("N_c1_slice_ci_flatters", cfg(phase_lo=0.00, phase_slice=(0.918, 0.849, 1.0),
                                       phase_auc=0.778, acq=0.86, bg=0.50,
                                       scramble=0.50),
         "NOT SUPPORTED", {"C1": "fail"}),
        # [1] ... and the cluster-level CI is itself unusable on a 4-patient fold.
        ("O_c1_too_few_clusters", cfg(n_clusters=4, n_pos_clusters=3,
                                      phase_lo=0.85, phase_slice=(0.918, 0.849, 1.0),
                                      phase_auc=0.90, acq=0.86, bg=0.50,
                                      scramble=0.50),
         "INCONCLUSIVE", {"C1": "missing"}),
        # [5] headline inside the empirical null; C1 and C3 both still pass.
        ("P_headline_inside_null", cfg(perm_aucs=D5_NULL, phase_auc=0.62,
                                       phase_lo=0.55, phase_slice=(0.62, 0.55, 0.71),
                                       bg=0.45, scramble=0.45, acq=0.65,
                                       confound=0.45),
         "NOT SUPPORTED", {"C0": "pass", "C1": "pass", "C2": "pass", "C3": "pass",
                           "C4": "pass", "C5": "pass", "C6": "pass", "C7": "pass",
                           "C8": "fail"}),
        # [6] scanner fingerprint: controls high, and with no bootstrap block.
        ("Q_scanner_fingerprint", cfg(phase_auc=0.90, phase_lo=0.80,
                                      bg=0.79, scramble=0.79, confound=0.79,
                                      acq=0.86, control_ci=False),
         "NOT SUPPORTED", {"C4": "fail", "C6": "fail", "C7": "fail"}),
        # [7] a second, non-pre-registered cohort cannot be SUPPORTED.
        ("R_exploratory_cohort", cfg(cohorts=(PRIMARY_COHORT, other)),
         {PRIMARY_COHORT: "SUPPORTED", other: "INCONCLUSIVE"},
         {PRIMARY_COHORT: {"C0": "pass", "C1": "pass"},
          other: {"C0": "missing", "C1": "pass", "C2": "pass", "C3": "pass",
                  "C4": "pass", "C5": "pass", "C6": "pass", "C7": "pass",
                  "C8": "pass"}}),
        # [5] boundary: 19 replicates cannot reach p < 0.05, so C8 is MISSING.
        ("S_perm_too_coarse_for_p", cfg(n_permutations=MIN_PERM_REPLICATES_FOR_P - 1),
         "INCONCLUSIVE", {"C3": "pass", "C8": "missing"}),

        # ---- regression tests for the SEVEN aggregation-layer defects --------
        # Every one of these was a route to SUPPORTED after the per-control
        # THRESHOLDS had been fixed. None of them attacks a threshold: each
        # attacks the step that builds the number the threshold is applied to.
        # In all of them the headline is 0.900 and the destroy-controls sit at
        # 0.790 -- a scanner fingerprint, the alternative explanation the whole
        # study exists to exclude -- or the control is simply unmeasured.
        #
        # [A1] one junk run enveloped with a clean one. min(lo)/max(hi) over the
        # two hulls the clean run's [0.660, 0.920] into [0.000, 0.920], which
        # "contains 0.500", so C4 recorded the fingerprint as a collapse.
        ("A1_envelope_launder",
         cfg(phase_auc=0.90, phase_lo=0.80, acq=0.86, bg=0.79,
             bg_extra={"auc": 0.79, "seed": 43, "ci_override": {"lo": 0.0, "hi": 1.0}}),
         "INCONCLUSIVE", {"C0": "pass", "C1": "pass", "C2": "pass", "C3": "pass",
                          "C4": "missing", "C5": "pass", "C6": "pass",
                          "C7": "pass", "C8": "pass"}),
        # [A2] a single control whose interval is simply too wide to decide
        # anything. Noise is REWARDED by "the interval contains 0.500".
        ("A2_wide_ci_control",
         cfg(phase_auc=0.90, phase_lo=0.80, acq=0.86, bg=0.79, scramble=0.79,
             destroy_ci={"lo": 0.39, "hi": 1.0}),
         "INCONCLUSIVE", {"C4": "missing", "C7": "missing", "C6": "pass"}),
        # [A2b] the limiting case of A2: the whole AUC axis.
        ("A2b_ci_0_to_1",
         cfg(phase_auc=0.90, phase_lo=0.80, acq=0.86, bg=0.79, scramble=0.79,
             destroy_ci={"lo": 0.0, "hi": 1.0}),
         "INCONCLUSIVE", {"C4": "missing", "C7": "missing"}),
        # [A9] +/-inf is not NaN, so `has_ci` was True and lo <= 0.5 <= hi held
        # trivially. JSON round-trips Infinity, so this shape reaches disk.
        ("A9_infinite_ci",
         cfg(phase_auc=0.90, phase_lo=0.80, acq=0.86, bg=0.79, scramble=0.79,
             destroy_ci={"lo": float("-inf"), "hi": float("inf")}),
         "INCONCLUSIVE", {"C4": "missing", "C7": "missing"}),
        # [A13] the C1 power floor, absent from C4/C7: a two-subject cluster
        # bootstrap. The interval looks perfectly ordinary; it is measured on
        # two people.
        ("A13_two_cluster_control",
         cfg(destroy_ci={"n_clusters": 2}),
         "INCONCLUSIVE", {"C4": "missing", "C7": "missing", "C5": "pass"}),
        # [DG] every resample returned the same AUC: a zero-width "interval"
        # sitting exactly on chance, which trivially contains chance.
        ("DG_tied",
         cfg(destroy_ci={"lo": 0.5, "hi": 0.5}),
         "INCONCLUSIVE", {"C4": "missing", "C7": "missing"}),
        # [DG] a bootstrap block with a plausible interval and ZERO valid
        # resamples behind it.
        ("DG_empty_bootstrap",
         cfg(destroy_ci={"n_boot_ok": 0, "n_boot_degenerate": 2000}),
         "INCONCLUSIVE", {"C4": "missing", "C7": "missing"}),
        # [A3] one permutation replicate stored under 20 filenames. Every file
        # is a valid stage-5 payload; only the COUNT is false, and the count is
        # what makes C8 evaluable at all ((0+1)/(20+1) = 0.048 < 0.05).
        ("A3_dup_perm_reps", cfg(perm_duplicate=True),
         "INCONCLUSIVE", {"C3": "missing", "C8": "missing"}),
        # [A4] one split direction written under two variant strings. "Both
        # directions" was a count of distinct strings, not of distinct
        # directions, so the reverse split never had to be run.
        ("A4_one_direction_twice",
         cfg(acq_variants=(("A2B", 0.86), ("A_to_B", 0.86))),
         "INCONCLUSIVE", {"C5": "missing", "C4": "pass", "C7": "pass"}),
        # [A5] C6 is a MAXIMUM over confound targets, so a target dropped for
        # malformed semantics can only lower it. The coil-count run -- the
        # measured mechanism of this whole study, at AUC 0.97 -- was dropped
        # with a log line and C6 passed on the survivor at 0.45.
        ("A5_confound_semantics_drop",
         cfg(confound_specs=(("receiver_channels", 0.97, "diagnosis"),
                             ("institution", 0.45, "confound:institution"))),
         "INCONCLUSIVE", {"C6": "missing", "C4": "pass", "C7": "pass"}),
        # [A-slug] the controls tree names its cohort `prostatet2`; the results
        # tree names it `prostate_t2`. `_slug` deleted the separator and matched
        # them, crediting a different cohort's controls to the primary one.
        ("A_slug_cohort_near_miss",
         cfg(controls_cohort_alias=_slug(PRIMARY_COHORT)),
         "INCONCLUSIVE", {"C0": "pass", "C1": "pass", "C2": "pass",
                          "C3": "missing", "C4": "missing", "C5": "missing",
                          "C6": "missing", "C7": "missing", "C8": "missing"}),
    ]

    print("=" * 82)
    print("s06 SELF-TEST -- synthetic stage-3/4/5 outputs in the real formats")
    print("=" * 82)

    for name, c, want_verdict, want_status in scenarios:
        case = root / name
        res = case / "results"
        res.mkdir(parents=True, exist_ok=True)
        merged_stats: Optional[dict] = None
        for co in c["cohorts"]:
            for cond, target in (("magnitude", 0.35), ("phase", 0.55), ("both", 0.58)):
                for seed in (42, 123, 7):
                    r = _synth_run(co, cond, seed, target)
                    (res / f"{co}_{cond}_seed{seed}.json").write_text(json.dumps(r))
            if c["stats"]:
                st = _synth_stats(co, c["phase_lo"], c["p_holm"],
                                  preferred_evaluable=c["preferred_evaluable"],
                                  n_clusters=c["n_clusters"],
                                  n_pos_clusters=c["n_pos_clusters"],
                                  phase_slice=c["phase_slice"],
                                  phase_auc=c["phase_auc"])
                if merged_stats is None:
                    merged_stats = st
                else:
                    for k in ("runs", "across_seeds", "comparisons"):
                        merged_stats[k].extend(st[k])
            if c["controls"]:
                # Written where s05 really puts them: a sibling of results/.
                _write_control_payloads(
                    case / "controls" / "results",
                    _synth_control_payloads(c["controls_cohort_alias"] or co,
                                            c["perm"], c["bg"], c["scramble"],
                                            c["acq"], c["confound"],
                                            n_permutations=c["n_permutations"],
                                            control_ci=c["control_ci"],
                                            perm_aucs=c["perm_aucs"],
                                            n_clusters=c["n_clusters"],
                                            destroy_ci=c["destroy_ci"],
                                            bg_extra=c["bg_extra"],
                                            perm_duplicate=c["perm_duplicate"],
                                            acq_variants=c["acq_variants"],
                                            confound_specs=c["confound_specs"]))
        if merged_stats is not None:
            (res / "statistics.json").write_text(json.dumps(merged_stats))

        ns = argparse.Namespace(
            results_dir=str(res), cache_dir=str(case / "cache"),
            cohorts_dir=str(case / "cohorts"), out=str(case / "report"),
            stats=None, controls=None, bootstrap=200, seed=0, dpi=dpi,
        )
        rc, ctx = build_report(ns)
        by_cohort = {cv.cohort: cv for cv in ctx["verdicts"]}
        md = (case / "report" / "RESULTS.md").read_text()
        claimed = _verdicts_claimed_in_markdown(md)
        vj = (json.loads((case / "report" / "verdict.json").read_text())
              if (case / "report" / "verdict.json").exists() else None)

        # Normalise "one cohort" and "several cohorts" into the same shape.
        want_v = want_verdict if isinstance(want_verdict, dict) else {cohort: want_verdict}
        want_s = (want_status if all(isinstance(v, dict) for v in want_status.values())
                  else {cohort: want_status})

        problems: List[str] = []
        for co in c["cohorts"]:
            cv = by_cohort.get(co)
            if cv is None:
                problems.append(f"{co}: no verdict produced")
                continue
            by_code = {c_.code: c_.status for c_ in cv.criteria}
            for code, want in (want_s.get(co) or {}).items():
                if by_code.get(code) != want:
                    problems.append(f"{co}/{code} is {by_code.get(code)}, expected {want}")
            wv = want_v.get(co, "__any_but_supported__")
            if wv is None or wv == "__any_but_supported__":
                if cv.verdict == "SUPPORTED":
                    problems.append(f"{co}: verdict is SUPPORTED but should not be")
            elif cv.verdict != wv:
                problems.append(f"{co}: verdict {cv.verdict}, expected {wv}")
            if claimed.get(co) != cv.verdict:
                problems.append(f"{co}: RESULTS.md claims {claimed.get(co)!r}, engine "
                                f"said {cv.verdict!r}")
            if claimed.get(co) == "SUPPORTED" and any(s != "pass" for s in by_code.values()):
                problems.append(f"{co}: RESULTS.md claims SUPPORTED with a "
                                f"non-passing criterion")
            if vj is None:
                problems.append("verdict.json not written")
            elif vj["cohorts"][co]["verdict"] != cv.verdict:
                problems.append(f"{co}: verdict.json disagrees with the engine")
            # An exploratory cohort must be labelled as such wherever it is read.
            if not cv.is_primary:
                if cv.verdict == "SUPPORTED":
                    problems.append(f"{co}: SUPPORTED on a non-primary cohort")
                if vj is not None and vj["cohorts"][co].get("role") != "exploratory":
                    problems.append(f"{co}: verdict.json does not mark it exploratory")
                if "exploratory" not in md:
                    problems.append(f"{co}: RESULTS.md does not label it exploratory")
        if not sorted((case / "report" / "figures").glob("*.png")):
            problems.append("no figures produced")

        shown = by_cohort.get(cohort) or ctx["verdicts"][0]
        flagged = ",".join(f"{k}:{v}" for k, v in
                           sorted({c_.code: c_.status for c_ in shown.criteria}.items())
                           if v != "pass")
        print(f"  {name:<26} {shown.verdict:<14} {flagged or 'all pass':<40} "
              f"{'OK' if not problems else 'FAILED'}")
        for pr in problems:
            print(f"      - {pr}")
        failures += bool(problems)

    # Direct unit checks on the invariant everything else rests on.
    def _full(status: str = "pass") -> List[Criterion]:
        return [Criterion(k, code, r, status, "d") for code, k, r in CRITERIA_ORDER]

    bogus_cases = [
        ("fail + SUPPORTED",
         CohortVerdict("x", "SUPPORTED", [Criterion("k", "C1", "r", "fail", "d")],
                       "bogus", is_primary=True)),
        ("missing + SUPPORTED",
         CohortVerdict("x", "SUPPORTED", [Criterion("k", "C1", "r", "missing", "d")],
                       "bogus", is_primary=True)),
        ("fail + INCONCLUSIVE",
         CohortVerdict("x", "INCONCLUSIVE", [Criterion("k", "C1", "r", "fail", "d")],
                       "bogus", is_primary=True)),
        # Defect [7]: an exploratory cohort must never carry a confirmatory
        # verdict, even if every criterion it was given happens to pass.
        ("all pass + SUPPORTED on a non-primary cohort",
         CohortVerdict("x", "SUPPORTED", _full("pass"), "bogus", is_primary=False)),
        # A criterion silently dropped from the list must not buy a SUPPORTED.
        ("SUPPORTED with C8 never evaluated",
         CohortVerdict("x", "SUPPORTED", [c for c in _full("pass") if c.code != "C8"],
                       "bogus", is_primary=True)),
        # ---- the three headline invariants (fourth adversarial round) -------
        # A point estimate outside its own interval is what a headline read from
        # two different scopes looks like in print: the real report said
        # "headline 0.794 [0.600, 0.780]".
        ("a headline whose point lies outside its own interval",
         CohortVerdict("x", "NOT SUPPORTED",
                       [Criterion("k", "C4", "r", "fail", "d",
                                  headline_key="x/phase@slice", headline_point=0.794)],
                       "bogus", is_primary=True,
                       headlines={"x/phase@slice": (0.794, 0.600, 0.780)})),
        # Two criteria quoting two different headlines for one cohort: C4 read
        # 0.794 while C8 read 0.691, in the same file, and neither could see it.
        ("two criteria quoting different headlines for one cohort",
         CohortVerdict("x", "NOT SUPPORTED",
                       [Criterion("k", "C4", "r", "fail", "d",
                                  headline_key="x/phase@slice", headline_point=0.794),
                        Criterion("k", "C8", "r", "fail", "d",
                                  headline_key="x/phase@slice", headline_point=0.691)],
                       "bogus", is_primary=True,
                       headlines={"x/phase@slice": (0.691, 0.600, 0.780)})),
        # A criterion that differenced a control against the headline and passed
        # without checking the two share a test set.
        ("PASS on an unverified difference against the headline",
         CohortVerdict("x", "SUPPORTED",
                       [Criterion(k, code, r, "pass", "d", differenced=(code == "C4"),
                                  test_set_verified=False)
                        for code, k, r in CRITERIA_ORDER],
                       "bogus", is_primary=True)),
        # ---- aggregate provenance (fifth adversarial round) ----------------
        # A pool reporting the fingerprint of ONE of its members. This is the
        # shape every route in the fifth round took, and it must not depend on
        # the verdict: a NOT SUPPORTED report whose controls borrowed a test set
        # is broken too.
        ("an envelope reporting a member's test set as the group's",
         CohortVerdict("x", "NOT SUPPORTED", _full("fail"), "bogus", is_primary=True,
                       aggregates=[{"what": "2 background-only run(s)",
                                    "reported": {"n": 560, "n_clusters": 70},
                                    "members": [{"n": 560, "n_clusters": 70},
                                                {"n": 40, "n_clusters": 200}]}])),
        # ... and the silent-member half of the same rule: a member that
        # declares nothing cannot be shown to share what the group reports.
        ("a null pool reporting a fingerprint one replicate never declared",
         CohortVerdict("x", "INCONCLUSIVE", _full("missing"), "bogus", is_primary=True,
                       aggregates=[{"what": "20 label-permutation replicate(s)",
                                    "reported": {"n_clusters": 70},
                                    "members": [{"n_clusters": 70},
                                                {"n_clusters": None}]}])),
    ]
    checks = 0
    for label, bogus in bogus_cases:
        try:
            _assert_verdict_consistent(bogus)
            print(f"  invariant_guard            FAILED ({label} accepted)")
            failures += 1
        except AssertionError:
            checks += 1
    print(f"  invariant_guard            OK ({checks}/{len(bogus_cases)} illegal "
          f"combinations rejected)")

    # ---- controls kept UNDER results/ must not be pooled into the headline ---
    # `<results-dir>/controls` is one of the directories s06 searches for stage-5
    # output, and it is inside the tree load_runs() walks. A stage-5 payload is a
    # stage-3 payload plus two keys, so without the `control` guard in
    # _looks_like_run it is indistinguishable from a training run: the scramble
    # and acquisition payloads carry region="full" and the confound payloads
    # carry SCANNER labels, and all of them would be pooled into the phase
    # headline the controls exist to falsify.
    case = root / "T_controls_inside_results"
    res = case / "results"
    res.mkdir(parents=True, exist_ok=True)
    for cond, target in (("magnitude", 0.35), ("phase", 0.55), ("both", 0.58)):
        for seed in (42, 123, 7):
            (res / f"{cohort}_{cond}_seed{seed}.json").write_text(
                json.dumps(_synth_run(cohort, cond, seed, target)))
    ctl_payloads = _synth_control_payloads(cohort, 0.50, 0.55, 0.55, 0.74, 0.55)
    _write_control_payloads(res / "controls", ctl_payloads)     # INSIDE results/
    ns = argparse.Namespace(results_dir=str(res), cache_dir=str(case / "cache"),
                            cohorts_dir=str(case / "cohorts"), out=str(case / "report"),
                            stats=None, controls=None, bootstrap=100, seed=0, dpi=dpi)
    _, ctx_t = build_report(ns)
    pooled_t = ctx_t["pooled"].get((cohort, HEADLINE_CONDITION))
    # n_subj * slices from _synth_run; pool_runs AVERAGES seeds over the shared
    # fold rather than concatenating, so three seeds still give 120 rows. The
    # control payloads carry cache_idx 0..39, and pool_runs intersects indices,
    # so contamination shows up as a SHRUNK pooled fold (40) whose probabilities
    # are the mean of the model and its own falsification controls.
    clean_n = 4 * 30
    t_problems = []
    if pooled_t is None:
        t_problems.append("no pooled headline predictions at all")
    elif pooled_t.n != clean_n:
        t_problems.append(f"headline pooled {pooled_t.n} rows, expected {clean_n} "
                          f"-- stage-5 control payloads were loaded as stage-3 runs")
    # ... and the controls must still be FOUND there, or the guard has simply
    # hidden them instead of separating them.
    if not ctx_t["verdicts"] or all(
            c.status == "missing" for c in ctx_t["verdicts"][0].criteria
            if c.code in {"C3", "C4", "C7"}):
        t_problems.append("controls under results/controls were not discovered")
    print(f"  controls_not_pooled        {'OK' if not t_problems else 'FAILED'}"
          + (f" (headline rows {pooled_t.n if pooled_t else 'n/a'} == {clean_n})"
             if not t_problems else ""))
    for pr in t_problems:
        print(f"      - {pr}")
    failures += bool(t_problems)

    # ======================================================================
    # Confound cohorts and cross-validation
    # ======================================================================
    extra_checks = 0

    def _run_case(name: str, build, bootstrap: int = 120
                  ) -> Tuple[dict, str, Optional[dict]]:
        """
        Materialise a results tree, run build_report, return (ctx, md, verdict.json).

        `bootstrap` is a parameter because a control POOLED out-of-fold gets its
        interval from s06's own cluster bootstrap, so the resample count is the
        one this call passes rather than the `n_boot_ok` a per-fold payload
        declares -- and MIN_BOOT_VALID_CONTROL (200) is a real floor that 120
        resamples do not clear. The real report runs at 2000.
        """
        case = root / name
        res = case / "results"
        res.mkdir(parents=True, exist_ok=True)
        build(res)
        ns_ = argparse.Namespace(
            results_dir=str(res), cache_dir=str(case / "cache"),
            cohorts_dir=str(case / "cohorts"), out=str(case / "report"),
            stats=None, controls=None, bootstrap=int(bootstrap), seed=0, dpi=dpi)
        _, c_ = build_report(ns_)
        md_ = (case / "report" / "RESULTS.md").read_text()
        vj_ = json.loads((case / "report" / "verdict.json").read_text())
        return c_, md_, vj_

    def _report(name: str, probs: List[str], note: str = "") -> None:
        nonlocal failures, extra_checks
        extra_checks += 1
        print(f"  {name:<26} {'OK' if not probs else 'FAILED'}"
              + (f" ({note})" if not probs and note else ""))
        for pr_ in probs:
            print(f"      - {pr_}")
        failures += bool(probs)

    def _write_clean_clinical(res: Path, cohort_: str = PRIMARY_COHORT) -> None:
        """The A_all_pass tree: a cohort that reaches SUPPORTED on its own."""
        for cond, target in (("magnitude", 0.35), ("phase", 0.55), ("both", 0.58)):
            for sd in (42, 123, 7):
                (res / f"{cohort_}_{cond}_seed{sd}.json").write_text(
                    json.dumps(_synth_run(cohort_, cond, sd, target)))
        (res / "statistics.json").write_text(json.dumps(
            _synth_stats(cohort_, base["phase_lo"], base["p_holm"])))
        _write_control_payloads(
            res.parent / "controls" / "results",
            _synth_control_payloads(cohort_, base["perm"], base["bg"], base["scramble"],
                                    base["acq"], base["confound"]))

    def _write_confound(res: Path, cohort_: str, phase_auc: float,
                        mag_auc: float = 0.55) -> None:
        """A confound cohort: unpaired, one class per subject, 20 test subjects."""
        d = res / f"confound_{cohort_}"
        d.mkdir(parents=True, exist_ok=True)
        for cond, target in (("magnitude", mag_auc), ("phase", phase_auc),
                             ("both", phase_auc)):
            for sd in (42, 123):
                (d / f"{cohort_}_{cond}_seed{sd}.json").write_text(json.dumps(
                    _synth_run(cohort_, cond, sd, 0.55, n_subj=20, n_pos_subj=10,
                               slices=4, pos_rate=1.0, auc_exact=target)))

    # ---- [CF1] a confound cohort carries no verdict, and tightens C6 ---------
    # brain has no tumour label. It must be absent from the verdict table and
    # from verdict.json["cohorts"], and its measured coil-count predictability
    # must FAIL C6 on the clinical cohort -- the one direction the wiring runs.
    def _b(res: Path) -> None:
        _write_clean_clinical(res)
        _write_confound(res, "brain", phase_auc=0.90)
    ctx_cf, md_cf, vj_cf = _run_case("CF1_confound_no_verdict", _b)
    p_cf: List[str] = []
    if any(v.cohort == "brain" for v in ctx_cf["verdicts"]):
        p_cf.append("brain was given a diagnostic verdict")
    if "brain" in (vj_cf.get("cohorts") or {}):
        p_cf.append("verdict.json lists brain among the diagnostic cohorts")
    if "brain" not in (vj_cf.get("confound_cohorts") or {}):
        p_cf.append("verdict.json does not record brain as a confound cohort")
    if re.search(r"^### brain: \*\*", md_cf, re.M):
        p_cf.append("RESULTS.md gives brain a verdict heading")
    if ctx_cf["confound_cohorts"] != ["brain"]:
        p_cf.append(f"confound cohorts are {ctx_cf['confound_cohorts']}, expected ['brain']")
    if not ctx_cf["external_confounds"]:
        p_cf.append("the brain measurement was not wired into C6 at all")
    prim = next((v for v in ctx_cf["verdicts"] if v.cohort == PRIMARY_COHORT), None)
    c6 = next((c for c in (prim.criteria if prim else []) if c.code == "C6"), None)
    if c6 is None or c6.status != "fail":
        p_cf.append(f"C6 is {c6.status if c6 else 'absent'}, expected fail "
                    f"(a directly measured confound at 0.90 must fail it)")
    if prim is None or prim.verdict != "NOT SUPPORTED":
        p_cf.append(f"{PRIMARY_COHORT} verdict is "
                    f"{prim.verdict if prim else 'absent'}, expected NOT SUPPORTED")
    # ... and the inversion must be stated in words, not left to the reader.
    for phrase in ("evidence AGAINST", "no tumour label"):
        if phrase not in md_cf:
            p_cf.append(f"RESULTS.md never says {phrase!r}")
    _report("confound_no_verdict", p_cf,
            "brain excluded from the verdict table; its 0.90 coil-count AUC fails C6")

    # ---- [CF2] ... and it can never make C6 PASS ----------------------------
    # Same tree with the confound sitting at 0.55 and NO stage-5 controls. C6
    # must stay MISSING: a low external number is not a substitute for a control
    # that was never run on this cohort.
    def _b2(res: Path) -> None:
        for cond, target in (("magnitude", 0.35), ("phase", 0.55), ("both", 0.58)):
            for sd in (42, 123, 7):
                (res / f"{PRIMARY_COHORT}_{cond}_seed{sd}.json").write_text(
                    json.dumps(_synth_run(PRIMARY_COHORT, cond, sd, target)))
        (res / "statistics.json").write_text(json.dumps(
            _synth_stats(PRIMARY_COHORT, base["phase_lo"], base["p_holm"])))
        _write_confound(res, "brain", phase_auc=0.52, mag_auc=0.52)
    ctx_c2, md_c2, _ = _run_case("CF2_external_cannot_pass_c6", _b2)
    p_c2: List[str] = []
    prim2 = next((v for v in ctx_c2["verdicts"] if v.cohort == PRIMARY_COHORT), None)
    c6b = next((c for c in (prim2.criteria if prim2 else []) if c.code == "C6"), None)
    if c6b is None or c6b.status != "missing":
        p_c2.append(f"C6 is {c6b.status if c6b else 'absent'}, expected missing -- a low "
                    f"external confound must not stand in for an unrun control")
    if prim2 is not None and prim2.verdict == "SUPPORTED":
        p_c2.append("SUPPORTED reached with no stage-5 controls at all")
    _report("external_cannot_pass_c6", p_c2,
            "a low external confound leaves C6 MISSING, never PASS")

    # ---- [CF3] the phase>magnitude ORDERING claim is gated on a paired CI ----
    # The mechanism sentence ("phase is the BETTER predictor of the acquisition
    # property") used to fire on the bare sign of the point difference, so a
    # hair's-breadth gap whose interval spans zero rendered as a finding. On the
    # real brain cohort that gap is +0.007 with a paired CI of [-0.038, +0.051].
    # A tie must read as a tie; a real ordering must still read as one, so both
    # directions are asserted -- a gate that always hedges is also broken.
    BETTER = "BETTER predictor of the acquisition property"

    def _b_tie(res: Path) -> None:
        _write_clean_clinical(res)
        _write_confound(res, "brain", phase_auc=0.905, mag_auc=0.900)
    _, md_tie, _ = _run_case("CF3_confound_ordering_tie", _b_tie)
    p_c3: List[str] = []
    if BETTER in md_tie:
        p_c3.append("a +0.005 difference whose paired CI includes zero still "
                    "rendered as 'phase is the BETTER predictor'")
    if "no ordering between them is claimed" not in md_tie:
        p_c3.append("the tie is not reported as a tie")
    _report("confound_ordering_needs_interval", p_c3,
            "a point difference whose paired CI spans zero is reported as a tie, "
            "not as a finding")

    def _b_real(res: Path) -> None:
        _write_clean_clinical(res)
        _write_confound(res, "brain", phase_auc=0.95, mag_auc=0.50)
    _, md_real, _ = _run_case("CF4_confound_ordering_real", _b_real)
    p_c4: List[str] = []
    if BETTER not in md_real:
        p_c4.append("a 0.95-vs-0.50 separation was NOT reported as phase being the "
                    "better predictor -- the gate is hedging away a real ordering")
    _report("confound_ordering_still_reported", p_c4,
            "a separation the data actually support is still claimed")

    # ---- [CV1] out-of-fold pooling: full power, one estimate ----------------
    def _b3(res: Path) -> None:
        for f in range(3):
            d = res / f"{PRIMARY_COHORT}_cv{f}"
            d.mkdir(parents=True, exist_ok=True)
            for cond, target in (("magnitude", 0.35), ("phase", 0.55), ("both", 0.58)):
                for sd in (42, 123):
                    (d / f"{PRIMARY_COHORT}_{cond}_seed{sd}.json").write_text(json.dumps(
                        _synth_run(PRIMARY_COHORT, cond, sd, target,
                                   n_subj=5, n_pos_subj=3, slices=10,
                                   subj_offset=5 * f, idx_offset=1000 * f)))
        (res / "statistics.json").write_text(json.dumps(
            _synth_stats(PRIMARY_COHORT, base["phase_lo"], base["p_holm"])))
    ctx_cv, md_cv, vj_cv = _run_case("CV1_out_of_fold_pooling", _b3)
    p_cv: List[str] = []
    pool_cv = ctx_cv["pooled"].get((PRIMARY_COHORT, HEADLINE_CONDITION))
    if pool_cv is None or not pool_cv.is_pooled_oof:
        p_cv.append("the headline is not the pooled out-of-fold vector")
    else:
        if pool_cv.n != 150:
            p_cv.append(f"pooled {pool_cv.n} slices, expected 150 (3 folds x 5 subjects "
                        f"x 10 slices) -- folds were not concatenated")
        if pool_cv.n_clusters != 15:
            p_cv.append(f"pooled {pool_cv.n_clusters} subjects, expected 15")
        if len(pool_cv.per_fold) != 3:
            p_cv.append(f"{len(pool_cv.per_fold)} per-fold records, expected 3")
    if len(ctx_cv["cv_rows"]) != 3:
        p_cv.append(f"{len(ctx_cv['cv_rows'])} pooled rows, expected one per condition")
    if not ctx_cv["fold_rows"]:
        p_cv.append("no per-fold dispersion table was produced")
    if "every subject is tested exactly once" not in md_cv:
        p_cv.append("RESULTS.md does not state that every subject is tested once")
    # ... and stage 4, which described a different fold, must not decide C1/C2.
    prim3 = next((v for v in ctx_cv["verdicts"] if v.cohort == PRIMARY_COHORT), None)
    by_code3 = {c.code: c.status for c in (prim3.criteria if prim3 else [])}
    if by_code3.get("C1") != "missing" or by_code3.get("C2") != "missing":
        p_cv.append(f"C1={by_code3.get('C1')}, C2={by_code3.get('C2')}; both must be "
                    f"MISSING while statistics.json describes a different fold")
    if prim3 is not None and prim3.verdict == "SUPPORTED":
        p_cv.append("SUPPORTED reached on stage-4 numbers for a fold not headlined")
    if not (vj_cv.get("cross_validation") or {}).get(PRIMARY_COHORT):
        p_cv.append("verdict.json records no cross-validation block")
    _report("cv_out_of_fold_pooling", p_cv,
            "3 disjoint folds -> 150 slices / 15 subjects, one estimate; C1/C2 capped")

    # ---- [CV2] overlapping folds are refused, not silently pooled -----------
    # Same subjects and same cache rows in two "folds". Concatenating them would
    # test everyone twice and shrink every interval by ~sqrt(2).
    def _b4(res: Path) -> None:
        for f in range(2):
            d = res / f"{PRIMARY_COHORT}_cv{f}"
            d.mkdir(parents=True, exist_ok=True)
            for cond, target in (("magnitude", 0.35), ("phase", 0.55), ("both", 0.58)):
                (d / f"{PRIMARY_COHORT}_{cond}_seed42.json").write_text(json.dumps(
                    _synth_run(PRIMARY_COHORT, cond, 42, target,
                               n_subj=5, n_pos_subj=3, slices=10)))   # no offsets
    ctx_ov, md_ov, _ = _run_case("CV2_overlapping_folds", _b4)
    p_ov: List[str] = []
    pool_ov = ctx_ov["pooled"].get((PRIMARY_COHORT, HEADLINE_CONDITION))
    if pool_ov is not None:
        p_ov.append(f"a headline estimate was produced anyway ({pool_ov.n} rows, "
                    f"scheme {pool_ov.scheme!r}); with overlapping folds and no "
                    f"official-split run there is nothing to headline")
    if not any("NOT disjoint" in d for d in ctx_ov["degraded"]):
        p_ov.append("the overlap was not reported in the degraded banner")
    if not any("NO headline estimate" in d for d in ctx_ov["degraded"]):
        p_ov.append("the report does not say the cohort has no headline")
    # The per-fold numbers must survive the refusal -- they are all there is.
    if not ctx_ov["fold_rows"]:
        p_ov.append("the per-fold diagnostic was dropped along with the pooling")
    ov_v = next((v for v in ctx_ov["verdicts"] if v.cohort == PRIMARY_COHORT), None)
    if ov_v is not None and ov_v.verdict == "SUPPORTED":
        p_ov.append("SUPPORTED reached with no headline estimate at all")
    _report("cv_overlapping_folds_refused", p_ov,
            "shared subjects across folds -> no headline, overlap reported, "
            "per-fold diagnostic kept")

    # ======================================================================
    # Split families, fold coverage, figure labels, cluster unit
    #
    # Regression tests for six confirmed defects. Every one of them is a
    # constructed tree that used to reach a flattering answer, and none of them
    # touches a threshold constant: each attacks the step that decides WHICH
    # numbers the thresholds are applied to.
    # ======================================================================
    CV_K, CV_SUBJ, CV_SLICES = 5, 14, 8          # folds, subjects/fold, slices/subject
    CV_N, CV_CL = CV_K * CV_SUBJ * CV_SLICES, CV_K * CV_SUBJ      # 560 slices, 70 subj
    CV_PHASE_SLICE = (0.6909, 0.60, 0.78)

    def _cv_runs(res: Path, drop=(), folds=CV_K) -> None:
        """
        `folds` disjoint CV folds x 3 conditions x 2 seeds. `drop`: (cond, fold).

        pos_rate=1.0 so the LABELS are identical across conditions, as they are
        in a real tree: the three conditions are three input channels for one set
        of subjects, not three datasets. Without it the synthetic labels differ
        per condition and the figures would legitimately report three different
        positive counts, which would hide the thing being tested.
        """
        for f in range(folds):
            d = res / f"{PRIMARY_COHORT}_cv{f}"
            d.mkdir(parents=True, exist_ok=True)
            for cond, target in (("magnitude", 0.35), ("phase", 0.55), ("both", 0.58)):
                if (cond, f) in drop:
                    continue
                for sd in (42, 123):
                    (d / f"{PRIMARY_COHORT}_{cond}_seed{sd}.json").write_text(json.dumps(
                        _synth_run(PRIMARY_COHORT, cond, sd, target,
                                   n_subj=CV_SUBJ, n_pos_subj=7, slices=CV_SLICES,
                                   pos_rate=1.0,
                                   subj_offset=CV_SUBJ * f, idx_offset=1000 * f)))

    def _cv_stats(per_cond_size=None, n=CV_N, ncl=CV_CL, folds=tuple(range(CV_K))) -> dict:
        return _synth_stats(PRIMARY_COHORT, 0.60, 0.01, seeds=(42, 123),
                            n_clusters=ncl, n_pos_clusters=35,
                            phase_slice=CV_PHASE_SLICE, phase_auc=CV_PHASE_SLICE[0],
                            n_slices=n, n_pos_slices=35,
                            split_family=CV_FAMILY, folds=list(folds),
                            per_cond_size=per_cond_size)

    def _cv_controls(case: Path, bg=0.55, scramble=0.55, n=CV_N, ncl=CV_CL) -> None:
        _write_control_payloads(
            case / "controls" / "results",
            _synth_control_payloads(PRIMARY_COHORT, 0.50, bg, scramble, 0.74, 0.52,
                                    n_permutations=20, n_clusters=ncl,
                                    ci_n=n, ci_n_pos=35))

    def _s04_order(st: dict) -> dict:
        """s04 writes `runs` sorted by str(unit_key) -- (cohort, region, family, ...)."""
        st["runs"].sort(key=lambda r: str((r.get("cohort"), r.get("region", "full"),
                                           Stats.family_of(r), r.get("condition"),
                                           r.get("seed"))))
        return st

    def _add_leftover(st: dict, res: Path, subdir: str, declare: bool = True) -> dict:
        """
        One leftover single-split phase run: 4 subjects, AUC 1.000.

        `subdir=''` is the results root (split family '.'); anything else is that
        directory's name. Both are written because whether the OLD size guard
        fired depended on where the alphabetical sort put the family key relative
        to 'cv' -- which is a property of the directory name and of nothing else.

        `declare=False` strips `split_families` and `mixes_split_families` from
        the across-seeds row, i.e. an s04 that simply does not write them. The
        blended mean is still there; only the label saying it is a blend is gone.
        """
        d = res / subdir if subdir else res
        d.mkdir(parents=True, exist_ok=True)
        (d / f"{PRIMARY_COHORT}_phase_seed7.json").write_text(json.dumps(
            _synth_run(PRIMARY_COHORT, "phase", 7, 0.99, n_subj=4, n_pos_subj=3,
                       slices=8, subj_offset=900, idx_offset=900_000)))
        fam = subdir or "."
        src = next(r for r in st["runs"]
                   if r["condition"] == "phase" and r["seed"] == 42)
        one = json.loads(json.dumps(src))
        one.update({"tag": f"{PRIMARY_COHORT}_phase_seed7", "seed": 7,
                    "split_family": fam, "pooled": False, "folds": None,
                    "n_folds": None})
        for lvl in ("slice_level", "patient_level_mean", "patient_level_max"):
            one[lvl] = dict(one[lvl], auc=1.0, ci_lo=0.95, ci_hi=1.0,
                            n_slices=32 if lvl == "slice_level" else 4,
                            n_pos_slices=12, n_clusters=4, n_pos_clusters=3)
        st["runs"].append(one)
        # s04's aggregate_across_seeds groups by (cohort, condition, region) ONLY,
        # so the leftover lands in the same across row -- and the row says so.
        for a in st["across_seeds"]:
            if a["condition"] != "phase":
                continue
            vals = [CV_PHASE_SLICE[0], CV_PHASE_SLICE[0], 1.0]
            pv = [CV_PHASE_SLICE[0] + 0.02, CV_PHASE_SLICE[0] + 0.02, 1.0]
            a["slice_auc"] = dict(a["slice_auc"], n=3, mean=sum(vals) / 3,
                                  min=min(vals), max=max(vals), values=vals)
            a["patient_mean_auc"] = dict(a["patient_mean_auc"], n=3, mean=sum(pv) / 3,
                                         min=min(pv), max=max(pv), values=pv)
            if declare:
                a.update(split_families=sorted({fam, CV_FAMILY}),
                         mixes_split_families=True)
            else:
                a.pop("split_families", None)
                a.pop("mixes_split_families", None)
        return st

    def _codes(ctx_: dict, cohort_: str = PRIMARY_COHORT) -> Tuple[str, Dict[str, str]]:
        cv_ = next((v for v in ctx_["verdicts"] if v.cohort == cohort_), None)
        return ((cv_.verdict if cv_ else "n/a"),
                {c.code: c.status for c in (cv_.criteria if cv_ else [])})

    # ---- [SF] a leftover single-split row is NEVER averaged in ---------------
    # Three trees with byte-identical CV folds and byte-identical stage-5
    # controls. The only difference is one leftover 4-subject official-split file
    # at AUC 1.000, present in two of them and in two different directories.
    #
    # Before: Stats.estimate keyed on (cohort, condition, region), so it averaged
    # the pooled out-of-fold row with the leftover and called the blend "3
    # seed(s)". The 70-subject headline moved 0.691 -> 0.794, which is more than
    # the whole margin C4 and C7 test, and both flipped FAIL -> PASS: NOT
    # SUPPORTED became SUPPORTED while the report went on describing the number
    # as the pooled out-of-fold estimate over 70 subjects.
    #
    # Also the DEFECT-5 half: whether the size guard noticed depended on whether
    # the leftover's directory name sorted before or after 'cv'. All three trees
    # must now give the same criteria and the same verdict.
    sf_out = {}
    for tag, subdir in (("clean", None), ("official", "official"), ("root", "")):
        def _bsf(res: Path, _sub=subdir) -> None:
            _cv_runs(res)
            st = _cv_stats()
            if _sub is not None:
                st = _add_leftover(st, res, _sub)
            (res / "statistics.json").write_text(json.dumps(_s04_order(st)))
            # Destroy-controls 0.061 below the true headline: FAIL on the honest
            # number, PASS on the blended one.
            _cv_controls(res.parent, bg=0.6299, scramble=0.6299)
        sf_out[tag] = _run_case(f"SF_split_family_{tag}", _bsf)

    p_sf: List[str] = []
    base_v, base_c = _codes(sf_out["clean"][0])
    if base_v != "NOT SUPPORTED" or base_c.get("C4") != "fail" or base_c.get("C7") != "fail":
        p_sf.append(f"the clean cv-only tree gives {base_v} with C4={base_c.get('C4')}, "
                    f"C7={base_c.get('C7')}; expected NOT SUPPORTED with both failing "
                    f"-- the scenario no longer tests what it was built to test")
    for tag in ("official", "root"):
        v, c = _codes(sf_out[tag][0])
        if v == "SUPPORTED":
            p_sf.append(f"{tag}: one leftover 4-subject file reached SUPPORTED")
        if (v, c) != (base_v, base_c):
            p_sf.append(f"{tag}: verdict/criteria differ from the identical cv-only "
                        f"tree ({v} {sorted(c.items())})")
        if not any("split famil" in d for d in sf_out[tag][0]["degraded"]):
            p_sf.append(f"{tag}: the second split family is not named in the degraded "
                        f"banner")
    # ... and the estimator itself must refuse to blend, whatever the caller does.
    merged = json.loads((root / "SF_split_family_official" / "results"
                         / "statistics.json").read_text())
    sx = Stats(merged, None)
    if sx.families_for(PRIMARY_COHORT, HEADLINE_CONDITION) != ["cv", "official"]:
        p_sf.append(f"families_for reports "
                    f"{sx.families_for(PRIMARY_COHORT, HEADLINE_CONDITION)}")
    if sx.estimate(PRIMARY_COHORT, HEADLINE_CONDITION, "slice").ok:
        p_sf.append("estimate(family=None) returned a number for a file holding two "
                    "split families instead of refusing")
    e_cv = sx.estimate(PRIMARY_COHORT, HEADLINE_CONDITION, "slice", family="cv")
    e_of = sx.estimate(PRIMARY_COHORT, HEADLINE_CONDITION, "slice", family="official")
    if abs(e_cv.point - CV_PHASE_SLICE[0]) > 1e-9:
        p_sf.append(f"cv-family estimate is {e_cv.point:.4f}, expected "
                    f"{CV_PHASE_SLICE[0]} (the across-seed row that mixes families "
                    f"must not be used)")
    if abs(e_of.point - 1.0) > 1e-9 or e_of.n_clusters != 4:
        p_sf.append(f"official-family estimate is {e_of.point:.4f} on "
                    f"{e_of.n_clusters} clusters, expected 1.000 on 4 -- the two "
                    f"families must be readable separately, not merged")
    _report("split_family_never_blended", p_sf,
            "one leftover 4-subject row at AUC 1.000 leaves the 70-subject headline "
            "at 0.691; verdict identical in all three trees")
    _report("split_family_guard_not_sort_dependent",
            [] if _codes(sf_out["official"][0]) == _codes(sf_out["root"][0]) else
            ["the verdict still depends on whether the stale directory name sorts "
             "before or after 'cv'"],
            "'official' and '.' give the same answer")

    # ---- [CVM] a fold that died caps the criteria and is stated everywhere ---
    # phase loses fold 4; magnitude and both keep all five. stage 4 describes
    # exactly what is on disk (4 folds, 56 subjects for phase), so nothing else
    # flags it. Before: SUPPORTED on 56 of 70 subjects, an empty coverage banner,
    # and section 5a asserting "one prediction per subject over the whole cohort"
    # directly above a 4-fold row sitting beside a 5-fold one.
    PH_N, PH_CL = 4 * CV_SUBJ * CV_SLICES, 4 * CV_SUBJ

    def _bcvm(res: Path) -> None:
        _cv_runs(res, drop=(("phase", 4),))
        st = _cv_stats(per_cond_size={"phase": (PH_N, PH_CL, [0, 1, 2, 3])})
        (res / "statistics.json").write_text(json.dumps(_s04_order(st)))
        _cv_controls(res.parent, n=PH_N, ncl=PH_CL)
    ctx_cvm, md_cvm, vj_cvm = _run_case("CVM_missing_fold", _bcvm)
    v_cvm, c_cvm = _codes(ctx_cvm)
    p_cvm: List[str] = []
    if v_cvm == "SUPPORTED":
        p_cvm.append("SUPPORTED reached on a cross-validation missing a fold")
    for code in ("C1", "C2"):
        if c_cvm.get(code) != "missing":
            p_cvm.append(f"{code} is {c_cvm.get(code)}, expected missing -- it was "
                         f"decided on 56 of 70 subjects")
    pcvm = ctx_cvm["pooled"][(PRIMARY_COHORT, HEADLINE_CONDITION)]
    if pcvm.coverage_complete or pcvm.missing_folds != [4]:
        p_cvm.append(f"coverage_complete={pcvm.coverage_complete}, "
                     f"missing_folds={pcvm.missing_folds}, expected False / [4]")
    if "INCOMPLETE" not in pcvm.scheme_label or "every subject is tested exactly once" \
            in pcvm.scheme_label:
        p_cvm.append(f"the scheme label still claims full coverage: "
                     f"{pcvm.scheme_label!r}")
    if not any("does NOT cover the cross-validation design" in d
               for d in ctx_cvm["degraded"]):
        p_cvm.append("the missing fold is not in the degraded banner")
    if "INCOMPLETE" not in md_cvm:
        p_cvm.append("RESULTS.md never says the cross-validation is incomplete")
    cvblk = (vj_cvm.get("cross_validation") or {}).get(PRIMARY_COHORT) or {}
    if cvblk.get("coverage_complete") is not False or cvblk.get("missing_folds") != [4]:
        p_cvm.append(f"verdict.json records coverage_complete="
                     f"{cvblk.get('coverage_complete')}, "
                     f"missing_folds={cvblk.get('missing_folds')}")
    _report("cv_missing_fold_caps_criteria", p_cvm,
            "phase pooled over 4 of 5 folds -> C1/C2 MISSING, banner + 5a + "
            "verdict.json all say so")

    # ... and the same tree with every fold present must still reach SUPPORTED,
    # or the coverage guard is simply a blanket refusal rather than a check.
    def _bcvc(res: Path) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        _cv_controls(res.parent)
    ctx_cvc, md_cvc, _ = _run_case("CVC_complete_folds", _bcvc)
    v_cvc, c_cvc = _codes(ctx_cvc)
    p_cvc: List[str] = []
    if v_cvc != "SUPPORTED":
        p_cvc.append(f"a complete 5-fold sweep gives {v_cvc} "
                     f"({sorted(k for k, s in c_cvc.items() if s != 'pass')}); the "
                     f"coverage guard must not fire when coverage is complete")
    if "every subject is tested exactly once" not in md_cvc:
        p_cvc.append("RESULTS.md drops the 'tested exactly once' statement even when "
                     "it is true")
    _report("cv_complete_folds_not_penalised", p_cvc,
            "5 of 5 folds -> coverage complete, criteria decided normally")

    # ---- [CVD] the DECLARED design is authoritative, not what survived -------
    # Three folds on disk for every condition, and the cv<k>_split columns stage
    # 1/2 wrote declare five. The cross-condition fallback cannot see this: every
    # condition agrees, because the same two folds died for all of them. Only the
    # design table knows 28 of 70 subjects were never tested.
    D_N, D_CL = 3 * CV_SUBJ * CV_SLICES, 3 * CV_SUBJ

    def _bcvd(res: Path) -> None:
        _cv_runs(res, folds=3)
        st = _cv_stats(n=D_N, ncl=D_CL, folds=(0, 1, 2),
                       per_cond_size={c: (D_N, D_CL, [0, 1, 2])
                                      for c in CONDITION_ORDER})
        (res / "statistics.json").write_text(json.dumps(_s04_order(st)))
        _cv_controls(res.parent, n=D_N, ncl=D_CL)
        # The stage-2 cache index, with the cv<k>_split columns describing all
        # five folds -- written before any model ran.
        cache = res.parent / "cache"
        cache.mkdir(parents=True, exist_ok=True)
        cols = [f"cv{k}_split" for k in range(CV_K)]
        lines = ["idx,patient_id,subject_id,file,slice," + ",".join(cols)]
        for f in range(CV_K):
            for s in range(CV_SUBJ):
                subj = f"{PRIMARY_COHORT[:4].upper()}{CV_SUBJ * f + s:03d}"
                for sl in range(CV_SLICES):
                    idx = 1000 * f + s * CV_SLICES + sl
                    splits = ",".join("test" if k == f else "training"
                                      for k in range(CV_K))
                    lines.append(f"{idx},{subj},{subj},/data/{subj}.h5,{sl},{splits}")
        (cache / f"{PRIMARY_COHORT}_index.csv").write_text("\n".join(lines) + "\n")
    ctx_cvd, md_cvd, vj_cvd = _run_case("CVD_design_table", _bcvd)
    v_cvd, c_cvd = _codes(ctx_cvd)
    p_cvd: List[str] = []
    exp = ctx_cvd["cv_expectation"].get(PRIMARY_COHORT)
    if exp is None or exp.folds != list(range(CV_K)):
        p_cvd.append(f"the declared fold set was not read from the cache index "
                     f"({exp.folds if exp else None}); the guard fell back to the "
                     f"folds that happened to survive")
    pcvd = ctx_cvd["pooled"].get((PRIMARY_COHORT, HEADLINE_CONDITION))
    if pcvd is None or pcvd.missing_folds != [3, 4]:
        p_cvd.append(f"missing folds recorded as "
                     f"{pcvd.missing_folds if pcvd else None}, expected [3, 4]")
    if pcvd is not None and pcvd.expected_subjects != CV_CL:
        p_cvd.append(f"expected subject count {pcvd.expected_subjects}, "
                     f"expected {CV_CL}")
    if v_cvd == "SUPPORTED":
        p_cvd.append("SUPPORTED reached with 28 of 70 subjects never tested")
    for code in ("C1", "C2"):
        if c_cvd.get(code) != "missing":
            p_cvd.append(f"{code} is {c_cvd.get(code)}, expected missing")
    _report("cv_design_table_is_authoritative", p_cvd,
            "cv<k>_split declares 5 folds, 3 are on disk -> caught even though every "
            "condition agrees")

    # ---- [FL] figures label every curve/bar with ITS OWN sample size ---------
    # Reusing the CVM tree, where phase covers 4 folds / 56 subjects and the other
    # two cover 5 / 70. The figures used to carry ONE size -- the first usable
    # condition's -- for the whole plot.
    fl = ctx_cvm.get("fig_labels") or {}
    p_fl: List[str] = []
    roc_note = (fl.get("roc") or [""])[0]
    if "DIFFERS BY CONDITION" not in roc_note:
        p_fl.append(f"the ROC annotation does not say the curves are on different "
                    f"test sets: {roc_note!r}")
    for cond, want in (("phase", PH_CL), ("magnitude", CV_CL), ("both", CV_CL)):
        if f"{want} test subjects" not in roc_note:
            p_fl.append(f"the ROC annotation never states {cond}'s own "
                        f"{want}-subject count")
    bars = " || ".join(fl.get("auc_bars") or [])
    if "SAMPLE SIZE DIFFERS BY CONDITION" not in bars:
        p_fl.append(f"the AUC-bar panel title asserts one sample size for bars that "
                    f"do not share one: {bars!r}")
    cvf = " || ".join(fl.get("cv_folds") or [])
    if "DIFFERENT fold sets" not in cvf:
        p_fl.append(f"the per-fold panel title does not say the conditions cover "
                    f"different fold sets: {cvf!r}")
    if "each tested exactly once" in cvf:
        p_fl.append("the per-fold panel still claims every subject was tested once")
    # ... and on a tree where they DO agree, the shared one-liner comes back: a
    # label that always hedges carries no information either.
    fl_ok = ctx_cvc.get("fig_labels") or {}
    if "DIFFERS BY CONDITION" in (fl_ok.get("roc") or [""])[0]:
        p_fl.append("the ROC annotation claims a discrepancy on a tree where every "
                    "condition covers the same folds")
    if not any("each tested exactly once" in t for t in (fl_ok.get("cv_folds") or [])):
        p_fl.append("the per-fold panel drops the 'tested exactly once' statement "
                    "even when it is true")
    _report("figures_label_each_condition", p_fl,
            "ROC/bars/fold panels state per-condition sizes when they differ and the "
            "shared one when they do not")

    # ---- [CU] the cluster unit comes from the column stage 2 writes ----------
    # The cache index carries subject_id; the stage-1 cohort CSV has one basename
    # that does not join. s06 used to demand the join and drop the WHOLE cohort to
    # patient_id on a single miss -- which on breast does not collapse the
    # repeat-scan groups, so two scans of one woman were bootstrapped as two
    # independent patients and every interval came out too narrow.
    cu = root / "CU_cluster_unit"
    (cu / "cache").mkdir(parents=True, exist_ok=True)
    (cu / "cohorts").mkdir(parents=True, exist_ok=True)
    rows = ["idx,patient_id,subject_id,file,slice"] + [
        f"{i},P{i // 2},SUBJ{i // 4},/data/scan{i}.h5,{i}" for i in range(8)]
    (cu / "cache" / f"{PRIMARY_COHORT}_index.csv").write_text("\n".join(rows) + "\n")
    coh = ["subject_id,file"] + [
        f"SUBJ{i // 4}," + ("/data/scan3_RENAMED.h5" if i == 3 else f"/data/scan{i}.h5")
        for i in range(8)]
    (cu / "cohorts" / f"{PRIMARY_COHORT}_cohort.csv").write_text("\n".join(coh) + "\n")
    cmap, unit = build_cluster_map(PRIMARY_COHORT, cu / "cache", cu / "cohorts")
    p_cu: List[str] = []
    if unit != "subject_id":
        p_cu.append(f"cluster unit is {unit!r} although the cache index has a "
                    f"subject_id column")
    if sorted(set(cmap.values())) != ["SUBJ0", "SUBJ1"]:
        p_cu.append(f"4 patient_ids did not collapse to 2 subject_ids: "
                    f"{sorted(set(cmap.values()))}")
    # The old join path must still work when the index has no subject_id ...
    (cu / "cache2").mkdir(parents=True, exist_ok=True)
    (cu / "cache2" / f"{PRIMARY_COHORT}_index.csv").write_text(
        "\n".join(["idx,patient_id,file,slice"]
                  + [f"{i},P{i // 2},/data/scan{i}.h5,{i}" for i in range(8)]) + "\n")
    (cu / "cohorts2").mkdir(parents=True, exist_ok=True)
    (cu / "cohorts2" / f"{PRIMARY_COHORT}_cohort.csv").write_text(
        "\n".join(["subject_id,file"]
                  + [f"SUBJ{i // 4},/data/scan{i}.h5" for i in range(8)]) + "\n")
    cmap2, unit2 = build_cluster_map(PRIMARY_COHORT, cu / "cache2", cu / "cohorts2")
    if unit2 != "subject_id" or sorted(set(cmap2.values())) != ["SUBJ0", "SUBJ1"]:
        p_cu.append(f"the cohort-CSV join stopped working: {unit2!r} "
                    f"{sorted(set(cmap2.values()))}")
    # ... and with neither, the fallback still names itself.
    cmap3, unit3 = build_cluster_map(PRIMARY_COHORT, cu / "cache2", cu / "cohorts")
    if "patient_id" not in unit3:
        p_cu.append(f"the fallback does not name itself: {unit3!r}")
    _report("cluster_unit_from_cache_index", p_cu,
            "subject_id read from {cohort}_index.csv; one unmatched basename no "
            "longer costs the whole cohort its clustering unit")

    # ======================================================================
    # Fourth adversarial round: R1, R2b, R3, R4, R5.
    #
    # Every one of these is a tree that printed SUPPORTED, and every one of them
    # is the same species of bug: a number was compared against another number
    # without establishing that the two describe the same experiment. The
    # baseline they all share is the honest five-fold tree used above -- headline
    # 0.6909 over 70 subjects, destroy-controls 0.061 below it -- which correctly
    # gives NOT SUPPORTED with C4 and C7 failing.
    # ======================================================================

    # ---- [R1] the point estimate may not be overwritten after run selection --
    # `Stats.estimate` scopes run selection to one split family and then took
    # s04's across-seeds mean when the across row `family is None or not
    # acr_fams` -- i.e. whenever the row did not SAY what it covered. An s04 that
    # simply does not write `split_families` therefore re-opened the whole defect
    # the family scoping exists to close: with one leftover 4-subject file at AUC
    # 1.000 on disk, the 70-subject headline went 0.691 -> 0.794, C4 and C7
    # flipped FAIL -> PASS, and RESULTS.md printed "headline 0.794 [0.600,
    # 0.780]" -- a point estimate outside its own interval -- in the C4 evidence
    # while C8 quoted 0.691 for the same cohort in the same file.
    def _br1(res: Path) -> None:
        _cv_runs(res)
        st = _add_leftover(_cv_stats(), res, "official", declare=False)
        (res / "statistics.json").write_text(json.dumps(_s04_order(st)))
        _cv_controls(res.parent, bg=0.6299, scramble=0.6299)
    ctx_r1, md_r1, _ = _run_case("R1_across_row_undeclared", _br1)
    v_r1, c_r1 = _codes(ctx_r1)
    p_r1: List[str] = []
    sx1 = Stats(json.loads((root / "R1_across_row_undeclared" / "results"
                            / "statistics.json").read_text()), None)
    e_r1 = sx1.estimate(PRIMARY_COHORT, HEADLINE_CONDITION, "slice", family=CV_FAMILY)
    if abs(e_r1.point - CV_PHASE_SLICE[0]) > 1e-9:
        p_r1.append(f"the cv-family estimate is {e_r1.point:.4f}, expected "
                    f"{CV_PHASE_SLICE[0]}: an across-seeds row that does not declare "
                    f"its split families still overwrote the family-scoped point")
    if v_r1 == "SUPPORTED":
        p_r1.append("SUPPORTED reached on a headline blended with a 4-subject file")
    for code in ("C4", "C7"):
        if c_r1.get(code) != "fail":
            p_r1.append(f"{code} is {c_r1.get(code)}, expected fail on the honest "
                        f"0.691 headline")
    # ... and the two invariants the same report handed us. Every criterion that
    # quotes the slice-level headline must quote ONE number, and that number must
    # lie inside its own interval.
    cv_r1 = next(v for v in ctx_r1["verdicts"] if v.cohort == PRIMARY_COHORT)
    for key, (pt, lo, hi) in cv_r1.headlines.items():
        if not (lo <= pt <= hi):
            p_r1.append(f"{key} is {pt} with interval [{lo}, {hi}]")
    quoted_pts = {round(c.headline_point, 6) for c in cv_r1.criteria
                  if c.headline_key.endswith("@slice")}
    if len(quoted_pts) > 1:
        p_r1.append(f"criteria quote {len(quoted_pts)} different slice-level "
                    f"headlines for one cohort: {sorted(quoted_pts)}")
    for m in re.finditer(r"headline ([0-9.]+) \[([0-9.]+), ([0-9.]+)\]", md_r1):
        pt, lo, hi = (float(x) for x in m.groups())
        if not (lo <= pt <= hi):
            p_r1.append(f"RESULTS.md prints {pt} outside its own interval "
                        f"[{lo}, {hi}]")
            break
    _report("headline_not_overwritten_after_scoping", p_r1,
            "an across-seeds row that declares no split family cannot overwrite the "
            "family-scoped point; headline stays 0.691 and every criterion quotes it")

    # ---- [R2b] a guard defeated by DELETING information ---------------------
    # C8's fold-match test read `if a is not None and b is not None and a != b`,
    # so the SAME permutation null on the SAME wrong fold was refused when its
    # payload declared `n_clusters` and accepted when the key was absent. Both
    # trees below hold a null computed on 20 of the 70 subjects; only the
    # declaration differs. C8 must be MISSING in both.
    def _br2(res: Path, declare: bool) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        payloads = _synth_control_payloads(
            PRIMARY_COHORT, 0.50, 0.55, 0.55, 0.74, 0.52, n_permutations=20,
            n_clusters=CV_CL, ci_n=CV_N, ci_n_pos=35)
        for pl in payloads:
            if pl["control"] != "label_permutation":
                continue
            ci = pl["control_detail"]["test_auc_ci95"]
            if declare:
                ci.update(n=160, n_clusters=20)      # honest about the wrong fold
            else:
                ci.pop("n", None)                    # same run, provenance deleted
                ci.pop("n_clusters", None)
        _write_control_payloads(res.parent / "controls" / "results", payloads)
    p_r2: List[str] = []
    r2_states = {}
    for tag, declare in (("declared", True), ("silent", False)):
        ctx_r2, _, _ = _run_case(f"R2b_perm_null_{tag}",
                                 lambda r, _d=declare: _br2(r, _d))
        v_r2, c_r2 = _codes(ctx_r2)
        r2_states[tag] = c_r2.get("C8")
        if c_r2.get("C8") != "missing":
            p_r2.append(f"{tag}: C8 is {c_r2.get('C8')}, expected missing -- the null "
                        f"was computed on 20 of the 70 subjects the headline covers")
        if v_r2 == "SUPPORTED":
            p_r2.append(f"{tag}: SUPPORTED reached against a null from another fold")
    if r2_states.get("declared") != r2_states.get("silent"):
        p_r2.append(f"deleting the size keys changed the answer: declared -> "
                    f"{r2_states.get('declared')}, silent -> {r2_states.get('silent')}")
    # ... and where the null IS on the headline's fold, C8 must still be decided,
    # or the guard is a blanket refusal rather than a check (CVC_complete_folds
    # above is that tree; assert it here too, next to the refusal it pairs with).
    if _codes(ctx_cvc)[1].get("C8") != "pass":
        p_r2.append(f"C8 is {_codes(ctx_cvc)[1].get('C8')} on a tree where the null "
                    f"and the headline share a fold; the check refuses everything")
    # The same shape in the replicate counter: `pseed is not None and pseed in
    # seen` compared permutation seeds only where the payload carried one, so 20
    # files declaring no `permutation_seed` counted as 20 draws from the null on
    # the strength of their predictions being byte-different -- which two
    # training seeds of ONE permutation also are. Here the fold matches, so
    # nothing else can be what makes C8 unevaluable.
    def _br2c(res: Path) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        payloads = _synth_control_payloads(
            PRIMARY_COHORT, 0.50, 0.55, 0.55, 0.74, 0.52, n_permutations=20,
            n_clusters=CV_CL, ci_n=CV_N, ci_n_pos=35)
        for pl in payloads:
            if pl["control"] == "label_permutation":
                pl["control_detail"].pop("permutation_seed", None)
        _write_control_payloads(res.parent / "controls" / "results", payloads)
    ctx_r2c, _, _ = _run_case("R2b_perm_no_seed", _br2c)
    v_r2c, c_r2c = _codes(ctx_r2c)
    for code in ("C3", "C8"):
        if c_r2c.get(code) != "missing":
            p_r2.append(f"unidentified replicates: {code} is {c_r2c.get(code)}, "
                        f"expected missing -- 20 files that declare no "
                        f"`permutation_seed` are not 20 draws from the null")
    if v_r2c == "SUPPORTED":
        p_r2.append("SUPPORTED reached on a null of unidentifiable replicates")
    _report("missing_size_fails_closed", p_r2,
            "a null on another fold is refused whether or not it declares its size, "
            "replicates that cannot identify themselves are not counted, and a null "
            "on the headline's own fold is still evaluated")

    # ---- [R3] C4/C5/C6/C7 must apply the check C8 already applies ------------
    # These four subtracted a control AUC from the headline AUC with no check
    # that the two were scored on the same test set, so one RESULTS.md could
    # refuse the comparison as meaningless for C8 and pass the other four on
    # exactly it. Live on the real tree: the only prostate_t2 background control
    # sits in controls/results/prostate_t2_cv0, scored on that fold's subjects,
    # and was differenced against the 67-subject pooled out-of-fold headline.
    def _br3(res: Path) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        _cv_controls(res.parent, n=CV_SUBJ * CV_SLICES, ncl=CV_SUBJ)   # ONE fold
    ctx_r3, _, _ = _run_case("R3_control_on_another_fold", _br3)
    v_r3, c_r3 = _codes(ctx_r3)
    p_r3: List[str] = []
    for code in ("C4", "C5", "C6", "C7"):
        if c_r3.get(code) != "missing":
            p_r3.append(f"{code} is {c_r3.get(code)}, expected missing -- its control "
                        f"was scored on {CV_SUBJ} of the {CV_CL} subjects the headline "
                        f"covers, which is the comparison C8 refuses")
    if c_r3.get("C8") != "missing":
        p_r3.append(f"C8 is {c_r3.get('C8')} on the same mismatch it has always "
                    f"refused")
    if v_r3 == "SUPPORTED":
        p_r3.append("SUPPORTED reached on controls from a different test set")
    # The same tree with the controls on the headline's own fold must still
    # decide them: a guard that refuses unconditionally is not a check. CVC is
    # that tree.
    for code in ("C4", "C5", "C6", "C7"):
        if _codes(ctx_cvc)[1].get(code) != "pass":
            p_r3.append(f"{code} is {_codes(ctx_cvc)[1].get(code)} on a tree where "
                        f"the controls and the headline share a test set")
    # ... and a control that is demonstrably ALIVE still FAILS on the wrong
    # fold. PASS -> MISSING only; a wrong provenance never upgrades a criterion.
    def _br3b(res: Path) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        _cv_controls(res.parent, bg=0.66, scramble=0.66,
                     n=CV_SUBJ * CV_SLICES, ncl=CV_SUBJ)
    ctx_r3b, _, _ = _run_case("R3_alive_control_still_fails", _br3b)
    v_r3b, c_r3b = _codes(ctx_r3b)
    for code in ("C4", "C7"):
        if c_r3b.get(code) != "fail":
            p_r3b_msg = (f"{code} is {c_r3b.get(code)}, expected fail: a control "
                         f"0.031 below the headline is still too close whatever fold "
                         f"it ran on, and MISSING would be an upgrade")
            p_r3.append(p_r3b_msg)
    _report("control_headline_share_a_test_set", p_r3,
            "C4/C5/C6/C7 refuse a control from another fold exactly as C8 does, still "
            "pass when it is the same fold, and still FAIL a control that survived")

    # ---- [R4] the OTHER side of the comparison ------------------------------
    # `comparison_levels` scoped records by split family and never read the
    # `folds_a` / `folds_b` they carry, and the stage-4 size guard ran on
    # HEADLINE_CONDITION only -- so nothing at all checked the reference
    # condition. Below: five folds on disk for every condition, and stage 4
    # describing magnitude over four. C2 was a Holm-adjusted p-value between
    # phase on 70 subjects and magnitude on 56.
    R4_N, R4_CL = 4 * CV_SUBJ * CV_SLICES, 4 * CV_SUBJ

    def _br4(res: Path) -> None:
        _cv_runs(res)
        st = _cv_stats(per_cond_size={REFERENCE_CONDITION: (R4_N, R4_CL, [0, 1, 2, 3])})
        (res / "statistics.json").write_text(json.dumps(_s04_order(st)))
        _cv_controls(res.parent)
    ctx_r4, md_r4, _ = _run_case("R4_reference_coverage", _br4)
    v_r4, c_r4 = _codes(ctx_r4)
    p_r4: List[str] = []
    if c_r4.get("C2") != "missing":
        p_r4.append(f"C2 is {c_r4.get('C2')}, expected missing -- stage 4 compared "
                    f"phase over 5 folds against magnitude over 4")
    if v_r4 == "SUPPORTED":
        p_r4.append("SUPPORTED reached on a comparison between two test sets")
    if not any("not between two numbers on one test set" in d
               for d in ctx_r4["degraded"]):
        p_r4.append("the reference condition's coverage is not in the degraded banner")

    # ... and the same tree with the fold declarations DELETED. Missing
    # provenance fails closed here too: a comparison record that does not say
    # which folds it covers cannot demonstrate it covers these.
    def _br4b(res: Path) -> None:
        _cv_runs(res)
        st = _s04_order(_cv_stats())
        for c_ in st["comparisons"]:
            c_.pop("folds_a", None)
            c_.pop("folds_b", None)
        (res / "statistics.json").write_text(json.dumps(st))
        _cv_controls(res.parent)
    ctx_r4b, _, _ = _run_case("R4_comparison_declares_no_folds", _br4b)
    v_r4b, c_r4b = _codes(ctx_r4b)
    if c_r4b.get("C2") != "missing":
        p_r4.append(f"C2 is {c_r4b.get('C2')} on comparison records that do not say "
                    f"which folds they cover; an undeclared scope is not a matching "
                    f"scope")
    if v_r4b == "SUPPORTED":
        p_r4.append("SUPPORTED reached on comparison records with no fold provenance")
    # ... and where they DO declare the headlined folds, C2 is decided normally.
    if _codes(ctx_cvc)[1].get("C2") != "pass":
        p_r4.append(f"C2 is {_codes(ctx_cvc)[1].get('C2')} on a tree whose comparison "
                    f"records declare exactly the folds this report headlines")
    lv_r4 = Stats(json.loads((root / "R4_reference_coverage" / "results"
                              / "statistics.json").read_text()), None) \
        .comparison_levels(PRIMARY_COHORT, HEADLINE_CONDITION, REFERENCE_CONDITION,
                           family=CV_FAMILY)
    pm_r4 = lv_r4.get("patient_mean") or {}
    if pm_r4.get("folds_a") != [tuple(range(CV_K))] or \
            pm_r4.get("folds_b") != [(0, 1, 2, 3)]:
        p_r4.append(f"comparison_levels does not surface the per-side fold sets: "
                    f"{pm_r4.get('folds_a')} vs {pm_r4.get('folds_b')}")
    _report("comparison_covers_one_test_set", p_r4,
            "phase over 5 folds vs magnitude over 4 caps C2 at MISSING, as does a "
            "comparison record that declares no folds at all")

    # ---- [R5] two cross-validation sweeps are two experiments ---------------
    # `_split_family` collapsed every fold-tagged run into `cv` regardless of the
    # directory it lived in, and the pooler grouped folds by index alone, so
    # `sweepA/<cohort>_cv0` and `sweepB/<cohort>_cv0` landed in one bucket and
    # pool_runs AVERAGED their probability vectors (they share cache rows).
    # Measured: an honest sweep at AUC 0.63 blended with an optimistic one at
    # 0.995 gave one "pooled out-of-fold" headline of 0.938 over the same 70
    # subjects, and the destroy-controls that FAIL against 0.63 PASS against it.
    def _br5(res: Path, subdirs) -> None:
        for sub, auc in subdirs:
            for f in range(CV_K):
                d = res / sub / f"{PRIMARY_COHORT}_cv{f}"
                d.mkdir(parents=True, exist_ok=True)
                for cond, target in (("magnitude", 0.35), ("phase", 0.55),
                                     ("both", 0.58)):
                    for sd in (42, 123):
                        (d / f"{PRIMARY_COHORT}_{cond}_seed{sd}.json").write_text(
                            json.dumps(_synth_run(
                                PRIMARY_COHORT, cond, sd, target, n_subj=CV_SUBJ,
                                n_pos_subj=7, slices=CV_SLICES, pos_rate=1.0,
                                subj_offset=CV_SUBJ * f, idx_offset=1000 * f,
                                auc_exact=auc)))
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        _cv_controls(res.parent, bg=0.6299, scramble=0.6299)
    ctx_r5, _, _ = _run_case("R5_two_cv_sweeps",
                             lambda r: _br5(r, (("sweepA", 0.63), ("sweepB", 0.995))))
    v_r5, c_r5 = _codes(ctx_r5)
    p_r5: List[str] = []
    fams_r5 = sorted({r["_split_family"] for r in
                      load_runs(root / "R5_two_cv_sweeps" / "results")})
    if fams_r5 != ["cv@sweepA", "cv@sweepB"]:
        p_r5.append(f"the two sweeps were assigned split families {fams_r5}; folds in "
                    f"different subdirectories are different experiments")
    if (PRIMARY_COHORT, HEADLINE_CONDITION) in ctx_r5["pooled"]:
        pr5 = ctx_r5["pooled"][(PRIMARY_COHORT, HEADLINE_CONDITION)]
        p_r5.append(f"a headline was pooled from both sweeps anyway ({pr5.n_runs} runs, "
                    f"AUC {_safe_auc(pr5.labels, pr5.probs):.3f} over "
                    f"{pr5.n_clusters} subjects)")
    if v_r5 == "SUPPORTED":
        p_r5.append("SUPPORTED reached on a headline blended from two sweeps")
    for code in ("C4", "C7"):
        if c_r5.get(code) == "pass":
            p_r5.append(f"{code} PASSES against a headline the report declined to build")
    if not any("different results subdirectories" in d for d in ctx_r5["degraded"]):
        p_r5.append("the second sweep is not named in the degraded banner")
    # ... and the ordinary one-sweep layout is untouched: folds directly under
    # the results root are still the plain `cv` family stage 4 writes, and that
    # tree still reaches SUPPORTED (CVC_complete_folds, above).
    fams_ok = sorted({r["_split_family"] for r in
                      load_runs(root / "CVC_complete_folds" / "results")})
    if fams_ok != [CV_FAMILY]:
        p_r5.append(f"the ordinary layout no longer yields the plain {CV_FAMILY!r} "
                    f"family: {fams_ok}")
    if _codes(ctx_cvc)[0] != "SUPPORTED":
        p_r5.append("the single-sweep tree stopped reaching SUPPORTED")
    _report("cv_sweeps_never_merged", p_r5,
            "folds in two subdirectories are two experiments: no blended headline, "
            "banner says so, and the ordinary root layout is unchanged")

    # ======================================================================
    # Fifth adversarial round: AG1..AG5 -- aggregate provenance.
    #
    # Same species again, one level below the fourth round's guards. Those
    # guards ask an object for its test-set fingerprint; every object they ask
    # is a POOL, and each pool used to synthesise its fingerprint out of
    # whichever member happened to have one -- `good[0].n`, `min(clusters)`,
    # `_control_estimate(runs[0])`, `_first(group[0], "n_clusters")`. Every tree
    # below printed SUPPORTED on the honest five-fold baseline (headline 0.6909
    # over 70 subjects / 560 slices, destroy-controls at 0.6299 so that the
    # tree WITHOUT the attack is NOT SUPPORTED with C4 and C7 failing).
    # ======================================================================

    def _cv_control_payloads(bg=0.55, scramble=0.55, n=CV_N, ncl=CV_CL) -> List[dict]:
        return _synth_control_payloads(PRIMARY_COHORT, 0.50, bg, scramble, 0.74, 0.52,
                                       n_permutations=20, n_clusters=ncl,
                                       ci_n=n, ci_n_pos=35)

    def _restated(base: dict, auc: float, seed: int, n=None, ncl=None,
                  half: float = 0.13, silent: bool = False) -> dict:
        """A copy of one control run, re-scored and re-declaring its test set."""
        p = json.loads(json.dumps(base))
        p["seed"] = seed
        p["test"] = dict(p["test"], auc=auc)
        p["best_selection_score"] = auc
        ci = dict(p["control_detail"]["test_auc_ci95"])
        ci.update({"auc": auc, "lo": max(0.0, auc - half), "hi": min(1.0, auc + half),
                   "n_boot_ok": 1180})
        if silent:                      # declares no size at all
            ci.pop("n", None)
            ci.pop("n_clusters", None)
        else:
            ci.update({"n": n, "n_clusters": ncl, "n_pos": 7})
        p["control_detail"] = dict(p["control_detail"], test_auc_ci95=ci)
        return p

    def _bag(res: Path, second: Optional[dict]) -> None:
        """The honest tree; `second` = kwargs for an extra destroy-control run."""
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        pls = _cv_control_payloads(bg=0.6299, scramble=0.6299)
        if second is not None:
            for name in ("background_only", "phase_scramble"):
                src = next(p for p in pls if p["control"] == name)
                pls.append(_restated(src, seed=43, **second))
        _write_control_payloads(res.parent / "controls" / "results", pls)

    # ---- [AG1/AG2] an envelope may not borrow one member's test set ---------
    # `n = good[0].n, n_pos = good[0].n_pos` took the FIRST member's slice count
    # and `n_clusters = min(clusters)` the smallest, so the hull over an honest
    # 560-slice / 70-subject control at 0.6299 and a second run at 0.5300 scored
    # on 40 slices of 200 other subjects reported "560 slices / 70 subjects".
    # `_test_set_mismatch` confirmed it, and C4/C7 -- which FAIL at 0.6299 --
    # passed on the blended 0.580. AG2 is the version `min()` cannot see at all:
    # the same 70 subjects, a different 200-slice selection.
    ag_specs = [
        ("AG1_envelope_borrows_a_member",
         {"auc": 0.5300, "n": 40, "ncl": 200},
         "a second control run on 40 slices / 200 other subjects"),
        ("AG2_envelope_same_subjects_other_slices",
         {"auc": 0.5300, "n": 200, "ncl": CV_CL},
         "a second control run on 200 of the 560 slices"),
        ("AG2b_envelope_member_declares_nothing",
         {"auc": 0.5300, "silent": True},
         "a second control run that declares no size at all"),
    ]
    p_ag1: List[str] = []
    for name, spec, blurb in ag_specs:
        ctx_ag, _, _ = _run_case(name, lambda r, _s=spec: _bag(r, _s))
        v_ag, c_ag = _codes(ctx_ag)
        for code in ("C4", "C7"):
            if c_ag.get(code) != "missing":
                p_ag1.append(f"{name}: {code} is {c_ag.get(code)}, expected missing -- "
                             f"the envelope pools {blurb} and cannot report the "
                             f"headline's test set as its own")
        if v_ag == "SUPPORTED":
            p_ag1.append(f"{name}: SUPPORTED reached on an envelope whose members were "
                         f"scored on different test sets")
    # ... and the same second run scored on the SAME test set must still be
    # decided, or the rule is a blanket refusal rather than a consensus.
    ctx_agok, _, _ = _run_case(
        "AG1_negative_control",
        lambda r: _bag(r, {"auc": 0.5300, "n": CV_N, "ncl": CV_CL}))
    v_agok, c_agok = _codes(ctx_agok)
    for code in ("C4", "C7"):
        if c_agok.get(code) != "pass":
            p_ag1.append(f"AG1_negative_control: {code} is {c_agok.get(code)}, expected "
                         f"pass -- both runs declare 560 slices / 70 subjects, so the "
                         f"envelope has a consensus and the difference is decidable")
    # ... and the baseline the attack is measured against still FAILS, or the
    # scenario has stopped testing what it was built to test.
    ctx_agbase, _, _ = _run_case("AG_baseline_alive", lambda r: _bag(r, None))
    v_agbase, c_agbase = _codes(ctx_agbase)
    for code in ("C4", "C7"):
        if c_agbase.get(code) != "fail":
            p_ag1.append(f"AG_baseline_alive: {code} is {c_agbase.get(code)}, expected "
                         f"fail -- a control 0.061 below the headline is still alive")
    if v_agbase != "NOT SUPPORTED":
        p_ag1.append(f"AG_baseline_alive: {v_agbase}, expected NOT SUPPORTED")
    _report("envelope_fingerprint_is_consensus", p_ag1,
            "an envelope over runs on different test sets carries no fingerprint "
            "(C4/C7 MISSING); on one test set it still decides them")

    # ---- [AG3] a null pool may not borrow one replicate's test set ----------
    # `base = _control_estimate(runs[0])` gave the pool whatever the first file
    # on disk declared. Nineteen replicates on a 4-subject fold plus ONE on the
    # headline's own 70-subject out-of-fold set reported 560 slices / 70
    # subjects, C8's fold check was satisfied by it, and p = (0+1)/(20+1) = 0.048
    # passed on a null measured almost entirely somewhere else.
    AG3_NULLS = [0.35, 0.38, 0.41, 0.44, 0.47, 0.50, 0.53, 0.55, 0.42, 0.46,
                 0.36, 0.39, 0.43, 0.45, 0.48, 0.51, 0.54, 0.40, 0.44, 0.49]

    def _bag3(res: Path, on_headline_fold) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        pls = [p for p in _cv_control_payloads()
               if p["control"] != "label_permutation"]
        tmpl = next(p for p in _cv_control_payloads()
                    if p["control"] == "label_permutation")
        for i, a in enumerate(AG3_NULLS):
            p = json.loads(json.dumps(tmpl))
            p["seed"] = 1000 + i
            p["control_detail"] = dict(p["control_detail"], variant=f"perm{i:02d}",
                                       permutation_seed=1000 + i)
            p["test"] = dict(p["test"], auc=a,
                             probs=[float((x + i) % 97) / 97.0 for x in range(40)])
            ci = dict(p["control_detail"]["test_auc_ci95"])
            ci.update({"auc": a, "lo": max(0.0, a - 0.13), "hi": min(1.0, a + 0.13)})
            if i not in on_headline_fold:
                ci.update({"n": 40, "n_clusters": 4})       # a different fold
            p["control_detail"] = dict(p["control_detail"], test_auc_ci95=ci)
            pls.append(p)
        _write_control_payloads(res.parent / "controls" / "results", pls)

    p_ag3: List[str] = []
    for tag, on_fold, want_c3 in (("all_on_another_fold", set(), "pass"),
                                  ("one_on_the_headlines_fold", {0}, "missing")):
        ctx_a3, _, _ = _run_case(f"AG3_null_pool_{tag}",
                                 lambda r, _o=on_fold: _bag3(r, _o))
        v_a3, c_a3 = _codes(ctx_a3)
        if c_a3.get("C8") != "missing":
            p_ag3.append(f"{tag}: C8 is {c_a3.get('C8')}, expected missing -- the null "
                         f"was not measured on the test set it is differenced against")
        if c_a3.get("C3") != want_c3:
            p_ag3.append(f"{tag}: C3 is {c_a3.get('C3')}, expected {want_c3}")
        if v_a3 == "SUPPORTED":
            p_ag3.append(f"{tag}: SUPPORTED reached on a null pooled over two test sets")
    # ... and a null whose replicates DO agree is still evaluated (CVC).
    if _codes(ctx_cvc)[1].get("C8") != "pass" or _codes(ctx_cvc)[1].get("C3") != "pass":
        p_ag3.append(f"C3/C8 are {_codes(ctx_cvc)[1].get('C3')}/"
                     f"{_codes(ctx_cvc)[1].get('C8')} on a tree whose replicates all "
                     f"share the headline's fold; the check refuses everything")
    _report("null_pool_fingerprint_is_consensus", p_ag3,
            "19 replicates on a 4-subject fold and 1 on the headline's give the pool "
            "no fingerprint (C3/C8 MISSING); 20 that agree still decide it")

    # ---- [AG4] a comparison group has TWO sides, and both must be checkable --
    # The fold check ran only when BOTH conditions had a pooled prediction
    # vector, so a report holding no magnitude predictions at all -- none on
    # disk, or two sweeps the pooler refused to merge -- skipped it entirely and
    # C2 was decided on a stage-4 Holm p-value whose reference side this report
    # cannot identify.
    def _bag4_absent(res: Path) -> None:
        _cv_runs(res)
        for f in range(CV_K):
            for sd in (42, 123):
                (res / f"{PRIMARY_COHORT}_cv{f}"
                 / f"{PRIMARY_COHORT}_{REFERENCE_CONDITION}_seed{sd}.json").unlink()
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        _cv_controls(res.parent)

    def _bag4_unpoolable(res: Path) -> None:
        _cv_runs(res)
        for f in range(CV_K):
            src = res / f"{PRIMARY_COHORT}_cv{f}"
            alt = res / "sweepB" / f"{PRIMARY_COHORT}_cv{f}"
            alt.mkdir(parents=True, exist_ok=True)
            for sd in (42, 123):
                q = src / f"{PRIMARY_COHORT}_{REFERENCE_CONDITION}_seed{sd}.json"
                (alt / q.name).write_text(q.read_text())
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        _cv_controls(res.parent)

    p_ag4: List[str] = []
    for tag, build in (("reference_absent", _bag4_absent),
                       ("reference_unpoolable", _bag4_unpoolable)):
        ctx_a4, _, _ = _run_case(f"AG4_comparison_{tag}", build)
        v_a4, c_a4 = _codes(ctx_a4)
        if c_a4.get("C2") != "missing":
            p_ag4.append(f"{tag}: C2 is {c_a4.get('C2')}, expected missing -- stage 4's "
                         f"comparison names a reference condition this report has no "
                         f"predictions for, so its test set cannot be checked")
        if v_a4 == "SUPPORTED":
            p_ag4.append(f"{tag}: SUPPORTED reached on a comparison whose reference "
                         f"side the report cannot identify")
        if not any("no pooled predictions" in d for d in ctx_a4["degraded"]):
            p_ag4.append(f"{tag}: the unidentifiable reference side is not in the "
                         f"degraded banner")
    # ... and where both sides ARE pooled over the headlined folds, C2 decides.
    if _codes(ctx_cvc)[1].get("C2") != "pass":
        p_ag4.append(f"C2 is {_codes(ctx_cvc)[1].get('C2')} on a tree where both "
                     f"conditions are pooled over exactly the folds headlined")
    _report("comparison_group_has_two_checkable_sides", p_ag4,
            "a comparison whose reference condition has no predictions in this report "
            "caps C2 at MISSING; one where both sides are pooled still decides it")

    # ---- [AG5] the count of DISTINCT replicates is not a denylist question ---
    # `_replicate_fingerprint` hashes the whole payload minus a hand-written list
    # of volatile keys, so a field the list has never heard of counts as
    # substance. Twenty copies of ONE replicate were correctly collapsed to one
    # draw; the same twenty with a per-epoch wall clock varied counted as twenty,
    # and p = 1/21 = 0.048 passed C8 out of a single permutation.
    def _bag5(res: Path, vary_history: bool) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        pls = [p for p in _cv_control_payloads()
               if p["control"] != "label_permutation"]
        tmpl = next(p for p in _cv_control_payloads()
                    if p["control"] == "label_permutation")
        tmpl["test"] = dict(tmpl["test"], auc=0.5)
        tmpl["control_detail"]["test_auc_ci95"].update({"auc": 0.5, "lo": 0.37,
                                                        "hi": 0.63})
        for i in range(20):
            p = json.loads(json.dumps(tmpl))
            p["seed"] = 1000 + i
            p["control_detail"] = dict(p["control_detail"], variant=f"perm{i:02d}",
                                       permutation_seed=1000 + i)
            if vary_history:
                p["history"][0] = dict(p["history"][0], seconds=float(i))
            pls.append(p)
        _write_control_payloads(res.parent / "controls" / "results", pls)

    p_ag5: List[str] = []
    for tag, vary in (("pure_copies", False), ("one_incidental_field", True)):
        ctx_a5, _, _ = _run_case(f"AG5_near_duplicate_{tag}",
                                 lambda r, _v=vary: _bag5(r, _v))
        v_a5, c_a5 = _codes(ctx_a5)
        for code in ("C3", "C8"):
            if c_a5.get(code) != "missing":
                p_ag5.append(f"{tag}: {code} is {c_a5.get(code)}, expected missing -- "
                             f"20 files holding one set of predictions are one draw "
                             f"from the null, whatever else differs between them")
        if v_a5 == "SUPPORTED":
            p_ag5.append(f"{tag}: SUPPORTED reached on a null of one replicate")
    # ... and 20 genuinely different permutations are still 20 (CVC).
    if _codes(ctx_cvc)[1].get("C8") != "pass":
        p_ag5.append("C8 is no longer decided on a tree of 20 distinct replicates; the "
                     "prediction hash is collapsing runs that really do differ")
    _report("replicate_count_from_predictions", p_ag5,
            "one replicate under 20 filenames is one draw whether or not an incidental "
            "field differs; 20 real permutations still count as 20")

    # ======================================================================
    # SIXTH ROUND: a control that ran PER FOLD is one measurement, not five
    #
    # Every check below is a per-fold stage-5 controls tree -- the layout the
    # real pipeline writes, `controls/results/<cohort>_cv<k>/` -- differenced
    # against a pooled out-of-fold headline. All four defects they pin were live
    # on the real prostate_t2 tree, and all four are in the same family: an
    # aggregation over folds that is not the aggregation the headline got.
    # ======================================================================

    def _cv_control(cohort_: str, fold: int, control: str, auc: float, *,
                    variant: Optional[str] = None, seed: int = 42,
                    condition: str = HEADLINE_CONDITION, region: str = "full",
                    label_semantics: str = DIAGNOSTIC_SEMANTICS,
                    detail: Optional[dict] = None, with_ci: bool = True,
                    declare_split_col: bool = True) -> dict:
        """
        ONE stage-5 control payload for ONE cross-validation fold.

        Built out of `_synth_run` with the SAME `subj_offset`/`idx_offset` the
        headline's fold uses, so the control is scored on exactly the subjects
        and cache rows that fold of the headline was scored on -- which is what a
        real per-fold control is, and what makes the pooled control and the
        pooled headline differenceable. `auc_exact` sets the AUC, and the seed
        enters the generator's key, so two replicates of one permutation index on
        two folds carry genuinely different predictions.

        `test_auc_ci95` declares the FOLD's own sizes, exactly as stage 5 writes
        them. Those are the sizes the per-fold reading reports and the pooled
        reading replaces; nothing here declares the pool's.
        """
        r = _synth_run(cohort_, condition, seed, 0.55,
                       n_subj=CV_SUBJ, n_pos_subj=7, slices=CV_SLICES, pos_rate=1.0,
                       subj_offset=CV_SUBJ * fold, idx_offset=1000 * fold,
                       auc_exact=auc)
        det: Dict[str, Any] = {"variant": variant, "label_semantics": label_semantics,
                               "subject_col": "subject_id", "transform": None,
                               "wall_seconds": 1.0}
        if declare_split_col:
            det["split_col"] = f"cv{fold}_split"
        if with_ci:
            blk = r["test"]
            det["test_auc_ci95"] = {
                "auc": float(blk["auc"]), "lo": max(0.0, float(blk["auc"]) - 0.13),
                "hi": min(1.0, float(blk["auc"]) + 0.13),
                "n": int(blk["n"]), "n_pos": int(blk["n_pos"]),
                "n_clusters": CV_SUBJ, "n_boot_ok": 1180, "n_boot_degenerate": 820,
                "method": "subject_id-clustered percentile bootstrap, B=2000"}
        det.update(detail or {})
        r = dict(r)
        r.update({"region": region, "val": None, "control": control,
                  "control_detail": det})
        return r

    def _write_cv_controls(root_: Path, payloads: Sequence[Tuple[int, dict]]) -> None:
        """Write (fold, payload) pairs into `<root>/<cohort>_cv<fold>/`."""
        for fold, pl in payloads:
            _write_control_payloads(root_ / f"{pl['cohort']}_cv{fold}", [pl])

    # Fold 0 of the permutation control scores WELL ABOVE chance and folds 1-4
    # sit below it. That is the real prostate_t2 shape (per-fold means 0.622 /
    # 0.484 / 0.695 / 0.605 / 0.678, i.e. a null whose answer depends entirely on
    # which fold you read) with the spread widened so the two readings land on
    # opposite sides of the C3 threshold.
    PERM_FOLD_AUC = {0: 0.86, 1: 0.44, 2: 0.42, 3: 0.43, 4: 0.45}

    def _perm_payloads(cohort_: str = PRIMARY_COHORT, n_idx: int = 20,
                       extra_copy: bool = False) -> List[Tuple[int, dict]]:
        out: List[Tuple[int, dict]] = []
        for i in range(n_idx):
            for f in range(CV_K):
                out.append((f, _cv_control(
                    cohort_, f, "label_permutation", PERM_FOLD_AUC[f],
                    variant=f"perm{i:02d}", seed=1000 + i,
                    detail={"permutation_seed": 1000 + i,
                            "permutation_unit": "subject_id"})))
        if extra_copy:
            # A GENUINE duplicate: the same permutation index AND the same fold,
            # written twice. This is the one thing the old key caught and the new
            # key must go on catching.
            out.append(out[0])
        return out

    def _destroy_and_rest(cohort_: str, bg: Dict[int, float], scr: Dict[int, float],
                          root_: Path) -> None:
        """Per-fold background/scramble, plus the fold-less acquisition/confound."""
        _write_cv_controls(root_, [
            (f, _cv_control(cohort_, f, "background_only", bg[f], region="background"))
            for f in range(CV_K)])
        _write_cv_controls(root_, [
            (f, _cv_control(cohort_, f, "phase_scramble", scr[f],
                            detail={"scramble_seed": 20240517,
                                    "scope": "within body mask only"}))
            for f in range(CV_K)])
        _write_control_payloads(root_, [
            p for p in _synth_control_payloads(cohort_, 0.50, 0.55, 0.55, 0.74, 0.52,
                                               n_permutations=0, n_clusters=CV_CL,
                                               ci_n=CV_N, ci_n_pos=35)
            if p["control"] in ("acquisition_split", "confound_predictability")])

    def _pooled_auc_of(payloads: Sequence[Tuple[int, dict]]) -> float:
        """The AUC of the concatenated per-fold test blocks. Computed here, from
        the files, so the assertion does not read the number it is checking."""
        lab = np.concatenate([np.array(pl["test"]["labels"], dtype=int)
                              for _f, pl in payloads])
        pr = np.concatenate([np.array(pl["test"]["probs"], dtype=float)
                             for _f, pl in payloads])
        return _safe_auc(lab, pr)

    # ---- [A] a replicate is a PERMUTATION AND A FOLD ------------------------
    # `Controls.permutation_null` de-duplicated on `permutation_seed` alone, and
    # stage 5 assigns that seed by replicate INDEX, independent of fold. On the
    # real tree that is 100 distinct payloads per cohort -- 20 indices x 5 folds,
    # five DISJOINT subject sets, 100 distinct probability vectors -- of which it
    # kept the 20 `sorted(rglob)` reached first (all of fold 0) and reported the
    # other 80 as duplicates. The reported prostate_t2 null 0.622 [0.503, 0.719]
    # was cv0 alone; the five fold means are 0.622 / 0.484 / 0.695 / 0.605 /
    # 0.678, so had cv1 sorted first C3 would have PASSED instead of FAILED. A
    # criterion outcome decided by directory sort order is not a criterion.
    perms_a = _perm_payloads()

    def _b_a(res: Path) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        croot = res.parent / "controls" / "results"
        _write_cv_controls(croot, perms_a)
        _destroy_and_rest(PRIMARY_COHORT, {f: 0.55 for f in range(CV_K)},
                          {f: 0.55 for f in range(CV_K)}, croot)
    ctx_a, md_a, _ = _run_case("SIX_A_perm_index_and_fold", _b_a)
    v_a, c_a = _codes(ctx_a)
    ctl_a = ctx_a["controls"]
    null_a = ctl_a.permutation_null(PRIMARY_COHORT, HEADLINE_CONDITION)
    fold0_mean = float(np.mean([PERM_FOLD_AUC[0]] * 1))
    pooled_means = [_pooled_auc_of([(f, pl) for f, pl in perms_a
                                    if pl["control_detail"]["permutation_seed"] == 1000 + i])
                    for i in range(20)]
    p_a: List[str] = []
    if int(null_a.get("n", 0)) != 20:
        p_a.append(f"the null counts {null_a.get('n')} replicate(s) over 100 files of "
                   f"20 indices x 5 folds; it must count 20")
    if int(null_a.get("n_duplicates", -1)) != 0:
        p_a.append(f"{null_a.get('n_duplicates')} of the 100 payloads were called "
                   f"duplicates; they are 20 indices on 5 DISJOINT folds and none of "
                   f"them is a copy of another")
    if int(null_a.get("n_pooled", 0)) != 20:
        p_a.append(f"only {null_a.get('n_pooled')} of the counted replicates were "
                   f"pooled out-of-fold; each index must be pooled across its folds "
                   f"exactly as the headline is")
    if abs(float(null_a.get("mean", 0.0)) - float(np.mean(pooled_means))) > 0.01:
        p_a.append(f"the null mean is {null_a.get('mean'):.4f}; the mean of the 20 "
                   f"per-index out-of-fold AUCs recomputed from the files is "
                   f"{float(np.mean(pooled_means)):.4f}")
    if abs(float(null_a.get("mean", 0.0)) - fold0_mean) < 0.10:
        p_a.append("the pooled null is indistinguishable from the fold-0-only null, so "
                   "this tree cannot tell the two readings apart")
    for m in (null_a.get("members") or []):
        if m.get("n") != CV_N or m.get("n_clusters") != CV_CL:
            p_a.append(f"a counted replicate declares {m.get('n')} slices / "
                       f"{m.get('n_clusters')} subjects; every one of them is pooled "
                       f"over the whole cohort and must declare {CV_N}/{CV_CL}")
            break
    if c_a.get("C3") != "pass":
        p_a.append(f"C3 is {c_a.get('C3')}; the pooled null sits at chance "
                   f"({null_a.get('mean'):.3f}, range [{null_a.get('min'):.3f}, "
                   f"{null_a.get('max'):.3f}]) and C3 must be decided on it, not on "
                   f"fold 0's {fold0_mean:.3f}")
    if c_a.get("C8") == "missing":
        p_a.append("C8 is MISSING: the null is pooled onto the headline's own test "
                   "set, so the comparison is no longer between two test sets")
    # ... and a REAL duplicate -- same index AND same fold -- is still caught.
    def _b_a2(res: Path) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        croot = res.parent / "controls" / "results"
        _write_cv_controls(croot, _perm_payloads(extra_copy=True))
        _destroy_and_rest(PRIMARY_COHORT, {f: 0.55 for f in range(CV_K)},
                          {f: 0.55 for f in range(CV_K)}, croot)
    ctx_a2, _, _ = _run_case("SIX_A_dup_same_index_and_fold", _b_a2)
    null_a2 = ctx_a2["controls"].permutation_null(PRIMARY_COHORT, HEADLINE_CONDITION)
    if int(null_a2.get("n_duplicates", 0)) < 1 or int(null_a2.get("n", 0)) != 20:
        p_a.append(f"a second copy of one (index, fold) gave n={null_a2.get('n')}, "
                   f"duplicates={null_a2.get('n_duplicates')}; the same permutation on "
                   f"the same fold written twice is one draw")
    if v_a == "SUPPORTED":
        p_a.append("SUPPORTED reached on this tree")
    _report("null_replicate_is_index_and_fold", p_a,
            f"100 files = 20 indices x 5 folds -> 20 out-of-fold replicates on the "
            f"headline's own {CV_N}/{CV_CL} test set (fold 0 alone would have said "
            f"{fold0_mean:.2f}); a repeat of one (index, fold) is still one draw")

    # ---- [B] controls are pooled out-of-fold, like the headline -------------
    # C4/C5/C7 differenced a control against the headline while the control was
    # the MEAN OF ITS PER-FOLD AUCs and the headline was a pooled out-of-fold
    # vector. Two different estimators of two different quantities, so part of
    # every gap they reported was the aggregation, not the control. Measured on
    # the real prostate_t2 tree: background per-fold mean 0.5956 vs pooled
    # out-of-fold 0.6038, i.e. the C4 gap moves 0.033 -> 0.025 on the same five
    # files, and the per-fold envelope had no fingerprint at all (its members
    # disagree on every size), which is why C7 was MISSING rather than decided.
    BG_FOLD = {0: 0.72, 1: 0.42, 2: 0.44, 3: 0.43, 4: 0.45}
    SCR_FOLD = {0: 0.70, 1: 0.43, 2: 0.45, 3: 0.44, 4: 0.42}

    def _b_b(res: Path) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        croot = res.parent / "controls" / "results"
        _write_cv_controls(croot, _perm_payloads())
        _destroy_and_rest(PRIMARY_COHORT, BG_FOLD, SCR_FOLD, croot)
    ctx_b, md_b, vj_b = _run_case("SIX_B_controls_pooled_oof", _b_b, bootstrap=400)
    v_b, c_b = _codes(ctx_b)
    ctl_b = ctx_b["controls"]
    p_b: List[str] = []
    for canon, per_fold in (("background", BG_FOLD), ("scramble", SCR_FOLD)):
        est_b = ctl_b.estimate(PRIMARY_COHORT, canon)
        want = _pooled_auc_of([(f, _cv_control(
            PRIMARY_COHORT, f,
            "background_only" if canon == "background" else "phase_scramble",
            per_fold[f], region="background" if canon == "background" else "full",
            detail=None if canon == "background" else {
                "scramble_seed": 20240517, "scope": "within body mask only"}))
            for f in range(CV_K)])
        mean_of_folds = float(np.mean(list(per_fold.values())))
        if abs(est_b.point - want) > 0.01:
            p_b.append(f"the {canon} control reports {est_b.point:.4f}; the AUC of its "
                       f"five folds concatenated is {want:.4f}")
        if abs(est_b.point - mean_of_folds) < 0.02:
            p_b.append(f"the {canon} control's pooled AUC {est_b.point:.4f} cannot be "
                       f"told apart from the mean of its per-fold AUCs "
                       f"{mean_of_folds:.4f}; this tree does not test the difference")
        if est_b.n != CV_N or est_b.n_clusters != CV_CL:
            p_b.append(f"the pooled {canon} control declares {est_b.n} slices / "
                       f"{est_b.n_clusters} subjects, not the headline's "
                       f"{CV_N}/{CV_CL}")
        if (est_b.detail or {}).get("is_envelope"):
            p_b.append(f"the {canon} control is still an envelope over per-fold runs")
    crit_b = {c.code: c for c in next(v for v in ctx_b["verdicts"]
                                      if v.cohort == PRIMARY_COHORT).criteria}
    for code in ("C4", "C7"):
        if crit_b[code].status == "missing":
            p_b.append(f"{code} is MISSING on a control pooled onto the headline's own "
                       f"test set: {crit_b[code].detail[:160]}")
        if not crit_b[code].test_set_verified:
            p_b.append(f"{code} did not verify that the two sides share a test set")
    # The acquisition arms are NOT merged -- they are disjoint protocol groups by
    # design -- and the criterion has to say which basis it differenced on.
    acq_b = ctl_b.estimate(PRIMARY_COHORT, "acquisition")
    if "comparison_basis" not in (acq_b.detail or {}):
        p_b.append("the acquisition control carries no comparison basis")
    if "COMPARISON BASIS" not in crit_b["C5"].evidence:
        p_b.append("C5's evidence does not state the basis it differenced on")
    # The point must be ONE direction's own AUC (the worse of 0.74 / 0.80), not
    # the AUC of the two concatenated: merging disjoint protocol groups trained
    # on opposite halves would produce a number that is an estimate of nothing.
    if abs(acq_b.point - 0.74) > 1e-6:
        p_b.append(f"the acquisition control reports {acq_b.point:.4f}; the worse of "
                   f"its two directions is 0.74 and the two are never merged")
    # This tree is the POSITIVE half: every control really does collapse, on the
    # headline's own test set, so the criteria must be decided and the verdict
    # must be reachable. A pooling rule that refuses everything is not a check.
    if v_b != "SUPPORTED":
        p_b.append(f"verdict is {v_b} on a tree where every control collapses on the "
                   f"headline's own test set; the pooled reading is refusing runs it "
                   f"should decide ({', '.join(sorted(k + ':' + val for k, val in c_b.items() if val != 'pass'))})")
    _report("controls_pooled_out_of_fold", p_b,
            "background/scramble are pooled out-of-fold onto the headline's own test "
            "set (not averaged over folds) so C4/C7 are decided; the acquisition arms "
            "stay separate and C5 states its comparison basis")

    # ---- [B2] the drawn curve is the estimand the table reports -------------
    # `pool_runs` averages seeds into ONE probability vector. That vector is the
    # right thing to bootstrap (concatenating seeds would fake independence and
    # shrink intervals by sqrt(N)) but it is a 2-model ENSEMBLE, and its AUC is
    # not the mean of the per-seed AUCs that stage 4 reports. The ROC figure
    # drew the ensemble and labelled it with stage 4's number: on the real
    # prostate_t2 tree that put the curve 0.021-0.033 above its own headline
    # (magnitude 0.683 -> 0.715, phase 0.629 -> 0.650, both 0.594 -> 0.627) and
    # sent every clinical cohort DEGRADED. The same error made C5's matched
    # comparison difference a 2-seed ensemble against a 1-seed stage-5 control.
    p_b2: List[str] = []
    from sklearn.metrics import roc_auc_score as _ras

    def _seeded_run(seed_: int, probs_: List[float], labels_: List[int],
                    idx_: List[int]) -> dict:
        return {"cohort": "t", "condition": "phase", "seed": seed_,
                "_split_family": "cv",
                "test": {"probs": probs_, "labels": labels_, "cache_idx": idx_,
                         "patient_ids": [f"S{i}" for i in idx_],
                         "auc": _ras(labels_, probs_)}}

    # Complementary errors: each seed misranks a block the other gets right, so
    # averaging beats BOTH constituents and ensemble > mean-of-seeds strictly.
    _lab = [0] * 8 + [1] * 8
    _idx = list(range(16))
    _pa = [0.1, 0.2, 0.3, 0.4, 0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6, 0.6, 0.7, 0.8, 0.9, 1.0]
    _pb = [0.9, 0.9, 0.9, 0.9, 0.1, 0.2, 0.3, 0.4, 0.7, 0.8, 0.9, 1.0, 0.6, 0.6, 0.6, 0.6]
    _pooled = pool_runs([_seeded_run(42, _pa, _lab, _idx),
                         _seeded_run(123, _pb, _lab, _idx)], {}, "patient_id")
    if _pooled is None or sorted(_pooled.per_seed_probs) != [42, 123]:
        p_b2.append("pool_runs kept no per-seed view of two seeds over identical rows")
    else:
        _mean = float(np.mean([_ras(_lab, _pa), _ras(_lab, _pb)]))
        _ens = _ras(_pooled.labels, _pooled.probs)
        _sm = seed_mean_auc(_pooled)
        if _sm is None or abs(_sm[0] - _mean) > 1e-9 or _sm[1] != 2:
            p_b2.append(f"seed_mean_auc gave {_sm} for a mean-over-seeds of {_mean:.6f}")
        if _ens <= _mean + 1e-6:
            p_b2.append(
                f"the fixture does not separate the two estimands (ensemble {_ens:.4f} "
                f"vs mean-over-seeds {_mean:.4f}); it cannot detect the defect")
        # Area under the vertically averaged ROC IS the mean of the per-seed
        # AUCs, which is what lets the figure show one and print the other.
        _g = np.linspace(0, 1, 2001)
        _tpr = seed_mean_roc(_pooled, _g)
        if _tpr is None or abs(float(np.trapezoid(_tpr, _g)) - _mean) > 2e-3:
            p_b2.append("the vertically averaged ROC's area is not the mean of the "
                        "per-seed AUCs, so the drawn curve and the printed number "
                        "are different quantities")
        # The matched comparison reports the one-model basis and says so.
        _m = _matched_headline(_pooled, _idx[:12])
        _msel = list(range(12))
        _mexp = float(np.mean([_ras(_lab[:12], _pa[:12]), _ras(_lab[:12], _pb[:12])]))
        if _m is None or abs(_m[0] - _mexp) > 1e-9 or _m[2] != 2:
            p_b2.append(f"_matched_headline gave {_m}; the seed-mean on those rows "
                        f"is {_mexp:.6f} over 2 seeds")
    # Seeds that scored DIFFERENT rows: pool_runs already intersects to the rows
    # every run scored, so the pooled vector is those 12 and the per-seed view is
    # over exactly them. What must never happen is a per-seed vector of a
    # different length from the pooled one -- that is the size-borrowing this
    # file refuses everywhere else.
    _ragged = pool_runs([_seeded_run(42, _pa, _lab, _idx),
                         _seeded_run(123, _pb[:12], _lab[:12], _idx[:12])],
                        {}, "patient_id")
    if _ragged is None or _ragged.n != 12:
        p_b2.append(f"pooling seeds over 16 and 12 rows gave "
                    f"{None if _ragged is None else _ragged.n} rows, not the 12 both scored")
    elif any(len(v) != _ragged.n for v in _ragged.per_seed_probs.values()):
        p_b2.append("a per-seed vector is a different length from the pooled vector "
                    "it is supposed to be an alternative reading of")
    # FAIL CLOSED on a collision: two runs claiming the SAME seed and the same
    # rows are not two seeds to average -- one is a re-run this function cannot
    # tell apart -- so the per-seed view is dropped rather than silently keeping
    # whichever landed last.
    _collide = pool_runs([_seeded_run(42, _pa, _lab, _idx),
                          _seeded_run(42, _pb, _lab, _idx),
                          _seeded_run(123, _pb, _lab, _idx)], {}, "patient_id")
    if _collide is not None and _collide.per_seed_probs:
        p_b2.append("two runs claiming the same seed over the same rows still produced "
                    "a per-seed view; one of them is an untellable re-run")
    if _collide is not None and seed_mean_auc(_collide) is not None:
        p_b2.append("seed_mean_auc did not fail closed on a duplicated seed")
    # A single seed has no mean-over-seeds to report and must not invent one --
    # this is every stage-5 control, which runs at one seed.
    _one = pool_runs([_seeded_run(42, _pa, _lab, _idx)], {}, "patient_id")
    if _one is not None and (_one.per_seed_probs or seed_mean_auc(_one) is not None):
        p_b2.append("a single-seed pool reported a mean over seeds")
    # Pooling folds keeps the per-seed view row-aligned, and refuses it when the
    # folds ran different seeds ("seed 42" here and "seed 7" there is not one
    # vector). Folds must be disjoint in rows AND subjects to pool at all.
    _f0 = pool_runs([_seeded_run(42, _pa, _lab, _idx),
                     _seeded_run(123, _pb, _lab, _idx)], {}, "patient_id")
    _idx2 = [i + 100 for i in _idx]
    _f1 = pool_runs([_seeded_run(42, _pa, _lab, _idx2),
                     _seeded_run(123, _pb, _lab, _idx2)], {}, "patient_id")
    _oof, _ = pool_folds_oof({0: _f0, 1: _f1})
    if _oof is None or sorted(_oof.per_seed_probs) != [42, 123]:
        p_b2.append("pooling two folds dropped the per-seed view")
    elif any(len(v) != _oof.n for v in _oof.per_seed_probs.values()):
        p_b2.append("the pooled per-seed vectors are not aligned with the pooled rows")
    _f1b = pool_runs([_seeded_run(42, _pa, _lab, _idx2),
                      _seeded_run(7, _pb, _lab, _idx2)], {}, "patient_id")
    _oof2, _ = pool_folds_oof({0: _f0, 1: _f1b})
    if _oof2 is not None and _oof2.per_seed_probs:
        p_b2.append("folds trained on different seed sets were concatenated into a "
                    "per-seed view; 'seed 42' would be seed 42 in one fold and a "
                    "different model in the other")
    # ...and the FIGURE itself, end to end. The three checks above all pass with
    # `fig_roc` reverted to plotting the ensemble, because they exercise the
    # helpers rather than the caller -- so the curve is drawn here against a
    # stage-4 band holding the mean-over-seeds, and the disagreement guard that
    # first caught this on the real tree must stay silent.
    if _pooled is not None and _pooled.per_seed_probs:
        _mm: List[str] = []
        _band = {"phase": {"point": float(np.mean([_ras(_lab, _pa), _ras(_lab, _pb)])),
                           "lo": 0.40, "hi": 0.80, "_from_s04": True}}
        _fig_dir = root / "roc_estimand"
        _fig_dir.mkdir(parents=True, exist_ok=True)
        fig_roc("t", {"phase": _pooled}, _band, _fig_dir, 60, mismatches=_mm)
        if _mm:
            p_b2.append("the drawn curve disagrees with the stage-4 point it is "
                        "labelled with: " + "; ".join(_mm))

    _report("roc_curve_is_the_reported_estimand", p_b2,
            "the drawn curve is the mean-over-seeds ROC (area == stage 4's point), not "
            "the higher-scoring seed ENSEMBLE; matched comparisons use the same "
            "one-model basis as the single-seed controls; ragged seeds pool to the rows "
            "both scored, and a single-seed, duplicated-seed or mixed-seed pool reports "
            "no per-seed view at all")

    # ---- [C] the test-set check reads all five fingerprint components -------
    # `_test_set_mismatch` compared `n` and `n_clusters` and stopped, while
    # `split_family`, `folds` and `subjects` are first-class members of
    # `_FP_ORDER`, are read out of every control payload, and were carried the
    # whole way here to be ignored. Two test sets can agree exactly on both
    # counts and share not one subject.
    p_c: List[str] = []

    def _est_with(**fp) -> Estimate:
        e = Estimate(point=0.6, lo=0.5, hi=0.7,
                     n=fp.get("n"), n_clusters=fp.get("n_clusters"))
        e.detail = {"test_set": dict(fp)}
        return e

    same = dict(split_family="cv", folds=[0, 1, 2, 3, 4], n=CV_N, n_clusters=CV_CL,
                subjects=[f"S{i}" for i in range(CV_CL)])
    if _test_set_mismatch(_est_with(**same), _est_with(**same)):
        p_c.append("two identical fingerprints are reported as a mismatch; a check "
                   "that refuses everything is not a check")
    for comp, other_val in (("split_family", "cv@sweepB"),
                            ("folds", [0, 1, 2, 3, 5]),
                            ("subjects", [f"T{i}" for i in range(CV_CL)])):
        alt = dict(same, **{comp: other_val})
        if not _test_set_mismatch(_est_with(**same), _est_with(**alt)):
            p_c.append(f"a control differing from the headline on {comp} -- with the "
                       f"identical slice and subject COUNTS -- is accepted as the same "
                       f"test set")
    # ... and end to end: the same five-fold controls, in a SUBDIRECTORY of the
    # controls tree. That is a second sweep. Every count matches the headline's;
    # only the split family does not.
    def _b_c(res: Path) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        croot = res.parent / "controls" / "results"
        _write_cv_controls(croot / "sweepB", _perm_payloads())
        _destroy_and_rest(PRIMARY_COHORT, BG_FOLD, SCR_FOLD, croot / "sweepB")
    ctx_c, _, _ = _run_case("SIX_C_control_sweep_elsewhere", _b_c)
    v_c, c_c = _codes(ctx_c)
    bg_c = ctx_c["controls"].estimate(PRIMARY_COHORT, "background")
    if bg_c.n != CV_N or bg_c.n_clusters != CV_CL:
        p_c.append(f"the second sweep's background control declares {bg_c.n}/"
                   f"{bg_c.n_clusters}, so this tree does not exercise the case where "
                   f"only the family differs")
    for code in ("C4", "C7", "C8"):
        if c_c.get(code) not in ("missing", "fail"):
            p_c.append(f"{code} is {c_c.get(code)} on a control from a DIFFERENT "
                       f"cross-validation sweep with identical sizes")
    if c_c.get("C4") == "pass" or v_c == "SUPPORTED":
        p_c.append("a control sweep in another directory was accepted as this one")
    _report("test_set_check_reads_all_five", p_c,
            "split family, fold set and subject-id set are compared alongside the two "
            "counts; identical fingerprints still match")

    # ---- [D] a scanner-identity AUC is never a diagnostic control ------------
    # The Controls docstring has always claimed confound-cohort AUCs are never
    # mixed with diagnostic ones, and `select` filtered on cohort/control/
    # condition only. A payload carrying `label_semantics="confound:<target>"`
    # filed under background_only was selected, differenced against a cancer AUC,
    # and PASSED C4 -- on a number that says nothing whatever about the
    # background channel.
    def _b_d(res: Path, sem: str) -> None:
        _cv_runs(res)
        (res / "statistics.json").write_text(json.dumps(_s04_order(_cv_stats())))
        croot = res.parent / "controls" / "results"
        _write_cv_controls(croot, _perm_payloads())
        _write_cv_controls(croot, [
            (f, _cv_control(PRIMARY_COHORT, f, "background_only", 0.50,
                            region="background", label_semantics=sem))
            for f in range(CV_K)])
        _write_cv_controls(croot, [
            (f, _cv_control(PRIMARY_COHORT, f, "phase_scramble", SCR_FOLD[f],
                            detail={"scramble_seed": 20240517,
                                    "scope": "within body mask only"}))
            for f in range(CV_K)])
        _write_control_payloads(croot, [
            p for p in _synth_control_payloads(PRIMARY_COHORT, 0.50, 0.55, 0.55,
                                               0.74, 0.52, n_permutations=0,
                                               n_clusters=CV_CL, ci_n=CV_N,
                                               ci_n_pos=35)
            if p["control"] in ("acquisition_split", "confound_predictability")])
    ctx_d, _, _ = _run_case(
        "SIX_D_confound_semantics_in_background",
        lambda res: _b_d(res, "confound:receiver_channels"), bootstrap=400)
    v_d, c_d = _codes(ctx_d)
    ctx_d2, _, _ = _run_case("SIX_D_diagnostic_background",
                             lambda res: _b_d(res, DIAGNOSTIC_SEMANTICS),
                             bootstrap=400)
    v_d2, c_d2 = _codes(ctx_d2)
    p_d: List[str] = []
    if c_d.get("C4") != "missing":
        p_d.append(f"C4 is {c_d.get('C4')} on a background-only control whose only "
                   f"runs carry label_semantics='confound:receiver_channels'; a "
                   f"scanner-identity AUC is not a measurement of the background "
                   f"channel and the criterion is unmeasured, not passed")
    if ctx_d["controls"].select(PRIMARY_COHORT, "background", HEADLINE_CONDITION):
        p_d.append("select() still returns the confound-semantics runs to the "
                   "diagnostic criteria")
    if c_d2.get("C4") != "pass":
        p_d.append(f"C4 is {c_d2.get('C4')} on the SAME runs carrying diagnostic "
                   f"semantics; the filter is rejecting everything rather than "
                   f"rejecting the wrong thing")
    # C6 must be untouched: it is the one reader that has to SEE every semantics,
    # because a confound run it cannot score can only lower the maximum it takes.
    if c_d.get("C6") != c_d2.get("C6"):
        p_d.append(f"C6 changed ({c_d2.get('C6')} -> {c_d.get('C6')}) when a "
                   f"background run's semantics changed; C6 reads the confound runs "
                   f"and nothing else")
    if not ctx_d["controls"].select(PRIMARY_COHORT, "confound", HEADLINE_CONDITION,
                                    semantics=None):
        p_d.append("the confound runs are no longer reachable at all, so C6 cannot "
                   "report a run it could not score")
    if "SUPPORTED" == v_d:
        p_d.append("SUPPORTED reached on a background control that is a scanner AUC")
    _report("controls_never_mix_label_semantics", p_d,
            "a confound-semantics run filed under background_only is not a diagnostic "
            "control (C4 MISSING); the same runs with diagnostic semantics still "
            "decide it, and C6 still sees every confound run")

    total = len(scenarios) + 2 + extra_checks
    print("=" * 82)
    print(f"self-test: {total - failures}/{total} checks passed")
    if failures:
        print(f"artifacts under {root}")
    print("=" * 82)
    if failures == 0:
        shutil.rmtree(root, ignore_errors=True)
    return 1 if failures else 0


# ==========================================================================
# CLI
# ==========================================================================

def parse_args(argv=None):
    default_out = (common.OUT_ROOT if common else Path("pipeline_out"))
    p = argparse.ArgumentParser(
        description="PhaseDx stage 6: figures, RESULTS.md, and the verdict",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--results-dir", default=str(default_out / "results"),
                   help="directory holding stage-3 run JSONs and statistics.json")
    p.add_argument("--cache-dir", default=str(default_out / "cache"),
                   help="stage-2 cache, used for the qualitative panel")
    p.add_argument("--cohorts-dir", default=str(default_out / "cohorts"),
                   help="stage-1 output, used for subject_id and the cohort table")
    p.add_argument("--out", default=str(default_out / "report"),
                   help="report directory (RESULTS.md, verdict.json, figures/)")
    p.add_argument("--stats", default=None, help="explicit path to statistics.json")
    p.add_argument("--controls", default=None,
                   help="extra DIRECTORY of stage-5 control run JSONs to search "
                        "(searched recursively, in addition to the default "
                        "<out-root>/controls/results locations). s05 writes no "
                        "aggregate controls.json; passing a file here finds nothing.")
    p.add_argument("--bootstrap", type=int, default=DEFAULT_BOOTSTRAP,
                   help="resamples for the s06 fallback cluster bootstrap")
    p.add_argument("--seed", type=int, default=0, help="bootstrap seed")
    p.add_argument("--dpi", type=int, default=300, help="figure resolution")
    p.add_argument("--log-level", default="INFO")
    p.add_argument("--self-test", action="store_true",
                   help="run the synthetic end-to-end verdict test and exit")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = parse_args(argv)
    if args.self_test:
        return self_test()
    logging.basicConfig(level=getattr(logging, str(args.log_level).upper(), logging.INFO),
                        format="%(asctime)s %(levelname)-7s %(message)s",
                        datefmt="%H:%M:%S")
    rc, ctx = build_report(args)

    print()
    print("=" * 78)
    print("STAGE 6 SUMMARY")
    print("=" * 78)
    for cv in ctx["verdicts"]:
        print(f"  {cv.cohort:<16} {cv.verdict}")
        print(f"  {'':<16} {cv.reason}")
    if not ctx["verdicts"]:
        print("  no cohort could be evaluated -- no stage-3 results found")
    if ctx["degraded"]:
        print()
        print("  DEGRADED:")
        for d in ctx["degraded"]:
            print(f"    - {d}")
    print()
    print(f"  report  -> {Path(args.out) / 'RESULTS.md'}")
    print(f"  verdict -> {Path(args.out) / 'verdict.json'}")
    print(f"  figures -> {Path(args.out) / 'figures'}")
    print("=" * 78)
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
