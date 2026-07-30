"""
s15_positional_theory.py
------------------------
The ANALYTICAL companion to s14_trivialbaselines: what the positional null model's
AUROC *must* be, derived from the label file, before any model is trained.

WHY THIS EXISTS
    s14 measures. It takes a published label table, fits P(label | relative slice
    position) on the training rows, scores the test rows, and reports a number. That is
    an empirical claim of the form "this happens, here is how often".

    This module makes the number PREDICTABLE. It shows that the slice-level AUROC of
    the binned positional baseline is a closed-form functional of a quantity that can be
    read off a label file with no model, no training and no split: the positional risk
    profile pi(z) = P(label = 1 | relative slice position z). Given that profile, the
    achievable AUROC is fixed by arithmetic. So a benchmark designer can compute one
    number from their own labels and read off how much slice-level inflation their
    evaluation protocol is about to certify.

    It also shows, as an exact algebraic fact rather than a statistical tendency, WHY
    the slice-to-patient collapse happens, and states the precondition under which it
    must happen and the precondition under which it must not.

===========================================================================
1. THE ESTIMATOR, EXACTLY AS IMPLEMENTED
===========================================================================
Everything below is about the estimator s14/s12 actually run, not an idealised
continuum analogue. To fix notation, that estimator is:

    relative position   r_i = (z_i - min_v z) / (max_v z - min_v z) within volume v,
                        or a published normalised-z column when one exists;
    bins                B equal-width bins of [0, 1], b(r) = clip(digitize(r, e) - 1,
                        0, B-1) with e_k = k/B, so the last bin is closed on the right;
    fit                 p_hat_b = mean label over TRAINING rows in bin b, falling back
                        to the training prevalence when bin b holds no training row;
    score               s_hat_i = p_hat_{b(r_i)} for TEST rows;
    metric              mid-rank AUROC (Mann-Whitney with ties at one half).

===========================================================================
2. PROPOSITION 1 (EXACT). The AUROC is a B x B bilinear form.
===========================================================================
Let a_b and c_b be the numbers of positive and negative TEST slices in bin b, with
A = sum_b a_b and C = sum_b c_b. Then, exactly and with no approximation of any kind,

    AUC  =  (1 / (A C)) * sum_{b,b'} a_b c_b' * K_{b b'},
    K_{b b'} = 1{p_hat_b > p_hat_b'} + (1/2) 1{p_hat_b = p_hat_b'}.

PROOF. The mid-rank AUROC equals the proportion of (positive, negative) test pairs in
which the positive scores higher, counting ties as one half. The score is constant
within a bin, so every pair with the positive in bin b and the negative in bin b'
contributes the same amount K_{b b'}; there are a_b c_b' such pairs. []

Three consequences that matter:

  (1a) The fitted rates enter ONLY through their order-with-ties. Any strictly
       increasing recalibration of p_hat leaves the AUROC unchanged. The estimator's
       calibration is irrelevant; only its ranking of the B bins is used.
  (1b) The sufficient statistic is 2B test counts plus one ordering of B objects. A
       slice-level AUROC on 750,000 slices is a function of 40 numbers and a
       permutation of 20.
  (1c) The same identity holds for pooled out-of-fold predictions if the groups are
       taken to be (fold, bin) pairs rather than bins, because the score is still
       constant on each such group. Everything below that says "bin" can be read as
       "score-constant group".

===========================================================================
3. PROPOSITION 2 (CLOSED FORM). The population AUROC is a Gini coefficient.
===========================================================================
Define the POSITIONAL RISK PROFILE of a label table:

    q_b   = P(bin = b)                     the positional density of slices
    pi_b  = P(label = 1 | bin = b)         the positional risk
    theta = sum_b q_b pi_b                 the slice prevalence

Substituting the population targets into Proposition 1 -- p_hat_b -> pi_b, and the test
counts by their expectations a_b/A -> q_b pi_b / theta, c_b/C -> q_b (1 - pi_b)/(1 -
theta) -- gives the ORACLE positional AUROC

    A(q, pi) = 1/(theta (1-theta)) * sum_{b,b'} q_b q_b' pi_b (1 - pi_b')
                                     * [1{pi_b > pi_b'} + (1/2) 1{pi_b = pi_b'}].

Write K = 1/2 + (1/2) sgn(pi_b - pi_b'). The 1/2 part sums to exactly 1/2 because
sum_b q_b pi_b = theta and sum_b q_b (1 - pi_b) = 1 - theta. Pairing (b, b') with
(b', b) in the sgn part collapses pi_b(1-pi_b') - pi_b'(1-pi_b) to pi_b - pi_b'. Hence

    ---------------------------------------------------------------
    A  =  1/2  +  ( sum_{b < b'} q_b q_b' |pi_b - pi_b'| )
                  / ( 2 theta (1 - theta) )
    ---------------------------------------------------------------

The numerator is one half of the GINI MEAN DIFFERENCE of the random variable pi_beta
where beta ~ q. Writing G for the Gini coefficient of pi_beta (mean absolute difference
divided by twice the mean, the usual definition), the whole thing reduces to

    ===============================================
      A  =  (1/2) * ( 1 + G / (1 - theta) )
    ===============================================

THE POSITIONAL AUROC CEILING OF A LABEL FILE IS ONE HALF TIMES ONE PLUS THE GINI
COEFFICIENT OF ITS POSITIONAL RISK PROFILE, DIVIDED BY ONE MINUS PREVALENCE.

That is the screening statistic. It needs a label column, a slice index and a volume
id. It needs no split, no training, no model, no images and no GPU.

Sanity, both cases verified analytically in --self-test:
  * label independent of position: pi_b == theta for all b, every |pi_b - pi_b'| is 0,
    G = 0, and A = 0.500 exactly;
  * perfectly positionally separable label: pi_b in {0, 1}, the double sum is
    theta(1-theta), G = 1 - theta, and A = 1.000 exactly.

Four further properties, all used below:
  (2a) CEILING, not merely prediction. By Neyman-Pearson the ranking induced by the
       likelihood ratio maximises AUROC, and within the sigma-algebra generated by the
       bin index that ranking is the ranking by pi_b. So A is the LARGEST AUROC any
       function of B-binned slice position can attain. A measured positional baseline
       above A + sampling error is evidence of a bug or of leakage through some other
       channel, not of a better positional model.
  (2b) MONOTONE UNDER REFINEMENT. Refining the bins can only enlarge the sigma-algebra,
       so A is non-decreasing in B in the population. Any decrease along s14's measured
       5/10/20/50 bin sweep is therefore finite-sample, not structural.
  (2c) A >= 1/2 always, with equality iff pi is q-almost-surely constant.
  (2d) PREVALENCE MATTERS. For fixed G, A rises with theta. Two benchmarks with the
       same positional concentration are not equally inflatable; the rarer the label,
       the more concentration it takes to move the AUROC.

===========================================================================
4. FINITE SAMPLES: TWO BIASES, IN OPPOSITE DIRECTIONS
===========================================================================
A is a population quantity. Two things separate it from what s14 measures, and they
push opposite ways, which is why a naive comparison can look better than the theory
deserves.

4.1 ORDER-RECOVERY LOSS (pushes the MEASURED value DOWN)
    Proposition 1 says only the fitted ORDER is used. The fit is on M training slices,
    so for a pair of bins whose true rates are close the fitted order can invert. Take
    a pair with pi_b > pi_b' and let rho_{bb'} = P(p_hat_b > p_hat_b') + (1/2)
    P(p_hat_b = p_hat_b'). A correctly ordered pair contributes + q_b q_b'
    |pi_b - pi_b'|; a reversed one contributes the negative of that. Therefore

        E[AUC]  =  1/2 + ( sum_{b<b'} q_b q_b' |pi_b - pi_b'| (2 rho_{bb'} - 1) )
                          / ( 2 theta (1 - theta) )

    which is exact given the test counts at their expectations, and reduces to
    Proposition 2 when every pair is ordered correctly. rho is approximated by

        rho_{bb'} ~ Phi( |pi_b - pi_b'| / sqrt(v_b + v_b') ),
        v_b = pi_b (1 - pi_b) * deff_b / m_b,   m_b = expected TRAINING slices in bin b,

    with deff_b = 1 + (nbar_b - 1) * icc the design effect from within-volume label
    correlation and nbar_b the mean number of training slices per volume landing in bin
    b. This is the ONLY approximation in the chain, and it is validated by simulation
    rather than asserted.

    Note what deff does here. The brief for this section said clustering "affects the
    variance but not the expectation". That is true of the ORACLE predictor, whose
    ordering is fixed. It is NOT true of the fitted estimator: clustering inflates v_b,
    lowers rho, and so lowers E[AUC]. The correction is first order in deff. We state
    that rather than repeat the tidier claim.

    Note also what binning does to clustering. With B = 20 bins and a 30-slice stack,
    each volume contributes about 1.5 slices per bin, so nbar_b is near 1 and deff_b is
    near 1 whatever the ICC. Positional binning largely dissolves the clustering it is
    exposed to. That is a happy accident of the estimator, not a general fact.

4.2 PROFILE SELECTION BIAS (pushes the PREDICTED value UP)
    A itself has to be estimated. Plugging the empirical pi_hat_b into Proposition 2
    over-states A, because |.| is convex: E|pi_hat_b - pi_hat_b'| >= |pi_b - pi_b'| by
    Jensen, with equality only in the limit. Concretely, for D ~ N(delta, tau^2),

        E|D| = tau sqrt(2/PI) exp(-delta^2 / 2 tau^2) + delta (2 Phi(delta/tau) - 1),

    the folded-normal mean, which at delta = 0 equals tau sqrt(2/PI) > 0. So a label
    with NO positional structure still yields a plug-in ceiling above 0.5, by roughly
    sqrt(2/PI) * mean(tau) / (2 theta (1-theta)) -- the apparent performance of a
    saturated B-level model on its own training data. Two remedies, both implemented:

      * ANALYTIC: invert the folded-normal mean pairwise to recover an estimate of
        |delta| = |pi_b - pi_b'| from the observed |pi_hat_b - pi_hat_b'|;
      * CROSS-FITTING: split subjects in two, take the ordering from one half and the
        counts from the other, and apply Proposition 1. Honest by construction.

    A permutation null (labels shuffled within volume, which destroys the position-label
    link and preserves prevalence, depth and clustering) is also computed, exactly as
    s14 calibrates its baselines, so the plug-in ceiling can be read against its own
    null rather than against 0.5.

4.3 THE PREDICTION ACTUALLY REPORTED
    pred = folded-normal-debiased profile, put through the order-recovery-corrected
    expectation of section 4.1 at the true training size. Both corrections applied. The
    raw plug-in ceiling is reported alongside as the CERTIFICATE (property 2a).

===========================================================================
5. PROPOSITION 3 (EXACT). Patient-level aggregation annihilates position.
===========================================================================
This is the mechanistic heart of the paper and it is not a statistical statement.

Suppose relative position is computed by within-volume min-max rescaling, which is what
s14 does whenever a dataset does not publish a normalised z of its own. Then for a
volume v the vector of relative positions of its slices, and hence the multiset of bins
it occupies,

    Bset(v) = { b(r) : r in v },

is a deterministic function of v's SLICE INDEX PATTERN and of nothing else. It does not
depend on the patient, on the label, on the scanner, or on anything the model could
exploit. For a volume of n consecutively indexed slices it is a function of n alone:

    Bset(n) = { clip(floor(B k / (n-1)), 0, B-1) : k = 0 .. n-1 }.

Consequently the aggregated patient score is a deterministic function of the patient's
volume depths:

    max-aggregated  patient score = max over Bset  of p_hat,
    mean-aggregated patient score = mean over Bset (with multiplicity) of p_hat.

COROLLARY 3.1 (ANNIHILATION). If every volume has n >= B + 1 slices, then consecutive
    k differ by B/(n-1) <= 1 in the floor's argument, so the floor advances by at most
    one at each step, starts at 0 and ends at B-1: every bin is occupied by every
    volume. The max-aggregated score is then max_b p_hat_b for EVERY patient -- a
    constant. All patient pairs tie, and the max-aggregated patient AUROC is EXACTLY
    0.500 by the mid-rank convention. Not approximately, not on average: exactly. The
    same holds for MEAN aggregation whenever all volumes share a depth, and generally
    reduces the mean-aggregated score to a function of depth alone (Corollary 3.2).

    A FLOATING-POINT CAVEAT, because it is real and the code has to satisfy the claim
    it makes. At n = B + 1 exactly, every relative position k/B lands exactly on a bin
    edge. np.linspace's edge and the quotient k/B differ in the last bit for some k, and
    np.digitize then sends those slices one bin low, so a bin can be left empty. This is
    the only integer n above B for which the implication fails, measured exhaustively
    for B in {5, 10, 20, 50}. The bound stated in the paper is therefore n >= B + 2,
    which is what the released code actually satisfies; the exact-arithmetic version
    (n >= B + 1) is available as canonical_bins_exact and is tested separately.

COROLLARY 3.2 (DEPTH IS THE ONLY SURVIVOR). In general, the patient-level AUROC of the
    positional baseline equals the AUROC of a DEPTH-ONLY predictor: score each patient
    by the deterministic map from their volumes' slice counts to the aggregated rate.
    Positional information contributes nothing at patient level. Whatever patient-level
    signal remains -- above or below 0.5 -- is stack depth in disguise, and stack depth
    is a protocol fingerprint (scanner, coverage, era), not anatomy.

COROLLARY 3.3 (NO PARADOX). The slice-level AUROC can be arbitrarily close to 1 while
    the patient-level AUROC is exactly 0.5. This is not regression to the mean and not
    a small-sample artefact. Slice-level positional signal lives entirely in WITHIN-
    volume contrasts, and patient-level aggregation integrates over exactly those
    contrasts. A benchmark reporting slice-level AUROC is measuring a quantity that
    aggregation is guaranteed to destroy.

COROLLARY 3.4 (THE PRECONDITION IS DISCRIMINATING). If relative position comes from a
    published column that is NOT within-volume normalised -- DeepLesion's
    Normalized_lesion_location, LUNA16's world z in millimetres -- then volumes do not
    span [0, 1], Bset depends on the volume's actual anatomical extent, and the
    annihilation does NOT apply. The theory therefore predicts high patient-level AUROC
    for exactly those benchmarks and 0.500 for the others. Both halves are checked
    against measurement in --real, and both hold.

===========================================================================
6. RESULTS AS RUN (2026-07-29; regenerate with --all)
===========================================================================
6.1 SIMULATION, 240 cells crossing positional concentration (Gaussian width 0.06 to
    flat), bins (5/10/20/50), cohort size (60/200/400 subjects), depth (12/24/40
    slices), within-subject correlation (sd 0 / 0.20) and prevalence (0.15 / 0.40),
    30 replicates per cell:

        predicted (both corrections)    MAE 0.0087   bias +0.0087   max 0.0441
        uncorrected plug-in ceiling     MAE 0.0223   bias +0.0223   max 0.0970
        cross-fitted ceiling            MAE 0.0041   bias +0.0041   max 0.0207
        R^2 of predicted vs measured    0.9945

    Two honest readings. First, the bias is POSITIVE everywhere: the closed form
    over-states, never under-states, which for a screening statistic is the safe
    direction but is still a bias and is not corrected away by fitting a constant to
    this table. Second, CROSS-FITTING BEATS THE CLOSED-FORM CORRECTION by a factor of
    two. If a subject-level split of the label file is available -- and from a label
    file it always is -- cross-fitting is the better estimator, and the closed form's
    job is to explain the number rather than to be the most accurate way of getting it.

6.2 REAL DATA, 26 benchmark arms, every label table on disk:

        prediction error                mean +0.0047  MAE 0.0119  max 0.0492
        uncorrected ceiling error       mean +0.0109  MAE 0.0160  max 0.0715
        Proposition 1 identity gap      0.000e+00 (exact on all 26)
        replay vs committed s14 cards   0.000e+00 slice, 0.000e+00 patient, 21 cards

6.3 RSNA ICH under Burduja, Ionescu & Verga's own split geometry (744 held-out scans,
    100 draws, the paper's ONLY peer-reviewed slice-level comparator), predicted from
    the label file with no model against the Monte-Carlo mean of the measurement:

        label              Gini   predicted   measured    error   published   triv.frac
        any               0.408      0.7379     0.7385  -0.0006      0.9843       0.492
        epidural          0.432      0.7150     0.7172  -0.0022      0.9851       0.448
        intraparenchymal  0.480      0.7522     0.7520  +0.0002      0.9927       0.512
        intraventricular  0.590      0.8057     0.8057  +0.0001      0.9970       0.615
        subarachnoid      0.366      0.6918     0.6924  -0.0006      0.9821       0.399
        subdural          0.414      0.7205     0.7212  -0.0006      0.9682       0.472
                                          MAE 0.0007, max 0.0022

6.4 PROPOSITION 3, measured:
      * on the 13 arms whose annihilation precondition holds, the WITHIN-partition
        max-aggregated patient AUROC deviates from 0.500 by at most 0.000e+00;
      * on the 13 arms whose volumes are consecutively indexed, the DEPTH-ONLY
        surrogate reproduces the measured patient AUROC to 0.000e+00 -- slice position
        contributes literally nothing at patient level;
      * on the 11 arms whose precondition fails (DeepLesion's published normalised z,
        our breast and brain indices, LUNA16's candidate lists) the max-aggregated
        patient AUROC runs 0.476 to 0.963 and the depth surrogate misses by up to
        0.433. The annihilation is absent exactly where the theory says it must be.

6.5 WHERE THE PREDICTION FAILS, and why. The six worst arms are, in order,
    phasedx_prostate_dwi (+0.049, 122 test slices, 11 positive, 4 subjects),
    phasedx_breast (+0.044, 16 subjects), phasedx_prostate_t2 (-0.037, 7 subjects),
    rsna_ich_epidural (+0.029, prevalence 0.0035), phasedx_brain (+0.020) and
    fastmri_prostate_t2_published (-0.019). Two mechanisms, and they are separable:

      SMALL TEST ARM. MAE is 0.0191 where the test arm holds fewer than 300 positive
        slices and 0.0081 where it holds more. Our own five cohorts have official test
        arms of 4 to 136 subjects; on those the MEASUREMENT is the noisy quantity, not
        the prediction, and the residual is not evidence against the theory. The
        diagnostic printed alongside each failure is the ceiling recomputed on the TEST
        ROWS ONLY: for phasedx_prostate_dwi that is 0.921 against a measured 0.792, so
        the test arm's own positional profile implies a HIGHER value than was measured
        and the gap is order-recovery loss on 11 positive slices, not a wrong formula.

      EXTREME PREVALENCE. MAE is 0.0228 on the two arms with slice prevalence below
        0.02 (RSNA epidural at 0.0035, LUNA16 at 0.0017). The folded-normal debias
        assumes a normal approximation to a bin rate estimated from a handful of
        positives, and at 4 positives per 1000 slices that approximation is the weakest
        link in the chain. Note this is a limit of the CORRECTION, not of Propositions
        1 to 3, which remain exact: on the same RSNA epidural arm under Burduja's own
        larger-file protocol the error falls to -0.0022.

    Nothing here was selected. The 26 arms are every label table on disk that s14 has
    audited, reported in full, including the arms where the baseline finds nothing
    (PI-CAI at exactly 0.500, LUNA16 at 0.534) and the arms where we miss.

===========================================================================
7. WHAT IS EXACT AND WHAT IS NOT
===========================================================================
    EXACT, no approximation:  Propositions 1, 2, 3 and Corollaries 3.1-3.4.
    APPROXIMATE:              rho (normal approximation to the order-recovery
                              probability), deff (one-way ANOVA ICC), and the
                              folded-normal debias.
    NOT AVAILABLE IN CLOSED FORM: the sampling distribution of the measured AUROC.
                              Its spread is obtained by simulation and by s14's
                              subject-clustered bootstrap, never from a formula.

Usage:
    python pipeline/s15_positional_theory.py --self-test
    python pipeline/s15_positional_theory.py --simulate
    python pipeline/s15_positional_theory.py --real
    python pipeline/s15_positional_theory.py --rsna-burduja
    python pipeline/s15_positional_theory.py --all
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import s04_stats            # noqa: E402  -- one place for AUC/bootstrap in this study
import s14_trivialbaselines as s14  # noqa: E402  -- the estimator under analysis

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BINS = s14.DEFAULT_BINS          # 20, the harness default
SQRT2 = math.sqrt(2.0)
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


# ==========================================================================
# Small numerics: normal cdf and the folded-normal mean, without scipy
# ==========================================================================
# s14 is deliberately numpy+pandas only so it can run inside a data enclave with no
# package installs. This module keeps that promise.

_ERF = np.frompyfunc(math.erf, 1, 1)


def _phi_cdf(x) -> np.ndarray:
    """Standard normal CDF, vectorised, via math.erf."""
    x = np.asarray(x, dtype=float)
    return 0.5 * (1.0 + _ERF(x / SQRT2).astype(float))


def folded_normal_mean(delta, tau) -> np.ndarray:
    """
    E|D| for D ~ N(delta, tau^2). Strictly increasing in |delta|, with a floor of
    tau*sqrt(2/pi) at delta = 0 -- the floor that section 4.2 is about.
    """
    delta, tau = np.broadcast_arrays(np.asarray(delta, dtype=float),
                                     np.asarray(tau, dtype=float))
    delta = np.abs(delta).astype(float)
    tau = np.asarray(tau, dtype=float)
    out = delta.copy()
    ok = tau > 0
    if np.any(ok):
        d, t = delta[ok], tau[ok]
        out[ok] = (t * SQRT_2_OVER_PI * np.exp(-(d ** 2) / (2.0 * t ** 2))
                   + d * (2.0 * _phi_cdf(d / t) - 1.0))
    return out


def invert_folded_normal(observed, tau, n_iter: int = 60) -> np.ndarray:
    """
    Recover |delta| from an observed |D| by inverting the folded-normal mean.

    The plug-in |pi_hat_b - pi_hat_b'| estimates E|D|, not |delta|, and E|D| >= |delta|
    with a floor of tau*sqrt(2/pi) at delta = 0. Inverting removes that floor. Where the
    observation sits at or below the floor the answer is 0: the data cannot distinguish
    the two bins' rates at all, and pretending otherwise is exactly the bias we are
    removing. Monotone bisection, vectorised; no scipy.
    """
    observed = np.abs(np.asarray(observed, dtype=float))
    tau = np.asarray(tau, dtype=float)
    lo = np.zeros_like(observed)
    hi = observed + 8.0 * np.maximum(tau, 1e-12)
    floor = tau * SQRT_2_OVER_PI
    for _ in range(n_iter):
        mid = 0.5 * (lo + hi)
        too_small = folded_normal_mean(mid, tau) < observed
        lo = np.where(too_small, mid, lo)
        hi = np.where(too_small, hi, mid)
    out = 0.5 * (lo + hi)
    return np.where(observed <= floor, 0.0, out)


# ==========================================================================
# The positional risk profile
# ==========================================================================

def position_bins(relpos, n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """
    Bin index, bit-identical to s14.positional_scores / s12_rempe.

    Kept as a separate function so the theory and the measurement provably share one
    binning rule; the self-test asserts they agree on random inputs including the
    endpoints 0.0 and 1.0.
    """
    edges = np.linspace(0.0, 1.0, int(n_bins) + 1)
    return np.clip(np.digitize(np.asarray(relpos, dtype=float), edges) - 1,
                   0, int(n_bins) - 1)


def positional_profile(relpos, labels, n_bins: int = DEFAULT_BINS) -> dict:
    """
    (q_b, pi_b, theta) -- everything Proposition 2 needs, from a label file alone.

    Empty bins get q_b = 0 and are assigned pi_b = theta so that nothing downstream has
    to special-case a NaN; they contribute zero weight to every sum by construction.
    """
    y = np.asarray(labels, dtype=float)
    b = position_bins(relpos, n_bins)
    n = np.bincount(b, minlength=n_bins).astype(float)
    npos = np.bincount(b, weights=y, minlength=n_bins)
    total = float(n.sum())
    if total <= 0:
        raise ValueError("empty table: no slices to profile")
    theta = float(npos.sum() / total)
    q = n / total
    pi = np.where(n > 0, npos / np.maximum(n, 1.0), theta)
    return {"n_bins": int(n_bins), "n": n, "n_pos": npos, "q": q, "pi": pi,
            "theta": theta, "n_slices": int(total),
            "n_empty_bins": int((n == 0).sum())}


def gini_mean_difference_half(q, pi) -> float:
    """
    sum_{b < b'} q_b q_b' |pi_b - pi_b'|, in O(B log B).

    Sorting ascending by pi lets the absolute value be dropped:
        sum_j q_j pi_j * (cumulative q below j) - sum_j q_j * (cumulative q*pi below j).
    The self-test checks this against the naive O(B^2) double loop.
    """
    q = np.asarray(q, dtype=float)
    pi = np.asarray(pi, dtype=float)
    order = np.argsort(pi, kind="mergesort")
    qs, ps = q[order], pi[order]
    cum_q = np.concatenate(([0.0], np.cumsum(qs)[:-1]))
    cum_qp = np.concatenate(([0.0], np.cumsum(qs * ps)[:-1]))
    return float(np.sum(qs * ps * cum_q) - np.sum(qs * cum_qp))


def _gmd_half_naive(q, pi) -> float:
    """O(B^2) reference implementation, used only by the self-test."""
    q = np.asarray(q, dtype=float)
    pi = np.asarray(pi, dtype=float)
    return float(sum(q[i] * q[j] * abs(pi[i] - pi[j])
                     for i in range(len(q)) for j in range(i + 1, len(q))))


def positional_ceiling(q, pi) -> dict:
    """
    Proposition 2. The oracle positional AUROC and the concentration statistics.

        auroc  = 1/2 + M / (2 theta (1-theta)),  M = sum_{b<b'} q_b q_b' |pi_b - pi_b'|
               = (1/2)(1 + G / (1 - theta)),     G = Gini coefficient of pi_beta

    Returns NaN for the AUROC when the table is single class, because AUROC is not
    defined there and returning 0.5 would be a lie of convenience.
    """
    q = np.asarray(q, dtype=float)
    pi = np.asarray(pi, dtype=float)
    theta = float(np.sum(q * pi))
    m = gini_mean_difference_half(q, pi)
    gini = float("nan") if theta <= 0 else float(m / theta)
    if not (0.0 < theta < 1.0):
        auroc = float("nan")
    else:
        auroc = 0.5 + m / (2.0 * theta * (1.0 - theta))
    return {"auroc": auroc, "theta": theta, "gmd_half": float(m), "gini": gini,
            "kappa": float("nan") if not np.isfinite(auroc) else float(2 * auroc - 1)}


def ceiling_from_table(relpos, labels, n_bins: int = DEFAULT_BINS) -> dict:
    """The screening statistic: label file in, positional AUROC ceiling out."""
    prof = positional_profile(relpos, labels, n_bins)
    out = positional_ceiling(prof["q"], prof["pi"])
    out.update({"n_bins": prof["n_bins"], "n_slices": prof["n_slices"],
                "n_empty_bins": prof["n_empty_bins"]})
    return out


# ==========================================================================
# Proposition 1: the exact identity
# ==========================================================================

def auc_exact_from_groups(n_pos, n_neg, scores) -> float:
    """
    Proposition 1. Mid-rank AUROC from per-group positive/negative counts and the group
    score, exactly. Groups are bins, or (fold, bin) pairs for pooled out-of-fold scores.

    O(G log G): sort by score, walk the tie classes, credit each class with all the
    negatives strictly below it plus half the negatives tied with it.
    """
    a = np.asarray(n_pos, dtype=float)
    c = np.asarray(n_neg, dtype=float)
    s = np.asarray(scores, dtype=float)
    A, C = float(a.sum()), float(c.sum())
    if A <= 0 or C <= 0:
        return float("nan")
    order = np.argsort(s, kind="mergesort")
    a, c, s = a[order], c[order], s[order]
    starts = np.concatenate(([True], s[1:] != s[:-1]))
    grp = np.cumsum(starts) - 1
    ga = np.bincount(grp, weights=a)
    gc = np.bincount(grp, weights=c)
    below = np.concatenate(([0.0], np.cumsum(gc)[:-1]))
    return float(np.sum(ga * (below + 0.5 * gc)) / (A * C))


def _auc_exact_naive(n_pos, n_neg, scores) -> float:
    """O(G^2) reference implementation, used only by the self-test."""
    a = np.asarray(n_pos, float)
    c = np.asarray(n_neg, float)
    s = np.asarray(scores, float)
    A, C = a.sum(), c.sum()
    if A <= 0 or C <= 0:
        return float("nan")
    tot = 0.0
    for i in range(len(a)):
        for j in range(len(c)):
            k = 1.0 if s[i] > s[j] else (0.5 if s[i] == s[j] else 0.0)
            tot += a[i] * c[j] * k
    return float(tot / (A * C))


def groups_from_scores(labels, scores) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Collapse a score vector into its score-constant groups (Proposition 1c)."""
    y = np.asarray(labels, dtype=int)
    s = s14.snap_ties(scores)
    uniq, inv = np.unique(s, return_inverse=True)
    npos = np.bincount(inv, weights=(y == 1).astype(float), minlength=len(uniq))
    nneg = np.bincount(inv, weights=(y == 0).astype(float), minlength=len(uniq))
    return npos, nneg, uniq


# ==========================================================================
# Finite-sample corrections
# ==========================================================================

def within_volume_icc(volumes, labels, bins=None, n_bins: int = DEFAULT_BINS) -> float:
    """
    One-way ANOVA intraclass correlation of the label within volume, after removing the
    bin mean so that positional structure is not counted as clustering.

    icc = (MSB - MSW) / (MSB + (n0 - 1) MSW), clipped to [0, 1). This is the crude
    estimator on purpose: it feeds a design-effect multiplier inside a normal
    approximation, and a more refined ICC would not survive that approximation.
    """
    y = np.asarray(labels, dtype=float)
    v = pd.factorize(np.asarray(volumes))[0]
    if bins is not None:
        b = np.asarray(bins, dtype=int)
        bm = np.bincount(b, weights=y, minlength=n_bins) / np.maximum(
            np.bincount(b, minlength=n_bins), 1)
        r = y - bm[b]
    else:
        r = y - y.mean()
    k = int(v.max()) + 1 if len(v) else 0
    n = len(r)
    if k < 2 or n <= k:
        return 0.0
    cnt = np.bincount(v).astype(float)
    grp_sum = np.bincount(v, weights=r)
    grp_mean = grp_sum / cnt
    grand = r.mean()
    ssb = float(np.sum(cnt * (grp_mean - grand) ** 2))
    ssw = float(np.sum((r - grp_mean[v]) ** 2))
    msb, msw = ssb / (k - 1), ssw / (n - k)
    if msw <= 0:
        return 0.0
    n0 = (n - float(np.sum(cnt ** 2)) / n) / (k - 1)
    denom = msb + (n0 - 1.0) * msw
    if denom <= 0:
        return 0.0
    return float(min(max((msb - msw) / denom, 0.0), 0.999))


def design_effect_per_bin(bins, volumes, icc: float, n_bins: int = DEFAULT_BINS):
    """deff_b = 1 + (nbar_b - 1) * icc, nbar_b = slices per volume landing in bin b."""
    b = np.asarray(bins, dtype=int)
    v = pd.factorize(np.asarray(volumes))[0]
    n_b = np.bincount(b, minlength=n_bins).astype(float)
    nvol_b = np.array([len(np.unique(v[b == k])) if n_b[k] > 0 else 0
                       for k in range(n_bins)], dtype=float)
    nbar = np.where(nvol_b > 0, n_b / np.maximum(nvol_b, 1.0), 1.0)
    return 1.0 + np.maximum(nbar - 1.0, 0.0) * float(icc)


def order_recovery(pi, m_train, deff=None) -> np.ndarray:
    """
    rho_{bb'} = P(the fitted rates order this pair the way the true rates do), with
    ties counted as one half, under a normal approximation to the fitted bin rates.

    m_train is the expected number of TRAINING slices in each bin, deff the per-bin
    design effect from within-volume label correlation.
    """
    pi = np.asarray(pi, dtype=float)
    m = np.maximum(np.asarray(m_train, dtype=float), 1.0)
    d = np.ones_like(pi) if deff is None else np.maximum(np.asarray(deff, float), 1.0)
    v = pi * (1.0 - pi) * d / m
    tau = np.sqrt(v[:, None] + v[None, :])
    delta = np.abs(pi[:, None] - pi[None, :])
    rho = np.full(tau.shape, 0.5)
    ok = tau > 0
    rho[ok] = _phi_cdf(delta[ok] / tau[ok])
    # tau == 0 means both bins are estimated without error: the order is recovered
    # perfectly when the rates differ, and is a genuine tie when they do not.
    fixed = (~ok) & (delta > 0)
    rho[fixed] = 1.0
    return rho


def predicted_auc(q, pi, m_train, deff=None, rho=None) -> float:
    """Section 4.1: the ceiling with each pair discounted by its order recovery."""
    q = np.asarray(q, dtype=float)
    pi = np.asarray(pi, dtype=float)
    theta = float(np.sum(q * pi))
    if not (0.0 < theta < 1.0):
        return float("nan")
    if rho is None:
        rho = order_recovery(pi, m_train, deff)
    w = np.outer(q, q) * np.abs(pi[:, None] - pi[None, :]) * (2.0 * rho - 1.0)
    return 0.5 + float(np.sum(np.triu(w, 1))) / (2.0 * theta * (1.0 - theta))


def debias_profile_gmd(q, n_per_bin, pi_hat, deff=None) -> float:
    """
    Section 4.2: the folded-normal-debiased sum_{b<b'} q_b q_b' |pi_b - pi_b'|.

    tau_{bb'} is the sampling sd of the difference of the two empirical bin rates,
    inflated by the design effect, and the observed absolute difference is mapped back
    through the folded-normal mean.
    """
    q = np.asarray(q, dtype=float)
    n = np.maximum(np.asarray(n_per_bin, dtype=float), 1.0)
    p = np.asarray(pi_hat, dtype=float)
    d = np.ones_like(p) if deff is None else np.maximum(np.asarray(deff, float), 1.0)
    v = p * (1.0 - p) * d / n
    tau = np.sqrt(v[:, None] + v[None, :])
    obs = np.abs(p[:, None] - p[None, :])
    delta = invert_folded_normal(obs, tau)
    return float(np.sum(np.triu(np.outer(q, q) * delta, 1)))


def debiased_profile(prof: dict, deff=None) -> tuple[np.ndarray, float, float]:
    """
    The risk profile with the section 4.2 selection bias removed.

    The pairwise debias gives a debiased Gini mean difference but not a debiased profile
    vector, and the order-recovery correction of section 4.1 needs a vector. We shrink
    pi towards theta by the factor that reproduces the debiased Gini mean difference
    exactly; because the sum is quadratic in the deviations, that factor is the square
    root of the ratio. Shrinking towards theta is the right direction: it is the profile
    the null would produce, and it preserves the ordering, which by Proposition 1a is
    all the AUROC uses.

    Returns (pi_debiased, gmd_raw, gmd_debiased).
    """
    q, pi, theta = prof["q"], prof["pi"], prof["theta"]
    gmd_raw = gini_mean_difference_half(q, pi)
    gmd_db = debias_profile_gmd(q, prof["n"], pi, deff)
    shrink = math.sqrt(max(gmd_db, 0.0) / gmd_raw) if gmd_raw > 0 else 0.0
    return theta + shrink * (np.asarray(pi, dtype=float) - theta), gmd_raw, gmd_db


def permute_labels_within_volume(volumes, labels, rng) -> np.ndarray:
    """
    Shuffle labels inside each volume: kills the position-label link, keeps prevalence,
    volume depth, subject clustering and each volume's positive count. This is the same
    null s14.calibrate_nulls uses for the positional baseline.
    """
    v = pd.factorize(np.asarray(volumes))[0]
    y = np.asarray(labels)
    key = rng.random(len(y))
    src = np.lexsort((key, v))     # grouped by volume, random inside
    dst = np.argsort(v, kind="stable")  # grouped by volume, original order inside
    out = np.empty_like(y)
    out[dst] = y[src]
    return out


def null_ceiling(relpos, labels, volumes, n_bins: int = DEFAULT_BINS,
                 n_perm: int = 25, seed: int = 0) -> dict:
    """Permutation null of the plug-in ceiling. Its floor is not 0.5; see section 4.2."""
    rng = np.random.default_rng(seed)
    vals = []
    for _ in range(int(n_perm)):
        yp = permute_labels_within_volume(volumes, labels, rng)
        vals.append(ceiling_from_table(relpos, yp, n_bins)["auroc"])
    vals = np.asarray([v for v in vals if np.isfinite(v)], dtype=float)
    if len(vals) == 0:
        return {"median": float("nan"), "p05": float("nan"), "p95": float("nan"),
                "n": 0, "reason": "no non-degenerate permutation draw"}
    return {"median": float(np.median(vals)), "p05": float(np.percentile(vals, 5)),
            "p95": float(np.percentile(vals, 95)), "n": int(len(vals))}


def crossfit_ceiling(relpos, labels, subjects, n_bins: int = DEFAULT_BINS,
                     seed: int = 0) -> float:
    """
    Two-fold subject-level cross-fitted ceiling: ordering from one half, counts from the
    other, combined through Proposition 1 and averaged over the two directions. Honest
    by construction -- no bin's rate is ever ranked against the rows that produced it.
    """
    y = np.asarray(labels, dtype=int)
    b = position_bins(relpos, n_bins)
    subj = pd.Series(np.asarray(subjects).astype(str))
    assign = s14.subject_folds(
        pd.Series(y).groupby(subj).max(), 2, seed)
    fold = subj.map(assign).to_numpy()
    outs = []
    for k in (0, 1):
        tr, te = fold != k, fold == k
        if tr.sum() == 0 or te.sum() == 0:
            continue
        num = np.bincount(b[tr], weights=y[tr].astype(float), minlength=n_bins)
        den = np.bincount(b[tr], minlength=n_bins).astype(float)
        prior = float(y[tr].mean())
        rate = np.where(den > 0, num / np.maximum(den, 1.0), prior)
        a = np.bincount(b[te], weights=(y[te] == 1).astype(float), minlength=n_bins)
        c = np.bincount(b[te], weights=(y[te] == 0).astype(float), minlength=n_bins)
        v = auc_exact_from_groups(a, c, rate)
        if np.isfinite(v):
            outs.append(v)
    return float(np.mean(outs)) if outs else float("nan")


# ==========================================================================
# Proposition 3: patient-level aggregation
# ==========================================================================

def canonical_bins(n_slices: int, n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """Bins occupied by a volume of n consecutively indexed slices (Proposition 3)."""
    n = int(n_slices)
    if n <= 1:
        return np.zeros(1, dtype=int)
    return position_bins(np.arange(n, dtype=float) / (n - 1.0), n_bins)


def canonical_bins_exact(n_slices: int, n_bins: int = DEFAULT_BINS) -> np.ndarray:
    """
    The same thing in exact integer arithmetic: min(floor(B k / (n-1)), B-1).

    Kept separate from canonical_bins because the two DISAGREE at n = B + 1, and that
    disagreement is a real property of the released estimator, not a bug to hide. See
    spans_all_bins.
    """
    n, b = int(n_slices), int(n_bins)
    if n <= 1:
        return np.zeros(1, dtype=int)
    k = np.arange(n, dtype=np.int64)
    return np.minimum((b * k) // (n - 1), b - 1).astype(int)


def spans_all_bins(n_slices: int, n_bins: int = DEFAULT_BINS,
                   exact: bool = False) -> bool:
    """
    Corollary 3.1's precondition, computed rather than assumed.

    In exact arithmetic n >= B + 1 is sufficient: consecutive slices advance the floor's
    argument by B/(n-1) <= 1, so no bin can be skipped. In IEEE double the case
    n = B + 1 is DEGENERATE and the implication can fail, because then every relative
    position k/B lands exactly on a bin edge, np.linspace's edge and the quotient k/B
    differ in the last bit for some k, and np.digitize sends those slices one bin low.
    Measured over B in {5, 10, 20, 50}, n = B + 1 is the only n above B that fails, and
    every n >= B + 2 spans. The paper states the bound as n >= B + 2 for that reason: it
    is the claim the code actually satisfies.
    """
    f = canonical_bins_exact if exact else canonical_bins
    return len(np.unique(f(n_slices, n_bins))) == int(n_bins)


def patient_scores_from_rates(bins, subjects, rate, how: str = "mean") -> tuple:
    """Aggregate slice scores rate[bin] to the subject, exactly as s14 does."""
    s = pd.Series(rate[np.asarray(bins, dtype=int)])
    g = s.groupby(pd.Series(np.asarray(subjects).astype(str)))
    agg = g.max() if how == "max" else g.mean()
    return agg.index.to_numpy(), s14.snap_ties(agg.to_numpy())


def patient_level_analysis(df_bins, subjects, volumes, labels, rates, part=None,
                           slices=None, n_bins: int = DEFAULT_BINS) -> dict:
    """
    Measured patient AUROC of the positional baseline, and the surrogates Proposition 3
    says must reproduce it.

      DEPTH SURROGATE   replace each volume's occupied bins by the canonical bins of a
                        consecutive stack of the same depth, then aggregate. If this
                        reproduces the measurement, DEPTH is the only carrier and
                        position has been annihilated (Corollary 3.2). It is EXACT when
                        relative position is within-volume min-max on a consecutively
                        indexed stack, and only approximate otherwise -- so the fraction
                        of consecutively indexed volumes is reported next to it rather
                        than left for a reviewer to wonder about.

    Corollary 3.1's precondition is checked TWICE and the difference matters:
      * on the ACTUAL bins each volume occupies -- the honest sufficient condition, and
        the one the prediction of 0.500 is issued from;
      * on the DEPTH MODEL's canonical bins -- what the idealised statement assumes.
    They differ exactly when a dataset labels a non-consecutive subset of slices
    (LUNA16's candidate lists, our own brain index), and that is where the idealised
    corollary over-claims.

    `rates` is the list of per-partition fitted rate vectors and `part` says which
    partition each row's score came from, so the measurement here is the SAME pooled
    out-of-fold quantity s14 reports rather than a single fold's. Because the split is
    subject-level, every volume sits entirely in one partition, so the surrogates are
    well defined partition by partition.
    """
    b = np.asarray(df_bins, dtype=int)
    subj = np.asarray(subjects).astype(str)
    vol = np.asarray(volumes).astype(str)
    y = np.asarray(labels, dtype=int)
    rates = np.atleast_2d(np.asarray(rates, dtype=float))
    part = (np.zeros(len(b), dtype=int) if part is None
            else np.asarray(part, dtype=int))
    ylab = pd.Series(y).groupby(pd.Series(subj)).max()
    # s14 snaps the SLICE scores before aggregating and again after; reproduce both
    # steps or the pooled patient AUROC differs from the committed card in the fifth
    # decimal, because a 1e-12 tie broken the other way moves it by 1/(n_pos n_neg).
    slice_scores = s14.snap_ties(rates[part, b])

    def _pat_auc(scores, labels, clusters, how):
        """Bit-identical to s14.evaluate_scores' patient-level path."""
        agg = s04_stats.aggregate_by_cluster(np.asarray(labels, dtype=int),
                                             s14.snap_ties(scores),
                                             np.asarray(clusters, dtype=object),
                                             how=how)
        return float(s04_stats.auc_midrank(agg["labels"],
                                           s14.snap_ties(agg["scores"])))

    out = {}
    for how in ("mean", "max"):
        out[f"measured_patient_auc_{how}"] = _pat_auc(slice_scores, y, subj, how)

    # Corollary 3.1 is a statement about ONE fitted rate vector. Pooling out-of-fold
    # predictions across folds whose fitted maximum differs re-introduces a rankable
    # quantity -- fold identity -- exactly as s14's prevalence_baseline_check warns for
    # the constant predictor. So the corollary is also checked WITHIN each partition,
    # where it has to hold exactly.
    per_part = []
    for k in np.unique(part):
        m = part == k
        if ylab.loc[pd.unique(subj[m])].nunique() < 2:
            continue
        v = _pat_auc(slice_scores[m], y[m], subj[m], "max")
        if np.isfinite(v):
            per_part.append(v)
    out["patient_auc_max_within_partition"] = (
        [float(v) for v in per_part] if per_part else [])
    out["patient_auc_max_within_partition_maxdev"] = (
        float(np.max(np.abs(np.asarray(per_part) - 0.5))) if per_part
        else float("nan"))

    frame = pd.DataFrame({"v": vol, "s": subj, "b": b, "p": part})
    depth = frame.groupby("v")["b"].size()
    first_subject = frame.groupby("v")["s"].first()
    first_part = frame.groupby("v")["p"].first()
    rows_subj, rows_score = [], []
    for v in depth.index:
        cb = canonical_bins(int(depth[v]), n_bins)
        rows_subj.append(np.full(len(cb), first_subject[v], dtype=object))
        rows_score.append(rates[int(first_part[v]), cb])
    ss = np.concatenate(rows_subj) if rows_subj else np.array([], dtype=object)
    sc_all = np.concatenate(rows_score) if rows_score else np.array([], dtype=float)
    y_sur = ylab.loc[ss].to_numpy()
    for how in ("mean", "max"):
        out[f"depth_surrogate_patient_auc_{how}"] = _pat_auc(sc_all, y_sur, ss, how)

    depths = depth.to_numpy()
    argmax_bin = [int(np.argmax(r)) for r in rates]
    actual_sets = frame.groupby("v")["b"].apply(lambda s: set(s.tolist()))
    hits_actual = np.array([argmax_bin[int(first_part[v])] in actual_sets[v]
                            for v in depth.index])
    spans_actual = np.array([len(s) == n_bins for s in actual_sets])
    hits_depth = np.array([argmax_bin[int(first_part[v])]
                           in set(canonical_bins(int(depth[v]), n_bins).tolist())
                           for v in depth.index])

    out["n_volumes"] = int(len(depths))
    out["min_volume_depth"] = int(depths.min()) if len(depths) else 0
    out["median_volume_depth"] = float(np.median(depths)) if len(depths) else 0.0
    out["frac_volumes_spanning_all_bins"] = (
        float(spans_actual.mean()) if len(depths) else 0.0)
    out["frac_volumes_hitting_argmax_bin"] = (
        float(hits_actual.mean()) if len(depths) else 0.0)
    out["frac_volumes_hitting_argmax_bin_depthmodel"] = (
        float(hits_depth.mean()) if len(depths) else 0.0)
    out["annihilation_precondition_holds"] = bool(len(depths) and hits_actual.all())
    out["predicted_patient_auc_max"] = (
        0.5 if out["annihilation_precondition_holds"] else float("nan"))
    if slices is not None:
        z = pd.to_numeric(pd.Series(np.asarray(slices)), errors="coerce")
        cons = pd.DataFrame({"v": vol, "z": z.to_numpy()}).groupby("v")["z"].apply(
            lambda s: bool(len(s) == 1 or np.all(np.diff(np.sort(s.dropna().to_numpy()))
                                                 == 1)))
        out["frac_volumes_consecutive_slice_index"] = float(cons.mean())
    return out


# ==========================================================================
# Simulation
# ==========================================================================

def simulate_cohort(n_subjects: int, n_slices: int, width: float, theta: float,
                    centre: float = 0.55, subject_sd: float = 0.0,
                    rng=None, jitter_depth: int = 0, case_fraction: float = 1.0):
    """
    A synthetic cohort with a controlled positional concentration.

    pi(z) = clip(c * exp(-(z - centre)^2 / (2 width^2)), 0, 1) with c chosen so that the
    mean prevalence is theta. Small width = tightly concentrated label; width >= 3 is
    numerically flat and is the analytic null. subject_sd adds a per-subject offset,
    which induces within-volume label correlation without moving the positional profile
    in expectation -- the lever for the clustering claim.

    case_fraction < 1 makes only that fraction of subjects eligible to carry a positive
    slice, so the PATIENT-level label varies and patient-level AUROC is defined. With
    the default of 1 and a concentrated profile, essentially every subject ends up
    positive and the patient-level metric is degenerate, which is itself a fact about
    slice-level benchmarks worth knowing.
    """
    rng = rng or np.random.default_rng(0)
    subj, vol, sl, rel, y = [], [], [], [], []
    scale = 1.0 / max(float(case_fraction), 1e-12)
    for i in range(n_subjects):
        n = n_slices if jitter_depth <= 0 else int(
            n_slices + rng.integers(-jitter_depth, jitter_depth + 1))
        n = max(n, 3)
        z = np.arange(n, dtype=float)
        r = z / (n - 1.0)
        shape = np.exp(-((r - centre) ** 2) / (2.0 * width ** 2))
        c = theta * scale / max(shape.mean(), 1e-12)
        p = np.clip(c * shape, 0.0, 1.0)
        if subject_sd > 0:
            p = np.clip(p + rng.normal(0.0, subject_sd), 0.0, 1.0)
        if case_fraction < 1.0 and rng.random() >= case_fraction:
            p = np.zeros_like(p)
        y.append((rng.random(n) < p).astype(int))
        subj.append(np.full(n, f"S{i:05d}", dtype=object))
        vol.append(np.full(n, f"S{i:05d}|v0", dtype=object))
        sl.append(z)
        rel.append(r)
    return pd.DataFrame({
        "subject": np.concatenate(subj), "volume": np.concatenate(vol),
        "slice": np.concatenate(sl), "relpos": np.concatenate(rel),
        "label": np.concatenate(y)})


def measure_positional(df: pd.DataFrame, n_bins: int, seed: int, folds: int = 2):
    """
    Run the s14 estimator on a synthetic cohort and return the pooled out-of-fold
    slice AUROC, plus the per-bin training counts the prediction needs.
    """
    y = df["label"].to_numpy()
    b = position_bins(df["relpos"].to_numpy(), n_bins)
    subj = df["subject"].astype(str)
    assign = s14.subject_folds(pd.Series(y).groupby(subj).max(), folds, seed)
    fold = subj.map(assign).to_numpy()
    scores = np.empty(len(y), dtype=float)
    m_train = np.zeros(n_bins, dtype=float)
    for k in range(folds):
        tr, te = fold != k, fold == k
        if tr.sum() == 0 or te.sum() == 0 or len(np.unique(y[tr])) < 2:
            scores[te] = np.nan
            continue
        num = np.bincount(b[tr], weights=y[tr].astype(float), minlength=n_bins)
        den = np.bincount(b[tr], minlength=n_bins).astype(float)
        prior = float(y[tr].mean())
        rate = np.where(den > 0, num / np.maximum(den, 1.0), prior)
        scores[te] = rate[b[te]]
        m_train += den / folds
    ok = np.isfinite(scores)
    return (float(s04_stats.auc_midrank(y[ok], s14.snap_ties(scores[ok]))), m_train)


def run_simulation(reps: int = 30, seed: int = 0, quick: bool = False) -> dict:
    """
    Predicted vs measured across a grid of positional concentration, bin count, cohort
    size, slices per patient and within-patient correlation.

    Reported quantitatively: the mean and maximum absolute discrepancy between the
    prediction and the mean measured AUROC over replicates, plus the replicate spread,
    which is what tells you whether an apparent agreement is meaningful.
    """
    widths = (0.06, 0.12, 0.25, 0.50, 5.0) if not quick else (0.10, 0.30, 5.0)
    bins_grid = (5, 10, 20, 50) if not quick else (10, 20)
    sizes = ((60, 12), (200, 24), (400, 40)) if not quick else ((150, 20),)
    sds = (0.0, 0.20) if not quick else (0.0,)
    theta_grid = (0.15, 0.40) if not quick else (0.25,)
    reps = int(reps if not quick else max(6, reps // 4))

    rows = []
    for width in widths:
        for nb in bins_grid:
            for (ns, nsl) in sizes:
                for sd in sds:
                    for theta in theta_grid:
                        meas, ceil_pl, ceil_xf, pred, nulls = [], [], [], [], []
                        for r in range(reps):
                            rng = np.random.default_rng(
                                hash((width, nb, ns, nsl, sd, theta, r, seed))
                                % (2 ** 32))
                            df = simulate_cohort(ns, nsl, width, theta,
                                                 subject_sd=sd, rng=rng)
                            if df["label"].nunique() < 2:
                                continue
                            m, m_train = measure_positional(df, nb, seed=r)
                            if not np.isfinite(m):
                                continue
                            rel = df["relpos"].to_numpy()
                            lab = df["label"].to_numpy()
                            prof = positional_profile(rel, lab, nb)
                            b = position_bins(rel, nb)
                            icc = within_volume_icc(df["volume"], lab, b, nb)
                            deff = design_effect_per_bin(b, df["volume"], icc, nb)
                            cp = positional_ceiling(prof["q"], prof["pi"])["auroc"]
                            pi_db, _, _ = debiased_profile(prof, deff)
                            pr = predicted_auc(prof["q"], pi_db, m_train, deff)
                            meas.append(m)
                            ceil_pl.append(cp)
                            pred.append(pr)
                            ceil_xf.append(crossfit_ceiling(
                                rel, lab, df["subject"], nb, seed=r))
                            if r < 3:
                                nulls.append(null_ceiling(rel, lab, df["volume"],
                                                          nb, n_perm=5, seed=r
                                                          )["median"])
                        if not meas:
                            continue
                        rows.append({
                            "width": width, "n_bins": nb, "n_subjects": ns,
                            "slices_per_subject": nsl, "subject_sd": sd,
                            "theta": theta, "reps": len(meas),
                            "measured_mean": float(np.mean(meas)),
                            "measured_sd": float(np.std(meas, ddof=1))
                            if len(meas) > 1 else 0.0,
                            "ceiling_plugin": float(np.mean(ceil_pl)),
                            "ceiling_xfit": float(np.nanmean(ceil_xf)),
                            "predicted": float(np.mean(pred)),
                            "null_ceiling": float(np.nanmean(nulls)) if nulls else float("nan"),
                        })
    d = pd.DataFrame(rows)
    if len(d):
        d["err_pred"] = d["predicted"] - d["measured_mean"]
        d["err_ceiling"] = d["ceiling_plugin"] - d["measured_mean"]
        d["err_xfit"] = d["ceiling_xfit"] - d["measured_mean"]
    return {"grid": d, "summary": _sim_summary(d)}


def _sim_summary(d: pd.DataFrame) -> dict:
    if not len(d):
        return {}
    out = {"n_cells": int(len(d))}
    for k in ("pred", "ceiling", "xfit"):
        e = d[f"err_{k}"].to_numpy()
        out[k] = {"mae": float(np.mean(np.abs(e))), "bias": float(np.mean(e)),
                  "max_abs": float(np.max(np.abs(e))),
                  "p95_abs": float(np.percentile(np.abs(e), 95))}
    m, p = d["measured_mean"].to_numpy(), d["predicted"].to_numpy()
    ss_res = float(np.sum((m - p) ** 2))
    ss_tot = float(np.sum((m - m.mean()) ** 2))
    out["r2_predicted_vs_measured"] = float(1 - ss_res / ss_tot) if ss_tot > 0 else float("nan")
    flat = d[d["width"] >= 3.0]
    if len(flat):
        out["flat_profile_cells"] = {
            "n": int(len(flat)),
            "measured_mean": float(flat["measured_mean"].mean()),
            "ceiling_plugin_mean": float(flat["ceiling_plugin"].mean()),
            "predicted_mean": float(flat["predicted"].mean()),
            "note": "width >= 3 is a numerically flat profile: the analytic answer is "
                    "0.500 and any excess in ceiling_plugin is the section 4.2 bias",
        }
    return out


# ==========================================================================
# Real data
# ==========================================================================
# Every target below is a label table that already sits on disk and that s14 has already
# audited, so predicted-vs-measured is a comparison against a number in a committed
# JSON, not against a number this module invented.

SCRATCH = Path("/private/tmp/claude-501/-Users-sathvikloke-Downloads-PhaseDx/"
               "20765690-1294-4135-b633-a1e4b2778ed4/scratchpad/audit/labels")

_DL = dict(label="lesion_type", volume="pseudo_volume", relpos_col="norm_z")
_RSNA = dict(subject="patient_id", volume="series_id", slice_col="slice",
             metadata=["plane", "rescale_slope", "rescale_intercept"])


def _targets() -> list[dict]:
    """(name, path, s14 column arguments, s14 payload to check against)."""
    t: list[dict] = []
    # --- published fastMRI prostate label files (the paper's anchor) --------
    for arm in ("dwi", "t2"):
        t.append({"name": f"fastmri_prostate_{arm}_published",
                  "path": SCRATCH / f"{arm}_slice_level_labels.csv",
                  "kw": dict(label="PIRADS", positive_if=">2"),
                  "payload": f"fastmri_prostate_{arm}_published"})
    # --- our five cohort indices -------------------------------------------
    for c in ("brain", "breast", "knee", "prostate_dwi", "prostate_t2"):
        t.append({"name": f"phasedx_{c}",
                  "path": REPO / "pipeline_out" / "cache" / f"{c}_index.csv",
                  "kw": {}, "payload": f"phasedx_{c}"})
    # --- DeepLesion, eight one-vs-rest arms, published normalised z ---------
    for code, organ in ((1, "bone"), (2, "abdomen"), (3, "mediastinum"), (4, "liver"),
                        (5, "lung"), (6, "kidney"), (7, "softtissue"), (8, "pelvis")):
        t.append({"name": f"deeplesion_{organ}_vs_rest",
                  "path": SCRATCH / "deeplesion_tidy.csv",
                  "kw": dict(positive_if=f"=={code}", **_DL),
                  "payload": f"deeplesion_{organ}_vs_rest"})
    # --- the two NOT-MATCHED benchmarks, reported as prominently as the hits -
    t.append({"name": "luna16_fp_reduction_candidates",
              "path": SCRATCH / "luna16_tidy.csv",
              "kw": dict(subject="subject_id", volume="series_id", slice_col="slice",
                         label="label",
                         exclude=("coordX", "coordY", "z_mm", "n_candidates_in_scan")),
              "payload": "luna16_fp_reduction_candidates"})
    t.append({"name": "picai_case_level", "path": SCRATCH / "picai_case_level.csv",
              "kw": dict(subject="patient_id", volume="", label="label"),
              "payload": "picai_case_level"})
    # --- fastMRI+ knee and Duke breast --------------------------------------
    t.append({"name": "fastmriplus_knee_meniscus_tear",
              "path": SCRATCH / "fastmriplus_knee_meniscus.csv",
              "kw": dict(subject="subject_id", volume="volume_id", label="label",
                         exclude=("any_finding",)),
              "payload": "fastmriplus_knee_meniscus_tear"})
    t.append({"name": "fastmriplus_knee_any_finding",
              "path": SCRATCH / "fastmriplus_knee_meniscus.csv",
              "kw": dict(subject="subject_id", volume="volume_id", label="any_finding",
                         exclude=("label",)),
              "payload": "fastmriplus_knee_any_finding"})
    t.append({"name": "duke_breast_owner_slice_task",
              "path": SCRATCH / "duke_breast_slices.csv",
              "kw": dict(subject="patient_id", volume="", label="label"),
              "payload": "duke_breast_owner_slice_task"})
    # --- RSNA ICH: the any arm s14 audited, plus the five official subtypes --
    sub = REPO / "pipeline_out" / "audit_data" / "rsna_ich_slices_sub1500.csv"
    t.append({"name": "rsna_ich_any_slice", "path": sub,
              "kw": dict(label="label", **_RSNA), "payload": "rsna_ich_any_slice"})
    for s in ("epidural", "intraparenchymal", "intraventricular", "subarachnoid",
              "subdural"):
        t.append({"name": f"rsna_ich_{s}", "path": sub,
                  "kw": dict(label=s, **_RSNA), "payload": None})
    return t


def prepare_frame(path: Path, positive_if=None, relpos_col=None, **colargs):
    """
    Rebuild exactly the frame s14.audit builds, up to the point of splitting.

    This duplicates s14.audit's preparation block rather than refactoring s14, because
    s14 is the released artefact with 53 self-tests and a Zenodo DOI: it is not touched
    by this module. The duplication is checked -- validate_target asserts the resulting
    row count and prevalence against the committed s14 payload.
    """
    raw = s14.load_table(Path(path))
    if relpos_col is not None:
        colargs = dict(colargs)
        colargs["exclude"] = tuple(colargs.get("exclude") or ()) + (relpos_col,)
    colmap = s14.resolve_columns(raw, **colargs)
    if relpos_col is not None and relpos_col in colmap.metadata:
        colmap.metadata.remove(relpos_col)
    df = raw.copy()
    y, rule = s14.binarise(df[colmap.label], positive_if)
    df[s14.C_LABEL] = y
    df[s14.C_SUBJ] = df[colmap.subject].astype(str)
    df[s14.C_SLICE] = pd.to_numeric(df[colmap.slice], errors="coerce")
    df = df[df[s14.C_SLICE].notna()].copy()
    df[s14.C_VOL] = (df[s14.C_SUBJ] if colmap.volume is None
                     else df[s14.C_SUBJ] + "|" + df[colmap.volume].astype(str))
    if relpos_col is None:
        df = s14.add_relative_position(df, s14.C_VOL, s14.C_SLICE)
    else:
        r = pd.to_numeric(df[relpos_col], errors="coerce")
        df = df[r.notna()].copy()
        r = r[r.notna()]
        lo, hi = float(r.min()), float(r.max())
        df[s14.C_RELPOS] = ((r - lo) / (hi - lo if hi > lo else 1.0)
                            if (lo < 0.0 or hi > 1.0) else r)
    df = s14.add_volume_size(df, s14.C_VOL)
    return df.reset_index(drop=True), colmap, rule


def build_partitions(df: pd.DataFrame, colmap, cv_folds: int = 5, seed: int = 0,
                     val_as: str = "exclude"):
    """s14.audit's split rule: the dataset's own split when it has one, else 5-fold
    subject-level CV. Returns (parts, kind)."""
    if colmap.split is not None:
        canon, _ = s14.normalise_split_values(df[colmap.split])
        if val_as == "train":
            canon = canon.replace("val", "train")
        elif val_as == "test":
            canon = canon.replace("val", "test")
        tr = df.index[canon == "train"]
        te = df.index[canon == "test"]
        if len(tr) and len(te):
            return [(tr, te)], "official_split"
    assign = s14.subject_folds(df.groupby(s14.C_SUBJ)[s14.C_LABEL].max(),
                               cv_folds, seed)
    fold = df[s14.C_SUBJ].map(assign)
    return [(df.index[fold != k], df.index[fold == k])
            for k in range(cv_folds)], "subject_cv"


def replay_positional(df, parts, n_bins: int = DEFAULT_BINS):
    """
    Re-run s14's positional baseline over the same partitions, collecting everything
    the theory needs: the pooled score vector, the per-partition fitted rate vectors and
    the per-bin training counts.
    """
    b_all = position_bins(df[s14.C_RELPOS].to_numpy(), n_bins)
    y_all = df[s14.C_LABEL].to_numpy().astype(int)
    ys, ss, sj, bs, rates, m_train, n_used = [], [], [], [], [], np.zeros(n_bins), 0
    vl, pt, rows = [], [], []
    for tr_idx, te_idx in parts:
        tr = df.index.get_indexer(tr_idx)
        te = df.index.get_indexer(te_idx)
        if len(tr) == 0 or len(te) == 0:
            continue
        if len(np.unique(y_all[tr])) < 2 or len(np.unique(y_all[te])) < 2:
            continue
        num = np.bincount(b_all[tr], weights=y_all[tr].astype(float), minlength=n_bins)
        den = np.bincount(b_all[tr], minlength=n_bins).astype(float)
        prior = float(y_all[tr].mean())
        rate = np.where(den > 0, num / np.maximum(den, 1.0), prior)
        ys.append(y_all[te])
        ss.append(rate[b_all[te]])
        sj.append(df[s14.C_SUBJ].to_numpy()[te])
        vl.append(df[s14.C_VOL].to_numpy()[te])
        bs.append(b_all[te])
        pt.append(np.full(len(te), n_used, dtype=int))
        rows.append(te)
        rates.append(rate)
        m_train += den
        n_used += 1
    if not ys:
        return None
    return {"y": np.concatenate(ys), "score": np.concatenate(ss),
            "subject": np.concatenate(sj), "volume": np.concatenate(vl),
            "bin": np.concatenate(bs), "part": np.concatenate(pt),
            "test_rows": np.concatenate(rows),
            "rates": np.array(rates, dtype=float),
            "m_train": m_train / max(n_used, 1), "n_partitions": n_used}


def validate_target(t: dict, n_bins: int = DEFAULT_BINS, n_perm: int = 15,
                    seed: int = 0) -> dict:
    """Predicted vs measured for one label table."""
    path = Path(t["path"])
    if not path.exists():
        return {"dataset": t["name"], "status": "missing", "path": str(path)}
    kw = dict(t["kw"])
    positive_if = kw.pop("positive_if", None)
    relpos_col = kw.pop("relpos_col", None)
    df, colmap, rule = prepare_frame(path, positive_if=positive_if,
                                     relpos_col=relpos_col, **kw)
    if df[s14.C_LABEL].nunique() < 2:
        return {"dataset": t["name"], "status": "single_class"}

    rel = df[s14.C_RELPOS].to_numpy()
    y = df[s14.C_LABEL].to_numpy().astype(int)
    vol = df[s14.C_VOL].to_numpy()
    b = position_bins(rel, n_bins)

    # ---- screening statistics: label file only, no split, no model --------
    prof = positional_profile(rel, y, n_bins)
    ceil = positional_ceiling(prof["q"], prof["pi"])
    icc = within_volume_icc(vol, y, b, n_bins)
    deff = design_effect_per_bin(b, vol, icc, n_bins)

    # ---- measurement: replay s14's estimator on s14's partitions ----------
    parts, kind = build_partitions(df, colmap, seed=seed)
    rp = replay_positional(df, parts, n_bins)
    if rp is None:
        return {"dataset": t["name"], "status": "no_evaluable_partition"}
    measured = float(s04_stats.auc_midrank(rp["y"], s14.snap_ties(rp["score"])))

    # ---- Proposition 1: the exact identity, to machine precision ----------
    npos, nneg, sc = groups_from_scores(rp["y"], rp["score"])
    exact = auc_exact_from_groups(npos, nneg, sc)

    # ---- the prediction: both corrections, at the real training size ------
    pi_db, gmd_raw, gmd_db = debiased_profile(prof, deff)
    predicted = predicted_auc(prof["q"], pi_db, rp["m_train"], deff)
    ceiling_db = positional_ceiling(prof["q"], pi_db)["auroc"]

    xfit = crossfit_ceiling(rel, y, df[s14.C_SUBJ], n_bins, seed=seed)
    null = null_ceiling(rel, y, vol, n_bins, n_perm=n_perm, seed=seed)

    # ---- Proposition 3, on the SAME pooled out-of-fold rows s14 reports ----
    pat = patient_level_analysis(
        rp["bin"], rp["subject"], rp["volume"], rp["y"], rp["rates"],
        part=rp["part"], slices=df[s14.C_SLICE].to_numpy()[rp["test_rows"]],
        n_bins=n_bins)

    # ---- how much of the residual is train/test profile shift? ------------
    prof_te = positional_profile(rel[rp["test_rows"]], rp["y"], n_bins)
    ceil_te = positional_ceiling(prof_te["q"], prof_te["pi"])["auroc"]

    out = {
        "dataset": t["name"], "status": "ok", "labels_file": str(path),
        "label_rule": rule, "protocol": kind, "n_partitions": rp["n_partitions"],
        "n_rows": int(len(df)), "n_subjects": int(df[s14.C_SUBJ].nunique()),
        "n_volumes": int(df[s14.C_VOL].nunique()),
        "relpos_source": "published column" if relpos_col else "within-volume min-max",
        "n_bins": n_bins,
        "slice_prevalence": float(prof["theta"]),
        "gini_of_profile": float(ceil["gini"]),
        "gmd_half_raw": float(gmd_raw),
        "gmd_half_debiased": float(gmd_db),
        "kappa": float(ceil["kappa"]),
        "icc_within_volume": float(icc),
        "mean_deff": float(np.mean(deff)),
        "n_test_slices": int(len(rp["y"])),
        "n_test_pos_slices": int(rp["y"].sum()),
        "n_test_subjects": int(pd.unique(rp["subject"]).size),
        "ceiling_plugin": float(ceil["auroc"]),
        "ceiling_debiased": float(ceiling_db),
        "ceiling_test_rows_only": float(ceil_te),
        "err_ceiling_test_rows_only": float(ceil_te - measured),
        "ceiling_xfit": float(xfit),
        "null_ceiling_median": float(null["median"]),
        "predicted": float(predicted),
        "measured": float(measured),
        "exact_identity": float(exact),
        "exact_identity_gap": float(abs(exact - measured)),
        "err_predicted": float(predicted - measured),
        "err_ceiling": float(ceil["auroc"] - measured),
        "err_xfit": float(xfit - measured),
    }
    out.update(pat)
    out["err_patient_depth_surrogate_mean"] = float(
        pat["depth_surrogate_patient_auc_mean"] - pat["measured_patient_auc_mean"])
    # cross-check against the committed s14 card where one exists
    if t.get("payload"):
        p = REPO / "pipeline_out" / "trivial_baselines" / f"{t['payload']}.json"
        if p.exists():
            pay = json.loads(p.read_text())
            ev = pay["evaluations"][pay["headline_evaluation"]]
            key = f"positional_{n_bins}bin"
            if key in ev["baselines"]:
                bl = ev["baselines"][key]
                out["s14_card_slice_auc"] = float(bl["slice_auc"])
                out["s14_card_gap"] = float(abs(bl["slice_auc"] - measured))
                out["s14_card_patient_auc"] = float(bl["patient_auc"])
                out["s14_card_patient_auc_maxagg"] = float(bl["patient_auc_maxagg"])
            out["s14_card_n_rows"] = int(pay["n_rows"])
            out["s14_card_prevalence"] = float(pay["slice_prevalence"])
    return out


def run_real(n_bins: int = DEFAULT_BINS, n_perm: int = 15, seed: int = 0,
             only=None) -> pd.DataFrame:
    rows = []
    for t in _targets():
        if only and t["name"] not in only:
            continue
        try:
            rows.append(validate_target(t, n_bins=n_bins, n_perm=n_perm, seed=seed))
        except Exception as exc:                      # noqa: BLE001
            rows.append({"dataset": t["name"], "status": f"error: {type(exc).__name__}: {exc}"})
        print(f"  ... {rows[-1]['dataset']:<36s} {rows[-1].get('status')}", flush=True)
    return pd.DataFrame(rows)


# ==========================================================================
# RSNA ICH under Burduja et al.'s own split geometry
# ==========================================================================

RSNA_PUBLISHED = {  # Burduja, Ionescu & Verga, Sensors 2020;20(19):5611, Table 3
    "label": ("any", 0.9843), "epidural": ("epidural", 0.9851),
    "intraparenchymal": ("intraparenchymal", 0.9927),
    "intraventricular": ("intraventricular", 0.9970),
    "subarachnoid": ("subarachnoid", 0.9821), "subdural": ("subdural", 0.9682),
}


def run_rsna_burduja(path: Path | None = None, n_bins: int = DEFAULT_BINS,
                     n_rep: int = 100, n_held_out: int = 744) -> pd.DataFrame:
    """
    The paper's only peer-reviewed slice-level comparator, predicted rather than only
    measured.

    This is the sharpest available test of the theory on real data, because the measured
    quantity is itself a Monte-Carlo average over n_rep random draws of the held-out
    split (Burduja et al. publish the split's geometry, not the draw). Averaging removes
    split-to-split noise, so what is left to explain is exactly the expectation the
    theory predicts.
    """
    path = Path(path or REPO / "pipeline_out" / "audit_data" / "rsna_ich_slices.csv")
    if not path.exists():
        return pd.DataFrame([{"status": "missing", "path": str(path)}])
    d = pd.read_csv(path, usecols=["patient_id", "series_id", "slice"]
                    + list(RSNA_PUBLISHED))
    g = d.groupby("series_id")["slice"]
    lo, hi = g.transform("min").to_numpy(), g.transform("max").to_numpy()
    span = np.where(hi > lo, hi - lo, 1.0)
    relpos = (d["slice"].to_numpy() - lo) / span
    b = position_bins(relpos, n_bins)
    series = d["series_id"].to_numpy()
    keys = np.unique(series)
    rows = []
    for col, (name, published) in RSNA_PUBLISHED.items():
        y = d[col].to_numpy().astype(float)
        prof = positional_profile(relpos, y, n_bins)
        ceil = positional_ceiling(prof["q"], prof["pi"])
        icc = within_volume_icc(series, y, b, n_bins)
        deff = design_effect_per_bin(b, series, icc, n_bins)
        frac_train = 1.0 - n_held_out / len(keys)
        m_train = prof["n"] * frac_train
        pi_db, _, _ = debiased_profile(prof, deff)
        predicted = predicted_auc(prof["q"], pi_db, m_train, deff)
        aucs = []
        for rep in range(int(n_rep)):
            rng = np.random.default_rng(rep)
            held = set(rng.permutation(keys)[:n_held_out].tolist())
            is_test = np.fromiter((k in held for k in series), bool, len(series))
            tr = ~is_test
            num = np.bincount(b[tr], weights=y[tr], minlength=n_bins)
            den = np.bincount(b[tr], minlength=n_bins).astype(float)
            rate = np.where(den > 0, num / np.maximum(den, 1.0), y[tr].mean())
            a = np.bincount(b[is_test], weights=(y[is_test] == 1).astype(float),
                            minlength=n_bins)
            c = np.bincount(b[is_test], weights=(y[is_test] == 0).astype(float),
                            minlength=n_bins)
            aucs.append(auc_exact_from_groups(a, c, rate))
        aucs = np.asarray(aucs, dtype=float)
        rows.append({
            "label": name, "published": published,
            "prevalence": float(prof["theta"]), "gini": float(ceil["gini"]),
            "icc_within_series": float(icc),
            "ceiling_plugin": float(ceil["auroc"]),
            "predicted": float(predicted),
            "measured_mean": float(np.nanmean(aucs)),
            "measured_sd": float(np.nanstd(aucs, ddof=1)),
            "err_predicted": float(predicted - np.nanmean(aucs)),
            "err_ceiling": float(ceil["auroc"] - np.nanmean(aucs)),
            "trivial_fraction": float((np.nanmean(aucs) - 0.5) / (published - 0.5)),
            "predicted_trivial_fraction": float((predicted - 0.5) / (published - 0.5)),
            "n_rep": int(n_rep),
        })
        print(f"  ... rsna {name:<20s} pred {rows[-1]['predicted']:.4f} "
              f"meas {rows[-1]['measured_mean']:.4f}", flush=True)
    return pd.DataFrame(rows)


# ==========================================================================
# Self-test: cases whose answer is known analytically
# ==========================================================================

def _check(ok: bool, msg: str, fails: list):
    print(f"  [{'PASS' if ok else 'FAIL'}] {msg}")
    if not ok:
        fails.append(msg)


def self_test(quick: bool = False) -> int:
    fails: list[str] = []
    rng = np.random.default_rng(11)
    print("s15_positional_theory self-test")
    print("=" * 72)

    print("\n-- binning agrees with the estimator under analysis")
    r = np.concatenate([rng.random(5000), [0.0, 1.0, 0.5, 1.0 - 1e-15]])
    for nb in (5, 10, 20, 50):
        train = pd.DataFrame({"relpos": r, "label": (rng.random(len(r)) < 0.3).astype(int)})
        edges = np.linspace(0, 1, nb + 1)
        ref = np.clip(np.digitize(r, edges) - 1, 0, nb - 1)
        _check(np.array_equal(position_bins(r, nb), ref),
               f"position_bins matches s14's digitize rule at B={nb}", fails)
        # and the scores themselves agree with s14.positional_scores
        sc_ref = s14.positional_scores(train, train, n_bins=nb)
        prof = positional_profile(train["relpos"], train["label"], nb)
        _check(np.allclose(sc_ref, prof["pi"][position_bins(r, nb)], atol=1e-12),
               f"per-bin rates reproduce s14.positional_scores at B={nb}", fails)

    print("\n-- ANALYTIC CASE 1: label independent of position must give 0.500")
    for nb in (5, 20, 50):
        q = rng.random(nb); q /= q.sum()
        pi = np.full(nb, 0.27)
        a = positional_ceiling(q, pi)
        _check(abs(a["auroc"] - 0.5) < 1e-12,
               f"constant risk profile, B={nb}: A = {a['auroc']:.15f}", fails)
        _check(abs(a["gini"]) < 1e-12, f"  its Gini coefficient is 0 (B={nb})", fails)

    print("\n-- ANALYTIC CASE 2: perfect positional separation must give 1.000")
    for nb in (5, 20, 50):
        q = rng.random(nb); q /= q.sum()
        pi = (np.arange(nb) >= nb // 3).astype(float)
        a = positional_ceiling(q, pi)
        _check(abs(a["auroc"] - 1.0) < 1e-12,
               f"two-valued risk profile, B={nb}: A = {a['auroc']:.15f}", fails)
        _check(abs(a["gini"] - (1 - a["theta"])) < 1e-12,
               f"  and G = 1 - theta exactly (B={nb})", fails)

    print("\n-- the closed form equals the double sum it was derived from")
    for _ in range(8):
        nb = int(rng.integers(3, 40))
        q = rng.random(nb); q /= q.sum()
        pi = rng.random(nb)
        theta = float(np.sum(q * pi))
        brute = 0.0
        for i in range(nb):
            for j in range(nb):
                k = 1.0 if pi[i] > pi[j] else (0.5 if pi[i] == pi[j] else 0.0)
                brute += q[i] * q[j] * pi[i] * (1 - pi[j]) * k
        brute /= theta * (1 - theta)
        _check(abs(brute - positional_ceiling(q, pi)["auroc"]) < 1e-12,
               f"B={nb}: Gini form == oracle double sum ({brute:.12f})", fails)
        _check(abs(gini_mean_difference_half(q, pi) - _gmd_half_naive(q, pi)) < 1e-14,
               f"B={nb}: O(B log B) Gini mean difference == O(B^2) reference", fails)

    print("\n-- PROPOSITION 1: the bilinear form is exact against mid-rank AUROC")
    for _ in range(12):
        n = int(rng.integers(50, 4000))
        nb = int(rng.integers(2, 30))
        b = rng.integers(0, nb, n)
        y = (rng.random(n) < 0.25).astype(int)
        rate = np.round(rng.random(nb), int(rng.integers(1, 4)))  # forces ties
        sc = rate[b]
        ref = s04_stats.auc_midrank(y, sc)
        npos = np.bincount(b, weights=(y == 1).astype(float), minlength=nb)
        nneg = np.bincount(b, weights=(y == 0).astype(float), minlength=nb)
        got = auc_exact_from_groups(npos, nneg, rate)
        naive = _auc_exact_naive(npos, nneg, rate)
        if not np.isfinite(ref):
            continue
        _check(abs(got - ref) < 1e-12, f"n={n} B={nb}: exact form == auc_midrank "
                                       f"({got:.15f} vs {ref:.15f})", fails)
        _check(abs(got - naive) < 1e-12, "  and == the O(G^2) reference", fails)

    print("\n-- PROPOSITION 1c: pooled out-of-fold scores collapse to their groups")
    n, nb = 3000, 12
    b = rng.integers(0, nb, n)
    fold = rng.integers(0, 5, n)
    y = (rng.random(n) < 0.3).astype(int)
    sc = rng.random(nb * 5)[fold * nb + b]
    npos, nneg, uniq = groups_from_scores(y, sc)
    _check(abs(auc_exact_from_groups(npos, nneg, uniq)
               - s04_stats.auc_midrank(y, s14.snap_ties(sc))) < 1e-12,
           "pooled 5-fold score vector: grouped form == auc_midrank", fails)

    print("\n-- PROPOSITION 3: annihilation under within-volume normalisation")
    for nb in (5, 10, 20, 50):
        _check(all(spans_all_bins(n, nb, exact=True)
                   for n in range(nb + 1, 6 * nb)),
               f"B={nb}: EXACT arithmetic, every n >= B+1 occupies every bin", fails)
        _check(all(spans_all_bins(n, nb) for n in range(nb + 2, 6 * nb)),
               f"B={nb}: IEEE double, every n >= B+2 occupies every bin", fails)
        _check(not spans_all_bins(nb + 1, nb),
               f"B={nb}: n = B+1 is the documented float knife-edge and does NOT span",
               fails)
    # the operational consequence, measured rather than asserted
    for nb in (10, 20):
        df = simulate_cohort(400, nb + 12, width=0.08, theta=0.10, case_fraction=0.35,
                             rng=np.random.default_rng(3))
        bb = position_bins(df["relpos"].to_numpy(), nb)
        prof = positional_profile(df["relpos"], df["label"], nb)
        ylab_all = df.groupby("subject")["label"].max()
        slice_a = positional_ceiling(prof["q"], prof["pi"])["auroc"]
        _check(ylab_all.nunique() == 2,
               f"B={nb}: the test cohort has both patient classes "
               f"({int(ylab_all.sum())}/{len(ylab_all)} positive)", fails)
        for how in ("max", "mean"):
            ids, sc = patient_scores_from_rates(bb, df["subject"], prof["pi"], how=how)
            a = s04_stats.auc_midrank(ylab_all.loc[ids].to_numpy(), sc)
            _check(abs(a - 0.5) < 1e-12,
                   f"B={nb}: slice ceiling {slice_a:.3f} -> {how}-agg patient AUROC "
                   f"{a:.15f} (exactly 0.5, every volume spans)", fails)

    print("\n-- folded-normal inversion round-trips")
    tau = np.array([0.01, 0.05, 0.1, 0.2])
    for delta in (0.0, 0.02, 0.1, 0.5):
        obs = folded_normal_mean(np.full(4, delta), tau)
        got = invert_folded_normal(obs, tau)
        _check(np.all(np.abs(got - delta) < 2e-3),
               f"delta={delta}: inverted {np.round(got, 4).tolist()}", fails)

    print("\n-- the plug-in ceiling's null is above 0.5, as section 4.2 predicts")
    df = simulate_cohort(150, 24, width=5.0, theta=0.25, rng=np.random.default_rng(7))
    c = ceiling_from_table(df["relpos"], df["label"], 20)["auroc"]
    nl = null_ceiling(df["relpos"], df["label"], df["volume"], 20, n_perm=20, seed=1)
    _check(c > 0.5 and nl["median"] > 0.5,
           f"flat synthetic cohort: plug-in ceiling {c:.4f}, permutation null median "
           f"{nl['median']:.4f}; neither is 0.5 and the difference is what counts", fails)
    xf = crossfit_ceiling(df["relpos"], df["label"], df["subject"], 20, seed=1)
    _check(abs(xf - 0.5) < 0.05,
           f"  cross-fitted ceiling on the same null cohort: {xf:.4f} (near 0.5)", fails)

    print("\n-- end-to-end: predicted tracks measured on a concentrated cohort")
    df = simulate_cohort(300, 28, width=0.10, theta=0.2, rng=np.random.default_rng(5))
    m, m_train = measure_positional(df, 20, seed=0)
    prof = positional_profile(df["relpos"], df["label"], 20)
    b = position_bins(df["relpos"].to_numpy(), 20)
    icc = within_volume_icc(df["volume"], df["label"], b, 20)
    deff = design_effect_per_bin(b, df["volume"], icc, 20)
    pr = predicted_auc(prof["q"], prof["pi"], m_train, deff)
    _check(abs(pr - m) < 0.03,
           f"predicted {pr:.4f} vs measured {m:.4f} (|diff| {abs(pr - m):.4f} < 0.03)",
           fails)

    print("\n-- monotonicity under bin refinement (property 2b), in the population")
    df = simulate_cohort(4000, 60, width=0.12, theta=0.25, rng=np.random.default_rng(9))
    vals = [ceiling_from_table(df["relpos"], df["label"], nb)["auroc"]
            for nb in (5, 10, 20, 40)]
    _check(all(vals[i] <= vals[i + 1] + 5e-3 for i in range(len(vals) - 1)),
           f"ceiling non-decreasing in B on a large cohort: "
           f"{[round(v, 4) for v in vals]}", fails)

    print("\n" + "=" * 72)
    if fails:
        print(f"{len(fails)} CHECK(S) FAILED")
        for f in fails:
            print("  - " + f)
        return 1
    print("all checks passed")
    return 0


# ==========================================================================
# Reporting
# ==========================================================================

def _fmt(v, nd=4):
    if v is None:
        return "-"
    try:
        f = float(v)
    except (TypeError, ValueError):
        return str(v)
    return "nan" if not np.isfinite(f) else f"{f:.{nd}f}"


def print_real_table(d: pd.DataFrame) -> None:
    ok = d[d["status"] == "ok"] if "status" in d else d
    if not len(ok):
        print("no rows")
        return
    hdr = (f"{'dataset':<34}{'n_rows':>9}{'prev':>8}{'Gini':>8}{'ceiling':>9}"
           f"{'null':>8}{'xfit':>8}{'PRED':>8}{'MEAS':>8}{'err':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in ok.iterrows():
        print(f"{r['dataset']:<34}{int(r['n_rows']):>9,}"
              f"{_fmt(r['slice_prevalence'], 3):>8}{_fmt(r['gini_of_profile'], 3):>8}"
              f"{_fmt(r['ceiling_plugin'], 3):>9}{_fmt(r['null_ceiling_median'], 3):>8}"
              f"{_fmt(r['ceiling_xfit'], 3):>8}{_fmt(r['predicted'], 3):>8}"
              f"{_fmt(r['measured'], 3):>8}{_fmt(r['err_predicted'], 3):>8}")
    e = ok["err_predicted"].to_numpy(dtype=float)
    e = e[np.isfinite(e)]
    print("-" * len(hdr))
    print(f"prediction error: mean {np.mean(e):+.4f}  MAE {np.mean(np.abs(e)):.4f}  "
          f"max |err| {np.max(np.abs(e)):.4f}  over {len(e)} benchmarks")
    ec = ok["err_ceiling"].to_numpy(dtype=float)
    ec = ec[np.isfinite(ec)]
    print(f"uncorrected ceiling error: mean {np.mean(ec):+.4f}  "
          f"MAE {np.mean(np.abs(ec)):.4f}  max {np.max(np.abs(ec)):.4f}")
    g = ok["exact_identity_gap"].to_numpy(dtype=float)
    g = g[np.isfinite(g)]
    if len(g):
        print(f"Proposition 1 identity gap vs auc_midrank: max {np.max(g):.3e}")
    if "s14_card_gap" in ok:
        s = ok["s14_card_gap"].to_numpy(dtype=float)
        s = s[np.isfinite(s)]
        if len(s):
            print(f"replay vs committed s14 card: max |gap| {np.max(s):.3e} "
                  f"over {len(s)} cards")
    if "s14_card_patient_auc" in ok:
        s = (ok["measured_patient_auc_mean"] - ok["s14_card_patient_auc"]).to_numpy(
            dtype=float)
        s = s[np.isfinite(s)]
        if len(s):
            print(f"patient-level replay vs committed s14 card: max |gap| "
                  f"{np.max(np.abs(s)):.3e} over {len(s)} cards")


def print_failure_analysis(d: pd.DataFrame, k: int = 6) -> None:
    """
    Where the prediction misses, and the decomposition that says why.

    A theory section that never mispredicts has not been tested. These are the worst
    rows, printed with the three diagnostics that separate the candidate explanations:
    the size of the test arm, the label prevalence, and how much of the residual
    survives when the ceiling is recomputed on the TEST rows alone (which isolates
    train/test positional profile shift from everything else).
    """
    ok = d[d["status"] == "ok"].copy() if "status" in d else d.copy()
    if not len(ok):
        return
    ok["abs_err"] = ok["err_predicted"].abs()
    worst = ok.sort_values("abs_err", ascending=False).head(k)
    hdr = (f"{'dataset':<34}{'err':>8}{'n_test':>9}{'pos':>7}{'subj':>6}{'prev':>8}"
           f"{'ceil_te':>9}{'err_te':>8}{'shift':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in worst.iterrows():
        shift = float(r["ceiling_plugin"]) - float(r["ceiling_test_rows_only"])
        print(f"{r['dataset']:<34}{r['err_predicted']:>+8.3f}"
              f"{int(r['n_test_slices']):>9,}{int(r['n_test_pos_slices']):>7,}"
              f"{int(r['n_test_subjects']):>6}{r['slice_prevalence']:>8.4f}"
              f"{r['ceiling_test_rows_only']:>9.3f}"
              f"{r['err_ceiling_test_rows_only']:>+8.3f}{shift:>+8.3f}")
    print("-" * len(hdr))
    small = ok[ok["n_test_pos_slices"] < 300]
    big = ok[ok["n_test_pos_slices"] >= 300]
    if len(small) and len(big):
        print(f"MAE where the test arm holds < 300 positive slices "
              f"(n={len(small)}): {small['abs_err'].mean():.4f}")
        print(f"MAE where it holds >= 300           "
              f"(n={len(big)}): {big['abs_err'].mean():.4f}")
    rare = ok[ok["slice_prevalence"] < 0.02]
    if len(rare):
        print(f"MAE where slice prevalence < 0.02   (n={len(rare)}): "
              f"{rare['abs_err'].mean():.4f}")


def print_patient_table(d: pd.DataFrame) -> None:
    ok = d[d["status"] == "ok"] if "status" in d else d
    if not len(ok):
        return
    hdr = (f"{'dataset':<34}{'relpos':>12}{'cons%':>7}{'hit%':>6}{'sliceA':>8}"
           f"{'pat_mean':>10}{'depth_sur':>11}{'d_err':>8}"
           f"{'pat_max':>9}{'pred_max':>9}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in ok.iterrows():
        print(f"{r['dataset']:<34}"
              f"{('pubcol' if 'published' in str(r['relpos_source']) else 'within-vol'):>12}"
              f"{100 * float(r.get('frac_volumes_consecutive_slice_index', np.nan)):>7.0f}"
              f"{100 * float(r['frac_volumes_hitting_argmax_bin']):>6.1f}"
              f"{_fmt(r['measured'], 3):>8}"
              f"{_fmt(r['measured_patient_auc_mean'], 3):>10}"
              f"{_fmt(r['depth_surrogate_patient_auc_mean'], 3):>11}"
              f"{_fmt(r['err_patient_depth_surrogate_mean'], 3):>8}"
              f"{_fmt(r['measured_patient_auc_max'], 3):>9}"
              f"{_fmt(r['predicted_patient_auc_max'], 3):>9}")
    m = ok["measured_patient_auc_max"].to_numpy(dtype=float)
    hold = ok["annihilation_precondition_holds"].to_numpy(dtype=bool)
    fin = np.isfinite(m)
    print("-" * len(hdr))
    if (hold & fin).any():
        sel = ok[hold & fin]
        v = sel["measured_patient_auc_max"].to_numpy(dtype=float)
        w = sel["patient_auc_max_within_partition_maxdev"].to_numpy(dtype=float)
        w = w[np.isfinite(w)]
        print(f"Corollary 3.1: on the {len(sel)} benchmarks whose precondition holds, "
              f"WITHIN-partition max-aggregated patient AUROC deviates from 0.500 by "
              f"at most {np.max(w) if len(w) else float('nan'):.3e}")
        print(f"               pooled across folds it runs {np.min(v):.4f}-"
              f"{np.max(v):.4f}: pooling out-of-fold predictions whose fitted maxima "
              f"differ makes fold identity rankable, the same artefact s14's "
              f"prevalence_baseline_check reports for the constant predictor")
    if ((~hold) & fin).any():
        v = m[(~hold) & fin]
        print(f"Corollary 3.4: on the {int(((~hold) & fin).sum())} benchmarks whose "
              f"precondition FAILS, it runs {np.min(v):.3f}-{np.max(v):.3f} "
              f"-- the annihilation is absent exactly where the theory says it "
              f"should be")
    if "frac_volumes_consecutive_slice_index" in ok:
        cons = ok["frac_volumes_consecutive_slice_index"].to_numpy(dtype=float)
        e = ok["err_patient_depth_surrogate_mean"].to_numpy(dtype=float)
        sel = np.isfinite(e) & (cons >= 0.98)
        oth = np.isfinite(e) & (cons < 0.98)
        if sel.any():
            print(f"Corollary 3.2: on the {int(sel.sum())} benchmarks whose volumes are "
                  f"consecutively indexed, the DEPTH-ONLY surrogate reproduces the "
                  f"measured patient AUROC to {np.max(np.abs(e[sel])):.3e} -- position "
                  f"contributes nothing at patient level")
        if oth.any():
            print(f"               on the {int(oth.sum())} whose volumes are not, it "
                  f"misses by up to {np.max(np.abs(e[oth])):.3f}, because the bin set "
                  f"is then a function of the index pattern and not of depth alone")


def print_sim_table(res: dict) -> None:
    d = res["grid"]
    if not len(d):
        print("no simulation cells")
        return
    hdr = (f"{'width':>7}{'B':>4}{'S':>6}{'n/pt':>6}{'sd_u':>6}{'theta':>7}"
           f"{'ceiling':>9}{'xfit':>8}{'PRED':>8}{'MEAS':>8}{'sd':>7}{'err':>8}")
    print(hdr)
    print("-" * len(hdr))
    for _, r in d.iterrows():
        print(f"{r['width']:>7.2f}{int(r['n_bins']):>4}{int(r['n_subjects']):>6}"
              f"{int(r['slices_per_subject']):>6}{r['subject_sd']:>6.2f}"
              f"{r['theta']:>7.2f}{r['ceiling_plugin']:>9.3f}{r['ceiling_xfit']:>8.3f}"
              f"{r['predicted']:>8.3f}{r['measured_mean']:>8.3f}"
              f"{r['measured_sd']:>7.3f}{r['err_pred']:>+8.3f}")
    s = res["summary"]
    print("-" * len(hdr))
    print(f"cells {s['n_cells']}   "
          f"predicted: MAE {s['pred']['mae']:.4f} bias {s['pred']['bias']:+.4f} "
          f"max {s['pred']['max_abs']:.4f}   R2 {s['r2_predicted_vs_measured']:.4f}")
    print(f"          uncorrected ceiling: MAE {s['ceiling']['mae']:.4f} "
          f"bias {s['ceiling']['bias']:+.4f} max {s['ceiling']['max_abs']:.4f}")
    print(f"          cross-fitted ceiling: MAE {s['xfit']['mae']:.4f} "
          f"bias {s['xfit']['bias']:+.4f} max {s['xfit']['max_abs']:.4f}")


def render_card(bundle: dict) -> str:
    """A markdown card for the paper, in the same spirit as s14.render_card."""
    L: list[str] = ["# Positional theory: predicted vs measured", "",
                    f"*s15_positional_theory v{bundle.get('version')} - "
                    f"{bundle.get('generated_utc')}, B = {bundle.get('n_bins')} bins*",
                    "",
                    "Positional AUROC ceiling of a label table, from Proposition 2:",
                    "",
                    "    A = 1/2 + sum_{b<b'} q_b q_b' |pi_b - pi_b'| "
                    "/ (2 theta (1-theta))",
                    "      = (1/2) (1 + G / (1 - theta))",
                    "",
                    "with q the positional density of slices, pi the positional risk "
                    "profile, theta the",
                    "slice prevalence and G the Gini coefficient of pi. No model, no "
                    "split, no images.", ""]
    s = (bundle.get("simulation") or {}).get("summary") or {}
    if s:
        L += ["## Simulation", "",
              f"| estimator | MAE | bias | max abs |", "|---|---|---|---|",
              f"| predicted (both corrections) | {s['pred']['mae']:.4f} | "
              f"{s['pred']['bias']:+.4f} | {s['pred']['max_abs']:.4f} |",
              f"| uncorrected plug-in ceiling | {s['ceiling']['mae']:.4f} | "
              f"{s['ceiling']['bias']:+.4f} | {s['ceiling']['max_abs']:.4f} |",
              f"| cross-fitted ceiling | {s['xfit']['mae']:.4f} | "
              f"{s['xfit']['bias']:+.4f} | {s['xfit']['max_abs']:.4f} |", "",
              f"{s['n_cells']} grid cells; R^2 of predicted vs measured "
              f"{s['r2_predicted_vs_measured']:.4f}.", ""]
    rows = [r for r in (bundle.get("real") or []) if r.get("status") == "ok"]
    if rows:
        L += ["## Real data", "",
              "| dataset | n rows | prevalence | Gini | ceiling | null | cross-fit | "
              "predicted | measured | error |", "|---|---|---|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(f"| {r['dataset']} | {int(r['n_rows']):,} | "
                     f"{_fmt(r['slice_prevalence'], 3)} | {_fmt(r['gini_of_profile'], 3)} | "
                     f"{_fmt(r['ceiling_plugin'], 3)} | "
                     f"{_fmt(r['null_ceiling_median'], 3)} | "
                     f"{_fmt(r['ceiling_xfit'], 3)} | **{_fmt(r['predicted'], 3)}** | "
                     f"**{_fmt(r['measured'], 3)}** | {_fmt(r['err_predicted'], 3)} |")
        e = np.array([r["err_predicted"] for r in rows], dtype=float)
        e = e[np.isfinite(e)]
        L += ["", f"Prediction error: mean {np.mean(e):+.4f}, MAE "
                  f"{np.mean(np.abs(e)):.4f}, max {np.max(np.abs(e)):.4f} over "
                  f"{len(e)} benchmark arms.", ""]
        L += ["### Proposition 3, patient level", "",
              "| dataset | rel. position | consecutive | precondition | patient AUROC "
              "(mean) | depth-only surrogate | patient AUROC (max) |",
              "|---|---|---|---|---|---|---|"]
        for r in rows:
            L.append(
                f"| {r['dataset']} | "
                f"{'published column' if 'published' in str(r['relpos_source']) else 'within-volume'} | "
                f"{100 * float(r.get('frac_volumes_consecutive_slice_index', float('nan'))):.0f}% | "
                f"{'holds' if r.get('annihilation_precondition_holds') else 'FAILS'} | "
                f"{_fmt(r['measured_patient_auc_mean'], 3)} | "
                f"{_fmt(r['depth_surrogate_patient_auc_mean'], 3)} | "
                f"{_fmt(r['measured_patient_auc_max'], 3)} |")
        L.append("")
    rs = bundle.get("rsna_burduja") or []
    if rs and "label" in (rs[0] or {}):
        L += ["## RSNA ICH under Burduja et al.'s published split geometry", "",
              "| label | prevalence | Gini | predicted | measured (mean over draws) | "
              "error | published | trivial fraction |",
              "|---|---|---|---|---|---|---|---|"]
        for r in rs:
            L.append(f"| {r['label']} | {r['prevalence']:.4f} | {r['gini']:.3f} | "
                     f"**{r['predicted']:.4f}** | **{r['measured_mean']:.4f}** "
                     f"(sd {r['measured_sd']:.4f}) | {r['err_predicted']:+.4f} | "
                     f"{r['published']:.4f} | {r['trivial_fraction']:.3f} |")
        e = np.array([r["err_predicted"] for r in rs], dtype=float)
        L += ["", f"Prediction error: MAE {np.mean(np.abs(e)):.4f}, max "
                  f"{np.max(np.abs(e)):.4f}.", ""]
    L += ["## Limits", "",
          "* The ceiling is a CERTIFICATE, not a forecast of any particular model: by "
          "Neyman-Pearson no",
          "  function of the binned slice position can beat it, and a measured "
          "positional baseline above it",
          "  (beyond sampling error) indicates a bug or a second leakage channel, not "
          "a better positional model.",
          "* The plug-in ceiling is optimistically biased; its permutation null is "
          "reported next to it and is",
          "  not 0.5. Cross-fitting removes most of the bias and is the recommended "
          "estimator when a",
          "  subject-level split of the label file can be made.",
          "* A high ceiling says the EVALUATION PROTOCOL is inflatable without pixels. "
          "It does not say the",
          "  published model learned nothing. See s14's TRIVIAL_FRACTION_LIMITS, which "
          "applies unchanged.", ""]
    return "\n".join(L)


def parse_args(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__.split("Usage:")[0],
        formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--simulate", action="store_true")
    p.add_argument("--real", action="store_true")
    p.add_argument("--rsna-burduja", action="store_true")
    p.add_argument("--all", action="store_true")
    p.add_argument("--quick", action="store_true")
    p.add_argument("--bins", type=int, default=DEFAULT_BINS)
    p.add_argument("--reps", type=int, default=30, help="simulation replicates per cell")
    p.add_argument("--rsna-reps", type=int, default=100)
    p.add_argument("--null-permutations", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--only", default="", help="comma-separated dataset names")
    p.add_argument("--out-dir", default=None)
    p.add_argument("--no-write", action="store_true")
    return p.parse_args(argv)


def main(argv=None) -> int:
    a = parse_args(argv)
    if not any((a.self_test, a.simulate, a.real, a.rsna_burduja, a.all)):
        print("nothing to do: pass --self-test, --simulate, --real, --rsna-burduja "
              "or --all", file=sys.stderr)
        return 2
    out_dir = Path(a.out_dir) if a.out_dir else REPO / "pipeline_out" / "positional_theory"
    rc = 0
    bundle = {"tool": "s15_positional_theory", "version": "1.0",
              "generated_utc": _dt.datetime.now(_dt.timezone.utc).isoformat(
                  timespec="seconds"),
              "n_bins": a.bins, "seed": a.seed}

    if a.self_test or a.all:
        rc |= self_test(quick=a.quick)

    if a.simulate or a.all:
        print("\n" + "=" * 72)
        print("SIMULATION: predicted vs measured across controlled concentrations")
        print("=" * 72)
        res = run_simulation(reps=a.reps, seed=a.seed, quick=a.quick)
        print_sim_table(res)
        bundle["simulation"] = {"summary": res["summary"],
                                "grid": res["grid"].to_dict(orient="records")}

    if a.real or a.all:
        print("\n" + "=" * 72)
        print("REAL DATA: every label table on disk")
        print("=" * 72)
        only = [s.strip() for s in a.only.split(",") if s.strip()] or None
        d = run_real(n_bins=a.bins, n_perm=a.null_permutations, seed=a.seed, only=only)
        print()
        print_real_table(d)
        print("\nWHERE THE PREDICTION FAILS")
        print_failure_analysis(d)
        print("\nPROPOSITION 3, patient level")
        print_patient_table(d)
        bad = d[d["status"] != "ok"] if "status" in d else d.iloc[0:0]
        if len(bad):
            print("\nnot scored:")
            for _, r in bad.iterrows():
                print(f"  {r['dataset']:<36} {r['status']}")
        bundle["real"] = d.to_dict(orient="records")

    if a.rsna_burduja or a.all:
        print("\n" + "=" * 72)
        print("RSNA ICH under Burduja et al.'s published split geometry")
        print("=" * 72)
        r = run_rsna_burduja(n_bins=a.bins,
                             n_rep=(20 if a.quick else a.rsna_reps))
        if "status" in r.columns:
            print(f"  skipped: {r.iloc[0].to_dict()}")
        else:
            hdr = (f"{'label':<20}{'prev':>8}{'Gini':>8}{'ceiling':>9}{'PRED':>8}"
                   f"{'MEAS':>8}{'sd':>7}{'err':>8}{'published':>11}"
                   f"{'triv_meas':>10}{'triv_pred':>10}")
            print(hdr)
            print("-" * len(hdr))
            for _, x in r.iterrows():
                print(f"{x['label']:<20}{x['prevalence']:>8.4f}{x['gini']:>8.4f}"
                      f"{x['ceiling_plugin']:>9.4f}{x['predicted']:>8.4f}"
                      f"{x['measured_mean']:>8.4f}{x['measured_sd']:>7.4f}"
                      f"{x['err_predicted']:>+8.4f}{x['published']:>11.4f}"
                      f"{x['trivial_fraction']:>10.3f}"
                      f"{x['predicted_trivial_fraction']:>10.3f}")
            e = r["err_predicted"].to_numpy(dtype=float)
            print("-" * len(hdr))
            print(f"prediction error: mean {np.mean(e):+.4f}  "
                  f"MAE {np.mean(np.abs(e)):.4f}  max {np.max(np.abs(e)):.4f}")
        bundle["rsna_burduja"] = r.to_dict(orient="records")

    if not a.no_write and any(k in bundle for k in
                              ("simulation", "real", "rsna_burduja")):
        out_dir.mkdir(parents=True, exist_ok=True)
        p = out_dir / "positional_theory.json"
        p.write_text(json.dumps(bundle, indent=2, default=s14._json_default))
        m = out_dir / "positional_theory.md"
        m.write_text(render_card(bundle))
        print(f"\n  wrote {p}")
        print(f"  wrote {m}")
    return rc


if __name__ == "__main__":
    raise SystemExit(main())
