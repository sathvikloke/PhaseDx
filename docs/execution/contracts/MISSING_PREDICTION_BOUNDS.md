# External CT: sensitivity to missing predictions

**Current applicability note (October 15 handoff):** The historical design text below predates the comparator amendment. Its references to the “original geometry comparator” are superseded by [the October 14 external protocol](EXTERNAL_PROTOCOL_20261014.md): use `G2_tree_transported_rank_reconstructed_frozen`, explicitly as an adaptation. The finite-cohort bounds and fixed synthetic checks apply unchanged to that named comparison. This note does not authorize new execution or retrospectively change the historical design record.

Codex design addition, October 14, 2026. Prepared before external outcome access. This resolves an analysis choice for the pending external study; it is not an observed result. It supplements CT_EXTERNAL_DESIGN_DECISION.md and does not replace the original paired complete-case primary endpoint. Claude implements the bounded synthetic checks and later, only when authorized, applies the fixed diagnostic to the external cohort.

## Scientific question and fixed population

An image-versus-geometry AUROC difference among successfully processed examinations can differ from the comparison across the intended cohort if model failures are selective. Counting failures is necessary but does not quantify this uncertainty. Report a conservative range over the unknown scores of failed examinations in addition to the complete-case comparison.

Define the candidate roster before model execution. Apply the scientifically justified header/series eligibility rules without outcomes, then retain the intended eligible examination population. Reference-standard availability is a separate state, not a model eligibility criterion. The bounds below refer to the intended eligible examinations with an authenticated observed binary reference label. Count eligible examinations without labels separately; these bounds do not cover their unknown outcomes. A source-level exclusion, eligibility failure, missing reference label, and model prediction failure must not be merged into one category.

The primary remains the seed42 image maximum-slice score minus the frozen original geometry comparator on identical paired evaluable examinations. Both absolute AUROCs are reported. Seed123 remains descriptive. The estimand is examination-weighted discrimination; clustering the bootstrap by person accounts for repeated-examination dependence but does not turn that estimand into an equally weighted person-level AUROC. No claim of clinical utility follows from discrimination alone.

## Derivation, including ties

Let N+ and N- be positive and negative examination counts in the intended eligible, labeled population. Let m+ and m- be counts in the paired complete subset, where both models have valid complete examination scores. Put T = N+ N-, E = m+ m-, and U = T - E. T counts all eligible labeled positive-negative examination pairs; E counts pairs for which both models' scores are observed.

For model k, let W_k be the sum over observed complete positive-negative pairs of h(s_positive, s_negative), where h is 1 for a correctly ordered pair, 1/2 for a tie, and 0 otherwise. W_k need not be an integer. Every unobserved pair contributes a value in [0,1], so:

    AUC_k_full in [W_k / T, (W_k + U) / T].

For the paired model difference D = AUC_image - AUC_geometry, the observed-pair numerator is W_image - W_geometry. Each unobserved pair contributes a difference in [-1,1], giving:

    D_full in [(W_image - W_geometry - U) / T,
               (W_image - W_geometry + U) / T].

When E > 0, define c = E/T and D_complete = (W_image-W_geometry)/E. Then the same interval is:

    [c D_complete - (1-c), c D_complete + (1-c)].

This is an algebraic bound for this finite labeled cohort. It is conservative when any known unpaired model scores or extra score constraints could narrow the range; do not advertise universal sharpness. Missing scores are not generated, imputed, fitted, or assigned outcome-informed values for analysis. Extreme values appear only in the mathematical argument.

Pair coverage c is not the fraction of examinations with predictions: it is (m+/N+) times (m-/N-). Failure concentrated in the smaller class can matter disproportionately. That motivates reporting missingness by the observed reference label after the frozen outcome join, without changing eligibility or trying alternative rules to improve coverage.

## Interpretation and result-independent rules

- With full pair coverage, U=0 and both bounds reduce to the corresponding empirical complete-cohort value.
- With no observed paired positive-negative pair but T>0, component bounds are [0,1] and the difference bound is [-1,1]. The complete-case AUROC remains undefined. Do not invent a point estimate at the midpoint.
- If either full-cohort class is empty, T=0; all AUROC quantities and bounds are undefined with a structured reason.
- If the lower difference bound exceeds zero, every unrestricted completion consistent with the retained paired scores preserves a positive empirical difference within the labeled eligible cohort. This is not a confidence statement, evidence of clinical usefulness, or a population-level guarantee.
- If the bound includes zero, an advantage across the intended labeled cohort is not identified from the available paired scores alone. This is not proof that the actual difference is zero or negative.
- Bounds and bootstrap confidence intervals answer different questions. Display them in separate columns and use separate labels. These bounds make no missing-at-random assumption and contain no sampling-confidence level. The original 2000-draw paired cluster bootstrap continues to describe sampling uncertainty conditional on the observed complete subset. Do not bootstrap the bounds or add a second interval family in this assignment.
- Known repeated patients affect sampling uncertainty; the deterministic finite-examination-cohort inequality itself does not require independent examinations. If repeats are unidentifiable, retain the existing cluster-assumption limitation.
- An unavailable/nonportable original geometry model is an applicability failure. Do not replace it silently with a new comparator or treat synthetic bounds as an executed external comparison.

## Minimal output attached to the external result

One fixed row for primary seed42: candidate examinations, header-eligible examinations, labeled eligible examinations, N+/N-, m+/m-, image failures, geometry failures, paired failures, missing labels, T, E, pair coverage, complete-case component AUROCs and difference, component completion bounds, difference completion bound, and reason/status fields. Report overlaps among failure categories or a mutually exclusive partition so counts cannot be double-counted. No subgroup performance hunting, additional seeds, thresholds, p-values, or new bootstrap draws are introduced. An optional displayed seed123 row uses its own explicitly named paired subset and remains descriptive.

## Bounded synthetic checks for implementation

Use hand-specified small score arrays, not a simulation sweep. Check: complete data; one failed positive; one failed negative; failures in both classes; asymmetric model failure; tied retained scores; no complete pair; and one-class cohort. Verify the observed W values by direct positive-negative comparisons. For a tiny fixture, enumerate a few finite score completions and verify their full-cohort values lie inside the bounds; this checks the implementation, not the proof. Use stable example identifiers and the real full-roster scorer. No new clinical data or model inference is needed.

## Practical consequence

The external study will provide an interpretable complete-case comparison and show whether its sign is robust to otherwise unobserved model failures in the labeled intended population. A clean positive comparison on a small surviving subset will no longer be reported without quantifying how incomplete prediction coverage limits that conclusion. These bounds do not repair selection into the source dataset, undocumented original cohort selection, missing labels, or incompatible reference standards.
