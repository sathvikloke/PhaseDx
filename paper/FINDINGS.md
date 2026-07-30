# FINDINGS.md — every established finding, its number, its interval, its file

Rewritten 2026-07-29 after the adjudication remedy, the access-recovery pass, the full re-code
under codebook v1.2, and four reserve blocks. **This is the document the manuscript is written
from.** If a number is not here with a file path next to it, it does not go in the paper.

Ordering rule for this file and for the manuscript: **weakest first.** A reader who stops after
§1 must have read the things that most limit the claim.

Every proportion is a Wilson score 95% two-sided interval, pre-specified for every proportion in
`paper/screen_protocol.md` §8 before any analysis-sample record was read. Every prevalence number
below is reproduced by one command:

```bash
/Users/sathvikloke/Downloads/PhaseDx/venv/bin/python \
  /Users/sathvikloke/Downloads/PhaseDx/paper/screen/analysis/pool_final.py
```

which reads only committed files and writes `paper/screen/analysis/pooled_final.json`.

---

## 0. The five sentences that govern everything below

1. **The extension did not fix the access rate, and one previously published number in this
   repository was wrong in our favour.** Full text was unreachable for **32.6% [25.3%, 40.9%]**
   of the eligible-looking set after 250 screened records — against 36.4% at n = 100. Screening
   150 more papers moved censoring by under four percentage points. The 29.6% figure previously
   carried in this file and in `paper/prisma_flow.json` was an artefact: the access-recovery
   overlay was applied without codebook rule **D1**, which forbids excluding a paper at full text
   when the full text was never obtained. Five such papers were sitting in the "excluded" column.
   Correctly classified, the main sample's censoring is **35.6%**, not 29.6%, and the recovery
   pass bought 0.8 points, not 6.8. See §1.1.
2. **Because censoring still exceeds 15%, the protocol's own rule makes the bound the headline.**
   The reportable primary result is **P1 ∈ [0.0%, 32.6%]** over 135 eligible papers. The
   complete-case point estimate is 0/91 = 0.0% [0.0%, 4.1%] and is reported *alongside* the
   bound, never in place of it.
3. **The pre-registered sample-size target was met.** 91 included and reachable papers against a
   target of 75; the extension rule fired and stopped at permutation position 260. A fifth block
   (R4, positions 261–310) was screened past the stopping point and is reported separately as a
   **post-hoc** extension, never pooled into the pre-registered denominator.
4. **The censoring-free result is total and needs no denominator.** Across **345 independently
   coded records over 300 distinct sampled papers**, the four zero-image sub-flags —
   constant/prevalence, positional, acquisition-metadata, permuted-label — are TRUE **zero
   times**. Not one paper in this literature measures whether its benchmark is solvable without
   pixels.
5. **Agreement now clears its floor, but the number that clears it is a re-encoding, not a
   re-rating.** Fleiss' κ on the primary flag moves −0.015 → **0.932 [0.777, 1.000]** and raw
   agreement 65.6% → **95.6%**, both above the pre-registered floors. That is computed on the
   *same four sealed files* re-expressed under two amendments that only add a missing level. A
   genuine post-amendment reliability estimate needs a fresh independent four-screener re-coding
   under v1.2, and **that has not been run.** All four reserve blocks are single-coded, so they
   contribute no agreement information at all. See §1.2.

---

## 1. WEAKEST FIRST — the five things that cap every claim in this paper

### 1.1 The access rate is the binding constraint, and enlarging the sample did not touch it

Source: `paper/screen/analysis/pooled_final.json` → `flow_by_block`, `primary_pre_registered`;
rule in `paper/screen_frame.json` → `missing_and_unreachable.rule_4`; ladder in
`paper/screen_protocol.md` §7.

| block | positions | screened | included | unreachable | unreachable % of eligible |
|---|---|---|---|---|---|
| main (re-coded v1.2 + recovery) | 1–100 | 100 | 38 | 21 | **35.6%** |
| R1 | 111–160 | 50 | 17 | 9 | 34.6% |
| R2 | 161–210 | 50 | 17 | 5 | 22.7% |
| R3 | 211–260 | 50 | 19 | 9 | 32.1% |
| **pooled, pre-registered** | | **250** | **91** | **44** | **32.6% [25.3%, 40.9%]** |
| *R4 — post-hoc, beyond the stopping rule* | *261–310* | *50* | *23* | *12* | *34.3%* |
| *pooled including R4 — post-hoc* | | *300* | *114* | *56* | *32.9% [26.3%, 40.3%]* |

Four independent blocks, four separate screening sessions, and the unreachable rate sits between
23% and 36% in every one of them. **The censoring is a property of the literature's access
conditions, not of a sample size or of one screener's environment, and no further screening can
narrow it.** Only recovered full texts can.

**A correction that runs against us, stated plainly.** `paper/prisma_flow.json`,
`paper/prisma_flow.md`, `paper/figures/prisma_flow.svg` and the previous §1.4a of this file report
the main sample as 38 included / 16 unreachable / 54 eligible, S6 = 29.6%. That is wrong. Those
artefacts were built from `paper/screen/analysis/recovery_out.json`, which predates the v1.2
re-code, and they never applied rule **D1** — *"unreachable dominates included; an unreachable
record may be coded `excluded` only if `stage1_decision='exclude'`"* (`paper/screen_adjudication.md`
line 111). Five records had been given a full-text exclusion code while
`fulltext_reachable = unreachable_paywalled` and `stage1_decision = go_to_fulltext`, i.e. a screener
assigned an exclusion reason to a paper nobody had opened:

| PMID | batch | code assigned without the full text |
|---|---|---|
| 33937792 | A | E-SEG |
| 42162744 | A | E-DERIV |
| 35641181 | D | E-DERIV |
| 35787928 | D | E-DERIV |
| 41874622 | D | E-SEG |

`paper/screen_recoded.json` applies D1 and moves all five into `eligible_unreachable`, which is what
`_no_number_was_improved_by_a_rule_change` in that file already records as a change made *against
us*. Its own endpoint block gives 38 included / 21 unreachable / 59 eligible. **The re-code is
authoritative; the PRISMA artefacts are stale.** The one-line consequence: the access recovery
narrowed censoring from 36.4% to 35.6%, not to 29.6%.

**The direction of the residual bias was stated before the result was seen** (protocol §7):
paywalled articles skew toward higher-tier clinical journals, which are the venues most likely to
demand a comparator arm, so silently dropping them would push P1 *downward* — in the direction that
flatters our thesis. The exploratory venue cut in §5.2 confirms the mechanism: censoring is
**36.0% [26.7%, 46.6%]** in clinical/radiology journals against **11.5% [4.0%, 29.0%]** in
engineering/computing journals. This is why both bounding analyses are unconditional.

**What was and was not permitted.** Sci-Hub, LibGen and every other infringing source were refused
in every block (`screen_reserve_R3.json` → `sources_refused`, `screen_reserve_R4.json` →
`sources_refused`). Bot-detection was not circumvented. Where a paper was reachable only through
such a source it stayed unreachable. The one access lever that *did* work is worth naming: the
Europe PMC render-PDF rung recovered five records inside block R4 and took that block's S6 from
60.7% to 34.3% (`screen_reserve_R4.json` → `access_recovery_in_this_block`). That is the lever a
submitted version needs applied to all 44.

### 1.2 The post-amendment κ is a re-encoding of the same four files, not an independent re-rating

Source: `paper/screen/analysis/adjudication_out.json`; script
`paper/screen/analysis/adjudicate.py`; audit trail `paper/screen_adjudication.md`; floor
pre-registered in `paper/screen_protocol.md` §6.

15 overlap papers, 4 screeners, bootstrap percentile 95% over 2,000 resamples, seed 20260729 — all
fixed before coding. **Three numbers exist and only two are measurements.**

| | raw agreement | Fleiss' κ | Gwet's AC1 | unanimous |
|---|---|---|---|---|
| **(1) pre-reconciliation, v1.0 as sealed** | **65.6% [50.0, 80.0]** | **−0.015 [−0.164, 0.120]** | 0.479 [0.119, 0.740] | 6/15 |
| **(2) counterfactual v1.2 encoding** | **95.6% [86.7, 100.0]** | **0.932 [0.777, 1.000]** | 0.934 [0.800, 1.000] | 14/15 |
| (2′) same, collapsed to TRUE / not-TRUE | 100% | 1.000 | 1.000 | 15/15 |
| (3) post-adjudication consensus | 1.000 **by construction** | undefined | — | 15/15 |

**The floor is assessed against (2) and against nothing else.** κ 0.932 ≥ 0.60 and raw 95.6% ≥ 0.90,
so **both pre-registered floors are now met** (`adjudication_out.json` → `floor`). (3) is a
construction and carries no information. (1) is reported first and is never replaced.

Pairwise Cohen's κ on the primary flag, all six pairs:

| pair | (1) pre, % agree / κ | (2) counterfactual v1.2, % agree / κ |
|---|---|---|
| S1–S2 | 73.3% / 0.000 | 93.3% / 0.898 |
| S1–S3 | 40.0% / 0.000 | 93.3% / 0.898 |
| S1–S4 | 100% / undefined (one category) | 100% / 1.000 |
| S2–S3 | 66.7% / 0.390 | 100% / 1.000 |
| S2–S4 | 73.3% / 0.000 | 93.3% / 0.898 |
| S3–S4 | 40.0% / 0.000 | 93.3% / 0.898 |

**Why the original failure was a codebook defect and not a screener failure.** Restricted to the six
overlap papers all four screeners both *obtained* and *included* — the only papers on which a P1
code is defined — agreement is **100%, AC1 = 1.000, 6/6 unanimous**, and all four screeners produce
identical vectors on **all six** `trivial_baseline` sub-flags on **all six** papers
(`adjudication_out.json` → `restricted_to_core`). The disagreement lived entirely in the placeholder
the form forced screeners to type for "could not be assessed": S1 wrote `false`, S2 `null`, S3 the
string `"unclear"`, S4 `false`. `trivial_baseline` was declared boolean with no third level.
Fourteen rules D1–D14 close that and thirteen other gaps.

**What this caps.** Three things, all of which must appear in the paper:

- Reliability rests entirely on **15 papers coded by 4 raters**. The 200 reserve records are
  **single-coded** (R1 S1, R2 S1, R3 S3, R4 S1 — `pooled_final.json` →
  `agreement.reserve_block_screeners`). No reserve record contributes agreement information, so the
  pooled n = 91 inherits its reliability estimate from a 15-paper subsample of its first 100.
- The protocol's §6 provision for a **20% within-batch re-code by the next screener in the cycle**
  outside the overlap set has **not been executed**. Disagreement outside the overlap set is
  invisible by construction.
- A genuine post-amendment reliability estimate — fresh independent four-screener coding under
  v1.2 — **has not been run** (`screen_protocol.md` §6.1 states this as an outstanding action).

`split_unit` is the one field whose disagreement is substantive rather than clerical: κ 0.637
[0.430, 0.824] even after the amendment, 9/15 unanimous. **Every statement of S4 must carry κ ≈ 0.64
in the same sentence** and cannot be presented as a precise prevalence.

### 1.3 The primary endpoint's *evidenced* denominator is 79, not 91

`paper/screen_frame.json` → `reading_effort` forbids an unevidenced negative: before
`trivial_baseline` may be coded all-false, the screener must run all 14 named terms over the full
text **and the supplement** and record having done so. Of the 91 complete-case papers, **79 carry a
fully evidenced 14-term search** including the supplement. The 12 that do not are 10 main-sample
papers coded `fulltext_search = main_text_only`, plus one record each in R2 and R4 whose supplement
exists but could not be searched (R2 33713959, video-only supplement; R4's PMC deposit is a
scanned-manuscript PDF). Restricting to the evidenced set changes nothing about the numerator:
**0/79 = 0.0% [0.0%, 4.6%]**.

### 1.4 Block R4 is outside the pre-registered stopping rule

Protocol §3.1 stops the extension at 75 included papers. The running total was 38 (main) → 55 (R1)
→ 72 (R2) → **91 (R3)**, so the rule fired at permutation position 260 and did not authorise a
fifth block. R4 was screened on explicit instruction and is coded to the same standard, but
continuing past a stopping rule that has already fired is a data-dependent continuation, and pooling
it silently would convert pre-specified inverse sampling into optional stopping.
`screen_reserve_R4.json` → `STATUS_OF_THIS_BLOCK_UNDER_THE_PRE_SPECIFIED_EXTENSION_RULE` says so in
its own file. **Both figures are reported side by side and labelled**; nothing about R4 changes the
primary result in either direction.

### 1.5 What the screen still does not establish

- It does not establish that the sampled papers are *wrong*. It establishes that they do not report
  the check.
- It does not establish prevalence in the slice-level literature specifically. Slice-headline papers
  are 17/91 = 18.7% of the eligible population; any sentence of the form "of slice-level papers,
  X%" rests on n = 19 or smaller.
- The **venue-type subgroup is not the pre-specified one.** Protocol §8 requires classification from
  each journal's scope statement before unblinding; that was never done, and §5.2 substitutes a
  keyword heuristic over journal names with a 23-paper `unclassified` bucket. It is labelled
  provisional wherever it appears.
- Six of the 15 overlap papers retain **between-screener differences on fields the adjudication did
  not settle** — `n_positive_reported`, `uncertainty_interval_reported`, `input_representation`,
  `split_disjointness_verified`, `label_broadcast_to_slices`, `headline_selection_rule`
  (`pooled_final.json` → `unsettled_fields_in_overlap_papers`). These are reported per screener
  rather than collapsed. None of the disputed values would enter the numerator of S8 or S9 as the
  codebook defines them, so those two endpoints are insensitive to the residual; the looser S9
  reading in §3.9 could move by at most one record.

---

## 2. FLOW — frame to included, pre-registered pool

Figure: **`paper/figures/prisma_flow_pooled.svg`**, generated by
`paper/screen/analysis/flow_figure.py` from `pooled_final.json`. Counts below are
`flow_pre_registered`.

```
PubMed, one frozen Boolean query, run 2026-07-29 (UTC)
  n = 9,979 records     SHA-256 d611def0785f3a5e7b7489364959f1d3471b61651f98a3ed049252654264374b
  duplicates removed 0 (PMID-unique) - registers 0 - other methods 0
        |
        |  seeded permutation (seed 20260729, SHA-256 dad12a30b77d...)
        |  never drawn 9,579 - pilot excluded a priori 10 - reserve beyond position 260: 90
        v
RANDOMLY SAMPLED FOR SCREENING          n = 250     positions 1-100, 111-260
        |                                           extension rule stopped after block R3
        v
RECORDS SCREENED (title + abstract)     n = 250
        |
        |-------------->  EXCLUDED at title/abstract        n =  79
        |                   E-SEG 33 - E-DERIV 16 - E-NONMED 9 - E-NOCLF 8
        |                   E-2D 8 - E-TYPE 4 - E-PROJ 1
        v
REPORTS SOUGHT FOR RETRIEVAL            n = 171
        |
        |-------------->  NOT RETRIEVED - full text unreachable      n =  44
        |                   NOT excluded. Eligibility unresolved. Carried into
        |                   BOTH bounding analyses (protocol section 7).
        v
REPORTS ASSESSED FOR ELIGIBILITY        n = 127
        |
        |-------------->  EXCLUDED at full text              n =  36
        |                   E-DERIV 16  <- inside the query, outside the failure mode
        |                   E-2D 6 - E-SEG 6 - E-NOCLF 5 - E-PROJ 2 - E-TYPE 1
        v
INCLUDED AND REACHABLE                  n =  91   <- complete-case denominator
                                                     pre-registered target 75: MET
                                                     79 of 91 fully evidenced (section 1.3)

ELIGIBLE-LOOKING SET = 91 included + 44 unreachable = n = 135
        S6 unreachable = 44/135 = 32.6% [25.3%, 40.9%]  >  15%  =>  THE BOUND IS THE HEADLINE
```

Exclusions at both stages, by reason: E-SEG 39, E-DERIV 32, E-2D 14, E-NOCLF 13, E-NONMED 9,
E-TYPE 5, E-PROJ 3; total 115. **E-DERIV is reported separately per protocol §9**: those 32 papers —
radiomics feature vectors, connectivity matrices, volumetry tables — are inside the query and
outside the failure mode, because no image reaches the model.

De-duplication: **0 duplicates** across all five blocks; 300 distinct PMIDs over 300 records
(`pooled_final.json` → `deduplication`). Codebook conformance: **0 D1 violations** in the pooled set
(`codebook_conformance`).

Adding the post-hoc R4 block gives 300 screened → 94 excluded at title/abstract → 206 sought → 56
unreachable → 150 assessed → 36 excluded at full text → **114 included**, eligible-looking 170,
S6 = 32.9% [26.3%, 40.3%].

---

## 3. PREVALENCE — the final numbers

Frame: PubMed, frozen query in `paper/screen/frame_meta.json`, `esearch_count = 9,979`.
Sample: `paper/screen_sample.json` v1.1. Codebook: `paper/screen_frame.json` **v1.2**.
Records: `paper/screen_recoded.json` (positions 1–100, post-adjudication consensus) and
`paper/screen_reserve_R{1,2,3,4}.json`. Analysis: `paper/screen/analysis/pool_final.py` →
`pooled_final.json`.

### 3.1 P1 (primary): zero-image baselines — THE HEADLINE IS THE BOUND

| analysis | k/n | estimate | Wilson 95% |
|---|---|---|---|
| **HEADLINE — bounding interval, protocol §7 rule_4** | | **[0.0%, 32.6%]** | outer envelope [0.0%, 40.9%] |
| lower bound (all 44 unreachable = no baseline) | 0/135 | 0.0% | [0.0%, 2.8%] |
| upper bound (all 44 unreachable = has baseline) | 44/135 | 32.6% | [25.3%, 40.9%] |
| complete case — reported, **not** the headline | 0/91 | 0.0% | [0.0%, 4.1%] |
| complete case, evidence-restricted (§1.3) | 0/79 | 0.0% | [0.0%, 4.6%] |
| *post-hoc with R4: bound* | | *[0.0%, 32.9%]* | |
| *post-hoc with R4: complete case* | *0/114* | *0.0%* | *[0.0%, 3.3%]* |

The complete-case interval at n = 91 is **[0.0%, 4.1%]**, inside the 4.9% precision the protocol set
as its target at n = 75. **"Fewer than 1 in 20" is now supportable as the complete-case statement —
and it is still not the headline, because 32.6% of the eligible set was never read.**

### 3.2 The censoring-free statement — the most robust form of the result

Across **345 independently coded records** covering **300 distinct sampled papers** — every
screener, every block, including excluded and unreachable papers — **not one record carries a single
TRUE on any of the four P1 sub-flags.** No constant-or-prevalence baseline, no positional baseline,
no acquisition-metadata baseline, no permutation null, anywhere (`pooled_final.json` →
`censoring_free`; corroborated independently by `screen_recoded.json.summary.headline`,
`screen_reserve_R3.json.block_tallies`, `screen_reserve_R4.json.primary_result`).

This statement has no denominator and is unaffected by the unreachable imputation, because it is a
statement about what was *found*. **It is the strongest form of the result the screen supports and
should carry the abstract.**

Two sub-flags outside the P1 family are TRUE somewhere, and both are the comparison the codebook
deliberately keeps out of the primary:

- `clinical_or_demographic_only` TRUE in **7** papers: 39061744, 38337016, 38784688, 40121941,
  37222638 (unreachable at pooling, so outside the complete case), and — in post-hoc R4 — 36646808,
  34765542.
- `other_non_imaging` TRUE in **1** paper: 37679806 (R1).

### 3.3 Near misses, and why each is coded FALSE

These belong in the manuscript as quotations, not as counts. All are in
`screen_reserve_R4.json` → `primary_result` unless noted.

| PMID | what it did | why it is not a P1 positive |
|---|---|---|
| **41481488** *Radiol Imaging Cancer* 2026 | built a clinical-**and-location**-only model (age, sex, tumour volume, centroid distance to hip bone) reaching macro AUC **0.867 [0.810, 0.909]**, **beating** the image-only DenseNet121 at 0.851 [0.775, 0.902]; Table 3 reports all six arms | the volume and centroid come from the paper's own segmentation network, i.e. from the pixels, so the arm is not pixel-free; the codebook's `does_not_count` list rules out a hand-crafted-feature model that uses pixels. The authors read the result as "location information improves identification", not as evidence the benchmark is largely solvable from size and position. |
| **33239711** *Sci Rep* 2020 | Table 1 tabulates label frequency against relative intracranial height band (21–30% … 71–80%) **and** reports a position-stratified AUC per band (0.838, 0.870, 0.903, 0.845, 0.764, 0.825); the authors write that the best results came from the band "which had the most number of positive cases", and dropped four of ten bands for low incidence | this is the **S5 positive** for R4 — a paper that measured the positional label distribution, saw performance track it, and still reported no positional baseline |
| **42266879** | trains on OASIS with 67,222 of 86,437 images in one class — a **77.8% no-information rate** — and reports 95.87% accuracy | the codebook requires a **measured** value; the majority-class rate is never computed, so `constant_or_prevalence` is FALSE. The clearest instance in the sample of a headline number uninterpretable without a baseline the paper does not report. |
| **42406258** | "The diagonal dotted lines in the ROC curves represent random chance performance" | chance **asserted, never measured**. FALSE on every sub-flag, recorded in its own field. Five such records exist in the pre-registered pool: 40093990, 38298725, 41568076 (main), 35562596, 41657565 (R2). |
| 38591974, 39846055 | "baseline" appears in the text | a *radiomics* model in an upgrade chain (uses pixels) and an *architectural ablation* of the authors' own network, respectively — both explicitly on the codebook's `does_not_count` list |

### 3.4 S1: any non-imaging baseline at all, including clinical-only

| analysis | k/n | estimate | Wilson 95% |
|---|---|---|---|
| complete case | 5/91 | 5.5% | [2.4%, 12.2%] |
| lower bound | 5/135 | 3.7% | [1.6%, 8.4%] |
| upper bound | 49/135 | 36.3% | [28.7%, 44.7%] |
| *post-hoc with R4* | *7/114* | *6.1%* | *[3.0%, 12.1%]* |

Four of the five are the "clinical model" arm of a clinical-plus-radiomics/DL nomogram; one
(37679806) is coded `other_non_imaging`. **The practice that exists is "does imaging beat clinical
variables". The practice that does not exist anywhere in 300 papers is "does this benchmark beat a
model with no information at all".**

### 3.5 S2 / S3: the unit of evaluation

| endpoint | k/n | estimate | Wilson 95% |
|---|---|---|---|
| S2 headline unit is the slice | 17/91 | 18.7% | [12.0%, 27.9%] |
| any slice-level metric reported (slice or both) | 19/91 | 20.9% | [13.8%, 30.3%] |
| **evaluation unit is below the patient** (slice, both, lesion, volume-not-aggregated) | **40/91** | **44.0%** | **[34.2%, 54.2%]** |
| S3 among slice-reporting papers, also report patient-level | 6/19 | 31.6% | [15.4%, 54.0%] |

Complete-case distribution of `evaluation_unit_reported` (n = 91): patient 29, unclear 19, slice 13,
lesion 12, volume-or-scan-not-patient 9, both 6, other 3. `headline_unit`:
`na_only_one_unit_reported` 85, slice 4, patient 2 — i.e. **93% of papers report exactly one unit and
never contrast two.**

`input_representation` (n = 91): 2D slice 36, 3D volume 22, unclear 13, 3D patch 12, mixed 5,
2.5D stack 3. Two in five papers on volumetric acquisitions feed the network single 2D slices.

### 3.6 S4: split unit — report with its κ attached

| analysis | k/n | estimate | Wilson 95% |
|---|---|---|---|
| complete case, explicit subject-level split | 29/91 | 31.9% | [23.2%, 42.0%] |
| lower bound | 29/135 | 21.5% | [15.4%, 29.1%] |
| upper bound | 73/135 | 54.1% | [45.7%, 62.3%] |

Distribution (n = 91): `patient_subject` 29, `random_unit_not_stated` 22, `slice_or_image` 17,
`external_cohort_only` 7, `lesion_or_roi` 6, `unclear` 5, `site_or_centre` 3, `scan_or_study` 2.

**17 of 91 papers — 18.7% — split at an image or slice unit**, so the same patient can appear on both
sides. 22 more say only "randomly split" and never name a unit; per codebook rule that is *not*
upgraded to patient-level because the word "patients" appears elsewhere. And
`split_disjointness_verified` is `not_stated` for **49 of 91**, `stated_only` for 34, and
`stated_and_checked` for **8**.

**This number carries κ = 0.637 [0.430, 0.824] (§1.2) and must never be stated without it.**

### 3.7 S5: does anyone look at where the labels sit along the stack?

| analysis | k/n | estimate | Wilson 95% |
|---|---|---|---|
| complete case, figure/table or text-with-numbers | **1/91** | **1.1%** | **[0.2%, 6.0%]** |
| complete case, including qualitative text | 2/91 | 2.2% | [0.6%, 7.7%] |
| upper bound | 45/135 | 33.3% | [25.9%, 41.6%] |
| *post-hoc with R4* | *2/114* | *1.8%* | *[0.5%, 6.2%]* |

`positional_distribution_reported` is `no` for **89 of 91**. The one positive is **42130124** (main
sample), a vertebral-level fracture table coded under adjudication rule D9; the one qualitative
mention is **36200353**. In the post-hoc block, **33239711** is a second and stronger positive
(§3.3).

**This is the screen's most direct support for the paper's thesis: the confound is not argued about
and dismissed in this literature, it is not looked at.** Note that S5 moving off zero makes the
literature look *better* on the very endpoint this paper accuses it of ignoring; D9 was adopted for
that reason, not despite it (`screen_recoded.json` → `_no_number_was_improved_by_a_rule_change`).

### 3.8 S7: headline unit × P1, full table per protocol §8

| headline_unit | P1 = TRUE | P1 = FALSE |
|---|---|---|
| `na_only_one_unit_reported` | 0 | 85 |
| slice | 0 | 4 |
| patient | 0 | 2 |
| **total** | **0** | **91** |

There is no subgroup — not slice-level, not patient-level — in which zero-image baselines are
reported.

### 3.9 S8 / S9: how the numbers are reported

| endpoint | k/n | estimate | Wilson 95% |
|---|---|---|---|
| S8 subject-clustered uncertainty interval | 2/91 | 2.2% | [0.6%, 7.7%] |
| S9 reports n positive **patients and slices** (codebook definition) | 11/91 | 12.1% | [6.9%, 20.4%] |
| S9 alt: any patient-level positive count (looser reading of protocol §8) | 45/91 | 49.5% | [39.4%, 59.5%] |

Interval practice (n = 91): **none 48**, unspecified-method CI 30, sd across folds 11,
subject-clustered CI 2 (41357810 main, 42225843 R2). **More than half the papers report a point
estimate with no uncertainty at all.**

`n_positive_reported`: patients_only 34, slices_only 24, neither 22, patients_and_slices 11.
`code_availability`: not_stated 74, public link stated 12, public link verified working 4, on
request 1.

Two readings of S9 exist because protocol §8 and the codebook word it differently — §8 says "reports
n positive patients and not only n positive slices", the codebook pins it to the enum level
`patients_and_slices`. **The codebook's operational definition is primary; the looser reading is
given alongside so no reader can be surprised by the gap.**

---

## 4. AGREEMENT — the table for the paper

Reported for every pre-specified agreement field, both pre- and post-reconciliation, from
`paper/screen/analysis/adjudication_out.json`. **Fleiss' κ is primary** (four raters); the six
pairwise Cohen's κ are in §1.2. Raw agreement and Gwet's AC1 were pre-specified in §6 **before any
coding**, against the kappa paradox, and they earned it: `headline_unit` and
`positional_distribution_reported` on the six-paper restricted set give 5/6 unanimous, raw 91.7%,
AC1 0.909 and **κ = −0.043**.

| field | (1) pre v1.0 raw / κ / AC1 | (2) counterfactual v1.2 raw / κ / AC1 |
|---|---|---|
| **P1 zero-image baseline flag** | **65.6% / −0.015 [−0.164, 0.120] / 0.479** | **95.6% / 0.932 [0.777, 1.000] / 0.934** |
| P1 flag, collapsed TRUE vs not | — | 100% / 1.000 / 1.000 |
| final_inclusion | 86.7% / 0.785 [0.544, 1.000] / 0.807 | 86.7% / 0.785 / 0.807 |
| fulltext_obtained | 88.9% / 0.769 [0.484, 1.000] / 0.786 | 88.9% / 0.769 / 0.786 |
| evaluation_unit_reported | 76.7% / 0.685 [0.465, 0.828] / 0.714 | 87.8% / 0.816 [0.565, 1.000] / 0.859 |
| headline_unit | 76.7% / 0.425 [0.085, 0.683] / 0.607 | 87.8% / 0.762 [0.473, 1.000] / 0.836 |
| **split_unit** | 64.4% / 0.498 [0.267, 0.692] / 0.586 | **76.7% / 0.637 [0.430, 0.824] / 0.722** |
| positional_distribution_reported | 82.2% / 0.648 [0.324, 0.870] / 0.762 | 87.8% / 0.783 [0.555, 1.000] / 0.850 |
| six-subflag vector, restricted to the 6 core papers | 100% / 1.000 / 1.000 | 100% / 1.000 / 1.000 |

**Verdict on the floor: MET, under (2), the encoding the protocol nominates.** κ 0.932 ≥ 0.60 and raw
95.6% ≥ 0.90.

**What this does not license, stated in the same breath:** (2) is a re-expression of the same four
sealed files under two amendments that add a missing level and cannot change a reading; no screener's
reading of any paper was altered, and `screen_recoded.json._sealed_files_modified` is `false`. It
answers "was the failure a codebook defect or a screener failure?" — it is not an independent
re-rating. The reserve blocks are single-coded. **The claim is capped at: the codebook defect is
fixed and the fix is verified on the 15 overlap papers; the reliability of the amended codebook in
fresh hands is untested.**

---

## 5. EXPLORATORY SUBGROUPS — labelled exploratory, no tests, no multiplicity correction

`pooled_final.json` → `subgroups_exploratory_pre_registered`. Pre-specified strata only
(protocol §8). P1 is 0/n in **every** stratum, so the informative column is the censoring rate.

### 5.1 Publication year

| stratum | included | P1 complete case | S6 unreachable |
|---|---|---|---|
| 2019–2022 | 33 | 0/33 = 0.0% [0.0%, 10.4%] | 9/42 = 21.4% [11.7%, 35.9%] |
| 2023–2026 | 58 | 0/58 = 0.0% [0.0%, 6.2%] | 35/93 = **37.6%** [28.5%, 47.8%] |

Censoring is **worse in the recent literature**, which is the half a reviewer will care most about.

### 5.2 Venue type — PROVISIONAL, not the pre-specified classification

| stratum | included | P1 complete case | S6 unreachable |
|---|---|---|---|
| clinical / radiology | 55 | 0/55 = 0.0% [0.0%, 6.5%] | 31/86 = **36.0%** [26.7%, 46.6%] |
| engineering / computing | 23 | 0/23 = 0.0% [0.0%, 14.3%] | 3/26 = **11.5%** [4.0%, 29.0%] |
| unclassified | 13 | 0/13 = 0.0% [0.0%, 22.8%] | 10/23 = 43.5% [25.6%, 63.2%] |

**Caveat that must travel with this table:** protocol §8 requires classification from each journal's
scope statement *before* unblinding. That was not done. This is a keyword heuristic over journal
names with a 23-paper `unclassified` bucket
(`subgroups_exploratory_pre_registered._venue_class_caveat`). The 3× gap in censoring is nonetheless
the mechanism protocol §7 predicted in advance.

### 5.3 Modality, dataset provenance, evaluation unit (included papers only)

| modality | n | P1 | | dataset | n | P1 |
|---|---|---|---|---|---|---|
| CT | 46 | 0/46 [0.0%, 7.7%] | | private | 49 | 0/49 [0.0%, 7.3%] |
| MRI | 34 | 0/34 [0.0%, 10.2%] | | public | 33 | 0/33 [0.0%, 10.4%] |
| PET/CT | 4 | 0/4 | | mixed | 9 | 0/9 |
| OCT | 4 | 0/4 | | | | |
| multiple | 2 | 0/2 | | | | |
| CBCT | 1 | 0/1 | | | | |

By evaluation unit: patient 29, unclear 19, slice 13, lesion 12, volume-not-patient 9, both 6,
other 3 — **P1 = 0 in all seven.** These three cuts are included-only and carry no S6 column, because
a paper nobody could open has no coded modality, provenance or evaluation unit; inventing a
`not_applicable` stratum for them would manufacture a 100%-censored cell.

---

## 6. BENCHMARKS — how many rows meet all three requirements

Source: `paper/audit_results.md` §2.1, §2.3, §8 item 7; per-run cards in
`pipeline_out/trivial_baselines/*.json`; logs in `pipeline_out/audit_logs/`.

The three requirements: **(a)** the audit runs from a pixel-free published label file;
**(b)** the comparator is **peer-reviewed** and on the **same metric** and the **same unit**;
**(c)** the split condition is adequate or an explicit approximation statement is made.

### 6.1 The count, stated plainly

**Four benchmarks meet all three requirements: RSNA 2019 ICH, LUNA16, DeepLesion, PI-CAI**
(12 scored rows). That is up from three, because RSNA ICH was reached this run.

**Zero of those four reach the MATCHED verdict.**

| benchmark | comparator | venue | rows | verdict | trivial fraction |
|---|---|---|---|---|---|
| RSNA 2019 ICH | Burduja, Ionescu & Verga, *Sensors* 2020;20(19):5611 | MDPI journal, PubMed-indexed | 16–21 | **PARTIAL** | 0.398–0.615 |
| DeepLesion | Yan et al., CVPR 2018 | IEEE/CVF proceedings (conference, not journal) | 7–9 | **PARTIAL** | 0.480–0.889 |
| LUNA16 | Setio et al., *Medical Image Analysis* 2017;42:1–13 | Elsevier journal | 10 | **NOT MATCHED** | ≈0 |
| PI-CAI | Saha et al., *Lancet Oncol* 2024;25:879–887 | journal | 11–12 | **NOT MATCHED** | 0.467, 0.532 |
| *fastMRI Prostate* | *Rempe et al., arXiv:2407.06165* | ***arXiv preprint*** | *1–6* | *MATCHED ×6* | *0.973–1.655* |

**The precondition named in `PAPER_PLAN.md` §3.3 item 1 — "three or more independent benchmarks
MATCHED against peer-reviewed published numbers on the same metric and unit" — is not met.
Under a strict reading the count is zero, not one.** The count of *valid peer-reviewed
comparisons* is four; the count of *matches* among them is zero. Both numbers must appear;
reporting only the four is the over-claim this file exists to prevent.

Rempe et al. was re-queried on the arXiv API on 2026-07-29: no `journal_ref`, no DOI, no Europe
PMC record, two years after posting (`audit_results.md` §2.3). **Never present a MATCHED row
without saying its comparator is a preprint, in the same sentence.**

### 6.2 What the peer-reviewed comparisons *do* support

Not "matches", but a quantity:

> A pixel-blind model reaches **40–62%** of the margin over chance on RSNA ICH, **48–89%** on
> DeepLesion, and essentially none on LUNA16 and PI-CAI.

### 6.3 The result that needs no comparator at all — slice vs patient

`audit_results.md` §4. Every cell is our own computation on a published label file, so no
comparability objection touches it.

| dataset | zero-image positional, slice AUROC | patient AUROC | position-stratified slice AUROC (remedy) |
|---|---|---|---|
| **RSNA ICH, any haemorrhage** | **0.731 [0.723, 0.739]** | **0.462 [0.431, 0.491]** | — |
| fastMRI Prostate T2 | 0.854 [0.812, 0.891] | 0.506 [0.381, 0.632] | 0.546 (5 strata) |
| fastMRI Prostate DWI | 0.851 [0.816, 0.887] | 0.424 [0.298, 0.547] | 0.539 (6 strata) |
| fastMRI+ knee, meniscus tear | 0.873 [0.858, 0.886] | 0.510 [0.428, 0.592] | — |
| fastMRI+ knee, any finding | 0.801 [0.779, 0.824] | 0.558 [0.470, 0.648] | — |
| Duke breast, owner slice task | 0.823 [0.811, 0.834] | undefined (all 922 patients positive) | — |
| DeepLesion, pelvis vs rest | 0.977 [0.969, 0.984] | 0.954 [0.939, 0.967] | — |
| LUNA16 candidates | 0.534 [0.514, 0.558] | 0.581 [0.538, 0.613] | — |
| PI-CAI (case level) | n/a | 0.692 [0.626, 0.755] metadata only | — |

**RSNA ICH is the flagship**: 752,802 slices, 18,938 patients, on the benchmark whose official
metric is per-slice and whose organisers stated on the record that the released fields could
not determine whether an image contains haemorrhage — against a slice AUROC of 0.738 on their
own label file (row 16). The other collapses are on 46–199 subjects.

**Two exceptions reported at equal prominence.** DeepLesion does not collapse (its labels are
anatomical regions, so they *are* patient-level facts). LUNA16 is at chance at both units.

**One honest qualification** (`audit_results.md` §4): the ICH collapse is bin-robust at the
slice level (0.707→0.740 over 5→50 bins) but **not** at the patient level, where the sweep runs
5→0.425, 10→0.435, 20→0.462, 50→**0.652**. At 50 bins the bin index approximates the raw slice
index and the patient aggregate starts tracking volume length, which is itself weakly
predictive (0.599 patient AUROC alone). The 20-bin setting is pre-specified and is what the
table reports. Say this in the results, not the limitations.

### 6.4 The release-practice pair

- **RSNA ICH**: slice ordering not in the official release; recovered from a public
  MIT-licensed pixel-free mirror and verified by a falsifiable run-length test
  (`audit_results.md` §3.7, `pipeline_out/audit_logs/rsna_ich_prep.log`).
- **RSNA-STR PE 2020**: official `train.csv` obtained and verified genuine, then *measured* to
  carry no positional information at all — run-length ratio 0.974 and 1.001 against random
  (`audit_results.md` §3.8, `pipeline_out/audit_logs/rsna_pe_position_test.log`). A benchmark
  can publish a per-slice metric and a per-slice label file and still make the slice
  unlocatable.

Report these two together as one finding about release practice.

---

## 7. The anchor result (unchanged, do not re-derive)

`pipeline_out/trivial_baselines/fastmri_prostate_t2_published.json`,
`fastmri_prostate_dwi_published.json`; tool `pipeline/s14_trivialbaselines.py` (5 baselines,
53 self-tests, permutation-calibrated).

Binned P(label | relative slice position), fitted on train, applied to test, no pixels, on
Rempe et al.'s own published label file and their own patient-disjoint split:

| arm | slice AUROC | patient AUROC | published | trivial fraction |
|---|---|---|---|---|
| T2 | 0.854 [0.812, 0.891] | 0.506 [0.381, 0.632] | 0.861 | 0.981 [0.865, 1.084] |
| DWI | 0.851 [0.816, 0.887] | 0.424 [0.298, 0.547] | 0.809 | 0.973 [0.876, 1.073] |

98% of the margin over chance needs no pixels. Their split **is** patient-disjoint, so this is
not a train/test leakage result. Comparator is a preprint.

---

## 8. Prior art — cite, never re-present as ours

Established in `paper/audit_results.md` §5 and the project brief. The construction is prior
art; our contribution is formalisation, statistics, the patient-level contrast, prevalence, and
the released tool.

| source | what it did | what it did not do |
|---|---|---|
| **Yan et al., CVPR 2018**, "Deep Lesion Graphs in the Wild", Table 1 | published a "Baseline: Location feature" at **59.7%** 8-class accuracy vs their 90.5% — a position-only baseline in DeepLesion's own defining paper | position is *image-derived* (self-supervised body-part regressor), used as a retrieval feature, not offered as a critique of the benchmark |
| **OsciiArt**, Kaggle "Baseline with no image", RSNA-STR PE, 2020-10-10, gold medal, 5,759 views | bins P(PE \| relative slice location) on train, applies to test with no pixels; 0.33 public LB vs 0.44 constant. **The same construction, four years earlier** | no AUROC, no CI, no patient-level contrast, no claim about anyone's published performance |
| **Tomoo Inubushi**, Kaggle, RSNA ICH, 2019 | showed relative slice position distribution differs by class (sd 0.314 normals vs 0.187 positives) | no classifier, no metric |
| **RSNA ICH organisers, 2019** | on the record: "the available fields do not contain information that can determine if an image contains intracranial hemorrhage" — the fields include `ImagePositionPatient` | the best quotable foil in the paper; now refuted with a number (0.738 slice AUROC) |
| Badgeley 2019 *npj Digit Med*; Zech 2018; Geirhos 2020; Yagis 2021; Tampu 2022; Wen 2020 | shortcut learning / leakage, established | none quantify the positional null under a correct split |
| **Ong Ly et al. 2024 *npj Digit Med*** | 13 datasets, up to 20% overestimation from acquisition shortcuts, released estimator (PEst) | **closest competitor.** Does not address the slice-vs-patient unit collapse and requires pixel access |

Handle the ICH organisers' sentence fairly: it is defensible as a claim about *individual*
certainty and indefensible as a claim about *aggregate rankability*. Say so.

---

## 9. Reproducing every number in this file

```bash
PY=/Users/sathvikloke/Downloads/PhaseDx/venv/bin/python

# FINAL pooled prevalence, flow, agreement, subgroups  -> paper/screen/analysis/pooled_final.json
$PY /Users/sathvikloke/Downloads/PhaseDx/paper/screen/analysis/pool_final.py

# the pooled flow figure                                -> paper/figures/prisma_flow_pooled.svg
$PY /Users/sathvikloke/Downloads/PhaseDx/paper/screen/analysis/flow_figure.py

# agreement, pre- and post-amendment                    -> paper/screen/analysis/adjudication_out.json
$PY /Users/sathvikloke/Downloads/PhaseDx/paper/screen/analysis/adjudicate.py

# the frame and the permutation, offline digest check
$PY /Users/sathvikloke/Downloads/PhaseDx/paper/screen/reproduce_frame.py --verify
```

| artefact | file | status |
|---|---|---|
| **FINAL pooled prevalence, flow, agreement, subgroups** | `paper/screen/analysis/pooled_final.json` | **authoritative** |
| final analysis script | `paper/screen/analysis/pool_final.py` | authoritative |
| pooled flow figure | `paper/figures/prisma_flow_pooled.svg` (+ `flow_figure.py`) | authoritative |
| analysis-sample re-code under v1.2 (100 papers, 145 records) | `paper/screen_recoded.json` / `.md` | authoritative for positions 1–100 |
| reserve blocks, pre-registered | `paper/screen_reserve_R{1,2,3}.json` / `.md` | authoritative |
| reserve block, **post-hoc** | `paper/screen_reserve_R4.json` / `.md` | label as post-hoc wherever used |
| agreement, pre- and post-amendment | `paper/screen/analysis/adjudication_out.json` | authoritative |
| adjudication audit trail, rules D1–D14 | `paper/screen_adjudication.md` | authoritative |
| access-recovery record | `paper/screen/access_recovery.json` | authoritative for *which* texts were recovered |
| four sealed screener files, never edited | `paper/screen_batch_{A,B,C,D}.json` | frozen |
| pre-registered codebook v1.2 | `paper/screen_frame.json` | frozen + changelog |
| pre-registered protocol v1.3 | `paper/screen_protocol.md` | frozen + changelog |
| sampling frame + permutation | `paper/screen/frame_meta.json`, `frame_pmids.txt`, `permutation.txt` | frozen |
| ~~main-sample flow, v1.0/v1.2/v1.3~~ | ~~`paper/prisma_flow.json` / `.md`, `paper/figures/prisma_flow.svg`~~ | **SUPERSEDED — reports 16 unreachable / 54 eligible / 29.6%; does not apply rule D1. Do not cite. See §1.1.** |
| ~~pre-adjudication pooled analysis~~ | ~~`paper/screen/analysis/analysis_out.json`~~ | superseded by `pooled_final.json`; retained as the v1.0 record |
| benchmark audit, all 21 rows | `paper/audit_results.md` | authoritative |
| per-benchmark cards | `pipeline_out/trivial_baselines/*.json`, `*.md` | authoritative |
| the tool | `pipeline/s14_trivialbaselines.py`, packaged as `trivialbaselines/` | released |

---

## 10. Open items, in the order they change the paper's tier

1. **Attack the 44 unreachable full texts.** This is now the *only* lever on the headline
   (§1.1). One screener with institutional access, or the Europe PMC render-PDF rung applied to
   all 44, or protocol rung 5 (author requests, 21-day clock). Getting censoring below 15% would
   promote the complete-case 0/91 = 0.0% [0.0%, 4.1%] to the headline and turn the paper's
   central claim from a bound into an estimate. **No further screening can do this.** Nothing
   else on this list moves the paper as much.
2. **Run a fresh independent four-screener re-code under v1.2** on a new 15-paper overlap set
   (§1.2). Until then the post-amendment κ is a re-encoding, and a methods reviewer will say so.
3. **Execute the protocol's §6 20% within-batch cross-check.** Disagreement outside the overlap
   set is currently invisible; the reserve blocks are entirely single-coded.
4. **Regenerate or retract `paper/prisma_flow.json` / `.md` / `figures/prisma_flow.svg`** so the
   repository does not carry two incompatible flows (§1.1). `prisma_flow.py` must apply D1.
5. **Classify venues from scope statements**, as protocol §8 requires, replacing the keyword
   heuristic in §5.2 and its 23-paper `unclassified` bucket.
6. **A peer-reviewed anchor for a MATCHED row** (§6.1). Either Rempe et al. gets published, or a
   fourth benchmark with a peer-reviewed comparator produces a match.
7. **Deposit the pre-registration on OSF** with the freezing commit hash, as protocol §11 says
   should have happened before screening began. It did not, and that is a reportable deviation.
8. **Re-audit prior art against CVPR / MICCAI / MIDL / ML4H** (`audit_results.md` §8). If Yan
   et al. 2018 exists, others likely do.
