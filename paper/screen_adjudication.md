# Adjudication round — the pre-registered agreement remedy

**Date: 2026-07-29. Protocol v1.1 → v1.2. Codebook `screen_frame.json` v1.0 → v1.2.**

This document exists because a pre-registered threshold was breached and the pre-registered
remedy became mandatory. It is written so that a reader who trusts none of our judgement can
check every one of them: for each of the fifteen overlap papers it lays out what all four
screeners coded, where they diverged, what the adjudicated code is, and **which rule decides
it** — an existing rule where one existed, a new rule where the codebook was silent.

Nothing in the four sealed screener files was edited. They were read; they were not written.
The recomputation is `paper/screen/analysis/adjudicate.py` → `screen/analysis/adjudication_out.json`.

---

## 0. Why this round happened

`screen_frame.json` → `agreement.threshold_and_remedy`, fixed before any coding:

> If Fleiss' kappa on the P1 flag is below 0.60 (or, if the paradox guard triggers, if raw
> agreement is below 90%): a documented adjudication round is held, the codebook is amended in
> the changelog, and EVERY already-coded paper is re-coded under the amended codebook.

Pre-reconciliation, over the fifteen overlap papers at permutation positions 1–15:

| statistic | value | floor | met? |
|---|---|---|---|
| Fleiss' κ, P1 flag | **−0.015** [−0.164, 0.120] | 0.60 | **no** |
| raw pairwise agreement, P1 flag | **65.6%** [50.0, 80.0] | 90% (paradox guard) | **no** |
| Gwet's AC1, P1 flag | 0.479 [0.119, 0.740] | — | — |
| unanimous items | 6/15 | — | — |

Both floors failed. The remedy fired.

---

## 1. Where the disagreement actually was

The task brief's hypothesis was that the divergence sits in **reachability and eligibility**
rather than in the baseline flag. The codes confirm it, and more sharply than expected.

### 1.1 Nobody disagreed about the primary flag. Nobody could have.

**Across all 145 coded records, not one screener coded `true` on any of the four zero-image
sub-flags** — `constant_or_prevalence`, `positional`, `acquisition_metadata`,
`permuted_or_shuffled_label`. The set of true codes in the P1 family is empty. There is no
paper on which two screeners looked at the same evidence and reached opposite conclusions about
whether a zero-image baseline was reported.

Restricted to the **six** overlap papers all four screeners both *obtained* and *included* —
PMIDs 36776294, 42130124, 39061744, 31093705, 36016875, 41068276, the only papers on which a
`trivial_baseline` code is even defined:

| field | raw agreement | Fleiss' κ | Gwet's AC1 | unanimous |
|---|---|---|---|---|
| P1 flag | **100%** | n/d (one category) | **1.000** | 6/6 |
| **all six sub-flags as a vector** | **100%** | — | — | **6/6** |
| `evaluation_unit_reported` | 91.7% | 0.892 | 0.897 | 5/6 |
| `headline_unit` | 91.7% | **−0.043** | 0.909 | 5/6 |
| `positional_distribution_reported` | 91.7% | **−0.043** | 0.909 | 5/6 |
| `split_unit` | 63.9% | 0.464 | 0.534 | 2/6 |

Only two distinct six-flag vectors appear across 24 screener-paper cells: all-false (20 cells)
and clinical-only-true (4 cells, all of them PMID 39061744, on which all four screeners agree).

Two of those rows are the kappa paradox in miniature — 5/6 unanimous, raw 91.7%, AC1 0.909, and
κ = −0.043. Pre-specifying AC1 and raw agreement in §6 *before* any coding is what makes those
rows readable rather than alarming.

### 1.2 The measured disagreement was a placeholder collision

`trivial_baseline` was declared as six **booleans**. The codebook gave no level for *"this
record's full text was never obtained, so the mandatory 14-term search could not be run"*, and
no level for *"this record is excluded, so the field does not exist"*. Four screeners
independently invented four conventions:

| screener | what it wrote when the flag could not be assessed |
|---|---|
| S1 | `false`, with `searches_run` = "NOT RUN — full text unreachable" |
| S2 | `null`, with "an unevidenced negative on P1 is not accepted by the codebook" |
| S3 | the string `"unclear"` |
| S4 | `false`, with `searches_run` = "all 14 terms run over title+abstract ONLY" |

The pre-reconciliation pairwise agreement matrix on the P1 flag is the fingerprint of that
collision, not of any disagreement about papers:

| pair | raw | Cohen's κ |
|---|---|---|
| S1–S4 | **100%** | undefined (both wrote one category throughout) |
| S2–S3 | 66.7% | 0.390 |
| S1–S2, S2–S4 | 73.3% | 0.000 |
| S1–S3, S3–S4 | **40.0%** | 0.000 |

S1 and S4 agree perfectly with each other and badly with S3 — because S1 and S4 share a
convention and S3 does not. That is a property of the form, not of the literature.

This divergence occurs on **9 of the 15** overlap papers: 41617832, 39423605, 36789248,
40335658, 40194851, 42489954, 36072854, 37222638, 40239684. Every one is either unreachable or
excluded. **Confirmed: the disagreement is about reachability and eligibility.**

---

## 2. The fourteen rules

Full text of each is in `screen_frame.json` → `eligibility.ambiguous_case_rules`. D1–D9 and D14
close gaps where codebook v1.0 **did not determine an answer** — those divergences are codebook
defects, not screener errors. D10–D13 correct or complete a rule that already existed.

| id | what it fixes | resolves |
|---|---|---|
| **D1** | unreachable **dominates** included; an unreachable record may be `excluded` only if `stage1_decision='exclude'` | 41617832, 37222638, 42489954 |
| **D2** | `trivial_baseline` sub-flags become **three-valued**; TRUE may rest on an abstract quote *carrying the value*, FALSE requires the full-text search, otherwise `not_assessable` | the whole P1-flag disagreement |
| **D3** | new level `not_applicable` on every descriptive field, available **only** where `final_inclusion≠included` | 9 papers, all descriptive fields |
| **D4** | access ladder is not climbed for a stage-1 exclusion | 40335658, 40194851, 40239684 |
| **D5** | Methods/Results **govern over the Abstract** on a factual contradiction | 36016875 |
| **D6** | `patient_subject` needs a patient-naming noun *in the splitting sentence*; word list given | 42130124 |
| **D7** | a unit named in a table caption **is** a named unit | 41068276 |
| **D8** | new `split_unit` level `lesion_or_roi`; a split deferred to an unsampled companion paper is `unclear` | 36776294, 31093705 |
| **D9** | the positional-distribution test is the **ordering of the classification unit**, not the vocabulary | 42130124 |
| **D10** | E-SEG's "no categorical class decision" qualifier **binds**; a human-reader class decision gives E-NOCLF | 40335658 |
| **D11** | trigger stated for `stage1_decision='include_provisional'` | S3's five papers |
| **D12** | `modality` is the modality of the **input** | 40335658 |
| **D13** | new `code_availability` level `not_stated` | defect found in passing |
| **D14** | `headline_unit='unclear'` only when ≥2 units are reported | 41068276 |

### 2.1 The amendment does not buy agreement with permissiveness

Three guards, all checkable:

- **`not_applicable` is unavailable on an included record.** On an included record `unclear`
  remains the only answer for an undeterminable field, and `unclear` is still reported as its
  own category, never merged, never imputed. D3 *narrows* the admissible codes on exactly the
  records that enter an endpoint.
- **D2 makes a negative harder and a positive no easier.** Coding a sub-flag `false` now
  requires the full-text search to have been run; the placeholder-`false` that S1 and S4 used on
  unreachable records is no longer a legal code. D2's asymmetry cuts both ways in practice: it
  *admits* PMID 37222638's clinical-only flag on an abstract quote carrying three measured AUCs,
  and it *refuses* PMID 41617832's, whose abstract names a clinical arm but reports no value for
  it.
- **Two rules move our own headline numbers in opposite directions.** D6 takes a paper out of
  S4; D9 puts a paper into S5. Neither was chosen for its direction.

### 2.2 A rule that makes the literature look better, adopted anyway

**D9 takes endpoint S5 from 0/35 to 1/35.** S5 is the proportion of papers reporting the
positional distribution of labels — the endpoint on which this paper accuses the literature of
silence, and the construction our own zero-image baseline exploits. PMID 42130124's Table 2
tabulates fracture counts against vertebral level (L1 57/16, L2 53/12, L3 45/7, L4 29/3, L5
16/2) while the **vertebra is the scored unit**, so the table is literally the distribution of
the label over the ordered unit index. Screener S1 coded it `figure_or_table` and was outvoted
3–1; S3 recorded the same tension in its own note and coded `no` anyway. S1 was right. The
adjudicated code is `figure_or_table` and S5 is no longer zero.

---

## 3. Paper by paper

`fulltext` abbreviated: **OA** = `oa_pmc_or_publisher`, **UP** = `unreachable_paywalled`,
**NA1** = `not_attempted_excluded_at_stage1`. Screener order is S1 | S2 | S3 | S4 throughout.

### pos 1 — PMID 36776294 · CT lymph-node metastasis · **included**

Unanimous on every agreement field: `lesion` / `na_only_one_unit_reported` / `slice_or_image` /
`no`, all six sub-flags false, full text obtained by all four.

**Adjudicated:** included, OA, P1 false, `lesion`, `na_only_one_unit_reported`, **`lesion_or_roi`**, `no`.
**Rule:** D8. All four had written `slice_or_image` because `split_unit` had no lesion level —
S1 logged the forced mapping in its own file header. Unanimous before and after; the change
buys no agreement, it buys accuracy.

### pos 2 — PMID 41617832 · CT + laryngoscopy, LSCC · **unreachable**

| field | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| `final_inclusion` | **included** | unresolved | unresolved | **included** |
| `evaluation_unit` | patient | patient | **unclear** | patient |
| `split_unit` | patient_subject | patient_subject | **unclear** | patient_subject |
| `clinical_or_demographic_only` | **true** | null | "unclear" | **true** |

All four exhausted rungs 1–4; Springer served abstract only.

**Adjudicated:** `unreachable_eligibility_unresolved`, UP, P1 `not_assessable`, all descriptive
fields `not_applicable`, **`clinical_or_demographic_only` = `not_assessable`, not true**.
**Rules:** D1, D2, D3.
The eligibility call is *forced by a rule the codebook already had*: an `included` record must
carry an evidenced `trivial_baseline` code, and the mandatory search cannot be run without the
full text. On the clinical flag, S1 and S4 over-coded: the abstract names "a clinical logistic
regression model [CL]" and says the fused model outperformed all single-modality models
(p < 0.05), but gives **no AUC for the CL arm**. The codebook already says an assertion with no
number is false for every flag; D2 makes that `not_assessable` rather than false. S2 and S3
correct.

### pos 3 — PMID 39423605 · MRI + speech, Alzheimer's · **unreachable**

Unanimous `unreachable_eligibility_unresolved`, unanimous `unclear` on every descriptive field.
The only divergence was the sub-flag placeholder.

**Adjudicated:** unresolved, UP, P1 `not_assessable`, descriptive fields `not_applicable`.
**Rules:** D2, D3. All four independently noted the same unresolved question — whether
"pre-processing, feature extraction and detection" means a spatially resolved image ever reaches
the network, i.e. whether this is E-DERIV. It stays unresolved.

### pos 4 — PMID 42130124 · CT lumbar vertebral fracture · **included** · *two overturns, opposite directions*

| field | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| `split_unit` | **patient_subject** | **patient_subject** | random_unit_not_stated | random_unit_not_stated |
| `positional_distribution` | **figure_or_table** | no | no | no |

**Adjudicated:** included, OA, P1 false, `other`, `na_only_one_unit_reported`,
**`random_unit_not_stated`**, **`figure_or_table`**.

- **`split_unit` → `random_unit_not_stated` (D6).** The paper says "10% of all **cases** were
  randomly selected". S1 and S2 upgraded on the strength of a per-case demographics table
  elsewhere. The codebook's own CRITICAL DISTINCTION already forbade exactly that
  ("do NOT upgrade this to patient-level because the sentence mentions patients elsewhere");
  D6 makes it operational with a word list. **S3 and S4 correct.** This removes a paper from
  S4's numerator — it makes the literature look *worse*.
- **`positional_distribution` → `figure_or_table` (D9).** See §2.2. **S1 correct, three
  screeners overturned.** This adds a paper to S5's numerator — it makes the literature look
  *better*.

Also of record: the paper's internal counts do not reconcile (240 cases vs "216 L-OVCF
patients" vs a 204-vertebra test confusion matrix), noted independently by S2 and S3.

### pos 5 — PMID 36789248 · CT · **excluded, E-DERIV** (unanimous)

Divergence only in what was written into fields that do not exist on an excluded record:
`evaluation_unit` `unclear` | slice | slice | slice; `split_unit` `random_unit_not_stated` |
`slice_or_image` | `random_unit_not_stated` | `random_unit_not_stated`.

**Adjudicated:** excluded (E-DERIV), OA, everything else `not_applicable`. **Rule:** D3.

### pos 6 — PMID 40335658 · MRI→synthetic CT, cervical spine · **excluded** · *exclusion code overturned*

| field | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| `exclusion_code` | E-SEG | **E-NOCLF** | E-SEG | E-SEG |
| `fulltext_reachable` | UP | **NA1** | UP | UP |
| `modality` | MRI | MRI | MRI | **multiple** |

**Adjudicated:** excluded, **E-NOCLF**, **NA1**, modality MRI, everything else `not_applicable`.
**Rules:** D10, D4, D12, D3.

S2 was right, and its reasoning is worth preserving verbatim in substance: E-SEG is qualified
*"with NO categorical class decision evaluated"*, and a categorical decision (AO Spine class)
**was** evaluated — by five human readers. E-SEG's qualifier therefore fails, and the first code
that actually applies is E-NOCLF, "purely descriptive or reader study with no model". Three
screeners applied the first-in-listed-order rule mechanically without checking that E-SEG's own
condition held. The ordering rule is unchanged; D10 fixes which codes *apply*.

### pos 7 — PMID 40194851 · MRI · **excluded, E-DERIV** (unanimous)

Divergence: `fulltext_reachable` UP | NA1 | UP | UP, and `positional_distribution` `unclear` |
no | `unclear` | no.
**Adjudicated:** excluded (E-DERIV), NA1, everything else `not_applicable`. **Rules:** D4, D3.

### pos 8 — PMID 42489954 · structural MRI, graph attention network · **unreachable** · *the one genuine survivor*

| field | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| `stage1_decision` | go_to_fulltext | go_to_fulltext | go_to_fulltext | go_to_fulltext |
| `final_inclusion` | **excluded (E-DERIV)** | unresolved | unresolved | **excluded (E-DERIV)** |
| `screener_confidence` | low | low | low | low |

**Adjudicated:** `unreachable_eligibility_unresolved`, UP, P1 `not_assessable`, everything else
`not_applicable`. **Rules:** the *existing* stage-1 rule, plus D1 and D11.

This one needs no new rule. Codebook v1.0 already permitted a stage-1 exclusion "ONLY when an
exclusion code is unambiguous from the abstract" — and **all four screeners coded
`stage1_decision='go_to_fulltext'`**, i.e. all four recorded that it was not unambiguous. S1 and
S4 then excluded anyway, both at low confidence, and both asked in their own notes for exactly
this adjudication: S1 wrote "LOW CONFIDENCE / ADJUDICATION REQUESTED", S4 wrote "the single most
likely disagreement in my overlap set". The abstract's "dual-branch graph attention network to
extract complementary global statistical and local topological features from structural MRI"
points hard at E-DERIV, but a graph attention network can also run over spatially resolved
patches. It stays unresolved and it is a priority for the access remedy.

**This is the only P1-flag cell that remains non-unanimous after the counterfactual re-encoding
(S1/S4 `not_applicable` vs S2/S3 `not_assessable`), and it should be: it is a real eligibility
disagreement, and the transform is designed to preserve real disagreements.**

### pos 9 — PMID 39061744 · CT · **included** (unanimous on everything)

`patient` / `na_only_one_unit_reported` / `patient_subject` / `no`; all four coded
`clinical_or_demographic_only = true` and all four P1 sub-flags false. No rule needed.
The only divergence in the record was `screener_confidence` (medium | high | high | high).

### pos 10 — PMID 31093705 · MRI hepatic lesions, Part II of two · **included**

`split_unit`: **`slice_or_image`** | `unclear` | `unclear` | `unclear`.

**Adjudicated:** included, OA, P1 false, `lesion`, `na_only_one_unit_reported`, **`unclear`**,
`no`. **Rule:** D8(b). The new `lesion_or_roi` level exists, but this paper states no split unit
of its own — it defers to Part I (PMID 31016442), which is not in the sample. Importing a method
from an unsampled paper is not screening, and would make the coded unit depend on which half of
a two-part paper the permutation happened to draw. **S2, S3, S4 correct.**

### pos 11 — PMID 36016875 · CT temporal bone, cholesteatoma · **included** · *Abstract vs Methods*

| field | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| `evaluation_unit` | **patient** | slice | slice | slice |
| `split_unit` | **patient_subject** | slice_or_image | slice_or_image | slice_or_image |

The Abstract says "70% of cases for training and 30% of cases for validation". The Methods split
85/15 over n = 2,070 and n = 388 units — 2,458 in all, against **119 patients**. The two
statements cannot both be true, and the arithmetic settles it: 2,458 units is not a patient-level
split.

**Adjudicated:** included, OA, P1 false, **`slice`**, `na_only_one_unit_reported`,
**`slice_or_image`**, `no`. **Rule:** D5, a rule the codebook lacked entirely (A5 governs only
which *number* is the headline, not which section wins a factual contradiction). **S2, S3, S4
correct.**

Direction, stated plainly: this moves a paper *into* the slice-level stratum, which flatters our
thesis. It is adopted because the arithmetic forces it, and D5 is written direction-neutrally so
it will fire the other way when the other way is correct.

### pos 12 — PMID 36072854 · CT COVID-19 · **excluded, E-SEG** (unanimous)

Divergence only in the undefined fields: `split_unit` `unclear` | `slice_or_image` | `unclear` |
**`patient_subject`** — S4 recorded the paper's real split from the full text, which is correct
information in a field that carries no meaning on an excluded record.
**Adjudicated:** excluded (E-SEG), OA, everything else `not_applicable`. **Rule:** D3.
Worth keeping: all four flagged this as the trap the codebook warns about — the abstract
promises to "classify, identify, and segment" and reports an "accuracy", but the accuracy is
per-pixel.

### pos 13 — PMID 37222638 · MRI placenta accreta spectrum · **unreachable** · *a positive survives*

| field | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| `final_inclusion` | **included** | unresolved | unresolved | **included** |
| `split_unit` | patient_subject | **site_or_centre** | **unclear** | patient_subject |
| `clinical_or_demographic_only` | true | true | **"unclear"** | true |

**Adjudicated:** `unreachable_eligibility_unresolved`, UP, four P1 sub-flags `not_assessable`,
descriptive fields `not_applicable`, and **`clinical_or_demographic_only` = TRUE, recorded and
held pending eligibility resolution.** **Rules:** D1, D2, D3.

This is where D2's asymmetry earns its place. The abstract itself reports the measured values:
"The MRI-based DLR model had a higher area under the curve than the clinical model in three
datasets (0.880 vs. **0.741**, 0.861 vs. **0.772**, 0.852 vs. **0.675**)", where the clinical
model is "different clinical characteristics between PAS and non-PAS groups" — pixel-free, three
measured AUCs, same metric. A **positive** may rest on that; only a **negative** requires the
full-text search. S3's blanket `"unclear"` discarded evidence it had itself quoted. S1, S2 and
S4 correct on the flag. It counts toward S1, never P1.

### pos 14 — PMID 40239684 · OCT, mouse skin · **excluded, E-SEG** (unanimous)

Divergence: `fulltext_reachable` UP | NA1 | UP | NA1; `fulltext_version_used` abstract_only ×3 |
na. **Adjudicated:** excluded (E-SEG), **NA1**, everything else `not_applicable`.
**Rules:** D4, D3. All four independently noted that E-NONMED also applies and that E-SEG comes
first in the listed order — a case where the ordering rule worked exactly as intended.

### pos 15 — PMID 41068276 · Kaggle Alzheimer's MRI · **included**

| field | S1 | S2 | S3 | S4 |
|---|---|---|---|---|
| `split_unit` | **random_unit_not_stated** | slice_or_image | slice_or_image | slice_or_image |
| `headline_unit` | na_only_one | na_only_one | na_only_one | **unclear** |

**Adjudicated:** included, OA, P1 false, `unclear`, **`na_only_one_unit_reported`**,
**`slice_or_image`**, `no`. **Rules:** D7, D14.
Table 2 is headed "Distribution of MRI images by dataset split (70/15/15)" with per-class image
counts — the unit **is** named, in a table caption. `random_unit_not_stated` is for when it is
not named anywhere. On `headline_unit`, exactly one unit is reported; that it is itself
`unclear` is a different fact, answered by a different field.

---

## 4. Recomputed agreement

### 4.1 What each number is allowed to claim

- **(1) Pre-reconciliation** — the four sealed files as coded under v1.0. The honest reliability
  of the original round. Never replaced.
- **(2) Counterfactual v1.2 encoding** — the **same four sealed files**, with **no screener's
  reading of any paper altered**, re-expressed under **D2 and D3 only**: the two amendments that
  add a missing *level* and cannot change a reading. The transform is driven by each screener's
  **own** `final_inclusion` and **own** `fulltext_reachable`, so genuine eligibility
  disagreements survive it — and one does. This answers *"was the failure a codebook defect or a
  screener failure?"*, and **the floor is assessed against this.**
- **(3) Post-adjudication consensus** — one code per paper. Agreement is **1.000 by
  construction**. It is not a reliability statistic, it is not evidence of anything, and the
  floor is **not** assessed against it. Reported only so nobody mistakes it for (2).

Rules D5–D12 are deliberately **excluded** from (2). Applying an adjudicated substantive reading
to all four screeners forces them to agree by construction; folding that into a reliability
number would be circular. They appear only in (3).

### 4.2 Pre-reconciliation vs counterfactual v1.2 encoding

15 papers, 4 raters, bootstrap percentile 95% CI over 2,000 resamples, seed 20260729 — exactly
as pre-registered.

| field | raw (v1.0) | raw (v1.2) | Fleiss' κ (v1.0) | Fleiss' κ (v1.2) | AC1 (v1.0) | AC1 (v1.2) | unan. |
|---|---|---|---|---|---|---|---|
| **P1 flag** | **65.6%** [50.0, 80.0] | **95.6%** [86.7, 100] | **−0.015** [−0.164, 0.120] | **0.932** [0.777, 1.000] | 0.479 [0.119, 0.740] | **0.934** [0.800, 1.000] | 6/15 → **14/15** |
| P1 flag, not-coded collapsed | — | 100% [100, 100] | — | 1.000 [1.000, 1.000] | — | 1.000 | → 15/15 |
| `evaluation_unit_reported` | 76.7% [63.3, 90.0] | 87.8% [74.4, 100] | 0.685 [0.465, 0.828] | 0.816 [0.565, 1.000] | 0.714 | 0.859 | 8 → 12/15 |
| `headline_unit` | 76.7% [63.3, 90.0] | 87.8% [74.4, 100] | 0.425 [0.085, 0.683] | 0.762 [0.473, 1.000] | 0.607 | 0.836 | 8 → 12/15 |
| `split_unit` | 64.4% [48.9, 80.0] | 76.7% [62.2, 92.2] | 0.498 [0.267, 0.692] | 0.637 [0.430, 0.824] | 0.586 | 0.722 | 6 → 9/15 |
| `positional_distribution_reported` | 82.2% [67.8, 93.3] | 87.8% [75.6, 100] | 0.648 [0.324, 0.870] | 0.783 [0.555, 1.000] | 0.762 | 0.849 | 10 → 12/15 |
| `final_inclusion` | 86.7% [73.3, 100] | 86.7% (untouched) | 0.785 [0.544, 1.000] | 0.785 | 0.807 | 0.807 | 12/15 |
| `fulltext_obtained` | 88.9% [75.6, 100] | 88.9% (untouched) | 0.769 [0.484, 1.000] | 0.769 | 0.786 | 0.786 | 12/15 |

`final_inclusion` and `fulltext_obtained` are deliberately **unchanged** — D1 and D4 would move
them, and both decide substantive readings, so both are held back to tier (3). Their v1.0 values
stand as the honest measure of how much the four screeners really disagreed about eligibility.

### 4.3 Pairwise Cohen's matrix, P1 flag

| pair | raw v1.0 | κ v1.0 | raw v1.2 | κ v1.2 |
|---|---|---|---|---|
| S1–S2 | 73.3% | 0.000 | 93.3% | 0.898 |
| S1–S3 | 40.0% | 0.000 | 93.3% | 0.898 |
| S1–S4 | 100% | undefined (one category) | 100% | 1.000 |
| S2–S3 | 66.7% | 0.390 | 100% | 1.000 |
| S2–S4 | 73.3% | 0.000 | 93.3% | 0.898 |
| S3–S4 | 40.0% | 0.000 | 93.3% | 0.898 |

### 4.4 Is the floor met? — honestly

**Yes, on the number the floor should be assessed against, and no, not as an independent
replication.**

| | value | floor | met |
|---|---|---|---|
| Fleiss' κ, P1 flag, counterfactual v1.2 | **0.932** [0.777, 1.000] | 0.60 | **yes** |
| raw agreement, P1 flag, counterfactual v1.2 | **95.6%** [86.7, 100] | 90% | **yes** |
| Gwet's AC1, P1 flag, counterfactual v1.2 | 0.934 [0.800, 1.000] | — | — |
| Fleiss' κ, pre-reconciliation | −0.015 | 0.60 | **no** — reported unchanged |

The bootstrap lower bound on κ is 0.777, comfortably clear of 0.60, so the conclusion does not
rest on a point estimate.

**Three caveats stated because they are true, not because a reviewer will ask:**

1. **This is not an independent re-rating.** It is the same four sealed files, mechanically
   re-expressed. It establishes that the v1.0 disagreement was a defect in the form, which is
   what the diagnostic question was; it does not establish that four screeners coding afresh
   under v1.2 would agree to κ = 0.93. **A fresh independent re-coding of the overlap set under
   v1.2 by four screeners remains outstanding**, and the paper must say so wherever this number
   appears.
2. **κ = 0.932 is computed over three categories** (`False`, `not_assessable`,
   `not_applicable`), which is a less skewed distribution than the two-category v1.0 read and is
   therefore less exposed to the kappa paradox. That is a real property of the amended coding,
   not a trick, but it is why raw agreement and AC1 are reported beside it, as pre-specified.
3. **The residual disagreement is real and is not swept up.** PMID 42489954 remains 2–2 on
   whether it is excluded or unresolved. The transform preserves it deliberately.

---

## 5. Effect on the endpoints

Overlap set re-adjudicated; the 85 unique records are re-coded under the mechanical rules
D2/D3/D4 alone, which touch no numerator and no denominator.

| endpoint | before | after | |
|---|---|---|---|
| flow: included & reachable / unreachable / excluded | 35 / 20 / 45 | 35 / 20 / 45 | unchanged |
| **P1, complete case** | **0/35 = 0.0%** [0.0, 9.9] | **0/35 = 0.0%** [0.0, 9.9] | **unchanged** |
| **P1, headline bounding interval** | **[0.0%, 36.4%]** | **[0.0%, 36.4%]** | **unchanged** |
| S2, headline unit = slice | 8/35 = 22.9% [12.1, 39.0] | 8/35 = 22.9% | unchanged |
| S4, explicit subject-level split | 13/35 = 37.1% [23.2, 53.7] | **12/35 = 34.3%** [20.8, 50.9] | **↓ D6** |
| S5, positional distribution reported | 0/35 = 0.0% [0.0, 9.9] | **1/35 = 2.9%** [0.5, 14.5] | **↑ D9** |
| S6, unreachable | 20/55 = 36.4% [24.9, 49.6] | 20/55 = 36.4% | unchanged |

**The primary endpoint did not move.** The censoring-free statement survives the adjudication
intact: of the 145 coded records, not one carries a positive code on any of the four zero-image
sub-flags, and the reportable headline remains the bounding interval **[0.0%, 36.4%]**, because
unreachability at 36.4% is still far above the §7 15% threshold. **The binding constraint is
still access, and this remedy did not touch it.**

The two secondaries that moved, moved by one record each and in opposite directions. S5 leaving
zero is the more consequential of the two for how the paper reads, and it is the one that goes
against us.

---

## 6. Did the amendment find a paper that reports a zero-image baseline?

The honesty rules require looking for this as hard as for the absence, because one such paper
would materially change the result. The search was run over all 145 coded records, not only the
overlap set: every `trivial_baseline_quote`, `trivial_baseline_other_description`, `notes` and
`positional_distribution_quote` field, matched against majority-class, no-information-rate,
chance-level, random-guess, permutation-test, shuffled/permuted-label, prevalence-baseline,
constant-predictor, always-positive, slice-index/slice-position/relative-position/z-coordinate/
ImagePositionPatient, and scanner/manufacturer/DICOM-header-as-input.

**No. Not one.** The four closest calls, and why each is correctly negative:

| PMID | what it has | why it is not a P1 baseline |
|---|---|---|
| **41568076** | a **binary lung-mask** arm at balanced accuracy 0.56–0.63, which the authors report as "only 10% (or less) below the performance achieved with the CT contrast", concluding CNNs over-rely on shape | a mask is still pixels — a shape-only control, not a pixel-free one. The right instinct; it does not reach P1. Worth quoting in our Discussion. |
| **41568076**, **40093990** | an asserted chance level (0.33 for 3 classes; MCC 0 = random) | asserted, not measured; the codebook already routes this to `chance_asserted_without_measurement`, and both screeners set that field true |
| **34003056** | the **PanCan 2b** comparator | its predictors — nodule size, type, spiculation, emphysema, upper-lobe location — are all read off the image |
| **42100397** | a "Clinic" arm, AUC 0.720/0.692 | the retained features are radiologist-read CT semantic features, i.e. pixels; correctly coded false rather than as a clinical-only arm |
| **35626379** | slice-level headline AUC on **RSNA-STR PE**, the very dataset whose public "Baseline with no image" notebook bins P(PE \| relative slice location) | the paper itself reports no such arm — which is the finding |

`clinical_or_demographic_only` (secondary S1, never primary) is true on seven records: 37222638,
38337016, 38784688, 39061744, 39200968, 39513126, 41617832. D2 removes 41617832 (named without
a value) and confirms 37222638 (three measured AUCs in the abstract).

---

## 7. What this round did not do

- It did **not** touch the frame, the permutation, the sample, the seed, or the batch allocation.
- It did **not** edit any of the four sealed screener files.
- It did **not** change any endpoint definition, the Wilson interval method, the bootstrap
  specification, the bounding-analysis rules, or the 15% unreachability threshold.
- It did **not** fix the access failure. Full text is still unreachable for 36.4% of the
  eligible set, the bounding interval [0.0%, 36.4%] is still the headline, and **enlarging the
  sample cannot narrow it — only recovering full texts can.** Rung 5 of the §7 access ladder
  (interlibrary loan / author request, 21-day clock) has not been initiated by any screener.
  That is the next remedy and it is a separate piece of work.
- It did **not** produce an independent post-amendment reliability estimate. A fresh blind
  re-coding of the overlap set by four screeners under v1.2 is outstanding.

### Priority list for the access remedy, from this round

PMID **42489954** (the only surviving eligibility disagreement), then **37222638** and
**41617832** (both have a clinical-only arm visible in the abstract and both are S1 candidates
whose eligibility is unresolved), then **39423605** (E-DERIV vs included cannot be settled from
the abstract). Screeners S2, S3 and S4 all recorded that Wiley, Springer and Elsevier returned
HTTP 403 to automated requests **including for articles Unpaywall and OpenAlex report as open
access** — so several of these are environment failures rather than true paywalls and should be
retried by a screener with browser or institutional access before rung 5 is invoked.
