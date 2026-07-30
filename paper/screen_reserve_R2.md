# Reserve block R2 -- prevalence screen, permutation positions 161-210

**Protocol** `paper/screen_protocol.md` v1.3 | **Codebook** `paper/screen_frame.json` v1.2 (D1-D14) | **Sample** `paper/screen_sample.json` v1.1, reserve records 161-210 | **Screener** S1 | **Submitted** 2026-07-29T16:45:00Z

This block was screened under the extension rule pre-specified in `screen_protocol.md` section 3.1, before any outcome was seen: *"If the number of included papers is below 75, continue into the reserve in permutation order, in blocks of 50 records, until either 75 included papers are reached or position 400 is exhausted. Every record in a started block is screened and reported; blocks are never truncated part-way once begun."* All 50 records were screened and all 50 are reported. Machine-readable codes, verbatim quotes and the term-by-term search record for every field are in `paper/screen_reserve_R2.json`; every number below is computed from that file.

> **A second, independent coding of this same block already existed in the working tree when this pass began, and it is preserved.** It is now `paper/screen_reserve_R2_pass1.md`. Section 8 reports the agreement between the two passes field by field. This is the first genuine independent duplicate coding under v1.2 anywhere in the workflow, which `screen_protocol.md` section 6.1 lists as an outstanding action, and it is reported as such rather than absorbed.

---

## 1. Headline tallies

| | n | of |
|---|---|---|
| Records screened | **50** | 50 assigned |
| Included | **17** | 50 |
| Excluded | **28** | 50 |
| Unreachable (`unreachable_eligibility_unresolved`) | **5** | 50 |
| Eligible-looking set (included + unreachable) | 22 | 50 |
| **Reporting ANY zero-image baseline (P1 family)** | **0** | 17 included |
| Reporting any non-imaging baseline at all (S1 family) | **1** | 17 included |

**The primary answer for this block is zero.** Not one of the 17 included papers reports a measured value for a constant/prevalence predictor, a positional (slice-index) predictor, an acquisition-metadata predictor, or a permuted-label null. Across all 50 records and all 300 sub-flag cells, `constant_or_prevalence`, `positional`, `acquisition_metadata` and `permuted_or_shuffled_label` are true **nowhere** — 17 cells of each are evidenced `false`, 33 are `not_applicable` under D3, and none is `true`.

Block P1 = 0/17, Wilson 95% **0.0% [0.0%, 18.4%]**. Block S6 (unreachable) = 5/22, Wilson 95% **22.7% [10.1%, 43.4%]** — above the 15% threshold at which section 7 makes the bounding interval the headline.

### Cumulative position after this block

Combining with the cumulative figures recorded at the end of `paper/screen_reserve_R1.md` (55 included, 25 unreachable, 80 eligible-looking):

| | before R2 | R2 | cumulative |
|---|---|---|---|
| Included | 55 | +17 | **72** |
| Unreachable | 25 | +5 | **30** |
| Eligible-looking | 80 | +22 | **102** |
| P1 numerator | 0 | +0 | **0** |

- **P1 complete-case 0/72, Wilson 95% 0.0% [0.0%, 5.1%].** Still exactly zero.
- **S6 unreachable 30/102 = 29.4% [21.4%, 38.9%].** Above 15%, so section 7 still binds and the bounding interval remains the headline.
- **Headline bounding interval [0.0%, 29.4%]** (lower arm: every unreachable paper coded as not reporting a zero-image baseline; upper arm: every one coded as reporting one).

**The extension rule has not stopped.** 72 included is below the 75 target and position 210 is well short of 400, so the rule requires block R3 (positions 211-260) next. On this block's yield of 17 per 50, three more included papers needs roughly nine more records; one more block will pass the target comfortably.

**And the binding constraint moved the right way, slightly, for the first time.** The complete-case interval narrowed from 0/55 → [0.0%, 6.5%] to 0/72 → [0.0%, 5.1%], but the complete-case estimate is not the reportable figure. The reportable figure is the bounding interval, and it improved from **[0.0%, 31.2%] to [0.0%, 29.4%]** — because this block's unreachable fraction (22.7%) is *lower* than the running rate, not because the sample got bigger. That is the whole point of section 3.1's warning: sample size does not move this bound. Access does. A block with a better access rate moves it; a block with a worse one moves it back.

---

## 2. The one positive, in full

**PMID 40121941** (position 190) — Song B, et al., *EBioMedicine* 2025, "Deep learning informed multimodal fusion of radiology and pathology to predict outcomes in HPV-associated oropharyngeal squamous cell carcinoma."

> "SMuRF outperformed the clinical-based models for both endpoints on the test set (Supplementary Fig. S3, C-index for DFS: 0.79 vs. 0.57; **AUC for grade: 0.74 vs. 0.60**)." — Results, *Interpretabilities of SMuRF*

The supplement names what is in the pixel-free arm:

> "Fig S3. The multivariable Cox regression model including **clinical factors only (age, smoking PY, T-stage and AJCC 8th stage)** on the test set (A). ... ROC curves for grade classification for the clinical factors only and clinical factor + SMuRF on the test set (C)." — supplementary file `mmc1.docx`, figure legends

A model with no pixels, a measured value, the same metric, the same 105-patient test set. `clinical_or_demographic_only` is coded **true**.

**This is an S1 positive, not a P1 positive.** `screen_protocol.md` section 1 restricts the primary to constant/prevalence, positional, acquisition-metadata and permuted-label, and says why: *"A clinical-variables-only nomogram is deliberately **not** in the primary: it is a different comparison and it does not test whether the benchmark is solvable without pixels."* This record therefore does **not** move P1 off zero.

It is flagged for adjudication on one point: T-stage and AJCC stage are assigned by clinicians partly from imaging, so a screener could argue the arm is not truly pixel-free and code the flag false. That argument changes S1 and nothing else; P1 is zero on either reading.

**Note how it was found.** Not by the term list. `clinical-only` and `clinical model` both return **zero** hits in this article — the authors write "clinical-based models" — and the arm's composition appears only in a supplementary figure legend. It was found by reading the Results and then downloading and reading the supplement, which is what `screen_frame.json` `reading_effort` actually requires. This is the second block in a row where the fourteen mandated terms did not surface the block's only positive (R1's came out of a results table). The term list is a floor, not a substitute for reading.

---

## 3. Three near-misses and a false alarm

Recorded because staying silent about them would be dishonest.

**PMID 42225843** (position 165, *Sci Rep*, AIS on non-contrast CT) is the closest any paper in this block comes to the concept — and it comes at it from the wrong side. The word `metadata` hits three times, every time as a statement that the model **does not** use it: *"the diagnostic performance of NCCT-based AI without integration of clinical metadata"*, *"All nine readers were strictly blinded to clinical metadata, including patient age, sex, symptoms, and National Institutes of Health Stroke Scale (NIHSS) scores."* An acquisition-metadata baseline with a measured value would be `acquisition_metadata` true; a statement that metadata was withheld is not one. Coded **false**.

**PMID 35571081** (position 183, *Front Pharmacol*, PET/CT EGFR status) builds exactly the structure that usually carries a clinical-only arm — *"smoking was combined with DS for multivariable logistic regression to establish a hybrid model"* — and never reports smoking's own AUC. Under the rule that an assertion with no number is false for every flag, `clinical_or_demographic_only` is **false**. This paper returns **zero hits on all fourteen terms** in its main text, the only record in the block to do so.

**PMID 41087406** (position 182) is the one paper in all fifty that actually models an ordered index and quantifies label frequency against it — it classifies tibial slices into epiphysis, growth plate, primary and secondary spongiosa and fits "a regional probability distribution method to detect the transitional landmarks between these compartments". It is a **mouse** study, excluded `E-NONMED`, so under D3 all six sub-flags are `not_applicable` and it enters no denominator. Recorded in full in the JSON so the reason it is absent is visible rather than silent.

**The false alarm: PMID 36010183** (position 181) returns four hits on `permut` and looks, on a term count alone, like a permutation null. Every hit is the paper's permutation-based **voting** mechanism over SVM, Gaussian naive Bayes and XGBoost — combinations of which classifiers vote, not shuffles of the labels. `permuted_or_shuffled_label` is **false**. Together with the R1 case that went the other way, this is the argument for reading hits rather than counting them.

---

## 4. Secondary endpoints in this block

| id | statement | this block | Wilson 95% |
|---|---|---|---|
| S1 | any non-imaging baseline (P1 family + clinical-only + other) | 1/17 | 5.9% [1.0%, 27.0%] |
| S2 | headline evaluation unit is the slice | 2/17 | 11.8% [3.3%, 34.3%] |
| S3 | among papers reporting any slice-level metric, also report patient-level | 1/3 | 33.3% [6.1%, 79.2%] |
| S4 | explicitly states a subject-level split | 8/17 | 47.1% [26.2%, 69.0%] |
| S5 | reports or discusses the positional distribution of labels | **0/17** | 0.0% [0.0%, 18.4%] |
| S6 | unreachable, over included + unreachable | 5/22 | 22.7% [10.1%, 43.4%] |
| S8 | subject-clustered uncertainty interval | 1/17 | 5.9% [1.0%, 27.0%] |
| S9 | reports n positive patients as well as n positive slices | 2/17 | 11.8% [3.3%, 34.3%] |

**S5 is zero.** Not one of the seventeen relates its label to position along the acquisition axis — no histogram of positive-slice position, no mean relative position, no position-stratified metric. Two of the seventeen score the slice itself as the classification unit and neither of those does it either.

**S7, headline unit crossed with the P1 flag, exact counts, all cells:**

| effective headline unit | n | P1 true | P1 false |
|---|---|---|---|
| patient (`patient`, or the only unit reported is patient) | 7 | 0 | 7 |
| slice (only unit reported is slice) | 2 | 0 | 2 |
| unclear (the only unit reported cannot be identified) | 8 | 0 | 8 |
| **total** | **17** | **0** | **17** |

Distributions on the fields the endpoints rest on, `unclear` reported as its own category and never merged or imputed:

- **`evaluation_unit_reported`**: unclear 8, patient 6, slice 2, both 1
- **`split_unit`**: patient_subject 8, random_unit_not_stated 6, slice_or_image 2, external_cohort_only 1
- **`input_representation`**: 2D_slice 7, unclear 6, patch_3D 2, 3D_volume 1, 2.5D_stack 1
- **`uncertainty_interval_reported`**: none 8, ci_unspecified_method 7, ci_clustered_by_subject 1, sd_across_folds 1
- **`code_availability`**: not_stated 16, public_link_works 1
- **`modality`**: CT 9, MRI 6, PET_CT 1, multiple 1
- **`dataset_public`**: public 9, private 7, mixed 1

**Eight of seventeen included papers never say what one scored unit is,** and six of seventeen split with no unit named anywhere. Those are not coding failures — the codebook says so explicitly and requires them to be reported as their own category — and they are the finding.

### The record that shows why the endpoints are constructed this way

**PMID 40002836** (position 200, *Biomedicines*) analyses **3,194 image units drawn from 60 patients**, splits them 80-10-10 with the unit unnamed, and then asserts leakage-freedom on the strength of that split alone:

> "A stratified 80-10-10 split strategy was implemented, ensuring that training, validation and test subsets were completely separate." ... "**No data leakage occurred**, as the train-validation-test split (80-10-10) ensured that test data remained completely unseen during training." ... "the observed 100% classification accuracy is likely due to optimal feature separability rather than data leakage."

Roughly 53 units per patient, no patient-level check offered, and the reported result is **100% accuracy, AUC 1.0000, 95% CI [1.0000, 1.0000]** from 1,000 bootstrap resamples. No constant, prevalence, positional or permuted-label comparator is reported that would let a reader interpret that number. It is a textbook instance of the configuration the parent paper formalises, and its P1 flag is correctly zero — because the paper reports no such baseline at all.

### The record that shows the opposite

**PMID 42225843** (position 165, *Sci Rep*) is the counter-example the paper should quote as well. It reports **both** units and the gap between them — *"we observed a significant discrepancy between patient-wise (69.66%) and slice-wise (44.95%) sensitivities"* — trains on slice-level expert annotations, validates on four institutions distinct from the two development sites, and is the only record in the block to cluster its uncertainty by subject:

> "Generalized linear mixed models (GLMM) with a logit link function were used for patient-wise analysis ... patient and reader were included as random intercepts to account for **patient-level clustering** across paired reads of the same case ... For slice-wise analysis, **generalized estimating equations (GEE) with an exchangeable working correlation structure** were applied to account for the clustered nature of multi-slice data and repeated measures across readers."

It still reports no zero-image baseline. Doing everything else right does not imply doing this.

---

## 5. Exclusions and access

Exclusion codes individually, as protocol section 9 requires:

| code | n | what they were |
|---|---|---|
| `E-SEG` | 9 | segmentation, bounding-box detection, landmark localisation, catheter/object localisation, super-resolution — no categorical class decision evaluated |
| `E-NONMED` | 6 | mouse tibia micro-CT, mouse skin FF-OCT, canine middle-ear CT, ovine tissue spectroscopy, microarray gene expression, Thai clinical-note NER |
| `E-DERIV` | 6 | the fitted classifier eats a radiomics vector, a connectivity graph, ICA loadings or a thickness table — the image is discarded |
| `E-2D` | 5 | fundus photographs, intraoral periapical radiographs, knee radiographs, H&E whole slides, single horizontal OCT B-scans |
| `E-NOCLF` | 2 | no classifier fitted — the class decision belongs to human readers (D10, ×2) |

Six `E-DERIV` records in fifty is the frame's deliberate breadth doing what section 2.1 said it would; these papers are inside the query and outside the failure mode, and are reported separately in the flow diagram.

**The access ladder** (section 7) was worked in order for all 27 records that reached stage 2. Per **D4** it was not climbed for eligibility purposes on the 23 records excluded at stage 1, so S6 measures the reachability of the eligible-looking literature.

| rung that worked | n |
|---|---|
| `oa_pmc_or_publisher` (rung 1) | 36 |
| `not_attempted_excluded_at_stage1` (D4) | 8 |
| `unreachable_paywalled` (rungs 1-5 exhausted) | 5 |
| `repository_or_accepted_manuscript` (rung 4) | 1 |

**Two rung-1 recoveries by indirect routes**, both worth recording for reproducibility. PMID **35562596**'s PMC deposit carries front matter and abstract only (`pmc-prop-open-access=no`, no `<body>`) and Springer shows *"This is a preview of subscription content"* — the Europe PMC rendered PDF served the complete typeset version of record, pages 2773-2780. PMID **42225843**, a 2026 *Sci Rep* article not yet in PMC, is gold OA at nature.com but the PDF is labelled *"We are providing an unedited version of this manuscript"*, so it is coded `accepted_manuscript` and reserved for the version-of-record sensitivity analysis.

**One rung-4 recovery.** PMID **33713959** (Elsevier, `is_oa=false` at both Unpaywall and OpenAlex) was recovered as an accepted manuscript from the University of West London repository (`repository.uwl.ac.uk/id/eprint/7765`) and coded in full including the fourteen-term search. Also flagged for the version-of-record sensitivity analysis.

**Five unreachable, every failed rung named in the JSON:** 32232524 (Springer, closed at every index); 38721876 (SAGE 403; the HAL record OpenAlex points to reports `submitType_s="notice"` and holds no file); 41655629 (ScienceDirect 403, no repository copy); 42184237 (JoVE returns HTTP 202 with an empty body to every client here); 42159478 (RSNA 403 — the same publisher-side refusal recorded in changelog v1.3 — and the Radboud repository record's only action is "Upload full text").

Two of the five would very probably have been *included* on full text: 41655629 (CBCT, diagnostic AUC 94.4% internal, 85.2-90.0% external) and 42184237, whose abstract states a patient-level split in so many words. **D1** forbids coding them included anyway, for the reason the codebook gives — an included record must carry an *evidenced* `trivial_baseline`, and the mandatory search cannot be run on a text nobody has. That is precisely the cost S6 exists to measure.

**No infringing source was used at any point.** Sci-Hub and equivalents were not accessed. No CAPTCHA, bot-detection or proof-of-work challenge was bypassed — including the PMC proof-of-work gate that blocked the one supplementary `.xlsx` for PMID 41633187, which is disclosed in the JSON even though that record is excluded and the missing file affects no endpoint.

---

## 6. How the negative was evidenced

An unevidenced negative is not accepted. For all 17 included records, every one of the fourteen terms — *baseline, chance, random, majority, prevalence, constant, trivial, metadata, clinical-only, clinical model, position, location, slice index, permut* — was run over the complete full text, and `searches_run` records the hit counts term by term together with what each hit actually was.

Supplements, per record: **five obtained and searched in full** (42225843, 40676122, 35571081, 40121941, 35562596 — two of them required going to the publisher's static-content host after PMC served a download shell); **eleven declare no supplementary material, so none exists**; **one** (34109325) declares a supplement that is a *video* of a reconstructed cardiac cycle, containing no searchable text. **No all-false code in this file rests on an unsearched supplement that its article declares.**

`chance_asserted_without_measurement` is true on two included records — 41657565, which draws and names the ROC diagonal three different ways (*"the diagonal line representing random chance"*, *"The curve is plotted against a random guess line"*, *"A dashed diagonal line indicates random performance"*) without measuring one, and 35562596, whose ROC figure legend labels the diagonal *"Random curve"*. Both are false for every sub-flag, per the codebook, and routed to this field instead.

---

## 7. Amended-codebook rules that fired

| rule | where it fired | effect |
|---|---|---|
| **D1** unreachable dominates included | 32232524, 38721876, 41655629, 42184237, 42159478 | five abstracts consistent with inclusion (two strongly so) are `unreachable_eligibility_unresolved`, not `included` |
| **D2** positive/negative asymmetry | all 17 included (false requires the search) and the one true (40121941, quote carries AUC 0.60) | no unevidenced negative and no unevidenced positive anywhere |
| **D3** `not_applicable` on non-included | all 33 excluded and unreachable records | machine-checked: `not_applicable` appears on **no** included record |
| **D4** ladder not climbed for stage-1 exclusions | 23 records, 8 of which never had a full text fetched | keeps S6 measuring the eligible-looking literature; where PMC had the text anyway, the rung that worked is recorded instead |
| **D5** Methods govern over Abstract | 40676122 (images vs patients), 40121941 (grade AUC 0.75 vs 0.74) | the Results value is coded and the contradiction recorded verbatim |
| **D6** patient-naming-noun word list | *in*: 40134559, 34109325, 33713959, 35571081, 35562596, 40121941, 41444372, 40342492 — *out*: 40783612, 34828081, 36010183, 36672641, 40601647, 40002836 | S4 = 8/17; "data", "dataset" and "sections" never upgrade a split |
| **D7** a unit named in the split description is a named unit | 41657565 ("The images were organized into a training and a testing set") | one record leaves `random_unit_not_stated` |
| **D9** ordering of the classification unit, not vocabulary | 42225843 (slice thickness), 40134559 (ROI placement), 40121941 (sub-volume placement), 41657565 and 40601647 (positional encoding) | S5 stays 0/17 on the correct test |
| **D10** E-SEG's qualifier binds | *to* E-NOCLF: 40544715, 41899885 — *away from* E-SEG: 32691326, 35748898, 40365495 | reader-made class decisions give E-NOCLF; evaluated class decisions on feature vectors give E-DERIV |
| **D11** trigger for `include_provisional` | whole block | used **zero** times; every record is `exclude` or `go_to_fulltext` |
| **D12** modality is the modality of the input | 41444372 (`multiple`), 40121941 and 40342492 (`CT` + `mixed_modality`) | |
| **D13** `not_stated` for silence on code | 16 of 17 included | without D13 all 16 would have been forced to the inference `none` |
| **D14** `na_only_one_unit_reported` vs `unclear` | 16 of 17 included | applies even where the single unit is itself `unclear` |

D8 did not fire: no record splits over lesions or ROIs and none defers its Methods to an unsampled companion paper.

**One codebook gap is logged rather than resolved unilaterally,** in addition to the reserve-block `batch` enum gap R1 already logged: D2(b) makes a `false` sub-flag conditional on searching the supplement, but the form has no field recording whether a supplement was *reachable*, and no rule for an included record whose supplement exists and cannot be obtained. One record here came within a whisker of that state (41633187, one `.xlsx` behind a proof-of-work gate) and happens to be excluded, so D3 disposed of it. Read literally, D2(b) would have forced `not_assessable` on all six sub-flags and removed the record from both the P1 numerator *and* its denominator. That case needs a rule before it decides an endpoint.

---

## 8. Two independent passes over the same 50 records

A prior independent coding of block R2 was already present in the working tree when this pass began, and this pass was carried out without reading it. It is preserved unchanged as **`paper/screen_reserve_R2_pass1.md`**; its JSON did not survive, and that is recorded as a loss rather than glossed over.

**Where the two passes agree — including every number the paper reports:**

| | pass 1 | pass 2 (this file) |
|---|---|---|
| records screened | 50 | 50 |
| included / excluded / unreachable | 17 / 28 / 5 | 17 / 28 / 5 |
| **P1 numerator** | **0/17** | **0/17** |
| S1 numerator | 1/17 | 1/17 |
| the S1 positive | 40121941, `clinical_or_demographic_only` | 40121941, `clinical_or_demographic_only` |
| the quote it rests on | "AUC for grade: 0.74 vs. 0.60" | "AUC for grade: 0.74 vs. 0.60" |
| S5 | 0/17 | 0/17 |
| S6 | 5/22 | 5/22 |
| the five unreachable PMIDs | 32232524, 38721876, 41655629, 42184237, 42159478 | identical |
| S8 | 1/17 (42225843) | 1/17 (42225843) |

**Where they differ:**

| record | pass 1 | pass 2 | consequence |
|---|---|---|---|
| 41331277 (188) | **included**, flagged on I3 | **excluded** `E-2D`, flagged | genuine eligibility disagreement; single horizontal OCT cross-hair B-scan per visit — is that a "volumetric OCT B-scan stack"? Both passes ran the full-text search and both found no baseline, so P1 is 0 either way |
| 40601647 (199) | **excluded** `E-DERIV` | **included**, `screener_confidence=low` | genuine eligibility disagreement on an internally contradictory paper (hand-crafted features per section 3, but a 3D-conv first layer in Table 5 and Grad-CAM on the input MRI). Section 4.1 of the protocol forbids excluding for being under-described, which drove pass 2. P1 false either way |
| 39956834 (193) | `E-NOCLF` | `E-DERIV` | exclusion-code choice only; both excluded, no endpoint effect |
| 33713959 (179) | `patient_subject` | `patient_subject` **after correction** | **pass 1 was right and this pass was initially wrong** — see below |
| 42225843 (165) | `patient_subject` | `external_cohort_only` | level choice on a paper whose sets are whole separate hospitals; moves S4 by one and is flagged |
| S4 | 10/17 | 8/17 | the split-unit codes above, plus the different included sets |

**The correction, logged rather than quietly fixed.** On PMID 33713959 this pass first coded `split_unit=random_unit_not_stated` from the splitting sentence alone (*"The OASIS dataset is randomly broken down into five sections"*). The **next** sentence reads *"The experiments in this paper secured that the patient-wise division is taken into account"* — and "patient-wise" is one of the exact phrases the codebook enumerates as satisfying `patient_subject`. Checking pass 1's disagreement surfaced it; the code was changed to `patient_subject` before submission and the correction is recorded in the record's own `notes`. Two other pass-1 claims were checked the same way against the full texts and did **not** hold (no patient-naming noun exists in the split descriptions of 34828081 or 36010183), so those codes stand.

**What this comparison is and is not.** It is not a pre-registered agreement statistic: it is two passes by the same screener identity over one non-overlap block, not the fresh four-screener re-coding under v1.2 that section 6.1 says is still outstanding. What it is worth saying is narrower and still useful: **two independent passes over these 50 records, differing on two eligibility calls, one exclusion code and two split-unit levels, returned the identical primary result, the identical single S1 positive and the identical quote underneath it.** The disagreements are where the earlier rounds' disagreements always were — reachability and eligibility — and not in the primary flag. That is the same diagnosis the v1.2 adjudication reached, arrived at independently.

---

## 9. Record-by-record

| pos | PMID | year / venue | decision | code / P1 | one line |
|---|---|---|---|---|---|
| 161 | 41861368 | 2026 JMIR Form Res | excluded | `E-NONMED` | Thai discharge-summary NER; no image anywhere. Matched the frame on "CT" and "F1". |
| 162 | 40544715 | 2025 Eur J Radiol | excluded | `E-NOCLF` | DLIR reconstruction; four radiologists make every diagnosis. D10 → E-NOCLF, not E-SEG. |
| 163 | 40783612 | 2025 Sci Rep | **included** | P1 false | Kaggle brain-stroke CT, 2,501 "CT Images", accuracy 99.09%. No patient count, no split ratio, no split unit, no baseline. |
| 164 | 32691326 | 2020 Phys Eng Sci Med | excluded | `E-DERIV` | SVM/kNN/ANN on histogram+GLCM+NGTDM features; the image is discarded before the class decision. |
| 165 | 42225843 | 2026 Sci Rep | **included** | P1 false | AIS on NCCT. Patient-wise **and** slice-wise sensitivity (69.66% vs 44.95%), GLMM/GEE subject-clustered intervals, four external institutions — and still no zero-image baseline. Accepted manuscript, flagged. |
| 166 | 41899885 | 2026 Bioengineering | excluded | `E-NOCLF` | Pre-trained nnU-Net segments; six readers decide. D10. Full text read before excluding. |
| 167 | 34729970 | 2021 J Biomed Opt | excluded | `E-NONMED` | Ovine tissue, light-scattering spectra. Not human, not imaging. |
| 168 | 40676122 | 2025 Sci Rep | **included** | P1 false | AI observer over >750,000 RSNA-style 2D axial slices. **Calls the same 1,000 objects "images" and "patients" in adjacent sentences** (D5). The only "baseline" is the full-view image reconstruction. |
| 169 | 32232524 | 2020 Abdom Radiol | unreachable | not_applicable | Springer, closed at every index. Rungs 1-5 exhausted. D1 → unresolved. |
| 170 | 34828081 | 2021 Entropy | **included** | P1 false | COVID-CT (volumetric arm) + two X-ray sets; A2 applies. Accuracy 0.783 on the CT arm. Split unit never named. |
| 171 | 40134559 | 2024 J Bone Oncol | **included** | P1 false | 3DResUNet on QCT vertebrae, 749 patients split patient-wise, AUC 0.966 from thresholded vBMD. Flagged on I2. |
| 172 | 40745009 | 2025 Sci Rep | excluded | `E-2D` | Fundus photographs are the model input; the CT and CMR are outcomes. |
| 173 | 31001458 | 2019 Cureus | excluded | `E-NONMED` | ANN on a 12-gene microarray vector. No imaging. |
| 174 | 39656660 | 2025 Dentomaxillofac Radiol | excluded | `E-2D` | Intraoral periapical radiographs; CBCT is only the ground-truth arbiter. |
| 175 | 34109325 | 2020 MICCAI | **included** | P1 false | Cine CMR short-axis stack → U-Net masks → VAE + subject-level classifier, sensitivity 88.43%. Flagged on I3 (the classifier sees masks, not intensities). |
| 176 | 38893629 | 2024 Diagnostics | excluded | `E-SEG` | Mask R-CNN on PE-positive patients only — no negative class, so the sen/spec are pixel-wise. The codebook's warning case exactly. |
| 177 | 38905892 | 2024 Comput Biol Med | excluded | `E-SEG` | 3D U-Net; Dice and voxel precision only. Title says detection, evaluation is segmentation (A4). |
| 178 | 32484573 | 2020 Med Phys | excluded | `E-SEG` | Catheter localisation in MRI, evaluated in millimetres. |
| 179 | 33713959 | 2021 Comput Methods Programs Biomed | **included** | P1 false | OASIS AD, GoogLeNet transfer 93.02%, "patient-wise division" claimed — while over-sampling and augmenting *before* the CV split. Three mutually inconsistent dataset counts. Rung-4 recovery, flagged. |
| 180 | 39562310 | 2024 Hum Brain Mapp | excluded | `E-DERIV` | ICA loadings + sFNC connectivity + SNPs. The pilot's E-DERIV construction with a genomics arm. |
| 181 | 36010183 | 2022 Diagnostics | **included** | P1 false | Kaggle AD 4-class, 6,200 images, accuracy 91.75%. Four `permut` hits are a **voting ensemble**, not a permutation null. |
| 182 | 41087406 | 2025 Sci Rep | excluded | `E-NONMED` | Mouse tibia micro-CT — and the only paper in the block that models an ordered index and its label distribution. Excluded on species; recorded in full. |
| 183 | 35571081 | 2022 Front Pharmacol | **included** | P1 false | PET/CT EGFR status, 3D CNN on 64³ cuboids, validation AUC 0.85. **Zero hits on all fourteen terms** in the main text; supplement searched. 194 vs 138+57=195 recorded under D5. |
| 184 | 39768288 | 2024 Life | excluded | `E-SEG` | Latent-diffusion PET super-resolution; no categorical decision of any kind. |
| 185 | 41353071 | 2026 Acad Radiol | excluded | `E-SEG` | Cardiac MRI planning landmarks, millimetres and degrees. |
| 186 | 35562596 | 2022 J Cancer Res Clin Oncol | **included** | P1 false | CT cervical nodes, 276 patients split patient-wise, accuracy 87.50%. Draws a "Random curve" and never measures one. Recovered via the Europe PMC rendered PDF. |
| 187 | 41633187 | 2026 Comput Med Imaging Graph | excluded | `E-SEG` | YOLOv11 CRC detection, **no negative class in the data**. Reports slice-level vs patient-level splitting side by side (recall 0.9949 vs 0.8092) — worth citing even though excluded. |
| 188 | 41331277 | 2025 Sci Rep | excluded | `E-2D` | Spectralis cross-hair scan: one horizontal B-scan per visit, no slice axis. **Disagrees with pass 1**; flagged. Full-text search run anyway — no baseline either way. |
| 189 | 40714864 | 2025 Vet Radiol Ultrasound | excluded | `E-NONMED` | Canine middle-ear CT. Every other criterion met; not human. |
| 190 | 40121941 | 2025 EBioMedicine | **included** | P1 false, **S1 true** | CT + H&E fusion for OPSCC. **The block's only positive**: clinical-factors-only AUC 0.60 against 0.74, same metric, same test set. Found in the supplement, not by the term list. |
| 191 | 33662804 | 2021 Comput Methods Programs Biomed | excluded | `E-SEG` | COVID-19 CT segmentation; accuracy 0.994 beside DSC 0.704 gives the voxel denominator away. |
| 192 | 36672641 | 2023 Biomedicines | **included** | P1 false | 17,194 slices from the "CT Scan Slice Dataset", 70:20:10 with no unit named, no patient count, accuracy 97.12%. |
| 193 | 39956834 | 2025 Sci Rep | excluded | `E-DERIV` | OCT thickness tables → five regressors. Pass 1 coded E-NOCLF; both exclude. |
| 194 | 38721876 | 2024 J Endovasc Ther | unreachable | not_applicable | SAGE 403; the HAL record holds no file. Also raises an unresolved I2 question (commercial device, fitted elsewhere). |
| 195 | 34626908 | 2021 Comput Med Imaging Graph | excluded | `E-NONMED` | Mouse skin FF-OCT. |
| 196 | 35748898 | 2022 Eur Radiol | excluded | `E-DERIV` | nnU-Net segments; SVM over seven radiomics features decides. |
| 197 | 41657565 | 2025 Front Med | **included** | P1 false | CerevianNet over five Kaggle sets; 98.30% on one, 75.63% on another. Names the chance line three ways and measures none of them. |
| 198 | 41655629 | 2026 J Endod | unreachable | not_applicable | ScienceDirect 403, no repository copy. Would very likely have been included. |
| 199 | 40601647 | 2025 PLoS One | **included** | P1 false | HAETN for TLE. Internally contradictory about whether the network eats features or voxels; included per section 4.1, `screener_confidence=low`, mandatory adjudication. **Disagrees with pass 1.** No patient count, no image count, no class balance. |
| 200 | 40002836 | 2025 Biomedicines | **included** | P1 false | 3,194 images from 60 patients, 80-10-10 with the unit unnamed, 100.00% accuracy, AUC 1.0000 [1.0000, 1.0000], and *"No data leakage occurred"*. |
| 201 | 41037545 | 2026 IEEE Trans Biomed Eng | excluded | `E-DERIV` | rs-fMRI brain graphs + phenotypic data. Identical construction to pilot PMID 34924987. |
| 202 | 41444372 | 2025 Sci Rep | **included** | P1 false | Federated CT+MRI+histology+genomics, 221,347 "records", 99.2%. Claims division "by patient" while never stating a patient count. Flagged. |
| 203 | 42184237 | 2026 J Vis Exp | unreachable | not_applicable | JoVE returns an empty body. The abstract states a patient-level split in so many words — information now out of reach. |
| 204 | 39811011 | 2025 Eur Heart J Imaging Methods Pract | excluded | `E-2D` | H&E-stained myocardial whole slides. |
| 205 | 35830745 | 2022 Med Image Anal | excluded | `E-SEG` | Anatomy-guided object recognition; centroid, scale and wall-distance errors. |
| 206 | 42459762 | 2026 Front Artif Intell | excluded | `E-2D` | Knee radiographs; CT appears once, in the background sentence. |
| 207 | 40342492 | 2025 J Bone Oncol | **included** | P1 false | CT + pathology Swin fusion, 215 patients, AUC 0.966. The most explicit subject-level split in the block: both modalities from a patient stay on the same side of every fold. |
| 208 | 42159478 | 2026 Radiol Artif Intell | unreachable | not_applicable | RSNA 403; Radboud repository holds no bitstream. Eligibility genuinely unresolved in both directions (registration task, but a pCR classifier with AUC 0.81). |
| 209 | 40576676 | 2026 J Craniofac Surg | excluded | `E-SEG` | Sella turcica coordinate regression from 3D stereophotograph meshes; millimetres of deviation. |
| 210 | 40365495 | 2025 Front Med | excluded | `E-DERIV` | UPerNet segments; a radiomic signature decides (AUC 0.891/0.892). |

---

## 10. What this block adds to the argument, and what it does not

**Adds.** Seventeen more included papers, none of which reports a zero-image baseline, taking the complete-case primary to **0/72** and its Wilson upper bound below 5.1% for the first time. Seventeen more papers, none of which relates its labels to position along the acquisition axis. Two clean illustrative records at opposite ends: one that splits 3,194 slices from 60 patients at the file level, declares no leakage, and reports AUC 1.0000; and one that reports patient-wise and slice-wise sensitivity side by side with GEE-clustered intervals and still reports no pixel-free comparator.

**Does not add.** It does not narrow the reportable figure by sampling. S6 is still 29.4% cumulative, still far above the 15% threshold, so the bounding interval **[0.0%, 29.4%]** remains the headline and no further block will change that. Five more papers went unreachable here, four of them because a publisher refused this environment rather than because the article is closed to a human reader.

**And it does not settle the reliability question.** Section 6.1 still records a fresh four-screener independent re-coding under v1.2 as outstanding. Section 8 above is a two-pass comparison over one block, not that study — but it is evidence, and it points the same way the adjudication did: the disagreements are about reachability and eligibility, and the primary flag is not where they live.
