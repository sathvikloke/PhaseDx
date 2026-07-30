# Prevalence screen — reserve block R4 (permutation positions 261–310)

**Screener S1 · submitted 2026-07-30 · codebook `paper/screen_frame.json` v1.2 (amended, rules D1–D14) · protocol `paper/screen_protocol.md` v1.3**

Machine-readable records: `paper/screen_reserve_R4.json`. Fifty reserve records, positions 261–310, taken in
permutation order, none skipped and none substituted.

---

## 0. Read this first: R4 is outside the pre-registered stopping rule

THIS BLOCK IS OUTSIDE THE PRE-REGISTERED STOPPING RULE AND MUST BE REPORTED SEPARATELY FROM THE PRE-REGISTERED ANALYSIS SAMPLE. Protocol section 3.1 fixes the extension rule before any outcome was seen: "continue into the reserve in permutation order, in blocks of 50 records, until either 75 included papers are reached or position 400 is exhausted, whichever comes first." The running included total was re-derived independently for this report from the committed files rather than taken on trust: main analysis sample after the v1.3 access recovery = 38; screen_reserve_R1.json = 17 (positions 111-160); screen_reserve_R2.json = 17 (161-210); screen_reserve_R3.json = 19 (211-260); total 91. Ninety-one is at or above the target of 75, so the rule STOPPED at permutation position 260 and block R4 was not triggered by it. Block R4 was screened because the workflow explicitly instructed it. That instruction is honoured and the work is complete and coded to the same standard, but continuing past a stopping rule that has already fired is a data-dependent continuation, and pooling R4 into the pre-registered denominator without saying so would convert a pre-specified inverse-sampling design into an optional-stopping one. RECOMMENDATION: report R4 as a labelled POST-HOC EXTENSION, give the pre-registered figure at n=91 and the extended figure at n=114 side by side, and say which is which. Nothing about R4 changes the primary result in either direction: P1 is zero here as it is everywhere.

| block | positions | included | running total |
|---|---|---|---|
| main sample (post v1.3 recovery) | 1–100 | 38 | 38 |
| R1 | 111–160 | 17 | 55 |
| R2 | 161–210 | 17 | 72 |
| R3 | 211–260 | 19 | **91 — target of 75 met, rule stops here** |
| **R4 (this block)** | **261–310** | **23** | **114 — post-hoc** |

---

## 1. The primary result

> NOT ONE of the 50 records screened in block R4 reports a zero-image baseline. Across all 50 records and all six trivial_baseline sub-flags, the four P1-family sub-flags - constant/prevalence, positional, acquisition-metadata, permuted-label - are TRUE ZERO TIMES, on 23 fully evidenced complete-case searches with no included record left "not_assessable".

| endpoint | block R4 |
|---|---|
| **P1 — any zero-image baseline (complete case)** | **0/23 = 0.0% [0.0%, 14.3%]** |
| P1 lower bound (all unreachable negative) | 0/35 = 0.0% [0.0%, 9.9%] |
| P1 upper bound (all unreachable positive) | 12/35 = 34.3% |
| **P1 headline bounding interval (§7 binds)** | **[0.0%, 34.3%]** |
| S1 any non-imaging baseline | 2/23 = 8.7% [2.4%, 26.8%] |
| S2 headline unit is the slice | 6/23 = 26.1% [12.5%, 46.5%] |
| S3 slice papers also reporting patient | 1/7 = 14.3% [2.6%, 51.3%] |
| S4 explicit subject-level split | 6/23 = 26.1% [12.5%, 46.5%] |
| S5 positional distribution reported | 1/23 = 4.3% [0.8%, 21.0%] |
| S6 unreachable | 12/35 = 34.3% [20.8%, 50.8%] |
| S8 subject-clustered interval | 1/23 = 4.3% [0.8%, 21.0%] |
| S9 n positive patients *and* slices | 1/23 = 4.3% [0.8%, 21.0%] |

All intervals are Wilson score 95%, two-sided, as protocol §8 fixes for every proportion in the paper.

### Disposition

| | n |
|---|---|
| screened | 50 |
| included | 23 |
| excluded | 15 |
| unreachable, eligibility unresolved | 12 |
| eligible-looking denominator (included + unreachable) | 35 |

Exclusion codes: `E-SEG` 4, `E-DERIV` 4, `E-2D` 3, `E-NOCLF` 2, `E-NONMED` 1, `E-TYPE` 1.

Reachability: `oa_pmc_or_publisher` 30, `unreachable_paywalled` 12, `not_attempted_excluded_at_stage1` 7, `preprint_version_only` 1.

---

## 2. What was looked for as hard as the absence

### 2.1 The closest thing to a positional baseline in the whole workflow — PMID 41481488 (pos 302)

PMID 41481488 (position 302), Radiology: Imaging Cancer 2026, and it is the most useful record this workflow has produced so far. The authors build an explicit comparison model from CLINICAL AND LOCATION DATA ONLY - age, sex, tumour volume, and the tumour centroid's distance to the hip bone - and it reaches macro average AUC 0.867 (0.810, 0.909) on the validation set, BEATING the image-only DenseNet121 at 0.851 (0.775, 0.902), against the full image+clinical+location model at 0.891. Table 3 reports all six arms side by side. The authors read this as "location information improves the identification of different sacral tumor types" rather than as evidence that a six-class benchmark is largely solvable from tumour size and position. IT IS CODED FALSE, and the reason is exact: tumour volume and centroid distance are computed by the paper's OWN segmentation network from the pixels ("From this segmentation, the tumor volume is calculated, and its relative position to the hip bone is determined"), so the arm is not pixel-free and the codebook's does_not_count list rules out a hand-crafted-feature model that uses pixels. This is a near-miss to QUOTE IN THE MANUSCRIPT, not a P1 count.

### 2.2 The S5 positive — PMID 33239711 (pos 278)

PMID 33239711 (position 278), Sci Rep 2020, is the block's S5 positive and is stronger than the D9 precedent that put S5 on the board in the first place. Table 1 tabulates LABEL FREQUENCY against the ordered index of the classification unit within one acquisition - positive and negative case counts for each relative-intracranial-height band from 21-30% to 71-80% - AND reports a position-stratified AUC for each band (0.838, 0.870, 0.903, 0.845, 0.764, 0.825). The authors then state the association themselves: "we obtained the best ICH detection results in the subdivision with relative height of 41-50%, which had the most number of positive cases". They also dropped four of ten bands "since they have a very low incidence of ICH". This is a paper that measured the positional label distribution, noticed that performance tracks it, and still reported no positional baseline.

### 2.3 A headline number with no baseline to interpret it against — PMID 42266879 (pos 279)

PMID 42266879 (position 279) trains on OASIS with 67,222 of 86,437 images in one class - a 77.8% no-information rate - and reports 95.87% accuracy without ever computing the majority-class rate. The codebook requires a MEASURED value, so constant_or_prevalence is FALSE; but this is the single clearest instance in R4 of a paper whose headline number is uninterpretable without a baseline it does not report.

### 2.4 Non-imaging baselines that were found

Two included records carry a pixel-free-enough comparator with a measured value, and BOTH are clinical-variables-only arms, which the protocol deliberately keeps out of the primary endpoint and reports under secondary endpoint S1: PMID 36646808 (clinical model on age, PSA, zonal location and PI-RADS; exact-correct rate 46.7% versus the deep model's 49.2%, Fig 5a) and PMID 34765542 (clinical model on CEA, CA19-9 and CT-reported LN status; AUC 0.727 training and 0.741 external test, Supplementary Figure S6, read visually from the plotted legend because the values appear nowhere in the text). Both carry the R4-G6 caveat that one of their variables is a radiologist's imaging read. Two further clinical arms were found and coded FALSE because they contain pixel-computed quantities: PMID 37869523 (Clinical-signature AUC 0.66, but four of its eight factors are SUVmax and lesion measurements from the PET/CT) and PMID 41481488 (C-SVM macro AUC 0.774, but it contains tumour volume from the paper's own segmentation).

### 2.5 Chance asserted, never measured

One included record, PMID 42406258 (position 304): "The diagonal dotted lines in the ROC curves represent random chance performance". Asserted, never measured, so FALSE for every sub-flag and recorded in its own field, as the codebook directs.

---

## 3. Access — the binding constraint, and what moved it

FIVE records that the sample's automated oa_status hint called 'oa_pmc' turn out NOT to be in the PMC open-access subset - oa.fcgi returns idIsNotOpenAccess for all five - so Europe PMC's fullTextXML returned metadata with no body and the PMC PDF front end returned an HTML interstitial. All five were then recovered legitimately from Europe PMC's own free render-PDF endpoint (europepmc.org/articles/PMCnnnnnnn?pdf=render), which Europe PMC advertises for these records as 'Free/pdf/Europe_PMC' in its fullTextUrlList: PMIDs 36646808 (pos 264), 38423747 (269), 38164538 (287), 39557735 (305), 40879858 (306). Without that rung, four included records and one exclusion would have been coded unreachable and S6 for this block would have been 17/28 = 60.7% instead of 12/35 = 34.3%. This vindicates protocol section 10's warning that oa_status is an automated hint and not a finding, and it is the same class of recovery the v1.3 pass made.

screen_protocol.md section 7 rungs attempted in order for every record that reached stage 2: (1) PubMed Central / Europe PMC / publisher OA, including the Europe PMC free render-PDF endpoint, which recovered FIVE records the PMC front end refused; (2) the publisher's site directly via DOI resolution with a browser user-agent; (4) repository, accepted manuscript or preprint, searched through Unpaywall, OpenAlex locations, Semantic Scholar and the arXiv API by title. Rung 3 (institutional subscription) is not available to this screener and rung 5 (interlibrary loan / author request) cannot complete inside one session, so records that failed rungs 1, 2 and 4 are coded unreachable_paywalled rather than left pending - the same honest, conservative convention the earlier batches used. Under amendment D4 the ladder was NOT climbed for records whose stage1_decision was 'exclude'; where such a record happened to be in PMC anyway, the rung that worked is recorded, as D4 directs.

**Sci-Hub, LibGen and every other unauthorised full-text source. None was used and no record is reported as reachable because one of them holds it. Bot-detection and Cloudflare/JavaScript challenges were NOT circumvented: iopscience.iop.org (served a perfdrive.com challenge for PMIDs 32906091 and 34260415), linkinghub.elsevier.com, www.sciencedirect.com, pubs.rsna.org, academic.oup.com, ajnr.org, link.springer.com, ieeexplore.ieee.org, ovid.com and pmc.ncbi.nlm.nih.gov's HTML front end all served challenges, HTTP 403 or abstract-only pages to this environment and were left unread. ajnr.org and link.springer.com are additionally blocked by this environment's browsing policy, which was not worked around.**

---

## 4. The evidenced negative

Residual gap G1 from the v1.2 re-coding (a supplement that exists but cannot be searched makes a negative INADMISSIBLE, forcing 'not_assessable') was applied in full. Every included record's supplement was either retrieved and text-searched, or confirmed not to exist. Two records needed extra work: PMID 39820581, whose supplementary file 43856_2025_732_MOESM12_ESM.pdf is an image-only PDF with no text layer, was RASTERISED and all four pages READ VISUALLY (it is the Nature Portfolio Reporting Summary and contains no baseline); and PMID 42406258, whose Europe PMC supplementary endpoint returned HTTP 404, was recovered from static-content.springer.com. One further case is recorded rather than assumed: PMID 38753596's only supplementary file is a zip of image and mask files with no text document, so there is no supplementary prose that could carry a baseline. RESULT: not one included record in block R4 carries a 'not_assessable' sub-flag, so the complete-case P1 denominator of 23 is fully evidenced.

21 of the 23 included records carry an all-false trivial_baseline on a recorded 14-term search including the supplement; the other 2 carry a TRUE on clinical_or_demographic_only with the other five false on the same recorded search. Zero included records carry a "not_assessable" sub-flag, so the complete-case P1 denominator of 23 is fully evidenced.

---

## 5. Codebook conformance, rule by rule

- **D1.** Applied to all twelve unreachable records: unreachable dominates included. None of the twelve is coded "included" however clear its abstract is.
- **D2.** Applied throughout. Sub-flags are three-valued. FALSE appears only where the 14-term search including the supplement was run and is recorded in searches_run. "not_assessable" appears on all twelve unreachable records and on no included record. Two TRUE codes exist, both on clinical_or_demographic_only, both resting on a quote that itself carries the measured value (PMIDs 36646808 and 34765542); neither counts toward P1.
- **D3.** "not_applicable" used on every descriptive field of every excluded and unreachable record, and NOWHERE on an included record. Verified programmatically before writing this file.
- **D4.** All fifteen excluded records had stage1_decision="exclude". Seven of them carry fulltext_reachable="not_attempted_excluded_at_stage1" (PMIDs 41903680, 34859922, 38467345, 39992333, 35914993, 38761987, 42224315). For the other eight the full text was in hand anyway - all eight are in PMC or were recovered from Europe PMC - so the rung that worked is recorded instead, exactly as D4 directs (PMIDs 39462483, 32391274, 36627354, 38164538, 37273912, 36514476, 41595557, 40879858). "unreachable_*" is reserved for the twelve records that reached stage 2, so S6 measures the reachability of the eligible-looking literature and is not inflated by records that were never eligible.
- **D5.** Did not fire: no record in this block contradicts itself between Abstract and Methods/Results on a matter of fact.
- **D6.** Applied strictly. "patient_subject" coded only where a patient-naming noun sits inside the splitting sentence itself (six records: 39820581, 38423747, 33239711, 33194680, 37370944, 40040863). Refused on PMIDs 36646808 ("the data"), 34765542 (unnamed), 41481488 ("each tumor category"), 40506998 (unnamed), 42406258 (unnamed), 36950474 ("the initial data sets"), 37284168 ("the training dataset"), 36033909 (unnamed), 41150022 (unnamed). PMID 37370944 is the G3 configuration and, as in the sealed re-coding, the LITERAL word-list test is applied; here the literal and substantive readings agree because there is one image per patient.
- **D7.** Used on PMIDs 42266879 (Table 1 column headers "Training images"/"Testing images") and 38753596 (Table 2 per-nodule train/test counts) to read the split unit off a table.
- **D8.** "lesion_or_roi" used once, on PMID 38753596, whose split is over annotated nodules. D8(b) did not fire: no record defers its Methods to an unsampled companion paper.
- **D9.** Applied to every positional_distribution_reported code and it FIRED ONCE, on PMID 33239711, whose Table 1 tabulates positive and negative case counts against the ordered relative-intracranial-height band that IS the classification unit, and reports a position-stratified AUC for each band. Every other record came out "no"; the near misses (tumour-centroid-to-hip-bone distance in 41481488, imaging plane in 40363369, RGB superposition of adjacent slice positions in 37284168, slices-per-scan in 39820581 and 40506998, liver-metastasis distribution in 37869523, anatomical zones in 36646808 and 42147868) are all anatomical-region or scan-geometry axes, which D9 codes "no".
- **D10.** Applied via the operational test recorded in paper/screen_recoded.json: E-NOCLF, not E-SEG, where a categorical class decision IS scored with an I4-list metric but is not produced by a fitted classifier, including where the decider is a threshold on a continuous quantity the network produced. Two records: 41903680 (">=5 AI-positive slices" threshold on nnU-Net output) and 36627354 (regression, no class decision at all). E-SEG was kept where its qualifier genuinely holds (34859922, 38164538, 39992333, 37273912) and residual gap G5 is the nearest precedent for 38164538.
- **D11.** "include_provisional" was never used. Every record is "exclude" (only where an exclusion code is unambiguous from the abstract alone) or "go_to_fulltext".
- **D12.** modality is the modality of the input acquisition throughout. mixed_modality is true only where two acquisition modalities enter the model (PMID 36033909, chest X-ray and CT into the same architectures on separate datasets). PET in PMID 38761987 and MRI in PMID 35914993 are reference standards, not inputs, and do not make those records "multiple".
- **D13.** "not_stated" used wherever a paper is simply silent about code; "none" was never used. "on_request" used once, on PMID 37284168.
- **D14.** "na_only_one_unit_reported" used on 22 of the 23 included records, including the three whose evaluation_unit_reported is itself "unclear" (41150022 is "slice", but 40363369, 42266879 and 37370944 are "unclear" and still take na_only_one_unit_reported). "patient" is used on the one record reporting two units, PMID 33239711.

---

## 6. Residual gaps found in this block

### R4-G6 (material)

**Case.** The codebook admits "the clinical-only arm of a clinical+radiomics nomogram" as clinical_or_demographic_only TRUE, but the sub-flag's own definition requires "with no imaging". Real clinical-only arms routinely contain a RADIOLOGIST-REPORTED imaging variable - PI-RADS score, CT-reported lymph-node status, zonal location - which is imaging information but is not a quantity any model computes from pixels. The codebook does not say which way that goes.

**Records.** PMID 36646808, PMID 34765542

**Rule applied uniformly in this block.** A radiologist-reported categorical imaging assessment alongside labs and demographics does NOT disqualify the arm (coded TRUE, caveat recorded); a quantity COMPUTED FROM PIXELS by the paper's own pipeline - tumour volume, tumour centroid distance, SUVmax, lesion diameter, lesion count, mean ADC - DOES disqualify it, because does_not_count rules out "a radiomics or hand-crafted-feature model - it uses pixels" (coded FALSE on PMIDs 41481488, 37869523 and 33194680).

**Direction.** The rule raises S1 on two records and lowers it on two others, so it is not applied selectively. If an adjudicator reads "with no imaging" strictly, S1 for this block falls from 2/23 to 0/23. P1 is unaffected under every reading.

**Recommendation.** A future amendment should state explicitly whether a radiologist-reported imaging finding used as a tabular covariate counts as "imaging" for this sub-flag.

### R4-G7 (minor)

**Case.** Rule A5 fixes the headline as the number in the abstract's results sentence, but rule A2 requires a mixed 2D/3D paper to be coded on its VOLUMETRIC arm. When the abstract reports only the 2D arm, the two rules point at different numbers.

**Records.** PMID 36033909

**Consequence.** A5's own fallback ("If the abstract gives none, take the first row of the first results table for the proposed method") was applied to the CT arm, giving accuracy 0.9859 with headline_selection_rule="first_results_table_row".

**Recommendation.** State in A5 that "the abstract's results sentence" means the abstract's results sentence FOR THE CODED ARM.

### R4-G8 (minor)

**Case.** oa_status in screen_sample.json called five records "oa_pmc" that are PMC author-manuscript deposits outside the open-access subset, so no machine-readable full text exists at the endpoint the hint implies.

**Records.** PMID 36646808, PMID 38423747, PMID 38164538, PMID 39557735, PMID 40879858

**Consequence.** All five were recovered from Europe PMC's free render-PDF endpoint, so none is lost; but a screener who trusted the hint and stopped at fullTextXML would have coded four included records unreachable.

**Recommendation.** Add the Europe PMC render-PDF endpoint to the section 7 rung-1 procedure explicitly, and re-run it over the still-unreachable records of the main sample and of blocks R1-R3.

---

## 7. Every record

`incl.` = final_inclusion (I included / X excluded / U unreachable). `P1` is the four-flag zero-image family; it is `0` on every row.

| pos | PMID | year | venue | incl. | code | eval unit | headline | split unit | P1 | S1 | S5 |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 261 | 41150022 | 2025 | J Imaging | I | — | slice | na_only_one_unit_reported | random_unit_not_stated | 0 | no | no |
| 262 | 39820581 | 2025 | Commun Med (Lond) | I | — | patient | na_only_one_unit_reported | patient_subject | 0 | no | no |
| 263 | 34642754 | 2021 | Cereb Cortex | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 264 | 36646808 | 2023 | Br J Cancer | I | — | patient | na_only_one_unit_reported | random_unit_not_stated | 0 | **yes** | no |
| 265 | 31425026 | 2020 | IEEE Trans Med Imaging | I | — | lesion | na_only_one_unit_reported | scan_or_study | 0 | no | no |
| 266 | 39462483 | 2025 | J Biophotonics | X | E-NONMED | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 267 | 42136201 | 2026 | Invest Radiol | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 268 | 40363369 | 2025 | Sensors (Basel) | I | — | unclear | na_only_one_unit_reported | random_unit_not_stated | 0 | no | no |
| 269 | 38423747 | 2024 | AJNR Am J Neuroradiol | I | — | patient | na_only_one_unit_reported | patient_subject | 0 | no | no |
| 270 | 36125375 | 2023 | Radiology | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 271 | 39128599 | 2024 | J Neurosci Methods | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 272 | 32315264 | 2020 | Radiology | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 273 | 41903680 | 2026 | J Endod | X | E-NOCLF | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 274 | 36240594 | 2022 | Comput Biol Med | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 275 | 34859922 | 2022 | NMR Biomed | X | E-SEG | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 276 | 32391274 | 2020 | Front Oncol | X | E-DERIV | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 277 | 36627354 | 2023 | Sci Rep | X | E-NOCLF | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 278 | 33239711 | 2020 | Sci Rep | I | — | both | patient | patient_subject | 0 | no | **figure_or_table** |
| 279 | 42266879 | 2026 | Front Artif Intell | I | — | unclear | na_only_one_unit_reported | slice_or_image | 0 | no | no |
| 280 | 42147868 | 2026 | Quant Imaging Med Surg | I | — | patient | na_only_one_unit_reported | unclear | 0 | no | no |
| 281 | 32906091 | 2020 | J Neural Eng | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 282 | 33194680 | 2020 | Front Oncol | I | — | patient | na_only_one_unit_reported | patient_subject | 0 | no | no |
| 283 | 37370944 | 2023 | Diagnostics (Basel) | I | — | unclear | na_only_one_unit_reported | patient_subject | 0 | no | no |
| 284 | 33677163 | 2021 | Epilepsy Res | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 285 | 38467345 | 2024 | Neuroimage | X | E-DERIV | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 286 | 36950474 | 2023 | Eur J Radiol Open | I | — | slice | na_only_one_unit_reported | random_unit_not_stated | 0 | no | no |
| 287 | 38164538 | 2023 | AJNR Am J Neuroradiol | X | E-SEG | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 288 | 39992333 | 2024 | Fa Yi Xue Za Zhi | X | E-SEG | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 289 | 38810473 | 2024 | Comput Biol Med | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 290 | 37273912 | 2023 | Neural Comput Appl | X | E-SEG | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 291 | 34260415 | 2021 | Biomed Phys Eng Express | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 292 | 35914993 | 2022 | Ultrasound Med Biol | X | E-2D | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 293 | 40040863 | 2025 | EClinicalMedicine | I | — | patient | na_only_one_unit_reported | patient_subject | 0 | no | no |
| 294 | 36514476 | 2022 | Healthc Technol Lett | X | E-DERIV | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 295 | 40790336 | 2025 | Sci Rep | I | — | slice | na_only_one_unit_reported | slice_or_image | 0 | no | no |
| 296 | 40506998 | 2025 | Diagnostics (Basel) | I | — | patient | na_only_one_unit_reported | random_unit_not_stated | 0 | no | no |
| 297 | 34765542 | 2021 | Front Oncol | I | — | patient | na_only_one_unit_reported | random_unit_not_stated | 0 | **yes** | no |
| 298 | 41595557 | 2025 | Biomedicines | X | E-TYPE | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 299 | 37284168 | 2023 | Comput Math Methods Med | I | — | slice | na_only_one_unit_reported | random_unit_not_stated | 0 | no | no |
| 300 | 36033909 | 2022 | Inform Med Unlocked | I | — | slice | na_only_one_unit_reported | random_unit_not_stated | 0 | no | no |
| 301 | 38753596 | 2024 | PLoS One | I | — | lesion | na_only_one_unit_reported | lesion_or_roi | 0 | no | no |
| 302 | 41481488 | 2026 | Radiol Imaging Cancer | I | — | patient | na_only_one_unit_reported | random_unit_not_stated | 0 | no | no |
| 303 | 41271174 | 2026 | Radiother Oncol | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 304 | 42406258 | 2026 | Brain Inform | I | — | patient | na_only_one_unit_reported | random_unit_not_stated | 0 | no | no |
| 305 | 39557735 | 2025 | J Imaging Inform Med | I | — | slice | na_only_one_unit_reported | slice_or_image | 0 | no | no |
| 306 | 40879858 | 2026 | J Imaging Inform Med | X | E-DERIV | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 307 | 36403310 | 2023 | Med Image Anal | U | — | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 308 | 38761987 | 2024 | J Am Soc Echocardiogr | X | E-2D | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |
| 309 | 37869523 | 2023 | EClinicalMedicine | I | — | patient | na_only_one_unit_reported | external_cohort_only | 0 | no | no |
| 310 | 42224315 | 2026 | IEEE Trans Biomed Eng | X | E-2D | not_applicable | not_applicable | not_applicable | 0 | — | not_applicable |

---

## 8. Included records in detail

### pos 261 — PMID 41150022 — *J Imaging* (2025, CT)

Optimized Lung Nodule Classification Using CLAHE-Enhanced CT Imaging and Swin Transformer-Based Deep Feature Extraction.

- **dataset** LIDC-IDRI (public) · **organ** lung · **n patients** None · **n test** None · **n slices/images** 11417
- **headline** accuracy = 0.958 on the cross_validation set (abstract_sentence), scope single_modality_arm, interval sd_across_folds
- **evaluation unit** `slice` — "Figure 4 Example of feature extraction process. ( a ) CLAHE-enhanced CT slice." and "Table 1 Distribution of images after applying the labeling strategy. Class Number of Images Malignant 6568 Benign 4849" (Results, Table 1 and Fig 4 caption). 11,417 scored images from 1,018 scans, and the input is a 512x512 grayscale CT slice.
- **headline unit** `na_only_one_unit_reported` — Only the image (slice) unit is scored anywhere; no patient-level metric is reported. D14 applies.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "To ensure fairness and reproducibility, stratified train-test splits were employed to preserve class distributions." (Methods, Classification). No unit is named in the splitting sentence or in any table caption describing the split.
- **positional distribution** `no` — No histogram, mean/sd or position-stratified metric along the slice axis appears anywhere. The only "position" hit is "positional" in the transformer architecture description, which is scan geometry of the network, not label position (D9 codes this "no").
- **input** `2D_slice` · label broadcast `true` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** Nodule malignancy scores are broadcast to the scan and the scored unit is the image. n_patients coded NULL: the paper states "1018 thoracic CT scans" and never states a distinct patient count, and the codebook forbids back-computing. Data availability names LIDC-IDRI only; the paper is silent on code, so D13 gives not_stated.

### pos 262 — PMID 39820581 — *Commun Med (Lond)* (2025, CT)

Harnessing deep learning to detect bronchiolitis obliterans syndrome from chest CT.

- **dataset** private_single_centre (private) · **organ** lung · **n patients** 75 · **n test** 75 · **n slices/images** 391
- **headline** AUC = 0.9 on the cross_validation set (abstract_sentence), scope single_modality_arm, interval ci_clustered_by_subject
- **evaluation unit** `patient` — "We evaluated this approach on CT scans from 75 post-transplant patients, including 26 with BOS, and used a ROC-AUC metric to assess performance." (Abstract, Methods) with aggregation to patient described in Methods.
- **headline unit** `na_only_one_unit_reported` — Only the patient-level ROC-AUC is reported; no slice-level metric appears.
- **split unit** `patient_subject` (disjointness stated_and_checked) — "We randomly divided the cohort of 26 patients with BOS and 49 non-BOS patients into four splits of 5 patients with BOS and 10 patients without BOS and one split of 6 patients with BOS and 9 patients without BOS." (Supplementary Reporting Summary, Replication; same procedure in Methods). A patient-naming noun sits inside the splitting sentence and is the divided unit (D6 satisfied).
- **positional distribution** `no` — The paper discusses how many slices are retained per scan ("retaining too many slices defeats the purpose of discarding slices", Supplement) - scan geometry, not label position along the slice axis. D9 codes this "no".
- **input** `3D_volume` · label broadcast `unclear` · code `public_link_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - all 12 supplementary files retrieved from Europe PMC and searched; the one image-only PDF (43856_2025_732_MOESM12_ESM.pdf, the Nature Portfolio Reporting Summary) was RASTERISED and all four pages READ VISUALLY, per residual gap G1. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - all 12 supplementary files retrieved from Europe PMC and searched; the one image-only PDF (43856_2025_732_MOESM12_ESM.pdf, the Nature Portfolio Reporting Summary) was RASTERISED and all four pages READ VISUALLY, per residual gap G1. no match indicating any zero-image baseline.
- **notes** Uncertainty: "We therefore resorted to bootstrapping by hierarchical case resampling... ten thousand simulations to estimate the 95% confidence intervals" - a subject-clustered interval, so this record is in S8. Supplement G1 handled by visual reading of the image-only PDF; it is a reporting checklist and contains no baseline.

### pos 264 — PMID 36646808 — *Br J Cancer* (2023, MRI)

High-throughput precision MRI assessment with integrated stack-ensemble deep learning can enhance the preoperative prediction of prostate cancer Gleason grade.

- **dataset** private_multi_centre (private) · **organ** prostate · **n patients** 1442 · **n test** 539 · **n slices/images** None
- **headline** AUC = 0.762 on the external set (abstract_multiple_took_external), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `patient` — "challenging to achieve a per-lesion imaging correlation with prostatectomy specimens in retrospective data, the unit of assessment in this study was per-patient." (Discussion).
- **headline unit** `na_only_one_unit_reported` — Only the per-patient unit is scored; no slice-level metric is reported.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "we randomly split the data from centre 1 into training (n = 672) for model development and test (n = 231) cohort for internal validation, respectively." (Methods). The divided noun is "the data"; no patient-naming noun sits inside the splitting sentence, so D6 refuses the upgrade even though Table 1 counts patients elsewhere.
- **positional distribution** `no` — "zonal location" and "PI-RADS" are anatomical-zone descriptors, not the ordered index of the classification unit within one acquisition; D9 codes an anatomical region that is not the unit ordering as "no".
- **input** `2D_slice` · label broadcast `unclear` · code `public_link_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": true, "other_non_imaging": false}`
- **trivial_baseline evidence** "the performance of PRISK versus a clinical model using age, PSA, zonal location and PI-RADS score for ISUP GGs is plotted. It shows that PRISK results in a relatively similar upgrade rate (11.9% vs 11.8%), lower downgrade rate (10.1% vs 11.2%)" and Fig 5a confusion matrices give the same metric for both arms: "correct: 709 (49.2%)" for PRISK versus "correct: 673 (46.7%)" for the Clinical model (Results, Fig 5a).
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none retrievable for this record (PMC deposit is a scanned-manuscript PDF with no separate supplementary package); the version-of-record PDF was searched in full including all figures and tables. no match indicating any zero-image baseline.
- **notes** S1 ONLY, NOT P1. The clinical comparator has a MEASURED value on the same metric as the proposed model (exact-correct classification rate, 46.7% vs 49.2%). CAVEAT RECORDED, see block-level residual gap R4-G6: two of the four clinical variables (zonal location, PI-RADS score) are a radiologist reading of the MRI, so the arm is not strictly pixel-free. It is coded TRUE because the codebook explicitly admits "the clinical-only arm of a clinical+radiomics nomogram" and because no quantity in the arm is computed from pixels by any model. Recovered at rung 1 via the Europe PMC free render-PDF endpoint after the PMC front end and nature.com both refused this environment.

### pos 265 — PMID 31425026 — *IEEE Trans Med Imaging* (2020, CT)

Automatic Pulmonary Nodule Detection in CT Scans Using Convolutional Neural Networks Based on Maximum Intensity Projection.

- **dataset** LIDC-IDRI; LUNA16 (public) · **organ** lung · **n patients** None · **n test** None · **n slices/images** None
- **headline** sensitivity = 0.9267 on the cross_validation set (abstract_sentence), scope single_modality_arm, interval none
- **evaluation unit** `lesion` — "Our proposed method achieves sensitivity of 92.67% with 1 false positive per scan and sensitivity of 94.19% with 2 false positives per scan for lung nodule detection on 888 scans in the LIDC-IDRI dataset." (Abstract). The scored unit is the nodule candidate.
- **headline unit** `na_only_one_unit_reported` — Only the nodule-candidate unit is scored; no patient-level or slice-level metric is reported.
- **split unit** `scan_or_study` (disjointness not_stated) — "In the nodule candidate detection stage, the whole dataset is equally split into 10 subsets in the LUNA16 competition. We perform 10-fold cross-validation... For each fold, we use 63% of the dataset for training, 27% of the dataset for validation, and 10% of the dataset for testing." (Methods, Training Process). The LUNA16 subsets are scan-wise; no patient-naming noun appears in the splitting sentence, so D6 refuses patient_subject.
- **positional distribution** `no` — Slab thickness (5/10/15 mm MIP) is scan geometry, not label position along the slice axis; D9 codes this "no".
- **input** `mixed` · label broadcast `na` · code `not_stated` · access `preprint_version_only` (preprint)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists for the arXiv preprint. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists for the arXiv preprint. no match indicating any zero-image baseline.
- **notes** E-PROJ CONSIDERED AND REJECTED: the input is not a collapsed projection with no slice axis - one of the four streams is "1 mm axial section slices" and the MIP streams are thin slabs at ordered z positions, so the slice axis survives. E-SEG CONSIDERED AND REJECTED: the false-positive-reduction stage is a fitted CNN classifier assigning nodule/non-nodule to each candidate and is evaluated with sensitivity, a metric on the I4 list, so E-SEG's qualifier "with NO categorical class decision evaluated" fails; a classifier IS fitted, so D10/E-NOCLF does not apply either. CODED FROM A PREPRINT (arXiv:1904.05956, rung 4) because IEEE Xplore refused this environment; the preprint carries the banner "This work has been submitted to the IEEE for possible publication", and the published abstract's sensitivities (92.7%/94.2%) match the preprint's (92.67%/94.19%). This record belongs in the version-of-record sensitivity analysis.

### pos 268 — PMID 40363369 — *Sensors (Basel)* (2025, MRI)

Multimodal MRI Image Fusion for Early Automatic Staging of Endometrial Cancer.

- **dataset** private_multi_centre (private) · **organ** other · **n patients** 122 · **n test** None · **n slices/images** None
- **headline** accuracy = 1.0 on the internal_held_out set (abstract_sentence), scope single_modality_arm, interval none
- **evaluation unit** `unclear` — "magnetic resonance imaging (MRI) images in each of the three planes (sagittal, coronal, and transverse) are cropped, enhanced, and classified" (Abstract). The paper reports metrics over "images" and never states whether one scored image is one slice or one patient; the codebook makes this a substantive finding, not a coding failure.
- **headline unit** `na_only_one_unit_reported` — Exactly one unit is scored, so D14 gives na_only_one_unit_reported even though that unit is itself coded unclear.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "Experimental data were randomly divided 8:2 into training and test datasets." (Methods). No unit is named in the splitting sentence or in any table caption describing the split.
- **positional distribution** `no` — "position" occurs 39 times, every one of them meaning the imaging plane (sagittal/coronal/transverse position) or the transformer's positional encoding; neither is the ordered index of the classification unit within one acquisition, so D9 codes "no".
- **input** `2D_slice` · label broadcast `unclear` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** Reported accuracy, recall and specificity are all exactly 1.000 after three-plane fusion on 122 patients with an unstated split unit - the configuration this project's thesis predicts, though the paper reports no zero-image baseline that would test it.

### pos 269 — PMID 38423747 — *AJNR Am J Neuroradiol* (2024, MRI)

Identifying Patients with CSF-Venous Fistula Using Brain MRI: A Deep Learning Approach.

- **dataset** private_single_centre (private) · **organ** brain · **n patients** 129 · **n test** 129 · **n slices/images** None
- **headline** AUC = 0.8668 on the cross_validation set (abstract_sentence), scope single_modality_arm, interval sd_across_folds
- **evaluation unit** `patient` — "In discriminating between positive and negative cases for CSF-venous fistulas, the classifier demonstrated an average area under the receiver operating characteristic curve of 0.8668 with a standard deviation of 0.0254 across the folds." (Results); the data "was split into 5 folds at the patient level".
- **headline unit** `na_only_one_unit_reported` — Only the patient unit is scored; no slice-level metric is reported.
- **split unit** `patient_subject` (disjointness stated_and_checked) — "The data set was split into 5 folds at the patient level and stratified by label." and "The data were split into 5 folds at the patient level by using the GroupKfold module from the scikit-learn package" (Methods).
- **positional distribution** `no` — No histogram, mean/sd of relative position, or position-stratified metric along the slice axis appears anywhere in the paper.
- **input** `3D_volume` · label broadcast `false` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** Recovered at rung 1 via the Europe PMC free render-PDF endpoint; the PMC deposit carries no machine-readable body and ajnr.org returned HTTP 403 and is blocked by this environment's browsing policy. GroupKFold at the patient level is the strongest split statement in this block.

### pos 278 — PMID 33239711 — *Sci Rep* (2020, CT)

Detection and classification of intracranial haemorrhage on CT images using a novel deep-learning algorithm.

- **dataset** private_single_centre (private) · **organ** brain · **n patients** 250 · **n test** 84 · **n slices/images** 9085
- **headline** AUC = 0.859 on the internal_held_out set (abstract_sentence), scope single_modality_arm, interval none
- **evaluation unit** `both` — "For the detection of ICH with the summation of all the computed tomography (CT) images for each case, the area under the ROC curve (AUC) was 0.859" (Abstract) AND Table 1, which reports AUC per subdivision, where "the CT images were divided into 10 subdivisions based on the intracranial height, where each subdivision comprised 3–4 CT brain slices for a case" (Methods). The same metric (AUC) is reported at two units.
- **headline unit** `patient` — The abstract's first results sentence is the case-level number: "For the detection of ICH with the summation of all the computed tomography (CT) images for each case, the area under the ROC curve (AUC) was 0.859, and the sensitivity and the specificity were 78.0% and 80.0%, respectively."
- **split unit** `patient_subject` (disjointness stated_only) — "The 250 patients were randomly divided into 166 for the training set and 84 for the validation set." (Methods, Subjects). A patient-naming noun sits inside the splitting sentence and is the divided unit, so D6 is satisfied.
- **positional distribution** `figure_or_table` — VERBATIM, Results, Table 1: "Table 1 ICH detection results using the validation set for the six subdivisions. Subdivision Resolution Positive case Negative case Hidden nodes AUC Sensitivity % Specificity % Accuracy % 21–30% 28 × 28 12 72 40 0.838 91.7 70.8 73.8 31–40% 24 × 24 27 57 80 0.870 92.6 73.7 79.8 41–50% 30 × 30 40 44 40 0.903 82.5 84.1 83.3 51–60% 80 × 80 37 47 120 0.845 70.3 87.2 79.8 61–70% 30 × 30 31 53 240 0.764 83.9 69.8 75.0 71–80% 30 × 30 21 63 40 0.825 81.0 71.4 73.8 Average – – – – 0.841 83.7 76.2 77.6" Also verbatim, Methods: "Of the 10 subdivisions based on the intracranial height, the subdivisions with relative heights of 0–10%, 11–20%, 81–90%, and 91–100% were excluded from this study since they have a very low incidence of ICH." And verbatim, Results: "Among the six subdivisions, we obtained the best ICH detection results in the subdivision with relative height of 41–50%, which had the most number of positive cases".
- **input** `2.5D_stack` · label broadcast `true` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** THE STRONGEST S5 POSITIVE IN BLOCK R4, AND STRONGER THAN THE D9 PRECEDENT (PMID 42130124). Table 1 tabulates LABEL FREQUENCY (positive cases and negative cases) against the ORDERED INDEX OF THE CLASSIFICATION UNIT WITHIN ONE ACQUISITION (relative intracranial height band, 21-30% through 71-80%), AND reports a position-stratified AUC for each band, which the codebook names explicitly as qualifying. The authors then observe the association themselves: the best-performing band is the one with the most positive cases. THIS IS NOT A ZERO-IMAGE BASELINE - the model still sees pixels in every band - but it is the closest any record in R4 comes to measuring what this project formalises, and it is reported for that reason. E-PROJ CONSIDERED AND REJECTED: the headline arm sums the whole stack into one image, which alone would be a collapsed projection, but the per-subdivision arm sums only 3-4 adjacent slices at a defined ordered height, which is 2.5D under A8 and retains the slice axis; A2's logic admits the record on the arm that is reported separately.

### pos 279 — PMID 42266879 — *Front Artif Intell* (2026, MRI)

Ensemble Deep Learning Denoising (EDLD) model and optimized OTSU segmentation for Alzheimer's disease diagnosis using MRI images.

- **dataset** OASIS (public) · **organ** brain · **n patients** None · **n test** None · **n slices/images** 86437
- **headline** accuracy = 0.9587 on the internal_held_out set (abstract_sentence), scope single_modality_arm, interval none
- **evaluation unit** `unclear` — "Table 1 Dataset details. Classes Training images Testing images Total MD 4,726 276 5,002 MOD 420 68 488 NOD 65,563 1,659 67,222 VMD 12,567 1,158 13,725 Total 83,276 3,161 86,437" (Methods 3.1). The scored unit is the "image" and the paper never states whether one image is one slice or one subject.
- **headline unit** `na_only_one_unit_reported` — Exactly one unit is scored, so D14 gives na_only_one_unit_reported even though that unit is itself coded unclear.
- **split unit** `slice_or_image` (disjointness not_stated) — "Table 1 Dataset details. Classes Training images Testing images Total" with per-class train/test image counts (Methods 3.1). Under D7 a unit named in a table column header IS a named unit, so this is not random_unit_not_stated.
- **positional distribution** `no` — The 14 "position" hits are all grasshopper-swarm optimiser positions in the LGOA equations, not label position along the slice axis; D9 codes this "no".
- **input** `2D_slice` · label broadcast `unclear` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** n_patients coded NULL: the paper never states a subject count for OASIS, only 86,437 images, and the codebook forbids back-computing from image counts. Severe class imbalance (NOD 67,222 of 86,437 = 77.8%) with no majority-class baseline reported anywhere - the paper reports 95.87% accuracy against a 77.8% no-information rate it never computes. This is a near-miss for constant_or_prevalence but the codebook requires a MEASURED value and none is given, so the flag is FALSE.

### pos 280 — PMID 42147868 — *Quant Imaging Med Surg* (2026, CT)

Ensemble deep learning model based on CT scans: differentiating and subtype-classifying pancreatic inflammations and tumors, and predicting pancreatic lesion invasiveness.

- **dataset** private_multi_centre (private) · **organ** pancreas · **n patients** 6740 · **n test** None · **n slices/images** None
- **headline** accuracy = 0.958 on the external set (abstract_multiple_took_external), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `patient` — "The ensemble DL model showed high accuracy in both differentiating inflammatory and tumor lesions (internally 95.1%, externally 95.8%)" over "6,740 patients' pancreatic CT images" (Abstract; Methods, Dataset description).
- **headline unit** `na_only_one_unit_reported` — Only the patient/lesion-per-patient unit is scored for classification; no slice-level classification metric is reported (Dice and IoU belong to the segmentation arm and A3 requires them to be ignored).
- **split unit** `unclear` (disjointness not_stated) — "This multicenter study comprised three strategically partitioned cohorts to ensure robust model development and validation. An internal training cohort was utilized for the primary construction of the integrated DL framework." (Methods, Dataset description). No unit and no ratio are named anywhere in the split description, its tables or its figure legends.
- **positional distribution** `no` — "The regions of model focus were divided into three parts: peri-pancreatic, intra-pancreatic, and intra-tumoral" - anatomical regions that are not the unit ordering, which D9 codes explicitly as "no".
- **input** `2D_slice` · label broadcast `unclear` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - all four supplementary files retrieved from Europe PMC and searched (70,818 characters). no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - all four supplementary files retrieved from Europe PMC and searched (70,818 characters). no match indicating any zero-image baseline.
- **notes** A3 applied: multi-task segmentation plus classification, classification arm coded, Dice and IoU ignored entirely.

### pos 282 — PMID 33194680 — *Front Oncol* (2020, MRI)

A Deep Learning Model to Predict the Response to Neoadjuvant Chemoradiotherapy by the Pretreatment Apparent Diffusion Coefficient Images of Locally Advanced Rectal Cancer.

- **dataset** private_single_centre (private) · **organ** other · **n patients** 700 · **n test** 200 · **n slices/images** None
- **headline** AUC = 0.851 on the internal_held_out set (abstract_sentence), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `patient` — "All participants (n = 700) were divided into a training group (n = 500, from December 2009 to March 2015) and a test group (n = 200, from March 2015 to July 2016) chronically." with AUC reported over the 200-participant test group (Methods; Results).
- **headline unit** `na_only_one_unit_reported` — Only the participant unit is scored; no slice-level metric is reported.
- **split unit** `patient_subject` (disjointness stated_only) — "All participants (n = 700) were divided into a training group (n = 500, from December 2009 to March 2015) and a test group (n = 200, from March 2015 to July 2016) chronically." (Methods). "participants" is on D6's admitted word list and is the divided unit.
- **positional distribution** `no` — No histogram, mean/sd of relative position or position-stratified metric along the slice axis appears anywhere.
- **input** `2D_slice` · label broadcast `unclear` · code `public_link_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** NEAR MISS, CODED FALSE: the paper compares the deep model against "the prediction by mean ADC value" (AUC 0.723 vs 0.851, DeLong p = 0.018). Mean ADC is a hand-crafted quantity computed FROM PIXELS inside the delineated tumour, and the codebook's does_not_count list rules out "a radiomics or hand-crafted-feature model - it uses pixels". It is therefore not a zero-image baseline and no sub-flag is set. A chronological rather than random split is used, which is stronger than most of this block.

### pos 283 — PMID 37370944 — *Diagnostics (Basel)* (2023, MRI)

Visual Cascaded-Progressive Convolutional Neural Network (C-PCNN) for Diagnosis of Meniscus Injury.

- **dataset** private_single_centre (private) · **organ** musculoskeletal · **n patients** 1396 · **n test** 1396 · **n slices/images** 1396
- **headline** accuracy = 0.898 on the cross_validation set (abstract_sentence), scope single_modality_arm, interval none
- **evaluation unit** `unclear` — "A total of 1396 images collected in the hospital were used for training and testing." and "We divided the images of all 1396 patients into five parts" (Abstract; Methods, 5-Fold Cross Validation). One image per patient, but the paper never states whether an image is a slice or a series.
- **headline unit** `na_only_one_unit_reported` — Exactly one unit is scored, so D14 applies.
- **split unit** `patient_subject` (disjointness stated_only) — "We divided the images of all 1396 patients into five parts labeled as K1, K2, K3, K4, and K5." (Methods, 5-Fold Cross Validation Method). D6's literal word-list test is satisfied - "patients" sits inside the splitting sentence - and the 1:1 correspondence between the 1396 images and the 1396 patients means the literal reading and the substantive reading agree here. Residual gap G3 of the v1.2 re-coding kept the literal test, and it is applied the same way here.
- **positional distribution** `no` — Anterior horn versus posterior horn is an anatomical sub-region of the meniscus, not the ordered index of the classification unit within one acquisition; D9 codes this "no".
- **input** `2D_slice` · label broadcast `unclear` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** The only "baseline" hit is "Among the different baseline networks, ResNet50 demonstrated the best performance" - a comparison against another imaging network, which the does_not_count list rules out explicitly.

### pos 286 — PMID 36950474 — *Eur J Radiol Open* (2023, MRI)

Optimizing MRI-based brain tumor classification and detection using AI: A comparative analysis of neural networks, transfer learning, data augmentation, and the cross-transformer network.

- **dataset** Figshare brain tumor dataset (BTD); Brain MRI Images for Brain Tumor Detection (MRI-D); TCGA-LGG (public) · **organ** brain · **n patients** 596 · **n test** None · **n slices/images** 7246
- **headline** accuracy = 0.97 on the internal_held_out set (abstract_sentence), scope pooled_multi_dataset, interval none
- **evaluation unit** `slice` — "Table 1 Dataset used for training convolutional neural networks for brain tumor detection. Dataset Subjects Sequences Slices Classes Images per class | BTD 233 T1-Gd Axial, coronal and sagittal Meningioma 708 Glioma 1426 Pituitary 930 | MRI-D 253 T1WI Axial Tumors 155 Not tumors 98 | TCGA-LGG 110 T1W1, T1-Gd, FLAIR Axial Tumors 1373 Not tumors 2556" (Methods, Table 1). Subjects and slices are counted separately and the scored unit is the slice.
- **headline unit** `na_only_one_unit_reported` — Only the slice unit is scored; no patient-level metric is reported anywhere.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "The initial data sets were divided into two sets with 80 % and 20 % proportions for training and testing, respectively." (Methods, Experimental design). The divided noun is "the initial data sets"; no unit is named in the splitting sentence or in any table caption describing the split.
- **positional distribution** `no` — Table 1 records the acquisition planes (axial, coronal, sagittal) and slice counts per dataset - scan geometry, which D9 codes explicitly as "no" - and no label-versus-position tabulation appears.
- **input** `2D_slice` · label broadcast `unclear` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - all 25 supplementary files retrieved from Europe PMC and searched (78,561 characters). no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - all 25 supplementary files retrieved from Europe PMC and searched (78,561 characters). no match indicating any zero-image baseline.
- **notes** Slices from the same subject are split across train and test with a plain 80/20 division and 596 subjects contribute 7,246 slices, so subject-level leakage is structurally possible and unaddressed; the paper reports no zero-image baseline that would detect it. The only supplement "baseline" hit is the EfficientNet paper abstract quoted in the reference material.

### pos 293 — PMID 40040863 — *EClinicalMedicine* (2025, MRI)

Development of an MRI based artificial intelligence model for the identification of underlying atrial fibrillation after ischemic stroke: a multicenter proof-of-concept analysis.

- **dataset** private_multi_centre (private) · **organ** brain · **n patients** 758 · **n test** 175 · **n slices/images** None
- **headline** AUC = 0.85 on the external set (abstract_multiple_took_external), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `patient` — "Fivefold cross-validation was implemented by randomly assigning patients to five datasets." with AUC reported per cohort of patients (Methods; Findings).
- **headline unit** `na_only_one_unit_reported` — Only the patient unit is scored; no slice-level metric is reported.
- **split unit** `patient_subject` (disjointness stated_only) — "Fivefold cross-validation was implemented by randomly assigning patients to five datasets." (Methods). "patients" sits inside the splitting sentence and is the assigned unit, so D6 is satisfied.
- **positional distribution** `no` — The paper discusses "brain ischemic lesion pattern" as an anatomical/vascular-territory notion, not position along the acquired slice stack; D9 codes an anatomical axis that is not the unit ordering as "no".
- **input** `3D_volume` · label broadcast `unclear` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - retrieved from Europe PMC and searched. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - retrieved from Europe PMC and searched. no match indicating any zero-image baseline.
- **notes** A combined classifier over pre-defined radiomics features AND de novo CNN features; the CNN takes the image, so E-DERIV does not apply. No clinical-only comparator arm with a measured value is reported despite CHA2DS2-VASc being available in the cohort - a conspicuous omission recorded here, not a coded positive.

### pos 295 — PMID 40790336 — *Sci Rep* (2025, MRI)

Enhanced MRI brain tumor detection using deep learning in conjunction with explainable AI SHAP based diverse and multi feature analysis.

- **dataset** Figshare brain tumor dataset; Kaggle Brain Tumor MRI Dataset (public) · **organ** brain · **n patients** None · **n test** None · **n slices/images** 7023
- **headline** accuracy = 0.989 on the internal_held_out set (abstract_sentence), scope pooled_multi_dataset, interval ci_unspecified_method
- **evaluation unit** `slice` — "a large training dataset is used, which consists of 7023 MRI images split into 5712 images for training and 1311 images for testing" with per-class image counts for pituitary, meningioma and glioma (Methods, Dataset).
- **headline unit** `na_only_one_unit_reported` — Only the image (slice) unit is scored; no patient-level metric is reported.
- **split unit** `slice_or_image` (disjointness not_stated) — "a large training dataset is used, which consists of 7023 MRI images split into 5712 images for training and 1311 images for testing" (Methods). "images" is named inside the splitting sentence, so this is slice_or_image, not random_unit_not_stated.
- **positional distribution** `no` — The three "position" hits are all Local Binary Pattern / Fourier positional descriptors within a single image, not label position along the slice axis; D9 codes this "no".
- **input** `2D_slice` · label broadcast `unclear` · code `public_link_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** n_patients coded NULL: the paper never states a subject count, only image counts. Slices from the same subject can fall on both sides of the image-level split; no zero-image baseline is reported that would detect it.

### pos 296 — PMID 40506998 — *Diagnostics (Basel)* (2025, MRI)

GM-VGG-Net: A Gray Matter-Based Deep Learning Network for Autism Classification.

- **dataset** ABIDE (public) · **organ** brain · **n patients** 272 · **n test** None · **n slices/images** None
- **headline** accuracy = 0.96 on the internal_held_out set (abstract_sentence), scope single_modality_arm, interval none
- **evaluation unit** `patient` — "We included a total of 272 subjects, with 132 individuals diagnosed with ASD and 140 matched normal controls." and "The input layer consists of preprocessed gray matter (GM) from MRI T1-weighted images... we selected the best 70 slices that contain the most brain regions (256 x 70 x 256)" - one 3D GM volume per subject, so the scored unit is the subject (Results 3.1; Methods 2.4.1).
- **headline unit** `na_only_one_unit_reported` — Only the subject unit is scored; no slice-level metric is reported.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "split into 70% for training and 30% for validation" (Methods). No unit is named in the splitting sentence or in any table caption describing the split; the demographics table elsewhere does not upgrade the code (D6).
- **positional distribution** `no` — "we selected the best 70 slices that contain the most brain regions" is a statement about scan geometry and slice selection, which D9 codes explicitly as "no"; no label-versus-position statement appears.
- **input** `3D_volume` · label broadcast `false` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** The reported number is a VALIDATION accuracy with no independent test set; the authors say so themselves: "this work was focused on evaluating the proposed GM-VGG-Net using validation and training accuracy and error measures obtained via TensorBoard with Keras. Although these metrics offer a useful baseline, future work will combine more broad evaluation measures such as F1-score, AUC-ROC, and confusion matrices." That "baseline" refers to their own metric set, not a pixel-free comparator, so no sub-flag is set. ABIDE is multi-site and no site/scanner-metadata baseline is reported.

### pos 297 — PMID 34765542 — *Front Oncol* (2021, CT)

Deep Learning Radiomics to Predict Regional Lymph Node Staging for Hilar Cholangiocarcinoma.

- **dataset** private_multi_centre (private) · **organ** liver · **n patients** 179 · **n test** 21 · **n slices/images** None
- **headline** AUC = 0.87 on the external set (abstract_multiple_took_external), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `patient` — "Of the 179 enrolled HC patients, 90 were pathologically diagnosed with lymph node metastasis." with AUCs reported over patient cohorts (Abstract; Results).
- **headline unit** `na_only_one_unit_reported` — Only the patient unit is scored; no slice-level metric is reported.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "randomly divided at a ratio of 80%:20% for training and internal validation" (Methods). The divided noun is unnamed; no patient-naming noun sits inside the splitting sentence, so D6 refuses the upgrade despite Table 1 counting patients.
- **positional distribution** `no` — ROIs were segmented "according to the maximum cross-section layer of the arterial phase images of lesions" - a slice-selection rule, i.e. scan geometry, which D9 codes as "no". No label-versus-position tabulation appears.
- **input** `2D_slice` · label broadcast `false` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": true, "other_non_imaging": false}`
- **trivial_baseline evidence** "A clinical model was also constructed to confirm the special contribution of DLRS to the fusion model ( Supplementary Figure S6 )." (Results), and Supplementary Figure S6's own legend carries the measured values: "Training Cohort (AUC: 0.727, 95% CI: 0.648-0.806)" and "External Test Cohort (AUC: 0.741, 95% CI: 0.522-0.960)". The figure was retrieved from the supplementary DataSheet_1.docx, extracted as an image and READ VISUALLY, because the AUC values appear only inside the plotted legend and nowhere in the text.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - DataSheet_1.docx retrieved from Europe PMC and searched (33,993 characters); its embedded figure images were additionally extracted and read visually. no match indicating any zero-image baseline.
- **notes** S1 ONLY, NOT P1. CAVEAT RECORDED, see block-level residual gap R4-G6: the three selected clinical characteristics are "the CEA level, CA 19-9 level, and CT-reported LN status", and the last is a radiologist's reading of the CT, so the arm is not strictly pixel-free. Coded TRUE because the codebook explicitly admits "the clinical-only arm of a clinical+radiomics nomogram" and because no quantity in the arm is computed from pixels by any model. An adjudicator who reads "with no imaging" strictly would code this FALSE and S1 for the block would fall from 2/23 to 1/23; P1 is unaffected either way.

### pos 299 — PMID 37284168 — *Comput Math Methods Med* (2023, CT)

Enhancing Disease Classification in Abdominal CT Scans through RGB Superposition Methods and 2D Convolutional Neural Networks: A Study of Appendicitis and Diverticulitis.

- **dataset** private_single_centre (private) · **organ** abdomen_general · **n patients** None · **n test** None · **n slices/images** None
- **headline** accuracy = 0.9198 on the cross_validation set (abstract_sentence), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `slice` — "We propose a deep learning method, utilizing red, green, and blue (RGB) channel superposition images reconstructed from three slices of sequence images. Using the RGB superposition image as the input image of the model, the average accuracy was shown as 90.98% in EfficietNetB0, 91.27% in EfficietNetB2, and 91.98% in EfficietNetB4." (Abstract). Three adjacent slices predicting the centre slice's label is 2.5D, which rule A8 makes slice-level.
- **headline unit** `na_only_one_unit_reported` — Only the slice unit is scored; no patient-level metric is reported anywhere.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "splitting the training dataset into k subsets of roughly equal size and calling each subset like 'fold,' in this study, we divided 5 folds." (Methods). The divided noun is "the training dataset"; no unit is named in the splitting sentence or in any table caption describing the split.
- **positional distribution** `no` — The 53 "position" hits are all RGB channel superposition of adjacent slice positions within one input image - the construction of the input, i.e. scan geometry - not a tabulation of label frequency against the ordered slice index; D9 codes this "no".
- **input** `2.5D_stack` · label broadcast `true` · code `on_request` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** THE SHARPEST STRUCTURAL NEAR-MISS FOR THIS PROJECT'S THESIS IN BLOCK R4 AFTER 33239711: a 2.5D slice-level classifier, folds split over an unnamed unit, and no patient count stated anywhere, so slices from one patient can sit on both sides of every fold boundary and nothing in the paper would reveal it. The paper reports no zero-image baseline, and the 14-term search over the full text confirms the negative.

### pos 300 — PMID 36033909 — *Inform Med Unlocked* (2022, CT)

SEL-COVIDNET: An intelligent application for the diagnosis of COVID-19 from chest X-rays and CT-scans.

- **dataset** SARS-CoV-2 CT-scan dataset (CT dataset 4); COVID-19 Radiography Database and two other chest X-ray sets (public) · **organ** lung · **n patients** None · **n test** None · **n slices/images** None
- **headline** accuracy = 0.9859 on the internal_held_out set (first_results_table_row), scope single_modality_arm, interval none
- **evaluation unit** `slice` — "Table 9 Evaluation performance of binary-class DL models used in the SEL-COVIDNET on CT dataset 4 (0: COVID-19, 1: No-finding)... DensNet121 0 0.992 0.996 0.989 0.989 0.992 0.984 98.59" (Results, Table 9). The CT arm scores individual CT slice images.
- **headline unit** `na_only_one_unit_reported` — Only the image unit is scored in the CT arm; no patient-level metric is reported.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "divided 80-20 to begin the training phase of one of nine deep learning models that have been tuned" (Methods). No unit is named in the splitting sentence or in any table caption describing the split.
- **positional distribution** `no` — No histogram, mean/sd of relative position or position-stratified metric along the slice axis appears anywhere.
- **input** `2D_slice` · label broadcast `unclear` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** A2 applied: a volumetric (CT) arm exists AND is reported separately in Table 9, so the record is included, the volumetric arm is coded, and mixed_modality is true (X-ray and CT both enter the same architectures on separate datasets, which is what D12 requires). HEADLINE RULE NOTE: the abstract's results sentence reports only the X-ray arm (98.52% multi-class, 99.77% binary), which is the ineligible modality, so A5's fallback fires and the headline is the first row of the first CT results table. This is flagged because A5 does not explicitly cover the case where the abstract's number belongs to the arm A2 excludes.

### pos 301 — PMID 38753596 — *PLoS One* (2024, CT)

Attention pyramid pooling network for artificial diagnosis on pulmonary nodules.

- **dataset** LIDC-IDRI (public) · **organ** lung · **n patients** None · **n test** None · **n slices/images** None
- **headline** AUC = 0.914 on the internal_held_out set (abstract_sentence), scope single_modality_arm, interval none
- **evaluation unit** `lesion` — "False negative ( FN ) represents false negative (the number of malignant nodules that are predicted as benign nodules)." and "Table 3 Experimental results of the APPN on lung_all. Training set Testing set Sensitivity (%) 99.94 87.59 ... Accuracy (%) 99.72 88.47" (Methods; Results, Table 3). The scored unit is the nodule.
- **headline unit** `na_only_one_unit_reported` — Only the nodule unit is scored; no patient-level metric is reported.
- **split unit** `lesion_or_roi` (disjointness not_stated) — "The LIDC-IDRI dataset is a large publicly available dataset of lung CT, it has a total of 1018 cases and each case includes several CT images containing annotated lesions from lung cancer patients." followed by "Table 2 The description of the datasets. Training Testing lung_all..." giving per-nodule train/test counts (Results 3.1, Dataset preprocessing and splitting). D8(a) supplies lesion_or_roi for a split performed over sub-scan annotations; D7 makes the table counts a named unit.
- **positional distribution** `no` — No histogram, mean/sd of relative position or position-stratified metric along the slice axis appears; the pyramid-pooling "spatial" language refers to within-image scales.
- **input** `2D_slice` · label broadcast `na` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - the PLOS supplementary package was retrieved from Europe PMC; its only content is pone.0302641.s001.zip, a minimal dataset of image and mask files with no text document, so there is no supplementary prose that could carry a baseline. Recorded rather than assumed. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - the PLOS supplementary package was retrieved from Europe PMC; its only content is pone.0302641.s001.zip, a minimal dataset of image and mask files with no text document, so there is no supplementary prose that could carry a baseline. Recorded rather than assumed. no match indicating any zero-image baseline.
- **notes** The two "baseline" hits compare APPN against an Autoencoder - another imaging network, which the does_not_count list rules out. The "trivial" hit is "achieving high accuracy on this dataset is not a trivial task", not a trivial baseline. Nodules from the same patient can fall on both sides of the split and no patient count is stated.

### pos 302 — PMID 41481488 — *Radiol Imaging Cancer* (2026, CT)

Assessing an Automated Noncontrast CT-based Pipeline for Sacral Tumor Classification Using a Hip Bone Reference Frame.

- **dataset** private_multi_centre (private) · **organ** spine · **n patients** 690 · **n test** None · **n slices/images** None
- **headline** AUC = 0.87 on the external set (abstract_multiple_took_external), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `patient` — "In all, 690 patients (mean age, 46 years +/- 17 [SD]; 377 male patients) were included... The CL-MedImageNet classifier attained macro average AUCs of 0.89 (95% CI: 0.83, 0.93), 0.88 (95% CI: 0.84, 0.92), and 0.87 (95% CI: 0.79, 0.92) in validation, internal, and external test sets" (Results).
- **headline unit** `na_only_one_unit_reported` — Only the patient unit is scored for classification; no slice-level classification metric is reported (Dice belongs to the segmentation arm and A3 requires it to be ignored).
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "Each tumor category from center 1 was placed into training, validation, and test sets in a 7:1:2 ratio, consistent with the dataset construction method used for the segmentation model." (Methods). The placed noun is "each tumor category"; no patient-naming noun sits inside the splitting sentence, so D6 refuses the upgrade even though Table 1 counts patients.
- **positional distribution** `no` — The paper's location variable is "the distance from the tumor centroid to the hip bone" in a hip-bone reference frame, i.e. an anatomical 3D position of the lesion, not the ordered index of the classification unit within one acquisition. D9's test is the ordering of the classification unit, and the classification unit here is the patient, which has no ordering, so this is "no".
- **input** `3D_volume` · label broadcast `false` · code `public_link_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** Searches run and recorded; ALL SIX SUB-FLAGS FALSE, but this record is the closest thing to a positional baseline anywhere in block R4 and the evidence is quoted in full so an adjudicator can overturn it. "Additionally, we developed comparison models based on clinical, location, and imaging data. Using clinical information alone and a combination of clinical and location data, we built clinical models using a support vector machine (SVM)." (Methods). "Table 3: Model Performance Comparison for Sacral Tumor Classification | Test Set C-SVM CL-SVM Densnet121 ... | Validation set Macro average AUC 0.774 (0.701, 0.827) 0.867 (0.810, 0.909) 0.851 (0.775, 0.902)". CODED FALSE because the codebook's does_not_count list rules out "a radiomics or hand-crafted-feature model - it uses pixels", and BOTH comparison arms contain quantities the paper's own segmentation network computes from the pixels: "Clinical data (age, sex, and tumor volume) and location information were standardized", and "The process begins with model 1 performing automatic tumor and hip bone segmentation. From this segmentation, the tumor volume is calculated, and its relative position to the hip bone is determined." Neither arm is pixel-free, so neither is a zero-image baseline.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - all four supplementary files retrieved from Europe PMC and searched (57,630 characters). no match indicating any zero-image baseline.
- **notes** READ THIS ONE. The nearly-pixel-free CL-SVM arm (age, sex, tumour volume, and tumour position relative to the hip bone - a two-feature geometric summary plus two demographics) reaches macro AUC 0.867 on the validation set and BEATS the image-only DenseNet121 at 0.851, and reaches 0.85 externally against the full model's 0.87. The authors read this as evidence that "location information improves the identification of different sacral tumor types" rather than as evidence that the benchmark is largely solvable from tumour size and position. Under the codebook as written this is NOT a zero-image baseline, because volume and centroid distance are computed from the segmentation, i.e. from pixels; it is nevertheless the single most useful record in R4 for the paper's argument and should be quoted in the manuscript as a near-miss rather than as a P1 or S1 count.

### pos 304 — PMID 42406258 — *Brain Inform* (2026, MRI)

Generalizable and explainable deep learning for brain MRI: a multi-cohort evaluation of 3D architectures for age and sex prediction.

- **dataset** UK Biobank; Dallas Lifespan Brain Study (DLBS); PPMI; IXI (public) · **organ** brain · **n patients** 47949 · **n test** 559 · **n slices/images** None
- **headline** AUC = 1.0 on the internal_held_out set (abstract_sentence), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `patient` — "Table 2 Performance measures Cohort Model AUC (95% CI) AUPRC (95% CI) MAE (95% CI) Pearson r (95% CI)" over the four cohorts UKB (n = 47,390), DLBS (n = 132), PPMI (n = 108 controls) and IXI (n = 319), one T1-weighted volume per participant.
- **headline unit** `na_only_one_unit_reported` — Only the participant unit is scored; no slice-level metric is reported.
- **split unit** `random_unit_not_stated` (disjointness not_stated) — "split (2:1) with a fixed random seed (42)" (Methods). No unit is named in the splitting sentence or in any table caption describing the split.
- **positional distribution** `no` — Explainability maps show "task-specific and spatially consistent attention patterns" - anatomical attention within the volume, not a tabulation of label frequency against the ordered index of the classification unit; D9 codes this "no".
- **input** `3D_volume` · label broadcast `false` · code `public_link_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - 40708_2026_316_MOESM1_ESM.docx retrieved from Springer static content after the Europe PMC supplementary endpoint returned HTTP 404, and searched (8,410 characters). no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - 40708_2026_316_MOESM1_ESM.docx retrieved from Springer static content after the Europe PMC supplementary endpoint returned HTTP 404, and searched (8,410 characters). no match indicating any zero-image baseline.
- **notes** chance_asserted_without_measurement is TRUE: "The diagonal dotted lines in the ROC curves represent random chance performance" (Fig 2 caption). The codebook rules a stated chance level with no measurement FALSE for every sub-flag and records it here instead. This is the only such assertion among the 23 included records in R4. The age-prediction arm is regression and is ignored; the sex-classification arm satisfies I2 and I4 and is what is coded. An AUC of exactly 1.00 [1.00-1.00] for internal sex classification with an unnamed split unit is the configuration a zero-image baseline would most usefully interrogate, and none is reported.

### pos 305 — PMID 39557735 — *J Imaging Inform Med* (2025, CBCT)

A Comparison of Deep Learning vs. Dental Implantologists in Cone-Beam Computed Tomography-Based Bone Quality Classification.

- **dataset** private_single_centre (private) · **organ** head_neck · **n patients** 163 · **n test** None · **n slices/images** 1100
- **headline** accuracy = 0.86 on the internal_held_out set (abstract_sentence), scope single_modality_arm, interval none
- **evaluation unit** `slice` — "This investigation analyzed 1100 cross-sectional slices of CBCT data obtained from 163 patients" and "Five pre-trained DL models were trained on 1000 images using MATLAB, with 100 images reserved for testing." (Abstract; Methods, Data Collection). The scored unit is the CBCT cross-sectional slice.
- **headline unit** `na_only_one_unit_reported` — Only the slice unit is scored; no patient-level metric is reported.
- **split unit** `slice_or_image` (disjointness not_stated) — "To maintain a balanced distribution of samples for training and validation, each category was adjusted to contain precisely 250 images. Additionally, a random selection of 100 images from the year 2023 were independently re-categorized by the same radiologists to serve as a testing dataset." (Methods). "images" is named inside the splitting sentence, so this is slice_or_image.
- **positional distribution** `no` — Maxillary versus mandibular edentulous site is an anatomical region, not the ordered index of the classification unit within one acquisition; D9 codes this "no".
- **input** `2D_slice` · label broadcast `unclear` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: none exists. no match indicating any zero-image baseline.
- **notes** 1,100 slices come from only 163 patients, so roughly seven slices per patient are split at the image level with nothing said about patient disjointness - a textbook configuration for slice-level leakage, and the paper reports no zero-image baseline that would detect it. Recovered at rung 1 via the Europe PMC free render-PDF endpoint; the PMC deposit has no machine-readable body and link.springer.com refused this environment.

### pos 309 — PMID 37869523 — *EClinicalMedicine* (2023, PET_CT)

Deep radiomics-based fusion model for prediction of bevacizumab treatment response and outcome in patients with colorectal cancer liver metastases: a multicentre cohort study.

- **dataset** BECOME trial (NCT01972490); private_multi_centre (private) · **organ** liver · **n patients** 307 · **n test** 102 · **n slices/images** None
- **headline** AUC = 0.83 on the external set (abstract_multiple_took_external), scope single_modality_arm, interval ci_unspecified_method
- **evaluation unit** `patient` — "For image features, we directly used the feature extraction network to predict, and the classification results of a patient were determined by majority voting of the prediction from all PET/CT image pairs." (Methods, Establishment of single-scale models).
- **headline unit** `na_only_one_unit_reported` — Only the patient unit is scored; the per-image-pair predictions are aggregated by majority vote and are never reported as a metric of their own.
- **split unit** `external_cohort_only` (disjointness stated_only) — "The training cohort (n = 103) for the DERBY model consisted of patients of arm A (mFOLFOX6 plus bevacizumab) from the BECOME study. The internal validation cohort (n = 65) were collected from consecutive patients with CRLM from ZSH cohort" and "external validation cohort (n = 102) was derived from ZSHX and HWMU cohort" (Results, Study cohort). Cohorts are separated by study and by centre rather than by a within-cohort random split.
- **positional distribution** `no` — "the distribution of liver metastases" is an anatomical/lobar distribution among liver segments, not the ordered index of the classification unit within one acquisition; D9 codes an anatomical axis that is not the unit ordering as "no".
- **input** `2D_slice` · label broadcast `true` · code `not_stated` · access `oa_pmc_or_publisher` (version_of_record)
- **trivial_baseline** `{"constant_or_prevalence": false, "positional": false, "acquisition_metadata": false, "permuted_or_shuffled_label": false, "clinical_or_demographic_only": false, "other_non_imaging": false}`
- **trivial_baseline evidence** All six sub-flags FALSE. Per the codebook, the negative is evidenced by the recorded search: Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - mmc1.docx retrieved from Europe PMC and searched (62,592 characters). no match indicating any zero-image baseline. NO MATCH: no measured value for any constant/prevalence, positional, acquisition-metadata, permuted-label, clinical-only or other pixel-free comparator appears anywhere in the full text or its supplement.
- **searches_run** Ran all 14 required terms over the full text: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut. supplement: yes - mmc1.docx retrieved from Europe PMC and searched (62,592 characters). no match indicating any zero-image baseline.
- **notes** NEAR MISS, CODED FALSE, and the reasoning is the same test applied to PMID 41481488, so it is not being applied selectively. The paper reports a single-scale clinical arm with a measured value: "the DERBY + model (AUC: 0.83) presented higher accuracy than individual predictor types (AUC: Clinical-signature:0.66, Imaging-signature:0.72, Histological-signature:0.72, DERBY: 0.77), in the external validation cohort" (Results). But the eight candidate clinical factors are "SUVmax of PET/CT imaging, clinical risk score, pre-operative CEA, pre-operative CA19-9, the site of primary tumour, the numbers of liver metastases, maximum diameter of liver metastases, and the distribution of liver metastases" - four of the eight are quantities measured FROM the PET/CT pixels, so the Clinical-signature arm is not pixel-free and does_not_count rules out a hand-crafted-feature model that uses pixels. clinical_or_demographic_only is therefore FALSE. DIRECTION NOTE: this ruling LOWERS S1, i.e. it makes the literature look worse, and it is adopted for the same reason the two TRUE codes in this block (PMIDs 36646808 and 34765542) are adopted despite raising S1.

---

## 9. Excluded records

**pos 266 — PMID 39462483 — `E-NONMED`** — Deep Learning With Optical Coherence Tomography for Melanoma Identification and Risk Prediction.

> "This study develops and evaluates a convolutional neural network (CNN) for melanoma identification and risk prediction using optical coherence tomography (OCT) imaging of mice skin. Longitudinal tests are performed on four animal models: melanoma mice, dysplastic nevus mice, and their respective controls." (Abstract).

Preclinical animal-only imaging. E-NONMED is the first applicable code in listed order: the task IS classification (so not E-SEG), the acquisition IS volumetric OCT (so not E-2D or E-PROJ), the input IS the image (so not E-DERIV), a classifier IS fitted (so not E-NOCLF), and sensitivity/specificity ARE reported (so not E-NOMET). Full text was in PMC and the rung that worked is recorded, as D4 directs.

**pos 273 — PMID 41903680 — `E-NOCLF`** — Automated Detection of Middle Mesial Canals in Mandibular Molars on CBCT using nnU-Net: A Retrospective Diagnostic Accuracy Study.

> "A 3D nnU-Net-based model was trained using canal-focused patch sampling and a combined Dice plus weighted cross-entropy loss... For case-level MMC detection using a prespecified threshold (>=5 AI-positive slices), performance was sensitivity 94.7%, specificity 100%, accuracy 96.4%, and kappa = 0.920." (Abstract, Methods and Results).

D10 applied via the operational test recorded in screen_recoded.json: a categorical class decision IS scored with I4-list metrics but is NOT produced by a fitted classifier - the decider is an explicit threshold (">=5 AI-positive slices") on a quantity a segmentation network produced. E-SEG's qualifier "with NO categorical class decision evaluated" therefore fails and E-NOCLF is the correct code. Unambiguous from the abstract alone, so D11 permits stage1 exclude and D4 forbids climbing the ladder.

**pos 275 — PMID 34859922 — `E-SEG`** — 3D asymmetric expectation-maximization attention network for brain tumor segmentation.

> "Automatic brain tumor segmentation on MRI is a prerequisite... we present in this work a novel encoder-decoder neural network, ie a 3D asymmetric expectation-maximization attention network (AEMA-Net), to automatically segment brain tumors. We extensively evaluate AEMA-Net on three MRI brain tumor segmentation benchmarks of BraTS 2018, 2019 and 2020 datasets." (Abstract).

Segmentation only; no categorical class decision is evaluated anywhere, so E-SEG's qualifier holds and E-SEG is the first applicable code. Unambiguous from the abstract, so D4 forbids climbing the ladder.

**pos 276 — PMID 32391274 — `E-DERIV`** — Differentiating Peripherally-Located Small Cell Lung Cancer From Non-small Cell Lung Cancer Using a CT Radiomic Approach.

> "Radiomic features were extracted from histogram-based statistics, textural analysis of tumor images and their wavelet transforms. A minimal-redundancy-maximal-relevance method was used for feature selection. The predictive model was constructed with a multilayer artificial neural network." (Abstract; identical in Methods).

The classifier input is a selected radiomics feature vector with the image discarded - pilot amendment A5's exact construction. E-DERIV precedes E-NOCLF and E-NOMET in listed order. Full text was in PMC and the rung that worked is recorded, as D4 directs.

**pos 277 — PMID 36627354 — `E-NOCLF`** — Optical force estimation for interactions between tool and soft tissues.

> "We present a multi-input deep learning network for processing of local elasticity estimates and volumetric image data. Our results demonstrate that accounting for elastic properties is critical for accurate image-based force estimation... Joint processing of local elasticity information yields the best performance throughout our phantom study." (Abstract).

Force estimation is a regression task; no categorical label is assigned and no classification metric is reported, so criterion I2 fails. E-NOCLF precedes E-NOMET and E-NONMED in listed order, so the phantom/ex-vivo nature of the samples does not decide the code.

**pos 285 — PMID 38467345 — `E-DERIV`** — Tinnitus classification based on resting-state functional connectivity using a convolutional neural network architecture.

> "We established a convolutional neural network (CNN) model based on rs-fMRI FC to distinguish tinnitus patients from healthy controls" and "A CNN architecture was trained on rs-fMRI data from 100 tinnitus patients and 100 healthy controls using an asymmetric convolutional layer... each of the three models was tested on three different brain atlases." (Abstract).

The classifier input is an atlas-parcellated functional-connectivity matrix, not a spatially resolved image - pilot amendment A5's founding case (PMID 34924987, ABIDE connectivity matrices). The dependence of the result on which of three atlases is used confirms that the image never reaches the model. Unambiguous from the abstract, so D4 forbids climbing the ladder; the record is nonetheless demonstrably OA at Elsevier, which refused this environment.

**pos 287 — PMID 38164538 — `E-SEG`** — Application of a Denoising High-Resolution Deep Convolutional Neural Network to Improve Conspicuity of CSF-Venous Fistulas on Photon-Counting CT Myelography.

> "Here, we describe a novel deep-learning-based algorithm used to denoise photon-counting detector CT myelographic images, allowing the sharpest and thinnest quantitative reconstruction available on the scanner to be used to enhance diagnostic image quality... This algorithm has the potential to increase the sensitivity of photon-counting detector CT myelography for detecting CSF-venous fistulas" (Abstract; Discussion).

Denoising is on E-SEG's own list and E-SEG's qualifier holds: no categorical class decision of any kind is evaluated, and the only sensitivity mentioned is a hoped-for future gain with no number. Residual gap G5 of the v1.2 re-coding is the nearest precedent and it kept E-SEG in exactly this configuration. Recovered at rung 1 via the Europe PMC free render-PDF endpoint; ajnr.org returned HTTP 403 and is blocked by browsing policy.

**pos 288 — PMID 39992333 — `E-SEG`** — Intelligent Recognition and Segmentation of Blunt Craniocerebral Injury CT Images Based on DeepLabV3+ Model.

> Chinese: "根据盲测集的准确率、精确率和F 1值评价模型对 5类颜脑损伤的分割性能。" / English abstract: "According to the accuracy, precision and F1 value of the blind test set, the segmentation performance of the model for five types of BCI was evaluated." (Abstract, Methods).

The paper states in both languages that accuracy, precision and F1 evaluate the SEGMENTATION performance of DeepLabV3+; no categorical class decision on an imaging unit is evaluated, so E-SEG's qualifier holds and E-SEG precedes E-LANG in listed order. Unambiguous from the abstract, so D4 forbids climbing the ladder. The publisher landing page at fyxzz.cn was reachable and confirms the abstract; its PDF endpoint returned HTTP 403.

**pos 290 — PMID 37273912 — `E-SEG`** — Automated semantic lung segmentation in chest CT images using deep neural network.

> "This work aims to develop a computationally efficient and robust deep learning model for lung segmentation using chest computed tomography (CT) images with DeepLabV3 + networks for two-class (background and lung field) and four-class (ground-glass opacities, background, consolidation, and lung field)... The segmentation performance has been assessed using five performance measures: Intersection of Union (IoU), Weighted IoU, Balance F1 score, pixel accuracy, and global accuracy." (Abstract).

Semantic segmentation only; every reported metric is pixel-wise. No categorical class decision on an imaging unit is evaluated, so E-SEG's qualifier holds. Full text was in PMC and the rung that worked is recorded, as D4 directs.

**pos 292 — PMID 35914993 — `E-2D`** — Comparative Study of Raw Ultrasound Data Representations in Deep Learning to Classify Hepatic Steatosis.

> "US radiofrequency (RF) frames (raw data) and clinical B-mode images were acquired. Intermediate image formation stages were modeled from RF data... Co-registered patches were used to independently train 1-, 2- and 3-D convolutional neural networks (CNNs)" (Abstract).

The acquisition is single-frame B-mode/RF ultrasound, which E-2D names explicitly. The "3-D" CNNs operate on patches of 2D RF frames, not on a volumetric acquisition, so criterion I3 fails. MRI appears only as the fat-fraction reference standard and never enters the model. E-2D precedes E-DERIV in listed order.

**pos 294 — PMID 36514476 — `E-DERIV`** — Alzheimer's disease classification using cluster-based labelling for graph neural network on heterogeneous data.

> "The final dataset comprised 224 features: 7 sociodemographic and medical history features, 40 cognitive and functional assessments' (CFAs) scores, and 177 neuroimaging features (from combined MRI and tau PET imaging data)" and "the dataset of 224 features and 559 samples was subjected to dimension reduction using the uniform manifold approximation and project (UMAP)" (Methods).

The GNN classifier input is a 224-dimensional feature vector in which the imaging contribution is 177 regional summary features; the image itself is discarded. Pilot amendment A5's construction, E-DERIV. Full text was in PMC and the rung that worked is recorded, as D4 directs.

**pos 298 — PMID 41595557 — `E-TYPE`** — Comparison of Artificial Intelligence and Radiologists in MRI-Based Prostate Cancer Diagnosis: A Meta-Analysis of Accuracy and Effectiveness.

> "This meta-analysis aims to evaluate whether AI can achieve diagnostic performance that is comparable to that of radiologists... Following PRISMA 2020 guidelines, we searched PubMed for studies directly comparing AI and radiologists in MRI-based detection of csPCa. Ten studies (20,423 patients) were included, and quality was assessed using QUADAS-2." (Abstract).

A meta-analysis that re-tabulates other papers' numbers and fits no model of its own; E-TYPE names both "meta-analysis" and "a survey/benchmark paper that only re-tabulates other people's numbers". PubMed carries no meta-analysis publication type on this record, which is why the frame's negative publication-type filter did not remove it - exactly the case exclusion code E-TYPE exists to catch (protocol section 2.1).

**pos 306 — PMID 40879858 — `E-DERIV`** — Multi-regional Multiparametric Deep Learning Radiomics for Diagnosis of Clinically Significant Prostate Cancer.

> "Radiomics features are then extracted and selected from multiparametric MRI at the PZ, TZ, and their combined area to develop a multi-regional multiparametric radiomics diagnostic model" and "To address redundancy and overfitting caused by high-dimensional radiomics features, we propose a hybrid algorithm which combines the least absolute shrinkage and selection operator (LASSO) and Shapley additive explanation (SHAP)... where ym represents the label of the m-th sample, Xm is the feature vector of the m-th sample" (Abstract; Methods, Feature Selection and Classification).

The deep-learning component (CCT-Unet) performs zonal SEGMENTATION only; the diagnostic classifier's input is a selected radiomics feature vector of morphological, intensity and texture features, with the image discarded. E-DERIV precedes E-NOCLF in listed order, and E-SEG does not apply because a categorical class decision IS evaluated by a fitted classifier. Recovered at rung 1 via the Europe PMC free render-PDF endpoint. stage1_decision is "exclude" on abstract evidence alone - the abstract itself says the radiomics features, not the images, are what the diagnostic model is built from - and the full text, recovered anyway, confirms it; the rung that worked is therefore recorded rather than "not_attempted", as D4 directs.

**pos 308 — PMID 38761987 — `E-2D`** — Mental Stress-Induced Myocardial Ischemia Detected by Global Longitudinal Strain and Quantitative Myocardial Contrast Echocardiography in Women With Nonobstructive Coronary Artery Disease.

> "This study aims to assess the diagnostic performance of novel echocardiographic techniques, including automated strain and quantitative myocardial contrast echocardiography (MCE) with dedicated software and deep neural network model, for MSIMI detection... Mental stress-induced myocardial ischemia was defined as a summed difference score >=3 on PET." (Abstract).

The deep neural network's input is 2D contrast echocardiography, which E-2D names explicitly ("2D echocardiography"). PET is the reference standard defining the label, not a model input, so D12 keeps modality on the input acquisition and the record does not become "multiple". E-2D precedes E-NOCLF in listed order, so the fact that the reported ROC analyses threshold continuous echo parameters (deltaGLS, beta reserve) does not decide the code.

**pos 310 — PMID 42224315 — `E-2D`** — Liver Nodule Anomaly Detection Using Ultra-sound Radiofrequency Signals and Variational Autoencoders.

> "This prospective, cross-sectional study aims to evaluate the performance of a reconstruction based deep learning model for detection of liver nodules using one dimensional US radiofrequency signals (1D RF) of two-dimensional images... Variational autoencoders were trained on adjacent 1D RF lines of images" (Abstract).

The acquisition is two-dimensional ultrasound and the model input is 1D RF lines of those 2D images, so criterion I3 fails on both counts; E-2D names single-frame ultrasound explicitly and precedes E-DERIV and E-NOCLF in listed order. MRI and histopathology appear only as reference standards. (Had the record survived E-2D it would also have failed I2: the variational autoencoder is unsupervised and the class decision is a threshold on an anomaly score, which D10 makes E-NOCLF.)

---

## 10. Unreachable records (eligibility unresolved)

Every one of these reached stage 2, every rung of the §7 ladder available to this screener was worked, and D1 forbids
coding any of them `included` however clear the abstract is, because the mandatory 14-term full-text search cannot be run
without the full text. All six sub-flags are `not_assessable` on all twelve, never `false`.

**pos 263 — PMID 34642754 — *Cereb Cortex* 2021** — Integrating Multilevel Functional Characteristics Reveals Aberrant Neural Patterns during Audiovisual Emotional Processing in Depression.

- *why it matters:* Audiovisual-emotion fMRI in depression; SVM on task-evoked activation, task-modulated connectivity and their combination. Whether the classifier input is a spatially resolved image or an ROI/connectivity feature vector (E-DERIV, pilot amendment A5) cannot be settled from the abstract, so stage1 is go_to_fulltext and D1 forces unreachable_eligibility_unresolved.
- *access ladder:* rung 1 Europe PMC/PMC: not in PMC, isOpenAccess=N; rung 2 publisher (academic.oup.com via doi.org): HTTP 403 to this environment; rung 3 institutional subscription: not held by this screener; rung 4 repository/preprint: Unpaywall is_oa=false, OpenAlex lists no OA location, arXiv title search returns no hit; rung 5 ILL/author request: cannot complete within this session.

**pos 267 — PMID 42136201 — *Invest Radiol* 2026** — Artificial Intelligence-Enhanced Identification of Incidental Findings in Prostate MRI.

- *why it matters:* nnU-Net segmentation of incidental findings in prostate MRI with model-level confusion matrices, sensitivity, specificity and accuracy on a quantitative test set, plus a radiologist reader test set. Whether the categorical presence decision is produced by a fitted classifier or by a threshold on the segmentation output (D10 -> E-NOCLF) cannot be settled from the abstract, so stage1 is go_to_fulltext and D1 forces unreachable_eligibility_unresolved.
- *access ladder:* rung 1 Europe PMC/PMC: no PMCID, not open access; rung 2 publisher: doi.org resolved to ovid.com and returned the abstract page only, behind "Check Access"; rung 3 institutional subscription: not held; rung 4 repository/preprint: Unpaywall is_oa=false, OpenAlex lists no OA location, arXiv title search returns no hit; rung 5: cannot complete within this session.

**pos 270 — PMID 36125375 — *Radiology* 2023** — Machine Learning for Adrenal Gland Segmentation and Classification of Normal and Adrenal Masses at CT.

- *why it matters:* Adrenal gland segmentation plus normal/mass classification with sensitivity and specificity reported for the classification arm; A3 would make this an include if the full text confirmed a fitted classifier, but eligibility cannot be confirmed without it, so D1 governs.
- *access ladder:* rung 1 Europe PMC/PMC: no PMCID, isOpenAccess=N; rung 2 publisher (pubs.rsna.org via doi.org): HTTP 403 to this environment and blocked by browsing policy; rung 3: not held; rung 4: Unpaywall is_oa=false, OpenAlex lists no OA location, arXiv title search returns no hit; rung 5: cannot complete within this session.

**pos 271 — PMID 39128599 — *J Neurosci Methods* 2024** — Neuro-XAI: Explainable deep learning framework based on deeplabV3+ and bayesian optimization for segmentation and classification of brain tumor in MRI scans.

- *why it matters:* Neuro-XAI: DeepLabV3+ segmentation plus Darknet53/MobileNetV2 features into an SVM, 97% classification accuracy. A3 suggests include on the classification arm, but whether the SVM input is an image-derived deep feature or a non-spatial descriptor (E-DERIV) needs the Methods. D1 governs.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (Elsevier linkinghub via doi.org): returned a 2.8 kB redirect stub, no full text; rung 3: not held; rung 4: Unpaywall is_oa=false, no OA location in OpenAlex, no arXiv hit; rung 5: cannot complete within this session.

**pos 272 — PMID 32315264 — *Radiology* 2020** — Predicting Rectal Cancer Response to Neoadjuvant Chemoradiotherapy Using Deep Learning of Diffusion Kurtosis MRI.

- *why it matters:* Diffusion-kurtosis MRI deep learning for rectal cancer pCR, AUC 0.99 in a 93-patient test cohort. Abstract is consistent with inclusion; D1 forbids coding it included without the mandatory 14-term full-text search.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (pubs.rsna.org via doi.org): HTTP 403 and blocked by browsing policy; rung 3: not held; rung 4: no OA location in Unpaywall or OpenAlex, no arXiv hit; rung 5: cannot complete within this session.

**pos 274 — PMID 36240594 — *Comput Biol Med* 2022** — Patient-level grading prediction of prostate cancer from mp-MRI via GMINet.

- *why it matters:* GMINet, patient-level Gleason grade-group prediction from multi-slice mp-MRI with slice-to-slice correlations, five-class accuracy 81.1% and csPCa AUC 0.801. Abstract is consistent with inclusion; D1 governs.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (Elsevier linkinghub via doi.org): 2.7 kB redirect stub only; rung 3: not held; rung 4: no OA location in Unpaywall or OpenAlex, no arXiv hit; rung 5: cannot complete within this session.

**pos 281 — PMID 32906091 — *J Neural Eng* 2020** — 'When' and 'what' did you see? A novel fMRI-based visual decoding framework.

- *why it matters:* fMRI visual decoding with an RNN over fMRI activity patterns in eVC and hVC. Whether the classifier input is a spatially resolved volume or an ROI voxel-pattern vector (E-DERIV) cannot be settled from the abstract. NOTE FOR ADJUDICATION: the abstract states "The average decoding accuracy across five subjects was over 19 times the chance level" - a chance level expressed as a ratio with no absolute measured value for a chance arm, which the codebook rules FALSE for every sub-flag; but the record is unreachable, so D2(c) gives not_assessable, not false.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (iopscience.iop.org via doi.org): redirected to a perfdrive.com bot-detection challenge, which was NOT circumvented; rung 3: not held; rung 4: no OA location in Unpaywall or OpenAlex, no arXiv hit; rung 5: cannot complete within this session.

**pos 284 — PMID 33677163 — *Epilepsy Res* 2021** — Automated detection and segmentation of focal cortical dysplasias (FCDs) with artificial intelligence: Presentation of a novel convolutional neural network and its prospective clinical validation.

- *why it matters:* FCD detection and segmentation with a 3D CNN, prospectively validated on 100 routine MRIs and reporting sensitivity 77.8% and specificity 5.5%. Whether a categorical per-MRI class decision is evaluated (include, A3) or the metrics belong to lesion segmentation (E-SEG, rule A4) cannot be settled from the abstract, so D1 governs.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (Elsevier linkinghub via doi.org): 2.8 kB redirect stub only; rung 3: not held; rung 4: no OA location in Unpaywall or OpenAlex, no arXiv hit; rung 5: cannot complete within this session.

**pos 289 — PMID 38810473 — *Comput Biol Med* 2024** — Comprehensive quantitative radiogenomic evaluation reveals novel radiomic subtypes with distinct immune pattern in glioma.

- *why it matters:* Radiogenomic study: Mask R-CNN identifies tumour regions (accuracy 88.3%/83%), subtypes come from unsupervised clustering of radiomic features, and "Three machine learning-based classifiers showed that radiomic and genomic co-features better predicted the radiomic subtypes". Whether the first applicable code is E-SEG (detection metrics only), E-DERIV (classifier on radiomic feature vectors) or include cannot be settled from the abstract, so D1 governs.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (Elsevier linkinghub via doi.org): 2.8 kB redirect stub only; rung 3: not held; rung 4: no OA location in Unpaywall or OpenAlex, no arXiv hit; rung 5: cannot complete within this session.

**pos 291 — PMID 34260415 — *Biomed Phys Eng Express* 2021** — Efficient brain tumor detection and classification using magnetic resonance imaging.

- *why it matters:* Brain tumour detection and classification from MRI with an "Absolute Classification-Detection Model"; the abstract names accuracy, precision, sensitivity and classification time but reports no value, so even I4 is unconfirmed. D1 governs.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (iopscience.iop.org via doi.org): redirected to a perfdrive.com bot-detection challenge, which was NOT circumvented; rung 3: not held; rung 4: no OA location in Unpaywall or OpenAlex, no arXiv hit; rung 5: cannot complete within this session.

**pos 303 — PMID 41271174 — *Radiother Oncol* 2026** — A foundation model for brain tumor MRI analysis: WHO grading and subtype classification.

- *why it matters:* UMBIF self-supervised foundation model for glioma grading and histological subtype classification on MRI, with accuracies and AUCs reported per grade. Abstract is consistent with inclusion; D1 forbids coding it included without the mandatory 14-term full-text search.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (Elsevier linkinghub via doi.org): 2.7 kB redirect stub only; rung 3: not held; rung 4: no OA location in Unpaywall or OpenAlex, no arXiv hit; rung 5: cannot complete within this session.

**pos 307 — PMID 36403310 — *Med Image Anal* 2023** — Deep semi-supervised multiple instance learning with self-correction for DME classification from OCT images.

- *why it matters:* Deep semi-supervised multiple-instance learning for DME classification from OCT VOLUMETRIC images, with B-scan instances inside volume-level bags - squarely inside the eligible population and inside the failure mode this project studies. Abstract is consistent with inclusion; D1 forbids coding it included without the mandatory 14-term full-text search. THE MOST REGRETTABLE LOSS IN BLOCK R4.
- *access ladder:* rung 1: no PMCID, isOpenAccess=N; rung 2 publisher (Elsevier linkinghub via doi.org): 2.7 kB redirect stub only; rung 3: not held; rung 4: OpenAlex lists a HKUST institutional-repository record (repository.hkust.edu.hk/ir/Record/1783.1-124632) which returned HTTP 302 and no retrievable file, Unpaywall is_oa=false, no arXiv hit; rung 5: cannot complete within this session.

---

## 11. Cumulative arithmetic and what it does not change

- **included_running_total** — main sample after the v1.3 access recovery 38; + R1 (111-160) 17 = 55; + R2 (161-210) 17 = 72; + R3 (211-260) 19 = 91; + R4 (261-310) 23 = 114.
- **verdict** — THE PRE-SPECIFIED EXTENSION RULE HAD ALREADY STOPPED AT PERMUTATION POSITION 260. Ninety-one included papers at the end of R3 is at or above the target of 75, so protocol section 3.1 did not authorise a fifth block. Block R4 was screened on explicit instruction and is complete and honestly coded, but it is a POST-HOC EXTENSION and must be labelled one wherever it is reported. Do not silently pool it into the pre-registered denominator.
- **P1_cumulative_pre_registered_n91** — 0/91 = 0.0% [0.0%, 4.1%] Wilson - the pre-registered figure, inside the 4.9% target precision at n=75.
- **P1_cumulative_with_R4_post_hoc** — 0/114 = 0.0% [0.0%, 3.3%] Wilson - POST-HOC, report as such.
- **S6_cumulative_with_R4_post_hoc** — unreachable 16 + 9 + 5 + 8 + 12 = 50 over an eligible-looking denominator of 54 + 26 + 22 + 27 + 35 = 164, i.e. 30.5%.
- **section_7_still_binds** — 34.3% unreachable in this block alone, and about 30.5% cumulatively, both far above the 15% threshold in protocol section 7, so the bounding interval REMAINS the headline number and no increase in sample size can change that. What DID move the bound in this block was access work, not sample size: the Europe PMC render-PDF rung recovered five records and took S6 for R4 from 60.7% to 34.3%. That is the lever, and section 7 says so.

- **how this was computed** — The aggregate included counts of screen_reserve_R1/R2/R3.json and the post-recovery count for the main sample were re-derived from the committed files by this screener AFTER all fifty R4 records were frozen. No individual record of any earlier block was read. The parent should re-derive these totals rather than trust this line.

---

## 12. Independence

All fifty records were coded from the papers themselves under codebook v1.2. No other screener's output was read while coding: paper/screen_batch_{A,B,C,D}.json, paper/screen_recoded.json's per-record 'records' array, and screen_reserve_R1/R2/R3.json's per-record arrays were not opened for any record in this block. Read as CODEBOOK material before coding, which the protocol requires: screen_protocol.md, screen_frame.json v1.2, and from screen_recoded.json only the two header keys '_operational_test_used_for_D10' and 'residual_gaps_found_during_the_recode' (G1-G5). The aggregate included/unreachable COUNTS of R1, R2 and R3 were read once, AFTER all fifty R4 records were written and frozen, and used only for the extension-rule arithmetic above; no individual record of any earlier block was read at any point.

