# Reserve block R1 -- prevalence screen, permutation positions 111-160

**Protocol** `paper/screen_protocol.md` v1.3 | **Codebook** `paper/screen_frame.json` v1.2 (D1-D14) | **Sample** `paper/screen_sample.json` v1.1, reserve records 111-160 | **Screener** S1 | **Submitted** 2026-07-29T14:30:00Z

This block was screened under the extension rule pre-specified in `screen_protocol.md` section 3.1, before any outcome was seen: *"If the number of included papers is below 75, continue into the reserve in permutation order, in blocks of 50 records, until either 75 included papers are reached or position 400 is exhausted. Every record in a started block is screened and reported; blocks are never truncated part-way once begun."* All 50 records were screened and all 50 are reported.

---

## 1. Headline tallies

| | n | of |
|---|---|---|
| Records screened | **50** | 50 assigned |
| Included | **17** | 50 |
| Excluded | **24** | 50 |
| Unreachable (`unreachable_eligibility_unresolved`) | **9** | 50 |
| Eligible-looking set (included + unreachable) | 26 | 50 |
| **Reporting ANY zero-image baseline (P1 family)** | **0** | 17 included |
| Reporting any non-imaging baseline at all (S1 family) | **1** | 17 included |

**The primary answer for this block is zero.** Not one of the 17 included papers reports a measured value for a constant/prevalence predictor, a positional (slice-index) predictor, an acquisition-metadata predictor, or a permuted-label null. Across all 50 records and all 300 sub-flag cells, `constant_or_prevalence`, `positional`, `acquisition_metadata` and `permuted_or_shuffled_label` are true **nowhere**.

Block P1 = 0/17, Wilson 95% **0.0% [0.0%, 18.4%]**. Block S6 (unreachable) = 9/26, Wilson 95% **34.6% [19.4%, 53.8%]** -- again far above the 15% threshold at which section 7 makes the bounding interval the headline.

### Cumulative position after this block

Combining with the post-recovery numbers recorded in `screen_protocol.md` changelog v1.3 (38 included, 16 unreachable, 54 eligible-looking):

| | before R1 | R1 | cumulative |
|---|---|---|---|
| Included | 38 | +17 | **55** |
| Unreachable | 16 | +9 | **25** |
| Eligible-looking | 54 | +26 | **80** |
| P1 numerator | 0 | +0 | **0** |

- **P1 complete-case 0/55, Wilson 95% 0.0% [0.0%, 6.5%].** Still exactly zero.
- **S6 unreachable 25/80 = 31.2% [22.2%, 42.1%].** Above 15%, so section 7 still binds and the bounding interval remains the headline.
- **Headline bounding interval [0.0%, 31.2%]** (lower arm: every unreachable paper coded as not reporting a zero-image baseline; upper arm: every one coded as reporting one).

**The extension rule has not yet stopped.** 55 included is below the 75 target, and position 160 is well short of 400, so the rule requires block R2 (positions 161-210) next. On this block's yield of 17 included per 50 screened, reaching 75 needs roughly 60 more records, i.e. probably two more blocks.

**And the binding constraint has not moved.** Enlarging the sample did narrow the complete-case interval -- 0/38 gave [0.0%, 9.2%], 0/55 gives [0.0%, 6.5%] -- but the complete-case estimate is not the reportable figure. The reportable figure is the bounding interval, and it went the wrong way: **[0.0%, 29.6%] to [0.0%, 31.2%]**, because this block's unreachable fraction (34.6%) is higher than the post-recovery rate of the main sample (29.6%). This is the arithmetic the protocol warned about in section 3.1: adding records adds unreachable records in the same proportion, so no amount of extra sampling narrows the bound. Recovering full texts is the only thing that does.

---

## 2. The one positive, in full

**PMID 37679806** (position 126) -- Wang Y, Ding Y, Liu X, et al., *Cancer Imaging* 2023, "Preoperative CT-based radiomics combined with tumour spread through air spaces can accurately predict early recurrence of stage I lung adenocarcinoma: a multicentre retrospective cohort study."

Table 2 reports a **STAS-alone arm** with a measured AUC in all three cohorts, against the combined model:

| arm | training | test | validation |
|---|---|---|---|
| STAS alone (no CT pixels) | **0.727** (0.654-0.799) | **0.606** (0.433-0.780) | **0.713** (0.555-0.872) |
| RAISm (radiomics + STAS) | 0.847 (0.762-0.932) | 0.750 (0.531-0.969) | 0.817 (0.625-1.000) |

STAS -- tumour spread through air spaces -- is a binary variable read off postoperative H&E sections. It needs no CT voxel. Reported alone with an AUC on the same metric, it is a pixel-free comparator with a measured value, so `other_non_imaging` is coded **true**.

**This is an S1 positive, not a P1 positive.** `screen_protocol.md` section 1 defines P1 as constant/prevalence, positional, acquisition-metadata or permuted-label; `other_non_imaging` and `clinical_or_demographic_only` sit in secondary endpoint S1 by design, "so the two can be read apart". This record therefore does **not** move P1 off zero.

The sub-flag choice is flagged for adjudication. A screener could reasonably code `clinical_or_demographic_only` instead -- structurally the STAS row *is* "the clinical-only arm of a clinical+radiomics nomogram", which that sub-flag names explicitly. It was coded `other_non_imaging` because STAS is not a demographic or laboratory variable of the age/sex/PSA kind the sub-flag enumerates; it comes off a microscope slide. **Either coding gives the same endpoint result**, because both sub-flags live in S1 and neither enters P1.

Note how it was found: the fourteen mandated search terms did **not** surface it. `baseline` hits five times in this paper and every hit is "baseline characteristics" or "baseline CT". The STAS arm is a row in a results table. That is an argument for the protocol's reading effort -- Methods in full *plus Results tables and figures* -- and against treating the term list as sufficient on its own.

---

## 3. The near-miss that deserves a second screener

**PMID 36087795** (position 131) -- Weir-McCall JR, et al., *Chest* 2023, LCP-CNN on solitary pulmonary nodules. The paper reports:

> "Both PET with CT scan imaging and the LCP-CNN were significantly more accurate than the Mayo Clinic model (AUC, 0.73; 95% CI, 0.67-0.79; P < .002 for both)"

A measured comparator, on the same metric, from a model usually described as clinical. It is coded **false** for `clinical_or_demographic_only`, for a specific reason: the Mayo Clinic (Swensen) model is **not pixel-free**. Its predictors are age, smoking history and extrathoracic-cancer history *together with* nodule diameter, spiculation and upper-lobe location -- three CT findings read off the scan by a human. The codebook defines the sub-flag as "patient variables only ... with no imaging" and puts "a radiomics or hand-crafted-feature model -- it uses pixels" under `does_not_count`. A radiologist-scored nodule descriptor is a hand-crafted image feature. Table 1 confirms the paper measured them ("Size, mm 15.8 +/- 6.0").

If a second screener disagrees, this becomes a second S1 positive. It does **not** enter P1 on either reading.

A third case is recorded for completeness even though it is excluded and enters no denominator: **PMID 36077686** (position 155) reports measured 5-year AUCs for TNM stage (0.561 +/- 0.042) and tumour grade (0.573 +/- 0.044) against its own model (0.817 +/- 0.037). Excluded E-DERIV, so under D3 all six sub-flags are `not_applicable`. Recorded because staying silent about it would be dishonest.

And the sharpest observation in the block: **the only record that draws a prevalence baseline at all is the one that never uses a pixel.** PMID 41899050 (position 123), an EMR sepsis model excluded E-DERIV, captions a precision-recall figure "The dashed horizontal line represents the baseline precision corresponding to the prevalence of the positive class." Seventeen imaging papers drew no such line. One non-imaging paper did.

---

## 4. Flow

```
50 reserve records screened (permutation positions 111-160)
  |
  |-- 19 excluded at stage 1 from the abstract alone (D11)
  |      |-- 7  ladder therefore NOT climbed (D4) -> not_attempted_excluded_at_stage1
  |      \-- 12 full text in hand anyway via PMC/publisher, so the rung that worked
  |             is recorded (D4) and the stage-1 code was VERIFIED against the full text
  |
  \-- 31 reached stage 2 (go_to_fulltext)
         |-- 22 full text obtained
         |      |-- 5  excluded on full text
         |      \-- 17 INCLUDED
         \-- 9  unreachable_eligibility_unresolved (D1)
```

34 full texts were read in total (12 + 22). The 7 stage-1 exclusions whose ladder was not climbed are the only records in the block for which no full text was sought; D4 keeps them out of the S6 denominator so that S6 measures the reachability of the *eligible-looking* literature. S6's denominator is the 26 eligible-looking records (17 included + 9 unreachable).

### Exclusion codes (24 excluded records)

| code | n | what these papers actually were |
|---|---|---|
| `E-SEG` | 10 | segmentation, super-resolution, synthesis or reconstruction with no evaluated class decision |
| `E-DERIV` | 7 | classifier input is a feature vector, connectome, volumetry table or EMR record; no image reaches the model |
| `E-2D` | 2 | histopathology whole-slide (41617821) and laser-speckle camera video (40337176) |
| `E-TYPE` | 2 | a narrative review (35680755) and a survey (35578678), neither caught by the frame's negative publication-type filter |
| `E-NOCLF` | 2 | unsupervised VAE abnormality score (31003928); DL segmentation whose only class decision was made by human readers (34776805, D10) |
| `E-PROJ` | 1 | unfolded universal-atrial-coordinate maps of simulated fibrosis (40290188) |

The 34% inclusion rate (17/50) sits just below the 30-50% band the pilot predicted (section 4.2), and the gap is entirely the 9 unreachable records, which are eligible-looking but unconfirmable: counted as eligible, the rate is 26/50 = 52%. The criteria are travelling.

---

## 5. Access: what worked, what did not, and what is disclosed

| rung | outcome |
|---|---|
| 1. PMC / publisher OA | 31 full texts. Includes two Europe PMC rendered PDFs (34117783, 36087795) obtained after `pmc.ncbi.nlm.nih.gov` served this environment a **reCAPTCHA challenge**, which was **not** bypassed. |
| 2. Publisher site direct | 2 full texts (34341737, 35655831), both *Quantitative Imaging in Medicine and Surgery*, whose PMC deposits carry front matter only but whose own site serves complete HTML free. |
| 3. Institutional subscription | Not held by this screener. |
| 4. Repository / accepted manuscript / preprint | Worked against Unpaywall, arXiv and named repositories for all 19 records failing rungs 1-2. **One** usable recovery: 39515189 via arXiv:2311.08908v1. |
| 5. ILL / author request | Cannot complete inside one session; the 21-day clock cannot elapse. Records failing rungs 1-4 are coded `unreachable_paywalled`, the conservative code. |

**Sci-Hub and every other infringing source were not used at any point.** Where a paper was reachable only through one, it stayed unreachable. No CAPTCHA or bot-detection challenge was bypassed.

**Disclosed rung-4 near-misses** -- cases where a legitimate open copy demonstrably exists and this environment could not read it. All four are excluded at stage 1 or otherwise resolved, so none of them inflates or deflates S6:

- **35487442** -- Cardiff University ORCA holds a CC-BY-NC-ND accepted manuscript; `orca.cardiff.ac.uk` returned HTTP 403 from a bot-detection interposer.
- **39302179** -- University of Geneva archive-ouverte holds a CC-BY copy; the landing page is a client-side-rendered SPA that served no document to any scripted request.
- **37742486** -- Vrije Universiteit Brussel repository PDF; HTTP 503 on every attempt.
- **33741850** -- Fujita Health University repository has a genuine OA record, but it is a **one-page Japanese doctoral-thesis summary** (論文内容の要旨), not the article. A thesis abstract is neither the version of record nor an accepted manuscript, so it was not coded from and the record stays **unreachable**.

### Four records not coded from the version of record

All four belong in the version-of-record sensitivity analysis section 7 requires.

| PMID | version used | why |
|---|---|---|
| 39515189 | **preprint (substituted)** | Version of record closed at Elsevier; coded from arXiv:2311.08908v1. The published version may differ. |
| 40766163 | preprint (the record itself) | The sampled record *is* a medRxiv preprint; PubMed publication type "Preprint". |
| 42078359 | preprint (the record itself) | Same. |
| 36087795 | accepted manuscript | PMC author manuscript deposit. |

---

## 6. Secondary endpoints, block-level

Reported for this block alone, over its 17 included papers. Wilson 95%.

| endpoint | block R1 |
|---|---|
| **P1** any zero-image baseline | **0/17** 0.0% [0.0%, 18.4%] |
| S1 any non-imaging baseline | 1/17 5.9% [1.0%, 27.0%] |
| S2 headline unit is the slice | 3/17 17.6% [6.2%, 41.0%] |
| -- of which `headline_unit` = `na_only_one_unit_reported` | 16/17 (only one paper reports two units at all) |
| S3 among papers reporting a slice-level metric, also reporting patient-level | 1/3 |
| S4 explicit subject-level split | 4/17 23.5% [9.6%, 47.3%] |
| S5 positional distribution of labels reported | **0/17** 0.0% [0.0%, 18.4%] |
| S6 unreachable | 9/26 34.6% [19.4%, 53.8%] |
| S8 subject-clustered interval | **0/17** 0.0% [0.0%, 18.4%] |
| S9 reports n positive patients, not only slices | 7/17 41.2% [21.6%, 64.0%] |

Three of these deserve a sentence.

**S4 = 4/17.** Only four of seventeen included papers state a subject-level split: 40721771 ("All splits were performed at the patient level to avoid data leakage"), 40627160 ("split on a patient level ... ensured this ratio for the patients as well as for the lesions"), 40766163 ("at the participant level to no individual contributed to both sets"), 37679806 ("226 patients ... were randomly split"). The other thirteen split "samples", "cases", "data", "images", "lesions", by centre, or over an external cohort -- or, in four cases, say nothing a reader can act on.

**S8 = 0/17.** No included paper reports a subject-clustered uncertainty interval. Six report an interval of unspecified method, three report a spread across folds, and eight report no uncertainty at all -- including papers claiming 99.4% and 99.79477% accuracy.

**S5 = 0/17.** No included paper reports how its labels distribute along the slice axis. Every candidate hit in the block was an anatomical region (lobe, peripheral zone) or scan geometry (slice selection, patient repositioning), which D9 codes "no". D9 was written for a case where the classification unit is itself ordered within one acquisition; no paper in this block has that structure.

**8 of 17 included papers state no patient count at all** (`n_patients` null), and **3 of 17 never define what one scored unit is** (`evaluation_unit_reported` = unclear). Those are findings, coded as the codebook requires rather than guessed away.

---

## 7. Three papers, one dataset, no unit

Positions 111 (34149118), 145 (36619376) and 156 (37360135) all classify the same public SARS-CoV-2 CT-scan dataset (Soares et al., Kaggle) and report 94.07%, 99.4% and 99.79477% accuracy respectively. **None of the three states whether one of its 2,482 units is a cross-section or a volume, and none states a patient count.** All three are coded `evaluation_unit_reported = unclear` and `n_patients = null`.

Position 135 (34117783), in this same block, uses that dataset too -- and prints a table headed "Statistics of the datasets: Dataset | Patients | Slices", giving **120 patients, 2,482 slices**. So the answer exists, in a paper drawn by the same permutation.

That cross-reference was **not** used to upgrade the other three records. The governing principles forbid inferring a code from what the authors probably did, so `unclear` stands. But 34117783 then splits at the image level anyway -- "we divided each dataset into 5 pieces ... the proportion of the training set, validation set, and test set was 3:1:1", with the unit being the CT image -- so slices from one of its 120 patients fall on both sides of the split, in a paper that had every number needed to prevent it.

---

## 8. Two stage-1 traps worth carrying into block R2

**PMID 41617821** (position 152), *Sci Rep* 2026, "Dynamic graph convolution ... for precise lymph node metastasis detection". The abstract names CT and PET ("current clinical practices, including CT, PET imaging, and microscopic examination"), which is why the frame matched it. The full text: *"This study is developed and evaluated exclusively on H&E-stained histopathology WSIs from the CAMELYON17 dataset."* No CT. No PET. Coding from the abstract would have produced a false include with a reported 98.65% accuracy. This is a concrete argument for the protocol's "when in doubt, go_to_fulltext" default.

**PMID 34776805** (position 137) is the block's D10 case. E-SEG is first in the listed order and names segmentation, but its text is qualified "with NO categorical class decision evaluated" -- and here a class decision *was* evaluated (abscess, tract and internal-orifice localisation, 84%/80%/92% vs 60%/68%/72%), just by human readers rather than by a fitted model. D10 routes it to `E-NOCLF`. Without D10 this record would have been coded E-SEG mechanically, exactly as four screeners did to PMID 40335658 in the main sample.

---

## 9. Codebook gap found in this block

The `batch` field enum is `["overlap","batch_A","batch_B","batch_C","batch_D"]` and has **no level for a reserve block**, although section 3.1 pre-specifies reserve blocks and `screen_sample.json` pre-assigns every reserve record round-robin via `reserve_assigned_to`. No enum level was invented. `batch` carries the sample file's own `reserve_assigned_to` value and a separate non-enum key `reserve_block = "R1"` records the block. Logged for a future amendment rather than resolved unilaterally, in keeping with the rule that nothing is edited silently.

---

## 10. Flagged for adjudication

29 of 50 records carry `flag_for_adjudication = true`, and every one of them has its reasoning written out in the record's `notes` field. The ones that could move a number:

| PMID | what a second screener should check | could it move an endpoint? |
|---|---|---|
| 37679806 | `other_non_imaging` vs `clinical_or_demographic_only` for the STAS arm | No -- both are S1, neither is P1 |
| 36087795 | Is the Mayo Clinic model pixel-free? | Yes -- S1 would go 1/17 to 2/17. P1 unaffected |
| 39280340 | Include (a CNN sees the slices) vs `E-DERIV` (final classifier is LR over 9 features) | Yes -- the included denominator |
| 39022286 | Same question for a radiomics GBDT-LR alongside a DL arm | Yes -- the included denominator |
| 37782590, 37742486, 39302179, 35487442, 37436866 | Were these five safe to exclude from the abstract alone (D11)? If not, they reach stage 2 and become unreachable | Yes -- S6 would rise from 9/26 toward 14/31 |
| 38324428 | Converse check: was it right to send this one to stage 2 rather than exclude it? | Yes -- S6 |
| 35655831 | `site_or_centre` vs `random_unit_not_stated` for the split | Yes -- S4 |

The fifth row is the one that matters most, and it is stated plainly because it cuts against this screener's own numbers: **the stage-1 exclusions are the single largest discretionary lever on S6 in this block.** Sensitivity, worked explicitly: overturning the five flagged ones sends S6 from 9/26 = 34.6% to 14/31 = **45.2%**; overturning all seven sends it to 16/33 = **48.5%**. (The two unflagged ones are 35680755, whose own abstract says "we ... reviewed the status and prospects", and 41273425, which reports only DSC, Hausdorff distance and area difference; neither looks arguable.) D4 exists precisely so that S6 measures the reachability of the *eligible-looking* literature, and D11 sets the bar for using it -- an exclusion code unambiguous from the abstract alone. Each of the seven exclusion quotes is reproduced in full in the JSON so the call can be checked rather than taken on trust. Note the direction: every one of these overturns makes the headline bounding interval WIDER, i.e. makes this paper's own result weaker, which is why the lever is disclosed rather than left implicit.

---

## 11. Record-by-record index

| pos | PMID | venue | decision | code | access |
|---|---|---|---|---|---|
| 111 | 34149118 | Comput Commun 2021 | included | - | oa_pmc_or_publisher |
| 112 | 40721771 | BMC Med Imaging 2025 | included | - | oa_pmc_or_publisher |
| 113 | 40627160 | Eur J Nucl Med Mol Imaging 2025 | included | - | oa_pmc_or_publisher |
| 114 | 41929976 | J Alzheimers Dis Rep 2026 | included | - | oa_pmc_or_publisher |
| 115 | 37782590 | IEEE Trans Med Imaging 2024 | excluded | E-SEG | not_attempted_excluded_at_stage1 |
| 116 | 40766163 | medRxiv 2025 | included | - | oa_pmc_or_publisher |
| 117 | 39515189 | Comput Med Imaging Graph 2024 | included | - | preprint_version_only |
| 118 | 35181263 | Respir Investig 2022 | unreachable | - | unreachable_paywalled |
| 119 | 34341737 | Quant Imaging Med Surg 2021 | included | - | oa_pmc_or_publisher |
| 120 | 39280340 | Front Bioeng Biotechnol 2024 | included | - | oa_pmc_or_publisher |
| 121 | 35680755 | J Nucl Cardiol 2023 | excluded | E-TYPE | not_attempted_excluded_at_stage1 |
| 122 | 35774412 | IEEE J Transl Eng Health Med 2022 | excluded | E-SEG | oa_pmc_or_publisher |
| 123 | 41899050 | J Clin Med 2026 | excluded | E-DERIV | oa_pmc_or_publisher |
| 124 | 35694573 | Comput Intell Neurosci 2022 | excluded | E-SEG | oa_pmc_or_publisher |
| 125 | 39797353 | J Clin Med 2025 | excluded | E-SEG | oa_pmc_or_publisher |
| 126 | 37679806 | Cancer Imaging 2023 | included | - | oa_pmc_or_publisher |
| 127 | 39302179 | Med Phys 2024 | excluded | E-SEG | not_attempted_excluded_at_stage1 |
| 128 | 35487442 | Methods 2022 | excluded | E-DERIV | not_attempted_excluded_at_stage1 |
| 129 | 38786273 | Diagnostics (Basel) 2024 | excluded | E-SEG | oa_pmc_or_publisher |
| 130 | 42078359 | medRxiv 2026 | included | - | oa_pmc_or_publisher |
| 131 | 36087795 | Chest 2023 | included | - | oa_pmc_or_publisher |
| 132 | 40337176 | J Biomed Opt 2025 | excluded | E-2D | oa_pmc_or_publisher |
| 133 | 39022286 | Quant Imaging Med Surg 2024 | included | - | oa_pmc_or_publisher |
| 134 | 40463652 | Neurooncol Adv 2025 | included | - | oa_pmc_or_publisher |
| 135 | 34117783 | Med Phys 2021 | included | - | oa_pmc_or_publisher |
| 136 | 42443645 | J Imaging Inform Med 2026 | unreachable | - | unreachable_paywalled |
| 137 | 34776805 | Contrast Media Mol Imaging 2021 | excluded | E-NOCLF | oa_pmc_or_publisher |
| 138 | 37015600 | IEEE J Biomed Health Inform 2023 | unreachable | - | unreachable_paywalled |
| 139 | 35578678 | SN Comput Sci 2022 | excluded | E-TYPE | oa_pmc_or_publisher |
| 140 | 37742486 | Comput Methods Programs Biomed 2023 | excluded | E-SEG | not_attempted_excluded_at_stage1 |
| 141 | 37587160 | Sci Rep 2023 | excluded | E-SEG | oa_pmc_or_publisher |
| 142 | 40290188 | Front Cardiovasc Med 2025 | excluded | E-PROJ | oa_pmc_or_publisher |
| 143 | 39227330 | J Clin Neurol 2024 | excluded | E-DERIV | oa_pmc_or_publisher |
| 144 | 31003928 | EBioMedicine 2019 | excluded | E-NOCLF | oa_pmc_or_publisher |
| 145 | 36619376 | Chemometr Intell Lab Syst 2023 | included | - | oa_pmc_or_publisher |
| 146 | 36086447 | Annu Int Conf IEEE Eng Med Biol Soc 2022 | unreachable | - | unreachable_paywalled |
| 147 | 40232413 | Abdom Radiol (NY) 2026 | unreachable | - | unreachable_paywalled |
| 148 | 34668251 | NMR Biomed 2022 | excluded | E-DERIV | oa_pmc_or_publisher |
| 149 | 40277031 | Mycoses 2025 | unreachable | - | unreachable_paywalled |
| 150 | 41273425 | Eur Radiol 2026 | excluded | E-SEG | not_attempted_excluded_at_stage1 |
| 151 | 35592700 | Front Aging Neurosci 2022 | excluded | E-DERIV | oa_pmc_or_publisher |
| 152 | 41617821 | Sci Rep 2026 | excluded | E-2D | oa_pmc_or_publisher |
| 153 | 38324428 | IEEE Trans Med Imaging 2024 | unreachable | - | unreachable_paywalled |
| 154 | 40828373 | EJNMMI Phys 2025 | excluded | E-SEG | oa_pmc_or_publisher |
| 155 | 36077686 | Cancers (Basel) 2022 | excluded | E-DERIV | oa_pmc_or_publisher |
| 156 | 37360135 | Wirel Pers Commun 2023 | included | - | oa_pmc_or_publisher |
| 157 | 33741850 | Nucl Med Commun 2021 | unreachable | - | unreachable_paywalled |
| 158 | 37436866 | IEEE Trans Biomed Eng 2024 | excluded | E-DERIV | not_attempted_excluded_at_stage1 |
| 159 | 35655831 | Quant Imaging Med Surg 2022 | included | - | oa_pmc_or_publisher |
| 160 | 40505211 | Med Image Anal 2025 | unreachable | - | unreachable_paywalled |

Full coded records, with a verbatim quote and location behind every coded field and the complete 14-term search string behind every negative, are in `paper/screen_reserve_R1.json`.

