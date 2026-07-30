# Reserve block R3 -- permutation positions 211-260

Fifty records, screened under `paper/screen_frame.json` **v1.2** and `paper/screen_protocol.md` (v1.0 frozen, amended v1.2, access pass v1.3).
Machine-readable codes, quotes and search records: **`paper/screen_reserve_R3.json`**. This file is the human-readable companion; every number in it is computed from that JSON, not typed by hand.

The block was taken by permutation position from `paper/screen_sample.json`, in order, with no substitution and no truncation, as the pre-registered extension rule (protocol sec.3.1) requires. The codebook and the protocol were read in full before any record in this block was read.

---

## 1. The headline

**Not one of the 19 included papers reports a zero-image baseline.** Across all 50 records and all six `trivial_baseline` sub-flags, **not one sub-flag is coded true by anyone, anywhere** -- not the four P1-family flags, not `clinical_or_demographic_only`, not `other_non_imaging`.

| endpoint | this block | Wilson 95% |
|---|---|---|
| **P1** -- reports >=1 zero-image baseline with a measured value | **0/19** | 0.0% [0.0%, 16.8%] |
| S1 -- reports >=1 non-imaging baseline of any kind (P1 family + clinical-only) | 0/19 | 0.0% [0.0%, 16.8%] |
| S6 -- unreachable, over the eligible-looking set | 9/28 | 32.1% [17.9%, 50.7%] |

**S1 is zero as well**, which R1 and R2 were not -- each of those blocks carried exactly one S1 positive (R1: PMID 37679806 on `other_non_imaging`; R2: PMID 40121941 on `clinical_or_demographic_only`). R3 carries none. The block contains exactly one genuine near-miss and it is coded FALSE on the evidence:
PMID **40338065** (*J Comput Assist Tomogr*) reports a model its own legend calls the "clinical imaging model" with measured AUCs of 0.712 (training) and 0.690 (internal validation) -- but the four features that survived selection are *Age, Cirrhosis, AFP and* ***Intratumor vascularity***, and intratumoral vascularity is a radiologist-read imaging sign. The codebook requires `clinical_or_demographic_only` to use "patient variables only ... with no imaging", so the flag is false. Coding it true would have put an S1 positive on the board cheaply; it would have been wrong.
The second near-miss, PMID **37745691**, runs a parallel "semantic" analysis of sex, smoking history and performance status against mutation status and concludes the two approaches are comparable -- but it reports that analysis as chi-square tests and odds ratios, never as accuracy or AUC on the model's metric, and its variable list includes imaging findings (ring enhancement, meningeal involvement). No measured value on the same metric, no pixel-free model: false.

**S6 is 32.1% [17.9%, 50.7%], above the 15% threshold in protocol sec.7.** Within this block the reportable form of the primary is therefore the bounding interval, not the point estimate: **[0.0%, 32.1%]** over the 28 eligible-looking records, the lower bound coding all 9 unreachable papers as not reporting a baseline and the upper bound coding all 9 as reporting one. Adding records does not narrow that. Recovering full texts does.

---

## 2. Tallies

| | n |
|---|---|
| records screened | 50 |
| excluded at stage 1 (abstract alone unambiguous, D11) | 14 |
| excluded at stage 2 (full text obtained and read first) | 8 |
| **excluded, total** | **22** |
| **unreachable, eligibility unresolved** | **9** |
| **included** | **19** |
| eligible-looking denominator (included + unreachable) | 28 |
| **reporting ANY zero-image baseline** | **0** |

Exclusion codes, reported individually as protocol sec.9 requires:

| code | n | what they were |
|---|---|---|
| `E-SEG` | 7 | segmentation, region-detection, landmark localisation, denoising, registration or synthesis with no categorical class decision evaluated |
| `E-2D` | 5 | no slice axis reaches the model: chest radiographs, H&E whole slides, LC25000 histopathology, and **two records where a volumetric-capable modality was acquired as a single cross-section** (see sec.5) |
| `E-DERIV` | 5 | the fitted classifier eats a radiomics vector, an RNFL-thickness table, a volumetry table, CFD flow features or an fMRI connectivity matrix -- not an image |
| `E-TYPE` | 2 | a journal 'Practice and Policy' review and a systematic literature review |
| `E-PROJ` | 1 | SPECT bull's-eye polar maps -- a flattened surface map with no slice axis (pilot amendment A1) |
| `E-NOCLF` | 1 | MRI reconstruction where the only class decision (intraplaque haemorrhage) is a human reader's, coded under D10 rather than E-SEG |
| `E-NONMED` | 1 | macaque fMRI and monkey face photographs |

Five `E-DERIV` and seven `E-SEG` records in fifty is the frame's deliberate breadth doing exactly what protocol sec.2.1 said it would. `E-DERIV` is reported separately because those papers are inside the query and outside the failure mode.

---

## 3. Secondary endpoints in this block

| id | statement | this block | Wilson 95% |
|---|---|---|---|
| S1 | any non-imaging baseline | 0/19 | 0.0% [0.0%, 16.8%] |
| S2 | headline unit is the slice | 3/19 | 15.8% [5.5%, 37.6%] |
| S3 | among papers reporting any slice-level metric, also report patient-level | 0/3 | 0.0% [0.0%, 56.1%] |
| S4 | explicitly states a subject-level split | 5/19 | 26.3% [11.8%, 48.8%] |
| S5 | reports or discusses the positional distribution of labels | 0/19 | 0.0% [0.0%, 16.8%] |
| S6 | unreachable, over included + unreachable | 9/28 | 32.1% [17.9%, 50.7%] |
| S8 | subject-clustered uncertainty interval | 0/19 | 0.0% [0.0%, 16.8%] |
| S9 | reports n positive patients as well as n positive slices | 4/19 | 21.1% [8.5%, 43.3%] |

**S3 is 0/3.** All three slice-level papers report the slice metric and nothing else: not one of them also reports a patient-level number. **S8 is zero**: not one of the 19 included papers reports a subject-clustered interval, and 14 report no interval of any kind.

**S5 is zero, and the nearest miss is worth quoting** because it is the exact construction our audit formalises. PMID **40301455** selects its dataset *by slice index*: "Since PD is associated with the substantia nigra, which is located in the middle of the brain. Therefore, we were interested in middle-order slices ... we took the mode of each study and picked median slices with two standard deviation slices." That quantifies which indices were *sampled*; it says nothing about how the label is distributed along the axis, which is what D9's test asks, so it is coded `no`. The paper then splits those 2,490 slices 70/10/20 with no subject separation stated and reports 96% slice-level accuracy over 498 subjects. A constant or positional baseline is precisely the check that design needs, and it is not there.

**S7, headline unit crossed with the P1 flag** (exact counts, all cells):

| effective headline unit | n | P1 true | P1 false |
|---|---|---|---|
| patient | 7 | 0 | 7 |
| unclear | 4 | 0 | 4 |
| volume_or_scan_not_patient | 3 | 0 | 3 |
| slice | 3 | 0 | 3 |
| other | 1 | 0 | 1 |
| lesion | 1 | 0 | 1 |
| **total** | **19** | **0** | **19** |

Distributions on the fields the endpoints rest on, `unclear` reported as its own category and never merged:

- **`evaluation_unit_reported`**: patient 7, unclear 4, volume_or_scan_not_patient 3, slice 3, other 1, lesion 1
- **`split_unit`**: patient_subject 5, random_unit_not_stated 5, slice_or_image 4, lesion_or_roi 2, scan_or_study 1, unclear 1, external_cohort_only 1
- **`split_disjointness_verified`**: not_stated 13, stated_only 3, stated_and_checked 3
- **`modality`**: CT 8, MRI 7, OCT 1, multiple 1, CBCT 1, PET_CT 1
- **`input_representation`**: 2D_slice 5, patch_3D 4, 3D_volume 4, unclear 3, mixed 2, 2.5D_stack 1
- **`code_availability`**: not_stated 15, public_link_stated 3, public_link_works 1
- **`uncertainty_interval_reported`**: none 14, ci_unspecified_method 3, sd_across_folds 2
- **`dataset_public`**: private 10, public 7, mixed 2
- **`n_positive_reported`**: patients_only 9, slices_only 4, patients_and_slices 4, neither 2
- **`label_broadcast_to_slices`**: unclear 7, na 6, false 3, true 3

Four of 19 included papers never say what one scored unit is. Five split with no unit named at all and one defers the question entirely. Thirteen of 19 never state whether the split respected the subject. These are not coding failures; the codebook says so explicitly, and they are the finding.

---

## 4. Access

The five-rung ladder in protocol sec.7 was worked in order for all 36 records that reached stage 2. Per **D4** it was **not** climbed for the 7 closed-access records excluded at stage 1, so S6 measures the reachability of the eligible-looking literature and is not inflated by records that were never eligible.

| rung that worked | n |
|---|---|
| `oa_pmc_or_publisher` | 34 |
| `unreachable_paywalled` | 9 |
| `not_attempted_excluded_at_stage1` (D4) | 7 |

Seven of the fourteen stage-1 exclusions were in PMC, so the full text was in hand anyway and was used to *verify* the exclusion code; D4's second sentence permits recording the rung that worked, and it is recorded.

**No recoveries at rungs 3-5, and every failed rung is named in the JSON.** All 9 unreachable records were checked against Unpaywall, OpenAlex, Semantic Scholar, Europe PMC (`fullTextUrlList` plus an exact-title search), OpenAIRE and arXiv. Eight of the nine are genuinely closed -- all four independent sources agree `is_oa=false` and no repository, accepted manuscript or preprint exists anywhere.

**Two disclosures, in the direction that costs us rather than the direction that flatters us:**

1. **PMID 38723886 (*J Nucl Cardiol*) is demonstrably open access and is counted unreachable anyway.** Elsevier's own article API returns `<openaccessArticle>true</openaccessArticle>`, `<openaccessType>Full</openaccessType>` and `<openaccessUserLicense>http://creativecommons.org/licenses/by-nc-nd/4.0/</openaccessUserLicense>`; Unpaywall and Semantic Scholar both report `is_oa=true`. It is unreachable only because sciencedirect.com and journalofnuclearcardiology.org return HTTP 403 to every client in this environment and the Elsevier full-text API returns 401 without a subscriber key. It is counted unreachable because **no full text was read**, and the cause is disclosed rather than charged to the literature. Recovering it elsewhere would give 8/28 = 28.6%, still above 15%.
2. **Three records were refused by a JavaScript bot-detection challenge, which was not circumvented.** link.springer.com and nature.com served a `Client Challenge` interstitial for PMIDs 41361534, 33428062 and 38433144. All three are independently confirmed closed by Unpaywall, OpenAlex, Semantic Scholar and Europe PMC, so the challenge is not what makes them unreachable -- but the refusal is recorded rather than papered over.

**No infringing source was used at any point.** Sci-Hub, LibGen and equivalents were not accessed. Where a paper was reachable only through one, it stayed unreachable.

At least three of the nine would very probably have been *included* on full text -- 34674280 (multi-task nodule segmentation + invasiveness classification, AUC 93.8% over 1,626 patients), 33428062 (chest CT, three histological classes, accuracy 57.7% vs 34.2%) and 38723886 (1,038 patients, gated SPECT, AUC 0.82 with a 95% CI). They are counted unreachable because no full text was read. **D1** forbids coding them included on abstract evidence, for the reason the codebook gives: an included record must carry an *evidenced* `trivial_baseline`, and the mandatory fourteen-term search cannot be run on a text nobody has. That is the whole cost of the access constraint, and it is the constraint that binds this screen.

---

## 5. The two judgement calls that move this block's denominator

Both are flagged in the JSON (`flag_for_adjudication: true`, `screener_confidence: medium`) and both are the same codebook gap.

Criterion **I3** requires the classifier's input to come from "a VOLUMETRIC acquisition, i.e. one natively reconstructed as an ordered stack of 2D cross-sections". Two records use a modality that is *on I3's qualifying list* but acquire a **single cross-section**, so no slice axis exists:

- **PMID 39166052** (*Heliyon*, anterior-segment OCT). "All images were acquired in the 'anterior segment quadrant' mode at the 0deg-180deg, 45deg-225deg, 90deg-270deg, and 135deg-315deg meridians" on a Zeiss Visante Model 1000. 11,035 images / 2,833 eyes = 3.9 images per eye: four discrete meridional B-scans, not a stack.
- **PMID 40478199** (*Liver Int*, UK Biobank liver MRI). "a magnitude image and its corresponding phase image were captured as **single transverse slices** during end-expiration breath-hold ... voxel size 1.719 x 1.719 x 10.0 mm". One 10 mm slice per participant.

Both are coded `E-2D` on that code's **headline rule** ("Imaging is natively 2D") rather than on its example list, which contains only natively-2D *modalities* and does not contemplate this case. `E-PROJ` was considered and rejected: a B-scan is not a collapsed projection.

**The sensitivity analysis is exact, not hypothetical.** The fourteen-term search was run on both records, main text *and* supplement, before they were excluded. Neither reports a measured pixel-free comparator. So if adjudication overturns both exclusions:

| | as coded | if 39166052 and 40478199 are included instead |
|---|---|---|
| included | 19 | 21 |
| **P1** | **0/19** = 0.0% [0.0%, 16.8%] | **0/21** = 0.0% [0.0%, 15.5%] |
| S6 | 9/28 = 32.1% [17.9%, 50.7%] | 9/30 = 30.0% [16.7%, 47.9%] |

The primary result does not move either way. Recording the call, and its cost, is the point.

PMID 39166052 also carries a finding worth preserving even though it is out of the denominator: "The training and test sets comprised images split at the image level and were not categorized by eye or patient type" -- an explicit image-level split with no eye or patient separation, in a paper reporting AUC 0.980.

---

## 6. What was flagged, and why

| PMID | disposition | why flagged |
|---|---|---|
| 39166052 | excluded (E-2D) | criterion-I3 judgement call (single-cross-section AS-OCT); see sec.5 |
| 40478199 | excluded (E-2D) | criterion-I3 judgement call (single-slice UK Biobank liver MRI); see sec.5 |
| 41767013 | included | `screener_confidence: low`, which the codebook's low_confidence_rule makes an automatic adjudication trigger. Meets I1-I4 on its face and is included, but section 4.2 describes "Video samples incorporated temporal jitter", reports "top-1 and top-5 recognition accuracy", and section 4.3 discusses results "within the Kinetics Dataset" -- a video action-recognition benchmark that appears nowhere in its dataset section. A data-integrity concern, not an eligibility criterion, so it is flagged rather than acted on unilaterally. |
| 42433232 | included | the only included record whose declared attachments could not be obtained. They are administrative only (TRIPOD checklist, COI form, data-sharing statement, peer-review file); Europe PMC returns 404 for this PMCID and PMC serves a download interstitial. Disclosed rather than glossed. |
| 38988988 | included | self-contradictory reporting -- 99.40% accuracy described as achieved "in segmentation tasks" while tabulated against other papers' classification accuracies -- and PubMed records an Erratum (PMID 40529250). n_patients is never stated. |
| 36588765 | excluded (E-TYPE) | a systematic literature review that also runs one original experiment. Coded E-TYPE; the direction of the call is recorded (it removes a record that reports no baseline, so it cannot flatter P1's numerator). |
| 35620201 | included | abstract names five injury categories but reports six accuracies, and the venue has a large retraction record. PubMed and Europe PMC were checked on 2026-07-29: no retraction or expression of concern for this PMID. |

---

## 7. What this block adds to the extension

R3 contributes **19 included papers** and **9 unreachable** to the running totals. Whether the extension rule's target of 75 included papers is met is computed at analysis time across R1+R2+R3 and the analysis sample, not here.

Three things in this block are worth carrying into the paper:

1. **The three records that most need a zero-image baseline are the three that come closest to being unable to detect it, and none of them reports one.** PMID 35360446 (COVID-Net CT-2) does the split correctly at the patient level, broadcasts the volume label to every abnormal slice, and then reports 99.0% accuracy over ~194,922 *slices*. PMID 40301455 selects slices by index around the substantia nigra, splits 2,490 slices 70/10/20 with no subject separation stated, and reports 96% slice-level accuracy over 498 subjects. PMID 35203433 splits 3,064 slices from 233 patients 60/20/20 with a table headed "Number of Slices". Zero baselines between them.
2. **The strongest methodological paper in the block is also a clean negative.** PMID 37152810 (2,648 patients, 14 institutions, three external hold-out cohorts, code released) fuses two *non-imaging* features -- patient age and atlas-derived tumour location -- into the network and runs a four-way ablation over them. Every arm of that ablation still contains the image. There is no age-only, no location-only and no age+location-only arm, which is exactly the arm a paper fusing non-imaging features has the strongest reason to fit.
3. **Reading the search hits instead of counting them kept six records off the numerator.** Six contain a term from the mandatory list in a construction that looks like a trivial baseline and is not: PMID 31941918's seven-CNN "Majority Voting rule", PMID 38988988's description of another paper's KNN-RF-DT majority vote, PMID 39369213's "95% permutation interval" (a confidence interval on the model's own AUC, not a permuted-label null), PMID 37960407's "Adam and SGD are chosen as two baselines" (optimisers), PMID 33846450's "highest accuracy permutation of the pre-training and augmentation variations" (an experimental grid), and PMID 41767013's "ResNet baseline 46 ms" (an inference-time comparison). A screen that grepped and tallied would have reported a non-zero P1 here.

---

## 8. Honesty statement

- The pre-registered protocol governs. Nothing in the frame, the permutation, the sample, the endpoint definitions, the Wilson interval method or the 15% threshold was altered to produce these numbers, and no codebook rule was amended in the course of coding this block. The two I3 judgement calls in sec.5 are applications of an existing rule to a case it does not cover, are flagged as such, and are reported with their exact cost.
- **No unevidenced negative was accepted.** Every one of the 19 included records carries the fourteen-term search over the full text, and the supplement position is stated for each: obtained and searched where one exists, "none exists" where the OA package holds figures only, and "declared but not obtained" for the single record (42433232) where that is the truth.
- Every record that could not be reached is coded `not_assessable` on all six sub-flags, never `false`. Not one FALSE is entered anywhere in this file without the fourteen-term search behind it.
- **Sci-Hub and every other infringing source were not used**, and bot-detection challenges were not circumvented. Where a paper was reachable only through one, it stayed unreachable.
- A positive was looked for as hard as the absence: every hit on every one of the fourteen terms in every reachable full text and supplement was read in context. The one construction in the whole block that comes closest to a trivial baseline is in an *excluded* record and is recorded there anyway -- PMID 36997134's "classification performance of model (6) to 64 randomly generated white-noise images (mu=0, sigma=1) (denoted as `baseline`)" -- and it would still not qualify, because white noise is an image.
- This file is one operator's coding. It enters no agreement statistic, and it does not discharge the outstanding action named in `screen_frame.json`: a fresh independent four-screener re-coding under v1.2.
