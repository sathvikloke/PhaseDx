# Screening results — BATCH B + shared OVERLAP SET

**Screener:** S2  **Coded:** 2026-07-29  **Records:** 36 (15 overlap, positions 1–15; 21 batch B, positions 38–58)

Protocol `paper/screen_protocol.md` v1.0 (frozen). Codebook `paper/screen_frame.json` v1.0. Sample `paper/screen_sample.json` v1.1.
Machine-readable records, with every field and its supporting quote, are in `paper/screen_batch_B.json`.

**Independence.** The overlap set was coded without reading any other screener's output. No file named screen_batch_A.json / screen_batch_C.json / screen_batch_D.json was opened, and no overlap-set paper was discussed with another screener before this file was written. The only project files read before coding were paper/screen_protocol.md, paper/screen_frame.json and paper/screen_sample.json.

---

## 1. Headline tallies

| | overlap (15) | batch B (21) | total (36) |
|---|---|---|---|
| included | 6 | 5 | **11** |
| excluded | 5 | 12 | **17** |
| unreachable, eligibility unresolved | 4 | 4 | **8** |

Exclusion codes (17): **E-DERIV 7**, **E-SEG 6**, **E-NOCLF 2**, **E-TYPE 1**, **E-2D 1**.

Full text reached: 18 via PMC/publisher OA, 1 via an institutional repository (rung 4), 9 not attempted because the record was excluded at stage 1, **8 unreachable behind a paywall**.

21 of 36 records carry `flag_for_adjudication = true`; 8 are `screener_confidence = low` (all 8 are the unreachable records).

## 2. Endpoints, for these 36 records only

> These are **not** the study estimate. They cover 36 of the 100 sampled records, 15 of which are shared with three other screeners and must be deduplicated before pooling. The §3.1 extension rule is not applied here. Wilson intervals are shown only to indicate the precision available at this batch size.

| endpoint | k/n | % | Wilson 95% |
|---|---|---|---|
| **P1 — reports a zero-image baseline** | **0/11** | **0.0** | **0.0–25.9** |
| S1 — any non-imaging baseline (incl. clinical-only) | 2/11 | 18.2 | 5.1–47.7 |
| S2 — headline unit is the slice | 3/11 | 27.3 | 9.7–56.6 |
| S3 — of papers reporting a slice metric, also report patient | 2/4 | 50.0 | 15.0–85.0 |
| S4 — explicit subject-level split | 4/11 | 36.4 | 15.2–64.6 |
| S5 — positional distribution of labels reported | 0/11 | 0.0 | 0.0–25.9 |
| S6 — unreachable | 8/19 | 42.1 | 23.1–63.7 |
| S8 — subject-clustered uncertainty interval | 0/11 | 0.0 | 0.0–25.9 |
| S9 — n positive patients *and* slices | 2/11 | 18.2 | 5.1–47.7 |

**P1 bounding analysis (reported unconditionally, per §7 rule 3).** Over included + unreachable (n = 19): lower bound 0/19 = 0.0% [0.0, 16.8]; upper bound 8/19 = 42.1% [23.1, 63.7].

**§7 rule 4 is triggered in this batch.** Unreachable records are 42.1% of the eligible-looking set, far above the 15% threshold, so for these rows the bounding interval — not the complete-case 0/11 — is the honest summary. Eight of the eight unreachable papers sit behind Springer, Elsevier, Wiley, Oxford, IEEE or RSNA paywalls; as §7 rule 5 predicts, these skew toward higher-tier clinical journals, i.e. exactly the venues most likely to demand a comparator arm, so silent exclusion would push P1 *down*, in the direction that flatters our thesis.

### S7 — headline unit against the P1 flag (included + reachable, n = 11)

| headline unit | n | P1 true |
|---|---|---|
| slice (unit chosen by rule A5 among two reported) | 1 | 0 |
| patient (unit chosen by rule A5 among two reported) | 1 | 0 |
| only one unit reported — lesion | 3 | 0 |
| only one unit reported — slice | 2 | 0 |
| only one unit reported — patient | 2 | 0 |
| only one unit reported — other (per-vertebra) | 1 | 0 |
| only one unit reported — unclear | 1 | 0 |

## 3. Distributions among the 11 included, reachable papers

- **evaluation unit reported**: `lesion` 3, `both` 2, `patient` 2, `slice` 2, `other` 1, `unclear` 1
- **split unit**: `patient_subject` 4, `slice_or_image` 4, `external_cohort_only` 1, `random_unit_not_stated` 1, `unclear` 1
- **split disjointness verified**: `not_stated` 6, `stated_only` 5
- **positional distribution reported**: `no` 11
- **uncertainty interval reported**: `ci_unspecified_method` 7, `none` 3, `sd_across_folds` 1
- **input representation**: `2D_slice` 7, `patch_3D` 3, `mixed` 1
- **n positive reported**: `patients_only` 4, `slices_only` 3, `patients_and_slices` 2, `na` 1, `neither` 1
- **label broadcast to slices**: `na` 5, `true` 3, `false` 2, `unclear` 1
- **modality**: `CT` 8, `MRI` 3
- **dataset public/private**: `private` 8, `public` 3
- **headline test set**: `internal_held_out` 9, `external` 1, `unclear` 1
- **code availability**: `none` 9, `public_link_stated` 2

Three facts stand out and each is fully quoted in the JSON:

1. **Not one of the 11 reports any zero-image baseline** — no constant/prevalence arm, no positional arm, no acquisition-metadata arm, no permutation null. The two non-imaging comparators that exist are both clinical-variables nomogram arms (S1, not P1).
2. **Not one of the 11 reports the positional distribution of its labels**, and not one reports a subject-clustered interval, although 7 of 11 report a confidence interval of some kind.
3. **Four of the 11 split at the image/slice level and one names no unit at all.** Only 4 of 11 state a subject-level split, and none verifies disjointness.

## 4. Records

### 4.1 Overlap set (positions 1–15)

#### pos 1 — PMID 36776294 — INCLUDED

*Diagnosis of cervical lymph node metastasis with thyroid carcinoma by deep learning application to CT images.* — Front Oncol 2023. <https://pubmed.ncbi.nlm.nih.gov/36776294/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `lesion` — "The 676 lymph nodes were randomly divided into 70% of the training set (73 benign and 401 malignant lymph nodes) and 30% of the test set (30 benign and 172 malignant lymph nodes)." (Results, 'Dataset'; repeated in Abstract-Results). Metrics ACC/TPR/TNR/AUC in Table 3 are computed over the 202 test lymph nodes.
- **headline unit**: `na_only_one_unit_reported` — Only lymph-node-level performance is reported anywhere; no patient-level metric appears in the paper.
- **split unit**: `slice_or_image` (disjointness `not_stated`) — "For detection, the data set was randomly subdivided into a training set (70%) and a testing set (30%)" and "For classification, ... The training and testing sets for classification were set as same as the detection." (Methods, Dataset). The units split are the 676 lymph-node ROIs taken from 574 axial CT images of 196 patients; the word 'patient' never appears in any split sentence.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists (no supplementary material in the PMC record)
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists
- **positional distribution**: `no` — The only hits for 'position' are network internals: "Nreg is the number of anchor positions" (Methods, loss function) and "long-range dependencies with precise positional information" (Methods, coordinate attention). No distribution of the label along the slice axis is shown or stated.
- **headline**: accuracy = 0.96 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); A-ResNet50-W classification network, 202-node internal test set; the same sentence also gives AUC 0.894
- **cohort**: dataset `private_single_centre` (private), modality `CT` (mixed=false), region `head_neck`; n_patients=196, n_patients_test=—, n_slices_or_images=574, n_positive_reported=`neither`
- **other**: interval `none`; input `2D_slice`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: JUDGEMENT CALL on split_unit: the split is at the lymph-node level, which is not one of the enum levels. 'slice_or_image' was chosen because the nodes are ROIs cropped from axial CT images and a patient contributes several nodes/images; it is certainly NOT patient-level. A second screener could reasonably code 'random_unit_not_stated'. n_positive_reported coded 'neither' because positives (103 benign / 573 malignant) are counted per lymph node, neither per patient nor per slice.

#### pos 2 — PMID 41617832 — UNREACHABLE_ELIGIBILITY_UNRESOLVED

*Multimodal deep learning for laryngeal squamous cell carcinoma staging using CT and laryngoscopy.* — Eur Radiol 2026. <https://pubmed.ncbi.nlm.nih.gov/41617832/>

- **full text**: `unreachable_paywalled` / version used `abstract_only`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `patient` — "This retrospective multicenter study included 450 patients ... They were divided into training (n = 235), internal validation (n = 101), and external validation (n = 114) cohorts." (Abstract, Materials and Methods)
- **headline unit**: `na_only_one_unit_reported` — Only patient-level AUCs appear in the abstract; no sub-patient unit is mentioned.
- **split unit**: `patient_subject` (disjointness `stated_only`) — "450 patients ... were divided into training (n = 235), internal validation (n = 101), and external validation (n = 114) cohorts." (Abstract)
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=—, positional=—, acquisition_metadata=—, permuted_or_shuffled_label=—, clinical_or_demographic_only=—, other_non_imaging=—
  - evidence: NOT DETERMINABLE. The abstract names "a clinical logistic regression model [CL]" as a single-modality comparator and says the integrated model outperformed "all single- and dual-modality models (p < 0.05)", which implies a measured AUC for the clinical-only arm, but NO numeric value for CL is given in the abstract and the full text could not be obtained. Coded NULL rather than TRUE or FALSE.
  - searches_run: REQUIRED SEARCHES NOT RUN - full text unreachable (see screen_protocol.md sec.7 access ladder: PMC/publisher OA no; publisher site HTTP 403/paywall; no institutional subscription held by this screener; OpenAlex/Unpaywall/Crossref/arXiv searched for a repository, accepted-manuscript or preprint copy, none found; interlibrary loan / author request not initiated). The primary-endpoint flags are therefore recorded as NULL, not FALSE: an unevidenced negative on P1 is not accepted by the codebook.
- **positional distribution**: `unclear` — Not determinable from the abstract.
- **headline**: AUC = 0.888 (external, scope `pooled_2D_and_3D`, rule `abstract_multiple_took_external`); integrated multimodal model (clinical + CT + laryngoscopy), external validation cohort; internal cohort AUC 0.902
- **cohort**: dataset `private_multi_centre` (private), modality `CT` (mixed=true), region `head_neck`; n_patients=450, n_patients_test=114, n_slices_or_images=—, n_positive_reported=`neither`
- **other**: interval `ci_unspecified_method`; input `unclear`; label_broadcast_to_slices `unclear`; code `none`; chance_asserted_without_measurement=false
- **confidence** `low`; flag_for_adjudication=true
- **notes**: UNREACHABLE. Rule A2 applies: a volumetric CT arm (CT-DL) exists and was compared, but the headline number is the fused CT + 2D laryngoscopy + clinical model, so headline_value_scope='pooled_2D_and_3D' and the record is flagged. code_availability='none' is the enum's only option here and is NOT evidenced - the field has no 'unclear' level and the abstract says nothing about code.

#### pos 3 — PMID 39423605 — UNREACHABLE_ELIGIBILITY_UNRESOLVED

*Federated learning and deep learning framework for MRI image and speech signal-based multi-modal depression detection.* — Comput Biol Chem 2024. <https://pubmed.ncbi.nlm.nih.gov/39423605/>

- **full text**: `unreachable_paywalled` / version used `abstract_only`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `unclear` — The abstract never states what one scored unit is: "the DL model considered two modalities of inputs, such as speech signal and Magnetic Resonance Imaging (MRI) image ... Finally, both the outputs are fused utilizing the overlap coefficient."
- **headline unit**: `unclear` — Not determinable from the abstract.
- **split unit**: `unclear` (disjointness `not_stated`) — No split is described anywhere in the abstract.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=—, positional=—, acquisition_metadata=—, permuted_or_shuffled_label=—, clinical_or_demographic_only=—, other_non_imaging=—
  - evidence: NOT DETERMINABLE - full text unreachable.
  - searches_run: REQUIRED SEARCHES NOT RUN - full text unreachable (see screen_protocol.md sec.7 access ladder: PMC/publisher OA no; publisher site HTTP 403/paywall; no institutional subscription held by this screener; OpenAlex/Unpaywall/Crossref/arXiv searched for a repository, accepted-manuscript or preprint copy, none found; interlibrary loan / author request not initiated). The primary-endpoint flags are therefore recorded as NULL, not FALSE: an unevidenced negative on P1 is not accepted by the codebook.
- **positional distribution**: `unclear` — Not determinable from the abstract.
- **headline**: accuracy = 0.98 (unclear, scope `unclear`, rule `abstract_sentence`); ExpAPO-DCNN, fused MRI + speech output; no separate MRI-arm value is given
- **cohort**: dataset `unclear` (unclear), modality `MRI` (mixed=true), region `brain`; n_patients=—, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`neither`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `unclear`; code `none`; chance_asserted_without_measurement=false
- **confidence** `low`; flag_for_adjudication=true
- **notes**: UNREACHABLE, and eligibility genuinely unresolved: the abstract says 'pre-processing, feature extraction and detection' were applied to the MRI, which leaves open whether a spatially resolved image ever reached the network (I3) or whether it is an E-DERIV feature-vector paper. The abstract also reports only the fused MRI+speech output, so rule A2's requirement of a separately reported volumetric arm cannot be checked.

#### pos 4 — PMID 42130124 — INCLUDED

*Development and Validation of an AI-Integrated System for Automated Fracture Detection and Pedicle Puncture Planning in Lumbar Osteoporotic Vertebral Compression Fractures Based on the Nine-Grid Area Division Method.* — Orthop Surg 2026. <https://pubmed.ncbi.nlm.nih.gov/42130124/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `other` — "The fracture recognition model was trained and tested using 1018 vertebral samples derived from 216 L-OVCF patients." and Figure 7 legend: "Confusion matrix (A) illustrating the model's diagnostic performance on normal and fracture samples (total n = 204)." The scored unit is the individual vertebral body, not the patient and not the scan.
- **headline unit**: `na_only_one_unit_reported` — Only per-vertebra classification performance is reported; no patient-level fracture metric appears.
- **split unit**: `patient_subject` (disjointness `stated_only`) — "After labeling was completed, 10% of all cases were randomly selected to serve as an independent test set, excluded from model training." (Methods, Dataset Partitioning). 'Cases' are patients: Table 2 gives Age and Gender per case for the internal (n = 200) and external (n = 40) datasets.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists || SEARCH NOTE: The four 'baseline' hits are (a) the Table 2 demographic 'Baseline Characteristics' and (b) "the proposed two-stage algorithm significantly outperformed the baseline single-stage nnU-Net model (DSC = 0.853)" - an imaging-model comparison on a segmentation metric, which the codebook lists under does_not_count.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists
- **positional distribution**: `no` — Table 2 tabulates the 'Fracture segment' by vertebral level (L1 57/16, L2 53/12, L3 45/7, L4 29/3, L5 16/2) and the text says "Fracture segments were primarily distributed in the upper lumbar region". Per the codebook this is an ANATOMICAL statement (vertebral level), not the position of the label within the acquired slice stack, so it is coded 'no'.
- **headline**: AUC = 0.918 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); 3D ResNet50 fracture-identification module (the classification arm; the segmentation DSC of 0.934 is ignored per rule A3)
- **cohort**: dataset `private_multi_centre` (private), modality `CT` (mixed=false), region `spine`; n_patients=240, n_patients_test=—, n_slices_or_images=1018, n_positive_reported=`patients_only`
- **other**: interval `ci_unspecified_method`; input `patch_3D`; label_broadcast_to_slices `false`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: Multi-task paper included under rule A3; only the classification arm is coded. TWO JUDGEMENT CALLS. (1) evaluation_unit coded 'other' (per-vertebra); 'lesion' is a defensible alternative. (2) split_unit coded 'patient_subject' on the strength of 'cases' = patients (Table 2 is per-case demographics); a stricter screener could code 'random_unit_not_stated' because the sentence does not use the word patient. Internal inconsistency: 240 cases in Table 2 but '216 L-OVCF patients' for the fracture model, and the test confusion matrix has n = 204 vertebrae.

#### pos 5 — PMID 36789248 — EXCLUDED (E-DERIV)

*Grayscale Image Statistical Attributes Effectively Distinguish the Severity of Lung Abnormalities in CT Scan Slices of COVID-19 Patients.* — SN Comput Sci 2023. <https://pubmed.ncbi.nlm.nih.gov/36789248/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **exclusion `E-DERIV`** — evidence: "Values of 12 of the 13 statistics derived for each image, omitting the number of pixels in each image, are used as the input variables in this study." (Methods, 'Machine and Deep Learning Algorithms Applied to Grayscale Statistics'). Also "The total of 513 data records (one for each extract image with twelve grayscale statistics and a VS class) are assessed using multiple ML/DL algorithms". The 'CNN' is a 1D convolutional network over the 12 scalars, so no spatially resolved image reaches any classifier.
- **evaluation unit**: `slice` — "Five hundred and thirteen quadrilateral CT-image-slice extracts were evaluated in total" - recorded for completeness only; the record is excluded.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `slice_or_image` (disjointness `not_stated`) — "a split of 80% training subset: 20% testing subset worked well for the dataset evaluated and this division was randomly applied for this study" - the units are the 513 image extracts (average 8-15 per person), so extracts from one patient appear in both subsets.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists
- **positional distribution**: `no` — The 'position' hits concern where in the lung an extract was cropped ("the image extract position, and any image cropping conducted, has minimal impact on the grayscale statistics"), not the distribution of the label along the slice axis.
- **headline**: accuracy = 0.96 (cross_validation, scope `single_modality_arm`, rule `abstract_sentence`); 5-class visual score; excluded record, value recorded for completeness
- **cohort**: dataset `private_single_centre` (private), modality `CT` (mixed=false), region `lung`; n_patients=57, n_patients_test=—, n_slices_or_images=513, n_positive_reported=`slices_only`
- **other**: interval `sd_across_folds`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Clean E-DERIV: the image is reduced to 12 grayscale summary statistics before any classifier sees it, exactly the case pilot amendment A5 was written for. Note this paper would also have been an interesting slice-level record (513 extracts from 57 people, split at extract level) had it been eligible.

#### pos 6 — PMID 40335658 — EXCLUDED (E-NOCLF)

*Radiological evaluation and clinical implications of deep learning- and MRI-based synthetic CT for the assessment of cervical spine injuries.* — Eur Radiol 2025. <https://pubmed.ncbi.nlm.nih.gov/40335658/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-NOCLF`** — evidence: "A panel of five clinicians independently reviewed the images for diagnostic accuracy, lesion characterization (AO Spine classification), and soft tissue trauma." (Abstract, Methods). The deep-learning component generates synthetic CT images; the categorical decision (fracture present/absent, AO Spine class) is made by human readers, so no supervised classifier is fitted and criterion I2 fails.
- **evaluation unit**: `patient` — "Thirty-seven patients (44 cervical spine fractures) were enrolled." - reader-level agreement statistics, not model classification.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `no_held_out_test_set` (disjointness `na`) — No model train/test split is described in the abstract; the study is a prospective reader comparison of sCT against CT.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1; no zero-image baseline is claimed in the abstract.
  - searches_run: Not run - excluded at stage 1 (codebook requires the search only before coding trivial_baseline as absent in an INCLUDED paper).
- **positional distribution**: `no` — Nothing in the abstract refers to label position within the acquired stack.
- **headline**: sensitivity = 0.973 (unclear, scope `unclear`, rule `abstract_sentence`); reader sensitivity for visualizing fractures on synthetic CT - a reader metric, not a classifier metric
- **cohort**: dataset `private_multi_centre` (private), modality `MRI` (mixed=true), region `spine`; n_patients=37, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: JUDGEMENT CALL on the exclusion code. E-SEG is checked first and lists 'synthesis', but it is qualified 'with NO categorical class decision evaluated' - and a categorical decision (AO Spine class) WAS evaluated, just by humans. So the first code that actually applies is E-NOCLF, whose text is an exact description of this study ('purely descriptive or reader study with no model'). Coded identically to PMID 37962500 (batch B, GAN-based CTA) for consistency. ACCESS NOTE: Unpaywall lists a green-OA copy at https://dspace.library.uu.nl/handle/1874/466035 but the repository returned HTTP 403 from this environment, so the exclusion rests on the abstract alone; a second screener with access should confirm.

#### pos 7 — PMID 40194851 — EXCLUDED (E-DERIV)

*Severity Classification of Pediatric Spinal Cord Injuries Using Structural MRI Measures and Deep Learning: A Comprehensive Analysis across All Vertebral Levels.* — AJNR Am J Neuroradiol 2025. <https://pubmed.ncbi.nlm.nih.gov/40194851/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-DERIV`** — evidence: "Deep convolutional neural networks (CNNs) were utilized to classify participants into SCI or TD groups and determine their AIS classification based on structural parameters and demographic factors such as age and height." (Abstract, Materials and Methods). The classifier input is a table of cord measurements (CSA, AP width, RL width per vertebral level) plus age and height - the image is discarded.
- **evaluation unit**: `patient` — "Sixty-one pediatric participants (ages 6-18), including 20 with chronic SCI and 41 TD, were enrolled"; accuracy is reported per participant.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `unclear` (disjointness `not_stated`) — No split is described in the abstract.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1.
  - searches_run: Not run - excluded at stage 1.
- **positional distribution**: `no` — The abstract describes measurements 'across all vertebral levels', which is anatomical level, not the position of a label within the acquired slice stack.
- **headline**: accuracy = 0.9659 (unclear, scope `unclear`, rule `abstract_sentence`); SCI vs typically developing; AIS category classification 94.92%
- **cohort**: dataset `private_single_centre` (private), modality `MRI` (mixed=false), region `spine`; n_patients=61, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=false
- **notes**: The PMC deposit (PMC12633662) contains the abstract only - 2.6 kB of text, no Methods. The exclusion does not depend on the full text: the abstract states in terms that the CNN classified 'based on structural parameters and demographic factors', i.e. a shape/volumetry descriptor table, which is the E-DERIV definition verbatim.

#### pos 8 — PMID 42489954 — UNREACHABLE_ELIGIBILITY_UNRESOLVED

*Multi-Scale Structural MRI Features Reveal Task-Based Functional Connectivity and Its Alterations in Psychiatric Disorders: A Collaborative Graph Attention Network Approach.* — Brain Topogr 2026. <https://pubmed.ncbi.nlm.nih.gov/42489954/>

- **full text**: `unreachable_paywalled` / version used `abstract_only`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `patient` — "Validated on the Consortium for Neuropsychiatric Phenomics dataset (152 participants, three tasks, four diagnostic groups)" - macro F1 is reported per participant.
- **headline unit**: `na_only_one_unit_reported` — Only participant-level macro F1 is given.
- **split unit**: `unclear` (disjointness `not_stated`) — No split is described in the abstract.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=—, positional=—, acquisition_metadata=—, permuted_or_shuffled_label=—, clinical_or_demographic_only=—, other_non_imaging=—
  - evidence: NOT DETERMINABLE - full text unreachable. The abstract says the model "outperforms single-scale baselines", but those are ablations of the authors' own imaging model, which the codebook lists under does_not_count; no pixel-free comparator is named.
  - searches_run: REQUIRED SEARCHES NOT RUN - full text unreachable (see screen_protocol.md sec.7 access ladder: PMC/publisher OA no; publisher site HTTP 403/paywall; no institutional subscription held by this screener; OpenAlex/Unpaywall/Crossref/arXiv searched for a repository, accepted-manuscript or preprint copy, none found; interlibrary loan / author request not initiated). The primary-endpoint flags are therefore recorded as NULL, not FALSE: an unevidenced negative on P1 is not accepted by the codebook.
- **positional distribution**: `unclear` — Not determinable from the abstract.
- **headline**: F1 = 0.68 (unclear, scope `unclear`, rule `abstract_sentence`); "macro F1 scores of 0.68-0.75 across three psychiatric disorders"; the lower bound of the stated range is recorded, the abstract gives no single value
- **cohort**: dataset `Consortium for Neuropsychiatric Phenomics (CNP/UCLA)` (public), modality `MRI` (mixed=false), region `brain`; n_patients=152, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `unclear`; code `none`; chance_asserted_without_measurement=false
- **confidence** `low`; flag_for_adjudication=true
- **notes**: UNREACHABLE and deliberately NOT excluded at stage 1, although the abstract points strongly towards E-DERIV: "a dual-branch graph attention network to extract complementary global statistical and local topological features from structural MRI" describes a graph over derived descriptors rather than voxels. The codebook permits a stage-1 exclusion only when the code is UNAMBIGUOUS from the abstract, and this one is not (a voxel-consuming branch cannot be ruled out), so the record goes to the unreachable stratum. A screener with access should re-code it; the expected code is E-DERIV.

#### pos 9 — PMID 39061744 — INCLUDED

*Identification of Calculous Pyonephrosis by CT-Based Radiomics and Deep Learning.* — Bioengineering (Basel) 2024. <https://pubmed.ncbi.nlm.nih.gov/39061744/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `patient` — "A total of 53 patients with pyonephrosis and 129 patients with hydronephrosis were enrolled and all patients were randomly assigned to the training cohort or the testing cohort in a ratio of 7:3 (123:59)." (Results 3.1); all AUCs in Tables 3-4 are per patient.
- **headline unit**: `na_only_one_unit_reported` — Only patient-level AUC is reported anywhere.
- **split unit**: `patient_subject` (disjointness `stated_only`) — "These participants were randomly divided into two independent cohorts: training cohort (n = 123) and testing cohort (n = 59), based on a 7:3 ratio." (Methods 2.1, Patient Selection)
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=true, other_non_imaging=false
  - evidence: clinical_or_demographic_only = TRUE: "The clinical model based on the three clinical risk factors above exhibited an AUC of 0.904 (95% CI 0.837-0.950) ... In the testing cohort, it yielded an AUC of 0.889 (95% CI 0.781-0.956)" (Results 3.1). The three factors are fever, blood neutrophils and urine leukocytes - no pixels. This counts toward S1 only, NOT toward P1. All four P1 sub-flags are FALSE; Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists || SEARCH NOTE: The 17 'clinical model' hits are all this arm. The 'HU' arm (AUC 0.578) is NOT a zero-image baseline: the Hounsfield value is measured from the pixels inside the ROI.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists
- **positional distribution**: `no` — 'location' appears only as a stone characteristic in the clinical-variable list ("stone characteristics (laterality, location, and size)") - anatomical, not slice-axis.
- **headline**: AUC = 0.967 (internal_held_out, scope `single_modality_arm`, rule `abstract_multiple_took_external`); clinical machine-learning model (Rad-score + 3 clinical factors), testing cohort; the pure 3D-CNN arm reached only 0.599 in the same cohort
- **cohort**: dataset `private_single_centre` (private), modality `CT` (mixed=false), region `kidney`; n_patients=182, n_patients_test=59, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `ci_unspecified_method`; input `patch_3D`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Included because a genuine 3D-CNN consumes the image ("The final input size of images and masks into the CNN were all 128 x 128 x 128"). headline_selection_rule 'abstract_multiple_took_external' fired in its most-held-out sense: the abstract gives several numbers and the authors' final proposed model on the testing cohort was taken; there is no external cohort in this study. Worth noting for the paper's narrative: the clinical-variables-only model (0.889) beat the pure imaging CNN (0.599) on the same test set.

#### pos 10 — PMID 31093705 — INCLUDED

*Deep learning for liver tumor diagnosis part II: convolutional neural network interpretation using radiologic imaging features.* — Eur Radiol 2019. <https://pubmed.ncbi.nlm.nih.gov/31093705/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `lesion` — "A test set of 60 lesions was labeled with the most prominent imaging features in each image (1-4 features per lesion)." (Methods, Radiological feature selection); "the model obtained a PPV of 76.5 +/- 2.2% and Sn of 82.9 +/- 2.6% in identifying the 1-4 correct radiological features for the 60 manually labeled test lesions over 20 iterations".
- **headline unit**: `na_only_one_unit_reported` — All metrics are per lesion (or per lesion-feature); no patient-level metric appears in this paper.
- **split unit**: `unclear` (disjointness `not_stated`) — "The specific methods for patient selection, lesion reference standard, MRI technique, image processing techniques, and DL model are described in Part I of this study [5]." (Methods). This paper names only lesion counts (494 training lesions, 60 test lesions) and never states the unit at which train and test were separated, so the unit is coded unclear rather than upgraded.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists in the PMC record (a 'Supplementary Material' heading is present but empty) || SEARCH NOTE: 'baseline', 'chance', 'majority', 'prevalence', 'trivial', 'metadata', 'clinical-only', 'clinical model', 'position', 'slice index' and 'permut' return zero hits; 'random' refers only to random selection of sample lesions; 'constant' refers to enhancement patterns being constant across phases.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists in the PMC record
- **positional distribution**: `no` — 'location' hits all refer to where a feature is localised inside a lesion's feature map ("well localized and consistent with their location in the original image"), not to the distribution of labels along the slice axis.
- **headline**: other = 0.765 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); positive predictive value for identifying the correct radiological features per test lesion, averaged over 20 iterations (sensitivity 82.9%); the underlying 6-class lesion classifier is reported in Part I
- **cohort**: dataset `private_single_centre` (private), modality `MRI` (mixed=false), region `liver`; n_patients=296, n_patients_test=—, n_slices_or_images=494, n_positive_reported=`na`
- **other**: interval `sd_across_folds`; input `patch_3D`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: JUDGEMENT CALL on inclusion. This is Part II of a two-part report; the classifier itself (six hepatic tumour entities on multi-phasic MRI, 24 x 24 x 12 volumes) is fitted and its errors are reported here ("The model misclassified 12% of lesions"; "The lesion classifier performed better on both cysts (Sn = 99.5%, Sp = 99.9%) ... relative to HCCs (Sn = 82.0%, Sp = 96.5%)"), and PPV/sensitivity are in the I4 metric list, so I1-I4 are met. A screener who treated Part II as a methods-only companion could argue E-TYPE; I do not, because an original experiment with its own test set is reported. Because the split is defined in Part I, split_unit is coded 'unclear' rather than inferred.

#### pos 11 — PMID 36016875 — INCLUDED

*An in-depth discussion of cholesteatoma, middle ear Inflammation, and langerhans cell histiocytosis of the temporal bone, based on diagnostic results.* — Front Pediatr 2022. <https://pubmed.ncbi.nlm.nih.gov/36016875/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `slice` — "A random selection of 85% of the dataset (n = 2,070) was used during the validation process ... The remaining 15% of the data (n = 388) were stored and could be used to evaluate the performance of the model after the training was complete." (Methods, Classification task). The dataset units are individual axial CT sections: "the number of axial CT sections per scan ranged from 30 to 50. The total number of scans performed was 2,588", each "saved in the prescribed 224 x 224-pixel jpg format".
- **headline unit**: `na_only_one_unit_reported` — Every reported number (Table 1 confusion matrix, Table 2 precision/recall/F1/accuracy, the ROC AUCs) is computed over images. No patient-level metric is reported anywhere.
- **split unit**: `slice_or_image` (disjointness `not_stated`) — "A random selection of 85% of the dataset (n = 2,070) was used during the validation process ... The remaining 15% of the data (n = 388) were stored" - 2,070 + 388 = 2,458 image units, not the 119 patients. Sections from the same patient can therefore fall on both sides of the split.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists || SEARCH NOTE: 'baseline', 'chance', 'majority', 'prevalence', 'constant', 'trivial', 'metadata', 'clinical-only', 'clinical model', 'slice index' and 'permut' return zero hits; the single 'random' hit is the 85/15 split sentence; 'position'/'location' refer to where cholesteatoma arises anatomically ("the location of the tympanic sinus").
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists
- **positional distribution**: `no` — "Cholesteatoma is generally manifested in the location of the tympanic sinus ... These two entities may appear in different positions of the tympanic chamber" (Introduction) - anatomical, not position within the acquired stack.
- **headline**: AUC = 0.98 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); one-vs-rest AUC for cholesteatoma, the first value in the abstract's results sentence (LCH 0.99, MEI 0.99); the Results text also states an overall network AUC of 0.99
- **cohort**: dataset `private_single_centre` (private), modality `CT` (mixed=false), region `head_neck`; n_patients=119, n_patients_test=—, n_slices_or_images=2588, n_positive_reported=`patients_only`
- **other**: interval `none`; input `2D_slice`; label_broadcast_to_slices `true`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: A textbook slice-level record: patient-level histopathology labels broadcast to every axial section ("The distribution of clinical labels for each ear was first arranged according to each patient's diagnostic record"), an 85/15 split over images, and no patient-level metric. uncertainty_interval_reported is coded 'none' even though the Methods claim "a 95% CI was estimated using the Tak Long Estate Algorithm" (i.e. DeLong) - no interval value appears in any result. Internal inconsistency flagged: the test set is stated as n = 388 but the Table 1 confusion matrix contains 573 images.

#### pos 12 — PMID 36072854 — EXCLUDED (E-SEG)

*COVID-19 CT image segmentation method based on swin transformer.* — Front Physiol 2022. <https://pubmed.ncbi.nlm.nih.gov/36072854/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `exclude`
- **exclusion `E-SEG`** — evidence: "To analyze the segmentation performance of the trained model, we used three common performance metrics: mean intersection over union (mIoU), DSC, and mean pixel accuracy (mPA)." (Methods 2.5.3, Evaluation indicators). No categorical class decision on an imaging unit is evaluated; 'mean pixel accuracy' is a per-voxel overlap measure.
- **evaluation unit**: `other` — Voxel/pixel-level segmentation of four regions (background, lung, ground-glass opacity, lung parenchyma).
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `slice_or_image` (disjointness `not_stated`) — "The training set of the CC-CCII dataset is divided into ten groups, each time nine groups of images are used as the training set and one group is used as the validation set."
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded.
  - searches_run: Not run - excluded (segmentation-only paper).
- **positional distribution**: `no` — No statement about label position within the stack.
- **headline**: other = 0.8827 (cross_validation, scope `single_modality_arm`, rule `abstract_sentence`); Dice similarity coefficient (segmentation) - recorded for completeness only
- **cohort**: dataset `CC-CCII` (public), modality `CT` (mixed=false), region `lung`; n_patients=150, n_patients_test=—, n_slices_or_images=750, n_positive_reported=`slices_only`
- **other**: interval `none`; input `2D_slice`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Unambiguous E-SEG; confirmed against the full text rather than the abstract alone.

#### pos 13 — PMID 37222638 — UNREACHABLE_ELIGIBILITY_UNRESOLVED

*Prenatal Diagnosis of Placenta Accreta Spectrum Disorders: Deep Learning Radiomics of Pelvic MRI.* — J Magn Reson Imaging 2024. <https://pubmed.ncbi.nlm.nih.gov/37222638/>

- **full text**: `unreachable_paywalled` / version used `abstract_only`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `patient` — "324 pregnant women (mean age, 33.3 years) suspected PAS (170 training and 72 validation from institution 1, 82 external validation from institution 2) with clinicopathologically proved PAS (206 PAS, 118 non-PAS)" (Abstract, Population).
- **headline unit**: `na_only_one_unit_reported` — Only patient-level AUCs are given.
- **split unit**: `site_or_centre` (disjointness `stated_only`) — "170 training and 72 validation from institution 1, 82 external validation from institution 2" (Abstract, Population). The internal 170/72 split unit is not named.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=—, positional=—, acquisition_metadata=—, permuted_or_shuffled_label=—, clinical_or_demographic_only=true, other_non_imaging=—
  - evidence: clinical_or_demographic_only = TRUE on abstract evidence: "The MRI-based DLR model had a higher area under the curve than the clinical model in three datasets (0.880 vs. 0.741, 0.861 vs. 0.772, 0.852 vs. 0.675, respectively)" (Abstract, Results), where the clinical model is defined as "clinical model (different clinical characteristics between PAS and non-PAS groups)". Three measured AUCs for a pixel-free clinical-variables arm. The four P1 sub-flags remain NULL because the required full-text searches could not be run. NOTE: the 'MRI morphologic model' (0.760/0.781) is NOT pixel-free - it is the radiologists' binary read of the MRI.
  - searches_run: REQUIRED SEARCHES NOT RUN - full text unreachable (see screen_protocol.md sec.7 access ladder: PMC/publisher OA no; publisher site HTTP 403/paywall; no institutional subscription held by this screener; OpenAlex/Unpaywall/Crossref/arXiv searched for a repository, accepted-manuscript or preprint copy, none found; interlibrary loan / author request not initiated). The primary-endpoint flags are therefore recorded as NULL, not FALSE: an unevidenced negative on P1 is not accepted by the codebook.
- **positional distribution**: `unclear` — Not determinable from the abstract.
- **headline**: AUC = 0.852 (external, scope `single_modality_arm`, rule `abstract_multiple_took_external`); MRI-based deep-learning-radiomics model on the external validation dataset (institution 2); internal 0.880 / 0.861
- **cohort**: dataset `private_multi_centre` (private), modality `MRI` (mixed=false), region `other`; n_patients=324, n_patients_test=82, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `unclear`; input `unclear`; label_broadcast_to_slices `unclear`; code `none`; chance_asserted_without_measurement=false
- **confidence** `low`; flag_for_adjudication=true
- **notes**: UNREACHABLE (Wiley returned HTTP 403 to every route including the pdfdirect URL that Unpaywall and OpenAlex both list as open access - recorded as a discrepancy between automated OA metadata and actual reachability, which is exactly the limitation stated in screen_protocol.md sec.10). Eligibility is coded unresolved per sec.7 even though the abstract looks eligible. The clinical-only flag is set TRUE because the numbers are visible in the abstract; it feeds S1, not P1.

#### pos 14 — PMID 40239684 — EXCLUDED (E-SEG)

*DCA-U-Net: a deep learning network for segmentation of laser-induced thermal damage regions in mouse skin OCT images.* — Biomed Phys Eng Express 2025. <https://pubmed.ncbi.nlm.nih.gov/40239684/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-SEG`** — evidence: "we propose an efficient and lightweight segmentation model, Dilated ConvNeXT Attention U-Net (DCA-U-Net) ... Experimental results on two different sections of mouse skin laser thermal damage Optical Coherence Tomography (OCT) datasets show that our model has better segmentation performance" (Abstract). Segmentation only; no categorical class decision is evaluated.
- **evaluation unit**: `other` — Pixel-level segmentation of damage regions.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `unclear` (disjointness `not_stated`) — Not described in the abstract.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1.
  - searches_run: Not run - excluded at stage 1.
- **positional distribution**: `no` — Nothing in the abstract refers to label position within the stack.
- **headline**: other = — (unclear, scope `unclear`, rule `other`); no numeric classification metric is reported in the abstract
- **cohort**: dataset `private_single_centre` (private), modality `OCT` (mixed=false), region `other`; n_patients=—, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`na`
- **other**: interval `none`; input `2D_slice`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Two exclusion codes apply and E-SEG comes first in the codebook's fixed order; E-NONMED (mouse skin, preclinical animal-only) would also have applied. Recorded here so a second screener sees the same reasoning.

#### pos 15 — PMID 41068276 — INCLUDED

*IoMT driven Alzheimer's prediction model empowered with transfer learning and explainable AI approach in healthcare 5.0.* — Sci Rep 2025. <https://pubmed.ncbi.nlm.nih.gov/41068276/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `unclear` — The paper uses 'MRI scans' and 'MRI images' interchangeably and never says whether one unit is a slice or a volume: "The dataset used in this study is the publicly available Kaggle Augmented Alzheimer MRI Dataset (33,984 MRI scans across four classes)" vs Table 2 "Distribution of MRI images by dataset split (70/15/15, before augmentation)". Per the codebook's ambiguous-case rule this is coded 'unclear' - a substantive finding, not a coding failure.
- **headline unit**: `na_only_one_unit_reported` — One unit only; no patient-level metric is reported and no patient count exists in the paper.
- **split unit**: `slice_or_image` (disjointness `not_stated`) — Table 2 caption: "Distribution of MRI images by dataset split (70/15/15, before augmentation)", with per-class image counts (Mild Demented 8960 -> 6272/1344/1344 etc., Total 33,984 -> 23,789/5097/5098). The split unit is the image; the paper contains no subject identifier at all.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: 2 mentions of supplementary material in text but no supplementary instance files are attached to the PMC record; nothing further to search || SEARCH NOTE: The two 'baseline' hits are the ablation row "Baseline ResNet152 (no augmentation, no XAI) 92.1" (Table 4) - an ablation of the authors' own imaging network, which the codebook lists under does_not_count. The single 'chance' hit is the rhetorical "a scant chance of regaining health" in the Introduction.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: no supplementary files attached to the PMC record
- **positional distribution**: `no` — The only 'position' hit is "These results position the framework as a practical candidate" (Abstract).
- **headline**: accuracy = 0.9777 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); 4-class (Non-Demented / Very Mild / Mild / Moderate Demented), ResNet152-TL-XAI
- **cohort**: dataset `Kaggle Augmented Alzheimer MRI Dataset` (public), modality `MRI` (mixed=false), region `brain`; n_patients=—, n_patients_test=—, n_slices_or_images=33984, n_positive_reported=`slices_only`
- **other**: interval `none`; input `2D_slice`; label_broadcast_to_slices `unclear`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: n_patients is NULL because the paper never states one - the finding itself. input_representation is coded '2D_slice' on the strength of "unlike 3D-CNN or DenseNet models ..., it avoids heavy volumetric preprocessing" plus the ImageNet-initialised 2D ResNet152, while evaluation_unit stays 'unclear' because the paper never defines its unit; this deliberate mismatch is what the codebook's ambiguous-case rule produces. The Kaggle set is pre-augmented, so near-duplicate images of the same subject can straddle the 70/15/15 image split; the authors' own leakage safeguard covers only the WGAN-GP synthetic images.

### 4.2 Batch B (positions 38–58)

#### pos 38 — PMID 34229143 — EXCLUDED (E-DERIV)

*The detection of mild traumatic brain injury in paediatrics using artificial neural networks.* — Comput Biol Med 2021. <https://pubmed.ncbi.nlm.nih.gov/34229143/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-DERIV`** — evidence: "RF ranked ten clinical demographic features and twelve CT-findings; the hybrid RF-ANN model achieved ..." and "This is the first study to investigate deep ANN in a paediatric cohort with mTBI using clinical and non-imaging data" (Abstract). The classifier inputs are tabulated clinical variables and coded radiologist CT findings from the PECARN dataset; no image reaches the network.
- **evaluation unit**: `patient` — "The models were conducted using 15,271 patients under the age of 18 years with mTBI and had a head CT report."
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `random_unit_not_stated` (disjointness `not_stated`) — "The dataset was divided into two subsets: 80% for training and 20% for testing using five-fold cross-validation." - the unit is not named.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1.
  - searches_run: Not run - excluded at stage 1.
- **positional distribution**: `no` — Nothing in the abstract refers to position within an image stack.
- **headline**: accuracy = 0.999 (cross_validation, scope `unclear`, rule `abstract_sentence`); deep ANN; recorded for completeness only
- **cohort**: dataset `PECARN` (public), modality `CT` (mixed=false), region `brain`; n_patients=15271, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`neither`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: The paper says so itself ('using clinical and non-imaging data'), so the E-DERIV code is unambiguous from the abstract and no full text was sought.

#### pos 39 — PMID 39107903 — UNREACHABLE_ELIGIBILITY_UNRESOLVED

*Automated detection of maxillary sinus opacifications compatible with sinusitis from CT images.* — Dentomaxillofac Radiol 2024. <https://pubmed.ncbi.nlm.nih.gov/39107903/>

- **full text**: `unreachable_paywalled` / version used `abstract_only`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `lesion` — "Of the 1080 randomly selected coronal-view CT images, including 2158 maxillary sinuses, datasets of maxillary sinus lesions comprised 1138 normal sinuses, 366 cysts, and 654 sinusitis based on radiographic findings" (Abstract, Methods). The scored unit is the individual sinus within a coronal CT image.
- **headline unit**: `na_only_one_unit_reported` — Only per-sinus precision/F1 is reported in the abstract.
- **split unit**: `slice_or_image` (disjointness `not_stated`) — "Of the 1080 randomly selected coronal-view CT images ... [datasets] were divided into training (n = 648 CT images), validation (n = 216), and test (n = 216) sets." (Abstract). The split unit is the CT image; no patient count appears in the abstract at all.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=—, positional=—, acquisition_metadata=—, permuted_or_shuffled_label=—, clinical_or_demographic_only=—, other_non_imaging=—
  - evidence: NOT DETERMINABLE - full text unreachable.
  - searches_run: REQUIRED SEARCHES NOT RUN - full text unreachable (see screen_protocol.md sec.7 access ladder: PMC/publisher OA no; publisher site HTTP 403/paywall; no institutional subscription held by this screener; OpenAlex/Unpaywall/Crossref/arXiv searched for a repository, accepted-manuscript or preprint copy, none found; interlibrary loan / author request not initiated). The primary-endpoint flags are therefore recorded as NULL, not FALSE: an unevidenced negative on P1 is not accepted by the codebook.
- **positional distribution**: `unclear` — Not determinable from the abstract.
- **headline**: other = 0.971 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); overall precision of the YOLOv8-nano detector on the test set (per-class: normal 96.9%, cyst 95.2%, sinusitis 99.2%; average F1 95.4%)
- **cohort**: dataset `private_single_centre` (private), modality `CT` (mixed=false), region `head_neck`; n_patients=—, n_patients_test=—, n_slices_or_images=1080, n_positive_reported=`slices_only`
- **other**: interval `none`; input `2D_slice`; label_broadcast_to_slices `unclear`; code `none`; chance_asserted_without_measurement=false
- **confidence** `low`; flag_for_adjudication=true
- **notes**: UNREACHABLE. Deliberately NOT excluded as E-SEG: although the model is a YOLO object detector, a genuine 3-way categorical decision (normal / cyst / sinusitis) is evaluated with per-class precision and F1, so the E-SEG qualifier 'with NO categorical class decision evaluated' is not met. A split at the level of the coronal CT image with no patient count anywhere in the abstract would make this a highly relevant record if it could be reached; a screener with Oxford access should re-code it.

#### pos 40 — PMID 42462969 — EXCLUDED (E-TYPE)

*Diagnostic performance of deep learning for automated mandibular canal segmentation on CBCT images: A systematic review and meta-analysis.* — J Stomatol Oral Maxillofac Surg 2026. <https://pubmed.ncbi.nlm.nih.gov/42462969/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-TYPE`** — evidence: "This systematic review and meta-analysis aimed to evaluate the diagnostic performance ... A comprehensive literature search was conducted across PubMed, Scopus, Web of Science, IEEE Xplore, and Embase databases in accordance with PRISMA guidelines ... A total of 38 unique studies comprising over 8420 CBCT volumes were included in the quantitative synthesis." (Abstract). A meta-analysis re-tabulating other people's numbers; no original experiment.
- **evaluation unit**: `other` — Pooled Dice across 38 primary studies.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `no_held_out_test_set` (disjointness `na`) — No model is fitted.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1.
  - searches_run: Not run - excluded at stage 1.
- **positional distribution**: `no` — Not applicable.
- **headline**: other = 0.82 (unclear, scope `pooled_multi_dataset`, rule `abstract_sentence`); pooled Dice similarity coefficient (segmentation); recorded for completeness only
- **cohort**: dataset `na` (unclear), modality `CBCT` (mixed=false), region `head_neck`; n_patients=—, n_patients_test=—, n_slices_or_images=8420, n_positive_reported=`na`
- **other**: interval `ci_unspecified_method`; input `3D_volume`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: The frame's negative-only publication-type filter cannot drop an untyped record, and PubMed types this one as 'Journal Article' only - so it reached the sample and is caught here by E-TYPE exactly as screen_protocol.md sec.2.1 anticipated. E-SEG would also apply to the underlying task.

#### pos 41 — PMID 37591161 — EXCLUDED (E-SEG)

*Segmentation quality assessment by automated detection of erroneous surface regions in medical images.* — Comput Biol Med 2023. <https://pubmed.ncbi.nlm.nih.gov/37591161/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **exclusion `E-SEG`** — evidence: "We develop a segmentation quality assessment (SQA) model using U-Net based architecture to predict the error mask E_pred, which marks the erroneous regions for the predicted segmentation I_seg on the input image I_in." (Methods 2.1) and "We evaluated our method based on two evaluation metrics: (1) Sensitivity, (2) Positive predictive value (PPV), (3) Detected rate of predicted erroneous surface." (Results, Evaluation metrics). Sensitivity/PPV are computed over surface patches, i.e. this is segmentation-quality assessment and error localisation - the exact case the codebook warns about (three of ten pilot papers reported sensitivity/specificity without being classification papers).
- **evaluation unit**: `other` — Erroneous surface patches on the boundary of a segmented object: "The sensitivity and PPV scores were calculated over the whole testing dataset for each object class j."
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `unclear` (disjointness `not_stated`) — Train/test partition is described per dataset (OAI knee MRI, calf muscle MRI) without naming the unit.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded.
  - searches_run: Not run for the primary endpoint - excluded record. (The term sweep was nevertheless executed and returned no baseline/chance/majority/prevalence/constant/trivial/metadata/clinical/position/slice-index/permutation hits; 'location' hits all concern error localisation.)
- **positional distribution**: `no` — 'location' refers to where segmentation errors occur on an object surface, not to the position of a label in the slice stack.
- **headline**: sensitivity = 0.95 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); erroneous-surface-region detection on knee cartilage (0.92 on calf muscle); recorded for completeness only
- **cohort**: dataset `Osteoarthritis Initiative (OAI) 3D knee MRI; 3D calf muscle MRI` (mixed), modality `MRI` (mixed=false), region `musculoskeletal`; n_patients=—, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`na`
- **other**: interval `none`; input `3D_volume`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Title says 'detection' and the metrics are sensitivity and PPV, but the evaluated task is voxel/surface error localisation for segmentation QA. Rule A4 applies: code the evaluated task.

#### pos 42 — PMID 40768653 — INCLUDED

*Predictive Modeling of Osteonecrosis of the Femoral Head Progression Using MobileNetV3_Large and Long Short-Term Memory Network: Novel Approach.* — JMIR Med Inform 2025. <https://pubmed.ncbi.nlm.nih.gov/40768653/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `slice` — "In total, 1200 slices were generated, including 675 slices with lesions and 225 normal slices. Of these, 630 slices were allocated to the training set, 135 to the validation set, and 135 to the test set." (Methods, Dataset Division)
- **headline unit**: `na_only_one_unit_reported` — No patient-level metric is reported anywhere; all accuracy/recall/AUC figures are over slices.
- **split unit**: `slice_or_image` (disjointness `not_stated`) — "The annotated image dataset was divided into training (70%), validation (15%), and test (15%) sets ... Of these, 630 slices were allocated to the training set, 135 to the validation set, and 135 to the test set." (Methods, Dataset Division). The 1200 slices come from only 30 patients (21 bilateral, 9 unilateral), so slices of the same hip of the same patient are on both sides of the split; the word 'patient' never appears in a split sentence.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: Multimedia Appendices 1-3 exist (Figures S1, S2, Tables S1-S3) but could NOT be retrieved - the PMC supplementary-file endpoint returned an HTML error stub from this environment. The negative is therefore evidenced over the main text only. || SEARCH NOTE: The two 'baseline' hits are "the patients' baseline characteristics" and "The baseline characteristics of all patients are summarized in Table S1"; 'chance', 'majority', 'prevalence', 'constant', 'trivial', 'metadata', 'clinical-only', 'clinical model', 'position', 'slice index', 'permut' and 'random' return zero hits.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: Multimedia Appendices 1-3 exist but could not be downloaded (PMC supplementary endpoint blocked); main text only
- **positional distribution**: `no` — The single 'location' hit is "the coarse segmentation network extracts the features at the corresponding locations" (PointRend description).
- **headline**: accuracy = 0.965 (unclear, scope `single_modality_arm`, rule `abstract_sentence`); MobileNetV3_Large in ONFH diagnosis, 95% CI 95.1-97.8%
- **cohort**: dataset `Open Biomedical Imaging Archive (OBIA)` (public), modality `MRI` (mixed=false), region `musculoskeletal`; n_patients=30, n_patients_test=—, n_slices_or_images=1200, n_positive_reported=`slices_only`
- **other**: interval `ci_unspecified_method`; input `2D_slice`; label_broadcast_to_slices `false`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: Slice-level throughout: 1200 slices from 30 patients, split 630/135/135 by slice, no patient-level metric. headline_test_set is coded 'unclear' because the abstract's 96.5% cannot be matched to any table: the Results text reports "Accuracy significantly improved to 91.3% (P<.05)" for the same model, and the ten compared architectures listed in the Methods differ from the ten listed in the Results. A 95% CI is quoted for an accuracy computed on 135 test slices, with no clustering by patient.

#### pos 43 — PMID 38337016 — INCLUDED

*Predicting hematoma expansion in acute spontaneous intracerebral hemorrhage: integrating clinical factors with a multitask deep learning model for non-contrast head CT.* — Neuroradiology 2024. <https://pubmed.ncbi.nlm.nih.gov/38337016/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `both` — SLICE: "For the task of hematoma slice classification, the model put forth an excellent performance with an accuracy of 97.7% ... the AUC of the receiver operating characteristic curve for this model stood high at 99.4%." PATIENT: "The Image-to-HE model predicts a patient-specific HE outcome using the normalized DL score (nDL score)", with Table 3 giving accuracy/sensitivity/specificity/AUC over 107 test patients. AUC is therefore reported at two units.
- **headline unit**: `slice` — Both units appear in the abstract and the codebook takes whichever is stated first: "For hematoma detection, the diagnostic performance of the developed multi-task model was excellent (AUC, 0.99). For expansion prediction, three models were evaluated ..." The first number is the slice-level detection AUC.
- **split unit**: `patient_subject` (disjointness `stated_only`) — "From a total of 24,238 slices from 572 patients, 6044 slices were identified from 569 patients as having hematoma. The training set comprised 4834 slices from 458 patients, while the test set consisted of 1210 slices from 111 patients." (Methods, Dataset)
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=true, other_non_imaging=false
  - evidence: clinical_or_demographic_only = TRUE: "The Clinical-to-HE model forecasts HE using multivariate logistic regression on clinical variables, notably without incorporating image data." (Methods, Deep learning algorithm), with a measured AUC on the same metric: Table 3, "Clinical-to-HE 74.8 (80/107) [65.8, 82.0] ... 0.81 [0.69, 0.93]". This counts toward S1 only. All four P1 sub-flags FALSE; Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: Supplementary Information exists (Supplementary Tables 1-2, one MOESM PDF) and could NOT be retrieved - the PMC supplementary endpoint returned an HTML error stub from this environment. Negative evidenced over the main text only. || SEARCH NOTE: 'baseline', 'chance', 'majority', 'prevalence', 'trivial', 'metadata', 'position', 'slice index' and 'permut' return zero hits in the main text; 'constant' refers to acquisition parameters being 'relatively constant'.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: Supplementary Information exists but could not be downloaded (PMC supplementary endpoint blocked); main text only
- **positional distribution**: `no` — A slice-classification model is used to SELECT hematoma-bearing slices ("the trained hematoma slice classification model was used to extract slices indicative of hematoma"), but no distribution of the label along the slice axis is shown or stated.
- **headline**: AUC = 0.99 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); slice-level hematoma detection (99.4% in the Results text); the paper's own primary task, patient-level hematoma-expansion prediction, reached AUC 0.83 [0.72, 0.95] for the integrated model and 0.81 [0.69, 0.93] for the clinical-variables-only model
- **cohort**: dataset `private_single_centre` (private), modality `CT` (mixed=false), region `brain`; n_patients=572, n_patients_test=111, n_slices_or_images=24238, n_positive_reported=`patients_and_slices`
- **other**: interval `ci_unspecified_method`; input `2D_slice`; label_broadcast_to_slices `true`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: JUDGEMENT CALL on headline_unit. Rule A5 makes the headline the first number in the abstract's results sentence, which is the slice-level detection AUC of 0.99, even though the paper's stated purpose is patient-level HE prediction. Coded by the rule and flagged. Substantively this is one of the most informative records in the batch: the split IS patient-level and stated, positive counts ARE given at both patient and slice level, and a pixel-free clinical model (AUC 0.81) beats the image-only model (AUC 0.76) - but no zero-image baseline in the P1 family (constant/positional/metadata/permuted) is reported.

#### pos 44 — PMID 33310694 — EXCLUDED (E-SEG)

*3D PBV-Net: An automated prostate MRI data segmentation method.* — Comput Biol Med 2021. <https://pubmed.ncbi.nlm.nih.gov/33310694/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-SEG`** — evidence: "This study proposes an automated prostate MRI data segmentation approach using bicubic interpolation with improved 3D V-Net (dubbed 3D PBV-Net) ... Our approach generates promising segmentation results, which have achieved 97.65% and 98.29% of average accuracy, 0.9613 and 0.9765 of Dice metric, 3.120 mm and 0.9382 mm of Hausdorff distance, and average boundary distance of 1.708, 0.7950 on PROMISE 12 and TPHOH datasets" (Abstract). 'Average accuracy' here is a voxel-overlap measure alongside Dice/Hausdorff/boundary distance; no categorical class decision on an imaging unit is evaluated.
- **evaluation unit**: `other` — Voxel-level segmentation of the prostate gland.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `external_cohort_only` (disjointness `not_stated`) — "we evaluate the proposed 3D PBV-Net on two clinical prostate MRI data datasets, i.e., PROMISE 12 and TPHOH" - not determinable further from the abstract.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1.
  - searches_run: Not run - excluded at stage 1.
- **positional distribution**: `no` — Nothing in the abstract refers to label position within the stack.
- **headline**: other = 0.9613 (unclear, scope `pooled_multi_dataset`, rule `abstract_sentence`); Dice on PROMISE12 (segmentation); recorded for completeness only
- **cohort**: dataset `PROMISE12; TPHOH` (mixed), modality `MRI` (mixed=false), region `prostate`; n_patients=—, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`na`
- **other**: interval `none`; input `3D_volume`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: A segmentation paper that reports 'accuracy' - the exact trap flagged in codebook section 4.1. Excluded on the evaluated task, not on the metric name.

#### pos 45 — PMID 36291282 — EXCLUDED (E-DERIV)

*Multi-Perspective Feature Extraction and Fusion Based on Deep Latent Space for Diagnosis of Alzheimer's Diseases.* — Brain Sci 2022. <https://pubmed.ncbi.nlm.nih.gov/36291282/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **exclusion `E-DERIV`** — evidence: "Construction of the dFC: To describe the dynamic changes in brain regions and construct the dFC, we divided the time series of ROIs obtained by rs-fMRI into M overlapping sliding windows" and "we use M dFCs from each subject as input to the global convolutional [network]" (Methods 3.1). The classifier input is a stack of Pearson functional-connectivity matrices; the image is discarded. This is the case pilot amendment A5 was created for (PMID 34924987, ABIDE connectivity matrices).
- **evaluation unit**: `patient` — "we used the rs-fMRI data of 174 subjects, the ADNI dataset, including 48 Normal Control (NC), 50 patients with early MCI (eMCI), 45 patients with late MCI (lMCI), and 31 patients with AD."
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `unclear` (disjointness `not_stated`) — "we randomly used the dFCs of the subjects in the training set as input to the autoencoder" - a training set is referred to but its construction and unit are never specified.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded. (The three 'baseline' hits are comparison architectures such as ROIs+SVM and sFC+SVM, which are image-derived models, not zero-image baselines.)
  - searches_run: Term sweep executed but not used for the primary endpoint - excluded record.
- **positional distribution**: `no` — No reference to position within an acquired stack.
- **headline**: accuracy = 0.951 (unclear, scope `unclear`, rule `abstract_sentence`); NC/AD binary task; recorded for completeness only
- **cohort**: dataset `ADNI` (public), modality `MRI` (mixed=false), region `brain`; n_patients=174, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Clean E-DERIV, confirmed against the full text.

#### pos 46 — PMID 40442294 — EXCLUDED (E-DERIV)

*Comparative analysis of natural language processing methodologies for classifying computed tomography enterography reports in Crohn's disease patients.* — NPJ Digit Med 2025. <https://pubmed.ncbi.nlm.nih.gov/40442294/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **exclusion `E-DERIV`** — evidence: "Here we evaluate natural language processing to classify Crohn's disease (CD) on CTE" and "2839 CTE reports of patients with CD and controls collected from all healthcare and diagnostic imaging facilities across Alberta ... were available" and "CTE reports were extracted and split into training- (n = 1568), development- (n = 196), and testing (n = 198) datasets each with around 200 words" (Abstract; Methods, report inclusion). The classifier input is the free-text radiology report; no image ever reaches the model, so criterion I3 fails.
- **evaluation unit**: `other` — One CTE report (~200 words) per scored unit.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `slice_or_image` (disjointness `not_stated`) — "split into training- (n = 1568), development- (n = 196), and testing (n = 198) datasets" - the unit is the report, not the patient.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded. (For information: the paper does report a rule-based classifier as a comparator, but that is a text model, not a zero-image baseline in this codebook's sense.)
  - searches_run: Term sweep executed but not used for the primary endpoint - excluded record.
- **positional distribution**: `no` — No reference to position within an acquired stack.
- **headline**: accuracy = 0.912 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); LLaMA-3.3-70B-Instruct; recorded for completeness only
- **cohort**: dataset `private_multi_centre (Alberta IBD registry)` (private), modality `CT` (mixed=false), region `abdomen_general`; n_patients=1962, n_patients_test=—, n_slices_or_images=2839, n_positive_reported=`neither`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: JUDGEMENT CALL on the code. A radiology report is not one of E-DERIV's enumerated inputs, but it is squarely 'a non-spatial derived representation with the image discarded', and E-DERIV precedes E-NONMED in the fixed order. A second screener could reasonably code E-NONMED ('non-imaging signal ... despite matching the query'). Either way the paper is outside the failure mode and inside the query, which is why the codebook asks for E-DERIV to be reported separately.

#### pos 47 — PMID 34655238 — EXCLUDED (E-DERIV)

*3D gray density coding feature for benign-malignant pulmonary nodule classification on chest CT.* — Med Phys 2021. <https://pubmed.ncbi.nlm.nih.gov/34655238/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-DERIV`** — evidence: "feature descriptor is obtained by coding the pulmonary nodule with codebook, and 3D GDC feature is the result of histogram statistics on feature descriptor. Second, geometric features are extracted for fusion feature. Finally, random forest is performed for benign-malignant pulmonary nodule classification with fusion feature of the 3D gray density coding feature and the geometric features." (Abstract, Methods). The classifier is a random forest whose input is a hand-crafted histogram plus geometric descriptors - a feature vector alone, with the image discarded.
- **evaluation unit**: `lesion` — "it contains a total of 238 lung nodules from 203 patients" (private ZSHD dataset); metrics are per nodule.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `unclear` (disjointness `not_stated`) — Not described in the abstract beyond '3D BD is balanced and randomly selecting from benign and malignant pulmonary nodules of training data'.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1.
  - searches_run: Not run - excluded at stage 1.
- **positional distribution**: `no` — Nothing in the abstract refers to slice position.
- **headline**: AUC = 0.9753 (cross_validation, scope `single_modality_arm`, rule `abstract_sentence`); LIDC-IDRI, mean +/- SD 97.53 +/- 1.62%; recorded for completeness only
- **cohort**: dataset `LIDC-IDRI; ZSHD (private)` (mixed), modality `CT` (mixed=false), region `lung`; n_patients=203, n_patients_test=—, n_slices_or_images=238, n_positive_reported=`neither`
- **other**: interval `sd_across_folds`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: JUDGEMENT CALL. The features are computed from 3D image blocks, so a screener could argue the image does reach the pipeline; but the CLASSIFIER's input (criterion I3) is a histogram/descriptor vector, which is E-DERIV's definition. Contrast PMIDs 39061744 and 42100397, both INCLUDED in this batch, where a CNN genuinely consumes voxels alongside the radiomics arm. Flagged for adjudication; full text unreachable so the code rests on the abstract, which is explicit about the random-forest input.

#### pos 48 — PMID 39699671 — UNREACHABLE_ELIGIBILITY_UNRESOLVED

*Evaluation of a deep learning prostate cancer detection system on biparametric MRI against radiological reading.* — Eur Radiol 2025. <https://pubmed.ncbi.nlm.nih.gov/39699671/>

- **full text**: `unreachable_paywalled` / version used `abstract_only`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `both` — "A 3D nnU-Net was trained on bpMRI for lesion detection, evaluated using histopathology-based annotations, and assessed with patient- and lesion-level metrics, along with lesion volume, and GGG." (Abstract, Materials and Methods). Patient-level AUC 0.83 and lesion-level sensitivity/average precision are both reported.
- **headline unit**: `patient` — "The model achieved an AUC of 0.83 (95% CI: 0.80, 0.87). Lesion-level sensitivity was 0.85 ..." - the patient-level AUC is stated first in the abstract's results.
- **split unit**: `external_cohort_only` (disjointness `stated_only`) — "The training dataset included 4381 bpMRI cases (3800 positive and 581 negative) across three continents ... The testing set comprised 328 cases from the PROSTATEx dataset" (Abstract).
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=—, positional=—, acquisition_metadata=—, permuted_or_shuffled_label=—, clinical_or_demographic_only=—, other_non_imaging=—
  - evidence: NOT DETERMINABLE - full text unreachable. The abstract's only comparator is a group of non-expert radiologists, which the codebook lists under does_not_count.
  - searches_run: REQUIRED SEARCHES NOT RUN - full text unreachable (see screen_protocol.md sec.7 access ladder: PMC/publisher OA no; publisher site HTTP 403/paywall; no institutional subscription held by this screener; OpenAlex/Unpaywall/Crossref/arXiv searched for a repository, accepted-manuscript or preprint copy, none found; interlibrary loan / author request not initiated). The primary-endpoint flags are therefore recorded as NULL, not FALSE: an unevidenced negative on P1 is not accepted by the codebook.
- **positional distribution**: `unclear` — Not determinable from the abstract.
- **headline**: AUC = 0.83 (external, scope `single_modality_arm`, rule `abstract_sentence`); patient-level csPCa detection on the PROSTATEx test set, 95% CI 0.80-0.87
- **cohort**: dataset `PI-CAI-style multi-cohort training set; PROSTATEx (test)` (mixed), modality `MRI` (mixed=false), region `prostate`; n_patients=4381, n_patients_test=328, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `ci_unspecified_method`; input `3D_volume`; label_broadcast_to_slices `unclear`; code `public_link_stated`; chance_asserted_without_measurement=false
- **confidence** `low`; flag_for_adjudication=true
- **notes**: UNREACHABLE (Springer paywall; no repository or preprint copy found via OpenAlex, Unpaywall, Crossref or arXiv). This is the record in the batch most directly comparable to our own prostate audit, so it is worth prioritising for interlibrary loan. code_availability is coded 'public_link_stated' on the abstract's "offers public PI-RADS annotations" claim, which is data rather than code - a weak code, flagged.

#### pos 49 — PMID 40001707 — EXCLUDED (E-SEG)

*A Comprehensive AI Framework for Superior Diagnosis, Cranial Reconstruction, and Implant Generation for Diverse Cranial Defects.* — Bioengineering (Basel) 2025. <https://pubmed.ncbi.nlm.nih.gov/40001707/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **exclusion `E-SEG`** — evidence: "A diverse set of performance metrics was utilized to comprehensively evaluate the model's accuracy and robustness in cranial reconstruction tasks. The metrics used were DSC, JSC, HD, precision, recall, and specificity." followed by "Precision: The proportion of true positives out of all positive predictions" and "DSC: Measures the overlap between predicted and ground truth regions" (Results 3.4). Every metric is a voxel-overlap measure for skull reconstruction / implant generation; no categorical class decision on an imaging unit is evaluated.
- **evaluation unit**: `other` — Voxelised skull volumes; "averaged over a testing dataset of 720 files".
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `unclear` (disjointness `not_stated`) — "trained on a diverse dataset of 2160 images, which was prepared by simulating cylindrical, cubical, spherical, and triangular prism-shaped defects across five skull regions" - defects are synthetic; the split unit is not stated.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded.
  - searches_run: Term sweep executed but not used for the primary endpoint - excluded record.
- **positional distribution**: `no` — Region-wise performance is reported by skull region (top, back, etc.) - anatomical, not slice-axis.
- **headline**: other = 0.9948 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); Dice similarity coefficient for cranial reconstruction; recorded for completeness only
- **cohort**: dataset `private/simulated (2160 images with synthetic defects)` (unclear), modality `CT` (mixed=false), region `head_neck`; n_patients=—, n_patients_test=—, n_slices_or_images=2160, n_positive_reported=`na`
- **other**: interval `none`; input `3D_volume`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Title promises 'Superior Diagnosis' but the evaluated task is reconstruction/implant generation (rule A4). Precision/recall/specificity are voxel-level.

#### pos 50 — PMID 32372385 — EXCLUDED (E-SEG)

*Detecting the occluding contours of the uterus to automatise augmented laparoscopy: score, loss, dataset, evaluation and user study.* — Int J Comput Assist Radiol Surg 2020. <https://pubmed.ncbi.nlm.nih.gov/32372385/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-SEG`** — evidence: "we propose a complete framework for object-class occluding contour detection (OC2D) ... Our first contribution is a new distance-based evaluation score ... Evaluation shows that the proposed detector has a similar false false-negative rate to existing methods but substantially reduces both false-positive rate and response thickness." (Abstract). Contour delineation evaluated with a distance-based score and pixel-thickness; no categorical class decision on an imaging unit.
- **evaluation unit**: `other` — Pixel-level contour responses on laparoscopy frames.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `slice_or_image` (disjointness `not_stated`) — "a dataset of 3818 carefully labelled laparoscopy images of the uterus, which was used to train and evaluate our detector".
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1.
  - searches_run: Not run - excluded at stage 1.
- **positional distribution**: `no` — Nothing in the abstract refers to slice position.
- **headline**: other = — (unclear, scope `unclear`, rule `other`); distance-based contour score; no numeric classification metric in the abstract
- **cohort**: dataset `private (3818 labelled laparoscopy images)` (private), modality `other` (mixed=true), region `other`; n_patients=—, n_patients_test=—, n_slices_or_images=3818, n_positive_reported=`na`
- **other**: interval `none`; input `2D_slice`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Two codes apply and E-SEG comes first in the fixed order; E-2D (laparoscopy/endoscopy) would also apply, since the classifier input is the 2D laparoscopic video frame and the MRI appears only as a preoperative model to be registered. An OA copy exists at https://hal.science/hal-02884670 but was not needed.

#### pos 51 — PMID 34926501 — INCLUDED

*Clinical Applicable AI System Based on Deep Learning Algorithm for Differentiation of Pulmonary Infectious Disease.* — Front Med (Lausanne) 2021. <https://pubmed.ncbi.nlm.nih.gov/34926501/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `both` — "The proposed bi-classifier achieved an average AUROC of 0.984 (95% CI, 0.983-0.985) on slice-level and 0.988 (95% CI, 0.977-0.997) on patient-level, respectively." (Results, Deep Learning-Based Pathogen Identification); Table 2 reports AUC/accuracy/sensitivity/specificity at both units.
- **headline unit**: `patient` — The abstract's results sentence - "The median AUC of DL models for differentiating pulmonary infection was 99.5% (COVID-19), 98.6% (viral pneumonia), 98.4% (bacterial pneumonia), 99.1% (fungal pneumonia)" - matches the PATIENT-level quad-classifier row of Table 2 exactly (0.995 / 0.986 / 0.984 / 0.991). The slice-level row (0.983 / 0.987 / 0.990 / 0.979) does not appear in the abstract.
- **split unit**: `random_unit_not_stated` (disjointness `not_stated`) — "The given data was split into three sets with an 8:1:1 ratio for training, validation, and testing" (Methods, Overview of the AI System). No unit is named. Per the codebook this is NOT upgraded to patient-level merely because patient-level metrics are reported elsewhere.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists || SEARCH NOTE: The single 'baseline' hit is "Two groups of doctors ... were asked to evaluate pneumonia cases solely on CT scans independently and blindly to establish a comparative baseline for our AI system" - a human-reader comparison, which the codebook lists under does_not_count. The ML arm combines DL-quantified CT features WITH clinical indicators, so it is not a clinical-variables-only model either; 'chance', 'prevalence', 'constant', 'trivial', 'metadata', 'clinical-only', 'clinical model', 'slice index' and 'permut' return zero hits.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists
- **positional distribution**: `no` — Table 3 gives "Location: distance from lesion to pulmonary pleurae | Lesion distance (mm) | 4.8 +/- 4.3 ..." and the text notes "lesion location (right upper lung or, 1.12; 95% CI: 1.03-1.25)". Both are ANATOMICAL locations, not the position of the label within the acquired slice stack, so the codebook's rule gives 'no'.
- **headline**: AUC = 0.995 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); one-vs-rest patient-level AUROC for SARS-CoV-2 in the four-pathogen quad-classifier (common virus 0.986, bacterial 0.984, fungal 0.991)
- **cohort**: dataset `private_multi_centre (three institutions)` (private), modality `CT` (mixed=false), region `lung`; n_patients=1431, n_patients_test=—, n_slices_or_images=3463, n_positive_reported=`slices_only`
- **other**: interval `ci_unspecified_method`; input `2D_slice`; label_broadcast_to_slices `true`; code `public_link_stated`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: One of only two included papers in this batch that report the SAME metric at both slice and patient level (the other is PMID 38337016) - directly relevant to endpoint S3. label_broadcast_to_slices is TRUE: "By majority voting, the final score of the CNN classifier's prediction for all abnormal CT slices was merged to generate a patient-level CT volume prediction." split_unit is the flagged case: patient-level results are reported but the split sentence names no unit, and the codebook forbids upgrading. Code at https://github.com/chiehchiu/CAAS (link stated, not verified by this screener). The relation between 1,431 patients and 3,463 'CT images' is never explained.

#### pos 52 — PMID 37276106 — UNREACHABLE_ELIGIBILITY_UNRESOLVED

*Automatic Evaluating of Multi-Phase Cranial CTA Collateral Circulation Based on Feature Fusion Attention Network Model.* — IEEE Trans Nanobioscience 2023. <https://pubmed.ncbi.nlm.nih.gov/37276106/>

- **full text**: `unreachable_paywalled` / version used `abstract_only`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `unclear` — "Tested on a dataset of multi-phase cranial CTA images, the accuracy rate exceeding 90.43%." The abstract never states whether a scored unit is a patient, a scan or a slice.
- **headline unit**: `unclear` — Not determinable from the abstract.
- **split unit**: `unclear` (disjointness `not_stated`) — No split is described in the abstract.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=—, positional=—, acquisition_metadata=—, permuted_or_shuffled_label=—, clinical_or_demographic_only=—, other_non_imaging=—
  - evidence: NOT DETERMINABLE - full text unreachable.
  - searches_run: REQUIRED SEARCHES NOT RUN - full text unreachable (see screen_protocol.md sec.7 access ladder: PMC/publisher OA no; publisher site HTTP 403/paywall; no institutional subscription held by this screener; OpenAlex/Unpaywall/Crossref/arXiv searched for a repository, accepted-manuscript or preprint copy, none found; interlibrary loan / author request not initiated). The primary-endpoint flags are therefore recorded as NULL, not FALSE: an unevidenced negative on P1 is not accepted by the codebook.
- **positional distribution**: `unclear` — Not determinable from the abstract.
- **headline**: accuracy = 0.9043 (unclear, scope `unclear`, rule `abstract_sentence`); collateral circulation grading from multi-phase CTA
- **cohort**: dataset `unclear` (unclear), modality `CT` (mixed=false), region `brain`; n_patients=—, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`neither`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `unclear`; code `none`; chance_asserted_without_measurement=false
- **confidence** `low`; flag_for_adjudication=true
- **notes**: UNREACHABLE (IEEE Xplore paywall; no OA copy in OpenAlex/Unpaywall/Crossref/arXiv). Abstract is consistent with inclusion (multi-phase CTA is volumetric, a categorical collateral grade is predicted, accuracy is reported) but determines almost no extraction field.

#### pos 53 — PMID 37754951 — EXCLUDED (E-2D)

*A Deep Learning-Based Model for Classifying Osteoporotic Lumbar Vertebral Fractures on Radiographs: A Retrospective Model Development and Validation Study.* — J Imaging 2023. <https://pubmed.ncbi.nlm.nih.gov/37754951/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `exclude`
- **exclusion `E-2D`** — evidence: "In this study, we first automatically detected each lumbar vertebra from lateral radiographs. Then, after preliminary image processing, each vertebra was classified into normal, old, or fresh OLVF" (Section 2, overview). The classifier input is a projection radiograph, a natively 2D acquisition; MRI is used only to set the reference standard.
- **evaluation unit**: `other` — "A total of 3481 LV images for training, validation, and testing and 662 LV images for external validation were collected" - one lumbar vertebra on a lateral radiograph.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `slice_or_image` (disjointness `not_stated`) — Split is over LV images (3481 internal / 662 external).
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded.
  - searches_run: Term sweep executed but not used for the primary endpoint - excluded record.
- **positional distribution**: `no` — No reference to position within an acquired stack (there is no stack - the acquisition is a projection radiograph).
- **headline**: accuracy = 0.89 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); internal test set (external validation 0.84); recorded for completeness only
- **cohort**: dataset `private_multi_centre (two institutions)` (private), modality `other` (mixed=false), region `spine`; n_patients=—, n_patients_test=—, n_slices_or_images=3481, n_positive_reported=`slices_only`
- **other**: interval `none`; input `2D_slice`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Matched the frame because MRI appears in the abstract, but the classifier never sees a volumetric acquisition. Clean E-2D, confirmed against the full text.

#### pos 54 — PMID 34003056 — INCLUDED

*Deep Learning for Malignancy Risk Estimation of Pulmonary Nodules Detected at Low-Dose Screening CT.* — Radiology 2021. <https://pubmed.ncbi.nlm.nih.gov/34003056/>

- **full text**: `repository_or_accepted_manuscript` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `lesion` — Figure 3 legend: "Receiver operating characteristic curves of the deep learning algorithm and Pan-Canadian Early Detection of Lung Cancer (PanCan) model for discrimination of malignant nodules from benign nodules in the full Danish Lung Cancer Screening Trial (DLCST) cohort of 883 nodules." All metrics are per nodule.
- **headline unit**: `na_only_one_unit_reported` — Only nodule-level metrics are reported; there is no patient-level or slice-level metric anywhere.
- **split unit**: `external_cohort_only` (disjointness `stated_only`) — "The algorithm was externally validated in the DLCST cohorts." (Methods, Algorithm Development and Validation). Development used NLST with "10-fold cross validation", whose unit is not stated; the headline number comes from an entirely separate screening trial.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: Appendix E1 (online supplemental material) exists and could NOT be retrieved (RSNA supplement is behind the same paywall as the article; the repository deposit contains the article only). Negative evidenced over the full version-of-record text. || SEARCH NOTE: The five 'baseline' hits all refer to baseline screening-round CT images. The comparator arms are (a) the PanCan model 2b and (b) 11 clinicians. NEITHER is a zero-image baseline: PanCan 2b's predictors include nodule size, nodule type, spiculation, emphysema and upper-lobe location, all read off the image, and clinicians are human readers (does_not_count). 'chance', 'majority', 'trivial', 'metadata', 'clinical-only', 'clinical model', 'slice index' and 'permut' return zero hits.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: Appendix E1 exists but is paywalled and was not retrieved; main article searched in full
- **positional distribution**: `no` — "The NLST database lists the nodules found during screening with lobar locations and CT section numbers, but exact nodule coordinates are not available" and "They had access to the lobe location and CT section number and were instructed to locate all nodules" (Methods, NLST cohort). CT SECTION NUMBER is available to the annotators, but the paper never reports or analyses how the label is distributed along the slice axis, so the code is 'no'.
- **headline**: AUC = 0.93 (external, scope `single_modality_arm`, rule `abstract_multiple_took_external`); full external DLCST cohort (883 nodules), 95% CI 0.89-0.96; internal NLST 10-fold CV AUC 0.91
- **cohort**: dataset `NLST (development); DLCST (external validation)` (public), modality `CT` (mixed=false), region `lung`; n_patients=5881, n_patients_test=599, n_slices_or_images=16960, n_positive_reported=`patients_and_slices`
- **other**: interval `ci_unspecified_method`; input `mixed`; label_broadcast_to_slices `na`; code `public_link_stated`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Reached at rung 4 of the access ladder: version-of-record PDF (CC BY-NC-ND) deposited at the University of Copenhagen repository, https://curis.ku.dk/ws/files/314071118/radiol.2021204433.pdf - the Radiology site itself returned HTTP 403. n_patients (5,881) and n_patients_test (599) are summed from the paper's own stated component counts (NLST 686 + 4,596; DLCST 59 + 540), not back-computed from image counts; n_slices_or_images is the nodule total (16,077 + 883). The model is an ensemble of 2D and 3D CNNs, hence input_representation='mixed'. The PanCan comparator is the closest thing in this batch to a non-imaging baseline and it is NOT one - it consumes radiologist-read image features.

#### pos 55 — PMID 31634769 — UNREACHABLE_ELIGIBILITY_UNRESOLVED

*An expert system for brain tumor detection: Fuzzy C-means with super resolution and convolutional neural network with extreme learning machine.* — Med Hypotheses 2020. <https://pubmed.ncbi.nlm.nih.gov/31634769/>

- **full text**: `unreachable_paywalled` / version used `abstract_only`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `unclear` — "In the proposed method, 98.33% accuracy rate has been detected in the diagnosis of segmented brain tumors using SR-FCM." The abstract never says what one scored unit is.
- **headline unit**: `unclear` — Not determinable from the abstract.
- **split unit**: `unclear` (disjointness `not_stated`) — No split is described in the abstract.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=—, positional=—, acquisition_metadata=—, permuted_or_shuffled_label=—, clinical_or_demographic_only=—, other_non_imaging=—
  - evidence: NOT DETERMINABLE - full text unreachable. The abstract's only comparison is against the same pipeline without super-resolution, which is an ablation of an imaging model (does_not_count).
  - searches_run: REQUIRED SEARCHES NOT RUN - full text unreachable (see screen_protocol.md sec.7 access ladder: PMC/publisher OA no; publisher site HTTP 403/paywall; no institutional subscription held by this screener; OpenAlex/Unpaywall/Crossref/arXiv searched for a repository, accepted-manuscript or preprint copy, none found; interlibrary loan / author request not initiated). The primary-endpoint flags are therefore recorded as NULL, not FALSE: an unevidenced negative on P1 is not accepted by the codebook.
- **positional distribution**: `unclear` — Not determinable from the abstract.
- **headline**: accuracy = 0.9833 (unclear, scope `unclear`, rule `abstract_sentence`); SqueezeNet features + extreme learning machine on SR-FCM-segmented tumours
- **cohort**: dataset `unclear` (unclear), modality `MRI` (mixed=false), region `brain`; n_patients=—, n_patients_test=—, n_slices_or_images=—, n_positive_reported=`neither`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `unclear`; code `none`; chance_asserted_without_measurement=false
- **confidence** `low`; flag_for_adjudication=true
- **notes**: UNREACHABLE (Elsevier paywall, no OA copy anywhere). Abstract is consistent with inclusion because a pretrained SqueezeNet consumes the segmented tumour image before the ELM classifier, but neither the cohort, the split nor the evaluation unit can be determined. Note also that the journal (Medical Hypotheses) does not peer-review in the conventional sense; the record is nonetheless typed 'Journal Article' in PubMed and is not excludable under E-TYPE, which covers reviews, editorials and protocols.

#### pos 56 — PMID 37962500 — EXCLUDED (E-NOCLF)

*Generative Adversarial Network-based Noncontrast CT Angiography for Aorta and Carotid Arteries.* — Radiology 2023. <https://pubmed.ncbi.nlm.nih.gov/37962500/>

- **full text**: `not_attempted_excluded_at_stage1` / version used `abstract_only`; stage-1 decision `exclude`
- **exclusion `E-NOCLF`** — evidence: "To develop an ICA-free deep learning imaging model for synthesizing CTA-like images ... In addition, two senior radiologists scored the visual quality on a three-point scale (3 = good) and determined the vascular diagnosis." (Abstract, Purpose and Materials and Methods). The deep-learning model performs image synthesis; the categorical vascular diagnosis (accuracy 94%, macro F1 91%) is produced by human readers, so no supervised classifier is fitted and criterion I2 fails.
- **evaluation unit**: `patient` — "CT scans from 1749 patients ... 212 for testing. The external validation set comprised CT scans from 42 patients" - reader diagnoses per patient.
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `patient_subject` (disjointness `stated_only`) — "CT scans from 1749 patients ... were included in the internal data set: 1137 for training, 400 for validation, and 212 for testing."
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded at stage 1.
  - searches_run: Not run - excluded at stage 1.
- **positional distribution**: `no` — Nothing in the abstract refers to label position within the stack.
- **headline**: accuracy = 0.94 (internal_held_out, scope `single_modality_arm`, rule `abstract_sentence`); radiologists' diagnostic accuracy reading synthetic CTA on the internal test set (external 86%) - a reader metric, not a classifier metric
- **cohort**: dataset `private_single_centre plus external set` (private), modality `CT` (mixed=false), region `whole_body`; n_patients=1791, n_patients_test=212, n_slices_or_images=—, n_positive_reported=`neither`
- **other**: interval `none`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: Same judgement call as PMID 40335658 in the overlap set and coded identically. E-SEG is checked first and lists 'synthesis', but its qualifier 'with NO categorical class decision evaluated' is not met (a vascular diagnosis WAS evaluated, by humans), so the first code that applies is E-NOCLF - 'reader study with no model'. A second screener may prefer E-SEG; flagged so the disagreement is visible.

#### pos 57 — PMID 30922901 — EXCLUDED (E-DERIV)

*Machine Learning for the Prediction of Cervical Spondylotic Myelopathy: A Post Hoc Pilot Study of 28 Participants.* — World Neurosurg 2019. <https://pubmed.ncbi.nlm.nih.gov/30922901/>

- **full text**: `oa_pmc_or_publisher` / version used `accepted_manuscript`; stage-1 decision `go_to_fulltext`
- **exclusion `E-DERIV`** — evidence: "Images were reviewed in a post-hoc fashion and the degree of cord compression was graded using 3 common literature scales (Kang, Nagata and Chang) alongside 3 MRI measurements (sagittal canal width, vertebral body height to vertebral disk height ratio and the C5 vertebral body sagittal width) all at the point of greatest compression on MRI. These six features were used to train a deep neural network (DNN) classification model." (Methods, Model 1). Model 2 likewise takes "a total of 23 input variables" of tract volumetry plus gender, age, height, weight and level. No image reaches either network.
- **evaluation unit**: `patient` — "A total of 14 patients with CSM and 14 controls underwent imaging of the cervical spine."
- **headline unit**: `na_only_one_unit_reported`
- **split unit**: `random_unit_not_stated` (disjointness `not_stated`) — "The model was trained and tested using cross-validation, in which the data were randomly partitioned into training (n=18) and testing (n=10) datasets" - with 28 participants the units are almost certainly participants, but the sentence does not say so.
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: Not applicable - excluded.
  - searches_run: Term sweep executed but not used for the primary endpoint - excluded record.
- **positional distribution**: `no` — Measurements are taken 'at the point of greatest compression' - an anatomical, per-subject location, not a distribution of labels along the slice axis.
- **headline**: accuracy = 0.865 (cross_validation, scope `single_modality_arm`, rule `abstract_sentence`); mean cross-validated accuracy over 200 random partitions, 95% CI 85.16-87.83%; recorded for completeness only
- **cohort**: dataset `private_single_centre` (private), modality `MRI` (mixed=false), region `spine`; n_patients=28, n_patients_test=10, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `ci_unspecified_method`; input `unclear`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `high`; flag_for_adjudication=false
- **notes**: Clean E-DERIV, confirmed against the full text: both networks consume scalar measurement tables. Worth noting in passing that the reported 95% CI is over 200 random re-partitions of 28 subjects, not over subjects.

#### pos 58 — PMID 42100397 — INCLUDED

*Differentiation between benign and malignant orbital tumors using deep transfer learning features and hand-crafted radiomics features from traditional CT imaging.* — Front Oncol 2026. <https://pubmed.ncbi.nlm.nih.gov/42100397/>

- **full text**: `oa_pmc_or_publisher` / version used `version_of_record`; stage-1 decision `go_to_fulltext`
- **evaluation unit**: `patient` — Figure 1 legend: "out of 176 patients, 12 were excluded for lesions less than five millimeters, 19 excluded for artifacts, leaving 145 patients randomly assigned into 115 for the training cohort and 30 for the test cohort." All AUCs in Tables 3-4 are per patient.
- **headline unit**: `na_only_one_unit_reported` — Only patient-level metrics are reported.
- **split unit**: `patient_subject` (disjointness `stated_only`) — "leaving 145 patients randomly assigned into 115 for the training cohort and 30 for the test cohort" (Figure 1 legend); "To prevent data leakage, the entire dataset was first randomly divided into a training cohort (n=115) and a testing cohort (n=30). All subsequent feature selection steps were performed exclusively in the training cohort" (Methods, Feature selection and fusion).
- **trivial_baseline (PRIMARY)**: constant_or_prevalence=false, positional=false, acquisition_metadata=false, permuted_or_shuffled_label=false, clinical_or_demographic_only=false, other_non_imaging=false
  - evidence: no match. Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists || SEARCH NOTE: IMPORTANT: the 'Clinic' arm in Table 4 (training AUC 0.720, test 0.692) is NOT a pixel-free clinical baseline. "homogeneous enhancement and ill-defined/infiltrative - the features retained as independent in the multivariate model - were selected for integration into the subsequent clinical-semantic model and the final nomogram" (Results); these are radiologist-read CT semantic features, so the arm uses pixels and clinical_or_demographic_only is coded FALSE. The seven 'baseline' hits are the Table 1 baseline-characteristics table.
  - searches_run: Terms searched case-insensitively over the complete text (all sections, tables and figure captions): baseline; chance; random; majority; majority class; prevalence; constant; trivial; metadata; clinical-only; clinical only; clinical model; position; location; slice index; permut; no-information; guess. supplement: none exists
- **positional distribution**: `no` — "Lesion Location (intraconal, extraconal, or trans-spatial)" is an anatomical compartment, not a position within the acquired slice stack.
- **headline**: AUC = 0.837 (internal_held_out, scope `single_modality_arm`, rule `abstract_multiple_took_external`); deep-learning-radiomics nomogram on the test cohort, 95% CI 0.684-0.990 (fused DLR model 0.811, DL 0.826, radiomics 0.816, clinical-semantic 0.692); the abstract's 0.975/0.986 figures are training-cohort values
- **cohort**: dataset `private_single_centre` (private), modality `CT` (mixed=false), region `head_neck`; n_patients=145, n_patients_test=30, n_slices_or_images=—, n_positive_reported=`patients_only`
- **other**: interval `ci_unspecified_method`; input `2D_slice`; label_broadcast_to_slices `na`; code `none`; chance_asserted_without_measurement=false
- **confidence** `medium`; flag_for_adjudication=true
- **notes**: Included because a real CNN consumes a real image: "Before extracting deep learning (DL) features, the region of interest (ROI) with the largest sagittal plane area was selected for cropping. The input images were resampled to a size of 64x64 ... ResNet50 was selected as the base model for transfer learning." That is a single 2D sagittal section of a volumetric CT, so E-PROJ does not apply (it is a cross-section, not a projection). JUDGEMENT CALL on the headline: the abstract lists four models x two cohorts; the nomogram is the authors' final proposal and its test-cohort value was taken. A screener could instead take the fused DLR model (0.811). Contrast with PMID 34655238 in the same batch, excluded as E-DERIV because no network sees the image there.

---

## 5. Judgement calls a reviewer should look at first

Every one of these is recorded in the record's `notes` field and flagged.

1. **Image synthesis + human reader study → `E-NOCLF` or `E-SEG`?** PMIDs 40335658 (overlap) and 37962500 (batch B). `E-SEG` is checked first and lists 'synthesis', but it is qualified *'with NO categorical class decision evaluated'* — and in both papers a categorical decision *was* evaluated, by humans. So the first code that actually applies is `E-NOCLF` ('reader study with no model'). Both are coded the same way; a screener who reads the qualifier differently will code both `E-SEG`, which would show up as a clean 2-record disagreement rather than noise.
2. **Hand-crafted-feature classifiers → `E-DERIV`?** PMID 34655238 (3D gray density coding + random forest) is excluded because the *classifier's* input is a histogram vector. Contrast PMIDs 39061744 and 42100397, both **included**, where a CNN genuinely consumes voxels alongside a radiomics arm. The line drawn is: does a network ever see a spatially resolved image.
3. **A radiology-report NLP paper → `E-DERIV` or `E-NONMED`?** PMID 40442294 (*npj Digit Med*). A free-text report is not in E-DERIV's enumerated list but is squarely 'a non-spatial derived representation with the image discarded', and E-DERIV precedes E-NONMED.
4. **`headline_unit` for a multi-task paper.** PMID 38337016 exists to predict hematoma expansion at patient level (AUC 0.83), but rule A5 makes the headline the *first* number in the abstract's results sentence, which is the slice-level hematoma-detection AUC of 0.99. Coded by the rule and flagged.
5. **`split_unit` where 'cases' means patients.** PMID 42130124 splits '10% of all cases'; Table 2 gives per-case age and sex, so cases are patients and it is coded `patient_subject`. A stricter reading gives `random_unit_not_stated`.
6. **`split_unit` where the paper reports patient-level results but names no split unit.** PMID 34926501 — *'The given data was split into three sets with an 8:1:1 ratio'* — is coded `random_unit_not_stated`, because the codebook explicitly forbids upgrading. This is the single most likely source of disagreement on the overlap set.
7. **Lesion- and vertebra-level units have no home in the `split_unit` enum.** PMIDs 36776294 (lymph nodes) and 42130124 (vertebrae) split at a sub-patient anatomical unit that is neither slice nor scan nor patient. Coded `slice_or_image` and `patient_subject` respectively, both flagged. **A codebook amendment adding a `lesion_or_object` level to `split_unit` would remove a recurring ambiguity** — but it is not made here, because the protocol is frozen and amendments belong in the changelog after adjudication.
8. **`evaluation_unit_reported = 'unclear'` alongside `input_representation = '2D_slice'`.** PMID 41068276 never defines whether one of its 33,984 'MRI scans'/'MRI images' is a slice or a volume, so the unit is coded `unclear` per the ambiguous-case rule, even though the network is plainly 2D. The mismatch is what the codebook produces by design.
9. **Anatomical position is not slice position.** PMIDs 42130124 (fracture level L1–L5), 34926501 (lesion distance to pleura), 42100397 (intraconal/extraconal) all give numeric positional information, and all are coded `positional_distribution_reported = 'no'` because none refers to position *within the acquired stack*. This is the codebook's rule, applied uniformly; it is worth an explicit line in the paper because it is the code most likely to be challenged.
10. **PMID 31093705 is Part II of a two-part report.** Included, because an original experiment with its own 60-lesion test set and its own PPV/sensitivity is reported; `split_unit` is coded `unclear` because the split is defined in Part I and this paper does not restate it. A screener who treats Part II as a methods companion would code `E-TYPE`.

## 6. Known limitations of this batch

- **Eight records could not be read.** All five permitted rungs of the §7 ladder were attempted; only one produced a copy (PMID 34003056, Copenhagen repository). For all eight, the four P1 sub-flags are recorded as **null, not false** — an unevidenced negative on the primary endpoint is not accepted — and the required 14-term search is recorded as NOT RUN.
- **Two of those eight are listed as open access by Unpaywall and OpenAlex and are not reachable in practice** (PMIDs 37222638 via Wiley pdfdirect, 40335658 via Utrecht DSpace; both HTTP 403). This is a concrete instance of the §10 caveat that `oa_status` is an automated hint, and it should be reported as such.
- **Supplementary material could not be downloaded for four included papers** (PMIDs 38337016, 40768653, 34003056, 40442294): both PMC supplement endpoints returned error stubs. Their negative P1 codes rest on the complete main text only.
- **PMID 42489954 was deliberately left unresolved rather than excluded.** The abstract points strongly to `E-DERIV` (a graph attention network over derived structural features), but the codebook permits a stage-1 exclusion only when the code is *unambiguous*, and a voxel-consuming branch cannot be ruled out from the abstract. Expected code on re-screening: `E-DERIV`.
- **Internal inconsistencies were found and recorded, not silently resolved**, in PMIDs 36016875 (test set n = 388 vs a confusion matrix of 573 images), 40768653 (abstract accuracy 96.5% vs results text 91.3%, and two different lists of ten compared architectures) and 42130124 (240 cases vs 216 patients vs 204 test vertebrae).

## 7. Priority for adjudication and interlibrary loan

- **Interlibrary loan first:** PMID 39699671 (*Eur Radiol*, deep-learning csPCa detection on bpMRI, PROSTATEx test set) — the record in this batch closest to our own prostate audit; and PMID 39107903 (*Dentomaxillofac Radiol*), which splits at the level of the coronal CT image with no patient count anywhere in the abstract.
- **Second-screener adjudication (mandatory under §6):** the 8 `low`-confidence records, plus the 13 `medium`-confidence records carrying `flag_for_adjudication = true`.
