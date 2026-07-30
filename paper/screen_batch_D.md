# Batch D + overlap set — screening results (screener S4)

Protocol: `paper/screen_protocol.md` v1.0 (FROZEN 2026-07-29). Codebook: `paper/screen_frame.json` v1.0.
Machine-readable records, with every coded field and its supporting quote: `paper/screen_batch_D.json`.

**Independence.** The overlap set (permutation positions 1–15) was coded without reading any other screener's
output. `paper/screen_batch_A/_B/_C.json` did not exist in the repository when screening of this batch began
(verified by directory listing at the start of the session). They were written concurrently by the other screeners
and appeared on disk while this batch was being coded; they were never opened, and no batch other than D was read
at any point.

**36 records screened**: 15 overlap + 21 batch D.

---

## 1. Headline tallies

### Batch D (21 records)

| flow | n |
|---|---|
| screened | 21 |
| excluded | 8 |
| included | 13 |
| &nbsp;&nbsp;of which full text reachable (**complete-case set**) | 8 |
| &nbsp;&nbsp;of which included but full text unreachable (bounding analysis only) | 5 |
| `unreachable_eligibility_unresolved` | 0 |

Exclusion codes, individually (protocol §9): `E-DERIV` 5, `E-NONMED` 1, `E-SEG` 2.

| endpoint | complete-case value |
|---|---|
| **P1 — reports ≥1 zero-image baseline with a measured value** | **0/8 = 0.0% [0.0%, 32.4%]** |
| S1 — reports any non-imaging baseline (P1 family + clinical/demographic-only) | 1/8 = 12.5% [2.2%, 47.1%] |
| S2 — headline evaluation unit is the slice | 1/8 = 12.5% [2.2%, 47.1%] |
| S3 — of papers reporting any slice-level metric, also report patient-level | 1/1 = 100.0% [20.7%, 100.0%] |
| S4 — explicitly states a subject-level split | 2/8 = 25.0% [7.1%, 59.1%] |
| S5 — reports/discusses the positional distribution of labels | 0/8 = 0.0% [0.0%, 32.4%] |
| S6 — unreachable, over (included + eligibility-unresolved) | 5/13 = 38.5% [17.7%, 64.5%] |
| S8 — reports a subject-clustered uncertainty interval | 1/8 = 12.5% [2.2%, 47.1%] |
| S9 — reports n positive **patients** as well as n positive slices | 0/8 = 0.0% [0.0%, 32.4%] |

### Overlap set (15 records)

| flow | n |
|---|---|
| screened | 15 |
| excluded | 6 |
| included | 8 |
| &nbsp;&nbsp;of which full text reachable (**complete-case set**) | 6 |
| &nbsp;&nbsp;of which included but full text unreachable (bounding analysis only) | 2 |
| `unreachable_eligibility_unresolved` | 1 |

Exclusion codes, individually (protocol §9): `E-DERIV` 3, `E-SEG` 3.

| endpoint | complete-case value |
|---|---|
| **P1 — reports ≥1 zero-image baseline with a measured value** | **0/6 = 0.0% [0.0%, 39.0%]** |
| S1 — reports any non-imaging baseline (P1 family + clinical/demographic-only) | 1/6 = 16.7% [3.0%, 56.4%] |
| S2 — headline evaluation unit is the slice | 1/6 = 16.7% [3.0%, 56.4%] |
| S3 — of papers reporting any slice-level metric, also report patient-level | 0/1 = 0.0% [0.0%, 79.3%] |
| S4 — explicitly states a subject-level split | 1/6 = 16.7% [3.0%, 56.4%] |
| S5 — reports/discusses the positional distribution of labels | 0/6 = 0.0% [0.0%, 39.0%] |
| S6 — unreachable, over (included + eligibility-unresolved) | 3/9 = 33.3% [12.1%, 64.6%] |
| S8 — reports a subject-clustered uncertainty interval | 0/6 = 0.0% [0.0%, 39.0%] |
| S9 — reports n positive **patients** as well as n positive slices | 0/6 = 0.0% [0.0%, 39.0%] |

### All 36 records

| flow | n |
|---|---|
| screened | 36 |
| excluded | 14 |
| included | 21 |
| &nbsp;&nbsp;of which full text reachable (**complete-case set**) | 14 |
| &nbsp;&nbsp;of which included but full text unreachable (bounding analysis only) | 7 |
| `unreachable_eligibility_unresolved` | 1 |

Exclusion codes, individually (protocol §9): `E-DERIV` 8, `E-NONMED` 1, `E-SEG` 5.

| endpoint | complete-case value |
|---|---|
| **P1 — reports ≥1 zero-image baseline with a measured value** | **0/14 = 0.0% [0.0%, 21.5%]** |
| S1 — reports any non-imaging baseline (P1 family + clinical/demographic-only) | 2/14 = 14.3% [4.0%, 39.9%] |
| S2 — headline evaluation unit is the slice | 2/14 = 14.3% [4.0%, 39.9%] |
| S3 — of papers reporting any slice-level metric, also report patient-level | 1/2 = 50.0% [9.5%, 90.5%] |
| S4 — explicitly states a subject-level split | 3/14 = 21.4% [7.6%, 47.6%] |
| S5 — reports/discusses the positional distribution of labels | 0/14 = 0.0% [0.0%, 21.5%] |
| S6 — unreachable, over (included + eligibility-unresolved) | 8/22 = 36.4% [19.7%, 57.0%] |
| S8 — reports a subject-clustered uncertainty interval | 1/14 = 7.1% [1.3%, 31.5%] |
| S9 — reports n positive **patients** as well as n positive slices | 0/14 = 0.0% [0.0%, 21.5%] |

**S7 (cross-tabulation of headline unit against the P1 flag), all 36, complete-case set only:**

| headline_unit | P1 true | P1 false |
|---|---|---|
| na_only_one_unit_reported | 0 | 11 |
| slice | 0 | 1 |
| unclear | 0 | 2 |

**P1 is 0 of 14 in the complete-case set (0.0%, 95% Wilson [0.0%, 21.5%]).** No paper in these 36 records
reported a constant/prevalence, positional, acquisition-metadata or permuted-label baseline with a measured value.
One paper (PMID 41568076) *asserts* a chance level without measuring it, which the codebook codes as
`chance_asserted_without_measurement=true` and every P1 sub-flag false.

---

## 2. Record-by-record

### Overlap set (positions 1–15)

| pos | PMID | venue | decision | reachability | eval unit | split unit | P1 | trivial_baseline flags | conf |
|---|---|---|---|---|---|---|---|---|---|
| 1 | [36776294](https://pubmed.ncbi.nlm.nih.gov/36776294/) | Front Oncol 2023 | included | oa_pmc_or_publisher | lesion | slice_or_image | no | none | high |
| 2 | [41617832](https://pubmed.ncbi.nlm.nih.gov/41617832/) | Eur Radiol 2026 | included | unreachable_paywalled | patient | patient_subject | no | clinical_or_demographic_only | medium ⚑ |
| 3 | [39423605](https://pubmed.ncbi.nlm.nih.gov/39423605/) | Comput Biol Chem 2024 | unreachable_eligibility_unresolved | unreachable_paywalled | unclear | unclear | no | none | low ⚑ |
| 4 | [42130124](https://pubmed.ncbi.nlm.nih.gov/42130124/) | Orthop Surg 2026 | included | oa_pmc_or_publisher | other | random_unit_not_stated | no | none | medium |
| 5 | [36789248](https://pubmed.ncbi.nlm.nih.gov/36789248/) | SN Comput Sci 2023 | excluded / E-DERIV | oa_pmc_or_publisher | slice | random_unit_not_stated | no | none | high |
| 6 | [40335658](https://pubmed.ncbi.nlm.nih.gov/40335658/) | Eur Radiol 2025 | excluded / E-SEG | unreachable_paywalled | unclear | unclear | no | none | medium ⚑ |
| 7 | [40194851](https://pubmed.ncbi.nlm.nih.gov/40194851/) | AJNR Am J Neuroradiol 2025 | excluded / E-DERIV | unreachable_paywalled | patient | unclear | no | none | medium ⚑ |
| 8 | [42489954](https://pubmed.ncbi.nlm.nih.gov/42489954/) | Brain Topogr 2026 | excluded / E-DERIV | unreachable_paywalled | patient | unclear | no | none | low ⚑ |
| 9 | [39061744](https://pubmed.ncbi.nlm.nih.gov/39061744/) | Bioengineering (Basel) 2024 | included | oa_pmc_or_publisher | patient | patient_subject | no | clinical_or_demographic_only | high |
| 10 | [31093705](https://pubmed.ncbi.nlm.nih.gov/31093705/) | Eur Radiol 2019 | included | oa_pmc_or_publisher | lesion | unclear | no | none | medium |
| 11 | [36016875](https://pubmed.ncbi.nlm.nih.gov/36016875/) | Front Pediatr 2022 | included | oa_pmc_or_publisher | slice | slice_or_image | no | none | medium |
| 12 | [36072854](https://pubmed.ncbi.nlm.nih.gov/36072854/) | Front Physiol 2022 | excluded / E-SEG | oa_pmc_or_publisher | unclear | patient_subject | no | none | high |
| 13 | [37222638](https://pubmed.ncbi.nlm.nih.gov/37222638/) | J Magn Reson Imaging 2024 | included | unreachable_paywalled | patient | patient_subject | no | clinical_or_demographic_only | medium ⚑ |
| 14 | [40239684](https://pubmed.ncbi.nlm.nih.gov/40239684/) | Biomed Phys Eng Express 2025 | excluded / E-SEG | not_attempted_excluded_at_stage1 | unclear | unclear | no | none | high |
| 15 | [41068276](https://pubmed.ncbi.nlm.nih.gov/41068276/) | Sci Rep 2025 | included | oa_pmc_or_publisher | unclear | slice_or_image | no | none | medium |

### Batch D (positions 80–100)

| pos | PMID | venue | decision | reachability | eval unit | split unit | P1 | trivial_baseline flags | conf |
|---|---|---|---|---|---|---|---|---|---|
| 80 | [35061759](https://pubmed.ncbi.nlm.nih.gov/35061759/) | PLoS One 2022 | included | oa_pmc_or_publisher | both | slice_or_image | no | none | high |
| 81 | [42052229](https://pubmed.ncbi.nlm.nih.gov/42052229/) | Orthop J Sports Med 2026 | excluded / E-SEG | oa_pmc_or_publisher | unclear | unclear | no | none | high |
| 82 | [35864986](https://pubmed.ncbi.nlm.nih.gov/35864986/) | Front Neurosci 2022 | excluded / E-DERIV | oa_pmc_or_publisher | patient | patient_subject | no | none | high |
| 83 | [41357810](https://pubmed.ncbi.nlm.nih.gov/41357810/) | IEEE Access 2025 | included | oa_pmc_or_publisher | patient | patient_subject | no | none | high |
| 84 | [38584366](https://pubmed.ncbi.nlm.nih.gov/38584366/) | ACS Sens 2024 | excluded / E-NONMED | not_attempted_excluded_at_stage1 | unclear | unclear | no | none | high |
| 85 | [41559509](https://pubmed.ncbi.nlm.nih.gov/41559509/) | J Imaging Inform Med 2026 | included | unreachable_paywalled | unclear | unclear | no | none | medium ⚑ |
| 86 | [40883444](https://pubmed.ncbi.nlm.nih.gov/40883444/) | Sci Rep 2025 | included | oa_pmc_or_publisher | unclear | random_unit_not_stated | no | none | low ⚑ |
| 87 | [41874622](https://pubmed.ncbi.nlm.nih.gov/41874622/) | Eur Radiol 2026 | excluded / E-SEG | unreachable_paywalled | unclear | unclear | no | none | low ⚑ |
| 88 | [40232605](https://pubmed.ncbi.nlm.nih.gov/40232605/) | Med Biol Eng Comput 2025 | included | oa_pmc_or_publisher | lesion | slice_or_image | no | none | high |
| 89 | [36200353](https://pubmed.ncbi.nlm.nih.gov/36200353/) | Clin Otolaryngol 2023 | included | unreachable_paywalled | both | unclear | no | none | medium ⚑ |
| 90 | [39200968](https://pubmed.ncbi.nlm.nih.gov/39200968/) | J Clin Med 2024 | excluded / E-DERIV | oa_pmc_or_publisher | patient | patient_subject | no | clinical_or_demographic_only | high |
| 91 | [40147601](https://pubmed.ncbi.nlm.nih.gov/40147601/) | Neuroimage 2025 | included | unreachable_paywalled | patient | unclear | no | none | medium ⚑ |
| 92 | [35787928](https://pubmed.ncbi.nlm.nih.gov/35787928/) | Int J Radiat Oncol Biol Phys 2022 | excluded / E-DERIV | unreachable_paywalled | other | unclear | no | none | low ⚑ |
| 93 | [30921550](https://pubmed.ncbi.nlm.nih.gov/30921550/) | Comput Med Imaging Graph 2019 | excluded / E-DERIV | oa_pmc_or_publisher | volume_or_scan_not_patient | patient_subject | no | none | high |
| 94 | [35401411](https://pubmed.ncbi.nlm.nih.gov/35401411/) | Front Neurol 2022 | included | oa_pmc_or_publisher | patient | patient_subject | no | none | high |
| 95 | [36170844](https://pubmed.ncbi.nlm.nih.gov/36170844/) | Ophthalmic Res 2023 | included | unreachable_paywalled | slice | slice_or_image | no | none | low ⚑ |
| 96 | [38784688](https://pubmed.ncbi.nlm.nih.gov/38784688/) | Cureus 2024 | included | oa_pmc_or_publisher | patient | random_unit_not_stated | no | clinical_or_demographic_only | high |
| 97 | [35641181](https://pubmed.ncbi.nlm.nih.gov/35641181/) | Cereb Cortex 2023 | excluded / E-DERIV | unreachable_paywalled | patient | unclear | no | none | low ⚑ |
| 98 | [32907561](https://pubmed.ncbi.nlm.nih.gov/32907561/) | BMC Med Inform Decis Mak 2020 | included | oa_pmc_or_publisher | patient | unclear | no | none | medium |
| 99 | [39846055](https://pubmed.ncbi.nlm.nih.gov/39846055/) | Health Inf Sci Syst 2025 | included | unreachable_paywalled | unclear | unclear | no | none | medium ⚑ |
| 100 | [41568076](https://pubmed.ncbi.nlm.nih.gov/41568076/) | Eur J Radiol Open 2026 | included | oa_pmc_or_publisher | patient | site_or_centre | no | none | high |

⚑ = `flag_for_adjudication=true`.

---

## 3. Notes on every record

### Overlap set

**1. PMID 36776294 — Diagnosis of cervical lymph node metastasis with thyroid carcinoma by deep learning application to CT images.**  
*Front Oncol 2023.* included; oa_pmc_or_publisher.
- evaluation unit `lesion` — "The 676 lymph nodes were randomly divided into 70% of the training set (73 benign and 401 malignant lymph nodes) and 30% of the test set (30 benign and 172 malignant lymph nodes). The classification method showed superior performance over other state-of-the-art methods with an accuracy of 96%" (Abstract, Results; repeated in Methods 'For detection, the 676 lymph nodes were randomly divided...'). One scored row = one lymph node.
- split unit `slice_or_image` (`not_stated`) — "For detection, the 676 lymph nodes were randomly divided into 70% of the training set (73 benign and 401 malignant lymph nodes) and 30% of the testing set (30 benign and 172 malignant lymph nodes)." (Methods) and "The training and testing sets for classification were set as same as the detection." No patient-level statement anywhere.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: baseline 0, chance 0, majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0; 'random' (3) all refer to random data splitting/augmentation.
- positional distribution `no` — No match for 'slice index', 'relative position', 'slice position', or any label-vs-position analysis. 'position' (2) and 'location' (3) refer to node localisation/detection, e.g. "how to find LNs on the CT images".
- headline: accuracy = 0.96 (internal_held_out, rule `abstract_sentence`); binary benign vs malignant lymph-node classification; abstract also gives AUC 0.894
- dataset `private_single_centre` (private), CT, head_neck, n_patients=196, n_patients_test=None, n_images=574
- **Notes:** 574 axial CT images / 676 nodes from 196 patients, split at the NODE level, so nodes from the same patient can fall in both train and test. split_unit coded 'slice_or_image' because the enum has no lesion level and the split is explicitly below the subject. Augmentation applied to BOTH sets: "Each CT image in the training and testing sets was expanded to 20 images by rotation, mirror image, changing brightness, and Gaussian noise."

**2. PMID 41617832 — Multimodal deep learning for laryngeal squamous cell carcinoma staging using CT and laryngoscopy.**  
*Eur Radiol 2026.* included; unreachable_paywalled.
- evaluation unit `patient` — "A total of 450 patients were included (median age, 62 years [range, 31-88]; 365 men). The integrated multimodal model achieved AUCs of 0.902 (0.833-0.954) in the internal cohort and 0.888 (0.826-0.944) in the external cohort" (Abstract, Results). Cohorts are counted in patients, so one scored row = one patient.
- split unit `patient_subject` (`stated_only`) — "This retrospective multicenter study included 450 patients with pathologically confirmed LSCC from two Chinese medical centers... They were divided into training (n = 235), internal validation (n = 101), and external validation (n = 114) cohorts." (Abstract, Materials and Methods)
- trivial_baseline — clinical_or_demographic_only=TRUE on the strength of: "Three single-modality models (CT-based deep learning [CT-DL], laryngoscopy-based multiple instance learning [L-MIL], and a clinical logistic regression model [CL]) and their combinations were compared... Performance was evaluated by AUC, accuracy, sensitivity, specificity" plus "outperforming all single- and dual-modality models (p < 0.05)" (Abstract). The numeric CL AUC itself is in the unreachable full text; the p-value comparison establishes that it was measured on the same metric. NOTE: this flag counts toward S1 only, never P1. All four P1 sub-flags coded FALSE and could not be searched in full text.
- positional distribution `unclear` — Not determinable from the abstract; full text unreachable.
- headline: AUC = 0.888 (external, rule `abstract_multiple_took_external`); integrated multimodal model (CL + CT + L), external validation cohort
- dataset `private_multi_centre` (private), CT, head_neck, n_patients=450, n_patients_test=114, n_images=None
- **Notes:** UNREACHABLE. Springer (Eur Radiol) serves abstract + paywall preview only; Unpaywall and OpenAlex both report is_oa=False; not in PMC or Europe PMC. Included because eligibility (I1-I4) is fully determinable from the abstract, but it is OUTSIDE the complete-case denominator per the endpoint definition and enters the bounding analysis. Mixed 2D/3D per rule A2: CT is volumetric, laryngoscopy is not; the headline number is the fused model, hence headline_value_scope='pooled_2D_and_3D'. code_availability='none' could not be verified.

**3. PMID 39423605 — Federated learning and deep learning framework for MRI image and speech signal-based multi-modal depression detection.**  
*Comput Biol Chem 2024.* unreachable_eligibility_unresolved; unreachable_paywalled.
- evaluation unit `unclear` — "The ExpAPO-DCNN obtained accuracy, Loss, Root mean Squared error (RMSE), Mean Squared error (MSE), True Negative rate (TNR), and True Positive rate (TPR) of 98.00 %, 0.023, 0.058, 0.240, 97.90 %, and 96.30 %, respectively." (Abstract) — no unit of evaluation is named anywhere in the abstract.
- split unit `unclear` (`not_stated`) — The abstract contains no statement of any train/test split, cross-validation, or held-out set. Full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search; a full-text negative could NOT be evidenced, so this negative is weak and the paper is not in the complete-case denominator.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: accuracy = 0.98 (unclear, rule `abstract_sentence`); ExpAPO-DCNN under the federated-learning framework; unit of evaluation unstated
- dataset `not_stated` (unclear), MRI, brain, n_patients=None, n_patients_test=None, n_images=None
- **Notes:** UNREACHABLE (Elsevier, Comput Biol Chem; not OA per Unpaywall; not in PMC/Europe PMC; publisher returns HTTP 403 to every automated request from this environment). Coded 'unreachable_eligibility_unresolved' rather than 'included' because I3 cannot be resolved: the abstract says the model takes 'Magnetic Resonance Imaging (MRI) image' inputs but also describes 'pre-processing, feature extraction and detection' carried out separately for MRI and speech, so whether a spatially resolved image ever reaches the classifier (vs a feature vector, which would be E-DERIV) is undeterminable.

**4. PMID 42130124 — Development and Validation of an AI-Integrated System for Automated Fracture Detection and Pedicle Puncture Planning in Lumbar Osteoporotic Vertebral Compression Fractures Based on the Nine-Grid Area Division Method.**  
*Orthop Surg 2026.* included; oa_pmc_or_publisher.
- evaluation unit `other` — "After segmentation, each individual vertebra was cropped from the original CT scan and resampled to a consistent resolution of 64 x 64 x 64 voxels. Fracture classification was implemented using a 3D ResNet50 network... A global average pooling layer aggregated spatial information before the final fully connected layer produced the binary probability of 'fractured' versus 'normal'." (Methods 2.5.2) — one scored row = one vertebra (L1-L5), i.e. ~5 rows per patient.
- split unit `random_unit_not_stated` (`not_stated`) — "2.4 Dataset Partitioning. After labeling was completed, 10% of all cases were randomly selected to serve as an independent test set, excluded from model training. Test set was used exclusively to compare model diagnostic performance with that of experienced spine surgeons after final model development. A ten-fold cross-validation method was used during model training."
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. The four 'baseline' hits are (a) 'Baseline Characteristics'/'Table 2 Baseline demographic' and (b) "the proposed two-stage algorithm significantly outperformed the baseline single-stage nnU-Net model (DSC = 0.853)" — an imaging segmentation comparator, which the codebook's does_not_count list excludes. chance 0, majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — No positional analysis of the label along the slice axis. The 7 'position' hits are anatomical/puncture-path geometry ("Confirmation of pedicle projection location"). Table 2 reports fracture counts by vertebral LEVEL (L1 57, L2 53, L3 45, L4 29, L5 16) — anatomical level, not position within the acquired stack, so 'no' per the codebook.
- headline: AUC = 0.918 (internal_held_out, rule `abstract_sentence`); 3D ResNet50 L-OVCF (fracture) identification, per vertebra
- dataset `private_multi_centre` (private), CT, spine, n_patients=240, n_patients_test=None, n_images=None
- **Notes:** JUDGEMENT CALL on split_unit. The paper splits '10% of all cases'; 'cases' is used interchangeably with patients elsewhere, but the codebook explicitly forbids upgrading to patient_subject on the strength of the word appearing elsewhere and lists only explicit phrasings ('patient-wise', 'subject-disjoint', ...). Coded random_unit_not_stated. Note the mismatch of units: the split is per case while the metric is per vertebra (~5 vertebrae per case). Multi-task paper (segmentation + classification): coded on the classification arm only, per amendment A3.

**5. PMID 36789248 — Grayscale Image Statistical Attributes Effectively Distinguish the Severity of Lung Abnormalities in CT Scan Slices of COVID-19 Patients.**  
*SN Comput Sci 2023.* excluded (`E-DERIV`); oa_pmc_or_publisher.
> Exclusion evidence: "Values of 12 of the 13 statistics derived for each image, omitting the number of pixels in each image, are used as the input variables in this study." and "The total of 513 data records (one for each extract image with twelve grayscale statistics and a VS class) are assessed using multiple ML/DL algorithms configured to optimize VS classification." (Methods, 'Machine and Deep Learning Algorithms Applied to Grayscale Statistics'). Table 1 confirms the CNN is a 1-D network: "Convolutional Neural Network (CNN) 1D Convolutional layer = 5 (filters = 200; size = 3...)". No spatially resolved image reaches any classifier — this is exactly pilot amendment A5.
- evaluation unit `slice` — "A convolutional neural network achieves this with better than 96% accuracy (only 18 images misclassified out of 513)" (Abstract); the 513 records are CT-slice extracts: "thoracic CT-image slices were collected for multiple individuals".
- split unit `random_unit_not_stated` (`not_stated`) — "The multi-K-fold cross validation results obtained from the analysis conducted suggested that a split of 80% training subset: 20% testing subset worked well for the dataset evaluated and this division was randomly applied for this study." (Methods). No unit of separation is named; 513 image extracts come from only 57 individuals.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: baseline 0, chance 0, majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — No match for slice index / relative position / positional distribution of the label.
- headline: accuracy = 0.9649 (cross_validation, rule `abstract_sentence`); CNN, 12-variable model, visual-score (5-class) prediction; 18 errors of 513
- dataset `private_single_centre` (private), CT, lung, n_patients=57, n_patients_test=None, n_images=513
- **Notes:** Textbook E-DERIV: the abstract's phrase 'image attributes as inputs' is literal — a 12-number vector per slice. input_representation coded 'unclear' because the enum has no level for a non-image input. Also worth recording for the paper: 513 slice extracts from 57 individuals split 80/20 with no subject-level separation.

**6. PMID 40335658 — Radiological evaluation and clinical implications of deep learning- and MRI-based synthetic CT for the assessment of cervical spine injuries.**  
*Eur Radiol 2025.* excluded (`E-SEG`); unreachable_paywalled.
> Exclusion evidence: "We sought to evaluate the diagnostic validity of magnetic resonance imaging (MRI)-based synthetic CT (sCT) compared with conventional computed tomography (CT) for cervical spine injuries... A panel of five clinicians independently reviewed the images for diagnostic accuracy, lesion characterization (AO Spine classification), and soft tissue trauma." (Abstract, Methods). The deep-learning component performs image SYNTHESIS; every reported number is human-reader performance ("sCT demonstrated a sensitivity of 97.3% for visualizing fractures"), inter-reader agreement (ICC, Fleiss' kappa) or image similarity (HU mean absolute error, cortical surface distance). No supervised classifier assigns a categorical label. E-SEG explicitly covers 'reconstruction, denoising, synthesis ... with NO categorical class decision evaluated' and precedes E-NOCLF in the ordered list.
- evaluation unit `unclear` — "Thirty-seven patients (44 cervical spine fractures) were enrolled. sCT demonstrated a sensitivity of 97.3% for visualizing fractures." (Abstract) — the unit is the fracture as seen by human readers, not a model output.
- split unit `unclear` (`na`) — No train/test split of any model is described in the abstract; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: sensitivity = 0.973 (unclear, rule `abstract_sentence`); reader sensitivity for fracture visualisation on synthetic CT (human readers, not a classifier)
- dataset `private_multi_centre` (private), multiple, spine, n_patients=37, n_patients_test=None, n_images=None
- **Notes:** JUDGEMENT CALL made without full text (Springer paywall; Unpaywall lists only a submittedVersion in the Utrecht DSpace repository, which also returned HTTP 403 from this environment). Excluded as image synthesis + reader study. If the full text turns out to contain a fitted classifier with a numeric classification metric, this row should flip to 'included'. Flagged for adjudication.

**7. PMID 40194851 — Severity Classification of Pediatric Spinal Cord Injuries Using Structural MRI Measures and Deep Learning: A Comprehensive Analysis across All Vertebral Levels.**  
*AJNR Am J Neuroradiol 2025.* excluded (`E-DERIV`); unreachable_paywalled.
> Exclusion evidence: "T2-weighted MRI scans were utilized to measure CSA, AP width, and RL widths along the entire cervical and thoracic cord. These measures were automatically extracted at every vertebral level of the spinal cord by using the spinal cord toolbox. Deep convolutional neural networks (CNNs) were utilized to classify participants into SCI or TD groups and determine their AIS classification based on structural parameters and demographic factors such as age and height." (Abstract, Materials and Methods). The classifier's input is a table of derived morphometric measures plus demographics — a non-spatial derived representation with the image discarded (pilot amendment A5).
- evaluation unit `patient` — "The CNN-based models demonstrated high performance, achieving 96.59% accuracy in distinguishing SCI from TD participants." (Abstract, Results)
- split unit `unclear` (`not_stated`) — No split is described in the abstract; full text unreachable (PMC record PMC12633662 exists but is under embargo, live=false; www.ajnr.org returns HTTP 403).
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search. NOTE: the CNN itself uses demographic factors (age, height) alongside the morphometric measures, but no demographics-ONLY arm with a measured value is reported in the abstract, so clinical_or_demographic_only is FALSE.
- positional distribution `no` — "Significant differences (P < .05) were found in CSA, AP width, and RL width between SCI and TD participants" measured 'at every vertebral level' — this is an ANATOMICAL level-stratified comparison, not the distribution of the label along the acquired slice stack, so 'no' per the codebook's explicit anatomical-statement rule. Recorded as a judgement call.
- headline: accuracy = 0.9659 (unclear, rule `abstract_sentence`); SCI vs typically developing; a second task (AIS category) reaches 94.92%
- dataset `private_single_centre` (private), MRI, spine, n_patients=61, n_patients_test=None, n_images=None
- **Notes:** UNREACHABLE but excluded on an explicit abstract statement. 61 participants (20 SCI, 41 TD) with 96.59% accuracy and no stated split is itself notable. Flagged because the exclusion rests on the abstract alone.

**8. PMID 42489954 — Multi-Scale Structural MRI Features Reveal Task-Based Functional Connectivity and Its Alterations in Psychiatric Disorders: A Collaborative Graph Attention Network Approach.**  
*Brain Topogr 2026.* excluded (`E-DERIV`); unreachable_paywalled.
> Exclusion evidence: "CoGA-MTN employs a dual-branch graph attention network to extract complementary global statistical and local topological features from structural MRI, and a cross-modal task-coordinated learning mechanism that enables task-conditioned FC prediction alongside multi-disease classification." (Abstract). A graph attention network operates on a node/edge graph of regional descriptors, i.e. a non-spatial derived representation (connectome/feature vector), which is E-DERIV per pilot amendment A5.
- evaluation unit `patient` — "Validated on the Consortium for Neuropsychiatric Phenomics dataset (152 participants, three tasks, four diagnostic groups), CoGA-MTN ... achieves macro F1 scores of 0.68-0.75 across three psychiatric disorders." (Abstract)
- split unit `unclear` (`not_stated`) — No split described in the abstract; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search. 'single-scale baselines' are mentioned ("outperforms single-scale baselines in task-conditioned FC prediction") but these are competing model variants on the same derived features, which the does_not_count list excludes.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: F1 = NULL (unclear, rule `abstract_sentence`); macro F1 reported only as a range 0.68-0.75 across three psychiatric disorders, so headline_value is NULL per the codebook
- dataset `Consortium for Neuropsychiatric Phenomics (CNP)` (public), MRI, brain, n_patients=152, n_patients_test=None, n_images=None
- **Notes:** JUDGEMENT CALL, LOW CONFIDENCE, full text unreachable (Springer, Brain Topogr, closed). Excluded as E-DERIV because the stated architecture (graph attention over 'global statistical and local topological features') implies a graph/feature input rather than a voxel grid. If the network in fact ingests voxel data this must be re-coded as 'included'. This is the single most likely disagreement in my overlap set.

**9. PMID 39061744 — Identification of Calculous Pyonephrosis by CT-Based Radiomics and Deep Learning.**  
*Bioengineering (Basel) 2024.* included; oa_pmc_or_publisher.
- evaluation unit `patient` — "These participants were randomly divided into two independent cohorts: training cohort (n = 123) and testing cohort (n = 59), based on a 7:3 ratio." (Methods) — AUCs in Tables 3 and 4 are computed over patients.
- split unit `patient_subject` (`stated_only`) — "A total of 53 patients with pyonephrosis and 129 patients with hydronephrosis were enrolled and all patients were randomly assigned to the training cohort or the testing cohort in a ratio of 7:3 (123:59)." (Results)
- trivial_baseline — clinical_or_demographic_only=TRUE: "The clinical model based on the three clinical risk factors above exhibited an AUC of 0.904 (95% CI 0.837-0.950) with sensitivity and specificity of 0.853 and 0.865, respectively, in the training cohort (Table 3, Figure 3)." and "In the testing cohort, it yielded an AUC of 0.889 (95% CI 0.781-0.956)". The three factors are fever, blood neutrophils and urine leukocytes — no pixels. This counts toward S1 only. All four P1 sub-flags FALSE: full-text counts baseline 0, chance 0, majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — No match for slice index/relative position/positional distribution of the label.
- headline: AUC = 0.967 (internal_held_out, rule `abstract_sentence`); comprehensive clinical machine-learning model (clinical factors + radiomics), testing cohort
- dataset `private_single_centre` (private), CT, kidney, n_patients=182, n_patients_test=59, n_images=None
- **Notes:** A clean S1 row: a clinical-variables-only arm with a measured AUC on the same metric, and it beats the 3D-CNN (0.904/0.889 vs 0.599 testing). Note also the imaging-only 3D-CNN AUC of 1.000 in training vs 0.599 in testing. Included because the 3D-CNN arm takes a delineated CT ROI (an image); the radiomics arm alone would have been E-DERIV.

**10. PMID 31093705 — Deep learning for liver tumor diagnosis part II: convolutional neural network interpretation using radiologic imaging features.**  
*Eur Radiol 2019.* included; oa_pmc_or_publisher.
- evaluation unit `lesion` — "A post hoc algorithm inferred the presence of these features in a test set of 60 lesions by analyzing activation patterns of the pre-trained CNN model." and "With a mean number of 2.6 labeled features per lesion, the model achieved a precision of 76.5 +/- 2.2% with a recall of 82.9 +/- 2.6% (see Table 3)."
- split unit `unclear` (`not_stated`) — "A test set of 60 lesions was labeled with the most prominent imaging features in each image (1-4 features per lesion). This test set was the same as that used to conduct the reader study in Part I." No unit of separation is stated anywhere in Part II; the split is inherited from Part I, which was not sampled.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: baseline 0, chance 0, majority 0, prevalence 0, trivial 0, metadata 0, slice index 0, permut 0; the single 'constant' hit is unrelated. Comparisons in the paper are against radiologist-defined features, which does_not_count covers.
- positional distribution `no` — No match for slice index/relative position/label-vs-position analysis.
- headline: other = 0.765 (internal_held_out, rule `abstract_sentence`); positive predictive value for identifying the correct radiological features in each test lesion, mean over 20 iterations (recall 82.9%); the six-class lesion accuracy itself is reported only as 'the model misclassified 12% of lesions'
- dataset `private_single_centre` (private), MRI, liver, n_patients=296, n_patients_test=None, n_images=None
- **Notes:** Part II of a two-part series; the lesion classifier itself is reported in Part I (PMID 31016442), which is NOT in this sample. Coded on what Part II reports. 494 training lesions, 60 test lesions, 296 patients; lesion volumes are 24 x 24 x 12 voxels.

**11. PMID 36016875 — An in-depth discussion of cholesteatoma, middle ear Inflammation, and langerhans cell histiocytosis of the temporal bone, based on diagnostic results.**  
*Front Pediatr 2022.* included; oa_pmc_or_publisher.
- evaluation unit `slice` — "The scans were performed on sections where lesions were present, and the number of axial CT sections per scan ranged from 30 to 50. The total number of scans performed was 2,588." together with "A random selection of 85% of the dataset (n = 2,070) was used during the validation process... The remaining 15% of the data (n = 388) were stored and could be used to evaluate the performance of the model after the training was complete." 2,070 + 388 = 2,458 units against only 119 patients, so the scored unit is the axial CT section.
- split unit `slice_or_image` (`not_stated`) — "A random selection of 85% of the dataset (n = 2,070) was used during the validation process... The remaining 15% of the data (n = 388) were stored and could be used to evaluate the performance of the model after the training was complete." (Methods). The abstract instead says "(70% of cases for training and 30% of cases for validation)" — the two statements are inconsistent.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: baseline 0, chance 0, majority 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0; both 'prevalence' hits are in reference titles.
- positional distribution `no` — No match for slice index/relative position/label-vs-position analysis.
- headline: AUC = 0.98 (internal_held_out, rule `abstract_sentence`); ROC for the cholesteatoma class (framework vs physician 0.98 vs 0.91); LCH 0.99, MEI 0.99
- dataset `private_single_centre` (private), CT, head_neck, n_patients=119, n_patients_test=None, n_images=2588
- **Notes:** A strong example for the paper: ~2,458 axial sections from 119 patients split 85/15 at the IMAGE level, so sections from the same patient necessarily appear in both sets, and the histology label is a per-patient label broadcast to every section. Abstract (70/30 of cases) and Methods (85/15 of images) disagree. 95% CI attributed to a 'Tak Long Estate Algorithm', which is not a recognised method.

**12. PMID 36072854 — COVID-19 CT image segmentation method based on swin transformer.**  
*Front Physiol 2022.* excluded (`E-SEG`); oa_pmc_or_publisher.
> Exclusion evidence: "In this study, we propose a new method to improve U-Net for lesion segmentation in the chest CT images of COVID-19 patients." and "The results of ablation experiments demonstrate that this method achieved significant performance gain, in which the mean pixel accuracy is 87.62%, mean intersection over union is 80.6%, and dice similarity coefficient is 88.27%." Every reported number is a segmentation metric; 'mean pixel accuracy' is pixel-wise, not a categorical class decision on an imaging unit.
- evaluation unit `unclear` — Metrics are pixel-wise (mPA, mIoU, Dice); no imaging unit carries a class decision.
- split unit `patient_subject` (`stated_only`) — "Patients were randomly assigned to a training set (60%), an internal validation set (20%) or a test set (20%)." (Methods 2.5)
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: baseline 0, chance 0, majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — No match.
- headline: other = NULL (internal_held_out, rule `abstract_sentence`); mean pixel accuracy 87.62% / mIoU 80.6% / Dice 88.27% — segmentation metrics only; headline_value left NULL so this row cannot leak into any classification endpoint
- dataset `CC-CCII` (public), CT, lung, n_patients=150, n_patients_test=None, n_images=750
- **Notes:** Exactly the trap the codebook warns about: the abstract says the method will 'classify, identify, and segment' four regions and reports an 'accuracy', but the accuracy is pixel-wise. E-SEG.

**13. PMID 37222638 — Prenatal Diagnosis of Placenta Accreta Spectrum Disorders: Deep Learning Radiomics of Pelvic MRI.**  
*J Magn Reson Imaging 2024.* included; unreachable_paywalled.
- evaluation unit `patient` — "324 pregnant women (mean age, 33.3 years) suspected PAS (170 training and 72 validation from institution 1, 82 external validation from institution 2) with clinicopathologically proved PAS (206 PAS, 118 non-PAS)" (Abstract, Population) — cohorts and outcomes are counted in women.
- split unit `patient_subject` (`stated_only`) — "324 pregnant women ... (170 training and 72 validation from institution 1, 82 external validation from institution 2)" (Abstract, Population).
- trivial_baseline — clinical_or_demographic_only=TRUE: "The MRI-based DLR model had a higher area under the curve than the clinical model in three datasets (0.880 vs. 0.741, 0.861 vs. 0.772, 0.852 vs. 0.675, respectively)" (Abstract, Results); the clinical model is defined as "different clinical characteristics between PAS and non-PAS groups" — no pixels, measured on the same metric. S1 only, not P1. All four P1 sub-flags FALSE (abstract-only search).
- positional distribution `unclear` — Not determinable from the abstract.
- headline: AUC = 0.852 (external, rule `abstract_multiple_took_external`); MRI-based deep-learning-radiomics model, external validation dataset (institution 2)
- dataset `private_multi_centre` (private), MRI, other (placenta / pelvis), n_patients=324, n_patients_test=82, n_images=None
- **Notes:** UNREACHABLE (Wiley, JMRI). Unpaywall reports is_oa=True with a publisher pdfdirect link, but onlinelibrary.wiley.com returned HTTP 403 to every automated request from this environment, and reader proxies are blocked as well — so this is an environment access failure, not necessarily a true paywall. Should be retried by a screener with browser access. Eligibility and the S1 flag are both fully evidenced from the abstract.

**14. PMID 40239684 — DCA-U-Net: a deep learning network for segmentation of laser-induced thermal damage regions in mouse skin OCT images.**  
*Biomed Phys Eng Express 2025.* excluded (`E-SEG`); not_attempted_excluded_at_stage1.
> Exclusion evidence: "we propose an efficient and lightweight segmentation model, Dilated ConvNeXT Attention U-Net (DCA-U-Net), based on U-Net... Experimental results on two different sections of mouse skin laser thermal damage Optical Coherence Tomography (OCT) datasets show that our model has better segmentation performance with insufficient or sufficient amount of data." (Abstract). Only segmentation is evaluated; no categorical class decision, no classification metric.
- evaluation unit `unclear` — Segmentation only; no class decision on any imaging unit.
- split unit `unclear` (`na`) — No split described in the abstract; full text not attempted (excluded at stage 1).
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Stage-1 exclusion; no full-text search required or run.
- positional distribution `unclear` — Not applicable.
- headline: other = NULL (unclear, rule `other`); segmentation metrics only
- dataset `private (mouse skin OCT)` (private), OCT, other (mouse skin), n_patients=None, n_patients_test=None, n_images=None
- **Notes:** TWO codes apply: E-SEG (segmentation only) and E-NONMED (preclinical animal-only: mouse skin). E-SEG is first in the codebook's listed order, so E-SEG is recorded; E-NONMED noted here.

**15. PMID 41068276 — IoMT driven Alzheimer's prediction model empowered with transfer learning and explainable AI approach in healthcare 5.0.**  
*Sci Rep 2025.* included; oa_pmc_or_publisher.
- evaluation unit `unclear` — "The publicly available Kaggle Alzheimer's MRI dataset, comprising 33,984 images across four classes (Non-Demented, Very Mild, Mild, and Moderate Demented) was employed." The paper counts only 'images' and never says whether an image is a slice or a volume, and never mentions patients — the codebook's explicit 'unclear' case.
- split unit `slice_or_image` (`not_stated`) — "The dataset was first divided into training, Validation, and testing subsets (70/15/15) using the original class distributions shown in Table 2." (Methods) — the dataset's enumerated unit is the image; no patient identifiers exist in this Kaggle release, so subject-level separation is impossible.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. The two 'baseline' hits are an imaging ablation: "Starting from a baseline ResNet152 classifier without augmentation or XAI, each module was added sequentially" and Table 4 row "Baseline ResNet152 (no augmentation, no XAI) 92.1" — the does_not_count list excludes network-component ablations. The single 'chance' hit is the everyday word ("With a scant chance of regaining health"). majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — No match for slice index/relative position/label-vs-position analysis; the single 'position' hit is unrelated.
- headline: accuracy = 0.9777 (internal_held_out, rule `abstract_sentence`); ResNet152-TL-XAI, four-class
- dataset `Kaggle Augmented Alzheimer MRI Dataset` (public), MRI, brain, n_patients=None, n_patients_test=None, n_images=33984
- **Notes:** n_patients is NULL and that is the finding: the paper never states a patient count because the Kaggle release carries no subject identifiers. The release is additionally PRE-augmented (its name is 'augmented-alzheimer-mri-dataset'), so near-duplicates of the same underlying scan can straddle the 70/15/15 image-level split; the paper adds a Conditional WGAN-GP on top, stated as applied to the training set only. Code link: only the dataset URL is given, no code repository, hence code_availability='none'.

### Batch D

**80. PMID 35061759 — Multi-channel convolutional neural network architectures for thyroid cancer detection.**  
*PLoS One 2022.* included; oa_pmc_or_publisher.
- evaluation unit `both` — Per-image: "reached a diagnostic accuracy rate of 0.989 with ultrasound images and 0.975 with computed tomography scans through the single input dual-channel architecture". Per-patient: "the patient-specific design was implemented for thyroid cancer detection and has obtained an accuracy of 0.95 for double inputs dual-channel architecture and 0.94 for four-channel architecture" and "the segmented left-side and right-side CT scans extracted from one patient will travel through two convolutional channels" (Methodology). Same metric (accuracy) at two units.
- split unit `slice_or_image` (`not_stated`) — "This study, therefore, adopted the 2, 352 CT scans, with 1, 224 left-side and 1, 128 right-side were separately split into training and testing sets." and "we have adopted a 10-fold stratified cross-validation... to obtain the best ratio of the benign and malignant class in both training and testing sets". Table 1 shows those 2,352 CT images come from 578 patients, so the split is below the subject.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. The three 'baseline' hits are single-channel Xception comparisons: "we have established a baseline result with the single-channel architecture to compare to our developed multi-channel architectures" — an imaging-model comparison, excluded by does_not_count. The one 'chance' hit is the everyday word. majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — No match ('position' 0, 'location' 0, 'slice index' 0).
- headline: accuracy = 0.975 (cross_validation, rule `abstract_sentence`); single-input dual-channel (SIDC) Xception on CT, stratified 10-fold CV; the volumetric (CT) arm is reported separately from the ultrasound arm per rule A2
- dataset `private_single_centre (Hospital_X CT); DDTI (ultrasound, public)` (mixed), CT, head_neck (thyroid), n_patients=578, n_patients_test=None, n_images=2352
- **Notes:** Rule A2 applied: mixed ultrasound (2D, not eligible) + CT (volumetric); the CT arm has its own numbers, so included with modality=CT and mixed_modality=true. Table 1 gives CT Hospital_X = 578 patients / 2,352 images (514 benign, 1,838 malignant). Useful S3 row: BOTH per-image (0.975) and per-patient (0.95/0.94) accuracies are reported, and the per-patient number is 2.5 points lower. GitHub link https://github.com/Amyyy-z/Multi-channel-DCNN returned HTTP 200 on 2026-07-29.

**81. PMID 42052229 — Deep Learning-Based Automatic Glenohumeral Joint Segmentation for Determining Whether the Hill-Sachs Lesion Is On-Track or Off-Track.**  
*Orthop J Sports Med 2026.* excluded (`E-SEG`); oa_pmc_or_publisher.
> Exclusion evidence: "Segmentation performance and measurement method reliability were evaluated using the Dice similarity coefficient and intraclass correlation coefficient (ICC), respectively." (Abstract, Methods). Full-text term counts confirm the absence of any classification metric: AUC 0, sensitivity 0, specificity 0, F1 0, kappa 0. The on-track/off-track determination is a geometric rule applied to measurements, not a fitted classifier, and is reported only as workflow time saved: "The semi-automated determination of on-track/off-track status improved workflow efficiency, saving approximately 2 hours".
- evaluation unit `unclear` — No categorical class decision is scored; Dice and ICC are per structure/measurement.
- split unit `unclear` (`not_stated`) — No train/test split is described. "we trained and fine-tuned the model using a dataset of 100 healthy shoulder CT scans" then "we evaluated the model using data from patients with anterior shoulder dislocation" (43 cases) — two separate cohorts, but no split statement.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: baseline 0, chance 0, prevalence 0, trivial 0, metadata 0, slice index 0, permut 0; 'majority' 1 and 'constant' 1 are incidental prose.
- positional distribution `no` — No match.
- headline: other = NULL (unclear, rule `other`); Dice 0.958 (humerus) / 0.950 (scapula) and ICC — segmentation and agreement metrics only
- dataset `private_single_centre` (private), CT, musculoskeletal (shoulder), n_patients=43, n_patients_test=None, n_images=None
- **Notes:** Another instance of the metric-is-not-a-task trap: the title contains 'Determining Whether the Hill-Sachs Lesion Is On-Track or Off-Track' (a categorical decision) but nothing categorical is scored with a classification metric. E-SEG per amendment A4.

**82. PMID 35864986 — Graph Empirical Mode Decomposition-Based Data Augmentation Applied to Gifted Children MRI Analysis.**  
*Front Neurosci 2022.* excluded (`E-DERIV`); oa_pmc_or_publisher.
> Exclusion evidence: "Then, the data are split into training and test sets; the training set is augmented using GEMD; the structural connectivity of the data is calculated and used to feed the deep learning model. Finally, the structural connectivity is also derived for the samples of the test set to demonstrate the capability of the classifier." (Methods). The classifier (BrainNetCNN) input is a 308x308 structural-connectivity matrix built from regional morphometric features — a connectivity matrix, the exact case of pilot amendment A5 (PMID 34924987).
- evaluation unit `patient` — "Therefore, we randomly selected 14 subjects (7 from the gifted group and 7 from the control group) as the original MRI data for the training set. The rest of the subjects are used as the test set, containing 15 subjects."
- split unit `patient_subject` (`stated_only`) — "we randomly selected 14 subjects (7 from the gifted group and 7 from the control group) as the original MRI data for the training set. The rest of the subjects are used as the test set, containing 15 subjects."
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Both 'baseline' hits mean the non-augmented condition: "The average of the 10 sessions' best accuracy using GEMD achieves 78%, which is better than using SMOTE (74.7%) and the baseline case (55.7%)" — an augmentation ablation on the same input, excluded by does_not_count. chance 0, majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — The 22 'position' hits are 'decomposition'/'superposition' substrings and cortical-region prose; no analysis of label distribution along a slice axis (the model never sees a slice axis).
- headline: accuracy = 0.78 (internal_held_out, rule `abstract_sentence`); average of the best accuracy over 10 independent sessions with GEMD augmentation (best single case 93.3%)
- dataset `OpenNeuro ds001988 (gifted children MRI)` (public), MRI, brain, n_patients=29, n_patients_test=15, n_images=None
- **Notes:** 15 gifted vs 14 controls, 29 subjects total, accuracies of 78-93.3%. Excluded as E-DERIV, counted separately in the flow diagram. GitHub link returned HTTP 301 (repository appears to have been renamed/moved), so coded public_link_stated rather than public_link_works.

**83. PMID 41357810 — A Multimodal Adaptive Inter-Region Attention-Guided Network for Brain Tumor Classification.**  
*IEEE Access 2025.* included; oa_pmc_or_publisher.
- evaluation unit `patient` — "We adopted a leave-one-out cross-validation (LOOCV) strategy at the patient level for both training and evaluation. In each fold, we reserved all MRI scans (DW-MRI and T2-MRI) from a single patient for testing, while using data from the remaining patients for training." (Methods)
- split unit `patient_subject` (`stated_and_checked`) — "We adopted a leave-one-out cross-validation (LOOCV) strategy at the patient level... This approach ensured that no data from the same patient appeared in both the training and testing sets, thereby preventing data leakage and providing an unbiased assessment of the model generalizability across different individuals."
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. The two 'baseline' hits are competing imaging architectures: "we computed the mean F1-score difference between our proposed model and the baseline methods using the same bootstrap sample" and "we performed an internal ablation study under consistent training settings, rather than benchmarking against heterogeneous external baselines" — both excluded by does_not_count. chance 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0; the single 'majority' hit is clinical prose about meningioma grades.
- positional distribution `no` — No match ('position' 0, 'location' 0, 'slice index' 0).
- headline: accuracy = 0.9286 (cross_validation, rule `abstract_sentence`); three-class (normal / benign / malignant), LOOCV over 70 patients; sensitivity 80.00%, specificity 94.12%
- dataset `private_single_centre (Mansoura University Hospitals)` (private), MRI, brain, n_patients=70, n_patients_test=70, n_images=None
- **Notes:** The best-specified split in my batch: patient-level LOOCV with disjointness explicitly asserted. uncertainty_interval_reported coded ci_clustered_by_subject because the bootstrap resamples LOOCV predictions and there is exactly one prediction per patient, so the resampling unit IS the subject: "To compute the 95% confidence intervals (CIs) for the mean F1-score and Cohen's kappa, we applied bootstrap resampling to the predictions obtained from the LOOCV folds." A stricter screener could code ci_unspecified_method; recorded as a judgement call. n_patients_test=70 because every patient is a test case exactly once under LOOCV.

**84. PMID 38584366 — Deep Learning-Assisted Colorimetric/Electrical Dual-Sensing System for Ultrafast Detection of Hydrogen Sulfide.**  
*ACS Sens 2024.* excluded (`E-NONMED`); not_attempted_excluded_at_stage1.
> Exclusion evidence: "This study presents a colorimetric/electrical dual-sensing system (CEDS) for low-power, high-precision, adaptable, and real-time detection of hydrogen sulfide (H2S) gas. The lead acetate/poly(vinyl alcohol) (Pb(Ac)2/PVA) nanofiber film was transferred onto a polyethylene terephthalate (PET) flexible substrate by electrospinning" (Abstract). Not human medical imaging at all; matched the frame only because 'PET' (polyethylene terephthalate), 'CNN'/'DNN' and 'detection' occur in the abstract.
- evaluation unit `unclear` — Not applicable — no medical imaging.
- split unit `unclear` (`na`) — Not applicable — excluded at stage 1.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Stage-1 exclusion; no full-text search required or run.
- positional distribution `unclear` — Not applicable.
- headline: other = NULL (unclear, rule `other`); limit of detection 0.1 ppm H2S
- dataset `na` (unclear), other, other (none - gas sensor), n_patients=None, n_patients_test=None, n_images=None
- **Notes:** A clean illustration of the frame's deliberate imprecision (protocol section 2.1): the query's 'PET' term matched polyethylene terephthalate. Report under E-NONMED in the flow diagram.

**85. PMID 41559509 — Customized CNN Architectures Outperform Pre-Trained Models in Differentiating Normal Brain Tissues, Glioma, Meningioma, and Pituitary Tumors.**  
*J Imaging Inform Med 2026.* included; unreachable_paywalled.
- evaluation unit `unclear` — "When evaluated on the CE-MRI Figshare dataset containing 3064 T1-weighted contrast-enhanced images, the model achieved a validation accuracy of 97.01%" (Abstract) — the abstract counts only 'images' and never defines whether an image is a slice or a volume, and never mentions patients.
- split unit `unclear` (`not_stated`) — The abstract describes no split beyond reporting a 'validation accuracy'; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search. The comparisons named in the abstract (ResNet50 89.15%, MobileNetV2 92.89%, VGG16 96.76%) are all imaging models, excluded by does_not_count.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: accuracy = 0.9701 (internal_held_out, rule `abstract_sentence`); customized CNN, four-class (normal / glioma / meningioma / pituitary); reported as 'validation accuracy'
- dataset `CE-MRI Figshare brain tumour dataset (Cheng et al.)` (public), MRI, brain, n_patients=None, n_patients_test=None, n_images=3064
- **Notes:** UNREACHABLE (Springer, J Imaging Inform Med; is_oa=False per Unpaywall and OpenAlex; not in PMC/Europe PMC). Eligibility is determinable from the abstract: the Figshare CE-MRI release consists of 2D slices extracted from T1-weighted contrast-enhanced MRI volumes, so I3 is met. The dataset's 3,064 slices come from 233 patients in the original release, but this paper reports no patient count and no patient-level split, which is precisely the pattern the study is measuring. Not in the complete-case denominator.

**86. PMID 40883444 — Multimodal feature distinguishing and deep learning approach to detect lung disease from MRI images.**  
*Sci Rep 2025.* included; oa_pmc_or_publisher.
- evaluation unit `unclear` — "The medical decathlon MRI inputs from [39] are used in this article for identifying lung infections. This dataset presents malignant and benign stages of lung tumors with 197 test inputs and 1295 training inputs." (Results, Dataset description) — the scored unit is called only an 'input' and is never defined as a slice, a volume or a patient.
- split unit `random_unit_not_stated` (`not_stated`) — "This dataset presents malignant and benign stages of lung tumors with 197 test inputs and 1295 training inputs." plus "Stratified cross-validation was utilized to test performance across varied data subsets to ensure fair generalization and reduce selection or preparation bias across the detection pipeline." No unit of separation is named anywhere.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: chance 0, majority 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0; the 2 'baseline' and 2 'prevalence' hits are prose/reference titles. Comparisons in Figs 5-7 are against other published imaging methods, excluded by does_not_count.
- positional distribution `no` — No match for slice index/relative position/label-vs-position analysis.
- headline: accuracy = NULL (cross_validation, rule `first_results_table_row`); NULL because every performance number appears only in figures: "The comparative study of the accuracy metric is graphically represented in Fig. 5", "Fig. 6 Precision Comparisons", "Fig. 7 Sensitivity Comparisons". The abstract's "8.78% of sensitivity, 8.81% of precision" are relative improvements, not values.
- dataset `Medical Segmentation Decathlon (lung task)` (public), MRI, lung, n_patients=None, n_patients_test=None, n_images=1492
- **Notes:** LOW CONFIDENCE, FLAGGED. Included because a supervised classifier assigning benign/malignant is fitted and accuracy/precision/sensitivity are reported (in figures), so I4 is arguably met; a screener who reads I4 as requiring a number in text or a table would code E-NOMET. Two further problems recorded for the audit trail: the paper calls the Medical Segmentation Decathlon lung task an MRI dataset when that task is CT, and its abstract's headline numbers are improvement percentages, not performance values. Single-author paper.

**87. PMID 41874622 — Automatic framework for evaluating osteoarthritic cartilage severity: high-resolution cartilage thickness mapping and scoring.**  
*Eur Radiol 2026.* excluded (`E-SEG`); unreachable_paywalled.
> Exclusion evidence: "A 3D-UNet was trained to segment femoro-tibial bones and cartilages using MRI from baseline, 1-, 2-, 3-, 4-, 6-, and 8-year follow-ups. CTh-Maps were created for each knee. A ResNet model trained on CTh-Maps assigned a CTh-Score ranging from 0 (healthy cartilage) to 100 (end-stage OA)." and "Both CTh-Maps and CTh-Score showed excellent reproducibility (ICC > 0.98). The CTh-Score demonstrated strong correlations (r = 0.81) with expert assessments of cartilage loss" (Abstract). The evaluated outputs are a segmentation and a CONTINUOUS score; no categorical class decision and no classification metric (no AUC, accuracy, sensitivity, specificity, F1 or kappa) appears in the abstract.
- evaluation unit `unclear` — Scores and ICCs are per knee; no class decision is scored.
- split unit `unclear` (`not_stated`) — No split described in the abstract; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: other = NULL (external, rule `abstract_sentence`); ICC > 0.98 (reproducibility) and r = 0.81 (correlation with MOAKS) — no classification metric
- dataset `Osteoarthritis Initiative (OAI)` (public), MRI, musculoskeletal (knee), n_patients=4796, n_patients_test=None, n_images=None
- **Notes:** LOW CONFIDENCE, FLAGGED, full text unreachable (Springer, Eur Radiol, closed). Two codes were in play: E-SEG (segmentation plus a continuous score, no categorical decision) and E-NOCLF (regression-only outcome); E-SEG is first in the codebook's order. If the full text contains a KL-grade or MOAKS classification arm with a numeric classification metric, this row must flip to 'included'. The paper does publish a project page (lausannekneestudy.org/cthscore) and a Zenodo dataset DOI.

**88. PMID 40232605 — Automated pulmonary nodule classification from low-dose CT images using ERBNet: an ensemble learning approach.**  
*Med Biol Eng Comput 2025.* included; oa_pmc_or_publisher.
- evaluation unit `lesion` — "It should be noted that all evaluations were conducted on a per-lesion basis, ensuring a detailed and specific evaluation for each individual lesion." (Methods)
- split unit `slice_or_image` (`not_stated`) — "The training process involved randomly selecting 60% of the data, comprising 1200 volumes (3D patches) that included 600 nodules and 600 non-nodules." and "To validate the models, 400 nodules and 400 non-nodules were randomly selected and fed into the different models." The split unit is the 64x64x64 patch drawn from 888 patient CT volumes; no patient-level statement anywhere.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. The single 'baseline' hit is an imaging comparison: "the sensitivity of the full-dose network evaluated using the full-dose network was 97.7% which could be regarded as a baseline to relatively assess other low-dose models" — a dose-level ablation on images, excluded by does_not_count. chance 0, majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — No match; the 2 'location' hits are unrelated.
- headline: accuracy = 0.97 (internal_held_out, rule `abstract_sentence`); nodule vs non-nodule on full-dose CT; the ensemble across dose levels reaches 95.0% and the in-house external set 85.5%
- dataset `LUNA16; private in-house (Khatam PET/CT Center, Tehran)` (mixed), CT, lung, n_patients=888, n_patients_test=47, n_images=1000
- **Notes:** LUNA16 — one of the two datasets where our own positional baseline did NOT match, so this row is directly relevant to the audit. Balanced 1:1 nodule/non-nodule design by construction (600/600 train, 400/400 test), split at the patch level with no subject separation stated. code_availability='none' because the statement is "The code used in this work will be publicly available on GitHub upon publication of the paper" with no URL.

**89. PMID 36200353 — Differentiation of eosinophilic and non-eosinophilic chronic rhinosinusitis on preoperative computed tomography using deep learning.**  
*Clin Otolaryngol 2023.* included; unreachable_paywalled.
- evaluation unit `both` — "The average area under the curve and mean accuracy values of the four networks were 0.848 and 0.762 for models trained using a single image as a unit, while the corresponding values for models trained using each patient as a unit were 0.893 and 0.853, respectively." (Abstract, Results) — the same metric at two units, explicitly named.
- split unit `unclear` (`not_stated`) — The abstract states no unit of train/test separation; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: AUC = 0.848 (internal_held_out, rule `abstract_sentence`); average AUC of four transferred networks, per-image models (per-patient models reach 0.893)
- dataset `private_single_centre (Renmin Hospital of Wuhan University)` (private), CT, head_neck (paranasal sinuses), n_patients=878, n_patients_test=None, n_images=None
- **Notes:** UNREACHABLE (Wiley, Clin Otolaryngol; is_oa=False; onlinelibrary.wiley.com HTTP 403 from this environment). Even so this is one of the most informative rows in the batch: the authors deliberately compare an image-as-unit model with a patient-as-unit model and conclude "labeling each patient to build a dataset for classification may be more reliable than labeling each medical image". A per-patient histological label is broadcast to every axial slice, hence label_broadcast_to_slices=true.

**90. PMID 39200968 — Preoperative OCT Characteristics Contributing to Prediction of Postoperative Visual Acuity in Eyes with Macular Hole.**  
*J Clin Med 2024.* excluded (`E-DERIV`); oa_pmc_or_publisher.
> Exclusion evidence: "Instead of using the baseline OCT image as a feature, we annotated the OCT image and used the raster and vector data in that region as handcrafted features." (Methods) and "From 32 handcraft features extracted from the OCT images and 9 features of clinical information..." — the classifier is a logistic regression over a table of hand-measured descriptors (BDM, hole-min, ONL_DL, OPL_DL, preoperative BCVA, age, sex). No spatially resolved image reaches the model: E-DERIV per amendment A5.
- evaluation unit `patient` — "The classifier was developed using preoperative clinical information and the optical coherence tomographic (OCT) findings of 43 eyes of 43 patients who had undergone a vitrectomy." — one eye per patient.
- split unit `patient_subject` (`stated_only`) — "For the test data, 100 pairs were randomly selected from all combinations of patients by performing 5 stratified cross-validation 100 times." and "A stratified cross-validation method for both test and parameter tuning was used so that the ratio of Group A to Group B was the same for each block when the data were divided."
- trivial_baseline — clinical_or_demographic_only=TRUE, and it is an unusually explicit one: "In a control experiment where the OCT features were removed from the explanatory variables, the AUC was 0.82 +/- 0.12 (Figure 7), and the OCT features contributed only slightly to the improvement in the prediction of the postoperative BCVA in eyes with a MH." (Results/Discussion) — a clinical-variables-only arm, measured on the same metric (AUC), against the full model's 0.84 +/- 0.12. All four P1 sub-flags FALSE.
- positional distribution `no` — No match for slice index/relative position/label-vs-position analysis.
- headline: AUC = 0.84 (cross_validation, rule `abstract_sentence`); full model (clinical + OCT-derived features) for good vs poor 6-month BCVA; the clinical-only control reaches 0.82
- dataset `private_single_centre` (private), OCT, retina, n_patients=43, n_patients_test=43, n_images=None
- **Notes:** Excluded (E-DERIV) so it does NOT enter S1, but worth quoting in the paper's Discussion: this is exactly the ablation our checklist asks for, it was run, and it showed the imaging contributed almost nothing (0.84 with OCT features vs 0.82 without). Recorded here so the evidence is not lost behind the exclusion.

**91. PMID 40147601 — Ensemble network using oblique coronal MRI for Alzheimer's disease diagnosis.**  
*Neuroimage 2025.* included; unreachable_paywalled.
- evaluation unit `patient` — "To achieve subject-wise classification based on 2D slices, rather than image-wise classification, we employed ensemble learning methods. This approach fused classification results from different modality images or different positions of the same modality images, constructing a more reliable ensemble classification model." (Abstract)
- split unit `unclear` (`not_stated`) — The abstract states no unit of train/test separation; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search.
- positional distribution `unclear` — The abstract mentions fusing "different positions of the same modality images" but gives no distribution of the label along the slice axis; the full text could not be read to resolve this, so 'unclear'.
- headline: accuracy = 0.975 (unclear, rule `abstract_sentence`); weighted-voting decision fusion on oblique coronal slices, CN vs AD (CN vs MCI 100%, MCI vs AD 94.83%)
- dataset `ADNI` (public), MRI, brain, n_patients=None, n_patients_test=None, n_images=None
- **Notes:** UNREACHABLE despite being gold OA (CC BY-NC-ND) per Unpaywall/OpenAlex: sciencedirect.com and linkinghub.elsevier.com return HTTP 403 to every automated request from this environment, the DOAJ record is 403, and reader proxies are blocked. This is an environment access failure and should be retried by a screener with browser access. Eligibility is clear from the abstract; the paper is explicitly aware of the slice-vs-subject distinction, which is a directly quotable point for the paper. A reported 100% accuracy for CN vs MCI is itself worth flagging.

**92. PMID 35787928 — An Automated Treatment Planning Framework for Spinal Radiation Therapy and Vertebral-Level Second Check.**  
*Int J Radiat Oncol Biol Phys 2022.* excluded (`E-DERIV`); unreachable_paywalled.
> Exclusion evidence: "Features from the CT and auto-contours were input into a random forest classifier to predict whether vertebrae were correctly labeled. This classifier was trained using auto-contours from cone beam computed tomography, positron emission tomography/CT, simulation CT, and diagnostic CT images (n = 56 CT scans, 751 contours)." (Abstract, Methods). The only numeric classification metric in the paper ("The random forest classifier predicted mislabeling across various CT scan types with an area under the curve of 0.82") belongs to a random forest over a derived feature vector; everything else evaluated is segmentation (Dice 85.0-93.7%) or plan quality scored by radiation oncologists.
- evaluation unit `other` — "n = 56 CT scans, 751 contours" — one scored row = one vertebral auto-contour.
- split unit `unclear` (`not_stated`) — No unit of train/test separation is stated in the abstract; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search.
- positional distribution `unclear` — Not determinable from the abstract. NOTE: the whole paper is about vertebral LEVEL identification (C1-L5, including atypical anatomy T13/L6), i.e. position along the spine, which a second screener might read as positional information; but no distribution of the class label along the acquired slice stack is reported in the abstract.
- headline: AUC = 0.82 (unclear, rule `abstract_sentence`); random forest predicting vertebral mislabeling, across CBCT / PET-CT / simulation CT / diagnostic CT
- dataset `private_single_centre` (private), multiple, spine, n_patients=None, n_patients_test=None, n_images=None
- **Notes:** LOW CONFIDENCE, FLAGGED, full text unreachable despite CC-BY per Unpaywall (redjournal.org / sciencedirect return HTTP 403 from this environment). Rule A3 was applied first (multi-task segmentation + classification -> code the classification arm), and the classification arm's input is 'features from the CT and auto-contours', which reads as a derived feature vector -> E-DERIV. If those 'features' turn out to be image patches, this must flip to 'included'. n_patients is NULL because the abstract counts CT scans (220 for contouring, 56 for the classifier, 60 for the planning study), never patients.

**93. PMID 30921550 — Identification of the presence of ischaemic stroke lesions by means of texture analysis on brain magnetic resonance images.**  
*Comput Med Imaging Graph 2019.* excluded (`E-DERIV`); oa_pmc_or_publisher.
> Exclusion evidence: "We used 1800 3D sets of MRI data from three prospective studies... evaluated 114 textural features in WMH, cerebrospinal fluid, deep grey and normal-appearing white matter, and attempted to classify the scans using a random forest and support vector machine classifiers with and without feature selection." (Abstract). The classifier input is a radiomics texture-feature vector with the image discarded — E-DERIV names 'a radiomics feature vector alone' explicitly.
- evaluation unit `volume_or_scan_not_patient` — "We explore the use of radiomics in identifying whether a brain magnetic resonance imaging (MRI) scan belongs to an individual that had a stroke or not." (Abstract) — the scored unit is the scan.
- split unit `patient_subject` (`stated_only`) — "This process was repeated ten times to reduce the variance of the cross validation results and to avoid possible bias in the random separation of the folds (Kuhn and Johnson, 2013c), so at the end 50 models (5 test folds x 10 repetitions) were built using different sets of patients for training and testing each time." (Methods 2.6.3)
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: baseline 0, majority 0, prevalence 0, trivial 0, metadata 0, slice index 0, permut 0; the single 'chance' hit is part of an address ('Chancellor's Building') and the single 'constant' hit is unrelated. The paper does correlate classification success with age but reports no age-only classifier with a measured metric.
- positional distribution `no` — No match; features are tissue-region based, not slice-position based.
- headline: AUC = NULL (cross_validation, rule `abstract_sentence`); NULL because the abstract gives only a range: "the presence of a stroke-type lesion can be ascertained with accuracies ranging from 0.7 < AUC < 0.83"; the best single value in the Results is AUC = 0.667 +/- 0.117 (GLCM, T2W, NAWM, linear SVM)
- dataset `private_multi_centre (one stroke-mechanism study and two cognitive-ageing studies)` (private), MRI, brain, n_patients=None, n_patients_test=None, n_images=1800
- **Notes:** Excluded as E-DERIV (radiomics feature vector), and counted separately per protocol section 9. Note the internal inconsistency between the abstract's 0.7-0.83 range and the Results' best AUC of 0.667. n_patients NULL: the paper counts 1,800 3D MRI datasets and gives per-study patient numbers (e.g. 100 in the first study) but no single total.

**94. PMID 35401411 — Fully Automatic Classification of Brain Atrophy on NCCT Images in Cerebral Small Vessel Disease: A Pilot Study Using Deep Learning Models.**  
*Front Neurol 2022.* included; oa_pmc_or_publisher.
- evaluation unit `patient` — "A total of 385 subjects such as 107 no-atrophy brain, 185 mild atrophy, and 93 severe atrophy were collected and randomly separated into training set (n = 308) and test set (n = 77)." (Abstract, Methods) — one NCCT scan per subject, so the scored unit is the subject.
- split unit `patient_subject` (`stated_only`) — "A total of 385 subjects ... were collected and randomly separated into training set (n = 308) and test set (n = 77)." (Abstract) and "All NCCT scans were randomly separated into a training set (n = 308) and a test set (n = 77) for both two-type and three-type classification tasks." (Methods)
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: baseline 0, chance 0, majority 0, prevalence 0, trivial 0, metadata 0, slice index 0, permut 0. Age and sex are ADDED to the imaging models ("Applying patient age and gender information improved classification performances of both 2D and 3D models") but no age+sex-only model with a measured value is reported, so clinical_or_demographic_only is FALSE.
- positional distribution `no` — The paper has a 'key slice detection' module (a ResNet34 that finds the slices carrying the linear measurements) but reports no distribution of the atrophy label along the slice axis; no match for slice index / relative position.
- headline: AUC = 0.953 (internal_held_out, rule `abstract_sentence`); two-type (atrophy vs no atrophy) classification, authors' proposed 2D linear-measurement model; the end-to-end 3D CNN comparator reaches 0.941 (p = 0.250)
- dataset `private_single_centre` (private), CT, brain, n_patients=385, n_patients_test=77, n_images=None
- **Notes:** Included because the 3D end-to-end CNN arm consumes the image volume directly; the authors' own 2D arm is a pipeline of automated linear measurements (which in isolation would be E-DERIV), hence input_representation='mixed'. A near-miss for our thesis: nine automated linear measurements plus age and sex reach AUC 0.953, statistically indistinguishable from the end-to-end CNN.

**95. PMID 36170844 — Automated Detection of Epiretinal Membranes in OCT Images Using Deep Learning.**  
*Ophthalmic Res 2023.* included; unreachable_paywalled.
- evaluation unit `slice` — "The image-level accuracy was 95.65%, and the ERM region-level accuracy was 90.14%, respectively." (Abstract, Results) — the primary scored unit is the OCT image.
- split unit `slice_or_image` (`not_stated`) — "A total of 422 images (90%) and the remainig 46 images (10%) were used as the training dataset and validation dataset for deep learning algorithm training and validation, respectively." (Abstract, Methods). 468 images come from 468 eyes of 404 patients, so 64 patients contribute both eyes and the split is below the subject.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: accuracy = 0.9565 (internal_held_out, rule `abstract_sentence`); image-level accuracy for identifying/locating epiretinal membrane; region-level accuracy 90.14%
- dataset `private_single_centre` (private), OCT, retina, n_patients=404, n_patients_test=None, n_images=468
- **Notes:** LOW CONFIDENCE, FLAGGED, full text unreachable despite diamond OA per Unpaywall/OpenAlex (karger.com returns HTTP 403 from this environment; the article-page URL patterns tried also 404'd). Two unresolved issues: (1) I3 - with exactly one image per eye (468 images / 468 eyes) it cannot be confirmed from the abstract whether these B-scans come from a volumetric OCT stack or from single line scans, which decides eligibility vs E-PROJ/E-2D; (2) there is no held-out test set distinct from the 46-image validation set. Included per amendment A3 (segmentation + image-level classification -> code the classification arm) and per the rule that under-described papers are not excluded for being under-described.

**96. PMID 38784688 — Use of Artificial Intelligence in the Prediction of Chiari Malformation Type 1 Recurrence After Posterior Fossa Decompressive Surgery.**  
*Cureus 2024.* included; oa_pmc_or_publisher.
- evaluation unit `patient` — "This study included 57 patients who underwent CM1 decompression. The recurrence rate was 30%. The combined model incorporating MRI, pre-operative SF-12 physical component scale (PCS), and extent of cerebellar ectopia performed best with an area under the curve (AUC) of 0.71 and an F1 score of 0.74." (Abstract, Results) — the label (symptom recurrence) is a patient-level outcome and CLAM aggregates each patient's slices into one bag-level prediction.
- split unit `random_unit_not_stated` (`not_stated`) — "Within each of the five folds, the dataset was split randomly into training, validation, and test datasets in a ratio of 60:20:20, respectively." (Methods) — no unit of separation is named. "Data augmentation was performed at the image level such that multiple augmented versions were created for each MRI slice" and "The test set was not augmented."
- trivial_baseline — clinical_or_demographic_only=TRUE: "Five-fold cross-validation was used for the development of MRI only, clinical features only, and a combined machine learning model." (Abstract) with measured values in the results table: "All clinical features 65.0% (20.5) [sensitivity] 60.0% (19.9) [specificity] 63.5% (13.3) [accuracy] 0.70 (0.15) [F1] 0.61 (0.16) [ROC AUC]" against "MRI ... 0.68 (0.19)" — a no-pixels arm on the same metric. S1 only. All four P1 sub-flags FALSE.
- positional distribution `no` — No match for slice index / relative position / label-vs-position analysis.
- headline: AUC = 0.71 (cross_validation, rule `abstract_sentence`); combined MRI + PCS + cerebellar ectopia model; MRI-only 0.68, clinical-features-only 0.61
- dataset `private_single_centre` (private), MRI, brain, n_patients=57, n_patients_test=None, n_images=None
- **Notes:** A clean S1 row with the numbers in the paper's own table: clinical-features-only AUC 0.61 vs MRI-only 0.68 vs combined 0.71, on 57 patients with 17 events. Note split_unit: CLAM works on per-patient bags so the effective unit is probably the patient, but the paper never says so, and the codebook forbids inferring it. label_broadcast_to_slices=true because the patient-level SF-12 recurrence label is attached to every MRI slice for the weakly supervised MIL training.

**97. PMID 35641181 — Classification of major depressive disorder using an attention-guided unified deep convolutional neural network and individual structural covariance network.**  
*Cereb Cortex 2023.* excluded (`E-DERIV`); unreachable_paywalled.
> Exclusion evidence: "Thus, establishing an attention-guided unified classification framework with deep learning and individual structural covariance networks in a large multisite dataset could facilitate developing an accurate diagnosis strategy." and "the discriminative features of regional covariance connectivities and local structural characteristics were found to be mainly located in prefrontal cortex, insula, superior temporal cortex, and cingulate cortex" (Abstract). The classifier input is an individual structural covariance NETWORK - a region-by-region covariance matrix - which is the connectivity-matrix case named in pilot amendment A5.
- evaluation unit `patient` — "Our results showed that attention-guided classification could improve the classification accuracy from primary 75.1% to ultimate 76.54%." (Abstract) — MDD vs healthy control is a subject-level label.
- split unit `unclear` (`not_stated`) — No unit of train/test separation is stated in the abstract; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: accuracy = 0.7654 (unclear, rule `abstract_sentence`); attention-guided unified framework, MDD vs control, multisite
- dataset `large multisite dataset (not named in the abstract)` (unclear), MRI, brain, n_patients=None, n_patients_test=None, n_images=None
- **Notes:** LOW CONFIDENCE, FLAGGED, full text unreachable (OUP, Cereb Cortex, closed; academic.oup.com HTTP 403). Excluded as E-DERIV because the stated input is an individual structural covariance network plus regional structural characteristics. The abstract does say 'local grayscale information' is what OTHER methods use, which leaves open that this model also ingests voxels; if so it must flip to 'included'. Second most likely disagreement in my batch after PMID 42489954.

**98. PMID 32907561 — Semi-supervised method for image texture classification of pituitary tumors via CycleGAN and optimized feature extraction.**  
*BMC Med Inform Decis Mak 2020.* included; oa_pmc_or_publisher.
- evaluation unit `patient` — "An image sequence represents a patient and only one sequence-level label is needed." (Methods, 'Semi-supervised classification of spatial sequence images based on CRNN') and "It needs only sequence-level label, instead of frame-level label, to complete the training for subtype classification of pituitary tumors."
- split unit `unclear` (`not_stated`) — No train/test split is described anywhere in the paper. Table 1 reports Train / Verification / Test accuracies (98.8 / 92.82 / 91.78 for the multi-sequence model) without ever saying how the 152 labelled patients were divided or at what unit.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Full-text counts: chance 0, majority 0, prevalence 0, constant 0, trivial 0, metadata 0, slice index 0, permut 0; the 2 'baseline' hits are unrelated prose. All comparisons are against other imaging feature-extraction methods, excluded by does_not_count.
- positional distribution `no` — No match for slice index / relative position / label-vs-position analysis.
- headline: accuracy = 0.9178 (internal_held_out, rule `abstract_sentence`); multi-sequence CRNN, soft vs hard pituitary tumour texture, test set
- dataset `private_single_centre` (private), MRI, brain (pituitary), n_patients=374, n_patients_test=None, n_images=None
- **Notes:** split_unit='unclear' rather than 'random_unit_not_stated' because the paper contains NO split statement at all, not merely one with an unnamed unit. Two further concerns recorded: the CycleGAN domain converter was trained on the image data of all 374 patients - including the 152 labelled ones later used for train/verification/test - before any split, and the same generator was then used to synthesise the missing modality for the labelled set. input_representation coded 3D_volume: a 2D CNN runs per slice and an RNN runs over the ordered 24-slice (12 T1 + 12 T2) stack, with the label attached at sequence level.

**99. PMID 39846055 — A 3D decoupling Alzheimer's disease prediction network based on structural MRI.**  
*Health Inf Sci Syst 2025.* included; unreachable_paywalled.
- evaluation unit `unclear` — "The accuracy of our model is 0.985 for the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset and 0.963 for the Australian Imaging, Biomarker & Lifestyle (AIBL) dataset" (Abstract, Results) — no unit of evaluation (subject, scan or image) is named.
- split unit `unclear` (`not_stated`) — No unit of train/test separation is stated in the abstract; full text unreachable.
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. Abstract-only search.
- positional distribution `unclear` — Not determinable from the abstract.
- headline: accuracy = 0.963 (external, rule `abstract_multiple_took_external`); 3D decoupling self-attention network on AIBL, the more external of the two datasets reported (ADNI 0.985); tasks are NC vs AD and sMCI vs pMCI
- dataset `ADNI; AIBL` (public), MRI, brain, n_patients=None, n_patients_test=None, n_images=None
- **Notes:** UNREACHABLE (Springer, Health Inf Sci Syst). PMC11748674 exists but is embargoed (idconv live=false; efetch returns front matter only), and link.springer.com serves abstract + paywall preview. Unpaywall's only 'green' location is the PubMed abstract page, which is not a full text. Eligibility is clear from the abstract (3D sMRI, supervised classifier, accuracy reported); the paper is included but sits outside the complete-case denominator. An accuracy of 0.985 for NC vs AD on ADNI with no stated split unit is the pattern of interest.

**100. PMID 41568076 — Automated diagnosis of usual interstitial pneumonia on chest CT via the mean curvature of isophotes.**  
*Eur J Radiol Open 2026.* included; oa_pmc_or_publisher.
- evaluation unit `patient` — "The training and inference for the SOFIA and EfficientNet-V2 pipelines were done on a per-montage (per-slice) basis, and the final patient-level diagnosis was achieved through majority voting" and "Both Table 3, Table 4 report patient-level diagnostic performance" (Methods 2.8 and Results).
- split unit `site_or_centre` (`stated_and_checked`) — "All models were trained on data from the first institution and evaluated on data from the second institution" (Abstract) and "The final training dataset (Institution 1) consisted of 158 patients, and the final testing dataset (Institution 2) comprised 76 patients." (Results)
- trivial_baseline — ALL SUB-FLAGS FALSE. Searches run (see searches_run): no match for any zero-image or non-imaging comparator with a measured value. chance_asserted_without_measurement=TRUE instead: "In a 3-class task, a random classifier is expected to result in a balanced accuracy of 0.33 purely by chance. Our results in Table 3 indicate that the binary mask images achieve a much higher value, between 0.56 and 0.63, which is nearly double the value expected by pure chance" (Results). The 0.33 is ASSERTED, not measured, so every trivial_baseline sub-flag stays FALSE per the codebook. The binary-lung-mask arm (0.56-0.63) is an image-based ablation - it still uses pixels (shape) - so it does not count either. Full-text counts: baseline 0, majority 1 (majority voting), prevalence 1 (discussion of another study), constant 0, trivial 0, metadata 0, slice index 0, permut 0.
- positional distribution `no` — Slice POSITION enters the sampling scheme - "dividing the axial scan into 4 blocks ... then selecting at random one slice from each block and joining each selected slice into a 2 x 2 slice montage" and "This process is repeated 500 times for each 3D scan" - but no distribution of the diagnostic label along the slice axis is reported, so 'no' per the codebook.
- headline: sensitivity = 0.83 (external, rule `abstract_sentence`); recall-macro (macro-averaged over 3 classes) for the MCI-transformed images vs 0.57 for original CT; precision-macro 0.81 vs 0.50 and F1-macro 0.80 vs 0.49
- dataset `private_multi_centre (two independent institutions)` (private), CT, lung, n_patients=234, n_patients_test=76, n_images=17425
- **Notes:** The most rigorously evaluated paper in my batch: trained at one institution, tested at another, patient-level reporting, code public (https://github.com/petersv2/MCI_UIP/ returned HTTP 200 on 2026-07-29). It also comes closest of anything in the batch to our own argument without getting there: the authors run a binary-lung-MASK arm and find balanced accuracy 0.56-0.63, 'only 10% (or less) below the performance achieved with the CT contrast', concluding "clearly, CNNs over-rely on shape and under-exploit CT texture". That is a shape-only, not a pixel-free, control, so it does not satisfy P1 - but it is the right instinct and is worth quoting in our Discussion. label_broadcast_to_slices=true: the patient diagnosis is attached to all 500 montages per scan and aggregated back by majority vote (amendment A9).

---

## 4. Access: what could not be reached, and why

Access ladder per protocol section 7 was executed in order: (1) PMC/publisher OA via NCBI efetch — succeeded for 22 records; (2) publisher site — link.springer.com served abstract+paywall preview only, and onlinelibrary.wiley.com, sciencedirect.com, linkinghub.elsevier.com, academic.oup.com, karger.com, ajnr.org and redjournal.org all returned HTTP 403 to every automated request from this environment, including for articles that Unpaywall and OpenAlex report as OA; (3) institutional access — none held by this screener; (4) repository/preprint — Europe PMC, the NCBI PMC ID converter, Unpaywall and OpenAlex were all queried, and the only repository locations found (a Utrecht DSpace submitted version, a DOAJ record) were also 403; (5) interlibrary/author request — not initiated. Sci-Hub and equivalent sources were NOT used. Records coded 'unreachable_paywalled' therefore include several that are genuinely open access but unreachable from this environment; each says so in its notes and is flagged.

**14 of 36 records could not be read in full text.** Broken down by whether the article is genuinely closed:

| PMID | venue | Unpaywall `is_oa` | what happened |
|---|---|---|---|
| 41617832 | Eur Radiol | False | HTTP 403 / paywall preview only |
| 39423605 | Comput Biol Chem | False | HTTP 403 / paywall preview only |
| 40335658 | Eur Radiol | True (repository, submittedVersion) | HTTP 403 / paywall preview only |
| 40194851 | AJNR Am J Neuroradiol | True (hybrid) | HTTP 403 / paywall preview only |
| 42489954 | Brain Topogr | False | HTTP 403 / paywall preview only |
| 37222638 | J Magn Reson Imaging | True (publisher, publishedVersion) | HTTP 403 / paywall preview only |
| 41559509 | J Imaging Inform Med | False | HTTP 403 / paywall preview only |
| 41874622 | Eur Radiol | False | HTTP 403 / paywall preview only |
| 36200353 | Clin Otolaryngol | False | HTTP 403 / paywall preview only |
| 40147601 | Neuroimage | True (gold, CC BY-NC-ND) | HTTP 403 / paywall preview only |
| 35787928 | Int J Radiat Oncol Biol Phys | True (hybrid, CC BY) | HTTP 403 / paywall preview only |
| 36170844 | Ophthalmic Res | True (diamond) | HTTP 403 / paywall preview only |
| 35641181 | Cereb Cortex | False | HTTP 403 / paywall preview only |
| 39846055 | Health Inf Sci Syst | True (green — but the only location is the PubMed abstract page) | HTTP 403 / paywall preview only |

Seven of the fourteen unreachable articles are recorded by Unpaywall/OpenAlex as **open access** (for two of those
seven the only 'OA' location is a PubMed abstract page or a repository *submitted* version, so five are genuinely
supposed to be free at the publisher). They are unreachable
because every publisher host tried (Wiley, Elsevier/ScienceDirect, Karger, AJNR, OUP, Elsevier's redjournal) returns
HTTP 403 to automated requests from this environment, and public reader proxies are blocked from this network as well.
That is a screener-environment limitation, **not** a statement about the article. Every such record says so in its
`notes` and carries `flag_for_adjudication=true`; they should all be retried by a screener with browser or
institutional access before the bounding analysis is finalised.

Consequence for the analysis: unreachable papers are **8 of 22** eligible-looking records here (S6 = 36.4%,
95% Wilson [19.7%, 57.0%]), far above the 15% threshold in protocol §7 rule 4. For this batch the bounding interval —
not the complete-case estimate — would therefore be the headline number:

- **lower bound P1 = 0/22 = 0.0%, 95% Wilson [0.0%, 14.9%]** (every unreachable paper assumed NOT to report one)
- **upper bound P1 = 8/22 = 36.4%, 95% Wilson [19.7%, 57.0%]** (every unreachable paper assumed to report one)

These are batch-level figures only; the analysis is defined over the pooled sample and the extension-rule reserve.

---

## 5. Judgement calls, flagged

15 of 36 records carry `flag_for_adjudication=true`; 
7 carry `screener_confidence='low'` (which triggers mandatory adjudication).
The substantive calls, in descending order of how likely I think they are to be disputed:

1. **PMID 42489954 (overlap, pos 8) — E-DERIV vs included.** Coded excluded because a *graph attention network* over
   "global statistical and local topological features from structural MRI" implies a graph/feature input, not a voxel
   grid. Full text unreachable. This is the single most likely disagreement in the overlap set.
2. **PMID 35641181 (batch D, pos 97) — E-DERIV vs included.** Same shape of call: input is an "individual structural
   covariance network". The abstract's contrast with "local grayscale information" leaves open that voxels are also used.
3. **PMID 40335658 (overlap, pos 6) — E-SEG.** MRI→synthetic-CT generation evaluated by five human readers. I read
   every reported number as reader performance or image similarity, so no classifier is scored. Unreachable full text.
4. **PMID 41874622 (batch D, pos 87) — E-SEG vs E-NOCLF vs included.** Segmentation plus a *continuous* 0–100 score;
   no classification metric in the abstract. E-SEG chosen because it is first in the codebook's ordered list.
5. **PMID 35787928 (batch D, pos 92) — E-DERIV.** Rule A3 says code the classification arm; that arm is a random
   forest over "features from the CT and auto-contours". If those features are image patches this flips to included.
6. **PMID 40883444 (batch D, pos 86) — included vs E-NOMET.** Every performance number lives in a figure; the abstract
   quotes only relative improvements. I read I4 as satisfied and set `headline_value=NULL`.
7. **PMID 42130124 (overlap, pos 4) — `split_unit`.** The paper splits "10% of all cases". I coded
   `random_unit_not_stated`, not `patient_subject`, because "cases" is not one of the codebook's accepted phrasings and
   the codebook explicitly forbids upgrading on the strength of "patients" appearing elsewhere. Applied uniformly:
   a split sentence whose unit noun is *patients/subjects/participants* → `patient_subject`; *cases* → not stated;
   *images/scans* → `slice_or_image`/`scan_or_study`.
8. **PMID 41357810 (batch D, pos 83) — `ci_clustered_by_subject`.** The bootstrap resamples LOOCV predictions and there
   is exactly one prediction per patient, so the resampling unit is the subject. A stricter screener would code
   `ci_unspecified_method`. This is the only S8-positive record in the batch, so the endpoint is sensitive to it.
9. **PMID 36170844 (batch D, pos 95) — I3.** 468 OCT images for 468 eyes = one image per eye; whether these B-scans
   come from a volumetric stack (eligible) or single line scans (E-PROJ/E-2D) is not decidable from the abstract, and
   the full text is unreachable. Included per the rule that under-described papers are not excluded for being
   under-described.
10. **PMID 40194851 (overlap, pos 7) — `positional_distribution_reported='no'`.** The paper compares cord
    cross-sectional measures "at every vertebral level". I read vertebral level as *anatomical*, which the codebook
    sends to 'no'; another screener could reasonably read it as position along the slice axis.

---

## 6. Two things worth carrying into the paper's Discussion

Both are quotable, both are in the complete-case set, and neither satisfies P1 — which is the point.

**PMID 41568076 (Savadjiev et al., *Eur J Radiol Open* 2026) gets closest.** The authors run a binary-lung-**mask** arm
and report: *"the mask images achieve a much higher balanced accuracy value than what is expected by chance. Furthermore,
the balanced accuracy value achieved with binary mask images is only 10 % (or less) below the performance achieved with
the CT contrast"*, concluding *"clearly, CNNs over-rely on shape and under-exploit CT texture"*. That is a shape-only
control, not a pixel-free one, so P1 stays false — but it is the same instinct as ours, arrived at independently.

**PMID 39200968 (Mase et al., *J Clin Med* 2024) actually ran the ablation and it came out against the imaging.**
*"In a control experiment where the OCT features were removed from the explanatory variables, the AUC was 0.82 ± 0.12
… and the OCT features contributed only slightly to the improvement in the prediction of the postoperative BCVA"* —
against 0.84 ± 0.12 with them. The paper is **excluded** (E-DERIV: the model consumes hand-measured OCT descriptors,
not pixels), so it contributes to neither P1 nor S1, but the evidence should not be lost behind the exclusion code.

Three further patterns recur and are worth a sentence each in the Results:

- **`n_patients` is NULL in 5 of 21 included records** because the paper counts only "images" and never a patient
  (e.g. PMID 41068276, a 33,984-image Kaggle release with no subject identifiers at all).
- **`split_unit` is `unclear` or `random_unit_not_stated` in 20 of 36 records**, and explicitly subject-level in only
  3 of 14 complete-case papers.
- **Zero of 14 complete-case papers report the positional distribution of the label** (S5 = 0/14).
