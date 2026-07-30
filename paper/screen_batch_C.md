# Screening results -- BATCH C + shared OVERLAP SET

**Screener:** S3 | **Coded:** 2026-07-29 | **Protocol:** `paper/screen_protocol.md` v1.1 (FROZEN) | **Codebook:** `paper/screen_frame.json` v1.0 (FROZEN)

**Records coded: 36** = 15 overlap (permutation positions 1-15) + 21 batch C (positions 59-79). Machine-readable version with every field and its supporting quote: `paper/screen_batch_C.json`.

**Independence.** The overlap set was coded without reading any other screener's output. No `screen_batch_A/B/D.*` file was opened at any point. The only shared artefacts touched were raw article XML/HTML fetched from NCBI and publishers.

---

## 1. Headline tallies

| | overlap (15) | batch C (21) | total (36) |
|---|---|---|---|
| **included** | 6 | 7 | 13 |
| **excluded** | 5 | 11 | 16 |
| **unreachable, eligibility unresolved** | 4 | 3 | 7 |

Inclusion yield among *resolved* records: 13/29 = 45%. Counting the 7 unresolved records in the denominator gives 36%. Both are inside the 30-50% band the pilot predicted (protocol sec.4.2).

### Endpoints, this batch only (raw counts -- no intervals; pooling and Wilson intervals belong to the analysis stage)

| endpoint | count | denominator | note |
|---|---|---|---|
| **P1 -- reports >=1 zero-image baseline with a measured value** | **0** | 13 included and reachable | not one paper measured a constant/prevalence, positional, acquisition-metadata or permuted-label comparator |
| S1 -- any non-imaging baseline (P1 family + clinical/demographic-only) | 1 | 13 | PMID 39061744 only |
| S2 -- headline evaluation unit is the slice | 1 | 13 | PMID 36016875 only |
| S3 -- of papers with any slice-level metric, also report patient level | 0 | 1 | the one slice-level paper reports no patient-level metric |
| S4 -- explicitly states a subject-level split | 5 | 13 | 39061744, 36229338, 35872945, 40093990, 38298725 |
| S5 -- reports/discusses the positional distribution of labels | 0 | 13 | zero, on the codebook's own definition; see sec.4 |
| S6 -- eligible-looking but full text unreachable | 7 | 20 = 35% | **above the 15% trigger in protocol sec.7** |
| S8 -- subject-clustered uncertainty interval | 0 | 13 | 4 papers give a 95% CI of unspecified method; none is clustered |
| S9 -- reports n positive patients as well as n positive slices | 1 | 13 | PMID 36229338 only |

**S7 cross-tabulation (headline unit x P1 flag), included and reachable papers:**

| headline unit | P1 true | P1 false | total |
|---|---|---|---|
| na_only_one_unit_reported | 0 | 13 | 13 |
| **total** | **0** | **13** | **13** |

Every included paper reports exactly one evaluation unit, so `headline_unit` is `na_only_one_unit_reported` throughout and the informative cross-tabulation is against `evaluation_unit_reported` instead:

| evaluation unit | n | P1 true |
|---|---|---|
| lesion | 3 | 0 |
| other | 1 | 0 |
| patient | 3 | 0 |
| slice | 1 | 0 |
| unclear | 2 | 0 |
| volume_or_scan_not_patient | 3 | 0 |

---

## 2. Flow, by exclusion code

| code | n | PMIDs |
|---|---|---|
| E-SEG | 8 | 40335658, 36072854, 40239684, 34324463, 41547664, 34603980, 39384719, 34888191 |
| E-PROJ | 1 | 42367846 |
| E-DERIV | 4 | 36789248, 40194851, 34476208, 30734849 |
| E-NOCLF | 2 | 38082902, 36762417 |
| E-NONMED | 1 | 36539234 |
| **total excluded** | **16** | |

`E-DERIV` (4 papers: 36789248, 40194851, 34476208, 30734849) is reported separately per protocol sec.9: these are inside the query and outside the failure mode, because the image never reaches a model. `E-SEG` at 8/16 confirms the pilot's warning -- five of those eight report a metric whose name sounds like classification (`mean pixel accuracy`, voxel-wise `AUC`, reader `sensitivity`, `classification accuracy by location`, `misclassified in 34/399 cases`) on a task that is segmentation, synthesis, reconstruction or measurement.

The single most quotable frame-imprecision case in this batch is **PMID 36539234**, *A deep learning approach to identify missing is-a relations in SNOMED CT* (JAMIA). It matched on "deep learning" + "CT" + "classification" + "F1". A full-text search for `image` and `imaging` returns **zero hits**.

---

## 3. The primary endpoint: 0 of 13

Not one included, reachable paper measured a zero-image baseline. The negatives are evidenced: all 14 required search terms (`baseline, chance, random, majority, prevalence, constant, trivial, metadata, clinical-only, clinical model, position, location, slice index, permut`) were run over each full text and the hits recorded in `trivial_baseline_quote`. What the hits actually were:

- `baseline` -- almost always **the baseline visit** (32714766: "These databases contain baseline brain MR imaging") or a **Baseline Characteristics table** (42130124, 34476208), or an **architecture ablation** (41068276: "Baseline ResNet152 (no augmentation, no XAI) 92.1"), or an **XAI reference image** (40093990: "Using a black image as the baseline"). None is a pixel-free comparator.
- `majority` -- the ensemble's **majority-vote classifier** (34976558) or the **majority class** in an imbalance discussion (38298725). Never a majority-class predictor with a measured score.
- `permut` -- 8 hits in 42367846, all **permutations of simulated ligament injury states**, not a label permutation null.
- `chance` -- 2 papers assert a chance level and measure nothing: 40093990 ("0 signifying no better than random chance", for MCC) and 38298725 ("A value of 0.5 implies random concordance", for the C-index). Both are recorded in `chance_asserted_without_measurement`, which is exactly what that field exists for.

**The one S1 hit, PMID 39061744** (calculous pyonephrosis, CT), is worth the paper's attention even though it does not count toward P1. Its clinical-variables-only model -- fever, blood neutrophils, urine leukocytes, no pixels -- scored **AUC 0.889 (95% CI 0.781-0.956)** on the test cohort, while the 3D-CNN on the same test cohort scored **0.599** after reaching **1.000** in training. The authors report this without comment. It is a clean, already-published instance of a no-pixel model beating the deep model, produced by a team that was not looking for one.

Two more measured clinical-only arms sit just outside the endpoint: **34476208** (excluded E-DERIV) reports a clinical model at AUC 0.737 / 0.560 / 0.585 across three cohorts, and **37222638** (unreachable) and **41617832** (unreachable) both describe one in the abstract. If either of the latter two is obtained at rung 5, S1 in this batch rises.

---

## 4. Where I had to make a judgement call

21 of 36 records carry `flag_for_adjudication = true`. The calls that could move a number:

**(a) Positional distribution: two papers tabulate the label along an ordered anatomical axis, and I coded both `no`.** PMID 42130124's Table 2 gives fracture counts by vertebral level (L1 57, L2 53, L3 45, L4 29, L5 16) and the vertebra *is* the classification unit, sorted along the z-axis by the authors' own pipeline. PMID 35378943's Table 2 gives nodule counts by lung lobe. Both are label distributions along the cranio-caudal axis and a reasonable screener could code them `figure_or_table`. I coded `no` because the codebook states flatly that "Anatomical statements with no reference to position WITHIN the acquired stack ... are 'no'", and I would rather follow the frozen rule than improvise a better one. **This is the single most likely source of disagreement on S5 in the overlap set** and the protocol authors may want to sharpen the rule in the changelog.

**(b) Two retracted papers, and the codebook has no rule for retractions.** PMID 35872945 and PMID 35378943 both carry PubMed publication type *Retracted Publication*. `E-TYPE` lists reviews, editorials, errata, protocols and dataset descriptors -- not retracted primary research -- and protocol sec.4.1 forbids dropping papers for being poorly described. I **included** both and flagged them. If a retraction rule is adopted, my included n drops from 13 to 11 and P1 stays 0/11.

**(c) PMID 34603980 (thoracic aorta) is the closest exclusion call in the batch.** A dilatation category is derived and evaluated ("An aneurysm was misclassified in 34/399 cases (8.5%)"), but it comes from thresholding a measured diameter at 45 mm / 40 mm, not from a fitted supervised classifier, so I2 fails and E-SEG applies. A second screener could defensibly include the dilatation arm.

**(d) PMID 31093705 (liver tumour, Part II).** The six-class CNN was fitted and evaluated in Part I; this paper evaluates a post-hoc feature-identification algorithm over that pre-trained model. Included because a categorical label (feature present/absent) is still assigned to a lesion image and PPV/sensitivity are reported. Not `E-DUP` -- Part I is not in the sample.

**(e) PMID 36229338 (cataract on SS-OCT).** Two calls: the model is pixel-wise but an eye-level categorical decision is evaluated with ROC AUC (amendment A3 -> include, classification arm only); and the acquisition is a set of **radial** B-scans through the lens rather than a parallel stack, which I judged to satisfy I3.

**(f) PMID 38298725 (renal cancer).** The abstract's results sentence contains no classification metric at all -- C-index, IBS and AUC are all survival metrics -- so rule A5's fallback fired and the headline came from the first results table. That table is transposed, so "first row" is itself ambiguous.

**(g) PMID 42367846 (wrist ligaments, bioRxiv).** Coded `E-PROJ` because the classifier input is a conformal bone-surface proximity map with no slice axis. `E-NONMED` also applies on a strict reading -- the model never sees a patient image, only finite-element simulation output derived from two asymptomatic volunteers. E-PROJ wins on the codebook's first-applicable-code rule. Preprints are also not addressed in the codebook.

**(h) PMID 42489954 (graph attention on structural MRI)** reads like `E-DERIV` from its abstract, but a graph attention network can also run over spatially resolved patches, so the abstract does not make it unambiguous and stage-1 exclusion would have violated the codebook. It stays unresolved rather than excluded -- the conservative direction.

---

## 5. Reachability -- above the 15% trigger, and the ladder was not fully executed

**7 of 20 eligible-looking records (35%) could not be reached.** That is above the 15% threshold at which protocol sec.7 says the bounding interval replaces the complete-case estimate as the headline number. Whether the trigger fires depends on the pooled sample, not on my batch alone, but it should be watched.

Unreachable: 41617832, 39423605, 42489954, 37222638 (overlap); 42153825, 40081198, 38591974 (batch C).

**Rungs 1-4 were exhausted for every one of them** -- PMC via E-utilities *and* via the PMC website, Europe PMC `fullTextXML`, the publisher site directly, the OpenAlex open-access location list, and every repository or preprint version OpenAlex indexed. Publishers returned **HTTP 403** (RSNA, Springer, Wiley, Elsevier, SAGE, AJNR) or **HTTP 418** (IEEE). **Rung 5 -- interlibrary loan or a direct request to the corresponding author, with the 21-day wait -- could not be executed in this session.** These records should therefore be read as *unreachable to screener S3 within this session* and re-attempted before the flow diagram is finalised. This is a stated deviation from the frozen protocol, not a silent one.

Two access notes worth recording. **PMID 42153825 and PMID 38591974 both carry a CC BY licence in their own abstracts** ("Published under a CC BY 4.0 license") and are still unreachable, because pubs.rsna.org returns 403 to every client and neither has a PMC record -- so `oa_status` in `screen_sample.json` would have been misleading in the opposite direction from the one protocol sec.10 anticipates. **PMID 40194851** has a PMC record (PMC12633662) that is embargoed: "This article will be available in PMC on October 01, 2026".

Also worth logging for reproducibility: the PMC website served a reCAPTCHA interstitial on two of my requests (PMC12092874) and the E-utilities endpoint returned HTTP 429 twice. Both succeeded on retry with backoff. A third party reproducing this screen should expect to need retries, not to conclude the article is unavailable.

---

## 6. What the included papers actually look like

| field | distribution across the 13 included, reachable papers |
|---|---|
| evaluation unit | lesion 3, patient 3, volume_or_scan_not_patient 3, unclear 2, other 1, slice 1 |
| split unit | patient_subject 5, slice_or_image 3, random_unit_not_stated 2, unclear 2, external_cohort_only 1 |
| split disjointness | not_stated 7, stated_only 5, stated_and_checked 1 |
| positional distribution | no 13 |
| uncertainty interval | none 8, ci_unspecified_method 4, sd_across_folds 1 |
| n positive reported | patients_only 6, slices_only 5, na 1, patients_and_slices 1 |
| input representation | 2D_slice 4, 3D_volume 4, patch_3D 3, unclear 2 |
| modality | CT 8, MRI 4, OCT 1 |
| dataset | private 8, public 3, mixed 2 |
| code availability | none 11, public_link_stated 2 |

Three observations that bear on the paper's argument.

**Only 5 of 13 say they split by subject.** The other eight split by image (36016875: 2,070 vs 388 *axial sections* from 119 patients; 41068276: 33,984 *images*, no patient count anywhere in the paper), by lesion (36776294: 676 lymph nodes from 196 patients, split by node), by an unnamed "case" (42130124), or do not describe a partition at all (34976558, 35378943, 31093705).

**Two papers never state a patient count at all** (41068276, 34976558) and are coded `n_patients = NULL`. A third (34976558) uses images scraped from the figures of other papers, so the subject structure is not merely unstated but unknowable.

**Not one paper reports a subject-clustered interval.** Four report a 95% CI of unspecified method, one reports SD across retraining iterations, eight report no interval at all.

The strongest single example for the paper's thesis in this batch is **PMID 36016875** (temporal bone CT, Front Pediatr): one histology-proven diagnosis per patient is broadcast to every axial section ("Each of the 119 patients was matched with one-third of the disease labels"), 2,588 sections from 119 patients are split **at the image level**, and image-level accuracies of 0.99 with AUCs of 0.98-0.99 are reported against clinicians -- with no patient-level number anywhere. Its abstract and its Methods also disagree about the split (70/30 of "cases" vs 85/15 of n=2,070/388 images).

---

## 7. Record-by-record

### Overlap set (positions 1-15)

| pos | PMID | year, venue | decision | eval unit | split unit | P1 | conf |
|---|---|---|---|---|---|---|---|
| 1 | 36776294 | 2023, Front Oncol | **included** | lesion | slice_or_image | false | high |
| 2 | 41617832 | 2026, Eur Radiol | *unreachable* :triangular_flag_on_post: | unclear | unclear | - | low |
| 3 | 39423605 | 2024, Comput Biol Chem | *unreachable* :triangular_flag_on_post: | unclear | unclear | - | low |
| 4 | 42130124 | 2026, Orthop Surg | **included** :triangular_flag_on_post: | other | random_unit_not_stated | false | medium |
| 5 | 36789248 | 2023, SN Comput Sci | excluded `E-DERIV` | slice | random_unit_not_stated | - | high |
| 6 | 40335658 | 2025, Eur Radiol | excluded `E-SEG` | unclear | unclear | - | medium |
| 7 | 40194851 | 2025, AJNR | excluded `E-DERIV` :triangular_flag_on_post: | patient | unclear | - | medium |
| 8 | 42489954 | 2026, Brain Topogr | *unreachable* :triangular_flag_on_post: | unclear | unclear | - | low |
| 9 | 39061744 | 2024, Bioengineering | **included** | patient | patient_subject | false | high |
| 10 | 31093705 | 2019, Eur Radiol | **included** :triangular_flag_on_post: | lesion | unclear | false | medium |
| 11 | 36016875 | 2022, Front Pediatr | **included** :triangular_flag_on_post: | slice | slice_or_image | false | medium |
| 12 | 36072854 | 2022, Front Physiol | excluded `E-SEG` | unclear | unclear | - | high |
| 13 | 37222638 | 2024, J Magn Reson Imaging | *unreachable* :triangular_flag_on_post: | patient | unclear | - | low |
| 14 | 40239684 | 2025, Biomed Phys Eng Express | excluded `E-SEG` | unclear | unclear | - | high |
| 15 | 41068276 | 2025, Sci Rep | **included** | unclear | slice_or_image | false | medium |

**1. PMID 36776294** -- *Diagnosis of cervical lymph node metastasis with thyroid carcinoma by deep learning application to CT images* (2023, Front Oncol)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `lesion`: "The 676 lymph nodes were randomly divided into 70% of the training set (73 benign and 401 malignant lymph nodes) and 30% of the test set (30 benign and 172 malignant lymph nodes)." (Results / Sec.2 Materials)
- Split unit `slice_or_image` (disjointness `not_stated`): "For detection, the 676 lymph nodes were randomly divided into 70% of the training set ... and 30% of the testing set ... The training and testing sets for classification were set as same as the detection." (Sec.2 Materials)
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. 'baseline'=0, 'chance'=0, 'majority'=0, 'prevalence'=0, 'constant'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0. The only comparators are other CNN architectures and three radiologists.
- Chance asserted without measurement: `False`
- Positional distribution `no`: Figure 2 shows "The size and shape histograms of LNs" and a "Joint distribution map of the size and shape of LNs" -- size/aspect ratio only, no reference to position within the acquired stack.
- Headline: `accuracy` = 0.96 (internal_held_out, rule `abstract_sentence`) -- binary benign-vs-malignant lymph node classification; AUC 0.894 also reported in the same abstract sentence
- Cohort: private_single_centre (`private`), CT, head_neck; n_patients=196, n_patients_test=None, n_slices_or_images=574, n_positive_reported=`slices_only`
- Input `2D_slice`, label broadcast `na`, uncertainty `none`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `high`, flagged `False`. Two-stage system (Faster R-CNN detection + A-ResNet50-W classification); classification arm coded per amendment A3. 196 patients contribute 676 nodes and the split is by node, so a patient can appear in both partitions; the paper never says otherwise. split_unit coded 'slice_or_image' because the named split unit is the lymph-node ROI, for which the enumeration has no exact level.

**2. PMID 41617832** -- *Multimodal deep learning for laryngeal squamous cell carcinoma staging using CT and laryngoscopy* (2026, Eur Radiol)

- Decision: `unreachable_eligibility_unresolved`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Evaluation unit `unclear`: Abstract reports "AUCs of 0.902 (0.833-0.954) in the internal cohort and 0.888 (0.826-0.944) in the external cohort" for 450 patients; the unit is presumably the patient but the full text could not be reached to confirm.
- Split unit `unclear` (disjointness `not_stated`): "They were divided into training (n = 235), internal validation (n = 101), and external validation (n = 114) cohorts." (Abstract, Materials and Methods) -- the counts are patients, but the paper's own wording on split unit could not be checked.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: NOT EVIDENCED. Abstract states "Three single-modality models (CT-based deep learning [CT-DL], laryngoscopy-based multiple instance learning [L-MIL], and a clinical logistic regression model [CL]) and their combinations were compared" and "Performance was evaluated by AUC" -- this is consistent with a measured clinical-variables-only arm (secondary endpoint S1) but no number for the CL model appears in the abstract, so no sub-flag is set to true.
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `AUC` = 0.888 (external, rule `abstract_multiple_took_external`) -- integrated multimodal model (CL + CT + L), external validation cohort
- Cohort: private_multi_centre (`private`), CT, head_neck; n_patients=450, n_patients_test=114, n_slices_or_images=None, n_positive_reported=`na`
- Input `unclear`, label broadcast `unclear`, uncertainty `ci_unspecified_method`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `low`, flagged `True`. UNREACHABLE. Springer/Eur Radiol returned HTTP 403 from this environment; no PMC record, no Europe PMC full text, no OA copy in OpenAlex, no repository/preprint version found. Rung 5 of the access ladder (ILL / author request with 21-day wait) was not available within this session -- this is a deviation from screen_protocol.md sec.7 and is reported as such. Mixed modality: volumetric CT plus 2D white-light laryngoscopy, with a separately reported CT-DL arm, so amendment A2 would make it eligible. STRONG CANDIDATE FOR S1 (clinical-only logistic regression arm) if the full text is ever obtained.

**3. PMID 39423605** -- *Federated learning and deep learning framework for MRI image and speech signal-based multi-modal depression detection* (2024, Comput Biol Chem)

- Decision: `unreachable_eligibility_unresolved`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Evaluation unit `unclear`: "The ExpAPO-DCNN obtained accuracy, Loss, Root mean Squared error (RMSE), Mean Squared error (MSE), True Negative rate (TNR), and True Positive rate (TPR) of 98.00 %, 0.023, 0.058, 0.240, 97.90 %, and 96.30 %, respectively." (Abstract) -- no unit is named.
- Split unit `unclear` (disjointness `not_stated`): The abstract describes no train/test partition at all: "The processing steps used for this research are pre-processing, feature extraction and detection." (Abstract)
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: NOT EVIDENCED -- full text unreachable. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `accuracy` = 0.98 (unclear, rule `abstract_sentence`) -- ExpAPO-DCNN, fused MRI + speech branches
- Cohort: unclear (`unclear`), MRI, brain; n_patients=None, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`neither`
- Input `unclear`, label broadcast `unclear`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `low`, flagged `True`. UNREACHABLE (Elsevier 403; no PMC, no Europe PMC full text, not OA in OpenAlex). Eligibility genuinely unresolved: the abstract says "feature extraction and detection" for both MRI and speech, so the classifier input may be a derived feature vector (E-DERIV) rather than a spatially resolved image. Also a mixed imaging + non-imaging (speech) fusion, which the codebook's mixed_modality field does not cover; mixed_modality set false because only one imaging modality is used.

**4. PMID 42130124** -- *Development and Validation of an AI-Integrated System for Automated Fracture Detection and Pedicle Puncture Planning in Lumbar OVCF* (2026, Orthop Surg)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `other`: "The automatic fracture recognition model was trained and tested using 1018 vertebral samples derived from 216 L-OVCF patients. On the test dataset, the model demonstrated excellent diagnostic performance, achieving an area under the receiver operating characteristic curve (AUC) of 0.918 (95% CI: 0.885-0.925)" and "Confusion matrix (A) illustrating the model's diagnostic performance on normal and fracture samples (total n = 204)" (Results 3.2, Figure 7). Unit = the individual vertebral body, one of five per scan.
- Split unit `random_unit_not_stated` (disjointness `not_stated`): "After labeling was completed, 10% of all cases were randomly selected to serve as an independent test set, excluded from model training. ... A ten-fold cross-validation method was used during model training." (Sec.2.4 Dataset Partitioning) -- 'cases' is never defined as patients, and the test-set vertebra count (n = 204) is not reconcilable with 10% of 240 cases.
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. The four 'baseline' hits are all "3.1 Baseline Characteristics" / "baseline demographic and clinical characteristics"; 'chance'=0, 'majority'=0, 'prevalence'=0, 'constant'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0. Comparators are nnU-Net (segmentation) and senior surgeons.
- Chance asserted without measurement: `False`
- Positional distribution `no`: "Fracture segments were primarily distributed in the upper lumbar region, consistent with the typical clinical localization pattern of L-OVCF." and Table 2 tabulates fracture counts by vertebral level (L1 57/16, L2 53/12, L3 45/7, L4 29/3, L5 16/2). JUDGEMENT CALL: coded 'no' under the codebook's rule that "Anatomical statements with no reference to position WITHIN the acquired stack ... are 'no'", even though L1-L5 is an ordered cranio-caudal index and the vertebra is the classification unit.
- Headline: `AUC` = 0.918 (internal_held_out, rule `abstract_sentence`) -- binary fractured-vs-normal vertebra, 3D ResNet50 arm; segmentation DSC 0.934 ignored per amendment A3
- Cohort: private_multi_centre (`private`), CT, spine; n_patients=240, n_patients_test=None, n_slices_or_images=1018, n_positive_reported=`slices_only`
- Input `patch_3D`, label broadcast `na`, uncertainty `ci_unspecified_method`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `medium`, flagged `True`. Multi-task (two-stage U-Net segmentation + 3D ResNet50 classification); classification arm coded per amendment A3, Dice/HD ignored. Internal inconsistency: "A database of 240 cases (200 internal and 40 external)" vs "1018 vertebral samples derived from 216 L-OVCF patients"; n_patients recorded as 240. Flagged because positional_distribution_reported is a judgement call (Table 2 = fracture count by vertebral level).

**5. PMID 36789248** -- *Grayscale Image Statistical Attributes Effectively Distinguish the Severity of Lung Abnormalities in CT Scan Slices of COVID-19 Patients* (2023, SN Comput Sci)

- Decision: `excluded` -- `E-DERIV`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Exclusion evidence: "Values of 12 of the 13 statistics derived for each image, omitting the number of pixels in each image, are used as the input variables in this study." (Methods, 'Machine and Deep Learning Algorithms Applied to Grayscale Statistics'). Table of control parameters confirms even the 'CNN' is a 1-D network over those 12 scalars: "Convolutional Neural Network (CNN) 1D Convolutional layer = 5 (filters = 200; size = 3; activation = relu)".
- Evaluation unit `slice`: "513 extract images taken from pulmonary computed tomography (CT) scan slices of 57 individuals" (Abstract) -- coded for completeness only; the paper is excluded.
- Split unit `random_unit_not_stated` (disjointness `not_stated`): "Four distinct K-folds are considered (fourfold involving 75% training: 25% testing splits; fivefold with 80%: 20% splits; tenfold with 90%: 10% splits; and 15-fold with 93%: 7% splits)." (Methods) -- unit never named; images from the same patient are not kept together.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- paper excluded at stage 2 under E-DERIV. not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Chance asserted without measurement: `False`
- Positional distribution `no`
- Headline: `accuracy` = 0.965 (cross_validation, rule `abstract_sentence`) -- 5-class visual score; reported for completeness, paper excluded
- Cohort: private_single_centre (Shiraz, Iran) (`private`), CT, lung; n_patients=57, n_patients_test=None, n_slices_or_images=513, n_positive_reported=`patients_only`
- Input `unclear`, label broadcast `na`, uncertainty `none`, code `none`
- Searches run: not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Confidence `high`, flagged `False`. The word 'CNN' and the CT slices make this look eligible from the title, but the image is discarded before any model: the inputs are 12 grayscale summary statistics per image extract. Exactly the pilot amendment A5 case (cf. PMID 34924987). Reported separately in the flow diagram per screen_protocol.md sec.9.

**6. PMID 40335658** -- *Radiological evaluation and clinical implications of deep learning- and MRI-based synthetic CT for cervical spine injuries* (2025, Eur Radiol)

- Decision: `excluded` -- `E-SEG`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Exclusion evidence: "We sought to evaluate the diagnostic validity of magnetic resonance imaging (MRI)-based synthetic CT (sCT) compared with conventional computed tomography (CT) for cervical spine injuries." and "A panel of five clinicians independently reviewed the images for diagnostic accuracy, lesion characterization (AO Spine classification), and soft tissue trauma." (Abstract, Objective/Methods). The deep-learning task is MRI-to-CT image synthesis; every categorical decision reported (fracture visibility, AO Spine class) is made by human readers, not by a fitted classifier.
- Evaluation unit `unclear`: not applicable -- no fitted classifier; reader-level agreement statistics only ("Inter-reader ICCs were good to excellent").
- Split unit `unclear` (disjointness `na`): not applicable -- no model train/test split is reported for a classification task.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded at stage 1 under E-SEG. not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `sensitivity` = 0.973 (unclear, rule `abstract_sentence`) -- human-reader sensitivity for visualising fractures on sCT; not a model metric
- Cohort: private_multi_centre (`private`), MRI, spine; n_patients=37, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `na`, label broadcast `na`, uncertainty `unclear`, code `none`
- Searches run: not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Confidence `medium`, flagged `False`. Excluded at stage 1: image synthesis + reader study, no supervised classifier. Full text is paywalled (Springer 403) but the abstract is unambiguous on the evaluated task, which is what amendment A4 requires. E-SEG chosen because it is the first applicable code in the listed order (E-SEG explicitly covers 'synthesis'); E-NOCLF would also apply.

**7. PMID 40194851** -- *Severity Classification of Pediatric Spinal Cord Injuries Using Structural MRI Measures and Deep Learning* (2025, AJNR)

- Decision: `excluded` -- `E-DERIV`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Exclusion evidence: "Deep convolutional neural networks (CNNs) were utilized to classify participants into SCI or TD groups and determine their AIS classification based on structural parameters and demographic factors such as age and height." and "These measures were automatically extracted at every vertebral level of the spinal cord by using the spinal cord toolbox." (Abstract, Materials and Methods). The classifier input is a table of cross-sectional area / AP width / RL width plus demographics -- a shape and volumetry descriptor table, not a spatially resolved image.
- Evaluation unit `patient`: "The CNN-based models demonstrated high performance, achieving 96.59% accuracy in distinguishing SCI from TD participants." (Abstract, Results)
- Split unit `unclear` (disjointness `not_stated`): No split is described in the abstract; full text unreachable.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded at stage 1 under E-DERIV. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `accuracy` = 0.9659 (unclear, rule `abstract_sentence`) -- SCI vs typically developing; AIS category accuracy 94.92% also reported
- Cohort: private_single_centre (`private`), MRI, spine; n_patients=61, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `na`, label broadcast `na`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `medium`, flagged `True`. PMC record PMC12633662 exists but is embargoed until 2026-10-01 ('This article will be available in PMC on October 01, 2026'); AJNR site returns 403. Coded from the abstract, which states the CNN input explicitly. Flagged because it is an abstract-only exclusion. Note the demographic inputs (age, height) mean this model is itself close to a non-imaging baseline -- worth revisiting for the paper's discussion once the embargo lifts.

**8. PMID 42489954** -- *Multi-Scale Structural MRI Features Reveal Task-Based Functional Connectivity ... Collaborative Graph Attention Network* (2026, Brain Topogr)

- Decision: `unreachable_eligibility_unresolved`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Evaluation unit `unclear`: "Validated on the Consortium for Neuropsychiatric Phenomics dataset (152 participants, three tasks, four diagnostic groups) ... achieves macro F1 scores of 0.68-0.75 across three psychiatric disorders." (Abstract) -- unit presumably the participant, unconfirmed.
- Split unit `unclear` (disjointness `not_stated`): No split described in the abstract; full text unreachable.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: NOT EVIDENCED -- full text unreachable. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced Abstract notes "CoGA-MTN outperforms single-scale baselines", but those baselines are imaging models and do not count.
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `F1` = 0.75 (unclear, rule `abstract_sentence`) -- macro F1, range 0.68-0.75 across three psychiatric disorders; upper end taken
- Cohort: Consortium for Neuropsychiatric Phenomics (CNP) (`public`), MRI, brain; n_patients=152, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `unclear`, label broadcast `unclear`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `low`, flagged `True`. UNREACHABLE (Springer 403; no PMC/Europe PMC/OA copy). PROVISIONAL READ, NOT ACTED ON: "a dual-branch graph attention network to extract complementary global statistical and local topological features from structural MRI" suggests a graph/derived representation and therefore E-DERIV (pilot amendment A5), but a graph attention network can also be applied over spatially resolved patches, so the abstract does not make the exclusion unambiguous. Per the codebook, stage-1 exclusion requires unambiguity, so this stays unresolved rather than being excluded.

**9. PMID 39061744** -- *Identification of Calculous Pyonephrosis by CT-Based Radiomics and Deep Learning* (2024, Bioengineering)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `patient`: "Finally, a total of 182 patients (84 females/98 males, mean age 53 +/- 13 years ...) were enrolled. These participants were randomly divided into two independent cohorts: training cohort (n = 123) and testing cohort (n = 59)" (Sec.2.1); all AUCs in Tables 3-4 are per patient.
- Split unit `patient_subject` (disjointness `stated_only`): "These participants were randomly divided into two independent cohorts: training cohort (n = 123) and testing cohort (n = 59), based on a 7:3 ratio." (Sec.2.1 Patients)
- Trivial baseline: **clinical_or_demographic_only** | P1 flag: **false**
  - Evidence: TRUE for clinical_or_demographic_only: "The clinical model based on the three clinical risk factors above exhibited an AUC of 0.904 (95% CI 0.837-0.950) with sensitivity and specificity of 0.853 and 0.865, respectively, in the training cohort (Table 3, Figure 3). In the testing cohort, it yielded an AUC of 0.889 (95% CI 0.781-0.956)" (Results 3.1). The three variables are fever, blood neutrophils and urine leukocytes -- no pixels, same metric (AUC). All four P1 sub-flags remain FALSE; searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available ('baseline'=0, 'chance'=0, 'majority'=0, 'prevalence'=0, 'constant'=0, 'trivial'=0, 'metadata'=0, 'permut'=0, 'slice index'=0).
- Chance asserted without measurement: `False`
- Positional distribution `no`: The only position-related sentence concerns HU measurement: "previous studies measured the HU ... in the single slice with the maximal collecting system surface area, whereas we measured the average HU across all slices" (Discussion) -- not a distribution of the label along the slice axis.
- Headline: `AUC` = 0.967 (internal_held_out, rule `abstract_multiple_took_external`) -- comprehensive clinical machine-learning model (radiomics + clinical), testing cohort
- Cohort: private_single_centre (`private`), CT, kidney; n_patients=182, n_patients_test=59, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `patch_3D`, label broadcast `na`, uncertainty `ci_unspecified_method`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `high`, flagged `False`. COUNTS TOWARD SECONDARY ENDPOINT S1, NOT P1: the clinical-only arm is a clinical-variables nomogram comparator (fever, neutrophils, urine leukocytes), which the primary endpoint deliberately excludes. Worth noting for the paper: the clinical-only model (AUC 0.889) beat the 3D-CNN on the same test set (AUC 0.599), and the CNN's training AUC was 1.000.

**10. PMID 31093705** -- *Deep learning for liver tumor diagnosis part II: CNN interpretation using radiologic imaging features* (2019, Eur Radiol)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `lesion`: "A test set of 60 lesions was labeled with the most prominent imaging features in each image (1-4 features per lesion). This test set was the same as that used to conduct the reader study in Part I." (Methods)
- Split unit `unclear` (disjointness `not_stated`): The paper never describes how the data were partitioned; it defers to Part I: "Characteristics of the 296 patients included in this study are described in Part I of this article series. CNN model classification performance is also described in detail in Part I." (Results)
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. 'baseline'=0, 'chance'=0, 'majority'=0, 'prevalence'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0; the single 'constant' hit is "constant across all phases" (Discussion, about imaging features).
- Chance asserted without measurement: `False`
- Positional distribution `no`: Table 2 reports "Frequency in the test set" per radiological feature (e.g. arterial phase hyperenhancement 19/60) -- class frequency, not position along the slice axis.
- Headline: `other` = 0.765 (internal_held_out, rule `abstract_sentence`) -- positive predictive value for identifying the correct radiological features in each test lesion (76.5 +/- 2.2%); sensitivity 82.9% also given
- Cohort: private_single_centre (`private`), MRI, liver; n_patients=296, n_patients_test=None, n_slices_or_images=494, n_positive_reported=`na`
- Input `3D_volume`, label broadcast `na`, uncertainty `sd_across_folds`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available -- NOTE: 'Supplement 1' is referenced in the text and was not obtainable, so the negative is evidenced on the main text only.
- Confidence `medium`, flagged `True`. JUDGEMENT CALL: this is Part II of a two-part series; the six-class lesion classifier itself was fitted and evaluated in Part I, and Part II evaluates a post-hoc feature-identification algorithm over that pre-trained CNN. It still satisfies I2 (a categorical label -- feature present/absent -- is assigned to a lesion image) and I4 (PPV, sensitivity), so it is included. Not E-DUP: Part I is not in this sample and reports different experiments. Uncertainty is SD over 20 retraining iterations: "the model obtained a PPV of 76.5 +/- 2.2% and Sn of 82.9 +/- 2.6% ... over 20 iterations".

**11. PMID 36016875** -- *An in-depth discussion of cholesteatoma, middle ear inflammation, and Langerhans cell histiocytosis of the temporal bone* (2022, Front Pediatr)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `slice`: "A random selection of 85% of the dataset (n = 2,070) was used during the validation process ... The remaining 15% of the data (n = 388) were stored and could be used to evaluate the performance of the model after the training was complete." (Methods) -- n = 2,070 / 388 are axial CT sections, not patients (only 119 patients exist; "The total number of scans performed was 2,588").
- Split unit `slice_or_image` (disjointness `not_stated`): "A random selection of 85% of the dataset (n = 2,070) was used during the validation process ... The remaining 15% of the data (n = 388)" (Methods). The abstract contradicts this with "70% of cases for training and 30% of cases for validation".
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. 'baseline'=0, 'chance'=0, 'majority'=0, 'constant'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0; the two 'prevalence' hits refer to disease prevalence. The only comparator is the clinicians' own diagnoses.
- Chance asserted without measurement: `False`
- Positional distribution `no`: "The scans were performed on sections where lesions were present, and the number of axial CT sections per scan ranged from 30 to 50." (Methods) -- a scan-length statement, not a distribution of the label along the slice axis.
- Headline: `AUC` = 0.98 (internal_held_out, rule `abstract_sentence`) -- one-vs-rest ROC for the cholesteatoma class (model 0.98 vs physician 0.91); LCH 0.99, MEI 0.99
- Cohort: private_single_centre (`private`), CT, head_neck; n_patients=119, n_patients_test=None, n_slices_or_images=2588, n_positive_reported=`patients_only`
- Input `2D_slice`, label broadcast `true`, uncertainty `ci_unspecified_method`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `medium`, flagged `True`. A textbook slice-level design: one histology-proven diagnosis per patient is broadcast to every 224x224 axial section ("Each of the 119 patients was matched with one-third of the disease labels"), 2,588 sections from 119 patients are split at the image level, and near-perfect image-level accuracies (0.99) are reported with no patient-level aggregation. Flagged because the abstract (70/30 of 'cases') and the Methods (85/15 of n=2,070/388 images) disagree on the split; coded on the Methods numbers, which can only be images.

**12. PMID 36072854** -- *COVID-19 CT image segmentation method based on swin transformer* (2022, Front Physiol)

- Decision: `excluded` -- `E-SEG`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Exclusion evidence: "we propose a new method to improve U-Net for lesion segmentation in the chest CT images of COVID-19 patients" and "this method achieved significant performance gain, in which the mean pixel accuracy is 87.62%, mean intersection over union is 80.6%, and dice similarity coefficient is 88.27%" (Abstract). The word 'classify' in "selected to classify, identify, and segment the background area, lung area, ground glass opacity, and lung parenchyma" refers to semantic (per-pixel) classes; no image- or patient-level categorical decision is evaluated.
- Evaluation unit `unclear`: not applicable -- per-pixel segmentation metrics only.
- Split unit `unclear` (disjointness `na`): not applicable -- excluded; no classification split coded.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded under E-SEG. not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (unclear, rule `other`) -- mean pixel accuracy 87.62% / mIoU 80.6% / DSC 88.27% -- segmentation metrics, not classification
- Cohort: CC-CCII (`public`), CT, lung; n_patients=150, n_patients_test=None, n_slices_or_images=750, n_positive_reported=`neither`
- Input `2D_slice`, label broadcast `na`, uncertainty `none`, code `none`
- Searches run: not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Confidence `high`, flagged `False`. Clean E-SEG. Note 'mean pixel accuracy' is an accuracy-named metric on a segmentation task -- exactly the trap the codebook warns about (3 of 10 pilot papers did this).

**13. PMID 37222638** -- *Prenatal Diagnosis of Placenta Accreta Spectrum Disorders: Deep Learning Radiomics of Pelvic MRI* (2024, J Magn Reson Imaging)

- Decision: `unreachable_eligibility_unresolved`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Evaluation unit `patient`: "324 pregnant women (mean age, 33.3 years) suspected PAS (170 training and 72 validation from institution 1, 82 external validation from institution 2) with clinicopathologically proved PAS (206 PAS, 118 non-PAS)" (Abstract, Population) -- counts are pregnancies/patients.
- Split unit `unclear` (disjointness `not_stated`): "170 training and 72 validation from institution 1, 82 external validation from institution 2" (Abstract) -- the unit is the patient but the paper's own split wording could not be checked.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: NOT EVIDENCED -- full text unreachable. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced Abstract states "The MRI-based DLR model had a higher area under the curve than the clinical model in three datasets (0.880 vs. 0.741, 0.861 vs. 0.772, 0.852 vs. 0.675, respectively)", where 'clinical model' = "different clinical characteristics between PAS and non-PAS groups". That reads as a measured clinical-variables-only arm on the same metric (secondary endpoint S1), but because eligibility itself is unresolved the sub-flags are left 'unclear' rather than set.
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `AUC` = 0.852 (external, rule `abstract_multiple_took_external`) -- MRI-based deep learning radiomics model, external validation dataset
- Cohort: private_multi_centre (`private`), MRI, other (placenta / pelvis); n_patients=324, n_patients_test=82, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `unclear`, label broadcast `unclear`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `low`, flagged `True`. UNREACHABLE (Wiley 403 on both the article page and the OA pdfdirect link OpenAlex reports; no PMC record). Second strong S1 candidate in the overlap set if the full text is obtained.

**14. PMID 40239684** -- *DCA-U-Net: segmentation of laser-induced thermal damage regions in mouse skin OCT images* (2025, Biomed Phys Eng Express)

- Decision: `excluded` -- `E-SEG`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Exclusion evidence: "we propose an efficient and lightweight segmentation model, Dilated ConvNeXT Attention U-Net (DCA-U-Net), based on U-Net" and "Experimental results on two different sections of mouse skin laser thermal damage Optical Coherence Tomography (OCT) datasets show that our model has better segmentation performance" (Abstract).
- Evaluation unit `unclear`: not applicable -- segmentation only.
- Split unit `unclear` (disjointness `na`): not applicable -- excluded at stage 1.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded at stage 1 under E-SEG. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (unclear, rule `other`) -- segmentation performance; no numeric classification metric in the abstract
- Cohort: private (mouse skin OCT) (`private`), OCT, other (skin, murine); n_patients=None, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`na`
- Input `na`, label broadcast `na`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `high`, flagged `False`. Two exclusion codes apply (E-SEG and E-NONMED: mouse skin, preclinical animal-only). E-SEG recorded because the codebook requires the FIRST applicable code in the listed order. Full text paywalled (IOP) but the abstract is unambiguous.

**15. PMID 41068276** -- *IoMT driven Alzheimer's prediction model empowered with transfer learning and explainable AI* (2025, Sci Rep)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `unclear`: The paper calls the same objects both images and scans and never defines either: "The publicly available Kaggle Alzheimer's MRI dataset, comprising 33,984 images across four classes" (Abstract) vs "the dataset size (33,984 MRI scans) provided sufficient samples per class" (Methods, Prediction layer). Table 2 is headed "Distribution of MRI images by dataset split (70/15/15, before augmentation)". Coded 'unclear' per the codebook's ambiguous-case rule.
- Split unit `slice_or_image` (disjointness `not_stated`): "The augmented dataset was divided into training (70%), validation (15%), and testing (15%) subsets." (Methods, Prediction layer) with "Table 2 Distribution of MRI images by dataset split (70/15/15, before augmentation). Class Original Samples Train (70%) Validation (15%) Test (15%) Mild Demented 8960 6272 1344 1344 ..." -- the split is over images.
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. The two 'baseline' hits are an ablation row -- "Baseline ResNet152 (no augmentation, no XAI) 92.1" (Table 4) -- which is an imaging model and is explicitly excluded by the codebook's does_not_count list. 'chance' hit is "a scant chance of regaining health"; 'majority'=0, 'prevalence'=0, 'constant'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0.
- Chance asserted without measurement: `False`
- Positional distribution `no`: No statement about where in the volume the classes fall; the only related text is a literature note that another study "Focuses only on mid-sagittal slices" (Table of related work).
- Headline: `accuracy` = 0.9777 (internal_held_out, rule `abstract_sentence`) -- 4-class (Non-, Very Mild, Mild, Moderate Demented), ResNet152-TL-XAI
- Cohort: Kaggle Alzheimer's MRI dataset (`public`), MRI, brain; n_patients=None, n_patients_test=None, n_slices_or_images=33984, n_positive_reported=`slices_only`
- Input `unclear`, label broadcast `unclear`, uncertainty `none`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `medium`, flagged `False`. No patient count exists anywhere in the paper -- n_patients is NULL, which is itself a finding. The dataset is a Kaggle re-post whose provenance and subject structure are never described, so subject-level disjointness between the 70/15/15 image splits cannot be assessed. A conditional WGAN generates synthetic training images; the authors state "The validation and testing subsets remained unchanged to ensure unbiased model evaluation and to prevent data leakage from synthetic samples."

### Batch C (positions 59-79)

| pos | PMID | year, venue | decision | eval unit | split unit | P1 | conf |
|---|---|---|---|---|---|---|---|
| 59 | 36229338 | 2022, J Optom | **included** :triangular_flag_on_post: | volume_or_scan_not_patient | patient_subject | false | medium |
| 60 | 34324463 | 2022, Invest Radiol | excluded `E-SEG` | unclear | unclear | - | medium |
| 61 | 32714766 | 2020, Adv Sci | **included** | patient | external_cohort_only | false | medium |
| 62 | 41547664 | 2026, EJNMMI Phys | excluded `E-SEG` | unclear | unclear | - | high |
| 63 | 34976558 | 2020, IEEE Access | **included** :triangular_flag_on_post: | unclear | random_unit_not_stated | false | medium |
| 64 | 42153825 | 2026, Radiology | *unreachable* :triangular_flag_on_post: | patient | unclear | - | low |
| 65 | 35872945 | 2022, Comput Math Methods Med | **included** :triangular_flag_on_post: | volume_or_scan_not_patient | patient_subject | false | low |
| 66 | 34476208 | 2021, Front Oncol | excluded `E-DERIV` | patient | site_or_centre | - | high |
| 67 | 38082902 | 2023, EMBC | excluded `E-NOCLF` :triangular_flag_on_post: | patient | unclear | - | medium |
| 68 | 34603980 | 2021, Quant Imaging Med Surg | excluded `E-SEG` :triangular_flag_on_post: | other | no_held_out_test_set | - | medium |
| 69 | 36762417 | 2023, Acta Radiol | excluded `E-NOCLF` :triangular_flag_on_post: | patient | unclear | - | medium |
| 70 | 42367846 | 2026, bioRxiv | excluded `E-PROJ` :triangular_flag_on_post: | other | random_unit_not_stated | - | medium |
| 71 | 40093990 | 2025, EClinicalMedicine | **included** | patient | patient_subject | false | high |
| 72 | 30734849 | 2019, Eur Radiol | excluded `E-DERIV` :triangular_flag_on_post: | patient | unclear | - | medium |
| 73 | 40081198 | 2025, Comput Methods Programs Biomed | *unreachable* :triangular_flag_on_post: | unclear | unclear | - | low |
| 74 | 39384719 | 2025, J Imaging Inform Med | excluded `E-SEG` | unclear | unclear | - | high |
| 75 | 38298725 | 2024, Heliyon | **included** :triangular_flag_on_post: | volume_or_scan_not_patient | patient_subject | false | medium |
| 76 | 36539234 | 2023, J Am Med Inform Assoc | excluded `E-NONMED` | other | unclear | - | high |
| 77 | 38591974 | 2024, Radiology | *unreachable* :triangular_flag_on_post: | lesion | unclear | - | low |
| 78 | 35378943 | 2022, J Healthc Eng | **included** :triangular_flag_on_post: | lesion | unclear | false | low |
| 79 | 34888191 | 2021, Quant Imaging Med Surg | excluded `E-SEG` | unclear | unclear | - | high |

**59. PMID 36229338** -- *Development and validation of a pixel wise deep learning model to detect cataract on swept-source OCT images* (2022, J Optom)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `volume_or_scan_not_patient`: "The Cataract Fraction (CF) defined as the number of pixels classified as 'Cataract' divided by the total number of pixels representing the lens was calculated for each image. CF was averaged over all radial scans available for each eye." (Metrics and statistics) -- the ROC unit is the eye (validation set: 132 clear + 89 cataract eyes from 114 patients), and eyes are never aggregated to patients.
- Split unit `patient_subject` (disjointness `stated_and_checked`): "The first set of patients constituted the development set which was randomly split into a training set and a test set with an 80%/20% ratio at the patient level, not at the image level. Both eyes of each patient were included in the same set." (Methods, Study design)
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. Every one of the 14 terms returned 0 hits in the full text ('baseline'=0, 'chance'=0, 'random' only in data augmentation and in the split sentence, 'majority'=0, 'prevalence'=0, 'constant'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'position'=0, 'slice index'=0, 'permut'=0).
- Chance asserted without measurement: `False`
- Positional distribution `no`: No statement about where along the radial scan sequence cataract appears; CF is averaged over all scans of an eye without position stratification.
- Headline: `AUC` = 0.98 (internal_held_out, rule `abstract_sentence`) -- cataract vs clear lens, per eye, validation set; optimal CF threshold 0.14, sensitivity 94.4%, specificity 94.7%
- Cohort: private_single_centre (Rothschild Foundation, Paris) (`private`), OCT, other (crystalline lens, anterior segment); n_patients=157, n_patients_test=114, n_slices_or_images=1830, n_positive_reported=`patients_and_slices`
- Input `2D_slice`, label broadcast `true`, uncertainty `none`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `medium`, flagged `True`. TWO JUDGEMENT CALLS, both flagged. (1) Eligibility: the model is pixel-wise (segmentation-like), but a categorical eye-level decision IS evaluated with ROC AUC / sensitivity / specificity, so amendment A3 makes it an include with the classification arm coded and Dice ignored. (2) I3: SS-OCT of the crystalline lens acquired as a set of radial B-scans -- a volumetric acquisition with a cross-section axis, but radial rather than a parallel stack. (3) label_broadcast_to_slices=true: "All lens pixels of a given image were labeled 'Normal' for the clear lens patients or 'Cataract' for the cataract patients" -- the patient-level diagnosis is broadcast to every pixel of every image. This is one of only two included papers in the whole assignment with an explicit patient-level split statement.

**60. PMID 34324463** -- *Can Deep Learning Replace Gadolinium in Neuro-Oncology?: A Reader Study* (2022, Invest Radiol)

- Decision: `excluded` -- `E-SEG`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Exclusion evidence: "A deep network was trained to process the precontrast and low-dose sequences to predict 'virtual' surrogate images for contrast-enhanced T1." and "The discrepancies between the predicted virtual images and the standard-dose MRIs were qualitatively and quantitatively evaluated using both automated voxel-wise metrics and a reader study, where 2 radiologists graded image qualities and marked all visible enhancing lesions." (Abstract, Materials and Methods). The deep-learning task is image synthesis; the reported "area under the curve of 96.4% +/- 3.1%" is voxel-wise, and all lesion-level decisions are made by human readers.
- Evaluation unit `unclear`: Voxel-wise metrics plus human-reader lesion marking; no fitted classifier assigns a categorical label to an imaging unit.
- Split unit `unclear` (disjointness `na`): "A total of 145 patients were included: 107 formed the training sample ... and 38 the separate test sample" (Abstract) -- a synthesis-model split, not a classification split.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded at stage 1 under E-SEG. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (internal_held_out, rule `other`) -- voxel-wise AUC 96.4% for the synthesis task; reader-study F1 63-88% for lesion detection by humans
- Cohort: private_single_centre (Gustave Roussy) (`private`), MRI, brain; n_patients=145, n_patients_test=38, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `na`, label broadcast `na`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `medium`, flagged `False`. Excluded at stage 1: gadolinium-dose-reduction image synthesis plus a radiologist reader study. E-SEG explicitly covers 'synthesis'. Full text paywalled (Wolters Kluwer / Invest Radiol; no PMC, no OA copy), but the abstract states the model's task and the evaluation design unambiguously.

**61. PMID 32714766** -- *Generalizable, Reproducible, and Neuroscientifically Interpretable Imaging Biomarkers for Alzheimer's Disease* (2020, Adv Sci)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `patient`: "In total, 1832 subjects from our in-house multi-center database (n = 716) and the Alzheimer's Disease Neuroimaging Initiative (ADNI) database (n = 1116) were employed in this study" and "The input is the normalized 3D gray matter density image and the output is a probability for each individual obtained by a soft-max classifier" (Results 2.1; Methods).
- Split unit `external_cohort_only` (disjointness `stated_only`): "For the AD versus normal control (NC) classification, we conducted cross-validations between the ADNI and the in-house databases. For each strategy, one of the two databases was used as the training set and the other as the testing set." (Results 2.1). Leave-centre-out cross-validation within the in-house database is also used; the headline 92.1% comes from a within-ADNI tenfold cross-validation whose split unit is not named.
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. Both 'baseline' hits mean the baseline visit ("using only the baseline data to predict whether or not an MCI subject would convert", "These databases contain baseline brain MR imaging"); 'chance'=0, 'random'=0, 'majority'=0, 'prevalence'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0. The three 'position' hits are all about voxel spatial position in the attention module.
- Chance asserted without measurement: `False`
- Positional distribution `no`: Attention maps are reported over brain regions, not over slice position: "The attention score map ... indicating the discriminative power of various brain regions for AD diagnosis" (Figure 1B).
- Headline: `accuracy` = 0.92 (cross_validation, rule `abstract_sentence`) -- "an accuracy up to 92%" = 92.1% from tenfold cross-validation within ADNI for AD vs NC (AUC 0.941)
- Cohort: in-house multi-centre database; ADNI (`mixed`), MRI, brain; n_patients=1832, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `3D_volume`, label broadcast `na`, uncertainty `none`, code `public_link_stated`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available -- Supporting Information (Figures S1-S5, Table S1) referenced but not obtainable; negative evidenced on the main text.
- Confidence `medium`, flagged `False`. Well-designed cross-database validation (each database in turn as the external test set) plus leave-centre-out CV. split_unit coded 'external_cohort_only' for the design the paper foregrounds; note that the headline number the abstract quotes ("up to 92%") is the LEAST external of the four validations. Code stated: "The code can be downloaded at https://github.com/YongLiuLab" (Methods) -- link not verified.

**62. PMID 41547664** -- *Clinical validation of a unified data-driven respiratory motion correction technique in 18F-FDG PET/CT* (2026, EJNMMI Phys)

- Decision: `excluded` -- `E-SEG`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Exclusion evidence: "This study aimed to prospectively evaluate the clinical utility of the unified data-driven respiratory motion correction (uRMC) algorithm utilizing deep learning neural networks for diagnosing upper abdominal lesions." and "For each patient, a 3-point Likert scale was employed to evaluate the overall motion artifacts, and up to 3 lesions were selected to evaluate PET-CT alignment and lesion distortion" (Purpose; Methods). The deep-learning task is motion-corrected image reconstruction; every reported outcome is an image-quality score, a SUVmax/MTV/TBR measurement or a physician's lesion count.
- Evaluation unit `unclear`: not applicable -- no fitted classifier; reader Likert scores and semi-quantitative measurements only.
- Split unit `unclear` (disjointness `na`): not applicable -- no model was trained in this study; a vendor algorithm was evaluated.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded under E-SEG. not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (unclear, rule `other`) -- no classification metric reported; outcomes are Likert image quality, SUVmax, MTV, TBR, HV_ratio and lesion counts
- Cohort: private_single_centre (Xijing Hospital) (`private`), PET_CT, abdomen_general; n_patients=100, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`neither`
- Input `na`, label broadcast `na`, uncertainty `unclear`, code `none`
- Searches run: not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Confidence `high`, flagged `False`. E-SEG applies first ('reconstruction'); E-NOCLF and E-NOMET would also apply. A clear instance of the query's imprecision: 'deep learning' + 'PET/CT' + 'detection' + 'accuracy' all appear, but no classifier exists.

**63. PMID 34976558** -- *Novel Feature Selection and Voting Classifier Algorithms for COVID-19 Classification in CT Images* (2020, IEEE Access)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `unclear`: "The first is the COVID-19-dataset which has 334 CT images containing clinical findings of COVID-19. While the second is the non-COVID-19-dataset that has extra 794 CT images with clinical cases that have no COVID-19." (Sec.III.A Datasets). The paper says only 'CT images' and never states whether an image is a slice or a volume, and never reports a patient count -- coded 'unclear' per the codebook's ambiguous-case rule.
- Split unit `random_unit_not_stated` (disjointness `not_stated`): The only partition statement in the entire paper is a row in an algorithm-configuration table: "K-fold cross-validation 10" (Table 5, Proposed SFS-Guided WOA Algorithm Configuration). Searches for 'training set', 'test set', 'testing set', 'validation set', 'hold-out', 'partition' return no data-partition description.
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. 'baseline'=0, 'prevalence'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0. The 5 'majority' hits are the ensemble's "majority-vote classifier", not a majority-class baseline; the 3 'chance' hits are "increases the chance that individual classifiers ... show significant discrepancies"; the 4 'constant' hits are algorithm constants ("Acceleration constants").
- Chance asserted without measurement: `False`
- Positional distribution `no`: The 30 'position' hits are all optimizer particle positions (PSO/WOA); there is no statement about the position of findings within a scan.
- Headline: `AUC` = 0.995 (cross_validation, rule `abstract_sentence`) -- PSO-Guided-WOA voting classifier; the text calls it "an AUC with binary predictions (balanced accuracy) result of 0.995"
- Cohort: COVID-CT-Dataset (Zhao et al., arXiv:2003.13865); non-COVID-19 CT set (`public`), CT, lung; n_patients=None, n_patients_test=None, n_slices_or_images=1128, n_positive_reported=`slices_only`
- Input `unclear`, label broadcast `unclear`, uncertainty `none`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `medium`, flagged `True`. Included because the pipeline starts from images: AlexNet extracts features from the CT images (so the image does reach a model) and the selected features feed the voting classifier -- this is NOT the E-DERIV case, where the image is discarded before any model. Flagged because the paper never describes a data partition in prose, never reports patient counts, and the images are figure crops harvested from COVID-19 papers (medRxiv/bioRxiv/NEJM/JAMA/Lancet), so subject structure is unknowable.

**64. PMID 42153825** -- *Interpretable MRI-based Multiparametric Radiomics for Preoperative Prediction of CMS4 Colorectal Cancer* (2026, Radiology)

- Decision: `unreachable_eligibility_unresolved`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Evaluation unit `patient`: "This study included 253 patients (median age, 63 years; IQR, 55-69 years; 163 men). The merged MRC4s ... achieved areas under the receiver operating characteristic curve (AUCs) of 0.85 (95% CI: 0.63, 1.00) in the internal and 0.84 (95% CI: 0.73, 0.95) in the external test sets" (Abstract, Results).
- Split unit `unclear` (disjointness `not_stated`): "A subgroup of patients was randomly divided into a training and an internal test set, whereas another subgroup constituted the external test set." (Abstract, Materials and Methods) -- the unit of the random division is not named in the abstract.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: NOT EVIDENCED -- full text unreachable. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `AUC` = 0.84 (external, rule `abstract_multiple_took_external`) -- merged MRI radiomics CMS4 score (MRC4s), external test set
- Cohort: private_multi_centre (`private`), MRI, other (colorectum / rectum); n_patients=253, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `unclear`, label broadcast `unclear`, uncertainty `ci_unspecified_method`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `low`, flagged `True`. UNREACHABLE despite being marked CC BY: pubs.rsna.org returns HTTP 403 to every client tried, there is no PMC record, and Europe PMC has no full text. ELIGIBILITY IS GENUINELY UNRESOLVED AND THE REASON IS INTERESTING: the headline model is a radiomics feature-vector model (candidate E-DERIV under amendment A5), but the paper also implements ResNet50/VGG16/DenseNet201 "as comparators" (AUC range 0.70-0.75) which presumably take images. If those image models exist, the paper is eligible and only the headline arm is radiomics; if they were run on feature vectors, E-DERIV applies. The abstract cannot settle it, so exclusion at stage 1 would violate the codebook's unambiguity requirement.

**65. PMID 35872945** -- *Prediction of Conversion from CIS to Clinically Definite Multiple Sclerosis Using CNNs* (2022, Comput Math Methods Med)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `volume_or_scan_not_patient`: "As previously mentioned, scans from 9 of the patients were used for testing purposes. A range of metrics was applied to each of these scans, and the averages were calculated (see Table 2)." (Sec.3.2 Results) -- each of the 9 test patients contributes two scans (baseline and one-year), and the metrics are averaged over scans, not aggregated to patients.
- Split unit `patient_subject` (disjointness `stated_only`): "The scans of 40 of these patients were then used to train the algorithm, with this training dataset then being randomly divided into subsets for training and validation at a ratio of 80 : 20. Scans from the remaining 9 patients were used for testing purposes." (Sec.2.1)
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. 'baseline'=0, 'chance'=0, 'majority'=0, 'constant'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'position'=0, 'slice index'=0, 'permut'=0; the single 'prevalence' hit is disease prevalence. Comparators are other published imaging models (random forest of Zhang et al.).
- Chance asserted without measurement: `False`
- Positional distribution `no`: 'position' returns 0 hits in the full text; nothing is said about where along the 80-slice stack the informative signal lies.
- Headline: `accuracy` = 0.888 (internal_held_out, rule `abstract_sentence`) -- prediction of CIS-to-CDMS conversion; AUC 91% also reported in the same abstract sentence
- Cohort: private (PRISMA and VARIO scanner cohorts, Univ. of Newcastle); ADNI used for pretraining only (`mixed`), MRI, brain; n_patients=49, n_patients_test=9, n_slices_or_images=7360, n_positive_reported=`patients_only`
- Input `2D_slice`, label broadcast `true`, uncertainty `none`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `low`, flagged `True`. RETRACTED PUBLICATION (PubMed publication type 'Retracted Publication'). The codebook has NO RULE for retractions: E-TYPE covers reviews, editorials, errata and protocols but not retracted primary research, and screen_protocol.md sec.4.1 forbids dropping papers for being under-described. Coded as INCLUDED and flagged for adjudication; the protocol authors should decide whether to add a retraction rule to the changelog (2 of my 36 records are retracted: 35872945 and 35378943). Substantively: 49 patients / 98 volumetric scans / 7,360 slices, patient-level split (a genuine strength), but the patient-level conversion label is broadcast to every slice and the paper mixes classification metrics with a segmentation DSC formula, so the evaluation unit is coded from the one explicit sentence about the test set.

**66. PMID 34476208** -- *MRI-Radiomics Prediction for Cytokeratin 19-Positive Hepatocellular Carcinoma: A Multicenter Study* (2021, Front Oncol)

- Decision: `excluded` -- `E-DERIV`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Exclusion evidence: "A total of 968 radiomics features were extracted from preoperative multisequence MR images. The maximum relevance minimum redundancy algorithm was applied for feature selection. Multiple logistic regression, support vector machine, random forest, and artificial neural network (ANN) algorithms were used to construct the radiomics model" (Abstract) and "We established three radiomics-based models, i.e., T2, DWI, and combined (both T2 and DWI radiomics features), using an artificial neural network (ANN) algorithm (hidden size = 2 ...)" (Methods). Searches for 'convolutional' and 'deep learning' return ZERO hits in the full text: no model ever sees a pixel.
- Evaluation unit `patient`: "A multicenter and time-independent cohort of 257 patients were retrospectively enrolled (training cohort, n = 143; validation cohort A, n = 75; validation cohort B, n = 39)." (Abstract) -- coded for completeness only; the paper is excluded.
- Split unit `site_or_centre` (disjointness `stated_only`): "FAH-ZJU, from which the majority of patients were enrolled, was set as the training cohort, and SLH and LSCH were the independent external validation cohorts." (Methods)
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded as an endpoint (paper excluded under E-DERIV), but recorded for transparency: the paper DOES report a measured clinical-variables-only arm -- "The predictive ability of the clinical model was worse than that of the radiomics-based model in the training and independent validation cohorts [FAH-ZJU: 0.737 (95% CI: 0.654-0.819); SLH: 0.560 (95% CI: 0.425-0.694); LSCH: 0.585 (95% CI: 0.313-0.857)]" (Results, Clinical Classifier Construction), built from "sex, lymph node status, total bilirubin, direct bilirubin, alpha-fetoprotein, and log alpha-fetoprotein". This would have been an S1 hit had the paper been eligible. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Chance asserted without measurement: `False`
- Positional distribution `no`: 'position' returns 0 hits in the full text.
- Headline: `AUC` = 0.79 (external, rule `abstract_sentence`) -- ANN combined radiomics classifier, validation cohort B; reported for completeness, paper excluded
- Cohort: private_multi_centre (FAH-ZJU, SLH, LSCH) (`private`), MRI, liver; n_patients=257, n_patients_test=114, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `na`, label broadcast `na`, uncertainty `ci_unspecified_method`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `high`, flagged `False`. Entered the frame on 'artificial neural network' + MRI + classification + AUROC, but the ANN runs on a 968-dimensional radiomics vector, so the image is discarded: E-DERIV (pilot amendment A5). Counted separately in the flow diagram per screen_protocol.md sec.9.

**67. PMID 38082902** -- *The Influence of Brain MRI Defacing Algorithms on Brain-Age Predictions via 3D CNNs* (2023, EMBC)

- Decision: `excluded` -- `E-NOCLF`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Exclusion evidence: "Here, we evaluated 4 popular defacing methods to identify the effects of defacing on 'brain age' prediction - a common benchmarking task of predicting a subject's chronological age from their 3D T1-weighted brain MRI." and "Significant differences were present when comparing average per-subject error rates between algorithms in both the defaced brain data and the extracted facial tissue." (Abstract). The task is continuous age regression; the abstract reports error rates and no categorical label or classification metric.
- Evaluation unit `patient`: "average per-subject error rates" (Abstract) -- coded for completeness only.
- Split unit `unclear` (disjointness `not_stated`): No split described in the abstract; full text unreachable.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded at stage 1 under E-NOCLF. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (unclear, rule `other`) -- brain-age prediction error; no classification metric in the abstract
- Cohort: unclear (`unclear`), MRI, brain; n_patients=None, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`na`
- Input `3D_volume`, label broadcast `na`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `medium`, flagged `True`. UNREACHABLE full text (IEEE Xplore returns HTTP 418/403 even for the 'free pdf' link Europe PMC advertises; no PMC record). Excluded on the abstract because brain-age prediction is regression. Flagged: a supplementary categorical analysis cannot be ruled out from the abstract alone. SUBSTANTIVELY RELEVANT TO OUR PAPER even though excluded -- "We obtained better performance in age prediction when using the extracted face portion alone than images of the brain", i.e. a non-target-region shortcut beating the intended signal.

**68. PMID 34603980** -- *Fully automated guideline-compliant diameter measurements of the thoracic aorta on ECG-gated CT angiography* (2021, Quant Imaging Med Surg)

- Decision: `excluded` -- `E-SEG`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Exclusion evidence: "The DL-algorithm prototype detected aortic landmarks (deep reinforcement learning) and segmented the lumen of the thoracic aorta (multi-layer convolutional neural network). It performed measurements according to AHA-guidelines and created visual outputs." (Abstract, Methods). The deep-learning task is landmark detection plus segmentation plus diameter measurement; agreement with radiologists is the outcome ("2,778/3,192 (87.0%) of DL-algorithm's measurements were coherent").
- Evaluation unit `other`: "Table 3. Measurement and classification accuracy by location" -- per measurement location (3,192 locations across 405 exams); coded for completeness only.
- Split unit `no_held_out_test_set` (disjointness `na`): The DL prototype was a pre-existing vendor algorithm evaluated on 405 consecutive exams; no training/test partition of these data is described.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded under E-SEG. not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (unclear, rule `other`) -- 87.0% of measurement locations coherent with radiologists; mean differences in mm
- Cohort: private_single_centre (Basel) (`private`), CT, cardiac; n_patients=371, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`neither`
- Input `3D_volume`, label broadcast `na`, uncertainty `ci_unspecified_method`, code `none`
- Searches run: not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Confidence `medium`, flagged `True`. BORDERLINE, flagged. A dilatation category IS derived and evaluated -- "an analysis of classification change (dilatation versus no dilatation) between DL-measurements and original reports was performed", "An aneurysm was misclassified in 34/399 cases (8.5%)" -- but that category comes from thresholding a measured diameter (>=45 mm / >=40 mm), not from a fitted supervised classifier, so I2 is not met. E-SEG is the first applicable code (segmentation / landmark localisation). A second screener could plausibly argue for inclusion of the dilatation arm.

**69. PMID 36762417** -- *Deep learning-extracted CT imaging phenotypes predict response to total resection in colorectal cancer* (2023, Acta Radiol)

- Decision: `excluded` -- `E-NOCLF`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Exclusion evidence: "To evaluate the prognostic value of preoperative computed tomography (CT) image texture features and deep learning self-learning high-throughput features (SHF) on postoperative overall survival" and "The overall recognition ability and accuracy of CoxPH and N-MTLR model were evaluated by C-index and Integrated Brier Score (IBS)." (Abstract, Purpose / Material and Methods). The outcome is overall survival; the only metrics reported are C-index and IBS, neither of which is in the I4 list, and the high/low risk grouping is a threshold on a continuous RAD score rather than a fitted categorical classifier.
- Evaluation unit `patient`: "The dataset consisted of 810 enrolled patients with CRC" (Abstract) -- coded for completeness only.
- Split unit `unclear` (disjointness `not_stated`): No split described in the abstract; full text unreachable.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded at stage 1 under E-NOCLF. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (unclear, rule `other`) -- C-index 0.884 (SHF, CoxPH); not a classification metric
- Cohort: private_single_centre (`private`), CT, other (colorectum); n_patients=810, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`na`
- Input `3D_volume`, label broadcast `na`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `medium`, flagged `True`. UNREACHABLE (SAGE 403; no PMC/OA copy). Excluded on the abstract as survival-only. Flagged because a supplementary categorical analysis with an I4 metric cannot be ruled out from the abstract.

**70. PMID 42367846** -- *AI Models for Classifying Wrist Ligament Injuries Using Synthetically-Generated Joint Proximity Maps from FEMs* (2026, bioRxiv)

- Decision: `excluded` -- `E-PROJ`
- Reachability: `oa_pmc_or_publisher` / version used: `preprint`
- Exclusion evidence: "a significant portion of this work focused on developing methods to convert proximity maps into image formats that AI models can utilize more efficiently" and "the RGB field captured radial/ulnar translation (red, x), dorsal/volar translation (green, y), and proximal/distal translation (blue, z). ... The final resolution of the RGB conformal proximity map images was 64x64 pixels." (Methods). The classifier input is a conformally mapped bone-surface field -- an unfolded/flattened surface map with no slice axis -- generated by finite element simulation, not a cross-section of the 4DCT acquisition.
- Evaluation unit `other`: "In total, FEMs performed over 7,500 simulations, yielding over 9,000,000 labeled RGB images and associated descriptive metrics." -- the unit is a simulated proximity-map image; coded for completeness only.
- Split unit `random_unit_not_stated` (disjointness `not_stated`): "Data were divided into training (80%), validation (10%), and testing (10%) portions" (Methods) -- unit not named; only two human participants contributed anatomy, so the two subjects' simulations are certainly present in all three partitions.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded under E-PROJ. Searches were nevertheless run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. The 8 'permut' hits are permutations of LIGAMENT INJURY STATES in the Monte Carlo design ("each sample created a random permutation of injury to the five primary stabilizers"), NOT a permuted-label null; the 'chance' hit is "each of the five ligamentous stabilizers had a 33% chance of being injured" (design, not a chance-level measurement).
- Chance asserted without measurement: `False`
- Positional distribution `no`
- Headline: `AUC` = 0.757 (internal_held_out, rule `abstract_sentence`) -- average AUROC across all injury types and kinematics; reported for completeness, paper excluded
- Cohort: synthetic (finite element models from 4DCT of 2 asymptomatic participants) (`private`), CT, musculoskeletal; n_patients=2, n_patients_test=None, n_slices_or_images=9000000, n_positive_reported=`slices_only`
- Input `na`, label broadcast `na`, uncertainty `none`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `medium`, flagged `True`. bioRxiv PREPRINT (PubMed publication type 'Preprint'), reachable in PMC. E-PROJ is the first applicable code (flattened surface map). E-NONMED would also apply on a strict reading -- the classifier never sees any patient image, only finite-element simulation output from two asymptomatic volunteers. Flagged because the E-PROJ / E-NONMED choice and the 'is a synthetic FEM image a medical image' question are both judgement calls, and because preprints are not addressed in the codebook.

**71. PMID 40093990** -- *Explainable deep learning algorithm for identifying CVST-related hemorrhage from spontaneous ICH using CT* (2025, EClinicalMedicine)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `patient`: "Table 2 The performance of the classification model on the internal and external testing datasets. ... Internal testing data External testing data CVST-ICH sICH CVST-ICH sICH Cases - - 26 76 38 119 AUC [95% CI] ... 0.9352 [0.8674, 0.9818] 0.8476 [0.7629, 0.9228]" -- one row per patient case, one admission CT per patient.
- Split unit `patient_subject` (disjointness `stated_only`): "The matched data was then randomly split into a training set (75%) and an internal testing set (25%) used for model development and internal testing, respectively." and "The entire training dataset was randomly divided into five groups (61 cases per group) for training (four groups) and validation (one group) using the five-fold cross-validation method" (Methods) -- the units divided are propensity-matched patient cases (Table 1 counts patients).
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. All 5 'baseline' hits are the Integrated-Gradients reference image ("Using a black image as the baseline, a gradient linear path from the baseline image to input image was created"), which is an XAI construct, not a comparator arm; 'majority'=0, 'constant'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0. Comparators are nine doctors, which the codebook excludes.
- Chance asserted without measurement: `True`
- Positional distribution `no`: Table 1 gives "Number of axial slices" and "Slices per case, mean [SD] 29.2 [3.48]" -- scan geometry, not the distribution of the label along the slice axis.
- Headline: `AUC` = 0.85 (external, rule `abstract_multiple_took_external`) -- CVST-ICH vs spontaneous ICH, external dataset (0.8476 [0.7629, 0.9228] in Table 2); internal testing AUC 0.94
- Cohort: private_multi_centre (5 hospitals, Zhejiang/Fujian) (`private`), CT, brain; n_patients=565, n_patients_test=157, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `3D_volume`, label broadcast `na`, uncertainty `ci_unspecified_method`, code `public_link_stated`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `high`, flagged `False`. Multi-task (3D U-Net hematoma segmentation as a pretext task + classification head); classification arm coded per amendment A3. chance_asserted_without_measurement=TRUE: "The MCC value spans from -1 to +1, with +1 denoting perfect prediction accuracy, 0 signifying no better than random chance" -- an asserted chance level with no measured chance-level arm, which is exactly what that field exists to capture. n_patients = 408 internal (102 CVST-ICH + 306 propensity-matched sICH, from 102 + 683 screened) plus 157 external. Code: "The proposed model is provided as an open-sourced package at Github: https://github.com/CVST-Research/CVST-ICH_Classify" (Data sharing statement); link not verified.

**72. PMID 30734849** -- *Machine learning identifies "rsfMRI epilepsy networks" in temporal lobe epilepsy* (2019, Eur Radiol)

- Decision: `excluded` -- `E-DERIV`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Exclusion evidence: "Probabilistic independent component analysis (PICA) was applied to rsfMRI data from 132 subjects (42 TLE patients + 90 healthy controls) and 88 independent components (ICs) were obtained following standard procedures. Elastic net-selected features were used as inputs to support vector machine (SVM)." (Abstract, Methods). The SVM input is a vector of IC network strengths, not a spatially resolved image -- the same construction as pilot paper PMID 34924987 (functional-connectivity matrix), which created code E-DERIV.
- Evaluation unit `patient`: "SVM could classify individuals with epilepsy with 97.5% accuracy (sensitivity = 100%, specificity = 94.4%)." (Abstract, Results) -- coded for completeness only.
- Split unit `unclear` (disjointness `not_stated`): No split described in the abstract; full text unreachable.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded at stage 1 under E-DERIV. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `accuracy` = 0.975 (unclear, rule `abstract_sentence`) -- TLE vs healthy control; reported for completeness, paper excluded
- Cohort: private_single_centre (NIMHANS) (`private`), MRI, brain; n_patients=132, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `na`, label broadcast `na`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `medium`, flagged `True`. UNREACHABLE (Springer 403 on both the article and the 'free pdf' link Europe PMC advertises). Excluded on the abstract, which names the SVM input explicitly. Flagged because it is an abstract-only exclusion.

**73. PMID 40081198** -- *MCNEL: A multi-scale convolutional network and ensemble learning for Alzheimer's disease diagnosis* (2025, Comput Methods Programs Biomed)

- Decision: `unreachable_eligibility_unresolved`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Evaluation unit `unclear`: "Extensive experiments on the Alzheimer's Disease Neuroimaging Initiative (ADNI) dataset validate the effectiveness of our solution, which achieves average accuracies of 96.67% for ADNI-1 and 96.20% for ADNI-2" (Abstract, Results) -- the unit is not named. The abstract says the feature extractor is "capable of extracting features from multi-view slices", which raises but does not settle whether accuracy is per slice or per subject.
- Split unit `unclear` (disjointness `not_stated`): No split described in the abstract; full text unreachable.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: NOT EVIDENCED -- full text unreachable. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `accuracy` = 0.9667 (unclear, rule `abstract_sentence`) -- MCNEL ensemble, ADNI-1
- Cohort: ADNI-1; ADNI-2 (`public`), MRI, brain; n_patients=None, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`neither`
- Input `unclear`, label broadcast `unclear`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `low`, flagged `True`. UNREACHABLE (Elsevier 403; no PMC record, not OA in OpenAlex). HIGH-VALUE TARGET IF IT CAN EVER BE OBTAINED: an ADNI slice-based ensemble reporting 96.67%/96.20% accuracy, i.e. exactly the configuration (public dataset, multi-view slices, near-ceiling accuracy, split unit unstated) where a positional baseline is most likely to be informative.

**74. PMID 39384719** -- *Dual Energy CT for Deep Learning-Based Segmentation and Volumetric Estimation of Early Ischemic Infarcts* (2025, J Imaging Inform Med)

- Decision: `excluded` -- `E-SEG`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Exclusion evidence: "A self-configuring 3D nnU-Net was trained for segmentation on (1) standard 120 kV mixed-images (2) 190 keV virtual monochromatic images and (3) 120 kV + 190 keV images as dual channel inputs. Algorithm performance was assessed with Dice scores with paired t-tests on a test set." (Abstract) and "The primary performance metric was Dice score which was evaluated with Python." (Statistical Analysis). No categorical class decision is evaluated and no I4 metric is reported.
- Evaluation unit `unclear`: not applicable -- Dice per case and global aggregate Dice only.
- Split unit `unclear` (disjointness `na`): "Final evaluation was independently performed using the hold-out test set." (Methods) -- a segmentation split; not coded.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded under E-SEG. not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (internal_held_out, rule `other`) -- global aggregate Dice 0.616 / 0.645 / 0.665; segmentation metric only
- Cohort: private_single_centre (`private`), CT, brain; n_patients=330, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `3D_volume`, label broadcast `na`, uncertainty `sd_across_folds`, code `none`
- Searches run: not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Confidence `high`, flagged `False`. Titled around 'Segmentation and Volumetric Estimation' and evaluates exactly that. PMC full text was blocked by a reCAPTCHA interstitial on the first two attempts and retrieved on the third; coded from the version of record.

**75. PMID 38298725** -- *Deep learning-assisted survival prognosis in renal cancer: A CT scan-based personalized approach* (2024, Heliyon)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `volume_or_scan_not_patient`: "Table 1 Number of training, validation, and test set after augmentation. Fold Training Validation Test 1 1150 184 659 2 1133 184 676 3 1151 184 658" and "By augmenting the test set, we aimed to more rigorously evaluate the model's robustness to these typical discrepancies found in scans." (Sec.5.1, Discussion). The classification metrics in Table 3 are computed over 659-676 augmented kidney volumes per fold, several per patient, never aggregated back to the patient.
- Split unit `patient_subject` (disjointness `stated_only`): "We used three-fold cross-validation to train the classification and survival network. We divided our dataset into three folds depending on the number of dead and censored patients." and "The consistent ratios across folds also avoided introducing data leakage or bias." (Sec.5.2 Data splitting)
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. 'baseline'=0, 'prevalence'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0; the 2 'majority' hits are the majority ISUP2 class (imbalance), the 2 'constant' hits are "constant hazard ratios" and a "constant density function".
- Chance asserted without measurement: `True`
- Positional distribution `no`: "Within our augmentation strategy, we have specifically utilized position and noise tr[ansforms]" -- data augmentation, not a distribution of the label along the slice axis.
- Headline: `F1` = 0.61 (cross_validation, rule `first_results_table_row`) -- ISUP 4-class grading, fold 1 of 3 (Table 3: F1 0.61 / 0.84 / 0.67). Selection rule 'first_results_table_row' fired because the abstract's results sentence contains no classification metric at all -- "an average concordance index of 0.72, an integrated Brier score of 0.15, and an area under the curve value of 0.71" are all survival metrics ("the average C-index, IBS, and AUC were all 0.72, 0.15, and 0.71").
- Cohort: KiTS21 (`public`), CT, kidney; n_patients=244, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`patients_only`
- Input `patch_3D`, label broadcast `na`, uncertainty `none`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `medium`, flagged `True`. Two-network paper (ISUP-grade classifier feeding a survival network); the classification arm is coded and the survival arm ignored. Flagged for two reasons: (1) the headline had to be taken from the first results table because the abstract reports only survival metrics, and Table 3 is transposed so 'first row' is ambiguous; (2) the TEST SET IS AUGMENTED -- 244 patients become 658-676 test samples -- so the reported metrics are over augmented copies, an unusual evaluation-unit choice that a second screener may code differently. chance_asserted_without_measurement=TRUE: "A value of 0.5 implies random concordance, whereas a value of 1 shows ideal concordance". The only GitHub link is the third-party pycox library, so code_availability='none'.

**76. PMID 36539234** -- *A deep learning approach to identify missing is-a relations in SNOMED CT* (2023, J Am Med Inform Assoc)

- Decision: `excluded` -- `E-NONMED`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Exclusion evidence: "SNOMED CT is the largest clinical terminology worldwide. ... In this work, we introduce a deep learning-based approach to uncover missing is-a relations in SNOMED CT." and "The model is a binary classifier leveraging concept name features, hierarchical features, enriched lexical attribute features, and logical definition features." (Abstract). A full-text search for 'image' and 'imaging' returns ZERO hits: there is no imaging of any kind in this paper.
- Evaluation unit `other`: Concept pairs in a terminology: "a total of 1661 potential candidates" (Abstract) -- coded for completeness only.
- Split unit `unclear` (disjointness `na`): "We introduce a cross-validation inspired approach to identify missing is-a relations among all hierarchically unrelated containment concept-pairs." (Abstract) -- not an imaging split.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded under E-NONMED. not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Chance asserted without measurement: `False`
- Positional distribution `no`
- Headline: `F1` = 0.8279 (cross_validation, rule `abstract_sentence`) -- reported for completeness; paper excluded
- Cohort: SNOMED CT Clinical finding subhierarchy (September 2019 US edition) (`public`), other, other (not imaging); n_patients=None, n_patients_test=None, n_slices_or_images=None, n_positive_reported=`na`
- Input `na`, label broadcast `na`, uncertainty `none`, code `none`
- Searches run: not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Confidence `high`, flagged `False`. The cleanest illustration of the frame's imprecision in my whole assignment: 'CT' in the query matched 'SNOMED CT'. Deep learning + 'CT' + classification + F1 all present, zero images. Worth naming explicitly in the PRISMA flow narrative.

**77. PMID 38591974** -- *Predicting Invasiveness of Lung Adenocarcinoma at Chest CT with Deep Learning Ternary Classification Models* (2024, Radiology)

- Decision: `unreachable_eligibility_unresolved`
- Reachability: `unreachable_paywalled` / version used: `abstract_only`
- Evaluation unit `lesion`: "A total of 4929 nodules from 4483 patients (mean age, 50.1 years +/- 9.5 [SD]; 2806 female) were divided into training (n = 3384), validation (n = 579), and internal (n = 966) test sets. A total of 361 pGGNs from 281 patients ... formed the external test set." (Abstract, Results) -- the counts divided are nodules.
- Split unit `unclear` (disjointness `not_stated`): "A total of 4929 nodules from 4483 patients ... were divided into training (n = 3384), validation (n = 579), and internal (n = 966) test sets." (Abstract) -- the split is over nodules and the abstract does not say whether a patient's nodules were kept together; the full text could not be reached to check.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: NOT EVIDENCED -- full text unreachable. NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `accuracy` = 0.85 (external, rule `abstract_multiple_took_external`) -- model 6 (with adjudication), minimally invasive adenocarcinoma class, external test set of 361 pure ground-glass nodules
- Cohort: private_multi_centre (`private`), CT, lung; n_patients=4483, n_patients_test=281, n_slices_or_images=4929, n_positive_reported=`patients_and_slices`
- Input `unclear`, label broadcast `unclear`, uncertainty `unclear`, code `none`
- Searches run: NOT RUN -- full text unreachable (rungs 1-4 of screen_protocol.md sec.7 exhausted); coded from abstract only, so a negative on trivial_baseline is NOT evidenced
- Confidence `low`, flagged `True`. UNREACHABLE despite the CC BY licence line in the abstract: pubs.rsna.org returns HTTP 403 to every client tried, no PMC record, no Europe PMC full text; the HKU institutional-repository record OpenAlex lists also returns 403. Rung 5 (ILL / author request) was unavailable in-session. Note 4,929 nodules from 4,483 patients means ~10% of patients contribute more than one nodule, so nodule-level splitting is a live leakage question the abstract cannot settle.

**78. PMID 35378943** -- *Models of Artificial Intelligence-Assisted Diagnosis of Lung Cancer Pathology Based on Deep Learning Algorithms* (2022, J Healthc Eng)

- Decision: `included`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Evaluation unit `lesion`: "Of the 652 patients included, there were a total of 674 pulmonary nodules. ... the improved 3D U-net network-assisted diagnosis system detected a total of 674 target nodules" and "The AUC area of the improved 3D U-net system was 0.729" (Sec.3.2, Results) -- the ROC unit is the nodule.
- Split unit `unclear` (disjointness `not_stated`): The paper describes NO data partition anywhere. Full-text searches return: 'training set'=0, 'testing set'=0, 'validation'=0, 'verification'=0, 'partition'=0; the single 'test set' hit is in the literature review ("For the external test set, the transferred model has good generalization ability [3]") and the single 'fold' hit is "a folded neural network".
- Trivial baseline: all six false | P1 flag: **false**
  - Evidence: no match. Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available. 'baseline'=0, 'majority'=0, 'prevalence'=0, 'trivial'=0, 'metadata'=0, 'clinical-only'=0, 'clinical model'=0, 'slice index'=0, 'permut'=0; the 'chance' hit is "increase the chance of curing lung cancer patients" and the 3 'constant' hits are "The normalization constant is F" and "new signs are constantly being explored". The only comparators are the original 3D U-net and radiologists.
- Chance asserted without measurement: `False`
- Positional distribution `no`: Table 2 tabulates nodule counts by lobe ("69 nodules in the middle right lung, and 132 nodules in the lower right lung"). Coded 'no' under the codebook rule that anatomical statements with no reference to position within the acquired stack are 'no' -- the same rule applied to PMID 42130124's vertebral-level table.
- Headline: `accuracy` = 0.923 (unclear, rule `abstract_sentence`) -- "an accuracy rate of 92.3% for predicting malignant lung nodules and an accuracy rate of 82.8% for benign lung nodules"; the paper's own ROC analysis gives AUC 0.729 for the same system
- Cohort: private_multi_centre (`private`), CT, lung; n_patients=652, n_patients_test=None, n_slices_or_images=674, n_positive_reported=`slices_only`
- Input `3D_volume`, label broadcast `na`, uncertainty `none`, code `none`
- Searches run: baseline; chance; random; majority; prevalence; constant; trivial; metadata; clinical-only; clinical model; position; location; slice index; permut -- all 14 run over the full text; supplement: none available
- Confidence `low`, flagged `True`. RETRACTED PUBLICATION -- the PMC record carries the retraction notice ("Retracted: Models of Artificial Intelligence-Assisted Diagnosis of Lung Cancer Pathology Based on Deep Learning Algorithms", J Healthc Eng 2023;9874292). As with PMID 35872945 the codebook has no retraction rule, so it is INCLUDED and flagged; the protocol authors need to decide and log a rule. The paper is also internally inconsistent: the abstract says "two hospitals", the Methods say "three tertiary hospitals"; the classifier input is chest CT nodules while H-E pathology is the reference standard (so NOT E-2D); and the abstract's 92.3% accuracy sits alongside an AUC of 0.729 for the same system. screener_confidence='low'.

**79. PMID 34888191** -- *Deep learning-based bone suppression in chest radiographs using CT-derived features* (2021, Quant Imaging Med Surg)

- Decision: `excluded` -- `E-SEG`
- Reachability: `oa_pmc_or_publisher` / version used: `version_of_record`
- Exclusion evidence: "This study aims to develop a deep learning-based bone suppression method using CT-derived features to reduce the reliance on the bone-free dataset." and "The synthesized bone-suppressed radiographs were compared with the bone-suppressed reference in terms of peak signal-to-noise ratio (PSNR), mean absolute error (MAE), structural similarity index measure (SSIM), and Spearman's correlation coefficient." (Abstract). The task is image-to-image synthesis; no categorical class decision is evaluated and no I4 metric is reported.
- Evaluation unit `unclear`: not applicable -- image-similarity metrics only.
- Split unit `unclear` (disjointness `na`): "59 high-resolution lung CT scans were processed ... for the training and internal validation ... In external validation, the trained CCNN was evaluated using 30 chest radiographs." (Abstract) -- a synthesis split; not coded.
- Trivial baseline: not coded / unclear | P1 flag: **false**
  - Evidence: not coded -- excluded under E-SEG. not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Chance asserted without measurement: `False`
- Positional distribution `unclear`
- Headline: `other` = None (external, rule `other`) -- MAE 0.0087, SSIM 0.8458, PSNR 20.86 -- image-similarity metrics, not classification
- Cohort: private_single_centre (`private`), CT, lung; n_patients=59, n_patients_test=30, n_slices_or_images=None, n_positive_reported=`neither`
- Input `na`, label broadcast `na`, uncertainty `sd_across_folds`, code `none`
- Searches run: not required -- paper excluded before stage-2 baseline coding; trivial_baseline not coded
- Confidence `high`, flagged `False`. E-SEG applies first (synthesis); E-2D (the output domain is the 2D chest radiograph) and E-NOMET would also apply. Full text reached by scraping the PMC HTML page after the E-utilities OA route returned front matter only.

---

## 8. For the adjudicator

Rank-ordered list of what a second reader should look at first:

1. **The two retracted papers** (35872945, 35378943) -- a codebook rule is needed, and it changes n.
2. **Positional-distribution coding of 42130124 and 35378943** -- ordered anatomical level vs position in the stack. Affects S5, and 42130124 is in the *overlap set*, so any disagreement here will surface directly in the Fleiss' kappa for `positional_distribution_reported`.
3. **34603980** -- threshold-derived dilatation category: E-SEG or include?
4. **42153825** -- radiomics headline with deep-learning image comparators: E-DERIV or include? Unresolvable without the full text.
5. **The 7 unreachable records** -- rung 5 of the access ladder still owes them an attempt.
6. **38298725** -- headline taken from a transposed results table because the abstract has no classification metric; and the test set is augmented, which makes the evaluation unit unusual.
