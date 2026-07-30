# Reserve block R2 -- permutation positions 161-210

Fifty records, screened under `paper/screen_frame.json` **v1.2** and `paper/screen_protocol.md` (v1.0 frozen, amended v1.2, access pass v1.3). 
Machine-readable codes, quotes and search records: **`paper/screen_reserve_R2.json`**. This file is the human-readable companion; every number in it is computed from that JSON, not typed by hand.

The block was taken by permutation position from `paper/screen_sample.json`, in order, with no substitution and no truncation, as the pre-registered extension rule (protocol sec.3.1) requires. Every PMID here was re-checked against the sample file position-by-position before coding.

---

## 1. The headline

**Not one of the 17 included papers reports a zero-image baseline.** Across all 50 records, not one sub-flag in the P1 family -- constant/prevalence, positional, acquisition-metadata, permuted-label -- is coded true by anyone, anywhere.

| endpoint | this block | Wilson 95% |
|---|---|---|
| **P1** -- reports >=1 zero-image baseline with a measured value | **0/17** | 0.0% [0.0%, 18.4%] |
| S1 -- reports >=1 non-imaging baseline of any kind (P1 family + clinical-only) | 1/17 | 5.9% [1.0%, 27.0%] |
| S6 -- unreachable, over the eligible-looking set | 5/22 | 22.7% [10.1%, 43.4%] |

The single S1 positive is PMID **40121941** (*EBioMedicine*), which fits multivariable Cox and logistic models **using only clinical factors** and reports their numbers on the same metric: *"SMuRF outperformed the clinical-based models for both endpoints on the test set (Supplementary Fig. S3, C-index for DFS: 0.79 vs. 0.57; AUC for grade: 0.74 vs. 0.60)."* That is a measured pixel-free comparator, so `clinical_or_demographic_only` is true. It does **not** enter P1, by the primary endpoint's own rule: a clinical-variables model is a different comparison and does not test whether the benchmark is solvable without pixels.

**S6 is 22.7%, above the 15% threshold in protocol sec.7.** Within this block the reportable form of the primary is therefore the bounding interval, not the point estimate: **[0.0%, 22.7%]** over the 22 eligible-looking records, with the lower bound coding all five unreachable papers as not reporting a baseline and the upper bound coding all five as reporting one. Adding records does not narrow that. Recovering full texts does.

---

## 2. Tallies

| | n |
|---|---|
| records screened | 50 |
| excluded at stage 1 (abstract alone unambiguous, D11) | 20 |
| excluded at stage 2 (full text obtained and read first) | 8 |
| **excluded, total** | **28** |
| **unreachable, eligibility unresolved** | **5** |
| **included** | **17** |
| eligible-looking denominator (included + unreachable) | 22 |
| **reporting ANY zero-image baseline** | **0** |

Exclusion codes, reported individually as protocol sec.9 requires:

| code | n | what they were |
|---|---|---|
| `E-SEG` | 9 | segmentation, bounding-box detection, landmark localisation or synthesis with no categorical class decision evaluated |
| `E-NONMED` | 6 | not human medical imaging -- mouse tibia micro-CT, mouse skin FF-OCT, canine middle-ear CT, ovine tissue spectroscopy, microarray expression, Thai clinical text |
| `E-DERIV` | 6 | the fitted classifier eats a feature vector, connectivity graph or descriptor table, not an image |
| `E-2D` | 4 | natively 2D imaging -- fundus photographs, intraoral radiographs, knee radiographs, H&E whole slides |
| `E-NOCLF` | 3 | no supervised classifier fitted -- the class decision is a human reader's (x2, D10) or the outcome is regression only (x1) |

Six `E-DERIV` records in fifty is worth recording on its own: it is the frame's deliberate breadth doing exactly what protocol sec.2.1 said it would, and these papers are inside the query and outside the failure mode.

---

## 3. Secondary endpoints in this block

| id | statement | this block | Wilson 95% |
|---|---|---|---|
| S1 | any non-imaging baseline | 1/17 | 5.9% [1.0%, 27.0%] |
| S2 | headline unit is the slice | 4/17 | 23.5% [9.6%, 47.3%] |
| S3 | among papers reporting any slice-level metric, also report patient-level | 1/5 | 20.0% [3.6%, 62.4%] |
| S4 | explicitly states a subject-level split | 10/17 | 58.8% [36.0%, 78.4%] |
| S5 | reports or discusses the positional distribution of labels | 0/17 | 0.0% [0.0%, 18.4%] |
| S6 | unreachable, over included + unreachable | 5/22 | 22.7% [10.1%, 43.4%] |
| S8 | subject-clustered uncertainty interval | 1/17 | 5.9% [1.0%, 27.0%] |
| S9 | reports n positive patients as well as n positive slices | 5/17 | 29.4% [13.3%, 53.1%] |

**S5 is zero here.** Not one of the seventeen relates its label to position along the acquisition axis, by figure, table or number. The one record in the whole block that does model an ordered slice index -- PMID 41087406, whose classes are ordered anatomical compartments of the tibia and which explicitly fits 'a regional probability distribution method to detect the transitional landmarks between these compartments' -- is a **mouse** study and is excluded by `E-NONMED`. That is recorded in the JSON in full so the reason it is not in the denominator is visible rather than silent.

**S7, headline unit crossed with the P1 flag** (exact counts, all cells):

| effective headline unit | n | P1 true | P1 false |
|---|---|---|---|
| unclear | 7 | 0 | 7 |
| patient | 6 | 0 | 6 |
| slice | 4 | 0 | 4 |
| **total** | **17** | **0** | **17** |

Distributions on the fields the endpoints rest on, `unclear` reported as its own category and never merged:

- **`evaluation_unit_reported`**: unclear 7, patient 5, slice 4, both 1
- **`split_unit`**: patient_subject 10, random_unit_not_stated 3, slice_or_image 3, unclear 1
- **`modality`**: CT 9, MRI 5, PET_CT 1, OCT 1, multiple 1
- **`input_representation`**: unclear 9, 2D_slice 5, patch_3D 3
- **`code_availability`**: not_stated 15, public_link_stated 2
- **`uncertainty_interval_reported`**: none 11, ci_unspecified_method 5, ci_clustered_by_subject 1

Seven of seventeen included papers never say what one scored unit is, and one more names a unit (`records`) that cannot be mapped to anything. Three split with no unit named at all. These are not coding failures; the codebook says so explicitly, and they are the finding.

---

## 4. Access

The five-rung ladder in protocol sec.7 was worked in order for all 30 records that reached stage 2. Per **D4**, it was **not** climbed for the 20 records excluded at stage 1, so S6 measures the reachability of the eligible-looking literature and is not inflated by records that were never eligible.

| rung that worked | n |
|---|---|
| `oa_pmc_or_publisher` | 24 |
| `unreachable_paywalled` | 5 |
| `repository_or_accepted_manuscript` | 1 |
| `not_attempted_excluded_at_stage1` (D4) | 20 |

**One recovery at rung 4.** PMID 33713959 (*Comput Methods Programs Biomed*, closed at Elsevier, `is_oa=False` at both Unpaywall and OpenAlex) was recovered as an **accepted manuscript** from the University of West London institutional repository and coded in full, including the fourteen-term search. It is flagged for the version-of-record sensitivity analysis.

**One publisher OA copy that is not the version of record.** PMID 42225843 (*Sci Rep*) is served by the publisher only as an unedited accepted manuscript -- *"We are providing an unedited version of this manuscript to give early access to its findings."* Rung 1, but `fulltext_version_used='accepted_manuscript'`, flagged, and reserved for the same sensitivity analysis.

**Five unreachable, every failed rung named in the JSON.** 32232524 (Springer, genuinely closed -- Unpaywall, OpenAlex, Europe PMC and Crossref all agree, no green copy exists); 38721876 (SAGE 403; the HAL record OpenAlex points to reports `openAccess_bool=false` and holds no file, and its landing page served an anti-scraping proof-of-work challenge that was **not** circumvented); 41655629 (Elsevier bot-detection challenge, **not** circumvented, no repository copy); 42184237 (JoVE returns HTTP 202 with an empty body to every client here); 42159478 (RSNA 403 -- the same publisher-side refusal recorded in changelog v1.3 -- and the Radboud repository record exposes no bitstream; arXiv returns `totalResults=0` for the exact title).

**No infringing source was used at any point.** Sci-Hub, LibGen and equivalents were not accessed. Where a paper was reachable only through one, it stayed unreachable. Bot-detection and proof-of-work challenges were not bypassed.

Two of the five would very probably have been *included* on full text -- 41655629 (CBCT, AUC 94.4% internal / 85.2-90.0% external) and 42184237, whose abstract even states a patient-level split in so many words. They are counted unreachable because no full text was read. **D1** forbids coding them included on abstract evidence, for the reason the codebook gives: an included record must carry an *evidenced* `trivial_baseline`, and the mandatory fourteen-term search cannot be run on a text nobody has. That is the whole cost of the access failure, and it is what S6 exists to measure.

---

## 5. How the negative was evidenced

An unevidenced negative is not accepted. For every one of the 17 included records, all fourteen terms -- *baseline, chance, random, majority, prevalence, constant, trivial, metadata, clinical-only, clinical model, position, location, slice index, permut* -- were run over the complete full text, and `searches_run` records the hit counts term by term with what each hit actually was.

Supplements, per record: **5 obtained and searched** (42225843, 40676122, 35571081, 40121941, 35562596 -- the last two required going to the publisher's static-content host after PMC served a download shell); **11 declare no supplementary material, so none exists**; **1** (34109325) declares a supplement that is a *video* of a reconstructed cardiac cycle, containing no searchable text and no results. No all-false code in this file rests on an unsearched supplement that its article declares.

**Three hits would have produced a false positive if the search had been counted rather than read.** This is the part of the procedure that cannot be automated:

1. **PMID 36010183** returns four hits on `permut` -- *"a permutation-based machine learning (ML) voting classifier"*, *"evaluated by means of permutations of the voting mechanism"*. Every one is enumeration of combinations in a **voting ensemble** of SVM, Gaussian NB and XGBoost. No label is ever shuffled. `permuted_or_shuffled_label` is **false**.
2. **PMID 41657565** returns *"The curve is plotted against a random guess line"* and *"The curves are far above the diagonal baseline"*. That is the **ROC diagonal**, asserted and never measured -- which the codebook classes as false for every sub-flag and routes to `chance_asserted_without_measurement`, where it is recorded.
3. **PMID 40676122** returns *"We also repeated the same analysis for the original images as a baseline without sparse-view artifacts"* -- the full-view **image** reconstruction. It uses pixels; `does_not_count` covers it.

And the one that went the other way: **PMID 40121941**'s clinical-only arm was found through the `clinical` stem, not through the literal terms `clinical-only` or `clinical model`, both of which return zero hits in that article. It is the block's only true sub-flag.

---

## 6. Amended-codebook rules that actually fired

| rule | where it fired | effect |
|---|---|---|
| **D1** unreachable dominates included | 32232524, 38721876, 41655629, 42184237, 42159478 | five abstracts consistent with inclusion (two strongly so) are coded `unreachable_eligibility_unresolved`, not `included` |
| **D2** positive/negative asymmetry; three-valued sub-flags | all 5 unreachable records | sub-flags are `not_assessable`, never false; the failed ladder rung is named in each `trivial_baseline_quote` |
| **D3** `not_applicable` on non-included records | all 28 excluded records | descriptive fields carry `not_applicable` with a one-line reason, and enter no numerator or denominator |
| **D4** ladder not climbed for stage-1 exclusions | 20 records | keeps S6 measuring the eligible-looking literature; **seven** of the twenty carry a closed or text-mining-only licence and would otherwise have inflated it |
| **D5** Methods govern over Abstract on a fact | 40121941 (grade AUC 0.75 vs 0.74), 36010183 (6,200 vs 6,126 images) | the Results value is coded and the contradiction recorded verbatim |
| **D6** patient-naming-noun word list | 40134559, 35562596, 40121941, 35571081, 41331277, 40342492 (in), 40365495 (undefined `cases`, out) | S4 is 10/17, and `cases` alone never upgrades a split |
| **D7** a unit named in a table is a named unit | 34828081 (`# patients` per split), 36010183 (`number of images`) | two records leave `random_unit_not_stated` |
| **D9** ordering of the *classification unit*, not vocabulary | 42225843 (slice thickness = scan geometry -> `no`), 40121941 (sub-volume crop position -> `no`) | S5 stays at 0/17 on the correct test |
| **D10** E-SEG's 'no categorical class decision' qualifier binds | 40544715, 41899885 | both go to `E-NOCLF`, not `E-SEG`: a class decision *was* evaluated, by human readers |
| **D11** trigger for `include_provisional` | whole block | `include_provisional` was used **zero** times; every record is `exclude` or `go_to_fulltext` |
| **D12** modality is the modality of the *input* | 35571081 (`PET_CT`, not `multiple`), 41444372 (`multiple`, two acquisitions entering) | |
| **D13** `not_stated` for silence on code | 15 of 17 included records | only 2 state a code link; without D13 all 15 would have been forced to the inference `none` |
| **D14** `na_only_one_unit_reported` vs `unclear` | 16 of 17 included records | applies even where the single unit is itself `unclear` |

---

## 7. Record-by-record

`R2-A`/`R2-B`/`R2-C`/`R2-D`/`R2-E` refer to the coding conventions stated in the JSON header. Full quotes for every coded field, and the term-by-term search record, are in the JSON.

| pos | PMID | year / venue | decision | code / P1 | one line |
|---|---|---|---|---|---|
| 161 | 41861368 | 2026 JMIR Form Res | excluded | `E-NONMED` | Thai discharge-summary NER; no imaging anywhere. Matches the frame only via 'SNOMED-CT' and 'F1'. |
| 162 | 40544715 | 2025 Eur J Radiol | excluded | `E-NOCLF` | DLIR reconstruction; four radiologists make every diagnosis. D10 -> E-NOCLF, not E-SEG. |
| 163 | 40783612 | 2025 Sci Rep | **included** | P1 false | Kaggle brain-stroke CT, 2,501 'CT Images', accuracy 99.09%. No patient count, no split unit, no baseline. R2-A. |
| 164 | 32691326 | 2020 Phys Eng Sci Med | excluded | `E-DERIV` | SVM/kNN/ANN on histogram+GLCM+NGTDM features; the image is discarded before the class decision. R2-C. |
| 165 | 42225843 | 2026 Sci Rep | **included** | P1 false | AIS on NCCT. Reports patient-wise **and** slice-wise sensitivity (69.66% vs 44.95%), GLMM/GEE clustered CIs, patient-level split -- and no zero-image baseline. Accepted manuscript, flagged. |
| 166 | 41899885 | 2026 Bioengineering (Basel) | excluded | `E-NOCLF` | nnU-Net segments and normalises; six readers decide. D10 -> E-NOCLF. Full text read before excluding. |
| 167 | 34729970 | 2021 J Biomed Opt | excluded | `E-NONMED` | Ovine tissue, light-scattering spectra. Not human, not imaging. |
| 168 | 40676122 | 2025 Sci Rep | **included** | P1 false | ResNet-50 AI observer over RSNA ICH 2D axial slices; the only 'baseline' is the full-view image reconstruction. |
| 169 | 32232524 | 2020 Abdom Radiol (NY) | unreachable | P1 `not_assessable` | Springer, genuinely closed at every source. Rungs 1-5 all fail. D1 -> unresolved. |
| 170 | 34828081 | 2021 Entropy (Basel) | **included** | P1 false | COVID-CT (volumetric arm) + two X-ray sets; MobileNetV3 embeddings + Aquila + KNN, accuracy 0.783. A2 applies; split unit flagged. |
| 171 | 40134559 | 2024 J Bone Oncol | **included** | P1 false | 3DResUNet on vertebral ROIs, 749 patients split patient-wise, AUC 0.966 from thresholded vBMD. Flagged on the headline metric's provenance. |
| 172 | 40745009 | 2025 Sci Rep | excluded | `E-2D` | Fundus photographs are the model input; the CT/CMR are outcomes. |
| 173 | 31001458 | 2019 Cureus | excluded | `E-NONMED` | ANN on a 12-gene microarray vector. No imaging. |
| 174 | 39656660 | 2025 Dentomaxillofac Radiol | excluded | `E-2D` | Intraoral periapical radiographs; CBCT is only the ground-truth arbiter. |
| 175 | 34109325 | 2020 Med Image Comput Comput Assist Interv | **included** | P1 false | Cine CMR short-axis stack, VAE + subject-level classifier, sens 88.43%. The ablated imaging 'baseline' beats it on balanced accuracy and the authors say so. |
| 176 | 38893629 | 2024 Diagnostics (Basel) | excluded | `E-SEG` | Mask R-CNN on a dataset of PE-containing sections only -- no negative class, so sen/spec are pixel-wise. E-SEG. |
| 177 | 38905892 | 2024 Comput Biol Med | excluded | `E-SEG` | 3D U-Net; Dice and voxel precision only. Title says detection, evaluation is segmentation (A4). |
| 178 | 32484573 | 2020 Med Phys | excluded | `E-SEG` | Catheter localisation in MRI, evaluated in millimetres. |
| 179 | 33713959 | 2021 Comput Methods Programs Biomed | **included** | P1 false | OASIS AD classification, GoogLeNet transfer 93.02%, and an explicit 'patient-wise division'. Recovered at rung 4 from the UWL repository; flagged. |
| 180 | 39562310 | 2024 Hum Brain Mapp | excluded | `E-DERIV` | ICA loadings + sFNC connectivity + GWAS SNPs. The pilot's E-DERIV construction with a genomics arm. |
| 181 | 36010183 | 2022 Diagnostics (Basel) | **included** | P1 false | Kaggle AD 4-class, 6,126 2D images. Four `permut` hits are a **voting ensemble**, not a permutation null. D5 fires on the image count. |
| 182 | 41087406 | 2025 Sci Rep | excluded | `E-NONMED` | Mouse tibia micro-CT -- and the only paper in the block that models an ordered slice index and its label distribution. Excluded on species; recorded in full. |
| 183 | 35571081 | 2022 Front Pharmacol | **included** | P1 false | PET/CT EGFR status, two-channel 3D CNN on 64^3 cuboids, validation AUC 0.85. Zero hits on all fourteen terms in main text; supplement searched. |
| 184 | 39768288 | 2024 Life (Basel) | excluded | `E-SEG` | Latent-diffusion PET super-resolution; no categorical decision of any kind (D10's final clause). |
| 185 | 41353071 | 2026 Acad Radiol | excluded | `E-SEG` | Cardiac MRI planning landmarks, millimetres and degrees. |
| 186 | 35562596 | 2022 J Cancer Res Clin Oncol | **included** | P1 false | CT cervical nodes, 276 patients split patient-wise, accuracy 87.50% -- over a denominator the paper never states. Flagged. |
| 187 | 41633187 | 2026 Comput Med Imaging Graph | excluded | `E-SEG` | YOLOv11 CRC detection. Reads like two-unit classification until the Methods: *'TNs were set to 0'*. E-SEG, flagged. |
| 188 | 41331277 | 2025 Sci Rep | **included** | P1 false | Spectralis cross-hair B-scans, VGG16 five-class VA, macro AUC 0.772, explicit patient-level partition. Flagged on I3 (line scan, not a volume stack). |
| 189 | 40714864 | 2025 Vet Radiol Ultrasound | excluded | `E-NONMED` | Canine middle-ear CT. Every other criterion met; not human. |
| 190 | 40121941 | 2025 EBioMedicine | **included** | P1 false (S1 **true**) | CT + H&E fusion for OPSCC. **The block's only S1 positive**: clinical-factors-only models, AUC 0.60 vs 0.74 on the same metric. D5 fires on 0.75 vs 0.74. |
| 191 | 33662804 | 2021 Comput Methods Programs Biomed | excluded | `E-SEG` | COVID-19 CT segmentation; accuracy 0.994 alongside DSC 0.704 gives the voxel denominator away. |
| 192 | 36672641 | 2023 Biomedicines | **included** | P1 false | 17,194 slices from the 'CT Scan Slice Dataset', 70:20:10 with no unit named, no patient count, accuracy 97.12%. |
| 193 | 39956834 | 2025 Sci Rep | excluded | `E-NOCLF` | OCT thickness tables -> regression only. E-DERIV describes the input but presupposes a classifier; E-NOCLF is the code that applies. |
| 194 | 38721876 | 2024 J Endovasc Ther | unreachable | P1 `not_assessable` | SAGE 403; the HAL record OpenAlex points to holds no file. Anti-scraping challenge not circumvented. |
| 195 | 34626908 | 2021 Comput Med Imaging Graph | excluded | `E-NONMED` | Mouse skin FF-OCT. E-PROJ checked and rejected -- a depth axis exists. |
| 196 | 35748898 | 2022 Eur Radiol | excluded | `E-DERIV` | nnU-Net segments, SVM over 7 radiomics features decides. R2-C -> E-DERIV. |
| 197 | 41657565 | 2025 Front Med (Lausanne) | **included** | P1 false | CerevianNet over five Kaggle sets; AUC 1.00 on one, 74.11% validation on another. Draws the ROC 'random guess line' and never measures one. |
| 198 | 41655629 | 2026 J Endod | unreachable | P1 `not_assessable` | Elsevier bot challenge, not circumvented; no repository copy. Would very likely have been included. |
| 199 | 40601647 | 2025 PLoS One | excluded | `E-DERIV` | Texture/shape/colour features -> DGWO selection -> HAETN. The transformer eats a feature vector. |
| 200 | 40002836 | 2025 Biomedicines | **included** | P1 false | 3,194 images from 60 patients split **by image file path**; 100.00% accuracy, AUC 1.0000 [1.0000, 1.0000], and *'No data leakage occurred'*. |
| 201 | 41037545 | 2026 IEEE Trans Biomed Eng | excluded | `E-DERIV` | rs-fMRI brain graphs + phenotypic data. Identical to pilot PMID 34924987. |
| 202 | 41444372 | 2025 Sci Rep | **included** | P1 false | Federated CT+MRI+histology+genomics, 221,347 'records', 99.2% accuracy. Scored unit, split unit and patient count all undecidable. Flagged. |
| 203 | 42184237 | 2026 J Vis Exp | unreachable | P1 `not_assessable` | JoVE returns an empty body. Abstract states a patient-level split in so many words -- information now lost. |
| 204 | 39811011 | 2025 Eur Heart J Imaging Methods Pract | excluded | `E-2D` | H&E-stained myocardial whole slides. |
| 205 | 35830745 | 2022 Med Image Anal | excluded | `E-SEG` | Anatomy-guided object recognition; centroid, scale and wall-distance errors. |
| 206 | 42459762 | 2026 Front Artif Intell | excluded | `E-2D` | Knee radiographs; CT appears once, in the background sentence. |
| 207 | 40342492 | 2025 J Bone Oncol | **included** | P1 false | CT + pathology Swin fusion, 215 patients, *'patient-level stratification was strictly enforced'*, AUC 0.966. Class counts never given. |
| 208 | 42159478 | 2026 Radiol Artif Intell | unreachable | P1 `not_assessable` | RSNA 403 and a Radboud record with no bitstream. Eligibility genuinely unresolved: registration paper, but with a pCR AUC of 0.81. |
| 209 | 40576676 | 2026 J Craniofac Surg | excluded | `E-SEG` | Sella turcica coordinate regression from 3D surface meshes; millimetres. |
| 210 | 40365495 | 2025 Front Med (Lausanne) | excluded | `E-DERIV` | UPerNet segments, logistic regression over nine PyRadiomics features decides. 'cases' as a split unit -> D6. |

---

## 8. What this block does and does not establish

**Does.** Seventeen more papers, drawn by a frozen permutation from a pre-registered 9,979-record frame, read in full, searched term by term including supplements, and coded against an amended codebook -- and the count of zero-image baselines is still exactly zero. The censoring-free statement holds and gets stronger: across the 50 records here, no P1-family sub-flag is true anywhere.

**Does not.** Three things, stated plainly:

1. **This is one operator, not four.** No record here enters any agreement statistic, and this file does not discharge the outstanding action named in `screen_frame.json` -- a fresh independent four-screener re-coding under v1.2. `screener_id` is an administrative label.
2. **The access constraint still binds.** 22.7% unreachable in this block is above the 15% threshold, so within the block the bounding interval remains the reportable form. Enlarging the sample cannot fix that; only recovering full texts can. Two of the five unreachable records look like they would have been included, and one of those states a patient-level split in its abstract -- information that is simply lost.
3. **Eight records are flagged for adjudication** (`flag_for_adjudication=true`): 42225843 and 33713959 on version-of-record grounds; 41331277 on a genuine I3 question (a Spectralis *cross-hair* protocol yields isolated B-scans, not a volume stack); 41633187, where the abstract's patient-level and slice-level recall/precision read like a two-unit classification result until the Methods reveal *"Since the dataset in this study included only CT data from CRC patients, TNs were set to 0"*; 34828081 on a split unit where a table names both patients and images; 40134559, where the headline AUC comes from thresholding a regression output; 35562596, where the split is patient-level but the metric denominator cannot be recovered; and 41444372, where the scored unit, the split unit and the patient count are all undecidable. None of the eight changes P1, which is zero under every reading of all eight.

**The most valuable single find available here was a paper that reports a zero-image baseline. There isn't one.** The closest anything comes is PMID 40121941's clinical-factors-only model, which measures a pixel-free comparator and reports its AUC -- and still never asks whether its imaging benchmark is solvable from position or acquisition metadata alone. The second closest is PMID 41087406, which does model an ordered slice index and its label distribution, and is a study of mouse tibiae.

The record that states this project's case most economically needs no gloss. PMID 40002836 splits 3,194 MRI images from 60 patients **by image file path** with `train_test_split`, reports 100.00% accuracy and an AUC of 1.0000 with a 95% CI of [1.0000, 1.0000], and writes: *"No data leakage occurred, as the train-validation-test split (80-10-10) ensured that test data remained completely unseen during training."* No constant baseline, no positional baseline, no permutation null, no patient-level split, anywhere in the paper.

