# Audit target list — zero-image null models on public medical imaging benchmarks

Compiled 2026-07-28. Companion machine-readable file: `paper/audit_targets.json`.

**Scope.** This document identifies public medical imaging datasets on which the two
zero-image null models already implemented in this repository can be run, the published
performance numbers those nulls would be compared against, and the prior art that
constrains what we may claim is new.

The two nulls, restated so the entry criteria are unambiguous:

| null | inputs it needs | implementation |
|---|---|---|
| POSITIONAL | subject id, slice index (or ordering), binary label, train/test split | `pipeline/s12_rempe.py::positional_scores`, `::run_positional_baseline_on_their_labels` |
| METADATA | subject id, one or more acquisition/provenance fields, label, split | `pipeline/s08_belowchance.py` |

**Entry criterion for Part 1.** A dataset qualifies as a *label-file-only* audit target if
the four positional fields above can be obtained **without downloading pixel data and
without a data-use agreement covering pixels**. This is the property that makes an audit
of benchmarks we do not hold possible at all, so it is recorded explicitly and
pessimistically for every entry.

---

## 0. A correction that must be made before anything else is written

The brief circulated for this task quotes the anchor result as

> ZERO-IMAGE POSITIONAL BASELINE 0.851 [0.821, 0.880] slice-level
> the same baseline, PATIENT level 0.424 [0.298, 0.547]

Those numbers are correct, but they are computed on the **DWI** label file
(`dwi_slice_level_labels.csv`), not the T2 file. I re-downloaded both label files from
the public GitHub repository today and re-ran `run_positional_baseline_on_their_labels`
against each:

| published label file | slice AUC | naive 95% CI | patient AUC | clustered 95% CI |
|---|---|---|---|---|
| `dwi_slice_level_labels.csv` | **0.8514** | [0.8207, 0.8804] | **0.4240** | [0.2982, 0.5471] |
| `t2_slice_level_labels.csv`  | **0.8542** | [0.8220, 0.8843] | **0.5058** | [0.3808, 0.6316] |

Both reproduce exactly from a clean download, which is the strongest possible form of the
claim. But the persisted artefact `pipeline_out/rempe/positional_baseline.json` contains
the **T2** run (`"source": "/Volumes/Research/fastmridatasets/prostate/labels/t2_slice_level_labels.csv"`),
while the docstring waterfall in `pipeline/s12_rempe.py` lines 272-278 quotes the **DWI**
numbers. The number in the manuscript and the number in the released artefact currently
disagree. Rempe et al. work on prostate diffusion data, so DWI is the correct arm; the
persisted JSON needs to be regenerated against the DWI file before submission, or the
manuscript must say T2. Either is defensible. Silently shipping both is not.

Both arms support the same conclusion (slice-level ≈ their headline, patient-level ≈
chance), so this is a bookkeeping fix, not a result change.

---

## Part 1 — Public datasets with slice-level labels

Verification status is recorded per row. `VERIFIED` means I downloaded the label file or
read the column headers myself in this session. `PAGE` means a dataset page or paper
stated it and I quote the source. `UNVERIFIED` means I could not confirm it and it must
be checked before use.

### Tier 1 — label file downloaded and schema confirmed this session

#### 1.1 fastMRI Prostate (Tibrewala et al. 2024) — the anchor, already run
- Modality/organ: 3T bp-MRI (T2 and DWI), prostate.
- n patients: **312**. n slices: **9,508** (T2), **9,490** (DWI). `VERIFIED` — row counts
  of the downloaded CSVs; matches Rempe et al.'s "312 subject and a total of 9508 slices".
- Per-slice labels public: **yes**, as PI-RADS per slice.
- Exact location: `https://raw.githubusercontent.com/cai2r/fastMRI_prostate/main/fastmri_prostate/data/t2_slice_level_labels.csv`
  and `.../dwi_slice_level_labels.csv` (also `volume_exam_labels.csv`, 312 rows).
- Schema `VERIFIED`: `,fastmri_pt_id,slice,PIRADS,fastmri_rawfile,data_split,folder`
- Official split: **yes**, in-file `data_split` column — training 6,647 / validation 1,462
  / test 1,399 slices (T2). Patient-disjoint.
- Access: label CSVs are in a public GitHub repo, no registration, no DUA. Pixel data needs
  NYU registration; **we do not need it.**
- Why it matters: this is the existence proof for the whole method. The audit needs one
  `curl` and no GPU.

#### 1.2 fastMRI+ (Zhao et al. 2022, Sci Data) — knee and brain pathology, per slice
- Modality/organ: MRI knee (coronal PD/PDFS) and brain (AXFLAIR/AXT1/AXT2).
- Per-slice labels public: **yes** — bounding boxes carrying a `slice` index.
- Exact location: `https://raw.githubusercontent.com/microsoft/fastmri-plus/main/Annotations/knee.csv`
  and `.../brain.csv`; volume rosters at `.../knee_file_list.csv`, `.../brain_file_list.csv`.
- Schema `VERIFIED`: `file,slice,study_level,x,y,width,height,label` — 16,167 knee
  annotation rows, 8,213 brain rows, 1,172 knee volumes in the file list.
- Reported scale (`PAGE`, Nature Sci Data): "16154 subspecialist expert bounding box
  annotations and 13 study-level labels for 22 different pathology categories on the
  fastMRI knee dataset"; ~7,570 boxes and 30 categories for brain.
- Official split: **no** classification split published in the annotations repo; the
  underlying fastMRI train/val split applies to the volumes.
- **Feasibility caveat, stated plainly:** the annotation file lists *positive* slices only.
  Negatives are implicit, so computing relative slice position requires the slice count per
  volume, which comes from the fastMRI HDF5 headers (free registration; header read, not a
  pixel download). We hold fastMRI, so this is cheap for us and *moderately* expensive for
  a third party. Do not describe this one as "label file only" without the caveat.
- Repo note (`PAGE`): the maintainers themselves warn the labels are "an indicatition of
  where a pathology could be present" rather than adjudicated ground truth.

#### 1.3 DeepLesion (Yan et al. 2018, NIH CC) — strongest pure label-file target
- Modality/organ: CT, whole body, 8 coarse lesion types.
- Counts (`PAGE`, arXiv:1710.01766 / dataset docs): "32,735 lesions in 32,120 bookmarked CT
  slices from 10,594 studies of 4,427 unique patients". `VERIFIED`: `DL_info.csv` has
  32,735 data rows and 4,427 distinct `Patient_index`.
- Exact location: `DL_info.csv`, mirrored at
  `https://huggingface.co/datasets/farrell236/DeepLesion/resolve/main/DL_info.csv`
  (8.5 MB); original NIH Box release. **No pixel download required.**
- Schema `VERIFIED`: `File_name, Patient_index, Study_index, Series_ID, Key_slice_index,
  Measurement_coordinates, Bounding_boxes, Lesion_diameters_Pixel_,
  Normalized_lesion_location, Coarse_lesion_type, Possibly_noisy, Slice_range,
  Spacing_mm_px_, Image_size, DICOM_windows, Patient_gender, Patient_age, Train_Val_Test`
- Official split: **yes**, in-file `Train_Val_Test` — 22,919 / 4,889 / 4,927 rows.
  `VERIFIED` patient-disjoint between the val and test typed subsets (0 shared patients).
- Licence: CC BY-SA 4.0 on the HuggingFace mirror; NIH terms on the original.
- **The label file publishes the confound directly.** `Normalized_lesion_location` is the
  lesion centroid in normalised volume coordinates — the z component *is* the positional
  feature. No inference of position is required at all.
- **Feasibility probe run this session** (see Part 2.4 for the honest reading): the 8-class
  `Coarse_lesion_type` labels exist only in the val and test splits (4,889 / 4,927 rows,
  patient-disjoint). Fitting a 20-bin `P(type | normalized z)` on val and applying to test:
  **56.0% accuracy vs 22.3% majority class**, zero pixels.

#### 1.4 Duke Breast Cancer MRI (Saha et al. / Mazurowski lab, TCIA)
- Modality/organ: DCE-MRI, breast. n patients: **922** (`PAGE`, TCIA: "The breast MRI
  dataset contains 922 patients gathered in Duke Hospital from 1 January, 2000 to 23 March,
  2014").
- Per-slice labels public: **yes, derivable exactly** — the annotation file gives the
  inclusive slice range of the tumour box per patient.
- Exact location: `Annotation_Boxes.csv` at
  `https://www.cancerimagingarchive.net/tcia-downloads/duke-breast-cancer-mri-da-other-1/annotation_boxes/`
  (community mirror used for schema check:
  `https://raw.githubusercontent.com/kinyanjjui/BreastTumorMRI_Classification/main/Annotation_Boxes.csv`).
- Schema `VERIFIED`: `Patient ID,Start Row,End Row,Start Column,End Column,Start Slice,End Slice`
  — 922 data rows.
- Official split: **no**.
- Licence/access: TCIA, CC BY-NC 4.0 class terms, no DUA for the supporting tabular files.
- **A canonical slice-level task already exists and is defined by the data owners.** The
  Mazurowski lab tutorial (`PAGE`): "take all 2D slices that contain a tumor bounding box
  to be **positive**, (labeled with "1"), and all other slices at least five slices away
  from positive slices to be negative ("0")". That definition is *purely positional by
  construction* and is exactly what the null tests. Their worked example extracts
  "2,600 examples per class".
- Caveat: total slices per series must come from the TCIA series metadata CSV (tabular, no
  pixels). Also every patient in this cohort has cancer, so the slice-level task is
  within-patient localisation, not diagnosis — say so.

### Tier 2 — label file public, but a join or a download of masks is required

#### 1.5 RSNA 2019 Intracranial Haemorrhage Detection — biggest impact, real friction
- Modality/organ: non-contrast head CT, 5 ICH subtypes + `any`.
- Counts (`PAGE`, RSNA ATLAS card): "874,035 images" from "over 25,000 CT brain scans",
  three contributing institutions.
- Per-slice labels public: **yes** — this benchmark's *official competition metric is
  per-image (per-slice) weighted multi-label log loss*, which is precisely the evaluation
  unit our critique targets. That makes it the highest-impact single target on this list.
- Exact location: `stage_2_train.csv` via Kaggle
  (`https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data`) and
  RSNA MIRA (`https://mira.rsna.org/dataset/1`).
- Official split (`PAGE`, ATLAS card): "Data from each institution were divided into sets of
  500 examinations, and the last 100 examinations in each segment were selected for the test
  and validation sets", "disjoint at the patient level".
- Licence: RSNA MIRA Dataset Research Use Agreement (click-through; free).
- **Blocking caveat, do not paper over it.** `stage_2_train.csv` is keyed by
  `ID_<SOPInstanceUID>_<subtype>` and carries *only* the label. It contains no study id, no
  patient id and no slice position. Those live in the DICOM headers. So this is **not** a
  label-file-only audit as released: it needs either (a) header extraction from the ~450 GB
  image download, or (b) a third-party published metadata CSV mapping SOPInstanceUID →
  StudyInstanceUID + ImagePositionPatient. Several such CSVs circulate from the competition
  era; **any one we use must be provenance-checked and cited, and if we cannot verify one,
  we should say the audit was not feasible from the label file alone.** That is itself a
  reportable finding about benchmark release practice.
- The ATLAS card also documents label noise worth quoting: "under-labeling" occurred when
  "a single image label" was mistakenly used "to reflect the entire examination".

#### 1.6 PI-CAI (Saha et al., Grand Challenge) — likely a NEGATIVE result, and that is useful
- Modality/organ: bp-MRI, prostate, csPCa.
- Counts (`PAGE`, Zenodo 6624726): "1500 anonymized prostate biparametric MRI scans from
  1476 patients, acquired between 2012-2021"; 425 positive / 1,075 negative cases.
- Labels public **without images**: **yes** — `https://github.com/DIAGNijmegen/picai_labels`
  ships human-expert and AI-derived csPCa lesion delineations as standalone `.nii.gz` label
  volumes (1,295/1,500 human-expert cases) plus `clinical_information/marksheet.csv`.
  Per-slice positivity is derivable from the masks; the masks are label volumes, not pixels.
- Official split: **yes** — `picai_baseline` publishes 5-fold CV splits ("we prepared 5-fold
  cross-validation splits of all cases with an expert-derived csPCa annotation"), and
  challenge submissions "were trained via 5-fold cross-validation using the exact same splits".
- Licence: CC BY-NC 4.0.
- **Why it belongs on the list even though we expect the null to fail here.** PI-CAI's
  official metrics are *patient-level AUROC* and *lesion-level average precision* — it does
  not report a slice-level number at all. If the positional null lands near chance on the
  patient-level metric while matching a slice-level metric we compute ourselves, that is a
  clean demonstration that the remedy works, using the most prominent current prostate
  benchmark as the positive control. Report it as a benchmark that already does the right
  thing. Do not round this one up.

#### 1.7 RSNA 2023 Abdominal Trauma Detection (RATIC)
- Modality/organ: contrast-enhanced abdominal CT.
- Per-slice labels public: **yes** — `image_level_labels.csv`, described on the competition
  data page (`PAGE`) as "Train only. Identifies specific images that contain either bowel
  or extravasation injuries", keyed by `patient_id` and `series_id`.
- Location: `https://www.kaggle.com/competitions/rsna-2023-abdominal-trauma-detection/data`;
  dataset paper `https://pubs.rsna.org/doi/10.1148/ryai.240101`.
- Official split: competition train/test; test labels withheld.
- Feasibility: `image_level_labels.csv` carries `patient_id`/`series_id` **and** an instance
  number, so subject id and ordering are present. Slice counts per series still need the
  series manifest. Better than RSNA ICH, worse than DeepLesion.

#### 1.8 RSNA 2022 Cervical Spine Fracture Detection
- Counts (`PAGE`, RSNA dataset paper `10.1148/ryai.230034`): 3,112 CT scans; 1,445 studies
  with fractures; **235 studies with bounding-box annotations**.
- Per-slice labels: `train_bounding_boxes.csv` — "contains information regarding the
  fracture bounding boxes for a subset of the training set", with a slice/instance number.
  `train.csv` itself is study-level per-vertebra (C1–C7), not per slice.
- Feasibility: the per-slice subset is only 235 studies, so power is limited. Mid-tier.

#### 1.9 LUNA16 / LIDC-IDRI
- LIDC-IDRI (`PAGE`, TCIA): "all 1,010 patients", 1,018 cases, licence CC BY 3.0; nodule
  annotations distributed as separate XML files alongside the DICOMs.
- LUNA16 (`PAGE`, grand-challenge Data page): "888 CT scans are included", "Creative Commons
  Attribution 4.0 International License"; `annotations.csv` and `candidates_V2.csv` publish
  nodule and candidate **world coordinates including z (mm)** plus a `seriesuid`, and
  LUNA16 publishes an official 10-fold subset split.
- Feasibility for the positional null: **good at candidate level** — `candidates_V2.csv`
  gives ~750k candidates with a z coordinate, a series id and a binary label, no pixels.
- Impact caveat: the benchmark metric is FROC on candidates, not slice-level AUC, and z in
  a chest CT is again strongly anatomical. Treat as a supporting case, not a headline.

### Tier 3 — investigated and deliberately excluded, with reasons

These are as important to the paper's credibility as the inclusions.

- **CQ500** (`PAGE`: 491 scans, 193,317 slices, Chilamkurthy et al.): labels in `reads.csv`
  are **scan-level** (three radiologist reads per scan), not per-slice. The positional null
  as specified does not apply. Slice-level masks exist only via the third-party Seg-CQ500
  release (Zenodo 8063221). **Excluded from the positional audit.**
- **BraTS, KiTS, Medical Segmentation Decathlon, AMOS, TotalSegmentator**: these are
  *segmentation* benchmarks scored by Dice/HD95. Per-slice labels are derivable from the
  released masks, but (a) the masks are generally distributed in the same archive as the
  images, so the "no pixel download" property fails, and (b) there is no published
  slice-level classification number to compare a null against. A positional null against a
  Dice score is not a meaningful comparison. **Excluded**; mention in the paper only as the
  boundary of the method's applicability.
- **PROSTATEx**: `ProstateX-Findings-Train.csv` gives finding position in patient
  coordinates (mm), which requires DICOM headers to convert to a slice index, and the
  challenge metric is per-finding AUC. Low feasibility, mid impact. **Deferred.**
- **Prostate158**: small (n≈158), segmentation-oriented, no slice-level classification
  benchmark. **Deferred.**
- **MRNet (Stanford knee)**: exam-level labels only. Out of scope by construction.

### Unverified candidates — check before citing
`CT-ICH` (PhysioNet, ~82 patients with per-slice haemorrhage labels), `INSTANCE 2022`,
`COVIDx CT-3`, `MosMedData`, `SPIDER`, `VerSe`. I did not confirm their label files this
session. They are listed in the JSON with `"verification": "UNVERIFIED"` and must not enter
the manuscript without a download.

---

## Part 2 — Published slice-level performance numbers

Every number below carries the source and the evaluation unit. Where I could not obtain an
exact figure, the row says so rather than guessing.

### 2.1 fastMRI Prostate / Rempe et al. 2024 — the comparison already established
Rempe M, Hörst F, Seibold C, Hadaschik B, Schlimbach M, Egger J, Kröninger K, Breuer F,
Blaimer M, Kleesiek J. *Tumor likelihood estimation on MRI prostate data by utilizing
k-Space information.* arXiv:2407.06165.

Abstract, verbatim: the dataset is "a publicly available MRI raw dataset with 312 subject
and a total of 9508 slices"; the full k-space approach reaches "AUROC of $86.1\%\pm1.8\%$";
the R=16 PCA arm reaches "AUROC of $71.4\%\pm2.9\%$".

The full Table II transcription is already in the repository at
`pipeline/s12_rempe.py::REPORTED` (headline 86.1±1.8; image-only x0 85.7±1.6; PCA x2
magnitude 81.3±2.2; PCA x2 magnitude+phase 80.9±2.1; PCA x2 magnitude+k-space 81.1±1.9;
GRAPPA x2 arms 80.7/80.7/78.3). Evaluation unit recorded in the same file as
"slice-level AUROC, unclustered bootstrap (1000 iterations)". No patient-level number is
reported by the authors — that absence is the point.

**Null vs published:** 0.851 (DWI) / 0.854 (T2) slice-level, zero images, against a
published slice-level headline of 0.861. Patient-level 0.424 / 0.506.

### 2.2 RSNA 2019 ICH — published numbers found so far
Ngo DT, Nguyen TTB, Nguyen HT, Nguyen DB, Nguyen HQ, Pham HH. *Slice-level Detection of
Intracranial Hemorrhage on CT Using Deep Descriptors of Adjacent Slices.* arXiv:2208.03403,
IEEE SSP 2022.
- RSNA ICH: weighted multi-label log loss **0.05341**, single ResNet-50, private test set of
  3,518 studies; the authors state they "obtain a single model in the top 4%
  best-performing solutions of the RSNA ICH challenge". Evaluation unit: **per-slice**
  (the official competition metric). Split: official competition split.
- CQ500 (external): Table I, "Experimental results measured by AUC score on CQ500 dataset",
  evaluation unit **study-level** (max probability over slices), 490 of 500 studies:
  ICH-any 0.9419 → 0.9612; intraparenchymal 0.9544 → 0.9691; intraventricular 0.9310 →
  0.9832; subarachnoid 0.9574 → 0.9596; subdural 0.9521 → 0.9694; extradural 0.9731 →
  0.9814; **mean 0.9520 → 0.9710**. (Left column is their reproduction of the Chilamkurthy
  baseline, right column their method.) Trained on the RSNA public training set only.
- Also located but not yet extracted: arXiv:2005.10992 reports "our best single model
  achieves a weighted log loss of 0.0522" on the RSNA challenge; arXiv:2205.07556 is a
  transformer solution claiming to beat the first-place single model. **Both need their
  exact tables read before citation.**

**Note for the write-up.** Because the RSNA metric is a log loss on slice-level
probabilities, the natural null comparison is the log loss of the positional prior, not an
AUC. That is a cleaner and more damaging comparison than an AUC if we can compute it — the
competition's own scoreboard becomes the yardstick. It depends entirely on resolving the
SOPInstanceUID → position join in 1.5.

### 2.3 fastMRI+ — numbers not yet pinned down
The fastMRI+ data descriptor (Zhao et al., *Sci Data* 2022, `10.1038/s41597-022-01255-z`)
releases annotations rather than a benchmark leaderboard. Downstream slice-level
classification papers exist but I did not obtain an exact table this session. **Do not cite
a fastMRI+ classification number until it is read from the source table.**

### 2.4 DeepLesion — feasibility probe, and the honest reading
Zero-image result computed this session (val→test, patient-disjoint, official
`Train_Val_Test` field, 20-bin `P(Coarse_lesion_type | normalized z)`):

| predictor | 8-class accuracy on the 4,927-row test split |
|---|---|
| published normalised z-position alone, no pixels | **0.5602** |
| majority class | 0.2233 |

**Read this conservatively.** DeepLesion's eight coarse classes are *bone, abdomen,
mediastinum, liver, lung, kidney, soft tissue, pelvis* — they are anatomical regions. A
z-coordinate predicting an anatomical region is not a confound; it is the task. So this
result is a good demonstration that the null is *cheap and informative*, and a good
argument that papers should report it, but it is **not** evidence that DeepLesion
lesion-type papers are unsound. If we use it, frame it as "the reference level a lesion-type
classifier must clear", not as a debunking. A benchmark where the null is legitimately high
is exactly the kind of case the paper should present honestly.

Published DeepLesion lesion-type accuracies were **not** located this session; the
retrievable numbers are detection sensitivities ("sensitivity of 81.1% with five false
positives per image") which are a different task. **Needed before Part 2 is complete.**

### 2.5 Duke Breast MRI — numbers not yet pinned down
The data owners' own tutorial defines the slice-level task but reports no metric ("The
tutorial does not report any accuracy or AUC values" in part 1). Downstream papers using the
Duke slice task exist; none produced an extractable metric this session. **Open.**

### 2.6 PI-CAI — reported at the right unit, by design
PI-CAI reports patient-level AUROC and lesion-level AP. There is no official slice-level
number to attack. This is the paper's positive example.

**Summary of Part 2 completeness.** Fully sourced and quotable: 2.1 (Rempe, complete table
already transcribed in code) and 2.2 (Ngo et al., complete tables). Computed by us: 2.4.
Open and explicitly flagged: 2.3, 2.5, and the two additional RSNA papers in 2.2. Given the
brief's warning that "a misattributed number would be a serious error in a paper that
criticises other people's numbers", I have left the open ones open rather than filling them
from memory.

---

## Part 3 — Prior art, stated against our own interest

### 3.1 What is already firmly established in the literature

**Shortcut learning is a named, well-documented phenomenon.**
Geirhos et al., *Shortcut learning in deep neural networks*, Nat Mach Intell 2020, is the
canonical statement. In medical imaging specifically:
- Zech et al. 2018 (PLOS Med) — chest radiograph models keyed on hospital system.
  *(well known; verify DOI before citing, I did not confirm it this session)*
- Badgeley MA, Zech JR, Oakden-Rayner L, et al. *Deep learning predicts hip fracture using
  confounding patient and healthcare variables.* npj Digit Med 2019;2:31.
  DOI `10.1038/s41746-019-0105-1`. **This is the closest prior art to our METADATA null.**
  Verbatim from the abstract: fracture "was predicted moderately well from the image
  (AUC = 0.78) and better when combining image features with patient data (AUC = 0.86) or
  patient data plus hospital process features (AUC = 0.91)"; and on a test set balanced
  across patient and process variables "the model performed randomly (AUC = 0.52, 95% CI
  0.46–0.58), indicating that these variables were the main source of the model's fracture
  predictions." Scanner model was predictable from the radiograph at AUC = 1.00.
- DeGrave AJ, Janizek JD, Lee S-I. *AI for radiographic COVID-19 detection selects shortcuts
  over signal.* Nat Mach Intell 2021;3(7). DOI `10.1038/s42256-021-00338-7`.
- Oakden-Rayner L, Dunnmon J, Carneiro G, Ré C. *Hidden stratification causes clinically
  meaningful failures in machine learning for medical imaging.* ACM CHIL 2020.
  DOI `10.1145/3368555.3384468`.
- Hill BG et al., Sci Rep 2024 — ResNet18 predicts whether a patient avoids refried beans
  (AUC 0.63) or beer (AUC 0.73) from a knee radiograph. The reductio version of our argument.
- Ong Ly C et al., npj Digit Med 2024 — across 13 datasets, "model performance is frequently
  overestimated by up to 20% on average due to shortcut learning of hidden data acquisition
  biases", and they release a bias-corrected estimator (PEst).
- Lin et al., *Shortcut Learning in Medical Image Segmentation*, MICCAI 2024,
  arXiv:2403.06748.

**Slice-level evaluation of 3D volumes is already known to inflate performance.**
- Yagis E, Atnafu SW, García Seco de Herrera A, Marzi C, Scheda R, Giannelli M, Tessa C,
  Citi L, Diciotti S. *Effect of data leakage in brain MRI classification using 2D
  convolutional neural networks.* Sci Rep 2021;11:22544. DOI `10.1038/s41598-021-01681-w`.
  Slice-level CV "erroneously boosted the average slice level accuracy on the test set by
  30% on OASIS, 29% on ADNI, 48% on PPMI and 55% on a local de-novo PD Versilia dataset",
  and on **randomly labelled** data produced "about 96% of (erroneous) accuracy (slice-level
  split) and 50% accuracy (subject-level split)".
- Tampu IE et al. *Inflation of test accuracy due to data leakage in deep learning-based
  classification of OCT images.* Sci Data 2022. Inflation of MCC by 0.07–0.43.
- Wen J, Thibeau-Sutre E, Diaz-Melo M, et al. *Convolutional neural networks for
  classification of Alzheimer's disease: Overview and reproducible evaluation.* Med Image
  Anal 2020;63:101694. DOI `10.1016/j.media.2020.101694`. Two findings that bear directly on
  us: "more than half of the surveyed papers may have suffered from data leakage and thus
  reported biased performance", and — a trivial-baseline result — "the different CNN
  approaches did not perform better than a SVM with voxel-based features."

**Leakage as a cross-disciplinary reproducibility failure.**
- Kapoor S, Narayanan A. *Leakage and the reproducibility crisis in machine-learning-based
  science.* Patterns 2023;4(9). Affects "at least 294 papers across 17 disciplines";
  proposes model info sheets. Our tool is a domain-specific instance of their programme.

**Field-level critiques that already recommend better baselines.**
- Varoquaux G, Cheplygina V. *Machine learning for medical imaging: methodological failures
  and recommendations for the future.* npj Digit Med 2022;5:48.
  DOI `10.1038/s41746-022-00592-y`.
- Roberts M et al. *Common pitfalls and recommendations for using machine learning to detect
  and prognosticate for COVID-19 using chest radiographs and CT scans.* Nat Mach Intell
  2021;3. Headline: "none of the models identified are of potential clinical use due to
  methodological flaws and/or underlying biases."

### 3.2 What those works leave open

1. **They diagnose the mechanism; they do not quantify the share.** Yagis, Tampu and Wen all
   measure the *inflation caused by an incorrect split*. None of them asks the different
   question we ask: under a **correct, patient-disjoint split**, how much of a published
   slice-level number is recoverable from slice position alone? Rempe et al.'s split *is*
   patient-disjoint. Our 0.851 is therefore not a leakage result in the Yagis sense at all,
   and the paper must be explicit about that distinction or a reviewer will conflate them
   and reject us for redundancy.
2. **Badgeley's control is a negative control; ours is a positive control.** Badgeley
   balanced the confounders and watched the model collapse to 0.52. We fit a model on the
   confounder alone and watch it match the published number. Same phenomenon, opposite
   direction, and the positive-control form is the one an auditor can run without the
   images. That is a genuine methodological difference, but it is a difference in *form*,
   not in *discovery*. Say so.
3. **No one has run these nulls systematically across benchmarks.** Every prior work is
   one dataset family (brain MRI, OCT, chest X-ray, AD) or one review of others' papers.
   There is no released tool that takes a label file and returns the null.
4. **The "audit without the pixels" property is, as far as I can tell, unexploited.** The
   observation that the four fields needed are exactly the fields most benchmarks publish
   openly — and that this makes third-party auditing possible at zero data cost — is not
   something I found stated anywhere.

### 3.3 What is genuinely new, and what is not — stated plainly

**Not new.** All of the following are already published and must be cited as such, not
presented as our findings:
- that shortcut learning occurs in medical imaging;
- that models exploit acquisition and process metadata (Badgeley 2019 is prior art for our
  metadata arm, and it is close prior art);
- that slice-level evaluation of volumetric data inflates reported performance;
- that patient-level is the correct evaluation unit;
- that leakage is widespread and under-reported across ML-based science;
- that trivial baselines sometimes match deep models (Wen 2020 already showed SVM ≈ CNN
  for AD).

**Plausibly new, and defensible:**
- (a) **The quantification under a correct split.** "A zero-image positional predictor
  attains 0.851 against a published 0.861 on the authors' own label file and split" is, to
  the best of my search, an unpublished class of measurement. The prior literature measures
  the cost of a *wrong* split; we measure the residual inflation of a *right* split.
- (b) **The family framing.** Positional and metadata nulls as two instances of one
  construct — "predictors that see no pixels" — applied uniformly, with the same
  bootstrap and the same reporting, across heterogeneous benchmarks.
- (c) **The zero-cost audit.** Auditing benchmarks we will never hold, from published label
  files, with no DUA and no GPU. This is the strongest novelty claim and the one that turns
  a Letter into an article.
- (d) **The released tool and the remedy** (report the null alongside every slice-level
  number; report patient-level; report the position-stratified metric, which is already
  implemented as `slice_auc_position_stratified` and collapses the DWI/T2 null to ~0.55).

**The single biggest reviewer risk.** A reviewer who knows Yagis 2021 and Badgeley 2019 will
ask: "you have combined two known results and run them on more datasets — where is the
new idea?" The answer has to be (c), and (c) only survives if Part 1 delivers enough
benchmarks that the audit is genuinely systematic. **If the final audit covers only two or
three datasets, this should be submitted as a Letter, not an article.** That is the decision
this list exists to inform, and on today's evidence the credible count of *label-file-only*
targets is four (fastMRI prostate, fastMRI+, DeepLesion, Duke breast), with RSNA ICH as a
fifth conditional on the metadata join. Four to five is enough for an article; two is not.

### 3.4 Searches that returned nothing, recorded so the claim is falsifiable
I ran targeted searches for any published work using slice position alone as a predictive
baseline, including: `"slice number" OR "slice index" alone predicts diagnosis baseline no
image medical imaging null model`; `predicting pathology from "slice location" alone without
image features baseline AUC CT MRI confounder anatomical prior`; `"null model" OR "trivial
baseline" medical imaging AI benchmark audit zero-image chance performance`. The first and
third returned **zero results**; the second returned only body-part-regression work
(predicting position *from* the image, the inverse problem). PubMed queries combining
slice-level with leakage/shortcut/confound returned 6 records, none of which construct a
position-only predictor.

This is absence of evidence from a handful of queries, not proof of novelty. Before
submission someone should search Google Scholar and the MICCAI/MIDL/ML4H proceedings
directly, and check the RSNA ICH Kaggle solution write-ups — the competition community very
likely knew that slice position is predictive, and a public forum post predating us would
not sink the paper but must be acknowledged.

---

## Ranking: feasibility × impact

Feasibility 1–5 (5 = label file downloads in one command, schema confirmed, official split).
Impact 1–5 (5 = widely used benchmark whose *published* metric is slice-level).

| # | target | feas | imp | score | status |
|---|---|---|---|---|---|
| 1 | fastMRI Prostate (Rempe) | 5 | 5 | **25** | DONE — anchor; fix the DWI/T2 artefact mismatch |
| 2 | DeepLesion | 5 | 4 | **20** | probe run (0.560 vs 0.223); needs published comparators |
| 3 | RSNA 2019 ICH | 3 | 5 | **15** | highest impact; blocked on SOPInstanceUID→position join |
| 4 | fastMRI+ knee | 4 | 4 | **16** | schema confirmed; needs per-volume slice counts |
| 5 | Duke Breast MRI | 4 | 4 | **16** | schema confirmed; owner-defined slice task exists |
| 6 | fastMRI+ brain | 4 | 3 | **12** | as knee, smaller |
| 7 | PI-CAI | 3 | 4 | **12** | expected NEGATIVE — run it, report it as the positive example |
| 8 | RSNA 2023 abdominal trauma | 3 | 4 | **12** | `image_level_labels.csv` is genuinely per-slice |
| 9 | LUNA16 / LIDC-IDRI | 4 | 3 | **12** | candidate-level z is public; metric mismatch (FROC) |
| 10 | RSNA 2022 cervical spine | 3 | 3 | **9** | only 235 box-annotated studies |
| 11 | PROSTATEx | 2 | 3 | **6** | needs DICOM headers |
| 12 | CQ500 | 1 | 3 | **3** | scan-level labels only — **exclude** |
| 13 | BraTS/KiTS/MSD/AMOS/TotalSegmentator | 1 | 2 | **2** | segmentation benchmarks — **exclude** |

**Recommended execution order for the next phase:** 1 (fix) → 2 → 4 → 5 → 7 → 3 → 8.
Run PI-CAI (7) *before* the harder RSNA ICH join, because a null that fails is the fastest
available protection against the accusation that the tool only ever confirms itself.
