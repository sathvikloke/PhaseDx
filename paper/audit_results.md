# Audit results — zero-image null models against published medical imaging benchmarks

Run 2026-07-29. Tool: `pipeline/s14_trivialbaselines.py` (self-test passed before and
after the one change made during this run; see §7). Label-table preparation scripts:
`pipeline/audit_prep/`. Machine-readable results: `pipeline_out/trivial_baselines/*.json`,
one card per run in `*.md`.

---

## 0. Headline, stated before any table

**Six benchmarks were audited on seven label files, producing fifteen rows. Twelve rows
carry a defensible published comparator: six are MATCHED, three are PARTIAL, three are
NOT MATCHED. The remaining three rows are NON-COMPARABLE and are not scored.** All six
MATCHED rows come from a single benchmark — fastMRI Prostate under Rempe et al.'s
protocol. Every other benchmark audited either resisted the null outright or was matched
only in part.

That distribution is the finding, and it changes the framing the paper can support:

* **The claim "trivial baselines match published performance on medical imaging
  benchmarks" is not supported as a general statement.** It is supported for one
  benchmark, strongly and reproducibly, and partially for a second.
* **What generalises further than the match is the gap between the slice-level and the
  patient-level number.** On three of the six benchmarks a pixel-blind positional model
  scores 0.80–0.87 at the slice level and falls to chance at the patient level: fastMRI
  Prostate T2 0.854 → 0.506, DWI 0.851 → 0.424, fastMRI+ knee meniscus 0.873 → 0.510.
  Duke breast reaches 0.823 at the slice level and cannot be evaluated at the patient
  level at all, because all 922 patients are positive — a fourth way the same protocol
  problem shows up. Crucially this did *not* depend on whether the published number was
  matched, so it is not hostage to any comparability argument. It is the most defensible
  general claim the audit supports.
  **It is not universal.** DeepLesion stays high at both units (pelvis 0.977 slice, 0.954
  patient) because its labels are anatomical regions, and LUNA16 is at chance at both
  (0.534 slice, 0.581 patient). Both exceptions must be reported alongside the rule.
* **Two benchmarks are clean.** LUNA16 is decisively unmatched on its own metric (CPM
  0.0020 against a published >0.95 sensitivity at <1 FP/scan) and PI-CAI is unmatched at
  its own evaluation unit. Both belong in the paper at equal prominence.
* **A published location-only baseline already exists on DeepLesion.** Yan et al. (CVPR
  2018) Table 1 reports a "Location feature" baseline at 59.7% 8-class accuracy against
  their own 90.5%. This is closer prior art than `paper/audit_targets.md` §3.4 records,
  and the novelty section must be rewritten to acknowledge it. See §5.

---

## 1. How a verdict is assigned

The comparison statistic is the tool's trivial fraction,

> trivial fraction = (best zero-image baseline − chance) / (published − chance)

with chance = 0.5 for AUROC and the majority-class rate for multi-class accuracy. Its
interval propagates uncertainty in the *baseline only*; the published number enters as a
constant, so the interval is too narrow as a statement about the ratio.

| verdict | rule |
|---|---|
| **MATCHED** | the upper bound of the baseline's clustered 95% CI reaches or exceeds the published number (equivalently, trivial-fraction CI covers or exceeds 1) |
| **PARTIAL** | trivial fraction ≥ 0.30 but its CI lies wholly below 1 |
| **NOT MATCHED** | trivial fraction < 0.30, or the baseline is statistically indistinguishable from chance |
| **NON-COMPARABLE** | the published number is on a different cohort, split, label definition or metric and could not be reconstructed; no verdict is issued |

A MATCHED verdict licenses exactly one sentence: *this published evaluation protocol
certifies a number that a model with no access to the pixels also reaches.* It does not
license "the model learned nothing". Every card repeats this.

---

## 2. Results — one row per (dataset, published number)

### 2.1 Scored rows

| # | dataset | published number | source | our best zero-image baseline | trivial fraction [CI] | verdict |
|---|---|---|---|---|---|---|
| 1 | fastMRI Prostate **T2** | **0.861** slice AUROC | Rempe et al. 2024, arXiv:2407.06165, Table II gold standard (image+k-space) | **0.854** [0.812, 0.891] positional 20-bin | 0.981 [0.865, 1.084] | **MATCHED** |
| 2 | fastMRI Prostate **T2** | **0.809** slice AUROC | Rempe et al., Table II, PCA ×2 magnitude+phase | 0.854 [0.812, 0.891] | 1.146 [1.011, 1.266] | **MATCHED** (exceeds) |
| 3 | fastMRI Prostate **T2** | **0.714** slice AUROC | Rempe et al., R=16 PCA coil combination | 0.854 [0.812, 0.891] | 1.655 [1.459, 1.829] | **MATCHED** (exceeds) |
| 4 | fastMRI Prostate **DWI** | 0.861 slice AUROC | as row 1 | **0.851** [0.816, 0.887] positional 20-bin | 0.973 [0.876, 1.073] | **MATCHED** |
| 5 | fastMRI Prostate **DWI** | 0.809 slice AUROC | as row 2 | 0.851 [0.816, 0.887] | 1.137 [1.023, 1.253] | **MATCHED** (exceeds) |
| 6 | fastMRI Prostate **DWI** | 0.714 slice AUROC | as row 3 | 0.851 [0.816, 0.887] | 1.642 [1.478, 1.810] | **MATCHED** (exceeds) |
| 7 | DeepLesion | **0.905 ± 0.002** 8-class accuracy | Yan et al., CVPR 2018, arXiv:1711.10535, Table 1, "Triplet with type + location + size" | **0.557** [0.524, 0.578] positional 20-bin on published normalised z | 0.480 [0.431, 0.511] | **PARTIAL** |
| 8 | DeepLesion | **0.862** 8-class accuracy | same table, "Baseline: Multi-scale ImageNet feature" | 0.557 [0.524, 0.578] | 0.513 [0.460, 0.546] | **PARTIAL** |
| 9 | DeepLesion | **0.597** 8-class accuracy | same table, "Baseline: Location feature" (*their own* image-derived location baseline) | 0.557 [0.524, 0.578] | 0.889 [0.799, 0.947] | **PARTIAL** (see §5) |
| 10 | LUNA16 (FP-reduction track) | **">95% sensitivity at <1.0 FP/scan"** | Setio et al. 2017, LUNA16 challenge summary, arXiv:1612.08012 (combined solutions) | **CPM 0.0020**; sensitivity 0.0006 at 1 FP/scan | ≈0 | **NOT MATCHED** |
| 11 | PI-CAI | **0.91** (95% CI 0.87–0.94) case-level AUROC, AI system | Saha et al., *Lancet Oncol* 2024;25:879-887, [DOI](https://doi.org/10.1016/S1470-2045(24)00220-1) | **0.692** [0.626, 0.755] metadata CART, case level | 0.467 [0.307, 0.622] | **NOT MATCHED** ¹ |
| 12 | PI-CAI | **0.86** (0.83–0.89) case-level AUROC, 62 radiologists PI-RADS 2.1 | same | 0.692 [0.626, 0.755] | 0.532 [0.350, 0.708] | **NOT MATCHED** ¹ |

¹ Rows 11–12 carry a cohort caveat that would justify calling them NON-COMPARABLE; see
§3.5. They are scored as NOT MATCHED because the caveat cuts *against* the null (our
baseline had the easier cohort and still lost), so scoring them is the conservative
choice. Rows 7–9 use the majority class (0.236) as the chance anchor, not 0.5.

### 2.2 Non-comparable rows — audited, but no defensible published comparator

| # | dataset | zero-image result | why no verdict |
|---|---|---|---|
| 13 | fastMRI+ knee, "Meniscus Tear" per slice | positional 20-bin **0.873** [0.858, 0.886] slice AUROC; **0.510** [0.428, 0.592] patient | fastMRI+ is a data descriptor; no published slice-level classification number was located, and `paper/audit_targets.md` §2.3 already flags this as open. Also 199 of the 1,173 roster volumes (§3.3). |
| 14 | fastMRI+ knee, "any annotated finding" per slice | positional 20-bin **0.801** [0.779, 0.824] slice; **0.558** [0.470, 0.648] patient | as above |
| 15 | Duke Breast Cancer MRI, owner-defined slice task | positional 20-bin **0.823** [0.811, 0.834] slice AUROC; patient AUROC **undefined** | the Mazurowski lab tutorial defines the task but publishes no metric. No downstream number with the same task definition was located. |

---

## 3. Comparability, line by line

Nothing below is a hedge. Each item is a condition that had to hold for the row above to
mean anything, and where it did not hold the row was demoted.

### 3.1 fastMRI Prostate (rows 1–6) — fully comparable, and one correction to make

* **Same file.** The audit ran on the authors' own published label CSVs, downloaded from
  `github.com/cai2r/fastMRI_prostate`. SHA-256 of the T2 file is `d248d41c9915c3fe…`,
  DWI `e22a354132cce884…`; both match the copies used in the earlier session, so the
  numbers reproduce from a clean download.
* **Same split.** The in-file `data_split` column, patient-disjoint: 6,647 training /
  1,462 validation / 1,399 test slices (T2). Validation rows were excluded from both
  arms, as the tool's default.
* **Same label.** PI-RADS > 2 per slice.
* **Same evaluation unit.** Slice-level AUROC, which is what Rempe et al. report and the
  only unit they report.
* **Test arm is small.** 46 patients, 1,399 slices, 68 positive slices (T2). The
  slice-level interval is clustered on patient; the patient-level interval on 46 subjects
  is wide and is reported as such.
* **We still cannot reproduce their pipeline.** Their protocol on our data gives 0.616,
  not 0.809. The claim these six rows support is that *their evaluation protocol* is
  matched by a zero-image baseline — not that their model learned nothing.
* **Correction to `paper/audit_targets.json`.** Its `anchor_correction` block asserts
  "Rempe et al. work on prostate diffusion data, so DWI is the correct arm". The evidence
  points the other way: Rempe et al.'s abstract says "312 subject and a total of 9508
  slices", and 9,508 is the exact row count of `t2_slice_level_labels.csv` (DWI has
  9,490). **T2 is the correct arm, the persisted artefact
  `pipeline_out/rempe/positional_baseline.json` is already right, and it is the docstring
  waterfall at `pipeline/s12_rempe.py:272-278` that quotes the wrong arm.** Both arms are
  reported here so the conclusion does not depend on resolving it, but the recommendation
  in `audit_targets.json` should be reversed before submission.

### 3.2 DeepLesion (rows 7–9) — their conditions were reconstructed, not assumed

The first attempt at this row would have been wrong and is worth recording. Yan et al.'s
Table 1 test set has 4,927 samples, which is *exactly* the row count of DeepLesion's
official `Train_Val_Test == 3` split — a coincidence that invites a false match. Their
own text says otherwise, verbatim: *"Among the labeled samples, we randomly select 25% as
training seeds to predict pseudo-labels, 25% as the validation set, and the other 50% as
the test set. There is no patient-level overlap between all subsets."*

So the reported row is **not** the official-split number. `pipeline/audit_prep/
deeplesion_yan_conditions.py` rebuilds their partition — a random patient-disjoint
25/25/50 split of the 9,816 type-labelled rows, fitting on the 25% seed set only —
and repeats it over 200 draws so the comparison is not hostage to one seed. Mean seed
size 2,454 rows, mean test size 4,900 (they report 4,927). Under those conditions the
zero-image accuracy is **0.5571**, sd 0.0131, [0.5243, 0.5778] over partitions, against a
majority class of 0.2361.

For reference, the official-split number (fit on the 4,889-row validation split, applied
to the 4,927-row test split, 0 shared patients) is **0.5602** [0.5344, 0.5868] with a
patient-clustered bootstrap — nearly identical, which is reassuring but is not what
appears in the scored rows.

**Read DeepLesion's eight classes conservatively.** They are *bone, abdomen, mediastinum,
liver, lung, kidney, soft tissue, pelvis* — anatomical regions. A z-coordinate predicting
an anatomical region is the task, not a confound. This row is the reference level a
lesion-type classifier must clear. It is not evidence that DeepLesion papers are unsound,
and the paper must not use it that way.

**Metadata finding worth its own sentence.** On the official split, one-vs-rest for lung
lesions is reached at slice AUROC **0.911** by the `DICOM_windows` column alone — the
window/level stored in the header, which is `-1500, 500` for lung-reconstructed series
and `-175, 275` otherwise. Position alone gives 0.872; the two together give 0.962. Per
class, the best zero-image ceiling on the official split runs: pelvis 0.982, lung 0.962,
mediastinum 0.957, kidney 0.896, abdomen 0.886, liver 0.876, bone 0.832, soft tissue
0.831. No published per-class AUROC was located, so these are reference levels only.

### 3.3 fastMRI+ knee (rows 13–14) — not a label-file-only target, and a partial cohort

fastMRI+ publishes positive annotations only. Negative slices are implicit, so the table
cannot be built from the annotation file alone: the slice count of each volume comes from
the fastMRI HDF5 headers. That is a header read, not a pixel download, but it needs the
fastMRI registration and the archive. **Do not describe fastMRI+ as label-file-only.**

Worse for coverage: the fastMRI+ knee roster is 1,173 volumes and we hold 199 of them
(the fastMRI knee validation set). 155 of those 199 carry at least one annotation. The
audit therefore runs on 199 volumes / 7,135 slices, a 17% subset that is *not* the subset
any published number would use. Evaluation is 5-fold subject-level CV; there is no
official classification split.

The maintainers themselves warn the labels are "an indicatition of where a pathology
could be present" rather than adjudicated ground truth.

fastMRI+ **brain** was investigated and dropped: only 73 of the 1,001 roster volumes are
held locally, which is too few to report.

### 3.4 Duke breast (row 15) — the task is positional by construction

Positives and negatives here are *defined* by slice position relative to the tumour box
(the tutorial's own rule: inside the box is positive, ≥5 slices away is negative,
everything between is discarded). A high positional null is therefore a tautology, not a
discovery, and the row is reported to quantify the tautology rather than to indict
anyone. Slice counts came from the TCIA `getSeries` metadata (tabular, CC BY-NC 4.0, no
DUA); the modal `ImageCount` per patient was validated against the annotation file —
for all 922 patients the annotated end slice is strictly inside the series, and the modal
and maximum counts agree.

Every patient in this cohort has cancer, so **patient-level AUROC is undefined** (922 of
922 subjects positive) and the harness correctly reports it as unavailable rather than
inventing a value. The slice task is within-patient localisation, not diagnosis.

### 3.5 PI-CAI (rows 11–12) — different cohort, and the positional null does not apply

* **The published 0.91/0.86 are on the hidden 1,000-case testing cohort** (400-case
  subset for the reader comparison), from four centres in the Netherlands and Norway.
  **Our baseline is on the public 1,500-case Training and Development set.** These are
  different cohorts. A strict reading makes rows 11–12 non-comparable, and that reading is
  defensible.
* They are scored anyway because the caveat runs against the null: our baseline had the
  larger, more heterogeneous public cohort and the benchmark's own official 5-fold splits,
  and it still landed at 0.692 against 0.91. Reporting it as NOT MATCHED is the
  conservative call, not the generous one.
* **The positional null is not applicable to PI-CAI as released.** The marksheet has one
  row per case and no slice index; the harness measured the positional baseline at exactly
  0.500 across every bin setting, which is the correct registration of "inapplicable", not
  a computed result. Per-slice positivity would require downloading the 1,295 human-expert
  lesion delineation *volumes* from `picai_labels` and a NIfTI reader — a real download,
  not one `curl` of a CSV. That was not attempted here.
* **Column discipline.** `prostate_volume` and `psad` were excluded because they are
  measured from the MRI; including either would have broken the zero-image guarantee and
  inflated this row. `case_ISUP`, `lesion_ISUP`, `lesion_GS`, `lesion_PIRADS` and
  `histopath_type` were excluded as outcome-derived. What remains is `patient_age`, `psa`
  (a blood test), `center` and the acquisition year. Best single column: `patient_age` at
  0.639, then `psa` at 0.638.
* **This is the paper's positive example.** PI-CAI evaluates at the patient level, by
  design, and publishes no slice-level number to attack. It should be presented as a
  benchmark doing it right.

### 3.6 LUNA16 (row 10) — the strongest negative, with one honest asterisk

* **Scored on LUNA16's own scale.** Comparing a positional AUROC (0.534 [0.513, 0.558])
  against a published CPM would be exactly the incomparable comparison this audit exists
  to refuse. `pipeline/audit_prep/luna16_cpm.py` therefore scores the same 20-bin
  positional estimator on the competition performance metric: sensitivity at 1/8, 1/4,
  1/2, 1, 2, 4, 8 false positives per scan, out-of-fold on a scan-disjoint 5-fold split.
  Result: **CPM 0.0020**, sensitivity 0.0006 at 1 FP/scan, against a random-score
  reference of 0.0027. The positional baseline is not merely worse than the published
  system, it is *at or below chance on this benchmark*.
* **The asterisk.** The FP-reduction track is conditioned on `candidates_V2.csv`, a
  candidate list produced by image-based detectors. "Zero-image" here means "zero image
  *given the published candidate list*". The label being predicted — is this candidate a
  nodule — is not predictable from where the candidate sits in the scan, and that is the
  finding, but the setup is not pixel-free in the same clean sense as fastMRI Prostate.
* World z was used as the position, rescaled within each scan; with ~850 candidates per
  scan the endpoints are well determined, so no supplied position column was needed.

---

## 4. Slice-level versus patient-level, measured on every benchmark

This table is the paper's real result. Every cell is our own computation on a published
label file, so it depends on no published number and none of the comparability objections
in §3 apply to it. It is reported for all eight dataset-arms, including the two that do
not show the effect.

| dataset | zero-image positional, **slice** AUROC | zero-image positional, **patient** AUROC | position-stratified slice AUROC (the remedy) |
|---|---|---|---|
| fastMRI Prostate T2 | 0.854 [0.812, 0.891] | **0.506** [0.381, 0.632] | **0.546** (5 strata) |
| fastMRI Prostate DWI | 0.851 [0.816, 0.887] | **0.424** [0.298, 0.547] | **0.539** (6 strata) |
| fastMRI+ knee, meniscus tear | 0.873 [0.858, 0.886] | **0.510** [0.428, 0.592] | — |
| fastMRI+ knee, any finding | 0.801 [0.779, 0.824] | **0.558** [0.470, 0.648] | — |
| Duke breast, owner slice task | 0.823 [0.811, 0.834] | undefined (all patients positive) | — |
| DeepLesion, pelvis vs rest | 0.977 [0.969, 0.984] | 0.954 [0.939, 0.967] | — |
| PI-CAI (case level) | not applicable | metadata only, 0.692 [0.626, 0.755] | — |
| LUNA16 candidates | 0.534 [0.514, 0.558] | 0.581 [0.538, 0.613] | — |

Two benchmarks show the collapse outright — fastMRI Prostate (both arms) and fastMRI+
knee (both label definitions). Duke breast is a third variant of the same protocol
problem: 0.823 at the slice level, and no patient-level number is computable at all
because every patient in the cohort is positive. DeepLesion does not collapse, and should
not: its labels are anatomical regions, so they *are* patient-level facts about where
lesions were found, and position predicts them at both units. LUNA16 is at chance at both
units. PI-CAI has no slice-level structure to collapse. Stating all five outcomes is what
makes the first three credible.

The remedy column is the constructive half. Stratifying the slice-level AUROC by
relative position collapses the fastMRI Prostate null from 0.854 to 0.546 and from 0.851
to 0.539 — within noise of chance. That is the metric the paper should ask reviewers to
require.

---

## 5. Prior art discovered during this run — the novelty section needs revising

`paper/audit_targets.md` §3.4 records that targeted searches for a position-only
predictive baseline "returned zero results", with the caveat that this is absence of
evidence. This run found one, and it is on a dataset already in the target list.

**Yan K, Wang X, Lu L, Zhang L, Harrison AP, Bagheri M, Summers RM. "Deep Lesion Graphs
in the Wild", CVPR 2018 (arXiv:1711.10535), Table 1** includes a row labelled
**"Baseline: Location feature"** scoring **59.7%** 8-class lesion-type accuracy, against
their full method's 90.5%. Their location feature is (x, y, z) where z comes from a
self-supervised body-part regressor run on the image — so it is *image-derived* position,
not pixel-free position, and it is used as a retrieval/clustering feature rather than
offered as a critique of the benchmark. But it is a published position-only baseline on a
benchmark in our list, and our 0.557 sits just below it.

**What this changes.** It does not sink the paper, and it should not be buried. It
sharpens what is left as new:

* Not new: that position alone predicts lesion type on DeepLesion, or that a
  location-only feature makes a useful baseline. Yan et al. 2018 published that.
* Still defensible: that the position can be taken from the *published label file* with
  no image and no body-part regressor, and that the resulting number is within four
  points of the image-derived version (0.557 vs 0.597).
* Still defensible: the systematic application across benchmarks with identical
  reporting, and the released tool.
* The novelty claim must be re-audited against the CVPR/MICCAI/MIDL literature before
  submission, not against a handful of web queries. If Yan et al. 2018 exists, others
  likely do.

---

## 6. What was and was not reached

**Audited (7 label files, 6 distinct benchmarks):** fastMRI Prostate T2, fastMRI Prostate
DWI, DeepLesion, fastMRI+ knee, Duke Breast Cancer MRI, PI-CAI, LUNA16.

**Not reached, with reasons:**

| target | why not |
|---|---|
| **RSNA 2019 Intracranial Haemorrhage** (highest impact on the list) | Two independent blockers. (a) `stage_2_train.csv` is keyed by `ID_<SOPInstanceUID>_<subtype>` and carries only the label — no patient id, no study id, no slice position. The join needs DICOM headers from the ~450 GB image release or an unverified third-party metadata CSV, and no provenance-checkable mapping was found. (b) Access is behind a click-through Research Use Agreement; accepting terms on the user's behalf is outside what this session may do. **Both are reportable findings about benchmark release practice**: the benchmark whose official metric is per-slice publishes a label file from which the slice cannot be located. |
| **RSNA 2023 Abdominal Trauma / RSNA 2022 Cervical Spine** | Kaggle-hosted; same click-through-agreement blocker. `image_level_labels.csv` is genuinely per-slice and these remain the best next targets for someone who accepts the agreement. |
| **fastMRI+ brain** | Only 73 of 1,001 roster volumes held locally. Underpowered; not run. |
| **PI-CAI slice-level arm** | Would require downloading 1,295 lesion-delineation volumes and a NIfTI reader (neither `nibabel` nor `SimpleITK` is in the venv). Not attempted; the case-level arm was run instead. |
| **PROSTATEx, CQ500, BraTS/KiTS/MSD/AMOS/TotalSegmentator** | Excluded for the reasons already recorded in `paper/audit_targets.md` Tier 3 (DICOM-header dependence, scan-level-only labels, segmentation metrics). Not revisited. |

**Label files used — provenance, size and licence.** All are tabular; no pixel data was
downloaded for any target.

| file | bytes | sha256 (first 16) | source | licence |
|---|---|---|---|---|
| `t2_slice_level_labels.csv` | 760,340 | `d248d41c9915c3fe` | github.com/cai2r/fastMRI_prostate | MIT (repo), no DUA for the CSVs |
| `dwi_slice_level_labels.csv` | 796,852 | `e22a354132cce884` | same | same |
| `DL_info.csv` | 8,479,888 | `a8f57b4b1164c9ed` | HuggingFace `farrell236/DeepLesion` | CC BY-SA 4.0 (mirror); NIH terms on original |
| `knee.csv` | 918,105 | `c1f4a083646cec81` | github.com/microsoft/fastmri-plus | MIT (repo) |
| `knee_file_list.csv` | 14,074 | `4b09e5523709815d` | same | MIT (repo) |
| `Annotation_Boxes.csv` (`duke_boxes.csv`) | 35,508 | `52752a20f4ec47ea` | TCIA Duke-Breast-Cancer-MRI supporting file | CC BY-NC 4.0 |
| TCIA `getSeries` metadata | 2,894,891 | `fa6b3ee2cc457402` | services.cancerimagingarchive.net NBIA API | CC BY-NC 4.0 (stated per row in the file) |
| `picai_marksheet.csv` | 97,708 | `23eab23790886258` | github.com/DIAGNijmegen/picai_labels | CC BY-NC 4.0 |
| PI-CAI official CV folds ×5 | ~7.5 kB each | — | github.com/DIAGNijmegen/picai_baseline | Apache 2.0 (repo) |
| `candidates_V2.csv` | 71,374,684 | `2e0f79bbee9a3ba7` | Zenodo 3723295 (LUNA16) | CC BY 4.0 (Zenodo record metadata) |
| `annotations.csv` (LUNA16) | 136,986 | `db9adb75b381f3e9` | Zenodo 3723295 | CC BY 4.0 |

Files not obtained in this session were already present from the prior session's
downloads and their hashes match, which is why rows 1–6 reproduce exactly.

---

## 7. One change to the tool, and why it was necessary

`pipeline/s14_trivialbaselines.py` gained a `--relpos-col` option. The harness derives
relative position by rescaling the slice index within its volume, which requires enough
slices per volume for the endpoints to be meaningful. DeepLesion contributes ~2.2
annotated lesions per series, so that rescaling is degenerate there — it put the
positional baseline for lung-vs-rest at 0.658 when the published normalised z gives
0.872. Benchmarks that publish position directly (DeepLesion's
`Normalized_lesion_location`, LUNA16's world z) need the column used verbatim.

The option is recorded in every payload as `relative_position_provenance`, the supplied
column is removed from the metadata pool so it cannot be counted twice, and the self-test
passes before and after. The two fastMRI Prostate anchors were re-run after the change
and reproduce to four decimal places (T2 0.8542 [0.8123, 0.8913]; DWI 0.8514 [0.8162,
0.8873]), confirming the edit is inert on the default path.

---

## 8. Consequences for the paper

1. **Retitle.** "Trivial baselines match published performance on medical imaging
   benchmarks" over-claims on this evidence. What the audit supports is closer to *"How
   much of a slice-level medical imaging benchmark can be reached without the pixels: a
   label-file audit of six public benchmarks"*. The strong version survives only for
   fastMRI Prostate.
2. **Lead with the unit-of-evaluation collapse (§4), not with the match.** It is the only
   result that holds across benchmarks and it needs no published comparator, so it cannot
   be attacked on comparability.
3. **Give LUNA16 and PI-CAI first-class space.** Two of six benchmarks resisted the null.
   A paper that reports that is much harder to dismiss than one that does not.
4. **Rewrite the novelty section around Yan et al. 2018** (§5) and re-run the prior-art
   search properly.
5. **Reverse the T2/DWI recommendation in `audit_targets.json`** (§3.1).
6. **On today's count the label-file-only claim supports five targets** (fastMRI Prostate
   ×2 arms, DeepLesion, Duke breast tabular, PI-CAI marksheet, LUNA16) — fastMRI+ is
   *not* one of them, and RSNA ICH is the case that proves benchmarks can publish a
   slice-level metric and still make the slice unlocatable from the label file.
