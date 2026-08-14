# Benchmark Eligibility — the universe of datasets considered, and why each was kept or dropped

Answers external-review FINDING 4 ("benchmark selection looks convenient rather than
systematic: seven datasets are listed with no defined universe").

**Terminology guard.** The submission's hard exclusion bans the token "screen" and its
variants, because that token belongs to a different, excluded analysis. This artefact is
therefore called **Benchmark Eligibility** everywhere, and the manuscript-facing LaTeX
(`tables_appendix.tex`, Table E1) contains none of the banned tokens. The `.md` filename
was fixed by the task request; it is a working file and does not travel with the
submission. Verify before upload with:
`grep -nEi 'screen|PRISMA|protocol-frozen|bounding interval' paper/tex/rsna/*.tex`

**Provenance of this document.** Reconstructed from `paper/audit_targets.json` (the
structured candidate record, `"generated": "2026-07-28"`), `paper/audit_targets.md`
(its companion prose, Tier 1 / Tier 2 / Tier 3 / Unverified), `paper/audit_results.md`
§3.7, §3.8 and §6, the 23 per-benchmark payloads in `pipeline_out/trivial_baselines/`,
and filesystem/payload timestamps. Nothing here is recalled; every row cites the file it
came from.

---

## 1. The entry criterion, verbatim from the record

From `paper/audit_targets.json`, field `entry_criterion`, recorded 2026-07-28:

> A dataset qualifies as a label-file-only target if subject id, slice index, label and
> split can be obtained WITHOUT downloading pixel data and WITHOUT a DUA covering pixels.

and field `nulls.positional.needs`:

> `["subject_id", "slice_index", "label", "train_test_split"]`

Those are the four required fields used as columns below. Two points of discipline the
criterion does **not** settle, and which the manuscript must state, because the audit's
inclusions and exclusions are only consistent under one reading of each:

1. **"Obtainable" does not mean "in the benchmark's own release."** RSNA ICH qualifies
   only because a third party republished the identifiers and the ordering in a pixel-free
   tabular file. Under the stricter reading — fields must come from the benchmark's own
   release — RSNA ICH would be excluded along with RSNA PE, and the flagship would
   disappear. The audit used the looser reading. Say so.
2. **"Without pixel data" was applied to the *download*, not to the *provenance* of every
   field.** fastMRI+ knee needs per-volume slice counts read from image-file headers, and
   LUNA16's candidate list was produced by image-based detectors. Both are already
   disclosed in the manuscript's limitations; they are recorded again here as eligibility
   qualifications rather than as afterthoughts.

---

## 2. What the record can and cannot establish about ordering

**It can establish this.** A candidate universe with a stated entry criterion, per-candidate
field inventories and per-candidate exclusion reasons was written down **before** the run
that produced 7 of the 8 included label files. `paper/audit_targets.json` carries
`"generated": "2026-07-28"` and a filesystem mtime of 2026-07-28 23:37 local
(2026-07-29 04:37 UTC). The payloads for fastMRI Prostate T2/DWI, all eight DeepLesion
body-part arms, fastMRI+ knee ×2, Duke Breast, PI-CAI and LUNA16 carry
`generated_utc` between **2026-07-29 05:09:41** and **05:33:00** — 32 to 56 minutes
after the candidate list. RSNA ICH is later still (07:21:34 subsample, 18:09:02 full).

**It cannot establish this.** Every one of those files entered git in a *single* commit,
`a64d202` "Replace pipeline; correct the published claim", 2026-07-29 02:12:02 −0500.
`git log --diff-filter=A` returns that one commit for `audit_targets.md`,
`audit_targets.json`, `audit_results.md`, `protocol.md` and `checklist.md` alike. **There
is no commit trail that orders anything inside the run.** The ordering above rests on
filesystem mtimes and on timestamps the analysis scripts wrote into their own outputs.
Both are mutable and neither is a registration. The correct description of this audit is
**"eligibility criteria and a candidate universe recorded in the repository before most
arms were scored, with no external timestamp,"** not "pre-specified."

**Three specific departures from pre-specification, named rather than smoothed over.**

| # | What happened | Evidence |
|---|---|---|
| 1 | **fastMRI Prostate carried results before the candidate list was written.** Its entry is `"status": "DONE_ANCHOR"` and the companion prose calls it "the anchor, already run" with the slice/patient AUCs already printed in `audit_targets.md` §1.1 and in the JSON's `anchor_correction` block. | `audit_targets.json` `datasets[0].status`; `anchor_correction.t2.slice_auc = 0.8542` |
| 2 | **DeepLesion carried a feasibility-probe result before the candidate list was written.** Its entry is `"status": "PROBE_RUN"`, and `audit_targets.md` §1.3 records "**56.0% accuracy vs 22.3% majority class**, zero pixels" as already measured. So DeepLesion was not selected blind either. | `audit_targets.json` `datasets[1].status`; `audit_targets.md` lines 118–121 |
| 3 | **RSNA-STR Pulmonary Embolism 2020 is absent from the candidate list entirely and was added mid-run.** It appears in no entry of the 13-entry `datasets` array, in no Tier of `audit_targets.md`, and in no position of `recommended_execution_order`. `audit_results.md` §6 nevertheless calls it "the priority target", and the test that excluded it ran at 2026-07-29 07:29 UTC — *after* the seven public benchmarks were scored at 05:09–05:33 and after RSNA ICH's first result at 07:21. | `grep -i "pulmonary embolism\|RSPECT" paper/audit_targets.md` → no match; mtime of `pipeline/audit_prep/rsna_pe_position_test.py` |

Departure 3 pushes **against** the paper's interest, not for it: PE was added late and then
excluded, so it adds a negative case to the universe rather than a favourable arm. Departures
1 and 2 do run in the paper's favour and must be disclosed as such. `paper/PAPER_PLAN.md`
§8.14 already anticipated the reviewer objection and the answer it proposed is the one
adopted here: *"do not claim it is [systematic]. Claim ... a stated entry criterion and a
published exclusion list."*

---

## 3. The eligibility table

Field key: **S** = subject identifier · **Z** = slice index or z position · **L** = per-slice
label · **P** = train/test assignment. `Y` = present in a pixel-free public file. `Y*` =
obtainable, but not from the benchmark's own release or not without a header read.
`N` = not obtainable without pixels or without a DUA covering pixels.

### 3a. Included — 7 benchmarks, 8 label files, 16 scored arms

| Candidate | S | Z | L | P | What was available, pixel-free | Disposition |
|---|---|---|---|---|---|---|
| **RSNA 2019 Intracranial Hemorrhage** | Y\* | Y\* | Y | Y | Official `stage_2_train.csv` carries **label only**, keyed `ID_<SOPInstanceUID>_<subtype>`. S, Z and series id come from a third-party MIT-licensed pixel-free mirror (`slice_labels.csv`, 61.7 MB; `rescale_values.csv`, 1.29 MB). Official split geometry published in the ATLAS card. | **INCLUDED**, 6 label columns. Flagship. Eligible only under the looser reading of "obtainable" (§1). Ordering validated, not assumed — see §5 and `rsna_mirror_provenance.json` |
| **fastMRI Prostate, T2** | Y | Y | Y | Y | `t2_slice_level_labels.csv` on public GitHub: `fastmri_pt_id, slice, PIRADS, fastmri_rawfile, data_split, folder`. 9,508 slices, 312 patients. In-file `data_split`, patient-disjoint. | **INCLUDED**. The only arm with a matched published slice-level comparator |
| **fastMRI Prostate, DWI** | Y | Y | Y | Y | `dwi_slice_level_labels.csv`, same schema, 9,490 slices | **INCLUDED** |
| **DeepLesion** | Y | Y | Y | Y | `DL_info.csv`, 8.48 MB, 32,735 rows, 4,427 patients. Publishes `Normalized_lesion_location` — the z confound **directly**, no inference needed — plus in-file `Train_Val_Test` | **INCLUDED**, 8 body-part arms + 1 eight-class arm |
| **fastMRI+ knee** | Y | Y\* | Y | N | `knee.csv` (16,167 annotation rows) + `knee_file_list.csv` (1,172 volumes) on public GitHub. **Positives only**; negatives implicit, so relative position needs per-volume slice counts from fastMRI HDF5 **headers** (registration, header read, not a pixel download), and only 199 of 1,173 roster volumes were held. No published classification split | **INCLUDED with a stated qualification**, 2 arms. Split assigned by 5-fold subject CV in the absence of a published one |
| **Duke Breast Cancer MRI** | Y | Y | Y | N | `Annotation_Boxes.csv`, 922 rows, giving the inclusive tumour slice range per patient. Slice counts from the TCIA `getSeries` metadata CSV (tabular). No official split | **INCLUDED**, 1 arm. Slice task is the **data owners' own** published definition. Every patient is positive, so patient-level AUC is undefined and is reported as undefined |
| **PI-CAI, case level** | Y | N | Y | Y | `marksheet.csv` (97.7 kB) + official 5-fold splits from `picai_baseline`. Per-slice positivity needs the 1,295 `.nii.gz` label volumes and a NIfTI reader; neither `nibabel` nor `SimpleITK` is in the venv | **INCLUDED at case level only**, 1 arm. Field Z is missing, which is *why* the locked positional baseline reads exactly 0.500 here |
| **LUNA16** | Y | Y | Y | Y | `candidates_V2.csv` (71.4 MB, ~750k candidates with world z), `annotations.csv`; official 10-fold subset split | **INCLUDED**, 1 arm. Candidate list is detector-produced, so "zero image" means "given the published candidate list" |

### 3b. Excluded on eligibility — recorded 2026-07-28, before that run's results

| Candidate | S | Z | L | P | What was available | Exclusion reason |
|---|---|---|---|---|---|---|
| **CQ500** | Y | N | N | N | `reads.csv`: three radiologist reads **per scan**. 491 scans, 193,317 slices | **L fails.** Labels are scan-level; the positional model as specified has nothing to score. Slice-level masks exist only via a third-party release (Seg-CQ500) |
| **BraTS** | Y | Y\* | Y\* | Y | Voxel masks | **Pixel-download fails, and no comparator exists.** Masks ship in the same archive as the images, and the benchmark metric is Dice/HD95. A positional null against a Dice score is not a comparison |
| **KiTS** | Y | Y\* | Y\* | Y | Voxel masks | same |
| **Medical Segmentation Decathlon** | Y | Y\* | Y\* | Y | Voxel masks | same |
| **AMOS** | Y | Y\* | Y\* | Y | Voxel masks | same |
| **TotalSegmentator** | Y | Y\* | Y\* | Y | Voxel masks | same |
| **PROSTATEx** | Y | N | Y | Y | `ProstateX-Findings-Train.csv` gives finding position in **patient coordinates (mm)** | **Z fails.** Converting mm to a slice index requires DICOM headers. Metric is per-finding AUC |
| **Prostate158** | Y | Y\* | Y\* | N | n ≈ 158, segmentation-oriented | **L fails and power fails.** No slice-level classification benchmark exists to audit |
| **MRNet (Stanford knee)** | Y | N | N | Y | Exam-level labels | **Z and L fail** by construction |

### 3c. Eligible or near-eligible, not reached — an access or power decision, not an eligibility failure

Recording these separately matters: calling them "excluded" would overstate how systematic
the universe is.

| Candidate | S | Z | L | P | Status | Why not reached |
|---|---|---|---|---|---|---|
| **RSNA 2023 Abdominal Trauma (RATIC)** | Y | Y | Y | Y | `"status": "CANDIDATE"`, 5th in `recommended_execution_order` | **All four fields present.** `image_level_labels.csv` carries `patient_id`, `series_id` and an instance number. Blocked only by the Kaggle click-through agreement, which was not accepted. Per-series slice counts would still need the series manifest. *This is the strongest untaken target and should be named as such* |
| **RSNA 2022 Cervical Spine Fracture** | Y | Y | Y | Y | `"status": "CANDIDATE"` | Per-slice labels exist for only **235 of 3,112** studies (`train_bounding_boxes.csv`); `train.csv` is study-level per-vertebra. Underpowered, plus the same Kaggle blocker |
| **fastMRI+ brain** | Y | Y\* | Y | N | Label file downloaded (8,213 rows) | Only **73 of 1,001** roster volumes held locally. Underpowered; not run |
| **PI-CAI, slice level** | Y | Y\* | Y | Y | Case-level arm run instead | Requires 1,295 NIfTI label volumes and a reader absent from the venv. The *labels* are pixel-free (they are label volumes, not images), so this is a tooling gap, not an eligibility failure |

### 3d. Added to the universe after results were seen, then excluded on a measurement

| Candidate | S | Z | L | P | What was available | Exclusion reason |
|---|---|---|---|---|---|---|
| **RSNA-STR Pulmonary Embolism 2020** | Y | **N** | Y | Y | Official `train.csv` **was obtained** without Kaggle credentials, from a public GitHub mirror (119,970,071 bytes, dated 2020-09-07, still on disk at `pipeline_out/audit_data/train.csv`). Genuine: 1,790,594 rows, 7,279 studies, prevalence 0.05392, matching Hu et al. *npj Digit Med* 2025 Table 4 exactly | **Z fails, and it was measured rather than assumed.** Row order gives a run-length ratio of **0.974** and SOPInstanceUID order **1.001** — neither carries any positional information. Independently, requirement (b) fails: the peer-reviewed numbers are exam-level on a private test set whose labels were never released |

### 3e. Named but never verified — must not enter the manuscript as either included or excluded

`CT-ICH` (PhysioNet), `INSTANCE 2022`, `COVIDx CT-3`, `MosMedData`, `SPIDER`, `VerSe`.
All carry `"verification": "UNVERIFIED"` in `audit_targets.json`; no label file was
downloaded and no field inventory exists. `audit_targets.md` line 252 already says they
"must not enter the manuscript without a download." They are listed here so the universe
is complete, and they must be described as *unassessed*, not as excluded.

---

## 4. Counts, for the body sentence

- **25 datasets named** (counting the five segmentation benchmarks individually).
- **19 assessed** against the four fields; **6 unassessed** (§3e).
- **8 label files across 7 benchmarks included**, yielding **16 arms** plus the six label
  columns of RSNA ICH.
- **9 excluded on eligibility**, every one for a *named field failure*: L fails for 3
  (CQ500, Prostate158, MRNet), Z fails for 2 (PROSTATEx, MRNet), the pixel-free-download
  requirement fails for 5 (the segmentation benchmarks), and RSNA PE's Z failure was
  measured.
- **4 eligible-or-near-eligible but not reached**, for access or power, not eligibility.
- The **exclusion reason is a field failure, not a result**, in every case. No candidate
  was dropped after its baseline was computed.

That last sentence is the one that answers FINDING 4, and it is checkable: the only two
candidates whose results predate the candidate list (fastMRI Prostate, DeepLesion) were
both **included**, so the failure mode "score it, then delete it if it does not fire" left
no trace it could have left.

---

## 5. Cross-reference

The RSNA ICH mirror's validation — which is what makes row 1 of §3a admissible at all — is
in `pipeline_out/trivial_baselines/rsna_mirror_provenance.json`, produced by
`pipeline/audit_prep/frozen/rsna_mirror_provenance.py`. The three-way variable taxonomy
answering FINDING 5 is Table E2 in `tables_appendix.tex`.
