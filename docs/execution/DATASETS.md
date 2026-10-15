# Dataset and evidence-source registry

This inventory covers the active October study, datasets used by the older repository and identified historical candidates. **An entry is not an instruction to download it.** Active, reused, historical and excluded roles are explicit. Different subsets/modalities from one release are not independent external datasets.

E02 resolves exact local paths and versions privately. E36 decides the role of older artifacts under H01. Pin source release/repository commit, exact filenames, digest, acquisition date, inclusion/split and prior exposure in the private input registry. Do not invent a missing release version. None of the raw/individual-level data are included in this GitHub handoff.

## Active clinical source families

| ID | Source and authoritative entry point | Existing study use / required input | Units, status and tasks |
|---|---|---|---|
| D01 | [fastMRI Prostate](https://github.com/cai2r/fastMRI_prostate); [dataset paper](https://doi.org/10.1038/s41597-024-03252-w) | T2 k-space/reference/reader-derived labels, retained cache and C2 scores | Local retained 67 subjects/2039 slices; source release larger. PI-RADS-derived suspicion target, not established pathology-confirmed cancer. E02/E30/E33/E34, H06 |
| D02 | Same fastMRI Prostate release | DWI k-space/reference/labels, retained C2 inputs/scores | 45 subjects/1359 slices; 38 recorded identifiers overlap D01. Do not count as independent cohort. Same tasks as D01 |
| D03 | [fastMRI Breast paper](https://doi.org/10.1148/ryai.240345); [source reconstruction code](https://github.com/eddysolo/demo_dce_recon) | Existing breast workbook, cache/input map, magnitude/phase/combined scores, original retained uncertainty arrays | 70 local cases, 140 cached files, 2240 slice records; 54 positive/16 negative case labels broadcast to slices. Source case/person and ascertainment mapping remains a Methods task. E31/H06 |
| D08 | [RSNA ICH source dataset](https://mira.rsna.org/dataset/1); [official challenge](https://www.kaggle.com/competitions/rsna-intracranial-hemorrhage-detection/data); [dataset paper](https://doi.org/10.1148/ryai.2020190211) | Original challenge label release, mirror `slice_labels.csv`/`rescale_values.csv` and source reconciliation records | Older label-only full-cohort audit and newer subset share a source. Mirror filename ordering is not automatically official anatomical truth. E02/E32/E36 |
| D09 | Local imaging subset derived from D08, **not a new source dataset** | Original selected image members, source-to-patient/study mapping, seeds42/123 models, frozen geometry and original bootstrap inputs | Current internal comparison: 283 studies/250 recorded patients, 117 positive/166 negative studies. Later sensitivity analyses remain separate. E16/E32/E34 |
| D15 | [CQ500 release](https://academictorrents.com/details/47e9d8aab761e75fd0a81982fa62bddf3a173831); [primary methods](https://arxiv.org/html/1803.05854); [curator account](https://www.qure.ai/us/blog/challenges-of-development-validation-of-deep-learning-for-radiology) | Planned independent CT images first; authenticated reader-reference file only at the locked outcome stage | No external clinical results established in this audit. Source roster, reference, repeat-exam and clinical-series rules are in the frozen contract. E07–E29, H02–H09 |

**fastMRI access:** use the user's existing authorized local data. The [official access page](https://fastmri.med.nyu.edu/) requires dataset-specific agreements and restricts redistribution, including derived files. Private download links must not enter Git. Source approvals are not the project's ethics determination. Permission to publish images or weights is not inferred from permission to analyze them. These observations were checked October15; record the actual agreement governing each acquired release.

**RSNA access:** preserve the actual Kaggle/MIRA and mirror release terms/identities from the acquired records. This audit did not accept terms, obtain credentials or prove that every mirror object is official. A blank browser rendering of MIRA is not evidence the dataset is unavailable.

## Exact CQ500 acquisition specification

The saved release identity is torrent infohash **`47e9d8aab761e75fd0a81982fa62bddf3a173831`**, multi-file name `qure.headct.study`. The saved source-list image SHA-256 is **`39960c565b3efc786bc3bfa7cdf40057bd89c70f2d15593e2ac37cd577752f07`**; E07 verifies it against the actual local list.

- The image stage contains **491 ZIP files, 28,660,207,451 bytes** in total. Expanded size is not established and must be measured/reserved; do not assume the compressed total is the storage requirement.
- The full release has 493 entries. `reads.csv` is excluded from the image stage and handled separately in E14. `prediction_probabilities.csv` is excluded from the project evaluation entirely.
- Saved webseed composition: `https://s3.ap-south-1.amazonaws.com/qure.headct.study/<exact source filename>`. Use the release list, not a guessed contiguous numeric range. Do not infer endpoint ownership or current availability from URL shape alone.
- The release record names CC-BY-NC-SA4.0; the curator describes public research release. Additional EULA obligations were not established. No new agreement is accepted by this handoff.
- No peer-upload client is needed. Prior HEAD observations do not establish real ranged-transfer/content correctness. E06/E07 provide the narrow implementation completion before the actual permitted operation.
- Previously approved project capacity: 400 GB stored / 200 GB cumulative downloaded in the chosen Research project area. Prior consumption must be established; a nonexistent ledger does not imply zero.

## Historical datasets and supporting resources

These remain documented so no prior study work is lost. **Do not acquire/retrain them by default.** Their current-paper disposition is E36/H01; E47 runs only a specifically missing retained artifact.

| ID | Source | Actual historical use / relevant code | Scope cautions and access status |
|---|---|---|---|
| D04 | [fastMRI brain](https://fastmri.med.nyu.edu/) | Coil/acquisition-property controls; `pipeline/s02_brainknee.py`, `s09_robustness.py`, `s11_archzoo.py` | Not a tumor-diagnosis cohort. Existing raw data/derived caches under their agreement; no new architecture sweep queued |
| D05 | [fastMRI knee](https://fastmri.med.nyu.edu/) | Pulse-sequence/acquisition controls and cache/reconstruction work | Distinct from pathology annotations D06; preserve actual source IDs privately |
| D06 | [fastMRI+ annotations](https://github.com/microsoft/fastmri-plus) | Knee slice annotations plus D05 volume shapes; `pipeline/audit_prep/prep_fastmriplus_knee.py` | Positive annotations alone do not define all negative slices; preserve scan lengths. Annotation repository and raw images have distinct terms/version identities |
| D07 | [Rempe source code/splits](https://github.com/TIO-IKIM/tumor-prediction-on-undersampled-MRI-kSpace); [paper](https://arxiv.org/abs/2407.06165) | Published prostate label/split artifacts; `pipeline/s12_rempe.py`, frozen prostate audit | Uses the same prostate source family; do not call an existing public split external independent clinical validation. Verify paper version before claims |
| D10 | [NIH DeepLesion source](https://nihcc.app.box.com/v/DeepLesion); `DL_info.csv` | Label/metadata-only eight-class benchmark; `prep_deeplesion.py`, `deeplesion_multiclass.py`, `deeplesion_yan_conditions.py` | Location is released lesion location, not independent pixel measurement; no raw image download needed for the historical label-only task. Direct source retrieval unavailable to this audit tool; do not infer missing data or permissions |
| D11 | [LUNA16](https://luna16.grand-challenge.org/) | `candidates_V2.csv`, official folds and original evaluation files; `prep_luna16.py`, `luna16_cpm.py`, `frozen/frozen_luna16_froc.py` | Candidate-level AUROC is not CPM/FROC. Preserve scan grouping and evaluation unit. Official page returned403 in this audit; use existing lawful records, not a bypass |
| D12 | [PI-CAI labels](https://github.com/DIAGNijmegen/picai_labels) and official baseline split repository referenced there | Marksheet/case labels/official fold definitions; `prep_picai.py` | Do not use outcome-derived ISUP/Gleason/PIRADS as pixel-free predictors. Case classification is not slice performance |
| D13 | Same PI-CAI source family | Human-expert delineation volumes for the historical slice audit; `prep_picai_slices.py` | Not an independent cohort from D12. Masks are actual licensed data, not merely a CSV. No new mask download by default |
| D14 | [Duke Breast Cancer MRI, TCIA](https://www.cancerimagingarchive.net/collection/duke-breast-cancer-mri/), DOI `10.7937/TCIA.e3sv-re93` | `Annotation_Boxes.csv` and TCIA series metadata; `prep_duke.py` | The historical task is within-cancer slice localization, not cancer diagnosis. Preserve original positive/negative margin definition and source terms; no 369GB imaging download needed for this label-only task |
| D16 | [PubMed](https://pubmed.ncbi.nlm.nih.gov/) / lawful full-text sources | Frozen PMID frame, permutation, screening codes and access-recovery records in `paper/screen*` | Historical screen excluded from older RSNA main paper; not prospectively registered. E48/E49/H15 conditional only. Copyrighted full texts are not redistributed in Git |
| D17 | Project-generated controlled transport data | B25 fixed generator/configs, 24,000 retained records | Synthetic, no patients. Preserve fixed seed streams and paired design; already completed |
| D18 | Project-generated precision-feasibility data | B25 fixed108-cell design and500 replicates/cell | Synthetic hypothetical precision, not achieved clinical power. Already completed |
| D21 | [openFDA 510(k) metadata](https://open.fda.gov/apis/device/510k/) and FDA public summary documents | `paper/fda_scoping/sample_frame.json`, `paper/FDA_PATH.md`; recorded535-record frame/30-record exploratory sample | Regulatory-scoping history, not a current clinical result or validation dataset. Preserve sample identity; no new FDA audit is required for this paper |

## Candidates and exclusions, not silently imported into scope

- **D19 RSNA-STR pulmonary embolism:** prior-art/label-position investigation; `pipeline/audit_prep/rsna_pe_position_test.py`. Do not turn it into a new benchmark or download imaging without a scientific scope decision.
- **D20 RSNA2023 abdominal trauma:** investigated candidate requiring its actual data-access process; not an executed addition in the inspected current study. No agreement acceptance or download is queued automatically.
- **D22 RSNA2022 cervical spine fracture:** historical candidate in `paper/audit_targets.md`; do not imply it was evaluated.
- **Previously excluded/deferred families:** BraTS, KiTS, Medical Segmentation Decathlon, AMOS, TotalSegmentator, PROSTATEx, Prostate158 and MRNet were considered for the older positional audit and not added to this execution queue. Segmentation metrics cannot be compared directly to slice-classification AUROC. The old CQ500 exclusion concerned that earlier slice-label audit; the current examination-level external study is a separately specified use.
- **Unverified historical candidates:** CT-ICH, INSTANCE2022, COVIDx CT-3, MosMedData, SPIDER and VerSe were search candidates, not established executed inputs. Their existence in an old list is not a reason to acquire them now.
- Other discarded/unverified candidates in `paper/audit_targets.md` remain historical search records. E36 includes their disposition; they do not add mandatory experiments. If an overlooked **actually used** source is found during E02, append its exact identity/role here before execution rather than treating this registry as permission to omit it.

## Private registry fields required per source

`dataset_id`, parent source family, release/version/commit, official URL/citation, actual acquired filename/digest, local authorized path, applicable terms record, unit and target, label/reference source, split/grouping source, observed exposure date, current-study role, inclusion/exclusion rule, output lineage and distribution permission. Unknowns remain explicit. Do not publish private registry rows containing participant-level identities or sensitive paths.
