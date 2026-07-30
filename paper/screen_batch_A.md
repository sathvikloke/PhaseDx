# Screening results — BATCH A + shared OVERLAP SET

**Screener:** S1 **Submitted (UTC):** 2026-07-29T03:05:00Z
**Protocol:** `paper/screen_protocol.md` v1.0 (FROZEN) · **Codebook:** `paper/screen_frame.json` v1.0 · **Sample:** `paper/screen_sample.json` v1.1
**Machine-readable companion:** `paper/screen_batch_A.json` (one record per paper, every field plus its supporting quote)

**Records coded:** 37 = 15 overlap (permutation positions 1–15) + 22 batch A (positions 16–37). Every position and PMID was checked against `screen_sample.json`; zero mismatches.

## Independence statement

Coded without reading any other screener's output. No file named `screen_batch_B/C/D.*` was opened and no other screener's codes were consulted for any record, including the 15 overlap records. Full texts were downloaded into a shared scratchpad directory that also held files for PMIDs outside this assignment (other screeners appear to be working concurrently in the same directory); every one of those was ignored, and only the 37 records listed below were read.

---

## 1. Headline tallies

### Flow

| | overlap (15) | batch A (22) | total (37) |
|---|---|---|---|
| included | 8 | 12 | **20** |
| excluded | 6 | 9 | **15** |
| unreachable, eligibility unresolved | 1 | 1 | **2** |
| — of the included, full text reachable | 6 | 9 | **15** (complete-case set) |

### Exclusion codes (individually, per §9 of the protocol)

| code | overlap | batch A | total | PMIDs |
|---|---|---|---|---|
| `E-SEG` | 3 | 5 | **8** | 40335658, 36072854, 40239684 / 37293874, 35071275, 34003996, 41274092, 33937792 |
| `E-DERIV` | 3 | 3 | **6** | 36789248, 40194851, 42489954 / 42162744, 39513126, 41574043 |
| `E-2D` | 0 | 1 | **1** | 37908848 |
| `E-PROJ`, `E-NOCLF`, `E-NOMET`, `E-TYPE`, `E-NONMED`, `E-LANG`, `E-DUP` | 0 | 0 | **0** | — |

`E-DERIV` is reported separately per §9: those six papers are inside the query and outside the failure mode.

### Endpoints, this batch only (Wilson score 95%, two-sided)

Denominator for P1 and the secondaries is the **complete-case set: included AND reachable, n = 15**, as specified in `screen_frame.json → endpoints._denominator_default`.

| endpoint | count | proportion | Wilson 95% |
|---|---|---|---|
| **P1** — ≥1 zero-image baseline (constant/positional/acquisition-metadata/permuted-label) with a measured value | **0 / 15** | **0.000** | **[0.000, 0.204]** |
| S1 — ≥1 non-imaging baseline of any kind (P1 family + clinical/demographic-only) | 1 / 15 | 0.067 | [0.012, 0.298] |
| S2 — headline evaluation unit is the slice | 4 / 15 | 0.267 | [0.109, 0.520] |
| S3 — among papers reporting any slice-level metric, proportion also reporting patient-level | 0 / 4 | 0.000 | [0.000, 0.490] |
| S4 — explicitly states a subject-level split | 6 / 15 | 0.400 | [0.198, 0.643] |
| S5 — reports/discusses the positional distribution of labels (figure/table or text with numbers) | 1 / 15 | 0.067 | [0.012, 0.298] |
| S6 — full text unreachable, among eligible-looking papers (included + unresolved, n = 22) | 7 / 22 | 0.318 | [0.164, 0.527] |
| S8 — reports a subject-clustered uncertainty interval | 0 / 15 | 0.000 | [0.000, 0.204] |
| S9 — reports n positive **patients** as well as n positive slices | 0 / 15 | 0.000 | [0.000, 0.204] |

Sub-tallies, for reassembly with the other batches: overlap complete-case P1 = 0/6, batch A complete-case P1 = 0/9. Counting the 5 included-but-paywalled papers as P1-positive (upper bound) would give 5/20 = 0.250; counting them as negative (lower bound) gives 0/20 = 0.000. **S6 is 31.8% here, well above the 15% threshold in §7 rule 4** — if that holds across all four batches the bounding interval, not the complete-case estimate, becomes the headline number.

### S7 — headline unit × P1 flag (complete-case, n = 15)

| headline unit | P1 = true | P1 = false | total |
|---|---|---|---|
| `na_only_one_unit_reported` | 0 | 15 | 15 |
| any other level | 0 | 0 | 0 |

Every single complete-case paper reports its metric at exactly one unit. Not one reports the same metric at two units, so `headline_unit` never had to be adjudicated — and endpoint S3 is 0/4 by construction: of the four papers with a slice-level metric, **none** also reports a patient-level one.

### Other descriptive tallies (complete-case, n = 15)

| field | distribution |
|---|---|
| `evaluation_unit_reported` | patient 4 · slice 4 · lesion 2 · other (sub-volume) 2 · unclear 2 · volume_or_scan_not_patient 1 |
| `split_unit` | patient_subject 6 · slice_or_image 5 · random_unit_not_stated 3 · external_cohort_only 1 |
| `split_disjointness_verified` | stated_only 6 · not_stated 9 · **stated_and_checked 0** |
| `positional_distribution_reported` | no 14 · figure_or_table 1 |
| `uncertainty_interval_reported` | none 8 · ci_unspecified_method 6 · sd_across_folds 1 · **ci_clustered_by_subject 0** |
| `n_positive_reported` | neither 6 · patients_only 5 · slices_only 4 · **patients_and_slices 0** |
| `modality` | MRI 7 · CT 6 · OCT 2 |
| `dataset_public` | private 10 · public 4 · mixed 1 |
| `code_availability` | none 13 · public_link_stated 2 · **public_link_works 0** |

---

## 2. What the primary endpoint actually looks like in this batch

**Zero papers, in either group, report a constant/prevalence, positional, acquisition-metadata or permuted-label baseline with a measured value.** The nearest approaches, and why each is coded false:

- The comparator in every included paper is another imaging model, another architecture, an ablation, or a human reader — all four are explicitly listed under `trivial_baseline.does_not_count`.
- PMID 41068276 reports an ablation row literally labelled "Baseline ResNet152 (no augmentation, no XAI) 92.1" (Table 4). That is an ablation of network components, not a pixel-free comparator.
- PMID 35626379 writes the AUC definition as "the probability by which our classifier preferred a randomly selected PE instance over a negative one". That is a definition, not a measured chance arm; `chance_asserted_without_measurement` is false throughout the batch (0/37).

Three papers do report a **clinical/demographic-variables-only** arm with measured values — S1, never P1:

| PMID | quote |
|---|---|
| 39061744 | "The clinical machine-learning model surpassed the clinical model in both the training (0.975 vs. 0.904, p = 0.019) and testing (0.967 vs. 0.889, p = 0.045) cohorts." (Abstract) |
| 37222638 | "The MRI-based DLR model had a higher area under the curve than the clinical model in three datasets (0.880 vs. 0.741, 0.861 vs. 0.772, 0.852 vs. 0.675, respectively)." (Abstract) |
| 41617832 | "Three single-modality models (CT-based deep learning [CT-DL], … and a clinical logistic regression model [CL]) … outperforming all single- and dual-modality models (p < 0.05)." (Abstract) |

Only 39061744 is in the complete-case set; the other two are paywalled, which is why complete-case S1 is 1/15 while all-included S1 is 3/20. A fourth, PMID 39513126, plots a clinical-features-only ROC (Figure 2c) but is **excluded** (`E-DERIV`), so it contributes to no endpoint — recorded here only so the trail is complete.

One incidental finding worth the paper's Discussion: in **PMID 39061744** the pixel-based 3D-CNN reaches AUC 0.599 on the test cohort while a three-variable clinical model (fever, blood neutrophils, urine leukocytes) reaches 0.889 on the same cohort. The authors report both without remarking on it.

---

## 3. Record-by-record summary

`FT` = full text: **OA** = PMC/publisher open access · **PW** = paywalled, abstract only · **PP** = preprint version used.
`P1` / `S1` = zero-image / any non-imaging baseline flag.

### Overlap set (positions 1–15)

| # | PMID | venue, year | decision | FT | eval unit | split unit | P1 | S1 | positional | headline |
|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 36776294 | Front Oncol 2023 | included | OA | lesion | slice_or_image | ✗ | ✗ | no | accuracy 0.96 |
| 2 | 41617832 | Eur Radiol 2026 | included | PW | patient | patient_subject | ✗ | **✓** | unclear | AUC 0.888 (ext) |
| 3 | 39423605 | Comput Biol Chem 2024 | **unresolved** | PW | unclear | unclear | ✗ | ✗ | unclear | accuracy 0.98 |
| 4 | 42130124 | Orthop Surg 2026 | included | OA | other (vertebra) | patient_subject | ✗ | ✗ | **figure_or_table** | AUC 0.918 |
| 5 | 36789248 | SN Comput Sci 2023 | **excluded `E-DERIV`** | OA | — | — | — | — | — | — |
| 6 | 40335658 | Eur Radiol 2025 | **excluded `E-SEG`** | PW | — | — | — | — | — | — |
| 7 | 40194851 | AJNR 2025 | **excluded `E-DERIV`** | PW | — | — | — | — | — | — |
| 8 | 42489954 | Brain Topogr 2026 | **excluded `E-DERIV`** | PW | — | — | — | — | — | — |
| 9 | 39061744 | Bioengineering 2024 | included | OA | patient | patient_subject | ✗ | **✓** | no | AUC 0.967 |
| 10 | 31093705 | Eur Radiol 2019 | included | OA | lesion | slice_or_image | ✗ | ✗ | no | PPV 0.765 |
| 11 | 36016875 | Front Pediatr 2022 | included | OA | patient | patient_subject | ✗ | ✗ | no | AUC 0.98 |
| 12 | 36072854 | Front Physiol 2022 | **excluded `E-SEG`** | OA | — | — | — | — | — | — |
| 13 | 37222638 | J Magn Reson Imaging 2024 | included | PW | patient | patient_subject | ✗ | **✓** | unclear | AUC 0.852 (ext) |
| 14 | 40239684 | Biomed Phys Eng Express 2025 | **excluded `E-SEG`** | PW | — | — | — | — | — | — |
| 15 | 41068276 | Sci Rep 2025 | included | OA | unclear | random_unit_not_stated | ✗ | ✗ | no | accuracy 0.978 |

### Batch A (positions 16–37)

| # | PMID | venue, year | decision | FT | eval unit | split unit | P1 | S1 | positional | headline |
|---|---|---|---|---|---|---|---|---|---|---|
| 16 | 37293874 | Med Phys 2023 | **excluded `E-SEG`** | PW | — | — | — | — | — | — |
| 17 | 35071275 | Front Med 2022 | **excluded `E-SEG`** | OA | — | — | — | — | — | — |
| 18 | 38083399 | IEEE EMBC 2023 | included | PW | **both** (lesion/artery/patient) | unclear | ✗ | ✗ | unclear | accuracy 0.84 |
| 19 | 34003996 | Transl Vis Sci Technol 2021 | **excluded `E-SEG`** | OA | — | — | — | — | — | — |
| 20 | 37908848 | Front Med 2023 | **excluded `E-2D`** | OA | — | — | — | — | — | — |
| 21 | 35626379 | Diagnostics 2022 | included | OA | **slice** | patient_subject | ✗ | ✗ | no | AUC 0.84 |
| 22 | 41262491 | Front Radiol 2025 | included | OA | **slice** | patient_subject | ✗ | ✗ | no | accuracy 0.915 |
| 23 | 41274092 | J Med Imaging Radiat Sci 2026 | **excluded `E-SEG`** | PW | — | — | — | — | — | — |
| 24 | 40903384 | Radiography 2025 | included | PW | **slice** | slice_or_image | ✗ | ✗ | unclear | accuracy 0.755 |
| 25 | 32452907 | J Chin Med Assoc 2020 | included | OA | **slice** (B-scan) | slice_or_image | ✗ | ✗ | no | accuracy 0.931 |
| 26 | 41999029 | Med Sci Monit 2026 | included | OA | patient | patient_subject | ✗ | ✗ | no | accuracy 0.95 |
| 27 | 38082966 | IEEE EMBC 2023 | included | PW | unclear | random_unit_not_stated | ✗ | ✗ | unclear | accuracy 0.85 |
| 28 | 34136394 | Front Oncol 2021 | included | OA | **slice** (patch) | random_unit_not_stated | ✗ | ✗ | no | AUC 0.98 |
| 29 | 35247336 | Am J Ophthalmol 2022 | included | **PP** | volume_or_scan_not_patient | external_cohort_only | ✗ | ✗ | no | accuracy 0.827 (ext) |
| 30 | 41782094 | BMC Nephrol 2026 | included | OA | patient | random_unit_not_stated | ✗ | ✗ | no | AUC 0.930 |
| 31 | 36244303 | Comput Biol Med 2022 | included | OA | other (3D crop) | slice_or_image | ✗ | ✗ | no | AUC 0.971 |
| 32 | 35684918 | Sensors 2022 | included | OA | unclear | slice_or_image | ✗ | ✗ | no | accuracy 0.991 |
| 33 | 42162744 | Radiother Oncol 2026 | **excluded `E-DERIV`** | PW | — | — | — | — | — | — |
| 34 | 39513126 | J Cancer 2024 | **excluded `E-DERIV`** | OA | — | — | — | — | — | — |
| 35 | 41740680 | J Neurosci Methods 2026 | **unresolved** | PW | unclear | unclear | ✗ | ✗ | unclear | accuracy 0.990 |
| 36 | 33937792 | Radiol Artif Intell 2019 | **excluded `E-SEG`** | PW | — | — | — | — | — | — |
| 37 | 41574043 | Eur Heart J Digit Health 2025 | **excluded `E-DERIV`** | OA | — | — | — | — | — | — |

Every code above is backed by a verbatim quote with its section/table location in `screen_batch_A.json`. Negatives on the primary endpoint carry the `searches_run` string listing all fourteen mandatory terms and what each one hit.

---

## 4. Access: what was tried, and what failed

The §7 ladder was walked in order for all 37 records.

- **Rung 1 (PMC / publisher OA)** succeeded for 21 records, fetched as JATS XML via NCBI E-utilities.
- **Rung 2 (publisher site direct)** was attempted for all 16 non-OA records and for the two PMC records that turned out to hold front matter and abstract only. It succeeded for none: Springer returned abstract-only landing pages (41617832, 40335658, 42489954); Elsevier returned redirect stubs (39423605, 41274092, 40903384, 35247336, 42162744, 41740680); Wiley returned a Cloudflare interstitial (37222638, 37293874); IEEE Xplore returned HTTP 202 with an empty body (38083399, 38082966); IOP returned a Radware bot-manager CAPTCHA (40239684); `www.ajnr.org` and `pubs.rsna.org` returned HTTP 403 (40194851, 33937792). **No CAPTCHA was solved and no unauthorised source was used.**
- **Rung 3 (institutional subscription)** — not held by this screener.
- **Rung 4 (repository / preprint)** succeeded once: **PMID 35247336** was coded from **arXiv:2111.03997v2**, same title and author list. `fulltext_version_used = preprint`; per §7 it must be reported separately and in the sensitivity analysis. arXiv was also checked for the other Elsevier and IEEE records and returned nothing.
- **Rung 5 (interlibrary loan / author request)** cannot complete inside one session (21-day window). Rather than leave 15 records pending indefinitely I coded them `unreachable_paywalled`, which is the conservative choice: it keeps them out of the complete-case P1 denominator and into the bounding analyses.

Two records sit in PMC but hold **front matter and abstract only** under publisher embargo — 40194851 (PMC12633662, released 2026-10-01) and 33937792 (PMC8017422) — so `oa_status = oa_pmc` in `screen_sample.json` overstates reachability for both. That is exactly the automated-hint caveat in §10, and it is worth reporting: the sample's OA flag was wrong in 2 of the 23 records it marked as PMC-available in this batch.

**Consequence for the primary endpoint.** For the 5 included-but-paywalled papers the mandatory 14-term full-text search could not be run. Their P1 sub-flags are written `false` in the JSON, but `searches_run` says plainly that the search was **NOT RUN** and that the codebook does not accept that as evidence for a negative. They must not be counted in the complete-case P1 denominator. The two `unreachable_eligibility_unresolved` records are handled the same way.

---

## 5. Judgement calls and uncertainty (24 of 37 records flagged)

Six records are `screener_confidence = low` and require adjudication regardless of batch: 41617832, 39423605, 42489954, 38083399, 38082966, 41740680 — all six are paywalled and coded from abstracts.

### 5.1 Calls that decide inclusion

| PMID | call | why I coded it this way | what a second screener could reasonably do instead |
|---|---|---|---|
| **34003996** | `E-SEG` | The paper reports a fluid-detection AUC of 0.97/0.95/0.99 per B-scan, which looks like an evaluated class decision. But no classifier is fitted — "This quantity was then thresholded to the presence or absence of each fluid type" — and pilot record **PMID 41897586** ("U-Net segmentation of OCT foci — reports AUC 0.8411") was excluded on exactly this construction. | Include and code the classification arm, on the literal wording "with NO categorical class decision evaluated". |
| **42489954** | `E-DERIV` | "a dual-branch graph attention network to extract complementary global statistical and local topological features from structural MRI" — a graph descriptor, matching pilot exclusion PMID 34924987 (ABIDE connectivity matrices). Abstract only. | Include with `input_representation = unclear`, if the GAT is read as operating on a voxel grid. |
| **37908848** | `E-2D` | Classifier input is an individual en-face in-vivo confocal microscopy frame; confocal microscopy is not in I3's modality enumeration. | `E-PROJ` — the frames come from a 40-plane Z-stack, so a slice axis arguably exists. Either way the paper is out. |
| **42162744**, **41574043** | `E-DERIV`, not `E-SEG` | Both are nnU-Net segmentation papers, but both DO evaluate a categorical decision (low-overlap risk AUC 0.89; PH diagnosis AUC 0.88), so E-SEG's "with NO categorical class decision evaluated" clause fails and E-DERIV becomes the first applicable code — the classifier input is a derived measurement table in both. Coded identically for consistency. | `E-SEG` on the grounds that the deep-learning contribution is segmentation. |
| **41274092** | `E-SEG` | Three codes apply (E-SEG image-quality assessment, E-NOCLF no classifier, E-NONMED phantom-only); the codebook mandates the first in listed order. | `E-NOCLF`, which is more informative for the flow diagram. |
| **33937792** | `E-SEG` | FROC and LROC are detection/localisation metrics; the four nodule types are a stratification, not an evaluated task. Consistent with pilot exclusion PMID 39389801. Abstract only. | Include, if the full text turns out to contain a per-nodule benign/malignant classification arm. |
| **36244303** | included, **not** `E-DERIV` | The GCN consumes graph-structured data, but that graph is built inside the same pipeline from pixel-level CNN features — the image is not discarded. | `E-DERIV`, reading "graph-structured data" as the classifier input. |
| **41740680** | **unresolved** | Abstract lists "dice score, Intersection over Union (IoU), accuracy, precision, sensitivity, specificity, and F1-score" in one sentence; I cannot tell whether the headline 99.03% accuracy is a classification metric (include under A3) or a pixel-wise segmentation accuracy (exclude under A4). | Either; this is precisely what `unreachable_eligibility_unresolved` exists for. |
| **39423605** | **unresolved** | Abstract is consistent with inclusion (MRI input, classifier, numeric accuracy) but "feature extraction" before "detection" leaves E-DERIV live. | Either. |

### 5.2 Enum-mapping conventions I had to adopt (recorded so they can be re-mapped)

The codebook's enums do not cover three configurations that occur repeatedly in this sample. I chose one mapping and applied it consistently rather than improvising per paper; the verbatim wording is preserved in every quote field so an adjudicator can re-map without re-reading.

1. **`split_unit` has no `lesion` level.** Lesion- and ROI-level splits are coded `slice_or_image` as the nearest sub-patient image unit — PMIDs 36776294 ("The 676 lymph nodes were randomly divided into 70% of the training set…") and 31093705 ("A test set of 60 lesions was labeled…"). This is the mapping most likely to disagree with other screeners.
2. **`evaluation_unit_reported` mapping.** 2D object from one cross-section (slice, B-scan, frame, patch-from-slice) → `slice`; explicitly named lesion/node → `lesion`; 3D sub-volume smaller than the scan (a vertebra, a stepped 3D crop) → `other`; whole scan not aggregated to patient → `volume_or_scan_not_patient`.
3. **"cases" for `split_unit`.** Read as patient-level only where the paper's own tables make one case = one patient — PMID 42130124 ("10% of all cases were randomly selected…", with per-case age/sex in Table 2) and PMID 36016875 ("70% of cases for training and 30% of cases for validation", 119 patients / 119 scans). Both flagged. Everything vaguer stayed at `random_unit_not_stated`; in particular **PMID 32452907** ("80% (2768) of images were randomly selected (based on different patients)") was **not** upgraded to `patient_subject` on the strength of a parenthetical, per the codebook's CRITICAL DISTINCTION. That single record is my best guess at where the overlap-set disagreement will land.

### 5.3 One positional call

**PMID 42130124** is the batch's only `positional_distribution_reported` positive. Table 2 tabulates fracture counts by vertebral level (L1 57, L2 53, L3 45, L4 29, L5 16) and the text says "Fracture segments were primarily distributed in the upper lumbar region". Vertebral level is an ordered position along the acquired stack and the counts are given, so I coded `figure_or_table`. A screener reading L1–L5 as a purely anatomical statement — the codebook's "lesions occur in the peripheral zone" case — would code `no`, and S5 for this batch would fall to 0/15.

---

## 6. Observations for the manuscript (not endpoints)

- **PMID 35626379** (Diagnostics 2022) is the most on-topic record in the batch: a **slice-level headline AUC of 0.84 on the RSNA-STR Pulmonary Embolism CT Dataset** — the same benchmark in which the prior-art Kaggle notebook "Baseline with no image" bins P(PE | relative slice location). The paper splits patient-wise, reports no patient-level metric, and reports no zero-image baseline. It is a clean illustration of the gap the checklist is meant to close.
- **PMID 34136394** is the batch's clearest label-broadcast case: a per-patient molecular label (SYP expression, dichotomised at the median) is broadcast to 7,266 tumour patches, the split is "The data were divided into a training set and a test set in a ratio of 8:2" with no unit named, and the headline AUC of 0.98 is reported per patch with no patient-level aggregation.
- **Nothing in this batch reports a subject-clustered interval (0/15) and nothing reports positive counts at both patient and slice level (0/15).** Both are the floor, not a near-miss.
- **`split_disjointness_verified = stated_and_checked` is 0/15.** Six papers state a subject-level split; none reports having verified it.
- **PMID 41999029** has the most careful evaluation design in the batch (explicit patient-level split, explicit patient-level unit of analysis, bootstrap CIs) and still has no zero-image comparator — on a task whose label is a geometric distance measured on the same MRI, where a metadata or positional baseline would be highly informative.
