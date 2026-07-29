# PAPER_PLAN.md

Written 2026-07-29, after reading `paper/audit_results.md`, `paper/audit_targets.md`,
`paper/protocol.md`, `paper/checklist.md`, `pipeline_out/report/RESULTS.md`,
`pipeline_out/robustness/s09_robustness.json`, `pipeline_out/recon_fidelity/`,
`pipeline_out/rempe/` and all 20 cards in `pipeline_out/trivial_baselines/`.

Everything below is written on the assumption that a hostile reviewer will read it, and
that the most dangerous person in the review pool is someone who already knows Yagis 2021,
Badgeley 2019 and Ong Ly 2024. Section 8 is the list of things they will say.

---

## 0. Bottom line, before the argument

**On today's evidence this is not an IF 15+ paper, and it should not be sent to one.**

The audit returned six MATCHED rows and all six come from a single benchmark
(`paper/audit_results.md` §0). Two of six benchmarks refused the null outright. The one
benchmark that is matched is matched against a **preprint** (arXiv:2407.06165, v2 dated
14 Apr 2025; arXiv shows no journal-ref as of 2026-07-29 — verified by fetching the abstract
page). The one place we found a genuinely large positional effect on a widely used
benchmark, DeepLesion, already has a published position-only baseline in its own defining
paper (Yan et al., CVPR 2018, Table 1, "Baseline: Location feature", 59.7%;
`paper/audit_results.md` §5).

What survives all of that is real, useful and publishable: a measured, reproducible,
cross-benchmark demonstration that the *unit of evaluation* is doing work that the reported
number is credited to; a remedy metric that removes exactly that work; a checklist; and a
tool with two dependencies that lets anyone run the audit on a benchmark they will never
hold. That is a strong methods/audit paper for a specialist journal. It is not a Nature
Medicine paper, and pretending otherwise wastes six months.

Recommended venue: **Radiology: Artificial Intelligence** (primary), with a pre-submission
enquiry to **npj Digital Medicine** sent first because the impact is higher and the
topical fit is better — but npj DM also carries the single largest desk-reject risk, for
reasons given in §3.

---

## 1. The one honest sentence

> Across six public medical imaging benchmarks audited from their published label files
> alone, a model that never sees a pixel — fitted only on where a slice sits in its stack,
> or on acquisition and release metadata — reaches 0.80–0.87 slice-level AUROC on three of
> them and falls to chance when the same scores are read at the patient level, matching the
> published slice-level headline of one benchmark (0.854 [0.812, 0.891] against a reported
> 0.861) while failing outright on two others.

If only one sentence can be kept, keep the middle clause. **The slice-to-patient collapse
is the result that generalises.** It is computed entirely from our own runs on published
label files, so it depends on no published comparator and cannot be attacked on
comparability grounds — which is the attack that can be made against every MATCHED row.

Second sentence, for the abstract:

> The effect is not universal: on DeepLesion, whose labels are anatomical regions, the
> positional model is high at *both* units (0.977 slice / 0.954 patient for pelvis), and on
> LUNA16 it is at chance at both (0.534 / 0.581) and scores 0.0020 on the challenge's own
> CPM metric against a published >0.95 sensitivity at <1 false positive per scan.

---

## 2. Claims the evidence does NOT support — write these on the wall

Each of these is a sentence a draft will drift towards. None of them may appear.

| Forbidden claim | Why it fails | What may be said instead |
|---|---|---|
| "Trivial baselines match published performance on medical imaging benchmarks." | Six MATCHED rows, all from one benchmark. Three NOT MATCHED. Three PARTIAL. (`audit_results.md` §0, §2.1) | "A zero-image baseline matched the published slice-level number on one of six benchmarks audited, and reached ≥0.80 at the slice level on three." |
| "Rempe et al.'s model learned nothing." | We could not reproduce their pipeline: their protocol on our prostate DWI cache gives 0.616 against their reported 0.809 (`pipeline_out/s12_waterfall_magphase.log:431`), and 0.574 against 0.813 (`s12_arm_mag.log:131`). A label file cannot support a claim about a model's internals. | "Their *evaluation protocol* certifies a number that a pixel-blind model also reaches." Repeat verbatim in abstract, results, discussion and every figure legend. |
| "We discovered that slice position confounds slice-level evaluation." | Known. Geirhos 2020; Yagis 2021; Wen 2020; Badgeley 2019; Kapoor & Narayanan 2023; and for position specifically Yan et al. CVPR 2018 already published a location-only baseline. | "Known in principle, unquantified in practice under a correct split. We measure how much of a published number it accounts for, and release the means to check." |
| "Published medical imaging benchmarks are broadly invalid." | Two of six resisted. PI-CAI is a benchmark doing it right. | "Benchmarks differ, and the difference is measurable in one command." |
| "Deep learning on medical images is no better than trivial baselines." | Not tested. We never trained a competitive model on any audited benchmark. | Nothing. Delete the sentence. |
| "Phase carries no diagnostic information in MRI." | Our three clinical cohorts are 45–70 subjects, single-institution, single-vendor, 3 T. A null on 67 patients is a null on 67 patients. | "In our cohorts, under a pre-registered nine-criterion protocol, the hypothesis was NOT SUPPORTED on 6–7 of 9 criteria in all three." |
| "The audit covers benchmarks we do not hold." | True for five targets, false for fastMRI+ knee (needs HDF5 headers, `audit_results.md` §3.3) and false for RSNA ICH (`§6`). | "Five of seven label tables were sufficient on their own; two were not, and the failure of one of them is itself a finding about release practice." |
| "The trivial fraction decomposes the published number." | It is a ratio of two margins, not a decomposition. The baseline and the model may exploit the same, different or overlapping shortcuts. (`protocol.md` Rule 7 limits) | "It bounds the part of a reported margin reachable without pixels." |
| "The zero-image baseline's CI shows the published number is not significantly better." | The interval propagates uncertainty in the **baseline only**; the published number enters as a constant. | State the limitation in the same sentence as the fraction, every time. |

---

## 3. Venue recommendation

### 3.1 What kind of paper this actually is

Weigh the three candidate framings the brief names:

* **Methods/tools article.** Fits. There is a tool (`trivialbaselines/`, numpy + pandas,
  MIT, self-tested), a protocol (`paper/protocol.md`, seven rules, each with a measured
  failure behind it), a checklist (`paper/checklist.md`), and a worked example. The
  weakness is that the tool is not hard — it is ~100 lines of statistics plus careful
  bookkeeping — so the contribution has to be the *discipline*, not the code.
* **Benchmark audit.** Fits partially. Six benchmarks is a respectable N for an audit, but
  only one produced a defensible match, and only two produced a defensible published
  comparator at all on the same metric and unit (fastMRI Prostate, DeepLesion). An audit
  whose scored comparisons rest on one preprint and one CVPR table is thin as an *audit*.
* **Perspective with an empirical core.** Fits the material well, and would let the paper
  lead with "here is what a benchmark should publish" and use the audit as evidence. But
  perspectives are usually invited, do not carry a software release well, and would waste
  the strongest asset — the fact that every number here is reproducible from a public CSV.

**Recommendation: write it as a methods/tools article with an audit as its evidence base.**
Title the audit honestly (six benchmarks, seven label files, fifteen rows, six matched) and
put the protocol and the tool in the abstract. That framing is the only one where the two
NOT MATCHED benchmarks are an asset rather than an embarrassment.

### 3.2 Ranked venues

**1. Radiology: Artificial Intelligence (RSNA). Impact factor ~9–10. RECOMMENDED PRIMARY.**

Reasons:
- The audience is exactly the population whose behaviour the checklist is meant to change:
  radiologists and medical-imaging ML researchers who publish slice-level AUROCs.
- RSNA has an institutional appetite for reporting standards (CLAIM, and the RSNA dataset
  papers cited in `audit_targets.md` §1.7, §1.8). A checklist is a *native artefact* there.
- Four of the seven label files audited come from datasets RSNA readers use directly
  (fastMRI Prostate, fastMRI+, DeepLesion, LUNA16/LIDC).
- The closest prior art is **not** in this journal, so the "you already published this"
  desk rejection is much less likely.
- The two NOT MATCHED benchmarks read as rigour to this audience rather than as a weak
  result.

Risk: reviewers will be clinically literate and will ask the anatomy question (see §8.3).
That is answerable, and §7 of `COLLABORATORS.md` asks for exactly that answer.

**2. npj Digital Medicine. Impact factor ~12–15. HIGHER IMPACT, HIGHER RISK. Send a
pre-submission enquiry before writing to their format.**

Reasons for:
- Best topical fit of any journal. It published Badgeley et al. 2019 (`10.1038/s41746-019-0105-1`)
  and Ong Ly et al. 2024, the two closest works to ours, and Varoquaux & Cheplygina 2022.
  The editors already believe the problem matters.
- The "audit without the pixels" property is exactly the kind of scalable-governance point
  that journal likes.

Reason against, stated plainly:
- **Ong Ly et al. 2024 is a direct competitor and it is in this journal.** They report that
  across 13 datasets performance is overestimated by up to 20% on average due to shortcut
  learning of acquisition biases, and they release an estimator (PEst). An editor can
  reasonably say "we ran this in 2024". Our differentiators — zero-image so no pixel access
  is needed, the slice-versus-patient unit collapse which PEst does not address, and the
  positional null specifically — are real but must be argued in the cover letter, not
  discovered by the editor.
- Mitigation: a 200-word pre-submission enquiry that names Ong Ly 2024 in the first
  sentence and states the three differentiators. If the editor is cold, go to Radiology: AI
  immediately and do not spend a submission cycle finding out.

**3. Medical Image Analysis (IF ~11) or MELBA.** Methods audience, tool-friendly, and the
reviewers will scrutinise the statistics harder than anyone else — which will improve the
paper. Lower clinical reach, so the checklist lands with fewer of the people who need it.
Good third choice, and a good *first* choice if the biostatistics review (§`COLLABORATORS.md`
§2) turns up problems that need a specialist referee.

**4. Patterns (Cell Press, IF ~7).** Published Kapoor & Narayanan 2023. Would take the
"leakage + released instrument" framing happily. Lower impact, near-certain acceptance.

**5. Nature Machine Intelligence / Nature Medicine / Lancet Digital Health / Nature
Biomedical Engineering. NOT ON THIS EVIDENCE.** Sending there now costs 3–6 months and
returns a desk rejection whose stated reason will be "the phenomenon is established and the
new evidence rests on a single benchmark". That reason would be correct.

### 3.3 What would make it an IF 15+ paper

Named concretely so this is a plan and not a consolation:

1. **Three or more independent benchmarks MATCHED against peer-reviewed published numbers
   on the same metric and unit.** Today: one, against a preprint. The two best routes are
   RSNA 2019 ICH (whose *official competition metric is per-slice*, so a positional log-loss
   null would be scored on the competition's own scoreboard — `audit_targets.md` §2.2) and
   RSNA 2023 Abdominal Trauma, whose `image_level_labels.csv` is genuinely per-slice. Both
   need a human to accept a click-through Research Use Agreement.
2. **A survey of how many published slice-level papers report the patient-level number.**
   A structured screen of, say, 100 papers reporting slice-level AUROC in 2022–2025, scoring
   them against `paper/checklist.md`. That converts "here is a failure mode" into "here is
   its prevalence", which is the step from a methods note to a field-level result. It costs
   labour, not data access, and it is the single highest-value missing piece.
3. **Rempe et al. peer-reviewed, or replaced by a peer-reviewed anchor.** If arXiv:2407.06165
   is still a preprint at submission, the strongest row in the paper is a critique of a
   preprint and every reviewer will say so.
4. **A clean prior-art re-audit** (§8.1) showing that no MICCAI/MIDL/ML4H paper and no
   Kaggle write-up already published the positional null. If one has, the paper is a
   confirmation-plus-tool, which is a Radiology: AI paper and not more.

If items 1 and 2 land, revisit npj Digital Medicine and Nature Machine Intelligence. Until
then, do not.

---

## 4. Title

Recommended:

> **What a slice-level benchmark certifies without the pixels: a label-file audit of six
> public medical imaging benchmarks**

Runner-up (leads on the remedy, better for Radiology: AI):

> **Slice-level AUROC certifies stack geometry: pixel-blind baselines, a
> position-stratified remedy, and an audit of six public benchmarks**

Reject: *"Trivial baselines match published performance on medical imaging benchmarks."*
It over-claims on this evidence, and `audit_results.md` §8.1 says so already.

---

## 5. Section-by-section outline, with every number and its source

Every numeric claim in the draft must appear in this table with a file path. If it is not
here, it does not go in the paper.

### 5.1 Abstract (~250 words)

| claim | number | source |
|---|---|---|
| benchmarks audited | 6 benchmarks / 7 label files / 15 rows / 12 scored | `paper/audit_results.md` §0, §2 |
| verdict distribution | 6 MATCHED, 3 PARTIAL, 3 NOT MATCHED, 3 NON-COMPARABLE | `audit_results.md` §2.1–2.2 |
| the match | 0.854 [0.812, 0.891] vs published 0.861 | `pipeline_out/trivial_baselines/fastmri_prostate_t2_published.md` |
| the collapse | 0.854 → 0.506; 0.851 → 0.424; 0.873 → 0.510 | same, `..._dwi_published.md`, `fastmriplus_knee_meniscus_tear.md` |
| the exceptions | DeepLesion 0.977 / 0.954; LUNA16 0.534 / 0.581 | `deeplesion_pelvis_vs_rest.md`, `luna16_fp_reduction_candidates.md` |
| the remedy | 0.854 → 0.546; 0.851 → 0.539 | `pipeline_out/rempe/positional_baseline.json`, `..._dwi_labels.json` |
| the tool | numpy + pandas only, MIT | `trivialbaselines/pyproject.toml` |

### 5.2 Introduction (~900 words)

Four paragraphs, in this order. The order matters: prior art comes **second**, not in a
"related work" section at the end, because the reviewer's first question is "what is new".

1. **The setup.** 3D acquisitions labelled per slice; performance pooled over slices; the
   clinical question is about a patient. State the arithmetic problem in two sentences.
2. **What is already known, cited generously and against our own interest.** Shortcut
   learning (Geirhos 2020); acquisition and process metadata (Badgeley 2019 — fracture
   AUC 0.78 from image, 0.91 with hospital process features, 0.52 on a balanced test set,
   scanner model predictable at AUC 1.00; `audit_targets.md` §3.1); slice-level split
   inflation (Yagis 2021 — 30%/29%/48%/55%, and ~96% accuracy on **randomly labelled** data;
   Tampu 2022; Wen 2020, in which CNNs "did not perform better than a SVM with voxel-based
   features"); leakage as a cross-disciplinary failure (Kapoor & Narayanan 2023, 294 papers
   across 17 disciplines); field-level critiques (Varoquaux & Cheplygina 2022; Roberts 2021);
   and **a published position-only baseline** (Yan et al., CVPR 2018, Table 1, "Baseline:
   Location feature", 59.7% against their 90.5%). This paragraph must be longer than a
   reviewer expects. It is the single best protection against a redundancy rejection.
3. **The gap.** Prior work measures the cost of a *wrong* split. We ask a different
   question: under a **correct, patient-disjoint** split, how much of a published
   slice-level number is reachable from the label file alone? Rempe et al.'s split *is*
   patient-disjoint (`audit_results.md` §3.1), so our 0.854 is not a leakage result in the
   Yagis sense. Say this explicitly or a reviewer will conflate the two and reject for
   redundancy (`audit_targets.md` §3.2.1).
4. **What this paper does.** Three deliverables: a family of pixel-blind nulls that need
   only four columns (subject id, slice index, label, split); an audit of six benchmarks
   reporting matches *and* failures; a remedy metric and a one-page checklist. State the
   headline distribution of verdicts in the introduction, not only in the results — a paper
   that hides its negative rows until page 9 reads like advocacy.

### 5.3 Methods (~1,600 words)

**5.3.1 The zero-image family.** Five baselines, all `fit(train)` / `score(test)`:
`prevalence`, `positional_20bin`, `volume_size`, `metadata_tree` (depth-limited CART on
acquisition/administrative columns), `combined_position_metadata`
(`trivialbaselines/README.md`, "The baselines"). Relative position is
`(slice − min_slice_in_volume) / (max − min)`.

**5.3.2 Column discipline.** Outcome-derived columns (the label under another name) and
image-derived columns (they break the zero-image premise) are excluded by default, by a
fallible name heuristic, and every included and excluded column is written to the JSON.
Concrete instance to report in the text: on PI-CAI, `prostate_volume` and `psad` were
excluded because they are measured *from the MRI*, and `case_ISUP`, `lesion_ISUP`,
`lesion_GS`, `lesion_PIRADS`, `histopath_type` were excluded as outcome-derived
(`audit_results.md` §3.5). Naming the exclusions is what makes the guarantee checkable.

**5.3.3 Evaluation.** Both units always, from one score vector. Subject-clustered
percentile bootstrap, 2,000 replicates, seed recorded. Degenerate replicates counted, not
dropped silently. The naive slice-level interval is computed too and reported *as the wrong
one*, so the width difference is visible.

Coverage, measured on simulated data with a closed-form truth
(Φ(μ/√(2σ²ᵤ+2σ²ₑ)) = 0.6880; 200 datasets × 20 patients × 15 slices, 500 bootstrap
replicates each): subject-clustered **91.5%** coverage, mean width 0.370; naive slice-level
**46.5%**, mean width 0.117; ratio **3.18×**.
*Source: `python pipeline/s04_stats.py --self-test`, block [6], re-run and confirmed
2026-07-29.*

**5.3.4 The trivial fraction and the verdict rule.**
`(best zero-image baseline − chance) / (published − chance)`, chance = 0.5 for AUROC and
the majority-class rate for multi-class accuracy. Verdict rule stated **before** any result
(`audit_results.md` §1): MATCHED if the baseline's clustered 95% upper bound reaches the
published number; PARTIAL if fraction ≥ 0.30 with CI wholly below 1; NOT MATCHED if
fraction < 0.30 or the baseline is indistinguishable from chance; NON-COMPARABLE if the
published number is on a different cohort, split, label or metric.

**5.3.5 The remedy: position-stratified AUROC.** Mann–Whitney within bins of relative
position, so only same-position positive/negative pairs contribute
(`trivialbaselines.stratified_auc`). This removes exactly the share of a slice-level AUROC
that stack geometry paid for, and nothing else.

**5.3.6 Each baseline's own permutation null.** Not automatically 0.5. An out-of-fold
metadata model on a subject-level label sits systematically *below* chance because the rate
fitted is anti-correlated with the rate scored; on a synthetic dataset whose label is by
construction invisible to metadata it measures **0.424**. Judged against 0.5 that is a
manufactured below-chance "finding"; judged against its own null it is correctly reported as
no effect. *Source: `trivialbaselines --self-test`; `protocol.md` Rule 4.*

**5.3.7 Benchmark selection.** State the entry criterion (four columns obtainable without
pixel download and without a DUA covering pixels) and report the exclusions with reasons —
CQ500 (scan-level labels only), BraTS/KiTS/MSD/AMOS/TotalSegmentator (segmentation metrics,
masks shipped with images), PROSTATEx (needs DICOM headers), MRNet (exam-level)
(`audit_targets.md` Tier 3). The exclusions are as load-bearing as the inclusions.

**5.3.8 Provenance table.** Every label file with bytes, SHA-256 prefix, source URL and
licence (`audit_results.md` §6). This is Table 1.

**5.3.9 The worked-example cohorts.** The k-space phase study: 3 clinical cohorts
(prostate T2 n=67, prostate DWI n=45, breast n=70), 2 confound cohorts (brain n=454, knee
n=96), 102 training runs (`pipeline_out/results`, 103 JSONs of which one is
`statistics.json`), 456 control runs (`pipeline_out/controls`), 5-fold subject-level CV,
2 seeds. Reconstruction validated against the vendor references shipped in the same HDF5:
brain r = 1.000 (2,270 slices / 454 files), knee r = 1.000 (995 / 199), prostate T2
r = 0.9982 (2,039 / 67), prostate DWI **per-file low-b magnitude-averaged r = 0.9835**
(per cached slice 0.8866, and the reason for the gap is stated), breast r = 0.9772 —
**and the breast reference is not ground truth**: `temptv` is the vendor's temporal-TV
*regularised* reconstruction of the same radial k-space, so that r is agreement between two
estimators. *Source: `pipeline_out/recon_fidelity/recon_fidelity_summary.json` and
`run_streamA.log:160-206`.* Report the breast caveat in the main text, not a footnote.

**5.3.10 Software.** `trivialbaselines` v1.0, MIT, `numpy` + `pandas` only; console script
`trivial-baselines`; `--self-test` with known answers; every run writes a JSON payload and a
markdown card. **Before submission, either publish to PyPI or change every
`pip install trivialbaselines` in `protocol.md` and `checklist.md` to the
`git clone && pip install .` form that `trivialbaselines/README.md` actually uses.** Right
now those two documents disagree, and a reviewer who tries the command in the protocol will
get a 404.

### 5.4 Results (~2,200 words)

**R1. The audit, stated as a distribution before any individual row.**
6 benchmarks, 7 label files, 15 rows; 12 carry a defensible published comparator: 6 MATCHED,
3 PARTIAL, 3 NOT MATCHED; 3 NON-COMPARABLE and unscored. All 6 MATCHED rows come from one
benchmark. *Source: `audit_results.md` §0, §2.1.* → **Table 2**, **Figure 4**.

**R2. fastMRI Prostate — the matched benchmark.**

| quantity | value | source |
|---|---|---|
| published headline (image + k-space) | 0.861 ± 1.8 slice AUROC | Rempe et al. Table II, transcribed at `pipeline/s12_rempe.py::REPORTED` |
| published PCA ×2 magnitude + phase | 0.809 ± 2.1 | same |
| published PCA ×2 magnitude | 0.813 ± 2.2 | same |
| published R=16 PCA | 0.714 ± 2.9 | same |
| **T2 positional 20-bin, slice** | **0.854 [0.812, 0.891]** | `fastmri_prostate_t2_published.md` |
| T2 positional 20-bin, patient | 0.506 [0.381, 0.632] | same |
| **DWI positional 20-bin, slice** | **0.851 [0.816, 0.887]** | `fastmri_prostate_dwi_published.md` |
| DWI positional 20-bin, patient | 0.424 [0.298, 0.547] | same |
| trivial fraction vs 0.861 | T2 0.981 [0.865, 1.084]; DWI 0.973 [0.876, 1.073] | both cards |
| trivial fraction vs 0.809 | T2 1.146 [1.011, 1.266]; DWI 1.137 [1.023, 1.253] | `audit_results.md` §2.1 |
| trivial fraction vs 0.714 | T2 1.655 [1.459, 1.829]; DWI 1.642 [1.478, 1.810] | same |
| test arm | T2 1,399 slices / 46 subjects / 68 positive slices / 20 positive subjects | card |
| training arm | 6,647 slices / 218 subjects (T2); 6,637 / 218 (DWI) | `positional_baseline*.json` |
| bin sweep, T2 | 5=0.835, 10=0.848, 20=0.854, 30=0.854, 50=0.856 | `positional_baseline.json` |
| **no-fit centrality**, T2 / DWI | 0.825 / 0.841 | same JSONs |

The no-fit centrality number deserves its own sentence: **−|relative position − 0.5|, which
uses no training data at all, reaches 0.841 on the DWI file against a published 0.861.**
That is the cheapest possible statement of the result and it is immune to any accusation of
fitting.

**Correction that must be made before submission.** `paper/audit_targets.json`'s
`anchor_correction` block asserts DWI is the correct arm. The evidence points the other
way: Rempe et al.'s abstract says "312 subject and a total of 9508 slices", and 9,508 is the
exact row count of `t2_slice_level_labels.csv` (DWI has 9,490). **T2 is the correct arm**;
the persisted `pipeline_out/rempe/positional_baseline.json` is already the T2 run and is
right; the docstring waterfall at `pipeline/s12_rempe.py:272-278` quotes the wrong arm.
Both arms are reported so the conclusion does not depend on it. *Source:
`audit_results.md` §3.1.*

**R3. The unit-of-evaluation collapse — the paper's real result.** → **Table 3**, **Figure 2**.
Every cell is our own computation on a published label file; no published number enters.

| dataset-arm | slice | patient |
|---|---|---|
| fastMRI Prostate T2 | 0.854 [0.812, 0.891] | 0.506 [0.381, 0.632] |
| fastMRI Prostate DWI | 0.851 [0.816, 0.887] | 0.424 [0.298, 0.547] |
| fastMRI+ knee, meniscus tear | 0.873 [0.858, 0.886] | 0.510 [0.428, 0.592] |
| fastMRI+ knee, any finding | 0.801 [0.779, 0.824] | 0.558 [0.470, 0.648] |
| Duke breast, owner slice task | 0.823 [0.811, 0.834] | **undefined** (922/922 patients positive) |
| DeepLesion, pelvis vs rest | 0.977 [0.969, 0.984] | 0.954 [0.939, 0.967] |
| LUNA16 candidates | 0.534 [0.513, 0.558] | 0.581 [0.538, 0.613] |
| PI-CAI, case level | not applicable (no slice index in the marksheet) | metadata only, 0.692 [0.626, 0.755] |

*Sources: the seven cards named in §5.1 plus `duke_breast_owner_slice_task.md`,
`fastmriplus_knee_any_finding.md`, `picai_case_level.md`; all reproduced in
`audit_results.md` §4.*

Report all eight rows. The two that do not collapse are what make the six that do credible.

**R4. Two benchmarks that resist the null — equal prominence, not a footnote.**

*LUNA16.* Scored on the competition's own metric rather than on a convenient one:
`pipeline/audit_prep/luna16_cpm.py` runs the same 20-bin positional estimator through the
challenge performance metric — sensitivity at 1/8, 1/4, 1/2, 1, 2, 4, 8 FP/scan,
out-of-fold on a scan-disjoint 5-fold split. Result **CPM 0.0020**, sensitivity **0.0006**
at 1 FP/scan, against a random-score reference of 0.0027 and a published >0.95 sensitivity
at <1 FP/scan (Setio et al. 2017, arXiv:1612.08012). 754,975 candidate rows, 888 scans,
1,557 positives. The asterisk must travel with it: the FP-reduction track is conditioned on
`candidates_V2.csv`, produced by image-based detectors, so "zero-image" here means "zero
image *given the published candidate list*" (`audit_results.md` §3.6). Also report the
harness's own warning: the constant predictor scored 0.483, not 0.500, because pooling
out-of-fold across folds with different training prevalence makes fold identity rankable
(`luna16_fp_reduction_candidates.md`, Warnings).

*PI-CAI.* Published 0.91 (0.87–0.94) case-level AUROC for the AI system and 0.86
(0.83–0.89) for 62 radiologists reading PI-RADS 2.1 (Saha et al., *Lancet Oncol*
2024;25:879-887). Our best zero-image baseline, at the case level its authors report, is
**0.692 [0.626, 0.755]** (metadata CART); trivial fractions 0.467 [0.307, 0.622] and
0.532 [0.350, 0.708] — NOT MATCHED. The **positional** baseline is exactly **0.500** at
every bin setting, which is the correct registration of "inapplicable" (the marksheet has
one row per case and no slice index), not a computed result. Best single columns:
`patient_age` 0.639, `psa` 0.638. Cohort caveat, stated because it cuts against us and not
for us: the published numbers are on the hidden 1,000-case testing cohort; ours is on the
public 1,500-case Training and Development set with the benchmark's own official 5-fold
splits. A strict reading makes these rows non-comparable; they are scored anyway because our
baseline had the easier cohort and still lost. *Source: `audit_results.md` §3.5;
`picai_case_level.md`.*

**PI-CAI is the paper's positive example and should be labelled as such in the abstract:
a benchmark that evaluates at the unit it should, and has no slice-level number to attack.
Its metadata baseline is still 0.692, which is the point that fixing the unit does not fix
acquisition confounding — Rules 2 and 6 are independent.**

**R5. DeepLesion — where the positional null is the task, not a confound.**
Yan et al.'s conditions were reconstructed rather than assumed. Their Table 1 test set has
4,927 samples, which is *exactly* the row count of the official `Train_Val_Test == 3`
split — a coincidence that invites a false match. Their own text says otherwise: a random
patient-disjoint 25/25/50 partition of the type-labelled rows, fitting on the 25% seed set.
`pipeline/audit_prep/deeplesion_yan_conditions.py` rebuilds that partition over 200 draws
(mean seed 2,454 rows, mean test 4,900 against their 4,927). Zero-image accuracy **0.5571**,
sd 0.0131, [0.5243, 0.5778] over partitions, against a majority class of **0.2361**.
Trivial fractions: 0.480 [0.431, 0.511] against their 0.905 ± 0.002, 0.513 [0.460, 0.546]
against the 0.862 ImageNet-feature baseline, and **0.889 [0.799, 0.947] against their own
"Location feature" baseline of 0.597** — all PARTIAL.

Read conservatively, and say so in the text: the eight classes are *bone, abdomen,
mediastinum, liver, lung, kidney, soft tissue, pelvis* — anatomical regions. A z-coordinate
predicting an anatomical region is the task, not a confound. **This row is the reference
level a lesion-type classifier must clear. It is not evidence that DeepLesion papers are
unsound and the paper must not use it that way.**

One metadata finding worth its own sentence: on the official split, one-vs-rest for lung
lesions reaches slice AUROC **0.911** from the `DICOM_windows` header column alone
(`-1500, 500` for lung-reconstructed series, `-175, 275` otherwise); position alone gives
0.872; together 0.962 [0.949, 0.973]. Per-class zero-image ceilings on the official split:
pelvis 0.982, lung 0.962, mediastinum 0.957, kidney 0.896, abdomen 0.886, liver 0.876, bone
0.832, soft tissue 0.831. No published per-class AUROC was located, so these are reference
levels only. *Sources: `audit_results.md` §3.2; `deeplesion_*_vs_rest.md`.*

**R6. Metadata baselines beating a trained network — from the worked example, and labelled
as such.**

| cohort | field | field → label | trained phase network → label |
|---|---|---|---|
| breast (n=70) | `folder` — the release batch / download tarball, 7 levels | **0.743** | **0.633** (seed 42), 0.630 (seed 123) |
| prostate T2 (n=67) | `kspace_shape` — acquisition matrix, 14 levels | **0.609** | 0.483 (seed 42), 0.462 (seed 123) |

Both columns are subject-level AUROCs on the same subjects. In the breast cohort the field
also explains more of the *model's score variance* than the true label does: η² = 0.108 for
`folder` against 0.033 for the label (seed 42); 0.151 against 0.046 (seed 123). Which
tarball a scan arrived in has no causal relationship to whether the patient has cancer.
*Source: `python pipeline/s08_belowchance.py --cohort breast --condition phase`, re-run and
confirmed 2026-07-29; `pipeline_out/s08_belowchance.log` for prostate T2.*

**R7. The remedy.** Position-stratified AUROC on the same score vectors:

| | raw slice | position-stratified |
|---|---|---|
| zero-image positional, T2 label file | 0.854 | **0.546** (5 strata) |
| zero-image positional, DWI label file | 0.851 | **0.539** (6 strata) |
| our reimplementation, magnitude arm | 0.574 | **0.467** |
| our reimplementation, magnitude + phase arm | 0.616 | **0.562** |

*Sources: `pipeline_out/rempe/positional_baseline.json` and `..._dwi_labels.json`
(`slice_auc_position_stratified`); `pipeline_out/s12_arm_mag.log:134` and
`pipeline_out/s12_waterfall_magphase.log:434` (rung W4s).*
→ **Figure 5**.

**R8. What the label files could not support — reportable failures of release practice.**
RSNA 2019 ICH is the highest-impact target on the list and its **official competition metric
is per-image**. Its `stage_2_train.csv` is keyed by `ID_<SOPInstanceUID>_<subtype>` and
carries only the label: no patient id, no study id, no slice position. The join needs DICOM
headers from the ~450 GB image release or an unprovenanced third-party CSV. **A benchmark
whose official metric is per-slice publishes a label file from which the slice cannot be
located.** That sentence is the paper's argument for the three-field recommendation, and it
costs nothing to make. Access is additionally behind a click-through Research Use Agreement
that this analysis did not accept. *Source: `audit_results.md` §6.*
fastMRI+ is the other honest failure: it publishes positive annotations only, so slice
counts come from the fastMRI HDF5 headers. **Do not call fastMRI+ a label-file-only target.**
And the coverage is a 199-of-1,173-volume subset (17%), which is not the subset any
published number would use (`audit_results.md` §3.3).

**R9. Worked example: a study audited under its own protocol.** Kept short — 400 words in
the main text, the rest in a supplement. The point is that the protocol was applied to our
own work first and returned a null, not that MRI phase is or is not informative.

- Pre-registered primary cohort `prostate_t2` (n=67), chosen on size and on reconstruction
  validated at r = 0.998, both fixed before any verdict. **NOT SUPPORTED**, failing 7 of 9
  criteria (C1, C2, C3, C4, C5, C6, C8). `prostate_dwi` (n=45) NOT SUPPORTED on 7
  (C1, C2, C4, C5, C6, C7, C8). `breast` (n=70) NOT SUPPORTED on 6
  (C1, C2, C4, C5, C6, C7). *Source: `pipeline_out/report/RESULTS.md` §2.*
- Phase, patient-level, pooled out-of-fold: prostate T2 **0.380 [0.244, 0.530]**;
  prostate DWI 0.442 [0.254, 0.642]; breast 0.631 [0.457, 0.800]. Magnitude, prostate T2:
  0.473 [0.321, 0.632]. *`RESULTS.md` §7.*
- The permutation null on the primary cohort does not contain chance: observed range
  [0.548, 0.645] over 20 distinct replicates, so C3 fails and nothing downstream of it can
  be believed. *`RESULTS.md` §8.* Report this. It is the most damaging single fact about our
  own pipeline and burying it would be exactly the behaviour the paper criticises.
- Training on **air alone** (anatomy removed) does not collapse: prostate T2 0.604 against a
  headline of 0.629 (0.025 below, and the control's own CI [0.528, 0.673] lies entirely
  above chance); prostate DWI 0.595 against 0.586 — the control is **above** the headline;
  breast 0.549 against 0.587 (0.037 below). *`RESULTS.md` §8.*
  **Do not write "within 0.018".** 0.018 is the mean of the three gaps and no cohort
  attains it; quote the three numbers.
- Phase predicts receive-coil count at **0.921 [0.870, 0.966]** on 136 independent test
  subjects, against 0.913 [0.872, 0.950] for magnitude; the difference is +0.007
  [−0.038, +0.051] and includes zero, so **no ordering between the channels is claimed —
  the level is the point**. *`RESULTS.md` §4b.*
- Stratifying within site, phase still predicts the coil bucket at **0.979 [0.953, 0.996]**
  (seed 42) / 0.974 [0.940, 0.997] (seed 123) against an unstratified 0.926 / 0.923. But it
  is **not** separable from scanner model, device id or coil array: no scanner model carries
  enough subjects in both coil buckets. So the paper may say "phase encodes hardware, not
  merely site", and may **not** decompose that hardware into coil count as distinct from the
  scanner it is attached to. *`pipeline_out/robustness/s09_robustness.json`,
  `coil_vs_site.verdict`.*
- Aggregation sensitivity: 7 distinct schemes (mean, max, top-1/2/3/5 mean, q75, q90), and
  no scheme lifts any clinical cohort's cross-seed CI lower bound above 0.500. The
  selection-aware envelope — the 2.5th percentile of the best-of-7 AUC within each subject
  resample — tops out at **0.476**. Sixteen confound-cohort results *do* clear chance under
  the same schemes, so the sweep can detect an effect and the clinical null is informative
  rather than underpowered machinery. *`s09_robustness.json`,
  `aggregation_sensitivity.verdict`.*

### 5.5 Discussion (~1,200 words)

1. **What a high trivial fraction licenses, in one paragraph, stated three times in the
   paper.** It is a statement about an evaluation protocol. It is not a decomposition and
   not a claim about a model's internals.
2. **Why the collapse, not the match, is the general finding.** The match rests on one
   preprint's Table II; the collapse rests on eight of our own computations on public label
   files. Concede the first and keep the second.
3. **Where the null legitimately fires and where it legitimately does not.** DeepLesion's
   labels are anatomical regions; Duke breast's slice task is positional *by construction*
   (the owners' rule: inside the tumour box is positive, ≥5 slices away is negative,
   everything between discarded), so its 0.823 quantifies a tautology rather than indicting
   anyone (`audit_results.md` §3.4). Saying this loudly is what makes the fastMRI Prostate
   row believable.
4. **For benchmark publishers.** Three fields in the label file you already release —
   subject id, slice index or z position, official split assignment — make every rule
   auditable by people who will never be granted the pixels. PI-CAI already reports at case
   and lesion level by design. RSNA ICH is the counter-example.
5. **What the protocol does not do.** It cannot detect shortcuts that live in the pixels —
   scanner texture, burned-in annotation, body-part framing. Those need the images. It
   bounds only the part reachable *without* them, which is the part a third party can check
   for free.
6. **Relation to prior work, again, at the end.** Badgeley's control is a negative control
   (balance the confounder, watch the model collapse to 0.52); ours is a positive control
   (fit the confounder alone, see what it reaches). Same phenomenon, opposite direction, and
   the positive-control form is the one an auditor can run without the images. That is a
   difference in *form*, not in *discovery* (`audit_targets.md` §3.2.2).

### 5.6 Limitations — see §7. This is a full section, not a paragraph.

### 5.7 Data and code availability

- Tool: `trivialbaselines/`, MIT, `numpy` + `pandas`. Resolve the PyPI-vs-git install
  discrepancy first.
- Every audit payload: `pipeline_out/trivial_baselines/*.json` + `*.md`, one card per run.
- Label-table preparation: `pipeline/audit_prep/` (8 scripts).
- Label file provenance with SHA-256: Table 1.
- The worked example's full report: `pipeline_out/report/RESULTS.md` and `verdict.json`.
- **`manuscript/*.docx` must not be released.** `manuscript/DO_NOT_SUBMIT.md` records that
  those files report numbers no code in the repository has ever produced. Delete them from
  any public archive or keep them only in a clearly quarantined directory.

---

## 6. Figures and tables

### Figures

**Figure 1 — What the audit needs.** Schematic. Left: the four columns (subject id, slice
index, label, split) as they appear in a real published CSV. Middle: the five baselines.
Right: the card the tool emits. One annotation: "no pixels, no DUA, no GPU; numpy + pandas".
*Source: `trivialbaselines/README.md`; schema in `audit_targets.md` §1.1.*
New artwork required.

**Figure 2 — The unit-of-evaluation collapse.** Slope chart: eight dataset-arms, slice-level
AUROC on the left axis joined to patient-level AUROC on the right, subject-clustered CIs on
both. Colour: three collapsing (fastMRI Prostate ×2, fastMRI+ knee ×2 — five lines), one
undefined at the patient level (Duke breast, drawn as a line running off the panel with the
reason printed), two not collapsing (DeepLesion pelvis, LUNA16). **The exceptions must be in
the same panel, same colour scale, not in a supplement.**
*Source: `audit_results.md` §4; the eight cards in `pipeline_out/trivial_baselines/`.*
New artwork required.

**Figure 3 — The fastMRI Prostate waterfall.** Eight rungs on one axis:
W0h their headline 0.861 [0.843, 0.879]; W0 their PCA arm (0.813 magnitude / 0.809
magnitude+phase); W1 zero-image positional on their labels 0.851 [0.821, 0.880]; W1p the
same scores at patient level 0.424 [0.298, 0.547]; W2 our reimplementation at their
evaluation level 0.574 / 0.616; W3 the same predictions with a subject-clustered interval
0.574 [0.489, 0.667] / 0.616 [0.528, 0.691]; W4 patient level 0.524 [0.348, 0.690] / 0.528
[0.356, 0.696]; W4s position held fixed 0.467 / 0.562.
Annotation, mandatory: *"W2 does not reproduce W0. This figure is about the evaluation
protocol, not about their model."*
*Source: `pipeline_out/s12_arm_mag.log:127-134`;
`pipeline_out/s12_waterfall_magphase.log:427-435`.*
New artwork required; data already persisted.

**Figure 4 — Trivial fraction across the 12 scored rows.** Forest plot, one row per
(dataset, published number), ordered by fraction, with a vertical line at 1.0 and verdict
labels. Rows 1–6 fastMRI Prostate, 7–9 DeepLesion, 10 LUNA16, 11–12 PI-CAI. LUNA16's row is
its CPM comparison, drawn on its own scale with the scale change made obvious.
*Source: `audit_results.md` §2.1.* New artwork required.

**Figure 5 — The remedy.** Paired bars: raw slice AUROC vs position-stratified AUROC for
four score vectors (zero-image T2 0.854→0.546; zero-image DWI 0.851→0.539; our magnitude arm
0.574→0.467; our magnitude+phase arm 0.616→0.562), plus a small inset showing the bin sweep
(5/10/20/50) and the no-fit centrality value so binning cannot be blamed.
*Source: `pipeline_out/rempe/positional_baseline*.json`; `s12_*` logs, rung W4s.*
New artwork required.

**Figure 6 (worked example, main text or supplement) — the confound cohorts.** Phase vs
magnitude at predicting an acquisition property, with the interpretation inverted and the
inversion printed on the panel: brain, receive-coil count, phase 0.921 [0.870, 0.966] vs
magnitude 0.913 [0.872, 0.950], 136 test subjects; knee, pulse sequence, 0.999 vs 1.000, 29
paired subjects. **Never on the same axis as a diagnostic AUROC.**
*Existing artwork: `pipeline_out/report/figures/fig6_confound_predictability.png`.*

**Supplementary Figure S1 — Bootstrap coverage.** Clustered 91.5% (width 0.370) vs naive
46.5% (width 0.117), 3.18×, on simulated data with true AUC 0.6880.
*Source: `pipeline/s04_stats.py --self-test` block [6].* New artwork required.

**Supplementary Figure S2 — Reconstruction fidelity, worked example.**
*Existing artwork: `pipeline_out/recon_fidelity/fig_recon_fidelity.png` / `.pdf`.*

**Supplementary Figure S3 — Positional label-rate histograms.** Label rate against relative
slice position, training set only, one panel per audited arm. This is Rule 5's own
deliverable and the paper should ship it for its own data.
*Source: the `positional` blocks of `pipeline_out/trivial_baselines/*.json`.*

### Tables

**Table 1 — The seven label files.** File, bytes, SHA-256 prefix, source URL, licence, and
a column "sufficient on its own?" with the honest values (fastMRI Prostate yes ×2,
DeepLesion yes, Duke breast yes-with-TCIA-series-metadata, PI-CAI yes, LUNA16 yes,
fastMRI+ **no — needs HDF5 headers**).
*Source: `audit_results.md` §6.*

**Table 2 — The 15 rows and their verdicts.** Dataset, published number, source of the
published number, our best zero-image baseline, trivial fraction with CI, verdict; the three
NON-COMPARABLE rows in a separate block with the reason.
*Source: `audit_results.md` §2.1, §2.2.*

**Table 3 — Slice vs patient vs position-stratified, all eight dataset-arms.**
*Source: `audit_results.md` §4.*

**Table 4 — The protocol, seven rules, each with the measured failure behind it and the
checklist items it maps to.**
*Source: `paper/protocol.md`, `paper/checklist.md`.*

**Supplementary Table S1 — Per-column metadata results** for every audited arm (best single
columns with the multiplicity warning attached), plus the exclusion lists.
*Source: the "Single metadata columns" block of every card; `audit_results.md` §3.5.*

**Supplementary Table S2 — The worked example's nine criteria** for all three clinical
cohorts, verbatim from `pipeline_out/report/RESULTS.md` §2, including the two PASSes
(breast C3/C8, prostate T2 C7) so it cannot be read as a table built to fail.

**Supplementary Table S3 — DeepLesion per-class zero-image ceilings**, official split, with
the "reference level, not a debunking" caveat in the caption.
*Source: `deeplesion_*_vs_rest.md`.*

---

## 7. Limitations — the full section, written to be pasted

Write it in the first person plural, in the main text, before the conclusion, and do not
soften any of it.

**We could not reproduce the pipeline of the paper we audit most closely.** Our
implementation of Rempe et al.'s protocol on our own prostate DWI cache reaches 0.616
slice-level AUROC against their reported 0.809 for the magnitude+phase arm, and 0.574
against their reported 0.813 for the magnitude arm
(`pipeline_out/s12_waterfall_magphase.log:431`, `pipeline_out/s12_arm_mag.log:131`). We
therefore have no standing to make any claim about their model. Everything we say about
their benchmark is a claim about the *evaluation protocol*: that a model with no access to
the pixels reaches 0.854 [0.812, 0.891] on their own published label file and split, against
their reported 0.861. Whether their network learned tumour signal is a question a label file
cannot answer and we do not attempt it.

**The single MATCHED benchmark's published comparator is a preprint.** arXiv:2407.06165,
v2 dated 14 Apr 2025, carries no journal reference as of 2026-07-29. If it remains
unpublished at submission, the strongest row in this paper is a comparison against
non-peer-reviewed numbers and should be described that way in the results, not only in the
limitations.

**The phenomenon is known; the measurement is what is new, and even that has closer prior
art than we first recorded.** Shortcut learning (Geirhos 2020), acquisition and process
confounding (Badgeley 2019; DeGrave 2021; Ong Ly 2024), slice-level split inflation (Yagis
2021; Tampu 2022; Wen 2020) and leakage as a reproducibility failure (Kapoor & Narayanan
2023) are all established. During this audit we found a published position-only baseline on
one of our own targets: Yan et al., CVPR 2018, Table 1, "Baseline: Location feature", 59.7%
8-class accuracy on DeepLesion against their own 90.5%. Their location feature is
image-derived — z comes from a self-supervised body-part regressor — and is used as a
retrieval feature rather than as a critique, and our pixel-free version lands at 0.557; but
the idea of a location-only baseline on this benchmark is theirs and not ours. What remains
ours is that the position can be taken from the *published label file* with no image and no
regressor, the uniform application across benchmarks with identical reporting, and the
released tool. **Before submission the prior-art search must be redone properly against
Google Scholar, the MICCAI/MIDL/ML4H proceedings and the RSNA ICH Kaggle write-ups, not
against the handful of web queries recorded in `audit_targets.md` §3.4.**

**Only one benchmark was matched, and all six MATCHED rows are that benchmark.** Twelve of
fifteen rows carry a defensible published comparator; six matched, three were partial, three
were not matched. The general statement "trivial baselines match published performance"
is not supported by these data, and we do not make it.

**Two audited benchmarks are not pixel-free in the same clean sense.** fastMRI+ needs slice
counts from the fastMRI HDF5 headers because it publishes positive annotations only, and our
coverage is 199 of 1,173 roster volumes (17%), which is not the subset any published number
would use. LUNA16's FP-reduction track is conditioned on `candidates_V2.csv`, a candidate
list produced by image-based detectors, so "zero-image" there means "zero image given the
published candidate list".

**Two audited rows are positional by construction, not by accident.** Duke breast's slice
task is defined by distance from the tumour box, so a high positional null there is a
tautology we quantify rather than a confound we discover; and every patient in that cohort
has cancer, so the patient-level AUROC is undefined (922 of 922 positive) and we report it
as unavailable. DeepLesion's eight classes are anatomical regions, so position predicting
them is the task.

**The PI-CAI comparison is across cohorts.** The published 0.91/0.86 are on the hidden
1,000-case testing cohort (400-case subset for the reader comparison); our baseline is on the
public 1,500-case Training and Development set. A strict reading makes those rows
non-comparable. We score them because the caveat runs against the null.

**The trivial fraction's interval is too narrow as a statement about the ratio.** It
propagates uncertainty in the baseline only; the published number enters as a fixed
constant, because we almost never have its sampling distribution. Where the publication
gives a half-width (Rempe et al. report ±1.8 on 0.861) we report it but do not combine it,
because the resampling unit behind their interval is not stated to be the subject.

**Our own clinical cohorts are small, single-institution, single-vendor and 3 T only.**
Prostate T2 n=67, prostate DWI n=45, breast n=70. Official test folds are 4–7 subjects,
which is why the cross-validated pooled out-of-fold estimate is the headline and the
official split is reported only as a labelled secondary analysis. A null on 67 patients is
a null on 67 patients, not a statement about MRI phase.

**Our own pipeline fails one of its own controls on its primary cohort.** The label
permutation null for prostate T2 has an observed range of [0.548, 0.645] over 20 distinct
replicates, which does not contain 0.500. That is a failure of criterion C3, and it means
nothing downstream of it on that cohort can be believed on its own terms. We report the
cohort as NOT SUPPORTED partly for that reason, and we state it here rather than in a
supplement.

**The remedy is validated on four score vectors, not on a benchmark suite.**
Position-stratified AUROC collapses the zero-image nulls (0.854→0.546, 0.851→0.539) and
moves our two trained arms (0.574→0.467, 0.616→0.562). We have not demonstrated that it
preserves a genuine effect, because we do not have a benchmark with a demonstrated genuine
slice-level effect to test it on. That is the most important missing validation in this
paper and it should be named as future work in those words.

**Reconstruction fidelity is validated for magnitude only.** Every vendor reference in these
releases is a magnitude image. The phase channel is never directly validated and inherits
credibility only through sharing the same complex image. The breast reference (`temptv`) is
the vendor's temporal-TV-regularised reconstruction of the same radial k-space, not an
independent ground truth, so r = 0.977 there is agreement between two estimators and is the
weakest of the five comparisons.

**The audit is a snapshot of what one analyst could obtain without accepting a data-use
agreement.** RSNA 2019 ICH, RSNA 2023 Abdominal Trauma and RSNA 2022 Cervical Spine are all
behind click-through agreements that were not accepted; PI-CAI's slice-level arm would need
1,295 lesion-delineation volumes and a NIfTI reader. Those are the obvious next targets and
their absence is a limit on the audit's breadth, not evidence about those benchmarks.

---

## 8. What a reviewer will attack, and how the paper pre-empts it

### 8.1 "This is known. Yagis 2021 and Badgeley 2019 already showed it."

*Pre-empt:* the introduction's second paragraph, before any of our results, and again in the
discussion. Concede fully, then make the distinction that survives: prior work measures the
inflation caused by a *wrong* split; we measure what remains under a **correct,
patient-disjoint** split. Rempe et al.'s split is patient-disjoint. State this in one
sentence in the abstract.
*Residual risk:* high. This is the most likely rejection reason and the concession must be
generous enough that the reviewer feels read rather than argued with.

### 8.2 "Yan et al. 2018 already published a location-only baseline on DeepLesion."

*Pre-empt:* we found it ourselves and it is in the limitations, in the DeepLesion results
paragraph, and cited in the introduction. Our 0.557 sits *below* their 0.597, and we say so.
The remaining difference is pixel-free versus image-derived position, and it is stated as a
difference in form.
*Residual risk:* medium, and rising if the proper prior-art search finds more. Do that search
before submission (§`COLLABORATORS.md` is not the right place for it — it is our job).

### 8.3 "Prostate cancer really does concentrate in the mid-gland. Your positional 'null' is anatomy."

*Pre-empt:* three moves. (i) Agree that it is anatomy — that is the point: an evaluation
protocol that credits anatomy to a model is the problem, not the anatomy. (ii) Show that the
patient-level reading of the *same scores* is 0.506 / 0.424, so whatever the model ranks
correctly at the slice level it does not rank patients. (iii) Show the remedy: stratifying
on position collapses the null to 0.546 / 0.539, which is the metric we ask reviewers to
require. Add a clinician co-author's one-paragraph statement on the anatomical expectation
(`COLLABORATORS.md` §3) so the concession is authoritative rather than defensive.
*Residual risk:* medium-low once the clinical paragraph is in.

### 8.4 "You compare against a preprint."

*Pre-empt:* say it in the results the first time the number appears, not only in the
limitations. Check the publication status again the week of submission. Offer Rempe et al.
sight of the critique in advance and record their response (draft email in
`COLLABORATORS.md` §4) — a reviewer who sees "the authors were shown this and their reply is
in the supplement" cannot use it as a rejection reason.
*Residual risk:* medium. If a stronger peer-reviewed anchor can be obtained, get it.

### 8.5 "Your tool only ever confirms itself."

*Pre-empt:* three built-in negatives, all in the main text. LUNA16 at CPM 0.0020 on the
challenge's own metric. PI-CAI NOT MATCHED at the unit its authors report, with the
positional null exactly 0.500. The synthetic clean control shipped with the tool, trivial
fraction −0.041 [−0.338, 0.304]. Plus the two confound cohorts from the worked example
(brain positional 0.480 [0.446, 0.513], knee positional 0.500).
*Residual risk:* low. This is the paper's strongest structural defence and it should be in
the abstract.

### 8.6 "Slice-level AUROC is a legitimate metric for localisation, not diagnosis."

*Pre-empt:* agree, and carve the scope explicitly in the methods. The critique is of
slice-level numbers presented as evidence about *patients*. Duke breast is the worked
example of a genuinely within-patient localisation task, and we say its 0.823 is a
tautology. Add one sentence: "where the clinical question really is localisation, report it
as localisation and report the patient-level number too, because readers will otherwise
assume it."
*Residual risk:* low.

### 8.7 "The trivial fraction's confidence interval is wrong."

*Pre-empt:* stated as a limitation in `protocol.md` Rule 7, in every generated card, and in
the methods. It propagates baseline uncertainty only. Ask the biostatistician
(`COLLABORATORS.md` §2) whether a defensible two-source interval is available when the
publication reports only a ± half-width without naming the resampling unit; if not, say so
in the paper and keep the one-sided reading (the interval is a statement about our baseline,
not about the ratio).
*Residual risk:* medium. A statistically strong referee will press here, and the honest
answer — "we cannot, and here is why" — is acceptable only if it is offered first.

### 8.8 "MATCHED is defined by a CI upper bound touching a point estimate. That is not a test."

*Pre-empt:* the rule is stated before any result (`audit_results.md` §1) and it is
deliberately generous to the *benchmark* in one direction and to us in another. Say plainly
that it is a descriptive decision rule, not a hypothesis test, and that no p-value is claimed
from it. Ask the biostatistician whether an equivalence framing (TOST against a margin) would
be stronger and whether it would change any verdict.
*Residual risk:* medium. This is the second most likely statistical objection.

### 8.9 "n = 46 test patients. The intervals are enormous."

*Pre-empt:* they are, and they are reported clustered on patient. 1,399 slices from 46
subjects with 68 positive slices and 20 positive patients (T2). Report positive-patient
counts everywhere, as the checklist demands of everyone else. Note that this is *their*
official test split, not one we chose, and that the small-n objection applies with equal
force to the published 0.861.
*Residual risk:* low, and it cuts both ways.

### 8.10 "You audit a benchmark you also use as your own study's dataset. That is circular."

*Pre-empt:* separate the two roles explicitly in the methods. The audit uses Rempe et al.'s
published 312-patient label CSV and their own `data_split` column; our study uses a
locally reconstructed 45/67-patient cache with its own subject-level CV. Different tables,
different splits, different subjects. The waterfall figure shows both and labels which rung
comes from which.
*Residual risk:* low if the figure is labelled; high if it is not.

### 8.11 "Your own study is a null result on 45–70 patients. Why is it in the paper?"

*Pre-empt:* say what it is for in the first sentence of that section: it is the worked
example of applying the protocol to one's own work, including reporting a control our own
pipeline failed. It is not evidence about MRI phase. Cap it at ~400 words in the main text.
If a reviewer still objects, it moves to the supplement in full and the main text keeps only
the metadata-baseline row (breast `folder` 0.743 vs network 0.633) and the coil-count
result (0.921), both of which are protocol evidence rather than phase evidence.
*Residual risk:* medium. Be ready to move it.

### 8.12 "The 20-bin choice is arbitrary."

*Pre-empt:* the bin sweep is reported for every arm (T2: 5=0.835, 10=0.848, 20=0.854,
30=0.854, 50=0.856) and the *no-fit* centrality score — which uses no training data at all —
reaches 0.825 (T2) and 0.841 (DWI). The result does not depend on fitting anything.
*Residual risk:* very low.

### 8.13 "`pip install trivialbaselines` does not work."

*Pre-empt:* fix it before submission. Either publish to PyPI or change the command in
`protocol.md` and `checklist.md` to match `trivialbaselines/README.md`. A reviewer who tries
the advertised command and gets a 404 will distrust everything else.
*Residual risk:* certain if unfixed, zero if fixed.

### 8.14 "Six benchmarks is not systematic."

*Pre-empt:* do not claim it is. Claim seven label files, fifteen rows, uniform reporting, a
stated entry criterion and a published exclusion list — and name the targets not reached and
why (`audit_results.md` §6). An audit that reports what it could not reach is more credible
than one that reports only what it could.
*Residual risk:* medium, and it is the reason for the venue recommendation in §3.

---

## 9. Blockers to clear before submission

Ordered by how much damage each does if missed.

1. **Redo the prior-art search properly.** Google Scholar, MICCAI/MIDL/ML4H proceedings,
   RSNA ICH Kaggle solution write-ups. Yan et al. 2018 was found only during the audit;
   assume there are others. (§8.2)
2. **Check whether arXiv:2407.06165 has been published**, the week of submission. (§8.4)
3. **Fix the install command** so `protocol.md`, `checklist.md` and
   `trivialbaselines/README.md` agree. (§8.13)
4. **Reverse the T2/DWI recommendation in `paper/audit_targets.json`** — T2 is the correct
   arm; the persisted artefact is already right; the docstring at
   `pipeline/s12_rempe.py:272-278` is wrong. (§5.4 R2)
5. **Send the three collaborator requests** in `paper/COLLABORATORS.md`, with the Rempe
   email first because it has the longest turnaround.
6. **Delete or quarantine `manuscript/*.docx`** from anything public.
   (`manuscript/DO_NOT_SUBMIT.md`)
7. **Replace every occurrence of "within 0.018"** with the three per-cohort background-only
   gaps (0.025 / −0.009 / 0.037). The 0.018 is a mean across cohorts and no cohort attains
   it. (§5.4 R9)
8. **Decide, in writing, whether the worked example is main text or supplement**, and hold
   to it. (§8.11)
9. **Reconcile three small numeric inconsistencies between our own documents**, because a
   reviewer who finds one in a paper about other people's numbers will look for more:
   - the fastMRI+ knee roster is **1,173 volumes** in `audit_results.md` §3.3 and **1,172**
     in `audit_targets.md` §1.2. Recount from `knee_file_list.csv` and fix both;
   - LUNA16's positional slice interval is **[0.513, 0.558]** in the persisted card and the
     JSON (`slice_ci_clustered` = 0.5135–0.5580) and **[0.514, 0.558]** in
     `audit_results.md` §4. The card is authoritative; `DRAFT.md` already uses it;
   - the DeepLesion majority-class anchor is **0.2361** in the Yan-conditions run and
     **0.2233** in the earlier official-split probe (`audit_targets.md` §2.4). They are
     different partitions and both are correct, but the draft must never put them in the
     same sentence without saying which is which.
