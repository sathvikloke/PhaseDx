# What a slice-level benchmark certifies without the pixels: a label-file audit of six public medical imaging benchmarks

**Draft 1, 2026-07-29.**

> **Provenance markers.** Every number in this draft carries a marker of the form
> `[→ file]` naming the artefact it was read from. They are for internal verification and
> must be stripped before submission. If a number has no marker, it must not be in the
> paper. Framing, venue reasoning and the reviewer-attack analysis live in
> `paper/PAPER_PLAN.md`; the seven-rule protocol in `paper/protocol.md`; the one-page
> checklist in `paper/checklist.md`; the full audit record in `paper/audit_results.md`.

**Authors:** [to be completed — see `paper/COLLABORATORS.md` for the two co-author roles
that are required and not yet filled: a senior radiologist and a biostatistician.]

---

## Abstract

**Background.** Three-dimensional medical images are frequently labelled and evaluated one
slice at a time, and performance is reported by pooling slices. Shortcut learning,
acquisition confounding and the inflation caused by slice-level data splits are all
documented. What has not been measured is how much of a published slice-level number
survives as reachable *without any image* under a split that is correct.

**Methods.** We define a family of pixel-blind null models that require only four columns a
benchmark usually publishes — subject identifier, slice index, label and train/test
assignment — and fit them on training slices, score them on test slices, and read them at
both the slice and the patient level with subject-clustered bootstrap intervals. We audited
six public benchmarks across seven label files: fastMRI Prostate (T2 and DWI), DeepLesion,
fastMRI+ knee, Duke Breast Cancer MRI, PI-CAI and LUNA16. Verdict rules and the comparison
statistic were fixed before any result. We release the implementation as `trivialbaselines`
(MIT; `numpy` and `pandas` only), and a seven-rule reporting protocol with a one-page
checklist.

**Results.** Fifteen rows were produced; twelve carry a defensible published comparator. Six
are matched, three partial, three not matched; the remaining three are non-comparable and
unscored. **All six matched rows come from one benchmark.** On Rempe et al.'s own published
fastMRI Prostate label file and patient-disjoint split, a 20-bin estimate of
P(label | relative slice position) reaches slice-level AUROC 0.854 [0.812, 0.891] (T2) and
0.851 [0.816, 0.887] (DWI) against their reported 0.861, and a score using no training data
at all — negative distance from the middle of the stack — reaches 0.841 (DWI)
[→ `pipeline_out/trivial_baselines/fastmri_prostate_{t2,dwi}_published.md`;
`pipeline_out/rempe/positional_baseline*.json`]. Read at the patient level, the identical
scores give 0.506 [0.381, 0.632] and 0.424 [0.298, 0.547]. The same divergence appears on
fastMRI+ knee (0.873 [0.858, 0.886] slice, 0.510 [0.428, 0.592] patient) and, in a fourth
form, on Duke breast, where the slice-level baseline reaches 0.823 [0.811, 0.834] and the
patient-level AUROC is undefined because all 922 patients are positive. It is not universal:
on DeepLesion, whose labels are anatomical regions, the positional model is high at both
units (0.977 / 0.954 for pelvis), and on LUNA16 it is at chance at both (0.534 / 0.581) and
scores CPM 0.0020 on the challenge's own metric against a published sensitivity above 0.95 at
under one false positive per scan. PI-CAI, evaluated at the case level its authors report,
is not matched (0.692 [0.626, 0.755] against 0.91), and its positional baseline is exactly
0.500. Stratifying the slice-level statistic within bins of relative position collapses the
matched nulls to 0.546 and 0.539.

**Conclusion.** On one of six benchmarks audited, a published slice-level evaluation protocol
certifies a number that a model with no access to the pixels also reaches. More generally,
and independently of any published comparator, slice-level and patient-level readings of the
same pixel-blind score vector disagree by 0.3–0.4 AUROC on three of six benchmarks. This is
checkable by a third party in one command, from a label file, without a data-use agreement
for pixels and without a GPU.

---

## 1. Introduction

A three-dimensional acquisition is often labelled slice by slice — this slice contains the
lesion, that one does not — and a classifier's performance is then reported by pooling
slices into a single AUROC. The clinical question, almost always, is about a patient. The
gap between those two things is arithmetic, not opinion: a slice-level ranking can be
dominated by *where* a slice sits in the stack, because findings are not uniformly
distributed along an organ, and a model that learns only that geometry ranks slices well and
patients not at all.

None of this is new, and the paper is written on the assumption that the reader already
knows it. Shortcut learning is a named and canonical phenomenon [Geirhos et al., *Nat Mach
Intell* 2020]. In medical imaging specifically, Badgeley et al. showed that hip fracture was
predicted at AUC 0.78 from the radiograph, 0.91 with hospital process features added, and
0.52 — chance — on a test set balanced across patient and process variables, with scanner
model predictable from the radiograph at AUC 1.00 [*npj Digit Med* 2019;2:31]. DeGrave et al.
showed radiographic COVID-19 models selecting shortcuts over signal [*Nat Mach Intell*
2021]. Ong Ly et al. found across thirteen datasets that performance is frequently
overestimated by up to 20% on average through shortcut learning of hidden acquisition biases
[*npj Digit Med* 2024]. On the evaluation unit, Yagis et al. measured that slice-level
cross-validation boosted test accuracy by 30% (OASIS), 29% (ADNI), 48% (PPMI) and 55% (a
local cohort), and that on **randomly labelled** data a slice-level split reached about 96%
accuracy against 50% for a subject-level split [*Sci Rep* 2021;11:22544]. Tampu et al.
reported the same inflation in OCT [*Sci Data* 2022]. Wen et al. found that more than half
of surveyed Alzheimer's classification papers may have suffered from data leakage, and — a
trivial-baseline result of exactly the kind we generalise — that the CNN approaches did not
outperform an SVM on voxel-based features [*Med Image Anal* 2020;63:101694]. Kapoor and
Narayanan placed leakage in a cross-disciplinary frame, affecting at least 294 papers across
17 disciplines [*Patterns* 2023]. Varoquaux and Cheplygina, and Roberts et al., have already
recommended better baselines at field level [*npj Digit Med* 2022;5:48; *Nat Mach Intell*
2021].

Position-only baselines are not new either. Yan et al., in the paper that defines the
DeepLesion lesion-type task, report a "Baseline: Location feature" row at 59.7% eight-class
accuracy against their full method's 90.5% [CVPR 2018, Table 1]. Their location feature is
image-derived — the z coordinate comes from a self-supervised body-part regressor — and it is
offered as a retrieval feature rather than as a critique of the benchmark, but the idea of a
location-only baseline on that benchmark is theirs.

What the prior literature measures is the cost of a *wrong* split. The question we ask is
different, and it is not answered anywhere we could find: **under a correct, patient-disjoint
split, how much of a published slice-level number is reachable from the benchmark's own label
file, with no image at all?** Rempe et al.'s split is patient-disjoint
[→ `paper/audit_results.md` §3.1], so the number we report on their benchmark is not a
leakage result in the Yagis sense. It is the residue that survives doing the split correctly.

Asking it turns out to be nearly free, and that is the second contribution. The inputs a
positional null needs — subject identifier, slice index, label, split assignment — are
exactly the fields most public benchmarks publish in a CSV with no data-use agreement and no
pixels. So a benchmark can be audited by someone who will never be granted its images, on a
laptop, in one command.

This paper reports three things.

1. **A family of pixel-blind null models** — a constant predictor, a positional model, a
   volume-size model, a metadata model over acquisition and administrative fields, and their
   combination — applied uniformly, with identical reporting, to six public benchmarks across
   seven label files.
2. **An audit that reports its failures at the same prominence as its successes.** Twelve of
   fifteen rows carry a defensible published comparator: six are matched, three partial, three
   not matched. All six matched rows come from a single benchmark. Two benchmarks refused the
   null outright, and one of those, PI-CAI, is presented as a benchmark that already evaluates
   at the unit it should.
3. **A remedy and an instrument.** A position-stratified AUROC that removes exactly the share
   of a slice-level statistic that stack geometry paid for, a seven-rule reporting protocol
   in which each rule exists because a concrete failure was measured, a one-page checklist,
   and a released tool with two dependencies.

We state at the outset the sentence this evidence licenses and the sentence it does not. A
high trivial fraction says: *this published evaluation protocol certifies a number that a
model with no access to the pixels also reaches.* It does **not** say that the published
model learned nothing. We could not reproduce the pipeline of the benchmark we audit most
closely — our implementation of its protocol on our own data reaches 0.616 against its
reported 0.809 [→ `pipeline_out/s12_waterfall_magphase.log:431`] — and a label file cannot
support a claim about a model's internals in any case.

---

## 2. Methods

### 2.1 The zero-image family

Five null models, each implementing `fit` on the training rows and `score` on the test rows
[→ `trivialbaselines/README.md`, "The baselines"]:

| name | what it knows | what it stands in for |
|---|---|---|
| `prevalence` | nothing; a constant | the chance anchor, and a check that the harness is not rewarding a degenerate model |
| `positional_20bin` | P(label \| relative slice position), binned, fitted on train | stack geometry |
| `volume_size` | how many slices the volume has | protocol, scanner, acquisition era |
| `metadata_tree` | acquisition and administrative columns, depth-limited CART | release batch, matrix size, site, coil count |
| `combined_position_metadata` | position and metadata in one tree | the ceiling reachable with no pixels |

Relative position is `(slice − min_slice_in_volume) / (max − min)`, so volumes of different
depth are comparable. The positional baseline is reported over a bin sweep (5/10/20/50) and
alongside a *no-fit* variant, `−|relative position − 0.5|`, which uses no training data
whatsoever; a result that survives both cannot be a binning artefact.

### 2.2 Column discipline

Two classes of column are excluded from the metadata pool by default: **outcome-derived**
columns, which are the label under another name, and **image-derived** columns, which break
the zero-image premise. The exclusion is a fallible name heuristic, so every included and
excluded column is printed and written to the run's JSON payload, and can be set explicitly.

The exclusions are not decorative. On PI-CAI we excluded `prostate_volume` and `psad`
because both are measured *from the MRI*, and `case_ISUP`, `lesion_ISUP`, `lesion_GS`,
`lesion_PIRADS` and `histopath_type` as outcome-derived; what remained was `patient_age`,
`psa` (a blood test), `center` and the acquisition year
[→ `paper/audit_results.md` §3.5]. Including either image-derived column would have inflated
that row and broken the guarantee the paper rests on.

### 2.3 Evaluation, and the interval

Every baseline produces one score vector per test set, which is read at **both** units:
slice-level AUROC, and patient-level AUROC after aggregating each subject's slice scores.
Intervals are percentile bootstrap over **subjects** — a subject drawn twice contributes all
of their slices twice — with 2,000 replicates unless stated, seeds recorded, and degenerate
replicates counted rather than silently dropped. The naive slice-level interval is computed
as well and reported in the JSON as `slice_ci_naive`, explicitly labelled as the incorrect
one, so that the width difference is visible rather than asserted.

The size of that difference was measured on simulated data where the true AUC is available
in closed form, Φ(μ/√(2σ²ᵤ + 2σ²ₑ)) = 0.6880, over 200 datasets of 20 patients × 15 slices
with 500 bootstrap replicates each: the subject-clustered interval covered the truth **91.5%**
of the time at a nominal 95% with mean width 0.370; the naive slice-level interval covered
it **46.5%** of the time with mean width 0.117, a factor of **3.18** narrower
[→ `pipeline/s04_stats.py --self-test`, block [6], re-run 2026-07-29]. A nominal 95%
interval that covers 46.5% of the time is not a conservative approximation; it is a
different claim from the one being written down.

### 2.4 Each baseline's own permutation null

A null model's chance level is not automatically 0.5. Fit a metadata model out of fold on a
subject-level label and the rate fitted is anti-correlated with the rate scored, because
positives are a finite population: a level that was positive-rich in training is
positive-poor in the fold left out. On a synthetic dataset whose label is by construction
invisible to metadata, the metadata baseline measures **0.424**, not 0.500
[→ `trivial-baselines --self-test`; `paper/protocol.md` Rule 4]. Judged against 0.5 that
would be a below-chance "finding" manufactured out of arithmetic. Every baseline is therefore
reported against its own permutation null, and where a permutation cannot change anything —
shuffling labels within a single-class volume — the null is reported as unavailable rather
than as *p* = 1.

### 2.5 The comparison statistic and the verdict rule, both fixed before any result

```
trivial fraction = (best zero-image baseline − chance) / (published − chance)
```

with chance = 0.5 for AUROC and the majority-class rate for multi-class accuracy. Verdicts
[→ `paper/audit_results.md` §1]:

* **MATCHED** — the upper bound of the baseline's clustered 95% interval reaches or exceeds
  the published number.
* **PARTIAL** — trivial fraction ≥ 0.30 with its interval wholly below 1.
* **NOT MATCHED** — trivial fraction < 0.30, or the baseline is statistically
  indistinguishable from chance.
* **NON-COMPARABLE** — the published number is on a different cohort, split, label definition
  or metric and could not be reconstructed; no verdict is issued.

The rule is a descriptive decision rule, not a hypothesis test, and no *p*-value is claimed
from it. Its interval propagates uncertainty in the **baseline only**; the published number
enters as a fixed constant, because a publication's sampling distribution is almost never
available. The fraction is undefined when the published number is at or below chance, and
values above 1 — the baseline exceeded the published number — are left unclipped, because
they are real outcomes.

### 2.6 The remedy: position-stratified AUROC

`stratified_auc` computes the Mann–Whitney statistic *within* strata of relative slice
position, so only same-position positive/negative pairs contribute. It removes exactly the
share of a slice-level AUROC that stack geometry paid for, and nothing else. It is called on
a paper's own test predictions and needs no access to the tool's baselines.

### 2.7 Benchmark selection

A dataset entered the audit if the four positional fields could be obtained **without
downloading pixel data and without a data-use agreement covering pixels**. Datasets
investigated and excluded, with reasons, are as important as those included
[→ `paper/audit_targets.md`, Tier 3]: CQ500 publishes scan-level reads only; BraTS, KiTS,
the Medical Segmentation Decathlon, AMOS and TotalSegmentator are segmentation benchmarks
whose masks ship with the images and which publish no slice-level classification number;
PROSTATEx gives finding position in patient coordinates and needs DICOM headers to convert;
MRNet is exam-level by construction.

Targets that were in scope but not reached are reported in Results §3.7 rather than omitted.

### 2.8 The worked-example cohorts

To demonstrate the protocol on data where every control could be run, we applied it to our
own study of whether MRI phase carries tumour signal beyond magnitude. Three clinical cohorts
(prostate T2 n = 67, prostate DWI n = 45, breast n = 70) and two confound cohorts whose label
is an acquisition property rather than a diagnosis (brain n = 454, label: receive-coil count
≥ 16; knee n = 96, label: pulse sequence). 102 training runs
[→ `pipeline_out/results`, 103 JSONs of which one is `statistics.json`] and 456 control runs
[→ `pipeline_out/controls`], five-fold subject-level cross-validation, two seeds, pooled
out-of-fold estimation with each subject tested exactly once.

Reconstructions were validated against the vendor reference images shipped in the same HDF5
files [→ `pipeline_out/recon_fidelity/recon_fidelity_summary.json`;
`run_streamA.log:190-206`]: brain *r* = 1.000 (2,270 slices / 454 files), knee *r* = 1.000
(995 / 199), prostate T2 *r* = 0.9982 (2,039 / 67), prostate DWI *r* = 0.9835 per file with
the low-*b* volumes magnitude-averaged as the vendor does (0.8866 per cached slice, because
the vendor trace averages roughly 14 acquisitions and the cache stores one), breast
*r* = 0.9772. **The breast comparison is the weakest of the five and is not a ground truth:**
`temptv` is the vendor's temporal-TV-*regularised* reconstruction of the same radial k-space,
so that correlation is agreement between two estimators. Every vendor reference in these
releases is a magnitude image, so these numbers validate the magnitude reconstruction only;
the phase channel is never directly validated and inherits credibility only through sharing
the same complex image.

### 2.9 Software

`trivialbaselines` v1.0, MIT licensed, depends on `numpy` and `pandas` and nothing else
[→ `trivialbaselines/pyproject.toml`]. The absence of `torch` and `scikit-learn` is a
deliberate property, not an accident: the premise is that a benchmark can be audited with no
images, no data-use agreement and no GPU, and a reader should be able to verify that from the
dependency list rather than take it on trust. The rank statistics, the clustered bootstrap and
the depth-limited CART are implemented against `numpy`. A `--self-test` runs synthetic data
with known answers. Every run writes a JSON payload with every number traceable and a
markdown card suitable for pasting into a supplement.

---

## 3. Results

### 3.1 The audit as a distribution, stated before any individual row

Six benchmarks were audited on seven label files, producing fifteen rows. Twelve carry a
defensible published comparator: **six MATCHED, three PARTIAL, three NOT MATCHED**. The
remaining three are NON-COMPARABLE and are not scored. **All six MATCHED rows come from a
single benchmark, fastMRI Prostate under Rempe et al.'s protocol.** Every other benchmark
audited either resisted the null outright or was matched only in part
[→ `paper/audit_results.md` §0, §2] (Table 2, Figure 4).

That distribution is the finding, and it constrains what may be said. The general claim that
trivial baselines match published performance on medical imaging benchmarks is **not**
supported by these data. It is supported for one benchmark, strongly and reproducibly, and
partially for a second.

### 3.2 fastMRI Prostate: a published slice-level protocol matched without pixels

The audit ran on the authors' own published label CSVs, downloaded from the public
repository, using the in-file `data_split` column: 6,647 training / 1,462 validation / 1,399
test slices for T2, patient-disjoint, with validation rows excluded from both arms. The label
is PI-RADS > 2 per slice. The evaluation unit is slice-level AUROC, which is what the authors
report and the only unit they report [→ `paper/audit_results.md` §3.1;
`pipeline_out/trivial_baselines/fastmri_prostate_t2_published.md`].

| | slice-level AUROC |
|---|---|
| published headline, image + k-space (their Table II) | 0.861 ± 1.8 |
| published PCA ×2, magnitude | 0.813 ± 2.2 |
| published PCA ×2, magnitude + phase | 0.809 ± 2.1 |
| published R = 16, PCA coil combination | 0.714 ± 2.9 |
| **zero-image positional baseline, T2 label file** | **0.854 [0.812, 0.891]** |
| **zero-image positional baseline, DWI label file** | **0.851 [0.816, 0.887]** |

Published values are transcribed from their Table II, not recomputed
[→ `pipeline/s12_rempe.py::REPORTED`]. Baseline values and intervals are from
[→ `pipeline_out/trivial_baselines/fastmri_prostate_{t2,dwi}_published.md`].

Trivial fractions against the headline are **0.981 [0.865, 1.084]** (T2) and
**0.973 [0.876, 1.073]** (DWI); against the PCA magnitude+phase arm, 1.146 [1.011, 1.266]
and 1.137 [1.023, 1.253]; against the R = 16 arm, 1.655 [1.459, 1.829] and
1.642 [1.478, 1.810] [→ `paper/audit_results.md` §2.1]. Fractions above 1 mean the
zero-image baseline exceeded the published number, and are reported unclipped.

Two features of this result matter more than the point estimate.

**It does not depend on fitting anything.** The bin sweep gives 0.835 (5 bins), 0.848 (10),
0.854 (20), 0.854 (30), 0.856 (50) on T2, and the *no-fit* centrality score —
`−|relative position − 0.5|`, which uses no training data at all — reaches **0.825** on T2
and **0.841** on DWI [→ `pipeline_out/rempe/positional_baseline{,_dwi_labels}.json`,
`bin_sweep` and `centrality_no_fit`]. A published slice-level headline of 0.861 is
approached by a function of the slice index alone.

**The test arm is small and is theirs, not ours.** 46 patients, 1,399 slices, 68 positive
slices and 20 positive patients (T2); 1,395 slices, 83 positive slices and 27 positive
patients (DWI). The interval is clustered on patient and is reported as such. That small-n
objection applies with equal force to the published 0.861.

**What this licenses.** That *their evaluation protocol* certifies a number a pixel-blind
model also reaches. It does not say their model learned nothing. Our implementation of their
protocol on our own prostate DWI cache reaches 0.574 [0.516, 0.629] for the magnitude arm
against their reported 0.813, and 0.616 [0.559, 0.672] for magnitude + phase against their
reported 0.809 [→ `pipeline_out/s12_arm_mag.log:131`;
`pipeline_out/s12_waterfall_magphase.log:431`]. We do not reproduce their pipeline, and we
therefore make no claim about it (Figure 3).

*Note for revision.* `paper/audit_targets.json` currently recommends DWI as the correct arm.
The evidence points the other way: the authors' abstract states "312 subject and a total of
9508 slices", and 9,508 is the exact row count of `t2_slice_level_labels.csv` (DWI has
9,490). **T2 is the correct arm.** Both arms are reported here so that no conclusion depends
on resolving it [→ `paper/audit_results.md` §3.1].

### 3.3 The unit of evaluation: the result that generalises

Every cell in Table 3 is our own computation on a published label file. No published number
enters, so none of the comparability objections that can be raised against §3.2 apply.

| dataset-arm | zero-image positional, **slice** | zero-image positional, **patient** |
|---|---|---|
| fastMRI Prostate T2 | 0.854 [0.812, 0.891] | **0.506** [0.381, 0.632] |
| fastMRI Prostate DWI | 0.851 [0.816, 0.887] | **0.424** [0.298, 0.547] |
| fastMRI+ knee, meniscus tear | 0.873 [0.858, 0.886] | **0.510** [0.428, 0.592] |
| fastMRI+ knee, any annotated finding | 0.801 [0.779, 0.824] | **0.558** [0.470, 0.648] |
| Duke breast, owner-defined slice task | 0.823 [0.811, 0.834] | **undefined** (922 of 922 patients positive) |
| DeepLesion, pelvis vs rest | 0.977 [0.969, 0.984] | 0.954 [0.939, 0.967] |
| LUNA16 candidates | 0.534 [0.513, 0.558] | 0.581 [0.538, 0.613] |
| PI-CAI, case level | not applicable (no slice index in the marksheet) | 0.692 [0.626, 0.755] (metadata only) |

[→ `paper/audit_results.md` §4; the eight cards in `pipeline_out/trivial_baselines/`.]

Nothing changes between the two columns except the unit at which the ranking is performed.
On three benchmarks the same score vector looks like a working detector at the slice level
and like nothing at all at the patient level. Duke breast is a fourth form of the same
protocol problem: a slice-level number is computable and a patient-level number is not,
because every patient in that cohort has cancer, and the harness reports it as unavailable
rather than inventing a value.

**Two benchmarks do not show the effect, and both are reported here rather than in a
supplement.** DeepLesion does not collapse, and should not: its labels are anatomical
regions, so they are patient-level facts about where lesions were found, and position
predicts them at both units. LUNA16 is at chance at both. Stating all the outcomes is what
makes the first three credible (Figure 2).

### 3.4 Two benchmarks that resist the null

**LUNA16.** Comparing a positional AUROC against a published competition performance metric
would be exactly the incomparable comparison this audit exists to refuse. We therefore scored
the same 20-bin positional estimator on the challenge's own metric — sensitivity at 1/8, 1/4,
1/2, 1, 2, 4 and 8 false positives per scan, out of fold on a scan-disjoint five-fold split
[→ `pipeline/audit_prep/luna16_cpm.py`]. Result: **CPM 0.0020**, sensitivity **0.0006** at
one false positive per scan, against a random-score reference of 0.0027, and against a
published combined-solution sensitivity above 0.95 at under one false positive per scan
[Setio et al., arXiv:1612.08012]. The positional baseline is not merely worse than the
published system; on this benchmark it is at or below chance. As an AUROC on 754,975
candidates from 888 scans it reaches 0.534 [0.513, 0.558] at the slice level and 0.581
[0.538, 0.613] at the patient level, with the best zero-image combination at 0.539
[0.520, 0.565] [→ `pipeline_out/trivial_baselines/luna16_fp_reduction_candidates.md`].

One asterisk travels with it. The false-positive-reduction track is conditioned on
`candidates_V2.csv`, a candidate list produced by image-based detectors, so "zero-image" here
means "zero image *given the published candidate list*". The label being predicted — is this
candidate a nodule — is not predictable from where the candidate sits in the scan, and that
is the finding, but the setup is not pixel-free in the same clean sense as fastMRI Prostate
[→ `paper/audit_results.md` §3.6]. The harness's own protocol check should also be reported:
the constant predictor scored 0.483 rather than 0.500 here, because pooling out-of-fold
predictions across folds whose training prevalence differs makes fold identity rankable on
its own. That is the floor of what any pooled number on this file can mean.

**PI-CAI.** The published numbers are 0.91 (95% CI 0.87–0.94) case-level AUROC for the AI
system and 0.86 (0.83–0.89) for 62 radiologists reading PI-RADS 2.1 [Saha et al., *Lancet
Oncol* 2024;25:879-887]. Our best zero-image baseline, at the case level its authors report,
is **0.692 [0.626, 0.755]** — trivial fractions 0.467 [0.307, 0.622] and 0.532 [0.350, 0.708],
both **NOT MATCHED** [→ `pipeline_out/trivial_baselines/picai_case_level.md`;
`paper/audit_results.md` §2.1]. The strongest single columns are `patient_age` at 0.639 and
`psa` at 0.638.

The **positional** baseline on PI-CAI is exactly **0.500** at every bin setting. That is the
correct registration of "inapplicable" — the marksheet has one row per case and no slice
index — and not a computed result.

A cohort caveat runs with these two rows and is stated because it cuts against the null
rather than for it: the published numbers are on the hidden 1,000-case testing cohort (a
400-case subset for the reader comparison), while our baseline is on the public 1,500-case
Training and Development set, using the benchmark's own official five-fold splits. A strict
reading makes the rows non-comparable and that reading is defensible; we score them anyway,
because our baseline had the larger and more heterogeneous cohort and still lost.

**PI-CAI should be read as the paper's positive example.** It evaluates at the patient level
by design and publishes no slice-level number to attack, which is exactly what removes the
positional exposure. But its metadata baseline still reaches 0.692. Fixing the reporting unit
does not fix acquisition confounding; the two problems are independent, and a benchmark can
solve one and not the other.

### 3.5 DeepLesion: where the positional null is the task

Yan et al.'s evaluation conditions were reconstructed rather than assumed, and the first
attempt would have been wrong. Their Table 1 test set has 4,927 samples, which is *exactly*
the row count of DeepLesion's official `Train_Val_Test == 3` split — a coincidence that
invites a false match. Their own text describes something else: a random patient-disjoint
25/25/50 partition of the type-labelled rows, fitting on the 25% seed set.
`pipeline/audit_prep/deeplesion_yan_conditions.py` rebuilds that partition and repeats it
over 200 draws so the comparison is not hostage to one seed (mean seed 2,454 rows, mean test
4,900, against their reported 4,927). Under those conditions the zero-image eight-class
accuracy is **0.5571**, sd 0.0131, range [0.5243, 0.5778] over partitions, against a majority
class of **0.2361** [→ `paper/audit_results.md` §3.2]. For reference, the official-split
number is 0.5602 [0.5344, 0.5868] with a patient-clustered bootstrap, which is reassuringly
similar but is not what is scored.

| published comparator (Yan et al., CVPR 2018, Table 1) | value | trivial fraction | verdict |
|---|---|---|---|
| Triplet with type + location + size | 0.905 ± 0.002 | 0.480 [0.431, 0.511] | PARTIAL |
| Baseline: multi-scale ImageNet feature | 0.862 | 0.513 [0.460, 0.546] | PARTIAL |
| **Baseline: Location feature** (their own, image-derived) | 0.597 | 0.889 [0.799, 0.947] | PARTIAL |

**This row must be read conservatively and we say so in the results, not only in the
discussion.** DeepLesion's eight coarse classes are *bone, abdomen, mediastinum, liver, lung,
kidney, soft tissue, pelvis* — anatomical regions. A z-coordinate predicting an anatomical
region is the task, not a confound. This is the reference level a lesion-type classifier must
clear. It is **not** evidence that DeepLesion papers are unsound.

The comparison also establishes the honest position on novelty. Yan et al.'s own
location-only baseline reaches 0.597; ours, taken from the published label file with no image
and no body-part regressor, reaches 0.557. Our number is *below* theirs. What is left as new
is that the position can be had from the label file at zero cost, not that position predicts
lesion type.

One metadata finding on this benchmark deserves its own sentence. On the official split,
one-vs-rest classification of lung lesions reaches slice AUROC **0.911** from the
`DICOM_windows` header column alone — the window and level stored in the header, which is
`−1500, 500` for lung-reconstructed series and `−175, 275` otherwise. Position alone gives
0.872; the two together give 0.962 [0.949, 0.973]. Per-class zero-image ceilings on the
official split run: pelvis 0.982, lung 0.962, mediastinum 0.957, kidney 0.896, abdomen 0.886,
liver 0.876, bone 0.832, soft tissue 0.831 [→ `paper/audit_results.md` §3.2;
`pipeline_out/trivial_baselines/deeplesion_*_vs_rest.md`]. No published per-class AUROC was
located, so these are reference levels only.

### 3.6 Metadata alone can beat a trained network

The positional null is one member of the family; the metadata null is another, and it fails
differently. In the worked-example cohorts, where both the acquisition fields and the trained
model's out-of-fold predictions are available on the same subjects:

| cohort | field | field predicts label | trained phase network predicts label |
|---|---|---|---|
| breast (n = 70) | `folder` — the release batch / download tarball, 7 levels | **0.743** | **0.633** (seed 42), 0.630 (seed 123) |
| prostate T2 (n = 67) | `kspace_shape` — acquisition matrix, 14 levels | **0.609** | 0.483 (seed 42), 0.462 (seed 123) |

Both columns are subject-level AUROCs on the same subjects and are therefore directly
comparable [→ `python pipeline/s08_belowchance.py --cohort breast --condition phase`, re-run
2026-07-29; `pipeline_out/s08_belowchance.log` for prostate T2]. In the breast cohort the
release batch also explains more of the *model's score variance* than the true label does:
η² = 0.108 for `folder` against 0.033 for the label at seed 42, and 0.151 against 0.046 at
seed 123.

Which tarball a scan was downloaded in has no causal relationship to whether the patient has
cancer. It predicts the label at 0.743 because releases are assembled over time and enriched
differently. That is why the checklist asks specifically for the release batch, source
directory or download tarball to be among the fields tested; it is the field authors are
least likely to think of and it is administrative rather than physical, so a model that reads
it is reading nothing about the patient at all.

### 3.7 What the label files could not support

Three targets in scope were not reached, and the reasons are findings about release practice
rather than about the benchmarks' scientific quality [→ `paper/audit_results.md` §6].

**RSNA 2019 Intracranial Haemorrhage** is the highest-impact target on the list because its
**official competition metric is per-image**, which is exactly the evaluation unit this paper
concerns. Its `stage_2_train.csv` is keyed by `ID_<SOPInstanceUID>_<subtype>` and carries only
the label: no patient identifier, no study identifier, no slice position. Locating the slice
requires DICOM headers from the roughly 450 GB image release, or an unprovenanced third-party
metadata CSV. **A benchmark whose official metric is per-slice publishes a label file from
which the slice cannot be located.** Access is additionally behind a click-through Research
Use Agreement, which this analysis did not accept.

**fastMRI+** publishes positive annotations only, so negative slices are implicit and the
table cannot be built from the annotation file alone: the slice count of each volume comes
from the fastMRI HDF5 headers. That is a header read rather than a pixel download, but it
needs registration and the archive. **fastMRI+ is therefore not a label-file-only target and
must not be described as one.** Our coverage is also partial: the knee roster is 1,173 volumes
and we hold 199 (17%), of which 155 carry at least one annotation, so the audit runs on a
subset no published number would use. The maintainers themselves describe the labels as an
indication of where a pathology could be present rather than adjudicated ground truth.

**Duke breast** required the total slice count per series, taken from the TCIA `getSeries`
metadata (tabular, CC BY-NC 4.0, no data-use agreement). The modal `ImageCount` per patient
was validated against the annotation file: for all 922 patients the annotated end slice is
strictly inside the series, and the modal and maximum counts agree. Its slice task is
positional *by construction* — the data owners' rule is that slices inside the tumour box are
positive and slices at least five away are negative, with everything between discarded — so
the 0.823 quantifies a tautology rather than indicting anyone.

### 3.8 The remedy

Stratifying the slice-level statistic within bins of relative position removes exactly the
pairs that stack geometry wins, and nothing else.

| score vector | raw slice AUROC | position-stratified |
|---|---|---|
| zero-image positional, T2 label file | 0.854 | **0.546** (5 strata) |
| zero-image positional, DWI label file | 0.851 | **0.539** (6 strata) |
| our reimplementation, magnitude arm | 0.574 | **0.467** |
| our reimplementation, magnitude + phase arm | 0.616 | **0.562** |

[→ `pipeline_out/rempe/positional_baseline{,_dwi_labels}.json`, field
`slice_auc_position_stratified`; `pipeline_out/s12_arm_mag.log:134` and
`pipeline_out/s12_waterfall_magphase.log:434`, rung W4s.]

The two zero-image rows fall to within noise of chance, which is the correct behaviour: a
model whose only input is position should score at chance once position is held fixed. The
remedy also applies to trained models, and the two rows below show it moving a real score
vector in the expected direction (Figure 5).

We have **not** demonstrated that the stratified statistic preserves a genuine effect, because
we do not hold a benchmark with a demonstrated genuine slice-level effect to test it on. That
is the most important missing validation in this work and it is named as such in §5.

### 3.9 Worked example: applying the protocol to our own study

The protocol was applied first to our own work, and it returned a null. The point of this
section is the demonstration, not the biology.

The question was whether MRI phase carries tumour signal beyond what the magnitude image
already provides, tested under nine criteria fixed in advance, with `prostate_t2`
pre-registered as the primary cohort on the basis of size (67 patients) and reconstruction
fidelity (*r* = 0.998), both fixed independently of any result. All three clinical cohorts
were **NOT SUPPORTED**: `prostate_t2` failed seven of nine criteria (C1, C2, C3, C4, C5, C6,
C8), `prostate_dwi` seven (C1, C2, C4, C5, C6, C7, C8), `breast` six (C1, C2, C4, C5, C6, C7)
[→ `pipeline_out/report/RESULTS.md` §2].

Four results from it are protocol evidence rather than phase evidence, and they are what
belongs in this paper.

**The permutation control on the primary cohort failed.** Over 20 distinct label-permutation
replicates, pooled out of fold exactly as the headline is pooled, the observed null range for
`prostate_t2` was [0.548, 0.645], which does not contain 0.500
[→ `pipeline_out/report/RESULTS.md` §8]. A pipeline that scores above chance on scrambled
labels cannot support anything downstream of it, and we report the cohort as NOT SUPPORTED
partly for that reason. We state this rather than bury it, because burying it is precisely the
behaviour the rest of the paper criticises.

**Training on air alone did not collapse.** With the anatomy removed, the background-only
control reached 0.604 against a headline of 0.629 on prostate T2 — 0.025 below, with the
control's own interval [0.528, 0.673] lying entirely above chance — 0.595 against 0.586 on
prostate DWI, where the control is *above* the headline, and 0.549 against 0.587 on breast
[→ `pipeline_out/report/RESULTS.md` §8]. A diagnostic signal that survives deleting the
patient is not a diagnostic signal.

**The input channel encodes the hardware.** On a cohort whose label contains no pathology of
any kind, a network reading phase alone predicts receive-coil count ≥ 16 at AUROC **0.921
[0.870, 0.966]** on 136 independent test subjects, against 0.913 [0.872, 0.950] for magnitude;
the paired difference is +0.007 with a 95% interval of [−0.038, +0.051], which includes zero,
so **no ordering between the channels is claimed — the level is the point**
[→ `pipeline_out/report/RESULTS.md` §4b]. Stratifying within site, the same prediction reaches
**0.979 [0.953, 0.996]** (seed 42) and 0.974 [0.940, 0.997] (seed 123), against an
unstratified 0.926 and 0.923, so the effect is not merely site
[→ `pipeline_out/robustness/s09_robustness.json`, `coil_vs_site.verdict.within_stratum.site`].
It is **not** separable from scanner model, device identity or coil array — no scanner model
carries enough subjects in both coil buckets — so we may say that phase encodes hardware, and
may not decompose that hardware into coil count as distinct from the scanner it is attached
to.

**No aggregation rescues the null.** Seven distinct patient-aggregation schemes (mean, max,
top-1, top-2, top-3, top-5 mean, 75th and 90th percentile) were swept. No scheme lifts any
clinical cohort's cross-seed interval lower bound above 0.500, and the selection-aware
envelope — the 2.5th percentile of the best-of-seven AUROC within each subject resample, which
dominates every individual scheme — tops out at **0.476**. Sixteen confound-cohort results do
clear chance under the same schemes, so the sweep is capable of detecting an effect and the
clinical null is informative rather than underpowered machinery
[→ `pipeline_out/robustness/s09_robustness.json`, `aggregation_sensitivity.verdict`].

---

## 4. Discussion

### 4.1 What a matched row licenses

A high trivial fraction is a statement about an **evaluation protocol**. It says that a
pixel-blind model reaches that much of a reported margin over chance under the same protocol.
It is not a decomposition: the baseline and the published model may exploit the same shortcut,
different shortcuts, or overlapping ones, and the fraction cannot distinguish those cases. It
is not a claim about a model's internals, and no analysis of a label file could be. We could
not reproduce the pipeline of the benchmark that produced our six matched rows, and we say so
in the abstract, the results and the limitations.

### 4.2 Why the collapse, not the match, is the general finding

The six matched rows rest on one preprint's Table II. The eight rows in Table 3 rest on our
own computations on published label tables, and no published number enters them. If a reader
grants nothing else in this paper, the sentence that survives is: *on three of six benchmarks,
the slice-level and patient-level readings of one pixel-blind score vector differ by 0.3–0.4
AUROC, and only the first of those readings is what a paper would print.*

### 4.3 Where the null legitimately fires and where it legitimately does not

Two of our rows are positional by construction. DeepLesion's classes are anatomical regions,
so position predicting them is the task and not a confound. Duke breast's slice task is
defined by distance from the tumour box, so a high positional null there is a tautology that
we quantify rather than a defect that we discover. Two benchmarks refused the null: LUNA16
decisively, on its own metric, and PI-CAI at the unit its authors report. Saying all of this
plainly is what makes the fastMRI Prostate row believable; a tool that only ever fires is not
a measurement.

### 4.4 For benchmark publishers

Three fields, in the label file that is already released, make every rule in the protocol
auditable by anyone — including people who will never be granted the pixels: a subject
identifier, a slice index or z position, and the official train/test assignment. Publishing
them costs nothing. PI-CAI already reports at case and lesion level by design, and has no
slice-level number to attack; that is the target. RSNA 2019 ICH is the counter-example, and
the fact that a benchmark whose official metric is per-slice releases a label file from which
the slice cannot be located is, on its own, an argument for the recommendation.

### 4.5 Relation to prior work

Badgeley et al.'s control is a **negative** control: balance the confounders, watch the model
collapse to 0.52. Ours is a **positive** control: fit a model on the confounder alone, and see
what it reaches. Same phenomenon, opposite direction — and the positive-control form is the one
an auditor can run without the images, which is the property that makes third-party auditing
possible at zero data cost. That is a difference in form, not in discovery, and it is stated
that way.

Against Yagis et al. and the leakage literature, the distinction is the split. Their
measurements are of the inflation caused by a *wrong* split. Every number in this paper was
obtained under a correct, patient-disjoint split; following the split rule does not protect a
benchmark from the rest.

Against Yan et al., the distinction is the source of the position. Theirs comes from a
self-supervised body-part regressor run on the image; ours comes from a column in the
published CSV. Their location-only baseline scores higher than ours.

### 4.6 What the protocol does not do

It cannot detect shortcuts that live in the pixels — scanner-specific texture, burned-in
annotation, body-part framing. Those need the images. It bounds only the part of a reported
number that is reachable *without* them, which is the part that can be checked for free, at
scale, by a third party.

---

## 5. Limitations

**We could not reproduce the pipeline of the paper we audit most closely.** Our
implementation of Rempe et al.'s protocol on our own prostate DWI cache reaches 0.616
slice-level AUROC against their reported 0.809 for the magnitude+phase arm, and 0.574 against
their reported 0.813 for the magnitude arm
[→ `pipeline_out/s12_waterfall_magphase.log:431`; `pipeline_out/s12_arm_mag.log:131`]. We
therefore have no standing to make any claim about their model. Everything we say about their
benchmark is a claim about the evaluation protocol.

**The single matched benchmark's published comparator is a preprint.** arXiv:2407.06165, v2
dated 14 April 2025, carried no journal reference at the time of writing. If it remains
unpublished, the strongest rows in this paper are comparisons against non-peer-reviewed
numbers.

**The phenomenon is known, and the closest prior art is closer than we first recorded.**
Shortcut learning, acquisition and process confounding, slice-level split inflation and
leakage as a reproducibility failure are all established and cited above. During this audit we
found a published position-only baseline on one of our own targets — Yan et al., CVPR 2018,
Table 1, 59.7% against their own 90.5%. Their feature is image-derived and used for retrieval
rather than critique, and our pixel-free version scores lower, but the idea of a location-only
baseline on that benchmark is theirs. What remains ours is that the position can be taken from
the published label file with no image and no regressor, the uniform application across
benchmarks with identical reporting, and the released tool.

**Only one benchmark was matched.** Six matched rows, all from fastMRI Prostate. The general
statement that trivial baselines match published performance on medical imaging benchmarks is
not supported by these data and is not made.

**Two audited benchmarks are not pixel-free in the same clean sense.** fastMRI+ needs slice
counts from HDF5 headers, and our coverage is 199 of 1,173 roster volumes. LUNA16's
false-positive-reduction track is conditioned on a candidate list produced by image-based
detectors.

**The PI-CAI comparison is across cohorts.** The published values are on the hidden 1,000-case
testing cohort; our baseline is on the public 1,500-case Training and Development set. A strict
reading makes those rows non-comparable; we score them because the caveat runs against the
null.

**The trivial fraction's interval is too narrow as a statement about the ratio.** It propagates
uncertainty in the baseline only, because a publication's sampling distribution is almost never
available. Where a half-width is published — Rempe et al. report ±1.8 on 0.861 — we report it
but do not combine it, because the resampling unit behind it is not stated to be the subject.

**The MATCHED rule is a descriptive decision rule, not a test.** It asks whether the upper
bound of the baseline's interval reaches a published point estimate. No *p*-value is claimed
from it and none should be read into it.

**Our own clinical cohorts are small, single-institution, single-vendor and 3 T only.**
Prostate T2 n = 67, prostate DWI n = 45, breast n = 70; official test folds are 4–7 subjects,
which is why the pooled out-of-fold cross-validated estimate is the headline and the official
split appears only as a labelled secondary analysis. A null on 67 patients is a null on 67
patients.

**Our own pipeline fails one of its own controls on its primary cohort.** The label-permutation
null for prostate T2 spans [0.548, 0.645] over 20 distinct replicates and does not contain
0.500.

**The remedy is validated on four score vectors, not on a benchmark suite.** We show that
position-stratified AUROC collapses a pixel-blind null and moves two trained arms. We have not
shown that it preserves a genuine effect, because we do not hold a benchmark with a
demonstrated genuine slice-level effect to test it on. This is the most important missing
validation in the work.

**Reconstruction fidelity was validated for magnitude only,** because every vendor reference in
these releases is a magnitude image; the phase channel is never directly validated. The breast
reference is the vendor's temporal-TV-regularised reconstruction of the same k-space rather than
an independent ground truth, so *r* = 0.977 there is agreement between two estimators.

**The audit is a snapshot of what one analyst could obtain without accepting a data-use
agreement.** RSNA 2019 ICH, RSNA 2023 Abdominal Trauma and RSNA 2022 Cervical Spine are all
behind click-through agreements that were not accepted; PI-CAI's slice-level arm would require
1,295 lesion-delineation volumes and a NIfTI reader. Their absence limits the audit's breadth
and is not evidence about those benchmarks.

---

## 6. Conclusion

A slice-level AUROC on a three-dimensional benchmark can be reached, in part or in whole, by a
model that never sees a pixel. On one of six public benchmarks audited here it is reached
almost entirely: 0.854 [0.812, 0.891] against a published 0.861, from a function of the slice
index, on the authors' own label file and split. On two others it is not reached at all, and
those results are reported at the same prominence. What generalises further than either is the
unit of evaluation: on three of six benchmarks, the same pixel-blind score vector reads as
0.80–0.87 at the slice level and as chance at the patient level.

Three fields in the label file a benchmark already publishes — subject, slice index, split —
make all of this checkable by anyone, for free, without the images. Reporting the pixel-blind
baselines beside every headline number, at the patient level, with subject-clustered intervals
and a position-stratified statistic, costs one command and removes an entire class of
ambiguity from the literature.

---

## Data and code availability

- **Tool.** `trivialbaselines` v1.0, MIT licensed, `numpy` and `pandas` only, with
  `--self-test`. [*Resolve the install path before submission: `paper/protocol.md` and
  `paper/checklist.md` advertise `pip install trivialbaselines` while
  `trivialbaselines/README.md` documents a `git clone && pip install .`. They must agree.*]
- **Audit payloads.** `pipeline_out/trivial_baselines/*.json` and `*.md`, one JSON payload and
  one human-readable card per run, 20 runs.
- **Label-table preparation.** `pipeline/audit_prep/` (eight scripts, one per benchmark plus
  the DeepLesion Yan-conditions rebuild and the LUNA16 CPM scorer).
- **Label-file provenance.** Table 1: source URL, byte count, SHA-256 prefix and licence for
  every file used. No pixel data was downloaded for any audited target.
- **Worked-example report.** `pipeline_out/report/RESULTS.md` and `verdict.json`, generated by
  `pipeline/s06_report.py`, with every number traced to its source JSON.
- **Not released.** The `manuscript/` directory contains an earlier draft whose numbers no code
  in the repository has ever produced; it is retained only as a record and is excluded from any
  archive [→ `manuscript/DO_NOT_SUBMIT.md`].

---

## Tables

### Table 1 — Label files used: provenance, size and licence

| file | bytes | sha256 (first 16) | source | licence | sufficient on its own? |
|---|---|---|---|---|---|
| `t2_slice_level_labels.csv` | 760,340 | `d248d41c9915c3fe` | github.com/cai2r/fastMRI_prostate | MIT (repo); no DUA for the CSVs | yes |
| `dwi_slice_level_labels.csv` | 796,852 | `e22a354132cce884` | same | same | yes |
| `DL_info.csv` | 8,479,888 | `a8f57b4b1164c9ed` | HuggingFace `farrell236/DeepLesion` | CC BY-SA 4.0 (mirror); NIH terms on original | yes |
| `knee.csv` | 918,105 | `c1f4a083646cec81` | github.com/microsoft/fastmri-plus | MIT (repo) | **no — needs HDF5 headers** |
| `knee_file_list.csv` | 14,074 | `4b09e5523709815d` | same | MIT (repo) | as above |
| `Annotation_Boxes.csv` | 35,508 | `52752a20f4ec47ea` | TCIA Duke-Breast-Cancer-MRI | CC BY-NC 4.0 | yes, with TCIA series metadata |
| TCIA `getSeries` metadata | 2,894,891 | `fa6b3ee2cc457402` | services.cancerimagingarchive.net | CC BY-NC 4.0 | — |
| `picai_marksheet.csv` | 97,708 | `23eab23790886258` | github.com/DIAGNijmegen/picai_labels | CC BY-NC 4.0 | yes |
| PI-CAI official CV folds ×5 | ~7.5 kB each | — | github.com/DIAGNijmegen/picai_baseline | Apache 2.0 | — |
| `candidates_V2.csv` | 71,374,684 | `2e0f79bbee9a3ba7` | Zenodo 3723295 (LUNA16) | CC BY 4.0 | yes |
| `annotations.csv` (LUNA16) | 136,986 | `db9adb75b381f3e9` | Zenodo 3723295 | CC BY 4.0 | — |

[→ `paper/audit_results.md` §6.]

### Table 2 — Fifteen rows, twelve verdicts

*Scored rows.*

| # | dataset | published number | source of the published number | our best zero-image baseline | trivial fraction [CI] | verdict |
|---|---|---|---|---|---|---|
| 1 | fastMRI Prostate **T2** | 0.861 slice AUROC | Rempe et al. 2024, arXiv:2407.06165, Table II gold standard | 0.854 [0.812, 0.891] positional 20-bin | 0.981 [0.865, 1.084] | **MATCHED** |
| 2 | fastMRI Prostate **T2** | 0.809 slice AUROC | same, PCA ×2 magnitude + phase | 0.854 [0.812, 0.891] | 1.146 [1.011, 1.266] | **MATCHED** (exceeds) |
| 3 | fastMRI Prostate **T2** | 0.714 slice AUROC | same, R = 16 PCA coil combination | 0.854 [0.812, 0.891] | 1.655 [1.459, 1.829] | **MATCHED** (exceeds) |
| 4 | fastMRI Prostate **DWI** | 0.861 slice AUROC | as row 1 | 0.851 [0.816, 0.887] positional 20-bin | 0.973 [0.876, 1.073] | **MATCHED** |
| 5 | fastMRI Prostate **DWI** | 0.809 slice AUROC | as row 2 | 0.851 [0.816, 0.887] | 1.137 [1.023, 1.253] | **MATCHED** (exceeds) |
| 6 | fastMRI Prostate **DWI** | 0.714 slice AUROC | as row 3 | 0.851 [0.816, 0.887] | 1.642 [1.478, 1.810] | **MATCHED** (exceeds) |
| 7 | DeepLesion | 0.905 ± 0.002 8-class accuracy | Yan et al., CVPR 2018, Table 1, triplet + type + location + size | 0.557 [0.524, 0.578] | 0.480 [0.431, 0.511] | PARTIAL |
| 8 | DeepLesion | 0.862 8-class accuracy | same table, multi-scale ImageNet feature | 0.557 [0.524, 0.578] | 0.513 [0.460, 0.546] | PARTIAL |
| 9 | DeepLesion | 0.597 8-class accuracy | same table, **their own** location-feature baseline | 0.557 [0.524, 0.578] | 0.889 [0.799, 0.947] | PARTIAL |
| 10 | LUNA16 (FP-reduction) | >0.95 sensitivity at <1 FP/scan | Setio et al. 2017, arXiv:1612.08012 | **CPM 0.0020**; 0.0006 at 1 FP/scan | ≈ 0 | **NOT MATCHED** |
| 11 | PI-CAI | 0.91 (0.87–0.94) case-level AUROC, AI system | Saha et al., *Lancet Oncol* 2024;25:879-887 | 0.692 [0.626, 0.755] metadata CART | 0.467 [0.307, 0.622] | **NOT MATCHED** |
| 12 | PI-CAI | 0.86 (0.83–0.89) case-level AUROC, 62 radiologists | same | 0.692 [0.626, 0.755] | 0.532 [0.350, 0.708] | **NOT MATCHED** |

Rows 7–9 use the majority class (0.236) as the chance anchor, not 0.5. Rows 11–12 carry the
cohort caveat set out in §3.4.

*Non-comparable rows — audited, no defensible published comparator, no verdict issued.*

| # | dataset | zero-image result | why no verdict |
|---|---|---|---|
| 13 | fastMRI+ knee, meniscus tear per slice | 0.873 [0.858, 0.886] slice; 0.510 [0.428, 0.592] patient | data descriptor, no published slice-level classification number located; also a 199-of-1,173 volume subset |
| 14 | fastMRI+ knee, any annotated finding | 0.801 [0.779, 0.824] slice; 0.558 [0.470, 0.648] patient | as above |
| 15 | Duke Breast Cancer MRI, owner-defined slice task | 0.823 [0.811, 0.834] slice; patient undefined | the data owners define the task but publish no metric |

[→ `paper/audit_results.md` §2.]

### Table 3 — Slice level, patient level, and the remedy, on all eight dataset-arms

| dataset-arm | slice AUROC | patient AUROC | position-stratified slice AUROC |
|---|---|---|---|
| fastMRI Prostate T2 | 0.854 [0.812, 0.891] | 0.506 [0.381, 0.632] | **0.546** (5 strata) |
| fastMRI Prostate DWI | 0.851 [0.816, 0.887] | 0.424 [0.298, 0.547] | **0.539** (6 strata) |
| fastMRI+ knee, meniscus tear | 0.873 [0.858, 0.886] | 0.510 [0.428, 0.592] | — |
| fastMRI+ knee, any finding | 0.801 [0.779, 0.824] | 0.558 [0.470, 0.648] | — |
| Duke breast, owner slice task | 0.823 [0.811, 0.834] | undefined (all patients positive) | — |
| DeepLesion, pelvis vs rest | 0.977 [0.969, 0.984] | 0.954 [0.939, 0.967] | — |
| PI-CAI, case level | not applicable | 0.692 [0.626, 0.755] (metadata) | — |
| LUNA16 candidates | 0.534 [0.513, 0.558] | 0.581 [0.538, 0.613] | — |

[→ `paper/audit_results.md` §4.]

### Table 4 — The protocol: seven rules, each with the failure it was written for

| # | rule | the measured failure behind it |
|---|---|---|
| 1 | Split at the subject level, and state the unit | slice-level CV inflated accuracy by 30–55%, and reached ~96% on randomly labelled data (Yagis 2021) — the one rule here backed by others' numbers |
| 2 | Report patient level as primary | one score vector, two readings: 0.851 slice vs 0.424 patient on a published label file |
| 3 | Subject-clustered intervals, never the slice-level bootstrap | nominal-95% coverage 46.5% vs 91.5%; 3.18× too narrow |
| 4 | Report the zero-image baselines beside every headline | 0.854 against a published 0.861, with no pixels |
| 5 | Publish the positional label distribution and stratify on it | 0.851 → 0.539 when position is held fixed |
| 6 | Test whether metadata predicts the label | release batch predicts breast cancer status at 0.743 against 0.633 for the trained network |
| 7 | Report the trivial fraction, including when it is small | LUNA16's best zero-image baseline is 0.539 [0.520, 0.565]; PI-CAI's positional baseline is exactly 0.500 |

[→ `paper/protocol.md`; the one-page reviewer version is `paper/checklist.md`.]

---

## Figure legends

**Figure 1. What the audit needs.** The four columns a positional null requires — subject
identifier, slice index, label, train/test assignment — shown as they appear in a published
label CSV; the five pixel-blind baselines fitted from them; and the card the tool emits. No
pixels, no data-use agreement for images, no GPU; `numpy` and `pandas` are the entire
dependency list.

**Figure 2. The unit of evaluation.** Slice-level AUROC joined to patient-level AUROC for the
same pixel-blind score vector, on eight dataset-arms, with subject-clustered 95% intervals on
both. Five lines collapse (fastMRI Prostate T2 and DWI, fastMRI+ knee under two label
definitions); one has no patient-level value at all because all 922 Duke breast patients are
positive; two do not collapse (DeepLesion pelvis, whose labels are anatomical regions, and
LUNA16, which is at chance at both units). The two exceptions are drawn in the same panel and
on the same scale as the rest.

**Figure 3. The fastMRI Prostate waterfall.** Eight readings on one axis. W0h, their published
headline 0.861 [0.843, 0.879]; W0, their PCA arm (0.813 magnitude, 0.809 magnitude + phase);
W1, the zero-image positional baseline on their published labels, 0.851 [0.821, 0.880]; W1p,
the same scores read at the patient level, 0.424 [0.298, 0.547]; W2, our reimplementation of
their protocol at their evaluation level, 0.574 / 0.616; W3, the same predictions with a
subject-clustered interval, 0.574 [0.489, 0.667] / 0.616 [0.528, 0.691]; W4, patient level,
0.524 [0.348, 0.690] / 0.528 [0.356, 0.696]; W4s, position held fixed, 0.467 / 0.562.
**W2 does not reproduce W0. This figure is about the evaluation protocol, not about their
model.**

**Figure 4. Trivial fraction across the twelve scored rows.** Forest plot ordered by fraction,
with a reference line at 1.0 and verdict labels. Values above 1 mean the zero-image baseline
exceeded the published number and are shown unclipped. LUNA16's row is its CPM comparison and
is drawn on its own scale, with the scale change marked.

**Figure 5. The remedy.** Raw slice-level AUROC beside position-stratified AUROC for four
score vectors: the zero-image baseline on the T2 and DWI label files (0.854 → 0.546,
0.851 → 0.539) and our two trained arms (0.574 → 0.467, 0.616 → 0.562). Inset: the positional
baseline over a 5/10/20/50 bin sweep and the no-fit centrality score, which uses no training
data at all.

**Figure 6. Worked example: what the input channel predicts when the label is the scanner.**
Phase versus magnitude at predicting an acquisition property on two cohorts with no pathology
in the label: brain, receive-coil count ≥ 16, phase 0.921 [0.870, 0.966] against magnitude
0.913 [0.872, 0.950] on 136 independent test subjects; knee, pulse sequence, 0.999 against
1.000 on 29 paired subjects. **A high value here is the bad result.** These are
acquisition-identity AUROCs and are never drawn on the same axis as a diagnostic AUROC.

**Supplementary Figure S1. Bootstrap coverage.** Coverage of the true AUC (0.6880, known in
closed form) by a nominal 95% interval, over 200 simulated datasets of 20 patients × 15
slices: subject-clustered 91.5% (mean width 0.370), naive slice-level 46.5% (mean width
0.117).

**Supplementary Figure S2. Reconstruction fidelity, worked example.** Correlation between each
reconstructed magnitude slice and the vendor reference shipped in the same file, per cohort.

**Supplementary Figure S3. Positional label distributions.** Label rate against relative slice
position, training rows only, one panel per audited arm — the artefact Rule 5 asks every
benchmark to publish.

---

## References

*To be completed in the target journal's style. The works that must appear, with the exact
claims they are cited for, are enumerated in `paper/audit_targets.md` §3.1 and reproduced in
§1 above: Geirhos 2020; Badgeley 2019; DeGrave 2021; Oakden-Rayner 2020; Ong Ly 2024; Lin
2024; Yagis 2021; Tampu 2022; Wen 2020; Kapoor & Narayanan 2023; Varoquaux & Cheplygina 2022;
Roberts 2021; Yan et al. CVPR 2018; Rempe et al. arXiv:2407.06165; Tibrewala et al.
arXiv:2304.09254 (fastMRI Prostate); Zhao et al. Sci Data 2022 (fastMRI+); Saha et al. Lancet
Oncol 2024 (PI-CAI); Setio et al. arXiv:1612.08012 (LUNA16); Saha/Mazurowski TCIA
Duke-Breast-Cancer-MRI.*

**Before submission the prior-art search must be redone against Google Scholar, the
MICCAI/MIDL/ML4H proceedings and the RSNA ICH Kaggle solution write-ups.** Yan et al. 2018 was
found only during the audit itself; the search recorded in `paper/audit_targets.md` §3.4 was a
handful of web queries and is not sufficient for a paper whose novelty claim rests on absence.
