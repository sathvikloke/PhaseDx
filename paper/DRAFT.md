# What a slice-level benchmark certifies without the pixels: a label-file audit of seven public benchmarks, and a pre-registered screen of how often the check is reported

**Draft 1, 2026-07-29. Abstract, Methods §2.10, Results §3.10 and Limitations revised
2026-07-29 (evening) after the prevalence screen and the revised benchmark audit.**

> **Provenance markers.** Every number in this draft carries a marker of the form
> `[→ file]` naming the artefact it was read from. They are for internal verification and
> must be stripped before submission. If a number has no marker, it must not be in the
> paper. Framing, venue reasoning and the reviewer-attack analysis live in
> `paper/PAPER_PLAN.md`; the seven-rule protocol in `paper/protocol.md`; the one-page
> checklist in `paper/checklist.md`; the full audit record in `paper/audit_results.md`;
> **every established number with its interval and its source file in
> `paper/FINDINGS.md`, which now governs.**

> **REVISION NOTICE — read before editing any section below.** The Abstract, §2.5, §3.1,
> §3.2, §3.3, §4.1, §4.2, §2.10, §3.10, §5, Tables 2–3 and Tables 5–6 are **current** as of
> the 2026-07-29 evening re-anchoring pass. Sections 3.4–3.9 and the figure legends were
> written against the *six*-benchmark audit and have **not yet been brought forward**; three
> things changed and must be propagated: (a) **RSNA 2019 Intracranial Haemorrhage was
> reached**, recomputed on all 18,938 patients, and is now the flagship — slice AUROC 0.737
> [0.735, 0.740] collapsing to 0.453 [0.445, 0.461] at the patient level on the same score
> vector, on all six of its labels — displacing fastMRI Prostate, which is now a *worked
> example against a preprint*; (b) the audit is **seven benchmarks, eight label files, 24
> scored (benchmark, comparator) rows**, not six/seven/15/12; (c) the categorical verdict
> count has been **replaced as the headline by the distribution of the trivial fraction**,
> and the verdict retained as a secondary column.

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
**seven** public benchmarks across **eight** label files: RSNA 2019 Intracranial Haemorrhage,
fastMRI Prostate (T2 and DWI), DeepLesion, fastMRI+ knee, Duke Breast Cancer MRI, PI-CAI and
LUNA16. Verdict rules and the comparison statistic were fixed before any result. Separately,
we ran a **pre-registered prevalence screen**: from a frozen 9,979-record PubMed frame of
volumetric-imaging classification papers we drew a seeded random permutation and coded
positions 1–100 against a frozen extraction form, with four screeners working independently
and a 15-paper overlap set for agreement. We release the implementation as `trivialbaselines`
(MIT; `numpy` and `pandas` only), a seven-rule reporting protocol with a one-page checklist,
and the screen's protocol, codebook, sealed screener files and analysis script.

**Results — audit.** The primary result requires no published comparator. On RSNA ICH —
752,802 slices from **18,938 patients**, the benchmark whose official metric is per-slice —
one pixel-blind score vector, a 20-bin estimate of P(label | relative slice position) fitted
on the training slices of a subject-disjoint five-fold split, reaches slice AUROC **0.737
[0.735, 0.740]** and **0.453 [0.445, 0.461] at the patient level**, an interval lying wholly
below chance; aggregating by the patient's most suspicious slice instead gives 0.500. Nothing
changes between the two readings except the unit. All six official labels behave the same way
(gaps 0.205–0.307), and the result was recomputed by a second implementation sharing no code
with the first. The same divergence appears on fastMRI Prostate (0.854 [0.812, 0.891] slice /
0.506 [0.381, 0.632] patient for T2; 0.851 [0.816, 0.887] / 0.424 [0.298, 0.547] for DWI), on
fastMRI+ knee (0.873 [0.858, 0.886] / 0.510 [0.428, 0.592]) and, in a fourth form, on Duke
breast, where the slice-level baseline reaches 0.823 [0.811, 0.834] and the patient-level
AUROC is undefined because all 922 patients are positive. It is not universal: on DeepLesion,
whose labels are anatomical regions, the positional model is high at both units (0.977 / 0.954
for pelvis), and on LUNA16 it is at chance at both (0.534 / 0.581). Against published numbers
we report the trivial fraction — the share of a published margin over chance that a
pixel-blind model reaches — as a **distribution over 24 (benchmark, comparator) rows** rather
than as a verdict count. Against peer-reviewed comparators, taking each benchmark-arm's
strongest published system, the median is **0.469 (IQR 0.437–0.490, range −0.002 to 0.613,
n = 9)**; the six RSNA ICH subtype rows against a peer-reviewed slice-level comparator on the
identical cohort span **0.395–0.613**. Two benchmarks do not fire, and are reported at equal
prominence: LUNA16 scores CPM 0.0020 on the challenge's own metric — *below* its random-score
reference of 0.0027 — for a trivial fraction of −0.002, and PI-CAI's positional baseline is
exactly 0.500, the correct registration of "inapplicable". The only rows reaching a fraction
of 1 are six rows from one benchmark whose comparator is an arXiv preprint unreviewed two
years after posting; **no benchmark with a peer-reviewed comparator is matched**, and that
row-set is presented as a worked example, labelled as such, not as the headline. Stratifying
the slice-level statistic within bins of relative position collapses those nulls to 0.546 and
0.539.

**Results — does the unit change which method wins?** We tested the stronger claim on 21
method configurations across five cohorts with one shared subject-clustered bootstrap, comparing
between-unit rank disagreement against the within-unit disagreement produced by resampling
subjects with the unit held fixed. It does not clear that bar: of **447** method pairs, 204
point in opposite directions at the two units and **none** survives a Holm correction, and in no
cohort does between-unit disagreement exceed both within-unit noise floors. On three of five
cohorts two disjoint halves of the same subjects rank the same methods in opposite orders
(split-half Kendall τ = −0.42, −0.24 and −1.00 at the patient level), so **these benchmarks
cannot rank methods at either unit** — a null we report as a result. One narrower unit effect is
supported: on the confound cohort, where methods do separate, aggregating from slice to patient
shifts the paired magnitude-versus-phase contrast by **+0.028 AUROC [0.015, 0.041], p = 0.001**,
in the same direction for all six architectures, reversing its sign for two of them, though
neither ordering is individually distinguishable from a tie. In the published literature —
2,934 open-access full texts scanned, plus challenge leaderboards — rank inversions between
units are real and change reported winners (CAMELYON16's own two boards, τ = 0.754; a 13-method
reproduction of them in which the winner changes, τ_a = 0.603), but only two published cases
report intervals at both units and in both the inversion lies inside them.

**Results — prevalence.** The pre-registered extension rule was run to its stopping point. Of
250 sampled papers, 115 were excluded, **44 (32.6% [25.3%, 40.9%] of the eligible-looking set)
could not be obtained in full text**, and 91 were included and readable against a
pre-registered target of 75. **None of the 91 reported a measured zero-image baseline of any
kind** (0/91, 0.0% [0.0%, 4.1%]; unconditional bounding interval over the unreachable papers
**[0.0%, 32.6%]**, which the protocol makes the headline because censoring exceeded 15%), and
**no such baseline appears in any of 345 coded records covering 300 distinct sampled papers**,
including the excluded and unreachable ones. Enlarging the sample did not reduce the censoring
and could not: four separately screened blocks returned unreachable rates of 35.6%, 34.6%,
22.7% and 32.1%. **One of 91 reported the positional distribution of its labels** (1.1%
[0.2%, 6.0%]). Forty-four percent evaluated at a unit below the patient (44.0% [34.2%, 54.2%])
and 85 of 91 reported exactly one unit; of the nineteen reporting any slice-level metric, six
also reported a patient-level one (31.6% [15.4%, 54.0%]). The only non-imaging comparators found
in the readable set were five clinical-variables arms (5.5% [2.4%, 12.2%]). Agreement on the
primary flag **failed its pre-specified threshold as originally sealed** (raw 65.6%, Fleiss'
κ −0.015 [−0.164, 0.120] against a 90% / 0.60 floor); the failure is attributable to the
extraction form having no level for "could not be assessed", and on the six overlap papers where
the flag is defined all four screeners agreed unanimously and negatively. The pre-registered
remedy — adjudication, codebook amendment, re-code of every coded record — was executed, and the
amended encoding of the same four sealed files gives raw 95.6%, Fleiss' κ 0.932 [0.777, 1.000],
meeting both floors; that is a re-encoding rather than an independent re-rating, and we say so.

**Conclusion.** Slice-level and patient-level readings of the same pixel-blind score vector
disagree by 0.20–0.43 AUROC on four of seven benchmarks, and on the largest of them — 18,938
patients, no published number involved — the slice-level reading looks like a working detector
while the patient-level reading is below chance. Whether the unit also changes *which method
wins* we could not establish, and neither can any published two-unit comparison we located: at
realistic cohort sizes the ranking does not reproduce across two halves of the same subjects at
either unit, which means architecture choices are being justified by orderings that are not
identified. Against peer-reviewed published numbers, on
the same metric and the same unit, a pixel-blind model reaches a median of 47% of the reported
margin over chance, and on two benchmarks it reaches none of it. A published slice-level
protocol can certify a number that such a model also reaches outright, though the only
benchmark where that match is demonstrated has a preprint comparator and is reported here as a
worked example rather than as evidence. In a pre-registered random sample of the literature
that produces such numbers, none of these checks was reported in any of the papers that could
be read. All of it is checkable by a third party in one command, from a label file, without a
data-use agreement for pixels and without a GPU.

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
   combination — applied uniformly, with identical reporting, to seven public benchmarks
   across eight label files.
2. **A measurement of how much of a published number survives without the pixels, reported as
   a distribution rather than as a verdict.** Twenty-four (benchmark, published comparator)
   rows; against peer-reviewed comparators the median share of the published margin over
   chance that a pixel-blind model reaches is 0.469 (IQR 0.437–0.490, n = 9 benchmark-arms).
   **An audit that reports its failures at the same prominence as its successes.** Two
   benchmarks refused the null outright — LUNA16 at a fraction of −0.002 on the challenge's
   own metric, and PI-CAI, which is presented as a benchmark that already evaluates at the
   unit it should. The only rows reaching a fraction of 1 come from one benchmark whose
   comparator is an unreviewed preprint, and they are labelled as such wherever they appear.
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

### 2.5 The comparison statistic, reported as a distribution; the verdict rule, kept as a secondary summary

```
trivial fraction = (best zero-image baseline − chance) / (published − chance)
```

with chance = 0.5 for AUROC and the majority-class rate for multi-class accuracy. **It is a
continuous quantity and we report it as one**: for every (benchmark, published comparator)
pair, with an interval, and summarised as a distribution
[→ `paper/trivial_fraction_distribution.{json,md}`]. Rows are not independent — a single
paper's table can supply several comparator systems for one benchmark — so the primary
summary takes each benchmark-arm's **strongest** published system, which is the conservative
choice because a stronger comparator enlarges the denominator and shrinks the fraction.

The categorical verdict is retained as a **secondary** column, unchanged, so this analysis
can be reconciled with the earlier one rather than replacing it silently
[→ `paper/audit_results.md` §1]:

* **MATCHED** — the upper bound of the baseline's clustered 95% interval reaches or exceeds
  the published number.
* **PARTIAL** — trivial fraction ≥ 0.30 with its interval wholly below 1.
* **NOT MATCHED** — trivial fraction < 0.30, or the baseline is statistically
  indistinguishable from chance.
* **NON-COMPARABLE** — the published number is on a different cohort, split, label definition
  or metric and could not be reconstructed; no verdict is issued.

The rule is a descriptive decision rule, not a hypothesis test, and no *p*-value is claimed
from it. We demote it for a reason we can exhibit: applied mechanically it returns PARTIAL on
the two PI-CAI rows, where an earlier hand-assignment recorded NOT MATCHED on the strength of
a cohort caveat the rule has no slot for. The verdict flipped; the fraction (0.467, 0.532) did
not move. Both are reported [→ `paper/audit_results.md` §1, §2.1].

**Behaviour of the statistic at its extremes**, exercised on the implementation rather than
asserted [→ `paper/trivial_fraction_distribution.md`, "Behaviour at the extremes"]. The
fraction is **undefined**, and returned as such with a reason string, when the published
number is at or below chance. Values above 1 — the baseline exceeded the published number —
are left unclipped in the reported value, with a clipped copy kept only for plotting. A
baseline below chance yields a negative value, which is likewise kept. The statistic's one
real fragility is a small denominator: at a published number 0.021 above chance it returns
4.76, arithmetically correct and practically meaningless. The implementation guards this with
a minimum-headroom threshold of 0.02, which is arbitrary; in this audit the guard is never
load-bearing, because the smallest denominator across all 24 rows is 0.214. A trivial fraction
against a published number near chance should be replaced by the two margins themselves.

Its interval propagates uncertainty in the **baseline only**; the published number enters as a
fixed constant, because a publication's sampling distribution is almost never available. On
the RSNA ICH rows, where the baseline interval is ±0.002, this means essentially all of the
real uncertainty in the fraction is uncertainty about the published constant and none of it is
displayed. The statistic is **not a decomposition**: baseline and published model may exploit
the same shortcut, different shortcuts or overlapping ones, and no fraction distinguishes
those cases.

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

### 2.10 The prevalence screen

The audit establishes that a benchmark *can* be reached without pixels. It says nothing about
whether anyone checks. We therefore ran a separate, pre-registered screen of the literature
that produces slice-level numbers. Protocol and codebook were frozen before any
analysis-sample record was read, and both are released
[→ `paper/screen_protocol.md`, `paper/screen_frame.json`].

**Frame.** A single PubMed query for deep-learning classification papers on volumetric
modalities reporting a numeric classification metric, 2019-01-01 to 2026-12-31, English,
excluding reviews and case reports. Executed 2026-07-29 via E-utilities; 9,979 records
retrieved and hashed [→ `paper/screen/frame_meta.json`, `frame_sha256 = d611def0…`]. The
frozen list governs; the query re-run later will drift and that drift is reported, not
absorbed.

**Sample.** A seeded permutation of the frame [→ `paper/screen/permutation.txt`]. Positions
1–100 were screened. Ten pilot records at positions 101–110 were used to fix the codebook and
are permanently excluded from the analysis sample because the protocol author read them. The
protocol fixes a target of **75 included papers** and an extension rule — continue into the
reserve in permutation order, in blocks of 50, until 75 or position 400 — chosen because at
n = 75 a zero count carries a Wilson upper bound of 4.9%, which is the precision at which
"fewer than 1 in 20" can be claimed. **The extension was triggered and executed**: reserve
blocks at positions 111–160, 161–210 and 211–260 were screened in permutation order and none
was truncated part-way, and the rule stopped at position 260, where the running total of
included papers first exceeded 75. A fourth reserve block at positions 261–310 was screened
after that stopping point and is reported throughout as a labelled post-hoc extension, never
pooled into the pre-registered denominator, because continuing past a stopping rule that has
already fired is a data-dependent continuation.

**Extraction.** Two stages: title-and-abstract, then a Methods-in-full read with the Results
tables, any baseline or statistical-analysis subsection, and the supplement. Every coded field
that bears on a result requires a verbatim quote with its location; a code without a quote is
not submitted. Before a paper may be coded as reporting no trivial baseline, the screener must
run fourteen prescribed full-text searches (*baseline, chance, random, majority, prevalence,
constant, trivial, metadata, clinical-only, clinical model, position, location, slice index,
permut*) and record that they were run. **A negative without the search is invalid.** Absence
of a statement is recorded as absence, never inferred into a positive.

**Primary endpoint (P1).** The proportion of included papers reporting at least one
**zero-image** baseline with a measured value on the same metric: a constant / majority-class
/ prevalence predictor, a positional model, an acquisition-metadata model, or a permuted-label
null. A clinical-variables-only arm is deliberately **excluded** from P1 and reported
separately (S1), because a clinical nomogram tests whether imaging adds to clinical data and
not whether the benchmark is solvable without information. An assertion with no number — "an
AUC of 0.5 represents chance" — counts as negative and is recorded in its own field.
Secondary endpoints cover the evaluation unit, the split unit, the positional distribution of
labels, the unreachable count, interval practice and positive-count reporting.

**Unreachable papers are never dropped.** An access ladder is followed in order — PMC or
publisher open access, publisher site, institutional subscription, repository or preprint,
interlibrary loan or author request with a 21-day wait — and the rung reached is recorded.
Sci-Hub and equivalent sources were not used. A paper whose abstract is consistent with
inclusion but whose full text could not be obtained is coded `unreachable`, enters the flow
diagram, and enters **two bounding analyses that are reported unconditionally**: a lower bound
with every unreachable paper coded as not reporting a baseline, and an upper bound with every
one coded as reporting one. Where unreachable papers exceed 15% of the eligible set, the
protocol makes the **bounding interval the headline**. The direction of the bias was stated
before the result: paywalled articles skew toward higher-tier clinical journals, the venues
most likely to require a comparator arm, so silently dropping them would push P1 downward — in
the direction that flatters our own hypothesis. That is why the bounds are unconditional
rather than contingent on the missingness looking bad.

**Agreement.** Fifteen papers at permutation positions 1–15 were coded independently by all
four screeners, each submitting a sealed file before seeing any other. Fleiss' κ across four
raters is the primary statistic; the six pairwise Cohen's κ are reported alongside it. Because
the primary flag was expected to be extremely skewed, under which κ collapses toward zero even
at near-perfect agreement, **raw percent agreement and Gwet's AC1 were pre-specified in the
same table**, before any coding. Intervals are bootstrap percentile 95% over 2,000 resamples
of the fifteen papers, seed 20260729. A remedy was pre-specified for κ below 0.60 or, where
the skew guard applies, raw agreement below 90%: a documented adjudication round, a codebook
amendment, and re-coding of every already-coded paper. That remedy fired and was executed; the
adjudication round produced fourteen decision rules, the sealed files were never modified, and
the re-code is an analysis-time overlay recorded separately
[→ `paper/screen_adjudication.md`, `paper/screen_recoded.json`]. Reserve-block records are
coded by a single screener and therefore contribute no agreement information.

**Pooling.** The overlap set is de-duplicated by majority of the four independent codes. Ties
are broken **against our own hypothesis** — toward the code that flatters the literature — so
that no reported number can be an artefact of an arbitrary tie-break. Five ties occurred. A
paper appearing in more than one block would be counted once; none did. All proportions carry
Wilson score 95% intervals, fixed for every proportion in the paper so that no interval method
is chosen after seeing a result. No significance test is pre-specified; this is an estimation
study. The pooling script is released [→ `paper/screen/analysis/pool_final.py`], and it writes
every number reported below [→ `paper/screen/analysis/pooled_final.json`].

---

## 3. Results

### 3.1 The audit as a distribution, stated before any individual row

Seven benchmarks were audited on eight label files, producing **24 (benchmark, published
comparator) rows** — more rows than benchmarks because a single paper's table can supply
several comparator systems for one benchmark. We report the trivial fraction across all of
them as a distribution rather than as a count of verdicts, because the verdict discards
exactly the information the continuous statistic exists to carry: a benchmark at 0.61 and one
at 0.31 are both "PARTIAL" [→ `paper/trivial_fraction_distribution.{json,md}`;
`paper/audit_results.md` §0.2, §2.0] (Table 2, Figure 4).

| set of rows | n | min | Q1 | **median** | Q3 | max |
|---|---|---|---|---|---|---|
| **peer-reviewed comparator, strongest system per benchmark-arm** | 9 | −0.002 | 0.437 | **0.469** | 0.490 | 0.613 |
| peer-reviewed comparator, all rows | 18 | −0.002 | 0.455 | 0.485 | 0.514 | 0.889 |
| strongest system per benchmark-arm, any comparator | 11 | −0.002 | 0.452 | 0.480 | 0.562 | 0.981 |
| all rows | 24 | −0.002 | 0.469 | 0.512 | 0.910 | 1.655 |
| preprint comparator only (§3.2) | 6 | 0.973 | 1.020 | 1.142 | 1.518 | 1.655 |

**Against peer-reviewed numbers, on the same metric and the same evaluation unit, the median
share of the published margin over chance that a pixel-blind model reaches is 0.469.** The
distribution is tight, not diffuse: eight of those nine rows lie between 0.395 and 0.613, and
the ninth is LUNA16 at −0.002. The six RSNA ICH subtype rows — a peer-reviewed slice-level
comparator on the identical cohort, metric and unit — span 0.395, 0.437, 0.469, 0.490, 0.510,
0.613 against the stronger of the two systems in that table, and 0.410–0.615 against the
weaker one. Roughly **40–60% of a published margin over chance, per subtype, in peer-reviewed
work, with no pixels.**

The categorical verdict is retained as a secondary summary and is reported here so that
nothing is hidden: six MATCHED, seventeen PARTIAL, one NOT MATCHED across all rows; **zero
MATCHED once the preprint comparator is removed**. That is why the count cannot be the
headline. The general claim that trivial baselines *match* published performance on medical
imaging benchmarks is **not** supported by these data, and is not supported against the
peer-reviewed literature at all. What is supported is quantitative, and it is the table above.

### 3.2 Worked example, against a preprint: fastMRI Prostate, a published slice-level protocol matched without pixels

**This subsection is a worked example, and its comparator has not been peer-reviewed.** Rempe
et al. was re-queried on the arXiv API on 2026-07-29: no journal reference, no DOI, no Europe
PMC record, two years after posting [→ `paper/audit_results.md` §2.3]. It supplies the only
six rows in the audit that reach a trivial fraction of 1, and it is presented here because it
is the clearest illustration of what a matched row looks like — not as evidence for a general
claim. Every sentence about it must carry the preprint label.

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

### 3.3 The unit of evaluation: the principal result, and it requires no published number

Every cell in this section is our own computation on a published label file. No published
number enters, so none of the comparability objections that can be raised against §3.2 apply,
and no comparator's peer-review status can reach it.

**RSNA 2019 Intracranial Haemorrhage, the official training file, all 18,938 patients.** The
benchmark's own official metric is per-slice. We fit a 20-bin estimate of P(label | relative
slice position) on the training slices of a subject-disjoint five-fold split, apply it out of
fold, and read the resulting score vector at three units. Intervals are 95% percentile
intervals from 2,000 bootstrap replicates resampling **patients**, so a patient drawn twice
contributes all of their slices twice
[→ `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json`;
`pipeline_out/audit_logs/rsna_ich_unit_collapse.log`].

| label | slice prevalence | patient prevalence | **slice** AUROC | **patient** AUROC (mean) | patient (max) | gap |
|---|---|---|---|---|---|---|
| **any haemorrhage** | 0.143 | 0.404 | **0.737** [0.735, 0.740] | **0.453** [0.445, 0.461] | 0.500 | **0.284** |
| epidural | 0.004 | 0.016 | 0.712 [0.700, 0.725] | 0.492 [0.461, 0.524] | 0.486 | 0.220 |
| intraparenchymal | 0.048 | 0.247 | 0.751 [0.747, 0.755] | 0.480 [0.471, 0.490] | 0.503 | 0.271 |
| intraventricular | 0.035 | 0.174 | 0.805 [0.802, 0.808] | 0.497 [0.487, 0.508] | 0.495 | 0.307 |
| subarachnoid | 0.047 | 0.182 | 0.690 [0.686, 0.695] | 0.485 [0.475, 0.496] | 0.505 | 0.205 |
| subdural | 0.063 | 0.178 | 0.720 [0.717, 0.723] | 0.476 [0.466, 0.487] | 0.500 | 0.243 |

752,802 slices, 21,744 series, 18,938 patients. **Nothing changes between the slice and
patient columns except the unit at which the ranking is performed, and on the `any` label the
difference is 0.284 AUROC.** Four of the six patient-level intervals lie wholly below chance;
none reaches 0.55. The max-aggregated column — score each patient by their most suspicious
slice, which is what a triage system would do — is 0.486–0.505 throughout.

Three controls run on the same path. The positional baseline's own permutation null, with
labels shuffled *within each series* so that prevalence, subject clustering and stack depth
all survive, sits at 0.502 at the slice level, so the excess is 0.236; at the patient level
that null is 0.523, meaning the observed 0.453 is not merely below chance but 0.070 below what
the same estimator reaches when there is nothing to find. The constant predictor scores 0.492
slice / 0.501 patient here and 0.498 / 0.500 in the released tool's own full-cohort card — on
the 1,500-patient subsample it scored 0.487, which triggered a protocol warning about pooling
out of fold across folds of differing training prevalence; on the full cohort that deviation
falls to 0.002 and the warning does not fire. And the naive slice-resampled interval, which we
compute and refuse to use, is 1.5–2.0× too narrow on these six rows.

**The number was verified, not carried forward.** It was first produced by the audit harness
on a seeded 1,500-patient subsample. The table above comes from a second implementation that
shares no code with it — its own fold assignment and seed, its own binning, `scikit-learn`'s
AUROC in place of ours, its own clustered bootstrap. Four routes were run:

| route | cohort | slice | patient |
|---|---|---|---|
| audit harness, five-fold subject CV | seeded 1,500-patient subsample | 0.7313 [0.723, 0.739] | 0.4616 [0.431, 0.491] |
| independent implementation, different folds | same subsample | 0.7311 [0.723, 0.739] | 0.4580 [0.420, 0.486] |
| **independent implementation** (the table above) | **all 18,938 patients** | **0.7374 [0.7351, 0.7398]** | **0.4533 [0.4454, 0.4613]** |
| **audit harness** (the released tool) | **all 18,938 patients** | **0.7376 [0.7352, 0.7399]** | **0.4561 [0.4478, 0.4640]** |
| replication at the published paper's own split geometry, 200 re-draws | all 18,938 patients | 0.7381 [0.727, 0.750] | — |

Four routes, one answer, agreeing to 0.003 AUROC at both units. The released tool's
full-cohort card supplies three further checks: the effect is not a binning artefact (slice
AUROC 0.716 / 0.733 / 0.738 / 0.745 over 5 / 10 / 20 / 50 bins, and a fit-free centrality
score −|relative position − 0.5|, which uses no training data at all, reaches 0.735); the null
model is not itself overfitting (apparent training AUROC 0.7379 against a held-out 0.7376);
and acquisition metadata adds nothing here (the combined position + metadata tree reaches
0.718 against its own permutation null of 0.718, an excess of exactly zero)
[→ `pipeline_out/trivial_baselines/rsna_ich_any_slice_full.json`].

**One qualification travels with the patient-level number and must be quoted with it.** The
collapse is bin-robust at the slice level but not at the patient level, where the full-cohort
sweep runs 0.437 / 0.445 / 0.456 / 0.632 over 5 / 10 / 20 / 50 bins. At 50 bins over volumes
of 20–60 slices the bin index is nearly the raw slice index, so the per-patient aggregate
begins tracking volume length, which is itself weakly predictive here (0.591 patient AUROC
alone). The positional signal collapses at the patient level; a separate and weaker
volume-size signal does not, and the 50-bin patient number is measuring the latter. Twenty
bins is the pre-specified setting throughout.

The same divergence appears on every other benchmark that has a slice axis and a patient
axis:

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

[→ `paper/audit_results.md` §4; the cards in `pipeline_out/trivial_baselines/`.]

Duke breast is a further form of the same protocol problem: a slice-level number is computable
and a patient-level number is not, because every patient in that cohort has cancer, and the
harness reports it as unavailable rather than inventing a value.

**Two benchmarks do not show the effect, and both are reported here rather than in a
supplement.** DeepLesion does not collapse, and should not: its labels are anatomical
regions, so they are patient-level facts about where lesions were found, and position
predicts them at both units. LUNA16 is at chance at both. Stating all the outcomes is what
makes the rest credible (Figure 2).

**Why this is the paper's principal result rather than §3.2.** It rests on no published
number, so no reproduction dispute and no comparator's peer-review status touches it. Its
cohort is 18,938 patients — roughly 400× the 46-patient test arm that carried the claim
before. It holds on all six labels of the benchmark rather than one convenient arm. And it is
the only result in the paper computed twice by two implementations sharing no code.

### 3.3bis Does the unit change *which method wins*? A pre-specified test, and a null

§3.3 shows the unit changes the *number*. The claim with consequences is that it changes the
*ordering*: if a slice-level benchmark ranks A above B and a patient-level benchmark ranks B
above A, a slice-level benchmark is not merely optimistic, it selects the wrong method. We
tested that directly, on the only material where we hold every method's per-slice predictions:
our own architecture zoo of seven architectures × three input conditions (magnitude, phase,
both), giving 3–21 method configurations per cohort depending on which cells were run, each
method scored on identical slices of identical subjects, all read from one shared
subject-clustered bootstrap of 2,000 replicates
[→ `pipeline/s16_rankinversion.py` (65/65 self-tests); `pipeline_out/rankinversion.json`;
full analysis in `paper/rank_inversion.md`].

**The test is designed around the obvious trap.** A ranking computed on noisy estimates will
differ between two units by chance, so "slice-rank ≠ patient-rank" proves nothing. We
therefore measure rank disagreement as `D = 1 − τ_b` and compare the *between-unit*
disagreement against the *within-unit* disagreement produced by resampling subjects with the
unit held fixed, paired inside each bootstrap replicate. The unit is credited only if the
paired difference excludes zero against **both** noise floors. Named inversion pairs must in
addition have both orderings individually excluded from zero after Holm adjustment over every
pair examined at that unit. Thresholds (τ = 0.50, top-1 reproducibility = 0.50) were set in the
file before the cohorts were run.

**The answer is a null, and it is a stronger null than "the rankings agree".**

| cohort | methods | subjects | ρ(slice, patient) | between-unit `D` | δ vs slice-level noise floor | δ vs patient-level noise floor | split-half τ, slice / patient | verdict |
|---|---|---|---|---|---|---|---|---|
| brain (confound) | 13 | 136 | 0.852 [0.748, 0.989] | 0.282 | −0.018 [−0.256, 0.231] | +0.026 [−0.333, 0.282] | 0.58 / 0.66 | cannot name a winner |
| prostate T2 | 21 | 67 | −0.311 [−0.356, 0.635] | 1.173 | +0.514 [0.010, 0.962] | +0.175 [−0.334, 0.655] | 0.21 / **−0.42** | cannot rank |
| prostate DWI | 18 | 45 | 0.067 [−0.152, 0.790] | 0.928 | +0.259 [−0.405, 0.804] | +0.131 [−0.431, 0.667] | −0.03 / **−0.24** | cannot rank |
| breast | 3 | 70 | 0.500 [−0.500, 1.000] | 0.667 | 0.000 [−2.000, 0.667] | −0.667 [−2.000, 1.333] | −0.33 / **−1.00** | cannot rank |
| knee | 3 | 29 | undefined (patient AUROC = 1.000 for all three) | — | — | — | — | no estimate |

**In no cohort does between-unit disagreement exceed within-unit resampling noise at both
units.** Of **447** method pairs examined, 204 point in opposite directions at the two units
and **none** survives the inversion test. The most quotable numbers in the table are the least
trustworthy: prostate T2 and prostate DWI share *none* of their top-three methods between
units, and prostate T2's rank correlation is negative — but two disjoint halves of the same
subjects rank those methods at split-half τ = 0.21 (slice) and **−0.42** (patient), so the two
units disagree because at least one of them is noise, not because they encode two orderings.
On prostate T2 the between-unit disagreement does clear the *slice*-level floor (δ = +0.514
[0.010, 0.962]) and not the patient-level one; that asymmetry is the signature of a ranking
with no ordering information, and all 21 patient-level AUROCs there lie between 0.381 and
0.539. The honest statement is not that the unit changes the ranking. It is that **a benchmark
of this size cannot rank methods at either unit** — and that is a reportable result, because it
is the state most published two-unit comparisons are in.

**One unit effect is supported, and it is narrower than a rank inversion.** On the brain
cohort — the only one where methods separate at all, AUROC 0.52–0.96 — the paired
unit × condition interaction `I = (AUC_agg,mag − AUC_agg,phase) − (AUC_slice,mag −
AUC_slice,phase)` is **+0.028 [0.015, 0.041], p = 0.001**, positive for all six architectures
and individually Holm-supported for four. Aggregating to the patient shifts the
magnitude-versus-phase verdict towards magnitude, because magnitude scores carry per-slice
noise that averages out within a subject while phase scores are already close to a
subject-level property (mean aggregation gain +0.045 for magnitude cells against +0.017 for
phase cells). For two architectures the shift crosses zero — `densenet121/imagenet` (phase
ahead by 0.004 at slice level, magnitude ahead by 0.023 at patient level) and
`resnet18/imagenet` (0.006 / 0.013) — so the two units return literally opposite answers to
this paper's own question. But neither ordering is individually distinguishable from a tie at
either unit, so what is established is a reproducible shift in the estimate, not a
demonstration that a slice-level benchmark would confidently select a different method. The
same interaction is unsupported on all four other cohorts (0 of 7, 0 of 6, 0 of 1, and one
cell excluded for saturating the aggregated unit).

**Published two-unit comparisons reach the same verdict, from other people's tables.** We
searched for published rankings of two or more methods reported at both a fine and a coarse
unit: 3,462 unique open-access records harvested and 2,934 full texts scanned through the
Europe PMC API, plus the arXiv API and direct fetches of challenge leaderboards
[→ `paper/published_inversions.json`, `paper/published_inversions_round2.json`]. Inversions
exist and they change reported winners. CAMELYON16's organisers scored the same 32 submissions
at two units and published both boards: Kendall τ = 0.754, 61 of 496 pairs discordant, and two
methods from the same group reverse order (3rd by slide AUC / 6th by lesion FROC against 8th /
3rd). Ruan et al. reproduce 13 of those methods in a single table where the winner itself
changes (τ_a = 0.603, best by AUC 0.9935 against best by FROC 0.8533, which is 3rd by AUC), and
explain the mechanism themselves: the method that wins at the fine unit is the one trained at
the fine unit. Islam et al. report six architectures on RSNA-STR PE at image and exam level
(τ_a = 0.467) and write that the optimal architecture differs between the two. Two further
cases invert on accuracy at slice and patient level (τ = −0.600 and 0.300).

**But only two published cases report intervals at both units, and in both the inversion sits
inside them.** Jarkman et al. publish 95% CIs for four model variants on CAMELYON16 where the
AUC ordering is almost exactly the reverse of the FROC ordering — and the extremes are
0.988 [0.965–1.000] against 0.969 [0.926–0.998] on AUC, 0.838 [0.757–0.913] against
0.817 [0.730–0.896] on FROC; on the same paper's own local cohort the same four models at the
same two units agree perfectly. Guo et al. publish CIs for six entries at τ_a = 0.133, where a
claim to beat the CAMELYON16 champion holds only at the lesion unit and the intervals overlap
heavily. A strong counter-example exists too: eleven methods on CAMELYON16 at both units with
τ_a = 0.927 and the same winner at both. And the ranking-instability literature sets the bar
we must clear — Maier-Hein et al. show that τ in the 0.74–0.85 range is already enough for
"critical changes in the ranking", which is exactly where CAMELYON16's between-unit τ of 0.754
sits. Taken with our own null, the defensible claim is that **published orderings do change
with the unit, and no published two-unit comparison — including ours — has yet shown that the
change exceeds sampling noise.**

**Two-unit reporting is also rare enough to count.** Of 2,642 full texts scanned for a
fine-unit and a coarse-unit column on the same metric, 18 had both and only **1** carried two
or more comparable model rows; of 350 papers mentioning CAMELYON, PI-CAI or LUNA16, only **7**
tabulated both units for three or more methods. For most published models the ranking at the
other unit is not under-reported but unrecoverable.

**This result is reported because it is what the analysis returned.** It is also the analysis
whose weakness we most want a reader to see: our "methods" are configurations we trained
ourselves under one pipeline and one seed, not independently published systems (§5).

### 3.4 Two benchmarks that resist the null — reported at the same prominence as the hits

A measure that returns a large number on every benchmark measures nothing. These two rows are
the only evidence in the paper that the trivial fraction discriminates, and they carry the
same weight as §3.1's distribution. **LUNA16's trivial fraction is −0.002 and PI-CAI's
positional baseline is exactly 0.500** — the single value at the floor and the single value
registering "inapplicable", out of the nine peer-reviewed head-to-heads whose other eight lie
in 0.395–0.613 [→ `paper/audit_results.md` §4bis.1].

Two negatives out of nine rows is an **existence proof that the measure can fail to fire**,
and nothing more. No claim about its false-positive rate is made or would be supportable at
this n. The two are also not the same kind of null, and collapsing them into one verdict label
loses that: LUNA16 is a benchmark where position genuinely carries nothing about the label,
while PI-CAI is a benchmark that does not expose the axis the baseline needs at all.

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
is **0.692 [0.626, 0.755]** — trivial fractions 0.467 [0.307, 0.623] and 0.532 [0.350, 0.710]
[→ `pipeline_out/trivial_baselines/picai_case_level.md`; `paper/audit_results.md` §2.1]. The
strongest single columns are `patient_age` at 0.639 and `psa` at 0.638. The categorical
verdict on these two rows is the one place where the mechanical rule (PARTIAL, since both
fractions exceed 0.30) and a hand-assignment weighing the cohort caveat (NOT MATCHED) disagree.
Both are reported; the fraction is the same under either, which is the argument for leading
with it.

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

### 3.10 How often is any of this reported? A pre-registered prevalence screen

Everything above establishes that a benchmark *can* be reached without pixels and that the
reading unit does the work. Whether the field checks is a separate question, and it is
answered here. Every number in this section is written from
`paper/screen/analysis/pooled_final.json`, which the released script `pool_final.py`
regenerates from the committed coding files, and the flow is drawn in Figure 7
(`paper/figures/prisma_flow_pooled.svg`).

**Flow, and the censoring, first.** The pre-registered extension rule was executed: screening
began at permutation positions 1–100 and continued into the reserve in blocks of fifty until
the target of 75 included papers was reached, which happened at position 260. **Of 250 papers
screened, 115 were excluded** — 39 as segmentation studies whose reported sensitivity,
specificity or AUC does not make them classification studies, 32 as papers whose classifier
input is a derived feature vector or connectivity matrix with the image discarded, and 44 for
the remaining reasons. **Forty-four papers were eligible on their abstract but could not be
obtained in full text: 32.6% [25.3%, 40.9%] of the eligible-looking set.** Ninety-one were
included and readable, against a pre-registered target of 75, which is met.

**Enlarging the sample did not reduce the censoring, and could not.** Four blocks were screened
in four separate sessions and returned unreachable rates of 35.6%, 34.6%, 22.7% and 32.1%;
adding 150 papers to the original 100 moved the pooled figure from 36.4% to 32.6%. The access
notes record the same cause in every block: publisher sites returned HTTP 403 to every
automated request, no screener held institutional access, and the interlibrary-loan rung cannot
complete inside a working session. Several records that Unpaywall reports as open access were
among them. No infringing source was used and no bot-detection challenge was circumvented; a
paper reachable only through such a source stayed unreachable. **Censoring here is a property
of the literature's access conditions, not of the sample size, and only recovered full texts
can reduce it.** The one legitimate lever that worked at scale — retrieving publisher-deposited
render PDFs through Europe PMC — took a single block's unreachable rate from 60.7% to 34.3%.

Because censoring exceeds the pre-specified 15% threshold, the protocol makes the **bounding
interval, not the complete-case estimate, the headline** for every affected endpoint. We report
both throughout, and never the point estimate alone.

**Agreement, before any prevalence number.** On the fifteen-paper overlap set, as originally
sealed, the primary flag returned raw pairwise agreement of **65.6% [50.0, 80.0]** and Fleiss'
κ **−0.015 [−0.164, 0.120]**, against a pre-specified floor of 90% raw or κ 0.60. **The
threshold failed and the pre-registered remedy was triggered.** Reporting the failure is not
optional and the diagnosis does not excuse it, but the diagnosis is clean and is itself a
finding: the frozen extraction form declares the six baseline sub-flags as booleans and provides
**no level for "could not be assessed"** — no instruction for a paper excluded before stage-2
coding, and none for a paper whose full text was never obtained. Four screeners independently
invented four different conventions for those records (a literal `false`, a null, the string
`"unclear"`, and a literal `false` again). Every disagreement on the flag falls on a record
where the field is undefined. **Restricted to the six overlap papers all four screeners
obtained and included — the only papers on which the code is defined — agreement is 100%, four
of four raters, unanimously negative on all six sub-flags.**

**The remedy was executed.** A documented adjudication round produced fourteen decision rules,
including the missing `not_assessable` level, and every already-coded record was re-coded under
the amended codebook. Re-expressed under the two amendments that add a missing level and cannot
change any screener's reading of any paper, the primary flag gives raw **95.6% [86.7, 100.0]**,
Fleiss' κ **0.932 [0.777, 1.000]**, Gwet's AC1 **0.934 [0.800, 1.000]**, fourteen of fifteen
papers unanimous, and the six pairwise Cohen's κ move from {0.000, 0.000, undefined, 0.390,
0.000, 0.000} to {0.898, 0.898, 1.000, 1.000, 0.898, 0.898}. **Both pre-registered floors are
met.** We state plainly what that is and is not: it is the *same four sealed files* re-encoded,
not an independent re-rating, and no sealed file was modified. A genuine post-amendment
reliability estimate requires a fresh independent four-screener coding under the amended
codebook, which we have not run; and the 200 reserve records are single-coded, so the reliability
of the whole pooled sample rests on fifteen papers.

One field disagrees for real, and the amendment does not remove it. `split_unit` returned κ
**0.498 [0.267, 0.692]** as sealed and **0.637 [0.430, 0.824]** after the amendment, with nine
of fifteen papers unanimous; on one paper a single screener read a patient-level split where
three read a slice-level one, and on another the four split two-two between "patient-level" and
"randomly split, unit not stated". Four trained readers working from the same Methods section
cannot reliably determine what unit a paper split at. That is a substantive statement about how
these papers are written, and it is also a measurement ceiling on our own split-unit estimate,
which is therefore reported with its κ attached every time.

**Primary endpoint.** **No paper reported a measured zero-image baseline of any kind: 0 of 91,
0.0% [0.0%, 4.1%].** The bounding analysis over the forty-four unreachable papers gives
**[0.0%, 32.6%]**, and that is the headline the protocol requires; the complete-case estimate is
reported beside it and never instead of it. Restricting to the 79 papers whose fourteen-term
search over the full text *and the supplement* is fully recorded — the codebook does not accept
an unevidenced negative — the numerator is unchanged: 0 of 79, 0.0% [0.0%, 4.6%].

A form of the result that does not depend on the imputation at all is available and is stronger:
across **all 345 coded records covering 300 distinct sampled papers** — every screener, every
block, including all excluded and all unreachable papers — **not one carries a single positive
code on any of the four zero-image sub-flags.** No constant or prevalence predictor with a
measured value, no positional model, no acquisition-metadata model, no permutation null. This is
a statement about what was found rather than about a denominator, and the censoring does not
touch it.

**The near misses are worth naming, because they show what the absence looks like from inside.**
One paper builds an explicit comparison model from clinical variables and tumour location alone
and reports that it *beats* the image-only network (macro AUC 0.867 [0.810, 0.909] against 0.851
[0.775, 0.902]) — and reads the result as evidence that location information improves
identification, not as evidence that its six-class benchmark is largely solvable from size and
position. It is coded negative here only because the location features are computed by the
paper's own segmentation network and therefore are not pixel-free. A second tabulates label
frequency against relative intracranial height, reports a position-stratified AUC for each band,
observes that its best band "had the most number of positive cases", and still reports no
positional baseline. A third reports 95.87% accuracy on a cohort with a 77.8% no-information
rate that it never computes. In one further paper chance is asserted in a figure legend — "the
diagonal dotted lines in the ROC curves represent random chance performance" — and never
measured; five such papers appear in the sample.

**What exists instead.** Five of 91 papers (5.5% [2.4%, 12.2%]) reported any non-imaging
comparator, and four of those five are the "clinical model" arm of a clinical-plus-radiomics or
clinical-plus-deep-learning nomogram. That comparison is a real and useful one, and it is not
this one: it asks whether imaging adds to clinical data, not whether the benchmark is separable
without information. The practice that exists in this literature is *imaging versus clinical
variables*. The practice that does not appear anywhere in 300 sampled papers is *the benchmark
versus nothing*.

**The positional distribution is not merely under-reported; it is nearly unexamined.** One of
the 91 readable papers showed the positional distribution of its labels in a figure or table
(**1.1% [0.2%, 6.0%]**) — a vertebral-level fracture table — and one more mentioned it
qualitatively. None showed a histogram of positive-slice position, reported a mean or standard
deviation of relative position, or gave a position-stratified metric. The confound this paper
measures is not argued about and dismissed in this literature. It is essentially not looked at.

**The reading unit.** Forty-four percent of papers evaluate below the patient: 40 of 91, **44.0%
[34.2%, 54.2%]**, counting slice, lesion, and per-scan metrics that are never aggregated to the
patient. Seventeen (18.7% [12.0%, 27.9%]) headline a slice-level number. **Eighty-five of 91
report exactly one evaluation unit and never contrast two.** Of the nineteen that report any
slice-level metric at all, **six also report a patient-level one — 31.6% [15.4%, 54.0%] on
n = 19**, an interval wide enough that it constrains little and is reported for completeness
rather than as an estimate. Cross-tabulating the headline unit against the primary flag returns
zero in every cell: there is no subgroup, at any evaluation unit, in which zero-image baselines
are reported.

**The split, and the intervals.** Twenty-nine of 91 papers (**31.9% [23.2%, 42.0%]**, at
κ 0.64) state a subject-level split. Seventeen (18.7%) split at an image or slice unit, so the
same patient can appear on both sides. Twenty-two more say only that the data were "randomly
split" and never name a unit; the codebook does not upgrade these to patient-level merely
because the word "patients" appears elsewhere in the paper. Disjointness of the split is
verified in eight papers, stated without verification in 34, and not stated at all in 49. Two
papers in 91 (2.2% [0.6%, 7.7%]) report a subject-clustered uncertainty interval; **forty-eight
report no uncertainty of any kind.** Eleven (12.1% [6.9%, 20.4%]) report the number of positive
*patients* as well as the number of positive slices.

**Exploratory subgroups, and what they show.** Labelled exploratory, with no multiplicity
correction and no tests, the primary endpoint is 0/n in every pre-specified stratum: both year
bands, all six modalities, public and private datasets, and all seven evaluation units. The
informative variation is in censoring, and it runs in the direction the protocol predicted
before the result was seen: unreachability is 36.0% [26.7%, 46.6%] in clinical and radiology
journals against 11.5% [4.0%, 29.0%] in engineering and computing ones, and 37.6%
[28.5%, 47.8%] in 2023–2026 against 21.4% [11.7%, 35.9%] in 2019–2022. Paywalled articles skew
toward the venues most likely to demand a comparator arm, so silently dropping the unreachable
papers would bias the primary endpoint *downward* — toward our own thesis — which is why both
bounding analyses are reported unconditionally rather than as a sensitivity. The venue
classification here is a provisional keyword heuristic over journal names, not the
scope-statement reading the protocol specified, and 23 papers remain unclassified.

**One further block was screened past the stopping point and is reported separately.** After 91
included papers the extension rule had already stopped; a fifth block of fifty records
(positions 261–310) was nonetheless screened. Pooled with it, the flow is 300 screened, 114
included, 56 unreachable (32.9% [26.3%, 40.3%]), and the primary endpoint is 0/114, 0.0%
[0.0%, 3.3%], bound [0.0%, 32.9%]. Continuing past a stopping rule that has already fired is a
data-dependent continuation, so this block is labelled post-hoc wherever it appears and is never
pooled into the pre-registered denominator. It changes the primary result in neither direction.

**What this screen does and does not support.** It supports the statement that in a
pre-registered random sample of the volumetric-imaging classification literature, the check this
paper describes was reported in none of the 91 papers that could be read and in none of 345
coded records, and that the positional structure it exploits was examined in one paper. It does
not support "no published paper reports a zero-image baseline" — Yan et al. published one in
2018, outside this frame — and it does not support a precise prevalence, because a third of the
eligible sample is unread, which is why the reportable primary result is an interval 33 points
wide and not a point.

---

## 4. Discussion

### 4.1 What a trivial fraction licenses

A trivial fraction is a statement about an **evaluation protocol**. It says that a pixel-blind
model reaches that much of a reported margin over chance under the same protocol. It is not a
decomposition: the baseline and the published model may exploit the same shortcut, different
shortcuts, or overlapping ones, and the fraction cannot distinguish those cases. It is not a
claim about a model's internals, and no analysis of a label file could be. We could not
reproduce the pipeline of the benchmark that produced our six matched rows, and we say so in
the abstract, the results and the limitations.

Reporting it continuously rather than as a verdict changes what may be said, in both
directions. It licenses the quantitative claim — *a median of 47% of the peer-reviewed
published margin over chance is reachable with no pixels, and 40–60% per subtype on the one
benchmark where we have six peer-reviewed head-to-heads on an identical cohort* — which is
stronger and more defensible than "one benchmark matched". It also forbids the reverse:
because eight of the nine peer-reviewed rows sit near 0.5 rather than near 1, this paper
cannot say that published slice-level numbers are *accounted for* by trivial baselines. Half
a margin is half a margin.

### 4.2 Why the collapse, not the match, is the general finding

The six matched rows rest on one preprint's Table II. The unit-of-evaluation result rests on
our own computations on published label tables, and no published number enters it. If a reader
grants nothing else in this paper, the sentence that survives is: *on the RSNA 2019
Intracranial Haemorrhage benchmark — 752,802 slices, 18,938 patients, the benchmark whose own
official metric is per-slice — the slice-level and patient-level readings of one pixel-blind
score vector are 0.737 [0.735, 0.740] and 0.453 [0.445, 0.461], the same holds on all six of
its labels, and only the first of those readings is what a paper would print.* The effect
recurs on three further benchmarks with gaps of 0.20–0.43 AUROC, and is absent on two, and
both facts are reported.

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

Against Maier-Hein et al., the distinction is which source of instability is being measured,
and their result is the reason §3.3bis is written as a null. They show that challenge rankings
are not robust to the test data, the ranking scheme or the observer, and that τ in the
0.74–0.85 range is already enough for critical changes in ranking. The between-unit τ of the
strongest published inversion we located (CAMELYON16, 0.754) sits inside that band. So the
evaluation unit is not a newly discovered source of ranking instability of larger magnitude
than the ones they document; it is another member of the same family, and the correct question
is not whether the two units disagree but whether either ordering is identified at all. On our
cohorts it is not, and we say so.

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

**The rank-inversion analysis compares configurations we trained, not published systems.** All
21 method cells in §3.3bis share one training pipeline, one preprocessing path, one optimiser
schedule and one seed (42), and they are 7 architectures × 3 input conditions rather than 21
independent methods. A configuration set assembled by one group has far less between-method
spread than a challenge leaderboard — CAMELYON16's 32 independent submissions span slide AUC
0.78–0.99 — which is the most likely reason our between-unit signal sits inside the noise while
theirs does not. Our inversion arm is therefore an internal consistency check on this project's
own models, not evidence about the field, and it must not be read as the latter. Closing that
gap needs per-algorithm scores at both units from a benchmark that publishes both; PI-CAI is
the obvious target, since its official ranking is already the mean of a lesion-level average
precision and a patient-level AUROC over 293 submissions on a hidden 1,000-scan cohort, but its
per-algorithm table is not published in machine-readable form and we did not obtain it.

**The rank-inversion null is a statement about these cohorts and this design, not an absence.**
The bootstrap resamples subjects, not training runs, so seed-to-seed variability appears in
neither the between-unit nor the within-unit term. That omission makes the within-unit noise
floor too *low* and therefore biases the test towards declaring a unit effect; we found none
regardless, which is the conservative direction, but the floors in §3.3bis
(`D` ≈ 0.21–0.71 in rank distance) should be read as the minimum detectable effect rather than
as evidence that no unit effect exists. Two cohorts (knee, breast) carry three methods each, at
which width Kendall τ spans [−1, 1] and the knee cohort is reported as no estimate because all
three methods saturate the aggregated unit.

**The published-inversion search never had a working web search.** Both passes ran with the
WebSearch tool returning API errors, so coverage came from the Europe PMC full-text API, the
arXiv API, the GitHub API and direct leaderboard fetches. Kaggle is JavaScript-rendered and its
solution discussions — which sometimes report both image-level and exam-level validation scores
in prose — were not searched. The rarity denominators in §3.3bis are lower bounds on coverage
and must be quoted with the search method attached. Every FROC-versus-AUC case also confounds
the unit with the metric, since AUC is not definable at the lesion level; that confound is
entailed by the unit change but it means those cases speak to *reported* rankings rather than to
the unit in isolation.

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
agreement.** RSNA 2023 Abdominal Trauma and RSNA 2022 Cervical Spine are behind click-through
agreements that were not accepted; PI-CAI's slice-level arm would require 1,295
lesion-delineation volumes and a NIfTI reader. Their absence limits the audit's breadth and is
not evidence about those benchmarks. (RSNA 2019 ICH was reached in the revision pass, via a
public pixel-free mirror and with no agreement accepted.)

### 5.1 Limitations of the prevalence screen

These are stated separately because the screen is a different kind of evidence with a
different set of failure modes, and because three of them are severe enough that they must be
read before any prevalence figure is quoted.

**A third of the eligible sample could not be read, and enlarging the sample did not help.**
Forty-four of 135 eligible papers, 32.6% [25.3%, 40.9%] — more than double the 15% threshold at
which our own protocol makes the bounding interval the headline. The honest primary estimate is
therefore an interval from 0.0% to 32.6%, and the upper end of that interval is set entirely by
missingness rather than by anything observed. **This is the binding limitation of the screen and
it is not a sample-size problem**: four blocks screened in four separate sessions returned
unreachable rates of 35.6%, 34.6%, 22.7% and 32.1%, and adding 150 papers to the original 100
moved the pooled figure by under four points. Only recovered full texts can narrow it. The
direction of the bias runs against us and the data confirm it: unreachability is 36.0% in
clinical and radiology journals against 11.5% in engineering and computing ones, and paywalled
papers skew toward the venues most likely to demand a comparator arm, so dropping them would
have flattered our hypothesis. That is why the bounds are unconditional. It remains true that a
screen with institutional access would produce a much tighter number, and that we did not have
one.

**Inter-rater agreement failed its own pre-specified threshold on the primary field as
originally sealed, and the number that now clears it is a re-encoding rather than a re-rating.**
Raw agreement 65.6%, Fleiss' κ −0.015, against a floor of 90% or 0.60. The failure is fully
attributable to the extraction form having no level for "could not be assessed", and agreement
is 100% on the six overlap papers where the field is defined. The pre-registered remedy
(adjudicate, amend the codebook, re-code every already-coded record) **was executed**, and under
the amended encoding the flag gives raw 95.6% and κ 0.932, meeting both floors. That encoding
re-expresses the *same four sealed files* under two amendments that add a missing level and
cannot change a reading; it is not an independent re-rating, and a fresh four-screener coding
under the amended codebook has not been run. Nor has the protocol's provision for a 20%
within-batch cross-check outside the overlap set, and every one of the 200 reserve records is
single-coded — so the reliability of the pooled sample of 91 papers rests on fifteen. The
codebook defect is worth stating on its own account: a form frozen after a ten-paper pilot still
failed to anticipate the most common record type in the sample, which is a caution to anyone
designing a similar screen.

**A fifth block of records lies outside the pre-registered stopping rule.** The extension rule
stopped at permutation position 260, where 91 included papers exceeded the target of 75. A
further fifty records (positions 261–310) were screened afterwards. Continuing past a stopping
rule that has already fired is a data-dependent continuation, so that block is reported as a
labelled post-hoc extension throughout and never pooled into the pre-registered denominator. It
moves no endpoint materially in either direction.

**Absence in a sample is not absence in a literature.** The screen licenses a statement about
a random sample of 9,979 frame records, not a universal claim. Yan et al. reported a
position-only baseline on DeepLesion in 2018, and a Kaggle notebook constructed the pixel-free
positional predictor in 2020; neither falls inside this frame. Any sentence beginning "no
published paper" is unsupported by this evidence and does not appear in this manuscript.

**The frame is broader than the failure mode, and the slice-level subgroup is small.** The
query selects volumetric-imaging classification papers generally. Papers reporting any
slice-level metric are 20.9% of the included set, so the secondary endpoint conditioned on them
rests on n = 19 and its interval spans 15% to 54%. It constrains little and is reported for
completeness.

**One coded field is not reliably codeable.** Split unit returned κ 0.498 as sealed and 0.637
after the codebook amendment. The estimate that 31.9% of papers state a subject-level split
should be read as a description of what four readers could extract, not as a property of the
literature measured without error.

**The venue subgroup is not the classification the protocol specified.** Protocol §8 requires
each journal to be classified from its scope statement before unblinding. That was not done, and
the clinical-versus-engineering split reported above is a keyword heuristic over journal names
that leaves 23 papers unclassified. It is labelled exploratory and provisional, and the
censoring contrast it shows should be read as consistent with the pre-stated bias direction
rather than as a measurement of it.

**Screeners were not blinded to the study hypothesis.** The protocol mitigates this with
mandatory verbatim quotes, mandatory full-text searches before any negative code, sealed
independent submission of the overlap set, and a pooling rule whose ties break against the
hypothesis. It does not eliminate it.

---

## 6. Conclusion

The unit at which a three-dimensional benchmark is read decides what its number means. On the
RSNA 2019 Intracranial Haemorrhage benchmark — 752,802 slices from 18,938 patients, the
benchmark whose own official metric is per-slice — a single pixel-blind score vector reads as
0.737 [0.735, 0.740] at the slice level and 0.453 [0.445, 0.461] at the patient level, below
chance, and does so on all six of its labels. Nothing changes between the two readings except
the unit; only the first is what a paper would print; and no published number enters either.
The same divergence appears on three further benchmarks with gaps of 0.20–0.43 AUROC, and is
absent on two.

How much of a *published* number a pixel-blind model reaches is best reported as a
distribution rather than as a verdict. Against peer-reviewed comparators, on the same metric
and the same evaluation unit, the median is 0.469 of the reported margin over chance (IQR
0.437–0.490, range −0.002 to 0.613 over nine benchmark-arms), and on the one benchmark
supplying six peer-reviewed head-to-heads on an identical cohort the fractions run 0.395–0.613
across its subtypes. On one benchmark a pixel-blind model reaches the published number
outright — 0.854 [0.812, 0.891] against 0.861, from a function of the slice index, on the
authors' own label file and split — but that comparator is an unreviewed preprint and it is
reported here as a worked example, not as evidence. On two benchmarks nothing is reached at
all, and those results carry the same prominence, because a measure that always fires measures
nothing.

In a pre-registered random sample of the literature that produces such numbers, none of the 35
papers that could be read reported a zero-image baseline of any kind, and none reported where
along the stack their labels fall. Neither figure is precise — a third of the eligible sample
was unreadable and the sample is half its target size — but neither is close to the values that
would make this a solved problem.

Three fields in the label file a benchmark already publishes — subject, slice index, split —
make all of this checkable by anyone, for free, without the images. Reporting the pixel-blind
baselines beside every headline number, at the patient level, with subject-clustered intervals
and a position-stratified statistic, costs one command and removes an entire class of ambiguity
from the literature.

---

## Data and code availability

- **Tool.** `trivialbaselines` v1.0, MIT licensed, `numpy` and `pandas` only, with
  `--self-test`. [*Resolve the install path before submission: `paper/protocol.md` and
  `paper/checklist.md` advertise `pip install trivialbaselines` while
  `trivialbaselines/README.md` documents a `git clone && pip install .`. They must agree.*]
- **Audit payloads.** `pipeline_out/trivial_baselines/*.json` and `*.md`, one JSON payload and
  one human-readable card per run, 20 runs.
- **The principal result, and its independent recomputation.**
  `pipeline/audit_prep/rsna_ich_unit_collapse.py` recomputes the slice-to-patient divergence on
  all 18,938 RSNA ICH patients with an implementation sharing no code with the audit harness —
  its own fold assignment and seed, its own binning, `scikit-learn`'s AUROC, its own clustered
  bootstrap — and writes `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` plus the
  console log at `pipeline_out/audit_logs/rsna_ich_unit_collapse.log`. Inside the bootstrap the
  AUROC is computed from per-patient count vectors over the distinct score values, which is
  exact midrank arithmetic and is asserted equal to the `scikit-learn` value before any
  resampling; that is what makes 2,000 replicates on 750k rows tractable.
- **The trivial-fraction distribution.**
  `pipeline/audit_prep/trivial_fraction_distribution.py` assembles every (benchmark, published
  comparator) row, computes the fraction and its interval, applies the verdict rule
  mechanically, exercises the definition at its extremes, and writes
  `paper/trivial_fraction_distribution.{json,md}`. It re-fits no baseline: every value is read
  from an artefact named on the row.
- **Label-table preparation.** `pipeline/audit_prep/` (one script per benchmark plus the
  DeepLesion Yan-conditions rebuild, the LUNA16 CPM scorer, the RSNA ICH split-geometry
  replication, the RSNA PE row-order test, the unit-collapse recomputation and the
  distribution assembler).
- **Label-file provenance.** Table 1: source URL, byte count, SHA-256 prefix and licence for
  every file used. No pixel data was downloaded for any audited target.
- **Worked-example report.** `pipeline_out/report/RESULTS.md` and `verdict.json`, generated by
  `pipeline/s06_report.py`, with every number traced to its source JSON.
- **Prevalence screen, released in full.** The frozen protocol (`paper/screen_protocol.md`)
  and extraction form (`paper/screen_frame.json`), both with their changelogs; the sampling
  frame, its SHA-256 and the seeded permutation (`paper/screen/frame_meta.json`,
  `frame_pmids.txt`, `permutation.txt`) with the script that regenerates them
  (`paper/screen/reproduce_frame.py`); the **four sealed, independently submitted screener
  files** with every verbatim quote, every search string run and every access-ladder note
  (`paper/screen_batch_{A,B,C,D}.json`); the adjudication audit trail and the re-code overlay
  (`paper/screen_adjudication.md`, `paper/screen_recoded.json`); the four reserve blocks
  (`paper/screen_reserve_R{1,2,3,4}.json`); and the pooling script
  (`paper/screen/analysis/pool_final.py`) with its output
  (`paper/screen/analysis/pooled_final.json`). Every prevalence and agreement number in this
  paper is regenerated by one command from those inputs. The 44 unreachable records are listed
  by PMID so that a reader with better access can complete the screen.
- **Master findings ledger.** `paper/FINDINGS.md` — every established number, its interval and
  the file it was read from. Internal; not for submission.
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

### Table 2 — The trivial fraction across every benchmark and every published comparator

*Distribution (the headline). Rows are not independent: one paper's table can supply several
comparator systems for one benchmark, so the primary line takes each benchmark-arm's
strongest published system, which enlarges the denominator and is therefore conservative.*

| set of rows | n | min | Q1 | **median** | Q3 | max | ≤0.05 | 0.30–0.70 | ≥1 |
|---|---|---|---|---|---|---|---|---|---|
| **peer-reviewed comparator, strongest system per benchmark-arm** | 9 | −0.002 | 0.437 | **0.469** | 0.490 | 0.613 | 1 | 8 | 0 |
| peer-reviewed comparator, all rows | 18 | −0.002 | 0.455 | 0.485 | 0.514 | 0.889 | 1 | 16 | 0 |
| strongest system per benchmark-arm, any comparator | 11 | −0.002 | 0.452 | 0.480 | 0.562 | 0.981 | 1 | 8 | 0 |
| all rows | 24 | −0.002 | 0.469 | 0.512 | 0.910 | 1.655 | 1 | 16 | 4 |
| preprint comparator only (§3.2) | 6 | 0.973 | 1.020 | 1.142 | 1.518 | 1.655 | 0 | 0 | 4 |

*Rows. The verdict column is secondary and is retained, not deleted.*

| # | dataset | published number | source of the published number | peer-reviewed? | our best zero-image baseline | **trivial fraction** [CI] | verdict |
|---|---|---|---|---|---|---|---|
| 1 | RSNA ICH, **any** | 0.9843 slice ROC AUC | Burduja et al., *Sensors* 2020;20(19):5611, Table 3, ResNeXt-101+BiLSTM | yes | 0.737 [0.735, 0.740] positional 20-bin | **0.490** [0.485, 0.495] | PARTIAL |
| 2 | RSNA ICH, epidural | 0.9851 | same | yes | 0.712 [0.700, 0.725] | **0.437** [0.411, 0.464] | PARTIAL |
| 3 | RSNA ICH, intraparenchymal | 0.9927 | same | yes | 0.751 [0.747, 0.755] | **0.510** [0.502, 0.518] | PARTIAL |
| 4 | RSNA ICH, intraventricular | 0.9970 | same | yes | 0.805 [0.802, 0.808] | **0.613** [0.607, 0.620] | PARTIAL |
| 5 | RSNA ICH, subarachnoid | 0.9821 | same | yes | 0.690 [0.686, 0.695] | **0.395** [0.386, 0.404] | PARTIAL |
| 6 | RSNA ICH, subdural | 0.9682 | same | yes | 0.720 [0.717, 0.723] | **0.469** [0.463, 0.476] | PARTIAL |
| 7–12 | RSNA ICH, six labels | 0.9752 / 0.9703 / 0.9883 / 0.9953 / 0.9644 / 0.9576 | same table, plain ResNeXt-101 (no LSTM) | yes | as rows 1–6 | 0.500 / 0.451 / 0.515 / 0.615 / 0.410 / 0.480 | PARTIAL |
| 13 | DeepLesion | 0.905 ± 0.002 8-class accuracy | Yan et al., CVPR 2018, Table 1, triplet + type + location + size | yes | 0.557 [0.524, 0.578] | **0.480** [0.431, 0.511] | PARTIAL |
| 14 | DeepLesion | 0.862 8-class accuracy | same table, multi-scale ImageNet feature | yes | 0.557 [0.524, 0.578] | **0.513** [0.460, 0.546] | PARTIAL |
| 15 | DeepLesion | 0.597 8-class accuracy | same table, **their own** location-feature baseline | yes | 0.557 [0.524, 0.578] | **0.889** [0.799, 0.947] | PARTIAL |
| 16 | PI-CAI | 0.91 (0.87–0.94) case AUROC, AI system | Saha et al., *Lancet Oncol* 2024;25:879-887 | yes | 0.692 [0.626, 0.755] metadata CART | **0.467** [0.307, 0.623] | PARTIAL / NOT MATCHED † |
| 17 | PI-CAI | 0.86 (0.83–0.89) case AUROC, 62 radiologists | same | yes | 0.692 [0.626, 0.755] | **0.532** [0.350, 0.710] | PARTIAL / NOT MATCHED † |
| 18 | LUNA16 (FP-reduction) | >0.95 sensitivity at <1 FP/scan | Setio et al., *Med Image Anal* 2017;42:1–13 | yes | 0.0006 at 1 FP/scan; **CPM 0.0020** vs random-score 0.0027 | **−0.002** | **NOT MATCHED** |
| 19 | fastMRI Prostate **T2** | 0.861 slice AUROC | Rempe et al. 2024, arXiv:2407.06165, Table II gold standard | **no — preprint** | 0.854 [0.812, 0.891] positional 20-bin | 0.981 [0.865, 1.084] | **MATCHED** |
| 20 | fastMRI Prostate **T2** | 0.809 slice AUROC | same, PCA ×2 magnitude + phase | **no — preprint** | 0.854 [0.812, 0.891] | 1.146 [1.011, 1.266] | **MATCHED** (exceeds) |
| 21 | fastMRI Prostate **T2** | 0.714 slice AUROC | same, R = 16 PCA coil combination | **no — preprint** | 0.854 [0.812, 0.891] | 1.655 [1.459, 1.829] | **MATCHED** (exceeds) |
| 22 | fastMRI Prostate **DWI** | 0.861 slice AUROC | as row 19 | **no — preprint** | 0.851 [0.816, 0.887] positional 20-bin | 0.973 [0.876, 1.073] | **MATCHED** |
| 23 | fastMRI Prostate **DWI** | 0.809 slice AUROC | as row 20 | **no — preprint** | 0.851 [0.816, 0.887] | 1.137 [1.023, 1.253] | **MATCHED** (exceeds) |
| 24 | fastMRI Prostate **DWI** | 0.714 slice AUROC | as row 21 | **no — preprint** | 0.851 [0.816, 0.887] | 1.642 [1.478, 1.810] | **MATCHED** (exceeds) |

Rows 13–15 use the majority class (0.236) as the chance anchor, not 0.5; row 18 uses the
measured random-score reference (0.0027). † The verdict rule applied mechanically returns
PARTIAL on rows 16–17; a hand-assignment weighing the cohort caveat of §3.4 returns NOT
MATCHED. Both are shown; the fraction is identical either way.

*Non-comparable rows — audited, no defensible published comparator, no fraction computable.*

| # | dataset | zero-image result | why no comparator |
|---|---|---|---|
| A | fastMRI+ knee, meniscus tear per slice | 0.873 [0.858, 0.886] slice; 0.510 [0.428, 0.592] patient | data descriptor, no published slice-level classification number located; also a 199-of-1,173 volume subset |
| B | fastMRI+ knee, any annotated finding | 0.801 [0.779, 0.824] slice; 0.558 [0.470, 0.648] patient | as above |
| C | Duke Breast Cancer MRI, owner-defined slice task | 0.823 [0.811, 0.834] slice; patient undefined | the data owners define the task but publish no metric |

[→ `paper/trivial_fraction_distribution.{json,md}`; `paper/audit_results.md` §2.]

### Table 3 — Slice level versus patient level, with the same score vector and the same estimator

*Panel A — RSNA ICH, all 18,938 patients / 752,802 slices, five-fold subject-disjoint CV,
2,000 patient-clustered bootstrap replicates. This is the paper's principal result and it
involves no published number.*

| label | slice prev. | patient prev. | **slice** AUROC | **patient** AUROC (mean-agg) | patient (max-agg) | gap |
|---|---|---|---|---|---|---|
| **any haemorrhage** | 0.143 | 0.404 | **0.737** [0.735, 0.740] | **0.453** [0.445, 0.461] | 0.500 | **0.284** |
| epidural | 0.004 | 0.016 | 0.712 [0.700, 0.725] | 0.492 [0.461, 0.524] | 0.486 | 0.220 |
| intraparenchymal | 0.048 | 0.247 | 0.751 [0.747, 0.755] | 0.480 [0.471, 0.490] | 0.503 | 0.271 |
| intraventricular | 0.035 | 0.174 | 0.805 [0.802, 0.808] | 0.497 [0.487, 0.508] | 0.495 | 0.307 |
| subarachnoid | 0.047 | 0.182 | 0.690 [0.686, 0.695] | 0.485 [0.475, 0.496] | 0.505 | 0.205 |
| subdural | 0.063 | 0.178 | 0.720 [0.717, 0.723] | 0.476 [0.466, 0.487] | 0.500 | 0.243 |

Within-series label-permutation null: 0.502 slice / 0.523 patient. Constant predictor: 0.492
slice / 0.501 patient. The naive slice-resampled interval, computed and refused, is 1.5–2.0×
too narrow. The released tool, run independently on the same file, gives 0.7376 [0.7352,
0.7399] slice and 0.4561 [0.4478, 0.4640] patient on the `any` row, with a bin sweep of
0.716 / 0.733 / 0.738 / 0.745 (slice) and 0.437 / 0.445 / 0.456 / 0.632 (patient) over
5 / 10 / 20 / 50 bins. [→ `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json`,
`rsna_ich_any_slice_full.json`.]

*Panel B — the other seven dataset-arms, and the remedy.*

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

### Table 3bis — Does the unit change the ranking? Between-unit disagreement against the within-unit noise floor

Twenty-one method configurations (7 architectures × 3 input conditions), five cohorts, one
shared subject-clustered bootstrap of 2,000 replicates. `D = 1 − τ_b`, so `D/2` is the fraction
of method pairs ordered differently; δ is the paired within-replicate difference between
between-unit and within-unit disagreement, and the unit is credited only if δ clears both
floors. Split-half τ resamples two disjoint halves of the subjects, 200 replicates.

| cohort | methods | subjects | ρ(slice, patient) [95%] | between-unit `D` | δ vs slice floor [95%] | δ vs patient floor [95%] | split-half τ slice / patient | top-1 reproduces slice / patient | pairs: examined / discordant / **surviving** | verdict |
|---|---|---|---|---|---|---|---|---|---|---|
| brain (confound) | 13 | 136 | 0.852 [0.748, 0.989] | 0.282 | −0.018 [−0.256, 0.231] | +0.026 [−0.333, 0.282] | 0.58 / 0.66 | 32% / 75% | 78 / 11 / **0** | cannot name a winner |
| prostate T2 | 21 | 67 | −0.311 [−0.356, 0.635] | 1.173 | +0.514 [0.010, 0.962] | +0.175 [−0.334, 0.655] | 0.21 / **−0.42** | 47% / 23% | 210 / 121 / **0** | cannot rank |
| prostate DWI | 18 | 45 | 0.067 [−0.152, 0.790] | 0.928 | +0.259 [−0.405, 0.804] | +0.131 [−0.431, 0.667] | −0.03 / **−0.24** | 40% / 23% | 153 / 71 / **0** | cannot rank |
| breast | 3 | 70 | 0.500 [−0.500, 1.000] | 0.667 | 0.000 [−2.000, 0.667] | −0.667 [−2.000, 1.333] | −0.33 / **−1.00** | 60% / 39% | 3 / 1 / **0** | cannot rank |
| knee | 3 | 29 | undefined | undefined | — | — | — | — | 3 / 0 / **0** | no estimate (patient AUROC 1.000 for all three) |
| **total** | | | | | | | | | **447 / 204 / 0** | |

**No cohort's between-unit disagreement exceeds its within-unit resampling noise at both units,
and no inversion pair survives.** Panel B, the paired unit × condition interaction
`I = (AUC_agg,mag − AUC_agg,phase) − (AUC_slice,mag − AUC_slice,phase)` on the brain cohort:

| architecture | `d_slice` | `d_agg` | `I` [95%] | Holm p | sign flip |
|---|---|---|---|---|---|
| complex_small/scratch | −0.303 | −0.270 | +0.033 [−0.020, +0.085] | 0.426 | no |
| convnext_tiny/scratch | −0.154 | −0.134 | +0.020 [−0.015, +0.054] | 0.426 | no |
| densenet121/imagenet | −0.004 | +0.023 | **+0.027 [+0.009, +0.046]** | **0.012** | **yes** |
| resnet18/imagenet | −0.006 | +0.013 | **+0.020 [+0.009, +0.032]** | **0.006** | **yes** |
| resnet50/imagenet | +0.001 | +0.029 | **+0.028 [+0.012, +0.045]** | **0.006** | no |
| vit_b_16/scratch | −0.197 | −0.155 | **+0.042 [+0.012, +0.070]** | **0.024** | no |
| **mean over the six** | | | **+0.028 [+0.015, +0.041]**, p = 0.001 | | 2 of 6 |

Neither individual ordering is distinguishable from a tie at either unit; the supported
quantity is the shift, not a selection.
[→ `pipeline_out/rankinversion.json`; `pipeline/s16_rankinversion.py`;
`paper/rank_inversion.md` §3–§4.]

### Table 3ter — Published comparisons of two or more methods at two evaluation units

Located by scanning 2,934 open-access full texts through the Europe PMC API plus the arXiv API
and direct leaderboard fetches; WebSearch was unavailable and Kaggle discussions are unsearched.
Negative controls are in the same table as the positives, by design.

| case | unit pair | n | τ | top-1 changes | CIs at both units | unit ⟂ metric? | the caveat that travels with it |
|---|---|---|---|---|---|---|---|
| CAMELYON16 official boards | lesion FROC / slide AUC | 32 | 0.754 | top-5 membership changes | no | no | organisers' own boards; no intervals published |
| Ruan 2021, PLoS One | lesion FROC / slide AUC | 13 | 0.603 (τ_a) | **yes** | no | no | authors state the mechanism: the fine-unit winner was trained at the fine unit |
| Islam 2021, MLMI (RSNA-STR PE) | image AUC / exam AUC | 6 | 0.467 (τ_a) | **yes** | no | partly | exam figure is a mean over nine labels; exam features derive from the image models |
| Jarkman 2022, Cancers | lesion FROC / slide AUC | 4 | **−0.667** (τ_a) | **yes** | **yes** | no | all four intervals overlap; the same models agree perfectly on the paper's own local cohort |
| Guo 2019, Sci Rep | lesion FROC / slide AUC | 6 | 0.133 (τ_a) | **yes** | **yes** | no | intervals overlap heavily; "beats the champion" holds only at the lesion unit |
| ADNI slice/scan, IJERPH 2026 | slice acc / scan acc | 5 | **−0.600** | **yes** | no | **yes** | scan arm covers only the 5 best slice models; max-confidence aggregation; split not subject-disjoint |
| Acute pancreatitis, Diagnostics 2026 | slice acc / patient acc | 5 | 0.300 (τ_a) | **yes** | patient only | **yes** | 37 test patients; two models tie at the patient unit; winner also flips with augmentation |
| **Chen 2025, Sci Rep (negative control)** | lesion FROC / slide AUC | 11 | **0.927** (τ_a) | **no** | no | no | the strongest counter-example: 2 of 55 pairs invert, same winner at both units |
| **LGI1 encephalitis 2023 (negative control)** | slice AUC / patient AUC | 3 | ranking preserved | **no** | no | no | DeLong tests reported; ordering identical on AUC and accuracy |

[→ `paper/published_inversions.json`, `paper/published_inversions_round2.json`;
`paper/rank_inversion.md` §5.]

### Table 4 — The protocol: seven rules, each with the failure it was written for

| # | rule | the measured failure behind it |
|---|---|---|
| 1 | Split at the subject level, and state the unit | slice-level CV inflated accuracy by 30–55%, and reached ~96% on randomly labelled data (Yagis 2021) — the one rule here backed by others' numbers |
| 2 | Report patient level as primary | one score vector, two readings: **0.737 slice vs 0.453 patient on 18,938 patients**, on all six labels of the benchmark whose own metric is per-slice |
| 3 | Subject-clustered intervals, never the slice-level bootstrap | nominal-95% coverage 46.5% vs 91.5%; 3.18× too narrow in our own cohorts, and 1.5–2.0× too narrow on the RSNA ICH label file |
| 4 | Report the zero-image baselines beside every headline | 0.854 against a published 0.861, with no pixels |
| 5 | Publish the positional label distribution and stratify on it | 0.851 → 0.539 when position is held fixed |
| 6 | Test whether metadata predicts the label | release batch predicts breast cancer status at 0.743 against 0.633 for the trained network |
| 7 | Report the trivial fraction **as a number, including when it is small** | median 0.469 against peer-reviewed comparators (IQR 0.437–0.490, n = 9 benchmark-arms), and −0.002 on LUNA16 — the same statistic distinguishes the two, a MATCHED/NOT-MATCHED label does not |

[→ `paper/protocol.md`; the one-page reviewer version is `paper/checklist.md`.]

### Table 5 — Prevalence screen: flow and endpoints

250 papers sampled from a frozen 9,979-record PubMed frame at permutation positions 1–100 and
111–260, the second range drawn under the pre-registered extension rule, which stopped at
position 260 once 75 included papers had been exceeded. Four independent screeners on the
15-paper overlap set; codebook v1.2. Complete case = included and full text obtained. Bounding
analyses impute every unreachable paper both ways and are reported unconditionally. All
intervals Wilson score 95%.
[→ `paper/screen/analysis/pooled_final.json`, regenerated by
`paper/screen/analysis/pool_final.py`; flow drawn in Figure 7.]

**Flow**

| | n |
|---|---|
| screened (permutation positions 1–100, 111–260) | 250 |
| excluded at title/abstract | 79 |
| eligible on abstract, **full text unreachable** | **44** |
| assessed for eligibility at full text | 127 |
| excluded at full text | 36 |
| excluded, both stages, by reason | 115 |
| — segmentation only (E-SEG) | 39 |
| — derived non-spatial input, e.g. radiomics vector or connectivity matrix (E-DERIV) | 32 |
| — natively 2D imaging (E-2D) | 14 |
| — no supervised classifier (E-NOCLF) | 13 |
| — not human medical imaging (E-NONMED) | 9 |
| — publication type (E-TYPE) | 5 |
| — collapsed projection input (E-PROJ) | 3 |
| eligible-looking set (included + unreachable) | 135 |
| **included and readable (analysis denominator)** | **91** |
| pre-registered target | 75 — **met**; extension rule executed to its stopping point |

**Endpoints**

| endpoint | complete case | Wilson 95% | lower bound | upper bound |
|---|---|---|---|---|
| **P1 any zero-image baseline** | **0/91 = 0.0%** | **[0.0, 4.1]** | 0/135 = 0.0% [0.0, 2.8] | 44/135 = 32.6% [25.3, 40.9] |
| P1, evidence-restricted denominator | 0/79 = 0.0% | [0.0, 4.6] | — | — |
| **S5 positional distribution of labels reported** | **1/91 = 1.1%** | **[0.2, 6.0]** | 1/135 = 0.7% [0.1, 4.1] | 45/135 = 33.3% [25.9, 41.6] |
| S1 any non-imaging baseline incl. clinical-only | 5/91 = 5.5% | [2.4, 12.2] | 5/135 = 3.7% [1.6, 8.4] | 49/135 = 36.3% [28.7, 44.7] |
| evaluation unit below the patient | 40/91 = 44.0% | [34.2, 54.2] | — | — |
| S2 headline unit is the slice | 17/91 = 18.7% | [12.0, 27.9] | — | — |
| S3 slice-reporting papers also reporting patient level | 6/19 = 31.6% | [15.4, 54.0] | — | — |
| S4 explicit subject-level split (κ 0.64 — see Table 6) | 29/91 = 31.9% | [23.2, 42.0] | 29/135 = 21.5% [15.4, 29.1] | 73/135 = 54.1% [45.7, 62.3] |
| S6 unreachable among eligible | 44/135 = 32.6% | [25.3, 40.9] | — | — |
| S8 subject-clustered uncertainty interval | 2/91 = 2.2% | [0.6, 7.7] | — | — |
| S9 positive **patients** reported as well as slices | 11/91 = 12.1% | [6.9, 20.4] | — | — |

Censoring exceeds the pre-specified 15% threshold, so for P1, S1, S4 and S5 the **bounding
interval is the headline** and the complete-case column is reported alongside it, not instead
of it. Zero-image baselines appear in **no cell** of the headline-unit × P1 cross-tabulation,
and in **none of the 345 coded records over 300 distinct sampled papers**, including the
excluded and unreachable ones. A fifth block of fifty records (positions 261–310) was screened
after the stopping rule had already fired and is reported as a labelled post-hoc extension: 114
included, S6 = 32.9% [26.3, 40.3], P1 = 0/114 = 0.0% [0.0, 3.3], bound [0.0, 32.9].

### Table 6 — Prevalence screen: inter-rater agreement, 15 overlap papers, 4 screeners

Fleiss' κ is primary. Raw percent agreement and Gwet's AC1 were pre-specified in this table
before any coding, because the primary flag was expected to be extremely skewed and κ collapses
toward zero under skew even at near-perfect agreement. Intervals are bootstrap percentile 95%
over 2,000 resamples, seed 20260729. Column (1) is the four sealed files exactly as submitted
under codebook v1.0; column (2) is the **same four files** re-expressed under the two amendments
that add a missing level and cannot change any screener's reading. **The pre-registered floor
(κ ≥ 0.60 or raw ≥ 90%) is assessed against column (2).**
[→ `paper/screen/analysis/adjudication_out.json`.]

| field | (1) as sealed: raw / κ / AC1 | (2) amended encoding: raw / κ / AC1 |
|---|---|---|
| **P1 zero-image baseline flag** | **65.6% [50.0, 80.0] / −0.015 [−0.164, 0.120] / 0.479** | **95.6% [86.7, 100] / 0.932 [0.777, 1.000] / 0.934** |
| P1 flag, collapsed TRUE vs not | — | 100% / 1.000 / 1.000 |
| **split_unit** | 64.4% [48.9, 80.0] / **0.498 [0.267, 0.692]** / 0.586 | **76.7% / 0.637 [0.430, 0.824] / 0.722** |
| evaluation_unit_reported | 76.7% [63.3, 90.0] / 0.685 [0.465, 0.828] / 0.714 | 87.8% / 0.816 [0.565, 1.000] / 0.859 |
| headline_unit | 76.7% [63.3, 90.0] / 0.425 [0.085, 0.683] / 0.607 | 87.8% / 0.762 [0.473, 1.000] / 0.836 |
| positional_distribution_reported | 82.2% [67.8, 93.3] / 0.648 [0.324, 0.870] / 0.762 | 87.8% / 0.783 [0.555, 1.000] / 0.850 |
| final_inclusion | 86.7% [73.3, 100.0] / 0.785 [0.544, 1.000] / 0.807 | 86.7% / 0.785 / 0.807 |
| fulltext_obtained | 88.9% [75.6, 100.0] / 0.769 [0.484, 1.000] / 0.786 | 88.9% / 0.769 / 0.786 |
| six-subflag vector, restricted to the 6 papers all four obtained and included | 100% / 1.000 / 1.000 | 100% / 1.000 / 1.000 |

Pairwise Cohen's κ on the primary flag over the six screener pairs moves from {0.000, 0.000,
undefined, 0.390, 0.000, 0.000} to {0.898, 0.898, 1.000, 1.000, 0.898, 0.898}.

The original P1 failure is located entirely on records where the field is undefined: the frozen
extraction form declares the baseline sub-flags as booleans and supplies no level for "could not
be assessed", and the four screeners adopted four different conventions for excluded and
unreachable records (`false`, null, the string `"unclear"`, and `false`). Under the naive reading
in which the string is truthy, Fleiss' κ falls further, to −0.176 [−0.277, −0.091]. The
pre-registered remedy — adjudication, codebook amendment, and re-coding of every already-coded
record — **was executed**, and column (2) shows both floors met. Two qualifications travel with
that column and are not relegated to a footnote: it is a re-encoding of the same four sealed
files rather than an independent re-rating, so a fresh four-screener coding under the amended
codebook remains outstanding; and all 200 reserve records are single-coded, so the reliability of
the pooled sample of 91 rests entirely on these fifteen papers. `split_unit` is the one field
whose disagreement is substantive and survives the amendment (κ 0.637, 9/15 unanimous).

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

**Figure 7. Prevalence screen: flow from frame to included papers.** PRISMA 2020 adapted for a
random-sample meta-research screen. 9,979-record frozen PubMed frame → 250 records sampled at
permutation positions 1–100 and 111–260 → 79 excluded at title and abstract → 171 reports
sought → **44 not retrieved, carried into both bounding analyses rather than excluded** → 127
assessed at full text → 36 excluded with reasons → **91 included and reachable**. The
eligible-looking denominator is 91 + 44 = 135 and S6 = 32.6% [25.3%, 40.9%], above the 15%
threshold at which the protocol makes the bounding interval the headline. The per-block panel
shows that the unreachable rate does not fall as the sample grows. The post-hoc fifth block is
drawn greyed. Every count is produced by `paper/screen/analysis/flow_figure.py` from
`pooled_final.json`.

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
