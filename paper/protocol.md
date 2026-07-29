# An evaluation protocol for slice-labelled 3D medical imaging benchmarks

**Scope.** Any benchmark where a label is attached to individual 2D slices of a 3D
acquisition — MRI, CT, OCT, ultrasound volumes — and performance is reported by pooling
slices. Seven rules. Each one exists because a concrete failure was measured; the
measurement is given under the rule, with the file it lives in. Six of the seven rest on
our own measurements; Rule 1 rests on published ones and says so. A rule with no measured
failure behind it is not in this document.

Nothing here is a new idea. Shortcut learning, acquisition confounding and the
slice-versus-patient distinction are all documented (Geirhos 2020; Badgeley 2019;
Yagis 2021; Wen 2020; Kapoor & Narayanan 2023). What is new is that the amount of
published performance these account for is now measurable from a label file alone, which
makes the rules below cheap enough to be mandatory rather than aspirational.

---

## The protocol

> 1. **Split at the subject level.** State the unit in the same sentence as the split.
> 2. **Report patient level as primary.** Slice level, if reported, is secondary and
>    labelled as such. State *n* patients and *n* positive patients, not just *n* slices.
> 3. **Interval every number with a subject-clustered bootstrap.** Never the slice-level
>    bootstrap.
> 4. **Report the zero-image baselines next to every headline number.** Same table, same
>    metric, same test set, same unit.
> 5. **Publish the positional distribution of the label**, and the position-stratified
>    AUROC alongside the raw one.
> 6. **Test whether acquisition and administrative metadata predict the label**, and
>    report the strongest field with its AUROC.
> 7. **Report the trivial fraction**, including when it is small.

Rules 4, 6 and 7 are one command:

```bash
pip install trivialbaselines
trivial-baselines --labels your_labels.csv --published 0.861
```

Rule 5's remedy metric is `trivialbaselines.stratified_auc`, called on your own test
predictions. Rules 1–3 are things you do, not things you run.

---

## Rule 1 — Split at the subject level, and state the unit

Assign every volume of a subject to exactly one arm. Verify and report that the train and
test subject sets are disjoint. Write the unit down: *"split at the patient level;
218 training / 46 test patients"*, not *"80/20 split"*.

**The failure it prevents.** Slice-level splitting puts adjacent slices of the same
volume on both sides of the split, and the resulting number is a measure of interpolation,
not diagnosis. Yagis et al. (Sci Rep 2021;11:22544) measured the inflation directly:
slice-level cross-validation boosted test accuracy by 30% (OASIS), 29% (ADNI), 48% (PPMI)
and 55% (a local Parkinson's cohort), and on **randomly labelled** data reached ~96%
accuracy under a slice-level split against 50% under a subject-level split.

**This measurement is not ours** — it is cited, not reproduced here — and Rule 1 is the
one rule in this document backed by someone else's numbers. It is included anyway, for a
reason that matters for how the rest of the document is read: **Rule 1 is necessary and
not sufficient.** Every number in Rules 2–7 below was obtained under a *correct,
patient-disjoint* split. Rempe et al.'s split is patient-disjoint, and a zero-image model
still reached 0.851 against their 0.861. Following Rule 1 does not protect you from Rules
2–7.

---

## Rule 2 — Report at the patient level as primary

The clinical question is about a patient. Aggregate slice scores to one score per patient
(state whether by mean or by max; report both if they disagree), compute the metric there,
and give that number the headline position. Report *n* patients and *n* positive patients.
A slice-level number may accompany it, labelled as secondary.

**The failure it prevents.** Slice-level and patient-level AUROC computed from the *same
score vector* on the *same test set* can point in opposite directions.

On Rempe et al. (2024)'s own published prostate DWI label file and split (218 training /
46 test patients; test arm 1395 slices, 83 positive slices, 27 positive patients), a
zero-image positional baseline gives:

| reading of one score vector | AUROC | 95% CI (subject-clustered) |
|---|---|---|
| slice level | **0.851** | [0.816, 0.887] |
| patient level | **0.424** | [0.298, 0.547] |

Published headline on the same benchmark: 0.861 (slice level). The slice-level reading is
indistinguishable from state of the art; the patient-level reading of the identical scores
is below chance. Nothing changed between the two rows except which unit the ranking was
performed at.

The same divergence appears in our own reimplementation of their protocol on our prostate
DWI cache — trained models this time, one score vector each, read at two units:

| arm | slice level | patient level |
|---|---|---|
| magnitude | 0.574 [0.516, 0.629] | 0.524 [0.348, 0.690] |
| magnitude + phase | 0.616 [0.559, 0.672] | 0.528 [0.356, 0.696] |

*Source:* `pipeline_out/rempe/positional_baseline_dwi_labels.json`,
`pipeline_out/trivial_baselines/fastmri_prostate_dwi_published.md`,
`pipeline_out/s12_arm_mag.log` and `pipeline_out/s12_waterfall_magphase.log`
(waterfall rungs W1/W1p/W2/W4).

---

## Rule 3 — Subject-clustered intervals, never the slice-level bootstrap

Resample **subjects** with replacement; a subject drawn twice contributes all of their
slices twice. Report the percentile interval from that. If you also show the slice-level
interval, label it as the incorrect one and say why it is there.

**The failure it prevents.** The slice-level bootstrap treats correlated slices as
independent observations, so it produces an interval that is far too narrow and does not
cover the truth. We measured the coverage directly, on simulated data where the true AUC
is known in closed form (Φ(μ/√(2σ²ᵤ+2σ²ₑ)) = 0.6880; 200 datasets of 20 patients × 15
slices, 500 bootstrap replicates each):

| interval | coverage at nominal 95% | mean width |
|---|---|---|
| subject-clustered bootstrap | **91.5%** | 0.370 |
| naive slice-level bootstrap | **46.5%** | 0.117 |

The wrong interval is 3.2× narrower and misses the truth more often than it catches it.
A 95% interval that covers 46.5% of the time is not a conservative approximation; it is a
different claim than the one being written down.

On real predictions — same predictions, same test set, only the resampling unit changed —
it is sometimes the difference between a finding and no finding, and always a material
widening:

| arm | naive slice interval | subject-clustered interval |
|---|---|---|
| magnitude (AUROC 0.574) | [0.516, 0.629] — excludes 0.5 | [0.489, 0.667] — **includes 0.5** |
| magnitude + phase (AUROC 0.616) | [0.559, 0.672] — excludes 0.5 | [0.528, 0.691] — still excludes 0.5, 1.4× wider |

Both arms are shown because in one the conclusion flips and in the other it does not.
The correct interval is not guaranteed to overturn a result; it is guaranteed to be the
interval you are entitled to quote.

*Source:* `pipeline/s04_stats.py --self-test`, block `[6]`, reproduced 2026-07-29;
`pipeline_out/s12_arm_mag.log` and `pipeline_out/s12_waterfall_magphase.log` (W2/W3).

---

## Rule 4 — Report the zero-image baselines next to every headline number

In the same table as your model, on the same metric, the same test set and the same
evaluation unit, report at minimum: the constant (prevalence) predictor; a positional
baseline, P(label | relative slice position) binned and fitted on training slices only;
and a metadata baseline fitted on acquisition/administrative columns alone.

**The failure it prevents.** Without the baseline row, a reader cannot distinguish a
number that required pixels from a number that did not. On Rempe et al.'s published label
files and splits, using no pixels, no k-space and no phase:

| | slice-level AUROC |
|---|---|
| published headline (gold-standard ADC+trace dual network) | 0.861 |
| published PCA + phase arm | 0.809 |
| published PCA magnitude arm | 0.813 |
| **zero-image positional baseline (DWI label file)** | **0.851** [0.816, 0.887] |
| **zero-image positional baseline (T2 label file)** | **0.854** [0.812, 0.891] |

Published values are transcribed from their Table II, not recomputed.

Every baseline must be fitted on the training rows and scored on the test rows, with the
apparent (training) performance reported next to the test performance so that overfitting
of the *null model* is visible. Each baseline must also be judged against **its own
permutation null**, which is not automatically 0.5: an out-of-fold metadata model on a
subject-level label sits systematically *below* chance, because the rate you fitted is
anti-correlated with the rate you score — positives are a finite population, so a level
that was positive-rich in training is positive-poor in the fold left out. On a synthetic
dataset whose label was by construction invisible to metadata, the metadata baseline
measured **0.424**, not 0.500. Judged against 0.5 that is a below-chance "finding"
manufactured out of arithmetic; judged against its own permutation null it is correctly
reported as no effect.

**What this licenses you to say.** That the *evaluation protocol* certifies a number a
pixel-blind model also reaches. It does **not** say the published model learned nothing —
that is a different claim and a label file cannot support it. We could not reproduce
Rempe et al.'s pipeline: their protocol on our prostate DWI cache gives 0.616 against
their reported 0.809 for the magnitude+phase arm, and 0.574 against their reported 0.813
for the magnitude arm. The only defensible statement we make is about the protocol.

*Source:* `pipeline_out/trivial_baselines/fastmri_prostate_dwi_published.md` and
`..._t2_published.md`.

---

## Rule 5 — Publish the positional distribution of the label, and stratify on it

Ship a histogram of the label rate against relative slice position, computed on the
training set. Then report, next to the raw slice-level AUROC, the **position-stratified**
AUROC: the Mann-Whitney statistic computed within bins of relative position, so only
same-position positive/negative pairs contribute.

**The failure it prevents.** When positives concentrate in the middle of the stack, a
large share of the pairs a slice-level AUROC counts are won by geometry. Stratification
removes exactly that share and nothing else. On the DWI label file, all three numbers from
one zero-image score vector:

| reading | AUROC |
|---|---|
| slice level | 0.851 |
| patient level | 0.424 |
| **position-stratified slice level** (10 bins, 6 populated) | **0.539** |

The positional result is not an artefact of the binning: the bin sweep gives 0.834 (5),
0.842 (10), 0.851 (20), 0.841 (50), and a *no-fit* centrality score — literally
−|relative position − 0.5|, using no training data at all — reaches **0.841**.

The remedy applies to trained models too. Our reimplementation of their protocol scores
0.574 → **0.467** (magnitude arm) and 0.616 → **0.562** (magnitude + phase arm) when
position is held fixed.

```python
from trivialbaselines import position_strata, stratified_auc
stratified_auc(labels, scores, position_strata(relative_position, n_strata=10))
```

*Source:* `pipeline_out/rempe/positional_baseline_dwi_labels.json`
(`slice_auc_position_stratified` = 0.5392, 6 strata; `bin_sweep`; `centrality_no_fit`);
`pipeline_out/s12_arm_mag.log` (W4s = 0.467) and
`pipeline_out/s12_waterfall_magphase.log` (W4s = 0.562).

---

## Rule 6 — Test whether metadata predicts the label

For every acquisition or administrative field you hold — scanner, coil count, matrix
size, TR, institution, file size, **and the release batch or download tarball the volume
arrived in** — fit a model on that field alone, on training subjects, and score the test
subjects. Report the strongest field and its AUROC. Exclude outcome-derived fields (they
are the label under another name) and image-derived fields (they break the zero-image
premise), and list what you excluded.

**The failure it prevents.** An administrative field can outperform the trained network,
which means the network's number cannot be separated from it.

Both columns below are subject-level AUROCs on the same subjects, so they are directly
comparable. The network column is the phase-channel model, pooled out of fold, per seed.

| cohort | field | field predicts label | network predicts label |
|---|---|---|---|
| breast (n=70) | `folder` — the release batch / download tarball, 7 levels | **0.743** | **0.633** (seed 42), 0.630 (seed 123) |
| prostate T2 (n=67) | `kspace_shape` — acquisition matrix, 14 levels | **0.609** | **0.483** (seed 42), 0.462 (seed 123) |

Which tarball a scan was downloaded in has no causal relationship to whether the patient
has cancer. It predicts the label at 0.743 because releases are assembled over time and
enriched differently. In the same cohorts, the field also explains more of the *model's
score variance* than the true label does (breast: η² = 0.108 for `folder` against 0.033
for the label).

*Source:* `python pipeline/s08_belowchance.py --cohort breast --condition phase`, run
2026-07-29; `pipeline_out/s08_belowchance.log` for prostate T2.

---

## Rule 7 — Report the trivial fraction, including when it is small

```
trivial fraction = (best zero-image baseline − chance) / (published − chance)
```

Chance is the value the constant predictor attains under the metric: 0.5 for AUROC, the
test positive rate for average precision. Report the value, its interval, and the limits
below.

**The failure it prevents.** Two benchmarks both reporting 0.86 are not comparable
without it. Measured:

| benchmark | published | best zero-image | trivial fraction |
|---|---|---|---|
| fastMRI prostate DWI label file, vs headline | 0.861 | 0.851 | **0.973** [0.876, 1.073] |
| fastMRI prostate T2 label file, vs headline | 0.861 | 0.854 | **0.981** [0.865, 1.084] |

A fraction above 1 means the zero-image baseline **exceeded** the published number. That
is a real outcome and is reported unclipped. It happens here: against their PCA+phase arm
(0.809) rather than their gold-standard headline, the same DWI baseline gives **1.137**
[1.023, 1.253] — reproduce with `--published 0.809` on the same label file.

**Report it when it is small.** The rule is worthless if only positive audits are
published. Benchmarks on which the nulls fail are reported here as such:

| benchmark | best zero-image baseline |
|---|---|
| LUNA16 false-positive-reduction candidates | **0.539** [0.520, 0.565] |
| our brain confound cohort, positional | **0.480** [0.446, 0.513] |
| our knee confound cohort, positional | **0.500** |
| the synthetic clean control shipped with the tool | trivial fraction **−0.041** [−0.338, 0.304] |

A benchmark on which the null models fail is evidence *for* that benchmark, and it is the
result that makes the positive ones credible.

PI-CAI is the instructive mixed case, and it should be reported as mixed rather than as a
clean win. Evaluated at the case level its authors intended, its **positional** baseline
is exactly **0.500** — there is no stack geometry to exploit, which is precisely what
Rule 2 buys. But its **metadata** baseline still reaches **0.692 [0.626, 0.755]**. Fixing
the reporting unit does not fix acquisition confounding; Rules 2 and 6 are independent and
a benchmark can pass one and fail the other.

**Limits, all of which must travel with the number.**

- The published number must be on the same metric, the same unit and a comparable test
  set. A slice-level baseline against a patient-level publication is meaningless.
- The fraction is undefined when the published number is at or below chance.
- It is **not a decomposition**. The baseline and the published model may exploit the
  same shortcut, different shortcuts, or overlapping ones. A fraction of 0.97 does not
  license "the model learned nothing"; it licenses "this evaluation protocol certifies a
  number that a pixel-blind model also reaches".
- The interval propagates uncertainty in the **baseline only**; the published number
  enters as a fixed constant, so the interval is too narrow as a statement about the ratio.
- The baseline is fitted on the training rows of the same table. If the published model
  was trained on a different or larger set, the comparison is approximate.

*Source:* `pipeline_out/trivial_baselines/fastmri_prostate_dwi_published.md`,
`fastmri_prostate_t2_published.md`, `luna16_fp_reduction_candidates.md`,
`picai_case_level.md`, `phasedx_brain.md`, `phasedx_knee.md`;
`trivial-baselines --labels trivialbaselines/examples/clean_benchmark.csv --published 0.88`.

---

## For dataset publishers

Three fields, in the label file you already release, make every rule above auditable by
anyone — including people who will never be granted the pixels:

1. a subject identifier,
2. a slice index or z position,
3. the official train/test assignment.

Publishing these costs nothing and is the single highest-leverage change a benchmark can
make. PI-CAI already reports at case level and lesion level by design and has no
slice-level number to attack; that is the target.

## What this protocol does not do

It does not detect shortcuts that live in the pixels — scanner-specific texture, burned-in
annotation, body-part framing. Those need the images. It bounds only the part of a
reported number that is reachable *without* them, which is the part that can be checked
for free, at scale, by a third party.

## Limits of the evidence behind these rules

Rules 2, 4 and 5 are anchored on one external benchmark (fastMRI prostate, Rempe et al.
2024) and on our own cohorts, which are small (45–70 subjects), single-institution,
single-vendor and 3 T only. Rule 6's measurements come from those same small cohorts, so
its two examples are two cohorts, not a survey. Rule 7 draws on a wider set of public
label files (LUNA16, PI-CAI and others in `pipeline_out/trivial_baselines/`), but only
fastMRI prostate has a published number we were able to source and match on the same
metric and unit, so it is the only benchmark for which a trivial fraction is quoted.
Rule 1 rests on published work, not on our measurements. Rule 3's coverage figure is a
simulation with a known truth, which is the only setting where coverage *can* be measured.

---

*Companion artefacts: `paper/checklist.md` (one page, for authors and reviewers);
`trivialbaselines/` (the tool: `pip install`, numpy + pandas, no GPU).*
