# Audit results — zero-image null models against published medical imaging benchmarks

Run 2026-07-29, revised later the same day after RSNA ICH was reached and every
comparator's peer-review status was re-checked (§2.3, §3.7, §3.8), and **revised again the
same evening to re-anchor the audit on its peer-review-independent result and to replace
the categorical headline with a continuous one** (§0, §1, §2.0, §2.1, §3.7, §4, §4bis).
Tool: `pipeline/s14_trivialbaselines.py`. **The tool has not been modified in either
revision pass**; its self-test is re-run after each and all checks pass. The `--relpos-col`
option described in §7 was added in the first pass of the day. Label-table preparation
scripts: `pipeline/audit_prep/`. Machine-readable results:
`pipeline_out/trivial_baselines/*.json`, one card per run in `*.md`. Every number added in
a revision pass is reproduced by a logged script under `pipeline_out/audit_logs/`:

| log | produces |
|---|---|
| `rsna_ich_prep.log` | the ordering test and the built ICH slice table (§3.7) |
| `rsna_ich_burduja_conditions.log` | the split-geometry replication, full 752,802-slice file |
| `rsna_ich_s14_card.log` | the s14 harness card, seeded 1,500-patient subsample |
| `rsna_ich_s14_card_full.log` | **the s14 harness card on the FULL 752,802-slice / 18,938-patient file** — bin sweep, metadata baselines, permutation calibration (§3.7) |
| `rsna_ich_unit_collapse.log` | **the flagship: all 18,938 patients, both units, all six labels, independent implementation** (§0, §3.7, §4) |
| `rsna_pe_position_test.log` | the RSNA-STR PE negative (§3.8) |

Two artefacts were added in the second revision pass and are the sources for §0 and §4bis:

| artefact | what it is |
|---|---|
| `pipeline/audit_prep/rsna_ich_unit_collapse.py` → `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json` | a from-scratch recomputation of the slice-to-patient collapse on the full cohort, sharing no code with `s14`: its own fold assignment, its own binning, `sklearn`'s AUROC rather than ours, its own clustered bootstrap |
| `pipeline/audit_prep/trivial_fraction_distribution.py` → `paper/trivial_fraction_distribution.{json,md}` | the trivial fraction over every (benchmark, published comparator) pair, with intervals, summarised as a distribution; re-fits nothing, reads every value from a named artefact |

---

## 0. Headline, stated before any table

### 0.1 The result that needs no published comparator, and it is the largest thing here

On the RSNA 2019 Intracranial Haemorrhage official training file — **752,802 slices,
21,744 series, 18,938 patients** — a model that never sees a pixel produces **one** score
vector, a 20-bin estimate of P(haemorrhage | relative slice position) fitted on the
training slices of a subject-disjoint 5-fold split and applied out of fold. Read that one
vector at two units:

| unit | AUROC | 95% CI (2,000 patient-clustered bootstrap replicates) |
|---|---|---|
| **slice** — the unit this benchmark's own official metric uses | **0.737** | [0.735, 0.740] |
| **patient** — mean-aggregated within patient | **0.453** | [0.445, 0.461] |
| patient — max-aggregated ("take the most suspicious slice") | 0.500 | — |

**Nothing changes between the first two rows except the unit at which the ranking is
performed, and the difference is 0.284 AUROC.** The patient-level interval lies entirely
below 0.5. The same score vector that looks like a working triage tool at the slice level
is worse than a coin toss at the level a patient is actually treated at.

This is the number the paper should lead with, for four reasons.

1. **No published comparator enters it.** Both columns are our own computation on a public
   label file. There is no reproduction dispute available, and no comparator's peer-review
   status can touch it.
2. **It is the whole cohort.** 18,938 patients — roughly 400× the 46-patient prostate test
   arm that carried this claim before, and 12.6× the seeded 1,500-patient subsample the
   earlier revision used.
3. **It was verified rather than carried forward, by four routes.**
   `rsna_ich_unit_collapse.py` shares no code with the released harness — its own fold
   assignment and seed, its own binning, `sklearn`'s AUROC, its own clustered bootstrap.

   | route | cohort | slice AUROC | patient AUROC |
   |---|---|---|---|
   | s14 harness (the released tool), 5-fold subject CV | seeded 1,500-patient subsample | 0.7313 [0.723, 0.739] | 0.4616 [0.431, 0.491] |
   | independent implementation, different folds | same subsample | 0.7311 [0.723, 0.739] | 0.4580 [0.420, 0.486] |
   | **independent implementation** | **all 18,938 patients** | **0.7374 [0.7351, 0.7398]** | **0.4533 [0.4454, 0.4613]** |
   | **s14 harness (the released tool)** | **all 18,938 patients** | **0.7376 [0.7352, 0.7399]** | **0.4561 [0.4478, 0.4640]** |
   | split-geometry replication, 200 re-draws at the published paper's own geometry | all 18,938 patients | 0.7381 [0.727, 0.750] | — |

   Four routes, one answer, agreeing to 0.003 AUROC at both units. The six-label table below
   comes from the independent implementation, which is the only route run on all six labels;
   the harness was run on `any`, and supplies the bin sweep, the metadata baselines and the
   permutation calibration.
4. **It holds on every label, not just the convenient one.** All six official labels
   collapse: `any` 0.737→0.453, epidural 0.712→0.492, intraparenchymal 0.751→0.480,
   intraventricular 0.805→0.497, subarachnoid 0.690→0.485, subdural 0.720→0.476. The gap
   runs 0.205–0.307. [→ `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json`]

Controls, run on the same path: the within-series label permutation null — which destroys
the position–label link and preserves prevalence, clustering and stack depth — sits at
**0.502** at the slice level, so the excess is 0.236 (harness on the same full cohort: null
0.505, excess 0.233). The constant predictor scores 0.492 slice / 0.501 patient in the
independent implementation and **0.498 / 0.500** in the harness. That last number matters: on
the seeded 1,500-patient subsample the harness raised a protocol warning because the constant
predictor scored 0.487, 0.013 off chance, an artefact of pooling out of fold across folds with
different training prevalence. **On the full cohort that deviation falls to 0.002 and the
warning does not fire** — the full-cohort card carries no warning beyond the standing metadata
caveat. Moving to the whole cohort removed the one protocol blemish on this row.

### 0.2 The audit is a distribution, not a verdict count

The audit previously led with a count — six MATCHED, nine PARTIAL, three NOT MATCHED. That
headline is weak in two independent ways. It throws away the information in the PARTIAL
rows, where a benchmark at 0.61 and a benchmark at 0.31 are recorded identically; and it
makes the paper's strength hostage to one threshold and, through it, to one preprint,
because the only rows crossing the MATCHED threshold have a preprint comparator. **The
headline is now the distribution of the trivial fraction; the categorical verdict is kept
as a secondary column and is not deleted.**
[→ `paper/trivial_fraction_distribution.md`]

| set of (benchmark, comparator) rows | n | min | Q1 | **median** | Q3 | max |
|---|---|---|---|---|---|---|
| **peer-reviewed comparator, strongest system per benchmark-arm** | 9 | −0.002 | 0.437 | **0.469** | 0.490 | 0.613 |
| peer-reviewed comparator, all rows | 18 | −0.002 | 0.455 | 0.485 | 0.514 | 0.889 |
| strongest system per benchmark-arm, any comparator | 11 | −0.002 | 0.452 | 0.480 | 0.562 | 0.981 |
| all rows | 24 | −0.002 | 0.469 | 0.512 | 0.910 | 1.655 |
| **preprint comparator (Rempe et al.) only** | 6 | 0.973 | 1.020 | 1.142 | 1.518 | 1.655 |

Read the first line. **Against peer-reviewed numbers, on the same metric and the same
evaluation unit, the median *share* of the published margin over chance that a pixel-blind
model reaches is 0.469 — a little under half.** The distribution is tight rather than diffuse: eight of
those nine rows fall between 0.395 and 0.613, and the ninth is LUNA16 at −0.002. That is a
stronger and far more defensible sentence than "one benchmark matched", and it survives
losing the preprint entirely.

The six RSNA ICH subtype rows are the core of it, because they are against a peer-reviewed
comparator on the identical cohort, metric and unit: 0.395, 0.437, 0.469, 0.490, 0.510,
0.613 against Burduja et al.'s BiLSTM column; 0.410, 0.451, 0.480, 0.500, 0.515, 0.615
against their plain ResNeXt column in the same table. Roughly **40–60% of the published
margin over chance, per subtype, in peer-reviewed work, with no pixels.**

The preprint line is reported at equal prominence and must always be labelled: all six
values above 0.97 come from one arXiv-only comparator, four of them exceed 1 (the other two
are 0.973 and 0.981), and they are the only rows in the audit anywhere near 1. Note also what
those four mean — a fraction of 1.655 is the baseline exceeding Rempe et al.'s weakest
reported arm, not the audit finding 165% of anything.

### 0.3 The rest of the run, unchanged in substance

**RSNA 2019 Intracranial Haemorrhage was reached**, having been recorded as unreachable in
the previous run. It is the benchmark whose official metric is per-slice and whose
organisers stated on the record that the released fields could not determine whether an
image contains haemorrhage. See §3.7.

**A peer-review audit of every comparator (§2.3) found that the only MATCHED rows in the
entire audit rest on a preprint.** Rempe et al. was re-queried on 2026-07-29 and is still
arXiv-only, two years after posting. Meanwhile two comparators were being under-cited:
LUNA16's is a *Medical Image Analysis* 2017 paper and DeepLesion's is a CVPR 2018 paper,
both cited here previously by their arXiv numbers. Stated plainly:

> **Every peer-reviewed comparison in this audit is PARTIAL or NOT MATCHED. The only
> MATCHED rows rest on a preprint comparator.**

That is why the headline moved. It changes the framing the paper can support:

* **The claim "trivial baselines match published performance on medical imaging
  benchmarks" is not supported as a general statement, and is not supported against the
  peer-reviewed literature at all.** It is supported for one benchmark, strongly and
  reproducibly, whose comparator is a preprint. Against peer-reviewed numbers the
  supportable claim is quantitative rather than absolute, and it is §0.2's distribution.
* **What generalises further than the match is the gap between the slice-level and the
  patient-level number**, and that is §0.1. It also appears on fastMRI Prostate T2
  0.854 → 0.506, DWI 0.851 → 0.424, and fastMRI+ knee meniscus 0.873 → 0.510. Duke breast
  reaches 0.823 at the slice level and cannot be evaluated at the patient level at all,
  because all 922 patients are positive — a further way the same protocol problem shows up.
  Crucially none of this depends on whether a published number was matched, so it is not
  hostage to any comparability argument. See the qualification about bin choice in §4.
  **It is not universal.** DeepLesion stays high at both units (pelvis 0.977 slice, 0.954
  patient) because its labels are anatomical regions, and LUNA16 is at chance at both
  (0.534 slice, 0.581 patient). Both exceptions must be reported alongside the rule.
* **Two benchmarks are clean, and this is reported at equal prominence with the hits.**
  LUNA16's positional baseline scores **sensitivity 0.0006 at 1 FP/scan and CPM 0.0020**
  on the challenge's own metric, *below* a random-score reference of 0.0027, against a
  published >0.95 sensitivity — a trivial fraction of **−0.002**, the one value in the
  whole distribution at zero. PI-CAI's positional baseline is **exactly 0.500** at every
  bin setting, which is the correct registration of "inapplicable" rather than a computed
  result, and its best zero-image model reaches 0.692 against a published 0.91 at the unit
  its own authors report. **A measure that always fires measures nothing.** These two rows
  are what make the other nine believable, and they demonstrate that the trivial fraction
  discriminates between benchmarks rather than condemning all of them. See §4bis for what
  that does and does not establish about specificity.
* **Two of the three largest per-slice benchmarks publish a label file from which the
  slice cannot be located.** For RSNA ICH the ordering was recoverable from a third-party
  pixel-free mirror and is verified here by a run-length test (§3.7). For RSNA-STR
  Pulmonary Embolism it was not: the official `train.csv` was obtained and its row order
  *measured* to carry no positional information at all (§3.8). That pair of results is a
  finding about benchmark release practice and should be reported as one.
* **A published location-only baseline already exists on DeepLesion.** Yan et al. (CVPR
  2018) Table 1 reports a "Location feature" baseline at 59.7% 8-class accuracy against
  their own 90.5%. This is closer prior art than `paper/audit_targets.md` §3.4 records,
  and the novelty section must be rewritten to acknowledge it. See §5.

---

## 1. The measure, and the verdict that is now secondary to it

The comparison statistic is the tool's trivial fraction,

> trivial fraction = (best zero-image baseline − chance) / (published − chance)

with chance = 0.5 for AUROC and the majority-class rate for multi-class accuracy. **It is a
continuous measure and is reported as one.** Its interval propagates uncertainty in the
*baseline only*; the published number enters as a constant, so the interval is too narrow
as a statement about the ratio. §4bis exercises the definition at its extremes and states
where it stops meaning anything.

The categorical verdict is retained, unchanged, as a **secondary** column. It is a coarse
summary of the measure, not a substitute for it, and it exists so that this revision can be
reconciled against the previous one rather than replacing it silently.

| verdict | rule |
|---|---|
| **MATCHED** | the upper bound of the baseline's clustered 95% CI reaches or exceeds the published number (equivalently, trivial-fraction CI covers or exceeds 1) |
| **PARTIAL** | trivial fraction ≥ 0.30 but its CI lies wholly below 1 |
| **NOT MATCHED** | trivial fraction < 0.30, or the baseline is statistically indistinguishable from chance |
| **NON-COMPARABLE** | the published number is on a different cohort, split, label definition or metric and could not be reconstructed; no verdict is issued |

A MATCHED verdict licenses exactly one sentence: *this published evaluation protocol
certifies a number that a model with no access to the pixels also reaches.* It does not
license "the model learned nothing". Every card repeats this.

**Why the verdict is being demoted, in one concrete case.**
`trivial_fraction_distribution.py` applies the rule above mechanically, and it disagrees
with the hand-assigned verdicts in the previous revision on exactly two rows: PI-CAI's,
whose trivial fractions are 0.467 and 0.532. Those are ≥ 0.30, so the *written* rule
returns PARTIAL, while §2.1 recorded NOT MATCHED on the strength of a cohort caveat the
rule has no slot for. Both readings are defensible; neither is hidden. The point is that a
threshold-plus-judgement verdict flipped while the underlying fraction did not move at all.
Both are shown in §2.1. That is the argument for leading with the number.

---

## 2. Results — one row per (dataset, published number)

### 2.0 The distribution, before the rows

Generated by `pipeline/audit_prep/trivial_fraction_distribution.py`, which re-fits nothing
and reads every value from a named artefact
[→ `paper/trivial_fraction_distribution.{json,md}`].

**Rows are not independent.** A single benchmark contributes several rows when a paper
reports several systems in one table: Rempe et al.'s Table II has three arms, Burduja et
al.'s Table 3 has two model columns across six labels. The primary reading is therefore the
*strongest published system per benchmark-arm*, which is the conservative choice — a
stronger comparator makes the denominator larger and the fraction smaller.

| set | n | min | Q1 | **median** | Q3 | max | ≤0.05 | 0.30–0.70 | ≥1 |
|---|---|---|---|---|---|---|---|---|---|
| **peer-reviewed comparator, strongest system per benchmark-arm** | 9 | −0.002 | 0.437 | **0.469** | 0.490 | 0.613 | 1 | 8 | 0 |
| peer-reviewed comparator, all rows | 18 | −0.002 | 0.455 | 0.485 | 0.514 | 0.889 | 1 | 16 | 0 |
| strongest system per benchmark-arm, any comparator | 11 | −0.002 | 0.452 | 0.480 | 0.562 | 0.981 | 1 | 8 | 0 |
| all rows | 24 | −0.002 | 0.469 | 0.512 | 0.910 | 1.655 | 1 | 16 | 4 |
| preprint comparator (Rempe et al.) only | 6 | 0.973 | 1.020 | 1.142 | 1.518 | 1.655 | 0 | 0 | 4 |

Verdict counts, kept as the secondary summary they now are:

| set | MATCHED | PARTIAL | NOT MATCHED |
|---|---|---|---|
| all rows | 6 | 17 | 1 |
| **peer-reviewed comparators only** | **0** | 17 | 1 |

The distribution and the verdict count are two summaries of the same 24 numbers. The
distribution is the one that survives deleting the preprint: strike Rempe et al. and the
verdict count loses every MATCHED row it had, while the peer-reviewed distribution does not
move at all, because none of its rows were ever near the threshold.

### 2.1 Scored rows

Two RSNA ICH row-sets are shown. **Rows 16–21 are the primary ones** and were recomputed on
the full cohort in the second revision pass; rows 16v–21v are the split-geometry
replication that was previously primary, retained as a robustness check. They agree to
within 0.002 AUROC.

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
| 11 | PI-CAI | **0.91** (95% CI 0.87–0.94) case-level AUROC, AI system | Saha et al., *Lancet Oncol* 2024;25:879-887, [DOI](https://doi.org/10.1016/S1470-2045(24)00220-1) | **0.692** [0.626, 0.755] metadata CART, case level | 0.467 [0.307, 0.623] | **NOT MATCHED** ¹ (rule: PARTIAL) |
| 12 | PI-CAI | **0.86** (0.83–0.89) case-level AUROC, 62 radiologists PI-RADS 2.1 | same | 0.692 [0.626, 0.755] | 0.532 [0.350, 0.710] | **NOT MATCHED** ¹ (rule: PARTIAL) |
| 16 | RSNA ICH, **any haemorrhage** | **0.9843** slice ROC AUC | Burduja, Ionescu & Verga, *Sensors* 2020;20(19):5611, [DOI](https://doi.org/10.3390/s20195611), Table 3, ResNeXt-101 32×8d + BiLSTM | **0.737** [0.735, 0.740] positional 20-bin | 0.490 [0.485, 0.495] | **PARTIAL** ² |
| 17 | RSNA ICH, epidural | 0.9851 slice ROC AUC | same table, same row | 0.712 [0.700, 0.725] | 0.437 [0.411, 0.464] | **PARTIAL** ² |
| 18 | RSNA ICH, intraparenchymal | 0.9927 slice ROC AUC | same | 0.751 [0.747, 0.755] | 0.510 [0.502, 0.518] | **PARTIAL** ² |
| 19 | RSNA ICH, intraventricular | 0.9970 slice ROC AUC | same | **0.805** [0.802, 0.808] | 0.613 [0.607, 0.620] | **PARTIAL** ² |
| 20 | RSNA ICH, subarachnoid | 0.9821 slice ROC AUC | same | 0.690 [0.686, 0.695] | 0.395 [0.386, 0.404] | **PARTIAL** ² |
| 21 | RSNA ICH, subdural | 0.9682 slice ROC AUC | same | 0.720 [0.717, 0.723] | 0.469 [0.463, 0.476] | **PARTIAL** ² |

*Second published system in the same table.* Burduja et al.'s Table 3 also reports a plain
ResNeXt-101 32×8d without the bidirectional LSTM, on the same metric, unit and cohort.
Against that column the six fractions are, in the same order: **0.500, 0.451, 0.515, 0.615,
0.410, 0.480**. Both columns are reported for the same reason rows 1–3 report three arms of
Rempe et al.'s table; the BiLSTM column is treated as primary because it is the stronger
system and therefore the harder denominator.

*Split-geometry replication (previously primary, now a robustness check).*

| # | dataset | our zero-image baseline, their 744-scan split geometry | trivial fraction | agreement with the primary row |
|---|---|---|---|---|
| 16v | RSNA ICH, any | 0.738 [0.727, 0.750] | 0.492 | +0.001 |
| 17v | RSNA ICH, epidural | 0.719 [0.649, 0.776] | 0.451 | +0.006 |
| 18v | RSNA ICH, intraparenchymal | 0.753 [0.732, 0.772] | 0.513 | +0.001 |
| 19v | RSNA ICH, intraventricular | 0.806 [0.791, 0.820] | 0.615 | +0.001 |
| 20v | RSNA ICH, subarachnoid | 0.692 [0.665, 0.710] | 0.398 | +0.002 |
| 21v | RSNA ICH, subdural | 0.721 [0.707, 0.735] | 0.472 | +0.002 |

¹ Rows 11–12 carry a cohort caveat that would justify calling them NON-COMPARABLE; see
§3.5. They are scored as NOT MATCHED because the caveat cuts *against* the null (our
baseline had the easier cohort and still lost), so scoring them is the conservative
choice. **The written verdict rule in §1, applied mechanically, returns PARTIAL for both**
— 0.467 and 0.532 are ≥ 0.30 — and both readings are shown here rather than one being
quietly preferred. The trivial fraction itself is identical under either. Rows 7–9 use the
majority class (0.236) as the chance anchor, not 0.5.

² **Rows 16–21 were recomputed on the full cohort in the second revision pass and their
interval is now the same object as rows 1–12's.** They come from
`pipeline/audit_prep/rsna_ich_unit_collapse.py` on all 752,802 slices / 18,938 patients:
5-fold subject-disjoint CV, pooled out of fold, 95% percentile interval from 2,000
bootstrap replicates **resampling patients**, so a patient drawn twice contributes all of
their slices twice. The naive slice-resampled interval is reported alongside in the JSON
and is 1.5–2.0× too narrow on these six rows, which is the concrete size of the error Rule
3 of the protocol exists to prevent.

Rows 16v–21v are the earlier estimator and are retained because they match the published
paper's split *geometry* rather than ours: Burduja et al. publish that geometry (744
held-out scans, 24,290 slices) but not the draw, so that script repeats the draw 200 times
and its interval is split-to-split spread, **not** sampling error. The two estimators
differ by at most 0.006 in trivial fraction. Two further routes agree: the s14 harness run on
the **full** file gives 0.7376 [0.7352, 0.7399] and a trivial fraction of 0.491 [0.486, 0.495]
on the `any` arm [→ `rsna_ich_any_slice_full.json`], and the same harness on a seeded
1,500-patient subsample gives 0.731 [0.723, 0.739]. Four routes, one answer — see §3.7 for the
table.

### 2.2 Non-comparable rows — audited, but no defensible published comparator

| # | dataset | zero-image result | why no verdict |
|---|---|---|---|
| 13 | fastMRI+ knee, "Meniscus Tear" per slice | positional 20-bin **0.873** [0.858, 0.886] slice AUROC; **0.510** [0.428, 0.592] patient | fastMRI+ is a data descriptor; no published slice-level classification number was located, and `paper/audit_targets.md` §2.3 already flags this as open. Also 199 of the 1,173 roster volumes (§3.3). |
| 14 | fastMRI+ knee, "any annotated finding" per slice | positional 20-bin **0.801** [0.779, 0.824] slice; **0.558** [0.470, 0.648] patient | as above |
| 15 | Duke Breast Cancer MRI, owner-defined slice task | positional 20-bin **0.823** [0.811, 0.834] slice AUROC; patient AUROC **undefined** | the Mazurowski lab tutorial defines the task but publishes no metric. No downstream number with the same task definition was located. |

### 2.3 Peer-review status of every comparator — checked one by one on 2026-07-29

The audit's rows are only as good as the numbers they are compared against. Each
comparator was re-checked against the literature databases on the date of this run, not
taken on trust from the earlier session's notes. Three corrections came out of it, two
in our favour and one against.

| benchmark | comparator | venue | peer-reviewed? | same metric? | same unit? | split condition | counts as a benchmarked row? |
|---|---|---|---|---|---|---|---|
| fastMRI Prostate T2 / DWI (rows 1–6) | Rempe et al., arXiv:2407.06165 | arXiv | **NO** | yes, slice AUROC | yes, slice | their own in-file split | **NO — fails the peer-review test** |
| DeepLesion (rows 7–9) | Yan et al., CVPR 2018 | IEEE/CVF CVPR proceedings | **yes**, conference not journal | yes, 8-class accuracy | yes, lesion | their partition reconstructed, 200 draws | **YES** |
| LUNA16 (row 10) | Setio et al., *Medical Image Analysis* 2017;42:1–13 | Elsevier journal | **yes** | yes, sensitivity at FP/scan | yes, candidate | approximate, stated in §3.6 | **YES** |
| PI-CAI (rows 11–12) | Saha et al., *Lancet Oncol* 2024;25:879–887 | journal | **yes** | yes, case AUROC | yes, case | **different cohort**, stated in §3.5 | **YES, weakest of the four** |
| RSNA ICH (rows 16–21, new) | Burduja, Ionescu & Verga, *Sensors* 2020;20(19):5611 | MDPI journal, PubMed-indexed, CC BY | **yes** | yes, slice ROC AUC | yes, slice | same file, split geometry published and replicated, stated in §3.7 | **YES** |
| fastMRI+ knee (rows 13–14) | none located | — | — | — | — | — | no |
| Duke breast (row 15) | none located | — | — | — | — | — | no |
| RSNA PE | see §3.8 | — | yes, but **exam-level on a non-public private test set** | **no** | **no** | — | no |

**Two corrections in our favour.** The audit previously cited LUNA16's comparator as
`arXiv:1612.08012` and DeepLesion's as `arXiv:1711.10535`. Both are the preprints of
peer-reviewed papers, and both should be cited as the published versions: Setio et al.
appeared in *Medical Image Analysis* 2017;42:1–13
([DOI](https://doi.org/10.1016/j.media.2017.06.015)), whose abstract carries the exact
claim the audit compares against, verbatim — *"The combination of these solutions
achieved an excellent sensitivity of over 95% at fewer than 1.0 false positives per
scan."* Yan et al. appeared at CVPR 2018; DBLP records it as a conference paper with the
arXiv entry as the separate preprint. So two rows that looked preprint-backed are in fact
journal- and proceedings-backed.

**One correction against us, and it is the important one.** Rempe et al. was re-queried
on the arXiv API on 2026-07-29: no `journal_ref`, no DOI, no Europe PMC record, two years
after posting. It is still a preprint. Since rows 1–6 are the **only MATCHED rows in the
entire audit**, the consequence has to be stated without softening:

> **Every peer-reviewed comparison in this audit is PARTIAL or NOT MATCHED. The only
> MATCHED rows rest on a preprint.**

That is a materially different paper from the one the earlier draft implied. It does not
weaken the audit — it relocates the claim. What survives peer-reviewed comparison is
*"a pixel-blind model reaches a large and quantified fraction of published slice-level
performance"*, not *"a pixel-blind model matches it"*. The matching result remains real
and reportable, but must be labelled as resting on a preprint comparator every time it
appears.

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
* **Which arm is theirs: DWI. Settled 2026-07-30 from the paper's own text.** An earlier
  revision of this section argued for T2 from the slice count, and was wrong. Rempe et
  al.'s section IIIC says verbatim: *"The dataset comes with k-Space data of both T2 and
  DWI. To show the feasibility of our approach, we only work on the DWI data, as it needs
  more extensive post-processing steps."* Three further details in the same section are
  DWI-only — matrix 100 × 100 with a 200 mm FOV, the b=50/b=1000/b=0 direction-and-average
  counts, and the GRAPPA comparison, which is needed because fastMRI prostate DWI ships 2×
  undersampled. **DWI is the correct arm.** `paper/audit_targets.json`'s
  `anchor_correction` block and the waterfall docstring at `pipeline/s12_rempe.py:272-278`
  are both right as written; no reversal is needed.

  The slice count that misled the earlier revision is a real discrepancy, but it is in
  *their* reporting, not ours. Their abstract and section IIIC both give the cohort as
  "312 subject and a total of 9508 slices". 9,508 is the row count of
  `t2_slice_level_labels.csv`; the diffusion file they state they use has 9,490 rows. Both
  files cover the same 312 patients, so the patient count cannot discriminate between them.
  The manuscript therefore leads with the DWI arm and reports the T2 arm beside it, naming
  this discrepancy as the reason both are shown.

### 3.2 DeepLesion (rows 7–9) — their conditions were reconstructed, not assumed

The first attempt at this row would have been wrong and is worth recording. Yan et al.'s
Table 1 test set has 4,927 samples, which is *exactly* the row count of DeepLesion's
official `Train_Val_Test == 3` split — a coincidence that invites a false match. Their
own text says otherwise, verbatim: *"Among the labeled samples, we randomly select 25% as
training seeds to predict pseudo-labels, 25% as the validation set, and the other 50% as
the test set. There is no patient-level overlap between all subsets."*

So the reported row is **not** the official-split number.
`pipeline/audit_prep/deeplesion_yan_conditions.py` rebuilds their partition — a random patient-disjoint
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

### 3.7 RSNA 2019 Intracranial Haemorrhage (rows 16–21) — the blocker was removed, and the organisers' claim is now refuted with a number

The previous run recorded this benchmark as unreachable because `stage_2_train.csv` is
keyed by `ID_<SOPInstanceUID>_<subtype>` and carries no patient id, no study id and no
slice position. **That blocker is gone.** The join is recoverable from a public,
MIT-licensed, pixel-free tabular file.

* **Source.** `ianpan/rsna-intracranial-hemorrhage-16bit-png` on HuggingFace, a derived
  mirror of the Kaggle release. Two tabular files were downloaded: `slice_labels.csv`
  (61,729,905 bytes, sha256 `72885546ba8f55fb…`) and `rescale_values.csv` (1,294,907
  bytes, sha256 `ca1d0d583ed6a41d…`). **`images.tar` (144 GB) was not downloaded and no
  pixel data was touched.** The repo is MIT-licensed.
* **The file is the official training set.** 752,802 rows, 18,938 patients, 21,744
  series, slice prevalence 0.14338. Burduja et al. describe the official training set as
  752,803 slices in 21,744 scans. The series count matches exactly and the slice count is
  one short — consistent with the one series (of 21,744) whose `IM` counter is not
  contiguous, i.e. a single dropped file. That is a known feature of this release, not a
  silent subsetting, but it is one row we cannot account for and it is recorded here as
  such.
* **The whole row rests on one claim, and the claim is tested rather than assumed.** The
  filename encodes `IM%06d`, a counter within the series. Everything depends on that
  counter being the anatomical z-order. `prep_rsna_ich.py::verify_ordering` tests it
  falsifiably: a haemorrhage is a spatially contiguous object, so a z-ordered index makes
  positive slices form very few runs, while an arbitrary index makes them form as many
  runs as random placement predicts. **Observed 1.384 runs per series against 7.167
  expected under random placement; shuffling the labels within each series returns 7.170,
  recovering the random expectation to three decimal places.** The index is z-ordered.
  The test re-runs on every invocation and the script aborts if it stops holding. It
  fixes order but not orientation — whether `IM000001` is the vertex or the skull base is
  undetermined, and does not matter, because the positional baseline is invariant to
  reversing the axis and is applied per series.
* **This is the benchmark whose organisers said it could not be done.** On the record:
  *"the available fields do not contain information that can determine if an image
  contains intracranial hemorrhage"* — and the fields include ImagePositionPatient. A
  20-bin estimate of P(haemorrhage | relative slice position), fitted on training scans
  and applied to held-out scans, reaches **slice AUROC 0.738** on the `any` label. That
  is not a determination of every image, and the organisers' sentence should be read as a
  claim about individual certainty rather than about aggregate rankability. But as a
  statement about what the metadata carries it is quantitatively wrong, and this is the
  cleanest foil in the paper because the number is computed on the organisers' own file.
* **Comparability to Burduja et al. is unusually good, and approximate in exactly one
  respect.** Same cohort (the identical official training file), same label (`any`), same
  metric (ROC AUC), same evaluation unit (the slice), and they publish the geometry of
  their split verbatim: *"We further split the official training data into a training set
  of 728,513 slices and a validation set 24,290 slices… the training set contains 21,000
  CT scans, the validation set contains 744 CT scans."* What they do not publish is the
  draw. So the audit repeats the draw 200 times at their geometry and reports the
  distribution. **The comparison is approximate in the choice of held-out scans and in
  nothing else**, and the spread it induces is sd 0.006.
* **Their split leaks patients; it does not matter, and we checked rather than assumed.**
  Their 21,744 scans come from 18,938 patients, so a scan-level split puts some patients
  in both arms and the null benefits. Re-running patient-disjoint gives 0.7376 against
  0.7381 — a difference of 0.0005. Both are reported by the script; the audit quotes the
  scan-level number because that is their protocol, and the patient-disjoint number is
  the honest one, and here they agree.
* **A stronger-venue comparator was sought and rejected on the merits.** The obvious
  candidate was Wu Y, Iorga M, …, Hill VB, *Radiology: Artificial Intelligence*
  2024;6:e230296,
  [DOI](https://doi.org/10.1148/ryai.230296), which is explicitly about image-level ICH
  localisation. It does not qualify: the RSNA dataset is used only for **pretraining**,
  and its AUC of 0.96 is reported on a local held-out cohort of 7,243 scans and an
  external set of 491, neither of which we hold. Comparing against it would have been the
  cross-cohort error this audit exists to refuse. *Sensors* is an MDPI journal and some
  reviewers discount MDPI; it is nonetheless the correct comparator here, because it is
  the one located paper reporting slice-level ROC AUC on this exact cohort with a
  published split geometry. It is PubMed-indexed, CC BY, and its Table 2 also reports the
  official competition metric, placing it, in their words, "in the top 30 ranking from a total of 1345
  participants" — so it is not a
  weak system being cherry-picked as an easy target.
* **The full cohort is now used for the headline, twice, and the number was verified rather
  than carried forward.** The first pass ran the s14 harness on a seeded 1,500-patient
  subsample because the full 752,802-row file drove the machine into swap exhaustion. The
  second pass did two things. It wrote a **separate implementation**,
  `pipeline/audit_prep/rsna_ich_unit_collapse.py`, sharing no code with `s14`: its own fold
  assignment (balanced on the patient-level label, seed 20260729 rather than 0), its own
  binning, `sklearn.metrics.roc_auc_score` rather than our midrank routine for the point
  estimates, and its own patient-clustered bootstrap. Inside that bootstrap the AUROC is
  evaluated from per-patient count vectors over the distinct score values instead of
  re-ranking 750k rows on every replicate — exact midrank arithmetic, not an approximation,
  asserted equal to the `sklearn` value before any resampling is done — which is what makes
  2,000 replicates on the whole cohort cheap. And it then **re-ran the s14 harness itself on
  the full file**, successfully, in 30 minutes at a peak of 2.0 GB resident
  [→ `pipeline_out/trivial_baselines/rsna_ich_any_slice_full.json`;
  `pipeline_out/audit_logs/rsna_ich_s14_card_full.log`]. The earlier swap exhaustion was a
  machine-state problem, not a property of the file, and the note claiming otherwise is
  withdrawn.

  | route | cohort | slice | patient |
  |---|---|---|---|
  | s14 harness, 5-fold subject CV | seeded 1,500-patient subsample | 0.7313 [0.723, 0.739] | 0.4616 [0.431, 0.491] |
  | independent implementation, 5-fold subject CV | same subsample, different folds | 0.7311 [0.723, 0.739] | 0.4580 [0.420, 0.486] |
  | **independent implementation**, 5-fold subject CV, 2,000 boot | **all 18,938 patients** | **0.7374 [0.7351, 0.7398]** | **0.4533 [0.4454, 0.4613]** |
  | **s14 harness**, 5-fold subject CV, 1,000 boot | **all 18,938 patients** | **0.7376 [0.7352, 0.7399]** | **0.4561 [0.4478, 0.4640]** |
  | split-geometry replication, 200 re-draws | all 18,938 patients | 0.7381 [0.727, 0.750] | — |

  Four routes, agreeing to 0.003 at both units. The subsample was not misleading; the full
  cohort tightens the intervals by about 3× and moves the patient-level number further below
  chance. The six-label table in §4 comes from the independent implementation, the only route
  run on all six labels.
* **The patient-level interval now excludes chance.** On the subsample it was [0.431,
  0.491]; on the full cohort it is [0.445, 0.461] (independent) / [0.448, 0.464] (harness),
  entirely below 0.5 either way. The max-aggregated reading — "score the patient by their
  most suspicious slice", which is what a triage system would actually do — is **0.5000**,
  i.e. exactly chance, in both. Both aggregations are reported.
* **Not a binning artefact, on the full cohort.** Bin sweep on the slice unit: 5→0.716,
  10→0.733, 20→0.738, 50→0.745, and a fit-free centrality score −|relpos − 0.5| reaches
  **0.735** with no training data at all. Apparent (training) slice AUROC is 0.7379 against a
  held-out 0.7376, so the null model is not itself overfitting.
* **Permutation-calibrated on the full cohort.** The positional baseline's own null — labels
  shuffled *within each series*, which destroys the position–label link and preserves
  prevalence, subject clustering, series depth and all metadata — sits at **0.5047** over 20
  draws (harness; range 0.5023–0.5065) and **0.5018** in the independent implementation, so
  the excess is 0.233–0.236. **The protocol warning that fired on the subsample no longer
  fires:** the constant predictor scored 0.487 there (0.013 off chance, from pooling out of
  fold across folds of differing training prevalence) and scores **0.4981** on the full
  cohort, a deviation of 0.002. The full-cohort card carries no warning beyond the standing
  metadata caveat.
* **The permutation null at the patient level is above 0.5, not at it, and that matters.**
  It is 0.523 (independent) / 0.561 (harness) for `any`. So the patient-level 0.453–0.456 is
  not merely below chance, it is 0.07–0.10 below the value the *same estimator* reaches when
  there is nothing to find. The positional baseline is anti-predictive at patient level:
  patients with more mid-stack slices are not the patients with haemorrhage. This is also why
  the harness reports each baseline against its own permutation null rather than against 0.5.
* **Acquisition metadata is nearly inert here, unlike DeepLesion.** On the full cohort the
  metadata tree reaches 0.524 slice / 0.534 patient against a null of 0.498. The only
  metadata column that carries anything is the number of slices in the series (0.572 slice /
  0.591 patient); `rescale_intercept` gives 0.538/0.559; `plane` and `rescale_slope` are
  single-valued in this cohort and score exactly 0.498/0.500. The combined position+metadata
  tree reaches 0.718 slice — *below* position alone at 0.738 — against a null of 0.718, an
  excess of exactly zero. That null is the metadata-block permutation, which scrambles
  metadata while leaving the position–label link intact, so a zero excess is the harness
  correctly registering that **metadata adds nothing to position on this benchmark**, and
  that the combined tree's 0.718 is the positional signal alone, slightly degraded by
  spending tree depth on inert columns.

### 3.8 RSNA-STR Pulmonary Embolism 2020 — NOT REACHED, and the reason is now a measurement rather than a guess

This was the priority target. It is not reached, and the negative is sharper than the
previous "we could not get the file" because the file was obtained and then shown to be
insufficient.

* **The official label file was obtained without Kaggle credentials.** `train.csv` is
  committed to a public GitHub repository, `darraghdog/rsnastr`, at `data/train.csv.zip`
  (16,512,900 bytes zipped; 119,970,071 bytes expanded, dated 2020-09-07). It is
  genuine: 1,790,594 rows, 7,279 studies, `pe_present_on_image` prevalence 0.05392. Those
  three numbers match Table 4 of Hu Z, Lin HM, …, Colak E, *npj Digital Medicine*
  2025;8:254 ([DOI](https://doi.org/10.1038/s41746-025-01594-2)) exactly — it reports the
  RSPECT train split as 96,540 positive of 1,790,594 slices, 5.39%, over 7,279 exams.
* **Requirement (a) fails, and this was measured.** `train.csv` carries
  StudyInstanceUID, SeriesInstanceUID, SOPInstanceUID and the labels, and **no slice
  index and no z position**. The obvious hope is that the file's row order preserves
  acquisition order. It does not. Applying the same run-length test used on RSNA ICH:
  observed 31.090 runs of positive slices per series against 31.913 expected under random
  placement, a ratio of **0.974**. Sorting by SOPInstanceUID instead gives 31.938, ratio
  **1.001**. **Neither the row order nor the identifier order carries any positional
  information whatsoever.** No public pixel-free source of RSNA PE slice positions was
  located (HuggingFace, Zenodo, and the twelve public solution repositories were
  checked; the second-place solution ships a metadata-extraction *script* that requires
  the DICOMs, and the top solution ships only image-derived lung bounding boxes, which
  would break the zero-image guarantee).
* **Requirement (b) fails independently.** The peer-reviewed numbers located on this
  benchmark are at the **examination level**, not the slice level, and are computed on
  the **private test set, whose labels were never released**. Hu et al.'s AUCs of 0.928
  (semi-weak, 27.5% of slice labels) and 0.932 (fully supervised) are exam-level on the
  RSPECT private split (385,238 slices, of which 18,846 positive). So even with
  slice positions in hand, there would be nothing on the same metric and unit and cohort
  to compare against.
* **So RSNA PE fails (a) and (b) for independent reasons**, and the descriptor paper
  (Colak et al., *Radiology: Artificial Intelligence* 2021;3:e200254) is a dataset
  descriptor that publishes no per-slice classification metric.
* **This is worth a sentence in the paper, not a footnote.** RSNA ICH and RSNA PE are the
  two largest per-slice-labelled benchmarks in medical imaging. Both publish a label file
  from which the slice cannot be located. For ICH a third party reconstructed the
  ordering and we could test and use it; for PE nobody has, and we can now demonstrate
  the absence rather than assert it. OsciiArt's 2020 notebook (the prior art recorded in
  `paper/audit_targets.md`) did this on PE with the DICOM headers in hand, which is
  precisely the resource a label-file-only auditor does not have.

---

## 4. Slice-level versus patient-level, measured on every benchmark

This table is the paper's real result. Every cell is our own computation on a published
label file, so it depends on no published number and none of the comparability objections
in §3 apply to it. It is reported for all eight dataset-arms, including the two that do
not show the effect.

**The RSNA ICH block is the headline and is reported first, on the full cohort, at both
units with the same estimator and the same interval construction.**
[→ `pipeline_out/trivial_baselines/rsna_ich_unit_collapse.json`;
`pipeline_out/audit_logs/rsna_ich_unit_collapse.log`]

| RSNA ICH label | slice prev. | patient prev. | **slice** AUROC | **patient** AUROC (mean-agg) | patient (max-agg) | gap |
|---|---|---|---|---|---|---|
| **any** | 0.1434 | 0.4041 | **0.7374** [0.7351, 0.7398] | **0.4533** [0.4454, 0.4613] | 0.5001 | **0.284** |
| epidural | 0.0042 | 0.0164 | 0.7122 [0.6996, 0.7249] | 0.4921 [0.4613, 0.5244] | 0.4856 | 0.220 |
| intraparenchymal | 0.0480 | 0.2468 | 0.7514 [0.7474, 0.7553] | 0.4804 [0.4708, 0.4900] | 0.5025 | 0.271 |
| intraventricular | 0.0348 | 0.1744 | 0.8048 [0.8017, 0.8081] | 0.4974 [0.4870, 0.5077] | 0.4948 | 0.307 |
| subarachnoid | 0.0474 | 0.1822 | 0.6905 [0.6859, 0.6948] | 0.4855 [0.4749, 0.4960] | 0.5045 | 0.205 |
| subdural | 0.0627 | 0.1782 | 0.7195 [0.7166, 0.7227] | 0.4764 [0.4657, 0.4870] | 0.4996 | 0.243 |

752,802 slices, 21,744 series, **18,938 patients**; 5-fold subject-disjoint CV pooled out of
fold; 95% percentile intervals from 2,000 bootstrap replicates resampling **patients**. One
score vector per row, read at three units. Four of the six patient-level intervals lie
wholly below 0.5; the other two contain it; none reaches 0.55. The max-aggregated column
is the reading a deployed triage tool would use, and it runs 0.486–0.505 across the six
labels — chance, on every one of them.

*The naive slice-resampled interval, which this audit computes and refuses to use:* on the
`any` row it is [0.7360, 0.7387], width 0.0027, against the clustered width of 0.0048. The
wrong interval is **1.76×** too narrow here, and 1.51–1.99× across the six labels. That is
the concrete size of the error Rule 3 of the protocol exists to prevent, measured on a
published label file rather than argued from theory.

The remaining benchmarks, on their own cards:

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

Three benchmarks show the collapse outright — **RSNA ICH on all six of its labels**, fastMRI
Prostate (both arms) and fastMRI+ knee (both label definitions). Duke breast is a fourth
variant of the same protocol problem: 0.823 at the slice level, and no patient-level number
is computable at all because every patient in the cohort is positive. DeepLesion does not
collapse, and should not: its labels are anatomical regions, so they *are* patient-level
facts about where lesions were found, and position predicts them at both units. LUNA16 is
at chance at both units. PI-CAI has no slice-level structure to collapse. Stating all six
outcomes is what makes the first three credible.

**RSNA ICH is the single most useful result in this document** and it should lead the paper.
The other collapses are on cohorts of 46 to 199 subjects. This one is on 18,938 patients —
roughly 400× the prostate test arm — on the benchmark whose official metric is per-slice and
whose organisers stated on the record that the metadata could not do this. **It needs no
comparability argument and no published comparator at all**: every column is our own
computation on one public label file, so no reproduction dispute and no comparator's
peer-review status can reach it. It is also the only row here that has been computed twice
by two implementations sharing no code (§3.7).

**One honest qualification on the ICH patient-level number, confirmed rather than removed by
the full-cohort rerun.** The collapse is bin-robust at the slice level (0.716→0.745 over 5→50
bins on the full cohort) but *not* at the patient level, where the full-cohort sweep runs
5→0.437, 10→0.445, 20→0.456, 50→**0.632** [→ `rsna_ich_any_slice_full.json`,
`positional_bin_sweep`; the subsample gave 0.425 / 0.435 / 0.462 / 0.652]. At 50 bins over
volumes of 20–60 slices the bin index is nearly the raw slice index, so the per-patient
aggregate starts tracking volume length, and volume length is itself weakly predictive here
(0.591 patient AUROC on its own, full cohort). The honest statement is that the *positional*
signal collapses at the patient level while a separate and weaker *volume-size* signal does
not, and that the 50-bin patient number is measuring the latter. The 20-bin setting is the
pre-specified one and is what the tables report. Any paper quoting the patient-level collapse
must quote this sweep with it.

The remedy column is the constructive half. Stratifying the slice-level AUROC by
relative position collapses the fastMRI Prostate null from 0.854 to 0.546 and from 0.851
to 0.539 — within noise of chance. That is the metric the paper should ask reviewers to
require.

---

## 4bis. The trivial fraction as a measure: specificity, extremes, and where it stops meaning anything

### 4bis.1 It discriminates — the non-firing rows are evidence, not an appendix

A measure that returns a large number on every benchmark measures nothing, and the two
benchmarks where the zero-image family does not fire are therefore reported at the same
prominence as the ones where it does.

**LUNA16 is the strongest negative.** Scored on the competition's own metric rather than a
convenient one, the identical 20-bin positional estimator reaches **sensitivity 0.0006 at 1
false positive per scan and CPM 0.0020**, against a random-score reference CPM of **0.0027**
and a published combined-solution sensitivity above 0.95. The trivial fraction is
**−0.002**: the baseline is not merely worse than the published system, it is *below the
random-score reference on the published scale*. As an AUROC on 754,975 candidates from 888
scans it is 0.534 at the slice level and 0.581 at the patient level — and, tellingly, it
does **not** collapse between units, because there is nothing there to collapse.

**PI-CAI's positional baseline is exactly 0.500 at every bin setting.** That is the correct
registration of "inapplicable" — the marksheet has one row per case and no slice index — not
a computed near-chance result, and the audit says so rather than reporting 0.500 as a
finding. Its metadata CART reaches 0.692 [0.626, 0.755] against a published 0.91, a fraction
of 0.467. **PI-CAI is the paper's positive example**: a benchmark evaluating at the unit it
should, with no slice-level number to attack.

Three things follow, and only three.

1. **The measure has a floor and can reach it.** One of nine peer-reviewed head-to-heads
   sits at −0.002. The other eight cluster in 0.395–0.613. That spread is what makes the
   eight credible.
2. **The two behaviours are not the same kind of null.** LUNA16 is a benchmark where
   position genuinely carries nothing about the label. PI-CAI is a benchmark that does not
   expose the axis the baseline needs. Reporting both as "NOT MATCHED" loses that
   distinction; reporting the fractions and the reasons keeps it.
3. **This is not a validation of the measure's specificity in any statistical sense.** Two
   negatives out of nine rows is an existence proof that it can fail to fire, nothing more.
   No claim about false-positive rate is made or supportable from n = 9.

Two further rows behave the way a well-behaved measure should and are worth naming for the
same reason. DeepLesion's fraction against Yan et al.'s own image-derived location baseline
is **0.889** — our pixel-free position nearly reproduces their image-derived position, which
is exactly what should happen and is a sanity check rather than a scandal. And the fraction
against their full method is 0.480, i.e. the measure separates "reproduces the location
baseline" from "reproduces the published system" on the same benchmark, at the same time.

### 4bis.2 The definition at its extremes, exercised rather than asserted

Run on the tool's own `trivial_fraction()`; the full table is in
`paper/trivial_fraction_distribution.md` and the JSON, with baseline CI = baseline ± 0.02
throughout.

| case | baseline | chance | published | value | clipped | behaviour |
|---|---|---|---|---|---|---|
| published far above chance | 0.7374 | 0.500 | 0.9843 | 0.4902 | 0.4902 | the ordinary case |
| **published just above chance** (headroom 0.021) | 0.600 | 0.500 | 0.521 | **4.7619** | 1.0000 | **unstable — see below** |
| published exactly at chance | 0.600 | 0.500 | 0.500 | **undefined** | — | refused, with a reason string |
| published *below* chance | 0.600 | 0.500 | 0.450 | **undefined** | — | refused |
| baseline above published | 0.854 | 0.500 | 0.714 | 1.6542 | 1.0000 | reported unclipped in `value` |
| baseline exactly at chance | 0.500 | 0.500 | 0.900 | 0.0000 | 0.0000 | exact zero |
| baseline *below* chance | 0.4533 | 0.500 | 0.9843 | −0.0964 | 0.0000 | negative, kept |
| baseline equals published | 0.900 | 0.500 | 0.900 | 1.0000 | 1.0000 | exact one |
| published near-perfect | 0.7374 | 0.500 | 1.0000 | 0.4748 | 0.4748 | stable |
| non-AUROC anchor (AP, prevalence 0.1434) | 0.260 | 0.1434 | 0.600 | 0.2554 | 0.2554 | anchor moves with the metric |

**It behaves correctly at both ends the audit actually visits.** A published number at or
below chance yields `None` with an explanatory reason rather than a huge or negative number;
a baseline that exceeds the published number yields a value above 1 that is left unclipped
in `value`, with the clipped copy kept separately for plotting only; a baseline below chance
yields a negative value that is likewise kept. Nothing is silently coerced.

**The one real fragility is a small denominator, and it must be stated as a limit.** At a
published number 0.021 above chance the fraction is 4.76 with an interval entirely above 1 —
arithmetically correct and practically meaningless, because dividing by 0.021 amplifies
everything. The tool guards this with `min_headroom = 0.02`, but that threshold is arbitrary
and, being a floating-point comparison, 0.521 slips past it by one unit in the last place.
**In this audit the guard is never load-bearing**: the smallest denominator anywhere in the
24 rows is **0.214** (fastMRI Prostate against Rempe et al.'s R = 16 arm, published 0.714
against an AUROC anchor of 0.5), ten times the guard threshold; the smallest among the
peer-reviewed rows is 0.36 (PI-CAI, published 0.86) and the smallest non-AUROC one is 0.361
(DeepLesion, published 0.597 against a majority-class anchor of 0.236). The limit is
therefore recorded as a caveat on the *measure*, not on these results, and a paper reporting
a trivial fraction against a published number near chance should report the two margins
instead of their ratio.

### 4bis.3 The limits that travel with every fraction, restated

These are the tool's own, and they must appear wherever a fraction does.

* The published number must be on the **same metric, the same evaluation unit and a
  comparable test set**. A slice-level baseline against a patient-level publication is
  meaningless. This is the constraint that excluded RSNA-STR PE (§3.8) and that makes the
  PI-CAI rows arguable (§3.5).
* **It is not a decomposition.** The baseline and the published model may exploit the same
  shortcut, different shortcuts, or overlapping ones. A fraction of 0.9 does not license
  "the model learned nothing"; it licenses "this evaluation protocol certifies a number that
  a pixel-blind model also reaches".
* **The interval is too narrow as a statement about the ratio.** It propagates uncertainty
  in the baseline only; the published number enters as a fixed constant because its sampling
  distribution is almost never available. On the RSNA ICH rows the baseline interval is
  ±0.002, so essentially all of the real uncertainty in those fractions is uncertainty about
  the published constant, and none of it is shown.
* **The baseline is fitted on the training rows of the same table.** Where the published
  model was trained on a different or larger set the comparison is approximate, and §3 says
  for each row exactly how approximate.
* **The fraction says nothing about the patient level.** All 24 rows are against
  slice-level, lesion-level or case-level published numbers, at the unit each paper reports.
  The patient-level trivial fractions on RSNA ICH are all negative (−0.005 to −0.096),
  which is arithmetically true and rhetorically useless; §4 reports the patient-level
  AUROCs directly instead, and that is the right way to report them.

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

**Audited (8 label files, 7 distinct benchmarks):** fastMRI Prostate T2, fastMRI Prostate
DWI, DeepLesion, fastMRI+ knee, Duke Breast Cancer MRI, PI-CAI, LUNA16, **RSNA 2019
Intracranial Haemorrhage (new)**.

**Reached this run, having been listed as unreachable in the previous one:**

| target | how the blocker was removed |
|---|---|
| **RSNA 2019 Intracranial Haemorrhage** | Both blockers recorded previously turned out to be avoidable. The slice-position join is available in a public MIT-licensed HuggingFace mirror as a pixel-free CSV, and no click-through agreement was accepted because nothing was downloaded from Kaggle. The third-party mapping is no longer "unverified": its slice ordering is confirmed by a falsifiable run-length test (§3.7), and its patient/series counts reconcile with the peer-reviewed literature. See §3.7. |

**Not reached, with reasons:**

| target | why not |
|---|---|
| **RSNA-STR Pulmonary Embolism 2020** (was the priority target) | The official `train.csv` **was** obtained, from a public GitHub mirror, with no Kaggle credentials, and verified genuine against published counts. It still fails: it carries no slice position, and its row order and identifier order were **measured** to carry none (run-length ratio 0.974 and 1.001 against random). Independently, the peer-reviewed numbers on this benchmark are exam-level on a test set whose labels were never released. Fails (a) and (b) separately. See §3.8. |
| **RSNA 2023 Abdominal Trauma / RSNA 2022 Cervical Spine** | Kaggle-hosted; click-through-agreement blocker. `image_level_labels.csv` is genuinely per-slice and these remain the best next targets for someone who accepts the agreement. Worth noting after §3.7 that a public pixel-free mirror may exist for these too, and that possibility was not exhausted. |
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
| `slice_labels.csv` (RSNA ICH) | 61,729,905 | `72885546ba8f55fb` | HuggingFace `ianpan/rsna-intracranial-hemorrhage-16bit-png` | MIT (repo) |
| `rescale_values.csv` (RSNA ICH) | 1,294,907 | `ca1d0d583ed6a41d` | same | MIT (repo) |
| `train.csv` (RSNA-STR PE) | 119,970,071 | — | github.com/darraghdog/rsnastr, `data/train.csv.zip` | no licence file in that repo; **obtained, verified, and then not used** — see §3.8 |

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
   label-file audit of seven public benchmarks"*. The strong version survives only for
   fastMRI Prostate, whose comparator is a preprint (§2.3).
2. **Lead with the unit-of-evaluation collapse (§0.1, §4), not with the match, and lead the
   collapse with RSNA ICH on the full cohort.** It is the only result that holds across
   benchmarks and it needs no published comparator, so it cannot be attacked on
   comparability. RSNA ICH gives it 18,938 patients — 400× the prostate arm — on the
   most-cited slice-level benchmark in the field, computed twice by two implementations
   sharing no code, on all six of its labels. **The single sentence is: 0.737 [0.735, 0.740]
   at the slice level, 0.453 [0.445, 0.461] at the patient level, same score vector.**
2a. **Replace the verdict count with the distribution (§0.2, §2.0).** "Six MATCHED, nine
   PARTIAL, three NOT MATCHED" discards the information the continuous statistic exists to
   carry and stakes the paper on a threshold that only a preprint's rows cross. The
   defensible headline is: *against peer-reviewed comparators, on the same metric and unit,
   the median trivial fraction is 0.469 (IQR 0.437–0.490, range −0.002 to 0.613)*, with the
   six RSNA ICH subtype rows spanning 0.395–0.613 — roughly 40–60% of the published margin
   over chance, per subtype, with no pixels. Keep the verdict as a secondary column; the
   PI-CAI disagreement between the mechanical rule and the hand-assigned verdict (§1, §2.1)
   is itself the argument for the demotion and should be reported, not resolved.
3. **Give LUNA16 and PI-CAI first-class space (§4bis.1).** Two of seven benchmarks resisted
   the null: LUNA16 at a trivial fraction of −0.002 on the challenge's own metric, below its
   own random-score reference, and PI-CAI at exactly 0.500 for the positional arm, which is
   an "inapplicable" rather than a result. A paper that reports that is much harder to
   dismiss than one that does not — and it is the only evidence in the paper that the
   measure discriminates rather than always firing. State plainly that two negatives out of
   nine rows is an existence proof, not a specificity estimate.
3d. **Carry the measure's limits wherever the measure goes (§4bis.2, §4bis.3).** In
   particular: the fraction is undefined when the published number is at or below chance;
   it explodes when the denominator is small (4.76 at a headroom of 0.021), though the
   smallest denominator in this audit is 0.214 and the guard is never load-bearing; the
   interval propagates baseline uncertainty only, which on the RSNA ICH rows means almost
   all the real uncertainty is invisible; and it is not a decomposition.
3a. **Never present a MATCHED row without saying its comparator is a preprint.** This is
   now the paper's largest exposure. A reviewer who checks the six MATCHED rows will find
   they all trace to one arXiv posting that has not been peer-reviewed in two years. Say
   it first, in the results, not in a limitations paragraph.
3b. **Use the RSNA ICH organisers' statement as the paper's opening foil, now that it has
   a number attached.** "The available fields do not contain information that can
   determine if an image contains intracranial hemorrhage" against a slice AUROC of 0.738
   computed on their own released label file is the most economical possible statement of
   the paper's thesis. Handle it fairly: their sentence is defensible as a claim about
   individual certainty and indefensible as a claim about aggregate rankability, and the
   paper should say so rather than treat it as a simple error.
3c. **Report the RSNA PE negative next to the ICH positive.** Obtaining the official PE
   label file and then *measuring* that its row order carries no positional information
   (ratio 0.974 against random) is a stronger and more interesting result than not
   obtaining it. It also demonstrates the run-length test as a reusable instrument, which
   is a small methodological contribution the tool can carry.
4. **Rewrite the novelty section around Yan et al. 2018** (§5) and re-run the prior-art
   search properly.
5. **Reverse the T2/DWI recommendation in `audit_targets.json`** (§3.1).
6. **On today's count the label-file-only claim supports six targets** (fastMRI Prostate
   ×2 arms, DeepLesion, Duke breast tabular, PI-CAI marksheet, LUNA16, **RSNA ICH**) —
   fastMRI+ is *not* one of them. RSNA ICH moves from the "unlocatable" column to the
   "locatable, but only via a third party" column, and **RSNA-STR PE now occupies the
   column ICH vacated**: it is the case that proves a benchmark can publish a per-slice
   metric and a per-slice label file and still make the slice unlocatable.
7. **State the count that this run was commissioned to produce.** Rows meeting all three
   of (a) pixel-free label file, (b) peer-reviewed comparator on the same metric and unit,
   (c) adequate split information or an explicit approximation statement: **four
   benchmarks** — RSNA ICH, LUNA16, DeepLesion, PI-CAI. Two of those four carry a named
   caveat (DeepLesion's comparator is peer-reviewed conference proceedings rather than a
   journal; PI-CAI's is on a different cohort). fastMRI Prostate, which supplies all six
   MATCHED rows, is **not** among the four, because its comparator is a preprint. See
   §2.3 for the row-by-row determination.
