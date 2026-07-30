# Does the evaluation unit change WHICH method wins?

Written 2026-07-29, after `pipeline/s16_rankinversion.py` (our own data) and the two-pass
published-literature search (`paper/published_inversions.json`,
`paper/published_inversions_round2.json`) both returned.

Everything here is traced to a file in §10. Nothing was regenerated for this document except
two arithmetic checks on columns already published in `pipeline_out/rankinversion.json`,
which are marked as such.

---

## 0. The answer, weakest finding first

**In our own data there is no well-supported rank inversion, and the reason is not that the
rankings agree — it is that they are not identified at either unit.** Across five cohorts,
447 method pairs were examined and **zero** survived the inversion test. In no cohort does
between-unit rank disagreement exceed the disagreement that resampling subjects produces on
its own with the unit held fixed. Three cohorts cannot rank their methods at either unit
(`prostate_t2`, `prostate_dwi`, `breast`); one can order strong methods against weak ones but
cannot name a winner at either unit (`brain`); one is saturated and admits no comparison
(`knee`). The honest claim our data licenses is **"this benchmark cannot stably rank methods
at either unit"**, not "the unit changes the ranking".

**There is one supported unit effect, and it is narrower than a rank inversion.** On the
brain confound cohort the paired unit × condition interaction — how much the
magnitude-versus-phase answer moves when only the evaluation unit changes, architecture held
fixed — is **+0.028 AUROC [0.015, 0.041], p = 0.001**, positive for all six architectures and
individually Holm-supported for four. For two architectures (`densenet121/imagenet`,
`resnet18/imagenet`) the shift crosses zero, so the two units return literally opposite
answers to the paper's own question. But at neither unit is either ordering individually
distinguishable from a tie, so this is a *reproducible shift in the estimate*, not a
demonstration that a slice-level benchmark would confidently select a different method.

**In the published literature the inversion is real and it changes winners.** The strongest
case is CAMELYON16, where the organisers scored the same 32 submissions at two units and
published both boards (Kendall τ = 0.754; two methods from the same group reverse order), and
Ruan et al. 2021 reproduce 13 of those methods in one table where the winner itself changes
with the unit (τ_a = 0.603). Six further published two-unit comparisons were located, four
positive and two negative.

**But the published cases fail the same test ours does.** Only two of them publish confidence
intervals at both units, and in both the reordering sits inside the intervals: Jarkman et al.
2022 (near-complete reversal on CAMELYON16, all four intervals overlapping) and Guo et al.
2019 (τ_a = 0.133, FROC 83.5 [75.5–91.1] vs 80.7 [73.2–88.9]). The literature therefore
documents that published orderings *do* change with the unit while providing no case in which
the change has been shown to exceed sampling noise — which is exactly the state our own data
is in, reached independently.

**Consequence for the paper, stated plainly: this is a null on the headline hypothesis.** It
does not make npj Digital Medicine a target. Radiology: Artificial Intelligence remains the
right primary venue, and the null goes in as a reported result — see §8.

---

## 1. The question and the trap

The paper already shows slice-level numbers are inflated relative to patient-level numbers.
Inflation is a caveat; a reviewer can answer it with "everyone knows benchmarks are
optimistic". The claim with consequences is that the unit changes the *ordering*: if a
slice-level benchmark ranks A above B and a patient-level benchmark ranks B above A, then the
slice-level benchmark is not merely optimistic, it selects the wrong method, and every paper
that used one to justify an architectural choice inherits the error.

The trap is that a ranking computed on noisy estimates will differ between two units **by
chance**. With 21 methods whose AUCs sit inside one another's intervals, any two re-estimates
disagree. So "slice-rank ≠ patient-rank" proves nothing on its own. The analysis has to
separate

> "the unit changes the ranking"  from  "the rankings are too noisy to be stable at all"

and the only honest way is to measure the **within-unit** rank variability on the same data
and compare between-unit disagreement against it. Both outcomes are publishable; only one is
what we hoped for. What follows is whichever is true.

---

## 2. Method

`pipeline/s16_rankinversion.py`, 2,098 lines, 65 self-tests, all passing
(`./venv/bin/python pipeline/s16_rankinversion.py --self-test` → `65 passed, 0 failed`, run
2026-07-29).

**What counts as a method.** One (architecture, initialisation, input condition) cell, scored
by its pooled out-of-fold prediction vector from `s04_stats.pool_folds`, which refuses to pool
unless every subject is tested exactly once. Sources: the seven-architecture zoo
(`pipeline_out/results_arch/{complex_small, convnext_tiny, densenet121, resnet18,
resnet18_scratch, resnet50, vit_b_16}`) plus the main sweep (`pipeline_out/results`). Only
training seed 42 is admitted, the seed the zoo was run at; duplicate (arch, condition, seed)
cells appearing in both trees are taken once and the duplication is logged.

**Alignment.** All methods in a cohort are aligned on `cache_idx`, so every method is scored
on exactly the same slices of exactly the same subjects; any slice non-finite for *any* method
is dropped from *all* methods. Comparing rankings computed over different case sets would be
meaningless, and the self-test checks that a method scored on a different case set cannot
silently enter the ranking.

**One joint clustered bootstrap drives everything.** Each of 2,000 replicates resamples
**subjects** with replacement and, from that single draw, recomputes the slice-level and
patient-level AUC of every method. Because the draw is shared, the two units and all methods
move together as they do in the real dependence structure, and every between-method and
between-unit comparison is properly paired. The estimator is `s04_stats`' own
(`auc_midrank`, `_cluster_index`, `aggregate_by_cluster`); only the sharing of one draw across
methods and units is new, and the self-test checks the resulting intervals against s04's
`cluster_bootstrap_auc` and `cluster_bootstrap_diff`. No second estimator was written.

**The three quantities that decide the question.** Rank disagreement is `D = 1 − τ_b`, so
`D = 0` is an identical ordering, `D = 2` a perfect reversal, and `D/2` is the fraction of
method pairs ordered differently.

| symbol | meaning |
|---|---|
| `D_between(b)` | how far apart the two units put the methods, inside replicate *b* |
| `D_within_slice(b)` | how far the slice ranking moves when the **unit is held fixed** and only the sampled subjects change |
| `D_within_patient(b)` | the same at the aggregated unit |

The headline is the **paired** difference `δ_u(b) = D_between(b) − D_within_u(b)`, computed
inside each replicate so both terms share a draw. The unit is declared to change the ranking
only if δ lies entirely above zero against **both** references. Clearing only one is the
signature of a ranking that is pure noise at that one unit — a different finding, and not
dressed up as an inversion. The raw exceedance `P(D_within ≥ D_between_obs)` is reported
alongside.

**Two stability references, kept apart because they mean different things.** A whole-order
failure (median τ of a resampled ranking to the point ranking < 0.50, or split-half τ < 0.50
between two disjoint halves of the subjects) means the benchmark cannot separate methods at
all. A top-1 failure with a reproducible whole order means it can tell strong from weak but
cannot name a winner. The split-half reference (200 replicates, disjoint subject halves) is
the decisive one for "would another cohort rank these the same way", because disjoint halves
share no subjects and cannot be propped up by sample overlap.

**Named inversion pairs, and the bar they must clear.** (i) the point estimates disagree in
sign between the units; (ii) the paired clustered-bootstrap interval for the difference
excludes zero at slice level **and** at patient level; (iii) after **Holm adjustment over
every pair examined at that unit** — not just the ones that flipped — both p-values stay below
0.05. Step (iii) is what stops this being a fishing expedition: with 21 methods there are 210
pairs and two units, and ~21 would clear step (ii) in each direction by chance. Surviving and
strongest candidate pairs are re-run through `s04_stats.cluster_bootstrap_diff` on an
independent RNG stream as a check on this file.

**The unit × condition interaction.** `inversion_pairs` asks whether the unit reorders two
*arbitrary* methods, which over 210 pairs is a multiplicity problem with no power here. The
question a reader actually asks is "does phase beat magnitude?", with architecture held fixed.
That contrast is paired, there is one per architecture rather than 210, and it is the decision
a downstream paper would make. With architecture *a*:

```
d_slice(a) = AUC_slice(magnitude, a) − AUC_slice(phase, a)
d_agg(a)   = AUC_agg  (magnitude, a) − AUC_agg  (phase, a)
I(a)       = d_agg(a) − d_slice(a)
```

`I(a) > 0` means aggregating to the patient unit shifts the verdict towards magnitude. A
**sign flip** — `d_slice` and `d_agg` of opposite sign — means the two units return opposite
answers for that architecture. Every `I(a)` is read off the same joint bootstrap columns, so
it is a paired clustered-bootstrap difference-of-differences; Holm runs over the architectures
examined. Cells where both conditions reach ≥ 0.995 at the aggregated unit are flagged
`ceilinged` and excluded from the supported count, because there `d_agg` is pinned at zero by
arithmetic and the bootstrap would happily call the interaction significant when what it has
detected is the ceiling.

Command that produced the results file (defaults throughout: `--n-boot 2000 --seed 0
--alpha 0.05 --seeds 42`):

```
./venv/bin/python pipeline/s16_rankinversion.py      # → pipeline_out/rankinversion.json
```

---

## 3. Result 1 — the negative, which is the main result

### 3.1 Rank agreement between the two units

| cohort | methods | slices | subjects | cases | Spearman ρ [95%] | Kendall τ [95%] | top-1 slice | top-1 aggregated | top-3 overlap |
|---|---|---|---|---|---|---|---|---|---|
| brain | 13 | 680 | 136 | 136 | 0.852 [0.748, 0.989] | 0.718 [0.590, 0.949] | resnet50/imagenet/magnitude | *same* | 1 of 3 |
| knee | 3 | 290 | 29 | 58 scans | undefined | undefined | resnet18/imagenet/both | 3-way tie at 1.000 | 3 of 3 |
| prostate_t2 | 21 | 2,039 | 67 | 67 | −0.311 [−0.356, 0.635] | −0.173 [−0.257, 0.453] | densenet121/imagenet/magnitude | convnext_tiny/scratch/magnitude | **0 of 3** |
| prostate_dwi | 18 | 1,359 | 45 | 45 | 0.067 [−0.152, 0.790] | 0.072 [−0.111, 0.608] | resnet18/imagenet/both | convnext_tiny/scratch/phase | **0 of 3** |
| breast | 3 | 2,240 | 70 | 70 | 0.500 [−0.500, 1.000] | 0.333 [−0.333, 1.000] | resnet18/imagenet/magnitude | resnet18/imagenet/phase | 3 of 3 |

Read naively this looks like the result we wanted: on `prostate_t2` the two units share **none**
of their top three, ρ is *negative*, and the best method by slice AUROC (densenet121/magnitude,
0.703) ranks 4th at the patient level while the patient-level winner (convnext_tiny/magnitude,
0.539) ranks 18th of 21 at slice level. On `prostate_dwi` the top-3 sets are likewise disjoint.

That reading is wrong, and §3.2 is why.

### 3.2 The decisive comparison: between-unit disagreement against within-unit noise

| cohort | `D_between_obs` | `D_within_slice` median [95%] | `D_within_agg` median [95%] | δ vs slice, median [95%], p | δ vs aggregated, median [95%], p | exceeds both floors? |
|---|---|---|---|---|---|---|
| brain | 0.282 | 0.205 [0.077, 0.359] | 0.128 [0.026, 0.462] | −0.018 [−0.256, 0.231], p = 0.991 | +0.026 [−0.333, 0.282], p = 0.856 | **no** |
| prostate_t2 | 1.173 | 0.371 [0.181, 0.619] | 0.712 [0.471, 0.933] | +0.514 [0.010, 0.962], p = 0.046 | +0.175 [−0.334, 0.655], p = 0.508 | **no** (slice only) |
| prostate_dwi | 0.928 | 0.458 [0.196, 0.915] | 0.593 [0.314, 0.954] | +0.259 [−0.405, 0.804], p = 0.449 | +0.131 [−0.431, 0.667], p = 0.677 | **no** |
| breast | 0.667 | 0.667 [0.000, 2.000] | 0.667 [0.000, 2.000] | 0.000 [−2.000, 0.667], p = 1.000 | −0.667 [−2.000, 1.333], p = 0.839 | **no** |
| knee | undefined (every replicate fully tied at the aggregated unit) | — | — | — | — | — |

**Not one cohort clears the bar.** `P(D_within ≥ D_between_obs)`, the fraction of noise-only
perturbations that move the ranking at least as far as switching the unit does, is 0.200
(brain), 0.038 (prostate_dwi), 0.731 (breast).

`prostate_t2` needs a sentence of its own, because it is the one row that could be misread as
a win. Its observed between-unit disagreement (1.173, i.e. 59% of all method pairs ordered
differently) exceeds *every* noise-only replicate at both units, and the paired δ against the
slice-level floor does exclude zero (p = 0.046). But the paired δ against the **aggregated**
floor spans zero (p = 0.508). That asymmetry is the signature of a ranking with no ordering
information: at the aggregated unit this cohort's ranking reproduces itself at only τ = 0.29
under subject resampling, its top-1 method reproduces in 23% of resamples, and two disjoint
halves of the subjects rank the methods at split-half τ = **−0.42**. The two units disagree
because one of them is noise, not because they encode two different orderings. All 21
aggregated AUCs lie between 0.381 and 0.539.

### 3.3 Are the rankings stable at all, unit aside?

| cohort | τ(resample, point) slice | τ(resample, point) agg | split-half τ slice [95%] | split-half τ agg [95%] | top-1 reproduces, slice | top-1 reproduces, agg | verdict |
|---|---|---|---|---|---|---|---|
| brain | 0.79 | 0.87 | 0.581 [0.385, 0.796] | 0.658 [0.359, 0.876] | **32%** | 75% | `CANNOT_NAME_WINNER` |
| prostate_t2 | 0.63 | **0.29** | **0.205** [−0.067, 0.515] | **−0.421** [−0.644, −0.121] | 47% | 23% | `CANNOT_RANK` |
| prostate_dwi | 0.54 | **0.41** | **−0.033** [−0.438, 0.334] | **−0.242** [−0.560, 0.208] | 40% | 23% | `CANNOT_RANK` |
| breast | **0.33** | **0.33** | **−0.333** [−1.000, 0.333] | **−1.000** [−1.000, 0.357] | 60% | 39% | `CANNOT_RANK` |
| knee | — | — | — | — | — | — | `NO_ESTIMATE` |

Thresholds are pre-set in the file at τ = 0.50 and top-1 = 0.50 (`STABLE_TAU`, `STABLE_TOP1`),
not chosen after seeing these numbers.

The split-half columns are the ones to quote. On three cohorts, two disjoint halves of the
subjects rank the same methods **in opposite orders on average**. A benchmark in that state
cannot be used to choose an architecture at any unit, and the between-unit question does not
arise.

`brain` is the exception worth stating precisely, because it is the only cohort where methods
genuinely separate (AUROCs 0.52–0.96 on coil-count prediction): the *broad order* reproduces
(τ = 0.79 slice, 0.87 aggregated; split-half 0.58 and 0.66, both above threshold), but the
*identity of the best method* does not — slice-level top-1 reproduces in 32% of resamples.
The benchmark can separate strong methods from weak ones without being able to name a winner.

### 3.4 Named inversion pairs: none

| cohort | pairs examined | sign-discordant candidates | survivors after Holm |
|---|---|---|---|
| brain | 78 | 11 | **0** |
| knee | 3 | 0 | **0** |
| prostate_t2 | 210 | 121 | **0** |
| prostate_dwi | 153 | 71 | **0** |
| breast | 3 | 1 | **0** |
| **total** | **447** | **204** | **0** |

204 of 447 pairs point in opposite directions at the two units. Not one has both directions
individually supported. The strongest candidate anywhere is
`resnet18/imagenet/magnitude` vs `convnext_tiny/scratch/both` on `prostate_t2`: slice-level
difference +0.277 [0.192, 0.360], raw p = 0.0005 — but Holm-adjusted p = 0.105 over the 210-pair
family, and the patient-level "reversal" it is paired with is **0.0036** AUROC,
[−0.167, +0.147], p = 0.983. That is a tie, not a reversal. Five candidates per cohort were
re-run through `s04_stats.cluster_bootstrap_diff` with an independent RNG stream; all agree
with the joint bootstrap to within Monte-Carlo error (e.g. that same pair: joint slice
[0.192, 0.360] p = 0.0005 vs s04 [0.192, 0.360] p = 0.0005; joint patient [−0.167, 0.147]
p = 0.983 vs s04 [−0.169, 0.153] p = 1.000).

---

## 4. Result 2 — the one supported unit effect, and exactly what it is

The pairwise test above has no power because it asks about arbitrary pairs. The paired
unit × condition interaction asks the paper's own question with architecture held fixed, and
on one cohort it answers.

**brain, contrast (magnitude − phase), 6 architectures:**

| architecture | `d_slice` | `d_agg` | `I = d_agg − d_slice` [95%] | p | Holm p | sign flip | supported |
|---|---|---|---|---|---|---|---|
| complex_small/scratch | −0.303 | −0.270 | +0.033 [−0.020, +0.085] | 0.213 | 0.426 | no | no |
| convnext_tiny/scratch | −0.154 | −0.134 | +0.020 [−0.015, +0.054] | 0.242 | 0.426 | no | no |
| **densenet121/imagenet** | **−0.004** | **+0.023** | **+0.027 [+0.009, +0.046]** | 0.003 | **0.012** | **YES** | **yes** |
| **resnet18/imagenet** | **−0.006** | **+0.013** | **+0.020 [+0.009, +0.032]** | 0.001 | **0.006** | **YES** | **yes** |
| resnet50/imagenet | +0.001 | +0.029 | +0.028 [+0.012, +0.045] | 0.001 | 0.006 | no | yes |
| vit_b_16/scratch | −0.197 | −0.155 | +0.042 [+0.012, +0.070] | 0.008 | 0.024 | no | yes |
| **mean over the six** | | | **+0.028 [+0.015, +0.041]** | **0.001** | | 2 of 6 | 4 of 6 |

All six interactions are positive. **Aggregating from slice to patient systematically shifts
the magnitude-versus-phase verdict towards magnitude**, by about 0.02–0.04 AUROC. For
`densenet121/imagenet` and `resnet18/imagenet` the shift crosses zero: phase is ahead at slice
level (by 0.004 and 0.006) and magnitude is ahead at patient level (by 0.023 and 0.013). Those
are the paper's two named sign-flip pairs, and they are the only unit-driven reversals in the
project that survive a multiplicity correction — as *interactions*.

**The mechanism, from the same table.** Aggregation gain (patient AUROC − slice AUROC) on
brain averages **+0.045** over the six magnitude cells and **+0.017** over the six phase cells
(arithmetic on the `table` block of `pipeline_out/rankinversion.json`). Magnitude scores carry
more per-slice noise that averages out within a subject; phase scores are already close to a
subject-level property, so averaging adds little. That is a structural fact about the two
input channels, and it is why the unit moves the contrast.

**The obvious alternative explanation, tested.** Aggregation gain is larger for weaker
conditions in general — over all 13 brain methods, corr(slice AUROC, gain) = **−0.695** — and
magnitude is the weaker condition in 5 of 6 architectures, so the interaction could be
compression against the ceiling rather than a unit × condition effect. It is not, for the
three architectures that matter: at essentially identical slice AUROC (0.912–0.924) the
magnitude cells gain +0.032 / +0.023 / +0.034 and the phase cells gain +0.005 / +0.004 /
+0.006. Same level, six-fold difference in gain. (Both figures are arithmetic on the published
`table` columns.)

**What this does *not* establish.** At neither unit is either individual ordering significant:
for `densenet121`, phase vs magnitude is +0.004 [−0.041, +0.046] p = 0.816 at slice level and
−0.023 [−0.078, +0.027] p = 0.413 at patient level. So the *difference of the differences* is
supported while *each difference* is a tie. That is the statistically correct way round — the
paired difference-of-differences is the right remedy for the "difference between significant
and non-significant" error, not an instance of it — but it caps the claim at: **the unit
reliably moves the estimate; it does not let anyone confidently pick a different method at
either unit.**

**And it appears on one cohort only.** `prostate_t2` 0 of 7 architectures supported (mean
I = +0.015 [−0.048, +0.075], p = 0.679); `prostate_dwi` 0 of 6 (mean −0.022 [−0.109, +0.064],
p = 0.591); `breast` 0 of 1 (sign flip present, I = −0.035 [−0.096, +0.019], p = 0.201);
`knee` ceilinged and excluded by rule (both conditions ≥ 0.995 at the aggregated unit). The
one cohort that answers is the one whose task is coil-count prediction — a confound label,
not a diagnosis.

---

## 5. Result 3 — the published literature

Two search passes, 3,462 unique PMCIDs harvested and 2,934 full texts scanned via the Europe
PMC REST API, plus arXiv/ar5iv and direct fetches of challenge leaderboards. **WebSearch was
unavailable in both sessions** and Kaggle is JavaScript-rendered and could not be read, so
Kaggle discussion write-ups remain the one genuinely unsearched arm.

### 5.1 Positive cases

| case | unit pair | n methods | τ | winner changes? | intervals at both units? |
|---|---|---|---|---|---|
| **CAMELYON16 official boards** (Ehteshami Bejnordi 2017; live leaderboard) | lesion FROC vs slide AUC | 32 | τ = 0.754, ρ = 0.903, 61/496 discordant | top-5 membership changes; HMS/MGH M1 3rd by AUC (0.9650) / 6th by FROC (0.5963), HMS/MGH M2 8th by AUC (0.9082) / 3rd by FROC (0.7289) | **no** |
| **Ruan 2021, PLoS One 16(5):e0251521, Table 4** | lesion FROC vs slide AUC | 13 | τ_a = 0.603, 15/78 discordant | **yes** — HMS and MIT best by AUC (0.9935), Fast ScanNet-16 best by FROC (0.8533, 3rd by AUC) | **no** |
| **Islam 2021, MLMI/LNCS 12966:692–702 (RSNA-STR PE)** | image AUC vs exam AUC | 6 | τ_a = 0.467, ρ = 0.543, 4/15 discordant | **yes** — authors write that SeXception is optimal at image level and Xception at exam level | no (run-to-run SDs only) |
| **Jarkman 2022, Cancers 14(21):5424, Table 4** | lesion FROC vs slide AUC | 4 | τ_a = **−0.667** on CAMELYON16 | **yes** — AUC-best is FROC-last and vice versa | **YES** |
| **Guo 2019, Sci Rep 9:882, Table 5** | lesion FROC vs slide AUC | 6 | τ_a = 0.133 | **yes** — the "we beat the CAMELYON16 champion" claim holds only at the lesion unit | **YES** |
| **ADNI slice/scan, IJERPH 2026, Table 8** | slice acc vs scan acc | 5 | τ = **−0.600**, ρ = −0.800, 8/10 discordant | **yes** — slice-best model is scan-worst | no |
| **Acute pancreatitis, Diagnostics 2026** | slice acc vs patient acc | 5 | τ_a = 0.300, 3/10 discordant | **yes** — Swin best at slice, ViT best at patient | patient only |

Two of these deserve a sentence each.

**Ruan 2021 contains the mechanism, stated by authors who were not arguing for it:** "The label
of the patch-based training sample of Fast ScanNet was pixel-level and our classifier used
patch level, it is not difficult to understand that the former performed well in FROC of the
pixel-level detection." That is the claim — the evaluation unit selects the method optimised
for that unit — in print, in a paper about something else.

**Jarkman 2022 is the most useful case for the trap, and it cuts against us.** It publishes 95%
CIs at both units. On CAMELYON16 the ordering is almost exactly reversed, and the intervals
overlap almost completely: AUC extremes 0.988 [0.965–1.000] vs 0.969 [0.926–0.998], FROC
extremes 0.838 [0.757–0.913] vs 0.817 [0.730–0.896]. The within-unit interval width (~0.16 on
FROC) dwarfs the between-unit rank movement. And on the same paper's own local sentinel-node
cohort, the same four models at the same two units agree **perfectly** (τ_a = 1.000).

### 5.2 Negative controls, reported at the same prominence

* **Chen 2025, Sci Rep (PMC12657889), Table 1.** Eleven methods on CAMELYON16 at both units:
  τ_a = **0.927**, only 2 of 55 pairs invert, the same method (CAMCSA) is best at both units.
  The strongest counter-example located. Reporting both units does **not** reliably produce an
  inversion.
* **LGI1 encephalitis, Vis Comput Ind Biomed Art 2023;6:17.** Three architectures at both
  units with DeLong tests; ranking preserved on AUC and on accuracy.
* **LUNA16.** Checked and negative: it does not rank at two units at all — one FROC score,
  per-nodule sensitivity against per-scan false positives. (Worth citing for a different
  reason: its intervals are bootstrapped by scan-level resampling, the analogue of our
  subject-clustered bootstrap.)

### 5.3 Adjacent evidence that must not be conflated

* **Yagis 2021, Sci Rep 11:22544** — the best architecture flips on 4 of 6 datasets, 9 of 18
  pairs reverse. This is the **split** axis (slice-level vs subject-level splitting), not the
  evaluation axis. Same lesson, different mechanism. Same for the nasopharyngeal-carcinoma
  paper (Diagnostics 2022;12:2478).
* **Maier-Hein 2018, Nat Commun 9:5217** — the comparator our analysis must beat, and the
  citation that makes our null respectable: rankings of biomedical-image challenges are not
  robust to test data, ranking scheme or observer, and "Even for relatively high values of
  Kendall's tau (τ = 0.74; 0.85), critical changes in the ranking may occur". CAMELYON16's
  between-unit τ of 0.754 sits *inside* that band. So the honest reading of the strongest
  published case is that the unit change produces instability of the **same order** as the
  aggregation-scheme instability already documented — not larger.
* **PI-CAI** ranks with `Overall Ranking Score = (AP + AUROC)/2`, i.e. an average of a
  lesion-level and a patient-level score. Averaging is only necessary if the two can disagree;
  the formula is an institutional admission that the unit matters. Per-algorithm values are on
  a JavaScript leaderboard that could not be read, so the reordering could not be measured —
  only shown to be anticipated by the organisers. With 293 submissions this is by far the
  best-powered two-unit ranking in existence and it is one email away (§9).

### 5.4 The rarity denominator, which is itself a finding

Of 2,642 full texts scanned by the same-metric scanner, **18** presented a fine-unit and a
coarse-unit column for the same metric, and only **1** carried ≥ 2 comparable model rows. Of
350 papers mentioning CAMELYON / PI-CAI / LUNA16, only **7** published a table with both a
lesion-level and a slide/patient-level metric for ≥ 3 methods. Most papers report one unit, so
the ranking at the other unit is not merely unreported — it is unrecoverable from the
publication. That is a gap this paper can state with a denominator, and it belongs next to the
prevalence screen in §3.10 of the draft.

---

## 6. What all this licenses the paper to claim

**Licensed:**

1. In five cohorts, 447 method pairs and 2,000 subject-resampled replicates, between-unit rank
   disagreement never exceeds within-unit resampling noise, and no inversion pair survives a
   Holm correction over the pairs examined. On three cohorts two disjoint halves of the
   subjects rank the same methods in opposite orders. *A benchmark of this size cannot rank
   methods at either unit.*
2. The evaluation unit nevertheless moves the paper's own comparison by a reproducible amount
   where methods separate at all: +0.028 AUROC [0.015, 0.041] on the brain cohort, all six
   architectures in the same direction, four Holm-supported, two crossing zero.
3. Published rank inversions between evaluation units exist, change reported winners, and are
   documented in a major challenge's own leaderboards — and in the only two published cases
   that report intervals at both units, the inversion is not separable from sampling noise.
4. Papers reporting both units for two or more methods are rare enough to count: 1 of 2,642
   scanned full texts on the same metric, 7 of 350 benchmark-anchored papers.

**Not licensed, and must be written on the wall:**

* "The unit changes which method wins." Our data does not support it; the two published cases
  with intervals do not support it either.
* "Slice-level benchmarks select the wrong architecture." No evidence anywhere establishes
  that a slice-level benchmark's *winner* is wrong, because no located two-unit comparison can
  identify a winner at either unit.
* "Our 21 methods disagree between units, therefore benchmarks disagree between units."
  Twenty-one is 7 architectures × 3 input conditions on 67 patients; §7 is what a reviewer
  will do with that.
* Anything derived from `prostate_t2`'s disjoint top-3 sets, or from `prostate_dwi`'s. Those
  are the most quotable numbers in the file and they are chance reorderings, so labelled in
  the results file itself.

**The synthesis sentence for the paper:**

> Published orderings do change when the evaluation unit changes, in a major challenge's own
> leaderboards and in six further published comparisons; but no located two-unit comparison —
> including ours, which is powered by 2,000 subject-clustered bootstrap replicates and a
> pre-set stability threshold — can establish that the change exceeds sampling noise, because
> at these sample sizes the ranking is not identified at either unit. Architecture choices are
> being justified by rankings that do not reproduce across two halves of the same cohort.

That is a weaker claim than "the unit selects the wrong method" and a stronger one than "slice
numbers are optimistic". It is also directly actionable: a benchmark that publishes one unit
and no ranking-stability estimate is not licensing the architectural conclusion drawn from it.

---

## 7. What a reviewer will attack about this specific analysis

**1. "Your 'methods' are architectures you trained yourself, not independently published
systems."** This is the strongest objection and it is correct. All 21 cells share one training
pipeline, one preprocessing path, one augmentation policy, one optimiser schedule and one
seed. They are not a sample of *methods*; they are a sample of *configurations*, and a
configuration set assembled by one group has far less between-method spread than a challenge
leaderboard does — which is exactly why the between-unit signal is inside the noise here and
is not inside the noise on CAMELYON16, where 32 independent teams span AUC 0.78–0.99.
**What would close it:** per-algorithm scores at both units from a challenge that publishes
both. PI-CAI is the target — 293 submissions, hidden 1,000-scan test cohort, an official
ranking that already averages a lesion-level AP and a patient-level AUROC, and the same organ
as two of our cohorts. Ask the organisers for the per-algorithm AP and AUROC table, or read
the leaderboard in a real browser. Second best: re-run three or more published systems' released
weights on one cohort and score both units ourselves. Until one of those lands, our inversion
arm is an internal consistency check, not evidence about the field, and the paper must say so
in those words.

**2. "n = 45–136 subjects. Your null is a power statement."** Also correct, and the paper should
concede rather than argue. The remedy is to report what effect *could* have been detected: the
within-unit noise floors in §3.2 are the answer (a unit effect would have had to exceed
D ≈ 0.21–0.71 in rank distance to clear them), and they should be stated as a minimum
detectable effect, not left implicit.

**3. "Only one training seed. Your noise floor is not the whole noise."** True: the bootstrap
resamples subjects, not training runs, so seed-to-seed variability is absent from both the
between-unit and the within-unit terms. Note the direction, though — omitting a noise source
from the *within*-unit floor makes the floor too low and therefore biases the test **towards**
declaring a unit effect. We found none anyway. That is the conservative direction and should
be stated.

**4. "The one interaction you do have is on a confound task."** Yes. The brain cohort's label
is receive-coil count ≥ 16, not a diagnosis. The defence is that it is the only cohort where
methods separate at all (0.52–0.96), which is precisely why a ranking question can be asked
there and cannot be asked on the clinical cohorts — but the reader must be told that the one
supported unit effect in the paper concerns a hardware label.

**5. "Your interaction is a significant difference between two non-significant results."** The
answer is that the interaction is computed as a paired difference-of-differences on shared
bootstrap draws, which is the standard remedy for that error rather than an instance of it.
The concession that goes with it is in §4: neither individual ordering is established, so no
method selection follows.

**6. "The published inversions confound unit with metric."** True for every FROC-vs-AUC case
(AUC is not definable at lesion level). Islam 2021 avoids it partially — AUC at both units,
though the exam-level figure is a mean over nine labels. The ADNI and pancreatitis cases avoid
it entirely (accuracy at both units) and are the weakest cases on other grounds. This should be
tabulated, not buried: unit-and-metric-confounded cases are evidence about *reported* rankings,
which is what a downstream reader actually consumes, but not about the unit in isolation.

**7. "τ over 3 methods is not a statistic."** Right — `knee` and `breast` carry 3 methods each
and their τ intervals span [−1, 1]. They should be reported as unavailable rather than as
weak evidence, which is what the results file already does for `knee` (`NO_ESTIMATE`).

**8. "The ADNI case was selected on the slice axis."** Their scan-level evaluation covers only
the five best slice-level models, which truncates slice-level variance relative to scan-level
variance and inflates apparent discordance. τ = −0.600 is an upper bound on the effect. Their
split is also scan-based rather than subject-based. Both caveats must travel with the number.

---

## 8. Venue consequence, stated straight

The user has asked repeatedly, so: **this is the null, and the null does not change the
venue.**

* **Radiology: Artificial Intelligence remains the primary target.** Nothing in this analysis
  weakens the paper — it adds a properly-powered negative result and a literature denominator
  — and nothing in it supplies the finding that would have justified moving up. The rank
  analysis enters as a Results subsection and a Limitations paragraph, not as a headline.
* **npj Digital Medicine stays a shot, not a target.** The pre-submission enquiry
  (`paper/enquiry_npj.md`) should still be sent, and should still lead with the prevalence
  screen, which remains the strongest field-level object in the project. The rank-inversion
  work belongs in that letter as one clause — "and we show that at realistic cohort sizes the
  ranking is not identified at either unit, so the unit question cannot be settled by
  single-centre data" — because it is honest, it pre-empts the reviewer who would ask, and it
  is not a claim the editor will treat as the paper's core.
* **What would have changed the call, for the record:** a Holm-surviving inversion pair in our
  own data on a clinical cohort, or a published inversion with non-overlapping intervals at
  both units. Neither exists. Case 3 in §5.1 came closest and its intervals overlap almost
  completely.
* **What could still change it:** the PI-CAI per-algorithm table (§7, attack 1). With 293
  submissions and published intervals, a between-unit rank analysis there would be
  well-powered, would use genuinely independent methods, and would settle the question in
  either direction. That is one email and a fortnight, and it is the highest tier-per-hour
  item left in the project. It is also the only route that would make npj Digital Medicine a
  genuine target rather than a shot.

---

## 9. Limits of this analysis

1. **Configurations, not methods** (§7.1). Everything downstream inherits it.
2. **One seed** (seed 42 only; `seeds_admitted: [42]`). Method estimates carry training-run
   variance that no interval here represents.
3. **Small cohorts.** 45–136 subjects. Three of five cohorts cannot rank at either unit; that
   is a statement about these cohorts, not about slice-level benchmarks in general.
4. **`knee` is uninformative by construction.** Matched-pairs design, every subject carries
   both labels, so the aggregated unit is the scan (58 scans, 2.0 per subject, subject retained
   as the bootstrap cluster) and all three methods reach 1.000 at that unit.
5. **`breast` and `knee` carry 3 methods.** τ is nearly uninformative at that width.
6. **The literature search never had WebSearch.** Both passes failed with an API error; Europe
   PMC full-text search, the arXiv API, the GitHub API and direct leaderboard fetches were the
   substitutes. Kaggle discussion write-ups — which do sometimes report both image-level and
   exam-level validation scores in prose — are unsearched. The rarity denominators in §5.4 are
   therefore lower bounds on coverage, and the "we found only 7" figures must be quoted with
   the search method attached.
7. **No published prediction files were obtainable.** The leading Kaggle solution repositories
   for RSNA-STR PE and RSNA ICH ship code but no out-of-fold or test prediction files (218,
   505, 55 and 0 candidate files across four repos), and both competitions withhold test
   labels, so a two-unit recomputation from someone else's predictions is not currently
   possible.
8. **One correction carried forward.** In the pass-1 pancreatitis record, ResNet50 and Swin are
   **tied** at 78.38% patient accuracy; the recorded ranks 3 and 4 impose an order the data do
   not support. τ_a = 0.300 and 3/10 discordant are unaffected; τ_b = 0.316 and the tie should
   be stated wherever that case is cited.

---

## 10. Every number, and the file it comes from

| where | file |
|---|---|
| all own-data numbers in §3 and §4 | `/Users/sathvikloke/Downloads/PhaseDx/pipeline_out/rankinversion.json` (schema `phasedx.s16.rankinversion.v1`, `n_boot` 2000, `seed` 0, `alpha` 0.05, `seeds_admitted` [42], written 2026-07-29 23:16) |
| the estimator, the trap test, the verdict rule, the thresholds | `/Users/sathvikloke/Downloads/PhaseDx/pipeline/s16_rankinversion.py` (`stability_comparison`, `inversion_pairs`, `condition_interaction`; `STABLE_TAU = 0.50`, `STABLE_TOP1 = 0.50`) |
| pooling, clustered bootstrap, DeLong, Holm | `/Users/sathvikloke/Downloads/PhaseDx/pipeline/s04_stats.py` (`pool_folds`, `auc_midrank`, `aggregate_by_cluster`, `cluster_bootstrap_auc`, `cluster_bootstrap_diff`, `holm_adjust`) |
| self-test evidence | `./venv/bin/python pipeline/s16_rankinversion.py --self-test` → `self-test: 65 passed, 0 failed` |
| per-method AUCs and ranks quoted in §3.1, §4 | `cohorts[*].table` in `rankinversion.json` |
| agreement (ρ, τ, top-1, top-3) | `cohorts[*].agreement` |
| the decisive comparison (§3.2) and stability (§3.3) | `cohorts[*].stability` (`d_between`, `d_within_slice`, `d_within_patient`, `delta_vs_slice`, `delta_vs_patient`, `p_exceed*`, `split_half`, `verdict`, `verdict_text`) |
| pairs, candidates, survivors, s04 cross-check (§3.4) | `cohorts[*].inversions` |
| the interaction table (§4) | `cohorts[*].interaction` |
| aggregation-gain means and the matched-level subset (§4) | arithmetic on `cohorts[brain].table` (`auc_patient − auc_slice`), recomputed 2026-07-29 |
| model sources | `pipeline_out/results_arch/{complex_small,convnext_tiny,densenet121,resnet18,resnet18_scratch,resnet50,vit_b_16}` and `pipeline_out/results` |
| published cases, pass 1 (CAMELYON16 boards, RSNA-STR PE, ADNI, pancreatitis; LGI1 negative control; Yagis and Maier-Hein adjacent) | `/Users/sathvikloke/Downloads/PhaseDx/paper/published_inversions.json` |
| published cases, pass 2 (Ruan, Jarkman, Guo; Chen negative control; PI-CAI, LUNA16, CAMELYON16/17; rarity denominators; verification of pass-1 cases; Kaggle prediction-file search) | `/Users/sathvikloke/Downloads/PhaseDx/paper/published_inversions_round2.json` |
| the scanners behind the literature denominators | `/Users/sathvikloke/Downloads/PhaseDx/paper/screen/inversion_scan.py`, `/Users/sathvikloke/Downloads/PhaseDx/paper/screen/inversion_harvest.py` |
