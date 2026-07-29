# trivialbaselines

**Audit a slice-level medical imaging benchmark using nothing but its label file.**

A slice-level classification benchmark is usually reported as a slice-level AUROC over a
test set of volumes. That number can be reached, in part or in whole, by a model that
never sees a pixel — because the label is correlated with *where* a slice sits in its
stack, with *how* the volume was acquired, or with *which release batch* it came from.

`trivialbaselines` fits that family of pixel-blind null models on a benchmark's own
published label table, evaluates them at both the slice and the patient level with
subject-clustered intervals, and reports how much of the published headline number they
account for.

No images. No k-space. No data-use agreement for pixels. No GPU. **No torch** —
`numpy` and `pandas` are the entire dependency list, which is checkable in
`pyproject.toml` and is the point: a benchmark whose images you will never hold can
still be audited, because almost every dataset publishes its labels.

---

## 60-second quickstart

```bash
git clone https://github.com/sathvikloke/PhaseDx && cd PhaseDx/trivialbaselines
python -m venv .venv && source .venv/bin/activate
pip install .

trivial-baselines --labels examples/shortcut_benchmark.csv --published 0.88
```

`examples/shortcut_benchmark.csv` is a synthetic 200-patient benchmark that contains a
real per-patient disease state *plus* two artefacts that occur in real releases: lesion
slices cluster mid-stack, and the label rate differs between download batches. Suppose a
paper reported a slice-level AUROC of **0.88** on it. Here is what the tool says:

```
============================================================================================
ZERO-IMAGE BASELINES  shortcut_benchmark
============================================================================================
  6011 slices / 200 subjects / 200 volumes   prevalence 0.083 slice, 0.440 patient
  label rule: lesion already binary {0,1}
  evaluation: the dataset's own 'data_split' column; validation rows: exclude
  metadata columns: 3  (release_batch, scanner_model, TR)

  baseline                       slice AUC  slice 95% CI       pat AUC  patient 95% CI       null  excess
  ----------------------------------------------------------------------------------------
  prevalence                         0.500  [0.500, 0.500]       0.500  [0.500, 0.500]          -       -
  positional_20bin                   0.840  [0.794, 0.886]       0.324  [0.156, 0.503]      0.489   0.351
  volume_size                        0.402  [0.268, 0.554]       0.406  [0.238, 0.598]      0.485  -0.084
  metadata_tree                      0.654  [0.523, 0.781]       0.679  [0.486, 0.851]      0.528   0.126
  combined_position_metadata         0.845  [0.796, 0.892]       0.647  [0.441, 0.836]      0.805   0.039
  'null' is this baseline's own permutation null, which is NOT always 0.5; 'excess' is observed - null.

  bin sweep (slice AUC): 5=0.807  10=0.832  20=0.840  50=0.838   no-fit centrality=0.839

  best single metadata columns (slice AUC / patient AUC)  [max over 4 columns, not multiplicity-corrected]:
      release_batch                  0.736     0.815   (categorical, 4 levels)
      scanner_model                  0.559     0.521   (categorical, 3 levels)
      TR                             0.501     0.537   (numeric, 10 levels)
      n_slices_per_volume            0.402     0.406   (numeric, 10 levels)

  chance anchor          0.500   (constant predictor measured at 0.500)
  best zero-image        0.845 [0.796, 0.892]  (combined_position_metadata)
  published              0.880
  TRIVIAL FRACTION       0.907 [0.780, 1.032]   -> 91% of the margin over chance needs no pixels
```

Three things to read off that table, in order of importance:

1. **`positional_20bin` reaches 0.840 slice-level and 0.324 patient-level.** The same
   scores, the same test set, read at two different units. The slice-level number looks
   like a working detector; the patient-level number is below chance. Whatever is being
   ranked is stack geometry, not patients.
2. **`release_batch` alone reaches 0.736.** An administrative field — which tarball the
   volume was downloaded in — outperforms most of the rest of the table.
3. **The trivial fraction is 0.907.** 91% of this benchmark's margin over chance is
   reachable without pixels.

Now run the control, which is the more important demonstration:

```bash
trivial-baselines --labels examples/clean_benchmark.csv --published 0.88
```

Same schema, same prevalence, same per-patient disease state — but lesion slices are
uniform in relative position and the batch is independent of the label. Every baseline
lands near 0.5, the best is `metadata_tree` at 0.484 [0.372, 0.615], and the trivial
fraction comes out **-0.041 [-0.338, 0.304]**. The tool does not simply always fire. A
benchmark on which the null models fail is a real, reportable, publishable result.

Verify the whole thing at once:

```bash
trivial-baselines --self-test        # ~45 s; synthetic data with known answers
trivial-baselines --self-test --quick # ~25 s; fewer bootstrap replicates
```

Each run also writes `./trivial_baselines/<name>.json` (the full payload, every number
traceable) and `./trivial_baselines/<name>.md` (a card you can paste into a supplement).
Add `--no-write` for console only, `--out-dir DIR` to put them elsewhere.

---

## What it needs

One tidy table — CSV, TSV, or Parquet — with, at minimum:

| role | what it is | auto-detected from |
|---|---|---|
| subject | patient / subject identifier | `patient_id`, `subject_id`, `fastmri_pt_id`, `case_id`, … |
| slice | slice index or z position | `slice`, `slice_idx`, `instance_number`, `z`, … |
| label | the target | `label`, `lesion`, `PIRADS`, `malignant`, `diagnosis`, … |

and optionally:

| role | what it is | auto-detected from |
|---|---|---|
| split | the dataset's own train/test column | `data_split`, `split`, `partition`, … |
| volume | series / file identifier, when a subject has several stacks | `fastmri_rawfile`, `series_id`, `file`, … |
| metadata | any number of acquisition or administrative fields | everything left over |

Every one of these is overridable. Auto-detection is a convenience, never a requirement:

```bash
trivial-baselines --labels t2_slice_level_labels.csv \
    --subject-col fastmri_pt_id --slice-col slice \
    --label-col PIRADS --positive-if '>2' \
    --split-col data_split \
    --name rempe_t2 --published 0.861 \
    --published-label 'Rempe et al. 2024 headline, slice-level AUC'
```

`--positive-if` accepts `>2`, `>=3`, `==1`, `in:3,4,5`. Without it, a column already in
`{0,1}` is used as-is and anything else raises rather than guessing.

---

## The baselines

| name | what it knows | what it is a proxy for |
|---|---|---|
| `prevalence` | nothing — a constant | the chance anchor, and a check that the harness is not rewarding a degenerate model |
| `positional_20bin` | P(label \| relative slice position), binned, fitted on train | stack geometry: findings live in the middle of the organ |
| `volume_size` | how many slices are in the volume | protocol, scanner, acquisition era |
| `metadata_tree` | acquisition/administrative columns, depth-limited CART | release batch, matrix size, site, coil count |
| `combined_position_metadata` | position + metadata in one tree | the ceiling reachable with no pixels at all |

Relative position is `(slice - min_slice_in_volume) / (max - min)`, so volumes of
different depth are comparable. The positional baseline is reported with a bin sweep
(5/10/20/50) and a *no-fit* variant — `-(|relative position - 0.5|)`, which uses no
training data whatsoever — so the result cannot be waved away as a binning artefact.

### Discipline the harness enforces

- **Fit on train, score on test.** Every baseline implements `fit(train)` / `score(rows)`.
- **Subject-level splits, never a random slice split.** The dataset's own split column is
  preferred when one exists; the harness asserts and records that the train and test
  subject sets are disjoint.
- **Both units, always.** Slice-level and patient-level AUROC for the same score vector.
  The divergence between them is usually the finding.
- **Subject-clustered bootstrap intervals.** The naive slice-level interval is computed
  too — and reported as `slice_ci_naive` in the JSON — but only so a report can show how
  much narrower the wrong interval would have been. In simulation (20 subjects × 15
  slices, 200 datasets) the naive interval covered the true AUC **46.5%** of the time at
  a nominal 95%, against 91.5% for the clustered one, and was 3.2× narrower.
- **Apparent (train) performance next to test performance**, so overfitting of the null
  model itself is visible rather than assumed away.
- **A permutation null per baseline.** A baseline's null is *not* automatically 0.5. Fit
  a metadata model out-of-fold on a subject-level label and the rate you fitted is
  anti-correlated with the rate you score, so it lands *below* chance — the metadata
  baseline measures 0.424 on a synthetic dataset whose label is by construction invisible
  to metadata (see the `--self-test` output). Judging against 0.5 there would have
  manufactured a below-chance "finding" out of arithmetic; judging against the baseline's
  own permutation null reports it correctly as no effect. Where a permutation cannot
  change anything (shuffling labels within
  a single-class volume) the null is reported as *unavailable*, not as p = 1.0.
- **Outcome-derived and image-derived columns are excluded from metadata by default.**
  A column that is the label under another name (PI-RADS, grade, receptor status) is
  tautological; a column computed *from* the pixels (SNR, SSIM, mask fraction) breaks the
  zero-image guarantee. The exclusion is a fallible name heuristic, so every included and
  excluded column is printed and written to the JSON. Use `--metadata-cols` to state the
  set explicitly when it matters.

---

## The trivial fraction, and what it does not licence

```
trivial_fraction = (baseline - chance) / (published - chance)
```

Read it as: *this published evaluation certifies X% of its own margin over chance to a
model that never saw an image.*

**It is a statement about an evaluation protocol, not about a published model's
internals.** It does not say the model learned nothing, and a paper that writes it that
way is wrong. The baseline and the published model may exploit the same shortcut,
different shortcuts, or overlapping ones; the fraction cannot distinguish these.

The other limits, all reproduced in the "Interpretation" block of every generated card
and in `trivialbaselines.TRIVIAL_FRACTION_LIMITS`:

- The published number must be on the **same metric**, the **same evaluation unit**
  (slice vs patient) and a comparable test set. A slice-level baseline compared against a
  patient-level publication is meaningless.
- The fraction is **undefined** when the published number is at or below chance, and is
  reported as `null` rather than as a large or negative number.
- Values **above 1** mean the zero-image baseline exceeded the published number. That is
  a real outcome, not an error, and it is left unclipped.
- The interval propagates uncertainty in the **baseline only**. We almost never have the
  published number's sampling distribution, so it enters as a fixed constant and the
  interval is too narrow as a statement about the ratio.
- The baseline is fitted on the **training rows of the same table**. If the published
  model was trained on a different or larger set, the comparison is approximate.

---

## The remedy metric: position-stratified AUROC

Diagnosis is not enough. `stratified_auc` computes the Mann-Whitney statistic *within*
strata of relative slice position, so only same-position positive/negative pairs
contribute and exactly the share of a slice-level AUROC that came from stack geometry is
removed — nothing else is.

`audit()` never sees your model's predictions, so this one is called directly:

```python
from trivialbaselines import position_strata, stratified_auc

rel = (slice_idx - lo) / (hi - lo)          # relative position within each volume
print(stratified_auc(labels, scores, position_strata(rel, n_strata=10)))
```

On Rempe et al. (2024)'s own published prostate DWI label file and split, all three
numbers below come from **one** score vector produced by the zero-image positional
baseline:

| reading of the same scores | AUROC |
|---|---|
| slice-level | **0.851** |
| patient-level | **0.424** |
| position-stratified slice-level (10 bins, 6 populated) | **0.539** |

Report all three. The gap between the first and the third is the part of the headline
that stack geometry paid for.

---

## Python API

```python
from trivialbaselines import audit, render_card

payload = audit("examples/shortcut_benchmark.csv",
                name="mybench", label="lesion", published=0.88)

ev = payload["evaluations"][payload["headline_evaluation"]]
pos = ev["baselines"]["positional_20bin"]
print("positional slice / patient:", round(pos["slice_auc"], 3), round(pos["patient_auc"], 3))
print("trivial fraction:", round(payload["headline"]["trivial_fraction"]["value"], 3))

open("card.md", "w").write(render_card(payload))
```

```
positional slice / patient: 0.84 0.324
trivial fraction: 0.907
```

For a real benchmark, add the column overrides: `label="PIRADS", positive_if=">2",
subject="fastmri_pt_id", split="data_split"`.

`audit()` takes the same knobs as the CLI. The returned payload is plain JSON-serialisable
data: every number in the console output and in the card is in there, alongside the column
resolution, the exclusions, the split accounting, the permutation nulls, the sha256 of the
label file, and the exact command line.

---

## Reproducibility notes

- **`trivialbaselines/core.py` is `pipeline/s14_trivialbaselines.py` from the PhaseDx
  study**, with four changes and no others: the statistics import points at the vendored
  copy, the `sys.path` hack is gone, output defaults to `./trivial_baselines` instead of
  `pipeline_out/`, and the usage lines name the installed CLI. `diff` the two files.
- **`trivialbaselines/stats.py` is extracted verbatim** from `pipeline/s04_stats.py` by
  AST, function by function — `compute_midrank`, `auc_midrank`, `average_precision`,
  `aggregate_by_cluster`, `cluster_bootstrap_auc`, `naive_slice_bootstrap_auc`. It is
  copied rather than imported so that this package installs standalone. The function
  bodies are byte-identical to the ones every number in the paper was computed with.
- **`trivialbaselines/stratified.py` is extracted verbatim** the same way from
  `pipeline/s12_rempe.py` (`stratified_auc`, `position_strata`), with the statistics
  import repointed. It is the only thing in this package that is not reachable from
  `s14_trivialbaselines.py`, and it is here because the protocol the paper proposes asks
  for it. Installed in a clean venv it reproduces the study's persisted values on Rempe
  et al.'s published DWI label file to four decimals: slice 0.8514, patient 0.4240,
  position-stratified 0.5392 over 6 populated strata.
- The `tool` field in emitted JSON reads `s14_trivialbaselines` on purpose: a payload
  produced by this package and one produced inside the study are directly comparable.
- `--seed` fixes the bootstrap and the CV assignment; two runs with the same seed and the
  same input file produce identical payloads apart from the timestamp.
- scikit-learn, if present, is used by the self-test *only*, to cross-check the built-in
  CART against `DecisionTreeClassifier` on an XOR problem. The self-test passes without
  it and falls back to a weaker assertion, which it announces.

## Citation

If this tool changes what you report, cite the paper it came from, and — more usefully —
publish your own card. See `paper/protocol.md` and `paper/checklist.md` in the parent
repository for the evaluation protocol these baselines are meant to support.

## License

MIT. See `LICENSE`.
