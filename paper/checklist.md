# Slice-level benchmark checklist

**One page. For authors before submitting, and reviewers before recommending acceptance,
any paper reporting performance on slice-labelled 3D medical images.** Every item is
answerable from a label file and a table of predictions. None requires a GPU. Full
rationale and the measured failures behind each item: `paper/protocol.md`.

---

### Split

- [ ] **S1.** The split is at the **subject** level. Every volume of a subject is in exactly one arm.
- [ ] **S2.** Train and test subject sets are stated to be **disjoint**, and this was checked, not assumed.
- [ ] **S3.** The paper says *which unit* was split, in words, next to the split sizes.

### Reporting unit

- [ ] **U1.** The **primary** number is at the **patient** level. Any slice-level number is labelled secondary.
- [ ] **U2.** The **aggregation rule** (mean / max / other) is stated and was chosen before seeing results.
- [ ] **U3.** Sample size is reported as **n patients and n positive patients**, not only n slices.
- [ ] **U4.** If slice-level and patient-level readings of the same scores disagree, both are shown.

### Uncertainty

- [ ] **I1.** Intervals come from a **subject-clustered** bootstrap (resample subjects, not slices).
- [ ] **I2.** No slice-level bootstrap interval is presented as the headline interval.
- [ ] **I3.** Degenerate bootstrap replicates (single-class, single-cluster) are counted and reported.

### Baselines — the zero-image family

- [ ] **B1.** A **constant (prevalence)** predictor is reported. Its measured value is shown, not assumed.
- [ ] **B2.** A **positional** baseline, P(label | relative slice position), fitted on **training** slices and scored on test slices, is reported on the same metric and unit as the headline.
- [ ] **B3.** A **metadata** baseline, fitted on acquisition/administrative fields alone, is reported.
- [ ] **B4.** Every baseline is fitted on train and scored on test; **apparent (train) performance** is shown next to test performance.
- [ ] **B5.** Each baseline is compared against **its own permutation null**, not against 0.5. (Out-of-fold metadata baselines sit *below* chance by construction: on a synthetic label invisible to metadata, one measured 0.424.)
- [ ] **B6.** Metadata columns exclude **outcome-derived** fields (the label under another name) and **image-derived** fields (they break the zero-image premise). The exclusion list is published.

### Positional structure

- [ ] **P1.** The **label rate against relative slice position** is published as a histogram (training set).
- [ ] **P2.** The **position-stratified** AUROC is reported next to the raw slice-level AUROC.
- [ ] **P3.** The positional baseline is reported over a **bin sweep** (e.g. 5/10/20/50) so it is not a binning artefact.

### Metadata confounding

- [ ] **M1.** Each available acquisition field was tested against the label; the **strongest** is named with its AUROC.
- [ ] **M2.** **Release batch / download tarball / source directory** is among the fields tested. (Measured: it predicts the label at 0.743 in our breast cohort, against 0.633 for the trained network.)
- [ ] **M3.** If any field beats or matches the model, the paper says so in the results, not only in the limitations.

### The headline comparison

- [ ] **T1.** The **trivial fraction** — (best zero-image baseline − chance) / (published − chance) — is reported with an interval.
- [ ] **T2.** It is reported **even when small**. A benchmark on which the nulls fail is a result. (LUNA16's best zero-image baseline is 0.539 [0.520, 0.565]; PI-CAI's positional baseline, at the case level its authors report, is exactly 0.500.)
- [ ] **T3.** The comparison is on the **same metric, same unit, comparable test set**.
- [ ] **T4.** No sentence claims the model "learned nothing". A high trivial fraction is a statement about an **evaluation protocol**, not about a model's internals.

### Reproducibility of the audit itself

- [ ] **R1.** The label file (or its schema and split column) is public, so a third party can rerun B1–B3 without the pixels.
- [ ] **R2.** The seed, the number of bootstrap replicates, and the exact command are recorded.

---

### Run most of this in one command

```bash
pip install trivialbaselines
trivial-baselines --labels your_labels.csv --published 0.861
```

Covers B1–B6, P3, M1–M2, T1–T4 and R2, and writes a JSON payload plus a card you can
paste into a supplement. B4 and B6 land in the JSON (`apparent_slice_auc_on_train`,
`columns.excluded`) rather than in the printed card; R2 is `settings` plus `command`.
Requires `numpy` and `pandas`. No torch, no GPU, no images.
For P2, call `trivialbaselines.stratified_auc` on your own test predictions.

---

### The three numbers that motivate this page

All measured, all on published artefacts or our own cohorts:

| | |
|---|---|
| A zero-image positional baseline on Rempe et al. (2024)'s own published prostate DWI label file and split | **0.851** slice-level, against their **0.861** headline — and **0.424** at the patient level, from the same scores |
| Release batch (which tarball a scan was downloaded in) predicting breast cancer status | **0.743**, against **0.633** for the trained network on the same cohort |
| Coverage of a nominal 95% slice-level bootstrap interval, simulation with known truth | **46.5%** (subject-clustered: 91.5%) |

---

*Cite as: the evaluation protocol in `paper/protocol.md`. Reuse freely — this page is
more useful copied into your own supplement than cited from ours.*
