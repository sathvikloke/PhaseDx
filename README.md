# PhaseDx

MRI scanners collect k-space — complex numbers with both magnitude and phase. Standard
reconstruction throws the phase away, and essentially every published AI tumour classifier
trains on what's left. A 2024 paper reported that keeping phase improved prostate cancer
classification on DWI. This project asked whether that generalises.

**It does not — not in this data. And what the models actually learned was the scanner.**

> **Note on earlier versions of this README.** Until 2026-07-29 this file reported prostate
> phase-only AUCs of 0.883/0.818 against magnitude 0.779/0.636 and described that as a
> stronger result than the 2024 paper. **Those numbers were wrong.** They came from a reader
> that assumed 4-D k-space, ignored that prostate DWI is 2× undersampled with the zeros
> stored — so a plain inverse FFT yields two copies of the pelvis folded together — and
> evaluated at slice level with file-level splits. See [`legacy/README.md`](legacy/README.md).

## Findings

Pooled out-of-fold over 5-fold subject-level cross-validation, subject-clustered bootstrap
intervals. **Patient-level AUROC — every interval covers or sits below 0.5:**

| cohort | subjects | magnitude | phase | both |
|---|---|---|---|---|
| prostate T2 (pre-registered primary) | 67 | 0.473 | 0.380 | 0.492 |
| breast DCE | 70 | 0.642 | 0.631 | 0.615 |
| prostate DWI | 45 | 0.465 | 0.442 | 0.457 |

All three are **NOT SUPPORTED** on six or seven of nine pre-registered criteria. The null
holds across seven architectures, including a complex-valued network that consumes the
complex image natively: the best phase result over every architecture × cohort is
0.503 [0.369, 0.646].

**The falsification suite explains why:**

- Training on the **air outside the body**, anatomy deleted, scores within **0.018** of
  training on the anatomy.
- The label-permutation null sits at **0.595 [0.548, 0.645]**, not 0.5.
- Holding acquisition protocol out across the split erases the effect.
- Phase predicts the **receive-coil configuration at 0.979 [0.953, 0.996]** within site, on
  454 brain subjects. Trained from scratch, a complex-valued network reads it at 0.821 while
  sitting at chance (0.519) on magnitude.

**A trivial baseline reaches published performance.** On the 2024 paper's own label file and
its own patient-disjoint split, a predictor reading **no image data at all** — only where a
slice sits in its stack — reaches **0.851 [0.816, 0.887]** slice-level AUROC against a
reported 0.809–0.861, and falls to **0.424 [0.298, 0.547]** at the patient level.

The positional construction is **not new**: it was publicly disclosed in a 2020 Kaggle
notebook ("Baseline with no image", RSNA-STR PE Detection), and DeepLesion's own defining
paper reports a location-only baseline. Prior art is catalogued in
[`paper/audit_targets.md`](paper/audit_targets.md).

## `trivialbaselines`

The auditor, generalised and released. Give it a label table — subject id, slice index,
label, split — and it reports what is reachable with no pixels: positional, metadata-only,
volume-size, prevalence, and a combined ceiling, each at slice and patient level, with
clustered intervals and calibrated against its own permutation null.

```bash
pip install ./trivialbaselines
trivial-baselines --labels your_labels.csv --published 0.86
```

No images, no GPU, no data-use agreement. numpy and pandas only.
See [`trivialbaselines/README.md`](trivialbaselines/README.md).

## Layout

```
pipeline/           s00 inventory -> s14 trivial baselines; the full study
trivialbaselines/   the released tool
paper/              manuscript, pre-registered screening protocol, audit results
legacy/             the original code, and a record of what it got wrong
manuscript/         superseded drafts (not published; see DO_NOT_SUBMIT.md)
```

`python pipeline/run_all.py --help` for the orchestrator; most stages take `--self-test`.

## Data

NYU fastMRI, under a data use agreement: <https://fastmri.med.nyu.edu>.

Practical notes, unchanged and still accurate: download `labels.tar` first for prostate;
don't pull everything, it's terabytes. For breast, malignant cases concentrate in patients
131–160 and 261–300, so those blocks give the most signal per gigabyte.

**No imaging data or derived cohort table is in this repository.** `pipeline_out/` is
excluded in full — the cohort tables carry coded patient identifiers alongside institution
and device fields taken from scan headers.

## Reproducibility

Every reconstruction is validated against the vendor reference shipped inside the same HDF5
file: r = 1.000 (brain, knee, vs `reconstruction_rss`), 0.998 (prostate T2), 0.984 (prostate
DWI, low-b-averaged, vs `trace_b50`), 0.97 (breast, vs `temptv`). Persisted per slice and per
file in `pipeline_out/recon_fidelity/`, not asserted in a comment.

**383+ self-test checks across eight suites.** The report layer was audited adversarially six
times by agents whose objective was to make it print a positive verdict while a falsification
control had failed. Every successful attack is now a permanent regression test, and each new
guard is mutation-verified: disable it and its test must fail.

## Status

Work in progress. The findings above are stable and reproducible; the manuscript is not
final. See [`paper/PAPER_PLAN.md`](paper/PAPER_PLAN.md) for the honest novelty position and
target venue.

## References

- Rempe et al. (2024). *Tumor likelihood estimation on MRI prostate data by utilizing k-Space
  information.* arXiv:2407.06165
- Zbontar et al. (2018). *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI.*
  arXiv:1811.08839
- Tibrewala et al. (2024). *FastMRI Prostate.* Scientific Data 11, 404.
- Solomon et al. (2025). *FastMRI Breast.* Radiology: Artificial Intelligence.
- Badgeley et al. (2019). *Deep learning predicts hip fracture using confounding patient and
  healthcare variables.* npj Digital Medicine 2:31.
