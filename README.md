# Phase-Informed MRI Tumor Classification
### RSNA 2026 Abstract Project

---

## What this project does

This code tests whether MRI phase information — data that is routinely discarded
during standard image reconstruction — improves tumor classification accuracy
across three organs: brain, prostate, and breast.

Every published MRI tumor AI paper uses only the magnitude image (what a
radiologist sees on screen). MRI scanners actually collect raw k-space data
containing both magnitude AND phase. One 2024 paper showed phase improves
prostate cancer classification. This study asks: does that finding generalize
to brain and breast as well?

We test three input conditions using the same ResNet-18 classifier:
  - Condition A: magnitude only      (baseline — what every prior paper does)
  - Condition B: phase only          (control)
  - Condition C: magnitude + phase   (our main experiment)

If Condition C consistently beats Condition A across all three organs,
phase is a universal diagnostic signal that current AI systems are ignoring.

---

## Dataset: FastMRI

This project uses the FastMRI dataset from NYU Langone Health.
You must apply for access at: https://fastmri.med.nyu.edu/

### What to download (do NOT download everything — it's terabytes)

| Organ    | What to download              | Approx size  | Has tumor labels?         |
|----------|-------------------------------|--------------|---------------------------|
| Prostate | All splits, T2 + DWI folders  | ~20–30 GB    | Yes — Pi-RADS scores      |
| Brain    | multicoil_train only          | ~100–150 GB  | Partial (contrast T1)     |
| Breast   | Training split only           | ~500 GB      | Yes — benign/malignant    |

Knee dataset: DO NOT download — it has no tumor labels and is not used.

Recommended starting point: download prostate first. It is the smallest,
has the best labels, and is the organ from the 2024 paper you are replicating.

---

## Project structure

```
phase_mri/
├── README.md                  ← you are here
├── requirements.txt           ← Python dependencies
├── run_experiment.py          ← main entry point — run this
├── train.py                   ← training loop
├── evaluate.py                ← evaluation, results table, ROC curves
├── utils/
│   ├── __init__.py
│   └── data_utils.py          ← k-space loading, phase extraction, datasets
└── models/
    ├── __init__.py
    └── models.py              ← ResNet-18 adapted for 1 or 2 input channels
```

---

## Setup

### 1. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 2. Smoke test (no FastMRI data needed)

Run this first to verify the full pipeline works on your machine
before downloading any data:

```bash
python run_experiment.py --smoke_test
```

You should see training logs for 9 runs (3 organs x 3 conditions)
and a results table printed at the end. Check the output folder
for ROC curve plots and training history charts.

### 3. Real experiment (once you have FastMRI data)

```bash
python run_experiment.py \
  --prostate_dir /path/to/fastmri/prostate \
  --brain_dir    /path/to/fastmri/brain \
  --breast_dir   /path/to/fastmri/breast \
  --output_dir   ./results \
  --epochs       20 \
  --batch_size   16 \
  --num_workers  0
```

You do not have to provide all three organs. If you only have prostate
downloaded, just pass --prostate_dir and omit the others. The code
will skip missing organs gracefully.

### 4. Mac-specific notes

- Set --num_workers 0 (default). MacBooks have issues with h5py + multiprocessing.
- Mixed precision (AMP) is disabled by default. MPS (Apple Silicon) does not
  fully support it yet.
- The code auto-detects your device: CUDA > MPS (Apple M4) > CPU.
  On an M4 MacBook it will use MPS automatically.
- Each training run takes roughly 1–3 hours on M4 depending on dataset size.

---

## Output files

After running, your output_dir will contain:

```
results/
├── all_results.json           ← test AUC for all 9 conditions in JSON
├── roc_brain.png              ← ROC curves for brain (3 conditions overlaid)
├── roc_prostate.png           ← ROC curves for prostate
├── roc_breast.png             ← ROC curves for breast
├── history_brain_magnitude.png
├── history_brain_phase.png
├── history_brain_both.png     ← training curves per run
└── ... (one history plot per run)

checkpoints/
├── brain_magnitude_best.pt    ← saved model weights (best val AUC)
├── brain_phase_best.pt
├── brain_both_best.pt
└── ... (one checkpoint per run)
```

The key output is the results table printed to the terminal and saved
in all_results.json. It looks like this:

```
================================================================
TEST AUC RESULTS
================================================================
Organ       Magnitude only (A)    Phase only (B)    Mag + Phase (C)
----------------------------------------------------------------
Brain       0.XXXX                0.XXXX            0.XXXX
Prostate    0.XXXX                0.XXXX            0.XXXX
Breast      0.XXXX                0.XXXX            0.XXXX
================================================================
```

If Condition C > Condition A across organs, phase adds diagnostic value.
Your prostate Condition C result should be close to the 2024 paper's
86.1% AUC if your pipeline is working correctly (the number won't
match exactly since we use a different classifier — that is expected).

---

## Important label notes per organ

### Prostate
Labels come from Pi-RADS scores stored in the h5 file attributes.
Pi-RADS >= 3 = label 1 (cancer risk), Pi-RADS <= 2 = label 0 (low risk).
The FastMRI prostate dataset also provides a separate CSV with labels —
if the h5 attributes don't contain Pi-RADS scores, use make_label_fn_from_dict()
in data_utils.py to load labels from that CSV instead.

### Brain
FastMRI brain does not have explicit tumor labels. The current code uses
contrast-enhanced T1 acquisitions as a proxy for pathology cases. This is
imperfect — for a stronger study you would need to supplement with the
fastMRI+ dataset which adds radiologist annotations.

### Breast
Labels (negative / benign / malignant) are stored as case-level attributes
in each h5 file. The code binarizes these to malignant=1, else=0.

---

## How to read your results

### Did we replicate the 2024 paper?
Look at prostate Condition C vs Condition A. If C > A, yes.
The 2024 paper got 86.1% AUC for phase-informed prostate DWI classification.
Your number will differ (different classifier) but the direction should match.

### Does phase generalize across organs?
If Condition C > Condition A for brain and breast as well, phase is a
general phenomenon. If only prostate shows improvement, the 2024 finding
may be prostate/DWI-specific — also an important and publishable result.

### Is the improvement statistically significant?
For the RSNA abstract you report AUC values. For a full paper you would
run DeLong's test to confirm the AUC difference is not due to chance.
This can be added as a follow-up step.

---

## The novelty claim (for your abstract)

Every prior MRI tumor AI paper uses magnitude images only. One 2024 paper
showed phase improves prostate classification. This is the first study to
systematically test whether that finding generalizes across brain, prostate,
and breast using FastMRI — the only public dataset providing labeled raw
k-space for all three organs.

---

## Prior art to cite

The key paper your study builds on:
  Rempe et al. (2024). "Tumor likelihood estimation on MRI prostate data
  by utilizing k-Space information." arXiv:2407.06165

FastMRI dataset papers:
  Zbontar et al. (2018). fastMRI: An Open Dataset and Benchmarks for
  Accelerated MRI. arXiv:1811.08839

  Tibrewala et al. (2024). FastMRI Prostate: A public, biparametric MRI
  dataset to advance machine learning for prostate cancer imaging.
  Scientific Data 11, 404.

  Solomon et al. (2025). FastMRI Breast: A Publicly Available Radial
  k-Space Dataset of Breast DCE-MRI. Radiology: Artificial Intelligence.

---

## Questions?

RSNA 2026 abstract deadline: May 6, 2026 at 12pm CT
Submit at: https://www.rsna.org/annual-meeting/abstract-submission
