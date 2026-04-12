# PhaseDx

MRI scanners collect data in a format called k-space — a grid of complex numbers with both magnitude and phase components. Every standard reconstruction pipeline throws the phase away. Every AI tumor classifier ever published trains on what's left: the magnitude image.

A 2024 paper showed that keeping phase information improved prostate cancer classification on DWI MRI. We wanted to know if that finding was real and generalizable, or specific to one organ and one sequence.

This project runs a three-condition ablation — magnitude only, phase only, and magnitude + phase — across prostate and breast MRI using the FastMRI dataset, which is one of the only public datasets that provides raw k-space data with tumor labels.

---

## What we found (prostate)

| Input | AUC (seed 1) | AUC (seed 2) |
|-------|-------------|-------------|
| Magnitude only | 0.779 | 0.636 |
| Phase only | 0.883 | 0.818 |
| Magnitude + phase | 0.766 | 0.766 |

Phase alone consistently outperformed magnitude alone across both random seeds. The combined model didn't outperform phase alone, which suggests magnitude may be adding noise rather than signal in this setting. This is a stronger result than the 2024 paper, which only tested phase combined with magnitude — never in isolation.

---

## Dataset

This uses the [FastMRI dataset](https://fastmri.med.nyu.edu/) from NYU Langone Health. You need to apply for access — it's free but requires a data use agreement.

**What to download:**

| Organ | Files | Size | Labels |
|-------|-------|------|--------|
| Prostate | T2 + DWI folders | ~500 GB total | Pi-RADS scores (CSV) |
| Breast | Select high-malignancy blocks | ~85 GB each | Benign/malignant (XLSX) |

Don't download everything — the full dataset is several terabytes. For prostate, the labels come in a separate `labels.tar` file — download that first.

For breast, the malignant cases are concentrated in patients 131–160 and 261–300. Downloading those blocks gives you the most signal per gigabyte.

---

## Setup

```bash
git clone https://github.com/sathvikloke/PhaseDx.git
cd PhaseDx
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Run the smoke test first (no data needed):

```bash
python run_experiment.py --smoke_test
```

---

## Running experiments

**Prostate only:**
```bash
python run_experiment.py \
  --prostate_dir /path/to/fastmri/prostate \
  --output_dir ./results \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4
```

**Breast only:**
```bash
python run_experiment.py \
  --breast_dir /path/to/fastmri/breast \
  --output_dir ./results \
  --epochs 30 \
  --batch_size 8 \
  --lr 1e-4
```

You don't need both organs — the code skips whichever directory you don't provide.

**Apple Silicon (M1/M2/M3/M4):** The code auto-detects MPS and uses it automatically. Set `--num_workers 0` (already the default) to avoid h5py multiprocessing issues.

---

## Project structure

```
PhaseDx/
├── run_experiment.py     — entry point
├── train.py              — training loop with early stopping
├── evaluate.py           — AUC evaluation and ROC curves
├── utils/
│   └── data_utils.py     — k-space loading, phase extraction, dataset classes
└── models/
    └── models.py         — ResNet-18 adapted for 1 or 2 input channels
```

The core of the project is in `data_utils.py`. Phase extraction uses PCA virtual coil compression followed by `np.angle()` on the resulting complex image. For prostate DWI the k-space is Cartesian; for breast DCE it's radial, so we apply a 1D iFFT along the readout dimension instead.

---

## How labels work

**Prostate:** Slice-level Pi-RADS scores from `dwi_slice_level_labels.csv` and `t2_slice_level_labels.csv`. We use exam-level labels from `volume_exam_labels.csv` for training — one label per patient, Pi-RADS ≥ 3 = positive.

**Breast:** Case-level labels from `fastMRI_breast_labels.xlsx`. Lesion status 1 (malignancy) = positive, everything else = negative.

---

## References

**The paper this builds on:**
Rempe et al. (2024). *Tumor likelihood estimation on MRI prostate data by utilizing k-Space information.* arXiv:2407.06165

**FastMRI dataset papers:**
- Zbontar et al. (2018). *fastMRI: An Open Dataset and Benchmarks for Accelerated MRI.* arXiv:1811.08839
- Tibrewala et al. (2024). *FastMRI Prostate: A public, biparametric MRI dataset to advance machine learning for prostate cancer imaging.* Scientific Data 11, 404.
- Solomon et al. (2025). *FastMRI Breast: A Publicly Available Radial k-Space Dataset of Breast DCE-MRI.* Radiology: Artificial Intelligence.
