"""
data_utils.py — FastMRI prostate k-space loading, phase extraction, datasets.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, List, Dict
import logging
import pandas as pd

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# k-space math
# ---------------------------------------------------------------------------

def ifft2c(kspace):
    """Centered inverse 2D FFT. Input/output: (..., H, W) complex."""
    return np.fft.ifftshift(
        np.fft.ifft2(np.fft.ifftshift(kspace, axes=(-2, -1)), axes=(-2, -1)),
        axes=(-2, -1)
    )


def coil_combine_rss(kspace):
    """
    RSS coil combination.
    Input:  (1, coils, H, W) complex
    Output: (H, W) float32
    """
    imgs = ifft2c(kspace[0])           # (coils, H, W) complex
    magnitude = np.sqrt(np.sum(np.abs(imgs) ** 2, axis=0))
    return magnitude.astype(np.float32)


def extract_phase_pca(kspace):
    """
    PCA virtual coil then extract phase.
    Input:  (1, coils, H, W) complex
    Output: (H, W) float32 in [-pi, pi]
    """
    imgs = ifft2c(kspace[0])           # (coils, H, W) complex
    n_coils = imgs.shape[0]
    X = imgs.reshape(n_coils, -1)      # (coils, H*W)
    X_ri = np.concatenate([X.real, X.imag], axis=0)   # (2*coils, H*W)
    cov = X_ri @ X_ri.T / (X_ri.shape[1] + 1e-8)
    _, vecs = np.linalg.eigh(cov)
    top = vecs[:, -1]
    w = (top[:n_coils] + 1j * top[n_coils:])
    w /= (np.linalg.norm(w) + 1e-8)
    virtual = np.einsum("c,chw->hw", w, imgs)
    return np.angle(virtual).astype(np.float32)


def normalize(arr, eps=1e-8):
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + eps)


def resize_2d(arr, target=(160, 160)):
    th, tw = target
    H, W = arr.shape
    # Height
    if H >= th:
        s = (H - th) // 2
        arr = arr[s: s + th, :]
    else:
        p = th - H
        arr = np.pad(arr, ((p // 2, p - p // 2), (0, 0)))
    # Width
    if W >= tw:
        s = (W - tw) // 2
        arr = arr[:, s: s + tw]
    else:
        p = tw - W
        arr = np.pad(arr, ((0, 0), (p // 2, p - p // 2)))
    return arr


def kspace_to_channels(kspace_raw, mode):
    """
    Convert raw kspace slice to image channels.
    kspace_raw: (coils, H, W) or (coils, dirs, H, W) complex
    Returns: list of (H, W) float32 arrays
    """
    k = kspace_raw.astype(np.complex64)
    if k.ndim == 4:
        k = k.mean(axis=1)   # average over diffusion directions
    k = k[np.newaxis]        # (1, coils, H, W)

    channels = []
    if mode in ("magnitude", "both"):
        mag = coil_combine_rss(k)
        channels.append(resize_2d(normalize(mag)))
    if mode in ("phase", "both"):
        phase = extract_phase_pca(k)
        phase = (phase + np.pi) / (2 * np.pi)   # map to [0, 1]
        channels.append(resize_2d(phase))
    return channels


# ---------------------------------------------------------------------------
# Label loading
# ---------------------------------------------------------------------------

def load_exam_labels(labels_dir: str) -> Dict[int, int]:
    """
    Load exam-level labels from volume_exam_labels.csv.
    Returns {patient_id: binary_label} where 1 = cancer risk (Pi-RADS >= 3).
    Uses the more conservative exam_level column.
    """
    path = Path(labels_dir) / "volume_exam_labels.csv"
    df = pd.read_csv(path)
    result = {}
    for _, row in df.iterrows():
        pt_id = int(row["fastmri_pt_id"])
        # exam_level is the most reliable — radiologist consensus across sequences
        pirads = int(row["exam_level"])
        result[pt_id] = 1 if pirads >= 3 else 0
    pos = sum(v for v in result.values())
    logger.info(f"Loaded exam labels: {len(result)} patients "
                f"({pos} positive, {len(result)-pos} negative)")
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProstateExamDataset(Dataset):
    """
    One sample per patient h5 file.
    Loads n_slices evenly from the middle of the volume,
    processes each to image channels, then averages across slices.
    Works for both DIFF and T2 sequences.
    """

    def __init__(self, h5_dir, labels_dir, mode="both",
                 target_size=(160, 160), n_slices_per_exam=3):
        assert mode in ("magnitude", "phase", "both")
        self.mode = mode
        self.target_size = target_size
        self.n_slices = n_slices_per_exam
        self.in_channels = 2 if mode == "both" else 1

        self.exam_labels = load_exam_labels(labels_dir)
        self.samples = []   # list of (h5_path, label)
        self._build_index(h5_dir)

    def _build_index(self, h5_dir):
        files = sorted(Path(h5_dir).rglob("*.h5"))
        files = [f for f in files if not f.name.startswith("._")]
        for f in files:
            try:
                pt_id = int(f.stem.split("_")[-1])
            except ValueError:
                continue
            if pt_id not in self.exam_labels:
                continue
            self.samples.append((str(f), self.exam_labels[pt_id]))

        pos = sum(l for _, l in self.samples)
        neg = len(self.samples) - pos
        logger.info(f"ProstateExamDataset [{self.mode}]: "
                    f"{len(self.samples)} exams ({pos}+, {neg}-)")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        dummy = torch.zeros(self.in_channels, *self.target_size)
        dummy_label = torch.tensor(label, dtype=torch.long)

        try:
            with h5py.File(path, "r") as f:
                n_total = f["kspace"].shape[0]
                start = max(1, n_total // 4)
                end   = min(n_total - 1, 3 * n_total // 4)
                if end <= start:
                    return dummy, dummy_label
                idxs = np.linspace(start, end, self.n_slices, dtype=int)
                raw_slices = [f["kspace"][i] for i in idxs]
        except Exception as e:
            logger.debug(f"Skip {path}: {e}")
            return dummy, dummy_label

        try:
            slice_tensors = []
            for raw in raw_slices:
                chs = kspace_to_channels(raw, self.mode)
                if len(chs) == 0:
                    continue
                slice_tensors.append(np.stack(chs, axis=0))

            if not slice_tensors:
                return dummy, dummy_label

            # Average across slices then convert to tensor
            image = np.mean(slice_tensors, axis=0).astype(np.float32)
            return torch.from_numpy(image), dummy_label

        except Exception as e:
            logger.debug(f"Process error {path}: {e}")
            return dummy, dummy_label
