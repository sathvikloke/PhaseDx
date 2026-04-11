"""
data_utils.py
-------------
Handles loading raw k-space from FastMRI h5 files,
extracting magnitude and phase maps, and building
labeled slice-level datasets for prostate and breast.

Prostate labels come from the FastMRI CSV files:
  - dwi_slice_level_labels.csv
  - t2_slice_level_labels.csv
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List, Dict
import logging
import pandas as pd

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# k-space utilities
# ---------------------------------------------------------------------------

def coil_combine_rss(kspace: np.ndarray) -> np.ndarray:
    """
    Root-Sum-of-Squares coil combination.
    Input:  (slices, coils, H, W) complex
    Output: (slices, H, W) float32 magnitude
    """
    image_coils = np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    )
    magnitude = np.sqrt(np.sum(np.abs(image_coils) ** 2, axis=1))
    return magnitude.astype(np.float32)


def extract_phase_pca(kspace: np.ndarray) -> np.ndarray:
    """
    PCA coil compression then extract phase.
    Input:  (slices, coils, H, W) complex
    Output: (slices, H, W) float32 phase in [-pi, pi]
    """
    n_slices, n_coils, H, W = kspace.shape
    image_coils = np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    )
    phase_maps = []
    for s in range(n_slices):
        X = image_coils[s].reshape(n_coils, -1)
        X_ri = np.concatenate([X.real, X.imag], axis=0)
        cov = X_ri @ X_ri.T / (X_ri.shape[1] + 1e-8)
        _, eigenvectors = np.linalg.eigh(cov)
        top_vec = eigenvectors[:, -1]
        w = (top_vec[:n_coils] + 1j * top_vec[n_coils:])
        w = w / (np.linalg.norm(w) + 1e-8)
        virtual_coil = np.einsum("c,chw->hw", w, image_coils[s])
        phase_maps.append(np.angle(virtual_coil))
    return np.stack(phase_maps, axis=0).astype(np.float32)


def normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    arr_min, arr_max = arr.min(), arr.max()
    return (arr - arr_min) / (arr_max - arr_min + eps)


def resize_2d(arr: np.ndarray, target: Tuple[int, int] = (320, 320)) -> np.ndarray:
    th, tw = target
    H, W = arr.shape
    if H >= th:
        start = (H - th) // 2
        arr = arr[start: start + th, :]
    else:
        pad = th - H
        arr = np.pad(arr, ((pad // 2, pad - pad // 2), (0, 0)))
    if W >= tw:
        start = (W - tw) // 2
        arr = arr[:, start: start + tw]
    else:
        pad = tw - W
        arr = np.pad(arr, ((0, 0), (pad // 2, pad - pad // 2)))
    return arr


# ---------------------------------------------------------------------------
# Label loading from CSV
# ---------------------------------------------------------------------------

def load_prostate_labels(labels_dir: str) -> Dict[str, Dict[int, int]]:
    """
    Load prostate slice-level labels from FastMRI CSVs.

    Returns dict: {h5_filename: {slice_idx: binary_label}}
    where binary_label = 1 if PIRADS >= 3, else 0.

    slice_idx is 0-based (CSV uses 1-based, we convert).
    """
    labels_dir = Path(labels_dir)
    result = {}

    for csv_name in ["dwi_slice_level_labels.csv", "t2_slice_level_labels.csv"]:
        csv_path = labels_dir / csv_name
        if not csv_path.exists():
            logger.warning(f"Label CSV not found: {csv_path}")
            continue

        df = pd.read_csv(csv_path)
        for _, row in df.iterrows():
            fname = str(row["fastmri_rawfile"]).strip()
            slice_idx = int(row["slice"]) - 1  # convert 1-based to 0-based
            pirads = int(row["PIRADS"])
            label = 1 if pirads >= 3 else 0

            if fname not in result:
                result[fname] = {}
            result[fname][slice_idx] = label

    logger.info(f"Loaded labels for {len(result)} h5 files from CSVs")
    return result


def load_breast_labels(labels_path: str) -> Dict[str, int]:
    """
    Load breast case-level labels.
    Returns dict: {h5_filename: binary_label}
    where binary_label = 1 if malignant, else 0.
    """
    df = pd.read_csv(labels_path)
    result = {}
    for _, row in df.iterrows():
        fname = str(row.get("file", row.get("filename", ""))).strip()
        status = str(row.get("lesion_status", row.get("label", "negative"))).lower()
        result[fname] = 1 if "malignant" in status else 0
    logger.info(f"Loaded breast labels for {len(result)} files")
    return result


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class ProstateSliceDataset(Dataset):
    """
    Slice-level dataset for FastMRI Prostate.
    Uses slice-level CSV labels (PIRADS per slice).

    mode: 'magnitude' | 'phase' | 'both'
    """

    def __init__(
        self,
        h5_dir: str,
        slice_labels: Dict[str, Dict[int, int]],
        mode: str = "both",
        target_size: Tuple[int, int] = (320, 320),
    ):
        assert mode in ("magnitude", "phase", "both")
        self.mode = mode
        self.target_size = target_size
        self.slice_labels = slice_labels

        # Build flat list of (h5_path, slice_idx, label)
        self.samples: List[Tuple[str, int, int]] = []
        self._build_index(h5_dir)

    def _build_index(self, h5_dir: str):
        h5_files = sorted(Path(h5_dir).rglob("*.h5"))
        # Filter out macOS metadata files
        h5_files = [f for f in h5_files if not f.name.startswith("._")]

        for h5_path in h5_files:
            fname = h5_path.name
            if fname not in self.slice_labels:
                continue
            slice_label_map = self.slice_labels[fname]
            for slice_idx, label in slice_label_map.items():
                self.samples.append((str(h5_path), slice_idx, label))

        pos = sum(1 for _, _, l in self.samples if l == 1)
        neg = sum(1 for _, _, l in self.samples if l == 0)
        logger.info(
            f"ProstateSliceDataset: {len(self.samples)} slices "
            f"({pos} positive, {neg} negative), mode='{self.mode}'"
        )

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        path, slice_idx, label = self.samples[idx]

        with h5py.File(path, "r") as f:
            kspace_slice = f["kspace"][slice_idx]  # (coils, H, W)

        kspace = kspace_slice.astype(np.complex64)
        # DWI has shape (coils, directions, H, W) — average across directions
        if kspace.ndim == 4:
            kspace = kspace.mean(axis=1)  # (coils, H, W)
        kspace = kspace[np.newaxis]  # (1, coils, H, W)

        channels = []
        if self.mode in ("magnitude", "both"):
            mag = coil_combine_rss(kspace)[0]
            mag = resize_2d(normalize(mag), self.target_size)
            channels.append(mag)

        if self.mode in ("phase", "both"):
            phase = extract_phase_pca(kspace)[0]
            phase = (phase + np.pi) / (2 * np.pi)
            phase = resize_2d(phase, self.target_size)
            channels.append(phase)

        image = torch.from_numpy(
            np.stack(channels, axis=0).astype(np.float32)
        )
        return image, torch.tensor(label, dtype=torch.long)
