"""
data_utils.py
-------------
Handles loading raw k-space from FastMRI h5 files,
extracting magnitude and phase maps, and building
labeled slice-level datasets for brain, prostate, and breast.
"""

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path
from typing import Tuple, Optional, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Low-level k-space utilities
# ---------------------------------------------------------------------------

def load_kspace(h5_path: str) -> np.ndarray:
    """
    Load raw multi-coil k-space from a FastMRI h5 file.
    Returns complex64 array of shape (slices, coils, H, W).
    """
    with h5py.File(h5_path, "r") as f:
        if "kspace" not in f:
            raise KeyError(f"'kspace' key not found in {h5_path}. "
                           f"Available keys: {list(f.keys())}")
        kspace = f["kspace"][()]  # shape: (slices, coils, H, W)
    return kspace.astype(np.complex64)


def coil_combine_rss(kspace: np.ndarray) -> np.ndarray:
    """
    Root-Sum-of-Squares coil combination.
    Input:  (slices, coils, H, W) complex
    Output: (slices, H, W) real — magnitude image
    """
    # Inverse FFT along spatial dims (last two axes)
    image_coils = np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    )
    # RSS across coil dimension
    magnitude = np.sqrt(np.sum(np.abs(image_coils) ** 2, axis=1))
    return magnitude.astype(np.float32)  # (slices, H, W)


def extract_phase_pca(kspace: np.ndarray, n_components: int = 1) -> np.ndarray:
    """
    PCA coil compression then extract phase.
    Reduces multi-coil k-space to a single virtual coil via PCA,
    then returns the phase of the resulting image.

    Input:  (slices, coils, H, W) complex
    Output: (slices, H, W) real — phase map in [-pi, pi]
    """
    n_slices, n_coils, H, W = kspace.shape

    # Transform to image domain per coil
    image_coils = np.fft.ifftshift(
        np.fft.ifft2(
            np.fft.ifftshift(kspace, axes=(-2, -1)),
            axes=(-2, -1)
        ),
        axes=(-2, -1)
    )  # (slices, coils, H, W) complex

    phase_maps = []
    for s in range(n_slices):
        # Flatten spatial dims: shape (coils, H*W)
        X = image_coils[s].reshape(n_coils, -1)  # complex (coils, pixels)

        # Stack real/imag for PCA
        X_ri = np.concatenate([X.real, X.imag], axis=0)  # (2*coils, pixels)

        # Simple covariance-based PCA — find dominant direction
        cov = X_ri @ X_ri.T / X_ri.shape[1]
        eigenvalues, eigenvectors = np.linalg.eigh(cov)
        # Take eigenvector corresponding to largest eigenvalue
        top_vec = eigenvectors[:, -1]  # (2*coils,)

        # Project: split back to real/imag weights
        w_real = top_vec[:n_coils]
        w_imag = top_vec[n_coils:]
        w_complex = w_real + 1j * w_imag  # (n_coils,)
        w_complex = w_complex / (np.linalg.norm(w_complex) + 1e-8)

        # Virtual coil image: weighted sum across coils
        virtual_coil = np.einsum("c,chw->hw", w_complex, image_coils[s])
        phase_maps.append(np.angle(virtual_coil))  # (H, W)

    return np.stack(phase_maps, axis=0).astype(np.float32)  # (slices, H, W)


def normalize(arr: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Min-max normalize array to [0, 1]."""
    arr_min = arr.min()
    arr_max = arr.max()
    return (arr - arr_min) / (arr_max - arr_min + eps)


# ---------------------------------------------------------------------------
# Dataset classes
# ---------------------------------------------------------------------------

class FastMRISliceDataset(Dataset):
    """
    Slice-level dataset for FastMRI brain/prostate/breast.

    mode:
      'magnitude' -> 1-channel input  (Condition A)
      'phase'     -> 1-channel input  (Condition B)
      'both'      -> 2-channel input  (Condition C)

    label_fn: callable(h5_path) -> int
        User-supplied function that returns 0/1 label for a given h5 file.
    """

    def __init__(
        self,
        file_paths: List[str],
        label_fn,
        mode: str = "both",
        slice_range: Optional[Tuple[int, int]] = None,
        transform=None,
        target_size: Tuple[int, int] = (320, 320),
    ):
        assert mode in ("magnitude", "phase", "both"), \
            f"mode must be 'magnitude', 'phase', or 'both', got '{mode}'"

        self.mode = mode
        self.label_fn = label_fn
        self.transform = transform
        self.target_size = target_size
        self.slice_range = slice_range

        # Build flat list of (file_path, slice_idx, label)
        self.samples: List[Tuple[str, int, int]] = []
        self._build_index(file_paths)

    def _build_index(self, file_paths: List[str]):
        for path in file_paths:
            try:
                label = self.label_fn(path)
                with h5py.File(path, "r") as f:
                    if "kspace" not in f:
                        logger.warning(f"No kspace in {path}, skipping.")
                        continue
                    n_slices = f["kspace"].shape[0]

                start = self.slice_range[0] if self.slice_range else 0
                end = self.slice_range[1] if self.slice_range else n_slices
                end = min(end, n_slices)

                for sl in range(start, end):
                    self.samples.append((path, sl, label))

            except Exception as e:
                logger.warning(f"Skipping {path}: {e}")

        logger.info(f"Built dataset with {len(self.samples)} slices "
                    f"from {len(file_paths)} files, mode='{self.mode}'")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, slice_idx, label = self.samples[idx]

        with h5py.File(path, "r") as f:
            # Load single slice k-space: (coils, H, W)
            kspace_slice = f["kspace"][slice_idx]  # complex

        # Add slice dim for utility functions: (1, coils, H, W)
        kspace_slice = kspace_slice[np.newaxis]

        channels = []

        if self.mode in ("magnitude", "both"):
            mag = coil_combine_rss(kspace_slice)[0]  # (H, W)
            mag = normalize(mag)
            mag = self._resize(mag)
            channels.append(mag)

        if self.mode in ("phase", "both"):
            phase = extract_phase_pca(kspace_slice)[0]  # (H, W) in [-pi, pi]
            # Normalize phase to [0, 1]
            phase = (phase + np.pi) / (2 * np.pi)
            phase = self._resize(phase)
            channels.append(phase)

        # Stack channels: (C, H, W)
        image = np.stack(channels, axis=0).astype(np.float32)
        tensor = torch.from_numpy(image)

        if self.transform:
            tensor = self.transform(tensor)

        return tensor, label

    def _resize(self, arr: np.ndarray) -> np.ndarray:
        """Resize 2D array to target_size using simple crop/pad."""
        H, W = arr.shape
        th, tw = self.target_size

        # Center crop or pad height
        if H >= th:
            start = (H - th) // 2
            arr = arr[start: start + th, :]
        else:
            pad = th - H
            arr = np.pad(arr, ((pad // 2, pad - pad // 2), (0, 0)))

        # Center crop or pad width
        if W >= tw:
            start = (W - tw) // 2
            arr = arr[:, start: start + tw]
        else:
            pad = tw - W
            arr = np.pad(arr, ((0, 0), (pad // 2, pad - pad // 2)))

        return arr


# ---------------------------------------------------------------------------
# Label functions for each organ
# ---------------------------------------------------------------------------

def brain_label_fn(h5_path: str) -> int:
    """
    For FastMRI brain: 1 if file metadata indicates pathology, else 0.
    FastMRI brain files store acquisition metadata in 'ismrmrd_header'.
    Since brain FastMRI doesn't have explicit tumor labels, we use
    contrast agent presence as a proxy (contrast-enhanced T1 = likely pathology).
    
    NOTE: For a real study you would use radiologist annotations.
    This is a placeholder showing the pattern — replace with your label source.
    """
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        # Check if acquisition type suggests pathology scan
        acq = str(attrs.get("acquisition", "")).lower()
        # T1 post-contrast scans are more likely to be pathology workups
        return 1 if "t1" in acq and "post" in acq else 0


def prostate_label_fn(h5_path: str) -> int:
    """
    FastMRI Prostate: binarize Pi-RADS score.
    Pi-RADS >= 3 -> clinically significant cancer risk -> label 1
    Pi-RADS <= 2 -> low risk -> label 0
    
    The FastMRI prostate dataset stores labels in a separate CSV.
    This function expects the label to be stored in h5 attributes,
    or you can pass a pre-built label dict instead.
    """
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        pirads = int(attrs.get("pirads_score", 0))
    return 1 if pirads >= 3 else 0


def breast_label_fn(h5_path: str) -> int:
    """
    FastMRI Breast: 1 if malignant, 0 if benign or negative.
    The breast dataset stores case-level lesion status in h5 attributes.
    """
    with h5py.File(h5_path, "r") as f:
        attrs = dict(f.attrs)
        status = str(attrs.get("lesion_status", "negative")).lower()
    return 1 if status == "malignant" else 0


def make_label_fn_from_dict(label_dict: dict):
    """
    Factory: create a label function from a pre-built {filename: label} dict.
    Useful when labels come from a CSV (e.g. FastMRI prostate label CSV).
    
    Usage:
        labels = {"file001.h5": 1, "file002.h5": 0, ...}
        fn = make_label_fn_from_dict(labels)
        dataset = FastMRISliceDataset(paths, label_fn=fn, mode="both")
    """
    def label_fn(h5_path: str) -> int:
        key = Path(h5_path).name
        if key not in label_dict:
            raise KeyError(f"No label found for {key} in label_dict")
        return int(label_dict[key])
    return label_fn
