"""
run_experiment.py
-----------------
Main entry point. Runs all 9 training conditions
(3 organs x 3 modes) and produces the final results table.

Usage:
    python run_experiment.py \
        --brain_dir    /path/to/fastmri/brain \
        --prostate_dir /path/to/fastmri/prostate \
        --breast_dir   /path/to/fastmri/breast \
        --output_dir   ./results \
        --epochs       20 \
        --batch_size   16

For a quick smoke test with dummy data (no FastMRI files needed):
    python run_experiment.py --smoke_test
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Callable

import numpy as np
import torch
from torch.utils.data import random_split

# Local imports
sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import (
    FastMRISliceDataset,
    brain_label_fn,
    prostate_label_fn,
    breast_label_fn,
    make_label_fn_from_dict,
)
from models.models import build_model
from train import train
from evaluate import (
    run_test_evaluation,
    print_results_table,
    plot_roc_curves,
    plot_training_history,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODES  = ["magnitude", "phase", "both"]
ORGANS = ["brain", "prostate", "breast"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def get_h5_files(directory: str) -> List[str]:
    """Recursively find all .h5 files in a directory."""
    p = Path(directory)
    if not p.exists():
        raise FileNotFoundError(f"Directory not found: {directory}")
    files = sorted(p.rglob("*.h5"))
    if not files:
        raise FileNotFoundError(f"No .h5 files found in {directory}")
    logger.info(f"Found {len(files)} h5 files in {directory}")
    return [str(f) for f in files]


def split_files(
    files: List[str],
    train_frac: float = 0.7,
    val_frac: float = 0.15,
    seed: int = 42,
) -> tuple:
    """Split file list into train/val/test."""
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(files))
    n_train = int(len(files) * train_frac)
    n_val   = int(len(files) * val_frac)
    train_files = [files[i] for i in idx[:n_train]]
    val_files   = [files[i] for i in idx[n_train: n_train + n_val]]
    test_files  = [files[i] for i in idx[n_train + n_val:]]
    return train_files, val_files, test_files


def build_datasets(
    train_files: List[str],
    val_files: List[str],
    test_files: List[str],
    label_fn: Callable,
    slice_range=None,
) -> Dict[str, Dict[str, FastMRISliceDataset]]:
    """
    Build train/val/test datasets for all 3 modes.
    Returns: {"train": {mode: dataset}, "val": {...}, "test": {...}}
    """
    splits = {}
    for split_name, file_list in [
        ("train", train_files),
        ("val",   val_files),
        ("test",  test_files),
    ]:
        splits[split_name] = {}
        for mode in MODES:
            splits[split_name][mode] = FastMRISliceDataset(
                file_paths=file_list,
                label_fn=label_fn,
                mode=mode,
                slice_range=slice_range,
            )
    return splits


# ---------------------------------------------------------------------------
# Smoke test with synthetic data
# ---------------------------------------------------------------------------

class SyntheticDataset(torch.utils.data.Dataset):
    """
    Generates random (C, 320, 320) tensors with random binary labels.
    Used to verify the full pipeline runs end-to-end without FastMRI data.
    """

    def __init__(self, mode: str, n_samples: int = 64):
        self.mode = mode
        self.n_samples = n_samples
        in_channels = 2 if mode == "both" else 1
        self.images = torch.randn(n_samples, in_channels, 64, 64)
        # Store as torch.long tensor — avoids Python 3.14 int collation bugs
        self.label_tensor = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        self.labels = self.label_tensor.tolist()
        # Expose .samples so make_balanced_sampler works
        self.samples = [(None, i, int(self.labels[i])) for i in range(n_samples)]

    def __len__(self):
        return self.n_samples

    def __getitem__(self, idx):
        # Return torch.long label — DataLoader collates this correctly
        return self.images[idx], self.label_tensor[idx]


def run_smoke_test(output_dir: str):
    """Run a full end-to-end test with synthetic random data."""
    logger.info("=" * 60)
    logger.info("SMOKE TEST — using synthetic random data")
    logger.info("=" * 60)

    checkpoint_dir = str(Path(output_dir) / "checkpoints")
    results_dir    = str(Path(output_dir) / "results")

    all_train_results = {}

    for organ in ORGANS:
        all_train_results[organ] = {}
        for mode in MODES:
            run_name = f"{organ}_{mode}"
            logger.info(f"\n--- {run_name} ---")

            train_ds = SyntheticDataset(mode, n_samples=80)
            val_ds   = SyntheticDataset(mode, n_samples=20)

            model = build_model(mode=mode, pretrained=True)

            result = train(
                model=model,
                train_dataset=train_ds,
                val_dataset=val_ds,
                checkpoint_dir=checkpoint_dir,
                run_name=run_name,
                num_epochs=3,
                batch_size=8,
                lr=1e-4,
                num_workers=0,
                use_amp=False,
                patience=5,
            )
            all_train_results[organ][mode] = result
            plot_training_history(
                result["history"], run_name, results_dir
            )

    # Evaluation on synthetic test sets
    test_datasets = {
        organ: {mode: SyntheticDataset(mode, n_samples=30) for mode in MODES}
        for organ in ORGANS
    }

    # Give synthetic datasets a .samples attribute for the evaluator
    all_results = run_test_evaluation(
        test_datasets=test_datasets,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        batch_size=8,
        num_workers=0,
    )

    print_results_table(all_results)
    plot_roc_curves(all_results, results_dir)
    logger.info("Smoke test complete. Check output_dir for results.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def run_prostate(args, checkpoint_dir, results_dir):
    """Run all 3 conditions for prostate using CSV labels."""
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from utils.data_utils import load_prostate_labels, ProstateSliceDataset
    from models.models import build_model
    from train import train
    from evaluate import plot_training_history

    labels_dir = str(Path(args.prostate_dir) / "labels")
    if not Path(labels_dir).exists():
        # Try finding labels folder anywhere under prostate_dir
        found = list(Path(args.prostate_dir).rglob("dwi_slice_level_labels.csv"))
        if found:
            labels_dir = str(found[0].parent)
        else:
            logger.error(f"Cannot find labels folder under {args.prostate_dir}")
            return {}

    slice_labels = load_prostate_labels(labels_dir)
    h5_dir = args.prostate_dir

    all_results = {}
    for mode in MODES:
        run_name = f"prostate_{mode}"
        logger.info(f"\n--- {run_name} ---")

        full_ds = ProstateSliceDataset(h5_dir, slice_labels, mode=mode)

        if len(full_ds) == 0:
            logger.error(f"No samples found for {run_name}. Check your paths.")
            continue

        # 70/15/15 split
        n = len(full_ds)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        n_test  = n - n_train - n_val

        generator = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n_test], generator=generator
        )

        # Expose .samples for WeightedRandomSampler
        train_ds.samples = [full_ds.samples[i] for i in train_ds.indices]
        val_ds.samples   = [full_ds.samples[i] for i in val_ds.indices]
        test_ds.samples  = [full_ds.samples[i] for i in test_ds.indices]

        model = build_model(mode=mode, pretrained=True)
        result = train(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            patience=5,
        )
        all_results[mode] = result
        plot_training_history(result["history"], run_name, results_dir)

    return all_results

def main():
    parser = argparse.ArgumentParser(
        description="Phase-informed MRI tumor classification experiment"
    )
    parser.add_argument("--brain_dir",    type=str, default=None)
    parser.add_argument("--prostate_dir", type=str, default=None)
    parser.add_argument("--breast_dir",   type=str, default=None)
    parser.add_argument("--output_dir",   type=str, default="./experiment_output")
    parser.add_argument("--epochs",       type=int, default=20)
    parser.add_argument("--batch_size",   type=int, default=16)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int, default=0,
                        help="Set 0 on Mac to avoid h5py multiprocessing issues")
    parser.add_argument("--smoke_test",   action="store_true",
                        help="Run with synthetic data to verify pipeline")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    results_dir    = str(Path(args.output_dir) / "results")

    if args.smoke_test:
        run_smoke_test(args.output_dir)
        return

    # Validate directories
    organ_dirs = {
        "brain":    args.brain_dir,
        "prostate": args.prostate_dir,
        "breast":   args.breast_dir,
    }
    label_fns = {
        "brain":    brain_label_fn,
        "prostate": prostate_label_fn,
        "breast":   breast_label_fn,
    }

    missing = [o for o, d in organ_dirs.items() if d is None]
    if missing:
        logger.warning(
            f"No data directory provided for: {missing}. "
            f"These organs will be skipped."
        )

    # ------------------------------------------------------------------
    # Build datasets and train all conditions
    # ------------------------------------------------------------------
    all_train_results: Dict = {}
    test_datasets: Dict = {}

    for organ in ORGANS:
        data_dir = organ_dirs[organ]
        if data_dir is None:
            continue

        logger.info(f"\n{'='*60}\nProcessing organ: {organ.upper()}\n{'='*60}")

        try:
            files = get_h5_files(data_dir)
        except FileNotFoundError as e:
            logger.error(str(e))
            continue

        train_files, val_files, test_files = split_files(
            files, train_frac=0.7, val_frac=0.15, seed=args.seed
        )
        logger.info(
            f"Split: {len(train_files)} train / "
            f"{len(val_files)} val / {len(test_files)} test files"
        )

        datasets = build_datasets(
            train_files, val_files, test_files,
            label_fn=label_fns[organ],
            slice_range=(5, 25),  # skip first/last slices (often blank)
        )

        all_train_results[organ] = {}
        test_datasets[organ] = datasets["test"]

        for mode in MODES:
            run_name = f"{organ}_{mode}"
            logger.info(f"\n--- Training: {run_name} ---")

            model = build_model(mode=mode, pretrained=True)

            result = train(
                model=model,
                train_dataset=datasets["train"][mode],
                val_dataset=datasets["val"][mode],
                checkpoint_dir=checkpoint_dir,
                run_name=run_name,
                num_epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                num_workers=args.num_workers,
                use_amp=False,  # MPS on Mac doesn't support amp
                patience=5,
            )

            all_train_results[organ][mode] = result
            plot_training_history(
                result["history"], run_name, results_dir
            )

    # ------------------------------------------------------------------
    # Final test evaluation
    # ------------------------------------------------------------------
    if not test_datasets:
        logger.error("No organs were successfully processed. Exiting.")
        sys.exit(1)

    logger.info("\n" + "=" * 60)
    logger.info("FINAL TEST EVALUATION")
    logger.info("=" * 60)

    all_results = run_test_evaluation(
        test_datasets=test_datasets,
        checkpoint_dir=checkpoint_dir,
        results_dir=results_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    print_results_table(all_results)
    plot_roc_curves(all_results, results_dir)

    logger.info(f"\nAll outputs saved to: {args.output_dir}")
    logger.info("Done.")


if __name__ == "__main__":
    main()
