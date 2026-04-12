"""
run_experiment.py — Phase-informed MRI tumor classification.
Usage:
    python run_experiment.py --prostate_dir ~/fastmri_local/prostate --output_dir ./results
    python run_experiment.py --smoke_test
"""

import argparse
import logging
import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils.data_utils import ProstateExamDataset
from models.models import build_model
from train import train
from evaluate import plot_training_history

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

MODES = ["magnitude", "phase", "both"]


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, mode, n_samples=64):
        self.mode = mode
        in_channels = 2 if mode == "both" else 1
        self.images = torch.randn(n_samples, in_channels, 64, 64)
        self.label_tensor = torch.randint(0, 2, (n_samples,), dtype=torch.long)
        self.samples = [(None, int(self.label_tensor[i])) for i in range(n_samples)]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return self.images[idx], self.label_tensor[idx]


def run_smoke_test(output_dir):
    logger.info("SMOKE TEST — synthetic data")
    ckpt_dir = str(Path(output_dir) / "checkpoints")
    res_dir  = str(Path(output_dir) / "results")
    Path(res_dir).mkdir(parents=True, exist_ok=True)

    for mode in MODES:
        run_name = f"smoke_{mode}"
        train_ds = SyntheticDataset(mode, 80)
        val_ds   = SyntheticDataset(mode, 20)
        model = build_model(mode=mode, pretrained=True)
        result = train(model, train_ds, val_ds, ckpt_dir, run_name,
                       num_epochs=2, batch_size=8, lr=1e-4,
                       num_workers=0, patience=5)
        logger.info(f"[{run_name}] best_val_auc={result['best_val_auc']:.4f}")
    logger.info("Smoke test complete.")


# ---------------------------------------------------------------------------
# Prostate experiment
# ---------------------------------------------------------------------------

def run_prostate(args, checkpoint_dir, results_dir):
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    # Find labels folder
    labels_dir = str(Path(args.prostate_dir) / "labels")
    if not Path(labels_dir).exists():
        found = list(Path(args.prostate_dir).rglob("volume_exam_labels.csv"))
        if not found:
            logger.error(f"Cannot find labels folder under {args.prostate_dir}")
            return
        labels_dir = str(found[0].parent)

    logger.info(f"Using labels from: {labels_dir}")

    all_results = {}
    for mode in MODES:
        run_name = f"prostate_{mode}"
        logger.info(f"\n{'='*50}\n{run_name}\n{'='*50}")

        full_ds = ProstateExamDataset(
            h5_dir=args.prostate_dir,
            labels_dir=labels_dir,
            mode=mode,
            n_slices_per_exam=3,
        )

        if len(full_ds) == 0:
            logger.error(f"No samples found for {run_name}.")
            continue

        # Stratified-ish split — shuffle with fixed seed then split
        n = len(full_ds)
        n_train = int(n * 0.70)
        n_val   = int(n * 0.15)
        n_test  = n - n_train - n_val

        g = torch.Generator().manual_seed(42)
        train_ds, val_ds, test_ds = torch.utils.data.random_split(
            full_ds, [n_train, n_val, n_test], generator=g
        )

        # Expose .samples so make_balanced_sampler works
        train_ds.samples = [full_ds.samples[i] for i in train_ds.indices]
        val_ds.samples   = [full_ds.samples[i] for i in val_ds.indices]
        test_ds.samples  = [full_ds.samples[i] for i in test_ds.indices]

        logger.info(f"Split: {len(train_ds)} train / {len(val_ds)} val / {len(test_ds)} test")

        model = build_model(mode=mode, pretrained=True)

        # Count trainable params
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in model.parameters())
        logger.info(f"Trainable params: {trainable:,} / {total:,} ({100*trainable/total:.1f}%)")

        result = train(
            model=model,
            train_dataset=train_ds,
            val_dataset=val_ds,
            checkpoint_dir=checkpoint_dir,
            run_name=run_name,
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            num_workers=args.num_workers,
            patience=7,
        )
        all_results[mode] = result
        plot_training_history(result["history"], run_name, results_dir)
        logger.info(f"[{run_name}] FINAL best_val_auc = {result['best_val_auc']:.4f}")

    # Print summary table
    logger.info("\n" + "="*50)
    logger.info("RESULTS SUMMARY")
    logger.info("="*50)
    for mode in MODES:
        if mode in all_results:
            auc = all_results[mode]["best_val_auc"]
            logger.info(f"  {mode:12s}: AUC = {auc:.4f}")
    logger.info("="*50)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--prostate_dir", type=str, default=None)
    parser.add_argument("--output_dir",   type=str, default="./results")
    parser.add_argument("--epochs",       type=int, default=30)
    parser.add_argument("--batch_size",   type=int, default=8)
    parser.add_argument("--lr",           type=float, default=1e-4)
    parser.add_argument("--num_workers",  type=int, default=0)
    parser.add_argument("--smoke_test",   action="store_true")
    parser.add_argument("--seed",         type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    checkpoint_dir = str(Path(args.output_dir) / "checkpoints")
    results_dir    = str(Path(args.output_dir) / "results")

    if args.smoke_test:
        run_smoke_test(args.output_dir)
        return

    if args.prostate_dir is None:
        logger.error("Please provide --prostate_dir")
        sys.exit(1)

    run_prostate(args, checkpoint_dir, results_dir)
    logger.info(f"All outputs saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
