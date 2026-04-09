"""
evaluate.py
-----------
Load saved checkpoints and compute final test AUC for all
9 conditions (3 organs x 3 input modes).
Also generates the results table and ROC curve plots.
"""

import torch
import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.metrics import roc_auc_score, roc_curve
from torch.utils.data import DataLoader
from pathlib import Path
import json
import logging
from typing import Dict, List

from models.models import build_model
from train import evaluate, _get_device

logger = logging.getLogger(__name__)


ORGANS = ["brain", "prostate", "breast"]
MODES  = ["magnitude", "phase", "both"]
MODE_LABELS = {
    "magnitude": "Magnitude only (A)",
    "phase":     "Phase only (B)",
    "both":      "Mag + Phase (C)",
}


def load_checkpoint(checkpoint_path: str, mode: str, device: torch.device):
    """Load model from checkpoint."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = build_model(mode=mode, pretrained=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()
    return model


def run_test_evaluation(
    test_datasets: Dict[str, Dict[str, object]],
    checkpoint_dir: str,
    results_dir: str,
    batch_size: int = 16,
    num_workers: int = 0,
) -> Dict:
    """
    Evaluate all 9 (organ, mode) combinations on their test sets.

    test_datasets: {
        "brain":    {"magnitude": dataset, "phase": dataset, "both": dataset},
        "prostate": {...},
        "breast":   {...},
    }

    Returns nested dict of results.
    """
    device = _get_device()
    Path(results_dir).mkdir(parents=True, exist_ok=True)

    all_results = {}

    for organ in ORGANS:
        all_results[organ] = {}
        for mode in MODES:
            run_name = f"{organ}_{mode}"
            checkpoint_path = str(
                Path(checkpoint_dir) / f"{run_name}_best.pt"
            )

            if not Path(checkpoint_path).exists():
                logger.warning(f"Checkpoint not found: {checkpoint_path}")
                all_results[organ][mode] = {"test_auc": float("nan")}
                continue

            dataset = test_datasets[organ][mode]
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
            )

            model = load_checkpoint(checkpoint_path, mode, device)
            metrics = evaluate(model, loader, device)

            all_results[organ][mode] = {
                "test_auc": metrics["auc"],
                "all_probs": metrics["all_probs"].tolist(),
                "all_labels": metrics["all_labels"].tolist(),
            }
            logger.info(
                f"[{run_name}] Test AUC = {metrics['auc']:.4f}"
            )

    # Save full results to JSON
    results_path = str(Path(results_dir) / "all_results.json")
    # Convert numpy values for JSON serialization
    json_results = {
        organ: {
            mode: {"test_auc": all_results[organ][mode]["test_auc"]}
            for mode in MODES
        }
        for organ in ORGANS
    }
    with open(results_path, "w") as f:
        json.dump(json_results, f, indent=2)
    logger.info(f"Results saved to {results_path}")

    return all_results


def print_results_table(all_results: Dict):
    """Print the 3x3 AUC table to stdout."""
    header = f"{'Organ':<12}" + "".join(
        f"{MODE_LABELS[m]:<28}" for m in MODES
    )
    print("\n" + "=" * (12 + 28 * 3))
    print("TEST AUC RESULTS")
    print("=" * (12 + 28 * 3))
    print(header)
    print("-" * (12 + 28 * 3))

    for organ in ORGANS:
        row = f"{organ.capitalize():<12}"
        for mode in MODES:
            auc = all_results[organ][mode].get("test_auc", float("nan"))
            val = f"{auc:.4f}" if not np.isnan(auc) else "  N/A "
            row += f"{val:<28}"
        print(row)
    print("=" * (12 + 28 * 3) + "\n")


def plot_roc_curves(all_results: Dict, results_dir: str):
    """
    Generate one ROC curve plot per organ, overlaying all 3 conditions.
    Saves to results_dir/roc_{organ}.png
    """
    colors = {
        "magnitude": "#888780",
        "phase":     "#BA7517",
        "both":      "#1D9E75",
    }

    for organ in ORGANS:
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)

        for mode in MODES:
            result = all_results[organ][mode]
            probs = result.get("all_probs")
            labels = result.get("all_labels")

            if probs is None or labels is None:
                continue

            probs = np.array(probs)
            labels = np.array(labels)

            if len(np.unique(labels)) < 2:
                continue

            auc = result["test_auc"]
            fpr, tpr, _ = roc_curve(labels, probs)
            ax.plot(
                fpr, tpr,
                color=colors[mode],
                lw=2,
                label=f"{MODE_LABELS[mode]} (AUC={auc:.3f})",
            )

        ax.set_xlabel("False Positive Rate", fontsize=12)
        ax.set_ylabel("True Positive Rate", fontsize=12)
        ax.set_title(f"ROC Curves — {organ.capitalize()} Tumor Classification",
                     fontsize=13)
        ax.legend(loc="lower right", fontsize=10)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, 1.02])
        ax.grid(alpha=0.3)

        out_path = str(Path(results_dir) / f"roc_{organ}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"ROC curve saved: {out_path}")


def plot_training_history(history: List[Dict], run_name: str, results_dir: str):
    """Plot train/val loss and val AUC curves for one run."""
    epochs = [h["epoch"] for h in history]
    train_loss = [h["train_loss"] for h in history]
    val_loss = [h["val_loss"] for h in history]
    val_auc = [h["val_auc"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(epochs, train_loss, label="Train loss", color="#534AB7")
    ax1.plot(epochs, val_loss, label="Val loss", color="#D85A30")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Loss")
    ax1.set_title(f"{run_name} — Loss")
    ax1.legend()
    ax1.grid(alpha=0.3)

    ax2.plot(epochs, val_auc, color="#1D9E75", lw=2)
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Val AUC")
    ax2.set_title(f"{run_name} — Validation AUC")
    ax2.set_ylim([0, 1])
    ax2.grid(alpha=0.3)

    fig.tight_layout()
    out_path = str(Path(results_dir) / f"history_{run_name}.png")
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
