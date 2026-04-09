"""
train.py
--------
Training loop for one condition (magnitude / phase / both) on one organ.
Saves best checkpoint based on validation AUC.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path
import logging
from typing import Optional
import time

logger = logging.getLogger(__name__)


def make_balanced_sampler(dataset) -> WeightedRandomSampler:
    """
    Creates a WeightedRandomSampler so every batch is balanced
    between positive and negative examples, even with class imbalance.
    """
    labels = [dataset.samples[i][2] for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [class_weights[l] for l in labels]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True,
    )
    return sampler


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
) -> float:
    """Run one training epoch. Returns mean loss."""
    model.train()
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast(device_type=device.type):
                logits = model(images)
                loss = criterion(logits, labels)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        n_batches += 1

    return total_loss / max(n_batches, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict:
    """
    Run evaluation. Returns dict with:
        auc: float
        loss: float
        all_probs: np.ndarray
        all_labels: np.ndarray
    """
    model.eval()
    criterion = nn.CrossEntropyLoss()

    all_probs = []
    all_labels = []
    total_loss = 0.0
    n_batches = 0

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits, dim=1)[:, 1].cpu().numpy()
        all_probs.extend(probs.tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        total_loss += loss.item()
        n_batches += 1

    all_probs = np.array(all_probs)
    all_labels = np.array(all_labels)

    # Guard against single-class batches during early epochs
    if len(np.unique(all_labels)) < 2:
        auc = float("nan")
        logger.warning("Only one class present in eval set — AUC undefined.")
    else:
        auc = roc_auc_score(all_labels, all_probs)

    return {
        "auc": auc,
        "loss": total_loss / max(n_batches, 1),
        "all_probs": all_probs,
        "all_labels": all_labels,
    }


def train(
    model: nn.Module,
    train_dataset,
    val_dataset,
    checkpoint_dir: str,
    run_name: str,
    num_epochs: int = 20,
    batch_size: int = 16,
    lr: float = 1e-4,
    weight_decay: float = 1e-4,
    num_workers: int = 0,
    use_amp: bool = False,
    patience: int = 5,
) -> dict:
    """
    Full training run for one condition on one organ.

    Returns dict with best val AUC and path to best checkpoint.

    Parameters
    ----------
    num_workers : int
        Set to 0 on MacBook to avoid multiprocessing issues with h5py.
    use_amp : bool
        Mixed precision. On M4 Mac, set False (MPS doesn't fully support amp).
    patience : int
        Early stopping patience in epochs.
    """
    device = _get_device()
    logger.info(f"Training '{run_name}' on {device}")

    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(Path(checkpoint_dir) / f"{run_name}_best.pt")

    # Dataloaders
    sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=lr, weight_decay=weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs
    )
    scaler = torch.amp.GradScaler() if (use_amp and device.type == "cuda") else None

    best_auc = 0.0
    epochs_no_improve = 0
    history = []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss = train_one_epoch(
            model, train_loader, optimizer, criterion, device, scaler
        )
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        elapsed = time.time() - t0

        val_auc = val_metrics["auc"]
        logger.info(
            f"[{run_name}] Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | "
            f"val_loss={val_metrics['loss']:.4f} | "
            f"val_auc={val_auc:.4f} | {elapsed:.1f}s"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_metrics["loss"],
            "val_auc": val_auc,
        })

        # Save best checkpoint
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc = val_auc
            epochs_no_improve = 0
            torch.save({
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "val_auc": best_auc,
                "run_name": run_name,
            }, checkpoint_path)
            logger.info(f"  -> Saved best checkpoint (AUC={best_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    return {
        "run_name": run_name,
        "best_val_auc": best_auc,
        "checkpoint_path": checkpoint_path,
        "history": history,
    }


def _get_device() -> torch.device:
    """Auto-select best available device: CUDA > MPS (Apple M4) > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    # MPS = Metal Performance Shaders (Apple Silicon)
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
