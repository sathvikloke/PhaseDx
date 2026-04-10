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

def make_balanced_sampler(dataset):
    labels = [int(dataset.samples[i][2]) for i in range(len(dataset))]
    class_counts = np.bincount(labels)
    class_weights = 1.0 / (class_counts + 1e-8)
    sample_weights = [float(class_weights[l]) for l in labels]
    return WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)

def train_one_epoch(model, loader, optimizer, criterion, device, scaler=None):
    model.train()
    total_loss, n_batches = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = torch.clamp(model(images), min=-50.0, max=50.0)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
        n_batches += 1
    return total_loss / max(n_batches, 1)

@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_probs, all_labels = [], []
    total_loss, n_batches = 0.0, 0
    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        logits = torch.clamp(model(images), min=-50.0, max=50.0)
        loss = criterion(logits, labels)
        for p in torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy():
            all_probs.append(float(p))
        for lb in labels.detach().cpu().numpy():
            all_labels.append(int(lb))
        total_loss += loss.item()
        n_batches += 1
    probs  = np.clip(np.nan_to_num(np.array(all_probs,  dtype=np.float64), nan=0.5), 1e-7, 1-1e-7)
    labels = np.array(all_labels, dtype=np.int32)
    unique = np.unique(labels)
    if len(unique) < 2:
        auc = float("nan")
        logger.warning(f"Only one class {unique} — AUC undefined.")
    else:
        auc = float(roc_auc_score(labels.tolist(), probs.tolist()))
    return {"auc": auc, "loss": total_loss / max(n_batches, 1), "all_probs": probs, "all_labels": labels}

def train(model, train_dataset, val_dataset, checkpoint_dir, run_name,
          num_epochs=20, batch_size=16, lr=1e-4, weight_decay=1e-4,
          num_workers=0, use_amp=False, patience=5):
    device = _get_device()
    logger.info(f"Training '{run_name}' on {device}")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    checkpoint_path = str(Path(checkpoint_dir) / f"{run_name}_best.pt")
    sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size, shuffle=False,  num_workers=num_workers)
    model = model.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    best_auc, epochs_no_improve, history = 0.0, 0, []
    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss  = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_metrics = evaluate(model, val_loader, device)
        scheduler.step()
        val_auc = val_metrics["auc"]
        logger.info(f"[{run_name}] Epoch {epoch:02d}/{num_epochs} | train_loss={train_loss:.4f} | val_auc={val_auc:.4f} | {time.time()-t0:.1f}s")
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_metrics["loss"], "val_auc": val_auc})
        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc, epochs_no_improve = val_auc, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(), "val_auc": best_auc}, checkpoint_path)
            logger.info(f"  -> Checkpoint saved (AUC={best_auc:.4f})")
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break
    return {"run_name": run_name, "best_val_auc": best_auc, "checkpoint_path": checkpoint_path, "history": history}

def _get_device():
    if torch.cuda.is_available(): return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cpu")
