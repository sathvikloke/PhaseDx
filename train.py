import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from sklearn.metrics import roc_auc_score
import numpy as np
from pathlib import Path
import logging
import time
from tqdm import tqdm

logger = logging.getLogger(__name__)


def make_balanced_sampler(dataset):
    """Oversample minority class so each batch is ~50/50."""
    labels = [int(dataset.samples[i][-1]) for i in range(len(dataset))]
    counts = np.bincount(labels)
    weights = 1.0 / (counts + 1e-8)
    sample_weights = [float(weights[l]) for l in labels]
    return WeightedRandomSampler(sample_weights, len(sample_weights), replacement=True)


def train_one_epoch(model, loader, optimizer, criterion, device, epoch, num_epochs, run_name):
    model.train()
    total_loss, n = 0.0, 0
    pbar = tqdm(loader, desc=f"[{run_name}] Epoch {epoch}/{num_epochs}", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = torch.clamp(model(images), -50.0, 50.0)
        loss = criterion(logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        total_loss += loss.item()
        n += 1
        pbar.set_postfix(loss=f"{loss.item():.4f}")
    return total_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, device, run_name):
    model.eval()
    criterion = nn.CrossEntropyLoss()
    all_probs, all_labels = [], []
    total_loss, n = 0.0, 0

    pbar = tqdm(loader, desc=f"[{run_name}] eval", leave=False)
    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.long().to(device, non_blocking=True)
        logits = torch.clamp(model(images), -50.0, 50.0)
        loss = criterion(logits, labels)

        probs = torch.softmax(logits.float(), dim=1)[:, 1].detach().cpu().numpy()
        labs  = labels.detach().cpu().numpy()

        for p in probs:
            all_probs.append(float(p))
        for lb in labs:
            v = int(lb)
            all_labels.append(v if v in (0, 1) else 0)

        total_loss += loss.item()
        n += 1

    probs_arr  = np.clip(np.nan_to_num(
        np.array(all_probs,  dtype=np.float64), nan=0.5), 1e-7, 1-1e-7)
    labels_arr = np.array(all_labels, dtype=np.int32)

    if len(np.unique(labels_arr)) < 2:
        auc = float("nan")
        logger.warning("Only one class in eval set — AUC undefined.")
    else:
        auc = float(roc_auc_score(labels_arr.tolist(), probs_arr.tolist()))

    return {"auc": auc, "loss": total_loss / max(n, 1),
            "all_probs": probs_arr, "all_labels": labels_arr}


def train(model, train_dataset, val_dataset, checkpoint_dir, run_name,
          num_epochs=30, batch_size=8, lr=1e-4, weight_decay=1e-3,
          num_workers=0, use_amp=False, patience=7):

    device = _get_device()
    logger.info(f"Training '{run_name}' on {device}")
    Path(checkpoint_dir).mkdir(parents=True, exist_ok=True)
    ckpt_path = str(Path(checkpoint_dir) / f"{run_name}_best.pt")

    sampler = make_balanced_sampler(train_dataset)
    train_loader = DataLoader(train_dataset, batch_size=batch_size,
                              sampler=sampler, num_workers=num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=batch_size,
                              shuffle=False,  num_workers=num_workers)

    model = model.to(device)
    criterion = nn.CrossEntropyLoss()

    # Only optimize unfrozen parameters
    trainable = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)

    # Warmup for 3 epochs then cosine decay
    def lr_lambda(epoch):
        warmup = 3
        if epoch < warmup:
            return (epoch + 1) / warmup
        progress = (epoch - warmup) / max(num_epochs - warmup, 1)
        return 0.5 * (1 + np.cos(np.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)

    best_auc, no_improve, history = 0.0, 0, []

    for epoch in range(1, num_epochs + 1):
        t0 = time.time()
        train_loss  = train_one_epoch(model, train_loader, optimizer,
                                       criterion, device, epoch, num_epochs, run_name)
        val_metrics = evaluate(model, val_loader, device, run_name)
        scheduler.step()

        val_auc = val_metrics["auc"]
        elapsed = time.time() - t0
        logger.info(
            f"[{run_name}] Epoch {epoch:02d}/{num_epochs} | "
            f"train_loss={train_loss:.4f} | val_loss={val_metrics['loss']:.4f} | "
            f"val_auc={val_auc:.4f} | {elapsed:.1f}s"
        )
        history.append({"epoch": epoch, "train_loss": train_loss,
                         "val_loss": val_metrics["loss"], "val_auc": val_auc})

        if not np.isnan(val_auc) and val_auc > best_auc:
            best_auc, no_improve = val_auc, 0
            torch.save({"epoch": epoch, "model_state_dict": model.state_dict(),
                        "val_auc": best_auc, "run_name": run_name}, ckpt_path)
            logger.info(f"  -> Checkpoint saved (AUC={best_auc:.4f})")
        else:
            no_improve += 1
            if no_improve >= patience:
                logger.info(f"Early stopping at epoch {epoch}.")
                break

    return {"run_name": run_name, "best_val_auc": best_auc,
            "checkpoint_path": ckpt_path, "history": history}


def _get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")
