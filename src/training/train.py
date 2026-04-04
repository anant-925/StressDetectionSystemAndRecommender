"""
src/training/train.py

Training loop for MultichannelCNNWithAttention.

Features:
  - AdamW optimiser
  - Class-weighted BCEWithLogitsLoss to handle label imbalance
  - Early stopping (patience-based)
  - Saves the best checkpoint to models/checkpoints/best_model.pt
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.models.dataset import StressDataset
from src.models.model import MultichannelCNNWithAttention

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

REPO_ROOT = Path(__file__).resolve().parents[2]
CHECKPOINT_DIR = REPO_ROOT / "models" / "checkpoints"
DATA_PATH = REPO_ROOT / "data" / "unified_training_data.csv"


# ── Early Stopping ────────────────────────────────────────────────────────────

class EarlyStopping:
    """Stop training when validation loss stops improving.

    Parameters
    ----------
    patience:
        Epochs to wait after last improvement before stopping.
    min_delta:
        Minimum change to qualify as an improvement.
    """

    def __init__(self, patience: int = 5, min_delta: float = 1e-4) -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.best_loss: float = float("inf")
        self.counter: int = 0
        self.best_state: Optional[dict] = None

    def step(self, val_loss: float, model: nn.Module) -> bool:
        """Returns True if training should stop."""
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            # Deep-copy the best weights
            self.best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            self.counter += 1

        return self.counter >= self.patience

    def restore_best(self, model: nn.Module) -> None:
        """Load the best weights back into *model*."""
        if self.best_state is not None:
            model.load_state_dict(self.best_state)


# ── Training helpers ──────────────────────────────────────────────────────────

def compute_pos_weight(labels: list[float]) -> torch.Tensor:
    """Compute the positive-class weight for BCEWithLogitsLoss."""
    arr = np.array(labels)
    n_pos = float((arr >= 0.5).sum())
    n_neg = float((arr < 0.5).sum())
    if n_pos == 0:
        return torch.tensor(1.0)
    pos_weight = n_neg / n_pos
    logger.info("Class weight – pos_weight: %.4f  (neg=%d, pos=%d)", pos_weight, int(n_neg), int(n_pos))
    return torch.tensor(pos_weight, dtype=torch.float32)


def train_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimiser: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for batch in loader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["label"].to(device).unsqueeze(1)   # (B, 1)

        optimiser.zero_grad()
        outputs = model(input_ids, attention_mask)
        loss = criterion(outputs["logits"], labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimiser.step()

        total_loss += loss.item() * input_ids.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple[float, float]:
    """Returns (val_loss, val_accuracy)."""
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch in loader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["label"].to(device).unsqueeze(1)

            outputs = model(input_ids, attention_mask)
            loss = criterion(outputs["logits"], labels)
            total_loss += loss.item() * input_ids.size(0)

            preds = (outputs["probs"] >= 0.5).float()
            correct += (preds == labels).sum().item()
            total += labels.size(0)

    return total_loss / len(loader.dataset), correct / total


# ── Main training function ────────────────────────────────────────────────────

def train(
    data_path: Path = DATA_PATH,
    tokenizer_name: str = "bert-base-uncased",
    chunk_size: int = 200,
    stride: int = 50,
    embed_dim: int = 128,
    num_filters: int = 128,
    kernel_sizes: tuple[int, ...] = (2, 3, 5),
    dropout: float = 0.5,
    batch_size: int = 32,
    epochs: int = 20,
    lr: float = 2e-4,
    weight_decay: float = 1e-2,
    patience: int = 5,
    val_split: float = 0.15,
    test_split: float = 0.10,
    max_vocab: int = 30522,   # bert-base vocab size as a sensible upper bound
    seed: int = 42,
    checkpoint_dir: Path = CHECKPOINT_DIR,
) -> None:
    """End-to-end training procedure."""

    torch.manual_seed(seed)
    np.random.seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Using device: %s", device)

    # ── Load data ─────────────────────────────────────────────────────────────
    logger.info("Loading data from %s …", data_path)
    df = pd.read_csv(data_path)
    df = df.dropna(subset=["text", "label"])
    texts = df["text"].tolist()
    labels = df["label"].tolist()

    # ── Train / val / test split ──────────────────────────────────────────────
    X_trainval, X_test, y_trainval, y_test = train_test_split(
        texts, labels, test_size=test_split, random_state=seed, stratify=None
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval,
        y_trainval,
        test_size=val_split / (1 - test_split),
        random_state=seed,
    )
    logger.info(
        "Split: train=%d  val=%d  test=%d", len(X_train), len(X_val), len(X_test)
    )

    # ── Tokeniser & Datasets ──────────────────────────────────────────────────
    logger.info("Loading tokenizer: %s …", tokenizer_name)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)

    train_ds = StressDataset(X_train, y_train, tokenizer, chunk_size, stride)
    val_ds = StressDataset(X_val, y_val, tokenizer, chunk_size, stride)
    test_ds = StressDataset(X_test, y_test, tokenizer, chunk_size, stride)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=0)

    # ── Model ─────────────────────────────────────────────────────────────────
    vocab_size = tokenizer.vocab_size
    model = MultichannelCNNWithAttention(
        vocab_size=vocab_size,
        embed_dim=embed_dim,
        num_filters=num_filters,
        kernel_sizes=kernel_sizes,
        dropout=dropout,
    ).to(device)

    logger.info(
        "Model parameters: %d",
        sum(p.numel() for p in model.parameters() if p.requires_grad),
    )

    # ── Optimiser & Loss ──────────────────────────────────────────────────────
    pos_weight = compute_pos_weight(y_train).to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimiser, mode="min", factor=0.5, patience=2
    )

    early_stop = EarlyStopping(patience=patience)

    # ── Training loop ─────────────────────────────────────────────────────────
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, epochs + 1):
        train_loss = train_epoch(model, train_loader, optimiser, criterion, device)
        val_loss, val_acc = eval_epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        logger.info(
            "Epoch %02d/%02d  train_loss=%.4f  val_loss=%.4f  val_acc=%.4f",
            epoch,
            epochs,
            train_loss,
            val_loss,
            val_acc,
        )

        if early_stop.step(val_loss, model):
            logger.info("Early stopping triggered at epoch %d", epoch)
            break

    # ── Restore best and evaluate on test ─────────────────────────────────────
    early_stop.restore_best(model)
    test_loss, test_acc = eval_epoch(model, test_loader, criterion, device)
    logger.info("Test  loss=%.4f  acc=%.4f", test_loss, test_acc)

    # ── Save checkpoint ───────────────────────────────────────────────────────
    ckpt_path = checkpoint_dir / "best_model.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "vocab_size": vocab_size,
            "embed_dim": embed_dim,
            "num_filters": num_filters,
            "kernel_sizes": list(kernel_sizes),
            "dropout": dropout,
            "tokenizer_name": tokenizer_name,
            "chunk_size": chunk_size,
            "stride": stride,
            "test_accuracy": test_acc,
        },
        ckpt_path,
    )
    logger.info("Checkpoint saved to %s", ckpt_path)


if __name__ == "__main__":
    train()
