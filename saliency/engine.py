"""Training loop, early stopping and a warmup+cosine learning-rate scheduler."""

from __future__ import annotations

import itertools
from typing import Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .data.augment import mixup_data


class EarlyStopping:
    """Stop training when a monitored score stops improving."""

    def __init__(self, patience: int = 10, min_delta: float = 1e-4, mode: str = "min") -> None:
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False

    def __call__(self, score: float) -> bool:
        if self.best_score is None:
            self.best_score = score
        elif (self.mode == "min" and score > self.best_score - self.min_delta) or (
            self.mode == "max" and score < self.best_score + self.min_delta
        ):
            self.counter += 1
            self.early_stop = self.counter >= self.patience
        else:
            self.best_score = score
            self.counter = 0
        return self.early_stop


class WarmupCosineLR:
    """Linear warmup for ``warmup_epochs`` then cosine decay to ``min_lr``."""

    def __init__(self, optimizer, base_lr: float, warmup_epochs: int = 5, total_epochs: int = 100, min_lr: float = 1e-6) -> None:
        self.optimizer = optimizer
        self.base_lr = base_lr
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr

    def step(self, epoch: int) -> float:
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / max(self.warmup_epochs, 1)
        else:
            progress = (epoch - self.warmup_epochs) / max(self.total_epochs - self.warmup_epochs, 1)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))
        for group in self.optimizer.param_groups:
            group["lr"] = lr
        return lr


def _as_total_loss(loss_out):
    """Accept either a tensor or a ``(tensor, components)`` tuple."""
    return loss_out[0] if isinstance(loss_out, tuple) else loss_out


def train(
    model, train_loader: DataLoader, val_loader: DataLoader, optimizer, criterion, device,
    num_epochs: int = 30, num_steps: int = 300, scheduler: Optional["WarmupCosineLR"] = None,
    early_stopping: Optional["EarlyStopping"] = None, accumulation_steps: int = 1,
    use_mixup: bool = False, mixup_alpha: float = 0.2, save_path: Optional[str] = None,
):
    """Step-based training loop with optional grad accumulation / mixup / scheduling.

    ``criterion(pred, target)`` may return a tensor or ``(tensor, components)``.
    The best checkpoint (lowest validation loss) is saved to ``save_path``.
    """
    history = {"train_loss": [], "val_loss": []}
    best_val_loss = float("inf")

    for epoch in range(num_epochs):
        if scheduler is not None:
            lr = scheduler.step(epoch)
            print(f"Epoch {epoch + 1}/{num_epochs} | lr {lr:.2e}")

        model.train()
        optimizer.zero_grad()
        train_losses = []
        loop = tqdm(itertools.islice(train_loader, num_steps), total=num_steps, desc="train")
        for step, (frames, saliency, _) in enumerate(loop):
            frames, saliency = frames.to(device), saliency.to(device)
            if use_mixup and np.random.random() < 0.5:
                frames, saliency = mixup_data(frames, saliency, mixup_alpha)

            loss = _as_total_loss(criterion(model(frames), saliency)) / accumulation_steps
            loss.backward()
            if (step + 1) % accumulation_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            train_losses.append(loss.item() * accumulation_steps)
            loop.set_postfix(loss=np.mean(train_losses[-10:]))

        model.eval()
        val_losses = []
        with torch.no_grad():
            for frames, saliency, _ in tqdm(val_loader, desc="val", leave=False):
                frames, saliency = frames.to(device), saliency.to(device)
                val_losses.append(_as_total_loss(criterion(model(frames), saliency)).item())

        train_loss, val_loss = float(np.mean(train_losses)), float(np.mean(val_losses))
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        print(f"  train loss {train_loss:.4f} | val loss {val_loss:.4f}")

        if save_path and val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({"model_state_dict": model.state_dict()}, save_path)
            print(f"  saved best -> {save_path} (val loss {val_loss:.4f})")

        if early_stopping is not None and early_stopping(val_loss):
            print(f"Early stopping at epoch {epoch + 1}")
            break

    return model, history
