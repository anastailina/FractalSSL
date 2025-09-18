"""Reusable training utilities for self-supervised experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Dataset

from ..utils import logging as logging_utils

__all__ = [
    "TrainingConfig",
    "create_dataloader",
    "build_optimizer",
    "train_backbone",
    "train_ssl_model",
]


@dataclass
class TrainingConfig:
    epochs: int
    lr: float = 1e-3
    weight_decay: float = 0.0
    optimizer: str = "adam"
    log_every: int = 10
    checkpoint_dir: Optional[Path] = Path("runs")
    checkpoint_prefix: str = "model"
    checkpoint_interval: int = 10
    grad_clip_norm: Optional[float] = None


def create_dataloader(dataset: Dataset, *, batch_size: int, shuffle: bool = True, num_workers: int = 0, drop_last: bool = True) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, drop_last=drop_last)


def build_optimizer(model: torch.nn.Module, config: TrainingConfig) -> torch.optim.Optimizer:
    if config.optimizer.lower() == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    if config.optimizer.lower() == "sgd":
        return torch.optim.SGD(model.parameters(), lr=config.lr, momentum=0.9, weight_decay=config.weight_decay)
    return torch.optim.Adam(model.parameters(), lr=config.lr, weight_decay=config.weight_decay)


def _log(message: str) -> None:
    import logging

    logging_utils.configure()
    logging.getLogger(__name__).info(message)


def _save_checkpoint(model: torch.nn.Module, checkpoint_dir: Optional[Path], prefix: str, epoch: int) -> None:
    if checkpoint_dir is None:
        return
    checkpoint_dir = Path(checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    path = checkpoint_dir / f"{prefix}_epoch{epoch:03d}.pt"
    torch.save(model.state_dict(), path)


def train_backbone(model: torch.nn.Module, dataloader: DataLoader, config: TrainingConfig, *, device: torch.device) -> Path:
    model = model.to(device)
    model.train()
    optimizer = build_optimizer(model, config)
    best_path = None
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for step, (batch, _) in enumerate(dataloader, 1):
            batch = batch.to(device)
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
            if step % config.log_every == 0:
                _log(f"[backbone] epoch={epoch} step={step} loss={loss.item():.4f}")
        epoch_loss /= len(dataloader.dataset)
        _log(f"[backbone] epoch {epoch}/{config.epochs} avg_loss={epoch_loss:.4f}")
        if config.checkpoint_dir is not None and epoch % config.checkpoint_interval == 0:
            _save_checkpoint(model, config.checkpoint_dir, config.checkpoint_prefix, epoch)
            best_path = Path(config.checkpoint_dir) / f"{config.checkpoint_prefix}_epoch{epoch:03d}.pt"
    if config.checkpoint_dir is not None and best_path is None:
        best_path = Path(config.checkpoint_dir) / f"{config.checkpoint_prefix}_final.pt"
        torch.save(model.state_dict(), best_path)
    return best_path


def train_ssl_model(model: torch.nn.Module, dataloader: DataLoader, config: TrainingConfig, *, device: torch.device) -> Path:
    model = model.to(device)
    model.train()
    optimizer = build_optimizer(model, config)
    best_path = None
    for epoch in range(1, config.epochs + 1):
        epoch_loss = 0.0
        for step, (batch, _) in enumerate(dataloader, 1):
            batch = batch.to(device)
            loss = model(batch)
            optimizer.zero_grad()
            loss.backward()
            if config.grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip_norm)
            optimizer.step()
            epoch_loss += loss.item() * len(batch)
            if step % config.log_every == 0:
                _log(f"[ssl] epoch={epoch} step={step} loss={loss.item():.4f}")
        epoch_loss /= len(dataloader.dataset)
        _log(f"[ssl] epoch {epoch}/{config.epochs} avg_loss={epoch_loss:.4f}")
        if config.checkpoint_dir is not None and epoch % config.checkpoint_interval == 0:
            _save_checkpoint(model, config.checkpoint_dir, config.checkpoint_prefix, epoch)
            best_path = Path(config.checkpoint_dir) / f"{config.checkpoint_prefix}_epoch{epoch:03d}.pt"
    if config.checkpoint_dir is not None and best_path is None:
        best_path = Path(config.checkpoint_dir) / f"{config.checkpoint_prefix}_final.pt"
        torch.save(model.state_dict(), best_path)
    return best_path
