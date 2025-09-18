"""Dataset abstractions used across experiments."""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple, Type

import numpy as np
import torch
from torch.utils.data import Dataset

from . import utils

__all__ = [
    "SlidingWindowConfig",
    "DatasetSpec",
    "WindowedDataset",
]


@dataclass
class SlidingWindowConfig:
    """Configuration for transforming raw sequences into sliding windows."""

    window_size: int
    step_size: int
    standardize: bool = True


@dataclass
class DatasetSpec:
    """Metadata describing how to prepare and load a dataset."""

    name: str
    dataset_cls: Type["WindowedDataset"]
    prepare_fn: Callable[..., Path]
    cache_file: str


class WindowedDataset(Dataset):
    """Simple dataset backed by an ``npz`` cache of shape ``(N, T, C)``."""

    def __init__(self, cache_path: Path):
        cache_path = Path(cache_path)
        if not cache_path.exists():
            raise FileNotFoundError(cache_path)
        data = np.load(cache_path, allow_pickle=True)
        self.x: np.ndarray = data["x"]
        self.y: Optional[np.ndarray] = data.get("y")

    def __len__(self) -> int:
        return len(self.x)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        x = torch.from_numpy(self.x[idx]).float()
        if self.y is None:
            return x, torch.full((), -1, dtype=torch.long)
        return x, torch.tensor(self.y[idx])


def asdict(config: SlidingWindowConfig) -> Dict[str, Any]:
    return dataclasses.asdict(config)
