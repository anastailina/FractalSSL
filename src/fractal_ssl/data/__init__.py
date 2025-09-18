"""Dataset registry and high level helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, Type

from ..utils.registry import Registry
from .base import DatasetSpec, SlidingWindowConfig, WindowedDataset

__all__ = [
    "DATASETS",
    "register_dataset",
    "get_dataset_spec",
    "available_datasets",
    "prepare_dataset",
    "load_dataset",
]

DATASETS: Registry[DatasetSpec] = Registry("dataset")


def register_dataset(spec: DatasetSpec) -> None:
    DATASETS.register(spec.name, spec)


def get_dataset_spec(name: str) -> DatasetSpec:
    return DATASETS.get(name)


def available_datasets() -> Iterable[str]:
    return DATASETS.keys()


def prepare_dataset(name: str, *, cache_dir: Path, **kwargs) -> Path:
    spec = get_dataset_spec(name)
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    return spec.prepare_fn(cache_dir=cache_dir, **kwargs)


def load_dataset(name: str, *, cache_dir: Path, dataset_kwargs: Optional[Dict] = None) -> WindowedDataset:
    spec = get_dataset_spec(name)
    dataset_kwargs = dataset_kwargs or {}
    cache_path = Path(cache_dir) / spec.cache_file
    return spec.dataset_cls(cache_path, **dataset_kwargs)


# Import datasets to populate registry
from . import pamap2  # noqa: F401
from . import pads  # noqa: F401
