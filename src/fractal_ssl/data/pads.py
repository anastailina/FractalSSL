"""PADS (Parkinson's disease) dataset helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from . import register_dataset
from .base import DatasetSpec, SlidingWindowConfig, WindowedDataset

__all__ = ["PadsDataset", "prepare", "DEFAULT_CACHE_FILE"]

DEFAULT_CACHE_FILE = "pads_windows.npz"


class PadsDataset(WindowedDataset):
    """Thin wrapper around the cached PADS windows."""

    pass


def prepare(*, cache_dir: Path, **_: object) -> Path:
    """Placeholder implementation.

    The official PADS dataset ships with preprocessed windows already saved as
    ``.npz`` files. Because the exact folder structure can vary depending on how
    you mirrored your Google Drive, this repository does not hard-code the
    expected layout. Instead, export your preferred preprocessing pipeline to an
    ``npz`` file named :data:`pads_windows.npz` and place it inside
    ``cache_dir``.

    See ``notebooks/legacy/PADs_FractalSSL.ipynb`` for the original exploratory
    notebook, or adapt ``Pamap2``'s ``prepare`` function.
    """

    cache_path = Path(cache_dir) / DEFAULT_CACHE_FILE
    if not cache_path.exists():
        raise FileNotFoundError(
            "Expected preprocessed cache at "
            f"{cache_path}. Please generate it using your custom pipeline."
        )
    return cache_path


register_dataset(
    DatasetSpec(
        name="pads",
        dataset_cls=PadsDataset,
        prepare_fn=prepare,
        cache_file=DEFAULT_CACHE_FILE,
    )
)
