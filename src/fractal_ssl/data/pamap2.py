"""PAMAP2 dataset preparation and loading."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

import numpy as np
import pandas as pd

from . import register_dataset
from .base import DatasetSpec, SlidingWindowConfig, WindowedDataset
from .utils import majority_vote, sliding_window

__all__ = ["Pamap2Dataset", "prepare", "DEFAULT_CACHE_FILE"]

SENSOR_COLS = list(range(3, 54))
LABEL_COL = 1
DEFAULT_CACHE_FILE = "pamap2_windows.npz"


class Pamap2Dataset(WindowedDataset):
    """Dataset backed by the cached PAMAP2 windows."""

    pass


@dataclass
class Pamap2PrepareConfig:
    data_root: Path
    window: SlidingWindowConfig
    participants: Optional[Iterable[int]] = None


def _iter_files(data_root: Path, participants: Optional[Iterable[int]]) -> Iterable[Path]:
    if participants is None:
        yield from sorted(data_root.glob("*.dat"))
    else:
        participants = {int(p) for p in participants}
        for dat in sorted(data_root.glob("*.dat")):
            try:
                pid = int(dat.stem.split(".")[-1])
            except ValueError:
                continue
            if pid in participants:
                yield dat


def prepare(*, cache_dir: Path, data_root: Path, window: Optional[SlidingWindowConfig] = None, participants: Optional[Iterable[int]] = None) -> Path:
    data_root = Path(data_root)
    if window is None:
        window = SlidingWindowConfig(window_size=400, step_size=200, standardize=True)

    all_windows: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    for dat in _iter_files(data_root, participants):
        df = pd.read_csv(dat, sep=" ", header=None)
        sensors = df.iloc[:, SENSOR_COLS].replace({"NaN": np.nan}).astype(float)
        labels = df.iloc[:, LABEL_COL].astype(int)
        mask = sensors.dropna().reset_index(drop=True)
        lbl = labels.loc[mask.index].reset_index(drop=True)
        x = mask.to_numpy(dtype=np.float32)
        if window.standardize:
            mu = x.mean(axis=0)
            sigma = x.std(axis=0) + 1e-6
            x = (x - mu) / sigma
        win = sliding_window(x, window.window_size, window.step_size)
        y_windows = sliding_window(lbl.to_numpy(), window.window_size, window.step_size)
        all_windows.append(win)
        all_labels.append(y_windows)

    if not all_windows:
        raise FileNotFoundError(f"No PAMAP2 files found under {data_root}")

    X = np.concatenate(all_windows, axis=0)
    y = np.concatenate(all_labels, axis=0)
    y_mode = majority_vote(y)

    cache_path = Path(cache_dir) / DEFAULT_CACHE_FILE
    np.savez_compressed(cache_path, x=X, y=y_mode)
    return cache_path


register_dataset(
    DatasetSpec(
        name="pamap2",
        dataset_cls=Pamap2Dataset,
        prepare_fn=prepare,
        cache_file=DEFAULT_CACHE_FILE,
    )
)
