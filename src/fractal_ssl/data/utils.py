"""Utility helpers for dataset preprocessing."""

from __future__ import annotations

import numpy as np

__all__ = ["sliding_window", "majority_vote"]


def sliding_window(a: np.ndarray, win_size: int, step: int = 1) -> np.ndarray:
    """Return a view over ``a`` with sliding windows."""
    if a.ndim != 2:
        raise ValueError("expected array of shape (n_samples, n_features)")
    n_samples, n_feat = a.shape
    if win_size > n_samples:
        raise ValueError("win_size larger than input length")
    if step <= 0:
        raise ValueError("step must be positive")
    n_windows = 1 + (n_samples - win_size) // step
    stride0, stride1 = a.strides
    return np.lib.stride_tricks.as_strided(
        a,
        shape=(n_windows, win_size, n_feat),
        strides=(step * stride0, stride0, stride1),
        writeable=False,
    )


def majority_vote(labels: np.ndarray) -> np.ndarray:
    """Compute the majority label for each window."""
    if labels.ndim != 2:
        raise ValueError("expected array of shape (n_windows, window_size)")
    modes = np.empty(labels.shape[0], dtype=labels.dtype)
    for i, row in enumerate(labels):
        values, counts = np.unique(row, return_counts=True)
        modes[i] = values[counts.argmax()]
    return modes
