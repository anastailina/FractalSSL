"""Simple downstream evaluation via logistic regression."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

from ..data.base import WindowedDataset

__all__ = ["linear_probe", "LinearProbeResult"]


@dataclass
class LinearProbeResult:
    accuracy: float
    report: str


def linear_probe(backbone: torch.nn.Module, dataset: WindowedDataset, *, device: torch.device, train_ratio: float = 0.8, random_state: int = 0, max_iter: int = 1000) -> LinearProbeResult:
    if dataset.y is None:
        raise ValueError("Dataset does not contain labels. Cannot run linear probe.")

    X = torch.from_numpy(dataset.x).float().to(device)
    y = dataset.y

    backbone = backbone.to(device)
    backbone.eval()
    with torch.no_grad():
        feats = backbone.encode(X).max(dim=1).values.cpu().numpy()

    n = len(feats)
    rng = np.random.default_rng(random_state)
    indices = rng.permutation(n)
    split = int(train_ratio * n)
    train_idx, test_idx = indices[:split], indices[split:]

    clf = LogisticRegression(max_iter=max_iter)
    clf.fit(feats[train_idx], y[train_idx])
    y_pred = clf.predict(feats[test_idx])
    acc = accuracy_score(y[test_idx], y_pred)
    report = classification_report(y[test_idx], y_pred)
    return LinearProbeResult(accuracy=acc, report=report)
