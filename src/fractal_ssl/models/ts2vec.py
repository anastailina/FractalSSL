"""Wrapper utilities for the TS2Vec backbone."""

from __future__ import annotations

from typing import Any

import torch

from . import register_backbone

__all__ = ["build_ts2vec"]


try:  # pragma: no cover - dependency optional during docs/tests
    from ts2vec import TS2Vec
except ImportError as exc:  # pragma: no cover - handled at runtime
    TS2Vec = None
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


def build_ts2vec(*, device: torch.device, **kwargs: Any) -> TS2Vec:
    if TS2Vec is None:
        raise ImportError(
            "TS2Vec not installed. Run `pip install ts2vec` or add the project to PYTHONPATH."
        ) from _IMPORT_ERROR
    model = TS2Vec(device=device, **kwargs)
    if hasattr(model, "to"):
        model = model.to(device)
    return model


register_backbone("ts2vec", build_ts2vec)
