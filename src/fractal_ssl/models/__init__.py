"""Model registry and builders."""

from __future__ import annotations

from typing import Any, Dict, Iterable, Optional

from ..utils.registry import Registry

__all__ = [
    "BACKBONES",
    "WRAPPERS",
    "register_backbone",
    "register_wrapper",
    "build_backbone",
    "build_wrapper",
    "available_backbones",
    "available_wrappers",
]

BACKBONES: Registry = Registry("backbone")
WRAPPERS: Registry = Registry("ssl_wrapper")


def register_backbone(name: str, builder) -> None:
    BACKBONES.register(name, builder)


def register_wrapper(name: str, builder) -> None:
    WRAPPERS.register(name, builder)


def build_backbone(name: str, *, device, **kwargs):
    builder = BACKBONES.get(name)
    return builder(device=device, **kwargs)


def build_wrapper(name: str, *, backbone, **kwargs):
    builder = WRAPPERS.get(name)
    return builder(backbone=backbone, **kwargs)


def available_backbones() -> Iterable[str]:
    return BACKBONES.keys()


def available_wrappers() -> Iterable[str]:
    return WRAPPERS.keys()


# Register built-in models
from . import ts2vec  # noqa: F401
from . import simclr  # noqa: F401
from . import fractal  # noqa: F401
