"""Logging configuration utilities."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

__all__ = ["configure"]


def configure(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Initialise a root logger with console + optional file handler."""
    if logging.getLogger().handlers:
        # Already configured – leave as-is so Jupyter notebooks stay clean.
        return

    fmt = "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
    logging.basicConfig(level=level, format=fmt)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        fh = logging.FileHandler(log_file)
        fh.setLevel(level)
        fh.setFormatter(logging.Formatter(fmt))
        logging.getLogger().addHandler(fh)
