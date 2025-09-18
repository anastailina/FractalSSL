#!/usr/bin/env python
"""Command line utility to preprocess raw datasets into cached windows."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Any

from fractal_ssl.data import available_datasets, prepare_dataset
from fractal_ssl.data.base import SlidingWindowConfig


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw time-series datasets")
    parser.add_argument("dataset", choices=list(available_datasets()))
    parser.add_argument("--data-root", type=Path, help="Path to the raw dataset (if required)")
    parser.add_argument("--cache-dir", type=Path, default=None, help="Folder where the cache will be stored")
    parser.add_argument("--window-size", type=int, default=400, help="Sliding window size")
    parser.add_argument("--step-size", type=int, default=200, help="Sliding window stride")
    parser.add_argument("--no-standardize", action="store_true", help="Disable per-sensor standardisation")
    parser.add_argument("--participants", type=int, nargs="*", help="Optional PAMAP2 participant IDs to include")
    parser.add_argument("--config", type=Path, help="Optional JSON with additional keyword arguments")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cache_dir = args.cache_dir or Path("cache") / args.dataset
    extra: Dict[str, Any] = {}
    if args.config is not None:
        with open(args.config) as fh:
            extra.update(json.load(fh))

    if args.dataset == "pamap2":
        if args.data_root is None:
            raise SystemExit("--data-root is required for PAMAP2")
        window = SlidingWindowConfig(
            window_size=args.window_size,
            step_size=args.step_size,
            standardize=not args.no_standardize,
        )
        extra.setdefault("data_root", args.data_root)
        extra.setdefault("window", window)
        if args.participants:
            extra.setdefault("participants", args.participants)
    elif args.data_root is not None:
        extra.setdefault("data_root", args.data_root)

    path = prepare_dataset(args.dataset, cache_dir=cache_dir, **extra)
    print(f"Cached {args.dataset} windows at {path}")


if __name__ == "__main__":
    main()
