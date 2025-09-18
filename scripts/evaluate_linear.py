#!/usr/bin/env python
"""Run a linear probe on cached embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from fractal_ssl.data import load_dataset
from fractal_ssl.evaluation import linear_probe
from fractal_ssl.models import build_backbone


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Linear probe evaluation")
    parser.add_argument("--config", type=Path, required=True, help="JSON config with dataset + backbone description")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Checkpoint containing the backbone weights")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--random-state", type=int, default=0)
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def main() -> None:
    args = parse_args()
    config = load_config(args.config)

    dataset_cfg = config["dataset"]
    dataset = load_dataset(dataset_cfg["name"], cache_dir=Path(dataset_cfg["cache_dir"]), dataset_kwargs=dataset_cfg.get("kwargs"))

    model_cfg = config["model"]["backbone"]
    backbone_kwargs = model_cfg.get("kwargs", {}).copy()
    backbone_kwargs.setdefault("input_dims", dataset.x.shape[-1])

    device = torch.device(args.device)
    backbone = build_backbone(model_cfg["name"], device=device, **backbone_kwargs)
    state = torch.load(args.checkpoint, map_location=device)
    backbone.load_state_dict(state)

    result = linear_probe(
        backbone,
        dataset,
        device=device,
        train_ratio=args.train_ratio,
        random_state=args.random_state,
    )
    print(f"Accuracy: {result.accuracy * 100:.2f}%")
    print(result.report)


if __name__ == "__main__":
    main()
