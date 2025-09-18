#!/usr/bin/env python
"""Train self-supervised models using the modular FractalSSL toolkit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

import torch

from fractal_ssl.data import load_dataset
from fractal_ssl.models import build_backbone, build_wrapper
from fractal_ssl.training import TrainingConfig, create_dataloader, train_backbone, train_ssl_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="FractalSSL training entry point")
    parser.add_argument("--config", type=Path, required=True, help="JSON config describing dataset/model/training setup")
    parser.add_argument("--stage", choices=["backbone", "fractal_ssl"], help="Training stage to run")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu", help="Device identifier")
    parser.add_argument("--backbone-checkpoint", type=Path, help="Optional checkpoint to warm-start the wrapper stage")
    return parser.parse_args()


def load_config(path: Path) -> Dict[str, Any]:
    with open(path) as fh:
        return json.load(fh)


def build_training_config(cfg: Dict[str, Any], default_prefix: str) -> TrainingConfig:
    checkpoint_dir = cfg.get("checkpoint_dir")
    if checkpoint_dir is not None:
        checkpoint_dir = Path(checkpoint_dir)
    return TrainingConfig(
        epochs=cfg["epochs"],
        lr=cfg.get("lr", 1e-3),
        weight_decay=cfg.get("weight_decay", 0.0),
        optimizer=cfg.get("optimizer", "adam"),
        log_every=cfg.get("log_every", 10),
        checkpoint_dir=checkpoint_dir,
        checkpoint_prefix=cfg.get("checkpoint_prefix", default_prefix),
        checkpoint_interval=cfg.get("checkpoint_interval", 10),
        grad_clip_norm=cfg.get("grad_clip_norm"),
    )


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    stage = args.stage or config.get("stage", "backbone")

    dataset_cfg = config["dataset"]
    dataset = load_dataset(dataset_cfg["name"], cache_dir=Path(dataset_cfg["cache_dir"]), dataset_kwargs=dataset_cfg.get("kwargs"))

    dataloader_cfg = config.get("dataloader", {})
    batch_size = dataloader_cfg.get("batch_size", 256)
    num_workers = dataloader_cfg.get("num_workers", 0)
    drop_last = dataloader_cfg.get("drop_last", True)
    dataloader = create_dataloader(dataset, batch_size=batch_size, num_workers=num_workers, drop_last=drop_last)

    model_cfg = config["model"]
    backbone_cfg = model_cfg["backbone"]
    backbone_kwargs = backbone_cfg.get("kwargs", {}).copy()
    backbone_kwargs.setdefault("input_dims", dataset.x.shape[-1])
    device = torch.device(args.device)

    backbone = build_backbone(backbone_cfg["name"], device=device, **backbone_kwargs)

    training_cfgs = config.get("training", {})
    if isinstance(training_cfgs, dict) and stage in training_cfgs:
        training_cfg = build_training_config(training_cfgs[stage], default_prefix=stage)
    else:
        training_cfg = build_training_config(training_cfgs, default_prefix=stage)

    if stage == "backbone":
        train_backbone(backbone, dataloader, training_cfg, device=device)
    elif stage == "fractal_ssl":
        if args.backbone_checkpoint is not None:
            state = torch.load(args.backbone_checkpoint, map_location=device)
            backbone.load_state_dict(state)
        wrapper_cfg = model_cfg.get("wrapper")
        if wrapper_cfg is None:
            raise SystemExit("Config missing 'model.wrapper' section for fractal_ssl stage")
        wrapper_kwargs = wrapper_cfg.get("kwargs", {}).copy()
        model = build_wrapper(wrapper_cfg["name"], backbone=backbone, **wrapper_kwargs)
        train_ssl_model(model, dataloader, training_cfg, device=device)
    else:  # pragma: no cover - should never happen due to argparse choices
        raise SystemExit(f"Unknown stage {stage}")


if __name__ == "__main__":
    main()
