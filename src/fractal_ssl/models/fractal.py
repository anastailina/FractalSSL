"""Fractal SSL view generator and wrapper modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch
import torch.nn as nn

from . import register_wrapper

__all__ = [
    "FractalViewConfig",
    "FractalViewGenerator",
    "FractalSSL",
    "build_fractal_ssl",
]


@dataclass
class FractalViewConfig:
    levels: int = 4
    noise_std: float = 0.01


class FractalViewGenerator:
    """Generates self-similar subsequences of exponentially decreasing length."""

    def __init__(self, config: FractalViewConfig):
        if config.levels < 2:
            raise ValueError('FractalViewGenerator requires at least two levels for contrastive pairs.')
        self.config = config

    def __call__(self, x: torch.Tensor) -> List[torch.Tensor]:
        B, T, _ = x.shape
        views: List[torch.Tensor] = []
        for level in range(self.config.levels):
            seg_len = max(1, T // (2 ** level))
            start = torch.randint(0, T - seg_len + 1, (B,), device=x.device)
            for b in range(B):
                seg = x[b, start[b] : start[b] + seg_len]
                seg = seg + self.config.noise_std * torch.randn_like(seg)
                views.append(seg)
        return views


class FractalSSL(nn.Module):
    """Fractal self-supervised learner that wraps a backbone encoder."""

    def __init__(self, backbone: nn.Module, proj_dim: int = 128, temperature: float = 0.2, view_config: FractalViewConfig | None = None):
        super().__init__()
        self.backbone = backbone
        self.view_gen = FractalViewGenerator(view_config or FractalViewConfig())
        if not hasattr(backbone, "repr_dims"):
            # Backbones that do not expose repr_dims fall back to projector input size.
            raise AttributeError("Backbone must define `repr_dims` attribute for FractalSSL to build a projector.")
        repr_dims = getattr(backbone, "repr_dims")
        self.projector = nn.Sequential(
            nn.Linear(repr_dims, proj_dim),
            nn.ReLU(inplace=True),
            nn.Linear(proj_dim, proj_dim),
        )
        self.temperature = temperature

    def info_nce(self, z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
        z1 = nn.functional.normalize(z1, dim=-1)
        z2 = nn.functional.normalize(z2, dim=-1)
        sim = (z1 @ z2.T) / self.temperature
        labels = torch.arange(z1.size(0), device=z1.device)
        return nn.functional.cross_entropy(sim, labels)

    def encode_views(self, views: List[torch.Tensor]) -> List[torch.Tensor]:
        encodings: List[torch.Tensor] = []
        batch_size = len(views)
        for view in views:
            if view.dim() == 2:
                view = view.unsqueeze(0)
            padded = nn.utils.rnn.pad_sequence(view, batch_first=True) if isinstance(view, list) else view
            features = self.backbone.encode(padded) if hasattr(self.backbone, "encode") else self.backbone(padded)
            if features.dim() > 2:
                features = features.max(dim=1).values
            encodings.append(self.projector(features))
        return encodings

    def forward(self, batch: torch.Tensor) -> torch.Tensor:
        views = self.view_gen(batch)
        B = batch.size(0)
        losses = []
        ref_features = None
        for level in range(0, len(views), B):
            chunk = views[level : level + B]
            padded = nn.utils.rnn.pad_sequence(chunk, batch_first=True)
            feats = self.backbone.encode(padded) if hasattr(self.backbone, "encode") else self.backbone(padded)
            feats = feats.max(dim=1).values
            proj = self.projector(feats)
            if level == 0:
                ref_features = proj
            else:
                losses.append(self.info_nce(proj, ref_features))
        if not losses:
            return torch.tensor(0.0, device=batch.device, requires_grad=True)
        return torch.stack(losses).mean()


def build_fractal_ssl(*, backbone, proj_dim: int = 128, temperature: float = 0.2, levels: int = 4, noise_std: float = 0.01, **_: object) -> FractalSSL:
    config = FractalViewConfig(levels=levels, noise_std=noise_std)
    return FractalSSL(backbone=backbone, proj_dim=proj_dim, temperature=temperature, view_config=config)


register_wrapper("fractal_ssl", build_fractal_ssl)
