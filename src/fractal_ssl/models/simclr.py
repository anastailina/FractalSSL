"""Placeholder SimCLR-style backbone for time-series experiments."""

from __future__ import annotations

import torch
import torch.nn as nn

from . import register_backbone

__all__ = ["SimCLRBackbone", "build_simclr"]


class SimCLRBackbone(nn.Module):
    """Very small temporal encoder intended as a baseline."""

    def __init__(self, input_dims: int, hidden_dims: int = 128, output_dims: int = 128, num_layers: int = 3, **_: object):
        super().__init__()
        layers = []
        in_dim = input_dims
        for _ in range(num_layers):
            layers.append(nn.Conv1d(in_dim, hidden_dims, kernel_size=5, padding=2))
            layers.append(nn.BatchNorm1d(hidden_dims))
            layers.append(nn.ReLU(inplace=True))
            in_dim = hidden_dims
        self.encoder = nn.Sequential(*layers)
        self.projector = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(hidden_dims, output_dims),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.transpose(1, 2)
        h = self.encoder(x)
        return self.projector(h)


def build_simclr(*, device: torch.device, **kwargs):
    model = SimCLRBackbone(**kwargs)
    return model.to(device)


register_backbone("simclr", build_simclr)
