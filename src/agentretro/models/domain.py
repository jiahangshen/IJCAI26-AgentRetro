from __future__ import annotations

import torch
from torch import nn


class _GradientReversalFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = scale
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient: torch.Tensor):
        return -ctx.scale * gradient, None


class GradientReversal(nn.Module):
    def __init__(self, scale: float = 1.0):
        super().__init__()
        self.scale = float(scale)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return _GradientReversalFunction.apply(value, self.scale)


class DomainClassifier(nn.Module):
    def __init__(self, feature_dim: int, hidden_dim: int = 256, num_domains: int = 13):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim, num_domains),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features)


class TextFusion(nn.Module):
    def __init__(self, text_dim: int, feature_dim: int):
        super().__init__()

        self.projection = nn.Linear(text_dim, feature_dim, bias=False)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.norm = nn.LayerNorm(feature_dim)

    def forward(self, features: torch.Tensor, text: torch.Tensor | None) -> torch.Tensor:
        if text is None:
            text = features.new_zeros((features.shape[0], self.projection.in_features))
        projected = self.projection(text)

        fused = self.norm(projected) - self.norm(torch.zeros_like(projected))
        if features.ndim == 3:
            fused = fused.unsqueeze(1)

        return features + torch.tanh(self.gate) * fused


def masked_mean(features: torch.Tensor, padding_mask: torch.Tensor | None = None) -> torch.Tensor:
    if features.ndim == 2:
        return features
    if padding_mask is None:
        return features.mean(dim=1)
    keep = (~padding_mask).to(features.dtype).unsqueeze(-1)
    return (features * keep).sum(1) / keep.sum(1).clamp_min(1)
