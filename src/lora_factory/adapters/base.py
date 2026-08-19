"""Base adapter class and shared metadata used by all variants."""
from __future__ import annotations

from dataclasses import dataclass
import torch
import torch.nn as nn


@dataclass
class AdapterMeta:
    kind: str
    rank: int
    alpha: float
    dropout: float


class AdapterModule(nn.Module):
    """Base adapter wrapping a Linear layer.

    子类实现 adapter_forward(x) 返回 Δy（将被加到基线输出上）。
    """

    def __init__(self, wrapped: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__()
        if not isinstance(wrapped, nn.Linear):
            raise TypeError("Only nn.Linear is supported by this minimal factory.")
        self.wrapped = wrapped
        self.in_features = wrapped.in_features
        self.out_features = wrapped.out_features
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / max(1, rank)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        # Freeze base weights/bias
        for p in self.wrapped.parameters():
            p.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = self.wrapped(x)
        return y + self.adapter_forward(self.drop(x))

    def adapter_forward(self, x: torch.Tensor) -> torch.Tensor:
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f"in={self.in_features}, out={self.out_features}, rank={self.rank}, alpha={self.alpha}"
