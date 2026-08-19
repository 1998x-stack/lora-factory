"""VeRA adapter: frozen shared A/B with trainable per-layer vectors."""
from __future__ import annotations

import math
from typing import Dict, Tuple

import torch
import torch.nn as nn

from .base import AdapterModule

_VERA_SHARED: Dict[Tuple[int, int, int, int], Tuple[torch.Tensor, torch.Tensor]] = {}


def _get_shared_AB(out_f: int, in_f: int, r: int, seed: int, device, dtype) -> Tuple[torch.Tensor, torch.Tensor]:
    key = (out_f, in_f, r, seed)
    if key in _VERA_SHARED:
        A, B = _VERA_SHARED[key]
        return A.to(device=device, dtype=dtype), B.to(device=device, dtype=dtype)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    A = torch.empty((r, in_f), device=device, dtype=dtype).normal_(0, 1.0, generator=gen) / math.sqrt(in_f)
    B = torch.empty((out_f, r), device=device, dtype=dtype).normal_(0, 1.0, generator=gen) / math.sqrt(r)
    A.requires_grad_(False); B.requires_grad_(False)
    _VERA_SHARED[key] = (A.cpu(), B.cpu())
    return A, B


class VeRALinear(AdapterModule):
    """VeRA: ΔW = diag(b)·B·diag(d)·A，A/B 冻结共享，只学习向量 b,d。"""

    def __init__(self, wrapped: nn.Linear, rank: int, alpha: float, dropout: float = 0.0, seed: int = 42) -> None:
        super().__init__(wrapped, rank, alpha, dropout)
        self.seed = seed
        self.vera_b = nn.Parameter(torch.ones(self.out_features))
        self.vera_d = nn.Parameter(torch.ones(self.rank))
        self.register_buffer("_affinity", torch.empty(0))

    def adapter_forward(self, x: torch.Tensor) -> torch.Tensor:
        device, dtype = x.device, x.dtype
        A, B = _get_shared_AB(self.out_features, self.in_features, self.rank, self.seed, device, dtype)
        u = x.matmul(A.t())         # (B, r)
        u = u * self.vera_d         # (B, r)
        v = u.matmul(B.t())         # (B, out)
        v = v * self.vera_b         # (B, out)
        return self.scaling * v