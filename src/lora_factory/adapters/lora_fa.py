"""LoRA-FA adapter: A is frozen after initialization, only B is trained."""
from __future__ import annotations

import math
import torch
import torch.nn as nn

from .base import AdapterModule


class LoRA_FALinear(AdapterModule):
    """LoRA-FA (Frozen-A): 仅训练 B，A 冻结。"""

    def __init__(self, wrapped: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__(wrapped, rank, alpha, dropout)
        r = rank
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        for p in self.lora_A.parameters():
            p.requires_grad_(False)

    def adapter_forward(self, x: torch.Tensor) -> torch.Tensor:
        with torch.no_grad():
            z = self.lora_A(x)
        return self.scaling * self.lora_B(z)
