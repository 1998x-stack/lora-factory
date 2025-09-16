from __future__ import annotations

import math
import torch
import torch.nn as nn

from .base import AdapterModule


class LoRALinear(AdapterModule):
    """Standard LoRA: ΔW = B @ A; forward adds scaling * (xA)B."""

    def __init__(self, wrapped: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__(wrapped, rank, alpha, dropout)
        r = rank
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B.weight)

    def adapter_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * self.lora_B(self.lora_A(x))
