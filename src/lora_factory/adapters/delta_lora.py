"""Delta-LoRA adapter: trains A/B and merges the accumulated delta into the base weight each step."""
from __future__ import annotations

import torch
import torch.nn as nn

from .base import AdapterModule


class DeltaLoRALinear(AdapterModule):
    """Delta-LoRA: 训练 A,B；每步后将 Δ(AB) 累加并合入基线权重 W。"""

    def __init__(self, wrapped: nn.Linear, rank: int, alpha: float, dropout: float = 0.0) -> None:
        super().__init__(wrapped, rank, alpha, dropout)
        r = rank
        self.lora_A = nn.Linear(self.in_features, r, bias=False)
        self.lora_B = nn.Linear(r, self.out_features, bias=False)
        nn.init.kaiming_uniform_(self.lora_A.weight)
        nn.init.zeros_(self.lora_B.weight)
        self.register_buffer("A_prev", self.lora_A.weight.detach().clone())
        self.register_buffer("B_prev", self.lora_B.weight.detach().clone())

    @torch.no_grad()
    def accumulate_delta_into_base(self) -> None:
        A_cur = self.lora_A.weight.detach()
        B_cur = self.lora_B.weight.detach()
        delta = B_cur @ A_cur - (self.B_prev @ self.A_prev)
        self.wrapped.weight.data.add_(delta)
        self.A_prev.copy_(A_cur)
        self.B_prev.copy_(B_cur)

    def adapter_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.scaling * self.lora_B(self.lora_A(x))
