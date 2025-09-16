from __future__ import annotations

import torch
import torch.nn as nn

from lora_factory.config import AdapterBuildConfig, TargetSpec
from lora_factory.nn.inject import apply_adapters
from lora_factory.utils.optim import split_lora_params_for_lora_plus
from lora_factory.adapters.base import AdapterModule


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x): return self.block(x)


def test_lora_injection_and_freeze():
    m = Toy()
    cfg = AdapterBuildConfig(
        kind="lora",
        targets=[TargetSpec(module_names=["block.0", "block.2"], rank=4, alpha=8, dropout=0.0)],
    )
    patched = apply_adapters(m, cfg)
    assert "block.0" in patched and "block.2" in patched
    # wrapped layers are AdapterModule
    assert isinstance(m.block[0], AdapterModule) and isinstance(m.block[2], AdapterModule)
    # base weights are frozen
    for p in m.block[0].wrapped.parameters():
        assert not p.requires_grad
    # split params groups
    base, a, b = split_lora_params_for_lora_plus(m)
    # There should be some A and B params present
    assert len(a) > 0 and len(b) > 0

    # basic forward
    x = torch.randn(4, 16)
    y = m(x)
    assert y.shape == (4, 8)
