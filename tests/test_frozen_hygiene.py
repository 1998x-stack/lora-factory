from __future__ import annotations

import torch
import torch.nn as nn

from lora_factory.adapters.base import AdapterModule
from lora_factory.adapters.lora_fa import LoRA_FALinear
from lora_factory.adapters.vera import VeRALinear
from lora_factory.config import AdapterBuildConfig, TargetSpec
from lora_factory.nn.inject import apply_adapters
from lora_factory.utils.optim import build_optimizer_with_lora_plus


class Toy(nn.Module):
    def __init__(self):
        super().__init__()
        torch.manual_seed(0)
        self.block = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x):
        return self.block(x)


def _cfg(kind: str) -> AdapterBuildConfig:
    return AdapterBuildConfig(
        kind=kind,
        targets=[TargetSpec(module_names=["block.0", "block.2"], rank=4, alpha=8)],
    )


def test_wrapped_base_weights_frozen_for_all_kinds():
    for kind in ("lora", "lora_fa", "vera", "delta_lora"):
        m = Toy()
        apply_adapters(m, _cfg(kind))
        for name, sub in m.named_modules():
            if isinstance(sub, AdapterModule):
                for p in sub.wrapped.parameters():
                    assert not p.requires_grad, (kind, name)


def test_lora_fa_freezes_a_only():
    m = Toy()
    apply_adapters(m, _cfg("lora_fa"))
    for idx in (0, 2):
        sub = m.block[idx]
        assert isinstance(sub, LoRA_FALinear)
        assert not sub.lora_A.weight.requires_grad
        assert sub.lora_B.weight.requires_grad


def test_vera_trainable_params_are_only_band_vectors():
    m = Toy()
    apply_adapters(m, _cfg("vera"))
    for idx in (0, 2):
        sub = m.block[idx]
        assert isinstance(sub, VeRALinear)
        trainable = {n for n, p in sub.named_parameters() if p.requires_grad}
        assert trainable <= {"vera_b", "vera_d"}, trainable


def test_apply_adapters_twice_does_not_double_wrap():
    m = Toy()
    first = apply_adapters(m, _cfg("lora"))
    assert sorted(first) == ["block.0", "block.2"]
    # Second call wraps nothing new (already adapted) and must not double-wrap.
    second = apply_adapters(m, _cfg("lora"))
    assert second == []
    n_adapters = sum(1 for sub in m.modules() if isinstance(sub, AdapterModule))
    assert n_adapters == 2


def test_optimizer_lora_plus_split_excludes_frozen_and_ratio():
    m = Toy()
    apply_adapters(m, _cfg("lora"))
    opt = build_optimizer_with_lora_plus(m, base_lr=1e-4, lora_plus_lr_ratio=16.0)
    high = 1e-4 * 16.0
    assert high in {g["lr"] for g in opt.param_groups}
    seen = [p for g in opt.param_groups for p in g["params"]]
    assert all(p.requires_grad for p in seen)
    # frozen wrapped base weights must not appear in any optimizer group (identity check)
    for idx in (0, 2):
        for p in m.block[idx].wrapped.parameters():
            assert all(p is not q for q in seen)