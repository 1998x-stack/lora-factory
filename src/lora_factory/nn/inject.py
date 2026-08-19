from __future__ import annotations

from typing import List, Tuple

import torch.nn as nn

from ..config import AdapterBuildConfig, TargetSpec
from ..utils.seed import set_seed
from ..adapters.registry import build_adapter
from ..adapters.delta_lora import DeltaLoRALinear
from ..adapters.base import AdapterModule
from ..adapters.vera import VeRALinear


def _match_module(name: str, module: nn.Module, spec: TargetSpec) -> bool:
    # avoid double-wrapping
    if isinstance(module, AdapterModule):
        return False
    if not any(t.lower() in type(module).__name__.lower() for t in spec.module_types):
        return False
    return any(sub in name for sub in spec.module_names)


def _wrap_linear(module: nn.Linear, kind: str, spec: TargetSpec, seed: int) -> nn.Module:
    ctor = build_adapter(kind)
    if kind == "vera":
        return ctor(module, rank=spec.rank, alpha=spec.alpha, dropout=spec.dropout, seed=seed)
    return ctor(module, rank=spec.rank, alpha=spec.alpha, dropout=spec.dropout)


def preview_targets(model: nn.Module, target: TargetSpec) -> List[str]:
    """List matching module names (for inspection before patching)."""
    hits = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear) and _match_module(name, module, target):
            hits.append(name)
    return hits


def apply_adapters(model: nn.Module, config: AdapterBuildConfig) -> List[str]:
    """Patch target Linear modules in-place. Returns list of patched module names."""
    set_seed(config.seed)
    patched = []
    for name, module in list(model.named_modules()):
        for spec in config.targets:
            if isinstance(module, nn.Linear) and _match_module(name, module, spec):
                parent, attr = _find_parent_with_attr(model, name)
                setattr(parent, attr, _wrap_linear(module, config.kind, spec, config.seed))
                patched.append(name)
                break
    return patched


def apply_post_step_updates(model: nn.Module) -> None:
    """Call after each optimizer.step() for variants that need it (Delta-LoRA)."""
    for m in model.modules():
        if isinstance(m, DeltaLoRALinear):
            m.accumulate_delta_into_base()


def _find_parent_with_attr(root: nn.Module, dotted_name: str) -> Tuple[nn.Module, str]:
    parts = dotted_name.split(".")
    if not parts or not all(parts):
        raise ValueError(f"Invalid module name: {dotted_name!r}")
    parent = root
    for p in parts[:-1]:
        if not hasattr(parent, p):
            raise ValueError(
                f"Cannot resolve module name {dotted_name!r}: no attribute {p!r} on {type(parent).__name__}"
            )
        parent = getattr(parent, p)
    attr = parts[-1]
    if not hasattr(parent, attr):
        raise ValueError(
            f"Cannot resolve module name {dotted_name!r}: no attribute {attr!r} on {type(parent).__name__}"
        )
    return parent, attr
