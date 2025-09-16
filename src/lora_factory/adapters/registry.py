from __future__ import annotations

from typing import Callable, Dict

import torch.nn as nn

from .lora import LoRALinear
from .lora_fa import LoRA_FALinear
from .vera import VeRALinear
from .delta_lora import DeltaLoRALinear

_REGISTRY: Dict[str, Callable[..., nn.Module]] = {
    "lora": LoRALinear,
    "lora_fa": LoRA_FALinear,
    "vera": VeRALinear,
    "delta_lora": DeltaLoRALinear,
}


def build_adapter(kind: str) -> Callable[..., nn.Module]:
    k = kind.lower()
    if k not in _REGISTRY:
        raise KeyError(f"Unknown adapter kind: {kind}")
    return _REGISTRY[k]
